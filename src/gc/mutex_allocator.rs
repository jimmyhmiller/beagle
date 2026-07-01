use std::{
    error::Error,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::types::BuiltInTypes;

use super::{AllocateAction, Allocator, AllocatorOptions};

pub struct MutexAllocator<Alloc: Allocator> {
    alloc: Alloc,
    mutex: Mutex<()>,
    options: AllocatorOptions,
    registered_threads: AtomicUsize,
}

impl<Alloc: Allocator> MutexAllocator<Alloc> {
    pub fn with_locked_alloc<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Alloc) -> R,
    {
        // Same fast-path interop as try_allocate: sync the inner allocator's
        // frontier from the current thread's MutatorState before the caller
        // touches it, then push the new frontier back afterwards. Without
        // this, direct callers of `alloc.try_allocate` (e.g. runtime-internal
        // Thread struct allocation) would hand out memory that overlaps with
        // objects the JIT fast path already placed at a higher offset.
        //
        // sync_from_mutator_state / refresh_mutator_state are both no-ops
        // when `registered_threads != 0`, matching try_allocate: in that
        // regime the shared frontier is authoritative and MutatorState is
        // disarmed, so we skip the lock+sync dance.
        self.sync_from_mutator_state();
        let lock = self.mutex.lock().unwrap();
        let result = f(&mut self.alloc);
        drop(lock);
        self.refresh_mutator_state();
        result
    }

    /// Inline fast-path interop: every `try_allocate*` call begins by
    /// syncing the inner allocator's bump frontier from the current
    /// thread's `MutatorState.alloc_ptr`, because the JIT inline fast
    /// path has been bumping that pointer without telling the inner
    /// allocator. Without this sync, slow-path allocations would happen
    /// at the inner allocator's stale offset and overlap with fast-path
    /// allocations.
    ///
    /// A `MutatorState.alloc_ptr` of 0 (bootstrap, or disarmed via
    /// multi-mutator registration) is handled as a no-op by the inner
    /// allocator's `sync_allocator_frontier`.
    fn sync_from_mutator_state(&mut self) {
        // Multi-mutator safety: when more than one thread is registered,
        // the inline fast path is disarmed (alloc_end = alloc_ptr in
        // every MutatorState) and the shared allocator frontier is
        // authoritative. Syncing from any one thread's stale
        // MutatorState would clobber progress made by the others.
        if self.registered_threads.load(Ordering::Acquire) != 0 {
            return;
        }
        let ms_alloc_ptr = unsafe { (*crate::runtime::current_mutator_state()).alloc_ptr };
        self.alloc.sync_allocator_frontier(ms_alloc_ptr);
    }

    /// Multi-threaded allocation via a thread-local allocation buffer (TLAB).
    /// First tries a lock-free bump of the calling thread's current TLAB
    /// (`MutatorState.alloc_ptr/alloc_end`); only on overflow does it take the
    /// alloc mutex to carve a fresh TLAB. This removes the per-object mutex that
    /// otherwise serialises (and degrades) parallel allocation.
    fn tlab_allocate(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
        zeroed: bool,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let ms = crate::runtime::current_mutator_state();
        // 1. Lock-free bump within the current TLAB.
        let (ptr, end) = unsafe { ((*ms).alloc_ptr, (*ms).alloc_end) };
        if let Some((obj, new_ptr)) = self.alloc.tlab_bump(ptr, end, words, kind, zeroed) {
            unsafe { (*ms).alloc_ptr = new_ptr };
            return Ok(AllocateAction::Allocated(obj as *const u8));
        }
        // 2. TLAB full (or unarmed): refill under the lock.
        let lock = self.mutex.lock().unwrap();
        let slice = self.alloc.grab_tlab(words);
        let result = match slice {
            Some((start, tlab_end)) => {
                // Place the first object at the front and arm the remainder as
                // the thread's new TLAB.
                let (obj, new_ptr) = self
                    .alloc
                    .tlab_bump(start, tlab_end, words, kind, zeroed)
                    .expect("fresh TLAB must fit the object it was sized for");
                unsafe {
                    (*ms).alloc_ptr = new_ptr;
                    (*ms).alloc_end = tlab_end;
                    (*ms).tlab_start = start;
                }
                Ok(AllocateAction::Allocated(obj as *const u8))
            }
            // No TLAB available: young-gen exhausted, or object too big for a
            // TLAB. Fall back to a direct per-object allocation (routes large
            // objects to tenured; returns Gc when the young gen is full).
            None => {
                if zeroed {
                    self.alloc.try_allocate_zeroed(words, kind)
                } else {
                    self.alloc.try_allocate(words, kind)
                }
            }
        };
        drop(lock);
        result
    }

    /// Direct per-object allocation under the lock, bypassing TLABs. Used for
    /// the `ensure_space_for`/`allocate_no_gc` sequence (which pre-reserves an
    /// exact word count and must not waste it on TLAB slack) and as the
    /// large-object / exhaustion fallback.
    pub fn try_allocate_direct(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
        zeroed: bool,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        // Single-threaded: this must keep the inline frontier consistent exactly
        // like the normal single-thread `try_allocate` path — absorb any inline
        // bumps first (sync) and re-expose the frontier after (refresh).
        // Otherwise the inner allocator's cursor and the MutatorState window
        // diverge and later inline allocations overlap direct ones.
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            self.sync_from_mutator_state();
            let result = if zeroed {
                self.alloc.try_allocate_zeroed(words, kind)
            } else {
                self.alloc.try_allocate(words, kind)
            };
            self.refresh_mutator_state();
            return result;
        }
        // Multi-threaded: direct per-object allocation under the lock, bypassing
        // this thread's TLAB (which stays armed and untouched).
        let lock = self.mutex.lock().unwrap();
        let result = if zeroed {
            self.alloc.try_allocate_zeroed(words, kind)
        } else {
            self.alloc.try_allocate(words, kind)
        };
        drop(lock);
        result
    }

    /// Reserve an EXCLUSIVE thread-local region of at least `bytes` for the
    /// current thread's upcoming no-GC allocation sequence
    /// (`ensure_space_for` + `allocate_no_gc`). Returns false if the young gen
    /// can't grant it (caller GCs/grows and retries). Multi-thread only: makes
    /// the reservation immune to other workers consuming the space (the TOCTOU
    /// that crashes once workers actually run in parallel).
    pub fn reserve_no_gc_region(&mut self, bytes: usize) -> bool {
        let ms = crate::runtime::current_mutator_state();
        unsafe {
            let room = (*ms).alloc_end.saturating_sub((*ms).alloc_ptr);
            if (*ms).alloc_ptr != 0 && room >= bytes {
                return true; // current TLAB already has room
            }
        }
        let lock = self.mutex.lock().unwrap();
        let slice = self.alloc.grab_tlab_sized(bytes);
        drop(lock);
        match slice {
            Some((start, end)) => {
                unsafe {
                    (*ms).alloc_ptr = start;
                    (*ms).alloc_end = end;
                    (*ms).tlab_start = start;
                }
                true
            }
            None => false,
        }
    }

    /// Allocate one object from the current thread's reserved region (bump only,
    /// no lock, no GC). Returns the untagged pointer, or None if it doesn't fit
    /// (which means `ensure_space_for` under-reserved — a bug).
    pub fn alloc_from_region(
        &self,
        words: usize,
        kind: BuiltInTypes,
        zeroed: bool,
    ) -> Option<usize> {
        let ms = crate::runtime::current_mutator_state();
        unsafe {
            let (obj, new_ptr) =
                self.alloc
                    .tlab_bump((*ms).alloc_ptr, (*ms).alloc_end, words, kind, zeroed)?;
            (*ms).alloc_ptr = new_ptr;
            Some(obj)
        }
    }

    /// Inline fast-path interop: after every `try_allocate*` call,
    /// push the inner allocator's new frontier back to the current
    /// thread's `MutatorState.alloc_ptr`/`alloc_end`, so subsequent
    /// JIT inline fast-path reads see the up-to-date window.
    fn refresh_mutator_state(&mut self) {
        // Multi-mutator safety: when more than one thread is registered,
        // do not expose the shared frontier to the inline fast path —
        // two threads would race on the bump. Write (0, 0) to disarm.
        let (alloc_ptr, alloc_end) = if self.registered_threads.load(Ordering::Acquire) == 0 {
            self.alloc.allocator_frontier()
        } else {
            (0, 0)
        };
        unsafe {
            let state = crate::runtime::current_mutator_state();
            (*state).alloc_ptr = alloc_ptr;
            (*state).alloc_end = alloc_end;
        }
    }
}

impl<Alloc: Allocator> Allocator for MutexAllocator<Alloc> {
    fn new(options: AllocatorOptions) -> Self {
        MutexAllocator {
            alloc: Alloc::new(options),
            mutex: Mutex::new(()),
            options,
            registered_threads: AtomicUsize::new(0),
        }
    }
    fn try_allocate(
        &mut self,
        bytes: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        // Single-threaded fast path: skip the pthread_mutex when there are no
        // registered child threads. `register_thread` is only called for spawned
        // threads; the main thread is never counted, so `registered_threads == 0`
        // means only the main thread is live and there is no contention.
        // Checking `<= 1` is wrong: count=1 means main + one child = two mutators
        // both racing on the lock-free path.
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            self.sync_from_mutator_state();
            let result = self.alloc.try_allocate(bytes, kind);
            self.refresh_mutator_state();
            return result;
        }
        // Multi-threaded: allocate from this thread's TLAB (lock-free), taking
        // the mutex only to refill.
        self.tlab_allocate(bytes, kind, false)
    }

    fn try_allocate_zeroed(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            self.sync_from_mutator_state();
            let result = self.alloc.try_allocate_zeroed(words, kind);
            self.refresh_mutator_state();
            return result;
        }
        self.tlab_allocate(words, kind, true)
    }

    fn gc(&mut self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]) {
        // Sync before GC so moving collectors (compacting's Cheney)
        // see every fast-path-allocated object when iterating from_space
        // to sweep finalizers or compute the live frontier. Without this
        // `from_space.allocation_offset` is stale, the fast-path region
        // goes unwalked, and objects there can be left in the old space
        // while roots reference post-swap-to-space addresses.
        self.sync_from_mutator_state();
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc(gc_frame_tops, extra_roots);
        drop(lock);
        // Refresh after GC: compacting swaps from/to spaces, so the old
        // MutatorState.alloc_ptr is now a dangling pointer into the
        // cleared to-space. Re-expose the new inner frontier so the
        // next fast-path bump lands in the current from-space.
        self.refresh_mutator_state();
    }

    fn grow(&mut self) {
        self.sync_from_mutator_state();
        let lock = self.mutex.lock().unwrap();
        self.alloc.grow();
        drop(lock);
        self.refresh_mutator_state();
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }

    fn register_thread(&mut self, _thread_id: std::thread::ThreadId) {
        // Close the fast-path race: before we flip into multi-mutator mode,
        // push the spawning thread's latest MutatorState.alloc_ptr into the
        // inner allocator's frontier and then disarm the spawning thread's
        // MutatorState. Without this, the inline fast path may have bumped
        // `alloc_ptr` past the inner allocator's stale `allocation_offset`;
        // once the counter is non-zero, `sync_from_mutator_state` becomes a
        // no-op and the allocator would hand out overlapping regions.
        //
        // `sync_allocator_frontier` tolerates a `ms_alloc_ptr` of 0 (the
        // MutatorState may never have been armed on this thread), so calling
        // it unconditionally is safe.
        let ms_alloc_ptr = unsafe { (*crate::runtime::current_mutator_state()).alloc_ptr };
        self.alloc.sync_allocator_frontier(ms_alloc_ptr);
        unsafe {
            let state = crate::runtime::current_mutator_state();
            (*state).alloc_ptr = 0;
            (*state).alloc_end = 0;
        }
        self.registered_threads.fetch_add(1, Ordering::AcqRel);
    }

    fn remove_thread(&mut self, _thread_id: std::thread::ThreadId) {
        let previous = self.registered_threads.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(
            previous > 0,
            "remove_thread called with no registered threads"
        );
    }

    fn write_barrier(&mut self, object_ptr: usize, new_value: usize) {
        // No lock needed - write barrier is called after the write has happened,
        // and the remembered set is only read during GC (which holds the lock).
        self.alloc.write_barrier(object_ptr, new_value);
    }

    fn get_card_table_biased_ptr(&self) -> *mut u8 {
        // Delegate to inner allocator to get the actual card table pointer
        self.alloc.get_card_table_biased_ptr()
    }

    fn mark_card_unconditional(&mut self, object_ptr: usize) {
        // Delegate to inner allocator to mark the card
        // No lock needed - card marking is atomic and only read during GC (which holds the lock)
        self.alloc.mark_card_unconditional(object_ptr);
    }

    fn supports_tlab(&self) -> bool {
        self.alloc.supports_tlab()
    }

    fn register_finalizable(&mut self, tagged_ptr: usize) {
        let _lock = self.mutex.lock().unwrap();
        self.alloc.register_finalizable(tagged_ptr);
    }

    fn begin_concurrent(
        &mut self,
        gc_frame_tops: &[usize],
        extra_roots: &[(*mut usize, usize)],
    ) -> bool {
        // Called at STW#1 (no contention). Lock for &mut access / consistency.
        let _lock = self.mutex.lock().unwrap();
        self.alloc.begin_concurrent(gc_frame_tops, extra_roots)
    }

    fn concurrent_trace(&mut self) {
        // Hold the allocation mutex for the ENTIRE trace: this makes the
        // allocator exclusive to the GC thread while mutators run, so mutators
        // block only on `try_allocate` (allocation), never on mutation or the
        // lock-free write barrier / card marking. That is what makes the trace
        // sound without a read barrier.
        let _lock = self.mutex.lock().unwrap();
        self.alloc.concurrent_trace();
    }

    fn finish_concurrent(&mut self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]) {
        // Called at STW#2 (no contention).
        let _lock = self.mutex.lock().unwrap();
        self.alloc.finish_concurrent(gc_frame_tops, extra_roots);
    }

    fn can_allocate(&self, words: usize, kind: BuiltInTypes) -> bool {
        self.alloc.can_allocate(words, kind)
    }

    fn bytes_in_use(&self) -> usize {
        // Lock for a consistent snapshot — this is a diagnostic read, not on
        // the allocation hot path.
        let _lock = self.mutex.lock().unwrap();
        self.alloc.bytes_in_use()
    }

    fn allocator_frontier(&self) -> (usize, usize) {
        // Multi-mutator safety: when more than one thread is registered,
        // the shared bump frontier cannot be exposed to the JIT's inline
        // fast path — two threads bumping the same `alloc_ptr` without
        // synchronisation would race. Return (0, 0) so callers forcing
        // slow path take over until the thread count returns to one.
        if self.registered_threads.load(Ordering::Acquire) != 0 {
            return (0, 0);
        }
        self.alloc.allocator_frontier()
    }

    fn sync_allocator_frontier(&mut self, alloc_ptr: usize) {
        self.alloc.sync_allocator_frontier(alloc_ptr);
    }
}
