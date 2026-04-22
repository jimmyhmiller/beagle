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
        let _lock = self.mutex.lock().unwrap();
        f(&mut self.alloc)
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
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.try_allocate(bytes, kind);
        drop(lock);
        result
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
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.try_allocate_zeroed(words, kind);
        drop(lock);
        result
    }

    fn gc(&mut self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc(gc_frame_tops, extra_roots);
        drop(lock)
    }

    fn grow(&mut self) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.grow();
        drop(lock)
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }

    fn register_thread(&mut self, _thread_id: std::thread::ThreadId) {
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

    fn register_finalizable(&mut self, tagged_ptr: usize) {
        let _lock = self.mutex.lock().unwrap();
        self.alloc.register_finalizable(tagged_ptr);
    }

    fn can_allocate(&self, words: usize, kind: BuiltInTypes) -> bool {
        self.alloc.can_allocate(words, kind)
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
