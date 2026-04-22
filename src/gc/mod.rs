use std::{error::Error, thread::ThreadId};

use crate::{CommandLineArguments, types::BuiltInTypes};

// Re-export get_page_size from mmap_utils for backward compatibility
pub use crate::mmap_utils::get_page_size;

pub mod compacting;
pub mod finalizers;
pub mod generational;
#[cfg(feature = "heap-dump")]
pub mod heap_dump;
pub mod mark_and_sweep;
pub mod mutex_allocator;
pub mod stack_walker;
pub mod usdt_probes;

pub const STACK_SIZE: usize = 1024 * 1024 * 128;

#[derive(Debug, Clone, Copy)]
pub struct AllocatorOptions {
    pub gc: bool,
    pub print_stats: bool,
    pub gc_always: bool,
}

pub enum AllocateAction {
    Allocated(*const u8),
    Gc,
}

pub fn get_allocate_options(command_line_arguments: &CommandLineArguments) -> AllocatorOptions {
    AllocatorOptions {
        gc: !command_line_arguments.no_gc,
        print_stats: command_line_arguments.show_gc_times,
        gc_always: command_line_arguments.gc_always,
    }
}

pub trait Allocator {
    fn new(options: AllocatorOptions) -> Self;

    // TODO: I probably want something like kind, but not actually kind
    // I might need to allocate things differently based on type
    fn try_allocate(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>>;

    /// Allocate with zeroed memory (for arrays that don't initialize all fields)
    fn try_allocate_zeroed(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        // Default: just use regular allocation
        self.try_allocate(words, kind)
    }

    /// Allocate a long-lived heap object for runtime infrastructure.
    /// For generational GC: allocates directly in old generation.
    /// For other GCs: same as try_allocate.
    /// Returns tagged pointer or error if allocation failed.
    fn allocate_for_runtime(&mut self, words: usize) -> Result<usize, Box<dyn Error>> {
        // Default: use regular allocation
        match self.try_allocate(words, BuiltInTypes::HeapObject)? {
            AllocateAction::Allocated(ptr) => {
                Ok(BuiltInTypes::HeapObject.tag(ptr as isize) as usize)
            }
            AllocateAction::Gc => Err("Need GC to allocate runtime object".into()),
        }
    }

    /// Run garbage collection.
    ///
    /// `gc_frame_tops` contains the GC frame chain top for each thread.
    /// GC walks each chain using the prev-pointer linked list in frame headers.
    ///
    /// `extra_roots` contains `(slot_address, value)` pairs from shadow stacks,
    /// head_block pointers, and namespace roots.
    /// The slot address points into the Rust-side Vec/AtomicUsize buffer,
    /// allowing moving GCs to update pointers in-place.
    fn gc(&mut self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]);

    fn grow(&mut self);

    #[allow(unused)]
    fn get_pause_pointer(&self) -> usize {
        0
    }

    fn register_thread(&mut self, _thread_id: ThreadId) {}

    fn remove_thread(&mut self, _thread_id: ThreadId) {}

    fn register_parked_thread(&mut self, _thread_id: ThreadId, _stack_pointer: usize) {}

    fn get_allocation_options(&self) -> AllocatorOptions;

    /// Check whether the allocator can satisfy an allocation of `words`
    /// without triggering garbage collection. Used by `ensure_space_for`
    /// to pre-trigger GC so that subsequent allocations are GC-free.
    fn can_allocate(&self, words: usize, kind: BuiltInTypes) -> bool;

    /// Returns `(alloc_ptr, alloc_end)` — the byte addresses of the
    /// current bump-allocator frontier and its upper limit.
    ///
    /// This is the window the inline JIT allocator is permitted to
    /// operate within: it may bump `alloc_ptr` up to (but not past)
    /// `alloc_end`. When a bump would overflow, the JIT calls back into
    /// Rust's slow path which returns an updated `(alloc_ptr, alloc_end)`
    /// via this trait.
    ///
    /// Allocators without a bump-pointer fast path (the mark-and-sweep
    /// free-list) return `(0, 0)` — the inline comparison will always
    /// fall to the slow path for them.
    ///
    /// The default implementation returns `(0, 0)` so that a backend
    /// that hasn't opted into inline allocation stays correct-by-default.
    fn allocator_frontier(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Sync the allocator's internal frontier to the supplied
    /// `alloc_ptr` — used by the Rust slow path to absorb JIT-side
    /// bumps that happened before the slow-path entry. The slow path
    /// would otherwise allocate from a stale offset and overlap with
    /// fast-path allocations.
    ///
    /// If `alloc_ptr` isn't within this allocator's bump region
    /// (e.g., `alloc_ptr == 0` on the first allocation), the call is
    /// a no-op. The default implementation does nothing so that
    /// non-bump-pointer allocators stay correct.
    fn sync_allocator_frontier(&mut self, _alloc_ptr: usize) {}

    /// Write barrier for generational GC.
    ///
    /// Called after writing a pointer value into a heap object's field.
    /// For generational GC, this records old-to-young pointers in a remembered set
    /// so they can be traced during minor GC.
    ///
    /// Parameters:
    /// - `object_ptr`: Tagged pointer to the object being written to
    /// - `new_value`: The value being written (may or may not be a heap pointer)
    ///
    /// Default implementation does nothing (for non-generational GCs).
    fn write_barrier(&mut self, _object_ptr: usize, _new_value: usize) {
        // Default: no-op for non-generational GCs
    }

    /// Get the card table biased pointer for generated code write barriers.
    ///
    /// For generational GC, returns a biased pointer such that:
    /// `biased_ptr[addr >> 9] = 1` marks the card containing `addr` as dirty.
    ///
    /// For non-generational GCs, returns null (no card marking needed).
    fn get_card_table_biased_ptr(&self) -> *mut u8 {
        std::ptr::null_mut()
    }

    /// Mark the card for an object unconditionally if it's in old gen.
    /// Used by generated code write barriers where we don't know the value being written.
    ///
    /// This is conservative - we mark the card even if the written value is not a young gen pointer.
    /// The GC will scan the object and find no young gen references, which is fine.
    ///
    /// For non-generational GCs, this is a no-op.
    fn mark_card_unconditional(&mut self, _object_ptr: usize) {
        // Default: no-op for non-generational GCs
    }

    /// Register a freshly-constructed heap object whose struct type has a
    /// finalizer (e.g. FFI Buffer/Cell/TypedArray). For generational GC, this
    /// adds the object to a young-gen side list so minor GC can find dead
    /// finalizables in O(finalizable count) instead of O(young allocations).
    ///
    /// Default is a no-op: other allocators discover finalizables via their
    /// own per-object sweep walk.
    fn register_finalizable(&mut self, _tagged_ptr: usize) {}
}
