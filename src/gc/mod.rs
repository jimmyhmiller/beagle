use std::{error::Error, thread::ThreadId};

use bincode::{Decode, Encode};
use nanoserde::SerJson;

use crate::{CommandLineArguments, types::BuiltInTypes};

// Re-export get_page_size from mmap_utils for backward compatibility
pub use crate::mmap_utils::get_page_size;

pub mod compacting;
pub mod generational;
#[cfg(feature = "heap-dump")]
pub mod heap_dump;
pub mod mark_and_sweep;
pub mod mutex_allocator;
pub mod stack_walker;
pub mod usdt_probes;

#[derive(Debug, Encode, Decode, SerJson, Clone)]
pub struct StackMapDetails {
    pub function_name: Option<String>,
    pub number_of_locals: usize,
    pub current_stack_size: usize,
    pub max_stack_size: usize,
}

pub const STACK_SIZE: usize = 1024 * 1024 * 128;

#[derive(Debug, Clone)]
pub struct StackMap {
    details: Vec<(usize, StackMapDetails)>,
}

impl Default for StackMap {
    fn default() -> Self {
        Self::new()
    }
}

impl StackMap {
    pub fn new() -> Self {
        Self { details: vec![] }
    }

    pub fn find_stack_data(&self, pointer: usize) -> Option<&StackMapDetails> {
        // Stack map now stores the exact return address (recorded after the call instruction)
        // No adjustment needed - just match the pointer directly
        for (key, value) in self.details.iter() {
            if *key == pointer {
                return Some(value);
            }
        }
        None
    }

    pub fn extend(&mut self, translated_stack_map: Vec<(usize, StackMapDetails)>) {
        self.details.extend(translated_stack_map);
    }

    /// Find stack data without debug output (for heap dump)
    pub fn find_stack_data_no_debug(&self, pointer: usize) -> Option<&StackMapDetails> {
        for (key, value) in self.details.iter() {
            if *key == pointer {
                return Some(value);
            }
        }
        None
    }

    /// Get all stack map details (for heap dump)
    pub fn details(&self) -> &[(usize, StackMapDetails)] {
        &self.details
    }
}

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

    /// GC with explicit return address for the first frame.
    /// The tuple is (stack_base, frame_pointer, gc_return_addr).
    /// gc_return_addr is the return address of the gc() call, which is the
    /// safepoint in the stack map describing the caller's frame.
    /// If gc_return_addr is 0, the stack walker falls back to [FP+8] lookup.
    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize, usize)]);

    fn grow(&mut self);

    #[allow(unused)]
    fn get_pause_pointer(&self) -> usize {
        0
    }

    fn register_thread(&mut self, _thread_id: ThreadId) {}

    fn remove_thread(&mut self, _thread_id: ThreadId) {}

    fn register_parked_thread(&mut self, _thread_id: ThreadId, _stack_pointer: usize) {}

    fn get_allocation_options(&self) -> AllocatorOptions;

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
}
