use std::{error::Error, thread::ThreadId};

use bincode::{Decode, Encode};
use nanoserde::SerJson;

use crate::{CommandLineArguments, types::BuiltInTypes};

use crate::collections::{HandleArenaPtr, RootSetPtr};

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

    /// GC with explicit return address for the first frame.
    /// The tuple is (stack_base, frame_pointer, gc_return_addr).
    /// gc_return_addr is the return address of the gc() call, which is the
    /// safepoint in the stack map describing the caller's frame.
    /// If gc_return_addr is 0, the stack walker falls back to [FP+8] lookup.
    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize, usize)]);

    /// Register a RootSet to be processed during GC.
    /// Returns an ID to unregister later.
    fn register_root_set(&mut self, roots: RootSetPtr) -> usize;

    /// Unregister a previously registered RootSet.
    fn unregister_root_set(&mut self, id: usize);

    /// Register a thread-local HandleArena to be processed during GC.
    /// The thread_id is used to unregister the arena when the thread exits.
    /// Returns an ID (unused for now).
    fn register_handle_arena(&mut self, arena: HandleArenaPtr, thread_id: ThreadId) -> usize;

    /// Unregister the HandleArena for a thread that is exiting.
    /// This must be called before the thread's stack is destroyed.
    fn unregister_handle_arena_for_thread(&mut self, thread_id: ThreadId);

    fn grow(&mut self);
    fn gc_add_root(&mut self, old: usize);
    fn register_temporary_root(&mut self, root: usize) -> usize;
    fn unregister_temporary_root(&mut self, id: usize) -> usize;
    fn peek_temporary_root(&self, id: usize) -> usize;
    fn add_namespace_root(&mut self, namespace_id: usize, root: usize);
    fn remove_namespace_root(&mut self, namespace_id: usize, root: usize) -> bool;
    // TODO: Get rid of allocation
    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)>;

    /// Register a Thread object as a root for a running thread.
    /// The thread_object is a tagged pointer to the Thread struct.
    fn add_thread_root(&mut self, thread_id: ThreadId, thread_object: usize);

    /// Remove a thread root when the thread finishes.
    fn remove_thread_root(&mut self, thread_id: ThreadId);

    /// Get the current Thread object pointer (may have been relocated by GC).
    fn get_thread_root(&self, thread_id: ThreadId) -> Option<usize>;

    #[allow(unused)]
    fn get_pause_pointer(&self) -> usize {
        0
    }

    fn register_thread(&mut self, _thread_id: ThreadId) {}

    fn remove_thread(&mut self, _thread_id: ThreadId) {}

    fn register_parked_thread(&mut self, _thread_id: ThreadId, _stack_pointer: usize) {}

    fn get_allocation_options(&self) -> AllocatorOptions;
}
