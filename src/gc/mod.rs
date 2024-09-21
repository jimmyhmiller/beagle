use std::{error::Error, thread::ThreadId};

use bincode::{Decode, Encode};

use crate::types::BuiltInTypes;

pub mod compacting;
pub mod mutex_allocator;
pub mod simple_generation;
pub mod simple_mark_and_sweep;

#[derive(Debug, Encode, Decode, Clone)]
pub struct StackMapDetails {
    pub number_of_locals: usize,
    pub current_stack_size: usize,
    pub max_stack_size: usize,
}

pub const STACK_SIZE: usize = 1024 * 1024 * 32;

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
        for (key, value) in self.details.iter() {
            if *key == pointer.saturating_sub(4) {
                return Some(value);
            }
        }
        None
    }

    pub fn extend(&mut self, translated_stack_map: Vec<(usize, StackMapDetails)>) {
        self.details.extend(translated_stack_map);
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

pub trait Allocator {
    fn new() -> Self;

    // TODO: I probably want something like kind, but not actually kind
    // I might need to allocate things differently based on type
    fn allocate(
        &mut self,
        bytes: usize,
        kind: BuiltInTypes,
        options: AllocatorOptions,
    ) -> Result<AllocateAction, Box<dyn Error>>;
    
    fn gc(
        &mut self,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize)],
        options: AllocatorOptions,
    );

    fn grow(&mut self, options: AllocatorOptions);
    fn gc_add_root(&mut self, old: usize, young: usize);
    fn add_namespace_root(&mut self, namespace_id: usize, root: usize);
    // TODO: Get rid of allocation
    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)>;

    fn get_pause_pointer(&self) -> usize {
        0
    }

    fn register_thread(&mut self, _thread_id: ThreadId) {}

    // TODO: I think this won't work because of my read write lock
    // I probably need to change that.
    fn register_parked_thread(&mut self, _thread_id: ThreadId, _stack_pointer: usize) {}
}
