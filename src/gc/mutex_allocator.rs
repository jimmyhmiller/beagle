use std::sync::Mutex;

use crate::types::BuiltInTypes;

use super::{AllocateAction, Allocator, AllocatorOptions, StackMap};

pub struct MutexAllocator<Alloc: Allocator> {
    alloc: Alloc,
    mutex: Mutex<()>,
    options: AllocatorOptions,
}

impl<Alloc: Allocator> Allocator for MutexAllocator<Alloc> {
    fn new(options: AllocatorOptions) -> Self {
        MutexAllocator {
            alloc: Alloc::new(options),
            mutex: Mutex::new(()),
            options,
        }
    }
    fn try_allocate(
        &mut self,
        bytes: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn std::error::Error>> {
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.try_allocate(bytes, kind);
        drop(lock);
        result
    }

    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc(stack_map, stack_pointers);
        drop(lock)
    }

    fn grow(&mut self) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.grow();
        drop(lock)
    }

    fn gc_add_root(&mut self, old: usize) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc_add_root(old);
        drop(lock)
    }

    fn add_namespace_root(&mut self, namespace_id: usize, root: usize) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.add_namespace_root(namespace_id, root);
        drop(lock)
    }

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.get_namespace_relocations();
        drop(lock);
        result
    }
    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }
}
