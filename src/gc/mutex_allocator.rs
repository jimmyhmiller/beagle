use std::sync::Mutex;

use crate::types::BuiltInTypes;

use super::{AllocateAction, Allocator, AllocatorOptions, StackMap};

pub struct MutexAllocator<Alloc: Allocator> {
    alloc: Alloc,
    mutex: Mutex<()>,
}

impl<Alloc: Allocator> Allocator for MutexAllocator<Alloc> {
    fn new() -> Self {
        MutexAllocator {
            alloc: Alloc::new(),
            mutex: Mutex::new(()),
        }
    }
    fn allocate(
        &mut self,
        bytes: usize,
        kind: BuiltInTypes,
        options: AllocatorOptions,
    ) -> Result<AllocateAction, Box<dyn std::error::Error>> {
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.allocate(bytes, kind, options);
        drop(lock);
        result
    }

    fn gc(
        &mut self,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize)],
        options: AllocatorOptions,
    ) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc(stack_map, stack_pointers, options);
        drop(lock)
    }

    fn grow(&mut self, options: AllocatorOptions) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.grow(options);
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
}
