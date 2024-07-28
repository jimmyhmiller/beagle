use std::sync::Mutex;

use crate::{runtime::Allocator, types::BuiltInTypes};

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
        options: crate::runtime::AllocatorOptions,
    ) -> Result<crate::runtime::AllocateAction, Box<dyn std::error::Error>> {
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.allocate(bytes, kind, options);
        drop(lock);
        result
    }

    fn gc(
        &mut self,
        stack_map: &crate::runtime::StackMap,
        stack_pointers: &[(usize, usize)],
        options: crate::runtime::AllocatorOptions,
    ) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc(stack_map, stack_pointers, options);
        drop(lock)
    }

    fn grow(&mut self, options: crate::runtime::AllocatorOptions) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.grow(options);
        drop(lock)
    }

    fn gc_add_root(&mut self, old: usize, young: usize) {
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc_add_root(old, young);
        drop(lock)
    }
}
