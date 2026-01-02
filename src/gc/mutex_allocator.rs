use std::{
    error::Error,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread::ThreadId,
};

use crate::types::BuiltInTypes;

use crate::collections::{HandleArenaPtr, RootSetPtr};

use super::{AllocateAction, Allocator, AllocatorOptions, StackMap};

pub struct MutexAllocator<Alloc: Allocator> {
    alloc: Alloc,
    mutex: Mutex<()>,
    options: AllocatorOptions,
    registered_threads: AtomicUsize,
}

impl<Alloc: Allocator> MutexAllocator<Alloc> {
    #[cfg(feature = "thread-safe")]
    pub fn with_locked_alloc<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Alloc) -> R,
    {
        let _lock = self.mutex.lock().unwrap();
        f(&mut self.alloc)
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
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.try_allocate(bytes, kind);
        }

        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.try_allocate(bytes, kind);
        drop(lock);
        result
    }

    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize, usize)]) {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.gc(stack_map, stack_pointers);
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc(stack_map, stack_pointers);
        drop(lock)
    }

    fn grow(&mut self) {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.grow();
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.grow();
        drop(lock)
    }

    fn gc_add_root(&mut self, old: usize) {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.gc_add_root(old);
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc_add_root(old);
        drop(lock)
    }

    fn register_temporary_root(&mut self, root: usize) -> usize {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.register_temporary_root(root);
        }
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.register_temporary_root(root);
        drop(lock);
        result
    }

    fn unregister_temporary_root(&mut self, id: usize) -> usize {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.unregister_temporary_root(id);
        }
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.unregister_temporary_root(id);
        drop(lock);
        result
    }

    fn peek_temporary_root(&self, id: usize) -> usize {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.peek_temporary_root(id);
        }
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.peek_temporary_root(id);
        drop(lock);
        result
    }

    fn add_namespace_root(&mut self, namespace_id: usize, root: usize) {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.add_namespace_root(namespace_id, root);
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.add_namespace_root(namespace_id, root);
        drop(lock)
    }

    fn remove_namespace_root(&mut self, namespace_id: usize, root: usize) -> bool {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.remove_namespace_root(namespace_id, root);
        }
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.remove_namespace_root(namespace_id, root);
        drop(lock);
        result
    }

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.get_namespace_relocations();
        }
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.get_namespace_relocations();
        drop(lock);
        result
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

    fn add_thread_root(&mut self, thread_id: ThreadId, thread_object: usize) {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.add_thread_root(thread_id, thread_object);
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.add_thread_root(thread_id, thread_object);
        drop(lock)
    }

    fn remove_thread_root(&mut self, thread_id: ThreadId) {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.remove_thread_root(thread_id);
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.remove_thread_root(thread_id);
        drop(lock)
    }

    fn get_thread_root(&self, thread_id: ThreadId) -> Option<usize> {
        // We need to lock here too to ensure we see the updated thread_root
        // after GC relocates the Thread object. Without the lock, there's a
        // data race between GC updating thread_roots and threads reading them.
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.get_thread_root(thread_id);
        }
        // SAFETY: We need to take the lock even for reads to ensure visibility
        // of writes from GC. Cast to &mut to acquire lock (the lock is for
        // synchronization, not mutation).
        let lock = unsafe { &*(&self.mutex as *const Mutex<()> as *mut Mutex<()>) }
            .lock()
            .unwrap();
        let result = self.alloc.get_thread_root(thread_id);
        drop(lock);
        result
    }

    fn register_root_set(&mut self, roots: RootSetPtr) -> usize {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.register_root_set(roots);
        }
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.register_root_set(roots);
        drop(lock);
        result
    }

    fn unregister_root_set(&mut self, id: usize) {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.unregister_root_set(id);
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.unregister_root_set(id);
        drop(lock)
    }

    fn register_handle_arena(&mut self, arena: HandleArenaPtr, thread_id: ThreadId) -> usize {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.register_handle_arena(arena, thread_id);
        }
        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.register_handle_arena(arena, thread_id);
        drop(lock);
        result
    }

    fn unregister_handle_arena_for_thread(&mut self, thread_id: ThreadId) {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.unregister_handle_arena_for_thread(thread_id);
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.unregister_handle_arena_for_thread(thread_id);
        drop(lock);
    }
}
