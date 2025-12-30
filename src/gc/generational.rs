use std::{collections::HashMap, error::Error, ffi::c_void, io, thread::ThreadId};

use libc::mprotect;

use super::get_page_size;

use crate::types::{BuiltInTypes, Header, HeapObject, Word};

use crate::collections::{HandleArenaPtr, RootSetPtr};

use super::{
    AllocateAction, Allocator, AllocatorOptions, StackMap, mark_and_sweep::MarkAndSweep,
    stack_walker::StackWalker,
};

const DEFAULT_PAGE_COUNT: usize = 1024;
// Aribtary number that should be changed when I have
// better options for gc
const MAX_PAGE_COUNT: usize = 1000000;

struct Space {
    start: *const u8,
    page_count: usize,
    allocation_offset: usize,
}

unsafe impl Send for Space {}
unsafe impl Sync for Space {}

impl Space {
    #[allow(unused)]
    fn word_count(&self) -> usize {
        (self.page_count * get_page_size()) / 8
    }

    fn byte_count(&self) -> usize {
        self.page_count * get_page_size()
    }

    fn contains(&self, pointer: *const u8) -> bool {
        let start = self.start as usize;
        let end = start + self.byte_count();
        let pointer = pointer as usize;
        pointer >= start && pointer < end
    }

    #[allow(unused)]
    fn copy_data_to_offset(&mut self, data: &[u8]) -> isize {
        unsafe {
            let start = self.start.add(self.allocation_offset);
            let new_pointer = start as isize;
            self.allocation_offset += data.len();
            if self.allocation_offset % 8 != 0 {
                panic!("Heap offset is not aligned");
            }
            std::ptr::copy_nonoverlapping(data.as_ptr(), start as *mut u8, data.len());
            new_pointer
        }
    }

    fn write_object(&mut self, offset: usize, size: Word) -> *const u8 {
        let mut heap_object = HeapObject::from_untagged(unsafe { self.start.add(offset) });

        assert!(self.contains(heap_object.get_pointer()));
        heap_object.write_header(size);

        heap_object.get_pointer()
    }

    fn allocate(&mut self, size: Word) -> *const u8 {
        let offset = self.allocation_offset;
        // Large objects need 16-byte header, small objects need 8-byte header
        let header_size = if size.to_words() > Header::MAX_INLINE_SIZE {
            16
        } else {
            8
        };
        let full_size = size.to_bytes() + header_size;
        let pointer = self.write_object(offset, size);
        self.increment_current_offset(full_size);
        pointer
    }

    fn increment_current_offset(&mut self, size: usize) {
        self.allocation_offset += size;
    }

    fn clear(&mut self) {
        self.allocation_offset = 0;
    }

    fn new(default_page_count: usize) -> Self {
        let pre_allocated_space = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                get_page_size() * MAX_PAGE_COUNT,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        Self::commit_memory(pre_allocated_space, default_page_count * get_page_size()).unwrap();
        Self {
            start: pre_allocated_space as *const u8,
            page_count: default_page_count,
            allocation_offset: 0,
        }
    }

    fn commit_memory(addr: *mut c_void, size: usize) -> Result<(), io::Error> {
        unsafe {
            if mprotect(addr, size, libc::PROT_READ | libc::PROT_WRITE) != 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        }
    }

    #[allow(unused)]
    fn double_committed_memory(&mut self) {
        let new_page_count = self.page_count * 2;
        Self::commit_memory(self.start as *mut c_void, new_page_count * get_page_size()).unwrap();
        self.page_count = new_page_count;
    }

    fn can_allocate(&self, size: Word) -> bool {
        // Large objects need 16-byte header, small objects need 8-byte header
        let header_size = if size.to_words() > Header::MAX_INLINE_SIZE {
            16
        } else {
            8
        };
        let alloc_size = size.to_bytes() + header_size;
        let new_offset = self.allocation_offset + alloc_size;
        if new_offset > self.byte_count() {
            return false;
        }
        true
    }
}

pub struct GenerationalGC {
    young: Space,
    old: MarkAndSweep,
    copied: Vec<HeapObject>,
    gc_count: usize,
    full_gc_frequency: usize,
    // TODO: This may not be the most efficient way
    // but given the way I'm dealing with mutability
    // right now it should work fine.
    // There should be very few atoms
    // But I will probably want to revist this
    additional_roots: Vec<usize>,
    namespace_roots: Vec<(usize, usize)>,
    relocated_namespace_roots: Vec<(usize, Vec<(usize, usize)>)>,
    temporary_roots: Vec<Option<usize>>,
    root_sets: Vec<Option<RootSetPtr>>,
    handle_arenas: Vec<Option<HandleArenaPtr>>,
    handle_arena_threads: HashMap<ThreadId, usize>,
    thread_roots: HashMap<ThreadId, usize>,
    atomic_pause: [u8; 8],
    options: AllocatorOptions,
}

impl Allocator for GenerationalGC {
    fn new(options: AllocatorOptions) -> Self {
        let young = Space::new(DEFAULT_PAGE_COUNT * 10);
        let old = MarkAndSweep::new_with_page_count(DEFAULT_PAGE_COUNT * 100, options);
        Self {
            young,
            old,
            copied: vec![],
            gc_count: 0,
            full_gc_frequency: 100,
            additional_roots: vec![],
            namespace_roots: vec![],
            relocated_namespace_roots: vec![],
            temporary_roots: vec![],
            root_sets: vec![],
            handle_arenas: vec![],
            handle_arena_threads: HashMap::new(),
            thread_roots: HashMap::new(),
            atomic_pause: [0; 8],
            options,
        }
    }

    fn try_allocate(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let pointer = self.allocate_inner(words, kind)?;
        Ok(pointer)
    }

    fn gc(&mut self, stack_map: &super::StackMap, stack_pointers: &[(usize, usize, usize)]) {
        // TODO: Need to figure out when to do a Major GC
        if !self.options.gc {
            return;
        }
        if self.gc_count % self.full_gc_frequency == 0 {
            self.gc_count = 0;
            self.full_gc(stack_map, stack_pointers);
        } else {
            self.minor_gc(stack_map, stack_pointers);
        }
        self.gc_count += 1;
    }

    fn register_root_set(&mut self, roots: RootSetPtr) -> usize {
        // Find an empty slot or push a new one
        for (i, slot) in self.root_sets.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(roots);
                return i;
            }
        }
        self.root_sets.push(Some(roots));
        self.root_sets.len() - 1
    }

    fn unregister_root_set(&mut self, id: usize) {
        if id < self.root_sets.len() {
            self.root_sets[id] = None;
        }
    }

    fn register_handle_arena(&mut self, arena: HandleArenaPtr, thread_id: ThreadId) -> usize {
        for (i, slot) in self.handle_arenas.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(arena);
                self.handle_arena_threads.insert(thread_id, i);
                return i;
            }
        }
        let idx = self.handle_arenas.len();
        self.handle_arenas.push(Some(arena));
        self.handle_arena_threads.insert(thread_id, idx);
        idx
    }

    fn unregister_handle_arena_for_thread(&mut self, thread_id: ThreadId) {
        if let Some(idx) = self.handle_arena_threads.remove(&thread_id) {
            if idx < self.handle_arenas.len() {
                self.handle_arenas[idx] = None;
            }
        }
    }

    fn grow(&mut self) {
        self.old.grow();
    }

    fn gc_add_root(&mut self, old: usize) {
        self.additional_roots.push(old);
    }

    #[allow(unused)]
    fn get_pause_pointer(&self) -> usize {
        self.atomic_pause.as_ptr() as usize
    }

    fn add_namespace_root(&mut self, namespace_id: usize, root: usize) {
        self.namespace_roots.push((namespace_id, root));
    }

    fn remove_namespace_root(&mut self, namespace_id: usize, root: usize) -> bool {
        if let Some(pos) = self
            .namespace_roots
            .iter()
            .position(|(ns, r)| *ns == namespace_id && *r == root)
        {
            self.namespace_roots.swap_remove(pos);
            true
        } else {
            false
        }
    }

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        std::mem::take(&mut self.relocated_namespace_roots)
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }

    fn register_temporary_root(&mut self, root: usize) -> usize {
        debug_assert!(self.temporary_roots.len() < 10, "Too many temporary roots");
        for (i, temp_root) in self.temporary_roots.iter_mut().enumerate() {
            if temp_root.is_none() {
                *temp_root = Some(root);
                return i;
            }
        }
        self.temporary_roots.push(Some(root));
        self.temporary_roots.len() - 1
    }

    fn unregister_temporary_root(&mut self, id: usize) -> usize {
        let value = self.temporary_roots[id];
        self.temporary_roots[id] = None;
        value.unwrap()
    }

    fn peek_temporary_root(&self, id: usize) -> usize {
        self.temporary_roots[id].unwrap()
    }

    fn add_thread_root(&mut self, thread_id: ThreadId, thread_object: usize) {
        self.thread_roots.insert(thread_id, thread_object);
    }

    fn remove_thread_root(&mut self, thread_id: ThreadId) {
        self.thread_roots.remove(&thread_id);
    }

    fn get_thread_root(&self, thread_id: ThreadId) -> Option<usize> {
        self.thread_roots.get(&thread_id).copied()
    }
}

impl GenerationalGC {
    fn allocate_inner(
        &mut self,
        words: usize,
        _kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let size = Word::from_word(words);
        if self.young.can_allocate(size) {
            Ok(AllocateAction::Allocated(self.young.allocate(size)))
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    fn minor_gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize, usize)]) {
        let start = std::time::Instant::now();

        self.process_temporary_roots();
        self.process_root_sets();
        self.process_handle_arenas();
        self.process_additional_roots();
        self.process_namespace_roots();
        self.process_thread_roots();
        self.update_old_generation_namespace_roots();
        self.update_old_generation_thread_roots();
        self.process_stack_roots(stack_map, stack_pointers);

        self.young.clear();

        if self.options.print_stats {
            println!("Minor GC took {:?}", start.elapsed());
        }
    }

    fn process_temporary_roots(&mut self) {
        let roots_to_copy: Vec<(usize, usize)> = self
            .temporary_roots
            .iter()
            .enumerate()
            .filter_map(|(i, root)| root.map(|r| (i, r)))
            .collect();

        for (index, root) in roots_to_copy {
            // Only copy heap pointers - skip tagged integers and other non-heap values
            if BuiltInTypes::is_heap_pointer(root) {
                let new_root = unsafe { self.copy(root) };
                self.temporary_roots[index] = Some(new_root);
            }
        }
        self.copy_remaining();
    }

    /// Process all registered RootSets - update roots in-place if they point to moved objects.
    fn process_root_sets(&mut self) {
        // Collect pointers first to avoid borrowing self while iterating
        let root_set_ptrs: Vec<RootSetPtr> =
            self.root_sets.iter().filter_map(|slot| *slot).collect();

        for roots_ptr in root_set_ptrs {
            // Safety: The caller guarantees the RootSet is valid while registered
            let roots = unsafe { &mut *roots_ptr.0 };
            for root in roots.roots_mut() {
                if BuiltInTypes::is_heap_pointer(*root) {
                    *root = unsafe { self.copy(*root) };
                }
            }
        }
        self.copy_remaining();
    }

    /// Process all registered HandleArenas - update roots in-place if they point to moved objects.
    fn process_handle_arenas(&mut self) {
        // Collect pointers first to avoid borrowing self while iterating
        let arena_ptrs: Vec<HandleArenaPtr> =
            self.handle_arenas.iter().filter_map(|slot| *slot).collect();

        for arena_ptr in arena_ptrs {
            // Safety: The caller guarantees the HandleArena is valid while registered
            let arena = unsafe { &mut *arena_ptr.0 };
            for root in arena.roots_mut() {
                if BuiltInTypes::is_heap_pointer(*root) {
                    *root = unsafe { self.copy(*root) };
                }
            }
        }
        self.copy_remaining();
    }

    fn process_additional_roots(&mut self) {
        let additional_roots = std::mem::take(&mut self.additional_roots);
        for old in additional_roots.into_iter() {
            self.move_objects_referenced_from_old_to_old(&mut HeapObject::from_tagged(old));
        }
    }

    fn process_namespace_roots(&mut self) {
        let namespace_roots = std::mem::take(&mut self.namespace_roots);
        // There has to be a better answer than this. But it does seem to work.
        for (namespace_id, root) in namespace_roots.into_iter() {
            if !BuiltInTypes::is_heap_pointer(root) {
                continue;
            }
            let mut heap_object = HeapObject::from_tagged(root);
            if self.young.contains(heap_object.get_pointer()) && heap_object.marked() {
                // We have already copied this object, so the first field points to the new location
                let new_pointer = heap_object.get_field(0);
                self.namespace_roots.push((namespace_id, new_pointer));
                self.relocated_namespace_roots
                    .push((namespace_id, vec![(root, new_pointer)]));
            } else if self.young.contains(heap_object.get_pointer()) {
                let new_pointer = unsafe { self.copy(root) };
                self.relocated_namespace_roots
                    .push((namespace_id, vec![(root, new_pointer)]));
                self.namespace_roots.push((namespace_id, new_pointer));
                self.move_objects_referenced_from_old_to_old(&mut HeapObject::from_tagged(
                    new_pointer,
                ));
            } else {
                self.namespace_roots.push((namespace_id, root));
                self.move_objects_referenced_from_old_to_old(&mut heap_object);
            }
        }
    }

    fn update_old_generation_namespace_roots(&mut self) {
        self.old.clear_namespace_roots();
        for (namespace_id, root) in self.namespace_roots.iter() {
            self.old.add_namespace_root(*namespace_id, *root);
        }
    }

    fn process_thread_roots(&mut self) {
        let thread_roots: Vec<(ThreadId, usize)> = self.thread_roots.drain().collect();
        for (thread_id, root) in thread_roots.into_iter() {
            if !BuiltInTypes::is_heap_pointer(root) {
                self.thread_roots.insert(thread_id, root);
                continue;
            }
            let mut heap_object = HeapObject::from_tagged(root);
            if self.young.contains(heap_object.get_pointer()) && heap_object.marked() {
                // Already copied, first field points to new location
                let new_pointer = heap_object.get_field(0);
                self.thread_roots.insert(thread_id, new_pointer);
            } else if self.young.contains(heap_object.get_pointer()) {
                let new_pointer = unsafe { self.copy(root) };
                self.thread_roots.insert(thread_id, new_pointer);
                self.move_objects_referenced_from_old_to_old(&mut HeapObject::from_tagged(
                    new_pointer,
                ));
            } else {
                self.thread_roots.insert(thread_id, root);
                self.move_objects_referenced_from_old_to_old(&mut heap_object);
            }
        }
    }

    fn update_old_generation_thread_roots(&mut self) {
        self.old.clear_thread_roots();
        for (thread_id, root) in self.thread_roots.iter() {
            self.old.add_thread_root(*thread_id, *root);
        }
    }

    fn full_gc(&mut self, stack_map: &super::StackMap, stack_pointers: &[(usize, usize, usize)]) {
        self.minor_gc(stack_map, stack_pointers);
        self.old.gc(stack_map, stack_pointers);
    }

    fn process_stack_roots(
        &mut self,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize, usize)],
    ) {
        for (stack_base, frame_pointer, gc_return_addr) in stack_pointers.iter() {
            // Gather young roots AND old roots that need their fields updated
            let (young_roots, old_roots) = self.gather_roots_with_old(*stack_base, stack_map, *frame_pointer, *gc_return_addr);

            // Process young roots - copy them to old generation
            let new_roots: Vec<usize> = young_roots.iter().map(|x| x.1).collect();
            let new_roots = unsafe { self.copy_all(new_roots) };

            self.copy_remaining();

            // With FP-chain based walking, roots contain (slot_address, value) pairs
            // Write new roots directly to their addresses
            for (i, (slot_addr, _)) in young_roots.iter().enumerate() {
                debug_assert!(
                    BuiltInTypes::untag(new_roots[i]) % 8 == 0,
                    "Pointer is not aligned"
                );
                unsafe {
                    *(*slot_addr as *mut usize) = new_roots[i];
                }
            }

            // Process old roots - update their fields if they point to young objects
            for old_root in old_roots {
                self.move_objects_referenced_from_old_to_old(&mut HeapObject::from_tagged(old_root));
            }
        }
    }

    pub fn gather_roots(
        &mut self,
        stack_base: usize,
        stack_map: &StackMap,
        frame_pointer: usize,
        gc_return_addr: usize,
    ) -> Vec<(usize, usize)> {
        let (young_roots, _) = self.gather_roots_with_old(stack_base, stack_map, frame_pointer, gc_return_addr);
        young_roots
    }

    /// Gather roots from the stack, returning both young roots (that need copying)
    /// and old roots (that need their fields checked for young pointers).
    pub fn gather_roots_with_old(
        &mut self,
        stack_base: usize,
        stack_map: &StackMap,
        frame_pointer: usize,
        gc_return_addr: usize,
    ) -> (Vec<(usize, usize)>, Vec<usize>) {
        let mut young_roots: Vec<(usize, usize)> = Vec::with_capacity(36);
        let mut old_roots: Vec<usize> = Vec::with_capacity(36);

        StackWalker::walk_stack_roots_with_return_addr(
            stack_base,
            frame_pointer,
            gc_return_addr,
            stack_map,
            |offset, pointer| {
                let untagged = BuiltInTypes::untag(pointer);
                if self.young.contains(untagged as *const u8) {
                    #[cfg(feature = "debug-gc")]
                    {
                        // Check for potentially problematic pointers
                        let raw_header = unsafe { *(untagged as *const usize) };
                        if raw_header == 0x7 {
                            let prev_word = unsafe { *((untagged - 8) as *const usize) };
                            let tag = pointer & 0x7;
                            let kind = BuiltInTypes::get_kind(pointer);
                            eprintln!(
                                "[GC DEBUG] SUSPICIOUS root: slot_addr={:#x}, tagged={:#x}, untagged={:#x}, tag={}, kind={:?}, raw_header={:#x}, prev_word={:#x}",
                                offset, pointer, untagged, tag, kind, raw_header, prev_word
                            );
                        }
                    }
                    young_roots.push((offset, pointer));
                } else if BuiltInTypes::is_heap_pointer(pointer) {
                    // Old generation object - we need to check its fields for young pointers
                    old_roots.push(pointer);
                }
            },
        );

        (young_roots, old_roots)
    }

    unsafe fn copy_all(&mut self, roots: Vec<usize>) -> Vec<usize> {
        unsafe {
            let mut new_roots = vec![];
            for root in roots.iter() {
                new_roots.push(self.copy(*root));
            }

            self.copy_remaining();

            new_roots
        }
    }

    fn copy_remaining(&mut self) {
        while let Some(mut object) = self.copied.pop() {
            if object.marked() {
                panic!("We are copying to this space, nothing should be marked");
            }

            for datum in object.get_fields_mut() {
                if BuiltInTypes::is_heap_pointer(*datum) {
                    *datum = unsafe { self.copy(*datum) };
                }
            }
        }
    }

    unsafe fn copy(&mut self, root: usize) -> usize {
        unsafe {
            let heap_object = HeapObject::from_tagged(root);

            if !self.young.contains(heap_object.get_pointer()) {
                return root;
            }

            // Debug: Check for corrupt headers to help diagnose root cause
            #[cfg(feature = "debug-gc")]
            {
                let raw_header = *(heap_object.get_pointer() as *const usize);
                if raw_header == 0x7 {
                    let prev_word = *((heap_object.get_pointer() as usize - 8) as *const usize);
                    eprintln!(
                        "[GC DEBUG] CORRUPT ROOT DETECTED: root={:#x} ptr={:?}, raw_header={:#x}, prev_word={:#x}",
                        root, heap_object.get_pointer(), raw_header, prev_word
                    );
                    eprintln!("[GC DEBUG] This pointer points to field[0] of an object, not the header!");
                    // This should no longer happen after the fix for old-to-young pointers
                }
            }

            // if it is marked we have already copied it
            // We now know that the first field is a pointer
            if heap_object.marked() {
                let first_field = heap_object.get_field(0);
                #[cfg(feature = "debug-gc")]
                if !BuiltInTypes::is_heap_pointer(first_field) {
                    let raw_header = *(heap_object.get_pointer() as *const usize);
                    eprintln!(
                        "[GC DEBUG] COPY ERROR: marked object at {:?} has non-heap first_field={:#x}, raw_header={:#x}",
                        heap_object.get_pointer(),
                        first_field,
                        raw_header
                    );
                }
                assert!(BuiltInTypes::is_heap_pointer(first_field));
                assert!(
                    !self
                        .young
                        .contains(BuiltInTypes::untag(first_field) as *const u8)
                );
                return first_field;
            }

            let data = heap_object.get_full_object_data();
            let new_pointer = self.old.copy_data_to_offset(data);
            debug_assert!(new_pointer as usize % 8 == 0, "Pointer is not aligned");
            // update header of original object to now be the forwarding pointer
            let tagged_new = BuiltInTypes::get_kind(root).tag(new_pointer as isize) as usize;

            if heap_object.is_zero_size() {
                // Zero-size objects don't have space for forwarding pointer
                return tagged_new;
            }

            // For opaque objects (strings, keywords), we still need to mark them
            // and write the forwarding pointer to prevent duplicate copies
            if !heap_object.is_opaque_object() {
                let first_field = heap_object.get_field(0);
                if let Some(heap_object) = HeapObject::try_from_tagged(first_field)
                    && self.young.contains(heap_object.get_pointer())
                {
                    self.copy(first_field);
                }
            }

            // Write forwarding pointer to first word (safe even for opaque objects
            // since we've already copied the data to the new location)
            heap_object.write_field(0, tagged_new);
            heap_object.mark();
            self.copied.push(HeapObject::from_untagged(new_pointer));
            tagged_new
        }
    }

    fn move_objects_referenced_from_old_to_old(&mut self, old_object: &mut HeapObject) {
        if self.young.contains(old_object.get_pointer()) {
            return;
        }
        let data = old_object.get_fields_mut();
        for datum in data.iter_mut() {
            if BuiltInTypes::is_heap_pointer(*datum) {
                let untagged = BuiltInTypes::untag(*datum);
                if !self.young.contains(untagged as *const u8) {
                    continue;
                }
                let new_pointer = unsafe { self.copy(*datum) };
                *datum = new_pointer;
            }
        }
    }
}
