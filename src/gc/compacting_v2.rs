use core::slice;
use std::{ffi::c_void, io::Error, mem};

use libc::{mprotect, vm_page_size};

use crate::types::{BuiltInTypes, HeapObject, Word};

use super::{AllocateAction, Allocator, AllocatorOptions, StackMap};

const DEFAULT_PAGE_COUNT: usize = 1024;
// Aribtary number that should be changed when I have
// better options for gc
const MAX_PAGE_COUNT: usize = 1000000;

struct Space {
    start: *const u8,
    page_count: usize,
    allocation_offset: usize,
    protected: bool,
}

unsafe impl Send for Space {}
unsafe impl Sync for Space {}

impl Space {
    fn word_count(&self) -> usize {
        (self.page_count * unsafe { vm_page_size }) / 8
    }

    fn byte_count(&self) -> usize {
        self.page_count * unsafe { vm_page_size }
    }

    fn contains(&self, pointer: *const u8) -> bool {
        let start = self.start as usize;
        let end = start + self.byte_count();
        let pointer = pointer as usize;
        pointer >= start && pointer < end
    }

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

    fn allocate(&mut self, words: usize) -> *const u8 {
        let offset = self.allocation_offset;
        let size = Word::from_word(words);
        let full_size = size.to_bytes() + HeapObject::header_size();
        let pointer = self.write_object(offset, size);
        self.increment_current_offset(full_size);
        pointer
    }

    fn increment_current_offset(&mut self, size: usize) {
        self.allocation_offset += size;
    }

    fn object_iter_from_position(&self, offset: usize) -> impl Iterator<Item = HeapObject> + use<> {
        ObjectIterator {
            space: self,
            offset,
        }
    }

    fn protect(&mut self) {
        unsafe {
            mprotect(
                self.start as *mut _,
                self.byte_count() - 1024,
                libc::PROT_NONE,
            )
        };
        self.protected = true;
    }

    fn unprotect(&mut self) {
        unsafe {
            mprotect(
                self.start as *mut _,
                self.byte_count() - 1024,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        };

        self.protected = false;
    }

    fn clear(&mut self) {
        self.allocation_offset = 0;
    }

    fn new(default_page_count: usize) -> Self {
        let pre_allocated_space = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                vm_page_size * MAX_PAGE_COUNT,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        Self::commit_memory(
            pre_allocated_space,
            default_page_count * unsafe { vm_page_size },
        )
        .unwrap();
        Self {
            start: pre_allocated_space as *const u8,
            page_count: default_page_count,
            allocation_offset: 0,
            protected: false,
        }
    }

    fn commit_memory(addr: *mut c_void, size: usize) -> Result<(), Error> {
        unsafe {
            if mprotect(addr, size, libc::PROT_READ | libc::PROT_WRITE) != 0 {
                Err(Error::last_os_error())
            } else {
                Ok(())
            }
        }
    }

    fn double_committed_memory(&mut self) {
        let new_page_count = self.page_count * 2;
        Self::commit_memory(
            self.start as *mut c_void,
            new_page_count * unsafe { vm_page_size },
        )
        .unwrap();
        self.page_count = new_page_count;
    }
}

struct ObjectIterator {
    space: *const Space,
    offset: usize,
}

impl Iterator for ObjectIterator {
    type Item = HeapObject;

    fn next(&mut self) -> Option<Self::Item> {
        let space = unsafe { &*self.space };

        if self.offset >= space.allocation_offset {
            return None;
        }

        if space.allocation_offset == 0 {
            return None;
        }

        let pointer = unsafe { space.start.add(self.offset) };
        let object = HeapObject::from_untagged(pointer);
        let size = object.full_size();

        self.offset += size;
        if self.offset % 8 != 0 {
            panic!("Heap offset is not aligned");
        }
        Some(object)
    }
}

pub struct CompactingHeapV2 {
    to_space: Space,
    from_space: Space,
    namespace_roots: Vec<(usize, usize)>,
    temporary_roots: Vec<Option<usize>>,
    namespace_relocations: Vec<(usize, Vec<(usize, usize)>)>,
    options: AllocatorOptions,
}
impl CompactingHeapV2 {
    fn copy_using_cheneys_algorithm(&mut self, heap_object: HeapObject) -> usize {
        let untagged = heap_object.get_pointer() as usize;

        debug_assert!(
            self.to_space.contains(untagged as *const u8)
                || self.from_space.contains(untagged as *const u8),
            "Pointer is not in to space"
        );

        if self.to_space.contains(untagged as *const u8) {
            debug_assert!(untagged % 8 == 0, "Pointer is not aligned");
            return heap_object.tagged_pointer();
        }

        if !heap_object.is_zero_size() && !heap_object.is_opaque_object() {
            // TODO(DuplicatingOpaque)
            let first_field = heap_object.get_field(0);
            if BuiltInTypes::is_heap_pointer(heap_object.get_field(0)) {
                let untagged_data = BuiltInTypes::untag(first_field);
                if self.to_space.contains(untagged_data as *const u8) {
                    debug_assert!(untagged_data % 8 == 0, "Pointer is not aligned");
                    return first_field;
                }
            }
        }

        // TODO: I want to change up this setup to use my mark bit to know
        // if objects have already been forwarded rather than trying to parse
        // their first field
        if heap_object.is_zero_size() || heap_object.is_opaque_object() {
            // TODO(DuplicatingOpaque)
            let data = heap_object.get_full_object_data();
            let new_pointer = self.to_space.copy_data_to_offset(data);
            let tagged_new = heap_object.get_object_type().unwrap().tag(new_pointer) as usize;
            return tagged_new;
        }
        let first_field = heap_object.get_field(0);
        if BuiltInTypes::is_heap_pointer(first_field) {
            let untagged = BuiltInTypes::untag(first_field);
            if !self.from_space.contains(untagged as *const u8) {
                let first_field = HeapObject::from_tagged(first_field);
                heap_object.write_field(0, self.copy_using_cheneys_algorithm(first_field));
            }
        }
        let data = heap_object.get_full_object_data();
        let new_pointer = self.to_space.copy_data_to_offset(data);
        debug_assert!(new_pointer % 8 == 0, "Pointer is not aligned");
        // update header of original object to now be the forwarding pointer
        let tagged_new = heap_object.get_object_type().unwrap().tag(new_pointer) as usize;
        heap_object.write_field(0, tagged_new);
        // heap_object.mark();
        tagged_new
    }

    unsafe fn copy_all(&mut self, roots: Vec<usize>) -> Vec<usize> {
        unsafe {
            let start_offset = self.to_space.allocation_offset;
            // TODO: Is this vec the best way? Probably not
            // I could hand this the pointers to the stack location
            // then resolve what they point to and update them?
            // I should think about how to get rid of this allocation at the very least.
            let mut new_roots = vec![];
            for root in roots.iter() {
                let heap_object = HeapObject::from_tagged(*root);
                new_roots.push(self.copy_using_cheneys_algorithm(heap_object));
            }

            self.copy_remaining(start_offset);

            new_roots
        }
    }

    unsafe fn copy_remaining(&mut self, start_offset: usize) {
        for mut object in self.to_space.object_iter_from_position(start_offset) {
            if object.marked() {
                panic!("We are copying to this space, nothing should be marked");
            }
            if object.is_opaque_object() || object.is_zero_size() {
                // TODO(DuplicatingOpaque): I think right now I'm duplicating opaque objects
                // Once I have a good means of visualizing that, it would be obvious
                continue;
            }
            for datum in object.get_fields_mut() {
                if BuiltInTypes::is_heap_pointer(*datum) {
                    let heap_object = HeapObject::from_tagged(*datum);
                    *datum = self.copy_using_cheneys_algorithm(heap_object);
                }
            }
        }
    }

    // Stolen from original compacting
    pub fn gather_roots(
        &mut self,
        stack_base: usize,
        stack_map: &StackMap,
        stack_pointer: usize,
    ) -> Vec<(usize, usize)> {
        // I'm adding to the end of the stack I've allocated so I only need to go from the end
        // til the current stack
        let stack = get_live_stack(stack_base, stack_pointer);

        let mut to_mark: Vec<usize> = Vec::with_capacity(128);
        let mut roots: Vec<(usize, usize)> = Vec::with_capacity(36);

        let mut i = 0;
        while i < stack.len() {
            let value = stack[i];

            if let Some(details) = stack_map.find_stack_data(value) {
                let frame_size = details.max_stack_size + details.number_of_locals;
                let padding = frame_size % 2;
                let active_frame_size = details.current_stack_size + details.number_of_locals;

                let diff = frame_size - active_frame_size;

                for (j, slot) in stack
                    .iter()
                    .enumerate()
                    .skip(i + padding + 1)
                    .skip(diff)
                    .take(active_frame_size)
                {
                    if BuiltInTypes::is_heap_pointer(*slot) {
                        roots.push((j, *slot));
                        let untagged = BuiltInTypes::untag(*slot);
                        debug_assert!(untagged % 8 == 0, "Pointer is not aligned");
                        to_mark.push(*slot);
                    }
                }
                i = i + padding + 1 + frame_size;
                continue;
            }
            i += 1;
        }
        roots
    }
}

impl Allocator for CompactingHeapV2 {
    fn new(options: AllocatorOptions) -> Self {
        let to_space = Space::new(DEFAULT_PAGE_COUNT / 2);
        let from_space = Space::new(DEFAULT_PAGE_COUNT / 2);

        Self {
            to_space,
            from_space,
            namespace_roots: vec![],
            temporary_roots: vec![],
            namespace_relocations: vec![],
            options,
        }
    }

    fn try_allocate(
        &mut self,
        words: usize,
        _kind: crate::types::BuiltInTypes,
    ) -> Result<super::AllocateAction, Box<dyn std::error::Error>> {
        if words > self.from_space.word_count() {
            // TODO: Grow should take an allocation size
            self.grow();
        }

        if self.from_space.allocation_offset + words * 8 >= self.from_space.byte_count() {
            return Ok(AllocateAction::Gc);
        }

        // TODO: Actually allocate
        let pointer = self.from_space.allocate(words);

        Ok(AllocateAction::Allocated(pointer))
    }

    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        if !self.options.gc {
            return;
        }
        #[cfg(debug_assertions)]
        {
            self.to_space.unprotect();
        }

        let start_offset = self.to_space.allocation_offset;
        let mut temporary_roots_to_update: Vec<(usize, usize)> = vec![];
        for (i, root) in self.temporary_roots.clone().iter().enumerate() {
            if let Some(root) = root {
                if BuiltInTypes::is_heap_pointer(*root) {
                    let heap_object = HeapObject::from_tagged(*root);
                    debug_assert!(self.from_space.contains(heap_object.get_pointer()));
                    let new_root = self.copy_using_cheneys_algorithm(heap_object);
                    temporary_roots_to_update.push((i, new_root));
                }
            }
        }

        unsafe { self.copy_remaining(start_offset) };

        for (i, new_root) in temporary_roots_to_update.iter() {
            self.temporary_roots[*i] = Some(*new_root);
        }

        for (stack_base, stack_pointer) in stack_pointers.iter() {
            let roots = self.gather_roots(*stack_base, stack_map, *stack_pointer);
            let new_roots = unsafe { self.copy_all(roots.iter().map(|x| x.1).collect()) };

            let stack_buffer = get_live_stack(*stack_base, *stack_pointer);
            for (i, (stack_offset, _)) in roots.iter().enumerate() {
                debug_assert!(
                    BuiltInTypes::untag(new_roots[i]) % 8 == 0,
                    "Pointer is not aligned"
                );
                stack_buffer[*stack_offset] = new_roots[i];
            }
        }

        let start_offset = self.to_space.allocation_offset;
        let namespace_roots = std::mem::take(&mut self.namespace_roots);
        // There has to be a better answer than this. But it does seem to work.
        for (namespace_id, root) in namespace_roots.into_iter() {
            if BuiltInTypes::is_heap_pointer(root) {
                let heap_object = HeapObject::from_tagged(root);
                let new_pointer = self.copy_using_cheneys_algorithm(heap_object);
                self.namespace_relocations
                    .push((namespace_id, vec![(root, new_pointer)]));
                self.namespace_roots.push((namespace_id, new_pointer));
            }
        }
        unsafe { self.copy_remaining(start_offset) };

        mem::swap(&mut self.from_space, &mut self.to_space);

        self.to_space.clear();
        // Only do this when debug mode
        #[cfg(debug_assertions)]
        {
            self.to_space.protect();
        }
    }

    fn grow(&mut self) {
        // From space is never protected
        self.from_space.double_committed_memory();

        #[cfg(debug_assertions)]
        {
            let currently_protect = self.to_space.protected;
            if currently_protect {
                self.to_space.unprotect();
            }
            self.to_space.double_committed_memory();
            if currently_protect {
                self.to_space.protect();
            }
        }

        #[cfg(not(debug_assertions))]
        {
            self.to_space.double_committed_memory();
        }
    }

    fn gc_add_root(&mut self, _old: usize) {
        // Don't need this because this is a write barrier for generational
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

    fn add_namespace_root(&mut self, namespace_id: usize, root: usize) {
        self.namespace_roots.push((namespace_id, root));
    }

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        self.namespace_relocations.drain(0..).collect()
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }
}

unsafe fn buffer_between<'a, T>(start: *mut T, end: *mut T) -> &'a mut [T] {
    unsafe {
        let len = end.offset_from(start);
        slice::from_raw_parts_mut(start, len as usize)
    }
}

fn get_live_stack<'a>(stack_base: usize, stack_pointer: usize) -> &'a mut [usize] {
    unsafe {
        buffer_between(
            (stack_pointer as *mut usize).sub(1),
            stack_base as *mut usize,
        )
    }
}
