use core::panic;
use std::{error::Error, mem, slice};

use libc::mprotect;
use mmap_rs::{MmapMut, MmapOptions};

use crate::types::{BuiltInTypes, HeapObject, Word};

use super::{AllocateAction, Allocator, AllocatorOptions, StackMap};

struct Segment {
    memory: MmapMut,
    offset: usize,
    size: usize,
    memory_range: std::ops::Range<*const u8>,
}

unsafe impl Send for Segment {}
unsafe impl Sync for Segment {}

impl Segment {
    fn new(size: usize) -> Self {
        let memory = MmapOptions::new(size)
            .unwrap()
            .map_mut()
            .unwrap()
            .make_mut()
            .unwrap_or_else(|(_map, e)| {
                panic!("Failed to make mmap executable: {}", e);
            });
        let memory_range = memory.as_ptr_range();
        Self {
            memory,
            offset: 0,
            size,
            memory_range,
        }
    }
}

struct Space {
    segments: Vec<Segment>,
    segment_offset: usize,
    segment_size: usize,
    scale_factor: usize,
}

struct ObjectIterator {
    space: *const Space,
    segment_index: usize,
    offset: usize,
}

impl Iterator for ObjectIterator {
    type Item = HeapObject;

    fn next(&mut self) -> Option<Self::Item> {
        let space = unsafe { &*self.space };
        if self.offset >= space.segments[self.segment_index].offset {
            self.segment_index += 1;
            self.offset = 0;
        }
        if self.segment_index == space.segments.len() {
            return None;
        }
        let segment = &space.segments[self.segment_index];
        if segment.offset == 0 {
            return None;
        }
        let pointer = unsafe { segment.memory.as_ptr().add(self.offset) };
        let object = HeapObject::from_untagged(pointer);
        let size = object.full_size();

        self.offset += size;
        if self.offset % 8 != 0 {
            panic!("Heap offset is not aligned");
        }
        Some(object)
    }
}

impl Space {
    fn new(segment_size: usize, scale_factor: usize) -> Self {
        let space = vec![Segment::new(segment_size)];
        Self {
            segments: space,
            segment_offset: 0,
            segment_size,
            scale_factor,
        }
    }

    fn object_iter_from_position(
        &self,
        segment_index: usize,
        offset: usize,
    ) -> impl Iterator<Item = HeapObject> + use<> {
        ObjectIterator {
            space: self,
            segment_index,
            offset,
        }
    }

    fn current_position(&self) -> (usize, usize) {
        (
            self.segment_offset,
            self.segments[self.segment_offset].offset,
        )
    }

    fn contains(&self, pointer: *const u8) -> bool {
        for segment in self.segments.iter() {
            if segment.memory_range.contains(&pointer) {
                return true;
            }
        }
        false
    }

    fn copy_data_to_offset(&mut self, data: &[u8]) -> isize {
        if !self.can_allocate(data.len()) {
            self.resize();
        }
        let segment = self.segments.get_mut(self.segment_offset).unwrap();
        let buffer = &mut segment.memory[segment.offset..segment.offset + data.len()];
        assert!(buffer.as_ptr().is_aligned());
        buffer.copy_from_slice(data);
        let pointer = buffer.as_ptr() as isize;
        self.increment_current_offset(data.len());
        pointer
    }

    fn write_object(&mut self, segment_offset: usize, offset: usize, size: Word) -> *const u8 {
        let memory = &mut self.segments[segment_offset].memory;

        let mut heap_object = HeapObject::from_untagged(unsafe { memory.as_ptr().add(offset) });
        heap_object.write_header(size);

        heap_object.get_pointer()
    }

    fn increment_current_offset(&mut self, size: usize) {
        self.segments[self.segment_offset].offset += size;
        // align to 8 bytes
        self.segments[self.segment_offset].offset =
            (self.segments[self.segment_offset].offset + 7) & !7;
        debug_assert!(
            self.segments[self.segment_offset].offset % 8 == 0,
            "Heap offset is not aligned"
        );
    }

    fn can_allocate(&mut self, words: usize) -> bool {
        let bytes = Word::from_word(words).to_bytes() + HeapObject::header_size();
        let segment = self.segments.get(self.segment_offset).unwrap();
        let current_segment = segment.offset + bytes < segment.size;
        if current_segment {
            return true;
        }
        while self.segment_offset < self.segments.len() {
            let segment = self.segments.get(self.segment_offset).unwrap();
            if segment.offset + bytes < segment.size {
                return true;
            }
            self.segment_offset += 1;
        }
        if self.segment_offset == self.segments.len() {
            self.segment_offset = self.segments.len() - 1;
        }
        false
    }

    fn allocate(&mut self, words: usize) -> Result<*const u8, Box<dyn Error>> {
        let segment = self.segments.get_mut(self.segment_offset).unwrap();
        let mut offset = segment.offset;
        let size = Word::from_word(words);
        let full_size = size.to_bytes() + HeapObject::header_size();
        if offset + full_size > segment.size {
            self.segment_offset += 1;
            if self.segment_offset == self.segments.len() {
                self.segments.push(Segment::new(self.segment_size));
            }
            offset = 0;
        }
        let pointer = self.write_object(self.segment_offset, offset, size);
        self.increment_current_offset(full_size);
        assert!(pointer as usize % 8 == 0, "Pointer is not aligned");
        Ok(pointer)
    }

    fn clear(&mut self) {
        for segment in self.segments.iter_mut() {
            segment.offset = 0;
        }
        self.segment_offset = 0;
    }

    fn resize(&mut self) {
        let offset = self.segment_offset;
        for _ in 0..self.scale_factor {
            self.segments.push(Segment::new(self.segment_size));
        }
        self.segment_offset = offset + 1;
        self.scale_factor *= 2;
        self.scale_factor = self.scale_factor.min(64);
    }

    fn protect(&mut self) {
        for segment in self.segments.iter_mut() {
            // for each segment we are going to mprotect None
            // so that we can't write to it anymore
            unsafe {
                mprotect(
                    segment.memory.as_ptr() as *mut _,
                    segment.size,
                    libc::PROT_NONE,
                )
            };
        }
    }

    fn unprotect(&mut self) {
        for segment in self.segments.iter_mut() {
            // for each segment we are going to mprotect None
            // so that we can't write to it anymore
            unsafe {
                mprotect(
                    segment.memory.as_ptr() as *mut _,
                    segment.size,
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            };
        }
    }
}

pub struct CompactingHeap {
    from_space: Space,
    to_space: Space,
    namespace_roots: Vec<(usize, usize)>,
    temporary_roots: Vec<Option<usize>>,
    namespace_relocations: Vec<(usize, Vec<(usize, usize)>)>,
    options: AllocatorOptions,
}

impl Allocator for CompactingHeap {
    fn new(options: AllocatorOptions) -> Self {
        let segment_size = MmapOptions::page_size() * 100;
        let from_space = Space::new(segment_size, 1);
        let to_space = Space::new(segment_size, 1);
        Self {
            from_space,
            to_space,
            namespace_roots: vec![],
            namespace_relocations: vec![],
            options,
            temporary_roots: vec![],
        }
    }

    fn try_allocate(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let pointer = self.allocate_inner(words, kind, self.options)?;

        Ok(pointer)
    }

    // TODO: Still got bugs here
    // Simple cases work, but not all cases
    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        if !self.options.gc {
            return;
        }

        #[cfg(debug_assertions)]
        {
            self.to_space.unprotect();
        }

        let start = std::time::Instant::now();
        // TODO: Make this better. I don't like the need to remember
        // to copy_remaining
        let (start_segment, start_offset) = self.to_space.current_position();
        let mut temporary_roots_to_update: Vec<(usize, usize)> = vec![];
        for (i, root) in self.temporary_roots.clone().iter().enumerate() {
            if let Some(root) = root {
                if BuiltInTypes::is_heap_pointer(*root) {
                    let untagged = BuiltInTypes::untag(*root);
                    if !self.to_space.contains(untagged as *const u8)
                        && !self.from_space.contains(untagged as *const u8)
                    {
                        panic!("Pointer is not in either space");
                    }
                    let new_root = unsafe { self.copy_using_cheneys_algorithm(*root) };
                    temporary_roots_to_update.push((i, new_root));
                }
            }
        }
        unsafe { self.copy_remaining(start_segment, start_offset) };

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

        let (start_segment, start_offset) = self.to_space.current_position();
        let namespace_roots = std::mem::take(&mut self.namespace_roots);
        // There has to be a better answer than this. But it does seem to work.
        for (namespace_id, root) in namespace_roots.into_iter() {
            if BuiltInTypes::is_heap_pointer(root) {
                let new_pointer = unsafe { self.copy_using_cheneys_algorithm(root) };
                self.namespace_relocations
                    .push((namespace_id, vec![(root, new_pointer)]));
                self.namespace_roots.push((namespace_id, new_pointer));
            }
        }
        unsafe { self.copy_remaining(start_segment, start_offset) };

        mem::swap(&mut self.from_space, &mut self.to_space);

        self.to_space.clear();
        // Only do this when debug mode
        #[cfg(debug_assertions)]
        {
            self.to_space.protect();
        }
        if self.options.print_stats {
            println!("GC took: {:?}", start.elapsed());
        }
    }

    fn gc_add_root(&mut self, _old: usize) {
        // We don't need to do anything because all roots are gathered
        // from the stack.
        // Maybe we should do something though?
        // I guess this could be useful for c stuff,
        // but for right now I'm not going to do anything.
    }

    fn add_namespace_root(&mut self, namespace_id: usize, root: usize) {
        self.namespace_roots.push((namespace_id, root));
    }

    fn grow(&mut self) {
        self.from_space.resize();
    }

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        let mut relocations = vec![];
        mem::swap(&mut self.namespace_relocations, &mut relocations);
        relocations
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
}

impl CompactingHeap {
    #[allow(clippy::too_many_arguments)]
    fn allocate_inner(
        &mut self,
        words: usize,
        _kind: BuiltInTypes,
        _options: AllocatorOptions,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        if self.from_space.can_allocate(words) {
            Ok(AllocateAction::Allocated(self.from_space.allocate(words)?))
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    unsafe fn copy_all(&mut self, roots: Vec<usize>) -> Vec<usize> {
        unsafe {
            let (start_segment, start_offset) = self.to_space.current_position();
            // TODO: Is this vec the best way? Probably not
            // I could hand this the pointers to the stack location
            // then resolve what they point to and update them?
            // I should think about how to get rid of this allocation at the very least.
            let mut new_roots = vec![];
            for root in roots.iter() {
                let untagged = BuiltInTypes::untag(*root);
                if !self.to_space.contains(untagged as *const u8)
                    && !self.from_space.contains(untagged as *const u8)
                {
                    panic!("Pointer is not in either space");
                }
                new_roots.push(self.copy_using_cheneys_algorithm(*root));
            }

            self.copy_remaining(start_segment, start_offset);

            new_roots
        }
    }

    unsafe fn copy_remaining(&mut self, start_segment: usize, start_offset: usize) {
        unsafe {
            for mut object in self
                .to_space
                .object_iter_from_position(start_segment, start_offset)
            {
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
                        *datum = self.copy_using_cheneys_algorithm(*datum);
                    }
                }
            }
        }
    }

    unsafe fn copy_using_cheneys_algorithm(&mut self, root: usize) -> usize {
        unsafe {
            let heap_object = HeapObject::from_tagged(root);
            let untagged = BuiltInTypes::untag(root);

            debug_assert!(
                self.to_space.contains(untagged as *const u8)
                    || self.from_space.contains(untagged as *const u8),
                "Pointer is not in to space"
            );

            if self.to_space.contains(untagged as *const u8) {
                debug_assert!(untagged % 8 == 0, "Pointer is not aligned");
                return root;
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
            if heap_object.is_zero_size() || heap_object.is_opaque_object() {
                // TODO(DuplicatingOpaque)
                let data = heap_object.get_full_object_data();
                let new_pointer = self.to_space.copy_data_to_offset(data);
                let tagged_new = BuiltInTypes::get_kind(root).tag(new_pointer) as usize;
                return tagged_new;
            }
            let first_field = heap_object.get_field(0);
            if BuiltInTypes::is_heap_pointer(first_field) {
                let untagged = BuiltInTypes::untag(first_field);
                if !self.from_space.contains(untagged as *const u8) {
                    heap_object.write_field(0, self.copy_using_cheneys_algorithm(first_field));
                }
            }
            let data = heap_object.get_full_object_data();
            let new_pointer = self.to_space.copy_data_to_offset(data);
            debug_assert!(new_pointer % 8 == 0, "Pointer is not aligned");
            // update header of original object to now be the forwarding pointer
            let tagged_new = BuiltInTypes::get_kind(root).tag(new_pointer) as usize;
            heap_object.write_field(0, tagged_new);
            // heap_object.mark();
            tagged_new
        }
    }

    // Stolen from simple mark and sweep
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
