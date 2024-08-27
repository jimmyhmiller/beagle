use std::{error::Error};

use mmap_rs::{MmapMut, MmapOptions};

use crate::types::{BuiltInTypes, HeapObject, Word};

use super::{
    simple_mark_and_sweep::SimpleMarkSweepHeap, AllocateAction, Allocator, AllocatorOptions,
    StackMap, STACK_SIZE,
};

struct Segment {
    memory: MmapMut,
    offset: usize,
    size: usize,
    memory_range: std::ops::Range<*const u8>,
}

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
}

impl Space {
    fn new(segment_size: usize) -> Self {
        let space = vec![Segment::new(segment_size)];
        Self {
            segments: space,
            segment_offset: 0,
            segment_size,
        }
    }

    fn contains(&self, pointer: *const u8) -> bool {
        for segment in self.segments.iter() {
            if segment.memory_range.contains(&pointer) {
                return true;
            }
        }
        false
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

    fn can_allocate(&mut self, size: usize) -> bool {
        let segment = self.segments.get(self.segment_offset).unwrap();
        let current_segment = segment.offset + size + 8 < segment.size;
        if current_segment {
            return true;
        }
        while self.segment_offset < self.segments.len() {
            let segment = self.segments.get(self.segment_offset).unwrap();
            if segment.offset + size + 8 < segment.size {
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
}

pub struct SimpleGeneration {
    young: Space,
    old: SimpleMarkSweepHeap,
    copied: Vec<HeapObject>,
    gc_count: usize,
    full_gc_frequency: usize,
    // TODO: This may not be the most efficient way
    // but given the way I'm dealing with mutability
    // right now it should work fine.
    // There should be very few atoms
    // But I will probably want to revist this
    additional_roots: Vec<(usize, usize)>,
    namespace_roots: Vec<(usize, usize)>,
    relocated_namespace_roots: Vec<(usize, Vec<(usize, usize)>)>,
    atomic_pause: [u8; 8],
}

impl Allocator for SimpleGeneration {
    #[allow(unused)]
    fn new() -> Self {
        // TODO: Make these configurable and play with configurations
        let young_size = MmapOptions::page_size() * 10000;
        let young = Space::new(young_size);
        let old = SimpleMarkSweepHeap::new_with_count(10);
        let copied = vec![];
        let gc_count = 0;
        let full_gc_frequency = 10;
        Self {
            young,
            old,
            copied,
            gc_count,
            full_gc_frequency,
            additional_roots: vec![],
            namespace_roots: vec![],
            relocated_namespace_roots: vec![],
            atomic_pause: [0; 8],
        }
    }

    fn allocate(
        &mut self,
        bytes: usize,
        kind: BuiltInTypes,
        options: AllocatorOptions,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let pointer = self.allocate_inner(bytes, kind, options)?;
        Ok(pointer)
    }

    fn gc(
        &mut self,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize)],
        options: AllocatorOptions,
    ) {
        // TODO: Need to figure out when to do a Major GC
        if !options.gc {
            return;
        }
        if self.gc_count % self.full_gc_frequency == 0 {
            self.full_gc(stack_map, stack_pointers, options);
        } else {
            self.minor_gc(stack_map, stack_pointers, options);
        }
        self.gc_count += 1;
    }

    fn grow(&mut self, options: AllocatorOptions) {
        self.old.grow(options);
    }

    fn gc_add_root(&mut self, old: usize, young: usize) {
        self.additional_roots.push((old, young));
    }

    fn get_pause_pointer(&self) -> usize {
        self.atomic_pause.as_ptr() as usize
    }

    fn add_namespace_root(&mut self, namespace_id: usize, root: usize) {
        self.namespace_roots.push((namespace_id, root));
    }

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        
        std::mem::take(&mut self.relocated_namespace_roots)
    }
}

impl SimpleGeneration {
    fn allocate_inner(
        &mut self,
        bytes: usize,
        _kind: BuiltInTypes,
        _options: AllocatorOptions,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        if self.young.can_allocate(bytes) {
            Ok(AllocateAction::Allocated(self.young.allocate(bytes)?))
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    fn minor_gc(
        &mut self,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize)],
        options: AllocatorOptions,
    ) {
        let start = std::time::Instant::now();
        for (stack_base, stack_pointer) in stack_pointers.iter() {
            let roots = self.gather_roots(*stack_base, stack_map, *stack_pointer);
            let new_roots: Vec<usize> = roots.iter().map(|x| x.1).collect();
            let new_roots = new_roots
                .into_iter()
                .chain(self.additional_roots.iter().map(|x| &x.1).copied())
                .chain(self.namespace_roots.iter().map(|x| &x.1).copied())
                .collect();
            let new_roots = unsafe { self.copy_all(new_roots) };
            let stack_buffer = get_live_stack(*stack_base, *stack_pointer);
            for (i, (stack_offset, _)) in roots.iter().enumerate() {
                debug_assert!(
                    BuiltInTypes::untag(new_roots[i]) % 8 == 0,
                    "Pointer is not aligned"
                );
                stack_buffer[*stack_offset] = new_roots[i];
            }
        }
        self.young.clear();
        if options.print_stats {
            println!("Minor GC took {:?}", start.elapsed());
        }
    }

    fn full_gc(
        &mut self,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize)],
        options: AllocatorOptions,
    ) {
        self.minor_gc(stack_map, stack_pointers, options);
        self.old.gc(stack_map, stack_pointers, options);
    }

    // TODO: I need to change this into a copy from roots to heap
    // not a segment.
    // That means I need be able to capture the state before I start adding objects
    // and then be able to iterate over the new ones added.
    // Right now, this would cause problems, because the objects alive from the roots
    // will probably not fit in one segment.

    // I also should move this to a new struct

    // I really want to experiment more with gc, but it feels so bogged down in the implementation
    // details right now.
    unsafe fn copy_all(&mut self, roots: Vec<usize>) -> Vec<usize> {
        let mut new_roots = vec![];
        for root in roots.iter() {
            new_roots.push(self.copy(*root));
        }

        while let Some(mut object) = self.copied.pop() {
            if object.marked() {
                panic!("We are copying to this space, nothing should be marked");
            }

            for datum in object.get_fields_mut() {
                if BuiltInTypes::is_heap_pointer(*datum) {
                    *datum = self.copy(*datum);
                }
            }
        }

        new_roots
    }

    unsafe fn copy(&mut self, root: usize) -> usize {
        let heap_object = HeapObject::from_tagged(root);

        if !self.young.contains(heap_object.get_pointer()) {
            return root;
        }

        // If the first field points into the old generation, we can just return the pointer
        // because this is a forwarding pointer.
        let first_field = heap_object.get_field(0);
        if BuiltInTypes::is_heap_pointer(first_field) {
            let untagged_data = BuiltInTypes::untag(first_field);
            if !self.young.contains(untagged_data as *const u8) {
                debug_assert!(untagged_data % 8 == 0, "Pointer is not aligned");
                return first_field;
            }
        }

        let data = heap_object.get_full_object_data();
        let new_pointer = self.old.copy_data_to_offset(data);
        debug_assert!(new_pointer as usize % 8 == 0, "Pointer is not aligned");
        // update header of original object to now be the forwarding pointer
        let tagged_new = BuiltInTypes::get_kind(root).tag(new_pointer as isize) as usize;

        for (old, young) in self.additional_roots.iter() {
            if root == *young {
                let mut object = HeapObject::from_tagged(*old);
                let data = object.get_fields_mut();

                for datum in data.iter_mut() {
                    if datum == young {
                        *datum = tagged_new;
                    }
                }
            }
        }
        for (namespace_id, old_root) in self.namespace_roots.iter() {
            if root == *old_root {
                // check if namespace_id is in relocated_namespace_roots
                // if it is add to list, otherwise create a new list
                let mut found = false;
                for (id, roots) in self.relocated_namespace_roots.iter_mut() {
                    if *id == *namespace_id {
                        roots.push((*old_root, tagged_new));
                        found = true;
                        break;
                    }
                }
                if !found {
                    self.relocated_namespace_roots
                        .push((*namespace_id, vec![(*old_root, tagged_new)]));
                }
            }
        }
        heap_object.write_field(0, tagged_new);
        self.copied.push(HeapObject::from_untagged(new_pointer));
        tagged_new
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

        let mut roots: Vec<(usize, usize)> = Vec::with_capacity(36);

        let mut i = 0;
        while i < stack.len() {
            let value = stack[i];

            if let Some(details) = stack_map.find_stack_data(value) {
                let mut frame_size = details.max_stack_size + details.number_of_locals;
                if frame_size % 2 != 0 {
                    frame_size += 1;
                }

                let bottom_of_frame = i + frame_size + 1;
                let _top_of_frame = i + 1;

                let active_frame = details.current_stack_size + details.number_of_locals;

                i = bottom_of_frame;

                for (j, slot) in stack
                    .iter()
                    .enumerate()
                    .take(bottom_of_frame)
                    .skip(bottom_of_frame - active_frame)
                {
                    if BuiltInTypes::is_heap_pointer(*slot) {
                        let untagged = BuiltInTypes::untag(*slot);
                        debug_assert!(untagged % 8 == 0, "Pointer is not aligned");
                        if !self.young.contains(untagged as *const u8) {
                            continue;
                        }
                        roots.push((j, *slot));
                    }
                }
                continue;
            }
            i += 1;
        }

        roots
    }
}

fn get_live_stack<'a>(stack_base: usize, stack_pointer: usize) -> &'a mut [usize] {
    let stack_end = stack_base;
    // let current_stack_pointer = current_stack_pointer & !0b111;
    let distance_till_end = stack_end - stack_pointer;
    let num_64_till_end = (distance_till_end / 8) + 1;
    let len = STACK_SIZE / 8;
    let stack_begin = stack_end - STACK_SIZE;
    let stack =
        unsafe { std::slice::from_raw_parts_mut(stack_begin as *mut usize, STACK_SIZE / 8) };

    (&mut stack[len - num_64_till_end..]) as _
}

// TODO: I can borrow the code here to get to a generational gc
// That should make a significant difference in performance
// I think to get there, I just need to mark things when I compact them
// Then those those that are marked get copied to the old generation
// I should probably read more about a proper setup for this
// to try and get the details right.
