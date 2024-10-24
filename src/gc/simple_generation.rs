use std::{error::Error, sync::Once};

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
    #[allow(unused)]
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

    fn can_allocate(&mut self, size: Word) -> bool {
        let segment = self.segments.get(self.segment_offset).unwrap();
        let current_segment =
            segment.offset + size.to_bytes() + HeapObject::header_size() < segment.size;
        if current_segment {
            return true;
        }
        while self.segment_offset < self.segments.len() {
            let segment = self.segments.get(self.segment_offset).unwrap();
            if segment.offset + size.to_bytes() + HeapObject::header_size() < segment.size {
                return true;
            }
            self.segment_offset += 1;
        }
        if self.segment_offset == self.segments.len() {
            self.segment_offset = self.segments.len() - 1;
        }
        false
    }

    fn allocate(&mut self, size: Word) -> Result<*const u8, Box<dyn Error>> {
        let segment = self.segments.get_mut(self.segment_offset).unwrap();
        let offset = segment.offset;
        let full_size = size.to_bytes() + HeapObject::header_size();
        if offset + full_size > segment.size {
            panic!("We should only be here if we think we can allocate: full_size: {}, offset: {}, segment.size: {}, diff {}", full_size, offset, segment.size, segment.size - offset);
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

static WARN_MEMORY: Once = Once::new();

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
    additional_roots: Vec<usize>,
    namespace_roots: Vec<(usize, usize)>,
    relocated_namespace_roots: Vec<(usize, Vec<(usize, usize)>)>,
    atomic_pause: [u8; 8],
    options: AllocatorOptions,
}

impl Allocator for SimpleGeneration {
    fn new(options: AllocatorOptions) -> Self {
        // TODO: Make these configurable and play with configurations
        let young_size = MmapOptions::page_size() * 10000;
        let young = Space::new(young_size);
        let old = SimpleMarkSweepHeap::new_with_count(options, 10);
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
            options,
        }
    }

    fn try_allocate(
        &mut self,
        bytes: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let pointer = self.allocate_inner(bytes, kind)?;
        Ok(pointer)
    }

    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        // TODO: Need to figure out when to do a Major GC
        if !self.options.gc {
            return;
        }
        if self.gc_count % self.full_gc_frequency == 0 {
            self.full_gc(stack_map, stack_pointers);
        } else {
            self.minor_gc(stack_map, stack_pointers);
        }
        self.gc_count += 1;
    }

    fn grow(&mut self) {
        if cfg!(debug_assertions) && self.old.segment_count() > 1000 {
            WARN_MEMORY.call_once(|| println!("Warning, memory growing dramatically"));
        }
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

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        std::mem::take(&mut self.relocated_namespace_roots)
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }
}

impl SimpleGeneration {
    fn allocate_inner(
        &mut self,
        bytes: usize,
        _kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let size = Word::from_word(bytes);
        if self.young.can_allocate(size) {
            Ok(AllocateAction::Allocated(self.young.allocate(size)?))
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    fn minor_gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        let start = std::time::Instant::now();
        for (stack_base, stack_pointer) in stack_pointers.iter() {
            let roots = self.gather_roots(*stack_base, stack_map, *stack_pointer);
            let new_roots: Vec<usize> = roots.iter().map(|x| x.1).collect();
            let new_roots = unsafe { self.copy_all(new_roots) };

            let additional_roots = std::mem::take(&mut self.additional_roots);
            for old in additional_roots.into_iter() {
                self.move_objects_referenced_from_old_to_old(&mut HeapObject::from_tagged(old));
            }

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
                    self.move_objects_referenced_from_old_to_old(&mut heap_object);
                }
            }
            self.copy_remaining();

            // TODO: Do better
            self.old.clear_namespace_roots();
            for (namespace_id, root) in self.namespace_roots.iter() {
                self.old.add_namespace_root(*namespace_id, *root);
            }

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
        if self.options.print_stats {
            println!("Minor GC took {:?}", start.elapsed());
        }
    }

    fn full_gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        self.minor_gc(stack_map, stack_pointers);
        self.old.gc(stack_map, stack_pointers);
    }

    unsafe fn copy_all(&mut self, roots: Vec<usize>) -> Vec<usize> {
        let mut new_roots = vec![];
        for root in roots.iter() {
            new_roots.push(self.copy(*root));
        }

        self.copy_remaining();

        new_roots
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
        let heap_object = HeapObject::from_tagged(root);

        if !self.young.contains(heap_object.get_pointer()) {
            return root;
        }

        // if it is marked we have already copied it
        // We now know that the first field is a pointer
        if heap_object.marked() {
            let first_field = heap_object.get_field(0);
            assert!(BuiltInTypes::is_heap_pointer(first_field));
            assert!(!self
                .young
                .contains(BuiltInTypes::untag(first_field) as *const u8));
            return first_field;
        }

        let data = heap_object.get_full_object_data();
        let new_pointer = self.old.copy_data_to_offset(data);
        debug_assert!(new_pointer as usize % 8 == 0, "Pointer is not aligned");
        // update header of original object to now be the forwarding pointer
        let tagged_new = BuiltInTypes::get_kind(root).tag(new_pointer as isize) as usize;

        if heap_object.is_zero_size() {
            return tagged_new;
        }
        let first_field = heap_object.get_field(0);
        if let Some(heap_object) = HeapObject::try_from_tagged(first_field) {
            if !self.young.contains(heap_object.get_pointer()) {
                return tagged_new;
            }
            self.copy(first_field);
        }

        heap_object.write_field(0, tagged_new);
        heap_object.mark();
        self.copied.push(HeapObject::from_untagged(new_pointer));
        tagged_new
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
