use crate::{
    Data, Message,
    builtins::debugger,
    types::{BuiltInTypes, HeapObject, Word},
};
use mmap_rs::{MmapMut, MmapOptions};
use std::error::Error;

use super::{AllocateAction, Allocator, AllocatorOptions, STACK_SIZE, StackMap};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct FreeListEntry {
    segment: usize,
    offset: usize,
    size: usize,
}

impl FreeListEntry {
    fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.size
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SegmentAction {
    Increment,
    AllocateMore,
}
struct Segment {
    memory: MmapMut,
    offset: usize,
    size: usize,
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
        Self {
            memory,
            offset: 0,
            size,
        }
    }
}

pub struct Space {
    segments: Vec<Segment>,
    segment_offset: usize,
    segment_size: usize,
    scale_factor: usize,
}

impl Space {
    #[allow(unused)]
    fn new(segment_size: usize, scale_factor: usize) -> Self {
        let mut space = vec![Segment::new(segment_size)];
        Self {
            segments: space,
            segment_offset: 0,
            segment_size,
            scale_factor,
        }
    }
}

pub struct SimpleMarkSweepHeap {
    space: Space,
    free_list: Vec<FreeListEntry>,
    namespace_roots: Vec<(usize, usize)>,
    options: AllocatorOptions,
    temporary_roots: Vec<Option<usize>>,
}

impl Allocator for SimpleMarkSweepHeap {
    fn new(options: AllocatorOptions) -> Self {
        Self::new_with_count(options, 1)
    }

    fn try_allocate(
        &mut self,
        bytes: usize,
        _kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        if self.can_allocate(bytes) {
            self.allocate_inner(Word::from_word(bytes), 0, None)
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        self.mark_and_sweep(stack_map, stack_pointers);
    }

    fn grow(&mut self) {
        self.create_more_segments();
    }

    fn gc_add_root(&mut self, _old: usize) {}

    fn add_namespace_root(&mut self, namespace_id: usize, root: usize) {
        self.namespace_roots.push((namespace_id, root));
    }

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        // Simple mark and sweep doesn't relocate
        // so we don't have any relocations
        vec![]
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

impl SimpleMarkSweepHeap {
    pub fn new_with_count(options: AllocatorOptions, initial_segment_count: usize) -> Self {
        let segment_size = MmapOptions::page_size() * 100;
        let mut segments = vec![];
        for _ in 0..initial_segment_count {
            segments.push(Segment::new(segment_size));
        }
        Self {
            space: Space {
                segments,
                segment_offset: 0,
                segment_size,
                scale_factor: 1,
            },
            free_list: vec![],
            namespace_roots: vec![],
            options,
            temporary_roots: vec![],
        }
    }

    pub fn segment_count(&self) -> usize {
        self.space.segments.len()
    }

    fn segment_pointer(&self, arg: usize) -> usize {
        let segment = self.space.segments.get(arg).unwrap();
        segment.memory.as_ptr() as usize
    }

    fn switch_to_available_segment(&mut self, size: usize) -> bool {
        for (segment_index, segment) in self.space.segments.iter().enumerate() {
            if segment.size - segment.offset > size {
                self.space.segment_offset = segment_index;
                return true;
            }
        }
        false
    }

    pub fn clear_namespace_roots(&mut self) {
        self.namespace_roots.clear();
    }

    fn switch_or_create_segments(&mut self, bytes: usize) -> SegmentAction {
        let size = (bytes + 1) * 8;
        if self.switch_to_available_segment(size) {
            return SegmentAction::Increment;
        }

        for (segment_index, segment) in self.space.segments.iter().enumerate() {
            if segment.offset + size < segment.size {
                self.space.segment_offset = segment_index;
                return SegmentAction::Increment;
            }
        }

        self.create_more_segments()
    }

    fn create_more_segments(&mut self) -> SegmentAction {
        self.space.segment_offset = self.space.segments.len();

        for i in 0..self.space.scale_factor {
            self.space
                .segments
                .push(Segment::new(self.space.segment_size));
            let segment_pointer = self.segment_pointer(self.space.segment_offset + i);
            debugger(Message {
                kind: "HeapSegmentPointer".to_string(),
                data: Data::HeapSegmentPointer {
                    pointer: segment_pointer,
                },
            });
        }

        self.space.scale_factor *= 2;
        self.space.scale_factor = self.space.scale_factor.min(64);
        SegmentAction::AllocateMore
    }

    fn write_object(
        &mut self,
        segment_offset: usize,
        offset: usize,
        size: Word,
        data: Option<&[u8]>,
    ) -> *const u8 {
        assert!(offset % 8 == 0, "Offset is not aligned");
        let memory = &mut self.space.segments[segment_offset].memory;
        let pointer = memory.as_ptr();
        let pointer = unsafe { pointer.add(offset) };
        let mut object = HeapObject::from_untagged(pointer);

        if let Some(data) = data {
            object.write_full_object(data);
        } else {
            object.write_header(size);
        }
        pointer
    }

    fn free_are_disjoint(entry1: &FreeListEntry, entry2: &FreeListEntry) -> bool {
        entry1.segment != entry2.segment
            || entry1.offset + entry1.size <= entry2.offset
            || entry2.offset + entry2.size <= entry1.offset
    }

    fn all_disjoint(&self) -> bool {
        for i in 0..self.free_list.len() {
            for j in 0..self.free_list.len() {
                if i == j {
                    continue;
                }
                if !Self::free_are_disjoint(&self.free_list[i], &self.free_list[j]) {
                    return false;
                }
            }
        }
        true
    }

    fn add_frees(&mut self, entries: Vec<FreeListEntry>) {
        let mut remaining_entries = vec![];
        for current_entry in self.free_list.iter_mut() {
            for entry in entries.iter() {
                if current_entry == entry {
                    println!("Double free!");
                }
                if current_entry.segment != entry.segment {
                    continue;
                }

                if current_entry.segment == entry.segment
                    && current_entry.offset + current_entry.size == entry.offset
                {
                    current_entry.size += entry.size;
                    return;
                }
                if current_entry.segment == entry.segment
                    && entry.offset + entry.size == current_entry.offset
                {
                    current_entry.offset = entry.offset;
                    current_entry.size += entry.size;
                    return;
                }
                remaining_entries.push(entry);
            }
        }

        for entry in remaining_entries {
            if entry.offset == 0 && entry.size == self.space.segments[entry.segment].offset {
                self.space.segments[entry.segment].offset = 0;
            } else {
                self.free_list.push(*entry);
            }
        }
    }

    fn current_offset(&self) -> usize {
        self.space.segments[self.space.segment_offset].offset
    }

    fn current_segment_size(&self) -> usize {
        self.space.segments[self.space.segment_offset].size
    }

    fn increment_current_offset(&mut self, size: usize) {
        self.space.segments[self.space.segment_offset].offset += size;
        // align to 8 bytes
        self.space.segments[self.space.segment_offset].offset =
            (self.space.segments[self.space.segment_offset].offset + 7) & !7;
        debug_assert!(
            self.space.segments[self.space.segment_offset].offset % 8 == 0,
            "Heap offset is not aligned"
        );
    }

    pub fn mark_and_sweep(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        let start = std::time::Instant::now();
        for (stack_base, stack_pointer) in stack_pointers {
            self.mark(*stack_base, stack_map, *stack_pointer);
        }
        self.sweep();
        if self.options.print_stats {
            println!("Mark and sweep took {:?}", start.elapsed());
        }
    }

    pub fn mark(&mut self, stack_base: usize, stack_map: &StackMap, stack_pointer: usize) {
        // I'm adding to the end of the stack I've allocated so I only need to go from the end
        // til the current stack
        let stack_end = stack_base;
        // let current_stack_pointer = current_stack_pointer & !0b111;
        let distance_till_end = stack_end - stack_pointer;
        let num_64_till_end = (distance_till_end / 8) + 1;
        let stack_begin = stack_end - STACK_SIZE;
        let stack =
            unsafe { std::slice::from_raw_parts(stack_begin as *const usize, STACK_SIZE / 8) };
        let stack = &stack[stack.len() - num_64_till_end..];

        let mut to_mark: Vec<HeapObject> = Vec::with_capacity(128);

        for (_, root) in self.namespace_roots.iter() {
            if !BuiltInTypes::is_heap_pointer(*root) {
                continue;
            }
            to_mark.push(HeapObject::from_tagged(*root));
        }

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

                for slot in stack
                    .iter()
                    .take(bottom_of_frame)
                    .skip(bottom_of_frame - active_frame)
                {
                    if BuiltInTypes::is_heap_pointer(*slot) {
                        // println!("{} {}", slot, BuiltInTypes::untag(*slot));
                        let untagged = BuiltInTypes::untag(*slot);
                        if untagged % 8 != 0 {
                            panic!("Not aligned");
                        }
                        to_mark.push(HeapObject::from_tagged(*slot));
                    }
                }
                continue;
            }
            i += 1;
        }

        while let Some(object) = to_mark.pop() {
            if object.marked() {
                continue;
            }

            object.mark();
            for object in object.get_heap_references() {
                to_mark.push(object);
            }
        }
    }

    fn sweep(&mut self) {
        let mut free_entries: Vec<FreeListEntry> = Vec::with_capacity(128);
        for (segment_index, segment) in self.space.segments.iter_mut().enumerate() {
            if segment.offset == 0 {
                continue;
            }
            let mut free_in_segment: Vec<&FreeListEntry> = self
                .free_list
                .iter()
                .filter(|x| x.segment == segment_index)
                .collect();

            free_in_segment.sort_by_key(|x| x.offset);
            let mut offset = 0;
            let segment_range = segment.offset;
            // TODO: I'm scanning whole segment even if unused
            let pointer = segment.memory.as_mut_ptr();
            while offset < segment_range {
                for free in free_in_segment.iter() {
                    if free.range().contains(&offset) {
                        offset = free.range().end;
                    }
                    if free.offset > offset {
                        break;
                    }
                }
                if offset >= segment_range {
                    break;
                }
                unsafe {
                    let pointer = pointer.add(offset);
                    let object = HeapObject::from_untagged(pointer);

                    if object.marked() {
                        object.unmark()
                    } else {
                        let entry = FreeListEntry {
                            segment: segment_index,
                            offset,
                            size: object.full_size(),
                        };
                        // We have no defined hard cap yet.
                        // but this is probably a bug
                        // assert!(entry.size < 1000);
                        let mut entered = false;
                        for current_entry in free_entries.iter_mut().rev() {
                            if current_entry.segment == entry.segment
                                && current_entry.offset + current_entry.size == entry.offset
                            {
                                current_entry.size += entry.size;
                                entered = true;
                                break;
                            }
                            if current_entry.segment == entry.segment
                                && entry.offset + entry.size == current_entry.offset
                            {
                                current_entry.offset = entry.offset;
                                current_entry.size += entry.size;
                                entered = true;
                                break;
                            }
                        }
                        if !entered {
                            free_entries.push(entry);
                        }
                        // println!("Found garbage!");
                    }

                    let size = object.full_size();
                    // debug_assert!(size > 8, "Size is less than 8");
                    // println!("size: {}", size);
                    offset += size;
                    offset = (offset + 7) & !7;
                }
            }
        }

        self.add_frees(free_entries);
        debug_assert!(self.all_disjoint(), "Free list is not disjoint");
    }

    fn can_allocate(&mut self, bytes: usize) -> bool {
        let size = (bytes + 1) * 8;

        if self.current_offset() + size < self.current_segment_size() {
            return true;
        }
        if self.switch_to_available_segment(size) {
            return true;
        }

        let spot = self
            .free_list
            .iter_mut()
            .enumerate()
            .find(|(_, x)| x.size >= size);
        spot.is_some()
    }

    fn allocate_inner(
        &mut self,
        size: Word,
        depth: usize,
        data: Option<&[u8]>,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        if depth > 1 {
            panic!("Too deep");
        }

        let size_bytes = size.to_bytes() + 8;

        if self.current_offset() + size_bytes < self.current_segment_size() {
            let pointer =
                self.write_object(self.space.segment_offset, self.current_offset(), size, data);
            self.increment_current_offset(size_bytes);
            return Ok(AllocateAction::Allocated(pointer));
        }

        if self.switch_to_available_segment(size_bytes) {
            return self.allocate_inner(size, depth + 1, data);
        }

        debug_assert!(
            !self.space.segments.iter().any(|x| x.offset == 0),
            "Available segment not being used"
        );

        let mut spot = self
            .free_list
            .iter_mut()
            .enumerate()
            .find(|(_, x)| x.size >= size_bytes);

        if spot.is_none() {
            if self.switch_to_available_segment(size_bytes) {
                return self.allocate_inner(size, depth + 1, data);
            }

            spot = self
                .free_list
                .iter_mut()
                .enumerate()
                .find(|(_, x)| x.size >= size_bytes);

            if spot.is_none() {
                // TODO: I should consider gc rather than growing here
                self.switch_or_create_segments(size_bytes);
                return self.allocate_inner(size, depth + 1, data);
            }
        }

        let (spot_index, spot) = spot.unwrap();

        let mut spot_clone = *spot;
        spot_clone.size = size_bytes;
        spot.size -= size_bytes;
        spot.offset += size_bytes;
        if spot.size == 0 {
            self.free_list.remove(spot_index);
        }

        let pointer = self.write_object(spot_clone.segment, spot_clone.offset, size, data);
        Ok(AllocateAction::Allocated(pointer))
    }

    pub fn copy_data_to_offset(&mut self, data: &[u8]) -> *const u8 {
        // TODO: I could amortize this by copying lazily and coalescing
        // the copies together if they are continuouss
        let pointer = self
            .allocate_inner(Word::from_bytes(data.len() - 8), 0, Some(data))
            .unwrap();

        if let AllocateAction::Allocated(pointer) = pointer {
            pointer
        } else {
            panic!("Failed to allocate");
        }
    }
}
