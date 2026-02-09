use std::{error::Error, ffi::c_void, io};

use libc::mprotect;

use super::get_page_size;

use crate::{
    collections::TYPE_ID_CONTINUATION,
    runtime::ContinuationObject,
    types::{BuiltInTypes, Header, HeapObject, Word},
};

use super::{
    AllocateAction, Allocator, AllocatorOptions, StackMap,
    continuation_walker::ContinuationSegmentWalker, stack_walker::StackWalker,
};

const DEFAULT_PAGE_COUNT: usize = 1024;
// Aribtary number that should be changed when I have
// better options for gc
const MAX_PAGE_COUNT: usize = 1000000;

struct Space {
    start: *const u8,
    page_count: usize,
    highmark: usize,
    #[allow(unused)]
    protected: bool,
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

    fn copy_data_to_offset(&mut self, offset: usize, data: &[u8]) -> isize {
        unsafe {
            let start = self.start.add(offset);
            let new_pointer = start as isize;
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

    #[allow(unused)]
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

    #[allow(unused)]
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
            highmark: 0,
            protected: false,
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

    fn double_committed_memory(&mut self) {
        let new_page_count = self.page_count * 2;
        Self::commit_memory(self.start as *mut c_void, new_page_count * get_page_size()).unwrap();
        self.page_count = new_page_count;
    }

    fn update_highmark(&mut self, highmark: usize) {
        if highmark > self.highmark {
            self.highmark = highmark;
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct FreeListEntry {
    offset: usize,
    size: usize,
}

impl FreeListEntry {
    pub fn end(&self) -> usize {
        self.offset + self.size
    }

    pub fn can_hold(&self, size: usize) -> bool {
        self.size >= size
    }

    pub fn contains(&self, offset: usize) -> bool {
        self.offset <= offset && offset < self.end()
    }
}

pub struct FreeList {
    ranges: Vec<FreeListEntry>, // always sorted by start
}

impl FreeList {
    fn new(starting_range: FreeListEntry) -> Self {
        FreeList {
            ranges: vec![starting_range],
        }
    }

    fn insert(&mut self, range: FreeListEntry) {
        let mut i = match self
            .ranges
            .binary_search_by_key(&range.offset, |r| r.offset)
        {
            Ok(i) | Err(i) => i,
        };

        // Coalesce with previous if adjacent
        if i > 0 && self.ranges[i - 1].end() == range.offset {
            i -= 1;
            self.ranges[i].size += range.size;
        } else {
            self.ranges.insert(i, range);
        }

        // Coalesce with next if adjacent
        if i + 1 < self.ranges.len() && self.ranges[i].end() == self.ranges[i + 1].offset {
            self.ranges[i].size += self.ranges[i + 1].size;
            self.ranges.remove(i + 1);
        }
    }

    fn allocate(&mut self, size: usize) -> Option<usize> {
        for (i, r) in self.ranges.iter_mut().enumerate() {
            if r.can_hold(size) {
                let addr = r.offset;
                if addr % 8 != 0 {
                    panic!("Heap offset is not aligned");
                }

                r.offset += size;
                r.size -= size;

                if r.size == 0 {
                    self.ranges.remove(i);
                }

                return Some(addr);
            }
        }
        None
    }

    fn iter(&self) -> impl Iterator<Item = &FreeListEntry> {
        self.ranges.iter()
    }

    fn find_entry_contains(&self, offset: usize) -> Option<&FreeListEntry> {
        self.ranges.iter().find(|&entry| entry.contains(offset))
    }
}

pub struct MarkAndSweep {
    space: Space,
    free_list: FreeList,
    options: AllocatorOptions,
}

// TODO: I got an issue with my freelist
impl MarkAndSweep {
    /// Check if a pointer is within this allocator's space
    pub fn contains(&self, pointer: *const u8) -> bool {
        self.space.contains(pointer)
    }

    /// Get the start address of this heap space
    pub fn heap_start(&self) -> usize {
        self.space.start as usize
    }

    /// Get the size of this heap space in bytes
    pub fn heap_size(&self) -> usize {
        self.space.byte_count()
    }

    fn can_allocate(&self, words: usize) -> bool {
        let words = Word::from_word(words);
        // Large objects need 16-byte header, small objects need 8-byte header
        let header_size = if words.to_words() > Header::MAX_INLINE_SIZE {
            16
        } else {
            8
        };
        let size = words.to_bytes() + header_size;
        let spot = self
            .free_list
            .iter()
            .enumerate()
            .find(|(_, x)| x.size >= size);
        spot.is_some()
    }

    fn allocate_inner(
        &mut self,
        words: Word,
        data: Option<&[u8]>,
        kind: crate::types::BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        // Large objects need 16-byte header, small objects need 8-byte header
        let header_size = if words.to_words() > Header::MAX_INLINE_SIZE {
            16
        } else {
            8
        };
        let size_bytes = words.to_bytes() + header_size;

        let offset = self.free_list.allocate(size_bytes);
        if let Some(offset) = offset {
            self.space.update_highmark(offset);
            if let Some(data) = data {
                // When data is provided, copy it directly without first writing
                // a temporary non-opaque header via write_object. The data already
                // contains the correct header (e.g., opaque for floats).
                assert_eq!(
                    data.len(),
                    size_bytes,
                    "data.len()={} != size_bytes={} (words={}, header_size={})",
                    data.len(),
                    size_bytes,
                    words.to_words(),
                    header_size
                );
                let pointer = unsafe { self.space.start.add(offset) as *const u8 };
                assert!(self.space.contains(pointer));
                self.space.copy_data_to_offset(offset, data);
                return Ok(AllocateAction::Allocated(pointer));
            }
            let pointer = self.space.write_object(offset, words);
            // Float objects are opaque (their field is a raw f64, not a pointer).
            // Set the opaque bit immediately so GC never sees a non-opaque float.
            if kind == crate::types::BuiltInTypes::Float {
                unsafe {
                    *(pointer as *mut usize) |= 0x2; // Set opaque bit (bit 1)
                }
            }
            return Ok(AllocateAction::Allocated(pointer));
        }

        Ok(AllocateAction::Gc)
    }

    #[allow(unused)]
    pub fn copy_data_to_offset(&mut self, data: &[u8]) -> *const u8 {
        // TODO: I could amortize this by copying lazily and coalescing
        // the copies together if they are continuous

        // Read the header from the data to determine if it's a large object.
        // Large objects have 16-byte headers, small objects have 8-byte headers.
        let header_value = usize::from_ne_bytes(data[0..8].try_into().unwrap());
        let header_size = if Header::is_large_object_bit_set(header_value) {
            16
        } else {
            8
        };

        let pointer = self
            .allocate_inner(
                Word::from_bytes(data.len() - header_size),
                Some(data),
                crate::types::BuiltInTypes::HeapObject,
            )
            .unwrap();

        if let AllocateAction::Allocated(pointer) = pointer {
            pointer
        } else {
            #[cfg(feature = "debug-gc")]
            eprintln!(
                "[GC DEBUG] copy_data_to_offset: allocation failed, data.len={}, header_size={}, space.page_count={}, space.byte_count={}",
                data.len(),
                header_size,
                self.space.page_count,
                self.space.byte_count()
            );
            self.grow();
            self.copy_data_to_offset(data)
        }
    }

    fn mark(
        &self,
        stack_base: usize,
        stack_map: &super::StackMap,
        frame_pointer: usize,
        gc_return_addr: usize,
    ) {
        let mut to_mark: Vec<HeapObject> = Vec::with_capacity(128);

        // Note: namespace_roots removed - bindings now stored in heap-based PersistentMap
        // which is traced automatically via GlobalObject roots

        // Temporary roots (including Thread objects) are now in GlobalObjectBlocks
        // which are traced via the stack walker.
        // GlobalObject blocks are found via stack walking - no special handling needed.
        // The block pointer is stored on the stack at a known location.

        // Use the stack walker with explicit return address
        StackWalker::walk_stack_roots_with_return_addr(
            stack_base,
            frame_pointer,
            gc_return_addr,
            stack_map,
            |_, pointer| {
                to_mark.push(HeapObject::from_tagged(pointer));
            },
        );

        // Scan continuation segments for heap pointers
        self.mark_continuation_roots(stack_map, &mut to_mark);

        while let Some(object) = to_mark.pop() {
            if object.marked() {
                continue;
            }

            object.mark();
            if object.get_type_id() == TYPE_ID_CONTINUATION as usize {
                let tagged = object.tagged_pointer();
                if let Some(cont) = ContinuationObject::from_tagged(tagged) {
                    cont.with_segment_bytes(|segment| {
                        if segment.is_empty() {
                            return;
                        }
                        ContinuationSegmentWalker::walk_segment_roots(
                            segment,
                            cont.original_sp(),
                            cont.original_fp(),
                            cont.prompt_stack_pointer(),
                            stack_map,
                            |_offset, pointer| {
                                to_mark.push(HeapObject::from_tagged(pointer));
                            },
                        );
                    });
                }
            }
            for child in object.get_heap_references() {
                to_mark.push(child);
            }
        }
    }

    /// Mark extra roots from shadow stacks (HandleScope handles).
    /// These are heap pointers stored in Rust-side Vec buffers.
    fn mark_extra_roots(&self, extra_roots: &[(*mut usize, usize)], stack_map: &StackMap) {
        let mut to_mark: Vec<HeapObject> = Vec::new();
        for &(_slot_addr, value) in extra_roots {
            if BuiltInTypes::is_heap_pointer(value) {
                to_mark.push(HeapObject::from_tagged(value));
            }
        }
        // Same marking loop as in `mark` — transitively mark all reachable objects
        while let Some(object) = to_mark.pop() {
            if object.marked() {
                continue;
            }
            object.mark();
            if object.get_type_id() == TYPE_ID_CONTINUATION as usize {
                let tagged = object.tagged_pointer();
                if let Some(cont) = ContinuationObject::from_tagged(tagged) {
                    cont.with_segment_bytes(|segment| {
                        if segment.is_empty() {
                            return;
                        }
                        ContinuationSegmentWalker::walk_segment_roots(
                            segment,
                            cont.original_sp(),
                            cont.original_fp(),
                            cont.prompt_stack_pointer(),
                            stack_map,
                            |_offset, pointer| {
                                to_mark.push(HeapObject::from_tagged(pointer));
                            },
                        );
                    });
                }
            }
            for child in object.get_heap_references() {
                to_mark.push(child);
            }
        }
    }

    fn mark_continuation_roots(&self, stack_map: &StackMap, to_mark: &mut Vec<HeapObject>) {
        let runtime = crate::get_runtime().get();

        // Scan InvocationReturnPoint saved frames (for multi-shot continuations)
        for (_thread_id, rps) in runtime.invocation_return_points.iter() {
            for rp in rps {
                if rp.saved_stack_frame.is_empty() {
                    continue;
                }

                ContinuationSegmentWalker::walk_segment_roots(
                    &rp.saved_stack_frame,
                    rp.stack_pointer,
                    rp.frame_pointer,
                    rp.frame_pointer, // upper bound is FP
                    stack_map,
                    |_offset, pointer| {
                        to_mark.push(HeapObject::from_tagged(pointer));
                    },
                );
            }
        }

        // Mark saved_continuation_ptr values as roots.
        // These are continuation objects saved during invoke_continuation_runtime.
        for (_thread_id, cont_ptr) in runtime.saved_continuation_ptr.iter() {
            if *cont_ptr == 0 {
                continue;
            }
            if BuiltInTypes::is_heap_pointer(*cont_ptr) {
                to_mark.push(HeapObject::from_tagged(*cont_ptr));
            }
        }
    }

    fn sweep(&mut self) {
        let mut offset = 0;

        loop {
            if offset > self.space.highmark {
                break;
            }
            if let Some(entry) = self.free_list.find_entry_contains(offset) {
                offset = entry.end();
                continue;
            }
            let heap_object = HeapObject::from_untagged(unsafe { self.space.start.add(offset) });

            let full_size = heap_object.full_size();

            if heap_object.marked() {
                heap_object.unmark();
                offset += full_size;
                offset = (offset + 7) & !7;
                continue;
            }
            let size = full_size;
            let entry = FreeListEntry { offset, size };
            self.free_list.insert(entry);
            offset += size;
            offset = (offset + 7) & !7;
            if offset % 8 != 0 {
                panic!("Heap offset is not aligned");
            }

            if offset > self.space.byte_count() {
                panic!("Heap offset is out of bounds");
            }
        }
    }

    #[allow(unused)]
    pub fn new_with_page_count(page_count: usize, options: AllocatorOptions) -> Self {
        let space = Space::new(page_count);
        let size = space.byte_count();
        Self {
            space,
            free_list: FreeList::new(FreeListEntry { offset: 0, size }),
            options,
        }
    }

    /// Walk all live objects in the heap, calling the provided function for each one.
    /// Returns the object's address and a mutable HeapObject reference.
    pub fn walk_objects_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut HeapObject),
    {
        let mut offset = 0;
        loop {
            if offset > self.space.highmark {
                break;
            }
            if let Some(entry) = self.free_list.find_entry_contains(offset) {
                offset = entry.end();
                continue;
            }
            let ptr = unsafe { self.space.start.add(offset) };
            let mut heap_object = HeapObject::from_untagged(ptr);
            f(ptr as usize, &mut heap_object);
            offset += heap_object.full_size();
            offset = (offset + 7) & !7;
        }
    }

    /// Walk all live objects in the heap, calling the provided function for each one.
    #[cfg(feature = "debug-gc")]
    #[allow(unused)]
    pub fn walk_objects<F>(&self, mut f: F)
    where
        F: FnMut(&HeapObject),
    {
        let mut offset = 0;
        loop {
            if offset > self.space.highmark {
                break;
            }
            if let Some(entry) = self.free_list.find_entry_contains(offset) {
                offset = entry.end();
                continue;
            }
            let heap_object = HeapObject::from_untagged(unsafe { self.space.start.add(offset) });
            f(&heap_object);
            offset += heap_object.full_size();
            offset = (offset + 7) & !7;
        }
    }
}

impl Allocator for MarkAndSweep {
    fn new(options: AllocatorOptions) -> Self {
        let page_count = DEFAULT_PAGE_COUNT;
        Self::new_with_page_count(page_count, options)
    }

    fn try_allocate(
        &mut self,
        words: usize,
        kind: crate::types::BuiltInTypes,
    ) -> Result<super::AllocateAction, Box<dyn std::error::Error>> {
        if self.can_allocate(words) {
            self.allocate_inner(Word::from_word(words), None, kind)
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    fn try_allocate_zeroed(
        &mut self,
        words: usize,
        kind: crate::types::BuiltInTypes,
    ) -> Result<super::AllocateAction, Box<dyn std::error::Error>> {
        let result = self.try_allocate(words, kind)?;
        if let AllocateAction::Allocated(ptr) = result {
            // Zero the field area (skip header) so GC doesn't trace garbage data
            let heap_object = HeapObject::from_untagged(ptr);
            let header_size = heap_object.header_size();
            let field_bytes = words * 8;
            unsafe {
                std::ptr::write_bytes((ptr as *mut u8).add(header_size), 0, field_bytes);
            }
            Ok(AllocateAction::Allocated(ptr))
        } else {
            Ok(result)
        }
    }

    fn gc(
        &mut self,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize, usize)],
        extra_roots: &[(*mut usize, usize)],
    ) {
        if !self.options.gc {
            return;
        }
        let start = std::time::Instant::now();
        for (stack_base, frame_pointer, gc_return_addr) in stack_pointers {
            self.mark(*stack_base, stack_map, *frame_pointer, *gc_return_addr);
        }

        // Mark extra roots from shadow stacks
        self.mark_extra_roots(extra_roots, stack_map);

        self.sweep();
        if self.options.print_stats {
            println!("Mark and sweep took {:?}", start.elapsed());
        }
    }

    fn grow(&mut self) {
        let current_max_offset = self.space.byte_count();
        self.space.double_committed_memory();
        let after_max_offset = self.space.byte_count();
        self.free_list.insert(FreeListEntry {
            offset: current_max_offset,
            size: after_max_offset - current_max_offset,
        });
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }
}

// Helper methods for heap dump
impl MarkAndSweep {
    /// Collect all objects for heap dump
    #[cfg(feature = "heap-dump")]
    pub fn collect_objects_for_dump(
        &self,
        classifier: &super::heap_dump::PointerClassifier,
    ) -> Vec<super::heap_dump::ObjectSnapshot> {
        use super::heap_dump::*;
        use crate::types::{BuiltInTypes, Header, HeapObject};

        let mut objects = Vec::new();
        let mut offset = 0;

        loop {
            if offset > self.space.highmark {
                break;
            }

            // Check if this offset is in the free list
            if let Some(entry) = self.free_list.find_entry_contains(offset) {
                offset = entry.end();
                continue;
            }

            let ptr = unsafe { self.space.start.add(offset) };
            let header_raw = unsafe { *(ptr as *const usize) };
            let header = Header::from_usize(header_raw);

            let header_size = if header.large { 16 } else { 8 };
            let fields_size = if header.large {
                let size_ptr = unsafe { (ptr as *const usize).add(1) };
                unsafe { *size_ptr * 8 }
            } else {
                header.size as usize * 8
            };
            let full_size = header_size + fields_size;

            if full_size == 0 {
                offset += 8;
                continue;
            }

            let heap_obj = HeapObject::from_untagged(ptr);

            let mut field_snapshots = Vec::new();
            if !header.opaque {
                let fields = heap_obj.get_fields();
                for (i, &field_value) in fields.iter().enumerate() {
                    field_snapshots.push(FieldSnapshot {
                        index: i,
                        value: format!("{:#x}", field_value),
                        tag: tag_name_local(field_value),
                        is_heap_ptr: BuiltInTypes::is_heap_pointer(field_value),
                        points_to: classifier.classify(field_value),
                    });
                }
            }

            objects.push(ObjectSnapshot {
                tagged_ptr: format!("{:#x}", BuiltInTypes::HeapObject.tag(ptr as isize)),
                address: format!("{:#x}", ptr as usize),
                offset,
                tag_type: "HeapObject".to_string(),
                header_raw: format!("{:#x}", header_raw),
                header: HeaderSnapshot {
                    type_id: header.type_id,
                    type_data: header.type_data,
                    size: header.size,
                    opaque: header.opaque,
                    marked: header.marked,
                    large: header.large,
                },
                full_size,
                fields: field_snapshots,
            });

            offset += full_size;
            offset = (offset + 7) & !7;
        }

        objects
    }
}

#[cfg(feature = "heap-dump")]
fn tag_name_local(value: usize) -> String {
    use crate::types::BuiltInTypes;
    match BuiltInTypes::get_kind(value) {
        BuiltInTypes::Int => "Int".to_string(),
        BuiltInTypes::Float => "Float".to_string(),
        BuiltInTypes::String => "String".to_string(),
        BuiltInTypes::Bool => "Bool".to_string(),
        BuiltInTypes::Function => "Function".to_string(),
        BuiltInTypes::Closure => "Closure".to_string(),
        BuiltInTypes::HeapObject => "HeapObject".to_string(),
        BuiltInTypes::Null => "Null".to_string(),
    }
}
