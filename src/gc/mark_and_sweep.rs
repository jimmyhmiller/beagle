use std::{collections::HashMap, error::Error, ffi::c_void, io, thread::ThreadId};

use libc::mprotect;

use super::get_page_size;

use crate::types::{BuiltInTypes, Header, HeapObject, Word};

use crate::collections::{HandleArenaPtr, RootSetPtr};

use super::{AllocateAction, Allocator, AllocatorOptions, StackMap, stack_walker::StackWalker};

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
    namespace_roots: Vec<(usize, usize)>,
    thread_roots: HashMap<ThreadId, usize>,
    options: AllocatorOptions,
    temporary_roots: Vec<Option<usize>>,
    root_sets: Vec<Option<RootSetPtr>>,
    handle_arenas: Vec<Option<HandleArenaPtr>>,
    handle_arena_threads: HashMap<ThreadId, usize>,
}

// TODO: I got an issue with my freelist
impl MarkAndSweep {
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
            let pointer = self.space.write_object(offset, words);
            if let Some(data) = data {
                self.space.copy_data_to_offset(offset, data);
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
            .allocate_inner(Word::from_bytes(data.len() - header_size), Some(data))
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

        for (_, root) in self.namespace_roots.iter() {
            if !BuiltInTypes::is_heap_pointer(*root) {
                continue;
            }
            to_mark.push(HeapObject::from_tagged(*root));
        }

        // Mark temporary roots (used by builtins to protect values during allocation)
        for temp_root in self.temporary_roots.iter() {
            if let Some(root) = temp_root
                && BuiltInTypes::is_heap_pointer(*root)
            {
                to_mark.push(HeapObject::from_tagged(*root));
            }
        }

        // Mark thread roots (Thread objects for running threads)
        for (_, root) in self.thread_roots.iter() {
            if BuiltInTypes::is_heap_pointer(*root) {
                to_mark.push(HeapObject::from_tagged(*root));
            }
        }

        // Mark roots from registered RootSets (used by AllocationContext)
        for roots_ptr in self.root_sets.iter().flatten() {
            let roots = unsafe { &*roots_ptr.0 };
            for root in roots.roots() {
                if BuiltInTypes::is_heap_pointer(*root) {
                    to_mark.push(HeapObject::from_tagged(*root));
                }
            }
        }

        // Mark roots from registered HandleArenas (thread-local handle storage)
        for arena_ptr in self.handle_arenas.iter().flatten() {
            let arena = unsafe { &*arena_ptr.0 };
            for root in arena.roots() {
                if BuiltInTypes::is_heap_pointer(*root) {
                    to_mark.push(HeapObject::from_tagged(*root));
                }
            }
        }

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

        while let Some(object) = to_mark.pop() {
            if object.marked() {
                continue;
            }

            object.mark();
            for child in object.get_heap_references() {
                to_mark.push(child);
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

            if heap_object.marked() {
                heap_object.unmark();
                offset += heap_object.full_size();
                offset = (offset + 7) & !7;
                continue;
            }
            let size = heap_object.full_size();
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
    pub fn clear_namespace_roots(&mut self) {
        self.namespace_roots.clear();
    }

    #[allow(unused)]
    pub fn clear_thread_roots(&mut self) {
        self.thread_roots.clear();
    }

    pub fn new_with_page_count(page_count: usize, options: AllocatorOptions) -> Self {
        let space = Space::new(page_count);
        let size = space.byte_count();
        Self {
            space,
            free_list: FreeList::new(FreeListEntry { offset: 0, size }),
            namespace_roots: vec![],
            thread_roots: HashMap::new(),
            options,
            temporary_roots: vec![],
            root_sets: vec![],
            handle_arenas: vec![],
            handle_arena_threads: HashMap::new(),
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
        _kind: crate::types::BuiltInTypes,
    ) -> Result<super::AllocateAction, Box<dyn std::error::Error>> {
        if self.can_allocate(words) {
            self.allocate_inner(Word::from_word(words), None)
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize, usize)]) {
        if !self.options.gc {
            return;
        }
        let start = std::time::Instant::now();
        for (stack_base, frame_pointer, gc_return_addr) in stack_pointers {
            self.mark(*stack_base, stack_map, *frame_pointer, *gc_return_addr);
        }
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

    fn gc_add_root(&mut self, _old: usize) {}

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

    fn register_temporary_root(&mut self, root: usize) -> usize {
        debug_assert!(
            self.temporary_roots.len() < 1024,
            "Too many temporary roots {}",
            self.temporary_roots.len()
        );
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

    fn register_root_set(&mut self, roots: RootSetPtr) -> usize {
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
        if let Some(idx) = self.handle_arena_threads.remove(&thread_id)
            && idx < self.handle_arenas.len()
        {
            self.handle_arenas[idx] = None;
        }
    }

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        // This mark and sweep doesn't relocate
        // so we don't have any relocations
        vec![]
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
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

// Helper methods for heap dump
impl MarkAndSweep {
    pub fn space_start(&self) -> usize {
        self.space.start as usize
    }

    pub fn space_byte_count(&self) -> usize {
        self.space.byte_count()
    }

    pub fn highmark(&self) -> usize {
        self.space.highmark
    }

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
