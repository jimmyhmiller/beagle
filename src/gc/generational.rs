use std::{collections::HashMap, error::Error, ffi::c_void, io, thread::ThreadId};

use libc::mprotect;

use super::get_page_size;

use crate::types::{BuiltInTypes, Header, HeapObject, Word};

use crate::collections::{HandleArenaPtr, RootSetPtr};

use super::{AllocateAction, Allocator, AllocatorOptions, StackMap, mark_and_sweep::MarkAndSweep};

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

    /// Check if pointer is within the ALLOCATED portion of the space.
    /// This is more strict than contains() - it checks allocation_offset, not the full range.
    fn contains_allocated(&self, pointer: *const u8) -> bool {
        let start = self.start as usize;
        let end = start + self.allocation_offset;
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
        if let Some(idx) = self.handle_arena_threads.remove(&thread_id)
            && idx < self.handle_arenas.len()
        {
            self.handle_arenas[idx] = None;
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

        // Heap dump before GC
        #[cfg(feature = "heap-dump")]
        {
            static GC_COUNT: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let count = GC_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            if let Ok(dump_dir) = std::env::var("BEAGLE_HEAP_DUMP_DIR") {
                // Check if we should dump this GC (BEAGLE_HEAP_DUMP_GC=N means dump GC #N)
                let should_dump = std::env::var("BEAGLE_HEAP_DUMP_GC")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .is_some_and(|n| n == count);

                // Or dump all GCs (BEAGLE_HEAP_DUMP_ALL=1)
                let dump_all = std::env::var("BEAGLE_HEAP_DUMP_ALL").is_ok();

                if should_dump || dump_all {
                    let label = format!("gc_{}_before", count);
                    self.dump_before_after_gc(&label, stack_map, stack_pointers, &dump_dir);
                }
            }
        }

        self.process_temporary_roots();
        self.process_root_sets();
        self.process_handle_arenas();
        self.process_additional_roots();
        self.process_namespace_roots();
        self.process_thread_roots();
        self.update_old_generation_namespace_roots();
        self.update_old_generation_thread_roots();
        self.process_stack_roots(stack_map, stack_pointers);

        // Verify no stale pointers BEFORE clearing young generation
        #[cfg(feature = "debug-gc")]
        self.verify_no_young_pointers(stack_map, stack_pointers);

        // Heap dump after GC (before clearing young)
        #[cfg(feature = "heap-dump")]
        {
            static GC_COUNT_AFTER: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let count = GC_COUNT_AFTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            if let Ok(dump_dir) = std::env::var("BEAGLE_HEAP_DUMP_DIR") {
                let should_dump = std::env::var("BEAGLE_HEAP_DUMP_GC")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .is_some_and(|n| n == count);
                let dump_all = std::env::var("BEAGLE_HEAP_DUMP_ALL").is_ok();

                if should_dump || dump_all {
                    let label = format!("gc_{}_after", count);
                    self.dump_before_after_gc(&label, stack_map, stack_pointers, &dump_dir);
                }
            }
        }

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
        #[cfg(feature = "debug-gc")]
        eprintln!("[GC] process_stack_roots: {} stacks", stack_pointers.len());

        for (_idx, (stack_base, frame_pointer, gc_return_addr)) in stack_pointers.iter().enumerate()
        {
            let roots = self.gather_roots(*stack_base, stack_map, *frame_pointer, *gc_return_addr);

            #[cfg(feature = "debug-gc")]
            {
                eprintln!(
                    "[GC] Stack[{}]: gathered {} roots from FP={:#x}",
                    _idx,
                    roots.len(),
                    frame_pointer
                );
                for (slot_addr, val) in roots.iter() {
                    eprintln!("[GC]   root: slot={:#x} val={:#x}", slot_addr, val);
                }
            }

            let new_roots: Vec<usize> = roots.iter().map(|x| x.1).collect();
            let new_roots = unsafe { self.copy_all(new_roots) };

            self.copy_remaining();

            // With FP-chain based walking, roots contain (slot_address, value) pairs
            // Write new roots directly to their addresses
            for (i, (slot_addr, _old_val)) in roots.iter().enumerate() {
                debug_assert!(
                    BuiltInTypes::untag(new_roots[i]) % 8 == 0,
                    "Pointer is not aligned"
                );

                #[cfg(feature = "debug-gc")]
                {
                    let still_young = self
                        .young
                        .contains(BuiltInTypes::untag(new_roots[i]) as *const u8);
                    if still_young {
                        eprintln!(
                            "[GC] WARNING: copy returned young ptr! slot={:#x} old={:#x} new={:#x}",
                            slot_addr, _old_val, new_roots[i]
                        );
                    }
                }

                unsafe {
                    *(*slot_addr as *mut usize) = new_roots[i];
                }
            }
        }
    }

    /// Gather young roots from the stack, and also trace into old gen objects
    /// to find young children that need to be copied.
    pub fn gather_roots(
        &mut self,
        stack_base: usize,
        stack_map: &StackMap,
        frame_pointer: usize,
        gc_return_addr: usize,
    ) -> Vec<(usize, usize)> {
        let mut roots: Vec<(usize, usize)> = Vec::with_capacity(36);
        let mut old_gen_objects: Vec<HeapObject> = Vec::with_capacity(16);

        // Custom stack walk with detailed slot tracking for debugging
        let mut fp = frame_pointer;
        let mut pending_return_addr = gc_return_addr;

        while fp != 0 && fp < stack_base {
            let caller_fp = unsafe { *(fp as *const usize) };
            let return_addr_for_caller = unsafe { *((fp + 8) as *const usize) };

            if pending_return_addr != 0
                && let Some(details) = stack_map.find_stack_data(pending_return_addr)
            {
                let active_slots = details.number_of_locals + details.current_stack_size;

                for i in 0..active_slots {
                    let slot_addr = fp - 8 - (i * 8);
                    let slot_value = unsafe { *(slot_addr as *const usize) };

                    if BuiltInTypes::is_heap_pointer(slot_value) {
                        let untagged = BuiltInTypes::untag(slot_value);

                        // Skip unaligned pointers
                        if untagged % 8 != 0 {
                            continue;
                        }

                        if self.young.contains(untagged as *const u8) {
                            // Validate the pointer BEFORE adding to roots
                            let is_local = i < details.number_of_locals;
                            let slot_type = if is_local { "LOCAL" } else { "STACK_SPILL" };
                            let slot_index_in_type = if is_local {
                                i
                            } else {
                                i - details.number_of_locals
                            };

                            // Check if pointer is within allocated portion
                            if !self.young.contains_allocated(untagged as *const u8) {
                                eprintln!(
                                    "[GC BUG] {} slot contains pointer beyond allocation_offset!",
                                    slot_type
                                );
                                eprintln!("  function: {:?}", details.function_name);
                                eprintln!("  slot[{}] = {}[{}]", i, slot_type, slot_index_in_type);
                                eprintln!(
                                    "  number_of_locals={}, current_stack_size={}",
                                    details.number_of_locals, details.current_stack_size
                                );
                                eprintln!("  slot_addr={:#x}, value={:#x}", slot_addr, slot_value);
                                eprintln!(
                                    "  young_start={:#x}, allocation_offset={:#x}",
                                    self.young.start as usize, self.young.allocation_offset
                                );
                                eprintln!(
                                    "  pointer offset: {:#x}",
                                    untagged - self.young.start as usize
                                );
                                panic!(
                                    "Stack slot contains pointer beyond young gen allocation_offset"
                                );
                            }

                            // Parse header and check for suspicious states
                            let heap_object = HeapObject::from_tagged(slot_value);
                            let header = heap_object.get_header();
                            let tag = BuiltInTypes::get_kind(slot_value);

                            // Note: marked=true is valid here. By the time we process stack roots,
                            // earlier root processing steps (additional_roots, namespace_roots, etc.)
                            // may have already copied this object and marked the original.
                            // The copy() function handles this correctly by following the forwarding pointer.
                            let _ = (header, tag); // silence unused warnings

                            // Closures should never have large flag set
                            if matches!(tag, BuiltInTypes::Closure) && header.large {
                                let header_raw = unsafe { *(untagged as *const usize) };
                                eprintln!(
                                    "[GC BUG] {} slot contains Closure with large=true!",
                                    slot_type
                                );
                                eprintln!("  function: {:?}", details.function_name);
                                eprintln!("  slot[{}] = {}[{}]", i, slot_type, slot_index_in_type);
                                eprintln!(
                                    "  number_of_locals={}, current_stack_size={}",
                                    details.number_of_locals, details.current_stack_size
                                );
                                eprintln!("  slot_addr={:#x}, value={:#x}", slot_addr, slot_value);
                                eprintln!(
                                    "  header_raw={:#x}, header: type_id={}, size={}, large={}, marked={}",
                                    header_raw,
                                    header.type_id,
                                    header.size,
                                    header.large,
                                    header.marked
                                );
                                panic!("Stack slot contains Closure with impossible large flag");
                            }

                            roots.push((slot_addr, slot_value));
                        } else if BuiltInTypes::is_heap_pointer(slot_value) {
                            // Old gen object on stack - need to trace into it for young children
                            if let Some(heap_obj) = HeapObject::try_from_tagged(slot_value) {
                                old_gen_objects.push(heap_obj);
                            }
                        }
                    }
                }
            }

            if caller_fp != 0 && caller_fp <= fp {
                break;
            }

            fp = caller_fp;
            pending_return_addr = return_addr_for_caller;
        }

        // Trace into old gen objects to find young children
        for mut obj in old_gen_objects {
            self.move_objects_referenced_from_old_to_old(&mut obj);
        }

        roots
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

            // Check if the address is within the ALLOCATED portion of young gen.
            // If not, this is a stale pointer to unallocated memory.
            if !self.young.contains_allocated(heap_object.get_pointer()) {
                #[cfg(feature = "debug-gc")]
                eprintln!(
                    "[GC] STALE: root={:#x} in young but NOT allocated (offset > {:#x})",
                    root, self.young.allocation_offset
                );
                return root;
            }

            // Closures (tag 5) should never be large objects.
            // If the header says large=true for a Closure, something is wrong.
            let tag = BuiltInTypes::get_kind(root);
            let header = heap_object.get_header();
            if matches!(tag, BuiltInTypes::Closure) && header.large {
                let offset_in_young = heap_object.untagged() - self.young.start as usize;
                let header_raw = *(heap_object.untagged() as *const usize);
                eprintln!("[GC BUG] Closure with large=true!");
                eprintln!("  root={:#x} (tag=5=Closure)", root);
                eprintln!("  untagged={:#x}", heap_object.untagged());
                eprintln!(
                    "  young_start={:#x}, young_end={:#x}",
                    self.young.start as usize,
                    self.young.start as usize + self.young.allocation_offset
                );
                eprintln!(
                    "  offset_in_young={:#x} (alloc_offset={:#x})",
                    offset_in_young, self.young.allocation_offset
                );
                eprintln!("  header_raw={:#x}", header_raw);
                eprintln!(
                    "  header_raw looks like heap ptr: {}",
                    BuiltInTypes::is_heap_pointer(header_raw)
                );
                eprintln!(
                    "  header parsed: type_id={}, size={}, large={}, marked={}",
                    header.type_id, header.size, header.large, header.marked
                );
                panic!("Closure with large=true - this should be impossible");
            }

            #[cfg(feature = "debug-gc")]
            {
                // Dump raw memory at this address to understand what we're looking at
                let ptr = heap_object.untagged() as *const usize;
                let header_raw = *ptr;
                let word1 = *ptr.add(1);
                let word2 = *ptr.add(2);
                let header = heap_object.get_header();

                // Only log if this looks problematic
                if header.marked || header.large {
                    eprintln!(
                        "[GC COPY] Examining root={:#x}: header_raw={:#x}, word1={:#x}, word2={:#x}",
                        root, header_raw, word1, word2
                    );
                    eprintln!(
                        "[GC COPY]   header: marked={}, large={}, size={}, type_id={}",
                        header.marked, header.large, header.size, header.type_id
                    );
                }
            }

            // if it is marked we have already copied it
            // The first field contains the forwarding pointer to old gen
            if heap_object.marked() {
                let first_field = heap_object.get_field(0);
                if !BuiltInTypes::is_heap_pointer(first_field) {
                    // Invalid forwarding pointer - skip
                    #[cfg(feature = "debug-gc")]
                    eprintln!(
                        "[GC] marked but invalid fwd: root={:#x}, first_field={:#x}",
                        root, first_field
                    );
                    return root;
                }
                // Forwarding pointer should point to old generation
                if self
                    .young
                    .contains(BuiltInTypes::untag(first_field) as *const u8)
                {
                    #[cfg(feature = "debug-gc")]
                    eprintln!(
                        "[GC] marked but fwd still young: root={:#x}, first_field={:#x}",
                        root, first_field
                    );
                    return root;
                }
                #[cfg(feature = "debug-gc")]
                eprintln!(
                    "[GC] already marked: root={:#x} -> fwd={:#x}",
                    root, first_field
                );
                return first_field;
            }

            // Verify large objects have reasonable extended size
            let header = heap_object.get_header();
            if header.large {
                let untagged = heap_object.untagged();
                let size_ptr = (untagged as *const usize).add(1);
                let extended_size = *size_ptr;
                // If extended_size looks like a tagged pointer, this isn't valid
                if BuiltInTypes::is_heap_pointer(extended_size) || extended_size > 1_000_000 {
                    #[cfg(feature = "debug-gc")]
                    eprintln!(
                        "[GC] invalid large obj: root={:#x}, extended_size={:#x}",
                        root, extended_size
                    );
                    return root;
                }
            }

            let data = heap_object.get_full_object_data();
            #[cfg(feature = "debug-gc")]
            {
                if data.len() > 1024 * 1024 {
                    let young_start = self.young.start as usize;
                    let young_end = young_start + self.young.byte_count();
                    let ptr = heap_object.get_pointer() as *const usize;
                    let raw_words: Vec<usize> = (0..4).map(|i| *ptr.add(i)).collect();
                    eprintln!(
                        "[GC DEBUG] copy: HUGE object! root={:#x}, ptr={:?}, data.len={}, header={:?}",
                        root,
                        heap_object.get_pointer(),
                        data.len(),
                        heap_object.get_header()
                    );
                    eprintln!(
                        "[GC DEBUG] young range: {:#x} - {:#x}, ptr in range: {}",
                        young_start,
                        young_end,
                        self.young.contains(heap_object.get_pointer())
                    );
                    eprintln!(
                        "[GC DEBUG] raw memory at ptr: [{:#x}, {:#x}, {:#x}, {:#x}]",
                        raw_words[0], raw_words[1], raw_words[2], raw_words[3]
                    );
                }
            }
            let new_pointer = self.old.copy_data_to_offset(data);
            debug_assert!(new_pointer as usize % 8 == 0, "Pointer is not aligned");
            // update header of original object to now be the forwarding pointer
            let tagged_new = BuiltInTypes::get_kind(root).tag(new_pointer as isize) as usize;

            if heap_object.is_zero_size() {
                // Zero-size objects don't have space for forwarding pointer
                return tagged_new;
            }

            // Save the first field BEFORE we overwrite it with the forwarding pointer
            let first_field = if !heap_object.is_opaque_object() {
                Some(heap_object.get_field(0))
            } else {
                None
            };

            // CRITICAL: Mark the object and write forwarding pointer BEFORE
            // recursively copying children. This prevents infinite loops with
            // cyclic references (A -> B -> A would otherwise copy A twice).
            heap_object.write_field(0, tagged_new);
            heap_object.mark();

            #[cfg(feature = "debug-gc")]
            {
                // Verify the write actually happened
                let verify = heap_object.get_field(0);
                if verify != tagged_new {
                    eprintln!(
                        "[GC COPY] BUG! write_field failed: wrote {:#x} but read back {:#x}",
                        tagged_new, verify
                    );
                }
            }

            self.copied.push(HeapObject::from_untagged(new_pointer));

            // Now safe to recursively copy children - if we encounter this
            // object again, it will be marked and we'll return the forwarding pointer
            if let Some(first_field) = first_field
                && let Some(child) = HeapObject::try_from_tagged(first_field)
                && self.young.contains(child.get_pointer())
            {
                self.copy(first_field);
            }

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

    /// Verify no stale young pointers remain after GC processing.
    /// This is a debug function to help identify where stale pointers come from.
    #[cfg(feature = "debug-gc")]
    fn verify_no_young_pointers(
        &self,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize, usize)],
    ) {
        let mut errors = Vec::new();

        // Check stacks
        for (i, (stack_base, frame_pointer, gc_return_addr)) in stack_pointers.iter().enumerate() {
            StackWalker::walk_stack_roots_with_return_addr(
                *stack_base,
                *frame_pointer,
                *gc_return_addr,
                stack_map,
                |slot_addr, pointer| {
                    if BuiltInTypes::is_heap_pointer(pointer) {
                        let untagged = BuiltInTypes::untag(pointer);
                        if self.young.contains(untagged as *const u8) {
                            errors.push(format!(
                                "STACK[{}]: slot={:#x} still points to young={:#x}",
                                i, slot_addr, pointer
                            ));
                        }
                    }
                },
            );
        }

        // Check temporary roots
        for (i, root) in self.temporary_roots.iter().enumerate() {
            if let Some(ptr) = root {
                if BuiltInTypes::is_heap_pointer(*ptr) {
                    let untagged = BuiltInTypes::untag(*ptr);
                    if self.young.contains(untagged as *const u8) {
                        errors.push(format!(
                            "TEMP_ROOT[{}]: still points to young={:#x}",
                            i, ptr
                        ));
                    }
                }
            }
        }

        // Check root sets
        for (i, slot) in self.root_sets.iter().enumerate() {
            if let Some(roots_ptr) = slot {
                let roots = unsafe { &*roots_ptr.0 };
                for (j, root) in roots.roots().iter().enumerate() {
                    if BuiltInTypes::is_heap_pointer(*root) {
                        let untagged = BuiltInTypes::untag(*root);
                        if self.young.contains(untagged as *const u8) {
                            errors.push(format!(
                                "ROOT_SET[{}][{}]: still points to young={:#x}",
                                i, j, root
                            ));
                        }
                    }
                }
            }
        }

        // Check handle arenas
        for (i, slot) in self.handle_arenas.iter().enumerate() {
            if let Some(arena_ptr) = slot {
                let arena = unsafe { &*arena_ptr.0 };
                for (j, root) in arena.roots().iter().enumerate() {
                    if BuiltInTypes::is_heap_pointer(*root) {
                        let untagged = BuiltInTypes::untag(*root);
                        if self.young.contains(untagged as *const u8) {
                            errors.push(format!(
                                "HANDLE_ARENA[{}][{}]: still points to young={:#x}",
                                i, j, root
                            ));
                        }
                    }
                }
            }
        }

        // Check namespace roots
        for (ns_id, root) in self.namespace_roots.iter() {
            if BuiltInTypes::is_heap_pointer(*root) {
                let untagged = BuiltInTypes::untag(*root);
                if self.young.contains(untagged as *const u8) {
                    errors.push(format!(
                        "NAMESPACE_ROOT[{}]: still points to young={:#x}",
                        ns_id, root
                    ));
                }
            }
        }

        // Check thread roots
        for (tid, root) in self.thread_roots.iter() {
            if BuiltInTypes::is_heap_pointer(*root) {
                let untagged = BuiltInTypes::untag(*root);
                if self.young.contains(untagged as *const u8) {
                    errors.push(format!(
                        "THREAD_ROOT[{:?}]: still points to young={:#x}",
                        tid, root
                    ));
                }
            }
        }

        // Check the old generation heap for references to young
        // Walk all objects in old gen and check their fields
        let young = &self.young;
        self.old.walk_objects(|heap_object| {
            for field in heap_object.get_fields() {
                if BuiltInTypes::is_heap_pointer(*field) {
                    let untagged = BuiltInTypes::untag(*field);
                    if young.contains(untagged as *const u8) {
                        errors.push(format!(
                            "OLD_HEAP[{:?}]: field points to young={:#x}",
                            heap_object.get_pointer(),
                            field
                        ));
                    }
                }
            }
        });

        if !errors.is_empty() {
            eprintln!("\n=== POST-GC VERIFICATION FAILED ===");
            eprintln!(
                "Young generation: {:#x} - {:#x}",
                self.young.start as usize,
                self.young.start as usize + self.young.byte_count()
            );
            for err in &errors {
                eprintln!("  {}", err);
            }
            eprintln!("Total stale pointers: {}", errors.len());
            eprintln!("===================================\n");
        }
    }

    /// Create a heap dump for debugging
    #[cfg(feature = "heap-dump")]
    pub fn create_heap_dump(
        &self,
        label: &str,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize, usize)],
    ) -> super::heap_dump::HeapDump {
        use super::heap_dump::*;
        use std::time::SystemTime;

        let classifier = PointerClassifier::new(
            self.young.start as usize,
            self.young.byte_count(),
            self.young.allocation_offset,
            self.old.space_start(),
            self.old.space_byte_count(),
        );

        // Snapshot young gen
        let young_objects =
            walk_space_objects(self.young.start, self.young.allocation_offset, &classifier);

        let young_gen = SpaceSnapshot {
            name: "young".to_string(),
            start: format!("{:#x}", self.young.start as usize),
            byte_count: self.young.byte_count(),
            allocation_offset: self.young.allocation_offset,
            objects: young_objects,
        };

        // Snapshot old gen
        let old_objects = self.old.collect_objects_for_dump(&classifier);

        let old_gen = SpaceSnapshot {
            name: "old".to_string(),
            start: format!("{:#x}", self.old.space_start()),
            byte_count: self.old.space_byte_count(),
            allocation_offset: 0, // Mark and sweep doesn't track this the same way
            objects: old_objects,
        };

        // Snapshot stacks
        let mut stacks = Vec::new();
        for (idx, (stack_base, frame_pointer, gc_return_addr)) in stack_pointers.iter().enumerate()
        {
            stacks.push(snapshot_stack(
                idx,
                *stack_base,
                *frame_pointer,
                *gc_return_addr,
                stack_map,
                &classifier,
            ));
        }

        // Snapshot roots
        let mut temporary_roots = Vec::new();
        for (i, root) in self.temporary_roots.iter().enumerate() {
            if let Some(ptr) = root {
                temporary_roots.push(RootSnapshot {
                    source: "temporary".to_string(),
                    index: format!("{}", i),
                    value: format!("{:#x}", ptr),
                    tag: tag_name(*ptr),
                    is_heap_ptr: BuiltInTypes::is_heap_pointer(*ptr),
                    points_to: classifier.classify(*ptr),
                });
            }
        }

        let mut namespace_roots = Vec::new();
        for (ns_id, root) in self.namespace_roots.iter() {
            namespace_roots.push(RootSnapshot {
                source: "namespace".to_string(),
                index: format!("{}", ns_id),
                value: format!("{:#x}", root),
                tag: tag_name(*root),
                is_heap_ptr: BuiltInTypes::is_heap_pointer(*root),
                points_to: classifier.classify(*root),
            });
        }

        let mut thread_roots = Vec::new();
        for (tid, root) in self.thread_roots.iter() {
            thread_roots.push(RootSnapshot {
                source: "thread".to_string(),
                index: format!("{:?}", tid),
                value: format!("{:#x}", root),
                tag: tag_name(*root),
                is_heap_ptr: BuiltInTypes::is_heap_pointer(*root),
                points_to: classifier.classify(*root),
            });
        }

        let mut additional_roots = Vec::new();
        for (i, root) in self.additional_roots.iter().enumerate() {
            additional_roots.push(RootSnapshot {
                source: "additional".to_string(),
                index: format!("{}", i),
                value: format!("{:#x}", root),
                tag: tag_name(*root),
                is_heap_ptr: BuiltInTypes::is_heap_pointer(*root),
                points_to: classifier.classify(*root),
            });
        }

        HeapDump {
            timestamp: format!("{:?}", SystemTime::now()),
            label: label.to_string(),
            young_gen,
            old_gen,
            stacks,
            roots: RootsSnapshot {
                temporary_roots,
                namespace_roots,
                thread_roots,
                additional_roots,
            },
        }
    }

    /// Dump heap before and after GC for debugging
    #[cfg(feature = "heap-dump")]
    pub fn dump_before_after_gc(
        &self,
        label: &str,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize, usize)],
        dump_dir: &str,
    ) {
        use std::fs;
        fs::create_dir_all(dump_dir).ok();

        // Save JSON dump
        let dump = self.create_heap_dump(label, stack_map, stack_pointers);
        let path = format!("{}/{}.json", dump_dir, label.replace(" ", "_"));
        if let Err(e) = dump.save(&path) {
            eprintln!("Failed to save heap dump to {}: {}", path, e);
        } else {
            eprintln!("Saved heap dump to {}", path);
        }

        // Save binary dumps
        let binary_dump = self.create_binary_dump(stack_pointers);
        if let Err(e) = binary_dump.save_all(dump_dir, label) {
            eprintln!("Failed to save binary dump: {}", e);
        } else {
            eprintln!("Saved binary dumps to {}/", dump_dir);
        }
    }

    /// Create raw binary dumps of memory regions
    #[cfg(feature = "heap-dump")]
    pub fn create_binary_dump(
        &self,
        stack_pointers: &[(usize, usize, usize)],
    ) -> super::heap_dump::BinaryDumpSet {
        use super::heap_dump::{BinaryDumpSet, RawMemoryDump};

        // Dump young generation (only allocated portion)
        let young_data =
            unsafe { std::slice::from_raw_parts(self.young.start, self.young.allocation_offset) };
        let young_gen = RawMemoryDump {
            label: "young".to_string(),
            start_addr: self.young.start as usize,
            data: young_data.to_vec(),
        };

        // Dump old generation (up to highmark)
        let old_data = unsafe {
            let start = self.old.space_start() as *const u8;
            let len = self.old.highmark();
            std::slice::from_raw_parts(start, len)
        };
        let old_gen = RawMemoryDump {
            label: "old".to_string(),
            start_addr: self.old.space_start(),
            data: old_data.to_vec(),
        };

        // Dump stacks (from frame pointer to stack base)
        let mut stacks = Vec::new();
        for (i, (stack_base, frame_pointer, _gc_return_addr)) in stack_pointers.iter().enumerate() {
            if *frame_pointer < *stack_base {
                let len = stack_base - frame_pointer;
                let stack_data =
                    unsafe { std::slice::from_raw_parts(*frame_pointer as *const u8, len) };
                stacks.push(RawMemoryDump {
                    label: format!("stack_{}", i),
                    start_addr: *frame_pointer,
                    data: stack_data.to_vec(),
                });
            }
        }

        BinaryDumpSet {
            young_gen,
            old_gen,
            stacks,
        }
    }
}

#[cfg(feature = "heap-dump")]
fn tag_name(value: usize) -> String {
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
