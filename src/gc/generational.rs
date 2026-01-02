use std::{collections::HashMap, error::Error, ffi::c_void, io, thread::ThreadId};

use libc::mprotect;

use super::get_page_size;
use super::usdt_probes;

use crate::types::{BuiltInTypes, Header, HeapObject, Word};

use crate::collections::{HandleArenaPtr, RootSetPtr};

use super::{AllocateAction, Allocator, AllocatorOptions, StackMap, mark_and_sweep::MarkAndSweep};

/// Represents a reference to a GC root that needs updating after collection.
/// Each variant knows how to read its current value and write back the new value.
enum RootRef {
    /// Points to a mutable slot holding the root value (stack, RootSet, HandleArena)
    Slot(*mut usize),

    /// Temporary root with index for later update
    Temporary { index: usize, value: usize },

    /// Namespace root that needs relocation tracking
    Namespace { namespace_id: usize, value: usize },

    /// Thread root that needs relocation tracking
    Thread { thread_id: ThreadId, value: usize },
}

impl RootRef {
    /// Get the current value of this root
    fn value(&self) -> usize {
        match self {
            RootRef::Slot(ptr) => unsafe { **ptr },
            RootRef::Temporary { value, .. } => *value,
            RootRef::Namespace { value, .. } => *value,
            RootRef::Thread { value, .. } => *value,
        }
    }
}

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

        // Zero the full object memory (header + fields) to prevent stale pointers
        // from previous GC cycles being seen as valid heap pointers
        let header_size = if size.to_words() > Header::MAX_INLINE_SIZE {
            16
        } else {
            8
        };
        let full_size = size.to_bytes() + header_size;
        unsafe {
            std::ptr::write_bytes(self.start.add(offset) as *mut u8, 0, full_size);
        }

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
        if self.gc_count != 0 && self.gc_count % self.full_gc_frequency == 0 {
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

    fn gc_add_root(&mut self, _old: usize) {
        // No-op: All old-gen roots are now handled uniformly during GC.
        // This function is kept for trait compatibility.
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
            let ptr = self.young.allocate(size);
            Ok(AllocateAction::Allocated(ptr))
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    // ==================== GATHER FUNCTIONS ====================

    /// Gather temporary roots as RootRefs
    fn gather_temporary_root_refs(&self) -> Vec<RootRef> {
        self.temporary_roots
            .iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.map(|value| RootRef::Temporary { index: i, value }))
            .collect()
    }

    /// Gather roots from RootSets and HandleArenas as Slot refs
    fn gather_slot_refs(&self) -> Vec<RootRef> {
        let mut slots = Vec::new();

        // Gather from RootSets
        for root_set_ptr in self.root_sets.iter().filter_map(|slot| *slot) {
            let roots = unsafe { &mut *root_set_ptr.0 };
            for root in roots.roots_mut() {
                slots.push(RootRef::Slot(root as *mut usize));
            }
        }

        // Gather from HandleArenas
        for arena_ptr in self.handle_arenas.iter().filter_map(|slot| *slot) {
            let arena = unsafe { &mut *arena_ptr.0 };
            for root in arena.roots_mut() {
                slots.push(RootRef::Slot(root as *mut usize));
            }
        }

        slots
    }

    /// Gather namespace roots, taking ownership of the current list
    fn gather_namespace_root_refs(&mut self) -> Vec<RootRef> {
        std::mem::take(&mut self.namespace_roots)
            .into_iter()
            .map(|(ns_id, val)| RootRef::Namespace {
                namespace_id: ns_id,
                value: val,
            })
            .collect()
    }

    /// Gather thread roots, draining the current map
    fn gather_thread_root_refs(&mut self) -> Vec<RootRef> {
        self.thread_roots
            .drain()
            .map(|(tid, val)| RootRef::Thread {
                thread_id: tid,
                value: val,
            })
            .collect()
    }

    /// Gather stack roots as Slot refs.
    /// Returns (slot_refs, old_gen_values) - old gen values need field updates.
    fn gather_stack_root_refs(
        &self,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize, usize)],
    ) -> (Vec<RootRef>, Vec<usize>) {
        let mut slots = Vec::new();
        let mut old_gen_values = Vec::new();

        for (stack_base, frame_pointer, gc_return_addr) in stack_pointers {
            let (young_roots, old_roots) = self.gather_stack_roots_inner(
                *stack_base,
                stack_map,
                *frame_pointer,
                *gc_return_addr,
            );

            // Convert (slot_addr, value) pairs to RootRef::Slot
            for (slot_addr, _value) in young_roots {
                slots.push(RootRef::Slot(slot_addr as *mut usize));
            }

            old_gen_values.extend(old_roots);
        }

        (slots, old_gen_values)
    }

    /// Inner function to gather roots from a single stack
    fn gather_stack_roots_inner(
        &self,
        stack_base: usize,
        stack_map: &StackMap,
        frame_pointer: usize,
        gc_return_addr: usize,
    ) -> (Vec<(usize, usize)>, Vec<usize>) {
        let mut roots: Vec<(usize, usize)> = Vec::with_capacity(36);
        let mut old_gen_objects: Vec<usize> = Vec::with_capacity(16);

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

                        if untagged % 8 != 0 {
                            continue;
                        }

                        if self.young.contains(untagged as *const u8) {
                            if !self.young.contains_allocated(untagged as *const u8) {
                                continue;
                            }

                            let heap_object = HeapObject::from_tagged(slot_value);
                            let header = heap_object.get_header();
                            let tag = BuiltInTypes::get_kind(slot_value);

                            if matches!(tag, BuiltInTypes::Closure)
                                && header.large
                                && !header.marked
                            {
                                continue;
                            }

                            roots.push((slot_addr, slot_value));
                        } else {
                            // CRITICAL: Verify this is actually in old gen, not just "not young"
                            // A heap pointer that's neither in young nor old is invalid (e.g., stack address)
                            if !self.old.contains(untagged as *const u8) {
                                continue;
                            }

                            let heap_object = HeapObject::from_tagged(slot_value);
                            let header = heap_object.get_header();

                            // Skip values that look like interior pointers (corrupted headers)
                            if header.size > 100 {
                                continue;
                            }

                            old_gen_objects.push(slot_value);
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

        (roots, old_gen_objects)
    }

    // ==================== UNIFIED ROOT PROCESSING ====================

    /// Re-insert a root that doesn't need updating (for namespace/thread roots)
    fn reinsert_root(&mut self, root: &RootRef, value: usize) {
        match root {
            RootRef::Namespace { namespace_id, .. } => {
                self.namespace_roots.push((*namespace_id, value));
            }
            RootRef::Thread { thread_id, .. } => {
                self.thread_roots.insert(*thread_id, value);
            }
            _ => {} // Slot/Temporary don't need reinsertion
        }
    }

    /// Update a root with its new value after GC processing
    fn update_root(&mut self, root: &RootRef, old_value: usize, new_value: usize) {
        match root {
            RootRef::Slot(ptr) => unsafe {
                **ptr = new_value;
            },
            RootRef::Temporary { index, .. } => {
                self.temporary_roots[*index] = Some(new_value);
            }
            RootRef::Namespace { namespace_id, .. } => {
                self.namespace_roots.push((*namespace_id, new_value));
                // Track the relocation so runtime can update its namespace variables
                if old_value != new_value {
                    // Find or create the entry for this namespace
                    if let Some(entry) = self
                        .relocated_namespace_roots
                        .iter_mut()
                        .find(|(ns, _)| *ns == *namespace_id)
                    {
                        entry.1.push((old_value, new_value));
                    } else {
                        self.relocated_namespace_roots
                            .push((*namespace_id, vec![(old_value, new_value)]));
                    }
                }
            }
            RootRef::Thread { thread_id, .. } => {
                self.thread_roots.insert(*thread_id, new_value);
            }
        }
    }

    /// Process all roots uniformly - copy young gen objects to old gen
    fn process_all_roots(&mut self, roots: Vec<RootRef>) {
        for root_ref in roots {
            let old_value = root_ref.value();

            if !BuiltInTypes::is_heap_pointer(old_value) {
                self.reinsert_root(&root_ref, old_value);
                continue;
            }

            let heap_object = HeapObject::from_tagged(old_value);

            // Skip if not in young gen
            if !self.young.contains(heap_object.get_pointer()) {
                self.reinsert_root(&root_ref, old_value);
                continue;
            }

            // Copy to old gen
            let new_value = self.copy(old_value);
            self.update_root(&root_ref, old_value, new_value);
        }
    }

    fn minor_gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize, usize)]) {
        let start = std::time::Instant::now();
        usdt_probes::fire_gc_minor_start(self.gc_count);

        self.gc_count += 1;

        // 1. GATHER all roots
        let mut all_roots = Vec::new();
        all_roots.extend(self.gather_temporary_root_refs());
        all_roots.extend(self.gather_slot_refs());
        all_roots.extend(self.gather_namespace_root_refs());
        all_roots.extend(self.gather_thread_root_refs());

        // Stack roots are handled separately because they return old-gen values too
        let (stack_roots, stack_old_gen) = self.gather_stack_root_refs(stack_map, stack_pointers);
        all_roots.extend(stack_roots);

        // 2. PROCESS all roots uniformly
        self.process_all_roots(all_roots);

        // 3. Update fields of old-gen objects found on the stack
        for old_root in stack_old_gen {
            let heap_obj = HeapObject::from_tagged(old_root);
            let header = heap_obj.get_header();
            if header.size > 100 {
                continue;
            }
            self.move_objects_referenced_from_old_to_old(&mut HeapObject::from_tagged(old_root));
        }
        self.copy_remaining();

        // 4. SYNC to old generation GC
        // These syncs are CRITICAL for old.gc() to mark all live objects.
        // Without them, objects protected by these roots could be swept!
        self.update_old_generation_namespace_roots();
        self.update_old_generation_thread_roots();
        self.old.sync_temporary_roots(&self.temporary_roots);
        self.old.sync_root_sets(&self.root_sets);
        self.old.sync_handle_arenas(&self.handle_arenas);

        // Reset young gen for new allocations
        self.young.clear();

        usdt_probes::fire_gc_minor_end(self.gc_count);
        if self.options.print_stats {
            println!("Minor gc took {:?}", start.elapsed());
        }
    }

    fn update_old_generation_namespace_roots(&mut self) {
        for (namespace_id, root) in &self.namespace_roots {
            self.old.add_namespace_root(*namespace_id, *root);
        }
    }

    fn update_old_generation_thread_roots(&mut self) {
        for (thread_id, root) in self.thread_roots.iter() {
            self.old.add_thread_root(*thread_id, *root);
        }
    }

    // ==================== COPY LOGIC ====================

    fn copy(&mut self, root: usize) -> usize {
        if !BuiltInTypes::is_heap_pointer(root) {
            return root;
        }

        let heap_object = HeapObject::from_tagged(root);
        let tag = BuiltInTypes::get_kind(root);

        // Skip if not in young gen
        if !self.young.contains(heap_object.get_pointer()) {
            return root;
        }

        // Check if already forwarded (forwarding bit set in header)
        let untagged = heap_object.untagged();
        let pointer = untagged as *mut usize;
        let header_data = unsafe { *pointer };
        if Header::is_forwarding_bit_set(header_data) {
            // The header contains the forwarding pointer with forwarding bit set
            return Header::clear_forwarding_bit(header_data);
        }

        let header = heap_object.get_header();

        // Skip objects with invalid size
        if header.size > 100 {
            return root;
        }

        // Copy object data to old generation
        let data = heap_object.get_full_object_data();
        let new_pointer = self.old.copy_data_to_offset(data);

        // Get the new object and add to processing queue
        let new_object = HeapObject::from_untagged(new_pointer);
        self.copied.push(new_object);

        // Store forwarding pointer in header for all objects (works even for 0-field objects)
        let tagged_new = tag.tag(new_pointer as isize) as usize;
        unsafe { *pointer = Header::set_forwarding_bit(tagged_new) };

        tagged_new
    }

    fn copy_remaining(&mut self) {
        while let Some(mut object) = self.copied.pop() {
            for field in object.get_fields_mut().iter_mut() {
                if BuiltInTypes::is_heap_pointer(*field) {
                    let heap_obj = HeapObject::from_tagged(*field);
                    if self.young.contains(heap_obj.get_pointer()) {
                        *field = self.copy(*field);
                    }
                }
            }
        }
    }

    fn move_objects_referenced_from_old_to_old(&mut self, old_object: &mut HeapObject) {
        let object_ptr = old_object.get_pointer();

        // Skip if in young gen
        if self.young.contains(object_ptr) {
            return;
        }

        let header = old_object.get_header();

        // Skip objects with invalid size
        if header.size > 100 {
            return;
        }

        let data = old_object.get_fields_mut();

        for field in data.iter_mut() {
            if BuiltInTypes::is_heap_pointer(*field) {
                let heap_obj = HeapObject::from_tagged(*field);

                // Only copy if in young gen
                if self.young.contains(heap_obj.get_pointer()) {
                    let new_value = self.copy(*field);
                    *field = new_value;
                }
            }
        }
    }

    fn full_gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize, usize)]) {
        usdt_probes::fire_gc_full_start(self.gc_count);
        self.minor_gc(stack_map, stack_pointers);
        self.old.gc(stack_map, stack_pointers);
        usdt_probes::fire_gc_full_end(self.gc_count);
    }
}
