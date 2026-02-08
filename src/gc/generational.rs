use std::{error::Error, ffi::c_void, io};

use libc::mprotect;

use super::get_page_size;
use super::usdt_probes;

use crate::{
    collections::TYPE_ID_CONTINUATION,
    runtime::ContinuationObject,
    types::{BuiltInTypes, Header, HeapObject, Word},
};

use super::{
    AllocateAction, Allocator, AllocatorOptions, StackMap,
    continuation_walker::ContinuationSegmentWalker, mark_and_sweep::MarkAndSweep,
    stack_walker::StackWalker,
};

/// Represents a reference to a GC root that needs updating after collection.
/// Points to a mutable slot holding the root value (stack slots, GlobalObjectBlock entries).
struct RootRef(*mut usize);

impl RootRef {
    /// Get the current value of this root
    fn value(&self) -> usize {
        unsafe { *self.0 }
    }
}

const DEFAULT_PAGE_COUNT: usize = 1024;
// Aribtary number that should be changed when I have
// better options for gc
const MAX_PAGE_COUNT: usize = 1000000;

/// Card size in bytes (512 = 2^9)
const CARD_SIZE_LOG2: usize = 9;
const CARD_SIZE: usize = 1 << CARD_SIZE_LOG2;

/// Card table for write barrier tracking.
///
/// Each byte in the table represents one 512-byte "card" of the old generation heap.
/// When a heap store occurs, the card containing the destination is marked dirty.
/// During minor GC, only dirty cards need to be scanned for old-to-young references.
///
/// Card values: 0 = clean, non-zero = dirty
pub struct CardTable {
    /// The card table memory
    cards: Vec<u8>,
    /// Start address of the heap region this table covers
    heap_start: usize,
    /// Number of cards in the table
    card_count: usize,
    /// Biased pointer for fast card marking: cards.as_ptr() - (heap_start >> CARD_SIZE_LOG2)
    /// This allows codegen to compute: biased_ptr[addr >> 9] = 1
    biased_ptr: *mut u8,
    /// Track which cards have been marked dirty (for efficient iteration)
    dirty_card_indices: Vec<usize>,
}

unsafe impl Send for CardTable {}
unsafe impl Sync for CardTable {}

impl CardTable {
    /// Create a new card table covering the given heap range.
    fn new(heap_start: usize, heap_size: usize) -> Self {
        let card_count = heap_size.div_ceil(CARD_SIZE);
        let mut cards = vec![0u8; card_count];
        let biased_ptr = unsafe { cards.as_mut_ptr().sub(heap_start >> CARD_SIZE_LOG2) };
        Self {
            cards,
            heap_start,
            card_count,
            biased_ptr,
            dirty_card_indices: Vec::with_capacity(64),
        }
    }

    /// Mark the card containing the given address as dirty.
    /// This is the fast path used by generated code.
    #[inline]
    pub fn mark_dirty(&mut self, addr: usize) {
        let card_index = (addr - self.heap_start) >> CARD_SIZE_LOG2;
        if card_index < self.card_count {
            // Only add to dirty list if not already dirty
            if self.cards[card_index] == 0 {
                self.cards[card_index] = 1;
                self.dirty_card_indices.push(card_index);
            }
        }
    }

    /// Resize the card table to cover a larger heap.
    /// Called when the old generation grows.
    pub fn resize(&mut self, new_heap_size: usize) {
        let new_card_count = new_heap_size.div_ceil(CARD_SIZE);
        if new_card_count > self.card_count {
            // Extend the card table with clean cards for the new region
            self.cards.resize(new_card_count, 0);
            self.card_count = new_card_count;
            // Recalculate biased pointer (the Vec may have reallocated)
            self.biased_ptr = unsafe {
                self.cards
                    .as_mut_ptr()
                    .sub(self.heap_start >> CARD_SIZE_LOG2)
            };
        }
    }

    /// Check if a card is dirty.
    #[inline]
    #[allow(unused)]
    pub fn is_dirty(&self, card_index: usize) -> bool {
        card_index < self.card_count && self.cards[card_index] != 0
    }

    /// Get the biased pointer for codegen.
    /// Generated code can do: biased_ptr[addr >> 9] = 1
    pub fn biased_ptr(&self) -> *mut u8 {
        self.biased_ptr
    }

    /// Get the start address of a card.
    #[allow(unused)]
    fn card_start(&self, card_index: usize) -> usize {
        self.heap_start + (card_index << CARD_SIZE_LOG2)
    }

    /// Get the end address of a card (exclusive).
    #[allow(unused)]
    fn card_end(&self, card_index: usize) -> usize {
        self.card_start(card_index) + CARD_SIZE
    }

    /// Get the list of dirty card indices (O(1) instead of scanning whole table).
    pub fn dirty_card_indices(&self) -> &[usize] {
        &self.dirty_card_indices
    }

    /// Clear all dirty cards and the tracking list.
    pub fn clear(&mut self) {
        for &card_index in &self.dirty_card_indices {
            self.cards[card_index] = 0;
        }
        self.dirty_card_indices.clear();
    }

    /// Check if there are any dirty cards.
    pub fn has_dirty_cards(&self) -> bool {
        !self.dirty_card_indices.is_empty()
    }
}

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
            if !self.allocation_offset.is_multiple_of(8) {
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

    fn write_object_zeroed(&mut self, offset: usize, size: Word) -> *const u8 {
        let mut heap_object = HeapObject::from_untagged(unsafe { self.start.add(offset) });
        assert!(self.contains(heap_object.get_pointer()));

        // Zero the full object memory (header + fields)
        // Used for arrays which don't initialize all fields
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

    fn allocate_zeroed(&mut self, size: Word) -> *const u8 {
        let offset = self.allocation_offset;
        let header_size = if size.to_words() > Header::MAX_INLINE_SIZE {
            16
        } else {
            8
        };
        let full_size = size.to_bytes() + header_size;
        let pointer = self.write_object_zeroed(offset, size);
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
    atomic_pause: [u8; 8],
    options: AllocatorOptions,
    /// Remembered set: old gen objects that contain pointers to young gen.
    /// Each entry is a tagged pointer to an old gen object whose fields need scanning.
    /// Note: This is used for Rust code write barriers. Card table is used for generated code.
    remembered_set: Vec<usize>,
    /// Card table for tracking writes to old generation from generated code.
    /// Each 512-byte region ("card") of old gen has one byte in this table.
    card_table: CardTable,
}

impl GenerationalGC {
    fn update_continuation_segments(&mut self, stack_map: &StackMap) {
        let runtime = crate::get_runtime().get_mut();

        // Fast path: skip if no invocation return points
        if runtime.invocation_return_points.is_empty() {
            return;
        }

        // Process InvocationReturnPoint saved frames
        for (_thread_id, rps) in runtime.invocation_return_points.iter_mut() {
            for rp in rps.iter_mut() {
                if rp.saved_stack_frame.is_empty() {
                    continue;
                }
                self.update_segment_young_gen_pointers(
                    &mut rp.saved_stack_frame,
                    rp.stack_pointer,
                    rp.frame_pointer,
                    rp.frame_pointer,
                    stack_map,
                );
            }
        }
    }

    /// Update saved_continuation_ptr values that point to young gen objects.
    /// These are continuation objects saved during invoke_continuation_runtime
    /// that may be collected or moved during GC.
    fn update_saved_continuation_ptrs(&mut self) {
        let runtime = crate::get_runtime().get_mut();

        for (_thread_id, cont_ptr) in runtime.saved_continuation_ptr.iter_mut() {
            if *cont_ptr == 0 {
                continue;
            }
            if !BuiltInTypes::is_heap_pointer(*cont_ptr) {
                continue;
            }
            let untagged = BuiltInTypes::untag(*cont_ptr);
            // Only process if in young gen
            if self.young.contains(untagged as *const u8) {
                // Copy to old gen and update the pointer
                let new_ptr = self.copy(*cont_ptr);
                *cont_ptr = new_ptr;
            }
        }
    }

    fn update_segment_young_gen_pointers(
        &mut self,
        segment: &mut [u8],
        original_sp: usize,
        original_fp: usize,
        prompt_sp: usize,
        stack_map: &StackMap,
    ) {
        ContinuationSegmentWalker::update_segment_pointers(
            segment,
            original_sp,
            original_fp,
            prompt_sp,
            stack_map,
            |old_value| self.copy(old_value),
        );
    }

    fn update_continuation_segment(&mut self, cont: &ContinuationObject, stack_map: &StackMap) {
        cont.with_segment_bytes_mut(|segment| {
            if segment.is_empty() {
                return;
            }
            self.update_segment_young_gen_pointers(
                segment,
                cont.original_sp(),
                cont.original_fp(),
                cont.prompt_stack_pointer(),
                stack_map,
            );
        });
    }
}

impl Allocator for GenerationalGC {
    fn new(options: AllocatorOptions) -> Self {
        let young = Space::new(DEFAULT_PAGE_COUNT * 10);
        let old = MarkAndSweep::new_with_page_count(DEFAULT_PAGE_COUNT * 100, options);
        // Create card table covering the old generation
        let card_table = CardTable::new(old.heap_start(), old.heap_size());
        Self {
            young,
            old,
            copied: vec![],
            gc_count: 0,
            full_gc_frequency: 100,
            atomic_pause: [0; 8],
            options,
            remembered_set: Vec::with_capacity(64),
            card_table,
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

    fn try_allocate_zeroed(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let pointer = self.allocate_inner_zeroed(words, kind)?;
        Ok(pointer)
    }

    fn gc(&mut self, stack_map: &super::StackMap, stack_pointers: &[(usize, usize, usize)]) {
        if !self.options.gc {
            return;
        }
        if self.gc_count != 0 && self.gc_count.is_multiple_of(self.full_gc_frequency) {
            self.gc_count = 0;
            self.full_gc(stack_map, stack_pointers);
        } else {
            self.minor_gc(stack_map, stack_pointers);
        }
        self.gc_count += 1;
    }

    fn grow(&mut self) {
        self.old.grow();
        // Resize card table to cover the expanded old generation
        self.card_table.resize(self.old.heap_size());
    }

    /// Allocate a long-lived runtime object directly in old generation.
    fn allocate_for_runtime(&mut self, words: usize) -> Result<usize, Box<dyn std::error::Error>> {
        // Allocate in old generation - these objects are long-lived
        match self.old.try_allocate(words, BuiltInTypes::HeapObject)? {
            super::AllocateAction::Allocated(ptr) => {
                Ok(BuiltInTypes::HeapObject.tag(ptr as isize) as usize)
            }
            super::AllocateAction::Gc => Err("Need GC to allocate runtime object".into()),
        }
    }

    #[allow(unused)]
    fn get_pause_pointer(&self) -> usize {
        self.atomic_pause.as_ptr() as usize
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }

    /// Write barrier: record old-to-young pointers.
    ///
    /// Called after writing a pointer into a heap object. If the object is in
    /// old gen, we mark its card as dirty so minor GC will scan it.
    ///
    /// ## Two Mechanisms
    ///
    /// 1. **Remembered set** (precise): Used by Rust code for exact tracking
    /// 2. **Card table** (fast): Used by generated code for cheap marking
    ///
    /// Both are processed during minor GC. The card table is also marked here
    /// so that Rust code writes are tracked by both mechanisms.
    fn write_barrier(&mut self, object_ptr: usize, new_value: usize) {
        // Only care about heap pointer values
        if !BuiltInTypes::is_heap_pointer(new_value) {
            return;
        }

        let new_value_untagged = BuiltInTypes::untag(new_value);

        // Only care if the new value is in young gen
        if !self.young.contains(new_value_untagged as *const u8) {
            return;
        }

        // Only care if the object being written to is in old gen
        if !BuiltInTypes::is_heap_pointer(object_ptr) {
            return;
        }

        let object_untagged = BuiltInTypes::untag(object_ptr);
        if !self.old.contains(object_untagged as *const u8) {
            return;
        }

        // Mark the card as dirty (for generated code compatibility)
        self.card_table.mark_dirty(object_untagged);

        // Also add to remembered set (precise tracking for Rust code)
        if !self.remembered_set.contains(&object_ptr) {
            #[cfg(feature = "debug-gc")]
            eprintln!(
                "[GC DEBUG] write_barrier: adding old-gen object {:#x} to remembered set (points to young-gen {:#x})",
                object_ptr, new_value
            );
            self.remembered_set.push(object_ptr);
        }
    }

    fn get_card_table_biased_ptr(&self) -> *mut u8 {
        self.card_table.biased_ptr()
    }

    fn mark_card_unconditional(&mut self, object_ptr: usize) {
        // Only care if the object is a heap pointer
        if !BuiltInTypes::is_heap_pointer(object_ptr) {
            return;
        }

        let object_untagged = BuiltInTypes::untag(object_ptr);

        // Only mark if the object is in old gen (card table only covers old gen)
        if self.old.contains(object_untagged as *const u8) {
            self.card_table.mark_dirty(object_untagged);
        }
    }
}

impl GenerationalGC {
    /// Check if an object is too large to ever fit in the young generation,
    /// even when it's completely empty.
    fn too_large_for_young(&self, words: usize) -> bool {
        let header_size = if words > Header::MAX_INLINE_SIZE {
            16
        } else {
            8
        };
        let alloc_size = words * 8 + header_size;
        alloc_size > self.young.byte_count()
    }

    fn allocate_inner(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        // Large objects that can never fit in young gen go directly to old gen
        if self.too_large_for_young(words) {
            return self.old.try_allocate(words, kind);
        }
        let size = Word::from_word(words);
        if self.young.can_allocate(size) {
            let ptr = self.young.allocate(size);
            // Float objects are opaque (their field is a raw f64, not a pointer).
            // Set the opaque bit immediately so GC never sees a non-opaque float.
            if kind == BuiltInTypes::Float {
                unsafe {
                    *(ptr as *mut usize) |= 0x2; // Set opaque bit (bit 1)
                }
            }
            Ok(AllocateAction::Allocated(ptr))
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    fn allocate_inner_zeroed(
        &mut self,
        words: usize,
        _kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        // Large objects that can never fit in young gen go directly to old gen
        if self.too_large_for_young(words) {
            return self.old.try_allocate_zeroed(words, _kind);
        }
        let size = Word::from_word(words);
        if self.young.can_allocate(size) {
            let ptr = self.young.allocate_zeroed(size);
            Ok(AllocateAction::Allocated(ptr))
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    // ==================== GATHER FUNCTIONS ====================

    /// Gather stack roots as RootRefs.
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

            for (slot_addr, _value) in young_roots {
                slots.push(RootRef(slot_addr as *mut usize));
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

        StackWalker::walk_stack_roots_with_return_addr(
            stack_base,
            frame_pointer,
            gc_return_addr,
            stack_map,
            |slot_addr, slot_value| {
                let untagged = BuiltInTypes::untag(slot_value);

                // Skip values where the untagged pointer is 0 (e.g., value 0b110 = 6)
                if untagged == 0 {
                    return;
                }

                if self.young.contains(untagged as *const u8) {
                    assert!(
                        self.young.contains_allocated(untagged as *const u8),
                        "Young gen pointer {:#x} not in allocated region",
                        untagged
                    );
                    roots.push((slot_addr, slot_value));
                } else {
                    assert!(
                        self.old.contains(untagged as *const u8),
                        "Heap pointer {:#x} (tagged {:#x}) neither in young nor old gen. Stack slot @ {:#x}",
                        untagged,
                        slot_value,
                        slot_addr
                    );

                    old_gen_objects.push(slot_value);
                }
            },
        );

        (roots, old_gen_objects)
    }

    // ==================== UNIFIED ROOT PROCESSING ====================

    /// Update a root slot with its new value after GC processing
    fn update_root(&self, root: &RootRef, new_value: usize) {
        unsafe {
            *root.0 = new_value;
        }
    }

    /// Process all roots - copy young gen objects to old gen
    fn process_all_roots(&mut self, roots: Vec<RootRef>) {
        for root_ref in roots {
            let old_value = root_ref.value();

            if !BuiltInTypes::is_heap_pointer(old_value) {
                continue;
            }

            let heap_object = HeapObject::from_tagged(old_value);

            // Skip if not in young gen
            if !self.young.contains(heap_object.get_pointer()) {
                continue;
            }

            // Copy to old gen
            let new_value = self.copy(old_value);

            self.update_root(&root_ref, new_value);
        }
    }

    fn minor_gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize, usize)]) {
        let start = std::time::Instant::now();
        usdt_probes::fire_gc_minor_start(self.gc_count);

        self.gc_count += 1;

        let (stack_roots, stack_old_gen) = self.gather_stack_root_refs(stack_map, stack_pointers);
        self.process_all_roots(stack_roots);

        // Update continuation segments
        self.update_continuation_segments(stack_map);

        // Update saved_continuation_ptr values (continuation objects saved in Rust runtime)
        self.update_saved_continuation_ptrs();

        // Process old gen objects found on stack - update their young gen references.
        // This scans one level deep for old gen objects directly referenced from stack.
        for old_root in stack_old_gen {
            self.process_old_gen_object(old_root, stack_map);
        }

        // Process remembered set - old gen objects that were mutated to point to young gen.
        // The write barrier recorded these when pointers were written.
        // Take ownership of the remembered set so we can clear it after processing.
        let remembered = std::mem::take(&mut self.remembered_set);
        #[cfg(feature = "debug-gc")]
        if !remembered.is_empty() {
            eprintln!(
                "[GC DEBUG] Processing {} remembered set entries",
                remembered.len()
            );
        }
        for old_object in remembered {
            #[cfg(feature = "debug-gc")]
            eprintln!("[GC DEBUG] Processing remembered object {:#x}", old_object);
            self.process_old_gen_object(old_object, stack_map);
        }

        // Process dirty cards - cards marked by generated code write barriers.
        // We need to scan all objects in dirty cards for young gen references.
        self.process_dirty_cards(stack_map);

        self.copy_remaining(stack_map);

        self.young.clear();

        // Clear only the dirty cards (much faster than clearing entire table)
        self.card_table.clear();

        usdt_probes::fire_gc_minor_end(self.gc_count);
        if self.options.print_stats {
            println!("Minor gc took {:?}", start.elapsed());
        }
    }

    /// Process dirty cards from the card table.
    /// Scans all objects in old gen that are in dirty cards for young gen references.
    fn process_dirty_cards(&mut self, stack_map: &StackMap) {
        // Use the tracked dirty card indices (O(1) to get, no scanning)
        if !self.card_table.has_dirty_cards() {
            return;
        }

        // Copy the dirty card indices to a HashSet for O(1) lookup
        let dirty_cards: std::collections::HashSet<usize> = self
            .card_table
            .dirty_card_indices()
            .iter()
            .copied()
            .collect();

        #[cfg(feature = "debug-gc")]
        eprintln!("[GC DEBUG] Processing {} dirty cards", dirty_cards.len());

        // Collect objects in dirty cards
        // We need to do this in two passes because we can't borrow old mutably
        // while also borrowing card_table
        let old_start = self.old.heap_start();
        let mut objects_to_process: Vec<usize> = Vec::new();

        self.old.walk_objects_mut(|obj_addr, heap_obj| {
            let card_index = (obj_addr - old_start) >> CARD_SIZE_LOG2;
            if dirty_cards.contains(&card_index) {
                // Tag the object address for processing
                let tagged = BuiltInTypes::HeapObject.tag(obj_addr as isize) as usize;
                objects_to_process.push(tagged);

                #[cfg(feature = "debug-gc")]
                eprintln!(
                    "[GC DEBUG] Object at {:#x} is in dirty card {}",
                    obj_addr, card_index
                );
                let _ = heap_obj; // suppress unused warning
            }
        });

        // Now process each object
        for old_object in objects_to_process {
            self.process_old_gen_object(old_object, stack_map);
        }
    }

    /// Process an old gen object's fields, copying any young gen references to old gen.
    fn process_old_gen_object(&mut self, old_object: usize, stack_map: &StackMap) {
        let mut heap_obj = HeapObject::from_tagged(old_object);

        // Process this old gen object's fields
        let data = heap_obj.get_fields_mut();
        let mut continuation_fields = Vec::new();
        #[cfg(feature = "debug-gc")]
        eprintln!(
            "[GC DEBUG] process_old_gen_object {:#x}: {} fields",
            old_object,
            data.len()
        );
        for (i, field) in data.iter_mut().enumerate() {
            let _ = i; // Suppress unused variable warning when debug-gc is disabled
            if BuiltInTypes::is_heap_pointer(*field) {
                let field_obj = HeapObject::from_tagged(*field);
                let field_ptr = field_obj.get_pointer();

                if self.young.contains(field_ptr) {
                    #[cfg(feature = "debug-gc")]
                    eprintln!(
                        "[GC DEBUG]   field[{}] = {:#x} is in young gen, copying",
                        i, *field
                    );
                    // Young gen reference - copy to old gen and update field
                    let new_value = self.copy(*field);
                    *field = new_value;
                    #[cfg(feature = "debug-gc")]
                    eprintln!("[GC DEBUG]   -> new value: {:#x}", new_value);
                }
                if field_obj.get_type_id() == TYPE_ID_CONTINUATION as usize {
                    continuation_fields.push(*field);
                }
            }
        }

        if heap_obj.get_type_id() == TYPE_ID_CONTINUATION as usize
            && let Some(cont) = ContinuationObject::from_heap_object(heap_obj)
        {
            self.update_continuation_segment(&cont, stack_map);
        }

        for cont_ptr in continuation_fields {
            if let Some(cont) = ContinuationObject::from_tagged(cont_ptr) {
                self.update_continuation_segment(&cont, stack_map);
            }
        }
    }

    // ==================== COPY LOGIC ====================

    /// Scan all objects in young gen for corrupted float headers.
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

    fn copy_remaining(&mut self, stack_map: &StackMap) {
        #[cfg(feature = "debug-gc")]
        let mut iterations = 0;
        while let Some(mut object) = self.copied.pop() {
            #[cfg(feature = "debug-gc")]
            {
                iterations += 1;
                eprintln!(
                    "[GC DEBUG] copy_remaining iteration {}: processing object at {:#x}",
                    iterations,
                    object.untagged()
                );
            }
            let mut continuation_fields = Vec::new();
            for field in object.get_fields_mut().iter_mut() {
                if BuiltInTypes::is_heap_pointer(*field) {
                    let heap_obj = HeapObject::from_tagged(*field);
                    if self.young.contains(heap_obj.get_pointer()) {
                        #[cfg(feature = "debug-gc")]
                        eprintln!("[GC DEBUG]   copying young-gen field {:#x}", *field);
                        *field = self.copy(*field);
                    }
                    if heap_obj.get_type_id() == TYPE_ID_CONTINUATION as usize {
                        continuation_fields.push(*field);
                    }
                }
            }

            if object.get_type_id() == TYPE_ID_CONTINUATION as usize
                && let Some(cont) = ContinuationObject::from_heap_object(object)
            {
                self.update_continuation_segment(&cont, stack_map);
            }

            for cont_ptr in continuation_fields {
                if let Some(cont) = ContinuationObject::from_tagged(cont_ptr) {
                    self.update_continuation_segment(&cont, stack_map);
                }
            }
        }
        #[cfg(feature = "debug-gc")]
        eprintln!(
            "[GC DEBUG] copy_remaining done after {} iterations",
            iterations
        );
    }

    fn full_gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize, usize)]) {
        usdt_probes::fire_gc_full_start(self.gc_count);
        self.minor_gc(stack_map, stack_pointers);
        self.old.gc(stack_map, stack_pointers);
        usdt_probes::fire_gc_full_end(self.gc_count);
    }
}
