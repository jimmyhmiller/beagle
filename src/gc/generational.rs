use std::{error::Error, ffi::c_void, io};

use libc::mprotect;

use super::get_page_size;
use super::usdt_probes;

use crate::types::{BuiltInTypes, Header, HeapObject, Word};

use super::{
    AllocateAction, Allocator, AllocatorOptions, mark_and_sweep::MarkAndSweep,
    stack_walker::StackWalker,
};
use crate::builtins::reset_shift::ContinuationObject;
use crate::collections::TYPE_ID_FRAME;

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

    fn allocation_offset(&self) -> usize {
        self.allocation_offset
    }

    fn base_address(&self) -> usize {
        self.start as usize
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
    fn scan_continuation_segment<F>(&self, object: &HeapObject, mut callback: F)
    where
        F: FnMut(usize, usize),
    {
        let object = HeapObject::from_untagged(object.untagged() as *const u8);
        let cont_obj_addr = object.untagged();
        let Some(cont) = ContinuationObject::from_heap_object(object) else {
            return;
        };
        let Some((segment_base, segment_top, gc_frame_top)) = cont.segment_gc_frame_info() else {
            return;
        };
        if gc_frame_top >= segment_base && gc_frame_top < segment_top {
            StackWalker::walk_segment_gc_roots(
                gc_frame_top,
                segment_base,
                segment_top,
                |slot_addr, slot_value| {
                    let untagged = BuiltInTypes::untag(slot_value);
                    if untagged != 0
                        && self.young.contains(untagged as *const u8)
                        && !self.young.contains_allocated(untagged as *const u8)
                    {
                        let mut owner_frame = None;
                        let mut header_cursor = gc_frame_top;
                        while header_cursor >= segment_base
                            && header_cursor < segment_top
                            && header_cursor != 0
                        {
                            let owner_header =
                                Header::from_usize(unsafe { *(header_cursor as *const usize) });
                            if owner_header.type_id != TYPE_ID_FRAME {
                                break;
                            }
                            let owner_slots = owner_header.size as usize;
                            let owner_low = header_cursor
                                .saturating_sub(16)
                                .saturating_sub(owner_slots.saturating_sub(1) * 8);
                            let owner_high = header_cursor.saturating_sub(16);
                            if slot_addr >= owner_low && slot_addr <= owner_high {
                                let slot_index = (owner_high - slot_addr) / 8;
                                let owner_fp = header_cursor + 8;
                                let owner_ret = unsafe { *((owner_fp + 8) as *const usize) };
                                let owner_fn = crate::get_runtime()
                                    .get()
                                    .get_function_containing_pointer(owner_ret as *const u8)
                                    .map(|(function, offset)| {
                                        format!("{}+{:#x}", function.name, offset)
                                    })
                                    .unwrap_or_else(|| "unknown".to_string());
                                owner_frame = Some((
                                    header_cursor,
                                    slot_index,
                                    owner_slots,
                                    owner_ret,
                                    owner_fn,
                                ));
                                break;
                            }
                            let prev = unsafe { *((header_cursor - 8) as *const usize) };
                            if prev < segment_base || prev >= segment_top {
                                break;
                            }
                            header_cursor = prev;
                        }
                        let header_addr = slot_addr + 16;
                        let frame_pointer = header_addr + 8;
                        let return_addr = unsafe { *((frame_pointer + 8) as *const usize) };
                        let dump = (-4isize..=12isize)
                            .map(|index| {
                                let addr = (frame_pointer as isize + index * 8) as usize;
                                let value = unsafe { *(addr as *const usize) };
                                format!("{:#x}:{:#x}", addr, value)
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        if let Some((function, offset)) = crate::get_runtime()
                            .get()
                            .get_function_containing_pointer(return_addr as *const u8)
                        {
                            eprintln!(
                                "[cont-segment-invalid-young-root] cont_obj={:#x} slot_addr={:#x} slot_value={:#x} fp={:#x} ret={:#x} fn={}+{:#x}{} segment_base={:#x} segment_top={:#x} words={}",
                                cont_obj_addr,
                                slot_addr,
                                slot_value,
                                frame_pointer,
                                return_addr,
                                function.name,
                                offset,
                                owner_frame
                                    .map(|(header, slot_index, owner_slots, owner_ret, owner_fn)| format!(
                                        " owner_header={:#x} owner_fp={:#x} owner_ret={:#x} owner_fn={} slot_index={} owner_slots={}",
                                        header,
                                        header + 8,
                                        owner_ret,
                                        owner_fn,
                                        slot_index,
                                        owner_slots
                                    ))
                                    .unwrap_or_default(),
                                segment_base,
                                segment_top,
                                dump,
                            );
                        } else {
                            eprintln!(
                                "[cont-segment-invalid-young-root] cont_obj={:#x} slot_addr={:#x} slot_value={:#x} fp={:#x} ret={:#x}{} segment_base={:#x} segment_top={:#x} words={}",
                                cont_obj_addr,
                                slot_addr,
                                slot_value,
                                frame_pointer,
                                return_addr,
                                owner_frame
                                    .map(|(header, slot_index, owner_slots, owner_ret, owner_fn)| format!(
                                        " owner_header={:#x} owner_fp={:#x} owner_ret={:#x} owner_fn={} slot_index={} owner_slots={}",
                                        header,
                                        header + 8,
                                        owner_ret,
                                        owner_fn,
                                        slot_index,
                                        owner_slots
                                    ))
                                    .unwrap_or_default(),
                                segment_base,
                                segment_top,
                                dump,
                            );
                        }
                    }
                    callback(slot_addr, slot_value)
                },
            );
        }
        // Restore relative offsets after GC scanning
        cont.make_fp_links_relative_again();
    }

    fn collect_continuation_segment_slots(&self, object: &HeapObject) -> Vec<(usize, usize)> {
        let mut slots = Vec::new();
        self.scan_continuation_segment(object, |slot_addr, slot_value| {
            slots.push((slot_addr, slot_value));
        });
        slots
    }

    fn should_collect_old_before_minor(&self) -> bool {
        let promotion_budget = self.young.allocation_offset();
        promotion_budget != 0 && self.old.free_bytes() < promotion_budget
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

    fn gc(&mut self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]) {
        if !self.options.gc {
            return;
        }
        if self.options.gc_always {
            self.gc_count = 0;
            self.full_gc(gc_frame_tops, extra_roots);
            return;
        }
        let periodic_full_gc =
            self.gc_count != 0 && self.gc_count.is_multiple_of(self.full_gc_frequency);
        if periodic_full_gc || self.should_collect_old_before_minor() {
            self.gc_count = 0;
            self.full_gc(gc_frame_tops, extra_roots);
        } else {
            self.minor_gc(gc_frame_tops, extra_roots);
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

    fn can_allocate(&self, words: usize, _kind: BuiltInTypes) -> bool {
        if self.too_large_for_young(words) {
            // Conservative: large objects go to old gen, can't cheaply check free list
            return false;
        }
        self.young.can_allocate(Word::from_word(words))
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
    fn gather_stack_root_refs(&self, gc_frame_tops: &[usize]) -> (Vec<RootRef>, Vec<usize>) {
        let mut slots = Vec::new();
        let mut old_gen_values = Vec::new();

        for &gc_frame_top in gc_frame_tops {
            let (young_roots, old_roots) = self.gather_stack_roots_inner(gc_frame_top);

            for (slot_addr, _value) in young_roots {
                slots.push(RootRef(slot_addr as *mut usize));
            }

            old_gen_values.extend(old_roots);
        }

        (slots, old_gen_values)
    }

    /// Inner function to gather roots from a single thread's GC frame chain.
    /// Uses StackWalker and classifies roots into young gen (to copy) and old gen (to scan fields).
    fn gather_stack_roots_inner(&self, gc_frame_top: usize) -> (Vec<(usize, usize)>, Vec<usize>) {
        let mut roots: Vec<(usize, usize)> = Vec::with_capacity(36);
        let mut old_gen_objects: Vec<usize> = Vec::with_capacity(16);

        StackWalker::walk_stack_roots(gc_frame_top, |slot_addr, slot_value| {
            let untagged = BuiltInTypes::untag(slot_value);
            if untagged == 0 {
                return;
            }
            if self.young.contains(untagged as *const u8) {
                assert!(
                    self.young.contains_allocated(untagged as *const u8),
                    "Stale young gen pointer {:#x} (offset={}) found on stack at {:#x}, \
                         young alloc_offset={}",
                    slot_value,
                    untagged - self.young.base_address(),
                    slot_addr,
                    self.young.allocation_offset(),
                );
                roots.push((slot_addr, slot_value));
            } else if self.old.contains(untagged as *const u8) {
                old_gen_objects.push(slot_value);
            }
        });

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

            // copy() handles non-heap, misaligned, and non-young-gen checks
            let new_value = self.copy(old_value);

            if new_value != old_value {
                self.update_root(&root_ref, new_value);
            }
        }
    }

    fn minor_gc(&mut self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]) {
        let start = std::time::Instant::now();
        usdt_probes::fire_gc_minor_start(self.gc_count);

        let (mut stack_roots, mut stack_old_gen) = self.gather_stack_root_refs(gc_frame_tops);

        // Classify extra_roots (shadow stack handles) into young/old gen
        for &(slot_addr, value) in extra_roots {
            let untagged = BuiltInTypes::untag(value);
            if untagged == 0 {
                continue;
            }
            if self.young.contains(untagged as *const u8) {
                stack_roots.push(RootRef(slot_addr));
            } else {
                stack_old_gen.push(value);
            }
        }

        self.process_all_roots(stack_roots);

        // Process old gen objects found on stack - update their young gen references.
        // This scans one level deep for old gen objects directly referenced from stack.
        for old_root in stack_old_gen {
            self.process_old_gen_object(old_root);
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
            self.process_old_gen_object(old_object);
        }

        // Process dirty cards - cards marked by generated code write barriers.
        // We need to scan all objects in dirty cards for young gen references.
        self.process_dirty_cards();

        self.copy_remaining();

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
    fn process_dirty_cards(&mut self) {
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
            self.process_old_gen_object(old_object);
        }
    }

    /// Process an old gen object's fields, copying any young gen references to old gen.
    fn process_old_gen_object(&mut self, old_object: usize) {
        let mut heap_obj = HeapObject::from_tagged(old_object);
        let container_type_id = heap_obj.get_type_id();
        let container_untagged = heap_obj.untagged();

        // Process this old gen object's fields
        {
            let data = heap_obj.get_fields_mut();
            #[cfg(feature = "debug-gc")]
            eprintln!(
                "[GC DEBUG] process_old_gen_object {:#x}: {} fields",
                old_object,
                data.len()
            );
            #[allow(unused_variables)]
            for (i, field) in data.iter_mut().enumerate() {
                if let Some(field_obj) = HeapObject::try_from_tagged(*field) {
                    let field_ptr = field_obj.get_pointer();

                    if self.young.contains(field_ptr) {
                        assert!(
                            self.young.contains_allocated(field_ptr),
                            "Stale young gen pointer {:#x} found in old object type {} at {:#x} field {} (young alloc_offset={})",
                            *field,
                            container_type_id,
                            container_untagged,
                            i,
                            self.young.allocation_offset(),
                        );
                        if std::env::var("BEAGLE_DEBUG_PROMOTION").is_ok() {
                            let pointer = field_obj.untagged() as *const usize;
                            let raw_header = unsafe { *pointer };
                            let header = Header::from_usize(raw_header);
                            let large_size_words = if header.large {
                                Some(unsafe { *pointer.add(1) })
                            } else {
                                None
                            };
                            if large_size_words.is_some_and(|words| words > (1024 * 1024)) {
                                eprintln!(
                                    "[promotion-old-field] container_type={} container={:#x} field_index={} field={:#x} raw_header={:#x} type_id={} inline_size={} large_size_words={:?} opaque={}",
                                    container_type_id,
                                    container_untagged,
                                    i,
                                    *field,
                                    raw_header,
                                    header.type_id,
                                    header.size,
                                    large_size_words,
                                    header.opaque
                                );
                            }
                        }
                        #[cfg(feature = "debug-gc")]
                        eprintln!(
                            "[GC DEBUG]   field[{}] = {:#x} is in young gen, copying",
                            i, *field
                        );
                        *field = self.copy(*field);
                        #[cfg(feature = "debug-gc")]
                        eprintln!("[GC DEBUG]   -> new value: {:#x}", *field);
                    }
                }
            }
        }

        let segment_slots = self.collect_continuation_segment_slots(&heap_obj);
        for (slot_addr, slot_value) in segment_slots {
            let untagged = BuiltInTypes::untag(slot_value);
            if untagged != 0 && self.young.contains(untagged as *const u8) {
                assert!(
                    self.young.contains_allocated(untagged as *const u8),
                    "Stale young gen pointer {:#x} found in continuation segment for object type {} at {:#x} slot {:#x} (young alloc_offset={})",
                    slot_value,
                    container_type_id,
                    container_untagged,
                    slot_addr,
                    self.young.allocation_offset(),
                );
                if std::env::var("BEAGLE_DEBUG_PROMOTION").is_ok() {
                    let pointer = untagged as *const usize;
                    let raw_header = unsafe { *pointer };
                    let header = Header::from_usize(raw_header);
                    let large_size_words = if header.large {
                        Some(unsafe { *pointer.add(1) })
                    } else {
                        None
                    };
                    if large_size_words.is_some_and(|words| words > (1024 * 1024)) {
                        eprintln!(
                            "[promotion-old-segment] container_type={} container={:#x} slot_addr={:#x} slot_value={:#x} raw_header={:#x} type_id={} inline_size={} large_size_words={:?} opaque={}",
                            container_type_id,
                            container_untagged,
                            slot_addr,
                            slot_value,
                            raw_header,
                            header.type_id,
                            header.size,
                            large_size_words,
                            header.opaque
                        );
                    }
                }
                unsafe {
                    *(slot_addr as *mut usize) = self.copy(slot_value);
                }
            }
        }
    }

    // ==================== COPY LOGIC ====================

    /// Copy a young gen object to old gen, returning the new tagged pointer.
    fn copy(&mut self, root: usize) -> usize {
        if !BuiltInTypes::is_heap_pointer(root) {
            return root;
        }
        let untagged = BuiltInTypes::untag(root);
        if !untagged.is_multiple_of(8) || untagged == 0 {
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
            let result = Header::clear_forwarding_bit(header_data);
            // Preserve the original root's tag - different references to the same
            // object may use different tags (e.g., Closure vs HeapObject)
            let result = (result & !7) | (root & 7);
            return result;
        }

        if std::env::var("BEAGLE_DEBUG_PROMOTION").is_ok() {
            let header = Header::from_usize(header_data);
            let large_size_words = if header.large {
                Some(unsafe { *pointer.add(1) })
            } else {
                None
            };
            if large_size_words.is_some_and(|words| words > (1024 * 1024)) {
                eprintln!(
                    "[promotion-raw] root={:#x} tag={:?} ptr={:#x} raw_header={:#x} type_id={} inline_size={} large_size_words={:?} opaque={}",
                    root,
                    tag,
                    heap_object.untagged(),
                    header_data,
                    header.type_id,
                    header.size,
                    large_size_words,
                    header.opaque
                );
            }
        }

        // Check if this user struct needs migration to a new shape
        // Closures also have type_id=0 but must NOT be migrated as structs
        let new_pointer = if heap_object.get_type_id() == 0 && !heap_object.is_opaque_object() {
            let is_closure = heap_object.get_object_type() == Some(BuiltInTypes::Closure);
            if !is_closure {
                let struct_id = heap_object.get_struct_id();
                let layout_version = heap_object.get_layout_version();
                let runtime = crate::get_runtime().get_mut();
                if runtime.structs.has_pending_migrations() {
                    if let Some(plan) = runtime
                        .structs
                        .migration_plan_for(struct_id, layout_version)
                    {
                        self.copy_with_migration(&heap_object, plan)
                    } else {
                        // Normal copy
                        let data = heap_object.get_full_object_data();
                        self.old.copy_data_to_offset(data)
                    }
                } else {
                    // Normal copy
                    let data = heap_object.get_full_object_data();
                    self.old.copy_data_to_offset(data)
                }
            } else {
                // Normal copy for closures
                let data = heap_object.get_full_object_data();
                self.old.copy_data_to_offset(data)
            }
        } else {
            // Normal copy
            let data = heap_object.get_full_object_data();
            self.old.copy_data_to_offset(data)
        };

        // Get the new object and add to processing queue
        let new_object = HeapObject::from_untagged(new_pointer);
        self.copied.push(new_object);

        // Store forwarding pointer in header for all objects (works even for 0-field objects)
        let tagged_new = tag.tag(new_pointer as isize) as usize;
        let forwarding = Header::set_forwarding_bit(tagged_new);
        unsafe { *pointer = forwarding };

        tagged_new
    }

    fn copy_with_migration(
        &mut self,
        old_object: &HeapObject,
        plan: &crate::runtime::MigrationPlan,
    ) -> *const u8 {
        // Build the new object data as a byte buffer and use copy_data_to_offset
        // struct_id stays the same (stable ID), only layout version changes
        let old_header = old_object.get_header();
        let new_header = Header {
            type_id: old_header.type_id,
            type_data: old_header.type_data, // struct_id unchanged
            size: plan.new_field_count as u16,
            opaque: old_header.opaque,
            marked: false,
            large: false,
            type_flags: plan.new_layout_version,
        };

        let total_words = 1 + plan.new_field_count; // header + fields
        let mut data = vec![0u8; total_words * 8];

        // Write header
        data[0..8].copy_from_slice(&new_header.to_usize().to_ne_bytes());

        // Write fields
        let null_val = BuiltInTypes::null_value() as usize;
        for (new_idx, mapping) in plan.field_map.iter().enumerate() {
            let value = match mapping {
                Some(old_idx) => old_object.get_field(*old_idx),
                None => null_val,
            };
            let offset = (1 + new_idx) * 8;
            data[offset..offset + 8].copy_from_slice(&value.to_ne_bytes());
        }

        self.old.copy_data_to_offset(&data)
    }

    fn copy_remaining(&mut self) {
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
            let container_type_id = object.get_type_id();
            let container_untagged = object.untagged();
            let initial_fields = object.get_fields().to_vec();
            let container_struct_info = if container_type_id == 0 {
                let struct_id = object.get_struct_id();
                let runtime = crate::get_runtime().get();
                let struct_name = runtime
                    .get_struct_by_id(struct_id)
                    .map(|s| s.name.as_str())
                    .unwrap_or("<unknown>");
                format!(" struct_id={} struct_name={}", struct_id, struct_name)
            } else {
                String::new()
            };
            for (field_index, field) in object.get_fields_mut().iter_mut().enumerate() {
                if let Some(heap_obj) = HeapObject::try_from_tagged(*field)
                    && self.young.contains(heap_obj.get_pointer())
                {
                    if !self.young.contains_allocated(heap_obj.get_pointer()) {
                        let fields = initial_fields
                            .iter()
                            .enumerate()
                            .map(|(i, value)| format!("{}:{:#x}", i, value))
                            .collect::<Vec<_>>()
                            .join(" ");
                        panic!(
                            "Stale young gen pointer {:#x} found while scanning copied object type {} at {:#x} field {} (field_count={}, young alloc_offset={}){} fields=[{}]",
                            *field,
                            container_type_id,
                            container_untagged,
                            field_index,
                            initial_fields.len(),
                            self.young.allocation_offset(),
                            container_struct_info,
                            fields,
                        );
                    }
                    if std::env::var("BEAGLE_DEBUG_PROMOTION").is_ok() {
                        let pointer = heap_obj.untagged() as *const usize;
                        let raw_header = unsafe { *pointer };
                        let header = Header::from_usize(raw_header);
                        let large_size_words = if header.large {
                            Some(unsafe { *pointer.add(1) })
                        } else {
                            None
                        };
                        if large_size_words.is_some_and(|words| words > (1024 * 1024)) {
                            eprintln!(
                                "[promotion-field] container_type={} container={:#x} field_index={} field={:#x} raw_header={:#x} type_id={} inline_size={} large_size_words={:?} opaque={}",
                                container_type_id,
                                container_untagged,
                                field_index,
                                *field,
                                raw_header,
                                header.type_id,
                                header.size,
                                large_size_words,
                                header.opaque
                            );
                        }
                    }
                    #[cfg(feature = "debug-gc")]
                    eprintln!("[GC DEBUG]   copying young-gen field {:#x}", *field);
                    *field = self.copy(*field);
                }
            }
            let segment_slots = self.collect_continuation_segment_slots(&object);
            for (slot_addr, slot_value) in segment_slots {
                let untagged = BuiltInTypes::untag(slot_value);
                if untagged != 0 && self.young.contains(untagged as *const u8) {
                    assert!(
                        self.young.contains_allocated(untagged as *const u8),
                        "Stale young gen pointer {:#x} found while scanning copied continuation segment type {} at {:#x} slot {:#x} (young alloc_offset={})",
                        slot_value,
                        container_type_id,
                        container_untagged,
                        slot_addr,
                        self.young.allocation_offset(),
                    );
                    if std::env::var("BEAGLE_DEBUG_PROMOTION").is_ok() {
                        let pointer = untagged as *const usize;
                        let raw_header = unsafe { *pointer };
                        let header = Header::from_usize(raw_header);
                        let large_size_words = if header.large {
                            Some(unsafe { *pointer.add(1) })
                        } else {
                            None
                        };
                        if large_size_words.is_some_and(|words| words > (1024 * 1024)) {
                            eprintln!(
                                "[promotion-segment-slot] container_type={} container={:#x} slot_addr={:#x} slot_value={:#x} raw_header={:#x} type_id={} inline_size={} large_size_words={:?} opaque={}",
                                container_type_id,
                                container_untagged,
                                slot_addr,
                                slot_value,
                                raw_header,
                                header.type_id,
                                header.size,
                                large_size_words,
                                header.opaque
                            );
                        }
                    }
                    unsafe {
                        *(slot_addr as *mut usize) = self.copy(slot_value);
                    }
                }
            }
        }
        #[cfg(feature = "debug-gc")]
        eprintln!(
            "[GC DEBUG] copy_remaining done after {} iterations",
            iterations
        );
    }

    fn full_gc(&mut self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]) {
        usdt_probes::fire_gc_full_start(self.gc_count);
        self.minor_gc(gc_frame_tops, extra_roots);
        self.old.gc(gc_frame_tops, extra_roots);
        if std::env::var("BEAGLE_DEBUG_CONT_LIVE").is_ok() {
            let mut continuation_count = 0usize;
            let mut continuation_segment_bytes = 0usize;
            self.old.walk_objects_mut(|_, heap_obj| {
                let object = HeapObject::from_untagged(heap_obj.untagged() as *const u8);
                if let Some(cont) = ContinuationObject::from_heap_object(object) {
                    continuation_count += 1;
                    continuation_segment_bytes += cont.segment_size();
                }
            });
            eprintln!(
                "[cont-live] gc_count={} continuations={} segment_bytes={} old_free={}",
                self.gc_count,
                continuation_count,
                continuation_segment_bytes,
                self.old.free_bytes()
            );
        }
        usdt_probes::fire_gc_full_end(self.gc_count);
    }
}

#[cfg(all(test, feature = "generational"))]
mod tests {
    use super::*;
    use crate::builtins::reset_shift::ContinuationObject;
    use crate::types::HeapObject;
    use std::sync::{Mutex, OnceLock};

    static RUNTIME_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn continuation_segment_ptrs(alloc: &mut GenerationalGC) -> Vec<usize> {
        let mut segment_ptrs = Vec::new();

        let mut offset = 0usize;
        while offset < alloc.young.allocation_offset() {
            let ptr = unsafe { alloc.young.start.add(offset) };
            let heap_obj = HeapObject::from_untagged(ptr);
            let full_size = heap_obj.full_size();
            let object = HeapObject::from_untagged(ptr);
            if !object.is_zero_size()
                && let Some(cont) = ContinuationObject::from_heap_object(object)
            {
                segment_ptrs.push(cont.segment_ptr());
            }
            offset += full_size;
            offset = (offset + 7) & !7;
        }

        alloc.old.walk_objects_mut(|_, heap_obj| {
            let object = HeapObject::from_untagged(heap_obj.untagged() as *const u8);
            if let Some(cont) = ContinuationObject::from_heap_object(object) {
                segment_ptrs.push(cont.segment_ptr());
            }
        });

        segment_ptrs
    }

    #[test]
    fn continuation_segments_are_heap_objects_and_gc_reclaimable() {
        let _guard = RUNTIME_TEST_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap();

        let probe_source = r#"
namespace continuation_heap_gc_probe

use beagle.core as core
use beagle.effect as effect

enum Yield {
    Value {}
}

let saved = atom(null)

struct YieldHandler {}

extend YieldHandler with effect/Handler(Yield) {
    fn handle(self, op, resume) {
        reset!(saved, resume)
        123
    }
}

fn capture-once() {
    let handler = YieldHandler {}
    handle effect/Handler(Yield) with handler {
        perform Yield.Value {}
    }
}

fn clear-saved() {
    reset!(saved, null)
    core/gc()
    core/gc()
}

fn main() {
    capture-once()
    println("ok")
}
"#;

        let probe_path = std::env::temp_dir().join(format!(
            "continuation_heap_gc_probe_{}.bg",
            std::process::id()
        ));
        std::fs::write(&probe_path, probe_source).expect("failed to write continuation probe");

        let args =
            crate::CommandLineArguments::for_run(probe_path.to_string_lossy().into_owned(), vec![]);
        crate::main_inner(args).expect("probe program should run");

        {
            let ptd = crate::runtime::per_thread_data();
            assert!(
                ptd.captured_segments.is_empty(),
                "legacy captured_segments map should stay unused"
            );
            assert!(
                ptd.pending_captured_segment_handles.is_empty(),
                "legacy pending captured handles should stay unused"
            );
            assert!(
                ptd.pending_heap_segments.is_empty(),
                "pending heap segments should be drained once the continuation object exists"
            );
        }

        let runtime = crate::get_runtime().get_mut();
        let live_segment_ptrs = runtime
            .heap_for_testing()
            .with_locked_alloc(continuation_segment_ptrs);
        assert!(
            !live_segment_ptrs.is_empty(),
            "expected a rooted continuation to remain reachable from the global atom"
        );
        for seg_ptr in live_segment_ptrs.iter().copied() {
            assert!(
                crate::types::BuiltInTypes::is_heap_pointer(seg_ptr),
                "captured continuation segment must be stored as a tagged heap pointer"
            );
            let seg_obj = HeapObject::from_tagged(seg_ptr);
            assert!(
                seg_obj.is_opaque_object(),
                "captured continuation segment should be an opaque heap object"
            );
        }

        let clear_fn = runtime
            .get_function0("continuation_heap_gc_probe/clear-saved")
            .expect("clear-saved function should exist");
        clear_fn();

        let remaining_segment_ptrs = runtime
            .heap_for_testing()
            .with_locked_alloc(continuation_segment_ptrs);
        assert!(
            remaining_segment_ptrs.is_empty(),
            "continuations should be reclaimed after clearing the last root and forcing GC"
        );

        let _ = std::fs::remove_file(probe_path);
    }
}
