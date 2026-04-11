use std::{ffi::c_void, io::Error, mem};

use libc::mprotect;

use super::get_page_size;

use super::{AllocateAction, Allocator, AllocatorOptions, stack_walker::StackWalker};
use crate::builtins::reset_shift::ContinuationObject;
use crate::types::{BuiltInTypes, Header, HeapObject, Word};

const DEFAULT_PAGE_COUNT: usize = 1024;
// Aribtary number that should be changed when I have
// better options for gc
const MAX_PAGE_COUNT: usize = 1000000;

struct Space {
    start: *const u8,
    page_count: usize,
    allocation_offset: usize,
    #[cfg(debug_assertions)]
    protected: bool,
}

unsafe impl Send for Space {}
unsafe impl Sync for Space {}

impl Space {
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

    /// Check if a pointer falls within the currently allocated (used) portion of this space.
    /// Unlike `contains()` which checks the entire reserved virtual memory region,
    /// this only checks up to `allocation_offset` — the portion that actually holds objects.
    fn contains_allocated(&self, pointer: *const u8) -> bool {
        let start = self.start as usize;
        let end = start + self.allocation_offset;
        let pointer = pointer as usize;
        pointer >= start && pointer < end
    }

    /// Ensure `size` bytes from the current offset fit in the committed region.
    /// Grows the committed space if needed.
    fn ensure_capacity(&mut self, size: usize) {
        while self.allocation_offset + size > self.byte_count() {
            self.double_committed_memory();
        }
    }

    fn copy_data_to_offset(&mut self, data: &[u8]) -> isize {
        // Grow to_space if the live set won't fit in the currently committed pages.
        self.ensure_capacity(data.len());

        unsafe {
            let start = self.start.add(self.allocation_offset);
            let new_pointer = start as isize;
            self.allocation_offset += data.len();
            if !self.allocation_offset.is_multiple_of(8) {
                panic!("Heap offset is not aligned");
            }
            std::ptr::copy_nonoverlapping(data.as_ptr(), start as *mut u8, data.len());
            // Objects in to-space must start unmarked. Mark bits belong to the
            // from-space traversal state and must not survive the copy.
            let header_ptr = start as *mut usize;
            *header_ptr = Header::clear_marked_bit(*header_ptr);
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

    fn allocate(&mut self, words: usize) -> *const u8 {
        let offset = self.allocation_offset;
        let size = Word::from_word(words);
        // Large objects need 16-byte header, small objects need 8-byte header
        let header_size = if words > Header::MAX_INLINE_SIZE {
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

    fn object_iter_from_position(&self, offset: usize) -> impl Iterator<Item = HeapObject> + use<> {
        ObjectIterator {
            space: self,
            offset,
        }
    }

    #[cfg(debug_assertions)]
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

    #[cfg(debug_assertions)]
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
            #[cfg(debug_assertions)]
            protected: false,
        }
    }

    fn commit_memory(addr: *mut c_void, size: usize) -> Result<(), Error> {
        unsafe {
            if mprotect(addr, size, libc::PROT_READ | libc::PROT_WRITE) != 0 {
                Err(Error::last_os_error())
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
}

struct ObjectIterator {
    space: *const Space,
    offset: usize,
}

impl Iterator for ObjectIterator {
    type Item = HeapObject;

    fn next(&mut self) -> Option<Self::Item> {
        let space = unsafe { &*self.space };

        if self.offset >= space.allocation_offset {
            return None;
        }

        if space.allocation_offset == 0 {
            return None;
        }

        let pointer = unsafe { space.start.add(self.offset) };
        let object = HeapObject::from_untagged(pointer);
        let size = object.full_size();

        self.offset += size;
        if !self.offset.is_multiple_of(8) {
            panic!("Heap offset is not aligned");
        }
        Some(object)
    }
}

pub struct CompactingHeap {
    to_space: Space,
    from_space: Space,
    options: AllocatorOptions,
}
impl CompactingHeap {
    fn is_plausible_from_space_object(&self, tagged_ptr: usize) -> bool {
        let untagged = BuiltInTypes::untag(tagged_ptr);
        if !untagged.is_multiple_of(8) || !self.from_space.contains_allocated(untagged as *const u8)
        {
            return false;
        }

        let start = self.from_space.start as usize;
        let allocated_end = start + self.from_space.allocation_offset;
        let remaining_bytes = allocated_end.saturating_sub(untagged);
        if remaining_bytes < 8 {
            return false;
        }

        let header_data = unsafe { *(untagged as *const usize) };
        if Header::is_forwarding_bit_set(header_data) {
            return true;
        }

        let header = Header::from_usize(header_data);
        let (header_size, field_words) = if header.large {
            if remaining_bytes < 16 {
                return false;
            }
            let field_words = unsafe { *((untagged as *const usize).add(1)) };
            (16usize, field_words)
        } else {
            (8usize, header.size as usize)
        };

        let Some(fields_size) = field_words.checked_mul(8) else {
            return false;
        };
        let Some(full_size) = header_size.checked_add(fields_size) else {
            return false;
        };

        full_size <= remaining_bytes && full_size.is_multiple_of(8)
    }

    fn scan_continuation_segment<F>(&mut self, object: &HeapObject, mut callback: F)
    where
        F: FnMut(usize, usize),
    {
        let object = HeapObject::from_untagged(object.untagged() as *const u8);
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
                |slot_addr, slot_value| callback(slot_addr, slot_value),
            );
        }
    }

    fn collect_continuation_segment_slots(&mut self, object: &HeapObject) -> Vec<(usize, usize)> {
        let mut slots = Vec::new();
        self.scan_continuation_segment(object, |slot_addr, slot_value| {
            slots.push((slot_addr, slot_value));
        });
        slots
    }

    fn copy_using_cheneys_algorithm(&mut self, tagged_ptr: usize) -> usize {
        let untagged = BuiltInTypes::untag(tagged_ptr);

        // Skip misaligned pointers - these are values (like floats) whose bit
        // pattern happens to have the heap pointer tag. They aren't real objects.
        if !untagged.is_multiple_of(8) {
            return tagged_ptr;
        }

        // If the pointer is not in either space's allocated region, it may be an
        // embedded value (e.g., inline float) or a pointer to runtime-allocated memory
        // outside the GC heap. Return it unchanged — it doesn't need to be moved.
        // We use contains_allocated() for from_space to avoid false positives: values
        // (like callee-saved register spills) whose bit pattern happens to fall within
        // the reserved-but-unallocated portion of from_space's virtual memory.
        if !self.to_space.contains(untagged as *const u8)
            && !self.from_space.contains_allocated(untagged as *const u8)
        {
            return tagged_ptr;
        }

        // If already in to_space, it's been copied
        if self.to_space.contains(untagged as *const u8) {
            return tagged_ptr;
        }

        // Stack/continuation scanning is conservative enough that stale words can
        // occasionally look like heap pointers. Only copy values that still decode
        // to an object fully contained in allocated from-space.
        if !self.is_plausible_from_space_object(tagged_ptr) {
            return tagged_ptr;
        }

        // Now we know it's a valid heap pointer in from_space — safe to construct HeapObject
        let heap_object = HeapObject::from_tagged(tagged_ptr);

        // If forwarded, object has been moved - get forwarding pointer from header
        // We check the forwarding bit (bit 3) which doesn't conflict with type tags
        let untagged = heap_object.untagged();
        if untagged == 0 {
            // Null heap pointer (tagged 6 with address 0) — skip
            return tagged_ptr;
        }
        let pointer = untagged as *mut usize;
        let header_data = unsafe { *pointer };

        if Header::is_forwarding_bit_set(header_data) {
            // The header contains the forwarding pointer with forwarding bit set
            // Clear the forwarding bit to get the clean tagged pointer
            let result = Header::clear_forwarding_bit(header_data);
            // Preserve the original root's tag - different references to the same
            // object may use different tags (e.g., Closure vs HeapObject)
            return (result & !7) | (tagged_ptr & 7);
        }

        // Check if this user struct needs migration to a new layout
        // IMPORTANT: Skip closures — they have type_id=0 but are NOT user structs.
        // Closures are tagged as Closure (tag 5), while user structs are HeapObject (tag 6).
        if heap_object.get_type_id() == 0 && !heap_object.is_opaque_object() {
            let object_type = heap_object.get_object_type();
            let is_closure = object_type == Some(BuiltInTypes::Closure);
            if !is_closure {
                let struct_id = heap_object.get_struct_id();
                let layout_version = heap_object.get_layout_version();
                let runtime = crate::get_runtime().get_mut();
                if runtime.structs.has_pending_migrations()
                    && let Some(plan) = runtime
                        .structs
                        .migration_plan_for(struct_id, layout_version)
                {
                    let new_pointer = self.copy_with_migration(&heap_object, plan);
                    debug_assert!(new_pointer % 8 == 0, "Pointer is not aligned");
                    let tagged_new = object_type.unwrap().tag(new_pointer) as usize;
                    unsafe { *pointer = Header::set_forwarding_bit(tagged_new) };
                    return tagged_new;
                }
            }
        }

        // Copy the object to to_space
        let data = heap_object.get_full_object_data();
        let new_pointer = self.to_space.copy_data_to_offset(data);
        debug_assert!(new_pointer % 8 == 0, "Pointer is not aligned");

        // Store forwarding pointer in header for all objects
        let tagged_new = heap_object.get_object_type().unwrap().tag(new_pointer) as usize;
        // Set the forwarding bit to mark this as a forwarding pointer
        unsafe { *pointer = Header::set_forwarding_bit(tagged_new) };

        tagged_new
    }

    fn copy_with_migration(
        &mut self,
        old_object: &HeapObject,
        plan: &crate::runtime::MigrationPlan,
    ) -> isize {
        // Build new header with updated layout version and field count
        // struct_id stays the same (stable ID)
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

        // Allocate space: header (8 bytes) + fields
        let total_bytes = 8 + plan.new_field_count * 8;
        self.to_space.ensure_capacity(total_bytes);
        let new_pointer =
            unsafe { self.to_space.start.add(self.to_space.allocation_offset) } as isize;

        // Write header
        unsafe {
            let header_ptr = new_pointer as *mut usize;
            *header_ptr = new_header.to_usize();
        }

        // Write fields: map old fields to new layout, null for missing fields
        let null_val = BuiltInTypes::null_value() as usize;
        for (new_idx, mapping) in plan.field_map.iter().enumerate() {
            let value = match mapping {
                Some(old_idx) => old_object.get_field(*old_idx),
                None => null_val,
            };
            unsafe {
                let field_ptr = (new_pointer as *mut usize).add(1 + new_idx);
                *field_ptr = value;
            }
        }

        self.to_space.allocation_offset += total_bytes;
        debug_assert!(
            self.to_space.allocation_offset.is_multiple_of(8),
            "Heap offset is not aligned"
        );
        new_pointer
    }

    unsafe fn copy_all(&mut self, roots: Vec<usize>) -> Vec<usize> {
        unsafe {
            let start_offset = self.to_space.allocation_offset;
            let mut new_roots = vec![];
            for root in roots.iter() {
                new_roots.push(self.copy_using_cheneys_algorithm(*root));
            }

            self.copy_remaining(start_offset);

            new_roots
        }
    }

    unsafe fn copy_remaining(&mut self, start_offset: usize) {
        for mut object in self.to_space.object_iter_from_position(start_offset) {
            if object.marked() {
                panic!("We are copying to this space, nothing should be marked");
            }
            if object.is_zero_size() {
                continue;
            }
            for datum in object.get_fields_mut().iter_mut() {
                if BuiltInTypes::is_heap_pointer(*datum) {
                    *datum = self.copy_using_cheneys_algorithm(*datum);
                }
            }
            let segment_slots = self.collect_continuation_segment_slots(&object);
            for (slot_addr, slot_value) in segment_slots {
                let new_value = self.copy_using_cheneys_algorithm(slot_value);
                unsafe {
                    *(slot_addr as *mut usize) = new_value;
                }
            }
        }
    }

    fn gc_continuations(&mut self) {
        let runtime = crate::get_runtime().get_mut();

        // GC runs while all threads are paused at safepoints, so it's safe to
        // iterate the registry and dereference each thread's per-thread data.
        let registry = runtime.per_thread_registry.lock().unwrap();
        for ptd_ptr in registry.iter() {
            let ptd = unsafe { &mut *ptd_ptr.0 };

            for saved_ptr in ptd.saved_continuation_ptrs.iter_mut() {
                if *saved_ptr != 0 && BuiltInTypes::is_heap_pointer(*saved_ptr) {
                    *saved_ptr = self.copy_using_cheneys_algorithm(*saved_ptr);
                }
            }
        }
    }
}

impl Allocator for CompactingHeap {
    fn new(options: AllocatorOptions) -> Self {
        let to_space = Space::new(DEFAULT_PAGE_COUNT / 2);
        let from_space = Space::new(DEFAULT_PAGE_COUNT / 2);

        Self {
            to_space,
            from_space,
            options,
        }
    }

    fn try_allocate(
        &mut self,
        words: usize,
        kind: crate::types::BuiltInTypes,
    ) -> Result<super::AllocateAction, Box<dyn std::error::Error>> {
        if words > self.from_space.word_count() {
            // TODO: Grow should take an allocation size
            self.grow();
        }

        if self.from_space.allocation_offset + words * 8 >= self.from_space.byte_count() {
            return Ok(AllocateAction::Gc);
        }

        // TODO: Actually allocate
        let pointer = self.from_space.allocate(words);

        // Float objects are opaque (their field is a raw f64, not a pointer).
        // Set the opaque bit immediately so GC never sees a non-opaque float.
        if kind == crate::types::BuiltInTypes::Float {
            unsafe {
                *(pointer as *mut usize) |= 0x2; // Set opaque bit (bit 1)
            }
        }

        Ok(AllocateAction::Allocated(pointer))
    }

    fn gc(&mut self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]) {
        if !self.options.gc {
            return;
        }
        let had_pending_migrations = crate::get_runtime().get().structs.has_pending_migrations();

        for &gc_frame_top in gc_frame_tops.iter() {
            let roots = StackWalker::collect_stack_roots(gc_frame_top);
            let new_roots = unsafe { self.copy_all(roots.iter().map(|x| x.1).collect()) };

            for (i, (slot_addr, _)) in roots.iter().enumerate() {
                debug_assert!(
                    BuiltInTypes::untag(new_roots[i]).is_multiple_of(8),
                    "Pointer is not aligned"
                );
                unsafe {
                    *(*slot_addr as *mut usize) = new_roots[i];
                }
            }
        }

        // Process extra roots from shadow stacks
        if !extra_roots.is_empty() {
            let values: Vec<usize> = extra_roots.iter().map(|&(_, v)| v).collect();
            let new_values = unsafe { self.copy_all(values) };
            for (i, &(slot_addr, _)) in extra_roots.iter().enumerate() {
                unsafe {
                    *slot_addr = new_values[i];
                }
            }
        }

        // Capture offset BEFORE continuation processing so that copy_remaining
        // will transitively process any objects newly copied by gc_continuations.
        let start_offset = self.to_space.allocation_offset;

        // Process suspended caller-frame roots
        self.gc_continuations();

        unsafe { self.copy_remaining(start_offset) };

        if had_pending_migrations {
            crate::get_runtime()
                .get_mut()
                .structs
                .complete_pending_migrations();
        }

        mem::swap(&mut self.from_space, &mut self.to_space);

        self.to_space.clear();
    }

    fn grow(&mut self) {
        // From space is never protected
        self.from_space.double_committed_memory();

        #[cfg(debug_assertions)]
        {
            let currently_protect = self.to_space.protected;
            if currently_protect {
                self.to_space.unprotect();
            }
            self.to_space.double_committed_memory();
            if currently_protect {
                self.to_space.protect();
            }
        }

        #[cfg(not(debug_assertions))]
        {
            self.to_space.double_committed_memory();
        }
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }
}
