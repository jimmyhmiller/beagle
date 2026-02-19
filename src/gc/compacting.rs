use std::{ffi::c_void, io::Error, mem};

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
    fn copy_using_cheneys_algorithm(&mut self, tagged_ptr: usize) -> usize {
        let untagged = BuiltInTypes::untag(tagged_ptr);

        // Skip misaligned pointers - these are values (like floats) whose bit
        // pattern happens to have the heap pointer tag. They aren't real objects.
        if !untagged.is_multiple_of(8) {
            return tagged_ptr;
        }

        // If the pointer is not in either space, it may be an embedded value
        // (e.g., inline float) or a pointer to runtime-allocated memory outside the GC heap.
        // Return it unchanged — it doesn't need to be moved.
        if !self.to_space.contains(untagged as *const u8)
            && !self.from_space.contains(untagged as *const u8)
        {
            return tagged_ptr;
        }

        // If already in to_space, it's been copied
        if self.to_space.contains(untagged as *const u8) {
            return tagged_ptr;
        }

        // Now we know it's a valid heap pointer in from_space — safe to construct HeapObject
        let heap_object = HeapObject::from_tagged(tagged_ptr);

        // If forwarded, object has been moved - get forwarding pointer from header
        // We check the forwarding bit (bit 3) which doesn't conflict with type tags
        let untagged = heap_object.untagged();
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

    unsafe fn copy_all(&mut self, roots: Vec<usize>, stack_map: &StackMap) -> Vec<usize> {
        unsafe {
            let start_offset = self.to_space.allocation_offset;
            // TODO: Is this vec the best way? Probably not
            // I could hand this the pointers to the stack location
            // then resolve what they point to and update them?
            // I should think about how to get rid of this allocation at the very least.
            let mut new_roots = vec![];
            for root in roots.iter() {
                new_roots.push(self.copy_using_cheneys_algorithm(*root));
            }

            self.copy_remaining(start_offset, stack_map);

            new_roots
        }
    }

    unsafe fn copy_remaining(&mut self, start_offset: usize, stack_map: &StackMap) {
        for mut object in self.to_space.object_iter_from_position(start_offset) {
            if object.marked() {
                panic!("We are copying to this space, nothing should be marked");
            }
            if object.is_zero_size() {
                continue;
            }
            for datum in object.get_fields_mut() {
                if BuiltInTypes::is_heap_pointer(*datum) {
                    *datum = self.copy_using_cheneys_algorithm(*datum);
                }
            }
            if object.get_type_id() == TYPE_ID_CONTINUATION as usize
                && let Some(cont) = ContinuationObject::from_heap_object(object)
            {
                cont.with_segment_bytes_mut(|segment| {
                    if segment.is_empty() {
                        return;
                    }
                    self.gc_continuation_segment(
                        segment,
                        cont.original_sp(),
                        cont.original_fp(),
                        cont.prompt_stack_pointer(),
                        cont.resume_address(),
                        stack_map,
                    );
                });
            }
        }
    }

    fn gc_continuations(&mut self, stack_map: &StackMap) {
        let runtime = crate::get_runtime().get_mut();

        // Fast path: skip if no invocation return points
        if runtime.invocation_return_points.is_empty() {
            return;
        }

        // Process InvocationReturnPoint saved frames.
        // These are single frames, not frame chains — use the return address for stack map lookup.
        for (_thread_id, rps) in runtime.invocation_return_points.iter_mut() {
            for rp in rps.iter_mut() {
                if rp.saved_stack_frame.is_empty() {
                    continue;
                }
                // Collect roots from the saved frame
                let mut roots = Vec::new();
                ContinuationSegmentWalker::walk_saved_frame_roots(
                    &rp.saved_stack_frame,
                    rp.stack_pointer,
                    rp.frame_pointer,
                    rp.return_address,
                    stack_map,
                    |offset, tagged_value| {
                        roots.push((offset, tagged_value));
                    },
                );
                // Copy objects and update pointers
                let new_values: Vec<usize> = roots
                    .iter()
                    .map(|(_offset, tagged_value)| self.copy_using_cheneys_algorithm(*tagged_value))
                    .collect();
                for (i, (offset, _)) in roots.iter().enumerate() {
                    unsafe {
                        let ptr = rp.saved_stack_frame.as_mut_ptr().add(*offset) as *mut usize;
                        *ptr = new_values[i];
                    }
                }
            }
        }
    }

    /// Update saved_continuation_ptr values after copying.
    /// These are continuation objects saved during invoke_continuation_runtime.
    fn gc_saved_continuation_ptrs(&mut self) {
        let runtime = crate::get_runtime().get_mut();

        for (_thread_id, cont_ptr) in runtime.saved_continuation_ptr.iter_mut() {
            if *cont_ptr == 0 {
                continue;
            }
            if !BuiltInTypes::is_heap_pointer(*cont_ptr) {
                continue;
            }
            // Copy the object and update the pointer
            let new_ptr = self.copy_using_cheneys_algorithm(*cont_ptr);
            *cont_ptr = new_ptr;
        }
    }

    fn gc_continuation_segment(
        &mut self,
        segment: &mut [u8],
        original_sp: usize,
        original_fp: usize,
        prompt_sp: usize,
        resume_address: usize,
        stack_map: &StackMap,
    ) {
        // Collect heap pointers and their offsets
        let mut roots = Vec::new();
        ContinuationSegmentWalker::walk_segment_roots(
            segment,
            original_sp,
            original_fp,
            prompt_sp,
            resume_address,
            stack_map,
            |offset, tagged_value| {
                roots.push((offset, tagged_value));
            },
        );

        // Copy objects to to_space using Cheney's algorithm
        let new_values: Vec<usize> = roots
            .iter()
            .map(|(_offset, tagged_value)| self.copy_using_cheneys_algorithm(*tagged_value))
            .collect();

        // Update pointers in segment
        for (i, (offset, _)) in roots.iter().enumerate() {
            unsafe {
                let ptr = segment.as_mut_ptr().add(*offset) as *mut usize;
                *ptr = new_values[i];
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

    fn gc(
        &mut self,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize, usize)],
        extra_roots: &[(*mut usize, usize)],
    ) {
        if !self.options.gc {
            return;
        }

        for (stack_base, frame_pointer, gc_return_addr) in stack_pointers.iter() {
            let roots = StackWalker::collect_stack_roots_with_return_addr(
                *stack_base,
                *frame_pointer,
                *gc_return_addr,
                stack_map,
            );
            let new_roots =
                unsafe { self.copy_all(roots.iter().map(|x| x.1).collect(), stack_map) };

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
            let new_values = unsafe { self.copy_all(values, stack_map) };
            for (i, &(slot_addr, _)) in extra_roots.iter().enumerate() {
                unsafe {
                    *slot_addr = new_values[i];
                }
            }
        }

        // Process continuation segments
        self.gc_continuations(stack_map);

        // Process saved_continuation_ptr values (continuation objects saved in Rust runtime)
        self.gc_saved_continuation_ptrs();

        let start_offset = self.to_space.allocation_offset;

        unsafe { self.copy_remaining(start_offset, stack_map) };

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
