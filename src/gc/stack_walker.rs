use crate::ir::CONTINUATION_MARKER_PADDING_SIZE;
use crate::types::BuiltInTypes;

use super::{STACK_SIZE, StackMap};

/// A simple abstraction for walking the stack and finding heap pointers
pub struct StackWalker;

impl StackWalker {
    /// Get the live portion of the stack as a slice
    pub fn get_live_stack(stack_base: usize, stack_pointer: usize) -> &'static [usize] {
        let stack_end = stack_base;
        let distance_till_end = stack_end - stack_pointer;
        let num_64_till_end = (distance_till_end / 8) + 1;
        let stack_begin = stack_end - STACK_SIZE;

        unsafe {
            let stack = std::slice::from_raw_parts(stack_begin as *const usize, STACK_SIZE / 8);
            &stack[stack.len() - num_64_till_end..]
        }
    }

    /// Walk the stack and call a callback for each heap pointer found
    /// The callback receives (stack_offset, heap_pointer_value)
    pub fn walk_stack_roots<F>(
        stack_base: usize,
        stack_pointer: usize,
        stack_map: &StackMap,
        mut callback: F,
    ) where
        F: FnMut(usize, usize),
    {
        let stack = Self::get_live_stack(stack_base, stack_pointer);

        let mut i = 0;
        while i < stack.len() {
            let value = stack[i];

            if let Some(details) = stack_map.find_stack_data(value) {
                let mut frame_size = details.max_stack_size + details.number_of_locals;
                if frame_size % 2 != 0 {
                    frame_size += 1;
                }

                // Account for continuation padding that's physically allocated in prelude
                // The padding is allocated but not added to local calculations anymore
                let bottom_of_frame =
                    i + frame_size + 1 + CONTINUATION_MARKER_PADDING_SIZE as usize;
                // Active frame size includes locals and current stack, but padding is already
                // accounted for in the bottom_of_frame calculation
                let active_frame = details.current_stack_size
                    + details.number_of_locals;

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
                        callback(j, *slot);
                    }
                }
                continue;
            }
            i += 1;
        }
    }

    /// Collect all heap pointers from the stack into a vector
    /// Returns (stack_offset, heap_pointer_value) pairs
    pub fn collect_stack_roots(
        stack_base: usize,
        stack_pointer: usize,
        stack_map: &StackMap,
    ) -> Vec<(usize, usize)> {
        let mut roots = Vec::with_capacity(32);
        Self::walk_stack_roots(stack_base, stack_pointer, stack_map, |offset, pointer| {
            roots.push((offset, pointer));
        });
        roots
    }

    /// Get a mutable slice of the live stack for updating pointers after GC
    pub fn get_live_stack_mut(stack_base: usize, stack_pointer: usize) -> &'static mut [usize] {
        let stack_end = stack_base;
        let distance_till_end = stack_end - stack_pointer;
        let num_64_till_end = (distance_till_end / 8) + 1;
        let stack_begin = stack_end - STACK_SIZE;
        let len = STACK_SIZE / 8;

        unsafe {
            let start_ptr = (stack_begin as *mut usize).add(len - num_64_till_end);
            std::slice::from_raw_parts_mut(start_ptr, num_64_till_end)
        }
    }
}
