use crate::types::BuiltInTypes;

use super::StackMap;

/// A simple abstraction for walking the stack and finding heap pointers
/// Uses frame pointer chain traversal for accurate stack walking.
pub struct StackWalker;

impl StackWalker {
    /// Get the live portion of the stack as a slice (used for pointer updates in compacting GC)
    #[allow(dead_code)]
    pub fn get_live_stack(stack_base: usize, frame_pointer: usize) -> &'static [usize] {
        let distance_till_end = stack_base - frame_pointer;
        let num_words = (distance_till_end / 8) + 1;

        unsafe { std::slice::from_raw_parts(frame_pointer as *const usize, num_words) }
    }

    /// Collect all heap pointers from the stack.
    /// Walks the frame pointer chain and scans each Beagle frame.
    pub fn collect_stack_roots_with_return_addr(
        stack_base: usize,
        frame_pointer: usize,
        gc_return_addr: usize,
        stack_map: &StackMap,
    ) -> Vec<(usize, usize)> {
        let mut roots = Vec::with_capacity(32);
        Self::walk_stack_roots_with_return_addr(
            stack_base,
            frame_pointer,
            gc_return_addr,
            stack_map,
            |addr, pointer| {
                roots.push((addr, pointer));
            },
        );
        roots
    }

    /// Walk the stack using the frame pointer chain.
    ///
    /// For the first frame, we use gc_return_addr (the return address from the builtin call)
    /// to look up stack map info. For subsequent frames, we use [FP+8] which gives us
    /// the return address for that frame's caller.
    ///
    /// Key insight: [FP+8] describes where THIS frame returns TO (i.e., a location in the caller).
    /// So for frame N, [FP+8] gives us the safepoint in frame N-1 where frame N was called.
    /// We use this to scan frame N-1's slots.
    pub fn walk_stack_roots_with_return_addr<F>(
        stack_base: usize,
        frame_pointer: usize,
        gc_return_addr: usize,
        stack_map: &StackMap,
        mut callback: F,
    ) where
        F: FnMut(usize, usize),
    {
        let mut fp = frame_pointer;
        // For the first frame, use gc_return_addr (the return address from the gc/builtin call)
        // For subsequent frames, use the return address from the previous frame
        let mut pending_return_addr = gc_return_addr;

        #[cfg(feature = "debug-gc")]
        eprintln!(
            "[GC DEBUG] walk_stack_roots: stack_base={:#x}, frame_pointer={:#x}, gc_return_addr={:#x}",
            stack_base, frame_pointer, gc_return_addr
        );

        while fp != 0 && fp < stack_base {
            let caller_fp = unsafe { *(fp as *const usize) };
            // This return address describes the CALLER frame (where we return to)
            let return_addr_for_caller = unsafe { *((fp + 8) as *const usize) };

            #[cfg(feature = "debug-gc")]
            eprintln!(
                "[GC DEBUG] Frame at FP={:#x}, pending_return_addr={:#x}, caller_fp={:#x}, return_addr_for_caller={:#x}",
                fp, pending_return_addr, caller_fp, return_addr_for_caller
            );

            // Use pending_return_addr to look up stack map for the CURRENT frame (fp)
            // pending_return_addr is the address where fp was called from
            if let Some(details) = stack_map.find_stack_data(pending_return_addr) {
                #[cfg(feature = "debug-gc")]
                eprintln!(
                    "[GC DEBUG] Scanning Beagle frame at FP={:#x}: fn={:?}, locals={}, max_stack={}, cur_stack={}",
                    fp,
                    details.function_name,
                    details.number_of_locals,
                    details.max_stack_size,
                    details.current_stack_size
                );

                let active_slots = details.number_of_locals + details.current_stack_size;

                for i in 0..active_slots {
                    // Locals are at [fp-8], [fp-16], etc.
                    let slot_addr = fp - 8 - (i * 8);
                    let slot_value = unsafe { *(slot_addr as *const usize) };

                    #[cfg(feature = "debug-gc")]
                    {
                        let tag = slot_value & 0x7;
                        let untagged = slot_value >> 3;
                        let tag_name = match tag {
                            0 => "Int",
                            1 => "Float",
                            2 => "String",
                            3 => "Bool",
                            4 => "Function",
                            5 => "Closure",
                            6 => "HeapObject",
                            7 => "Null",
                            _ => "Unknown",
                        };
                        eprintln!(
                            "[GC DEBUG]   slot[{}] @ {:#x} = {:#x} (tag={} [{}], untagged={:#x}), is_heap_ptr={}",
                            i,
                            slot_addr,
                            slot_value,
                            tag,
                            tag_name,
                            untagged,
                            BuiltInTypes::is_heap_pointer(slot_value)
                        );
                    }

                    if BuiltInTypes::is_heap_pointer(slot_value) {
                        let untagged = BuiltInTypes::untag(slot_value);
                        // Skip unaligned pointers - these are likely stale/garbage values
                        // that happen to match heap pointer tag patterns
                        if untagged % 8 != 0 {
                            #[cfg(feature = "debug-gc")]
                            eprintln!(
                                "[GC DEBUG]   SKIPPING unaligned pointer: slot[{}] @ {:#x} = {:#x}",
                                i, slot_addr, slot_value
                            );
                            continue;
                        }
                        callback(slot_addr, slot_value);
                    }
                }
            } else {
                #[cfg(feature = "debug-gc")]
                eprintln!(
                    "[GC DEBUG] Skipping non-Beagle frame at FP={:#x} (pending_return_addr={:#x} not in stack map)",
                    fp, pending_return_addr
                );
            }

            if caller_fp != 0 && caller_fp <= fp {
                #[cfg(feature = "debug-gc")]
                eprintln!(
                    "[GC DEBUG] FP chain invalid: caller_fp={:#x} <= fp={:#x}, stopping",
                    caller_fp, fp
                );
                break;
            }

            // Move to caller frame
            fp = caller_fp;
            // The return address we read describes the caller (now current fp)
            pending_return_addr = return_addr_for_caller;
        }

        #[cfg(feature = "debug-gc")]
        eprintln!("[GC DEBUG] walk_stack_roots done");
    }

    /// Get a mutable slice of the live stack for updating pointers after GC
    #[allow(dead_code)]
    pub fn get_live_stack_mut(stack_base: usize, frame_pointer: usize) -> &'static mut [usize] {
        let distance_till_end = stack_base - frame_pointer;
        let num_words = (distance_till_end / 8) + 1;

        unsafe { std::slice::from_raw_parts_mut(frame_pointer as *mut usize, num_words) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_slice_calculation() {
        // Test the calculation logic with known values
        let stack_base = 0x1000;
        let frame_pointer = 0x0F00; // 256 bytes down from base

        let slice = StackWalker::get_live_stack(stack_base, frame_pointer);

        // Distance is 256 bytes = 32 words, +1 = 33 words
        assert_eq!(slice.len(), 33);
        // Slice should start at frame_pointer
        assert_eq!(slice.as_ptr() as usize, frame_pointer);
    }
}
