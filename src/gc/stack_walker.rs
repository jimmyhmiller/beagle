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

    /// Walk the stack using the frame pointer chain and call a callback for each heap pointer found.
    /// The callback receives (slot_address, heap_pointer_value).
    ///
    /// ARM64 frame layout (stack grows down):
    ///   [FP+8]  = return address (LR) - tells us about the CALLER's frame
    ///   [FP]    = saved frame pointer (the CALLER's FP)
    ///   [FP-8]  = local 0 of CURRENT frame
    ///
    /// When we find a return address in the stack map, it describes the CALLER's frame,
    /// so we scan relative to the saved FP (caller's FP), not the current FP.
    pub fn walk_stack_roots<F>(
        stack_base: usize,
        frame_pointer: usize,
        stack_map: &StackMap,
        mut callback: F,
    ) where
        F: FnMut(usize, usize),
    {
        let mut fp = frame_pointer;

        #[cfg(feature = "debug-gc")]
        eprintln!(
            "[GC DEBUG] walk_stack_roots (FP-chain): stack_base={:#x}, frame_pointer={:#x}",
            stack_base, frame_pointer
        );

        // Follow the frame pointer chain
        while fp != 0 && fp < stack_base {
            // Read the return address at [FP+8] - this is where we return TO (in the caller)
            let return_addr = unsafe { *((fp + 8) as *const usize) };
            // Read the saved frame pointer at [FP] - this IS the caller's FP
            let caller_fp = unsafe { *(fp as *const usize) };

            #[cfg(feature = "debug-gc")]
            eprintln!(
                "[GC DEBUG] Frame at FP={:#x}, return_addr={:#x}, caller_fp={:#x}",
                fp, return_addr, caller_fp
            );

            // Check if this return address is a known Beagle safepoint
            // If so, the stack map describes the CALLER's frame (where we'll return to)
            if let Some(details) = stack_map.find_stack_data(return_addr) {
                #[cfg(feature = "debug-gc")]
                eprintln!(
                    "[GC DEBUG] Found Beagle frame: fn={:?}, locals={}, max_stack={}, cur_stack={}",
                    details.function_name,
                    details.number_of_locals,
                    details.max_stack_size,
                    details.current_stack_size
                );

                // Calculate how many slots to scan in the CALLER's frame
                // Locals are at [caller_fp-8], [caller_fp-16], etc.
                let active_slots = details.number_of_locals + details.current_stack_size;

                // Scan the active slots relative to the CALLER's FP
                for i in 0..active_slots {
                    let slot_addr = caller_fp - 8 - (i * 8);
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
                        debug_assert!(untagged % 8 == 0, "Pointer is not aligned");
                        callback(slot_addr, slot_value);
                    }
                }
            }

            #[cfg(feature = "debug-gc")]
            eprintln!("[GC DEBUG] Next FP: {:#x}", caller_fp);

            // Safety check: FP should be moving towards stack_base (higher addresses)
            if caller_fp != 0 && caller_fp <= fp {
                #[cfg(feature = "debug-gc")]
                eprintln!(
                    "[GC DEBUG] FP chain invalid: caller_fp={:#x} <= fp={:#x}, stopping",
                    caller_fp, fp
                );
                break;
            }

            fp = caller_fp;
        }

        #[cfg(feature = "debug-gc")]
        eprintln!("[GC DEBUG] walk_stack_roots done");
    }

    /// Collect all heap pointers from the stack into a vector
    /// Returns (slot_address, heap_pointer_value) pairs
    pub fn collect_stack_roots(
        stack_base: usize,
        frame_pointer: usize,
        stack_map: &StackMap,
    ) -> Vec<(usize, usize)> {
        let mut roots = Vec::with_capacity(32);
        Self::walk_stack_roots(stack_base, frame_pointer, stack_map, |addr, pointer| {
            roots.push((addr, pointer));
        });
        roots
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
