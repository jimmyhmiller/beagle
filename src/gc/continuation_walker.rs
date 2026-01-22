use crate::types::BuiltInTypes;

use super::StackMap;

/// Walker for continuation stack segments.
///
/// Continuation segments are raw Vec<u8> buffers containing copied stack frames.
/// This walker parses the frame pointer chain within the segment to find heap pointers,
/// similar to how StackWalker processes live stacks.
pub struct ContinuationSegmentWalker;

impl ContinuationSegmentWalker {
    /// Walk a continuation segment and invoke callback for each heap pointer found.
    ///
    /// # Arguments
    /// * `segment` - The raw bytes of the captured stack segment
    /// * `original_sp` - Stack pointer at capture time (base of segment)
    /// * `original_fp` - Frame pointer at capture time (first frame in segment)
    /// * `prompt_sp` - Upper bound (prompt handler's stack pointer)
    /// * `stack_map` - Stack map for looking up frame metadata
    /// * `callback` - Called with (offset_in_segment, tagged_heap_pointer)
    ///
    /// # Safety
    /// The segment must be a valid captured stack with proper frame chain structure.
    pub fn walk_segment_roots<F>(
        segment: &[u8],
        original_sp: usize,
        original_fp: usize,
        prompt_sp: usize,
        stack_map: &StackMap,
        mut callback: F,
    ) -> usize
    where
        F: FnMut(usize, usize),
    {
        if segment.is_empty() {
            return 0; // Empty segment fast path
        }

        let mut frame_count = 0;
        let mut fp = original_fp;

        #[cfg(feature = "debug-continuation-gc")]
        eprintln!(
            "[CONT-GC] Walking segment: sp={:#x}, fp={:#x}, prompt_sp={:#x}, size={}",
            original_sp, original_fp, prompt_sp, segment.len()
        );

        // Walk the frame pointer chain
        while fp < prompt_sp {
            // Convert absolute FP to segment offset
            let fp_offset = fp.checked_sub(original_sp);
            if fp_offset.is_none() || fp_offset.unwrap() >= segment.len() {
                #[cfg(feature = "debug-continuation-gc")]
                eprintln!("[CONT-GC] FP {:#x} out of segment bounds, stopping", fp);
                break;
            }
            let fp_offset = fp_offset.unwrap();

            // Read caller FP and return address from segment
            let caller_fp = Self::read_usize_from_segment(segment, fp_offset);
            let return_addr_offset = fp_offset.checked_add(8);
            if return_addr_offset.is_none() || return_addr_offset.unwrap() + 8 > segment.len() {
                break;
            }
            let return_addr = Self::read_usize_from_segment(segment, return_addr_offset.unwrap());

            #[cfg(feature = "debug-continuation-gc")]
            eprintln!(
                "[CONT-GC] Frame #{}: FP={:#x}, return_addr={:#x}, caller_fp={:#x}",
                frame_count, fp, return_addr, caller_fp
            );

            // Look up frame metadata in stack map
            if let Some(details) = stack_map.find_stack_data(return_addr) {
                let active_slots = details.number_of_locals + details.current_stack_size;

                #[cfg(feature = "debug-continuation-gc")]
                eprintln!(
                    "[CONT-GC]   Function: {:?}, locals={}, stack={}, active={}",
                    details.function_name,
                    details.number_of_locals,
                    details.current_stack_size,
                    active_slots
                );

                // Scan each slot for heap pointers
                for i in 0..active_slots {
                    // Locals are at [fp-8], [fp-16], etc.
                    let slot_addr = fp.wrapping_sub(8).wrapping_sub(i * 8);
                    if slot_addr < original_sp {
                        continue; // Slot is before segment start
                    }

                    let slot_offset = slot_addr - original_sp;
                    if slot_offset + 8 > segment.len() {
                        continue; // Slot extends beyond segment
                    }

                    let slot_value = Self::read_usize_from_segment(segment, slot_offset);

                    // Check if this is a heap pointer
                    if BuiltInTypes::is_heap_pointer(slot_value) {
                        let untagged = BuiltInTypes::untag(slot_value);
                        // Skip unaligned pointers (likely stale/garbage values)
                        if !untagged.is_multiple_of(8) {
                            #[cfg(feature = "debug-continuation-gc")]
                            eprintln!(
                                "[CONT-GC]   SKIP unaligned: slot[{}] @ offset {:#x} = {:#x}",
                                i, slot_offset, slot_value
                            );
                            continue;
                        }

                        #[cfg(feature = "debug-continuation-gc")]
                        eprintln!(
                            "[CONT-GC]   HEAP PTR: slot[{}] @ offset {:#x} = {:#x}",
                            i, slot_offset, slot_value
                        );

                        callback(slot_offset, slot_value);
                    }
                }
            } else {
                #[cfg(feature = "debug-continuation-gc")]
                eprintln!("[CONT-GC]   No stack map entry for return_addr {:#x}", return_addr);
            }

            // Validate FP chain
            if caller_fp != 0 && caller_fp <= fp {
                #[cfg(feature = "debug-continuation-gc")]
                eprintln!(
                    "[CONT-GC] FP chain invalid: caller_fp={:#x} <= fp={:#x}, stopping",
                    caller_fp, fp
                );
                break;
            }

            if caller_fp >= prompt_sp {
                // Reached prompt boundary
                break;
            }

            fp = caller_fp;
            frame_count += 1;
        }

        #[cfg(feature = "debug-continuation-gc")]
        eprintln!("[CONT-GC] Walked {} frames", frame_count);

        frame_count
    }

    /// Update pointers in a continuation segment after copying GC.
    ///
    /// This walks the segment, finds all heap pointers, and applies the updater function
    /// to get the new pointer value, then writes it back to the segment.
    ///
    /// # Arguments
    /// * `segment` - Mutable reference to the segment bytes
    /// * `original_sp` - Stack pointer at capture time
    /// * `original_fp` - Frame pointer at capture time
    /// * `prompt_sp` - Upper bound
    /// * `stack_map` - Stack map for looking up frame metadata
    /// * `updater` - Function that maps old_pointer -> new_pointer
    pub fn update_segment_pointers<F>(
        segment: &mut [u8],
        original_sp: usize,
        original_fp: usize,
        prompt_sp: usize,
        stack_map: &StackMap,
        mut updater: F,
    ) where
        F: FnMut(usize) -> usize,
    {
        if segment.is_empty() {
            return; // Empty segment fast path
        }

        let mut fp = original_fp;

        // Walk the frame pointer chain
        while fp < prompt_sp {
            let fp_offset = match fp.checked_sub(original_sp) {
                Some(offset) if offset < segment.len() => offset,
                _ => break,
            };

            // Read return address
            let return_addr_offset = match fp_offset.checked_add(8) {
                Some(offset) if offset + 8 <= segment.len() => offset,
                _ => break,
            };
            let return_addr = Self::read_usize_from_segment(segment, return_addr_offset);

            // Read caller FP for next iteration
            let caller_fp = Self::read_usize_from_segment(segment, fp_offset);

            // Look up frame metadata
            if let Some(details) = stack_map.find_stack_data(return_addr) {
                let active_slots = details.number_of_locals + details.current_stack_size;

                // Update each heap pointer slot
                for i in 0..active_slots {
                    let slot_addr = fp.wrapping_sub(8).wrapping_sub(i * 8);
                    if slot_addr < original_sp {
                        continue;
                    }

                    let slot_offset = slot_addr - original_sp;
                    if slot_offset + 8 > segment.len() {
                        continue;
                    }

                    let old_value = Self::read_usize_from_segment(segment, slot_offset);

                    if BuiltInTypes::is_heap_pointer(old_value) {
                        let untagged = BuiltInTypes::untag(old_value);
                        if !untagged.is_multiple_of(8) {
                            continue; // Skip unaligned
                        }

                        let new_value = updater(old_value);

                        // Write back the updated pointer
                        if new_value != old_value {
                            Self::write_usize_to_segment(segment, slot_offset, new_value);

                            #[cfg(feature = "debug-continuation-gc")]
                            eprintln!(
                                "[CONT-GC] Updated pointer @ offset {:#x}: {:#x} -> {:#x}",
                                slot_offset, old_value, new_value
                            );
                        }
                    }
                }
            }

            // Validate and advance FP chain
            if caller_fp != 0 && caller_fp <= fp {
                break;
            }
            if caller_fp >= prompt_sp {
                break;
            }

            fp = caller_fp;
        }
    }

    /// Read a usize value from a segment at the given offset.
    ///
    /// # Safety
    /// Caller must ensure offset + 8 <= segment.len()
    #[inline]
    fn read_usize_from_segment(segment: &[u8], offset: usize) -> usize {
        debug_assert!(offset + 8 <= segment.len(), "Read out of bounds");
        unsafe {
            let ptr = segment.as_ptr().add(offset) as *const usize;
            *ptr
        }
    }

    /// Write a usize value to a segment at the given offset.
    ///
    /// # Safety
    /// Caller must ensure offset + 8 <= segment.len()
    #[inline]
    fn write_usize_to_segment(segment: &mut [u8], offset: usize, value: usize) {
        debug_assert!(offset + 8 <= segment.len(), "Write out of bounds");
        unsafe {
            let ptr = segment.as_mut_ptr().add(offset) as *mut usize;
            *ptr = value;
        }
    }
}
