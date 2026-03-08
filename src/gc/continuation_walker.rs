use crate::collections::TYPE_ID_FRAME;
use crate::types::{BuiltInTypes, Header};

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
    /// * `callback` - Called with (offset_in_segment, tagged_heap_pointer)
    ///
    /// # Safety
    /// The segment must be a valid captured stack with proper frame chain structure.
    pub fn walk_segment_roots<F>(
        segment: &[u8],
        original_sp: usize,
        original_fp: usize,
        prompt_sp: usize,
        mut callback: F,
    ) -> usize
    where
        F: FnMut(usize, usize),
    {
        if segment.is_empty() {
            return 0;
        }

        let mut frame_count = 0;
        let mut fp = original_fp;

        #[cfg(feature = "debug-continuation-gc")]
        eprintln!(
            "[CONT-GC] Walking segment: sp={:#x}, fp={:#x}, prompt_sp={:#x}, size={}",
            original_sp, original_fp, prompt_sp, segment.len()
        );

        while fp < prompt_sp {
            let fp_offset = fp.checked_sub(original_sp);
            if fp_offset.is_none() || fp_offset.unwrap() >= segment.len() {
                break;
            }
            let fp_offset = fp_offset.unwrap();

            let caller_fp = Self::read_usize_from_segment(segment, fp_offset);

            // Read frame header at [fp-8] from the segment
            let header_offset = fp.wrapping_sub(8).checked_sub(original_sp);
            if header_offset.is_none() || header_offset.unwrap() + 8 > segment.len() {
                break;
            }
            let header_value = Self::read_usize_from_segment(segment, header_offset.unwrap());
            let header = Header::from_usize(header_value);

            if header.type_id == TYPE_ID_FRAME {
                let num_slots = header.size as usize;

                #[cfg(feature = "debug-continuation-gc")]
                eprintln!(
                    "[CONT-GC] Frame #{}: FP={:#x}, num_slots={}",
                    frame_count, fp, num_slots
                );

                for i in 0..num_slots {
                    let slot_addr = fp.wrapping_sub(16).wrapping_sub(i * 8);
                    if slot_addr < original_sp {
                        continue;
                    }

                    let slot_offset = slot_addr - original_sp;
                    if slot_offset + 8 > segment.len() {
                        continue;
                    }

                    let slot_value = Self::read_usize_from_segment(segment, slot_offset);

                    if BuiltInTypes::is_heap_pointer(slot_value) {
                        let untagged = BuiltInTypes::untag(slot_value);
                        if !untagged.is_multiple_of(8) {
                            continue;
                        }
                        callback(slot_offset, slot_value);
                    }
                }
            }

            if caller_fp != 0 && caller_fp <= fp {
                break;
            }
            if caller_fp >= prompt_sp {
                break;
            }

            fp = caller_fp;
            frame_count += 1;
        }

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
    /// * `updater` - Function that maps old_pointer -> new_pointer
    pub fn update_segment_pointers<F>(
        segment: &mut [u8],
        original_sp: usize,
        original_fp: usize,
        prompt_sp: usize,
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

            // Read caller FP for next iteration
            let caller_fp = Self::read_usize_from_segment(segment, fp_offset);

            // Read frame header at [fp-8] from the segment
            let header_offset = match fp.wrapping_sub(8).checked_sub(original_sp) {
                Some(offset) if offset + 8 <= segment.len() => offset,
                _ => break,
            };
            let header_value = Self::read_usize_from_segment(segment, header_offset);
            let header = Header::from_usize(header_value);

            if header.type_id == TYPE_ID_FRAME {
                let num_slots = header.size as usize;

                // Update each heap pointer slot
                for i in 0..num_slots {
                    // Locals at [fp-16], [fp-24], etc. ([fp-8] is frame header)
                    let slot_addr = fp.wrapping_sub(16).wrapping_sub(i * 8);
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

    /// Walk a saved stack frame (single frame, not a frame chain) and invoke callback
    /// for each heap pointer found.
    ///
    /// Unlike `walk_segment_roots` which follows a frame pointer chain through a
    /// continuation segment, this scans a single frame's locals using the frame
    /// header at [FP-8].
    ///
    /// # Arguments
    /// * `segment` - The saved frame bytes (from stack_pointer to frame_pointer)
    /// * `stack_pointer` - The original stack pointer (base of the saved frame)
    /// * `frame_pointer` - The original frame pointer (top of the saved frame)
    /// * `callback` - Called with (offset_in_segment, tagged_heap_pointer)
    pub fn walk_saved_frame_roots<F>(
        segment: &[u8],
        stack_pointer: usize,
        frame_pointer: usize,
        mut callback: F,
    ) where
        F: FnMut(usize, usize),
    {
        if segment.is_empty() {
            return;
        }

        // Read frame header at [fp-8] from the segment
        let header_offset = match frame_pointer.wrapping_sub(8).checked_sub(stack_pointer) {
            Some(offset) if offset + 8 <= segment.len() => offset,
            _ => {
                // No frame header accessible — conservatively scan all words
                for offset in (0..segment.len()).step_by(8) {
                    if offset + 8 > segment.len() {
                        break;
                    }
                    let value = Self::read_usize_from_segment(segment, offset);
                    if BuiltInTypes::is_heap_pointer(value) {
                        let untagged = BuiltInTypes::untag(value);
                        if untagged != 0 && untagged.is_multiple_of(8) {
                            callback(offset, value);
                        }
                    }
                }
                return;
            }
        };

        let header_value = Self::read_usize_from_segment(segment, header_offset);
        let header = Header::from_usize(header_value);

        let num_slots = if header.type_id == TYPE_ID_FRAME {
            header.size as usize
        } else {
            // No valid frame header — conservatively scan all words
            for offset in (0..segment.len()).step_by(8) {
                if offset + 8 > segment.len() {
                    break;
                }
                let value = Self::read_usize_from_segment(segment, offset);
                if BuiltInTypes::is_heap_pointer(value) {
                    let untagged = BuiltInTypes::untag(value);
                    if untagged != 0 && untagged.is_multiple_of(8) {
                        callback(offset, value);
                    }
                }
            }
            return;
        };

        for i in 0..num_slots {
            // Locals at [fp-16], [fp-24], etc. ([fp-8] is frame header)
            let slot_addr = frame_pointer.wrapping_sub(16).wrapping_sub(i * 8);
            if slot_addr < stack_pointer {
                continue;
            }
            let slot_offset = slot_addr - stack_pointer;
            if slot_offset + 8 > segment.len() {
                continue;
            }
            let value = Self::read_usize_from_segment(segment, slot_offset);
            if BuiltInTypes::is_heap_pointer(value) {
                let untagged = BuiltInTypes::untag(value);
                if untagged != 0 && untagged.is_multiple_of(8) {
                    callback(slot_offset, value);
                }
            }
        }
    }

    /// Update pointers in a saved stack frame (single frame).
    /// Same as `walk_saved_frame_roots` but applies an updater function and writes back.
    pub fn update_saved_frame_pointers<F>(
        segment: &mut [u8],
        stack_pointer: usize,
        frame_pointer: usize,
        mut updater: F,
    ) where
        F: FnMut(usize) -> usize,
    {
        if segment.is_empty() {
            return;
        }

        // Read frame header at [fp-8] from the segment
        let header_offset = match frame_pointer.wrapping_sub(8).checked_sub(stack_pointer) {
            Some(offset) if offset + 8 <= segment.len() => offset,
            _ => {
                // No frame header accessible — conservatively scan all words
                for offset in (0..segment.len()).step_by(8) {
                    if offset + 8 > segment.len() {
                        break;
                    }
                    let value = Self::read_usize_from_segment(segment, offset);
                    if BuiltInTypes::is_heap_pointer(value) {
                        let untagged = BuiltInTypes::untag(value);
                        if untagged != 0 && untagged.is_multiple_of(8) {
                            let new_value = updater(value);
                            if new_value != value {
                                Self::write_usize_to_segment(segment, offset, new_value);
                            }
                        }
                    }
                }
                return;
            }
        };

        let header_value = Self::read_usize_from_segment(segment, header_offset);
        let header = Header::from_usize(header_value);

        let num_slots = if header.type_id == TYPE_ID_FRAME {
            header.size as usize
        } else {
            // No valid frame header — conservatively scan all words
            for offset in (0..segment.len()).step_by(8) {
                if offset + 8 > segment.len() {
                    break;
                }
                let value = Self::read_usize_from_segment(segment, offset);
                if BuiltInTypes::is_heap_pointer(value) {
                    let untagged = BuiltInTypes::untag(value);
                    if untagged != 0 && untagged.is_multiple_of(8) {
                        let new_value = updater(value);
                        if new_value != value {
                            Self::write_usize_to_segment(segment, offset, new_value);
                        }
                    }
                }
            }
            return;
        };

        for i in 0..num_slots {
            // Locals at [fp-16], [fp-24], etc. ([fp-8] is frame header)
            let slot_addr = frame_pointer.wrapping_sub(16).wrapping_sub(i * 8);
            if slot_addr < stack_pointer {
                continue;
            }
            let slot_offset = slot_addr - stack_pointer;
            if slot_offset + 8 > segment.len() {
                continue;
            }
            let value = Self::read_usize_from_segment(segment, slot_offset);
            if BuiltInTypes::is_heap_pointer(value) {
                let untagged = BuiltInTypes::untag(value);
                if untagged != 0 && untagged.is_multiple_of(8) {
                    let new_value = updater(value);
                    if new_value != value {
                        Self::write_usize_to_segment(segment, slot_offset, new_value);
                    }
                }
            }
        }
    }

    /// Read a usize value from a segment at the given offset.
    ///
    /// # Safety
    /// Caller must ensure offset + 8 <= segment.len()
    #[inline]
    pub(crate) fn read_usize_from_segment(segment: &[u8], offset: usize) -> usize {
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
    pub(crate) fn write_usize_to_segment(segment: &mut [u8], offset: usize, value: usize) {
        debug_assert!(offset + 8 <= segment.len(), "Write out of bounds");
        unsafe {
            let ptr = segment.as_mut_ptr().add(offset) as *mut usize;
            *ptr = value;
        }
    }
}
