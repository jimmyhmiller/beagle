use crate::collections::TYPE_ID_FRAME;
use crate::types::{BuiltInTypes, Header};

/// A simple abstraction for walking the stack and finding heap pointers
/// Uses frame pointer chain traversal for accurate stack walking.
pub struct StackWalker;

impl StackWalker {
    /// Collect all heap pointers from the stack.
    pub fn collect_stack_roots(
        stack_base: usize,
        frame_pointer: usize,
    ) -> Vec<(usize, usize)> {
        let mut roots = Vec::with_capacity(32);
        Self::walk_stack_roots(
            stack_base,
            frame_pointer,
            |addr, pointer| {
                roots.push((addr, pointer));
            },
        );
        roots
    }

    /// Walk the stack using the frame pointer chain, reading frame headers at [FP-8]
    /// to determine how many slots each frame has.
    ///
    /// `stack_base` is used as an upper bound to prevent walking beyond the Beagle
    /// stack into Rust frames. It is NOT used to scan GlobalObjectBlock slots —
    /// those are handled via extra_roots (ThreadGlobal.head_block).
    ///
    /// Each compiled Beagle function writes a heap-object-style header at [FP-8] with
    /// type_id == TYPE_ID_FRAME. The header's `size` field gives the number of
    /// traced slots (locals + eval stack) starting at [FP-16].
    /// Frames without a frame header (trampolines, Rust frames) are skipped.
    pub fn walk_stack_roots<F>(
        stack_base: usize,
        frame_pointer: usize,
        mut callback: F,
    ) where
        F: FnMut(usize, usize),
    {
        let mut fp = frame_pointer;

        #[cfg(feature = "debug-gc")]
        eprintln!(
            "[GC DEBUG] walk_stack_roots: stack_base={:#x}, frame_pointer={:#x}",
            stack_base, frame_pointer
        );

        while fp != 0 && fp < stack_base {
            let caller_fp = unsafe { *(fp as *const usize) };

            // Read frame header at [FP-8]
            let header_value = unsafe { *((fp - 8) as *const usize) };
            let header = Header::from_usize(header_value);

            #[cfg(feature = "debug-gc")]
            eprintln!(
                "[GC DEBUG] Frame at FP={:#x}, caller_fp={:#x}, header type_id={}, size={}",
                fp, caller_fp, header.type_id, header.size
            );

            // Only scan frames with a valid frame header
            if header.type_id == TYPE_ID_FRAME {
                let num_slots = header.size as usize;

                #[cfg(feature = "debug-gc")]
                eprintln!(
                    "[GC DEBUG] Scanning frame at FP={:#x}: num_slots={}",
                    fp, num_slots
                );

                for i in 0..num_slots {
                    // Locals are at [fp-16], [fp-24], etc. ([fp-8] is the frame header)
                    let slot_addr = fp - 16 - (i * 8);
                    let slot_value = unsafe { *(slot_addr as *const usize) };

                    #[cfg(feature = "debug-gc")]
                    {
                        let tag = slot_value & 0x7;
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
                            "[GC DEBUG]   slot[{}] @ {:#x} = {:#x} (tag={} [{}]), is_heap_ptr={}",
                            i, slot_addr, slot_value, tag, tag_name,
                            BuiltInTypes::is_heap_pointer(slot_value)
                        );
                    }

                    if BuiltInTypes::is_heap_pointer(slot_value) {
                        let untagged = BuiltInTypes::untag(slot_value);
                        if !untagged.is_multiple_of(8) {
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
            }

            #[cfg(feature = "debug-gc")]
            eprintln!("[GC DEBUG] Next FP: {:#x}", caller_fp);

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
        }

        #[cfg(feature = "debug-gc")]
        eprintln!("[GC DEBUG] walk_stack_roots done");
    }
}
