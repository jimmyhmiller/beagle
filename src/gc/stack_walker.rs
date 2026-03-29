use crate::collections::TYPE_ID_FRAME;
use crate::types::{BuiltInTypes, Header};

/// Stack walker that traverses the GC frame linked list (Henderson/Pizderson frames).
///
/// Each compiled Beagle function links its frame into a per-thread linked list via
/// `gc_frame_link` in the prologue and `gc_frame_unlink` in the epilogue.
///
/// Frame layout:
///   [FP+8]  : saved return address
///   [FP+0]  : saved caller FP
///   [FP-8]  : frame header (TYPE_ID_FRAME, size=num_slots)  ← header_addr
///   [FP-16] : prev pointer (raw address of previous frame's header, or 0)
///   [FP-24] : local[0]
///   [FP-32] : local[1]
///   ...
///
/// GC walks the prev chain starting from GC_FRAME_TOP, independent of the FP chain.
/// This means captured continuation frames are naturally excluded from scanning
/// (they were unlinked when captured), and restored frames are included (they were
/// re-linked during restoration).
pub struct StackWalker;

impl StackWalker {
    /// Collect all heap pointers from the GC frame chain for a single thread.
    /// `gc_frame_top` is the address of the topmost frame's header (or 0 for empty).
    pub fn collect_stack_roots(gc_frame_top: usize) -> Vec<(usize, usize)> {
        let mut roots = Vec::with_capacity(32);
        Self::walk_stack_roots(gc_frame_top, |addr, pointer| {
            roots.push((addr, pointer));
        });
        roots
    }

    /// Walk the GC frame linked list, scanning each frame's local slots for heap pointers.
    ///
    /// `gc_frame_top` is the address of the topmost frame's header word.
    /// The prev pointer is at `header_addr - 8` (i.e., [FP-16]).
    /// Locals start at `header_addr - 16` (i.e., [FP-24]).
    pub fn walk_stack_roots<F>(gc_frame_top: usize, mut callback: F)
    where
        F: FnMut(usize, usize),
    {
        let mut header_addr = gc_frame_top;
        #[cfg(feature = "debug-gc")]
        let mut fast = gc_frame_top;

        #[cfg(feature = "debug-gc")]
        eprintln!(
            "[GC DEBUG] walk_stack_roots: gc_frame_top={:#x}",
            gc_frame_top
        );

        while header_addr != 0 {
            // Read frame header
            let header_value = unsafe { *(header_addr as *const usize) };
            let header = Header::from_usize(header_value);

            #[cfg(feature = "debug-gc")]
            eprintln!(
                "[GC DEBUG] Frame header at {:#x}, type_id={}, size={}",
                header_addr, header.type_id, header.size
            );

            // Only scan frames with a valid frame header
            if header.type_id == TYPE_ID_FRAME {
                let num_slots = header.size as usize;

                #[cfg(feature = "debug-gc")]
                eprintln!(
                    "[GC DEBUG] Scanning frame at header={:#x}: num_slots={}",
                    header_addr, num_slots
                );

                for i in 0..num_slots {
                    // Locals are at [header_addr - 16], [header_addr - 24], etc.
                    // (header_addr - 8 is the prev pointer)
                    let slot_addr = header_addr - 16 - (i * 8);
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
                            i,
                            slot_addr,
                            slot_value,
                            tag,
                            tag_name,
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

            // Follow prev pointer at [header_addr - 8]
            let prev_value = unsafe { *((header_addr - 8) as *const usize) };

            #[cfg(feature = "debug-gc")]
            eprintln!("[GC DEBUG] Next prev: {:#x}", prev_value);

            // prev_value is a raw address (not tagged), pointing to previous frame's header
            // A value of 0 means end of chain
            header_addr = prev_value;

            #[cfg(feature = "debug-gc")]
            {
                // Floyd's cycle detection: advance hare two steps per tortoise step.
                // If we ever detect a cycle, panic — it means the GC prev fix in
                // continuation restore has a gap we need to investigate.
                for _ in 0..2 {
                    if fast != 0 {
                        fast = unsafe { *((fast - 8) as *const usize) };
                    }
                }
                if header_addr != 0 && header_addr == fast {
                    panic!(
                        "BUG: cycle in GC frame chain at {:#x} — saved_gc_prev fix didn't cover this case",
                        header_addr
                    );
                }
            }
        }

        #[cfg(feature = "debug-gc")]
        eprintln!("[GC DEBUG] walk_stack_roots done");
    }

    /// Walk stack roots in a detached stack segment by following the saved FP chain.
    ///
    /// `frame_pointer` is the innermost active Beagle frame within the detached segment.
    /// `segment_base..segment_top` is the mapped stack region for that segment.
    pub fn walk_segment_roots<F>(
        frame_pointer: usize,
        segment_base: usize,
        segment_top: usize,
        mut callback: F,
    ) where
        F: FnMut(usize, usize),
    {
        let mut fp = frame_pointer;
        #[cfg(feature = "debug-gc")]
        let mut fast = frame_pointer;

        while fp >= segment_base && fp < segment_top && fp != 0 {
            let header_addr = fp.saturating_sub(8);
            let header_value = unsafe { *(header_addr as *const usize) };
            let header = Header::from_usize(header_value);

            if header.type_id != TYPE_ID_FRAME {
                break;
            }

            let num_slots = header.size as usize;
            for i in 0..num_slots {
                let slot_addr = header_addr - 16 - (i * 8);
                let slot_value = unsafe { *(slot_addr as *const usize) };
                if BuiltInTypes::is_heap_pointer(slot_value) {
                    let untagged = BuiltInTypes::untag(slot_value);
                    if untagged != 0 && untagged.is_multiple_of(8) {
                        callback(slot_addr, slot_value);
                    }
                }
            }

            let caller_fp = unsafe { *(fp as *const usize) };
            if caller_fp == 0 || caller_fp <= fp {
                break;
            }
            if caller_fp < segment_base || caller_fp >= segment_top {
                break;
            }
            fp = caller_fp;

            #[cfg(feature = "debug-gc")]
            {
                for _ in 0..2 {
                    if fast >= segment_base && fast < segment_top && fast != 0 {
                        fast = unsafe { *(fast as *const usize) };
                    }
                }
                if fp != 0 && fp == fast {
                    panic!("BUG: cycle in detached segment FP chain at {:#x}", fp);
                }
            }
        }
    }
}
