use crate::types::{BuiltInTypes, HeapObject};

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

    /// Collect all heap pointers from the stack, with an explicit return address for the first frame.
    /// gc_return_addr is the return address of the gc() call - this is the safepoint that describes
    /// the first frame's locals. If gc_return_addr is 0, falls back to [FP+8] lookup.
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

    /// Walk the stack using the frame pointer chain, with an explicit return address for the first frame.
    ///
    /// Key insight: The return address at [FP+8] describes the CALLER's frame, not the current frame.
    /// So we track the "pending" return address from the previous frame to know how to scan the current frame.
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
        // pending_return_addr describes the CURRENT frame (fp).
        // For the first frame, this is gc_return_addr.
        // For subsequent frames, this is the return address read from the PREVIOUS frame.
        let mut pending_return_addr = gc_return_addr;

        #[cfg(feature = "debug-gc")]
        eprintln!(
            "[GC DEBUG] walk_stack_roots_with_return_addr: stack_base={:#x}, frame_pointer={:#x}, gc_return_addr={:#x}",
            stack_base, frame_pointer, gc_return_addr
        );

        // Check the GlobalObjectBlock slot at stack_base - 8
        // This is where the Runtime stores the GlobalObjectBlock pointer for this thread.
        // The trampoline skips this slot, so it's preserved throughout execution.
        // TODO: Ideally this would be a regular stack local tracked by the stack map
        // at the base frame, removing the need for this special case.
        //
        // IMPORTANT: We walk the entire chain of GlobalObjectBlocks here.
        // Each block has field 0 as next_block pointer. If we only trace the first
        // block, subsequent blocks' entries won't be found by GC and young gen
        // pointers in them won't be updated after copying.
        let global_block_slot_addr = stack_base - 8;
        let global_block_value = unsafe { *(global_block_slot_addr as *const usize) };
        if BuiltInTypes::is_heap_pointer(global_block_value) {
            let untagged = BuiltInTypes::untag(global_block_value);
            if untagged.is_multiple_of(8) {
                #[cfg(feature = "debug-gc")]
                eprintln!(
                    "[GC DEBUG] GlobalObjectBlock slot @ {:#x} = {:#x}",
                    global_block_slot_addr, global_block_value
                );
                callback(global_block_slot_addr, global_block_value);

                // Walk the chain of GlobalObjectBlocks.
                // Field 0 of each block is the next_block pointer.
                // Field 1 is count (tagged int), fields 2..66 are entries.
                let mut current_block = global_block_value;
                loop {
                    let heap_obj = HeapObject::from_tagged(current_block);
                    let next_block = heap_obj.get_field(0);

                    // Check if next_block is a valid heap pointer (not null/free)
                    // null_value is 0b111, and free slots also use 0b111
                    if !BuiltInTypes::is_heap_pointer(next_block) || next_block == 0b111 {
                        break;
                    }

                    let next_untagged = BuiltInTypes::untag(next_block);
                    if !next_untagged.is_multiple_of(8) {
                        break;
                    }

                    #[cfg(feature = "debug-gc")]
                    eprintln!(
                        "[GC DEBUG] GlobalObjectBlock chain: next block @ {:#x}",
                        next_block
                    );

                    // The slot address for this block is field 0 of the previous block.
                    // We need to get the address of field 0 in the heap object.
                    let prev_heap_obj = HeapObject::from_tagged(current_block);
                    let slot_addr = prev_heap_obj.get_field_ptr(0) as usize;
                    callback(slot_addr, next_block);

                    current_block = next_block;
                }
            }
        }

        while fp != 0 && fp < stack_base {
            let caller_fp = unsafe { *(fp as *const usize) };
            // Read return address from current frame - this describes the CALLER (caller_fp)
            let return_addr_for_caller = unsafe { *((fp + 8) as *const usize) };

            #[cfg(feature = "debug-gc")]
            eprintln!(
                "[GC DEBUG] Frame at FP={:#x}, pending_return_addr={:#x}, caller_fp={:#x}, return_addr_for_caller={:#x}",
                fp, pending_return_addr, caller_fp, return_addr_for_caller
            );

            // Use pending_return_addr to scan the CURRENT frame (fp)
            if pending_return_addr != 0
                && let Some(details) = stack_map.find_stack_data(pending_return_addr)
            {
                #[cfg(feature = "debug-gc")]
                eprintln!(
                    "[GC DEBUG] Scanning frame at FP={:#x}: fn={:?}, locals={}, max_stack={}, cur_stack={}, callee_saved={}",
                    fp,
                    details.function_name,
                    details.number_of_locals,
                    details.max_stack_size,
                    details.current_stack_size,
                    details.num_callee_saved
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

                // NOTE: Callee-saved register spill area is NOT scanned.
                // The spill area at the bottom of the frame (near SP) contains values
                // from the CALLER's registers, which may originate from Rust code
                // anywhere up the call chain (not just the immediate caller). Since we
                // can't reliably distinguish Beagle values from Rust values in these
                // slots, scanning them can cause crashes (null deref, invalid pointers).
                // The register allocator ensures all live Beagle values are backed by
                // local/eval stack slots, which ARE scanned.
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
            // The return address we read describes the caller (now current fp)
            pending_return_addr = return_addr_for_caller;
        }

        #[cfg(feature = "debug-gc")]
        eprintln!("[GC DEBUG] walk_stack_roots_with_return_addr done");
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
