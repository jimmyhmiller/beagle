use super::*;

pub fn record_gc_chain_event(event: String) {
    if std::env::var("BEAGLE_DEBUG_GC_CHAIN_TRACE").is_err() {
        return;
    }
    GC_CHAIN_TRACE.with(|trace| {
        let mut trace = trace.borrow_mut();
        if trace.len() >= 128 {
            trace.pop_front();
        }
        trace.push_back(event);
    });
}

pub fn dump_gc_chain_trace() {
    if std::env::var("BEAGLE_DEBUG_GC_CHAIN_TRACE").is_err() {
        return;
    }
    GC_CHAIN_TRACE.with(|trace| {
        eprintln!("[gc-chain-trace] recent events:");
        for event in trace.borrow().iter() {
            eprintln!("  {}", event);
        }
    });
}

pub fn skip_empty_gc_frames(mut header_addr: usize) -> usize {
    while header_addr != 0 {
        if !header_addr.is_multiple_of(8) {
            break;
        }
        let header = Header::from_usize(unsafe { *(header_addr as *const usize) });
        if header.type_id != TYPE_ID_FRAME || header.size != 0 {
            break;
        }
        let next = unsafe { *((header_addr - 8) as *const usize) };
        if next == header_addr {
            return 0;
        }
        header_addr = next;
    }
    header_addr
}

pub fn gc_chain_prev_for_restored_segment(
    segment_start: usize,
    segment_len: usize,
    outermost_fp: usize,
) -> usize {
    let current_top = skip_empty_gc_frames(get_gc_frame_top());
    let segment_end = segment_start.saturating_add(segment_len);

    // If the current GC top already points into the segment we are restoring,
    // that segment is being rewritten in place. Preserve the outermost frame's
    // original predecessor instead of linking the segment back under its own top.
    if current_top >= segment_start && current_top < segment_end {
        unsafe { *((outermost_fp - 16) as *const usize) }
    } else {
        current_top
    }
}

pub fn gc_chain_anchor_for_invocation(beagle_fp: usize) -> usize {
    let header = Header::from_usize(unsafe { *((beagle_fp - 8) as *const usize) });
    if header.type_id == TYPE_ID_FRAME {
        skip_empty_gc_frames(unsafe { *((beagle_fp - 16) as *const usize) })
    } else {
        skip_empty_gc_frames(get_gc_frame_top())
    }
}

pub fn frame_pointer_in_active_segment(frame_pointer: usize) -> bool {
    let ptd = crate::runtime::per_thread_data();
    ptd.active_segments
        .iter()
        .any(|active| frame_pointer >= active.segment.base && frame_pointer < active.segment.top)
}

pub fn segment_frame_pointer_is_valid(
    frame_pointer: usize,
    segment_base: usize,
    segment_top: usize,
) -> bool {
    if frame_pointer < segment_base.saturating_add(8)
        || frame_pointer.saturating_add(8) >= segment_top
    {
        return false;
    }

    let header = Header::from_usize(unsafe { *((frame_pointer - 8) as *const usize) });
    if header.type_id != TYPE_ID_FRAME {
        return false;
    }

    let saved_fp = unsafe { *(frame_pointer as *const usize) };
    if saved_fp >= segment_base.saturating_add(8)
        && saved_fp < segment_top
        && saved_fp <= frame_pointer
    {
        return false;
    }

    let return_addr = unsafe { *((frame_pointer + 8) as *const usize) };
    if return_addr < 0x1000 {
        return false;
    }

    true
}

pub fn find_segment_innermost_frame_pointer(
    stack_pointer: usize,
    segment_top: usize,
    gc_frame_top: usize,
    raw_frame_pointer: usize,
) -> Option<usize> {
    let gc_frame_pointer = gc_frame_top.saturating_add(8);
    if gc_frame_top >= stack_pointer
        && gc_frame_top < segment_top
        && segment_frame_pointer_is_valid(gc_frame_pointer, stack_pointer, segment_top)
    {
        return Some(gc_frame_pointer);
    }

    if segment_frame_pointer_is_valid(raw_frame_pointer, stack_pointer, segment_top) {
        return Some(raw_frame_pointer);
    }

    None
}

/// Called by JIT prologue AFTER arguments have been saved to locals.
/// Links the new frame into the GC frame chain.
/// Returns the old gc_frame_top (to be stored as the prev pointer at [FP-16]).
#[unsafe(no_mangle)]
pub extern "C" fn gc_frame_link(frame_header_addr: usize) -> usize {
    let prev = GC_FRAME_TOP.with(|cell| cell.get());
    if std::env::var("BEAGLE_DEBUG_GC_CHAIN_TRACE").is_ok() {
        let header = Header::from_usize(unsafe { *(frame_header_addr as *const usize) });
        let fp = frame_header_addr + 8;
        let saved_fp = unsafe { *(fp as *const usize) };
        let return_addr = unsafe { *((fp + 8) as *const usize) };
        let function = crate::get_runtime()
            .get()
            .get_function_containing_pointer(return_addr as *const u8)
            .map(|(function, offset)| format!(" {}+{:#x}", function.name, offset))
            .unwrap_or_default();
        record_gc_chain_event(format!(
            "gc_frame_link header={:#x} fp={:#x} type_id={} size={} saved_fp={:#x} ret={:#x} prev={:#x}{}",
            frame_header_addr,
            fp,
            header.type_id,
            header.size,
            saved_fp,
            return_addr,
            prev,
            function
        ));
    }
    if std::env::var("BEAGLE_DEBUG_GC_CHAIN_WRITES").is_ok() {
        let header = Header::from_usize(unsafe { *(frame_header_addr as *const usize) });
        let fp = frame_header_addr + 8;
        let saved_fp = unsafe { *(fp as *const usize) };
        let return_addr = unsafe { *((fp + 8) as *const usize) };
        if header.type_id != TYPE_ID_FRAME || saved_fp == 0 || return_addr < 0x1000 {
            if let Some((function, offset)) = crate::get_runtime()
                .get()
                .get_function_containing_pointer(return_addr as *const u8)
            {
                eprintln!(
                    "[gc-top-write] via=gc_frame_link header={:#x} fp={:#x} type_id={} size={} saved_fp={:#x} ret={:#x} prev={:#x} fn={}+{:#x}",
                    frame_header_addr,
                    fp,
                    header.type_id,
                    header.size,
                    saved_fp,
                    return_addr,
                    prev,
                    function.name,
                    offset
                );
            } else {
                eprintln!(
                    "[gc-top-write] via=gc_frame_link header={:#x} fp={:#x} type_id={} size={} saved_fp={:#x} ret={:#x} prev={:#x}",
                    frame_header_addr, fp, header.type_id, header.size, saved_fp, return_addr, prev
                );
            }
        }
    }
    GC_FRAME_TOP.with(|cell| cell.set(frame_header_addr));
    if prev == frame_header_addr {
        // Resumed/rewritten frames can reuse the same header slot. In that case,
        // preserve the predecessor already stored in the frame instead of
        // linking the frame under itself and creating a one-node cycle.
        unsafe { *((frame_header_addr - 8) as *const usize) }
    } else {
        prev
    }
}

/// Called by JIT epilogue BEFORE restoring the return value.
/// Unlinks the current frame from the GC frame chain.
#[unsafe(no_mangle)]
pub extern "C" fn gc_frame_unlink(prev: usize) {
    if std::env::var("BEAGLE_DEBUG_GC_CHAIN_TRACE").is_ok() {
        let current_top = GC_FRAME_TOP.with(|cell| cell.get());
        record_gc_chain_event(format!(
            "gc_frame_unlink current_top={:#x} new_top={:#x}",
            current_top, prev
        ));
    }
    GC_FRAME_TOP.with(|cell| cell.set(prev));
}
