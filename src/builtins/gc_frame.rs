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
