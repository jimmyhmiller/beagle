use super::*;
use crate::save_gc_context;
use crate::trace;

pub fn current_saved_continuation_ptr(fallback: usize) -> usize {
    crate::runtime::per_thread_data()
        .saved_continuation_ptrs
        .last()
        .copied()
        .unwrap_or(fallback)
}

pub fn debug_cont_state(label: &str, prompt_id: usize) {
    if std::env::var("BEAGLE_DEBUG_CONT_STATE").is_err() {
        return;
    }
    let ptd = crate::runtime::per_thread_data();
    eprintln!(
        "[cont-state] {} prompt_id={} prompts={} active_segments={} rps={} suspended={} saved_conts={}",
        label,
        prompt_id,
        ptd.prompt_handlers.len(),
        ptd.active_segments.len(),
        ptd.invocation_return_points.len(),
        ptd.suspended_frames.len(),
        ptd.saved_continuation_ptrs.len()
    );
}

pub fn debug_captured_continuation_refs(cont_ptr: usize) {
    if std::env::var("BEAGLE_DEBUG_CONT_REFS").is_err() {
        return;
    }
    let Some(cont) = ContinuationObject::from_tagged(cont_ptr) else {
        return;
    };
    let mut refs = Vec::new();
    if let Some((segment_base, segment_top, gc_frame_top)) = cont.segment_gc_frame_info() {
        if gc_frame_top >= segment_base && gc_frame_top < segment_top {
            crate::gc::stack_walker::StackWalker::walk_segment_gc_roots(
                gc_frame_top,
                segment_base,
                segment_top,
                |slot_addr, slot_value| {
                    if let Some(obj) = HeapObject::try_from_tagged(slot_value)
                        && ContinuationObject::from_heap_object(obj).is_some()
                    {
                        refs.push((slot_addr, slot_value));
                    }
                },
            );
        }
    }
    if !refs.is_empty() {
        eprintln!(
            "[cont-refs] cont={:#x} prompt_id={} refs={:?}",
            cont_ptr,
            cont.prompt_id(),
            refs
        );
    }
}

pub fn decode_arm_mov_imm3(words: &[u32], reg: u8) -> Option<usize> {
    if words.len() < 3 {
        return None;
    }
    let mut value = 0usize;
    for &word in words.iter().take(3) {
        let rd = (word & 0x1f) as u8;
        if rd != reg {
            return None;
        }
        let imm16 = ((word >> 5) & 0xffff) as usize;
        let hw = ((word >> 21) & 0x3) as usize;
        value |= imm16 << (hw * 16);
    }
    Some(value)
}

#[inline]
pub unsafe fn allocate_string_or_throw(
    runtime: &mut Runtime,
    stack_pointer: usize,
    s: String,
) -> usize {
    match runtime.allocate_string(stack_pointer, s) {
        Ok(ptr) => ptr.into(),
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate string - out of memory".to_string(),
            );
        },
    }
}

// ============================================================================
// Delimited Continuation Builtins
// ============================================================================

/// Push a prompt handler for delimited continuations.
/// Similar to push_exception_handler but for continuation capture.
/// REFACTOR A stub. `push-prompt` was the v1 prompt entry point used by
/// `handle`, `perform`, and `try/catch` with resumable exceptions. It is
/// no longer wired to anything — the new reset/shift machinery doesn't
/// use it. Kept as a named entry point so Ast::Handle / Ast::Perform /
/// Ast::TryResumable compile-time builtin lookups still succeed; any
/// program that actually reaches this code path aborts with a clear
/// error. Delete alongside Refactor B when those AST nodes are rebuilt.
pub unsafe extern "C" fn push_prompt_runtime(
    _handler_address: usize,
    _result_local: isize,
    _link_register: usize,
    stack_pointer: usize,
    _frame_pointer: usize,
) -> usize {
    unsafe {
        throw_runtime_error(
            stack_pointer,
            "RuntimeError",
            "handle/perform/resumable-try is temporarily disabled (Refactor A in progress)"
                .to_string(),
        );
    }
}

/// REFACTOR A stub. Pairs with `push_prompt_runtime`. See its docstring.
pub unsafe extern "C" fn pop_prompt_runtime(
    stack_pointer: usize,
    _frame_pointer: usize,
    _result_value: usize,
) -> usize {
    unsafe {
        throw_runtime_error(
            stack_pointer,
            "RuntimeError",
            "pop-prompt is temporarily disabled (Refactor A in progress)".to_string(),
        );
    }
}

/// Capture a continuation up to the nearest prompt.
/// Prompt-delimited execution runs on a prompt-owned segment, so capture
/// detaches that segment and records it by segment handle.
///
/// Arguments:
/// - stack_pointer: current SP
/// - frame_pointer: current FP
/// - resume_address: where to resume when continuation is invoked
/// - result_local_offset: offset where the invoked value should be stored
///
/// Returns: a tagged continuation pointer that can be invoked later
pub unsafe extern "C" fn capture_continuation_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    resume_address: usize,
    result_local_offset: isize,
) -> usize {
    unsafe {
        capture_continuation_runtime_inner(
            stack_pointer,
            frame_pointer,
            resume_address,
            result_local_offset,
        )
    }
}

/// REFACTOR A stub. `resume-tail` was a tail-position continuation invoker
/// used by the async implicit-handler runtime. Referenced by stdlib at
/// compile time; any runtime call aborts. Signature matches the v1
/// definition so install.rs registration stays valid.
pub unsafe extern "C" fn resume_tail_runtime(
    stack_pointer: usize,
    _frame_pointer: usize,
    _resume_closure: usize,
    _value: usize,
) -> ! {
    unsafe {
        throw_runtime_error(
            stack_pointer,
            "RuntimeError",
            "resume-tail is temporarily disabled (Refactor A in progress)".to_string(),
        );
    }
}

/// REFACTOR A stub. `invoke-continuation` was the direct (non-closure-wrapped)
/// continuation invoker. Referenced by stdlib at compile time; any runtime
/// call aborts.
pub unsafe extern "C" fn invoke_continuation_runtime(
    stack_pointer: usize,
    _frame_pointer: usize,
    _cont_ptr: usize,
    _value: usize,
) -> ! {
    unsafe {
        throw_runtime_error(
            stack_pointer,
            "RuntimeError",
            "invoke-continuation is temporarily disabled (Refactor A in progress)".to_string(),
        );
    }
}

pub unsafe extern "C" fn capture_continuation_runtime_with_saved_regs(
    stack_pointer: usize,
    frame_pointer: usize,
    resume_address: usize,
    result_local_offset: isize,
    _saved_regs_ptr: *const usize,
) -> usize {
    // saved_regs_ptr is no longer used — callee-saved register values are
    // stored in root slots by the codegen save loop and restored at the
    // resume point via ReloadRootSlots.
    unsafe {
        capture_continuation_runtime_inner(
            stack_pointer,
            frame_pointer,
            resume_address,
            result_local_offset,
        )
    }
}

/// Walks the FP chain from `from_fp` upward, looking for the topmost body
/// frame — the frame whose saved return address lies inside the code range
/// of `beagle.core/__reset__`. That frame is immediately above the reset
/// frame in the call chain; its FP is the last live frame of the reset
/// body, and its saved FP points to the reset frame itself.
///
/// Returns (topmost_body_fp, reset_fp) on success. Returns None if the walk
/// terminates (reaches the stack base, hits a 0 FP, or detects a cycle /
/// out-of-order chain) without finding a reset frame.
unsafe fn find_enclosing_reset_frame(from_fp: usize) -> Option<(usize, usize)> {
    let runtime = crate::get_runtime().get_mut();
    let mut fp = from_fp;
    // Defensive iteration cap — no realistic stack has this many frames.
    for _ in 0..100_000 {
        if fp == 0 {
            return None;
        }
        let saved_lr = unsafe { *((fp + 8) as *const usize) };
        if runtime.is_pc_in_reset_function(saved_lr) {
            let reset_fp = unsafe { *(fp as *const usize) };
            return Some((fp, reset_fp));
        }
        let saved_fp = unsafe { *(fp as *const usize) };
        if saved_fp == 0 || saved_fp <= fp {
            return None;
        }
        fp = saved_fp;
    }
    None
}

pub unsafe fn capture_continuation_runtime_inner(
    stack_pointer: usize,
    frame_pointer: usize,
    resume_address: usize,
    result_local_offset: isize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "capture_continuation");
    trace!(
        "continuation-detail",
        "capture_continuation: resume_addr={:#x} result_offset={}",
        resume_address,
        result_local_offset
    );

    // Walk the FP chain to find the prompt boundary — the outermost
    // captured body frame is the topmost frame whose saved return address
    // points into `__reset__`. That frame was called directly by
    // `__reset__` (i.e., body_thunk). The captured bytes run from the
    // current SP (innermost) up through the outermost body frame's saved
    // FP+LR pair.
    let (outermost_body_fp, _reset_fp) = unsafe { find_enclosing_reset_frame(frame_pointer) }
        .unwrap_or_else(|| {
            panic!(
                "capture_continuation: shift without an enclosing reset. Walked FP chain \
                 from fp={:#x} and found no __reset__ frame. This is a compiler bug — \
                 shift must be inside reset.",
                frame_pointer
            )
        });

    // Captured byte range: from the current SP up to and including the
    // saved FP+LR pair of the outermost body frame.
    let capture_top = outermost_body_fp + 16;
    let stack_size = capture_top.saturating_sub(stack_pointer);

    // `segment_frame_pointer_offset` is the offset from capture base to the
    // INNERMOST frame (the frame shift is running in). This is the canonical
    // "resume FP" — where execution picks up when the continuation is
    // invoked — and also where the saved-FP-chain relocation walk starts,
    // since the chain goes from innermost upward to outermost.
    //
    // `segment_gc_frame_offset` is reused here to store the OUTERMOST frame
    // offset. Invoke needs this to place the copy at exactly the right
    // destination so that the outermost frame's saved FP/LR slots overlay
    // the trampoline's own saved caller FP/LR slots — making normal body
    // return flow back to the invoker automatic with no slot patching.
    let innermost_fp_offset = frame_pointer - stack_pointer;
    let outermost_fp_offset = outermost_body_fp - stack_pointer;
    let segment_frame_pointer_offset = innermost_fp_offset;
    let segment_gc_frame_offset = outermost_fp_offset;

    let runtime = get_runtime().get_mut();

    // Synthetic "prompt" record for ContinuationObject::initialize. In the
    // new model these fields are not read by anything — prompt SP/FP/LR are
    // derived fresh from the FP walk at return_from_shift time — so we
    // leave them zero. The only live fields going forward are the segment
    // bytes, fp_offset, gc_frame_offset, size, resume_address, and
    // result_local.
    let synthetic_prompt = crate::runtime::PromptHandler {
        handler_address: 0,
        stack_pointer: 0,
        frame_pointer: 0,
        link_register: 0,
        result_local: 0,
        prompt_id: runtime
            .prompt_id_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
    };

    // Allocate the continuation object.
    let cont_ptr = match runtime.allocate(21, stack_pointer, BuiltInTypes::HeapObject) {
        Ok(ptr) => ptr,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate continuation object - out of memory".to_string(),
            );
        },
    };

    let mut cont_obj = HeapObject::from_tagged(cont_ptr);
    ContinuationObject::initialize(
        &mut cont_obj,
        resume_address,
        result_local_offset,
        &synthetic_prompt,
        BuiltInTypes::null_value() as usize,
        0,
        0,
        0,
    );

    // Root the continuation across the segment allocation.
    crate::runtime::per_thread_data()
        .saved_continuation_ptrs
        .push(cont_ptr);

    let segment_words = (stack_size + 7) / 8;
    let mut segment_data_base_at_capture = 0usize;
    let segment_heap_ptr = if segment_words > 0 {
        match runtime.allocate(segment_words, stack_pointer, BuiltInTypes::HeapObject) {
            Ok(ptr) => {
                let seg_obj = HeapObject::from_tagged(ptr);
                let header_ptr = seg_obj.untagged() as *mut usize;
                let mut header_val = unsafe { *header_ptr };
                header_val |= 0x2; // opaque bit
                unsafe {
                    *header_ptr = header_val;
                }

                let data_ptr = seg_obj.untagged() + seg_obj.header_size();
                segment_data_base_at_capture = data_ptr;

                // Copy live frames [stack_pointer, capture_top) into the heap
                // object. The live frames remain in place on the main stack;
                // this is a snapshot for later resume.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        stack_pointer as *const u8,
                        data_ptr as *mut u8,
                        stack_size,
                    );
                }

                // Relocate saved-FP links inside the copied bytes from their
                // original stack addresses to the heap object addresses.
                let fp_offset = segment_frame_pointer_offset;
                crate::runtime::relocate_segment_caller_fp_links(
                    data_ptr,
                    stack_size,
                    fp_offset,
                    stack_pointer,
                );
                let _ = crate::runtime::rebuild_segment_gc_prev_links_from_caller_chain(
                    data_ptr, stack_size, fp_offset, 0,
                );

                ptr
            }
            Err(_) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate captured segment - out of memory".to_string(),
                );
            },
        }
    } else {
        BuiltInTypes::null_value() as usize
    };

    // Re-read the cont pointer in case GC moved it during segment allocation.
    let cont_ptr = crate::runtime::per_thread_data()
        .saved_continuation_ptrs
        .last()
        .copied()
        .unwrap_or(cont_ptr);

    let mut cont = ContinuationObject::from_tagged(cont_ptr).unwrap();
    cont.set_segment_ptr_with_barrier(runtime, segment_heap_ptr);
    cont.set_segment_frame_pointer_offset(segment_frame_pointer_offset);
    cont.set_segment_gc_frame_offset(segment_gc_frame_offset);
    cont.set_segment_size(stack_size);
    if segment_heap_ptr != BuiltInTypes::null_value() as usize {
        cont.set_segment_original_data_base(segment_data_base_at_capture);
    }

    let cont_ptr = crate::runtime::per_thread_data()
        .saved_continuation_ptrs
        .last()
        .copied()
        .unwrap_or(cont_ptr);

    // Drop the temporary root.
    let _ = crate::runtime::per_thread_data()
        .saved_continuation_ptrs
        .pop();

    cont_ptr
}

pub unsafe fn return_from_shift_runtime_inner(
    _stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
    _cont_ptr: usize,
    _fallback_prompt: Option<crate::runtime::PromptHandler>,
) -> ! {
    print_call_builtin(get_runtime().get(), "return_from_shift");

    // Walk the FP chain to locate the enclosing `__reset__` frame. The
    // topmost body frame is the one whose saved LR points into `__reset__`;
    // __reset__'s own FP is topmost_body_fp's saved FP.
    let (topmost_body_fp, reset_fp) = unsafe { find_enclosing_reset_frame(frame_pointer) }
        .unwrap_or_else(|| {
            panic!(
                "return_from_shift: no enclosing reset found walking FP chain from fp={:#x}",
                frame_pointer
            )
        });

    // Derive the longjmp target: simulate `__reset__` returning normally to
    // its caller with `value` as the return register.
    //
    //   new_fp = __reset__'s saved caller FP       = [reset_fp + 0]
    //   new_ra = __reset__'s saved return address  = [reset_fp + 8]
    //   new_sp = post-__reset__-return SP          = reset_fp + 16
    //
    // Nothing is snapshotted in advance. These reads happen now because
    // __reset__'s frame is still live below shift body's execution until
    // the jump fires.
    let new_fp = unsafe { *(reset_fp as *const usize) };
    let new_ra = unsafe { *((reset_fp + 8) as *const usize) };
    let new_sp = reset_fp + 16;
    let _ = topmost_body_fp;

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            let runtime = get_runtime().get();
            let handler_jump_fn = runtime
                .get_function_by_name("beagle.builtin/handler-jump")
                .expect("handler-jump function not found");
            let ptr: *const u8 = handler_jump_fn.pointer.into();
            // x86_64 handler-jump signature currently only restores sp/fp/ra.
            // The return value needs to land in the calling convention's
            // return register. handler_jump is a 3-arg trampoline here; if
            // this path is exercised on x86_64 we'll need to extend it to
            // also set rax = value. For now this arch is ignored by the
            // active build.
            let _ = value;
            let handler_jump: extern "C" fn(usize, usize, usize) -> ! =
                unsafe { std::mem::transmute(ptr) };
            handler_jump(new_sp, new_fp, new_ra);
        } else {
            // ARM64: use return-jump trampoline. It takes (new_sp, new_fp,
            // new_lr, jump_target, callee_saved_ptr, value). `value` is
            // written into X0 before the branch, making it the return value
            // of `__reset__`.
            let runtime = get_runtime().get();
            let return_jump_fn = runtime
                .get_function_by_name("beagle.builtin/return-jump")
                .expect("return-jump function not found");
            let ptr: *const u8 = return_jump_fn.pointer.into();
            let return_jump: extern "C" fn(usize, usize, usize, usize, *const usize, usize) -> ! =
                unsafe { std::mem::transmute(ptr) };
            // We want control to land AT new_ra with sp/fp restored, as if
            // __reset__ had executed its epilogue. Pass new_ra as the jump
            // target. The new_lr arg is unused by this path (the handler
            // doesn't need to know its own caller's LR because it's just
            // resuming a normal return).
            return_jump(new_sp, new_fp, 0, new_ra, std::ptr::null(), value);
        }
    }
}

/// Return from shift body to the enclosing reset.
/// This pops the prompt and jumps to the prompt handler with the given value.
pub unsafe extern "C" fn return_from_shift_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
    cont_ptr: usize,
) -> ! {
    unsafe { return_from_shift_runtime_inner(stack_pointer, frame_pointer, value, cont_ptr, None) }
}

/// Return from shift body to the enclosing reset, specifically for handler returns.
/// This is called after `call-handler` in perform. It sets the is_handler_return flag
/// so that return_from_shift_runtime skips popping InvocationReturnPoints.
/// This prevents nested handlers from incorrectly consuming outer handler return points.
pub unsafe extern "C" fn return_from_shift_handler_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
    cont_ptr: usize,
) -> ! {
    if std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
        eprintln!(
            "[return_from_shift_handler] value={:#x} cont_ptr={:#x}",
            value, cont_ptr
        );
    }
    // SAFETY: We are in an unsafe function and caller guarantees valid stack/frame pointers
    // Set flag in a scoped block so the &mut borrow is dropped before calling
    // return_from_shift_runtime (which also accesses per-thread data).
    unsafe {
        {
            crate::runtime::per_thread_data().is_handler_return = true;
        }
        return_from_shift_runtime_inner(stack_pointer, frame_pointer, value, cont_ptr, None)
    }
}


// ============================================================================

/// Trampoline function for continuation closures.
/// When a continuation is captured, it's wrapped in a closure with this function as its body.
/// When called, it extracts the continuation pointer from the closure and invokes it.
///
/// Layout: closure_ptr points to a closure with:
/// - header (8 bytes)
/// - function pointer (8 bytes) - points to this trampoline
/// - cont_ptr (8 bytes) - the captured continuation heap object pointer (tagged)
///
/// Note: This is called as a regular closure body, so we receive (closure_ptr, value)
/// and need to get SP/FP ourselves.
/// Invoke a captured continuation.
///
/// Called as the body of a cont-closure: `k(v)` where `k` is the closure
/// returned by shift. The closure's single free variable is the
/// ContinuationObject; `value` is the value being plugged into the shift
/// point's result slot.
///
/// The design is:
/// 1. Compute a destination address `dst` on this trampoline's own stack
///    such that the outermost captured frame's FP slot lands exactly at
///    trampoline's own FP. The trampoline's prologue already wrote the
///    invoker's FP at [trampoline_fp + 0] and the invoker's LR at
///    [trampoline_fp + 8]. By overlaying the bottom frame on those slots,
///    the resumed body's eventual "return past bottom" unwinds directly
///    into the invoker — no return stubs, no prompt pushes, no side
///    tables.
/// 2. Copy the captured bytes (minus the bottom 16) from the heap segment
///    into [dst, dst + copy_size).
/// 3. Walk the copy and relocate saved-FP links from heap-addresses to
///    stack-addresses by a constant delta.
/// 4. Rebuild the GC prev chain in the copy so GC can walk it as normal
///    stack frames.
/// 5. Write `value` into the resume point's result slot.
/// 6. Use `return-jump` to set SP/FP and branch to resume_address.
///
/// Multi-shot is automatic: the heap segment's bytes are never mutated,
/// so each invocation copies fresh bytes to a fresh destination.
///
/// Safety: All writes to `dst` happen while Rust is mid-function. The
/// `dst` region overlaps Rust's own local area but Rust has already
/// packed every value it needs into local variables (which the compiler
/// keeps in registers for the remaining straight-line code). Between the
/// first dst write and the final `return_jump` call, no helper function
/// is invoked — so no frame gets pushed below Rust's SP into a region
/// that might overlap dst. Once `return_jump` runs, SP is set to `dst`
/// and control transfers to the resumed body; the old Rust frame is
/// abandoned.
#[allow(unused_variables)]
pub unsafe extern "C" fn continuation_trampoline(closure_ptr: usize, value: usize) -> ! {
    // Extract the continuation from the closure's free variables.
    // Closure layout: header(8) + fn_ptr(8) + num_free(8) + num_locals(8) + free_var[0]...
    let untagged_closure = BuiltInTypes::untag(closure_ptr);
    let cont_ptr = unsafe { *((untagged_closure + 32) as *const usize) };
    let cont = ContinuationObject::from_tagged(cont_ptr)
        .expect("continuation_trampoline: closure free var is not a ContinuationObject");

    // Normalize segment — this handles GC moves by re-relocating the
    // in-segment saved-FP chain against the current heap data base.
    let (seg_base, _seg_top, _innermost_fp_heap) = cont
        .segment_frame_info()
        .expect("continuation_trampoline: continuation has no segment data");
    let seg_size = cont.segment_size();
    let innermost_offset = cont.segment_frame_pointer_offset();
    // segment_gc_frame_offset is repurposed to store the outermost offset.
    let outermost_offset = cont.segment_gc_frame_offset();
    let resume_address = cont.resume_address();
    let result_local_offset = cont.result_local();

    // Read the trampoline's own FP via `read-fp` — a 2-instruction
    // Beagle-side JIT trampoline that returns x29. This function call
    // pushes a frame below trampoline's SP; that frame is popped before
    // we return here, so it doesn't touch any memory we care about.
    let trampoline_fp = {
        let runtime = get_runtime().get();
        let fn_entry = runtime
            .get_function_by_name("beagle.builtin/read-fp")
            .expect("read-fp trampoline not found");
        let read_fp: extern "C" fn() -> usize =
            unsafe { std::mem::transmute::<_, _>(fn_entry.pointer) };
        read_fp()
    };

    // Read return-jump's pointer here, before we start writing dst, so
    // that the final call doesn't need any function lookups.
    let return_jump_ptr: *const u8 = {
        let runtime = get_runtime().get();
        let fn_entry = runtime
            .get_function_by_name("beagle.builtin/return-jump")
            .expect("return-jump trampoline not found");
        fn_entry.pointer.into()
    };

    // Destination placement: outermost_fp_in_dst must equal trampoline_fp.
    //   outermost_fp_in_dst = dst + outermost_offset
    //   => dst = trampoline_fp - outermost_offset
    //
    // copy_size = outermost_offset. We do NOT copy the bottom 16 bytes
    // (the outermost frame's saved FP+LR pair). Those slots at
    // [trampoline_fp, trampoline_fp+16) already hold the invoker's FP
    // and LR (written by the trampoline's own prologue), which is exactly
    // what the outermost frame's saved FP/LR need to be.
    let dst = trampoline_fp - outermost_offset;
    let copy_size = outermost_offset;

    // ------------------------------------------------------------------
    // From here until the final `return_jump` call, NO function calls
    // may happen. All writes go to `dst` (which may overlap Rust's own
    // frame); a function call would push a frame below trampoline's SP,
    // into a region of `dst` we're actively writing.
    // ------------------------------------------------------------------

    // 1. Copy bytes: [seg_base, seg_base + copy_size) → [dst, dst + copy_size).
    let mut i = 0usize;
    while i < copy_size {
        let src_word = unsafe { *((seg_base + i) as *const usize) };
        unsafe { *((dst + i) as *mut usize) = src_word };
        i += 8;
    }

    // 2. Relocate saved-FP chain in dst. Walk from innermost upward via
    //    saved_fp, adjusting each by delta until a saved_fp falls outside
    //    the source range.
    let delta = dst.wrapping_sub(seg_base);
    let mut fp = dst + innermost_offset;
    loop {
        let saved_slot = fp as *mut usize;
        let saved = unsafe { *saved_slot };
        if saved < seg_base || saved >= seg_base + seg_size {
            break;
        }
        let relocated = saved.wrapping_add(delta);
        unsafe { *saved_slot = relocated };
        fp = relocated;
    }

    // 3. Rebuild GC prev chain in dst. For each frame header (at [fp-8]),
    //    write to [header-8] (the GC prev slot) the address of the parent
    //    frame's header, or 0 if the parent is outside the copy range
    //    (i.e., this is the outermost frame, which logically chains into
    //    the invoker — GC will pick that up naturally via GC_FRAME_TOP
    //    once we update it below).
    //
    //    Range check uses the header address (fp - 8), not fp, because the
    //    outermost frame's fp equals dst + copy_size and is therefore not
    //    strictly less than dst + copy_size. Its header at dst + copy_size
    //    - 8 IS within range.
    let copy_end = dst + copy_size;
    let mut fp = dst + innermost_offset;
    loop {
        let header_addr = fp.wrapping_sub(8);
        if header_addr < dst || header_addr >= copy_end {
            break;
        }
        let saved_fp = unsafe { *(fp as *const usize) };
        let parent_header = saved_fp.wrapping_sub(8);
        let parent_in_range = parent_header >= dst && parent_header < copy_end;
        let prev_val = if parent_in_range { parent_header } else { 0 };
        unsafe { *((header_addr - 8) as *mut usize) = prev_val };
        if !parent_in_range {
            break;
        }
        fp = saved_fp;
    }

    // 4. Write value to the resume point's result slot.
    let innermost_fp_in_dst = dst + innermost_offset;
    if result_local_offset != 0 {
        let result_ptr =
            (innermost_fp_in_dst as isize).wrapping_add(result_local_offset) as *mut usize;
        unsafe { *result_ptr = value };
    }

    // 5. Update GC_FRAME_TOP to the innermost frame's header in dst.
    //    This is a simple thread-local cell write — no function call.
    //    We cannot call set_gc_frame_top because that's a Rust function.
    //    The cell is accessed via a macro; we replicate its write here.
    GC_FRAME_TOP.with(|cell| cell.set(innermost_fp_in_dst - 8));

    // 6. Final jump via return-jump. Signature:
    //    return_jump(new_sp, new_fp, new_lr, jump_target, callee_saved_ptr, value)
    //    It sets sp/fp/lr/x0 and branches. noreturn.
    let return_jump: extern "C" fn(usize, usize, usize, usize, *const usize, usize) -> ! =
        unsafe { std::mem::transmute(return_jump_ptr) };
    return_jump(
        dst,
        innermost_fp_in_dst,
        0,
        resume_address,
        std::ptr::null(),
        0,
    );
}

