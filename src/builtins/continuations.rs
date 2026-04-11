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
pub unsafe extern "C" fn push_prompt_runtime(
    handler_address: usize,
    result_local: isize,
    link_register: usize,
    stack_pointer: usize,
    frame_pointer: usize,
) -> usize {
    print_call_builtin(get_runtime().get(), "push_prompt");
    let runtime = get_runtime().get_mut();

    // Generate a unique prompt ID to distinguish this handle block from others
    let prompt_id = runtime
        .prompt_id_counter
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    let handler = crate::runtime::PromptHandler {
        handler_address,
        stack_pointer,
        frame_pointer,
        link_register,
        result_local,
        prompt_id,
    };

    let segment = crate::runtime::per_thread_data().allocate_segment();
    let segment_top = segment.top & !0xF;
    {
        let ptd = crate::runtime::per_thread_data();
        ptd.push_active_segment(prompt_id, segment);
    }

    runtime.push_prompt_handler(handler);

    segment_top
}

/// Chez-style prompt push: records the prompt marker without allocating a
/// separate execution segment. The body runs on the caller's stack.
///
/// Stores handler_address (the post-body code label inside the enclosing
/// function), stack_pointer, and frame_pointer — these are the values to
/// restore when perform longjmps back to the post-body code path. Body
/// runs on the normal stack, so SP/FP at the post-body point equal the
/// values recorded here at push time.
#[allow(dead_code)]
pub unsafe extern "C" fn push_prompt_runtime_v2(
    handler_address: usize,
    _result_local: isize,
    _link_register: usize,
    stack_pointer: usize,
    frame_pointer: usize,
) -> usize {
    print_call_builtin(get_runtime().get(), "push_prompt_v2");
    let runtime = get_runtime().get_mut();

    let prompt_id = runtime
        .prompt_id_counter
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    let handler = crate::runtime::PromptHandler {
        handler_address,
        stack_pointer,
        frame_pointer,
        // Not used on the v2 path; the post-body code path is reached by
        // direct jump and the handler result flows through pending_perform
        // state rather than a dedicated result local.
        link_register: 0,
        result_local: 0,
        prompt_id,
    };

    runtime.push_prompt_handler(handler);

    // No SP switch. Return the caller's SP unchanged so the IR codegen's
    // stack-pointer reassignment becomes a no-op.
    stack_pointer
}

/// Chez-style perform: captures the body frames between the perform site
/// and the handle wrapper's frame, stores operation state in the
/// per-thread pending_perform slots, and longjmps back to the handle
/// call site as if the handle function returned normally.
///
/// The handle wrapper's IR will check pending_perform after its body call
/// returns and dispatch to the handler at that point — so this function
/// does not call the handler directly.
///
/// Multi-shot / resume is not yet implemented on this path.
#[allow(dead_code)]
pub unsafe extern "C" fn perform_effect_runtime_v2(
    stack_pointer: usize,
    frame_pointer: usize,
    enum_type_ptr: usize,
    op_value: usize,
    resume_address: usize,
    result_local_offset_raw: usize,
    _saved_regs_ptr: *const usize,
) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);
    let result_local_offset = result_local_offset_raw as isize;

    // Find the topmost prompt handler. Do NOT pop it — the prompt stays
    // active so that when the handler calls resume and the body re-enters
    // the prompt scope, subsequent performs can still find it. The prompt
    // is popped by pop_prompt_runtime_v2 when the handle wrapper exits.
    let prompt = {
        let ptd = crate::runtime::per_thread_data();
        ptd.prompt_handlers.last().cloned().unwrap_or_else(|| {
            panic!("perform_effect_runtime_v2 called without an enclosing handle block")
        })
    };

    // Root op_value and enum_type_ptr across allocations.
    let runtime = get_runtime().get_mut();
    let op_root = if BuiltInTypes::is_heap_pointer(op_value) {
        Some(runtime.register_temporary_root(op_value))
    } else {
        None
    };
    let enum_type_root = if BuiltInTypes::is_heap_pointer(enum_type_ptr) {
        Some(runtime.register_temporary_root(enum_type_ptr))
    } else {
        None
    };

    // Walk the FP chain to find the topmost body frame — the one whose
    // saved caller FP equals the prompt's handle_fp. Body frames are the
    // frames that will be captured into the continuation segment.
    let handle_fp = prompt.frame_pointer;
    let mut fp = frame_pointer;
    let topmost_body_fp = loop {
        if fp == 0 || fp >= handle_fp {
            panic!(
                "perform_effect_runtime_v2: walked past handle_fp ({:#x}) without finding it (fp={:#x})",
                handle_fp, fp
            );
        }
        let saved_fp = unsafe { *(fp as *const usize) };
        if saved_fp == handle_fp {
            break fp;
        }
        if saved_fp <= fp {
            panic!(
                "perform_effect_runtime_v2: corrupt FP chain at {:#x} -> {:#x}",
                fp, saved_fp
            );
        }
        fp = saved_fp;
    };

    // Captured byte range: [stack_pointer, topmost_body_fp + 16).
    // +16 includes the saved FP+LR pair of the topmost body frame.
    let segment_size = topmost_body_fp + 16 - stack_pointer;
    let segment_fp_offset = topmost_body_fp - stack_pointer;
    let prompt_sp = prompt.stack_pointer;

    // Allocate the opaque-bytes heap object for the captured segment.
    let seg_tagged: usize =
        match runtime.allocate_opaque_bytes_from_bytes(prompt_sp, &vec![0u8; segment_size]) {
            Ok(ptr) => ptr.into(),
            Err(_) => unsafe {
                throw_runtime_error(
                    prompt_sp,
                    "AllocationError",
                    "Failed to allocate continuation segment".to_string(),
                );
            },
        };

    // Copy the body frame bytes into the segment.
    let seg_obj = HeapObject::from_tagged(seg_tagged);
    let data_ptr = seg_obj.untagged() + seg_obj.header_size();
    unsafe {
        std::ptr::copy_nonoverlapping(
            stack_pointer as *const u8,
            data_ptr as *mut u8,
            segment_size,
        );
    }

    // Root the segment across the continuation allocation.
    let seg_root = runtime.register_temporary_root(seg_tagged);

    // Allocate the continuation object.
    let cont_ptr = match runtime.allocate(21, prompt_sp, BuiltInTypes::HeapObject) {
        Ok(ptr) => ptr,
        Err(_) => unsafe {
            throw_runtime_error(
                prompt_sp,
                "AllocationError",
                "Failed to allocate continuation object".to_string(),
            );
        },
    };
    let seg_tagged = runtime.peek_temporary_root(seg_root);
    runtime.unregister_temporary_root(seg_root);

    let mut cont_obj = HeapObject::from_tagged(cont_ptr);
    ContinuationObject::initialize(
        &mut cont_obj,
        resume_address,
        result_local_offset,
        &prompt,
        seg_tagged,
        segment_fp_offset,
        0, // segment_gc_frame_offset — not used on v2 path
        segment_size,
    );

    // Build a resume closure wrapping the continuation.
    let cont_root = runtime.register_temporary_root(cont_ptr);
    let trampoline_fn = runtime
        .get_function_by_name("beagle.builtin/continuation-trampoline-v2")
        .expect("continuation-trampoline-v2 not found");
    let tagged_trampoline =
        BuiltInTypes::Function.tag(usize::from(trampoline_fn.pointer) as isize) as usize;
    let cont_ptr_now = runtime.peek_temporary_root(cont_root);
    let resume_closure = match runtime.make_closure(prompt_sp, tagged_trampoline, &[cont_ptr_now]) {
        Ok(closure) => closure,
        Err(_) => unsafe {
            throw_runtime_error(
                prompt_sp,
                "AllocationError",
                "Failed to allocate resume closure".to_string(),
            );
        },
    };
    runtime.unregister_temporary_root(cont_root);

    // Refresh op_value and enum_type_ptr after allocations.
    let op_value = op_root
        .map(|r| runtime.peek_temporary_root(r))
        .unwrap_or(op_value);
    let enum_type_ptr = enum_type_root
        .map(|r| runtime.peek_temporary_root(r))
        .unwrap_or(enum_type_ptr);
    if let Some(r) = op_root {
        runtime.unregister_temporary_root(r);
    }
    if let Some(r) = enum_type_root {
        runtime.unregister_temporary_root(r);
    }

    // Record pending perform state; handle_completed_runtime will pick it
    // up after the longjmp returns control to the handle wrapper.
    {
        let ptd = crate::runtime::per_thread_data();
        ptd.set_pending_perform(op_value, resume_closure, enum_type_ptr);
    }

    // Longjmp back to the post-body code path in the enclosing function.
    // prompt.stack_pointer/frame_pointer are the values SP/FP had when
    // push_prompt was called — since body runs on the normal stack as a
    // regular function call, these are also the correct values for the
    // post-body code path (SP/FP are preserved across function call and
    // return). prompt.handler_address is the post-body code label.
    let runtime = get_runtime().get();
    let return_jump_fn = runtime
        .get_function_by_name("beagle.builtin/return-jump")
        .expect("return-jump trampoline not found");
    let ptr: *const u8 = return_jump_fn.pointer.into();

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            // x86-64 return-jump signature:
            //   (new_sp, new_fp, value, return_address, callee_saved_ptr, frame_src, frame_size)
            let return_jump: extern "C" fn(
                usize, usize, usize, usize, *const usize, *const u8, usize,
            ) -> ! = unsafe { std::mem::transmute(ptr) };
            return_jump(
                prompt.stack_pointer,
                prompt.frame_pointer,
                0,
                prompt.handler_address,
                std::ptr::null(),
                std::ptr::null(),
                0,
            );
        } else {
            // ARM64 return-jump signature:
            //   (new_sp, new_fp, new_lr, jump_target, callee_saved_ptr, value)
            let return_jump: extern "C" fn(
                usize, usize, usize, usize, *const usize, usize,
            ) -> ! = unsafe { std::mem::transmute(ptr) };
            return_jump(
                prompt.stack_pointer,
                prompt.frame_pointer,
                0,
                prompt.handler_address,
                std::ptr::null(),
                0,
            );
        }
    }
}

/// Chez-style continuation resume: called as a regular closure body from
/// the handler when it invokes `resume(v)`. Does not return — instead
/// copies the captured frames back onto the stack, writes `value` into
/// the resume point's result slot, and jumps to the resume address.
///
/// The body frames are copied at [prompt_sp - segment_size, prompt_sp),
/// directly above the handle site. When the body eventually completes,
/// it returns normally via the stack frame chain back to the handle
/// wrapper's post-body code in the enclosing function — bypassing the
/// handler's stack entirely (tail-call semantics for resume).
///
/// Multi-shot where the handler calls `resume` non-tail (then continues
/// execution after the resume call) is NOT supported on this path — the
/// handler's stack frames are implicitly destroyed when SP is moved.
#[allow(dead_code)]
pub unsafe extern "C" fn continuation_trampoline_v2(closure_ptr: usize, value: usize) -> ! {
    // Extract cont_ptr from closure free variables (layout matches
    // continuation_trampoline: header + fn_ptr + cont_ptr at offset 32).
    let untagged_closure = BuiltInTypes::untag(closure_ptr);
    let cont_ptr = unsafe { *((untagged_closure + 32) as *const usize) };

    let cont = ContinuationObject::from_tagged(cont_ptr)
        .expect("continuation_trampoline_v2: invalid continuation pointer");
    let segment_ptr = cont.segment_ptr();
    let segment_size = cont.segment_size();
    let segment_fp_offset = cont.segment_frame_pointer_offset();
    let resume_address = cont.resume_address();
    let result_local_offset = cont.result_local();
    let prompt_sp = cont.prompt_stack_pointer();

    // Compute placement: body frames go at [prompt_sp - segment_size, prompt_sp).
    // Delegate to the resume-jump-v2 assembly trampoline so the copy loop
    // runs in registers only — the copy may overwrite the memory we're
    // currently executing in, which is only safe to do from code that
    // makes no stack accesses.
    let new_sp = prompt_sp - segment_size;
    let new_fp = new_sp + segment_fp_offset;

    let seg_obj = HeapObject::from_tagged(segment_ptr);
    let src_ptr = seg_obj.untagged() + seg_obj.header_size();
    let dst_ptr = new_sp;

    let runtime = get_runtime().get();
    let resume_jump_fn = runtime
        .get_function_by_name("beagle.builtin/resume-jump-v2")
        .expect("resume-jump-v2 trampoline not found");
    let ptr: *const u8 = resume_jump_fn.pointer.into();

    // Signature: (src, dst, size, new_sp, new_fp, jump_target, value, result_local_offset)
    let resume_jump: extern "C" fn(usize, usize, usize, usize, usize, usize, usize, isize) -> ! =
        unsafe { std::mem::transmute(ptr) };
    resume_jump(
        src_ptr,
        dst_ptr,
        segment_size,
        new_sp,
        new_fp,
        resume_address,
        value,
        result_local_offset,
    );
}

/// Chez-style prompt pop: removes the prompt marker without segment
/// bookkeeping. Returns the caller's SP unchanged so the IR codegen's
/// stack-pointer reassignment becomes a no-op.
#[allow(dead_code)]
pub unsafe extern "C" fn pop_prompt_runtime_v2(
    stack_pointer: usize,
    _frame_pointer: usize,
    _result_value: usize,
) -> usize {
    print_call_builtin(get_runtime().get(), "pop_prompt_v2");
    let runtime = get_runtime().get_mut();
    let _ = runtime.pop_prompt_handler();
    stack_pointer
}

/// The return context of a handle wrapper's frame, as seen by a perform
/// operation that wants to simulate a normal return from the handle call.
///
/// This is read directly from the handle wrapper's stack frame — no fields
/// need to be snapshotted in the PromptHandler itself.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct HandleReturnContext {
    /// The caller's FP, to restore on return. Read from [handle_fp + 0].
    pub saved_caller_fp: usize,
    /// Where to jump to on return. Read from [handle_fp + 8].
    pub return_address: usize,
    /// The SP the handle's caller had before calling handle. Computed as
    /// `handle_fp + 16` — this is the position immediately above the saved
    /// FP+LR pair, which is where SP was when the handle call's prelude
    /// executed `stp x29, x30, [sp, #-16]!`.
    pub sp_after_return: usize,
}

/// Read the return context from a handle wrapper's stack frame.
///
/// Given the FP of a handle wrapper function (the function that called
/// push_prompt), extract the SP/FP/return-address values needed to longjmp
/// back to the handle's caller as if the handle function returned normally.
///
/// Frame layout (standard AArch64):
///   [FP+8]  : saved return address
///   [FP+0]  : saved caller FP
///   [FP-8]  : frame header
///   ...
///
/// # Safety
/// `handle_fp` must point to a live stack frame that was set up by Beagle's
/// function prelude. The caller is responsible for ensuring the frame has
/// not been unwound or corrupted.
#[allow(dead_code)]
pub unsafe fn read_handle_return_context(handle_fp: usize) -> HandleReturnContext {
    let saved_caller_fp = unsafe { *(handle_fp as *const usize) };
    let return_address = unsafe { *((handle_fp + 8) as *const usize) };
    let sp_after_return = handle_fp + 16;
    HandleReturnContext {
        saved_caller_fp,
        return_address,
        sp_after_return,
    }
}

/// Pop the current prompt handler.
/// If there's an invocation return point (continuation was invoked), returns via
/// return_from_shift_runtime to enable multi-shot continuations.
pub unsafe extern "C" fn pop_prompt_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    result_value: usize,
) -> usize {
    print_call_builtin(get_runtime().get(), "pop_prompt");
    let runtime = get_runtime().get_mut();
    let debug_prompts = runtime.get_command_line_args().debug;

    // Read values from per-thread data in a scoped block to avoid holding
    // a &mut reference across the call to return_from_shift_runtime (which
    // also accesses per-thread data — overlapping &mut would be UB).
    let (current_prompt_id, current_prompt_sp, should_route_to_return_from_shift) = {
        let ptd = crate::runtime::per_thread_data();

        // Get the current prompt's ID BEFORE popping - this tells us which handle block is completing
        let current_prompt_id = ptd.prompt_handlers.last().map(|h| h.prompt_id);
        let current_prompt_sp = ptd.prompt_handlers.last().map(|h| h.stack_pointer);

        if debug_prompts {
            let rp_len = ptd.invocation_return_points.len();
            let top_rp = ptd.invocation_return_points.last().map(|rp| {
                (
                    rp.stack_pointer,
                    rp.frame_pointer,
                    rp.return_address,
                    rp.prompt_id,
                )
            });
            eprintln!(
                "[pop_prompt] current_prompt_id={:?} return_points={} top={:?} sp={:#x} fp={:#x}",
                current_prompt_id, rp_len, top_rp, stack_pointer, frame_pointer
            );
        }

        // Check if there's an invocation return point for THIS handle block.
        //
        // For empty segments: the prompt was already popped by capture_continuation (perform/shift),
        // and NOT re-pushed by invoke_continuation. So current_prompt_id will be None.
        // But we still have an InvocationReturnPoint that we need to route through.
        //
        // For non-empty segments: the prompt was popped by capture_continuation, then RE-PUSHED
        // by invoke_continuation at the relocated location. So current_prompt_id will match.
        //
        // Strategy: Check for return points FIRST, regardless of current_prompt_id.
        // Match on the prompt_id stored in the return point itself.
        let should_route = if let Some(top_point) = ptd.invocation_return_points.last() {
            current_prompt_id.is_none() || current_prompt_id == Some(top_point.prompt_id)
        } else {
            false
        };

        if should_route {
            ptd.return_from_shift_via_pop_prompt = true;
        }

        (current_prompt_id, current_prompt_sp, should_route)
    };
    // ptd borrow is dropped here

    if should_route_to_return_from_shift {
        unsafe {
            return_from_shift_runtime(
                stack_pointer,
                frame_pointer,
                result_value,
                BuiltInTypes::null_value() as usize,
            )
        };
    }

    // Normal path - no matching invocation return point, just pop and return
    let ptd = crate::runtime::per_thread_data();
    if !ptd.prompt_handlers.is_empty() {
        ptd.prompt_handlers.pop();
    }

    // Clear invocation return points that belong to THIS handle block.
    if let Some(prompt_id) = current_prompt_id {
        if let Some(segment) = ptd.pop_active_segment(prompt_id) {
            ptd.recycle_segment(segment);
        }
        ptd.invocation_return_points
            .retain(|rp| rp.prompt_id != prompt_id);
    }

    // When all prompts are done, clear ALL continuation state for this thread.
    // saved_continuation_ptr only exists to bridge a single perform/handler-return
    // when the stack local holding cont_ptr becomes stale after relocation.
    // Once the prompt stack is empty, any leftover pointer belongs to an already
    // completed invocation and must not leak into the next one.
    if ptd.prompt_handlers.is_empty() {
        ptd.invocation_return_points.clear();
        ptd.return_from_shift_via_pop_prompt = false;
        ptd.is_handler_return = false;
        ptd.clear_all_saved_continuations();
    }

    current_prompt_sp.unwrap_or(stack_pointer)
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

    // Walk the FP chain to find the prompt boundary — the innermost
    // `beagle.core/__reset__` frame. `topmost_body_fp` is the topmost frame
    // whose saved return address points into `__reset__`; it is the last
    // frame of the reset body that shift is capturing. `reset_fp` is
    // `__reset__`'s own FP.
    let (topmost_body_fp, _reset_fp) = unsafe { find_enclosing_reset_frame(frame_pointer) }
        .unwrap_or_else(|| {
            panic!(
                "capture_continuation: shift without an enclosing reset. Walked FP chain \
                 from fp={:#x} and found no __reset__ frame. This is a compiler bug — \
                 shift must be inside reset.",
                frame_pointer
            )
        });

    // Captured byte range: from the current SP up to and including the
    // saved FP+LR pair of the topmost body frame. That saved pair is what
    // returns "into __reset__" in the original execution — copying it
    // preserves the original chain, and invocation will rewrite it to
    // return to the invoker instead.
    let capture_top = topmost_body_fp + 16;
    let stack_size = capture_top.saturating_sub(stack_pointer);
    let segment_frame_pointer_offset = topmost_body_fp - stack_pointer;
    let segment_gc_frame_offset = (topmost_body_fp - 8) - stack_pointer;

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

pub unsafe extern "C" fn continue_return_from_shift_on_safe_stack() -> ! {
    let (mut value, return_point) = {
        let ptd = crate::runtime::per_thread_data();
        let ctx = ptd
            .safe_return_context
            .as_ref()
            .expect("missing safe return context for continuation return")
            .clone();
        (ctx.value, ctx.return_point)
    };

    value = unsafe {
        let ptd = crate::runtime::per_thread_data();
        ptd.safe_return_context
            .as_ref()
            .map(|ctx| std::ptr::read_volatile(&ctx.value))
            .unwrap_or(value)
    };

    let new_sp = return_point.stack_pointer;
    let new_fp = return_point.frame_pointer;
    let return_address = return_point.return_address;

    let suspended_frame = {
        let ptd = crate::runtime::per_thread_data();
        ptd.take_suspended_frame(return_point.suspended_frame_id)
    };
    let restored_frame = suspended_frame.is_some();

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
                let caller_frame_is_live = !restored_frame && frame_pointer_in_active_segment(new_fp);
                if let Some(suspended_frame) = suspended_frame.as_ref() {
                    let frame_size = suspended_frame.frame_size;
                    let frame_bottom = new_fp
                        .checked_add(16)
                        .and_then(|top| top.checked_sub(frame_size))
                        .expect("safe return frame restore underflow computing frame_bottom");
                    let gc_chain_prev =
                        gc_chain_prev_for_restored_segment(frame_bottom, frame_size, new_fp);
                    suspended_frame.restore_to_stack(new_fp, gc_chain_prev);
                }
                if restored_frame || caller_frame_is_live {
                    crate::builtins::set_gc_frame_top(new_fp.saturating_sub(8));
                } else {
                    crate::builtins::set_gc_frame_top(return_point.saved_gc_prev);
                }

            let runtime = get_runtime().get();
            let return_jump_fn = runtime
                .get_function_by_name("beagle.builtin/return-jump")
                .expect("return-jump function not found");
            let ptr: *const u8 = return_jump_fn.pointer.into();
            let return_jump_ptr: extern "C" fn(usize, usize, usize, usize, *const usize, *const u8, usize) -> ! =
                unsafe { std::mem::transmute(ptr) };
            crate::runtime::per_thread_data().safe_return_context = None;
            // Pass null for frame_src since we restored the suspended caller frame directly.
            return_jump_ptr(new_sp, new_fp, value, return_address, std::ptr::null(), std::ptr::null(), 0);
        } else {
                let caller_frame_is_live = !restored_frame && frame_pointer_in_active_segment(new_fp);
                if let Some(suspended_frame) = suspended_frame.as_ref() {
                    let frame_size = suspended_frame.frame_size;
                    let frame_bottom = new_fp
                        .checked_add(16)
                        .and_then(|top| top.checked_sub(frame_size))
                        .expect("safe return frame restore underflow computing frame_bottom");
                    let gc_chain_prev =
                        gc_chain_prev_for_restored_segment(frame_bottom, frame_size, new_fp);
                    suspended_frame.restore_to_stack(new_fp, gc_chain_prev);
                }
                if restored_frame || caller_frame_is_live {
                    crate::builtins::set_gc_frame_top(new_fp.saturating_sub(8));
                } else {
                    crate::builtins::set_gc_frame_top(return_point.saved_gc_prev);
                }

            crate::runtime::per_thread_data().safe_return_context = None;
            let runtime = get_runtime().get();
            let return_jump_fn = runtime
                .get_function_by_name("beagle.builtin/return-jump")
                .expect("return-jump function not found");
            let ptr: *const u8 = return_jump_fn.pointer.into();
            let return_jump: extern "C" fn(usize, usize, usize, usize, *const usize, usize) -> ! =
                unsafe { std::mem::transmute(ptr) };
            return_jump(new_sp, new_fp, 0, return_address, std::ptr::null(), value);
        }
    }
}

pub unsafe extern "C" fn continue_perform_on_safe_stack() -> ! {
    let ctx = {
        let ptd = crate::runtime::per_thread_data();
        ptd.safe_perform_context
            .take()
            .expect("missing safe perform context")
    };

    let handler = ctx.handler;
    let op_value = ctx.op_value;
    let resume_closure = ctx.resume_closure;
    let cont_ptr = ctx.cont_ptr;
    let fn_ptr = ctx.fn_ptr;
    let prompt_handler = ctx.prompt_handler;

    let handler_result = unsafe {
        call_beagle_fn_ptr3(
            get_runtime().get(),
            fn_ptr,
            handler,
            op_value,
            resume_closure,
        )
    };

    if std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
        eprintln!(
            "[perform_safe] handler returned result={:#x} cont_ptr={:#x}",
            handler_result, cont_ptr
        );
    }

    crate::runtime::per_thread_data().pop_native_perform_stack();

    let (stack_pointer, frame_pointer) = {
        let runtime = get_runtime().get();
        let fn_entry = runtime
            .get_function_by_name("beagle.builtin/read-sp-fp")
            .expect("read-sp-fp trampoline not found");
        let read_sp_fp: extern "C" fn() -> (usize, usize) =
            unsafe { std::mem::transmute::<_, _>(fn_entry.pointer) };
        read_sp_fp()
    };

    let resolved_cont_ptr = cont_ptr;

    unsafe {
        crate::runtime::per_thread_data().is_handler_return = true;
        return_from_shift_runtime_inner(
            stack_pointer,
            frame_pointer,
            handler_result,
            resolved_cont_ptr,
            Some(prompt_handler),
        )
    }
}

pub unsafe fn jump_to_safe_perform_stack() -> ! {
    let stack_top = {
        let ptd = crate::runtime::per_thread_data();
        let _ = ptd
            .safe_perform_context
            .as_ref()
            .expect("missing safe perform context");
        ptd.push_native_perform_stack()
    };

    let runtime = get_runtime().get();
    let switch_fn = runtime
        .get_function_by_name("beagle.builtin/stack-switch")
        .expect("stack-switch function not found");
    let ptr: *const u8 = switch_fn.pointer.into();
    let stack_switch: extern "C" fn(usize, usize) -> ! = unsafe { std::mem::transmute(ptr) };
    stack_switch(stack_top, continue_perform_on_safe_stack as usize);
}

pub unsafe fn jump_to_safe_return_stack() -> ! {
    let stack_top = {
        let ptd = crate::runtime::per_thread_data();
        ptd.ensure_native_scratch_stack_top()
    };

    let runtime = get_runtime().get();
    let switch_fn = runtime
        .get_function_by_name("beagle.builtin/stack-switch")
        .expect("stack-switch function not found");
    let ptr: *const u8 = switch_fn.pointer.into();
    let stack_switch: extern "C" fn(usize, usize) -> ! = unsafe { std::mem::transmute(ptr) };
    stack_switch(stack_top, continue_return_from_shift_on_safe_stack as usize);
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

pub fn invoke_segmented_continuation(
    continuation: &ContinuationObject,
    value: usize,
    debug_prompts: bool,
    return_point: crate::runtime::InvocationReturnPoint,
    push_return_point: bool,
) -> ! {
    let prompt_id = continuation.prompt_id();
    let mut resume_address = continuation.resume_address();
    let continuation_return_address = {
        let runtime = get_runtime().get();
        let return_stub = runtime
            .get_function_by_name("beagle.builtin/continuation-return-stub")
            .expect("continuation-return-stub function not found");
        let return_stub_ptr: usize = return_stub.pointer.into();
        return_stub_ptr
    };
    // Read the captured segment data from the heap-allocated opaque bytes object.
    let seg_tagged = continuation.segment_ptr();
    let seg_size = continuation.segment_size();
    let seg_fp_offset = continuation.segment_frame_pointer_offset();
    let seg_gc_offset = continuation.segment_gc_frame_offset();
    let seg_original_base = continuation.segment_original_data_base();
    if seg_tagged == 0 || seg_tagged == BuiltInTypes::null_value() as usize || seg_size == 0 {
        panic!(
            "invoke_segmented_continuation: continuation has no captured segment data (ptr={:#x} size={} fp_offset={} gc_offset={} original_base={:#x})",
            seg_tagged, seg_size, seg_fp_offset, seg_gc_offset, seg_original_base
        );
    }

    // segment_frame_info() normalizes the captured segment after any GC move by
    // relocating the saved caller-FP chain to the current data base.
    let (seg_data_base, _seg_data_top, seg_innermost_fp) =
        continuation.segment_frame_info().unwrap_or_else(|| {
        panic!(
            "invoke_segmented_continuation: continuation has no normalized segment data (ptr={:#x} size={} fp_offset={} gc_offset={} original_base={:#x})",
            seg_tagged, seg_size, seg_fp_offset, seg_gc_offset, seg_original_base
        );
    });

    if debug_prompts || std::env::var("BEAGLE_DEBUG_INVOKE").is_ok() {
        eprintln!(
            "[invoke_seg_cont] seg_tagged={:#x} seg_size={} gc_offset={} seg_data_base={:#x} resume={:#x} prompt_id={}",
            seg_tagged,
            seg_size,
            continuation.segment_gc_frame_offset(),
            seg_data_base,
            continuation.resume_address(),
            prompt_id,
        );
    }
    // Allocate an execution segment (mmap) to copy the frames into.
    // The resumed code will execute on this segment (RSP points into it).
    let cloned_segment = crate::runtime::per_thread_data().allocate_segment();

    // The captured frames are stored at [seg_data_base..seg_data_base+seg_size].
    // In the heap object, interior pointers were relocated relative to seg_data_base.
    // We need to copy them into the execution segment and relocate to the segment's base.
    //
    // The original capture stored frames starting at original_sp. In the heap object,
    // the data starts at offset 0. We place the frames at the TOP of the execution
    // segment (since stacks grow downward).
    // Place frames at the TOP of the execution segment, matching the original
    // layout where frames grew downward from segment_used_top.
    let exec_top = cloned_segment.top & !0xF; // same alignment as push_prompt_runtime
    let exec_base = exec_top - seg_size;

    unsafe {
        std::ptr::copy_nonoverlapping(seg_data_base as *const u8, exec_base as *mut u8, seg_size);
    }

    let sp_offset = 0; // SP was at the start of captured data
    let fp_offset = if seg_innermost_fp >= seg_data_base
        && seg_innermost_fp < seg_data_base + seg_size
    {
        seg_innermost_fp - seg_data_base
    } else {
        eprintln!(
            "[invoke_seg_cont] WARNING: innermost_fp {:#x} outside captured data [{:#x}..{:#x}]",
            seg_innermost_fp,
            seg_data_base,
            seg_data_base + seg_size
        );
        panic!("invoke_segmented_continuation: invalid frame pointer offset for captured segment");
    };

    let new_sp = exec_base + sp_offset;
    let new_fp = exec_base + fp_offset;
    let gc_offset = continuation.segment_gc_frame_offset();

    crate::runtime::relocate_segment_caller_fp_links(exec_base, seg_size, fp_offset, seg_data_base);

    // The prompt frame stays live while the detached continuation is suspended.
    // Replaying its snapshot here clobbers the resumer's live state and can
    // reintroduce stale prompt-local data. Keep the snapshot for now, but do
    // not restore it during invoke.

    if std::env::var("BEAGLE_DEBUG_RESUME").is_ok() {
        let runtime = get_runtime().get();
        let resume_address = continuation.resume_address() as *const u8;
        if let Some((function, offset)) = runtime.get_function_containing_pointer(resume_address) {
            let function_start: usize = function.pointer.into();
            eprintln!(
                "[resume] function={} start={:#x} offset={:#x} resume={:#x} source={:?}:{:?} locals={} size={:#x}",
                function.name,
                function_start,
                offset,
                continuation.resume_address(),
                function.source_file,
                function.source_line,
                function.number_of_locals,
                function.size,
            );
            let dump_start = continuation.resume_address().saturating_sub(0x20);
            let mut words = Vec::new();
            for i in 0..32usize {
                let addr = dump_start + i * 4;
                let word = unsafe { *(addr as *const u32) };
                words.push(format!("{:#010x}", word));
            }
            eprintln!(
                "[resume-code] start={:#x} words={}",
                dump_start,
                words.join(" ")
            );
        } else {
            eprintln!(
                "[resume] unknown resume={:#x}",
                continuation.resume_address(),
            );
        }

        let frame_words = [
            unsafe { *((new_fp - 40) as *const usize) },
            unsafe { *((new_fp - 32) as *const usize) },
            unsafe { *((new_fp - 24) as *const usize) },
            unsafe { *((new_fp - 16) as *const usize) },
            unsafe { *((new_fp - 8) as *const usize) },
            unsafe { *(new_fp as *const usize) },
            unsafe { *((new_fp + 8) as *const usize) },
        ];
        eprintln!(
            "[resume-frame] fp={:#x} slots[-40..+8]={:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}",
            new_fp,
            frame_words[0],
            frame_words[1],
            frame_words[2],
            frame_words[3],
            frame_words[4],
            frame_words[5],
            frame_words[6],
        );
        if BuiltInTypes::is_heap_pointer(frame_words[2]) {
            let env_base = BuiltInTypes::untag(frame_words[2]);
            let mut env_words = Vec::new();
            for i in 0..6usize {
                let addr = env_base + i * 8;
                let word = unsafe { *(addr as *const usize) };
                env_words.push(format!("{:#x}:{:#x}", addr, word));
            }
            let env_kind = BuiltInTypes::get_kind(frame_words[2]);
            eprintln!(
                "[resume-env] slot={:#x} kind={:?} base={:#x} words={}",
                frame_words[2],
                env_kind,
                env_base,
                env_words.join(" ")
            );
            if let Some(heap_obj) = HeapObject::try_from_tagged(frame_words[2]) {
                let mut fields = Vec::new();
                for i in 0..6usize {
                    fields.push(format!("{:#x}", heap_obj.get_field(i)));
                }
                eprintln!("[resume-env-fields] {}", fields.join(" "));
            }
        }
        let mut sp_words = Vec::new();
        for i in 0..20usize {
            let addr = new_sp + i * 8;
            let word = unsafe { *(addr as *const usize) };
            sp_words.push(format!("{:#x}:{:#x}", addr, word));
        }
        eprintln!(
            "[resume-stack] sp={:#x} words={}",
            new_sp,
            sp_words.join(" ")
        );
        let result_local = continuation.result_local();
        let result_addr = (new_fp as isize).wrapping_add(result_local) as usize;
        let result_word = unsafe { *(result_addr as *const usize) };
        eprintln!(
            "[resume-result-slot] offset={} addr={:#x} value={:#x} resume_value={:#x}",
            result_local, result_addr, result_word, value
        );

        let current_return = unsafe { *((new_fp + 8) as *const usize) };
        if let Some((function, offset)) =
            runtime.get_function_containing_pointer(current_return as *const u8)
        {
            eprintln!(
                "[resume-caller] current-frame-caller={} start={:#x} offset={:#x} return={:#x}",
                function.name,
                {
                    let function_start: usize = function.pointer.into();
                    function_start
                },
                offset,
                current_return
            );
            let dump_start = current_return.saturating_sub(0x20);
            let mut words = Vec::new();
            let mut raw_words = Vec::new();
            for i in 0..48usize {
                let addr = dump_start + i * 4;
                let word = unsafe { *(addr as *const u32) };
                raw_words.push(word);
                words.push(format!("{:#010x}", word));
            }
            eprintln!(
                "[resume-caller-code] start={:#x} words={}",
                dump_start,
                words.join(" ")
            );
            if let Some(cell_addr) = decode_arm_mov_imm3(&raw_words[0..3], 27) {
                let cell_value = unsafe { *(cell_addr as *const usize) };
                let mapped = ContinuationObject::from_tagged(cell_value)
                    .map(|_| "continuation".to_string())
                    .or_else(|| {
                        if BuiltInTypes::get_kind(cell_value) == BuiltInTypes::Function {
                            runtime
                                .get_function_by_pointer(
                                    BuiltInTypes::untag(cell_value) as *const u8
                                )
                                .map(|f| f.name.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| format!("kind={:?}", BuiltInTypes::get_kind(cell_value)));
                eprintln!(
                    "[resume-caller-cell] first addr={:#x} value={:#x} mapped={}",
                    cell_addr, cell_value, mapped
                );
            }
            if let Some(cell_addr) = decode_arm_mov_imm3(&raw_words[11..14], 27) {
                let cell_value = unsafe { *(cell_addr as *const usize) };
                let mapped = if BuiltInTypes::get_kind(cell_value) == BuiltInTypes::Function {
                    runtime
                        .get_function_by_pointer(BuiltInTypes::untag(cell_value) as *const u8)
                        .map(|f| f.name.clone())
                        .unwrap_or_else(|| "unknown-function".to_string())
                } else {
                    format!("kind={:?}", BuiltInTypes::get_kind(cell_value))
                };
                eprintln!(
                    "[resume-caller-cell] second addr={:#x} value={:#x} mapped={}",
                    cell_addr, cell_value, mapped
                );
            }
        } else {
            eprintln!(
                "[resume-caller] current-frame-caller unknown return={:#x}",
                current_return
            );
        }

        let caller_fp = unsafe { *(new_fp as *const usize) };
        if caller_fp != 0 {
            let caller_return = unsafe { *((caller_fp + 8) as *const usize) };
            if let Some((function, offset)) =
                runtime.get_function_containing_pointer(caller_return as *const u8)
            {
                eprintln!(
                    "[resume-caller] parent-frame-caller={} offset={:#x} return={:#x}",
                    function.name, offset, caller_return
                );
            } else {
                eprintln!(
                    "[resume-caller] parent-frame-caller unknown return={:#x}",
                    caller_return
                );
            }
        }
    }

    let saved_gc_prev = return_point.saved_gc_prev;
    let gc_chain_top = crate::runtime::rebuild_segment_gc_prev_links_from_caller_chain(
        exec_base,
        seg_size,
        fp_offset,
        saved_gc_prev,
    )
    .unwrap_or(0);

    // The caller-FP chain is the authoritative structure for detached/resumed segments.
    let mut outermost_fp = new_fp;
    loop {
        let caller_fp = unsafe { *(outermost_fp as *const usize) };
        if caller_fp < exec_base || caller_fp >= exec_base + seg_size {
            break;
        }
        outermost_fp = caller_fp;
    }
    if std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
        let thread_id = std::thread::current().id();
        let old_return = unsafe { *((outermost_fp + 8) as *const usize) };
        let old_ret_fn = get_runtime()
            .get()
            .get_function_containing_pointer(old_return as *const u8)
            .map(|(function, offset)| format!("{}+{:#x}", function.name, offset))
            .unwrap_or_else(|| "unknown".to_string());
        eprintln!(
            "[invoke_cont outermost][{:?}] prompt_id={} new_fp={:#x} outermost_fp={:#x} old_ret={:#x} ({}) new_ret={:#x}",
            thread_id,
            prompt_id,
            new_fp,
            outermost_fp,
            old_return,
            old_ret_fn,
            continuation_return_address
        );
        let current_return = unsafe { *((new_fp + 8) as *const usize) };
        let current_ret_fn = get_runtime()
            .get()
            .get_function_containing_pointer(current_return as *const u8)
            .map(|(function, offset)| format!("{}+{:#x}", function.name, offset))
            .unwrap_or_else(|| "unknown".to_string());
        eprintln!(
            "[invoke_cont current][{:?}] prompt_id={} new_fp={:#x} current_ret={:#x} ({}) saved_caller_fp={:#x}",
            thread_id,
            prompt_id,
            new_fp,
            current_return,
            current_ret_fn,
            unsafe { *(new_fp as *const usize) }
        );
    }
    unsafe {
        *((outermost_fp + 8) as *mut usize) = continuation_return_address;
    }

    let resumed_frame_is_outermost = outermost_fp == new_fp;

    #[cfg(target_arch = "aarch64")]
    {
        let words = unsafe {
            [
                *(resume_address as *const u32),
                *((resume_address + 4) as *const u32),
                *((resume_address + 8) as *const u32),
                *((resume_address + 12) as *const u32),
                *((resume_address + 16) as *const u32),
                *((resume_address + 20) as *const u32),
            ]
        };
        if words[0] == 0xf85e03b7
            && words[1] == 0xaa1703fc
            && words[2] == 0xaa1c03e0
            && words[3] == 0x5400002e
            && words[4] == 0xa9bf7be0
            && words[5] == 0xf85f03a0
            && resumed_frame_is_outermost
        {
            set_gc_frame_top(saved_gc_prev);
            resume_address = resume_address.wrapping_add(0x2c);
            if std::env::var("BEAGLE_DEBUG_RESUME").is_ok() {
                eprintln!(
                    "[resume-adjust] skipped gc_frame_unlink sequence new_resume={:#x} saved_gc_prev={:#x}",
                    resume_address, saved_gc_prev
                );
            }
        }
    }

    let result_local = continuation.result_local();
    let result_ptr = if result_local != 0 {
        (new_fp as isize).wrapping_add(result_local) as usize
    } else {
        0
    };
    if std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
        let thread_id = std::thread::current().id();
        eprintln!(
            "[invoke_cont result-slot][{:?}] prompt_id={} new_fp={:#x} result_local={} result_ptr={:#x} ret_slot={:#x} fp_slot={:#x} prev_slot={:#x} header_slot={:#x}",
            thread_id,
            prompt_id,
            new_fp,
            result_local,
            result_ptr,
            new_fp + 8,
            new_fp,
            new_fp.saturating_sub(16),
            new_fp.saturating_sub(8)
        );
    }
    if result_ptr != 0 {
        unsafe {
            *(result_ptr as *mut usize) = value;
        }
        if std::env::var("BEAGLE_DEBUG_RESUME").is_ok() {
            let written = unsafe { *(result_ptr as *const usize) };
            eprintln!(
                "[resume-result-write] addr={:#x} written={:#x}",
                result_ptr, written
            );
        }
    }

    let runtime = get_runtime().get_mut();
    if push_return_point {
        crate::runtime::per_thread_data()
            .invocation_return_points
            .push(return_point.clone());
    }
    debug_cont_state("invoke:ready", prompt_id);
    if push_return_point && std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
        let thread_id = std::thread::current().id();
        eprintln!(
            "[invoke_cont segmented-rp][{:?}] prompt_id={} beagle_sp={:#x} beagle_fp={:#x} ret={:#x}",
            thread_id,
            prompt_id,
            return_point.stack_pointer,
            return_point.frame_pointer,
            return_point.return_address
        );
    }
    runtime.push_prompt_handler(crate::runtime::PromptHandler {
        handler_address: continuation.handler_address(),
        stack_pointer: continuation.prompt_stack_pointer(),
        frame_pointer: continuation.prompt_frame_pointer(),
        link_register: continuation.prompt_link_register(),
        result_local: continuation.prompt_result_local(),
        prompt_id,
    });
    if continuation.exc_has_handler() {
        runtime.push_exception_handler(crate::runtime::ExceptionHandler {
            handler_address: continuation.exc_handler_address(),
            stack_pointer: continuation.prompt_stack_pointer(),
            frame_pointer: continuation.prompt_frame_pointer(),
            link_register: continuation.prompt_link_register(),
            result_local: continuation.exc_result_local(),
            handler_id: continuation.exc_handler_id(),
            is_resumable: true,
            resume_local: continuation.exc_resume_local(),
        });
    }
    crate::runtime::per_thread_data().push_active_segment(prompt_id, cloned_segment);
    if gc_chain_top != 0 {
        set_gc_frame_top(gc_chain_top);
    } else {
        set_gc_frame_top(new_fp.saturating_sub(8));
    }

    if debug_prompts || std::env::var("BEAGLE_DEBUG_INVOKE").is_ok() {
        eprintln!(
            "[invoke_cont] segmented prompt_id={} value={:#x} new_sp={:#x} new_fp={:#x} exec_base={:#x} exec_top={:#x} resume={:#x}",
            prompt_id, value, new_sp, new_fp, exec_base, exec_top, resume_address
        );
    }

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            let runtime = get_runtime().get();
            let jump_fn = runtime
                .get_function_by_name("beagle.builtin/invoke-continuation-jump")
                .expect("invoke-continuation-jump function not found");
            let ptr: *const u8 = jump_fn.pointer.into();
            let jump_ptr: extern "C" fn(usize, usize, usize, usize, usize, usize, usize) -> ! =
                unsafe { std::mem::transmute(ptr) };
            jump_ptr(
                0, // null callee_saved_regs — registers reloaded from root slots at resume point
                continuation.resume_address(),
                0,
                new_sp,
                new_fp,
                result_ptr,
                value,
            );
        } else {
            let runtime = get_runtime().get();
            let return_jump_fn = runtime
                .get_function_by_name("beagle.builtin/return-jump")
                .expect("return-jump function not found");
            let ptr: *const u8 = return_jump_fn.pointer.into();
            let return_jump: extern "C" fn(usize, usize, usize, usize, *const usize, usize) -> ! =
                unsafe { std::mem::transmute(ptr) };
            return_jump(new_sp, new_fp, continuation_return_address, resume_address, std::ptr::null(), value);
        }
    }
}

pub unsafe extern "C" fn segmented_continuation_return(value: usize) -> ! {
    if std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
        let ptd = crate::runtime::per_thread_data();
        let top_prompt = ptd.prompt_handlers.last().map(|p| p.prompt_id);
        let active_prompt = ptd.prompt_handlers.last().and_then(|prompt| {
            ptd.active_segment_for_prompt(prompt.prompt_id)
                .map(|_| prompt.prompt_id)
        });
        let top_rp = ptd.invocation_return_points.last().map(|rp| rp.prompt_id);
        eprintln!(
            "[cont_return] value={:#x} top_prompt={:?} active_prompt={:?} top_rp={:?} prompts={} active_segments={} rps={}",
            value,
            top_prompt,
            active_prompt,
            top_rp,
            ptd.prompt_handlers.len(),
            ptd.active_segments.len(),
            ptd.invocation_return_points.len()
        );
    }
    let prompt = {
        let ptd = crate::runtime::per_thread_data();
        ptd.prompt_handlers
            .last()
            .cloned()
            .expect("segmented continuation returned without an active prompt")
    };

    let return_point = {
        let ptd = crate::runtime::per_thread_data();
        let should_pop = ptd
            .invocation_return_points
            .last()
            .map(|rp| rp.prompt_id == prompt.prompt_id)
            .unwrap_or(false);
        if should_pop {
            ptd.invocation_return_points.pop()
        } else {
            None
        }
    };

    {
        let ptd = crate::runtime::per_thread_data();
        if let Some(segment) = ptd.pop_active_segment(prompt.prompt_id) {
            ptd.recycle_segment(segment);
        }
        if ptd
            .prompt_handlers
            .last()
            .map(|handler| handler.prompt_id == prompt.prompt_id)
            .unwrap_or(false)
        {
            ptd.prompt_handlers.pop();
        }
    }
    debug_cont_state("seg-ret:after-pop", prompt.prompt_id);

    if let Some(return_point) = return_point {
        if std::env::var("BEAGLE_DEBUG_RESUME").is_ok() {
            eprintln!(
                "[seg-ret] rp prompt_id={} sp={:#x} fp={:#x} ret={:#x} value={:#x}",
                return_point.prompt_id,
                return_point.stack_pointer,
                return_point.frame_pointer,
                return_point.return_address,
                value
            );
        }
        let restored_frame = if let Some(suspended_frame) =
            crate::runtime::per_thread_data().take_suspended_frame(return_point.suspended_frame_id)
        {
            let frame_size = suspended_frame.frame_size;
            let frame_bottom = return_point
                .frame_pointer
                .checked_add(16)
                .and_then(|top| top.checked_sub(frame_size))
                .expect("segmented return frame restore underflow computing frame_bottom");
            let gc_chain_prev = gc_chain_prev_for_restored_segment(
                frame_bottom,
                frame_size,
                return_point.frame_pointer,
            );
            suspended_frame.restore_to_stack(return_point.frame_pointer, gc_chain_prev);
            true
        } else {
            false
        };
        if std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
            let thread_id = std::thread::current().id();
            eprintln!(
                "[cont_return rp][{:?}] prompt_id={} sp={:#x} fp={:#x} ret={:#x} value={:#x}",
                thread_id,
                return_point.prompt_id,
                return_point.stack_pointer,
                return_point.frame_pointer,
                return_point.return_address,
                value
            );
        }
        let caller_frame_is_live =
            !restored_frame && frame_pointer_in_active_segment(return_point.frame_pointer);
        if restored_frame || caller_frame_is_live {
            set_gc_frame_top(return_point.frame_pointer.saturating_sub(8));
        } else {
            set_gc_frame_top(return_point.saved_gc_prev);
        }
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                let runtime = get_runtime().get();
                let return_jump_fn = runtime
                    .get_function_by_name("beagle.builtin/return-jump")
                    .expect("return-jump function not found");
                let ptr: *const u8 = return_jump_fn.pointer.into();
                let return_jump_ptr: extern "C" fn(usize, usize, usize, usize, *const usize, *const u8, usize) -> ! =
                    unsafe { std::mem::transmute(ptr) };
                return_jump_ptr(
                    return_point.stack_pointer,
                    return_point.frame_pointer,
                    value,
                    return_point.return_address,
                    std::ptr::null::<usize>(),
                    std::ptr::null(),
                    0,
                );
            } else {
                let runtime = get_runtime().get();
                let return_jump_fn = runtime
                    .get_function_by_name("beagle.builtin/return-jump")
                    .expect("return-jump function not found");
                let ptr: *const u8 = return_jump_fn.pointer.into();
                let return_jump: extern "C" fn(usize, usize, usize, usize, *const usize, usize) -> ! =
                    unsafe { std::mem::transmute(ptr) };
                return_jump(return_point.stack_pointer, return_point.frame_pointer, 0, return_point.return_address, std::ptr::null(), value);
            }
        }
    }

    set_gc_frame_top(prompt.frame_pointer.saturating_sub(8));

    if prompt.result_local != 0 {
        let result_ptr =
            (prompt.frame_pointer as isize).wrapping_add(prompt.result_local) as *mut usize;
        unsafe {
            *result_ptr = value;
        }
    }

    let handler_address = prompt.handler_address;
    let new_sp = prompt.stack_pointer;
    let new_fp = prompt.frame_pointer;
    let new_lr = prompt.link_register;

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            let _ = new_lr;
            let runtime = get_runtime().get();
            let handler_jump_fn = runtime
                .get_function_by_name("beagle.builtin/handler-jump")
                .expect("handler-jump function not found");
            let ptr: *const u8 = handler_jump_fn.pointer.into();
            let handler_jump: extern "C" fn(usize, usize, usize) -> ! =
                unsafe { std::mem::transmute(ptr) };
            handler_jump(new_sp, new_fp, handler_address);
        } else {
            let runtime = get_runtime().get();
            let return_jump_fn = runtime
                .get_function_by_name("beagle.builtin/return-jump")
                .expect("return-jump function not found");
            let ptr: *const u8 = return_jump_fn.pointer.into();
            let return_jump: extern "C" fn(usize, usize, usize, usize, *const usize, usize) -> ! =
                unsafe { std::mem::transmute(ptr) };
            return_jump(new_sp, new_fp, new_lr, handler_address, std::ptr::null(), 0);
        }
    }
}

/// Invoke a captured continuation with a value.
/// The callee_saved_regs parameter contains the callee-saved registers that Beagle was using
/// when it called k() - these are saved at the very start of continuation_trampoline.
#[allow(improper_ctypes_definitions, unused_variables)]
pub unsafe extern "C" fn invoke_continuation_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    cont_ptr: usize,
    value: usize,
) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "invoke_continuation");
    trace!(
        "continuation-detail",
        "invoke_continuation: cont_ptr={:#x} value={:#x}", cont_ptr, value
    );

    let runtime = get_runtime().get_mut();
    let debug_prompts = runtime.get_command_line_args().debug;
    let continuation = ContinuationObject::from_tagged(cont_ptr).unwrap_or_else(|| {
        panic!(
            "Invalid continuation pointer: {:#x}. This is a compiler bug - trying to invoke a continuation that doesn't exist.",
            cont_ptr
        );
    });

    if debug_prompts {
        eprintln!(
            "[invoke_cont] cont_ptr={:#x} cont_prompt_fp={:#x}",
            cont_ptr,
            continuation.prompt_frame_pointer()
        );
    }

    let prompt_id = continuation.prompt_id();
    if std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
        let thread_id = std::thread::current().id();
        let ptd = crate::runtime::per_thread_data();
        eprintln!(
            "[invoke_cont][{:?}] prompt_id={} rps={}",
            thread_id,
            prompt_id,
            ptd.invocation_return_points.len()
        );
    }

    // On x86-64, `invoke_continuation_runtime` is entered through the generated
    // `continuation-trampoline`. The passed `frame_pointer` is that trampoline's
    // native FP, so its caller frame is the exact Beagle frame that invoked
    // `k(value)`. Reconstruct the caller context directly from that frame instead
    // of walking back out through Rust frames.
    let beagle_fp = unsafe { *(frame_pointer as *const usize) };
    let beagle_return_address = unsafe { *((frame_pointer + 8) as *const usize) };
    let beagle_sp = frame_pointer + 16;
    unsafe {
        invoke_continuation_runtime_with_caller_context(
            cont_ptr,
            value,
            debug_prompts,
            beagle_sp,
            beagle_fp,
            beagle_return_address,
            false,
        )
    }
}

pub unsafe fn invoke_continuation_runtime_with_caller_context(
    cont_ptr: usize,
    value: usize,
    debug_prompts: bool,
    beagle_sp: usize,
    beagle_fp: usize,
    beagle_return_address: usize,
    tail_resume: bool,
) -> ! {
    let continuation = ContinuationObject::from_tagged(cont_ptr).unwrap_or_else(|| {
        panic!(
            "Invalid continuation pointer: {:#x}. This is a compiler bug - trying to invoke a continuation that doesn't exist.",
            cont_ptr
        )
    });

    let prompt_id = continuation.prompt_id();
    let reused_return_point = if tail_resume {
        crate::runtime::per_thread_data()
            .invocation_return_points
            .last()
            .filter(|rp| rp.prompt_id == prompt_id)
            .cloned()
    } else {
        None
    };
    let (return_point, push_return_point) = if let Some(existing) = reused_return_point {
        (existing, false)
    } else {
        let header = Header::from_usize(unsafe { *((beagle_fp - 8) as *const usize) });
        let saved_gc_prev = gc_chain_anchor_for_invocation(beagle_fp);
        let suspended_frame_id = if header.type_id == TYPE_ID_FRAME
            && !frame_pointer_in_active_segment(beagle_fp)
        {
            let num_slots = header.size as usize;
            if std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
                let thread_id = std::thread::current().id();
                let frame_ret = unsafe { *((beagle_fp + 8) as *const usize) };
                let frame_ret_fn = get_runtime()
                    .get()
                    .get_function_containing_pointer(frame_ret as *const u8)
                    .map(|(function, offset)| format!("{}+{:#x}", function.name, offset))
                    .unwrap_or_else(|| "unknown".to_string());
                eprintln!(
                    "[invoke_cont caller-frame][{:?}] beagle_sp={:#x} beagle_fp={:#x} frame_ret={:#x} ({}) saved_gc_prev={:#x}",
                    thread_id, beagle_sp, beagle_fp, frame_ret, frame_ret_fn, saved_gc_prev
                );
            }
            let frame =
                crate::runtime::SuspendedFrame::capture_from_stack(beagle_fp, num_slots, beagle_sp);
            crate::runtime::per_thread_data().store_suspended_frame(frame)
        } else {
            0
        };
        (
            crate::runtime::InvocationReturnPoint {
                stack_pointer: beagle_sp,
                frame_pointer: beagle_fp,
                return_address: beagle_return_address,
                suspended_frame_id,
                prompt_id,
                saved_gc_prev,
            },
            true,
        )
    };
    let seg_ptr = continuation.segment_ptr();
    if seg_ptr == 0 || seg_ptr == BuiltInTypes::null_value() as usize {
        panic!(
            "invoke_continuation_runtime got continuation without segment data: cont_ptr={:#x}",
            cont_ptr
        );
    }

    invoke_segmented_continuation(
        &continuation,
        value,
        debug_prompts,
        return_point,
        push_return_point,
    )
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
#[allow(unused_variables)]
pub unsafe extern "C" fn continuation_trampoline(closure_ptr: usize, value: usize) -> ! {
    // Get current stack pointer and frame pointer via JIT trampolines
    #[cfg(target_arch = "x86_64")]
    let stack_pointer: usize;
    let frame_pointer: usize;

    {
        let runtime = get_runtime().get();
        #[cfg(target_arch = "x86_64")]
        {
            let fn_entry = runtime
                .get_function_by_name("beagle.builtin/read-sp-fp")
                .expect("read-sp-fp trampoline not found");
            let read_sp_fp: extern "C" fn() -> (usize, usize) =
                unsafe { std::mem::transmute::<_, _>(fn_entry.pointer) };
            let (sp, fp) = read_sp_fp();
            stack_pointer = sp;
            frame_pointer = fp;
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let fn_entry = runtime
                .get_function_by_name("beagle.builtin/read-fp")
                .expect("read-fp trampoline not found");
            let read_fp: extern "C" fn() -> usize =
                unsafe { std::mem::transmute::<_, _>(fn_entry.pointer) };
            frame_pointer = read_fp();
        }
    }

    // Extract the continuation pointer from the closure's free variables
    // Closure layout: header(8) + fn_ptr(8) + num_free(8) + num_locals(8) + free_vars...
    // First free variable is at offset 32
    let untagged_closure = BuiltInTypes::untag(closure_ptr);

    // SAFETY: closure memory layout is known
    let cont_ptr = unsafe { *((untagged_closure + 32) as *const usize) };
    let cont_ptr = cont_ptr;

    if std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
        let thread_id = std::thread::current().id();
        let resume_fn = ContinuationObject::from_tagged(cont_ptr)
            .and_then(|cont| {
                get_runtime()
                    .get()
                    .get_function_containing_pointer(cont.resume_address() as *const u8)
                    .map(|(function, offset)| format!("{}+{:#x}", function.name, offset))
            })
            .unwrap_or_else(|| "unknown".to_string());
        eprintln!(
            "[continuation_trampoline][{:?}] closure={:#x} value={:#x} cont={:#x} resume={}",
            thread_id, closure_ptr, value, cont_ptr, resume_fn
        );
    }

    // Now invoke the continuation, passing the saved callee-saved registers
    // SAFETY: invoke_continuation_runtime is an unsafe function
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let beagle_fp = *(frame_pointer as *const usize);
        let beagle_return_address = *((frame_pointer + 8) as *const usize);
        let beagle_sp = frame_pointer + 16;
        invoke_continuation_runtime_with_caller_context(
            cont_ptr,
            value,
            get_runtime().get().get_command_line_args().debug,
            beagle_sp,
            beagle_fp,
            beagle_return_address,
            false,
        )
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        invoke_continuation_runtime(stack_pointer, frame_pointer, cont_ptr, value)
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn continuation_trampoline_with_saved_regs_and_context(
    closure_ptr: usize,
    value: usize,
    _saved_regs_ptr: *const usize,
    beagle_sp: usize,
    beagle_fp: usize,
    beagle_return_address: usize,
) -> ! {
    let untagged_closure = BuiltInTypes::untag(closure_ptr);
    let cont_ptr = unsafe { *((untagged_closure + 32) as *const usize) };
    let cont_ptr = cont_ptr;

    if std::env::var("BEAGLE_DEBUG_CONT_TRAMPOLINE").is_ok() {
        eprintln!(
            "[continuation_trampoline] closure={:#x} value={:#x} cont={:#x}",
            closure_ptr, value, cont_ptr
        );
    }

    unsafe {
        invoke_continuation_runtime_with_caller_context(
            cont_ptr,
            value,
            get_runtime().get().get_command_line_args().debug,
            beagle_sp,
            beagle_fp,
            beagle_return_address,
            false,
        )
    }
}

pub unsafe extern "C" fn resume_tail_runtime(
    _stack_pointer: usize,
    _frame_pointer: usize,
    resume_closure: usize,
    value: usize,
) -> ! {
    let untagged_closure = BuiltInTypes::untag(resume_closure);
    let cont_ptr = unsafe { *((untagged_closure + 32) as *const usize) };

    let rust_fp = get_current_rust_frame_pointer();
    let trampoline_fp = unsafe { *(rust_fp as *const usize) };
    let beagle_fp = unsafe { *(trampoline_fp as *const usize) };
    let beagle_return_address = unsafe { *((trampoline_fp + 8) as *const usize) };
    let beagle_sp = trampoline_fp + 16;

    crate::runtime::per_thread_data().pop_native_perform_stack();

    unsafe {
        invoke_continuation_runtime_with_caller_context(
            cont_ptr,
            value,
            get_runtime().get().get_command_line_args().debug,
            beagle_sp,
            beagle_fp,
            beagle_return_address,
            true,
        )
    }
}

/// Return trampoline for multi-shot continuations.
/// When a continuation body returns, this trampoline is called to route the
/// return value through `return_from_shift_runtime` so that multi-shot
/// continuations work correctly.
///
/// On entry: the return value is in x0 (ARM64) or rax (x86-64)
/// This gets SP/FP and calls return_from_shift_runtime.
#[allow(dead_code)]
#[allow(unused_variables)]
pub unsafe extern "C" fn continuation_return_trampoline(value: usize) -> ! {
    // On x86-64, the continuation-return-stub passes RAX (JIT return value) as the
    // first argument (RDI -> value), so we can use `value` directly on all platforms.

    let should_use_segmented_return = {
        let ptd = crate::runtime::per_thread_data();
        let top_prompt = ptd.prompt_handlers.last().map(|prompt| prompt.prompt_id);
        let has_active = top_prompt
            .and_then(|prompt_id| ptd.active_segment_for_prompt(prompt_id).map(|_| prompt_id));
        if std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
            let thread_id = std::thread::current().id();
            eprintln!(
                "[cont_return][{:?}] value={:#x} top_prompt={:?} active_prompt={:?} prompts={} active_segments={} rps={}",
                thread_id,
                value,
                top_prompt,
                has_active,
                ptd.prompt_handlers.len(),
                ptd.active_segments.len(),
                ptd.invocation_return_points.len()
            );
        }
        has_active.is_some()
    };

    if should_use_segmented_return {
        unsafe { segmented_continuation_return(value) };
    }

    // Get current stack pointer and frame pointer via JIT trampoline
    let (stack_pointer, frame_pointer) = {
        let runtime = get_runtime().get();
        let fn_entry = runtime
            .get_function_by_name("beagle.builtin/read-sp-fp")
            .expect("read-sp-fp trampoline not found");
        let read_sp_fp: extern "C" fn() -> (usize, usize) =
            unsafe { std::mem::transmute::<_, _>(fn_entry.pointer) };
        read_sp_fp()
    };

    // Route through return_from_shift_runtime so multi-shot works
    // SAFETY: return_from_shift_runtime is an unsafe function
    unsafe {
        return_from_shift_runtime(
            stack_pointer,
            frame_pointer,
            value,
            BuiltInTypes::null_value() as usize,
        )
    }
}
