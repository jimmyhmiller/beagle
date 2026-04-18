use super::*;

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

/// Push a prompt-tag record onto the per-thread side stack. Used by
/// the tag-aware reset path for effect handlers (Step E6+): each
/// `handle { ... } with h` generates a fresh u64 tag and pushes a
/// record capturing SP/FP/LR so a later `shift(tag)` knows where to
/// longjmp. Not called by plain reset/shift (which use the FP walker).
///
/// Arguments:
///   `tag` — u64 prompt tag, passed as usize.
///   `stack_pointer` — SP at push time (caller's SP before the reset).
///   `frame_pointer` — FP at push time.
///   `link_register` — address to longjmp to on shift-return.
///   `result_local_offset` — signed byte offset from the caller's frame
///     pointer to the local slot that should receive the shift/handler
///     return value on longjmp. Passed as `usize` over the FFI and
///     reinterpreted as `isize` inside.
///
/// Returns SP unchanged (the Beagle IR caller doesn't mutate it).
pub unsafe extern "C" fn push_prompt_tag_runtime(
    tag: usize,
    stack_pointer: usize,
    frame_pointer: usize,
    link_register: usize,
    result_local_offset: usize,
) -> usize {
    // Beagle passes the tag as a tagged integer (`fresh-tag` returns
    // `BuiltInTypes::Int.tag(raw)`). The prompt-tag stack keys on the
    // underlying u64, so untag before storing.
    let tag_raw = BuiltInTypes::untag(tag) as u64;
    let runtime = crate::get_runtime().get();
    runtime.push_prompt_tag(
        tag_raw,
        stack_pointer,
        frame_pointer,
        link_register,
        result_local_offset as isize,
    );
    stack_pointer
}

/// Pop the top prompt-tag record, asserting the tag matches. Called at
/// the end of a tagged-reset body on the normal-completion path.
pub unsafe extern "C" fn pop_prompt_tag_runtime(tag: usize) -> usize {
    // Mirror push_prompt_tag_runtime — untag the caller's tagged int
    // before comparing against the stored u64 key.
    let tag_raw = BuiltInTypes::untag(tag) as u64;
    let runtime = crate::get_runtime().get();
    runtime.pop_prompt_tag(tag_raw);
    BuiltInTypes::null_value() as usize
}

/// Tail-call a resume closure with the given value. Exposed to stdlib
/// as `builtin/resume-tail(resume, value)` so async handlers can mark
/// the tail-position invocation explicitly.
///
/// The closure is typically a `continuation_trampoline` wrapper, in
/// which case the call teleports and never returns. For non-trampoline
/// closures (e.g. the deep-handler-wrapped resume), it returns the
/// closure's result.
pub unsafe extern "C" fn resume_tail_runtime(
    _stack_pointer: usize,
    _frame_pointer: usize,
    resume_closure: usize,
    value: usize,
) -> usize {
    // Extract the closure's body fn pointer (layout: header, fn_ptr, ...).
    // Closure layout: header(8) + fn_ptr_tagged(8) + num_free(8) + num_locals(8) + free_var[0]...
    // Function-tagged pointers use shift-left-by-3 encoding, so untag
    // via >> 3 (BuiltInTypes::untag), not a low-bit mask.
    let untagged_closure = BuiltInTypes::untag(resume_closure);
    let fn_ptr_tagged = unsafe { *((untagged_closure + 8) as *const usize) };
    let raw_fn_ptr = BuiltInTypes::untag(fn_ptr_tagged) as *const u8;

    // Call the closure body directly. For continuation_trampoline,
    // this teleports and does not return via this frame — the resumed
    // body's outermost frame points at our caller's frame via the
    // Beagle ABI. For regular closures (e.g. deep-handler-wrapped),
    // the call returns normally with the closure's result.
    let closure_body: extern "C" fn(usize, usize) -> usize =
        unsafe { std::mem::transmute(raw_fn_ptr) };
    closure_body(resume_closure, value)
}

/// Entry point called from the x86-64 continuation trampoline generated
/// in `main.rs::compile_continuation_trampolines`. The arm64 backend
/// uses the pure-Rust `continuation_trampoline` in `reset_shift.rs`
/// instead and never reaches this function.
///
/// The generated x86-64 trampoline was written against the v1 reset/
/// shift machinery that captured callee-saved registers into a
/// `saved_regs` array; the new machinery doesn't need them. Until the
/// x86-64 trampoline is rewritten against the new design, calling
/// `k(value)` on x86-64 aborts here with a clear error.
#[cfg(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
))]
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
            "continuation invocation is not implemented on the x86-64 backend".to_string(),
        );
    }
}
