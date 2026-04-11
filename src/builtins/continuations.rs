use super::*;
use crate::save_gc_context;

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

/// REFACTOR A stub. `resume-tail` was a tail-position continuation
/// invoker used by the async implicit-handler runtime. Referenced by
/// stdlib at compile time; any runtime call aborts.
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

/// REFACTOR A stub. `invoke-continuation` was the direct
/// (non-closure-wrapped) continuation invoker. Referenced by stdlib
/// at compile time; any runtime call aborts.
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

/// REFACTOR A stub. Was called after `handle` dispatch to route a
/// handler's return value through the v1 return-from-shift path.
/// Kept as a thin wrapper over the new reset/shift return path so
/// the install.rs registration stays valid — not actually reachable
/// from live code after Refactor A.
pub unsafe extern "C" fn return_from_shift_handler_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
    _cont_ptr: usize,
) -> ! {
    unsafe {
        crate::builtins::reset_shift::return_from_shift_runtime_inner(
            stack_pointer,
            frame_pointer,
            value,
        )
    }
}
