use super::*;
use crate::save_gc_context;

// ============================================================================
// Effect Handler Builtins
//
// REFACTOR A stubs. The effect handler system (handle/perform) is
// disabled — the v1 prompt machinery it relied on has been removed.
// These entry points are kept so that Ast::Handle / Ast::Perform
// compile-time builtin lookups still succeed; any program that
// actually reaches these code paths aborts with a clear error.
// Refactor B will rebuild effects on top of reset/shift.
// ============================================================================

/// Stub: push a handler onto the thread-local handler stack
pub extern "C" fn push_handler_builtin(
    _protocol_key_ptr: usize,
    _handler_instance: usize,
) -> usize {
    panic!("push-handler is temporarily disabled (Refactor A in progress)");
}

/// Stub: pop a handler from the thread-local handler stack
pub extern "C" fn pop_handler_builtin(_protocol_key_ptr: usize) -> usize {
    panic!("pop-handler is temporarily disabled (Refactor A in progress)");
}

/// Stub: find a handler in the thread-local handler stack
pub extern "C" fn find_handler_builtin(stack_pointer: usize, _protocol_key_ptr: usize) -> usize {
    unsafe {
        throw_runtime_error(
            stack_pointer,
            "RuntimeError",
            "find-handler is temporarily disabled (Refactor A in progress)".to_string(),
        );
    }
}

/// Stub: get the enum type name for a value
pub extern "C" fn get_enum_type_builtin(
    stack_pointer: usize,
    _frame_pointer: usize,
    _value: usize,
) -> usize {
    unsafe {
        throw_runtime_error(
            stack_pointer,
            "RuntimeError",
            "get-enum-type is temporarily disabled (Refactor A in progress)".to_string(),
        );
    }
}

/// Stub: call handler.handle(op, resume) using protocol dispatch
pub extern "C" fn call_handler_builtin(
    stack_pointer: usize,
    _frame_pointer: usize,
    _handler: usize,
    _enum_type_ptr: usize,
    _op_value: usize,
    _resume: usize,
) -> usize {
    save_gc_context!(stack_pointer, _frame_pointer);
    unsafe {
        throw_runtime_error(
            stack_pointer,
            "RuntimeError",
            "call-handler is temporarily disabled (Refactor A in progress)".to_string(),
        );
    }
}

/// Stub: perform an effect operation
pub unsafe extern "C" fn perform_effect_runtime_with_saved_regs(
    stack_pointer: usize,
    _frame_pointer: usize,
    _enum_type_ptr: usize,
    _op_value: usize,
    _resume_address: usize,
    _result_local_offset_raw: usize,
    _saved_regs_ptr: *const usize,
) -> usize {
    unsafe {
        throw_runtime_error(
            stack_pointer,
            "RuntimeError",
            "perform is temporarily disabled (Refactor A in progress)".to_string(),
        );
    }
}
