use super::*;
use crate::save_gc_context;

// ============================================================================
// Effect Handler Builtins
//
// The per-thread effect-handler registry: a Vec<HandlerRegistryEntry>
// owned by `PerThreadData` (see src/runtime.rs). `handle { ... } with h`
// pushes; `perform op` looks up top-down by protocol key; exit pops.
//
// This file implements the registry and enum-type inspection. The
// continuation-capturing side of `perform` — `perform_effect` and
// `call_handler` — remains stubbed; those come in Step E6 once
// prompt tags are wired in.
// ============================================================================

/// Push a handler onto the thread-local effect-handler stack.
/// Called by `Ast::Handle` compilation at handle entry.
///
/// `protocol_key_ptr` is a tagged Beagle string like
/// `"beagle.effect/Handler<MyEffect>"`. `handler_instance` is the
/// tagged heap pointer to the struct implementing the handler protocol.
pub extern "C" fn push_handler_builtin(
    protocol_key_ptr: usize,
    handler_instance: usize,
) -> usize {
    let runtime = crate::get_runtime().get();
    let protocol_key = runtime.get_string_literal(protocol_key_ptr);
    let ptd = crate::runtime::per_thread_data();
    ptd.effect_handlers
        .push(crate::runtime::HandlerRegistryEntry {
            protocol_key,
            handler_instance,
            tag: 0, // Populated in Step E3 when tag-aware reset lands.
        });
    BuiltInTypes::null_value() as usize
}

/// Pop the most-recently-pushed handler for `protocol_key_ptr`.
/// Called by `Ast::Handle` compilation at handle exit.
///
/// Searches top-down — this handles nested handles of the same effect
/// correctly (the innermost one gets popped first).
pub extern "C" fn pop_handler_builtin(protocol_key_ptr: usize) -> usize {
    let runtime = crate::get_runtime().get();
    let protocol_key = runtime.get_string_literal(protocol_key_ptr);
    let ptd = crate::runtime::per_thread_data();
    if let Some(pos) = ptd
        .effect_handlers
        .iter()
        .rposition(|e| e.protocol_key == protocol_key)
    {
        ptd.effect_handlers.remove(pos);
    }
    BuiltInTypes::null_value() as usize
}

/// Find the nearest handler matching `protocol_key_ptr`, top-down.
/// Called by `perform op` compilation before shifting. Returns the
/// tagged `handler_instance` pointer, or null if no handler is
/// installed (caller handles the null case with an error message).
pub extern "C" fn find_handler_builtin(stack_pointer: usize, protocol_key_ptr: usize) -> usize {
    let runtime = crate::get_runtime().get();
    let protocol_key = runtime.get_string(stack_pointer, protocol_key_ptr);
    let ptd = crate::runtime::per_thread_data();
    ptd.effect_handlers
        .iter()
        .rev()
        .find(|e| e.protocol_key == protocol_key)
        .map(|e| e.handler_instance)
        .unwrap_or_else(|| BuiltInTypes::null_value() as usize)
}

/// Return the enum name (as a Beagle string) for a tagged enum-variant
/// value. `perform op` uses this to construct the protocol key at
/// runtime. Returns null if `value` is not a heap-allocated enum
/// variant — the AST-layer caller checks for null and raises
/// "perform requires an enum value".
pub extern "C" fn get_enum_type_builtin(
    stack_pointer: usize,
    _frame_pointer: usize,
    value: usize,
) -> usize {
    if !BuiltInTypes::is_heap_pointer(value) {
        return BuiltInTypes::null_value() as usize;
    }
    let heap_obj = HeapObject::from_tagged(value);
    let header = heap_obj.get_header();
    let struct_id: usize = header.type_id.into();

    let enum_name_opt = {
        let runtime = crate::get_runtime().get();
        runtime.get_enum_name_for_variant(struct_id).cloned()
    };

    match enum_name_opt {
        Some(enum_name) => {
            let runtime = crate::get_runtime().get_mut();
            match runtime.allocate_string(stack_pointer, enum_name) {
                Ok(ptr) => ptr.into(),
                Err(_) => BuiltInTypes::null_value() as usize,
            }
        }
        None => BuiltInTypes::null_value() as usize,
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
