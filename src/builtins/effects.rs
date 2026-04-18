use super::*;
use crate::save_gc_context;

// ============================================================================
// Effect Handler Builtins
//
// The per-thread effect-handler registry: a Vec<HandlerRegistryEntry>
// owned by `PerThreadData` (see src/runtime.rs). `handle { ... } with h`
// pushes; `perform op` looks up top-down by enum type_id; exit pops.
//
// The registry is keyed by the enum's struct_id (the pseudo-struct
// registered alongside the enum definition). No string allocation, no
// linear string compare — integer compare, matching the rest of the
// compiler's protocol dispatch strategy.
// ============================================================================

/// Push a handler onto the thread-local effect-handler stack.
/// Called by `Ast::Handle` compilation at handle entry.
///
/// `enum_type_id_tagged` is a tagged integer (the enum's struct_id).
/// `handler_instance` is the tagged heap pointer to the struct
/// implementing the handler protocol. `tag` is the tagged-integer prompt
/// tag associated with this handle's prompt boundary.
pub extern "C" fn push_handler_builtin(
    enum_type_id_tagged: usize,
    handler_instance: usize,
    tag: usize,
) -> usize {
    let enum_type_id = BuiltInTypes::untag(enum_type_id_tagged);
    let tag_val = BuiltInTypes::untag(tag) as u64;
    // Pin the handler in a GC root slot so it survives any GC that fires
    // while the handler is installed. The slot id is stored in the
    // registry; find/pop retrieve/release it.
    let runtime = crate::get_runtime().get_mut();
    let handler_root_id = runtime.register_temporary_root(handler_instance);
    let ptd = crate::runtime::per_thread_data();
    ptd.effect_handlers
        .push(crate::runtime::HandlerRegistryEntry {
            enum_type_id,
            handler_root_id,
            tag: tag_val,
        });
    BuiltInTypes::null_value() as usize
}

/// Pop the most-recently-pushed handler for `enum_type_id_tagged`.
/// Called by `Ast::Handle` compilation at handle exit.
pub extern "C" fn pop_handler_builtin(enum_type_id_tagged: usize) -> usize {
    let enum_type_id = BuiltInTypes::untag(enum_type_id_tagged);
    let ptd = crate::runtime::per_thread_data();
    if let Some(pos) = ptd
        .effect_handlers
        .iter()
        .rposition(|e| e.enum_type_id == enum_type_id)
    {
        let entry = ptd.effect_handlers.remove(pos);
        let runtime = crate::get_runtime().get_mut();
        runtime.unregister_temporary_root(entry.handler_root_id);
    }
    BuiltInTypes::null_value() as usize
}

/// Find the nearest handler matching `enum_type_id_tagged`, top-down.
/// Returns the tagged `handler_instance` pointer, or null if no handler
/// is installed.
pub extern "C" fn find_handler_builtin(enum_type_id_tagged: usize) -> usize {
    let enum_type_id = BuiltInTypes::untag(enum_type_id_tagged);
    let ptd = crate::runtime::per_thread_data();
    let root_id = match ptd
        .effect_handlers
        .iter()
        .rev()
        .find(|e| e.enum_type_id == enum_type_id)
        .map(|e| e.handler_root_id)
    {
        Some(id) => id,
        None => return BuiltInTypes::null_value() as usize,
    };
    // Read the current (possibly GC-moved) handler pointer from its root slot.
    crate::get_runtime().get().peek_temporary_root(root_id)
}

/// Find the tag for the nearest handler matching `enum_type_id_tagged`.
/// Returns the tag as a tagged integer, or null if no handler is
/// installed.
pub extern "C" fn find_handler_tag_builtin(enum_type_id_tagged: usize) -> usize {
    let enum_type_id = BuiltInTypes::untag(enum_type_id_tagged);
    let ptd = crate::runtime::per_thread_data();
    ptd.effect_handlers
        .iter()
        .rev()
        .find(|e| e.enum_type_id == enum_type_id)
        .map(|e| BuiltInTypes::Int.tag(e.tag as isize) as usize)
        .unwrap_or_else(|| BuiltInTypes::null_value() as usize)
}

/// Return a freshly allocated prompt tag as a tagged integer.
pub extern "C" fn fresh_tag_builtin() -> usize {
    use std::sync::atomic::Ordering;
    let runtime = crate::get_runtime().get();
    let raw = runtime.prompt_id_counter.fetch_add(1, Ordering::SeqCst);
    BuiltInTypes::Int.tag(raw as isize) as usize
}

/// Return the enum's struct_id (as a tagged integer) for a tagged
/// enum-variant value. Returns null if `value` is not a heap-allocated
/// enum variant — the AST-layer caller checks for null and raises
/// "perform requires an enum value".
pub extern "C" fn get_enum_type_builtin(value: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(value) {
        return BuiltInTypes::null_value() as usize;
    }
    let heap_obj = HeapObject::from_tagged(value);
    // Custom structs (enum variants are structs) have header.type_id == 0;
    // the real struct_id lives in header.type_data. Use get_struct_id.
    if heap_obj.get_type_id() != 0 {
        return BuiltInTypes::null_value() as usize;
    }
    let variant_struct_id = heap_obj.get_struct_id();

    let runtime = crate::get_runtime().get();
    match runtime.get_enum_id_for_variant(variant_struct_id) {
        Some(enum_id) => BuiltInTypes::Int.tag(enum_id as isize) as usize,
        None => BuiltInTypes::null_value() as usize,
    }
}

/// Stub: call handler.handle(op, resume) using protocol dispatch
pub extern "C" fn call_handler_builtin(
    stack_pointer: usize,
    _frame_pointer: usize,
    _handler: usize,
    _enum_type_id: usize,
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

/// Dispatch `handler.handle(op_value, resume_closure)` via the Handler
/// protocol's dispatch table, then longjmp back past the enclosing
/// `__reset__` with the handler's return value.
///
/// The IR-level compilation of `perform op` has already done the work
/// of capturing the continuation and wrapping it in a trampoline
/// closure. This builtin only performs the dispatch + return-from-shift.
///
/// Does not return. Either:
///   - the handler calls `resume(v)`, and `continuation_trampoline`
///     teleports back to the resume point (overwriting our frame); or
///   - the handler returns a value, and we longjmp back past
///     `__reset__` via `return_from_shift_runtime_inner`.
pub unsafe extern "C" fn perform_dispatch_and_return_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    handler: usize,
    op_value: usize,
    resume_closure: usize,
    _enum_type_id_tagged: usize,
) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);

    let runtime = crate::get_runtime().get();

    if BuiltInTypes::get_kind(resume_closure) == BuiltInTypes::Closure {
        let resume_obj = HeapObject::from_tagged(resume_closure);
        for free_var in &resume_obj.get_fields()[3..] {
            if BuiltInTypes::get_kind(*free_var) != BuiltInTypes::Closure {
                continue;
            }
            let raw_resume_obj = HeapObject::from_tagged(*free_var);
            let raw_fn_ptr = BuiltInTypes::untag(raw_resume_obj.get_field(0)) as *const u8;
            let is_continuation_trampoline = runtime
                .get_function_by_pointer(raw_fn_ptr)
                .map(|f| f.name == "beagle.builtin/continuation-trampoline")
                .unwrap_or(false);
            if is_continuation_trampoline {
                eprintln!(
                    "[perform-dispatch-raw-resume] wrapped={:#x} raw={:#x} raw_field3={:#x}",
                    resume_closure,
                    *free_var,
                    raw_resume_obj.get_field(3)
                );
            }
        }
    }

    // Call through the protocol dispatcher function `beagle.effect/handle`,
    // which handles inline-cache dispatch and struct-id → method lookup.
    // Same calling convention as user calls to `handle(h, op, resume)`.
    let handle_fn_entry = runtime
        .get_function_by_name("beagle.effect/handle")
        .unwrap_or_else(|| unsafe {
            throw_runtime_error(
                stack_pointer,
                "RuntimeError",
                "beagle.effect/handle dispatcher not compiled — was any Handler extension registered?"
                    .to_string(),
            );
        });
    let handle_fn_ptr = runtime
        .get_pointer(handle_fn_entry)
        .expect("handle dispatcher has no code pointer");

    let save_vr_fn_entry = runtime
        .get_function_by_name("beagle.builtin/save_volatile_registers3")
        .expect("save_volatile_registers3 trampoline not registered");
    let save_vr_ptr = runtime
        .get_pointer(save_vr_fn_entry)
        .expect("save_volatile_registers3 pointer not available");
    let save_vr: extern "C" fn(usize, usize, usize, usize) -> usize =
        unsafe { std::mem::transmute(save_vr_ptr) };

    let result = save_vr(handler, op_value, resume_closure, handle_fn_ptr as usize);

    // Handler returned without calling resume. Longjmp past the enclosing
    // __reset__ with `result` as its return value.
    unsafe {
        crate::builtins::reset_shift::return_from_shift_runtime_inner(
            stack_pointer,
            frame_pointer,
            result,
        )
    }
}

/// Tag-aware variant of `perform_dispatch_and_return_runtime`. Same
/// handler dispatch, but on normal handler return we longjmp through
/// the matching prompt-tag record via `return_from_shift_tagged` rather
/// than FP-walking for an enclosing `__reset__`. The tag comes from the
/// handler's registry entry (looked up at `perform`-time and threaded
/// through as an extra argument) so the tagged path doesn't need any
/// changes to the prompt-lookup infrastructure on this side.
pub unsafe extern "C" fn perform_dispatch_and_return_tagged_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    handler: usize,
    op_value: usize,
    resume_closure: usize,
    tag_tagged: usize,
) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);

    let runtime = crate::get_runtime().get();

    let handle_fn_entry = runtime
        .get_function_by_name("beagle.effect/handle")
        .unwrap_or_else(|| unsafe {
            throw_runtime_error(
                stack_pointer,
                "RuntimeError",
                "beagle.effect/handle dispatcher not compiled — was any Handler extension registered?"
                    .to_string(),
            );
        });
    let handle_fn_ptr = runtime
        .get_pointer(handle_fn_entry)
        .expect("handle dispatcher has no code pointer");

    let save_vr_fn_entry = runtime
        .get_function_by_name("beagle.builtin/save_volatile_registers3")
        .expect("save_volatile_registers3 trampoline not registered");
    let save_vr_ptr = runtime
        .get_pointer(save_vr_fn_entry)
        .expect("save_volatile_registers3 pointer not available");
    let save_vr: extern "C" fn(usize, usize, usize, usize) -> usize =
        unsafe { std::mem::transmute(save_vr_ptr) };

    let result = save_vr(handler, op_value, resume_closure, handle_fn_ptr as usize);

    // Handler returned without calling resume. Untag the prompt tag and
    // longjmp through the matching prompt-tag record.
    let tag = BuiltInTypes::untag(tag_tagged);
    unsafe {
        crate::builtins::reset_shift::return_from_shift_tagged_runtime(
            stack_pointer,
            frame_pointer,
            result,
            tag,
        )
    }
}
