use super::*;
use crate::save_gc_context;

// ============================================================================
// Effect Handler Builtins
// ============================================================================

/// Push a handler onto the thread-local handler stack
pub extern "C" fn push_handler_builtin(protocol_key_ptr: usize, handler_instance: usize) -> usize {
    let protocol_key = {
        let runtime = get_runtime().get();
        runtime.get_string_literal(protocol_key_ptr)
    };
    if std::env::var("BEAGLE_DEBUG_HANDLER_STACK").is_ok()
        && let Some(handler_obj) = HeapObject::try_from_tagged(handler_instance)
    {
        let raw_header = unsafe { *(handler_obj.untagged() as *const usize) };
        let header = handler_obj.get_header();
        let fields = handler_obj
            .get_fields()
            .iter()
            .take(4)
            .map(|value| format!("{:#x}", value))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "[push_handler] key={} handler={:#x} header={:#x} marked={} opaque={} large={} fields={}",
            protocol_key,
            handler_instance,
            raw_header,
            header.marked,
            header.opaque,
            header.large,
            fields
        );
    }
    crate::runtime::push_handler(protocol_key, handler_instance);
    BuiltInTypes::null_value() as usize
}

/// Pop a handler from the thread-local handler stack
pub extern "C" fn pop_handler_builtin(protocol_key_ptr: usize) -> usize {
    let protocol_key = {
        let runtime = get_runtime().get();
        runtime.get_string_literal(protocol_key_ptr)
    };
    crate::runtime::pop_handler(&protocol_key);
    BuiltInTypes::null_value() as usize
}

/// Find a handler in the thread-local handler stack
/// Returns the handler instance or null if not found
pub extern "C" fn find_handler_builtin(stack_pointer: usize, protocol_key_ptr: usize) -> usize {
    // Use get_string which handles both string literals and heap-allocated strings
    let protocol_key = {
        let runtime = get_runtime().get();
        runtime.get_string(stack_pointer, protocol_key_ptr)
    };
    match crate::runtime::find_handler(&protocol_key) {
        Some(handler) => {
            if std::env::var("BEAGLE_DEBUG_HANDLER_STACK").is_ok()
                && let Some(handler_obj) = HeapObject::try_from_tagged(handler)
            {
                let raw_header = unsafe { *(handler_obj.untagged() as *const usize) };
                let header = handler_obj.get_header();
                let fields = handler_obj
                    .get_fields()
                    .iter()
                    .take(4)
                    .map(|value| format!("{:#x}", value))
                    .collect::<Vec<_>>()
                    .join(" ");
                eprintln!(
                    "[find_handler] key={} handler={:#x} header={:#x} marked={} opaque={} large={} fields={}",
                    protocol_key,
                    handler,
                    raw_header,
                    header.marked,
                    header.opaque,
                    header.large,
                    fields
                );
            }
            handler
        }
        None => BuiltInTypes::null_value() as usize,
    }
}

pub fn resolve_effect_handler_method(
    runtime: &mut Runtime,
    stack_pointer: usize,
    enum_type_ptr: usize,
) -> (usize, usize) {
    let enum_type = runtime.get_string(stack_pointer, enum_type_ptr);
    let protocol_key = format!("beagle.effect/Handler<{}>", enum_type);

    let handler = match crate::runtime::find_handler(&protocol_key) {
        Some(h) => h,
        None => unsafe {
            throw_runtime_error(
                stack_pointer,
                "RuntimeError",
                format!("No handler found for protocol {}", protocol_key),
            );
        },
    };

    let dispatch_table_ptr = runtime.get_dispatch_table_ptr(&protocol_key, "handle");
    if dispatch_table_ptr.is_none() {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "RuntimeError",
                format!(
                    "No handler registered for protocol {}, dispatch key beagle.effect/handle",
                    protocol_key
                ),
            );
        }
    }

    let dispatch_table =
        unsafe { &*(dispatch_table_ptr.unwrap() as *const crate::runtime::DispatchTable) };

    let type_id = if BuiltInTypes::is_heap_pointer(handler) {
        let heap_obj = HeapObject::from_tagged(handler);
        let header_type_id = heap_obj.get_type_id();
        if header_type_id == 0 {
            heap_obj.get_struct_id()
        } else {
            let primitive_index = match header_type_id as u8 {
                TYPE_ID_STRING_SLICE | TYPE_ID_CONS_STRING => 2,
                _ => header_type_id,
            };
            0x8000_0000_0000_0000 | primitive_index
        }
    } else {
        let kind = BuiltInTypes::get_kind(handler);
        let tag = kind.get_tag() as usize;
        let primitive_index = if tag == 2 { 2 } else { tag + 16 };
        0x8000_0000_0000_0000 | primitive_index
    };

    let fn_ptr = if type_id & 0x8000_0000_0000_0000 != 0 {
        let primitive_index = type_id & 0x7FFF_FFFF_FFFF_FFFF;
        dispatch_table.lookup_primitive(primitive_index)
    } else {
        dispatch_table.lookup_struct(type_id)
    };

    if fn_ptr == 0 {
        let handler_repr = runtime
            .get_repr(handler, 0)
            .unwrap_or_else(|| "<unknown>".to_string());
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "RuntimeError",
                format!(
                    "Handler type does not implement {} protocol. Handler: {}",
                    protocol_key, handler_repr
                ),
            );
        }
    }

    (handler, fn_ptr)
}

pub unsafe fn call_beagle_fn_ptr3(
    runtime: &Runtime,
    fn_ptr: usize,
    arg0: usize,
    arg1: usize,
    arg2: usize,
) -> usize {
    let saved_ctx = save_current_gc_context();
    let apply_call = runtime
        .get_function_by_name("beagle.builtin/apply_call_3")
        .unwrap();
    let apply_call = runtime.get_pointer(apply_call).unwrap();
    let apply_call: fn(usize, usize, usize, usize) -> usize =
        unsafe { std::mem::transmute(apply_call) };
    let result = apply_call(fn_ptr, arg0, arg1, arg2);
    restore_gc_context(saved_ctx);
    result
}

/// Get the enum name for a value (by examining its struct_id/type_id)
/// Returns a string pointer to the enum name, or null if not an enum variant
pub extern "C" fn get_enum_type_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let _ = stack_pointer; // Used by save_gc_context!

    // Check if the value is a heap pointer
    if !BuiltInTypes::is_heap_pointer(value) {
        return BuiltInTypes::null_value() as usize;
    }

    // Get the struct_id from the heap object
    // For custom structs (including enum variants), header type_id is 0,
    // and the actual struct_id is stored separately
    let heap_obj = HeapObject::from_tagged(value);
    let header_type_id = heap_obj.get_type_id();

    // Only custom structs (type_id == 0) can be enum variants
    if header_type_id != 0 {
        return BuiltInTypes::null_value() as usize;
    }

    // Get struct_id directly (stored as raw value in header)
    let struct_id = heap_obj.get_struct_id();

    // Look up the enum name for this struct_id
    let runtime = get_runtime().get_mut();
    match runtime.get_enum_name_for_variant(struct_id) {
        Some(enum_name) => {
            // Allocate a string for the enum name
            match runtime.allocate_string(stack_pointer, enum_name.to_string()) {
                Ok(string_ptr) => usize::from(string_ptr),
                Err(_) => BuiltInTypes::null_value() as usize,
            }
        }
        None => BuiltInTypes::null_value() as usize,
    }
}

/// Call the `handle` method on a handler instance with the given operation and resume continuation.
///
/// This is used by `perform` to dispatch to the handler's `handle(op, resume)` method.
/// The protocol key is constructed from the enum type: "Handler<{enum_type}>"
///
/// # Arguments
/// * `stack_pointer` - Stack pointer for GC safety
/// * `frame_pointer` - Frame pointer for GC safety
/// * `handler` - The handler instance (implements Handler(T))
/// * `enum_type_ptr` - String pointer to the enum type name (e.g., "myns/Async")
/// * `op_value` - The operation value (enum variant)
/// * `resume` - The continuation closure
///
/// # Returns
/// The result of calling handler.handle(op, resume)
pub extern "C" fn call_handler_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    _handler: usize, // NOTE: Ignored! We re-read handler from GlobalObjectBlock below for GC safety
    enum_type_ptr: usize,
    op_value: usize,
    resume: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);

    let runtime = get_runtime().get_mut();
    let (handler, fn_ptr) = resolve_effect_handler_method(runtime, stack_pointer, enum_type_ptr);

    let _enum_type = runtime.get_string(stack_pointer, enum_type_ptr);
    trace!("handler", "call_handler: enum_type={}", _enum_type);

    // Call the handle function: fn handle(self, op, resume) -> result
    // The function pointer is tagged, untag it
    let fn_ptr = BuiltInTypes::untag(fn_ptr as usize);

    // The handler is compiled Beagle code, NOT a builtin.
    // Beagle functions do NOT take stack_pointer and frame_pointer as explicit args.
    // The signature is just: fn handle(self, op, resume) -> result
    let func: extern "C" fn(usize, usize, usize) -> usize = unsafe { std::mem::transmute(fn_ptr) };

    let result = func(handler, op_value, resume);
    trace!("handler", "call_handler returned: result={:#x}", result);
    result
}

pub unsafe extern "C" fn perform_effect_runtime_with_saved_regs(
    stack_pointer: usize,
    frame_pointer: usize,
    enum_type_ptr: usize,
    op_value: usize,
    resume_address: usize,
    result_local_offset_raw: usize,
    _saved_regs_ptr: *const usize,
) -> usize {
    // saved_regs_ptr is no longer used — callee-saved register values are
    // stored in root slots by the codegen save loop and restored at the
    // resume point via ReloadRootSlots.
    unsafe {
        perform_effect_runtime_inner(
            stack_pointer,
            frame_pointer,
            enum_type_ptr,
            op_value,
            resume_address,
            result_local_offset_raw,
        )
    }
}

pub unsafe fn perform_effect_runtime_inner(
    stack_pointer: usize,
    frame_pointer: usize,
    enum_type_ptr: usize,
    op_value: usize,
    resume_address: usize,
    result_local_offset_raw: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let result_local_offset = result_local_offset_raw as isize;

    if std::env::var("BEAGLE_DEBUG_PERFORM_ENV").is_ok() {
        let slot0_addr = frame_pointer.wrapping_sub(24);
        let slot0 = unsafe { *(slot0_addr as *const usize) };
        eprintln!(
            "[perform-env] fp={:#x} slot0@{:#x}={:#x} kind={:?}",
            frame_pointer,
            slot0_addr,
            slot0,
            BuiltInTypes::get_kind(slot0)
        );
        if let Some(heap_obj) = HeapObject::try_from_tagged(slot0) {
            let mut fields = Vec::new();
            for i in 0..6usize {
                fields.push(format!("{:#x}", heap_obj.get_field(i)));
            }
            eprintln!("[perform-env-fields] {}", fields.join(" "));
        }
    }

    let (op_root, enum_root) = {
        let runtime = get_runtime().get_mut();
        let op_root = if BuiltInTypes::is_heap_pointer(op_value) {
            Some(runtime.register_temporary_root(op_value))
        } else {
            None
        };
        let enum_root = if BuiltInTypes::is_heap_pointer(enum_type_ptr) {
            Some(runtime.register_temporary_root(enum_type_ptr))
        } else {
            None
        };
        (op_root, enum_root)
    };

    let cont_ptr = unsafe {
        capture_continuation_runtime_inner(
            stack_pointer,
            frame_pointer,
            resume_address,
            result_local_offset,
        )
    };

    let runtime = get_runtime().get_mut();
    let cont_root = runtime.register_temporary_root(cont_ptr);
    if let Some(cont) = ContinuationObject::from_tagged(cont_ptr) {
        crate::runtime::per_thread_data().clear_pending_captured_segment(cont.segment_handle_id());
    }

    // After capture, the original stack_pointer points into the detached segment.
    // Use the prompt's stack pointer for all subsequent allocations so that GC
    // sees valid (main-stack) pointers.
    let cont_obj = ContinuationObject::from_tagged(cont_ptr).unwrap_or_else(|| {
        panic!(
            "perform_effect_runtime got invalid continuation pointer {:#x}",
            cont_ptr
        )
    });
    let safe_sp = cont_obj.prompt_stack_pointer();

    let trampoline_fn = runtime
        .get_function_by_name("beagle.builtin/continuation-trampoline")
        .expect("continuation-trampoline builtin not found");
    let tagged_trampoline =
        BuiltInTypes::Function.tag(usize::from(trampoline_fn.pointer) as isize) as usize;

    let resume_closure = match runtime.make_closure(safe_sp, tagged_trampoline, &[cont_ptr]) {
        Ok(closure) => closure,
        Err(_) => unsafe {
            throw_runtime_error(
                safe_sp,
                "AllocationError",
                "Failed to allocate continuation closure".to_string(),
            );
        },
    };
    let resume_root = runtime.register_temporary_root(resume_closure);

    let cont_ptr = runtime.peek_temporary_root(cont_root);
    let op_value = op_root
        .map(|root| runtime.peek_temporary_root(root))
        .unwrap_or(op_value);
    let enum_type_ptr = enum_root
        .map(|root| runtime.peek_temporary_root(root))
        .unwrap_or(enum_type_ptr);
    let resume_closure = runtime.peek_temporary_root(resume_root);

    let continuation = ContinuationObject::from_tagged(cont_ptr).unwrap_or_else(|| {
        panic!(
            "perform_effect_runtime got invalid continuation pointer {:#x}",
            cont_ptr
        )
    });
    let prompt_stack_pointer = continuation.prompt_stack_pointer();

    if std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
        let thread_id = std::thread::current().id();
        let resume_fn = runtime
            .get_function_containing_pointer(resume_address as *const u8)
            .map(|(function, offset)| format!("{}+{:#x}", function.name, offset))
            .unwrap_or_else(|| "unknown".to_string());
        eprintln!(
            "[perform_effect][{:?}] cont={:#x} prompt_sp={:#x} prompt_fp={:#x} resume={:#x} ({}) op={:#x}",
            thread_id,
            cont_ptr,
            prompt_stack_pointer,
            continuation.prompt_frame_pointer(),
            resume_address,
            resume_fn,
            op_value
        );
    }

    let (handler, fn_ptr_tagged) = resolve_effect_handler_method(runtime, safe_sp, enum_type_ptr);
    let fn_ptr = BuiltInTypes::untag(fn_ptr_tagged);

    if std::env::var("BEAGLE_DEBUG_HANDLER_FIELDS").is_ok()
        && let Some(handler_obj) = HeapObject::try_from_tagged(handler)
    {
        let fields = handler_obj
            .get_fields()
            .iter()
            .take(4)
            .map(|value| format!("{:#x}", value))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "[perform_handler_fields] handler={:#x} fields={}",
            handler, fields
        );
    }

    if std::env::var("BEAGLE_DEBUG_PERFORM").is_ok() {
        eprintln!(
            "[perform_effect] handler={:#x} fn_tagged={:#x} fn_raw={:#x}",
            handler, fn_ptr_tagged, fn_ptr
        );
    }

    {
        let ptd = crate::runtime::per_thread_data();
        ptd.safe_perform_context = Some(crate::runtime::SafePerformContext {
            handler,
            op_value,
            resume_closure,
            cont_ptr,
            fn_ptr,
            prompt_handler: continuation.prompt_handler(),
        });
    }

    let runtime = get_runtime().get_mut();
    runtime.unregister_temporary_root(cont_root);
    if let Some(root) = op_root {
        runtime.unregister_temporary_root(root);
    }
    if let Some(root) = enum_root {
        runtime.unregister_temporary_root(root);
    }
    runtime.unregister_temporary_root(resume_root);

    unsafe { jump_to_safe_perform_stack() }
}
