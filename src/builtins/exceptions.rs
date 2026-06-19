use super::*;
use crate::save_gc_context;

pub unsafe extern "C" fn push_exception_handler_runtime(
    handler_address: usize,
    result_local: isize,
    link_register: usize,
    stack_pointer: usize,
    frame_pointer: usize,
) -> usize {
    print_call_builtin(get_runtime().get(), "push_exception_handler");
    let runtime = get_runtime().get_mut();

    // All values are passed as parameters since we can't reliably read them
    // inside this function (x30 gets clobbered by the call, SP/FP might be modified)
    let handler = crate::runtime::ExceptionHandler {
        handler_address,
        stack_pointer,
        frame_pointer,
        link_register,
        result_local,
        handler_id: 0,
        is_resumable: false,
        resume_local: 0,
        saved_marks_count: super::current_marks_count(),
        saved_frame_marks: unsafe { super::snapshot_frame_marks(frame_pointer) },
        saved_effect_handlers_len: crate::runtime::per_thread_data().effect_handlers.len(),
        saved_prompt_tags_len: crate::runtime::per_thread_data().prompt_tags.len(),
    };

    runtime.push_exception_handler(handler);
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn pop_exception_handler_runtime() -> usize {
    print_call_builtin(get_runtime().get(), "pop_exception_handler");
    let runtime = get_runtime().get_mut();
    runtime.pop_exception_handler();
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn push_resumable_exception_handler_runtime(
    handler_address: usize,
    result_local: isize,
    resume_local: isize,
    link_register: usize,
    stack_pointer: usize,
    frame_pointer: usize,
) -> usize {
    print_call_builtin(get_runtime().get(), "push_resumable_exception_handler");
    let runtime = get_runtime().get_mut();

    let handler_id = runtime
        .prompt_id_counter
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    let handler = crate::runtime::ExceptionHandler {
        handler_address,
        stack_pointer,
        frame_pointer,
        link_register,
        result_local,
        handler_id,
        is_resumable: true,
        resume_local,
        saved_marks_count: super::current_marks_count(),
        saved_frame_marks: unsafe { super::snapshot_frame_marks(frame_pointer) },
        saved_effect_handlers_len: crate::runtime::per_thread_data().effect_handlers.len(),
        saved_prompt_tags_len: crate::runtime::per_thread_data().prompt_tags.len(),
    };

    runtime.push_exception_handler(handler);
    BuiltInTypes::Int.tag(handler_id as isize) as usize
}

pub unsafe extern "C" fn pop_exception_handler_by_id_runtime(handler_id: usize) -> usize {
    print_call_builtin(get_runtime().get(), "pop_exception_handler_by_id");
    let expected_id = BuiltInTypes::untag_isize(handler_id as isize) as usize;
    let _runtime = get_runtime().get_mut();
    let ptd = crate::runtime::per_thread_data();
    if let Some(top) = ptd.exception_handlers.last()
        && top.handler_id == expected_id
    {
        ptd.exception_handlers.pop();
    }
    BuiltInTypes::null_value() as usize
}

/// Unwind the per-thread effect-handler and prompt-tag side stacks to
/// the depths recorded when `handler` was pushed. A throw that unwinds
/// past `handle` / `reset(tag)` frames abandons those extents — their
/// side-stack records hold SP/FP into now-dead stack and MUST not be
/// matched by a later `perform` (a stale match produces a zero-byte
/// continuation segment whose invocation aborts the process).
/// Handler-registry entries pin their handler instance in a GC root
/// slot; release those for every dropped entry, mirroring
/// `pop_handler_builtin`.
fn unwind_effect_state_to(handler: &crate::runtime::ExceptionHandler) {
    let ptd = crate::runtime::per_thread_data();
    if ptd.effect_handlers.len() > handler.saved_effect_handlers_len {
        let runtime = get_runtime().get_mut();
        for entry in ptd
            .effect_handlers
            .drain(handler.saved_effect_handlers_len..)
        {
            runtime.unregister_temporary_root(entry.handler_root_id);
        }
    }
    if ptd.prompt_tags.len() > handler.saved_prompt_tags_len {
        ptd.prompt_tags.truncate(handler.saved_prompt_tags_len);
    }
}

pub unsafe extern "C" fn throw_exception(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
    resume_address: usize,
    resume_local_offset: isize,
) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "throw_exception");

    // Create exception object
    let exception = {
        let runtime = get_runtime().get_mut();
        match runtime.create_exception(stack_pointer, value) {
            Ok(exc) => exc,
            Err(_) => {
                // Failed to allocate exception object - use the original value as the exception
                // This is a last resort to avoid a panic when out of memory
                value
            }
        }
    };

    // Look up the top handler. For RESUMABLE handlers we clone without
    // popping so that a subsequent throw inside the resumed body still
    // finds a handler (deep-handler semantics). For non-resumable
    // handlers we pop as before — abortive catch only fires once and
    // control never comes back inside the try body.
    //
    // The resumable handler is popped by `pop_exception_handler_by_id`
    // on the normal-completion path of the try, or by the catch-body's
    // `after_catch` cleanup when the catch finishes executing.
    let handler_opt: Option<crate::runtime::ExceptionHandler> = {
        let ptd = crate::runtime::per_thread_data();
        match ptd.exception_handlers.last() {
            Some(top) if top.is_resumable => Some(top.clone()),
            _ => {
                let runtime = get_runtime().get_mut();
                runtime.pop_exception_handler()
            }
        }
    };

    if let Some(handler) = handler_opt {
        if handler.is_resumable {
            // Resumable path: capture continuation like shift does, then jump to catch
            // capture_continuation_runtime pops the topmost prompt handler
            let cont_ptr = unsafe {
                capture_continuation_runtime(
                    stack_pointer,
                    frame_pointer,
                    resume_address,
                    resume_local_offset,
                )
            };

            // Write exception to handler's result_local (exception binding)
            let exception_ptr =
                (handler.frame_pointer as isize).wrapping_add(handler.result_local) as *mut usize;
            unsafe { *exception_ptr = exception };

            // Write raw continuation pointer to handler's resume_local
            let resume_ptr =
                (handler.frame_pointer as isize).wrapping_add(handler.resume_local) as *mut usize;
            unsafe { *resume_ptr = cont_ptr };

            // Unwind the GC frame chain to the handler's level.
            // Frames deeper than the handler are being abandoned (SP/FP are
            // restored to the handler's values), so their GC chain entries
            // must be removed. Otherwise, another thread triggering GC would
            // walk stale entries pointing into overwritten stack space.
            let handler_header = handler.frame_pointer - 8;
            {
                let mut hdr = get_gc_frame_top();
                while hdr != 0 && hdr < handler_header {
                    hdr = unsafe { *((hdr.wrapping_sub(8)) as *const usize) };
                }
                set_gc_frame_top(hdr);
            }

            // Roll continuation marks back to the handler's level.
            // Marks in abandoned (deeper) frames die with their frames,
            // but marks installed in the HANDLER's own frame (e.g. a
            // `binding` inside the try body — its mark local is hoisted
            // into the enclosing function's frame) survive the unwind
            // and must be truncated to the push-time snapshot, or the
            // dynamic var stays bound to a dead value for the rest of
            // the thread's life.
            super::set_marks_count(handler.saved_marks_count);
            unsafe { super::restore_frame_marks(handler.frame_pointer, handler.saved_frame_marks) };

            // Snapshot the handle scopes between the catch and the throw
            // (the records `unwind_effect_state_to` is about to truncate) onto
            // the continuation, so a resume back into the body re-establishes
            // them. Without this, a `perform` in the resumed body misses its
            // handler and falls through to an outer one. Must run BEFORE the
            // unwind, while the effect-handler roots are still registered.
            unsafe {
                crate::builtins::reset_shift::snapshot_boundary_records_onto_cont(
                    cont_ptr,
                    stack_pointer,
                    handler.saved_prompt_tags_len,
                    handler.saved_effect_handlers_len,
                );
            }

            // Drop effect-handler / prompt-tag records belonging to the
            // abandoned frames (see `unwind_effect_state_to`).
            unwind_effect_state_to(&handler);

            // Jump to catch handler with restored SP, FP, and LR
            let handler_address = handler.handler_address;
            let new_sp = handler.stack_pointer;
            let new_fp = handler.frame_pointer;
            let new_lr = handler.link_register;

            let runtime = get_runtime().get();
            let return_jump_fn = runtime
                .get_function_by_name("beagle.builtin/return-jump")
                .expect("return-jump function not found");
            let ptr: *const u8 = return_jump_fn.pointer.into();
            let return_jump: extern "C" fn(usize, usize, usize, usize, *const usize, usize) -> ! =
                unsafe { std::mem::transmute(ptr) };
            return_jump(new_sp, new_fp, new_lr, handler_address, std::ptr::null(), 0);
        } else {
            // Non-resumable path: existing behavior
            // Unwind GC frame chain (same as resumable path above)
            let handler_header = handler.frame_pointer - 8;
            {
                let mut hdr = get_gc_frame_top();
                while hdr != 0 && hdr < handler_header {
                    hdr = unsafe { *((hdr.wrapping_sub(8)) as *const usize) };
                }
                set_gc_frame_top(hdr);
            }

            // Roll continuation marks back to the handler's level
            // (same reasoning as the resumable path above).
            super::set_marks_count(handler.saved_marks_count);
            unsafe { super::restore_frame_marks(handler.frame_pointer, handler.saved_frame_marks) };

            // Drop effect-handler / prompt-tag records belonging to the
            // abandoned frames (see `unwind_effect_state_to`).
            unwind_effect_state_to(&handler);

            let handler_address = handler.handler_address;
            let new_sp = handler.stack_pointer;
            let new_fp = handler.frame_pointer;
            let new_lr = handler.link_register;
            let result_local_offset = handler.result_local;

            let result_ptr = (new_fp as isize).wrapping_add(result_local_offset) as *mut usize;
            unsafe { *result_ptr = exception };

            let runtime = get_runtime().get();
            let return_jump_fn = runtime
                .get_function_by_name("beagle.builtin/return-jump")
                .expect("return-jump function not found");
            let ptr: *const u8 = return_jump_fn.pointer.into();
            let return_jump: extern "C" fn(usize, usize, usize, usize, *const usize, usize) -> ! =
                unsafe { std::mem::transmute(ptr) };
            return_jump(new_sp, new_fp, new_lr, handler_address, std::ptr::null(), 0);
        }
    } else {
        // No try-catch handler found
        // Check per-thread uncaught exception handler (JVM-style)
        let thread_handler_fn = {
            let runtime = get_runtime().get();
            runtime.get_thread_exception_handler()
        };

        if let Some(handler_fn) = thread_handler_fn {
            // Call the per-thread uncaught exception handler
            let runtime = get_runtime().get();
            unsafe { call_beagle_fn_ptr(runtime, handler_fn, exception) };
            // Handler ran, now terminate since exception was uncaught
            println!("Uncaught exception after thread handler:");
            get_runtime().get_mut().println(exception).ok();
            unsafe { throw_error(stack_pointer, frame_pointer) };
        }

        // Check default (global) uncaught exception handler
        let default_handler_fn = {
            let runtime = get_runtime().get();
            runtime.default_exception_handler_fn
        };

        if let Some(handler_fn) = default_handler_fn {
            // Call the Beagle handler function
            let runtime = get_runtime().get();
            unsafe { call_beagle_fn_ptr(runtime, handler_fn, exception) };
            // Handler ran, now panic since exception was uncaught
            println!("Uncaught exception after default handler:");
            get_runtime().get_mut().println(exception).ok();
            unsafe { throw_error(stack_pointer, frame_pointer) };
        }

        // No handler at all - panic with stack trace
        println!("Uncaught exception:");
        get_runtime().get_mut().println(exception).ok();
        unsafe { throw_error(stack_pointer, frame_pointer) };
    }
}

pub unsafe extern "C" fn set_thread_exception_handler(handler_fn: usize) {
    let runtime = get_runtime().get_mut();
    runtime.set_thread_exception_handler(handler_fn);
}

pub unsafe extern "C" fn set_default_exception_handler(handler_fn: usize) {
    let runtime = get_runtime().get_mut();
    runtime.set_default_exception_handler(handler_fn);
}

/// Creates an Error struct on the heap
/// Returns tagged heap pointer to Error { kind, message, location }
pub unsafe extern "C" fn create_error(
    stack_pointer: usize,
    _frame_pointer: usize, // Frame pointer for GC stack walking (needed since create_struct can allocate)
    kind_str: usize, // Tagged string specifying the error variant (e.g., "StructError", "TypeError")
    message_str: usize, // Tagged string
    location_str: usize, // Tagged string or null
) -> usize {
    print_call_builtin(get_runtime().get(), "create_error");

    let runtime = get_runtime().get_mut();

    let kind_root = if BuiltInTypes::is_heap_pointer(kind_str) {
        Some(runtime.register_temporary_root(kind_str))
    } else {
        None
    };
    let message_root = if BuiltInTypes::is_heap_pointer(message_str) {
        Some(runtime.register_temporary_root(message_str))
    } else {
        None
    };
    let location_root = if BuiltInTypes::is_heap_pointer(location_str) {
        Some(runtime.register_temporary_root(location_str))
    } else {
        None
    };

    let kind_str = kind_root
        .map(|root| runtime.peek_temporary_root(root))
        .unwrap_or(kind_str);
    let message_str = message_root
        .map(|root| runtime.peek_temporary_root(root))
        .unwrap_or(message_str);
    let location_str = location_root
        .map(|root| runtime.peek_temporary_root(root))
        .unwrap_or(location_str);

    // Extract the error kind string to determine which variant to create
    // Use get_string which handles both string constants and heap-allocated strings
    let kind = runtime.get_string(stack_pointer, kind_str);
    // Use the general struct creation helper
    let fields = vec![message_str, location_str];
    let error = runtime
        .create_struct(
            "beagle.core/SystemError",
            Some(&kind),
            &fields,
            stack_pointer,
        )
        .expect("Failed to create SystemError");

    if let Some(root) = kind_root {
        runtime.unregister_temporary_root(root);
    }
    if let Some(root) = message_root {
        runtime.unregister_temporary_root(root);
    }
    if let Some(root) = location_root {
        runtime.unregister_temporary_root(root);
    }

    error
}

/// Helper to throw a runtime error with kind and message strings
/// This is a convenience function for Rust code to throw structured exceptions
pub unsafe fn throw_runtime_error(stack_pointer: usize, kind: &str, message: String) -> ! {
    // Get frame_pointer from thread-local (set by the builtin entry)
    let frame_pointer = get_saved_frame_pointer();

    // The saved return address points just after the call that raised this
    // error, so the function containing it is exactly where the error occurred.
    // Use it to populate the error's `location` field instead of leaving null.
    let resume_address = get_saved_gc_return_addr();
    let location = crate::builtins::location_for_address(resume_address);

    // Allocate strings with proper GC root protection. Each prior allocation is
    // rooted before the next, since allocation can trigger GC (which moves them).
    let (kind_str, message_str, location_str) = {
        let runtime = get_runtime().get_mut();
        let kind_str: usize = runtime
            .allocate_string(stack_pointer, kind.to_string())
            .expect("Failed to allocate kind string")
            .into();
        let kind_root_id = runtime.register_temporary_root(kind_str);

        let message_str: usize = runtime
            .allocate_string(stack_pointer, message)
            .expect("Failed to allocate message string")
            .into();
        let message_root_id = runtime.register_temporary_root(message_str);

        let location_str: usize = match location {
            Some(loc) => runtime
                .allocate_string(stack_pointer, loc)
                .expect("Failed to allocate location string")
                .into(),
            None => BuiltInTypes::Null.tag(0) as usize,
        };

        // Re-read the earlier pointers after the location allocation's possible GC.
        let message_str = runtime.unregister_temporary_root(message_root_id);
        let kind_str = runtime.unregister_temporary_root(kind_root_id);
        (kind_str, message_str, location_str)
    };
    // Runtime borrow is dropped here

    unsafe {
        let error = create_error(
            stack_pointer,
            frame_pointer,
            kind_str,
            message_str,
            location_str,
        );
        throw_exception(stack_pointer, frame_pointer, error, resume_address, 0);
    }
}
