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
    };

    runtime.push_exception_handler(handler);
    BuiltInTypes::Int.tag(handler_id as isize) as usize
}

pub unsafe extern "C" fn pop_exception_handler_by_id_runtime(handler_id: usize) -> usize {
    print_call_builtin(get_runtime().get(), "pop_exception_handler_by_id");
    let expected_id = BuiltInTypes::untag_isize(handler_id as isize) as usize;
    let _runtime = get_runtime().get_mut();
    let ptd = crate::runtime::per_thread_data();
    if let Some(top) = ptd.exception_handlers.last() {
        if top.handler_id == expected_id {
            ptd.exception_handlers.pop();
        }
    }
    BuiltInTypes::null_value() as usize
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

    // Pop handlers until we find one
    if let Some(handler) = {
        let runtime = get_runtime().get_mut();
        runtime.pop_exception_handler()
    } {
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

            // Store exception handler info in the continuation so it can be
            // re-pushed when the continuation is invoked (for multi-throw support)
            {
                let mut cont_obj = ContinuationObject::from_tagged(cont_ptr)
                    .expect("continuation pointer invalid after capture");
                cont_obj.set_exc_handler_info(
                    handler.handler_address,
                    handler.result_local,
                    handler.resume_local,
                    handler.handler_id,
                );
            }

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

            // Jump to catch handler with restored SP, FP, and LR
            let handler_address = handler.handler_address;
            let new_sp = handler.stack_pointer;
            let new_fp = handler.frame_pointer;
            let new_lr = handler.link_register;

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

            let handler_address = handler.handler_address;
            let new_sp = handler.stack_pointer;
            let new_fp = handler.frame_pointer;
            let new_lr = handler.link_register;
            let result_local_offset = handler.result_local;

            let result_ptr = (new_fp as isize).wrapping_add(result_local_offset) as *mut usize;
            unsafe { *result_ptr = exception };

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

    // Allocate strings with proper GC root protection
    // kind_str must be rooted before allocating message_str since allocation can trigger GC
    let (kind_str, message_str) = {
        let runtime = get_runtime().get_mut();
        let kind_str: usize = runtime
            .allocate_string(stack_pointer, kind.to_string())
            .expect("Failed to allocate kind string")
            .into();

        // Register kind_str as a root before allocating message_str
        let kind_root_id = runtime.register_temporary_root(kind_str);

        let message_str: usize = runtime
            .allocate_string(stack_pointer, message)
            .expect("Failed to allocate message string")
            .into();

        // Get the potentially updated kind_str after GC
        let kind_str = runtime.unregister_temporary_root(kind_root_id);
        (kind_str, message_str)
    };
    // Runtime borrow is dropped here

    let null_location = BuiltInTypes::Null.tag(0) as usize;

    // Create the Error struct and throw it
    // Use the saved return address (the instruction after the builtin call in Beagle code)
    // as the resume address, so that resuming from a caught runtime error works correctly.
    let resume_address = get_saved_gc_return_addr();
    unsafe {
        let error = create_error(
            stack_pointer,
            frame_pointer,
            kind_str,
            message_str,
            null_location,
        );
        throw_exception(stack_pointer, frame_pointer, error, resume_address, 0);
    }
}
