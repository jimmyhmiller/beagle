use super::*;
use crate::save_gc_context;

pub unsafe extern "C" fn println_value(value: usize) -> usize {
    if std::env::var("BEAGLE_DEBUG_RESUME").is_ok() {
        eprintln!("[println_value] arg={:#x}", value);
    }
    let runtime = get_runtime().get_mut();
    let result = runtime.println(value);
    if let Err(error) = result {
        let stack_pointer = get_current_stack_pointer();
        let frame_pointer = get_saved_frame_pointer();
        println!("Error: {:?}", error);
        unsafe { throw_error(stack_pointer, frame_pointer) };
    }
    0b111
}

pub unsafe extern "C" fn to_string(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let result = runtime.get_repr(value, 0);
    if result.is_none() {
        let stack_pointer = get_current_stack_pointer();
        unsafe { throw_error(stack_pointer, frame_pointer) };
    }
    let result = result.unwrap();
    unsafe { allocate_string_or_throw(runtime, stack_pointer, result) }
}

pub unsafe extern "C" fn repr(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let result = runtime.get_eval_repr(value, 1);
    if result.is_none() {
        let stack_pointer = get_current_stack_pointer();
        unsafe { throw_error(stack_pointer, frame_pointer) };
    }
    let result = result.unwrap();
    unsafe { allocate_string_or_throw(runtime, stack_pointer, result) }
}

pub unsafe extern "C" fn to_number(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let string = runtime.get_string(stack_pointer, value);
    if string.contains(".") {
        match string.parse::<f64>() {
            Ok(result) => {
                // Must heap-allocate the float, not use construct_float which creates
                // an inline tagged value that looks like a heap pointer to the GC but
                // contains raw f64 bits instead of a real address.
                let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
                    Ok(ptr) => ptr,
                    Err(_) => unsafe {
                        throw_runtime_error(
                            stack_pointer,
                            "AllocationError",
                            "Failed to allocate float - out of memory".to_string(),
                        );
                    },
                };
                let untagged = BuiltInTypes::untag(new_float_ptr);
                unsafe {
                    let float_ptr = untagged as *mut f64;
                    *float_ptr.add(1) = result;
                }
                new_float_ptr
            }
            Err(e) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "ParseError",
                    format!("Cannot parse '{}' as a float: {}", string, e),
                );
            },
        }
    } else {
        match string.parse::<isize>() {
            Ok(result) => BuiltInTypes::Int.tag(result) as usize,
            Err(e) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "ParseError",
                    format!("Cannot parse '{}' as a number: {}", string, e),
                );
            },
        }
    }
}

pub unsafe extern "C" fn print_value(value: usize) -> usize {
    let runtime = get_runtime().get_mut();
    runtime.print(value);
    0b111
}

pub extern "C" fn print_byte(value: usize) -> usize {
    let byte_value = BuiltInTypes::untag(value) as u8;
    let runtime = get_runtime().get_mut();
    runtime.printer.print_byte(byte_value);
    0b111
}

pub extern "C" fn wait_for_input(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let mut input = String::new();
    match std::io::stdin().read_line(&mut input) {
        Ok(_) => {
            let runtime = get_runtime().get_mut();
            match runtime.allocate_string(stack_pointer, input) {
                Ok(s) => s.into(),
                Err(_) => unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "AllocationError",
                        "Failed to allocate string for input".to_string(),
                    );
                },
            }
        }
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "IOError",
                format!("Failed to read input: {}", e),
            );
        },
    }
}

// Get the ASCII code of the first character of a string
pub extern "C" fn char_code(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let string = runtime.get_string(stack_pointer, string);
    if let Some(ch) = string.chars().next() {
        BuiltInTypes::Int.tag(ch as isize) as usize
    } else {
        // Empty string returns -1
        BuiltInTypes::Int.tag(-1) as usize
    }
}

// Create a single-character string from a Unicode code point
pub extern "C" fn char_from_code(stack_pointer: usize, frame_pointer: usize, code: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let code = BuiltInTypes::untag(code) as u32;
    let ch = match char::from_u32(code) {
        Some(c) => c,
        None => unsafe {
            throw_runtime_error(
                stack_pointer,
                "InvalidArgument",
                format!("Invalid Unicode code point: {}", code),
            );
        },
    };
    let runtime = get_runtime().get_mut();
    match runtime.allocate_string(stack_pointer, ch.to_string()) {
        Ok(ptr) => ptr.into(),
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate character string - out of memory".to_string(),
            );
        },
    }
}

// Read a line from stdin, stripping the trailing newline
// Returns null if EOF is reached
pub extern "C" fn read_line(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let mut input = String::new();
    match std::io::stdin().read_line(&mut input) {
        Ok(0) => {
            // EOF reached
            BuiltInTypes::null_value() as usize
        }
        Ok(_) => {
            // Remove trailing newline if present
            if input.ends_with('\n') {
                input.pop();
                if input.ends_with('\r') {
                    input.pop();
                }
            }
            let runtime = get_runtime().get_mut();
            match runtime.allocate_string(stack_pointer, input) {
                Ok(ptr) => ptr.into(),
                Err(_) => unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "AllocationError",
                        "Failed to allocate string for read_line - out of memory".to_string(),
                    );
                },
            }
        }
        Err(_) => {
            // Error - return null
            BuiltInTypes::null_value() as usize
        }
    }
}

// ---------------------------------------------------------------------------
// Rich REPL readline with editing, history, completion, highlighting, multi-line
// ---------------------------------------------------------------------------

thread_local! {
    static REPL_EDITOR: std::cell::RefCell<Option<(crate::repl::ReplEditor, std::sync::Arc<std::sync::atomic::AtomicUsize>)>> =
        const { std::cell::RefCell::new(None) };
}

pub fn with_repl_editor<F, R>(f: F) -> R
where
    F: FnOnce(&mut crate::repl::ReplEditor, &std::sync::Arc<std::sync::atomic::AtomicUsize>) -> R,
{
    REPL_EDITOR.with(|cell| {
        let mut editor_opt = cell.borrow_mut();

        // Initialize on first call
        if editor_opt.is_none() {
            match crate::repl::create_editor() {
                Ok(editor_and_pw) => {
                    *editor_opt = Some(editor_and_pw);
                }
                Err(e) => {
                    panic!("Failed to create REPL editor: {}", e);
                }
            }
        }

        let (rl, prompt_width) = editor_opt.as_mut().unwrap();
        f(rl, prompt_width)
    })
}

pub extern "C" fn repl_read_line(
    stack_pointer: usize,
    frame_pointer: usize,
    prompt_value: usize,
) -> usize {
    use std::sync::atomic::Ordering;

    save_gc_context!(stack_pointer, frame_pointer);

    // Get prompt string from Beagle value
    let runtime = get_runtime().get_mut();
    let prompt = runtime.get_string(stack_pointer, prompt_value);

    with_repl_editor(|rl, prompt_width| {
        // Refresh completions from runtime
        let runtime = get_runtime().get_mut();
        rl.helper_mut().unwrap().refresh(runtime);

        // Update prompt width for auto-indent
        prompt_width.store(prompt.len(), Ordering::Relaxed);
        rl.helper_mut().unwrap().prompt_width = prompt.len();

        // Register as c_calling before blocking in readline so GC triggered by
        // other threads can proceed without deadlocking on us.
        {
            let runtime = get_runtime().get_mut();
            let thread_state = runtime.thread_state.clone();
            let (lock, condvar) = &*thread_state;
            let mut state = lock.lock().unwrap();
            state.register_c_call(frame_pointer);
            condvar.notify_one();
        }

        let readline_result = rl.readline(&prompt);

        // Unregister from c_calling atomically using gc_lock to prevent a
        // race where GC starts between unregister and our next safepoint.
        // We use yield_now (not park) because the main thread is not in
        // memory.threads and won't be unparked by GC.
        {
            let runtime = get_runtime().get_mut();
            while runtime.is_paused() {
                thread::yield_now();
            }
            loop {
                match runtime.gc_lock.try_lock() {
                    Ok(_guard) => {
                        let thread_state = runtime.thread_state.clone();
                        let (lock, condvar) = &*thread_state;
                        let mut state = lock.lock().unwrap();
                        state.unregister_c_call();
                        condvar.notify_one();
                        break;
                    }
                    Err(_) => thread::yield_now(),
                }
            }
        }

        match readline_result {
            Ok(line) => {
                let runtime = get_runtime().get_mut();
                match runtime.allocate_string(stack_pointer, line) {
                    Ok(ptr) => ptr.into(),
                    Err(_) => unsafe {
                        throw_runtime_error(
                            stack_pointer,
                            "AllocationError",
                            "Failed to allocate string for repl_read_line".to_string(),
                        );
                    },
                }
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                // EOF (Ctrl-D)
                BuiltInTypes::null_value() as usize
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                // Ctrl-C — return special marker string so Beagle can handle it
                let runtime = get_runtime().get_mut();
                match runtime.allocate_string(stack_pointer, ":interrupted".to_string()) {
                    Ok(ptr) => ptr.into(),
                    Err(_) => BuiltInTypes::null_value() as usize,
                }
            }
            Err(_) => BuiltInTypes::null_value() as usize,
        }
    })
}

// Save REPL history to disk
pub extern "C" fn repl_save_history() -> usize {
    with_repl_editor(|rl, _| {
        let hist = crate::repl::history_path();
        let _ = rl.save_history(&hist);
    });
    BuiltInTypes::null_value() as usize
}

pub extern "C" fn read_full_file(
    stack_pointer: usize,
    frame_pointer: usize,
    file_name: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let file_name_str = runtime.get_string(stack_pointer, file_name);
    match std::fs::read_to_string(&file_name_str) {
        Ok(content) => match runtime.allocate_string(stack_pointer, content) {
            Ok(ptr) => ptr.into(),
            Err(_) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate string for file content - out of memory".to_string(),
                );
            },
        },
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "IOError",
                format!("Failed to read file '{}': {}", file_name_str, e),
            );
        },
    }
}

/// Write string content directly to a file (fast path)
/// Returns the number of bytes written, or -1 on error
pub extern "C" fn write_full_file(
    stack_pointer: usize,
    frame_pointer: usize,
    file_name: usize,
    content: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let file_name = runtime.get_string(stack_pointer, file_name);
    let content_str = runtime.get_string(stack_pointer, content);

    match std::fs::write(&file_name, &content_str) {
        Ok(()) => BuiltInTypes::construct_int(content_str.len() as isize) as usize,
        Err(_) => BuiltInTypes::construct_int(-1) as usize,
    }
}
