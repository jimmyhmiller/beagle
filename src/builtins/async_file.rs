use super::*;
use crate::save_gc_context;

/// Submit a file read operation, returns the operation handle
/// Returns -1 on error
pub extern "C" fn file_read_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::Read { path: path_str }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file write operation, returns the operation handle
/// Returns -1 on error
pub extern "C" fn file_write_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
    content: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);
    let content_str = runtime.get_string(stack_pointer, content);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::Write {
        path: path_str,
        content: content_str.into_bytes(),
    }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file delete operation, returns the operation handle
/// Returns -1 on error
pub extern "C" fn file_delete_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::Delete { path: path_str }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file stat operation, returns the operation handle
/// Returns -1 on error
pub extern "C" fn file_stat_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::Stat { path: path_str }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a readdir operation, returns the operation handle
/// Returns -1 on error
pub extern "C" fn file_readdir_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::ReadDir { path: path_str }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file append operation, returns the operation handle
pub extern "C" fn file_append_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
    content: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);
    let content_str = runtime.get_string(stack_pointer, content);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::Append {
        path: path_str,
        content: content_str.into_bytes(),
    }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file exists check, returns the operation handle
pub extern "C" fn file_exists_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::Exists { path: path_str }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file rename operation, returns the operation handle
pub extern "C" fn file_rename_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    old_path: usize,
    new_path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let old_path_str = runtime.get_string(stack_pointer, old_path);
    let new_path_str = runtime.get_string(stack_pointer, new_path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::Rename {
        old_path: old_path_str,
        new_path: new_path_str,
    }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file copy operation, returns the operation handle
pub extern "C" fn file_copy_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    src_path: usize,
    dest_path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let src_path_str = runtime.get_string(stack_pointer, src_path);
    let dest_path_str = runtime.get_string(stack_pointer, dest_path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::Copy {
        src_path: src_path_str,
        dest_path: dest_path_str,
    }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a mkdir operation, returns the operation handle
pub extern "C" fn file_mkdir_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::Mkdir { path: path_str }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a mkdir-all operation, returns the operation handle
pub extern "C" fn file_mkdir_all_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::MkdirAll { path: path_str }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a rmdir operation, returns the operation handle
pub extern "C" fn file_rmdir_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::Rmdir { path: path_str }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a rmdir-all operation, returns the operation handle
pub extern "C" fn file_rmdir_all_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::RmdirAll { path: path_str }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit an is-directory check, returns the operation handle
pub extern "C" fn file_is_dir_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::IsDir { path: path_str }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit an is-file check, returns the operation handle
pub extern "C" fn file_is_file_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::IsFile { path: path_str }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file open operation, returns the operation handle
pub extern "C" fn file_open_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    path: usize,
    mode: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);
    let mode_str = runtime.get_string(stack_pointer, mode);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::FileOpen {
        path: path_str,
        mode: mode_str,
    }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file close operation, returns the operation handle
pub extern "C" fn file_close_submit(loop_id: usize, handle_key: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let handle_key = BuiltInTypes::untag(handle_key) as u64;
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::FileClose { handle_key }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file handle read operation, returns the operation handle
pub extern "C" fn file_handle_read_submit(
    loop_id: usize,
    handle_key: usize,
    count: usize,
) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let handle_key = BuiltInTypes::untag(handle_key) as u64;
    let count = BuiltInTypes::untag(count);
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop
        .submit_file_op(crate::runtime::FileOperation::FileHandleRead { handle_key, count })
    {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file handle write operation, returns the operation handle
pub extern "C" fn file_handle_write_submit(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    handle_key: usize,
    content: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let handle_key = BuiltInTypes::untag(handle_key) as u64;
    let runtime = get_runtime().get_mut();
    let content_str = runtime.get_string(stack_pointer, content);

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::FileHandleWrite {
        handle_key,
        content: content_str.into_bytes(),
    }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file handle readline operation, returns the operation handle
pub extern "C" fn file_handle_readline_submit(loop_id: usize, handle_key: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let handle_key = BuiltInTypes::untag(handle_key) as u64;
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop
        .submit_file_op(crate::runtime::FileOperation::FileHandleReadLine { handle_key })
    {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Submit a file handle flush operation, returns the operation handle
pub extern "C" fn file_handle_flush_submit(loop_id: usize, handle_key: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let handle_key = BuiltInTypes::untag(handle_key) as u64;
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    match event_loop.submit_file_op(crate::runtime::FileOperation::FileHandleFlush { handle_key }) {
        Ok(handle) => BuiltInTypes::Int.tag(handle as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Get the count of completed file operations
pub extern "C" fn file_results_count(loop_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };

    BuiltInTypes::Int.tag(event_loop.file_results_count() as isize) as usize
}

/// Check if a result is ready for the given handle (without consuming it)
/// Returns true/false
pub extern "C" fn file_result_ready(loop_id: usize, handle: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let handle = BuiltInTypes::untag(handle) as u64;
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::construct_boolean(false) as usize,
    };

    BuiltInTypes::construct_boolean(event_loop.file_result_ready(handle)) as usize
}

/// Poll for a file result type by handle
/// Returns the type code if ready, 0 if not ready
/// After calling this, use file_result_get_* functions to get the data
/// 0 = not ready, 1 = ReadOk, 2 = ReadErr, 3 = WriteOk, 4 = WriteErr,
/// 5 = DeleteOk, 6 = DeleteErr, 7 = StatOk, 8 = StatErr, 9 = ReadDirOk, 10 = ReadDirErr
pub extern "C" fn file_result_poll_type(loop_id: usize, handle: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let handle = BuiltInTypes::untag(handle) as u64;
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };

    // Use the peek method to check without removing
    match event_loop.file_result_peek_type(handle) {
        Some(type_code) => BuiltInTypes::Int.tag(type_code as isize) as usize,
        None => BuiltInTypes::Int.tag(0) as usize,
    }
}

/// Get the string value from a completed file result (for ReadOk or any error)
/// Removes the result from the map
pub extern "C" fn file_result_get_string(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    handle: usize,
) -> usize {
    use crate::runtime::EventLoop;

    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let handle = BuiltInTypes::untag(handle) as u64;
    let runtime = get_runtime().get_mut();

    // Get and remove the result
    let result_data = {
        let event_loop = match runtime.event_loops.get(loop_id) {
            Some(el) => el,
            None => return BuiltInTypes::null_value() as usize,
        };
        event_loop.file_result_poll(handle)
    };

    let data = match result_data {
        Some(d) => d,
        None => return BuiltInTypes::null_value() as usize,
    };

    // Get the string data
    match EventLoop::file_result_string_data(&data) {
        Some(s) => runtime
            .allocate_string(stack_pointer, s.to_string())
            .map(|t| t.into())
            .unwrap_or(BuiltInTypes::null_value() as usize),
        None => BuiltInTypes::null_value() as usize,
    }
}

/// Get the numeric value from a completed file result (for WriteOk or StatOk)
/// Removes the result from the map
pub extern "C" fn file_result_get_value(loop_id: usize, handle: usize) -> usize {
    use crate::runtime::EventLoop;

    let loop_id = BuiltInTypes::untag(loop_id);
    let handle = BuiltInTypes::untag(handle) as u64;
    let runtime = get_runtime().get_mut();

    // Get and remove the result
    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };

    match event_loop.file_result_poll(handle) {
        Some(data) => {
            let value = EventLoop::file_result_value(&data);
            BuiltInTypes::Int.tag(value as isize) as usize
        }
        None => BuiltInTypes::Int.tag(0) as usize,
    }
}

/// Consume (remove) a file result by handle without getting its data
/// Use this for DeleteOk which has no data
pub extern "C" fn file_result_consume(loop_id: usize, handle: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let handle = BuiltInTypes::untag(handle) as u64;
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::construct_boolean(false) as usize,
    };

    let was_present = event_loop.file_result_poll(handle).is_some();
    BuiltInTypes::construct_boolean(was_present) as usize
}

/// Get directory entries from a ReadDirOk result
/// Returns a PersistentVec of strings, or null if not ready/wrong type
pub extern "C" fn file_result_get_entries(
    stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    handle: usize,
) -> usize {
    use crate::runtime::FileResultData;

    save_gc_context!(stack_pointer, frame_pointer);
    let loop_id = BuiltInTypes::untag(loop_id);
    let handle = BuiltInTypes::untag(handle) as u64;
    let runtime = get_runtime().get_mut();

    // Get and remove the result
    let result_data = {
        let event_loop = match runtime.event_loops.get(loop_id) {
            Some(el) => el,
            None => return BuiltInTypes::null_value() as usize,
        };
        event_loop.file_result_poll(handle)
    };

    let entries = match result_data {
        Some(FileResultData::ReadDirOk { entries }) => entries,
        _ => return BuiltInTypes::null_value() as usize,
    };

    // Create a PersistentVec from the entries
    // Start with an empty vector and push each string
    let mut vec: usize = match PersistentVec::empty(runtime, stack_pointer) {
        Ok(h) => h.as_tagged(),
        Err(_) => return BuiltInTypes::null_value() as usize,
    };

    // Register the vector as a temporary root so GC doesn't collect it
    let mut vec_root_id = runtime.register_temporary_root(vec);

    for entry in entries {
        // Allocate the string
        let string_tagged: usize = match runtime.allocate_string(stack_pointer, entry) {
            Ok(t) => t.into(),
            Err(_) => {
                runtime.unregister_temporary_root(vec_root_id);
                return BuiltInTypes::null_value() as usize;
            }
        };

        // Register string as root during push
        let string_root_id = runtime.register_temporary_root(string_tagged);

        // Get updated vec pointer (GC may have moved it)
        let vec_updated = runtime.peek_temporary_root(vec_root_id);
        let string_updated = runtime.peek_temporary_root(string_root_id);

        let vec_handle = GcHandle::from_tagged(vec_updated);

        // Push it onto the vector
        match PersistentVec::push(runtime, stack_pointer, vec_handle, string_updated) {
            Ok(new_vec) => {
                vec = new_vec.as_tagged();
            }
            Err(_) => {
                runtime.unregister_temporary_root(string_root_id);
                runtime.unregister_temporary_root(vec_root_id);
                return BuiltInTypes::null_value() as usize;
            }
        };

        runtime.unregister_temporary_root(string_root_id);
        runtime.unregister_temporary_root(vec_root_id);
        vec_root_id = runtime.register_temporary_root(vec);
    }

    runtime.unregister_temporary_root(vec_root_id);
    vec
}
