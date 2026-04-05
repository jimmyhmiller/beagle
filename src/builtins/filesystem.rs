use super::*;
use crate::save_gc_context;

pub extern "C" fn fs_unlink(stack_pointer: usize, frame_pointer: usize, path: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let c_path = match std::ffi::CString::new(path_str) {
        Ok(s) => s,
        Err(_) => return BuiltInTypes::construct_int(-1) as usize,
    };

    let result = unsafe { libc::unlink(c_path.as_ptr()) };
    BuiltInTypes::construct_int(result as isize) as usize
}

/// access builtin - Check file accessibility
/// mode: 0 = F_OK (existence), 1 = X_OK (execute), 2 = W_OK (write), 4 = R_OK (read)
/// Returns 0 on success, -1 on error
pub extern "C" fn fs_access(
    stack_pointer: usize,
    frame_pointer: usize,
    path: usize,
    mode: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);
    let mode_val = BuiltInTypes::untag(mode) as i32;

    let c_path = match std::ffi::CString::new(path_str) {
        Ok(s) => s,
        Err(_) => return BuiltInTypes::construct_int(-1) as usize,
    };

    let result = unsafe { libc::access(c_path.as_ptr(), mode_val) };
    BuiltInTypes::construct_int(result as isize) as usize
}

/// mkdir builtin - Create a directory
/// mode: permission bits (e.g., 0o755 = 493)
/// Returns 0 on success, negative errno on error
pub extern "C" fn fs_mkdir(
    stack_pointer: usize,
    frame_pointer: usize,
    path: usize,
    mode: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);
    let mode_val = BuiltInTypes::untag(mode) as libc::mode_t;

    let c_path = match std::ffi::CString::new(path_str) {
        Ok(s) => s,
        Err(_) => return BuiltInTypes::construct_int(-1) as usize,
    };

    let result = unsafe { libc::mkdir(c_path.as_ptr(), mode_val) };
    if result == 0 {
        BuiltInTypes::construct_int(0) as usize
    } else {
        // Return negative errno
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(-1);
        BuiltInTypes::construct_int(-errno as isize) as usize
    }
}

/// rmdir builtin - Remove an empty directory
/// Returns 0 on success, -1 on error
pub extern "C" fn fs_rmdir(stack_pointer: usize, frame_pointer: usize, path: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    let c_path = match std::ffi::CString::new(path_str) {
        Ok(s) => s,
        Err(_) => return BuiltInTypes::construct_int(-1) as usize,
    };

    let result = unsafe { libc::rmdir(c_path.as_ptr()) };
    BuiltInTypes::construct_int(result as isize) as usize
}

/// rename builtin - Rename/move a file or directory
/// Returns 0 on success, -1 on error
pub extern "C" fn fs_rename(
    stack_pointer: usize,
    frame_pointer: usize,
    old_path: usize,
    new_path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let old_path_str = runtime.get_string(stack_pointer, old_path);
    let new_path_str = runtime.get_string(stack_pointer, new_path);

    let c_old_path = match std::ffi::CString::new(old_path_str) {
        Ok(s) => s,
        Err(_) => return BuiltInTypes::construct_int(-1) as usize,
    };

    let c_new_path = match std::ffi::CString::new(new_path_str) {
        Ok(s) => s,
        Err(_) => return BuiltInTypes::construct_int(-1) as usize,
    };

    let result = unsafe { libc::rename(c_old_path.as_ptr(), c_new_path.as_ptr()) };
    BuiltInTypes::construct_int(result as isize) as usize
}

/// is_directory builtin - Check if path is a directory
/// Returns true if directory, false otherwise
pub extern "C" fn fs_is_directory(
    stack_pointer: usize,
    frame_pointer: usize,
    path: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    match std::fs::metadata(&path_str) {
        Ok(metadata) => {
            if metadata.is_dir() {
                BuiltInTypes::true_value() as usize
            } else {
                BuiltInTypes::false_value() as usize
            }
        }
        Err(_) => BuiltInTypes::false_value() as usize,
    }
}

/// is_file builtin - Check if path is a regular file
/// Returns true if file, false otherwise
pub extern "C" fn fs_is_file(stack_pointer: usize, frame_pointer: usize, path: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    match std::fs::metadata(&path_str) {
        Ok(metadata) => {
            if metadata.is_file() {
                BuiltInTypes::true_value() as usize
            } else {
                BuiltInTypes::false_value() as usize
            }
        }
        Err(_) => BuiltInTypes::false_value() as usize,
    }
}

/// readdir builtin - List directory contents
/// Returns an array of strings (filenames), or null on error
pub extern "C" fn fs_readdir(stack_pointer: usize, frame_pointer: usize, path: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    // Use std::fs::read_dir for safer directory reading
    match std::fs::read_dir(&path_str) {
        Ok(entries) => {
            let mut filenames: Vec<String> = Vec::new();
            for entry in entries {
                if let Ok(entry) = entry
                    && let Some(name) = entry.file_name().to_str()
                {
                    filenames.push(name.to_string());
                }
            }
            // Create a Beagle array of strings
            match runtime.create_string_array(stack_pointer, &filenames) {
                Ok(ptr) => ptr,
                Err(_) => unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "AllocationError",
                        "Failed to allocate string array for directory listing - out of memory"
                            .to_string(),
                    );
                },
            }
        }
        Err(_) => BuiltInTypes::null_value() as usize,
    }
}

/// file_size builtin - Get file size in bytes
/// Returns file size on success, -1 on error
pub extern "C" fn fs_file_size(stack_pointer: usize, frame_pointer: usize, path: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let path_str = runtime.get_string(stack_pointer, path);

    match std::fs::metadata(&path_str) {
        Ok(metadata) => {
            let size = metadata.len() as isize;
            BuiltInTypes::construct_int(size) as usize
        }
        Err(_) => BuiltInTypes::construct_int(-1) as usize,
    }
}
