use super::*;
use crate::save_gc_context;

pub extern "C" fn get_string_index(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    index: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "get_string_index");
    let runtime = get_runtime().get_mut();
    let index = BuiltInTypes::untag(index);

    if BuiltInTypes::get_kind(string) == BuiltInTypes::String {
        let s = runtime.get_str_literal(string);
        let bytes = s.as_bytes();
        // ASCII fast path: O(1) byte indexing, return cached single-char literal
        if s.is_ascii() {
            if index >= bytes.len() {
                unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "IndexError",
                        format!(
                            "String index {} out of bounds (length {})",
                            index,
                            bytes.len()
                        ),
                    );
                }
            }
            let byte = bytes[index];
            return runtime.get_ascii_char_literal(byte);
        }
        // Unicode: use char_indices to find the byte offset in one pass
        let char_count = s.chars().count();
        if index >= char_count {
            unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "IndexError",
                    format!(
                        "String index {} out of bounds (length {})",
                        index, char_count
                    ),
                );
            }
        }
        let ch = s.chars().nth(index).unwrap();
        let result = ch.to_string();
        unsafe { allocate_string_or_throw(runtime, stack_pointer, result) }
    } else {
        let heap_obj = HeapObject::from_tagged(string);
        let header = heap_obj.get_header();

        // Cons string: flatten then index
        if header.type_id == TYPE_ID_CONS_STRING {
            let bytes = runtime.get_string_bytes_vec(string);
            let is_ascii = header.type_flags & 1 != 0;
            if is_ascii {
                if index >= bytes.len() {
                    unsafe {
                        throw_runtime_error(
                            stack_pointer,
                            "IndexError",
                            format!(
                                "String index {} out of bounds (length {})",
                                index,
                                bytes.len()
                            ),
                        );
                    }
                }
                let byte = bytes[index];
                return runtime.get_ascii_char_literal(byte);
            }
            let s = unsafe { std::str::from_utf8_unchecked(&bytes) };
            let char_count = s.chars().count();
            if index >= char_count {
                unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "IndexError",
                        format!(
                            "String index {} out of bounds (length {})",
                            index, char_count
                        ),
                    );
                }
            }
            let ch = s.chars().nth(index).unwrap();
            let result = ch.to_string();
            return unsafe { allocate_string_or_throw(runtime, stack_pointer, result) };
        }

        let bytes = heap_obj.get_string_bytes();
        let is_ascii = if header.type_id == TYPE_ID_STRING_SLICE {
            let parent = HeapObject::from_tagged(heap_obj.get_field(0));
            parent.get_header().type_flags & 1 != 0
        } else {
            header.type_flags & 1 != 0
        };

        if is_ascii {
            if index >= bytes.len() {
                unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "IndexError",
                        format!(
                            "String index {} out of bounds (length {})",
                            index,
                            bytes.len()
                        ),
                    );
                }
            }
            let byte = bytes[index];
            return runtime.get_ascii_char_literal(byte);
        }

        let object_pointer_id = runtime.register_temporary_root(string);
        let s = unsafe { std::str::from_utf8_unchecked(bytes) };
        let char_count = s.chars().count();
        if index >= char_count {
            runtime.unregister_temporary_root(object_pointer_id);
            unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "IndexError",
                    format!(
                        "String index {} out of bounds (length {})",
                        index, char_count
                    ),
                );
            }
        }
        let ch = s.chars().nth(index).unwrap();
        let result_str = ch.to_string();
        let result = match runtime.allocate_string(stack_pointer, result_str) {
            Ok(ptr) => ptr.into(),
            Err(_) => {
                runtime.unregister_temporary_root(object_pointer_id);
                unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "AllocationError",
                        "Failed to allocate character string - out of memory".to_string(),
                    );
                }
            }
        };
        runtime.unregister_temporary_root(object_pointer_id);
        result
    }
}

pub extern "C" fn get_string_length(string: usize) -> usize {
    print_call_builtin(get_runtime().get(), "get_string_length");
    let runtime = get_runtime().get_mut();
    if BuiltInTypes::get_kind(string) == BuiltInTypes::String {
        let s = runtime.get_str_literal(string);
        if s.is_ascii() {
            return BuiltInTypes::Int.tag(s.len() as isize) as usize;
        }
        BuiltInTypes::Int.tag(s.chars().count() as isize) as usize
    } else {
        let heap_obj = HeapObject::from_tagged(string);
        let header = heap_obj.get_header();

        // Cons string: use type_data for ASCII length, flatten for Unicode
        if header.type_id == TYPE_ID_CONS_STRING {
            let is_ascii = header.type_flags & 1 != 0;
            if is_ascii {
                return BuiltInTypes::Int.tag(header.type_data as isize) as usize;
            }
            let bytes = runtime.get_string_bytes_vec(string);
            let s = unsafe { std::str::from_utf8_unchecked(&bytes) };
            return BuiltInTypes::Int.tag(s.chars().count() as isize) as usize;
        }

        let bytes = heap_obj.get_string_bytes();
        let is_ascii = if header.type_id == TYPE_ID_STRING_SLICE {
            let parent = HeapObject::from_tagged(heap_obj.get_field(0));
            parent.get_header().type_flags & 1 != 0
        } else {
            header.type_flags & 1 != 0
        };
        if is_ascii {
            return BuiltInTypes::Int.tag(bytes.len() as isize) as usize;
        }
        let s = unsafe { std::str::from_utf8_unchecked(bytes) };
        BuiltInTypes::Int.tag(s.chars().count() as isize) as usize
    }
}

pub extern "C" fn string_concat(
    stack_pointer: usize,
    frame_pointer: usize,
    a: usize,
    b: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "string_concat");
    let runtime = get_runtime().get_mut();

    // If either argument is not a string, fall back to converting both to strings
    let a_is_string = is_string_like(a);
    let b_is_string = is_string_like(b);
    if !a_is_string || !b_is_string {
        let a_str = runtime.get_string(stack_pointer, a);
        let b_str = runtime.get_string(stack_pointer, b);
        let result = a_str + &b_str;
        return unsafe { allocate_string_or_throw(runtime, stack_pointer, result) };
    }

    let a_len = runtime.get_string_byte_length(a);
    let b_len = runtime.get_string_byte_length(b);

    // If either side is empty, return the other (O(1))
    if a_len == 0 {
        return b;
    }
    if b_len == 0 {
        return a;
    }

    let total_len = a_len + b_len;

    // For small strings (<= 128 bytes), create a flat string directly
    if total_len <= 128 {
        let a_bytes = runtime.get_string_bytes_vec(a);
        let b_bytes = runtime.get_string_bytes_vec(b);
        let mut result = Vec::with_capacity(total_len);
        result.extend_from_slice(&a_bytes);
        result.extend_from_slice(&b_bytes);
        return match runtime.allocate_string_from_bytes(stack_pointer, &result) {
            Ok(ptr) => ptr.into(),
            Err(_) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate concatenated string - out of memory".to_string(),
                );
            },
        };
    }

    // For larger strings, create a cons string node (O(1))
    match runtime.allocate_cons_string(stack_pointer, a, b) {
        Ok(ptr) => ptr.into(),
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate cons string - out of memory".to_string(),
            );
        },
    }
}

pub extern "C" fn substring(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    start: usize,
    end: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "substring");
    let runtime = get_runtime().get_mut();
    let string_pointer = runtime.register_temporary_root(string);
    let start = BuiltInTypes::untag(start);
    let end = BuiltInTypes::untag(end);
    if end < start {
        unsafe {
            runtime.unregister_temporary_root(string_pointer);
            throw_runtime_error(
                stack_pointer,
                "IndexError",
                format!("substring end ({}) is less than start ({})", end, start),
            );
        }
    }
    let length = end - start;
    let result = match runtime.get_substring(stack_pointer, string, start, length) {
        Ok(s) => s.into(),
        Err(e) => unsafe {
            runtime.unregister_temporary_root(string_pointer);
            throw_runtime_error(
                stack_pointer,
                "StringError",
                format!("substring failed: {}", e),
            );
        },
    };
    runtime.unregister_temporary_root(string_pointer);
    result
}

pub extern "C" fn uppercase(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "uppercase");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let uppercased = string_value.to_uppercase();
    unsafe { allocate_string_or_throw(runtime, stack_pointer, uppercased) }
}

pub extern "C" fn lowercase(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "lowercase");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let lowercased = string_value.to_lowercase();
    unsafe { allocate_string_or_throw(runtime, stack_pointer, lowercased) }
}

pub extern "C" fn split(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    delimiter: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "split");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let delimiter_value = runtime.get_string(stack_pointer, delimiter);

    let parts: Vec<String> = string_value
        .split(&delimiter_value)
        .map(|s| s.to_string())
        .collect();

    match runtime.create_string_array(stack_pointer, &parts) {
        Ok(arr) => arr,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate string array - out of memory".to_string(),
            );
        },
    }
}

// join() is now implemented in std.bg using Indexed and Length protocols

pub extern "C" fn trim(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "trim");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let trimmed = string_value.trim();
    unsafe { allocate_string_or_throw(runtime, stack_pointer, trimmed.to_string()) }
}

pub extern "C" fn trim_left(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "trim-left");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let trimmed = string_value.trim_start();
    unsafe { allocate_string_or_throw(runtime, stack_pointer, trimmed.to_string()) }
}

pub extern "C" fn trim_right(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "trim-right");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let trimmed = string_value.trim_end();
    unsafe { allocate_string_or_throw(runtime, stack_pointer, trimmed.to_string()) }
}

pub extern "C" fn starts_with(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    prefix: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "starts-with?");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let prefix_value = runtime.get_string(stack_pointer, prefix);
    let result = string_value.starts_with(&prefix_value);
    BuiltInTypes::construct_boolean(result) as usize
}

pub extern "C" fn ends_with(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    suffix: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "ends-with?");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let suffix_value = runtime.get_string(stack_pointer, suffix);
    let result = string_value.ends_with(&suffix_value);
    BuiltInTypes::construct_boolean(result) as usize
}

pub extern "C" fn string_contains(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    substr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "contains?");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let substr_value = runtime.get_string(stack_pointer, substr);
    let result = string_value.contains(&substr_value);
    BuiltInTypes::construct_boolean(result) as usize
}

pub extern "C" fn index_of(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    substr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "index-of");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let substr_value = runtime.get_string(stack_pointer, substr);

    match string_value.find(&substr_value) {
        Some(byte_index) => {
            if string_value.is_ascii() {
                BuiltInTypes::construct_int(byte_index as isize) as usize
            } else {
                let char_index = string_value[..byte_index].chars().count();
                BuiltInTypes::construct_int(char_index as isize) as usize
            }
        }
        None => BuiltInTypes::construct_int(-1) as usize,
    }
}

pub extern "C" fn last_index_of(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    substr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "last-index-of");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let substr_value = runtime.get_string(stack_pointer, substr);

    match string_value.rfind(&substr_value) {
        Some(byte_index) => {
            if string_value.is_ascii() {
                BuiltInTypes::construct_int(byte_index as isize) as usize
            } else {
                let char_index = string_value[..byte_index].chars().count();
                BuiltInTypes::construct_int(char_index as isize) as usize
            }
        }
        None => BuiltInTypes::construct_int(-1) as usize,
    }
}

pub extern "C" fn replace_string(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    from: usize,
    to: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "replace");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let from_value = runtime.get_string(stack_pointer, from);
    let to_value = runtime.get_string(stack_pointer, to);

    let replaced = string_value.replace(&from_value, &to_value);

    unsafe { allocate_string_or_throw(runtime, stack_pointer, replaced) }
}

pub extern "C" fn blank_string(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "blank?");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);

    // A string is blank if it's empty or contains only whitespace
    let is_blank = string_value.trim().is_empty();

    BuiltInTypes::construct_boolean(is_blank) as usize
}

pub extern "C" fn replace_first_string(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    from: usize,
    to: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "replace-first");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let from_value = runtime.get_string(stack_pointer, from);
    let to_value = runtime.get_string(stack_pointer, to);

    let replaced = string_value.replacen(&from_value, &to_value, 1);

    unsafe { allocate_string_or_throw(runtime, stack_pointer, replaced) }
}

pub extern "C" fn pad_left_string(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    width: usize,
    pad_char: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "pad-left");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let width_value = BuiltInTypes::untag_isize(width as isize) as usize;
    let pad_char_value = runtime.get_string(stack_pointer, pad_char);

    // Get the first character from the pad string, or use space if empty
    let pad_ch = pad_char_value.chars().next().unwrap_or(' ');

    let current_len = string_value.chars().count();
    if current_len >= width_value {
        // Already at or exceeds desired width, return as-is
        return string;
    }

    let padding_needed = width_value - current_len;
    let padded = format!(
        "{}{}",
        pad_ch.to_string().repeat(padding_needed),
        string_value
    );

    unsafe { allocate_string_or_throw(runtime, stack_pointer, padded) }
}

pub extern "C" fn pad_right_string(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    width: usize,
    pad_char: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "pad-right");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let width_value = BuiltInTypes::untag_isize(width as isize) as usize;
    let pad_char_value = runtime.get_string(stack_pointer, pad_char);

    // Get the first character from the pad string, or use space if empty
    let pad_ch = pad_char_value.chars().next().unwrap_or(' ');

    let current_len = string_value.chars().count();
    if current_len >= width_value {
        // Already at or exceeds desired width, return as-is
        return string;
    }

    let padding_needed = width_value - current_len;
    let padded = format!(
        "{}{}",
        string_value,
        pad_ch.to_string().repeat(padding_needed)
    );

    unsafe { allocate_string_or_throw(runtime, stack_pointer, padded) }
}

pub extern "C" fn lines_string(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "lines");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);

    let line_vec: Vec<String> = string_value.lines().map(|s| s.to_string()).collect();

    match runtime.create_string_array(stack_pointer, &line_vec) {
        Ok(arr) => arr,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate string array - out of memory".to_string(),
            );
        },
    }
}

pub extern "C" fn words_string(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "words");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);

    let word_vec: Vec<String> = string_value
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();

    match runtime.create_string_array(stack_pointer, &word_vec) {
        Ok(arr) => arr,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate string array - out of memory".to_string(),
            );
        },
    }
}
