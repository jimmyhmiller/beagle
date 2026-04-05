use super::*;
use crate::collections::TYPE_ID_REGEX;
use crate::save_gc_context;
use ::regex::Regex;

/// Compile a regex pattern
/// Signature: (stack_pointer, frame_pointer, pattern) -> regex_handle
pub unsafe extern "C" fn regex_compile(
    stack_pointer: usize,
    frame_pointer: usize,
    pattern: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Get the pattern string
    let pattern_str = runtime.get_string(stack_pointer, pattern);

    // Try to compile the regex
    match Regex::new(&pattern_str) {
        Ok(regex) => {
            // Store the compiled regex and get its index
            let index = runtime.compiled_regexes.len();
            runtime.compiled_regexes.push(regex);

            // Allocate a heap object to hold the index
            // The object has 1 field: the index into compiled_regexes
            match runtime.allocate(1, stack_pointer, BuiltInTypes::HeapObject) {
                Ok(ptr) => {
                    let mut heap_obj = HeapObject::from_tagged(ptr);
                    heap_obj.write_type_id(TYPE_ID_REGEX as usize);
                    heap_obj.write_field(0, BuiltInTypes::Int.tag(index as isize) as usize);
                    ptr
                }
                Err(e) => unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "AllocationError",
                        format!("Failed to allocate regex object: {}", e),
                    );
                },
            }
        }
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "RegexError",
                format!("Invalid regex pattern '{}': {}", pattern_str, e),
            );
        },
    }
}

/// Get the regex from a heap object
fn get_regex(runtime: &Runtime, regex_ptr: usize) -> Option<&Regex> {
    if !BuiltInTypes::is_heap_pointer(regex_ptr) {
        return None;
    }
    let heap_obj = HeapObject::from_tagged(regex_ptr);
    if heap_obj.get_type_id() != TYPE_ID_REGEX as usize {
        return None;
    }
    let index = BuiltInTypes::untag(heap_obj.get_field(0));
    runtime.compiled_regexes.get(index)
}

/// Check if string matches regex
/// Signature: (regex, string) -> bool
pub unsafe extern "C" fn regex_matches(regex_ptr: usize, string: usize) -> usize {
    let runtime = get_runtime().get();

    let Some(regex) = get_regex(runtime, regex_ptr) else {
        return BuiltInTypes::false_value() as usize;
    };

    // Get the string - but we can't call get_string without stack_pointer
    // So we need to handle this differently
    let string_str = get_string_for_regex(runtime, string);
    let result = regex.is_match(&string_str);
    BuiltInTypes::construct_boolean(result) as usize
}

/// Helper to get a string without throwing (for non-allocating functions)
fn get_string_for_regex(runtime: &Runtime, value: usize) -> String {
    let tag = BuiltInTypes::get_kind(value);
    if tag == BuiltInTypes::String {
        runtime.get_string_literal(value)
    } else if tag == BuiltInTypes::HeapObject {
        let heap_object = HeapObject::from_tagged(value);
        let tid = heap_object.get_type_id();
        if tid != TYPE_ID_STRING as usize
            && tid != TYPE_ID_STRING_SLICE as usize
            && tid != TYPE_ID_CONS_STRING as usize
        {
            return String::new();
        }
        let bytes = runtime.get_string_bytes_vec(value);
        unsafe { String::from_utf8_unchecked(bytes) }
    } else {
        String::new()
    }
}

/// Find first match in string
/// Signature: (stack_pointer, frame_pointer, regex, string) -> match map or null
pub unsafe extern "C" fn regex_find(
    stack_pointer: usize,
    frame_pointer: usize,
    regex_ptr: usize,
    string: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let Some(regex) = get_regex(runtime, regex_ptr) else {
        return BuiltInTypes::null_value() as usize;
    };

    let string_str = runtime.get_string(stack_pointer, string);

    match regex.find(&string_str) {
        Some(m) => {
            // Return the matched string
            let matched = m.as_str().to_string();
            match runtime.allocate_string(stack_pointer, matched) {
                Ok(ptr) => ptr.into(),
                Err(_) => BuiltInTypes::null_value() as usize,
            }
        }
        None => BuiltInTypes::null_value() as usize,
    }
}

/// Find all matches in string
/// Signature: (stack_pointer, frame_pointer, regex, string) -> vector of strings
pub unsafe extern "C" fn regex_find_all(
    stack_pointer: usize,
    frame_pointer: usize,
    regex_ptr: usize,
    string: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let Some(regex) = get_regex(runtime, regex_ptr) else {
        return BuiltInTypes::null_value() as usize;
    };

    let string_str = runtime.get_string(stack_pointer, string);

    // Collect all matches
    let matches: Vec<String> = regex
        .find_iter(&string_str)
        .map(|m| m.as_str().to_string())
        .collect();

    // Create a PersistentVec with all matches
    use crate::collections::PersistentVec;

    let mut vec = match PersistentVec::empty(runtime, stack_pointer) {
        Ok(v) => v,
        Err(_) => return BuiltInTypes::null_value() as usize,
    };

    for matched in matches {
        let str_ptr = match runtime.allocate_string(stack_pointer, matched) {
            Ok(ptr) => ptr.into(),
            Err(_) => return BuiltInTypes::null_value() as usize,
        };
        vec = match PersistentVec::push(runtime, stack_pointer, vec, str_ptr) {
            Ok(v) => v,
            Err(_) => return BuiltInTypes::null_value() as usize,
        };
    }

    vec.as_tagged()
}

/// Replace first match
/// Signature: (stack_pointer, frame_pointer, regex, string, replacement) -> new_string
pub unsafe extern "C" fn regex_replace(
    stack_pointer: usize,
    frame_pointer: usize,
    regex_ptr: usize,
    string: usize,
    replacement: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let Some(regex) = get_regex(runtime, regex_ptr) else {
        return string; // Return original string if not a valid regex
    };

    let string_str = runtime.get_string(stack_pointer, string);
    let replacement_str = runtime.get_string(stack_pointer, replacement);

    let result = regex
        .replace(&string_str, replacement_str.as_str())
        .to_string();

    match runtime.allocate_string(stack_pointer, result) {
        Ok(ptr) => ptr.into(),
        Err(_) => BuiltInTypes::null_value() as usize,
    }
}

/// Replace all matches
/// Signature: (stack_pointer, frame_pointer, regex, string, replacement) -> new_string
pub unsafe extern "C" fn regex_replace_all(
    stack_pointer: usize,
    frame_pointer: usize,
    regex_ptr: usize,
    string: usize,
    replacement: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let Some(regex) = get_regex(runtime, regex_ptr) else {
        return string; // Return original string if not a valid regex
    };

    let string_str = runtime.get_string(stack_pointer, string);
    let replacement_str = runtime.get_string(stack_pointer, replacement);

    let result = regex
        .replace_all(&string_str, replacement_str.as_str())
        .to_string();

    match runtime.allocate_string(stack_pointer, result) {
        Ok(ptr) => ptr.into(),
        Err(_) => BuiltInTypes::null_value() as usize,
    }
}

/// Split string by regex
/// Signature: (stack_pointer, frame_pointer, regex, string) -> vector of strings
pub unsafe extern "C" fn regex_split(
    stack_pointer: usize,
    frame_pointer: usize,
    regex_ptr: usize,
    string: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let Some(regex) = get_regex(runtime, regex_ptr) else {
        return BuiltInTypes::null_value() as usize;
    };

    let string_str = runtime.get_string(stack_pointer, string);

    // Split the string
    let parts: Vec<&str> = regex.split(&string_str).collect();

    // Create a PersistentVec with all parts
    use crate::collections::PersistentVec;

    let mut vec = match PersistentVec::empty(runtime, stack_pointer) {
        Ok(v) => v,
        Err(_) => return BuiltInTypes::null_value() as usize,
    };

    for part in parts {
        let str_ptr = match runtime.allocate_string(stack_pointer, part.to_string()) {
            Ok(ptr) => ptr.into(),
            Err(_) => return BuiltInTypes::null_value() as usize,
        };
        vec = match PersistentVec::push(runtime, stack_pointer, vec, str_ptr) {
            Ok(v) => v,
            Err(_) => return BuiltInTypes::null_value() as usize,
        };
    }

    vec.as_tagged()
}

/// Get capture groups from first match
/// Signature: (stack_pointer, frame_pointer, regex, string) -> vector of strings or null
pub unsafe extern "C" fn regex_captures(
    stack_pointer: usize,
    frame_pointer: usize,
    regex_ptr: usize,
    string: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let Some(regex) = get_regex(runtime, regex_ptr) else {
        return BuiltInTypes::null_value() as usize;
    };

    let string_str = runtime.get_string(stack_pointer, string);

    match regex.captures(&string_str) {
        Some(caps) => {
            use crate::collections::PersistentVec;

            let mut vec = match PersistentVec::empty(runtime, stack_pointer) {
                Ok(v) => v,
                Err(_) => return BuiltInTypes::null_value() as usize,
            };

            for i in 0..caps.len() {
                let value = match caps.get(i) {
                    Some(m) => {
                        match runtime.allocate_string(stack_pointer, m.as_str().to_string()) {
                            Ok(ptr) => ptr.into(),
                            Err(_) => return BuiltInTypes::null_value() as usize,
                        }
                    }
                    None => BuiltInTypes::null_value() as usize,
                };
                vec = match PersistentVec::push(runtime, stack_pointer, vec, value) {
                    Ok(v) => v,
                    Err(_) => return BuiltInTypes::null_value() as usize,
                };
            }

            vec.as_tagged()
        }
        None => BuiltInTypes::null_value() as usize,
    }
}

/// Check if value is a regex
/// Signature: (value) -> bool
pub unsafe extern "C" fn is_regex(value: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(value) {
        return BuiltInTypes::false_value() as usize;
    }
    let heap_obj = HeapObject::from_tagged(value);
    let is_regex = heap_obj.get_type_id() == TYPE_ID_REGEX as usize;
    BuiltInTypes::construct_boolean(is_regex) as usize
}
