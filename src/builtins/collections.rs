use super::*;
use crate::collections::{GcHandle, MutableMap, PersistentMap, PersistentSet, PersistentVec};
use crate::save_gc_context;

/// Create an empty persistent vector
/// Signature: (stack_pointer, frame_pointer) -> tagged_ptr
pub unsafe extern "C" fn rust_vec_empty(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    match PersistentVec::empty(runtime, stack_pointer) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to create empty vector: {}", e),
            );
        },
    }
}

/// Get the count of a persistent vector
/// Signature: (vec_ptr) -> tagged_int
pub unsafe extern "C" fn rust_vec_count(vec_ptr: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(vec_ptr) {
        return BuiltInTypes::construct_int(0) as usize;
    }
    let vec = GcHandle::from_tagged(vec_ptr);
    let count = PersistentVec::count(vec);
    BuiltInTypes::construct_int(count as isize) as usize
}

/// Get a value from a persistent vector by index
/// Signature: (vec_ptr, index) -> tagged_value
pub unsafe extern "C" fn rust_vec_get(vec_ptr: usize, index: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(vec_ptr) {
        return BuiltInTypes::null_value() as usize;
    }
    // Validate that the index is an integer
    if BuiltInTypes::get_kind(index) != BuiltInTypes::Int {
        return BuiltInTypes::null_value() as usize;
    }
    let vec = GcHandle::from_tagged(vec_ptr);
    let idx = BuiltInTypes::untag(index);
    if let Ok(path) = std::env::var("BEAGLE_DEBUG_VEC_GET_FILE")
        && let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
    {
        use std::io::Write;
        let count = vec.get_field(0);
        let shift = vec.get_field(1);
        let root = vec.get_field(2);
        let tail = vec.get_field(3);
        let tail_fields = if BuiltInTypes::is_heap_pointer(tail) {
            let tail_handle = GcHandle::from_tagged(tail);
            (0..tail_handle.field_count())
                .map(|i| format!("{}:{:#x}", i, tail_handle.get_field(i)))
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            String::new()
        };
        let _ = writeln!(
            file,
            "[vec-get] vec={:#x} idx={} count={:#x} shift={:#x} root={:#x} tail={:#x} tail_fields=[{}]",
            vec_ptr, idx, count, shift, root, tail, tail_fields
        );
        let gc_top = crate::builtins::get_gc_frame_top();
        let _ = writeln!(file, "[vec-get-gc-top] header={:#x}", gc_top);
        let mut header_addr = gc_top;
        let mut frames_seen = 0usize;
        while header_addr != 0 && frames_seen < 16 {
            frames_seen += 1;
            let header_value = unsafe { *(header_addr as *const usize) };
            let header = crate::types::Header::from_usize(header_value);
            let frame_pointer = header_addr + 8;
            let return_addr = unsafe { *((frame_pointer + 8) as *const usize) };
            let frame_name = crate::get_runtime()
                .get()
                .get_function_containing_pointer(return_addr as *const u8)
                .map(|(function, offset)| format!("{}+{:#x}", function.name, offset))
                .unwrap_or_else(|| "<unknown>".to_string());
            let _ = writeln!(
                file,
                "[vec-get-frame] header={:#x} slots={} fn={}",
                header_addr, header.size, frame_name
            );
            let slots_to_log = header.size as usize;
            for i in 0..slots_to_log {
                let slot_addr = header_addr - 24 - (i * 8);
                let slot_value = unsafe { *(slot_addr as *const usize) };
                let _ = writeln!(
                    file,
                    "[vec-get-slot] fn={} slot={} addr={:#x} value={:#x}",
                    frame_name, i, slot_addr, slot_value
                );
            }
            header_addr = unsafe { *((header_addr - 8) as *const usize) };
        }
    }
    let result = PersistentVec::get(vec, idx);
    if std::env::var("BEAGLE_DEBUG_VEC_ATOM_BITS").is_ok() {
        if BuiltInTypes::is_heap_pointer(result) {
            let obj = GcHandle::from_tagged(result);
            if obj.get_type_id() == crate::collections::TYPE_ID_ATOM {
                eprintln!(
                    "[vec-get-atom] vec={:#x} index={} result={:#x} low4={:#x} field0={:#x}",
                    vec_ptr,
                    idx,
                    result,
                    result & 0xF,
                    obj.get_field(0)
                );
            }
        }
    }
    result
}

/// Push a value onto a persistent vector
/// Signature: (stack_pointer, frame_pointer, vec_ptr, value) -> tagged_ptr
pub unsafe extern "C" fn rust_vec_push(
    stack_pointer: usize,
    frame_pointer: usize,
    vec_ptr: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(vec_ptr) {
        return BuiltInTypes::null_value() as usize;
    }

    let vec = GcHandle::from_tagged(vec_ptr);

    if std::env::var("BEAGLE_DEBUG_VEC_ATOM_BITS").is_ok() {
        if BuiltInTypes::is_heap_pointer(value) {
            let obj = GcHandle::from_tagged(value);
            if obj.get_type_id() == crate::collections::TYPE_ID_ATOM {
                eprintln!(
                    "[vec-push-atom] vec={:#x} value={:#x} low4={:#x} field0={:#x}",
                    vec_ptr,
                    value,
                    value & 0xF,
                    obj.get_field(0)
                );
            }
        }
    }

    match PersistentVec::push(runtime, stack_pointer, vec, value) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to push to vector: {}", e),
            );
        },
    }
}

/// Update a value at an index in a persistent vector
/// Signature: (stack_pointer, frame_pointer, vec_ptr, index, value) -> tagged_ptr
pub unsafe extern "C" fn rust_vec_assoc(
    stack_pointer: usize,
    frame_pointer: usize,
    vec_ptr: usize,
    index: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(vec_ptr) {
        return BuiltInTypes::null_value() as usize;
    }

    let vec = GcHandle::from_tagged(vec_ptr);
    let idx = BuiltInTypes::untag(index);

    match PersistentVec::assoc(runtime, stack_pointer, vec, idx, value) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to update vector at index {}: {}", idx, e),
            );
        },
    }
}

// ========== Map builtins ==========

/// Create an empty persistent map
/// Signature: (stack_pointer, frame_pointer) -> tagged_ptr
pub unsafe extern "C" fn rust_map_empty(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Stay c_calling - HandleScope::allocate checks is_paused and participates in GC

    match PersistentMap::empty(runtime, stack_pointer) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to create empty map: {}", e),
            );
        },
    }
}

/// Get the count of a persistent map
/// Signature: (map_ptr) -> tagged_int
pub unsafe extern "C" fn rust_map_count(map_ptr: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        return BuiltInTypes::construct_int(0) as usize;
    }
    let map = GcHandle::from_tagged(map_ptr);
    let count = PersistentMap::count(map);
    BuiltInTypes::construct_int(count as isize) as usize
}

/// Get a value from a persistent map by key
/// Signature: (map_ptr, key) -> tagged_value
pub unsafe extern "C" fn rust_map_get(map_ptr: usize, key: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        return BuiltInTypes::null_value() as usize;
    }
    let runtime = get_runtime().get();
    let map = GcHandle::from_tagged(map_ptr);
    PersistentMap::get(runtime, map, key)
}

/// Associate a key-value pair in a persistent map
/// Signature: (stack_pointer, frame_pointer, map_ptr, key, value) -> tagged_ptr
pub unsafe extern "C" fn rust_map_assoc(
    stack_pointer: usize,
    frame_pointer: usize,
    map_ptr: usize,
    key: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        return BuiltInTypes::null_value() as usize;
    }

    // Stay c_calling - HandleScope::allocate checks is_paused and participates in GC

    match PersistentMap::assoc(runtime, stack_pointer, map_ptr, key, value) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to associate key in map: {}", e),
            );
        },
    }
}

/// Get all keys from a persistent map as a vector
// Signature: (stack_pointer, frame_pointer, map_ptr) -> tagged_ptr (vector)
pub unsafe extern "C" fn rust_map_keys(
    stack_pointer: usize,
    frame_pointer: usize,
    map_ptr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        // Empty map or invalid - return empty vector
        match PersistentVec::empty(runtime, stack_pointer) {
            Ok(handle) => return handle.as_tagged(),
            Err(e) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    format!("Failed to create empty vector for map keys: {}", e),
                );
            },
        }
    }

    let map = GcHandle::from_tagged(map_ptr);

    // Stay c_calling - HandleScope::allocate checks is_paused and participates in GC

    match PersistentMap::keys(runtime, stack_pointer, map) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to get map keys: {}", e),
            );
        },
    }
}

// ========== Set builtins ==========

/// Create an empty persistent set
/// Signature: (stack_pointer, frame_pointer) -> tagged_ptr
pub unsafe extern "C" fn rust_set_empty(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    match PersistentSet::empty(runtime, stack_pointer) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to create empty set: {}", e),
            );
        },
    }
}

/// Get the count of a persistent set
/// Signature: (set_ptr) -> tagged_int
pub unsafe extern "C" fn rust_set_count(set_ptr: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(set_ptr) {
        return BuiltInTypes::construct_int(0) as usize;
    }
    let set = GcHandle::from_tagged(set_ptr);
    let count = PersistentSet::count(set);
    BuiltInTypes::construct_int(count as isize) as usize
}

/// Check if an element is in a persistent set
/// Signature: (set_ptr, element) -> tagged_bool
pub unsafe extern "C" fn rust_set_contains(set_ptr: usize, element: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(set_ptr) {
        return BuiltInTypes::false_value() as usize;
    }
    let runtime = get_runtime().get();
    let set = GcHandle::from_tagged(set_ptr);
    if PersistentSet::contains(runtime, set, element) {
        BuiltInTypes::true_value() as usize
    } else {
        BuiltInTypes::false_value() as usize
    }
}

/// Add an element to a persistent set
/// Signature: (stack_pointer, frame_pointer, set_ptr, element) -> tagged_ptr
pub unsafe extern "C" fn rust_set_add(
    stack_pointer: usize,
    frame_pointer: usize,
    set_ptr: usize,
    element: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(set_ptr) {
        return BuiltInTypes::null_value() as usize;
    }

    match PersistentSet::add(runtime, stack_pointer, set_ptr, element) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to add element to set: {}", e),
            );
        },
    }
}

/// Get all elements from a persistent set as a vector
/// Signature: (stack_pointer, frame_pointer, set_ptr) -> tagged_ptr (vector)
pub unsafe extern "C" fn rust_set_elements(
    stack_pointer: usize,
    frame_pointer: usize,
    set_ptr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(set_ptr) {
        // Empty set or invalid - return empty vector
        match PersistentVec::empty(runtime, stack_pointer) {
            Ok(handle) => return handle.as_tagged(),
            Err(e) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    format!("Failed to create empty vector for set elements: {}", e),
                );
            },
        }
    }

    let set = GcHandle::from_tagged(set_ptr);

    match PersistentSet::elements(runtime, stack_pointer, set) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to get set elements: {}", e),
            );
        },
    }
}

// ========== Mutable Map builtins ==========

/// Create an empty mutable map with default capacity (16)
/// Signature: (stack_pointer, frame_pointer) -> tagged_ptr
pub unsafe extern "C" fn rust_mutable_map_empty(
    stack_pointer: usize,
    frame_pointer: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    match MutableMap::empty(runtime, stack_pointer, 16) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to create empty mutable map: {}", e),
            );
        },
    }
}

/// Create an empty mutable map with specified capacity
/// Signature: (stack_pointer, frame_pointer, capacity) -> tagged_ptr
pub unsafe extern "C" fn rust_mutable_map_with_capacity(
    stack_pointer: usize,
    frame_pointer: usize,
    capacity: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let cap = BuiltInTypes::untag(capacity);

    match MutableMap::empty(runtime, stack_pointer, cap) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to create mutable map with capacity {}: {}", cap, e),
            );
        },
    }
}

/// Put a key-value pair into a mutable map (mutates in place)
/// Signature: (stack_pointer, frame_pointer, map, key, value) -> null
pub unsafe extern "C" fn rust_mutable_map_put(
    stack_pointer: usize,
    frame_pointer: usize,
    map_ptr: usize,
    key: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        return BuiltInTypes::null_value() as usize;
    }

    match MutableMap::put(runtime, stack_pointer, map_ptr, key, value) {
        Ok(()) => BuiltInTypes::null_value() as usize,
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to put value in mutable map: {}", e),
            );
        },
    }
}

/// Increment the integer value for a key by 1, inserting 1 if absent.
/// Signature: (stack_pointer, frame_pointer, map, key) -> null
pub unsafe extern "C" fn rust_mutable_map_increment(
    stack_pointer: usize,
    frame_pointer: usize,
    map_ptr: usize,
    key: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        return BuiltInTypes::null_value() as usize;
    }

    match MutableMap::increment(runtime, stack_pointer, map_ptr, key) {
        Ok(()) => BuiltInTypes::null_value() as usize,
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to increment value in mutable map: {}", e),
            );
        },
    }
}

/// Get a value from a mutable map by key
/// Signature: (map, key) -> tagged_value
pub unsafe extern "C" fn rust_mutable_map_get(map_ptr: usize, key: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        return BuiltInTypes::null_value() as usize;
    }
    let runtime = get_runtime().get();
    let map = GcHandle::from_tagged(map_ptr);
    MutableMap::get(runtime, map, key)
}

/// Get the count of entries in a mutable map
/// Signature: (map) -> tagged_int
pub unsafe extern "C" fn rust_mutable_map_count(map_ptr: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        return BuiltInTypes::construct_int(0) as usize;
    }
    let map = GcHandle::from_tagged(map_ptr);
    let count = MutableMap::count(map);
    BuiltInTypes::construct_int(count as isize) as usize
}

/// Get all entries from a mutable map as array of [key, value] pairs
/// Signature: (stack_pointer, frame_pointer, map) -> tagged_ptr (array)
pub unsafe extern "C" fn rust_mutable_map_entries(
    stack_pointer: usize,
    frame_pointer: usize,
    map_ptr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        return BuiltInTypes::null_value() as usize;
    }

    let map = GcHandle::from_tagged(map_ptr);

    match MutableMap::entries(runtime, stack_pointer, map) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to get mutable map entries: {}", e),
            );
        },
    }
}
