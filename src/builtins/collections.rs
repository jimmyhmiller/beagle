use super::*;
use crate::collections::{
    GcHandle, HandleScope, MutableMap, PersistentMap, PersistentSet, PersistentVec,
    TYPE_ID_RAW_ARRAY,
};
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
    PersistentVec::get(vec, idx)
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

/// Remove the last element of a persistent vector, returning a new vector.
/// Popping a 0- or 1-element vector returns the empty vector.
/// Signature: (stack_pointer, frame_pointer, vec_ptr) -> tagged_ptr
pub unsafe extern "C" fn rust_vec_pop(
    stack_pointer: usize,
    frame_pointer: usize,
    vec_ptr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(vec_ptr) {
        return BuiltInTypes::null_value() as usize;
    }

    let vec = GcHandle::from_tagged(vec_ptr);

    match PersistentVec::pop(runtime, stack_pointer, vec) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to pop from vector: {}", e),
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

/// Copy a persistent vector's elements into a freshly-allocated raw mutable
/// array (type_id = 1). Used as the first step of sort — the algorithm runs
/// in place on the array, then `array-to-vec` rebuilds a persistent vector.
/// Signature: (stack_pointer, frame_pointer, vec_ptr) -> tagged_ptr (array)
pub unsafe extern "C" fn rust_vec_to_array(
    stack_pointer: usize,
    frame_pointer: usize,
    vec_ptr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(vec_ptr) {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                "vec-to-array: expected a persistent vector".to_string(),
            );
        }
    }

    let vec = GcHandle::from_tagged(vec_ptr);
    let count = PersistentVec::count(vec);

    let mut scope = HandleScope::new(runtime, stack_pointer);
    let vec_h = scope.alloc(vec_ptr);

    let array_h = match scope.allocate_typed_zeroed(count, TYPE_ID_RAW_ARRAY) {
        Ok(h) => h,
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to allocate array for vec-to-array: {}", e),
            );
        },
    };

    for i in 0..count {
        let vec = GcHandle::from_tagged(vec_h.get());
        let value = PersistentVec::get(vec, i);
        let array = array_h.to_gc_handle();
        array.set_field_with_barrier(scope.runtime(), i, value);
    }

    array_h.to_gc_handle().as_tagged()
}

/// Build a new PersistentVec from the first `len` elements of a raw mutable
/// array (type_id = 1). Counterpart to `vec-to-array` — the sort writes sorted
/// elements into the array and this rebuilds an immutable vector from them.
/// Signature: (stack_pointer, frame_pointer, array_ptr, len) -> tagged_ptr (vec)
pub unsafe extern "C" fn rust_array_to_vec(
    stack_pointer: usize,
    frame_pointer: usize,
    array_ptr: usize,
    len: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(array_ptr) {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                "array-to-vec: expected a raw array".to_string(),
            );
        }
    }
    if BuiltInTypes::get_kind(len) != BuiltInTypes::Int {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                "array-to-vec: len must be an integer".to_string(),
            );
        }
    }
    let len = BuiltInTypes::untag(len);

    let mut scope = HandleScope::new(runtime, stack_pointer);
    let array_h = scope.alloc(array_ptr);

    let empty = match PersistentVec::empty(scope.runtime(), stack_pointer) {
        Ok(h) => h,
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to create empty vec in array-to-vec: {}", e),
            );
        },
    };
    let vec_h = scope.alloc(empty.as_tagged());

    for i in 0..len {
        let array = array_h.to_gc_handle();
        let value = array.get_field(i);
        let current_vec = GcHandle::from_tagged(vec_h.get());
        let new_vec = match PersistentVec::push(scope.runtime(), stack_pointer, current_vec, value)
        {
            Ok(h) => h,
            Err(e) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    format!("Failed to push in array-to-vec: {}", e),
                );
            },
        };
        let tg = unsafe { &mut *crate::runtime::cached_thread_global_ptr() };
        tg.handle_stack[vec_h.slot()] = new_vec.as_tagged();
    }

    vec_h.get()
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
///
/// Returns null when the key is absent OR when the stored value is null — the
/// two are indistinguishable through this entry point, exactly like Clojure's
/// 2-arg `get`. Use `map-contains?`, `map-find`, or 3-arg `map-get-default` to
/// tell a stored null apart from a missing key.
pub unsafe extern "C" fn rust_map_get(map_ptr: usize, key: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        return BuiltInTypes::null_value() as usize;
    }
    let runtime = get_runtime().get();
    let map = GcHandle::from_tagged(map_ptr);
    let result = PersistentMap::get(runtime, map, key);
    if PersistentMap::is_not_found(result) {
        BuiltInTypes::null_value() as usize
    } else {
        result
    }
}

/// Get a value from a persistent map, falling back to a default when absent.
/// Signature: (map_ptr, key, default) -> tagged_value
///
/// Unlike `map-get`, this distinguishes a stored null (returned as-is) from a
/// missing key (returns `default`).
pub unsafe extern "C" fn rust_map_get_default(map_ptr: usize, key: usize, default: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        return default;
    }
    let runtime = get_runtime().get();
    let map = GcHandle::from_tagged(map_ptr);
    let result = PersistentMap::get(runtime, map, key);
    if PersistentMap::is_not_found(result) {
        default
    } else {
        result
    }
}

/// Return true if the map contains the key (even if its value is null).
/// Signature: (map_ptr, key) -> tagged_bool
pub unsafe extern "C" fn rust_map_contains(map_ptr: usize, key: usize) -> usize {
    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        return BuiltInTypes::false_value() as usize;
    }
    let runtime = get_runtime().get();
    let map = GcHandle::from_tagged(map_ptr);
    let result = PersistentMap::get(runtime, map, key);
    if PersistentMap::is_not_found(result) {
        BuiltInTypes::false_value() as usize
    } else {
        BuiltInTypes::true_value() as usize
    }
}

/// Find the entry for a key, returning a `[key, value]` vector or null if absent.
/// Signature: (stack_pointer, frame_pointer, map_ptr, key) -> tagged_ptr | null
pub unsafe extern "C" fn rust_map_find(
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

    let map = GcHandle::from_tagged(map_ptr);
    let result = PersistentMap::get(runtime, map, key);
    if PersistentMap::is_not_found(result) {
        return BuiltInTypes::null_value() as usize;
    }

    // Build the [key, value] entry vector. Protect key/value across the
    // allocations so a GC mid-build can't leave us holding stale pointers.
    let mut scope = HandleScope::new(runtime, stack_pointer);
    let key_h = scope.alloc(key);
    let value_h = scope.alloc(result);

    let entry = match PersistentVec::empty(scope.runtime(), stack_pointer) {
        Ok(handle) => handle,
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to create entry vector in map-find: {}", e),
            );
        },
    };
    let entry_h = scope.alloc(entry.as_tagged());

    for slot_handle in [&key_h, &value_h] {
        let current = entry_h.to_gc_handle();
        // Read the element fresh from its handle: a prior push may have GC'd and
        // relocated it, in which case the handle slot was updated but a cached
        // tagged value would be stale.
        let value = slot_handle.get();
        let pushed = match PersistentVec::push(scope.runtime(), stack_pointer, current, value) {
            Ok(handle) => handle,
            Err(e) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    format!("Failed to push entry in map-find: {}", e),
                );
            },
        };
        let tg = unsafe { &mut *crate::runtime::cached_thread_global_ptr() };
        tg.handle_stack[entry_h.slot()] = pushed.as_tagged();
    }

    entry_h.get()
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

/// Remove a key from a persistent map, returning a new map (the same map if the
/// key was absent). O(log32 n) native HAMT remove.
/// Signature: (stack_pointer, frame_pointer, map_ptr, key) -> tagged_ptr
pub unsafe extern "C" fn rust_map_dissoc(
    stack_pointer: usize,
    frame_pointer: usize,
    map_ptr: usize,
    key: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(map_ptr) {
        return map_ptr;
    }

    let map = GcHandle::from_tagged(map_ptr);

    match PersistentMap::without(runtime, stack_pointer, map, key) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to remove key from map: {}", e),
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

/// Remove an element from a persistent set, returning a new set (the same set if
/// absent). O(log32 n) native remove via the backing map.
/// Signature: (stack_pointer, frame_pointer, set_ptr, element) -> tagged_ptr
pub unsafe extern "C" fn rust_set_disj(
    stack_pointer: usize,
    frame_pointer: usize,
    set_ptr: usize,
    element: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if !BuiltInTypes::is_heap_pointer(set_ptr) {
        return set_ptr;
    }

    match PersistentSet::remove(runtime, stack_pointer, set_ptr, element) {
        Ok(handle) => handle.as_tagged(),
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                format!("Failed to remove element from set: {}", e),
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
