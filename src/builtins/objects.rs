use super::*;
use crate::save_gc_context;

pub extern "C" fn fill_object_fields(object_pointer: usize, value: usize) -> usize {
    print_call_builtin(get_runtime().get(), "fill_object_fields");
    let mut object = HeapObject::from_tagged(object_pointer);
    let raw_slice = object.get_fields_mut();
    raw_slice.fill(value);
    object_pointer
}

pub extern "C" fn make_closure(
    stack_pointer: usize,
    frame_pointer: usize,
    function: usize,
    num_free: usize,
    free_variable_pointer: usize,
) -> usize {
    #[cfg(debug_assertions)]
    {
        if std::env::var("BEAGLE_DEBUG_MAKE_CLOSURE_FRAME").is_ok() {
            let gc_top = get_gc_frame_top();
            let saved_fp = unsafe { *(frame_pointer as *const usize) };
            let return_addr = unsafe { *((frame_pointer + 8) as *const usize) };
            if let Some((function, offset)) = get_runtime()
                .get()
                .get_function_containing_pointer(return_addr as *const u8)
            {
                eprintln!(
                    "[make_closure-frame] sp={:#x} fp={:#x} saved_fp={:#x} ret={:#x} caller={}+{:#x}",
                    stack_pointer, frame_pointer, saved_fp, return_addr, function.name, offset
                );
            } else {
                eprintln!(
                    "[make_closure-frame] sp={:#x} fp={:#x} saved_fp={:#x} ret={:#x}",
                    stack_pointer, frame_pointer, saved_fp, return_addr
                );
            }
            eprintln!(
                "[make_closure-frame] fp_slots prev={:#x} header={:#x} local0={:#x}",
                unsafe { *((frame_pointer - 16) as *const usize) },
                unsafe { *((frame_pointer - 8) as *const usize) },
                unsafe { *((frame_pointer - 24) as *const usize) },
            );
            if gc_top != 0 {
                let gc_top_fp = gc_top + 8;
                let gc_top_ret = unsafe { *((gc_top_fp + 8) as *const usize) };
                eprintln!(
                    "[make_closure-frame] gc_top={:#x} gc_top_fp={:#x} gc_top_ret={:#x}",
                    gc_top, gc_top_fp, gc_top_ret
                );
            }
        }
    }
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "make_closure");
    let runtime = get_runtime().get_mut();

    // Extract the raw function pointer for closure creation.
    // Accept both Function-tagged raw pointers and HeapObject function struct objects.
    let function = match BuiltInTypes::get_kind(function) {
        BuiltInTypes::Function => function,
        BuiltInTypes::HeapObject => {
            let obj = HeapObject::from_tagged(function);
            if obj.get_type_id() == 0 && obj.get_struct_id() == runtime.function_struct_id {
                // Extract Int-tagged fn_ptr from field 0, re-tag as Function
                let int_tagged = obj.get_field(0);
                let raw = BuiltInTypes::untag(int_tagged);
                BuiltInTypes::Function.tag(raw as isize) as usize
            } else {
                unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "TypeError",
                        format!(
                            "Expected function, got HeapObject with type_id={}",
                            obj.get_type_id()
                        ),
                    );
                }
            }
        }
        _ => unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!(
                    "Expected function, got {:?}",
                    BuiltInTypes::get_kind(function)
                ),
            );
        },
    };

    let num_free = BuiltInTypes::untag(num_free);
    let free_variable_pointer = free_variable_pointer as *const usize;
    let start = unsafe { free_variable_pointer.sub(num_free.saturating_sub(1)) };
    let free_variables = unsafe { from_raw_parts(start, num_free) };
    match runtime.make_closure(stack_pointer, function, free_variables) {
        Ok(closure) => closure,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate closure - out of memory".to_string(),
            );
        },
    }
}

/// Create a FunctionObject from a function pointer.
/// Unlike closures, FunctionObjects have no free variables and don't pass self as arg0.
pub extern "C" fn make_function_object(
    stack_pointer: usize,
    frame_pointer: usize,
    function: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "make_function_object");
    let runtime = get_runtime().get_mut();

    // Accept both Function-tagged raw pointers and HeapObject function struct objects
    let function = match BuiltInTypes::get_kind(function) {
        BuiltInTypes::Function => function,
        BuiltInTypes::HeapObject => {
            let obj = HeapObject::from_tagged(function);
            if obj.get_type_id() == 0 && obj.get_struct_id() == runtime.function_struct_id {
                let int_tagged = obj.get_field(0);
                let raw = BuiltInTypes::untag(int_tagged);
                BuiltInTypes::Function.tag(raw as isize) as usize
            } else {
                unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "TypeError",
                        format!(
                            "make-function-object: Expected function, got HeapObject with type_id={}",
                            obj.get_type_id()
                        ),
                    );
                }
            }
        }
        _ => unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!(
                    "make-function-object: Expected function, got {:?}",
                    BuiltInTypes::get_kind(function)
                ),
            );
        },
    };

    match runtime.make_function_object(stack_pointer, function) {
        Ok(obj) => obj,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate function object - out of memory".to_string(),
            );
        },
    }
}

/// Takes tagged shape_id and tagged expected_id, returns tagged bool.
pub extern "C" fn check_struct_family(tagged_shape_id: usize, tagged_family_id: usize) -> usize {
    let shape_id = BuiltInTypes::untag(tagged_shape_id);
    let expected_id = BuiltInTypes::untag(tagged_family_id);
    if shape_id == expected_id {
        BuiltInTypes::true_value() as usize
    } else {
        BuiltInTypes::false_value() as usize
    }
}

pub extern "C" fn property_access(
    stack_pointer: usize,
    frame_pointer: usize,
    struct_pointer: usize,
    str_constant_ptr: usize,
    property_cache_location: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);

    // Check for null before accessing properties
    if struct_pointer == BuiltInTypes::null_value() as usize {
        let runtime = get_runtime().get_mut();
        let str_constant_idx: usize = BuiltInTypes::untag(str_constant_ptr);
        let property_name = &runtime.string_constants[str_constant_idx].str;
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!("Cannot access property '{}' on null", property_name),
            );
        }
    }

    let runtime = get_runtime().get_mut();
    #[cfg(debug_assertions)]
    let pre_result = if std::env::var("BEAGLE_DEBUG_PROPERTY_ACCESS").is_ok() {
        HeapObject::try_from_tagged(struct_pointer).map(|obj| obj.get_field(0))
    } else {
        None
    };
    let (result, index) = runtime
        .property_access(struct_pointer, str_constant_ptr)
        .unwrap_or_else(|error| unsafe {
            throw_runtime_error(stack_pointer, "FieldError", error.to_string());
        });
    #[cfg(debug_assertions)]
    {
        if std::env::var("BEAGLE_DEBUG_PROPERTY_ACCESS").is_ok() {
            let str_constant_idx = BuiltInTypes::untag(str_constant_ptr);
            let property_name = runtime.string_constants[str_constant_idx].str.clone();
            if property_name == "v" {
                let pre_header = HeapObject::try_from_tagged(struct_pointer).map(|obj| {
                    let raw_header = unsafe { *(obj.untagged() as *const usize) };
                    let header = obj.get_header();
                    (
                        raw_header,
                        header.marked,
                        header.opaque,
                        header.large,
                        obj.get_struct_id(),
                    )
                });
                let struct_name = HeapObject::try_from_tagged(struct_pointer)
                    .and_then(|obj| runtime.get_struct_by_id(obj.get_struct_id()))
                    .map(|def| def.name.clone())
                    .unwrap_or_else(|| "<unknown>".to_string());
                let post_result =
                    HeapObject::try_from_tagged(struct_pointer).map(|obj| obj.get_field(0));
                eprintln!(
                    "[property_access] struct={} object={:#x} property={} pre={:?} header={:?} result={:#x} post={:?} index={}",
                    struct_name,
                    struct_pointer,
                    property_name,
                    pre_result,
                    pre_header,
                    result,
                    post_result,
                    index
                );
            }
        }
    }
    // Don't cache if:
    // - field_index is usize::MAX (field was added after object creation, returned null)
    // - object has an old layout version (its field offsets differ from current layout)
    #[cfg(debug_assertions)]
    let should_cache =
        index != usize::MAX && std::env::var("BEAGLE_DISABLE_PROPERTY_CACHE").is_err();
    #[cfg(not(debug_assertions))]
    let should_cache = index != usize::MAX;
    if should_cache {
        let heap_obj = HeapObject::from_tagged(struct_pointer);
        let layout_version = heap_obj.get_layout_version();
        let struct_id = heap_obj.get_struct_id();
        let current_version = runtime.structs.get_current_layout_version(struct_id);
        if layout_version == current_version {
            // Store combined struct_id + layout_version so the fast path rejects
            // old-layout objects (which have the same struct_id but different version).
            let raw_header = unsafe { *(heap_obj.untagged() as *const usize) };
            let combined = raw_header & 0x00FFFFFFFF0000F0;
            let buffer = unsafe { from_raw_parts_mut(property_cache_location as *mut usize, 2) };
            buffer[0] = combined;
            buffer[1] = index * 8;
        }
    }
    result
}

pub extern "C" fn type_of(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "type_of");
    let runtime = get_runtime().get_mut();
    match runtime.type_of(stack_pointer, value) {
        Ok(t) => t,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate type symbol - out of memory".to_string(),
            );
        },
    }
}

pub extern "C" fn get_os(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "get_os");
    let runtime = get_runtime().get_mut();
    let os_name = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "unknown"
    };
    unsafe { allocate_string_or_throw(runtime, stack_pointer, os_name.to_string()) }
}

/// Return the raw usize value of any Beagle value (tagged pointer).
/// This is useful for using atoms as unique identifiers when coordinating
/// async operations across the Rust/Beagle boundary.
pub extern "C" fn atom_address(value: usize) -> usize {
    // Just return the raw tagged pointer value as an integer.
    // This works because each atom has a unique address.
    BuiltInTypes::Int.tag(value as isize) as usize
}

pub extern "C" fn equal(a: usize, b: usize) -> usize {
    print_call_builtin(get_runtime().get(), "equal");
    let runtime = get_runtime().get_mut();
    if runtime.equal(a, b) {
        BuiltInTypes::true_value() as usize
    } else {
        BuiltInTypes::false_value() as usize
    }
}

pub extern "C" fn write_field(
    stack_pointer: usize,
    frame_pointer: usize,
    struct_pointer: usize,
    str_constant_ptr: usize,
    property_cache_location: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    // write_field checks mutability and throws if not mutable
    let index = runtime.write_field(stack_pointer, struct_pointer, str_constant_ptr, value);
    // Write barrier for generational GC - mark the card containing this object
    runtime.mark_card_for_object(struct_pointer);
    let heap_obj = HeapObject::from_tagged(struct_pointer);
    let layout_version = heap_obj.get_layout_version();
    let struct_id = heap_obj.get_struct_id();
    let current_version = runtime.structs.get_current_layout_version(struct_id);
    // Cache layout: [struct_id_versioned, field_offset, is_mutable]
    // We only reach here if field is mutable (otherwise write_field would have thrown)
    // Only cache for current-layout objects to prevent stale offset reuse
    if layout_version == current_version {
        let raw_header = unsafe { *(heap_obj.untagged() as *const usize) };
        let combined = raw_header & 0x00FFFFFFFF0000F0;
        let buffer = unsafe { from_raw_parts_mut(property_cache_location as *mut usize, 3) };
        buffer[0] = combined;
        buffer[1] = index * 8;
        buffer[2] = 1; // is_mutable = true
    }
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn copy_object(
    stack_pointer: usize,
    frame_pointer: usize,
    object_pointer: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let object_pointer_id = runtime.register_temporary_root(object_pointer);
    let to_pointer = {
        let object = HeapObject::from_tagged(object_pointer);
        let header = object.get_header();
        let size = header.size as usize;
        let kind = BuiltInTypes::get_kind(object_pointer);
        match runtime.allocate(size, stack_pointer, kind) {
            Ok(ptr) => ptr,
            Err(_) => unsafe {
                runtime.unregister_temporary_root(object_pointer_id);
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate object copy - out of memory".to_string(),
                );
            },
        }
    };
    let object_pointer = runtime.unregister_temporary_root(object_pointer_id);
    let mut to_object = HeapObject::from_tagged(to_pointer);
    let object = HeapObject::from_tagged(object_pointer);
    match runtime.copy_object(object, &mut to_object) {
        Ok(ptr) => ptr,
        Err(error) => {
            let stack_pointer = get_current_stack_pointer();
            println!("Error: {:?}", error);
            unsafe { throw_error(stack_pointer, frame_pointer) };
        }
    }
}

/// Copies a struct for spread syntax, migrating to the current layout if needed.
/// The source object may have an old layout version — this function always produces
/// a copy with the CURRENT layout, so the caller's compile-time field indices are correct.
pub unsafe extern "C" fn copy_object_spread(
    stack_pointer: usize,
    frame_pointer: usize,
    object_pointer: usize,
    expected_struct_id: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);

    // Check that the value is actually a heap object (not an int, float, etc.)
    if !BuiltInTypes::is_heap_pointer(object_pointer) {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "StructError",
                "Spread source must be a struct".to_string(),
            );
        }
    }

    let actual_struct_id = HeapObject::from_tagged(object_pointer).get_struct_id();
    if actual_struct_id != expected_struct_id {
        let runtime = get_runtime().get_mut();
        let expected_name = runtime
            .get_struct_by_id(expected_struct_id)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| format!("<struct#{}>", expected_struct_id));
        let actual_name = runtime
            .get_struct_by_id(actual_struct_id)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| format!("<struct#{}>", actual_struct_id));
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "StructError",
                format!(
                    "Spread type mismatch: expected {}, got {}",
                    expected_name, actual_name
                ),
            );
        }
    }

    let source_object = HeapObject::from_tagged(object_pointer);
    let source_layout_version = source_object.get_layout_version();

    let runtime = get_runtime().get_mut();
    let current_version = runtime
        .structs
        .get_current_layout_version(expected_struct_id);

    if source_layout_version == current_version {
        // Source already has current layout — simple copy
        let object_pointer_id = runtime.register_temporary_root(object_pointer);
        let object = HeapObject::from_tagged(object_pointer);
        let header = object.get_header();
        let size = header.size as usize;
        let kind = BuiltInTypes::get_kind(object_pointer);
        let to_pointer = match runtime.allocate(size, stack_pointer, kind) {
            Ok(ptr) => ptr,
            Err(_) => unsafe {
                runtime.unregister_temporary_root(object_pointer_id);
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate object copy - out of memory".to_string(),
                );
            },
        };
        let object_pointer = runtime.unregister_temporary_root(object_pointer_id);
        let mut to_object = HeapObject::from_tagged(to_pointer);
        let object = HeapObject::from_tagged(object_pointer);
        match runtime.copy_object(object, &mut to_object) {
            Ok(ptr) => ptr,
            Err(error) => {
                let stack_pointer = get_current_stack_pointer();
                println!("Error: {:?}", error);
                unsafe { throw_error(stack_pointer, frame_pointer) };
            }
        }
    } else {
        // Source has old layout — migrate to current layout.
        // Look up current struct definition for target fields
        let current_fields: Vec<String> = runtime
            .get_struct_by_id(expected_struct_id)
            .expect("Struct must exist")
            .fields
            .clone();
        let current_field_count = current_fields.len();

        // Look up source layout fields
        let source_fields: Vec<String> = runtime
            .structs
            .get_old_definition(expected_struct_id, source_layout_version)
            .map(|d| d.fields.clone())
            .unwrap_or_else(|| {
                runtime
                    .get_struct_by_id(expected_struct_id)
                    .expect("Struct must exist")
                    .fields
                    .clone()
            });

        // Build field map: for each field in current layout, find its position in source
        let field_map: Vec<Option<usize>> = current_fields
            .iter()
            .map(|field| source_fields.iter().position(|f| f == field))
            .collect();

        let object_pointer_id = runtime.register_temporary_root(object_pointer);
        let kind = BuiltInTypes::get_kind(object_pointer);
        let to_pointer = match runtime.allocate(current_field_count, stack_pointer, kind) {
            Ok(ptr) => ptr,
            Err(_) => unsafe {
                runtime.unregister_temporary_root(object_pointer_id);
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate object copy - out of memory".to_string(),
                );
            },
        };
        let object_pointer = runtime.unregister_temporary_root(object_pointer_id);

        let source = HeapObject::from_tagged(object_pointer);
        let mut dest = HeapObject::from_tagged(to_pointer);

        // Write header with current layout version
        let source_header = source.get_header();
        let new_header = crate::types::Header {
            type_id: source_header.type_id,
            type_data: expected_struct_id as u32,
            size: current_field_count as u16,
            opaque: source_header.opaque,
            marked: false,
            large: false,
            type_flags: current_version,
        };
        dest.writer_header_direct(new_header);

        // Map fields from source to current layout, null-fill new fields
        let null_val = BuiltInTypes::null_value() as usize;
        for (new_idx, mapping) in field_map.iter().enumerate() {
            let value = match mapping {
                Some(old_idx) => source.get_field(*old_idx),
                None => null_val,
            };
            dest.write_field(new_idx as i32, value);
        }

        to_pointer
    }
}

pub unsafe extern "C" fn copy_from_to_object(from: usize, to: usize) -> usize {
    let runtime = get_runtime().get_mut();
    if from == BuiltInTypes::null_value() as usize {
        return to;
    }
    let from = HeapObject::from_tagged(from);
    let mut to_obj = HeapObject::from_tagged(to);
    match runtime.copy_object_except_header(from, &mut to_obj) {
        Ok(_) => to_obj.tagged_pointer(),
        Err(error) => {
            let stack_pointer = get_saved_stack_pointer();
            unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "CopyError",
                    format!("Failed to copy object: {:?}", error),
                );
            }
        }
    }
}

pub unsafe extern "C" fn copy_array_range(
    from: usize,
    to: usize,
    start: usize,
    count: usize,
) -> usize {
    let from_obj = HeapObject::from_tagged(from);
    let mut to_obj = HeapObject::from_tagged(to);

    let from_fields = from_obj.get_fields();
    let to_fields = to_obj.get_fields_mut();

    let start_idx = BuiltInTypes::untag(start);
    let count_val = BuiltInTypes::untag(count);

    // Use ptr::copy_nonoverlapping for fast memcpy
    unsafe {
        std::ptr::copy_nonoverlapping(
            from_fields.as_ptr().add(start_idx),
            to_fields.as_mut_ptr().add(start_idx),
            count_val,
        );
    }

    to
}
