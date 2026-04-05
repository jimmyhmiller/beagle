use super::*;
use crate::save_gc_context;

/// Slow path for protocol dispatch - looks up function in dispatch table and updates cache
/// Returns the function pointer to call
///
/// Type identification for dispatch:
/// - Tagged primitives (Int, Float, Bool): high bit marker + (tag + 16)
/// - Heap objects with type_id > 0 (Array=1, String=2, Keyword=3): high bit marker + type_id
/// - Heap objects with type_id = 0 (structs): struct_id from type_data (raw value)
pub extern "C" fn protocol_dispatch(
    stack_pointer: usize,
    frame_pointer: usize,
    first_arg: usize,
    cache_location: usize,
    dispatch_table_ptr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let type_id = if BuiltInTypes::is_heap_pointer(first_arg) {
        let heap_obj = HeapObject::from_tagged(first_arg);
        let header_type_id = heap_obj.get_type_id();

        if header_type_id == 0 {
            // Custom struct - use struct_id from type_data (raw value)
            heap_obj.get_struct_id()
        } else {
            // Built-in heap type (Array=1, String=2, Keyword=3)
            // Map variant type_ids to their base primitive index
            let primitive_index = match header_type_id as u8 {
                TYPE_ID_STRING_SLICE | TYPE_ID_CONS_STRING => 2, // → String
                _ => header_type_id,
            };
            0x8000_0000_0000_0000 | primitive_index
        }
    } else {
        // Tagged primitive (Int, Float, Bool, String constant, etc.)
        let kind = BuiltInTypes::get_kind(first_arg);
        let tag = kind.get_tag() as usize;
        // Map tags to primitive indices:
        // - String constant (tag 2) -> index 2 (same as heap String)
        // - Int (tag 0) -> index 16
        // - Float (tag 1) -> index 17
        // - Bool (tag 3) -> index 19
        let primitive_index = if tag == 2 {
            2 // String constant uses same index as heap String
        } else {
            tag + 16
        };
        0x8000_0000_0000_0000 | primitive_index
    };

    // Look up function pointer in dispatch table
    let dispatch_table = unsafe { &*(dispatch_table_ptr as *const DispatchTable) };
    let fn_ptr = if type_id & 0x8000_0000_0000_0000 != 0 {
        // Primitive/built-in type
        let primitive_index = type_id & 0x7FFF_FFFF_FFFF_FFFF;
        dispatch_table.lookup_primitive(primitive_index)
    } else {
        // Struct type - struct_id is stored as raw value in header
        dispatch_table.lookup_struct(type_id)
    };

    if fn_ptr == 0 {
        // Get a string representation of the value for the error message
        let runtime = get_runtime().get_mut();
        let value_repr = runtime
            .get_repr(first_arg, 0)
            .unwrap_or_else(|| "<unknown>".to_string());

        // Get type name for better error message
        let type_name = if type_id & 0x8000_0000_0000_0000 != 0 {
            let primitive_index = type_id & 0x7FFF_FFFF_FFFF_FFFF;
            match primitive_index {
                1 => "Array",
                2 => "String",
                3 => "Keyword",
                16 => "Int",
                17 => "Float",
                19 => "Bool",
                _ => "Unknown",
            }
        } else {
            "Struct"
        };
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!(
                    "Function not implemented for {} value: {}",
                    type_name, value_repr
                ),
            );
        }
    }

    // Update cache
    let buffer = unsafe { from_raw_parts_mut(cache_location as *mut usize, 2) };
    buffer[0] = type_id;
    buffer[1] = fn_ptr;

    fn_ptr
}

pub extern "C" fn hash(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let _ = stack_pointer; // Used by save_gc_context!
    print_call_builtin(get_runtime().get(), "hash");
    let runtime = get_runtime().get();
    let raw_hash = runtime.hash_value(value);
    BuiltInTypes::Int.tag(raw_hash as isize) as usize
}

pub extern "C" fn many_args(
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
    a6: usize,
    a7: usize,
    a8: usize,
    a9: usize,
    a10: usize,
    a11: usize,
) -> usize {
    let a1 = BuiltInTypes::untag(a1);
    let a2 = BuiltInTypes::untag(a2);
    let a3 = BuiltInTypes::untag(a3);
    let a4 = BuiltInTypes::untag(a4);
    let a5 = BuiltInTypes::untag(a5);
    let a6 = BuiltInTypes::untag(a6);
    let a7 = BuiltInTypes::untag(a7);
    let a8 = BuiltInTypes::untag(a8);
    let a9 = BuiltInTypes::untag(a9);
    let a10 = BuiltInTypes::untag(a10);
    let a11 = BuiltInTypes::untag(a11);
    let result = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11;
    BuiltInTypes::Int.tag(result as isize) as usize
}

/// Dispatch for multi-arity functions.
/// The multi-arity object layout (heap object):
/// - header: standard heap object header
/// - field 0: num_arities (tagged Int)
/// - entries: [arity (tagged), fn_ptr (tagged Function), is_variadic (tagged bool)] * num_arities
pub extern "C" fn dispatch_multi_arity(
    stack_pointer: usize,
    frame_pointer: usize,
    multi_arity_obj: usize,
    arg_count: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);

    // Check if this is a Function struct object - if so, extract fn_ptr directly
    let heap_obj = HeapObject::from_tagged(multi_arity_obj);
    if heap_obj.get_type_id() == 0 {
        let struct_id = heap_obj.get_struct_id();
        let runtime = get_runtime().get();
        if struct_id == runtime.function_struct_id {
            // Return the Int-tagged fn_ptr from field 0
            // Callers expect a Function-tagged result, so re-tag it
            let int_tagged_fn_ptr = heap_obj.get_field(0);
            let raw_fn_ptr = BuiltInTypes::untag(int_tagged_fn_ptr);
            return BuiltInTypes::Function.tag(raw_fn_ptr as isize) as usize;
        }
    }

    // Untag the heap object
    let ptr = BuiltInTypes::untag(multi_arity_obj);

    // Read num_arities from offset 1 (after header)
    let num_arities_raw = unsafe { *((ptr + 8) as *const usize) };
    let num_arities = BuiltInTypes::untag(num_arities_raw);

    // Untag the arg_count
    let arg_count = BuiltInTypes::untag(arg_count);

    // Collect available arities for error message
    let mut available_arities = Vec::new();

    // Search for matching arity - first try exact match for non-variadic
    for i in 0..num_arities {
        let base_offset = 16 + i * 24; // header(8) + num_arities(8) + i * (3 * 8)
        let arity = unsafe { *((ptr + base_offset) as *const usize) };
        let arity = BuiltInTypes::untag(arity);
        let jt_ptr = unsafe { *((ptr + base_offset + 8) as *const usize) };
        let is_variadic = unsafe { *((ptr + base_offset + 16) as *const usize) };
        let is_variadic = is_variadic == BuiltInTypes::true_value() as usize;

        available_arities.push(if is_variadic {
            format!("{}+", arity)
        } else {
            arity.to_string()
        });

        if !is_variadic && arity == arg_count {
            // Load through the jump table to get the current code pointer.
            // This indirection is what makes hot-reload work: eval updates
            // the jump table entry, so we always get the latest code.
            // The jump table stores Function-tagged pointers, so the loaded
            // value is already correctly tagged.
            let fn_ptr = unsafe { *(jt_ptr as *const usize) };
            return fn_ptr;
        }
    }

    // Then try variadic match (arg_count >= min_arity)
    for i in 0..num_arities {
        let base_offset = 16 + i * 24;
        let min_arity = unsafe { *((ptr + base_offset) as *const usize) };
        let min_arity = BuiltInTypes::untag(min_arity);
        let jt_ptr = unsafe { *((ptr + base_offset + 8) as *const usize) };
        let is_variadic = unsafe { *((ptr + base_offset + 16) as *const usize) };
        let is_variadic = is_variadic == BuiltInTypes::true_value() as usize;

        if is_variadic && arg_count >= min_arity {
            let fn_ptr = unsafe { *(jt_ptr as *const usize) };
            return fn_ptr;
        }
    }

    // No matching arity found - throw a catchable ArityError
    unsafe {
        throw_runtime_error(
            stack_pointer,
            "ArityError",
            format!(
                "No matching arity for multi-arity function call with {} argument(s). Available arities: {}",
                arg_count,
                available_arities.join(", ")
            ),
        );
    }
}
