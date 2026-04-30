use super::*;
use crate::save_gc_context;

pub extern "C" fn register_extension(
    struct_name: usize,
    protocol_name: usize,
    method_name: usize,
    f: usize,
) -> usize {
    let runtime = get_runtime().get_mut();
    // TOOD: For right now I'm going to store these at the runtime level
    // But I think I actually want to store this information in the
    // protocol struct instead of out of band
    let struct_name = runtime.get_string_literal(struct_name);
    let protocol_name = runtime.get_string_literal(protocol_name);
    let method_name = runtime.get_string_literal(method_name);

    let struct_name = runtime.resolve(struct_name);
    // Don't resolve protocol names that contain type parameters (e.g., Handler<ns/Type>)
    // as they are already fully qualified
    let protocol_name = if protocol_name.contains('<') {
        protocol_name
    } else {
        runtime.resolve(protocol_name)
    };

    runtime.add_protocol_info(&protocol_name, &struct_name, &method_name, f);

    runtime.compile_protocol_method(&protocol_name, &method_name);

    BuiltInTypes::null_value() as usize
}

pub extern "C" fn is_keyword(value: usize) -> usize {
    let tag = BuiltInTypes::get_kind(value);
    if tag != BuiltInTypes::HeapObject {
        return BuiltInTypes::construct_boolean(false) as usize;
    }
    let heap_object = HeapObject::from_tagged(value);
    let is_kw = heap_object.get_header().type_id == TYPE_ID_KEYWORD;
    BuiltInTypes::construct_boolean(is_kw) as usize
}

pub extern "C" fn keyword_to_string(
    stack_pointer: usize,
    frame_pointer: usize,
    keyword: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Check if it's a HeapObject before calling from_tagged
    let tag = BuiltInTypes::get_kind(keyword);
    if tag != BuiltInTypes::HeapObject {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!(
                    "keyword->string expects a keyword, got {:?} (raw value: {:#x})",
                    tag, keyword
                ),
            );
        }
    }

    let heap_object = HeapObject::from_tagged(keyword);

    if heap_object.get_header().type_id != TYPE_ID_KEYWORD {
        let type_id = heap_object.get_header().type_id;
        let type_name = match type_id {
            TYPE_ID_STRING | TYPE_ID_STRING_SLICE | TYPE_ID_CONS_STRING => "String".to_string(),
            TYPE_ID_KEYWORD => "Keyword".to_string(),
            TYPE_ID_MULTI_ARITY_FUNCTION => "MultiArityFunction".to_string(),
            _ => {
                let struct_id = heap_object.get_struct_id();
                let runtime2 = get_runtime().get();
                if let Some(s) = runtime2.get_struct_by_id(struct_id) {
                    format!("{} (type_id={}, struct_id={})", s.name, type_id, struct_id)
                } else {
                    format!("type_id={}, struct_id={}", type_id, struct_id)
                }
            }
        };
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!("keyword->string expects a keyword, got {}", type_name),
            );
        }
    }

    let bytes = heap_object.get_keyword_bytes();
    let keyword_text = unsafe { std::str::from_utf8_unchecked(bytes) };

    match runtime.allocate_string(stack_pointer, keyword_text.to_string()) {
        Ok(ptr) => ptr.into(),
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate string for keyword - out of memory".to_string(),
            );
        },
    }
}

pub extern "C" fn string_to_keyword(
    stack_pointer: usize,
    frame_pointer: usize,
    string_value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let keyword_text = runtime.get_string(stack_pointer, string_value);

    // Use intern_keyword to ensure same text = same pointer
    match runtime.intern_keyword(stack_pointer, keyword_text) {
        Ok(ptr) => ptr,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to intern keyword - out of memory".to_string(),
            );
        },
    }
}

pub extern "C" fn load_keyword_constant_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    index: usize,
) -> usize {
    use crate::types::BuiltInTypes;

    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Fast path: the keyword's cell is populated. Read it directly —
    // every keyword that has been seen at compile time has a cell;
    // every cell has been GC-traced if it held a heap pointer, so this
    // value is current.
    let cell = runtime.keyword_cells[index];
    let cached = unsafe { *(cell as *const usize) };
    if cached != BuiltInTypes::null_value() as usize {
        return cached;
    }

    // Slow path: first time this keyword is loaded. Intern it (which
    // allocates and writes the cell).
    let keyword_text = runtime.keyword_constants[index].str.clone();
    match runtime.intern_keyword(stack_pointer, keyword_text) {
        Ok(ptr) => ptr,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to intern keyword constant - out of memory".to_string(),
            );
        },
    }
}
