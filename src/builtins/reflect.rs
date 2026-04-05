use super::*;
use crate::save_gc_context;

pub fn get_type_info(
    runtime: &mut Runtime,
    stack_pointer: usize,
    value: usize,
) -> (String, String, Option<String>, Vec<String>, bool) {
    let tag = BuiltInTypes::get_kind(value);

    match tag {
        BuiltInTypes::Null => (
            "primitive".to_string(),
            "Null".to_string(),
            None,
            vec![],
            false,
        ),
        BuiltInTypes::Int => (
            "primitive".to_string(),
            "Int".to_string(),
            None,
            vec![],
            false,
        ),
        BuiltInTypes::Float => (
            "primitive".to_string(),
            "Float".to_string(),
            None,
            vec![],
            false,
        ),
        BuiltInTypes::Bool => (
            "primitive".to_string(),
            "Bool".to_string(),
            None,
            vec![],
            false,
        ),
        BuiltInTypes::String => (
            "primitive".to_string(),
            "String".to_string(),
            None,
            vec![],
            false,
        ),
        BuiltInTypes::Function => {
            let fn_ptr = BuiltInTypes::untag(value) as *const u8;
            if let Some(func) = runtime.get_function_by_pointer(fn_ptr) {
                (
                    "function".to_string(),
                    func.name.clone(),
                    func.docstring.clone(),
                    func.arg_names.clone(),
                    func.is_variadic,
                )
            } else {
                (
                    "function".to_string(),
                    "<unknown>".to_string(),
                    None,
                    vec![],
                    false,
                )
            }
        }
        BuiltInTypes::Closure => {
            // Closures wrap a function pointer in field 0
            if BuiltInTypes::is_heap_pointer(value) {
                let heap_obj = HeapObject::from_tagged(value);
                let fn_ptr = BuiltInTypes::untag(heap_obj.get_field(0)) as *const u8;
                if let Some(func) = runtime.get_function_by_pointer(fn_ptr) {
                    (
                        "function".to_string(),
                        func.name.clone(),
                        func.docstring.clone(),
                        func.arg_names.clone(),
                        func.is_variadic,
                    )
                } else {
                    (
                        "function".to_string(),
                        "<closure>".to_string(),
                        None,
                        vec![],
                        false,
                    )
                }
            } else {
                (
                    "function".to_string(),
                    "<closure>".to_string(),
                    None,
                    vec![],
                    false,
                )
            }
        }
        BuiltInTypes::HeapObject => {
            let heap_obj = HeapObject::from_tagged(value);
            let type_id = heap_obj.get_type_id();
            let struct_id = heap_obj.get_struct_id();

            // Check for built-in heap types (Array, Keyword, PersistentVector, etc.)
            match type_id {
                1 => (
                    "primitive".to_string(),
                    "Array".to_string(),
                    None,
                    vec![],
                    false,
                ),
                2 => (
                    "primitive".to_string(),
                    "String".to_string(),
                    None,
                    vec![],
                    false,
                ),
                3 => {
                    // Keyword - extract the actual keyword text (e.g., "struct" from :struct)
                    let keyword_bytes = heap_obj.get_keyword_bytes();
                    let keyword_text = unsafe { std::str::from_utf8_unchecked(keyword_bytes) };
                    (
                        "keyword".to_string(),
                        keyword_text.to_string(),
                        None,
                        vec![],
                        false,
                    )
                }
                20 => (
                    "primitive".to_string(),
                    "PersistentVector".to_string(),
                    None,
                    vec![],
                    false,
                ),
                22 => (
                    "primitive".to_string(),
                    "PersistentMap".to_string(),
                    None,
                    vec![],
                    false,
                ),
                28 => (
                    "primitive".to_string(),
                    "PersistentSet".to_string(),
                    None,
                    vec![],
                    false,
                ),
                29 => (
                    "primitive".to_string(),
                    "MultiArityFunction".to_string(),
                    None,
                    vec![],
                    false,
                ),
                30 => (
                    "primitive".to_string(),
                    "Regex".to_string(),
                    None,
                    vec![],
                    false,
                ),
                _ => {
                    // Check for Function struct objects
                    if type_id == 0 && struct_id == runtime.function_struct_id {
                        let fn_ptr_tagged = heap_obj.get_field(0);
                        let fn_ptr = BuiltInTypes::untag(fn_ptr_tagged) as *const u8;
                        if let Some(func) = runtime.get_function_by_pointer(fn_ptr) {
                            return (
                                "function".to_string(),
                                func.name.clone(),
                                func.docstring.clone(),
                                func.arg_names.clone(),
                                func.is_variadic,
                            );
                        } else {
                            let name_tagged = heap_obj.get_field(1);
                            let name_idx = BuiltInTypes::untag(name_tagged);
                            let name = runtime.string_constants[name_idx].str.clone();
                            return ("function".to_string(), name, None, vec![], false);
                        }
                    }

                    // Check if this is a type descriptor (Struct instance with id field at index 1)
                    // Type descriptors are instances of beagle.core/Struct and field 1 contains the type id:
                    // - Negative id_field: primitive type (e.g., Int=-3, Float=-4)
                    // - Positive id_field: user-defined struct (e.g., Point=50)
                    let fields_size = heap_obj.fields_size() / 8; // number of fields
                    let struct_struct_id =
                        runtime.get_struct("beagle.core/Struct").map(|(id, _)| id);
                    if struct_struct_id == Some(struct_id) && fields_size >= 2 {
                        let id_field = heap_obj.get_field(1);
                        if BuiltInTypes::get_kind(id_field) == BuiltInTypes::Int {
                            // This is a type descriptor - extract name from field 0
                            let name_ptr = heap_obj.get_field(0);
                            let full_name = runtime.get_string(stack_pointer, name_ptr);
                            // Extract local name (after / if present)
                            let name = full_name
                                .split('/')
                                .next_back()
                                .unwrap_or(&full_name)
                                .to_string();
                            let id_value = BuiltInTypes::untag_isize(id_field as isize);
                            // Determine kind: negative id = primitive, positive id = user struct
                            let kind = if id_value < 0 { "primitive" } else { "struct" };
                            // For user-defined structs, look up the struct definition to get fields
                            let fields = if id_value > 0 {
                                runtime
                                    .get_struct_by_id(id_value as usize)
                                    .map(|s| s.fields.clone())
                                    .unwrap_or_default()
                            } else {
                                vec![]
                            };
                            return (kind.to_string(), name, None, fields, false);
                        }
                    }

                    // Custom struct (type_id == 0) - look up the struct definition
                    if let Some(struct_def) = runtime.get_struct_by_id(struct_id) {
                        let name = struct_def.name.clone();
                        let local_name = name.split('/').next_back().unwrap_or(&name).to_string();
                        let doc = struct_def.docstring.clone();
                        let fields = struct_def.fields.clone();

                        // Check if this is actually an enum variant
                        if let Some(enum_name) = runtime.get_enum_name_for_variant(struct_id) {
                            // It's an enum variant - get the enum's info
                            if let Some(enum_def) = runtime.enums.get(enum_name) {
                                let variant_names: Vec<String> = enum_def
                                    .variants
                                    .iter()
                                    .map(|v| match v {
                                        crate::runtime::EnumVariant::StructVariant {
                                            name, ..
                                        } => name.clone(),
                                        crate::runtime::EnumVariant::StaticVariant { name } => {
                                            name.clone()
                                        }
                                    })
                                    .collect();
                                let enum_local_name = enum_name
                                    .split('/')
                                    .next_back()
                                    .unwrap_or(enum_name)
                                    .to_string();
                                return (
                                    "enum".to_string(),
                                    enum_local_name,
                                    enum_def.docstring.clone(),
                                    variant_names,
                                    false,
                                );
                            }
                        }

                        ("struct".to_string(), local_name, doc, fields, false)
                    } else {
                        (
                            "struct".to_string(),
                            "<unknown>".to_string(),
                            None,
                            vec![],
                            false,
                        )
                    }
                }
            }
        }
    }
}

/// reflect/type-of(value) - Get the type descriptor for any value
/// Uses runtime.type_of which returns proper type descriptors from beagle.core
pub extern "C" fn reflect_type_of(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    match runtime.type_of(stack_pointer, value) {
        Ok(descriptor) => descriptor,
        Err(_) => BuiltInTypes::null_value() as usize,
    }
}

/// reflect/kind(value) - Get the kind of type as a keyword
/// Works on values OR type descriptors - both give the same answer
pub extern "C" fn reflect_kind(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let (kind, _, _, _, _) = get_type_info(runtime, stack_pointer, value);
    runtime
        .intern_keyword(stack_pointer, kind)
        .unwrap_or(BuiltInTypes::null_value() as usize)
}

/// reflect/name(value) - Get the type name as a string
/// Works on values OR type descriptors
pub extern "C" fn reflect_name(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let (_, name, _, _, _) = get_type_info(runtime, stack_pointer, value);
    runtime
        .allocate_string(stack_pointer, name)
        .map(|s| s.into())
        .unwrap_or(BuiltInTypes::null_value() as usize)
}

/// reflect/doc(value) - Get the docstring for a type (or null)
pub extern "C" fn reflect_doc(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let (_, _, doc, _, _) = get_type_info(runtime, stack_pointer, value);
    match doc {
        Some(d) => runtime
            .allocate_string(stack_pointer, d)
            .map(|s| s.into())
            .unwrap_or(BuiltInTypes::null_value() as usize),
        None => BuiltInTypes::null_value() as usize,
    }
}

/// reflect/fields(value) - Get field names for struct types
pub extern "C" fn reflect_fields(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let (_, _, _, fields, _) = get_type_info(runtime, stack_pointer, value);
    build_string_vec(runtime, stack_pointer, &fields)
}

/// reflect/variants(value) - Get variant names for enum types
pub extern "C" fn reflect_variants(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let (_, _, _, variants, _) = get_type_info(runtime, stack_pointer, value);
    build_string_vec(runtime, stack_pointer, &variants)
}

/// reflect/args(value) - Get argument names for function types
pub extern "C" fn reflect_args(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let (_, _, _, args, _) = get_type_info(runtime, stack_pointer, value);
    build_string_vec(runtime, stack_pointer, &args)
}

/// Helper: Build a PersistentVec of strings from a Vec<String>, GC-safe.
pub fn build_string_vec(runtime: &mut Runtime, stack_pointer: usize, strings: &[String]) -> usize {
    use crate::collections::{HandleScope, PersistentVec};
    let mut scope = HandleScope::new(runtime, stack_pointer);
    let vec = match PersistentVec::empty(scope.runtime(), stack_pointer) {
        Ok(v) => v,
        Err(_) => return BuiltInTypes::null_value() as usize,
    };
    let vec_h = scope.alloc(vec.as_tagged());

    for s in strings.iter() {
        let str_val = match scope.runtime().allocate_string(stack_pointer, s.clone()) {
            Ok(v) => v.into(),
            Err(_) => return BuiltInTypes::null_value() as usize,
        };
        // Protect the string on shadow stack before push (which allocates)
        let str_h = scope.alloc(str_val);
        let current_vec = crate::collections::GcHandle::from_tagged(vec_h.get());
        match PersistentVec::push(scope.runtime(), stack_pointer, current_vec, str_h.get()) {
            Ok(new_vec) => {
                // Update vec handle on shadow stack
                let tg = unsafe { &mut *crate::runtime::cached_thread_global_ptr() };
                tg.handle_stack[vec_h.slot()] = new_vec.as_tagged();
            }
            Err(_) => return BuiltInTypes::null_value() as usize,
        }
    }
    vec_h.get()
}

/// reflect/variadic?(value) - Check if function is variadic
pub extern "C" fn reflect_variadic(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let (_, _, _, _, is_variadic) = get_type_info(runtime, stack_pointer, value);
    BuiltInTypes::construct_boolean(is_variadic) as usize
}

/// reflect/info(value) - Get complete type info as a map
pub extern "C" fn reflect_info(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let (kind, name, doc, _extra, is_variadic) = get_type_info(runtime, stack_pointer, value);

    use crate::collections::{HandleScope, PersistentMap};

    let result = (|| -> Result<usize, ()> {
        let mut scope = HandleScope::new(runtime, stack_pointer);

        let map = PersistentMap::empty(scope.runtime(), stack_pointer).map_err(|_| ())?;
        let map_h = scope.alloc(map.as_tagged());

        // Helper: assoc a key-value pair into the map, keeping map_h updated
        macro_rules! map_assoc {
            ($key:expr, $val:expr) => {{
                let key_h = scope.alloc($key);
                let val_h = scope.alloc($val);
                let new_map = PersistentMap::assoc(
                    scope.runtime(),
                    stack_pointer,
                    map_h.get(),
                    key_h.get(),
                    val_h.get(),
                )
                .map_err(|_| ())?;
                let tg = unsafe { &mut *crate::runtime::cached_thread_global_ptr() };
                tg.handle_stack[map_h.slot()] = new_map.as_tagged();
            }};
        }

        // Add :name
        let name_key = scope
            .runtime()
            .intern_keyword(stack_pointer, "name".to_string())
            .map_err(|_| ())?;
        let name_str: usize = scope
            .runtime()
            .allocate_string(stack_pointer, name)
            .map_err(|_| ())?
            .into();
        map_assoc!(name_key, name_str);

        // Add :kind
        let kind_key = scope
            .runtime()
            .intern_keyword(stack_pointer, "kind".to_string())
            .map_err(|_| ())?;
        let kind_kw = scope
            .runtime()
            .intern_keyword(stack_pointer, kind.clone())
            .map_err(|_| ())?;
        map_assoc!(kind_key, kind_kw);

        // Add :doc if present
        if let Some(d) = doc {
            let doc_key = scope
                .runtime()
                .intern_keyword(stack_pointer, "doc".to_string())
                .map_err(|_| ())?;
            let doc_str: usize = scope
                .runtime()
                .allocate_string(stack_pointer, d)
                .map_err(|_| ())?
                .into();
            map_assoc!(doc_key, doc_str);
        }

        // Add kind-specific fields
        if kind == "struct" || kind == "enum" {
            let fields_key = scope
                .runtime()
                .intern_keyword(stack_pointer, "fields".to_string())
                .map_err(|_| ())?;
            let fields_val = reflect_fields(stack_pointer, frame_pointer, value);
            map_assoc!(fields_key, fields_val);
        }

        if kind == "enum" {
            let variants_key = scope
                .runtime()
                .intern_keyword(stack_pointer, "variants".to_string())
                .map_err(|_| ())?;
            let variants_val = reflect_variants(stack_pointer, frame_pointer, value);
            map_assoc!(variants_key, variants_val);
        }

        if kind == "function" {
            let args_key = scope
                .runtime()
                .intern_keyword(stack_pointer, "args".to_string())
                .map_err(|_| ())?;
            let args_val = reflect_args(stack_pointer, frame_pointer, value);
            map_assoc!(args_key, args_val);

            let variadic_key = scope
                .runtime()
                .intern_keyword(stack_pointer, "variadic?".to_string())
                .map_err(|_| ())?;
            let variadic_val = BuiltInTypes::construct_boolean(is_variadic) as usize;
            map_assoc!(variadic_key, variadic_val);
        }

        Ok(map_h.get())
    })();

    result.unwrap_or(BuiltInTypes::null_value() as usize)
}

/// reflect/struct?(value) - Check if value is a struct instance
pub extern "C" fn reflect_is_struct(
    _stack_pointer: usize,
    _frame_pointer: usize,
    value: usize,
) -> usize {
    // A struct is a HeapObject with type_id == 0 (custom struct) and NOT an enum variant
    if !matches!(BuiltInTypes::get_kind(value), BuiltInTypes::HeapObject) {
        return BuiltInTypes::false_value() as usize;
    }
    let heap_obj = HeapObject::from_tagged(value);
    let type_id = heap_obj.get_type_id();
    let struct_id = heap_obj.get_struct_id();
    let runtime = get_runtime().get();
    // Custom struct (type_id == 0) that is NOT an enum variant and NOT a Function struct
    let is_struct = type_id == 0
        && runtime.get_enum_name_for_variant(struct_id).is_none()
        && struct_id != runtime.function_struct_id;
    BuiltInTypes::construct_boolean(is_struct) as usize
}

/// reflect/enum?(value) - Check if value is an enum variant
pub extern "C" fn reflect_is_enum(
    _stack_pointer: usize,
    _frame_pointer: usize,
    value: usize,
) -> usize {
    // An enum variant is a HeapObject with type_id == 0 that IS registered as an enum variant
    if !matches!(BuiltInTypes::get_kind(value), BuiltInTypes::HeapObject) {
        return BuiltInTypes::false_value() as usize;
    }
    let heap_obj = HeapObject::from_tagged(value);
    let type_id = heap_obj.get_type_id();
    let struct_id = heap_obj.get_struct_id();
    let runtime = get_runtime().get();
    // Custom struct (type_id == 0) that IS an enum variant
    let is_enum = type_id == 0 && runtime.get_enum_name_for_variant(struct_id).is_some();
    BuiltInTypes::construct_boolean(is_enum) as usize
}

/// reflect/function?(value) - Check if value is a function
pub extern "C" fn reflect_is_function(
    _stack_pointer: usize,
    _frame_pointer: usize,
    value: usize,
) -> usize {
    let kind = BuiltInTypes::get_kind(value);
    let is_fn = matches!(kind, BuiltInTypes::Function | BuiltInTypes::Closure);
    if !is_fn && matches!(kind, BuiltInTypes::HeapObject) {
        let heap_obj = HeapObject::from_tagged(value);
        let type_id = heap_obj.get_type_id() as u8;
        if type_id == crate::collections::TYPE_ID_FUNCTION_OBJECT
            || type_id == TYPE_ID_MULTI_ARITY_FUNCTION
        {
            return BuiltInTypes::true_value() as usize;
        }
        // Check for Function struct objects (type_id=0, struct_id=function_struct_id)
        if type_id == 0 {
            let struct_id = heap_obj.get_struct_id();
            let runtime = get_runtime().get();
            if struct_id == runtime.function_struct_id {
                return BuiltInTypes::true_value() as usize;
            }
        }
    }
    BuiltInTypes::construct_boolean(is_fn) as usize
}

/// reflect/primitive?(value) - Check if value is a primitive type instance
pub extern "C" fn reflect_is_primitive(
    _stack_pointer: usize,
    _frame_pointer: usize,
    value: usize,
) -> usize {
    let kind = BuiltInTypes::get_kind(value);
    let is_primitive = matches!(
        kind,
        BuiltInTypes::Int | BuiltInTypes::Float | BuiltInTypes::Bool | BuiltInTypes::Null
    ) || (matches!(kind, BuiltInTypes::HeapObject) && {
        let heap_obj = HeapObject::from_tagged(value);
        matches!(
            heap_obj.get_type_id(),
            1 | 2 | 3 | 20 | 22 | 28 | 29 | 30 // Array, String, Keyword, PersistentVector, PersistentMap, PersistentSet, MultiArityFunction, Regex
        )
    });
    BuiltInTypes::construct_boolean(is_primitive) as usize
}

/// reflect/namespace-members(ns) - List all members in a namespace
pub extern "C" fn reflect_namespace_members(
    stack_pointer: usize,
    frame_pointer: usize,
    namespace_name: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Get the namespace name string
    let ns_name = runtime.get_string(stack_pointer, namespace_name);

    // Collect matching local names first to avoid borrow issues
    let prefix = format!("{}/", ns_name);
    let local_names: Vec<String> = runtime
        .functions
        .iter()
        .filter(|func| func.name.starts_with(&prefix))
        .map(|func| {
            func.name
                .split('/')
                .next_back()
                .unwrap_or(&func.name)
                .to_string()
        })
        .collect();

    build_string_vec(runtime, stack_pointer, &local_names)
}

/// reflect/all-namespaces() - List all namespace names
pub extern "C" fn reflect_all_namespaces(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    use std::collections::HashSet;

    // Collect unique namespace names from functions
    let mut namespaces: HashSet<String> = HashSet::new();
    for func in runtime.functions.iter() {
        if let Some(ns) = func.name.split('/').next()
            && !ns.is_empty()
            && ns != func.name
        {
            namespaces.insert(ns.to_string());
        }
    }

    let ns_list: Vec<String> = namespaces.into_iter().collect();
    build_string_vec(runtime, stack_pointer, &ns_list)
}

/// reflect/apropos(query) - Search functions by name/doc substring
pub extern "C" fn reflect_apropos(
    stack_pointer: usize,
    frame_pointer: usize,
    query: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Get the query string
    let query_str = runtime.get_string(stack_pointer, query).to_lowercase();

    // Collect matching function names first to avoid borrow issues
    let matching_names: Vec<String> = runtime
        .functions
        .iter()
        .filter(|func| {
            let name_lower = func.name.to_lowercase();
            name_lower.contains(&query_str)
                || func
                    .docstring
                    .as_ref()
                    .is_some_and(|d| d.to_lowercase().contains(&query_str))
        })
        .map(|func| func.name.clone())
        .collect();

    build_string_vec(runtime, stack_pointer, &matching_names)
}

/// reflect/namespace-info(ns) - Get detailed info about a namespace
pub extern "C" fn reflect_namespace_info(
    stack_pointer: usize,
    frame_pointer: usize,
    namespace_name: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Get the namespace name string
    let ns_name = runtime.get_string(stack_pointer, namespace_name);
    let prefix = format!("{}/", ns_name);

    use crate::collections::{PersistentMap, PersistentVec};

    // Collect function info
    let function_info: Vec<_> = runtime
        .functions
        .iter()
        .filter(|func| func.name.starts_with(&prefix))
        .map(|func| {
            let local_name = func
                .name
                .split('/')
                .next_back()
                .unwrap_or(&func.name)
                .to_string();
            (
                local_name,
                func.docstring.clone(),
                func.arg_names.clone(),
                func.is_variadic,
            )
        })
        .collect();

    // Collect struct info
    let struct_info: Vec<_> = runtime
        .structs
        .iter()
        .filter(|s| {
            s.name.starts_with(&prefix) && {
                let local = s.name.strip_prefix(&prefix).unwrap_or(&s.name);
                !local.contains('.')
            }
        })
        .map(|s| {
            let local_name = s.name.split('/').next_back().unwrap_or(&s.name).to_string();
            (local_name, s.docstring.clone(), s.fields.clone())
        })
        .collect();

    // Collect enum info
    let enum_info: Vec<_> = runtime
        .enums
        .iter()
        .filter(|e| e.name.starts_with(&prefix))
        .map(|e| {
            let local_name = e.name.split('/').next_back().unwrap_or(&e.name).to_string();
            let variant_names: Vec<String> = e
                .variants
                .iter()
                .map(|v| match v {
                    crate::runtime::EnumVariant::StructVariant { name, .. } => name.clone(),
                    crate::runtime::EnumVariant::StaticVariant { name } => name.clone(),
                })
                .collect();
            (local_name, e.docstring.clone(), variant_names)
        })
        .collect();

    // Build the result map using HandleScope for GC safety
    use crate::collections::HandleScope;

    let result = (|| -> Result<usize, ()> {
        let mut scope = HandleScope::new(runtime, stack_pointer);
        let tg_ptr = crate::runtime::cached_thread_global_ptr();

        // Macro to assoc into a map handle, updating it in place
        macro_rules! map_assoc_h {
            ($map_h:expr, $key:expr, $val:expr) => {{
                let key_h = scope.alloc($key);
                let val_h = scope.alloc($val);
                let new_map = PersistentMap::assoc(
                    scope.runtime(),
                    stack_pointer,
                    $map_h.get(),
                    key_h.get(),
                    val_h.get(),
                )
                .map_err(|_| ())?;
                let tg = unsafe { &mut *tg_ptr };
                tg.handle_stack[$map_h.slot()] = new_map.as_tagged();
            }};
        }

        // Macro to push onto a vec handle, updating it in place
        macro_rules! vec_push_h {
            ($vec_h:expr, $val:expr) => {{
                let val_h = scope.alloc($val);
                let current_vec = crate::collections::GcHandle::from_tagged($vec_h.get());
                let new_vec =
                    PersistentVec::push(scope.runtime(), stack_pointer, current_vec, val_h.get())
                        .map_err(|_| ())?;
                let tg = unsafe { &mut *tg_ptr };
                tg.handle_stack[$vec_h.slot()] = new_vec.as_tagged();
            }};
        }

        // Create the main map
        let map = PersistentMap::empty(scope.runtime(), stack_pointer).map_err(|_| ())?;
        let map_h = scope.alloc(map.as_tagged());

        // Add :name
        let name_key = scope
            .runtime()
            .intern_keyword(stack_pointer, "name".to_string())
            .map_err(|_| ())?;
        let name_val = scope
            .runtime()
            .allocate_string(stack_pointer, ns_name.to_string())
            .map(|s| s.into())
            .map_err(|_| ())?;
        map_assoc_h!(map_h, name_key, name_val);

        // Build :functions vector
        let funcs_key = scope
            .runtime()
            .intern_keyword(stack_pointer, "functions".to_string())
            .map_err(|_| ())?;
        let funcs_key_h = scope.alloc(funcs_key);
        let funcs_vec = PersistentVec::empty(scope.runtime(), stack_pointer).map_err(|_| ())?;
        let funcs_vec_h = scope.alloc(funcs_vec.as_tagged());

        for (name, doc, args, variadic) in function_info {
            let func_map = PersistentMap::empty(scope.runtime(), stack_pointer).map_err(|_| ())?;
            let func_map_h = scope.alloc(func_map.as_tagged());

            let k = scope
                .runtime()
                .intern_keyword(stack_pointer, "name".to_string())
                .map_err(|_| ())?;
            let v = scope
                .runtime()
                .allocate_string(stack_pointer, name)
                .map(|s| s.into())
                .map_err(|_| ())?;
            map_assoc_h!(func_map_h, k, v);

            let k = scope
                .runtime()
                .intern_keyword(stack_pointer, "doc".to_string())
                .map_err(|_| ())?;
            let v = match doc {
                Some(d) => scope
                    .runtime()
                    .allocate_string(stack_pointer, d)
                    .map(|s| s.into())
                    .map_err(|_| ())?,
                None => BuiltInTypes::null_value() as usize,
            };
            map_assoc_h!(func_map_h, k, v);

            // Build args vector using build_string_vec
            let k = scope
                .runtime()
                .intern_keyword(stack_pointer, "args".to_string())
                .map_err(|_| ())?;
            let k_h = scope.alloc(k);
            let args_val = build_string_vec(scope.runtime(), stack_pointer, &args);
            map_assoc_h!(func_map_h, k_h.get(), args_val);

            let k = scope
                .runtime()
                .intern_keyword(stack_pointer, "variadic?".to_string())
                .map_err(|_| ())?;
            let v = if variadic {
                BuiltInTypes::true_value() as usize
            } else {
                BuiltInTypes::false_value() as usize
            };
            map_assoc_h!(func_map_h, k, v);

            vec_push_h!(funcs_vec_h, func_map_h.get());
        }

        map_assoc_h!(map_h, funcs_key_h.get(), funcs_vec_h.get());

        // Build :structs vector
        let structs_key = scope
            .runtime()
            .intern_keyword(stack_pointer, "structs".to_string())
            .map_err(|_| ())?;
        let structs_key_h = scope.alloc(structs_key);
        let structs_vec = PersistentVec::empty(scope.runtime(), stack_pointer).map_err(|_| ())?;
        let structs_vec_h = scope.alloc(structs_vec.as_tagged());

        for (name, doc, fields) in struct_info {
            let struct_map =
                PersistentMap::empty(scope.runtime(), stack_pointer).map_err(|_| ())?;
            let struct_map_h = scope.alloc(struct_map.as_tagged());

            let k = scope
                .runtime()
                .intern_keyword(stack_pointer, "name".to_string())
                .map_err(|_| ())?;
            let v = scope
                .runtime()
                .allocate_string(stack_pointer, name)
                .map(|s| s.into())
                .map_err(|_| ())?;
            map_assoc_h!(struct_map_h, k, v);

            let k = scope
                .runtime()
                .intern_keyword(stack_pointer, "doc".to_string())
                .map_err(|_| ())?;
            let v = match doc {
                Some(d) => scope
                    .runtime()
                    .allocate_string(stack_pointer, d)
                    .map(|s| s.into())
                    .map_err(|_| ())?,
                None => BuiltInTypes::null_value() as usize,
            };
            map_assoc_h!(struct_map_h, k, v);

            // Build fields vector using build_string_vec
            let k = scope
                .runtime()
                .intern_keyword(stack_pointer, "fields".to_string())
                .map_err(|_| ())?;
            let k_h = scope.alloc(k);
            let fields_val = build_string_vec(scope.runtime(), stack_pointer, &fields);
            map_assoc_h!(struct_map_h, k_h.get(), fields_val);

            vec_push_h!(structs_vec_h, struct_map_h.get());
        }

        map_assoc_h!(map_h, structs_key_h.get(), structs_vec_h.get());

        // Build :enums vector
        let enums_key = scope
            .runtime()
            .intern_keyword(stack_pointer, "enums".to_string())
            .map_err(|_| ())?;
        let enums_key_h = scope.alloc(enums_key);
        let enums_vec = PersistentVec::empty(scope.runtime(), stack_pointer).map_err(|_| ())?;
        let enums_vec_h = scope.alloc(enums_vec.as_tagged());

        for (name, doc, variants) in enum_info {
            let enum_map = PersistentMap::empty(scope.runtime(), stack_pointer).map_err(|_| ())?;
            let enum_map_h = scope.alloc(enum_map.as_tagged());

            let k = scope
                .runtime()
                .intern_keyword(stack_pointer, "name".to_string())
                .map_err(|_| ())?;
            let v = scope
                .runtime()
                .allocate_string(stack_pointer, name)
                .map(|s| s.into())
                .map_err(|_| ())?;
            map_assoc_h!(enum_map_h, k, v);

            let k = scope
                .runtime()
                .intern_keyword(stack_pointer, "doc".to_string())
                .map_err(|_| ())?;
            let v = match doc {
                Some(d) => scope
                    .runtime()
                    .allocate_string(stack_pointer, d)
                    .map(|s| s.into())
                    .map_err(|_| ())?,
                None => BuiltInTypes::null_value() as usize,
            };
            map_assoc_h!(enum_map_h, k, v);

            // Build variants vector using build_string_vec
            let k = scope
                .runtime()
                .intern_keyword(stack_pointer, "variants".to_string())
                .map_err(|_| ())?;
            let k_h = scope.alloc(k);
            let variants_val = build_string_vec(scope.runtime(), stack_pointer, &variants);
            map_assoc_h!(enum_map_h, k_h.get(), variants_val);

            vec_push_h!(enums_vec_h, enum_map_h.get());
        }

        map_assoc_h!(map_h, enums_key_h.get(), enums_vec_h.get());

        Ok(map_h.get())
    })();

    result.unwrap_or(BuiltInTypes::null_value() as usize)
}
