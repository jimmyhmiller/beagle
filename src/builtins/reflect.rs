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

/// Resolve a value (function, struct instance/descriptor, or enum
/// variant/descriptor) to the `source_text` stored on the matching
/// Function/Struct/Enum/Binding record. Returns `None` if no source is
/// available.
fn resolve_source_text(runtime: &Runtime, stack_pointer: usize, value: usize) -> Option<String> {
    // String argument: treat as a definition name (lets agents look up
    // top-level `let`-bindings, whose values can't be reverse-mapped).
    if value_is_string(value) {
        let name = runtime.get_string(stack_pointer, value);
        return resolve_by_name(runtime, &name).and_then(|d| d.source_text);
    }

    let tag = BuiltInTypes::get_kind(value);

    if matches!(tag, BuiltInTypes::Function) {
        let fn_ptr = BuiltInTypes::untag(value) as *const u8;
        return runtime
            .get_function_by_pointer(fn_ptr)
            .and_then(|f| f.source_text.clone());
    }

    if matches!(tag, BuiltInTypes::Closure) && BuiltInTypes::is_heap_pointer(value) {
        let heap_obj = HeapObject::from_tagged(value);
        let fn_ptr = BuiltInTypes::untag(heap_obj.get_field(0)) as *const u8;
        return runtime
            .get_function_by_pointer(fn_ptr)
            .and_then(|f| f.source_text.clone());
    }

    if matches!(tag, BuiltInTypes::HeapObject) {
        let heap_obj = HeapObject::from_tagged(value);
        let type_id = heap_obj.get_type_id();
        let struct_id = heap_obj.get_struct_id();

        // Function struct object (first-class fn via heap wrapper)
        if type_id == 0 && struct_id == runtime.function_struct_id {
            let fn_ptr_tagged = heap_obj.get_field(0);
            let fn_ptr = BuiltInTypes::untag(fn_ptr_tagged) as *const u8;
            return runtime
                .get_function_by_pointer(fn_ptr)
                .and_then(|f| f.source_text.clone());
        }

        // Type descriptor: an instance of beagle.core/Struct whose field 0
        // is the fully-qualified type name. Use it to look up the actual
        // struct or enum definition.
        let struct_struct_id = runtime.get_struct("beagle.core/Struct").map(|(id, _)| id);
        let fields_size = heap_obj.fields_size() / 8;
        if struct_struct_id == Some(struct_id) && fields_size >= 1 {
            let name_ptr = heap_obj.get_field(0);
            let full_name = runtime.get_string(stack_pointer, name_ptr);
            if let Some((_, s)) = runtime.get_struct(&full_name)
                && s.source_text.is_some()
            {
                return s.source_text.clone();
            }
            if let Some(e) = runtime.enums.get(&full_name)
                && e.source_text.is_some()
            {
                return e.source_text.clone();
            }
        }

        // Regular struct instance (or enum variant instance): look up by struct_id
        if let Some(s) = runtime.get_struct_by_id(struct_id) {
            if let Some(enum_name) = runtime.get_enum_name_for_variant(struct_id)
                && let Some(e) = runtime.enums.get(enum_name)
                && e.source_text.is_some()
            {
                return e.source_text.clone();
            }
            // Enum companion struct (same name as the enum) — prefer
            // the enum's source over the companion's empty source.
            if let Some(e) = runtime.enums.get(&s.name)
                && e.source_text.is_some()
            {
                return e.source_text.clone();
            }
            if s.source_text.is_some() {
                return s.source_text.clone();
            }
        }
    }

    None
}

/// reflect/source(value) - Return the original source text for a definition.
///
/// Accepts a function value, struct/enum value or type descriptor. Returns
/// `null` if no source text is stored (e.g. for builtins, foreign functions,
/// or anonymous closures).
pub extern "C" fn reflect_source(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    match resolve_source_text(runtime, stack_pointer, value) {
        Some(text) => runtime
            .allocate_string(stack_pointer, text)
            .map(|s| s.into())
            .unwrap_or(BuiltInTypes::null_value() as usize),
        None => BuiltInTypes::null_value() as usize,
    }
}

/// reflect/namespace-source(namespace) - Return the concatenated source text
/// of every definition in the given namespace, in registration order.
/// Members without stored source (builtins, foreign fns) are skipped.
/// Returns `null` if the namespace has no members with source.
pub extern "C" fn reflect_namespace_source(
    stack_pointer: usize,
    frame_pointer: usize,
    namespace_name: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let ns_name = runtime.get_string(stack_pointer, namespace_name);
    let prefix = format!("{}/", ns_name);

    let mut blocks: Vec<String> = Vec::new();

    // Structs (excluding enum variant structs, which are emitted as part of
    // the enum's source block).
    for s in runtime.structs.iter() {
        if !s.name.starts_with(&prefix) {
            continue;
        }
        let local = s.name.strip_prefix(&prefix).unwrap_or(&s.name);
        if local.contains('.') {
            continue;
        }
        if let Some(text) = &s.source_text {
            blocks.push(text.clone());
        }
    }

    // Enums
    for e in runtime.enums.iter() {
        if !e.name.starts_with(&prefix) {
            continue;
        }
        if let Some(text) = &e.source_text {
            blocks.push(text.clone());
        }
    }

    // Top-level `let`-bindings
    for b in runtime.bindings.iter() {
        if !b.name.starts_with(&prefix) {
            continue;
        }
        if let Some(text) = &b.source_text {
            blocks.push(text.clone());
        }
    }

    // Functions
    for f in runtime.functions.iter() {
        if !f.name.starts_with(&prefix) {
            continue;
        }
        if let Some(text) = &f.source_text {
            blocks.push(text.clone());
        }
    }

    if blocks.is_empty() {
        return BuiltInTypes::null_value() as usize;
    }

    let mut out = String::new();
    for (i, block) in blocks.iter().enumerate() {
        if i > 0 {
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push('\n');
        }
        out.push_str(block);
    }

    runtime
        .allocate_string(stack_pointer, out)
        .map(|s| s.into())
        .unwrap_or(BuiltInTypes::null_value() as usize)
}

/// Summary of what `resolve_definition` found for a value. Captures what
/// the two disk-facing builtins need: the definition's fully-qualified
/// name (for namespace-scoped recompilation), its current source text
/// (for the integrity check), and its on-disk location.
struct DefinitionInfo {
    full_name: String,
    source_text: Option<String>,
    disk_location: Option<DiskLocation>,
}

/// True if `value` is any kind of string (constant, heap, slice, cons).
/// Used so that `reflect/source(name)` / `reflect/write-source(name, ...)`
/// can accept a bare string name in place of the live value, which is
/// the only way to address `let`-bindings whose value is an integer,
/// map, or other type that can't be reverse-mapped to its binding slot.
fn value_is_string(value: usize) -> bool {
    match BuiltInTypes::get_kind(value) {
        BuiltInTypes::String => true,
        BuiltInTypes::HeapObject if BuiltInTypes::is_heap_pointer(value) => {
            let heap = HeapObject::from_tagged(value);
            let t = heap.get_header().type_id;
            t == TYPE_ID_STRING || t == TYPE_ID_STRING_SLICE || t == TYPE_ID_CONS_STRING
        }
        _ => false,
    }
}

/// Resolve a `DefinitionInfo` for a name string. Searches, in order:
/// Binding (top-level `let`), Function, Struct, Enum. The name is tried
/// verbatim first (e.g. `"my.ns/foo"`) then, if unqualified, against
/// the current namespace (e.g. `"foo"` → `"<current>/foo"`).
fn resolve_by_name(runtime: &Runtime, name: &str) -> Option<DefinitionInfo> {
    let candidates: Vec<String> = if name.contains('/') {
        vec![name.to_string()]
    } else {
        let ns = runtime.current_namespace_name();
        vec![format!("{}/{}", ns, name), name.to_string()]
    };

    for candidate in &candidates {
        if let Some(b) = runtime.bindings.get(candidate) {
            return Some(DefinitionInfo {
                full_name: b.name.clone(),
                source_text: b.source_text.clone(),
                disk_location: b.disk_location.clone(),
            });
        }
        if let Some(f) = runtime.functions.iter().find(|f| f.name == *candidate) {
            return Some(DefinitionInfo {
                full_name: f.name.clone(),
                source_text: f.source_text.clone(),
                disk_location: f.disk_location.clone(),
            });
        }
        if let Some((_, s)) = runtime.get_struct(candidate) {
            return Some(DefinitionInfo {
                full_name: s.name.clone(),
                source_text: s.source_text.clone(),
                disk_location: s.disk_location.clone(),
            });
        }
        if let Some(e) = runtime.enums.get(candidate) {
            return Some(DefinitionInfo {
                full_name: e.name.clone(),
                source_text: e.source_text.clone(),
                disk_location: e.disk_location.clone(),
            });
        }
    }
    None
}

/// Look up a runtime record for a value following the same dispatch logic
/// as `resolve_source_text`, and return a bundled `DefinitionInfo`.
fn resolve_definition(
    runtime: &Runtime,
    stack_pointer: usize,
    value: usize,
) -> Option<DefinitionInfo> {
    if value_is_string(value) {
        let name = runtime.get_string(stack_pointer, value);
        return resolve_by_name(runtime, &name);
    }
    let tag = BuiltInTypes::get_kind(value);

    if matches!(tag, BuiltInTypes::Function) {
        let fn_ptr = BuiltInTypes::untag(value) as *const u8;
        let f = runtime.get_function_by_pointer(fn_ptr)?;
        return Some(DefinitionInfo {
            full_name: f.name.clone(),
            source_text: f.source_text.clone(),
            disk_location: f.disk_location.clone(),
        });
    }

    if matches!(tag, BuiltInTypes::Closure) && BuiltInTypes::is_heap_pointer(value) {
        let heap_obj = HeapObject::from_tagged(value);
        let fn_ptr = BuiltInTypes::untag(heap_obj.get_field(0)) as *const u8;
        let f = runtime.get_function_by_pointer(fn_ptr)?;
        return Some(DefinitionInfo {
            full_name: f.name.clone(),
            source_text: f.source_text.clone(),
            disk_location: f.disk_location.clone(),
        });
    }

    if matches!(tag, BuiltInTypes::HeapObject) {
        let heap_obj = HeapObject::from_tagged(value);
        let type_id = heap_obj.get_type_id();
        let struct_id = heap_obj.get_struct_id();

        if type_id == 0 && struct_id == runtime.function_struct_id {
            let fn_ptr_tagged = heap_obj.get_field(0);
            let fn_ptr = BuiltInTypes::untag(fn_ptr_tagged) as *const u8;
            if let Some(f) = runtime.get_function_by_pointer(fn_ptr) {
                return Some(DefinitionInfo {
                    full_name: f.name.clone(),
                    source_text: f.source_text.clone(),
                    disk_location: f.disk_location.clone(),
                });
            }
        }

        let struct_struct_id = runtime.get_struct("beagle.core/Struct").map(|(id, _)| id);
        let fields_size = heap_obj.fields_size() / 8;
        if struct_struct_id == Some(struct_id) && fields_size >= 1 {
            let name_ptr = heap_obj.get_field(0);
            let full_name = runtime.get_string(stack_pointer, name_ptr);
            // Enums get registered both as an `Enum` and as a companion
            // `Struct` (used for variant dispatch); the real source and
            // disk origin live on the `Enum` record. Check enums first so
            // type descriptors for enums resolve to the right record.
            if let Some(e) = runtime.enums.get(&full_name) {
                return Some(DefinitionInfo {
                    full_name: e.name.clone(),
                    source_text: e.source_text.clone(),
                    disk_location: e.disk_location.clone(),
                });
            }
            if let Some((_, s)) = runtime.get_struct(&full_name) {
                return Some(DefinitionInfo {
                    full_name: s.name.clone(),
                    source_text: s.source_text.clone(),
                    disk_location: s.disk_location.clone(),
                });
            }
        }

        if let Some(s) = runtime.get_struct_by_id(struct_id) {
            // Variant instance → walk up to the enum.
            if let Some(enum_name) = runtime.get_enum_name_for_variant(struct_id) {
                if let Some(e) = runtime.enums.get(enum_name) {
                    return Some(DefinitionInfo {
                        full_name: e.name.clone(),
                        source_text: e.source_text.clone(),
                        disk_location: e.disk_location.clone(),
                    });
                }
            }
            // The enum's parent "companion" struct shares its name with
            // the enum. A bare reference like `Mode` evaluates to an
            // instance of this companion. Prefer the enum's source/disk
            // info when both exist.
            if let Some(e) = runtime.enums.get(&s.name) {
                return Some(DefinitionInfo {
                    full_name: e.name.clone(),
                    source_text: e.source_text.clone(),
                    disk_location: e.disk_location.clone(),
                });
            }
            return Some(DefinitionInfo {
                full_name: s.name.clone(),
                source_text: s.source_text.clone(),
                disk_location: s.disk_location.clone(),
            });
        }
    }

    None
}

/// Build the map `{:file, :byte-start, :byte-end, :line-start, :line-end}`
/// GC-safely using a HandleScope, mirroring the shape of `reflect/info`.
fn build_location_map(
    runtime: &mut Runtime,
    stack_pointer: usize,
    loc: &DiskLocation,
) -> Option<usize> {
    use crate::collections::{HandleScope, PersistentMap};

    let mut scope = HandleScope::new(runtime, stack_pointer);
    let tg_ptr = crate::runtime::cached_thread_global_ptr();

    let map = PersistentMap::empty(scope.runtime(), stack_pointer).ok()?;
    let map_h = scope.alloc(map.as_tagged());

    macro_rules! put {
        ($key:expr, $val:expr) => {{
            let k = scope
                .runtime()
                .intern_keyword(stack_pointer, $key.to_string())
                .ok()?;
            let k_h = scope.alloc(k);
            let v_h = scope.alloc($val);
            let new_map = PersistentMap::assoc(
                scope.runtime(),
                stack_pointer,
                map_h.get(),
                k_h.get(),
                v_h.get(),
            )
            .ok()?;
            let tg = unsafe { &mut *tg_ptr };
            tg.handle_stack[map_h.slot()] = new_map.as_tagged();
        }};
    }

    let file_str = scope
        .runtime()
        .allocate_string(stack_pointer, loc.file.clone())
        .ok()?
        .into();
    put!("file", file_str);
    put!(
        "byte-start",
        BuiltInTypes::construct_int(loc.byte_start as isize) as usize
    );
    put!(
        "byte-end",
        BuiltInTypes::construct_int(loc.byte_end as isize) as usize
    );
    put!(
        "line-start",
        BuiltInTypes::construct_int(loc.line_start as isize) as usize
    );
    put!(
        "line-end",
        BuiltInTypes::construct_int(loc.line_end as isize) as usize
    );

    Some(map_h.get())
}

/// reflect/location(value) - Return `{:file, :byte-start, :byte-end,
/// :line-start, :line-end}` for a definition that was loaded from disk,
/// or `null` for REPL/eval definitions, builtins, and foreign functions.
///
/// The byte range covers the full block as it lives on disk, including
/// any preceding `///` doc comment lines. Pair with
/// `reflect/write-source` to persist edits back to the file.
pub extern "C" fn reflect_location(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let info = match resolve_definition(runtime, stack_pointer, value) {
        Some(info) => info,
        None => return BuiltInTypes::null_value() as usize,
    };
    let Some(loc) = info.disk_location else {
        return BuiltInTypes::null_value() as usize;
    };
    build_location_map(runtime, stack_pointer, &loc).unwrap_or(BuiltInTypes::null_value() as usize)
}

/// Extract the namespace (the portion before the final `/`) from a
/// fully-qualified definition name like `"my.ns/foo"`. Returns `None`
/// for global/unqualified names.
fn namespace_of(full_name: &str) -> Option<&str> {
    full_name.rsplit_once('/').map(|(ns, _)| ns)
}

/// Shift the byte ranges of every runtime record whose disk origin lies
/// in `file`, at or after `after_byte`, by `delta` (positive = later in
/// file, negative = earlier). Called after a successful
/// `reflect/write-source` so subsequent edits in the same file use
/// up-to-date byte ranges.
fn shift_byte_ranges_after(runtime: &mut Runtime, file: &str, after_byte: usize, delta: isize) {
    let apply = |loc: &mut DiskLocation| {
        if loc.file != file || loc.byte_start < after_byte {
            return;
        }
        if delta >= 0 {
            let d = delta as usize;
            loc.byte_start = loc.byte_start.wrapping_add(d);
            loc.byte_end = loc.byte_end.wrapping_add(d);
        } else {
            let d = (-delta) as usize;
            loc.byte_start = loc.byte_start.saturating_sub(d);
            loc.byte_end = loc.byte_end.saturating_sub(d);
        }
    };

    for f in runtime.functions.iter_mut() {
        if let Some(loc) = f.disk_location.as_mut() {
            apply(loc);
        }
    }
    runtime.structs.for_each_mut(|s| {
        if let Some(loc) = s.disk_location.as_mut() {
            apply(loc);
        }
    });
    runtime.enums.for_each_mut(|e| {
        if let Some(loc) = e.disk_location.as_mut() {
            apply(loc);
        }
    });
    runtime.bindings.for_each_mut(|b| {
        if let Some(loc) = b.disk_location.as_mut() {
            apply(loc);
        }
    });
}

/// reflect/write-source(value, new-text) - Persist an edited definition
/// back to its source file and re-register it in the runtime.
///
/// Semantics:
///   1. Resolve `value` to a disk-resident Function/Struct/Enum.
///   2. Verify the file has not drifted since load — the bytes at the
///      recorded range must equal the stored `source_text`.
///   3. Splice `new-text` into the file at that byte range and write it.
///   4. Shift byte ranges of later definitions in the same file by the
///      length delta.
///   5. Compile `new-text` in the definition's namespace with file
///      context so the replacement's `disk_location` matches its new
///      position in the file.
///
/// Returns `true` on success. Throws with a descriptive string on any
/// failure (missing location, integrity mismatch, I/O error, compile
/// error). Existing definitions that existed before the edit but aren't
/// produced by the new text remain in the runtime unchanged.
pub extern "C" fn reflect_write_source(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
    new_text: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let info = match resolve_definition(runtime, stack_pointer, value) {
        Some(info) => info,
        None => {
            throw_write_source_error(
                stack_pointer,
                "reflect/write-source: value has no resolvable definition",
            );
        }
    };
    let Some(loc) = info.disk_location.clone() else {
        throw_write_source_error(
            stack_pointer,
            "reflect/write-source: definition has no on-disk origin (eval/REPL defs cannot be persisted)",
        );
    };
    let Some(original_source) = info.source_text.clone() else {
        throw_write_source_error(
            stack_pointer,
            "reflect/write-source: definition has no stored source text to verify against",
        );
    };

    let new_text_str = runtime.get_string(stack_pointer, new_text);

    // Read current file bytes.
    let file_contents = match std::fs::read_to_string(&loc.file) {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("reflect/write-source: failed to read {}: {}", loc.file, e);
            throw_write_source_error(stack_pointer, &msg);
        }
    };

    // Bounds + integrity check. If the file was edited outside the runtime
    // between load and now, we refuse rather than silently stomping.
    if loc.byte_end > file_contents.len() || loc.byte_start > loc.byte_end {
        throw_write_source_error(
            stack_pointer,
            "reflect/write-source: stored byte range is out of bounds for the current file",
        );
    }
    let current_slice = &file_contents[loc.byte_start..loc.byte_end];
    if current_slice != original_source {
        throw_write_source_error(
            stack_pointer,
            "reflect/write-source: file contents have changed since this definition was loaded (re-load and retry)",
        );
    }

    // Splice.
    let mut new_contents =
        String::with_capacity(file_contents.len() - original_source.len() + new_text_str.len());
    new_contents.push_str(&file_contents[..loc.byte_start]);
    new_contents.push_str(&new_text_str);
    new_contents.push_str(&file_contents[loc.byte_end..]);

    if let Err(e) = std::fs::write(&loc.file, &new_contents) {
        let msg = format!("reflect/write-source: failed to write {}: {}", loc.file, e);
        throw_write_source_error(stack_pointer, &msg);
    }

    // Shift subsequent definitions in the same file. `delta` is the
    // signed length change; earlier defs are untouched, later ones slide
    // by exactly this amount.
    let old_len = loc.byte_end - loc.byte_start;
    let new_len = new_text_str.len();
    let delta = new_len as isize - old_len as isize;
    shift_byte_ranges_after(runtime, &loc.file, loc.byte_end, delta);

    // The record we're about to overwrite in compile doesn't know its new
    // end yet — patch its range up-front so the compiler's upsert
    // (which leaves disk_location sticky) doesn't revert our patch.
    let new_byte_end = loc.byte_start + new_len;
    let new_line_end = loc.line_start + new_text_str.matches('\n').count();
    let patched_loc = DiskLocation {
        file: loc.file.clone(),
        byte_start: loc.byte_start,
        byte_end: new_byte_end,
        line_start: loc.line_start,
        line_end: new_line_end,
    };
    patch_disk_location(runtime, &info.full_name, patched_loc);

    // Recompile the edited text in the target namespace with file context
    // so the new definition's disk_location points at the updated byte
    // range. Line offset is `line_start - 1` because the fragment's own
    // first line maps to line 1 internally.
    let namespace = namespace_of(&info.full_name).unwrap_or("").to_string();
    // Snapshot the pending heap-binding queue length so we only flush
    // the entries the recompile itself queues, not accumulated cruft
    // from earlier compiler-thread reservations (which would otherwise
    // re-play stale null placeholders and clobber live bindings for
    // unrelated definitions in the same namespace).
    let pending_start = runtime.pending_heap_bindings_len();

    let top_level_ptr = match runtime.compile_string_with_file_context(
        &new_text_str,
        &namespace,
        &loc.file,
        loc.byte_start,
        loc.line_start.saturating_sub(1),
    ) {
        Ok(ptr) => ptr,
        Err(e) => {
            let msg = format!(
                "reflect/write-source: wrote {} but re-compile failed: {}",
                loc.file, e
            );
            throw_write_source_error(stack_pointer, &msg);
        }
    };

    // Flush only the bindings this recompile queued. Covers the function
    // case where `upsert_function` on the compiler thread adds the
    // new first-class wrapper via `add_binding` and queues the heap-side
    // update for later. Without flushing, reads of the edited name
    // would return the pre-edit wrapper pointing at the old code.
    runtime.flush_pending_heap_bindings_from(stack_pointer, pending_start);

    // Execute the compiled top-level so struct/enum definitions update
    // their type descriptor bindings in the namespace. Functions don't
    // strictly require this (upsert_function already rebinds), but
    // running it keeps the behavior uniform with `eval` and ensures the
    // edited name resolves to the new definition everywhere.
    if top_level_ptr != 0 {
        let top_level: fn() -> usize = unsafe { std::mem::transmute(top_level_ptr) };
        top_level();
    }

    BuiltInTypes::construct_boolean(true) as usize
}

/// Thin wrapper around `throw_runtime_error` that fills in the
/// `"write-source"` kind string, so we can raise with a single call
/// inside `reflect_write_source`.
fn throw_write_source_error(stack_pointer: usize, message: &str) -> ! {
    // `throw_runtime_error` expects a valid `SystemError` variant name
    // as the kind. `RuntimeError` is the catch-all for runtime failures
    // that aren't tied to parsing, typing, or I/O specifically.
    unsafe {
        crate::builtins::throw_runtime_error(stack_pointer, "RuntimeError", message.to_string())
    }
}

/// Overwrite the `disk_location` on whichever runtime record (fn, struct,
/// or enum) is registered under `full_name`. Used by write-source to
/// pre-emptively record the new range before recompilation, so that even
/// if the new text doesn't redefine the same name (e.g. a rename),
/// subsequent lookups under the old name see a sensible location.
fn patch_disk_location(runtime: &mut Runtime, full_name: &str, loc: DiskLocation) {
    if let Some(f) = runtime.functions.iter_mut().find(|f| f.name == full_name) {
        f.disk_location = Some(loc.clone());
        return;
    }
    if runtime.patch_struct_disk_location(full_name, loc.clone()) {
        return;
    }
    if runtime.patch_enum_disk_location(full_name, loc.clone()) {
        return;
    }
    runtime.patch_binding_disk_location(full_name, loc);
}
