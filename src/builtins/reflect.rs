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

/// Tier-up trampoline. Called from JIT'd code via the per-function
/// entry counter when a function has accumulated enough calls to
/// warrant specialization. `name_ptr` is a stable C-string holding
/// the function's fully-qualified name (allocated by
/// `Compiler::add_function_counter` and kept alive in
/// `Compiler::function_counter_names`).
///
/// Idempotent — the compiler skips already-specialized functions, so
/// the worst case for repeated firings (e.g. multiple threads racing
/// past the threshold) is one extra channel round-trip per race.
///
/// Uses a system-V calling convention: name_ptr is in the first arg
/// register. Beagle's normal arg registers x0–x7 are caller-saved
/// across this call by the function-entry tier-up check, so we don't
/// need to preserve them here ourselves — the caller has already
/// arranged for them to be in locals.
pub extern "C" fn tier_up_trampoline(name_ptr: *const std::ffi::c_char) {
    if name_ptr.is_null() {
        return;
    }
    let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) };
    if let Ok(name_str) = name.to_str() {
        // Spawn a short-lived thread to send the SpecializeFunction
        // message asynchronously — the channel is rendezvous-style and
        // would otherwise block the running Beagle function for the
        // duration of the recompile (~10ms on fib, more on bigger
        // bodies). The trampoline fires at most once per function
        // (counter never decrements past zero again), so this spawns
        // at most O(number-of-eligible-functions) short threads over
        // the program's lifetime.
        let owned = name_str.to_string();
        std::thread::spawn(move || {
            let runtime = get_runtime().get_mut();
            // Compile + STAGE the tier-2 install on the compiler thread.
            runtime.specialize_function(&owned);
            // Then apply it in a stop-the-world from THIS spawn thread (a
            // non-registered coordinator). The triggering mutator parks at its
            // next entry safepoint; the compiler thread stays free to service
            // other mutators' messages so they can park too (no deadlock).
            runtime.stop_world_and_apply_installs(0);
        });
    }
}

/// Sentinel returned by `osr_trampoline` to mean "did NOT transfer — keep
/// running tier-1." The OsrCheck codegen compares the trampoline's return
/// against this; on a match it falls through to the back-edge, otherwise it
/// returns the value (the OSR continuation's result) from the function.
/// `usize::MAX` is never a value the runtime produces as a Beagle return
/// (it decodes to a Null-tagged value with an all-ones payload, which the
/// canonical Null — `0b111` — never equals).
pub const OSR_NO_OSR: usize = usize::MAX;

/// Debug-only: time of the first OSR back-edge trip, for measuring
/// build-to-transfer latency under `BEAGLE_OSR_DEBUG`.
static OSR_FIRST_TRIP: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

/// OSR back-edge trampoline (Phase 1: logging only).
///
/// Emitted at hot loop latches in tier-1 code via the generic
/// `Instruction::TierUpCheck` primitive (decrement a per-loop counter;
/// when it reaches zero, call this with the loop's key string). Reuses
/// the same calling convention as `tier_up_trampoline`: the key C-string
/// pointer is in the first arg register, and Beagle's arg registers are
/// caller-saved across this call (the loop's live values live in
/// callee-saved registers `X19-X27`, preserved by this `extern "C"`
/// frame), so no preservation is needed here.
///
/// Fires at most once per loop instance (the counter never returns to
/// zero after the first trip). Later phases will, instead of logging,
/// compile a tier-2 OSR-entry variant and transfer the running frame
/// into it. For now this only confirms the instrumentation points fire
/// and lets us measure the per-iteration counter overhead.
pub extern "C" fn osr_trampoline(
    key_ptr: *const std::ffi::c_char,
    counter_addr: usize,
    sp: usize,
    fp: usize,
) -> usize {
    use crate::osr::OsrState;

    // How long to wait (loop iterations) before re-checking while the
    // continuation is still being built on the compiler thread.
    let recheck: i64 = std::env::var("BEAGLE_OSR_RECHECK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50_000);

    if std::env::var("BEAGLE_OSR_DEBUG").is_ok() {
        OSR_FIRST_TRIP.get_or_init(std::time::Instant::now);
    }
    if key_ptr.is_null() {
        return OSR_NO_OSR;
    }
    let key = match unsafe { std::ffi::CStr::from_ptr(key_ptr) }.to_str() {
        Ok(k) => k,
        Err(_) => return OSR_NO_OSR,
    };
    // Re-arm the back-edge counter so the check fires again later (the trip
    // that finds the continuation Ready will transfer instead).
    let rearm = |val: i64| unsafe { *(counter_addr as *mut i64) = val };

    match crate::osr::osr_lookup(key) {
        Some(OsrState::Ready {
            code_addr,
            live_in_slots,
        }) => {
            // Re-entrancy guard: if F_osr is already executing for this key on
            // this thread, we're inside a deopt re-invoke of generic F (which
            // re-runs the same loop). Decline so generic F completes in tier-1
            // instead of nesting another transfer (which could recurse without
            // bound if the guard keeps failing).
            if !crate::osr::osr_transfer_begin(key) {
                rearm(recheck);
                return OSR_NO_OSR;
            }
            // Transfer. The loop's live state is in F's frame at fixed
            // FP-relative slots; read it (in argument order), call the
            // optimized continuation, and return its result — which the
            // OsrCheck codegen returns from F. GC may run inside the call,
            // so publish F's frame first.
            save_gc_context!(sp, fp);
            // Pack the live-ins (read from F's frame, in slot order) into a
            // buffer and pass its pointer as F_osr's single argument. F_osr
            // unpacks it into slots before any safepoint, so the not-yet-rooted
            // buffer values can't be moved by GC mid-unpack.
            let buf: Vec<usize> = live_in_slots
                .iter()
                .map(|&k| unsafe { *((fp - (k + 3) * 8) as *const usize) })
                .collect();
            if std::env::var("BEAGLE_OSR_DEBUG").is_ok() {
                let elapsed = OSR_FIRST_TRIP.get().map(|t| t.elapsed());
                eprintln!(
                    "[OSR-xfer] {} addr={:#x} t_since_first_trip={:?} slots={:?}",
                    key, code_addr, elapsed, live_in_slots
                );
            }
            let buf_ptr = buf.as_ptr() as usize;
            let result = unsafe {
                crate::builtins::apply::call_with_args(
                    code_addr as *const u8,
                    &[buf_ptr],
                    1,
                    false,
                    0,
                )
            };
            drop(buf); // keep the buffer alive across the call
            crate::osr::osr_transfer_end(key);
            result
        }
        Some(OsrState::Failed) => {
            // Permanently ineligible — back off so the check effectively
            // never fires again for this loop.
            rearm(i64::MAX);
            OSR_NO_OSR
        }
        Some(OsrState::Requested) => {
            // Build in flight; check again soon.
            rearm(recheck);
            OSR_NO_OSR
        }
        None => {
            // First trip for this loop: request the build (exactly once),
            // then keep running tier-1 until it's ready.
            if crate::osr::osr_try_reserve(key) {
                if let Some((full, idx)) = key.rsplit_once("#L") {
                    if let Ok(idx) = idx.parse::<usize>() {
                        get_runtime()
                            .get()
                            .build_osr_variant(&full.to_string(), idx);
                    }
                }
            }
            rearm(recheck);
            OSR_NO_OSR
        }
    }
}

/// runtime/specialize-all() - Walk the arithmetic-feedback cache and
/// recompile every fully-monomorphic function with `*_with_bail`
/// specialization, atomically swapping each function's jump-table slot
/// to the new version. Returns the number of functions specialized.
///
/// Intended call shape from a benchmark:
///
///     fn main() {
///         warm_up()                         // populates feedback
///         let n = runtime/specialize-all()
///         println("specialized:", n)
///         measure()                         // hits specialized code
///     }
pub extern "C" fn runtime_specialize_all(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let count = runtime.specialize_all();
    BuiltInTypes::construct_int(count as isize) as usize
}

/// runtime/stop-the-world() - Drive a stop-the-world rendezvous from the
/// calling mutator (parking every other thread at its entry safepoint), observe
/// that the world is stopped, then resume. Returns the number of OTHER threads
/// that were parked. Scaffolding for race-free tier-2 install application.
pub extern "C" fn runtime_stop_the_world_observe(
    stack_pointer: usize,
    frame_pointer: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let parked = runtime.stop_the_world_observe(frame_pointer);
    BuiltInTypes::construct_int(parked as isize) as usize
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
        // Check enums BEFORE structs: an enum is registered both as an `Enum`
        // and as a companion `Struct` of the same name (used for variant
        // dispatch). The companion struct carries `source_text: None` /
        // `disk_location: None` — the real source and on-disk origin live on
        // the `Enum` record. Resolving the companion struct first would report
        // an enum as having no source/location (and make it un-persistable via
        // a string-name lookup). This mirrors the precedence in
        // `resolve_definition` for the value-based path.
        if let Some(e) = runtime.enums.get(candidate) {
            return Some(DefinitionInfo {
                full_name: e.name.clone(),
                source_text: e.source_text.clone(),
                disk_location: e.disk_location.clone(),
            });
        }
        if let Some((_, s)) = runtime.get_struct(candidate) {
            return Some(DefinitionInfo {
                full_name: s.name.clone(),
                source_text: s.source_text.clone(),
                disk_location: s.disk_location.clone(),
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

    let new_text_str = runtime.get_string(stack_pointer, new_text);

    match perform_splice(runtime, stack_pointer, &info, &new_text_str) {
        Ok(()) => BuiltInTypes::construct_boolean(true) as usize,
        Err(msg) => {
            throw_write_source_error(stack_pointer, &format!("reflect/write-source: {}", msg));
        }
    }
}

/// Resolve the authoritative byte range of a definition within the current
/// `file_contents`.
///
/// The recorded `disk_location` byte offsets are only a hint. Definitions that
/// were re-compiled from individually-extracted source fragments during a
/// multi-module load (anything pulled in via `use` / an `-I` include path) get
/// *fragment-local* (0-based) offsets, so `file[byte_start..byte_end]` can point
/// at the wrong region — or out of bounds — even when the file is byte-for-byte
/// unchanged. The reliable anchor is `original` (the exact `source_text`
/// captured when the def was loaded). Strategy:
///   1. Fast path — if the recorded range is in-bounds, on char boundaries, and
///      already equals `original`, trust it (preserves behaviour for files
///      loaded directly, e.g. the REPL server, and keeps the common case O(1)).
///   2. Otherwise locate `original` in the file. Exactly one occurrence is the
///      true range (self-healing). Zero occurrences = the def's text is gone
///      (genuine drift). More than one = ambiguous, so refuse rather than guess.
fn resolve_def_range(
    file_contents: &str,
    loc: &DiskLocation,
    original: &str,
) -> Result<(usize, usize), String> {
    if loc.byte_start <= loc.byte_end
        && loc.byte_end <= file_contents.len()
        && file_contents.is_char_boundary(loc.byte_start)
        && file_contents.is_char_boundary(loc.byte_end)
        && file_contents[loc.byte_start..loc.byte_end] == *original
    {
        return Ok((loc.byte_start, loc.byte_end));
    }
    if original.is_empty() {
        return Err("no stored source text to locate the definition".to_string());
    }
    let mut found: Option<usize> = None;
    let mut from = 0;
    while let Some(rel) = file_contents[from..].find(original) {
        let idx = from + rel;
        if found.is_some() {
            return Err(
                "stored source text occurs more than once in the file; cannot safely locate the definition to edit"
                    .to_string(),
            );
        }
        found = Some(idx);
        from = idx + 1;
    }
    found.map(|idx| (idx, idx + original.len())).ok_or_else(|| {
        "the definition's stored source text is no longer present in the file (it changed since it was loaded)"
            .to_string()
    })
}

/// Execute the full "edit an existing disk-resident definition" sequence:
/// verify the file hasn't drifted, splice the new text into the file,
/// shift subsequent byte ranges, patch the record's location, recompile
/// the fragment with file context, flush queued heap bindings, and run
/// the resulting top-level. Shared by `reflect/write-source` and
/// `reflect/persist` so both paths use identical semantics.
fn perform_splice(
    runtime: &mut Runtime,
    stack_pointer: usize,
    info: &DefinitionInfo,
    new_text: &str,
) -> Result<(), String> {
    let loc = info.disk_location.clone().ok_or_else(|| {
        "definition has no on-disk origin (eval/REPL defs cannot be persisted)".to_string()
    })?;
    let original_source = info
        .source_text
        .clone()
        .ok_or_else(|| "definition has no stored source text to verify against".to_string())?;

    let file_contents = std::fs::read_to_string(&loc.file)
        .map_err(|e| format!("failed to read {}: {}", loc.file, e))?;

    // The recorded byte range is only a hint (fragment-local offsets are
    // possible for `use`/`-I`-loaded defs); `original_source` is the real
    // anchor. resolve_def_range trusts the offsets when valid, else relocates
    // by the stored text. The resolved range is authoritative for the splice.
    let (byte_start, byte_end) = resolve_def_range(&file_contents, &loc, &original_source)
        .map_err(|e| format!("`{}` in {}: {}", info.full_name, loc.file, e))?;
    // Derive the real starting line from the resolved offset rather than the
    // (possibly fragment-local) recorded `line_start`, so the rewritten def's
    // location and the recompiled fragment's line base are both correct.
    let line_start = file_contents[..byte_start].matches('\n').count() + 1;

    let mut new_contents =
        String::with_capacity(file_contents.len() - (byte_end - byte_start) + new_text.len());
    new_contents.push_str(&file_contents[..byte_start]);
    new_contents.push_str(new_text);
    new_contents.push_str(&file_contents[byte_end..]);

    std::fs::write(&loc.file, &new_contents)
        .map_err(|e| format!("failed to write {}: {}", loc.file, e))?;

    let old_len = byte_end - byte_start;
    let new_len = new_text.len();
    let delta = new_len as isize - old_len as isize;
    shift_byte_ranges_after(runtime, &loc.file, byte_end, delta);

    let new_byte_end = byte_start + new_len;
    let new_line_end = line_start + new_text.matches('\n').count();
    let patched_loc = DiskLocation {
        file: loc.file.clone(),
        byte_start,
        byte_end: new_byte_end,
        line_start,
        line_end: new_line_end,
    };
    patch_disk_location(runtime, &info.full_name, patched_loc);
    // Keep the stored source text in sync with the bytes just written,
    // even if the recompile below fails — a corrective re-persist must
    // pass the drift check against what's really on disk.
    patch_source_text(runtime, &info.full_name, new_text);

    let namespace = namespace_of(&info.full_name).unwrap_or("").to_string();
    recompile_fragment_and_run(
        runtime,
        stack_pointer,
        new_text,
        &namespace,
        &loc.file,
        byte_start,
        line_start.saturating_sub(1),
    )
    .map_err(|e| format!("wrote {} but re-compile failed: {}", loc.file, e))
}

/// Append `new_text` to the end of `file` (with a blank line separator
/// so the appended def doesn't run into whatever trailed the file),
/// then compile the fragment with file context so its `disk_location`
/// matches its new position on disk. Used by `reflect/persist` for
/// top-level defs that don't yet exist on disk.
fn perform_append(
    runtime: &mut Runtime,
    stack_pointer: usize,
    namespace: &str,
    file: &str,
    new_text: &str,
) -> Result<(), String> {
    let file_contents =
        std::fs::read_to_string(file).map_err(|e| format!("failed to read {}: {}", file, e))?;

    // Ensure the appended fragment starts on its own line, with a blank
    // line separator from any prior content. Compute `separator` so that
    // the final file ends with at least "\n\n" before the fragment, but
    // we don't stack extra newlines if the file already ends with them.
    let trailing_newlines = file_contents
        .bytes()
        .rev()
        .take_while(|b| *b == b'\n')
        .count();
    let needed = 2usize.saturating_sub(trailing_newlines);
    let separator: String = "\n".repeat(if file_contents.is_empty() { 0 } else { needed });

    // Count lines already in the file so the fragment's line numbers
    // align with its real position. A file ending in "\n" has one more
    // logical "next line" than the raw newline count implies.
    let existing_line_count = file_contents.matches('\n').count();
    let separator_line_count = separator.matches('\n').count();
    let fragment_byte_offset = file_contents.len() + separator.len();
    let fragment_line_offset = existing_line_count + separator_line_count;

    let mut new_contents =
        String::with_capacity(file_contents.len() + separator.len() + new_text.len());
    new_contents.push_str(&file_contents);
    new_contents.push_str(&separator);
    new_contents.push_str(new_text);

    std::fs::write(file, &new_contents).map_err(|e| format!("failed to write {}: {}", file, e))?;

    recompile_fragment_and_run(
        runtime,
        stack_pointer,
        new_text,
        namespace,
        file,
        fragment_byte_offset,
        fragment_line_offset,
    )
    .map_err(|e| format!("wrote {} but re-compile failed: {}", file, e))
}

/// Recompile a single top-level fragment with file context, flush any
/// heap-binding updates queued by the recompile, and run the resulting
/// top-level. Extracted from the splice path so the append path can
/// reuse the identical tail sequence.
///
/// The top-level is entered through `save_volatile_registers0` because
/// the JIT prologue for `top_level` expects x28 to hold the per-thread
/// MutatorState pointer, and Rust is free to clobber x28 within this
/// function — the trampoline reloads it before branching into JIT code.
fn recompile_fragment_and_run(
    runtime: &mut Runtime,
    _stack_pointer: usize,
    fragment: &str,
    namespace: &str,
    file: &str,
    byte_offset: usize,
    line_offset: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let top_level_ptr = runtime.compile_string_with_file_context(
        fragment,
        namespace,
        file,
        byte_offset,
        line_offset,
    )?;

    if top_level_ptr != 0 {
        let save_vr0_entry = runtime
            .get_function_by_name("beagle.builtin/save_volatile_registers0")
            .expect("save_volatile_registers0 trampoline not registered");
        let save_vr0_ptr = runtime
            .get_pointer(save_vr0_entry)
            .expect("save_volatile_registers0 pointer not available");
        let save_vr0: extern "C" fn(usize) -> usize =
            unsafe { std::mem::transmute::<_, _>(save_vr0_ptr) };
        // The nested top-level run overwrites the per-thread GC context
        // with its own (soon-dead) frames. Restore the enclosing
        // builtin's context afterward so a later throw from this same
        // builtin invocation (e.g. a failed splice for the NEXT def in
        // a multi-def `reflect/persist`) walks live stack, not the
        // recompile's reused frames. Without this, a failing persist
        // aborted the whole process with "shift without an enclosing
        // reset" instead of raising a catchable error.
        let saved_ctx = crate::builtins::save_current_gc_context();
        save_vr0(top_level_ptr);
        crate::builtins::restore_gc_context(saved_ctx);
    }
    Ok(())
}

/// One top-level definition discovered inside the `text` argument of
/// `reflect/persist`: its unqualified name and the exact byte slice
/// that defines it (what will later be recompiled as a fragment).
struct PersistDef {
    name: String,
    fragment: String,
}

/// Pre-parse the `text` argument of `reflect/persist` to discover the
/// fn/struct/enum defs it contains and slice out each one's source text.
/// Skips `namespace` directives and any non-definition top-level
/// elements (the recompile step only handles fn/struct/enum).
fn extract_persist_defs(text: &str) -> Result<Vec<PersistDef>, String> {
    use crate::ast::Ast;
    use crate::parser::Parser;

    let mut parser = Parser::new("<persist>".to_string(), text.to_string())
        .map_err(|e| format!("parse error: {}", e))?;
    let ast = parser.parse().map_err(|e| format!("parse error: {}", e))?;
    let ranges = parser.get_definition_byte_ranges();

    let elements = match ast {
        Ast::Program { elements, .. } => elements,
        other => vec![other],
    };

    let mut defs = Vec::new();
    for el in elements {
        let (name, token_range) = match &el {
            Ast::Function {
                name: Some(n),
                token_range,
                ..
            } => (n.clone(), *token_range),
            Ast::Struct {
                name, token_range, ..
            } => (name.clone(), *token_range),
            Ast::Enum {
                name, token_range, ..
            } => (name.clone(), *token_range),
            _ => continue,
        };
        let (byte_start, byte_end) = ranges
            .get(&(token_range.start, token_range.end))
            .copied()
            .ok_or_else(|| format!("no byte range recorded for `{}`", name))?;
        if byte_end > text.len() || byte_start > byte_end {
            return Err(format!("byte range for `{}` is out of bounds", name));
        }
        defs.push(PersistDef {
            name,
            fragment: text[byte_start..byte_end].to_string(),
        });
    }
    Ok(defs)
}

/// Find any on-disk file that hosts a definition in `namespace`. Used
/// by `reflect/persist` to decide where to append brand-new defs. The
/// "one file per namespace" assumption holds in practice even though
/// it isn't enforced — if the namespace spans multiple files, the
/// first hit wins.
fn find_any_file_for_namespace(runtime: &Runtime, namespace: &str) -> Option<String> {
    let prefix = format!("{}/", namespace);
    if let Some(f) = runtime
        .functions
        .iter()
        .find(|f| f.name.starts_with(&prefix) && f.disk_location.is_some())
    {
        return f.disk_location.as_ref().map(|l| l.file.clone());
    }
    if let Some(s) = runtime
        .structs
        .iter()
        .find(|s| s.name.starts_with(&prefix) && s.disk_location.is_some())
    {
        return s.disk_location.as_ref().map(|l| l.file.clone());
    }
    if let Some(e) = runtime
        .enums
        .iter()
        .find(|e| e.name.starts_with(&prefix) && e.disk_location.is_some())
    {
        return e.disk_location.as_ref().map(|l| l.file.clone());
    }
    if let Some(b) = runtime
        .bindings
        .iter()
        .find(|b| b.name.starts_with(&prefix) && b.disk_location.is_some())
    {
        return b.disk_location.as_ref().map(|l| l.file.clone());
    }
    None
}

/// reflect/persist(namespace, text) - Persist one or more definitions
/// to disk, routing each to a splice or append path based on whether
/// it already exists in `namespace`.
///
/// Parses `text` to discover top-level fn/struct/enum defs. For each:
///   - if `namespace/<name>` has an on-disk location → splice path
///     (drift-check, write, shift trailing byte ranges, recompile).
///   - else → append path (append to the namespace's file, recompile
///     at the new byte offset).
///
/// All splice drift checks run up-front — if any fails, nothing is
/// written. After that, defs are processed in their textual order so
/// later defs see the runtime state left by earlier ones.
///
/// Returns a vector of maps `[{:name, :action}]` where `:action` is
/// `"updated"` or `"appended"`. Throws on parse failure, drift
/// failure, I/O failure, or compile failure of any fragment.
pub extern "C" fn reflect_persist(
    stack_pointer: usize,
    frame_pointer: usize,
    namespace_val: usize,
    text_val: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let namespace = runtime.get_string(stack_pointer, namespace_val);
    let text = runtime.get_string(stack_pointer, text_val);

    let defs = match extract_persist_defs(&text) {
        Ok(d) => d,
        Err(e) => throw_persist_error(stack_pointer, &format!("reflect/persist: {}", e)),
    };
    if defs.is_empty() {
        throw_persist_error(
            stack_pointer,
            "reflect/persist: no top-level fn/struct/enum definitions in text",
        );
    }

    enum Classified {
        Splice {
            full_name: String,
            info: DefinitionInfo,
            fragment: String,
        },
        Append {
            full_name: String,
            fragment: String,
        },
    }

    let mut classified: Vec<Classified> = Vec::with_capacity(defs.len());
    let mut splice_file: Option<String> = None;
    for def in defs {
        let full_name = format!("{}/{}", namespace, def.name);
        let info = resolve_by_name(runtime, &full_name);
        match info {
            Some(info) if info.disk_location.is_some() => {
                if splice_file.is_none() {
                    splice_file = info.disk_location.as_ref().map(|l| l.file.clone());
                }
                classified.push(Classified::Splice {
                    full_name,
                    info,
                    fragment: def.fragment,
                });
            }
            _ => {
                classified.push(Classified::Append {
                    full_name,
                    fragment: def.fragment,
                });
            }
        }
    }

    let has_appends = classified
        .iter()
        .any(|c| matches!(c, Classified::Append { .. }));
    let append_file = if has_appends {
        let f = splice_file
            .clone()
            .or_else(|| find_any_file_for_namespace(runtime, &namespace));
        match f {
            Some(f) => Some(f),
            None => throw_persist_error(
                stack_pointer,
                &format!(
                    "reflect/persist: namespace `{}` has no on-disk origin; cannot append new definitions",
                    namespace
                ),
            ),
        }
    } else {
        None
    };

    // Up-front drift check for every splice entry. If any fails we
    // abort before touching disk. We group by file and read once per
    // file so a multi-def persist against a single file reads that
    // file exactly once for the drift phase.
    use std::collections::HashMap;
    let mut file_cache: HashMap<String, String> = HashMap::new();
    for c in &classified {
        if let Classified::Splice {
            info, full_name, ..
        } = c
        {
            let loc = info.disk_location.as_ref().expect("classified as splice");
            let original = info.source_text.as_deref().unwrap_or("");
            let contents = match file_cache.get(&loc.file) {
                Some(s) => s,
                None => {
                    let s = match std::fs::read_to_string(&loc.file) {
                        Ok(s) => s,
                        Err(e) => throw_persist_error(
                            stack_pointer,
                            &format!("reflect/persist: failed to read {}: {}", loc.file, e),
                        ),
                    };
                    file_cache.entry(loc.file.clone()).or_insert(s)
                }
            };
            // Pre-flight: every splice target must be locatable in its file by
            // its stored text (recorded offsets are only a hint — see
            // resolve_def_range). Abort before touching disk if any can't be.
            if let Err(e) = resolve_def_range(contents, loc, original) {
                throw_persist_error(
                    stack_pointer,
                    &format!("reflect/persist: `{}` in {}: {}", full_name, loc.file, e),
                );
            }
        }
    }
    drop(file_cache);

    // Process defs in textual order. Per-def splice/append already
    // handles its own file I/O + recompile; we just dispatch.
    let mut results: Vec<(String, &'static str)> = Vec::with_capacity(classified.len());
    for c in classified {
        match c {
            Classified::Splice {
                full_name,
                info,
                fragment,
            } => {
                // Re-resolve at process time: an earlier splice in this
                // same persist may have shifted this definition's byte
                // range on disk. The `info` snapshot from classification
                // predates those shifts, so splicing with it would fail
                // the drift check ("file has changed") on every
                // multi-def persist whose earlier def changed length.
                let fresh_info = match resolve_by_name(runtime, &full_name) {
                    Some(fresh) if fresh.disk_location.is_some() => fresh,
                    _ => info,
                };
                if let Err(e) = perform_splice(runtime, stack_pointer, &fresh_info, &fragment) {
                    throw_persist_error(
                        stack_pointer,
                        &format!("reflect/persist: updating `{}`: {}", full_name, e),
                    );
                }
                results.push((full_name, "updated"));
            }
            Classified::Append {
                full_name,
                fragment,
            } => {
                let file = append_file
                    .as_ref()
                    .expect("append_file resolved when has_appends");
                if let Err(e) = perform_append(runtime, stack_pointer, &namespace, file, &fragment)
                {
                    throw_persist_error(
                        stack_pointer,
                        &format!("reflect/persist: appending `{}`: {}", full_name, e),
                    );
                }
                results.push((full_name, "appended"));
            }
        }
    }

    build_persist_result(runtime, stack_pointer, &results)
        .unwrap_or(BuiltInTypes::null_value() as usize)
}

/// Build the `[{:name, :action}]` return value for `reflect/persist`
/// as a PersistentVec of PersistentMaps, mirroring the GC-safe handle
/// pattern used by `reflect_namespace_info`.
fn build_persist_result(
    runtime: &mut Runtime,
    stack_pointer: usize,
    results: &[(String, &'static str)],
) -> Option<usize> {
    use crate::collections::{GcHandle, HandleScope, PersistentMap, PersistentVec};

    let mut scope = HandleScope::new(runtime, stack_pointer);
    let tg_ptr = crate::runtime::cached_thread_global_ptr();

    let vec = PersistentVec::empty(scope.runtime(), stack_pointer).ok()?;
    let vec_h = scope.alloc(vec.as_tagged());

    for (name, action) in results {
        let map = PersistentMap::empty(scope.runtime(), stack_pointer).ok()?;
        let map_h = scope.alloc(map.as_tagged());

        // :name
        let name_key = scope
            .runtime()
            .intern_keyword(stack_pointer, "name".to_string())
            .ok()?;
        let name_key_h = scope.alloc(name_key);
        let name_val = scope
            .runtime()
            .allocate_string(stack_pointer, name.clone())
            .ok()?
            .into();
        let name_val_h = scope.alloc(name_val);
        let map_after_name = PersistentMap::assoc(
            scope.runtime(),
            stack_pointer,
            map_h.get(),
            name_key_h.get(),
            name_val_h.get(),
        )
        .ok()?;
        let tg = unsafe { &mut *tg_ptr };
        tg.handle_stack[map_h.slot()] = map_after_name.as_tagged();

        // :action
        let action_key = scope
            .runtime()
            .intern_keyword(stack_pointer, "action".to_string())
            .ok()?;
        let action_key_h = scope.alloc(action_key);
        let action_val = scope
            .runtime()
            .allocate_string(stack_pointer, action.to_string())
            .ok()?
            .into();
        let action_val_h = scope.alloc(action_val);
        let map_after_action = PersistentMap::assoc(
            scope.runtime(),
            stack_pointer,
            map_h.get(),
            action_key_h.get(),
            action_val_h.get(),
        )
        .ok()?;
        let tg = unsafe { &mut *tg_ptr };
        tg.handle_stack[map_h.slot()] = map_after_action.as_tagged();

        // push map into vec
        let current_vec = GcHandle::from_tagged(vec_h.get());
        let new_vec =
            PersistentVec::push(scope.runtime(), stack_pointer, current_vec, map_h.get()).ok()?;
        let tg = unsafe { &mut *tg_ptr };
        tg.handle_stack[vec_h.slot()] = new_vec.as_tagged();
    }

    Some(vec_h.get())
}

/// Raise a runtime error from `reflect/persist`. Mirrors
/// `throw_write_source_error` but tags the message with the persist
/// path so users can distinguish the two sources in stack traces.
fn throw_persist_error(stack_pointer: usize, message: &str) -> ! {
    unsafe {
        crate::builtins::throw_runtime_error(stack_pointer, "RuntimeError", message.to_string())
    }
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

/// Overwrite the stored `source_text` on whichever runtime record (fn,
/// struct, enum, or binding) is registered under `full_name`. Called by
/// `perform_splice` right after the file write, BEFORE recompiling: if
/// the recompile fails, the record must still describe what is actually
/// on disk, or every follow-up persist of the same definition drift-fails
/// with "file has changed" and there is no way to fix a broken splice
/// through the persist API at all.
fn patch_source_text(runtime: &mut Runtime, full_name: &str, text: &str) {
    if let Some(f) = runtime.functions.iter_mut().find(|f| f.name == full_name) {
        f.source_text = Some(text.to_string());
        return;
    }
    if runtime.patch_struct_source_text(full_name, text.to_string()) {
        return;
    }
    if runtime.patch_enum_source_text(full_name, text.to_string()) {
        return;
    }
    runtime.patch_binding_source_text(full_name, text.to_string());
}
