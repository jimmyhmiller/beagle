use super::*;
use crate::save_gc_context;

pub unsafe extern "C" fn load_library(
    stack_pointer: usize,
    frame_pointer: usize,
    name: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let string = runtime.get_string_literal(name);
    let lib = match unsafe { libloading::Library::new(&string) } {
        Ok(lib) => lib,
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "FFIError",
                format!("Failed to load library '{}': {}", string, e),
            );
        },
    };
    let id = runtime.add_library(lib);

    unsafe {
        call_fn_1(
            runtime,
            "beagle.ffi/__make_lib_struct",
            BuiltInTypes::Int.tag(id as isize) as usize,
        )
    }
}

pub fn map_beagle_type_to_ffi_type(runtime: &Runtime, value: usize) -> Result<FFIType, String> {
    let heap_object = HeapObject::from_tagged(value);
    let struct_id = heap_object.get_struct_id();
    let struct_info = runtime.get_struct_by_id(struct_id);

    if struct_info.is_none() {
        return Err(format!("Could not find struct with id {}", struct_id));
    }

    let struct_info = struct_info.unwrap();
    let name = match struct_info.name.as_str().split_once("/") {
        Some((_, name)) => name,
        None => return Err(format!("Invalid struct name format: {}", struct_info.name)),
    };
    match name {
        "Type.U8" => Ok(FFIType::U8),
        "Type.U16" => Ok(FFIType::U16),
        "Type.U32" => Ok(FFIType::U32),
        "Type.U64" => Ok(FFIType::U64),
        "Type.I8" => Ok(FFIType::I8),
        "Type.I16" => Ok(FFIType::I16),
        "Type.I32" => Ok(FFIType::I32),
        "Type.I64" => Ok(FFIType::I64),
        "Type.F32" => Ok(FFIType::F32),
        "Type.F64" => Ok(FFIType::F64),
        "Type.Pointer" => Ok(FFIType::Pointer),
        "Type.MutablePointer" => Ok(FFIType::MutablePointer),
        "Type.String" => Ok(FFIType::String),
        "Type.Void" => Ok(FFIType::Void),
        "Type.Structure" => {
            let types = heap_object.get_field(0);
            let types = persistent_vector_to_vec(types);
            let fields: Result<Vec<FFIType>, String> = types
                .iter()
                .map(|t| map_beagle_type_to_ffi_type(runtime, *t))
                .collect();
            Ok(FFIType::Structure(fields?))
        }
        _ => Err(format!("Unknown type: {}", name)),
    }
}

fn persistent_vector_to_vec(vector: usize) -> Vec<usize> {
    use crate::collections::{GcHandle, PersistentVec};
    let vec_handle = GcHandle::from_tagged(vector);
    let count = PersistentVec::count(vec_handle);
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        result.push(PersistentVec::get(vec_handle, i));
    }
    result
}

/// Get a function from a dynamically loaded library and register it for FFI calls.
/// No longer uses libffi - just stores the function pointer and type information.
pub extern "C" fn get_function(
    stack_pointer: usize,
    frame_pointer: usize,
    library_struct: usize,
    function_name: usize,
    types: usize,
    return_type: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let library = runtime.get_library(library_struct);
    let function_name = runtime.get_string_literal(function_name);

    // Check if we already have this function registered
    if let Some(ffi_info_id) = runtime.find_ffi_info_by_name(&function_name) {
        let ffi_info_id = BuiltInTypes::Int.tag(ffi_info_id as isize) as usize;
        return unsafe { call_fn_1(runtime, "beagle.ffi/__create_ffi_function", ffi_info_id) };
    }

    // Get the function pointer from the library
    let func_ptr = match unsafe { library.get::<fn()>(function_name.as_bytes()) } {
        Ok(ptr) => ptr,
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "FFIError",
                format!("Failed to get symbol '{}': {}", function_name, e),
            );
        },
    };
    let code_ptr = match unsafe { func_ptr.try_as_raw_ptr() } {
        Some(ptr) => ptr,
        None => unsafe {
            throw_runtime_error(
                stack_pointer,
                "FFIError",
                format!("Invalid function pointer for symbol '{}'", function_name),
            );
        },
    };

    // Parse argument types
    let types: Vec<usize> = persistent_vector_to_vec(types);

    let beagle_ffi_types: Result<Vec<FFIType>, String> = types
        .iter()
        .map(|t| map_beagle_type_to_ffi_type(runtime, *t))
        .collect();
    let beagle_ffi_types = match beagle_ffi_types {
        Ok(types) => types,
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "FFIError", e);
        },
    };
    let number_of_arguments = beagle_ffi_types.len();

    // Parse return type
    let ffi_return_type = match map_beagle_type_to_ffi_type(runtime, return_type) {
        Ok(t) => t,
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "FFIError", e);
        },
    };

    // Register the function info (no Cif needed anymore)
    let ffi_info_id = runtime.add_ffi_function_info(FFIInfo {
        name: function_name.to_string(),
        function: RawPtr::new(code_ptr as *const u8),
        number_of_arguments,
        argument_types: beagle_ffi_types,
        return_type: ffi_return_type,
    });
    runtime.add_ffi_info_by_name(function_name, ffi_info_id);
    let ffi_info_id = BuiltInTypes::Int.tag(ffi_info_id as isize) as usize;

    unsafe { call_fn_1(runtime, "beagle.ffi/__create_ffi_function", ffi_info_id) }
}

/// Get a raw function pointer (symbol) from a loaded library without binding arg types.
/// Returns a Pointer struct wrapping the raw function address.
/// Used for variadic FFI calls where arg types are specified per-call.
pub extern "C" fn get_symbol(
    stack_pointer: usize,
    frame_pointer: usize,
    library_struct: usize,
    function_name: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let library = runtime.get_library(library_struct);
    let function_name = runtime.get_string_literal(function_name);

    // Get the function pointer from the library
    let func_ptr = match unsafe { library.get::<fn()>(function_name.as_bytes()) } {
        Ok(ptr) => ptr,
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "FFIError",
                format!("Failed to get symbol '{}': {}", function_name, e),
            );
        },
    };
    let code_ptr = match unsafe { func_ptr.try_as_raw_ptr() } {
        Some(ptr) => ptr,
        None => unsafe {
            throw_runtime_error(
                stack_pointer,
                "FFIError",
                format!("Invalid function pointer for symbol '{}'", function_name),
            );
        },
    };

    // Return as a Pointer struct (split into lo/hi halves to preserve all 64 bits)
    let raw = code_ptr as u64;
    let lo_tagged = BuiltInTypes::Int.tag((raw & 0xFFFFFFFF) as isize) as usize;
    let hi_tagged = BuiltInTypes::Int.tag(((raw >> 32) & 0xFFFFFFFF) as isize) as usize;
    unsafe {
        call_fn_2(
            runtime,
            "beagle.ffi/__make_pointer_struct",
            lo_tagged,
            hi_tagged,
        )
    }
}

/// Call a raw function pointer with per-call type information (variadic FFI).
/// Takes: func_pointer (Pointer struct), args_array (Beagle array), types_array (Beagle array of Type enum), return_type (Type enum).
pub unsafe extern "C" fn call_ffi_variadic(
    stack_pointer: usize,
    _frame_pointer: usize,
    func_pointer: usize,
    args_array: usize,
    types_array: usize,
    return_type_value: usize,
) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();

        // Extract raw function pointer from the Pointer struct (lo/hi halves)
        let ptr_object = HeapObject::from_tagged(func_pointer);
        let lo = BuiltInTypes::untag(ptr_object.get_field(0)) as u64;
        let hi = BuiltInTypes::untag(ptr_object.get_field(1)) as u64;
        let raw_ptr = (lo | (hi << 32)) as *const u8;

        // Parse argument types
        let types_vec: Vec<usize> = persistent_vector_to_vec(types_array);
        let arg_types: Vec<FFIType> = types_vec
            .iter()
            .map(|t| match map_beagle_type_to_ffi_type(runtime, *t) {
                Ok(ffi_type) => ffi_type,
                Err(e) => {
                    throw_runtime_error(stack_pointer, "FFIError", e);
                }
            })
            .collect();

        // Parse return type
        let return_type = match map_beagle_type_to_ffi_type(runtime, return_type_value) {
            Ok(t) => t,
            Err(e) => {
                throw_runtime_error(stack_pointer, "FFIError", e);
            }
        };

        // Read args from the Beagle persistent vector
        let args_vec: Vec<usize> = persistent_vector_to_vec(args_array);

        if args_vec.len() != arg_types.len() {
            throw_runtime_error(
                stack_pointer,
                "FFIError",
                format!(
                    "Variadic FFI call: {} arguments provided but {} types specified",
                    args_vec.len(),
                    arg_types.len()
                ),
            );
        }

        // Marshal arguments to native u64 values
        let mut native_args = Vec::with_capacity(arg_types.len());
        for (argument, ffi_type) in args_vec.iter().zip(arg_types.iter()) {
            native_args.push(marshal_ffi_argument(
                runtime,
                stack_pointer,
                *argument,
                ffi_type,
            ));
        }

        // Call the native function using the dynamic trampoline
        let (low, high) = dynamic_c_call(raw_ptr, &native_args, &arg_types, &return_type);

        // Unmarshal the return value
        let return_value = unmarshal_ffi_return(runtime, stack_pointer, low, high, &return_type);

        runtime.memory.clear_native_arguments();
        return_value
    }
}

/// Rust helper that callback trampolines call into.
/// Bridges from C calling convention back into Beagle.
/// Called from generated trampoline code.
///
/// # Safety
/// Must be called from a callback trampoline with valid arguments.
unsafe extern "C" fn invoke_beagle_callback(
    callback_index: usize,
    c_args_ptr: *const u64,
    num_args: usize,
) -> u64 {
    unsafe {
        let runtime = get_runtime().get_mut();
        let (beagle_fn, arg_types, return_type) = {
            let cb = runtime.get_callback(callback_index);
            (cb.beagle_fn, cb.arg_types.clone(), cb.return_type.clone())
        };

        // Get a valid stack pointer for GC context.
        // We're being called from C code, so get the current stack pointer.
        let stack_pointer: usize = {
            let fn_entry = runtime
                .get_function_by_name("beagle.builtin/read-sp")
                .expect("read-sp trampoline not found");
            let read_sp: extern "C" fn() -> usize = std::mem::transmute::<_, _>(fn_entry.pointer);
            read_sp()
        };

        // Unmarshal C args to Beagle values.
        // Heap-allocated args (Pointer structs, Floats) must be registered as temporary
        // roots so GC can update them if they move. Without this, a GC triggered by
        // unmarshalling arg N could move the heap object created for arg N-1.
        let mut beagle_args = Vec::with_capacity(num_args);
        let mut root_slots = Vec::new();
        for (i, arg_type) in arg_types.iter().enumerate().take(num_args) {
            let c_val = *c_args_ptr.add(i);
            let beagle_val = match arg_type {
                FFIType::U8 | FFIType::U16 | FFIType::U32 | FFIType::U64 => {
                    BuiltInTypes::Int.tag(c_val as isize) as usize
                }
                FFIType::I8 => {
                    let signed = c_val as i8 as isize;
                    BuiltInTypes::Int.tag(signed) as usize
                }
                FFIType::I16 => {
                    let signed = c_val as i16 as isize;
                    BuiltInTypes::Int.tag(signed) as usize
                }
                FFIType::I32 => {
                    let signed = c_val as i32 as isize;
                    BuiltInTypes::Int.tag(signed) as usize
                }
                FFIType::I64 => {
                    let signed = c_val as i64 as isize;
                    BuiltInTypes::Int.tag(signed) as usize
                }
                FFIType::Pointer | FFIType::MutablePointer => {
                    let raw = c_val;
                    let lo_tagged = BuiltInTypes::Int.tag((raw & 0xFFFFFFFF) as isize) as usize;
                    let hi_tagged =
                        BuiltInTypes::Int.tag(((raw >> 32) & 0xFFFFFFFF) as isize) as usize;
                    let val = call_fn_2(
                        runtime,
                        "beagle.ffi/__make_pointer_struct",
                        lo_tagged,
                        hi_tagged,
                    );
                    let slot = runtime.register_temporary_root(val);
                    root_slots.push((beagle_args.len(), slot));
                    val
                }
                FFIType::F32 => {
                    let f32_val = f32::from_bits(c_val as u32);
                    let f64_val = f32_val as f64;
                    let new_float_ptr =
                        match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
                            Ok(ptr) => ptr,
                            Err(_) => {
                                throw_runtime_error(
                                    stack_pointer,
                                    "AllocationError",
                                    "Failed to allocate float for FFI callback - out of memory"
                                        .to_string(),
                                );
                            }
                        };
                    let untagged_result = BuiltInTypes::untag(new_float_ptr);
                    let result_ptr = untagged_result as *mut f64;
                    *result_ptr.add(1) = f64_val;
                    let slot = runtime.register_temporary_root(new_float_ptr);
                    root_slots.push((beagle_args.len(), slot));
                    new_float_ptr
                }
                FFIType::F64 => {
                    let f64_val = f64::from_bits(c_val);
                    let new_float_ptr =
                        match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
                            Ok(ptr) => ptr,
                            Err(_) => {
                                throw_runtime_error(
                                    stack_pointer,
                                    "AllocationError",
                                    "Failed to allocate float for FFI callback - out of memory"
                                        .to_string(),
                                );
                            }
                        };
                    let untagged_result = BuiltInTypes::untag(new_float_ptr);
                    let result_ptr = untagged_result as *mut f64;
                    *result_ptr.add(1) = f64_val;
                    let slot = runtime.register_temporary_root(new_float_ptr);
                    root_slots.push((beagle_args.len(), slot));
                    new_float_ptr
                }
                _ => {
                    // For types we can't unmarshal, pass as tagged int
                    BuiltInTypes::Int.tag(c_val as isize) as usize
                }
            };
            beagle_args.push(beagle_val);
        }

        // Read back possibly-updated values from root slots and unregister them
        for (arg_index, slot) in root_slots {
            let updated = runtime.unregister_temporary_root(slot);
            beagle_args[arg_index] = updated;
        }

        // Call the Beagle function using the appropriate save_volatile_registers variant
        let saved_ctx = save_current_gc_context();

        // If beagle_fn is a function object (HeapObject with Function struct_id),
        // extract the raw function pointer from field 0 and treat as regular function.
        let (kind, resolved_fn) = {
            let k = BuiltInTypes::get_kind(beagle_fn);
            if matches!(k, BuiltInTypes::HeapObject) {
                let heap_obj = HeapObject::from_tagged(beagle_fn);
                if heap_obj.get_type_id() == 0 {
                    let rt = get_runtime().get();
                    if heap_obj.get_struct_id() == rt.function_struct_id {
                        // Extract Int-tagged fn_ptr, re-tag as Function
                        let int_tagged_fn_ptr = heap_obj.get_field(0);
                        let raw_fn_ptr = BuiltInTypes::untag(int_tagged_fn_ptr);
                        let fn_tagged = BuiltInTypes::Function.tag(raw_fn_ptr as isize) as usize;
                        (BuiltInTypes::Function, fn_tagged)
                    } else {
                        (k, beagle_fn)
                    }
                } else {
                    (k, beagle_fn)
                }
            } else {
                (k, beagle_fn)
            }
        };
        let result = if matches!(kind, BuiltInTypes::Closure) {
            // Closure: extract function pointer, pass closure as first implicit arg
            let untagged = BuiltInTypes::untag(beagle_fn);
            let closure = HeapObject::from_untagged(untagged as *const u8);
            let fp_tagged = closure.get_field(0);
            let function_pointer = BuiltInTypes::untag(fp_tagged);

            // For closures, arg layout is: closure, arg1, arg2, ..., fn_ptr
            // save_volatile_registersN takes N data args + fn_ptr
            // closure counts as first data arg
            let n = beagle_args.len() + 1; // +1 for closure itself
            match n {
                1 => {
                    let svr = get_svr::<fn(usize, usize) -> usize>(runtime, 1);
                    svr(beagle_fn, function_pointer)
                }
                2 => {
                    let svr = get_svr::<fn(usize, usize, usize) -> usize>(runtime, 2);
                    svr(beagle_fn, beagle_args[0], function_pointer)
                }
                3 => {
                    let svr = get_svr::<fn(usize, usize, usize, usize) -> usize>(runtime, 3);
                    svr(beagle_fn, beagle_args[0], beagle_args[1], function_pointer)
                }
                4 => {
                    let svr = get_svr::<fn(usize, usize, usize, usize, usize) -> usize>(runtime, 4);
                    svr(
                        beagle_fn,
                        beagle_args[0],
                        beagle_args[1],
                        beagle_args[2],
                        function_pointer,
                    )
                }
                5 => {
                    let svr = get_svr::<fn(usize, usize, usize, usize, usize, usize) -> usize>(
                        runtime, 5,
                    );
                    svr(
                        beagle_fn,
                        beagle_args[0],
                        beagle_args[1],
                        beagle_args[2],
                        beagle_args[3],
                        function_pointer,
                    )
                }
                _ => {
                    throw_runtime_error(
                        stack_pointer,
                        "ArgumentError",
                        format!(
                            "Callback with {} args not supported (max 4 for closures)",
                            beagle_args.len()
                        ),
                    );
                }
            }
        } else {
            // Regular function pointer (or resolved from function object)
            let function_pointer = BuiltInTypes::untag(resolved_fn);
            let n = beagle_args.len();
            match n {
                0 => {
                    let svr = get_svr::<fn(usize) -> usize>(runtime, 0);
                    svr(function_pointer)
                }
                1 => {
                    let svr = get_svr::<fn(usize, usize) -> usize>(runtime, 1);
                    svr(beagle_args[0], function_pointer)
                }
                2 => {
                    let svr = get_svr::<fn(usize, usize, usize) -> usize>(runtime, 2);
                    svr(beagle_args[0], beagle_args[1], function_pointer)
                }
                3 => {
                    let svr = get_svr::<fn(usize, usize, usize, usize) -> usize>(runtime, 3);
                    svr(
                        beagle_args[0],
                        beagle_args[1],
                        beagle_args[2],
                        function_pointer,
                    )
                }
                4 => {
                    let svr = get_svr::<fn(usize, usize, usize, usize, usize) -> usize>(runtime, 4);
                    svr(
                        beagle_args[0],
                        beagle_args[1],
                        beagle_args[2],
                        beagle_args[3],
                        function_pointer,
                    )
                }
                5 => {
                    let svr = get_svr::<fn(usize, usize, usize, usize, usize, usize) -> usize>(
                        runtime, 5,
                    );
                    svr(
                        beagle_args[0],
                        beagle_args[1],
                        beagle_args[2],
                        beagle_args[3],
                        beagle_args[4],
                        function_pointer,
                    )
                }
                _ => {
                    throw_runtime_error(
                        stack_pointer,
                        "ArgumentError",
                        format!(
                            "Callback with {} args not supported (max 5 for functions)",
                            n
                        ),
                    );
                }
            }
        };

        restore_gc_context(saved_ctx);

        // Marshal Beagle return value back to C

        match &return_type {
            FFIType::Void => 0u64,
            FFIType::U8 | FFIType::U16 | FFIType::U32 | FFIType::U64 => {
                BuiltInTypes::untag(result) as u64
            }
            FFIType::I8 | FFIType::I16 | FFIType::I32 | FFIType::I64 => {
                let val = BuiltInTypes::untag(result) as i64;
                val as u64
            }
            FFIType::F32 => {
                let ptr = BuiltInTypes::untag(result) as *const f64;
                let f64_val = *ptr.add(1);
                let f32_val = f64_val as f32;
                f32_val.to_bits() as u64
            }
            FFIType::F64 => {
                let ptr = BuiltInTypes::untag(result) as *const f64;
                let f64_val = *ptr.add(1);
                f64_val.to_bits()
            }
            FFIType::Pointer | FFIType::MutablePointer => {
                // Reconstruct 64-bit raw pointer from lo/hi halves
                let heap_object = HeapObject::from_tagged(result);
                let lo = BuiltInTypes::untag(heap_object.get_field(0)) as u64;
                let hi = BuiltInTypes::untag(heap_object.get_field(1)) as u64;
                lo | (hi << 32)
            }
            _ => BuiltInTypes::untag(result) as u64,
        }
    } // unsafe
}

/// Helper to get a save_volatile_registers function pointer.
/// Panics if the required function is not found - this is an internal runtime invariant.
unsafe fn get_svr<T>(runtime: &Runtime, n: usize) -> T {
    let name = format!("beagle.builtin/save_volatile_registers{}", n);
    let func = runtime
        .get_function_by_name(&name)
        .unwrap_or_else(|| panic!("Internal error: missing required builtin {}", name));
    let ptr = runtime
        .get_pointer(func)
        .unwrap_or_else(|_| panic!("Internal error: could not get pointer for {}", name));
    unsafe { std::mem::transmute_copy(&ptr) }
}

/// Build a per-callback ARM64 trampoline that:
/// 1. Saves C args (x0-x7) to the stack
/// 2. Calls invoke_beagle_callback(callback_index, args_ptr, num_args)
/// 3. Returns the result in x0
///
/// Returns a pointer to executable memory containing the trampoline code.
#[cfg(target_arch = "aarch64")]
fn build_callback_trampoline(
    callback_index: usize,
    helper_fn_ptr: *const u8,
    num_c_args: usize,
) -> *const u8 {
    use crate::machine_code::arm_codegen::*;

    let mut code: Vec<u32> = Vec::with_capacity(24);

    // === Prologue ===
    // stp x29, x30, [sp, #-16]!
    code.push(
        ArmAsm::StpGen {
            opc: 0b10,
            imm7: -2,
            rt2: X30,
            rn: SP,
            rt: X29,
            class_selector: StpGenSelector::PreIndex,
        }
        .encode(),
    );
    // mov x29, sp
    code.push(
        ArmAsm::MovAddAddsubImm {
            sf: 1,
            rn: SP,
            rd: X29,
        }
        .encode(),
    );
    // stp x19, x20, [sp, #-16]!  (save callee-saved)
    code.push(
        ArmAsm::StpGen {
            opc: 0b10,
            imm7: -2,
            rt2: X20,
            rn: SP,
            rt: X19,
            class_selector: StpGenSelector::PreIndex,
        }
        .encode(),
    );

    // === Save C args (x0-x7) to stack ===
    // sub sp, sp, #64  (8 regs * 8 bytes)
    code.push(
        ArmAsm::SubAddsubImm {
            sf: 1,
            sh: 0,
            imm12: 64,
            rn: SP,
            rd: SP,
        }
        .encode(),
    );
    // stp x0, x1, [sp]
    code.push(
        ArmAsm::StpGen {
            opc: 0b10,
            imm7: 0,
            rt2: X1,
            rn: SP,
            rt: X0,
            class_selector: StpGenSelector::SignedOffset,
        }
        .encode(),
    );
    // stp x2, x3, [sp, #16]
    code.push(
        ArmAsm::StpGen {
            opc: 0b10,
            imm7: 2,
            rt2: X3,
            rn: SP,
            rt: X2,
            class_selector: StpGenSelector::SignedOffset,
        }
        .encode(),
    );
    // stp x4, x5, [sp, #32]
    code.push(
        ArmAsm::StpGen {
            opc: 0b10,
            imm7: 4,
            rt2: X5,
            rn: SP,
            rt: X4,
            class_selector: StpGenSelector::SignedOffset,
        }
        .encode(),
    );
    // stp x6, x7, [sp, #48]
    code.push(
        ArmAsm::StpGen {
            opc: 0b10,
            imm7: 6,
            rt2: X7,
            rn: SP,
            rt: X6,
            class_selector: StpGenSelector::SignedOffset,
        }
        .encode(),
    );

    // === Set up call to invoke_beagle_callback(callback_index, c_args_ptr, num_args) ===

    // mov x0, #callback_index  (load callback index via movz/movk)
    let idx = callback_index as u64;
    code.push(
        ArmAsm::Movz {
            sf: 1,
            hw: 0,
            imm16: (idx & 0xFFFF) as i32,
            rd: X0,
        }
        .encode(),
    );
    if idx > 0xFFFF {
        code.push(
            ArmAsm::Movk {
                sf: 1,
                hw: 1,
                imm16: ((idx >> 16) & 0xFFFF) as i32,
                rd: X0,
            }
            .encode(),
        );
    }
    if idx > 0xFFFF_FFFF {
        code.push(
            ArmAsm::Movk {
                sf: 1,
                hw: 2,
                imm16: ((idx >> 32) & 0xFFFF) as i32,
                rd: X0,
            }
            .encode(),
        );
    }
    if idx > 0xFFFF_FFFF_FFFF {
        code.push(
            ArmAsm::Movk {
                sf: 1,
                hw: 3,
                imm16: ((idx >> 48) & 0xFFFF) as i32,
                rd: X0,
            }
            .encode(),
        );
    }

    // mov x1, sp  (pointer to saved args)
    code.push(
        ArmAsm::AddAddsubImm {
            sf: 1,
            sh: 0,
            imm12: 0,
            rn: SP,
            rd: X1,
        }
        .encode(),
    );

    // mov x2, #num_args
    code.push(
        ArmAsm::Movz {
            sf: 1,
            hw: 0,
            imm16: num_c_args as i32,
            rd: X2,
        }
        .encode(),
    );

    // Load helper function address (64-bit immediate via movz/movk into x9)
    let addr = helper_fn_ptr as u64;
    code.push(
        ArmAsm::Movz {
            sf: 1,
            hw: 0,
            imm16: (addr & 0xFFFF) as i32,
            rd: X9,
        }
        .encode(),
    );
    code.push(
        ArmAsm::Movk {
            sf: 1,
            hw: 1,
            imm16: ((addr >> 16) & 0xFFFF) as i32,
            rd: X9,
        }
        .encode(),
    );
    code.push(
        ArmAsm::Movk {
            sf: 1,
            hw: 2,
            imm16: ((addr >> 32) & 0xFFFF) as i32,
            rd: X9,
        }
        .encode(),
    );
    code.push(
        ArmAsm::Movk {
            sf: 1,
            hw: 3,
            imm16: ((addr >> 48) & 0xFFFF) as i32,
            rd: X9,
        }
        .encode(),
    );

    // blr x9  (call helper)
    code.push(ArmAsm::Blr { rn: X9 }.encode());

    // x0 now has C return value

    // === Epilogue ===
    // add sp, sp, #64  (restore stack past saved args)
    code.push(
        ArmAsm::AddAddsubImm {
            sf: 1,
            sh: 0,
            imm12: 64,
            rn: SP,
            rd: SP,
        }
        .encode(),
    );
    // ldp x19, x20, [sp], #16
    code.push(
        ArmAsm::LdpGen {
            opc: 0b10,
            imm7: 2,
            rt2: X20,
            rn: SP,
            rt: X19,
            class_selector: LdpGenSelector::PostIndex,
        }
        .encode(),
    );
    // ldp x29, x30, [sp], #16
    code.push(
        ArmAsm::LdpGen {
            opc: 0b10,
            imm7: 2,
            rt2: X30,
            rn: SP,
            rt: X29,
            class_selector: LdpGenSelector::PostIndex,
        }
        .encode(),
    );
    // ret
    code.push(ArmAsm::Ret { rn: X30 }.encode());

    // Convert to bytes and mmap as executable
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let size = bytes.len();
    let page_size = crate::mmap_utils::get_page_size();
    let alloc_size = (size + page_size - 1) & !(page_size - 1);

    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            alloc_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_JIT,
            -1,
            0,
        );
        assert!(
            ptr != libc::MAP_FAILED,
            "mmap failed for callback trampoline"
        );
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, size);
        let ret = libc::mprotect(ptr, alloc_size, libc::PROT_READ | libc::PROT_EXEC);
        assert!(ret == 0, "mprotect failed for callback trampoline");
        ptr as *const u8
    }
}

/// Build a per-callback x86-64 trampoline that:
/// 1. Saves C args (rdi, rsi, rdx, rcx, r8, r9) to the stack as u64 array
/// 2. Calls invoke_beagle_callback(callback_index, c_args_ptr, num_args)
/// 3. Returns the result in rax
///
/// Returns a pointer to executable memory containing the trampoline code.
#[cfg(target_arch = "x86_64")]
fn build_callback_trampoline(
    callback_index: usize,
    helper_fn_ptr: *const u8,
    num_c_args: usize,
) -> *const u8 {
    let mut code: Vec<u8> = Vec::new();

    // === Prologue ===
    // push rbp
    code.push(0x55);
    // mov rbp, rsp
    code.extend_from_slice(&[0x48, 0x89, 0xe5]);
    // push rbx (callee-saved, we use it as scratch)
    code.push(0x53);

    // === Save C args to stack ===
    // sub rsp, 56   (6 regs * 8 bytes + 8 bytes padding for 16-byte alignment)
    // At entry: rsp % 16 == 8 (return addr pushed by caller).
    // After push rbp + push rbx: rsp % 16 == 8.
    // sub 56 (odd multiple of 8) restores rsp % 16 == 0 before 'call'.
    code.extend_from_slice(&[0x48, 0x83, 0xec, 56]);

    // Save C args at [rsp+8..rsp+55], leaving [rsp+0..7] as alignment padding
    // mov [rsp+8], rdi
    code.extend_from_slice(&[0x48, 0x89, 0x7c, 0x24, 0x08]);
    // mov [rsp+16], rsi
    code.extend_from_slice(&[0x48, 0x89, 0x74, 0x24, 0x10]);
    // mov [rsp+24], rdx
    code.extend_from_slice(&[0x48, 0x89, 0x54, 0x24, 0x18]);
    // mov [rsp+32], rcx
    code.extend_from_slice(&[0x48, 0x89, 0x4c, 0x24, 0x20]);
    // mov [rsp+40], r8
    code.extend_from_slice(&[0x4c, 0x89, 0x44, 0x24, 0x28]);
    // mov [rsp+48], r9
    code.extend_from_slice(&[0x4c, 0x89, 0x4c, 0x24, 0x30]);

    // === Set up call to invoke_beagle_callback(callback_index, c_args_ptr, num_args) ===

    // mov rdi, callback_index (64-bit immediate)
    // movabs rdi, imm64
    code.extend_from_slice(&[0x48, 0xbf]);
    code.extend_from_slice(&(callback_index as u64).to_le_bytes());

    // lea rsi, [rsp+8]  (pointer to saved args array, skipping alignment padding)
    code.extend_from_slice(&[0x48, 0x8d, 0x74, 0x24, 0x08]);

    // mov rdx, num_c_args (64-bit immediate)
    // movabs rdx, imm64
    code.extend_from_slice(&[0x48, 0xba]);
    code.extend_from_slice(&(num_c_args as u64).to_le_bytes());

    // Load helper function address into rax and call it
    // movabs rax, imm64
    code.extend_from_slice(&[0x48, 0xb8]);
    code.extend_from_slice(&(helper_fn_ptr as u64).to_le_bytes());

    // call rax
    code.extend_from_slice(&[0xff, 0xd0]);

    // rax now has the C return value

    // === Epilogue ===
    // add rsp, 56  (restore stack past saved args + padding)
    code.extend_from_slice(&[0x48, 0x83, 0xc4, 56]);
    // pop rbx
    code.push(0x5b);
    // pop rbp
    code.push(0x5d);
    // ret
    code.push(0xc3);

    // mmap as executable
    let size = code.len();
    let page_size = crate::mmap_utils::get_page_size();
    let alloc_size = (size + page_size - 1) & !(page_size - 1);

    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            alloc_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        assert!(
            ptr != libc::MAP_FAILED,
            "mmap failed for callback trampoline"
        );
        std::ptr::copy_nonoverlapping(code.as_ptr(), ptr as *mut u8, size);
        let ret = libc::mprotect(ptr, alloc_size, libc::PROT_READ | libc::PROT_EXEC);
        assert!(ret == 0, "mprotect failed for callback trampoline");
        ptr as *const u8
    }
}

/// Create an FFI callback (C → Beagle function pointer).
/// Returns a Pointer struct whose raw pointer can be passed to C functions expecting callbacks.
#[cfg(target_arch = "aarch64")]
pub extern "C" fn create_callback(
    stack_pointer: usize,
    frame_pointer: usize,
    beagle_fn: usize,
    arg_types: usize,
    return_type_value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Parse argument types
    let types_vec: Vec<usize> = persistent_vector_to_vec(arg_types);
    let ffi_arg_types: Vec<FFIType> = types_vec
        .iter()
        .map(|t| match map_beagle_type_to_ffi_type(runtime, *t) {
            Ok(ffi_type) => ffi_type,
            Err(e) => unsafe {
                throw_runtime_error(stack_pointer, "FFIError", e);
            },
        })
        .collect();

    // Parse return type
    let ffi_return_type = match map_beagle_type_to_ffi_type(runtime, return_type_value) {
        Ok(t) => t,
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "FFIError", e);
        },
    };

    let num_c_args = ffi_arg_types.len();

    // Register beagle_fn as a GC root so it stays alive
    let gc_root_id = runtime.register_temporary_root(beagle_fn);

    // Build the trampoline first (before storing CallbackInfo)
    let helper_fn_ptr = invoke_beagle_callback as *const u8;
    let callback_index = runtime.callbacks.len(); // This will be the index
    let trampoline_ptr = build_callback_trampoline(callback_index, helper_fn_ptr, num_c_args);

    // Store callback info in runtime
    let info = crate::runtime::CallbackInfo {
        trampoline_ptr,
        beagle_fn,
        arg_types: ffi_arg_types,
        return_type: ffi_return_type,
        gc_root_id,
    };
    runtime.add_callback(info);

    // Return trampoline pointer as a Pointer struct (lo/hi halves)
    let raw = trampoline_ptr as u64;
    let lo_tagged = BuiltInTypes::Int.tag((raw & 0xFFFFFFFF) as isize) as usize;
    let hi_tagged = BuiltInTypes::Int.tag(((raw >> 32) & 0xFFFFFFFF) as isize) as usize;
    unsafe {
        call_fn_2(
            runtime,
            "beagle.ffi/__make_pointer_struct",
            lo_tagged,
            hi_tagged,
        )
    }
}

/// Create an FFI callback (C → Beagle function pointer) on x86-64.
/// Returns a Pointer struct whose raw pointer can be passed to C functions expecting callbacks.
#[cfg(target_arch = "x86_64")]
pub extern "C" fn create_callback(
    stack_pointer: usize,
    frame_pointer: usize,
    beagle_fn: usize,
    arg_types: usize,
    return_type_value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Parse argument types
    let types_vec: Vec<usize> = persistent_vector_to_vec(arg_types);
    let ffi_arg_types: Vec<FFIType> = types_vec
        .iter()
        .map(|t| match map_beagle_type_to_ffi_type(runtime, *t) {
            Ok(ffi_type) => ffi_type,
            Err(e) => unsafe {
                throw_runtime_error(stack_pointer, "FFIError", e);
            },
        })
        .collect();

    // Parse return type
    let ffi_return_type = match map_beagle_type_to_ffi_type(runtime, return_type_value) {
        Ok(t) => t,
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "FFIError", e);
        },
    };

    let num_c_args = ffi_arg_types.len();

    // Register beagle_fn as a GC root so it stays alive
    let gc_root_id = runtime.register_temporary_root(beagle_fn);

    // Build the trampoline first (before storing CallbackInfo)
    let helper_fn_ptr = invoke_beagle_callback as *const u8;
    let callback_index = runtime.callbacks.len(); // This will be the index
    let trampoline_ptr = build_callback_trampoline(callback_index, helper_fn_ptr, num_c_args);

    // Store callback info in runtime
    let info = crate::runtime::CallbackInfo {
        trampoline_ptr,
        beagle_fn,
        arg_types: ffi_arg_types,
        return_type: ffi_return_type,
        gc_root_id,
    };
    runtime.add_callback(info);

    // Return trampoline pointer as a Pointer struct (lo/hi halves)
    let raw = trampoline_ptr as u64;
    let lo_tagged = BuiltInTypes::Int.tag((raw & 0xFFFFFFFF) as isize) as usize;
    let hi_tagged = BuiltInTypes::Int.tag(((raw >> 32) & 0xFFFFFFFF) as isize) as usize;
    unsafe {
        call_fn_2(
            runtime,
            "beagle.ffi/__make_pointer_struct",
            lo_tagged,
            hi_tagged,
        )
    }
}

/// Marshal a Beagle value to a native u64 for FFI call.
/// This function converts tagged Beagle values to raw C-compatible values.
unsafe fn marshal_ffi_argument(
    runtime: &mut Runtime,
    stack_pointer: usize,
    argument: usize,
    ffi_type: &FFIType,
) -> u64 {
    let kind = BuiltInTypes::get_kind(argument);
    match kind {
        BuiltInTypes::Null => {
            // Null pointer
            0u64
        }
        BuiltInTypes::String => {
            // String literal - convert to C string
            let string = runtime.get_string_literal(argument);
            let c_string = runtime.memory.write_c_string(string);
            c_string as u64
        }
        BuiltInTypes::Int => match ffi_type {
            FFIType::U8
            | FFIType::U16
            | FFIType::U32
            | FFIType::U64
            | FFIType::I8
            | FFIType::I16
            | FFIType::I32
            | FFIType::I64 => {
                // Use signed shift so negative Beagle Ints sign-extend to 64 bits.
                // For I64 this is required (full width visible to C); for narrower
                // signed types C truncates the high bits, and unsigned types are
                // unaffected because Beagle Ints are at most 61 bits.
                BuiltInTypes::untag_isize(argument as isize) as i64 as u64
            }
            FFIType::F32 => {
                // Convert integer to f32 and get its bit representation
                let int_val = BuiltInTypes::untag(argument) as i64;
                let f32_val = int_val as f32;
                f32_val.to_bits() as u64
            }
            FFIType::F64 => {
                let int_val = BuiltInTypes::untag(argument) as i64;
                let f64_val = int_val as f64;
                f64_val.to_bits()
            }
            FFIType::Pointer => {
                if BuiltInTypes::untag(argument) == 0 {
                    0u64
                } else {
                    let heap_object = HeapObject::from_tagged(argument);
                    let buffer = BuiltInTypes::untag(heap_object.get_field(0));
                    buffer as u64
                }
            }
            _ => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "FFIError",
                    format!("Expected integer for FFI type {:?}", ffi_type),
                );
            },
        },
        BuiltInTypes::HeapObject => match ffi_type {
            FFIType::Pointer | FFIType::MutablePointer => {
                let heap_object = HeapObject::from_tagged(argument);
                let struct_id = heap_object.get_struct_id();
                let is_pointer_struct = runtime
                    .get_struct_by_id(struct_id)
                    .map(|s| s.name == "beagle.ffi/Pointer")
                    .unwrap_or(false);
                if is_pointer_struct {
                    // Pointer struct: reconstruct 64-bit raw pointer from lo/hi halves
                    let lo = BuiltInTypes::untag(heap_object.get_field(0)) as u64;
                    let hi = BuiltInTypes::untag(heap_object.get_field(1)) as u64;
                    lo | (hi << 32)
                } else {
                    // Buffer or other struct: field 0 is the full tagged pointer
                    BuiltInTypes::untag(heap_object.get_field(0)) as u64
                }
            }
            FFIType::String => {
                let string = runtime.get_string(stack_pointer, argument);
                let c_string = runtime.memory.write_c_string(string);
                c_string as u64
            }
            FFIType::Structure(_) => {
                let heap_object = HeapObject::from_tagged(argument);
                let buffer = BuiltInTypes::untag(heap_object.get_field(0));
                buffer as u64
            }
            _ => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "FFIError",
                    format!(
                        "Got HeapObject but expected matching FFI type, got {:?}",
                        ffi_type
                    ),
                );
            },
        },
        BuiltInTypes::Float => {
            // Float is heap-allocated: untag to get pointer, read f64 at offset 1
            let ptr = BuiltInTypes::untag(argument) as *const f64;
            let f64_val = unsafe { *ptr.add(1) };
            match ffi_type {
                FFIType::F32 => {
                    let f32_val = f64_val as f32;
                    f32_val.to_bits() as u64
                }
                FFIType::F64 => f64_val.to_bits(),
                FFIType::U8
                | FFIType::U16
                | FFIType::U32
                | FFIType::U64
                | FFIType::I8
                | FFIType::I16
                | FFIType::I32
                | FFIType::I64 => f64_val as i64 as u64,
                _ => unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "FFIError",
                        format!("Cannot convert Float to FFI type {:?}", ffi_type),
                    );
                },
            }
        }
        _ => unsafe {
            runtime.print(argument);
            throw_runtime_error(
                stack_pointer,
                "FFIError",
                format!("Unsupported FFI type: {:?}", kind),
            );
        },
    }
}

/// Build an ARM64 FFI trampoline using the existing assembler.
///
/// The trampoline takes:
///   x0 = func_ptr (the C function to call)
///   x1 = int_args_ptr (pointer to array of 8 u64 integer register values)
///   x2 = float_args_ptr (pointer to array of 8 u64 float arg bit patterns)
///
/// It loads x0-x7 from int_args_ptr, s0-s7 from float_args_ptr,
/// calls func_ptr, and returns (x0, x1) for struct return support.
/// Supports unlimited args: x0-x7 via int_args, s0-s7 via float_args,
/// overflow args placed on the stack per ARM64 AAPCS.
///
/// Signature: trampoline(func_ptr, int_args, float_args, overflow_args, num_overflow, stack_alloc_size)
///                        x0         x1         x2          x3             x4            x5
#[cfg(target_arch = "aarch64")]
fn build_ffi_trampoline() -> *const u8 {
    use crate::machine_code::arm_codegen::*;

    let xzr = Register {
        index: 31,
        size: Size::S64,
    };

    let mut code: Vec<u32> = Vec::with_capacity(48);

    // === Prologue ===
    // 0: stp x29, x30, [sp, #-16]!
    code.push(
        ArmAsm::StpGen {
            opc: 0b10,
            imm7: -2,
            rt2: X30,
            rn: SP,
            rt: X29,
            class_selector: StpGenSelector::PreIndex,
        }
        .encode(),
    );
    // 1: mov x29, sp
    code.push(
        ArmAsm::MovAddAddsubImm {
            sf: 1,
            rn: SP,
            rd: X29,
        }
        .encode(),
    );
    // 2: stp x19, x20, [sp, #-16]!
    code.push(
        ArmAsm::StpGen {
            opc: 0b10,
            imm7: -2,
            rt2: X20,
            rn: SP,
            rt: X19,
            class_selector: StpGenSelector::PreIndex,
        }
        .encode(),
    );
    // 3: stp x21, x22, [sp, #-16]!
    code.push(
        ArmAsm::StpGen {
            opc: 0b10,
            imm7: -2,
            rt2: X22,
            rn: SP,
            rt: X21,
            class_selector: StpGenSelector::PreIndex,
        }
        .encode(),
    );
    // 4: stp x23, x24, [sp, #-16]!
    code.push(
        ArmAsm::StpGen {
            opc: 0b10,
            imm7: -2,
            rt2: X24,
            rn: SP,
            rt: X23,
            class_selector: StpGenSelector::PreIndex,
        }
        .encode(),
    );

    // === Save inputs to callee-saved regs ===
    // 5: mov x19, x0  (func_ptr)
    code.push(
        ArmAsm::MovOrrLogShift {
            sf: 1,
            rm: X0,
            rd: X19,
        }
        .encode(),
    );
    // 6: mov x20, x1  (int_args_ptr)
    code.push(
        ArmAsm::MovOrrLogShift {
            sf: 1,
            rm: X1,
            rd: X20,
        }
        .encode(),
    );
    // 7: mov x21, x2  (float_args_ptr)
    code.push(
        ArmAsm::MovOrrLogShift {
            sf: 1,
            rm: X2,
            rd: X21,
        }
        .encode(),
    );
    // 8: mov x22, x3  (overflow_args_ptr)
    code.push(
        ArmAsm::MovOrrLogShift {
            sf: 1,
            rm: X3,
            rd: X22,
        }
        .encode(),
    );
    // 9: mov x23, x4  (num_overflow)
    code.push(
        ArmAsm::MovOrrLogShift {
            sf: 1,
            rm: X4,
            rd: X23,
        }
        .encode(),
    );
    // 10: mov x24, x5  (stack_alloc_size, pre-aligned to 16)
    code.push(
        ArmAsm::MovOrrLogShift {
            sf: 1,
            rm: X5,
            rd: X24,
        }
        .encode(),
    );

    // === Adjust SP down by x24 (may be 0, which is a no-op) ===
    // 11: add x9, sp, #0  (mov x9, sp)
    code.push(
        ArmAsm::AddAddsubImm {
            sf: 1,
            sh: 0,
            imm12: 0,
            rn: SP,
            rd: X9,
        }
        .encode(),
    );
    // 12: sub x9, x9, x24
    code.push(
        ArmAsm::SubAddsubShift {
            sf: 1,
            shift: 0,
            rm: X24,
            imm6: 0,
            rn: X9,
            rd: X9,
        }
        .encode(),
    );
    // 13: add sp, x9, #0  (mov sp, x9)
    code.push(
        ArmAsm::AddAddsubImm {
            sf: 1,
            sh: 0,
            imm12: 0,
            rn: X9,
            rd: SP,
        }
        .encode(),
    );

    // === Copy overflow args from x22 to stack ===
    // 14: movz x10, #0  (i = 0)
    code.push(
        ArmAsm::Movz {
            sf: 1,
            hw: 0,
            imm16: 0,
            rd: X10,
        }
        .encode(),
    );
    // .copy_loop (instruction 15):
    // 15: cmp x10, x23  (compare i with num_overflow)
    code.push(
        ArmAsm::CmpSubsAddsubShift {
            sf: 1,
            shift: 0,
            rm: X23,
            imm6: 0,
            rn: X10,
        }
        .encode(),
    );
    // 16: b.ge .copy_done  (if i >= num_overflow, skip to instruction 21; offset = 5)
    code.push(
        ArmAsm::BCond {
            imm19: 5,
            cond: 0b1010,
        }
        .encode(),
    ); // GE = 0b1010
    // 17: ldr x12, [x22, x10, lsl #3]  (load overflow_args[i])
    code.push(
        ArmAsm::LdrRegGen {
            size: 3,
            rm: X10,
            option: 0b011,
            s: 1,
            rn: X22,
            rt: X12,
        }
        .encode(),
    );
    // 18: str x12, [sp, x10, lsl #3]  (store to stack[i])
    code.push(
        ArmAsm::StrRegGen {
            size: 3,
            rm: X10,
            option: 0b011,
            s: 1,
            rn: SP,
            rt: X12,
        }
        .encode(),
    );
    // 19: add x10, x10, #1
    code.push(
        ArmAsm::AddAddsubImm {
            sf: 1,
            sh: 0,
            imm12: 1,
            rn: X10,
            rd: X10,
        }
        .encode(),
    );
    // 20: b .copy_loop  (unconditional jump back to instruction 15; offset = -5)
    code.push(
        ArmAsm::BCond {
            imm19: -5,
            cond: 0b1110,
        }
        .encode(),
    ); // AL = 0b1110

    // .copy_done (instruction 21):
    // === Load float args from x21 into d0-d7 ===
    // We always load 64 bits via fmov dN, x9. This is backward-compatible:
    // F32 values have upper 32 bits zero in the u64, and sN is bits[31:0] of dN,
    // so callees reading sN still get the correct f32 value.
    for i in 0..8u8 {
        let float_reg = Register {
            index: i,
            size: Size::S64,
        };
        // ldr x9, [x21, #i*8]
        code.push(
            ArmAsm::LdrImmGen {
                size: 3,
                imm9: 0,
                rn: X21,
                rt: Register {
                    index: 9,
                    size: Size::S64,
                },
                imm12: i as i32,
                class_selector: LdrImmGenSelector::UnsignedOffset,
            }
            .encode(),
        );
        // fmov dN, x9
        code.push(
            ArmAsm::FmovFloatGen {
                sf: 1,
                ftype: 0b01,
                rmode: 0b00,
                opcode: 0b111,
                rn: Register {
                    index: 9,
                    size: Size::S64,
                },
                rd: float_reg,
            }
            .encode(),
        );
    }

    // === Load integer args from x20 into x0-x7 ===
    for pair in 0..4 {
        let r1 = Register::from_index(pair * 2);
        let r2 = Register::from_index(pair * 2 + 1);
        code.push(
            ArmAsm::LdpGen {
                opc: 0b10,
                imm7: (pair as i32) * 2,
                rt2: r2,
                rn: X20,
                rt: r1,
                class_selector: LdpGenSelector::SignedOffset,
            }
            .encode(),
        );
    }

    // === Call the function ===
    code.push(ArmAsm::Blr { rn: X19 }.encode());

    // === Capture both integer and float return values ===
    // Save x0/x1 (integer/pointer return) to int_args buffer via x20,
    // then overwrite x0/x1 with d0/d1 bits (float return).
    // This way the trampoline's register return (x0:x1) carries d0:d1 for
    // float/struct returns, while int_args carries the original x0:x1 for
    // integer/pointer returns. The Rust caller picks the right one based
    // on return_type — no inline asm or compiler workarounds needed.

    // str x0, [x20]  (save integer return low to int_args[0])
    code.push(
        ArmAsm::StrImmGen {
            size: 3,
            imm9: 0,
            rn: X20,
            rt: X0,
            imm12: 0,
            class_selector: StrImmGenSelector::UnsignedOffset,
        }
        .encode(),
    );
    // str x1, [x20, #8]  (save integer return high to int_args[1])
    code.push(
        ArmAsm::StrImmGen {
            size: 3,
            imm9: 0,
            rn: X20,
            rt: X1,
            imm12: 1, // offset = 1 * 8 = 8 bytes
            class_selector: StrImmGenSelector::UnsignedOffset,
        }
        .encode(),
    );
    // fmov x0, d0  (float return bits → x0)
    code.push(
        ArmAsm::FmovFloatGen {
            sf: 1,
            ftype: 0b01,
            rmode: 0b00,
            opcode: 0b110, // float-to-general
            rn: Register {
                index: 0,
                size: Size::S64,
            }, // d0
            rd: X0,
        }
        .encode(),
    );
    // fmov x1, d1  (float return bits → x1)
    code.push(
        ArmAsm::FmovFloatGen {
            sf: 1,
            ftype: 0b01,
            rmode: 0b00,
            opcode: 0b110, // float-to-general
            rn: Register {
                index: 1,
                size: Size::S64,
            }, // d1
            rd: X1,
        }
        .encode(),
    );

    // === Restore SP (add x24 back) ===
    // add x9, sp, #0  (mov x9, sp)
    code.push(
        ArmAsm::AddAddsubImm {
            sf: 1,
            sh: 0,
            imm12: 0,
            rn: SP,
            rd: X9,
        }
        .encode(),
    );
    // add x9, x9, x24
    code.push(
        ArmAsm::AddAddsubShift {
            sf: 1,
            shift: 0,
            rm: X24,
            imm6: 0,
            rn: X9,
            rd: X9,
        }
        .encode(),
    );
    // add sp, x9, #0  (mov sp, x9)
    code.push(
        ArmAsm::AddAddsubImm {
            sf: 1,
            sh: 0,
            imm12: 0,
            rn: X9,
            rd: SP,
        }
        .encode(),
    );

    // === Epilogue ===
    // ldp x23, x24, [sp], #16
    code.push(
        ArmAsm::LdpGen {
            opc: 0b10,
            imm7: 2,
            rt2: X24,
            rn: SP,
            rt: X23,
            class_selector: LdpGenSelector::PostIndex,
        }
        .encode(),
    );
    // ldp x21, x22, [sp], #16
    code.push(
        ArmAsm::LdpGen {
            opc: 0b10,
            imm7: 2,
            rt2: X22,
            rn: SP,
            rt: X21,
            class_selector: LdpGenSelector::PostIndex,
        }
        .encode(),
    );
    // ldp x19, x20, [sp], #16
    code.push(
        ArmAsm::LdpGen {
            opc: 0b10,
            imm7: 2,
            rt2: X20,
            rn: SP,
            rt: X19,
            class_selector: LdpGenSelector::PostIndex,
        }
        .encode(),
    );
    // ldp x29, x30, [sp], #16
    code.push(
        ArmAsm::LdpGen {
            opc: 0b10,
            imm7: 2,
            rt2: X30,
            rn: SP,
            rt: X29,
            class_selector: LdpGenSelector::PostIndex,
        }
        .encode(),
    );
    // ret
    code.push(ArmAsm::Ret { rn: X30 }.encode());

    // Suppress unused variable warning
    let _ = xzr;

    // Convert to bytes and mmap as executable
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let size = bytes.len();
    let page_size = crate::mmap_utils::get_page_size();
    let alloc_size = (size + page_size - 1) & !(page_size - 1);

    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            alloc_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_JIT,
            -1,
            0,
        );
        assert!(ptr != libc::MAP_FAILED, "mmap failed for FFI trampoline");
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, size);
        let ret = libc::mprotect(ptr, alloc_size, libc::PROT_READ | libc::PROT_EXEC);
        assert!(ret == 0, "mprotect failed for FFI trampoline");
        ptr as *const u8
    }
}

/// Get the cached FFI trampoline function pointer.
#[cfg(target_arch = "aarch64")]
fn get_ffi_trampoline() -> *const u8 {
    use std::sync::OnceLock;
    static TRAMPOLINE: OnceLock<usize> = OnceLock::new();
    *TRAMPOLINE.get_or_init(|| build_ffi_trampoline() as usize) as *const u8
}

/// Call a native C function dynamically using the JIT'd trampoline.
/// Splits args into integer and float register arrays based on arg_types.
/// Args beyond 8 int or 8 float registers overflow to the stack per ARM64 AAPCS.
/// Returns (low, high) for struct return support.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn dynamic_c_call(
    func_ptr: *const u8,
    args: &[u64],
    arg_types: &[FFIType],
    return_type: &FFIType,
) -> (u64, u64) {
    // Split args into integer regs, float regs, and overflow (stack) based on ARM64 AAPCS
    let mut int_args: [u64; 8] = [0; 8];
    let mut float_args: [u64; 8] = [0; 8];
    let mut overflow_args: Vec<u64> = Vec::new();
    let mut int_idx = 0;
    let mut float_idx = 0;

    for (i, arg_type) in arg_types.iter().enumerate() {
        if i < args.len() {
            if matches!(arg_type, FFIType::F32 | FFIType::F64) {
                if float_idx < 8 {
                    float_args[float_idx] = args[i];
                    float_idx += 1;
                } else {
                    overflow_args.push(args[i]);
                }
            } else if int_idx < 8 {
                int_args[int_idx] = args[i];
                int_idx += 1;
            } else {
                overflow_args.push(args[i]);
            }
        }
    }

    let num_overflow = overflow_args.len();
    let stack_alloc_size = (num_overflow * 8 + 15) & !15; // 16-byte aligned

    // Call the trampoline. It returns d0:d1 bits in x0:x1 (the u128 return),
    // and saves the original x0:x1 (integer return) to int_args[0..1].
    // We pick the right pair based on return_type.
    let tramp_fn: unsafe extern "C" fn(
        u64,
        *mut u64,
        *const u64,
        *const u64,
        usize,
        usize,
    ) -> u128 = unsafe { std::mem::transmute(get_ffi_trampoline()) };
    let result = unsafe {
        tramp_fn(
            func_ptr as u64,
            int_args.as_mut_ptr(),
            float_args.as_ptr(),
            overflow_args.as_ptr(),
            num_overflow,
            stack_alloc_size,
        )
    };
    let low = result as u64;
    let high = (result >> 64) as u64;

    // For float/struct returns, the trampoline put d0:d1 bits into x0:x1,
    // which we get directly from (low, high).
    if let FFIType::Structure(fields) = return_type
        && fields
            .iter()
            .all(|f| matches!(f, FFIType::F32 | FFIType::F64))
    {
        return (low, if fields.len() > 1 { high } else { 0 });
    }
    if matches!(return_type, FFIType::F32 | FFIType::F64) {
        return (low, 0);
    }

    // For integer/pointer returns, the trampoline saved original x0:x1
    // into int_args[0..1].
    (int_args[0], int_args[1])
}

/// x86-64 fallback: use transmute-based dispatch (same as before but cleaned up).
#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn dynamic_c_call(
    func_ptr: *const u8,
    args: &[u64],
    arg_types: &[FFIType],
    _return_type: &FFIType,
) -> (u64, u64) {
    // On x86-64, integer and float registers are independently allocated:
    // Integer: rdi, rsi, rdx, rcx, r8, r9
    // Float: xmm0-xmm7
    // For now, use the simple transmute approach for x86-64.

    let has_float = arg_types
        .iter()
        .any(|t| matches!(t, FFIType::F32 | FFIType::F64));
    let num_args = args.len();

    // Pad args to 8 for indexing safety
    let mut padded: [u64; 8] = [0; 8];
    for (i, &a) in args.iter().enumerate().take(8) {
        padded[i] = a;
    }

    if has_float {
        fn to_f32(v: u64) -> f32 {
            f32::from_bits(v as u32)
        }
        fn to_f64(v: u64) -> f64 {
            f64::from_bits(v)
        }
        // float_mask tracks F32 args, double_mask tracks F64 args
        let mut float_mask: u8 = 0;
        let mut double_mask: u8 = 0;
        for (i, t) in arg_types.iter().enumerate() {
            if matches!(t, FFIType::F32) {
                float_mask |= 1 << i;
            }
            if matches!(t, FFIType::F64) {
                double_mask |= 1 << i;
            }
        }
        // If any F64 args, handle F64 patterns
        if double_mask != 0 {
            unsafe {
                match (num_args, double_mask) {
                    (1, 0b0001) => {
                        let f: extern "C" fn(f64) -> u64 = transmute(func_ptr);
                        return (f(to_f64(padded[0])), 0);
                    }
                    (2, 0b0010) => {
                        let f: extern "C" fn(u64, f64) -> u64 = transmute(func_ptr);
                        return (f(padded[0], to_f64(padded[1])), 0);
                    }
                    (2, 0b0011) => {
                        let f: extern "C" fn(f64, f64) -> u64 = transmute(func_ptr);
                        return (f(to_f64(padded[0]), to_f64(padded[1])), 0);
                    }
                    (4, 0b1111) => {
                        let f: extern "C" fn(f64, f64, f64, f64) -> u64 = transmute(func_ptr);
                        return (
                            f(
                                to_f64(padded[0]),
                                to_f64(padded[1]),
                                to_f64(padded[2]),
                                to_f64(padded[3]),
                            ),
                            0,
                        );
                    }
                    _ => {
                        let sp = get_saved_stack_pointer();
                        throw_runtime_error(
                            sp,
                            "FFIError",
                            format!(
                                "Unsupported f64 argument pattern on x86-64: {} args, double_mask=0b{:08b}",
                                num_args, double_mask
                            ),
                        );
                    }
                }
            }
        }
        // Handle common float patterns (same as old code)
        unsafe {
            match (num_args, float_mask) {
                (1, 0b0001) => {
                    let f: extern "C" fn(f32) -> u64 = transmute(func_ptr);
                    return (f(to_f32(padded[0])), 0);
                }
                (2, 0b0001) => {
                    let f: extern "C" fn(f32, u64) -> u64 = transmute(func_ptr);
                    return (f(to_f32(padded[0]), padded[1]), 0);
                }
                (2, 0b0010) => {
                    let f: extern "C" fn(u64, f32) -> u64 = transmute(func_ptr);
                    return (f(padded[0], to_f32(padded[1])), 0);
                }
                (2, 0b0011) => {
                    let f: extern "C" fn(f32, f32) -> u64 = transmute(func_ptr);
                    return (f(to_f32(padded[0]), to_f32(padded[1])), 0);
                }
                (3, 0b0001) => {
                    let f: extern "C" fn(f32, u64, u64) -> u64 = transmute(func_ptr);
                    return (f(to_f32(padded[0]), padded[1], padded[2]), 0);
                }
                (3, 0b0010) => {
                    let f: extern "C" fn(u64, f32, u64) -> u64 = transmute(func_ptr);
                    return (f(padded[0], to_f32(padded[1]), padded[2]), 0);
                }
                (3, 0b0100) => {
                    let f: extern "C" fn(u64, u64, f32) -> u64 = transmute(func_ptr);
                    return (f(padded[0], padded[1], to_f32(padded[2])), 0);
                }
                (4, 0b0001) => {
                    let f: extern "C" fn(f32, u64, u64, u64) -> u64 = transmute(func_ptr);
                    return (f(to_f32(padded[0]), padded[1], padded[2], padded[3]), 0);
                }
                (4, 0b0010) => {
                    let f: extern "C" fn(u64, f32, u64, u64) -> u64 = transmute(func_ptr);
                    return (f(padded[0], to_f32(padded[1]), padded[2], padded[3]), 0);
                }
                (4, 0b0100) => {
                    let f: extern "C" fn(u64, u64, f32, u64) -> u64 = transmute(func_ptr);
                    return (f(padded[0], padded[1], to_f32(padded[2]), padded[3]), 0);
                }
                (4, 0b1000) => {
                    let f: extern "C" fn(u64, u64, u64, f32) -> u64 = transmute(func_ptr);
                    return (f(padded[0], padded[1], padded[2], to_f32(padded[3])), 0);
                }
                (4, 0b0011) => {
                    let f: extern "C" fn(f32, f32, u64, u64) -> u64 = transmute(func_ptr);
                    return (
                        f(to_f32(padded[0]), to_f32(padded[1]), padded[2], padded[3]),
                        0,
                    );
                }
                (4, 0b0110) => {
                    let f: extern "C" fn(u64, f32, f32, u64) -> u64 = transmute(func_ptr);
                    return (
                        f(padded[0], to_f32(padded[1]), to_f32(padded[2]), padded[3]),
                        0,
                    );
                }
                (4, 0b1100) => {
                    let f: extern "C" fn(u64, u64, f32, f32) -> u64 = transmute(func_ptr);
                    return (
                        f(padded[0], padded[1], to_f32(padded[2]), to_f32(padded[3])),
                        0,
                    );
                }
                (5, 0b00100) => {
                    let f: extern "C" fn(u64, u64, f32, u64, u64) -> u64 = transmute(func_ptr);
                    return (
                        f(
                            padded[0],
                            padded[1],
                            to_f32(padded[2]),
                            padded[3],
                            padded[4],
                        ),
                        0,
                    );
                }
                (6, 0b000100) => {
                    let f: extern "C" fn(u64, u64, f32, u64, u64, u64) -> u64 = transmute(func_ptr);
                    return (
                        f(
                            padded[0],
                            padded[1],
                            to_f32(padded[2]),
                            padded[3],
                            padded[4],
                            padded[5],
                        ),
                        0,
                    );
                }
                _ => {
                    let sp = get_saved_stack_pointer();
                    throw_runtime_error(
                        sp,
                        "FFIError",
                        format!(
                            "Unsupported float argument pattern on x86-64: {} args, float_mask=0b{:06b}",
                            num_args, float_mask
                        ),
                    );
                }
            }
        }
    }

    // All integer args - simple dispatch
    unsafe {
        let result = match num_args {
            0 => {
                let f: extern "C" fn() -> u64 = transmute(func_ptr);
                f()
            }
            1 => {
                let f: extern "C" fn(u64) -> u64 = transmute(func_ptr);
                f(padded[0])
            }
            2 => {
                let f: extern "C" fn(u64, u64) -> u64 = transmute(func_ptr);
                f(padded[0], padded[1])
            }
            3 => {
                let f: extern "C" fn(u64, u64, u64) -> u64 = transmute(func_ptr);
                f(padded[0], padded[1], padded[2])
            }
            4 => {
                let f: extern "C" fn(u64, u64, u64, u64) -> u64 = transmute(func_ptr);
                f(padded[0], padded[1], padded[2], padded[3])
            }
            5 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64) -> u64 = transmute(func_ptr);
                f(padded[0], padded[1], padded[2], padded[3], padded[4])
            }
            6 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 = transmute(func_ptr);
                f(
                    padded[0], padded[1], padded[2], padded[3], padded[4], padded[5],
                )
            }
            _ => {
                let sp = get_saved_stack_pointer();
                throw_runtime_error(
                    sp,
                    "FFIError",
                    format!(
                        "Too many arguments ({}) for FFI call on x86-64. Maximum supported: 6 integer/pointer args.",
                        num_args
                    ),
                );
            }
        };
        (result, 0)
    }
}

/// Convert a native return value to a Beagle value.
/// Takes the (low, high) result pair from dynamic_c_call for struct return support.
unsafe fn unmarshal_ffi_return(
    runtime: &mut Runtime,
    stack_pointer: usize,
    low: u64,
    high: u64,
    return_type: &FFIType,
) -> usize {
    unsafe {
        match return_type {
            FFIType::Void => BuiltInTypes::null_value() as usize,
            FFIType::U8 | FFIType::U16 | FFIType::U32 | FFIType::U64 => {
                BuiltInTypes::Int.tag(low as isize) as usize
            }
            FFIType::I8 => {
                let signed_result = low as i8 as isize;
                BuiltInTypes::Int.tag(signed_result) as usize
            }
            FFIType::I16 => {
                let signed_result = low as i16 as isize;
                BuiltInTypes::Int.tag(signed_result) as usize
            }
            FFIType::I32 => {
                // Sign-extend I32 to isize properly
                let signed_result = low as i32 as isize;
                BuiltInTypes::Int.tag(signed_result) as usize
            }
            FFIType::I64 => {
                let signed_result = low as i64 as isize;
                BuiltInTypes::Int.tag(signed_result) as usize
            }
            FFIType::F32 => {
                // Convert f32 return to a heap-allocated Beagle float (f64)
                let f32_val = f32::from_bits(low as u32);
                let f64_val = f32_val as f64;
                let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
                    Ok(ptr) => ptr,
                    Err(_) => {
                        throw_runtime_error(
                            stack_pointer,
                            "AllocationError",
                            "Failed to allocate FFI float result - out of memory".to_string(),
                        );
                    }
                };
                let untagged_result = BuiltInTypes::untag(new_float_ptr);
                let result_ptr = untagged_result as *mut f64;
                *result_ptr.add(1) = f64_val;
                new_float_ptr
            }
            FFIType::F64 => {
                let f64_val = f64::from_bits(low);
                let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
                    Ok(ptr) => ptr,
                    Err(_) => {
                        throw_runtime_error(
                            stack_pointer,
                            "AllocationError",
                            "Failed to allocate FFI float result - out of memory".to_string(),
                        );
                    }
                };
                let untagged_result = BuiltInTypes::untag(new_float_ptr);
                let result_ptr = untagged_result as *mut f64;
                *result_ptr.add(1) = f64_val;
                new_float_ptr
            }
            FFIType::Pointer | FFIType::MutablePointer => {
                // Split 64-bit raw pointer into two 32-bit halves to avoid
                // truncation by Int.tag() (which shifts left by 3, losing top 3 bits).
                // This is critical for ObjC tagged pointers which use bit 63.
                let lo = (low & 0xFFFFFFFF) as isize;
                let hi = ((low >> 32) & 0xFFFFFFFF) as isize;
                let lo_tagged = BuiltInTypes::Int.tag(lo) as usize;
                let hi_tagged = BuiltInTypes::Int.tag(hi) as usize;
                call_fn_2(
                    runtime,
                    "beagle.ffi/__make_pointer_struct",
                    lo_tagged,
                    hi_tagged,
                )
            }
            FFIType::String => {
                if low == 0 {
                    return BuiltInTypes::null_value() as usize;
                }
                let c_string = CStr::from_ptr(low as *const i8);
                let string = match c_string.to_str() {
                    Ok(s) => s,
                    Err(e) => {
                        throw_runtime_error(
                            stack_pointer,
                            "EncodingError",
                            format!("FFI returned invalid UTF-8 string: {}", e),
                        );
                    }
                };
                match runtime.allocate_string(stack_pointer, string.to_string()) {
                    Ok(s) => s.into(),
                    Err(_) => {
                        throw_runtime_error(
                            stack_pointer,
                            "AllocationError",
                            "Failed to allocate string from FFI return".to_string(),
                        );
                    }
                }
            }
            FFIType::Structure(fields) => {
                let all_float_fields = fields
                    .iter()
                    .all(|f| matches!(f, FFIType::F32 | FFIType::F64));
                if all_float_fields {
                    // All-float struct (e.g., NSPoint): low/high contain f64 bit
                    // patterns from float registers d0/d1. Convert each field to
                    // a proper heap-allocated Beagle float.
                    let values: [u64; 2] = [low, high];
                    let mut tagged_fields = [BuiltInTypes::null_value() as usize; 2];
                    for (i, field_type) in fields.iter().enumerate().take(2) {
                        let f_val = match field_type {
                            FFIType::F32 => f32::from_bits(values[i] as u32) as f64,
                            FFIType::F64 => f64::from_bits(values[i]),
                            _ => unreachable!(),
                        };
                        let new_float_ptr =
                            match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
                                Ok(ptr) => ptr,
                                Err(_) => {
                                    throw_runtime_error(
                                        stack_pointer,
                                        "AllocationError",
                                        "Failed to allocate FFI float struct field - out of memory"
                                            .to_string(),
                                    );
                                }
                            };
                        let untagged = BuiltInTypes::untag(new_float_ptr);
                        let float_ptr = untagged as *mut f64;
                        *float_ptr.add(1) = f_val;
                        tagged_fields[i] = new_float_ptr;
                    }
                    call_fn_2(
                        runtime,
                        "beagle.ffi/__make_struct_return",
                        tagged_fields[0],
                        tagged_fields[1],
                    )
                } else {
                    // Non-HFA struct: low/high contain raw bytes from x0/x1.
                    call_fn_2(
                        runtime,
                        "beagle.ffi/__make_struct_return",
                        BuiltInTypes::Int.tag(low as isize) as usize,
                        BuiltInTypes::Int.tag(high as isize) as usize,
                    )
                }
            }
        }
    }
}

/// Call a foreign function through FFI.
/// Takes (stack_pointer, frame_pointer, ffi_info_id, args_array) where
/// args_array is a Beagle array (rest param) containing the actual arguments.
pub unsafe extern "C" fn call_ffi_info(
    stack_pointer: usize,
    _frame_pointer: usize,
    ffi_info_id: usize,
    args_array: usize,
) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        let ffi_info_id = BuiltInTypes::untag(ffi_info_id);

        // Extract function info
        let (func_ptr, number_of_arguments, argument_types, return_type) = {
            let ffi_info = runtime.get_ffi_info(ffi_info_id);
            (
                ffi_info.function.ptr,
                ffi_info.number_of_arguments,
                ffi_info.argument_types.clone(),
                ffi_info.return_type.clone(),
            )
        };

        // Read args from the Beagle array
        let args_obj = HeapObject::from_tagged(args_array);
        let fields = args_obj.get_fields();

        if fields.len() != number_of_arguments {
            throw_runtime_error(
                stack_pointer,
                "FFIError",
                format!(
                    "FFI function expects {} arguments, got {}",
                    number_of_arguments,
                    fields.len()
                ),
            );
        }

        // Marshal arguments to native u64 values
        let mut native_args = Vec::with_capacity(number_of_arguments);
        for (argument, ffi_type) in fields.iter().zip(argument_types.iter()) {
            native_args.push(marshal_ffi_argument(
                runtime,
                stack_pointer,
                *argument,
                ffi_type,
            ));
        }

        // Call the native function using the dynamic trampoline
        let (low, high) = dynamic_c_call(func_ptr, &native_args, &argument_types, &return_type);

        // Unmarshal the return value
        let return_value = unmarshal_ffi_return(runtime, stack_pointer, low, high, &return_type);

        runtime.memory.clear_native_arguments();
        return_value
    }
}

pub unsafe extern "C" fn ffi_allocate(size: usize) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        // TODO: I intentionally don't want to manage this memory on the heap
        // I probably need a better answer than this
        // but for now we are just going to leak memory
        let size = BuiltInTypes::untag(size);

        let mut buffer: Vec<u8> = vec![0; size];
        let buffer_ptr: *mut c_void = buffer.as_mut_ptr() as *mut c_void;
        std::mem::forget(buffer);

        let buffer = BuiltInTypes::Int.tag(buffer_ptr as isize) as usize;
        let size = BuiltInTypes::Int.tag(size as isize) as usize;
        call_fn_2(runtime, "beagle.ffi/__make_buffer_struct", buffer, size)
    }
}

/// Extract a raw pointer from a Beagle struct (Pointer or Buffer).
/// Pointer struct { lo, hi }: reconstruct from two 32-bit halves.
/// Buffer struct { ptr, size }: field 0 is the full tagged pointer.
pub unsafe fn extract_raw_ptr(tagged_struct: usize) -> *mut u8 {
    let heap_object = HeapObject::from_tagged(tagged_struct);
    let struct_id = heap_object.get_struct_id();
    let runtime = get_runtime().get_mut();
    let runtime = &*runtime;
    let is_pointer = runtime
        .get_struct_by_id(struct_id)
        .map(|s| s.name == "beagle.ffi/Pointer")
        .unwrap_or(false);
    if is_pointer {
        let lo = BuiltInTypes::untag(heap_object.get_field(0)) as u64;
        let hi = BuiltInTypes::untag(heap_object.get_field(1)) as u64;
        (lo | (hi << 32)) as *mut u8
    } else {
        BuiltInTypes::untag(heap_object.get_field(0)) as *mut u8
    }
}

pub unsafe extern "C" fn ffi_deallocate(buffer: usize) -> usize {
    unsafe {
        // deallocate is Buffer-only: field 0 = ptr, field 1 = size
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let size = BuiltInTypes::untag(buffer_object.get_field(1));
        let _buffer = Vec::from_raw_parts(buffer, size, size);
        BuiltInTypes::null_value() as usize
    }
}

pub unsafe extern "C" fn ffi_get_u32(buffer: usize, offset: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset) as *const u32);
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

pub unsafe extern "C" fn ffi_set_u8(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value);
        assert!(value <= u8::MAX as usize);
        let value = value as u8;
        *(buffer.add(offset)) = value;
        BuiltInTypes::null_value() as usize
    }
}

pub unsafe extern "C" fn ffi_get_u8(buffer: usize, offset: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset));
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

pub unsafe extern "C" fn ffi_set_i32(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value) as i32;
        *(buffer.add(offset) as *mut i32) = value;
        BuiltInTypes::null_value() as usize
    }
}

pub unsafe extern "C" fn ffi_set_i16(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value) as i16;
        *(buffer.add(offset) as *mut i16) = value;
        BuiltInTypes::null_value() as usize
    }
}

pub unsafe extern "C" fn ffi_get_i32(buffer: usize, offset: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset) as *const i32);
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

pub unsafe extern "C" fn ffi_set_i8(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value) as i8;
        *(buffer.add(offset) as *mut i8) = value;
        BuiltInTypes::null_value() as usize
    }
}

pub unsafe extern "C" fn ffi_get_i8(buffer: usize, offset: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset) as *const i8);
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

pub unsafe extern "C" fn ffi_set_u16(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value) as u16;
        *(buffer.add(offset) as *mut u16) = value;
        BuiltInTypes::null_value() as usize
    }
}

pub unsafe extern "C" fn ffi_get_u16(buffer: usize, offset: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset) as *const u16);
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

pub unsafe extern "C" fn ffi_get_i16(buffer: usize, offset: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset) as *const i16);
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

pub unsafe extern "C" fn ffi_set_u32(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value) as u32;
        *(buffer.add(offset) as *mut u32) = value;
        BuiltInTypes::null_value() as usize
    }
}

pub unsafe extern "C" fn ffi_set_i64(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value) as i64;
        *(buffer.add(offset) as *mut i64) = value;
        BuiltInTypes::null_value() as usize
    }
}

pub unsafe extern "C" fn ffi_get_i64(buffer: usize, offset: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset) as *const i64);
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

pub unsafe extern "C" fn ffi_set_u64(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value) as u64;
        *(buffer.add(offset) as *mut u64) = value;
        BuiltInTypes::null_value() as usize
    }
}

pub unsafe extern "C" fn ffi_get_u64(buffer: usize, offset: usize) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset) as *const u64);
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

/// Extract an f64 value from a Beagle Int or Float.
/// Throws FFIError for any other type.
unsafe fn ffi_value_as_f64(stack_pointer: usize, value: usize) -> f64 {
    match BuiltInTypes::get_kind(value) {
        BuiltInTypes::Int => BuiltInTypes::untag(value) as i64 as f64,
        BuiltInTypes::Float => {
            let ptr = BuiltInTypes::untag(value) as *const f64;
            unsafe { *ptr.add(1) }
        }
        kind => unsafe {
            throw_runtime_error(
                stack_pointer,
                "FFIError",
                format!("Expected Int or Float, got {:?}", kind),
            );
        },
    }
}

pub unsafe extern "C" fn ffi_set_f32(
    stack_pointer: usize,
    _frame_pointer: usize,
    buffer: usize,
    offset: usize,
    value: usize,
) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let f64_val = ffi_value_as_f64(stack_pointer, value);
        *(buffer.add(offset) as *mut f32) = f64_val as f32;
        BuiltInTypes::null_value() as usize
    }
}

pub unsafe extern "C" fn ffi_set_f64(
    stack_pointer: usize,
    _frame_pointer: usize,
    buffer: usize,
    offset: usize,
    value: usize,
) -> usize {
    unsafe {
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let f64_val = ffi_value_as_f64(stack_pointer, value);
        *(buffer.add(offset) as *mut f64) = f64_val;
        BuiltInTypes::null_value() as usize
    }
}

pub unsafe extern "C" fn ffi_get_f32(
    stack_pointer: usize,
    _frame_pointer: usize,
    buffer: usize,
    offset: usize,
) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        let buffer_ptr = extract_raw_ptr(buffer);
        let offset_val = BuiltInTypes::untag(offset);
        let f32_val = *(buffer_ptr.add(offset_val) as *const f32);
        let f64_val = f32_val as f64;
        let new_float_ptr = match (*runtime).allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate Float for get-f32 - out of memory".to_string(),
                );
            }
        };
        let untagged = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged as *mut f64;
        *result_ptr.add(1) = f64_val;
        new_float_ptr
    }
}

pub unsafe extern "C" fn ffi_get_f64(
    stack_pointer: usize,
    _frame_pointer: usize,
    buffer: usize,
    offset: usize,
) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        let buffer_ptr = extract_raw_ptr(buffer);
        let offset_val = BuiltInTypes::untag(offset);
        let f64_val = *(buffer_ptr.add(offset_val) as *const f64);
        let new_float_ptr = match (*runtime).allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate Float for get-f64 - out of memory".to_string(),
                );
            }
        };
        let untagged = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged as *mut f64;
        *result_ptr.add(1) = f64_val;
        new_float_ptr
    }
}

pub unsafe extern "C" fn ffi_get_string(
    stack_pointer: usize,
    _frame_pointer: usize, // Frame pointer for GC stack walking
    buffer: usize,
    offset: usize,
    len: usize,
) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        let buffer = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);
        let slice = std::slice::from_raw_parts(buffer.add(offset), len);
        let string = match std::str::from_utf8(slice) {
            Ok(s) => s,
            Err(e) => {
                throw_runtime_error(
                    stack_pointer,
                    "EncodingError",
                    format!("Buffer contains invalid UTF-8: {}", e),
                );
            }
        };
        match (*runtime).allocate_string(stack_pointer, string.to_string()) {
            Ok(s) => s.into(),
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate string from buffer".to_string(),
                );
            }
        }
    }
}

/// Combined get-string + deallocate: extracts string bytes and frees the native
/// buffer BEFORE allocating the Beagle string (which can trigger GC).
/// This avoids the caller needing to hold a Buffer struct reference across GC.
pub unsafe extern "C" fn ffi_get_string_and_free(
    stack_pointer: usize,
    _frame_pointer: usize,
    buffer: usize,
    offset: usize,
    len: usize,
) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        // Extract the raw pointer and size from the Buffer struct
        let buffer_object = HeapObject::from_tagged(buffer);
        let raw_ptr = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let buf_size = BuiltInTypes::untag(buffer_object.get_field(1));
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);

        // Copy string bytes into a Rust String (no GC)
        let slice = std::slice::from_raw_parts(raw_ptr.add(offset), len);
        let string_data = match std::str::from_utf8(slice) {
            Ok(s) => s.to_string(),
            Err(e) => {
                // Free the buffer before throwing error
                let _ = Vec::from_raw_parts(raw_ptr, buf_size, buf_size);
                throw_runtime_error(
                    stack_pointer,
                    "EncodingError",
                    format!("Buffer contains invalid UTF-8: {}", e),
                );
            }
        };

        // Free the native buffer NOW, before any GC can happen
        let _ = Vec::from_raw_parts(raw_ptr, buf_size, buf_size);

        // Allocate the Beagle string (may trigger GC, but Buffer is already freed)
        match (*runtime).allocate_string(stack_pointer, string_data) {
            Ok(s) => s.into(),
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate string from buffer".to_string(),
                );
            }
        }
    }
}

pub unsafe extern "C" fn ffi_create_array(
    stack_pointer: usize,
    _frame_pointer: usize, // Frame pointer for GC stack walking
    ffi_type: usize,
    array: usize,
) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        let ffi_type = match map_beagle_type_to_ffi_type(runtime, ffi_type) {
            Ok(t) => t,
            Err(e) => {
                throw_runtime_error(stack_pointer, "FFIError", e);
            }
        };
        let array = HeapObject::from_tagged(array);
        let fields = array.get_fields();
        let size = fields.len();
        let mut buffer: Vec<*mut i8> = Vec::with_capacity(size);
        for field in fields {
            match ffi_type {
                FFIType::U8 => {
                    let val = BuiltInTypes::untag(*field) as u8;
                    buffer.push(val as *mut i8);
                }
                FFIType::U16 => {
                    let val = BuiltInTypes::untag(*field) as u16;
                    buffer.push(val as *mut i8);
                }
                FFIType::U32 => {
                    let val = BuiltInTypes::untag(*field) as u32;
                    buffer.push(val as *mut i8);
                }
                FFIType::U64 => {
                    let val = BuiltInTypes::untag(*field) as u64;
                    buffer.push(val as *mut i8);
                }
                FFIType::I8 => {
                    let val = BuiltInTypes::untag(*field) as i8;
                    buffer.push(val as *mut i8);
                }
                FFIType::I16 => {
                    let val = BuiltInTypes::untag(*field) as i16;
                    buffer.push(val as *mut i8);
                }
                FFIType::I32 => {
                    let val = BuiltInTypes::untag(*field) as i32;
                    buffer.push(val as *mut i8);
                }
                FFIType::I64 => {
                    let val = BuiltInTypes::untag(*field) as i64;
                    buffer.push(val as *mut i8);
                }
                FFIType::F32 => {
                    throw_runtime_error(
                        stack_pointer,
                        "FFIError",
                        "FFI arrays of f32 not yet implemented".to_string(),
                    );
                }
                FFIType::F64 => {
                    throw_runtime_error(
                        stack_pointer,
                        "FFIError",
                        "FFI arrays of f64 not yet implemented".to_string(),
                    );
                }
                FFIType::Pointer => {
                    // Treat as raw pointer value
                    let val = BuiltInTypes::untag(*field);
                    buffer.push(val as *mut i8);
                }
                FFIType::MutablePointer => {
                    // Treat as raw pointer value
                    let val = BuiltInTypes::untag(*field);
                    buffer.push(val as *mut i8);
                }
                FFIType::Structure(_) => {
                    throw_runtime_error(
                        stack_pointer,
                        "FFIError",
                        "FFI arrays of structures not yet implemented".to_string(),
                    );
                }
                FFIType::String => {
                    let string = runtime.get_string(stack_pointer, *field);
                    let string_pointer = runtime.memory.write_c_string(string);
                    buffer.push(string_pointer);
                }
                FFIType::Void => {
                    throw_runtime_error(
                        stack_pointer,
                        "FFIError",
                        "Cannot create array of void type".to_string(),
                    );
                }
            }
        }
        // null terminate array
        buffer.push(std::ptr::null_mut());

        // For now we are intentionally leaking memory

        let buffer_ptr: *mut c_void = buffer.as_mut_ptr() as *mut c_void;
        std::mem::forget(buffer);
        let raw = buffer_ptr as u64;
        let lo_tagged = BuiltInTypes::Int.tag((raw & 0xFFFFFFFF) as isize) as usize;
        let hi_tagged = BuiltInTypes::Int.tag(((raw >> 32) & 0xFFFFFFFF) as isize) as usize;
        call_fn_2(
            runtime,
            "beagle.ffi/__make_pointer_struct",
            lo_tagged,
            hi_tagged,
        )
    }
}

// Copy bytes between FFI buffers with offsets
// ffi_copy_bytes(src, src_off, dst, dst_off, len) -> null
pub unsafe extern "C" fn ffi_copy_bytes(
    src: usize,
    src_off: usize,
    dst: usize,
    dst_off: usize,
    len: usize,
) -> usize {
    unsafe {
        let src_ptr = extract_raw_ptr(src) as *const u8;
        let src_off = BuiltInTypes::untag(src_off);

        let dst_ptr = extract_raw_ptr(dst);
        let dst_off = BuiltInTypes::untag(dst_off);

        let len = BuiltInTypes::untag(len);

        std::ptr::copy_nonoverlapping(src_ptr.add(src_off), dst_ptr.add(dst_off), len);

        BuiltInTypes::null_value() as usize
    }
}

// Reallocate an FFI buffer to a new size
// ffi_realloc(buffer, new_size) -> new_buffer
pub unsafe extern "C" fn ffi_realloc(buffer: usize, new_size: usize) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        let buffer_object = HeapObject::from_tagged(buffer);
        let old_ptr = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let old_size = BuiltInTypes::untag(buffer_object.get_field(1));
        let new_size_val = BuiltInTypes::untag(new_size);

        // Create new buffer
        let mut new_buffer: Vec<u8> = vec![0; new_size_val];
        let new_ptr: *mut u8 = new_buffer.as_mut_ptr();

        // Copy old data
        let copy_len = std::cmp::min(old_size, new_size_val);
        std::ptr::copy_nonoverlapping(old_ptr, new_ptr, copy_len);

        // Free old buffer
        let _old_buffer = Vec::from_raw_parts(old_ptr, old_size, old_size);

        // Forget new buffer (we're taking ownership)
        std::mem::forget(new_buffer);

        let new_ptr_tagged = BuiltInTypes::Int.tag(new_ptr as isize) as usize;
        let new_size_tagged = BuiltInTypes::Int.tag(new_size_val as isize) as usize;
        call_fn_2(
            runtime,
            "beagle.ffi/__make_buffer_struct",
            new_ptr_tagged,
            new_size_tagged,
        )
    }
}

// Get the size of an FFI buffer
// ffi_buffer_size(buffer) -> size
pub unsafe extern "C" fn ffi_buffer_size(buffer: usize) -> usize {
    let buffer_object = HeapObject::from_tagged(buffer);
    let size = BuiltInTypes::untag(buffer_object.get_field(1));
    BuiltInTypes::Int.tag(size as isize) as usize
}

// Write from buffer at offset to a file descriptor
// ffi_write_buffer_offset(fd, buffer, offset, len) -> bytes_written
pub unsafe extern "C" fn ffi_write_buffer_offset(
    fd: usize,
    buffer: usize,
    offset: usize,
    len: usize,
) -> usize {
    unsafe {
        let fd = BuiltInTypes::untag(fd) as i32;
        let buffer_ptr = extract_raw_ptr(buffer) as *const u8;
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);

        let result = libc::write(fd, buffer_ptr.add(offset) as *const libc::c_void, len);

        BuiltInTypes::Int.tag(result as isize) as usize
    }
}

// Translate bytes in buffer using a 256-byte lookup table
// ffi_translate_bytes(buffer, offset, len, table) -> null
pub unsafe extern "C" fn ffi_translate_bytes(
    buffer: usize,
    offset: usize,
    len: usize,
    table: usize,
) -> usize {
    unsafe {
        let buffer_ptr = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);

        let table_ptr = extract_raw_ptr(table) as *const u8;

        for i in 0..len {
            let byte = *buffer_ptr.add(offset + i);
            let translated = *table_ptr.add(byte as usize);
            *buffer_ptr.add(offset + i) = translated;
        }

        BuiltInTypes::null_value() as usize
    }
}

// Reverse bytes in buffer in place
// ffi_reverse_bytes(buffer, offset, len) -> null
pub unsafe extern "C" fn ffi_reverse_bytes(buffer: usize, offset: usize, len: usize) -> usize {
    unsafe {
        let buffer_ptr = extract_raw_ptr(buffer);
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);

        let mut left = 0;
        let mut right = len.saturating_sub(1);

        while left < right {
            let tmp = *buffer_ptr.add(offset + left);
            *buffer_ptr.add(offset + left) = *buffer_ptr.add(offset + right);
            *buffer_ptr.add(offset + right) = tmp;
            left += 1;
            right -= 1;
        }

        BuiltInTypes::null_value() as usize
    }
}

// Find first occurrence of a byte in buffer (like memchr)
// ffi_find_byte(buffer, offset, len, byte) -> index or -1
pub unsafe extern "C" fn ffi_find_byte(
    buffer: usize,
    offset: usize,
    len: usize,
    byte: usize,
) -> usize {
    unsafe {
        let buffer_ptr = extract_raw_ptr(buffer) as *const u8;
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);
        let byte = BuiltInTypes::untag(byte) as u8;

        let slice = std::slice::from_raw_parts(buffer_ptr.add(offset), len);
        match slice.iter().position(|&b| b == byte) {
            Some(pos) => BuiltInTypes::Int.tag((offset + pos) as isize) as usize,
            None => BuiltInTypes::Int.tag(-1) as usize,
        }
    }
}

// Copy bytes from src to dst, skipping instances of skip_byte
// Returns number of bytes written to dst
// ffi_copy_bytes_filter(src, src_off, dst, dst_off, len, skip_byte) -> bytes_written
pub unsafe extern "C" fn ffi_copy_bytes_filter(
    src: usize,
    src_off: usize,
    dst: usize,
    dst_off: usize,
    len: usize,
    skip_byte: usize,
) -> usize {
    unsafe {
        let src_ptr = extract_raw_ptr(src) as *const u8;
        let src_off = BuiltInTypes::untag(src_off);

        let dst_ptr = extract_raw_ptr(dst);
        let dst_off = BuiltInTypes::untag(dst_off);

        let len = BuiltInTypes::untag(len);
        let skip_byte = BuiltInTypes::untag(skip_byte) as u8;

        let mut written = 0;
        for i in 0..len {
            let byte = *src_ptr.add(src_off + i);
            if byte != skip_byte {
                *dst_ptr.add(dst_off + written) = byte;
                written += 1;
            }
        }

        BuiltInTypes::Int.tag(written as isize) as usize
    }
}
