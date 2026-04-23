use super::*;
use crate::save_gc_context;

/// Maximum arity we dispatch to JIT shims with. The shim we dispatch to
/// has a narrower declared signature, but the Rust→Beagle ABI lets the
/// caller pass extra trailing params the callee ignores: register args
/// beyond what the callee reads are noise in caller-owned registers,
/// and stack args beyond the callee's declared count sit in
/// caller-owned stack slots the callee never touches. This lets us use
/// ONE `transmute + call` site instead of a 12-arm match per call-kind.
///
/// The current max is dictated by `call_beagle_via_apply_call`
/// (fn_ptr + up to 11 target args) and the variadic / closure
/// trampolines below (fn_ptr + closure + up to 9 target args).
const SHIM_MAX_PARAMS: usize = 12;

/// Universal-arity dispatch to a JIT shim. `params` holds the exact
/// values the shim's declared signature expects, in order; unused
/// trailing slots must be zero (the shim won't read them, but keeping
/// them well-defined makes this function safe to inspect in a
/// debugger).
///
/// Replaces three parallel 10–12-arm transmute matches that previously
/// lived here. The bug that motivated the shim refactor —
/// `apply_call_N` missing its callee-saved save — was partly a
/// consequence of those matches hand-rolling the same pattern in three
/// places; reducing this to one call site removes the drift vector on
/// the Rust side as well.
#[inline(always)]
unsafe fn call_shim_padded(shim_ptr: usize, params: [usize; SHIM_MAX_PARAMS]) -> usize {
    type ShimFn = extern "C" fn(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) -> usize;
    let f: ShimFn = unsafe { std::mem::transmute(shim_ptr) };
    f(
        params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
        params[8], params[9], params[10], params[11],
    )
}

/// Call a JIT-compiled Beagle function from Rust, routing through the
/// `beagle.builtin/apply_call_N` shim trampoline so the call is a proper
/// Rust→Beagle boundary.
///
/// Why: Beagle function prologues assume X28 holds the current thread's
/// MutatorState pointer (inline `gc_frame_link` addresses `[x28, #16]`).
/// When Rust calls a Beagle function via `transmute + direct call`, X28
/// is whatever the Rust compiler happened to leave there — X28 is AAPCS
/// callee-saved so Rust is free to use it as a scratch inside the
/// function. The `apply_call_N` shim reloads X28 from
/// `jit_load_current_mutator_state` at entry, so calling *it* from Rust
/// is safe regardless of Rust's register usage.
///
/// `fn_ptr` is the untagged machine-code address of the target Beagle
/// function. `args` is the argument list the target should receive in
/// X0..X(N-1) (the shim shuffles our argument registers into place).
unsafe fn call_beagle_via_apply_call(runtime: &Runtime, fn_ptr: usize, args: &[usize]) -> usize {
    assert!(
        args.len() + 1 <= SHIM_MAX_PARAMS,
        "call_beagle_via_apply_call: {} args exceeds SHIM_MAX_PARAMS-1 = {}",
        args.len(),
        SHIM_MAX_PARAMS - 1
    );

    let shim_name = format!("beagle.builtin/apply_call_{}", args.len());
    let shim_entry = runtime
        .get_function_by_name(&shim_name)
        .unwrap_or_else(|| panic!("{} trampoline not compiled", shim_name));
    let shim_ptr = runtime
        .get_pointer(shim_entry)
        .unwrap_or_else(|_| panic!("{} has no code pointer", shim_name))
        as usize;

    // Shim signature: apply_call_N(fn_ptr, arg0, ..., argN-1).
    let mut params = [0usize; SHIM_MAX_PARAMS];
    params[0] = fn_ptr;
    for (i, a) in args.iter().enumerate() {
        params[i + 1] = *a;
    }
    unsafe { call_shim_padded(shim_ptr, params) }
}

pub unsafe extern "C" fn throw_error(stack_pointer: usize, frame_pointer: usize) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);
    print_stack(stack_pointer);
    std::process::exit(1);
}

pub unsafe extern "C" fn throw_type_error(stack_pointer: usize, frame_pointer: usize) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);

    let (kind_str, message_str) = {
        let runtime = get_runtime().get_mut();
        let kind = runtime
            .allocate_string(stack_pointer, "TypeError".to_string())
            .expect("Failed to allocate kind string")
            .into();
        let kind_root_id = runtime.register_temporary_root(kind);
        let msg = runtime
            .allocate_string(
                stack_pointer,
                "Type mismatch in arithmetic operation. To mix integers and floats, use to-float() to convert integers: e.g., 3.14 * to-float(2)".to_string(),
            )
            .expect("Failed to allocate message string")
            .into();
        let kind = runtime.unregister_temporary_root(kind_root_id);
        (kind, msg)
    };

    let null_location = BuiltInTypes::Null.tag(0) as usize;
    let resume_address = get_saved_gc_return_addr();
    unsafe {
        let error = create_error(
            stack_pointer,
            frame_pointer,
            kind_str,
            message_str,
            null_location,
        );
        throw_exception(stack_pointer, frame_pointer, error, resume_address, 0);
    }
}

pub unsafe extern "C" fn check_arity(
    stack_pointer: usize,
    frame_pointer: usize,
    function_pointer: usize,
    expected_args: isize,
) -> isize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get();

    // Function pointer is tagged, need to untag
    let untagged_ptr = (function_pointer >> BuiltInTypes::tag_size()) as *const u8;

    if let Some(function) = runtime.get_function_by_pointer(untagged_ptr) {
        // expected_args is a tagged integer, need to untag it
        let expected_args_untagged = BuiltInTypes::untag(expected_args as usize);

        if function.is_variadic {
            // For variadic functions, we need at least min_args arguments
            if expected_args_untagged < function.min_args {
                println!(
                    "Arity mismatch for variadic '{}': expected at least {} args, got {}",
                    function.name, function.min_args, expected_args_untagged
                );
                unsafe {
                    throw_error(stack_pointer, frame_pointer);
                }
            }
        } else {
            // For non-variadic functions, exact match required
            if function.number_of_args != expected_args_untagged {
                println!(
                    "Arity mismatch for '{}': expected {} args, got {}",
                    function.name, function.number_of_args, expected_args_untagged
                );
                unsafe {
                    throw_error(stack_pointer, frame_pointer);
                }
            }
        }
    }

    0 // Return value unused
}

/// Check if a function is variadic. Returns tagged boolean.
pub unsafe extern "C" fn is_function_variadic(function_pointer: usize) -> usize {
    let runtime = get_runtime().get();

    // Function pointer is tagged, need to untag
    let untagged_ptr = (function_pointer >> BuiltInTypes::tag_size()) as *const u8;

    if let Some(function) = runtime.get_function_by_pointer(untagged_ptr) {
        if function.is_variadic {
            BuiltInTypes::true_value() as usize
        } else {
            BuiltInTypes::false_value() as usize
        }
    } else {
        BuiltInTypes::false_value() as usize
    }
}

/// Get the min_args for a variadic function. Returns tagged int.
pub unsafe extern "C" fn get_function_min_args(function_pointer: usize) -> usize {
    let runtime = get_runtime().get();

    // Function pointer is tagged, need to untag
    let untagged_ptr = (function_pointer >> BuiltInTypes::tag_size()) as *const u8;

    if let Some(function) = runtime.get_function_by_pointer(untagged_ptr) {
        BuiltInTypes::Int.tag(function.min_args as isize) as usize
    } else {
        BuiltInTypes::Int.tag(0) as usize
    }
}

/// Build rest array from saved locals.
/// Used by variadic function prologues to build the rest parameter array
/// from arguments that were saved to dedicated local slots.
///
/// Arguments:
/// - stack_pointer: for GC
/// - frame_pointer: used to compute local addresses
/// - arg_count: total args passed including closure env (tagged int, from X9)
/// - min_args: named params count, NOT including closure env (tagged int)
/// - first_local_index: index of first saved arg local (untagged raw value)
/// - first_arg_index: 0 for top-level, 1 for closures (tagged int)
///
/// Returns: tagged array pointer containing the rest args
pub unsafe extern "C" fn build_rest_array_from_locals(
    stack_pointer: usize,
    frame_pointer: usize,
    arg_count: usize,
    min_args: usize,
    first_local_index: usize,
    first_arg_index: usize,
) -> usize {
    #[cfg(debug_assertions)]
    {
        if std::env::var("BEAGLE_DEBUG_RESUME").is_ok() {
            eprintln!(
                "[build_rest_array] sp={:#x} fp={:#x} arg_count={:#x} min_args={:#x} first_local={} first_arg={:#x}",
                stack_pointer,
                frame_pointer,
                arg_count,
                min_args,
                first_local_index,
                first_arg_index
            );
        }
    }
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let total = BuiltInTypes::untag(arg_count);
    let min = BuiltInTypes::untag(min_args);
    let first_arg = BuiltInTypes::untag(first_arg_index);

    // Number of user-visible args (excluding closure env)
    let total_user_args = total.saturating_sub(first_arg);
    // Number of rest args
    let num_extra = total_user_args.saturating_sub(min);

    if num_extra == 0 {
        // Return empty array
        let array_ptr = match runtime.allocate_zeroed(0, stack_pointer, BuiltInTypes::HeapObject) {
            Ok(ptr) => ptr,
            Err(_) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate rest array - out of memory".to_string(),
                );
            },
        };
        let mut heap_obj = HeapObject::from_tagged(array_ptr);
        heap_obj.write_type_id(1);
        return array_ptr;
    }

    // Allocate array for extra args (zeroed)
    let array_ptr =
        match runtime.allocate_zeroed(num_extra, stack_pointer, BuiltInTypes::HeapObject) {
            Ok(ptr) => ptr,
            Err(_) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate rest array - out of memory".to_string(),
                );
            },
        };

    // Set type_id to 1 (raw array)
    let mut heap_obj = HeapObject::from_tagged(array_ptr);
    heap_obj.write_type_id(1);

    let array_data = heap_obj.untagged() as *mut usize;

    // Read args from saved locals (register args) or the stack (overflow args).
    //
    // saved_arg_locals layout (allocated in AST for indices first_arg..8):
    //   saved_arg_locals[k] at local (first_local_index + k) = arg register (first_arg + k)
    //
    // We want user arg indices min..min+num_extra (0-based within user args).
    // The actual register index for user_arg_idx is: first_arg + user_arg_idx.
    // The saved local offset for that register is: user_arg_idx (since locals start at first_arg).
    const NUM_ARG_REGISTERS: usize = 8; // ARM64 uses X0-X7

    for i in 0..num_extra {
        let user_arg_idx = min + i; // 0-based user arg index
        let actual_register = first_arg + user_arg_idx; // actual arg register index

        let arg = if actual_register < NUM_ARG_REGISTERS {
            // Register arg: read from saved local
            let local_index = first_local_index + user_arg_idx;
            // +3 because locals start at FP-24: FP-8 = frame header, FP-16 = prev pointer
            let local_addr = frame_pointer.wrapping_sub((local_index + 3) * 8);
            unsafe { *(local_addr as *const usize) }
        } else {
            // Stack arg: read from caller's stack frame
            // Stack args are at [FP + (actual_register - 8 + 2) * 8]
            let stack_offset = (actual_register - NUM_ARG_REGISTERS + 2) * 8;
            let stack_addr = frame_pointer.wrapping_add(stack_offset);
            unsafe { *(stack_addr as *const usize) }
        };

        // Write to array field i (offset 1 to skip header)
        unsafe { *array_data.add(i + 1) = arg };
    }

    array_ptr
}

/// Pack variadic arguments from stack into an array.
/// stack_pointer: current stack pointer
/// args_base: pointer to first arg on stack (args are contiguous)
/// total_args: total number of args (tagged int)
/// min_args: minimum args before rest param (tagged int)
/// Returns: tagged array pointer containing args[min_args..total_args]
pub unsafe extern "C" fn pack_variadic_args_from_stack(
    stack_pointer: usize,
    frame_pointer: usize,
    args_base: usize,
    total_args: usize,
    min_args: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let total = BuiltInTypes::untag(total_args);
    let min = BuiltInTypes::untag(min_args);

    // Number of extra args to pack
    let num_extra = total.saturating_sub(min);

    // Allocate array for extra args (zeroed)
    let array_ptr =
        match runtime.allocate_zeroed(num_extra, stack_pointer, BuiltInTypes::HeapObject) {
            Ok(ptr) => ptr,
            Err(_) => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate variadic args array - out of memory".to_string(),
                );
            },
        };

    // Set type_id to 1 (raw array)
    let mut heap_obj = HeapObject::from_tagged(array_ptr);
    heap_obj.write_type_id(1);

    // Copy extra args from stack to array
    // Args on stack are at args_base, args_base+8, args_base+16, etc.
    let args_ptr = args_base as *const usize;
    let array_data = heap_obj.untagged() as *mut usize;

    for i in 0..num_extra {
        // Read arg at index (min + i) from args on stack
        // SAFETY: args_ptr and array_data are valid pointers set up by the caller,
        // and indices are bounds-checked via num_extra calculation
        unsafe {
            let arg = *args_ptr.add(min + i);
            // Write to array field i (offset 1 to skip header)
            *array_data.add(i + 1) = arg;
        }
    }

    array_ptr
}

/// Call a variadic function through a function value.
/// This handles the min_args dispatch at runtime, avoiding complex IR branching
/// that confuses the register allocator.
///
/// Arguments:
/// - stack_pointer: For GC safety during allocation
/// - frame_pointer: For GC stack walking
/// - function_ptr: Tagged function pointer
/// - args_array_ptr: Tagged pointer to array containing all call arguments
/// - is_closure: Tagged boolean - true if calling through a closure
/// - closure_ptr: Tagged closure pointer (or null if not a closure)
///
/// Returns: The function's return value
pub unsafe extern "C" fn call_variadic_function_value(
    stack_pointer: usize,
    frame_pointer: usize,
    function_ptr: usize,
    args_array_ptr: usize,
    is_closure: usize,
    closure_ptr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get();

    // Get function metadata
    let untagged_fn = (function_ptr >> BuiltInTypes::tag_size()) as *const u8;
    let function = match runtime.get_function_by_pointer(untagged_fn) {
        Some(f) => f,
        None => unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                "Invalid function pointer in call_variadic_function_value".to_string(),
            );
        },
    };
    let min_args = function.min_args;
    let is_closure_bool = is_closure == BuiltInTypes::true_value() as usize;

    // Get arg count before allocation (allocation can trigger GC)
    let args_heap = HeapObject::from_tagged(args_array_ptr);
    let total_args = args_heap.get_fields().len();

    // Build rest array for args[min_args..]
    let num_extra = total_args.saturating_sub(min_args);

    // Register args_array as a temporary root before allocation
    let runtime_mut = get_runtime().get_mut();
    let args_root_id = runtime_mut.register_temporary_root(args_array_ptr);

    // Allocate array for extra args (can trigger GC, zeroed)
    let rest_array =
        match runtime_mut.allocate_zeroed(num_extra, stack_pointer, BuiltInTypes::HeapObject) {
            Ok(ptr) => ptr,
            Err(_) => {
                runtime_mut.unregister_temporary_root(args_root_id);
                unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "AllocationError",
                        "Failed to allocate rest array - out of memory".to_string(),
                    );
                }
            }
        };

    // Get the updated args_array_ptr from the root (GC may have moved it)
    // then unregister the root
    let args_array_ptr = runtime_mut.unregister_temporary_root(args_root_id);

    // Set type_id to 1 (raw array)
    let mut rest_heap = HeapObject::from_tagged(rest_array);
    rest_heap.write_type_id(1);

    // Now get the args fields - using the updated pointer after potential GC
    let args_heap = HeapObject::from_tagged(args_array_ptr);
    let all_args = args_heap.get_fields();

    // Copy extra args into rest array
    let rest_fields = rest_heap.get_fields_mut();
    rest_fields[..num_extra].copy_from_slice(&all_args[min_args..(num_extra + min_args)]);

    // Dispatch: call the target with (optionally closure_ptr,)
    // all_args[0..min_args], rest_array. The apply_call_N shim loads X28,
    // tags the arg count in X9, and shuffles argument registers, so this
    // Rust→Beagle boundary doesn't depend on Rust preserving X28.
    if min_args > 7 {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "ArgumentError",
                format!(
                    "Unsupported min_args value {} for variadic function call (max supported: 7). \
                    Consider reducing the number of required parameters.",
                    min_args
                ),
            );
        }
    }
    let mut call_args: Vec<usize> = Vec::with_capacity(min_args + 2);
    if is_closure_bool {
        call_args.push(closure_ptr);
    }
    call_args.extend_from_slice(&all_args[..min_args]);
    call_args.push(rest_array);
    unsafe { call_beagle_via_apply_call(runtime, untagged_fn as usize, &call_args) }
}

/// Apply a function to an array of arguments.
/// Signature: (stack_pointer, frame_pointer, function, args_array) -> result
///
/// Supports:
/// - Regular functions (tagged as Function)
/// - Closures (tagged as Closure) - closure prepended as first arg
/// - Multi-arity functions (HeapObject with type_id=29) - dispatches to correct arity
/// - Variadic functions - bundles extra args into rest array
///
/// Args 0-10: Direct transmute calls
pub unsafe extern "C" fn apply_function(
    stack_pointer: usize,
    frame_pointer: usize,
    function: usize,
    args_array: usize,
) -> usize {
    use crate::collections::{GcHandle, PersistentVec};

    save_gc_context!(stack_pointer, frame_pointer);

    let runtime = get_runtime().get();

    // Determine the function type from the tag
    let tag = BuiltInTypes::get_kind(function);

    // Get arguments from the array
    // Could be a HeapObject (raw array, type_id=1) or PersistentVec (type_id=20)
    let (args, arg_count): (Vec<usize>, usize) =
        if BuiltInTypes::get_kind(args_array) == BuiltInTypes::HeapObject {
            let heap_obj = HeapObject::from_tagged(args_array);
            let type_id = heap_obj.get_type_id();
            if type_id == 20 {
                // PersistentVec
                let vec_handle = GcHandle::from_tagged(args_array);
                let count = PersistentVec::count(vec_handle);
                let mut v = Vec::with_capacity(count);
                for i in 0..count {
                    v.push(PersistentVec::get(vec_handle, i));
                }
                (v, count)
            } else {
                // Raw array (type_id=1) or other heap object with fields
                let fields = heap_obj.get_fields();
                (fields.to_vec(), fields.len())
            }
        } else {
            unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "TypeError",
                    format!(
                        "apply: expected array or vector as second argument, got {:?}",
                        BuiltInTypes::get_kind(args_array)
                    ),
                );
            }
        };

    unsafe {
        match tag {
            BuiltInTypes::Function => {
                // Regular function - look up metadata to check if variadic
                let fn_ptr = (function >> BuiltInTypes::tag_size()) as *const u8;

                if let Some(func_info) = runtime.get_function_by_pointer(fn_ptr) {
                    if func_info.is_variadic {
                        // Variadic function - bundle extra args into rest array
                        call_variadic_with_args(
                            stack_pointer,
                            fn_ptr,
                            &args,
                            arg_count,
                            func_info.min_args,
                            false,
                            0,
                        )
                    } else {
                        if arg_count != func_info.number_of_args {
                            throw_runtime_error(
                                stack_pointer,
                                "ArityError",
                                format!(
                                    "apply: function '{}' expects {} arguments, but got {}",
                                    func_info.name, func_info.number_of_args, arg_count
                                ),
                            );
                        }
                        call_with_args(fn_ptr, &args, arg_count, false, 0)
                    }
                } else {
                    // Unknown function, try direct call
                    call_with_args(fn_ptr, &args, arg_count, false, 0)
                }
            }
            BuiltInTypes::Closure => {
                // Closure - extract function pointer, prepend closure as first arg
                let heap_obj = HeapObject::from_tagged(function);
                let fn_ptr_tagged = heap_obj.get_field(0);
                let fn_ptr = BuiltInTypes::untag(fn_ptr_tagged) as *const u8;

                if let Some(func_info) = runtime.get_function_by_pointer(fn_ptr) {
                    if func_info.is_variadic {
                        // Variadic closure - bundle extra args into rest array
                        call_variadic_with_args(
                            stack_pointer,
                            fn_ptr,
                            &args,
                            arg_count,
                            func_info.min_args,
                            true,
                            function,
                        )
                    } else {
                        if arg_count != func_info.number_of_args {
                            throw_runtime_error(
                                stack_pointer,
                                "ArityError",
                                format!(
                                    "apply: closure '{}' expects {} arguments, but got {}",
                                    func_info.name, func_info.number_of_args, arg_count
                                ),
                            );
                        }
                        call_with_args(fn_ptr, &args, arg_count, true, function)
                    }
                } else {
                    call_with_args(fn_ptr, &args, arg_count, true, function)
                }
            }
            BuiltInTypes::HeapObject => {
                // Could be MultiArityFunction or FunctionObject
                let heap_obj = HeapObject::from_tagged(function);
                let type_id = heap_obj.get_type_id();

                if type_id == TYPE_ID_MULTI_ARITY_FUNCTION as usize {
                    // MultiArityFunction - dispatch to correct arity
                    let tagged_arg_count = BuiltInTypes::construct_int(arg_count as isize) as usize;
                    let fn_ptr_tagged = dispatch_multi_arity(
                        stack_pointer,
                        frame_pointer,
                        function,
                        tagged_arg_count,
                    );

                    // The returned fn_ptr is tagged as Function
                    let fn_ptr = (fn_ptr_tagged >> BuiltInTypes::tag_size()) as *const u8;

                    // Check if the dispatched function is variadic
                    if let Some(func_info) = runtime.get_function_by_pointer(fn_ptr) {
                        if func_info.is_variadic {
                            call_variadic_with_args(
                                stack_pointer,
                                fn_ptr,
                                &args,
                                arg_count,
                                func_info.min_args,
                                false,
                                0,
                            )
                        } else {
                            call_with_args(fn_ptr, &args, arg_count, false, 0)
                        }
                    } else {
                        call_with_args(fn_ptr, &args, arg_count, false, 0)
                    }
                } else if type_id == 10 {
                    // FunctionObject (TYPE_ID_FUNCTION_OBJECT = 10)
                    let fn_ptr_tagged = heap_obj.get_field(0);
                    let fn_ptr = BuiltInTypes::untag(fn_ptr_tagged) as *const u8;

                    if let Some(func_info) = runtime.get_function_by_pointer(fn_ptr) {
                        if func_info.is_variadic {
                            call_variadic_with_args(
                                stack_pointer,
                                fn_ptr,
                                &args,
                                arg_count,
                                func_info.min_args,
                                false,
                                0,
                            )
                        } else {
                            call_with_args(fn_ptr, &args, arg_count, false, 0)
                        }
                    } else {
                        call_with_args(fn_ptr, &args, arg_count, false, 0)
                    }
                } else if type_id == 0 {
                    // Could be a Function struct object
                    let struct_id = heap_obj.get_struct_id();
                    if struct_id == runtime.function_struct_id {
                        let fn_ptr_tagged = heap_obj.get_field(0);
                        let fn_ptr = BuiltInTypes::untag(fn_ptr_tagged) as *const u8;

                        if let Some(func_info) = runtime.get_function_by_pointer(fn_ptr) {
                            if func_info.is_variadic {
                                call_variadic_with_args(
                                    stack_pointer,
                                    fn_ptr,
                                    &args,
                                    arg_count,
                                    func_info.min_args,
                                    false,
                                    0,
                                )
                            } else {
                                call_with_args(fn_ptr, &args, arg_count, false, 0)
                            }
                        } else {
                            call_with_args(fn_ptr, &args, arg_count, false, 0)
                        }
                    } else {
                        throw_runtime_error(
                            stack_pointer,
                            "TypeError",
                            format!(
                                "apply: expected function, closure, or multi-arity function, got HeapObject with type_id={}",
                                type_id
                            ),
                        );
                    }
                } else {
                    throw_runtime_error(
                        stack_pointer,
                        "TypeError",
                        format!(
                            "apply: expected function, closure, or multi-arity function, got HeapObject with type_id={}",
                            type_id
                        ),
                    );
                }
            }
            _ => {
                throw_runtime_error(
                    stack_pointer,
                    "TypeError",
                    format!(
                        "apply: expected function, closure, or multi-arity function, got {:?}",
                        tag
                    ),
                );
            }
        }
    }
}

/// Helper function to call a variadic function with individual args.
/// Uses JIT-compiled trampolines that set X9 (ARM64) or R10 (x86-64) to the arg count.
#[inline(always)]
pub unsafe fn call_variadic_with_args(
    stack_pointer: usize,
    fn_ptr: *const u8,
    args: &[usize],
    arg_count: usize,
    _min_args: usize,
    is_closure: bool,
    closure_ptr: usize,
) -> usize {
    // The variadic calling convention:
    // 1. Pass all args individually in registers X0-X7 (ARM64) or RDI, RSI, etc (x86-64)
    // 2. Set X9 (ARM64) or R10 (x86-64) to the arg count (tagged)
    // 3. The function prologue reads X9 and builds the rest array from the args
    //
    // For closures, the closure pointer is prepended as arg0.
    //
    // We use JIT-compiled trampolines (apply_call_N) that handle the X9 setup.
    // Trampoline signature: apply_call_N(fn_ptr, arg0, arg1, ..., argN-1) -> result

    let effective_arg_count = if is_closure { arg_count + 1 } else { arg_count };

    // Look up the appropriate trampoline
    let trampoline_name = format!("beagle.builtin/apply_call_{}", effective_arg_count);
    let runtime = get_runtime().get();
    let trampoline = match runtime.get_function_by_name(&trampoline_name) {
        Some(t) => t,
        None => unsafe {
            throw_runtime_error(
                stack_pointer,
                "RuntimeError",
                format!(
                    "apply: cannot call function with {} arguments (trampoline {} not found)",
                    effective_arg_count, trampoline_name
                ),
            );
        },
    };
    let trampoline_ptr = usize::from(trampoline.pointer) as *const u8;

    // Build args array: [fn_ptr, arg0, arg1, ...]
    // For closures: [fn_ptr, closure_ptr, arg0, arg1, ...]
    unsafe {
        if is_closure {
            call_trampoline_with_closure(trampoline_ptr, fn_ptr, closure_ptr, args, arg_count)
        } else {
            call_trampoline(trampoline_ptr, fn_ptr, args, arg_count)
        }
    }
}

/// Call a trampoline with the given function pointer and args.
/// The trampoline expects: (fn_ptr, arg0, arg1, ..., argN-1)
#[inline(always)]
pub unsafe fn call_trampoline(
    trampoline_ptr: *const u8,
    fn_ptr: *const u8,
    args: &[usize],
    arg_count: usize,
) -> usize {
    if arg_count + 1 > SHIM_MAX_PARAMS {
        let sp = get_saved_stack_pointer();
        unsafe {
            throw_runtime_error(
                sp,
                "ArgumentError",
                format!(
                    "apply: too many arguments ({}), max supported is {}",
                    arg_count,
                    SHIM_MAX_PARAMS - 1
                ),
            );
        }
    }
    let mut params = [0usize; SHIM_MAX_PARAMS];
    params[0] = fn_ptr as usize;
    for (i, a) in args.iter().take(arg_count).enumerate() {
        params[i + 1] = *a;
    }
    unsafe { call_shim_padded(trampoline_ptr as usize, params) }
}

/// Call a trampoline with closure pointer prepended.
/// The trampoline expects: (fn_ptr, closure_ptr, arg0, arg1, ..., argN-1)
#[inline(always)]
pub unsafe fn call_trampoline_with_closure(
    trampoline_ptr: *const u8,
    fn_ptr: *const u8,
    closure_ptr: usize,
    args: &[usize],
    arg_count: usize,
) -> usize {
    if arg_count + 2 > SHIM_MAX_PARAMS {
        let sp = get_saved_stack_pointer();
        unsafe {
            throw_runtime_error(
                sp,
                "ArgumentError",
                format!(
                    "apply: too many arguments ({}) for closure, max supported is {}",
                    arg_count,
                    SHIM_MAX_PARAMS - 2
                ),
            );
        }
    }
    let mut params = [0usize; SHIM_MAX_PARAMS];
    params[0] = fn_ptr as usize;
    params[1] = closure_ptr;
    for (i, a) in args.iter().take(arg_count).enumerate() {
        params[i + 2] = *a;
    }
    unsafe { call_shim_padded(trampoline_ptr as usize, params) }
}

/// Helper function to call a function pointer with the given arguments.
/// Handles both regular functions and closures (closure_ptr prepended as first arg).
///
/// Routes the call through the `apply_call_N` shim (which loads X28 from
/// the current MutatorState on entry), so Rust is not required to have
/// preserved X28 by the time this is invoked.
#[inline(always)]
pub unsafe fn call_with_args(
    fn_ptr: *const u8,
    args: &[usize],
    arg_count: usize,
    is_closure: bool,
    closure_ptr: usize,
) -> usize {
    if arg_count > 10 {
        let sp = get_saved_stack_pointer();
        unsafe {
            throw_runtime_error(
                sp,
                "ArgumentError",
                format!(
                    "apply: too many arguments ({}), max supported is 10",
                    arg_count
                ),
            );
        }
    }
    let runtime = get_runtime().get();
    let mut call_args: Vec<usize> = Vec::with_capacity(arg_count + 1);
    if is_closure {
        call_args.push(closure_ptr);
    }
    call_args.extend_from_slice(&args[..arg_count]);
    unsafe { call_beagle_via_apply_call(runtime, fn_ptr as usize, &call_args) }
}

pub unsafe fn call_fn_0(runtime: &Runtime, function_name: &str) -> usize {
    print_call_builtin(
        runtime,
        format!("{} {}", "call_fn_0", function_name).as_str(),
    );

    // Save GC context before calling into Beagle - the Beagle code may call builtins
    // that update the saved GC context, making it point to now-deallocated frames
    let saved_ctx = save_current_gc_context();

    let save_volatile_registers = runtime
        .get_function_by_name("beagle.builtin/save_volatile_registers0")
        .unwrap();
    let save_volatile_registers = runtime.get_pointer(save_volatile_registers).unwrap();
    let save_volatile_registers: fn(usize) -> usize =
        unsafe { std::mem::transmute(save_volatile_registers) };

    let function = runtime.get_function_by_name(function_name).unwrap();
    let function = runtime.get_pointer(function).unwrap();
    let result = save_volatile_registers(function as usize);

    // Restore GC context after Beagle call returns
    restore_gc_context(saved_ctx);
    result
}

pub unsafe fn call_fn_1(runtime: &Runtime, function_name: &str, arg1: usize) -> usize {
    print_call_builtin(
        runtime,
        format!("{} {}", "call_fn_1", function_name).as_str(),
    );

    // Save GC context before calling into Beagle - the Beagle code may call builtins
    // that update the saved GC context, making it point to now-deallocated frames
    let saved_ctx = save_current_gc_context();

    let save_volatile_registers = runtime
        .get_function_by_name("beagle.builtin/save_volatile_registers1")
        .unwrap();
    let save_volatile_registers = runtime.get_pointer(save_volatile_registers).unwrap();
    let save_volatile_registers: fn(usize, usize) -> usize =
        unsafe { std::mem::transmute(save_volatile_registers) };

    let function = runtime.get_function_by_name(function_name).unwrap();
    let function = runtime.get_pointer(function).unwrap();
    let result = save_volatile_registers(arg1, function as usize);

    // Restore GC context after Beagle call returns
    restore_gc_context(saved_ctx);
    result
}

pub unsafe fn call_fn_2(runtime: &Runtime, function_name: &str, arg1: usize, arg2: usize) -> usize {
    print_call_builtin(
        runtime,
        format!("{} {}", "call_fn_2", function_name).as_str(),
    );

    // Save GC context before calling into Beagle - the Beagle code may call builtins
    // that update the saved GC context, making it point to now-deallocated frames
    let saved_ctx = save_current_gc_context();

    let save_volatile_registers = runtime
        .get_function_by_name("beagle.builtin/save_volatile_registers2")
        .unwrap();
    let save_volatile_registers = runtime.get_pointer(save_volatile_registers).unwrap();
    let save_volatile_registers: fn(usize, usize, usize) -> usize =
        unsafe { std::mem::transmute(save_volatile_registers) };

    let function = runtime.get_function_by_name(function_name).unwrap();
    let function = runtime.get_pointer(function).unwrap();
    let result = save_volatile_registers(arg1, arg2, function as usize);

    // Restore GC context after Beagle call returns
    restore_gc_context(saved_ctx);
    result
}

// Helper to call a Beagle function or closure with one argument
pub unsafe fn call_beagle_fn_ptr(runtime: &Runtime, fn_or_closure: usize, arg1: usize) {
    // Save GC context before calling into Beagle - the Beagle code may call builtins
    // that update the saved GC context, making it point to now-deallocated frames
    let saved_ctx = save_current_gc_context();

    // Check if this is a closure (has closure tag)
    let kind = BuiltInTypes::get_kind(fn_or_closure);

    if matches!(kind, BuiltInTypes::Closure) {
        // It's a closure - extract function pointer
        // Closure structure: [header][function_ptr][num_free_vars][free_vars...]
        let untagged = BuiltInTypes::untag(fn_or_closure);
        let closure = HeapObject::from_untagged(untagged as *const u8);
        // Function pointer is at field 0, but it's tagged so need to untag it
        let fp_tagged = closure.get_field(0);
        let function_pointer = BuiltInTypes::untag(fp_tagged);

        // Call with closure as first arg, exception as second
        // Closures MUST receive the closure object itself as the first argument
        let save_volatile_registers = runtime
            .get_function_by_name("beagle.builtin/save_volatile_registers2")
            .unwrap();
        let save_volatile_registers = runtime.get_pointer(save_volatile_registers).unwrap();
        let save_volatile_registers: fn(usize, usize, usize) -> usize =
            unsafe { std::mem::transmute(save_volatile_registers) };

        save_volatile_registers(fn_or_closure, arg1, function_pointer);
    } else {
        // It's a regular function pointer - just pass exception
        let function_pointer = BuiltInTypes::untag(fn_or_closure);

        let save_volatile_registers = runtime
            .get_function_by_name("beagle.builtin/save_volatile_registers1")
            .unwrap();
        let save_volatile_registers = runtime.get_pointer(save_volatile_registers).unwrap();
        let save_volatile_registers: fn(usize, usize) -> usize =
            unsafe { std::mem::transmute(save_volatile_registers) };

        save_volatile_registers(arg1, function_pointer);
    }

    // Restore GC context after Beagle call returns
    restore_gc_context(saved_ctx);
}
