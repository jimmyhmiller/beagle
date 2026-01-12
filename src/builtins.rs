use core::panic;
use std::{
    arch::asm,
    cell::Cell,
    error::Error,
    ffi::{CStr, c_void},
    mem::{self, transmute},
    slice::{from_raw_parts, from_raw_parts_mut},
    thread,
};


use crate::{
    Message, debug_only,
    gc::STACK_SIZE,
    get_runtime,
    runtime::{DispatchTable, FFIInfo, FFIType, RawPtr, Runtime},
    types::{BuiltInTypes, HeapObject},
};

use std::hint::black_box;

// Thread-local storage for the frame pointer and gc return address at builtin entry.
// This is set by builtins that receive frame_pointer from Beagle code.
// Used by gc() when triggered internally (e.g., during allocation).
thread_local! {
    static SAVED_FRAME_POINTER: Cell<usize> = const { Cell::new(0) };
    static SAVED_GC_RETURN_ADDR: Cell<usize> = const { Cell::new(0) };
}

/// Save the frame pointer for later use by gc().
/// Called by builtins that receive frame_pointer from Beagle.
pub fn save_frame_pointer(fp: usize) {
    SAVED_FRAME_POINTER.with(|cell| cell.set(fp));
}

/// Get the saved frame pointer.
/// Returns 0 if none has been saved (shouldn't happen in normal operation).
pub fn get_saved_frame_pointer() -> usize {
    SAVED_FRAME_POINTER.with(|cell| cell.get())
}

/// Save the gc return address for later use by gc().
/// Called by builtins that receive frame_pointer/stack_pointer from Beagle.
pub fn save_gc_return_addr(addr: usize) {
    SAVED_GC_RETURN_ADDR.with(|cell| cell.set(addr));
}

/// Get the saved gc return address.
/// Returns 0 if none has been saved.
pub fn get_saved_gc_return_addr() -> usize {
    SAVED_GC_RETURN_ADDR.with(|cell| cell.get())
}

/// Reset the saved frame pointer and gc return address.
/// Called by Runtime::reset() to clear stale values between test runs.
pub fn reset_saved_gc_context() {
    SAVED_FRAME_POINTER.with(|cell| cell.set(0));
    SAVED_GC_RETURN_ADDR.with(|cell| cell.set(0));
}

/// Saved GC context - used to save/restore around calls back into Beagle
pub struct SavedGcContext {
    pub frame_pointer: usize,
    pub gc_return_addr: usize,
}

/// Save the current GC context. Call this BEFORE calling back into Beagle.
/// The Beagle code may call builtins that update the saved GC context.
/// After the Beagle code returns, call restore_gc_context to restore it.
pub fn save_current_gc_context() -> SavedGcContext {
    SavedGcContext {
        frame_pointer: get_saved_frame_pointer(),
        gc_return_addr: get_saved_gc_return_addr(),
    }
}

/// Restore a previously saved GC context. Call this AFTER calling back into Beagle.
/// This ensures that if GC runs after the Beagle call returns, it uses the
/// correct (non-stale) frame pointer.
pub fn restore_gc_context(ctx: SavedGcContext) {
    save_frame_pointer(ctx.frame_pointer);
    save_gc_return_addr(ctx.gc_return_addr);
}

/// Macro to save frame pointer and gc return address for GC stack walking.
/// The gc_return_addr is the return address from the Beagle code that called
/// this builtin. On ARM64, this was in LR and is now saved at [Rust_FP + 8]
/// by Rust's prologue. On x86-64, it's also at [Rust_FP + 8].
///
/// The saved frame_pointer (Beagle FP) is used by __pause to check if the
/// saved gc_return_addr is still valid for the current call.
#[macro_export]
macro_rules! save_gc_context {
    ($stack_pointer:expr, $frame_pointer:expr) => {{
        $crate::builtins::save_frame_pointer($frame_pointer);
        // Get the return address from where Rust's prologue saved it
        // This is the address in Beagle code right after the builtin call
        let rust_fp = $crate::builtins::get_current_rust_frame_pointer();
        let gc_return_addr = unsafe { *((rust_fp + 8) as *const usize) };
        $crate::builtins::save_gc_return_addr(gc_return_addr);
    }};
}

/// Read the current frame pointer register.
/// Used by GC to walk the stack starting from the current Rust function's frame.
/// MUST be inlined so we read the caller's frame pointer, not this function's.
#[inline(always)]
pub fn get_current_rust_frame_pointer() -> usize {
    let fp: usize;
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            unsafe { core::arch::asm!("mov {}, rbp", out(reg) fp) };
        } else if #[cfg(target_arch = "aarch64")] {
            unsafe { core::arch::asm!("mov {}, x29", out(reg) fp) };
        } else {
            compile_error!("Unsupported architecture");
        }
    }
    fp
}

#[allow(unused)]
#[unsafe(no_mangle)]
#[inline(never)]
/// # Safety
///
/// This does nothing
pub unsafe extern "C" fn debugger_info(buffer: *const u8, length: usize) {
    // Hack to make sure this isn't inlined
    black_box(buffer);
    black_box(length);
}

#[macro_export]
macro_rules! debug_only {
    ($($code:tt)*) => {
        #[cfg(debug_assertions)]
        {
            $($code)*
        }
    };
}

#[macro_export]
macro_rules! debug_flag_only {
    ($($code:tt)*) => {
        {
            let runtime = get_runtime().get();
            if runtime.get_command_line_args().debug {
                $($code)*
            }
        }
    };
}

pub fn debugger(_message: Message) {
    debug_only! {
        let serialized_message : Vec<u8>;
        #[cfg(feature="json")] {
        use nanoserde::SerJson;
            let serialized : String = SerJson::serialize_json(&_message);
            serialized_message = serialized.into_bytes();
        }
        #[cfg(not(feature="json"))] {
            use crate::Serialize;
            serialized_message = _message.to_binary();
        }
        let ptr = serialized_message.as_ptr();
        let length = serialized_message.len();
        mem::forget(serialized_message);
        unsafe {
            debugger_info(ptr, length);
        }
        // TODO: Should clean up this memory
    }
}

pub unsafe extern "C" fn println_value(value: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let result = runtime.println(value);
    if let Err(error) = result {
        let stack_pointer = get_current_stack_pointer();
        let frame_pointer = get_saved_frame_pointer();
        println!("Error: {:?}", error);
        unsafe { throw_error(stack_pointer, frame_pointer) };
    }
    0b111
}

pub unsafe extern "C" fn to_string(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let result = runtime.get_repr(value, 0);
    if result.is_none() {
        let stack_pointer = get_current_stack_pointer();
        unsafe { throw_error(stack_pointer, frame_pointer) };
    }
    let result = result.unwrap();
    runtime
        .allocate_string(stack_pointer, result)
        .unwrap()
        .into()
}

pub unsafe extern "C" fn to_number(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let string = runtime.get_string(stack_pointer, value);
    if string.contains(".") {
        todo!()
    } else {
        let result = string.parse::<isize>().unwrap();
        BuiltInTypes::Int.tag(result) as usize
    }
}

#[inline(always)]
fn print_call_builtin(_runtime: &Runtime, _name: &str) {
    debug_only!(if _runtime.get_command_line_args().print_builtin_calls {
        println!("Calling: {}", _name);
    });
}

pub unsafe extern "C" fn print_value(value: usize) -> usize {
    let runtime = get_runtime().get_mut();
    runtime.print(value);
    0b111
}

pub extern "C" fn print_byte(value: usize) -> usize {
    let byte_value = BuiltInTypes::untag(value) as u8;
    let runtime = get_runtime().get_mut();
    runtime.printer.print_byte(byte_value);
    0b111
}

/// Mark the card containing an address as dirty for write barrier.
/// Called from generated code after heap stores to old gen objects.
/// Takes the untagged address of the object being written to.
///
/// This is a no-op for non-generational GCs (card_table_ptr will be null).
/// For generational GC, it only marks cards for addresses in old gen.
pub extern "C" fn mark_card(untagged_addr: usize) -> usize {
    let runtime = get_runtime().get_mut();
    // Tag the address as HeapObject for the mark_card_for_object call
    let tagged_addr = BuiltInTypes::HeapObject.tag(untagged_addr as isize) as usize;
    // Mark the card if object is in old gen (no-op for non-generational GCs)
    runtime.mark_card_for_object(tagged_addr);
    0b111 // Return null
}

extern "C" fn allocate(stack_pointer: usize, frame_pointer: usize, size: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let size = BuiltInTypes::untag(size);
    let runtime = get_runtime().get_mut();

    let result = runtime
        .allocate(size, stack_pointer, BuiltInTypes::HeapObject)
        .unwrap();

    debug_assert!(BuiltInTypes::is_heap_pointer(result));
    debug_assert!(BuiltInTypes::untag(result) % 8 == 0);
    result
}

/// Allocate with zeroed memory (for arrays that don't initialize all fields)
extern "C" fn allocate_zeroed(stack_pointer: usize, frame_pointer: usize, size: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let size = BuiltInTypes::untag(size);
    let runtime = get_runtime().get_mut();

    let result = runtime
        .allocate_zeroed(size, stack_pointer, BuiltInTypes::HeapObject)
        .unwrap();

    debug_assert!(BuiltInTypes::is_heap_pointer(result));
    debug_assert!(BuiltInTypes::untag(result) % 8 == 0);
    result
}

extern "C" fn allocate_float(stack_pointer: usize, frame_pointer: usize, size: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let value = BuiltInTypes::untag(size);

    let result = runtime
        .allocate(value, stack_pointer, BuiltInTypes::Float)
        .unwrap();

    debug_assert!(BuiltInTypes::get_kind(result) == BuiltInTypes::Float);
    debug_assert!(BuiltInTypes::untag(result) % 8 == 0);
    result
}

extern "C" fn get_string_index(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    index: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "get_string_index");
    let runtime = get_runtime().get_mut();
    if BuiltInTypes::get_kind(string) == BuiltInTypes::String {
        let string = runtime.get_string_literal(string);
        let index = BuiltInTypes::untag(index);
        let result = string.chars().nth(index).unwrap();
        let result = result.to_string();
        runtime
            .allocate_string(stack_pointer, result)
            .unwrap()
            .into()
    } else {
        let object_pointer_id = runtime.register_temporary_root(string);
        // we have a heap allocated string
        let string = HeapObject::from_tagged(string);
        // TODO: Type safety
        // We are just going to assert that the type_id == 2
        assert!(string.get_type_id() == 2);
        let index = BuiltInTypes::untag(index);
        let string = string.get_string_bytes();
        // TODO: This will break with unicode
        let result = string[index] as char;
        let result = result.to_string();
        let result = runtime
            .allocate_string(stack_pointer, result)
            .unwrap()
            .into();
        runtime.unregister_temporary_root(object_pointer_id);
        result
    }
}

extern "C" fn get_string_length(string: usize) -> usize {
    print_call_builtin(get_runtime().get(), "get_string_length");
    let runtime = get_runtime().get_mut();
    if BuiltInTypes::get_kind(string) == BuiltInTypes::String {
        // TODO: Make faster
        let string = runtime.get_string_literal(string);
        BuiltInTypes::Int.tag(string.len() as isize) as usize
    } else {
        // we have a heap allocated string
        let string = HeapObject::from_tagged(string);
        let length = string.get_type_data();
        BuiltInTypes::Int.tag(length as isize) as usize
    }
}

extern "C" fn string_concat(
    stack_pointer: usize,
    frame_pointer: usize,
    a: usize,
    b: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "string_concat");
    let runtime = get_runtime().get_mut();
    let a = runtime.get_string(stack_pointer, a);
    let b = runtime.get_string(stack_pointer, b);
    let result = a + &b;
    runtime
        .allocate_string(stack_pointer, result)
        .unwrap()
        .into()
}

extern "C" fn substring(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    start: usize,
    length: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "substring");
    let runtime = get_runtime().get_mut();
    let string_pointer = runtime.register_temporary_root(string);
    let start = BuiltInTypes::untag(start);
    let length = BuiltInTypes::untag(length);
    let string = runtime
        .get_substring(stack_pointer, string, start, length)
        .unwrap();
    runtime.unregister_temporary_root(string_pointer);
    string.into()
}

extern "C" fn uppercase(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "uppercase");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let uppercased = string_value.to_uppercase();
    runtime
        .allocate_string(stack_pointer, uppercased)
        .unwrap()
        .into()
}

extern "C" fn fill_object_fields(object_pointer: usize, value: usize) -> usize {
    print_call_builtin(get_runtime().get(), "fill_object_fields");
    let mut object = HeapObject::from_tagged(object_pointer);
    let raw_slice = object.get_fields_mut();
    raw_slice.fill(value);
    object_pointer
}

extern "C" fn make_closure(
    stack_pointer: usize,
    frame_pointer: usize,
    function: usize,
    num_free: usize,
    free_variable_pointer: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "make_closure");
    let runtime = get_runtime().get_mut();
    if BuiltInTypes::get_kind(function) != BuiltInTypes::Function {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!(
                    "Expected function, got {:?}",
                    BuiltInTypes::get_kind(function)
                ),
            );
        }
    }

    assert!(matches!(
        BuiltInTypes::get_kind(function),
        BuiltInTypes::Function
    ));

    let num_free = BuiltInTypes::untag(num_free);
    let free_variable_pointer = free_variable_pointer as *const usize;
    let start = unsafe { free_variable_pointer.sub(num_free.saturating_sub(1)) };
    let free_variables = unsafe { from_raw_parts(start, num_free) };
    runtime
        .make_closure(stack_pointer, function, free_variables)
        .unwrap()
}

pub fn get_current_stack_pointer() -> usize {
    use core::arch::asm;
    let sp: usize;
    unsafe {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                asm!(
                    "mov {0}, rsp",
                    out(reg) sp
                );
            } else {
                asm!(
                    "mov {0}, sp",
                    out(reg) sp
                );
            }
        }
    }
    sp
}

/// Get the current frame pointer (RBP on x86-64, X29 on ARM64).
/// Note: This is currently unused because Rust functions may not preserve
/// frame pointers, making FP-chain traversal unreliable for GC.
#[allow(dead_code)]
pub fn get_current_frame_pointer() -> usize {
    use core::arch::asm;
    let fp: usize;
    unsafe {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                asm!(
                    "mov {0}, rbp",
                    out(reg) fp
                );
            } else {
                asm!(
                    "mov {0}, x29",
                    out(reg) fp
                );
            }
        }
    }
    fp
}

extern "C" fn property_access(
    struct_pointer: usize,
    str_constant_ptr: usize,
    property_cache_location: usize,
) -> usize {
    let runtime = get_runtime().get_mut();
    let (result, index) = runtime
        .property_access(struct_pointer, str_constant_ptr)
        .unwrap_or_else(|error| {
            let stack_pointer = get_current_stack_pointer();
            let frame_pointer = get_saved_frame_pointer();
            let heap_object = HeapObject::from_tagged(struct_pointer);
            println!("Heap object: {:?}", heap_object.get_header());
            println!("Error: {:?}", error);
            unsafe {
                throw_error(stack_pointer, frame_pointer);
            };
        });
    let type_id = HeapObject::from_tagged(struct_pointer).get_struct_id();
    let buffer = unsafe { from_raw_parts_mut(property_cache_location as *mut usize, 2) };
    buffer[0] = type_id;
    buffer[1] = index * 8;
    result
}

/// Slow path for protocol dispatch - looks up function in dispatch table and updates cache
/// Returns the function pointer to call
///
/// Type identification for dispatch:
/// - Tagged primitives (Int, Float, Bool): high bit marker + (tag + 16)
/// - Heap objects with type_id > 0 (Array=1, String=2, Keyword=3): high bit marker + type_id
/// - Heap objects with type_id = 0 (structs): struct_id from type_data (tagged)
extern "C" fn protocol_dispatch(
    first_arg: usize,
    cache_location: usize,
    dispatch_table_ptr: usize,
) -> usize {
    let type_id = if BuiltInTypes::is_heap_pointer(first_arg) {
        let heap_obj = HeapObject::from_tagged(first_arg);
        let header_type_id = heap_obj.get_type_id();

        if header_type_id == 0 {
            // Custom struct - use struct_id from type_data (tagged)
            heap_obj.get_struct_id()
        } else {
            // Built-in heap type (Array=1, String=2, Keyword=3)
            // Use primitive dispatch with header_type_id
            0x8000_0000_0000_0000 | header_type_id
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
        // Struct type - struct_id is tagged in header, need to untag
        let struct_id = BuiltInTypes::untag(type_id);
        dispatch_table.lookup_struct(struct_id)
    };

    if fn_ptr == 0 {
        panic!(
            "Protocol dispatch failed: no implementation found for type_id={:#x}",
            type_id
        );
    }

    // Update cache
    let buffer = unsafe { from_raw_parts_mut(cache_location as *mut usize, 2) };
    buffer[0] = type_id;
    buffer[1] = fn_ptr;

    fn_ptr
}

extern "C" fn type_of(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "type_of");
    let runtime = get_runtime().get_mut();
    runtime.type_of(stack_pointer, value).unwrap()
}

extern "C" fn get_os(stack_pointer: usize, frame_pointer: usize) -> usize {
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
    runtime
        .allocate_string(stack_pointer, os_name.to_string())
        .unwrap()
        .into()
}

extern "C" fn equal(a: usize, b: usize) -> usize {
    print_call_builtin(get_runtime().get(), "equal");
    let runtime = get_runtime().get_mut();
    if runtime.equal(a, b) {
        BuiltInTypes::true_value() as usize
    } else {
        BuiltInTypes::false_value() as usize
    }
}

extern "C" fn write_field(
    stack_pointer: usize,
    frame_pointer: usize,
    struct_pointer: usize,
    str_constant_ptr: usize,
    property_cache_location: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let index = runtime.write_field(stack_pointer, struct_pointer, str_constant_ptr, value);
    let type_id = HeapObject::from_tagged(struct_pointer).get_struct_id();
    let buffer = unsafe { from_raw_parts_mut(property_cache_location as *mut usize, 2) };
    buffer[0] = type_id;
    buffer[1] = index * 8;
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn throw_error(stack_pointer: usize, frame_pointer: usize) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);
    print_stack(stack_pointer);
    panic!("Error!");
}

pub unsafe extern "C" fn throw_type_error(stack_pointer: usize, frame_pointer: usize) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);

    let (kind_str, message_str) = {
        let runtime = get_runtime().get_mut();
        let kind = runtime
            .allocate_string(stack_pointer, "TypeError".to_string())
            .expect("Failed to allocate kind string")
            .into();
        let msg = runtime
            .allocate_string(
                stack_pointer,
                "Type mismatch in arithmetic operation".to_string(),
            )
            .expect("Failed to allocate message string")
            .into();
        (kind, msg)
    };

    let null_location = BuiltInTypes::Null.tag(0) as usize;
    unsafe {
        let error = create_error(
            stack_pointer,
            frame_pointer,
            kind_str,
            message_str,
            null_location,
        );
        throw_exception(stack_pointer, frame_pointer, error);
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
    let array_ptr = runtime
        .allocate_zeroed(num_extra, stack_pointer, BuiltInTypes::HeapObject)
        .unwrap();

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
    let function = runtime
        .get_function_by_pointer(untagged_fn)
        .expect("Invalid function pointer in call_variadic_function_value");
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
    let rest_array = runtime_mut
        .allocate_zeroed(num_extra, stack_pointer, BuiltInTypes::HeapObject)
        .unwrap();

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

    // Dispatch based on (min_args, is_closure)
    // We support min_args 0-7 which covers typical use cases
    // SAFETY: We're transmuting function pointers to call them with the correct number of arguments.
    // The function pointer comes from get_function_by_pointer which validates it's a known function.
    unsafe {
        match (min_args, is_closure_bool) {
            (0, false) => {
                let f: fn(usize) -> usize = transmute(untagged_fn);
                f(rest_array)
            }
            (0, true) => {
                let f: fn(usize, usize) -> usize = transmute(untagged_fn);
                f(closure_ptr, rest_array)
            }
            (1, false) => {
                let f: fn(usize, usize) -> usize = transmute(untagged_fn);
                f(all_args[0], rest_array)
            }
            (1, true) => {
                let f: fn(usize, usize, usize) -> usize = transmute(untagged_fn);
                f(closure_ptr, all_args[0], rest_array)
            }
            (2, false) => {
                let f: fn(usize, usize, usize) -> usize = transmute(untagged_fn);
                f(all_args[0], all_args[1], rest_array)
            }
            (2, true) => {
                let f: fn(usize, usize, usize, usize) -> usize = transmute(untagged_fn);
                f(closure_ptr, all_args[0], all_args[1], rest_array)
            }
            (3, false) => {
                let f: fn(usize, usize, usize, usize) -> usize = transmute(untagged_fn);
                f(all_args[0], all_args[1], all_args[2], rest_array)
            }
            (3, true) => {
                let f: fn(usize, usize, usize, usize, usize) -> usize = transmute(untagged_fn);
                f(
                    closure_ptr,
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    rest_array,
                )
            }
            (4, false) => {
                let f: fn(usize, usize, usize, usize, usize) -> usize = transmute(untagged_fn);
                f(
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    rest_array,
                )
            }
            (4, true) => {
                let f: fn(usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    closure_ptr,
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    rest_array,
                )
            }
            (5, false) => {
                let f: fn(usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    all_args[4],
                    rest_array,
                )
            }
            (5, true) => {
                let f: fn(usize, usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    closure_ptr,
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    all_args[4],
                    rest_array,
                )
            }
            (6, false) => {
                let f: fn(usize, usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    all_args[4],
                    all_args[5],
                    rest_array,
                )
            }
            (6, true) => {
                let f: fn(usize, usize, usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    closure_ptr,
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    all_args[4],
                    all_args[5],
                    rest_array,
                )
            }
            (7, false) => {
                let f: fn(usize, usize, usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    all_args[4],
                    all_args[5],
                    all_args[6],
                    rest_array,
                )
            }
            (7, true) => {
                let f: fn(usize, usize, usize, usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    closure_ptr,
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    all_args[4],
                    all_args[5],
                    all_args[6],
                    rest_array,
                )
            }
            _ => panic!(
                "Unsupported min_args value {} for variadic function call through function value",
                min_args
            ),
        }
    }
}

fn print_stack(_stack_pointer: usize) {
    let runtime = get_runtime().get_mut();
    let stack_base = runtime.get_stack_base();
    let stack_begin = stack_base - STACK_SIZE;

    // Get the current frame pointer directly from the register
    let fp: usize;
    let mut current_frame_ptr = unsafe {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                asm!(
                    "mov {0}, rbp",
                    out(reg) fp
                );
            } else {
                asm!(
                    "mov {0}, x29",
                    out(reg) fp
                );
            }
        }
        fp
    };

    println!("Walking stack frames:");

    let mut frame_count = 0;
    const MAX_FRAMES: usize = 100; // Prevent infinite loops

    while frame_count < MAX_FRAMES && current_frame_ptr != 0 {
        // Validate frame pointer is within stack bounds
        if current_frame_ptr < stack_begin || current_frame_ptr >= stack_base {
            break;
        }

        // Frame layout in our calling convention:
        // [previous_X29] [X30/return_addr] <- X29 points here
        // [zero] [zero] [locals...]
        let frame = current_frame_ptr as *const usize;
        let previous_frame_ptr = unsafe { *frame.offset(0) }; // Previous X29
        let return_address = unsafe { *frame.offset(1) }; // X30 (return address)

        // Look up the function containing this return address
        for function in runtime.functions.iter() {
            let function_size = function.size;
            let function_start = usize::from(function.pointer);
            let range = function_start..function_start + function_size;
            if range.contains(&return_address) {
                println!("Function: {:?}", function.name);
                break;
            }
        }

        // Move to the previous frame
        current_frame_ptr = previous_frame_ptr;
        frame_count += 1;
    }
}

pub unsafe extern "C" fn gc(stack_pointer: usize, frame_pointer: usize) -> usize {
    // Save the GC context including the return address
    save_gc_context!(stack_pointer, frame_pointer);
    let gc_return_addr = get_saved_gc_return_addr();
    #[cfg(feature = "debug-gc")]
    {
        eprintln!(
            "DEBUG gc: stack_pointer={:#x}, frame_pointer={:#x}, gc_return_addr={:#x}",
            stack_pointer, frame_pointer, gc_return_addr
        );
    }
    let runtime = get_runtime().get_mut();
    runtime.gc_impl(stack_pointer, frame_pointer, gc_return_addr);
    BuiltInTypes::null_value() as usize
}

/// sqrt builtin - computes square root of a float
/// Takes a tagged float pointer, returns a new tagged float pointer with the result
pub unsafe extern "C" fn sqrt_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        // Read the float value from the heap object
        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1); // Float value is at offset 1 (after header)

        // Compute sqrt
        let result = float_value.sqrt();

        // Allocate a new float object for the result
        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        // Write the result
        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// floor builtin - computes floor of a float
pub unsafe extern "C" fn floor_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.floor();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// ceil builtin - computes ceiling of a float
pub unsafe extern "C" fn ceil_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.ceil();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// abs builtin - computes absolute value of a float
pub unsafe extern "C" fn abs_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.abs();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// sin builtin - computes sine of a float (in radians)
pub unsafe extern "C" fn sin_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.sin();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// cos builtin - computes cosine of a float (in radians)
pub unsafe extern "C" fn cos_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.cos();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// to_float builtin - converts an integer to a float
pub unsafe extern "C" fn to_float_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        // Get the integer value (tagged integers have 0b000 tag)
        let int_value = BuiltInTypes::untag(value) as i64;

        // Convert to f64
        let float_value = int_value as f64;

        // Allocate a new float object for the result
        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = float_value;

        new_float_ptr
    }
}

#[allow(unused)]
pub unsafe extern "C" fn new_thread(
    stack_pointer: usize,
    frame_pointer: usize,
    function: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    #[cfg(feature = "thread-safe")]
    {
        let runtime = get_runtime().get_mut();
        runtime.new_thread(function, stack_pointer, frame_pointer);
        BuiltInTypes::null_value() as usize
    }
    #[cfg(not(feature = "thread-safe"))]
    {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "ThreadError",
                "Threads are not supported in this build".to_string(),
            );
        }
    }
}

// I don't know what the deal is here

#[unsafe(no_mangle)]
pub unsafe extern "C" fn update_binding(namespace_slot: usize, value: usize) -> usize {
    print_call_builtin(get_runtime().get(), "update_binding");
    let runtime = get_runtime().get_mut();
    let namespace_slot = BuiltInTypes::untag(namespace_slot);
    let namespace_id = runtime.current_namespace_id();

    // Store binding in heap-based PersistentMap (no namespace_roots tracking needed!)
    let stack_pointer = get_current_stack_pointer();
    if let Err(e) = runtime.set_heap_binding(stack_pointer, namespace_id, namespace_slot, value) {
        eprintln!("Error in update_binding: {}", e);
    }

    // Also update the Rust-side HashMap for backwards compatibility during migration
    runtime.update_binding(namespace_id, namespace_slot, value);

    BuiltInTypes::null_value() as usize
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_binding(namespace: usize, slot: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let namespace = BuiltInTypes::untag(namespace);
    let slot = BuiltInTypes::untag(slot);

    // TODO: Flush pending bindings at a safer point (not during get_binding)
    // For now, rely on HashMap fallback for bindings added by compiler thread
    // let stack_pointer = get_current_stack_pointer();
    // runtime.flush_pending_heap_bindings(stack_pointer);

    // Try heap-based PersistentMap first
    let result = runtime.get_heap_binding(namespace, slot);
    if result != BuiltInTypes::null_value() as usize {
        return result;
    }

    // Fall back to Rust-side HashMap during migration
    runtime.get_binding(namespace, slot)
}

pub unsafe extern "C" fn set_current_namespace(namespace: usize) -> usize {
    let runtime = get_runtime().get_mut();
    runtime.set_current_namespace(namespace);
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn __pause(_stack_pointer: usize, frame_pointer: usize) -> usize {
    use crate::gc::usdt_probes::{self, ThreadStateCode};

    // Get the gc_return_addr. When called from Rust code (get_my_thread_obj, gc_impl),
    // save_gc_context! was already called with the correct Beagle return address.
    // When called directly from Beagle (via primitive/__pause), we capture it ourselves.
    //
    // To distinguish these cases, we compare the Beagle frame_pointer we received
    // with the saved frame_pointer from save_gc_context!. If they match, we're in
    // the same Beagle call chain and the saved gc_return_addr is valid.
    let saved_beagle_fp = get_saved_frame_pointer();
    let my_rust_fp = get_current_rust_frame_pointer();
    let captured_addr = unsafe { *((my_rust_fp + 8) as *const usize) };
    let saved_addr = get_saved_gc_return_addr();

    let gc_return_addr = if saved_beagle_fp != 0 && saved_beagle_fp == frame_pointer {
        // Same Beagle frame as when save_gc_context! was called - use saved addr
        if std::env::var("BEAGLE_PAUSE_DEBUG").is_ok() {
            eprintln!(
                "[PAUSE_DEBUG] Using saved gc_return_addr={:#x} (saved_fp={:#x} == frame_pointer={:#x})",
                saved_addr, saved_beagle_fp, frame_pointer
            );
        }
        saved_addr
    } else {
        // Called directly from Beagle with a different/no saved frame - capture our own
        if std::env::var("BEAGLE_PAUSE_DEBUG").is_ok() {
            eprintln!(
                "[PAUSE_DEBUG] Using captured gc_return_addr={:#x} (saved_fp={:#x} != frame_pointer={:#x})",
                captured_addr, saved_beagle_fp, frame_pointer
            );
        }
        captured_addr
    };

    let runtime = get_runtime().get_mut();

    // Fire USDT probe for thread state change
    usdt_probes::fire_thread_pause_enter();
    usdt_probes::fire_thread_state(ThreadStateCode::PausedForGc);

    let pause_start = std::time::Instant::now();

    // Use frame_pointer passed from Beagle code for FP-chain stack walking
    pause_current_thread(frame_pointer, gc_return_addr, runtime);

    // Memory barrier to ensure all writes are visible before parking
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    while runtime.is_paused() {
        // Park can unpark itself even if I haven't called unpark
        thread::park();
    }

    // Apparently, I can't count on this not unparking
    // I need some other mechanism to know that things are ready
    unpause_current_thread(runtime);

    let pause_duration_ns = pause_start.elapsed().as_nanos() as u64;

    // Fire USDT probe for thread resuming
    usdt_probes::fire_thread_pause_exit(pause_duration_ns);
    usdt_probes::fire_thread_state(ThreadStateCode::Running);

    // Memory barrier to ensure all GC updates are visible before continuing
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    BuiltInTypes::null_value() as usize
}

fn pause_current_thread(frame_pointer: usize, gc_return_addr: usize, runtime: &mut Runtime) {
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    let stack_base = runtime.get_stack_base();
    // Store (stack_base, frame_pointer, gc_return_addr) for stack walking
    state.pause((stack_base, frame_pointer, gc_return_addr));
    condvar.notify_one();
    drop(state);
}

fn unpause_current_thread(runtime: &mut Runtime) {
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    state.unpause();
    condvar.notify_one();
}

pub extern "C" fn register_c_call(_stack_pointer: usize, frame_pointer: usize) -> usize {
    // Capture our return address - this is the safepoint in Beagle code
    let gc_return_addr = get_current_rust_frame_pointer();
    let gc_return_addr = unsafe { *((gc_return_addr + 8) as *const usize) };

    // Use frame_pointer passed from Beagle code for FP-chain stack walking
    let runtime = get_runtime().get_mut();
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    let stack_base = runtime.get_stack_base();
    state.register_c_call((stack_base, frame_pointer, gc_return_addr));
    condvar.notify_one();
    BuiltInTypes::null_value() as usize
}

pub extern "C" fn unregister_c_call() -> usize {
    let runtime = get_runtime().get_mut();
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    state.unregister_c_call();
    condvar.notify_one();
    while runtime.is_paused() {
        // Park can unpark itself even if I haven't called unpark
        thread::park();
    }
    BuiltInTypes::null_value() as usize
}

/// Called from a newly spawned thread to safely get a closure from a temporary root and call it.
/// This function:
/// 1. Unregisters from C-call (so this thread can participate in GC safepoints)
/// 2. Peeks the current closure value from the temporary root (which may have been updated by GC)
/// 3. Unregisters the temporary root
/// 4. Calls the closure via __call_fn
///
/// The temporary_root_id is passed as a tagged integer.
/// Get a value from a temporary root and unregister it.
/// This is called from Beagle code after entering a managed context.
pub extern "C" fn get_and_unregister_temp_root(temporary_root_id: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let root_id = BuiltInTypes::untag(temporary_root_id);
    // Read and unregister in one operation
    runtime.unregister_temporary_root(root_id)
}

/// Run a thread by calling the no-argument __run_thread_start function.
/// The Thread object is accessed via the get_my_thread_obj builtin inside Beagle code,
/// ensuring GC can update stack slots properly.
///
/// Key invariant: We are NOT counted in registered_thread_count until we hold gc_lock
/// and increment it. This prevents GC from waiting for us before we're ready.
///
/// Flow:
/// 1. Acquire gc_lock (GC can't start while we hold it)
/// 2. Increment registered_thread_count (now GC will count us)
/// 3. Release gc_lock
/// 4. Run Beagle code (first instruction is __pause, handles any pending GC)
/// 5. On cleanup: acquire gc_lock, decrement count, remove thread root, release
pub extern "C" fn run_thread(_unused: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let my_thread_id = thread::current().id();

    // === STARTUP ===
    // Acquire gc_lock and register ourselves.
    // We use blocking lock() here because we are NOT yet counted in registered_thread_count,
    // so GC won't wait for us - no deadlock possible.
    {
        let _guard = runtime.gc_lock.lock().unwrap();
        // Now register ourselves - GC will count us from this point on
        let new_count = runtime
            .registered_thread_count
            .fetch_add(1, std::sync::atomic::Ordering::Release)
            + 1;
        // Fire USDT probes for thread start and registration
        crate::gc::usdt_probes::fire_thread_register(new_count);
        crate::gc::usdt_probes::fire_thread_start();
        // Lock released here - GC can now start, and it will count us
    }

    // Enter Beagle code - __run_thread_start calls __pause as first instruction!
    // If GC started right after we released gc_lock, __pause will handle it.
    let result = unsafe { call_fn_0(runtime, "beagle.core/__run_thread_start") };

    // === CLEANUP ===
    // We're still registered but we're in C code now, not Beagle code.
    // If GC starts, it will wait for us to pause, but we can't pause from C.
    // Solution: register as c_calling so GC counts us and proceeds.
    {
        let (lock, condvar) = &*runtime.thread_state.clone();
        let mut state = lock.lock().unwrap();
        state.register_c_call((0, 0, 0)); // No stack to scan
        condvar.notify_one();
    }

    // Now any GC will count us as c_calling and proceed.
    // Wait for any in-progress GC to finish, then unregister everything.
    loop {
        while runtime.is_paused() {
            thread::yield_now();
        }

        match runtime.gc_lock.try_lock() {
            Ok(_guard) => {
                // While holding lock: unregister from c_calling, decrement count, remove root
                {
                    let (lock, condvar) = &*runtime.thread_state.clone();
                    let mut state = lock.lock().unwrap();
                    state.unregister_c_call();
                    condvar.notify_one();
                }
                let new_count = runtime
                    .registered_thread_count
                    .fetch_sub(1, std::sync::atomic::Ordering::Release)
                    - 1;
                // Thread object is in our GlobalObjectBlock - cleanup happens
                // when thread_globals entry is removed
                runtime.memory.thread_globals.remove(&my_thread_id);
                // Fire USDT probes for thread unregistration and exit
                crate::gc::usdt_probes::fire_thread_unregister(new_count);
                crate::gc::usdt_probes::fire_thread_exit();
                break;
            }
            Err(_) => {
                thread::yield_now();
            }
        }
    }

    result
}

/// Get the current thread's Thread object from its GlobalObjectBlock.
/// Called from Beagle code in __run_thread_start.
/// Takes stack_pointer and frame_pointer so we can call __pause if needed.
///
/// CRITICAL: This function must return a pointer that won't become stale before
/// the caller can use it. We check is_paused() after releasing the lock to ensure
/// that if GC is about to run, we pause and get the updated pointer.
pub extern "C" fn get_my_thread_obj(stack_pointer: usize, frame_pointer: usize) -> usize {
    // CRITICAL: Save the gc context here so that if we call __pause,
    // the GC will have the correct return address pointing to Beagle code.
    save_gc_context!(stack_pointer, frame_pointer);

    let runtime = get_runtime().get_mut();
    let thread_id = thread::current().id();

    // Read the Thread object from our GlobalObjectBlock's reserved slot.
    // We need the gc_lock to ensure the value isn't stale during GC.
    let thread_obj = loop {
        // Check if GC needs us to pause first
        if runtime.is_paused() {
            unsafe { __pause(stack_pointer, frame_pointer) };
            continue;
        }

        // Try to get the lock
        let obj = match runtime.gc_lock.try_lock() {
            Ok(_guard) => runtime
                .memory
                .thread_globals
                .get(&thread_id)
                .map(|tg| tg.get_thread_object())
                .expect("ThreadGlobal not found in get_my_thread_obj"),
            Err(_) => {
                thread::yield_now();
                continue;
            }
        };

        // After releasing the lock, check if GC is pending.
        if runtime.is_paused() {
            unsafe { __pause(stack_pointer, frame_pointer) };
            continue;
        }

        break obj;
    };

    if std::env::var("BEAGLE_THREAD_DEBUG").is_ok() {
        eprintln!(
            "[THREAD_DEBUG] get_my_thread_obj: thread_id={:?} thread_obj={:#x}",
            thread_id, thread_obj
        );
        if BuiltInTypes::is_heap_pointer(thread_obj) {
            let heap_obj = HeapObject::from_tagged(thread_obj);
            let closure_field = heap_obj.get_field(0);
            eprintln!(
                "[THREAD_DEBUG]   closure_field={:#x} is_heap_ptr={}",
                closure_field,
                BuiltInTypes::is_heap_pointer(closure_field)
            );
            if BuiltInTypes::is_heap_pointer(closure_field) {
                let closure_tag = BuiltInTypes::get_kind(closure_field);
                eprintln!("[THREAD_DEBUG]   closure_tag={:?}", closure_tag);
                if matches!(closure_tag, BuiltInTypes::Closure) {
                    let closure_obj = HeapObject::from_tagged(closure_field);
                    let fn_ptr = closure_obj.get_field(0);
                    let fn_ptr_untagged = BuiltInTypes::untag(fn_ptr);
                    if let Some(function) =
                        runtime.get_function_by_pointer(fn_ptr_untagged as *const u8)
                    {
                        eprintln!(
                            "[THREAD_DEBUG]   closure fn={} args={}",
                            function.name, function.number_of_args
                        );
                    } else {
                        eprintln!("[THREAD_DEBUG]   closure fn ptr not found: {:#x}", fn_ptr);
                        panic!("Closure function pointer not found in runtime");
                    }
                }
            }
        }
    }

    thread_obj
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

pub unsafe extern "C" fn load_library(name: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let string = &runtime.get_string_literal(name);
    let lib = unsafe { libloading::Library::new(string).unwrap() };
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
    let struct_id = BuiltInTypes::untag(heap_object.get_struct_id());
    let struct_info = runtime.get_struct_by_id(struct_id);

    if struct_info.is_none() {
        return Err(format!("Could not find struct with id {}", struct_id));
    }

    let struct_info = struct_info.unwrap();
    let name = struct_info.name.as_str().split_once("/").unwrap().1;
    match name {
        "Type.U8" => Ok(FFIType::U8),
        "Type.U16" => Ok(FFIType::U16),
        "Type.U32" => Ok(FFIType::U32),
        "Type.U64" => Ok(FFIType::U64),
        "Type.I32" => Ok(FFIType::I32),
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
    let func_ptr = unsafe { library.get::<fn()>(function_name.as_bytes()).unwrap() };
    let code_ptr = unsafe { func_ptr.try_as_raw_ptr().unwrap() };

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
        BuiltInTypes::Int => {
            match ffi_type {
                FFIType::U8 | FFIType::U16 | FFIType::U32 | FFIType::U64 | FFIType::I32 => {
                    BuiltInTypes::untag(argument) as u64
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
                }
            }
        }
        BuiltInTypes::HeapObject => {
            match ffi_type {
                FFIType::Pointer | FFIType::MutablePointer => {
                    let heap_object = HeapObject::from_tagged(argument);
                    let buffer = BuiltInTypes::untag(heap_object.get_field(0));
                    buffer as u64
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
                        format!("Got HeapObject but expected matching FFI type, got {:?}", ffi_type),
                    );
                }
            }
        }
        _ => unsafe {
            runtime.print(argument);
            throw_runtime_error(
                stack_pointer,
                "FFIError",
                format!("Unsupported FFI type: {:?}", kind),
            );
        }
    }
}

/// Call a native function with the given number of arguments.
/// Uses transmute to cast the function pointer to the appropriate signature.
#[inline(never)]
unsafe fn call_native_function(
    func_ptr: *const u8,
    num_args: usize,
    args: [u64; 6],
) -> u64 {
    unsafe {
        match num_args {
            0 => {
                let f: extern "C" fn() -> u64 = transmute(func_ptr);
                f()
            }
            1 => {
                let f: extern "C" fn(u64) -> u64 = transmute(func_ptr);
                f(args[0])
            }
            2 => {
                let f: extern "C" fn(u64, u64) -> u64 = transmute(func_ptr);
                f(args[0], args[1])
            }
            3 => {
                let f: extern "C" fn(u64, u64, u64) -> u64 = transmute(func_ptr);
                f(args[0], args[1], args[2])
            }
            4 => {
                let f: extern "C" fn(u64, u64, u64, u64) -> u64 = transmute(func_ptr);
                f(args[0], args[1], args[2], args[3])
            }
            5 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64) -> u64 = transmute(func_ptr);
                f(args[0], args[1], args[2], args[3], args[4])
            }
            6 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 = transmute(func_ptr);
                f(args[0], args[1], args[2], args[3], args[4], args[5])
            }
            _ => panic!("Too many arguments for FFI call: {}", num_args),
        }
    }
}

/// Convert a native return value to a Beagle value.
unsafe fn unmarshal_ffi_return(
    runtime: &mut Runtime,
    stack_pointer: usize,
    result: u64,
    return_type: &FFIType,
) -> usize {
    unsafe {
        match return_type {
            FFIType::Void => BuiltInTypes::null_value() as usize,
            FFIType::U8 | FFIType::U16 | FFIType::U32 | FFIType::U64 => {
                BuiltInTypes::Int.tag(result as isize) as usize
            }
            FFIType::I32 => {
                // Sign-extend I32 to isize properly
                let signed_result = result as i32 as isize;
                BuiltInTypes::Int.tag(signed_result) as usize
            }
            FFIType::Pointer | FFIType::MutablePointer => {
                let pointer_value = BuiltInTypes::Int.tag(result as isize) as usize;
                call_fn_1(runtime, "beagle.ffi/__make_pointer_struct", pointer_value)
            }
            FFIType::String => {
                if result == 0 {
                    return BuiltInTypes::null_value() as usize;
                }
                let c_string = CStr::from_ptr(result as *const i8);
                let string = c_string.to_str().unwrap();
                runtime
                    .allocate_string(stack_pointer, string.to_string())
                    .unwrap()
                    .into()
            }
            FFIType::Structure(_) => {
                todo!("Structure return not yet implemented")
            }
        }
    }
}

/// Call a foreign function through FFI.
/// This implementation uses direct function pointer calls via transmute,
/// eliminating the need for libffi.
pub unsafe extern "C" fn call_ffi_info(
    stack_pointer: usize,
    _frame_pointer: usize,
    ffi_info_id: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
    a6: usize,
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

        let arguments = [a1, a2, a3, a4, a5, a6];
        let args = &arguments[..number_of_arguments];

        // Marshal arguments to native u64 values
        let mut native_args: [u64; 6] = [0; 6];
        for (i, (argument, ffi_type)) in args.iter().zip(argument_types.iter()).enumerate() {
            native_args[i] = marshal_ffi_argument(runtime, stack_pointer, *argument, ffi_type);
        }

        // Call the native function directly using transmute
        let result = call_native_function(func_ptr, number_of_arguments, native_args);

        // Unmarshal the return value
        let return_value = unmarshal_ffi_return(runtime, stack_pointer, result, &return_type);

        runtime.memory.clear_native_arguments();
        return_value
    }
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
        runtime.allocate(size, stack_pointer, kind).unwrap()
    };
    let object_pointer = runtime.unregister_temporary_root(object_pointer_id);
    let mut to_object = HeapObject::from_tagged(to_pointer);
    let object = HeapObject::from_tagged(object_pointer);
    let result = runtime.copy_object(object, &mut to_object);
    if let Err(error) = result {
        let stack_pointer = get_current_stack_pointer();
        println!("Error: {:?}", error);
        unsafe { throw_error(stack_pointer, frame_pointer) };
    } else {
        result.unwrap()
    }
}

pub unsafe extern "C" fn copy_from_to_object(from: usize, to: usize) -> usize {
    let runtime = get_runtime().get_mut();
    if from == BuiltInTypes::null_value() as usize {
        return to;
    }
    let from = HeapObject::from_tagged(from);
    let mut to = HeapObject::from_tagged(to);
    runtime.copy_object_except_header(from, &mut to).unwrap();
    to.tagged_pointer()
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

unsafe extern "C" fn ffi_allocate(size: usize) -> usize {
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

unsafe extern "C" fn ffi_deallocate(buffer: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let size = BuiltInTypes::untag(buffer_object.get_field(1));
        let _buffer = Vec::from_raw_parts(buffer, size, size);
        BuiltInTypes::null_value() as usize
    }
}

unsafe extern "C" fn ffi_get_u32(buffer: usize, offset: usize) -> usize {
    unsafe {
        // TODO: Make type safe
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset) as *const u32);
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

unsafe extern "C" fn ffi_set_u8(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value);
        assert!(value <= u8::MAX as usize);
        let value = value as u8;
        *(buffer.add(offset)) = value;
        BuiltInTypes::null_value() as usize
    }
}

unsafe extern "C" fn ffi_get_u8(buffer: usize, offset: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset));
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

unsafe extern "C" fn ffi_set_i32(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value) as i32;
        *(buffer.add(offset) as *mut i32) = value;
        BuiltInTypes::null_value() as usize
    }
}

unsafe extern "C" fn ffi_set_i16(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value) as i16;
        *(buffer.add(offset) as *mut i16) = value;
        BuiltInTypes::null_value() as usize
    }
}

unsafe extern "C" fn ffi_get_i32(buffer: usize, offset: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset) as *const i32);
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

unsafe extern "C" fn ffi_get_string(
    stack_pointer: usize,
    _frame_pointer: usize, // Frame pointer for GC stack walking
    buffer: usize,
    offset: usize,
    len: usize,
) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);
        let slice = std::slice::from_raw_parts(buffer.add(offset), len);
        let string = std::str::from_utf8(slice).unwrap();
        runtime
            .allocate_string(stack_pointer, string.to_string())
            .unwrap()
            .into()
    }
}

unsafe extern "C" fn ffi_create_array(
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
                FFIType::U8 => todo!(),
                FFIType::U16 => todo!(),
                FFIType::U32 => todo!(),
                FFIType::U64 => todo!(),
                FFIType::I32 => todo!(),
                FFIType::Pointer => {
                    todo!()
                }
                FFIType::MutablePointer => {
                    todo!()
                }
                FFIType::Structure(_) => {
                    todo!()
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
        let buffer = BuiltInTypes::Int.tag(buffer_ptr as isize) as usize;
        call_fn_1(runtime, "beagle.ffi/__make_pointer_struct", buffer)
    }
}

// Copy bytes between FFI buffers with offsets
// ffi_copy_bytes(src, src_off, dst, dst_off, len) -> null
unsafe extern "C" fn ffi_copy_bytes(
    src: usize,
    src_off: usize,
    dst: usize,
    dst_off: usize,
    len: usize,
) -> usize {
    unsafe {
        let src_object = HeapObject::from_tagged(src);
        let src_ptr = BuiltInTypes::untag(src_object.get_field(0)) as *const u8;
        let src_off = BuiltInTypes::untag(src_off);

        let dst_object = HeapObject::from_tagged(dst);
        let dst_ptr = BuiltInTypes::untag(dst_object.get_field(0)) as *mut u8;
        let dst_off = BuiltInTypes::untag(dst_off);

        let len = BuiltInTypes::untag(len);

        std::ptr::copy_nonoverlapping(src_ptr.add(src_off), dst_ptr.add(dst_off), len);

        BuiltInTypes::null_value() as usize
    }
}

// Reallocate an FFI buffer to a new size
// ffi_realloc(buffer, new_size) -> new_buffer
unsafe extern "C" fn ffi_realloc(buffer: usize, new_size: usize) -> usize {
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
unsafe extern "C" fn ffi_buffer_size(buffer: usize) -> usize {
    let buffer_object = HeapObject::from_tagged(buffer);
    let size = BuiltInTypes::untag(buffer_object.get_field(1));
    BuiltInTypes::Int.tag(size as isize) as usize
}

// Write from buffer at offset to a file descriptor
// ffi_write_buffer_offset(fd, buffer, offset, len) -> bytes_written
unsafe extern "C" fn ffi_write_buffer_offset(
    fd: usize,
    buffer: usize,
    offset: usize,
    len: usize,
) -> usize {
    unsafe {
        let fd = BuiltInTypes::untag(fd) as i32;
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer_ptr = BuiltInTypes::untag(buffer_object.get_field(0)) as *const u8;
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);

        let result = libc::write(fd, buffer_ptr.add(offset) as *const libc::c_void, len);

        BuiltInTypes::Int.tag(result as isize) as usize
    }
}

// Translate bytes in buffer using a 256-byte lookup table
// ffi_translate_bytes(buffer, offset, len, table) -> null
unsafe extern "C" fn ffi_translate_bytes(
    buffer: usize,
    offset: usize,
    len: usize,
    table: usize,
) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer_ptr = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);

        let table_object = HeapObject::from_tagged(table);
        let table_ptr = BuiltInTypes::untag(table_object.get_field(0)) as *const u8;

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
unsafe extern "C" fn ffi_reverse_bytes(buffer: usize, offset: usize, len: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer_ptr = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
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
unsafe extern "C" fn ffi_find_byte(buffer: usize, offset: usize, len: usize, byte: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer_ptr = BuiltInTypes::untag(buffer_object.get_field(0)) as *const u8;
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
unsafe extern "C" fn ffi_copy_bytes_filter(
    src: usize,
    src_off: usize,
    dst: usize,
    dst_off: usize,
    len: usize,
    skip_byte: usize,
) -> usize {
    unsafe {
        let src_object = HeapObject::from_tagged(src);
        let src_ptr = BuiltInTypes::untag(src_object.get_field(0)) as *const u8;
        let src_off = BuiltInTypes::untag(src_off);

        let dst_object = HeapObject::from_tagged(dst);
        let dst_ptr = BuiltInTypes::untag(dst_object.get_field(0)) as *mut u8;
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

extern "C" fn placeholder() -> usize {
    BuiltInTypes::null_value() as usize
}

extern "C" fn wait_for_input(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let runtime = get_runtime().get_mut();
    let string = runtime.allocate_string(stack_pointer, input);
    string.unwrap().into()
}

// Get the ASCII code of the first character of a string
extern "C" fn char_code(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let string = runtime.get_string(stack_pointer, string);
    if let Some(ch) = string.chars().next() {
        BuiltInTypes::Int.tag(ch as isize) as usize
    } else {
        // Empty string returns -1
        BuiltInTypes::Int.tag(-1) as usize
    }
}

// Create a single-character string from an ASCII code
extern "C" fn char_from_code(stack_pointer: usize, frame_pointer: usize, code: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let code = BuiltInTypes::untag(code) as u8;
    let ch = code as char;
    let runtime = get_runtime().get_mut();
    runtime
        .allocate_string(stack_pointer, ch.to_string())
        .unwrap()
        .into()
}

// Read a line from stdin, stripping the trailing newline
// Returns null if EOF is reached
extern "C" fn read_line(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let mut input = String::new();
    match std::io::stdin().read_line(&mut input) {
        Ok(0) => {
            // EOF reached
            BuiltInTypes::null_value() as usize
        }
        Ok(_) => {
            // Remove trailing newline if present
            if input.ends_with('\n') {
                input.pop();
                if input.ends_with('\r') {
                    input.pop();
                }
            }
            let runtime = get_runtime().get_mut();
            let string = runtime.allocate_string(stack_pointer, input);
            string.unwrap().into()
        }
        Err(_) => {
            // Error - return null
            BuiltInTypes::null_value() as usize
        }
    }
}

extern "C" fn read_full_file(
    stack_pointer: usize,
    frame_pointer: usize,
    file_name: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let file_name = runtime.get_string(stack_pointer, file_name);
    let file = std::fs::read_to_string(file_name).unwrap();
    let string = runtime.allocate_string(stack_pointer, file);
    string.unwrap().into()
}

extern "C" fn eval(stack_pointer: usize, frame_pointer: usize, code: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let code = match BuiltInTypes::get_kind(code) {
        BuiltInTypes::String => runtime.get_string_literal(code),
        BuiltInTypes::HeapObject => {
            let code = HeapObject::from_tagged(code);
            if code.get_header().type_id != 2 {
                unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "TypeError",
                        format!(
                            "Expected string, got heap object with type_id {}",
                            code.get_header().type_id
                        ),
                    );
                }
            }
            let bytes = code.get_string_bytes();
            let code = std::str::from_utf8(bytes).unwrap();
            code.to_string()
        }
        _ => unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!("Expected string, got {:?}", BuiltInTypes::get_kind(code)),
            );
        },
    };
    let result = match runtime.compile_string(&code) {
        Ok(result) => result,
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "CompileError",
                format!("Compilation failed: {}", e),
            );
        },
    };
    mem::forget(code);
    if result == 0 {
        return BuiltInTypes::null_value() as usize;
    }
    let f: fn() -> usize = unsafe { transmute(result) };
    f()
}

extern "C" fn sleep(time: usize) -> usize {
    let time = BuiltInTypes::untag(time);
    std::thread::sleep(std::time::Duration::from_millis(time as u64));
    BuiltInTypes::null_value() as usize
}

/// High-precision timer returning nanoseconds since an arbitrary epoch
/// Useful for benchmarking - subtract two values to get elapsed time
extern "C" fn time_now() -> usize {
    use std::time::Instant;
    // Use a static instant as the epoch to avoid overflow
    // This gives us relative timing which is what we need for benchmarks
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    let elapsed = start.elapsed().as_nanos() as isize;
    BuiltInTypes::Int.tag(elapsed) as usize
}

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
    let protocol_name = runtime.resolve(protocol_name);

    runtime.add_protocol_info(&protocol_name, &struct_name, &method_name, f);

    runtime.compile_protocol_method(&protocol_name, &method_name);

    BuiltInTypes::null_value() as usize
}

extern "C" fn hash(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let _ = stack_pointer; // Used by save_gc_context!
    print_call_builtin(get_runtime().get(), "hash");
    let runtime = get_runtime().get();
    let raw_hash = runtime.hash_value(value);
    BuiltInTypes::Int.tag(raw_hash as isize) as usize
}

pub extern "C" fn is_keyword(value: usize) -> usize {
    let tag = BuiltInTypes::get_kind(value);
    if tag != BuiltInTypes::HeapObject {
        return BuiltInTypes::construct_boolean(false) as usize;
    }
    let heap_object = HeapObject::from_tagged(value);
    let is_kw = heap_object.get_header().type_id == 3;
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
                "keyword->string expects a keyword".to_string(),
            );
        }
    }

    let heap_object = HeapObject::from_tagged(keyword);

    if heap_object.get_header().type_id != 3 {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                "keyword->string expects a keyword".to_string(),
            );
        }
    }

    let bytes = heap_object.get_keyword_bytes();
    let keyword_text = unsafe { std::str::from_utf8_unchecked(bytes) };

    runtime
        .allocate_string(stack_pointer, keyword_text.to_string())
        .unwrap()
        .into()
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
    runtime.intern_keyword(stack_pointer, keyword_text).unwrap()
}

pub extern "C" fn load_keyword_constant_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    index: usize,
) -> usize {
    use crate::types::BuiltInTypes;

    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Check heap-based PersistentMap first (survives GC relocation)
    let kw_ns = runtime.keyword_namespace_id();
    if kw_ns != 0 {
        let heap_ptr = runtime.get_heap_binding(kw_ns, index);
        if heap_ptr != BuiltInTypes::null_value() as usize {
            return heap_ptr;
        }
    }

    // Allocate and register in heap-based map
    let keyword_text = runtime.keyword_constants[index].str.clone();
    runtime.intern_keyword(stack_pointer, keyword_text).unwrap()
}

extern "C" fn many_args(
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

extern "C" fn pop_count(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "pop_count");
    let tag = BuiltInTypes::get_kind(value);
    match tag {
        BuiltInTypes::Int => {
            let value = BuiltInTypes::untag(value);
            let count = value.count_ones();
            BuiltInTypes::Int.tag(count as isize) as usize
        }
        _ => unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!("Expected int, got {:?}", tag),
            );
        },
    }
}

// Exception handling builtins
pub unsafe extern "C" fn push_exception_handler_runtime(
    handler_address: usize,
    result_local: isize,
    link_register: usize,
    stack_pointer: usize,
    frame_pointer: usize,
) -> usize {
    print_call_builtin(get_runtime().get(), "push_exception_handler");
    let runtime = get_runtime().get_mut();

    // All values are passed as parameters since we can't reliably read them
    // inside this function (x30 gets clobbered by the call, SP/FP might be modified)
    let handler = crate::runtime::ExceptionHandler {
        handler_address,
        stack_pointer,
        frame_pointer,
        link_register,
        result_local,
    };

    runtime.push_exception_handler(handler);
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn pop_exception_handler_runtime() -> usize {
    print_call_builtin(get_runtime().get(), "pop_exception_handler");
    let runtime = get_runtime().get_mut();
    runtime.pop_exception_handler();
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn throw_exception(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "throw_exception");

    // Create exception object
    let exception = {
        let runtime = get_runtime().get_mut();
        runtime.create_exception(stack_pointer, value).unwrap()
    };

    // Pop handlers until we find one
    if let Some(handler) = {
        let runtime = get_runtime().get_mut();
        runtime.pop_exception_handler()
    } {
        // Restore stack, frame, and link register pointers
        let handler_address = handler.handler_address;
        let new_sp = handler.stack_pointer;
        let new_fp = handler.frame_pointer;
        let new_lr = handler.link_register;
        let result_local_offset = handler.result_local;

        let result_ptr = (new_fp as isize).wrapping_add(result_local_offset) as *mut usize;
        unsafe {
            *result_ptr = exception;
        }

        // Jump to handler with restored SP, FP, and LR
        unsafe {
            cfg_if::cfg_if! {
                if #[cfg(target_arch = "x86_64")] {
                    // x86-64: restore RSP, RBP, and jump
                    // The return address is already on the stack at [RBP + 8]
                    // so we don't need to push it (unlike ARM64 which uses LR register)
                    let _ = new_lr; // unused on x86-64, return addr is on stack
                    asm!(
                        "mov rsp, {0}",
                        "mov rbp, {1}",
                        "jmp {2}",
                        in(reg) new_sp,
                        in(reg) new_fp,
                        in(reg) handler_address,
                        options(noreturn)
                    );
                } else {
                    // ARM64: restore SP, X29 (FP), X30 (LR), and branch
                    asm!(
                        "mov sp, {0}",
                        "mov x29, {1}",
                        "mov x30, {2}",
                        "br {3}",
                        in(reg) new_sp,
                        in(reg) new_fp,
                        in(reg) new_lr,
                        in(reg) handler_address,
                        options(noreturn)
                    );
                }
            }
        }
    } else {
        // No try-catch handler found
        // Check per-thread uncaught exception handler (JVM-style)
        let thread_handler_fn = {
            let runtime = get_runtime().get();
            runtime.get_thread_exception_handler()
        };

        if let Some(handler_fn) = thread_handler_fn {
            // Call the per-thread uncaught exception handler
            let runtime = get_runtime().get();
            unsafe { call_beagle_fn_ptr(runtime, handler_fn, exception) };
            // Handler ran, now terminate since exception was uncaught
            println!("Uncaught exception after thread handler:");
            get_runtime().get_mut().println(exception).ok();
            unsafe { throw_error(stack_pointer, frame_pointer) };
        }

        // Check default (global) uncaught exception handler
        let default_handler_fn = {
            let runtime = get_runtime().get();
            runtime.default_exception_handler_fn
        };

        if let Some(handler_fn) = default_handler_fn {
            // Call the Beagle handler function
            let runtime = get_runtime().get();
            unsafe { call_beagle_fn_ptr(runtime, handler_fn, exception) };
            // Handler ran, now panic since exception was uncaught
            println!("Uncaught exception after default handler:");
            get_runtime().get_mut().println(exception).ok();
            unsafe { throw_error(stack_pointer, frame_pointer) };
        }

        // No handler at all - panic with stack trace
        println!("Uncaught exception:");
        get_runtime().get_mut().println(exception).ok();
        unsafe { throw_error(stack_pointer, frame_pointer) };
    }
}

pub unsafe extern "C" fn set_thread_exception_handler(handler_fn: usize) {
    let runtime = get_runtime().get_mut();
    runtime.set_thread_exception_handler(handler_fn);
}

pub unsafe extern "C" fn set_default_exception_handler(handler_fn: usize) {
    let runtime = get_runtime().get_mut();
    runtime.set_default_exception_handler(handler_fn);
}

/// Creates an Error struct on the heap
/// Returns tagged heap pointer to Error { kind, message, location }
pub unsafe extern "C" fn create_error(
    stack_pointer: usize,
    _frame_pointer: usize, // Frame pointer for GC stack walking (needed since create_struct can allocate)
    kind_str: usize, // Tagged string specifying the error variant (e.g., "StructError", "TypeError")
    message_str: usize, // Tagged string
    location_str: usize, // Tagged string or null
) -> usize {
    print_call_builtin(get_runtime().get(), "create_error");

    let runtime = get_runtime().get_mut();

    // Extract the error kind string to determine which variant to create
    // Use get_string which handles both string constants and heap-allocated strings
    let kind = runtime.get_string(stack_pointer, kind_str);

    // Use the general struct creation helper
    let fields = vec![message_str, location_str];
    runtime
        .create_struct(
            "beagle.core/SystemError",
            Some(&kind),
            &fields,
            stack_pointer,
        )
        .expect("Failed to create SystemError")
}

/// Helper to throw a runtime error with kind and message strings
/// This is a convenience function for Rust code to throw structured exceptions
pub unsafe fn throw_runtime_error(stack_pointer: usize, kind: &str, message: String) -> ! {
    // Get frame_pointer from thread-local (set by the builtin entry)
    let frame_pointer = get_saved_frame_pointer();

    // Allocate strings with proper GC root protection
    // kind_str must be rooted before allocating message_str since allocation can trigger GC
    let (kind_str, message_str) = {
        let runtime = get_runtime().get_mut();
        let kind_str: usize = runtime
            .allocate_string(stack_pointer, kind.to_string())
            .expect("Failed to allocate kind string")
            .into();

        // Register kind_str as a root before allocating message_str
        let kind_root_id = runtime.register_temporary_root(kind_str);

        let message_str: usize = runtime
            .allocate_string(stack_pointer, message)
            .expect("Failed to allocate message string")
            .into();

        // Get the potentially updated kind_str after GC
        let kind_str = runtime.unregister_temporary_root(kind_root_id);
        (kind_str, message_str)
    };
    // Runtime borrow is dropped here

    let null_location = BuiltInTypes::Null.tag(0) as usize;

    // Create the Error struct and throw it
    unsafe {
        let error = create_error(
            stack_pointer,
            frame_pointer,
            kind_str,
            message_str,
            null_location,
        );
        throw_exception(stack_pointer, frame_pointer, error);
    }
}

// It is very important that this isn't inlined
// because other code is expecting that
// If we inline, we need to remove a skip frame
impl Runtime {
    pub fn install_builtins(&mut self) -> Result<(), Box<dyn Error>> {
        self.add_builtin_function(
            "beagle.__internal_test__/many_args",
            many_args as *const u8,
            false,
            11,
        )?;

        self.add_builtin_function("beagle.core/_println", println_value as *const u8, false, 1)?;

        self.add_builtin_function("beagle.core/_print", print_value as *const u8, false, 1)?;

        // to_string now takes (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp(
            "beagle.core/to-string",
            to_string as *const u8,
            true,
            true,
            3,
        )?;
        // to_number now takes (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp(
            "beagle.core/to-number",
            to_number as *const u8,
            true,
            true,
            3,
        )?;

        // allocate now takes (stack_pointer, frame_pointer, size)
        self.add_builtin_function_with_fp(
            "beagle.builtin/allocate",
            allocate as *const u8,
            true,
            true,
            3,
        )?;

        // allocate_zeroed for arrays (zeroes memory)
        self.add_builtin_function_with_fp(
            "beagle.builtin/allocate-zeroed",
            allocate_zeroed as *const u8,
            true,
            true,
            3,
        )?;

        // allocate_float now takes (stack_pointer, frame_pointer, size)
        self.add_builtin_function_with_fp(
            "beagle.builtin/allocate-float",
            allocate_float as *const u8,
            true,
            true,
            3,
        )?;

        self.add_builtin_function(
            "beagle.builtin/fill-object-fields",
            fill_object_fields as *const u8,
            false,
            2,
        )?;

        // copy_object now takes (stack_pointer, frame_pointer, object_pointer)
        self.add_builtin_function_with_fp(
            "beagle.builtin/copy-object",
            copy_object as *const u8,
            true,
            true,
            3,
        )?;

        self.add_builtin_function(
            "beagle.builtin/copy-from-to-object",
            copy_from_to_object as *const u8,
            false,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/copy-array-range",
            copy_array_range as *const u8,
            false,
            4,
        )?;

        // make_closure now takes (stack_pointer, frame_pointer, function, num_free, free_variable_pointer)
        self.add_builtin_function_with_fp(
            "beagle.builtin/make-closure",
            make_closure as *const u8,
            true,
            true,
            5,
        )?;

        self.add_builtin_function(
            "beagle.builtin/property-access",
            property_access as *const u8,
            false,
            3,
        )?;

        self.add_builtin_function(
            "beagle.builtin/protocol-dispatch",
            protocol_dispatch as *const u8,
            false,
            3, // first_arg, cache_location, dispatch_table_ptr
        )?;

        // type_of now takes (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp(
            "beagle.core/type-of",
            type_of as *const u8,
            true,
            true,
            3,
        )?;

        // get_os now takes (stack_pointer, frame_pointer)
        self.add_builtin_function_with_fp(
            "beagle.core/get-os",
            get_os as *const u8,
            true,
            true,
            2,
        )?;

        self.add_builtin_function("beagle.core/equal", equal as *const u8, false, 2)?;

        // write_field now takes (stack_pointer, frame_pointer, struct_pointer, str_constant_ptr, property_cache_location, value)
        self.add_builtin_function_with_fp(
            "beagle.builtin/write-field",
            write_field as *const u8,
            true,
            true,
            6, // stack_pointer + frame_pointer + 4 original args
        )?;

        // throw_error now takes (stack_pointer, frame_pointer)
        self.add_builtin_function_with_fp(
            "beagle.builtin/throw-error",
            throw_error as *const u8,
            true,
            true,
            2,
        )?;

        // throw_type_error takes (stack_pointer, frame_pointer) and throws catchable TypeError
        self.add_builtin_function_with_fp(
            "beagle.builtin/throw-type-error",
            throw_type_error as *const u8,
            true,
            true,
            2,
        )?;

        // check_arity now takes (stack_pointer, frame_pointer, function_pointer, expected_args)
        self.add_builtin_function_with_fp(
            "beagle.builtin/check-arity",
            check_arity as *const u8,
            true,
            true,
            4,
        )?;

        self.add_builtin_function(
            "beagle.builtin/is-function-variadic",
            is_function_variadic as *const u8,
            false,
            1, // function_pointer
        )?;

        self.add_builtin_function(
            "beagle.builtin/get-function-min-args",
            get_function_min_args as *const u8,
            false,
            1, // function_pointer
        )?;

        // pack_variadic_args_from_stack now takes (stack_pointer, frame_pointer, args_base, total_args, min_args)
        self.add_builtin_function_with_fp(
            "beagle.builtin/pack-variadic-args-from-stack",
            pack_variadic_args_from_stack as *const u8,
            true,
            true,
            5,
        )?;

        // call_variadic_function_value now takes (stack_pointer, frame_pointer, function_ptr, args_array, is_closure, closure_ptr)
        self.add_builtin_function_with_fp(
            "beagle.builtin/call-variadic-function-value",
            call_variadic_function_value as *const u8,
            true,
            true,
            6,
        )?;

        self.add_builtin_function(
            "beagle.builtin/push-exception-handler",
            push_exception_handler_runtime as *const u8,
            false,
            5, // handler_address, result_local, link_register, stack_pointer, frame_pointer
        )?;

        self.add_builtin_function(
            "beagle.builtin/pop-exception-handler",
            pop_exception_handler_runtime as *const u8,
            false,
            0,
        )?;

        // throw_exception now takes (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp(
            "beagle.builtin/throw-exception",
            throw_exception as *const u8,
            true,
            true,
            3,
        )?;

        self.add_builtin_function(
            "beagle.core/set-thread-exception-handler!",
            set_thread_exception_handler as *const u8,
            false,
            1, // handler_fn
        )?;

        self.add_builtin_function(
            "beagle.core/set-default-exception-handler!",
            set_default_exception_handler as *const u8,
            false,
            1, // handler_fn
        )?;

        self.add_builtin_function(
            "beagle.builtin/create-error",
            create_error as *const u8,
            true,
            4, // stack_pointer, kind_str, message_str, location_str
        )?;

        self.add_builtin_function("beagle.builtin/assert!", placeholder as *const u8, false, 0)?;

        // gc needs both stack_pointer and frame_pointer
        // stack_pointer is arg 0, frame_pointer is arg 1
        self.add_builtin_function_with_fp("beagle.core/gc", gc as *const u8, true, true, 2)?;

        // Math builtins - all now take (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp(
            "beagle.core/sqrt",
            sqrt_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/floor",
            floor_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/ceil",
            ceil_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/abs",
            abs_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/sin",
            sin_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/cos",
            cos_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/to-float",
            to_float_builtin as *const u8,
            true,
            true,
            3,
        )?;

        // new_thread now takes (stack_pointer, frame_pointer, function)
        self.add_builtin_function_with_fp(
            "beagle.core/thread",
            new_thread as *const u8,
            true,
            true,
            3,
        )?;

        self.add_builtin_function(
            "beagle.ffi/load-library",
            load_library as *const u8,
            false,
            1,
        )?;

        // get_function now takes (stack_pointer, frame_pointer, library_struct, function_name, types, return_type)
        self.add_builtin_function_with_fp(
            "beagle.ffi/get-function",
            get_function as *const u8,
            true,
            true,
            6,
        )?;

        self.add_builtin_function(
            "beagle.ffi/call-ffi-info",
            call_ffi_info as *const u8,
            true,
            8,
        )?;

        self.add_builtin_function("beagle.ffi/allocate", ffi_allocate as *const u8, false, 1)?;
        self.add_builtin_function(
            "beagle.ffi/deallocate",
            ffi_deallocate as *const u8,
            false,
            1,
        )?;

        self.add_builtin_function("beagle.ffi/get-u32", ffi_get_u32 as *const u8, false, 2)?;

        self.add_builtin_function("beagle.ffi/set-i16", ffi_set_i16 as *const u8, false, 3)?;

        self.add_builtin_function("beagle.ffi/set-i32", ffi_set_i32 as *const u8, false, 3)?;

        self.add_builtin_function("beagle.ffi/set-u8", ffi_set_u8 as *const u8, false, 3)?;

        self.add_builtin_function("beagle.ffi/get-u8", ffi_get_u8 as *const u8, false, 2)?;

        self.add_builtin_function("beagle.ffi/get-i32", ffi_get_i32 as *const u8, false, 2)?;

        self.add_builtin_function(
            "beagle.ffi/get-string",
            ffi_get_string as *const u8,
            true,
            4,
        )?;

        self.add_builtin_function(
            "beagle.ffi/create-array",
            ffi_create_array as *const u8,
            true,
            3,
        )?;

        self.add_builtin_function(
            "beagle.ffi/copy-bytes",
            ffi_copy_bytes as *const u8,
            false,
            5,
        )?;

        self.add_builtin_function("beagle.ffi/realloc", ffi_realloc as *const u8, false, 2)?;

        self.add_builtin_function(
            "beagle.ffi/buffer-size",
            ffi_buffer_size as *const u8,
            false,
            1,
        )?;

        self.add_builtin_function(
            "beagle.ffi/write-buffer-offset",
            ffi_write_buffer_offset as *const u8,
            false,
            4,
        )?;

        self.add_builtin_function(
            "beagle.ffi/translate-bytes",
            ffi_translate_bytes as *const u8,
            false,
            4,
        )?;

        self.add_builtin_function(
            "beagle.ffi/reverse-bytes",
            ffi_reverse_bytes as *const u8,
            false,
            3,
        )?;

        self.add_builtin_function("beagle.ffi/find-byte", ffi_find_byte as *const u8, false, 4)?;

        self.add_builtin_function(
            "beagle.ffi/copy-bytes_filter",
            ffi_copy_bytes_filter as *const u8,
            false,
            6,
        )?;

        // __pause needs both stack_pointer and frame_pointer for FP-chain walking
        self.add_builtin_function_with_fp(
            "beagle.builtin/__pause",
            __pause as *const u8,
            true,
            true,
            2,
        )?;

        // register_c_call needs both stack_pointer and frame_pointer for FP-chain walking
        self.add_builtin_function_with_fp(
            "beagle.builtin/__register_c_call",
            register_c_call as *const u8,
            true,
            true,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/__unregister_c_call",
            unregister_c_call as *const u8,
            false,
            0,
        )?;

        self.add_builtin_function(
            "beagle.builtin/__run_thread",
            run_thread as *const u8,
            false,
            1,
        )?;

        // __get_my_thread_obj needs stack_pointer and frame_pointer to call __pause
        self.add_builtin_function_with_fp(
            "beagle.builtin/__get_my_thread_obj",
            get_my_thread_obj as *const u8,
            true,
            true,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/__get_and_unregister_temp_root",
            get_and_unregister_temp_root as *const u8,
            false,
            1,
        )?;

        self.add_builtin_function(
            "beagle.builtin/update-binding",
            update_binding as *const u8,
            false,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/get-binding",
            get_binding as *const u8,
            false,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/set-current-namespace",
            set_current_namespace as *const u8,
            false,
            1,
        )?;

        // wait_for_input now takes (stack_pointer, frame_pointer)
        self.add_builtin_function_with_fp(
            "beagle.builtin/wait-for-input",
            wait_for_input as *const u8,
            true,
            true,
            2,
        )?;

        // read_line now takes (stack_pointer, frame_pointer)
        self.add_builtin_function_with_fp(
            "beagle.builtin/read-line",
            read_line as *const u8,
            true,
            true,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/print-byte",
            print_byte as *const u8,
            false,
            1,
        )?;

        // char_code now takes (stack_pointer, frame_pointer, string)
        self.add_builtin_function_with_fp(
            "beagle.builtin/char-code",
            char_code as *const u8,
            true,
            true,
            3,
        )?;

        // char_from_code now takes (stack_pointer, frame_pointer, code)
        self.add_builtin_function_with_fp(
            "beagle.builtin/char-from-code",
            char_from_code as *const u8,
            true,
            true,
            3,
        )?;

        // eval now takes (stack_pointer, frame_pointer, code)
        self.add_builtin_function_with_fp("beagle.core/eval", eval as *const u8, true, true, 3)?;

        self.add_builtin_function("beagle.core/sleep", sleep as *const u8, false, 1)?;

        self.add_builtin_function("beagle.core/time-now", time_now as *const u8, false, 0)?;

        self.add_builtin_function(
            "beagle.builtin/register-extension",
            register_extension as *const u8,
            false,
            4,
        )?;

        // get_string_index now takes (stack_pointer, frame_pointer, string, index)
        self.add_builtin_function_with_fp(
            "beagle.builtin/get-string-index",
            get_string_index as *const u8,
            true,
            true,
            4,
        )?;

        self.add_builtin_function(
            "beagle.builtin/get-string-length",
            get_string_length as *const u8,
            false,
            1,
        )?;

        // string_concat now takes (stack_pointer, frame_pointer, a, b)
        self.add_builtin_function_with_fp(
            "beagle.core/string-concat",
            string_concat as *const u8,
            true,
            true,
            4,
        )?;

        // substring now takes (stack_pointer, frame_pointer, string, start, length)
        self.add_builtin_function_with_fp(
            "beagle.core/substring",
            substring as *const u8,
            true,
            true,
            5,
        )?;
        // uppercase now takes (stack_pointer, frame_pointer, string)
        self.add_builtin_function_with_fp(
            "beagle.core/uppercase",
            uppercase as *const u8,
            true,
            true,
            3,
        )?;

        // hash now takes (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp("beagle.builtin/hash", hash as *const u8, true, true, 3)?;

        self.add_builtin_function(
            "beagle.builtin/is-keyword",
            is_keyword as *const u8,
            false,
            1,
        )?;

        // keyword_to_string now takes (stack_pointer, frame_pointer, keyword)
        self.add_builtin_function_with_fp(
            "beagle.builtin/keyword-to-string",
            keyword_to_string as *const u8,
            true,
            true,
            3,
        )?;

        // string_to_keyword now takes (stack_pointer, frame_pointer, string_value)
        self.add_builtin_function_with_fp(
            "beagle.builtin/string-to-keyword",
            string_to_keyword as *const u8,
            true,
            true,
            3,
        )?;

        // load_keyword_constant_runtime now takes (stack_pointer, frame_pointer, index)
        self.add_builtin_function_with_fp(
            "beagle.builtin/load-keyword-constant-runtime",
            load_keyword_constant_runtime as *const u8,
            true,
            true,
            3,
        )?;

        // pop_count now takes (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp(
            "beagle.builtin/pop-count",
            pop_count as *const u8,
            true,
            true,
            3,
        )?;

        // read_full_file now takes (stack_pointer, frame_pointer, file_name)
        self.add_builtin_function_with_fp(
            "beagle.core/read-full-file",
            read_full_file as *const u8,
            true,
            true,
            3,
        )?;

        // compiler_warnings now takes (stack_pointer, frame_pointer)
        self.add_builtin_function_with_fp(
            "beagle.core/compiler-warnings",
            compiler_warnings as *const u8,
            true,
            true,
            2,
        )?;

        self.install_rust_collection_builtins()?;

        Ok(())
    }
}

/// Allocates a Beagle struct from Rust using struct registry lookup.
///
/// WARNING: This function is NOT GC-safe if fields contain heap pointers!
/// The allocation can trigger GC, making the field values stale.
/// For GC-safe struct allocation, allocate first, then peek roots, then write fields.
///
/// # Arguments
/// * `struct_name` - Fully qualified name like "beagle.core/CompilerWarning"
/// * `fields` - Slice of pre-tagged Beagle values (must match struct field count)
///
/// # Returns
/// Tagged pointer to the allocated struct
#[allow(dead_code)]
unsafe fn allocate_struct(
    runtime: &mut Runtime,
    stack_pointer: usize,
    struct_name: &str,
    fields: &[usize],
) -> Result<usize, String> {
    // Look up struct definition from registry
    let (struct_id, struct_def) = runtime
        .get_struct(struct_name)
        .ok_or_else(|| format!("Struct {} not found", struct_name))?;

    let struct_id = BuiltInTypes::Int.tag(struct_id as isize) as usize;

    // Validate field count matches struct definition
    if fields.len() != struct_def.fields.len() {
        return Err(format!(
            "Expected {} fields for {}, got {}",
            struct_def.fields.len(),
            struct_name,
            fields.len()
        ));
    }

    // Allocate heap object (same as create_error line 1630-1632)
    let obj_ptr = runtime
        .allocate(fields.len(), stack_pointer, BuiltInTypes::HeapObject)
        .map_err(|e| format!("Allocation failed: {}", e))?;

    // Write struct_id to header's type_data field (same as create_error lines 1636-1653)
    let heap_obj = HeapObject::from_tagged(obj_ptr);

    let untagged = heap_obj.untagged();
    let header_ptr = untagged as *mut usize;

    // Write struct_id to type_data field (bytes 3-6) without changing other fields
    // Header layout (little-endian):
    //   Bits 0-7:   Byte 0 (flags)
    //   Bits 8-15:  Byte 1 (padding)
    //   Bits 16-23: Byte 2 (size) - MUST PRESERVE
    //   Bits 24-55: Bytes 3-6 (type_data) - WRITE HERE
    //   Bits 56-63: Byte 7 (type_id) - MUST PRESERVE
    unsafe {
        let current_header = *header_ptr;
        let mask = 0x00FFFFFFFF000000; // Mask for bits 24-55 (bytes 3-6, the type_data field)
        let shifted_type_id = struct_id << 24; // Shift to bit 24
        let new_header = (current_header & !mask) | shifted_type_id;
        *header_ptr = new_header;
    }
    // Write all fields (same as create_error lines 1656-1658)
    for (i, &field_value) in fields.iter().enumerate() {
        heap_obj.write_field(i as i32, field_value);
    }

    Ok(obj_ptr)
}

/// Converts a CompilerWarning to a Beagle struct.
/// Wrapper that ensures temporary roots are always cleaned up.
unsafe fn warning_to_struct(
    runtime: &mut Runtime,
    stack_pointer: usize,
    warning: &crate::compiler::CompilerWarning,
) -> Result<usize, String> {
    let mut temp_roots: Vec<usize> = Vec::new();

    // Do all the work that might fail
    // SAFETY: warning_to_struct_impl is unsafe for the same reasons as this function
    let result =
        unsafe { warning_to_struct_impl(runtime, stack_pointer, warning, &mut temp_roots) };

    // Always clean up temporary roots, whether success or failure
    for root_id in temp_roots {
        runtime.unregister_temporary_root(root_id);
    }

    result
}

/// Inner implementation that does the actual work.
/// Any early return via ? will be caught by the wrapper which cleans up temp_roots.
unsafe fn warning_to_struct_impl(
    runtime: &mut Runtime,
    stack_pointer: usize,
    warning: &crate::compiler::CompilerWarning,
    temp_roots: &mut Vec<usize>,
) -> Result<usize, String> {
    // Use line and column directly from the warning struct
    let line = warning.line;
    let column = warning.column;

    // Helper macro to allocate, register as temp root, and return the root INDEX
    // We store root IDs and retrieve updated values before use (GC safety)
    macro_rules! alloc_and_root {
        ($expr:expr) => {{
            let val: usize = $expr;
            let root_id = runtime.register_temporary_root(val);
            temp_roots.push(root_id);
            temp_roots.len() - 1 // Return the index into temp_roots
        }};
    }

    // Create kind string based on warning type
    let kind_str = match &warning.kind {
        crate::compiler::WarningKind::NonExhaustiveMatch { .. } => "NonExhaustiveMatch",
        crate::compiler::WarningKind::UnreachablePattern => "UnreachablePattern",
    };
    let kind_root_idx = alloc_and_root!(
        runtime
            .allocate_string(stack_pointer, kind_str.to_string())
            .map_err(|e| format!("Failed to create kind string: {}", e))?
            .into()
    );

    // Create file_name string
    let file_name_root_idx = alloc_and_root!(
        runtime
            .allocate_string(stack_pointer, warning.file_name.clone())
            .map_err(|e| format!("Failed to create file_name string: {}", e))?
            .into()
    );

    // Create message string
    let message_root_idx = alloc_and_root!(
        runtime
            .allocate_string(stack_pointer, warning.message.clone())
            .map_err(|e| format!("Failed to create message string: {}", e))?
            .into()
    );

    // Create line and column as tagged ints (no allocation, no rooting needed)
    let line_tagged = BuiltInTypes::Int.tag(line as isize) as usize;
    let column_tagged = BuiltInTypes::Int.tag(column as isize) as usize;

    // Handle optional fields based on warning kind
    let (enum_name_root_idx, missing_variants_root_idx) = match &warning.kind {
        crate::compiler::WarningKind::NonExhaustiveMatch {
            enum_name,
            missing_variants,
        } => {
            // Create enum_name string
            let enum_name_root_idx = alloc_and_root!(
                runtime
                    .allocate_string(stack_pointer, enum_name.clone())
                    .map_err(|e| format!("Failed to create enum_name string: {}", e))?
                    .into()
            );

            // Build persistent vector of variant strings (using Beagle implementation)
            let mut vec = unsafe { call_fn_1(runtime, "persistent-vector/vec", 0) };
            let mut vec_root_id = runtime.register_temporary_root(vec);
            // Track vec_root_id immediately so it gets cleaned up on early return
            temp_roots.push(vec_root_id);
            let vec_root_index = temp_roots.len() - 1;

            for variant in missing_variants {
                let variant_str: usize = runtime
                    .allocate_string(stack_pointer, variant.clone())
                    .map_err(|e| format!("Failed to create variant string: {}", e))?
                    .into();
                let variant_root_id = runtime.register_temporary_root(variant_str);
                // Get updated vec from root before calling push (GC may have moved it)
                vec = runtime.peek_temporary_root(vec_root_id);
                vec = unsafe { call_fn_2(runtime, "persistent-vector/push", vec, variant_str) };
                runtime.unregister_temporary_root(variant_root_id);
                // Update vec root to point to new vec
                runtime.unregister_temporary_root(vec_root_id);
                vec_root_id = runtime.register_temporary_root(vec);
                // Update the tracked root ID so cleanup uses the current one
                temp_roots[vec_root_index] = vec_root_id;
            }

            (Some(enum_name_root_idx), Some(vec_root_index))
        }
        crate::compiler::WarningKind::UnreachablePattern => {
            // Use null for both optional fields (no roots needed)
            (None, None)
        }
    };

    // Create line and column as tagged ints (no allocation needed)
    // These are safe since they don't require heap allocation

    // Allocate the struct FIRST (this can trigger GC)
    // Then peek all root values AFTER allocation
    let struct_ptr = {
        // Look up struct definition
        let (struct_id, struct_def) = runtime
            .get_struct("beagle.core/CompilerWarning")
            .ok_or_else(|| "Struct beagle.core/CompilerWarning not found".to_string())?;

        if struct_def.fields.len() != 7 {
            return Err(format!(
                "Expected 7 fields for CompilerWarning, got {}",
                struct_def.fields.len()
            ));
        }

        let struct_id_tagged = BuiltInTypes::Int.tag(struct_id as isize) as usize;

        // Allocate the struct - this can trigger GC!
        let obj_ptr = runtime
            .allocate(7, stack_pointer, BuiltInTypes::HeapObject)
            .map_err(|e| format!("Allocation failed: {}", e))?;

        // Write struct_id to header
        let heap_obj = HeapObject::from_tagged(obj_ptr);
        let untagged = heap_obj.untagged();
        let header_ptr = untagged as *mut usize;
        unsafe {
            let current_header = *header_ptr;
            let mask = 0x00FFFFFFFF000000;
            let shifted_type_id = struct_id_tagged << 24;
            let new_header = (current_header & !mask) | shifted_type_id;
            *header_ptr = new_header;
        }

        obj_ptr
    };

    // NOW peek all values from roots - AFTER allocation/GC
    // This is critical for GC correctness - we must get updated addresses
    let kind_tagged = runtime.peek_temporary_root(temp_roots[kind_root_idx]);
    let file_name_tagged = runtime.peek_temporary_root(temp_roots[file_name_root_idx]);
    let message_tagged = runtime.peek_temporary_root(temp_roots[message_root_idx]);
    let enum_name_tagged = enum_name_root_idx
        .map(|idx| runtime.peek_temporary_root(temp_roots[idx]))
        .unwrap_or(BuiltInTypes::null_value() as usize);
    let missing_variants_tagged = missing_variants_root_idx
        .map(|idx| runtime.peek_temporary_root(temp_roots[idx]))
        .unwrap_or(BuiltInTypes::null_value() as usize);

    // Write all fields to the struct
    let heap_obj = HeapObject::from_tagged(struct_ptr);
    heap_obj.write_field(0, kind_tagged);
    heap_obj.write_field(1, file_name_tagged);
    heap_obj.write_field(2, line_tagged);
    heap_obj.write_field(3, column_tagged);
    heap_obj.write_field(4, message_tagged);
    heap_obj.write_field(5, enum_name_tagged);
    heap_obj.write_field(6, missing_variants_tagged);

    Ok(struct_ptr)
}

pub unsafe extern "C" fn compiler_warnings(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Clone warnings to avoid holding the lock while processing
    let warnings = {
        let warnings_guard = runtime.compiler_warnings.lock().unwrap();
        warnings_guard.clone()
    };

    // Start with empty persistent vector (using Beagle implementation for compatibility)
    let mut vec = unsafe { call_fn_1(runtime, "persistent-vector/vec", 0) };

    // Register vec as a temporary root to protect it from GC during the loop
    let mut vec_root_id = runtime.register_temporary_root(vec);

    // Convert each warning to struct and add to persistent vector
    for warning in warnings.iter() {
        match unsafe { warning_to_struct(runtime, stack_pointer, warning) } {
            Ok(warning_struct) => {
                let warning_root_id = runtime.register_temporary_root(warning_struct);
                // Get updated values from roots before calling push (GC may have moved them)
                let vec_updated = runtime.peek_temporary_root(vec_root_id);
                let warning_struct_updated = runtime.peek_temporary_root(warning_root_id);
                vec = unsafe {
                    call_fn_2(
                        runtime,
                        "persistent-vector/push",
                        vec_updated,
                        warning_struct_updated,
                    )
                };
                runtime.unregister_temporary_root(warning_root_id);
                // Update the root to point to the new vec
                runtime.unregister_temporary_root(vec_root_id);
                vec_root_id = runtime.register_temporary_root(vec);
            }
            Err(e) => {
                // Log error but continue processing other warnings
                eprintln!("Warning: Failed to convert compiler warning: {}", e);
            }
        }
    }

    // Unregister final vec root before returning
    runtime.unregister_temporary_root(vec_root_id);

    vec
}

// ============================================================================
// Rust Collections Builtins
// ============================================================================

mod rust_collections {
    use super::*;
    use crate::collections::{GcHandle, PersistentMap, PersistentVec};

    /// Create an empty persistent vector
    /// Signature: (stack_pointer, frame_pointer) -> tagged_ptr
    pub unsafe extern "C" fn rust_vec_empty(stack_pointer: usize, frame_pointer: usize) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        match PersistentVec::empty(runtime, stack_pointer) {
            Ok(handle) => handle.as_tagged(),
            Err(e) => {
                eprintln!("rust_vec_empty error: {}", e);
                BuiltInTypes::null_value() as usize
            }
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
            Err(e) => {
                eprintln!("rust_vec_push error: {}", e);
                BuiltInTypes::null_value() as usize
            }
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
            Err(e) => {
                eprintln!("rust_vec_assoc error: {}", e);
                BuiltInTypes::null_value() as usize
            }
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
            Err(e) => {
                eprintln!("rust_map_empty error: {}", e);
                BuiltInTypes::null_value() as usize
            }
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
            Err(e) => {
                eprintln!("rust_map_assoc error: {}", e);
                BuiltInTypes::null_value() as usize
            }
        }
    }
}

impl Runtime {
    pub fn install_rust_collection_builtins(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use rust_collections::*;

        // Register the namespace so it can be imported
        self.reserve_namespace("beagle.collections".to_string());

        // rust-vec: Create empty vector (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/vec",
            rust_vec_empty as *const u8,
            true,
            true,
            2, // sp, fp (implicit to caller)
        )?;

        // rust-vec-count: Get count (no allocation needed)
        self.add_builtin_function(
            "beagle.collections/vec-count",
            rust_vec_count as *const u8,
            false,
            1, // vec
        )?;

        // rust-vec-get: Get by index (no allocation needed)
        self.add_builtin_function(
            "beagle.collections/vec-get",
            rust_vec_get as *const u8,
            false,
            2, // vec, index
        )?;

        // rust-vec-push: Push value (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/vec-push",
            rust_vec_push as *const u8,
            true,
            true,
            4, // sp, fp, vec, value
        )?;

        // rust-vec-assoc: Update at index (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/vec-assoc",
            rust_vec_assoc as *const u8,
            true,
            true,
            5, // sp, fp, vec, index, value
        )?;

        // ========== Map builtins ==========

        // rust-map: Create empty map (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/map",
            rust_map_empty as *const u8,
            true,
            true,
            2, // sp, fp
        )?;

        // rust-map-count: Get count (no allocation needed)
        self.add_builtin_function(
            "beagle.collections/map-count",
            rust_map_count as *const u8,
            false,
            1, // map
        )?;

        // rust-map-get: Get by key (no allocation needed)
        self.add_builtin_function(
            "beagle.collections/map-get",
            rust_map_get as *const u8,
            false,
            2, // map, key
        )?;

        // rust-map-assoc: Associate key-value (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/map-assoc",
            rust_map_assoc as *const u8,
            true,
            true,
            5, // sp, fp, map, key, value
        )?;

        Ok(())
    }
}
