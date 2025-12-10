use core::panic;
use std::{
    arch::asm,
    error::Error,
    ffi::{CStr, c_void},
    hash::{DefaultHasher, Hasher},
    mem::{self, transmute},
    slice::{from_raw_parts, from_raw_parts_mut},
    thread,
};

use libffi::{
    low::CodePtr,
    middle::{Cif, Type, arg},
};

use crate::{
    Message, debug_only,
    gc::{Allocator, STACK_SIZE},
    get_runtime,
    runtime::{FFIInfo, FFIType, RawPtr, Runtime, SyncWrapper},
    types::{BuiltInTypes, HeapObject},
};

use std::hash::Hash;
use std::hint::black_box;

pub unsafe extern "C" fn debug_stack_segments() -> usize {
    let runtime = get_runtime().get();

    println!("=== Stack Segments Debug ===");
    println!("Segment count: {}", runtime.get_stack_segment_count());

    for i in 0..runtime.get_stack_segment_count() {
        if let Some(segment) = runtime.get_stack_segment(i) {
            println!("Segment {}: {} bytes", i, segment.data.len());

            // Show raw bytes in hex format
            print!("Raw bytes: ");
            for (j, byte) in segment.data.iter().enumerate() {
                if j > 0 && j % 8 == 0 {
                    print!(" ");
                }
                if j > 0 && j % 32 == 0 {
                    println!();
                    print!("           ");
                }
                print!("{:02x}", byte);
            }
            println!();

            // Try to interpret as usize values (8-byte chunks)
            if segment.data.len() >= 8 {
                println!("As usize values:");
                for chunk in segment.data.chunks_exact(8) {
                    let value = u64::from_le_bytes(chunk.try_into().unwrap()) as usize;
                    println!("  0x{:016x} ({})", value, value);
                }
            }
            println!();
        }
    }

    0b111 // Return success
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

pub fn debugger(message: Message) {
    debug_only! {
        let serialized_message : Vec<u8>;
        #[cfg(feature="json")] {
        use nanoserde::SerJson;
            let serialized : String = SerJson::serialize_json(&message);
            serialized_message = serialized.into_bytes();
        }
        #[cfg(not(feature="json"))] {
            use crate::Serialize;
            serialized_message = message.to_binary();
        }
        let message = serialized_message;
        let ptr = message.as_ptr();
        let length = message.len();
        mem::forget(message);
        unsafe {
            debugger_info(ptr, length);
        }
        #[allow(unused)]
        let message = unsafe { from_raw_parts(ptr, length) };
        // Should make it is so we clean up this memory
    }
}

pub unsafe extern "C" fn println_value(value: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let result = runtime.println(value);
    if let Err(error) = result {
        let stack_pointer = get_current_stack_pointer();
        println!("Error: {:?}", error);
        unsafe { throw_error(stack_pointer) };
    }
    0b111
}

pub unsafe extern "C" fn to_string(stack_pointer: usize, value: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let result = runtime.get_repr(value, 0);
    if result.is_none() {
        let stack_pointer = get_current_stack_pointer();
        unsafe { throw_error(stack_pointer) };
    }
    let result = result.unwrap();
    runtime
        .allocate_string(stack_pointer, result)
        .unwrap()
        .into()
}

pub unsafe extern "C" fn to_number(stack_pointer: usize, value: usize) -> usize {
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
fn print_call_builtin(runtime: &Runtime, name: &str) {
    debug_only!(if runtime.get_command_line_args().print_builtin_calls {
        println!("Calling: {}", name);
    });
}

pub unsafe extern "C" fn print_value(value: usize) -> usize {
    let runtime = get_runtime().get_mut();
    runtime.print(value);
    0b111
}

extern "C" fn allocate(stack_pointer: usize, size: usize) -> usize {
    let size = BuiltInTypes::untag(size);
    let runtime = get_runtime().get_mut();

    let result = runtime
        .allocate(size, stack_pointer, BuiltInTypes::HeapObject)
        .unwrap();

    debug_assert!(BuiltInTypes::is_heap_pointer(result));
    debug_assert!(BuiltInTypes::untag(result) % 8 == 0);
    result
}

extern "C" fn allocate_float(stack_pointer: usize, size: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let value = BuiltInTypes::untag(size);

    let result = runtime
        .allocate(value, stack_pointer, BuiltInTypes::Float)
        .unwrap();

    debug_assert!(BuiltInTypes::get_kind(result) == BuiltInTypes::Float);
    debug_assert!(BuiltInTypes::untag(result) % 8 == 0);
    result
}

extern "C" fn get_string_index(stack_pointer: usize, string: usize, index: usize) -> usize {
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

extern "C" fn string_concat(stack_pointer: usize, a: usize, b: usize) -> usize {
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

extern "C" fn substring(stack_pointer: usize, string: usize, start: usize, length: usize) -> usize {
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

extern "C" fn fill_object_fields(object_pointer: usize, value: usize) -> usize {
    print_call_builtin(get_runtime().get(), "fill_object_fields");
    let mut object = HeapObject::from_tagged(object_pointer);
    let raw_slice = object.get_fields_mut();
    raw_slice.fill(value);
    object_pointer
}

extern "C" fn make_closure(
    stack_pointer: usize,
    function: usize,
    num_free: usize,
    free_variable_pointer: usize,
) -> usize {
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
            let heap_object = HeapObject::from_tagged(struct_pointer);
            println!("Heap object: {:?}", heap_object.get_header());
            println!("Error: {:?}", error);
            unsafe {
                throw_error(stack_pointer);
            };
        });
    let type_id = HeapObject::from_tagged(struct_pointer).get_struct_id();
    let buffer = unsafe { from_raw_parts_mut(property_cache_location as *mut usize, 2) };
    buffer[0] = type_id;
    buffer[1] = index * 8;
    result
}

extern "C" fn type_of(stack_pointer: usize, value: usize) -> usize {
    print_call_builtin(get_runtime().get(), "type_of");
    let runtime = get_runtime().get_mut();
    runtime.type_of(stack_pointer, value).unwrap()
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
    struct_pointer: usize,
    str_constant_ptr: usize,
    property_cache_location: usize,
    value: usize,
) -> usize {
    let runtime = get_runtime().get_mut();
    let index = runtime.write_field(stack_pointer, struct_pointer, str_constant_ptr, value);
    let type_id = HeapObject::from_tagged(struct_pointer).get_struct_id();
    let buffer = unsafe { from_raw_parts_mut(property_cache_location as *mut usize, 2) };
    buffer[0] = type_id;
    buffer[1] = index * 8;
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn throw_error(stack_pointer: usize) -> ! {
    print_stack(stack_pointer);
    panic!("Error!");
}

pub unsafe extern "C" fn check_arity(
    stack_pointer: usize,
    function_pointer: usize,
    expected_args: isize,
) -> isize {
    let runtime = get_runtime().get();

    // Function pointer is tagged, need to untag
    let untagged_ptr = (function_pointer >> BuiltInTypes::tag_size()) as *const u8;

    if let Some(function) = runtime.get_function_by_pointer(untagged_ptr) {
        // expected_args is a tagged integer, need to untag it
        let expected_args_untagged = BuiltInTypes::untag(expected_args as usize);
        if function.number_of_args != expected_args_untagged {
            println!(
                "Arity mismatch for '{}': expected {} args, got {}",
                function.name, function.number_of_args, expected_args_untagged
            );
            unsafe {
                throw_error(stack_pointer);
            }
        }
    }

    0 // Return value unused
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

pub unsafe extern "C" fn gc(stack_pointer: usize) -> usize {
    let runtime = get_runtime().get_mut();
    runtime.gc(stack_pointer);
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn gc_add_root(old: usize) -> usize {
    let runtime = get_runtime().get_mut();
    runtime.gc_add_root(old);
    BuiltInTypes::null_value() as usize
}

#[allow(unused)]
pub unsafe extern "C" fn new_thread(stack_pointer: usize, function: usize) -> usize {
    #[cfg(feature = "thread-safe")]
    {
        let runtime = get_runtime().get_mut();
        runtime.new_thread(function);
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
    runtime.memory.add_namespace_root(namespace_id, value);
    runtime.update_binding(namespace_id, namespace_slot, value);
    BuiltInTypes::null_value() as usize
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_binding(namespace: usize, slot: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let namespace = BuiltInTypes::untag(namespace);
    let slot = BuiltInTypes::untag(slot);
    runtime.get_binding(namespace, slot)
}

pub unsafe extern "C" fn set_current_namespace(namespace: usize) -> usize {
    let runtime = get_runtime().get_mut();
    runtime.set_current_namespace(namespace);
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn __pause(stack_pointer: usize) -> usize {
    let runtime = get_runtime().get_mut();

    pause_current_thread(stack_pointer, runtime);

    while runtime.is_paused() {
        // Park can unpark itself even if I haven't called unpark
        thread::park();
    }

    // Apparently, I can't count on this not unparking
    // I need some other mechanism to know that things are ready
    unpause_current_thread(runtime);

    BuiltInTypes::null_value() as usize
}

fn pause_current_thread(stack_pointer: usize, runtime: &mut Runtime) {
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    let stack_base = runtime.get_stack_base();
    state.pause((stack_base, stack_pointer));
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

pub extern "C" fn register_c_call(stack_pointer: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    let stack_base = runtime.get_stack_base();
    state.register_c_call((stack_base, stack_pointer));
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

pub unsafe fn call_fn_1(runtime: &Runtime, function_name: &str, arg1: usize) -> usize {
    print_call_builtin(
        runtime,
        format!("{} {}", "call_fn_1", function_name).as_str(),
    );
    let save_volatile_registers = runtime
        .get_function_by_name("beagle.builtin/save_volatile_registers1")
        .unwrap();
    let save_volatile_registers = runtime.get_pointer(save_volatile_registers).unwrap();
    let save_volatile_registers: fn(usize, usize) -> usize =
        unsafe { std::mem::transmute(save_volatile_registers) };

    let function = runtime.get_function_by_name(function_name).unwrap();
    let function = runtime.get_pointer(function).unwrap();
    save_volatile_registers(arg1, function as usize)
}

pub unsafe fn call_fn_2(runtime: &Runtime, function_name: &str, arg1: usize, arg2: usize) -> usize {
    print_call_builtin(
        runtime,
        format!("{} {}", "call_fn_2", function_name).as_str(),
    );
    let save_volatile_registers = runtime
        .get_function_by_name("beagle.builtin/save_volatile_registers2")
        .unwrap();
    let save_volatile_registers = runtime.get_pointer(save_volatile_registers).unwrap();
    let save_volatile_registers: fn(usize, usize, usize) -> usize =
        unsafe { std::mem::transmute(save_volatile_registers) };

    let function = runtime.get_function_by_name(function_name).unwrap();
    let function = runtime.get_pointer(function).unwrap();
    save_volatile_registers(arg1, arg2, function as usize)
}

// Helper to call a Beagle function or closure with one argument
pub unsafe fn call_beagle_fn_ptr(runtime: &Runtime, fn_or_closure: usize, arg1: usize) {
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

pub fn map_ffi_type(runtime: &Runtime, value: usize) -> Result<Type, String> {
    let heap_object = HeapObject::from_tagged(value);
    let struct_id = BuiltInTypes::untag(heap_object.get_struct_id());
    let struct_info = runtime.get_struct_by_id(struct_id);

    if struct_info.is_none() {
        return Err(format!("Could not find struct with id {}", struct_id));
    }

    let struct_info = struct_info.unwrap();
    let name = struct_info.name.as_str().split_once("/").unwrap().1;
    match name {
        "Type.U8" => Ok(Type::u8()),
        "Type.U16" => Ok(Type::u16()),
        "Type.U32" => Ok(Type::u32()),
        "Type.U64" => Ok(Type::u64()),
        "Type.I32" => Ok(Type::i32()),
        "Type.Pointer" => Ok(Type::pointer()),
        "Type.MutablePointer" => Ok(Type::pointer()),
        "Type.String" => Ok(Type::pointer()),
        "Type.Void" => Ok(Type::void()),
        "Type.Structure" => {
            let types = heap_object.get_field(0);
            let types = array_to_vec(persistent_vector_to_array(runtime, types));
            let fields: Result<Vec<Type>, String> =
                types.iter().map(|t| map_ffi_type(runtime, *t)).collect();
            Ok(Type::structure(fields?))
        }
        _ => Err(format!("Unknown type: {}", name)),
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
            let types = array_to_vec(persistent_vector_to_array(runtime, types));
            let fields: Result<Vec<FFIType>, String> = types
                .iter()
                .map(|t| map_beagle_type_to_ffi_type(runtime, *t))
                .collect();
            Ok(FFIType::Structure(fields?))
        }
        _ => Err(format!("Unknown type: {}", name)),
    }
}

fn persistent_vector_to_array(_runtime: &Runtime, vector: usize) -> HeapObject {
    // TODO: This isn't actually a safe thing to do. It allocates
    // which means that any pointers I had now could have moved.
    // I also am not sure I can from one of these runtime
    // functions end up in another runtime function
    // Because then I have multiple mutable references it runtime.
    let tagged = unsafe { call_fn_1(_runtime, "persistent_vector/to_array", vector) };
    HeapObject::from_tagged(tagged)
}

fn array_to_vec(object: HeapObject) -> Vec<usize> {
    object.get_fields().to_vec()
}

// TODO:
// I need to get the elements of this vector into
// a rust vector and then map the types

pub extern "C" fn get_function(
    stack_pointer: usize,
    library_struct: usize,
    function_name: usize,
    types: usize,
    return_type: usize,
) -> usize {
    let runtime = get_runtime().get_mut();
    let library = runtime.get_library(library_struct);
    let function_name = runtime.get_string_literal(function_name);

    // TODO: I should actually cache the closure, but I don't want to do that and mess up gc
    if let Some(ffi_info_id) = runtime.find_ffi_info_by_name(&function_name) {
        let ffi_info_id = BuiltInTypes::Int.tag(ffi_info_id as isize) as usize;
        return unsafe { call_fn_1(runtime, "beagle.ffi/__create_ffi_function", ffi_info_id) };
    }

    let func_ptr = unsafe { library.get::<fn()>(function_name.as_bytes()).unwrap() };

    let code_ptr = unsafe { CodePtr(func_ptr.try_as_raw_ptr().unwrap()) };

    // use std::ffi::c_void;
    // let code_ptr = if function_name == "SBTargetBreakpointCreateByName" {
    //     CodePtr(create_breakpointer_placeholder as *mut c_void)
    // } else {
    //     unsafe { CodePtr(func_ptr.try_as_raw_ptr().unwrap()) }
    // };

    let types: Vec<usize> = array_to_vec(persistent_vector_to_array(runtime, types));

    let lib_ffi_types: Result<Vec<Type>, String> =
        types.iter().map(|t| map_ffi_type(runtime, *t)).collect();
    let lib_ffi_types = match lib_ffi_types {
        Ok(types) => types,
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "FFIError", e);
        },
    };
    let number_of_arguments = lib_ffi_types.len();

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

    let lib_ffi_return_type = match map_ffi_type(runtime, return_type) {
        Ok(t) => t,
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "FFIError", e);
        },
    };
    let ffi_return_type = match map_beagle_type_to_ffi_type(runtime, return_type) {
        Ok(t) => t,
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "FFIError", e);
        },
    };

    let cif = Cif::new(lib_ffi_types, lib_ffi_return_type.clone());

    let ffi_info_id = runtime.add_ffi_function_info(FFIInfo {
        name: function_name.to_string(),
        function: RawPtr::new(code_ptr.0 as *const u8),
        cif: SyncWrapper::new(cif),
        number_of_arguments,
        argument_types: beagle_ffi_types,
        return_type: ffi_return_type,
    });
    runtime.add_ffi_info_by_name(function_name, ffi_info_id);
    let ffi_info_id = BuiltInTypes::Int.tag(ffi_info_id as isize) as usize;

    unsafe { call_fn_1(runtime, "beagle.ffi/__create_ffi_function", ffi_info_id) }
}

// In general, this code doesn't work with release mode... :(

// TODO:
// thread 'main' panicked at /Users/jimmyhmiller/.cargo/registry/src/index.crates.io-6f17d22bba15001f/libffi-3.2.0/src/middle/types.rs:151:8:
// misaligned pointer dereference: address must be a multiple of 0x8 but is 0x88263ae042be000d
// stack backtrace:
//    0: rust_begin_unwind
//              at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library/std/src/panicking.rs:652:5
//    1: core::panicking::panic_nounwind_fmt::runtime
//              at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library/core/src/panicking.rs:110:18
//    2: core::panicking::panic_nounwind_fmt
//              at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library/core/src/panicking.rs:120:5
//    3: core::panicking::panic_misaligned_pointer_dereference
//              at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library/core/src/panicking.rs:287:5
//    4: libffi::middle::types::ffi_type_clone
//              at /Users/jimmyhmiller/.cargo/registry/src/index.crates.io-6f17d22bba15001f/libffi-3.2.0/src/middle/types.rs:151:8
//    5: libffi::middle::types::ffi_type_array_clone
//              at /Users/jimmyhmiller/.cargo/registry/src/index.crates.io-6f17d22bba15001f/libffi-3.2.0/src/middle/types.rs:143:23
//    6: <libffi::middle::types::TypeArray as core::clone::Clone>::clone
//              at /Users/jimmyhmiller/.cargo/registry/src/index.crates.io-6f17d22bba15001f/libffi-3.2.0/src/middle/types.rs:204:40
//    7: <libffi::middle::Cif as core::clone::Clone>::clone
//              at /Users/jimmyhmiller/.cargo/registry/src/index.crates.io-6f17d22bba15001f/libffi-3.2.0/src/middle/mod.rs:91:19
//    8: <main::runtime::FFIInfo as core::clone::Clone>::clone
//              at ./src/runtime.rs:149:5
//    9: main::builtins::call_ffi_info
//              at ./src/builtins.rs:439:20
// note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
// thread caused non-unwinding panic. aborting.

// TODO: Fix this to allow multiple arguments
// instead of hardcoding 0
pub unsafe extern "C" fn call_ffi_info(
    stack_pointer: usize,
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
        let ffi_info = runtime.get_ffi_info(ffi_info_id).clone();
        let code_ptr = ffi_info.function;
        let arguments = [a1, a2, a3, a4, a5, a6];
        let args = &arguments[..ffi_info.number_of_arguments];
        let argument_types = ffi_info.argument_types;
        let mut argument_pointers = vec![];

        for (argument, ffi_type) in args.iter().zip(argument_types.iter()) {
            let kind = BuiltInTypes::get_kind(*argument);
            match kind {
                BuiltInTypes::Null => {
                    if ffi_type != &FFIType::Pointer {
                        throw_runtime_error(
                            stack_pointer,
                            "FFIError",
                            format!("Expected pointer type, got {:?}", ffi_type),
                        );
                    }
                    argument_pointers.push(arg(&std::ptr::null_mut::<c_void>()));
                }
                BuiltInTypes::String => {
                    if ffi_type != &FFIType::String {
                        throw_runtime_error(
                            stack_pointer,
                            "FFIError",
                            format!("Expected string type, got {:?}", ffi_type),
                        );
                    }
                    let string = runtime.get_string_literal(*argument);
                    let string = runtime.memory.write_c_string(string);
                    let pointer = runtime.memory.write_pointer(string as usize);
                    argument_pointers.push(arg(pointer));
                }
                BuiltInTypes::Int => match ffi_type {
                    FFIType::U8 => {
                        let pointer = runtime
                            .memory
                            .write_u8(BuiltInTypes::untag(*argument) as u8);
                        argument_pointers.push(arg(pointer));
                    }
                    FFIType::U16 => {
                        let pointer = runtime
                            .memory
                            .write_u16(BuiltInTypes::untag(*argument) as u16);
                        argument_pointers.push(arg(pointer));
                    }
                    FFIType::U32 => {
                        let pointer = runtime
                            .memory
                            .write_u32(BuiltInTypes::untag(*argument) as u32);
                        argument_pointers.push(arg(pointer));
                    }
                    FFIType::U64 => {
                        let pointer = runtime
                            .memory
                            .write_u64(BuiltInTypes::untag(*argument) as u64);
                        argument_pointers.push(arg(pointer));
                    }
                    FFIType::I32 => {
                        let pointer = runtime
                            .memory
                            .write_i32(BuiltInTypes::untag(*argument) as i32);
                        argument_pointers.push(arg(pointer));
                    }

                    FFIType::Pointer => {
                        if BuiltInTypes::untag(*argument) == 0 {
                            argument_pointers.push(arg(&std::ptr::null_mut::<c_void>()));
                        } else {
                            let heap_object = HeapObject::from_tagged(*argument);
                            let buffer = BuiltInTypes::untag(heap_object.get_field(0));
                            let pointer = runtime.memory.write_pointer(buffer);
                            argument_pointers.push(arg(pointer));
                        }
                    }

                    FFIType::MutablePointer
                    | FFIType::String
                    | FFIType::Void
                    | FFIType::Structure(_) => {
                        throw_runtime_error(
                            stack_pointer,
                            "FFIError",
                            format!("Expected integer for FFI type {:?}", ffi_type),
                        );
                    }
                },
                BuiltInTypes::HeapObject => {
                    match ffi_type {
                        FFIType::Pointer | FFIType::MutablePointer => {
                            let heap_object = HeapObject::from_tagged(*argument);
                            let buffer = BuiltInTypes::untag(heap_object.get_field(0));
                            let pointer = runtime.memory.write_pointer(buffer);
                            argument_pointers.push(arg(pointer));
                        }
                        FFIType::Structure(_types) => {
                            // We are going to asume for now now that we pass a buffer.
                            // We are going to write that buffer to our memory and then pass the pointer
                            // like we would any number
                            let heap_object = HeapObject::from_tagged(*argument);
                            let buffer = BuiltInTypes::untag(heap_object.get_field(0));
                            let size = BuiltInTypes::untag(heap_object.get_field(1));
                            let pointer = runtime.memory.write_buffer(buffer, size);
                            argument_pointers.push(arg(pointer));
                        }
                        FFIType::String => {
                            let string = runtime.get_string(stack_pointer, *argument);
                            let string = runtime.memory.write_c_string(string);
                            let pointer = runtime.memory.write_pointer(string as usize);
                            argument_pointers.push(arg(pointer));
                        }
                        _ => {
                            throw_runtime_error(
                                stack_pointer,
                                "FFIError",
                                format!(
                                    "Got HeapObject but expected matching FFI type, got {:?}",
                                    ffi_type
                                ),
                            );
                        }
                    }
                }
                _ => {
                    runtime.print(*argument);
                    throw_runtime_error(
                        stack_pointer,
                        "FFIError",
                        format!("Unsupported FFI type: {:?}", kind),
                    );
                }
            }
        }

        let return_value = match ffi_info.return_type {
            FFIType::Void => {
                ffi_info
                    .cif
                    .get()
                    .call::<()>(CodePtr(code_ptr.ptr as *mut c_void), &argument_pointers);
                BuiltInTypes::null_value() as usize
            }
            FFIType::U8 => {
                let result = ffi_info
                    .cif
                    .get()
                    .call::<u8>(CodePtr(code_ptr.ptr as *mut c_void), &argument_pointers);
                BuiltInTypes::Int.tag(result as isize) as usize
            }
            FFIType::U16 => {
                let result = ffi_info
                    .cif
                    .get()
                    .call::<u16>(CodePtr(code_ptr.ptr as *mut c_void), &argument_pointers);
                BuiltInTypes::Int.tag(result as isize) as usize
            }
            FFIType::U32 => {
                let result = ffi_info
                    .cif
                    .get()
                    .call::<u32>(CodePtr(code_ptr.ptr as *mut c_void), &argument_pointers);
                BuiltInTypes::Int.tag(result as isize) as usize
            }
            FFIType::U64 => {
                let result = ffi_info
                    .cif
                    .get()
                    .call::<u64>(CodePtr(code_ptr.ptr as *mut c_void), &argument_pointers);
                BuiltInTypes::Int.tag(result as isize) as usize
            }
            FFIType::I32 => {
                let result = ffi_info
                    .cif
                    .get()
                    .call::<i32>(CodePtr(code_ptr.ptr as *mut c_void), &argument_pointers);
                BuiltInTypes::Int.tag(result as isize) as usize
            }
            FFIType::Pointer => {
                let result = ffi_info
                    .cif
                    .get()
                    .call::<*mut u8>(CodePtr(code_ptr.ptr as *mut c_void), &argument_pointers);
                let pointer_value = BuiltInTypes::Int.tag(result as isize) as usize;
                call_fn_1(runtime, "beagle.ffi/__make_pointer_struct", pointer_value)
            }
            FFIType::MutablePointer => {
                let result = ffi_info
                    .cif
                    .get()
                    .call::<*mut u8>(CodePtr(code_ptr.ptr as *mut c_void), &argument_pointers);
                let pointer_value = BuiltInTypes::Int.tag(result as isize) as usize;
                call_fn_1(runtime, "beagle.ffi/__make_pointer_struct", pointer_value)
            }
            FFIType::String => {
                let result = ffi_info
                    .cif
                    .get()
                    .call::<*mut u8>(CodePtr(code_ptr.ptr as *mut c_void), &argument_pointers);
                if result.is_null() {
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
                todo!()
            }
        };
        runtime.memory.clear_native_arguments();
        return_value
    }
}

pub unsafe extern "C" fn copy_object(stack_pointer: usize, object_pointer: usize) -> usize {
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
        unsafe { throw_error(stack_pointer) };
    } else {
        result.unwrap()
    }
}

pub unsafe extern "C" fn copy_from_to_object(from: usize, to: usize) -> usize {
    let runtime = get_runtime().get_mut();
    if from == BuiltInTypes::null_value() as usize {
        return to;
    }
    // runtime.gc_add_root(from);
    // runtime.gc_add_root(to);
    let from = HeapObject::from_tagged(from);
    let mut to = HeapObject::from_tagged(to);
    runtime.copy_object_except_header(from, &mut to).unwrap();
    to.tagged_pointer()
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

extern "C" fn placeholder() -> usize {
    BuiltInTypes::null_value() as usize
}

extern "C" fn wait_for_input(stack_pointer: usize) -> usize {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let runtime = get_runtime().get_mut();
    let string = runtime.allocate_string(stack_pointer, input);
    string.unwrap().into()
}

extern "C" fn read_full_file(stack_pointer: usize, file_name: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let file_name = runtime.get_string(stack_pointer, file_name);
    let file = std::fs::read_to_string(file_name).unwrap();
    let string = runtime.allocate_string(stack_pointer, file);
    string.unwrap().into()
}

extern "C" fn eval(stack_pointer: usize, code: usize) -> usize {
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

extern "C" fn hash(stack_pointer: usize, value: usize) -> usize {
    print_call_builtin(get_runtime().get(), "hash");
    let tag = BuiltInTypes::get_kind(value);
    match tag {
        BuiltInTypes::Int => {
            let mut s = DefaultHasher::new();
            value.hash(&mut s);
            BuiltInTypes::Int.tag(s.finish() as isize) as usize
        }
        BuiltInTypes::HeapObject => {
            let heap_object = HeapObject::from_tagged(value);
            if heap_object.get_header().type_id == 2 {
                let bytes = heap_object.get_string_bytes();
                let string = unsafe { std::str::from_utf8_unchecked(bytes) };
                let mut s = DefaultHasher::new();
                string.hash(&mut s);
                return BuiltInTypes::Int.tag(s.finish() as isize) as usize;
            } else if heap_object.get_header().type_id == 3 {
                // Keywords: return cached hash (stable across GC)
                let hash = heap_object.get_keyword_hash();
                return BuiltInTypes::Int.tag(hash as isize) as usize;
            }
            let fields = heap_object.get_fields();
            let mut s = DefaultHasher::new();
            for field in fields {
                field.hash(&mut s);
            }
            BuiltInTypes::Int.tag(s.finish() as isize) as usize
        }
        BuiltInTypes::String => {
            let runtime = get_runtime().get_mut();
            let string = runtime.get_string_literal(value);
            let mut s = DefaultHasher::new();
            string.hash(&mut s);
            BuiltInTypes::Int.tag(s.finish() as isize) as usize
        }
        _ => unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!("Expected int or heap object for hash, got {:?}", tag),
            );
        },
    }
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

pub extern "C" fn keyword_to_string(stack_pointer: usize, keyword: usize) -> usize {
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

pub extern "C" fn string_to_keyword(stack_pointer: usize, string_value: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let keyword_text = runtime.get_string(stack_pointer, string_value);

    // Use intern_keyword to ensure same text = same pointer
    runtime.intern_keyword(stack_pointer, keyword_text).unwrap()
}

pub extern "C" fn load_keyword_constant_runtime(stack_pointer: usize, index: usize) -> usize {
    let runtime = get_runtime().get_mut();

    // Check if we already allocated this keyword
    if let Some(ptr) = runtime.keyword_heap_ptrs[index] {
        return ptr;
    }

    // Allocate and register as GC root
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

extern "C" fn pop_count(stack_pointer: usize, value: usize) -> usize {
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

pub unsafe extern "C" fn throw_exception(stack_pointer: usize, value: usize) -> ! {
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
            unsafe { throw_error(stack_pointer) };
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
            unsafe { throw_error(stack_pointer) };
        }

        // No handler at all - panic with stack trace
        println!("Uncaught exception:");
        get_runtime().get_mut().println(exception).ok();
        unsafe { throw_error(stack_pointer) };
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
    // Allocate strings and create error in a scoped block to avoid aliasing
    let (kind_str, message_str) = {
        let runtime = get_runtime().get_mut();
        let kind_str = runtime
            .allocate_string(stack_pointer, kind.to_string())
            .expect("Failed to allocate kind string")
            .into();
        let message_str = runtime
            .allocate_string(stack_pointer, message)
            .expect("Failed to allocate message string")
            .into();
        (kind_str, message_str)
    };
    // Runtime borrow is dropped here

    let null_location = BuiltInTypes::Null.tag(0) as usize;

    // Create the Error struct and throw it
    unsafe {
        let error = create_error(stack_pointer, kind_str, message_str, null_location);
        throw_exception(stack_pointer, error);
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

        self.add_builtin_function("beagle.core/to_string", to_string as *const u8, true, 2)?;
        self.add_builtin_function("beagle.core/to_number", to_number as *const u8, true, 2)?;

        self.add_builtin_function("beagle.builtin/allocate", allocate as *const u8, true, 2)?;

        self.add_builtin_function(
            "beagle.builtin/allocate_float",
            allocate_float as *const u8,
            true,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/fill_object_fields",
            fill_object_fields as *const u8,
            false,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/copy_object",
            copy_object as *const u8,
            true,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/copy_from_to_object",
            copy_from_to_object as *const u8,
            false,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/make_closure",
            make_closure as *const u8,
            true,
            4,
        )?;

        self.add_builtin_function(
            "beagle.builtin/property_access",
            property_access as *const u8,
            false,
            3,
        )?;

        self.add_builtin_function("beagle.core/type-of", type_of as *const u8, true, 2)?;

        self.add_builtin_function("beagle.core/equal", equal as *const u8, false, 2)?;

        self.add_builtin_function(
            "beagle.builtin/write_field",
            write_field as *const u8,
            true, // Now takes stack_pointer
            5,    // stack_pointer + 4 original args
        )?;

        self.add_builtin_function(
            "beagle.builtin/throw_error",
            throw_error as *const u8,
            true,
            1,
        )?;

        self.add_builtin_function(
            "beagle.builtin/check_arity",
            check_arity as *const u8,
            true, // needs_stack_pointer
            3,    // stack_pointer, function_pointer, expected_args
        )?;

        self.add_builtin_function(
            "beagle.builtin/push_exception_handler",
            push_exception_handler_runtime as *const u8,
            false,
            5, // handler_address, result_local, link_register, stack_pointer, frame_pointer
        )?;

        self.add_builtin_function(
            "beagle.builtin/pop_exception_handler",
            pop_exception_handler_runtime as *const u8,
            false,
            0,
        )?;

        self.add_builtin_function(
            "beagle.builtin/throw_exception",
            throw_exception as *const u8,
            true,
            2,
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
            "beagle.builtin/create_error",
            create_error as *const u8,
            true,
            4, // stack_pointer, kind_str, message_str, location_str
        )?;

        self.add_builtin_function("beagle.builtin/assert!", placeholder as *const u8, false, 0)?;

        self.add_builtin_function("beagle.core/gc", gc as *const u8, true, 1)?;

        self.add_builtin_function(
            "beagle.debug/stack_segments",
            debug_stack_segments as *const u8,
            false,
            0,
        )?;

        self.add_builtin_function(
            "beagle.builtin/gc_add_root",
            gc_add_root as *const u8,
            false,
            1,
        )?;

        self.add_builtin_function("beagle.core/thread", new_thread as *const u8, true, 2)?;

        self.add_builtin_function(
            "beagle.ffi/load_library",
            load_library as *const u8,
            false,
            1,
        )?;

        self.add_builtin_function(
            "beagle.ffi/get_function",
            get_function as *const u8,
            true,
            5,
        )?;

        self.add_builtin_function(
            "beagle.ffi/call_ffi_info",
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

        self.add_builtin_function("beagle.ffi/get_u32", ffi_get_u32 as *const u8, false, 2)?;

        self.add_builtin_function("beagle.ffi/set_i16", ffi_set_i16 as *const u8, false, 3)?;

        self.add_builtin_function("beagle.ffi/set_i32", ffi_set_i32 as *const u8, false, 3)?;

        self.add_builtin_function("beagle.ffi/set_u8", ffi_set_u8 as *const u8, false, 3)?;

        self.add_builtin_function("beagle.ffi/get_i32", ffi_get_i32 as *const u8, false, 2)?;

        self.add_builtin_function(
            "beagle.ffi/get_string",
            ffi_get_string as *const u8,
            true,
            4,
        )?;

        self.add_builtin_function(
            "beagle.ffi/create_array",
            ffi_create_array as *const u8,
            true,
            3,
        )?;

        self.add_builtin_function("beagle.builtin/__pause", __pause as *const u8, true, 1)?;

        self.add_builtin_function(
            "beagle.builtin/__register_c_call",
            register_c_call as *const u8,
            true,
            1,
        )?;

        self.add_builtin_function(
            "beagle.builtin/__unregister_c_call",
            unregister_c_call as *const u8,
            false,
            0,
        )?;

        self.add_builtin_function(
            "beagle.builtin/update_binding",
            update_binding as *const u8,
            false,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/get_binding",
            get_binding as *const u8,
            false,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/set_current_namespace",
            set_current_namespace as *const u8,
            false,
            1,
        )?;

        self.add_builtin_function(
            "beagle.builtin/wait_for_input",
            wait_for_input as *const u8,
            true,
            1,
        )?;

        self.add_builtin_function("beagle.core/eval", eval as *const u8, true, 2)?;

        self.add_builtin_function("beagle.core/sleep", sleep as *const u8, false, 1)?;

        self.add_builtin_function(
            "beagle.builtin/register_extension",
            register_extension as *const u8,
            false,
            4,
        )?;

        self.add_builtin_function(
            "beagle.builtin/get_string_index",
            get_string_index as *const u8,
            true,
            3,
        )?;

        self.add_builtin_function(
            "beagle.builtin/get_string_length",
            get_string_length as *const u8,
            false,
            1,
        )?;

        self.add_builtin_function(
            "beagle.core/string_concat",
            string_concat as *const u8,
            true,
            3,
        )?;

        self.add_builtin_function("beagle.core/substring", substring as *const u8, true, 4)?;

        self.add_builtin_function("beagle.builtin/hash", hash as *const u8, true, 2)?;

        self.add_builtin_function(
            "beagle.builtin/is_keyword",
            is_keyword as *const u8,
            false,
            1,
        )?;

        self.add_builtin_function(
            "beagle.builtin/keyword_to_string",
            keyword_to_string as *const u8,
            true,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/string_to_keyword",
            string_to_keyword as *const u8,
            true,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/load_keyword_constant_runtime",
            load_keyword_constant_runtime as *const u8,
            true,
            2,
        )?;

        self.add_builtin_function("beagle.builtin/pop_count", pop_count as *const u8, true, 2)?;

        self.add_builtin_function(
            "beagle.core/read_full_file",
            read_full_file as *const u8,
            true,
            2,
        )?;

        self.add_builtin_function(
            "beagle.core/compiler-warnings",
            compiler_warnings as *const u8,
            true,
            1,
        )?;

        Ok(())
    }
}

/// Allocates a Beagle struct from Rust using struct registry lookup.
///
/// # Arguments
/// * `struct_name` - Fully qualified name like "beagle.core/CompilerWarning"
/// * `fields` - Slice of pre-tagged Beagle values (must match struct field count)
///
/// # Returns
/// Tagged pointer to the allocated struct
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
        let shifted_type_id = (struct_id as usize) << 24; // Shift to bit 24
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
unsafe fn warning_to_struct(
    runtime: &mut Runtime,
    stack_pointer: usize,
    warning: &crate::compiler::CompilerWarning,
) -> Result<usize, String> {
    // Use line and column directly from the warning struct
    let line = warning.line;
    let column = warning.column;

    // Create kind string based on warning type
    let kind_str = match &warning.kind {
        crate::compiler::WarningKind::NonExhaustiveMatch { .. } => "NonExhaustiveMatch",
        crate::compiler::WarningKind::UnreachablePattern => "UnreachablePattern",
    };
    let kind_tagged = runtime
        .allocate_string(stack_pointer, kind_str.to_string())
        .map_err(|e| format!("Failed to create kind string: {}", e))?
        .into();

    // Create file_name string
    let file_name_tagged = runtime
        .allocate_string(stack_pointer, warning.file_name.clone())
        .map_err(|e| format!("Failed to create file_name string: {}", e))?
        .into();

    // Create message string
    let message_tagged = runtime
        .allocate_string(stack_pointer, warning.message.clone())
        .map_err(|e| format!("Failed to create message string: {}", e))?
        .into();

    // Create line and column as tagged ints
    let line_tagged = BuiltInTypes::Int.tag(line as isize) as usize;
    let column_tagged = BuiltInTypes::Int.tag(column as isize) as usize;

    // Handle optional fields based on warning kind
    let (enum_name_tagged, missing_variants_tagged) = match &warning.kind {
        crate::compiler::WarningKind::NonExhaustiveMatch {
            enum_name,
            missing_variants,
        } => {
            // Create enum_name string
            let enum_name_str = runtime
                .allocate_string(stack_pointer, enum_name.clone())
                .map_err(|e| format!("Failed to create enum_name string: {}", e))?
                .into();

            // Build persistent vector of variant strings
            let empty_vec = unsafe { call_fn_1(runtime, "persistent_vector/vec", 0) };
            let mut vec = empty_vec;
            for variant in missing_variants {
                let variant_str: usize = runtime
                    .allocate_string(stack_pointer, variant.clone())
                    .map_err(|e| format!("Failed to create variant string: {}", e))?
                    .into();
                vec = unsafe { call_fn_2(runtime, "persistent_vector/push", vec, variant_str) };
            }

            (enum_name_str, vec)
        }
        crate::compiler::WarningKind::UnreachablePattern => {
            // Use null for both optional fields
            (
                BuiltInTypes::null_value() as usize,
                BuiltInTypes::null_value() as usize,
            )
        }
    };

    // Allocate struct with all 7 fields in order
    let fields = [
        kind_tagged,
        file_name_tagged,
        line_tagged,
        column_tagged,
        message_tagged,
        enum_name_tagged,
        missing_variants_tagged,
    ];

    unsafe {
        allocate_struct(
            runtime,
            stack_pointer,
            "beagle.core/CompilerWarning",
            &fields,
        )
    }
}

pub unsafe extern "C" fn compiler_warnings(stack_pointer: usize) -> usize {
    let runtime = get_runtime().get_mut();

    // Clone warnings to avoid holding the lock while processing
    let warnings = {
        let warnings_guard = runtime.compiler_warnings.lock().unwrap();
        warnings_guard.clone()
    };

    // Start with empty persistent vector
    let mut vec = unsafe { call_fn_1(runtime, "persistent_vector/vec", 0) };

    // Convert each warning to struct and add to persistent vector
    for (_, warning) in warnings.iter().enumerate() {
        match unsafe { warning_to_struct(runtime, stack_pointer, warning) } {
            Ok(warning_struct) => {
                let root_id = runtime.register_temporary_root(warning_struct);
                vec = unsafe { call_fn_2(runtime, "persistent_vector/push", vec, warning_struct) };
                runtime.unregister_temporary_root(root_id);
            }
            Err(e) => {
                // Log error but continue processing other warnings
                eprintln!("Warning: Failed to convert compiler warning: {}", e);
            }
        }
    }

    vec
}
