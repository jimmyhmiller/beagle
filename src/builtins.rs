use core::panic;
use std::{
    error::Error,
    ffi::c_void,
    mem::{self, transmute},
    slice::{from_raw_parts, from_raw_parts_mut},
    thread,
};

use libffi::{
    low::CodePtr,
    middle::{arg, Cif, Type},
};

use crate::{
    gc::Allocator,
    runtime::{FFIInfo, FFIType, Runtime},
    types::{BuiltInTypes, HeapObject},
    Message, Serialize,
};

use std::hint::black_box;

#[allow(unused)]
#[no_mangle]
#[inline(never)]
/// # Safety
///
/// This does nothing
pub unsafe extern "C" fn debugger_info(buffer: *const u8, length: usize) {
    // Hack to make sure this isn't inlined
    black_box(buffer);
    black_box(length);
}

pub fn debugger(message: Message) {
    let message = message.to_binary();
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

pub unsafe extern "C" fn println_value<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    value: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    runtime.println(value);
    0b111
}

pub unsafe extern "C" fn print_value<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    value: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    runtime.print(value);
    0b111
}

extern "C" fn allocate<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    stack_pointer: usize,
    size: usize,
) -> usize {
    let size = BuiltInTypes::untag(size);
    let runtime = unsafe { &mut *runtime };

    let result = runtime
        .allocate(size, stack_pointer, BuiltInTypes::HeapObject)
        .unwrap();

    debug_assert!(BuiltInTypes::is_heap_pointer(result));
    debug_assert!(BuiltInTypes::untag(result) % 8 == 0);
    result
}

extern "C" fn allocate_float<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    stack_pointer: usize,
    size: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    let value = BuiltInTypes::untag(size);

    let result = runtime
        .allocate(value, stack_pointer, BuiltInTypes::Float)
        .unwrap();

    debug_assert!(BuiltInTypes::get_kind(result) == BuiltInTypes::Float);
    debug_assert!(BuiltInTypes::untag(result) % 8 == 0);
    result
}

extern "C" fn fill_object_fields<Alloc: Allocator>(
    _runtime: *mut Runtime<Alloc>,
    object_pointer: usize,
    value: usize,
) -> usize {
    let mut object = HeapObject::from_tagged(object_pointer);
    let raw_slice = object.get_fields_mut();
    raw_slice.fill(value);
    object_pointer
}

extern "C" fn make_closure<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    stack_pointer: usize,
    function: usize,
    num_free: usize,
    free_variable_pointer: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    if BuiltInTypes::get_kind(function) != BuiltInTypes::Function {
        panic!(
            "Expected function, got {:?}",
            BuiltInTypes::get_kind(function)
        );
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

extern "C" fn property_access<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    struct_pointer: usize,
    str_constant_ptr: usize,
    property_cache_location: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    let (result, index) = runtime
        .compiler
        .property_access(struct_pointer, str_constant_ptr);
    let type_id = HeapObject::from_tagged(struct_pointer).get_struct_id();
    let buffer = unsafe { from_raw_parts_mut(property_cache_location as *mut usize, 2) };
    buffer[0] = type_id;
    buffer[1] = index * 8;
    result
}

pub unsafe extern "C" fn throw_error<Alloc: Allocator>(
    _runtime: *mut Runtime<Alloc>,
    _stack_pointer: usize,
) -> usize {
    // let compiler = unsafe { &mut *compiler };
    panic!("Error!");
}

pub unsafe extern "C" fn gc<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    stack_pointer: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    runtime.gc(stack_pointer);
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn gc_add_root<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    old: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    runtime.gc_add_root(old);
    BuiltInTypes::null_value() as usize
}

#[allow(unused)]
pub unsafe extern "C" fn new_thread<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    function: usize,
) -> usize {
    #[cfg(feature = "thread-safe")]
    {
        let runtime = unsafe { &mut *runtime };
        runtime.new_thread(function);
        BuiltInTypes::null_value() as usize
    }
    #[cfg(not(feature = "thread-safe"))]
    {
        panic!("Threads are not supported in this build");
    }
}

pub unsafe extern "C" fn update_binding<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    namespace_slot: usize,
    value: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    let namespace_id = runtime.compiler.current_namespace_id();
    runtime.memory.add_namespace_root(namespace_id, value);
    runtime.compiler.update_binding(namespace_slot, value);
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn get_binding<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    namespace: usize,
    slot: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };

    runtime.compiler.get_binding(namespace, slot)
}

pub unsafe extern "C" fn set_current_namespace<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    namespace: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    runtime.compiler.set_current_namespace(namespace);
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn __pause<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    stack_pointer: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };

    pause_current_thread(stack_pointer, runtime);

    while runtime.is_paused() {
        // Park can unpark itself even if I haven't called unpark
        thread::park();
    }

    unpause_current_thread(runtime);

    // Apparently, I can't count on this not unparking
    // I need some other mechanism to know that things are ready
    BuiltInTypes::null_value() as usize
}

fn pause_current_thread<Alloc: Allocator>(stack_pointer: usize, runtime: &mut Runtime<Alloc>) {
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    let stack_base = runtime.get_stack_base();
    state.pause((stack_base, stack_pointer));
    condvar.notify_one();
    drop(state);
}

fn unpause_current_thread<Alloc: Allocator>(runtime: &mut Runtime<Alloc>) {
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    state.unpause();
    condvar.notify_one();
}

pub extern "C" fn register_c_call<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    stack_pointer: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    let stack_base = runtime.get_stack_base();
    state.register_c_call((stack_base, stack_pointer));
    condvar.notify_one();
    BuiltInTypes::null_value() as usize
}

pub extern "C" fn unregister_c_call<Alloc: Allocator>(runtime: *mut Runtime<Alloc>) -> usize {
    let runtime = unsafe { &mut *runtime };
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

pub unsafe fn call_fn_1<Alloc: Allocator>(
    runtime: &Runtime<Alloc>,
    function_name: &str,
    arg1: usize,
) -> usize {
    let save_volatile_registers = runtime
        .compiler
        .get_function_by_name("beagle.builtin/save_volatile_registers")
        .unwrap();
    let save_volatile_registers = runtime
        .compiler
        .get_pointer(save_volatile_registers)
        .unwrap();
    let save_volatile_registers: fn(usize, usize) -> usize =
        std::mem::transmute(save_volatile_registers);

    let function = runtime
        .compiler
        .get_function_by_name(function_name)
        .unwrap();
    let function = runtime.compiler.get_pointer(function).unwrap();
    save_volatile_registers(arg1, function)
}

pub unsafe extern "C" fn load_library<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    name: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    let string = &runtime.compiler.get_string(name);
    let lib = libloading::Library::new(string).unwrap();
    let id = runtime.add_library(lib);

    call_fn_1(
        runtime,
        "beagle.ffi/__make_lib_struct",
        BuiltInTypes::Int.tag(id as isize) as usize,
    )
}

pub fn map_ffi_type<Alloc: Allocator>(runtime: &Runtime<Alloc>, value: usize) -> Type {
    let heap_object = HeapObject::from_tagged(value);
    let struct_id = BuiltInTypes::untag(heap_object.get_struct_id());
    let struct_info = runtime
        .compiler
        .get_struct_by_id(struct_id)
        .unwrap_or_else(|| panic!("Could not find struct with id {}", struct_id));
    let name = struct_info.name.as_str().split_once("/").unwrap().1;
    match name {
        "Type.U8" => Type::u8(),
        "Type.U16" => Type::u16(),
        "Type.U32" => Type::u32(),
        "Type.I32" => Type::i32(),
        "Type.Pointer" => Type::pointer(),
        "Type.MutablePointer" => Type::pointer(),
        "Type.String" => Type::pointer(),
        "Type.Void" => Type::void(),
        _ => panic!("Unknown type: {}", name),
    }
}

pub fn map_beagle_type_to_ffi_type<Alloc: Allocator>(
    runtime: &Runtime<Alloc>,
    value: usize,
) -> FFIType {
    let heap_object = HeapObject::from_tagged(value);
    let struct_id = BuiltInTypes::untag(heap_object.get_struct_id());
    let struct_info = runtime
        .compiler
        .get_struct_by_id(struct_id)
        .unwrap_or_else(|| panic!("Could not find struct with id {}", struct_id));
    let name = struct_info.name.as_str().split_once("/").unwrap().1;
    match name {
        "Type.U8" => FFIType::U8,
        "Type.U16" => FFIType::U16,
        "Type.U32" => FFIType::U32,
        "Type.I32" => FFIType::I32,
        "Type.Pointer" => FFIType::Pointer,
        "Type.MutablePointer" => FFIType::MutablePointer,
        "Type.String" => FFIType::String,
        "Type.Void" => FFIType::Void,
        _ => panic!("Unknown type: {}", name),
    }
}

fn persistent_vector_to_array<Alloc: Allocator>(
    runtime: &Runtime<Alloc>,
    vector: usize,
) -> HeapObject {
    // TODO: This isn't actually a safe thing to do. It allocates
    // which means that any pointers I had now could have moved.
    // I also am not sure I can from one of these runtime
    // functions end up in another runtime function
    // Because then I have multiple mutable references it runtime.
    let tagged = unsafe { call_fn_1(runtime, "persistent_vector/to_array", vector) };
    HeapObject::from_tagged(tagged)
}

fn array_to_vec(object: HeapObject) -> Vec<usize> {
    object.get_fields().to_vec()
}

#[allow(unused)]
extern "C" fn create_window_placeholder(
    title: *const i8,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    flags: u32,
) -> usize {
    let title = unsafe { std::ffi::CStr::from_ptr(title).to_str().unwrap() };
    println!("Arguments {:?}", (title, x, y, w, h, flags));
    0
}

#[allow(unused)]
extern "C" fn sdl_render_fill_rect_placeholder(renderer: *const u32, rect: *const u32) -> usize {
    println!("Arguments {:?}", (renderer, rect));
    0
}

#[allow(unused)]
extern "C" fn sdl_poll_event(buffer: *const u32) -> usize {
    println!("Arguments {:?}", buffer);
    0
}

// TODO:
// I need to get the elements of this vector into
// a rust vector and then map the types

pub extern "C" fn get_function<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    library_struct: usize,
    function_name: usize,
    types: usize,
    return_type: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    let library = runtime.get_library(library_struct);
    let function_name = runtime.compiler.get_string(function_name);

    // TODO: I should actually cache the closure, but I don't want to do that and mess up gc
    if let Some(ffi_info_id) = runtime.find_ffi_info_by_name(&function_name) {
        let ffi_info_id = BuiltInTypes::Int.tag(ffi_info_id as isize) as usize;
        return unsafe { call_fn_1(runtime, "beagle.ffi/__create_ffi_function", ffi_info_id) };
    }

    let func_ptr = unsafe { library.get::<fn()>(function_name.as_bytes()).unwrap() };

    let code_ptr = unsafe { CodePtr(func_ptr.try_as_raw_ptr().unwrap()) };

    // use std::ffi::c_void;
    // let code_ptr = if function_name == "SDL_RenderFillRect" {
    //     CodePtr(sdl_render_fill_rect_placeholder as *mut c_void)
    // } else {
    //     unsafe { CodePtr(func_ptr.try_as_raw_ptr().unwrap()) }
    // };

    let types: Vec<usize> = array_to_vec(persistent_vector_to_array(runtime, types)).to_vec();

    let lib_ffi_types: Vec<Type> = types.iter().map(|t| map_ffi_type(runtime, *t)).collect();
    let number_of_arguments = lib_ffi_types.len();

    let beagle_ffi_types = types
        .iter()
        .map(|t| map_beagle_type_to_ffi_type(runtime, *t))
        .collect::<Vec<_>>();

    let lib_ffi_return_type = map_ffi_type(runtime, return_type);
    let ffi_return_type = map_beagle_type_to_ffi_type(runtime, return_type);

    let cif = Cif::new(lib_ffi_types, lib_ffi_return_type.clone());

    let ffi_info_id = runtime.add_ffi_function_info(FFIInfo {
        name: function_name.to_string(),
        function: code_ptr,
        cif,
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
pub unsafe extern "C" fn call_ffi_info<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    ffi_info_id: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
    a6: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
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
            BuiltInTypes::String => {
                if ffi_type != &FFIType::String {
                    panic!("Expected string, got {:?}", ffi_type);
                }
                let string = runtime.compiler.get_string(*argument);
                let string = runtime.memory.write_c_string(string);
                argument_pointers.push(arg(&string));
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
                FFIType::I32 => {
                    let pointer = runtime
                        .memory
                        .write_i32(BuiltInTypes::untag(*argument) as i32);
                    argument_pointers.push(arg(pointer));
                }

                FFIType::Pointer | FFIType::MutablePointer | FFIType::String | FFIType::Void => {
                    panic!("Expected pointer, got {:?}", ffi_type);
                }
            },
            BuiltInTypes::HeapObject => {
                if ffi_type != &FFIType::Pointer && ffi_type != &FFIType::MutablePointer {
                    panic!("Expected pointer, got {:?}", ffi_type);
                }
                // TODO: Make this type safe
                let heap_object = HeapObject::from_tagged(*argument);
                let buffer = BuiltInTypes::untag(heap_object.get_field(0));
                let pointer = runtime.memory.write_pointer(buffer);
                argument_pointers.push(arg(pointer));
            }
            _ => {
                runtime.print(*argument);
                panic!("Unsupported type: {:?}", kind)
            }
        }
    }

    let return_value = match ffi_info.return_type {
        FFIType::Void => {
            ffi_info.cif.call::<()>(code_ptr, &argument_pointers);
            BuiltInTypes::null_value() as usize
        }
        FFIType::U8 => {
            let result = ffi_info.cif.call::<u8>(code_ptr, &argument_pointers);
            BuiltInTypes::Int.tag(result as isize) as usize
        }
        FFIType::U16 => {
            let result = ffi_info.cif.call::<u16>(code_ptr, &argument_pointers);
            BuiltInTypes::Int.tag(result as isize) as usize
        }
        FFIType::U32 => {
            let result = ffi_info.cif.call::<u32>(code_ptr, &argument_pointers);
            BuiltInTypes::Int.tag(result as isize) as usize
        }
        FFIType::I32 => {
            let result = ffi_info.cif.call::<i32>(code_ptr, &argument_pointers);
            BuiltInTypes::Int.tag(result as isize) as usize
        }
        FFIType::Pointer => {
            let result = ffi_info.cif.call::<*mut u8>(code_ptr, &argument_pointers);
            let pointer_value = BuiltInTypes::Int.tag(result as isize) as usize;
            call_fn_1(runtime, "beagle.ffi/__make_pointer_struct", pointer_value)
        }
        FFIType::MutablePointer => {
            let result = ffi_info.cif.call::<*mut u8>(code_ptr, &argument_pointers);
            let pointer_value = BuiltInTypes::Int.tag(result as isize) as usize;
            call_fn_1(runtime, "beagle.ffi/__make_pointer_struct", pointer_value)
        }
        FFIType::String => {
            todo!()
        }
    };
    runtime.memory.clear_native_arguments();
    return_value
}

pub unsafe extern "C" fn copy_object<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    stack_pointer: usize,
    object_pointer: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    // runtime.gc_add_root(object_pointer);
    let object = HeapObject::from_tagged(object_pointer);
    let header = object.get_header();
    let size = header.size as usize;
    let kind = BuiltInTypes::get_kind(object_pointer);
    let to_pointer = runtime.allocate(size, stack_pointer, kind).unwrap();
    let mut to_object = HeapObject::from_tagged(to_pointer);
    runtime.copy_object(object, &mut to_object).unwrap()
}

pub unsafe extern "C" fn copy_from_to_object<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    from: usize,
    to: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
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

unsafe extern "C" fn ffi_allocate<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    size: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    // TODO: I intentionally don't want to manage this memory on the heap
    // I probably need a better answer than this
    // but for now we are just going to leak memory
    let size = BuiltInTypes::untag(size);

    let mut buffer: Vec<u8> = vec![0; size];
    let buffer_ptr: *mut c_void = buffer.as_mut_ptr() as *mut c_void;
    std::mem::forget(buffer);

    let buffer = BuiltInTypes::Int.tag(buffer_ptr as isize) as usize;
    call_fn_1(runtime, "beagle.ffi/__make_pointer_struct", buffer)
}

unsafe extern "C" fn ffi_get_u32<Alloc: Allocator>(
    _runtime: *mut Runtime<Alloc>,
    buffer: usize,
    offset: usize,
) -> usize {
    // TODO: Make type safe
    let buffer_object = HeapObject::from_tagged(buffer);
    let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
    let offset = BuiltInTypes::untag(offset);
    let value = *(buffer.add(offset) as *const u32);
    BuiltInTypes::Int.tag(value as isize) as usize
}

unsafe extern "C" fn ffi_set_i32<Alloc: Allocator>(
    _runtime: *mut Runtime<Alloc>,
    buffer: usize,
    offset: usize,
    value: usize,
) -> usize {
    let buffer_object = HeapObject::from_tagged(buffer);
    let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
    let offset = BuiltInTypes::untag(offset);
    let value = BuiltInTypes::untag(value) as i32;
    *(buffer.add(offset) as *mut i32) = value;
    BuiltInTypes::null_value() as usize
}

unsafe extern "C" fn ffi_set_i16<Alloc: Allocator>(
    _runtime: *mut Runtime<Alloc>,
    buffer: usize,
    offset: usize,
    value: usize,
) -> usize {
    let buffer_object = HeapObject::from_tagged(buffer);
    let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
    let offset = BuiltInTypes::untag(offset);
    let value = BuiltInTypes::untag(value) as i16;
    *(buffer.add(offset) as *mut i16) = value;
    BuiltInTypes::null_value() as usize
}

unsafe extern "C" fn ffi_get_i32<Alloc: Allocator>(
    _runtime: *mut Runtime<Alloc>,
    buffer: usize,
    offset: usize,
) -> usize {
    let buffer_object = HeapObject::from_tagged(buffer);
    let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
    let offset = BuiltInTypes::untag(offset);
    let value = *(buffer.add(offset) as *const i32);
    BuiltInTypes::Int.tag(value as isize) as usize
}

unsafe extern "C" fn ffi_get_string<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    buffer: usize,
    offset: usize,
    len: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    let buffer_object = HeapObject::from_tagged(buffer);
    let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
    let offset = BuiltInTypes::untag(offset);
    let len = BuiltInTypes::untag(len);
    let slice = std::slice::from_raw_parts(buffer.add(offset), len);
    let string = std::str::from_utf8(slice).unwrap();
    runtime
        .memory
        .alloc_string(string.to_string())
        .unwrap()
        .into()
}

extern "C" fn placeholder() -> usize {
    BuiltInTypes::null_value() as usize
}

extern "C" fn wait_for_input<Alloc: Allocator>(runtime: *mut Runtime<Alloc>) -> usize {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let runtime = unsafe { &mut *runtime };
    let string = runtime.memory.alloc_string(input);
    string.unwrap().into()
}

extern "C" fn eval<Alloc: Allocator>(runtime: *mut Runtime<Alloc>, code: usize) -> usize {
    let runtime = unsafe { &mut *runtime };
    let code = match BuiltInTypes::get_kind(code) {
        BuiltInTypes::String => runtime.compiler.get_string(code),
        BuiltInTypes::HeapObject => {
            let code = HeapObject::from_tagged(code);
            assert!(code.get_header().type_id == 2);
            let bytes = code.get_string_bytes();
            let code = std::str::from_utf8(bytes).unwrap();
            code.to_string()
        }
        _ => panic!("Expected string, got {:?}", BuiltInTypes::get_kind(code)),
    };
    let result = runtime.compiler.compile_string(&code).unwrap();
    mem::forget(code);
    if result == 0 {
        return BuiltInTypes::null_value() as usize;
    }
    let f: fn() -> usize = unsafe { transmute(result) };
    f()
}

extern "C" fn sleep<Alloc: Allocator>(_runtime: *mut Runtime<Alloc>, time: usize) -> usize {
    let time = BuiltInTypes::untag(time);
    std::thread::sleep(std::time::Duration::from_millis(time as u64));
    BuiltInTypes::null_value() as usize
}

impl<Alloc: Allocator> Runtime<Alloc> {
    pub fn install_builtins(&mut self) -> Result<(), Box<dyn Error>> {
        self.compiler.add_builtin_function(
            "beagle.core/println",
            println_value::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.core/print",
            print_value::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/allocate",
            allocate::<Alloc> as *const u8,
            true,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/allocate_float",
            allocate_float::<Alloc> as *const u8,
            true,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/fill_object_fields",
            fill_object_fields::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/copy_object",
            copy_object::<Alloc> as *const u8,
            true,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/copy_from_to_object",
            copy_from_to_object::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/make_closure",
            make_closure::<Alloc> as *const u8,
            true,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/property_access",
            property_access::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/throw_error",
            throw_error::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/assert!",
            placeholder as *const u8,
            false,
        )?;

        self.compiler
            .add_builtin_function("beagle.core/gc", gc::<Alloc> as *const u8, true)?;

        self.compiler.add_builtin_function(
            "beagle.builtin/gc_add_root",
            gc_add_root::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.core/thread",
            new_thread::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.ffi/load_library",
            load_library::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.ffi/get_function",
            get_function::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.ffi/call_ffi_info",
            call_ffi_info::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.ffi/allocate",
            ffi_allocate::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.ffi/get_u32",
            ffi_get_u32::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.ffi/set_i16",
            ffi_set_i16::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.ffi/set_i32",
            ffi_set_i32::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.ffi/get_i32",
            ffi_get_i32::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.ffi/get_string",
            ffi_get_string::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/__pause",
            __pause::<Alloc> as *const u8,
            true,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/__register_c_call",
            register_c_call::<Alloc> as *const u8,
            true,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/__unregister_c_call",
            unregister_c_call::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/update_binding",
            update_binding::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/get_binding",
            get_binding::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/set_current_namespace",
            set_current_namespace::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.builtin/wait_for_input",
            wait_for_input::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.core/eval",
            eval::<Alloc> as *const u8,
            false,
        )?;

        self.compiler.add_builtin_function(
            "beagle.core/sleep",
            sleep::<Alloc> as *const u8,
            false,
        )?;

        Ok(())
    }
}
