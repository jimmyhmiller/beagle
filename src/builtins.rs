use core::panic;
use std::{
    error::Error, ffi::CString, mem, os::raw::c_void, slice::{from_raw_parts, from_raw_parts_mut}, thread
};

use libffi::{
    low::CodePtr,
    middle::{Arg, Cif, Type},
};

use crate::{
    gc::Allocator,
    runtime::{FFIInfo, Runtime},
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

    let num_free = BuiltInTypes::untag(num_free);
    let free_variable_pointer = free_variable_pointer as *const usize;
    let start = unsafe { free_variable_pointer.sub(num_free - 1) };
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

    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    let stack_base = runtime.get_stack_base();
    state.pause((stack_base, stack_pointer));
    condvar.notify_one();
    drop(state);

    while runtime.is_paused() {
        // Park can unpark itself even if I haven't called unpark
        thread::park();
    }

    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    state.unpause();
    condvar.notify_one();

    // Apparently, I can't count on this not unparking
    // I need some other mechanism to know that things are ready
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn load_library<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    name: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    let string = &runtime.compiler.get_string(name);
    let lib = libloading::Library::new(string).unwrap();
    let id = runtime.add_library(lib);

    let call_fn = runtime
        .compiler
        .get_function_by_name("beagle.ffi/__make_lib_struct")
        .unwrap();
    let function_pointer = runtime.compiler.get_pointer(call_fn).unwrap();
    let function: fn(usize) -> usize = std::mem::transmute(function_pointer);
    function(id)
}

#[allow(unused)]
pub fn map_ffi_type<Alloc: Allocator>(runtime: &Runtime<Alloc>, value: usize) -> Type {
    let heap_object = HeapObject::from_tagged(value);
    let struct_id = BuiltInTypes::untag(heap_object.get_struct_id());
    let struct_info = runtime
        .compiler
        .get_struct_by_id(struct_id)
        .unwrap_or_else(|| panic!("Could not find struct with id {}", struct_id));
    let name = struct_info.name.as_str().split_once("/").unwrap().1;
    match name {
        "Type.U32" => Type::u32(),
        "Type.I32" => Type::i32(),
        "Type.Pointer" => Type::pointer(),
        "Type.MutablePointer" => Type::pointer(),
        "Type.String" => Type::pointer(),
        "Type.Void" => Type::void(),
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
    let make_array = runtime
        .compiler
        .get_function_by_name("persistent_vector/to_array")
        .unwrap();
    let function_pointer = runtime.compiler.get_pointer(make_array).unwrap();
    let function: fn(usize) -> usize = unsafe { std::mem::transmute(function_pointer) };
    let tagged = function(vector);
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

// TODO:
// I need to get the elements of this vector into
// a rust vector and then map the types

pub unsafe extern "C" fn get_function<Alloc: Allocator>(
    runtime: *mut Runtime<Alloc>,
    library_struct: usize,
    function_name: usize,
    types: usize,
    return_type: usize,
) -> usize {
    let runtime = unsafe { &mut *runtime };
    let library = runtime.get_library(library_struct);
    let function_name = runtime.compiler.get_string(function_name);
    let func_ptr = unsafe { library.get::<fn()>(function_name.as_bytes()).unwrap() };

    let code_ptr =  unsafe { CodePtr(func_ptr.try_as_raw_ptr().unwrap()) };
    
    // let code_ptr = if function_name == "SDL_CreateWindow" {
    //     CodePtr(create_window_placeholder as *mut c_void)
    // } else {
    //     unsafe { CodePtr(func_ptr.try_as_raw_ptr().unwrap()) }
    // };

    // If I am going to call into the language from the runtime
    // I am going to need to have something that saves and restores
    // callee saved registers.
    // Right now I'm not, which is why in release mode return_type is breaking.

    let types: Vec<Type> = array_to_vec(persistent_vector_to_array(runtime, types))
        .iter()
        .map(|x| map_ffi_type(runtime, *x))
        .collect();
    let number_of_arguments = types.len();

    let return_type = map_ffi_type(runtime, return_type);

    let cif = Cif::new(types, return_type);

    let ffi_info_id = runtime.add_ffi_function_info(FFIInfo {
        function: code_ptr,
        cif,
        number_of_arguments,
    });
    let ffi_info_id = BuiltInTypes::Int.tag(ffi_info_id as isize) as usize;

    let create_ffi_function = runtime
        .compiler
        .get_function_by_name("beagle.ffi/__create_ffi_function")
        .unwrap();
    let function_pointer = runtime.compiler.get_pointer(create_ffi_function).unwrap();
    let function: fn(usize) -> usize = std::mem::transmute(function_pointer);
    let result = function(ffi_info_id);
    result
}

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
    let ffi_info = runtime.get_ffi_info(ffi_info_id);
    let code_ptr = ffi_info.function;
    let arguments = [a1, a2, a3, a4, a5, a6];
    
    // TODO: I hate this and don't want to make things work this way.
    // I get the tradeoff that libffi is making, but not a fan.
    let mut string_values = Box::new(Vec::new());
    let mut i32_values = Box::new(Vec::new());
    let mut ptr_values = Box::new(Vec::new());

    let args = &arguments[..ffi_info.number_of_arguments];

    for argument in args.iter() {
        let kind = BuiltInTypes::get_kind(*argument);
        match kind {
            BuiltInTypes::String => {
                let string = runtime.compiler.get_string(*argument);
                string_values.push(CString::new(string).unwrap());
            }
            BuiltInTypes::Int => {
                i32_values.push(BuiltInTypes::untag(*argument) as i32);
            }
            BuiltInTypes::HeapObject => {
                // TODO: Make this type safe
                let heap_object = HeapObject::from_tagged(*argument);
                let buffer = BuiltInTypes::untag(heap_object.get_field(0));
                ptr_values.push(buffer as *mut c_void);
            }
            _ => panic!("Unsupported type: {:?}", kind),
        }
    }

    let mut passed_arguments = Vec::new();
    let mut string_index = 0;
    let mut i32_index = 0;
    let mut ptr_index = 0;
    for argument in args.iter() {
        let kind = BuiltInTypes::get_kind(*argument);
        match kind {
            BuiltInTypes::String => {
                let string = &string_values[string_index];
                passed_arguments.push(Arg::new(&(string.as_ptr() as *mut c_void)));
                string_index += 1;
            }
            BuiltInTypes::Int => {
                passed_arguments.push(Arg::new(i32_values.get(i32_index).unwrap()));
                i32_index += 1;
            }
            BuiltInTypes::HeapObject => {
                passed_arguments.push(Arg::new(ptr_values.get(ptr_index).unwrap()));
                ptr_index += 1;
            }
            _ => panic!("Unsupported type: {:?}", kind),
        }
    }

    // TODO: I'm just leaking this memory for now.
    // I don't know what to do with it yet. So I need to figure that out.
    Box::leak(string_values);
    Box::leak(i32_values);
    Box::leak(ptr_values);

    let result = ffi_info.cif.call::<u32>(code_ptr, &passed_arguments);


    result as usize
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
    let buffer = Box::new(vec![0u8; size]);
    let buffer_ptr = Box::into_raw(buffer);


    let call_fn = runtime
        .compiler
        .get_function_by_name("beagle.ffi/__make_buffer_struct")
        .unwrap();
    let function_pointer = runtime.compiler.get_pointer(call_fn).unwrap();
    let function: fn(usize) -> usize = std::mem::transmute(function_pointer);
    function(BuiltInTypes::Int.tag(buffer_ptr as isize) as usize)
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


extern "C" fn placeholder() -> usize {
    BuiltInTypes::null_value() as usize
}

extern "C" fn wait_for_input() -> usize {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    0
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
            "beagle.builtin/__pause",
            __pause::<Alloc> as *const u8,
            true,
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
            "beagle.core/wait_for_input",
            wait_for_input as *const u8,
            false,
        )?;

        Ok(())
    }
}
