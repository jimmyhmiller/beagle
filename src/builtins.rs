use std::{
    error::Error,
    mem,
    slice::{from_raw_parts, from_raw_parts_mut},
    thread,
};

use crate::{
    gc::Allocator,
    runtime::Runtime,
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
        .get_function_by_name("beagle.core/__make_lib_struct")
        .unwrap();
    let function_pointer = runtime.compiler.get_pointer(call_fn).unwrap();
    let function: fn(usize) -> usize = std::mem::transmute(function_pointer);
    function(id)
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

extern "C" fn placeholder() -> usize {
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
            "beagle.core/load_library",
            load_library::<Alloc> as *const u8,
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

        Ok(())
    }
}
