use super::*;
use crate::save_gc_context;

/// Full write barrier for generated code heap stores.
/// Takes the untagged address of the object being written to and the value written.
/// This mirrors Runtime::set_field_with_barrier for stores emitted directly by codegen.
pub extern "C" fn write_barrier(untagged_addr: usize, value: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let tagged_addr = BuiltInTypes::HeapObject.tag(untagged_addr as isize) as usize;
    runtime.write_barrier(tagged_addr, value);
    0b111
}

/// Register a freshly-constructed heap object as finalizable so the GC
/// can find it at minor-collection time without sweeping the whole young
/// space. Called from beagle.ffi.bg right after constructing a Buffer/
/// Cell/TypedArray. Returns the same pointer for convenient chaining.
pub extern "C" fn register_finalizable(tagged_ptr: usize) -> usize {
    let runtime = get_runtime().get_mut();
    runtime.register_finalizable(tagged_ptr);
    tagged_ptr
}

pub extern "C" fn allocate(stack_pointer: usize, frame_pointer: usize, size: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let size = BuiltInTypes::untag(size);
    let runtime = get_runtime().get_mut();

    let result = match runtime.allocate(size, stack_pointer, BuiltInTypes::HeapObject) {
        Ok(ptr) => ptr,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate heap object - out of memory".to_string(),
            );
        },
    };

    debug_assert!(BuiltInTypes::is_heap_pointer(result));
    debug_assert!(BuiltInTypes::untag(result).is_multiple_of(8));
    result
}

/// Allocate with zeroed memory (for arrays that don't initialize all fields)
pub extern "C" fn allocate_zeroed(
    stack_pointer: usize,
    frame_pointer: usize,
    size: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let size = BuiltInTypes::untag(size);
    let runtime = get_runtime().get_mut();

    let result = match runtime.allocate_zeroed(size, stack_pointer, BuiltInTypes::HeapObject) {
        Ok(ptr) => ptr,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate zeroed heap object - out of memory".to_string(),
            );
        },
    };

    debug_assert!(BuiltInTypes::is_heap_pointer(result));
    debug_assert!(BuiltInTypes::untag(result).is_multiple_of(8));
    result
}
