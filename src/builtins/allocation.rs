use super::*;
use crate::save_gc_context;

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

pub extern "C" fn allocate_float(stack_pointer: usize, frame_pointer: usize, size: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let value = BuiltInTypes::untag(size);

    let result = match runtime.allocate(value, stack_pointer, BuiltInTypes::Float) {
        Ok(ptr) => ptr,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate float - out of memory".to_string(),
            );
        },
    };

    debug_assert!(BuiltInTypes::get_kind(result) == BuiltInTypes::Float);
    debug_assert!(BuiltInTypes::untag(result).is_multiple_of(8));
    result
}
