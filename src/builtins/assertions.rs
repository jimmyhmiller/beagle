use super::*;
use crate::save_gc_context;

pub unsafe extern "C" fn assert_truthy(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    // Falsy values: false, null
    let is_falsy = value == BuiltInTypes::false_value() as usize
        || value == BuiltInTypes::null_value() as usize;

    if is_falsy {
        let runtime = get_runtime().get_mut();
        let repr = runtime
            .get_repr(value, 0)
            .unwrap_or_else(|| "???".to_string());
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "AssertError",
                format!("assert! failed: expected truthy value, got {}", repr),
            );
        }
    }
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn assert_eq(
    stack_pointer: usize,
    frame_pointer: usize,
    a: usize,
    b: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    if !runtime.equal(a, b) {
        let repr_a = runtime.get_repr(a, 0).unwrap_or_else(|| "???".to_string());
        let repr_b = runtime.get_repr(b, 0).unwrap_or_else(|| "???".to_string());
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "AssertError",
                format!(
                    "assert-eq! failed\n  expected: {}\n       got: {}",
                    repr_b, repr_a
                ),
            );
        }
    }
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn assert_ne(
    stack_pointer: usize,
    frame_pointer: usize,
    a: usize,
    b: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    if runtime.equal(a, b) {
        let repr_a = runtime.get_repr(a, 0).unwrap_or_else(|| "???".to_string());
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "AssertError",
                format!("assert-ne! failed: both values are {}", repr_a),
            );
        }
    }
    BuiltInTypes::null_value() as usize
}
