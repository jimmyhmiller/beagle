use super::*;
use crate::save_gc_context;

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
        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

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

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

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

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// abs builtin - computes absolute value of a number (int or float)
pub unsafe extern "C" fn abs_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let kind = BuiltInTypes::get_kind(value);

        match kind {
            BuiltInTypes::Int => {
                // For integers, the value is stored directly in the tagged value
                let int_value = BuiltInTypes::untag_isize(value as isize);
                let result = int_value.abs();
                BuiltInTypes::construct_int(result) as usize
            }
            BuiltInTypes::Float => {
                // For floats, we need to read from the heap
                let runtime = get_runtime().get_mut();

                let untagged = BuiltInTypes::untag(value);
                let float_ptr = untagged as *const f64;
                let float_value = *float_ptr.add(1);

                let result = float_value.abs();

                let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
                    Ok(ptr) => ptr,
                    Err(_) => {
                        throw_runtime_error(
                            stack_pointer,
                            "AllocationError",
                            "Failed to allocate float result - out of memory".to_string(),
                        );
                    }
                };

                let untagged_result = BuiltInTypes::untag(new_float_ptr);
                let result_ptr = untagged_result as *mut f64;
                *result_ptr.add(1) = result;

                new_float_ptr
            }
            _ => {
                // For other types, throw a type error
                throw_runtime_error(
                    stack_pointer,
                    "TypeError",
                    "abs requires a number (int or float)".to_string(),
                );
            }
        }
    }
}

/// round builtin - rounds a float to nearest integer
pub unsafe extern "C" fn round_builtin(
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

        let result = float_value.round();

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// truncate builtin - truncates a float towards zero
pub unsafe extern "C" fn truncate_builtin(
    _stack_pointer: usize,
    _frame_pointer: usize,
    value: usize,
) -> usize {
    unsafe {
        let kind = BuiltInTypes::get_kind(value);
        if kind == BuiltInTypes::Int {
            return value;
        }

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.trunc() as isize;
        BuiltInTypes::Int.tag(result) as usize
    }
}

/// max builtin - returns the maximum of two numbers
pub extern "C" fn max_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    a: usize,
    b: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);

    let a_kind = BuiltInTypes::get_kind(a);
    let b_kind = BuiltInTypes::get_kind(b);

    // Both are ints
    if a_kind == BuiltInTypes::Int && b_kind == BuiltInTypes::Int {
        let a_val = BuiltInTypes::untag_isize(a as isize);
        let b_val = BuiltInTypes::untag_isize(b as isize);
        return BuiltInTypes::construct_int(a_val.max(b_val)) as usize;
    }

    // At least one is a float - convert both to float and compare
    unsafe {
        let a_float = if a_kind == BuiltInTypes::Float {
            let untagged = BuiltInTypes::untag(a);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        } else {
            BuiltInTypes::untag_isize(a as isize) as f64
        };

        let b_float = if b_kind == BuiltInTypes::Float {
            let untagged = BuiltInTypes::untag(b);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        } else {
            BuiltInTypes::untag_isize(b as isize) as f64
        };

        let result = a_float.max(b_float);

        let runtime = get_runtime().get_mut();
        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// min builtin - returns the minimum of two numbers
pub extern "C" fn min_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    a: usize,
    b: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);

    let a_kind = BuiltInTypes::get_kind(a);
    let b_kind = BuiltInTypes::get_kind(b);

    // Both are ints
    if a_kind == BuiltInTypes::Int && b_kind == BuiltInTypes::Int {
        let a_val = BuiltInTypes::untag_isize(a as isize);
        let b_val = BuiltInTypes::untag_isize(b as isize);
        return BuiltInTypes::construct_int(a_val.min(b_val)) as usize;
    }

    // At least one is a float - convert both to float and compare
    unsafe {
        let a_float = if a_kind == BuiltInTypes::Float {
            let untagged = BuiltInTypes::untag(a);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        } else {
            BuiltInTypes::untag_isize(a as isize) as f64
        };

        let b_float = if b_kind == BuiltInTypes::Float {
            let untagged = BuiltInTypes::untag(b);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        } else {
            BuiltInTypes::untag_isize(b as isize) as f64
        };

        let result = a_float.min(b_float);

        let runtime = get_runtime().get_mut();
        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// clamp builtin - clamps a value between low and high bounds
pub extern "C" fn clamp_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
    low: usize,
    high: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let value_kind = BuiltInTypes::get_kind(value);
        let low_kind = BuiltInTypes::get_kind(low);
        let high_kind = BuiltInTypes::get_kind(high);

        // If all are integers, do integer clamping
        if value_kind == BuiltInTypes::Int
            && low_kind == BuiltInTypes::Int
            && high_kind == BuiltInTypes::Int
        {
            let value_int = BuiltInTypes::untag_isize(value as isize);
            let low_int = BuiltInTypes::untag_isize(low as isize);
            let high_int = BuiltInTypes::untag_isize(high as isize);

            let clamped = value_int.max(low_int).min(high_int);
            return BuiltInTypes::construct_int(clamped) as usize;
        }

        // Otherwise, convert to float and do float clamping
        let value_float = if value_kind == BuiltInTypes::Int {
            BuiltInTypes::untag_isize(value as isize) as f64
        } else {
            let untagged = BuiltInTypes::untag(value);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        };

        let low_float = if low_kind == BuiltInTypes::Int {
            BuiltInTypes::untag_isize(low as isize) as f64
        } else {
            let untagged = BuiltInTypes::untag(low);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        };

        let high_float = if high_kind == BuiltInTypes::Int {
            BuiltInTypes::untag_isize(high as isize) as f64
        } else {
            let untagged = BuiltInTypes::untag(high);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        };

        let result = value_float.max(low_float).min(high_float);

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// gcd builtin - computes greatest common divisor of two integers
pub extern "C" fn gcd_builtin(a: usize, b: usize) -> usize {
    let mut a_val = BuiltInTypes::untag_isize(a as isize).abs();
    let mut b_val = BuiltInTypes::untag_isize(b as isize).abs();

    // Euclidean algorithm
    while b_val != 0 {
        let temp = b_val;
        b_val = a_val % b_val;
        a_val = temp;
    }

    BuiltInTypes::construct_int(a_val) as usize
}

/// lcm builtin - computes least common multiple of two integers
pub extern "C" fn lcm_builtin(a: usize, b: usize) -> usize {
    let a_val = BuiltInTypes::untag_isize(a as isize).abs();
    let b_val = BuiltInTypes::untag_isize(b as isize).abs();

    if a_val == 0 || b_val == 0 {
        return BuiltInTypes::construct_int(0) as usize;
    }

    // LCM(a,b) = |a*b| / GCD(a,b)
    let gcd_result = gcd_builtin(a, b);
    let gcd_val = BuiltInTypes::untag_isize(gcd_result as isize);

    let lcm = (a_val * b_val) / gcd_val;

    BuiltInTypes::construct_int(lcm) as usize
}

/// random builtin - returns a random float between 0.0 and 1.0
pub unsafe extern "C" fn random_builtin(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();
        let mut rng = rand::thread_rng();
        let random_value: f64 = rng.gen_range(0.0..1.0);

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = random_value;

        new_float_ptr
    }
}

/// random-int builtin - returns a random integer from 0 to max-1
pub extern "C" fn random_int_builtin(max: usize) -> usize {
    let max_val = BuiltInTypes::untag_isize(max as isize);

    if max_val <= 0 {
        return BuiltInTypes::construct_int(0) as usize;
    }

    let mut rng = rand::thread_rng();
    let random_value: isize = rng.gen_range(0..max_val);

    BuiltInTypes::construct_int(random_value) as usize
}

/// random-range builtin - returns a random integer from min to max-1
pub extern "C" fn random_range_builtin(min: usize, max: usize) -> usize {
    let min_val = BuiltInTypes::untag_isize(min as isize);
    let max_val = BuiltInTypes::untag_isize(max as isize);

    if min_val >= max_val {
        return BuiltInTypes::construct_int(min_val) as usize;
    }

    let mut rng = rand::thread_rng();
    let random_value: isize = rng.gen_range(min_val..max_val);

    BuiltInTypes::construct_int(random_value) as usize
}

/// even? predicate - checks if an integer is even
pub extern "C" fn is_even(value: usize) -> usize {
    let value_kind = BuiltInTypes::get_kind(value);
    if value_kind == BuiltInTypes::Int {
        let int_val = BuiltInTypes::untag_isize(value as isize);
        BuiltInTypes::construct_boolean(int_val % 2 == 0) as usize
    } else {
        BuiltInTypes::construct_boolean(false) as usize
    }
}

/// odd? predicate - checks if an integer is odd
pub extern "C" fn is_odd(value: usize) -> usize {
    let value_kind = BuiltInTypes::get_kind(value);
    if value_kind == BuiltInTypes::Int {
        let int_val = BuiltInTypes::untag_isize(value as isize);
        BuiltInTypes::construct_boolean(int_val % 2 != 0) as usize
    } else {
        BuiltInTypes::construct_boolean(false) as usize
    }
}

/// positive? predicate - checks if a number is positive
pub extern "C" fn is_positive(value: usize) -> usize {
    let value_kind = BuiltInTypes::get_kind(value);

    if value_kind == BuiltInTypes::Int {
        let int_val = BuiltInTypes::untag_isize(value as isize);
        BuiltInTypes::construct_boolean(int_val > 0) as usize
    } else if value_kind == BuiltInTypes::Float {
        unsafe {
            let untagged = BuiltInTypes::untag(value);
            let float_ptr = untagged as *const f64;
            let float_value = *float_ptr.add(1);
            BuiltInTypes::construct_boolean(float_value > 0.0) as usize
        }
    } else {
        BuiltInTypes::construct_boolean(false) as usize
    }
}

/// negative? predicate - checks if a number is negative
pub extern "C" fn is_negative(value: usize) -> usize {
    let value_kind = BuiltInTypes::get_kind(value);

    if value_kind == BuiltInTypes::Int {
        let int_val = BuiltInTypes::untag_isize(value as isize);
        BuiltInTypes::construct_boolean(int_val < 0) as usize
    } else if value_kind == BuiltInTypes::Float {
        unsafe {
            let untagged = BuiltInTypes::untag(value);
            let float_ptr = untagged as *const f64;
            let float_value = *float_ptr.add(1);
            BuiltInTypes::construct_boolean(float_value < 0.0) as usize
        }
    } else {
        BuiltInTypes::construct_boolean(false) as usize
    }
}

/// zero? predicate - checks if a number is zero
pub extern "C" fn is_zero(value: usize) -> usize {
    let value_kind = BuiltInTypes::get_kind(value);

    if value_kind == BuiltInTypes::Int {
        let int_val = BuiltInTypes::untag_isize(value as isize);
        BuiltInTypes::construct_boolean(int_val == 0) as usize
    } else if value_kind == BuiltInTypes::Float {
        unsafe {
            let untagged = BuiltInTypes::untag(value);
            let float_ptr = untagged as *const f64;
            let float_value = *float_ptr.add(1);
            BuiltInTypes::construct_boolean(float_value == 0.0) as usize
        }
    } else {
        BuiltInTypes::construct_boolean(false) as usize
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

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

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

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// tan builtin - computes tangent of a float (in radians)
pub unsafe extern "C" fn tan_builtin(
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

        let result = float_value.tan();

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// asin builtin - computes arcsine of a float (returns radians)
pub unsafe extern "C" fn asin_builtin(
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

        let result = float_value.asin();

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// acos builtin - computes arccosine of a float (returns radians)
pub unsafe extern "C" fn acos_builtin(
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

        let result = float_value.acos();

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// atan builtin - computes arctangent of a float (returns radians)
pub unsafe extern "C" fn atan_builtin(
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

        let result = float_value.atan();

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// atan2 builtin - computes arctangent of y/x (returns radians)
pub unsafe extern "C" fn atan2_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    y: usize,
    x: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let y_untagged = BuiltInTypes::untag(y);
        let y_float_ptr = y_untagged as *const f64;
        let y_value = *y_float_ptr.add(1);

        let x_untagged = BuiltInTypes::untag(x);
        let x_float_ptr = x_untagged as *const f64;
        let x_value = *x_float_ptr.add(1);

        let result = y_value.atan2(x_value);

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// exp builtin - computes e^x
pub unsafe extern "C" fn exp_builtin(
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

        let result = float_value.exp();

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// log builtin - computes natural logarithm (base e)
pub unsafe extern "C" fn log_builtin(
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

        let result = float_value.ln();

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// log10 builtin - computes base-10 logarithm
pub unsafe extern "C" fn log10_builtin(
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

        let result = float_value.log10();

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// log2 builtin - computes base-2 logarithm
pub unsafe extern "C" fn log2_builtin(
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

        let result = float_value.log2();

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// pow builtin - computes base^exponent
pub unsafe extern "C" fn pow_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    base: usize,
    exponent: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let base_untagged = BuiltInTypes::untag(base);
        let base_float_ptr = base_untagged as *const f64;
        let base_value = *base_float_ptr.add(1);

        let exp_untagged = BuiltInTypes::untag(exponent);
        let exp_float_ptr = exp_untagged as *const f64;
        let exp_value = *exp_float_ptr.add(1);

        let result = base_value.powf(exp_value);

        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

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
        // Use signed right shift to preserve negative numbers
        let int_value = BuiltInTypes::untag_isize(value as isize) as i64;

        // Convert to f64
        let float_value = int_value as f64;

        // Allocate a new float object for the result
        let new_float_ptr = match runtime.allocate(1, stack_pointer, BuiltInTypes::Float) {
            Ok(ptr) => ptr,
            Err(_) => {
                throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate float result - out of memory".to_string(),
                );
            }
        };

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = float_value;

        new_float_ptr
    }
}

/// to_int builtin - converts a float to a tagged integer (truncating towards zero)
pub unsafe extern "C" fn to_int_builtin(
    _stack_pointer: usize,
    _frame_pointer: usize,
    value: usize,
) -> usize {
    unsafe {
        let kind = BuiltInTypes::get_kind(value);
        if kind == BuiltInTypes::Int {
            return value;
        }

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.trunc() as isize;
        BuiltInTypes::Int.tag(result) as usize
    }
}

pub extern "C" fn pop_count(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
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
