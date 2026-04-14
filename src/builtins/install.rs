use super::*;

// If we inline, we need to remove a skip frame
impl Runtime {
    pub fn install_builtins(&mut self) -> Result<(), Box<dyn Error>> {
        self.add_builtin_function(
            "beagle.__internal_test__/many_args",
            many_args as *const u8,
            false,
            11,
        )?;

        self.add_builtin_with_doc(
            "beagle.core/_println",
            println_value as *const u8,
            false,
            &["value"],
            "Print a value followed by a newline to standard output.",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/_print",
            print_value as *const u8,
            false,
            &["value"],
            "Print a value to standard output without a trailing newline.",
        )?;

        // Dynamic variable builtins
        self.add_builtin_with_doc(
            "beagle.core/_get_dynamic_var",
            get_dynamic_var as *const u8,
            false,
            &["namespace_id", "slot"],
            "Internal: Get the current value of a dynamic variable.",
        )?;

        // Continuation mark builtins (need frame pointer)
        self.add_builtin_function_with_fp(
            "beagle.core/_install_continuation_mark",
            install_continuation_mark as *const u8,
            true,
            true,
            6, // stack_pointer, frame_pointer, namespace_id, slot, value, mark_local_index
        )?;

        self.add_builtin_function_with_fp(
            "beagle.core/_uninstall_continuation_mark",
            uninstall_continuation_mark as *const u8,
            true,
            true,
            3, // stack_pointer, frame_pointer, mark_local_index
        )?;

        // Multi-arity dispatch builtin (needs stack/frame pointer for throwing ArityError)
        self.add_builtin_function_with_fp(
            "beagle.builtin/dispatch-multi-arity",
            dispatch_multi_arity as *const u8,
            true,
            true,
            4, // stack_pointer + frame_pointer + multi_arity_obj + arg_count
        )?;

        // to-string: Convert any value to its string representation
        self.add_builtin_with_doc(
            "beagle.core/to-string",
            to_string as *const u8,
            true,
            &["value"],
            "Convert any value to its string representation.\n\nExamples:\n  (to-string 42)     ; => \"42\"\n  (to-string true)   ; => \"true\"\n  (to-string [1 2])  ; => \"[1, 2]\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/repr",
            repr as *const u8,
            true,
            &["value"],
            "Return a string representation of a value that could be evaluated back.\nStrings are quoted, special characters are escaped.\n\nExamples:\n  (repr 42)        ; => \"42\"\n  (repr \"hello\")   ; => \"\\\"hello\\\"\"\n  (repr [1 \"a\"])   ; => \"[1, \\\"a\\\"]\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/current-namespace",
            get_current_namespace as *const u8,
            true,
            &[],
            "Return the name of the current namespace as a string.",
        )?;

        // to-number: Parse a string into a number
        self.add_builtin_with_doc(
            "beagle.core/to-number",
            to_number as *const u8,
            true,
            &["string"],
            "Parse a string into a number. Throws an error if the string is not a valid number.\n\nExamples:\n  (to-number \"42\")   ; => 42\n  (to-number \"-5\")   ; => -5",
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

        // copy_object_spread takes (stack_pointer, frame_pointer, object_pointer, expected_struct_id)
        self.add_builtin_function_with_fp(
            "beagle.builtin/copy-object-spread",
            copy_object_spread as *const u8,
            true,
            true,
            4,
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

        // make_function_object takes (stack_pointer, frame_pointer, function)
        self.add_builtin_function_with_fp(
            "beagle.builtin/make-function-object",
            make_function_object as *const u8,
            true,
            true,
            3,
        )?;

        self.add_builtin_function(
            "beagle.builtin/property-access",
            property_access as *const u8,
            true,
            4,
        )?;

        self.add_builtin_function(
            "beagle.builtin/check-struct-family",
            check_struct_family as *const u8,
            false,
            2,
        )?;

        self.add_builtin_function_with_fp(
            "beagle.builtin/protocol-dispatch",
            protocol_dispatch as *const u8,
            true,
            true,
            5, // stack_pointer, frame_pointer, first_arg, cache_location, dispatch_table_ptr
        )?;

        // type-of: Get the type of a value
        self.add_builtin_with_doc(
            "beagle.core/type-of",
            type_of as *const u8,
            true,
            &["value"],
            "Return a type descriptor for the given value.\n\nReturns a Struct instance representing the type (e.g., Int, String, Array).\n\nExamples:\n  (type-of 42)        ; => Int\n  (type-of \"hello\")   ; => String\n  (type-of [1 2 3])   ; => Array",
        )?;

        // get-os: Get the operating system name
        self.add_builtin_with_doc(
            "beagle.core/get-os",
            get_os as *const u8,
            true,
            &[],
            "Return the name of the current operating system.\n\nReturns one of: \"macos\", \"linux\", \"windows\", or \"unknown\".",
        )?;

        // atom-address: Get the memory address of a value
        self.add_builtin_with_doc(
            "beagle.core/atom-address",
            atom_address as *const u8,
            false,
            &["value"],
            "Return the raw memory address of a value as an integer.\n\nUseful for identity comparisons and async coordination.",
        )?;

        // equal: Deep equality comparison
        self.add_builtin_with_doc(
            "beagle.core/equal",
            equal as *const u8,
            false,
            &["a", "b"],
            "Compare two values for deep equality.\n\nReturns true if the values are structurally equal, false otherwise.\n\nExamples:\n  (equal 1 1)           ; => true\n  (equal [1 2] [1 2])   ; => true\n  (equal {:a 1} {:a 1}) ; => true",
        )?;

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

        // build_rest_array_from_locals takes (stack_pointer, frame_pointer, arg_count, min_args, first_local_index, first_arg_index)
        self.add_builtin_function_with_fp(
            "beagle.builtin/build-rest-array-from-locals",
            build_rest_array_from_locals as *const u8,
            true,
            true,
            6,
        )?;

        // call_variadic_function_value now takes (stack_pointer, frame_pointer, function_ptr, args_array, is_closure, closure_ptr)
        self.add_builtin_function_with_fp(
            "beagle.builtin/call-variadic-function-value",
            call_variadic_function_value as *const u8,
            true,
            true,
            6,
        )?;

        // Function application
        self.add_builtin_with_doc(
            "beagle.core/apply",
            apply_function as *const u8,
            true,
            &["f", "args"],
            "Apply a function to an array of arguments.\n\nExamples:\n  (apply + [1 2 3])      ; => 6\n  (apply max [3 7 2])    ; => 7\n  (apply my-fn [a b c])  ; equivalent to (my-fn a b c)",
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

        // throw_exception takes (stack_pointer, frame_pointer, value, resume_address, resume_local_offset)
        self.add_builtin_function_with_fp(
            "beagle.builtin/throw-exception",
            throw_exception as *const u8,
            true,
            true,
            5,
        )?;

        self.add_builtin_function(
            "beagle.builtin/push-resumable-exception-handler",
            push_resumable_exception_handler_runtime as *const u8,
            false,
            6, // handler_address, result_local, resume_local, link_register, stack_pointer, frame_pointer
        )?;

        self.add_builtin_function(
            "beagle.builtin/pop-exception-handler-by-id",
            pop_exception_handler_by_id_runtime as *const u8,
            false,
            1, // handler_id
        )?;

        // Delimited continuation builtins
        self.add_builtin_function(
            "beagle.builtin/push-prompt",
            push_prompt_runtime as *const u8,
            false,
            5, // handler_address, result_local, link_register, stack_pointer, frame_pointer
        )?;

        self.add_builtin_function(
            "beagle.builtin/pop-prompt",
            pop_prompt_runtime as *const u8,
            false,
            0,
        )?;

        // Tag-aware prompt push/pop for effect handlers. These mirror
        // push-prompt/pop-prompt above but store records on the
        // prompt-tag side-stack keyed by a fresh u64 tag. Used by the
        // lowering of `handle { ... } with h` (Step E6+).
        self.add_builtin_function(
            "beagle.builtin/push-prompt-tag",
            push_prompt_tag_runtime as *const u8,
            false,
            4, // tag, stack_pointer, frame_pointer, link_register
        )?;

        self.add_builtin_function(
            "beagle.builtin/pop-prompt-tag",
            pop_prompt_tag_runtime as *const u8,
            false,
            1, // tag
        )?;

        // Register return-from-shift-tagged for effect handlers.
        self.add_builtin_function_with_fp(
            "beagle.builtin/return-from-shift-tagged",
            return_from_shift_tagged_runtime as *const u8,
            true,
            true,
            4,
        )?;

        // Tag-aware capture: (sp, fp, resume_address, result_local_offset, tag)
        self.add_builtin_function_with_fp(
            "beagle.builtin/capture-continuation-tagged",
            capture_continuation_tagged_runtime as *const u8,
            true,
            true,
            5,
        )?;

        // capture-continuation takes
        // (stack_pointer, frame_pointer, resume_address, result_local_offset, saved_regs_ptr)
        self.add_builtin_function_with_fp(
            "beagle.builtin/capture-continuation",
            capture_continuation_runtime_with_saved_regs as *const u8,
            true,
            true,
            5,
        )?;

        // return-from-shift takes (stack_pointer, frame_pointer, value, cont_ptr)
        self.add_builtin_function_with_fp(
            "beagle.builtin/return-from-shift",
            return_from_shift_runtime as *const u8,
            true,
            true,
            4,
        )?;

        // return-from-shift-handler is for handler returns (after call-handler in perform)
        // It sets is_handler_return flag so return_from_shift skips InvocationReturnPoints
        self.add_builtin_function_with_fp(
            "beagle.builtin/return-from-shift-handler",
            return_from_shift_handler_runtime as *const u8,
            true,
            true,
            4,
        )?;

        // Refactor A stubs — kept registered so stdlib compile-time lookups
        // for `beagle.builtin/invoke-continuation` and
        // `beagle.builtin/resume-tail` continue to resolve. Any runtime
        // call aborts with a clear error.
        self.add_builtin_function_with_fp(
            "beagle.builtin/invoke-continuation",
            invoke_continuation_runtime as *const u8,
            true,
            true,
            4,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.builtin/resume-tail",
            resume_tail_runtime as *const u8,
            true,
            true,
            4,
        )?;

        // continuation-trampoline is the function body for continuation closures
        // takes (closure_ptr, value) - it's a closure body, not a regular builtin
        // NOTE: On x86-64, this trampoline will be replaced by a generated version
        // in main.rs to avoid optimizer issues with inline assembly in release builds
        self.add_builtin_function(
            "beagle.builtin/continuation-trampoline",
            continuation_trampoline as *const u8,
            false,
            2,
        )?;

        self.add_builtin_with_doc(
            "beagle.core/set-thread-exception-handler!",
            set_thread_exception_handler as *const u8,
            false,
            &["handler"],
            "Set an exception handler for the current thread.\n\nThe handler function receives the exception value when an uncaught exception occurs.",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/set-default-exception-handler!",
            set_default_exception_handler as *const u8,
            false,
            &["handler"],
            "Set the default exception handler for all threads.\n\nThis handler is used when a thread doesn't have its own handler set.",
        )?;

        self.add_builtin_function(
            "beagle.builtin/create-error",
            create_error as *const u8,
            true,
            4, // stack_pointer, kind_str, message_str, location_str
        )?;

        // Assertion builtins for test blocks
        self.add_builtin_with_doc(
            "beagle.core/assert!",
            assert_truthy as *const u8,
            true,
            &["value"],
            "Assert that a value is truthy. Throws AssertError if the value is false, null, or 0.",
        )?;
        self.add_builtin_with_doc(
            "beagle.core/assert-eq!",
            assert_eq as *const u8,
            true,
            &["actual", "expected"],
            "Assert that two values are equal. Throws AssertError showing both values on failure.",
        )?;
        self.add_builtin_with_doc(
            "beagle.core/assert-ne!",
            assert_ne as *const u8,
            true,
            &["a", "b"],
            "Assert that two values are not equal. Throws AssertError on failure.",
        )?;
        // Garbage collection
        self.add_builtin_with_doc(
            "beagle.core/gc",
            gc as *const u8,
            true,
            &[],
            "Trigger garbage collection manually.\n\nNormally GC runs automatically, but this can be useful for testing or freeing memory at specific points.",
        )?;

        // ============================================================================
        // Math Functions
        // ============================================================================

        self.add_builtin_with_doc(
            "beagle.core/sqrt",
            sqrt_builtin as *const u8,
            true,
            &["x"],
            "Return the square root of x.\n\nExamples:\n  (sqrt 4)    ; => 2.0\n  (sqrt 2)    ; => 1.414...",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/floor",
            floor_builtin as *const u8,
            true,
            &["x"],
            "Return the largest integer less than or equal to x.\n\nExamples:\n  (floor 3.7)   ; => 3\n  (floor -2.3)  ; => -3",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/ceil",
            ceil_builtin as *const u8,
            true,
            &["x"],
            "Return the smallest integer greater than or equal to x.\n\nExamples:\n  (ceil 3.2)   ; => 4\n  (ceil -2.7)  ; => -2",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/abs",
            abs_builtin as *const u8,
            true,
            &["x"],
            "Return the absolute value of x.\n\nExamples:\n  (abs -5)   ; => 5\n  (abs 3.2)  ; => 3.2",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/round",
            round_builtin as *const u8,
            true,
            &["x"],
            "Round x to the nearest integer.\n\nExamples:\n  (round 3.4)  ; => 3\n  (round 3.6)  ; => 4",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/truncate",
            truncate_builtin as *const u8,
            true,
            &["x"],
            "Truncate x toward zero (remove the fractional part).\n\nExamples:\n  (truncate 3.7)   ; => 3\n  (truncate -3.7)  ; => -3",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/max",
            max_builtin as *const u8,
            true,
            &["a", "b"],
            "Return the larger of two numbers.\n\nExamples:\n  (max 3 7)  ; => 7\n  (max -1 -5)  ; => -1",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/min",
            min_builtin as *const u8,
            true,
            &["a", "b"],
            "Return the smaller of two numbers.\n\nExamples:\n  (min 3 7)  ; => 3\n  (min -1 -5)  ; => -5",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/even?",
            is_even as *const u8,
            false,
            &["n"],
            "Return true if n is even.\n\nExamples:\n  (even? 4)  ; => true\n  (even? 3)  ; => false",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/odd?",
            is_odd as *const u8,
            false,
            &["n"],
            "Return true if n is odd.\n\nExamples:\n  (odd? 3)  ; => true\n  (odd? 4)  ; => false",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/positive?",
            is_positive as *const u8,
            false,
            &["n"],
            "Return true if n is positive (greater than zero).\n\nExamples:\n  (positive? 5)   ; => true\n  (positive? -3)  ; => false\n  (positive? 0)   ; => false",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/negative?",
            is_negative as *const u8,
            false,
            &["n"],
            "Return true if n is negative (less than zero).\n\nExamples:\n  (negative? -3)  ; => true\n  (negative? 5)   ; => false\n  (negative? 0)   ; => false",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/zero?",
            is_zero as *const u8,
            false,
            &["n"],
            "Return true if n is zero.\n\nExamples:\n  (zero? 0)  ; => true\n  (zero? 5)  ; => false",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/clamp",
            clamp_builtin as *const u8,
            true,
            &["x", "min_val", "max_val"],
            "Clamp x to be within the range [min_val, max_val].\n\nExamples:\n  (clamp 5 0 10)   ; => 5\n  (clamp -5 0 10)  ; => 0\n  (clamp 15 0 10)  ; => 10",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/gcd",
            gcd_builtin as *const u8,
            false,
            &["a", "b"],
            "Return the greatest common divisor of a and b.\n\nExamples:\n  (gcd 12 8)  ; => 4\n  (gcd 17 5)  ; => 1",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/lcm",
            lcm_builtin as *const u8,
            false,
            &["a", "b"],
            "Return the least common multiple of a and b.\n\nExamples:\n  (lcm 4 6)  ; => 12\n  (lcm 3 5)  ; => 15",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/random",
            random_builtin as *const u8,
            true,
            &[],
            "Return a random floating-point number between 0.0 (inclusive) and 1.0 (exclusive).\n\nExamples:\n  (random)  ; => 0.7234... (varies)",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/random-int",
            random_int_builtin as *const u8,
            false,
            &["max"],
            "Return a random integer from 0 (inclusive) to max (exclusive).\n\nExamples:\n  (random-int 10)  ; => 7 (varies, 0-9)",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/random-range",
            random_range_builtin as *const u8,
            false,
            &["min", "max"],
            "Return a random integer from min (inclusive) to max (exclusive).\n\nExamples:\n  (random-range 5 10)  ; => 7 (varies, 5-9)",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/sin",
            sin_builtin as *const u8,
            true,
            &["x"],
            "Return the sine of x (in radians).\n\nExamples:\n  (sin 0)       ; => 0.0\n  (sin (/ PI 2))  ; => 1.0",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/cos",
            cos_builtin as *const u8,
            true,
            &["x"],
            "Return the cosine of x (in radians).\n\nExamples:\n  (cos 0)   ; => 1.0\n  (cos PI)  ; => -1.0",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/tan",
            tan_builtin as *const u8,
            true,
            &["x"],
            "Return the tangent of x (in radians).\n\nExamples:\n  (tan 0)  ; => 0.0",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/asin",
            asin_builtin as *const u8,
            true,
            &["x"],
            "Return the arc sine (inverse sine) of x in radians.\n\nThe result is in the range [-PI/2, PI/2]. x must be in [-1, 1].",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/acos",
            acos_builtin as *const u8,
            true,
            &["x"],
            "Return the arc cosine (inverse cosine) of x in radians.\n\nThe result is in the range [0, PI]. x must be in [-1, 1].",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/atan",
            atan_builtin as *const u8,
            true,
            &["x"],
            "Return the arc tangent (inverse tangent) of x in radians.\n\nThe result is in the range [-PI/2, PI/2].",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/atan2",
            atan2_builtin as *const u8,
            true,
            &["y", "x"],
            "Return the arc tangent of y/x in radians, using signs to determine the quadrant.\n\nThe result is in the range [-PI, PI].",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/exp",
            exp_builtin as *const u8,
            true,
            &["x"],
            "Return e raised to the power x (e^x).\n\nExamples:\n  (exp 0)  ; => 1.0\n  (exp 1)  ; => 2.718...",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/log",
            log_builtin as *const u8,
            true,
            &["x"],
            "Return the natural logarithm (base e) of x.\n\nExamples:\n  (log 1)  ; => 0.0\n  (log E)  ; => 1.0",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/log10",
            log10_builtin as *const u8,
            true,
            &["x"],
            "Return the base-10 logarithm of x.\n\nExamples:\n  (log10 10)   ; => 1.0\n  (log10 100)  ; => 2.0",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/log2",
            log2_builtin as *const u8,
            true,
            &["x"],
            "Return the base-2 logarithm of x.\n\nExamples:\n  (log2 2)  ; => 1.0\n  (log2 8)  ; => 3.0",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/pow",
            pow_builtin as *const u8,
            true,
            &["base", "exponent"],
            "Return base raised to the power exponent.\n\nExamples:\n  (pow 2 3)  ; => 8.0\n  (pow 10 2) ; => 100.0",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/to-float",
            to_float_builtin as *const u8,
            true,
            &["x"],
            "Convert an integer to a floating-point number.\n\nExamples:\n  (to-float 42)  ; => 42.0",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/to-int",
            to_int_builtin as *const u8,
            true,
            &["x"],
            "Convert a float to an integer by truncating towards zero. If already an integer, returns unchanged.\n\nExamples:\n  (to-int 3.7)   ; => 3\n  (to-int -3.7)  ; => -3\n  (to-int 42)    ; => 42",
        )?;

        // ============================================================================
        // Threading
        // ============================================================================

        self.add_builtin_with_doc(
            "beagle.core/thread",
            new_thread as *const u8,
            true,
            &["f"],
            "Spawn a new thread to execute function f.\n\nReturns a Thread object that can be used with thread-join.\n\nExamples:\n  (let t (thread fn() { expensive-computation() }))\n  (thread-join t)  ; wait for result",
        )?;

        // ============================================================================
        // FFI (Foreign Function Interface)
        // ============================================================================

        self.add_builtin_with_doc(
            "beagle.ffi/load-library",
            load_library as *const u8,
            true,
            &["path"],
            "Load a dynamic library (shared object) from the given path.\n\nReturns a Library struct that can be used with get-function.\n\nExamples:\n  (let lib (ffi/load-library \"libm.dylib\"))",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/get-function",
            get_function as *const u8,
            true,
            &["library", "name", "arg_types", "return_type"],
            "Get a function from a loaded library.\n\narg_types is an array of Type values, return_type is a Type.\n\nExamples:\n  (let sqrt-fn (ffi/get-function lib \"sqrt\" [Type.F64] Type.F64))",
        )?;

        // Internal FFI call function: (sp, fp, ffi_info_id, args_array)
        self.add_builtin_function(
            "beagle.ffi/call-ffi-info",
            call_ffi_info as *const u8,
            true,
            3,
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/get-symbol",
            get_symbol as *const u8,
            true,
            &["library", "name"],
            "Get a raw function pointer (symbol) from a loaded library.\n\nReturns a Pointer struct. Used with call-variadic for per-call type specification.",
        )?;

        // Internal variadic FFI call: (sp, fp, func_ptr, args_array, types_array, return_type)
        self.add_builtin_function(
            "beagle.ffi/call-variadic-raw",
            call_ffi_variadic as *const u8,
            true,
            5,
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/create-callback",
            create_callback as *const u8,
            true,
            &["fn", "arg_types", "return_type"],
            "Create an FFI callback from a Beagle function.\n\nReturns a Pointer that can be passed to C functions expecting function pointers.\narg_types is an array of Type values for the C callback parameters.\nreturn_type is the C return type.\n\nCallbacks must be called from the main thread.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/allocate",
            ffi_allocate as *const u8,
            false,
            &["size"],
            "Allocate size bytes of unmanaged memory.\n\nReturns a Pointer. Must be freed with deallocate.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/deallocate",
            ffi_deallocate as *const u8,
            false,
            &["ptr"],
            "Free memory allocated with allocate.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/get-u32",
            ffi_get_u32 as *const u8,
            false,
            &["ptr", "offset"],
            "Read an unsigned 32-bit integer from memory at ptr + offset.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/set-i16",
            ffi_set_i16 as *const u8,
            false,
            &["ptr", "offset", "value"],
            "Write a signed 16-bit integer to memory at ptr + offset.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/set-i32",
            ffi_set_i32 as *const u8,
            false,
            &["ptr", "offset", "value"],
            "Write a signed 32-bit integer to memory at ptr + offset.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/set-u8",
            ffi_set_u8 as *const u8,
            false,
            &["ptr", "offset", "value"],
            "Write an unsigned 8-bit integer (byte) to memory at ptr + offset.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/get-u8",
            ffi_get_u8 as *const u8,
            false,
            &["ptr", "offset"],
            "Read an unsigned 8-bit integer (byte) from memory at ptr + offset.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/get-i32",
            ffi_get_i32 as *const u8,
            false,
            &["ptr", "offset"],
            "Read a signed 32-bit integer from memory at ptr + offset.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/get-string",
            ffi_get_string as *const u8,
            true,
            &["buffer", "offset", "length"],
            "Read a string from a buffer at the given offset with the specified length.\n\nExamples:\n  (ffi/get-string buf 0 10)  ; Read 10 bytes starting at offset 0",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/get-string-and-free",
            ffi_get_string_and_free as *const u8,
            true,
            &["buffer", "offset", "length"],
            "Read a string from a buffer and free the native buffer.\nCombines get-string + deallocate to avoid holding Buffer reference across GC.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/create-array",
            ffi_create_array as *const u8,
            true,
            &["size"],
            "Create a new FFI buffer/array of the given size in bytes.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/copy-bytes",
            ffi_copy_bytes as *const u8,
            false,
            &["src", "src_offset", "dest", "dest_offset", "length"],
            "Copy length bytes from src+src_offset to dest+dest_offset.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/realloc",
            ffi_realloc as *const u8,
            false,
            &["ptr", "new_size"],
            "Reallocate memory to a new size, preserving contents.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/buffer-size",
            ffi_buffer_size as *const u8,
            false,
            &["buffer"],
            "Get the size of a buffer in bytes.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/write-buffer-offset",
            ffi_write_buffer_offset as *const u8,
            false,
            &["buffer", "offset", "value", "size"],
            "Write a value to a buffer at a given offset.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/translate-bytes",
            ffi_translate_bytes as *const u8,
            false,
            &["buffer", "offset", "length", "table"],
            "Translate bytes using a lookup table.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/reverse-bytes",
            ffi_reverse_bytes as *const u8,
            false,
            &["buffer", "offset", "length"],
            "Reverse bytes in place in a buffer.",
        )?;

        self.add_builtin_with_doc(
            "beagle.ffi/find-byte",
            ffi_find_byte as *const u8,
            false,
            &["buffer", "offset", "length", "byte"],
            "Find the first occurrence of a byte in a buffer.\n\nReturns the offset or -1 if not found.",
        )?;

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
            true,
            3, // stack_pointer, namespace_slot, value
        )?;

        self.add_builtin_function_with_fp(
            "beagle.builtin/store-function-binding",
            store_function_binding as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, namespace_slot, function
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

        // Rich REPL readline: editing, history, completion, highlighting, multi-line
        // Takes (stack_pointer, frame_pointer, prompt_string) — 1 Beagle arg
        self.add_builtin_function_with_fp(
            "beagle.builtin/repl-read-line",
            repl_read_line as *const u8,
            true,
            true,
            3,
        )?;

        // Save REPL history to disk
        self.add_builtin_function(
            "beagle.builtin/repl-save-history",
            repl_save_history as *const u8,
            false,
            0,
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

        // ============================================================================
        // Runtime Evaluation and Introspection
        // ============================================================================

        self.add_builtin_with_doc(
            "beagle.core/eval",
            eval as *const u8,
            true,
            &["code"],
            "Evaluate a string as Beagle code at runtime.\n\nReturns the result of the evaluated expression.\n\nExamples:\n  eval(\"1 + 2\")  ; => 3",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/eval-in-ns",
            eval_in_ns as *const u8,
            true,
            &["code", "namespace"],
            "Evaluate a string as Beagle code in a specific namespace.\n\nThe code is compiled with access to all bindings and imports of the given namespace.\n\nExamples:\n  eval-in-ns(\"my-fn(42)\", \"my_app\")",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/sleep",
            sleep as *const u8,
            true,
            &["ms"],
            "Pause execution for the specified number of milliseconds.\n\nExamples:\n  (sleep 1000)  ; sleep for 1 second",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/time-now",
            time_now as *const u8,
            false,
            &[],
            "Return the current time in milliseconds since the Unix epoch.\n\nUseful for timing operations or generating timestamps.",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/thread-id",
            thread_id as *const u8,
            false,
            &[],
            "Return the ID of the current thread as an integer.",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/get-cpu-count",
            get_cpu_count as *const u8,
            false,
            &[],
            "Return the number of CPU cores available on the system.",
        )?;

        // Event loop builtins for async I/O
        self.add_builtin_function(
            "beagle.core/event-loop-create",
            event_loop_create as *const u8,
            false,
            1,
        )?;
        self.add_builtin_function(
            "beagle.core/event-loop-create-threaded",
            event_loop_create_threaded as *const u8,
            false,
            1,
        )?;
        self.add_builtin_function(
            "beagle.core/event-loop-run-once",
            event_loop_run_once as *const u8,
            true,
            3, // stack_pointer + loop_id + timeout_ms (frame_pointer added by adjustment)
        )?;
        self.add_builtin_function(
            "beagle.core/event-loop-wake",
            event_loop_wake as *const u8,
            false,
            1,
        )?;
        self.add_builtin_function(
            "beagle.core/event-loop-destroy",
            event_loop_destroy as *const u8,
            false,
            1,
        )?;

        // TCP networking builtins
        self.add_builtin_function_with_fp(
            "beagle.core/tcp-connect-async",
            tcp_connect_async as *const u8,
            true,
            true,
            6, // stack_pointer, frame_pointer, loop_id, host, port, future_atom
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/tcp-listen",
            tcp_listen as *const u8,
            true,
            true,
            6, // stack_pointer, frame_pointer, loop_id, host, port, backlog
        )?;
        self.add_builtin_function(
            "beagle.core/tcp-accept-async",
            tcp_accept_async as *const u8,
            false,
            3, // loop_id, listener_id, future_atom
        )?;
        self.add_builtin_function(
            "beagle.core/tcp-read-async",
            tcp_read_async as *const u8,
            false,
            4, // loop_id, socket_id, buffer_size, future_atom
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/tcp-write-async",
            tcp_write_async as *const u8,
            true,
            true,
            6, // stack_pointer, frame_pointer, loop_id, socket_id, data, future_atom
        )?;
        self.add_builtin_function(
            "beagle.core/tcp-close",
            tcp_close as *const u8,
            false,
            2, // loop_id, socket_id
        )?;
        self.add_builtin_function(
            "beagle.core/tcp-close-listener",
            tcp_close_listener as *const u8,
            false,
            2, // loop_id, listener_id
        )?;

        // TCP result polling builtins
        self.add_builtin_function(
            "beagle.core/tcp-results-count",
            tcp_results_count as *const u8,
            false,
            1, // loop_id
        )?;
        self.add_builtin_function(
            "beagle.core/tcp-result-pop",
            tcp_result_pop as *const u8,
            false,
            1, // loop_id
        )?;
        self.add_builtin_function(
            "beagle.core/tcp-result-pop-for-atom",
            tcp_result_pop_for_atom as *const u8,
            false,
            2, // loop_id, future_atom
        )?;
        self.add_builtin_function(
            "beagle.core/tcp-result-pop-for-op-id",
            tcp_result_pop_for_op_id as *const u8,
            false,
            2, // loop_id, op_id
        )?;
        self.add_builtin_function(
            "beagle.core/tcp-result-future-atom",
            tcp_result_future_atom as *const u8,
            false,
            1, // loop_id
        )?;
        self.add_builtin_function(
            "beagle.core/tcp-result-value",
            tcp_result_value as *const u8,
            false,
            1, // loop_id
        )?;
        self.add_builtin_function(
            "beagle.core/tcp-result-value-for-op-id",
            tcp_result_value_for_op_id as *const u8,
            false,
            2, // loop_id, op_id
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/tcp-result-data",
            tcp_result_data as *const u8,
            true,
            false,
            2, // stack_pointer, loop_id
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/tcp-result-data-for-op-id",
            tcp_result_data_for_op_id as *const u8,
            true,
            false,
            3, // stack_pointer, loop_id, op_id
        )?;
        self.add_builtin_function(
            "beagle.core/tcp-result-op-id",
            tcp_result_op_id as *const u8,
            false,
            1, // loop_id
        )?;
        self.add_builtin_function(
            "beagle.core/tcp-result-listener-id",
            tcp_result_listener_id as *const u8,
            false,
            1, // loop_id
        )?;

        // Timer builtins
        self.add_builtin_function(
            "beagle.core/timer-set",
            timer_set as *const u8,
            false,
            3, // loop_id, delay_ms, future_atom
        )?;
        self.add_builtin_function(
            "beagle.core/timer-cancel",
            timer_cancel as *const u8,
            false,
            2, // loop_id, timer_id
        )?;
        self.add_builtin_function(
            "beagle.core/timer-completed-count",
            timer_completed_count as *const u8,
            false,
            1, // loop_id
        )?;
        self.add_builtin_function(
            "beagle.core/timer-pop-completed",
            timer_pop_completed as *const u8,
            false,
            1, // loop_id
        )?;
        self.add_builtin_function(
            "beagle.core/timer-take-completed",
            timer_take_completed as *const u8,
            false,
            2, // loop_id, future_atom
        )?;

        // Handle-based async file I/O builtins (new API)
        self.add_builtin_function_with_fp(
            "beagle.core/file-read-submit",
            file_read_submit as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-write-submit",
            file_write_submit as *const u8,
            true,
            true,
            5, // stack_pointer, frame_pointer, loop_id, path, content -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-delete-submit",
            file_delete_submit as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-stat-submit",
            file_stat_submit as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-readdir-submit",
            file_readdir_submit as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-append-submit",
            file_append_submit as *const u8,
            true,
            true,
            5, // stack_pointer, frame_pointer, loop_id, path, content -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-exists-submit",
            file_exists_submit as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-rename-submit",
            file_rename_submit as *const u8,
            true,
            true,
            5, // stack_pointer, frame_pointer, loop_id, old_path, new_path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-copy-submit",
            file_copy_submit as *const u8,
            true,
            true,
            5, // stack_pointer, frame_pointer, loop_id, src_path, dest_path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-mkdir-submit",
            file_mkdir_submit as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-mkdir-all-submit",
            file_mkdir_all_submit as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-rmdir-submit",
            file_rmdir_submit as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-rmdir-all-submit",
            file_rmdir_all_submit as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-is-dir-submit",
            file_is_dir_submit as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-is-file-submit",
            file_is_file_submit as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, path -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-open-submit",
            file_open_submit as *const u8,
            true,
            true,
            5, // stack_pointer, frame_pointer, loop_id, path, mode -> handle
        )?;
        self.add_builtin_function(
            "beagle.core/file-close-submit",
            file_close_submit as *const u8,
            false,
            2, // loop_id, handle_key -> handle
        )?;
        self.add_builtin_function(
            "beagle.core/file-handle-read-submit",
            file_handle_read_submit as *const u8,
            false,
            3, // loop_id, handle_key, count -> handle
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-handle-write-submit",
            file_handle_write_submit as *const u8,
            true,
            true,
            5, // stack_pointer, frame_pointer, loop_id, handle_key, content -> handle
        )?;
        self.add_builtin_function(
            "beagle.core/file-handle-readline-submit",
            file_handle_readline_submit as *const u8,
            false,
            2, // loop_id, handle_key -> handle
        )?;
        self.add_builtin_function(
            "beagle.core/file-handle-flush-submit",
            file_handle_flush_submit as *const u8,
            false,
            2, // loop_id, handle_key -> handle
        )?;
        self.add_builtin_function(
            "beagle.core/file-results-count",
            file_results_count as *const u8,
            false,
            1, // loop_id
        )?;
        self.add_builtin_function(
            "beagle.core/file-result-ready",
            file_result_ready as *const u8,
            false,
            2, // loop_id, handle -> bool
        )?;
        self.add_builtin_function(
            "beagle.core/file-result-poll-type",
            file_result_poll_type as *const u8,
            false,
            2, // loop_id, handle -> type_code (0 if not ready)
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-result-get-string",
            file_result_get_string as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, handle -> string (consumes result)
        )?;
        self.add_builtin_function(
            "beagle.core/file-result-get-value",
            file_result_get_value as *const u8,
            false,
            2, // loop_id, handle -> int (consumes result)
        )?;
        self.add_builtin_function(
            "beagle.core/file-result-consume",
            file_result_consume as *const u8,
            false,
            2, // loop_id, handle -> bool (consumes result)
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/file-result-get-entries",
            file_result_get_entries as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, loop_id, handle -> [string] (consumes result)
        )?;

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

        // String functions
        self.add_builtin_with_doc(
            "beagle.core/substring",
            substring as *const u8,
            true,
            &["string", "start", "end"],
            "Extract a substring from a string.\n\nArguments:\n  string - The source string\n  start  - Starting index (0-based, inclusive)\n  end    - Ending index (exclusive)\n\nExamples:\n  (substring \"hello\" 1 4)  ; => \"ell\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/uppercase",
            uppercase as *const u8,
            true,
            &["string"],
            "Convert a string to uppercase.\n\nExamples:\n  (uppercase \"hello\")  ; => \"HELLO\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/lowercase",
            lowercase as *const u8,
            true,
            &["string"],
            "Convert a string to lowercase.\n\nExamples:\n  (lowercase \"HELLO\")  ; => \"hello\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/split",
            split as *const u8,
            true,
            &["string", "delimiter"],
            "Split a string into an array of substrings.\n\nExamples:\n  (split \"a,b,c\" \",\")  ; => [\"a\", \"b\", \"c\"]",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/trim",
            trim as *const u8,
            true,
            &["string"],
            "Remove leading and trailing whitespace from a string.\n\nExamples:\n  (trim \"  hello  \")  ; => \"hello\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/trim-left",
            trim_left as *const u8,
            true,
            &["string"],
            "Remove leading whitespace from a string.\n\nExamples:\n  (trim-left \"  hello  \")  ; => \"hello  \"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/trim-right",
            trim_right as *const u8,
            true,
            &["string"],
            "Remove trailing whitespace from a string.\n\nExamples:\n  (trim-right \"  hello  \")  ; => \"  hello\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/starts-with?",
            starts_with as *const u8,
            true,
            &["string", "prefix"],
            "Check if a string starts with a given prefix.\n\nExamples:\n  (starts-with? \"hello\" \"he\")  ; => true",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/ends-with?",
            ends_with as *const u8,
            true,
            &["string", "suffix"],
            "Check if a string ends with a given suffix.\n\nExamples:\n  (ends-with? \"hello\" \"lo\")  ; => true",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/contains?",
            string_contains as *const u8,
            true,
            &["string", "substr"],
            "Check if a string contains a substring.\n\nExamples:\n  (contains? \"hello\" \"ell\")  ; => true",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/index-of",
            index_of as *const u8,
            true,
            &["string", "substr"],
            "Find the first index of a substring in a string.\n\nReturns -1 if not found.\n\nExamples:\n  (index-of \"hello\" \"l\")  ; => 2",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/last-index-of",
            last_index_of as *const u8,
            true,
            &["string", "substr"],
            "Find the last index of a substring in a string.\n\nReturns -1 if not found.\n\nExamples:\n  (last-index-of \"hello\" \"l\")  ; => 3",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/replace",
            replace_string as *const u8,
            true,
            &["string", "from", "to"],
            "Replace all occurrences of a substring.\n\nExamples:\n  (replace \"hello\" \"l\" \"L\")  ; => \"heLLo\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/blank?",
            blank_string as *const u8,
            true,
            &["string"],
            "Check if a string is empty or contains only whitespace.\n\nExamples:\n  (blank? \"\")       ; => true\n  (blank? \"  \")     ; => true\n  (blank? \"hello\")  ; => false",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/replace-first",
            replace_first_string as *const u8,
            true,
            &["string", "from", "to"],
            "Replace the first occurrence of a substring.\n\nExamples:\n  (replace-first \"hello\" \"l\" \"L\")  ; => \"heLlo\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/pad-left",
            pad_left_string as *const u8,
            true,
            &["string", "width", "pad_char"],
            "Pad a string on the left to a given width.\n\nExamples:\n  (pad-left \"42\" 5 \"0\")  ; => \"00042\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/pad-right",
            pad_right_string as *const u8,
            true,
            &["string", "width", "pad_char"],
            "Pad a string on the right to a given width.\n\nExamples:\n  (pad-right \"42\" 5 \"0\")  ; => \"42000\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/lines",
            lines_string as *const u8,
            true,
            &["string"],
            "Split a string into an array of lines.\n\nExamples:\n  (lines \"a\\nb\\nc\")  ; => [\"a\", \"b\", \"c\"]",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/words",
            words_string as *const u8,
            true,
            &["string"],
            "Split a string into an array of words (whitespace-separated).\n\nExamples:\n  (words \"hello world\")  ; => [\"hello\", \"world\"]",
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

        // write_full_file: write string to file directly (fast path)
        self.add_builtin_function_with_fp(
            "beagle.core/write-full-file",
            write_full_file as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, file_name, content
        )?;

        // ============================================================================
        // Filesystem builtins for async I/O
        // ============================================================================

        // fs-unlink: delete a file
        self.add_builtin_function_with_fp(
            "beagle.core/fs-unlink",
            fs_unlink as *const u8,
            true,
            true,
            3, // stack_pointer, frame_pointer, path
        )?;

        // fs-access: check file accessibility
        self.add_builtin_function_with_fp(
            "beagle.core/fs-access",
            fs_access as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, path, mode
        )?;

        // fs-mkdir: create a directory
        self.add_builtin_function_with_fp(
            "beagle.core/fs-mkdir",
            fs_mkdir as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, path, mode
        )?;

        // fs-rmdir: remove an empty directory
        self.add_builtin_function_with_fp(
            "beagle.core/fs-rmdir",
            fs_rmdir as *const u8,
            true,
            true,
            3, // stack_pointer, frame_pointer, path
        )?;

        // fs-rename: rename/move a file or directory
        self.add_builtin_function_with_fp(
            "beagle.core/fs-rename",
            fs_rename as *const u8,
            true,
            true,
            4, // stack_pointer, frame_pointer, old_path, new_path
        )?;

        // fs-is-directory: check if path is a directory
        self.add_builtin_function_with_fp(
            "beagle.core/fs-is-directory?",
            fs_is_directory as *const u8,
            true,
            true,
            3, // stack_pointer, frame_pointer, path
        )?;

        // fs-is-file: check if path is a regular file
        self.add_builtin_function_with_fp(
            "beagle.core/fs-is-file?",
            fs_is_file as *const u8,
            true,
            true,
            3, // stack_pointer, frame_pointer, path
        )?;

        // fs-readdir: list directory contents
        self.add_builtin_function_with_fp(
            "beagle.core/fs-readdir",
            fs_readdir as *const u8,
            true,
            true,
            3, // stack_pointer, frame_pointer, path
        )?;

        // fs-file-size: get file size in bytes
        self.add_builtin_function_with_fp(
            "beagle.core/fs-file-size",
            fs_file_size as *const u8,
            true,
            true,
            3, // stack_pointer, frame_pointer, path
        )?;

        // ============================================================================
        // Future waiting builtins for efficient async await
        // ============================================================================

        // future-wait: wait for any future to complete with timeout
        self.add_builtin_function(
            "beagle.core/future-wait",
            future_wait as *const u8,
            false,
            1, // timeout_ms
        )?;

        // future-notify: notify all waiters that a future completed
        self.add_builtin_function(
            "beagle.core/future-notify",
            future_notify as *const u8,
            false,
            0,
        )?;

        // Diagnostic system builtins
        self.add_builtin_function_with_fp(
            "beagle.core/diagnostics",
            diagnostics as *const u8,
            true,
            true,
            2, // stack_pointer, frame_pointer
        )?;

        self.add_builtin_function_with_fp(
            "beagle.core/diagnostics-for-file",
            diagnostics_for_file as *const u8,
            true,
            true,
            3, // stack_pointer, frame_pointer, file_path
        )?;

        self.add_builtin_function_with_fp(
            "beagle.core/files-with-diagnostics",
            files_with_diagnostics as *const u8,
            true,
            true,
            2, // stack_pointer, frame_pointer
        )?;

        self.add_builtin_function_with_fp(
            "beagle.core/clear-diagnostics",
            clear_diagnostics as *const u8,
            true,
            true,
            2, // stack_pointer, frame_pointer
        )?;

        self.install_rust_collection_builtins()?;
        self.install_regex_builtins()?;

        // Effect handler builtins
        self.add_builtin_function(
            "beagle.builtin/push-handler",
            push_handler_builtin as *const u8,
            false,
            2, // protocol_key_str, handler_instance
        )?;

        self.add_builtin_function(
            "beagle.builtin/pop-handler",
            pop_handler_builtin as *const u8,
            false,
            1, // protocol_key_str
        )?;

        // find-handler needs stack_pointer for get_string error handling
        self.add_builtin_function_with_fp(
            "beagle.builtin/find-handler",
            find_handler_builtin as *const u8,
            true,  // needs stack_pointer
            false, // doesn't need frame_pointer
            2,     // stack_pointer, protocol_key_str
        )?;

        // get-enum-type needs stack_pointer and frame_pointer for GC-safe string allocation
        self.add_builtin_function_with_fp(
            "beagle.builtin/get-enum-type",
            get_enum_type_builtin as *const u8,
            true,
            true,
            3, // stack_pointer, frame_pointer, value
        )?;

        // call-handler calls handler.handle(op, resume) using protocol dispatch
        self.add_builtin_function_with_fp(
            "beagle.builtin/call-handler",
            call_handler_builtin as *const u8,
            true,
            true,
            6, // stack_pointer, frame_pointer, handler, enum_type_ptr, op_value, resume
        )?;

        self.add_builtin_function_with_fp(
            "beagle.builtin/perform-effect",
            perform_effect_runtime_with_saved_regs as *const u8,
            true,
            true,
            7, // stack_pointer, frame_pointer, enum_type_ptr, op_value, resume_address, result_local_offset, saved_regs_ptr
        )?;

        // ====================================================================
        // Reflection builtins for docstrings and namespace introspection
        // ====================================================================

        // ============================================================================
        // Reflect API - Type-centric Introspection (beagle.reflect namespace)
        // ============================================================================

        self.add_builtin_with_doc(
            "beagle.reflect/type-of",
            reflect_type_of as *const u8,
            true,
            &["value"],
            "Get a type descriptor for any value.\n\nReturns a type descriptor that can be used with other reflect functions.\n\nExamples:\n  (reflect/type-of 42)        ; => <type Int>\n  (reflect/type-of \"hello\")   ; => <type String>",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/kind",
            reflect_kind as *const u8,
            true,
            &["descriptor"],
            "Get the type kind from a type descriptor.\n\nReturns a keyword: :struct, :enum, :function, or :primitive.\n\nExamples:\n  (reflect/kind (reflect/type-of some-struct))  ; => :struct",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/name",
            reflect_name as *const u8,
            true,
            &["descriptor"],
            "Get the type name from a type descriptor.\n\nReturns the name as a string.\n\nExamples:\n  (reflect/name (reflect/type-of 42))  ; => \"Int\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/doc",
            reflect_doc as *const u8,
            true,
            &["descriptor"],
            "Get the docstring for a type or function.\n\nReturns the documentation string or null if none available.\n\nExamples:\n  (reflect/doc (reflect/type-of some-fn))  ; => \"Documentation...\"\n  (reflect/doc (reflect/type-of 42))  ; => null",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/fields",
            reflect_fields as *const u8,
            true,
            &["descriptor"],
            "Get the field names for a struct type.\n\nReturns a vector of field names, or null for non-struct types.\n\nExamples:\n  (reflect/fields (reflect/type-of my-struct))  ; => [\"field1\" \"field2\"]",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/variants",
            reflect_variants as *const u8,
            true,
            &["descriptor"],
            "Get the variant names for an enum type.\n\nReturns a vector of variant names, or null for non-enum types.\n\nExamples:\n  (reflect/variants (reflect/type-of Result.Ok))  ; => [\"Ok\" \"Err\"]",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/args",
            reflect_args as *const u8,
            true,
            &["descriptor"],
            "Get the argument names for a function.\n\nReturns a vector of argument names, or null for non-function types.\n\nExamples:\n  (reflect/args (reflect/type-of println))  ; => [\"value\"]",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/variadic?",
            reflect_variadic as *const u8,
            true,
            &["descriptor"],
            "Check if a function accepts variable arguments.\n\nReturns true if the function is variadic, false otherwise.\n\nExamples:\n  (reflect/variadic? (reflect/type-of +))  ; => true",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/info",
            reflect_info as *const u8,
            true,
            &["descriptor"],
            "Get complete type information as a map.\n\nReturns a map containing all available metadata about the type,\nincluding kind, name, docstring, fields/variants/args as appropriate.\n\nExamples:\n  (reflect/info (reflect/type-of my-fn))\n  ; => {:kind :function :name \"my-fn\" :args [...] ...}",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/struct?",
            reflect_is_struct as *const u8,
            true,
            &["value"],
            "Check if a value is a struct type or instance.\n\nExamples:\n  (reflect/struct? my-struct-instance)  ; => true\n  (reflect/struct? 42)  ; => false",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/enum?",
            reflect_is_enum as *const u8,
            true,
            &["value"],
            "Check if a value is an enum type or variant.\n\nExamples:\n  (reflect/enum? Result.Ok)  ; => true\n  (reflect/enum? 42)  ; => false",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/function?",
            reflect_is_function as *const u8,
            true,
            &["value"],
            "Check if a value is a function.\n\nExamples:\n  (reflect/function? println)  ; => true\n  (reflect/function? 42)  ; => false",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/primitive?",
            reflect_is_primitive as *const u8,
            true,
            &["value"],
            "Check if a value is a primitive type (Int, Float, String, Bool, Null).\n\nExamples:\n  (reflect/primitive? 42)  ; => true\n  (reflect/primitive? [1 2 3])  ; => false",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/namespace-members",
            reflect_namespace_members as *const u8,
            true,
            &["namespace-name"],
            "List all members defined in a namespace.\n\nReturns a vector of member names.\n\nExamples:\n  (reflect/namespace-members \"beagle.core\")  ; => [\"println\" \"map\" ...]",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/all-namespaces",
            reflect_all_namespaces as *const u8,
            true,
            &[],
            "List all namespace names in the runtime.\n\nReturns a vector of namespace name strings.\n\nExamples:\n  (reflect/all-namespaces)  ; => [\"beagle.core\" \"beagle.fs\" ...]",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/apropos",
            reflect_apropos as *const u8,
            true,
            &["query"],
            "Search for functions by name or docstring substring.\n\nReturns a vector of matching function names.\n\nExamples:\n  (reflect/apropos \"print\")  ; => [\"println\" \"print\" ...]",
        )?;

        self.add_builtin_with_doc(
            "beagle.reflect/namespace-info",
            reflect_namespace_info as *const u8,
            true,
            &["namespace-name"],
            "Get detailed information about a namespace.\n\nReturns a map with the namespace's functions, structs, and enums.\n\nExamples:\n  (reflect/namespace-info \"beagle.core\")\n  ; => {:functions [...] :structs [...] :enums [...]}",
        )?;

        // ============================================================================
        // JSON Serialization builtins
        // ============================================================================

        self.add_builtin_with_doc(
            "beagle.core/json-encode",
            json_encode as *const u8,
            true,
            &["value"],
            "Serialize a Beagle value to a JSON string.\n\nSupports primitives, vectors, maps, and nested structures.\n\nExamples:\n  (json-encode {:name \"alice\" :age 30})\n  ; => \"{\\\"name\\\":\\\"alice\\\",\\\"age\\\":30}\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.core/json-decode",
            json_decode as *const u8,
            true,
            &["json-string"],
            "Parse a JSON string to a Beagle value.\n\nJSON objects become maps with string keys (use `get` to access).\nJSON arrays become vectors.\n\nExamples:\n  (let data (json-decode \"{\\\"name\\\":\\\"alice\\\"}\"))\n  (get data \"name\")  ; => \"alice\"",
        )?;

        Ok(())
    }

    pub fn install_rust_collection_builtins(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use super::collections::*;

        // Register the namespace so it can be imported
        self.reserve_namespace("beagle.collections".to_string());

        // ============================================================================
        // Persistent Vector (HAMT-based immutable vector)
        // ============================================================================

        self.add_builtin_with_doc(
            "beagle.collections/vec",
            rust_vec_empty as *const u8,
            true,
            &[],
            "Create a new empty persistent vector.\n\nPersistent vectors are immutable - all operations return new vectors.\n\nExamples:\n  (let v (collections/vec))",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/vec-count",
            rust_vec_count as *const u8,
            false,
            &["vec"],
            "Return the number of elements in the vector.\n\nExamples:\n  (vec-count (push [] 1 2 3))  ; => 3",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/vec-get",
            rust_vec_get as *const u8,
            false,
            &["vec", "index"],
            "Get the element at index. Returns null if out of bounds.\n\nExamples:\n  (vec-get [1 2 3] 1)  ; => 2",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/vec-push",
            rust_vec_push as *const u8,
            true,
            &["vec", "value"],
            "Return a new vector with value appended.\n\nExamples:\n  (vec-push [1 2] 3)  ; => [1 2 3]",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/vec-assoc",
            rust_vec_assoc as *const u8,
            true,
            &["vec", "index", "value"],
            "Return a new vector with the value at index replaced.\n\nExamples:\n  (vec-assoc [1 2 3] 1 99)  ; => [1 99 3]",
        )?;

        // ============================================================================
        // Persistent Map (HAMT-based immutable hash map)
        // ============================================================================

        self.add_builtin_with_doc(
            "beagle.collections/map",
            rust_map_empty as *const u8,
            true,
            &[],
            "Create a new empty persistent map.\n\nPersistent maps are immutable - all operations return new maps.\n\nExamples:\n  (let m (collections/map))",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/map-count",
            rust_map_count as *const u8,
            false,
            &["m"],
            "Return the number of key-value pairs in the map.\n\nExamples:\n  (map-count {:a 1 :b 2})  ; => 2",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/map-get",
            rust_map_get as *const u8,
            false,
            &["m", "key"],
            "Get the value for key. Returns null if not found.\n\nExamples:\n  (map-get {:a 1} :a)  ; => 1\n  (map-get {:a 1} :b)  ; => null",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/map-assoc",
            rust_map_assoc as *const u8,
            true,
            &["m", "key", "value"],
            "Return a new map with the key-value pair added or updated.\n\nExamples:\n  (map-assoc {:a 1} :b 2)  ; => {:a 1 :b 2}",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/map-keys",
            rust_map_keys as *const u8,
            true,
            &["m"],
            "Return a vector of all keys in the map.\n\nExamples:\n  (map-keys {:a 1 :b 2})  ; => [:a :b]",
        )?;

        // ============================================================================
        // Persistent Set (HAMT-based immutable hash set)
        // ============================================================================

        self.add_builtin_with_doc(
            "beagle.collections/set",
            rust_set_empty as *const u8,
            true,
            &[],
            "Create a new empty persistent set.\n\nPersistent sets are immutable - all operations return new sets.\n\nExamples:\n  (let s (collections/set))",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/set-count",
            rust_set_count as *const u8,
            false,
            &["s"],
            "Return the number of elements in the set.\n\nExamples:\n  (set-count #{1 2 3})  ; => 3",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/set-contains?",
            rust_set_contains as *const u8,
            false,
            &["s", "element"],
            "Return true if the set contains the element.\n\nExamples:\n  (set-contains? #{1 2 3} 2)  ; => true",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/set-add",
            rust_set_add as *const u8,
            true,
            &["s", "element"],
            "Return a new set with the element added.\n\nExamples:\n  (set-add #{1 2} 3)  ; => #{1 2 3}",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/set-elements",
            rust_set_elements as *const u8,
            true,
            &["s"],
            "Return a vector of all elements in the set.\n\nExamples:\n  (set-elements #{1 2 3})  ; => [1 2 3]",
        )?;

        // ============================================================================
        // Mutable Map (open-addressing hash table)
        // ============================================================================

        self.add_builtin_with_doc(
            "beagle.collections/mutable-map",
            rust_mutable_map_empty as *const u8,
            true,
            &[],
            "Create a new empty mutable map with default capacity (16).\n\nMutable maps are modified in place for high-performance scenarios.\n\nExamples:\n  (let m (collections/mutable-map))",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/mutable-map-with-capacity",
            rust_mutable_map_with_capacity as *const u8,
            true,
            &["capacity"],
            "Create a new empty mutable map with the given capacity hint.\n\nExamples:\n  (let m (collections/mutable-map-with-capacity 1024))",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/mutable-map-put!",
            rust_mutable_map_put as *const u8,
            true,
            &["m", "key", "value"],
            "Insert or update a key-value pair in the mutable map. Mutates in place.\n\nExamples:\n  (mutable-map-put! m \"key\" 42)",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/mutable-map-get",
            rust_mutable_map_get as *const u8,
            false,
            &["m", "key"],
            "Get the value for key. Returns null if not found.\n\nExamples:\n  (mutable-map-get m \"key\")  ; => 42",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/mutable-map-increment!",
            rust_mutable_map_increment as *const u8,
            true,
            &["m", "key"],
            "Increment the integer value for key by 1, inserting 1 if absent.\n\nExamples:\n  (mutable-map-increment! m \"key\")",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/mutable-map-count",
            rust_mutable_map_count as *const u8,
            false,
            &["m"],
            "Return the number of key-value pairs in the mutable map.\n\nExamples:\n  (mutable-map-count m)  ; => 3",
        )?;

        self.add_builtin_with_doc(
            "beagle.collections/mutable-map-entries",
            rust_mutable_map_entries as *const u8,
            true,
            &["m"],
            "Return an array of [key, value] pairs from the mutable map.\n\nExamples:\n  (mutable-map-entries m)  ; => [[\"a\" 1] [\"b\" 2]]",
        )?;

        Ok(())
    }

    pub fn install_regex_builtins(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use super::regex::*;

        // Register the namespace so it can be imported
        self.reserve_namespace("beagle.regex".to_string());

        // ============================================================================
        // Regular Expressions
        // ============================================================================

        self.add_builtin_with_doc(
            "beagle.regex/compile",
            regex_compile as *const u8,
            true,
            &["pattern"],
            "Compile a regular expression pattern.\n\nReturns a Regex object that can be used with other regex functions.\n\nExamples:\n  (let re (regex/compile \"[0-9]+\"))",
        )?;

        self.add_builtin_with_doc(
            "beagle.regex/matches?",
            regex_matches as *const u8,
            false,
            &["regex", "string"],
            "Check if the entire string matches the regex.\n\nExamples:\n  (regex/matches? (regex/compile \"[0-9]+\") \"123\")  ; => true\n  (regex/matches? (regex/compile \"[0-9]+\") \"abc\")  ; => false",
        )?;

        self.add_builtin_with_doc(
            "beagle.regex/find",
            regex_find as *const u8,
            true,
            &["regex", "string"],
            "Find the first match in the string.\n\nReturns a map with :start, :end, and :match keys, or null if no match.\n\nExamples:\n  (regex/find (regex/compile \"[0-9]+\") \"abc123def\")\n  ; => {:start 3 :end 6 :match \"123\"}",
        )?;

        self.add_builtin_with_doc(
            "beagle.regex/find-all",
            regex_find_all as *const u8,
            true,
            &["regex", "string"],
            "Find all matches in the string.\n\nReturns a vector of match maps.\n\nExamples:\n  (regex/find-all (regex/compile \"[0-9]+\") \"a1b2c3\")\n  ; => [{:start 1 :end 2 :match \"1\"} ...]",
        )?;

        self.add_builtin_with_doc(
            "beagle.regex/replace",
            regex_replace as *const u8,
            true,
            &["regex", "string", "replacement"],
            "Replace the first match in the string with the replacement.\n\nExamples:\n  (regex/replace (regex/compile \"[0-9]+\") \"a1b2c3\" \"X\")\n  ; => \"aXb2c3\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.regex/replace-all",
            regex_replace_all as *const u8,
            true,
            &["regex", "string", "replacement"],
            "Replace all matches in the string with the replacement.\n\nExamples:\n  (regex/replace-all (regex/compile \"[0-9]+\") \"a1b2c3\" \"X\")\n  ; => \"aXbXcX\"",
        )?;

        self.add_builtin_with_doc(
            "beagle.regex/split",
            regex_split as *const u8,
            true,
            &["regex", "string"],
            "Split a string by the regex pattern.\n\nReturns a vector of strings.\n\nExamples:\n  (regex/split (regex/compile \",\\\\s*\") \"a, b, c\")\n  ; => [\"a\" \"b\" \"c\"]",
        )?;

        self.add_builtin_with_doc(
            "beagle.regex/captures",
            regex_captures as *const u8,
            true,
            &["regex", "string"],
            "Get capture groups from the first match.\n\nReturns a vector of captured strings (index 0 is the full match), or null if no match.\n\nExamples:\n  (regex/captures (regex/compile \"(\\\\w+)@(\\\\w+)\") \"user@host\")\n  ; => [\"user@host\" \"user\" \"host\"]",
        )?;

        self.add_builtin_with_doc(
            "beagle.regex/is-regex?",
            is_regex as *const u8,
            false,
            &["value"],
            "Check if a value is a compiled regex.\n\nExamples:\n  (regex/is-regex? (regex/compile \"test\"))  ; => true\n  (regex/is-regex? \"test\")  ; => false",
        )?;

        Ok(())
    }
}
