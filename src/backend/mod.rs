//! Pluggable code generation backends for Beagle.
//!
//! This module provides a trait-based abstraction for code generation,
//! allowing different backends (ARM64, x86-64, LLVM, Cranelift) to be
//! selected at compile time via Cargo feature flags.
//!
//! # Usage
//!
//! ```bash
//! # Default (ARM64)
//! cargo run -- program.bg
//!
//! # Explicit backend selection
//! cargo run --features backend-arm64 -- program.bg
//! cargo run --features backend-x86-64 -- program.bg
//! cargo run --features backend-llvm -- program.bg        # future
//! cargo run --features backend-cranelift -- program.bg   # future
//! ```

#[cfg(not(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
)))]
pub mod arm64;

#[cfg(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
))]
pub mod x86_64;

use std::collections::HashMap;

use crate::common::Label;
use crate::ir::Condition;

/// Core trait that all code generation backends must implement.
///
/// This trait abstracts over the differences between target architectures,
/// allowing the IR compiler to generate code without knowing the specifics
/// of the target platform.
pub trait CodegenBackend: Sized {
    /// Register type used by this backend.
    /// For native backends (ARM64, x86-64), this is the physical register type.
    /// For IR-based backends (LLVM, Cranelift), this may be a virtual register ID.
    type Register: Copy + Clone + std::fmt::Debug + PartialEq;

    /// Create a new backend instance.
    fn new() -> Self;

    // === Lifecycle methods ===

    /// Generate function prologue (save frame pointer, link register, allocate stack).
    fn prelude(&mut self);

    /// Generate function epilogue (restore frame pointer, link register, deallocate stack).
    fn epilogue(&mut self);

    /// Compile the accumulated instructions to machine code bytes.
    fn compile_to_bytes(&mut self) -> Vec<u8>;

    // === Register management ===

    /// Get the register for function argument at index (0-7 for ARM64, 0-5 for x86-64).
    fn arg(&self, index: u8) -> Self::Register;

    /// Number of registers available for function arguments.
    /// ARM64: 8 (X0-X7), x86-64: 6 (RDI, RSI, RDX, RCX, R8, R9)
    fn num_arg_registers(&self) -> usize;

    /// Get the return value register.
    fn ret_reg(&self) -> Self::Register;

    /// Allocate a callee-saved (volatile) register.
    fn volatile_register(&mut self) -> Self::Register;

    /// Allocate a temporary (caller-saved) register.
    fn temporary_register(&mut self) -> Self::Register;

    /// Free a previously allocated temporary register.
    fn free_temporary_register(&mut self, register: Self::Register);

    /// Free a previously allocated volatile register.
    fn free_register(&mut self, register: Self::Register);

    /// Reserve a register (mark it as in-use without allocating).
    fn reserve_register(&mut self, register: Self::Register);

    /// Clear all temporary registers (mark them as free).
    fn clear_temporary_registers(&mut self);

    // === Arithmetic operations ===

    fn add(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);
    fn sub(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);
    fn sub_imm(&mut self, dest: Self::Register, a: Self::Register, imm: i32);
    fn mul(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);
    fn div(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);

    // === Bitwise operations ===

    fn and(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);
    fn and_imm(&mut self, dest: Self::Register, a: Self::Register, imm: u64);
    fn or(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);
    fn xor(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);

    // === Shift operations ===

    fn shift_left(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);
    fn shift_left_imm(&mut self, dest: Self::Register, a: Self::Register, imm: i32);
    fn shift_right(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);
    fn shift_right_imm(&mut self, dest: Self::Register, a: Self::Register, imm: i32);
    fn shift_right_zero(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);

    // === Floating point operations ===

    fn fadd(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);
    fn fsub(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);
    fn fmul(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);
    fn fdiv(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register);

    /// Move value from general-purpose register to floating-point register.
    fn fmov_to_float(&mut self, dest: Self::Register, src: Self::Register);

    /// Move value from floating-point register to general-purpose register.
    fn fmov_from_float(&mut self, dest: Self::Register, src: Self::Register);

    // === Memory operations - Heap ===

    fn load_from_heap(&mut self, dest: Self::Register, src: Self::Register, offset: i32);
    fn store_on_heap(&mut self, ptr: Self::Register, val: Self::Register, offset: i32);
    fn load_from_heap_with_reg_offset(
        &mut self,
        dest: Self::Register,
        src: Self::Register,
        offset: Self::Register,
    );
    fn store_to_heap_with_reg_offset(
        &mut self,
        ptr: Self::Register,
        val: Self::Register,
        offset: Self::Register,
    );

    // === Memory operations - Stack ===

    fn push_to_stack(&mut self, reg: Self::Register);
    fn pop_from_stack(&mut self, reg: Self::Register);
    fn push_to_stack_indexed(&mut self, reg: Self::Register, offset: i32);
    fn pop_from_stack_indexed(&mut self, reg: Self::Register, offset: i32);
    fn pop_from_stack_indexed_raw(&mut self, reg: Self::Register, offset: i32);
    fn push_to_end_of_stack(&mut self, reg: Self::Register, offset: i32);
    fn load_local(&mut self, dest: Self::Register, offset: i32);
    fn store_local(&mut self, src: Self::Register, offset: i32);
    fn load_from_stack_beginning(&mut self, dest: Self::Register, offset: i32);

    // === Stack pointer operations ===

    fn get_stack_pointer(&mut self, dest: Self::Register, offset: Self::Register);
    fn get_stack_pointer_imm(&mut self, dest: Self::Register, offset: isize);
    fn get_current_stack_position(&mut self, dest: Self::Register);
    fn add_stack_pointer(&mut self, bytes: i32);
    fn sub_stack_pointer(&mut self, bytes: i32);

    // === Move/load operations ===

    fn mov(&mut self, dest: Self::Register, imm: u16);
    fn mov_64(&mut self, dest: Self::Register, imm: isize);
    fn mov_reg(&mut self, dest: Self::Register, src: Self::Register);

    // === Control flow ===

    fn ret(&mut self);
    fn call(&mut self, register: Self::Register);
    fn call_builtin(&mut self, register: Self::Register);
    fn recurse(&mut self, label: Label);
    fn jump(&mut self, label: Label);
    fn jump_equal(&mut self, label: Label);
    fn jump_not_equal(&mut self, label: Label);
    fn jump_greater(&mut self, label: Label);
    fn jump_greater_or_equal(&mut self, label: Label);
    fn jump_less(&mut self, label: Label);
    fn jump_less_or_equal(&mut self, label: Label);

    // === Labels ===

    fn new_label(&mut self, name: &str) -> Label;
    fn write_label(&mut self, label: Label);
    fn load_label_address(&mut self, dest: Self::Register, label: Label);
    fn get_label_by_name(&self, name: &str) -> Label;

    // === Comparison ===

    fn compare(&mut self, a: Self::Register, b: Self::Register);
    fn compare_bool(
        &mut self,
        condition: Condition,
        dest: Self::Register,
        a: Self::Register,
        b: Self::Register,
    );
    fn compare_float_bool(
        &mut self,
        condition: Condition,
        dest: Self::Register,
        a: Self::Register,
        b: Self::Register,
    );

    // === Tagged value operations (Beagle-specific) ===

    fn tag_value(&mut self, dest: Self::Register, value: Self::Register, tag: Self::Register);
    fn get_tag(&mut self, dest: Self::Register, value: Self::Register);
    fn guard_integer(&mut self, temp: Self::Register, value: Self::Register, error_label: Label);
    fn guard_float(&mut self, temp: Self::Register, value: Self::Register, error_label: Label);

    // === Atomic operations ===

    fn atomic_load(&mut self, dest: Self::Register, src: Self::Register);
    fn atomic_store(&mut self, ptr: Self::Register, val: Self::Register);
    fn compare_and_swap(
        &mut self,
        expected: Self::Register,
        new: Self::Register,
        ptr: Self::Register,
    );

    // === Stack management ===

    fn set_max_locals(&mut self, max_locals: usize);
    fn increment_stack_size(&mut self, size: i32);
    fn set_all_locals_to_null(&mut self, null_register: Self::Register);

    /// Reset the callee-saved register tracking for a new function.
    /// Called before compiling a new function to clear the used register set.
    fn reset_callee_saved_tracking(&mut self);

    /// Mark a callee-saved register as used by index.
    /// This is used by the register allocator to inform the backend which
    /// callee-saved registers are used and need to be saved in the prologue.
    fn mark_callee_saved_register_used(&mut self, index: usize);

    // === Stack map for GC ===

    fn translate_stack_map(&self, base_pointer: usize) -> Vec<(usize, usize)>;

    /// Get the current byte offset in the generated code.
    /// ARM64: instruction_count * 4 (fixed 4-byte instructions)
    /// x86-64: sum of all instruction byte sizes (variable length)
    fn current_byte_offset(&self) -> usize;

    /// Record a GC safepoint at the current position.
    /// Called before emitting call instructions to builtins.
    /// The safepoint records the byte offset and current stack size.
    fn record_gc_safepoint(&mut self);

    /// Get the adjustment needed when looking up return addresses in the stack map.
    /// ARM64: 4 (instruction size)
    /// x86-64: depends on call instruction size (typically 2-3 for indirect call)
    fn return_address_adjustment() -> usize;

    // === Debugging ===

    fn breakpoint(&mut self);
    fn current_position(&self) -> usize;

    // === Backend-specific registers ===

    /// Get the link register (return address). Used for exception handling.
    fn link_register(&self) -> Self::Register;

    /// Load the return address into a destination register.
    /// On ARM64, this copies from the link register (X30/LR).
    /// On x86-64, this loads from [RBP + 8] since there's no link register.
    fn load_return_address(&mut self, dest: Self::Register);

    /// Get the frame pointer register. Used for exception handling.
    fn frame_pointer(&self) -> Self::Register;

    /// Get the byte offset from frame pointer for a local variable.
    /// This is negative (locals are below FP).
    /// Used by exception handling to store the caught exception value.
    fn get_local_byte_offset(&self, local_index: usize) -> isize;

    /// Get the zero register if available (ARM64 has one, x86 doesn't).
    fn zero_register(&self) -> Self::Register;

    // === Public field accessors ===

    fn max_locals(&self) -> i32;
    fn max_stack_size(&self) -> i32;
    fn stack_size(&self) -> i32;

    // === Pair operations (ARM64-specific but abstracted) ===

    fn store_pair(
        &mut self,
        reg1: Self::Register,
        reg2: Self::Register,
        dest: Self::Register,
        offset: i32,
    );
    fn load_pair(
        &mut self,
        reg1: Self::Register,
        reg2: Self::Register,
        location: Self::Register,
        offset: i32,
    );

    // === Debug info ===

    fn share_label_info_debug(
        &self,
        function_pointer: usize,
    ) -> Result<(), crate::compiler::CompileError>;

    // === Access to internal state for IR compiler ===

    fn instructions_mut(&mut self) -> &mut Vec<Self::Instruction>;
    fn label_locations(&self) -> &HashMap<usize, usize>;
    fn labels(&self) -> &Vec<String>;

    /// Get a volatile register for temporary use (doesn't allocate).
    /// This is used for specific positions like the first callee-saved register.
    fn get_volatile_register(&self, index: usize) -> Self::Register;

    /// Register for label name
    fn register_label_name(&mut self, name: &str);

    /// Create a register from an index (used for virtual register to physical mapping).
    fn register_from_index(&self, index: usize) -> Self::Register;

    /// Instruction type for this backend
    type Instruction: std::fmt::Debug;

    /// Set the function name for debugging output (optional, default is no-op)
    fn set_function_name(&mut self, _name: &str) {}
}

// Select the backend type based on feature flags or target architecture
cfg_if::cfg_if! {
    if #[cfg(any(feature = "backend-x86-64", all(target_arch = "x86_64", not(feature = "backend-arm64"))))] {
        pub type Backend = x86_64::X86_64Backend;
    } else if #[cfg(feature = "backend-llvm")] {
        compile_error!("LLVM backend not yet implemented");
    } else if #[cfg(feature = "backend-cranelift")] {
        compile_error!("Cranelift backend not yet implemented");
    } else {
        // Default to ARM64 for aarch64 or when no specific backend is determined
        pub type Backend = arm64::Arm64Backend;
    }
}
