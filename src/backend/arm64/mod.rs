//! ARM64 (AArch64) code generation backend for Beagle.
//!
//! This module wraps the existing `LowLevelArm` implementation to conform
//! to the `CodegenBackend` trait, allowing it to be used interchangeably
//! with other backends.

use std::collections::HashMap;

use crate::arm::{FmovDirection, LowLevelArm};
use crate::backend::CodegenBackend;
use crate::common::Label;
use crate::compiler::CompileError;
use crate::ir::Condition;
use crate::machine_code::arm_codegen::{ArmAsm, Register, X29, X30, ZERO_REGISTER};

/// ARM64 backend implementation.
///
/// This struct wraps `LowLevelArm` and implements `CodegenBackend`,
/// delegating all operations to the underlying implementation.
#[derive(Debug)]
pub struct Arm64Backend {
    inner: LowLevelArm,
}

impl Default for Arm64Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl CodegenBackend for Arm64Backend {
    type Register = Register;
    type Instruction = ArmAsm;

    fn new() -> Self {
        Arm64Backend {
            inner: LowLevelArm::new(),
        }
    }

    // === Lifecycle methods ===

    fn prelude(&mut self) {
        self.inner.prelude();
    }

    fn epilogue(&mut self) {
        self.inner.epilogue();
    }

    fn compile_to_bytes(&mut self) -> Vec<u8> {
        self.inner.compile_to_bytes()
    }

    // === Register management ===

    fn arg(&self, index: u8) -> Self::Register {
        self.inner.arg(index)
    }

    fn num_arg_registers(&self) -> usize {
        8 // X0-X7
    }

    fn ret_reg(&self) -> Self::Register {
        self.inner.ret_reg()
    }

    fn arg_count_reg(&self) -> Self::Register {
        crate::machine_code::arm_codegen::X9
    }

    fn volatile_register(&mut self) -> Self::Register {
        self.inner.volatile_register()
    }

    fn temporary_register(&mut self) -> Self::Register {
        self.inner.temporary_register()
    }

    fn free_temporary_register(&mut self, register: Self::Register) {
        self.inner.free_temporary_register(register);
    }

    fn free_register(&mut self, register: Self::Register) {
        self.inner.free_register(register);
    }

    fn reserve_register(&mut self, register: Self::Register) {
        self.inner.reserve_register(register);
    }

    fn clear_temporary_registers(&mut self) {
        self.inner.clear_temporary_registers();
    }

    // === Arithmetic operations ===

    fn add(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.add(dest, a, b);
    }

    fn sub(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.sub(dest, a, b);
    }

    fn sub_imm(&mut self, dest: Self::Register, a: Self::Register, imm: i32) {
        self.inner.sub_imm(dest, a, imm);
    }

    fn mul(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.mul(dest, a, b);
    }

    fn div(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.div(dest, a, b);
    }

    fn modulo(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.modulo(dest, a, b);
    }

    // === Bitwise operations ===

    fn and(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.and(dest, a, b);
    }

    fn and_imm(&mut self, dest: Self::Register, a: Self::Register, imm: u64) {
        self.inner.and_imm(dest, a, imm);
    }

    fn or(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.or(dest, a, b);
    }

    fn xor(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.xor(dest, a, b);
    }

    // === Shift operations ===

    fn shift_left(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.shift_left(dest, a, b);
    }

    fn shift_left_imm(&mut self, dest: Self::Register, a: Self::Register, imm: i32) {
        self.inner.shift_left_imm(dest, a, imm);
    }

    fn shift_right(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.shift_right(dest, a, b);
    }

    fn shift_right_imm(&mut self, dest: Self::Register, a: Self::Register, imm: i32) {
        self.inner.shift_right_imm(dest, a, imm);
    }

    fn shift_right_zero(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.shift_right_zero(dest, a, b);
    }

    // === Floating point operations ===

    fn fadd(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.fadd(dest, a, b);
    }

    fn fsub(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.fsub(dest, a, b);
    }

    fn fmul(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.fmul(dest, a, b);
    }

    fn fdiv(&mut self, dest: Self::Register, a: Self::Register, b: Self::Register) {
        self.inner.fdiv(dest, a, b);
    }

    fn fmov_to_float(&mut self, dest: Self::Register, src: Self::Register) {
        self.inner
            .fmov(dest, src, FmovDirection::FromGeneralToFloat);
    }

    fn fmov_from_float(&mut self, dest: Self::Register, src: Self::Register) {
        self.inner
            .fmov(dest, src, FmovDirection::FromFloatToGeneral);
    }

    // === Memory operations - Heap ===

    fn load_from_heap(&mut self, dest: Self::Register, src: Self::Register, offset: i32) {
        self.inner.load_from_heap(dest, src, offset);
    }

    fn store_on_heap(&mut self, ptr: Self::Register, val: Self::Register, offset: i32) {
        self.inner.store_on_heap(ptr, val, offset);
    }

    fn load_from_heap_with_reg_offset(
        &mut self,
        dest: Self::Register,
        src: Self::Register,
        offset: Self::Register,
    ) {
        self.inner.load_from_heap_with_reg_offset(dest, src, offset);
    }

    fn store_to_heap_with_reg_offset(
        &mut self,
        ptr: Self::Register,
        val: Self::Register,
        offset: Self::Register,
    ) {
        self.inner.store_to_heap_with_reg_offset(ptr, val, offset);
    }

    // === Memory operations - Stack ===

    fn push_to_stack(&mut self, reg: Self::Register) {
        self.inner.push_to_stack(reg);
    }

    fn pop_from_stack(&mut self, reg: Self::Register) {
        self.inner.pop_from_stack(reg);
    }

    fn push_to_stack_indexed(&mut self, reg: Self::Register, offset: i32) {
        self.inner.push_to_end_of_stack(reg, offset);
    }

    fn pop_from_stack_indexed(&mut self, reg: Self::Register, offset: i32) {
        self.inner.pop_from_stack_indexed(reg, offset);
    }

    fn pop_from_stack_indexed_raw(&mut self, reg: Self::Register, offset: i32) {
        self.inner.pop_from_stack_indexed_raw(reg, offset);
    }

    fn push_to_end_of_stack(&mut self, reg: Self::Register, offset: i32) {
        self.inner.push_to_end_of_stack(reg, offset);
    }

    fn load_local(&mut self, dest: Self::Register, offset: i32) {
        self.inner.load_local(dest, offset);
    }

    fn store_local(&mut self, src: Self::Register, offset: i32) {
        self.inner.store_local(src, offset);
    }

    fn load_from_stack_beginning(&mut self, dest: Self::Register, offset: i32) {
        self.inner.load_from_stack_beginning(dest, offset);
    }

    // === Stack pointer operations ===

    fn get_stack_pointer(&mut self, dest: Self::Register, offset: Self::Register) {
        self.inner.get_stack_pointer(dest, offset);
    }

    fn get_stack_pointer_imm(&mut self, dest: Self::Register, offset: isize) {
        self.inner.get_stack_pointer_imm(dest, offset);
    }

    fn get_current_stack_position(&mut self, dest: Self::Register) {
        self.inner.get_current_stack_position(dest);
    }

    fn add_stack_pointer(&mut self, bytes: i32) {
        self.inner.add_stack_pointer(bytes);
    }

    fn sub_stack_pointer(&mut self, bytes: i32) {
        self.inner.sub_stack_pointer(bytes);
    }

    // === Move/load operations ===

    fn mov(&mut self, dest: Self::Register, imm: u16) {
        self.inner.mov(dest, imm);
    }

    fn mov_64(&mut self, dest: Self::Register, imm: isize) {
        self.inner.mov_64(dest, imm);
    }

    fn mov_reg(&mut self, dest: Self::Register, src: Self::Register) {
        self.inner.mov_reg(dest, src);
    }

    // === Control flow ===

    fn ret(&mut self) {
        self.inner.ret();
    }

    fn call(&mut self, register: Self::Register) {
        self.inner.call(register);
    }

    fn call_builtin(&mut self, register: Self::Register) {
        self.inner.call_builtin(register);
    }

    fn recurse(&mut self, label: Label) {
        self.inner.recurse(label);
    }

    fn jump(&mut self, label: Label) {
        self.inner.jump(label);
    }

    fn jump_equal(&mut self, label: Label) {
        self.inner.jump_equal(label);
    }

    fn jump_not_equal(&mut self, label: Label) {
        self.inner.jump_not_equal(label);
    }

    fn jump_greater(&mut self, label: Label) {
        self.inner.jump_greater(label);
    }

    fn jump_greater_or_equal(&mut self, label: Label) {
        self.inner.jump_greater_or_equal(label);
    }

    fn jump_less(&mut self, label: Label) {
        self.inner.jump_less(label);
    }

    fn jump_less_or_equal(&mut self, label: Label) {
        self.inner.jump_less_or_equal(label);
    }

    // === Labels ===

    fn new_label(&mut self, name: &str) -> Label {
        self.inner.new_label(name)
    }

    fn write_label(&mut self, label: Label) {
        self.inner.write_label(label);
    }

    fn load_label_address(&mut self, dest: Self::Register, label: Label) {
        self.inner.load_label_address(dest, label);
    }

    fn get_label_by_name(&self, name: &str) -> Label {
        self.inner.get_label_by_name(name)
    }

    // === Comparison ===

    fn compare(&mut self, a: Self::Register, b: Self::Register) {
        self.inner.compare(a, b);
    }

    fn compare_bool(
        &mut self,
        condition: Condition,
        dest: Self::Register,
        a: Self::Register,
        b: Self::Register,
    ) {
        self.inner.compare_bool(condition, dest, a, b);
    }

    fn compare_float_bool(
        &mut self,
        condition: Condition,
        dest: Self::Register,
        a: Self::Register,
        b: Self::Register,
    ) {
        self.inner.compare_float_bool(condition, dest, a, b);
    }

    // === Tagged value operations ===

    fn tag_value(&mut self, dest: Self::Register, value: Self::Register, tag: Self::Register) {
        self.inner.tag_value(dest, value, tag);
    }

    fn get_tag(&mut self, dest: Self::Register, value: Self::Register) {
        self.inner.get_tag(dest, value);
    }

    fn guard_integer(&mut self, temp: Self::Register, value: Self::Register, error_label: Label) {
        self.inner.guard_integer(temp, value, error_label);
    }

    fn guard_float(&mut self, temp: Self::Register, value: Self::Register, error_label: Label) {
        self.inner.guard_float(temp, value, error_label);
    }

    // === Atomic operations ===

    fn atomic_load(&mut self, dest: Self::Register, src: Self::Register) {
        self.inner.atomic_load(dest, src);
    }

    fn atomic_store(&mut self, ptr: Self::Register, val: Self::Register) {
        self.inner.atomic_store(ptr, val);
    }

    fn compare_and_swap(
        &mut self,
        expected: Self::Register,
        new: Self::Register,
        ptr: Self::Register,
    ) {
        self.inner.compare_and_swap(expected, new, ptr);
    }

    // === Stack management ===

    fn set_max_locals(&mut self, max_locals: usize) {
        self.inner.set_max_locals(max_locals);
    }

    fn increment_stack_size(&mut self, size: i32) {
        self.inner.increment_stack_size(size);
    }

    fn set_all_locals_to_null(&mut self, null_register: Self::Register) {
        self.inner.set_all_locals_to_null(null_register);
    }

    fn reset_callee_saved_tracking(&mut self) {
        self.inner.reset_callee_saved_tracking();
    }

    fn mark_callee_saved_register_used(&mut self, index: usize) {
        self.inner.mark_callee_saved_register_used(index);
    }

    // === Stack map for GC ===

    fn translate_stack_map(&self, base_pointer: usize) -> Vec<(usize, usize)> {
        self.inner.translate_stack_map(base_pointer)
    }

    fn current_byte_offset(&self) -> usize {
        self.inner.current_byte_offset()
    }

    fn record_gc_safepoint(&mut self) {
        self.inner.record_gc_safepoint();
    }

    fn return_address_adjustment() -> usize {
        LowLevelArm::return_address_adjustment()
    }

    // === Debugging ===

    fn breakpoint(&mut self) {
        self.inner.breakpoint();
    }

    fn current_position(&self) -> usize {
        self.inner.current_position()
    }

    // === Backend-specific registers ===

    fn link_register(&self) -> Self::Register {
        X30
    }

    fn load_return_address(&mut self, dest: Self::Register) {
        // On ARM64, return address is in the link register (X30/LR)
        self.mov_reg(dest, X30);
    }

    fn frame_pointer(&self) -> Self::Register {
        X29
    }

    fn get_local_byte_offset(&self, local_index: usize) -> isize {
        // On ARM64, locals are at [FP - (local_index + 1) * 8]
        -(((local_index + 1) * 8) as isize)
    }

    fn zero_register(&self) -> Self::Register {
        ZERO_REGISTER
    }

    // === Public field accessors ===

    fn max_locals(&self) -> i32 {
        self.inner.max_locals
    }

    fn max_stack_size(&self) -> i32 {
        self.inner.max_stack_size
    }

    fn stack_size(&self) -> i32 {
        self.inner.stack_size
    }

    // === Pair operations ===

    fn store_pair(
        &mut self,
        reg1: Self::Register,
        reg2: Self::Register,
        dest: Self::Register,
        offset: i32,
    ) {
        self.inner.store_pair(reg1, reg2, dest, offset);
    }

    fn load_pair(
        &mut self,
        reg1: Self::Register,
        reg2: Self::Register,
        location: Self::Register,
        offset: i32,
    ) {
        self.inner.load_pair(reg1, reg2, location, offset);
    }

    // === Debug info ===

    fn share_label_info_debug(&self, function_pointer: usize) -> Result<(), CompileError> {
        self.inner.share_label_info_debug(function_pointer)
    }

    // === Access to internal state ===

    fn instructions_mut(&mut self) -> &mut Vec<Self::Instruction> {
        &mut self.inner.instructions
    }

    fn label_locations(&self) -> &HashMap<usize, usize> {
        &self.inner.label_locations
    }

    fn labels(&self) -> &Vec<String> {
        &self.inner.labels
    }

    fn get_volatile_register(&self, index: usize) -> Self::Register {
        self.inner.canonical_volatile_registers[index]
    }

    fn register_label_name(&mut self, name: &str) {
        self.inner.register_label_name(name);
    }

    fn register_from_index(&self, index: usize) -> Self::Register {
        Register::from_index(index)
    }
}
