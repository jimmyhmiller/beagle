//! Low-level x86-64 code generation for Beagle
//!
//! This module provides the `LowLevelX86` struct which handles x86-64
//! instruction generation, register allocation, and label management.

use crate::{
    common::Label,
    compiler::CompileError,
    ir::Condition,
    machine_code::x86_codegen::{
        Condition as X86Cond, R8, R9, R10, R11, R12, R13, R14, R15, RAX, RBP, RBX, RCX, RDI, RDX,
        RSI, RSP, X86Asm, X86Register,
    },
    types::BuiltInTypes,
};

use std::collections::HashMap;

/// Low-level x86-64 code generator
#[derive(Debug)]
pub struct LowLevelX86 {
    pub instructions: Vec<X86Asm>,
    pub label_locations: HashMap<usize, usize>,
    pub label_index: usize,
    pub labels: Vec<String>,
    pub canonical_volatile_registers: Vec<X86Register>,
    pub free_volatile_registers: Vec<X86Register>,
    pub allocated_volatile_registers: Vec<X86Register>,
    pub stack_size: i32,
    pub max_stack_size: i32,
    pub max_locals: i32,
    pub stack_map: HashMap<usize, usize>,
    free_temporary_registers: Vec<X86Register>,
    allocated_temporary_registers: Vec<X86Register>,
    canonical_temporary_registers: Vec<X86Register>,
    current_function_name: Option<String>,
    /// Tracks which callee-saved registers are actually used in this function.
    /// This is a bitmask for R12-R15 and RBX (indices 12-16).
    used_callee_saved_registers: u8,
}

impl Default for LowLevelX86 {
    fn default() -> Self {
        Self::new()
    }
}

impl LowLevelX86 {
    pub fn new() -> Self {
        // Register allocation strategy (matching ARM64 approach):
        //
        // We use R12-R15 and RBX as "value storage" registers for the linear scan allocator.
        // Per System V AMD64 ABI, these are callee-saved, but we DON'T save them
        // in prologue/epilogue. Instead, the CallWithSaves mechanism explicitly
        // pushes/pops values that need to survive across function calls.
        //
        // Note: RBX has raw index 3 but uses virtual index 16 to avoid arg(3) conflict.
        //
        // Caller-saved registers:
        // - RAX: return value
        // - RDI, RSI, RDX, RCX, R8, R9: argument registers (used for call args)
        // - R10, R11: scratch registers (NOT argument registers, safe to use anytime)
        //
        // Temporary registers: Only use caller-saved (scratch) registers.
        // R10 and R11 are scratch registers per System V ABI, and RAX is the return value register.
        //
        // IMPORTANT: Argument registers (RDI, RSI, RDX, RCX, R8, R9) must NOT be used as temps
        // because during call setup, we first set the argument registers, then load the function
        // pointer into a temp. If an arg register is used as the function temp, it clobbers
        // the argument we just set.
        let canonical_volatile_registers = vec![R12, R13, R14, R15, RBX];
        let temporary_registers = vec![R10, R11, RAX];

        LowLevelX86 {
            instructions: vec![],
            label_locations: HashMap::new(),
            label_index: 0,
            labels: vec![],
            canonical_volatile_registers: canonical_volatile_registers.clone(),
            canonical_temporary_registers: temporary_registers.clone(),
            free_volatile_registers: canonical_volatile_registers,
            free_temporary_registers: temporary_registers,
            allocated_temporary_registers: vec![],
            allocated_volatile_registers: vec![],
            stack_size: 0,
            max_stack_size: 0,
            max_locals: 0,
            stack_map: HashMap::new(),
            current_function_name: None,
            used_callee_saved_registers: 0,
        }
    }

    pub fn set_function_name(&mut self, name: &str) {
        self.current_function_name = Some(name.to_string());
    }

    pub fn increment_stack_size(&mut self, size: i32) {
        self.stack_size += size;
        if self.stack_size > self.max_stack_size {
            self.max_stack_size = self.stack_size;
        }
    }

    /// Generate function prologue
    pub fn prelude(&mut self) {
        self.instructions.push(X86Asm::Push { reg: RBP });
        self.instructions.push(X86Asm::MovRR {
            dest: RBP,
            src: RSP,
        });
        self.instructions.push(X86Asm::SubRI {
            dest: RSP,
            imm: 0x1111_1111_u32 as i32,
        });
    }

    /// Generate function epilogue
    pub fn epilogue(&mut self) {
        self.instructions.push(X86Asm::AddRI {
            dest: RSP,
            imm: 0x1111_1111_u32 as i32,
        });
        self.instructions.push(X86Asm::Pop { reg: RBP });
    }

    pub fn get_label_index(&mut self) -> usize {
        let current_label_index = self.label_index;
        self.label_index += 1;
        current_label_index
    }

    pub fn breakpoint(&mut self) {
        self.instructions.push(X86Asm::Int3);
    }

    // === Move operations ===

    pub fn mov(&mut self, destination: X86Register, input: u16) {
        self.instructions.push(X86Asm::MovRI32 {
            dest: destination,
            imm: input as i32,
        });
    }

    pub fn mov_64(&mut self, destination: X86Register, input: isize) {
        if input >= i32::MIN as isize && input <= i32::MAX as isize {
            // Use 32-bit sign-extended move for smaller values
            self.instructions.push(X86Asm::MovRI32 {
                dest: destination,
                imm: input as i32,
            });
        } else {
            // Use full 64-bit move
            self.instructions.push(X86Asm::MovRI {
                dest: destination,
                imm: input as i64,
            });
        }
    }

    pub fn mov_reg(&mut self, destination: X86Register, source: X86Register) {
        if destination != source {
            self.instructions.push(X86Asm::MovRR {
                dest: destination,
                src: source,
            });
        }
    }

    // === Arithmetic operations ===

    pub fn add(&mut self, destination: X86Register, a: X86Register, b: X86Register) {
        // x86 ADD is destructive: dest = dest + src
        // We need to handle cases where dest might clobber operands
        if destination == a {
            // Simple case: dest = a + b -> ADD dest, b
            self.instructions.push(X86Asm::AddRR {
                dest: destination,
                src: b,
            });
        } else if destination == b {
            // dest = a + b where dest == b
            // ADD is commutative, so: dest = b + a -> ADD dest, a
            self.instructions.push(X86Asm::AddRR {
                dest: destination,
                src: a,
            });
        } else {
            // dest is different from both a and b
            // mov dest, a; add dest, b
            self.mov_reg(destination, a);
            self.instructions.push(X86Asm::AddRR {
                dest: destination,
                src: b,
            });
        }
    }

    pub fn sub(&mut self, destination: X86Register, a: X86Register, b: X86Register) {
        // x86 SUB is destructive: dest = dest - src
        // Subtraction is NOT commutative, so we need careful handling
        if destination == a {
            // Simple case: dest = a - b -> SUB dest, b
            self.instructions.push(X86Asm::SubRR {
                dest: destination,
                src: b,
            });
        } else if destination == b {
            // dest = a - b where dest == b
            // We can't swap operands (not commutative)
            // Use: NEG dest; ADD dest, a (since -b + a = a - b)
            self.instructions.push(X86Asm::Neg { reg: destination });
            self.instructions.push(X86Asm::AddRR {
                dest: destination,
                src: a,
            });
        } else {
            // dest is different from both a and b
            // mov dest, a; sub dest, b
            self.mov_reg(destination, a);
            self.instructions.push(X86Asm::SubRR {
                dest: destination,
                src: b,
            });
        }
    }

    pub fn sub_imm(&mut self, destination: X86Register, a: X86Register, imm: i32) {
        if destination != a {
            self.mov_reg(destination, a);
        }
        self.instructions.push(X86Asm::SubRI {
            dest: destination,
            imm,
        });
    }

    pub fn mul(&mut self, destination: X86Register, a: X86Register, b: X86Register) {
        // IMUL r64, r/m64 is: dest = dest * src
        // We want dest = a * b
        // Multiplication is commutative, so we can swap operands if needed
        if destination == a {
            // Simple case: dest = a * b -> IMUL dest, b
            self.instructions.push(X86Asm::ImulRR {
                dest: destination,
                src: b,
            });
        } else if destination == b {
            // dest = a * b where dest == b
            // Since mul is commutative: dest = b * a -> IMUL dest, a
            self.instructions.push(X86Asm::ImulRR {
                dest: destination,
                src: a,
            });
        } else {
            // dest is different from both a and b
            // mov dest, a; imul dest, b
            self.mov_reg(destination, a);
            self.instructions.push(X86Asm::ImulRR {
                dest: destination,
                src: b,
            });
        }
    }

    pub fn div(&mut self, destination: X86Register, a: X86Register, b: X86Register) {
        // IDIV uses RDX:RAX / divisor, quotient in RAX, remainder in RDX
        // Need to handle cases where divisor (b) might be RAX or RDX
        self.instructions.push(X86Asm::Push { reg: RDX });

        // If divisor is RAX, we need to save it before moving dividend to RAX
        // Use R11 as a temporary (caller-saved, safe to clobber)
        let actual_divisor = if b == RAX {
            self.mov_reg(R11, b); // Save divisor before clobbering RAX
            R11
        } else if b == RDX {
            // Divisor is RDX, but we just pushed RDX and will clobber it with CQO
            self.mov_reg(R11, b); // Save divisor
            R11
        } else {
            b
        };

        self.mov_reg(RAX, a);
        self.instructions.push(X86Asm::Cqo); // Sign-extend RAX to RDX:RAX
        self.instructions.push(X86Asm::Idiv {
            divisor: actual_divisor,
        });
        self.mov_reg(destination, RAX);
        self.instructions.push(X86Asm::Pop { reg: RDX });
    }

    /// Modulo operation: returns remainder of a / b
    /// On x86-64, IDIV puts quotient in RAX and remainder in RDX
    pub fn modulo(&mut self, destination: X86Register, a: X86Register, b: X86Register) {
        // IDIV uses RDX:RAX / divisor, quotient in RAX, remainder in RDX
        self.instructions.push(X86Asm::Push { reg: RDX });

        // If divisor is RAX, we need to save it before moving dividend to RAX
        let actual_divisor = if b == RAX {
            self.mov_reg(R11, b); // Save divisor before clobbering RAX
            R11
        } else if b == RDX {
            // Divisor is RDX, but we just pushed RDX and will clobber it with CQO
            self.mov_reg(R11, b); // Save divisor
            R11
        } else {
            b
        };

        self.mov_reg(RAX, a);
        self.instructions.push(X86Asm::Cqo); // Sign-extend RAX to RDX:RAX
        self.instructions.push(X86Asm::Idiv {
            divisor: actual_divisor,
        });
        // Remainder is in RDX - but we need to pop first since we pushed RDX
        if destination == RDX {
            // Destination is RDX - get remainder from stack-top, discard old value
            self.instructions.push(X86Asm::Pop { reg: R11 }); // Discard saved RDX
        // RDX already has remainder
        } else {
            self.mov_reg(destination, RDX); // Get remainder
            self.instructions.push(X86Asm::Pop { reg: RDX }); // Restore RDX
        }
    }

    // === Shift operations ===

    pub fn shift_right_imm(&mut self, destination: X86Register, a: X86Register, imm: i32) {
        if destination != a {
            self.mov_reg(destination, a);
        }
        self.instructions.push(X86Asm::SarRI {
            dest: destination,
            imm: imm as u8,
        });
    }

    pub fn shift_left_imm(&mut self, destination: X86Register, a: X86Register, imm: i32) {
        if destination != a {
            self.mov_reg(destination, a);
        }
        self.instructions.push(X86Asm::ShlRI {
            dest: destination,
            imm: imm as u8,
        });
    }

    pub fn shift_left(&mut self, dest: X86Register, a: X86Register, b: X86Register) {
        // SHL uses CL for variable shift
        if dest != a {
            self.mov_reg(dest, a);
        }
        self.mov_reg(RCX, b);
        self.instructions.push(X86Asm::ShlRCL { dest });
    }

    pub fn shift_right(&mut self, dest: X86Register, a: X86Register, b: X86Register) {
        // SAR (arithmetic shift right) uses CL
        if dest != a {
            self.mov_reg(dest, a);
        }
        self.mov_reg(RCX, b);
        self.instructions.push(X86Asm::SarRCL { dest });
    }

    pub fn shift_right_zero(&mut self, dest: X86Register, a: X86Register, b: X86Register) {
        // SHR (logical shift right) uses CL
        if dest != a {
            self.mov_reg(dest, a);
        }
        self.mov_reg(RCX, b);
        self.instructions.push(X86Asm::ShrRCL { dest });
    }

    // === Bitwise operations ===

    pub fn and(&mut self, destination: X86Register, a: X86Register, b: X86Register) {
        if destination != a {
            self.mov_reg(destination, a);
        }
        self.instructions.push(X86Asm::AndRR {
            dest: destination,
            src: b,
        });
    }

    pub fn and_imm(&mut self, destination: X86Register, a: X86Register, imm: u64) {
        if destination != a {
            self.mov_reg(destination, a);
        }
        // x86-64 AND with 64-bit immediate needs special handling
        if imm <= i32::MAX as u64 {
            self.instructions.push(X86Asm::AndRI {
                dest: destination,
                imm: imm as i32,
            });
        } else {
            // Need to load 64-bit immediate to temp register first
            let temp = self.temporary_register();
            self.mov_64(temp, imm as isize);
            self.instructions.push(X86Asm::AndRR {
                dest: destination,
                src: temp,
            });
        }
    }

    pub fn or(&mut self, destination: X86Register, a: X86Register, b: X86Register) {
        if destination != a {
            self.mov_reg(destination, a);
        }
        self.instructions.push(X86Asm::OrRR {
            dest: destination,
            src: b,
        });
    }

    pub fn xor(&mut self, dest: X86Register, a: X86Register, b: X86Register) {
        if dest != a {
            self.mov_reg(dest, a);
        }
        self.instructions.push(X86Asm::XorRR { dest, src: b });
    }

    // === Control flow ===

    pub fn ret(&mut self) {
        self.instructions.push(X86Asm::Ret);
    }

    pub fn compare(&mut self, a: X86Register, b: X86Register) {
        self.instructions.push(X86Asm::CmpRR { a, b });
    }

    pub fn compare_bool(
        &mut self,
        condition: Condition,
        dest: X86Register,
        a: X86Register,
        b: X86Register,
    ) {
        // Zero the destination first (BEFORE compare, since XOR modifies flags!)
        self.instructions.push(X86Asm::XorRR { dest, src: dest });
        // Now do the comparison (sets flags)
        self.compare(a, b);
        // Set byte based on condition flags
        let cond = match condition {
            Condition::Equal => X86Cond::E,
            Condition::NotEqual => X86Cond::NE,
            Condition::LessThan => X86Cond::L,
            Condition::LessThanOrEqual => X86Cond::LE,
            Condition::GreaterThan => X86Cond::G,
            Condition::GreaterThanOrEqual => X86Cond::GE,
        };
        self.instructions.push(X86Asm::Setcc { dest, cond });
    }

    pub fn compare_float_bool(
        &mut self,
        condition: Condition,
        dest: X86Register,
        a: X86Register,
        b: X86Register,
    ) {
        // Zero the destination first (BEFORE compare, since XOR modifies flags!)
        self.instructions.push(X86Asm::XorRR { dest, src: dest });
        // Now do the float comparison (sets flags)
        self.instructions.push(X86Asm::Ucomisd { a, b });
        // Set byte based on condition flags
        // Note: For UCOMISD, we need unsigned comparison flags (A/B/AE/BE)
        // because it sets CF and ZF (unsigned-style), not SF/OF (signed-style)
        let cond = match condition {
            Condition::Equal => X86Cond::E,
            Condition::NotEqual => X86Cond::NE,
            Condition::LessThan => X86Cond::B, // Below (unsigned)
            Condition::LessThanOrEqual => X86Cond::BE, // Below or Equal
            Condition::GreaterThan => X86Cond::A, // Above (unsigned)
            Condition::GreaterThanOrEqual => X86Cond::AE, // Above or Equal
        };
        self.instructions.push(X86Asm::Setcc { dest, cond });
    }

    pub fn jump(&mut self, destination: Label) {
        self.instructions.push(X86Asm::Jmp {
            label_index: destination.index,
        });
    }

    pub fn jump_equal(&mut self, destination: Label) {
        self.instructions.push(X86Asm::Jcc {
            label_index: destination.index,
            cond: X86Cond::E,
        });
    }

    pub fn jump_not_equal(&mut self, destination: Label) {
        self.instructions.push(X86Asm::Jcc {
            label_index: destination.index,
            cond: X86Cond::NE,
        });
    }

    pub fn jump_greater(&mut self, destination: Label) {
        self.instructions.push(X86Asm::Jcc {
            label_index: destination.index,
            cond: X86Cond::G,
        });
    }

    pub fn jump_greater_or_equal(&mut self, destination: Label) {
        self.instructions.push(X86Asm::Jcc {
            label_index: destination.index,
            cond: X86Cond::GE,
        });
    }

    pub fn jump_less(&mut self, destination: Label) {
        self.instructions.push(X86Asm::Jcc {
            label_index: destination.index,
            cond: X86Cond::L,
        });
    }

    pub fn jump_less_or_equal(&mut self, destination: Label) {
        self.instructions.push(X86Asm::Jcc {
            label_index: destination.index,
            cond: X86Cond::LE,
        });
    }

    pub fn call(&mut self, register: X86Register) {
        self.instructions.push(X86Asm::CallR { target: register });
        // Record safepoint after the call - this is the return address
        self.record_gc_safepoint();
    }

    pub fn call_builtin(&mut self, register: X86Register) {
        // Just use regular call - it records the safepoint
        self.call(register);
    }

    pub fn recurse(&mut self, label: Label) {
        self.instructions.push(X86Asm::CallRel {
            label_index: label.index,
        });
        // Record safepoint after the call - this is the return address
        self.record_gc_safepoint();
    }

    // === Memory operations ===

    pub fn load_from_heap(&mut self, dest: X86Register, src: X86Register, offset: i32) {
        // Offset is a field index, multiply by 8 for byte offset (64-bit values)
        self.instructions.push(X86Asm::MovRM {
            dest,
            base: src,
            offset: offset * 8,
        });
    }

    pub fn store_on_heap(&mut self, ptr: X86Register, val: X86Register, offset: i32) {
        // Offset is a field index, multiply by 8 for byte offset (64-bit values)
        self.instructions.push(X86Asm::MovMR {
            base: ptr,
            offset: offset * 8,
            src: val,
        });
    }

    pub fn load_from_heap_with_reg_offset(
        &mut self,
        dest: X86Register,
        src: X86Register,
        offset: X86Register,
    ) {
        // Use SIB addressing: MOV dest, [src + offset*1]
        // Single instruction instead of LEA + ADD + MOV
        self.instructions.push(X86Asm::MovRMIndexed {
            dest,
            base: src,
            index: offset,
        });
    }

    pub fn store_to_heap_with_reg_offset(
        &mut self,
        ptr: X86Register,
        val: X86Register,
        offset: X86Register,
    ) {
        // Use SIB addressing: MOV [ptr + offset*1], val
        // Single instruction instead of LEA + ADD + MOV
        self.instructions.push(X86Asm::MovMRIndexed {
            base: ptr,
            index: offset,
            src: val,
        });
    }

    // === Stack operations ===

    const CALLEE_SAVED_SIZE: i32 = 0;

    pub fn push_to_stack(&mut self, reg: X86Register) {
        self.increment_stack_size(1);
        let offset = Self::CALLEE_SAVED_SIZE + (self.max_locals + self.stack_size) * 8;
        self.instructions.push(X86Asm::MovMR {
            base: RBP,
            offset: -offset,
            src: reg,
        });
    }

    pub fn pop_from_stack(&mut self, reg: X86Register) {
        let offset = Self::CALLEE_SAVED_SIZE + (self.max_locals + self.stack_size) * 8;
        self.instructions.push(X86Asm::MovRM {
            dest: reg,
            base: RBP,
            offset: -offset,
        });
        self.stack_size -= 1;
    }

    pub fn load_local(&mut self, dest: X86Register, offset: i32) {
        self.instructions.push(X86Asm::MovRM {
            dest,
            base: RBP,
            offset: -Self::CALLEE_SAVED_SIZE - (offset + 1) * 8,
        });
    }

    pub fn store_local(&mut self, src: X86Register, offset: i32) {
        self.instructions.push(X86Asm::MovMR {
            base: RBP,
            offset: -Self::CALLEE_SAVED_SIZE - (offset + 1) * 8,
            src,
        });
    }

    pub fn push_to_stack_indexed(&mut self, reg: X86Register, offset: i32) {
        // Store to [RSP + offset * 8]
        self.instructions.push(X86Asm::MovMR {
            base: RSP,
            offset: offset * 8,
            src: reg,
        });
    }

    pub fn pop_from_stack_indexed(&mut self, reg: X86Register, offset: i32) {
        // Load from [RSP + offset * 8] and adjust stack
        self.instructions.push(X86Asm::MovRM {
            dest: reg,
            base: RSP,
            offset: offset * 8,
        });
        self.stack_size -= 1;
    }

    pub fn pop_from_stack_indexed_raw(&mut self, reg: X86Register, offset: i32) {
        // Load from [RSP + offset * 8] without stack size adjustment
        self.instructions.push(X86Asm::MovRM {
            dest: reg,
            base: RSP,
            offset: offset * 8,
        });
    }

    pub fn push_to_end_of_stack(&mut self, reg: X86Register, offset: i32) {
        // Ensure frame has space for outgoing stack arguments.
        // Like ARM64, we track this in max_stack_size so the prologue
        // allocates enough space below RSP.
        self.max_stack_size += 1;

        // Store to stack at offset from current position
        self.instructions.push(X86Asm::MovMR {
            base: RSP,
            offset: offset * 8,
            src: reg,
        });
    }

    pub fn load_from_stack_beginning(&mut self, dest: X86Register, offset: i32) {
        // Load from [RBP + offset * 8]
        // On x86-64, stack arguments are above the saved frame pointer and return address.
        // After prologue:
        //   [RBP + 0] = saved RBP
        //   [RBP + 8] = return address
        //   [RBP + 16] = arg 6 (first stack arg)
        //   [RBP + 24] = arg 7
        //   etc.
        // The offset from IR is (arg_index - 6) + 2, so for arg 6: offset = 2 -> [RBP + 16]
        self.instructions.push(X86Asm::MovRM {
            dest,
            base: RBP,
            offset: offset * 8,
        });
    }

    pub fn get_stack_pointer(&mut self, dest: X86Register, offset: X86Register) {
        self.instructions.push(X86Asm::Lea {
            dest,
            base: RSP,
            offset: 0,
        });
        self.add(dest, dest, offset);
    }

    pub fn get_stack_pointer_imm(&mut self, dest: X86Register, offset: isize) {
        self.instructions.push(X86Asm::Lea {
            dest,
            base: RSP,
            offset: offset as i32,
        });
    }

    pub fn get_current_stack_position(&mut self, dest: X86Register) {
        let offset = Self::CALLEE_SAVED_SIZE + (self.max_locals + self.stack_size + 1) * 8;
        self.instructions.push(X86Asm::Lea {
            dest,
            base: RBP,
            offset: -offset,
        });
    }

    pub fn add_stack_pointer(&mut self, bytes: i32) {
        self.instructions.push(X86Asm::AddRI {
            dest: RSP,
            imm: bytes,
        });
    }

    pub fn sub_stack_pointer(&mut self, bytes: i32) {
        self.instructions.push(X86Asm::SubRI {
            dest: RSP,
            imm: bytes,
        });
    }

    // === Register allocation ===

    pub fn volatile_register(&mut self) -> X86Register {
        let reg = self
            .free_volatile_registers
            .pop()
            .expect("No free volatile registers");
        // Track that this callee-saved register is used (for ABI compliance)
        self.mark_callee_saved_used(reg);
        reg
    }

    /// Mark a callee-saved register as used by this function.
    fn mark_callee_saved_used(&mut self, reg: X86Register) {
        // R12-R15 and RBX are callee-saved in System V AMD64 ABI
        // R12=12, R13=13, R14=14, R15=15, RBX has index 3
        let bit = match reg.index {
            12 => Some(0), // R12
            13 => Some(1), // R13
            14 => Some(2), // R14
            15 => Some(3), // R15
            3 => Some(4),  // RBX (index 3)
            _ => None,
        };
        if let Some(b) = bit {
            self.used_callee_saved_registers |= 1 << b;
        }
    }

    /// Reset callee-saved register tracking for a new function.
    pub fn reset_callee_saved_tracking(&mut self) {
        self.used_callee_saved_registers = 0;
    }

    /// Mark a callee-saved register as used by its index.
    pub fn mark_callee_saved_register_used(&mut self, index: usize) {
        // R12-R15 use indices 12-15, RBX uses virtual index 16
        let bit = match index {
            12 => Some(0),
            13 => Some(1),
            14 => Some(2),
            15 => Some(3),
            16 => Some(4), // RBX
            _ => None,
        };
        if let Some(b) = bit {
            self.used_callee_saved_registers |= 1 << b;
        }
    }

    /// Get the list of callee-saved registers that are actually used.
    /// Returns registers in order: R12, R13, R14, R15, RBX (based on bitmask).
    pub fn get_used_callee_saved_registers(&self) -> Vec<X86Register> {
        let mut result = Vec::new();
        // Bit 0 = R12 (index 12), Bit 1 = R13 (index 13), etc.
        let registers = [R12, R13, R14, R15, RBX];
        for (i, reg) in registers.iter().enumerate() {
            if self.used_callee_saved_registers & (1 << i) != 0 {
                result.push(*reg);
            }
        }
        result
    }

    pub fn temporary_register(&mut self) -> X86Register {
        if let Some(reg) = self.free_temporary_registers.pop() {
            self.allocated_temporary_registers.push(reg);
            reg
        } else {
            panic!("No free temporary registers")
        }
    }

    pub fn free_temporary_register(&mut self, register: X86Register) {
        if self.canonical_temporary_registers.contains(&register)
            && !self.free_temporary_registers.contains(&register)
        {
            self.free_temporary_registers.push(register);
            self.allocated_temporary_registers
                .retain(|r| *r != register);
        }
    }

    pub fn free_register(&mut self, register: X86Register) {
        if self.canonical_volatile_registers.contains(&register)
            && !self.free_volatile_registers.contains(&register)
        {
            self.free_volatile_registers.push(register);
            self.allocated_volatile_registers.retain(|r| *r != register);
        }
    }

    pub fn reserve_register(&mut self, register: X86Register) {
        self.free_volatile_registers.retain(|r| *r != register);
        if !self.allocated_volatile_registers.contains(&register) {
            self.allocated_volatile_registers.push(register);
        }
    }

    pub fn clear_temporary_registers(&mut self) {
        self.free_temporary_registers = self.canonical_temporary_registers.clone();
        self.allocated_temporary_registers.clear();
    }

    // === Label management ===

    pub fn new_label(&mut self, name: &str) -> Label {
        let index = self.get_label_index();
        self.labels.push(name.to_string());
        Label { index }
    }

    pub fn write_label(&mut self, label: Label) {
        self.label_locations
            .insert(label.index, self.instructions.len());
        self.instructions.push(X86Asm::Label { index: label.index });
    }

    pub fn load_label_address(&mut self, dest: X86Register, label: Label) {
        // LEA with RIP-relative addressing
        // Uses LeaRipRel which will be patched to correct offset
        // Store label index as i32 for now (will be replaced with displacement during patching)
        self.instructions.push(X86Asm::LeaRipRel {
            dest,
            label_index: label.index as i32,
        });
    }

    pub fn get_label_by_name(&self, name: &str) -> Label {
        let index = self
            .labels
            .iter()
            .position(|n| n == name)
            .expect("Label not found");
        Label { index }
    }

    pub fn register_label_name(&mut self, name: &str) {
        if !self.labels.contains(&name.to_string()) {
            self.labels.push(name.to_string());
        }
    }

    // === Floating point operations ===

    pub fn fadd(&mut self, dest: X86Register, a: X86Register, b: X86Register) {
        if dest != a {
            self.instructions.push(X86Asm::MovsdRR { dest, src: a });
        }
        self.instructions.push(X86Asm::Addsd { dest, src: b });
    }

    pub fn fsub(&mut self, dest: X86Register, a: X86Register, b: X86Register) {
        if dest != a {
            self.instructions.push(X86Asm::MovsdRR { dest, src: a });
        }
        self.instructions.push(X86Asm::Subsd { dest, src: b });
    }

    pub fn fmul(&mut self, dest: X86Register, a: X86Register, b: X86Register) {
        if dest != a {
            self.instructions.push(X86Asm::MovsdRR { dest, src: a });
        }
        self.instructions.push(X86Asm::Mulsd { dest, src: b });
    }

    pub fn fdiv(&mut self, dest: X86Register, a: X86Register, b: X86Register) {
        if dest != a {
            self.instructions.push(X86Asm::MovsdRR { dest, src: a });
        }
        self.instructions.push(X86Asm::Divsd { dest, src: b });
    }

    pub fn fmov_to_float(&mut self, dest: X86Register, src: X86Register) {
        self.instructions.push(X86Asm::MovqXR { dest, src });
    }

    pub fn fmov_from_float(&mut self, dest: X86Register, src: X86Register) {
        self.instructions.push(X86Asm::MovqRX { dest, src });
    }

    // === Tagged value operations ===

    pub fn tag_value(&mut self, dest: X86Register, value: X86Register, tag: X86Register) {
        // Tag = (value << tag_size) | tag
        if dest != value {
            self.mov_reg(dest, value);
        }
        self.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
        self.or(dest, dest, tag);
    }

    pub fn get_tag(&mut self, dest: X86Register, value: X86Register) {
        // Tag is in the low 3 bits
        if dest != value {
            self.mov_reg(dest, value);
        }
        self.and_imm(dest, dest, 0b111); // Tag mask = 0b111
    }

    pub fn guard_integer(&mut self, temp: X86Register, value: X86Register, error_label: Label) {
        self.get_tag(temp, value);
        self.instructions.push(X86Asm::CmpRI {
            reg: temp,
            imm: BuiltInTypes::Int.get_tag() as i32,
        });
        self.jump_not_equal(error_label);
    }

    pub fn guard_float(&mut self, temp: X86Register, value: X86Register, error_label: Label) {
        self.get_tag(temp, value);
        self.instructions.push(X86Asm::CmpRI {
            reg: temp,
            imm: BuiltInTypes::Float.get_tag() as i32,
        });
        self.jump_not_equal(error_label);
    }

    // === Atomic operations ===

    pub fn atomic_load(&mut self, dest: X86Register, src: X86Register) {
        // On x86-64, aligned loads are atomic and have acquire semantics by default
        // due to x86's strong memory model (loads are not reordered with other loads).
        // No MFENCE needed for acquire semantics - that would be extremely expensive.
        self.load_from_heap(dest, src, 0);
    }

    pub fn atomic_store(&mut self, ptr: X86Register, val: X86Register) {
        // On x86-64, aligned stores are atomic and have release semantics by default
        // due to x86's strong memory model (stores are not reordered with other stores).
        // No MFENCE needed for release semantics - that would be extremely expensive.
        self.store_on_heap(ptr, val, 0);
    }

    pub fn compare_and_swap(&mut self, expected: X86Register, new: X86Register, ptr: X86Register) {
        // LOCK CMPXCHG uses RAX for expected value
        // IMPORTANT: We must handle the case where new or ptr might be RAX,
        // since moving expected to RAX would clobber them.
        //
        // Strategy: Save any operand that's in RAX to R11 before moving expected to RAX.

        let (actual_new, actual_ptr) = if expected == RAX {
            // expected is already in RAX, no clobbering possible
            (new, ptr)
        } else if new == RAX && ptr == RAX {
            // Both new and ptr are RAX (unlikely but handle it)
            self.mov_reg(R11, RAX); // Save RAX to R11
            self.mov_reg(RAX, expected);
            (R11, R11)
        } else if new == RAX {
            // new is in RAX, save it to R11 before overwriting RAX
            self.mov_reg(R11, RAX);
            self.mov_reg(RAX, expected);
            (R11, ptr)
        } else if ptr == RAX {
            // ptr is in RAX, save it to R11 before overwriting RAX
            self.mov_reg(R11, RAX);
            self.mov_reg(RAX, expected);
            (new, R11)
        } else {
            // No conflicts, just move expected to RAX
            self.mov_reg(RAX, expected);
            (new, ptr)
        };

        self.instructions.push(X86Asm::LockCmpxchg {
            base: actual_ptr,
            src: actual_new,
        });
        // Result is in RAX
        self.mov_reg(expected, RAX);
    }

    // === Utility methods ===

    pub fn current_position(&self) -> usize {
        self.instructions.len()
    }

    pub fn set_max_locals(&mut self, max_locals: usize) {
        self.max_locals = max_locals as i32;
    }

    pub fn set_all_locals_to_null(&mut self, null_register: X86Register) {
        for i in 0..self.max_locals {
            self.store_local(null_register, i);
        }
    }

    pub fn arg(&self, index: u8) -> X86Register {
        // System V AMD64 ABI argument registers
        match index {
            0 => RDI,
            1 => RSI,
            2 => RDX,
            3 => RCX,
            4 => R8,
            5 => R9,
            _ => panic!("Too many arguments, stack passing not yet implemented"),
        }
    }

    pub fn ret_reg(&self) -> X86Register {
        RAX
    }

    pub fn register_from_index(&self, index: usize) -> X86Register {
        // Map virtual register indices to physical registers.
        //
        // The register allocator assigns:
        // - Indices 0-5 for function arguments (these map to arg registers)
        // - Indices 6-11 for scratch/callee-saved registers
        // - Indices 12-15 for callee-saved registers (R12-R15)
        //
        // On x86-64, argument registers are NOT sequential:
        // - arg(0) = RDI (raw index 7)
        // - arg(1) = RSI (raw index 6)
        // - arg(2) = RDX (raw index 2)
        // - arg(3) = RCX (raw index 1)
        // - arg(4) = R8  (raw index 8)
        // - arg(5) = R9  (raw index 9)
        //
        // IMPORTANT: We must ensure virtual indices 6-11 do NOT map to the same
        // physical registers as indices 0-5, otherwise we get register collisions.
        // Virtual 6-9 use scratch registers (R10, R11) and callee-saved (RBX, R12).
        // Virtual 10-11 continue with R13, R14.
        match index {
            0 => RDI,  // arg 0
            1 => RSI,  // arg 1
            2 => RDX,  // arg 2
            3 => RCX,  // arg 3
            4 => R8,   // arg 4
            5 => R9,   // arg 5
            6 => R10,  // scratch register
            7 => R11,  // scratch register
            8 => RBX,  // callee-saved (careful: also used as volatile register)
            9 => R12,  // callee-saved
            10 => R13, // callee-saved
            11 => R14, // callee-saved
            16 => RBX, // Additional callee-saved register (virtual index 16 -> RBX)
            // For indices 12-15, use the raw register index (maps to R12-R15)
            _ => X86Register::from_index(index),
        }
    }

    /// Compile instructions to bytes
    pub fn compile_to_bytes(&mut self) -> Vec<u8> {
        // First, patch prelude/epilogue (may insert instructions, shifting labels)
        self.patch_prelude_and_epilogue();
        // Then patch labels (after instruction insertions are done)
        self.patch_labels();

        // Debug: dump instructions before encoding
        if std::env::var("DUMP_X86").is_ok() {
            self.dump_instructions_named(self.current_function_name.as_deref());
        }

        // Finally, encode all instructions
        let mut bytes = Vec::new();
        for instr in &self.instructions {
            bytes.extend(instr.encode());
        }
        bytes
    }

    /// Dump all instructions for debugging, with optional function name
    pub fn dump_instructions_named(&self, function_name: Option<&str>) {
        // Check if we should filter by function name
        if let Ok(filter) = std::env::var("DUMP_X86_FILTER") {
            if let Some(name) = function_name {
                if !name.contains(&filter) {
                    return;
                }
            } else {
                return;
            }
        }

        let name_str = function_name.unwrap_or("<anonymous>");
        eprintln!(
            "\n=== X86 Instructions for {} ({} total) ===",
            name_str,
            self.instructions.len()
        );
        let mut byte_offset = 0;
        for (i, instr) in self.instructions.iter().enumerate() {
            let size = instr.size();
            let bytes = instr.encode();
            let hex: String = bytes.iter().map(|b| format!("{:02x} ", b)).collect();
            eprintln!(
                "{:4} [{:04x}] {:20} {:?}",
                i,
                byte_offset,
                hex.trim(),
                instr
            );
            byte_offset += size;
        }
        eprintln!("=== Total: {} bytes ===\n", byte_offset);
    }

    fn patch_labels(&mut self) {
        // Calculate byte offsets for each label
        let mut byte_offsets: HashMap<usize, usize> = HashMap::new();
        let mut current_offset = 0;

        for instr in self.instructions.iter() {
            if let X86Asm::Label { index } = instr {
                byte_offsets.insert(*index, current_offset);
            }
            current_offset += instr.size();
        }

        // Patch jump/call instructions
        current_offset = 0;
        for instr in self.instructions.iter_mut() {
            let instr_size = instr.size();
            match instr {
                X86Asm::Jmp { label_index } => {
                    if let Some(&target) = byte_offsets.get(label_index) {
                        // rel32 is relative to end of instruction
                        let rel = (target as i64) - (current_offset as i64 + instr_size as i64);
                        *instr = X86Asm::Jmp {
                            label_index: rel as usize,
                        };
                    }
                }
                X86Asm::Jcc { label_index, cond } => {
                    if let Some(&target) = byte_offsets.get(label_index) {
                        let rel = (target as i64) - (current_offset as i64 + instr_size as i64);
                        *instr = X86Asm::Jcc {
                            label_index: rel as usize,
                            cond: *cond,
                        };
                    }
                }
                X86Asm::CallRel { label_index } => {
                    if let Some(&target) = byte_offsets.get(label_index) {
                        let rel = (target as i64) - (current_offset as i64 + instr_size as i64);
                        *instr = X86Asm::CallRel {
                            label_index: rel as usize,
                        };
                    }
                }
                X86Asm::LeaRipRel { dest, label_index } => {
                    // label_index starts as a label index (cast to i32), lookup using it as usize
                    if let Some(&target) = byte_offsets.get(&(*label_index as usize)) {
                        // RIP-relative offset is from end of instruction
                        let rel = (target as i64) - (current_offset as i64 + instr_size as i64);
                        // Store the signed displacement as i32
                        *instr = X86Asm::LeaRipRel {
                            dest: *dest,
                            label_index: rel as i32,
                        };
                    }
                }
                _ => {}
            }
            current_offset += instr_size;
        }
    }

    fn patch_prelude_and_epilogue(&mut self) {
        // Get callee-saved registers that need to be saved
        let used_callee_saved = self.get_used_callee_saved_registers();
        let num_callee_saved = used_callee_saved.len();

        // Calculate stack size including space for callee-saved registers
        let mut slots = self.max_locals + self.max_stack_size + num_callee_saved as i32;
        if slots % 2 != 0 {
            slots += 1;
        }
        let aligned_size = slots * 8;

        // Find and replace the SUB instruction, inserting MOV instructions after it
        let sub_index = self.instructions.iter().position(|instr| {
            matches!(instr, X86Asm::SubRI { dest, imm } if *dest == RSP && *imm == 0x1111_1111_u32 as i32)
        });

        if let Some(index) = sub_index {
            // Replace the SUB with correct size
            self.instructions[index] = X86Asm::SubRI {
                dest: RSP,
                imm: aligned_size,
            };

            // Calculate byte offset at insertion point (after SUB instruction)
            let byte_offset_at_insert: usize =
                self.instructions[..=index].iter().map(|i| i.size()).sum();

            // Insert MOV instructions for callee-saved registers right after SUB
            // We store at increasing offsets from RSP
            let mut inserted_instructions = Vec::new();
            for (i, reg) in used_callee_saved.iter().enumerate() {
                // Store at [RSP + i*8]
                let instr = X86Asm::MovMR {
                    base: RSP,
                    offset: (i * 8) as i32,
                    src: *reg,
                };
                inserted_instructions.push(instr);
            }

            // CRITICAL FIX: Zero out local slots to prevent GC from seeing garbage.
            // When GC runs during allocation (before a local is assigned), uninitialized
            // local slots could contain interior pointers or other garbage from previous
            // stack usage. Initialize all local slots to null (0x7, which is tagged null).
            if self.max_locals > 0 {
                let null_value = BuiltInTypes::null_value() as i32;

                // Load null value into R11 (caller-saved, safe to use)
                inserted_instructions.push(X86Asm::MovRI32 {
                    dest: R11,
                    imm: null_value,
                });

                // Store null to each local slot at [RBP - (i+1)*8]
                for i in 0..self.max_locals {
                    inserted_instructions.push(X86Asm::MovMR {
                        base: RBP,
                        offset: -((i + 1) * 8),
                        src: R11,
                    });
                }
            }

            // Calculate total byte size of inserted instructions
            let num_inserted = inserted_instructions.len();
            let byte_delta: usize = inserted_instructions.iter().map(|i| i.size()).sum();

            // Insert the instructions
            for (i, instr) in inserted_instructions.into_iter().enumerate() {
                self.instructions.insert(index + 1 + i, instr);
            }

            // Shift label locations that come after the insertion point
            if num_inserted > 0 {
                for location in self.label_locations.values_mut() {
                    if *location > index {
                        *location += num_inserted;
                    }
                }

                // Shift stack_map byte offsets that come after the insertion point
                self.stack_map = self
                    .stack_map
                    .drain()
                    .map(|(k, v)| {
                        if k >= byte_offset_at_insert {
                            (k + byte_delta, v)
                        } else {
                            (k, v)
                        }
                    })
                    .collect();
            }
        }

        // Find and replace the ADD instruction, inserting load instructions before it
        // Note: After inserting saves above, indices have shifted
        let add_index = self.instructions.iter().rposition(|instr| {
            matches!(instr, X86Asm::AddRI { dest, imm } if *dest == RSP && *imm == 0x1111_1111_u32 as i32)
        });

        if let Some(index) = add_index {
            // Calculate byte offset at insertion point (before ADD instruction)
            let byte_offset_at_insert: usize =
                self.instructions[..index].iter().map(|i| i.size()).sum();

            // Create load instructions to restore callee-saved registers
            let mut inserted_instructions = Vec::new();
            for (i, reg) in used_callee_saved.iter().enumerate() {
                // Load from [RSP + i*8]
                let instr = X86Asm::MovRM {
                    dest: *reg,
                    base: RSP,
                    offset: (i * 8) as i32,
                };
                inserted_instructions.push(instr);
            }

            // Calculate total byte size of inserted instructions
            let byte_delta: usize = inserted_instructions.iter().map(|i| i.size()).sum();

            // Insert the instructions
            for (i, instr) in inserted_instructions.into_iter().enumerate() {
                self.instructions.insert(index + i, instr);
            }

            // Update the ADD instruction (now at index + num_callee_saved)
            let new_add_index = index + num_callee_saved;
            self.instructions[new_add_index] = X86Asm::AddRI {
                dest: RSP,
                imm: aligned_size,
            };

            // Shift label locations that come after the insertion point
            if num_callee_saved > 0 {
                for location in self.label_locations.values_mut() {
                    if *location > index {
                        *location += num_callee_saved;
                    }
                }

                // Shift stack_map byte offsets that come after the insertion point
                self.stack_map = self
                    .stack_map
                    .drain()
                    .map(|(k, v)| {
                        if k >= byte_offset_at_insert {
                            (k + byte_delta, v)
                        } else {
                            (k, v)
                        }
                    })
                    .collect();
            }
        }
    }

    /// Translate stack map for GC
    /// Stack map keys are already byte offsets
    pub fn translate_stack_map(&self, base_pointer: usize) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for (byte_offset, size) in &self.stack_map {
            result.push((base_pointer + byte_offset, *size));
        }
        result
    }

    /// Get the current byte offset in the generated code.
    /// x86-64 instructions are variable length, so we sum all sizes.
    pub fn current_byte_offset(&self) -> usize {
        self.instructions.iter().map(|i| i.size()).sum()
    }

    /// Record a GC safepoint at the current position.
    pub fn record_gc_safepoint(&mut self) {
        let byte_offset = self.current_byte_offset();
        let stack_size = self.stack_size as usize;
        self.stack_map.insert(byte_offset, stack_size);
    }

    /// Get the adjustment for return address lookup.
    /// x86-64 CALL reg instruction is typically 2-3 bytes.
    pub fn return_address_adjustment() -> usize {
        // CALL reg (FF /2) with REX prefix is 3 bytes
        // Without REX prefix it's 2 bytes
        // We use 3 to be safe since most of our calls use 64-bit registers
        3
    }

    /// Store a pair of registers (emulated for x86)
    pub fn store_pair(
        &mut self,
        reg1: X86Register,
        reg2: X86Register,
        dest: X86Register,
        offset: i32,
    ) {
        // x86 doesn't have STP, use two stores
        self.instructions.push(X86Asm::MovMR {
            base: dest,
            offset: offset * 8,
            src: reg1,
        });
        self.instructions.push(X86Asm::MovMR {
            base: dest,
            offset: (offset + 1) * 8,
            src: reg2,
        });
    }

    /// Load a pair of registers (emulated for x86)
    pub fn load_pair(
        &mut self,
        reg1: X86Register,
        reg2: X86Register,
        location: X86Register,
        offset: i32,
    ) {
        // x86 doesn't have LDP, use two loads
        self.instructions.push(X86Asm::MovRM {
            dest: reg1,
            base: location,
            offset: offset * 8,
        });
        self.instructions.push(X86Asm::MovRM {
            dest: reg2,
            base: location,
            offset: (offset + 1) * 8,
        });
    }

    /// Share label info for debugging (stub for now)
    pub fn share_label_info_debug(&self, _function_pointer: usize) -> Result<(), CompileError> {
        // TODO: Implement debug info sharing for x86-64
        Ok(())
    }
}
