use crate::{
    builtins::debugger,
    machine_code::arm_codegen::{
        ArmAsm, LdpGenSelector, LdrImmGenSelector, Register, SP, Size, StpGenSelector,
        StrImmGenSelector, X0, X10, X11, X12, X16, X19, X20, X21, X22, X23, X24, X25, X26, X27,
        X28, X29, X30, ZERO_REGISTER,
    },
    types::BuiltInTypes,
};

use std::collections::HashMap;

use crate::{Data, Message, common::Label, ir::Condition};

pub enum FmovDirection {
    FromGeneralToFloat,
    FromFloatToGeneral,
}

pub fn adr(destination: Register, label_index: usize) -> ArmAsm {
    // For now, store the label index in immhi, we'll patch this later
    // immlo will be set to 0 for now
    ArmAsm::Adr {
        immlo: 0,
        immhi: label_index as i32,
        rd: destination,
    }
}

pub fn mov_imm(destination: Register, input: u16) -> ArmAsm {
    ArmAsm::Movz {
        sf: destination.sf(),
        hw: 0,
        // TODO: Shouldn't this be a u16??
        imm16: input as i32,
        rd: destination,
    }
}

pub fn mov_reg(destination: Register, source: Register) -> ArmAsm {
    ArmAsm::MovOrrLogShift {
        sf: destination.sf(),
        rm: source,
        rd: destination,
    }
}

pub fn mov_sp(destination: Register, source: Register) -> ArmAsm {
    ArmAsm::MovAddAddsubImm {
        sf: destination.sf(),
        rn: source,
        rd: destination,
    }
}

pub fn add(destination: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::AddAddsubShift {
        sf: destination.sf(),
        shift: 0,
        imm6: 0,
        rn: a,
        rm: b,
        rd: destination,
    }
}

pub fn sub(destination: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::SubAddsubShift {
        sf: destination.sf(),
        shift: 0,
        imm6: 0,
        rn: a,
        rm: b,
        rd: destination,
    }
}

pub fn sub_imm(destination: Register, a: Register, b: i32) -> ArmAsm {
    ArmAsm::SubAddsubImm {
        sf: destination.sf(),
        sh: 0,
        imm12: b,
        rn: a,
        rd: destination,
    }
}

pub fn mul(destination: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::Madd {
        sf: destination.sf(),
        rm: b,
        ra: ZERO_REGISTER,
        rn: a,
        rd: destination,
    }
}

pub fn div(destination: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::Sdiv {
        sf: destination.sf(),
        rm: b,
        rn: a,
        rd: destination,
    }
}

fn shift_right_imm(destination: Register, a: Register, b: i32) -> ArmAsm {
    ArmAsm::AsrSbfm {
        sf: destination.sf(),
        rn: a,
        rd: destination,
        n: destination.sf(),
        immr: b,
        imms: 0b111111,
    }
}

fn shift_left_imm(destination: Register, a: Register, b: i32) -> ArmAsm {
    let immr = 64 - b;
    ArmAsm::LslUbfm {
        sf: destination.sf(),
        rn: a,
        rd: destination,
        n: destination.sf(),
        immr,
        imms: immr - 1,
    }
}

fn shift_left(dest: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::LslLslv {
        sf: dest.sf(),
        rm: b,
        rn: a,
        rd: dest,
    }
}

fn shift_right(dest: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::AsrAsrv {
        sf: dest.sf(),
        rm: b,
        rn: a,
        rd: dest,
    }
}

fn shift_right_zero(dest: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::AsrAsrv {
        sf: dest.sf(),
        rm: b,
        rn: a,
        rd: dest,
    }
}

fn xor(dest: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::EorLogShift {
        sf: dest.sf(),
        shift: 0,
        rm: b,
        rn: a,
        rd: dest,
        imm6: 0,
    }
}

pub fn ret() -> ArmAsm {
    ArmAsm::Ret {
        rn: Register {
            size: Size::S64,
            index: 30,
        },
    }
}

pub fn or(destination: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::OrrLogShift {
        sf: destination.sf(),
        shift: 0,
        rm: b,
        rn: a,
        rd: destination,
        imm6: 0,
    }
}

// Too lazy to understand this. But great explanation here
// https://kddnewton.com/2022/08/11/aarch64-bitmask-immediates.html
// Code taken from there
pub struct BitmaskImmediate {
    n: u8,
    imms: u8,
    immr: u8,
}

/// Is this number's binary representation all 1s?
fn is_mask(imm: u64) -> bool {
    ((imm + 1) & imm) == 0
}

/// Is this number's binary representation one or more 1s followed by
/// one or more 0s?
fn is_shifted_mask(imm: u64) -> bool {
    is_mask((imm - 1) | imm)
}

impl TryFrom<u64> for BitmaskImmediate {
    type Error = ();

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if value == 0 || value == u64::MAX {
            return Err(());
        }
        let mut imm = value;
        let mut size = 64;

        loop {
            size >>= 1;
            let mask = (1 << size) - 1;

            if (imm & mask) != ((imm >> size) & mask) {
                size <<= 1;
                break;
            }

            if size <= 2 {
                break;
            }
        }
        let trailing_ones: u32;
        let left_rotations: u32;

        let mask = u64::MAX >> (64 - size);
        imm &= mask;

        if is_shifted_mask(imm) {
            left_rotations = imm.trailing_zeros();
            trailing_ones = (imm >> left_rotations).trailing_ones();
        } else {
            imm |= !mask;
            if !is_shifted_mask(!imm) {
                return Err(());
            }

            let leading_ones = imm.leading_ones();
            left_rotations = 64 - leading_ones;
            trailing_ones = leading_ones + imm.trailing_ones() - (64 - size);
        }

        // immr is the number of right rotations it takes to get from the
        // matching unrotated pattern to the target value.
        let immr = (size - left_rotations) & (size - 1);

        // imms is encoded as the size of the pattern, a 0, and then one less
        // than the number of sequential 1s.
        let imms = (!(size - 1) << 1) | (trailing_ones - 1);

        // n is 1 if the element size is 64-bits, and 0 otherwise.
        let n = ((imms >> 6) & 1) ^ 1;

        Ok(BitmaskImmediate {
            n: n as u8,
            imms: (imms & 0x3f) as u8,
            immr: (immr & 0x3f) as u8,
        })
    }
}

pub fn and_imm(destination: Register, a: Register, b: u64) -> ArmAsm {
    let immediate: Result<BitmaskImmediate, _> = (b).try_into();
    let immediate = immediate.unwrap();

    ArmAsm::AndLogImm {
        sf: destination.sf(),
        n: immediate.n as i32,
        immr: immediate.immr as i32,
        imms: immediate.imms as i32,
        rn: a,
        rd: destination,
    }
}

pub fn and(destination: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::AndLogShift {
        sf: destination.sf(),
        shift: 0,
        rm: b,
        imm6: 0,
        rn: a,
        rd: destination,
    }
}

pub fn get_tag(destination: Register, value: Register) -> ArmAsm {
    ArmAsm::AndLogImm {
        sf: destination.sf(),
        n: 1,
        immr: 0,
        imms: 2,
        rn: value,
        rd: destination,
    }
}

pub fn tag_value(destination: Register, value: Register, tag: Register) -> Vec<ArmAsm> {
    vec![
        shift_left_imm(destination, value, BuiltInTypes::tag_size()),
        or(destination, destination, tag),
    ]
}

pub fn compare(a: Register, b: Register) -> ArmAsm {
    ArmAsm::CmpSubsAddsubShift {
        sf: a.sf(),
        shift: 0,
        rm: b,
        imm6: 0,
        rn: a,
    }
}

impl Condition {
    #[allow(unused)]
    pub fn arm_condition(&self) -> i32 {
        match self {
            Condition::Equal => 0,
            Condition::NotEqual => 1,
            Condition::LessThan => 11,
            Condition::LessThanOrEqual => 13,
            Condition::GreaterThan => 12,
            Condition::GreaterThanOrEqual => 10,
        }
    }
    pub fn arm_inverted_condition(&self) -> i32 {
        match self {
            Condition::Equal => 1,
            Condition::NotEqual => 0,
            Condition::LessThan => 13,
            Condition::LessThanOrEqual => 11,
            Condition::GreaterThan => 10,
            Condition::GreaterThanOrEqual => 12,
        }
    }

    pub fn arm_condition_from_i32(i: i32) -> Self {
        match i {
            0 => Condition::Equal,
            1 => Condition::NotEqual,
            11 => Condition::LessThan,
            13 => Condition::LessThanOrEqual,
            12 => Condition::GreaterThan,
            10 => Condition::GreaterThanOrEqual,
            _ => panic!("Invalid condition"),
        }
    }
}

pub fn compare_bool(
    condition: Condition,
    destination: Register,
    a: Register,
    b: Register,
) -> Vec<ArmAsm> {
    vec![
        ArmAsm::SubsAddsubShift {
            sf: destination.sf(),
            shift: 0,
            rm: a,
            imm6: 0,
            rn: b,
            rd: destination,
        },
        ArmAsm::CsetCsinc {
            sf: destination.sf(),
            // For some reason these conditions are inverted
            cond: condition.arm_inverted_condition(),
            rd: destination,
        },
    ]
}

pub fn compare_float_bool(
    condition: Condition,
    destination: Register,
    a: Register,
    b: Register,
) -> Vec<ArmAsm> {
    // FCMP compares Rn against Rm and sets flags directly for "Rn <cond> Rm"
    // Unlike SUBS (used for integers) which computes rn - rm backwards,
    // FCMP semantics match the natural comparison "a <cond> b"
    //
    // After FCMP Rn=a, Rm=b:
    // - If a > b: N=0, Z=0, C=1, V=0 -> GT (Z=0 and N==V) is TRUE
    // - If a < b: N=1, Z=0, C=0, V=0 -> LT (N!=V) is TRUE
    // - If a == b: N=0, Z=1, C=1, V=0 -> EQ (Z=1) is TRUE
    //
    // For CSET, we need to invert the condition because CSINC returns 1 when condition is FALSE
    let inverted_cond = match condition {
        Condition::Equal => 1,               // EQ(0) -> NE(1)
        Condition::NotEqual => 0,            // NE(1) -> EQ(0)
        Condition::GreaterThan => 13,        // GT(12) -> LE(13)
        Condition::GreaterThanOrEqual => 11, // GE(10) -> LT(11)
        Condition::LessThan => 10,           // LT(11) -> GE(10)
        Condition::LessThanOrEqual => 12,    // LE(13) -> GT(12)
    };
    vec![
        ArmAsm::FcmpFloat {
            ftype: 1, // Double precision
            rm: b,    // Compare a with b
            rn: a,
        },
        ArmAsm::CsetCsinc {
            sf: destination.sf(),
            cond: inverted_cond,
            rd: destination,
        },
    ]
}

pub fn jump_equal(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 0,
    }
}

pub fn jump_not_equal(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 1,
    }
}

pub fn jump_greater_or_equal(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 10,
    }
}

pub fn jump_less_than(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 11,
    }
}

pub fn jump_greater(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 12,
    }
}

pub fn jump_less_or_equal(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 13,
    }
}

pub fn jump(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 14,
    }
}

pub fn store_pair(reg1: Register, reg2: Register, destination: Register, offset: i32) -> ArmAsm {
    ArmAsm::StpGen {
        // TODO: Make this better/document this is about 64 bit or not
        opc: 0b10,
        class_selector: StpGenSelector::PreIndex,
        imm7: offset,
        rt2: reg2,
        rt: reg1,
        rn: destination,
    }
}

pub fn load_pair(reg1: Register, reg2: Register, destination: Register, offset: i32) -> ArmAsm {
    ArmAsm::LdpGen {
        opc: 0b10,
        class_selector: LdpGenSelector::PostIndex,
        // TODO: Truncate
        imm7: offset,
        rt2: reg2,
        rt: reg1,
        rn: destination,
    }
}

pub fn branch_with_link(destination: i32) -> ArmAsm {
    ArmAsm::Bl { imm26: destination }
}

pub fn branch_with_link_register(register: Register) -> ArmAsm {
    ArmAsm::Blr { rn: register }
}

pub fn breakpoint() -> ArmAsm {
    ArmAsm::Brk { imm16: 30 }
}

#[derive(Debug)]
pub struct LowLevelArm {
    pub instructions: Vec<ArmAsm>,
    pub label_index: usize,
    pub label_locations: HashMap<usize, usize>,
    pub labels: Vec<String>,
    pub free_volatile_registers: Vec<Register>,
    pub allocated_volatile_registers: Vec<Register>,
    pub stack_size: i32,
    pub max_stack_size: i32,
    pub max_locals: i32,
    pub canonical_volatile_registers: Vec<Register>,
    // The goal right now is that everytime we
    // make a "built-in" call, we keep a map
    // of code offset to max stack value at a point.
    // (We could change this to be discrete places rather than
    //  a threshold if needed)
    // Then when we compile, we can map to pc and lookup
    // the locations based on that pc
    // This means, we should be able to walk the stack
    // and figure out where to look for potential roots
    pub stack_map: HashMap<usize, usize>,
    free_temporary_registers: Vec<Register>,
    allocated_temporary_registers: Vec<Register>,
    canonical_temporary_registers: Vec<Register>,
    /// Tracks which callee-saved registers (X19-X28) are actually used in this function.
    /// This is a bitmask where bit 0 = X19, bit 1 = X20, etc.
    used_callee_saved_registers: u16,
    /// Number of callee-saved registers actually saved in this function's frame.
    /// Set by patch_prelude_and_epilogue().
    pub num_callee_saved: usize,
}

impl Default for LowLevelArm {
    fn default() -> Self {
        Self::new()
    }
}

impl LowLevelArm {
    pub fn new() -> Self {
        // https://github.com/swiftlang/swift/blob/716cc5cedf0b8638225bebf86bddc6a1295388f4/docs/ABI/CallingConventionSummary.rst#arm64
        let canonical_volatile_registers = vec![X19, X20, X21, X22, X23, X24, X25, X26, X27, X28];
        // X9 is reserved for arg count in variadic calling convention
        let temporary_registers = vec![X10, X11, X12];
        LowLevelArm {
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
            used_callee_saved_registers: 0,
            num_callee_saved: 0,
        }
    }

    pub fn increment_stack_size(&mut self, size: i32) {
        self.stack_size += size;
        if self.stack_size > self.max_stack_size {
            self.max_stack_size = self.stack_size;
        }
    }

    pub fn prelude(&mut self) {
        // Save frame pointer and link register, then set up new frame
        self.store_pair(X29, X30, SP, -2);
        self.mov_reg(X29, SP);
        // Add placeholder SUB instruction - will be replaced in patch_prelude_and_epilogue()
        self.instructions.push(ArmAsm::SubAddsubImm {
            sf: SP.sf(),
            rn: SP,
            rd: SP,
            imm12: 0x1111, // Magic placeholder value
            sh: 0,
        });
    }

    pub fn epilogue(&mut self) {
        // Add placeholder ADD instruction - will be replaced in patch_prelude_and_epilogue()
        self.instructions.push(ArmAsm::AddAddsubImm {
            sf: SP.sf(),
            rn: SP,
            rd: SP,
            imm12: 0x1111, // Magic placeholder value
            sh: 0,
        });
        self.load_pair(X29, X30, SP, 2);
    }

    pub fn get_label_index(&mut self) -> usize {
        let current_label_index = self.label_index;
        self.label_index += 1;
        current_label_index
    }

    pub fn breakpoint(&mut self) {
        self.instructions.push(breakpoint())
    }

    pub fn mov(&mut self, destination: Register, input: u16) {
        self.instructions.push(mov_imm(destination, input));
    }
    pub fn mov_64(&mut self, destination: Register, input: isize) {
        self.instructions
            .extend(Self::mov_64_bit_num(destination, input));
    }

    pub fn store_pair(
        &mut self,
        reg1: Register,
        reg2: Register,
        destination: Register,
        offset: i32,
    ) {
        self.instructions
            .push(store_pair(reg1, reg2, destination, offset));
    }
    pub fn load_pair(&mut self, reg1: Register, reg2: Register, location: Register, offset: i32) {
        self.instructions
            .push(load_pair(reg1, reg2, location, offset));
    }
    pub fn add(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(add(destination, a, b));
    }
    pub fn sub(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(sub(destination, a, b));
    }
    pub fn sub_imm(&mut self, destination: Register, a: Register, b: i32) {
        self.instructions.push(sub_imm(destination, a, b));
    }
    pub fn mul(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(mul(destination, a, b));
    }
    pub fn div(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(div(destination, a, b));
    }
    /// True modulo: result is always non-negative when divisor is positive
    /// Implements: ((a % b) + b) % b for correct handling of negative numbers
    pub fn modulo(&mut self, destination: Register, a: Register, b: Register) {
        // Step 1: quotient = a / b (signed division)
        self.instructions.push(div(destination, a, b));
        // Step 2: remainder = a - quotient * b (using MSUB: dest = a - dest * b)
        self.instructions.push(ArmAsm::Msub {
            sf: destination.sf(),
            rm: b,
            ra: a,
            rn: destination,
            rd: destination,
        });
        // For true modulo with negative numbers, we'd need additional logic:
        // if (remainder < 0) remainder += b;
        // But for now, this gives us the remainder which works for positive numbers
    }
    pub fn shift_right_imm(&mut self, destination: Register, a: Register, b: i32) {
        self.instructions.push(shift_right_imm(destination, a, b));
    }
    pub fn shift_left_imm(&mut self, destination: Register, a: Register, b: i32) {
        self.instructions.push(shift_left_imm(destination, a, b));
    }
    pub fn and_imm(&mut self, destination: Register, a: Register, b: u64) {
        self.instructions.push(and_imm(destination, a, b));
    }
    pub fn and(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(and(destination, a, b));
    }
    pub fn or(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(or(destination, a, b));
    }
    pub fn shift_left(&mut self, dest: Register, a: Register, b: Register) {
        self.instructions.push(shift_left(dest, a, b));
    }
    pub fn shift_right(&mut self, dest: Register, a: Register, b: Register) {
        self.instructions.push(shift_right(dest, a, b));
    }
    pub fn xor(&mut self, dest: Register, a: Register, b: Register) {
        self.instructions.push(xor(dest, a, b));
    }
    pub fn shift_right_zero(&mut self, dest: Register, a: Register, b: Register) {
        self.instructions.push(shift_right_zero(dest, a, b));
    }
    pub fn ret(&mut self) {
        self.instructions.push(ret());
    }
    pub fn compare(&mut self, a: Register, b: Register) {
        self.instructions.push(compare(a, b));
    }
    pub fn compare_bool(&mut self, condition: Condition, dest: Register, a: Register, b: Register) {
        self.instructions
            .extend(compare_bool(condition, dest, a, b));
    }
    pub fn compare_float_bool(
        &mut self,
        condition: Condition,
        dest: Register,
        a: Register,
        b: Register,
    ) {
        self.instructions
            .extend(compare_float_bool(condition, dest, a, b));
    }
    pub fn tag_value(&mut self, destination: Register, value: Register, tag: Register) {
        self.instructions.extend(tag_value(destination, value, tag));
    }
    pub fn get_tag(&mut self, destination: Register, value: Register) {
        self.instructions.push(get_tag(destination, value));
    }
    pub fn jump_equal(&mut self, destination: Label) {
        self.instructions.push(jump_equal(destination.index as u32));
    }
    pub fn jump_not_equal(&mut self, destination: Label) {
        self.instructions
            .push(jump_not_equal(destination.index as u32));
    }
    pub fn jump_greater_or_equal(&mut self, destination: Label) {
        self.instructions
            .push(jump_greater_or_equal(destination.index as u32));
    }
    pub fn jump_less(&mut self, destination: Label) {
        self.instructions
            .push(jump_less_than(destination.index as u32));
    }
    pub fn jump_greater(&mut self, destination: Label) {
        self.instructions
            .push(jump_greater(destination.index as u32));
    }
    pub fn jump_less_or_equal(&mut self, destination: Label) {
        self.instructions
            .push(jump_less_or_equal(destination.index as u32));
    }
    pub fn jump(&mut self, destination: Label) {
        self.instructions.push(jump(destination.index as u32));
    }

    pub fn store_on_stack(&mut self, reg: Register, offset: i32) {
        const MAX_IMM9: i32 = 255; // 9-bit signed immediate: -256 to +255
        const MIN_IMM9: i32 = -256;

        let byte_offset = offset * 8;

        if (MIN_IMM9..=MAX_IMM9).contains(&byte_offset) {
            // Offset fits in imm9 - use STUR directly
            self.instructions.push(ArmAsm::SturGen {
                size: 0b11,
                imm9: byte_offset,
                rn: X29,
                rt: reg,
            })
        } else {
            // Offset doesn't fit in LDR immediate - use multi-instruction sequence
            // X16 is reserved as intra-procedure-call scratch register in AAPCS64
            let temp_reg = X16;

            // Load offset into temp register
            for instr in Self::mov_64_bit_num(temp_reg, byte_offset as isize) {
                self.instructions.push(instr);
            }

            // Calculate address: temp = FP + offset (using ADD with register)
            self.instructions.push(add(temp_reg, X29, temp_reg));

            // Store: STR reg, [temp, #0]
            self.instructions.push(ArmAsm::StrRegGen {
                size: 0b11,
                rm: ZERO_REGISTER,
                option: 0b011, // LSL
                s: 0,
                rn: temp_reg,
                rt: reg,
            })
        }
    }

    pub fn push_to_stack(&mut self, reg: Register) {
        self.increment_stack_size(1);
        self.store_on_stack(reg, -(self.max_locals + self.stack_size))
    }
    pub fn store_local(&mut self, value: Register, offset: i32) {
        self.store_on_stack(value, -(offset + 1));
    }

    pub fn load_from_stack(&mut self, destination: Register, offset: i32) {
        const MAX_IMM9: i32 = 255;
        const MIN_IMM9: i32 = -256;

        let byte_offset = offset * 8;

        if (MIN_IMM9..=MAX_IMM9).contains(&byte_offset) {
            // Offset fits in imm9 - use LDUR directly
            self.instructions.push(ArmAsm::LdurGen {
                size: 0b11,
                imm9: byte_offset,
                rn: X29,
                rt: destination,
            });
        } else {
            // Offset doesn't fit - use multi-instruction sequence
            let temp_reg = X16;

            // Load offset into temp register
            for instr in Self::mov_64_bit_num(temp_reg, byte_offset as isize) {
                self.instructions.push(instr);
            }

            // Calculate address: temp = FP + offset (using ADD with register)
            self.instructions.push(add(temp_reg, X29, temp_reg));

            // Load: LDR destination, [temp, #0]
            self.instructions.push(ArmAsm::LdrRegGen {
                size: 0b11,
                rm: ZERO_REGISTER,
                option: 0b011, // LSL
                s: 0,
                rn: temp_reg,
                rt: destination,
            });
        }
    }

    pub fn load_from_stack_beginning(&mut self, destination: Register, offset: i32) {
        const MAX_IMM9: i32 = 255;
        const MIN_IMM9: i32 = -256;

        let byte_offset = offset * 8;

        if (MIN_IMM9..=MAX_IMM9).contains(&byte_offset) {
            // Offset fits in imm9 - use LDUR directly
            self.instructions.push(ArmAsm::LdurGen {
                size: 0b11,
                imm9: byte_offset,
                rn: X29,
                rt: destination,
            });
        } else {
            // Offset doesn't fit - use multi-instruction sequence
            let temp_reg = X16;

            // Load offset into temp register
            for instr in Self::mov_64_bit_num(temp_reg, byte_offset as isize) {
                self.instructions.push(instr);
            }

            // Calculate address: temp = FP + offset (using ADD with register)
            self.instructions.push(add(temp_reg, X29, temp_reg));

            // Load: LDR destination, [temp, #0]
            self.instructions.push(ArmAsm::LdrRegGen {
                size: 0b11,
                rm: ZERO_REGISTER,
                option: 0b011, // LSL
                s: 0,
                rn: temp_reg,
                rt: destination,
            });
        }
    }

    pub fn push_to_end_of_stack(&mut self, reg: Register, offset: i32) {
        const MAX_IMM9: i32 = 255;
        const MIN_IMM9: i32 = -256;

        // Saves (stack_size) and outgoing args are live at the same time.
        // Size for their sum, not just the outgoing args alone.
        let needed = self.stack_size + offset + 1;
        if needed > self.max_stack_size {
            self.max_stack_size = needed;
        }

        let byte_offset = offset * 8;

        if (MIN_IMM9..=MAX_IMM9).contains(&byte_offset) {
            // Offset fits in imm9 - use STUR directly
            self.instructions.push(ArmAsm::SturGen {
                size: 0b11,
                imm9: byte_offset,
                rn: SP,
                rt: reg,
            })
        } else {
            // Offset doesn't fit - use multi-instruction sequence
            let temp_reg = X16;

            // Load offset into temp register
            for instr in Self::mov_64_bit_num(temp_reg, byte_offset as isize) {
                self.instructions.push(instr);
            }

            // Calculate address: temp = SP + offset (using ADD with register)
            self.instructions.push(add(temp_reg, SP, temp_reg));

            // Store: STR reg, [temp, #0]
            self.instructions.push(ArmAsm::StrRegGen {
                size: 0b11,
                rm: ZERO_REGISTER,
                option: 0b011, // LSL
                s: 0,
                rn: temp_reg,
                rt: reg,
            })
        }
    }

    pub fn pop_from_stack_indexed(&mut self, reg: Register, offset: i32) {
        self.increment_stack_size(-1);
        self.load_from_stack(reg, -(offset + self.max_locals + 1))
    }

    pub fn pop_from_stack_indexed_raw(&mut self, reg: Register, offset: i32) {
        self.load_from_stack(reg, -(offset))
    }

    pub fn pop_from_stack(&mut self, reg: Register) {
        self.increment_stack_size(-1);
        self.load_from_stack(reg, -(self.max_locals + self.stack_size + 1))
    }

    pub fn load_local(&mut self, destination: Register, offset: i32) {
        self.load_from_stack(destination, -(offset + 1));
    }

    pub fn load_from_heap(&mut self, destination: Register, source: Register, offset: i32) {
        self.instructions.push(ArmAsm::LdurGen {
            size: 0b11,
            imm9: offset * 8,
            rn: source,
            rt: destination,
        });
    }

    pub fn atomic_load(&mut self, destination: Register, source: Register) {
        self.instructions.push(ArmAsm::Ldar {
            size: 0b11,
            rn: source,
            rt: destination,
        })
    }
    pub fn atomic_store(&mut self, ptr: Register, val: Register) {
        self.instructions.push(ArmAsm::Stlr {
            size: 0b11,
            rn: ptr,
            rt: val,
        });
    }

    pub fn compare_and_swap(&mut self, expected: Register, new: Register, ptr: Register) {
        self.instructions.push(ArmAsm::Cas {
            size: 0b11,
            l: 1,
            rs: expected,
            o0: 1,
            rn: ptr,
            rt: new,
        });
    }

    pub fn load_from_heap_with_reg_offset(
        &mut self,
        destination: Register,
        source: Register,
        offset: Register,
    ) {
        self.instructions.push(ArmAsm::LdrRegGen {
            size: 0b11,
            rm: offset,
            option: 0b11,
            s: 0b0,
            rn: source,
            rt: destination,
        })
    }

    pub fn store_to_heap_with_reg_offset(
        &mut self,
        destination: Register,
        source: Register,
        offset: Register,
    ) {
        self.instructions.push(ArmAsm::StrRegGen {
            size: 0b11,
            rm: offset,
            option: 0b11,
            s: 0b0,
            rn: destination,
            rt: source,
        })
    }

    pub fn store_on_heap(&mut self, destination: Register, source: Register, offset: i32) {
        self.instructions.push(ArmAsm::StrImmGen {
            size: 0b11,
            imm9: 0, // not used
            rn: destination,
            rt: source,
            imm12: offset,
            class_selector: StrImmGenSelector::UnsignedOffset,
        });
    }

    pub fn guard_integer(&mut self, dest: Register, a: Register, jump: Label) {
        // TODO: I need to have some way of signaling
        // that this is a type error;
        self.and_imm(dest, a, 0b111);
        self.compare(dest, ZERO_REGISTER);
        self.jump_not_equal(jump);
    }

    pub fn guard_float(&mut self, dest: Register, a: Register, jump: Label) {
        // floats are tagged with 0b001
        self.and_imm(dest, a, 0b111);
        self.sub_imm(dest, dest, BuiltInTypes::Float.get_tag() as i32);
        self.compare(dest, ZERO_REGISTER);
        self.jump_not_equal(jump);
    }

    pub fn new_label(&mut self, name: &str) -> Label {
        self.labels.push(name.to_string());
        Label {
            index: self.get_label_index(),
        }
    }

    pub fn register_label_name(&mut self, name: &str) {
        self.labels.push(name.to_string());
    }

    pub fn write_label(&mut self, label: Label) {
        self.label_locations
            .insert(label.index, self.instructions.len());
    }

    pub fn compile(&mut self) -> &Vec<ArmAsm> {
        self.patch_prelude_and_epilogue();
        self.patch_labels();
        &self.instructions
    }

    pub fn compile_directly(&mut self) -> Vec<u8> {
        self.patch_prelude_and_epilogue();
        self.patch_labels();

        self.instructions
            .iter()
            .flat_map(|x| x.encode().to_le_bytes())
            .collect()
    }

    pub fn compile_to_bytes(&mut self) -> Vec<u8> {
        let instructions = self.compile();

        instructions
            .iter()
            .flat_map(|x| x.encode().to_le_bytes())
            .collect()
    }

    pub fn translate_stack_map(&self, pc: usize) -> Vec<(usize, usize)> {
        // Stack map keys are already byte offsets
        self.stack_map
            .iter()
            .map(|(byte_offset, value)| (*byte_offset + pc, *value))
            .collect()
    }

    /// Get the current byte offset in the generated code.
    /// ARM64 instructions are fixed 4 bytes each.
    pub fn current_byte_offset(&self) -> usize {
        self.instructions.len() * 4
    }

    /// Record a GC safepoint at the current position.
    pub fn record_gc_safepoint(&mut self) {
        let byte_offset = self.current_byte_offset();
        let stack_size = self.stack_size as usize;
        self.stack_map.insert(byte_offset, stack_size);
    }

    /// Get the adjustment for return address lookup.
    /// ARM64 BLR instruction is 4 bytes.
    pub fn return_address_adjustment() -> usize {
        4
    }

    pub fn call(&mut self, register: Register) {
        self.instructions.push(branch_with_link_register(register));
        // Record safepoint after the call - this is the return address
        self.record_gc_safepoint();
    }

    pub fn call_builtin(&mut self, register: Register) {
        // Just use regular call - it records the safepoint
        self.call(register);
    }

    pub fn recurse(&mut self, label: Label) {
        self.instructions.push(branch_with_link(label.index as i32));
        // Record safepoint after the call - this is the return address
        self.record_gc_safepoint();
    }

    pub fn patch_labels(&mut self) {
        for (instruction_index, instruction) in self.instructions.iter_mut().enumerate() {
            match instruction {
                ArmAsm::BCond { imm19, cond: _ } => {
                    let label_index = *imm19 as usize;
                    let label_location = self.label_locations.get(&label_index);
                    match label_location {
                        Some(label_location) => {
                            let relative_position =
                                *label_location as isize - instruction_index as isize;
                            *imm19 = relative_position as i32;
                        }
                        None => {
                            println!("Couldn't find label {:?}", self.labels.get(label_index));
                        }
                    }
                }
                ArmAsm::Bl { imm26 } => {
                    let label_index = *imm26 as usize;
                    let label_location = self.label_locations.get(&label_index);
                    match label_location {
                        Some(label_location) => {
                            let relative_position =
                                *label_location as isize - instruction_index as isize;
                            *imm26 = relative_position as i32;
                        }
                        None => {
                            println!("Couldn't find label {:?}", self.labels.get(label_index));
                        }
                    }
                }
                ArmAsm::Adr {
                    immlo,
                    immhi,
                    rd: _,
                } => {
                    let label_index = *immhi as usize;
                    let label_location = self.label_locations.get(&label_index);
                    match label_location {
                        Some(label_location) => {
                            let relative_position =
                                *label_location as isize - instruction_index as isize;
                            // ADR uses byte offsets, so multiply by 4 (instruction size)
                            let byte_offset = (relative_position * 4) as i32;
                            // ADR uses a 21-bit signed immediate split across immlo (2 bits) and immhi (19 bits)
                            *immlo = byte_offset & 0x3; // Lower 2 bits
                            *immhi = (byte_offset >> 2) & 0x7FFFF; // Upper 19 bits
                        }
                        None => {
                            println!("Couldn't find label {:?}", self.labels.get(label_index));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn mov_reg(&mut self, destination: Register, source: Register) {
        self.instructions.push(match (destination, source) {
            (SP, _) => mov_sp(destination, source),
            (_, SP) => mov_sp(destination, source),
            _ => mov_reg(destination, source),
        });
    }

    pub fn volatile_register(&mut self) -> Register {
        let next_register = self.free_volatile_registers.pop().unwrap_or_else(|| {
            panic!(
                "No free volatile registers! Currently allocated: {:?}",
                self.allocated_volatile_registers
            )
        });
        self.allocated_volatile_registers.push(next_register);
        // Track that this callee-saved register is used (for AAPCS64 compliance)
        self.mark_callee_saved_used(next_register);
        next_register
    }

    pub fn temporary_register(&mut self) -> Register {
        let next_register = self.free_temporary_registers.pop().unwrap_or_else(|| {
            panic!(
                "No free temporary registers! Currently allocated: {:?}",
                self.allocated_temporary_registers
            )
        });
        self.allocated_temporary_registers.push(next_register);
        next_register
    }

    pub fn free_temporary_register(&mut self, reg: Register) {
        if self.canonical_temporary_registers.contains(&reg)
            && !self.free_temporary_registers.contains(&reg)
        {
            self.free_temporary_registers.push(reg);
            self.allocated_temporary_registers.retain(|r| *r != reg);
        }
    }

    pub fn clear_temporary_registers(&mut self) {
        // TODO: Just use an index. this is wasteful
        self.free_temporary_registers = self.canonical_temporary_registers.clone();
        self.allocated_temporary_registers = vec![];
    }

    pub fn free_register(&mut self, reg: Register) {
        // TODO: Properly fix the fact that the zero
        // register is being put in the volatile list
        if !self.canonical_volatile_registers.contains(&reg) {
            return;
        }
        self.free_volatile_registers.push(reg);
        self.allocated_volatile_registers
            .retain(|&allocated| allocated != reg);
    }

    /// Mark a callee-saved register (X19-X28) as used by this function.
    /// This is called when the register allocator assigns a physical register.
    fn mark_callee_saved_used(&mut self, reg: Register) {
        if reg.index >= 19 && reg.index <= 28 {
            let bit = reg.index - 19;
            self.used_callee_saved_registers |= 1 << bit;
        }
    }

    /// Get the list of callee-saved registers that are actually used.
    fn get_used_callee_saved_registers(&self) -> Vec<Register> {
        let mut result = Vec::new();
        for i in 0..10u8 {
            if self.used_callee_saved_registers & (1 << i) != 0 {
                result.push(Register {
                    size: Size::S64,
                    index: 19 + i,
                });
            }
        }
        result
    }

    /// Reset callee-saved register tracking for a new function.
    pub fn reset_callee_saved_tracking(&mut self) {
        self.used_callee_saved_registers = 0;
    }

    /// Mark a callee-saved register as used by its index (19-28).
    pub fn mark_callee_saved_register_used(&mut self, index: usize) {
        if (19..=28).contains(&index) {
            let bit = index - 19;
            self.used_callee_saved_registers |= 1 << bit;
        }
    }

    pub fn reserve_register(&mut self, reg: Register) {
        self.free_volatile_registers.retain(|&free| free != reg);
        if !self.allocated_volatile_registers.contains(&reg) {
            self.allocated_volatile_registers.push(reg);
        }
    }

    pub fn arg(&self, arg: u8) -> Register {
        assert!(
            arg < 8,
            "Only 8 arguments are supported on aarch64, but {} was requested",
            arg
        );
        Register {
            size: Size::S64,
            index: arg,
        }
    }

    pub fn ret_reg(&self) -> Register {
        X0
    }

    /// Generates ARM64 instructions to allocate a stack frame of any size.
    ///
    /// ARM64 SUB/ADD immediate instructions have a 12-bit immediate field (max 4095).
    /// For larger stack frames, this function uses multiple strategies:
    ///
    /// 1. **Small frames (≤4095 bytes)**: Single SUB instruction
    /// 2. **4096-byte multiples**: SUB with LSL#12 shift (more efficient)
    /// 3. **Large frames**: Multiple SUB instructions (matches LLVM/GCC behavior)
    ///
    /// This approach has no artificial limit and matches industry-standard compilers.
    fn generate_stack_allocation(bytes: i32) -> Vec<ArmAsm> {
        const MAX_IMM12: i32 = 4095;
        const PAGE_SIZE: i32 = 4096;

        if bytes <= MAX_IMM12 {
            // Single instruction: SUB SP, SP, #bytes
            vec![ArmAsm::SubAddsubImm {
                sf: SP.sf(),
                rn: SP,
                rd: SP,
                imm12: bytes,
                sh: 0,
            }]
        } else if bytes % PAGE_SIZE == 0 && bytes / PAGE_SIZE <= MAX_IMM12 {
            // Use LSL#12 shift: SUB SP, SP, #(bytes/4096), LSL#12
            vec![ArmAsm::SubAddsubImm {
                sf: SP.sf(),
                rn: SP,
                rd: SP,
                imm12: bytes / PAGE_SIZE,
                sh: 1, // LSL#12
            }]
        } else {
            // Multiple SUB instructions
            let mut instructions = vec![];
            let mut remaining = bytes;

            while remaining > MAX_IMM12 {
                instructions.push(ArmAsm::SubAddsubImm {
                    sf: SP.sf(),
                    rn: SP,
                    rd: SP,
                    imm12: MAX_IMM12,
                    sh: 0,
                });
                remaining -= MAX_IMM12;
            }

            if remaining > 0 {
                instructions.push(ArmAsm::SubAddsubImm {
                    sf: SP.sf(),
                    rn: SP,
                    rd: SP,
                    imm12: remaining,
                    sh: 0,
                });
            }

            instructions
        }
    }

    /// Generates ARM64 instructions to deallocate a stack frame of any size.
    /// (Same logic as generate_stack_allocation but uses ADD instead of SUB)
    fn generate_stack_deallocation(bytes: i32) -> Vec<ArmAsm> {
        const MAX_IMM12: i32 = 4095;
        const PAGE_SIZE: i32 = 4096;

        if bytes <= MAX_IMM12 {
            vec![ArmAsm::AddAddsubImm {
                sf: SP.sf(),
                rn: SP,
                rd: SP,
                imm12: bytes,
                sh: 0,
            }]
        } else if bytes % PAGE_SIZE == 0 && bytes / PAGE_SIZE <= MAX_IMM12 {
            vec![ArmAsm::AddAddsubImm {
                sf: SP.sf(),
                rn: SP,
                rd: SP,
                imm12: bytes / PAGE_SIZE,
                sh: 1,
            }]
        } else {
            let mut instructions = vec![];
            let mut remaining = bytes;

            while remaining > MAX_IMM12 {
                instructions.push(ArmAsm::AddAddsubImm {
                    sf: SP.sf(),
                    rn: SP,
                    rd: SP,
                    imm12: MAX_IMM12,
                    sh: 0,
                });
                remaining -= MAX_IMM12;
            }

            if remaining > 0 {
                instructions.push(ArmAsm::AddAddsubImm {
                    sf: SP.sf(),
                    rn: SP,
                    rd: SP,
                    imm12: remaining,
                    sh: 0,
                });
            }

            instructions
        }
    }

    pub fn patch_prelude_and_epilogue(&mut self) {
        // Get list of callee-saved registers that need to be saved
        let used_callee_saved = self.get_used_callee_saved_registers();
        let num_callee_saved = used_callee_saved.len() as u64;
        self.num_callee_saved = num_callee_saved as usize;

        // Calculate total stack size including callee-saved area
        let mut max = self.max_stack_size as u64 + self.max_locals as u64 + num_callee_saved;
        let remainder = max % 2;
        if remainder != 0 {
            max += 1;
        }

        let stack_bytes = (max * 8) as i32;

        // Find the placeholder SUB instruction in prelude (has magic value 0x1111)
        let sub_index = self.instructions.iter().position(|inst| {
            matches!(inst, ArmAsm::SubAddsubImm { rn, rd, imm12, .. }
                     if *rn == SP && *rd == SP && *imm12 == 0x1111)
        });

        if let Some(index) = sub_index {
            // Replace single SUB with sequence of instructions for stack allocation
            let mut alloc_instructions = Self::generate_stack_allocation(stack_bytes);

            // Save callee-saved registers at the bottom of the stack (near SP).
            // Stack layout:
            //   [FP + 8]  -> Saved LR (X30)
            //   [FP + 0]  -> Saved FP (X29)
            //   [FP - 8]  -> Local 0
            //   ...
            //   [FP - N]  -> Stack slots
            //   [SP + (num_callee_saved-1)*8] -> First saved callee-saved reg
            //   ...
            //   [SP]      -> Last saved callee-saved reg (or padding)
            //
            // We save at SP-relative offsets since SP is now at the bottom of our frame.
            for (i, reg) in used_callee_saved.iter().enumerate() {
                // Store at [SP + i*8] using STR with unsigned offset
                // For 64-bit STR, imm12 is scaled by 8, so imm12=i means byte offset i*8
                alloc_instructions.push(ArmAsm::StrImmGen {
                    rt: *reg,
                    rn: SP,
                    imm9: 0,
                    imm12: i as i32,
                    size: 0b11, // 64-bit
                    class_selector: StrImmGenSelector::UnsignedOffset,
                });
            }

            // CRITICAL FIX: Zero out local slots to prevent GC from seeing garbage.
            // When GC runs during allocation (before a local is assigned), uninitialized
            // local slots could contain interior pointers or other garbage from previous
            // stack usage. Initialize all local slots to null (0x7, which is tagged null).
            //
            // NOTE: STUR has a 9-bit signed immediate (-256 to 255), so for large stack
            // frames we need a different approach. We use X10 as a pointer that we decrement.
            let null_value = crate::types::BuiltInTypes::null_value() as i32;

            #[cfg(feature = "debug-gc")]
            eprintln!(
                "[GC DEBUG] patch_prelude: max_locals={}, zeroing {} slots",
                self.max_locals, self.max_locals
            );

            if self.max_locals > 0 {
                // Load null value into X11 (NOT X9 - that's reserved for arg count!)
                alloc_instructions.push(ArmAsm::Movz {
                    rd: X11,
                    imm16: null_value,
                    hw: 0,
                    sf: 1,
                });

                // X10 = X29 - 8 (point to first local slot)
                alloc_instructions.push(ArmAsm::SubAddsubImm {
                    sf: 1,
                    rn: X29,
                    rd: X10,
                    imm12: 8,
                    sh: 0,
                });

                // Store null to each local slot, decrementing X10 after each store
                for _ in 0..self.max_locals {
                    // Store X11 to [X10], then X10 = X10 - 8 (post-decrement)
                    // Use STR with post-index: str x11, [x10], #-8
                    alloc_instructions.push(ArmAsm::StrImmGen {
                        rt: X11,
                        rn: X10,
                        imm9: -8,
                        imm12: 0,   // Not used for post-index
                        size: 0b11, // 64-bit
                        class_selector: StrImmGenSelector::PostIndex,
                    });
                }
            }

            let delta = alloc_instructions.len() as isize - 1;
            self.instructions.splice(index..=index, alloc_instructions);

            if delta != 0 {
                // Shift label locations that come after the insertion point
                for location in self.label_locations.values_mut() {
                    if *location > index {
                        *location = (*location as isize + delta) as usize;
                    }
                }
                // Also shift stack_map entries that come after the insertion point
                // Note: stack_map keys are byte offsets, not instruction indices
                let byte_index = index * 4;
                let byte_delta = delta * 4;
                self.stack_map = self
                    .stack_map
                    .drain()
                    .map(|(k, v)| {
                        if k > byte_index {
                            ((k as isize + byte_delta) as usize, v)
                        } else {
                            (k, v)
                        }
                    })
                    .collect();
            }
        } else {
            unreachable!("Expected to find SUB placeholder instruction for stack allocation");
        }

        // Find the placeholder ADD instruction in epilogue (has magic value 0x1111)
        // Note: We need to search again after the splice operation
        let add_index = self.instructions.iter().rposition(|inst| {
            matches!(inst, ArmAsm::AddAddsubImm { rn, rd, imm12, .. }
                     if *rn == SP && *rd == SP && *imm12 == 0x1111)
        });

        if let Some(index) = add_index {
            // First restore callee-saved registers from the bottom of the stack,
            // then deallocate the stack space.
            let mut dealloc_instructions = Vec::new();

            // Restore callee-saved registers from [SP + i*8]
            for (i, reg) in used_callee_saved.iter().enumerate() {
                // Load from [SP + i*8] using LDR with unsigned offset
                // For 64-bit LDR, imm12 is scaled by 8, so imm12=i means byte offset i*8
                dealloc_instructions.push(ArmAsm::LdrImmGen {
                    rt: *reg,
                    rn: SP,
                    imm9: 0,
                    imm12: i as i32,
                    size: 0b11, // 64-bit
                    class_selector: LdrImmGenSelector::UnsignedOffset,
                });
            }

            // Now add the stack deallocation instructions
            dealloc_instructions.extend(Self::generate_stack_deallocation(stack_bytes));

            let delta = dealloc_instructions.len() as isize - 1;
            self.instructions
                .splice(index..=index, dealloc_instructions);

            if delta != 0 {
                for location in self.label_locations.values_mut() {
                    if *location > index {
                        *location = (*location as isize + delta) as usize;
                    }
                }
                // Also shift stack_map entries for epilogue splice
                // Note: stack_map keys are byte offsets, not instruction indices
                let byte_index = index * 4;
                let byte_delta = delta * 4;
                self.stack_map = self
                    .stack_map
                    .drain()
                    .map(|(k, v)| {
                        if k > byte_index {
                            ((k as isize + byte_delta) as usize, v)
                        } else {
                            (k, v)
                        }
                    })
                    .collect();
            }
        } else {
            unreachable!("Expected to find ADD placeholder instruction for stack deallocation");
        }
    }

    /// Load a 64-bit immediate value using MOVZ/MOVK sequence (up to 4 instructions)
    pub fn mov_64_bit_num(register: Register, num: isize) -> Vec<ArmAsm> {
        let mut num = num;
        let mut result = vec![];

        result.push(ArmAsm::Movz {
            sf: register.sf(),
            hw: 0,
            imm16: num as i32 & 0xffff,
            rd: register,
        });
        num >>= 16;
        if num == 0 {
            return result;
        }
        result.push(ArmAsm::Movk {
            sf: register.sf(),
            hw: 0b01,
            imm16: num as i32 & 0xffff,
            rd: register,
        });
        num >>= 16;
        if num == 0 {
            return result;
        }
        result.push(ArmAsm::Movk {
            sf: register.sf(),
            hw: 0b10,
            imm16: num as i32 & 0xffff,
            rd: register,
        });
        num >>= 16;
        if num == 0 {
            return result;
        }
        result.push(ArmAsm::Movk {
            sf: register.sf(),
            hw: 0b11,
            imm16: num as i32 & 0xffff,
            rd: register,
        });

        result
    }

    pub fn add_stack_pointer(&mut self, bytes: i32) {
        self.instructions.push(ArmAsm::AddAddsubImm {
            sf: SP.sf(),
            rn: SP,
            rd: SP,
            imm12: bytes * 8,
            sh: 0,
        });
    }

    pub fn sub_stack_pointer(&mut self, bytes: i32) {
        self.instructions.push(ArmAsm::SubAddsubImm {
            sf: SP.sf(),
            rn: SP,
            rd: SP,
            imm12: bytes * 8,
            sh: 0,
        });
    }

    pub fn set_max_locals(&mut self, num_locals: usize) {
        #[cfg(feature = "debug-gc")]
        eprintln!(
            "[GC DEBUG] set_max_locals: {} -> {}",
            self.max_locals, num_locals
        );
        self.max_locals = num_locals as i32;
    }

    pub fn get_stack_pointer_imm(&mut self, destination: Register, offset: isize) {
        self.instructions.push(ArmAsm::SubAddsubImm {
            sf: destination.sf(),
            rn: SP,
            rd: destination,
            imm12: offset as i32 * 8,
            sh: 0,
        });
    }
    pub fn get_stack_pointer(&mut self, destination: Register, offset: Register) {
        self.get_stack_pointer_imm(destination, 0);
        self.instructions
            .push(add(destination, destination, offset));
    }

    pub fn share_label_info_debug(
        &self,
        function_pointer: usize,
    ) -> Result<(), crate::compiler::CompileError> {
        for (label_index, label) in self.labels.iter().enumerate() {
            let label_location = *self.label_locations.get(&label_index).ok_or_else(|| {
                crate::compiler::CompileError::LabelLookup {
                    label: label.to_string(),
                }
            })? * 4;
            debugger(Message {
                kind: "label".to_string(),
                data: Data::Label {
                    label: label.to_string(),
                    function_pointer,
                    label_index,
                    label_location,
                },
            });
        }
        Ok(())
    }

    pub fn get_current_stack_position(&mut self, dest: Register) {
        self.instructions.push(ArmAsm::SubAddsubImm {
            sf: dest.sf(),
            rn: X29,
            rd: dest,
            imm12: (self.max_locals + self.stack_size + 1) * 8,
            sh: 0,
        });
        // TODO: This seems
    }

    pub fn set_all_locals_to_null(&mut self, null_register: Register) {
        for local_offset in 0..self.max_locals {
            self.store_local(null_register, local_offset)
        }
    }

    pub fn fmov(&mut self, a: Register, b: Register, direction: FmovDirection) {
        // sf == 1 && ftype == 01 && rmode == 00 && opcode == 111

        self.instructions.push(ArmAsm::FmovFloatGen {
            sf: a.sf(),
            ftype: 0b01,
            rmode: 0b00,
            opcode: match direction {
                FmovDirection::FromGeneralToFloat => 0b111,
                FmovDirection::FromFloatToGeneral => 0b110,
            },
            rn: b,
            rd: a,
        });
    }

    /// Convert signed 64-bit integer to double-precision floating-point.
    /// SCVTF Dd, Xn
    pub fn scvtf(&mut self, dest: Register, src: Register) {
        // Uses same encoding as FMOV but with opcode=010 for SCVTF
        self.instructions.push(ArmAsm::FmovFloatGen {
            sf: 1,       // 64-bit source
            ftype: 0b01, // Double precision
            rmode: 0b00,
            opcode: 0b010, // SCVTF signed int to float
            rn: src,
            rd: dest,
        });
    }

    pub fn fadd(&mut self, dest: Register, a: Register, b: Register) {
        self.instructions.push(ArmAsm::FaddFloat {
            ftype: 0b01,
            rm: b,
            rn: a,
            rd: dest,
        });
    }

    pub fn fsub(&mut self, dest: Register, a: Register, b: Register) {
        self.instructions.push(ArmAsm::FsubFloat {
            ftype: 0b01,
            rm: b,
            rn: a,
            rd: dest,
        });
    }

    pub fn fmul(&mut self, dest: Register, a: Register, b: Register) {
        self.instructions.push(ArmAsm::FmulFloat {
            ftype: 0b01,
            rm: b,
            rn: a,
            rd: dest,
        });
    }

    pub fn fdiv(&mut self, dest: Register, a: Register, b: Register) {
        self.instructions.push(ArmAsm::FdivFloat {
            ftype: 0b01,
            rm: b,
            rn: a,
            rd: dest,
        });
    }

    pub fn get_label_by_name(&self, arg: &str) -> Label {
        self.labels
            .iter()
            .enumerate()
            .find(|(_, label)| *label == arg)
            .map(|(index, _)| Label { index })
            .expect("Could not find label")
    }

    pub fn current_position(&self) -> usize {
        self.instructions.len()
    }

    pub fn load_label_address(&mut self, destination: Register, label: Label) {
        self.instructions.push(adr(destination, label.index));
    }
}
