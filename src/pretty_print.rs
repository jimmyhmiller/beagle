use std::collections::HashMap;

use crate::{
    ir::{Condition, Instruction, Value, VirtualRegister},
    machine_code::arm_codegen::{
        ArmAsm, LdpGenSelector, LdrImmGenSelector, Register, Size, StpGenSelector,
        StrImmGenSelector,
    },
};

pub trait PrettyPrint {
    fn pretty_print(&self) -> String;
}

impl PrettyPrint for Value {
    fn pretty_print(&self) -> String {
        match self {
            Value::Register(register) => register.pretty_print(),
            Value::RawValue(value) => format!("raw{}", value),
            Value::Pointer(value) => format!("ptr{}", value),
            Value::TaggedConstant(value) => format!("tagged_constant{}", value),
            Value::StringConstantPtr(value) => format!("string_constant_ptr{}", value),
            Value::Local(value) => format!("local{}", value),
            Value::Function(f) => format!("function{}", f),
            Value::True => "true".to_string(),
            Value::False => "false".to_string(),
            Value::Null => "null".to_string(),
            Value::Spill(value, index) => format!("spill({}, {})", value.pretty_print(), index),
            Value::Stack(value) => format!("stack{}", value),
        }
    }
}

impl PrettyPrint for VirtualRegister {
    fn pretty_print(&self) -> String {
        match self {
            VirtualRegister {
                argument: Some(argument),
                index,
                volatile: _,
                is_physical: _,
            } => {
                format!("arg{}r{}", argument, index)
            }
            VirtualRegister {
                argument: None,
                index,
                volatile: _,
                is_physical: true,
            } => {
                format!("pr{}", index)
            }
            VirtualRegister {
                argument: None,
                index,
                volatile: _,
                is_physical: _,
            } => {
                format!("r{}", index)
            }
        }
    }
}

impl PrettyPrint for Vec<Value> {
    fn pretty_print(&self) -> String {
        let mut result = String::new();
        for value in self {
            result.push_str(&value.pretty_print());
            result.push_str(", ");
        }
        result
    }
}

impl PrettyPrint for Instruction {
    fn pretty_print(&self) -> String {
        match self {
            Instruction::Label(label) => {
                format!("label{}:", label.index)
            }
            Instruction::Sub(value, value1, value2) => {
                format!(
                    "sub {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::AddInt(value, value1, value2) => {
                format!(
                    "add_int {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::Mul(value, value1, value2) => {
                format!(
                    "mul {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::Div(value, value1, value2) => {
                format!(
                    "div {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::Assign(virtual_register, value) => {
                format!(
                    "assign {}, {}",
                    virtual_register.pretty_print(),
                    value.pretty_print()
                )
            }
            Instruction::Recurse(value, vec) => {
                format!("recurse {}, {}", value.pretty_print(), vec.pretty_print())
            }
            Instruction::RecurseWithSaves(value, vec, saves) => {
                format!(
                    "recurse_with_saves {}, {}, {}",
                    value.pretty_print(),
                    vec.pretty_print(),
                    saves.pretty_print()
                )
            }
            Instruction::TailRecurse(value, vec) => {
                format!(
                    "tail_recurse {}, {}",
                    value.pretty_print(),
                    vec.pretty_print()
                )
            }
            Instruction::JumpIf(label, condition, value, value1) => {
                format!(
                    "jump_if {}, {:?}, {}, {}",
                    label.index,
                    condition,
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::Jump(label) => {
                format!("jump {}", label.index)
            }
            Instruction::Ret(value) => {
                format!("ret {}", value.pretty_print())
            }
            Instruction::Breakpoint => "breakpoint".to_string(),
            Instruction::Compare(value, value1, value2, condition) => {
                format!(
                    "compare {}, {}, {}, {:?}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print(),
                    condition
                )
            }
            Instruction::Tag(value, value1, value2) => {
                format!(
                    "tag {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::LoadTrue(value) => {
                format!("load_true {}", value.pretty_print())
            }
            Instruction::LoadFalse(value) => {
                format!("load_false {}", value.pretty_print())
            }
            Instruction::LoadConstant(value, value1) => {
                format!(
                    "load_constant {}, {}",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::Call(value, value1, vec, _) => {
                format!(
                    "{} <- call {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    vec.pretty_print()
                )
            }
            Instruction::CallWithSaves(value, value1, vec, _, saves) => {
                format!(
                    "{} <- call_with_saves {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    vec.pretty_print(),
                    saves.pretty_print()
                )
            }
            Instruction::HeapLoad(value, value1, _) => {
                format!(
                    "heap_load {}, {}, _",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::HeapLoadReg(value, value1, value2) => {
                format!(
                    "heap_load_reg {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::HeapStore(value, value1) => {
                format!(
                    "heap_store {}, {}",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::LoadLocal(value, value1) => {
                format!(
                    "load_local {}, {}",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::StoreLocal(value, value1) => {
                format!(
                    "store_local {}, {}",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::RegisterArgument(value) => {
                format!("register_argument {}", value.pretty_print())
            }
            Instruction::PushStack(value) => {
                format!("push_stack {}", value.pretty_print())
            }
            Instruction::PopStack(value) => {
                format!("pop_stack {}", value.pretty_print())
            }
            Instruction::GetStackPointer(value, value1) => {
                format!(
                    "get_stack_pointer {}, {}",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::GetStackPointerImm(value, _) => {
                format!("get_stack_pointer_imm {}, _", value.pretty_print())
            }
            Instruction::GetTag(value, value1) => {
                format!(
                    "get_tag {}, {}",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::Untag(value, value1) => {
                format!("untag {}, {}", value.pretty_print(), value1.pretty_print())
            }
            Instruction::HeapStoreOffset(value, value1, _) => {
                format!(
                    "heap_store_offset {}, {}, _",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::HeapStoreByteOffsetMasked(value, value1, _, _, _, _, _) => {
                format!(
                    "heap_store_byte_offset_masked {}, {}, _, _, _",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::CurrentStackPosition(value) => {
                format!("current_stack_position {}", value.pretty_print())
            }
            Instruction::ExtendLifeTime(value) => {
                format!("extend_lifetime {}", value.pretty_print())
            }
            Instruction::HeapStoreOffsetReg(value, value1, value2) => {
                format!(
                    "heap_store_offset_reg {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::AtomicLoad(value, value1) => {
                format!(
                    "atomic_load {}, {}",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::AtomicStore(value, value1) => {
                format!(
                    "atomic_store {}, {}",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::CompareAndSwap(value, value1, value2) => {
                format!(
                    "compare_and_swap {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::StoreFloat(value, value1, _) => {
                format!(
                    "store_float {}, {}, _",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::GuardInt(value, value1, label) => {
                format!(
                    "guard_int {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    label.index
                )
            }
            Instruction::GuardFloat(value, value1, label) => {
                format!(
                    "guard_float {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    label.index
                )
            }
            Instruction::FmovGeneralToFloat(value, value1) => {
                format!(
                    "fmov_general_to_float {}, {}",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::FmovFloatToGeneral(value, value1) => {
                format!(
                    "fmov_float_to_general {}, {}",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::AddFloat(value, value1, value2) => {
                format!(
                    "add_float {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::SubFloat(value, value1, value2) => {
                format!(
                    "sub_float {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::MulFloat(value, value1, value2) => {
                format!(
                    "mul_float {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::DivFloat(value, value1, value2) => {
                format!(
                    "div_float {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::ShiftRightImm(value, value1, _) => {
                format!(
                    "shift_right_imm {}, {}, _",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::ShiftRightImmRaw(value, value1, _) => {
                format!(
                    "shift_right_imm_raw {}, {}, _",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::AndImm(value, value1, _) => {
                format!(
                    "and_imm {}, {}, _",
                    value.pretty_print(),
                    value1.pretty_print()
                )
            }
            Instruction::ShiftLeft(value, value1, value2) => {
                format!(
                    "shift_left {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::ShiftRight(value, value1, value2) => {
                format!(
                    "shift_right {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::ShiftRightZero(value, value1, value2) => {
                format!(
                    "shift_right_zero {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::And(value, value1, value2) => {
                format!(
                    "and {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::Or(value, value1, value2) => {
                format!(
                    "or {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::Xor(value, value1, value2) => {
                format!(
                    "xor {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    value2.pretty_print()
                )
            }
            Instruction::SetContinuationMarker => "set_continuation_marker".to_string(),
            Instruction::SetContinuationHandlerAddress(label) => {
                format!("set_continuation_handler_address {}", label.index)
            }
            Instruction::ClearContinuationHandlerAddress => {
                "clear_continuation_handler_address".to_string()
            }
            Instruction::DelimitHandlerValue(dest) => {
                format!("delimit_handler_value {}", dest.pretty_print())
            }
            Instruction::DelimitHandlerContinuation(dest) => {
                format!("delimit_handler_continuation {}", dest.pretty_print())
            }
            Instruction::DelimitPrelude => "delimit_prelude".to_string(),
            Instruction::DelimitEpilogue => "delimit_epilogue".to_string(),
        }
    }
}

impl PrettyPrint for Vec<Instruction> {
    fn pretty_print(&self) -> String {
        let mut result = String::new();
        for instruction in self {
            result.push_str(&instruction.pretty_print());
            result.push('\n');
        }
        result
    }
}

impl PrettyPrint for Vec<ArmAsm> {
    fn pretty_print(&self) -> String {
        let mut result = String::new();
        for instruction in self {
            result.push_str(&instruction.pretty_print());
            result.push('\n');
        }
        result
    }
}

impl PrettyPrint for Register {
    fn pretty_print(&self) -> String {
        if self.size != Size::S64 {
            panic!("Need to deal with size since I'm using it now");
        }
        match self.index {
            0 => "x0".to_string(),
            1 => "x1".to_string(),
            2 => "x2".to_string(),
            3 => "x3".to_string(),
            4 => "x4".to_string(),
            5 => "x5".to_string(),
            6 => "x6".to_string(),
            7 => "x7".to_string(),
            8 => "x8".to_string(),
            9 => "x9".to_string(),
            10 => "x10".to_string(),
            11 => "x11".to_string(),
            12 => "x12".to_string(),
            13 => "x13".to_string(),
            14 => "x14".to_string(),
            15 => "x15".to_string(),
            16 => "x16".to_string(),
            17 => "x17".to_string(),
            18 => "x18".to_string(),
            19 => "x19".to_string(),
            20 => "x20".to_string(),
            21 => "x21".to_string(),
            22 => "x22".to_string(),
            23 => "x23".to_string(),
            24 => "x24".to_string(),
            25 => "x25".to_string(),
            26 => "x26".to_string(),
            27 => "x27".to_string(),
            28 => "x28".to_string(),
            29 => "x29".to_string(),
            30 => "x30".to_string(),
            31 => "xZ".to_string(),
            x => format!("??x{}", x),
        }
    }
}

impl PrettyPrint for Condition {
    fn pretty_print(&self) -> String {
        match self {
            Condition::LessThanOrEqual => "le".to_string(),
            Condition::LessThan => "lt".to_string(),
            Condition::Equal => "eq".to_string(),
            Condition::NotEqual => "ne".to_string(),
            Condition::GreaterThan => "gt".to_string(),
            Condition::GreaterThanOrEqual => "ge".to_string(),
        }
    }
}

impl PrettyPrint for ArmAsm {
    fn pretty_print(&self) -> String {
        match self {
            ArmAsm::Adr { immlo, immhi, rd } => {
                format!("adr {}, #{}", rd.pretty_print(), (*immhi << 2) | *immlo)
            }
            ArmAsm::AddAddsubImm {
                sf: _,
                sh,
                imm12,
                rn,
                rd,
            } => {
                if *sh != 0 {
                    panic!("Need to deal with shift since I'm using it now");
                }
                format!(
                    "add {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    imm12
                )
            }
            ArmAsm::AddAddsubShift {
                sf: _,
                shift,
                rm,
                imm6: _,
                rn,
                rd,
            } => {
                if *shift != 0 {
                    panic!("Need to deal with shift since I'm using it now");
                }
                format!(
                    "add {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::AndLogImm {
                sf: _,
                n: _,
                immr,
                imms: _,
                rn,
                rd,
            } => {
                // TODO: Print better
                format!("and {}, {}, {}", rd.pretty_print(), rn.pretty_print(), immr)
            }
            ArmAsm::AndLogShift {
                sf: _,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                if *shift != 0 || *imm6 != 0 {
                    panic!("Need to deal with shift and imm6 since I'm using it now");
                }
                format!(
                    "and {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::BCond { imm19, cond } => {
                if *cond == 14 {
                    return format!("b 0x{:x}", imm19);
                }
                let condition = Condition::arm_condition_from_i32(*cond);
                format!("b{} 0x{:x}", condition.pretty_print(), imm19)
            }
            ArmAsm::Bl { imm26 } => {
                format!("bl 0x{:x}", imm26)
            }
            ArmAsm::Blr { rn } => {
                format!("blr {}", rn.pretty_print())
            }
            ArmAsm::Brk { imm16 } => {
                format!("brk {}", imm16)
            }
            ArmAsm::Cas {
                size,
                l,
                rs,
                o0,
                rn,
                rt,
            } => {
                if *size != 0b11 || *l != 1 || *o0 != 1 {
                    panic!("Need to deal with size, l and o0 since I'm using it now");
                }
                format!(
                    "cas {}, {}, {}",
                    rt.pretty_print(),
                    rn.pretty_print(),
                    rs.pretty_print()
                )
            }
            ArmAsm::CmpSubsAddsubShift {
                sf: _,
                shift,
                rm,
                imm6,
                rn,
            } => {
                if *shift != 0 || *imm6 != 0 {
                    panic!("Need to deal with shift and imm6 since I'm using it now");
                }
                format!("cmp {}, {}", rn.pretty_print(), rm.pretty_print())
            }
            ArmAsm::CsetCsinc { sf: _, cond, rd } => {
                let condition = Condition::arm_condition_from_i32(*cond);
                format!("cset {}, {}", rd.pretty_print(), condition.pretty_print())
            }
            ArmAsm::Ldar { size, rn, rt } => {
                if *size != 0b11 {
                    panic!("Need to deal with size since I'm using it now");
                }
                format!("ldar {}, {}", rt.pretty_print(), rn.pretty_print())
            }
            ArmAsm::LdpGen {
                opc,
                imm7,
                rt2,
                rn,
                rt,
                class_selector,
            } => {
                if *opc != 0b10 || *class_selector != LdpGenSelector::PostIndex {
                    panic!("Need to deal with opc and class_selector since I'm using it now");
                }
                format!(
                    "ldp {}, {}, [{}], #{}",
                    rt.pretty_print(),
                    rt2.pretty_print(),
                    rn.pretty_print(),
                    imm7
                )
            }
            ArmAsm::LdrImmGen {
                size,
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => {
                if *size != 0b11
                    || *class_selector != LdrImmGenSelector::UnsignedOffset
                    || *imm9 != 0
                {
                    panic!(
                        "Need to deal with size and class_selector and imm9 since I'm using it now"
                    );
                }
                format!(
                    "ldr {}, [{} #{}], ",
                    rt.pretty_print(),
                    rn.pretty_print(),
                    imm12
                )
            }
            ArmAsm::LdrRegGen {
                size,
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                if *size != 0b11 || *option != 0b11 || *s != 0 {
                    panic!("Need to deal with size, option and s since I'm using it now");
                }
                format!(
                    "ldr {}, [{} {}]",
                    rt.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::LdurGen { size, imm9, rn, rt } => {
                if *size != 0b11 {
                    panic!("Need to deal with size since I'm using it now");
                }
                format!(
                    "ldur {}, [{} #{}]",
                    rt.pretty_print(),
                    rn.pretty_print(),
                    imm9
                )
            }
            ArmAsm::LslLslv { sf: _, rm, rn, rd } => {
                format!(
                    "lsl {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::LslUbfm {
                sf: _,
                n: _,
                immr: _,
                imms,
                rn,
                rd,
            } => {
                format!("lsl {}, {}, {}", rd.pretty_print(), rn.pretty_print(), imms)
            }
            ArmAsm::LsrUbfm {
                sf: _,
                n: _,
                immr: _,
                imms: _,
                rn: _,
                rd: _,
            } => {
                unimplemented!("Need to implement LsrUbfm")
            }
            ArmAsm::LsrLsrv {
                sf: _,
                rm: _,
                rn: _,
                rd: _,
            } => {
                unimplemented!("Need to implement LsrUbfm")
            }
            ArmAsm::AsrAsrv { sf: _, rm, rn, rd } => {
                format!(
                    "asr {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::AsrSbfm {
                sf: _,
                n,
                immr: _,
                imms,
                rn,
                rd,
            } => {
                // imms: 0b111111
                if *n != 1 || *imms != 0b111111 {
                    panic!("Need to deal with n and imms since I'm using it now");
                }
                format!("asr {}, {}, {}", rd.pretty_print(), rn.pretty_print(), imms)
            }
            ArmAsm::EorLogShift {
                sf: _,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                // shift: 0,
                // imm6: 0,
                if *shift != 0 || *imm6 != 0 {
                    panic!("Need to deal with shift and imm6 since I'm using it now");
                }
                format!(
                    "eor {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::Madd {
                sf: _,
                rm,
                ra,
                rn,
                rd,
            } => {
                format!(
                    "madd {}, {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    ra.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::MovAddAddsubImm { sf: _, rn, rd } => {
                format!("mov {}, {}", rd.pretty_print(), rn.pretty_print())
            }
            ArmAsm::MovOrrLogShift { sf: _, rm, rd } => {
                format!("mov {}, {}", rd.pretty_print(), rm.pretty_print())
            }
            ArmAsm::Movk {
                sf: _,
                hw,
                imm16,
                rd,
            } => {
                // MOVK  <Wd>, #<imm>{, LSL #<shift>}
                format!("movk {}, {} {{ #{} }}", rd.pretty_print(), imm16, hw)
            }
            ArmAsm::Movz {
                sf: _,
                hw,
                imm16,
                rd,
            } => {
                // hw: 0,
                if *hw != 0 {
                    panic!("Need to deal with hw since I'm using it now");
                }
                format!("movz {}, {}", rd.pretty_print(), imm16)
            }
            ArmAsm::OrrLogShift {
                sf: _,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                // shift: 0,
                // imm6: 0,
                if *shift != 0 || *imm6 != 0 {
                    panic!("Need to deal with shift and imm6 since I'm using it now");
                }
                format!(
                    "orr {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::Ret { rn } => {
                format!("ret {}", rn.pretty_print())
            }
            ArmAsm::Sdiv { sf: _, rm, rn, rd } => {
                format!(
                    "sdiv {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::Stlr { size, rn, rt } => {
                if *size != 0b11 {
                    panic!("Need to deal with size since I'm using it now");
                }
                format!("stlr {}, {}", rt.pretty_print(), rn.pretty_print())
            }
            ArmAsm::StpGen {
                opc,
                imm7,
                rt2,
                rn,
                rt,
                class_selector,
            } => {
                // opc: 0b10,
                // class_selector: StpGenSelector::PreIndex,
                if *opc != 0b10 || *class_selector != StpGenSelector::PreIndex {
                    panic!("Need to deal with opc and class_selector since I'm using it now");
                }
                format!(
                    "stp {}, {}, [{}], #{}",
                    rt.pretty_print(),
                    rt2.pretty_print(),
                    rn.pretty_print(),
                    imm7
                )
            }
            ArmAsm::StrImmGen {
                size,
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => {
                // size: 0b11,
                // imm9: 0, // not used
                // class_selector: StrImmGenSelector::UnsignedOffset,
                if *size != 0b11
                    || *class_selector != StrImmGenSelector::UnsignedOffset
                    || *imm9 != 0
                {
                    panic!(
                        "Need to deal with size and class_selector and imm9 since I'm using it now"
                    );
                }
                format!(
                    "str {}, [{} #{}], ",
                    rt.pretty_print(),
                    rn.pretty_print(),
                    imm12
                )
            }
            ArmAsm::StrRegGen {
                size,
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                // size: 0b11,
                // option: 0b11,
                // s: 0b0,
                if *size != 0b11 || *option != 0b11 || *s != 0 {
                    panic!("Need to deal with size, option and s since I'm using it now");
                }
                format!(
                    "str {}, [{} {}]",
                    rt.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::SturGen { size, imm9, rn, rt } => {
                // size: 0b11,
                // rn: X29,
                if *size != 0b11 {
                    panic!("Need to deal with size and rn since I'm using it now");
                }
                format!(
                    "stur {}, [{} #{}]",
                    rt.pretty_print(),
                    rn.pretty_print(),
                    imm9
                )
            }
            ArmAsm::SubAddsubImm {
                sf: _,
                sh,
                imm12,
                rn,
                rd,
            } => {
                // sh: 0,
                if *sh != 0 {
                    panic!("Need to deal with shift since I'm using it now");
                }
                format!(
                    "sub {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    imm12
                )
            }
            ArmAsm::SubAddsubShift {
                sf: _,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                // shift: 0,
                // imm6: 0,
                if *shift != 0 || *imm6 != 0 {
                    panic!("Need to deal with shift and imm6 since I'm using it now");
                }
                format!(
                    "sub {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::SubsAddsubShift {
                sf: _,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                // shift: 0,
                // imm6: 0,
                if *shift != 0 || *imm6 != 0 {
                    panic!("Need to deal with shift and imm6 since I'm using it now");
                }
                format!(
                    "subs {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::FmovFloat { ftype, rn, rd } => {
                if *ftype != 0b01 {
                    panic!("Need to deal with ftype since I'm using it now");
                }
                format!("fmov {}, {}", rd.pretty_print(), rn.pretty_print())
            }
            ArmAsm::FmovFloatGen {
                sf: _,
                ftype,
                rmode,
                opcode,
                rn,
                rd,
            } => {
                // ftype: 0b01,
                // rmode: 0b00,
                let direction = if *opcode == 0b111 {
                    "FromGeneralToFloat"
                } else if *opcode == 0b110 {
                    "FromFloatToGeneral"
                } else {
                    panic!("Need to deal with opcode since I'm using it now");
                };
                if *ftype != 0b01 || *rmode != 0b00 {
                    panic!("Need to deal with ftype and rmode since I'm using it now");
                }
                // TODO: Fix this to actually change registers
                format!(
                    "fmov {}, {}, {:?}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    direction
                )
            }
            ArmAsm::FaddFloat { ftype, rm, rn, rd } => {
                // ftype: 0b01,
                if *ftype != 0b01 {
                    panic!("Need to deal with ftype since I'm using it now");
                }
                format!(
                    "fadd {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::FsubFloat { ftype, rm, rn, rd } => {
                if *ftype != 0b01 {
                    panic!("Need to deal with ftype since I'm using it now");
                }
                format!(
                    "fsub {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::FmulFloat { ftype, rm, rn, rd } => {
                if *ftype != 0b01 {
                    panic!("Need to deal with ftype since I'm using it now");
                }
                format!(
                    "fmul {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
            ArmAsm::FdivFloat { ftype, rm, rn, rd } => {
                if *ftype != 0b01 {
                    panic!("Need to deal with ftype since I'm using it now");
                }
                format!(
                    "fdiv {}, {}, {}",
                    rd.pretty_print(),
                    rn.pretty_print(),
                    rm.pretty_print()
                )
            }
        }
    }
}

#[allow(unused)]
pub fn draw_lifetimes(lifetimes: &HashMap<VirtualRegister, (usize, usize)>) {
    // Find the maximum lifetime to set the width of the diagram
    let max_lifetime = lifetimes.values().map(|(_, end)| end).max().unwrap_or(&0);
    // sort lifetime by start
    let mut lifetimes: Vec<(VirtualRegister, (usize, usize))> =
        lifetimes.clone().into_iter().collect();
    lifetimes.sort_by_key(|(_, (start, _))| *start);

    for (register, (start, end)) in &lifetimes {
        // Print the register name
        print!("{:10} |", register.index);

        // Print the start of the lifetime
        for _ in 0..*start {
            print!(" ");
        }

        // Print the lifetime
        for _ in *start..*end {
            print!("-");
        }

        // Print the rest of the line
        for _ in *end..*max_lifetime {
            print!(" ");
        }

        println!("|");
    }
}
