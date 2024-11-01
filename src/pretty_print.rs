use std::collections::HashMap;

use crate::ir::{Instruction, Value, VirtualRegister};

pub trait PrettyPrint {
    fn pretty_print(&self) -> String;
}

impl PrettyPrint for Value {
    fn pretty_print(&self) -> String {
        match self {
            Value::Register(register) => register.pretty_print(),
            Value::RawValue(value) => format!("{}", value),
            Value::Pointer(value) => format!("ptr{}", value),
            Value::TaggedConstant(value) => format!("tagged_constant{}", value),
            Value::StringConstantPtr(value) => format!("string_constant_ptr{}", value),
            Value::Local(value) => format!("local{}", value),
            Value::Function(f) => format!("function{}", f),
            Value::FreeVariable(f) => format!("free_variable{}", f),
            Value::True => "true".to_string(),
            Value::False => "false".to_string(),
            Value::Null => "null".to_string(),
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
            Instruction::Breakpoint => {
                "breakpoint".to_string()
            }
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
                    "call {}, {}, {}",
                    value.pretty_print(),
                    value1.pretty_print(),
                    vec.pretty_print()
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
            Instruction::LoadFreeVariable(value, _) => {
                format!("load_free_variable {}, _", value.pretty_print())
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
