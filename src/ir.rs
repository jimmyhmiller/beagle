use std::collections::HashMap;

use crate::arm::FmovDirection;
use crate::machine_code::arm_codegen::{Register, X0, X1};

use crate::types::BuiltInTypes;
use crate::{arm::LowLevelArm, common::Label};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Condition {
    LessThanOrEqual,
    LessThan,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

#[derive(Debug, Copy, Clone)]
pub enum Value {
    Register(VirtualRegister),
    TaggedConstant(isize),
    RawValue(usize),
    // TODO: Think of a better representation
    StringConstantPtr(usize),
    Function(usize),
    Pointer(usize),
    Local(usize),
    FreeVariable(usize),
    True,
    False,
    Null,
}

impl Value {
    fn as_local(&self) -> usize {
        match self {
            Value::Local(local) => *local,
            _ => panic!("Expected local"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VirtualRegister {
    argument: Option<usize>,
    index: usize,
    volatile: bool,
}

impl From<VirtualRegister> for Value {
    fn from(val: VirtualRegister) -> Self {
        Value::Register(val)
    }
}

impl From<usize> for Value {
    fn from(val: usize) -> Self {
        Value::TaggedConstant(val as isize)
    }
}

// Probably don't want to just use rust strings
// could be confusing with lifetimes and such
#[derive(Debug, Clone)]
#[repr(C)]
pub struct StringValue {
    pub str: String,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Sub(Value, Value, Value),
    AddInt(Value, Value, Value),
    Mul(Value, Value, Value),
    Div(Value, Value, Value),
    Assign(VirtualRegister, Value),
    Recurse(Value, Vec<Value>),
    TailRecurse(Value, Vec<Value>),
    JumpIf(Label, Condition, Value, Value),
    Jump(Label),
    Ret(Value),
    Breakpoint,
    Compare(Value, Value, Value, Condition),
    Tag(Value, Value, Value),
    // Do I need these?
    LoadTrue(Value),
    LoadFalse(Value),
    LoadConstant(Value, Value),
    // bool is builtin?
    Call(Value, Value, Vec<Value>, bool),
    HeapLoad(Value, Value, i32),
    HeapLoadReg(Value, Value, Value),
    HeapStore(Value, Value),
    LoadLocal(Value, Value),
    StoreLocal(Value, Value),
    RegisterArgument(Value),
    PushStack(Value),
    PopStack(Value),
    LoadFreeVariable(Value, usize),
    GetStackPointer(Value, Value),
    GetStackPointerImm(Value, isize),
    GetTag(Value, Value),
    Untag(Value, Value),
    HeapStoreOffset(Value, Value, usize),
    HeapStoreByteOffsetMasked(Value, Value, usize, usize, usize),
    CurrentStackPosition(Value),
    ExtendLifeTime(Value),
    HeapStoreOffsetReg(Value, Value, Value),
    AtomicLoad(Value, Value),
    AtomicStore(Value, Value),
    CompareAndSwap(Value, Value, Value),
    StoreFloat(Value, Value, f64),
    // TODO: Move destination register
    // to inside arm instead of here
    GuardInt(Value, Value, Label),
    GuardFloat(Value, Value, Label),
    FmovGeneralToFloat(Value, Value),
    FmovFloatToGeneral(Value, Value),
    AddFloat(Value, Value, Value),
    SubFloat(Value, Value, Value),
    MulFloat(Value, Value, Value),
    DivFloat(Value, Value, Value),
    ShiftRightImm(Value, Value, i32),
    AndImm(Value, Value, u64),
    ShiftLeft(Value, Value, Value),
    ShiftRight(Value, Value, Value),
    ShiftRightZero(Value, Value, Value),
    And(Value, Value, Value),
    Or(Value, Value, Value),
    Xor(Value, Value, Value),
}

impl TryInto<VirtualRegister> for &Value {
    type Error = Value;

    fn try_into(self) -> Result<VirtualRegister, Self::Error> {
        match self {
            Value::Register(register) => Ok(*register),
            _ => Err(*self),
        }
    }
}
impl TryInto<VirtualRegister> for &VirtualRegister {
    type Error = ();

    fn try_into(self) -> Result<VirtualRegister, Self::Error> {
        Ok(*self)
    }
}

impl<T> From<*const T> for Value {
    fn from(val: *const T) -> Self {
        Value::Pointer(val as usize)
    }
}

macro_rules! get_register {
    ($x:expr) => {
        vec![get_registers!($x)].into_iter().flatten().collect()
    };
}
macro_rules! get_registers {
    ($x:expr) => {
        if let Ok(register) = $x.try_into() {
            Some(register)
        } else {
            None
        }
    };
    ($x:expr, $($xs:expr),+)  => {
        vec![get_registers!($x), $(get_registers!($xs)),+].into_iter().flatten().collect()
    };
}

impl Instruction {
    fn get_registers(&self) -> Vec<VirtualRegister> {
        match self {
            Instruction::Sub(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::AddInt(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::Mul(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::Div(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::ShiftLeft(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::ShiftRight(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::ShiftRightZero(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::And(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::Or(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::Xor(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::Assign(a, b) => {
                get_registers!(a, b)
            }
            Instruction::GuardInt(a, b, _) => {
                get_registers!(a, b)
            }
            Instruction::GuardFloat(a, b, _) => {
                get_registers!(a, b)
            }
            Instruction::FmovGeneralToFloat(a, b) => {
                get_registers!(a, b)
            }
            Instruction::FmovFloatToGeneral(a, b) => {
                get_registers!(a, b)
            }
            Instruction::AddFloat(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::SubFloat(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::MulFloat(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::DivFloat(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::ShiftRightImm(a, b, _) => {
                get_registers!(a, b)
            }
            Instruction::AndImm(a, b, _) => {
                get_registers!(a, b)
            }
            Instruction::Recurse(a, args) => {
                let mut result: Vec<VirtualRegister> =
                    args.iter().filter_map(|arg| get_registers!(arg)).collect();
                if let Ok(register) = a.try_into() {
                    result.push(register);
                }
                result
            }
            Instruction::TailRecurse(a, args) => {
                let mut result: Vec<VirtualRegister> =
                    args.iter().filter_map(|arg| get_registers!(arg)).collect();
                if let Ok(register) = a.try_into() {
                    result.push(register);
                }
                result
            }
            Instruction::Call(a, b, args, _) => {
                let mut result: Vec<VirtualRegister> =
                    args.iter().filter_map(|arg| get_registers!(arg)).collect();
                if let Ok(register) = a.try_into() {
                    result.push(register);
                }
                if let Ok(register) = b.try_into() {
                    result.push(register);
                }
                result
            }
            Instruction::LoadConstant(a, b) => {
                get_registers!(a, b)
            }
            Instruction::JumpIf(_, _, a, b) => {
                get_registers!(a, b)
            }
            Instruction::Ret(a) => {
                if let Ok(register) = a.try_into() {
                    vec![register]
                } else {
                    vec![]
                }
            }
            Instruction::Breakpoint => {
                vec![]
            }
            Instruction::Jump(_) => {
                vec![]
            }
            Instruction::Compare(a, b, c, _) => {
                get_registers!(a, b, c)
            }
            Instruction::Tag(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::HeapLoad(a, b, _) => {
                get_registers!(a, b)
            }
            Instruction::AtomicLoad(a, b) => {
                get_registers!(a, b)
            }
            Instruction::AtomicStore(a, b) => {
                get_registers!(a, b)
            }
            Instruction::CompareAndSwap(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::HeapLoadReg(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::HeapStore(a, b) => {
                get_registers!(a, b)
            }
            Instruction::HeapStoreOffset(a, b, _) => {
                get_registers!(a, b)
            }
            Instruction::HeapStoreByteOffsetMasked(a, b, _, _, _) => {
                get_registers!(a, b)
            }
            Instruction::HeapStoreOffsetReg(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::LoadTrue(a) => {
                get_register!(a)
            }
            Instruction::LoadFalse(a) => {
                get_register!(a)
            }
            Instruction::StoreFloat(a, b, _) => {
                get_registers!(a, b)
            }
            Instruction::LoadLocal(a, b) => {
                get_registers!(a, b)
            }
            Instruction::StoreLocal(a, b) => {
                get_registers!(a, b)
            }
            Instruction::RegisterArgument(a) => {
                get_register!(a)
            }
            Instruction::PushStack(a) => {
                get_register!(a)
            }
            Instruction::PopStack(a) => {
                get_register!(a)
            }
            Instruction::LoadFreeVariable(a, _) => {
                get_register!(a)
            }
            Instruction::GetStackPointer(a, b) => {
                get_registers!(a, b)
            }
            Instruction::CurrentStackPosition(a) => {
                get_register!(a)
            }
            Instruction::GetStackPointerImm(a, _) => {
                get_register!(a)
            }
            Instruction::GetTag(a, b) => {
                get_registers!(a, b)
            }
            Instruction::Untag(a, b) => {
                get_registers!(a, b)
            }
            Instruction::ExtendLifeTime(a) => {
                get_register!(a)
            }
        }
    }
}

struct RegisterAllocator {
    lifetimes: HashMap<VirtualRegister, (usize, usize)>,
    allocated_registers: HashMap<VirtualRegister, Register>,
}

impl RegisterAllocator {
    fn new(lifetimes: HashMap<VirtualRegister, (usize, usize)>) -> Self {
        Self {
            lifetimes,
            allocated_registers: HashMap::new(),
        }
    }

    fn allocate_register(
        &mut self,
        index: usize,
        register: VirtualRegister,
        lang: &mut LowLevelArm,
    ) -> Register {
        let (start, _end) = self.lifetimes.get(&register).unwrap();
        if index == *start {
            // Is it okay that the register is already allocated for the argument?
            if let Some(arg) = register.argument {
                let reg = lang.arg(arg as u8);
                self.allocated_registers.insert(register, reg);
                lang.reserve_register(reg);
                reg
            } else {
                assert!(!self.allocated_registers.contains_key(&register));
                let reg = lang.volatile_register();
                self.allocated_registers.insert(register, reg);
                reg
            }
        } else {
            assert!(self.allocated_registers.contains_key(&register));
            *self.allocated_registers.get(&register).unwrap()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Ir {
    register_index: usize,
    pub instructions: Vec<Instruction>,
    labels: Vec<Label>,
    label_names: Vec<String>,
    label_locations: HashMap<usize, usize>,
    pub num_locals: usize,
    compiler_pointer: usize,
    allocate_fn_pointer: usize,
    after_return: Label,
}

impl Ir {
    pub fn new(compiler_pointer: usize, allocate_fn_pointer: usize) -> Self {
        let mut me = Self {
            register_index: 0,
            instructions: vec![],
            labels: vec![],
            label_names: vec![],
            label_locations: HashMap::new(),
            num_locals: 0,
            compiler_pointer,
            allocate_fn_pointer,
            after_return: Label { index: 0 },
        };

        me.insert_label("after_return", me.after_return);
        me
    }

    fn next_register(&mut self, argument: Option<usize>, volatile: bool) -> VirtualRegister {
        let register = VirtualRegister {
            argument,
            index: self.register_index,
            volatile,
        };
        self.register_index += 1;
        register
    }

    pub fn arg(&mut self, n: usize) -> VirtualRegister {
        self.next_register(Some(n), true)
    }

    pub fn volatile_register(&mut self) -> VirtualRegister {
        self.next_register(None, true)
    }

    pub fn recurse<A>(&mut self, args: Vec<A>) -> Value
    where
        A: Into<Value>,
    {
        let register = self.volatile_register();
        let mut new_args: Vec<Value> = vec![];
        for arg in args.into_iter() {
            let value: Value = arg.into();
            let reg = self.assign_new(value);
            new_args.push(reg.into());
        }
        self.instructions
            .push(Instruction::Recurse(register.into(), new_args));
        Value::Register(register)
    }

    pub fn tail_recurse<A>(&mut self, args: Vec<A>) -> Value
    where
        A: Into<Value>,
    {
        let register = self.volatile_register();
        let mut new_args: Vec<Value> = vec![];
        for arg in args.into_iter() {
            let value: Value = arg.into();
            let reg = self.assign_new(value);
            new_args.push(reg.into());
        }
        self.instructions
            .push(Instruction::TailRecurse(register.into(), new_args));
        Value::Register(register)
    }

    pub fn sub_any<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        self.math_any(a, b, Self::sub_int::<Value, Value>, Self::sub_float)
    }

    pub fn sub_int<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let result = self.volatile_register();
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        self.instructions
            .push(Instruction::Sub(result.into(), a.into(), b.into()));
        Value::Register(result)
    }

    pub fn math_any<A, B, F, G>(&mut self, a: A, b: B, op_int: F, op_float: G) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
        F: FnOnce(&mut Ir, Value, Value) -> Value,
        G: FnOnce(&mut Ir, Value, Value) -> Value,
    {
        let result_register = self.volatile_register();
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        let add_float = self.label("add_float");
        let after_add = self.label("after_add");
        // self.breakpoint();
        self.guard_int(a.into(), add_float);
        self.guard_int(b.into(), add_float);
        let result = op_int(self, a.into(), b.into());
        self.assign(result_register, result);
        self.jump(after_add);
        self.write_label(add_float);

        self.guard_float(a.into(), self.after_return);
        self.guard_float(b.into(), self.after_return);
        let a = self.untag(a.into());
        let b = self.untag(b.into());
        let a = self.load_from_heap(a, 1);
        let b = self.load_from_heap(b, 1);
        let a = self.fmov_general_to_float(a);
        let b = self.fmov_general_to_float(b);
        let result = op_float(self, a, b);
        let result = self.fmov_float_to_general(result);
        // Allocate and store
        let size_reg = self.assign_new(1);
        let float_pointer = self.allocate(size_reg.into());
        let float_pointer = self.untag(float_pointer);
        self.write_small_object_header(float_pointer);
        self.heap_store_offset(float_pointer, result, 1);
        let tagged = self.tag(float_pointer, BuiltInTypes::Float.get_tag());
        self.assign(result_register, tagged);

        self.write_label(after_add);
        Value::Register(result_register)
    }

    pub fn add_any<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        self.math_any(a, b, Self::add_int::<Value, Value>, Self::add_float)
    }

    pub fn add_int<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let register = self.volatile_register();
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        self.instructions
            .push(Instruction::AddInt(register.into(), a.into(), b.into()));
        Value::Register(register)
    }

    pub fn mul<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let register = self.volatile_register();
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        self.instructions
            .push(Instruction::Mul(register.into(), a.into(), b.into()));
        Value::Register(register)
    }

    pub fn mul_any<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        self.math_any(a, b, Self::mul, Self::mul_float)
    }

    pub fn div<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let register = self.volatile_register();
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        self.instructions
            .push(Instruction::Div(register.into(), a.into(), b.into()));
        Value::Register(register)
    }

    pub fn div_any<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        self.math_any(a, b, Self::div, Self::div_float)
    }

    pub fn compare(&mut self, a: Value, b: Value, condition: Condition) -> Value {
        let register = self.volatile_register();
        let a = self.assign_new(a);
        let b = self.assign_new(b);
        let tag = self.assign_new(Value::RawValue(BuiltInTypes::Bool.get_tag() as usize));
        self.instructions.push(Instruction::Compare(
            register.into(),
            a.into(),
            b.into(),
            condition,
        ));
        self.instructions.push(Instruction::Tag(
            register.into(),
            register.into(),
            tag.into(),
        ));
        Value::Register(register)
    }

    pub fn jump_if<A, B>(&mut self, label: Label, condition: Condition, a: A, b: B)
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        self.instructions
            .push(Instruction::JumpIf(label, condition, a.into(), b.into()));
    }

    pub fn shift_right_imm(&mut self, a: Value, b: i32) -> Value {
        let a = self.assign_new(a);
        let destination = self.volatile_register();
        self.instructions
            .push(Instruction::ShiftRightImm(destination.into(), a.into(), b));
        destination.into()
    }

    pub fn and_imm(&mut self, a: Value, b: u64) -> Value {
        let a = self.assign_new(a);
        let destination = self.volatile_register();
        self.instructions
            .push(Instruction::AndImm(destination.into(), a.into(), b));
        destination.into()
    }

    pub fn shift_left(&mut self, a: Value, b: Value) -> Value {
        let a = self.assign_new(a);
        let b = self.assign_new(b);
        let destination = self.volatile_register();
        self.instructions.push(Instruction::ShiftLeft(
            destination.into(),
            a.into(),
            b.into(),
        ));
        destination.into()
    }

    pub fn shift_right(&mut self, a: Value, b: Value) -> Value {
        let a = self.assign_new(a);
        let b = self.assign_new(b);
        let destination = self.volatile_register();
        self.instructions.push(Instruction::ShiftRight(
            destination.into(),
            a.into(),
            b.into(),
        ));
        destination.into()
    }

    pub fn shift_right_zero(&mut self, a: Value, b: Value) -> Value {
        let a = self.assign_new(a);
        let b = self.assign_new(b);
        let destination = self.volatile_register();
        self.instructions.push(Instruction::ShiftRightZero(
            destination.into(),
            a.into(),
            b.into(),
        ));
        destination.into()
    }

    pub fn bitwise_and(&mut self, a: Value, b: Value) -> Value {
        let a = self.assign_new(a);
        let b = self.assign_new(b);
        let destination = self.volatile_register();
        self.instructions
            .push(Instruction::And(destination.into(), a.into(), b.into()));
        destination.into()
    }

    pub fn bitwise_or(&mut self, a: Value, b: Value) -> Value {
        let a = self.assign_new(a);
        let b = self.assign_new(b);
        let destination = self.volatile_register();
        self.instructions
            .push(Instruction::Or(destination.into(), a.into(), b.into()));
        destination.into()
    }

    pub fn bitwise_xor(&mut self, a: Value, b: Value) -> Value {
        let a = self.assign_new(a);
        let b = self.assign_new(b);
        let destination = self.volatile_register();
        self.instructions
            .push(Instruction::Xor(destination.into(), a.into(), b.into()));
        destination.into()
    }

    pub fn assign<A>(&mut self, dest: VirtualRegister, val: A)
    where
        A: Into<Value>,
    {
        self.instructions
            .push(Instruction::Assign(dest, val.into()));
    }

    pub fn assign_new_force<A>(&mut self, val: A) -> VirtualRegister
    where
        A: Into<Value>,
    {
        // We want to always get a new register.
        // This is useful if the register we are passing will be reassigned
        // like it is for atomics
        let val = val.into();
        let register = self.next_register(None, false);
        self.instructions.push(Instruction::Assign(register, val));
        register
    }

    pub fn assign_new<A>(&mut self, val: A) -> VirtualRegister
    where
        A: Into<Value>,
    {
        let val = val.into();
        if let Value::Register(register) = val {
            return register;
        }
        let register = self.next_register(None, false);
        self.instructions.push(Instruction::Assign(register, val));
        register
    }

    pub fn ret<A>(&mut self, n: A) -> Value
    where
        A: Into<Value>,
    {
        let val = n.into();
        self.instructions.push(Instruction::Ret(val));
        val
    }

    pub fn label(&mut self, arg: &str) -> Label {
        let label_index = self.labels.len();
        self.label_names.push(arg.to_string());
        let label = Label { index: label_index };
        self.labels.push(label);
        label
    }

    pub fn write_label(&mut self, early_exit: Label) {
        assert!(!self.label_locations.contains_key(&self.instructions.len()));
        self.label_locations
            .insert(self.instructions.len(), early_exit.index);
    }

    fn get_register_lifetime(&mut self) -> HashMap<VirtualRegister, (usize, usize)> {
        let mut result: HashMap<VirtualRegister, (usize, usize)> = HashMap::new();
        for (index, instruction) in self.instructions.iter().enumerate().rev() {
            for register in instruction.get_registers() {
                if let Some((_start, end)) = result.get(&register) {
                    result.insert(register, (index, *end));
                } else {
                    result.insert(register, (index, index));
                }
            }
        }

        result
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

    pub fn compile(
        &mut self,
        mut lang: LowLevelArm,
        error_fn_pointer: usize,
        compiler_ptr: usize,
    ) -> LowLevelArm {
        // println!("{:#?}", self.instructions);
        lang.set_max_locals(self.num_locals);
        // lang.breakpoint();

        let before_prelude = lang.new_label("before_prelude");
        lang.write_label(before_prelude);

        // zero is a placeholder because this will be patched
        lang.prelude(0);

        // I believe this is fine because it is volatile and we
        // are at the beginning of the function
        let register = lang.canonical_volatile_registers[0];
        lang.mov_64(register, BuiltInTypes::null_value());
        lang.set_all_locals_to_null(register);

        let after_prelude = lang.new_label("after_prelude");
        lang.write_label(after_prelude);

        let exit = lang.new_label("exit");

        self.compile_instructions(&mut lang, exit, before_prelude, after_prelude);

        lang.write_label(exit);
        // Zero is a placeholder because this will be patched
        lang.epilogue(0);
        lang.ret();
        // TODO: ugly
        let lang_after_return = lang.get_label_by_name("after_return");
        lang.write_label(lang_after_return);
        let register = lang.canonical_volatile_registers[0];
        lang.mov_64(register, error_fn_pointer as isize);
        lang.mov_64(X0, compiler_ptr as isize);
        lang.get_stack_pointer_imm(X1, 0);
        lang.call(register);

        lang
    }

    fn compile_instructions(
        &mut self,
        lang: &mut LowLevelArm,
        exit: Label,
        before_prelude: Label,
        after_prelude: Label,
    ) {
        let mut ir_label_to_lang_label: HashMap<Label, Label> = HashMap::new();
        let mut labels: Vec<&Label> = self.labels.iter().collect();
        labels.sort_by_key(|label| label.index);
        for label in labels.iter() {
            let new_label = lang.new_label(&self.label_names[label.index]);
            ir_label_to_lang_label.insert(**label, new_label);
        }
        let lifetimes = self.get_register_lifetime();
        // println!("compiling {}", name);
        // Self::draw_lifetimes(&lifetimes);
        let mut alloc = RegisterAllocator::new(lifetimes);

        let mut lifetimes2: Vec<(VirtualRegister, (usize, usize))> =
            alloc.lifetimes.iter().map(|(r, v)| (*r, *v)).collect();
        lifetimes2.sort_by_key(|(_, (start, _))| *start);
        for (index, instruction) in self.instructions.iter().enumerate() {
            for (register, (_start, end)) in lifetimes2.iter() {
                if index == end + 1 {
                    if let Some(register) = alloc.allocated_registers.get(register) {
                        lang.free_register(*register);
                    }
                }
            }
            let label = self.label_locations.get(&index);
            if let Some(label) = label {
                lang.write_label(ir_label_to_lang_label[&self.labels[*label]]);
            }
            // println!("instruction {:?}", instruction);
            match instruction {
                Instruction::Breakpoint => {
                    lang.breakpoint();
                }
                Instruction::ExtendLifeTime(_) => {}
                Instruction::Sub(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);

                    lang.guard_integer(dest, a, self.after_return);
                    lang.guard_integer(dest, b, self.after_return);

                    lang.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    lang.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    lang.sub(dest, a, b);
                    lang.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    lang.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    lang.shift_left_imm(b, b, BuiltInTypes::tag_size());
                }
                Instruction::AddInt(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);

                    lang.add(dest, a, b);
                }
                Instruction::Mul(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);

                    // lang.breakpoint();
                    lang.guard_integer(dest, a, self.after_return);
                    lang.guard_integer(dest, b, self.after_return);

                    lang.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    lang.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    lang.mul(dest, a, b);
                    lang.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    lang.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    lang.shift_left_imm(b, b, BuiltInTypes::tag_size());
                }
                Instruction::Div(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);

                    lang.guard_integer(dest, a, self.after_return);
                    lang.guard_integer(dest, b, self.after_return);

                    lang.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    lang.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    lang.div(dest, a, b);
                    lang.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    lang.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    lang.shift_left_imm(b, b, BuiltInTypes::tag_size());
                }
                Instruction::ShiftRightImm(dest, value, shift) => {
                    let value = value.try_into().unwrap();
                    let value = alloc.allocate_register(index, value, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);

                    lang.guard_integer(dest, value, self.after_return);

                    lang.shift_right_imm(dest, value, *shift);
                }
                Instruction::AndImm(dest, value, imm) => {
                    let value = value.try_into().unwrap();
                    let value = alloc.allocate_register(index, value, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.and_imm(dest, value, *imm);
                }
                Instruction::ShiftLeft(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);

                    lang.guard_integer(dest, a, self.after_return);
                    lang.guard_integer(dest, b, self.after_return);

                    lang.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    lang.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    lang.shift_left(dest, a, b);
                    lang.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    lang.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    lang.shift_left_imm(b, b, BuiltInTypes::tag_size());
                }
                Instruction::ShiftRight(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);

                    lang.guard_integer(dest, a, self.after_return);
                    lang.guard_integer(dest, b, self.after_return);

                    lang.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    lang.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    lang.shift_right(dest, a, b);
                    lang.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    lang.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    lang.shift_left_imm(b, b, BuiltInTypes::tag_size());
                }
                Instruction::ShiftRightZero(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);

                    lang.guard_integer(dest, a, self.after_return);
                    lang.guard_integer(dest, b, self.after_return);

                    lang.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    lang.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    lang.and_imm(a, a, 0xFFFFFFFF);
                    lang.shift_right_zero(dest, a, b);
                    lang.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    lang.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    lang.shift_left_imm(b, b, BuiltInTypes::tag_size());
                }
                Instruction::And(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.and(dest, a, b);
                }
                Instruction::Or(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.or(dest, a, b);
                }
                Instruction::Xor(dest, a, b) => {
                    let a: VirtualRegister = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.xor(dest, a, b);
                }
                Instruction::GuardInt(dest, value, label) => {
                    let value = value.try_into().unwrap();
                    let value = alloc.allocate_register(index, value, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.guard_integer(dest, value, ir_label_to_lang_label[label]);
                }
                Instruction::GuardFloat(dest, value, label) => {
                    let value = value.try_into().unwrap();
                    let value = alloc.allocate_register(index, value, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.guard_float(dest, value, ir_label_to_lang_label[label]);
                }
                Instruction::FmovGeneralToFloat(dest, src) => {
                    let src = src.try_into().unwrap();
                    let src = alloc.allocate_register(index, src, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.fmov(dest, src, FmovDirection::FromGeneralToFloat);
                }
                Instruction::FmovFloatToGeneral(dest, src) => {
                    let src = src.try_into().unwrap();
                    let src = alloc.allocate_register(index, src, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.fmov(dest, src, FmovDirection::FromFloatToGeneral);
                }
                Instruction::AddFloat(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.fadd(dest, a, b);
                }
                Instruction::SubFloat(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.fsub(dest, a, b);
                }
                Instruction::MulFloat(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.fmul(dest, a, b);
                }
                Instruction::DivFloat(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.fdiv(dest, a, b);
                }
                Instruction::Assign(dest, val) => match val {
                    Value::Register(virt_reg) => {
                        let register = alloc.allocate_register(index, *virt_reg, lang);
                        let dest = alloc.allocate_register(index, *dest, lang);
                        lang.mov_reg(dest, register);
                    }
                    Value::TaggedConstant(i) => {
                        let register = alloc.allocate_register(index, *dest, lang);
                        let tagged = BuiltInTypes::construct_int(*i);
                        lang.mov_64(register, tagged);
                    }
                    Value::StringConstantPtr(ptr) => {
                        let register = alloc.allocate_register(index, *dest, lang);
                        let tagged = BuiltInTypes::String.tag(*ptr as isize);
                        lang.mov_64(register, tagged);
                    }
                    Value::Function(id) => {
                        let register = alloc.allocate_register(index, *dest, lang);
                        let function = BuiltInTypes::Function.tag(*id as isize);
                        lang.mov_64(register, function);
                    }
                    Value::Pointer(ptr) => {
                        let register = alloc.allocate_register(index, *dest, lang);
                        lang.mov_64(register, *ptr as isize);
                    }
                    Value::RawValue(value) => {
                        let register = alloc.allocate_register(index, *dest, lang);
                        lang.mov_64(register, *value as isize);
                    }
                    Value::True => {
                        let register = alloc.allocate_register(index, *dest, lang);
                        lang.mov_64(register, BuiltInTypes::construct_boolean(true));
                    }
                    Value::False => {
                        let register = alloc.allocate_register(index, *dest, lang);
                        lang.mov_64(register, BuiltInTypes::construct_boolean(false));
                    }
                    Value::Local(local) => {
                        let register = alloc.allocate_register(index, *dest, lang);
                        lang.load_local(register, *local as i32)
                    }
                    Value::FreeVariable(free_variable) => {
                        let register = alloc.allocate_register(index, *dest, lang);
                        // The idea here is that I would store free variables after the locals on the stack
                        // Need to make sure I preserve that space
                        // and that at this point in the program I know how many locals there are.
                        lang.load_from_stack(
                            register,
                            -((*free_variable + self.num_locals + 1) as i32),
                        );
                    }
                    Value::Null => {
                        let register = alloc.allocate_register(index, *dest, lang);
                        lang.mov_64(register, 0b111_isize);
                    }
                },
                Instruction::LoadConstant(dest, val) => {
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.mov_reg(dest, val);
                }
                Instruction::LoadLocal(dest, local) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    let local = local.as_local();
                    lang.load_local(dest, local as i32);
                }
                Instruction::StoreLocal(dest, value) => {
                    let value = value.try_into().unwrap();
                    let value = alloc.allocate_register(index, value, lang);
                    lang.store_local(value, dest.as_local() as i32);
                }
                Instruction::LoadTrue(dest) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.mov_64(dest, BuiltInTypes::construct_boolean(true));
                }
                Instruction::LoadFalse(dest) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.mov_64(dest, BuiltInTypes::construct_boolean(false));
                }
                Instruction::StoreFloat(dest, temp, value) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    let temp = temp.try_into().unwrap();
                    let temp = alloc.allocate_register(index, temp, lang);
                    lang.mov_64(temp, value.to_bits() as isize);
                    // The header is the first field, so offset is 1
                    lang.store_on_heap(dest, temp, 1)
                }
                Instruction::Recurse(dest, args) => {
                    // TODO: Clean up duplication
                    let mut out_live_call_registers = vec![];
                    for (register, (start, end)) in alloc.lifetimes.iter() {
                        if *end < index {
                            continue;
                        }
                        if *start > index {
                            continue;
                        }
                        if index != *end {
                            if let Some(register) = alloc.allocated_registers.get(register) {
                                out_live_call_registers.push(*register);
                            }
                        }
                    }

                    for register in out_live_call_registers.iter() {
                        lang.push_to_stack(*register);
                    }
                    for (index, arg) in args.iter().enumerate().rev() {
                        let arg = arg.try_into().unwrap();
                        let arg = alloc.allocate_register(index, arg, lang);
                        lang.mov_reg(lang.arg(index as u8), arg);
                    }
                    lang.recurse(before_prelude);
                    let dest = dest.try_into().unwrap();
                    let register = alloc.allocate_register(index, dest, lang);
                    lang.mov_reg(register, lang.ret_reg());
                    for (index, register) in out_live_call_registers.iter().enumerate() {
                        lang.pop_from_stack_indexed(*register, index as i32);
                    }
                }
                Instruction::TailRecurse(dest, args) => {
                    for (index, arg) in args.iter().enumerate().rev() {
                        let arg = arg.try_into().unwrap();
                        let arg = alloc.allocate_register(index, arg, lang);
                        lang.mov_reg(lang.arg(index as u8), arg);
                    }
                    lang.jump(after_prelude);
                    let dest = dest.try_into().unwrap();
                    let register = alloc.allocate_register(index, dest, lang);
                    lang.mov_reg(register, lang.ret_reg());
                }
                Instruction::Call(dest, function, args, builtin) => {
                    // TODO: Clean up duplication
                    let mut out_live_call_registers = vec![];
                    for (register, (start, end)) in alloc.lifetimes.iter() {
                        if *end < index {
                            continue;
                        }
                        if *start > index {
                            continue;
                        }
                        if index != *end {
                            if let Some(register) = alloc.allocated_registers.get(register) {
                                out_live_call_registers.push(*register);
                            }
                        }
                    }

                    // I only need to store on stack those things that live past the call
                    // I think this is part of the reason why I have too many registers live at a time
                    for register in out_live_call_registers.iter() {
                        lang.push_to_stack(*register);
                    }
                    for (arg_index, arg) in args.iter().enumerate().rev() {
                        let arg = arg.try_into().unwrap();
                        let arg = alloc.allocate_register(index, arg, lang);
                        lang.mov_reg(lang.arg(arg_index as u8), arg);
                    }
                    // TODO: I am not actually checking any tags here
                    // or unmasking or anything. Just straight up calling it
                    let function =
                        alloc.allocate_register(index, function.try_into().unwrap(), lang);
                    lang.shift_right_imm(function, function, BuiltInTypes::tag_size());
                    if *builtin {
                        lang.call_builtin(function);
                    } else {
                        lang.call(function);
                    }

                    let dest = dest.try_into().unwrap();
                    let register = alloc.allocate_register(index, dest, lang);
                    lang.mov_reg(register, lang.ret_reg());
                    for register in out_live_call_registers.iter().rev() {
                        lang.pop_from_stack(*register);
                    }
                }
                Instruction::Compare(dest, a, b, condition) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.compare_bool(*condition, dest, a, b);
                }
                Instruction::Tag(destination, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let dest = destination.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.tag_value(dest, a, b);
                }
                Instruction::JumpIf(label, condition, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, lang);
                    let label = ir_label_to_lang_label.get(label).unwrap();
                    lang.compare(a, b);
                    match condition {
                        Condition::LessThanOrEqual => lang.jump_less_or_equal(*label),
                        Condition::LessThan => lang.jump_less(*label),
                        Condition::Equal => lang.jump_equal(*label),
                        Condition::NotEqual => lang.jump_not_equal(*label),
                        Condition::GreaterThan => lang.jump_greater(*label),
                        Condition::GreaterThanOrEqual => lang.jump_greater_or_equal(*label),
                    }
                }
                Instruction::Jump(label) => {
                    let label = ir_label_to_lang_label.get(label).unwrap();
                    lang.jump(*label);
                }
                Instruction::Ret(value) => match value {
                    Value::Register(virt_reg) => {
                        let register = alloc.allocate_register(index, *virt_reg, lang);
                        if register == lang.ret_reg() {
                            lang.jump(exit);
                        } else {
                            lang.mov_reg(lang.ret_reg(), register);
                            lang.jump(exit);
                        }
                    }
                    Value::TaggedConstant(i) => {
                        lang.mov_64(lang.ret_reg(), BuiltInTypes::construct_int(*i));
                        lang.jump(exit);
                    }
                    Value::StringConstantPtr(ptr) => {
                        lang.mov_64(lang.ret_reg(), *ptr as isize);
                        lang.jump(exit);
                    }
                    Value::Function(id) => {
                        lang.mov_64(lang.ret_reg(), *id as isize);
                        lang.jump(exit);
                    }
                    Value::Pointer(ptr) => {
                        lang.mov_64(lang.ret_reg(), *ptr as isize);
                        lang.jump(exit);
                    }
                    Value::True => {
                        lang.mov_64(lang.ret_reg(), BuiltInTypes::construct_boolean(true));
                        lang.jump(exit);
                    }
                    Value::False => {
                        lang.mov_64(lang.ret_reg(), BuiltInTypes::construct_boolean(false));
                        lang.jump(exit);
                    }
                    Value::RawValue(_) => {
                        panic!("Should we be returing a raw value?")
                    }
                    Value::Null => {
                        lang.mov_64(lang.ret_reg(), 0b111);
                        lang.jump(exit);
                    }
                    Value::Local(local) => {
                        lang.load_local(lang.ret_reg(), *local as i32);
                        lang.jump(exit);
                    }
                    Value::FreeVariable(free_variable) => {
                        lang.load_from_stack(
                            lang.ret_reg(),
                            -((*free_variable + self.num_locals + 1) as i32),
                        );
                        lang.jump(exit);
                    }
                },
                Instruction::HeapLoad(dest, ptr, offset) => {
                    let ptr = ptr.try_into().unwrap();
                    let ptr = alloc.allocate_register(index, ptr, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.load_from_heap(dest, ptr, *offset);
                }
                Instruction::AtomicLoad(dest, ptr) => {
                    let ptr = ptr.try_into().unwrap();
                    let ptr = alloc.allocate_register(index, ptr, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.atomic_load(dest, ptr);
                }
                Instruction::AtomicStore(ptr, val) => {
                    let ptr = ptr.try_into().unwrap();
                    let ptr = alloc.allocate_register(index, ptr, lang);
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, lang);
                    lang.atomic_store(ptr, val);
                }
                Instruction::CompareAndSwap(dest, ptr, val) => {
                    let ptr = ptr.try_into().unwrap();
                    let ptr = alloc.allocate_register(index, ptr, lang);
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.compare_and_swap(dest, ptr, val);
                }
                Instruction::HeapLoadReg(dest, ptr, offset) => {
                    let ptr = ptr.try_into().unwrap();
                    let ptr = alloc.allocate_register(index, ptr, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    let offset = offset.try_into().unwrap();
                    let offset = alloc.allocate_register(index, offset, lang);
                    lang.load_from_heap_with_reg_offset(dest, ptr, offset);
                }
                Instruction::HeapStore(ptr, val) => {
                    let ptr = ptr.try_into().unwrap();
                    let ptr = alloc.allocate_register(index, ptr, lang);
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, lang);
                    lang.store_on_heap(ptr, val, 0);
                }

                Instruction::HeapStoreOffset(ptr, val, offset) => {
                    let ptr = ptr.try_into().unwrap();
                    let ptr = alloc.allocate_register(index, ptr, lang);
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, lang);
                    lang.store_on_heap(ptr, val, *offset as i32);
                }
                Instruction::HeapStoreByteOffsetMasked(ptr, val, offset, byte_offset, mask) => {
                    // We are trying to write to a specific byte in a word
                    // We need to load the word, mask out the byte, or in the new value
                    // and then store it back
                    let ptr = ptr.try_into().unwrap();
                    let ptr = alloc.allocate_register(index, ptr, lang);
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, lang);
                    let dest = lang.volatile_register();
                    // lang.breakpoint();
                    lang.load_from_heap(dest, ptr, *offset as i32);
                    let mask_register = lang.volatile_register();
                    lang.mov_64(mask_register, *mask as isize);
                    lang.and(dest, dest, mask_register);
                    lang.free_register(mask_register);
                    lang.shift_left_imm(val, val, (byte_offset * 8) as i32);
                    lang.or(dest, dest, val);
                    lang.store_on_heap(ptr, dest, *offset as i32);
                    lang.free_register(dest);
                }
                Instruction::HeapStoreOffsetReg(ptr, val, offset) => {
                    let ptr = ptr.try_into().unwrap();
                    let ptr = alloc.allocate_register(index, ptr, lang);
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, lang);
                    let offset = offset.try_into().unwrap();
                    let offset = alloc.allocate_register(index, offset, lang);
                    lang.store_to_heap_with_reg_offset(ptr, val, offset);
                }
                Instruction::RegisterArgument(arg) => {
                    // This doesn't actually compile into any code
                    // it is here to say the argument is live from the beginning

                    let arg = arg.try_into().unwrap();
                    alloc.allocate_register(index, arg, lang);
                }
                Instruction::PushStack(val) => {
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, lang);
                    lang.push_to_stack(val);
                }
                Instruction::PopStack(val) => {
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, lang);
                    lang.pop_from_stack(val);
                }
                Instruction::LoadFreeVariable(dest, free_variable) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.load_from_stack(dest, (*free_variable + self.num_locals) as i32);
                }
                Instruction::GetStackPointer(dest, offset) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    let offset = offset.try_into().unwrap();
                    let offset = alloc.allocate_register(index, offset, lang);
                    lang.get_stack_pointer(dest, offset);
                }
                Instruction::GetStackPointerImm(dest, offset) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.get_stack_pointer_imm(dest, *offset);
                }
                Instruction::CurrentStackPosition(dest) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.get_current_stack_position(dest)
                }
                Instruction::GetTag(dest, value) => {
                    let value = value.try_into().unwrap();
                    let value = alloc.allocate_register(index, value, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.get_tag(dest, value);
                }
                Instruction::Untag(dest, value) => {
                    let value = value.try_into().unwrap();
                    let value = alloc.allocate_register(index, value, lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, lang);
                    lang.shift_right_imm(dest, value, BuiltInTypes::tag_size());
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn breakpoint(&mut self) {
        self.instructions.push(Instruction::Breakpoint);
    }

    pub fn jump(&mut self, label: Label) {
        self.instructions.push(Instruction::Jump(label));
    }

    pub fn load_string_constant(&mut self, string_constant: Value) -> Value {
        let string_constant = self.assign_new(string_constant);
        let register = self.volatile_register();
        self.instructions.push(Instruction::LoadConstant(
            register.into(),
            string_constant.into(),
        ));
        register.into()
    }

    pub fn heap_store<A, B>(&mut self, dest: A, source: B)
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let source = self.assign_new(source.into());
        let dest = self.assign_new(dest.into());
        self.instructions
            .push(Instruction::HeapStore(dest.into(), source.into()));
    }

    pub fn heap_store_offset<A, B>(&mut self, dest: A, source: B, offset: usize)
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let source = self.assign_new(source.into());
        let dest = self.assign_new(dest.into());
        self.instructions.push(Instruction::HeapStoreOffset(
            dest.into(),
            source.into(),
            offset,
        ));
    }

    pub fn heap_store_byte_offset_masked<A, B>(
        &mut self,
        dest: A,
        value: B,
        offset: usize,
        byte_offset: usize,
        mask: usize,
    ) where
        A: Into<Value>,
        B: Into<Value>,
    {
        let source = self.assign_new(value.into());
        let dest = self.assign_new(dest.into());
        self.instructions
            .push(Instruction::HeapStoreByteOffsetMasked(
                dest.into(),
                source.into(),
                offset,
                byte_offset,
                mask,
            ));
    }

    pub fn heap_load(&mut self, dest: Value, source: Value) -> Value {
        let source = self.assign_new(source);
        let dest = self.assign_new(dest);
        self.instructions
            .push(Instruction::HeapLoad(dest.into(), source.into(), 0));
        dest.into()
    }

    pub fn atomic_load(&mut self, dest: Value, source: Value) -> Value {
        let source = self.assign_new(source);
        let dest = self.assign_new(dest);
        self.instructions
            .push(Instruction::AtomicLoad(dest.into(), source.into()));
        dest.into()
    }
    pub fn atomic_store(&mut self, dest: Value, source: Value) {
        let source = self.assign_new(source);
        let dest = self.assign_new(dest);
        self.instructions
            .push(Instruction::AtomicStore(dest.into(), source.into()));
    }

    pub fn compare_and_swap(&mut self, expected: Value, new: Value, pointer: Value) {
        let expected = self.assign_new(expected);
        let new = self.assign_new(new);
        let pointer = self.assign_new(pointer);
        self.instructions.push(Instruction::CompareAndSwap(
            expected.into(),
            new.into(),
            pointer.into(),
        ));
    }

    pub fn heap_load_with_reg_offset(&mut self, source: Value, offset: Value) -> Value {
        let dest = self.volatile_register();
        let source = self.assign_new(source);
        let offset = self.assign_new(offset);
        self.instructions.push(Instruction::HeapLoadReg(
            dest.into(),
            source.into(),
            offset.into(),
        ));
        dest.into()
    }

    pub fn function(&mut self, function_index: Value) -> Value {
        let function = self.assign_new(function_index);
        function.into()
    }

    pub fn call(&mut self, function: Value, vec: Vec<Value>) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::Call(dest, function, vec, false));
        dest
    }

    pub fn call_builtin(&mut self, function: Value, vec: Vec<Value>) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::Call(dest, function, vec, true));
        dest
    }

    pub fn store_local(&mut self, local_index: usize, reg: VirtualRegister) {
        self.increment_locals(local_index);
        self.instructions.push(Instruction::StoreLocal(
            Value::Local(local_index),
            reg.into(),
        ));
    }

    pub fn load_local(&mut self, reg: VirtualRegister, local_index: usize) -> Value {
        self.increment_locals(local_index);
        self.instructions.push(Instruction::LoadLocal(
            reg.into(),
            Value::Local(local_index),
        ));
        reg.into()
    }

    pub fn push_to_stack(&mut self, reg: Value) {
        self.instructions.push(Instruction::PushStack(reg));
    }

    pub fn pop_from_stack(&mut self) -> Value {
        let reg = self.volatile_register().into();
        self.instructions.push(Instruction::PopStack(reg));
        reg
    }

    fn increment_locals(&mut self, index: usize) {
        if index >= self.num_locals {
            self.num_locals = index + 1;
        }
    }

    pub fn register_argument(&mut self, reg: VirtualRegister) {
        self.instructions
            .push(Instruction::RegisterArgument(reg.into()));
    }

    pub fn load_free_variable(&mut self, reg: VirtualRegister, index: usize) {
        self.instructions.push(Instruction::LoadLocal(
            reg.into(),
            Value::FreeVariable(index),
        ));
    }

    pub fn get_stack_pointer(&mut self, offset: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::GetStackPointer(dest, offset));
        dest
    }

    pub fn get_stack_pointer_imm(&mut self, num_free: isize) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::GetStackPointerImm(dest, num_free));
        dest
    }

    pub fn load_from_memory(&mut self, source: Value, offset: i32) -> Value {
        let dest = self.volatile_register();
        self.instructions
            .push(Instruction::HeapLoad(dest.into(), source, offset));
        dest.into()
    }

    pub fn get_tag(&mut self, value: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions.push(Instruction::GetTag(dest, value));
        dest
    }

    pub fn untag(&mut self, val: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions.push(Instruction::Untag(dest, val));
        dest
    }

    pub fn tag(&mut self, reg: Value, tag: isize) -> Value {
        let dest = self.volatile_register().into();
        let tag = self.assign_new(Value::RawValue(tag as usize));
        self.instructions
            .push(Instruction::Tag(dest, reg, tag.into()));
        dest
    }

    /// Gets the stack position of live values.
    /// This includes locals and any values we've pushed.
    /// It's is not the same as the actual SP because we
    /// update that at the beginning of the function.
    pub fn get_current_stack_position(&mut self) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::CurrentStackPosition(dest));
        dest
    }

    pub fn extend_register_life(&mut self, register: Value) {
        self.instructions
            .push(Instruction::ExtendLifeTime(register));
    }

    pub fn heap_store_with_reg_offset(&mut self, pointer: Value, value: Value, offset: Value) {
        self.instructions
            .push(Instruction::HeapStoreOffsetReg(pointer, value, offset));
    }

    pub fn write_float_literal(&mut self, float_pointer: Value, n: f64) {
        let temp_register = self.volatile_register();
        self.instructions.push(Instruction::StoreFloat(
            float_pointer,
            temp_register.into(),
            n,
        ))
    }

    fn guard_int(&mut self, a: Value, add_float: Label) {
        let dest = self.volatile_register();
        self.instructions
            .push(Instruction::GuardInt(dest.into(), a, add_float));
    }

    fn guard_float(&mut self, a: Value, add_float: Label) {
        let dest = self.volatile_register();
        self.instructions
            .push(Instruction::GuardFloat(dest.into(), a, add_float));
    }

    pub fn load_from_heap(&mut self, value: Value, arg: i32) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::HeapLoad(dest, value, arg));
        dest
    }

    fn fmov_general_to_float(&mut self, source: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::FmovGeneralToFloat(dest, source));
        dest
    }

    fn fmov_float_to_general(&mut self, source: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::FmovFloatToGeneral(dest, source));
        dest
    }

    fn add_float(&mut self, a: Value, b: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions.push(Instruction::AddFloat(dest, a, b));
        dest
    }

    fn sub_float(&mut self, a: Value, b: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions.push(Instruction::SubFloat(dest, a, b));
        dest
    }

    fn mul_float(&mut self, a: Value, b: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions.push(Instruction::MulFloat(dest, a, b));
        dest
    }

    fn div_float(&mut self, a: Value, b: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions.push(Instruction::DivFloat(dest, a, b));
        dest
    }

    fn allocate(&mut self, size: Value) -> Value {
        let compiler_pointer = self.assign_new(Value::Pointer(self.compiler_pointer));
        let stack_pointer = self.get_stack_pointer_imm(0);
        let f = self.assign_new(Value::Function(self.allocate_fn_pointer));
        self.call_builtin(f.into(), vec![compiler_pointer.into(), stack_pointer, size])
    }

    fn insert_label(&mut self, name: &str, label: Label) -> usize {
        let index = self.labels.len();
        assert!(index == label.index);
        self.labels.push(label);
        self.label_names.push(name.to_string());
        self.label_names.len() - 1
    }
}
