use std::cmp::Ordering;
use std::collections::HashMap;

use bincode::{Decode, Encode};

use crate::ast::IRRange;
use crate::backend::CodegenBackend;

// Backend-specific register imports for TryInto implementations
cfg_if::cfg_if! {
    if #[cfg(any(feature = "backend-x86-64", all(target_arch = "x86_64", not(feature = "backend-arm64"))))] {
        use crate::machine_code::x86_codegen::X86Register as Register;
    } else {
        use crate::machine_code::arm_codegen::Register;
    }
}

use crate::common::Label;
use crate::register_allocation::linear_scan::LinearScan;
use crate::types::BuiltInTypes;
use crate::pretty_print::PrettyPrint;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encode, Decode)]
pub enum Condition {
    LessThanOrEqual,
    LessThan,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

#[derive(Debug, Copy, Clone, Encode, Decode)]
pub enum Value {
    Register(VirtualRegister),
    Spill(VirtualRegister, usize),
    Stack(isize),
    TaggedConstant(isize),
    RawValue(usize),
    // TODO: Think of a better representation
    StringConstantPtr(usize),
    KeywordConstantPtr(usize),
    Function(usize),
    Pointer(usize),
    Local(usize),
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encode, Decode)]
pub struct VirtualRegister {
    pub argument: Option<usize>,
    pub index: usize,
    pub volatile: bool,
    // Hack to experiment with stuff
    pub is_physical: bool,
}

impl Ord for VirtualRegister {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.argument, other.argument) {
            (Some(a), Some(b)) => a.cmp(&b),
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (None, None) => match self.index.cmp(&other.index) {
                Ordering::Equal => match self.volatile.cmp(&other.volatile) {
                    Ordering::Equal => self.is_physical.cmp(&other.is_physical),
                    other => other,
                },
                other => other,
            },
        }
    }
}

impl PartialOrd for VirtualRegister {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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

#[derive(Debug, Clone, Encode, Decode)]
pub enum Instruction {
    Sub(Value, Value, Value),
    AddInt(Value, Value, Value),
    Mul(Value, Value, Value),
    Div(Value, Value, Value),
    Modulo(Value, Value, Value),
    Assign(Value, Value),
    Recurse(Value, Vec<Value>),
    RecurseWithSaves(Value, Vec<Value>, Vec<Value>),
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
    CallWithSaves(Value, Value, Vec<Value>, bool, Vec<Value>),
    HeapLoad(Value, Value, i32),
    HeapLoadReg(Value, Value, Value),
    HeapStore(Value, Value),
    LoadLocal(Value, Value),
    StoreLocal(Value, Value),
    RegisterArgument(Value),
    PushStack(Value),
    PopStack(Value),
    GetStackPointer(Value, Value),
    GetStackPointerImm(Value, isize),
    GetFramePointer(Value),
    GetTag(Value, Value),
    Untag(Value, Value),
    HeapStoreOffset(Value, Value, usize),
    HeapStoreByteOffsetMasked(Value, Value, Value, Value, usize, usize, usize),
    CurrentStackPosition(Value),
    ExtendLifeTime(Value),
    HeapStoreOffsetReg(Value, Value, Value),
    AtomicLoad(Value, Value),
    AtomicStore(Value, Value),
    CompareAndSwap(Value, Value, Value),
    StoreFloat(Value, Value, String),
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
    CompareFloat(Value, Value, Value, Condition),
    ShiftRightImm(Value, Value, i32),
    ShiftRightImmRaw(Value, Value, i32),
    AndImm(Value, Value, u64),
    ShiftLeft(Value, Value, Value),
    ShiftRight(Value, Value, Value),
    ShiftRightZero(Value, Value, Value),
    And(Value, Value, Value),
    Or(Value, Value, Value),
    Xor(Value, Value, Value),
    PushExceptionHandler(Label, Value, usize), // label, result_local, builtin_fn_ptr
    PopExceptionHandler(usize),                // builtin_fn_ptr
    Throw(Value, usize),                       // value, builtin_fn_ptr
    ReadArgCount(Value), // Read arg count register (X9/R10) for variadic functions
    Label(Label),
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

impl TryInto<VirtualRegister> for &mut Value {
    type Error = Value;

    fn try_into(self) -> Result<VirtualRegister, Self::Error> {
        match self {
            Value::Register(register) => Ok(*register),
            _ => Err(*self),
        }
    }
}

impl TryInto<Register> for Value {
    type Error = Value;

    fn try_into(self) -> Result<Register, Self::Error> {
        match self {
            Value::Register(register) => Ok(Register::from_index(register.index)),
            _ => Err(self),
        }
    }
}

impl TryInto<Register> for &Value {
    type Error = Value;

    fn try_into(self) -> Result<Register, Self::Error> {
        match self {
            Value::Register(register) => Ok(Register::from_index(register.index)),
            _ => Err(*self),
        }
    }
}

impl TryInto<Register> for &VirtualRegister {
    type Error = ();

    fn try_into(self) -> Result<Register, Self::Error> {
        Ok(Register::from_index(self.index))
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

macro_rules! replace_register {
    ($x:expr, $old_register:expr, $new_register:expr) => {
        if let Value::Register(register) = $x {
            if *register == $old_register {
                *$x = $new_register;
            }
        }
    };
    () => {};
}

impl Instruction {
    pub fn get_registers(&self) -> Vec<VirtualRegister> {
        match self {
            Instruction::Label(_) => vec![],
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
            Instruction::Modulo(a, b, c) => {
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
            Instruction::ShiftRightImmRaw(a, b, _) => {
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
            Instruction::RecurseWithSaves(a, args, saves) => {
                let mut result: Vec<VirtualRegister> =
                    args.iter().filter_map(|arg| get_registers!(arg)).collect();
                for save in saves {
                    if let Ok(register) = save.try_into() {
                        result.push(register);
                    }
                }
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
            Instruction::CallWithSaves(a, b, args, _, saves) => {
                let mut result: Vec<VirtualRegister> =
                    args.iter().filter_map(|arg| get_registers!(arg)).collect();
                for save in saves {
                    if let Ok(register) = save.try_into() {
                        result.push(register);
                    }
                }
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
            Instruction::CompareFloat(a, b, c, _) => {
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
            Instruction::HeapStoreByteOffsetMasked(a, b, c, d, _, _, _) => {
                get_registers!(a, b, c, d)
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
            Instruction::GetStackPointer(a, b) => {
                get_registers!(a, b)
            }
            Instruction::CurrentStackPosition(a) => {
                get_register!(a)
            }
            Instruction::GetStackPointerImm(a, _) => {
                get_register!(a)
            }
            Instruction::GetFramePointer(a) => {
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
            Instruction::PushExceptionHandler(_, local, _) => {
                get_register!(local)
            }
            Instruction::PopExceptionHandler(_) => {
                vec![]
            }
            Instruction::Throw(value, _) => {
                get_register!(value)
            }
            Instruction::ReadArgCount(a) => {
                get_register!(a)
            }
        }
    }

    // TODO: Replace with get_registers_mut
    pub fn replace_register(&mut self, old_register: VirtualRegister, new_register: Value) {
        match self {
            Instruction::Label(_) => {}
            Instruction::HeapStoreByteOffsetMasked(value, value1, value2, value3, _, _, _) => {
                replace_register!(value, old_register, new_register);
                replace_register!(value1, old_register, new_register);
                replace_register!(value2, old_register, new_register);
                replace_register!(value3, old_register, new_register);
            }

            Instruction::Sub(value, value1, value2)
            | Instruction::AddInt(value, value1, value2)
            | Instruction::Mul(value, value1, value2)
            | Instruction::Div(value, value1, value2)
            | Instruction::Modulo(value, value1, value2)
            | Instruction::HeapLoadReg(value, value1, value2)
            | Instruction::HeapStoreOffsetReg(value, value1, value2)
            | Instruction::ShiftLeft(value, value1, value2)
            | Instruction::ShiftRight(value, value1, value2)
            | Instruction::ShiftRightZero(value, value1, value2)
            | Instruction::And(value, value1, value2)
            | Instruction::Or(value, value1, value2)
            | Instruction::Xor(value, value1, value2)
            | Instruction::AddFloat(value, value1, value2)
            | Instruction::SubFloat(value, value1, value2)
            | Instruction::MulFloat(value, value1, value2)
            | Instruction::DivFloat(value, value1, value2)
            | Instruction::Compare(value, value1, value2, _)
            | Instruction::CompareFloat(value, value1, value2, _)
            | Instruction::Tag(value, value1, value2)
            | Instruction::CompareAndSwap(value, value1, value2) => {
                replace_register!(value, old_register, new_register);
                replace_register!(value1, old_register, new_register);
                replace_register!(value2, old_register, new_register);
            }

            Instruction::HeapLoad(value, value1, _)
            | Instruction::HeapStore(value, value1)
            | Instruction::LoadLocal(value, value1)
            | Instruction::StoreLocal(value, value1)
            | Instruction::GetStackPointer(value, value1)
            | Instruction::GetTag(value, value1)
            | Instruction::Untag(value, value1)
            | Instruction::HeapStoreOffset(value, value1, _)
            | Instruction::AtomicLoad(value, value1)
            | Instruction::AtomicStore(value, value1)
            | Instruction::StoreFloat(value, value1, _)
            | Instruction::GuardInt(value, value1, _)
            | Instruction::GuardFloat(value, value1, _)
            | Instruction::LoadConstant(value, value1)
            | Instruction::FmovGeneralToFloat(value, value1)
            | Instruction::FmovFloatToGeneral(value, value1)
            | Instruction::ShiftRightImm(value, value1, _)
            | Instruction::ShiftRightImmRaw(value, value1, _)
            | Instruction::AndImm(value, value1, _)
            | Instruction::JumpIf(_, _, value, value1) => {
                replace_register!(value, old_register, new_register);
                replace_register!(value1, old_register, new_register);
            }

            Instruction::Ret(value)
            | Instruction::LoadTrue(value)
            | Instruction::LoadFalse(value)
            | Instruction::RegisterArgument(value)
            | Instruction::PushStack(value)
            | Instruction::PopStack(value)
            | Instruction::GetStackPointerImm(value, _)
            | Instruction::GetFramePointer(value)
            | Instruction::CurrentStackPosition(value)
            | Instruction::ExtendLifeTime(value)
            | Instruction::ReadArgCount(value) => {
                replace_register!(value, old_register, new_register);
            }

            Instruction::Assign(virtual_register, value) => {
                replace_register!(virtual_register, old_register, new_register);
                replace_register!(value, old_register, new_register);
            }
            Instruction::Recurse(value, vec) => {
                replace_register!(value, old_register, new_register);
                for value in vec {
                    replace_register!(value, old_register, new_register);
                }
            }
            Instruction::RecurseWithSaves(value, vec, saves) => {
                replace_register!(value, old_register, new_register);
                for value in vec {
                    replace_register!(value, old_register, new_register);
                }
                for save in saves {
                    replace_register!(save, old_register, new_register);
                }
            }
            Instruction::TailRecurse(value, vec) => {
                replace_register!(value, old_register, new_register);
                for value in vec {
                    replace_register!(value, old_register, new_register);
                }
            }
            Instruction::Call(value, value1, vec, _) => {
                replace_register!(value, old_register, new_register);
                replace_register!(value1, old_register, new_register);
                for value in vec {
                    replace_register!(value, old_register, new_register);
                }
            }

            Instruction::CallWithSaves(value, value1, vec, _, saves) => {
                replace_register!(value, old_register, new_register);
                replace_register!(value1, old_register, new_register);
                for value in vec {
                    replace_register!(value, old_register, new_register);
                }
                for save in saves {
                    replace_register!(save, old_register, new_register);
                }
            }

            Instruction::Jump(_) | Instruction::Breakpoint => {}
            Instruction::PushExceptionHandler(_, value, _) => {
                replace_register!(value, old_register, new_register);
            }
            Instruction::PopExceptionHandler(_) => {}
            Instruction::Throw(value, _) => {
                replace_register!(value, old_register, new_register);
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MachineCodeRange {
    pub start: usize,
    pub end: usize,
}

impl MachineCodeRange {
    fn new(start_machine_code: usize, end_machine_code: usize) -> Self {
        Self {
            start: start_machine_code,
            end: end_machine_code,
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
    allocate_fn_pointer: usize,
    after_return: Label,
    pub ir_to_machine_code_range: Vec<(usize, MachineCodeRange)>,
    pub ir_range_to_token_range: Vec<(crate::ast::TokenRange, IRRange)>,
    /// Number of argument registers for the target architecture (8 for ARM64, 6 for x86-64)
    num_arg_registers: usize,
}

impl Ir {
    pub fn new(allocate_fn_pointer: usize) -> Self {
        // Determine number of argument registers based on target architecture
        #[cfg(any(
            feature = "backend-x86-64",
            all(target_arch = "x86_64", not(feature = "backend-arm64"))
        ))]
        let num_arg_registers = 6; // x86-64 SysV ABI: RDI, RSI, RDX, RCX, R8, R9
        #[cfg(not(any(
            feature = "backend-x86-64",
            all(target_arch = "x86_64", not(feature = "backend-arm64"))
        )))]
        let num_arg_registers = 8; // ARM64: X0-X7

        let mut me = Self {
            register_index: 0,
            instructions: vec![],
            labels: vec![],
            label_names: vec![],
            label_locations: HashMap::new(),
            num_locals: 0,
            allocate_fn_pointer,
            after_return: Label { index: 0 },
            ir_to_machine_code_range: vec![],
            ir_range_to_token_range: vec![],
            num_arg_registers,
        };

        me.insert_label("after_return", me.after_return);
        me
    }

    pub fn current_position(&self) -> usize {
        self.instructions.len()
    }

    fn next_register(&mut self, argument: Option<usize>, volatile: bool) -> VirtualRegister {
        let register = VirtualRegister {
            argument,
            index: self.register_index,
            volatile,
            is_physical: false,
        };
        self.register_index += 1;
        register
    }

    pub fn arg(&mut self, n: usize) -> Value {
        if n >= self.num_arg_registers {
            // Stack arguments are passed above the frame header
            // For ARM64: saved FP and LR = 2 words, so offset = (n - 8) + 2
            // For x86-64: return address + saved RBP = 2 words, so offset = (n - 6) + 2
            return Value::Stack((n as isize) - (self.num_arg_registers as isize) + 2);
        }
        let register = self.next_register(Some(n), true);
        Value::Register(register)
    }

    /// Read the argument count register (X9 for ARM64, R10 for x86-64).
    /// Used by variadic functions to determine how many arguments were passed.
    pub fn read_arg_count(&mut self) -> Value {
        let dest = self.next_register(None, false);
        self.instructions
            .push(Instruction::ReadArgCount(Value::Register(dest)));
        Value::Register(dest)
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
        // TODO: result registers like this can cause problems if not assigned
        // Need to think about this more
        let result_register = self.assign_new(Value::TaggedConstant(0));
        let a: VirtualRegister = self.assign_new(a.into());
        let b: VirtualRegister = self.assign_new(b.into());
        let add_float: Label = self.label("add_float");
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

        let size_reg = self.assign_new(1);
        let float_pointer = self.allocate(size_reg.into());
        let float_pointer = self.untag(float_pointer);

        let a = self.untag(a.into());
        let b = self.untag(b.into());
        let a = self.load_from_heap(a, 1);
        let b = self.load_from_heap(b, 1);
        let a = self.fmov_general_to_float(a);
        let b = self.fmov_general_to_float(b);
        let result = op_float(self, a, b);
        let result = self.fmov_float_to_general(result);
        // Allocate and store
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

    pub fn modulo<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let register = self.volatile_register();
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        self.instructions
            .push(Instruction::Modulo(register.into(), a.into(), b.into()));
        Value::Register(register)
    }

    pub fn modulo_any<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        // For now, only integer modulo is supported
        // True modulo: ((a % b) + b) % b to handle negative numbers correctly
        self.modulo(a, b)
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

    pub fn compare_any(&mut self, a: Value, b: Value, condition: Condition) -> Value {
        let result_register = self.assign_new(Value::TaggedConstant(0));
        let a: VirtualRegister = self.assign_new(a);
        let b: VirtualRegister = self.assign_new(b);
        let compare_float_label: Label = self.label("compare_float");
        let default_compare_label: Label = self.label("default_compare");
        let after_compare = self.label("after_compare");

        // Check if BOTH values are floats - only then use float comparison
        self.guard_float(a.into(), default_compare_label);
        self.guard_float(b.into(), default_compare_label);

        // Float path - both are floats
        self.write_label(compare_float_label);
        let a_untagged = self.untag(a.into());
        let b_untagged = self.untag(b.into());
        let a_raw = self.load_from_heap(a_untagged, 1);
        let b_raw = self.load_from_heap(b_untagged, 1);
        let a_float = self.fmov_general_to_float(a_raw);
        let b_float = self.fmov_general_to_float(b_raw);

        let tag2 = self.assign_new(Value::RawValue(BuiltInTypes::Bool.get_tag() as usize));
        let dest2 = self.volatile_register();
        self.instructions.push(Instruction::CompareFloat(
            dest2.into(),
            a_float,
            b_float,
            condition,
        ));
        self.instructions
            .push(Instruction::Tag(dest2.into(), dest2.into(), tag2.into()));
        self.assign(result_register, dest2);
        self.jump(after_compare);

        // Default path - use regular comparison (works for ints, bools, null, pointers, etc.)
        self.write_label(default_compare_label);
        let tag = self.assign_new(Value::RawValue(BuiltInTypes::Bool.get_tag() as usize));
        let dest = self.volatile_register();
        self.instructions.push(Instruction::Compare(
            dest.into(),
            a.into(),
            b.into(),
            condition,
        ));
        self.instructions
            .push(Instruction::Tag(dest.into(), dest.into(), tag.into()));
        self.assign(result_register, dest);

        self.write_label(after_compare);
        Value::Register(result_register)
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

    pub fn shift_right_imm_raw(&mut self, a: Value, b: i32) -> Value {
        let a = self.assign_new(a);
        let destination = self.volatile_register();
        self.instructions.push(Instruction::ShiftRightImmRaw(
            destination.into(),
            a.into(),
            b,
        ));
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
            .push(Instruction::Assign(dest.into(), val.into()));
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
        self.instructions
            .push(Instruction::Assign(Value::Register(register), val));
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
        self.instructions
            .push(Instruction::Assign(register.into(), val));
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

    pub fn write_label(&mut self, label: Label) {
        self.instructions.push(Instruction::Label(label));
        assert!(!self.label_locations.contains_key(&self.instructions.len()));
        self.label_locations
            .insert(self.instructions.len(), label.index);
    }

    pub fn compile<B: CodegenBackend>(&mut self, mut backend: B, error_fn_pointer: usize) -> B {
        debug_assert!(!self.ir_range_to_token_range.is_empty());

        // backend.breakpoint();

        let mut linear_scan = LinearScan::new(self.instructions.clone(), self.num_locals);
        linear_scan.allocate();

        self.instructions = linear_scan.instructions.clone();
        let num_spills = linear_scan.location.len();
        self.num_locals += num_spills;
        backend.set_max_locals(self.num_locals);

        // Inform the backend which callee-saved registers are used.
        // This allows the backend to save/restore them in prologue/epilogue
        // per the AAPCS64 / System V AMD64 ABI.
        backend.reset_callee_saved_tracking();
        for allocated in linear_scan.allocated_registers.values() {
            if allocated.is_physical {
                backend.mark_callee_saved_register_used(allocated.index);
            }
        }

        let before_prelude = backend.new_label("before_prelude");
        backend.write_label(before_prelude);

        backend.prelude();

        // NOTE: Local initialization is handled by patch_prelude_and_epilogue()
        // which uses temporary registers X9/X10 (not callee-saved registers).
        // Previously this code used X19 without saving it, corrupting callers' values.

        let after_prelude = backend.new_label("after_prelude");
        backend.write_label(after_prelude);

        let exit = backend.new_label("exit");

        // let mut simple_register_allocator = SimpleRegisterAllocator::new(
        //     self.instructions.clone(),
        //     self.num_locals,
        //     self.label_locations.clone(),
        //     self.ir_range_to_token_range.clone(),
        // );
        // simple_register_allocator.simplify_registers();
        // self.instructions = simple_register_allocator.resulting_instructions.clone();
        // self.num_locals = simple_register_allocator.max_num_locals;
        // self.label_locations = simple_register_allocator.label_locations.clone();
        // self.ir_range_to_token_range = simple_register_allocator.ir_range_to_token_range.clone();

        // eprintln!("{}", self.instructions.pretty_print());
        self.compile_instructions(&mut backend, exit, before_prelude, after_prelude);

        backend.write_label(exit);

        backend.epilogue();
        backend.ret();
        // TODO: ugly
        let backend_after_return = backend.get_label_by_name("after_return");
        backend.write_label(backend_after_return);
        let register = backend.get_volatile_register(0);
        backend.mov_64(register, error_fn_pointer as isize);
        backend.get_stack_pointer_imm(backend.arg(0), 0);
        // throw_error expects (stack_pointer, frame_pointer)
        backend.mov_reg(backend.arg(1), backend.frame_pointer());
        backend.call(register);
        backend
    }

    pub fn value_to_register<B: CodegenBackend>(
        &self,
        value: &Value,
        backend: &mut B,
    ) -> B::Register {
        match value {
            Value::Register(register) => backend.register_from_index(register.index),
            Value::Local(index) => {
                let temp_reg = backend.temporary_register();
                backend.load_local(temp_reg, *index as i32);
                temp_reg
            }
            Value::Spill(_register, index) => {
                let temp_reg = backend.temporary_register();
                backend.load_local(temp_reg, *index as i32);
                temp_reg
            }
            Value::RawValue(val) => {
                let temp_reg = backend.temporary_register();
                backend.mov_64(temp_reg, *val as isize);
                temp_reg
            }
            _ => panic!("Expected register got {:?}", value),
        }
    }

    fn store_spill<B: CodegenBackend>(
        &self,
        dest: B::Register,
        dest_spill: Option<usize>,
        backend: &mut B,
    ) {
        if let Some(dest_spill) = dest_spill {
            backend.store_local(dest, dest_spill as i32);
        }
    }

    fn dest_spill(&self, dest: &Value) -> Option<usize> {
        match dest {
            Value::Spill(_, index) => Some(*index),
            _ => None,
        }
    }

    fn compile_instructions<B: CodegenBackend>(
        &mut self,
        backend: &mut B,
        exit: Label,
        before_prelude: Label,
        after_prelude: Label,
    ) {
        let mut ir_label_to_lang_label: HashMap<Label, Label> = HashMap::new();
        let mut labels: Vec<&Label> = self.labels.iter().collect();
        labels.sort_by_key(|label| label.index);
        for label in labels.iter() {
            let new_label = backend.new_label(&self.label_names[label.index]);
            ir_label_to_lang_label.insert(**label, new_label);
        }

        for (index, instruction) in self.instructions.iter().enumerate() {
            let start_machine_code = backend.current_position();
            let label = self.label_locations.get(&index);
            if let Some(label) = label {
                backend.write_label(ir_label_to_lang_label[&self.labels[*label]]);
            }
            backend.clear_temporary_registers();
            // println!("instruction {:?}", instruction);
            match instruction {
                Instruction::Breakpoint => {
                    backend.breakpoint();
                }
                Instruction::Label(_) => {}
                Instruction::ExtendLifeTime(_) => {}
                Instruction::Sub(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);

                    // Use a temporary register for guard checks to avoid clobbering operands
                    // if dest happens to be the same register as a or b
                    let guard_temp = backend.temporary_register();
                    backend.guard_integer(guard_temp, a, self.after_return);
                    backend.guard_integer(guard_temp, b, self.after_return);
                    backend.free_temporary_register(guard_temp);

                    backend.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    backend.sub(dest, a, b);
                    backend.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    // Only re-tag operands if they're different from dest
                    if a != dest {
                        backend.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    }
                    if b != dest {
                        backend.shift_left_imm(b, b, BuiltInTypes::tag_size());
                    }
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::AddInt(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);

                    backend.add(dest, a, b);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::Mul(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);

                    // Use a temporary register for guard checks to avoid clobbering operands
                    // if dest happens to be the same register as a or b
                    let guard_temp = backend.temporary_register();
                    backend.guard_integer(guard_temp, a, self.after_return);
                    backend.guard_integer(guard_temp, b, self.after_return);
                    backend.free_temporary_register(guard_temp);

                    backend.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    backend.mul(dest, a, b);
                    backend.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    // Only re-tag operands if they're different from dest to avoid
                    // double-shifting the result
                    if a != dest {
                        backend.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    }
                    if b != dest {
                        backend.shift_left_imm(b, b, BuiltInTypes::tag_size());
                    }
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::Div(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);

                    // Use a temporary register for guard checks to avoid clobbering operands
                    // if dest happens to be the same register as a or b
                    let guard_temp = backend.temporary_register();
                    backend.guard_integer(guard_temp, a, self.after_return);
                    backend.guard_integer(guard_temp, b, self.after_return);
                    backend.free_temporary_register(guard_temp);

                    backend.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    backend.div(dest, a, b);
                    backend.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    // Only re-tag operands if they're different from dest
                    if a != dest {
                        backend.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    }
                    if b != dest {
                        backend.shift_left_imm(b, b, BuiltInTypes::tag_size());
                    }
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::Modulo(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);

                    // Use a temporary register for guard checks
                    let guard_temp = backend.temporary_register();
                    backend.guard_integer(guard_temp, a, self.after_return);
                    backend.guard_integer(guard_temp, b, self.after_return);
                    backend.free_temporary_register(guard_temp);

                    backend.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    // True modulo: result is always non-negative when divisor is positive
                    backend.modulo(dest, a, b);
                    backend.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    // Only re-tag operands if they're different from dest
                    if a != dest {
                        backend.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    }
                    if b != dest {
                        backend.shift_left_imm(b, b, BuiltInTypes::tag_size());
                    }
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::ShiftRightImm(dest, value, shift) => {
                    let value = self.value_to_register(value, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);

                    backend.guard_integer(dest, value, self.after_return);

                    backend.shift_right_imm(dest, value, *shift);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::ShiftRightImmRaw(dest, value, shift) => {
                    let value = self.value_to_register(value, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.shift_right_imm(dest, value, *shift);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::AndImm(dest, value, imm) => {
                    let value = self.value_to_register(value, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.and_imm(dest, value, *imm);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::ShiftLeft(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);

                    backend.guard_integer(dest, a, self.after_return);
                    backend.guard_integer(dest, b, self.after_return);

                    backend.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    backend.shift_left(dest, a, b);
                    backend.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    backend.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_left_imm(b, b, BuiltInTypes::tag_size());
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::ShiftRight(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);

                    backend.guard_integer(dest, a, self.after_return);
                    backend.guard_integer(dest, b, self.after_return);

                    backend.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    backend.shift_right(dest, a, b);
                    backend.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    backend.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_left_imm(b, b, BuiltInTypes::tag_size());
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::ShiftRightZero(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);

                    backend.guard_integer(dest, a, self.after_return);
                    backend.guard_integer(dest, b, self.after_return);

                    backend.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    backend.and_imm(a, a, 0xFFFFFFFF);
                    backend.shift_right_zero(dest, a, b);
                    backend.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    backend.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_left_imm(b, b, BuiltInTypes::tag_size());
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::And(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.and(dest, a, b);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::Or(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.or(dest, a, b);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::Xor(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.xor(dest, a, b);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::GuardInt(dest, value, label) => {
                    let value = self.value_to_register(value, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.guard_integer(dest, value, ir_label_to_lang_label[label]);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::GuardFloat(dest, value, label) => {
                    let value = self.value_to_register(value, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.guard_float(dest, value, ir_label_to_lang_label[label]);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::FmovGeneralToFloat(dest, src) => {
                    let src = self.value_to_register(src, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.fmov_to_float(dest, src);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::FmovFloatToGeneral(dest, src) => {
                    let src = self.value_to_register(src, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.fmov_from_float(dest, src);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::AddFloat(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.fadd(dest, a, b);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::SubFloat(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.fsub(dest, a, b);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::MulFloat(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.fmul(dest, a, b);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::DivFloat(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.fdiv(dest, a, b);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::Assign(dest, val) => {
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    match val {
                        Value::Register(_virt_reg) => {
                            let register = self.value_to_register(val, backend);
                            backend.mov_reg(dest, register);
                            self.store_spill(dest, dest_spill, backend);
                        }
                        Value::TaggedConstant(i) => {
                            let tagged = BuiltInTypes::construct_int(*i);
                            backend.mov_64(dest, tagged);
                            self.store_spill(dest, dest_spill, backend);
                        }
                        Value::StringConstantPtr(ptr) => {
                            let tagged = BuiltInTypes::String.tag(*ptr as isize);
                            backend.mov_64(dest, tagged);
                            self.store_spill(dest, dest_spill, backend);
                        }
                        Value::KeywordConstantPtr(ptr) => {
                            // Just pass the raw index, not tagged
                            backend.mov_64(dest, *ptr as isize);
                            self.store_spill(dest, dest_spill, backend);
                        }
                        Value::Function(id) => {
                            let function = BuiltInTypes::Function.tag(*id as isize);
                            backend.mov_64(dest, function);
                            self.store_spill(dest, dest_spill, backend);
                        }
                        Value::Pointer(ptr) => {
                            backend.mov_64(dest, *ptr as isize);
                            self.store_spill(dest, dest_spill, backend);
                        }
                        Value::RawValue(value) => {
                            backend.mov_64(dest, *value as isize);
                            self.store_spill(dest, dest_spill, backend);
                        }
                        Value::True => {
                            backend.mov_64(dest, BuiltInTypes::construct_boolean(true));
                            self.store_spill(dest, dest_spill, backend);
                        }
                        Value::False => {
                            backend.mov_64(dest, BuiltInTypes::construct_boolean(false));
                            self.store_spill(dest, dest_spill, backend);
                        }
                        Value::Local(local) => {
                            backend.load_local(dest, *local as i32);
                            self.store_spill(dest, dest_spill, backend);
                        }
                        Value::Null => {
                            backend.mov_64(dest, 0b111_isize);
                            self.store_spill(dest, dest_spill, backend);
                        }
                        Value::Spill(_register, index) => {
                            let temp_reg = backend.temporary_register();
                            backend.load_local(temp_reg, (*index) as i32);
                            backend.mov_reg(dest, temp_reg);
                            self.store_spill(dest, dest_spill, backend);
                        }
                        Value::Stack(offset) => {
                            backend.load_from_stack_beginning(dest, *offset as i32);
                            self.store_spill(dest, dest_spill, backend);
                        }
                    }
                }
                Instruction::LoadConstant(dest, val) => {
                    let val = self.value_to_register(val, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.mov_reg(dest, val);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::LoadLocal(dest, local) => {
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    let local = local.as_local();
                    backend.load_local(dest, local as i32);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::StoreLocal(dest, value) => {
                    let value = self.value_to_register(value, backend);
                    backend.store_local(value, dest.as_local() as i32);
                }
                Instruction::LoadTrue(dest) => {
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.mov_64(dest, BuiltInTypes::construct_boolean(true));
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::LoadFalse(dest) => {
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.mov_64(dest, BuiltInTypes::construct_boolean(false));
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::StoreFloat(dest, temp, value) => {
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    let temp = self.value_to_register(temp, backend);
                    // need to turn string to float precisely
                    let value: f64 = value.parse().unwrap();

                    backend.mov_64(temp, value.to_bits() as isize);
                    // The header is the first field, so offset is 1
                    backend.store_on_heap(dest, temp, 1);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::Recurse(dest, args) => {
                    // TODO: Clean up duplication
                    let num_arg_regs = backend.num_arg_registers();
                    for (arg_index, arg) in args.iter().enumerate().rev() {
                        let arg = self.value_to_register(arg, backend);
                        if arg_index < num_arg_regs {
                            backend.mov_reg(backend.arg(arg_index as u8), arg);
                        } else {
                            backend.push_to_end_of_stack(
                                arg,
                                (arg_index as i32) - (num_arg_regs as i32 - 1),
                            );
                        }
                    }

                    // Set arg_count for uniform variadic calling convention
                    let arg_count_tagged = (args.len() << BuiltInTypes::tag_size()) as isize;
                    backend.mov_64(backend.arg_count_reg(), arg_count_tagged);

                    backend.recurse(before_prelude);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.mov_reg(dest, backend.ret_reg());
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::RecurseWithSaves(dest, args, saves) => {
                    // Save registers using RBP-relative push_to_stack
                    for save in saves.iter() {
                        let save = self.value_to_register(save, backend);
                        backend.push_to_stack(save);
                    }

                    // Set up arguments - no adjustment for saves since they don't change RSP
                    let num_arg_regs = backend.num_arg_registers();
                    for (arg_index, arg) in args.iter().enumerate().rev() {
                        let arg = self.value_to_register(arg, backend);
                        if arg_index < num_arg_regs {
                            backend.mov_reg(backend.arg(arg_index as u8), arg);
                        } else {
                            // Recurse uses a different formula because it's a jump, not a call
                            backend.push_to_end_of_stack(
                                arg,
                                (arg_index as i32) - (num_arg_regs as i32 - 1),
                            );
                        }
                    }

                    // Set arg_count for uniform variadic calling convention
                    let arg_count_tagged = (args.len() << BuiltInTypes::tag_size()) as isize;
                    backend.mov_64(backend.arg_count_reg(), arg_count_tagged);

                    backend.recurse(before_prelude);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.mov_reg(dest, backend.ret_reg());
                    self.store_spill(dest, dest_spill, backend);

                    // Restore saved registers (in reverse order)
                    for save in saves.iter().rev() {
                        let save = self.value_to_register(save, backend);
                        backend.pop_from_stack(save);
                    }
                }
                Instruction::TailRecurse(dest, args) => {
                    let num_arg_regs = backend.num_arg_registers();
                    for (arg_index, arg) in args.iter().enumerate().rev() {
                        let arg = self.value_to_register(arg, backend);
                        if arg_index < num_arg_regs {
                            backend.mov_reg(backend.arg(arg_index as u8), arg);
                        } else {
                            backend.push_to_end_of_stack(
                                arg,
                                (arg_index as i32) - (num_arg_regs as i32 - 1),
                            );
                        }
                    }

                    // Set arg_count for uniform variadic calling convention
                    let arg_count_tagged = (args.len() << BuiltInTypes::tag_size()) as isize;
                    backend.mov_64(backend.arg_count_reg(), arg_count_tagged);

                    backend.jump(after_prelude);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.mov_reg(dest, backend.ret_reg());
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::Call(dest, function, args, builtin) => {
                    // TODO: I think I should never hit this with how my register allocator works
                    let num_arg_regs = backend.num_arg_registers();

                    for (arg_index, arg) in args.iter().enumerate().rev() {
                        let arg_reg = self.value_to_register(arg, backend);

                        if arg_index < num_arg_regs {
                            backend.mov_reg(backend.arg(arg_index as u8), arg_reg);
                        } else {
                            backend.push_to_end_of_stack(
                                arg_reg,
                                (arg_index as i32) - (num_arg_regs as i32 - 1),
                            );
                        }
                    }
                    // TODO: I am not actually checking any tags here
                    // or unmasking or anything. Just straight up calling it
                    let function = self.value_to_register(function, backend);
                    backend.shift_right_imm(function, function, BuiltInTypes::tag_size());

                    // Set arg_count for uniform variadic calling convention
                    // All calls set this so variadic functions can read it
                    let arg_count_tagged = (args.len() << BuiltInTypes::tag_size()) as isize;
                    backend.mov_64(backend.arg_count_reg(), arg_count_tagged);

                    if *builtin {
                        backend.call_builtin(function);
                    } else {
                        backend.call(function);
                    }

                    let dest_spill = self.dest_spill(dest);
                    // For dest, we need a register to store the result into.
                    // Don't use value_to_register for Spill because it loads from
                    // the spill slot, which might use RAX as a temp and clobber
                    // the function result that's currently in RAX.
                    let dest_reg = match dest {
                        Value::Register(register) => backend.register_from_index(register.index),
                        Value::Spill(_, _) => backend.temporary_register(),
                        _ => panic!("Unexpected dest type for call: {:?}", dest),
                    };
                    backend.mov_reg(dest_reg, backend.ret_reg());
                    self.store_spill(dest_reg, dest_spill, backend);
                    if matches!(dest, Value::Spill(_, _)) {
                        backend.free_temporary_register(dest_reg);
                    }
                }
                Instruction::CallWithSaves(dest, function, args, builtin, saves) => {
                    // The saves are stored using push_to_stack which uses RBP-relative
                    // addressing and doesn't modify RSP. Therefore, stack args should
                    // be placed at their standard RSP-relative positions WITHOUT
                    // accounting for saves.

                    // Check if function is in a register that will be saved
                    let function_in_save = if let Value::Register(reg) = function {
                        saves.iter().any(|save| {
                            if let Value::Register(save_reg) = save {
                                save_reg.index == reg.index
                            } else {
                                false
                            }
                        })
                    } else {
                        false
                    };

                    // If function is in a register that will be saved,
                    // we need to capture it BEFORE saving
                    let pre_captured_function = if function_in_save {
                        let function_reg = self.value_to_register(function, backend);
                        let temp = backend.temporary_register();
                        backend.mov_reg(temp, function_reg);
                        Some(temp)
                    } else {
                        None
                    };

                    // Save registers that are live across the call
                    // push_to_stack uses RBP-relative addressing, not RSP
                    for save in saves.iter() {
                        let save_reg = self.value_to_register(save, backend);
                        backend.push_to_stack(save_reg);
                    }

                    // Set up arguments
                    let num_arg_regs = backend.num_arg_registers();
                    for (arg_index, arg) in args.iter().enumerate().rev() {
                        let arg_reg = self.value_to_register(arg, backend);
                        if arg_index < num_arg_regs {
                            backend.mov_reg(backend.arg(arg_index as u8), arg_reg);
                        } else {
                            // Stack args at [RSP + (arg_index - num_arg_regs)*8]
                            // NO adjustment for saves because saves don't change RSP
                            backend.push_to_end_of_stack(
                                arg_reg,
                                (arg_index as i32) - num_arg_regs as i32,
                            );
                        }
                        backend.free_temporary_register(arg_reg);
                    }

                    // Get function pointer
                    let call_target = if let Some(temp) = pre_captured_function {
                        temp
                    } else {
                        self.value_to_register(function, backend)
                    };

                    // Untag function pointer and call
                    backend.shift_right_imm(call_target, call_target, BuiltInTypes::tag_size());

                    // Set arg_count for uniform variadic calling convention
                    let arg_count_tagged = (args.len() << BuiltInTypes::tag_size()) as isize;
                    backend.mov_64(backend.arg_count_reg(), arg_count_tagged);

                    if *builtin {
                        backend.call_builtin(call_target);
                    } else {
                        backend.call(call_target);
                    }
                    backend.free_temporary_register(call_target);

                    let dest_spill = self.dest_spill(dest);
                    let dest_reg = match dest {
                        Value::Register(register) => backend.register_from_index(register.index),
                        Value::Spill(_, _) => backend.temporary_register(),
                        _ => panic!("Unexpected dest type for call: {:?}", dest),
                    };
                    backend.mov_reg(dest_reg, backend.ret_reg());
                    self.store_spill(dest_reg, dest_spill, backend);
                    if matches!(dest, Value::Spill(_, _)) {
                        backend.free_temporary_register(dest_reg);
                    }

                    // Restore saved registers (in reverse order)
                    for save in saves.iter().rev() {
                        let save_reg = self.value_to_register(save, backend);
                        backend.pop_from_stack(save_reg);
                    }
                }
                Instruction::Compare(dest, a, b, condition) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.compare_bool(*condition, dest, a, b);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::CompareFloat(dest, a, b, condition) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.compare_float_bool(*condition, dest, a, b);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::Tag(dest, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.tag_value(dest, a, b);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::JumpIf(label, condition, a, b) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let label = ir_label_to_lang_label.get(label).unwrap();
                    backend.compare(a, b);
                    match condition {
                        Condition::LessThanOrEqual => backend.jump_less_or_equal(*label),
                        Condition::LessThan => backend.jump_less(*label),
                        Condition::Equal => backend.jump_equal(*label),
                        Condition::NotEqual => backend.jump_not_equal(*label),
                        Condition::GreaterThan => backend.jump_greater(*label),
                        Condition::GreaterThanOrEqual => backend.jump_greater_or_equal(*label),
                    }
                }
                Instruction::Jump(label) => {
                    let label = ir_label_to_lang_label.get(label).unwrap();
                    backend.jump(*label);
                }
                Instruction::Ret(value) => match value {
                    Value::Register(_virt_reg) => {
                        let register = self.value_to_register(value, backend);
                        if register == backend.ret_reg() {
                            backend.jump(exit);
                        } else {
                            backend.mov_reg(backend.ret_reg(), register);
                            backend.jump(exit);
                        }
                    }
                    Value::TaggedConstant(i) => {
                        backend.mov_64(backend.ret_reg(), BuiltInTypes::construct_int(*i));
                        backend.jump(exit);
                    }
                    Value::StringConstantPtr(ptr) => {
                        backend.mov_64(backend.ret_reg(), *ptr as isize);
                        backend.jump(exit);
                    }
                    Value::KeywordConstantPtr(ptr) => {
                        // Just pass the raw index
                        backend.mov_64(backend.ret_reg(), *ptr as isize);
                        backend.jump(exit);
                    }
                    Value::Function(id) => {
                        backend.mov_64(backend.ret_reg(), *id as isize);
                        backend.jump(exit);
                    }
                    Value::Pointer(ptr) => {
                        backend.mov_64(backend.ret_reg(), *ptr as isize);
                        backend.jump(exit);
                    }
                    Value::True => {
                        backend.mov_64(backend.ret_reg(), BuiltInTypes::construct_boolean(true));
                        backend.jump(exit);
                    }
                    Value::False => {
                        backend.mov_64(backend.ret_reg(), BuiltInTypes::construct_boolean(false));
                        backend.jump(exit);
                    }
                    Value::RawValue(_) => {
                        panic!("Should we be returing a raw value?")
                    }
                    Value::Null => {
                        backend.mov_64(backend.ret_reg(), 0b111);
                        backend.jump(exit);
                    }
                    Value::Local(local) => {
                        backend.load_local(backend.ret_reg(), *local as i32);
                        backend.jump(exit);
                    }
                    Value::Spill(_register, index) => {
                        let temp_reg = backend.temporary_register();
                        backend.load_local(temp_reg, (*index) as i32);
                        backend.mov_reg(backend.ret_reg(), temp_reg);
                        backend.jump(exit);
                    }
                    Value::Stack(offset) => {
                        backend.load_from_stack_beginning(backend.ret_reg(), *offset as i32);
                        backend.jump(exit);
                    }
                },
                Instruction::HeapLoad(dest, ptr, offset) => {
                    let ptr = self.value_to_register(ptr, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.load_from_heap(dest, ptr, *offset);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::AtomicLoad(dest, ptr) => {
                    // TODO: Does the spill work properly here?
                    let ptr = self.value_to_register(ptr, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.atomic_load(dest, ptr);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::AtomicStore(ptr, val) => {
                    let ptr = self.value_to_register(ptr, backend);
                    let val = self.value_to_register(val, backend);
                    backend.atomic_store(ptr, val);
                }
                Instruction::CompareAndSwap(dest, ptr, val) => {
                    let ptr = self.value_to_register(ptr, backend);
                    let val = self.value_to_register(val, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.compare_and_swap(dest, ptr, val);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::HeapLoadReg(dest, ptr, offset) => {
                    let ptr = self.value_to_register(ptr, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    let offset = self.value_to_register(offset, backend);
                    backend.load_from_heap_with_reg_offset(dest, ptr, offset);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::HeapStore(ptr, val) => {
                    let ptr = self.value_to_register(ptr, backend);
                    let val = self.value_to_register(val, backend);
                    backend.store_on_heap(ptr, val, 0);
                }

                Instruction::HeapStoreOffset(ptr, val, offset) => {
                    let ptr = self.value_to_register(ptr, backend);
                    let val = self.value_to_register(val, backend);
                    backend.store_on_heap(ptr, val, *offset as i32);
                }
                Instruction::HeapStoreByteOffsetMasked(
                    ptr,
                    val,
                    temp1,
                    temp2,
                    offset,
                    byte_offset,
                    mask,
                ) => {
                    // We are trying to write to a specific byte in a word
                    // We need to load the word, mask out the byte, or in the new value
                    // and then store it back
                    let ptr = self.value_to_register(ptr, backend);
                    let val = self.value_to_register(val, backend);
                    let dest = self.value_to_register(temp1, backend);

                    // backend.breakpoint();
                    backend.load_from_heap(dest, ptr, *offset as i32);
                    let mask_register = self.value_to_register(temp2, backend);
                    backend.mov_64(mask_register, *mask as isize);
                    backend.and(dest, dest, mask_register);
                    backend.free_register(mask_register);
                    backend.shift_left_imm(val, val, (byte_offset * 8) as i32);
                    backend.or(dest, dest, val);
                    backend.store_on_heap(ptr, dest, *offset as i32);
                    backend.free_register(dest);
                }
                Instruction::HeapStoreOffsetReg(ptr, val, offset) => {
                    let ptr = self.value_to_register(ptr, backend);
                    let val = self.value_to_register(val, backend);
                    let offset = self.value_to_register(offset, backend);
                    backend.store_to_heap_with_reg_offset(ptr, val, offset);
                }
                Instruction::RegisterArgument(_arg) => {}
                Instruction::PushStack(val) => {
                    let val = self.value_to_register(val, backend);
                    backend.push_to_stack(val);
                }
                Instruction::PopStack(val) => {
                    let val = self.value_to_register(val, backend);
                    backend.pop_from_stack(val);
                }
                Instruction::GetStackPointer(dest, offset) => {
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    let offset = self.value_to_register(offset, backend);
                    backend.get_stack_pointer(dest, offset);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::GetStackPointerImm(dest, offset) => {
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.get_stack_pointer_imm(dest, *offset);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::GetFramePointer(dest) => {
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.mov_reg(dest, backend.frame_pointer());
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::ReadArgCount(dest) => {
                    // Read the argument count register (X9 for ARM64, R10 for x86-64)
                    // Used by variadic functions to determine how many arguments were passed
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.mov_reg(dest, backend.arg_count_reg());
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::CurrentStackPosition(dest) => {
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.get_current_stack_position(dest);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::GetTag(dest, value) => {
                    let value = self.value_to_register(value, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.get_tag(dest, value);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::Untag(dest, value) => {
                    let value = self.value_to_register(value, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.shift_right_imm(dest, value, BuiltInTypes::tag_size());
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::PushExceptionHandler(label, result_local, builtin_fn) => {
                    // Call push_exception_handler builtin
                    // Arguments: (handler_address, result_local_offset, link_register, stack_pointer, frame_pointer)

                    // Get the ARM64 label for the catch block
                    let catch_label = ir_label_to_lang_label.get(label).unwrap();

                    // Load the address of the catch label into arg 0
                    backend.load_label_address(backend.arg(0), *catch_label);

                    // Load result_local offset into arg 1
                    // result_local is a Value::Local(index)
                    // Use backend-specific method to get the correct byte offset
                    let local_index = result_local.as_local();
                    let local_offset = backend.get_local_byte_offset(local_index);
                    backend.mov_64(backend.arg(1), local_offset);

                    // Load return address into arg 2 BEFORE calling the builtin
                    // On ARM64 this copies from LR, on x86-64 it loads from [RBP + 8]
                    backend.load_return_address(backend.arg(2));

                    // Get stack pointer into arg 3
                    backend.get_stack_pointer_imm(backend.arg(3), 0);

                    // Copy frame pointer to arg 4
                    backend.mov_reg(backend.arg(4), backend.frame_pointer());

                    // Call the push_exception_handler builtin
                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);
                }
                Instruction::PopExceptionHandler(builtin_fn) => {
                    // Call pop_exception_handler builtin - no arguments
                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);
                }
                Instruction::Throw(value, builtin_fn) => {
                    // Call throw_exception builtin with stack pointer, frame pointer, and value
                    // Arguments: (stack_pointer, frame_pointer, exception_value)

                    // Load stack pointer into arg 0
                    backend.get_stack_pointer_imm(backend.arg(0), 0);

                    // Load frame pointer into arg 1
                    backend.mov_reg(backend.arg(1), backend.frame_pointer());

                    // Load exception value into arg 2
                    let value_reg = self.value_to_register(value, backend);
                    backend.mov_reg(backend.arg(2), value_reg);

                    // Call the throw_exception builtin (does not return)
                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);
                    // Note: execution never continues past this point
                }
            }
            let end_machine_code = backend.current_position();
            self.ir_to_machine_code_range.push((
                index,
                MachineCodeRange::new(start_machine_code, end_machine_code),
            ));
        }
    }

    pub fn breakpoint(&mut self) {
        self.instructions.push(Instruction::Breakpoint);
    }

    pub fn jump(&mut self, label: Label) {
        self.instructions.push(Instruction::Jump(label));
    }

    pub fn push_exception_handler(
        &mut self,
        handler: Label,
        result_local: Value,
        builtin_fn: usize,
    ) {
        self.instructions.push(Instruction::PushExceptionHandler(
            handler,
            result_local,
            builtin_fn,
        ));
    }

    pub fn pop_exception_handler(&mut self, builtin_fn: usize) {
        self.instructions
            .push(Instruction::PopExceptionHandler(builtin_fn));
    }

    pub fn throw_value(&mut self, value: Value, builtin_fn: usize) {
        self.instructions
            .push(Instruction::Throw(value, builtin_fn));
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
        let temp1 = self.assign_new(Value::RawValue(0));
        let temp2 = self.assign_new(Value::RawValue(0));
        self.instructions
            .push(Instruction::HeapStoreByteOffsetMasked(
                dest.into(),
                source.into(),
                temp1.into(),
                temp2.into(),
                offset,
                byte_offset,
                mask,
            ));
    }

    pub fn heap_load(&mut self, source: Value) -> Value {
        let source = self.assign_new(source);
        let dest = self.volatile_register();
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

    pub fn store_local(&mut self, local_index: usize, reg: Value) {
        self.increment_locals(local_index);
        self.instructions
            .push(Instruction::StoreLocal(Value::Local(local_index), reg));
    }

    pub fn load_local(&mut self, local_index: usize) -> Value {
        let reg = self.volatile_register();
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

    pub fn get_stack_pointer(&mut self, offset: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::GetStackPointer(dest, offset));
        dest
    }

    pub fn get_stack_pointer_imm(&mut self, index: isize) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::GetStackPointerImm(dest, index));
        dest
    }

    pub fn get_frame_pointer(&mut self) -> Value {
        let dest = self.volatile_register().into();
        self.instructions.push(Instruction::GetFramePointer(dest));
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

    pub fn write_float_literal(&mut self, float_pointer: Value, n: String) {
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
        let stack_pointer = self.get_stack_pointer_imm(0);
        let frame_pointer = self.get_frame_pointer();
        let f = self.assign_new(Value::Function(self.allocate_fn_pointer));
        self.call_builtin(f.into(), vec![stack_pointer, frame_pointer, size])
    }

    fn insert_label(&mut self, name: &str, label: Label) -> usize {
        let index = self.labels.len();
        assert!(index == label.index);
        self.labels.push(label);
        self.label_names.push(name.to_string());
        self.label_names.len() - 1
    }
}
