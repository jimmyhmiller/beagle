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
use crate::pretty_print::PrettyPrint;
use crate::register_allocation::linear_scan::LinearScan;
use crate::types::BuiltInTypes;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encode, Decode)]
pub enum Condition {
    LessThanOrEqual,
    LessThan,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

#[derive(Debug, Copy, Clone, PartialEq, Encode, Decode)]
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
            _ => panic!("Expected local, got {:?}", self),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct SavedValue {
    pub source: Value,
    pub local: usize,
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
    Sub(Value, Value, Value, Label),
    AddInt(Value, Value, Value),
    Mul(Value, Value, Value, Label),
    Div(Value, Value, Value, Label),
    Modulo(Value, Value, Value, Label),
    Assign(Value, Value),
    Recurse(Value, Vec<Value>),
    RecurseWithSaves(Value, Vec<Value>, Vec<SavedValue>),
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
    CallWithSaves(Value, Value, Vec<Value>, bool, Vec<SavedValue>),
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
    IntToFloat(Value, Value), // Convert tagged int to float register (SCVTF/CVTSI2SD)
    FRoundToZero(Value, Value), // Float round toward zero (FRINTZ/ROUNDSD)
    AddFloat(Value, Value, Value),
    SubFloat(Value, Value, Value),
    MulFloat(Value, Value, Value),
    DivFloat(Value, Value, Value),
    CompareFloat(Value, Value, Value, Condition),
    ShiftRightImm(Value, Value, i32, Label),
    ShiftRightImmRaw(Value, Value, i32),
    AndImm(Value, Value, u64),
    ShiftLeft(Value, Value, Value, Label),
    ShiftRight(Value, Value, Value, Label),
    ShiftRightZero(Value, Value, Value, Label),
    And(Value, Value, Value),
    Or(Value, Value, Value),
    Xor(Value, Value, Value),
    PushExceptionHandler(Label, Value, usize), // label, result_local, builtin_fn_ptr
    PushResumableExceptionHandler(Value, Label, Value, Value, usize), // dest (handler_id), catch_label, exception_local, resume_local, builtin_fn_ptr
    PopExceptionHandler(usize),                                       // builtin_fn_ptr
    PopExceptionHandlerById(Value, usize), // handler_id_value, builtin_fn_ptr
    Throw(Value, Label, usize, usize), // value, resume_label, resume_local_index, builtin_fn_ptr
    ReadArgCount(Value),               // Read arg count register (X9/R10) for variadic functions
    Label(Label),
    // Delimited continuation instructions
    PushPromptHandler(Label, Value, usize), // prompt_handler_label, result_local, builtin_fn_ptr
    PopPromptHandler(Value, usize),         // result_value, builtin_fn_ptr
    PushPromptTag(Value, Label, Value, usize), // tag_value, abort_label, result_local (as Local), builtin_fn_ptr
    LoadLabelAddress(Value, Label), // dest, label - loads the address of a label into a register
    CaptureContinuation(Value, Label, usize, usize), // dest, resume_label, result_local_index, builtin_fn_ptr
    CaptureContinuationWithSaves(Value, Label, usize, usize, Vec<SavedValue>), // dest, resume_label, result_local_index, builtin_fn_ptr, saved live roots
    CaptureContinuationTagged(Value, Label, usize, usize, Value), // dest, resume_label, result_local_index, builtin_fn_ptr, tag_value
    CaptureContinuationTaggedWithSaves(Value, Label, usize, usize, Value, Vec<SavedValue>), // dest, resume_label, result_local_index, builtin_fn_ptr, tag_value, saved live roots
    PerformEffect(Value, Value, Value, Label, usize, usize), // handler, enum_type, op_value, resume_label, result_local_index, builtin_fn_ptr
    PerformEffectWithSaves(Value, Value, Value, Label, usize, usize, Vec<SavedValue>), // handler, enum_type, op_value, resume_label, result_local_index, builtin_fn_ptr, saved live roots
    ReturnFromShift(Value, Value, usize), // value, cont_ptr, builtin_fn_ptr - calls return_from_shift with current SP/FP
    RecordGcSafepoint, // Record a GC safepoint at the current position (for continuation resume points)
    ReloadRootSlots(Vec<SavedValue>), // Reload live registers from root slots at continuation resume points
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
            Instruction::Sub(a, b, c, _) => {
                get_registers!(a, b, c)
            }
            Instruction::AddInt(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::Mul(a, b, c, _) => {
                get_registers!(a, b, c)
            }
            Instruction::Div(a, b, c, _) => {
                get_registers!(a, b, c)
            }
            Instruction::Modulo(a, b, c, _) => {
                get_registers!(a, b, c)
            }
            Instruction::ShiftLeft(a, b, c, _) => {
                get_registers!(a, b, c)
            }
            Instruction::ShiftRight(a, b, c, _) => {
                get_registers!(a, b, c)
            }
            Instruction::ShiftRightZero(a, b, c, _) => {
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
            Instruction::IntToFloat(a, b) => {
                get_registers!(a, b)
            }
            Instruction::FRoundToZero(a, b) => {
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
            Instruction::ShiftRightImm(a, b, _, _) => {
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
                    if let Ok(register) = (&save.source).try_into() {
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
                    if let Ok(register) = (&save.source).try_into() {
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
            Instruction::Breakpoint | Instruction::RecordGcSafepoint => {
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
            Instruction::PushResumableExceptionHandler(
                dest,
                _,
                exception_local,
                resume_local,
                _,
            ) => {
                get_registers!(dest, exception_local, resume_local)
            }
            Instruction::PopExceptionHandler(_) => {
                vec![]
            }
            Instruction::PopExceptionHandlerById(handler_id, _) => {
                get_register!(handler_id)
            }
            Instruction::Throw(value, _, _, _) => {
                get_register!(value)
            }
            Instruction::ReadArgCount(a) => {
                get_register!(a)
            }
            Instruction::PushPromptHandler(_, local, _) => {
                get_register!(local)
            }
            Instruction::PopPromptHandler(result, _) => {
                get_register!(result)
            }
            Instruction::PushPromptTag(tag, _, local, _) => {
                get_registers!(tag, local)
            }
            Instruction::LoadLabelAddress(dest, _) => {
                get_register!(dest)
            }
            Instruction::CaptureContinuation(dest, _, _, _) => {
                get_register!(dest)
            }
            Instruction::CaptureContinuationWithSaves(dest, _, _, _, saves) => {
                let mut result: Vec<VirtualRegister> = get_register!(dest);
                for save in saves {
                    if let Ok(register) = (&save.source).try_into() {
                        result.push(register);
                    }
                }
                result
            }
            Instruction::CaptureContinuationTagged(dest, _, _, _, tag) => {
                get_registers!(dest, tag)
            }
            Instruction::CaptureContinuationTaggedWithSaves(dest, _, _, _, tag, saves) => {
                let mut result: Vec<VirtualRegister> = get_registers!(dest, tag);
                for save in saves {
                    if let Ok(register) = (&save.source).try_into() {
                        result.push(register);
                    }
                }
                result
            }
            Instruction::PerformEffect(handler, enum_type, op_value, _, _, _) => {
                get_registers!(handler, enum_type, op_value)
            }
            Instruction::PerformEffectWithSaves(handler, enum_type, op_value, _, _, _, saves) => {
                let mut result: Vec<VirtualRegister> = get_registers!(handler, enum_type, op_value);
                for save in saves {
                    if let Ok(register) = (&save.source).try_into() {
                        result.push(register);
                    }
                }
                result
            }
            Instruction::ReturnFromShift(value, cont_ptr, _) => {
                get_registers!(value, cont_ptr)
            }
            Instruction::ReloadRootSlots(saves) => saves
                .iter()
                .filter_map(|save| (&save.source).try_into().ok())
                .collect(),
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

            Instruction::Sub(value, value1, value2, _)
            | Instruction::AddInt(value, value1, value2)
            | Instruction::Mul(value, value1, value2, _)
            | Instruction::Div(value, value1, value2, _)
            | Instruction::Modulo(value, value1, value2, _)
            | Instruction::HeapLoadReg(value, value1, value2)
            | Instruction::HeapStoreOffsetReg(value, value1, value2)
            | Instruction::ShiftLeft(value, value1, value2, _)
            | Instruction::ShiftRight(value, value1, value2, _)
            | Instruction::ShiftRightZero(value, value1, value2, _)
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
            | Instruction::IntToFloat(value, value1)
            | Instruction::FRoundToZero(value, value1)
            | Instruction::ShiftRightImm(value, value1, _, _)
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
                    replace_register!(&mut save.source, old_register, new_register);
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
                    if let Value::Register(register) = save.source
                        && register == old_register
                    {
                        save.source = new_register;
                    }
                }
            }

            Instruction::Jump(_) | Instruction::Breakpoint | Instruction::RecordGcSafepoint => {}
            Instruction::ReloadRootSlots(saves) => {
                for save in saves {
                    if let Value::Register(register) = save.source
                        && register == old_register
                    {
                        save.source = new_register;
                    }
                }
            }
            Instruction::PushExceptionHandler(_, value, _) => {
                replace_register!(value, old_register, new_register);
            }
            Instruction::PushResumableExceptionHandler(
                dest,
                _,
                exception_local,
                resume_local,
                _,
            ) => {
                replace_register!(dest, old_register, new_register);
                replace_register!(exception_local, old_register, new_register);
                replace_register!(resume_local, old_register, new_register);
            }
            Instruction::PopExceptionHandler(_) => {}
            Instruction::PopExceptionHandlerById(handler_id, _) => {
                replace_register!(handler_id, old_register, new_register);
            }
            Instruction::Throw(value, _, _, _) => {
                replace_register!(value, old_register, new_register);
            }
            Instruction::PushPromptHandler(_, value, _) => {
                replace_register!(value, old_register, new_register);
            }
            Instruction::PopPromptHandler(result, _) => {
                replace_register!(result, old_register, new_register);
            }
            Instruction::PushPromptTag(tag, _, value, _) => {
                replace_register!(tag, old_register, new_register);
                replace_register!(value, old_register, new_register);
            }
            Instruction::LoadLabelAddress(dest, _) => {
                replace_register!(dest, old_register, new_register);
            }
            Instruction::CaptureContinuation(dest, _, _, _) => {
                replace_register!(dest, old_register, new_register);
            }
            Instruction::CaptureContinuationWithSaves(dest, _, _, _, saves) => {
                replace_register!(dest, old_register, new_register);
                for save in saves {
                    if let Value::Register(register) = save.source
                        && register == old_register
                    {
                        save.source = new_register;
                    }
                }
            }
            Instruction::CaptureContinuationTagged(dest, _, _, _, tag) => {
                replace_register!(dest, old_register, new_register);
                replace_register!(tag, old_register, new_register);
            }
            Instruction::CaptureContinuationTaggedWithSaves(dest, _, _, _, tag, saves) => {
                replace_register!(dest, old_register, new_register);
                replace_register!(tag, old_register, new_register);
                for save in saves {
                    if let Value::Register(register) = save.source
                        && register == old_register
                    {
                        save.source = new_register;
                    }
                }
            }
            Instruction::PerformEffect(handler, enum_type, op_value, _, _, _) => {
                replace_register!(handler, old_register, new_register);
                replace_register!(enum_type, old_register, new_register);
                replace_register!(op_value, old_register, new_register);
            }
            Instruction::PerformEffectWithSaves(handler, enum_type, op_value, _, _, _, saves) => {
                replace_register!(handler, old_register, new_register);
                replace_register!(enum_type, old_register, new_register);
                replace_register!(op_value, old_register, new_register);
                for save in saves {
                    if let Value::Register(register) = save.source
                        && register == old_register
                    {
                        save.source = new_register;
                    }
                }
            }
            Instruction::ReturnFromShift(value, cont_ptr, _) => {
                replace_register!(value, old_register, new_register);
                replace_register!(cont_ptr, old_register, new_register);
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
    pub debug_name: Option<String>,
    labels: Vec<Label>,
    label_names: Vec<String>,
    label_locations: HashMap<usize, usize>,
    pub num_locals: usize,
    allocate_fn_pointer: usize,
    pub error_fn_pointer: usize,
    pub ir_to_machine_code_range: Vec<(usize, MachineCodeRange)>,
    pub ir_range_to_token_range: Vec<(crate::ast::TokenRange, IRRange)>,
    /// Number of argument registers for the target architecture (8 for ARM64, 6 for x86-64)
    num_arg_registers: usize,
    /// If this function uses `binding`, the local index of the mark pointer.
    pub mark_local_index: Option<usize>,
    /// Stack of local indices used by push_to_stack/pop_from_stack.
    /// Values are stored in root-slotted locals instead of the eval stack.
    local_stack: Vec<usize>,
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

        Self {
            register_index: 0,
            instructions: vec![],
            debug_name: None,
            labels: vec![],
            label_names: vec![],
            label_locations: HashMap::new(),
            num_locals: 0,
            allocate_fn_pointer,
            error_fn_pointer: 0,
            ir_to_machine_code_range: vec![],
            ir_range_to_token_range: vec![],
            num_arg_registers,
            mark_local_index: None,
            local_stack: vec![],
        }
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
        let error_label = self.label("sub_type_error");
        let after_label = self.label("sub_after");
        self.instructions.push(Instruction::Sub(
            result.into(),
            a.into(),
            b.into(),
            error_label,
        ));
        self.jump(after_label);
        self.write_label(error_label);
        self.emit_type_error_with_resume(result, after_label);
        self.write_label(after_label);
        Value::Register(result)
    }

    pub fn math_any<A, B, F, G>(&mut self, a: A, b: B, op_int: F, op_float: G) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
        F: FnOnce(&mut Ir, Value, Value) -> Value,
        G: Fn(&mut Ir, Value, Value) -> Value,
    {
        let result_register = self.assign_new(Value::TaggedConstant(0));
        let a: VirtualRegister = self.assign_new(a.into());
        let b: VirtualRegister = self.assign_new(b.into());

        let slow_path: Label = self.label("slow_path");
        let after_op = self.label("after_op");

        // Fast path: Check if both are ints (most common case)
        self.guard_int(a.into(), slow_path);
        self.guard_int(b.into(), slow_path);

        // Both are ints - do integer operation (fast path)
        let result = op_int(self, a.into(), b.into());
        self.assign(result_register, result);
        self.jump(after_op);

        // Slow path: handle floats and mixed types
        self.write_label(slow_path);
        let float_result = self.math_any_slow_path(a, b, &op_float);
        self.assign(result_register, float_result);

        self.write_label(after_op);
        Value::Register(result_register)
    }

    /// Slow path for math operations involving floats.
    /// Handles: float+float, int+float, float+int
    fn math_any_slow_path<G>(
        &mut self,
        a: VirtualRegister,
        b: VirtualRegister,
        op_float: &G,
    ) -> Value
    where
        G: Fn(&mut Ir, Value, Value) -> Value,
    {
        let result_register = self.assign_new(Value::TaggedConstant(0));
        let a_is_float: Label = self.label("a_is_float");
        let both_floats: Label = self.label("both_floats");
        let after_slow: Label = self.label("after_slow");
        let type_error_1: Label = self.label("type_error_1");
        let type_error_2: Label = self.label("type_error_2");
        let type_error_3: Label = self.label("type_error_3");

        // Check if a is an int (we know at least one operand is not an int from fast path)
        self.guard_int(a.into(), a_is_float);

        // a is int, so b must be float (since we failed the fast path)
        self.guard_float(b.into(), type_error_1);

        // Case: a is int, b is float - convert a to float
        {
            // Allocate result heap object BEFORE computing the float result.
            // This ensures no raw f64 values are live in GPRs across the
            // allocation call, which could cause GC to misinterpret them
            // as tagged heap pointers.
            let size_reg = self.assign_new(1);
            let float_pointer = self.allocate(size_reg.into());
            let float_pointer_untagged = self.untag(float_pointer);
            self.write_small_object_header(float_pointer_untagged);
            let a_untagged = self.shift_right_imm_raw(a.into(), 3);
            let a_float = self.int_to_float(a_untagged);
            let b_untagged = self.untag(b.into());
            let b_val = self.load_from_heap(b_untagged, 1);
            let b_float = self.fmov_general_to_float(b_val);
            let float_result = op_float(self, a_float, b_float);
            let float_result_general = self.fmov_float_to_general(float_result);
            self.heap_store_offset(float_pointer_untagged, float_result_general, 1);
            let tagged = self.tag(float_pointer_untagged, BuiltInTypes::Float.get_tag());
            self.assign(result_register, tagged);
            self.jump(after_slow);
        }

        // a is not int - check if a is float
        self.write_label(a_is_float);
        self.guard_float(a.into(), type_error_2);

        // a is float - check if b is int or float
        self.guard_int(b.into(), both_floats);

        // Case: a is float, b is int - convert b to float
        {
            // Allocate before float computation to avoid raw f64 in GPRs across GC safepoint
            let size_reg = self.assign_new(1);
            let float_pointer = self.allocate(size_reg.into());
            let float_pointer_untagged = self.untag(float_pointer);
            self.write_small_object_header(float_pointer_untagged);
            let a_untagged = self.untag(a.into());
            let a_val = self.load_from_heap(a_untagged, 1);
            let a_float = self.fmov_general_to_float(a_val);
            let b_untagged = self.shift_right_imm_raw(b.into(), 3);
            let b_float = self.int_to_float(b_untagged);
            let float_result = op_float(self, a_float, b_float);
            let float_result_general = self.fmov_float_to_general(float_result);
            self.heap_store_offset(float_pointer_untagged, float_result_general, 1);
            let tagged = self.tag(float_pointer_untagged, BuiltInTypes::Float.get_tag());
            self.assign(result_register, tagged);
            self.jump(after_slow);
        }

        // Case: Both are floats
        self.write_label(both_floats);
        self.guard_float(b.into(), type_error_3);

        {
            // Allocate before float computation to avoid raw f64 in GPRs across GC safepoint
            let size_reg = self.assign_new(1);
            let float_pointer = self.allocate(size_reg.into());
            let float_pointer_untagged = self.untag(float_pointer);
            self.write_small_object_header(float_pointer_untagged);
            let a_untagged = self.untag(a.into());
            let b_untagged = self.untag(b.into());
            let a_val = self.load_from_heap(a_untagged, 1);
            let b_val = self.load_from_heap(b_untagged, 1);
            let a_float = self.fmov_general_to_float(a_val);
            let b_float = self.fmov_general_to_float(b_val);
            let float_result = op_float(self, a_float, b_float);
            let float_result_general = self.fmov_float_to_general(float_result);
            self.heap_store_offset(float_pointer_untagged, float_result_general, 1);
            let tagged = self.tag(float_pointer_untagged, BuiltInTypes::Float.get_tag());
            self.assign(result_register, tagged);
        }

        self.write_label(after_slow);

        // Emit type error handlers at the end (keeps the hot path compact)
        let after_errors: Label = self.label("after_type_errors");
        self.jump(after_errors);

        self.write_label(type_error_1);
        self.emit_type_error_with_resume(result_register, after_slow);

        self.write_label(type_error_2);
        self.emit_type_error_with_resume(result_register, after_slow);

        self.write_label(type_error_3);
        self.emit_type_error_with_resume(result_register, after_slow);

        self.write_label(after_errors);
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
        let error_label = self.label("mul_type_error");
        let after_label = self.label("mul_after");
        self.instructions.push(Instruction::Mul(
            register.into(),
            a.into(),
            b.into(),
            error_label,
        ));
        self.jump(after_label);
        self.write_label(error_label);
        self.emit_type_error_with_resume(register, after_label);
        self.write_label(after_label);
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
        let error_label = self.label("div_type_error");
        let after_label = self.label("div_after");
        self.instructions.push(Instruction::Div(
            register.into(),
            a.into(),
            b.into(),
            error_label,
        ));
        self.jump(after_label);
        self.write_label(error_label);
        self.emit_type_error_with_resume(register, after_label);
        self.write_label(after_label);
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
        let error_label = self.label("modulo_type_error");
        let after_label = self.label("modulo_after");
        self.instructions.push(Instruction::Modulo(
            register.into(),
            a.into(),
            b.into(),
            error_label,
        ));
        self.jump(after_label);
        self.write_label(error_label);
        self.emit_type_error_with_resume(register, after_label);
        self.write_label(after_label);
        Value::Register(register)
    }

    pub fn modulo_any<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        self.math_any(a, b, Self::modulo, Self::modulo_float)
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
        let default_compare_label: Label = self.label("default_compare");
        let after_compare = self.label("after_compare");

        // Fast path: check if both are ints
        let not_both_ints: Label = self.label("not_both_ints");
        self.guard_int(a.into(), not_both_ints);
        self.guard_int(b.into(), not_both_ints);

        // Both ints - integer comparison (fast path)
        {
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
            self.jump(after_compare);
        }

        // Slow path: handle floats and mixed int/float types
        self.write_label(not_both_ints);

        let a_is_float: Label = self.label("cmp_a_is_float");
        let both_floats: Label = self.label("cmp_both_floats");

        // Check if a is int (we know at least one is not int from fast path)
        self.guard_int(a.into(), a_is_float);

        // a is int, so b must be float (since fast path failed)
        self.guard_float(b.into(), default_compare_label);

        // Case: a is int, b is float - convert a to float and compare
        {
            let a_untagged = self.shift_right_imm_raw(a.into(), 3);
            let a_float = self.int_to_float(a_untagged);
            let b_untagged = self.untag(b.into());
            let b_val = self.load_from_heap(b_untagged, 1);
            let b_float = self.fmov_general_to_float(b_val);
            let tag = self.assign_new(Value::RawValue(BuiltInTypes::Bool.get_tag() as usize));
            let dest = self.volatile_register();
            self.instructions.push(Instruction::CompareFloat(
                dest.into(),
                a_float,
                b_float,
                condition,
            ));
            self.instructions
                .push(Instruction::Tag(dest.into(), dest.into(), tag.into()));
            self.assign(result_register, dest);
            self.jump(after_compare);
        }

        // a is not int - check if a is float
        self.write_label(a_is_float);
        self.guard_float(a.into(), default_compare_label);

        // a is float - check if b is int or float
        self.guard_int(b.into(), both_floats);

        // Case: a is float, b is int - convert b to float and compare
        {
            let a_untagged = self.untag(a.into());
            let a_val = self.load_from_heap(a_untagged, 1);
            let a_float = self.fmov_general_to_float(a_val);
            let b_untagged = self.shift_right_imm_raw(b.into(), 3);
            let b_float = self.int_to_float(b_untagged);
            let tag = self.assign_new(Value::RawValue(BuiltInTypes::Bool.get_tag() as usize));
            let dest = self.volatile_register();
            self.instructions.push(Instruction::CompareFloat(
                dest.into(),
                a_float,
                b_float,
                condition,
            ));
            self.instructions
                .push(Instruction::Tag(dest.into(), dest.into(), tag.into()));
            self.assign(result_register, dest);
            self.jump(after_compare);
        }

        // Case: both are floats
        self.write_label(both_floats);
        self.guard_float(b.into(), default_compare_label);

        {
            let a_untagged = self.untag(a.into());
            let b_untagged = self.untag(b.into());
            let a_raw = self.load_from_heap(a_untagged, 1);
            let b_raw = self.load_from_heap(b_untagged, 1);
            let a_float = self.fmov_general_to_float(a_raw);
            let b_float = self.fmov_general_to_float(b_raw);
            let tag = self.assign_new(Value::RawValue(BuiltInTypes::Bool.get_tag() as usize));
            let dest = self.volatile_register();
            self.instructions.push(Instruction::CompareFloat(
                dest.into(),
                a_float,
                b_float,
                condition,
            ));
            self.instructions
                .push(Instruction::Tag(dest.into(), dest.into(), tag.into()));
            self.assign(result_register, dest);
            self.jump(after_compare);
        }

        // Default path - neither operand is a number, or non-numeric types
        self.write_label(default_compare_label);
        let cmp_type_error_label = match condition {
            Condition::LessThan
            | Condition::LessThanOrEqual
            | Condition::GreaterThan
            | Condition::GreaterThanOrEqual => {
                let tag_a = self.get_tag(a.into());
                let tag_b = self.get_tag(b.into());
                let cmp_type_error = self.label("compare_type_error");
                self.jump_if(cmp_type_error, Condition::NotEqual, tag_a, tag_b);
                Some(cmp_type_error)
            }
            Condition::Equal | Condition::NotEqual => {
                // Equality/inequality can compare any types - no guard needed
                None
            }
        };
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
        if let Some(cmp_type_error) = cmp_type_error_label {
            self.jump(after_compare);
            self.write_label(cmp_type_error);
            self.emit_type_error_with_resume(result_register, after_compare);
        }

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
        let error_label = self.label("shr_imm_type_error");
        let after_label = self.label("shr_imm_after");
        self.instructions.push(Instruction::ShiftRightImm(
            destination.into(),
            a.into(),
            b,
            error_label,
        ));
        self.jump(after_label);
        self.write_label(error_label);
        self.emit_type_error_with_resume(destination, after_label);
        self.write_label(after_label);
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
        let error_label = self.label("shl_type_error");
        let after_label = self.label("shl_after");
        self.instructions.push(Instruction::ShiftLeft(
            destination.into(),
            a.into(),
            b.into(),
            error_label,
        ));
        self.jump(after_label);
        self.write_label(error_label);
        self.emit_type_error_with_resume(destination, after_label);
        self.write_label(after_label);
        destination.into()
    }

    pub fn shift_right(&mut self, a: Value, b: Value) -> Value {
        let a = self.assign_new(a);
        let b = self.assign_new(b);
        let destination = self.volatile_register();
        let error_label = self.label("shr_type_error");
        let after_label = self.label("shr_after");
        self.instructions.push(Instruction::ShiftRight(
            destination.into(),
            a.into(),
            b.into(),
            error_label,
        ));
        self.jump(after_label);
        self.write_label(error_label);
        self.emit_type_error_with_resume(destination, after_label);
        self.write_label(after_label);
        destination.into()
    }

    pub fn shift_right_zero(&mut self, a: Value, b: Value) -> Value {
        let a = self.assign_new(a);
        let b = self.assign_new(b);
        let destination = self.volatile_register();
        let error_label = self.label("shrz_type_error");
        let after_label = self.label("shrz_after");
        self.instructions.push(Instruction::ShiftRightZero(
            destination.into(),
            a.into(),
            b.into(),
            error_label,
        ));
        self.jump(after_label);
        self.write_label(error_label);
        self.emit_type_error_with_resume(destination, after_label);
        self.write_label(after_label);
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

    pub fn compile<B: CodegenBackend>(&mut self, mut backend: B, _error_fn_pointer: usize) -> B {
        debug_assert!(!self.ir_range_to_token_range.is_empty());

        // backend.breakpoint();

        let mut linear_scan = LinearScan::new(self.instructions.clone(), self.num_locals);
        linear_scan.allocate();
        #[cfg(debug_assertions)]
        {
            if std::env::var("BEAGLE_DEBUG_ROOT_SLOTS").is_ok() {
                eprintln!(
                    "[root-slots] function={} pre_num_locals={} post_stack_slot={} instructions={}",
                    self.debug_name.as_deref().unwrap_or("<anonymous>"),
                    self.num_locals,
                    linear_scan.stack_slot,
                    linear_scan.instructions.len()
                );
                let mut entries: Vec<_> = linear_scan
                    .root_slots
                    .iter()
                    .map(|(reg, slot)| {
                        let lifetime = linear_scan.lifetimes.get(reg).copied().unwrap_or((0, 0));
                        (
                            *slot,
                            reg.index,
                            reg.argument,
                            reg.is_physical,
                            lifetime.0,
                            lifetime.1,
                        )
                    })
                    .collect();
                entries.sort();
                for (slot, reg_index, argument, is_physical, start, end) in entries {
                    eprintln!(
                        "[root-slot] slot={} reg={} arg={:?} physical={} lifetime={}..{}",
                        slot, reg_index, argument, is_physical, start, end
                    );
                }
            }
            if std::env::var("BEAGLE_DEBUG_POST_ALLOC_IR").is_ok() {
                eprintln!(
                    "[post-alloc-ir] function={}",
                    self.debug_name.as_deref().unwrap_or("<anonymous>")
                );
                for (index, instruction) in linear_scan.instructions.iter().enumerate() {
                    eprintln!("{:04}: {}", index, instruction.pretty_print());
                }
            }
        }

        self.instructions = linear_scan.instructions.clone();
        self.num_locals = linear_scan.stack_slot;
        backend.set_max_locals(self.num_locals);
        if let Some(mark_idx) = self.mark_local_index {
            backend.set_mark_local_index(mark_idx);
        }

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
            Value::TaggedConstant(val) => {
                let temp_reg = backend.temporary_register();
                backend.mov_64(temp_reg, *val);
                temp_reg
            }
            Value::True => {
                let temp_reg = backend.temporary_register();
                backend.mov_64(temp_reg, BuiltInTypes::true_value());
                temp_reg
            }
            Value::False => {
                let temp_reg = backend.temporary_register();
                backend.mov_64(temp_reg, BuiltInTypes::false_value());
                temp_reg
            }
            Value::Null => {
                let temp_reg = backend.temporary_register();
                backend.mov_64(temp_reg, BuiltInTypes::null_value());
                temp_reg
            }
            Value::StringConstantPtr(val) | Value::KeywordConstantPtr(val) => {
                let temp_reg = backend.temporary_register();
                backend.mov_64(temp_reg, *val as isize);
                temp_reg
            }
            Value::Function(val) | Value::Pointer(val) => {
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

        // Build a map from resume labels to their saved registers. When we
        // encounter these labels during codegen, we emit reload instructions
        // so that registers are restored from root slots after continuation resume.
        let mut resume_label_saves: HashMap<Label, Vec<SavedValue>> = HashMap::new();
        for instruction in self.instructions.iter() {
            match instruction {
                Instruction::CaptureContinuationWithSaves(_, label, _, _, saves)
                    if !saves.is_empty() =>
                {
                    resume_label_saves.insert(*label, saves.clone());
                }
                Instruction::CaptureContinuationTaggedWithSaves(_, label, _, _, _, saves)
                    if !saves.is_empty() =>
                {
                    resume_label_saves.insert(*label, saves.clone());
                }
                Instruction::PerformEffectWithSaves(_, _, _, label, _, _, saves)
                    if !saves.is_empty() =>
                {
                    resume_label_saves.insert(*label, saves.clone());
                }
                _ => {}
            }
        }

        for (index, instruction) in self.instructions.iter().enumerate() {
            let start_machine_code = backend.current_position();
            let label = self.label_locations.get(&index);
            if let Some(label) = label {
                backend.write_label(ir_label_to_lang_label[&self.labels[*label]]);
                // At continuation resume labels, reload live registers from
                // their root slots. The stack segment was restored with
                // GC-updated values in the root slots.
                if let Some(saves) = resume_label_saves.get(&self.labels[*label]) {
                    let null_reg = self.value_to_register(&Value::Null, backend);
                    for save in saves.iter().rev() {
                        let save_reg = self.value_to_register(&save.source, backend);
                        backend.load_local(save_reg, save.local as i32);
                        backend.store_local(null_reg, save.local as i32);
                    }
                    backend.free_temporary_register(null_reg);
                }
            }
            backend.clear_temporary_registers();
            // println!("instruction {:?}", instruction);
            match instruction {
                Instruction::Breakpoint => {
                    backend.breakpoint();
                }
                Instruction::RecordGcSafepoint => {
                    // No-op: stack maps have been removed; GC scans all root
                    // slots in the frame header instead.
                }
                Instruction::ReloadRootSlots(_) => {
                    // Reloads are emitted at the label site (above) rather than
                    // here, because the label write happens before the instruction
                    // match. This arm is a no-op — the instruction exists only to
                    // satisfy exhaustive matching.
                }
                Instruction::Label(_) => {}
                Instruction::ExtendLifeTime(_) => {}
                Instruction::Sub(dest, a, b, error_label) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    let error_label = ir_label_to_lang_label[error_label];

                    // Use a temporary register for guard checks to avoid clobbering operands
                    // if dest happens to be the same register as a or b
                    let guard_temp = backend.temporary_register();
                    backend.guard_integer(guard_temp, a, error_label);
                    backend.guard_integer(guard_temp, b, error_label);
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
                Instruction::Mul(dest, a, b, error_label) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    let error_label = ir_label_to_lang_label[error_label];

                    // Use a temporary register for guard checks to avoid clobbering operands
                    // if dest happens to be the same register as a or b
                    let guard_temp = backend.temporary_register();
                    backend.guard_integer(guard_temp, a, error_label);
                    backend.guard_integer(guard_temp, b, error_label);
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
                Instruction::Div(dest, a, b, error_label) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    let error_label = ir_label_to_lang_label[error_label];

                    // Use a temporary register for guard checks to avoid clobbering operands
                    // if dest happens to be the same register as a or b
                    let guard_temp = backend.temporary_register();
                    backend.guard_integer(guard_temp, a, error_label);
                    backend.guard_integer(guard_temp, b, error_label);
                    backend.free_temporary_register(guard_temp);

                    backend.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    // Check for division by zero
                    let zero_reg = backend.temporary_register();
                    backend.mov_64(zero_reg, 0);
                    backend.compare(b, zero_reg);
                    backend.free_temporary_register(zero_reg);
                    backend.jump_equal(error_label);
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
                Instruction::Modulo(dest, a, b, error_label) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    let error_label = ir_label_to_lang_label[error_label];

                    // Use a temporary register for guard checks
                    let guard_temp = backend.temporary_register();
                    backend.guard_integer(guard_temp, a, error_label);
                    backend.guard_integer(guard_temp, b, error_label);
                    backend.free_temporary_register(guard_temp);

                    backend.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    // Check for division by zero
                    let zero_reg = backend.temporary_register();
                    backend.mov_64(zero_reg, 0);
                    backend.compare(b, zero_reg);
                    backend.free_temporary_register(zero_reg);
                    backend.jump_equal(error_label);
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
                Instruction::ShiftRightImm(dest, value, shift, error_label) => {
                    let value = self.value_to_register(value, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    let error_label = ir_label_to_lang_label[error_label];

                    backend.guard_integer(dest, value, error_label);

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
                Instruction::ShiftLeft(dest, a, b, error_label) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    let error_label = ir_label_to_lang_label[error_label];

                    backend.guard_integer(dest, a, error_label);
                    backend.guard_integer(dest, b, error_label);

                    backend.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    backend.shift_left(dest, a, b);
                    backend.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    backend.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_left_imm(b, b, BuiltInTypes::tag_size());
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::ShiftRight(dest, a, b, error_label) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    let error_label = ir_label_to_lang_label[error_label];

                    backend.guard_integer(dest, a, error_label);
                    backend.guard_integer(dest, b, error_label);

                    backend.shift_right_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_right_imm(b, b, BuiltInTypes::tag_size());
                    backend.shift_right(dest, a, b);
                    backend.shift_left_imm(dest, dest, BuiltInTypes::tag_size());
                    backend.shift_left_imm(a, a, BuiltInTypes::tag_size());
                    backend.shift_left_imm(b, b, BuiltInTypes::tag_size());
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::ShiftRightZero(dest, a, b, error_label) => {
                    let a = self.value_to_register(a, backend);
                    let b = self.value_to_register(b, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    let error_label = ir_label_to_lang_label[error_label];

                    backend.guard_integer(dest, a, error_label);
                    backend.guard_integer(dest, b, error_label);

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
                Instruction::IntToFloat(dest, src) => {
                    let src = self.value_to_register(src, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.int_to_float(dest, src);
                    self.store_spill(dest, dest_spill, backend);
                }
                Instruction::FRoundToZero(dest, src) => {
                    let src = self.value_to_register(src, backend);
                    let dest_spill = self.dest_spill(dest);
                    let dest = self.value_to_register(dest, backend);
                    backend.frintz(dest, src);
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
                    // Save registers to their root slots (frame locals)
                    for save in saves.iter() {
                        let save_reg = self.value_to_register(&save.source, backend);
                        backend.store_local(save_reg, save.local as i32);
                    }

                    // Set up arguments
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

                    // Restore saved registers from their root slots
                    let null_reg = self.value_to_register(&Value::Null, backend);
                    for save in saves.iter().rev() {
                        let save_reg = self.value_to_register(&save.source, backend);
                        backend.load_local(save_reg, save.local as i32);
                        backend.store_local(null_reg, save.local as i32);
                    }
                    backend.free_temporary_register(null_reg);
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
                    // Check if function is in a register that will be saved
                    let function_in_save = if let Value::Register(reg) = function {
                        saves.iter().any(|save| {
                            if let Value::Register(save_reg) = save.source {
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

                    // Save registers that are live across the call into frame locals
                    // so they are part of the GC-traced root set.
                    let mut null_save_reg = None;
                    for save in saves.iter() {
                        let save_reg = if matches!(save.source, Value::Null) {
                            *null_save_reg.get_or_insert_with(|| {
                                self.value_to_register(&Value::Null, backend)
                            })
                        } else {
                            self.value_to_register(&save.source, backend)
                        };
                        backend.store_local(save_reg, save.local as i32);
                    }
                    if let Some(reg) = null_save_reg {
                        backend.free_temporary_register(reg);
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

                    let result_reg = backend.temporary_register();
                    backend.mov_reg(result_reg, backend.ret_reg());

                    // Restore saved registers from their tracked local slots.
                    let null_reg = self.value_to_register(&Value::Null, backend);
                    for save in saves.iter().rev() {
                        if matches!(save.source, Value::Register(_)) {
                            let save_reg = self.value_to_register(&save.source, backend);
                            backend.load_local(save_reg, save.local as i32);
                        }
                        backend.store_local(null_reg, save.local as i32);
                    }
                    backend.free_temporary_register(null_reg);

                    let dest_spill = self.dest_spill(dest);
                    let dest_reg = match dest {
                        Value::Register(register) => backend.register_from_index(register.index),
                        Value::Spill(_, _) => backend.temporary_register(),
                        _ => panic!("Unexpected dest type for call: {:?}", dest),
                    };
                    backend.mov_reg(dest_reg, result_reg);
                    self.store_spill(dest_reg, dest_spill, backend);
                    backend.free_temporary_register(result_reg);
                    if matches!(dest, Value::Spill(_, _)) {
                        backend.free_temporary_register(dest_reg);
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
                Instruction::Throw(value, resume_label, resume_local_index, builtin_fn) => {
                    // Call throw_exception builtin with stack pointer, frame pointer, value,
                    // resume_address, and resume_local_offset
                    // Arguments: (stack_pointer, frame_pointer, exception_value, resume_address, resume_local_offset)

                    // Load stack pointer into arg 0
                    backend.get_stack_pointer_imm(backend.arg(0), 0);

                    // Load frame pointer into arg 1
                    backend.mov_reg(backend.arg(1), backend.frame_pointer());

                    // Load exception value into arg 2
                    let value_reg = self.value_to_register(value, backend);
                    backend.mov_reg(backend.arg(2), value_reg);

                    // Load address of resume label into arg 3
                    let resume_backend_label = ir_label_to_lang_label.get(resume_label).unwrap();
                    backend.load_label_address(backend.arg(3), *resume_backend_label);

                    // Load resume_local byte offset into arg 4
                    let local_offset = backend.get_local_byte_offset(*resume_local_index);
                    backend.mov_64(backend.arg(4), local_offset);

                    // Call the throw_exception builtin (does not return normally)
                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);
                    // Note: execution continues at resume_label only if handler is resumable and resume is called
                }
                Instruction::PushResumableExceptionHandler(
                    dest,
                    label,
                    exception_local,
                    resume_local,
                    builtin_fn,
                ) => {
                    // Call push_resumable_exception_handler builtin
                    // Arguments: (handler_address, result_local_offset, resume_local_offset, link_register, stack_pointer, frame_pointer)
                    // Returns: handler_id (tagged int)

                    let catch_label = ir_label_to_lang_label.get(label).unwrap();
                    backend.load_label_address(backend.arg(0), *catch_label);

                    let exception_index = exception_local.as_local();
                    let exception_offset = backend.get_local_byte_offset(exception_index);
                    backend.mov_64(backend.arg(1), exception_offset);

                    let resume_index = resume_local.as_local();
                    let resume_offset = backend.get_local_byte_offset(resume_index);
                    backend.mov_64(backend.arg(2), resume_offset);

                    backend.load_return_address(backend.arg(3));
                    backend.get_stack_pointer_imm(backend.arg(4), 0);
                    backend.mov_reg(backend.arg(5), backend.frame_pointer());

                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);

                    // Store returned handler_id in dest
                    let dest_spill = self.dest_spill(dest);
                    let dest_reg = match dest {
                        Value::Register(register) => backend.register_from_index(register.index),
                        Value::Spill(_, _) => backend.temporary_register(),
                        _ => panic!(
                            "Unexpected dest type for PushResumableExceptionHandler: {:?}",
                            dest
                        ),
                    };
                    backend.mov_reg(dest_reg, backend.ret_reg());
                    self.store_spill(dest_reg, dest_spill, backend);
                    if matches!(dest, Value::Spill(_, _)) {
                        backend.free_temporary_register(dest_reg);
                    }
                }
                Instruction::PopExceptionHandlerById(handler_id, builtin_fn) => {
                    let handler_id_reg = self.value_to_register(handler_id, backend);
                    backend.mov_reg(backend.arg(0), handler_id_reg);

                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);
                }
                Instruction::PushPromptHandler(label, result_local, builtin_fn) => {
                    // Call push_prompt builtin
                    // Arguments: (handler_address, result_local_offset, link_register, stack_pointer, frame_pointer)

                    // Get the backend label for the prompt handler
                    let prompt_label = ir_label_to_lang_label.get(label).unwrap();

                    // Load the address of the prompt handler label into arg 0
                    backend.load_label_address(backend.arg(0), *prompt_label);

                    // Load result_local offset into arg 1
                    let local_index = result_local.as_local();
                    let local_offset = backend.get_local_byte_offset(local_index);
                    backend.mov_64(backend.arg(1), local_offset);

                    // Load return address into arg 2
                    backend.load_return_address(backend.arg(2));

                    // Get stack pointer into arg 3
                    backend.get_stack_pointer_imm(backend.arg(3), 0);

                    // Copy frame pointer to arg 4
                    backend.mov_reg(backend.arg(4), backend.frame_pointer());

                    // Call the push_prompt builtin
                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);

                    // The builtin returns the active stack pointer for the prompt body.
                    // For segmented-stack execution this may differ from the caller's SP.
                    backend.mov_reg(backend.stack_pointer_reg(), backend.ret_reg());
                }
                Instruction::PopPromptHandler(result_value, builtin_fn) => {
                    // Call pop_prompt builtin with result value as argument
                    // arg0 = stack_pointer, arg1 = frame_pointer, arg2 = result_value

                    // Get stack pointer into arg0
                    backend.get_stack_pointer_imm(backend.arg(0), 0);

                    // Get frame pointer into arg1
                    backend.mov_reg(backend.arg(1), backend.frame_pointer());

                    // Get result value into arg2
                    let result_reg = self.value_to_register(result_value, backend);
                    backend.mov_reg(backend.arg(2), result_reg);

                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);

                    // Restore the caller-visible stack pointer on prompt exit.
                    backend.mov_reg(backend.stack_pointer_reg(), backend.ret_reg());
                }
                Instruction::PushPromptTag(tag_value, abort_label, result_local, builtin_fn) => {
                    // Call push_prompt_tag builtin with the record fields the
                    // tagged return/capture flow reads on perform/longjmp.
                    // Arguments: (tag, stack_pointer, frame_pointer, link_register,
                    //             result_local_offset)

                    // arg 0: tag (prompt id)
                    let tag_reg = self.value_to_register(tag_value, backend);
                    backend.mov_reg(backend.arg(0), tag_reg);

                    // arg 1: stack_pointer (caller's SP — the SP that the
                    // longjmp target expects to restore).
                    backend.get_stack_pointer_imm(backend.arg(1), 0);

                    // arg 2: frame_pointer (caller's FP — likewise the FP
                    // that the longjmp target expects).
                    backend.mov_reg(backend.arg(2), backend.frame_pointer());

                    // arg 3: link_register (address of the abort-landing
                    // label; return_from_shift_tagged jumps here with
                    // X0 = handler's return value).
                    let abort_backend_label = ir_label_to_lang_label.get(abort_label).unwrap();
                    backend.load_label_address(backend.arg(3), *abort_backend_label);

                    // arg 4: result_local byte offset (signed). 0 means
                    // "no local write"; any other value is interpreted
                    // relative to frame_pointer by return_from_shift_tagged.
                    let local_index = result_local.as_local();
                    let local_offset = backend.get_local_byte_offset(local_index);
                    backend.mov_64(backend.arg(4), local_offset);

                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);
                    // push-prompt-tag returns the caller's SP unchanged;
                    // discard the result — the SP register already has
                    // the live value.
                }
                Instruction::LoadLabelAddress(dest, label) => {
                    // Load the address of a label into a register
                    let dest_reg = self.value_to_register(dest, backend);
                    let backend_label = ir_label_to_lang_label.get(label).unwrap();
                    backend.load_label_address(dest_reg, *backend_label);
                }
                Instruction::CaptureContinuation(
                    dest,
                    resume_label,
                    result_local_index,
                    builtin_fn,
                ) => {
                    // Pass the real machine stack pointer. Continuation resume
                    // restores native execution state, so the saved SP must be
                    // the hardware SP, not a logical eval-stack position.
                    backend.get_stack_pointer_imm(backend.arg(0), 0);

                    // Get current frame pointer into arg 1
                    backend.mov_reg(backend.arg(1), backend.frame_pointer());

                    // Load address of resume label into arg 2
                    let resume_backend_label = ir_label_to_lang_label.get(resume_label).unwrap();
                    backend.load_label_address(backend.arg(2), *resume_backend_label);

                    // Load result_local byte offset into arg 3 (as signed value)
                    let local_offset = backend.get_local_byte_offset(*result_local_index);
                    backend.mov_64(backend.arg(3), local_offset);

                    // No live callee-saved register snapshot was materialized for
                    // this site, so pass a null saved_regs_ptr.
                    let saved_regs_ptr = backend.temporary_register();
                    backend.mov_64(saved_regs_ptr, 0);
                    backend.mov_reg(backend.arg(4), saved_regs_ptr);
                    backend.free_temporary_register(saved_regs_ptr);

                    // Call the builtin
                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);

                    // Store result in dest
                    let dest_spill = self.dest_spill(dest);
                    let dest_reg = match dest {
                        Value::Register(register) => backend.register_from_index(register.index),
                        Value::Spill(_, _) => backend.temporary_register(),
                        _ => panic!(
                            "Unexpected dest type for CaptureContinuationWithSaves: {:?}",
                            dest
                        ),
                    };
                    backend.mov_reg(dest_reg, backend.ret_reg());
                    self.store_spill(dest_reg, dest_spill, backend);
                    if matches!(dest, Value::Spill(_, _)) {
                        backend.free_temporary_register(dest_reg);
                    }
                }
                Instruction::CaptureContinuationWithSaves(
                    dest,
                    resume_label,
                    result_local_index,
                    builtin_fn,
                    saves,
                ) => {
                    // Save registers to their root slots (frame locals) so they
                    // survive GC and are part of the captured stack segment.
                    for save in saves.iter() {
                        let save_reg = self.value_to_register(&save.source, backend);
                        backend.store_local(save_reg, save.local as i32);
                    }

                    // Arguments: (stack_pointer, frame_pointer, resume_address, result_local_offset)

                    // Pass the real machine stack pointer. Continuation resume
                    // restores native execution state, so the saved SP must be
                    // the hardware SP, not a logical eval-stack position.
                    backend.get_stack_pointer_imm(backend.arg(0), 0);

                    // Get current frame pointer into arg 1
                    backend.mov_reg(backend.arg(1), backend.frame_pointer());

                    // Load address of resume label into arg 2
                    let resume_backend_label = ir_label_to_lang_label.get(resume_label).unwrap();
                    backend.load_label_address(backend.arg(2), *resume_backend_label);

                    // Load result_local byte offset into arg 3 (as signed value)
                    let local_offset = backend.get_local_byte_offset(*result_local_index);
                    backend.mov_64(backend.arg(3), local_offset);

                    // Pass null for saved_regs_ptr — registers are already
                    // stored in root slots by the save loop above.
                    let saved_regs_ptr = backend.temporary_register();
                    backend.mov_64(saved_regs_ptr, 0);
                    backend.mov_reg(backend.arg(4), saved_regs_ptr);
                    backend.free_temporary_register(saved_regs_ptr);

                    // Call the builtin
                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);

                    // Store result in dest
                    let dest_spill = self.dest_spill(dest);
                    let dest_reg = match dest {
                        Value::Register(register) => backend.register_from_index(register.index),
                        Value::Spill(_, _) => backend.temporary_register(),
                        _ => panic!("Unexpected dest type for CaptureContinuation: {:?}", dest),
                    };
                    backend.mov_reg(dest_reg, backend.ret_reg());
                    self.store_spill(dest_reg, dest_spill, backend);
                    if matches!(dest, Value::Spill(_, _)) {
                        backend.free_temporary_register(dest_reg);
                    }

                    // Restore saved registers from their root slots.
                    // On the initial capture path, GC may have moved objects
                    // during the builtin call — root slots were updated by GC.
                    let null_reg = self.value_to_register(&Value::Null, backend);
                    for save in saves.iter().rev() {
                        let save_reg = self.value_to_register(&save.source, backend);
                        backend.load_local(save_reg, save.local as i32);
                        backend.store_local(null_reg, save.local as i32);
                    }
                    backend.free_temporary_register(null_reg);
                }
                Instruction::CaptureContinuationTagged(
                    dest,
                    resume_label,
                    result_local_index,
                    builtin_fn,
                    tag_value,
                ) => {
                    // Tagged capture: passes a prompt tag instead of a
                    // saved-regs pointer as the last arg. Runtime looks up
                    // the matching prompt-tag record to determine
                    // capture_top and tag-stack cleanup.
                    backend.get_stack_pointer_imm(backend.arg(0), 0);
                    backend.mov_reg(backend.arg(1), backend.frame_pointer());

                    let resume_backend_label = ir_label_to_lang_label.get(resume_label).unwrap();
                    backend.load_label_address(backend.arg(2), *resume_backend_label);

                    let local_offset = backend.get_local_byte_offset(*result_local_index);
                    backend.mov_64(backend.arg(3), local_offset);

                    let tag_reg = self.value_to_register(tag_value, backend);
                    backend.mov_reg(backend.arg(4), tag_reg);

                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);

                    let dest_spill = self.dest_spill(dest);
                    let dest_reg = match dest {
                        Value::Register(register) => backend.register_from_index(register.index),
                        Value::Spill(_, _) => backend.temporary_register(),
                        _ => panic!(
                            "Unexpected dest type for CaptureContinuationTagged: {:?}",
                            dest
                        ),
                    };
                    backend.mov_reg(dest_reg, backend.ret_reg());
                    self.store_spill(dest_reg, dest_spill, backend);
                    if matches!(dest, Value::Spill(_, _)) {
                        backend.free_temporary_register(dest_reg);
                    }
                }
                Instruction::CaptureContinuationTaggedWithSaves(
                    dest,
                    resume_label,
                    result_local_index,
                    builtin_fn,
                    tag_value,
                    saves,
                ) => {
                    // Save live registers to their GC root slots so the
                    // captured segment (and any GC during the builtin)
                    // see consistent values.
                    for save in saves.iter() {
                        let save_reg = self.value_to_register(&save.source, backend);
                        backend.store_local(save_reg, save.local as i32);
                    }

                    backend.get_stack_pointer_imm(backend.arg(0), 0);
                    backend.mov_reg(backend.arg(1), backend.frame_pointer());

                    let resume_backend_label = ir_label_to_lang_label.get(resume_label).unwrap();
                    backend.load_label_address(backend.arg(2), *resume_backend_label);

                    let local_offset = backend.get_local_byte_offset(*result_local_index);
                    backend.mov_64(backend.arg(3), local_offset);

                    let tag_reg = self.value_to_register(tag_value, backend);
                    backend.mov_reg(backend.arg(4), tag_reg);

                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);

                    let dest_spill = self.dest_spill(dest);
                    let dest_reg = match dest {
                        Value::Register(register) => backend.register_from_index(register.index),
                        Value::Spill(_, _) => backend.temporary_register(),
                        _ => panic!(
                            "Unexpected dest type for CaptureContinuationTaggedWithSaves: {:?}",
                            dest
                        ),
                    };
                    backend.mov_reg(dest_reg, backend.ret_reg());
                    self.store_spill(dest_reg, dest_spill, backend);
                    if matches!(dest, Value::Spill(_, _)) {
                        backend.free_temporary_register(dest_reg);
                    }

                    let null_reg = self.value_to_register(&Value::Null, backend);
                    for save in saves.iter().rev() {
                        let save_reg = self.value_to_register(&save.source, backend);
                        backend.load_local(save_reg, save.local as i32);
                        backend.store_local(null_reg, save.local as i32);
                    }
                    backend.free_temporary_register(null_reg);
                }
                Instruction::PerformEffect(
                    handler,
                    enum_type,
                    op_value,
                    resume_label,
                    result_local_index,
                    builtin_fn,
                ) => {
                    backend.get_stack_pointer_imm(backend.arg(0), 0);
                    backend.mov_reg(backend.arg(1), backend.frame_pointer());

                    let _ = handler;

                    let enum_type_reg = self.value_to_register(enum_type, backend);
                    backend.mov_reg(backend.arg(2), enum_type_reg);

                    let op_reg = self.value_to_register(op_value, backend);
                    backend.mov_reg(backend.arg(3), op_reg);

                    let resume_backend_label = ir_label_to_lang_label.get(resume_label).unwrap();
                    backend.load_label_address(backend.arg(4), *resume_backend_label);

                    let local_offset = backend.get_local_byte_offset(*result_local_index);
                    backend.mov_64(backend.arg(5), local_offset);

                    let saved_regs_ptr = backend.temporary_register();
                    backend.mov_64(saved_regs_ptr, 0);
                    if backend.num_arg_registers() > 6 {
                        backend.mov_reg(backend.arg(6), saved_regs_ptr);
                    } else {
                        backend.push_to_end_of_stack(saved_regs_ptr, 0);
                    }
                    backend.free_temporary_register(saved_regs_ptr);

                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);
                }
                Instruction::PerformEffectWithSaves(
                    handler,
                    enum_type,
                    op_value,
                    resume_label,
                    result_local_index,
                    builtin_fn,
                    saves,
                ) => {
                    // Save registers to their root slots (frame locals) so they
                    // survive GC and are part of the captured stack segment.
                    for save in saves.iter() {
                        let save_reg = self.value_to_register(&save.source, backend);
                        backend.store_local(save_reg, save.local as i32);
                    }

                    backend.get_stack_pointer_imm(backend.arg(0), 0);
                    backend.mov_reg(backend.arg(1), backend.frame_pointer());

                    let _ = handler;

                    let enum_type_reg = self.value_to_register(enum_type, backend);
                    backend.mov_reg(backend.arg(2), enum_type_reg);

                    let op_reg = self.value_to_register(op_value, backend);
                    backend.mov_reg(backend.arg(3), op_reg);

                    let resume_backend_label = ir_label_to_lang_label.get(resume_label).unwrap();
                    backend.load_label_address(backend.arg(4), *resume_backend_label);

                    let local_offset = backend.get_local_byte_offset(*result_local_index);
                    backend.mov_64(backend.arg(5), local_offset);

                    // Pass null for saved_regs_ptr — registers are already
                    // stored in root slots by the save loop above.
                    let saved_regs_ptr = backend.temporary_register();
                    backend.mov_64(saved_regs_ptr, 0);
                    if backend.num_arg_registers() > 6 {
                        backend.mov_reg(backend.arg(6), saved_regs_ptr);
                    } else {
                        backend.push_to_end_of_stack(saved_regs_ptr, 0);
                    }
                    backend.free_temporary_register(saved_regs_ptr);

                    let fn_ptr = self.value_to_register(&Value::RawValue(*builtin_fn), backend);
                    backend.call_builtin(fn_ptr);
                }
                Instruction::ReturnFromShift(value, cont_ptr, builtin_fn) => {
                    // Call return_from_shift builtin (does not return)
                    // Arguments: (stack_pointer, frame_pointer, value, cont_ptr)

                    // Get current stack pointer into arg 0
                    backend.get_stack_pointer_imm(backend.arg(0), 0);

                    // Get current frame pointer into arg 1
                    backend.mov_reg(backend.arg(1), backend.frame_pointer());

                    // Load value into arg 2
                    let value_reg = self.value_to_register(value, backend);
                    backend.mov_reg(backend.arg(2), value_reg);

                    // Load cont_ptr into arg 3
                    let cont_reg = self.value_to_register(cont_ptr, backend);
                    backend.mov_reg(backend.arg(3), cont_reg);

                    // Call the builtin (does not return)
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

    pub fn push_resumable_exception_handler(
        &mut self,
        handler: Label,
        exception_local: Value,
        resume_local: Value,
        builtin_fn: usize,
    ) -> Value {
        let dest = self.volatile_register();
        self.instructions
            .push(Instruction::PushResumableExceptionHandler(
                dest.into(),
                handler,
                exception_local,
                resume_local,
                builtin_fn,
            ));
        dest.into()
    }

    pub fn pop_exception_handler(&mut self, builtin_fn: usize) {
        self.instructions
            .push(Instruction::PopExceptionHandler(builtin_fn));
    }

    pub fn pop_exception_handler_by_id(&mut self, handler_id: Value, builtin_fn: usize) {
        self.instructions
            .push(Instruction::PopExceptionHandlerById(handler_id, builtin_fn));
    }

    pub fn throw_value(
        &mut self,
        value: Value,
        resume_label: Label,
        resume_local_index: usize,
        builtin_fn: usize,
    ) {
        self.instructions.push(Instruction::Throw(
            value,
            resume_label,
            resume_local_index,
            builtin_fn,
        ));
    }

    pub fn push_prompt_handler(&mut self, handler: Label, result_local: Value, builtin_fn: usize) {
        self.instructions.push(Instruction::PushPromptHandler(
            handler,
            result_local,
            builtin_fn,
        ));
    }

    pub fn pop_prompt_handler(&mut self, result_value: Value, builtin_fn: usize) {
        self.instructions
            .push(Instruction::PopPromptHandler(result_value, builtin_fn));
    }

    /// Emit a tagged prompt-tag push. The abort label is the PC that
    /// `return_from_shift_tagged` will longjmp to if a matching perform
    /// unwinds this handle without calling resume. `result_local` is the
    /// local slot that receives the handler's return value on that
    /// longjmp — it also receives the body's normal return value on the
    /// non-abort path, so the post-label code can read it uniformly.
    pub fn push_prompt_tag(
        &mut self,
        tag_value: Value,
        abort_label: Label,
        result_local: Value,
        builtin_fn: usize,
    ) {
        self.instructions.push(Instruction::PushPromptTag(
            tag_value,
            abort_label,
            result_local,
            builtin_fn,
        ));
    }

    pub fn load_label_address(&mut self, dest: VirtualRegister, label: Label) {
        self.instructions
            .push(Instruction::LoadLabelAddress(dest.into(), label));
    }

    pub fn capture_continuation(
        &mut self,
        resume_label: Label,
        result_local_index: usize,
        builtin_fn: usize,
    ) -> Value {
        let dest = self.volatile_register();
        self.instructions.push(Instruction::CaptureContinuation(
            dest.into(),
            resume_label,
            result_local_index,
            builtin_fn,
        ));
        dest.into()
    }

    /// Tag-aware variant of `capture_continuation`. The runtime uses the
    /// tag to locate the matching prompt-tag record (pushed by the
    /// enclosing `handle`) and captures bytes from the current SP up to
    /// that record's `stack_pointer`. The linear-scan allocator rewrites
    /// this to `CaptureContinuationTaggedWithSaves` so live registers
    /// flush to their root slots before the builtin call.
    pub fn capture_continuation_tagged(
        &mut self,
        resume_label: Label,
        result_local_index: usize,
        builtin_fn: usize,
        tag_value: Value,
    ) -> Value {
        let dest = self.volatile_register();
        self.instructions
            .push(Instruction::CaptureContinuationTagged(
                dest.into(),
                resume_label,
                result_local_index,
                builtin_fn,
                tag_value,
            ));
        dest.into()
    }

    pub fn return_from_shift(&mut self, value: Value, cont_ptr: Value, builtin_fn: usize) {
        self.instructions
            .push(Instruction::ReturnFromShift(value, cont_ptr, builtin_fn));
    }

    pub fn perform_effect(
        &mut self,
        handler: Value,
        enum_type: Value,
        op_value: Value,
        resume_label: Label,
        result_local_index: usize,
        builtin_fn: usize,
    ) {
        self.instructions.push(Instruction::PerformEffect(
            handler,
            enum_type,
            op_value,
            resume_label,
            result_local_index,
            builtin_fn,
        ));
    }

    pub fn record_gc_safepoint(&mut self) {
        self.instructions.push(Instruction::RecordGcSafepoint);
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

    /// Inline `==` with a deep-equality fallback. Most equality checks at
    /// runtime are between values that are either identity-equal or have
    /// different tags (e.g. `node.left == null`). Both cases get answered by
    /// 1–2 inline instructions; only same-tag-non-trivial pairs and the few
    /// special cross-tag pairs (string ⇔ heap-string, int ⇔ float) actually
    /// need the deep-equality builtin.
    pub fn equal_with_deep_fallback(&mut self, a: Value, b: Value) -> Value {
        use crate::types::BuiltInTypes;
        let a: VirtualRegister = self.assign_new(a);
        let b: VirtualRegister = self.assign_new(b);
        let result_register = self.assign_new(Value::True);
        let call_builtin_label = self.label("eq_call_builtin");
        let after = self.label("eq_after");

        // Identity short-circuit: covers ints, bools, null, closures,
        // identity-equal heap pointers, interned strings.
        let a_val: Value = a.into();
        let b_val: Value = b.into();
        self.jump_if(after, Condition::Equal, a_val, b_val);

        let tag_a = self.get_tag(a.into());
        let tag_b = self.get_tag(b.into());

        let string_tag: Value = Value::RawValue(BuiltInTypes::String.get_tag() as usize);
        let heap_tag: Value = Value::RawValue(BuiltInTypes::HeapObject.get_tag() as usize);
        let int_tag: Value = Value::RawValue(BuiltInTypes::Int.get_tag() as usize);
        let float_tag: Value = Value::RawValue(BuiltInTypes::Float.get_tag() as usize);

        // Same tag. Only Float and HeapObject values can be non-identity-equal
        // (both are heap-allocated with deep semantics). For Int, Bool,
        // Closure, Function, Null, and String-literal tags, `equal` is just
        // pointer identity — and the identity check above already failed, so
        // they must be non-equal.
        let tags_differ = self.label("eq_tags_differ");
        self.jump_if(tags_differ, Condition::NotEqual, tag_a, tag_b);
        self.jump_if(call_builtin_label, Condition::Equal, tag_a, float_tag);
        self.jump_if(call_builtin_label, Condition::Equal, tag_a, heap_tag);
        self.assign(result_register, Value::False);
        self.jump(after);

        self.write_label(tags_differ);

        // String literal ↔ HeapObject string fallback (deep): heap-side could be
        // a String/StringSlice/ConsString. Punt to builtin for tag mismatches
        // that involve String tag with HeapObject tag.
        let after_str_heap = self.label("eq_after_str_heap");
        self.jump_if(after_str_heap, Condition::NotEqual, tag_a, string_tag);
        self.jump_if(call_builtin_label, Condition::Equal, tag_b, heap_tag);
        self.write_label(after_str_heap);

        let after_heap_str = self.label("eq_after_heap_str");
        self.jump_if(after_heap_str, Condition::NotEqual, tag_a, heap_tag);
        self.jump_if(call_builtin_label, Condition::Equal, tag_b, string_tag);
        self.write_label(after_heap_str);

        // Int ↔ Float mixed numeric.
        let after_int_float = self.label("eq_after_int_float");
        self.jump_if(after_int_float, Condition::NotEqual, tag_a, int_tag);
        self.jump_if(call_builtin_label, Condition::Equal, tag_b, float_tag);
        self.write_label(after_int_float);

        let after_float_int = self.label("eq_after_float_int");
        self.jump_if(after_float_int, Condition::NotEqual, tag_a, float_tag);
        self.jump_if(call_builtin_label, Condition::Equal, tag_b, int_tag);
        self.write_label(after_float_int);

        // No fast-path special case applies — they're not equal.
        self.assign(result_register, Value::False);
        self.jump(after);

        // Slow path: real deep equality via the builtin.
        self.write_label(call_builtin_label);
        let equal_fn =
            Value::RawValue((crate::builtins::equal as usize) << BuiltInTypes::tag_size());
        let a_v: Value = a.into();
        let b_v: Value = b.into();
        let call_result = self.call_builtin(equal_fn, vec![a_v, b_v]);
        self.assign(result_register, call_result);

        self.write_label(after);
        Value::Register(result_register)
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
        let local = self.num_locals;
        self.num_locals += 1;
        self.local_stack.push(local);
        self.instructions
            .push(Instruction::StoreLocal(Value::Local(local), reg));
    }

    pub fn pop_from_stack(&mut self) -> Value {
        let local = self
            .local_stack
            .pop()
            .expect("pop_from_stack: empty local stack");
        let reg = self.volatile_register();
        self.instructions
            .push(Instruction::LoadLocal(reg.into(), Value::Local(local)));
        reg.into()
    }

    /// Returns the local indices currently on the local stack (most recent push last).
    pub fn local_stack_indices(&self) -> &[usize] {
        &self.local_stack
    }

    /// Push to the real eval stack (not a local). Used only for closure free variables
    /// where `make_closure` needs a contiguous memory region accessed by pointer.
    pub fn push_to_eval_stack(&mut self, reg: Value) {
        self.instructions.push(Instruction::PushStack(reg));
    }

    /// Pop from the real eval stack. Pairs with `push_to_eval_stack`.
    pub fn pop_from_eval_stack(&mut self) -> Value {
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

    /// Convert a tagged integer to a float register.
    /// The integer should already be untagged (shifted right).
    fn int_to_float(&mut self, source: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::IntToFloat(dest, source));
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

    fn fround_to_zero(&mut self, a: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions.push(Instruction::FRoundToZero(dest, a));
        dest
    }

    /// Float modulo: a % b = a - trunc(a/b) * b
    fn modulo_float(&mut self, a: Value, b: Value) -> Value {
        let q = self.div_float(a, b);
        let q_trunc = self.fround_to_zero(q);
        let q_times_b = self.mul_float(q_trunc, b);
        self.sub_float(a, q_times_b)
    }

    /// Emit an inline type error call that supports resume.
    /// If the error is caught and resumed, the resume value becomes the result.
    /// After the call, jumps to `after_label`.
    fn emit_type_error_with_resume(
        &mut self,
        result_register: VirtualRegister,
        after_label: Label,
    ) {
        assert!(
            self.error_fn_pointer != 0,
            "throw-type-error builtin not registered: cannot emit guarded operation \
             without resume support. Ensure beagle.builtin/throw-type-error is installed \
             before compiling this function ({}).",
            self.debug_name.as_deref().unwrap_or("<anonymous>"),
        );
        let stack_pointer = self.get_stack_pointer_imm(0);
        let frame_pointer = self.get_frame_pointer();
        let f = self.assign_new(Value::Function(self.error_fn_pointer));
        let resumed_value = self.call_builtin(f.into(), vec![stack_pointer, frame_pointer]);
        // If we reach here, the error was caught and resumed with a value
        self.assign(result_register, resumed_value);
        self.jump(after_label);
    }

    fn allocate(&mut self, size: Value) -> Value {
        let stack_pointer = self.get_stack_pointer_imm(0);
        let frame_pointer = self.get_frame_pointer();
        let f = self.assign_new(Value::Function(self.allocate_fn_pointer));
        self.call_builtin(f.into(), vec![stack_pointer, frame_pointer, size])
    }
}
