#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::missing_safety_doc)]

// Backend-specific imports for runtime trampolines
// Use x86-64 backend if:
//   - Explicit feature flag is set, OR
//   - Building for x86_64 architecture (and no explicit ARM64 feature)
cfg_if::cfg_if! {
    if #[cfg(any(feature = "backend-x86-64", all(target_arch = "x86_64", not(feature = "backend-arm64"))))] {
        use crate::machine_code::x86_codegen::{RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11, RSP, X86Asm};
    } else {
        use crate::machine_code::arm_codegen::{SP, X0, X1, X2, X3, X4, X10};
    }
}
#[cfg(all(debug_assertions, not(feature = "json")))]
use bincode::config::standard;
use bincode::{Decode, Encode};
use clap::Parser as ClapParser;
use gc::{Allocator, StackMapDetails, get_allocate_options};
#[allow(unused)]
use gc::{
    compacting::CompactingHeap, generational::GenerationalGC, mark_and_sweep::MarkAndSweep,
    mutex_allocator::MutexAllocator,
};
use nanoserde::SerJson;
use runtime::{DefaultPrinter, Printer, Runtime, TestPrinter};

use std::{cell::UnsafeCell, env, error::Error, sync::OnceLock, time::Instant};

pub mod embedded_stdlib {
    pub fn get(name: &str) -> Option<&'static str> {
        match name {
            "std.bg" => Some(include_str!("../standard-library/std.bg")),
            "beagle.ffi.bg" => Some(include_str!("../standard-library/beagle.ffi.bg")),
            "beagle.io.bg" => Some(include_str!("../standard-library/beagle.io.bg")),
            "beagle.effect.bg" => Some(include_str!("../standard-library/beagle.effect.bg")),
            "beagle.async.bg" => Some(include_str!("../standard-library/beagle.async.bg")),
            "beagle.fs.bg" => Some(include_str!("../standard-library/beagle.fs.bg")),
            "beagle.timer.bg" => Some(include_str!("../standard-library/beagle.timer.bg")),
            "beagle.socket.bg" => Some(include_str!("../standard-library/beagle.socket.bg")),
            "beagle.stream.bg" => Some(include_str!("../standard-library/beagle.stream.bg")),
            "beagle.repl-session.bg" => {
                Some(include_str!("../standard-library/beagle.repl-session.bg"))
            }
            "beagle.repl.bg" => Some(include_str!("../standard-library/beagle.repl.bg")),
            "beagle.repl-main.bg" => Some(include_str!("../standard-library/beagle.repl-main.bg")),
            "beagle.repl-interactive.bg" => Some(include_str!(
                "../standard-library/beagle.repl-interactive.bg"
            )),
            "beagle.spawn.bg" => Some(include_str!("../standard-library/beagle.spawn.bg")),
            "beagle.mutable-array.bg" => {
                Some(include_str!("../standard-library/beagle.mutable-array.bg"))
            }
            _ => None,
        }
    }
}

#[cfg(not(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
)))]
mod arm;

#[macro_use]
mod trace;
pub mod ast;
pub mod backend;
mod builtins;
mod code_memory;
pub mod common;
mod compiler;
mod gc;
pub mod ir;
pub mod machine_code;
pub mod mmap_utils;
pub mod parser;
mod pretty_print;
mod primitives;
mod register_allocation;
pub mod runtime;
mod types;

#[cfg(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
))]
mod x86;

pub mod collections;
mod repl;

#[derive(Debug, Encode, Decode, Clone, SerJson)]
pub struct Message {
    kind: String,
    data: Data,
}

#[cfg(all(target_arch = "aarch64", not(feature = "backend-x86-64")))]
fn compile_arm_continuation_return_stub(runtime: &mut Runtime) {
    use crate::builtins::segmented_continuation_return;
    use crate::machine_code::arm_codegen::{
        ArmAsm, LdrImmGenSelector, SP, StrImmGenSelector, X2, X3, X4, X5, X6, X16, X19, X20, X21,
        X22, X23, X24, X25, X26, X27, X28, X29, X30,
    };

    let mut lang = arm::LowLevelArm::new();
    for instr in
        arm::LowLevelArm::mov_64_bit_num(X16, segmented_continuation_return as usize as isize)
    {
        lang.instructions.push(instr);
    }
    lang.instructions.push(ArmAsm::Br { rn: X16 });
    let code: Vec<u8> = lang
        .instructions
        .iter()
        .flat_map(|instr| instr.encode().to_le_bytes())
        .collect();

    runtime
        .add_function_mark_executable(
            "beagle.builtin/continuation-return-stub".to_string(),
            &code,
            0,
            1,
        )
        .unwrap();

    let function = runtime
        .get_function_by_name_mut("beagle.builtin/continuation-return-stub")
        .unwrap();
    function.is_builtin = true;
    if std::env::var("BEAGLE_DEBUG_RESUME").is_ok() {
        let ptr: usize = function.pointer.into();
        eprintln!("[arm-stub] continuation-return-stub={:#x}", ptr);
    }

    use crate::builtins::continuation_trampoline_with_saved_regs_and_context;

    let mut lang = arm::LowLevelArm::new();
    lang.mov_reg(X3, SP);
    lang.mov_reg(X4, X29);
    lang.mov_reg(X5, X30);
    lang.sub_stack_pointer(10);
    let regs = [X19, X20, X21, X22, X23, X24, X25, X26, X27, X28];
    for (i, reg) in regs.into_iter().enumerate() {
        lang.instructions.push(ArmAsm::StrImmGen {
            size: 0b11,
            imm9: 0,
            imm12: i as i32,
            rn: SP,
            rt: reg,
            class_selector: StrImmGenSelector::UnsignedOffset,
        });
    }
    lang.mov_reg(X2, SP);
    for instr in arm::LowLevelArm::mov_64_bit_num(
        X16,
        continuation_trampoline_with_saved_regs_and_context as usize as isize,
    ) {
        lang.instructions.push(instr);
    }
    lang.call(X16);
    lang.add_stack_pointer(10);
    lang.ret();
    let code: Vec<u8> = lang
        .instructions
        .iter()
        .flat_map(|instr| instr.encode().to_le_bytes())
        .collect();

    runtime
        .add_function_mark_executable(
            "beagle.builtin/continuation-trampoline".to_string(),
            &code,
            0,
            2,
        )
        .unwrap();

    let function = runtime
        .get_function_by_name_mut("beagle.builtin/continuation-trampoline")
        .unwrap();
    function.is_builtin = true;
    if std::env::var("BEAGLE_DEBUG_RESUME").is_ok() {
        let ptr: usize = function.pointer.into();
        eprintln!("[arm-stub] continuation-trampoline={:#x}", ptr);
    }

    use crate::builtins::capture_continuation_runtime_with_saved_regs;

    let mut lang = arm::LowLevelArm::new();
    // Keep SP 16-byte aligned across the helper call. We need 10 slots for
    // x19-x28 plus one saved LR slot, so reserve 12 words total.
    lang.sub_stack_pointer(12);
    for (i, reg) in regs.into_iter().enumerate() {
        lang.instructions.push(ArmAsm::StrImmGen {
            size: 0b11,
            imm9: 0,
            imm12: i as i32,
            rn: SP,
            rt: reg,
            class_selector: StrImmGenSelector::UnsignedOffset,
        });
    }
    lang.instructions.push(ArmAsm::StrImmGen {
        size: 0b11,
        imm9: 0,
        imm12: 10,
        rn: SP,
        rt: X30,
        class_selector: StrImmGenSelector::UnsignedOffset,
    });
    lang.mov_reg(X4, SP);
    for instr in arm::LowLevelArm::mov_64_bit_num(
        X16,
        capture_continuation_runtime_with_saved_regs as usize as isize,
    ) {
        lang.instructions.push(instr);
    }
    lang.call(X16);
    lang.instructions.push(ArmAsm::LdrImmGen {
        rt: X30,
        rn: SP,
        imm9: 0,
        imm12: 10,
        size: 0b11,
        class_selector: LdrImmGenSelector::UnsignedOffset,
    });
    lang.add_stack_pointer(12);
    lang.ret();
    let code: Vec<u8> = lang
        .instructions
        .iter()
        .flat_map(|instr| instr.encode().to_le_bytes())
        .collect();

    runtime
        .add_function_mark_executable(
            "beagle.builtin/capture-continuation".to_string(),
            &code,
            0,
            4,
        )
        .unwrap();

    let function = runtime
        .get_function_by_name_mut("beagle.builtin/capture-continuation")
        .unwrap();
    function.is_builtin = true;
    if std::env::var("BEAGLE_DEBUG_RESUME").is_ok() {
        let ptr: usize = function.pointer.into();
        eprintln!("[arm-stub] capture-continuation={:#x}", ptr);
    }

    use crate::builtins::perform_effect_runtime_with_saved_regs;

    let mut lang = arm::LowLevelArm::new();
    lang.sub_stack_pointer(10);
    for (i, reg) in regs.into_iter().enumerate() {
        lang.instructions.push(ArmAsm::StrImmGen {
            size: 0b11,
            imm9: 0,
            imm12: i as i32,
            rn: SP,
            rt: reg,
            class_selector: StrImmGenSelector::UnsignedOffset,
        });
    }
    lang.mov_reg(X6, SP);
    for instr in arm::LowLevelArm::mov_64_bit_num(
        X16,
        perform_effect_runtime_with_saved_regs as usize as isize,
    ) {
        lang.instructions.push(instr);
    }
    lang.call(X16);
    lang.add_stack_pointer(10);
    lang.ret();
    let code: Vec<u8> = lang
        .instructions
        .iter()
        .flat_map(|instr| instr.encode().to_le_bytes())
        .collect();

    runtime
        .add_function_mark_executable("beagle.builtin/perform-effect".to_string(), &code, 0, 6)
        .unwrap();

    let function = runtime
        .get_function_by_name_mut("beagle.builtin/perform-effect")
        .unwrap();
    function.is_builtin = true;
    if std::env::var("BEAGLE_DEBUG_RESUME").is_ok() {
        let ptr: usize = function.pointer.into();
        eprintln!("[arm-stub] perform-effect={:#x}", ptr);
    }

    // ========================================================================
    // ARM64 return-jump trampoline
    // Unified trampoline for all "restore registers + jump" operations.
    // Args: X0=new_sp, X1=new_fp, X2=new_lr, X3=jump_target,
    //       X4=callee_saved_ptr (NULL to skip), X5=value (placed in X0)
    // ========================================================================
    {
        use crate::machine_code::arm_codegen::{ArmAsm, LdrImmGenSelector, X0, X1, ZERO_REGISTER};
        let mut lang = arm::LowLevelArm::new();

        // Save jump_target to X16 (scratch) before we clobber argument registers
        lang.mov_reg(X16, X3);

        // Conditional callee-saved restore: skip if X4 == NULL
        // CMP X4, XZR sets flags; B.EQ skips past 10 LDR instructions
        // imm19=11 because the offset is PC-relative from B.EQ itself (need to skip 10 LDRs + land past them)
        lang.compare(X4, ZERO_REGISTER);
        lang.instructions.push(arm::jump_equal(11));

        // Load x19-x28 from array at X4
        let callee_regs = [X19, X20, X21, X22, X23, X24, X25, X26, X27, X28];
        for (i, reg) in callee_regs.iter().enumerate() {
            lang.instructions.push(ArmAsm::LdrImmGen {
                rt: *reg,
                rn: X4,
                imm9: 0,
                imm12: i as i32,
                size: 0b11,
                class_selector: LdrImmGenSelector::UnsignedOffset,
            });
        }

        // skip_restore lands here:
        // Set SP, FP, LR from arguments
        lang.mov_reg(SP, X0);
        lang.mov_reg(X29, X1);
        lang.mov_reg(X30, X2);

        // Set return value in X0
        lang.mov_reg(X0, X5);

        // Jump to target
        lang.instructions.push(ArmAsm::Br { rn: X16 });

        let code: Vec<u8> = lang
            .instructions
            .iter()
            .flat_map(|instr| instr.encode().to_le_bytes())
            .collect();
        runtime
            .add_function_mark_executable("beagle.builtin/return-jump".to_string(), &code, 0, 6)
            .unwrap();
        let function = runtime
            .get_function_by_name_mut("beagle.builtin/return-jump")
            .unwrap();
        function.is_builtin = true;
    }

    // ========================================================================
    // ARM64 stack-switch trampoline
    // Args: X0=stack_top, X1=target function pointer
    // ========================================================================
    {
        use crate::machine_code::arm_codegen::ArmAsm;
        let mut lang = arm::LowLevelArm::new();

        // Switch SP to new stack
        lang.mov_reg(SP, X0);
        // Save target to scratch register and call it
        lang.mov_reg(X16, X1);
        lang.call(X16);
        // Unreachable trap
        lang.instructions.push(ArmAsm::Brk { imm16: 0 });

        let code: Vec<u8> = lang
            .instructions
            .iter()
            .flat_map(|instr| instr.encode().to_le_bytes())
            .collect();
        runtime
            .add_function_mark_executable("beagle.builtin/stack-switch".to_string(), &code, 0, 2)
            .unwrap();
        let function = runtime
            .get_function_by_name_mut("beagle.builtin/stack-switch")
            .unwrap();
        function.is_builtin = true;
    }

    // ========================================================================
    // ARM64 read-fp trampoline: returns X29 (frame pointer)
    // ========================================================================
    {
        use crate::machine_code::arm_codegen::X0;
        let mut lang = arm::LowLevelArm::new();
        lang.mov_reg(X0, X29);
        lang.ret();
        let code: Vec<u8> = lang
            .instructions
            .iter()
            .flat_map(|instr| instr.encode().to_le_bytes())
            .collect();
        runtime
            .add_function_mark_executable("beagle.builtin/read-fp".to_string(), &code, 0, 0)
            .unwrap();
        let function = runtime
            .get_function_by_name_mut("beagle.builtin/read-fp")
            .unwrap();
        function.is_builtin = true;
    }

    // ========================================================================
    // ARM64 read-sp trampoline: returns SP (BLR doesn't modify SP)
    // ========================================================================
    {
        use crate::machine_code::arm_codegen::X0;
        let mut lang = arm::LowLevelArm::new();
        lang.mov_reg(X0, SP);
        lang.ret();
        let code: Vec<u8> = lang
            .instructions
            .iter()
            .flat_map(|instr| instr.encode().to_le_bytes())
            .collect();
        runtime
            .add_function_mark_executable("beagle.builtin/read-sp".to_string(), &code, 0, 0)
            .unwrap();
        let function = runtime
            .get_function_by_name_mut("beagle.builtin/read-sp")
            .unwrap();
        function.is_builtin = true;
    }

    // ========================================================================
    // ARM64 read-sp-fp trampoline: returns X0=SP, X1=X29
    // ========================================================================
    {
        use crate::machine_code::arm_codegen::{X0, X1};
        let mut lang = arm::LowLevelArm::new();
        lang.mov_reg(X0, SP);
        lang.mov_reg(X1, X29);
        lang.ret();
        let code: Vec<u8> = lang
            .instructions
            .iter()
            .flat_map(|instr| instr.encode().to_le_bytes())
            .collect();
        runtime
            .add_function_mark_executable("beagle.builtin/read-sp-fp".to_string(), &code, 0, 0)
            .unwrap();
        let function = runtime
            .get_function_by_name_mut("beagle.builtin/read-sp-fp")
            .unwrap();
        function.is_builtin = true;
    }

    // ========================================================================
    // ARM64 save-callee-regs trampoline: saves x19-x28 to [X0]
    // ========================================================================
    {
        use crate::machine_code::arm_codegen::{ArmAsm, StrImmGenSelector, X0};
        let mut lang = arm::LowLevelArm::new();
        let callee_regs = [X19, X20, X21, X22, X23, X24, X25, X26, X27, X28];
        for (i, reg) in callee_regs.iter().enumerate() {
            lang.instructions.push(ArmAsm::StrImmGen {
                size: 0b11,
                imm9: 0,
                imm12: i as i32,
                rn: X0,
                rt: *reg,
                class_selector: StrImmGenSelector::UnsignedOffset,
            });
        }
        lang.ret();
        let code: Vec<u8> = lang
            .instructions
            .iter()
            .flat_map(|instr| instr.encode().to_le_bytes())
            .collect();
        runtime
            .add_function_mark_executable(
                "beagle.builtin/save-callee-regs".to_string(),
                &code,
                0,
                1,
            )
            .unwrap();
        let function = runtime
            .get_function_by_name_mut("beagle.builtin/save-callee-regs")
            .unwrap();
        function.is_builtin = true;
    }
}

// TODO: This should really live on the debugger side of things
#[derive(Debug, Encode, Decode, Clone, SerJson)]
enum Data {
    ForeignFunction {
        name: String,
        pointer: usize,
    },
    BuiltinFunction {
        name: String,
        pointer: usize,
    },
    HeapSegmentPointer {
        pointer: usize,
    },
    UserFunction {
        name: String,
        pointer: usize,
        len: usize,
        number_of_arguments: usize,
    },
    Label {
        label: String,
        function_pointer: usize,
        label_index: usize,
        label_location: usize,
    },
    StackMap {
        pc: usize,
        name: String,
        stack_map: Vec<(usize, StackMapDetails)>,
    },
    Allocate {
        bytes: usize,
        stack_pointer: usize,
        kind: String,
    },
    Tokens {
        file_name: String,
        tokens: Vec<String>,
        token_line_column_map: Vec<(usize, usize)>,
    },
    Ir {
        function_pointer: usize,
        file_name: String,
        instructions: Vec<String>,
        token_range_to_ir_range: Vec<((usize, usize), (usize, usize))>,
    },
    Arm {
        function_pointer: usize,
        file_name: String,
        instructions: Vec<String>,
        ir_to_machine_code_range: Vec<(usize, (usize, usize))>,
    },
}

// Serialize is only used in debug builds without the json feature
#[cfg(all(debug_assertions, not(feature = "json")))]
trait Serialize {
    fn to_binary(&self) -> Vec<u8>;
}

#[cfg(all(debug_assertions, not(feature = "json")))]
impl<T: Encode + Decode<()>> Serialize for T {
    fn to_binary(&self) -> Vec<u8> {
        bincode::encode_to_vec(self, standard()).unwrap()
    }
}

const PADDING_FOR_ALIGNMENT: i64 = 2;

fn compile_trampoline(runtime: &mut Runtime) {
    cfg_if::cfg_if! {
        if #[cfg(any(feature = "backend-x86-64", all(target_arch = "x86_64", not(feature = "backend-arm64"))))] {
            use crate::machine_code::x86_codegen::{RBP, RBX, R12, R13, R14, R15};
            let mut lang = x86::LowLevelX86::new();

            lang.instructions.push(X86Asm::Push { reg: RBP });
            lang.instructions.push(X86Asm::MovRR { dest: RBP, src: RSP });
            lang.instructions.push(X86Asm::Push { reg: RBX });
            lang.instructions.push(X86Asm::Push { reg: R12 });
            lang.instructions.push(X86Asm::Push { reg: R13 });
            lang.instructions.push(X86Asm::Push { reg: R14 });
            lang.instructions.push(X86Asm::Push { reg: R15 });

            // Save the Rust stack pointer on the Rust stack so Beagle stack usage can't clobber it.
            lang.mov_reg(R10, RSP);
            lang.instructions.push(X86Asm::Push { reg: R10 }); // saved_rust_rsp at [RBP-48]

            lang.mov_reg(R10, RSP);
            lang.mov_reg(RSP, RDI);
            // Skip the GlobalObjectBlock slot at stack_base - 8
            // (Runtime pre-writes GlobalObjectBlock* there before calling trampoline)
            lang.instructions.push(X86Asm::SubRI { dest: RSP, imm: 8 });
            lang.instructions.push(X86Asm::Push { reg: R10 });
            // Stack is now 16-byte aligned (stack_base - 16), no extra sub needed

            lang.mov_reg(R10, RSI);
            lang.mov_reg(RDI, RDX);
            lang.mov_reg(RSI, RCX);
            lang.mov_reg(RDX, R8);

            lang.call(R10);

            // Pop old RSP (from stack_base - 16), RSP becomes stack_base - 8
            // GlobalBlock slot at stack_base - 8 is preserved
            lang.instructions.push(X86Asm::MovRM {
                dest: RSP,
                base: RBP,
                offset: -48,
            });

            lang.instructions.push(X86Asm::Pop { reg: R15 });
            lang.instructions.push(X86Asm::Pop { reg: R14 });
            lang.instructions.push(X86Asm::Pop { reg: R13 });
            lang.instructions.push(X86Asm::Pop { reg: R12 });
            lang.instructions.push(X86Asm::Pop { reg: RBX });
            lang.instructions.push(X86Asm::Pop { reg: RBP });
            lang.ret();

            runtime
                .add_function_mark_executable("trampoline".to_string(), &lang.compile_to_bytes(), 0, 3)
                .unwrap();
        } else {
            let mut lang = arm::LowLevelArm::new();

            lang.prelude();

            // Should I store or push?
            for (i, reg) in lang.canonical_volatile_registers.clone().iter().enumerate() {
                lang.store_on_stack(*reg, -((i + 4_usize) as i32));
            }

            lang.mov_reg(X10, SP);
            lang.mov_reg(SP, X0);
            // Skip the GlobalObjectBlock slot at stack_base - 8
            // (Runtime pre-writes GlobalObjectBlock* there before calling trampoline)
            lang.sub_stack_pointer(8);
            lang.push_to_stack(X10);

            lang.mov_reg(X10, X1);
            lang.mov_reg(X0, X2);
            lang.mov_reg(X1, X3);
            lang.mov_reg(X2, X4);

            lang.call(X10);

            lang.pop_from_stack_indexed(X10, 0);
            lang.mov_reg(SP, X10);
            for (i, reg) in lang
                .canonical_volatile_registers
                .clone()
                .iter()
                .enumerate()
                .rev()
            {
                lang.load_from_stack(*reg, -((i + 4_usize) as i32));
            }
            lang.epilogue();
            lang.ret();

            runtime
                .add_function_mark_executable("trampoline".to_string(), &lang.compile_directly(), 0, 3)
                .unwrap();
        }
    }

    let function = runtime.get_function_by_name_mut("trampoline").unwrap();
    function.is_builtin = true;
}

fn compile_save_volatile_registers_for(runtime: &mut Runtime, register_num: usize) {
    let function_name =
        "beagle.builtin/save_volatile_registers".to_owned() + &register_num.to_string();
    let arity = register_num + 1;

    cfg_if::cfg_if! {
        if #[cfg(any(feature = "backend-x86-64", all(target_arch = "x86_64", not(feature = "backend-arm64"))))] {
            let mut lang = x86::LowLevelX86::new();
            // Use lang.arg() to get correct argument register for x86-64 ABI
            // On x86-64, arg(n) maps to: RDI, RSI, RDX, RCX, R8, R9 (not sequential indices!)
            let call_register = lang.arg(register_num as u8);

            // We store volatile registers at local offsets 3-6, so need max_locals >= 7
            // (store_local at offset n stores at [RBP - (n+1)*8])
            let num_volatile = lang.canonical_volatile_registers.len();
            let max_offset = num_volatile + PADDING_FOR_ALIGNMENT as usize;
            lang.set_max_locals(max_offset + 1);

            lang.prelude();

            for (i, reg) in lang.canonical_volatile_registers.clone().iter().enumerate() {
                lang.store_local(*reg, (i + PADDING_FOR_ALIGNMENT as usize + 1) as i32);
            }

            lang.call(call_register);

            for (i, reg) in lang.canonical_volatile_registers.clone().iter().enumerate() {
                lang.load_local(*reg, (i + PADDING_FOR_ALIGNMENT as usize + 1) as i32);
            }

            lang.epilogue();
            lang.ret();

            runtime
                .add_function_mark_executable(
                    function_name.to_string(),
                    &lang.compile_to_bytes(),
                    0,
                    arity,
                )
                .unwrap();
        } else {
            use crate::machine_code::arm_codegen::Register;
            let call_register = Register {
                index: register_num as u8,
                size: crate::machine_code::arm_codegen::Size::S64,
            };
            let mut lang = arm::LowLevelArm::new();
            lang.prelude();

            lang.sub_stack_pointer(
                (lang.canonical_volatile_registers.len() + PADDING_FOR_ALIGNMENT as usize) as i32,
            );

            for (i, reg) in lang.canonical_volatile_registers.clone().iter().enumerate() {
                lang.store_on_stack(*reg, -((i + PADDING_FOR_ALIGNMENT as usize + 1) as i32));
            }

            lang.call(call_register);

            for (i, reg) in lang.canonical_volatile_registers.clone().iter().enumerate() {
                lang.load_from_stack(*reg, -((i + PADDING_FOR_ALIGNMENT as usize + 1) as i32));
            }

            lang.add_stack_pointer(
                (lang.canonical_volatile_registers.len() + PADDING_FOR_ALIGNMENT as usize) as i32,
            );

            lang.epilogue();
            lang.ret();

            runtime
                .add_function_mark_executable(
                    function_name.to_string(),
                    &lang.compile_directly(),
                    0,
                    arity,
                )
                .unwrap();
        }
    }

    let function = runtime.get_function_by_name_mut(&function_name).unwrap();
    function.is_builtin = true;
}

/// Generate apply call trampolines that set X9/R10 (arg count) before calling.
/// This is needed for calling variadic functions from Rust.
///
/// Signature: apply_call_N(fn_ptr, arg0, arg1, ..., argN-1) -> result
/// Where N is the number of arguments to pass.
///
/// The trampoline:
/// 1. Sets X9/R10 to N (tagged)
/// 2. Moves fn_ptr to a temp register
/// 3. Shifts args down (arg0 -> X0, arg1 -> X1, etc.)
/// 4. Calls the function via the temp register
/// 5. Returns the result
fn compile_apply_call_trampolines(runtime: &mut Runtime) {
    cfg_if::cfg_if! {
        if #[cfg(any(feature = "backend-x86-64", all(target_arch = "x86_64", not(feature = "backend-arm64"))))] {
            // x86-64: 6 arg registers (RDI, RSI, RDX, RCX, R8, R9)
            // We use last register (R9) for fn_ptr, so max 5 actual args
            // Plus we need to handle stack args for more
            compile_apply_call_trampolines_x86_64(runtime);
        } else {
            // ARM64: 8 arg registers (X0-X7)
            // We reserve X9 for arg count, use X10 for fn_ptr temporarily
            // So we can pass up to 8 args in registers
            compile_apply_call_trampolines_arm64(runtime);
        }
    }
}

#[cfg(not(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
)))]
fn compile_apply_call_trampolines_arm64(runtime: &mut Runtime) {
    use crate::machine_code::arm_codegen::{X0, X1, X2, X3, X4, X5, X6, X7, X9, X10, X11};
    use crate::types::BuiltInTypes;

    // Generate trampolines for 0-16 args
    // For 0-8 args: all in registers
    // For 9-16 args: first 8 in registers, rest on stack
    for num_args in 0..=16 {
        let function_name = format!("beagle.builtin/apply_call_{}", num_args);

        let mut lang = arm::LowLevelArm::new();

        // Function receives: fn_ptr in X0, then arg0..argN-1 in X1..X(N)
        // For N > 7: X0-X7 have fn_ptr and args[0-6], stack has args[7+]
        //
        // Target function expects:
        // - args[0-7] in X0-X7
        // - args[8+] on stack
        //
        // So we need to:
        // 1. Save fn_ptr from X0
        // 2. Shuffle X1-X7 -> X0-X6
        // 3. Load our stack args into X7 and push extras for target

        lang.prelude();

        // Save fn_ptr from X0 to X10 before we shuffle args
        lang.mov_reg(X10, X0);

        // Set X9 = num_args (tagged)
        let arg_count_tagged = (num_args << BuiltInTypes::tag_size()) as i64;
        lang.mov_64(X9, arg_count_tagged as isize);

        // Handle stack args for target function (args[8+])
        // Our args[8+] are at [FP+24], [FP+32], etc.
        // (FP+0 = saved FP, FP+8 = saved LR, FP+16 = our arg[7], FP+24 = our arg[8], ...)
        // Target expects them at [SP+0], [SP+8], etc.
        if num_args > 8 {
            for i in 8..num_args {
                // Our arg[i] is at offset (i - 7 + 2) from FP in 8-byte units
                // i=8 -> offset 3 (FP+24)
                // i=9 -> offset 4 (FP+32), etc.
                let load_offset = (i as i32) - 7 + 2;
                lang.load_from_stack(X11, load_offset);
                // Target expects arg[i] at [SP + (i-8)*8]
                let store_offset = (i as i32) - 8;
                lang.push_to_end_of_stack(X11, store_offset);
            }
        }

        // Shuffle register args: X1->X0, X2->X1, ..., X7->X6
        let regs = [X0, X1, X2, X3, X4, X5, X6, X7];
        for i in 0..num_args.min(7) {
            lang.mov_reg(regs[i], regs[i + 1]);
        }

        // If we have 8+ args, X7 needs to come from our stack
        // Our arg[7] (8th arg to target) is at FP+16
        if num_args >= 8 {
            lang.load_from_stack(X7, 2); // offset 2 = FP+16
        }

        // Call the function via X10
        lang.call(X10);

        // No stack cleanup needed - push_to_end_of_stack uses pre-allocated frame space

        lang.epilogue();
        lang.ret();

        runtime
            .add_function_mark_executable(
                function_name.clone(),
                &lang.compile_directly(),
                0,
                num_args + 1, // fn_ptr + num_args
            )
            .unwrap();

        let function = runtime.get_function_by_name_mut(&function_name).unwrap();
        function.is_builtin = true;
    }
}

#[cfg(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
))]
fn compile_apply_call_trampolines_x86_64(runtime: &mut Runtime) {
    use crate::machine_code::x86_codegen::{R8, R9, R10, R11, RBP, RCX, RDI, RDX, RSI, RSP};
    use crate::types::BuiltInTypes;

    // Generate trampolines for 0-16 args
    // For 0-6 args: all in registers
    // For 7-16 args: first 6 in registers, rest on stack
    for num_args in 0..=16 {
        let function_name = format!("beagle.builtin/apply_call_{}", num_args);

        let mut lang = x86::LowLevelX86::new();

        // Function receives: fn_ptr in RDI, then arg0..argN-1 in RSI, RDX, RCX, R8, R9, stack...
        // Our stack args start at [RBP+16] (after push RBP; mov RBP, RSP)
        // arg5 = [RBP+16], arg6 = [RBP+24], arg7 = [RBP+32], etc.
        //
        // Target function expects:
        // - args 0-5 in RDI, RSI, RDX, RCX, R8, R9
        // - args 6+ on stack at [RSP], [RSP+8], ...
        //
        // So we need to:
        // 1. Save fn_ptr from RDI to R11
        // 2. Push stack args for target (args 6+) in REVERSE order
        // 3. Shuffle register args down by 1
        // 4. Load arg5 from our stack into R9 (if num_args >= 6)
        // 5. Call via R11

        // Standard prologue
        lang.instructions.push(X86Asm::Push { reg: RBP });
        lang.instructions.push(X86Asm::MovRR {
            dest: RBP,
            src: RSP,
        });

        // Save fn_ptr from RDI to R11 before we shuffle args
        lang.mov_reg(R11, RDI);

        // Set R10 = num_args (tagged)
        let arg_count_tagged = (num_args << BuiltInTypes::tag_size()) as i64;
        lang.mov_64(R10, arg_count_tagged as isize);

        // Handle stack args for target function (args 6+)
        // We need to push them in reverse order so they appear correctly on stack
        // Our arg[i] is at [RBP + 16 + (i-5)*8] for i >= 5
        // arg5 = [RBP+16], arg6 = [RBP+24], arg7 = [RBP+32], etc.
        let num_stack_args = if num_args > 6 { num_args - 6 } else { 0 };
        let needs_alignment_pad = num_stack_args % 2 == 1;
        if num_stack_args > 0 {
            // Ensure 16-byte stack alignment before call.
            // After prologue (push RBP), RSP % 16 == 0.
            // Each pushed stack arg is 8 bytes. If we push an odd number,
            // RSP % 16 == 8 before the call, violating ABI alignment.
            // Add an 8-byte pad so the total pushed bytes are 16-byte aligned.
            if needs_alignment_pad {
                lang.instructions.push(X86Asm::SubRI { dest: RSP, imm: 8 });
            }
            // Push args in reverse order: argN-1, argN-2, ..., arg6
            for i in (6..num_args).rev() {
                // Our arg[i] is at [RBP + 16 + (i-5)*8]
                let offset = 16 + ((i as i32) - 5) * 8;
                // Use RDI as temp since we'll overwrite it in the shuffle below
                lang.instructions.push(X86Asm::MovRM {
                    dest: RDI,
                    base: RBP,
                    offset,
                });
                lang.instructions.push(X86Asm::Push { reg: RDI });
            }
        }

        // Shuffle args: RSI->RDI, RDX->RSI, RCX->RDX, R8->RCX, R9->R8
        let regs = [RDI, RSI, RDX, RCX, R8, R9];

        // Move register args down by 1
        for i in 0..num_args.min(5) {
            // Move regs[i+1] -> regs[i]
            lang.mov_reg(regs[i], regs[i + 1]);
        }

        // If we have 6+ args, R9 needs to come from our stack
        // arg5 = [RBP+16]
        if num_args >= 6 {
            lang.instructions.push(X86Asm::MovRM {
                dest: R9,
                base: RBP,
                offset: 16,
            });
        }

        // Call the function via R11
        lang.call(R11);

        // Clean up stack args and alignment padding we pushed (if any)
        if num_stack_args > 0 {
            let pad = if needs_alignment_pad { 8 } else { 0 };
            let stack_cleanup = (num_stack_args * 8 + pad) as i32;
            lang.instructions.push(X86Asm::AddRI {
                dest: RSP,
                imm: stack_cleanup,
            });
        }

        // Epilogue
        lang.instructions.push(X86Asm::Pop { reg: RBP });
        lang.ret();

        runtime
            .add_function_mark_executable(
                function_name.clone(),
                &lang.compile_to_bytes(),
                0,
                num_args + 1, // fn_ptr + num_args
            )
            .unwrap();

        let function = runtime.get_function_by_name_mut(&function_name).unwrap();
        function.is_builtin = true;
    }
}

/// Generate the continuation trampoline and return jump functions using the code generator.
/// This avoids inline assembly which gets broken by LLVM optimizer in release builds.
#[cfg(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
))]
fn compile_continuation_trampolines(runtime: &mut Runtime) {
    use crate::builtins::invoke_continuation_runtime;
    use crate::machine_code::x86_codegen::{R8, R11, R12, R13, R14, R15, RAX, RBP, RBX};
    use crate::types::BuiltInTypes;

    // Generate continuation_trampoline_generated
    // This is called when k(value) is invoked on a continuation closure
    // Arguments: RDI = closure_ptr, RSI = value
    {
        let mut lang = x86::LowLevelX86::new();

        // Standard function prologue
        lang.instructions.push(X86Asm::Push { reg: RBP });
        lang.instructions.push(X86Asm::MovRR {
            dest: RBP,
            src: RSP,
        });

        // Allocate stack space for saved_regs array (80 bytes = 10 * 8) + alignment
        // We need space for: saved_regs[0..5], closure_ptr, value, and padding
        lang.instructions.push(X86Asm::SubRI { dest: RSP, imm: 96 });

        // Save arguments to stack before we clobber any registers
        // closure_ptr (RDI) -> [RSP + 80]
        // value (RSI) -> [RSP + 88]
        lang.instructions.push(X86Asm::MovMR {
            base: RSP,
            offset: 80,
            src: RDI,
        });
        lang.instructions.push(X86Asm::MovMR {
            base: RSP,
            offset: 88,
            src: RSI,
        });

        // Save callee-saved registers (Beagle's values at the time k() was called)
        // saved_regs[0] = RBX, [1] = R12, [2] = R13, [3] = R14, [4] = R15
        lang.instructions.push(X86Asm::MovMR {
            base: RSP,
            offset: 0,
            src: RBX,
        });
        lang.instructions.push(X86Asm::MovMR {
            base: RSP,
            offset: 8,
            src: R12,
        });
        lang.instructions.push(X86Asm::MovMR {
            base: RSP,
            offset: 16,
            src: R13,
        });
        lang.instructions.push(X86Asm::MovMR {
            base: RSP,
            offset: 24,
            src: R14,
        });
        lang.instructions.push(X86Asm::MovMR {
            base: RSP,
            offset: 32,
            src: R15,
        });

        // Capture stack_pointer (RSP) and frame_pointer (RBP)
        lang.instructions.push(X86Asm::MovRR {
            dest: R14,
            src: RSP,
        }); // stack_pointer
        lang.instructions.push(X86Asm::MovRR {
            dest: R15,
            src: RBP,
        }); // frame_pointer

        // Extract continuation pointer from closure
        // closure_ptr is tagged, need to untag: untagged = closure_ptr >> 3
        lang.instructions.push(X86Asm::MovRM {
            dest: R12,
            base: RSP,
            offset: 80,
        }); // Load closure_ptr
        lang.instructions.push(X86Asm::ShrRI {
            dest: R12,
            imm: BuiltInTypes::tag_size() as u8,
        }); // Untag
        // cont_ptr is at offset 32 in the closure
        lang.instructions.push(X86Asm::MovRM {
            dest: RDX,
            base: R12,
            offset: 32,
        }); // cont_ptr -> RDX (arg3)

        // Prepare arguments for invoke_continuation_runtime
        // Args: (stack_pointer, frame_pointer, cont_ptr, value, saved_regs)
        // RDI = stack_pointer (R14)
        // RSI = frame_pointer (R15)
        // RDX = cont_ptr (already set)
        // RCX = value (from stack)
        // R8 = pointer to saved_regs array (RSP)
        lang.instructions.push(X86Asm::MovRR {
            dest: RDI,
            src: R14,
        }); // stack_pointer
        lang.instructions.push(X86Asm::MovRR {
            dest: RSI,
            src: R15,
        }); // frame_pointer
        lang.instructions.push(X86Asm::MovRM {
            dest: RCX,
            base: RBP,
            offset: -8,
        }); // value (use RBP-relative since we pushed args there... wait, no)

        // Actually, we saved value at [RSP + 88], but RSP changed after SubRI
        // Let's recalculate: after SubRI, RSP points to saved_regs[0]
        // So value is at [RSP + 88]
        lang.instructions.push(X86Asm::MovRM {
            dest: RCX,
            base: RSP,
            offset: 88,
        }); // value
        lang.instructions.push(X86Asm::MovRR { dest: R8, src: RSP }); // saved_regs pointer

        // Load function pointer and call
        let fn_ptr = invoke_continuation_runtime as *const u8 as i64;
        lang.instructions.push(X86Asm::MovRI {
            dest: RAX,
            imm: fn_ptr,
        });
        lang.instructions.push(X86Asm::CallR { target: RAX });

        // invoke_continuation_runtime never returns, but add Int3 for safety
        lang.instructions.push(X86Asm::Int3);

        let code = lang.compile_to_bytes();

        // Replace the existing continuation-trampoline function
        // add_function_mark_executable handles both writing the code and updating the function
        runtime
            .add_function_mark_executable(
                "beagle.builtin/continuation-trampoline".to_string(),
                &code,
                0,
                2, // 2 arguments: closure_ptr, value
            )
            .unwrap();
    }

    // Generate return_jump_generated
    // This is called from return_from_shift_runtime to restore state and jump back
    // We'll store this function pointer in a global so return_from_shift_runtime can call it
    {
        let mut lang = x86::LowLevelX86::new();

        // Arguments (System V AMD64 ABI):
        // RDI = new_sp
        // RSI = new_fp
        // RDX = value (return value)
        // RCX = return_address
        // R8 = callee_saved pointer
        // R9 = frame_src pointer (saved stack frame data)
        // [RSP+8] = frame_size (7th argument on stack)

        // Load frame_size from stack into R10
        lang.instructions.push(X86Asm::MovRM {
            dest: R10,
            base: RSP,
            offset: 8,
        });

        // Restore callee-saved registers from the array
        lang.instructions.push(X86Asm::MovRM {
            dest: RBX,
            base: R8,
            offset: 0,
        });
        lang.instructions.push(X86Asm::MovRM {
            dest: R12,
            base: R8,
            offset: 8,
        });
        lang.instructions.push(X86Asm::MovRM {
            dest: R13,
            base: R8,
            offset: 16,
        });
        lang.instructions.push(X86Asm::MovRM {
            dest: R14,
            base: R8,
            offset: 24,
        });
        lang.instructions.push(X86Asm::MovRM {
            dest: R15,
            base: R8,
            offset: 32,
        });

        // Restore stack state
        lang.instructions.push(X86Asm::MovRR {
            dest: RSP,
            src: RDI,
        }); // RSP = new_sp
        lang.instructions.push(X86Asm::MovRR {
            dest: RBP,
            src: RSI,
        }); // RBP = new_fp

        // Stack frame restoration for multi-shot continuations with non-empty segments.
        // Empty segments have frame_size=0 and frame_src=null, so they skip this.
        // Check if frame_size (R10) is non-zero
        lang.instructions.push(X86Asm::TestRR { a: R10, b: R10 });
        let skip_restore_label = lang.get_label_index();
        lang.instructions.push(X86Asm::Jcc {
            label_index: skip_restore_label,
            cond: crate::machine_code::x86_codegen::Condition::E, // Jump if zero
        });

        // Also check if frame_src (R9) is null
        lang.instructions.push(X86Asm::TestRR { a: R9, b: R9 });
        lang.instructions.push(X86Asm::Jcc {
            label_index: skip_restore_label,
            cond: crate::machine_code::x86_codegen::Condition::E, // Jump if null
        });

        // Restore the stack frame by copying frame_size bytes from frame_src to RSP
        // The loop uses non-callee-saved registers to avoid conflicts:
        // - R11 = offset counter
        // - RAX = temp for copying data
        // - R9 = frame_src (input, preserved)
        // - R10 = frame_size (input, preserved)
        // - RSP = destination (already set)
        // - RCX = return_address (input, must preserve)
        // - RDX = value (input, must preserve)

        // R11 = offset counter, start at 0
        lang.instructions.push(X86Asm::XorRR {
            dest: R11,
            src: R11,
        });

        let copy_loop = lang.get_label_index();
        lang.instructions.push(X86Asm::Label { index: copy_loop });

        // if offset >= frame_size, done
        lang.instructions.push(X86Asm::CmpRR { a: R11, b: R10 });
        let copy_done = lang.get_label_index();
        lang.instructions.push(X86Asm::Jcc {
            label_index: copy_done,
            cond: crate::machine_code::x86_codegen::Condition::GE,
        });

        // Load 8 bytes from [R9 + R11] into RAX
        // We need to compute the address first
        // Use R14 as temp (it's callee-saved and already restored, we'll restore it again after)
        lang.instructions.push(X86Asm::MovRR { dest: R14, src: R9 });
        lang.instructions.push(X86Asm::AddRR {
            dest: R14,
            src: R11,
        });
        lang.instructions.push(X86Asm::MovRM {
            dest: RAX,
            base: R14,
            offset: 0,
        });

        // Store 8 bytes to [RSP + R11]
        lang.instructions.push(X86Asm::MovRR {
            dest: R14,
            src: RSP,
        });
        lang.instructions.push(X86Asm::AddRR {
            dest: R14,
            src: R11,
        });
        lang.instructions.push(X86Asm::MovMR {
            base: R14,
            offset: 0,
            src: RAX,
        });

        // offset += 8
        lang.instructions.push(X86Asm::AddRI { dest: R11, imm: 8 });
        lang.instructions.push(X86Asm::Jmp {
            label_index: copy_loop,
        });

        lang.instructions.push(X86Asm::Label { index: copy_done });

        // Re-restore R14 from callee_saved array (we used it as temp)
        lang.instructions.push(X86Asm::MovRM {
            dest: R14,
            base: R8,
            offset: 24,
        });
        lang.instructions.push(X86Asm::Label {
            index: skip_restore_label,
        });

        // Put return value in RAX
        lang.instructions.push(X86Asm::MovRR {
            dest: RAX,
            src: RDX,
        });

        // Jump to return address (in RCX)
        lang.instructions.push(X86Asm::JmpR { target: RCX });

        let code = lang.compile_to_bytes();

        // Store this pointer somewhere accessible by return_from_shift_runtime
        // add_function_mark_executable handles both writing the code and registering the function
        runtime
            .add_function_mark_executable(
                "beagle.builtin/return-jump".to_string(),
                &code,
                0,
                7, // 7 arguments (6 in registers + 1 on stack)
            )
            .unwrap();
    }

    // Generate invoke_continuation_jump
    // This is called from invoke_continuation_runtime to jump to the continuation
    // Arguments (System V AMD64 ABI):
    // RDI = original_fp (used for empty stack segment case)
    // RSI = resume_address
    // RDX = stack_segment_size (0 for empty)
    // RCX = new_sp (for non-empty)
    // R8 = new_fp (for non-empty)
    // R9 = result_ptr (where to store the result value)
    // [RSP+8] = value (the result value to store)
    {
        let mut lang = x86::LowLevelX86::new();

        // Load value from stack into R11 (7th argument is at [RSP+8] after call)
        lang.instructions.push(X86Asm::MovRM {
            dest: R11,
            base: RSP,
            offset: 8, // 7th argument
        });

        // Check if stack_segment_size (RDX) is 0
        lang.instructions.push(X86Asm::TestRR { a: RDX, b: RDX });

        // If zero, jump to empty path
        let empty_label = lang.get_label_index();
        lang.instructions.push(X86Asm::Jcc {
            label_index: empty_label,
            cond: crate::machine_code::x86_codegen::Condition::E,
        });

        // Non-empty stack segment path:
        // Set RSP = new_sp (RCX), RBP = new_fp (R8)
        lang.instructions.push(X86Asm::MovRR {
            dest: RSP,
            src: RCX,
        });
        lang.instructions.push(X86Asm::MovRR { dest: RBP, src: R8 });
        // Write result value: [R9] = R11 (skip if R9 == 0, i.e. builtin-thrown error)
        let skip_write_nonempty = lang.get_label_index();
        lang.instructions.push(X86Asm::TestRR { a: R9, b: R9 });
        lang.instructions.push(X86Asm::Jcc {
            label_index: skip_write_nonempty,
            cond: crate::machine_code::x86_codegen::Condition::E,
        });
        lang.instructions.push(X86Asm::MovMR {
            base: R9,
            offset: 0,
            src: R11,
        });
        lang.instructions.push(X86Asm::Label {
            index: skip_write_nonempty,
        });
        // Set RAX = value so builtin-thrown errors resume with correct return value
        lang.instructions.push(X86Asm::MovRR {
            dest: RAX,
            src: R11,
        });
        // Jump to resume_address (RSI)
        lang.instructions.push(X86Asm::JmpR { target: RSI });

        // Empty stack segment path:
        // For empty segments, restore the original stack/frame pointers from when
        // the continuation was captured. RCX contains original_sp, R8 contains original_fp.
        lang.instructions.push(X86Asm::Label { index: empty_label });
        // Set RSP = original_sp (RCX), RBP = original_fp (R8)
        lang.instructions.push(X86Asm::MovRR {
            dest: RSP,
            src: RCX,
        });
        lang.instructions.push(X86Asm::MovRR { dest: RBP, src: R8 });
        // Write result value: [R9] = R11 (skip if R9 == 0, i.e. builtin-thrown error)
        let skip_write_empty = lang.get_label_index();
        lang.instructions.push(X86Asm::TestRR { a: R9, b: R9 });
        lang.instructions.push(X86Asm::Jcc {
            label_index: skip_write_empty,
            cond: crate::machine_code::x86_codegen::Condition::E,
        });
        lang.instructions.push(X86Asm::MovMR {
            base: R9,
            offset: 0,
            src: R11,
        });
        lang.instructions.push(X86Asm::Label {
            index: skip_write_empty,
        });
        // Set RAX = value so builtin-thrown errors resume with correct return value
        lang.instructions.push(X86Asm::MovRR {
            dest: RAX,
            src: R11,
        });
        // Jump to resume_address (RSI)
        lang.instructions.push(X86Asm::JmpR { target: RSI });

        let code = lang.compile_to_bytes();
        runtime
            .add_function_mark_executable(
                "beagle.builtin/invoke-continuation-jump".to_string(),
                &code,
                0,
                7, // 7 arguments
            )
            .unwrap();
    }

    // Generate continuation-return-stub
    // When a segmented continuation's outermost frame returns via `ret`, RSP is
    // 16-byte aligned (frame pointers are 16-aligned, and `leave; ret` adds 16).
    // But Rust functions expect entry with RSP ≡ 8 (mod 16) (as after a `call`).
    // This stub adjusts the alignment before jumping to continuation_return_trampoline.
    {
        use crate::builtins::continuation_return_trampoline;

        let mut lang = x86::LowLevelX86::new();
        // Pass the JIT return value (RAX) as the first argument (RDI) so
        // the Rust function receives it without needing inline asm to read RAX.
        lang.instructions.push(X86Asm::MovRR {
            dest: RDI,
            src: RAX,
        });
        // After Beagle `ret`, RSP is 16-aligned. Sub 8 to make it 8-mod-16.
        lang.instructions.push(X86Asm::SubRI { dest: RSP, imm: 8 });
        // Load and jump to continuation_return_trampoline (which is -> !)
        let trampoline_ptr = continuation_return_trampoline as *const u8 as i64;
        lang.instructions.push(X86Asm::MovRI {
            dest: R11,
            imm: trampoline_ptr,
        });
        lang.instructions.push(X86Asm::JmpR { target: R11 });

        let code = lang.compile_to_bytes();
        runtime
            .add_function_mark_executable(
                "beagle.builtin/continuation-return-stub".to_string(),
                &code,
                0,
                1,
            )
            .unwrap();

        let function = runtime
            .get_function_by_name_mut("beagle.builtin/continuation-return-stub")
            .unwrap();
        function.is_builtin = true;
    }

    // ========================================================================
    // x86-64 handler-jump trampoline
    // Simple SP/FP restore + jump (no callee-saved restore).
    // Args: RDI=new_sp, RSI=new_fp, RDX=jump_target
    // ========================================================================
    {
        let mut lang = x86::LowLevelX86::new();
        lang.instructions.push(X86Asm::MovRR {
            dest: RSP,
            src: RDI,
        });
        lang.instructions.push(X86Asm::MovRR {
            dest: RBP,
            src: RSI,
        });
        lang.instructions.push(X86Asm::JmpR { target: RDX });
        let code = lang.compile_to_bytes();
        runtime
            .add_function_mark_executable("beagle.builtin/handler-jump".to_string(), &code, 0, 3)
            .unwrap();
        let function = runtime
            .get_function_by_name_mut("beagle.builtin/handler-jump")
            .unwrap();
        function.is_builtin = true;
    }

    // ========================================================================
    // x86-64 stack-switch trampoline
    // Args: RDI=stack_top, RSI=target function pointer
    // ========================================================================
    {
        let mut lang = x86::LowLevelX86::new();
        // Switch to new stack and align to 16 bytes
        lang.instructions.push(X86Asm::MovRR {
            dest: RSP,
            src: RDI,
        });
        lang.instructions.push(X86Asm::AndRI {
            dest: RSP,
            imm: -16,
        });
        // Call target function
        lang.instructions.push(X86Asm::CallR { target: RSI });
        // Unreachable trap
        lang.instructions.push(X86Asm::Ud2);
        let code = lang.compile_to_bytes();
        runtime
            .add_function_mark_executable("beagle.builtin/stack-switch".to_string(), &code, 0, 2)
            .unwrap();
        let function = runtime
            .get_function_by_name_mut("beagle.builtin/stack-switch")
            .unwrap();
        function.is_builtin = true;
    }

    // ========================================================================
    // x86-64 read-fp trampoline: returns RBP
    // ========================================================================
    {
        let mut lang = x86::LowLevelX86::new();
        lang.instructions.push(X86Asm::MovRR {
            dest: RAX,
            src: RBP,
        });
        lang.instructions.push(X86Asm::Ret);
        let code = lang.compile_to_bytes();
        runtime
            .add_function_mark_executable("beagle.builtin/read-fp".to_string(), &code, 0, 0)
            .unwrap();
        let function = runtime
            .get_function_by_name_mut("beagle.builtin/read-fp")
            .unwrap();
        function.is_builtin = true;
    }

    // ========================================================================
    // x86-64 read-sp trampoline: returns caller's RSP (compensate for CALL push)
    // ========================================================================
    {
        let mut lang = x86::LowLevelX86::new();
        // LEA RAX, [RSP+8] to get caller's RSP (CALL pushed 8 bytes)
        lang.instructions.push(X86Asm::LeaRspOffset {
            dest: RAX,
            offset: 8,
        });
        lang.instructions.push(X86Asm::Ret);
        let code = lang.compile_to_bytes();
        runtime
            .add_function_mark_executable("beagle.builtin/read-sp".to_string(), &code, 0, 0)
            .unwrap();
        let function = runtime
            .get_function_by_name_mut("beagle.builtin/read-sp")
            .unwrap();
        function.is_builtin = true;
    }

    // ========================================================================
    // x86-64 read-sp-fp trampoline: returns RAX=caller's RSP, RDX=RBP
    // ========================================================================
    {
        let mut lang = x86::LowLevelX86::new();
        lang.instructions.push(X86Asm::LeaRspOffset {
            dest: RAX,
            offset: 8,
        });
        lang.instructions.push(X86Asm::MovRR {
            dest: RDX,
            src: RBP,
        });
        lang.instructions.push(X86Asm::Ret);
        let code = lang.compile_to_bytes();
        runtime
            .add_function_mark_executable("beagle.builtin/read-sp-fp".to_string(), &code, 0, 0)
            .unwrap();
        let function = runtime
            .get_function_by_name_mut("beagle.builtin/read-sp-fp")
            .unwrap();
        function.is_builtin = true;
    }

    // ========================================================================
    // x86-64 save-callee-regs trampoline: saves RBX, R12-R15 to [RDI]
    // ========================================================================
    {
        let mut lang = x86::LowLevelX86::new();
        lang.instructions.push(X86Asm::MovMR {
            base: RDI,
            offset: 0,
            src: RBX,
        });
        lang.instructions.push(X86Asm::MovMR {
            base: RDI,
            offset: 8,
            src: R12,
        });
        lang.instructions.push(X86Asm::MovMR {
            base: RDI,
            offset: 16,
            src: R13,
        });
        lang.instructions.push(X86Asm::MovMR {
            base: RDI,
            offset: 24,
            src: R14,
        });
        lang.instructions.push(X86Asm::MovMR {
            base: RDI,
            offset: 32,
            src: R15,
        });
        lang.instructions.push(X86Asm::Ret);
        let code = lang.compile_to_bytes();
        runtime
            .add_function_mark_executable(
                "beagle.builtin/save-callee-regs".to_string(),
                &code,
                0,
                1,
            )
            .unwrap();
        let function = runtime
            .get_function_by_name_mut("beagle.builtin/save-callee-regs")
            .unwrap();
        function.is_builtin = true;
    }
}

#[derive(Debug, Clone)]
pub struct CommandLineArguments {
    program: Option<String>,
    program_args: Vec<String>,
    show_times: bool,
    show_gc_times: bool,
    print_ast: bool,
    no_gc: bool,
    gc_always: bool,
    test: bool,
    debug: bool,
    verbose: bool,
    no_std: bool,
    print_parse: bool,
    print_builtin_calls: bool,
    update_snapshots: bool,
    include_paths: Vec<String>,
}

fn load_default_files(runtime: &mut Runtime) -> Result<Vec<String>, Box<dyn Error>> {
    let resource_files: [&str; 0] = [];
    let stdlib_files = [
        "std.bg",
        "beagle.ffi.bg",
        "beagle.io.bg",
        "beagle.effect.bg",
        "beagle.async.bg",
        "beagle.fs.bg",
        "beagle.timer.bg",
        "beagle.socket.bg",
        "beagle.stream.bg",
        "beagle.repl-session.bg",
        "beagle.repl.bg",
        "beagle.repl-interactive.bg",
    ];
    let mut all_top_levels = vec![];

    for file_name in resource_files {
        let file_path = find_resource_file(file_name)?;
        let top_levels = runtime.compile(&file_path)?;
        all_top_levels.extend(top_levels);
    }

    for file_name in stdlib_files {
        match find_stdlib_file(file_name) {
            Ok(file_path) => {
                let top_levels = runtime.compile(&file_path)?;
                all_top_levels.extend(top_levels);
            }
            Err(_) => {
                let source = embedded_stdlib::get(file_name)
                    .ok_or_else(|| format!("Could not find stdlib file: {}", file_name))?;
                let top_levels = runtime.compile_source(file_name, source)?;
                all_top_levels.extend(top_levels);
            }
        }
    }

    Ok(all_top_levels)
}

fn find_resource_file(file_name: &str) -> Result<String, Box<dyn Error>> {
    let mut exe_path = env::current_exe()?;
    exe_path = exe_path.parent().unwrap().to_path_buf();
    let mut candidates = Vec::new();
    candidates.push(exe_path.join(format!("resources/{}", file_name)));

    if let Some(parent) = exe_path.parent() {
        candidates.push(parent.join(format!("resources/{}", file_name)));
        if let Some(grandparent) = parent.parent() {
            candidates.push(grandparent.join(format!("resources/{}", file_name)));
            if let Some(great_grandparent) = grandparent.parent() {
                candidates.push(great_grandparent.join(format!("resources/{}", file_name)));
            }
        }
    }

    for candidate in candidates.into_iter() {
        if candidate.exists() {
            return Ok(candidate.to_str().unwrap().to_string());
        }
    }

    Err(format!("Could not find resource file: {}", file_name).into())
}

fn find_stdlib_file(file_name: &str) -> Result<String, Box<dyn Error>> {
    let mut exe_path = env::current_exe()?;
    exe_path = exe_path.parent().unwrap().to_path_buf();
    let mut candidates = Vec::new();
    candidates.push(exe_path.join(format!("standard-library/{}", file_name)));

    if let Some(parent) = exe_path.parent() {
        candidates.push(parent.join(format!("standard-library/{}", file_name)));
        if let Some(grandparent) = parent.parent() {
            candidates.push(grandparent.join(format!("standard-library/{}", file_name)));
            if let Some(great_grandparent) = grandparent.parent() {
                candidates.push(great_grandparent.join(format!("standard-library/{}", file_name)));
            }
        }
    }

    for path in candidates.into_iter() {
        if path.exists() {
            return Ok(path.to_str().unwrap().to_string());
        }
    }

    Err(format!("Could not find standard library file: {}", file_name).into())
}

// --- New user-facing CLI ---

#[derive(ClapParser, Debug)]
#[command(
    name = "beag",
    version,
    about = "The Beagle programming language",
    long_about = "Beagle is a dynamically-typed, functional programming language that compiles directly to native machine code."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Run a Beagle program
    Run(RunArgs),
    /// Start the interactive REPL
    Repl,
    /// Initialize a new Beagle project
    Init(InitArgs),
    /// Run tests in the current project
    Test(TestArgs),
    /// Export all documentation to JSON
    ExportDocs,
}

#[derive(clap::Args, Debug)]
struct RunArgs {
    /// The .bg file to run
    file: String,
    /// Arguments to pass to the program's main(args) function
    #[clap(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
    #[clap(long)]
    show_times: bool,
    #[clap(long)]
    show_gc_times: bool,
    #[clap(long)]
    print_ast: bool,
    #[clap(long)]
    no_gc: bool,
    #[clap(long)]
    gc_always: bool,
    #[clap(long)]
    debug: bool,
    #[clap(long)]
    verbose: bool,
    #[clap(long)]
    no_std: bool,
    #[clap(long)]
    print_parse: bool,
    #[clap(long)]
    print_builtin_calls: bool,
    /// Internal: run in test mode (used by `beag test` subprocesses)
    #[clap(long, hide = true)]
    test: bool,
    /// Update snapshot expectations to match actual output
    #[clap(long)]
    update_snapshots: bool,
    /// Additional directories to search for source files
    #[clap(short = 'I', long = "include")]
    include: Vec<String>,
}

#[derive(clap::Args, Debug)]
struct InitArgs {
    /// Project name (defaults to current directory name)
    name: Option<String>,
}

#[derive(clap::Args, Debug)]
struct TestArgs {
    /// File or directory to test (defaults to test/ or tests/)
    path: Option<String>,
    /// Update snapshot expectations to match actual output
    #[clap(long)]
    update_snapshots: bool,
}

impl CommandLineArguments {
    fn for_run(file: String, args: Vec<String>) -> Self {
        Self {
            program: Some(file),
            program_args: args,
            show_times: false,
            show_gc_times: false,
            print_ast: false,
            no_gc: false,
            gc_always: false,
            test: false,
            debug: false,
            verbose: false,
            no_std: false,
            print_parse: false,
            print_builtin_calls: false,
            update_snapshots: false,
            include_paths: vec![],
        }
    }

    fn default() -> Self {
        Self {
            program: None,
            program_args: vec![],
            show_times: false,
            show_gc_times: false,
            print_ast: false,
            no_gc: false,
            gc_always: false,
            test: false,
            debug: false,
            verbose: false,
            no_std: false,
            print_parse: false,
            print_builtin_calls: false,
            update_snapshots: false,
            include_paths: vec![],
        }
    }

    fn from_run_args(run_args: RunArgs) -> Self {
        Self {
            program: Some(run_args.file),
            program_args: run_args.args,
            show_times: run_args.show_times,
            show_gc_times: run_args.show_gc_times,
            print_ast: run_args.print_ast,
            no_gc: run_args.no_gc,
            gc_always: run_args.gc_always,
            test: run_args.test,
            debug: run_args.debug,
            verbose: run_args.verbose,
            no_std: run_args.no_std,
            print_parse: run_args.print_parse,
            print_builtin_calls: run_args.print_builtin_calls,
            update_snapshots: run_args.update_snapshots,
            include_paths: run_args.include,
        }
    }
}

fn cmd_init(init_args: InitArgs) -> Result<(), Box<dyn Error>> {
    let project_name = match init_args.name {
        Some(name) => {
            std::fs::create_dir_all(&name)?;
            std::env::set_current_dir(&name)?;
            name
        }
        None => {
            let cwd = std::env::current_dir()?;
            cwd.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("my-project")
                .to_string()
        }
    };

    let toml_content = format!(
        "[project]\nname = \"{}\"\nversion = \"0.1.0\"\n",
        project_name
    );
    std::fs::write("beagle.toml", toml_content)?;

    std::fs::create_dir_all("src")?;
    std::fs::write(
        "src/main.bg",
        format!(
            "namespace {}\n\nfn main() {{\n    println(\"Hello from {}!\")\n}}\n",
            project_name.replace('-', "_"),
            project_name,
        ),
    )?;

    std::fs::create_dir_all("test")?;
    std::fs::write(
        "test/main_test.bg",
        format!(
            "namespace {}_test\n\nfn main() {{\n    println(\"ok\")\n}}\n\n// @beagle.core.snapshot\n// ok\n",
            project_name.replace('-', "_"),
        ),
    )?;

    println!("Created new Beagle project: {}", project_name);
    println!();
    println!("  beagle.toml");
    println!("  src/main.bg");
    println!("  test/main_test.bg");
    println!();
    println!("Run your project:");
    println!("  beag run src/main.bg");
    Ok(())
}

fn discover_test_files(dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>, Box<dyn Error>> {
    let mut test_files = vec![];
    discover_test_files_recursive(dir, &mut test_files)?;
    test_files.sort();
    Ok(test_files)
}

/// Sort test files so that dependencies come before dependents.
/// Parses `namespace` declarations and `use` statements from each file to build
/// a dependency graph, then topological-sorts it. Files without dependencies
/// (or with only stdlib deps) retain their original alphabetical order.
fn sort_tests_by_deps(files: &mut [std::path::PathBuf]) {
    use std::collections::{HashMap, HashSet, VecDeque};

    // Build namespace→index map from each file's `namespace` declaration
    let mut ns_to_idx: HashMap<String, usize> = HashMap::new();
    let mut file_sources: Vec<String> = Vec::with_capacity(files.len());

    for (i, file) in files.iter().enumerate() {
        let source = std::fs::read_to_string(file).unwrap_or_default();
        // Extract namespace name from first `namespace` line
        if let Some(ns) = source.lines().find_map(|line| {
            let trimmed = line.trim();
            trimmed
                .strip_prefix("namespace ")
                .map(|rest| rest.trim().to_string())
        }) {
            ns_to_idx.insert(ns, i);
        }
        file_sources.push(source);
    }

    // Build adjacency list: edges[i] = set of indices that file i depends on
    let n = files.len();
    let mut deps: Vec<Vec<usize>> = vec![vec![]; n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for (i, source) in file_sources.iter().enumerate() {
        for line in source.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("use ") {
                // Parse `use <namespace> as <alias>` or `use <namespace>`
                let ns_name = rest.split_whitespace().next().unwrap_or("");
                if let Some(&dep_idx) = ns_to_idx.get(ns_name)
                    && dep_idx != i
                {
                    deps[i].push(dep_idx);
                    in_degree[dep_idx] += 0; // ensure entry exists
                }
            }
        }
    }

    // Kahn's algorithm for topological sort (stable: ties broken by original order)
    // We want dependencies first, so edges go from dependent → dependency.
    // Reverse: in_degree counts how many files depend on this file.
    // Actually, we want: if A depends on B, B should come first.
    // So edges: A → B means "A depends on B". We want B before A.
    // in_degree[i] = number of unsatisfied dependencies of i.
    let mut in_deg: Vec<usize> = vec![0; n];
    for (i, dep_list) in deps.iter().enumerate() {
        in_deg[i] = dep_list.len();
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    // Start with files that have no dependencies (in_deg == 0), in original order
    for (i, &deg) in in_deg.iter().enumerate().take(n) {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    // Build reverse adjacency: who depends on me?
    let mut dependents: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, dep_list) in deps.iter().enumerate() {
        for &dep in dep_list {
            dependents[dep].push(i);
        }
    }

    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(idx) = queue.pop_front() {
        order.push(idx);
        // Sort dependents to maintain stable order for ties
        let mut next: Vec<usize> = dependents[idx].clone();
        next.sort();
        for dep_idx in next {
            in_deg[dep_idx] -= 1;
            if in_deg[dep_idx] == 0 {
                queue.push_back(dep_idx);
            }
        }
    }

    // If there are cycles, just append remaining files in original order
    if order.len() < n {
        let in_order: HashSet<usize> = order.iter().copied().collect();
        for i in 0..n {
            if !in_order.contains(&i) {
                order.push(i);
            }
        }
    }

    // Reorder files according to topological order
    let original: Vec<std::path::PathBuf> = files.to_vec();
    for (i, &orig_idx) in order.iter().enumerate() {
        files[i] = original[orig_idx].clone();
    }
}

fn discover_test_files_recursive(
    dir: &std::path::Path,
    test_files: &mut Vec<std::path::PathBuf>,
) -> Result<(), Box<dyn Error>> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            discover_test_files_recursive(&path, test_files)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("bg") {
            let content = std::fs::read_to_string(&path)?;
            if content.contains("// @beagle.core.snapshot") || content.contains("test \"") {
                test_files.push(path);
            }
        }
    }
    Ok(())
}

fn cmd_test(test_args: TestArgs) -> Result<(), Box<dyn Error>> {
    let mut test_files: Vec<std::path::PathBuf> = vec![];

    if let Some(ref path_str) = test_args.path {
        let path = std::path::PathBuf::from(path_str);
        if !path.exists() {
            return Err(format!("Path not found: {}", path_str).into());
        }
        if path.is_dir() {
            test_files = discover_test_files(&path)?;
            if test_files.is_empty() {
                println!("No test files found in {}", path_str);
                return Ok(());
            }
        } else {
            test_files.push(path);
        }
    } else {
        // Default: look for test/ or tests/ directory
        let test_dir = if std::path::Path::new("test").is_dir() {
            "test"
        } else if std::path::Path::new("tests").is_dir() {
            "tests"
        } else {
            return Err(
                "No test/ or tests/ directory found. Create one with .bg test files.".into(),
            );
        };
        test_files = discover_test_files(std::path::Path::new(test_dir))?;
        if test_files.is_empty() {
            println!("No test files found in {}/", test_dir);
            return Ok(());
        }
    }

    sort_tests_by_deps(&mut test_files);

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    for test_file in &test_files {
        let file_str = test_file.to_str().unwrap();
        let source = std::fs::read_to_string(test_file)?;

        if source.contains("// Skip") {
            let skip_reason = source
                .lines()
                .find(|l| l.contains("// Skip"))
                .and_then(|l| l.split("// Skip:").nth(1))
                .map(|r| r.trim().to_string());
            println!(
                "  skip  {}{}",
                file_str,
                skip_reason.map_or(String::new(), |r| format!(" ({})", r))
            );
            skipped += 1;
            continue;
        }

        let gc_always = source.contains("// gc-always");
        let no_std = source.contains("// no-std");

        let exe = std::env::current_exe()?;
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("run").arg("--test").arg(file_str);
        if gc_always {
            cmd.arg("--gc-always");
        }
        if no_std {
            cmd.arg("--no-std");
        }
        if test_args.update_snapshots {
            cmd.arg("--update-snapshots");
        }

        let output = cmd.output()?;

        if output.status.success() {
            println!("  pass  {}", file_str);
            passed += 1;
        } else {
            println!("  FAIL  {}", file_str);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stderr.lines().chain(stdout.lines()) {
                println!("        {}", line);
            }
            failed += 1;
        }
    }

    println!();
    println!(
        "{} passed, {} failed, {} skipped ({} total)",
        passed,
        failed,
        skipped,
        test_files.len()
    );

    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let raw_args: Vec<String> = std::env::args().collect();

    // Bare file: beag file.bg [args...]
    if raw_args.len() > 1 && raw_args[1].ends_with(".bg") {
        let args = CommandLineArguments::for_run(raw_args[1].clone(), raw_args[2..].to_vec());
        return main_inner(args);
    }

    // Subcommand mode
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(run_args) => {
            let args = CommandLineArguments::from_run_args(run_args);
            main_inner(args)
        }
        Commands::Repl => {
            let args = CommandLineArguments::default();
            run_repl(args)
        }
        Commands::Init(init_args) => cmd_init(init_args),
        Commands::Test(test_args) => cmd_test(test_args),
        Commands::ExportDocs => export_docs(CommandLineArguments::default()),
    }
}

#[cfg(test)]
fn find_beag_binary() -> std::path::PathBuf {
    // In cargo test, current_exe() returns target/<profile>/deps/beag-<hash>.
    // The actual beag binary is at target/<profile>/beag.
    let test_exe = std::env::current_exe().expect("Failed to get current exe");
    let deps_dir = test_exe.parent().expect("No parent for test exe");
    let profile_dir = deps_dir.parent().expect("No parent for deps dir");
    profile_dir.join("beag")
}

#[cfg(test)]
fn run_all_tests(args: CommandLineArguments) -> Result<(), Box<dyn Error>> {
    let mut test_files = discover_test_files(std::path::Path::new("resources"))?;
    sort_tests_by_deps(&mut test_files);
    let exe = find_beag_binary();

    for test_file in &test_files {
        let path = test_file.to_str().unwrap();
        let source: String = std::fs::read_to_string(path)?;

        // Check for // Skip annotation (supports "// Skip" or "// Skip: reason")
        if let Some(skip_line) = source.lines().find(|line| line.starts_with("// Skip")) {
            let reason = skip_line
                .strip_prefix("// Skip")
                .unwrap()
                .trim_start_matches(':')
                .trim();
            if reason.is_empty() {
                println!("Skipping test: {}", path);
            } else {
                println!("Skipping test: {} ({})", path, reason);
            }
            continue;
        }

        eprintln!("[test] START {}", path);
        let gc_always = args.gc_always || source.contains("// gc-always");
        let no_std = args.no_std || source.contains("// no-std");

        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("run").arg("--test").arg(path);
        if gc_always {
            cmd.arg("--gc-always");
        }
        if no_std {
            cmd.arg("--no-std");
        }
        cmd.stdout(std::process::Stdio::piped());

        let start = std::time::Instant::now();
        let mut child = cmd.spawn()?;
        let pid = child.id();

        // Sample peak RSS while child runs
        let peak_rss_kb = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let peak_rss_clone = peak_rss_kb.clone();
        let monitor = std::thread::spawn(move || {
            let status_path = format!("/proc/{}/status", pid);
            loop {
                if let Ok(contents) = std::fs::read_to_string(&status_path) {
                    for line in contents.lines() {
                        if let Some(val) = line.strip_prefix("VmRSS:") {
                            if let Ok(kb) = val.trim().trim_end_matches(" kB").trim().parse::<u64>() {
                                peak_rss_clone.fetch_max(kb, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                } else {
                    break; // process exited
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });

        // Read stdout concurrently with wait to avoid deadlock when the pipe
        // buffer fills up (child blocks on write, parent blocks on wait).
        let stdout_handle = child.stdout.take().map(|stdout| {
            std::thread::spawn(move || {
                use std::io::Read;
                let mut buf = Vec::new();
                let mut stdout = stdout;
                let _ = stdout.read_to_end(&mut buf);
                buf
            })
        });
        let output_status = child.wait()?;
        let stdout = stdout_handle
            .map(|h| h.join().unwrap_or_default())
            .unwrap_or_default();
        let _ = monitor.join();
        let elapsed = start.elapsed();
        let rss = peak_rss_kb.load(std::sync::atomic::Ordering::Relaxed);

        if !output_status.success() {
            let stdout_str = String::from_utf8_lossy(&stdout);
            eprintln!("[test] FAIL {} ({:.2}s, peak_rss={}KB)", path, elapsed.as_secs_f64(), rss);
            return Err(format!("Test failed: {}\n{}", path, stdout_str).into());
        }
        eprintln!("[test] PASS {} ({:.2}s, peak_rss={}KB)", path, elapsed.as_secs_f64(), rss);
    }
    Ok(())
}

fn run_repl(args: CommandLineArguments) -> Result<(), Box<dyn Error>> {
    let args_clone = args.clone();

    RUNTIME.get_or_init(|| {
        let allocator = Alloc::new(get_allocate_options(&args_clone));
        let printer: Box<dyn Printer> = Box::new(DefaultPrinter);
        let runtime = Runtime::new(args_clone, allocator, printer);
        SyncUnsafeCell::new(runtime)
    });

    let runtime = RUNTIME.get().unwrap().get_mut();

    runtime.start_compiler_thread();

    compile_trampoline(runtime);
    compile_apply_call_trampolines(runtime);
    cfg_if::cfg_if! {
        if #[cfg(any(feature = "backend-x86-64", all(target_arch = "x86_64", not(feature = "backend-arm64"))))] {
            let max_wrapper_args = 5;
        } else {
            let max_wrapper_args = 6;
        }
    }
    for i in 0..=max_wrapper_args {
        compile_save_volatile_registers_for(runtime, i);
    }

    let pause_atom_ptr = runtime.pause_atom_ptr();
    runtime.set_pause_atom_ptr(pause_atom_ptr);

    runtime.initialize_thread_global()?;
    runtime.initialize_namespaces()?;
    runtime.register_function_struct();
    runtime.install_builtins()?;

    #[cfg(any(
        feature = "backend-x86-64",
        all(target_arch = "x86_64", not(feature = "backend-arm64"))
    ))]
    compile_continuation_trampolines(runtime);
    #[cfg(all(target_arch = "aarch64", not(feature = "backend-x86-64")))]
    compile_arm_continuation_return_stub(runtime);

    let mut top_levels = vec![];
    if !args.no_std {
        top_levels = load_default_files(runtime)?;
    }

    for top_level in top_levels {
        if let Some(f) = runtime.get_function0(&top_level) {
            f();
        }
    }

    // Run the Beagle REPL program through __main__ so it gets the async handler
    let repl_main = "beagle.repl-interactive/main";

    // Register main thread so child-triggered GC waits for us.
    {
        let _guard = runtime.gc_lock.lock().unwrap();
        let new_count = runtime
            .registered_thread_count
            .fetch_add(1, std::sync::atomic::Ordering::Release)
            + 1;
        gc::usdt_probes::fire_thread_register(new_count);
    }

    let async_main_wrapper = "beagle.async/__main__";
    let has_async_wrapper = runtime.get_function_arity(async_main_wrapper).is_some();

    if has_async_wrapper {
        let _stack_pointer = runtime.get_stack_base();
        let main_fn = runtime.find_function(repl_main).unwrap();
        let main_fn_ptr = runtime.get_function_pointer(main_fn).unwrap();
        let tagged_main_fn = ((main_fn_ptr as u64) << 3) | 0b100;
        let args_or_null = 0b111u64; // null for 0-arity main

        let wrapper = runtime.get_function2(async_main_wrapper).unwrap();
        wrapper(tagged_main_fn, args_or_null);
    } else {
        let f = runtime
            .get_function0(repl_main)
            .expect("beagle.repl-interactive/main not found");
        f();
    }

    runtime.wait_for_other_threads();
    runtime.event_loops.shutdown_all();

    Ok(())
}

cfg_if::cfg_if! {
    if #[cfg(feature = "compacting")] {
        pub type Alloc = MutexAllocator<CompactingHeap>;
    } else if #[cfg(feature = "mark-and-sweep")] {
        pub type Alloc = MutexAllocator<MarkAndSweep>;
    } else if #[cfg(feature = "generational")] {
        pub type Alloc = MutexAllocator<GenerationalGC>;
    } else {
        // Default to generational GC
        pub type Alloc = MutexAllocator<GenerationalGC>;
    }
}

fn export_docs(args: CommandLineArguments) -> Result<(), Box<dyn Error>> {
    let args_clone = args.clone();

    RUNTIME.get_or_init(|| {
        let allocator = Alloc::new(get_allocate_options(&args_clone));
        let printer: Box<dyn Printer> = Box::new(DefaultPrinter);
        let runtime = Runtime::new(args_clone, allocator, printer);
        SyncUnsafeCell::new(runtime)
    });

    let runtime = RUNTIME.get().unwrap().get_mut();

    runtime.start_compiler_thread();

    compile_trampoline(runtime);
    compile_apply_call_trampolines(runtime);
    runtime.register_function_struct();
    runtime.install_builtins()?;
    if !args.no_std {
        load_default_files(runtime)?;
    }

    // Export documentation as JSON
    let docs = runtime.export_documentation();
    println!("{}", docs);

    Ok(())
}

fn main_inner(mut args: CommandLineArguments) -> Result<(), Box<dyn Error>> {
    // Register USDT probes for DTrace
    if let Err(e) = gc::usdt_probes::register() {
        eprintln!("Warning: Failed to register USDT probes: {}", e);
    }

    if args.program.is_none() {
        println!("No program provided. Use --repl for interactive mode.");
        return Ok(());
    }
    let program = args.program.clone().unwrap();
    let source = std::fs::read_to_string(program.clone())?;
    // TODO: This is very ad-hoc
    // I should make it real functionality later
    // but right now I just want something working
    let has_expect = args.test && source.contains("// @beagle.core.snapshot");

    let args_clone = args.clone();

    if source.contains("// no-std") {
        args.no_std = true;
    }

    RUNTIME.get_or_init(|| {
        let allocator = Alloc::new(get_allocate_options(&args_clone.clone()));
        let printer: Box<dyn Printer> = if has_expect {
            Box::new(TestPrinter::new(Box::new(DefaultPrinter)))
        } else {
            Box::new(DefaultPrinter)
        };

        let runtime = Runtime::new(args_clone, allocator, printer);
        SyncUnsafeCell::new(runtime)
    });

    // let allocator = Alloc::new(get_allocate_options(&args));
    // let printer: Box<dyn Printer> = if has_expect {
    //     Box::new(TestPrinter::new(Box::new(DefaultPrinter)))
    // } else {
    //     Box::new(DefaultPrinter)
    // };

    let runtime = RUNTIME.get().unwrap().get_mut();

    runtime.start_compiler_thread();

    compile_trampoline(runtime);
    compile_apply_call_trampolines(runtime);
    // x86-64 has 6 arg registers (0-5), ARM64 has 8 (0-7)
    // The wrapper uses the last arg register for the function pointer
    cfg_if::cfg_if! {
        if #[cfg(any(feature = "backend-x86-64", all(target_arch = "x86_64", not(feature = "backend-arm64"))))] {
            let max_wrapper_args = 5; // Uses arg0-4 for data, arg5 for fn ptr
        } else {
            let max_wrapper_args = 6; // Uses arg0-5 for data, arg6 for fn ptr (ARM supports 8 args)
        }
    }
    for i in 0..=max_wrapper_args {
        compile_save_volatile_registers_for(runtime, i);
    }

    let pause_atom_ptr = runtime.pause_atom_ptr();
    runtime.set_pause_atom_ptr(pause_atom_ptr);

    // Initialize GlobalObject for main thread before any heap allocations that use roots
    runtime.initialize_thread_global()?;

    // Initialize the namespaces atom in GlobalObject slot 0
    runtime.initialize_namespaces()?;

    runtime.register_function_struct();
    runtime.install_builtins()?;

    // Generate continuation trampolines using the code generator (x86-64 only)
    // This replaces the Rust inline assembly version which gets broken by LLVM optimizer
    #[cfg(any(
        feature = "backend-x86-64",
        all(target_arch = "x86_64", not(feature = "backend-arm64"))
    ))]
    compile_continuation_trampolines(runtime);
    #[cfg(all(target_arch = "aarch64", not(feature = "backend-x86-64")))]
    compile_arm_continuation_return_stub(runtime);

    let compile_time = Instant::now();

    let mut top_levels = vec![];
    if !args.no_std {
        top_levels = load_default_files(runtime)?;
    }

    // TODO: Need better name for top_level
    // It should really be the top level of a namespace
    let new_top_levels = runtime.compile(&program)?;
    let current_namespace = runtime.current_namespace_id();
    top_levels.extend(new_top_levels);

    runtime.write_functions_to_pid_map();

    runtime.check_functions()?;
    if args.show_times {
        println!("Compile time {:?}", compile_time.elapsed());
    }

    let time = Instant::now();

    for top_level in top_levels {
        if let Some(f) = runtime.get_function0(&top_level) {
            f();
        } else {
            panic!(
                "Internal error: top-level initializer '{}' was compiled but not found at runtime. \
                This is a compiler bug.",
                top_level
            );
        }
    }
    runtime.set_current_namespace(current_namespace);

    let fully_qualified_main = runtime.current_namespace_name() + "/main";

    // Check if main exists and how many arguments it takes
    if let Some(arity) = runtime.get_function_arity(&fully_qualified_main) {
        // Register main thread so child-triggered GC waits for us.
        // This mirrors what run_thread does for child threads.
        {
            let _guard = runtime.gc_lock.lock().unwrap();
            let new_count = runtime
                .registered_thread_count
                .fetch_add(1, std::sync::atomic::Ordering::Release)
                + 1;
            gc::usdt_probes::fire_thread_register(new_count);
        }

        // Check if beagle.async/__main__ exists (implicit async handler wrapper)
        // If it does, call it with the main function pointer; otherwise call main directly
        let async_main_wrapper = "beagle.async/__main__";
        let has_async_wrapper = runtime.get_function_arity(async_main_wrapper).is_some();

        let result = if has_async_wrapper {
            // Call __main__(main_fn, args) - it will set up the async handler
            let stack_pointer = runtime.get_stack_base();
            let main_fn = runtime.find_function(&fully_qualified_main).unwrap();
            let main_fn_ptr = runtime.get_function_pointer(main_fn).unwrap();
            // Tag the function pointer as BuiltInTypes::Function (tag 0b100)
            let tagged_main_fn = ((main_fn_ptr as u64) << 3) | 0b100;

            let args_or_null = if arity == 1 {
                runtime
                    .create_string_array(stack_pointer, &args.program_args)
                    .expect("Failed to create args array") as u64
            } else {
                // null for 0-arity main (tag 0b111)
                0b111u64
            };

            let wrapper = runtime.get_function2(async_main_wrapper).unwrap();
            wrapper(tagged_main_fn, args_or_null)
        } else if arity == 0 {
            // No async wrapper, call main() directly with no arguments
            let f = runtime.get_function0(&fully_qualified_main).unwrap();
            f()
        } else if arity == 1 {
            // No async wrapper, call main(args) directly
            let stack_pointer = runtime.get_stack_base();
            let args_array = runtime
                .create_string_array(stack_pointer, &args.program_args)
                .expect("Failed to create args array");
            let f = runtime.get_function1(&fully_qualified_main).unwrap();
            f(args_array as u64)
        } else {
            eprintln!(
                "Error: main() must take 0 or 1 arguments, but yours takes {}.\n\
                Valid signatures:\n  \
                fn main() {{ ... }}\n  \
                fn main(args) {{ ... }}",
                arity
            );
            std::process::exit(1);
        };

        // Unregister main thread after Beagle code completes.
        // We're still registered but we're in Rust code now, not Beagle code.
        // If GC starts, it will wait for us to pause, but we can't pause from Rust.
        // Solution: register as c_calling so GC counts us and proceeds.
        {
            let (lock, condvar) = &*runtime.thread_state.clone();
            let mut state = lock.lock().unwrap();
            state.register_c_call(0); // No stack to scan
            condvar.notify_one();
        }

        // Now any GC will count us as c_calling and proceed.
        // Wait for any in-progress GC to finish, then unregister everything.
        loop {
            while runtime.is_paused() {
                std::thread::yield_now();
            }

            match runtime.gc_lock.try_lock() {
                Ok(_guard) => {
                    // While holding lock: unregister from c_calling, decrement count
                    {
                        let (lock, condvar) = &*runtime.thread_state.clone();
                        let mut state = lock.lock().unwrap();
                        state.unregister_c_call();
                        condvar.notify_one();
                    }
                    let new_count = runtime
                        .registered_thread_count
                        .fetch_sub(1, std::sync::atomic::Ordering::Release)
                        - 1;
                    gc::usdt_probes::fire_thread_unregister(new_count);
                    break;
                }
                Err(_) => {
                    std::thread::yield_now();
                }
            }
        }

        let _ = result; // Silence unused variable warning
    } else if args.debug {
        println!("No main function");
    }

    if args.show_times {
        println!("Time {:?}", time.elapsed());
    }

    // Wait for threads spawned by main() before inspecting snapshot output or running
    // test blocks. This avoids racing child-thread teardown against test-mode bookkeeping.
    runtime.wait_for_other_threads();

    if has_expect {
        let source = std::fs::read_to_string(program.clone())?;
        let expected = match get_expect(&source) {
            Some(e) => e,
            None => {
                return Err(format!(
                    "File '{}' was expected to have a // @beagle.core.snapshot marker but none was found",
                    program
                ).into());
            }
        };
        let expected = expected.trim();
        if expected.is_empty() {
            return Err(format!(
                "File '{}' has a // @beagle.core.snapshot marker but no expected output.\n\
                 Add expected output lines after the marker, prefixed with //:\n\n\
                 // @beagle.core.snapshot\n\
                 // expected output here\n\n\
                 If the marker appears in a comment unrelated to testing, \
                 reword it to avoid the exact string '// @beagle.core.snapshot'.",
                program
            )
            .into());
        }
        let printed = runtime.printer.get_output().join("").trim().to_string();
        if printed != expected {
            if args.update_snapshots {
                update_snapshot(&program, &printed)?;
            } else {
                return Err(format!(
                    "Snapshot mismatch:\nExpected:\n{}\nGot:\n{}",
                    expected, printed
                )
                .into());
            }
        }
    }

    // Run test blocks if in test mode
    if args.test {
        let test_names = runtime.get_test_function_names();
        if !test_names.is_empty() {
            let mut test_passed = 0;
            let mut test_failed = 0;
            for test_name in &test_names {
                // Extract the human-readable test name from __test_name__
                let short = test_name.rsplit('/').next().unwrap_or(test_name);
                let display_name = short
                    .strip_prefix("__test_")
                    .and_then(|s| s.strip_suffix("__"))
                    .unwrap_or(short)
                    .replace('_', " ");

                // Compile a wrapper that calls the test function with try/catch
                // Use the short function name since compile_string runs in the test's namespace
                let short_fn_name = test_name.rsplit('/').next().unwrap_or(test_name);
                let wrapper = format!(
                    "try {{ {}(); \"__PASS__\" }} catch (e) {{ to-string(e) }}",
                    short_fn_name
                );
                match runtime.compile_string(&wrapper) {
                    Ok(fn_ptr) => {
                        let result = runtime.call_via_trampoline(fn_ptr);
                        let result_str = runtime.get_string(0, result);
                        if result_str == "__PASS__" {
                            println!("  test pass: {}", display_name);
                            test_passed += 1;
                        } else {
                            println!("  test FAIL: {}", display_name);
                            println!("        {}", result_str);
                            test_failed += 1;
                        }
                    }
                    Err(e) => {
                        println!("  test FAIL: {} (compile error: {})", display_name, e);
                        test_failed += 1;
                    }
                }
            }
            if test_failed > 0 {
                return Err(format!(
                    "Test blocks: {} passed, {} failed",
                    test_passed, test_failed
                )
                .into());
            }
        }
    }

    runtime.wait_for_other_threads();

    // Clean up the main thread's per-thread continuation state after all threads have been joined.
    // Child threads clean up their own data in their exit path.
    {
        let ptd = crate::runtime::per_thread_data();
        ptd.invocation_return_points.clear();
        ptd.prompt_handlers.clear();
        ptd.return_from_shift_via_pop_prompt = false;
    }

    runtime.event_loops.shutdown_all();

    Ok(())
}

fn get_expect(source: &str) -> Option<String> {
    let start = source.find("// @beagle.core.snapshot")?;

    Some(
        source[start..]
            .lines()
            .skip(1)
            .take_while(|line| line.starts_with("//"))
            .map(|line| line.trim_start_matches("//").trim())
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn update_snapshot(file_path: &str, actual_output: &str) -> Result<(), Box<dyn Error>> {
    let source = std::fs::read_to_string(file_path)?;
    let marker = "// @beagle.core.snapshot";

    let marker_pos = source.find(marker).unwrap();
    // Find the start of the marker line
    let line_start = source[..marker_pos].rfind('\n').map_or(0, |p| p + 1);

    // Build new snapshot section
    let mut new_snapshot = String::new();
    new_snapshot.push_str(marker);
    new_snapshot.push('\n');
    for line in actual_output.lines() {
        if line.is_empty() {
            new_snapshot.push_str("//\n");
        } else {
            new_snapshot.push_str(&format!("// {}\n", line));
        }
    }

    // Replace everything from the marker line to end of file
    let mut new_source = source[..line_start].to_string();
    new_source.push_str(&new_snapshot);

    std::fs::write(file_path, new_source)?;
    Ok(())
}

#[test]
fn try_all_examples() -> Result<(), Box<dyn Error>> {
    let args = CommandLineArguments::default();
    run_all_tests(args)?;
    Ok(())
}

#[repr(transparent)]
pub struct SyncUnsafeCell<T: ?Sized> {
    value: UnsafeCell<T>,
}
impl<T> SyncUnsafeCell<T> {
    pub fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
        }
    }

    pub fn get(&self) -> &T {
        unsafe { &*self.value.get() }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn get_mut(&self) -> &mut T {
        unsafe { &mut *self.value.get() }
    }

    pub fn reset(&self, value: T) {
        unsafe {
            *self.value.get() = value;
        }
    }
}
unsafe impl<T: ?Sized + Sync> Sync for SyncUnsafeCell<T> {}

pub static RUNTIME: OnceLock<SyncUnsafeCell<Runtime>> = OnceLock::new();

pub fn get_runtime() -> &'static SyncUnsafeCell<Runtime> {
    RUNTIME.get().unwrap()
}
