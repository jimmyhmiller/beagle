#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::missing_safety_doc)]

// Backend-specific imports for runtime trampolines
// Use x86-64 backend if:
//   - Explicit feature flag is set, OR
//   - Building for x86_64 architecture (and no explicit ARM64 feature)
cfg_if::cfg_if! {
    if #[cfg(any(feature = "backend-x86-64", all(target_arch = "x86_64", not(feature = "backend-arm64"))))] {
        use crate::machine_code::x86_codegen::{RCX, RDX, RSI, RDI, R8, R9, R10, RSP, X86Asm};
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

#[cfg(not(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
)))]
mod arm;
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

#[derive(Debug, Encode, Decode, Clone, SerJson)]
pub struct Message {
    kind: String,
    data: Data,
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
        if num_args > 6 {
            // Push args in reverse order: argN-1, argN-2, ..., arg6
            for i in (6..num_args).rev() {
                // Our arg[i] is at [RBP + 16 + (i-5)*8]
                let offset = 16 + ((i as i32) - 5) * 8;
                // Load into R11 temporarily (we've saved fn_ptr already)
                // Actually we need a temp register that won't be clobbered
                // Use RDI since we'll overwrite it anyway in the shuffle
                lang.instructions.push(X86Asm::MovRM {
                    dest: RDI,
                    base: RBP,
                    offset,
                });
                lang.instructions.push(X86Asm::Push { reg: RDI });
            }
            // Reload fn_ptr into R11 since we clobbered RDI
            // Actually wait - we saved fn_ptr first, so R11 still has it
            // But we used RDI as temp, so we need to restore the shuffle logic
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

        // Clean up stack args we pushed (if any)
        if num_args > 6 {
            let stack_cleanup = ((num_args - 6) * 8) as i32;
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
        // Write result value: [R9] = R11
        lang.instructions.push(X86Asm::MovMR {
            base: R9,
            offset: 0,
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
        // Write result value: [R9] = R11
        lang.instructions.push(X86Asm::MovMR {
            base: R9,
            offset: 0,
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
}

#[derive(ClapParser, Debug, Clone)]
#[command(version, about, long_about = None)]
#[command(name = "beag")]
#[command(bin_name = "beag")]
pub struct CommandLineArguments {
    program: Option<String>,
    /// Arguments to pass to the Beagle program's main(args) function
    #[clap(trailing_var_arg = true, allow_hyphen_values = true)]
    program_args: Vec<String>,
    #[clap(long, default_value = "false")]
    show_times: bool,
    #[clap(long, default_value = "false")]
    show_gc_times: bool,
    #[clap(long, default_value = "false")]
    print_ast: bool,
    #[clap(long, default_value = "false")]
    no_gc: bool,
    #[clap(long, default_value = "false")]
    gc_always: bool,
    #[clap(long, default_value = "false")]
    all_tests: bool,
    #[clap(long, default_value = "false")]
    test: bool,
    #[clap(long, default_value = "false")]
    debug: bool,
    #[clap(long, default_value = "false")]
    verbose: bool,
    #[clap(long, default_value = "false")]
    no_std: bool,
    #[clap(long, default_value = "false")]
    print_parse: bool,
    #[clap(long, default_value = "false")]
    print_builtin_calls: bool,
    #[clap(long, default_value = "false")]
    repl: bool,
    /// Export all documentation to JSON and exit
    #[clap(long, default_value = "false")]
    export_docs: bool,
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
        "beagle.repl-session.bg",
        "beagle.repl.bg",
    ];
    let mut all_top_levels = vec![];

    for file_name in resource_files {
        let file_path = find_resource_file(file_name)?;
        let top_levels = runtime.compile(&file_path)?;
        all_top_levels.extend(top_levels);
    }

    for file_name in stdlib_files {
        let file_path = find_stdlib_file(file_name)?;
        let top_levels = runtime.compile(&file_path)?;
        all_top_levels.extend(top_levels);
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

fn main() -> Result<(), Box<dyn Error>> {
    let args = CommandLineArguments::parse();
    if args.export_docs {
        export_docs(args)
    } else if args.all_tests {
        run_all_tests(args)
    } else if args.repl {
        run_repl(args)
    } else {
        main_inner(args)
    }
}

fn run_all_tests(args: CommandLineArguments) -> Result<(), Box<dyn Error>> {
    for entry in std::fs::read_dir("resources")? {
        let entry = entry?;
        let mut path = entry.path();

        // Skip directories
        if path.is_dir() {
            continue;
        }

        if !path.exists() {
            path = path
                .parent()
                .unwrap()
                .join("resources")
                .join(path.file_name().unwrap());
        }
        let path = path.to_str().unwrap();
        if !path.ends_with(".bg") {
            continue;
        }

        let source: String = std::fs::read_to_string(path)?;

        if !source.contains("// Expect") {
            continue;
        }

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

        println!("Running test: {}", path);
        // Check for gc-always annotation - test requests gc on every allocation
        let gc_always = args.gc_always || source.contains("// gc-always");
        let args = CommandLineArguments {
            program: Some(path.to_string()),
            program_args: vec![], // Tests don't receive extra args
            show_times: args.show_times,
            show_gc_times: args.show_gc_times,
            print_ast: args.print_ast,
            no_gc: args.no_gc,
            gc_always,
            all_tests: false,
            test: true,
            debug: args.debug,
            verbose: args.verbose,
            no_std: args.no_std,
            print_parse: args.print_parse,
            print_builtin_calls: args.print_builtin_calls,
            repl: false,
            export_docs: false,
        };
        main_inner(args)?;
        RUNTIME.get().unwrap().get_mut().reset();
    }
    Ok(())
}

fn run_repl(args: CommandLineArguments) -> Result<(), Box<dyn Error>> {
    use std::io::{self, Write};

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

    runtime.install_builtins()?;

    // Generate continuation trampolines using the code generator (x86-64 only)
    // This replaces the Rust inline assembly version which gets broken by LLVM optimizer
    #[cfg(any(
        feature = "backend-x86-64",
        all(target_arch = "x86_64", not(feature = "backend-arm64"))
    ))]
    compile_continuation_trampolines(runtime);

    let mut top_levels = vec![];
    if !args.no_std {
        top_levels = load_default_files(runtime)?;
    }

    for top_level in top_levels {
        if let Some(f) = runtime.get_function0(&top_level) {
            f();
        }
    }

    // Set REPL to use "user" namespace for user code
    let user_namespace_id = runtime
        .get_namespace_id("user")
        .unwrap_or_else(|| runtime.reserve_namespace("user".to_string()));
    runtime.set_current_namespace(user_namespace_id);

    println!("Beagle REPL - Enter expressions to evaluate (Ctrl+C to exit)");

    loop {
        print!("beagle> ");
        io::stdout().flush()?;

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let input = input.trim();
                if input.is_empty() {
                    continue;
                }

                // Escape the input string for eval()
                let escaped_input = input.replace("\\", "\\\\").replace("\"", "\\\"");

                // Wrap user input in eval() with try/catch to handle all exceptions consistently
                // This ensures parse/compile errors throw SystemError.CompileError like explicit eval() calls
                let wrapped_input = format!(
                    "try {{ eval(\"{}\") }} catch (__repl_error__) {{ println(\"Uncaught exception:\"); println(__repl_error__); null }}",
                    escaped_input
                );

                match runtime.compile_string(&wrapped_input) {
                    Ok(function_pointer) => {
                        if function_pointer == 0 {
                            continue;
                        }
                        let f: fn() -> usize = unsafe { std::mem::transmute(function_pointer) };
                        let result = f();
                        runtime.println(result).unwrap();
                    }
                    Err(e) => {
                        // This should only happen if the wrapper code itself has syntax errors
                        eprintln!("Internal REPL error: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
    }

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
    let has_expect = args.test && source.contains("// Expect");

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

    runtime.install_builtins()?;

    // Generate continuation trampolines using the code generator (x86-64 only)
    // This replaces the Rust inline assembly version which gets broken by LLVM optimizer
    #[cfg(any(
        feature = "backend-x86-64",
        all(target_arch = "x86_64", not(feature = "backend-arm64"))
    ))]
    compile_continuation_trampolines(runtime);

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
                "We are supposed to have top level, but didn't find the function {}",
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
            panic!("main() must take 0 or 1 arguments, got {}", arity);
        };

        // Force cleanup of all continuation state before main returns
        // This prevents crashes during Runtime drop
        runtime.invocation_return_points.clear();
        runtime.prompt_handlers.clear();
        runtime.return_from_shift_via_pop_prompt.clear();

        // Unregister main thread after Beagle code completes.
        // We're still registered but we're in Rust code now, not Beagle code.
        // If GC starts, it will wait for us to pause, but we can't pause from Rust.
        // Solution: register as c_calling so GC counts us and proceeds.
        {
            let (lock, condvar) = &*runtime.thread_state.clone();
            let mut state = lock.lock().unwrap();
            state.register_c_call((0, 0, 0)); // No stack to scan
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

    if has_expect {
        let source = std::fs::read_to_string(program)?;
        let expected = get_expect(&source);
        let expected = expected.trim();
        let printed = runtime.printer.get_output().join("").trim().to_string();
        if printed != expected {
            println!("Expected: \n{}\n", expected);
            println!("Got: \n{}\n", printed);
            panic!("Test failed");
        }
        println!("Test passed");
    }

    loop {
        // take the list of threads so we are not holding a borrow on the compiler
        // use mem::replace to swap out the threads with an empty vec
        let threads = std::mem::take(&mut runtime.memory.join_handles);
        if threads.is_empty() {
            break;
        }
        for thread in threads {
            thread.join().unwrap();
        }
    }

    Ok(())
}

fn get_expect(source: &str) -> String {
    let start = source.find("// Expect").unwrap();
    // get each line as long as they start with //

    source[start..]
        .lines()
        .skip(1)
        .take_while(|line| line.starts_with("//"))
        .map(|line| line.trim_start_matches("//").trim())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn try_all_examples() -> Result<(), Box<dyn Error>> {
    let args = CommandLineArguments {
        program: None,
        program_args: vec![],
        show_times: false,
        show_gc_times: false,
        print_ast: false,
        no_gc: false,
        gc_always: false,
        all_tests: true,
        test: false,
        debug: false,
        verbose: false,
        no_std: false,
        print_parse: false,
        print_builtin_calls: false,
        repl: false,
        export_docs: false,
    };
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
