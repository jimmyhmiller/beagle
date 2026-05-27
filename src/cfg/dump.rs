//! Text + Graphviz pretty-printers for `CfgFunction`. Used by the
//! phase-staged dump harness in `build_cfg`: set
//! `BEAGLE_SSA_DUMP_DIR=<path>` and `BEAGLE_SSA_DUMP_FN=<function-name>`
//! to write per-phase `.cfg` text dumps (and a `.dot` file for the
//! final state) to disk, then diff between phases.
//!
//! The text format is intentionally compact and self-describing:
//!
//! ```text
//! function "name" entry=block0 blocks=4 vregs=12 slots=2
//!
//! block0(v0:gp, v1:gp):  preds=[]
//!   v2 = SlotLoad slot(0)
//!   v3 = AddInt v0 v1
//!   branch eq v2 v3 -> block1 [] else block2 []
//!
//! block1:  preds=[block0]
//!   v4 = ConstTaggedInt 42
//!   jump block3 [v4]
//! ```

#![allow(dead_code)]

use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::cfg::{
    BlockId, CallTarget, CfgFunction, ClobberSet, InlineBranchOp, Op, RegClass, TagSource,
    Terminator, VReg,
};

// ============================================================
// Text format
// ============================================================

pub fn dump_text(f: &CfgFunction) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "function {:?} entry=block{} blocks={} vregs={} slots={}",
        f.debug_name,
        f.entry.0,
        f.blocks.len(),
        f.vreg_classes.len(),
        f.num_slots
    )
    .unwrap();
    for (idx, block) in f.blocks.iter().enumerate() {
        writeln!(out).unwrap();
        write_block_header(&mut out, BlockId(idx as u32), block);
        for op in &block.body {
            write!(out, "  ").unwrap();
            write_op(&mut out, op);
            writeln!(out).unwrap();
        }
        write!(out, "  ").unwrap();
        write_terminator(&mut out, &block.terminator);
        writeln!(out).unwrap();
    }
    out
}

fn write_block_header(out: &mut String, bid: BlockId, block: &crate::cfg::Block) {
    write!(out, "block{}", bid.0).unwrap();
    if !block.params.is_empty() {
        write!(out, "(").unwrap();
        for (i, p) in block.params.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "{}", fmt_vreg(*p)).unwrap();
        }
        write!(out, ")").unwrap();
    }
    let preds: Vec<String> = block
        .predecessors
        .iter()
        .map(|p| format!("block{}", p.0))
        .collect();
    writeln!(out, ":  preds=[{}]", preds.join(", ")).unwrap();
}

fn write_op(out: &mut String, op: &Op) {
    match op {
        Op::SlotLoad { dst, slot } => {
            write!(out, "{} = SlotLoad slot({})", fmt_vreg(*dst), slot.0).unwrap()
        }
        Op::SlotStore { slot, src } => {
            write!(out, "SlotStore slot({}) <- {}", slot.0, fmt_vreg(*src)).unwrap()
        }
        Op::Move { dst, src } => {
            write!(out, "{} = Move {}", fmt_vreg(*dst), fmt_vreg(*src)).unwrap()
        }
        Op::ConstTaggedInt { dst, value } => {
            write!(out, "{} = ConstTaggedInt {}", fmt_vreg(*dst), value).unwrap()
        }
        Op::ConstStringPtr { dst, ptr } => {
            write!(out, "{} = ConstStringPtr 0x{:x}", fmt_vreg(*dst), ptr).unwrap()
        }
        Op::ConstKeywordPtr { dst, ptr } => {
            write!(out, "{} = ConstKeywordPtr 0x{:x}", fmt_vreg(*dst), ptr).unwrap()
        }
        Op::ConstFunctionId { dst, function_id } => {
            write!(out, "{} = ConstFunctionId {}", fmt_vreg(*dst), function_id).unwrap()
        }
        Op::ConstPointer { dst, ptr } => {
            write!(out, "{} = ConstPointer 0x{:x}", fmt_vreg(*dst), ptr).unwrap()
        }
        Op::ConstRawValue { dst, value } => {
            write!(out, "{} = ConstRawValue 0x{:x}", fmt_vreg(*dst), value).unwrap()
        }
        Op::ConstTrue { dst } => write!(out, "{} = ConstTrue", fmt_vreg(*dst)).unwrap(),
        Op::ConstFalse { dst } => write!(out, "{} = ConstFalse", fmt_vreg(*dst)).unwrap(),
        Op::ConstNull { dst } => write!(out, "{} = ConstNull", fmt_vreg(*dst)).unwrap(),
        Op::ConstLabelAddress { dst, target } => write!(
            out,
            "{} = ConstLabelAddress block{}",
            fmt_vreg(*dst),
            target.0
        )
        .unwrap(),

        Op::AddInt { dst, lhs, rhs } => write!(
            out,
            "{} = AddInt {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        )
        .unwrap(),
        Op::Compare {
            dst,
            lhs,
            rhs,
            cond,
        } => write!(
            out,
            "{} = Compare {:?} {} {}",
            fmt_vreg(*dst),
            cond,
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        )
        .unwrap(),
        Op::CompareFloat {
            dst,
            lhs,
            rhs,
            cond,
        } => write!(
            out,
            "{} = CompareFloat {:?} {} {}",
            fmt_vreg(*dst),
            cond,
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        )
        .unwrap(),

        Op::AddFloat { dst, lhs, rhs } => write!(
            out,
            "{} = AddFloat {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        )
        .unwrap(),
        Op::SubFloat { dst, lhs, rhs } => write!(
            out,
            "{} = SubFloat {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        )
        .unwrap(),
        Op::MulFloat { dst, lhs, rhs } => write!(
            out,
            "{} = MulFloat {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        )
        .unwrap(),
        Op::DivFloat { dst, lhs, rhs } => write!(
            out,
            "{} = DivFloat {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        )
        .unwrap(),

        Op::IntToFloat { dst, src } => {
            write!(out, "{} = IntToFloat {}", fmt_vreg(*dst), fmt_vreg(*src)).unwrap()
        }
        Op::FRoundToZero { dst, src } => {
            write!(out, "{} = FRoundToZero {}", fmt_vreg(*dst), fmt_vreg(*src)).unwrap()
        }
        Op::FmovGpToFp { dst, src } => {
            write!(out, "{} = FmovGpToFp {}", fmt_vreg(*dst), fmt_vreg(*src)).unwrap()
        }
        Op::FmovFpToGp { dst, src } => {
            write!(out, "{} = FmovFpToGp {}", fmt_vreg(*dst), fmt_vreg(*src)).unwrap()
        }

        Op::Tag {
            dst,
            src,
            tag_source,
        } => write!(
            out,
            "{} = Tag {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*src),
            match tag_source {
                TagSource::Register(t) => format!("reg({})", fmt_vreg(*t)),
                TagSource::Bits(b) => format!("bits(0x{:x})", b),
            }
        )
        .unwrap(),
        Op::Untag { dst, src } => {
            write!(out, "{} = Untag {}", fmt_vreg(*dst), fmt_vreg(*src)).unwrap()
        }
        Op::GetTag { dst, src } => {
            write!(out, "{} = GetTag {}", fmt_vreg(*dst), fmt_vreg(*src)).unwrap()
        }

        Op::And { dst, lhs, rhs } => write!(
            out,
            "{} = And {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        )
        .unwrap(),
        Op::Or { dst, lhs, rhs } => write!(
            out,
            "{} = Or {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        )
        .unwrap(),
        Op::Xor { dst, lhs, rhs } => write!(
            out,
            "{} = Xor {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        )
        .unwrap(),
        Op::AndImm { dst, src, imm } => write!(
            out,
            "{} = AndImm {} 0x{:x}",
            fmt_vreg(*dst),
            fmt_vreg(*src),
            imm
        )
        .unwrap(),
        Op::ShiftRightImmRaw { dst, src, imm } => write!(
            out,
            "{} = ShiftRightImmRaw {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*src),
            imm
        )
        .unwrap(),

        Op::HeapLoad { dst, base, offset } => write!(
            out,
            "{} = HeapLoad {} +{}",
            fmt_vreg(*dst),
            fmt_vreg(*base),
            offset
        )
        .unwrap(),
        Op::HeapLoadReg { dst, base, offset } => write!(
            out,
            "{} = HeapLoadReg {} +{}",
            fmt_vreg(*dst),
            fmt_vreg(*base),
            fmt_vreg(*offset)
        )
        .unwrap(),
        Op::HeapLoadByteReg { dst, base, offset } => write!(
            out,
            "{} = HeapLoadByteReg {} +{}",
            fmt_vreg(*dst),
            fmt_vreg(*base),
            fmt_vreg(*offset)
        )
        .unwrap(),
        Op::HeapStore { addr, src } => {
            write!(out, "HeapStore [{}] <- {}", fmt_vreg(*addr), fmt_vreg(*src)).unwrap()
        }
        Op::HeapStoreOffset { base, src, offset } => write!(
            out,
            "HeapStoreOffset [{} +{}] <- {}",
            fmt_vreg(*base),
            offset,
            fmt_vreg(*src)
        )
        .unwrap(),
        Op::HeapStoreOffsetReg { base, src, offset } => write!(
            out,
            "HeapStoreOffsetReg [{} +{}] <- {}",
            fmt_vreg(*base),
            fmt_vreg(*offset),
            fmt_vreg(*src)
        )
        .unwrap(),
        Op::HeapStoreByteOffsetReg { base, src, offset } => write!(
            out,
            "HeapStoreByteOffsetReg [{} +{}] <- {}",
            fmt_vreg(*base),
            fmt_vreg(*offset),
            fmt_vreg(*src)
        )
        .unwrap(),
        Op::HeapStoreByteOffsetMasked {
            ptr,
            val,
            temp1,
            temp2,
            offset,
            byte_offset,
            mask,
        } => write!(
            out,
            "HeapStoreByteOffsetMasked [{} +{}.{}] <- {} (mask=0x{:x}, temps={}, {})",
            fmt_vreg(*ptr),
            offset,
            byte_offset,
            fmt_vreg(*val),
            mask,
            fmt_vreg(*temp1),
            fmt_vreg(*temp2)
        )
        .unwrap(),

        Op::AtomicLoad { dst, src } => {
            write!(out, "{} = AtomicLoad [{}]", fmt_vreg(*dst), fmt_vreg(*src)).unwrap()
        }
        Op::AtomicStore { addr, src } => write!(
            out,
            "AtomicStore [{}] <- {}",
            fmt_vreg(*addr),
            fmt_vreg(*src)
        )
        .unwrap(),
        Op::CompareAndSwap {
            addr,
            expected,
            new,
        } => write!(
            out,
            "CompareAndSwap [{}] expect={} new={}",
            fmt_vreg(*addr),
            fmt_vreg(*expected),
            fmt_vreg(*new)
        )
        .unwrap(),
        Op::StoreFloatConstant {
            dest,
            temp,
            value_text,
        } => write!(
            out,
            "StoreFloatConstant [{}] = {:?} (temp={})",
            fmt_vreg(*dest),
            value_text,
            fmt_vreg(*temp)
        )
        .unwrap(),

        Op::PushStack { src } => write!(out, "PushStack {}", fmt_vreg(*src)).unwrap(),
        Op::PopStack { dst } => write!(out, "{} = PopStack", fmt_vreg(*dst)).unwrap(),
        Op::GetStackPointer { dst, offset } => write!(
            out,
            "{} = GetStackPointer +{}",
            fmt_vreg(*dst),
            fmt_vreg(*offset)
        )
        .unwrap(),
        Op::GetStackPointerImm { dst, offset } => {
            write!(out, "{} = GetStackPointerImm +{}", fmt_vreg(*dst), offset).unwrap()
        }
        Op::GetFramePointer { dst } => write!(out, "{} = GetFramePointer", fmt_vreg(*dst)).unwrap(),
        Op::CurrentStackPosition { dst } => {
            write!(out, "{} = CurrentStackPosition", fmt_vreg(*dst)).unwrap()
        }

        Op::ReadArgCount { dst } => write!(out, "{} = ReadArgCount", fmt_vreg(*dst)).unwrap(),

        Op::Breakpoint => write!(out, "Breakpoint").unwrap(),
        Op::ExtendLifetime { src } => write!(out, "ExtendLifetime {}", fmt_vreg(*src)).unwrap(),
        Op::RecordGcSafepoint => write!(out, "RecordGcSafepoint").unwrap(),
        Op::FeedbackOr { slot_addr, bits } => {
            write!(out, "FeedbackOr [0x{:x}] |= 0x{:x}", slot_addr, bits).unwrap()
        }
        Op::TierUpCheck {
            counter_addr,
            name_c_str_ptr,
            trampoline_fn_ptr,
        } => write!(
            out,
            "TierUpCheck counter=0x{:x} name=0x{:x} trampoline=0x{:x}",
            counter_addr, name_c_str_ptr, trampoline_fn_ptr
        )
        .unwrap(),

        Op::Call {
            dst,
            target,
            args,
            is_builtin,
            clobbers,
        } => write!(
            out,
            "{} = Call{} {} {} clobbers={}",
            fmt_vreg(*dst),
            if *is_builtin { "Builtin" } else { "" },
            fmt_call_target(target),
            fmt_args(args),
            fmt_clobbers(clobbers)
        )
        .unwrap(),
        Op::Recurse {
            dst,
            args,
            clobbers,
        } => write!(
            out,
            "{} = Recurse {} clobbers={}",
            fmt_vreg(*dst),
            fmt_args(args),
            fmt_clobbers(clobbers)
        )
        .unwrap(),

        Op::PushExceptionHandler {
            handler,
            result_slot,
            ..
        } => write!(
            out,
            "PushExceptionHandler handler=block{} result_slot={}",
            handler.0, result_slot.0
        )
        .unwrap(),
        Op::PushResumableExceptionHandler {
            dst,
            catch_block,
            exception_slot,
            resume_slot,
            ..
        } => write!(
            out,
            "{} = PushResumableExceptionHandler catch=block{} exc_slot={} resume_slot={}",
            fmt_vreg(*dst),
            catch_block.0,
            exception_slot.0,
            resume_slot.0
        )
        .unwrap(),
        Op::PopExceptionHandler { .. } => write!(out, "PopExceptionHandler").unwrap(),
        Op::PopExceptionHandlerById { handler_id, .. } => {
            write!(out, "PopExceptionHandlerById {}", fmt_vreg(*handler_id)).unwrap()
        }
        Op::PushPromptHandler {
            handler,
            result_slot,
            ..
        } => write!(
            out,
            "PushPromptHandler handler=block{} result_slot={}",
            handler.0, result_slot.0
        )
        .unwrap(),
        Op::PopPromptHandler { result, .. } => {
            write!(out, "PopPromptHandler result={}", fmt_vreg(*result)).unwrap()
        }
        Op::PushPromptTag {
            tag,
            abort_block,
            result_slot,
            ..
        } => write!(
            out,
            "PushPromptTag tag={} abort=block{} result_slot={}",
            fmt_vreg(*tag),
            abort_block.0,
            result_slot.0
        )
        .unwrap(),
        Op::CaptureContinuation {
            dst,
            resume_block,
            result_slot,
            ..
        } => write!(
            out,
            "{} = CaptureContinuation resume=block{} result_slot={}",
            fmt_vreg(*dst),
            resume_block.0,
            result_slot.0
        )
        .unwrap(),
        Op::CaptureContinuationTagged {
            dst,
            resume_block,
            result_slot,
            tag,
            ..
        } => write!(
            out,
            "{} = CaptureContinuationTagged resume=block{} result_slot={} tag={}",
            fmt_vreg(*dst),
            resume_block.0,
            result_slot.0,
            fmt_vreg(*tag)
        )
        .unwrap(),
        Op::PerformEffect {
            handler,
            enum_type,
            op_value,
            resume_block,
            result_slot,
            ..
        } => write!(
            out,
            "PerformEffect handler={} enum_type={} op={} resume=block{} result_slot={}",
            fmt_vreg(*handler),
            fmt_vreg(*enum_type),
            fmt_vreg(*op_value),
            resume_block.0,
            result_slot.0
        )
        .unwrap(),
        Op::ReturnFromShift {
            value, cont_ptr, ..
        } => write!(
            out,
            "ReturnFromShift value={} cont={}",
            fmt_vreg(*value),
            fmt_vreg(*cont_ptr)
        )
        .unwrap(),
    }
}

fn write_terminator(out: &mut String, t: &Terminator) {
    match t {
        Terminator::Jump { target, args } => {
            write!(out, "jump block{} {}", target.0, fmt_args(args)).unwrap();
        }
        Terminator::Branch {
            cond,
            lhs,
            rhs,
            t_target,
            t_args,
            f_target,
            f_args,
        } => write!(
            out,
            "branch {:?} {} {} -> block{} {} else block{} {}",
            cond,
            fmt_vreg(*lhs),
            fmt_vreg(*rhs),
            t_target.0,
            fmt_args(t_args),
            f_target.0,
            fmt_args(f_args)
        )
        .unwrap(),
        Terminator::InlineBranch {
            op,
            fall_through,
            fall_args,
            bail,
            bail_args,
        } => write!(
            out,
            "inline-branch {} -> block{} {} bail block{} {}",
            fmt_inline_branch_op(op),
            fall_through.0,
            fmt_args(fall_args),
            bail.0,
            fmt_args(bail_args)
        )
        .unwrap(),
        Terminator::Ret { value } => write!(out, "ret {}", fmt_vreg(*value)).unwrap(),
        Terminator::Throw {
            value,
            resume,
            resume_args,
            resume_local,
            ..
        } => write!(
            out,
            "throw {} -> resume block{} {} resume_local={}",
            fmt_vreg(*value),
            resume.0,
            fmt_args(resume_args),
            resume_local.0
        )
        .unwrap(),
        Terminator::Unreachable => write!(out, "unreachable").unwrap(),
    }
}

fn fmt_inline_branch_op(op: &InlineBranchOp) -> String {
    match op {
        InlineBranchOp::SubChecked { dst, lhs, rhs } => format!(
            "{} = sub-checked {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        ),
        InlineBranchOp::MulChecked { dst, lhs, rhs } => format!(
            "{} = mul-checked {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        ),
        InlineBranchOp::DivChecked { dst, lhs, rhs } => format!(
            "{} = div-checked {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        ),
        InlineBranchOp::ModuloChecked { dst, lhs, rhs } => format!(
            "{} = mod-checked {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        ),
        InlineBranchOp::ShiftLeftChecked { dst, lhs, rhs } => format!(
            "{} = shl-checked {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        ),
        InlineBranchOp::ShiftRightChecked { dst, lhs, rhs } => format!(
            "{} = shr-checked {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        ),
        InlineBranchOp::ShiftRightZeroChecked { dst, lhs, rhs } => format!(
            "{} = shrz-checked {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*lhs),
            fmt_vreg(*rhs)
        ),
        InlineBranchOp::ShiftRightImmChecked { dst, src, imm } => format!(
            "{} = shri-checked {} {}",
            fmt_vreg(*dst),
            fmt_vreg(*src),
            imm
        ),
        InlineBranchOp::GuardInt { dst, src } => {
            format!("{} = guard-int {}", fmt_vreg(*dst), fmt_vreg(*src))
        }
        InlineBranchOp::GuardFloat { dst, src } => {
            format!("{} = guard-float {}", fmt_vreg(*dst), fmt_vreg(*src))
        }
        InlineBranchOp::InlineBumpAllocate {
            dst,
            size_bytes,
            header,
        } => format!(
            "{} = inline-bump-allocate {}B header=0x{:x}",
            fmt_vreg(*dst),
            size_bytes,
            header
        ),
    }
}

fn fmt_vreg(v: VReg) -> String {
    format!(
        "v{}:{}",
        v.index,
        match v.class {
            RegClass::Gp => "gp",
            RegClass::Fp => "fp",
        }
    )
}

fn fmt_args(args: &[VReg]) -> String {
    if args.is_empty() {
        "[]".into()
    } else {
        format!(
            "[{}]",
            args.iter()
                .map(|v| fmt_vreg(*v))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn fmt_call_target(t: &CallTarget) -> String {
    match t {
        CallTarget::Register(v) => format!("reg({})", fmt_vreg(*v)),
        CallTarget::FunctionId(id) => format!("fn-id({})", id),
        CallTarget::Pointer(p) => format!("ptr(0x{:x})", p),
        CallTarget::Raw(v) => format!("raw(0x{:x})", v),
    }
}

fn fmt_clobbers(c: &ClobberSet) -> String {
    match c {
        ClobberSet::AllCallerSaved => "all-caller-saved".into(),
    }
}

// ============================================================
// Graphviz DOT
// ============================================================

pub fn dump_dot(f: &CfgFunction) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "digraph {:?} {{",
        f.debug_name.as_deref().unwrap_or("anon")
    )
    .unwrap();
    writeln!(out, "  node [shape=box fontname=Courier fontsize=10];").unwrap();
    for (idx, block) in f.blocks.iter().enumerate() {
        let mut label = String::new();
        let mut tmp_label = String::new();
        write_block_header(&mut tmp_label, BlockId(idx as u32), block);
        for op in &block.body {
            tmp_label.push_str("  ");
            write_op(&mut tmp_label, op);
            tmp_label.push('\n');
        }
        tmp_label.push_str("  ");
        write_terminator(&mut tmp_label, &block.terminator);
        // Escape for DOT.
        for c in tmp_label.chars() {
            match c {
                '"' => label.push_str("\\\""),
                '\n' => label.push_str("\\l"),
                '\\' => label.push_str("\\\\"),
                _ => label.push(c),
            }
        }
        writeln!(out, "  block{} [label=\"{}\"];", idx, label).unwrap();
        for succ in block.terminator.successors() {
            writeln!(out, "  block{} -> block{};", idx, succ.0).unwrap();
        }
    }
    writeln!(out, "}}").unwrap();
    out
}

// ============================================================
// Phase-staged dump harness for build_cfg
// ============================================================

/// Number of dumps written per function across the whole compile —
/// used to prefix filenames so the dump directory lists in phase
/// order even when one function is compiled multiple times (tier-up
/// specialization makes this common).
static DUMP_COUNTER: Mutex<usize> = Mutex::new(0);

/// If `BEAGLE_SSA_DUMP_FN=<fn_name>` matches `f.debug_name`, write the
/// CFG to `BEAGLE_SSA_DUMP_DIR/<NNN>-<sanitized-fn>-<phase>.cfg` (and
/// `.dot` for the final-phase dump). Returns silently if either env
/// var is unset or the function doesn't match.
///
/// Filename prefix is a monotonic counter so the directory listing
/// stays in chronological compile order — useful when the same
/// function is recompiled by tier-up specialization.
pub fn maybe_dump_phase(phase: &str, f: &CfgFunction, also_dot: bool) {
    let Ok(target) = std::env::var("BEAGLE_SSA_DUMP_FN") else {
        return;
    };
    let Some(name) = f.debug_name.as_deref() else {
        return;
    };
    if name != target {
        return;
    }
    let Ok(dir) = std::env::var("BEAGLE_SSA_DUMP_DIR") else {
        // No directory configured — print to stderr instead.
        eprintln!(
            "\n========== {} {} ==========\n{}",
            name,
            phase,
            dump_text(f)
        );
        return;
    };

    let counter = {
        let mut c = DUMP_COUNTER.lock().unwrap();
        let v = *c;
        *c += 1;
        v
    };

    let safe_name: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let safe_phase: String = phase
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();

    let _ = fs::create_dir_all(&dir);
    let base: PathBuf =
        Path::new(&dir).join(format!("{:04}-{}-{}", counter, safe_name, safe_phase));
    let _ = write_file(&base.with_extension("cfg"), &dump_text(f));
    if also_dot {
        let _ = write_file(&base.with_extension("dot"), &dump_dot(f));
    }
}

fn write_file(path: &Path, contents: &str) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(contents.as_bytes())?;
    Ok(())
}

/// If `BEAGLE_SSA_VERIFY_STAGES=1` is set, run the verifier and log
/// any failure to stderr with the phase name. Doesn't abort
/// compilation — the goal is to pinpoint which pipeline phase first
/// introduces a verification error.
pub fn maybe_verify_stage(phase: &str, f: &CfgFunction) {
    if std::env::var("BEAGLE_SSA_VERIFY_STAGES").is_err() {
        return;
    }
    if let Err(e) = crate::cfg::verify::verify(f) {
        let name = f.debug_name.as_deref().unwrap_or("<anon>");
        eprintln!("[stage-verify] {} after {}: {:?}", name, phase, e);
    }
}
