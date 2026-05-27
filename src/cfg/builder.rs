//! CFG construction from the legacy linear IR.
//!
//! Phase 1a ships only the **leader computation** — the algorithm that
//! decides where blocks begin in a flat `Vec<Instruction>`. Phase 1b
//! adds full construction (block bodies, terminator translation, VReg
//! threading). Phase 1c splits critical edges. Phase 1d rewrites tail
//! self-calls per **I8**.
//!
//! The single most important rule baked in here is the **orphan-block
//! fix** from `docs/SSA_ARCHITECTURE.md` (Phase 1 description, Invariant
//! I1): when computing leaders, the position immediately after a
//! terminator is NOT a leader if it lands on an `Instruction::Label(_)`.
//! The label's own jump target is already a leader via the label-targets
//! rule, so adding the position-of-the-label as a separate leader would
//! produce a 1-instruction orphan block with no predecessors. The prior
//! `ssa` branch saw 135 such orphans in `nbody/advance` alone; fixing
//! this rule in construction drops 3060 spurious merge phis to 117.
//!
//! Sub / Mul / Div / Modulo / Shift* / GuardInt / GuardFloat / etc. have
//! a bail-out label parameter. They are NOT terminators — they continue
//! straight-line after a successful fast path — but their bail-out label
//! IS a leader (the slow path begins a new block). Phase 1b will model
//! these as "inline-branch" ops inside a block body whose successor list
//! includes both the fall-through and the bail-out label.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::cfg::{BlockId, CfgFunction, Op, RegClass, SlotId, Terminator, VReg};
use crate::common::Label;
use crate::ir::{Instruction, Ir, Value, VirtualRegister};

/// Map from `Label::index` to the position of that label in the
/// instruction list. Built by scanning for `Instruction::Label`.
pub fn label_positions(instructions: &[Instruction]) -> HashMap<usize, usize> {
    let mut out = HashMap::new();
    for (pos, inst) in instructions.iter().enumerate() {
        if let Instruction::Label(label) = inst {
            out.insert(label.index, pos);
        }
    }
    out
}

/// Compute the set of leader positions for a CFG, sorted and deduped.
///
/// Rules:
/// 1. Position 0 is always a leader (if the list is non-empty).
/// 2. Every label position referenced by any instruction's label argument
///    is a leader.
/// 3. The position immediately after a true terminator (Jump, JumpIf, Ret,
///    Throw, TailRecurse) is a leader.
/// 4. **Exception (orphan-block fix):** if the position-after-a-terminator
///    lands on an `Instruction::Label(_)`, rule 3 is suppressed. The label
///    target itself is already a leader by rule 2; making the label
///    position a leader twice would produce an orphan block.
pub fn compute_leaders(instructions: &[Instruction]) -> Vec<usize> {
    if instructions.is_empty() {
        return Vec::new();
    }
    let labels = label_positions(instructions);
    let mut leaders: Vec<usize> = Vec::with_capacity(instructions.len() / 4 + 1);
    leaders.push(0);
    for (pos, inst) in instructions.iter().enumerate() {
        // Rule 2: targets of any label-naming instruction.
        for target_label in label_targets(inst) {
            if let Some(&target_pos) = labels.get(&target_label.index) {
                leaders.push(target_pos);
            }
        }
        // Rule 3 + 4: post-terminator position.
        if is_terminator(inst) {
            let next = pos + 1;
            if next < instructions.len() && !matches!(instructions[next], Instruction::Label(_)) {
                leaders.push(next);
            }
        }
    }
    leaders.sort_unstable();
    leaders.dedup();
    leaders
}

/// True for instructions that end a basic block.
///
/// True for: `Jump`, `JumpIf`, `Ret`, `Throw`, `TailRecurse`. False for
/// everything else — including `Recurse` (which returns and continues),
/// the math bail-out ops (which transfer to the slow path only on
/// failure), and the exception-handler-push ops (which register a handler
/// without transferring control).
pub fn is_terminator(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Jump(_)
            | Instruction::JumpIf(_, _, _, _)
            | Instruction::Ret(_)
            | Instruction::Throw(_, _, _, _)
            | Instruction::TailRecurse(_, _)
    )
}

/// Every label this instruction names as a control-flow target.
///
/// Includes:
/// - True terminators' targets (`Jump`, `JumpIf`, `Throw`, `TailRecurse`
///   does not name a label).
/// - Bail-out labels on math fast-paths (`Sub`, `Mul`, `Div`, `Modulo`,
///   `Shift*`, `GuardInt`, `GuardFloat`).
/// - Exception / prompt / continuation handler labels — these are
///   registered with the runtime and can be jumped to later, so they
///   must be leaders even though the registering op doesn't transfer
///   control itself.
/// - `LoadLabelAddress` — the address of a label is taken and stored;
///   anything could jump there.
///
/// Returns owned `Label` values so callers can index into the
/// label-positions map.
pub fn label_targets(inst: &Instruction) -> Vec<Label> {
    match inst {
        Instruction::Jump(l) => vec![*l],
        Instruction::JumpIf(l, _, _, _) => vec![*l],
        Instruction::Throw(_, l, _, _) => vec![*l],

        Instruction::Sub(_, _, _, l) => vec![*l],
        Instruction::Mul(_, _, _, l) => vec![*l],
        Instruction::Div(_, _, _, l) => vec![*l],
        Instruction::Modulo(_, _, _, l) => vec![*l],
        Instruction::ShiftLeft(_, _, _, l) => vec![*l],
        Instruction::ShiftRight(_, _, _, l) => vec![*l],
        Instruction::ShiftRightZero(_, _, _, l) => vec![*l],
        Instruction::ShiftRightImm(_, _, _, l) => vec![*l],

        Instruction::GuardInt(_, _, l) => vec![*l],
        Instruction::GuardFloat(_, _, l) => vec![*l],

        Instruction::PushExceptionHandler(l, _, _) => vec![*l],
        Instruction::PushResumableExceptionHandler(_, l, _, _, _) => vec![*l],
        Instruction::PushPromptHandler(l, _, _) => vec![*l],
        Instruction::PushPromptTag(_, l, _, _) => vec![*l],
        Instruction::CaptureContinuation(_, l, _, _) => vec![*l],
        Instruction::CaptureContinuationTagged(_, l, _, _, _) => vec![*l],
        Instruction::PerformEffect(_, _, _, l, _, _) => vec![*l],
        Instruction::InlineBumpAllocate(_, _, _, l) => vec![*l],
        Instruction::LoadLabelAddress(_, l) => vec![*l],

        _ => Vec::new(),
    }
}

// =========================================================================
// Phase 1b: full CFG construction from a legacy Ir
// =========================================================================

/// Failure modes for CFG construction. Distinguishes "we haven't taught
/// the translator about this op yet" from "the input IR is malformed in
/// a way Phase 1 cares about". Per the no-stubs rule, the translator
/// never emits a placeholder Op — anything it can't handle returns an
/// Err whose `variant` field names the legacy op for the caller's log.
#[derive(Debug, Clone)]
pub enum BuildError {
    /// A legacy `Instruction` variant has no translator arm yet.
    UnsupportedInstruction {
        position: usize,
        variant: &'static str,
    },
    /// A legacy `Value` variant appeared where Phase 1 expects a
    /// `Value::Register` or `Value::Local`. Most commonly arises when the
    /// IR emits a raw immediate / pointer where a vreg was expected.
    UnsupportedValueKind { position: usize, msg: String },
    /// A `JumpIf`'s fall-through target (the position immediately after
    /// the JumpIf) is not a leader. Means Phase 1a's leader computation
    /// and the construction's expectations disagree — bug, not a
    /// translator gap.
    FallThroughNotLeader { position: usize },
    /// A `Jump`/`JumpIf`/`Throw` target label resolves to a position that
    /// is not a leader. Also a leader-computation bug.
    LabelTargetNotLeader { position: usize, label: usize },
    /// A label was named by an instruction but never appeared as an
    /// `Instruction::Label(_)` marker in the IR. Indicates a broken IR
    /// builder upstream.
    UnknownLabel { position: usize, label: usize },
}

/// Build a `CfgFunction` from a legacy `Ir`.
///
/// Translates the linear IR to a CFG with explicit terminators and
/// block-param-style argument passing. Function arguments (read in the
/// entry by a run of `RegisterArgument` ops) become entry-block params;
/// this is the only "promote at construction" path allowed in Phase 1
/// (per I3), and it is OK because arguments are not locals (they're
/// inputs to the function, not mutable storage), so I6 doesn't apply.
///
/// All other locals (`Value::Local(n)`) stay materialized as
/// `Op::SlotLoad` / `Op::SlotStore` per I6; mem2reg in Phase 2 decides
/// whether to promote them.
///
/// Phase 1b-1 handles: `RegisterArgument`, `LoadLocal`, `StoreLocal`,
/// `AddInt`, `Ret`, `Jump`, `JumpIf`, `Label` (filtered). Every other
/// `Instruction` variant returns `Err(UnsupportedInstruction)`.
pub fn build_cfg(ir: &Ir) -> Result<CfgFunction, BuildError> {
    let mut f = CfgFunction::new(ir.debug_name.clone(), ir.num_locals as u32);

    // Register every VReg this function references, with its inferred
    // class. Phase 1b-1 has only GP-producing ops, so the table will be
    // all-GP today; 1b-2 adds FP-producing ops and the class inference
    // will start mattering.
    let class_table = classify_vregs(&ir.instructions);
    let max_vreg = class_table.keys().copied().max().unwrap_or(0);
    for idx in 0..=max_vreg {
        let class = class_table.get(&idx).copied().unwrap_or(RegClass::Gp);
        f.vreg_classes.push(class);
    }

    let leaders = compute_leaders(&ir.instructions);
    if leaders.is_empty() {
        return Ok(f);
    }

    let mut leader_to_block: HashMap<usize, BlockId> = HashMap::new();
    for &leader_pos in &leaders {
        let bid = f.new_block();
        leader_to_block.insert(leader_pos, bid);
    }
    f.entry = leader_to_block[&0];

    let label_pos = label_positions(&ir.instructions);

    for (i, &leader_pos) in leaders.iter().enumerate() {
        let end = leaders.get(i + 1).copied().unwrap_or(ir.instructions.len());
        let block_id = leader_to_block[&leader_pos];
        fill_block(
            &mut f,
            ir,
            block_id,
            leader_pos,
            end,
            &leader_to_block,
            &label_pos,
        )?;
    }

    rebuild_predecessors(&mut f);
    Ok(f)
}

/// Walk the [start, end) window of `ir.instructions`, populating the
/// block's params (for the entry block), body, and terminator.
fn fill_block(
    f: &mut CfgFunction,
    ir: &Ir,
    block_id: BlockId,
    start: usize,
    end: usize,
    leader_to_block: &HashMap<usize, BlockId>,
    label_pos: &HashMap<usize, usize>,
) -> Result<(), BuildError> {
    let mut idx = start;

    // Entry block: gather the run of `RegisterArgument(VR_n)` ops at the
    // top and promote each to a block param. This is the I3-compatible
    // way to introduce function arguments (block params, not phis).
    if block_id == f.entry {
        while idx < end {
            if let Instruction::RegisterArgument(Value::Register(vr)) = &ir.instructions[idx] {
                let v = VReg {
                    index: vr.index as u32,
                    class: RegClass::Gp,
                };
                f.block_mut(block_id).params.push(v);
                idx += 1;
            } else {
                break;
            }
        }
    }

    while idx < end {
        let inst = &ir.instructions[idx];

        // Skip Label markers — they served their purpose in leader
        // computation; I1 forbids them inside block bodies.
        if matches!(inst, Instruction::Label(_)) {
            idx += 1;
            continue;
        }
        // Skip RegisterArgument that didn't make it into the entry-param
        // run (e.g. a later block somehow contains one — shouldn't happen
        // in well-formed IR, but defensively ignore rather than translate).
        if matches!(inst, Instruction::RegisterArgument(_)) {
            idx += 1;
            continue;
        }

        if is_terminator(inst) {
            let term = translate_terminator(inst, idx, leader_to_block, label_pos)?;
            f.block_mut(block_id).terminator = term;
            idx += 1;
            continue;
        }

        let op = translate_op(inst, idx)?;
        f.block_mut(block_id).body.push(op);
        idx += 1;
    }

    // Implicit fall-through: legacy IR can let one block run into the
    // next without a terminator (e.g. when the last instruction of a
    // block is straight-line and the next instruction is a Label that
    // started a new leader). Synthesize a Jump to the fall-through block.
    if matches!(f.block(block_id).terminator, Terminator::Unreachable) {
        if let Some(&next_bid) = leader_to_block.get(&end) {
            f.block_mut(block_id).terminator = Terminator::Jump {
                target: next_bid,
                args: vec![],
            };
        }
        // No next leader => the function ends without a terminator on the
        // last block. The verifier will flag this with UnfilledTerminator.
    }
    Ok(())
}

fn translate_op(inst: &Instruction, position: usize) -> Result<Op, BuildError> {
    match inst {
        Instruction::LoadLocal(Value::Register(dst), Value::Local(idx)) => Ok(Op::SlotLoad {
            dst: VReg {
                index: dst.index as u32,
                class: RegClass::Gp,
            },
            slot: SlotId(*idx as u32),
        }),
        Instruction::StoreLocal(Value::Local(idx), Value::Register(src)) => Ok(Op::SlotStore {
            slot: SlotId(*idx as u32),
            src: VReg {
                index: src.index as u32,
                class: RegClass::Gp,
            },
        }),
        Instruction::AddInt(Value::Register(dst), Value::Register(lhs), Value::Register(rhs)) => {
            Ok(Op::AddInt {
                dst: gp_vreg(dst),
                lhs: gp_vreg(lhs),
                rhs: gp_vreg(rhs),
            })
        }
        // Explicit failure modes for mis-shaped values on otherwise
        // supported ops — surfaces them with the position rather than
        // matching a wider pattern and silently producing wrong CFG.
        Instruction::LoadLocal(..) | Instruction::StoreLocal(..) | Instruction::AddInt(..) => {
            Err(BuildError::UnsupportedValueKind {
                position,
                msg: format!("operand shape not handled for {}", instruction_name(inst)),
            })
        }
        _ => Err(BuildError::UnsupportedInstruction {
            position,
            variant: instruction_name(inst),
        }),
    }
}

fn translate_terminator(
    inst: &Instruction,
    position: usize,
    leader_to_block: &HashMap<usize, BlockId>,
    label_pos: &HashMap<usize, usize>,
) -> Result<Terminator, BuildError> {
    let resolve_label = |label: &Label| -> Result<BlockId, BuildError> {
        let pos = label_pos
            .get(&label.index)
            .ok_or(BuildError::UnknownLabel {
                position,
                label: label.index,
            })?;
        leader_to_block
            .get(pos)
            .copied()
            .ok_or(BuildError::LabelTargetNotLeader {
                position,
                label: label.index,
            })
    };
    match inst {
        Instruction::Ret(Value::Register(v)) => Ok(Terminator::Ret { value: gp_vreg(v) }),
        Instruction::Jump(l) => Ok(Terminator::Jump {
            target: resolve_label(l)?,
            args: vec![],
        }),
        Instruction::JumpIf(l, cond, lhs, rhs) => {
            let lhs_vr = require_register(lhs, position)?;
            let rhs_vr = require_register(rhs, position)?;
            let t_target = resolve_label(l)?;
            let f_target = leader_to_block
                .get(&(position + 1))
                .copied()
                .ok_or(BuildError::FallThroughNotLeader { position })?;
            Ok(Terminator::Branch {
                cond: *cond,
                lhs: gp_vreg(lhs_vr),
                rhs: gp_vreg(rhs_vr),
                t_target,
                t_args: vec![],
                f_target,
                f_args: vec![],
            })
        }
        // Operand-shape mismatch on a supported terminator.
        Instruction::Ret(..) => Err(BuildError::UnsupportedValueKind {
            position,
            msg: "Ret operand was not a Value::Register".to_string(),
        }),
        // Other terminators (Throw, TailRecurse) land in later sub-phases.
        _ => Err(BuildError::UnsupportedInstruction {
            position,
            variant: instruction_name(inst),
        }),
    }
}

fn require_register<'a>(
    value: &'a Value,
    position: usize,
) -> Result<&'a VirtualRegister, BuildError> {
    match value {
        Value::Register(vr) => Ok(vr),
        _ => Err(BuildError::UnsupportedValueKind {
            position,
            msg: format!("expected register, got {:?}", value),
        }),
    }
}

fn gp_vreg(vr: &VirtualRegister) -> VReg {
    VReg {
        index: vr.index as u32,
        class: RegClass::Gp,
    }
}

/// Recompute every block's `predecessors` list from the terminator graph.
/// Called after fill_block has populated all terminators.
fn rebuild_predecessors(f: &mut CfgFunction) {
    let n = f.blocks.len();
    let mut new_preds: Vec<Vec<BlockId>> = vec![Vec::new(); n];
    for (idx, block) in f.blocks.iter().enumerate() {
        let from = BlockId(idx as u32);
        for succ in block.terminator.successors() {
            new_preds[succ.0 as usize].push(from);
        }
    }
    for (idx, preds) in new_preds.into_iter().enumerate() {
        f.blocks[idx].predecessors = preds;
    }
}

/// Infer the RegClass of every VReg by scanning def sites.
///
/// Phase 1b-1: every translated op produces a GP value, so the table is
/// uniform GP. 1b-2 adds FP-producing ops (AddFloat, IntToFloat,
/// FmovGeneralToFloat, etc.) and the inference becomes meaningful.
fn classify_vregs(instructions: &[Instruction]) -> HashMap<u32, RegClass> {
    let mut classes = HashMap::new();
    for inst in instructions {
        for (vr, class) in def_class_of(inst) {
            classes.insert(vr, class);
        }
    }
    classes
}

fn def_class_of(inst: &Instruction) -> Vec<(u32, RegClass)> {
    let as_gp = |v: &Value| match v {
        Value::Register(vr) => Some((vr.index as u32, RegClass::Gp)),
        _ => None,
    };
    let mut out = Vec::new();
    match inst {
        Instruction::AddInt(dst, _, _) => out.extend(as_gp(dst)),
        Instruction::LoadLocal(dst, _) => out.extend(as_gp(dst)),
        Instruction::RegisterArgument(dst) => out.extend(as_gp(dst)),
        _ => {}
    }
    out
}

/// Wire-up helper: build the CFG and run the verifier, logging failures
/// under `BEAGLE_SSA_VERIFY=1`. Returns the constructed CFG on success
/// so callers can hand it to mem2reg / regalloc once those land. Never
/// aborts compilation — diagnostics go to stderr.
pub fn try_build_and_verify(ir: &Ir) -> Option<CfgFunction> {
    let enabled = std::env::var("BEAGLE_SSA_VERIFY")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);
    if !enabled {
        return None;
    }
    let name = ir.debug_name.as_deref().unwrap_or("<anonymous>");
    match build_cfg(ir) {
        Err(e) => {
            eprintln!("[cfg-build] {name}: {:?}", e);
            None
        }
        Ok(cfg) => match crate::cfg::verify::verify(&cfg) {
            Err(e) => {
                eprintln!("[cfg-verify] {name}: {:?}", e);
                None
            }
            Ok(()) => Some(cfg),
        },
    }
}

/// Diagnostic name for a legacy Instruction variant. Centralized here so
/// the failure messages from `build_cfg` always agree with the variant
/// names you can search for in `src/ir.rs`.
fn instruction_name(inst: &Instruction) -> &'static str {
    match inst {
        Instruction::Sub(..) => "Sub",
        Instruction::AddInt(..) => "AddInt",
        Instruction::Mul(..) => "Mul",
        Instruction::Div(..) => "Div",
        Instruction::Modulo(..) => "Modulo",
        Instruction::Assign(..) => "Assign",
        Instruction::Recurse(..) => "Recurse",
        Instruction::TailRecurse(..) => "TailRecurse",
        Instruction::JumpIf(..) => "JumpIf",
        Instruction::Jump(..) => "Jump",
        Instruction::Ret(..) => "Ret",
        Instruction::Breakpoint => "Breakpoint",
        Instruction::Compare(..) => "Compare",
        Instruction::Tag(..) => "Tag",
        Instruction::LoadTrue(..) => "LoadTrue",
        Instruction::LoadFalse(..) => "LoadFalse",
        Instruction::LoadConstant(..) => "LoadConstant",
        Instruction::Call(..) => "Call",
        Instruction::HeapLoad(..) => "HeapLoad",
        Instruction::HeapLoadReg(..) => "HeapLoadReg",
        Instruction::HeapLoadByteReg(..) => "HeapLoadByteReg",
        Instruction::HeapStore(..) => "HeapStore",
        Instruction::LoadLocal(..) => "LoadLocal",
        Instruction::StoreLocal(..) => "StoreLocal",
        Instruction::RegisterArgument(..) => "RegisterArgument",
        Instruction::PushStack(..) => "PushStack",
        Instruction::PopStack(..) => "PopStack",
        Instruction::GetStackPointer(..) => "GetStackPointer",
        Instruction::GetStackPointerImm(..) => "GetStackPointerImm",
        Instruction::GetFramePointer(..) => "GetFramePointer",
        Instruction::GetTag(..) => "GetTag",
        Instruction::Untag(..) => "Untag",
        Instruction::HeapStoreOffset(..) => "HeapStoreOffset",
        Instruction::HeapStoreByteOffsetMasked(..) => "HeapStoreByteOffsetMasked",
        Instruction::CurrentStackPosition(..) => "CurrentStackPosition",
        Instruction::ExtendLifeTime(..) => "ExtendLifeTime",
        Instruction::HeapStoreOffsetReg(..) => "HeapStoreOffsetReg",
        Instruction::HeapStoreByteOffsetReg(..) => "HeapStoreByteOffsetReg",
        Instruction::AtomicLoad(..) => "AtomicLoad",
        Instruction::AtomicStore(..) => "AtomicStore",
        Instruction::CompareAndSwap(..) => "CompareAndSwap",
        Instruction::StoreFloat(..) => "StoreFloat",
        Instruction::GuardInt(..) => "GuardInt",
        Instruction::GuardFloat(..) => "GuardFloat",
        Instruction::FmovGeneralToFloat(..) => "FmovGeneralToFloat",
        Instruction::FmovFloatToGeneral(..) => "FmovFloatToGeneral",
        Instruction::IntToFloat(..) => "IntToFloat",
        Instruction::FRoundToZero(..) => "FRoundToZero",
        Instruction::AddFloat(..) => "AddFloat",
        Instruction::SubFloat(..) => "SubFloat",
        Instruction::MulFloat(..) => "MulFloat",
        Instruction::DivFloat(..) => "DivFloat",
        Instruction::CompareFloat(..) => "CompareFloat",
        Instruction::ShiftRightImm(..) => "ShiftRightImm",
        Instruction::ShiftRightImmRaw(..) => "ShiftRightImmRaw",
        Instruction::AndImm(..) => "AndImm",
        Instruction::ShiftLeft(..) => "ShiftLeft",
        Instruction::ShiftRight(..) => "ShiftRight",
        Instruction::ShiftRightZero(..) => "ShiftRightZero",
        Instruction::And(..) => "And",
        Instruction::Or(..) => "Or",
        Instruction::Xor(..) => "Xor",
        Instruction::PushExceptionHandler(..) => "PushExceptionHandler",
        Instruction::PushResumableExceptionHandler(..) => "PushResumableExceptionHandler",
        Instruction::PopExceptionHandler(..) => "PopExceptionHandler",
        Instruction::PopExceptionHandlerById(..) => "PopExceptionHandlerById",
        Instruction::Throw(..) => "Throw",
        Instruction::ReadArgCount(..) => "ReadArgCount",
        Instruction::Label(..) => "Label",
        Instruction::PushPromptHandler(..) => "PushPromptHandler",
        Instruction::PopPromptHandler(..) => "PopPromptHandler",
        Instruction::PushPromptTag(..) => "PushPromptTag",
        Instruction::LoadLabelAddress(..) => "LoadLabelAddress",
        Instruction::CaptureContinuation(..) => "CaptureContinuation",
        Instruction::InlineBumpAllocate(..) => "InlineBumpAllocate",
        Instruction::CaptureContinuationTagged(..) => "CaptureContinuationTagged",
        Instruction::PerformEffect(..) => "PerformEffect",
        Instruction::ReturnFromShift(..) => "ReturnFromShift",
        Instruction::RecordGcSafepoint => "RecordGcSafepoint",
        Instruction::FeedbackOr(..) => "FeedbackOr",
        Instruction::TierUpCheck(..) => "TierUpCheck",
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Condition, Instruction, Value, VirtualRegister};

    fn vr(idx: usize) -> Value {
        Value::Register(VirtualRegister {
            argument: None,
            index: idx,
            volatile: false,
            is_physical: false,
        })
    }

    fn lbl(idx: usize) -> Label {
        Label { index: idx }
    }

    #[test]
    fn empty_has_no_leaders() {
        assert_eq!(compute_leaders(&[]), Vec::<usize>::new());
    }

    #[test]
    fn single_ret_has_one_leader() {
        let insts = vec![Instruction::Ret(vr(0))];
        assert_eq!(compute_leaders(&insts), vec![0]);
    }

    #[test]
    fn straight_line_has_one_leader() {
        let insts = vec![
            Instruction::AddInt(vr(0), vr(1), vr(2)),
            Instruction::AddInt(vr(3), vr(0), vr(1)),
            Instruction::Ret(vr(3)),
        ];
        assert_eq!(compute_leaders(&insts), vec![0]);
    }

    /// Vanilla pattern: Jump to a label with at least one other instruction
    /// between the Jump and the Label target. Both the post-Jump position
    /// and the label target become leaders.
    #[test]
    fn jump_with_intervening_instruction_creates_both_leaders() {
        // 0: Jump L0
        // 1: AddInt           (dead but still a block leader)
        // 2: Label L0
        // 3: Ret
        let insts = vec![
            Instruction::Jump(lbl(0)),
            Instruction::AddInt(vr(0), vr(1), vr(2)),
            Instruction::Label(lbl(0)),
            Instruction::Ret(vr(0)),
        ];
        assert_eq!(compute_leaders(&insts), vec![0, 1, 2]);
    }

    /// The orphan-block fix: a Jump immediately followed by the Label it
    /// targets must NOT produce two leaders at the same position. Without
    /// rule 4, post-Jump (pos 1) and Label-L0 (pos 1) would both be
    /// computed as leaders — which dedupes fine here, but the broader
    /// pattern (post-Jump-to-different-label landing on a Label marker)
    /// would create a real orphan. Rule 4 suppresses both consistently.
    #[test]
    fn jump_followed_immediately_by_label_skips_orphan_leader() {
        // 0: Jump L0
        // 1: Label L0
        // 2: Ret
        let insts = vec![
            Instruction::Jump(lbl(0)),
            Instruction::Label(lbl(0)),
            Instruction::Ret(vr(0)),
        ];
        assert_eq!(compute_leaders(&insts), vec![0, 1]);
    }

    /// The real-world orphan pattern from the prior `ssa` branch's
    /// orphan-blocks memo: a macro emits `Jump done; Label slow_path; ...`
    /// where Label slow_path is a *different* label from Jump's target.
    /// Without rule 4, post-Jump would be a leader at the Label position,
    /// producing a 1-instruction orphan wrapper before the slow body.
    #[test]
    fn jump_done_then_label_slowpath_produces_no_orphan() {
        // 0: <fast op with bail label L_slow>
        // 1: Jump L_done            (terminator)
        // 2: Label L_slow           (start of slow path — also a label target)
        // 3: <slow body>
        // 4: Label L_done
        // 5: Ret
        let insts = vec![
            Instruction::Sub(vr(0), vr(1), vr(2), lbl(0)), // L0 = L_slow
            Instruction::Jump(lbl(1)),                     // L1 = L_done
            Instruction::Label(lbl(0)),                    // L_slow target
            Instruction::AddInt(vr(3), vr(0), vr(1)),
            Instruction::Label(lbl(1)), // L_done target
            Instruction::Ret(vr(3)),
        ];
        // Expected leaders:
        //  0 (start; Sub at pos 0 is straight-line, not a terminator)
        //  2 (L_slow target; post-Jump-at-1 would otherwise also point here,
        //     but rule 4 suppresses that because pos 2 is a Label)
        //  4 (L_done target; post-AddInt-at-3 is pos 4 which is a Label —
        //     AddInt isn't a terminator anyway so rule 3 doesn't trigger)
        // Position 1 is NOT a leader (Sub didn't add it; rule 3 doesn't fire
        // on Sub because Sub isn't a terminator).
        assert_eq!(compute_leaders(&insts), vec![0, 2, 4]);
    }

    #[test]
    fn jumpif_creates_branch_and_fallthrough_leaders() {
        // 0: JumpIf L0 if Eq vr0 vr1   (terminator; branches to L0, falls
        //                                through to 1)
        // 1: AddInt                    (false branch)
        // 2: Jump L1                   (terminator)
        // 3: Label L0                  (true branch)
        // 4: AddInt
        // 5: Label L1                  (join)
        // 6: Ret
        let insts = vec![
            Instruction::JumpIf(lbl(0), Condition::Equal, vr(0), vr(1)),
            Instruction::AddInt(vr(2), vr(0), vr(1)),
            Instruction::Jump(lbl(1)),
            Instruction::Label(lbl(0)),
            Instruction::AddInt(vr(3), vr(0), vr(1)),
            Instruction::Label(lbl(1)),
            Instruction::Ret(vr(2)),
        ];
        // Leaders:
        //   0 (start)
        //   1 (post-JumpIf fall-through; pos 1 = AddInt, not a Label)
        //   3 (L0 target; post-Jump-at-2 is pos 3 = Label, suppressed)
        //   5 (L1 target; pos 5 = Label, post-AddInt-at-4 doesn't trigger
        //      rule 3 because AddInt isn't a terminator)
        assert_eq!(compute_leaders(&insts), vec![0, 1, 3, 5]);
    }

    #[test]
    fn loop_back_edge_creates_header_leader() {
        // 0: Label L_header
        // 1: AddInt
        // 2: JumpIf L_exit if Eq vr0 vr1
        // 3: Jump L_header             (back edge)
        // 4: Label L_exit
        // 5: Ret
        let insts = vec![
            Instruction::Label(lbl(0)),
            Instruction::AddInt(vr(2), vr(0), vr(1)),
            Instruction::JumpIf(lbl(1), Condition::Equal, vr(0), vr(1)),
            Instruction::Jump(lbl(0)),
            Instruction::Label(lbl(1)),
            Instruction::Ret(vr(2)),
        ];
        // Leaders:
        //   0 (start; also L_header target)
        //   3 (post-JumpIf fall-through; pos 3 = Jump, not a Label)
        //   4 (L_exit target; post-Jump-at-3 is pos 4 = Label, suppressed)
        assert_eq!(compute_leaders(&insts), vec![0, 3, 4]);
    }

    /// Throw is a terminator with a resume label argument. The resume
    /// label is a leader (rule 2). The position after the Throw is also
    /// a leader (rule 3) unless it lands on a Label (rule 4).
    #[test]
    fn throw_resume_label_becomes_leader() {
        // 0: Throw vr0 L_resume 0 0
        // 1: Label L_resume
        // 2: Ret
        let insts = vec![
            Instruction::Throw(vr(0), lbl(0), 0, 0),
            Instruction::Label(lbl(0)),
            Instruction::Ret(vr(0)),
        ];
        assert_eq!(compute_leaders(&insts), vec![0, 1]);
    }

    /// TailRecurse is a terminator with no label argument; it tail-jumps
    /// to function entry, which won't be expressed as a Label at this
    /// layer (entry is implicit at position 0). Rule 3 still triggers,
    /// producing a leader at post-TailRecurse — unless it lands on a Label.
    #[test]
    fn tail_recurse_then_dead_code_creates_post_leader() {
        // 0: TailRecurse vr0 [vr1]
        // 1: AddInt (dead, but a leader by rule 3)
        let insts = vec![
            Instruction::TailRecurse(vr(0), vec![vr(1)]),
            Instruction::AddInt(vr(2), vr(0), vr(1)),
        ];
        assert_eq!(compute_leaders(&insts), vec![0, 1]);
    }

    /// Bail-out ops (Sub with overflow label, GuardInt with type-mismatch
    /// label, etc.) are NOT terminators — they continue on success. But
    /// their bail-out label IS a leader (the slow path is its own block).
    #[test]
    fn guardint_bail_label_is_leader_but_guard_is_not_terminator() {
        // 0: GuardInt vr0 vr1 L_bail
        // 1: AddInt (continuation of fast path)
        // 2: Jump L_done
        // 3: Label L_bail
        // 4: AddInt (slow path body)
        // 5: Label L_done
        // 6: Ret
        let insts = vec![
            Instruction::GuardInt(vr(0), vr(1), lbl(0)),
            Instruction::AddInt(vr(2), vr(0), vr(1)),
            Instruction::Jump(lbl(1)),
            Instruction::Label(lbl(0)),
            Instruction::AddInt(vr(3), vr(0), vr(1)),
            Instruction::Label(lbl(1)),
            Instruction::Ret(vr(2)),
        ];
        // Leaders:
        //   0 (start; GuardInt is not a terminator so it doesn't fire rule 3)
        //   3 (L_bail target; post-Jump-at-2 is pos 3 = Label, suppressed)
        //   5 (L_done target; pos 5 = Label)
        assert_eq!(compute_leaders(&insts), vec![0, 3, 5]);
    }

    /// PushExceptionHandler doesn't transfer control, but its handler
    /// label is a leader because exception flow can jump there later.
    #[test]
    fn exception_handler_label_is_leader() {
        // 0: PushExceptionHandler L_handler vr0 0
        // 1: AddInt
        // 2: PopExceptionHandler 0
        // 3: Jump L_after
        // 4: Label L_handler
        // 5: AddInt
        // 6: Label L_after
        // 7: Ret
        let insts = vec![
            Instruction::PushExceptionHandler(lbl(0), vr(0), 0),
            Instruction::AddInt(vr(1), vr(0), vr(0)),
            Instruction::PopExceptionHandler(0),
            Instruction::Jump(lbl(1)),
            Instruction::Label(lbl(0)),
            Instruction::AddInt(vr(2), vr(0), vr(0)),
            Instruction::Label(lbl(1)),
            Instruction::Ret(vr(0)),
        ];
        // Leaders:
        //   0 (start)
        //   4 (L_handler target via PushExceptionHandler; also post-Jump-at-3
        //      lands here = Label, suppressed)
        //   6 (L_after target; pos 6 = Label)
        assert_eq!(compute_leaders(&insts), vec![0, 4, 6]);
    }

    // ---- build_cfg tests (Phase 1b-1) ---------------------------------

    fn ir_with(instructions: Vec<Instruction>, num_locals: usize, name: &str) -> Ir {
        let mut ir = Ir::new(0);
        ir.instructions = instructions;
        ir.num_locals = num_locals;
        ir.debug_name = Some(name.to_string());
        ir
    }

    /// `fn id(x) { x }` — one RegisterArgument, one Ret. The arg becomes
    /// an entry block param; body is empty; terminator is Ret(VR0).
    #[test]
    fn build_identity_function() {
        let ir = ir_with(
            vec![
                Instruction::RegisterArgument(vr(0)),
                Instruction::Ret(vr(0)),
            ],
            0,
            "id",
        );
        let cfg = build_cfg(&ir).expect("identity should build");
        crate::cfg::verify::verify(&cfg).expect("identity should verify");
        assert_eq!(cfg.num_blocks(), 1);
        let entry = cfg.block(cfg.entry);
        assert_eq!(entry.params.len(), 1);
        assert_eq!(entry.body.len(), 0);
        assert!(matches!(entry.terminator, Terminator::Ret { .. }));
    }

    /// `fn add(a, b) { a + b }` — two RegisterArguments, AddInt, Ret.
    /// Entry block has two params; body has one AddInt; Ret(VR2).
    #[test]
    fn build_add_function() {
        let ir = ir_with(
            vec![
                Instruction::RegisterArgument(vr(0)),
                Instruction::RegisterArgument(vr(1)),
                Instruction::AddInt(vr(2), vr(0), vr(1)),
                Instruction::Ret(vr(2)),
            ],
            0,
            "add",
        );
        let cfg = build_cfg(&ir).expect("add should build");
        crate::cfg::verify::verify(&cfg).expect("add should verify");
        let entry = cfg.block(cfg.entry);
        assert_eq!(entry.params.len(), 2);
        assert_eq!(entry.body.len(), 1);
        assert!(matches!(entry.body[0], Op::AddInt { .. }));
        assert!(matches!(entry.terminator, Terminator::Ret { .. }));
    }

    /// LoadLocal + StoreLocal translate to SlotLoad / SlotStore.
    /// Locals stay materialized as slots per I6.
    #[test]
    fn build_load_store_local() {
        let ir = ir_with(
            vec![
                Instruction::RegisterArgument(vr(0)),
                Instruction::StoreLocal(Value::Local(0), vr(0)),
                Instruction::LoadLocal(vr(1), Value::Local(0)),
                Instruction::Ret(vr(1)),
            ],
            1,
            "load_store",
        );
        let cfg = build_cfg(&ir).expect("load/store should build");
        crate::cfg::verify::verify(&cfg).expect("load/store should verify");
        let entry = cfg.block(cfg.entry);
        assert!(matches!(entry.body[0], Op::SlotStore { .. }));
        assert!(matches!(entry.body[1], Op::SlotLoad { .. }));
        assert_eq!(cfg.num_slots, 1);
    }

    /// JumpIf produces a Branch terminator with proper t/f targets, and
    /// the predecessors are stitched correctly. The diamond should
    /// verify (no critical edges because then/else each have one pred).
    #[test]
    fn build_jumpif_diamond() {
        // 0: RegisterArgument vr0
        // 1: RegisterArgument vr1
        // 2: JumpIf L_true Eq vr0 vr1
        // 3: AddInt vr2 vr0 vr1   (false branch)
        // 4: Jump L_join
        // 5: Label L_true
        // 6: AddInt vr2 vr0 vr1   (true branch; same dst in legacy is OK
        //    since we don't promote locals)
        // 7: Label L_join
        // 8: Ret vr2
        //
        // NOTE: vr2 is defined on both branches with the same index in
        // legacy IR (the IR builder reuses indices). For Phase 1b-1 the
        // verifier's dominance check would flag this if we constructed the
        // function such that vr2 is used after the join, because neither
        // def dominates the use across the join. To keep the diamond test
        // green we hold the result through a slot instead.
        let ir = ir_with(
            vec![
                Instruction::RegisterArgument(vr(0)),
                Instruction::RegisterArgument(vr(1)),
                Instruction::JumpIf(lbl(0), Condition::Equal, vr(0), vr(1)),
                Instruction::AddInt(vr(2), vr(0), vr(1)),
                Instruction::StoreLocal(Value::Local(0), vr(2)),
                Instruction::Jump(lbl(1)),
                Instruction::Label(lbl(0)),
                Instruction::AddInt(vr(3), vr(0), vr(1)),
                Instruction::StoreLocal(Value::Local(0), vr(3)),
                Instruction::Label(lbl(1)),
                Instruction::LoadLocal(vr(4), Value::Local(0)),
                Instruction::Ret(vr(4)),
            ],
            1,
            "diamond",
        );
        let cfg = build_cfg(&ir).expect("diamond should build");
        crate::cfg::verify::verify(&cfg).expect("diamond should verify");
        assert!(
            cfg.num_blocks() >= 3,
            "diamond should have at least entry/then/else/join blocks"
        );
    }

    /// Unsupported instruction → clear Err with the variant name.
    #[test]
    fn unsupported_instruction_returns_error() {
        let ir = ir_with(
            vec![Instruction::Breakpoint, Instruction::Ret(vr(0))],
            0,
            "with_breakpoint",
        );
        let err = build_cfg(&ir).expect_err("Breakpoint not yet supported");
        match err {
            BuildError::UnsupportedInstruction { variant, .. } => {
                assert_eq!(variant, "Breakpoint");
            }
            other => panic!("expected UnsupportedInstruction, got {:?}", other),
        }
    }

    /// Empty IR produces an empty CfgFunction (no blocks).
    #[test]
    fn empty_ir_produces_empty_function() {
        let ir = ir_with(vec![], 0, "empty");
        let cfg = build_cfg(&ir).expect("empty should build");
        assert_eq!(cfg.num_blocks(), 0);
    }
}
