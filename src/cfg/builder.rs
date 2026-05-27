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

use std::collections::{HashMap, HashSet};

use crate::cfg::{
    BlockId, CallTarget, CfgFunction, ClobberSet, InlineBranchOp, Op, RegClass, SlotId, TagSource,
    Terminator, VReg,
};
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
    // Labels named as targets by any instruction (per `label_targets`).
    // Rule 4 only suppresses post-terminator leaders if the Label at
    // position+1 is in this set — otherwise the Label is "dead" and not
    // a rule-2 leader, so suppressing rule 3 would leave no leader at
    // all and break fall-through resolution.
    let referenced: HashSet<usize> = instructions
        .iter()
        .flat_map(label_targets)
        .map(|l| l.index)
        .collect();

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
            if next < instructions.len() {
                let suppress = matches!(
                    &instructions[next],
                    Instruction::Label(l) if referenced.contains(&l.index)
                );
                if !suppress {
                    leaders.push(next);
                }
            }
        }
    }
    leaders.sort_unstable();
    leaders.dedup();
    leaders
}

/// True for instructions that end a basic block.
///
/// Includes:
/// - True terminators: `Jump`, `JumpIf`, `Ret`, `Throw`, `TailRecurse`.
/// - Bail-out arithmetic and guards (Sub, Mul, Div, Modulo, Shifts,
///   GuardInt, GuardFloat) — these have two successors (fall-through
///   and bail) and end a block in the CFG. The CFG layer models them
///   as `Terminator::InlineBranch`. The legacy IR runs them straight
///   line and lets the bail be a sibling block; we make the split
///   explicit so I1 holds.
/// - `InlineBumpAllocate` — same bail/fall-through structure.
///
/// False for: `Recurse` (returns and continues), exception-handler-push
/// ops (register a handler without transferring control), etc.
pub fn is_terminator(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Jump(_)
            | Instruction::JumpIf(_, _, _, _)
            | Instruction::Ret(_)
            | Instruction::Throw(_, _, _, _)
            | Instruction::TailRecurse(_, _)
            | Instruction::Sub(_, _, _, _)
            | Instruction::Mul(_, _, _, _)
            | Instruction::Div(_, _, _, _)
            | Instruction::Modulo(_, _, _, _)
            | Instruction::ShiftLeft(_, _, _, _)
            | Instruction::ShiftRight(_, _, _, _)
            | Instruction::ShiftRightZero(_, _, _, _)
            | Instruction::ShiftRightImm(_, _, _, _)
            | Instruction::GuardInt(_, _, _)
            | Instruction::GuardFloat(_, _, _)
            | Instruction::InlineBumpAllocate(_, _, _, _)
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

    // Classify every VReg the function references. The class table is the
    // single source of truth for what register class each VReg has, and is
    // consulted by `to_cfg_vreg` at every translation site so a VReg has
    // the same class everywhere it appears.
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
            &class_table,
        )?;
    }

    rebuild_predecessors(&mut f);
    crate::cfg::dump::maybe_dump_phase("00-after-cfg-ize", &f, false);
    crate::cfg::dump::maybe_verify_stage("00-after-cfg-ize", &f);

    // Phase 1c: split critical edges so I2 holds at every post-
    // construction phase boundary. Runs unconditionally — the spec
    // requires no critical edges to survive any pass.
    split_critical_edges(&mut f);
    crate::cfg::dump::maybe_dump_phase("01-after-split-critical-edges", &f, false);
    crate::cfg::dump::maybe_verify_stage("01-after-split-critical-edges", &f);

    // Phase 1d (light DCE): wipe out unreachable blocks. mem2reg's
    // dom-tree DFS only visits reachable blocks, so unreachable
    // predecessors of a reachable block would never get phi-arg
    // additions on their outgoing edges, producing ArgArityMismatch
    // when the verifier walks ALL terminators. Replacing unreachable
    // terminators with `Unreachable` removes their phantom edges from
    // the predecessor map.
    dce_unreachable_blocks(&mut f);
    crate::cfg::dump::maybe_dump_phase("02-after-dce", &f, false);
    crate::cfg::dump::maybe_verify_stage("02-after-dce", &f);

    // Phase 2a: lift cross-block legacy VRegs to slots. The legacy IR
    // emits VRegs whose def doesn't dominate every use (single-def, but
    // used across control-flow merges); SSA can't model this directly
    // (I5). The pass converts each such VReg's def to a SlotStore and
    // each use to a SlotLoad, producing slots that the subsequent
    // mem2reg run will lift back to SSA values + phi-params at
    // dominance frontiers.
    crate::cfg::lift_vregs::lift_cross_block_vregs(&mut f);
    crate::cfg::dump::maybe_dump_phase("03-after-lift", &f, false);
    crate::cfg::dump::maybe_verify_stage("03-after-lift", &f);

    // Phase 2b: mem2reg. Promotes profitable stack slots to SSA values
    // + block params at iterated dominance frontiers (Cytron-style).
    // Runs after lift so the slots created in Phase 2a are eligible for
    // promotion. Slots that don't pay (fewer than 2 reads) stay as
    // SlotLoad / SlotStore. Preserves I1–I8 by construction; phi
    // placement is "block params" not "Phi op" per I3 / F10.
    crate::cfg::mem2reg::promote_slots(&mut f);
    crate::cfg::dump::maybe_dump_phase("04-after-mem2reg", &f, true);
    crate::cfg::dump::maybe_verify_stage("04-after-mem2reg", &f);

    Ok(f)
}

/// Wipe out any block not reachable from `entry`. The block stays in
/// the `blocks` vec (so BlockId indices remain stable for everything
/// else that references them), but its body, params, and outgoing edges
/// are cleared and its terminator becomes `Unreachable`. Without this,
/// mem2reg's dom-tree DFS leaves unreachable predecessors' terminator-
/// edge args empty while the target block gains phi-params, triggering
/// `ArgArityMismatch` in the verifier.
fn dce_unreachable_blocks(f: &mut CfgFunction) {
    let reachable = crate::cfg::dom::compute_reachable(f);
    if reachable.len() == f.blocks.len() {
        return;
    }
    for (idx, block) in f.blocks.iter_mut().enumerate() {
        if !reachable.contains(&BlockId(idx as u32)) {
            block.params.clear();
            block.body.clear();
            block.terminator = Terminator::Unreachable;
        }
    }
    rebuild_predecessors(f);
}

/// Edge-position discriminator inside a terminator, used by the
/// critical-edge splitter to redirect exactly one outgoing edge of a
/// multi-successor terminator without disturbing its sibling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgePos {
    BranchTrue,
    BranchFalse,
    InlineFallThrough,
    InlineBail,
}

/// Split every critical edge — an edge whose source has >1 successors AND
/// whose target has >1 predecessors. Per **I2**, no critical edge may
/// survive any phase boundary. Phase 5 (edge resolution) relies on this:
/// parallel-copy implementation of block-param transfers is well-defined
/// only on single-pred-single-succ edges, eliminating the
/// const-before-branch clobber bug class (forbidden pattern F3).
///
/// For each critical edge, inserts a fresh empty block (`mid`) between
/// source and target. `mid`'s only contents are a `Jump(target, args)`
/// that forwards whatever args the source's terminator was passing along
/// the edge. For Phase 1 (no mem2reg promotion yet) the args are empty;
/// when mem2reg lands, this passthrough is already correct because
/// `mid` has no params of its own and the args flow source→mid→target
/// referencing source-block-dominated VRegs.
pub fn split_critical_edges(f: &mut CfgFunction) {
    // First pass: collect critical edges. Doing this in two passes avoids
    // mutating `f` while iterating its blocks.
    let mut to_split: Vec<(BlockId, EdgePos, BlockId)> = Vec::new();
    for (idx, block) in f.blocks.iter().enumerate() {
        let from = BlockId(idx as u32);
        match &block.terminator {
            Terminator::Branch {
                t_target, f_target, ..
            } => {
                if f.block(*t_target).predecessors.len() > 1 {
                    to_split.push((from, EdgePos::BranchTrue, *t_target));
                }
                if t_target != f_target && f.block(*f_target).predecessors.len() > 1 {
                    to_split.push((from, EdgePos::BranchFalse, *f_target));
                }
            }
            Terminator::InlineBranch {
                fall_through, bail, ..
            } => {
                if f.block(*fall_through).predecessors.len() > 1 {
                    to_split.push((from, EdgePos::InlineFallThrough, *fall_through));
                }
                if fall_through != bail && f.block(*bail).predecessors.len() > 1 {
                    to_split.push((from, EdgePos::InlineBail, *bail));
                }
            }
            _ => {} // Jump / Ret / Throw / Unreachable: <=1 successor
        }
    }

    // Second pass: split each. Each new `mid` block gets the highest
    // index allocated so far, so updating source's terminator field to
    // point at `mid` doesn't invalidate earlier BlockId references.
    for (source, pos, target) in to_split {
        let mid = f.new_block();
        let args = {
            let term = &mut f.block_mut(source).terminator;
            match (term, pos) {
                (
                    Terminator::Branch {
                        t_target, t_args, ..
                    },
                    EdgePos::BranchTrue,
                ) => {
                    let args = std::mem::take(t_args);
                    *t_target = mid;
                    args
                }
                (
                    Terminator::Branch {
                        f_target, f_args, ..
                    },
                    EdgePos::BranchFalse,
                ) => {
                    let args = std::mem::take(f_args);
                    *f_target = mid;
                    args
                }
                (
                    Terminator::InlineBranch {
                        fall_through,
                        fall_args,
                        ..
                    },
                    EdgePos::InlineFallThrough,
                ) => {
                    let args = std::mem::take(fall_args);
                    *fall_through = mid;
                    args
                }
                (
                    Terminator::InlineBranch {
                        bail, bail_args, ..
                    },
                    EdgePos::InlineBail,
                ) => {
                    let args = std::mem::take(bail_args);
                    *bail = mid;
                    args
                }
                _ => unreachable!("EdgePos / terminator mismatch in split_critical_edges"),
            }
        };
        f.block_mut(mid).terminator = Terminator::Jump { target, args };
    }

    rebuild_predecessors(f);
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
    classes: &HashMap<u32, RegClass>,
) -> Result<(), BuildError> {
    let mut idx = start;

    // Entry block: collect function-argument VRegs and promote them to
    // block params. The legacy IR doesn't emit an explicit op for each
    // argument — instead, `Ir::arg(n)` produces a `VirtualRegister` with
    // `argument: Some(n)` set, and the legacy regalloc binds those to
    // physical arg registers at function entry. We mirror that by
    // walking every VReg the function references and adding any with
    // an `argument` index as an entry block param (sorted by arg
    // position so the param order matches the ABI). This is the
    // I3-compatible introduction of function args — allowed to "promote
    // at construction" because arguments aren't locals.
    if block_id == f.entry {
        for v in collect_argument_vregs(&ir.instructions) {
            f.block_mut(block_id).params.push(to_cfg_vreg(&v, classes));
        }
    }

    while idx < end {
        let inst = &ir.instructions[idx];

        // Skip Label markers — I1 forbids them inside block bodies. Their
        // role was establishing leaders.
        if matches!(inst, Instruction::Label(_)) {
            idx += 1;
            continue;
        }
        // Skip RegisterArgument outside the entry-param run.
        if matches!(inst, Instruction::RegisterArgument(_)) {
            idx += 1;
            continue;
        }

        if is_terminator(inst) {
            let entry = f.entry;
            let mut pre_ops: Vec<Op> = Vec::new();
            let term = translate_terminator(
                inst,
                idx,
                leader_to_block,
                label_pos,
                classes,
                entry,
                f,
                &mut pre_ops,
            )?;
            for pre in pre_ops {
                f.block_mut(block_id).body.push(pre);
            }
            f.block_mut(block_id).terminator = term;
            idx += 1;
            continue;
        }

        let mut pre_ops: Vec<Op> = Vec::new();
        let op = translate_op(
            inst,
            idx,
            leader_to_block,
            label_pos,
            classes,
            f,
            &mut pre_ops,
        )?;
        for pre in pre_ops {
            f.block_mut(block_id).body.push(pre);
        }
        f.block_mut(block_id).body.push(op);
        idx += 1;
    }

    // Implicit fall-through: legacy IR can let one block run into the
    // next without a terminator. Synthesize a Jump to the fall-through
    // block. If no next leader exists, leave Unreachable — the verifier
    // will flag it with UnfilledTerminator.
    if matches!(f.block(block_id).terminator, Terminator::Unreachable) {
        if let Some(&next_bid) = leader_to_block.get(&end) {
            f.block_mut(block_id).terminator = Terminator::Jump {
                target: next_bid,
                args: vec![],
            };
        }
    }
    Ok(())
}

fn translate_op(
    inst: &Instruction,
    position: usize,
    leader_to_block: &HashMap<usize, BlockId>,
    label_pos: &HashMap<usize, usize>,
    classes: &HashMap<u32, RegClass>,
    f: &mut CfgFunction,
    pre_ops: &mut Vec<Op>,
) -> Result<Op, BuildError> {
    let v = |vr: &VirtualRegister| -> VReg { to_cfg_vreg(vr, classes) };
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
    use Instruction as I;
    match inst {
        // ---- Slots (I6) -------------------------------------------------
        I::LoadLocal(Value::Register(dst), Value::Local(idx)) => Ok(Op::SlotLoad {
            dst: v(dst),
            slot: SlotId(*idx as u32),
        }),
        I::StoreLocal(Value::Local(idx), Value::Register(src)) => Ok(Op::SlotStore {
            slot: SlotId(*idx as u32),
            src: v(src),
        }),

        // ---- Assign / LoadConstant: split by source kind ----------------
        I::Assign(Value::Register(dst), val) => translate_assign(dst, val, position, classes),
        I::LoadConstant(Value::Register(dst), val) => translate_assign(dst, val, position, classes),
        I::LoadTrue(Value::Register(dst)) => Ok(Op::ConstTrue { dst: v(dst) }),
        I::LoadFalse(Value::Register(dst)) => Ok(Op::ConstFalse { dst: v(dst) }),

        // ---- Integer arithmetic (no bail) ------------------------------
        I::AddInt(Value::Register(d), Value::Register(l), Value::Register(r)) => Ok(Op::AddInt {
            dst: v(d),
            lhs: v(l),
            rhs: v(r),
        }),

        // ---- Comparisons -----------------------------------------------
        I::Compare(Value::Register(d), Value::Register(l), Value::Register(r), cond) => {
            Ok(Op::Compare {
                dst: v(d),
                lhs: v(l),
                rhs: v(r),
                cond: *cond,
            })
        }
        I::CompareFloat(Value::Register(d), Value::Register(l), Value::Register(r), cond) => {
            Ok(Op::CompareFloat {
                dst: v(d),
                lhs: v(l),
                rhs: v(r),
                cond: *cond,
            })
        }

        // ---- FP arithmetic ---------------------------------------------
        I::AddFloat(Value::Register(d), Value::Register(l), Value::Register(r)) => {
            Ok(Op::AddFloat {
                dst: v(d),
                lhs: v(l),
                rhs: v(r),
            })
        }
        I::SubFloat(Value::Register(d), Value::Register(l), Value::Register(r)) => {
            Ok(Op::SubFloat {
                dst: v(d),
                lhs: v(l),
                rhs: v(r),
            })
        }
        I::MulFloat(Value::Register(d), Value::Register(l), Value::Register(r)) => {
            Ok(Op::MulFloat {
                dst: v(d),
                lhs: v(l),
                rhs: v(r),
            })
        }
        I::DivFloat(Value::Register(d), Value::Register(l), Value::Register(r)) => {
            Ok(Op::DivFloat {
                dst: v(d),
                lhs: v(l),
                rhs: v(r),
            })
        }

        // ---- Conversions / bit-moves between classes -------------------
        I::IntToFloat(Value::Register(d), Value::Register(s)) => Ok(Op::IntToFloat {
            dst: v(d),
            src: v(s),
        }),
        I::FRoundToZero(Value::Register(d), Value::Register(s)) => Ok(Op::FRoundToZero {
            dst: v(d),
            src: v(s),
        }),
        I::FmovGeneralToFloat(Value::Register(d), Value::Register(s)) => Ok(Op::FmovGpToFp {
            dst: v(d),
            src: v(s),
        }),
        I::FmovFloatToGeneral(Value::Register(d), Value::Register(s)) => Ok(Op::FmovFpToGp {
            dst: v(d),
            src: v(s),
        }),

        // ---- Tag bit manipulation --------------------------------------
        I::Tag(Value::Register(d), Value::Register(s), tag) => Ok(Op::Tag {
            dst: v(d),
            src: v(s),
            tag_source: extract_tag_source(tag, position, classes)?,
        }),
        I::Untag(Value::Register(d), Value::Register(s)) => Ok(Op::Untag {
            dst: v(d),
            src: v(s),
        }),
        I::GetTag(Value::Register(d), Value::Register(s)) => Ok(Op::GetTag {
            dst: v(d),
            src: v(s),
        }),

        // ---- Bit ops ---------------------------------------------------
        I::And(Value::Register(d), Value::Register(l), Value::Register(r)) => Ok(Op::And {
            dst: v(d),
            lhs: v(l),
            rhs: v(r),
        }),
        I::Or(Value::Register(d), Value::Register(l), Value::Register(r)) => Ok(Op::Or {
            dst: v(d),
            lhs: v(l),
            rhs: v(r),
        }),
        I::Xor(Value::Register(d), Value::Register(l), Value::Register(r)) => Ok(Op::Xor {
            dst: v(d),
            lhs: v(l),
            rhs: v(r),
        }),
        I::AndImm(Value::Register(d), Value::Register(s), imm) => Ok(Op::AndImm {
            dst: v(d),
            src: v(s),
            imm: *imm,
        }),
        I::ShiftRightImmRaw(Value::Register(d), Value::Register(s), imm) => {
            Ok(Op::ShiftRightImmRaw {
                dst: v(d),
                src: v(s),
                imm: *imm,
            })
        }

        // ---- Heap memory -----------------------------------------------
        I::HeapLoad(Value::Register(d), Value::Register(b), off) => Ok(Op::HeapLoad {
            dst: v(d),
            base: v(b),
            offset: *off,
        }),
        I::HeapLoadReg(Value::Register(d), Value::Register(b), Value::Register(o)) => {
            Ok(Op::HeapLoadReg {
                dst: v(d),
                base: v(b),
                offset: v(o),
            })
        }
        I::HeapLoadByteReg(Value::Register(d), Value::Register(b), Value::Register(o)) => {
            Ok(Op::HeapLoadByteReg {
                dst: v(d),
                base: v(b),
                offset: v(o),
            })
        }
        I::HeapStore(Value::Register(a), Value::Register(s)) => Ok(Op::HeapStore {
            addr: v(a),
            src: v(s),
        }),
        I::HeapStoreOffset(Value::Register(b), Value::Register(s), off) => {
            Ok(Op::HeapStoreOffset {
                base: v(b),
                src: v(s),
                offset: *off,
            })
        }
        I::HeapStoreOffsetReg(Value::Register(b), Value::Register(s), Value::Register(o)) => {
            Ok(Op::HeapStoreOffsetReg {
                base: v(b),
                src: v(s),
                offset: v(o),
            })
        }
        I::HeapStoreByteOffsetReg(Value::Register(b), Value::Register(s), Value::Register(o)) => {
            Ok(Op::HeapStoreByteOffsetReg {
                base: v(b),
                src: v(s),
                offset: v(o),
            })
        }
        I::HeapStoreByteOffsetMasked(
            Value::Register(p),
            Value::Register(va),
            Value::Register(t1),
            Value::Register(t2),
            offset,
            byte_offset,
            mask,
        ) => Ok(Op::HeapStoreByteOffsetMasked {
            ptr: v(p),
            val: v(va),
            temp1: v(t1),
            temp2: v(t2),
            offset: *offset,
            byte_offset: *byte_offset,
            mask: *mask,
        }),

        // ---- Atomic ----------------------------------------------------
        I::AtomicLoad(Value::Register(d), Value::Register(s)) => Ok(Op::AtomicLoad {
            dst: v(d),
            src: v(s),
        }),
        I::AtomicStore(Value::Register(a), Value::Register(s)) => Ok(Op::AtomicStore {
            addr: v(a),
            src: v(s),
        }),
        I::CompareAndSwap(Value::Register(a), Value::Register(e), Value::Register(n)) => {
            Ok(Op::CompareAndSwap {
                addr: v(a),
                expected: v(e),
                new: v(n),
            })
        }

        // ---- Float storage (parse string at compile time) --------------
        I::StoreFloat(Value::Register(d), Value::Register(t), text) => Ok(Op::StoreFloatConstant {
            dest: v(d),
            temp: v(t),
            value_text: text.clone(),
        }),

        // ---- Stack pointer manipulation --------------------------------
        I::PushStack(Value::Register(s)) => Ok(Op::PushStack { src: v(s) }),
        I::PopStack(Value::Register(d)) => Ok(Op::PopStack { dst: v(d) }),
        I::GetStackPointer(Value::Register(d), Value::Register(o)) => Ok(Op::GetStackPointer {
            dst: v(d),
            offset: v(o),
        }),
        I::GetStackPointerImm(Value::Register(d), off) => Ok(Op::GetStackPointerImm {
            dst: v(d),
            offset: *off,
        }),
        I::GetFramePointer(Value::Register(d)) => Ok(Op::GetFramePointer { dst: v(d) }),
        I::CurrentStackPosition(Value::Register(d)) => Ok(Op::CurrentStackPosition { dst: v(d) }),

        // ---- Variadic plumbing -----------------------------------------
        I::ReadArgCount(Value::Register(d)) => Ok(Op::ReadArgCount { dst: v(d) }),

        // ---- Misc no-VReg ops & lifetime markers -----------------------
        I::Breakpoint => Ok(Op::Breakpoint),
        I::ExtendLifeTime(Value::Register(s)) => Ok(Op::ExtendLifetime { src: v(s) }),
        I::RecordGcSafepoint => Ok(Op::RecordGcSafepoint),
        I::FeedbackOr(addr, bits) => Ok(Op::FeedbackOr {
            slot_addr: *addr,
            bits: *bits,
        }),
        I::TierUpCheck(c, n, t) => Ok(Op::TierUpCheck {
            counter_addr: *c,
            name_c_str_ptr: *n,
            trampoline_fn_ptr: *t,
        }),

        // ---- LoadLabelAddress: resolve to a BlockId --------------------
        I::LoadLabelAddress(Value::Register(d), l) => Ok(Op::ConstLabelAddress {
            dst: v(d),
            target: resolve_label(l)?,
        }),

        // ---- Calls (I7: per-call clobber set) --------------------------
        I::Call(Value::Register(dst), fn_ptr, args, is_builtin) => {
            let target = extract_call_target(fn_ptr, position, classes)?;
            let translated_args = translate_value_args(args, position, classes, f, pre_ops)?;
            Ok(Op::Call {
                dst: v(dst),
                target,
                args: translated_args,
                is_builtin: *is_builtin,
                clobbers: ClobberSet::AllCallerSaved,
            })
        }
        I::Recurse(Value::Register(dst), args) => {
            let translated_args = translate_value_args(args, position, classes, f, pre_ops)?;
            Ok(Op::Recurse {
                dst: v(dst),
                args: translated_args,
                clobbers: ClobberSet::AllCallerSaved,
            })
        }

        // ---- Exception handling ----------------------------------------
        I::PushExceptionHandler(handler_label, Value::Local(slot), builtin_fn_ptr) => {
            Ok(Op::PushExceptionHandler {
                handler: resolve_label(handler_label)?,
                result_slot: SlotId(*slot as u32),
                builtin_fn_ptr: *builtin_fn_ptr,
            })
        }
        I::PushResumableExceptionHandler(
            Value::Register(dst),
            catch_label,
            Value::Local(exc_slot),
            Value::Local(res_slot),
            builtin_fn_ptr,
        ) => Ok(Op::PushResumableExceptionHandler {
            dst: v(dst),
            catch_block: resolve_label(catch_label)?,
            exception_slot: SlotId(*exc_slot as u32),
            resume_slot: SlotId(*res_slot as u32),
            builtin_fn_ptr: *builtin_fn_ptr,
        }),
        I::PopExceptionHandler(builtin_fn_ptr) => Ok(Op::PopExceptionHandler {
            builtin_fn_ptr: *builtin_fn_ptr,
        }),
        I::PopExceptionHandlerById(Value::Register(id), builtin_fn_ptr) => {
            Ok(Op::PopExceptionHandlerById {
                handler_id: v(id),
                builtin_fn_ptr: *builtin_fn_ptr,
            })
        }

        // ---- Delimited continuations & prompts -------------------------
        I::PushPromptHandler(handler_label, Value::Local(slot), builtin_fn_ptr) => {
            Ok(Op::PushPromptHandler {
                handler: resolve_label(handler_label)?,
                result_slot: SlotId(*slot as u32),
                builtin_fn_ptr: *builtin_fn_ptr,
            })
        }
        I::PopPromptHandler(Value::Register(result), builtin_fn_ptr) => Ok(Op::PopPromptHandler {
            result: v(result),
            builtin_fn_ptr: *builtin_fn_ptr,
        }),
        I::PushPromptTag(Value::Register(tag), abort_label, Value::Local(slot), builtin_fn_ptr) => {
            Ok(Op::PushPromptTag {
                tag: v(tag),
                abort_block: resolve_label(abort_label)?,
                result_slot: SlotId(*slot as u32),
                builtin_fn_ptr: *builtin_fn_ptr,
            })
        }
        I::CaptureContinuation(Value::Register(dst), resume_label, res_slot, builtin_fn_ptr) => {
            Ok(Op::CaptureContinuation {
                dst: v(dst),
                resume_block: resolve_label(resume_label)?,
                result_slot: SlotId(*res_slot as u32),
                builtin_fn_ptr: *builtin_fn_ptr,
            })
        }
        I::CaptureContinuationTagged(
            Value::Register(dst),
            resume_label,
            res_slot,
            builtin_fn_ptr,
            Value::Register(tag),
        ) => Ok(Op::CaptureContinuationTagged {
            dst: v(dst),
            resume_block: resolve_label(resume_label)?,
            result_slot: SlotId(*res_slot as u32),
            builtin_fn_ptr: *builtin_fn_ptr,
            tag: v(tag),
        }),

        // ---- Algebraic effects -----------------------------------------
        I::PerformEffect(
            Value::Register(handler),
            Value::Register(enum_type),
            Value::Register(op_value),
            resume_label,
            res_slot,
            builtin_fn_ptr,
        ) => Ok(Op::PerformEffect {
            handler: v(handler),
            enum_type: v(enum_type),
            op_value: v(op_value),
            resume_block: resolve_label(resume_label)?,
            result_slot: SlotId(*res_slot as u32),
            builtin_fn_ptr: *builtin_fn_ptr,
        }),
        I::ReturnFromShift(Value::Register(value), Value::Register(cont_ptr), builtin_fn_ptr) => {
            Ok(Op::ReturnFromShift {
                value: v(value),
                cont_ptr: v(cont_ptr),
                builtin_fn_ptr: *builtin_fn_ptr,
            })
        }

        // ---- Mis-shaped operands on otherwise supported ops ------------
        I::LoadLocal(..)
        | I::StoreLocal(..)
        | I::AddInt(..)
        | I::Assign(..)
        | I::LoadConstant(..)
        | I::LoadTrue(..)
        | I::LoadFalse(..)
        | I::Compare(..)
        | I::CompareFloat(..)
        | I::AddFloat(..)
        | I::SubFloat(..)
        | I::MulFloat(..)
        | I::DivFloat(..)
        | I::IntToFloat(..)
        | I::FRoundToZero(..)
        | I::FmovGeneralToFloat(..)
        | I::FmovFloatToGeneral(..)
        | I::Tag(..)
        | I::Untag(..)
        | I::GetTag(..)
        | I::And(..)
        | I::Or(..)
        | I::Xor(..)
        | I::AndImm(..)
        | I::ShiftRightImmRaw(..)
        | I::HeapLoad(..)
        | I::HeapLoadReg(..)
        | I::HeapLoadByteReg(..)
        | I::HeapStore(..)
        | I::HeapStoreOffset(..)
        | I::HeapStoreOffsetReg(..)
        | I::HeapStoreByteOffsetReg(..)
        | I::HeapStoreByteOffsetMasked(..)
        | I::AtomicLoad(..)
        | I::AtomicStore(..)
        | I::CompareAndSwap(..)
        | I::StoreFloat(..)
        | I::PushStack(..)
        | I::PopStack(..)
        | I::GetStackPointer(..)
        | I::GetStackPointerImm(..)
        | I::GetFramePointer(..)
        | I::CurrentStackPosition(..)
        | I::ReadArgCount(..)
        | I::ExtendLifeTime(..)
        | I::LoadLabelAddress(..)
        | I::Call(..)
        | I::Recurse(..)
        | I::PushExceptionHandler(..)
        | I::PushResumableExceptionHandler(..)
        | I::PopExceptionHandlerById(..)
        | I::PushPromptHandler(..)
        | I::PopPromptHandler(..)
        | I::PushPromptTag(..)
        | I::CaptureContinuation(..)
        | I::CaptureContinuationTagged(..)
        | I::PerformEffect(..)
        | I::ReturnFromShift(..) => Err(BuildError::UnsupportedValueKind {
            position,
            msg: format!("operand shape not handled for {}", instruction_name(inst)),
        }),

        _ => Err(BuildError::UnsupportedInstruction {
            position,
            variant: instruction_name(inst),
        }),
    }
}

/// Split a legacy `Assign` or `LoadConstant` into the appropriate Op
/// variant based on the source kind.
fn translate_assign(
    dst: &VirtualRegister,
    val: &Value,
    position: usize,
    classes: &HashMap<u32, RegClass>,
) -> Result<Op, BuildError> {
    let dst_v = to_cfg_vreg(dst, classes);
    match val {
        Value::Register(src) => Ok(Op::Move {
            dst: dst_v,
            src: to_cfg_vreg(src, classes),
        }),
        Value::TaggedConstant(n) => Ok(Op::ConstTaggedInt {
            dst: dst_v,
            value: *n as i64,
        }),
        Value::StringConstantPtr(p) => Ok(Op::ConstStringPtr {
            dst: dst_v,
            ptr: *p,
        }),
        Value::KeywordConstantPtr(p) => Ok(Op::ConstKeywordPtr {
            dst: dst_v,
            ptr: *p,
        }),
        Value::Function(id) => Ok(Op::ConstFunctionId {
            dst: dst_v,
            function_id: *id,
        }),
        Value::Pointer(p) => Ok(Op::ConstPointer {
            dst: dst_v,
            ptr: *p,
        }),
        Value::RawValue(v_imm) => Ok(Op::ConstRawValue {
            dst: dst_v,
            value: *v_imm as u64,
        }),
        Value::True => Ok(Op::ConstTrue { dst: dst_v }),
        Value::False => Ok(Op::ConstFalse { dst: dst_v }),
        Value::Null => Ok(Op::ConstNull { dst: dst_v }),
        Value::Local(idx) => Ok(Op::SlotLoad {
            dst: dst_v,
            slot: SlotId(*idx as u32),
        }),
        Value::Spill(..) | Value::Stack(..) => Err(BuildError::UnsupportedValueKind {
            position,
            msg: format!("Assign/LoadConstant from {:?} not handled", val),
        }),
    }
}

fn translate_terminator(
    inst: &Instruction,
    position: usize,
    leader_to_block: &HashMap<usize, BlockId>,
    label_pos: &HashMap<usize, usize>,
    classes: &HashMap<u32, RegClass>,
    entry: BlockId,
    f: &mut CfgFunction,
    pre_ops: &mut Vec<Op>,
) -> Result<Terminator, BuildError> {
    let v = |vr: &VirtualRegister| -> VReg { to_cfg_vreg(vr, classes) };
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
    // For InlineBranch ops, the fall-through is the position right after
    // this op. We require it to be a leader (it always is — see Phase 1a
    // rule 3).
    let fall_through = || -> Result<BlockId, BuildError> {
        leader_to_block
            .get(&(position + 1))
            .copied()
            .ok_or(BuildError::FallThroughNotLeader { position })
    };
    use Instruction as I;
    match inst {
        // ---- True terminators -----------------------------------------
        I::Ret(value) => {
            let vr = value_to_vreg(value, position, classes, f, pre_ops)?;
            Ok(Terminator::Ret { value: vr })
        }
        I::Jump(l) => Ok(Terminator::Jump {
            target: resolve_label(l)?,
            args: vec![],
        }),
        I::JumpIf(l, cond, lhs, rhs) => {
            let lhs_vr = require_register(lhs, position)?;
            let rhs_vr = require_register(rhs, position)?;
            Ok(Terminator::Branch {
                cond: *cond,
                lhs: to_cfg_vreg(lhs_vr, classes),
                rhs: to_cfg_vreg(rhs_vr, classes),
                t_target: resolve_label(l)?,
                t_args: vec![],
                f_target: fall_through()?,
                f_args: vec![],
            })
        }
        // I8: tail self-call rewrites directly to a jump-to-entry. The
        // dst register is discarded (no return on a tail call).
        I::TailRecurse(_dst, args) => {
            let arg_vregs = translate_value_args(args, position, classes, f, pre_ops)?;
            Ok(Terminator::Jump {
                target: entry,
                args: arg_vregs,
            })
        }
        I::Throw(Value::Register(value), resume_label, resume_local_idx, builtin_fn_ptr) => {
            Ok(Terminator::Throw {
                value: v(value),
                resume: resolve_label(resume_label)?,
                resume_args: vec![],
                resume_local: SlotId(*resume_local_idx as u32),
                builtin_fn_ptr: *builtin_fn_ptr,
            })
        }

        // ---- InlineBranch family: bail-out arithmetic + guards + bump-
        //      allocate. fall_through is position+1; bail is the named
        //      label.
        I::Sub(Value::Register(d), Value::Register(l), Value::Register(r), bail) => {
            Ok(Terminator::InlineBranch {
                op: InlineBranchOp::SubChecked {
                    dst: v(d),
                    lhs: v(l),
                    rhs: v(r),
                },
                fall_through: fall_through()?,
                fall_args: vec![],
                bail: resolve_label(bail)?,
                bail_args: vec![],
            })
        }
        I::Mul(Value::Register(d), Value::Register(l), Value::Register(r), bail) => {
            Ok(Terminator::InlineBranch {
                op: InlineBranchOp::MulChecked {
                    dst: v(d),
                    lhs: v(l),
                    rhs: v(r),
                },
                fall_through: fall_through()?,
                fall_args: vec![],
                bail: resolve_label(bail)?,
                bail_args: vec![],
            })
        }
        I::Div(Value::Register(d), Value::Register(l), Value::Register(r), bail) => {
            Ok(Terminator::InlineBranch {
                op: InlineBranchOp::DivChecked {
                    dst: v(d),
                    lhs: v(l),
                    rhs: v(r),
                },
                fall_through: fall_through()?,
                fall_args: vec![],
                bail: resolve_label(bail)?,
                bail_args: vec![],
            })
        }
        I::Modulo(Value::Register(d), Value::Register(l), Value::Register(r), bail) => {
            Ok(Terminator::InlineBranch {
                op: InlineBranchOp::ModuloChecked {
                    dst: v(d),
                    lhs: v(l),
                    rhs: v(r),
                },
                fall_through: fall_through()?,
                fall_args: vec![],
                bail: resolve_label(bail)?,
                bail_args: vec![],
            })
        }
        I::ShiftLeft(Value::Register(d), Value::Register(l), Value::Register(r), bail) => {
            Ok(Terminator::InlineBranch {
                op: InlineBranchOp::ShiftLeftChecked {
                    dst: v(d),
                    lhs: v(l),
                    rhs: v(r),
                },
                fall_through: fall_through()?,
                fall_args: vec![],
                bail: resolve_label(bail)?,
                bail_args: vec![],
            })
        }
        I::ShiftRight(Value::Register(d), Value::Register(l), Value::Register(r), bail) => {
            Ok(Terminator::InlineBranch {
                op: InlineBranchOp::ShiftRightChecked {
                    dst: v(d),
                    lhs: v(l),
                    rhs: v(r),
                },
                fall_through: fall_through()?,
                fall_args: vec![],
                bail: resolve_label(bail)?,
                bail_args: vec![],
            })
        }
        I::ShiftRightZero(Value::Register(d), Value::Register(l), Value::Register(r), bail) => {
            Ok(Terminator::InlineBranch {
                op: InlineBranchOp::ShiftRightZeroChecked {
                    dst: v(d),
                    lhs: v(l),
                    rhs: v(r),
                },
                fall_through: fall_through()?,
                fall_args: vec![],
                bail: resolve_label(bail)?,
                bail_args: vec![],
            })
        }
        I::ShiftRightImm(Value::Register(d), Value::Register(s), imm, bail) => {
            Ok(Terminator::InlineBranch {
                op: InlineBranchOp::ShiftRightImmChecked {
                    dst: v(d),
                    src: v(s),
                    imm: *imm,
                },
                fall_through: fall_through()?,
                fall_args: vec![],
                bail: resolve_label(bail)?,
                bail_args: vec![],
            })
        }
        I::GuardInt(Value::Register(d), Value::Register(s), bail) => Ok(Terminator::InlineBranch {
            op: InlineBranchOp::GuardInt {
                dst: v(d),
                src: v(s),
            },
            fall_through: fall_through()?,
            fall_args: vec![],
            bail: resolve_label(bail)?,
            bail_args: vec![],
        }),
        I::GuardFloat(Value::Register(d), Value::Register(s), bail) => {
            Ok(Terminator::InlineBranch {
                op: InlineBranchOp::GuardFloat {
                    dst: v(d),
                    src: v(s),
                },
                fall_through: fall_through()?,
                fall_args: vec![],
                bail: resolve_label(bail)?,
                bail_args: vec![],
            })
        }
        I::InlineBumpAllocate(Value::Register(d), size_bytes, header, bail) => {
            Ok(Terminator::InlineBranch {
                op: InlineBranchOp::InlineBumpAllocate {
                    dst: v(d),
                    size_bytes: *size_bytes,
                    header: *header,
                },
                fall_through: fall_through()?,
                fall_args: vec![],
                bail: resolve_label(bail)?,
                bail_args: vec![],
            })
        }

        // ---- Mis-shaped operands on supported terminators ------------
        I::Throw(..)
        | I::Sub(..)
        | I::Mul(..)
        | I::Div(..)
        | I::Modulo(..)
        | I::ShiftLeft(..)
        | I::ShiftRight(..)
        | I::ShiftRightZero(..)
        | I::ShiftRightImm(..)
        | I::GuardInt(..)
        | I::GuardFloat(..)
        | I::InlineBumpAllocate(..) => Err(BuildError::UnsupportedValueKind {
            position,
            msg: format!("operand shape not handled for {}", instruction_name(inst)),
        }),

        _ => Err(BuildError::UnsupportedInstruction {
            position,
            variant: instruction_name(inst),
        }),
    }
}

/// Translate a `Vec<Value>` of args into a `Vec<VReg>`. Every Value must
/// be a `Value::Register` — non-register args (literals, locals, etc.)
/// Translate a `&[Value]` of args into `Vec<VReg>`. Values that aren't
/// already registers (`Local`, common constants) get materialized into
/// fresh VRegs via prepended ops on `pre_ops`. Used by TailRecurse and
/// the call ops.
fn translate_value_args(
    args: &[Value],
    position: usize,
    classes: &HashMap<u32, RegClass>,
    f: &mut CfgFunction,
    pre_ops: &mut Vec<Op>,
) -> Result<Vec<VReg>, BuildError> {
    args.iter()
        .map(|a| value_to_vreg(a, position, classes, f, pre_ops))
        .collect()
}

/// Resolve a `Value` to a `VReg`. If `value` is already a `Register`,
/// just look up its class. Otherwise materialize it into a fresh VReg
/// by prepending an appropriate op (`SlotLoad` for `Local`, `Const*`
/// for the common constant kinds). The prepended ops go onto
/// `pre_ops` — the caller (translate_op / translate_terminator)
/// arranges for those to land in the block body before the using op.
fn value_to_vreg(
    value: &Value,
    position: usize,
    classes: &HashMap<u32, RegClass>,
    f: &mut CfgFunction,
    pre_ops: &mut Vec<Op>,
) -> Result<VReg, BuildError> {
    match value {
        Value::Register(vr) => Ok(to_cfg_vreg(vr, classes)),
        Value::Local(idx) => {
            let dst = f.new_vreg(RegClass::Gp);
            pre_ops.push(Op::SlotLoad {
                dst,
                slot: SlotId(*idx as u32),
            });
            Ok(dst)
        }
        Value::TaggedConstant(n) => {
            let dst = f.new_vreg(RegClass::Gp);
            pre_ops.push(Op::ConstTaggedInt {
                dst,
                value: *n as i64,
            });
            Ok(dst)
        }
        Value::True => {
            let dst = f.new_vreg(RegClass::Gp);
            pre_ops.push(Op::ConstTrue { dst });
            Ok(dst)
        }
        Value::False => {
            let dst = f.new_vreg(RegClass::Gp);
            pre_ops.push(Op::ConstFalse { dst });
            Ok(dst)
        }
        Value::Null => {
            let dst = f.new_vreg(RegClass::Gp);
            pre_ops.push(Op::ConstNull { dst });
            Ok(dst)
        }
        Value::StringConstantPtr(p) => {
            let dst = f.new_vreg(RegClass::Gp);
            pre_ops.push(Op::ConstStringPtr { dst, ptr: *p });
            Ok(dst)
        }
        Value::KeywordConstantPtr(p) => {
            let dst = f.new_vreg(RegClass::Gp);
            pre_ops.push(Op::ConstKeywordPtr { dst, ptr: *p });
            Ok(dst)
        }
        Value::Function(id) => {
            let dst = f.new_vreg(RegClass::Gp);
            pre_ops.push(Op::ConstFunctionId {
                dst,
                function_id: *id,
            });
            Ok(dst)
        }
        Value::Pointer(p) => {
            let dst = f.new_vreg(RegClass::Gp);
            pre_ops.push(Op::ConstPointer { dst, ptr: *p });
            Ok(dst)
        }
        Value::RawValue(v) => {
            let dst = f.new_vreg(RegClass::Gp);
            pre_ops.push(Op::ConstRawValue {
                dst,
                value: *v as u64,
            });
            Ok(dst)
        }
        Value::Spill(..) | Value::Stack(..) => Err(BuildError::UnsupportedValueKind {
            position,
            msg: format!("cannot materialize {:?} as a VReg", value),
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

/// Build a `VReg` from a legacy `VirtualRegister`, looking up its class
/// from the classification table. The table is computed once via
/// `classify_vregs` and is authoritative for every VReg's class.
fn to_cfg_vreg(vr: &VirtualRegister, classes: &HashMap<u32, RegClass>) -> VReg {
    let class = classes
        .get(&(vr.index as u32))
        .copied()
        .unwrap_or(RegClass::Gp);
    VReg {
        index: vr.index as u32,
        class,
    }
}

/// Extract a 64-bit immediate from a `Value` that's expected to encode a
/// compile-time constant. Used for the tag-bits argument of `Tag` in the
/// common case (BuiltInType-derived constant).
fn extract_imm_u64(value: &Value, position: usize) -> Result<u64, BuildError> {
    match value {
        Value::TaggedConstant(n) => Ok(*n as u64),
        Value::RawValue(v) => Ok(*v as u64),
        _ => Err(BuildError::UnsupportedValueKind {
            position,
            msg: format!("expected constant immediate, got {:?}", value),
        }),
    }
}

/// Extract a `TagSource` from `Tag`'s third arg. Tag accepts either a
/// register (runtime-computed tag, e.g. from a struct id) or a
/// compile-time immediate (the common case).
fn extract_tag_source(
    value: &Value,
    position: usize,
    classes: &HashMap<u32, RegClass>,
) -> Result<TagSource, BuildError> {
    match value {
        Value::Register(vr) => Ok(TagSource::Register(to_cfg_vreg(vr, classes))),
        Value::TaggedConstant(n) => Ok(TagSource::Bits(*n as u64)),
        Value::RawValue(v) => Ok(TagSource::Bits(*v as u64)),
        _ => Err(BuildError::UnsupportedValueKind {
            position,
            msg: format!(
                "Tag third arg: expected register or immediate, got {:?}",
                value
            ),
        }),
    }
}

/// Extract a `CallTarget` from a Call's function-position arg. Accepts
/// register (computed function pointer) and the common constant kinds
/// (function id, raw pointer, raw value).
fn extract_call_target(
    value: &Value,
    position: usize,
    classes: &HashMap<u32, RegClass>,
) -> Result<CallTarget, BuildError> {
    match value {
        Value::Register(vr) => Ok(CallTarget::Register(to_cfg_vreg(vr, classes))),
        Value::Function(id) => Ok(CallTarget::FunctionId(*id)),
        Value::Pointer(p) => Ok(CallTarget::Pointer(*p)),
        Value::RawValue(v) => Ok(CallTarget::Raw(*v as u64)),
        _ => Err(BuildError::UnsupportedValueKind {
            position,
            msg: format!("Call fn position: unsupported value kind {:?}", value),
        }),
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
/// Two-pass algorithm:
///
/// 1. **Definite-class pass.** Every Instruction that unambiguously
///    determines its def's class (AddInt → GP, AddFloat → FP, etc.) sets
///    that class in the table.
/// 2. **Fixpoint pass.** `Assign`/`LoadConstant` of one VReg from another
///    propagates the source's class to the destination. Iterates until
///    stable, which handles chains like `vrA = AddFloat(...); vrB = vrA;
///    vrC = vrB` where the FP class flows from vrA through vrB to vrC.
///
/// VRegs not classified by either pass default to GP at `to_cfg_vreg`
/// lookup time. In practice this only happens for VRegs that the legacy
/// IR uses as undefined-elsewhere (which is malformed anyway) — the
/// verifier flags any mismatch when a downstream op disagrees.
fn classify_vregs(instructions: &[Instruction]) -> HashMap<u32, RegClass> {
    let mut classes = HashMap::new();

    // Pass 0: function-argument VRegs. The legacy IR doesn't emit an
    // explicit "register argument" op; arg VRegs are simply
    // `VirtualRegister` values with `argument: Some(n)` set. They're
    // "defined" by the calling convention at function entry and need to
    // be in the class table (and later, entry block params) even though
    // no Instruction in the body defines them.
    for vr in collect_argument_vregs(instructions) {
        classes.insert(vr.index as u32, RegClass::Gp);
    }

    // Pass 1: definite-class defs.
    for inst in instructions {
        for (vr, class) in def_class_of(inst) {
            classes.insert(vr, class);
        }
    }

    // Pass 2: default-GP for `Assign` / `LoadConstant` dsts that aren't
    // already classified. Without this, `Assign(Register, NonRegister)` —
    // e.g. Assign of a `TaggedConstant` — would leave dst unclassified
    // (def_class_of doesn't cover Assign, and the reg-reg fixpoint
    // doesn't fire on a non-register source). Pass 3's fixpoint can
    // override this with FP if a chained reg-reg move propagates FP from
    // an FP-producing op.
    for inst in instructions {
        let dst_idx = match inst {
            Instruction::Assign(Value::Register(d), _) => Some(d.index as u32),
            Instruction::LoadConstant(Value::Register(d), _) => Some(d.index as u32),
            _ => None,
        };
        if let Some(idx) = dst_idx {
            classes.entry(idx).or_insert(RegClass::Gp);
        }
    }

    // Pass 3: fixpoint over reg-to-reg Assign / LoadConstant moves.
    loop {
        let mut changed = false;
        for inst in instructions {
            let (dst_vr, src_vr) = match inst {
                Instruction::Assign(Value::Register(d), Value::Register(s)) => (d, s),
                Instruction::LoadConstant(Value::Register(d), Value::Register(s)) => (d, s),
                _ => continue,
            };
            if let Some(&src_class) = classes.get(&(src_vr.index as u32)) {
                let dst_idx = dst_vr.index as u32;
                if classes.get(&dst_idx) != Some(&src_class) {
                    classes.insert(dst_idx, src_class);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    classes
}

/// Collect the function-argument `VirtualRegister` values referenced by
/// any instruction in the function, sorted by argument position.
///
/// `Ir::arg(n)` returns a fresh `VirtualRegister` with `argument:
/// Some(n)` for the n'th function parameter. Multiple calls to
/// `arg(n)` produce different VRegs all carrying `argument: Some(n)`;
/// we take the lowest-indexed VReg per argument position (deterministic
/// and matches the first IR allocation order, which the legacy regalloc
/// uses for arg-register binding).
fn collect_argument_vregs(instructions: &[Instruction]) -> Vec<VirtualRegister> {
    // arg_index → (vreg_index, vreg). Keep the lowest vreg_index per arg.
    let mut by_arg: HashMap<usize, VirtualRegister> = HashMap::new();
    for inst in instructions {
        for vr in inst.get_registers() {
            if let Some(arg_idx) = vr.argument {
                by_arg
                    .entry(arg_idx)
                    .and_modify(|existing| {
                        if vr.index < existing.index {
                            *existing = vr;
                        }
                    })
                    .or_insert(vr);
            }
        }
    }
    let mut result: Vec<(usize, VirtualRegister)> = by_arg.into_iter().collect();
    result.sort_by_key(|&(arg_idx, _)| arg_idx);
    result.into_iter().map(|(_, vr)| vr).collect()
}

/// For each Instruction, return `(vreg_index, RegClass)` for every VReg
/// the instruction definitively defines with a known class. Assign and
/// LoadConstant are handled by the fixpoint in `classify_vregs`.
fn def_class_of(inst: &Instruction) -> Vec<(u32, RegClass)> {
    let as_class = |val: &Value, c: RegClass| -> Option<(u32, RegClass)> {
        match val {
            Value::Register(vr) => Some((vr.index as u32, c)),
            _ => None,
        }
    };
    let mut out = Vec::new();
    use Instruction as I;
    use RegClass::*;
    match inst {
        // GP defs ----------------------------------------------------
        I::AddInt(dst, _, _)
        | I::LoadLocal(dst, _)
        | I::RegisterArgument(dst)
        | I::Compare(dst, _, _, _)
        | I::CompareFloat(dst, _, _, _)
        | I::LoadTrue(dst)
        | I::LoadFalse(dst)
        | I::Tag(dst, _, _)
        | I::Untag(dst, _)
        | I::GetTag(dst, _)
        | I::And(dst, _, _)
        | I::Or(dst, _, _)
        | I::Xor(dst, _, _)
        | I::AndImm(dst, _, _)
        | I::ShiftRightImmRaw(dst, _, _)
        | I::HeapLoad(dst, _, _)
        | I::HeapLoadReg(dst, _, _)
        | I::HeapLoadByteReg(dst, _, _)
        | I::AtomicLoad(dst, _)
        | I::PopStack(dst)
        | I::GetFramePointer(dst)
        | I::CurrentStackPosition(dst)
        | I::GetStackPointer(dst, _)
        | I::GetStackPointerImm(dst, _)
        | I::ReadArgCount(dst)
        | I::LoadLabelAddress(dst, _)
        | I::FmovFloatToGeneral(dst, _)
        // Bail-out arithmetic terminators (Phase 1b-3 InlineBranch ops):
        | I::Sub(dst, _, _, _)
        | I::Mul(dst, _, _, _)
        | I::Div(dst, _, _, _)
        | I::Modulo(dst, _, _, _)
        | I::ShiftLeft(dst, _, _, _)
        | I::ShiftRight(dst, _, _, _)
        | I::ShiftRightZero(dst, _, _, _)
        | I::ShiftRightImm(dst, _, _, _)
        // GuardInt / GuardFloat: dst is a GP scratch used for the tag-bit
        // compare. Its value after the guard is junk, not the untagged
        // result — callers rely on the original src register going forward.
        | I::GuardInt(dst, _, _)
        | I::GuardFloat(dst, _, _)
        // InlineBumpAllocate: dst is a tagged HeapObject pointer (GP).
        | I::InlineBumpAllocate(dst, _, _, _)
        // Calls and recursion: dst holds the return value (tagged, GP).
        | I::Call(dst, _, _, _)
        | I::Recurse(dst, _)
        | I::TailRecurse(dst, _)
        // Exception / continuation handle defs (all tagged GP values).
        | I::PushResumableExceptionHandler(dst, _, _, _, _)
        | I::CaptureContinuation(dst, _, _, _)
        | I::CaptureContinuationTagged(dst, _, _, _, _) => out.extend(as_class(dst, Gp)),

        // FP defs ----------------------------------------------------
        I::AddFloat(dst, _, _)
        | I::SubFloat(dst, _, _)
        | I::MulFloat(dst, _, _)
        | I::DivFloat(dst, _, _)
        | I::IntToFloat(dst, _)
        | I::FRoundToZero(dst, _)
        | I::FmovGeneralToFloat(dst, _) => out.extend(as_class(dst, Fp)),

        // Multi-def ops ----------------------------------------------
        I::HeapStoreByteOffsetMasked(_, _, t1, t2, _, _, _) => {
            out.extend(as_class(t1, Gp));
            out.extend(as_class(t2, Gp));
        }
        I::StoreFloat(_, temp, _) => out.extend(as_class(temp, Gp)),

        // Everything else (terminators, stores, Assign, LoadConstant,
        // bail-out arithmetic, calls, exception/continuation, etc.)
        // defines nothing for class-inference purposes here.
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

    /// A VReg marked as the `arg_position`-th function argument. Mirrors
    /// what `Ir::arg(n)` produces in the real compiler — sets
    /// `argument: Some(arg_position)` so `collect_argument_vregs` picks
    /// it up as an entry block param.
    fn arg_vr(idx: usize, arg_position: usize) -> Value {
        Value::Register(VirtualRegister {
            argument: Some(arg_position),
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
    /// orphan-blocks memo: a macro emits `Jump done; Label slow_path; ...`.
    /// With Sub now modeled as an InlineBranch terminator (Phase 1b-3),
    /// the post-Sub position is also a leader (the fast-path continuation
    /// is its own block). Rule 4 still prevents the post-Jump position
    /// from doubling up the L_slow leader.
    #[test]
    fn jump_done_then_label_slowpath_produces_no_orphan() {
        // 0: Sub vr0 vr1 vr2 L_slow   (InlineBranch terminator)
        // 1: Jump L_done              (terminator)
        // 2: Label L_slow             (slow path)
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
        //  0 (start)
        //  1 (post-Sub terminator; pos 1 = Jump, not a Label)
        //  2 (L_slow target; post-Jump-at-1 would also point here but
        //     rule 4 suppresses it because pos 2 is a Label)
        //  4 (L_done target; pos 4 is Label)
        assert_eq!(compute_leaders(&insts), vec![0, 1, 2, 4]);
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
    /// label, etc.) are modeled as `Terminator::InlineBranch` in Phase
    /// 1b-3, so post-bail positions are leaders (rule 3) AND the bail
    /// label is a leader (rule 2). The split makes the fast-path
    /// continuation its own block, preserving I1.
    #[test]
    fn guardint_is_terminator_with_bail_and_fallthrough_leaders() {
        // 0: GuardInt vr0 vr1 L_bail   (InlineBranch terminator)
        // 1: AddInt                    (fast-path continuation block)
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
        //   0 (start)
        //   1 (post-GuardInt terminator; pos 1 = AddInt, not a Label)
        //   3 (L_bail target; post-Jump-at-2 lands on pos 3 = Label,
        //      suppressed by rule 4)
        //   5 (L_done target; pos 5 = Label)
        assert_eq!(compute_leaders(&insts), vec![0, 1, 3, 5]);
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

    /// `fn id(x) { x }` — entry takes one arg, returns it. The arg is a
    /// VReg with `argument: Some(0)` set (matching `Ir::arg(0)` semantics)
    /// and becomes an entry block param.
    #[test]
    fn build_identity_function() {
        let ir = ir_with(vec![Instruction::Ret(arg_vr(0, 0))], 0, "id");
        let cfg = build_cfg(&ir).expect("identity should build");
        crate::cfg::verify::verify(&cfg).expect("identity should verify");
        assert_eq!(cfg.num_blocks(), 1);
        let entry = cfg.block(cfg.entry);
        assert_eq!(entry.params.len(), 1);
        assert_eq!(entry.body.len(), 0);
        assert!(matches!(entry.terminator, Terminator::Ret { .. }));
    }

    /// `fn add(a, b) { a + b }` — two args, AddInt, Ret. Entry has two
    /// params; body has one AddInt; Ret(VR2).
    #[test]
    fn build_add_function() {
        let ir = ir_with(
            vec![
                Instruction::AddInt(vr(2), arg_vr(0, 0), arg_vr(1, 1)),
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

    /// LoadLocal + StoreLocal translate to SlotLoad / SlotStore. Locals
    /// stay materialized as slots per I6.
    #[test]
    fn build_load_store_local() {
        let ir = ir_with(
            vec![
                Instruction::StoreLocal(Value::Local(0), arg_vr(0, 0)),
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
                Instruction::JumpIf(lbl(0), Condition::Equal, arg_vr(0, 0), arg_vr(1, 1)),
                Instruction::AddInt(vr(2), arg_vr(0, 0), arg_vr(1, 1)),
                Instruction::StoreLocal(Value::Local(0), vr(2)),
                Instruction::Jump(lbl(1)),
                Instruction::Label(lbl(0)),
                Instruction::AddInt(vr(3), arg_vr(0, 0), arg_vr(1, 1)),
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

    /// Mis-shaped operand → clear UnsupportedValueKind. After operand-
    /// shape widening, most non-Register kinds are accepted and
    /// materialized into a fresh VReg via a prepended op (Local →
    /// SlotLoad, Null/True/False → Const*). The remaining unsupported
    /// kinds — Spill, Stack — are explicitly rejected.
    #[test]
    fn mis_shaped_operand_returns_error() {
        // Ret of a Stack-positioned value isn't materializable in the
        // CFG layer (Stack is for stack-frame addressing, not a value).
        let ir = ir_with(vec![Instruction::Ret(Value::Stack(0))], 0, "ret_with_stack");
        let err = build_cfg(&ir).expect_err("Ret(Stack(_)) is malformed");
        assert!(
            matches!(err, BuildError::UnsupportedValueKind { .. }),
            "expected UnsupportedValueKind, got {:?}",
            err
        );
    }

    /// Empty IR produces an empty CfgFunction (no blocks).
    #[test]
    fn empty_ir_produces_empty_function() {
        let ir = ir_with(vec![], 0, "empty");
        let cfg = build_cfg(&ir).expect("empty should build");
        assert_eq!(cfg.num_blocks(), 0);
    }

    // ---- Phase 1c/1d tests --------------------------------------------

    /// Phase 1c: a function whose CFG would have a critical edge
    /// (`entry → join` with `entry` having 2 successors and `join` having
    /// 2 predecessors) gets a `mid` block inserted automatically, and the
    /// resulting CFG passes the I2 verifier check.
    #[test]
    fn critical_edges_are_split_automatically() {
        // 0: RegisterArgument vr0
        // 1: JumpIf L_join Eq vr0 vr0      (Branch: t=L_join, f=post=pos 2)
        // 2: AddInt vr2 vr0 vr0            (false branch body)
        // 3: Jump L_join                   (false branch terminator)
        // 4: Label L_join                  (join — 2 preds: from Branch.t
        //                                    AND from Jump-at-3)
        // 5: Ret vr0
        //
        // Before split: entry → join (critical), entry → false_block,
        //               false_block → join.
        // After split:  entry → false_block (1 pred), entry → mid (1 pred),
        //               mid → join, false_block → join. No critical edges.
        let ir = ir_with(
            vec![
                Instruction::JumpIf(lbl(1), Condition::Equal, arg_vr(0, 0), arg_vr(0, 0)),
                Instruction::AddInt(vr(2), arg_vr(0, 0), arg_vr(0, 0)),
                Instruction::Jump(lbl(1)),
                Instruction::Label(lbl(1)),
                Instruction::Ret(arg_vr(0, 0)),
            ],
            0,
            "critical_edge_diamond",
        );
        let cfg = build_cfg(&ir).expect("builds");
        crate::cfg::verify::verify(&cfg).expect("verifies after critical-edge split");
        // 3 original leaders → 3 blocks pre-split, plus one mid block on
        // the critical edge.
        assert_eq!(
            cfg.num_blocks(),
            4,
            "expected 4 blocks (3 originals + 1 mid), got {}",
            cfg.num_blocks()
        );
    }

    /// Phase 1d: a tail self-call (`TailRecurse`) rewrites at construction
    /// time to `Terminator::Jump(entry, args)`, satisfying **I8** and
    /// keeping the function's deep-recursion benchmarks SIGBUS-free.
    #[test]
    fn tail_recurse_becomes_jump_to_entry() {
        // 0: AddInt vr1 = arg(0) + arg(0)
        // 1: TailRecurse vr2 [vr1]   (rewrites to Jump(entry, [vr1]))
        let ir = ir_with(
            vec![
                Instruction::AddInt(vr(1), arg_vr(0, 0), arg_vr(0, 0)),
                Instruction::TailRecurse(vr(2), vec![vr(1)]),
            ],
            0,
            "tail_loop",
        );
        let cfg = build_cfg(&ir).expect("builds");
        crate::cfg::verify::verify(&cfg).expect("verifies");
        let entry = cfg.entry;
        let entry_block = cfg.block(entry);
        assert_eq!(entry_block.params.len(), 1, "entry should have 1 param");
        match &entry_block.terminator {
            Terminator::Jump { target, args } => {
                assert_eq!(*target, entry, "tail call must jump to entry");
                assert_eq!(
                    args.len(),
                    1,
                    "tail call must pass 1 arg matching entry's params"
                );
            }
            other => panic!("expected Jump-to-entry, got {:?}", other),
        }
    }

    /// Phase 1a fix: an unreferenced `Label` immediately after a
    /// terminator no longer suppresses rule 3, so the post-terminator
    /// position is still a leader. Without this, `JumpIf x; Label dead;
    /// ...` left the fall-through with no resolvable leader.
    #[test]
    fn unreferenced_label_after_terminator_does_not_suppress_leader() {
        // 0: Jump L_done
        // 1: Label L_orphan          (NOT referenced anywhere)
        // 2: AddInt                   (dead code)
        // 3: Label L_done
        // 4: Ret
        let insts = vec![
            Instruction::Jump(lbl(1)),
            Instruction::Label(lbl(0)),
            Instruction::AddInt(vr(0), vr(0), vr(0)),
            Instruction::Label(lbl(1)),
            Instruction::Ret(vr(0)),
        ];
        let leaders = compute_leaders(&insts);
        assert!(
            leaders.contains(&1),
            "post-Jump (pos 1) should be a leader even though L_orphan is unreferenced; got {:?}",
            leaders
        );
        assert!(
            leaders.contains(&3),
            "L_done (pos 3) should be a leader; got {:?}",
            leaders
        );
    }
}
