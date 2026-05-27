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

use crate::common::Label;
use crate::ir::Instruction;

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
}
