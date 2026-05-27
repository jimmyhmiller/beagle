//! Phase 4f-2 — translate the post-SSA allocated CFG into a flat
//! `Vec<Instruction>` that the legacy `compile_instructions` loop can
//! consume.
//!
//! Why translate to the legacy Instruction shape rather than emit
//! machine code directly? The legacy lowering of every op (tag/untag
//! shifts, calling-convention plumbing, exception-handler scaffolding,
//! continuation save/restore, inline bump alloc, all the cmov-style
//! comparisons) is ~1500 lines of well-tested code. Re-deriving any
//! one of those is exactly the kind of divergence that produced the
//! prior SSA disasters this rewrite is fixing — two paths drifting
//! apart over time as bugs get fixed in one but not the other.
//!
//! The translator here consumes the result of `Phase 4f-1`
//! (`lower_to_allocated`):
//!
//! - Every `VReg.index` is a physical register's index (the
//!   `register_from_index` value the backend expects).
//! - Block params are empty; outgoing terminator args are empty.
//! - Edge transfers have been materialized as `Op::Move` ops in either
//!   the source's body (single-succ source) or the target's body
//!   (multi-succ source after critical-edge split).
//!
//! Output: a `TranslatedIr` holding the flat Instruction list, the
//! label table, label names, and label-location map keyed by
//! Instruction index — the four state pieces `compile_instructions`
//! reads.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::cfg::{BlockId, CfgFunction, InlineBranchOp, Op, RegClass, Terminator, VReg};
use crate::common::Label;
use crate::ir::{Instruction, Value, VirtualRegister};

/// Output of [`translate`]. Layout matches the four state pieces the
/// legacy `Ir::compile_instructions` reads.
pub struct TranslatedIr {
    pub instructions: Vec<Instruction>,
    pub labels: Vec<Label>,
    pub label_names: Vec<String>,
    /// Position-in-`instructions` → Label index, matching `Ir::label_locations`.
    pub label_locations: HashMap<usize, usize>,
    /// Stack slot count for the legacy `set_max_locals` call.
    pub num_locals: usize,
}

/// Translate a post-SSA allocated CFG into legacy Instructions.
///
/// `color_to_physical` maps `(color, class)` to the physical register
/// index the backend's `register_from_index` accepts. Caller supplies
/// this because the mapping depends on the coloring's choice of pool
/// (arg regs at 0..N for entry params vs. allocator-pool indices for
/// the rest), which is a coloring-side concern that Phase 4f-2 must
/// not redo inside the translator.
pub fn translate(
    cfg: &CfgFunction,
    color_to_physical: impl Fn(u32, RegClass) -> usize,
) -> TranslatedIr {
    let order = crate::cfg::dom::reverse_postorder(cfg);
    let mut t = Translator::new(cfg, &order, color_to_physical);
    t.run();
    TranslatedIr {
        instructions: t.instructions,
        labels: t.labels,
        label_names: t.label_names,
        label_locations: t.label_locations,
        num_locals: cfg.num_slots as usize,
    }
}

struct Translator<'a, F: Fn(u32, RegClass) -> usize> {
    cfg: &'a CfgFunction,
    order: &'a [BlockId],
    /// BlockId → its allocated Label. Pre-populated before any
    /// instruction emission so a Jump can resolve its target label
    /// regardless of forward/backward direction.
    block_label: HashMap<BlockId, Label>,
    /// BlockId → position in `order` (for fall-through optimization).
    block_position: HashMap<BlockId, usize>,
    instructions: Vec<Instruction>,
    labels: Vec<Label>,
    label_names: Vec<String>,
    label_locations: HashMap<usize, usize>,
    color_to_physical: F,
}

impl<'a, F: Fn(u32, RegClass) -> usize> Translator<'a, F> {
    fn new(cfg: &'a CfgFunction, order: &'a [BlockId], color_to_physical: F) -> Self {
        let mut t = Translator {
            cfg,
            order,
            block_label: HashMap::new(),
            block_position: HashMap::new(),
            instructions: Vec::new(),
            labels: Vec::new(),
            label_names: Vec::new(),
            label_locations: HashMap::new(),
            color_to_physical,
        };
        // Allocate labels and record positions up front so terminator
        // translation can resolve any block id.
        for (pos, &bid) in order.iter().enumerate() {
            let label = t.alloc_label(&format!("bb{}", bid.0));
            t.block_label.insert(bid, label);
            t.block_position.insert(bid, pos);
        }
        t
    }

    fn alloc_label(&mut self, name: &str) -> Label {
        let label = Label {
            index: self.labels.len(),
        };
        self.labels.push(label);
        self.label_names.push(name.to_string());
        label
    }

    /// Emit `Instruction::Label(L)` AND register its position so
    /// `compile_instructions` writes it to the backend (its loop checks
    /// `label_locations` keyed by instruction-position, not by walking
    /// `Instruction::Label`).
    fn emit_label(&mut self, label: Label) {
        let pos = self.instructions.len();
        self.instructions.push(Instruction::Label(label));
        self.label_locations.insert(pos, label.index);
    }

    fn run(&mut self) {
        for (i, &bid) in self.order.iter().enumerate() {
            let label = self.block_label[&bid];
            self.emit_label(label);
            let block = self.cfg.block(bid);
            for op in &block.body {
                self.emit_op(op);
            }
            let next_in_order = self.order.get(i + 1).copied();
            self.emit_terminator(&block.terminator, next_in_order);
        }
    }

    /// VReg → legacy Value::Register with `index` = physical register
    /// index. `argument`/`volatile`/`is_physical` are set the way the
    /// legacy regalloc would emit a post-allocation physical register
    /// (matches the `physical()` helper in `LinearScan`).
    fn reg(&self, v: VReg) -> Value {
        let index = (self.color_to_physical)(v.index, v.class);
        Value::Register(VirtualRegister {
            argument: None,
            index,
            volatile: true,
            is_physical: true,
        })
    }

    fn emit_op(&mut self, op: &Op) {
        use Instruction as I;
        match op {
            // ---- Moves & constants ----
            Op::Move { dst, src } => self
                .instructions
                .push(I::Assign(self.reg(*dst), self.reg(*src))),
            Op::ConstTaggedInt { dst, value } => self.instructions.push(I::Assign(
                self.reg(*dst),
                Value::TaggedConstant(*value as isize),
            )),

            // ---- Integer arithmetic (no bail) ----
            Op::AddInt { dst, lhs, rhs } => {
                self.instructions
                    .push(I::AddInt(self.reg(*dst), self.reg(*lhs), self.reg(*rhs)))
            }

            // ---- Comparisons ----
            Op::Compare {
                dst,
                lhs,
                rhs,
                cond,
            } => self.instructions.push(I::Compare(
                self.reg(*dst),
                self.reg(*lhs),
                self.reg(*rhs),
                *cond,
            )),

            // ---- Slots (locals) ----
            Op::SlotLoad { dst, slot } => self
                .instructions
                .push(I::LoadLocal(self.reg(*dst), Value::Local(slot.0 as usize))),
            Op::SlotStore { slot, src } => self
                .instructions
                .push(I::StoreLocal(Value::Local(slot.0 as usize), self.reg(*src))),

            // Everything else surfaces loudly so 4f-2d work is
            // self-discovering when a real corpus run hits it.
            other => todo!("emit_legacy: Op::{:?} not yet translated", op_tag(other)),
        }
    }

    fn emit_terminator(&mut self, term: &Terminator, next_in_order: Option<BlockId>) {
        use Instruction as I;
        match term {
            Terminator::Ret { value } => {
                self.instructions.push(I::Ret(self.reg(*value)));
            }
            Terminator::Jump { target, .. } => {
                // Fall-through if target is the next block in linearization.
                if Some(*target) == next_in_order {
                    return;
                }
                let label = self.block_label[target];
                self.instructions.push(I::Jump(label));
            }
            Terminator::Branch {
                cond,
                lhs,
                rhs,
                t_target,
                f_target,
                ..
            } => {
                let lhs_v = self.reg(*lhs);
                let rhs_v = self.reg(*rhs);
                let t_label = self.block_label[t_target];
                self.instructions
                    .push(I::JumpIf(t_label, *cond, lhs_v, rhs_v));
                // Fall-through to f_target if it's next; otherwise emit Jump.
                if Some(*f_target) != next_in_order {
                    let f_label = self.block_label[f_target];
                    self.instructions.push(I::Jump(f_label));
                }
            }
            Terminator::InlineBranch { op, .. } => {
                todo!(
                    "emit_legacy: Terminator::InlineBranch({:?}) not yet translated",
                    inline_op_tag(op)
                );
            }
            Terminator::Throw { .. } => {
                todo!("emit_legacy: Terminator::Throw not yet translated");
            }
            Terminator::Unreachable => {
                // No instruction; verifier should have caught any
                // surviving Unreachable in a reachable block before us.
            }
        }
    }
}

fn op_tag(op: &Op) -> &'static str {
    match op {
        Op::SlotLoad { .. } => "SlotLoad",
        Op::SlotStore { .. } => "SlotStore",
        Op::Move { .. } => "Move",
        Op::ConstTaggedInt { .. } => "ConstTaggedInt",
        Op::ConstStringPtr { .. } => "ConstStringPtr",
        Op::ConstKeywordPtr { .. } => "ConstKeywordPtr",
        Op::ConstFunctionId { .. } => "ConstFunctionId",
        Op::ConstPointer { .. } => "ConstPointer",
        Op::ConstRawValue { .. } => "ConstRawValue",
        Op::ConstTrue { .. } => "ConstTrue",
        Op::ConstFalse { .. } => "ConstFalse",
        Op::ConstNull { .. } => "ConstNull",
        Op::ConstLabelAddress { .. } => "ConstLabelAddress",
        Op::AddInt { .. } => "AddInt",
        Op::Compare { .. } => "Compare",
        Op::CompareFloat { .. } => "CompareFloat",
        Op::AddFloat { .. } => "AddFloat",
        Op::SubFloat { .. } => "SubFloat",
        Op::MulFloat { .. } => "MulFloat",
        Op::DivFloat { .. } => "DivFloat",
        Op::IntToFloat { .. } => "IntToFloat",
        Op::FRoundToZero { .. } => "FRoundToZero",
        Op::FmovGpToFp { .. } => "FmovGpToFp",
        Op::FmovFpToGp { .. } => "FmovFpToGp",
        Op::Tag { .. } => "Tag",
        Op::Untag { .. } => "Untag",
        Op::GetTag { .. } => "GetTag",
        Op::And { .. } => "And",
        Op::Or { .. } => "Or",
        Op::Xor { .. } => "Xor",
        Op::AndImm { .. } => "AndImm",
        Op::ShiftRightImmRaw { .. } => "ShiftRightImmRaw",
        Op::HeapLoad { .. } => "HeapLoad",
        Op::HeapLoadReg { .. } => "HeapLoadReg",
        Op::HeapLoadByteReg { .. } => "HeapLoadByteReg",
        Op::HeapStore { .. } => "HeapStore",
        Op::HeapStoreOffset { .. } => "HeapStoreOffset",
        Op::HeapStoreOffsetReg { .. } => "HeapStoreOffsetReg",
        Op::HeapStoreByteOffsetReg { .. } => "HeapStoreByteOffsetReg",
        Op::HeapStoreByteOffsetMasked { .. } => "HeapStoreByteOffsetMasked",
        Op::AtomicLoad { .. } => "AtomicLoad",
        Op::AtomicStore { .. } => "AtomicStore",
        Op::CompareAndSwap { .. } => "CompareAndSwap",
        Op::StoreFloatConstant { .. } => "StoreFloatConstant",
        Op::PushStack { .. } => "PushStack",
        Op::PopStack { .. } => "PopStack",
        Op::GetStackPointer { .. } => "GetStackPointer",
        Op::GetStackPointerImm { .. } => "GetStackPointerImm",
        Op::GetFramePointer { .. } => "GetFramePointer",
        Op::CurrentStackPosition { .. } => "CurrentStackPosition",
        Op::ReadArgCount { .. } => "ReadArgCount",
        Op::ExtendLifetime { .. } => "ExtendLifetime",
        Op::FeedbackOr { .. } => "FeedbackOr",
        Op::TierUpCheck { .. } => "TierUpCheck",
        Op::Call { .. } => "Call",
        Op::Recurse { .. } => "Recurse",
        Op::PushExceptionHandler { .. } => "PushExceptionHandler",
        Op::PushResumableExceptionHandler { .. } => "PushResumableExceptionHandler",
        Op::PopExceptionHandler { .. } => "PopExceptionHandler",
        Op::PopExceptionHandlerById { .. } => "PopExceptionHandlerById",
        Op::PushPromptHandler { .. } => "PushPromptHandler",
        Op::PopPromptHandler { .. } => "PopPromptHandler",
        Op::PushPromptTag { .. } => "PushPromptTag",
        Op::CaptureContinuation { .. } => "CaptureContinuation",
        Op::CaptureContinuationTagged { .. } => "CaptureContinuationTagged",
        Op::PerformEffect { .. } => "PerformEffect",
        Op::ReturnFromShift { .. } => "ReturnFromShift",
        Op::Breakpoint => "Breakpoint",
        Op::RecordGcSafepoint => "RecordGcSafepoint",
    }
}

fn inline_op_tag(op: &InlineBranchOp) -> &'static str {
    match op {
        InlineBranchOp::SubChecked { .. } => "SubChecked",
        InlineBranchOp::MulChecked { .. } => "MulChecked",
        InlineBranchOp::DivChecked { .. } => "DivChecked",
        InlineBranchOp::ModuloChecked { .. } => "ModuloChecked",
        InlineBranchOp::ShiftLeftChecked { .. } => "ShiftLeftChecked",
        InlineBranchOp::ShiftRightChecked { .. } => "ShiftRightChecked",
        InlineBranchOp::ShiftRightZeroChecked { .. } => "ShiftRightZeroChecked",
        InlineBranchOp::ShiftRightImmChecked { .. } => "ShiftRightImmChecked",
        InlineBranchOp::GuardInt { .. } => "GuardInt",
        InlineBranchOp::GuardFloat { .. } => "GuardFloat",
        InlineBranchOp::InlineBumpAllocate { .. } => "InlineBumpAllocate",
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CfgFunction, Op, RegClass, Terminator};

    /// Trivial identity test color mapping: pretend pool index = color.
    /// Real coloring chooses a physical-register pool per class; here we
    /// just verify the structural translation works.
    fn identity_color(v: u32, _class: RegClass) -> usize {
        v as usize
    }

    /// Single block with a Move and a Ret. Verifies block ordering,
    /// label emission, Op::Move → Instruction::Assign, and
    /// Terminator::Ret → Instruction::Ret.
    #[test]
    fn single_block_move_and_ret() {
        let mut f = CfgFunction::new(Some("id".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let x = f.new_vreg(RegClass::Gp);
        let y = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::Move { dst: y, src: x });
        f.block_mut(entry).terminator = Terminator::Ret { value: y };

        let t = translate(&f, identity_color);

        // bb0 label + Assign(y, x) + Ret(y) = 3 instructions.
        assert_eq!(t.instructions.len(), 3);
        assert!(matches!(t.instructions[0], Instruction::Label(_)));
        match &t.instructions[1] {
            Instruction::Assign(Value::Register(d), Value::Register(s)) => {
                assert_eq!(d.index, y.index as usize);
                assert_eq!(s.index, x.index as usize);
            }
            other => panic!("expected Assign, got {:?}", other),
        }
        match &t.instructions[2] {
            Instruction::Ret(Value::Register(v)) => {
                assert_eq!(v.index, y.index as usize);
            }
            other => panic!("expected Ret, got {:?}", other),
        }
        // Label registered for compile_instructions to write.
        assert_eq!(t.label_locations.get(&0), Some(&0));
    }

    /// Two blocks: entry computes a sum, jumps to tail which returns.
    /// Verifies AddInt, Jump fall-through (since tail is next in RPO),
    /// and that no spurious Jump is emitted.
    #[test]
    fn jump_fall_through() {
        let mut f = CfgFunction::new(Some("add".into()), 0);
        let entry = f.new_block();
        let tail = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let b = f.new_vreg(RegClass::Gp);
        let sum = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: sum,
            lhs: a,
            rhs: b,
        });
        f.block_mut(entry).terminator = Terminator::Jump {
            target: tail,
            args: vec![],
        };
        f.block_mut(tail).terminator = Terminator::Ret { value: sum };
        f.block_mut(tail).predecessors.push(entry);

        let t = translate(&f, identity_color);

        // bb0_label + AddInt + bb1_label + Ret  (no Jump because fall-through)
        assert_eq!(t.instructions.len(), 4, "got: {:?}", t.instructions);
        assert!(matches!(t.instructions[0], Instruction::Label(_)));
        assert!(matches!(t.instructions[1], Instruction::AddInt(_, _, _)));
        assert!(matches!(t.instructions[2], Instruction::Label(_)));
        assert!(matches!(t.instructions[3], Instruction::Ret(_)));
    }

    /// Branch with non-adjacent t_target emits a JumpIf + Jump.
    #[test]
    fn branch_emits_jumpif() {
        let mut f = CfgFunction::new(Some("br".into()), 0);
        let entry = f.new_block();
        let t_block = f.new_block();
        let f_block = f.new_block();
        f.entry = entry;
        let cond = f.new_vreg(RegClass::Gp);
        let v = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).terminator = Terminator::Branch {
            cond: crate::ir::Condition::Equal,
            lhs: cond,
            rhs: cond,
            t_target: t_block,
            t_args: vec![],
            f_target: f_block,
            f_args: vec![],
        };
        f.block_mut(t_block).terminator = Terminator::Ret { value: v };
        f.block_mut(f_block).terminator = Terminator::Ret { value: v };
        f.block_mut(t_block).predecessors.push(entry);
        f.block_mut(f_block).predecessors.push(entry);

        let t = translate(&f, identity_color);

        // RPO is entry, then one of {t_block, f_block} first based on DFS order;
        // successors are walked in source order so t_block comes first → that
        // becomes the fall-through and the other gets the explicit JumpIf.
        // What we strictly assert: at least one JumpIf exists, and the block
        // count is right.
        let jumpif_count = t
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::JumpIf(..)))
            .count();
        assert_eq!(jumpif_count, 1, "exactly one JumpIf for a Branch");
    }
}
