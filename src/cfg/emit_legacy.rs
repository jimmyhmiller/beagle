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

use std::collections::{HashMap, HashSet};

use crate::backend::CodegenBackend;
use crate::cfg::{
    BlockId, CallTarget, CfgFunction, ClobberSet, InlineBranchOp, Op, RegClass, SlotId, TagSource,
    Terminator, VReg,
};
use crate::common::Label;
use crate::ir::{Condition, Instruction, Ir, Value, VirtualRegister};

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
///
/// Returns `Err(SsaCompileError::UnsupportedOp(...))` when an op
/// hasn't been translated yet. The caller is expected to fall back
/// to the legacy pipeline on this error. Returning Err instead of
/// panicking lets the gate be opt-in without blowing up the compile
/// thread on every untranslated op encountered.
pub fn translate(
    cfg: &CfgFunction,
    color_to_physical: impl Fn(u32, RegClass) -> usize,
) -> Result<TranslatedIr, SsaCompileError> {
    let order = crate::cfg::dom::reverse_postorder(cfg);
    let mut t = Translator::new(cfg, &order, color_to_physical);
    t.run()?;
    Ok(TranslatedIr {
        instructions: t.instructions,
        labels: t.labels,
        label_names: t.label_names,
        label_locations: t.label_locations,
        num_locals: cfg.num_slots as usize,
    })
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
    /// Empty jump-only blocks that can be bypassed during final linear
    /// translation. This is a code-layout optimization only; the CFG keeps
    /// those blocks so earlier phases preserve the no-critical-edge invariant.
    jump_alias: HashMap<BlockId, BlockId>,
    instructions: Vec<Instruction>,
    labels: Vec<Label>,
    label_names: Vec<String>,
    label_locations: HashMap<usize, usize>,
    /// Per-block cache for post-allocation slot reload forwarding:
    /// after `SlotStore S <- v`, a later `SlotLoad S -> dst` can become
    /// a physical-register move while v's assigned physical register is
    /// still unmodified and no side-effecting op has intervened.
    slot_forward: HashMap<SlotId, VReg>,
    color_to_physical: F,
}

impl<'a, F: Fn(u32, RegClass) -> usize> Translator<'a, F> {
    fn new(cfg: &'a CfgFunction, order: &'a [BlockId], color_to_physical: F) -> Self {
        let mut t = Translator {
            cfg,
            order,
            block_label: HashMap::new(),
            block_position: HashMap::new(),
            jump_alias: compute_jump_aliases(cfg),
            instructions: Vec::new(),
            labels: Vec::new(),
            label_names: Vec::new(),
            label_locations: HashMap::new(),
            slot_forward: HashMap::new(),
            color_to_physical,
        };
        // Allocate labels for ALL blocks (not just RPO-reachable ones):
        // exception handler / continuation resume / prompt abort blocks
        // are referenced by `Op::PushExceptionHandler`, `Op::Throw`,
        // `Op::CaptureContinuation`, etc. — they're not normal CFG
        // successors so `reverse_postorder` doesn't visit them, but
        // their labels still need to exist for the LoadLabelAddress
        // emit at the call site (and for the runtime to jump to them
        // on throw/resume). Record position only for blocks in `order`
        // so the fall-through optimization in `emit_terminator` works.
        for bid_idx in 0..cfg.num_blocks() {
            let bid = BlockId(bid_idx as u32);
            let label = t.alloc_label(&format!("bb{}", bid.0));
            t.block_label.insert(bid, label);
        }
        for (pos, &bid) in order.iter().enumerate() {
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

    fn run(&mut self) -> Result<(), SsaCompileError> {
        // Phase A: emit blocks in RPO so fall-through optimization
        // applies to the common path.
        for (i, &bid) in self.order.iter().enumerate() {
            if self.jump_alias.contains_key(&bid) {
                continue;
            }
            let label = self.block_label[&bid];
            self.emit_label(label);
            let block = self.cfg.block(bid);
            self.slot_forward.clear();
            for op in &block.body {
                self.emit_op(op)?;
            }
            let next_in_order = self.next_emitted_block(i);
            self.emit_terminator(&block.terminator, next_in_order)?;
        }
        // Phase B: emit any blocks that aren't in RPO. These are
        // exception handler / continuation resume / prompt abort
        // blocks referenced by `Op::PushExceptionHandler` etc. — the
        // runtime jumps to them via the handler stack, not via normal
        // CFG edges, so RPO from entry skips them. Their labels still
        // need actual machine-code positions. Fall-through opt does
        // not apply here (every terminator emits its explicit Jump).
        let in_order: HashSet<BlockId> = self.order.iter().copied().collect();
        for bid_idx in 0..self.cfg.num_blocks() {
            let bid = BlockId(bid_idx as u32);
            if in_order.contains(&bid) {
                continue;
            }
            if self.jump_alias.contains_key(&bid) {
                continue;
            }
            // Skip blocks whose terminator is `Unreachable` AND have
            // an empty body — those are RPO-dead placeholders left by
            // `dce_unreachable_blocks` and have nothing to emit.
            let block = self.cfg.block(bid);
            if matches!(block.terminator, Terminator::Unreachable) && block.body.is_empty() {
                continue;
            }
            let label = self.block_label[&bid];
            self.emit_label(label);
            self.slot_forward.clear();
            for op in &block.body {
                self.emit_op(op)?;
            }
            self.emit_terminator(&block.terminator, None)?;
        }
        Ok(())
    }

    fn next_emitted_block(&self, pos: usize) -> Option<BlockId> {
        self.order
            .iter()
            .skip(pos + 1)
            .copied()
            .find(|bid| !self.jump_alias.contains_key(bid))
    }

    fn resolve_jump_alias(&self, mut target: BlockId) -> BlockId {
        let mut seen = HashSet::new();
        while let Some(&next) = self.jump_alias.get(&target) {
            if !seen.insert(target) {
                break;
            }
            target = next;
        }
        target
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

    fn physical_key(&self, v: VReg) -> (RegClass, usize) {
        (v.class, (self.color_to_physical)(v.index, v.class))
    }

    fn same_physical(&self, a: VReg, b: VReg) -> bool {
        self.physical_key(a) == self.physical_key(b)
    }

    fn invalidate_slot_forward_defs(&mut self, defs: &[VReg]) {
        if defs.is_empty() || self.slot_forward.is_empty() {
            return;
        }
        let def_keys: Vec<(RegClass, usize)> = defs.iter().map(|d| self.physical_key(*d)).collect();
        let color_to_physical = &self.color_to_physical;
        self.slot_forward.retain(|_, src| {
            let src_key = (src.class, color_to_physical(src.index, src.class));
            !def_keys.contains(&src_key)
        });
    }

    fn clear_slot_forward_after_side_effect(&mut self, op: &Op) {
        if op.has_side_effect() {
            self.slot_forward.clear();
        }
    }

    fn emit_op(&mut self, op: &Op) -> Result<(), SsaCompileError> {
        use Instruction as I;
        match op {
            Op::SlotLoad { dst, slot } => {
                if slot.is_unscanned() {
                    return Err(SsaCompileError::UnsupportedOp(format!(
                        "SlotLoad from unscanned slot {:?}: backend addressing \
                         for the non-scanned FP/raw region is not implemented \
                         yet (Phase 1 backend support pending)",
                        slot
                    )));
                }

                if let Some(src) = self.slot_forward.get(slot).copied()
                    && src.class == dst.class
                {
                    if !self.same_physical(*dst, src) {
                        self.instructions
                            .push(I::Assign(self.reg(*dst), self.reg(src)));
                        self.invalidate_slot_forward_defs(&[*dst]);
                    }
                } else {
                    self.instructions.push(I::LoadLocal(
                        self.reg(*dst),
                        Value::Local(slot.region_index() as usize),
                    ));
                    self.invalidate_slot_forward_defs(&[*dst]);
                }
                self.slot_forward.insert(*slot, *dst);
                return Ok(());
            }
            Op::SlotStore { slot, src } => {
                if slot.is_unscanned() {
                    return Err(SsaCompileError::UnsupportedOp(format!(
                        "SlotStore to unscanned slot {:?}: backend addressing \
                         for the non-scanned FP/raw region is not implemented \
                         yet (Phase 1 backend support pending)",
                        slot
                    )));
                }
                self.instructions.push(I::StoreLocal(
                    Value::Local(slot.region_index() as usize),
                    self.reg(*src),
                ));
                self.slot_forward.insert(*slot, *src);
                return Ok(());
            }
            _ => {}
        }

        match op {
            // ---- Moves & constants ----
            // An FP-class move must use an FP register move (`fmov Dd, Dn`);
            // a plain `Assign` lowers to an integer `mov` of the wrong (GP)
            // registers, silently corrupting the value. FP block-param
            // transfers on edges flow through here (Op::Move).
            Op::Move { dst, src } if dst.class == RegClass::Fp => self
                .instructions
                .push(I::MoveFloat(self.reg(*dst), self.reg(*src))),
            Op::Move { dst, src } => self
                .instructions
                .push(I::Assign(self.reg(*dst), self.reg(*src))),
            Op::ConstTaggedInt { dst, value } => self.instructions.push(I::Assign(
                self.reg(*dst),
                Value::TaggedConstant(*value as isize),
            )),
            Op::ConstStringPtr { dst, ptr } => self
                .instructions
                .push(I::Assign(self.reg(*dst), Value::StringConstantPtr(*ptr))),
            Op::ConstKeywordPtr { dst, ptr } => self
                .instructions
                .push(I::Assign(self.reg(*dst), Value::KeywordConstantPtr(*ptr))),
            Op::ConstFunctionId { dst, function_id } => self
                .instructions
                .push(I::Assign(self.reg(*dst), Value::Function(*function_id))),
            Op::ConstPointer { dst, ptr } => self
                .instructions
                .push(I::Assign(self.reg(*dst), Value::Pointer(*ptr))),
            Op::ConstRawValue { dst, value } => self
                .instructions
                .push(I::Assign(self.reg(*dst), Value::RawValue(*value as usize))),
            Op::ConstTrue { dst } => self.instructions.push(I::LoadTrue(self.reg(*dst))),
            Op::ConstFalse { dst } => self.instructions.push(I::LoadFalse(self.reg(*dst))),
            Op::ConstNull { dst } => self
                .instructions
                .push(I::Assign(self.reg(*dst), Value::Null)),
            Op::ConstLabelAddress { dst, target } => {
                let label = self.block_label[target];
                self.instructions
                    .push(I::LoadLabelAddress(self.reg(*dst), label));
            }

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
            Op::CompareFloat {
                dst,
                lhs,
                rhs,
                cond,
            } => self.instructions.push(I::CompareFloat(
                self.reg(*dst),
                self.reg(*lhs),
                self.reg(*rhs),
                *cond,
            )),

            // ---- Floating-point arithmetic ----
            Op::AddFloat { dst, lhs, rhs } => {
                self.instructions
                    .push(I::AddFloat(self.reg(*dst), self.reg(*lhs), self.reg(*rhs)))
            }
            Op::SubFloat { dst, lhs, rhs } => {
                self.instructions
                    .push(I::SubFloat(self.reg(*dst), self.reg(*lhs), self.reg(*rhs)))
            }
            Op::MulFloat { dst, lhs, rhs } => {
                self.instructions
                    .push(I::MulFloat(self.reg(*dst), self.reg(*lhs), self.reg(*rhs)))
            }
            Op::DivFloat { dst, lhs, rhs } => {
                self.instructions
                    .push(I::DivFloat(self.reg(*dst), self.reg(*lhs), self.reg(*rhs)))
            }

            // ---- Conversions / bit-moves between classes ----
            Op::IntToFloat { dst, src } => self
                .instructions
                .push(I::IntToFloat(self.reg(*dst), self.reg(*src))),
            Op::FRoundToZero { dst, src } => self
                .instructions
                .push(I::FRoundToZero(self.reg(*dst), self.reg(*src))),
            Op::FmovGpToFp { dst, src } => self
                .instructions
                .push(I::FmovGeneralToFloat(self.reg(*dst), self.reg(*src))),
            Op::FmovFpToGp { dst, src } => self
                .instructions
                .push(I::FmovFloatToGeneral(self.reg(*dst), self.reg(*src))),

            // ---- Tag bit manipulation ----
            Op::Tag {
                dst,
                src,
                tag_source,
            } => {
                let tag_v = match tag_source {
                    TagSource::Register(t) => self.reg(*t),
                    TagSource::Bits(bits) => Value::RawValue(*bits as usize),
                };
                self.instructions
                    .push(I::Tag(self.reg(*dst), self.reg(*src), tag_v));
            }
            Op::Untag { dst, src } => self
                .instructions
                .push(I::Untag(self.reg(*dst), self.reg(*src))),
            Op::GetTag { dst, src } => self
                .instructions
                .push(I::GetTag(self.reg(*dst), self.reg(*src))),

            // ---- Bit ops ----
            Op::And { dst, lhs, rhs } => {
                self.instructions
                    .push(I::And(self.reg(*dst), self.reg(*lhs), self.reg(*rhs)))
            }
            Op::Or { dst, lhs, rhs } => {
                self.instructions
                    .push(I::Or(self.reg(*dst), self.reg(*lhs), self.reg(*rhs)))
            }
            Op::Xor { dst, lhs, rhs } => {
                self.instructions
                    .push(I::Xor(self.reg(*dst), self.reg(*lhs), self.reg(*rhs)))
            }
            Op::AndImm { dst, src, imm } => {
                self.instructions
                    .push(I::AndImm(self.reg(*dst), self.reg(*src), *imm))
            }
            Op::ShiftRightImmRaw { dst, src, imm } => {
                self.instructions
                    .push(I::ShiftRightImmRaw(self.reg(*dst), self.reg(*src), *imm))
            }

            // ---- Slots (locals) ----
            // Root slots map to `Value::Local(index)` (GC-scanned). The
            // unscanned region (FP spills / raw scratch) has no backend
            // addressing yet; emit a clear hard error rather than a
            // bogus `Local(UNSCANNED_SLOT_BASE + i)` index. Nothing
            // produces unscanned slots on the wired path today (the
            // spiller isn't in `compile_via_ssa`), so this is the
            // forward guard for when Phase 1's backend support lands.
            Op::SlotLoad { dst, slot } => {
                if slot.is_unscanned() {
                    return Err(SsaCompileError::UnsupportedOp(format!(
                        "SlotLoad from unscanned slot {:?}: backend addressing \
                         for the non-scanned FP/raw region is not implemented \
                         yet (Phase 1 backend support pending)",
                        slot
                    )));
                }
                self.instructions.push(I::LoadLocal(
                    self.reg(*dst),
                    Value::Local(slot.region_index() as usize),
                ));
            }
            Op::SlotStore { slot, src } => {
                if slot.is_unscanned() {
                    return Err(SsaCompileError::UnsupportedOp(format!(
                        "SlotStore to unscanned slot {:?}: backend addressing \
                         for the non-scanned FP/raw region is not implemented \
                         yet (Phase 1 backend support pending)",
                        slot
                    )));
                }
                self.instructions.push(I::StoreLocal(
                    Value::Local(slot.region_index() as usize),
                    self.reg(*src),
                ));
            }

            // ---- Heap memory ----
            Op::HeapLoad { dst, base, offset } => {
                self.instructions
                    .push(I::HeapLoad(self.reg(*dst), self.reg(*base), *offset))
            }
            Op::HeapLoadReg { dst, base, offset } => self.instructions.push(I::HeapLoadReg(
                self.reg(*dst),
                self.reg(*base),
                self.reg(*offset),
            )),
            Op::HeapLoadByteReg { dst, base, offset } => self.instructions.push(
                I::HeapLoadByteReg(self.reg(*dst), self.reg(*base), self.reg(*offset)),
            ),
            Op::HeapStore { addr, src } => self
                .instructions
                .push(I::HeapStore(self.reg(*addr), self.reg(*src))),
            Op::HeapStoreOffset { base, src, offset } => self
                .instructions
                .push(I::HeapStoreOffset(self.reg(*base), self.reg(*src), *offset)),
            Op::HeapStoreOffsetReg { base, src, offset } => self.instructions.push(
                I::HeapStoreOffsetReg(self.reg(*base), self.reg(*src), self.reg(*offset)),
            ),
            Op::HeapStoreByteOffsetReg { base, src, offset } => self.instructions.push(
                I::HeapStoreByteOffsetReg(self.reg(*base), self.reg(*src), self.reg(*offset)),
            ),
            Op::HeapStoreByteOffsetMasked {
                ptr,
                val,
                temp1,
                temp2,
                offset,
                byte_offset,
                mask,
            } => self.instructions.push(I::HeapStoreByteOffsetMasked(
                self.reg(*ptr),
                self.reg(*val),
                self.reg(*temp1),
                self.reg(*temp2),
                *offset,
                *byte_offset,
                *mask,
            )),

            // ---- Atomic ----
            Op::AtomicLoad { dst, src } => self
                .instructions
                .push(I::AtomicLoad(self.reg(*dst), self.reg(*src))),
            Op::AtomicStore { addr, src } => self
                .instructions
                .push(I::AtomicStore(self.reg(*addr), self.reg(*src))),
            Op::CompareAndSwap {
                addr,
                expected,
                new,
            } => self.instructions.push(I::CompareAndSwap(
                self.reg(*addr),
                self.reg(*expected),
                self.reg(*new),
            )),

            // ---- Float storage ----
            Op::StoreFloatConstant {
                dest,
                temp,
                value_text,
            } => self.instructions.push(I::StoreFloat(
                self.reg(*dest),
                self.reg(*temp),
                value_text.clone(),
            )),

            // ---- Stack ----
            Op::PushStack { src } => self.instructions.push(I::PushStack(self.reg(*src))),
            Op::PopStack { dst } => self.instructions.push(I::PopStack(self.reg(*dst))),
            Op::GetStackPointer { dst, offset } => self
                .instructions
                .push(I::GetStackPointer(self.reg(*dst), self.reg(*offset))),
            Op::GetStackPointerImm { dst, offset } => self
                .instructions
                .push(I::GetStackPointerImm(self.reg(*dst), *offset)),
            Op::GetFramePointer { dst } => {
                self.instructions.push(I::GetFramePointer(self.reg(*dst)))
            }
            Op::CurrentStackPosition { dst } => self
                .instructions
                .push(I::CurrentStackPosition(self.reg(*dst))),

            // ---- Variadic plumbing ----
            Op::ReadArgCount { dst } => self.instructions.push(I::ReadArgCount(self.reg(*dst))),

            // ---- Misc no-VReg / lifetime markers ----
            Op::ExtendLifetime { src } => self.instructions.push(I::ExtendLifeTime(self.reg(*src))),
            Op::Breakpoint => self.instructions.push(I::Breakpoint),
            Op::RecordGcSafepoint => self.instructions.push(I::RecordGcSafepoint),
            Op::FeedbackOr { slot_addr, bits } => {
                self.instructions.push(I::FeedbackOr(*slot_addr, *bits))
            }
            Op::TierUpCheck {
                counter_addr,
                name_c_str_ptr,
                trampoline_fn_ptr,
            } => self.instructions.push(I::TierUpCheck(
                *counter_addr,
                *name_c_str_ptr,
                *trampoline_fn_ptr,
            )),

            // ---- Exception handlers ----
            Op::PushExceptionHandler {
                handler,
                result_slot,
                builtin_fn_ptr,
            } => self.instructions.push(I::PushExceptionHandler(
                self.block_label[handler],
                Value::Local(result_slot.0 as usize),
                *builtin_fn_ptr,
            )),
            Op::PushResumableExceptionHandler {
                dst,
                catch_block,
                exception_slot,
                resume_slot,
                builtin_fn_ptr,
            } => self.instructions.push(I::PushResumableExceptionHandler(
                self.reg(*dst),
                self.block_label[catch_block],
                Value::Local(exception_slot.0 as usize),
                Value::Local(resume_slot.0 as usize),
                *builtin_fn_ptr,
            )),
            Op::PopExceptionHandler { builtin_fn_ptr } => {
                self.instructions
                    .push(I::PopExceptionHandler(*builtin_fn_ptr));
            }
            Op::PopExceptionHandlerById {
                handler_id,
                builtin_fn_ptr,
            } => self.instructions.push(I::PopExceptionHandlerById(
                self.reg(*handler_id),
                *builtin_fn_ptr,
            )),

            // ---- Delimited continuations / prompts ----
            Op::PushPromptHandler {
                handler,
                result_slot,
                builtin_fn_ptr,
            } => self.instructions.push(I::PushPromptHandler(
                self.block_label[handler],
                Value::Local(result_slot.0 as usize),
                *builtin_fn_ptr,
            )),
            Op::PopPromptHandler {
                result,
                builtin_fn_ptr,
            } => self
                .instructions
                .push(I::PopPromptHandler(self.reg(*result), *builtin_fn_ptr)),
            Op::PushPromptTag {
                tag,
                abort_block,
                result_slot,
                builtin_fn_ptr,
            } => self.instructions.push(I::PushPromptTag(
                self.reg(*tag),
                self.block_label[abort_block],
                Value::Local(result_slot.0 as usize),
                *builtin_fn_ptr,
            )),
            Op::CaptureContinuation {
                dst,
                resume_block,
                result_slot,
                builtin_fn_ptr,
            } => self.instructions.push(I::CaptureContinuation(
                self.reg(*dst),
                self.block_label[resume_block],
                result_slot.0 as usize,
                *builtin_fn_ptr,
            )),
            Op::CaptureContinuationTagged {
                dst,
                resume_block,
                result_slot,
                builtin_fn_ptr,
                tag,
            } => self.instructions.push(I::CaptureContinuationTagged(
                self.reg(*dst),
                self.block_label[resume_block],
                result_slot.0 as usize,
                *builtin_fn_ptr,
                self.reg(*tag),
            )),

            // ---- Algebraic effects ----
            Op::PerformEffect {
                handler,
                enum_type,
                op_value,
                resume_block,
                result_slot,
                builtin_fn_ptr,
            } => self.instructions.push(I::PerformEffect(
                self.reg(*handler),
                self.reg(*enum_type),
                self.reg(*op_value),
                self.block_label[resume_block],
                result_slot.0 as usize,
                *builtin_fn_ptr,
            )),
            Op::ReturnFromShift {
                value,
                cont_ptr,
                builtin_fn_ptr,
            } => self.instructions.push(I::ReturnFromShift(
                self.reg(*value),
                self.reg(*cont_ptr),
                *builtin_fn_ptr,
            )),

            // ---- Calls ----
            Op::Call {
                dst,
                target,
                args,
                is_builtin,
                clobbers: ClobberSet::AllCallerSaved,
            } => {
                let fn_value = match target {
                    CallTarget::Register(vr) => self.reg(*vr),
                    CallTarget::FunctionId(id) => Value::Function(*id),
                    CallTarget::Pointer(p) => Value::Pointer(*p),
                    CallTarget::Raw(v) => Value::RawValue(*v as usize),
                };
                let arg_values: Vec<Value> = args.iter().map(|v| self.reg(*v)).collect();
                self.instructions
                    .push(I::Call(self.reg(*dst), fn_value, arg_values, *is_builtin));
            }
            Op::Recurse {
                dst,
                args,
                clobbers: ClobberSet::AllCallerSaved,
            } => {
                let arg_values: Vec<Value> = args.iter().map(|v| self.reg(*v)).collect();
                self.instructions
                    .push(I::Recurse(self.reg(*dst), arg_values));
            }
        }
        let defs = op.defs();
        self.invalidate_slot_forward_defs(&defs);
        self.clear_slot_forward_after_side_effect(op);
        Ok(())
    }

    fn emit_terminator(
        &mut self,
        term: &Terminator,
        next_in_order: Option<BlockId>,
    ) -> Result<(), SsaCompileError> {
        use Instruction as I;
        match term {
            Terminator::Ret { value } => {
                self.instructions.push(I::Ret(self.reg(*value)));
            }
            Terminator::Jump { target, args } => {
                let target = if args.is_empty() {
                    self.resolve_jump_alias(*target)
                } else {
                    *target
                };
                if Some(target) != next_in_order {
                    let label = self.block_label[&target];
                    self.instructions.push(I::Jump(label));
                }
            }
            Terminator::Branch {
                cond,
                lhs,
                rhs,
                t_target,
                t_args,
                f_target,
                f_args,
                ..
            } => {
                let lhs_v = self.reg(*lhs);
                let rhs_v = self.reg(*rhs);
                let t_target = if t_args.is_empty() {
                    self.resolve_jump_alias(*t_target)
                } else {
                    *t_target
                };
                let f_target = if f_args.is_empty() {
                    self.resolve_jump_alias(*f_target)
                } else {
                    *f_target
                };
                if t_target == f_target {
                    if Some(t_target) != next_in_order {
                        let label = self.block_label[&t_target];
                        self.instructions.push(I::Jump(label));
                    }
                } else if Some(t_target) == next_in_order {
                    let f_label = self.block_label[&f_target];
                    self.instructions.push(I::JumpIf(
                        f_label,
                        invert_condition(*cond),
                        lhs_v,
                        rhs_v,
                    ));
                } else {
                    let t_label = self.block_label[&t_target];
                    self.instructions
                        .push(I::JumpIf(t_label, *cond, lhs_v, rhs_v));
                    if Some(f_target) != next_in_order {
                        let f_label = self.block_label[&f_target];
                        self.instructions.push(I::Jump(f_label));
                    }
                }
            }
            Terminator::InlineBranch {
                op,
                fall_through,
                fall_args,
                bail,
                bail_args,
                ..
            } => {
                let fall_through = if fall_args.is_empty() {
                    self.resolve_jump_alias(*fall_through)
                } else {
                    *fall_through
                };
                let bail = if bail_args.is_empty() {
                    self.resolve_jump_alias(*bail)
                } else {
                    *bail
                };
                let bail_label = self.block_label[&bail];
                match op {
                    InlineBranchOp::SubChecked { dst, lhs, rhs } => {
                        self.instructions.push(I::Sub(
                            self.reg(*dst),
                            self.reg(*lhs),
                            self.reg(*rhs),
                            bail_label,
                        ));
                    }
                    InlineBranchOp::MulChecked { dst, lhs, rhs } => {
                        self.instructions.push(I::Mul(
                            self.reg(*dst),
                            self.reg(*lhs),
                            self.reg(*rhs),
                            bail_label,
                        ));
                    }
                    InlineBranchOp::DivChecked { dst, lhs, rhs } => {
                        self.instructions.push(I::Div(
                            self.reg(*dst),
                            self.reg(*lhs),
                            self.reg(*rhs),
                            bail_label,
                        ));
                    }
                    InlineBranchOp::ModuloChecked { dst, lhs, rhs } => {
                        self.instructions.push(I::Modulo(
                            self.reg(*dst),
                            self.reg(*lhs),
                            self.reg(*rhs),
                            bail_label,
                        ));
                    }
                    InlineBranchOp::ShiftLeftChecked { dst, lhs, rhs } => {
                        self.instructions.push(I::ShiftLeft(
                            self.reg(*dst),
                            self.reg(*lhs),
                            self.reg(*rhs),
                            bail_label,
                        ));
                    }
                    InlineBranchOp::ShiftRightChecked { dst, lhs, rhs } => {
                        self.instructions.push(I::ShiftRight(
                            self.reg(*dst),
                            self.reg(*lhs),
                            self.reg(*rhs),
                            bail_label,
                        ));
                    }
                    InlineBranchOp::ShiftRightZeroChecked { dst, lhs, rhs } => {
                        self.instructions.push(I::ShiftRightZero(
                            self.reg(*dst),
                            self.reg(*lhs),
                            self.reg(*rhs),
                            bail_label,
                        ));
                    }
                    InlineBranchOp::ShiftRightImmChecked { dst, src, imm } => {
                        self.instructions.push(I::ShiftRightImm(
                            self.reg(*dst),
                            self.reg(*src),
                            *imm,
                            bail_label,
                        ));
                    }
                    InlineBranchOp::GuardInt { dst, src } => {
                        self.instructions.push(I::GuardInt(
                            self.reg(*dst),
                            self.reg(*src),
                            bail_label,
                        ));
                    }
                    InlineBranchOp::GuardFloat { dst, src } => {
                        self.instructions.push(I::GuardFloat(
                            self.reg(*dst),
                            self.reg(*src),
                            bail_label,
                        ));
                    }
                    InlineBranchOp::InlineBumpAllocate {
                        dst,
                        size_bytes,
                        header,
                    } => {
                        self.instructions.push(I::InlineBumpAllocate(
                            self.reg(*dst),
                            *size_bytes,
                            *header,
                            bail_label,
                        ));
                    }
                }
                // Fall-through: if fall_through is the next block in
                // linearization, emit nothing; otherwise an explicit
                // Jump. The legacy semantics is "instruction succeeded,
                // control flows to the next address."
                if Some(fall_through) != next_in_order {
                    let fall_label = self.block_label[&fall_through];
                    self.instructions.push(I::Jump(fall_label));
                }
            }
            Terminator::Throw {
                value,
                resume,
                resume_args,
                resume_local,
                builtin_fn_ptr,
                ..
            } => {
                // `resume_args` are SSA edge transfers — cleared by
                // Phase 4f-1 `lower_to_allocated` and materialized as
                // Move ops at the resume target. Here we just emit the
                // Throw with the resume label and result-slot index.
                let resume = if resume_args.is_empty() {
                    self.resolve_jump_alias(*resume)
                } else {
                    *resume
                };
                let resume_label = self.block_label[&resume];
                self.instructions.push(I::Throw(
                    self.reg(*value),
                    resume_label,
                    resume_local.0 as usize,
                    *builtin_fn_ptr,
                ));
            }
            Terminator::Unreachable => {
                // No instruction; verifier should have caught any
                // surviving Unreachable in a reachable block before us.
            }
        }
        Ok(())
    }
}

fn compute_jump_aliases(cfg: &CfgFunction) -> HashMap<BlockId, BlockId> {
    let mut protected: HashSet<BlockId> = HashSet::new();
    for block in &cfg.blocks {
        for op in &block.body {
            protected.extend(op.block_refs());
        }
    }

    let mut aliases = HashMap::new();
    for bid_idx in 0..cfg.num_blocks() {
        let bid = BlockId(bid_idx as u32);
        if bid == cfg.entry || protected.contains(&bid) {
            continue;
        }

        let block = cfg.block(bid);
        if !block.params.is_empty() || !block.body.is_empty() {
            continue;
        }

        if let Terminator::Jump { target, args } = &block.terminator
            && args.is_empty()
            && *target != bid
        {
            aliases.insert(bid, *target);
        }
    }
    aliases
}

fn invert_condition(cond: Condition) -> Condition {
    match cond {
        Condition::LessThanOrEqual => Condition::GreaterThan,
        Condition::LessThan => Condition::GreaterThanOrEqual,
        Condition::Equal => Condition::NotEqual,
        Condition::NotEqual => Condition::Equal,
        Condition::GreaterThan => Condition::LessThanOrEqual,
        Condition::GreaterThanOrEqual => Condition::LessThan,
    }
}

// =========================================================================
// Driver: compile_via_ssa
// =========================================================================

/// Error from [`compile_via_ssa`] — caller may fall back to the
/// legacy pipeline if the SSA path can't handle the function yet.
#[derive(Debug)]
pub enum SsaCompileError {
    /// `build_cfg` rejected the legacy IR (e.g. unsupported instruction
    /// shape).
    BuildFailed(String),
    /// Greedy coloring exceeded the allocator pool and Phase 4d
    /// spilling can't currently cover it. Includes the function name
    /// for debugging.
    SpillOverflow(String),
    /// Op variant has no translation yet. Phase 4f-2d adds these as
    /// they're hit by real corpus runs.
    UnsupportedOp(String),
    /// Terminator variant has no translation yet (InlineBranch's
    /// checked-arithmetic family, Throw).
    UnsupportedTerminator(String),
}

/// Compile `ir` through the SSA pipeline:
///
/// 1. `build_cfg` → CFG with verified I1-I8.
/// 2. liveness → interference → chordal coloring.
/// 3. `assign_physical_registers` → physical-index coloring.
/// 4. `lower_to_allocated` (Phase 4f-1) — rewrites VRegs to physical
///    indices in place, materializes edge transfers as Moves.
/// 5. `translate` (Phase 4f-2a/b) → flat `Vec<Instruction>`.
/// 6. Installs the translated program on `ir` and runs the legacy
///    `compile_instructions` loop (which already handles every op's
///    backend lowering).
///
/// Returns the populated backend on success, or `SsaCompileError`
/// on bail so the caller can fall back to legacy.
pub fn compile_via_ssa<B: CodegenBackend>(
    ir: &mut Ir,
    mut backend: B,
) -> Result<B, (SsaCompileError, B)> {
    use crate::cfg::builder::build_cfg;
    use crate::cfg::regalloc::edge::Scratch;
    use crate::cfg::regalloc::emit::lower_to_allocated;
    use crate::cfg::regalloc::interference::cross_safepoint_values;
    use crate::cfg::regalloc::liveness::compute_liveness;
    use crate::cfg::regalloc::physical::{
        assign_physical_registers, current_layout, has_overflow, verify_clobber_safety,
    };
    use crate::cfg::regalloc::spill::{Budget, allocate_with_spilling};

    let mut cfg = match build_cfg(ir) {
        Ok(c) => c,
        Err(e) => return Err((SsaCompileError::BuildFailed(format!("{:?}", e)), backend)),
    };

    // Safety net (spec: a verifier failure falls back to legacy). A use
    // of a VReg with no def anywhere is a malformed CFG (e.g. the
    // variadic-prologue rest-array gap surfaced by count_items); emitting
    // code for it reads an undefined register → garbage. Bail to legacy,
    // which lowers these correctly.
    if let Some(v) = crate::cfg::verify::first_undefined_use(&cfg) {
        let name = ir.debug_name.clone().unwrap_or_else(|| "<anon>".into());
        return Err((
            SsaCompileError::BuildFailed(format!("use of undefined vreg {:?} in {}", v, name)),
            backend,
        ));
    }

    // Memory-safety guard. The interference graph is O(vregs × maxlive) and
    // liveness is O(blocks × maxlive); aggressive slot promotion can make a
    // single large, high-pressure function explode these (~80GB observed
    // under single-read promotion). Bail such a function to legacy *before*
    // building any quadratic structure. The principled fix is a
    // pressure-bounding live-range-splitting spiller (the parity plan's
    // remaining work); this is the safety valve until then. Caps are
    // env-tunable (0 disables): `BEAGLE_SSA_MAX_VREGS` (raw size, checked
    // before liveness), `BEAGLE_SSA_MAX_IG_EDGES` (pressure × size).
    let name = ir.debug_name.clone().unwrap_or_else(|| "<anon>".into());
    let nv = cfg.num_vregs();
    let nb = cfg.num_blocks();
    let log_size = std::env::var("BEAGLE_SSA_LOG_SIZE").is_ok();
    if log_size {
        eprintln!("[ssa-size] {} vregs={} blocks={}", name, nv, nb);
    }
    let raw_cap: usize = std::env::var("BEAGLE_SSA_MAX_VREGS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12000);
    if raw_cap != 0 && (nv > raw_cap || nb > raw_cap) {
        return Err((
            SsaCompileError::BuildFailed(format!(
                "oversized fn ({} vregs, {} blocks) in {} — bailed for memory safety",
                nv, nb, name
            )),
            backend,
        ));
    }
    // Pressure × size bound on the interference graph (liveness here is
    // bounded by the raw caps above, so it is safe to compute).
    let pre_liveness = compute_liveness(&cfg);
    let (mlgp, mlfp) = crate::cfg::regalloc::liveness::max_live(&cfg, &pre_liveness);
    let est_edges = (mlgp + mlfp).saturating_mul(nv);
    if log_size {
        eprintln!(
            "[ssa-size] {} maxlive_gp={} maxlive_fp={} est_edges={}",
            name, mlgp, mlfp, est_edges
        );
    }
    let edge_cap: usize = std::env::var("BEAGLE_SSA_MAX_IG_EDGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000_000);
    if edge_cap != 0 && est_edges > edge_cap {
        return Err((
            SsaCompileError::BuildFailed(format!(
                "interference too large (maxlive {}+{} × {} vregs ≈ {} edges) in {} — bailed for memory safety",
                mlgp, mlfp, nv, est_edges, name
            )),
            backend,
        ));
    }

    let layout = current_layout();

    // Phase 3: the real spiller. Color under the I7 clobber model and,
    // wherever register pressure exceeds the pool, spill the Belady
    // furthest-next-use value at the worst point — until the coloring
    // fits. GP spills go to GC-scanned root slots; the clobber model
    // keeps cross-safepoint values in callee-saved registers or root
    // slots (I9). `allocate_with_spilling` mutates `cfg` with the spill
    // SlotStore/SlotLoad rewrites.
    let budget = Budget {
        gp: layout.allocator_gp.len() as u32,
        fp: layout.allocator_fp.len() as u32,
    };
    let spill_result = allocate_with_spilling(&mut cfg, budget, layout.callee_saved_gp);
    if !spill_result.fits {
        // Pressure the body-op spiller can't lower (e.g. concentrated in
        // block params / loop-carried values) or that exceeded the spill
        // cap. Bail to legacy until block-param spilling / live-range
        // splitting lands. Tracked by the diff harness.
        let name = ir.debug_name.clone().unwrap_or_else(|| "<anon>".into());
        return Err((
            SsaCompileError::SpillOverflow(format!(
                "{} (spilled={} iters={})",
                name,
                spill_result.spilled.len(),
                spill_result.iterations
            )),
            backend,
        ));
    }
    let coloring = spill_result.coloring;
    let physical = assign_physical_registers(&cfg, &coloring, layout);

    // Postcondition (Phase 3): a fitting coloring can never overflow the
    // physical pool. If it does, the spiller's fit check and the pool
    // sizes disagree — a bug, not a spill situation.
    debug_assert!(
        !has_overflow(&physical),
        "spiller reported fit but physical assignment overflowed"
    );
    if has_overflow(&physical) {
        let name = ir.debug_name.clone().unwrap_or_else(|| "<anon>".into());
        return Err((SsaCompileError::SpillOverflow(name), backend));
    }

    // Cross-safepoint set on the post-spill CFG, for the clobber-safety
    // verifier below.
    let liveness = compute_liveness(&cfg);
    let cross_safepoint = cross_safepoint_values(&cfg, &liveness);

    // Verifier: no cross-safepoint GP value may have landed in a
    // caller-saved physical register (it would be clobbered by the
    // call). The color constraint guarantees this for allocator-pool
    // values; it can still catch a cross-safepoint *entry param* pinned
    // to an arg reg (the documented "clobber a live entry-param" case),
    // which the constraint doesn't control. Either way, bail to legacy
    // — never emit code that clobbers a live value.
    if let Err(bad) = verify_clobber_safety(&physical, &cross_safepoint, layout) {
        let name = ir.debug_name.clone().unwrap_or_else(|| "<anon>".into());
        return Err((
            SsaCompileError::BuildFailed(format!(
                "I7 clobber-safety: cross-safepoint {:?} got a caller-saved reg in {}",
                bad, name
            )),
            backend,
        ));
    }

    // Scratch registers for edge resolution's cycle breaks. Pick
    // values OUTSIDE the allocator pool so they can't collide with
    // any colored VReg. X9 (arg-count) and D31 are both reserved/
    // non-allocated.
    let scratch = Scratch { gp: 9, fp: 31 };
    lower_to_allocated(&mut cfg, &physical, scratch);
    crate::cfg::dump::maybe_dump_phase("06-after-lower-to-allocated", &cfg, false);

    // After lower_to_allocated, each VReg.index is its physical
    // register index. Translator's color_to_physical is identity.
    let translated = match translate(&cfg, |v, _| v as usize) {
        Ok(t) => t,
        Err(e) => return Err((e, backend)),
    };

    // Install translated program in `ir` and drive the legacy
    // compile_instructions loop. This is the bridge between the
    // SSA pipeline and the backend's per-op lowering.
    ir.install_translated_program(translated);

    backend.set_max_locals(ir.num_locals);
    if let Some(mark_idx) = ir.mark_local_index {
        backend.set_mark_local_index(mark_idx);
    }

    // Mark callee-saved registers in use. After physical assignment,
    // any VReg whose physical index is in the callee-saved set must
    // be saved/restored by the prologue/epilogue per AAPCS / SysV.
    backend.reset_callee_saved_tracking();
    for &phys in physical.colors.values() {
        backend.mark_callee_saved_register_used(phys as usize);
    }

    let before_prelude = backend.new_label("before_prelude");
    backend.write_label(before_prelude);
    backend.prelude();
    let after_prelude = backend.new_label("after_prelude");
    backend.write_label(after_prelude);
    let exit = backend.new_label("exit");

    ir.compile_via_legacy_emit(&mut backend, exit, before_prelude, after_prelude);

    backend.write_label(exit);
    backend.epilogue();
    backend.ret();

    Ok(backend)
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
    use crate::cfg::{CfgFunction, Op, RegClass, SlotId, Terminator};

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

        let t = translate(&f, identity_color).expect("trivial CFG translates");

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

        let t = translate(&f, identity_color).expect("trivial CFG translates");

        // bb0_label + AddInt + bb1_label + Ret  (no Jump because fall-through)
        assert_eq!(t.instructions.len(), 4, "got: {:?}", t.instructions);
        assert!(matches!(t.instructions[0], Instruction::Label(_)));
        assert!(matches!(t.instructions[1], Instruction::AddInt(_, _, _)));
        assert!(matches!(t.instructions[2], Instruction::Label(_)));
        assert!(matches!(t.instructions[3], Instruction::Ret(_)));
    }

    /// Empty jump-only blocks are final-code aliases after edge moves have
    /// already been materialized. Translation can bypass them without
    /// mutating the CFG's critical-edge-split shape.
    #[test]
    fn empty_jump_block_is_threaded_and_not_emitted() {
        let mut f = CfgFunction::new(Some("thread_empty_jump".into()), 0);
        let entry = f.new_block();
        let mid = f.new_block();
        let tail = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).terminator = Terminator::Jump {
            target: mid,
            args: vec![],
        };
        f.block_mut(mid).terminator = Terminator::Jump {
            target: tail,
            args: vec![],
        };
        f.block_mut(tail).terminator = Terminator::Ret { value: v };
        f.block_mut(mid).predecessors.push(entry);
        f.block_mut(tail).predecessors.push(mid);

        let t = translate(&f, identity_color).expect("trivial CFG translates");

        assert!(
            !t.instructions.iter().any(|i| matches!(
                i,
                Instruction::Label(label) if label.index == mid.0 as usize
            )),
            "alias block label should not be emitted: {:?}",
            t.instructions
        );
        assert!(
            !t.instructions
                .iter()
                .any(|i| matches!(i, Instruction::Jump(_))),
            "entry should fall through to threaded tail: {:?}",
            t.instructions
        );
    }

    /// After allocation, a slot store followed by a same-block reload can use
    /// a physical register move while keeping the store as the GC root.
    #[test]
    fn slot_reload_forwarded_after_store() {
        let mut f = CfgFunction::new(Some("slot_reload_forward".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let src = f.new_vreg(RegClass::Gp);
        let dst = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src,
        });
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: dst };

        let t = translate(&f, identity_color).expect("trivial CFG translates");

        assert!(
            t.instructions
                .iter()
                .any(|i| matches!(i, Instruction::StoreLocal(..))),
            "the root store must remain: {:?}",
            t.instructions
        );
        assert!(
            !t.instructions
                .iter()
                .any(|i| matches!(i, Instruction::LoadLocal(..))),
            "reload should be forwarded to a register move: {:?}",
            t.instructions
        );
        assert!(
            t.instructions
                .iter()
                .any(|i| matches!(i, Instruction::Assign(..))),
            "expected register move for forwarded reload: {:?}",
            t.instructions
        );
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

        let t = translate(&f, identity_color).expect("trivial CFG translates");

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

    /// If the true target is the next linear block, emit the inverted
    /// condition to the false target and let the true path fall through.
    #[test]
    fn branch_inverts_when_true_target_falls_through() {
        let mut f = CfgFunction::new(Some("br_true_fallthrough".into()), 0);
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
        // Make `f_block` reachable through the true block too. The side-effect
        // op keeps `t_block` from being treated as an empty jump alias. RPO
        // becomes entry, t_block, f_block, so the branch's true target is next.
        f.block_mut(t_block).body.push(Op::Breakpoint);
        f.block_mut(t_block).terminator = Terminator::Jump {
            target: f_block,
            args: vec![],
        };
        f.block_mut(f_block).terminator = Terminator::Ret { value: v };
        f.block_mut(t_block).predecessors.push(entry);
        f.block_mut(f_block).predecessors.push(entry);
        f.block_mut(f_block).predecessors.push(t_block);

        let t = translate(&f, identity_color).expect("trivial CFG translates");

        let jumpifs: Vec<_> = t
            .instructions
            .iter()
            .filter_map(|i| match i {
                Instruction::JumpIf(label, cond, _, _) => Some((*label, *cond)),
                _ => None,
            })
            .collect();
        assert_eq!(jumpifs.len(), 1, "got: {:?}", t.instructions);
        assert_eq!(jumpifs[0].0.index, f_block.0 as usize);
        assert_eq!(jumpifs[0].1, crate::ir::Condition::NotEqual);
        let jumpif_idx = t
            .instructions
            .iter()
            .position(|i| matches!(i, Instruction::JumpIf(..)))
            .expect("JumpIf exists");
        assert!(
            matches!(
                t.instructions.get(jumpif_idx + 1),
                Some(Instruction::Label(label)) if label.index == t_block.0 as usize
            ),
            "entry branch should fall through directly to true block: {:?}",
            t.instructions
        );
    }
}
