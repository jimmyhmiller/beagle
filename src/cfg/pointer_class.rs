//! Phase A of the type-aware GC-safety project: a **sound non-pointer
//! analysis**.
//!
//! Per **I9** (`docs/SSA_ARCHITECTURE.md`) the GC scans frame slots, not
//! registers, so any GP value live across a safepoint must be rooted in a
//! slot — *because it might be a relocatable heap pointer*. But a value that is
//! provably a tagged immediate (int / bool / null / raw small value) is **never
//! a heap pointer the GC relocates**, so it needs no root slot across a
//! safepoint. This module computes which SSA VRegs and which slots are provably
//! non-pointer; later phases use it to let `mem2reg` promote non-pointer slots
//! across safepoints (B/C) and to let regalloc keep them in registers across
//! calls without rooting.
//!
//! **Soundness is everything here.** A false "non-pointer" means the GC misses
//! a real root → use-after-free / SIGBUS under a compacting collector. So the
//! analysis is a *must* analysis with a maybe-pointer default: a VReg/slot is
//! non-pointer only when **every** definition / store that can reach it is
//! provably non-pointer. Anything unrecognized stays maybe-pointer.
//!
//! ## What is provably non-pointer
//!
//! Only a small, audited whitelist of producers — each yields a tagged
//! immediate or a raw non-pointer scalar, never a relocatable heap address:
//! - integer/bool/null constants (`ConstTaggedInt`, `ConstTrue`, `ConstFalse`,
//!   `ConstNull`);
//! - the non-bailing integer add (`AddInt`) and the checked integer ops
//!   (`SubChecked`/`MulChecked`/`DivChecked`/`ModuloChecked`/the shift-checked
//!   family, all `InlineBranch` terminators) — all produce tagged ints;
//! - the int guard's output (`GuardInt` dst) — a value proven int;
//! - comparisons (`Compare`/`CompareFloat`, raw 0/1) and raw bit ops
//!   (`GetTag`, `AndImm`, `ShiftRightImmRaw`).
//!
//! Deliberately **NOT** non-pointer: boxed floats. In the current value model a
//! float is a *tagged pointer to a heap box*, so `GuardFloat` / `Tag` results
//! are relocatable pointers. (Unboxed `f64` lives in the separate `Fp` register
//! class, already GC-exempt — that is Phase D's concern, not this set.) Also
//! maybe-pointer: `Untag` (raw heap address), `HeapLoad`, `Call`/`Recurse`
//! results, the `Const*Ptr`/`ConstFunctionId`/`ConstPointer`/`ConstRawValue`
//! family (pointers or opaque bit patterns), and anything else.
//!
//! ## Propagation
//!
//! Three def kinds carry the property from their sources, to a fixpoint:
//! - `Move { dst, src }` — `dst` is non-pointer iff `src` is;
//! - `SlotLoad { dst, slot }` — `dst` is non-pointer iff `slot` is;
//! - block params — non-pointer iff **every** predecessor's incoming arg is.
//!
//! A **slot** is non-pointer iff it has at least one store and **every**
//! `SlotStore` into it stores a non-pointer source. Entry-block params (function
//! arguments) and any VReg with no recognized def are maybe-pointer.
//!
//! Computed as greatest-fixpoint demotion: start everything optimistic
//! (non-pointer), then repeatedly demote any VReg/slot one of whose sources is
//! maybe-pointer until stable. Monotone (the maybe set only grows) and bounded
//! by VReg+slot count, so it terminates; at the fixpoint a VReg/slot is
//! non-pointer iff all reaching definitions are provably non-pointer.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::cfg::{BlockId, CfgFunction, InlineBranchOp, Op, SlotId, Terminator, VReg};

/// Result of the non-pointer analysis.
pub struct PointerClass {
    /// VRegs proven to never hold a relocatable heap pointer.
    pub non_pointer_vregs: HashSet<VReg>,
    /// Slots proven to only ever hold non-pointer values.
    pub non_pointer_slots: HashSet<SlotId>,
}

/// How a VReg gets its value, for the non-pointer fixpoint.
enum DefKind {
    /// Definitely non-pointer by its own producer (whitelist).
    NonPointer,
    /// Definitely maybe-pointer by its own producer, or undefined / entry param.
    MaybePointer,
    /// `Move`: non-pointer iff `src` is.
    FromVReg(VReg),
    /// `SlotLoad`: non-pointer iff the slot is.
    FromSlot(SlotId),
    /// Block param: non-pointer iff all incoming args are.
    FromParams(Vec<VReg>),
}

/// Compute the provably-non-pointer VRegs and slots of `f`.
pub fn analyze(f: &CfgFunction) -> PointerClass {
    // --- Gather every VReg's def kind and every slot's store sources. ---
    let mut def_kind: HashMap<VReg, DefKind> = HashMap::new();
    let mut slot_stores: HashMap<SlotId, Vec<VReg>> = HashMap::new();
    let mut all_slots: HashSet<SlotId> = HashSet::new();

    // Entry params are function arguments — unknown type, maybe-pointer.
    for &p in &f.block(f.entry).params {
        def_kind.insert(p, DefKind::MaybePointer);
    }

    let preds = predecessor_map(f);

    for (bi, block) in f.blocks.iter().enumerate() {
        let bid = BlockId(bi as u32);

        // Non-entry block params: propagate from every predecessor's arg.
        if bid != f.entry && !block.params.is_empty() {
            let plist = preds.get(&bid).cloned().unwrap_or_default();
            for (idx, &param) in block.params.iter().enumerate() {
                if plist.is_empty() {
                    def_kind.insert(param, DefKind::MaybePointer);
                    continue;
                }
                let mut incoming = Vec::with_capacity(plist.len());
                let mut ok = true;
                for &p in &plist {
                    match incoming_arg(f, p, bid, idx) {
                        Some(v) => incoming.push(v),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    def_kind.insert(param, DefKind::FromParams(incoming));
                } else {
                    def_kind.insert(param, DefKind::MaybePointer);
                }
            }
        }

        for op in &block.body {
            match op {
                Op::SlotStore { slot, src } => {
                    all_slots.insert(*slot);
                    slot_stores.entry(*slot).or_default().push(*src);
                }
                Op::SlotLoad { dst, slot } => {
                    all_slots.insert(*slot);
                    def_kind.insert(*dst, DefKind::FromSlot(*slot));
                }
                Op::Move { dst, src } => {
                    def_kind.insert(*dst, DefKind::FromVReg(*src));
                }
                _ => {
                    let np = op_produces_non_pointer(op);
                    for d in op.defs() {
                        def_kind.insert(d, kind_for(np));
                    }
                }
            }
        }

        // The InlineBranch terminator's op may define a (tagged-int) dst.
        if let Terminator::InlineBranch { op, .. } = &block.terminator {
            if let Some(dst) = inline_branch_dst(op) {
                def_kind.insert(dst, kind_for(inline_branch_produces_non_pointer(op)));
            }
        }
    }

    // --- Greatest-fixpoint demotion. ---
    // Everything starts optimistic (non-pointer); we demote to maybe-pointer.
    let mut maybe_vregs: HashSet<VReg> = HashSet::new();
    let mut maybe_slots: HashSet<SlotId> = HashSet::new();

    // A slot with no stores is uninitialized for analysis purposes → maybe.
    for &s in &all_slots {
        if !slot_stores.contains_key(&s) {
            maybe_slots.insert(s);
        }
    }
    // Seed definite-maybe VRegs.
    for (&v, k) in &def_kind {
        if matches!(k, DefKind::MaybePointer) {
            maybe_vregs.insert(v);
        }
    }

    let mut changed = true;
    while changed {
        changed = false;

        // A slot is maybe if any of its store sources is maybe (or undefined).
        for (&slot, srcs) in &slot_stores {
            if maybe_slots.contains(&slot) {
                continue;
            }
            if srcs
                .iter()
                .any(|s| is_maybe_vreg(*s, &def_kind, &maybe_vregs))
            {
                maybe_slots.insert(slot);
                changed = true;
            }
        }

        // A VReg is maybe if its def's source(s) are maybe.
        for (&v, k) in &def_kind {
            if maybe_vregs.contains(&v) {
                continue;
            }
            let demote = match k {
                DefKind::MaybePointer => true,
                DefKind::NonPointer => false,
                DefKind::FromVReg(src) => is_maybe_vreg(*src, &def_kind, &maybe_vregs),
                DefKind::FromSlot(slot) => maybe_slots.contains(slot),
                DefKind::FromParams(args) => args
                    .iter()
                    .any(|a| is_maybe_vreg(*a, &def_kind, &maybe_vregs)),
            };
            if demote {
                maybe_vregs.insert(v);
                changed = true;
            }
        }
    }

    // --- Invert to the non-pointer sets. ---
    let non_pointer_vregs = def_kind
        .keys()
        .copied()
        .filter(|v| !maybe_vregs.contains(v))
        .collect();
    let non_pointer_slots = all_slots
        .into_iter()
        .filter(|s| !maybe_slots.contains(s))
        .collect();

    PointerClass {
        non_pointer_vregs,
        non_pointer_slots,
    }
}

/// A VReg is maybe-pointer if it's in the demoted set, or has no recognized def
/// (defensive: an unknown VReg is conservatively a pointer).
fn is_maybe_vreg(v: VReg, def_kind: &HashMap<VReg, DefKind>, maybe: &HashSet<VReg>) -> bool {
    maybe.contains(&v) || !def_kind.contains_key(&v)
}

fn kind_for(non_pointer: bool) -> DefKind {
    if non_pointer {
        DefKind::NonPointer
    } else {
        DefKind::MaybePointer
    }
}

/// Whitelist of body ops that definitely produce a GP non-pointer (tagged
/// immediate or raw scalar). Everything not listed → maybe-pointer.
fn op_produces_non_pointer(op: &Op) -> bool {
    matches!(
        op,
        Op::ConstTaggedInt { .. }
            | Op::ConstTrue { .. }
            | Op::ConstFalse { .. }
            | Op::ConstNull { .. }
            | Op::AddInt { .. }
            | Op::Compare { .. }
            | Op::CompareFloat { .. }
            | Op::GetTag { .. }
            | Op::AndImm { .. }
            | Op::ShiftRightImmRaw { .. }
    )
}

/// The dst defined by an `InlineBranch` op on its fall-through, if any.
fn inline_branch_dst(op: &InlineBranchOp) -> Option<VReg> {
    match op {
        InlineBranchOp::SubChecked { dst, .. }
        | InlineBranchOp::MulChecked { dst, .. }
        | InlineBranchOp::DivChecked { dst, .. }
        | InlineBranchOp::ModuloChecked { dst, .. }
        | InlineBranchOp::ShiftLeftChecked { dst, .. }
        | InlineBranchOp::ShiftRightChecked { dst, .. }
        | InlineBranchOp::ShiftRightZeroChecked { dst, .. }
        | InlineBranchOp::ShiftRightImmChecked { dst, .. }
        | InlineBranchOp::GuardInt { dst, .. }
        | InlineBranchOp::GuardFloat { dst, .. } => Some(*dst),
        InlineBranchOp::InlineBumpAllocate { .. } => None,
    }
}

/// Whether an `InlineBranch` op's dst is a non-pointer. The checked integer ops
/// and the int guard yield tagged ints; the float guard yields a boxed float
/// (a relocatable pointer) → maybe-pointer.
fn inline_branch_produces_non_pointer(op: &InlineBranchOp) -> bool {
    matches!(
        op,
        InlineBranchOp::SubChecked { .. }
            | InlineBranchOp::MulChecked { .. }
            | InlineBranchOp::DivChecked { .. }
            | InlineBranchOp::ModuloChecked { .. }
            | InlineBranchOp::ShiftLeftChecked { .. }
            | InlineBranchOp::ShiftRightChecked { .. }
            | InlineBranchOp::ShiftRightZeroChecked { .. }
            | InlineBranchOp::ShiftRightImmChecked { .. }
            | InlineBranchOp::GuardInt { .. }
    )
}

/// Predecessor map (block -> its predecessor blocks), from terminators.
fn predecessor_map(f: &CfgFunction) -> HashMap<BlockId, Vec<BlockId>> {
    let mut m: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for (i, block) in f.blocks.iter().enumerate() {
        let from = BlockId(i as u32);
        for succ in block.terminator.successors() {
            m.entry(succ).or_default().push(from);
        }
    }
    m
}

/// The arg `pred` passes to `succ`'s block param at position `idx`.
fn incoming_arg(f: &CfgFunction, pred: BlockId, succ: BlockId, idx: usize) -> Option<VReg> {
    match &f.blocks[pred.0 as usize].terminator {
        Terminator::Jump { target, args } if *target == succ => args.get(idx).copied(),
        Terminator::Branch {
            t_target,
            t_args,
            f_target,
            f_args,
            ..
        } => {
            if *t_target == succ {
                t_args.get(idx).copied()
            } else if *f_target == succ {
                f_args.get(idx).copied()
            } else {
                None
            }
        }
        Terminator::InlineBranch {
            fall_through,
            fall_args,
            bail,
            bail_args,
            ..
        } => {
            if *fall_through == succ {
                fall_args.get(idx).copied()
            } else if *bail == succ {
                bail_args.get(idx).copied()
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CallTarget, ClobberSet, RegClass, Terminator};

    /// A slot fed only by int constants and AddInt is a non-pointer slot, and a
    /// load of it is a non-pointer VReg — even with a Call (safepoint) present.
    #[test]
    fn int_accumulator_slot_is_non_pointer() {
        let mut f = CfgFunction::new(Some("acc".into()), 0);
        let b = f.new_block();
        f.entry = b;
        let c0 = f.new_vreg(RegClass::Gp);
        let loaded = f.new_vreg(RegClass::Gp);
        let one = f.new_vreg(RegClass::Gp);
        let next = f.new_vreg(RegClass::Gp);
        let cd = f.new_vreg(RegClass::Gp);

        f.block_mut(b)
            .body
            .push(Op::ConstTaggedInt { dst: c0, value: 0 });
        f.block_mut(b).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: c0,
        });
        // A call in the middle (a safepoint) — must not affect the classification.
        f.block_mut(b).body.push(Op::Call {
            dst: cd,
            target: CallTarget::Pointer(0x1000),
            args: vec![],
            is_builtin: true,
            clobbers: ClobberSet::AllCallerSaved,
        });
        f.block_mut(b).body.push(Op::SlotLoad {
            dst: loaded,
            slot: SlotId(0),
        });
        f.block_mut(b)
            .body
            .push(Op::ConstTaggedInt { dst: one, value: 1 });
        f.block_mut(b).body.push(Op::AddInt {
            dst: next,
            lhs: loaded,
            rhs: one,
        });
        f.block_mut(b).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: next,
        });
        f.block_mut(b).terminator = Terminator::Ret { value: next };

        let pc = analyze(&f);
        assert!(
            pc.non_pointer_slots.contains(&SlotId(0)),
            "slot is int-only"
        );
        assert!(
            pc.non_pointer_vregs.contains(&loaded),
            "load is non-pointer"
        );
        assert!(
            pc.non_pointer_vregs.contains(&next),
            "AddInt is non-pointer"
        );
        // The call result is unknown-typed → maybe-pointer.
        assert!(
            !pc.non_pointer_vregs.contains(&cd),
            "call result is a pointer"
        );
    }

    /// A slot that ever stores a non-whitelisted (maybe-pointer) value is a
    /// pointer slot, and so is every load of it.
    #[test]
    fn pointer_store_taints_slot() {
        let mut f = CfgFunction::new(Some("ptr".into()), 0);
        let b = f.new_block();
        f.entry = b;
        let c0 = f.new_vreg(RegClass::Gp);
        let heap = f.new_vreg(RegClass::Gp);
        let base = f.new_vreg(RegClass::Gp);
        let loaded = f.new_vreg(RegClass::Gp);

        f.block_mut(b)
            .body
            .push(Op::ConstTaggedInt { dst: c0, value: 0 });
        f.block_mut(b).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: c0,
        });
        // A HeapLoad result is maybe-pointer; storing it taints the slot.
        f.block_mut(b).body.push(Op::ConstPointer {
            dst: base,
            ptr: 0x4000,
        });
        f.block_mut(b).body.push(Op::HeapLoad {
            dst: heap,
            base,
            offset: 0,
        });
        f.block_mut(b).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: heap,
        });
        f.block_mut(b).body.push(Op::SlotLoad {
            dst: loaded,
            slot: SlotId(0),
        });
        f.block_mut(b).terminator = Terminator::Ret { value: loaded };

        let pc = analyze(&f);
        assert!(
            !pc.non_pointer_slots.contains(&SlotId(0)),
            "a slot with a heap-load store is a pointer slot"
        );
        assert!(
            !pc.non_pointer_vregs.contains(&loaded),
            "load of a pointer slot is maybe-pointer"
        );
        assert!(!pc.non_pointer_vregs.contains(&heap));
    }

    /// Entry params (function args) are maybe-pointer; a loop-carried phi fed by
    /// an int const and an AddInt is non-pointer.
    #[test]
    fn loop_carried_int_phi_is_non_pointer() {
        // entry(p): jump H[c0]
        // H(iv):    next = iv + one ; jump H[next]   (self-loop)
        let mut f = CfgFunction::new(Some("loop".into()), 0);
        let entry = f.new_block();
        let h = f.new_block();
        f.entry = entry;
        let p = f.new_vreg(RegClass::Gp); // entry param (function arg)
        let c0 = f.new_vreg(RegClass::Gp);
        let iv = f.new_vreg(RegClass::Gp); // loop phi
        let one = f.new_vreg(RegClass::Gp);
        let next = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).params = vec![p];
        f.block_mut(entry)
            .body
            .push(Op::ConstTaggedInt { dst: c0, value: 0 });
        f.block_mut(entry).terminator = Terminator::Jump {
            target: h,
            args: vec![c0],
        };
        f.block_mut(h).params = vec![iv];
        f.block_mut(h)
            .body
            .push(Op::ConstTaggedInt { dst: one, value: 1 });
        f.block_mut(h).body.push(Op::AddInt {
            dst: next,
            lhs: iv,
            rhs: one,
        });
        f.block_mut(h).terminator = Terminator::Jump {
            target: h,
            args: vec![next],
        };
        f.block_mut(h).predecessors = vec![entry, h];

        let pc = analyze(&f);
        assert!(
            pc.non_pointer_vregs.contains(&iv),
            "loop int phi is non-pointer"
        );
        assert!(pc.non_pointer_vregs.contains(&next));
        assert!(
            !pc.non_pointer_vregs.contains(&p),
            "entry param is maybe-pointer"
        );
    }

    /// A boxed-float guard output is a pointer (to the heap box), so a slot fed
    /// by it is a pointer slot.
    #[test]
    fn boxed_float_is_pointer() {
        let mut f = CfgFunction::new(Some("flt".into()), 0);
        let b = f.new_block();
        f.entry = b;
        let src = f.new_vreg(RegClass::Gp);
        let g = f.new_vreg(RegClass::Gp);
        f.block_mut(b).body.push(Op::ConstPointer {
            dst: src,
            ptr: 0x8000,
        });
        f.block_mut(b).terminator = Terminator::InlineBranch {
            op: InlineBranchOp::GuardFloat { dst: g, src },
            fall_through: b,
            fall_args: vec![],
            bail: b,
            bail_args: vec![],
        };
        // Self-referential terminator just to satisfy the shape; analyze only
        // reads the op's dst classification.
        f.block_mut(b).predecessors = vec![b];

        let pc = analyze(&f);
        assert!(
            !pc.non_pointer_vregs.contains(&g),
            "GuardFloat output is a boxed-float pointer"
        );
    }
}
