//! Loop-invariant code motion (LICM) on the SSA CFG.
//!
//! Hoists pure, loop-invariant value ops out of a loop into its preheader so
//! they execute once instead of every iteration. Runs in the Phase-3 SSA
//! optimization fixpoint (`opt::optimize`), after mem2reg has lifted locals to
//! SSA values, so invariance is a clean dominance question.
//!
//! ## What is hoisted
//!
//! Only ops in [`is_hoistable`] — pure, non-trapping, **non-memory**,
//! **non-pointer** value producers: constant materialization, integer/float
//! arithmetic that can't bail (the bailing int ops Mul/Sub/Div/shifts are
//! `InlineBranch` *terminators*, never reached here), comparisons, and
//! int→float conversions. Excluded on purpose:
//! - memory loads (`HeapLoad`/`SlotLoad`/`AtomicLoad`): `has_side_effect` calls
//!   them pure, but the memory they read can be mutated in the loop, so their
//!   value is not loop-invariant;
//! - pointer producers (`Const*Ptr`, `Tag`, `Untag`, fmov/bit ops): hoisting a
//!   pointer (or a raw value that may be a pointer) above the loop could create
//!   a live reference across a loop-body GC safepoint that isn't root-slotted.
//!   The non-pointer set we hoist needs no rooting (the collector ignores FP
//!   and non-pointer GP values).
//!
//! ## Invariance
//!
//! In SSA a VReg is loop-invariant iff its single def is outside the loop body
//! (block params of loop blocks are loop-variant — they take back-edge values).
//! An op is hoistable iff it is in the pure set and every operand is invariant
//! or defined by an already-hoisted op. Computed to a fixpoint by rounds so a
//! hoisted op's def unlocks its dependents, and the collected order is a valid
//! dependency order for the preheader.
//!
//! ## Preheader
//!
//! We only hoist into an existing, unambiguous preheader: the loop header's
//! single predecessor that is *outside* the loop and whose terminator is a
//! `Jump` straight to the header (so the hoisted ops run exactly once on entry
//! and never on the back-edge). If a loop has no such block, it is skipped
//! (creating one is left to a later step). The invariant operands of a hoisted
//! op dominate the header and therefore the preheader, so the moved ops are
//! well-formed there.

use std::collections::{HashMap, HashSet};

use crate::cfg::gc_safety::{is_gc_safepoint_op, is_gc_safepoint_terminator};
use crate::cfg::loops::natural_loops;
use crate::cfg::{BlockId, CfgFunction, Op, SlotId, Terminator, VReg};

/// Hoist loop-invariant pure ops into loop preheaders. Returns whether any op
/// moved. Opt out with `BEAGLE_SSA_NO_LICM`.
pub fn loop_invariant_code_motion(f: &mut CfgFunction) -> bool {
    let loops = natural_loops(f);
    if loops.is_empty() {
        return false;
    }
    let def_block = def_block_map(f);
    let mut changed = false;
    // Process inner loops first (larger header id tends to be inner; but
    // hoisting from an inner loop into its preheader can then be hoisted again
    // by the enclosing loop on a later fixpoint iteration of `optimize`).
    for lp in &loops {
        let Some(preheader) = find_preheader(f, lp.header, &lp.body) else {
            continue;
        };
        // Can the loop body trigger a GC? If not, a hoisted value (even a
        // pointer) can't go stale inside the loop, so we may also hoist
        // invariant memory reads (`SlotLoad`) and other pointer-ish pure ops.
        let no_safepoint = !loop_can_gc(f, &lp.body);
        // Slots written anywhere in the loop are not loop-invariant.
        let slots_stored = slots_stored_in_loop(f, &lp.body);
        let hoisted = collect_hoistable(f, &lp.body, &def_block, no_safepoint, &slots_stored);
        if hoisted.is_empty() {
            continue;
        }
        move_ops_to_preheader(f, &hoisted, preheader);
        changed = true;
    }
    changed
}

/// Map every VReg to the block that defines it (block params + op defs).
fn def_block_map(f: &CfgFunction) -> HashMap<VReg, BlockId> {
    let mut m = HashMap::new();
    for (i, block) in f.blocks.iter().enumerate() {
        let bid = BlockId(i as u32);
        for &p in &block.params {
            m.insert(p, bid);
        }
        for op in &block.body {
            for d in op.defs() {
                m.insert(d, bid);
            }
        }
    }
    m
}

/// The loop's preheader, if it is unambiguous: a single predecessor of the
/// header outside the loop body whose terminator jumps straight to the header.
fn find_preheader(f: &CfgFunction, header: BlockId, body: &HashSet<BlockId>) -> Option<BlockId> {
    let hblock = &f.blocks[header.0 as usize];
    let entry_preds: Vec<BlockId> = hblock
        .predecessors
        .iter()
        .copied()
        .filter(|p| !body.contains(p))
        .collect();
    let [pre] = entry_preds.as_slice() else {
        return None;
    };
    match &f.blocks[pre.0 as usize].terminator {
        Terminator::Jump { target, .. } if *target == header => Some(*pre),
        _ => None,
    }
}

/// Whether any op or terminator in the loop body is a GC safepoint.
fn loop_can_gc(f: &CfgFunction, body: &HashSet<BlockId>) -> bool {
    body.iter().any(|&b| {
        let block = &f.blocks[b.0 as usize];
        block.body.iter().any(is_gc_safepoint_op) || is_gc_safepoint_terminator(&block.terminator)
    })
}

/// Slots written by a `SlotStore` anywhere in the loop body — their loaded
/// value is not loop-invariant.
fn slots_stored_in_loop(f: &CfgFunction, body: &HashSet<BlockId>) -> HashSet<SlotId> {
    let mut out = HashSet::new();
    for &b in body {
        for op in &f.blocks[b.0 as usize].body {
            if let Op::SlotStore { slot, .. } = op {
                out.insert(*slot);
            }
        }
    }
    out
}

/// An op identified for hoisting: its block and index within that block's body.
struct HoistSite {
    block: BlockId,
    index: usize,
}

/// Collect, in dependency order, the hoistable ops of the loop body.
fn collect_hoistable(
    f: &CfgFunction,
    body: &HashSet<BlockId>,
    def_block: &HashMap<VReg, BlockId>,
    no_safepoint: bool,
    slots_stored: &HashSet<SlotId>,
) -> Vec<HoistSite> {
    // A VReg is invariant if defined outside the loop body, or defined by an
    // op we have already decided to hoist.
    let invariant = |v: VReg, hoisted_defs: &HashSet<VReg>| -> bool {
        hoisted_defs.contains(&v)
            || match def_block.get(&v) {
                Some(b) => !body.contains(b),
                // No recorded def → a function entry param / external; treat as
                // invariant (it is defined before the function body).
                None => true,
            }
    };

    let mut sites: Vec<HoistSite> = Vec::new();
    let mut hoisted_defs: HashSet<VReg> = HashSet::new();
    let mut taken: HashSet<(u32, usize)> = HashSet::new();

    loop {
        let mut found = false;
        for &bid in body {
            let block = &f.blocks[bid.0 as usize];
            for (idx, op) in block.body.iter().enumerate() {
                if taken.contains(&(bid.0, idx)) || !is_hoistable(op, no_safepoint, slots_stored) {
                    continue;
                }
                if op.uses().iter().all(|&u| invariant(u, &hoisted_defs)) {
                    for d in op.defs() {
                        hoisted_defs.insert(d);
                    }
                    sites.push(HoistSite {
                        block: bid,
                        index: idx,
                    });
                    taken.insert((bid.0, idx));
                    found = true;
                }
            }
        }
        if !found {
            break;
        }
    }
    sites
}

/// Physically move the collected ops (in order) into the preheader, just
/// before its terminator, removing them from their original blocks.
fn move_ops_to_preheader(f: &mut CfgFunction, sites: &[HoistSite], preheader: BlockId) {
    // Pull the ops out (clone, then remove). Remove from the back of each block
    // so earlier indices stay valid.
    let mut moving: Vec<Op> = Vec::with_capacity(sites.len());
    for site in sites {
        moving.push(f.blocks[site.block.0 as usize].body[site.index].clone());
    }
    // Remove originals: group indices per block, remove descending.
    let mut by_block: HashMap<u32, Vec<usize>> = HashMap::new();
    for site in sites {
        by_block.entry(site.block.0).or_default().push(site.index);
    }
    for (bid, mut idxs) in by_block {
        idxs.sort_unstable();
        idxs.dedup();
        for &i in idxs.iter().rev() {
            f.blocks[bid as usize].body.remove(i);
        }
    }
    // Append to the preheader, preserving dependency order.
    let pre = &mut f.blocks[preheader.0 as usize];
    for op in moving {
        pre.body.push(op);
    }
}

/// Whether `op` is safe to hoist out of a loop, given whether the loop body
/// can GC (`no_safepoint`) and which slots are written in the loop.
///
/// Two tiers:
/// - **Always** (GC-safe even across safepoints): non-pointer value
///   computations — integer add, comparisons, and float arithmetic. Their
///   results are tagged ints / FP values that the collector never treats as
///   live pointers in a register, so extending their live range is harmless.
///   (Bailing int Mul/Sub/Div/shifts are `InlineBranch` terminators, not body
///   ops, so they don't appear here.)
/// - **Only when the loop cannot GC**: an invariant `SlotLoad` (slot never
///   written in the loop) and other pure ops that may yield a pointer/raw
///   value (`Tag`/`Untag`/bit ops/fmov). With no safepoint in the loop the
///   heap can't move while the hoisted value is live, so a possibly-pointer
///   register value can't go stale.
///
/// Constant materialization is intentionally never hoisted on its own — a
/// constant is free to rematerialize and pinning it in a register across the
/// loop only adds register pressure (measured as a small loss on mandelbrot).
/// An invariant arithmetic op that consumes a constant still pulls the
/// computation out.
fn is_hoistable(op: &Op, no_safepoint: bool, slots_stored: &HashSet<SlotId>) -> bool {
    match op {
        Op::AddInt { .. }
        | Op::Compare { .. }
        | Op::CompareFloat { .. }
        | Op::AddFloat { .. }
        | Op::SubFloat { .. }
        | Op::MulFloat { .. }
        | Op::DivFloat { .. }
        | Op::IntToFloat { .. }
        | Op::FRoundToZero { .. } => true,

        // Invariant memory read — only safe to lift past the loop's iterations
        // when the loop can't GC (pointer staleness) and the slot is constant.
        Op::SlotLoad { slot, .. } => no_safepoint && !slots_stored.contains(slot),

        // Possibly-pointer / raw pure ops: GC-safe to hoist only in a
        // safepoint-free loop.
        Op::Untag { .. }
        | Op::GetTag { .. }
        | Op::Tag { .. }
        | Op::And { .. }
        | Op::Or { .. }
        | Op::Xor { .. }
        | Op::AndImm { .. }
        | Op::ShiftRightImmRaw { .. }
        | Op::FmovGpToFp { .. }
        | Op::FmovFpToGp { .. } => no_safepoint,

        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{Condition, RegClass};

    #[test]
    fn hoists_invariant_addint_to_preheader() {
        // P (preheader): a=3; b=7; Jump H
        // H (loop):      t = a + b   <- loop-invariant; back-edge H->H
        //                Branch(t==t ? H : Exit)
        // Exit:          Ret t
        let mut f = CfgFunction::new(Some("licm_test".into()), 0);
        let p = f.new_block();
        let h = f.new_block();
        let exit = f.new_block();
        f.entry = p;

        let a = f.new_vreg(RegClass::Gp);
        let b = f.new_vreg(RegClass::Gp);
        let t = f.new_vreg(RegClass::Gp);

        f.block_mut(p)
            .body
            .push(Op::ConstTaggedInt { dst: a, value: 3 });
        f.block_mut(p)
            .body
            .push(Op::ConstTaggedInt { dst: b, value: 7 });
        f.block_mut(p).terminator = Terminator::Jump {
            target: h,
            args: vec![],
        };

        f.block_mut(h).body.push(Op::AddInt {
            dst: t,
            lhs: a,
            rhs: b,
        });
        f.block_mut(h).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: t,
            rhs: t,
            t_target: h,
            t_args: vec![],
            f_target: exit,
            f_args: vec![],
        };
        f.block_mut(exit).terminator = Terminator::Ret { value: t };

        f.block_mut(h).predecessors = vec![p, h];
        f.block_mut(exit).predecessors = vec![h];

        let changed = loop_invariant_code_motion(&mut f);
        assert!(changed, "invariant AddInt should be hoisted");
        assert!(
            !f.block(h)
                .body
                .iter()
                .any(|o| matches!(o, Op::AddInt { .. })),
            "AddInt must be removed from the loop body"
        );
        assert!(
            f.block(p)
                .body
                .iter()
                .any(|o| matches!(o, Op::AddInt { .. })),
            "AddInt must now be in the preheader"
        );
    }

    #[test]
    fn does_not_hoist_variant_op() {
        // t depends on a loop block param (variant) — must NOT hoist.
        let mut f = CfgFunction::new(Some("licm_variant".into()), 0);
        let p = f.new_block();
        let h = f.new_block();
        let exit = f.new_block();
        f.entry = p;

        let iv = f.new_vreg(RegClass::Gp); // loop-carried (header param)
        let one = f.new_vreg(RegClass::Gp);
        let t = f.new_vreg(RegClass::Gp);

        f.block_mut(p)
            .body
            .push(Op::ConstTaggedInt { dst: one, value: 1 });
        f.block_mut(p).terminator = Terminator::Jump {
            target: h,
            args: vec![one],
        };
        f.block_mut(h).params = vec![iv]; // iv is a loop param => variant
        f.block_mut(h).body.push(Op::AddInt {
            dst: t,
            lhs: iv,
            rhs: one,
        });
        f.block_mut(h).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: t,
            rhs: t,
            t_target: h,
            t_args: vec![t],
            f_target: exit,
            f_args: vec![],
        };
        f.block_mut(exit).terminator = Terminator::Ret { value: t };
        f.block_mut(h).predecessors = vec![p, h];
        f.block_mut(exit).predecessors = vec![h];

        let changed = loop_invariant_code_motion(&mut f);
        assert!(!changed, "op using a loop param must not be hoisted");
        assert!(
            f.block(h)
                .body
                .iter()
                .any(|o| matches!(o, Op::AddInt { .. })),
            "AddInt must stay in the loop body"
        );
    }
}
