//! Promote profitable stack slots to SSA values (Cytron-style mem2reg).
//!
//! Per **I6** in `docs/SSA_ARCHITECTURE.md`, CFG construction leaves
//! every `Value::Local` as a `SlotLoad` / `SlotStore`. This pass — Phase
//! 2 of the SSA pipeline — promotes slots whose reads pay for the
//! cost of phi-node insertion, replacing their loads with `Op::Move`
//! from the SSA value currently flowing through and inserting block
//! params (= phi nodes, in our model per **I3**) at the iterated
//! dominance frontier of each slot's write sites.
//!
//! Profitability gate: skip slots with fewer than 2 reads. The dynir
//! mem2reg uses the same gate; the prior `ssa` branch that DIDN'T have
//! this gate produced 3060 surviving non-trivial merge phis on
//! `nbody/advance` (vs 117 after the gate landed).
//!
//! Anti-patterns avoided (per the spec's forbidden list):
//! - F10 (Phi op in SSA layer): we emit block params, never a
//!   `Phi`-style op.
//! - F1 (Label in body): unchanged; mem2reg doesn't touch labels.
//! - F2 (dead-coded coalescer): not a coalescer, but the same
//!   discipline applies — the rewriter is wired into `build_cfg`, not
//!   shipped commented-out.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::cfg::dom::{
    compute_dominance_frontiers, compute_idoms, dominator_tree_children,
    iterated_dominance_frontier, reverse_postorder,
};
use crate::cfg::{BlockId, CfgFunction, Op, RegClass, SlotId, Terminator, VReg};

/// Run mem2reg on `f`. Mutates the function in place: promoted slots'
/// `SlotLoad` ops become `Op::Move` referencing the SSA value flowing
/// through; promoted slots' `SlotStore` ops are dropped (their src
/// becomes the new "current value" for the slot); block params are
/// appended at the iterated dominance frontier of each promoted slot's
/// write sites; terminator args are populated to pass the right values
/// along each outgoing edge.
///
/// Slots that don't pay (fewer than 2 reads) are left alone — their
/// `SlotLoad`/`SlotStore` ops survive unchanged.
pub fn promote_slots(f: &mut CfgFunction) {
    if f.blocks.is_empty() {
        return;
    }

    let (slot_writes, slot_reads) = collect_slot_sites(f);
    // Profitability gate: promote slots with >= `min_reads` reads. Default
    // is now **1** — promote even single-read slots, eliminating the
    // store→load round-trips the AST compiler emits for every intermediate.
    // This is the runtime win: it collapses the slot/memory traffic that made
    // SSA slower than legacy (specialized fib went 36 slot ops → 7), and is
    // only viable because pruned SSA (the φ-placement liveness gate) keeps
    // register pressure sane instead of exploding. Validated 364/364 with
    // peak ~0.05 GB. `BEAGLE_SSA_PROMOTE_MIN_READS=2` restores the
    // conservative gate for A/B.
    let min_reads: usize = std::env::var("BEAGLE_SSA_PROMOTE_MIN_READS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let read_filtered: HashSet<SlotId> = slot_writes
        .keys()
        .copied()
        .filter(|s| slot_reads.get(s).copied().unwrap_or(0) >= min_reads)
        .collect();
    if read_filtered.is_empty() {
        return;
    }

    let slot_class = infer_slot_classes(f, &read_filtered);

    // Handler-block gate: blocks reached only through the runtime
    // exception / continuation mechanism (handler / resume / abort
    // blocks) have no normal CFG predecessors, so they are absent from
    // the dominator tree that the rename walk (Phase 2) follows. For such
    // a block mem2reg would drop a promoted slot's `SlotStore` (in the
    // reachable region) but leave the block's `SlotLoad` un-rewritten —
    // the handler then reads an uninitialised (null) slot. Symptom: a
    // `catch` body that reads >=2 distinct closure free vars segfaults on
    // a null closure pointer (`load(untag(arg0), 4+idx)` with arg0 read
    // back from slot 0, which was promoted away). Any slot loaded in a
    // block unreachable from entry via normal edges must stay
    // materialized — regardless of class.
    let reachable: HashSet<BlockId> = reverse_postorder(f).into_iter().collect();
    let mut read_in_unreachable: HashSet<SlotId> = HashSet::new();
    for (idx, block) in f.blocks.iter().enumerate() {
        if reachable.contains(&BlockId(idx as u32)) {
            continue;
        }
        for op in &block.body {
            if let Op::SlotLoad { slot, .. } = op {
                read_in_unreachable.insert(*slot);
            }
        }
    }

    // I9 gate: GP slots whose live range crosses a GC-safepoint op
    // MUST stay materialized — Beagle's GC scans frame slots, not
    // registers. A promoted GP slot whose value lives in a register
    // across a Call is invisible to GC and (under compacting GC) goes
    // stale when the heap object is relocated. FP values are never
    // tagged heap pointers so they're exempt.
    //
    // See `docs/SSA_ARCHITECTURE.md` §I9 and `src/cfg/gc_safety.rs`.
    let log_mem2reg = std::env::var("BEAGLE_SSA_LOG_MEM2REG").is_ok();
    let mut promotable: HashSet<SlotId> = HashSet::new();
    let mut rejected: Vec<SlotId> = Vec::new();
    for s in &read_filtered {
        let class = slot_class.get(s).copied().unwrap_or(RegClass::Gp);
        let safe = !read_in_unreachable.contains(s)
            && (class == RegClass::Fp || crate::cfg::gc_safety::slot_is_gc_safe_to_promote(f, *s));
        if safe {
            promotable.insert(*s);
        } else {
            rejected.push(*s);
        }
    }
    if log_mem2reg {
        eprintln!(
            "[mem2reg] {} candidates={} promoted={} rejected_i9={:?}",
            f.debug_name.as_deref().unwrap_or("<anon>"),
            read_filtered.len(),
            promotable.len(),
            rejected,
        );
    }
    if promotable.is_empty() {
        return;
    }

    let rpo = reverse_postorder(f);
    let idom = compute_idoms(f, &rpo);
    let df = compute_dominance_frontiers(f, &idom);
    let dom_tree = dominator_tree_children(&idom);

    // Pruned SSA: per-block live-in set of promotable slots. A φ is placed
    // at an IDF block only if the slot is live-in there; otherwise it is a
    // dead φ that pins its incoming value live across the edge for nothing,
    // exploding register pressure (a loop header is in the IDF of every
    // slot written in the loop). See `docs/SSA_PRESSURE_EXPLOSION.md`.
    // `BEAGLE_SSA_NO_PRUNE=1` reverts to minimal SSA for A/B.
    let slot_live_in = compute_slot_live_in(f, &rpo, &promotable);
    let prune = std::env::var("BEAGLE_SSA_NO_PRUNE").is_err();

    // Phase 1: insert phi-style block params at IDF of write sites.
    let mut slot_phi_param: HashMap<(SlotId, BlockId), VReg> = HashMap::new();
    let mut phis_at_block: HashMap<BlockId, Vec<SlotId>> = HashMap::new();
    for &slot in &promotable {
        let write_blocks: HashSet<BlockId> = slot_writes[&slot].iter().copied().collect();
        let idf = iterated_dominance_frontier(&write_blocks, &df);
        let class = slot_class.get(&slot).copied().unwrap_or(RegClass::Gp);
        // Sort IDF blocks deterministically so output is reproducible.
        let mut idf_sorted: Vec<BlockId> = idf.into_iter().collect();
        idf_sorted.sort();
        for block in idf_sorted {
            if prune
                && !slot_live_in
                    .get(&block)
                    .is_some_and(|live| live.contains(&slot))
            {
                continue;
            }
            let phi_vr = f.new_vreg(class);
            f.block_mut(block).params.push(phi_vr);
            slot_phi_param.insert((slot, block), phi_vr);
            phis_at_block.entry(block).or_default().push(slot);
        }
    }

    // Phase 2: rename via dominator-tree DFS, walking from entry.
    // Maintain a per-slot stack of "current SSA value"; loads read the
    // top of stack, stores push a new value, exits pop.
    let mut stacks: HashMap<SlotId, Vec<VReg>> = HashMap::new();

    // Synthesize a zero-init at the start of entry for each promoted slot
    // that is live-in at entry. Legacy zero-initializes frame slots, so a
    // read-before-write must see 0 — but only slots actually read before
    // written on some path from entry can observe that. A slot written
    // before any read on every path never observes the zero, so its init
    // is dead; skipping it removes the instruction (and seeds nothing —
    // its rename stack is filled by the dominating store, which by that
    // same liveness fact precedes every read, so the stack is never empty
    // at a read or a phi-arg wiring). Uses slot liveness, not read counts.
    let entry_live = slot_live_in.get(&f.entry);
    let mut entry_inits: Vec<Op> = Vec::new();
    for &slot in &promotable {
        if !entry_live.is_some_and(|live| live.contains(&slot)) {
            continue;
        }
        let class = slot_class.get(&slot).copied().unwrap_or(RegClass::Gp);
        let init_vr = f.new_vreg(class);
        entry_inits.push(Op::ConstRawValue {
            dst: init_vr,
            value: 0,
        });
        stacks.entry(slot).or_default().push(init_vr);
    }
    let existing_entry_body = std::mem::take(&mut f.block_mut(f.entry).body);
    let mut new_entry_body = entry_inits;
    new_entry_body.extend(existing_entry_body);
    f.block_mut(f.entry).body = new_entry_body;

    rename_block(
        f,
        f.entry,
        &dom_tree,
        &promotable,
        &phis_at_block,
        &slot_phi_param,
        &mut stacks,
    );
}

/// Walk the dominator tree rooted at `block`, rewriting promoted-slot
/// ops and filling in successor block-param args.
#[allow(clippy::too_many_arguments)]
fn rename_block(
    f: &mut CfgFunction,
    block: BlockId,
    dom_tree: &HashMap<BlockId, Vec<BlockId>>,
    promotable: &HashSet<SlotId>,
    phis_at_block: &HashMap<BlockId, Vec<SlotId>>,
    slot_phi_param: &HashMap<(SlotId, BlockId), VReg>,
    stacks: &mut HashMap<SlotId, Vec<VReg>>,
) {
    let mut pushes_this_block: HashMap<SlotId, usize> = HashMap::new();

    // 1. The block's own phi-params become the new "current values" for
    //    their slots on entry.
    if let Some(slots) = phis_at_block.get(&block) {
        for slot in slots {
            let phi_vr = slot_phi_param[&(*slot, block)];
            stacks.entry(*slot).or_default().push(phi_vr);
            *pushes_this_block.entry(*slot).or_insert(0) += 1;
        }
    }

    // 2. Rewrite the block body. Promoted SlotLoad → Move from top of
    //    stack; promoted SlotStore → drop the op and push src onto the
    //    stack. Non-promoted slots and non-slot ops pass through.
    let body = std::mem::take(&mut f.block_mut(block).body);
    let mut new_body: Vec<Op> = Vec::with_capacity(body.len());
    for op in body {
        match op {
            Op::SlotLoad { dst, slot } if promotable.contains(&slot) => {
                let top = stacks
                    .get(&slot)
                    .and_then(|s| s.last())
                    .copied()
                    .unwrap_or_else(|| {
                        // Read-before-write on the entry path. The
                        // legacy IR shouldn't emit this, but if it does
                        // we surface it loudly per the no-stubs rule.
                        panic!(
                            "mem2reg: SlotLoad of uninitialized slot {:?} \
                             in block {:?} of {:?}",
                            slot, block, f.debug_name
                        );
                    });
                new_body.push(Op::Move { dst, src: top });
            }
            Op::SlotStore { slot, src } if promotable.contains(&slot) => {
                stacks.entry(slot).or_default().push(src);
                *pushes_this_block.entry(slot).or_insert(0) += 1;
                // Op is dropped (no push to new_body).
            }
            other => new_body.push(other),
        }
    }
    f.block_mut(block).body = new_body;

    // 3. For each successor, append phi-args to the terminator's
    //    outgoing edge — in the same order the phi-params were inserted
    //    on the successor block (phis_at_block preserves insertion
    //    order).
    let succs = f.block(block).terminator.successors();
    for succ in succs {
        if let Some(slots) = phis_at_block.get(&succ) {
            for slot in slots {
                let top = stacks
                    .get(slot)
                    .and_then(|s| s.last())
                    .copied()
                    .unwrap_or_else(|| {
                        panic!(
                            "mem2reg: no value for slot {:?} flowing into \
                             phi at block {:?} from block {:?} of {:?}",
                            slot, succ, block, f.debug_name
                        );
                    });
                push_edge_arg(&mut f.block_mut(block).terminator, succ, top);
            }
        }
    }

    // 4. Recurse on dominator-tree children before popping.
    if let Some(children) = dom_tree.get(&block) {
        let children = children.clone();
        for child in children {
            rename_block(
                f,
                child,
                dom_tree,
                promotable,
                phis_at_block,
                slot_phi_param,
                stacks,
            );
        }
    }

    // 5. Pop everything pushed in this block. Restores the stack so
    //    sibling subtrees see the pre-entry state.
    for (slot, count) in pushes_this_block {
        for _ in 0..count {
            stacks
                .get_mut(&slot)
                .expect("slot stack must exist if we pushed")
                .pop();
        }
    }
}

/// Append `vr` to the outgoing-args list of the edge from this
/// terminator to `succ`. If the terminator has multiple edges to the
/// same successor (e.g. `Branch { t_target == f_target }`), we push to
/// the first matching one — which is correct because the args come from
/// a single source-block stack-top and both edges receive the same
/// value.
fn push_edge_arg(term: &mut Terminator, succ: BlockId, vr: VReg) {
    match term {
        Terminator::Jump { target, args } if *target == succ => args.push(vr),
        Terminator::Branch {
            t_target,
            t_args,
            f_target,
            f_args,
            ..
        } => {
            if *t_target == succ {
                t_args.push(vr);
            } else if *f_target == succ {
                f_args.push(vr);
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
                fall_args.push(vr);
            } else if *bail == succ {
                bail_args.push(vr);
            }
        }
        Terminator::Throw {
            resume,
            resume_args,
            ..
        } if *resume == succ => resume_args.push(vr),
        _ => {}
    }
}

/// `(slot → blocks-that-write-it, slot → read-count)`. Used by the
/// profitability gate and the phi-placement worklist.
fn collect_slot_sites(f: &CfgFunction) -> (HashMap<SlotId, Vec<BlockId>>, HashMap<SlotId, usize>) {
    let mut writes: HashMap<SlotId, Vec<BlockId>> = HashMap::new();
    let mut reads: HashMap<SlotId, usize> = HashMap::new();
    for (idx, block) in f.blocks.iter().enumerate() {
        let bid = BlockId(idx as u32);
        for op in &block.body {
            match op {
                Op::SlotStore { slot, .. } => writes.entry(*slot).or_default().push(bid),
                Op::SlotLoad { slot, .. } => *reads.entry(*slot).or_insert(0) += 1,
                _ => {}
            }
        }
    }
    (writes, reads)
}

/// Per-block **live-in set of promotable slots** — a standard backward
/// live-variable dataflow keyed on `SlotId`. A slot is live-in at `B` if
/// some path from `B` reaches a `SlotLoad` of it before a `SlotStore`.
/// Turns minimal SSA into **pruned SSA** (see the caller). Restricted to
/// `promotable` so the live sets stay small.
fn compute_slot_live_in(
    f: &CfgFunction,
    rpo: &[BlockId],
    promotable: &HashSet<SlotId>,
) -> HashMap<BlockId, HashSet<SlotId>> {
    // gen[B] = slots loaded before any store in B (upward-exposed);
    // kill[B] = slots stored in B.
    let mut gen_set: HashMap<BlockId, HashSet<SlotId>> = HashMap::new();
    let mut kill: HashMap<BlockId, HashSet<SlotId>> = HashMap::new();
    let mut live_in: HashMap<BlockId, HashSet<SlotId>> = HashMap::new();
    for (idx, block) in f.blocks.iter().enumerate() {
        let bid = BlockId(idx as u32);
        let mut g: HashSet<SlotId> = HashSet::new();
        let mut k: HashSet<SlotId> = HashSet::new();
        for op in &block.body {
            match op {
                Op::SlotLoad { slot, .. } if promotable.contains(slot) => {
                    if !k.contains(slot) {
                        g.insert(*slot);
                    }
                }
                Op::SlotStore { slot, .. } if promotable.contains(slot) => {
                    k.insert(*slot);
                }
                _ => {}
            }
        }
        gen_set.insert(bid, g);
        kill.insert(bid, k);
        live_in.insert(bid, HashSet::new());
    }

    // Backward fixpoint: live_in[B] = gen[B] ∪ (⋃succ live_in[succ] − kill[B]).
    let order: Vec<BlockId> = rpo.iter().rev().copied().collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &bid in &order {
            let mut new_in = gen_set[&bid].clone();
            for succ in f.block(bid).terminator.successors() {
                if let Some(succ_in) = live_in.get(&succ) {
                    for s in succ_in {
                        if !kill[&bid].contains(s) {
                            new_in.insert(*s);
                        }
                    }
                }
            }
            if new_in != live_in[&bid] {
                live_in.insert(bid, new_in);
                changed = true;
            }
        }
    }
    live_in
}

/// Infer the `RegClass` of each promotable slot by looking at the class
/// of its `SlotLoad` destinations and `SlotStore` sources. They should
/// all agree — if they don't, default to GP and let the verifier flag
/// the inconsistency at the use sites.
fn infer_slot_classes(f: &CfgFunction, promotable: &HashSet<SlotId>) -> HashMap<SlotId, RegClass> {
    let mut classes: HashMap<SlotId, RegClass> = HashMap::new();
    for block in &f.blocks {
        for op in &block.body {
            match op {
                Op::SlotLoad { dst, slot } if promotable.contains(slot) => {
                    classes.entry(*slot).or_insert(dst.class);
                }
                Op::SlotStore { slot, src } if promotable.contains(slot) => {
                    classes.entry(*slot).or_insert(src.class);
                }
                _ => {}
            }
        }
    }
    classes
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CfgFunction, Op, RegClass, SlotId, Terminator};
    use crate::ir::Condition;

    /// `let x = 1; x + x` collapsed into one block: two reads of slot 0,
    /// one write. Should promote — reads >= 2.
    #[test]
    fn single_block_two_reads_promote() {
        let mut f = CfgFunction::new(Some("two_reads".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let stored = f.new_vreg(RegClass::Gp);
        let loaded1 = f.new_vreg(RegClass::Gp);
        let loaded2 = f.new_vreg(RegClass::Gp);
        let sum = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: stored,
        });
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: loaded1,
            slot: SlotId(0),
        });
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: loaded2,
            slot: SlotId(0),
        });
        f.block_mut(entry).body.push(Op::AddInt {
            dst: sum,
            lhs: loaded1,
            rhs: loaded2,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: sum };

        promote_slots(&mut f);

        // Slot 0 is stored before it is read → not live-in at entry → no
        // synthetic zero-init is emitted (it would be dead). SlotStore
        // dropped; SlotLoads become Moves from the stored value. Body is
        // 2 Moves + AddInt.
        let body = &f.block(entry).body;
        assert_eq!(
            body.len(),
            3,
            "2 Moves + AddInt; store dropped, dead init elided: {:?}",
            body
        );
        assert!(matches!(body[0], Op::Move { src, .. } if src == stored));
        assert!(matches!(body[1], Op::Move { src, .. } if src == stored));
        assert!(matches!(body[2], Op::AddInt { .. }));
    }

    /// Single read: profitability gate skips promotion.
    #[test]
    fn single_read_is_promoted_at_default_gate() {
        let mut f = CfgFunction::new(Some("one_read".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let stored = f.new_vreg(RegClass::Gp);
        let loaded = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: stored,
        });
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: loaded,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: loaded };

        promote_slots(&mut f);

        // The default profitability gate is now `>= 1` read, so even a
        // single-read slot is promoted: its load becomes a Move from the
        // stored value and the store is dropped. (Set
        // `BEAGLE_SSA_PROMOTE_MIN_READS=2` for the old conservative gate.)
        let body = &f.block(entry).body;
        assert!(
            body.iter()
                .any(|op| matches!(op, Op::Move { dst, src } if *dst == loaded && *src == stored)),
            "single-read slot should promote to a Move at the >=1 default: {:?}",
            body
        );
        assert!(
            !body
                .iter()
                .any(|op| matches!(op, Op::SlotLoad { .. } | Op::SlotStore { .. })),
            "promoted slot's load/store should be gone: {:?}",
            body
        );
    }

    /// Diamond: slot is written on both branches, read at the join.
    /// Cytron places a phi-param at the join; rename pass passes the
    /// correct value from each branch via the terminator args.
    #[test]
    fn diamond_writes_promote_with_join_phi() {
        // entry (Branch using arg) → then (writes slot 0 = v_then)
        //                          → else (writes slot 0 = v_else)
        // join (loads slot 0 twice → reads = 2, promotable) → Ret
        let mut f = CfgFunction::new(Some("diamond".into()), 1);
        let entry = f.new_block();
        let then_b = f.new_block();
        let else_b = f.new_block();
        let join = f.new_block();
        f.entry = entry;

        let arg = f.new_vreg(RegClass::Gp);
        let v_then = f.new_vreg(RegClass::Gp);
        let v_else = f.new_vreg(RegClass::Gp);
        let load_a = f.new_vreg(RegClass::Gp);
        let load_b = f.new_vreg(RegClass::Gp);
        let sum = f.new_vreg(RegClass::Gp);

        // Entry takes `arg` as a function-arg block param — gives us a
        // defined VReg without needing to touch slot 0 first.
        f.block_mut(entry).params.push(arg);
        f.block_mut(entry).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: arg,
            rhs: arg,
            t_target: then_b,
            t_args: vec![],
            f_target: else_b,
            f_args: vec![],
        };
        f.block_mut(then_b).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: v_then,
        });
        f.block_mut(then_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        f.block_mut(else_b).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: v_else,
        });
        f.block_mut(else_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        f.block_mut(join).body.push(Op::SlotLoad {
            dst: load_a,
            slot: SlotId(0),
        });
        f.block_mut(join).body.push(Op::SlotLoad {
            dst: load_b,
            slot: SlotId(0),
        });
        f.block_mut(join).body.push(Op::AddInt {
            dst: sum,
            lhs: load_a,
            rhs: load_b,
        });
        f.block_mut(join).terminator = Terminator::Ret { value: sum };
        // Wire predecessors.
        f.block_mut(then_b).predecessors.push(entry);
        f.block_mut(else_b).predecessors.push(entry);
        f.block_mut(join).predecessors.push(then_b);
        f.block_mut(join).predecessors.push(else_b);

        promote_slots(&mut f);

        // Slot 0 has 3 reads (entry + 2 in join) and 2 writes
        // (then + else). It's promotable. IDF of {then, else} = {join},
        // so join gets a phi-param.
        let join_params = &f.block(join).params;
        assert_eq!(
            join_params.len(),
            1,
            "join should have 1 phi-param for promoted slot, got {:?}",
            join_params
        );

        // The two loads in join should now be Moves from the phi-param.
        let phi = join_params[0];
        let body = &f.block(join).body;
        assert!(matches!(body[0], Op::Move { src, .. } if src == phi));
        assert!(matches!(body[1], Op::Move { src, .. } if src == phi));

        // The branches' Jumps now pass their stored value as the
        // outgoing arg to join.
        match &f.block(then_b).terminator {
            Terminator::Jump { args, .. } => {
                assert_eq!(
                    args,
                    &vec![v_then],
                    "then branch passes v_then to join's phi"
                );
            }
            other => panic!("expected Jump in then_b, got {:?}", other),
        }
        match &f.block(else_b).terminator {
            Terminator::Jump { args, .. } => {
                assert_eq!(
                    args,
                    &vec![v_else],
                    "else branch passes v_else to join's phi"
                );
            }
            other => panic!("expected Jump in else_b, got {:?}", other),
        }
    }
}
