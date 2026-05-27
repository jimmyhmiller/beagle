//! Lift cross-block legacy VRegs into stack slots.
//!
//! The legacy IR has VRegs whose def doesn't dominate every use — typically
//! defined in one branch of a control-flow structure and used after the
//! merge. Legacy regalloc handles this by spilling at the def and
//! reloading at the use; the dataflow is implicit in physical-register
//! state.
//!
//! SSA can't model this directly: every use must be dominated by its
//! def (**I5**). This pass converts each cross-block VReg into a stack
//! slot — its def is followed by a `SlotStore`, and every use is
//! preceded by a `SlotLoad`. After this, a subsequent mem2reg pass can
//! promote the slot back to SSA values + phi-style block params at
//! dominance frontiers (the spec's "Phase 2" path).
//!
//! Anti-patterns avoided:
//! - F10 (Phi op in SSA layer): we don't emit phis directly; mem2reg
//!   does it as block params on the next pass.
//! - F7 (silent stubs): a VReg whose lift requires extending past a
//!   terminator-only def site is handled correctly (SlotStore is
//!   inserted at the fall-through successor's start), not stubbed.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::cfg::dom::{compute_idoms, dominates, reverse_postorder};
use crate::cfg::{BlockId, CfgFunction, Op, SlotId, Terminator, VReg};

/// Lift every legacy VReg whose def doesn't dominate at least one of its
/// uses to a fresh stack slot. After this pass, mem2reg can promote the
/// slot back to SSA with proper phi-params.
pub fn lift_cross_block_vregs(f: &mut CfgFunction) {
    if f.blocks.is_empty() {
        return;
    }

    let rpo = reverse_postorder(f);
    let idom = compute_idoms(f, &rpo);

    let def_site = build_def_site_map(f);
    let needs_lift = find_lift_candidates(f, &def_site, &idom);
    if needs_lift.is_empty() {
        return;
    }

    let vreg_slot = allocate_slots_for_lifted(f, &needs_lift);

    // Combined pass: for each block, walk the original body once. For
    // each op: prepend SlotLoads + rewrite the op's lifted uses, emit
    // the op, then append SlotStores for any lifted defs. Doing it in
    // one pass avoids the bug where a separate "stores after defs"
    // step's SlotStore op would itself be picked up by the "rewrite
    // uses" step (the SlotStore's src IS a lifted VReg — the original
    // def whose value we're capturing).
    rewrite_body_and_terminator(f, &vreg_slot);

    // Terminator-defined VRegs (InlineBranchOp::dst): the SlotStore
    // goes at the start of the fall-through successor block. The def
    // is live only on the fall-through edge; critical-edge splitting
    // (Phase 1c) already guaranteed fall_through has source as its
    // sole pred, so source dominates fall_through and the VReg is in
    // scope there.
    insert_terminator_def_stores(f, &vreg_slot);
}

fn build_def_site_map(f: &CfgFunction) -> HashMap<VReg, (BlockId, usize)> {
    let mut def_site: HashMap<VReg, (BlockId, usize)> = HashMap::new();
    for (idx, block) in f.blocks.iter().enumerate() {
        let bid = BlockId(idx as u32);
        for &p in &block.params {
            def_site.insert(p, (bid, 0));
        }
        for (i, op) in block.body.iter().enumerate() {
            for d in op.defs() {
                def_site.insert(d, (bid, i + 1));
            }
        }
        for d in block.terminator.defs() {
            def_site.insert(d, (bid, block.body.len() + 1));
        }
    }
    def_site
}

fn find_lift_candidates(
    f: &CfgFunction,
    def_site: &HashMap<VReg, (BlockId, usize)>,
    idom: &HashMap<BlockId, BlockId>,
) -> HashSet<VReg> {
    let mut needs_lift: HashSet<VReg> = HashSet::new();
    let mut visit_use = |u: VReg, use_block: BlockId| {
        if let Some(&(def_b, _)) = def_site.get(&u) {
            if def_b != use_block && !dominates(idom, def_b, use_block) {
                needs_lift.insert(u);
            }
        }
    };
    for (idx, block) in f.blocks.iter().enumerate() {
        let bid = BlockId(idx as u32);
        for op in &block.body {
            for u in op.uses() {
                visit_use(u, bid);
            }
        }
        for u in block.terminator.uses() {
            visit_use(u, bid);
        }
    }
    needs_lift
}

fn allocate_slots_for_lifted(
    f: &mut CfgFunction,
    needs_lift: &HashSet<VReg>,
) -> HashMap<VReg, SlotId> {
    // Sort so slot allocation is deterministic.
    let mut sorted: Vec<VReg> = needs_lift.iter().copied().collect();
    sorted.sort();
    let mut map: HashMap<VReg, SlotId> = HashMap::new();
    for v in sorted {
        let slot = SlotId(f.num_slots);
        f.num_slots += 1;
        map.insert(v, slot);
    }
    map
}

/// For each block: walk the original body once, interleaving
/// SlotLoad-for-uses, the (rewritten) op, and SlotStore-for-defs.
/// Terminator handled similarly — uses get SlotLoads appended to the
/// end of body, then the terminator's uses are renamed. Terminator-
/// defined VRegs are stored at the successor block's start, NOT here
/// (see `insert_terminator_def_stores`).
fn rewrite_body_and_terminator(f: &mut CfgFunction, vreg_slot: &HashMap<VReg, SlotId>) {
    let num_blocks = f.blocks.len();
    for bid_idx in 0..num_blocks {
        let bid = BlockId(bid_idx as u32);

        let body = std::mem::take(&mut f.block_mut(bid).body);
        let mut new_body: Vec<Op> = Vec::with_capacity(body.len() * 2);
        for mut op in body {
            // Handle uses: SlotLoad each lifted use; rename the op's
            // uses to point at the load result. The op's DEFS keep
            // their original VReg names — those are the lifted values
            // we'll capture via SlotStore.
            let lifted_uses: Vec<VReg> = op
                .uses()
                .into_iter()
                .filter(|u| vreg_slot.contains_key(u))
                .collect();
            if !lifted_uses.is_empty() {
                let rename = build_rename_map(f, &lifted_uses, vreg_slot, &mut new_body);
                op.rename_uses(&rename);
            }

            // Emit the (possibly use-rewritten) op.
            let lifted_defs: Vec<(VReg, SlotId)> = op
                .defs()
                .into_iter()
                .filter_map(|d| vreg_slot.get(&d).map(|&s| (d, s)))
                .collect();
            new_body.push(op);

            // Capture any lifted defs into their slots, using the
            // ORIGINAL def VRegs (which are still what the op writes).
            for (vr, slot) in lifted_defs {
                new_body.push(Op::SlotStore { slot, src: vr });
            }
        }
        f.block_mut(bid).body = new_body;

        // Terminator uses.
        let term_uses = f.block(bid).terminator.uses();
        let lifted_term_uses: Vec<VReg> = term_uses
            .into_iter()
            .filter(|u| vreg_slot.contains_key(u))
            .collect();
        if !lifted_term_uses.is_empty() {
            let mut appended_loads: Vec<Op> = Vec::new();
            let rename = build_rename_map(f, &lifted_term_uses, vreg_slot, &mut appended_loads);
            f.block_mut(bid).body.extend(appended_loads);
            f.block_mut(bid).terminator.rename_uses(&rename);
        }
    }
}

/// Terminator-defined VRegs (InlineBranchOp::dst) live only on the
/// fall-through edge. The SlotStore goes at the start of fall_through;
/// after critical-edge splitting (Phase 1c), source dominates
/// fall_through and the VReg is in scope there.
fn insert_terminator_def_stores(f: &mut CfgFunction, vreg_slot: &HashMap<VReg, SlotId>) {
    let num_blocks = f.blocks.len();
    for bid_idx in 0..num_blocks {
        let bid = BlockId(bid_idx as u32);
        let term_defs = f.block(bid).terminator.defs();
        if term_defs.is_empty() {
            continue;
        }
        let fall_through = match &f.block(bid).terminator {
            Terminator::InlineBranch { fall_through, .. } => Some(*fall_through),
            _ => None,
        };
        let Some(ft) = fall_through else { continue };

        let mut prepend: Vec<Op> = Vec::new();
        for d in term_defs {
            if let Some(&slot) = vreg_slot.get(&d) {
                prepend.push(Op::SlotStore { slot, src: d });
            }
        }
        if !prepend.is_empty() {
            let existing = std::mem::take(&mut f.block_mut(ft).body);
            prepend.extend(existing);
            f.block_mut(ft).body = prepend;
        }
    }
}

/// For each lifted VReg in `lifted_uses`, allocate a fresh load-result
/// VReg, append a `SlotLoad` to `output_ops`, and record the rename
/// `original → new` in the returned map. Duplicates in `lifted_uses`
/// produce a single SlotLoad and a single map entry.
fn build_rename_map(
    f: &mut CfgFunction,
    lifted_uses: &[VReg],
    vreg_slot: &HashMap<VReg, SlotId>,
    output_ops: &mut Vec<Op>,
) -> HashMap<VReg, VReg> {
    let mut rename: HashMap<VReg, VReg> = HashMap::new();
    for &u in lifted_uses {
        if rename.contains_key(&u) {
            continue;
        }
        let slot = vreg_slot[&u];
        let new_vr = f.new_vreg(u.class);
        output_ops.push(Op::SlotLoad { dst: new_vr, slot });
        rename.insert(u, new_vr);
    }
    rename
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CfgFunction, Op, RegClass, Terminator};
    use crate::ir::Condition;

    /// `let r; if cond { r = a } else { r = b }; r + r` style — except
    /// expressed in legacy IR where `a` and `b` are SEPARATE VRegs
    /// defined in different branches, then both somehow used after the
    /// join. The legacy IR would do this via slot routing, but for the
    /// test we construct the SSA-illegal version: define V in the
    /// then-branch, use it after the join. Lift should add a SlotStore
    /// in the then-branch and a SlotLoad before the use.
    #[test]
    fn cross_block_vreg_is_lifted_to_slot() {
        let mut f = CfgFunction::new(Some("cross".into()), 0);
        let entry = f.new_block();
        let then_b = f.new_block();
        let else_b = f.new_block();
        let join = f.new_block();
        f.entry = entry;

        // Allocate VRegs.
        let arg = f.new_vreg(RegClass::Gp); // entry param
        let v = f.new_vreg(RegClass::Gp); // defined only in then-branch
        let zero = f.new_vreg(RegClass::Gp); // defined only in else-branch
        // (lifted too, used in join)

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
        f.block_mut(then_b).body.push(Op::AddInt {
            dst: v,
            lhs: arg,
            rhs: arg,
        });
        f.block_mut(then_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        f.block_mut(else_b).body.push(Op::ConstTaggedInt {
            dst: zero,
            value: 0,
        });
        f.block_mut(else_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        // Join uses v (from then-branch only) — this is the cross-block
        // case lift_vregs handles. Construct the (deliberately invalid
        // pre-lift) Ret of v.
        f.block_mut(join).terminator = Terminator::Ret { value: v };
        f.block_mut(then_b).predecessors.push(entry);
        f.block_mut(else_b).predecessors.push(entry);
        f.block_mut(join).predecessors.push(then_b);
        f.block_mut(join).predecessors.push(else_b);

        let slots_before = f.num_slots;
        lift_cross_block_vregs(&mut f);

        // A fresh slot should have been allocated for v.
        assert!(f.num_slots > slots_before, "lift should allocate >=1 slot");

        // then-branch should have AddInt + SlotStore.
        let then_body = &f.block(then_b).body;
        assert_eq!(then_body.len(), 2, "then has AddInt + SlotStore");
        assert!(matches!(then_body[0], Op::AddInt { .. }));
        assert!(matches!(then_body[1], Op::SlotStore { src, .. } if src == v));

        // join should have a prepended SlotLoad and a Ret of its dst.
        // Body has one SlotLoad; terminator references the load's dst.
        let join_body = &f.block(join).body;
        assert!(
            join_body.iter().any(|op| matches!(op, Op::SlotLoad { .. })),
            "join should have a SlotLoad"
        );
        match &f.block(join).terminator {
            Terminator::Ret { value } => assert_ne!(
                *value, v,
                "Ret's value should be the load-result VReg, not the original"
            ),
            other => panic!("expected Ret in join, got {:?}", other),
        }
    }

    /// End-to-end: lift + mem2reg on a synthetic loop produces a
    /// CFG that passes the verifier. Models the classic "loop-carried
    /// value" pattern: header reads V (live-in), body writes V, back-
    /// edge to header. Legacy IR is fine with this because the same
    /// VReg slot gets routed via physical registers; SSA needs lift to
    /// route through a slot, then mem2reg promotes to a phi-param at
    /// the header.
    #[test]
    fn loop_carried_vreg_verifies_after_lift_plus_mem2reg() {
        let mut f = CfgFunction::new(Some("loop_carry".into()), 0);
        let entry = f.new_block();
        let header = f.new_block();
        let body = f.new_block();
        let exit = f.new_block();
        f.entry = entry;

        // Initial value of V (produced in entry, used as the seed).
        let v_init = f.new_vreg(RegClass::Gp);
        // Loop-carried VReg V. Defined in body; read in header on each
        // iteration. body doesn't dominate header (header dominates
        // body) so this triggers the lift.
        let v = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: v_init,
            value: 0,
        });
        // Entry jumps to header with v_init as the seed. We model this
        // as a use of v_init in entry's terminator args — but header's
        // params get populated by mem2reg, not by lift. So this Jump
        // has empty args for now.
        f.block_mut(entry).terminator = Terminator::Jump {
            target: header,
            args: vec![],
        };
        // Header reads V (the cross-block USE).
        let read_in_header = f.new_vreg(RegClass::Gp);
        f.block_mut(header).body.push(Op::AddInt {
            dst: read_in_header,
            lhs: v,
            rhs: v_init,
        });
        // Header branches: continue to body or exit.
        f.block_mut(header).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: read_in_header,
            rhs: v_init,
            t_target: body,
            t_args: vec![],
            f_target: exit,
            f_args: vec![],
        };
        // Body defines V (a new iteration's value) and jumps back to
        // header. body's def of V doesn't dominate header → lift.
        f.block_mut(body).body.push(Op::AddInt {
            dst: v,
            lhs: read_in_header,
            rhs: v_init,
        });
        f.block_mut(body).terminator = Terminator::Jump {
            target: header,
            args: vec![],
        };
        f.block_mut(exit).terminator = Terminator::Ret {
            value: read_in_header,
        };
        // Predecessors.
        f.block_mut(header).predecessors.push(entry);
        f.block_mut(header).predecessors.push(body);
        f.block_mut(body).predecessors.push(header);
        f.block_mut(exit).predecessors.push(header);

        // Verify SHOULD fail with UseNotDominated before lift (V is read
        // in header, defined in body, body doesn't dominate header).
        let pre_err = crate::cfg::verify::verify(&f);
        assert!(
            matches!(
                pre_err,
                Err(crate::cfg::verify::VerifyError::UseNotDominated { .. })
            ),
            "expected UseNotDominated before lift, got {:?}",
            pre_err
        );

        // Run lift + mem2reg.
        lift_cross_block_vregs(&mut f);
        crate::cfg::mem2reg::promote_slots(&mut f);

        // Now verify SHOULD pass.
        crate::cfg::verify::verify(&f)
            .expect("after lift + mem2reg, loop-carried value should verify");
    }

    /// A VReg whose def dominates its only use shouldn't be lifted.
    #[test]
    fn fully_dominated_vreg_is_not_lifted() {
        let mut f = CfgFunction::new(Some("dom".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let arg = f.new_vreg(RegClass::Gp);
        let v = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(arg);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: v,
            lhs: arg,
            rhs: arg,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: v };

        let slots_before = f.num_slots;
        lift_cross_block_vregs(&mut f);
        assert_eq!(
            f.num_slots, slots_before,
            "no slot should be allocated for dominated uses"
        );
        // Body unchanged.
        let body = &f.block(entry).body;
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0], Op::AddInt { .. }));
    }
}
