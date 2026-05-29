//! CFG verifier. Enforces the architectural invariants from
//! `docs/SSA_ARCHITECTURE.md`. A verifier failure aborts the compile in
//! debug builds and under `BEAGLE_SSA_VERIFY=1`.
//!
//! Each check corresponds to one or more named invariants:
//!
//! | Function                  | Invariants enforced |
//! |---------------------------|---------------------|
//! | `check_predecessors`      | I1 (CFG integrity)  |
//! | `check_arg_arity`         | I3 (block params)   |
//! | `check_vreg_classes`      | I4 (RegClass)       |
//! | `check_no_critical_edges` | I2                  |
//! | `check_dominance`         | I5                  |
//! | `check_gc_slot_regions`   | I9 (slot regions)   |
//!
//! I1's "exactly one terminator per block" is enforced by the type system
//! (`Block::terminator` is a single `Terminator`, not a `Vec`). I6 (locals
//! stay in slots) is enforced at construction time, not here. I7 (per-call
//! clobber model) and I8 (tail self-call → Jump-to-entry) are enforced
//! in Phase 1 once the construction and call op variants exist; they have
//! no surface area to check in Phase 0.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::cfg::dom::{compute_idoms, compute_reachable, dominates, reverse_postorder};
use crate::cfg::{BlockId, CfgFunction, Terminator, VReg};

/// A verifier failure. Each variant maps to an invariant; the message is
/// what shows up when the verifier aborts the compile.
#[derive(Debug, Clone)]
pub enum VerifyError {
    /// I1: predecessors list out of sync with the terminator graph.
    PredecessorMismatch {
        block: BlockId,
        expected: Vec<BlockId>,
        got: Vec<BlockId>,
    },
    /// I2: a critical edge survived a pass that should have split it.
    CriticalEdge { from: BlockId, to: BlockId },
    /// I3: a Jump / Branch / Throw's arg count doesn't match the target
    /// block's param count.
    ArgArityMismatch {
        from: BlockId,
        to: BlockId,
        expected: usize,
        got: usize,
    },
    /// I4: a VReg referenced anywhere in the function has no registered
    /// class, or its class disagrees with the registry.
    RegClassError { vreg: VReg, msg: String },
    /// I5: a use of a VReg is not dominated by its def. Reports the def
    /// block and the use block; the verifier finds the first such use it
    /// encounters and stops.
    UseNotDominated {
        vreg: VReg,
        def_block: BlockId,
        use_block: BlockId,
    },
    /// I5 (intra-block): a use appears before its def in the same block.
    UseBeforeDef {
        vreg: VReg,
        block: BlockId,
        def_pos: usize,
        use_pos: usize,
    },
    /// Reached a block whose terminator is `Unreachable` but which has
    /// predecessors and is therefore on the reachable flow graph. The
    /// construction code should have filled this in before invoking the
    /// verifier.
    UnfilledTerminator { block: BlockId },
    /// I9 (slot-region split): a slot is used in a way that breaks the
    /// scanned/unscanned partition — an FP value in a GC-scanned slot
    /// (would be misread as a heap pointer), a GP value in an unscanned
    /// slot (the GC wouldn't see the pointer), a slot used with two
    /// different classes, or an out-of-range region index.
    SlotRegionError {
        slot: crate::cfg::SlotId,
        msg: String,
    },
}

/// Return the first VReg that is **used but never defined** anywhere in
/// the function (as a block param, body-op def, or terminator def), or
/// `None` if every use has a def.
///
/// This is the narrowest, false-positive-free slice of the dominance
/// check: a use with no def at all is always a malformed CFG (a
/// build/lowering bug), regardless of dominance subtleties around
/// soft-edge handler blocks. `compile_via_ssa` calls this and bails to
/// the legacy pipeline rather than emit a read of an undefined register
/// (which the spec mandates: a verifier failure falls back to legacy).
/// Surfaced by `variadic_recursive_test` once Phase 2's larger pool
/// stopped `count_items` from bailing on pressure — its variadic
/// prologue leaves the rest-array vreg undefined in the CFG.
pub fn first_undefined_use(f: &CfgFunction) -> Option<VReg> {
    use std::collections::HashSet;

    let mut defined: HashSet<VReg> = HashSet::new();
    for block in &f.blocks {
        for &p in &block.params {
            defined.insert(p);
        }
        for op in &block.body {
            for d in op.defs() {
                defined.insert(d);
            }
        }
        for d in block.terminator.defs() {
            defined.insert(d);
        }
    }

    for block in &f.blocks {
        for op in &block.body {
            for u in op.uses() {
                if !defined.contains(&u) {
                    return Some(u);
                }
            }
        }
        for u in block.terminator.uses() {
            if !defined.contains(&u) {
                return Some(u);
            }
        }
    }
    None
}

/// Run every check, returning the first failure.
///
/// Call this at every phase boundary in the pipeline. Cheap enough to run
/// on every function under `debug_assertions`; gated by
/// `BEAGLE_SSA_VERIFY=1` in release.
pub fn verify(f: &CfgFunction) -> Result<(), VerifyError> {
    check_terminator_filled(f)?;
    check_predecessors(f)?;
    check_arg_arity(f)?;
    check_vreg_classes(f)?;
    check_no_critical_edges(f)?;
    check_dominance(f)?;
    check_gc_slot_regions(f)?;
    Ok(())
}

/// I9 slot-region split (Phase 1 of the runtime-parity plan). The frame
/// slot space is partitioned into a GC-scanned root region (GP values
/// that may be live across a safepoint) and an unscanned region (FP
/// spills + raw scratch). The frame header's `size` counts only the
/// root region, so a value's class must match its slot's region:
///
/// - An FP value must live in an **unscanned** slot — a raw `f64` in a
///   scanned slot can be misread as a heap pointer (the documented
///   risk this split exists to remove).
/// - A GP value must live in a **root** slot — an unscanned slot is not
///   scanned, so the GC would miss the pointer.
/// - A single slot must not be used with two different classes.
/// - Region indices must be within the allocated counts.
fn check_gc_slot_regions(f: &CfgFunction) -> Result<(), VerifyError> {
    use crate::cfg::{Op, RegClass, SlotId};

    let mut seen: HashMap<SlotId, RegClass> = HashMap::new();

    let mut check = |slot: SlotId, class: RegClass| -> Result<(), VerifyError> {
        // Unscanned region bounds. The root region's upper bound
        // (`< num_slots`) is established by `build_cfg` from the legacy
        // local count, not re-checked here — hand-built test CFGs
        // routinely use root slot ids without bumping `num_slots`, and
        // that's harmless. The unscanned region is allocated only via
        // `alloc_unscanned_slot`, so its count is always exact and a
        // stray index there is a real spiller bug.
        if slot.is_unscanned() && slot.region_index() >= f.num_unscanned_slots {
            return Err(VerifyError::SlotRegionError {
                slot,
                msg: format!(
                    "unscanned slot index {} >= num_unscanned_slots {}",
                    slot.region_index(),
                    f.num_unscanned_slots
                ),
            });
        }
        // Class ↔ region.
        match class {
            RegClass::Fp if !slot.is_unscanned() => {
                return Err(VerifyError::SlotRegionError {
                    slot,
                    msg: "FP value stored in a GC-scanned slot (would be \
                          misread as a heap pointer); use an unscanned slot"
                        .into(),
                });
            }
            RegClass::Gp if slot.is_unscanned() => {
                return Err(VerifyError::SlotRegionError {
                    slot,
                    msg: "GP value stored in an unscanned slot (the GC would \
                          not see the pointer); use a root slot"
                        .into(),
                });
            }
            _ => {}
        }
        // Single class per slot.
        match seen.get(&slot) {
            Some(prev) if *prev != class => Err(VerifyError::SlotRegionError {
                slot,
                msg: format!("slot used with two classes: {:?} then {:?}", prev, class),
            }),
            _ => {
                seen.insert(slot, class);
                Ok(())
            }
        }
    };

    for block in &f.blocks {
        for op in &block.body {
            match op {
                Op::SlotStore { slot, src } => check(*slot, src.class)?,
                Op::SlotLoad { dst, slot } => check(*slot, dst.class)?,
                _ => {}
            }
        }
    }
    Ok(())
}

/// Every reachable block must have a real terminator. `Unreachable` is
/// allowed only for blocks that have no predecessors and are not the entry
/// (DCE will drop them).
fn check_terminator_filled(f: &CfgFunction) -> Result<(), VerifyError> {
    let reachable = compute_reachable(f);
    for (idx, block) in f.blocks.iter().enumerate() {
        let id = BlockId(idx as u32);
        if matches!(block.terminator, Terminator::Unreachable) && reachable.contains(&id) {
            return Err(VerifyError::UnfilledTerminator { block: id });
        }
    }
    Ok(())
}

/// Predecessors must equal the inverse of the terminator-successor graph.
fn check_predecessors(f: &CfgFunction) -> Result<(), VerifyError> {
    let mut expected: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for (idx, block) in f.blocks.iter().enumerate() {
        let from = BlockId(idx as u32);
        for succ in block.terminator.successors() {
            expected.entry(succ).or_default().push(from);
        }
    }
    for (idx, block) in f.blocks.iter().enumerate() {
        let id = BlockId(idx as u32);
        let mut got = block.predecessors.clone();
        got.sort();
        let mut want = expected.remove(&id).unwrap_or_default();
        want.sort();
        if got != want {
            return Err(VerifyError::PredecessorMismatch {
                block: id,
                expected: want,
                got,
            });
        }
    }
    Ok(())
}

/// Every Jump / Branch / Throw's arg count must equal the target block's
/// param count.
fn check_arg_arity(f: &CfgFunction) -> Result<(), VerifyError> {
    for (idx, block) in f.blocks.iter().enumerate() {
        let from = BlockId(idx as u32);
        match &block.terminator {
            Terminator::Jump { target, args } => {
                expect_arity(from, *target, args.len(), f.block(*target).params.len())?;
            }
            Terminator::Branch {
                t_target,
                t_args,
                f_target,
                f_args,
                ..
            } => {
                expect_arity(
                    from,
                    *t_target,
                    t_args.len(),
                    f.block(*t_target).params.len(),
                )?;
                expect_arity(
                    from,
                    *f_target,
                    f_args.len(),
                    f.block(*f_target).params.len(),
                )?;
            }
            Terminator::Throw {
                resume,
                resume_args,
                ..
            } => {
                expect_arity(
                    from,
                    *resume,
                    resume_args.len(),
                    f.block(*resume).params.len(),
                )?;
            }
            Terminator::InlineBranch {
                fall_through,
                fall_args,
                bail,
                bail_args,
                ..
            } => {
                expect_arity(
                    from,
                    *fall_through,
                    fall_args.len(),
                    f.block(*fall_through).params.len(),
                )?;
                expect_arity(from, *bail, bail_args.len(), f.block(*bail).params.len())?;
            }
            Terminator::Ret { .. } | Terminator::Unreachable => {}
        }
    }
    Ok(())
}

fn expect_arity(
    from: BlockId,
    to: BlockId,
    got: usize,
    expected: usize,
) -> Result<(), VerifyError> {
    if got == expected {
        Ok(())
    } else {
        Err(VerifyError::ArgArityMismatch {
            from,
            to,
            expected,
            got,
        })
    }
}

/// Every VReg referenced (as a def, use, block param, or terminator arg)
/// must have a class in the registry that matches the class stamped on the
/// VReg itself.
fn check_vreg_classes(f: &CfgFunction) -> Result<(), VerifyError> {
    let n = f.vreg_classes.len() as u32;
    let visit = |v: VReg| -> Result<(), VerifyError> {
        if v.index >= n {
            return Err(VerifyError::RegClassError {
                vreg: v,
                msg: format!("VReg index {} not registered", v.index),
            });
        }
        let stored = f.vreg_classes[v.index as usize];
        if stored != v.class {
            return Err(VerifyError::RegClassError {
                vreg: v,
                msg: format!(
                    "VReg carries class {:?} but registry has {:?}",
                    v.class, stored
                ),
            });
        }
        Ok(())
    };
    for block in &f.blocks {
        for &p in &block.params {
            visit(p)?;
        }
        for op in &block.body {
            for v in op.defs() {
                visit(v)?;
            }
            for v in op.uses() {
                visit(v)?;
            }
        }
        for v in block.terminator.uses() {
            visit(v)?;
        }
    }
    Ok(())
}

/// I2: no critical edges. An edge is critical when its source has >1
/// successors AND its target has >1 predecessors.
fn check_no_critical_edges(f: &CfgFunction) -> Result<(), VerifyError> {
    for (idx, block) in f.blocks.iter().enumerate() {
        let from = BlockId(idx as u32);
        let succs = block.terminator.successors();
        if succs.len() <= 1 {
            continue;
        }
        for succ in succs {
            if f.block(succ).predecessors.len() > 1 {
                return Err(VerifyError::CriticalEdge { from, to: succ });
            }
        }
    }
    Ok(())
}

/// I5: every use of a VReg is dominated by its def.
///
/// Intra-block check: in the same block, def position < use position.
/// Cross-block check: the def's block must dominate the use's block. The
/// dominator tree is computed with Cooper/Harvey/Kennedy iterative
/// data-flow over reverse postorder.
///
/// Block params count as definitions at position 0 of their block; values
/// passed as terminator args count as uses at the *end* of the
/// predecessor block.
fn check_dominance(f: &CfgFunction) -> Result<(), VerifyError> {
    if f.blocks.is_empty() {
        return Ok(());
    }
    let rpo = reverse_postorder(f);
    let idom = compute_idoms(f, &rpo);

    // Map every VReg to its def site: (block, position). Block-param defs
    // land at position 0; body-op defs at position (op_index + 1) so they
    // are strictly greater than the param position.
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
        // Terminator-defined VRegs (e.g. InlineBranch's dst) are defined
        // at the end of the source block (`body.len() + 1`). They are
        // live only on the fall-through path; the bail path is responsible
        // for not reading them. The verifier flags any use that isn't
        // dominated by this block.
        for d in block.terminator.defs() {
            def_site.insert(d, (bid, block.body.len() + 1));
        }
    }

    // Walk every use site and check dominance.
    for (idx, block) in f.blocks.iter().enumerate() {
        let use_block = BlockId(idx as u32);
        for (i, op) in block.body.iter().enumerate() {
            let use_pos = i + 1;
            for u in op.uses() {
                check_def_dominates_use(&def_site, &idom, u, use_block, use_pos)?;
            }
        }
        // Terminator uses happen at end-of-block. For dominance purposes
        // they live in `use_block` at a position larger than every body
        // op's def_pos.
        let term_pos = block.body.len() + 1;
        for u in block.terminator.uses() {
            check_def_dominates_use(&def_site, &idom, u, use_block, term_pos)?;
        }
    }
    Ok(())
}

fn check_def_dominates_use(
    def_site: &HashMap<VReg, (BlockId, usize)>,
    idom: &HashMap<BlockId, BlockId>,
    vreg: VReg,
    use_block: BlockId,
    use_pos: usize,
) -> Result<(), VerifyError> {
    let &(def_block, def_pos) = def_site
        .get(&vreg)
        .ok_or_else(|| VerifyError::RegClassError {
            vreg,
            msg: "use of VReg with no def site".to_string(),
        })?;
    if def_block == use_block {
        // Same-position def/use is legal: an op that both reads and
        // writes the same VReg (in-place arithmetic, in-place tag
        // manipulation, or InlineBranch where dst equals lhs/rhs)
        // semantically reads the operand first, then writes the result.
        // Only flag strict use-before-def.
        if def_pos > use_pos {
            return Err(VerifyError::UseBeforeDef {
                vreg,
                block: use_block,
                def_pos,
                use_pos,
            });
        }
        return Ok(());
    }
    if dominates(idom, def_block, use_block) {
        Ok(())
    } else {
        Err(VerifyError::UseNotDominated {
            vreg,
            def_block,
            use_block,
        })
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CfgFunction, Op, RegClass, SlotId, Terminator};
    use crate::ir::Condition;

    fn fresh(name: &str) -> CfgFunction {
        CfgFunction::new(Some(name.to_string()), 0)
    }

    /// A function with a single block, one SlotLoad def, and a Ret of
    /// that def. Should verify cleanly.
    #[test]
    fn single_block_load_and_ret_verifies() {
        let mut f = fresh("single_block");
        let entry = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: v };
        verify(&f).expect("clean function should verify");
    }

    #[test]
    fn empty_function_verifies() {
        let f = fresh("empty");
        verify(&f).expect("empty function should verify");
    }

    #[test]
    fn unfilled_terminator_on_entry_is_caught() {
        let mut f = fresh("unfilled");
        let entry = f.new_block();
        f.entry = entry;
        let err = verify(&f).expect_err("Unreachable terminator on entry should fail");
        assert!(matches!(err, VerifyError::UnfilledTerminator { .. }));
    }

    #[test]
    fn predecessor_mismatch_caught() {
        // entry → tail, but tail.predecessors is left empty.
        let mut f = fresh("preds");
        let entry = f.new_block();
        let tail = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Jump {
            target: tail,
            args: vec![],
        };
        f.block_mut(tail).terminator = Terminator::Ret { value: v };
        // (skip filling tail.predecessors on purpose)
        let err = verify(&f).expect_err("missing predecessor should fail");
        assert!(matches!(err, VerifyError::PredecessorMismatch { .. }));
    }

    #[test]
    fn arity_mismatch_caught() {
        let mut f = fresh("arity");
        let entry = f.new_block();
        let tail = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        // entry passes 1 arg; tail declares 0 params.
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Jump {
            target: tail,
            args: vec![v],
        };
        f.block_mut(tail).predecessors.push(entry);
        f.block_mut(tail).terminator = Terminator::Ret { value: v };
        let err = verify(&f).expect_err("arg/param arity mismatch should fail");
        assert!(matches!(err, VerifyError::ArgArityMismatch { .. }));
    }

    #[test]
    fn critical_edge_caught() {
        //         entry
        //        /     \
        //      then    join  ← second pred coming up
        //        \     /
        //         join
        // Construct entry with two successors, join with two preds. The
        // entry→join edge is critical.
        let mut f = fresh("critical");
        let entry = f.new_block();
        let then = f.new_block();
        let join = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: v,
            rhs: v,
            t_target: then,
            t_args: vec![],
            f_target: join,
            f_args: vec![],
        };
        f.block_mut(then).predecessors.push(entry);
        f.block_mut(then).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        f.block_mut(join).predecessors.push(entry);
        f.block_mut(join).predecessors.push(then);
        f.block_mut(join).terminator = Terminator::Ret { value: v };
        let err = verify(&f).expect_err("critical edge should be flagged");
        assert!(matches!(err, VerifyError::CriticalEdge { .. }));
    }

    #[test]
    fn use_before_def_caught() {
        let mut f = fresh("use_before_def");
        let entry = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        // Ret(v) appears before any def of v (SlotLoad is below in body
        // order, but a Ret is the terminator — so we move the Ret-equivalent
        // intra-block by using two SlotStores reading the same VReg before
        // its load.
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: v,
        });
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: v };
        let err = verify(&f).expect_err("use before def should fail");
        assert!(matches!(err, VerifyError::UseBeforeDef { .. }));
    }

    #[test]
    fn unregistered_vreg_caught() {
        let mut f = fresh("unregistered");
        let entry = f.new_block();
        f.entry = entry;
        let bogus = VReg {
            index: 999,
            class: RegClass::Gp,
        };
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: bogus,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: bogus };
        let err = verify(&f).expect_err("unregistered VReg should fail");
        assert!(matches!(err, VerifyError::RegClassError { .. }));
    }

    #[test]
    fn diamond_with_split_critical_edges_verifies() {
        //         entry
        //        /     \
        //     then     else
        //        \     /
        //         join
        // No critical edges (both pred branches have a unique edge into
        // join because the entry sends one path each to then and else).
        let mut f = fresh("diamond");
        let entry = f.new_block();
        let then_b = f.new_block();
        let else_b = f.new_block();
        let join = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: v,
            rhs: v,
            t_target: then_b,
            t_args: vec![],
            f_target: else_b,
            f_args: vec![],
        };
        f.block_mut(then_b).predecessors.push(entry);
        f.block_mut(then_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        f.block_mut(else_b).predecessors.push(entry);
        f.block_mut(else_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        f.block_mut(join).predecessors.push(then_b);
        f.block_mut(join).predecessors.push(else_b);
        f.block_mut(join).terminator = Terminator::Ret { value: v };
        verify(&f).expect("split-edge diamond should verify");
    }

    /// An FP value spilled to a properly-allocated unscanned slot
    /// verifies; the GP/root and FP/unscanned mapping is the happy path.
    /// `g` and `x` are entry params (clean defs that dominate all uses).
    #[test]
    fn fp_in_unscanned_and_gp_in_root_verifies() {
        let mut f = fresh("regions_ok");
        let entry = f.new_block();
        f.entry = entry;
        let g = f.new_vreg(RegClass::Gp);
        let x = f.new_vreg(RegClass::Fp);
        f.block_mut(entry).params.push(g);
        f.block_mut(entry).params.push(x);
        let root = f.alloc_root_slot();
        let un = f.alloc_unscanned_slot();
        f.block_mut(entry)
            .body
            .push(Op::SlotStore { slot: root, src: g });
        f.block_mut(entry)
            .body
            .push(Op::SlotStore { slot: un, src: x });
        f.block_mut(entry).terminator = Terminator::Ret { value: g };
        verify(&f).expect("GP→root, FP→unscanned should verify");
    }

    /// An FP value in a GC-scanned (root) slot is the exact bug the
    /// region split exists to prevent — the GC would misread the f64.
    #[test]
    fn fp_in_scanned_slot_is_caught() {
        let mut f = fresh("fp_scanned");
        let entry = f.new_block();
        f.entry = entry;
        let x = f.new_vreg(RegClass::Fp);
        f.block_mut(entry).params.push(x);
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId::root(0),
            src: x,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: x };
        let err = verify(&f).expect_err("FP in scanned slot must fail");
        assert!(
            matches!(err, VerifyError::SlotRegionError { .. }),
            "{:?}",
            err
        );
    }

    /// A GP value in an unscanned slot — the GC wouldn't see the pointer.
    #[test]
    fn gp_in_unscanned_slot_is_caught() {
        let mut f = fresh("gp_unscanned");
        let entry = f.new_block();
        f.entry = entry;
        let g = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(g);
        let un = f.alloc_unscanned_slot();
        f.block_mut(entry)
            .body
            .push(Op::SlotStore { slot: un, src: g });
        f.block_mut(entry).terminator = Terminator::Ret { value: g };
        let err = verify(&f).expect_err("GP in unscanned slot must fail");
        assert!(
            matches!(err, VerifyError::SlotRegionError { .. }),
            "{:?}",
            err
        );
    }

    /// `first_undefined_use` flags a use of a vreg with no def, and
    /// returns None for a well-formed function.
    #[test]
    fn undefined_use_detected() {
        // Well-formed: param defines x, Ret uses it.
        let mut ok = fresh("ok");
        let e = ok.new_block();
        ok.entry = e;
        let x = ok.new_vreg(RegClass::Gp);
        ok.block_mut(e).params.push(x);
        ok.block_mut(e).terminator = Terminator::Ret { value: x };
        assert_eq!(first_undefined_use(&ok), None);

        // Malformed: Ret uses y, which is never defined.
        let mut bad = fresh("bad");
        let e2 = bad.new_block();
        bad.entry = e2;
        let y = bad.new_vreg(RegClass::Gp);
        bad.block_mut(e2).terminator = Terminator::Ret { value: y };
        assert_eq!(first_undefined_use(&bad), Some(y));
    }

    /// An unscanned slot id past `num_unscanned_slots` is a spiller bug.
    #[test]
    fn out_of_range_unscanned_slot_is_caught() {
        let mut f = fresh("oob_unscanned");
        let entry = f.new_block();
        f.entry = entry;
        let x = f.new_vreg(RegClass::Fp);
        f.block_mut(entry).params.push(x);
        // No alloc_unscanned_slot call, so num_unscanned_slots == 0.
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId::unscanned(0),
            src: x,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: x };
        let err = verify(&f).expect_err("out-of-range unscanned slot must fail");
        assert!(
            matches!(err, VerifyError::SlotRegionError { .. }),
            "{:?}",
            err
        );
    }
}
