//! Phase 3 SSA optimizations — runs after lift + mem2reg, before
//! regalloc would consume the CFG.
//!
//! Three interacting passes, run to fixpoint:
//!
//! 1. **Trivial-block-param elimination.** A block param P is trivial
//!    when every predecessor's outgoing arg-at-this-position is either
//!    a single common VReg V or P itself (a self-loop with V as the
//!    only external value). Trivial params are removed; uses of P
//!    rewritten to V; every predecessor's outgoing arg at the
//!    corresponding index dropped. Cleans up mem2reg's "phi-where-
//!    all-incoming-are-the-same" output and the recursive
//!    self-loop-but-also-V cases.
//!
//! 2. **Copy coalesce.** Every `Op::Move { dst, src }` becomes a rename
//!    `dst → src`. The rename is resolved transitively then applied to
//!    every use across the function (body ops + terminators). The Move
//!    ops are then removed; their old `dst` VRegs become dead.
//!
//! 3. **DCE.** Mark-and-sweep over pure ops: side-effecting ops, block
//!    params, and terminator uses are roots; an op whose def is in the
//!    live set propagates use-liveness; sweep removes pure ops whose
//!    defs are all dead.
//!
//! Each pass returns `bool` (whether it changed anything); the
//! orchestrator loops until none of them change.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::cfg::{BlockId, CfgFunction, Op, Terminator, VReg};

/// Run all Phase 3 passes to fixpoint. Wired into `build_cfg` after
/// mem2reg.
pub fn optimize(f: &mut CfgFunction) {
    let no_trivial = std::env::var("BEAGLE_SSA_NO_TRIVIAL").is_ok();
    let no_coalesce = std::env::var("BEAGLE_SSA_NO_COALESCE").is_ok();
    let no_dce = std::env::var("BEAGLE_SSA_NO_DCE").is_ok();
    let mut changed = true;
    while changed {
        changed = false;
        if !no_trivial {
            changed |= eliminate_trivial_block_params(f);
        }
        if !no_coalesce {
            changed |= coalesce_copies(f);
        }
        if !no_dce {
            changed |= dead_code_elimination(f);
        }
    }
}

// =========================================================================
// Trivial block param elimination
// =========================================================================

/// Remove every block param whose incoming arg from every predecessor
/// is the same VReg (ignoring self-loops). Returns true if any param
/// was eliminated.
pub fn eliminate_trivial_block_params(f: &mut CfgFunction) -> bool {
    let mut any_removed = false;
    let mut changed_pass = true;
    while changed_pass {
        changed_pass = false;
        // Snapshot block count; we don't add or remove blocks here.
        let num_blocks = f.blocks.len();
        for bid_idx in 0..num_blocks {
            let bid = BlockId(bid_idx as u32);
            let num_params = f.block(bid).params.len();
            // Iterate from highest index down so per-block param
            // removal doesn't shift the indices of params we haven't
            // checked yet.
            for param_idx in (0..num_params).rev() {
                let Some(replacement) = try_trivial_param(f, bid, param_idx) else {
                    continue;
                };
                let param_vreg = f.block(bid).params[param_idx];

                // 1. Remove the param from B.
                f.block_mut(bid).params.remove(param_idx);

                // 2. Drop the corresponding arg from each predecessor's
                //    outgoing terminator-args list.
                let preds: Vec<BlockId> = f.block(bid).predecessors.clone();
                for pred in preds {
                    remove_terminator_arg(&mut f.block_mut(pred).terminator, bid, param_idx);
                }

                // 3. Rewrite every use of the removed param to the
                //    replacement VReg.
                let mut rename: HashMap<VReg, VReg> = HashMap::new();
                rename.insert(param_vreg, replacement);
                apply_rename_function_wide(f, &rename);

                any_removed = true;
                changed_pass = true;
            }
        }
    }
    any_removed
}

/// Returns Some(V) if the param at `param_idx` of block `bid` is
/// trivial — i.e. every predecessor's contribution is either V or
/// the param itself (self-loop). Returns None if any predecessor
/// passes a different value or there are no predecessors (entry args).
fn try_trivial_param(f: &CfgFunction, bid: BlockId, param_idx: usize) -> Option<VReg> {
    // Entry block params are function arguments — sourced from the
    // calling convention, not from any CFG predecessor. Folding them
    // would lose the implicit "caller" arg. Skip even when the
    // visible predecessors (e.g. tail-recurse back-edge) all pass the
    // same value, because the initial call still needs the param.
    if bid == f.entry {
        return None;
    }
    let preds = &f.block(bid).predecessors;
    if preds.is_empty() {
        return None;
    }
    let param_vreg = f.block(bid).params[param_idx];
    let mut unique_external: Option<VReg> = None;
    for pred in preds {
        let arg = extract_terminator_arg(&f.block(*pred).terminator, bid, param_idx)?;
        if arg == param_vreg {
            continue; // self-loop — ignore for trivial check
        }
        match unique_external {
            None => unique_external = Some(arg),
            Some(u) if u == arg => continue,
            Some(_) => return None, // multiple distinct external values → not trivial
        }
    }
    unique_external
}

// =========================================================================
// Copy coalesce
// =========================================================================

/// Replace `Op::Move { dst, src }` with a rename `dst → src` applied
/// globally; delete the Move ops. Returns true if any Move was
/// removed.
///
/// **Exception 1: `Op::CompareAndSwap`'s in/out operand.** The legacy
/// `Instruction::CompareAndSwap`'s first operand serves as both the
/// "expected" input and the "old value at addr" output — ARM `CAS`
/// and x86 `LOCK CMPXCHG` mutate that register in place. The IR
/// builder marks this by emitting `v_temp = Move v_expected` right
/// before the CAS, so the CAS mutates `v_temp` rather than
/// `v_expected`. The subsequent `branch Equal v_temp v_expected`
/// (== "did CAS succeed?") relies on `v_temp != v_expected` as
/// distinct VRegs. Coalescing the Move would replace `v_temp` with
/// `v_expected` everywhere, turning the branch into the trivially-
/// true `Equal v_expected v_expected` and making every CAS appear
/// to succeed. Caught by `atom.bg`'s second `compare-and-swap!`
/// returning `true` (expected `false`).
///
/// **Exception 2: Moves whose src is an entry block param.** Entry
/// block params are pre-colored to arg-register positions (X0..X7
/// on ARM64) per the calling convention. The AST compiler emits a
/// prologue copy `new_v = Move arg_0` whose `new_v` gets a callee-
/// saved color, so the long-lived value lives in X19..X27 instead.
/// Coalescing that Move would replace `new_v` with `arg_0` everywhere
/// — and `arg_0`'s color is X0, a caller-saved register that any
/// subsequent call clobbers. Result: SIGSEGV when a function arg
/// is used after a call (e.g. `sort_timsort_test.bg`).
pub fn coalesce_copies(f: &mut CfgFunction) -> bool {
    // Build the set of VRegs that are used as a CAS in/out operand
    // (the `addr` field per Op::CompareAndSwap's confused naming —
    // semantically it's the in/out "expected_then_old" slot).
    let mut cas_in_out: HashSet<VReg> = HashSet::new();
    for block in &f.blocks {
        for op in &block.body {
            if let Op::CompareAndSwap { addr, .. } = op {
                cas_in_out.insert(*addr);
            }
        }
    }

    // Build the set of entry block param VRegs. Moves whose `src`
    // is one of these must stay so the bridge to callee-saved is
    // preserved (see "Exception 2" above).
    let entry_params: HashSet<VReg> = f.block(f.entry).params.iter().copied().collect();

    // Build raw rename map from every Move EXCEPT the two exception
    // categories above.
    let mut rename: HashMap<VReg, VReg> = HashMap::new();
    for block in &f.blocks {
        for op in &block.body {
            if let Op::Move { dst, src } = op
                && !cas_in_out.contains(dst)
                && !entry_params.contains(src)
            {
                rename.insert(*dst, *src);
            }
        }
    }
    if rename.is_empty() {
        return false;
    }

    // Resolve transitively: if rename[a] = b and rename[b] = c, then
    // rename[a] = c. Each lookup walks the chain to the final value.
    let resolved = resolve_transitive_rename(&rename);

    // Apply the rename function-wide.
    apply_rename_function_wide(f, &resolved);

    // Delete only the Moves we actually coalesced — i.e., those
    // whose dst is in the rename map. Preserves both exception
    // categories (CAS in/out, entry-param-bridge).
    let mut removed_any = false;
    for block in f.blocks.iter_mut() {
        let before = block.body.len();
        block.body.retain(|op| match op {
            Op::Move { dst, .. } => !rename.contains_key(dst),
            _ => true,
        });
        if block.body.len() != before {
            removed_any = true;
        }
    }
    removed_any
}

fn resolve_transitive_rename(raw: &HashMap<VReg, VReg>) -> HashMap<VReg, VReg> {
    let mut resolved: HashMap<VReg, VReg> = HashMap::new();
    for (&start, _) in raw {
        let mut cur = start;
        let mut seen: HashSet<VReg> = HashSet::new();
        seen.insert(cur);
        while let Some(&next) = raw.get(&cur) {
            if !seen.insert(next) {
                // Cycle (shouldn't happen for Moves in valid SSA, but
                // bail out gracefully if it does).
                break;
            }
            cur = next;
        }
        if cur != start {
            resolved.insert(start, cur);
        }
    }
    resolved
}

// =========================================================================
// DCE — mark-and-sweep over pure ops
// =========================================================================

/// Remove pure ops whose defs are unused. Side-effecting ops stay
/// regardless of whether their dst (if any) is used. Block params and
/// terminator-used VRegs are roots. Returns true if any op was
/// removed.
pub fn dead_code_elimination(f: &mut CfgFunction) -> bool {
    // Initial live set: every VReg used by a side-effecting op or a
    // terminator, plus every block param (they're "imported" and may be
    // observed via successor terminators in ways we can't trace
    // perfectly here; trivial-param elimination handles redundant ones
    // separately).
    let mut live: HashSet<VReg> = HashSet::new();
    for block in &f.blocks {
        for &p in &block.params {
            live.insert(p);
        }
        for op in &block.body {
            if op.has_side_effect() {
                for u in op.uses() {
                    live.insert(u);
                }
            }
        }
        for u in block.terminator.uses() {
            live.insert(u);
        }
    }

    // Fixpoint: an op whose def is live makes its uses live.
    let mut changed = true;
    while changed {
        changed = false;
        for block in &f.blocks {
            for op in &block.body {
                let defs = op.defs();
                if defs.iter().any(|d| live.contains(d)) {
                    for u in op.uses() {
                        if live.insert(u) {
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // Sweep: remove side-effect-free ops whose defs are all dead.
    let mut removed_any = false;
    for block in f.blocks.iter_mut() {
        let before = block.body.len();
        block.body.retain(|op| {
            if op.has_side_effect() {
                return true;
            }
            let defs = op.defs();
            if defs.is_empty() {
                // Side-effect-free op with no defs is pointless but
                // shouldn't exist in our enum — keep defensively.
                return true;
            }
            defs.iter().any(|d| live.contains(d))
        });
        if block.body.len() != before {
            removed_any = true;
        }
    }
    removed_any
}

// =========================================================================
// Shared helpers
// =========================================================================

/// Walk every op (body + terminators in every block) and apply the
/// rename map via `Op::rename_uses` / `Terminator::rename_uses`. Defs
/// are NOT touched (that would change SSA identity).
fn apply_rename_function_wide(f: &mut CfgFunction, rename: &HashMap<VReg, VReg>) {
    for block in f.blocks.iter_mut() {
        for op in block.body.iter_mut() {
            op.rename_uses(rename);
        }
        block.terminator.rename_uses(rename);
    }
}

/// Read the outgoing arg at `idx` for the edge `term → succ`. Returns
/// None if `term` has no matching edge.
fn extract_terminator_arg(term: &Terminator, succ: BlockId, idx: usize) -> Option<VReg> {
    match term {
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
        Terminator::Throw {
            resume,
            resume_args,
            ..
        } if *resume == succ => resume_args.get(idx).copied(),
        _ => None,
    }
}

/// Drop the outgoing arg at `idx` for the edge `term → succ` (if any).
/// If `term` has multiple edges to `succ` (e.g. `Branch t==f`), the
/// arg is dropped from each matching list — they must stay in lockstep.
fn remove_terminator_arg(term: &mut Terminator, succ: BlockId, idx: usize) {
    match term {
        Terminator::Jump { target, args } if *target == succ => {
            if idx < args.len() {
                args.remove(idx);
            }
        }
        Terminator::Branch {
            t_target,
            t_args,
            f_target,
            f_args,
            ..
        } => {
            if *t_target == succ && idx < t_args.len() {
                t_args.remove(idx);
            }
            if *f_target == succ && idx < f_args.len() {
                f_args.remove(idx);
            }
        }
        Terminator::InlineBranch {
            fall_through,
            fall_args,
            bail,
            bail_args,
            ..
        } => {
            if *fall_through == succ && idx < fall_args.len() {
                fall_args.remove(idx);
            }
            if *bail == succ && idx < bail_args.len() {
                bail_args.remove(idx);
            }
        }
        Terminator::Throw {
            resume,
            resume_args,
            ..
        } if *resume == succ => {
            if idx < resume_args.len() {
                resume_args.remove(idx);
            }
        }
        _ => {}
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

    /// `c = const; v = Move c; ret v`: coalesce removes the Move,
    /// folds v → c. Uses a non-entry-param Move src so the "preserve
    /// entry-param bridge" exception doesn't apply.
    #[test]
    fn coalesce_removes_move_op() {
        let mut f = CfgFunction::new(Some("coalesce".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let c = f.new_vreg(RegClass::Gp);
        let v_copy = f.new_vreg(RegClass::Gp);
        f.block_mut(entry)
            .body
            .push(Op::ConstTaggedInt { dst: c, value: 42 });
        f.block_mut(entry).body.push(Op::Move {
            dst: v_copy,
            src: c,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: v_copy };

        let changed = coalesce_copies(&mut f);
        assert!(changed, "coalesce should have removed a Move");
        assert_eq!(
            f.block(entry).body.len(),
            1,
            "Move op should be gone; only ConstTaggedInt left"
        );
        match &f.block(entry).terminator {
            Terminator::Ret { value } => assert_eq!(*value, c, "Ret should now use c directly"),
            _ => panic!("expected Ret"),
        }
    }

    /// Entry-param Move is preserved: `block(arg) { v = Move arg; ret v }`.
    /// The Move bridges the arg-reg color (X0..X7) to a callee-saved
    /// vreg; coalescing it would extend `arg`'s lifetime into regions
    /// where its arg register is clobbered by intervening calls.
    /// See the "Exception 2" comment in `coalesce_copies`.
    #[test]
    fn coalesce_preserves_entry_param_move() {
        let mut f = CfgFunction::new(Some("arg_bridge".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let arg = f.new_vreg(RegClass::Gp);
        let v_copy = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(arg);
        f.block_mut(entry).body.push(Op::Move {
            dst: v_copy,
            src: arg,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: v_copy };

        let changed = coalesce_copies(&mut f);
        assert!(!changed, "Move from entry param must stay");
        assert_eq!(
            f.block(entry).body.len(),
            1,
            "Move op should still be present"
        );
    }

    /// A trivial block param: join block where both predecessors pass
    /// the same value. Trivial-param elim removes the param, drops the
    /// arg from each pred, rewrites uses.
    #[test]
    fn trivial_block_param_eliminated() {
        let mut f = CfgFunction::new(Some("trivial".into()), 0);
        let entry = f.new_block();
        let then_b = f.new_block();
        let else_b = f.new_block();
        let join = f.new_block();
        f.entry = entry;

        let arg = f.new_vreg(RegClass::Gp);
        let common = f.new_vreg(RegClass::Gp);
        let phi = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).params.push(arg);
        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: common,
            value: 7,
        });
        f.block_mut(entry).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: arg,
            rhs: arg,
            t_target: then_b,
            t_args: vec![],
            f_target: else_b,
            f_args: vec![],
        };
        f.block_mut(then_b).terminator = Terminator::Jump {
            target: join,
            args: vec![common],
        };
        f.block_mut(else_b).terminator = Terminator::Jump {
            target: join,
            args: vec![common],
        };
        f.block_mut(join).params.push(phi);
        f.block_mut(join).terminator = Terminator::Ret { value: phi };
        f.block_mut(then_b).predecessors.push(entry);
        f.block_mut(else_b).predecessors.push(entry);
        f.block_mut(join).predecessors.push(then_b);
        f.block_mut(join).predecessors.push(else_b);

        let changed = eliminate_trivial_block_params(&mut f);
        assert!(changed, "expected trivial-param elimination to fire");
        assert!(
            f.block(join).params.is_empty(),
            "join's trivial param removed"
        );
        match &f.block(then_b).terminator {
            Terminator::Jump { args, .. } => {
                assert!(args.is_empty(), "then's outgoing arg dropped")
            }
            _ => panic!("expected Jump"),
        }
        match &f.block(else_b).terminator {
            Terminator::Jump { args, .. } => {
                assert!(args.is_empty(), "else's outgoing arg dropped")
            }
            _ => panic!("expected Jump"),
        }
        match &f.block(join).terminator {
            Terminator::Ret { value } => assert_eq!(*value, common, "Ret rewritten to use common"),
            _ => panic!("expected Ret"),
        }
    }

    /// A self-loop trivial param (loop-header where the body re-passes
    /// the same value): trivial-param elim still folds it because only
    /// one distinct external value exists.
    #[test]
    fn self_loop_trivial_param_eliminated() {
        let mut f = CfgFunction::new(Some("self_loop".into()), 0);
        let entry = f.new_block();
        let header = f.new_block();
        let exit = f.new_block();
        f.entry = entry;

        let arg = f.new_vreg(RegClass::Gp);
        let phi = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).params.push(arg);
        f.block_mut(entry).terminator = Terminator::Jump {
            target: header,
            args: vec![arg],
        };
        f.block_mut(header).params.push(phi);
        // header branches to itself or exits; back-edge passes phi.
        f.block_mut(header).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: phi,
            rhs: phi,
            t_target: header,
            t_args: vec![phi], // self-loop: pass phi unchanged
            f_target: exit,
            f_args: vec![],
        };
        f.block_mut(exit).terminator = Terminator::Ret { value: phi };
        f.block_mut(header).predecessors.push(entry);
        f.block_mut(header).predecessors.push(header);
        f.block_mut(exit).predecessors.push(header);

        let changed = eliminate_trivial_block_params(&mut f);
        assert!(
            changed,
            "self-loop trivial param should fold to entry's arg"
        );
        assert!(f.block(header).params.is_empty(), "phi removed");
        match &f.block(exit).terminator {
            Terminator::Ret { value } => assert_eq!(*value, arg, "exit Ret now uses entry's arg"),
            _ => panic!("expected Ret"),
        }
    }

    /// DCE removes a pure op whose def is never used.
    #[test]
    fn dce_removes_dead_const() {
        let mut f = CfgFunction::new(Some("dce".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let arg = f.new_vreg(RegClass::Gp);
        let dead = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(arg);
        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: dead,
            value: 42,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: arg };

        let changed = dead_code_elimination(&mut f);
        assert!(changed, "DCE should remove the unused ConstTaggedInt");
        assert!(
            f.block(entry).body.is_empty(),
            "body should be empty after DCE"
        );
    }

    /// DCE keeps side-effecting ops even when their result is unused
    /// (e.g. SlotStore — no def at all, only side effect).
    #[test]
    fn dce_keeps_side_effecting_ops() {
        let mut f = CfgFunction::new(Some("dce_keep".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let arg = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(arg);
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: arg,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: arg };

        let changed = dead_code_elimination(&mut f);
        assert!(!changed, "no DCE — SlotStore is side-effecting");
        assert_eq!(f.block(entry).body.len(), 1);
    }

    /// Fixpoint: trivial-param elim creates dead Moves, coalesce kills
    /// them, DCE cleans up the const that fed them. Uses a non-entry-
    /// param Move src so the "preserve entry-param bridge" exception
    /// doesn't gate coalescing.
    #[test]
    fn optimize_fixpoint() {
        let mut f = CfgFunction::new(Some("fixpoint".into()), 0);
        let entry = f.new_block();
        let join = f.new_block();
        f.entry = entry;
        let dead_const = f.new_vreg(RegClass::Gp);
        let live_const = f.new_vreg(RegClass::Gp);
        let copy = f.new_vreg(RegClass::Gp);
        let phi = f.new_vreg(RegClass::Gp);

        // A dead constant + a live constant + a Move of the live const.
        // (No entry params; coalesce is free to fold the Move.)
        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: dead_const,
            value: 99,
        });
        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: live_const,
            value: 7,
        });
        f.block_mut(entry).body.push(Op::Move {
            dst: copy,
            src: live_const,
        });
        // Jump to join, passing arg via copy.
        f.block_mut(entry).terminator = Terminator::Jump {
            target: join,
            args: vec![copy],
        };
        f.block_mut(join).params.push(phi);
        f.block_mut(join).terminator = Terminator::Ret { value: phi };
        f.block_mut(join).predecessors.push(entry);

        optimize(&mut f);

        // After fixpoint:
        // - Move coalesced → copy gone, Jump args = [live_const].
        // - Trivial param: join has 1 pred passing live_const; phi → live_const.
        // - DCE: dead_const removed. live_const survives because Ret uses it.
        assert_eq!(
            f.block(entry).body.len(),
            1,
            "entry body has only live_const left",
        );
        assert!(
            matches!(f.block(entry).body[0], Op::ConstTaggedInt { value: 7, .. }),
            "the surviving op is the live_const",
        );
        assert!(f.block(join).params.is_empty(), "join param folded");
        match &f.block(entry).terminator {
            Terminator::Jump { args, .. } => assert!(args.is_empty(), "Jump args folded"),
            _ => panic!("expected Jump"),
        }
        match &f.block(join).terminator {
            Terminator::Ret { value } => assert_eq!(*value, live_const),
            _ => panic!("expected Ret"),
        }
    }
}
