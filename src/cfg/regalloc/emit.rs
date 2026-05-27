//! Phase 4f — lower the allocated CFG to post-SSA "linear allocated"
//! form by mutating the CFG in place.
//!
//! After `lower_to_allocated`:
//!
//! - Every VReg's `index` is its assigned color (= physical register
//!   number). The `class` is preserved (GP / FP).
//! - Block params are cleared. Each block's outgoing terminator-args
//!   are cleared too. The data transfer that the args used to encode
//!   is now realized as explicit `Op::Move` instructions, inserted by
//!   the parallel-copy resolution.
//! - For single-succ terminators (Jump, Throw), edge-transfer Moves
//!   are appended to the source block's body just before the
//!   terminator. For multi-succ terminators (Branch, InlineBranch),
//!   transfers are prepended to the TARGET block (which is
//!   unique-pred after Phase 1c's critical-edge split).
//!
//! The result is no longer SSA — it's a CFG of register-allocated
//! straight-line ops. Phase 4f-2 walks this form to drive the
//! existing backend's machine-code emission.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::cfg::regalloc::color::Coloring;
use crate::cfg::regalloc::edge::{EdgeTransfers, Scratch, resolve_edges};
use crate::cfg::{CfgFunction, Op, Terminator, VReg};

/// Transform `f` from SSA-form-with-coloring into post-SSA allocated
/// form (see module docs). Mutates in place.
pub fn lower_to_allocated(f: &mut CfgFunction, coloring: &Coloring, scratch: Scratch) {
    // 1. Compute edge transfers per the (still-SSA) CFG + coloring.
    let transfers = resolve_edges(f, coloring, scratch);

    // 2. Insert each EdgeTransfers as Op::Move ops at the right
    //    location, per terminator arity.
    for et in transfers {
        insert_transfers(f, et);
    }

    // 3. Rewrite every VReg in the function to its color. Defs, uses,
    //    block params, terminator args — all get renamed.
    let rename = build_color_rename(f, coloring);
    apply_color_rename(f, &rename);

    // 4. Drop block params and terminator args. The Moves inserted in
    //    step 2 carry the data; params/args are vestigial structure
    //    that the verifier would now flag as arity-mismatched (zero on
    //    each side after this clear keeps it consistent).
    for block in f.blocks.iter_mut() {
        block.params.clear();
        clear_terminator_args(&mut block.terminator);
    }
}

fn insert_transfers(f: &mut CfgFunction, et: EdgeTransfers) {
    let moves: Vec<Op> = et
        .moves
        .into_iter()
        .map(|pm| Op::Move {
            dst: VReg {
                index: pm.dst_color,
                class: pm.class,
            },
            src: VReg {
                index: pm.src_color,
                class: pm.class,
            },
        })
        .collect();
    if moves.is_empty() {
        return;
    }
    // Decide insertion location based on the source's terminator arity.
    let single_succ = matches!(
        f.block(et.from).terminator,
        Terminator::Jump { .. } | Terminator::Throw { .. } | Terminator::Ret { .. }
    );
    if single_succ {
        // End of source, before the terminator (which `body` doesn't
        // contain — terminator is a separate field). Just append.
        f.block_mut(et.from).body.extend(moves);
    } else {
        // Multi-succ: prepend to the target block. After critical-edge
        // split (Phase 1c), the target is unique-pred for the source
        // (or there's a `mid` block in between which is unique-pred).
        let existing = std::mem::take(&mut f.block_mut(et.to).body);
        let mut new_body = moves;
        new_body.extend(existing);
        f.block_mut(et.to).body = new_body;
    }
}

fn build_color_rename(f: &CfgFunction, coloring: &Coloring) -> HashMap<VReg, VReg> {
    let mut rename = HashMap::new();
    for v in 0..f.num_vregs() {
        let vr = VReg {
            index: v as u32,
            class: f.vreg_classes[v],
        };
        // Some VRegs introduced by transformations (e.g. a spill's
        // load-result that didn't get re-colored if the spiller bailed)
        // may not have a color entry. Skip those — leave them at their
        // original index, which is at least a deterministic placeholder
        // the verifier can flag downstream.
        let Some(color) = coloring.colors.get(&vr).copied() else {
            continue;
        };
        let new = VReg {
            index: color,
            class: vr.class,
        };
        if new != vr {
            rename.insert(vr, new);
        }
    }
    rename
}

fn apply_color_rename(f: &mut CfgFunction, rename: &HashMap<VReg, VReg>) {
    for block in f.blocks.iter_mut() {
        // Block params (defs at position 0)
        for p in block.params.iter_mut() {
            if let Some(&new) = rename.get(p) {
                *p = new;
            }
        }
        // Body ops (both defs and uses)
        for op in block.body.iter_mut() {
            op.rename_defs(rename);
            op.rename_uses(rename);
        }
        // Terminator (defs and uses)
        block.terminator.rename_defs(rename);
        block.terminator.rename_uses(rename);
    }
}

fn clear_terminator_args(term: &mut Terminator) {
    match term {
        Terminator::Jump { args, .. } => args.clear(),
        Terminator::Branch { t_args, f_args, .. } => {
            t_args.clear();
            f_args.clear();
        }
        Terminator::InlineBranch {
            fall_args,
            bail_args,
            ..
        } => {
            fall_args.clear();
            bail_args.clear();
        }
        Terminator::Throw { resume_args, .. } => resume_args.clear(),
        Terminator::Ret { .. } | Terminator::Unreachable => {}
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::regalloc::color::color;
    use crate::cfg::regalloc::interference::build_interference;
    use crate::cfg::regalloc::liveness::compute_liveness;
    use crate::cfg::{CfgFunction, Op, RegClass, Terminator};

    fn scratch() -> Scratch {
        Scratch { gp: 99, fp: 199 }
    }

    /// Identity function — single block, no edges, no transfers.
    /// After lowering: params dropped, body unchanged in structure,
    /// VRegs renamed to colors.
    #[test]
    fn identity_function_lowers_cleanly() {
        let mut f = CfgFunction::new(Some("id".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let x = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(x);
        f.block_mut(entry).terminator = Terminator::Ret { value: x };

        let liveness = compute_liveness(&f);
        let ig = build_interference(&f, &liveness);
        let coloring = color(&f, &ig);
        let x_color = coloring.color_of(x);

        lower_to_allocated(&mut f, &coloring, scratch());

        // Block params dropped.
        assert!(f.block(entry).params.is_empty(), "params cleared");
        // Body unchanged (no transfers needed for a Ret).
        assert!(f.block(entry).body.is_empty(), "no body ops");
        // Terminator's value VReg renamed to color.
        match &f.block(entry).terminator {
            Terminator::Ret { value } => {
                assert_eq!(value.index, x_color, "Ret value renamed to color");
            }
            _ => panic!("expected Ret"),
        }
    }

    /// Two blocks linked by Jump with a block-param transfer that
    /// requires an actual Move (source and dest colors differ).
    #[test]
    fn jump_edge_inserts_move_in_source() {
        let mut f = CfgFunction::new(Some("jump_xfer".into()), 0);
        let entry = f.new_block();
        let tail = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let p = f.new_vreg(RegClass::Gp);
        // Two more vregs in entry to force a > p's color
        let dummy = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(a);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: dummy,
            lhs: a,
            rhs: a,
        });
        // Pass `dummy` as the arg to tail's param p.
        f.block_mut(entry).terminator = Terminator::Jump {
            target: tail,
            args: vec![dummy],
        };
        f.block_mut(tail).params.push(p);
        f.block_mut(tail).terminator = Terminator::Ret { value: p };
        f.block_mut(tail).predecessors.push(entry);

        let liveness = compute_liveness(&f);
        let ig = build_interference(&f, &liveness);
        let coloring = color(&f, &ig);
        let dummy_color = coloring.color_of(dummy);
        let p_color = coloring.color_of(p);

        lower_to_allocated(&mut f, &coloring, scratch());

        // Both blocks' params dropped.
        assert!(f.block(entry).params.is_empty());
        assert!(f.block(tail).params.is_empty());

        // Entry's body: original AddInt + possibly an inserted Move
        // (if dummy and p got different colors).
        let entry_body = &f.block(entry).body;
        let move_count = entry_body
            .iter()
            .filter(|op| matches!(op, Op::Move { .. }))
            .count();
        if dummy_color != p_color {
            assert_eq!(move_count, 1, "Jump-edge Move inserted in source");
            // Verify the Move's colors match.
            let m = entry_body
                .iter()
                .find_map(|op| match op {
                    Op::Move { dst, src } => Some((*dst, *src)),
                    _ => None,
                })
                .unwrap();
            assert_eq!(m.0.index, p_color, "Move dst color = p's color");
            assert_eq!(m.1.index, dummy_color, "Move src color = dummy's color");
        } else {
            assert_eq!(move_count, 0, "same color → no Move needed");
        }

        // Entry's terminator args cleared.
        match &f.block(entry).terminator {
            Terminator::Jump { args, .. } => assert!(args.is_empty()),
            _ => panic!("expected Jump"),
        }
    }

    /// Branch with transfers on a target inserts moves at the START of
    /// the target block (multi-succ source → unique-pred target after
    /// critical-edge split).
    #[test]
    fn branch_edge_inserts_move_in_target() {
        let mut f = CfgFunction::new(Some("branch_xfer".into()), 0);
        let entry = f.new_block();
        let then_b = f.new_block();
        let else_b = f.new_block();
        f.entry = entry;
        let cond = f.new_vreg(RegClass::Gp);
        let pt = f.new_vreg(RegClass::Gp); // then's param
        let pe = f.new_vreg(RegClass::Gp); // else's param
        let xt = f.new_vreg(RegClass::Gp); // value passed on t edge
        let xe = f.new_vreg(RegClass::Gp); // value passed on f edge
        f.block_mut(entry).params.push(cond);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: xt,
            lhs: cond,
            rhs: cond,
        });
        f.block_mut(entry).body.push(Op::AddInt {
            dst: xe,
            lhs: cond,
            rhs: cond,
        });
        f.block_mut(entry).terminator = Terminator::Branch {
            cond: crate::ir::Condition::Equal,
            lhs: cond,
            rhs: cond,
            t_target: then_b,
            t_args: vec![xt],
            f_target: else_b,
            f_args: vec![xe],
        };
        f.block_mut(then_b).params.push(pt);
        f.block_mut(then_b).terminator = Terminator::Ret { value: pt };
        f.block_mut(else_b).params.push(pe);
        f.block_mut(else_b).terminator = Terminator::Ret { value: pe };
        f.block_mut(then_b).predecessors.push(entry);
        f.block_mut(else_b).predecessors.push(entry);

        let liveness = compute_liveness(&f);
        let ig = build_interference(&f, &liveness);
        let coloring = color(&f, &ig);

        lower_to_allocated(&mut f, &coloring, scratch());

        // Multi-succ source: transfers go in TARGET blocks (start),
        // not the source. Entry's body should be just its original
        // body ops; no Moves prepended/appended for the branch
        // transfers.
        let entry_body = &f.block(entry).body;
        assert!(
            !entry_body.iter().any(|op| matches!(op, Op::Move { .. })),
            "no Moves in entry: {:?}",
            entry_body
        );
        // Targets may have Moves at start (if their colors differ from
        // the source's args). We just check the Move count is reasonable
        // — at most 1 per target since each transfer is one Move.
        for tb in [then_b, else_b] {
            let body = &f.block(tb).body;
            let move_count = body
                .iter()
                .filter(|op| matches!(op, Op::Move { .. }))
                .count();
            assert!(move_count <= 1, "at most one Move per target");
        }
    }
}
