//! Global load CSE — cross-block redundant `SlotLoad` elimination.
//!
//! `forward_slot_stores` (in `opt.rs`) only forwards a store into a *same-block,
//! same-register* reload. But Beagle's guard-and-inline-bail lowering re-loads a
//! local once per operation: a monomorphic `a + b` becomes
//! `guard_int a; guard_int b; add`, and the guard split means `a` and `b` are
//! re-loaded from their slots in each guard block. The CFG dump for a hot kernel
//! shows `slot(0)` loaded 3× and `slot(1)` loaded 3×, each into a *distinct*
//! VReg. Because they are distinct VRegs, redundant-guard elimination only kills
//! one guard, LICM can't see the value as invariant, and the bail safepoints
//! survive.
//!
//! This pass collapses those reloads to a single SSA value so the rest of the
//! Phase-3 fixpoint can fire:
//!
//! ```text
//! v0 = SlotLoad slot(0)     v0 = SlotLoad slot(0)
//! ...                       ...
//! v1 = SlotLoad slot(0)  →  v1 = Move v0     (then coalesced away: v1 → v0)
//! ```
//!
//! → the value is now one VReg, guarded once → guard-elim drops the rest → DCE
//! removes the now-dead bail blocks → their safepoints disappear → LICM hoists.
//!
//! ## Algorithm — forward "available slot value" dataflow
//!
//! For every slot `S` we track which SSA VReg currently holds its value:
//!
//! - `OUT[b]` = the `SlotId → VReg` map at block exit.
//! - `IN[b]`  = the **intersection** over `b`'s predecessors of their `OUT`
//!   (keep `(S, v)` only when *every* predecessor agrees on the same `v`).
//!   Intersection is what makes a forwarded value safe at a merge: if all preds
//!   hold `S` in the same VReg `v`, then `v`'s single SSA def dominates every
//!   pred and therefore dominates the merge, so reusing it is well-formed.
//!
//! Transfer within a block, starting from `IN[b]`:
//! - `SlotStore(S, src)` → `avail[S] = src` (the slot now holds `src`).
//! - `SlotLoad(S) → dst`:
//!     - if `avail[S]` is some `v`, this load is **redundant**: rewrite it to
//!       `Move { dst, src: v }` (coalesce + DCE finish the job). `avail[S]`
//!       stays `v`.
//!     - else `avail[S] = dst` (this load establishes the value).
//! - a **GC safepoint** op/terminator clears *all* availability: per **I9** the
//!   collector scans frame slots, not registers, so a pointer cached in a VReg
//!   goes stale across a safepoint. The slot itself is updated, so a *fresh*
//!   reload after the safepoint is correct — we just must not forward the old
//!   register value across it. Clearing every slot is conservative (a known
//!   non-pointer could survive) but sound, and the keystone guard chains have no
//!   safepoint between their reloads.
//!
//! ## Soundness / termination
//!
//! Every `(S, v)` ever placed in any `OUT[b]` is *valid* — i.e. on every path
//! reaching that point `v` currently holds `S` and `v`'s register copy is not
//! stale — by induction: transfer only inserts a pair right after a
//! store/load that establishes it (no later kill in the block), and the
//! intersection meet only keeps a pair present-and-equal in all predecessors,
//! which (by the dominance argument above) stays valid across the merge. This
//! holds at *every* iteration, not just at convergence, so an early stop is
//! still sound — it can only *miss* CSE opportunities, never invent an unsound
//! one. The least-fixpoint-from-empty iteration is monotone in practice and
//! bounded; an iteration cap is a defensive backstop only.
//!
//! Like `coalesce_copies` and `eliminate_redundant_guards`, the function-wide
//! VReg substitution (done by the subsequent coalesce of the `Move`s we emit)
//! relies on the pipeline's **I9** invariant that a `SlotLoad` result is never
//! live across a safepoint in valid input — the same contract those passes
//! already depend on. We additionally never *forward* a value across a
//! safepoint (the dataflow clears at safepoints), so the forwarding region is
//! itself safepoint-free.

use std::collections::HashMap;

use crate::cfg::dom::reverse_postorder;
use crate::cfg::gc_safety::{is_gc_safepoint_op, is_gc_safepoint_terminator};
use crate::cfg::{Block, BlockId, CfgFunction, Op, SlotId, VReg};

/// Collapse cross-block redundant `SlotLoad`s into `Move`s of the dominating
/// load/store value. Returns whether anything changed. Opt out with
/// `BEAGLE_SSA_NO_LOAD_CSE`.
pub fn global_load_cse(f: &mut CfgFunction) -> bool {
    let n = f.num_blocks();
    if n == 0 {
        return false;
    }

    let rpo = reverse_postorder(f);

    // OUT[b]: slot -> VReg holding it at block exit. Least-fixpoint from empty.
    let mut out: Vec<HashMap<SlotId, VReg>> = vec![HashMap::new(); n];

    // Generous backstop; the analysis converges well within this in practice,
    // and an early stop is still sound (see module docs).
    let cap = n.saturating_mul(8).saturating_add(64);
    let mut iters = 0;
    let mut changed = true;
    while changed && iters < cap {
        iters += 1;
        changed = false;
        for &b in &rpo {
            let in_state = meet_predecessors(f, b, &out);
            let new_out = transfer(&f.blocks[b.0 as usize], in_state);
            if new_out != out[b.0 as usize] {
                out[b.0 as usize] = new_out;
                changed = true;
            }
        }
    }

    // Final pass: recompute IN per block from the converged OUT, walk the body,
    // and rewrite each redundant SlotLoad to a Move of the available value.
    let mut rewrites = 0usize;
    for b in 0..n {
        let bid = BlockId(b as u32);
        let mut avail = meet_predecessors(f, bid, &out);
        for op in f.blocks[b].body.iter_mut() {
            match op {
                Op::SlotStore { slot, src } => {
                    avail.insert(*slot, *src);
                }
                Op::SlotLoad { dst, slot } => match avail.get(slot) {
                    Some(&v) if v != *dst => {
                        *op = Op::Move { dst: *dst, src: v };
                        rewrites += 1;
                    }
                    Some(_) => {} // already this VReg; nothing to do
                    None => {
                        avail.insert(*slot, *dst);
                    }
                },
                other => {
                    if is_gc_safepoint_op(other) {
                        avail.clear();
                    }
                }
            }
        }
        if is_gc_safepoint_terminator(&f.blocks[b].terminator) {
            avail.clear();
        }
    }

    if rewrites > 0 && std::env::var("BEAGLE_LOAD_CSE_STATS").is_ok() {
        eprintln!(
            "[load-cse] {}: {} redundant SlotLoad(s) forwarded",
            f.debug_name.as_deref().unwrap_or("<anon>"),
            rewrites
        );
    }

    rewrites > 0
}

/// `IN[b]` = intersection over predecessors of their `OUT`, keeping `(S, v)`
/// only when every predecessor agrees on the same `v`. The entry block (and any
/// block recorded with no predecessors) starts empty.
fn meet_predecessors(
    f: &CfgFunction,
    b: BlockId,
    out: &[HashMap<SlotId, VReg>],
) -> HashMap<SlotId, VReg> {
    if b == f.entry {
        return HashMap::new();
    }
    let preds = &f.block(b).predecessors;
    let Some((first, rest)) = preds.split_first() else {
        return HashMap::new();
    };
    let mut acc = out[first.0 as usize].clone();
    for p in rest {
        let po = &out[p.0 as usize];
        acc.retain(|slot, v| po.get(slot) == Some(v));
        if acc.is_empty() {
            break;
        }
    }
    acc
}

/// Forward transfer: produce `OUT[b]` from `IN[b]` by walking the block body.
/// Mirrors the rewrite pass exactly (minus the actual mutation) so the
/// converged `OUT` matches what the final pass observes.
fn transfer(block: &Block, mut avail: HashMap<SlotId, VReg>) -> HashMap<SlotId, VReg> {
    for op in &block.body {
        match op {
            Op::SlotStore { slot, src } => {
                avail.insert(*slot, *src);
            }
            Op::SlotLoad { dst, slot } => {
                avail.entry(*slot).or_insert(*dst);
            }
            other => {
                if is_gc_safepoint_op(other) {
                    avail.clear();
                }
            }
        }
    }
    if is_gc_safepoint_terminator(&block.terminator) {
        avail.clear();
    }
    avail
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CallTarget, ClobberSet, RegClass, Terminator};
    use crate::ir::Condition;

    /// A straight-line dominator chain that reloads the same slot in each block
    /// — the keystone guard-chain shape. The later loads must become Moves of
    /// the first load's VReg.
    #[test]
    fn dominator_chain_reload_becomes_move() {
        let mut f = CfgFunction::new(Some("chain".into()), 1);
        let b0 = f.new_block();
        let b1 = f.new_block();
        let b2 = f.new_block();
        f.entry = b0;

        let v0 = f.new_vreg(RegClass::Gp);
        let v1 = f.new_vreg(RegClass::Gp);
        let v2 = f.new_vreg(RegClass::Gp);

        f.block_mut(b0).body.push(Op::SlotLoad {
            dst: v0,
            slot: SlotId(0),
        });
        f.block_mut(b0).terminator = Terminator::Jump {
            target: b1,
            args: vec![],
        };
        f.block_mut(b1).body.push(Op::SlotLoad {
            dst: v1,
            slot: SlotId(0),
        });
        f.block_mut(b1).terminator = Terminator::Jump {
            target: b2,
            args: vec![],
        };
        f.block_mut(b2).body.push(Op::SlotLoad {
            dst: v2,
            slot: SlotId(0),
        });
        f.block_mut(b2).terminator = Terminator::Ret { value: v2 };

        f.block_mut(b1).predecessors = vec![b0];
        f.block_mut(b2).predecessors = vec![b1];

        let changed = global_load_cse(&mut f);
        assert!(changed, "redundant reloads should be rewritten");
        assert!(
            matches!(f.block(b1).body[0], Op::Move { dst, src } if dst == v1 && src == v0),
            "b1's reload should become Move v1 <- v0"
        );
        assert!(
            matches!(f.block(b2).body[0], Op::Move { dst, src } if dst == v2 && src == v0),
            "b2's reload should become Move v2 <- v0"
        );
    }

    /// A store before a reload: the load must forward the stored source value,
    /// even cross-block and even when the load's dst differs from the source.
    #[test]
    fn store_forwards_to_later_load() {
        let mut f = CfgFunction::new(Some("store_fwd".into()), 1);
        let b0 = f.new_block();
        let b1 = f.new_block();
        f.entry = b0;

        let src = f.new_vreg(RegClass::Gp);
        let dst = f.new_vreg(RegClass::Gp);

        f.block_mut(b0).body.push(Op::SlotStore {
            slot: SlotId(3),
            src,
        });
        f.block_mut(b0).terminator = Terminator::Jump {
            target: b1,
            args: vec![],
        };
        f.block_mut(b1).body.push(Op::SlotLoad {
            dst,
            slot: SlotId(3),
        });
        f.block_mut(b1).terminator = Terminator::Ret { value: dst };
        f.block_mut(b1).predecessors = vec![b0];

        let changed = global_load_cse(&mut f);
        assert!(changed);
        assert!(
            matches!(f.block(b1).body[0], Op::Move { dst: d, src: s } if d == dst && s == src),
            "the reload should forward the stored source"
        );
    }

    /// An intervening store to the slot kills forwarding: the second load must
    /// stay a load (it observes the new value, not the first one).
    #[test]
    fn intervening_store_blocks_cse() {
        let mut f = CfgFunction::new(Some("kill_store".into()), 1);
        let b0 = f.new_block();
        f.entry = b0;

        let v0 = f.new_vreg(RegClass::Gp);
        let nv = f.new_vreg(RegClass::Gp);
        let v1 = f.new_vreg(RegClass::Gp);

        f.block_mut(b0).body.push(Op::SlotLoad {
            dst: v0,
            slot: SlotId(0),
        });
        f.block_mut(b0).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: nv,
        });
        f.block_mut(b0).body.push(Op::SlotLoad {
            dst: v1,
            slot: SlotId(0),
        });
        f.block_mut(b0).terminator = Terminator::Ret { value: v1 };

        let changed = global_load_cse(&mut f);
        // The second load now forwards the *stored* value `nv`, not `v0`.
        assert!(changed);
        assert!(
            matches!(f.block(b0).body[2], Op::Move { dst, src } if dst == v1 && src == nv),
            "reload after a store must forward the stored value, not the stale load"
        );
    }

    /// A safepoint (Call) between two loads kills forwarding: the cached
    /// register value could be a stale pointer after a relocating GC, so the
    /// second load must remain a load.
    #[test]
    fn safepoint_blocks_cse() {
        let mut f = CfgFunction::new(Some("kill_safepoint".into()), 1);
        let b0 = f.new_block();
        f.entry = b0;

        let v0 = f.new_vreg(RegClass::Gp);
        let cd = f.new_vreg(RegClass::Gp);
        let v1 = f.new_vreg(RegClass::Gp);

        f.block_mut(b0).body.push(Op::SlotLoad {
            dst: v0,
            slot: SlotId(0),
        });
        f.block_mut(b0).body.push(Op::Call {
            dst: cd,
            target: CallTarget::Pointer(0x1000),
            args: vec![],
            is_builtin: true,
            clobbers: ClobberSet::AllCallerSaved,
        });
        f.block_mut(b0).body.push(Op::SlotLoad {
            dst: v1,
            slot: SlotId(0),
        });
        f.block_mut(b0).terminator = Terminator::Ret { value: v1 };

        let changed = global_load_cse(&mut f);
        assert!(!changed, "must not forward a value across a safepoint");
        assert!(
            matches!(f.block(b0).body[2], Op::SlotLoad { .. }),
            "the post-safepoint reload must remain a load"
        );
    }

    /// A diamond where one branch stores to the slot: at the join the two
    /// predecessors disagree on the slot's value, so the intersection meet
    /// drops it and the join load must stay a load.
    #[test]
    fn disagreeing_merge_blocks_cse() {
        let mut f = CfgFunction::new(Some("diamond".into()), 1);
        let entry = f.new_block();
        let then_b = f.new_block();
        let else_b = f.new_block();
        let join = f.new_block();
        f.entry = entry;

        let v0 = f.new_vreg(RegClass::Gp);
        let cond = f.new_vreg(RegClass::Gp);
        let stored = f.new_vreg(RegClass::Gp);
        let jv = f.new_vreg(RegClass::Gp);

        // entry: load slot(0) -> v0, then branch.
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v0,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: cond,
            rhs: cond,
            t_target: then_b,
            t_args: vec![],
            f_target: else_b,
            f_args: vec![],
        };
        // then: store a different value into slot(0).
        f.block_mut(then_b).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: stored,
        });
        f.block_mut(then_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        // else: leave slot(0) as v0.
        f.block_mut(else_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        // join: reload slot(0) — value differs by path, so cannot CSE.
        f.block_mut(join).body.push(Op::SlotLoad {
            dst: jv,
            slot: SlotId(0),
        });
        f.block_mut(join).terminator = Terminator::Ret { value: jv };

        f.block_mut(then_b).predecessors = vec![entry];
        f.block_mut(else_b).predecessors = vec![entry];
        f.block_mut(join).predecessors = vec![then_b, else_b];

        let changed = global_load_cse(&mut f);
        assert!(!changed, "disagreeing merge must block CSE");
        assert!(
            matches!(f.block(join).body[0], Op::SlotLoad { .. }),
            "join reload must remain a load when predecessors disagree"
        );
    }

    /// A diamond where the slot value agrees on both paths: the join reload
    /// CSEs to the dominating load.
    #[test]
    fn agreeing_merge_allows_cse() {
        let mut f = CfgFunction::new(Some("diamond_ok".into()), 1);
        let entry = f.new_block();
        let then_b = f.new_block();
        let else_b = f.new_block();
        let join = f.new_block();
        f.entry = entry;

        let v0 = f.new_vreg(RegClass::Gp);
        let cond = f.new_vreg(RegClass::Gp);
        let jv = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v0,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: cond,
            rhs: cond,
            t_target: then_b,
            t_args: vec![],
            f_target: else_b,
            f_args: vec![],
        };
        f.block_mut(then_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        f.block_mut(else_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        f.block_mut(join).body.push(Op::SlotLoad {
            dst: jv,
            slot: SlotId(0),
        });
        f.block_mut(join).terminator = Terminator::Ret { value: jv };

        f.block_mut(then_b).predecessors = vec![entry];
        f.block_mut(else_b).predecessors = vec![entry];
        f.block_mut(join).predecessors = vec![then_b, else_b];

        let changed = global_load_cse(&mut f);
        assert!(changed, "agreeing merge should allow CSE");
        assert!(
            matches!(f.block(join).body[0], Op::Move { dst, src } if dst == jv && src == v0),
            "join reload should forward the dominating load's value"
        );
    }

    /// A loop whose body reloads the slot and has no store/safepoint: the
    /// least-fixpoint-from-empty result conservatively keeps the header reload
    /// as its own load (no unsound loop-carried bootstrap, no oscillation).
    #[test]
    fn loop_reload_terminates_and_is_sound() {
        let mut f = CfgFunction::new(Some("loop".into()), 1);
        let pre = f.new_block();
        let header = f.new_block();
        let exit = f.new_block();
        f.entry = pre;

        let v0 = f.new_vreg(RegClass::Gp);
        let hv = f.new_vreg(RegClass::Gp);
        let cond = f.new_vreg(RegClass::Gp);

        f.block_mut(pre).body.push(Op::SlotLoad {
            dst: v0,
            slot: SlotId(0),
        });
        f.block_mut(pre).terminator = Terminator::Jump {
            target: header,
            args: vec![],
        };
        f.block_mut(header).body.push(Op::SlotLoad {
            dst: hv,
            slot: SlotId(0),
        });
        f.block_mut(header).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: cond,
            rhs: cond,
            t_target: header,
            t_args: vec![],
            f_target: exit,
            f_args: vec![],
        };
        f.block_mut(exit).terminator = Terminator::Ret { value: hv };
        f.block_mut(header).predecessors = vec![pre, header];
        f.block_mut(exit).predecessors = vec![header];

        // Must terminate; header load stays a load (its value flows around the
        // back-edge as hv, which disagrees with pre's v0 at the meet).
        let _ = global_load_cse(&mut f);
        assert!(
            matches!(f.block(header).body[0], Op::SlotLoad { .. }),
            "loop header reload must remain a load under the conservative fixpoint"
        );
    }
}
