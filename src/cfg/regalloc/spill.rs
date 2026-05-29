//! Spill rewriter for the chordal-coloring regalloc.
//!
//! Given a single VReg `v` to spill:
//! - Allocate a fresh stack slot `s`.
//! - After every def of `v`, insert `SlotStore(s, v)` — `v`'s live
//!   range shrinks to the def→store edge.
//! - Before every use of `v` (body op + terminator), insert
//!   `SlotLoad(fresh, s)` and rename the use to `fresh` — the new
//!   load-result VReg lives only from load to use site (one
//!   instruction span).
//!
//! Net effect: `v` and each per-use load result have minimal live
//! ranges, slashing the simultaneous-live count at the pressure
//! point. Recomputing liveness + interference + coloring after the
//! rewrite typically reduces `max_color` by 1 per spill.
//!
//! **Scope limits (Phase 4d-1):**
//! - Body-op def sites only. Block params and `InlineBranchOp::dst`
//!   are not currently spillable — those require more care
//!   (entry-block params represent the calling convention; terminator
//!   defs would need SlotStore at the fall-through successor's
//!   start, like `lift_vregs::insert_terminator_def_stores`).
//! - One spill per iteration. Multiple high-pressure VRegs are
//!   handled by re-iterating the whole allocate-then-spill loop.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::cfg::regalloc::color::{Coloring, color};
use crate::cfg::regalloc::interference::build_interference;
use crate::cfg::regalloc::liveness::compute_liveness;
use crate::cfg::{BlockId, CfgFunction, Op, RegClass, SlotId, VReg};

/// Per-class budget of physical registers available to color into.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub gp: u32,
    pub fp: u32,
}

impl Budget {
    pub fn for_class(&self, class: RegClass) -> u32 {
        match class {
            RegClass::Gp => self.gp,
            RegClass::Fp => self.fp,
        }
    }
}

/// Final outcome of `allocate_with_spilling`.
#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub coloring: Coloring,
    /// VRegs that were spilled (rewritten through stack slots). Empty
    /// when the function fit in the budget on the first try.
    pub spilled: Vec<VReg>,
    /// Number of allocator iterations (= initial coloring + 1 per
    /// spill).
    pub iterations: u32,
}

/// Coloring with spilling: iterate compute_coloring → pick over-budget
/// candidate → spill (rewrite the CFG) → re-color, until everything
/// fits or no further candidate is available. The `f` is mutated by
/// the spilling rewrites; the returned `Coloring` applies to the
/// final state.
pub fn allocate_with_spilling(f: &mut CfgFunction, budget: Budget) -> AllocationResult {
    // Hard cap so a degenerate input (e.g. a function whose pressure
    // can't be reduced by body-op spilling alone) doesn't burn
    // unbounded compile time. The Phase 4d-2 work will improve the
    // spill heuristic to make progress detection more precise.
    let max_iter: u32 = std::env::var("BEAGLE_SSA_SPILL_MAX_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let mut spilled: Vec<VReg> = Vec::new();
    let mut iterations: u32 = 0;
    let mut prev_max_gp: Option<u32> = None;
    let mut prev_max_fp: Option<u32> = None;
    loop {
        iterations += 1;
        let liveness = compute_liveness(f);
        let ig = build_interference(f, &liveness);
        let coloring = color(f, &ig);
        if fits_budget(&coloring, budget) {
            return AllocationResult {
                coloring,
                spilled,
                iterations,
            };
        }
        if iterations >= max_iter {
            return AllocationResult {
                coloring,
                spilled,
                iterations,
            };
        }
        // No-progress check: if neither max_color budged after the
        // previous spill, our heuristic isn't making the function
        // colorable — bail rather than loop. The current naive
        // heuristic ("spill the over-budget vreg") often hits this
        // because the over-budget def's own site is the worst
        // pressure point.
        let cur_gp = coloring.max_color(crate::cfg::RegClass::Gp);
        let cur_fp = coloring.max_color(crate::cfg::RegClass::Fp);
        if let (Some(pg), Some(pf)) = (prev_max_gp, prev_max_fp) {
            if cur_gp >= pg && cur_fp >= pf {
                return AllocationResult {
                    coloring,
                    spilled,
                    iterations,
                };
            }
        }
        prev_max_gp = Some(cur_gp);
        prev_max_fp = Some(cur_fp);

        let Some(victim) = pick_spill_candidate(f, &coloring, budget) else {
            return AllocationResult {
                coloring,
                spilled,
                iterations,
            };
        };
        let _slot = spill_one(f, victim);
        spilled.push(victim);
    }
}

/// True if every VReg's color fits within its class's budget.
pub fn fits_budget(coloring: &Coloring, budget: Budget) -> bool {
    coloring.max_color(RegClass::Gp) < budget.gp && coloring.max_color(RegClass::Fp) < budget.fp
}

/// Pick a body-op-defined VReg whose color exceeds its class's budget.
/// Heuristic: lowest spill cost (= total uses + defs), then highest
/// color (the one most over budget). Returns None when no spillable
/// candidate exists.
fn pick_spill_candidate(f: &CfgFunction, coloring: &Coloring, budget: Budget) -> Option<VReg> {
    // Restrict to body-op-defined VRegs. Build the eligible set.
    let mut eligible: std::collections::HashSet<VReg> = std::collections::HashSet::new();
    for block in &f.blocks {
        for op in &block.body {
            for d in op.defs() {
                eligible.insert(d);
            }
        }
    }

    // Count uses + defs per VReg as a cheap spill-cost proxy.
    let mut cost: HashMap<VReg, u32> = HashMap::new();
    for block in &f.blocks {
        for op in &block.body {
            for u in op.uses() {
                *cost.entry(u).or_insert(0) += 1;
            }
            for d in op.defs() {
                *cost.entry(d).or_insert(0) += 1;
            }
        }
        for u in block.terminator.uses() {
            *cost.entry(u).or_insert(0) += 1;
        }
    }

    // Filter to eligible AND over-budget; sort by (cost asc, color desc).
    let mut candidates: Vec<(u32, u32, VReg)> = eligible
        .into_iter()
        .filter(|v| coloring.color_of(*v) >= budget.for_class(v.class))
        .map(|v| (cost.get(&v).copied().unwrap_or(0), coloring.color_of(v), v))
        .collect();
    if candidates.is_empty() {
        return None;
    }
    // Lowest cost first; if tied, highest color (= worst over-budget).
    candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
    Some(candidates[0].2)
}

/// Rewrite `f` to route `vreg` through a fresh stack slot. Returns
/// the slot id allocated. Caller is responsible for re-running
/// liveness / interference / coloring.
pub fn spill_one(f: &mut CfgFunction, vreg: VReg) -> SlotId {
    // Route by class so the destination is GC-correct (I9): GP spills
    // go to a scanned root slot (the GC must see the pointer), FP spills
    // go to the unscanned region (a raw f64 must not be misread as a
    // heap pointer). See `CfgFunction::alloc_slot_for`.
    let slot = f.alloc_slot_for(vreg.class);

    let num_blocks = f.blocks.len();
    for bid_idx in 0..num_blocks {
        let bid = BlockId(bid_idx as u32);

        // Rewrite body ops: prepend SlotLoad for any use, emit op,
        // append SlotStore for any def of `vreg`.
        let body = std::mem::take(&mut f.block_mut(bid).body);
        let mut new_body: Vec<Op> = Vec::with_capacity(body.len() * 2);
        for mut op in body {
            let uses_spilled = op.uses().iter().any(|u| *u == vreg);
            if uses_spilled {
                let fresh = f.new_vreg(vreg.class);
                new_body.push(Op::SlotLoad { dst: fresh, slot });
                let mut rename = HashMap::new();
                rename.insert(vreg, fresh);
                op.rename_uses(&rename);
            }
            let defs_spilled = op.defs().iter().any(|d| *d == vreg);
            new_body.push(op);
            if defs_spilled {
                new_body.push(Op::SlotStore { slot, src: vreg });
            }
        }
        f.block_mut(bid).body = new_body;

        // Terminator: prepend SlotLoad if the terminator uses `vreg`.
        if f.block(bid).terminator.uses().iter().any(|u| *u == vreg) {
            let fresh = f.new_vreg(vreg.class);
            f.block_mut(bid)
                .body
                .push(Op::SlotLoad { dst: fresh, slot });
            let mut rename = HashMap::new();
            rename.insert(vreg, fresh);
            f.block_mut(bid).terminator.rename_uses(&rename);
        }
    }

    slot
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CfgFunction, Op, RegClass, Terminator};

    /// Mechanical test of `spill_one`: a body-op def gets a SlotStore
    /// inserted right after its def, and a SlotLoad before each use.
    #[test]
    fn spill_one_inserts_store_and_loads() {
        let mut f = CfgFunction::new(Some("mech".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let v = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(a);
        // v = a + a (the def to spill)
        f.block_mut(entry).body.push(Op::AddInt {
            dst: v,
            lhs: a,
            rhs: a,
        });
        // Two uses of v in separate ops, to verify per-use loads.
        let r1 = f.new_vreg(RegClass::Gp);
        let r2 = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: r1,
            lhs: v,
            rhs: v,
        });
        f.block_mut(entry).body.push(Op::AddInt {
            dst: r2,
            lhs: v,
            rhs: r1,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: r2 };

        let slot = spill_one(&mut f, v);
        // Expect body to be:
        //   v = AddInt a a        (def of v)
        //   SlotStore slot <- v   (inserted)
        //   fresh1 = SlotLoad     (inserted before r1)
        //   r1 = AddInt fresh1 fresh1   (uses renamed)
        //   fresh2 = SlotLoad     (inserted before r2)
        //   r2 = AddInt fresh2 r1
        let body = &f.block(entry).body;
        assert!(
            body.iter().any(
                |op| matches!(op, Op::SlotStore { src, slot: s, .. } if *src == v && *s == slot)
            ),
            "SlotStore for v at slot {:?} missing: {:?}",
            slot,
            body
        );
        let slot_load_count = body
            .iter()
            .filter(|op| matches!(op, Op::SlotLoad { .. }))
            .count();
        assert_eq!(slot_load_count, 2, "two SlotLoads (one per use of v)");
        // After spill, no op (other than the one defining v and the
        // SlotStore) should reference v as a use.
        for op in body {
            if matches!(op, Op::SlotStore { src, .. } if *src == v) {
                continue;
            }
            if matches!(op, Op::AddInt { dst, .. } if *dst == v) {
                continue;
            }
            for u in op.uses() {
                assert_ne!(u, v, "use of v should have been renamed: {:?}", op);
            }
        }
    }

    /// Function that fits the budget without any spill — should not
    /// rewrite anything.
    #[test]
    fn fits_budget_skips_spill() {
        let mut f = CfgFunction::new(Some("fits".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let b = f.new_vreg(RegClass::Gp);
        let r = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(a);
        f.block_mut(entry).params.push(b);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: r,
            lhs: a,
            rhs: b,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: r };

        let result = allocate_with_spilling(&mut f, Budget { gp: 8, fp: 8 });
        assert_eq!(result.spilled.len(), 0, "no spill needed");
        assert_eq!(result.iterations, 1, "single coloring pass");
        assert_eq!(f.block(entry).body.len(), 1);
    }

    /// The iterative spill loop terminates: even when the budget is
    /// too tight for the function to fit, we don't loop forever
    /// (when no spill candidate exists, we return the over-budget
    /// coloring).
    #[test]
    fn allocator_terminates_when_unsatisfiable() {
        // 5 entry params all simultaneously live → need 5 cols.
        // Budget = 4 GP. Entry params aren't spillable (they're
        // function args; spilling them would break the calling
        // convention). The allocator should return with an over-
        // budget coloring rather than loop.
        let mut f = CfgFunction::new(Some("unsat".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let b = f.new_vreg(RegClass::Gp);
        let c = f.new_vreg(RegClass::Gp);
        let d = f.new_vreg(RegClass::Gp);
        let e = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(a);
        f.block_mut(entry).params.push(b);
        f.block_mut(entry).params.push(c);
        f.block_mut(entry).params.push(d);
        f.block_mut(entry).params.push(e);
        // Use all 5 to keep them all live at the body op.
        let r = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: r,
            lhs: a,
            rhs: b,
        });
        let s = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: s,
            lhs: c,
            rhs: d,
        });
        let t = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: t,
            lhs: r,
            rhs: s,
        });
        let u = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: u,
            lhs: t,
            rhs: e,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: u };

        let result = allocate_with_spilling(&mut f, Budget { gp: 4, fp: 8 });
        // Terminated. May have a spill or not, but didn't infinite-
        // loop. The over-budget situation is reflected in the
        // returned coloring's max_color.
        assert!(result.iterations < 100, "should terminate quickly");
    }
}
