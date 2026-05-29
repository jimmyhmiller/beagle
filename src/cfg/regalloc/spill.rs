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
//! rewrite reduces `max_color` toward the budget.
//!
//! **Phase 3 driver (`allocate_with_spilling`):** clobber-aware Belady.
//! Each iteration colors under the I7 clobber model; while over budget,
//! it spills the *batch* of furthest-next-use values at the worst
//! pressure point (one recompute per high-pressure region, not per
//! value — large functions made per-value recompute pathologically
//! slow). GP spills land in GC-scanned root slots, FP in the unscanned
//! region (`spill_one` routes by class, I9).
//!
//! **Scope limits:**
//! - Body-op def sites only. Block params and `InlineBranchOp::dst`
//!   are not spillable — those need other machinery (entry-block params
//!   are the calling convention; terminator defs would need SlotStore
//!   at the fall-through successor's start). Pressure concentrated in
//!   block params makes the function bail to legacy.
//! - `spill_one` reloads at *every* use, so heavily-over-pressure
//!   functions cascade (reloads re-create pressure). A `spill_cap`
//!   bails those fast; eliminating them needs live-range splitting
//!   (Phase 7), which keeps a value in a register across a region and
//!   reloads only across the gap.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::cfg::dom::reverse_postorder;
use crate::cfg::regalloc::color::{ClobberConstraints, Coloring, color_with_constraints};
use crate::cfg::regalloc::interference::{build_interference, cross_safepoint_values};
use crate::cfg::regalloc::liveness::{Liveness, compute_liveness};
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
    /// True if the final coloring fits the budget. False means the
    /// spiller ran out of spillable candidates while still over budget
    /// (e.g. pressure concentrated in block params, which `spill_one`
    /// can't lower) — the driver bails to legacy in that case.
    pub fits: bool,
}

/// Phase 3 spiller — clobber-aware Belady. Iterates
/// color → (over budget?) → spill the live value at the worst pressure
/// point with the **furthest next use** (Belady MIN) → re-color, until
/// the coloring fits the pool. `f` is mutated by the spill rewrites.
///
/// `callee_saved_gp` is the number of callee-saved GP colors (the
/// caller-saved sub-pool starts there). It feeds the I7 clobber model:
/// each iteration recomputes the cross-safepoint set (spilling changes
/// liveness) and colors under the constraint, so the fit check naturally
/// accounts for both budgets — total GP ≤ `budget.gp` and cross-safepoint
/// GP ≤ `callee_saved_gp` (a cross-safepoint value that can't get a
/// callee-saved color overflows past `budget.gp`, which the check sees).
///
/// GP spills land in **root slots** (GC-scanned) and FP spills in the
/// **unscanned region** — `spill_one` routes by class (I9). For the
/// current corpus only GP pressure ever overflows, so the unscanned-slot
/// backend support isn't on the critical path.
pub fn allocate_with_spilling(
    f: &mut CfgFunction,
    budget: Budget,
    callee_saved_gp: u32,
) -> AllocationResult {
    // Per-recompute the spiller is O(liveness + interference + color),
    // which is expensive on large functions — so spill a *batch* (the
    // whole excess at the worst point) per recompute, not one value.
    // That bounds recomputes to roughly the number of independent
    // high-pressure regions (a handful) instead of one-per-spilled-value.
    //
    // `spill_cap` bounds total spills. The basic `spill_one` mechanism
    // reloads at every use, so a heavily-over-pressure function cascades
    // (reloads re-create pressure) and never converges — and even when it
    // does, that many reloads lose to the legacy fallback. Measured: a
    // low cap captures the cheap-spill wins (the functions that fit in a
    // handful of spills, which run ≈ legacy) while bailing the cascade /
    // block-param cases *fast* — no wasted compile, no runtime
    // regression. Eliminating the remaining bails needs live-range
    // splitting (Phase 7) and block-param spilling, not a higher cap
    // (a higher cap fits a few more functions but their reload-heavy code
    // regresses vs legacy).
    let max_iter: u32 = std::env::var("BEAGLE_SSA_SPILL_MAX_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let spill_cap: usize = std::env::var("BEAGLE_SSA_SPILL_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let mut spilled: Vec<VReg> = Vec::new();
    let mut iterations: u32 = 0;
    loop {
        iterations += 1;
        let liveness = compute_liveness(f);
        let ig = build_interference(f, &liveness);
        let cross = cross_safepoint_values(f, &liveness);
        let constraints = ClobberConstraints {
            cross_safepoint: &cross,
            callee_saved_gp,
            pool_gp: budget.gp,
        };
        let coloring = color_with_constraints(f, &ig, Some(&constraints));
        if fits_budget(&coloring, budget) {
            return AllocationResult {
                coloring,
                spilled,
                iterations,
                fits: true,
            };
        }
        if iterations >= max_iter || spilled.len() >= spill_cap {
            return AllocationResult {
                coloring,
                spilled,
                iterations,
                fits: false,
            };
        }

        let batch = pick_belady_victims(f, &liveness, &cross, budget, callee_saved_gp);
        if batch.is_empty() {
            // No spillable value at the over-pressure point (all block
            // params / used-at-op). Can't make progress — bail.
            return AllocationResult {
                coloring,
                spilled,
                iterations,
                fits: false,
            };
        }
        for victim in batch {
            spill_one(f, victim);
            spilled.push(victim);
        }
    }
}

/// True if every VReg's color fits within its class's budget. With the
/// clobber constraint applied during coloring, a cross-safepoint GP
/// value that can't get a callee-saved color is pushed to a color
/// `>= budget.gp`, so this single check covers both the total-GP and
/// cross-safepoint-GP budgets.
pub fn fits_budget(coloring: &Coloring, budget: Budget) -> bool {
    coloring.max_color(RegClass::Gp) < budget.gp && coloring.max_color(RegClass::Fp) < budget.fp
}

/// Belady victim selection. Finds the program point whose register
/// pressure most exceeds the binding budget and returns a **batch** of
/// spillable values live there, the furthest-next-use first (Belady —
/// the ones we'll need last). The batch size is the point's excess over
/// budget, so one recompute clears the worst point. Returns an empty
/// vec if no spillable (body-op-defined, not used at that point) value
/// is live there.
///
/// Two budgets bind (per the I7 clobber model): total GP ≤ `budget.gp`
/// and cross-safepoint GP ≤ `callee_saved_gp`. We relieve whichever is
/// worse — preferring to spill cross-safepoint values when the
/// callee-saved budget is the violated one (only those compete for it).
fn pick_belady_victims(
    f: &CfgFunction,
    liveness: &Liveness,
    cross: &HashSet<VReg>,
    budget: Budget,
    callee_saved_gp: u32,
) -> Vec<VReg> {
    // Body-op-defined GP vregs are the spillable set (`spill_one` only
    // handles those — block params and entry args need other machinery).
    let mut spillable: HashSet<VReg> = HashSet::new();
    for block in &f.blocks {
        for op in &block.body {
            for d in op.defs() {
                if d.class == RegClass::Gp {
                    spillable.insert(d);
                }
            }
        }
    }
    if spillable.is_empty() {
        return Vec::new();
    }

    // Linear RPO position of every op, and per-vreg sorted use positions
    // (for next-use distance). Terminator uses sit just past the body.
    let order = reverse_postorder(f);
    let mut op_pos: HashMap<(BlockId, usize), usize> = HashMap::new();
    let mut use_positions: HashMap<VReg, Vec<usize>> = HashMap::new();
    let mut next_idx = 0usize;
    for &bid in &order {
        let block = f.block(bid);
        for (i, op) in block.body.iter().enumerate() {
            op_pos.insert((bid, i), next_idx);
            for u in op.uses() {
                use_positions.entry(u).or_default().push(next_idx);
            }
            next_idx += 1;
        }
        // Terminator position.
        for u in block.terminator.uses() {
            use_positions.entry(u).or_default().push(next_idx);
        }
        next_idx += 1;
    }
    for v in use_positions.values_mut() {
        v.sort_unstable();
    }

    // Find the worst pressure point: walk each block backward (the same
    // walk `build_interference` uses), tracking the live set after each
    // op. Record the point that most violates the binding budget.
    let mut worst: Option<WorstPoint> = None;

    for &bid in &order {
        let block = f.block(bid);
        let mut live: HashSet<VReg> = liveness.live_out(bid).clone();

        // Terminator first.
        for &d in &block.terminator.defs() {
            live.remove(&d);
        }
        for u in block.terminator.uses() {
            live.insert(u);
        }

        for (i, op) in block.body.iter().enumerate().rev() {
            // `live` here = values live after this op = the clique at the
            // op's program point. Defs born here join that clique.
            let mut here: HashSet<VReg> = live.clone();
            for &d in &op.defs() {
                here.insert(d);
            }
            let pos = op_pos[&(bid, i)];
            let used_here: HashSet<VReg> = op
                .uses()
                .into_iter()
                .filter(|u| u.class == RegClass::Gp)
                .collect();
            consider_point(
                &here,
                &used_here,
                cross,
                budget,
                callee_saved_gp,
                pos,
                &mut worst,
            );

            for &d in &op.defs() {
                live.remove(&d);
            }
            for u in op.uses() {
                live.insert(u);
            }
        }
    }

    let Some(worst) = worst else {
        return Vec::new();
    };
    if !worst.over_budget {
        return Vec::new();
    }

    // Candidates: spillable GP values live at the worst point that are
    // NOT used by the op there (those must be in a register). When
    // relieving the callee-saved budget, only cross-safepoint values
    // count — spilling a non-cross value wouldn't lower cross-safepoint
    // pressure.
    let used_here = &worst.used_here;
    let mut candidates: Vec<(usize, VReg)> = worst
        .live_gp
        .iter()
        .copied()
        .filter(|v| spillable.contains(v) && !used_here.contains(v))
        .filter(|v| !worst.relieve_callee_saved || cross.contains(v))
        .map(|v| {
            let next_use = use_positions
                .get(&v)
                .and_then(|ps| ps.iter().find(|&&p| p > worst.pos).copied())
                .unwrap_or(usize::MAX);
            (next_use, v)
        })
        .collect();

    // Furthest next use first (Belady MIN). Tie-break on vreg index for
    // determinism.
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.index.cmp(&b.1.index)));

    // Spill the whole excess at this point in one batch, so a single
    // recompute clears it.
    let batch_size = (worst.severity.max(0) as usize).min(candidates.len());
    candidates
        .into_iter()
        .take(batch_size)
        .map(|(_, v)| v)
        .collect()
}

/// The worst pressure point found so far during the backward walk.
struct WorstPoint {
    pos: usize,
    live_gp: Vec<VReg>,
    used_here: HashSet<VReg>,
    over_budget: bool,
    /// True if the violated budget is the callee-saved (cross-safepoint)
    /// one rather than the total-GP one.
    relieve_callee_saved: bool,
    /// Severity used to pick the single worst point.
    severity: i64,
}

/// Score one program point and update `worst` if it's the new worst
/// budget violation.
fn consider_point(
    here: &HashSet<VReg>,
    used_here: &HashSet<VReg>,
    cross: &HashSet<VReg>,
    budget: Budget,
    callee_saved_gp: u32,
    pos: usize,
    worst: &mut Option<WorstPoint>,
) {
    let mut total_gp = 0i64;
    let mut cs_gp = 0i64;
    for &v in here {
        if v.class == RegClass::Gp {
            total_gp += 1;
            if cross.contains(&v) {
                cs_gp += 1;
            }
        }
    }
    let total_excess = total_gp - budget.gp as i64;
    let cs_excess = cs_gp - callee_saved_gp as i64;
    // The binding violation is the larger excess. Cross-safepoint excess
    // is relieved by spilling a cross-safepoint value specifically.
    let (severity, relieve_callee_saved) = if cs_excess >= total_excess && cs_excess > 0 {
        (cs_excess, true)
    } else {
        (total_excess, false)
    };
    let over_budget = severity > 0;
    if !over_budget {
        return;
    }
    let better = match worst {
        None => true,
        Some(w) => severity > w.severity,
    };
    if better {
        let live_gp: Vec<VReg> = here
            .iter()
            .copied()
            .filter(|v| v.class == RegClass::Gp)
            .collect();
        *worst = Some(WorstPoint {
            pos,
            live_gp,
            used_here: used_here.clone(),
            over_budget,
            relieve_callee_saved,
            severity,
        });
    }
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

        let result = allocate_with_spilling(&mut f, Budget { gp: 8, fp: 8 }, 8);
        assert_eq!(result.spilled.len(), 0, "no spill needed");
        assert_eq!(result.iterations, 1, "single coloring pass");
        assert!(result.fits, "fits the budget");
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

        let result = allocate_with_spilling(&mut f, Budget { gp: 4, fp: 8 }, 4);
        // Terminated. The 5 simultaneously-live entry params can't be
        // spilled (body-op defs only), so the spiller runs out of
        // candidates and returns fits=false rather than looping forever.
        assert!(result.iterations < 100, "should terminate quickly");
        assert!(!result.fits, "unsatisfiable: params can't be spilled");
    }

    /// Belady core: six body-op values all live simultaneously, budget
    /// of 4 GP. The spiller must spill enough (furthest-next-use first)
    /// to bring the coloring under budget — and it succeeds (fits=true)
    /// because these are all spillable body-op defs.
    #[test]
    fn belady_spills_high_pressure_to_fit() {
        let mut f = CfgFunction::new(Some("pressure".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        // Six constants, then a chain that keeps them all live to the
        // end (each AddInt consumes the running sum + the next const, so
        // every const is live until its turn). The six consts are
        // simultaneously live across the first AddInt → pressure 6+.
        let consts: Vec<VReg> = (0..6).map(|_| f.new_vreg(RegClass::Gp)).collect();
        for (i, &c) in consts.iter().enumerate() {
            f.block_mut(entry).body.push(Op::ConstTaggedInt {
                dst: c,
                value: i as i64,
            });
        }
        // Build a tree that uses all six near the end so they stay live.
        let mut acc = consts[0];
        for &c in &consts[1..] {
            let next = f.new_vreg(RegClass::Gp);
            f.block_mut(entry).body.push(Op::AddInt {
                dst: next,
                lhs: acc,
                rhs: c,
            });
            acc = next;
        }
        f.block_mut(entry).terminator = Terminator::Ret { value: acc };

        // Sanity: unspilled pressure exceeds the budget.
        let before = {
            let l = compute_liveness(&f);
            let (gp, _) = crate::cfg::regalloc::liveness::max_live(&f, &l);
            gp
        };
        assert!(before > 4, "test needs pressure > budget, got {}", before);

        let result = allocate_with_spilling(&mut f, Budget { gp: 4, fp: 8 }, 4);
        assert!(result.fits, "spiller should bring it under budget");
        assert!(!result.spilled.is_empty(), "some values were spilled");
        assert!(
            result.coloring.max_color(RegClass::Gp) < 4,
            "final coloring fits 4 GP colors"
        );
    }
}
