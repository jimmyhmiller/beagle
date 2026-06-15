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
use crate::cfg::{BlockId, CfgFunction, Op, RegClass, SlotId, Terminator, VReg};

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
    /// VRegs that were *rematerialized* (Phase 4) instead of slot-spilled:
    /// their defining op was a pure immediate materialization, so it is
    /// re-emitted at each use — no slot, no store, no GC-root pressure.
    /// These do not count against the spill cap.
    pub rematerialized: Vec<VReg>,
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
    // Color first; spill only on overflow; compute the spill set in one
    // Belady pass (no re-coloring inside it).
    //
    // Spill *target* = `callee_saved_gp` (9), not the full pool (12),
    // even though the real fit check is against the pool. Two reasons:
    // (1) it leaves headroom so the reload temporaries `spill_one` adds
    // can't push the real coloring back over the pool — so a single spill
    // round suffices; (2) it sidesteps the constrained-coloring subtlety:
    // reducing MaxLive ≤ 12 does NOT guarantee the *cross-safepoint-
    // constrained* greedy coloring fits in 12 (a cross value forbidden
    // the caller-saved colors can still be squeezed out), but reducing
    // MaxLive ≤ 9 does — at ≤9 live everywhere, the coloring uses ≤9
    // colors, all callee-saved, so the constraint never binds. A little
    // over-spill on overflow functions in exchange for a guaranteed fit;
    // optimizing toward the full 12 is a later refinement.
    let spill_target = Budget {
        gp: callee_saved_gp,
        fp: budget.fp,
    };
    // One spill round suffices for functions whose pressure the
    // whole-value `spill_one` can converge (the target-9 headroom
    // absorbs reload temps). High-pressure functions cascade — reloads
    // re-create pressure faster than spilling clears it — so cap the
    // rounds low and let them bail to legacy fast (which produces better
    // code for them than a 100-reload SSA spill anyway). True zero-bails
    // needs a per-use-reload / live-range-splitting spiller (a W-set
    // single pass) that doesn't cascade; that's the next step.
    let max_rounds: u32 = std::env::var("BEAGLE_SSA_SPILL_MAX_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    // Total-spill cap. Block-param spilling (`deliver_param_via_memory`)
    // lets *every* function fit, so without a cap there are zero bails —
    // but a function needing many spills compiles to reload-heavy code
    // that loses to the legacy fallback (memory traffic in hot loops).
    // The cap captures the cheap wins (functions that fit in a few
    // spills, whose SSA code is competitive) and bails the rest to legacy.
    // A value of 0 disables the cap (force-fit everything — diagnostic).
    // This is the runtime-parity knob until Phase 4 (remat) + Phase 5
    // (coalescing) make heavily-spilled code competitive.
    let spill_cap: usize = std::env::var("BEAGLE_SSA_SPILL_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    // Block-param (phi) spilling — `deliver_param_via_memory`. This is the
    // mechanism that clears the dominant remaining bail cause (merge /
    // loop-header blocks whose live phi-set exceeds the pool). It is
    // CORRECT and reaches zero bails, but it is **off by default**: it
    // delivers loop-carried values through memory, which regresses hot
    // loops (e.g. nbody `advance`) vs the legacy fallback that keeps them
    // in registers. It becomes a net win only after Phase 4 (remat) +
    // Phase 5 (coalescing) shrink the spilled code; enable it then.
    let block_param_spill = std::env::var("BEAGLE_SSA_BLOCK_PARAM_SPILL").is_ok();

    // Phase 4: rematerialization. When a value chosen for spilling is
    // defined by a pure immediate materialization (`ConstTaggedInt`,
    // `ConstRawValue`, `GetFramePointer`, …), re-emit that op at each use
    // instead of routing through a slot — one cheap instruction, zero
    // slot/store/GC-root pressure, and the value stops crossing safepoints
    // (it no longer needs a callee-saved color). Strictly cheaper than a
    // slot spill, so it's on by default; `BEAGLE_SSA_NO_REMAT` disables it
    // for A/B measurement.
    let remat_enabled = std::env::var("BEAGLE_SSA_NO_REMAT").is_err();

    let mut spilled: Vec<VReg> = Vec::new();
    let mut rematerialized: Vec<VReg> = Vec::new();

    // GC correctness — MANDATORY, independent of the perf spill budget/cap:
    // a value that might be a relocatable heap pointer and is live across a GC
    // safepoint MUST be slot-backed. The conservative GC scans frame slots,
    // not registers, and the allocator pool is callee-saved — so such a value
    // left in a register survives the call but is invisible to the GC and goes
    // stale if a moving collection runs (nested-struct corruption like
    // `op.field == op`). Force-spill these up front (spill_one reloads at each
    // use, so the value stops crossing safepoints in a register) so the
    // FUNCTION STAYS IN SSA — we slot only the specific unsafe values instead
    // of bailing the whole function to legacy. Provably non-pointer values
    // (ints/floats, via pointer_class) are exempt, preserving numeric SSA
    // perf. The rare value spill_one can't deliver (e.g. an unspillable block
    // param) stays cross-safepoint and is caught by verify_clobber_safety,
    // which bails just that function.
    {
        let liveness0 = compute_liveness(f);
        let cross0 = cross_safepoint_values(f, &liveness0);
        let non_pointer = crate::cfg::pointer_class::analyze(f).non_pointer_vregs;
        let mut mandatory: Vec<VReg> = cross0
            .iter()
            .copied()
            .filter(|v| v.class == RegClass::Gp && !non_pointer.contains(v))
            .collect();
        mandatory.sort_by_key(|v| v.index); // deterministic order
        for v in mandatory {
            // Rematerialization would only apply to pure-immediate (hence
            // non-pointer) defs, which are already excluded; spill to a slot.
            // Not tracked in `spilled` so it doesn't count against the perf
            // cap — these slots are correctness-mandatory.
            let _ = spill_one(f, v);
        }
    }

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
                rematerialized,
                iterations,
                fits: true,
            };
        }
        if iterations > max_rounds
            || (block_param_spill && spill_cap != 0 && spilled.len() > spill_cap)
        {
            // Over the round budget, or (when block-param spilling is on)
            // this function needs more real (slot) spills than are worth it
            // — bail to the (faster) legacy fallback. Rematerialized values
            // are free and don't count against the cap.
            return AllocationResult {
                coloring,
                spilled,
                rematerialized,
                iterations,
                fits: false,
            };
        }
        let to_spill = compute_spill_set(
            f,
            &liveness,
            &cross,
            spill_target,
            callee_saved_gp,
            block_param_spill,
        );
        if to_spill.is_empty() {
            // Nothing left to spill but still over budget — pure
            // entry-param overflow (needs de-promotion). Bail.
            return AllocationResult {
                coloring,
                spilled,
                rematerialized,
                iterations,
                fits: false,
            };
        }
        let mut made_progress = false;
        for v in to_spill {
            // Prefer rematerialization (Phase 4): if `v` is a pure
            // immediate def, re-emit it at each use rather than slot-spill.
            // Falls back to `spill_one` for non-remat defs and block params.
            if remat_enabled && rematerialize(f, v) {
                rematerialized.push(v);
                made_progress = true;
            } else if spill_one(f, v).is_some() {
                spilled.push(v);
                made_progress = true;
            }
        }
        if !made_progress {
            // Every candidate this round was a param the spiller refused
            // to deliver through memory (entry params / non-Jump edges).
            // No further progress possible — bail.
            let liveness = compute_liveness(f);
            let ig = build_interference(f, &liveness);
            let cross = cross_safepoint_values(f, &liveness);
            let constraints = ClobberConstraints {
                cross_safepoint: &cross,
                callee_saved_gp,
                pool_gp: budget.gp,
            };
            let coloring = color_with_constraints(f, &ig, Some(&constraints));
            let fits = fits_budget(&coloring, budget);
            return AllocationResult {
                coloring,
                spilled,
                rematerialized,
                iterations,
                fits,
            };
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

/// One program point: the GP values simultaneously live there (the
/// interference clique), and the subset "touched" by the op (its uses
/// and defs, or a block's params at its entry point). `pos` is the
/// linear RPO position, used for next-use distances.
struct SpillPoint {
    pos: usize,
    live_gp: Vec<VReg>,
    touched: HashSet<VReg>,
    kind: &'static str,
}

/// Single-pass Belady spill-decision. Returns the set of GP values to
/// spill so that, after spilling, no program point exceeds the budget —
/// **without** re-coloring (the old design's O(V²)-per-spill recompute
/// was what made high-pressure functions bail on a cap). Decision only;
/// the caller applies `spill_one` and colors once.
///
/// Two budgets bind (I7 clobber model): total GP ≤ `budget.gp` and
/// cross-safepoint GP ≤ `callee_saved_gp`. We greedily relieve the
/// worst-violated point, spilling the **furthest-next-use** value live
/// *through* it (Belady MIN) — i.e. not touched by the op there (an
/// operand can't be spilled to relieve its own op; `spill_one` would
/// just reload it right back). Spilling a value lowers pressure at every
/// point it was live-through; the def/use points keep a short-lived
/// reload temp, so they're unchanged. Block params count as spillable at
/// the op points they're live through (`spill_one` stores them at block
/// entry); a point whose only live values are touched (e.g. an entry
/// crowded with params) is unfixable and left over budget — that's the
/// residual `fits=false` case.
fn compute_spill_set(
    f: &CfgFunction,
    liveness: &Liveness,
    cross: &HashSet<VReg>,
    budget: Budget,
    callee_saved_gp: u32,
    block_param_spill: bool,
) -> Vec<VReg> {
    // Spillable = GP values `spill_one` can route through a slot: block
    // params + body-op defs, i.e. everything except terminator-defined
    // values (InlineBranch dst).
    let mut term_defined: HashSet<VReg> = HashSet::new();
    for b in &f.blocks {
        for d in b.terminator.defs() {
            term_defined.insert(d);
        }
    }
    let is_spillable = |v: VReg| v.class == RegClass::Gp && !term_defined.contains(&v);

    // ---- Pass 1: linear positions + per-value sorted use positions. ----
    let order = reverse_postorder(f);
    let mut pos_counter = 0usize;
    let mut use_positions: HashMap<VReg, Vec<usize>> = HashMap::new();
    let mut entry_pos: HashMap<BlockId, usize> = HashMap::new();
    let mut op_pos: HashMap<(BlockId, usize), usize> = HashMap::new();
    let mut term_pos: HashMap<BlockId, usize> = HashMap::new();
    for &bid in &order {
        entry_pos.insert(bid, pos_counter);
        pos_counter += 1;
        let block = f.block(bid);
        for (i, op) in block.body.iter().enumerate() {
            op_pos.insert((bid, i), pos_counter);
            for u in op.uses() {
                use_positions.entry(u).or_default().push(pos_counter);
            }
            pos_counter += 1;
        }
        term_pos.insert(bid, pos_counter);
        for u in block.terminator.uses() {
            use_positions.entry(u).or_default().push(pos_counter);
        }
        pos_counter += 1;
    }
    for v in use_positions.values_mut() {
        v.sort_unstable();
    }

    // ---- Pass 2: build the clique + touched set at every point. ----
    let mut points: Vec<SpillPoint> = Vec::new();
    // Map each (non-fn-entry) GP block param to its block-entry point, so
    // a param spill can credit the entry-pressure relief that the generic
    // "decrement where live-through" loop skips (a param is *touched*, not
    // live-through, at its own entry).
    let mut param_entry_idx: HashMap<VReg, usize> = HashMap::new();
    let gp = |s: &HashSet<VReg>| -> Vec<VReg> {
        s.iter()
            .copied()
            .filter(|v| v.class == RegClass::Gp)
            .collect()
    };
    let gp_set = |s: HashSet<VReg>| -> HashSet<VReg> {
        s.into_iter().filter(|v| v.class == RegClass::Gp).collect()
    };
    for &bid in &order {
        let block = f.block(bid);
        let mut live: HashSet<VReg> = liveness.live_out(bid).clone();

        // Terminator point.
        let tdefs = block.terminator.defs();
        let mut clique = live.clone();
        clique.extend(tdefs.iter().copied());
        let mut touched: HashSet<VReg> = tdefs.iter().copied().collect();
        touched.extend(block.terminator.uses());
        points.push(SpillPoint {
            pos: term_pos[&bid],
            live_gp: gp(&clique),
            touched: gp_set(touched),
            kind: block.terminator.kind_name(),
        });
        for &d in &tdefs {
            live.remove(&d);
        }
        for u in block.terminator.uses() {
            live.insert(u);
        }

        // Body op points (reverse).
        for (i, op) in block.body.iter().enumerate().rev() {
            let defs = op.defs();
            let mut clique = live.clone();
            clique.extend(defs.iter().copied());
            let mut touched: HashSet<VReg> = defs.iter().copied().collect();
            touched.extend(op.uses());
            points.push(SpillPoint {
                pos: op_pos[&(bid, i)],
                live_gp: gp(&clique),
                touched: gp_set(touched),
                kind: op.kind_name(),
            });
            for &d in &defs {
                live.remove(&d);
            }
            for u in op.uses() {
                live.insert(u);
            }
        }

        // Block-entry point: `live` is now live-in. Params are defined
        // here (touched) and join the clique. A non-entry block's params
        // are spillable via memory (`deliver_param_via_memory`); the
        // function entry's params are the calling convention and are not.
        let mut clique = live.clone();
        clique.extend(block.params.iter().copied());
        let is_fn_entry = bid == f.entry;
        points.push(SpillPoint {
            pos: entry_pos[&bid],
            live_gp: gp(&clique),
            touched: gp_set(block.params.iter().copied().collect()),
            kind: if is_fn_entry {
                "fn-entry"
            } else {
                "block-entry"
            },
        });
        if !is_fn_entry {
            let idx = points.len() - 1;
            for &p in &block.params {
                if p.class == RegClass::Gp {
                    param_entry_idx.insert(p, idx);
                }
            }
        }
    }

    // ---- Pressure arrays + value→points inverse map. ----
    let n = points.len();
    let mut total: Vec<i64> = vec![0; n];
    let mut cross_p: Vec<i64> = vec![0; n];
    let mut value_points: HashMap<VReg, Vec<usize>> = HashMap::new();
    for (idx, p) in points.iter().enumerate() {
        for &v in &p.live_gp {
            total[idx] += 1;
            if cross.contains(&v) {
                cross_p[idx] += 1;
            }
            value_points.entry(v).or_default().push(idx);
        }
    }

    let next_use_after = |v: VReg, pos: usize| -> usize {
        use_positions
            .get(&v)
            .and_then(|ps| ps.iter().find(|&&p| p > pos).copied())
            .unwrap_or(usize::MAX)
    };

    // ---- Greedy: relieve the worst-violated fixable point until none. ----
    let mut spilled: Vec<VReg> = Vec::new();
    let mut spilled_set: HashSet<VReg> = HashSet::new();
    let mut dead: Vec<bool> = vec![false; n];
    loop {
        let mut best_idx: Option<usize> = None;
        let mut best_sev = 0i64;
        let mut best_cross = false;
        for idx in 0..n {
            if dead[idx] {
                continue;
            }
            let tot_ex = total[idx] - budget.gp as i64;
            let cs_ex = cross_p[idx] - callee_saved_gp as i64;
            let (sev, is_cross) = if cs_ex >= tot_ex && cs_ex > 0 {
                (cs_ex, true)
            } else {
                (tot_ex, false)
            };
            if sev > best_sev {
                best_sev = sev;
                best_idx = Some(idx);
                best_cross = is_cross;
            }
        }
        let Some(idx) = best_idx else { break };

        let pt_pos = points[idx].pos;
        let touched = &points[idx].touched;
        // At a (non-fn) merge/loop-header entry, the `touched` values are
        // the phi params, which ARE spillable via memory
        // (`deliver_param_via_memory`) — so don't exclude them there. At a
        // body/term point the touched values are operands that would just
        // be reloaded back, so the exclusion still holds.
        let entry_relax = block_param_spill && points[idx].kind == "block-entry";
        let victim = points[idx]
            .live_gp
            .iter()
            .copied()
            .filter(|v| {
                is_spillable(*v)
                    && !spilled_set.contains(v)
                    && (entry_relax || !touched.contains(v))
                    && (!best_cross || cross.contains(v))
            })
            .max_by_key(|v| next_use_after(*v, pt_pos));

        let Some(v) = victim else {
            // Nothing spillable here (e.g. an entry crowded with params).
            // Leave it over budget; the final color reports fits=false.
            if std::env::var("BEAGLE_SSA_BAIL_DIAG").is_ok() {
                let p = &points[idx];
                let n_touched = p.live_gp.iter().filter(|v| p.touched.contains(v)).count();
                let n_through = p.live_gp.len() - n_touched;
                eprintln!(
                    "[bail-diag] UNFIXABLE kind={} pos={} live_gp={} touched={} through={} cross_here={} (cross_budget={}) total_budget={}",
                    p.kind,
                    p.pos,
                    p.live_gp.len(),
                    n_touched,
                    n_through,
                    p.live_gp.iter().filter(|v| cross.contains(v)).count(),
                    callee_saved_gp,
                    budget.gp,
                );
            }
            dead[idx] = true;
            continue;
        };

        spilled_set.insert(v);
        spilled.push(v);
        let in_cross = cross.contains(&v);
        // Spilling v removes it from every point it was live *through*
        // (not touched); def/use points keep a short reload temp.
        if let Some(pts) = value_points.get(&v) {
            for &q in pts {
                if !points[q].touched.contains(&v) {
                    total[q] -= 1;
                    if in_cross {
                        cross_p[q] -= 1;
                    }
                }
            }
        }
        // A param delivered via memory also frees its register at its own
        // block-entry point — which is `touched` (a def, not a use), so
        // the loop above skipped it. Credit that relief explicitly, else
        // the entry point never drops below budget and the greedy loop
        // would over-spill the whole phi set chasing it.
        if let Some(&ep) = param_entry_idx.get(&v) {
            total[ep] -= 1;
            if in_cross {
                cross_p[ep] -= 1;
            }
        }
    }

    spilled
}

/// Rewrite `f` to route `vreg` through a fresh stack slot. Returns the
/// slot id allocated, or `None` if `vreg` is a block param the spiller
/// can't safely deliver through memory (entry params = calling
/// convention; non-`Jump` incoming edges) — the caller treats `None` as
/// "didn't spill" and the function bails to legacy.
///
/// Two shapes:
/// - **Body-op def:** insert `SlotStore` after the def and a per-use
///   `SlotLoad` before each use. `vreg`'s live range shrinks to def→store
///   plus per-use load spans.
/// - **Block param (phi):** the param has no register home of its own.
///   Each predecessor delivers it via memory: store the edge's outgoing
///   arg to the slot at the end of the predecessor, drop that arg from
///   the edge, and remove the param. Uses in the block reload per-use.
///   This is the only way to lower register pressure that concentrates
///   in a merge/loop-header's phi set (the dominant SSA bail cause), and
///   it relies on critical edges already being split (so every incoming
///   edge to a multi-pred block is a plain `Jump` with one arg list).
///
/// Caller re-runs liveness / interference / coloring afterward.
pub fn spill_one(f: &mut CfgFunction, vreg: VReg) -> Option<SlotId> {
    // Is `vreg` a block param? (A param is defined in exactly one block.)
    let param_site = f.blocks.iter().enumerate().find_map(|(bi, b)| {
        b.params
            .iter()
            .position(|p| *p == vreg)
            .map(|k| (BlockId(bi as u32), k))
    });

    // Route by class so the destination is GC-correct (I9): GP spills go
    // to a scanned root slot (the GC must see the pointer), FP spills go
    // to the unscanned region (a raw f64 must not be misread as a heap
    // pointer). See `CfgFunction::alloc_slot_for`.
    let slot = f.alloc_slot_for(vreg.class);
    if let Some((b, k)) = param_site {
        // Refuse (no mutation done yet) if delivery isn't safe.
        if !deliver_param_via_memory(f, vreg, b, k, slot) {
            return None;
        }
    }
    rewrite_uses_and_defs(f, vreg, slot);
    Some(slot)
}

/// Deliver phi param `vreg` (index `k` of block `b`) through `slot`
/// instead of a register: each predecessor stores its outgoing edge arg
/// to `slot` and drops it from the edge, and the param is removed from
/// `b`. Returns `false` (making no change) when the param can't be
/// delivered this way: `b` is the entry block (params are the calling
/// convention) or some predecessor reaches `b` via a non-`Jump`
/// terminator (only split-edge `Jump`s are safe to prepend a store to —
/// a `Branch`/`InlineBranch`/`Throw` edge to a multi-pred block would be
/// a critical edge, which the pipeline already split). After this, uses
/// of `vreg` in `b` are reloaded per-use by `rewrite_uses_and_defs`.
fn deliver_param_via_memory(
    f: &mut CfgFunction,
    vreg: VReg,
    b: BlockId,
    k: usize,
    slot: SlotId,
) -> bool {
    debug_assert_eq!(f.block(b).params.get(k), Some(&vreg));
    if b == f.entry {
        return false;
    }
    // Every incoming edge to `b` must be a plain `Jump` (one arg list).
    let num_blocks = f.blocks.len();
    for bid_idx in 0..num_blocks {
        let term = &f.block(BlockId(bid_idx as u32)).terminator;
        let edges = match term {
            Terminator::Jump { target, .. } => (*target == b) as usize,
            other => {
                // A non-Jump terminator targeting `b`: refuse.
                if other.successors().iter().any(|s| *s == b) {
                    return false;
                }
                0
            }
        };
        debug_assert!(edges <= 1);
    }
    // Safe: store each predecessor's arg `k` to the slot, drop it.
    for bid_idx in 0..num_blocks {
        let bid = BlockId(bid_idx as u32);
        let arg = match &mut f.block_mut(bid).terminator {
            Terminator::Jump { target, args } if *target == b => Some(args.remove(k)),
            _ => None,
        };
        if let Some(src) = arg {
            f.block_mut(bid).body.push(Op::SlotStore { slot, src });
        }
    }
    f.block_mut(b).params.remove(k);
    true
}

/// Insert per-use `SlotLoad`s (and, for body-op defs, a `SlotStore` after
/// each def) for `vreg` against `slot`, across every block and terminator.
fn rewrite_uses_and_defs(f: &mut CfgFunction, vreg: VReg, slot: SlotId) {
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
}

/// If `op` is a *rematerializable* definition — a pure materialization
/// whose value depends only on encoded immediates, with **no register
/// uses** — return an equivalent op writing `dst`. Such an op produces
/// the identical value wherever it's placed, so it can be re-emitted at a
/// use site instead of being spilled to a slot (Phase 4). Returns `None`
/// for anything with register operands or side effects.
///
/// `GetFramePointer` qualifies: FP is fixed for the frame's lifetime, so
/// reloading it is one register-to-register/immediate op. The stack
/// pointer ops are deliberately excluded — SP moves with push/pop.
fn remat_op_for(op: &Op, dst: VReg) -> Option<Op> {
    match op {
        Op::ConstTaggedInt { value, .. } => Some(Op::ConstTaggedInt { dst, value: *value }),
        Op::ConstRawValue { value, .. } => Some(Op::ConstRawValue { dst, value: *value }),
        Op::ConstPointer { ptr, .. } => Some(Op::ConstPointer { dst, ptr: *ptr }),
        Op::ConstStringPtr { ptr, .. } => Some(Op::ConstStringPtr { dst, ptr: *ptr }),
        Op::ConstKeywordPtr { ptr, .. } => Some(Op::ConstKeywordPtr { dst, ptr: *ptr }),
        Op::ConstFunctionId { function_id, .. } => Some(Op::ConstFunctionId {
            dst,
            function_id: *function_id,
        }),
        Op::ConstTrue { .. } => Some(Op::ConstTrue { dst }),
        Op::ConstFalse { .. } => Some(Op::ConstFalse { dst }),
        Op::ConstNull { .. } => Some(Op::ConstNull { dst }),
        Op::GetFramePointer { .. } => Some(Op::GetFramePointer { dst }),
        _ => None,
    }
}

/// Phase 4: rematerialize `vreg` instead of spilling it. If `vreg` is
/// defined by a rematerializable body op (`remat_op_for`), delete that
/// def and re-emit the op (with a fresh dst) immediately before every use
/// — each reload is one cheap instruction with zero slot/store/GC-root
/// pressure, and the value no longer crosses any safepoint. Returns
/// `true` on success; `false` (making **no** mutation) when `vreg` is not
/// a rematerializable body-op def (a block param, or a computed value) —
/// the caller falls back to `spill_one`.
///
/// Like `spill_one`, this shrinks the live range to per-use spans, so the
/// `compute_spill_set` pressure accounting (decrement at every
/// live-through point) holds unchanged. Caller re-runs liveness /
/// interference / coloring afterward.
fn rematerialize(f: &mut CfgFunction, vreg: VReg) -> bool {
    // Locate the single defining body op and capture its remat template
    // (read-only — bail before mutating if it isn't rematerializable).
    let mut template: Option<Op> = None;
    let mut def_site: Option<(BlockId, usize)> = None;
    'find: for bi in 0..f.blocks.len() {
        let bid = BlockId(bi as u32);
        for (i, op) in f.block(bid).body.iter().enumerate() {
            if op.defs().iter().any(|d| *d == vreg) {
                template = remat_op_for(op, vreg);
                def_site = Some((bid, i));
                break 'find;
            }
        }
    }
    let (Some(template), Some((def_bid, def_idx))) = (template, def_site) else {
        return false;
    };

    for bi in 0..f.blocks.len() {
        let bid = BlockId(bi as u32);
        let body = std::mem::take(&mut f.block_mut(bid).body);
        let mut new_body: Vec<Op> = Vec::with_capacity(body.len() + 4);
        for (i, mut op) in body.into_iter().enumerate() {
            // Drop the original def — after renaming all uses it is dead,
            // and removing it directly relieves pressure at the def point.
            if bid == def_bid && i == def_idx {
                continue;
            }
            if op.uses().iter().any(|u| *u == vreg) {
                let fresh = f.new_vreg(vreg.class);
                new_body
                    .push(remat_op_for(&template, fresh).expect("template is rematerializable"));
                let mut rename = HashMap::new();
                rename.insert(vreg, fresh);
                op.rename_uses(&rename);
            }
            new_body.push(op);
        }
        f.block_mut(bid).body = new_body;

        // Terminator use: re-emit at the end of the (rewritten) body.
        if f.block(bid).terminator.uses().iter().any(|u| *u == vreg) {
            let fresh = f.new_vreg(vreg.class);
            f.block_mut(bid)
                .body
                .push(remat_op_for(&template, fresh).expect("template is rematerializable"));
            let mut rename = HashMap::new();
            rename.insert(vreg, fresh);
            f.block_mut(bid).terminator.rename_uses(&rename);
        }
    }
    true
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

        let slot = spill_one(&mut f, v).expect("body-op def is always spillable");
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
        assert!(
            !result.spilled.is_empty() || !result.rematerialized.is_empty(),
            "some values were lowered (spilled or rematerialized)"
        );
        assert!(
            result.coloring.max_color(RegClass::Gp) < 4,
            "final coloring fits 4 GP colors"
        );
    }

    /// Phase 4: high pressure made entirely of constant defs is relieved
    /// by *rematerialization*, not slot spilling — no SlotStore/SlotLoad
    /// is emitted, the constants are simply re-materialized at their uses.
    #[test]
    fn rematerializes_consts_instead_of_spilling() {
        let mut f = CfgFunction::new(Some("remat".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        // Six tagged-int constants, kept simultaneously live by a tree
        // that consumes them near the end (same shape as the Belady test).
        let consts: Vec<VReg> = (0..6).map(|_| f.new_vreg(RegClass::Gp)).collect();
        for (i, &c) in consts.iter().enumerate() {
            f.block_mut(entry).body.push(Op::ConstTaggedInt {
                dst: c,
                value: i as i64,
            });
        }
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

        let result = allocate_with_spilling(&mut f, Budget { gp: 4, fp: 8 }, 4);
        assert!(result.fits, "remat should bring it under budget");
        assert!(
            !result.rematerialized.is_empty(),
            "constants should be rematerialized"
        );
        assert!(
            result.spilled.is_empty(),
            "no slot spill needed — all pressure was const-defined"
        );
        // No memory traffic was introduced.
        let body = &f.block(entry).body;
        assert!(
            !body
                .iter()
                .any(|op| matches!(op, Op::SlotStore { .. } | Op::SlotLoad { .. })),
            "rematerialization must not emit slot loads/stores: {:?}",
            body
        );
        assert!(
            result.coloring.max_color(RegClass::Gp) < 4,
            "final coloring fits 4 GP colors"
        );
    }
}
