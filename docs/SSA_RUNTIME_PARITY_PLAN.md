# SSA Runtime Parity Plan

## Purpose & status

`BEAGLE_USE_SSA=1` is correct (360/360) but **slower at runtime** than the
legacy allocator: fib ~25ms→~43ms (1.7×), nbody_full ~34ms→~66ms. This
document is the engineering plan to reach **runtime parity** (SSA ≤ legacy on
the benchmark set) and is scoped to runtime only — compile-time is explicitly
out of scope for this effort.

It is a companion to `docs/SSA_ARCHITECTURE.md` (the invariants/contract).
Where this plan touches `src/cfg/`, `src/ssa/`, `src/register_allocation/`,
`src/ir.rs`, `src/lir/`, `src/compiler.rs`, or backend regalloc/spill code, the
SSA_ARCHITECTURE invariants (I1–I10) and `/ssa-review` still govern.

Status: **Phase 0 landed; Phases 1–7 not started.** Phases land in order; each
is independently gated (see "Gating" below). The working-tree gate relaxation
(Phase 6) has been reverted per "Disposition" and will return last.

---

## Diagnosis (root cause)

The allocator forces every function into a lose-lose choice, because three
production-allocator capabilities are missing:

1. **Tiny register pool.** `regalloc/physical.rs` `allocator_gp = X19..X27` — **9
   GP registers, all callee-saved.** No caller-saved registers are used.
2. **No working spiller.** `regalloc/spill.rs::allocate_with_spilling` exists but
   is (a) not wired into `compile_via_ssa`, (b) naive "spill-everywhere", (c)
   bails on a no-progress heuristic, (d) body-op defs only, (e) not GC-aware.
   `compile_via_ssa` instead calls `has_overflow` → **bails the whole function
   to legacy** when coloring needs > 9 GP.
3. **Incomplete coalescing / no rematerialization.** Block-param transfers and
   `Move` chains survive as real register shuffles; constants are kept live
   instead of re-emitted.

So a function either:

- **Keeps values in slots** (mem2reg's conservative `>= 2`-read gate at
  `cfg/mem2reg.rs:53`) → heavy `SlotLoad`/`SlotStore` memory traffic → ~1.5–1.7×
  slower than legacy, which keeps the same values in registers; or
- **Promotes them to registers** → exceeds the 9-GP pool → SpillOverflow → bail
  to legacy (no win), or — where it fits — pays phi/edge-move + pressure costs
  that erase the win.

### Evidence

- **fib CFG dump** (`BEAGLE_SSA_DUMP_FN=fib/fib`): after mem2reg the body still
  has **25 `SlotStore` + 16 `SlotLoad`** for ~6 real live values. Slots 1/2/4/5
  are `SlotStore slot(k) <- v` immediately followed by `w = SlotLoad slot(k)`
  with no call between — pure round-trip traffic legacy keeps in registers.
- **`sample` profile on fib(37)** (prior investigation): 93% of samples are in
  the JIT'd fib code itself, not runtime functions — i.e. the SSA-emitted machine
  code is ~1.5× slower instruction-for-instruction on the same input IR.
- **Failed experiment (2026-05-29):** relaxing the gate `>= 2` → `>= 1` (GC-safe;
  see below) cut fib slot ops 41→7 but **regressed nbody 66ms→151ms (2.3×)** and
  pushed fib to SpillOverflow-bail. Proof that promotion only pays once the
  backend can absorb it — the gate relaxation must come **last**, not first.

### The theory we exploit

Hack 2006 / Braun-Hack 2009 (the basis of LLVM-greedy and Cranelift's regalloc2):
**in SSA, spilling and coloring decouple.** The interference graph is chordal, so
once register pressure (MaxLive) is ≤ k at every program point, chordal coloring
is **guaranteed** to succeed in k colors. `regalloc/color.rs` already does optimal
chordal coloring (dom-tree pre-order PEO); its `max_color` *is* the exact
pressure. The missing half is a spiller that lowers MaxLive to ≤ k — then there is
no "overflow," ever.

---

## What already exists and is correct (do not rebuild)

| Component | File | State |
|---|---|---|
| CFG + mem2reg + I1–I10 verifiers | `src/cfg/*`, `cfg/gc_safety.rs` | Solid |
| SSA chordal coloring (dom-tree PEO, optimal) | `regalloc/color.rs` | Correct; `max_color` = exact pressure |
| Biased coloring (block-param↔arg hints) | `color.rs::build_copy_hints` | Phase-5 start, partial |
| Interference graph (backward walk) | `regalloc/interference.rs` | Correct, but **no clobber modeling** |
| Parallel-copy edge resolution | `regalloc/edge.rs` | Correct (relies on I2 critical-edge split) |
| Spiller skeleton | `regalloc/spill.rs` | Naive; bails; not GC-aware; **not wired in** |
| Physical pool / overflow sentinel | `regalloc/physical.rs` | 9 GP callee-saved; overflow→bail |
| Lower + translate → legacy emit | `regalloc/emit.rs`, `cfg/emit_legacy.rs` | Works |

---

## GC facts that constrain the design (verified)

- The GC scans `header.size` contiguous frame slots and is **conservative-by-tag**:
  it follows a slot only if `is_heap_pointer(value)` (low 3 tag bits).
  (`gc/stack_walker.rs:106-148`)
- `header.size` is driven by `set_max_locals(cfg.num_slots)` (`emit_legacy.rs:79,863`).
  So **any slot in the scanned range holding a GP value is automatically a GC
  root** — exactly what I9 needs for cross-call spills — but **a raw f64 spilled
  there can be misread as a pointer.** FP spills must live *outside* the scanned
  region.
- `Op::Call` already carries `clobbers = all-caller-saved`, but
  `interference.rs` ignores it today.
- GC-safepoint set is centralized in `cfg/gc_safety.rs::is_gc_safepoint_op` /
  `is_gc_safepoint_terminator` — reuse it; do not re-enumerate safepoints.

### Why the gate relaxation is GC-safe (for the record)

The mem2reg `>= 2`-read profitability filter and the GC-safety gate are
**separate, sequential filters**. `slot_is_gc_safe_to_promote` (I9) and the
handler-read gate (I10) run *downstream* of the read-count filter and are
read-count-agnostic. Relaxing `>= 2` → `>= 1` only adds candidates; every one
still passes I9/I10. Verified: fib promotes 28/30, correctly keeping slot 3
(cross-call `fib(n-1)` result) and slot 0 (closure ptr) materialized. The
relaxation's problem is **runtime (pressure/moves)**, not GC.

---

## The plan — 8 phases

### Gating (applies to every phase)

The spec's promotion bar, enforced before a phase lands:

- 100% of `cargo run -- test resources/` pass — including `gc-always` / GC-stress
  files.
- `BEAGLE_SSA_VERIFY=1` on and passing (plus the new per-phase verifiers).
- `/ssa-review` clean (no new forbidden-pattern hits; spec updated in the same
  commit if an invariant must change).
- Phase-0 differential harness shows **no per-function spill/move/runtime
  regression** vs the prior phase, on the benchmark set.

Landing out of order is a blocking review failure (the order is load-bearing —
see "Why this order").

---

### Phase 0 — Measurement harness (no behavior change)

Extend `BEAGLE_SSA_REGALLOC_STATS` to emit per-function **MaxLive(GP/FP),
colors, spills, root-slots, edge-moves.** Build a differential runner over the
benchmark set (fib, nbody_full, btrees, large_concat, …) recording runtime +
counts for legacy vs SSA, that **fails if SSA regresses.** This is the oracle
that gates Phases 1–7. Every prior attempt regressed silently; this stops that.

**Done when:** harness runs in one command, prints a legacy-vs-SSA table, exits
nonzero on regression.

**Landed (2026-05-29).** `BEAGLE_SSA_REGALLOC_STATS` now emits per-function
`maxlive_gp/maxlive_fp colors_gp/colors_fp edges edge_moves root_slots`
(`src/cfg/regalloc/stats.rs`; `liveness::max_live` is the new MaxLive probe and
a chordal cross-check — MaxLive must equal `colors`). The differential runner is
`scripts/ssa_diff.sh` (baseline in `scripts/ssa_diff_baseline.tsv`):
`scripts/ssa_diff.sh` measures + gates; `--update-baseline` rewrites it;
`--heavy` adds the ~11s `bench_btrees_full`. It times only the benchmark's own
`core/time-now()` hot-loop measurement (runtime, not compile), and gates on SSA
runtime / `bails` (the pre-Phase-3 spill proxy) / `edge_moves` vs the saved
baseline, exiting nonzero on regression. Run in **release** — the per-instruction
debugger pretty-print hook (`debug_only!`) is debug-only and unrelated.

First baseline (release, ARM64, reps=2): fib_specialize SSA/legacy ≈ **1.12×**,
nbody_specialize ≈ 1.10×, nbody_full ≈ 0.88× (already at/under parity),
btrees_specialize ≈ 0.88–1.0× (noisy). Whole-compile aggregates: ~142–148
function bails to legacy, ~11 edge moves, peak `maxlive_gp` 125 (a stdlib
mega-function). These are the numbers Phases 1–7 must improve without
regressing.

---

### Phase 1 — GC-correct slot model (foundation, no perf change)

Split the frame slot space into a **scanned GP-root region** and a **non-scanned
region** (FP spills + raw non-pointer scratch); set `header.size` to scan only
the root region. Centralize root-slot allocation by porting the proven reuse
logic from `register_allocation/linear_scan.rs:209` (`assign_root_slots`,
free-list reuse of non-overlapping slots) into the CFG slot allocator.

**New verifier:** no FP/raw slot falls in the scanned range; every GP slot that
can hold a tagged pointer while live at a safepoint is in it.

**Rationale:** the spiller (Phase 3) must have a GC-correct destination *before*
it starts moving values, so it cannot introduce an I9 violation. This is the
GC foundation.

**Done when:** slot layout is split, `header.size` excludes FP/raw slots,
verifier passes, GC-stress green.

**Increment 1 landed — slot-region model + verifier.** `SlotId` now carries a
region (`UNSCANNED_SLOT_BASE` high-bit split; `is_unscanned()` /
`region_index()`); `CfgFunction` grows `num_unscanned_slots` and
`alloc_root_slot` / `alloc_unscanned_slot` / `alloc_slot_for(class)` (GP→root,
FP→unscanned). `spill_one` routes spills by class so FP spills are GC-correct by
construction. New verifier `check_gc_slot_regions` (I9) rejects FP-in-scanned,
GP-in-unscanned, mixed-class, and out-of-range unscanned slots; wired into
`verify()`. `translate` hard-errors on an unscanned slot (no silent bad
`Local` index) since backend addressing isn't in yet. No behavior change: the
spiller isn't wired into `compile_via_ssa`, so nothing emits unscanned slots;
full suite 364/364 green under legacy and SSA, no `SlotRegionError` across the
corpus (verified `build_cfg` never stores an FP value to a root slot).

**Increment 2 — deferred to Phase 3 (decision 2026-05-29).** The backend
frame-emit support — a `Value`-level unscanned-slot addressing mode placed
*below* the eval-stack region, SP reservation including it, and the
`assign_root_slots` free-list reuse port — will be co-developed with the spiller
in Phase 3, its only producer, so it is validated against real FP spills rather
than synthetic tests alone. The committed model + verifier already guarantee the
spiller cannot introduce an I9 violation at the model level (FP→unscanned,
GP→root enforced; `translate` hard-errors on an unscanned slot until the backend
addressing lands), which is the prerequisite Phase 3 needs.

Frame facts confirmed for Phase 3: total frame =
`2 + max_stack_size + max_locals + num_callee_saved` words; GC scans a
*contiguous* `header.size = max_locals + max_stack_size` prefix from `[FP-24]`;
callee-saved sit SP-relative at the bottom. The SSA path *does* use the eval
stack (`PushStack`/`PopStack`), so unscanned slots must sit below it — there is
no existing non-scanned region to reuse (the legacy "spill area" = root slots =
scanned). The catch Phase 3 must handle: an unscanned slot's FP-relative offset
depends on the *final* `max_stack_size`, which isn't known at body-emit time, so
it needs offset finalization (the codebase patches only the prologue `SUB SP`
today, not body load/store offsets).

---

### Phase 2 — Grow the pool + per-call clobber interference (I7)

Add a **caller-saved GP sub-pool** to `physical.rs`. Free immediately: **X13,
X14, X15**. Then a scratch-convention rework to reclaim X9/X10/X11/X12/X16/X17
(reserve a minimal fixed scratch set for lowering per **F11**; route op-scratch —
tag/modulo/guard/edge-cycle-break — through reserved regs only).

In `interference.rs`, **every value live across a safepoint op interferes with
the entire caller-saved sub-pool** (read the clobber from `Op::Call.clobbers`
and `gc_safety::is_gc_safepoint_op`). This forces cross-call values into
callee-saved registers or spills, and **models the clobber in the graph** instead
of the ad-hoc detection that made the previous caller-saved attempt clobber a
live entry-param (the documented revert in
`memory/project-ssa-perf-baseline`). Coloring biases short-lived (non-cross-call)
values to caller-saved → zero prologue cost — the case where legacy wins today.

**Rationale:** more registers + a correct clobber model = the function fits
without spilling in the common case, with cheap caller-saved temps.

**Done when:** pool is larger, per-call interference verifier passes
(`variadic_recursive_test` and the GC-stress suite specifically green — the
exact cases the prior attempt broke), no runtime regression.

**Landed (2026-05-29).** GP pool grown 9→12: `physical.rs` `allocator_gp` now
`[X19..X27 (callee-saved), X13, X14, X15 (caller-saved)]` with split point
`callee_saved_gp = 9` (X13–15 are AAPCS temporaries unused by the backend's
scratch convention — temps X10/X11/X12, scratch X16/X17 — so free to allocate;
`mark_callee_saved_register_used` no-ops on them → zero prologue cost). The
clobber model is realized as a **color constraint** (`color.rs`
`ClobberConstraints` / `color_with_constraints`): every value in
`interference::cross_safepoint_values` (live after any `gc_safety` safepoint op /
terminator — the I9 source of truth, conservative) is forbidden the caller-saved
color range, so a cross-call value gets a callee-saved color or overflows→bails —
identical to pre-Phase-2 behavior. **Provably non-regressing:** for any function
fitting in ≤9 GP the constraint never binds and caller-saved colors are never
reached, so the coloring is byte-identical to before; only previously-bailing
functions gain X13–15. End-to-end `verify_clobber_safety` (run in
`compile_via_ssa`, bail-to-legacy on violation) checks no cross-safepoint GP value
landed in a caller-saved reg *including* the arg regs (the entry-param case).

Result: **bails dropped ~145 → ~64** across the benchmark compile set, no runtime
regression, 364/364. Baseline re-cut to lock in the bail win.

Surfaced + fixed a latent bug: `variadic_recursive_test/count_items` (which used
to bail on pool pressure) compiles via SSA under the bigger pool, but
`build_cfg`'s variadic prologue leaves the rest-array vreg **undefined** in the
CFG → it printed garbage. Added `verify::first_undefined_use` (a false-positive-
free slice of the dominance check) wired into `compile_via_ssa` to bail any
malformed CFG to legacy (spec: a verifier failure falls back to legacy). The
underlying `build_cfg` variadic-lowering gap is a known limitation — variadic
functions with a named prefix param bail to legacy until it's fixed.

---

### Phase 3 — Real SSA spiller (Belady, decoupled, GC-aware) — the core

Replace spill-everywhere/bail with **MaxLive reduction**: at each program point
where pressure > k (per class), spill the value with the **furthest next use**
(Belady MIN), with **live-range splitting** at low-pressure points (Wimmer)
rather than whole-interval spilling. Reload is inserted before the next use.

- GP cross-safepoint spills → Phase-1 **root slots** (per-call clobber = spill
  around the call, reload after — I7).
- FP spills → Phase-1 **non-scanned region**.
- Add the **postcondition verifier**: pressure ≤ k everywhere ⟹ coloring cannot
  fail. **Delete the SpillOverflow→bail path** in `compile_via_ssa` (replace with
  a debug assertion it can't happen).
- Wire `allocate_with_spilling` into `compile_via_ssa` in place of
  `has_overflow`.

**Rationale:** the core. With a correct spiller no function ever bails to legacy;
it spills minimally instead. Belady is optimal for straight-line code; the SSA
extension is near-optimal. This is what production JITs do, and it removes the
double-compile for the ~15–23% of functions that bail today.

**Done when:** zero SpillOverflow bails over a full stdlib+test run, postcondition
verifier on, spill counts ≤ legacy ± tolerance, runtime ≤ legacy on functions
that previously bailed.

**Landed (2026-05-29) — spiller in, but "zero bails" deferred to Phase 7 (plan
deviation, see below).** `allocate_with_spilling` is now a clobber-aware Belady
spiller wired into `compile_via_ssa` in place of the old color-once→bail. Each
iteration colors under the I7 constraint (recomputing the cross-safepoint set,
since spilling changes liveness) and, while over budget, spills the **batch** of
furthest-next-use values at the worst pressure point — one recompute per
high-pressure region, not per value (per-value recompute was pathologically slow:
a function needing ~25 spills did 25 full liveness+interference+color passes;
fib didn't finish compiling in 180s). GP spills → GC-scanned root slots
(`spill_one` routes by class, I9); the fit check covers both budgets (total GP
≤ pool, cross-safepoint GP ≤ callee-saved) because the constraint pushes an
un-colorable cross-safepoint value past the pool size. Postcondition: `has_overflow`
after a reported fit is a `debug_assert!(false)`.

**Deviation from "delete the bail path."** The bail path is *kept*, not deleted.
The basic `spill_one` reloads at every use, so heavily-over-pressure functions
**cascade** (reloads re-create pressure and never converge) and bail at a
`spill_cap`; and ~11/53 bails are pressure concentrated in **block params**, which
`spill_one` can't spill at all. Eliminating those needs **live-range splitting
(Phase 7)** and block-param spilling — so true "zero bails" moves to Phase 7, not
Phase 3. Measured: a low `spill_cap` (default 8) is strictly best — it captures
the cheap-spill wins (bails **67→59 / 61→53** across the benchmark compile sets,
~8 functions now fit) while bailing cascade/block-param cases *fast* (no wasted
compile, no runtime regression). A higher cap fits a few more but their
reload-heavy code regresses vs the legacy fallback, so it's not used. Runtime is
≈ parity (SSA/legacy ratios: nbody 0.99, btrees 1.01, fib 1.12); 364/364 with the
spiller actively rewriting CFGs. The unscanned-FP-slot backend work stays deferred
— no function in the corpus FP-overflows (`maxlive_fp` is always 2).

---

### Phase 4 — Rematerialization

When a const / address-load would be spilled, **re-emit it at the reload point**
instead (1 cheap instruction, zero slot pressure). Rematerializable defs:
`ConstTaggedInt`, `ConstRawValue`, `ConstPointer`, `GetFramePointer`. `cfg/opt.rs`
already has a rematerialization *annotation* stub — implement the emit side and
have the spiller prefer remat over slot spill for marked defs.

**Rationale:** fib/nbody are saturated with constants; rematerializing removes
most spill pressure for free and is the cheapest possible reload.

**Done when:** spill count drops on const-heavy functions; runtime improves or
holds; no correctness regression.

---

### Phase 5 — Complete coalescing

Promote the biased hints (`color.rs::build_copy_hints`) to **full conservative
coalescing** (Briggs/George) over all copy-related pairs: block-param↔incoming
arg, `Move` chains, and two-address op constraints. Coalesced pairs share a color
→ edge resolution emits zero moves; the `v65→v66→v67→v68→v69` chains in the fib
dump disappear.

**Rationale:** closes the "phi/edge moves cost as much as the slot traffic" gap
that made the relaxed-gate nbody case regress 2.3×.

**Done when:** edge-move count per function ≤ legacy; no runtime regression.

---

### Phase 6 — *Now* relax the mem2reg gate (`>= 2` → `>= 1`)

Reintroduce the shelved one-line change at `cfg/mem2reg.rs:53`. With spiller +
remat + coalescing + bigger pool in place, promoting single-read store→load
round-trips finally pays: overflow → spill (not bail), transfers → coalesced
away.

**Rationale & evidence:** doing this *first* was a confirmed 2.3× nbody
regression. It is the **last** step, valid only once the backend can exploit the
extra promotions.

**Done when:** the nbody regression is gone; fib + nbody runtime ≤ legacy; full
suite + GC-stress green.

---

### Phase 7 — Dead-init + dead-phi cleanup

Add mem2reg's read-before-write analysis so the synthetic zero-inits
(`ConstRawValue 0`) are emitted **only** for slots with an actual
read-before-write path, and prune the dead phis they feed. (fib's entry showed
**10 dead `ConstRawValue 0x0`** that survived DCE as dead phi args and helped
tip the SpillOverflow.)

**Done when:** dead entry-inits gone for always-written-before-read slots;
pressure drops further; no correctness regression.

---

## Why this order

The sequence mirrors how production allocators are built and is the exact inverse
of what failed:

1. **GC-correct slot destination + bigger pool + clobber model** come *before* the
   spiller — the spiller needs a safe place to put things, and the pool defines k.
2. **Spiller** second — once pressure can be lowered to ≤ k, coloring never
   overflows and nothing bails.
3. **Remat + coalescing** third — they reduce the pressure and moves that
   promotion would otherwise create.
4. **mem2reg gate relaxation** last — proven by experiment to only pay once 1–5
   are in place.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Caller-saved reg clobbers a live cross-call value (the prior revert) | Model the clobber in the interference graph (Phase 2), not ad-hoc detection |
| Raw f64 spill misread as a heap pointer by GC | Slot-region split; FP spills outside the scanned range (Phase 1) |
| Spiller non-termination (the current no-progress bail) | Belady MaxLive reduction is single-pass per point — provably terminating, no color-respill loop |
| Silent regression | Phase-0 differential harness gates every phase |
| Handler/soft-edge blocks (I10) break under spilling | Spiller consults `gc_safety` + `reverse_postorder` reachable set, same as mem2reg |

---

## Disposition of the current working-tree change

The uncommitted `src/cfg/mem2reg.rs` change (gate `>= 2` → `>= 1`) **is Phase 6**.
It is a confirmed 2.3× nbody regression in isolation. **Reverted (2026-05-29);**
the gate is back at `>= 2`. Reintroduce at Phase 6 once Phases 0–5 land.

---

## References

- `docs/SSA_ARCHITECTURE.md` — invariants I1–I10, forbidden patterns, rollout.
- Hack, *Register Allocation for Programs in SSA Form* (2006) — chordal IG,
  decoupled spill/color.
- Braun & Hack, *Register Spilling and Live-Range Splitting for SSA-Form
  Programs* (2009) — the spilling algorithm Phase 3 implements.
- Wimmer & Mössenböck, *Optimized Interval Splitting in a Linear Scan Register
  Allocator* (2005) — live-range splitting at low-pressure points.
- Belady, *A study of replacement algorithms* (1966) — furthest-next-use (MIN).
- Briggs et al. / George & Appel — conservative coalescing (Phase 5).
