# Performance: Next Work

A prioritized, evidence-backed map of the remaining perf wins, where they are,
and where they provably aren't. Written after a session that landed load-CSE,
AST loop-invariant hoisting, tier-2 dead-feedback removal, the `x*x` backend
fix, Phase A non-pointer analysis, and deoptimization (on by default).

Companion docs: `docs/SSA_ARCHITECTURE.md` (invariants I1–I10, type-aware
GC-safety section, deopt feasibility + landing). Memory has the full iteration
history (`project_type_aware_gc_safety`, `project_float_unboxing`,
`project_tier2_dead_feedback`).

---

## Ground rules for any perf work here

1. **Probe the prize first.** Every win this session was confirmed by an A/B
   microbenchmark *before* building (and several no-gos were killed by a 10-min
   probe). Build a `resources/bench_*_probe.bg`, measure interleaved, only then
   commit to the build.
2. **Gate, validate, commit incrementally.** New optimizations go behind an env
   flag, default-off until validated, with the full suite green (`cargo run --
   test resources/`) and — for anything GC- or float-touching — the gc-stress
   tests and **bit-identical** float output at every step.
3. **GC soundness is absolute.** Beagle has a compacting GC and **no deopt of
   the OSR kind** (I9). A value live across a safepoint that *might* be a
   relocatable pointer must be slot-rooted. A single misclassification =
   use-after-free.
4. **No whole-program assumptions.** Code can redefine/load at runtime; any "I've
   seen all callers/constructors" inference is unsound.

---

## Current state (what's shipped / landed)

- **Float *locals* unboxing** — shipped (SSA tier-2 default-on): series 3.46×,
  fib 20%, btrees 23%. Local/intermediate floats stay in FP registers.
- **Deopt (reinvoke-for-pure-functions)** — `BEAGLE_SSA_DEOPT`, **on by
  default**. Removes the bail-rejoin for eligible pure functions so loop-carried
  ints promote to registers. ~25–35% on pure-no-call int loops. Validated
  (369/369 plain + tier-2 + gc-stress). **Narrow** — see below.
- **Phase A non-pointer analysis** — `src/cfg/pointer_class.rs`, sound, tested,
  committed. The type oracle. Currently consumed by nothing (Phase C reverted as
  inert pre-deopt).

---

## The honest landscape

| Lever | Verdict |
|---|---|
| Float locals unboxing | ✅ shipped |
| Deopt for pure int loops | ✅ shipped (narrow) |
| **Float-parameter versioning (mandelbrot)** | 🟢 **tractable, sound, ~15%, planner built** — the clear next prize |
| Region/loop versioning (impure loops) | 🟡 broadens deopt but reach still limited (pure loop *body* required); needs forward-resume for call-loops |
| Call-in-loop int accumulators | 🔴 needs forward-resume OSR deopt + Phase C — large |
| Field-fed floats (nbody) | 🔴 architecturally unreachable soundly without NaN-boxing or a value-rep rewrite |

**Why deopt is narrow:** it only removes *bail* safepoints. Real loops have
other exclusions — array indexing `arr[i]` is a call (`beagle.core/get`), field
writes are heap stores, floats are boxed. So deopt as built helps only synthetic
pure scalar-int loops. Broadening it is either marginal (region versioning) or
huge (forward-resume OSR).

---

## Priority 1 — Float-parameter versioning (the next prize)

**Why:** the one clearly-tractable, sound, *measured* remaining win. mandelbrot
~15% (probe in git history; `BEAGLE_FLOAT_ARG_PROBE` was the unsound proof). Hits
TCO'd local-float loops (mandelbrot-shape) that box today only because their
float values originate in *arguments*. **Sound** — the entry guard is read once
before the loop, so the nbody read-after-write hoist hazard does NOT apply.

**What exists:** `src/float_repr.rs::plan_float_param_version` (step 1, landed) —
returns a `FloatParamVersionPlan { guard_params, float_types }` when guarding
float params strictly enlarges the provably-float local set and the function
loops via `TailRecurse`. The float dataflow (`analyze_core` + `AnalyzeOpts`)
models the optimistic case (`assume_args_float`, `model_tailrecurse`). The
unboxing engine (`unbox_floats`, `LoadLocalFloat`/`StoreLocalFloat`, unscanned FP
slots) is shipped.

**What to build (step 2 — the versioned codegen):**
1. When a function has a `FloatParamVersionPlan`, emit at entry (after the
   arg→slot prologue): `GuardFloat` each `guard_params` slot. All pass → jump to
   the **fast** body; any fails → jump to the **slow** body.
2. **Fast body** = the function body with `plan.float_types` driving
   `unbox_floats` (float params unboxed, carried unboxed across the `TailRecurse`
   back-edge in FP slots).
3. **Slow body** = today's boxed lowering (unchanged).
4. Both reach the same return.

**Risks / where it bites:** this is IR-level body duplication (per iter-10's
finding: AST-emission versioning double-consumes feedback slots — do it at the
IR level in/after `unbox_floats`). Must handle label relabeling, the
`TailRecurse` back-edge in both versions, and the feedback-slot cursor. **Float
output must be bit-identical**; gc-stress must pass. Gate
`BEAGLE_SSA_FLOAT_PARAM_VERSION`, A/B on a mandelbrot probe.

**Caution (from memory):** versioning is repeatedly flagged as "the very large,
riskiest build that needs oversight." Build in the smallest gated, bit-identical
slices.

---

## Priority 2 — Region/loop versioning for deopt (broaden the int win)

**Why:** broadens deopt from "whole function pure" to "pure loop *body* in any
function" (e.g. read-input → compute-loop → print). Real but bounded reach —
still requires a call-free loop body.

**Design:** clone the hot loop into a fast (typed) + slow (generic, today's
rejoin) copy. Fast guard misses → deopt edge → slow header, **passing the
loop-carried phis as SSA block-params** (not slots — this avoids the mem2reg
slot-writeback coordination problem). Slow loop runs to the shared exit.
Deopt-to-slow-*header* re-runs the current iteration generically, sound when the
iteration body is pure-up-to-bail (same constraint as today). Reuses
`cfg/loops.rs::natural_loops`.

**Does NOT unlock** call-in-loop (the call is still a safepoint the loop-carried
values cross). For that, see Priority 4.

---

## Priority 3 — Re-enable Phase C on top of deopt

**Why:** deopt removes the bail-rejoin that made loop-carried values
`maybe-pointer` and made Phase C inert (the reason it was reverted). On a
deopt-rewritten loop, the accumulators are *provably non-pointer* (Phase A sees
them), so Phase C can keep them in registers **across remaining real-call
safepoints without rooting** — the piece a loop-with-a-call needs.

**What exists:** Phase A (`pointer_class.rs`). Phase C (the mem2reg gate
relaxation `non_pointer_slots`) is written and one revert away (see the reverted
commit / `SSA_ARCHITECTURE.md` Phase C note).

**Dependency:** only useful *after* Priority 2/4 makes call-loops deopt-eligible
(so their rejoin is gone too). Standalone today it re-confirms inert.

---

## Priority 4 — Forward-resume OSR deopt (the general win, large)

**Why:** the only thing that makes **call-in-loop** accumulators (`map`/`filter`/
`reduce`, anything calling a closure) promotable. Resume the generic version
*forward* from the exact miss point (no re-execution → no purity constraint).

**What's already easy here (discovered + measured this session):**
- **Deopt state map is the identity** — tier-1/tier-2 share a byte-identical slot
  layout (same AST, feedback-independent alloc; push_to_stack slots every value).
  A deopt only memcpies the slot region + writes promoted regs back.
- The generic code stays resident after the specialization jump-table swap.
- Ceiling measured at ~38% (`BEAGLE_SSA_DEOPT_CEILING`, reverted).

**The hard part:** mid-expression resume points (operand-stack reconstruction) —
the thing an interpreter gives for free and Beagle lacks. Either build
interpreter-grade per-position resume entries in the generic code, or restrict to
coarse boundaries (collapses toward versioning). Plus the frame transfer (resume
stubs / OSR). Big, careful, GC-critical.

---

## Do NOT re-investigate (closed, with evidence)

- **Field-fed float unboxing** (nbody, spectral) — ruled out twice: loop-body
  versioning hits genuine read-after-write field deps (commit 585718b); unboxed
  field *storage* needs deopt or NaN-boxing (commit dc32e31). Don't revisit
  without a deopt mechanism or NaN-boxing (both major architecture projects).
- **Field-CSE** — a probe showed 0 opportunities (iter 7).
- **load-CSE across safepoints** — measured *slower* (regalloc spills the
  extended live range); the conservative safepoint-clear is near-optimal.
- **Cold tier-2 non-pointer promotion (Phase C standalone)** — ~401 cold slots,
  zero hot win, GC risk. Inert until deopt removes the rejoin.

---

## Benchmark reality (warmed + tiered, benchmarksgame ports)

- `fannkuch_redux` ~16% from tier-2 (int regalloc)
- `mandelbrot` ~1.5% today / **~15% ceiling via Priority 1**
- `nbody`, `spectral_norm` ~0% (heap-fed floats — closed)
- `binary-trees` ~0% (alloc/GC bound — different domain; GC tuning, not covered here)

Reusable tested tooling: `cfg/loops.rs::natural_loops`,
`ir_loops.rs::{flat_ir_loops, versionable_float_loop}`,
`float_repr.rs::{analyze_core, plan_float_param_version, speculative_chain_stats}`,
`cfg/pointer_class.rs::analyze`.
