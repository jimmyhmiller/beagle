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
- **Float-parameter versioning** — **LANDED, default-on** (opt out with
  `BEAGLE_SSA_FLOAT_PARAM_VERSION=0`). Guards float params at entry, unboxes the
  fast body, re-invokes the resident generic on a guard miss. **mandelbrot ~2×
  (2.16→1.08s @ n=2000), bit-identical**; all other benchmarksgame ports
  bit-identical; suite 369/369 (plain + on + tier-2 harness). See Priority 1.
- **`Instruction::MoveFloat`** — a shared SSA-backend fix landed alongside: the
  edge resolver emitted an integer `mov` for FP block-param transfers (silently
  corrupting loop-carried floats with non-coalesced FP phis). Now emits
  `fmov`/`movsd`. Benefits **all** SSA float code, not just versioning.
- **De-specialize on redefinition** — landed (`Compiler::revert_all_specializations`,
  hooked into `compile_string*` + `add_struct`). When `eval`/REPL/reload redefines
  a struct/let/function, all tier-2 specializations revert to their retained tier-1
  code (jump-table swap-back) so no specialized code keeps a stale snapshot of the
  redefined world. Fixes the silent-wrong-result staleness at the default threshold.
- **Inlined array primitives** — `beagle.mutable-array/swap` and `/write-field`
  are now inline primitives (`src/primitives.rs`), like `get` already was. Each
  array op in a tight loop was a full wrapper-function call; inlining the
  type-guard + bounds + store(+barrier) and delegating non-array/OOB to the real
  wrapper on the slow path removes that overhead. **fannkuch_redux ~1.9× (4.9s →
  2.63s @ n=10)**, bit-identical; suite 369/369 (plain + tier-2 + generational GC).
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

## Priority 1 — Float-parameter versioning — ✅ LANDED, default-on

**Status (2026-06-02):** shipped, default-on (opt out `BEAGLE_SSA_FLOAT_PARAM_VERSION=0`).
**mandelbrot ~2× (2.16→1.08s @ n=2000), bit-identical.** All other benchmarksgame
ports bit-identical; suite 369/369 (plain + on + tier-2 harness). The original
~15%/~38% estimates were the *kernel* ceiling; the full benchmark (with IO and
draw_rows) realizes ~2× because `draw_rows` (the outer float loop) versions too.

**Design (Path B, no body duplication).** `mandel`/`draw_rows` are versioned:
guard each float param at entry; pass → unboxed fast body; miss → re-invoke the
resident generic (`generic_addr` from `deopt_info_for`) with the original args.
Sound because the guard is the first thing that runs (re-invoke from entry is
unobservable) and `float_types` is computed assuming float for *exactly* the
guarded params (an unguarded param never enters the unbox set).

**Implementation:**
- `src/float_repr.rs::plan_float_param_version` — handles `TailRecurse` and
  `while`/`loop` (flat back-edge) shapes. Three gates: (1) **necessity-minimised
  guards** (drop a guard whose float assumption doesn't enlarge the *derived*
  float set — excludes int params like a loop bound `n`); (2) **strict win** (more
  float locals than the sound baseline); (3) **arithmetic-benefit**
  (`float_types_feed_arithmetic`) — some guard-*unlocked* register (in
  `float_types.regs` but not `baseline.regs`, so not a mere float const) must feed
  a `FloatBinOp`. Gate (3) is what excludes `draw_row`, whose guarded `ci` is only
  copied and handed to `mandel(cr,ci)` (no arithmetic) — versioning it pushed an
  unboxed FP value across the call and corrupted it.
- `src/ir.rs::apply_float_param_versioning` — prepends `GuardFloat` per guard
  param (before the arg→slot prologue), appends the re-invoke slow block, sets
  `float_locals`/`float_regs` so `unbox_floats` unboxes the fast body.

**Two bugs found + fixed while landing (both in the shared SSA backend):**
1. **`Instruction::MoveFloat`** — the edge resolver lowered FP block-param
   transfers (`Op::Move` of FP class) to an integer `mov` of the wrong (GP)
   registers, silently corrupting any loop-carried float with a non-coalesced FP
   phi (minimal repro `/tmp/n6.bg`: `nzi = zr*zi + ci` carried). Added an FP
   register move (`fmov Dd,Dn` / `movsd`) wired through the backend trait. **This
   fixes all SSA float code**, not just versioning.
2. **draw_row over-version** — fixed by the arithmetic-benefit gate (above).

**Reusable diagnostics learned:** capture tier-2 asm with `--dump asm` and pick
the record by size/structure (tier-1 polymorphic is larger + has `scvtf`; tier-2
SSA is the lean one). `BEAGLE_SSA_DUMP_FN`/`BEAGLE_SSA_VERIFY_STAGES` dump/verify
per SSA phase. `runtime/specialize-all()` + `--no-auto-specialize` gives
deterministic tier-up for A/B (async trampoline tier-up is racy).

**Remaining (future):** `draw_row`-shape loops (unboxed float passed across a
fast-path call) stay boxed until the regalloc **preserves FP across calls** (the
spiller handles only GP today — `cfg::regalloc::spill`). That's the next unlock
for call-bearing float loops.

---

## Priority 1.5 — Hot-loop tier-up (OSR) — the biggest remaining systemic lever

**Finding (2026-06-02, measured).** Tier-up is triggered by the **entry
counter**. A function entered once but running a hot inner loop (fannkuch,
spectral_norm, any `main`-with-a-big-loop) **never tiers up** — it runs the whole
computation as unoptimized tier-1 (polymorphic ops, no int/float specialization,
no SSA regalloc). Confirmed: lowering the threshold to 1 doesn't help fannkuch
(specialization fires but applies to the *next* call, which never comes);
`runtime/specialize-all()` + a second call shows tier-2 would give **~32%**
(4.95s → ~3.35s on fannkuch n=10).

**Fix = OSR (on-stack replacement):** count loop back-edges in tier-1 code; when a
loop is hot, compile the tier-2 version and transfer the running frame into it at
the loop header. **Tractable here because tier-1 and tier-2 share a byte-identical
slot layout** (same AST, feedback-independent alloc — see the deopt notes): all
live loop state is already in stack slots, so OSR is essentially "jump from the
tier-1 loop header to the tier-2 loop header" with no register-state transfer (the
GP values are in slots; FP would need care). Hard parts: back-edge instrumentation
in tier-1, mapping tier-1→tier-2 loop-header addresses, the frame hand-off.

**Tier-2 activation races execution at low thresholds (BLOCKER for lowering the
threshold).** Tier-up runs on the compiler thread; the install (`flush_deferred_
functions`: jump-table swap + function-table mutation) races with the mutator
reading the function table (closure dispatch's ptr→index scan, etc.). At the
default 1000 it's rare (functions tier up when warm); at low thresholds a function
tiers up *during* its own first execution → crashes (uncaught exceptions via the
async wrapper — `map_utils_advanced` at thr=10, `struct_redefine_*`/`redefine_let`
at thr=1). **Tried TWICE & reverted both times: defer activation to a
stop-the-world** (compiler stages a ready install; a mutator applies it at STW).
The 2nd attempt fixed the thrash (relaxed-load gate before the atomic swap) so
it's correct and no longer thrashes — but it is a **confirmed dead end** for two
reasons measured on the real `beag` binary:
1. **It never activates on alloc-fast-path-heavy hot loops.** mandelbrot boxes
   its floats via the *inline* bump allocator in generated code, which never
   enters Rust `allocate()` where the `installs_ready` STW trigger lives. So
   tier-2 never installs and mandelbrot ran 2.89s (tier-1) instead of 1.04s — a
   ~37% regression. (`BEAGLE_DBG_INSTALL` trace: 0 activations.)
2. **Zero correctness benefit.** Full suite at `BEAGLE_SPECIALIZE_THRESHOLD=1`
   is 363/369 with STW-staging == 363/369 with HEAD's immediate-flush. The 6
   remaining thr=1 failures are pre-existing de-spec-at-thr1 bugs, not the
   install race.
Conclusion: keep immediate-flush. The right fix is **fine-grained
synchronization of the function-table reads** (lock/seqlock around the ptr→index
scan + the activation), NOT deferral — a separate careful project. Until then,
keep the threshold at the default.

**Current Beagle vs Node v24 (warmed, median of 3):** fannkuch_redux 21× (now
~12× after array inlining), nbody 18×, spectral_norm 8×, binary_trees 2.6×,
mandelbrot 1.7× (was ~3.5× pre-versioning). The big multipliers (fannkuch,
spectral) are dominated by **never tiering up** + array/V8-codegen gaps. OSR is
the lever that unlocks tier-2 for them.

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
