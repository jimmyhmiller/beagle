# OSR performance handoff — closing the F_osr-vs-warm gap

## RESOLVED (2026-06-04) — F_osr now ≈ warm for int loops

The gap is **closed for integer loops**. §4's unresolved question is answered
and the fix landed (no Phase-B/C GC-gate change was needed):

**§4 answer.** Warm and F_osr do *not* both keep the loop's bail safepoints.
Warm F is compiled with a `DeoptContextGuard` active, so `apply_deopt_rewrite`
redirects its type/overflow-guard bails to **terminal deopt edges** (re-invoke
generic F, return its result) — making the hot loop **safepoint-free**, which is
why mem2reg promotes its loop-carried slots. F_osr was compiled with **no** deopt
context, so its guard bails stayed as **polymorphic merge-backs** whose `Call`
results (and bail `Call` safepoints) sit on the main loop path → mem2reg's I9
gate rejects promotion *and* the merge-back stores a maybe-pointer into the
accumulator slot. The OSR-entry `HeapLoad` is a red herring; the merge-back is
the real blocker.

**The fix (landed).** Compile F_osr **with** a deopt context too:
- `Compiler::build_osr_variant_inner` wraps F_osr's compile in
  `DeoptContextGuard::enter_with_nargs(F_osr_name, generic=F's addr, pause,
  Some(F_arity))`. The new `reinvoke_nargs` override is needed because F_osr's
  own entry param is the live-in *buffer* (1 arg), but its deopt must re-invoke
  generic F with F's real args — which sit in slots `0..F_arity`, repopulated
  from the buffer by the OSR entry. Sound because F's params are live-ins and
  aren't reassigned in the loop (the deopt eligibility check enforces this).
- The OSR entry `GuardInt`s each **non-param int live-in** (classified via
  `pointer_class` on a deopt-applied analysis CFG of F) and stores the
  *proven-int* dst, so the loop-header φ is `known_int` and
  `eliminate_redundant_guards` drops the loop's redundant guards (≈18→1, matching
  warm). Entry guards are only added **when deopt actually applied** (gating on
  `build_cfg_for_int_analysis`'s `deopt_applied` flag) — adding them otherwise is
  pure overhead (regressed fannkuch 1.5→3.0s before the gate).
- A trampoline re-entrancy guard (`osr_transfer_begin/end`) stops the
  deopt→generic-F→re-OSR path from recursing without bound if a guard keeps
  failing.

**Measured (apples-to-apples).** multi 1.71→**0.26s** (warm ceiling 0.24),
sumloop 0.58→**0.21s**, fannkuch n=10 2.68→**1.56s** (deopt-ineligible — array
writes — so it gets the pre-existing non-deopt OSR, near warm 1.40). Suite
370/370 default *and* OSR-aggressive *and* generational; all benchmarksgame
bit-identical OSR on/off. Opt out with `BEAGLE_OSR_DEOPT=0`.

**Still open (float loops).** spectral/mandelbrot are flat — their live-ins are
boxed floats, which the int path doesn't guard/unbox. That's the documented
Phase-D follow-on (`GuardFloat` + `coerce_to_fp` at the entry, like
`apply_float_param_versioning`).

Everything below is the original investigation that led here, kept for context.

---

**Status:** OSR is implemented, correct, and landed behind `BEAGLE_OSR` (see
`docs/OSR_DESIGN.md`, `project_osr_landed` memory). It delivers real wins on
self-contained scalar loops but runs **~2-3× slower than a clean warm tier-2
compile of the same loop**. This doc is the map to closing that gap.

The whole investigation is real and reproduced; numbers and file:line refs below
are from working experiments, not theory. Read §4 (the one unresolved question)
before writing code — it decides whether the fix is small or the full §5 project.

---

## 1. The gap (measured, apples-to-apples wall clock)

| workload | tier-1 | **OSR** | warm tier-2 (ceiling) |
|---|---|---|---|
| `multi` (single loop, 10 int live-ins) | 1.58s | **0.74s** | **0.24s** |
| `sumloop` (single loop, 3 int live-ins) | 0.57s | 0.34s | ~0.20s |
| fannkuch_redux n=10 | 2.68s | 2.22s | 1.40s |

Repro: `/tmp/osr_multi.bg`, `/tmp/osr_test.bg` (sum loop), and
`benchmarks/benchmarksgame/fannkuch_redux.bg N warm` for the warm ceiling.
`BEAGLE_OSR=1 BEAGLE_OSR_THRESHOLD=10000 ./target/release/beag <file>`.

Even a **single, pure-int loop** is 3× off warm — so the gap is NOT about loop
selection, nesting, feedback, or the transfer mechanism. It's about the quality of
the code `F_osr` runs.

---

## 2. Root cause (established)

`F_osr`'s loop-carried live-ins are **not register-promoted** — they're written
back to their GC root slot **every iteration** — while warm tier-2 keeps them in
registers. Direct evidence from `mem2reg`'s own log
(`BEAGLE_SSA_LOG_MEM2REG=1`):

```
[mem2reg] multi_warm/multi        candidates=44 promoted=43 rejected_i9=[SlotId(0)]
[mem2reg] osr_multi/multi$osr0    candidates=72 promoted=62 rejected_i9=[Slot 0..9]   ← all live-ins
```

and the post-opt CFG dumps (`BEAGLE_SSA_DUMP_FN`):

```
warm  multi: SlotStore=1   guard-int=1    (loop-carried ints in registers)
F_osr multi: SlotStore=19  guard-int=47   (written back every iteration)
```

Two coupled symptoms, one cause:

- **No promotion.** `mem2reg`'s promotion gate is
  `cfg::gc_safety::slot_is_gc_safe_to_promote` (`src/cfg/mem2reg.rs:110-120`) — a
  **store→safepoint→load dataflow check, NOT pointer-class**. A slot is rejected
  iff its value is live across a GC safepoint (Beagle scans frame slots, not
  registers, so a maybe-pointer in a register across a safepoint can go stale under
  the compacting GC). The safepoints in these loops are the **bail `Call`s of the
  loop's int guards** (`beagle.bail/*`).
- **No guard elimination.** Warm has 1 guard-int (down from ~47) because
  `opt::eliminate_redundant_guards` (`src/cfg/opt.rs:488`) removed the dominated
  ones — the loop-carried values are *proven int* (defined by int ops + an int
  const init). In `F_osr` the loop-header φ merges the **untyped `HeapLoad` live-in**
  with the loop's int value → not proven int → guards stay → bail `Call`s stay →
  the slot crosses safepoints → I9 rejects promotion.

The common cause: **`F_osr`'s live-ins enter via an untyped `HeapLoad`** (the OSR
buffer unpack — `src/osr.rs::build_osr_variant_ir`). That single untyped def at the
top poisons both the type lattice (guard-elim) and the promotion gate (I9).

---

## 3. Why the obvious entry-side fix does NOT work (don't repeat)

Tried (2026-06-04) and reverted: `GuardInt` each live-in at the OSR entry to
"re-establish int-ness." Outcome: **correct + SSA, but ~no speedup** (0.74→0.65s).

It can't work, because **the I9 gate is safepoint-based, not type-based** —
establishing a value's int-ness does nothing to a dataflow that only asks "does a
store→...→safepoint→...→load chain exist for this slot." And guard-elim (which
*would* benefit from the entry type) runs **after** mem2reg, so it can't unblock
promotion either. The entry guard is necessary but not sufficient; the gate itself
must change. Also note while doing this:

- `guard_integer` (`src/arm.rs:1222`) does **not** untag — `_dest` is unused, the
  value passes through tagged. (An early "wrong result" was from storing the
  *unwritten dest*, not untagging.)
- `AddInt(v, 0)` to mark int-ness is **folded away** by opt → no effect.
- `pointer_class::analyze` (`src/cfg/pointer_class.rs`) returns an **empty**
  `non_pointer_slots` for these loops, because `op_produces_non_pointer`
  (`:250`) recognizes `AddInt` but **not** the `Tag` / checked-int-op outputs that
  the specialized `a = a + i` actually produces. So it can't currently drive a fix.

Also previously ruled out (see `project_osr_landed`): dead-pre-loop truncation
(unsound for nested loops — enclosing headers are reachable via back-edges), and
OSR-ing an outer loop from an inner trip (no consistent outer live-ins mid-inner).

---

## 4. THE UNRESOLVED QUESTION — answer this FIRST

**Warm and F_osr have the same ~47 guards + bail Calls at mem2reg time** (guard-elim
runs after mem2reg for both). Yet warm promotes the loop-carried slots and F_osr
does not. The *only* structural difference is F_osr's extra OSR-entry `SlotStore`
of each live-in (block 0). **Why does that flip the I9 verdict?**

Until this is answered, you don't know if the fix is small or large:

- **If** the OSR-entry store merely introduces an avoidable safepoint-crossing
  (e.g. it lengthens a live range so some slot is now live across a bail Call that
  warm's pre-loop store wasn't) → the fix may be **local** (restructure the entry,
  or pre-seed the slot so the dataflow matches warm).
- **If** warm genuinely promotes *across* the bail-Call safepoints by some property
  F_osr lacks → the fix is the full **§5 Phase-C** project.

**How to answer it:** instrument `slot_is_gc_safe_to_promote`
(`src/cfg/gc_safety.rs`) to log, for one slot (e.g. multi's `SlotId(1)`), the block
+ op where it transitions to `StorePending(true)` and the load that trips
`return false`. Run for both `multi_warm/multi` and `osr_multi/multi$osr0` and
diff. That pinpoints the exact safepoint and path that differs.

---

## 5. The fix plan (Phase-C-shaped — the likely path)

The general, sound unlock: **let provably-non-pointer slots promote across
safepoints.** A non-pointer in a register across a safepoint is safe — the GC only
cares about pointers. This was prototyped before as "Phase C" and reverted as inert
*pre-deopt*; for OSR it is not inert. Three composing steps, each independently
validatable:

### Step A — make `pointer_class` classify specialized-int values
`src/cfg/pointer_class.rs`. `op_produces_non_pointer` (`:250`) and
`inline_branch_produces_non_pointer` (`:286`) must additionally recognize, as
non-pointer:
- `Op::Tag { .. }` **only when the tag is the Int tag (0b000)** — re-tagging a raw
  int yields a non-pointer; tagging with a heap tag does not, so this MUST inspect
  the tag value or it's unsound (a false "non-pointer" = GC use-after-free).
- the checked-int op outputs (`SubChecked`/`MulChecked`/`AddChecked`/… — whatever
  the specialized `*_with_bail` lowers to) and the `GuardInt` fall-through dst
  (already listed at `:29` in the doc-comment but verify it's wired).

Validate: a unit test asserting `non_pointer_slots` now contains the accumulator
slot of a specialized `a = a + i` loop (mirror the existing tests at
`pointer_class.rs:359`).

### Step B — I9 gate consults the non-pointer analysis (the actual Phase C)
`src/cfg/mem2reg.rs:110-120`. Change the gate to:
```
let safe = !read_in_unreachable && !deopt_pinned
    && (class == Fp || non_pointer_slots.contains(s)   // ← new: promote across safepoints
        || slot_is_gc_safe_to_promote(f, *s));
```
Compute `non_pointer_slots` once via `pointer_class::analyze(f)` at the top of
`promote_slots`. **Soundness is everything** (`docs/SSA_ARCHITECTURE.md` §I9):
`pointer_class` is a sound *must*-non-pointer analysis, so a slot it proves
non-pointer is safe to keep in a register across a safepoint. Gate it behind an env
flag (`BEAGLE_SSA_PHASE_C=1`) and default-off until validated, like every prior GC
change. Validate: full suite + `gc-always` stress + bit-identical floats, with the
flag on, on a normal tier-2 run first (not just OSR).

### Step C — the OSR entry must give the live-in a non-pointer def
Even with A+B, `F_osr`'s slot has an extra writer — the untyped `HeapLoad`. For the
slot to be *proven* non-pointer, **every** writer must be non-pointer. So the OSR
entry must turn each int live-in's `HeapLoad` into a proven-int value before storing
it: `GuardInt` it (its fall-through dst is non-pointer per Step A) and store that.
This is the entry-guard work already prototyped in
`src/osr.rs::build_osr_variant_ir` (and reverted) — re-apply it, but now it *works*
because B makes promotion follow. Keep the canonical-arg-0 buffer base (already in
tree) and a sound slow-fallback for guard misses. Only guard slots that are int in
F (from `pointer_class::analyze(build_cfg(F))` — needs Step A to be non-empty); leave
pointer/float live-ins as plain loads.

With A+B+C: F_osr's int live-ins are proven non-pointer (entry guard + loop int ops),
I9 promotes them across the loop's bail-Call safepoints → register-resident, and
guard-elim (post-mem2reg, seeing the now-proven-int φ) drops the redundant loop
guards → F_osr ≈ warm.

### Expected payoff
multi 0.74→~0.24s, sumloop ~0.34→~0.20s, and the scalar parts of the benchmarksgame
ports. Float live-ins (FP class) are already promotable (`class == Fp`) but need the
unbox-at-entry handling (`coerce_to_fp` + `GuardFloat`, like
`apply_float_param_versioning`) — a follow-on after the int path lands.

---

## 6. Validation methodology (non-negotiable)
- **Correctness:** `cargo run -- test resources/` = 370/370, default AND
  `BEAGLE_OSR=1 BEAGLE_OSR_THRESHOLD=300 BEAGLE_OSR_RECHECK=30` (exercises transfers
  across all loop shapes). All `benchmarks/benchmarksgame/*` **bit-identical**
  (`md5`) OSR-on vs off.
- **GC soundness (Step B is GC-critical):** the suite under each GC feature
  (`--features generational`, `mark-and-sweep`, `compacting`) and `gc-always`
  tests. A misclassified non-pointer = silent heap corruption; only stress catches
  it.
- **Perf:** A/B wall-clock `multi`/`sumloop`/fannkuch OSR vs warm; confirm
  `SlotStore` count in the F_osr loop drops to ~warm's via `BEAGLE_SSA_DUMP_FN` and
  `[mem2reg] … rejected_i9=[]`.

---

## 7. Key code locations
- OSR entry construction: `src/osr.rs::build_osr_variant_ir` (Step C lives here).
- Lazy build / int-slot classification: `src/compiler.rs::build_osr_variant`
  (`build_cfg(&base_ir)` + `pointer_class::analyze` → int slots for Step C).
- Promotion gate (Step B): `src/cfg/mem2reg.rs:110-120`; safepoint dataflow
  `src/cfg/gc_safety.rs::slot_is_gc_safe_to_promote`.
- Non-pointer analysis (Step A): `src/cfg/pointer_class.rs`
  (`op_produces_non_pointer:250`, `inline_branch_produces_non_pointer:286`,
  `non_pointer_slots:223`).
- Guard elimination: `src/cfg/opt.rs::eliminate_redundant_guards:488`,
  `compute_known_types:617`.
- Pass order: `src/cfg/builder.rs:295-376` (cfg-ize → split-critical → lift →
  mem2reg → opt). Note mem2reg precedes guard-elim — central to §2.
- `guard_integer` (does not untag): `src/arm.rs:1222`. SSA GuardInt emit routes
  through it via `src/cfg/emit_legacy.rs:909`.
- Debug knobs: `BEAGLE_SSA_LOG_MEM2REG`, `BEAGLE_SSA_LOG_BAIL`, `BEAGLE_SSA_DUMP_FN`
  + `BEAGLE_SSA_DUMP_DIR`, `BEAGLE_SSA_VERIFY_STAGES`, `BEAGLE_OSR_DEBUG`,
  `BEAGLE_OSR_THRESHOLD`, `BEAGLE_OSR_RECHECK`.

---

## 8. One-line summary
F_osr's live-ins enter untyped (`HeapLoad`), which blocks both register promotion
(mem2reg's I9 safepoint gate) and guard elimination. The fix is to teach the
non-pointer analysis to classify specialized ints (A), let mem2reg promote proven
non-pointers across safepoints (B, the real Phase-C unlock), and give the OSR entry
a proven-int def per int live-in (C). Answer §4 first to confirm B is required.
