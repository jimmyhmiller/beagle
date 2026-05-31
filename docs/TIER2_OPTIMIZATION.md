# Tier-2 SSA optimization workflow

Beagle's tier-2 (the feedback-specialized recompile, optionally through the
SSA pipeline) is where we can be aggressive about codegen quality — the
function is known-hot, the recompile is off the critical path, and we have
type feedback. This document is the standing process for adding **sound,
correct** tier-2 optimizations without regressing.

We do **not** do OSR / deoptimization-to-baseline. Every optimization must
be sound under Beagle's per-op-bail model: never assume a runtime type we
haven't proven or guarded, and never produce a representation a bail path
can't satisfy. "Optimal but sound" — when in doubt, stay correct.

## The hard problem: standard tests never hit tier 2

A function only reaches tier 2 after `specialize-all` or ~1000 calls, and the
auto path is async — so the ordinary suite runs entirely on tier-1 code. A
tier-2 miscompile is invisible to it. Two tools close that gap:

### 1. Tier-2 differential test harness (`BEAGLE_TEST_TIER2=1`)

`test "..." { }` blocks run **twice**: once on tier-1, then — after
`specialize-all` — again on the specialized/SSA code. Both must pass.
Implemented in `src/main.rs` (the test-block loop runs as `run_blocks`,
called a second time when the env is set).

```
# Validate tier-2 SSA + float unboxing against the whole test-block corpus:
BEAGLE_SSA_TIER2=1 BEAGLE_TEST_TIER2=1 cargo run --release -- test resources/
```

A tier-2-only failure prints `test FAIL [tier2]: ...`. **Caveat:** tests that
call `specialize-all` themselves and assert on the count (`assert!(n > 0)`)
false-positive on the second pass (count is 0 — already specialized); such
meta-assertions were removed from the tier-up tests. Don't add new ones —
assert on *behaviour*, not on the specialization count.

### 2. Threshold override (`BEAGLE_SPECIALIZE_THRESHOLD=N`)

Lowers the tier-up counter so long-running functions tier up during a normal
run (`src/ast.rs`). Useful for benchmarks and long loops; less useful for
short unit tests (the async recompile may not finish before they end —
prefer the differential harness for those).

### 3. Snapshot differential re-run (opt-in: `// @tier2-rerun`)

A snapshot test marked `// @tier2-rerun` re-runs `main()` after
`specialize-all` under `BEAGLE_TEST_TIER2=1`, and the second (specialized/SSA)
run's output must match the first. **Opt-in**, because `specialize-all`
recompiles the whole program (slow — re-running *every* snapshot test made
the suite hang/crawl), and a re-run is only sound for an idempotent `main()`
(no file writes, accumulating atoms/state, or un-rejoinable threads). Mark
only pure-compute tests. `resources/tier2_snapshot_test.bg` is the model.

```
# fast: only test blocks + opt-in snapshots re-run on tier-2 (~1 min)
BEAGLE_SSA_TIER2=1 BEAGLE_TEST_TIER2=1 cargo run --release -- test resources/
```

The highest-leverage way to expand tier-2 coverage: add `test "..."` blocks
(differential for free) and `// @tier2-rerun` markers to pure snapshot tests,
plus a parity test per optimization (`resources/ssa_tier2_parity_test.bg`).

## Adding a tier-2 optimization — the checklist

1. **Prove the prize first.** Measure the win on a focused benchmark before
   building (e.g. `bench_phase0_*`, `bench_boxing_probe`). If it isn't there,
   stop.
2. **Establish soundness on paper.** What must be true at runtime? What
   proves it (feedback + guard, or a static invariant)? What does the bail
   path produce, and can the optimized representation hold it? Write it into
   `docs/SSA_ARCHITECTURE.md` if it touches the SSA layer.
3. **Gate it.** New optimizations ride `BEAGLE_SSA_TIER2` (tier-up only) or a
   dedicated env flag, so cold first-compiles and the default path stay
   untouched and the blast radius is bounded.
4. **Implement conservatively.** Bail to the safe lowering whenever an
   assumption isn't provable (see float unboxing's `unbox_safe` gate). A
   missed case must degrade, never miscompile.
5. **Test under the harness.** `BEAGLE_SSA_TIER2=1 BEAGLE_TEST_TIER2=1 test
   resources/` must stay green. Add a `test`-block parity test exercising the
   new shape. Run `/ssa-review`-style scrutiny against the spec.
6. **Measure and commit.** Confirm the win holds and bit-exact outputs; note
   bail rate (`BEAGLE_SSA_LOG_BAIL`).

## State assessment (after loop iters 1-3)

The high-value, tractable tier-2 wins are **landed**: tier-2 SSA regalloc
(~20-30% on hot loops) and float unboxing (~70-75% on float-domain loops).
The benchmarks now go through SSA tier-2 with **zero bails** (nbody 16/16 OK,
series 10/10, btrees 24/24), so the SSA path is already fully applied — there
is no bail-reduction win to chase on them.

What remains is **high-effort / diminishing-return**, and the autonomous loop
deliberately did NOT grind these (the "must not break things" bar):
- **field-CSE** — ~6% estimated on nbody (≈4 of ~30 field accesses redundant:
  `bi.mass`/`bj.mass` read 3× each). Requires deferring field *reads and
  writes* through core field-access codegen — a large build with real
  miscompile risk for a modest win. **Safety net is now in place:**
  `resources/field_access_parity_test.bg` exercises the patterns it must not
  break (repeated immutable reads, interleaved reads/writes = nbody shape,
  mutable re-read after write), re-run on tier-2 by the harness. Build design:
  `Instruction::FieldRead`/`FieldWrite` (FloatBinOp-style deferral) carrying
  resolved name-const-ptr + property-access builtin fn-ptr (so the slow-path
  call can be re-emitted from `Ir::expand`, since `call_builtin` is an
  AST-level method); AST-level CSE keyed on (object_var, field) with
  label-count barrier (reads AND writes deferred so the barrier counts only
  user control flow); immutability via struct_id>>24 → get_struct_by_id →
  is_field_mutable; restrict to non-`mut` object variables (no closure-mutation
  invalidation needed). **Progress:** (1) `lower_field_read` extracted from
  `Ast::PropertyAccess` — the idiom is now a reusable AST-level method, since
  the slow-path `call_builtin` forces expansion to live at the AST level, not
  in `Ir::compile` (commit 625aae2, behaviour-identical).
  (2) **DONE** (commit 352535a, behaviour-neutral): `Instruction::FieldRead`
  deferral + AST-level `expand_field_reads` rebuild pass that re-lowers each
  via `lower_field_read`. Emitted only under `in_tier_up_compile()`. KEY
  GOTCHA fixed: `expand_field_reads` must run in BOTH places an Ir is lowered
  — the end of `AstCompiler::compile` (synthetic top-level body) AND before
  the inner-function `self.ir.compile` in the `Ast::Function` arm
  (ast.rs:~1651). Top-level fns (weighted/step) compile through the
  `Ast::Function` path, so the inner one is the primary site; missing it leaks
  FieldRead to the backend (the `unreachable!` guards fire). 367/367 both ways.
  (3) **CSE measured, not built (iter 7):** probed opportunities first — 0 on
  nbody (structural), ≤1 even on a contrived `b.mass*b.mass` under a sound
  barrier. See "Tried, not worth it". Immutability decode that DID work and is
  worth remembering: struct_id = `(prior.0 >> 24) & 0xFFFFFFFF` (Header
  type_data lives in bits 24-55, see types.rs Header::to_usize); field_index =
  `prior.1 / 8`; then `self.compiler.get_struct_by_id(id).is_field_mutable(fi)`
  is authoritative. Do NOT use `prior.2` for read sites — the getter
  property_access builtin (builtins/objects.rs ~269) writes only 2 of the 3
  IC words, so the mutability word is unpopulated for reads.
- **guarded float speculation** (struct/array-fed loops, e.g. nbody's ~25%
  ceiling) — needs body-versioning (two copies of a region guarded at entry).
  Very large; the biggest remaining win but the riskiest build.
- **non-pointer → unscanned slots** — sound but the GC already tag-checks
  scanned slots, so the win (smaller scanned frame) is modest and hard to
  measure cleanly.
- **regalloc live-range splitting** (spec Phase 7) — general, but a complex,
  high-risk regalloc change.

Recommendation: these need explicit prioritization/oversight. Pick one
deliberately rather than grinding autonomously; each is a multi-session build
held to the hard gates below.

## Optimization pipeline (status)

- **done** — Tier-2 SSA path (`BEAGLE_SSA_TIER2`): ~20-30% on hot loops via
  better regalloc.
- **done** — Float unboxing (sound): definitely-float locals → unscanned FP
  slots, guard-free unboxed `FloatBinOp`s; ~70-75% on float-domain loops.
- **done** — Snapshot differential in the harness (opt-in `// @tier2-rerun`).
- **measured, not worth it** — Immutable-field-load CSE: probed the actual
  opportunity (iter 7) and found 0 on nbody (structural — no repeated
  same-object immutable reads; boxed-float values can't be cached across the
  safepoints that sit between interleaved field writes). See "Tried, not worth
  it" below. The `FieldRead` deferral foundation (352535a) is kept but has no
  consumer yet.
- **later** — Non-pointer values → unscanned slots (smaller GC root set);
  guarded float speculation for struct-fed loops (needs body-versioning).

### Tried, not worth it (don't re-attempt without new evidence)

- **Float-compare unboxing** — loop conditions usually compare a float local
  against a *parameter* (`while k < terms`), and params aren't provably float,
  so the compare can't be unboxed soundly. Narrow applicability.
- **Float-constant unbox CSE** — reusing one unboxed FP value for repeated
  uses of the same literal in a straight-line region. Implemented + measured:
  **~1% on `series` (within noise)**. The constants aren't shared enough and
  the unbox isn't the hot-path bottleneck. Reverted.
- **Immutable-field-load CSE** (iter 7) — MEASURED, not worth it; CSE not
  built. The `FieldRead` deferral foundation (commit 352535a) is kept (sound,
  behaviour-neutral, could enable other field opts), but a `BEAGLE_PROBE_
  FIELDCSE` instrumentation counted the actual CSE *opportunities* before
  building the CSE (checklist item 1) and found **ZERO on nbody**:
  `bench_phase4_nbody/advance` has 6 immutable field reads but 0 same-field
  repeats — even under the loosest sound barrier (clear only on
  control-flow/calls). It is STRUCTURAL: nbody reads each body's `mass` once
  per interaction; `bi.mass * bj.mass` are *different* objects, not a repeat.
  Even a contrived `b.mass * b.mass + b.mass` yields only 0–1 candidates under
  a sound barrier (the `*` is a deferred FloatBinOp = potential safepoint
  between the reads). Deeper reason it can't pay off on real float code: a
  cached immutable-field value that is a boxed-float *pointer* cannot survive a
  GC safepoint in a register (I9), so the cache must clear at every
  call/write-barrier — exactly where interleaved field writes sit. The win
  only exists for non-pointer immutable fields repeatedly read in a
  safepoint-free straight-line region, which the benchmarks don't contain.
  Do not re-attempt without a benchmark that actually has repeated same-object
  immutable-field reads between safepoints.
