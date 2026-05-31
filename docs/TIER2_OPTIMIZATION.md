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
  miscompile risk for a modest win. Needs a deliberate, human-overseen effort
  + a thorough field-CSE parity test, not an autonomous quick pass.
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
- **next** — Immutable-field-load CSE: non-`mut` fields are globally stable,
  so reads are CSE-able. Sound design: AST-level cache keyed on
  `(object_var, field)` with **written-label-count staleness** (reuse only if
  no label/call/reassignment occurred between — handles branch dominance by
  construction). Needs the harness to validate.
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
