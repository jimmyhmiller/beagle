# OSR default-on — handoff (the "super performant" perf push)

Goal: make single-function HOT LOOPS (fannkuch/spectral/main loops) tier up +
optimize instead of running unoptimized tier-1 — the path to beating Node
(currently ~12× fannkuch / 8× spectral SLOWER than Node v24 because the
entry-counter never tiers up a once-entered hot loop). OSR (on-stack replacement,
loop-back-edge tier-up) is the lever.

Branch: `perf-osr-default-on` (off the signed-off `s21-nondet-assertions` tip).

## TL;DR status

OSR is already implemented + correct + behind `BEAGLE_OSR`. Probe-first + the
measured-win bar reshaped the work:

- The OSR codegen WORKS: full suite **437/437** with OSR on (aggressive
  `BEAGLE_OSR=1 THRESHOLD=300 RECHECK=30`), all benchmarksgame **bit-identical**
  OSR on/off.
- **fannkuch n=11: 40.98s → 23.56s = ~1.74×** (n=10 = 1.50×) — the headline
  int-loop win, the gap to Node closing.
- The keystone "function-table sync" (originally A) was found **STALE** — the
  perf-doc's install-race crash repros don't reproduce (435/437 @
  `BEAGLE_SPECIALIZE_THRESHOLD=1`; the 2 failures are pre-existing
  `reflect_write_let*` disk-write races, NOT install-race). Mitigated since the
  doc by the L1/L2 dispatch redesign + stable-function-values + de-spec.

**UPDATE: both prerequisites are now landed.** Prereq 1 (A-lite) was signed off;
its 3 review nits are folded in. Prereq 2 (the benefit gate) is DONE and validated
(nbody regression eliminated → 1.00–1.02×; fannkuch 1.49×/1.72× preserved; all
bit-identical) — see "Prereq 2" below for the shipped gate and the two corrections
to this doc's predictions. Remaining before the flip: the full ×3-GC suite +
container stress re-validation, then the independent review of the gate + the flip.

So the flip is NOT a flag flip. Two prerequisites, built in PARALLEL, then flip:

### Prereq 1 — A-LITE (memory-safety): chunked `self.functions`  [IN PROGRESS]

Static trace of the CURRENT function-table read/write (empirical 0/5 stress
"looks closed", but static is the decider):
- **Mode 2 (in-place torn pointer): CLOSED — by STW exclusion.** Tier-2 installs
  (`overwrite_function`'s raw `f.pointer`/`f.size` writes) run through the staged
  + stop-the-world install path (`compiler.rs:1166` "so it can't race mutator
  table reads", commit 7c4cace). Mutators are paused; `get_function_by_pointer`'s
  raw `f.pointer` read is safe-by-exclusion (not by atomic). No fix needed.
- **Mode 1 (realloc UAF): LATENT — real, rare.** New-function / OSR-inner-fn
  registration (`cleanup_after_osr_compile` → `flush_deferred_functions`) is
  append-only BUT runs NON-STW on the compiler thread (a STW there deadlocks).
  An append `Vec::push` REALLOCS at capacity boundaries → moves the buffer →
  dangles a concurrent `get_function_by_pointer` reader (`make_closure`). Rare
  (doubling boundaries + infrequent registration + short window → 0/5), but a
  real use-after-free. **OSR-on INCREASES exposure** (more compiler-thread
  OSR-compiles → more append-pushes), so close it BEFORE the flip.

Fix = make `Runtime::functions` never realloc:
- **DONE (commit 6ea4b77):** `src/append_only_chunked.rs` — `AppendOnlyChunked<T>`:
  fixed array of `AtomicPtr` chunk slots + `AtomicLen`; chunks never move/free;
  existing elements address-stable across grows. Lock-free reads (Acquire len +
  chunk); `push` writes the element fully THEN Release-bumps len (the
  x86-sensitive publish order — readers never see a half-init element /
  unpublished chunk). No reclamation → no leak/epoch/hazard. Writes `&mut self`.
  3/3 unit tests incl the load-bearing `stable_addresses_across_grow`.
- **DONE (commit f15eb6b) — the rewiring:** swapped `Runtime::functions:
  Vec<Function>` → `AppendOnlyChunked<Function>`. Container gained `last()`,
  `truncate()` (compile-failure rollback — safe, rolled-back fns are unpublished),
  `IndexMut`, `IntoIterator` for `&`/`&mut`. `upsert_function`/`rebind`/
  `revert_function_pointer` restructured to find-index-via-immutable-scan-FIRST
  (the container's method indexing borrows differently than Vec's NLL). Builds
  clean; container unit tests 3/3; **default-GC full suite 437/437.** Minor: a
  redundant wrapper block left in `upsert_function` (trivial cosmetic cleanup).
  **VALIDATED (ready for the bar):** ×3-GC plain suite 437/437
  (gen/compacting/mark-sweep); OSR-on suite clean on all 3 (one low-frequency
  pre-existing flake, didn't recur in 3 re-runs); **x86 (Rosetta) suite 437/437 —
  the Acquire/Release publish is correct under x86-TSO**; container unit tests
  3/3. Remaining: the independent container reviewer (publish ordering +
  realloc-elimination genuine + rewiring behavior-identical + truncate-rollback
  safety). (Original site patterns, for reference:)
  - `.functions[idx]` → `Index` impl already provides this (works as-is).
  - `.functions.push(f)` → `.functions.push(f)` (returns idx; same).
  - `.functions.get(idx)` → same (`get` impl).
  - `.functions.len()` → same.
  - `.functions.iter()` → same (returns `impl Iterator<&T>`).
  - `.functions.iter_mut()` → same; NOTE: the in-place upsert mutation
    (`overwrite_function`) is STW-only, so the `&mut` path is sound (no concurrent
    reader). `iter_mut`/`get_mut` need `&mut self` (already the case at those
    sites — they're behind `get_runtime().get_mut()`).
  - bare `.functions` (e.g. passing `&self.functions`) → audit each; most want
    `.iter()`/`.get()`/`.len()`.
  Container API intentionally mirrors `Vec` (push/get/len/iter/iter_mut + `Index`)
  to minimize churn. After rewiring: `cargo build` clean, then validate.
- **VALIDATE:** suite ×3 GCs (generational/compacting/mark-sweep) + `gc-always`,
  WITH OSR-on folded in (that IS the B OSR-correctness validation — avoids
  redundant builds). **x86-validate** the Release/Acquire publish on
  computer.jimmyhmiller.com (the new-chunk publish + len-bump is the TSO-vs-ARM
  divergence point — a static memory-model argument is not enough, per the
  torn-read lesson). Then bring A-lite for independent review (chunked container
  correct + identical index/get/iter/push semantics + lock-free reads preserved +
  realloc-elimination genuine).

### A-lite review NITS — ALL DONE (folded in with the benefit gate)

A-lite SIGNED OFF (approve-with-nits; the reviewer's own concurrent stress —
1 writer × 6+ chunk boundaries + 4 readers — passed ARM64 20/20, x86-TSO 15/15).
All three nits landed:
1. ✅ **Cosmetic:** deleted the redundant wrapper block in `upsert_function`
   (`src/runtime.rs`); `cargo fmt` reindented (behavior-identical).
2. ✅ **Live-coding invariant:** `iter()` now uses `map_while(|i| self.get(i))`
   (`src/append_only_chunked.rs`) — stops cleanly when a concurrent compiler-thread
   `truncate` shrinks `len` mid-iteration, instead of the old `.expect` PANIC.
3. ✅ **Typo:** rewrote the garbled `chunk_ptr` doc-comment.

### Prereq 2 — BENEFIT GATE (no-regression)  ✅ DONE — but NOT the way this doc predicted

**Shipped gate** (`src/compiler.rs`, behind `BEAGLE_OSR_GATE`, default on). A loop
is skipped (stays tier-1 — never a regression, at worst a missed win) when ANY of:

1. **Tiers up via the entry counter** (cheap; checked at the TOP of
   `build_osr_variant_inner`, BEFORE the recompile). `osr_tiers_up_via_entry`:
   `specialized_names.contains(name)` OR the **live entry-counter call_count ≥
   half the tier-up threshold**. OSR exists only for hot loops in functions that
   DON'T tier up (entered once: fannkuch / main loops, call_count ~1). A call-hot
   function (nbody/`advance`) already gets warm tier-2, so OSR is redundant.
2. **Boxed-float function** — any float arithmetic in the CFG (whole-function
   scan). OSR can't unbox floats (Phase D), so the variant is pure overhead.
3. **Call-bound loop** — an `Op::Call` dominating a latch (the doc's original
   CFG-dominance idea; kept — perf-neutral on the benchmarks but guards unseen
   call-bound int-loop shapes).

**Measured (final, `BEAGLE_OSR_THRESHOLD=10000`, best-of-4, all bit-identical):**

| benchmark | off | on+gate | ratio |
|---|---|---|---|
| fannkuch n=10 | 2.94s | 1.97s | **1.49×** ✅ |
| fannkuch n=11 | 35.41s | 20.57s | **1.72×** ✅ |
| nbody n=200000 | 0.57s | 0.56s | **1.02×** ✅ (was 0.70×) |
| nbody n=500000 | 1.08s | 1.08s | **1.00×** ✅ (was 0.84×) |
| spectral n=500 | 0.58s | 0.60s | 0.97× (flat — boxed float, Phase D) |
| mandelbrot n=1500 | 0.70s | 0.70s | 1.00× (flat) |
| binary_trees n=14 | 0.20s | 0.20s | 1.00× |

**Two corrections to this doc's predictions** (probe-first earned both):

- **The regressor was NOT `run#L0` being call-bound.** Isolation
  (`THRESHOLD=1e9` = instrumentation-only = FREE 0.60s; transfer = 0.86s, with
  `run#L0` *already gated*) proved the cost is `advance`'s **boxed-float** loops.
  And pure CFG structure CANNOT distinguish them from fannkuch's int loops: Beagle
  compiles `dx*dx` as an inline int fast-path (dominates the latch, no Call) + a
  COLD bail to the generic handler (the Call, on the non-dominating bail edge) —
  an always-bailing float op is structurally identical to a rarely-bailing int
  op. Only call-frequency / type-feedback separates them, hence signal (1).
- **The gate had to run BEFORE the recompile, not after.** A gated loop still
  paid the full feedback-recompile (the expensive `compile_ast_with_feedback`),
  a FIXED ~0.17s for nbody's 3 gated `advance` loops — that *was* the residual
  regression even when nothing built. Moving the cheap entry-counter check ahead
  of the recompile (signals 2/3 still need the specialized IR, so they stay after)
  brought nbody to exact parity.

A-lite review nits (above): all 3 folded in.

---

#### Historical investigation (superseded by the shipped gate above)

The analysis below is the original handoff's gate design and the ATTEMPT-1 /
MEASURED / CFG-DOMINANCE reasoning. It is kept for the investigation trail; the
CFG-DOMINANCE conclusion it reaches was empirically disproven (see corrections
above). Read the shipped gate, not this, as the source of truth.

Measured A/B (OSR off→on, `THRESHOLD=10000`, all bit-identical):

| benchmark | off | on | ratio |
|---|---|---|---|
| fannkuch n=10 | 3.16s | 2.11s | **1.50×** ✅ |
| fannkuch n=11 | 40.98s | 23.56s | **1.74×** ✅ |
| spectral n=500 | 0.63s | 0.65s | 0.97× (flat — boxed float) |
| mandelbrot n=1500 | 0.76s | 0.78s | 0.97× (flat — boxed float) |
| **nbody n=200000** | 0.60s | 0.85s | **0.70× ❌ REGRESSION** |
| **nbody n=500000** | 1.14s | 1.44s | **0.79× ❌ REGRESSION** (confirmed 3×) |
| binary_trees n=14 | 0.24s | 0.24s | 1.00× (alloc-bound) |

Root cause: `build_osr_variant_inner` (`compiler.rs:1090-1140`) builds + publishes
F_osr **UNCONDITIONALLY** — no benefit gate. nbody = empty `int_slots` (float-fed,
nothing promotes) → F_osr is OSR-entry + transfer overhead with zero payoff →
slower, even per-iteration (regression persists at n=500000, so F_osr is genuinely
worse than tier-1 for this shape). BUT fannkuch ALSO has empty `int_slots`
(deopt-ineligible, array writes) yet BENEFITS via the non-deopt tier-2
(array-inlining + int regalloc). So **"int_slots empty" ≠ "no benefit"** — the
gate must capture the real benefit condition.

Gate design (Leader-locked): **conservative static-predict for v1.** Fire OSR ONLY
where CONFIDENT of a tier-2 win (the loop body has tier-2-improvable ops:
promotable-int live-ins OR inlinable-array ops, and is NOT boxed-float-dominated).
When UNSURE, **DON'T fire** → fall back to tier-1 (the safe baseline). This makes a
mis-predicted shape — INCLUDING shapes not in the 7 benchmarks (real programs have
shapes the suite doesn't) — get no-speedup, NEVER a regression. No-regression is
the HARD bar; a missed-speedup is acceptable. A/B-validate against the full
benchmark suite (fannkuch keeps ~1.5-1.74×, nbody not regressed, every benchmark
≥ tier-1). Measured/profile-guided is a future refinement IF v1 proves too
conservative. **This is the judgment-heavy piece — do it with a clear head / fresh
session, AFTER the mechanical rewiring.** Lives in `build_osr_variant_inner`
(after `int_slots` is computed, ~compiler.rs:1117 — `return None` to SKIP, which
the caller `build_osr_variant` turns into `osr_set_failed` → the loop stays
tier-1). Behind an env override `BEAGLE_OSR_GATE` (default on).

**ATTEMPT 1 (done, reverted to clean — the simple static signals DON'T separate
the cases; here's exactly why, so the fresh session starts past the dead end):**
Scan `base_ir.instructions` for op classes:
- ARRAY signal = the inlined mutable-array ops = register-indexed heap ops
  (`HeapLoadReg`/`HeapLoadByteReg`/`HeapStoreOffsetReg`/`HeapStoreByteOffsetReg`/
  `HeapStoreByteOffsetMasked`) + `CompareAndSwap`.
- FLOAT signal = `FloatBinOp`/`Add|Sub|Mul|DivFloat`/`CompareFloat`/`GuardFloat`/
  `IntToFloat`/`FRoundToZero`/`StoreFloat`/`LoadLocalFloat`/`StoreLocalFloat`/
  `MoveFloat`/`Fmov*`.
- CALL signal = `Call`/`Recurse`/`TailRecurse`.

Predicate `!int_slots.is_empty() || (has_array && !has_float)` → fannkuch n=11
**1.78× ✓** but **nbody still 0.72× ✗**. Debug (`BEAGLE_OSR_DEBUG=1`) shows WHY:
nbody's `advance` loops ARE correctly skipped (array+float), but **`nbody/run#L0`
fires** — the OUTER iteration loop indexes `bodies[]` (→ array=true) and the
floats live INSIDE `advance` (a separate fn, so run's own IR has float=false),
yet run's loop CALLS `advance` every iteration. A loop-carried value can't
register-promote across that call safepoint → F_osr is pure overhead → regression.
So `array && !float` MISFIRES on an incidental-array, call-dominated loop.

Adding `!has_call` to the predicate then BREAKS fannkuch (drops to 0.99×): the
inlined array ops have a SLOW-PATH (bounds-OOB) fallback `Call` to the real
wrapper, which is COLD (never taken on the hot path) but trips `has_call`. So a
blanket "call-free" check is wrong — fannkuch's calls are cold fallbacks, nbody's
`advance` call is hot.

**THE NUT (for the fresh session): distinguish a HOT-path call (nbody/advance,
on every iteration → no promotion → regress) from a COLD fallback call (fannkuch
array-OOB → skipped on the hot path → real win).**

**MEASURED approach — DISPROVEN by calibration (don't build it).** We assumed
"did the loop's values promote? / SlotStore count" was the cleaner signal. It is
NOT. Ran `BEAGLE_SSA_LOG_MEM2REG=1 BEAGLE_OSR=1` on the F_osr variants:
- `fannkuch$osr`: candidates=149, **promoted=125 (84%)** — promotes well, WINS.
- `nbody/run$osr0`: candidates=13, **promoted=10 (77%)** — promotes ALSO well, yet
  REGRESSES.
The promotion ratio does NOT separate win from regress (both ~77-84%). nbody/run's
regression is NOT failed promotion — it's OSR overhead (entry guards + buffer
transfer) NOT AMORTIZING on a CALL-DOMINATED loop: run's per-iteration cost is
dominated by the `advance()` CALL, which F_osr can't speed up (advance is its own
fn); promoting run's loop counter saves nothing. So SlotStore-count / mem2reg-result
is the wrong signal — discard it.

**CFG-DOMINANCE — the CONFIRMED path (this is what to build).** A perf heuristic
(reads CFG structure; does NOT alter gc-safety — gc-ADJACENT at most, and actually
not even that: it's pure control-flow structure). Gate: does the natural loop being
OSR'd contain a `Call` whose block **dominates a latch** (i.e. executes on EVERY
iteration → the loop is call-bound → OSR overhead won't amortize)? If yes → SKIP
(`return None`). fannkuch's cold OOB-fallback `Call`s are NOT latch-dominated (only
on the bounds-guard FAILURE edge, bypassed on the hot path) → don't gate; nbody/run's
`advance` call IS latch-dominated → gate out. INTEGRATION (in
`build_osr_variant_inner`, after `base_ir`+`info` are captured, ~compiler.rs:1061,
behind `BEAGLE_OSR_GATE`):
- `let cfg = crate::cfg::builder::build_cfg(&base_ir)?;` (or reuse the analysis CFG)
- `let loops = crate::cfg::loops::natural_loops(&cfg);` — `NaturalLoop { header,
  body: HashSet<BlockId>, latches: Vec<BlockId> }`. Find the loop whose `header`
  block corresponds to `info.header_label` (map the Label→BlockId; or, for a
  single-hot-loop fn, the sole/outermost loop).
- `let idom = crate::cfg::dom::compute_idoms(&cfg, &rpo);` + the `dominates(&idom,
  a, b)` helper (already in `cfg/loops.rs`/`cfg/dom.rs`).
- For each block in `loop.body` that contains an `Op::Call { .. }`: if it
  `dominates` any `loop.latches[i]` → the loop is call-bound → `return None`.
- Confirm the gate stays READ-ONLY on the CFG (a perf decision; no gc/IR mutation).
Then A/B-validate (fannkuch keeps ~1.5-1.74×, nbody NOT regressed, every benchmark
≥ tier-1 — the A/B is the backstop: too-aggressive → nbody regresses → caught;
too-conservative → fannkuch loses → caught), fold the 3 A-lite nits, re-validate
×3-GC + the container stress, then bring it + the flip for the bar.

### THE FLIP — gated on BOTH

Flip `BEAGLE_OSR` default-on (gate at `src/ast.rs:1296` env check). Ships ONLY
when A-lite (memory-safe — OSR-on increases mode-1 exposure) AND the benefit gate
(no regression) are both landed + reviewed. Bring the diff + measured-win numbers
(fannkuch ~1.5-1.74× vs Node v24, all benchmarks ≥ tier-1, bit-identical,
live-coding preserved — redefine-while-hot still takes effect via
`revert_all_specializations`) for independent review.

### Follow-on (NOT a flip blocker) — D: float OSR (Phase-D)

spectral/mandelbrot are FLAT because their live-ins are boxed floats the int path
doesn't unbox. The fix (OSR_PERF_HANDOFF.md §44-47): `GuardFloat` + `coerce_to_fp`
at the OSR entry (mirror `apply_float_param_versioning`); needs regalloc
FP-across-call preservation for call-bearing float loops. A separate win after the
flip — flat ≠ regression, so it does NOT gate the flip.

## Repro / measurement recipes
- OSR A/B: `BEAGLE_OSR=1 BEAGLE_OSR_THRESHOLD=10000 ./target/release/beag run <bench> <n>` vs without.
- OSR correctness: `BEAGLE_OSR=1 BEAGLE_OSR_THRESHOLD=300 BEAGLE_OSR_RECHECK=30 ./target/release/beag test resources/`.
- Function-table latent-race stress (0/5 = clean): `smoke/soak_starvation.sh` (or
  `BEAGLE_OSR=1 BEAGLE_SPECIALIZE_THRESHOLD=1` + soak_long under gc-always + saturation).
- Bit-identical: `diff` the OSR-off vs OSR-on stdout per benchmark.

## Key code locations
- A-lite container: `src/append_only_chunked.rs` (done). Rewire target:
  `Runtime::functions` (`src/runtime.rs:4767`), `get_function_by_pointer` (:9729),
  `upsert_function` (:9341), `overwrite_function`, `add_function` (:9511).
- Benefit gate (DONE): `src/compiler.rs::osr_tiers_up_via_entry` (cheap,
  pre-recompile) + `::osr_loop_skip_reason` (CFG: float + call-bound), both gated
  in `::build_osr_variant_inner` behind `BEAGLE_OSR_GATE`. CFG helpers:
  `src/cfg/builder.rs::build_cfg_for_osr_gate` + `::block_id_for_label`. Entry
  counts via `Compiler::entry_counter_slots` (populated in `add_function_counter`).
  The OSR trigger is in `src/ast.rs::maybe_emit_osr_backedge_check`.
- OSR machinery: `src/osr.rs`, `docs/OSR_DESIGN.md`, `docs/OSR_PERF_HANDOFF.md`
  (the int-path-near-warm + float Phase-D detail).
