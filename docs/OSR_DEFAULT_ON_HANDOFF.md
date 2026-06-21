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
- **TODO — the rewiring (mechanical, fully validated by review + suite ×3 GCs):**
  swap `Runtime::functions: Vec<Function>` → `AppendOnlyChunked<Function>`. ~50
  sites in `src/runtime.rs` + a few in `src/compiler.rs`, `src/repl.rs`,
  `src/builtins/reflect.rs`, `src/builtins/mod.rs`. Patterns to convert:
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

### Prereq 2 — BENEFIT GATE (no-regression): OSR fires only where tier-2 wins  [TODO — judgment-heavy]

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
session, AFTER the mechanical rewiring.** Likely lives in `build_osr_variant_inner`
(skip build/publish if not confident) and/or the OSR trigger in `src/ast.rs`.

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
- Benefit gate: `src/compiler.rs::build_osr_variant_inner` (:952, the
  unconditional build+publish), the OSR trigger in `src/ast.rs:1296`/:1336.
- OSR machinery: `src/osr.rs`, `docs/OSR_DESIGN.md`, `docs/OSR_PERF_HANDOFF.md`
  (the int-path-near-warm + float Phase-D detail).
