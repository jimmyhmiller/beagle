# SSA: state of things, the OOM, and the performance reality

**Date:** 2026-05-30. **Branch:** `ssa-foundation` (working tree on top of
commit `9ea4051`). **Author:** debugging session post-mortem.

This document is deliberately blunt. It records (1) what state the SSA work is
actually in, (2) the precise cause of the out-of-memory blow-ups, and (3) the
honest performance picture — including a number I reported as a win that turned
out to be measurement noise. It is the companion to
`docs/SSA_PRESSURE_EXPLOSION.md` (the pressure diagnosis) and
`docs/SSA_RUNTIME_PARITY_PLAN.md` (the phase plan).

---

## 0. Executive summary

- **Correctness: good.** Full `cargo run -- test resources/` is **364/364**
  under `BEAGLE_USE_SSA=1`, under both the default promotion gate (`≥2` reads)
  and aggressive promotion (`≥1`). Peak compiler memory ~0.05 GB. No OOM.
- **A real bug was found and fixed.** The OOM was a **variadic tail-call
  miscompilation** in CFG construction (`builder.rs`), pre-existing and
  unrelated to the register allocator. It made `variadic_recursive_test`'s
  `countdown` recurse forever, allocating without bound (the "190 GB").
- **Performance: SSA now beats legacy** (with `≥1` promotion, the default):
  **fib ~0.80×, btrees ~0.90×, nbody_specialize ~0.97×, nbody_full ~1.04×.**
  Net faster (geomean ≈0.92). Achieved by combining: pruned SSA (keeps
  pressure sane), the variadic-TCO correctness fix, the other contributor's
  slot-store forwarding + jump threading, and — the missing piece —
  **single-read slot promotion (`BEAGLE_SSA_PROMOTE_MIN_READS=1`)**, which
  collapses the store→load traffic (specialized fib went 36 slot ops → 7).
- **Correction to an earlier draft of this doc:** I previously wrote that the
  "fib 0.89×" reading was *noise* and that there was no runtime win. **That
  was wrong.** It was *real* `≥1` promotion — but the bisect-revert silently
  reverted the `PROMOTE_MIN_READS` env gate to a hardcoded `≥2`, so every
  later "≥1" run was actually `≥2` and read ~1.12×. Restoring the gate (and
  making `≥1` the default) reproduces the win stably (fib 0.78–0.83× across
  runs). The lesson: a reverted-during-bisect knob made the win look like
  noise. See §2.

So the session delivered a **correctness fix, a diagnosis, AND the speed**
the parity effort was after — on 3 of 4 benchmarks, with the 4th within 4%.

---

## 1. The OOM

### 1.1 Symptom

Running the corpus (or `bench_nbody_full`-style files) under
`BEAGLE_USE_SSA=1` with aggressive promotion drove the `beag` process to
**190 GB → 80 GB → 16 GB** of RSS across successive states, hammering the
machine. It was intermittent and looked like a compiler blow-up.

### 1.2 There were actually *two* distinct blow-ups, conflated

**(A) A compile-time O(G²) blow-up I introduced in Phase 5 coalescing.**
`regalloc/coalesce.rs::build_coalesce_groups` materialized, for each member of
a copy-affinity group of size *G*, a vector of all *G−1* other members —
**O(G²) memory**. Under `≥2` promotion, groups are tiny, so it never showed.
Under `≥1`, a single large copy chain became a ~150 K-member group →
~190 GB. **Fixed:** the function now returns an O(G) `member → representative`
map and coloring biases a whole group toward one representative color
(`color.rs`'s group-color logic). *(Note: that color.rs change was reverted
during bisection and is currently NOT re-applied — see §3.)*

**(B) A runtime infinite loop from a pre-existing variadic tail-call
miscompilation — the headline cause.** This is the one that kept recurring
after (A) was fixed, and it is *not* a memory-structure blow-up at all: the
generated code for `countdown` loops forever and allocates each iteration.

### 1.3 Root cause of (B): variadic tail-calls lower to a malformed CFG

`countdown(...nums)` is variadic and tail-recursive. Two facts collide:

1. **Variadic entry shape.** A variadic function's entry block has **8
   params** — the 8 incoming argument registers (X0–X7). The function learns
   how many args were actually passed by reading a separate **arg-count
   register** via `Op::ReadArgCount` — a register the *caller* sets before
   every call.

2. **I8 tail-call rewrite.** CFG construction (`builder.rs`) rewrites a tail
   self-call to `Jump(entry, args)` (Cranelift-style "loop back to entry").

For a variadic function this is **unsound**:

- The tail-call passes only the actual recursion args (0–3 for `countdown`),
  but entry has **8 params** → `ArgArityMismatch { expected: 8, got: 3 }`.
  Entry params 3–7 receive whatever happens to be in those registers →
  garbage.
- A `Jump` is not a call, so it **never re-establishes the arg-count
  register**. `ReadArgCount` inside the loop returns the *original* call's
  count, so the rest-array `nums` is rebuilt with the wrong length.

Either way the loop's control values are wrong. `countdown(3,2,1)` printed
`counting: 3 2 1` then looped on `counting: 1 1` forever (or skipped straight
to `done counting`, depending on register contents) — never terminating,
allocating on every iteration → OOM.

Verified directly from the dump: at `00_after_cfg_ize` (before mem2reg,
spilling, anything) `countdown`'s entry already shows
`block0(v2..v16) preds=[block124,126,129,132]` — 8 params, 4 tail-call
back-edges — and the verifier logs
`ArgArityMismatch { from: block124, to: block0, expected: 8, got: 3 }`.

### 1.4 Why it surfaced now (and why the spiller was blamed first)

On committed `HEAD`, `countdown` **bailed to legacy** (for unrelated
pressure/spill reasons), so the malformed SSA CFG was never emitted and the
test passed. The working tree's spiller changes reduced bailing, so
`countdown` started compiling *through* SSA — exposing the latent
`builder.rs` bug. That is why the bisection kept pointing at `spill.rs`: the
spiller was the thing that *changed whether the bug was reachable*, but it was
never itself wrong. The actual defect is in variadic tail-call lowering, and
it predates this entire session.

### 1.5 A secondary defect: the verifier is ignored

`compile_via_ssa` *runs* a verifier that *detects* the `ArgArityMismatch`,
logs it (`[cfg-verify] …`), and then **compiles the malformed CFG anyway**
(`[ssa-compile] OK`). The spec says a verifier failure should fall back to
legacy. Honoring that would have turned a silent miscompile into a safe (if
suboptimal) bail. This is worth closing independently of the real fix.

### 1.6 The fix

For a **variadic** function (detected by the presence of `Op::ReadArgCount`),
do **not** rewrite the tail self-call to `Jump(entry, …)`. Lower it to a real
self-`Recurse` (which sets the full calling convention, including the arg
count) and return its result:

```rust
// src/cfg/builder.rs, TailRecurse arm
I::TailRecurse(dst, args) => {
    let arg_vregs = translate_value_args(...)?;
    if is_variadic {
        let ret = /* dst */;
        pre_ops.push(Op::Recurse { dst: ret, args: arg_vregs, clobbers: AllCallerSaved });
        Ok(Terminator::Ret { value: ret })
    } else {
        Ok(Terminator::Jump { target: entry, args: arg_vregs })
    }
}
```

Correct, compiles through SSA, **no bail**. Cost: variadic tail-recursion
loses TCO (stack depth = recursion depth). Non-variadic TCO (fib,
`sum_helper`, etc.) is unaffected. This is a deliberate, correct trade — a
`Jump`-to-entry simply cannot model the variadic calling convention.

A complete fix that *preserves* variadic TCO would have to thread the
arg-count as an explicit value and re-establish all 8 arg-register params on
the back-edge — a larger change in the documented-fragile variadic path. Not
done here.

---

## 2. Performance — SSA now beats legacy

### 2.1 The numbers (12-core machine, quiet, `scripts/ssa_diff.sh`, `≥1` default)

| benchmark | SSA / legacy (repeated runs) | verdict |
|---|---|---|
| fib_specialize | 0.79, 0.79, 0.83 → **~0.80×** | SSA ~20% faster |
| btrees_specialize | 0.91, 0.89, 0.90 → **~0.90×** | SSA ~10% faster |
| nbody_specialize | 0.90, 0.96, 1.05 → **~0.97×** | ≈parity (mostly faster) |
| nbody_full | 1.04, 1.05, 1.02 → **~1.04×** | ~4% slower (last laggard) |

Net **geomean ≈0.92 (≈8% faster)**. Static metrics: maxlive_gp 10 (was 407
unpruned), bails 0, edge-moves 0. Validated **364/364** at `≥1`, peak ~0.05 GB.

### 2.2 What actually unlocked the win (and the "noise" mistake)

The win is **single-read slot promotion** (`PROMOTE_MIN_READS = 1`, now the
default). It collapses the store→load round-trips the AST compiler emits for
every intermediate: **specialized fib went from 36 slot ops → 7** (fewer than
legacy's 11). mem2reg promotion candidates jumped 6 → 30.

This only works because three other things are in place:
- **Pruned SSA** keeps pressure sane (no dead-φ explosion) so promoting
  everything doesn't blow up (maxlive 10, not 407).
- The **variadic-TCO fix** removes the OOM so `≥1` can run the full corpus.
- The other contributor's **slot-store forwarding + jump threading + branch
  inversion** trim the remaining traffic and control flow.

**The earlier "noise" diagnosis was wrong.** An intermediate draft of this
doc claimed the "fib 0.89×" reading was noise. It was real `≥1` promotion. The
confusion: during the OOM bisection I reverted `mem2reg.rs`, which silently
reverted the `PROMOTE_MIN_READS` env gate to a **hardcoded `≥2`**. Every later
"≥1" run was therefore actually `≥2` (candidates stayed at 6, slot ops at 36,
fib at ~1.12×) — so the win looked unreproducible. Restoring the env gate
(and defaulting it to `1`) reproduces fib ~0.80× stably. *Lesson: when a
result vanishes after a revert, check whether the revert disabled the very
knob that produced it.*

### 2.3 Why nbody_full is still ~4% slower

`nbody/advance` (the hot numeric loop) still has **276 slot ops vs legacy's
114**, because **65 of its slots are I9-rejected** — GP values live across
safepoints (the per-FP-op `InlineBranch` bail edges) that must stay slotted
for GC. Some are genuine heap pointers (the bodies array — correctly slotted);
some are int indices/counters that are *not* pointers and could live in
registers, but the I9 gate is conservative and slots all GP cross-safepoint
values.

Closing this needs one of (both substantial, both GC-critical):
- **A non-pointer I9 exemption** — prove a cross-safepoint GP value is a
  tagged non-pointer (the other contributor's `known_non_pointer_defs` is a
  start; it would need extending to guarded-int / int-arithmetic results) and
  let it stay in a callee-saved register across the call, as legacy does.
- **Live-range splitting** (a Braun–Hack spiller) — keep the value in a
  register where used, spill only across the actual safepoint.

Neither is worth the GC risk right now given this session's history. nbody_full
at 1.04× is the documented remaining gap, not a regression.

---

## 3. Current working-tree state

On top of `9ea4051`. `cargo build --release` is clean; 364/364 under SSA at
both gates.

| File | Change | Keep? |
|---|---|---|
| `src/cfg/builder.rs` | **Variadic tail-call fix** (the OOM fix) + `is_variadic` detection | **Yes — the real win** |
| `src/cfg/mem2reg.rs` | **Pruned SSA** (`compute_slot_live_in` + IDF gate, `BEAGLE_SSA_NO_PRUNE` toggle) + **dead-init elimination** (zero-init only if slot live-in at entry) | Yes — correct; viability groundwork |
| `src/cfg/regalloc/spill.rs` | **Phase 4 remat** + prior-session single-pass Belady spiller (`compute_spill_set`, block-param spilling gated off) | Yes, but large/unreviewed; see caveats |
| `src/cfg/emit_legacy.rs` | Memory **size guards** (`BEAGLE_SSA_MAX_VREGS`/`MAX_IG_EDGES`) — bail pathological functions before quadratic structures | Optional safety net; was OOM-chasing scaffolding |
| `src/cfg/mod.rs` | `kind_name()` diagnostics on `Op`/`Terminator` | Harmless |
| `src/cfg/regalloc/mod.rs` | `pub mod coalesce;` | **Dead — see below** |
| `src/cfg/regalloc/coalesce.rs` (untracked) | Phase 5 conservative coalescer (O(G) representative-map version) | **Currently UNWIRED / dead code** |
| `docs/SSA_PRESSURE_EXPLOSION.md` (untracked) | The pressure diagnosis (validated) | Yes |
| `docs/SSA_ARCHITECTURE.md`, `SSA_RUNTIME_PARITY_PLAN.md` | Diagnostics/env-var + status updates | Yes, but need a pass to reflect §2 (pruning ≠ speed) |
| `ssa_bail.log` (untracked) | stray log | Delete |

### Loose ends that must be resolved before any commit

1. **`coalesce.rs` is dead code.** During bisection I reverted `color.rs` to
   `HEAD`, so nothing calls `build_coalesce_groups`. That is exactly the
   forbidden **F2 "dead-coded coalescer"** smell. Either re-wire Phase 5
   (re-apply the `color.rs` group-color changes) or delete `coalesce.rs` +
   the `pub mod coalesce;` line. Given that committed pairwise coalescing
   already drives edge-moves to 0 on the benchmarks and Phase 5 showed no
   measurable runtime benefit, **deleting it is the simpler honest choice**
   unless a transitive-chain case is found that needs it.
2. **`spill.rs` is a large, partly-inherited rewrite** (prior session +
   Phase 4). It is correct on the corpus but has not been through
   `/ssa-review`, and it is the component whose bail-reduction exposed the
   variadic bug. Review before trusting.
3. **The verifier-ignored-`ArgArityMismatch` defect (§1.5)** should be closed:
   make `compile_via_ssa` bail malformed CFGs to legacy as a safety net.
4. **Docs** (`SSA_ARCHITECTURE.md`, parity plan) still imply pruning/`≥1` is a
   runtime win; update to match §2.

---

## 4. Recommendations

1. **Land the variadic tail-call fix on its own.** It is an isolated,
   validated correctness fix (the SSA path now compiles variadic recursion
   correctly instead of relying on a bail). Highest-value, lowest-risk commit.
2. **Land pruned SSA + dead-init elimination** as the pressure-explosion fix,
   described honestly as *viability groundwork, not a speedup*. Keep the
   `BEAGLE_SSA_NO_PRUNE` toggle.
3. **Resolve `coalesce.rs`** (delete or re-wire) — do not commit dead
   coalescer code.
4. **Do not flip the default promotion gate to `≥1`.** It is correct but not
   faster, and it costs compile time. Leave default `≥2`.
5. **Close the verifier safety-net gap** (§1.5).
6. **Treat the real perf work as its own effort** (§2.4): a Braun–Hack
   pressure-bounded spiller with live-range splitting is the prerequisite for
   beating legacy on numeric loops. Don't expect a win before that.

---

## 5. Reproduction

```sh
cargo build --release

# Correctness (both gates), with a memory watchdog for safety:
BEAGLE_USE_SSA=1 ./target/release/beag test resources/                     # >=2 default
BEAGLE_USE_SSA=1 BEAGLE_SSA_PROMOTE_MIN_READS=1 ./target/release/beag test resources/   # >=1

# The OOM repro (pre-fix): variadic tail recursion miscompiles.
BEAGLE_USE_SSA=1 ./target/release/beag run resources/variadic_recursive_test.bg
#   expect "counting: 3 2 1 / 2 1 / 1 / done counting"; pre-fix it loops forever.

# Inspect the malformed CFG / verifier:
BEAGLE_USE_SSA=1 BEAGLE_SSA_VERIFY=1 ./target/release/beag run resources/variadic_recursive_test.bg 2>&1 | grep -i ArgArity

# Pressure collapse from pruned SSA (advance):
BEAGLE_USE_SSA=1 BEAGLE_SSA_PROMOTE_MIN_READS=1 BEAGLE_SSA_REGALLOC_STATS=1 \
  ./target/release/beag run resources/bench_nbody_full.bg 2>&1 | grep 'advance'

# Perf gate (vs legacy, same run):
BEAGLE_SSA_PROMOTE_MIN_READS=1 ./scripts/ssa_diff.sh
#   note: its REGRESSION lines compare to a STALE baseline.tsv; read the
#   SSA/LEG column for the honest same-run ratio.
```

Relevant env vars: `BEAGLE_USE_SSA`, `BEAGLE_SSA_PROMOTE_MIN_READS` (default 2),
`BEAGLE_SSA_NO_PRUNE`, `BEAGLE_SSA_NO_REMAT`, `BEAGLE_SSA_VERIFY`,
`BEAGLE_SSA_LOG_BAIL`, `BEAGLE_SSA_REGALLOC_STATS`, `BEAGLE_SSA_MAX_VREGS`,
`BEAGLE_SSA_MAX_IG_EDGES`.
