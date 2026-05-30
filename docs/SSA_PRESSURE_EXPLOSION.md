# SSA register-pressure explosion: root cause and fix

**Status:** diagnosis confirmed; **Fix 1 (pruned SSA) implemented and measured —
see §7.** Fixes 2 and 3 still proposed.
**Audience:** reviewer auditing the diagnosis and the Fix-1 results.
**Branch:** `ssa-foundation`. **Date:** 2026-05-30.

This document is meant to be *checked*, not taken on faith. Every measured
claim has a reproduction command; every code claim has a `file:line`. Claims
that are predictions (not yet measured) are explicitly labelled **[PREDICTED]**.

---

## 1. The problem

`BEAGLE_USE_SSA=1` is correct (full `cargo run -- test resources/` passes) but
**slower at runtime than the legacy linear-scan allocator** on the benchmark
set. On a quiet machine (12 cores, no other load), same-run SSA-vs-legacy:

| benchmark | SSA / legacy |
|---|---|
| btrees_specialize | 1.03× (≈ parity) |
| fib_specialize | 1.11× |
| nbody_full | 1.50× |
| nbody_specialize | 1.63× |

The plan (`docs/SSA_RUNTIME_PARITY_PLAN.md`) bet that promoting more stack
slots into SSA values (relaxing the mem2reg profitability gate from "≥2 reads"
to "≥1 read" — its "Phase 6") would remove the stack-slot memory traffic that
makes the SSA code slow. **The opposite happened:** with the gate relaxed,
every benchmark got *worse*, and the register allocator's reported peak
pressure exploded.

| metric (whole-compile aggregate) | gate `≥2` | gate `≥1` |
|---|---|---|
| fib SSA/legacy | 1.11× | 1.21× |
| nbody_full SSA/legacy | 1.50× | 1.87× |
| nbody_specialize SSA/legacy | 1.63× | 2.98× |
| peak `maxlive_gp` (any fn) | 125 | **407** |
| edge moves | 0 | 2554 |
| function bails to legacy | 9 | ~90 |

The question this document answers: **why does promotion explode register
pressure, and precisely how do we fix it?**

---

## 2. How to reproduce the measurements

All commands assume a release build (`cargo build --release`) and the
env-flag that gates the promotion threshold (added for this investigation;
default is unchanged at `2`):

```
src/cfg/mem2reg.rs:  BEAGLE_SSA_PROMOTE_MIN_READS  (default 2)
```

**Per-function register pressure** (`maxlive_gp` = max #GP values live at any
program point = optimal #colors for the chordal interference graph):

```sh
# gate >=2 (current default)
BEAGLE_USE_SSA=1 BEAGLE_SSA_REGALLOC_STATS=1 \
  ./target/release/beag run resources/bench_nbody_full.bg 2>&1 \
  | grep 'regalloc-stats' | grep advance

# gate >=1 (promotion relaxed)
BEAGLE_USE_SSA=1 BEAGLE_SSA_PROMOTE_MIN_READS=1 BEAGLE_SSA_REGALLOC_STATS=1 \
  ./target/release/beag run resources/bench_nbody_full.bg 2>&1 \
  | grep 'regalloc-stats' | grep advance
```

Observed `bench_nbody_full/advance`:

| gate | `maxlive_gp` | interference edges |
|---|---|---|
| `≥2` | 20 then 36 | 24k then 128k |
| `≥1` | 121 then **285** | 217k then **1.4M** |

(`advance` is logged twice — it is compiled twice in this run. Both rows move
together; use the larger.)

**Dump the SSA form** to count block-params (φ-nodes) directly:

```sh
BEAGLE_USE_SSA=1 BEAGLE_SSA_PROMOTE_MIN_READS=1 \
  BEAGLE_SSA_DUMP_FN=bench_nbody_full/advance BEAGLE_SSA_DUMP_DIR=/tmp/d \
  ./target/release/beag run resources/bench_nbody_full.bg >/dev/null 2>&1
# the post-optimization CFG that feeds regalloc:
F=$(ls /tmp/d/*05_after_opts.cfg | head -1)

# total φ across all blocks (params are rendered `blockN(vK:gp, ...)`):
grep -E '^block[0-9]+\(' "$F" | grep -oE ':(gp|fp)' | wc -l
# φ on the single widest block:
grep -E '^block[0-9]+\(' "$F" | awk -F: '{n=gsub(/:(gp|fp)/,""); print n}' | sort -rn | head -1
# dead synthetic zero-inits:
grep -cE 'ConstRawValue 0x0$' "$F"
```

Observed for `advance`:

| | total φ | widest block φ | maxlive_gp | dead `ConstRawValue 0` |
|---|---|---|---|---|
| `≥2` | 99 | — | 36 | 37 |
| `≥1` | **1328** | **282** | **285** | 289 |

**The widest single block carries 282 φ-params, which equals the
maxlive_gp of 285** (282 φ + a few non-φ values live there). This is the
explosion, localized to one block.

---

## 3. Root cause: mem2reg builds *minimal* SSA, not *pruned* SSA

`advance` is a doubly-nested loop over body pairs. Its genuine register
pressure — the number of values a correct compiler must keep simultaneously
live — is small (tens, bounded by the `≥2` figure of 36, which already
promotes the multiply-read values). The 285 is **artificial**.

### The mechanism

`src/cfg/mem2reg.rs` places a φ for a promoted slot at the **iterated
dominance frontier (IDF) of that slot's write sites**, with **no check that
the slot is live there**:

```rust
// src/cfg/mem2reg.rs:140-152
for &slot in &promotable {
    let write_blocks = slot_writes[&slot]...;
    let idf = iterated_dominance_frontier(&write_blocks, &df);   // line 142
    ...
    for block in idf_sorted {
        let phi_vr = f.new_vreg(class);
        f.block_mut(block).params.push(phi_vr);                  // line 149 — unconditional
        ...
    }
}
```

This is textbook **minimal SSA** (Cytron et al. 1991): a φ at the IDF of every
definition. It is *correct* but places φ-nodes for variables that are **dead**
at the join.

A loop header is in the IDF of *every* slot written anywhere in the loop body.
So under `≥1`, all ~282 slots written inside `advance`'s loop get a φ at the
header — but only a handful (loop counter, accumulators) are genuinely
**loop-carried**. The other ~270 are **dead φ-nodes**: the slot is written and
read within a single iteration and never carried across the back-edge.

Dead φs would be harmless if they stayed dead. They do not, because the rename
pass wires an incoming argument for **every** φ from **every** predecessor,
unconditionally:

```rust
// src/cfg/mem2reg.rs:256-271
for succ in succs {
    if let Some(slots) = phis_at_block.get(&succ) {
        for slot in slots {
            let top = stacks.get(slot)...;       // the slot's current SSA value
            push_edge_arg(&mut ...terminator, succ, top);   // line 270
        }
    }
}
```

So every dead φ at the loop header forces its predecessor (the loop body's
back-edge) to **pass the slot's value along the edge** — which keeps that
value **live across the back-edge**. 282 dead/near-dead φs ⇒ 282 values pinned
simultaneously live at the header ⇒ `maxlive_gp = 285`.

### Why this also hurts the current `≥2` default

Even at `≥2`, `advance` already has **99 φ** and **37 dead `ConstRawValue 0`**
zero-inits, and `maxlive_gp = 36 > 12` (the GP pool). So the current default
*also* carries dead-φ pollution and *also* must spill — it is just less
severe. Pruned SSA helps both regimes.

---

## 4. The fix

Two independent, standard, well-understood changes. The first removes the
artificial pressure (the subject of this document); the second is required to
turn the now-sane pressure into a runtime win over legacy.

### Fix 1 — Pruned SSA (removes the explosion)

Place a φ for slot `S` at IDF block `B` **only if `S` is live-in at `B`.**

1. **Slot liveness.** Add a backward dataflow over the pre-SSA CFG, keyed on
   `SlotId` (not `VReg`):
   - `gen[B]`  = slots `SlotLoad`-ed in `B` before any `SlotStore` in `B`.
   - `kill[B]` = slots `SlotStore`-d in `B`.
   - `live_in[B] = gen[B] ∪ (live_out[B] − kill[B])`, iterate to fixpoint.
   This mirrors the existing VReg liveness in
   `src/cfg/regalloc/liveness.rs`, just over slots. ~50 lines.

2. **Gate φ placement** at `mem2reg.rs:147`:
   ```rust
   for block in idf_sorted {
       if !slot_live_in[&block].contains(&slot) { continue; }   // pruned SSA
       ...
   }
   ```

A φ is now created only when the slot's value actually reaches a use through
that join — i.e. for genuinely loop-carried / merge-live values. Dead φs are
never created, so no value is pinned across an edge where it is dead.

**[PREDICTED]** `advance`'s loop header drops from 282 φ to the ~5–15
loop-carried values; `maxlive_gp` falls **285 → ≈ genuine (30–40)**. The `≥2`
regime also sheds most of its 99 φ and 37 dead zero-inits.

**Correctness.** Pruned SSA is standard and correct: if a variable is not
live at a join, no downstream use observes a merge there, so omitting the φ
cannot change behavior. The rename pass already reads "current value" from the
dominating definition via its per-slot stack; with the φ absent it reads the
idom's value, which is the value that was live anyway. The I9/I10 GC gates
(`mem2reg.rs:98+`, the handler-block gate at `mem2reg.rs:73+`) are downstream
and read-count/liveness-agnostic, so pruning does not affect GC-safety.

### Fix 2 — Single-pass Belady spiller (turns sane pressure into a win)

Pruning brings pressure to the *genuine* ~36, still above the 12-register
pool, so spilling is still required. The current spiller is the wrong shape:
it iterates *color → check → spill a batch → recolor*, capped at
`BEAGLE_SSA_SPILL_CAP=8` spills and `BEAGLE_SSA_SPILL_MAX_ROUNDS=2` rounds
(`src/cfg/regalloc/spill.rs:151,165,213`), and **bails the whole function to
legacy** when it can't fit inside that cap. A function needing ~24 spills
bails instantly.

Replace it with the **Braun & Hack 2009 single-pass SSA spilling algorithm
("MIN")**:

- Walk each block maintaining a register set `W`, `|W| ≤ k`. On each
  instruction: reload operands not in `W`; whenever `|W| > k`, evict the value
  whose **next use is furthest in the future** (Belady MIN) to a slot.
- Loop headers seed `W` from a next-use-distance / loop-pressure analysis so
  loop-carried values are *not* reloaded every iteration (Braun §4.2).
- This bounds pressure to `≤ k` **by construction**, so the existing chordal
  coloring is then *guaranteed* to fit ⇒ **zero bails, ever.** Live-range
  splitting is inherent (a value may be in a register in one block and a slot
  in another). Rematerialization (already implemented:
  `spill.rs::remat_op_for`) and coalescing (`regalloc/coalesce.rs`) plug in
  unchanged.

**[PREDICTED]** Because Belady eviction is near-optimal, the values kept in
registers are at least as good as legacy linear-scan's, while the cold values
go to slots exactly as legacy already does — so SSA should reach **≤ legacy**
and, with the hottest values reliably in registers across the loop, beat it.

### Fix 3 — Dead-init cleanup (cheap, helps both regimes)

Emit a slot's synthetic zero-init (`ConstRawValue 0` at entry,
`mem2reg.rs:160+`) **only** for slots with an actual read-before-write path.
Removes 37 (`≥2`) / 289 (`≥1`) dead instructions from `advance`'s entry block.
Pruned SSA removes most of the dead φ that consume these; this removes the
inits themselves.

---

## 5. What is proven vs. predicted

**Proven by measurement (§2):**
- SSA is 1.03–1.63× slower than legacy at gate `≥2`; worse at `≥1`.
- Promotion (`≥1`) takes `advance` from `maxlive_gp` 36 → 285.
- That 285 is one block with 282 φ-params.
- mem2reg places φ at the IDF with no liveness gate (`mem2reg.rs:142,149`),
  and wires an edge arg for every φ unconditionally (`mem2reg.rs:256-271`).
- The current spiller bails past a small spill cap (`spill.rs:213`).

**Predicted, to be validated by implementing Fix 1 then re-running §2:**
- Pruned SSA collapses the 282-φ block to ~5–15 φ and `maxlive_gp` to ~30–40.
- With the single-pass spiller, SSA reaches and beats legacy runtime.

**The cleanest way for a reviewer to falsify the core claim:** implement Fix 1
(slot liveness + the one-line gate) and re-run the dump command in §2. If the
widest-block φ count does *not* drop dramatically, the "dead φ from non-pruned
SSA" diagnosis is wrong.

---

## 7. Results — Fix 1 implemented (measured)

Fix 1 (pruned SSA) is implemented in `src/cfg/mem2reg.rs`:
`compute_slot_live_in` (backward slot-liveness dataflow) plus a one-line gate
in the φ-placement loop (`if !slot_live_in[block].contains(&slot) { continue }`).
Promotion gate is still env-controlled (`BEAGLE_SSA_PROMOTE_MIN_READS`,
default 2); pruning is unconditional.

**Pressure collapse on `bench_nbody_full/advance` (gate `≥1`), measured:**

| | before pruning | after pruning |
|---|---|---|
| total φ | 1328 | 243 |
| widest block φ | 282 | **3** |
| maxlive_gp | 285 | **7** |

The diagnosis is therefore confirmed: the 285 was dead-φ pollution, not genuine
pressure. (Genuine GP pressure for `advance` is 7 — *below* the 12-register
pool, i.e. no spilling required.)

**Runtime, same-run SSA-vs-legacy (12-core machine, idle), via
`scripts/ssa_diff.sh`:**

| benchmark | SSA/legacy `≥2`+pruned | SSA/legacy **`≥1`+pruned** |
|---|---|---|
| fib_specialize | 1.13× | **0.89×** |
| btrees_specialize | 0.98× | **0.93×** |
| nbody_specialize | 1.46× | **1.04×** |
| nbody_full | 1.47× | **1.14×** |

Whole-compile static metrics under `≥1`+pruned: **maxlive_gp 10** (was 407),
**bails 0** (was ~90), **edge_moves ≈8** (was 2554).

So under aggressive promotion (`≥1`, which removes the stack-slot memory
traffic) + pruned SSA (which keeps pressure sane), **SSA beats legacy on
fib and btrees and is within 4–14% on nbody** — with zero bails and no
spilling. The `≥2`+pruned regime stays slower because it leaves single-read
values in slots; the win requires *both* aggressive promotion and pruning.

**Still open / not yet proven:**
- **Correctness** of `≥1`+pruned across the full `cargo run -- test resources/`
  is being validated (the benchmarks ran without crashing, a smoke test only).
- nbody's residual 1.04–1.14× — candidate for Fix 2 (better spilling) and/or
  remaining edge moves; needs its own measurement.
- Whether to flip the default `BEAGLE_SSA_PROMOTE_MIN_READS` to 1 (pending the
  correctness run and `/ssa-review`).
- `scripts/ssa_diff_baseline.tsv` is stale (captured when these functions
  bailed to legacy) — the gate's "REGRESSION" lines compare against it, not
  against legacy; it should be re-cut once the above lands.

## 6. References

- Cytron, Ferrante, Rosen, Wegman, Zadeck, *Efficiently Computing Static
  Single Assignment Form and the Control Dependence Graph*, TOPLAS 1991 —
  minimal SSA and the **pruned-SSA** variant (liveness-gated φ placement).
- Braun & Hack, *Register Spilling and Live-Range Splitting for SSA-Form
  Programs*, CC 2009 — the single-pass "MIN" spilling algorithm (Fix 2).
- Hack, *Register Allocation for Programs in SSA Form*, PhD thesis 2007 —
  chordal interference graphs; spill/color decoupling (`maxlive ≤ k ⇒
  k-colorable`).
- Belady, *A study of replacement algorithms for a virtual-storage computer*,
  IBM Systems Journal 1966 — furthest-next-use (MIN) eviction.
- In-repo: `docs/SSA_ARCHITECTURE.md` (invariants), `docs/SSA_RUNTIME_PARITY_PLAN.md`
  (phase plan), `src/cfg/mem2reg.rs`, `src/cfg/regalloc/{spill,color,coalesce,liveness}.rs`.
