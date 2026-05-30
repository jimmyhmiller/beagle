# SSA Architecture (Beagle)

## Why this document exists

Beagle has attempted SSA multiple times. Every attempt got stuck on the same
families of bug — orphan blocks, branch-arg clobber, FP-class loss,
loop-carried liveness, force-spill across calls, missing TCO, dead-coded
coalescer. The branches `ssa`, `ssa-attempt1`, and `ssa-take-1-million`
each picked up where the previous one died.

The root cause was always architectural, never coincidental: SSA was bolted
onto a linear IR that wasn't shaped for it, and symptoms were patched rather
than the shape being fixed.

This document is the contract for the next attempt. Any change that touches
`src/cfg/`, `src/ssa/`, `src/register_allocation/`, or the regalloc/spill
parts of the backends must comply. The `/ssa-review` skill enforces it.

If you have a strong reason to break an invariant, change this document
**first** in the same change, with the rationale. A diff that contradicts
this file is a blocking review failure.

---

## The invariants

Each invariant has a verifier. A verifier failure aborts the compile under
`debug_assertions` and under `BEAGLE_SSA_VERIFY=1`. In release builds it
logs and skips SSA codegen for that function (falls back to legacy IR).

### I1 — A function is a CFG, not an instruction list

- Module: `src/cfg/`.
- The CFG type is `Block { params, body, terminator, predecessors }`.
- Every block ends in exactly one terminator:
  `Jump | Branch | Ret | Throw | Unreachable`. No fall-through.
- **Forbidden:** `Label`-style marker instructions in block bodies. There
  is no `Instruction::Label` variant in the SSA / CFG layer. (This was the
  orphan-block bug.)

### I2 — No critical edges

- An edge is critical when its source has >1 successors AND its target has
  >1 predecessors.
- `split_critical_edges()` runs immediately after CFG construction and
  again after any pass that may create new edges.
- Verifier asserts no critical edge survives at any phase boundary.

### I3 — Block parameters, never classical phi nodes

- Cranelift / MLIR / dynir style. Control transfer carries the args:
  `Jump(BlockId, Vec<VReg>)`, `Branch(..., args_t, ..., args_f)`.
- Reads happen via `block_param(b, i)`.
- **Forbidden:** a `Phi(_)` op or any `insert_phi` / `phi_node` helper in
  the SSA layer.

### I4 — Every VReg has a register class

- `enum RegClass { Gp, Fp }` stored on `VReg`.
- Spill / reload / move picks slot width and instruction from the class,
  not from the surrounding op.
- **Forbidden:** any `to_breg` / `to_register` helper that drops the class
  during lowering. (This was the FP-spill bug — `STR Xn` emitted for an f64
  silently corrupted the value.)

### I5 — Every use is dominated by its def

- Verifier: Cooper/Harvey/Kennedy iterative idom; for each use, the def's
  block must dominate the use's block.
- Intra-block: in the same block, def position < use position.

### I6 — Locals stay in stack slots; mem2reg promotes selectively

- CFG construction does **not** promote `Value::Local(n)` to an SSA value.
  Every load/store of a Local remains an explicit `SlotLoad` / `SlotStore`.
- Mem2reg is a separate pass with a profitability gate. Slots with <2 reads
  or that escape are not promoted.
- This is the lesson from the prior `ssa` branch: surviving non-trivial
  merges in `nbody/advance` went 3060 → 117 from this single change.
- **Forbidden:** any "promote-at-construction" path (Braun-style on-the-fly
  SSA construction with implicit local promotion).

### I7 — Calls carry an explicit clobber set

- `Call { ..., clobbers: ClobberSet }`. The allocator decides per live
  value how to survive the call:
  - move to a callee-saved register, or
  - spill to a root slot around the call (existing root-slot machinery
    handles GC), or
  - rematerialize at the reload point.
- **Forbidden:** force-spilling every cross-call value for its entire
  interval. (Tier-2 regressed +19.9% from this; the fix is per-call
  decisions, not per-interval.)

### I8 — Tail self-calls become `Jump(entry, args)` at CFG construction

- Detected in Phase 1, not later.
- A tail-position self-call lowers to `Jump(entry_block, args)` directly.
- Without this, 1M-iter tail-recursive benchmarks SIGBUS. The legacy
  AST→IR path does this implicitly via `TailRecurse`; the SSA path must do
  it explicitly.

### I9 — GC sees roots through slots, not registers

- Beagle's GC scans `header.size` frame slots starting at `[FP-24]` per
  the layout in `src/gc/stack_walker.rs`. **It does not scan registers
  or the callee-saved spill area below the slots.** Compacting GC
  rewrites in-place at scanned slot addresses; it cannot update a
  value held only in a register.
- Therefore: **any GP-class SSA value live across a GC-safepoint op
  must also live in a slot at that point.** Failure → freed/relocated
  pointer in a register → SIGBUS or wrong-output bug.
- GC-safepoint ops in the CFG layer: `Op::Call`, `Op::Recurse`,
  `Op::InlineBumpAllocate`'s `Terminator::InlineBranch`, `Op::Throw`'s
  `Terminator::Throw`, all exception/handler/continuation/effect ops
  (each calls a builtin that may allocate), and `Op::RecordGcSafepoint`.
- The legacy IR satisfies I9 by AST-compiler discipline: every
  evaluated arg is pushed to a local-stack slot via `push_to_stack`
  before the next is evaluated. So the input CFG has no GP VReg live
  across a safepoint — until an optimization violates it.
- **`mem2reg` is the most dangerous violator** because it removes
  `SlotStore` ops, leaving the value only in SSA / registers. The
  promotion gate must therefore also enforce I9: a GP slot is
  promotable only if no `SlotStore→SlotLoad` path crosses a safepoint
  op. FP slots are exempt — FP values aren't tagged heap pointers and
  GC ignores them.
- `src/cfg/gc_safety.rs` is the single source of truth for the
  safepoint set and the per-slot dataflow. Other passes that extend
  SSA-value lifetimes (trivial-param elim, copy coalesce) must
  consult the same module before extending across a safepoint.
- **Forbidden:** any optimization that takes a `SlotStore`-backed GP
  value and removes the store without first checking
  `slot_is_gc_safe_to_promote`. Tier-1 SSA attempts crashed
  `gc_stress_single_thread.bg` because `mem2reg` violated this.
- **Rematerialization exemption.** The regalloc spiller may relieve a
  cross-safepoint GP value by *rematerializing* it (re-emitting its def
  at each use) instead of routing it through a root slot — but **only**
  for defs that produce a non-relocatable value: immediate constants
  (`ConstTaggedInt`/`ConstTrue`/`ConstFalse`/`ConstNull`/`ConstRawValue`)
  and pointers into non-moving regions the GC never relocates or frees
  (`ConstPointer`/`ConstStringPtr`/`ConstKeywordPtr`/`ConstFunctionId`,
  and `GetFramePointer`). This does not violate I9: remat *removes* the
  value's cross-safepoint liveness (it is re-derived after the call), so
  the "must be in a slot while live across a safepoint" clause no longer
  applies. A value defined by anything that loads or computes a live heap
  pointer is **not** rematerializable and must still go through a root
  slot. The rematerializable set lives in `regalloc/spill.rs::remat_op_for`.

### I10 — Slots read by soft-edge (handler) blocks stay materialized

- Handler / resume / abort blocks are reached only through the runtime
  exception / continuation mechanism, never through a terminator. They
  have no normal CFG predecessors (`preds = []`) and are therefore
  absent from the dominator tree the `mem2reg` rename walk follows.
- A pass that drops a `SlotStore` (promotion) but cannot rewrite the
  corresponding `SlotLoad` inside such a block leaves the handler reading
  an uninitialised (null) slot. Real bug: the closure pointer (`arg0`)
  lives in slot 0; promoting it away made a `catch` body that reads ≥2
  distinct closure free vars (`load(untag(arg0), 4+idx)`) segfault on a
  null base (`exception_*thread*`, `repl_resume`).
- **Forbidden:** promoting (or otherwise lifetime-extending in registers)
  any slot that has a `SlotLoad` in a block unreachable from `entry` via
  normal terminator edges. `mem2reg` gates on the `reverse_postorder`
  reachable set; regardless of `RegClass`. This is the dominance-tree
  analogue of I9.

---

## The pipeline

```
LegacyIR  (the linear IR on main today)
  │
  │  Phase 1: CFG-ize          src/cfg/builder.rs
  │           — leader computation skips post-terminator Labels (I1)
  │           — splits critical edges (I2)
  │           — rewrites tail self-calls to Jump-to-entry (I8)
  │           — assigns RegClass to every VReg (I4)
  │           — Value::Local stays as SlotLoad/SlotStore (I6 setup)
  ▼
BlockIR  (CFG + stack slots, NOT SSA yet)
  │
  │  Phase 2: mem2reg          src/ssa/mem2reg.rs
  │           — Cytron-style dominance frontiers
  │           — profitability gate: skip slots with <2 reads
  ▼
SSA  (block params, classed VRegs)  ← verifier: I3, I4, I5
  │
  │  Phase 3: SSA opts         src/ssa/opt.rs
  │           — copy coalesce pre-pass
  │           — constant rematerialization annotation (sets flag, no emit)
  │           — DCE
  │           — trivial-block-param elimination
  ▼
SSA (clean)
  │
  │  Phase 4: regalloc         src/register_allocation/ssa_linear_scan.rs
  │           — CFG-aware live intervals (RPO + natural-loop extension)
  │           — RegClass-respecting pools (I4)
  │           — live-range splitting at low-pressure points
  │           — rematerialization at use sites
  │           — coalesce hints on block-param incoming values
  │           — per-call clobber model (I7)
  ▼
Allocated SSA  (still SSA; assignments are colors, not destructive moves)
  │
  │  Phase 5: edge resolution   src/ssa/edge_resolve.rs
  │           — parallel-copy implementation of block-param transfers
  │           — safe because critical edges were split (I2)
  ▼
Linear allocated IR → existing backend (arm64 / x86_64)
```

Each `▼` is a verifier checkpoint. The relevant invariants are asserted at
that point.

---

## Anti-spill checklist (Phase 4)

If a benchmark regresses on spill count, one of these is missing. This list
is the canonical diagnostic order.

- [ ] CFG-aware live intervals (reverse-postorder + natural-loop extension)
- [ ] RegClass-aware spill/reload (I4)
- [ ] Live-range splitting at low-pressure points (Wimmer-style)
- [ ] Rematerialization of constants and address loads
- [ ] Coalesce hints on block-param incoming values
- [ ] Per-call clobber model (I7)
- [ ] Mem2reg profitability gate (I6) — don't promote slots that won't pay

If the box is checked and the regression persists, file a memory note with
the specific pattern that broke it before adding more machinery.

---

## Forbidden patterns (the prior-attempts hit list)

The critic (`/ssa-review`) flags any reintroduction. Each row has a
historical bug.

| #  | Pattern                                                                                       | Why forbidden                                          |
|----|-----------------------------------------------------------------------------------------------|--------------------------------------------------------|
| F1 | `Label`-style marker as an instruction inside a block body                                    | Reintroduces orphan blocks (I1)                        |
| F2 | A dead-coded coalescer: `let _ = build_coalesce_groups;` or `#[allow(dead_code)]` on coalesce | Prior attempts shipped this; spills exploded           |
| F3 | Materializing a constant in the same block as the branch that uses its dest's siblings        | Const-before-branch clobber; emit from a stub block    |
| F4 | A `to_breg` / `to_register` helper that strips RegClass                                       | The FP-spill bug (I4)                                  |
| F5 | Blanket dataflow liveness extension over the legacy linear IR                                 | Collides with InlineBumpAllocate physical-reg reuse    |
| F6 | Force-spill cross-call values for their entire interval                                       | Tier-2 -19.9% regression came from this                |
| F7 | TODO / `-1` / `null` stub returns instead of a hard error                                     | Global CLAUDE.md rule; stubs must throw clearly        |
| F8 | Disabling a verifier check "just for this test"                                               | Verifiers are the contract; fix the bug instead        |
| F9 | AST→SSA / IR→SSA lowering without tail-call rewriting                                         | 1M-iter benchmarks SIGBUS (I8)                         |
| F10| `Phi` op or `insert_phi` helper in the SSA IR                                                 | Violates I3                                            |
| F11| A backend emit helper that writes `dest` before reading all sources, when `dest` may equal a source | SSA coalesces dest with a dead source; the legacy allocator never did. Broke `tag_value` (dest==tag → untagged float), `modulo` (dest==b → `7%2==-2`), `guard_integer/float` (dest==value → value zeroed → `match`/`read_struct_id`), register shifts (dest==a → result re-shifted 8×). Use a reserved scratch (X16/X17) or a single read-all-then-write instruction; or skip the dead-operand restore. |
| F12| Promoting / register-lifetime-extending a slot read by a soft-edge (handler) block            | Violates I10 — handler reads a null slot              |

---

## Rollout phases — each gated on green tests + benchmark parity

1. **CFG only.** Lower CFG back to legacy linear before regalloc/emit.
2. **CFG + critical-edge split + tail-call rewrite** (I2, I8).
3. **CFG + mem2reg + SSA verifier** (I3, I5, I6); still lower to legacy for
   emit.
4. **SSA-aware regalloc (basic, no remat, no splitting).** Spills may be
   high; that's expected for this step.
5. **Coalesce hints.**
6. **Rematerialization.**
7. **Live-range splitting.**
8. **Per-call clobber model** (I7).
9. **Tier-up SSA path.** Treat as a separate rollout once tier-1 SSA is at
   parity.

**Promotion bar at each step:**
- All `cargo run -- test resources/` pass (100%, not "most").
- Spill count per function ≤ legacy baseline ± small tolerance.
- Benchmark suite runtime ≤ legacy baseline.
- The relevant verifier is on and passing.

Skipping ahead of the rollout order is a blocking review failure unless the
prerequisite phase is demonstrably green at the time of the skip.

---

## What is allowed to change without going through this checklist

- The existing legacy IR path (until SSA is the default and legacy is
  deleted).
- Backend code emitting instructions that already exist in the linear
  allocated IR.
- Optimizations that operate purely within the legacy linear IR.
- Anything outside `src/cfg/`, `src/ssa/`, `src/register_allocation/`,
  `src/lir/`, `src/ir.rs`, `src/compiler.rs`, and `src/backend/*/regalloc*`.

---

## Diagnostics

- `BEAGLE_SSA_VERIFY=1` — run all verifiers; abort on failure with function
  name.
- `BEAGLE_SSA_STATS_ALL=1` — per-function spill count, surviving non-trivial
  block-param count, worst merge.
- `BEAGLE_SSA_DOT=/tmp/ssa` — write graphviz dumps per function for
  `dot -Tpng`.
- `BEAGLE_USE_LEGACY=1` — force the legacy linear-IR path, for differential
  comparison.
- `BEAGLE_USE_SSA=1` — opt into the SSA pipeline (falls back to legacy per
  function on any bail). `BEAGLE_SSA_LOG_BAIL=1` logs each function's
  OK/BAIL outcome; `BEAGLE_SSA_ONLY=<substr>` / `BEAGLE_SSA_DENY=<substr>`
  restrict the SSA path to / from functions whose name matches.
- Spiller knobs (`regalloc/spill.rs`): `BEAGLE_SSA_NO_REMAT=1` disables
  rematerialization (forces slot spills — A/B measurement);
  `BEAGLE_SSA_SPILL_CAP=<n>` caps real (slot) spills before bailing (0 =
  uncapped); `BEAGLE_SSA_SPILL_MAX_ROUNDS=<n>` bounds color→spill rounds;
  `BEAGLE_SSA_BLOCK_PARAM_SPILL=1` enables phi/block-param spilling through
  memory (off by default — a net loss until remat + coalescing shrink the
  spilled code); `BEAGLE_SSA_BAIL_DIAG=1` prints the unfixable pressure
  point behind each bail.
- Coalescing: `BEAGLE_SSA_NO_REGCOALESCE=1` reverts the regalloc coalescer
  (`regalloc/coalesce.rs`) to the old pairwise block-param↔arg hints, to
  isolate Phase 5's effect in A/B. (Distinct from `BEAGLE_SSA_NO_COALESCE`,
  which disables the *SSA-level* `Op::Move`→rename pass in `cfg/opt.rs`.)

---

## Diff harness (parity test)

Before promoting a phase past its rollout step, run the diff harness:

1. Compile every `resources/*.bg` test through both pipelines.
2. Compare:
   - Test outcome (must match exactly).
   - Spill count per function (must not exceed legacy + tolerance).
   - Runtime on the benchmark set (must not regress vs legacy).
3. Any mismatch is a blocker until either the SSA path is fixed or the
   legacy path is shown to be the bug.
