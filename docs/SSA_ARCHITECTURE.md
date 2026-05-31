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

## Float unboxing (tier-2 SSA pass)

Floats are heap-boxed: each `f64` lives in a 1-word heap object behind a
tagged pointer. The feedback-specialized float fast path therefore lowers
`a + b` to ~12 ops — `GuardFloat ×2`, `InlineBumpAllocate` (a GC
safepoint), header store, `Untag ×2`, `HeapLoad ×2`, `FmovGpToFp ×2`,
the one real `AddFloat`, `FmovFpToGp`, `HeapStoreOffset`, `Tag`. In a loop
every intermediate allocates a fresh box. Measured tax: a float loop runs
~2.5–3× an identical int loop. Unboxing keeps the `f64` in an FP register
across chained float ops and boxes only at escapes.

**Why not a post-hoc SSA cancellation pass.** The boxed pointer is forced
through a **GC root slot** by the time the SSA layer sees it: I9 requires it
to survive the *next* op's allocation safepoint, so the producer's `Tag` and
the consumer's `Untag` are separated by `Move → SlotStore(root) → SlotLoad`.
A def-use peephole that tries to forward `Tag → Untag` therefore fires zero
times — it is fighting the I9 discipline at the wrong altitude. (Confirmed
empirically; see `project_float_unboxing` memory.)

**The approach: defer the boxed lowering so a high-level float op survives
to the SSA layer.** Float arithmetic is lowered to the boxed sequence in
`ir.rs` *during AST→IR emission* — before SSA, before root-slotting. Instead,
emission emits a single deferred `Instruction::FloatBinOp { op, dst, a, b,
feedback_slot, bail_table }`, and the lowering is chosen **per path**:
- **Legacy** expands it to the byte-identical boxed sequence
  (`Ir::expand_float_binops` → `lower_float_binop_boxed`).
- **SSA** lowers it keeping intermediates unboxed in `Fp` registers, boxing
  only at escapes — done on high-level SSA values *before* root-slotting, so
  escape = "a use that needs a tagged value" is answerable by def-use and I9
  only applies to genuine escapes.

This mirrors the SSA-vs-legacy split already in `Ir::compile` (same
instruction stream, two lowerings). Emission stays uniform: `FloatBinOp` is
emitted only when the SSA path will run (`want_high_level_float`), so a
pure-legacy compile is unchanged, and an SSA→legacy bail still works because
legacy can always expand the op.

**Escape = must box.** Box at: heap/struct stores, any `Call` (incl. the
bail helper), `Ret`, control-flow merges with a boxed value, and any consumer
that isn't a float op. Conservative by default; widen as tests confirm.

**Forbidden:** materializing the f64-bits (a `FmovFpToGp` result) in a GP
register live across a safepoint — those bits are not a pointer; root-slotting
them (I9) makes the GC trace garbage. Unboxed values stay `Fp`-class
(GC-exempt); spill them to **unscanned FP slots** when live across a safepoint.

**Soundness — only unbox *definitely*-float values (no guard needed).** A
`FloatBinOp` is float-specialized *speculation*: given non-float operands at
runtime it bails to the polymorphic helper, whose result may be a non-float
(e.g. a float-specialized `+` on two ints returns an int). So its result is
definitely-float **only if both operands are definitely-float**. The only
unconditional float witnesses are float-tagged constants; args, `HeapLoad`
(struct fields), and call results are treated as non-float. `analyze_float_types`
(`src/float_repr.rs`) computes the definitely-float locals/registers as an
optimistic fixpoint (loop accumulators start float, removed when a store is
shown non-float). A `FloatBinOp` with definitely-float operands is lowered
unboxed and **guard-free**; otherwise it keeps the guarded boxed lowering.
Definitely-float locals become unscanned FP slots; a value crossing the
boundary between representations is coerced (`coerce_to_fp` / `coerce_to_tagged`).

**Stages (each gated, each tested green before the next):**
1. **Deferred-lowering substrate (done).** `FloatBinOp` + per-path expansion;
   legacy byte-identical, SSA expands too (behaviour-neutral).
2. **Sound unboxed lowering (done).** `unbox_floats` (`src/ir.rs`): definitely-
   float locals → unscanned FP slots; definitely-float `FloatBinOp`s → unboxed
   guard-free `AddFloat`/…; box at escapes. Measured: series −70%, boxing probe
   −75%, fib unchanged, suite 365/365. Pure-float loops (const/float-local fed)
   win big.
3. **Speculative unboxing (next — prize CONFIRMED).** Struct-field-fed float
   ops (nbody: floats come from `HeapLoad`, so soundly stay boxed today, −2%)
   need a *guarded* unbox: guard the loaded value is a float, unbox on the fast
   path, fall back to the existing boxed lowering on the miss (per-op guarded
   branch — NOT OSR). Unlocks struct/array-heavy float code.
   **Measured prize (`resources/bench_field_unbox_probe.bg`, tier-2): 2.44×** —
   identical float arithmetic (bit-identical results) costs ~46 ns/iter through
   boxed struct fields vs ~19 ns/iter through unboxed float locals, i.e. field
   boxing is ~59% of that kernel. Real nbody recovers less (it also has sqrt /
   array gets / loop control) — est. ~25% ceiling — but it is the largest
   remaining tier-2 win and the opportunity is real (contrast field-CSE, probed
   to 0). Design: in `analyze_float_types`/`unbox_floats` (`float_repr.rs` /
   `ir.rs`), a `FloatBinOp` operand sourced from a field `HeapLoad` becomes a
   *speculatively*-float value: emit `GuardFloat` on the boxed ptr, unbox to an
   FP register, mark the operand definitely-float so the op and its downstream
   chain unbox; the final field write boxes once. Bail = current guarded boxed
   lowering. Gate behind a sub-flag; degrade to boxed on any unhandled shape.
   Build in the smallest slices (single field-read → single op first), each
   gated + harness-green + bit-identical, A/B on the probe.

   **CORRECTION (iter 9) — per-op guarded unbox is UNSOUND for chains; the win
   needs region/expression versioning.** Studied the code: a float-specialized
   op's bail path assigns the *polymorphic* result to `result_register`
   (ir.rs `lower_float_binop_boxed` slow path) and that result may be a
   non-float (int+int→int, etc.) — `float_repr.rs` already codifies "result is
   definitely-float ONLY if both operands are." So at each per-op fast/slow
   MERGE the value is tagged-but-maybe-non-float: you cannot forward an FP
   value past it nor mark the result definitely-float. Per-op guard+unbox
   therefore does NOT compose across a chain (`dx = bi.x-bj.x; dx*dx; …`) — the
   representation diverges at every bail merge, and a single field op alone has
   no chain to win (its result must be boxed to store anyway). The 2.44× prize
   comes from keeping a whole expression/loop-body unboxed, which requires
   **versioning**: guard all the field-read LEAVES once at region entry; the
   fast version runs the region with NO per-op bails (every value
   definitely-float by the entry guards) keeping intermediates in FP regs and
   boxing only at escapes (field writes); the slow version is today's boxed
   lowering; a guard miss at entry selects slow. Cleanest granularity = a float
   EXPRESSION TREE at AST emission (the tree is explicit in the recursion — the
   "emission-time type inference" path), bailing the whole expression to the
   current boxed emission on any leaf-guard miss (original tagged operands are
   still live at the bail point). Smallest MEASURABLE slice is thus a
   multi-op field-fed expression (e.g. the probe's `dx*dx + c`), not a single
   op. Build expression-tree versioning at emission, gated, A/B on the probe.

   **REFINED (iter 11) — dumped the real tier-2 IR; the region routes through
   LOCALS, so the sound build is full LOOP-BODY VERSIONING, not expression-
   local.** The speculative float region is not a contiguous register chain:
   intermediates round-trip through locals (`dx`→slot, `dx*dx`→slot,
   `mag`→slot) and interleave with the field-read IC idioms (guard + heap loads
   + slow-path property-access call). So the register-chain stat undercounts,
   and the intermediate locals would need DUAL representation (FP in the fast
   version, boxed in the slow). That means versioning the whole loop body:
   guard the field leaves at loop entry, duplicate the body (fast =
   float-locals-as-FP-slots + guarded field reads; slow = current boxed
   lowering), merge. This is the large/high-risk build flagged for deliberate
   scoping (not autonomous 60s grinding). Also: the 2.44× probe removed field
   reads entirely (`step_locals` has none), so versioning that KEEPS the field
   reads wins somewhat less — measure a tighter upper bound (boxing cost vs
   field-read IC cost) before committing to the build.

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
  OK/BAIL outcome (and `PANIC … falling back to legacy` for caught panics);
  `BEAGLE_SSA_ONLY=<substr>` / `BEAGLE_SSA_DENY=<substr>` restrict the SSA
  path to / from functions whose name matches.
- `BEAGLE_SSA_TIER2=1` — rollout phase 9 (tier-up SSA path). Routes *only*
  the hot tier-up / feedback recompiles through SSA, leaving cold
  first-compiles on legacy; orthogonal to `BEAGLE_USE_SSA`. Implemented by a
  thread-local `TierUpCompileGuard` (`src/ir.rs`) entered around
  `compile_ast_with_feedback` in `specialize_function` (`src/compiler.rs`);
  `Ir::compile` enables SSA when `BEAGLE_USE_SSA` is set globally **or**
  `BEAGLE_SSA_TIER2` is set *and* the current compile is a tier-up. Because
  these recompiles run on the shared compiler thread, the SSA attempt is
  wrapped in `catch_unwind`: a panic is caught, the function degrades to
  legacy, and the compiler thread survives. Correctness is covered by
  `resources/ssa_tier2_parity_test.bg` (run it under the flag).
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

### Stage 3 feasibility — explored at BOTH layers (iter 18); it's a multi-session refactor

The field-fed float win (loop-body versioning) was explored at both viable layers
and both confirm a large architectural build — no sound autonomous-increment slice:

- **Flat-IR layer (unbox_floats, ir.rs):** no loop structure visible; the
  speculative region routes through loop-local slots interleaved with field-read
  IC idioms (non-contiguous). Can't cleanly identify or duplicate a loop body.
- **CFG layer (src/cfg/):** HAS the right structure (SSA blocks with params/
  terminator/predecessors; dominators in dom.rs; RPO) — but `FloatBinOp` is
  PRE-LOWERED by unbox_floats before CFG construction, so the CFG only sees
  `AddFloat`/`SubFloat`/... (unboxed, definitely-float) or the expanded boxed
  sequence (`Tag`/`Untag`/`HeapLoad`/`InlineBumpAllocate`/`GuardFloat`) for
  speculative ops. The float region is thus an unidentifiable low-level mess at
  the CFG layer. Also: no reusable natural-loop utility exists yet (regalloc
  does loop extension implicitly via dominators+RPO).

**Therefore the sound build is a refactor**, roughly:
1. Make `FloatBinOp` (and field read/write markers) SURVIVE into the CFG: add
   them to the CFG `Op` enum; cfg/builder.rs translates the IR `FloatBinOp` to a
   CFG `FloatBinOp` instead of letting unbox_floats pre-lower it (gate to
   tier-2/flag so legacy path unaffected).
2. Add a reusable `natural_loops(cfg)` (dominators + back-edges).
3. A CFG pass: for a loop whose body has speculative field-fed `FloatBinOp`s,
   clone the body into fast (leaf `GuardFloat` -> exit-to-slow; intermediates in
   unscanned FP slots; box at field-write escapes) + slow (current boxed
   lowering); guard at loop entry / per-iteration selects; merge at loop exit.
4. Lower surviving `FloatBinOp`s per version, then mem2reg/regalloc/emit.

This is a deliberate multi-session build needing oversight, not 60s autonomous
grinding. Pure-float code already wins ~3.5x via SSA-default; this is the
narrower struct-field-fed case (nbody-shape).

### Stage 3 — architecture pivot (iter 19): do loop-body versioning at the FLAT-IR level, not CFG

Step-2 study finding: cfg/builder.rs builds blocks by LEADER computation over
the already-label-delimited linear IR (labels + post-terminator positions) — it
maps existing structure into blocks; it does NOT create new blocks from a single
op. So making `FloatBinOp` survive into the CFG and lower (boxed) THERE would
require cfg/builder to expand one op into a guard/fast/slow/merge multi-block
structure — a large architectural change to the builder. Rejected.

BETTER PATH (pivot): do the loop-body versioning at the FLAT-IR level, in/around
`unbox_floats` (ir.rs), BEFORE the boxed expansion:
- The flat IR DOES have loop structure (labels + backward Jump/JumpIf = back-
  edges) — the iter-18 "no loop structure" claim was wrong.
- At this point `FloatBinOp`s are still HIGH-LEVEL (not yet expanded), so each
  version lowers them cleanly (fast = emit_float_arith unboxed; slow =
  lower_float_binop_boxed) — no fragile boxed-sequence pattern-matching.
- A loop body is a CONTIGUOUS instruction range (header label .. back-edge
  jump), so cloning it = copy the range with fresh label ids + renamed temp
  registers. Non-contiguity was never the real blocker for *whole-body* cloning.
- Soundness for the bail: GUARD the field-read leaves at iteration TOP, before
  any field-WRITE side effects, so a guard miss bails to the slow body with no
  double side effects (nbody reads all fields before writing any).
Plan: (a) flat-IR loop detection (back-edge = jump to an earlier label that
dominates); (b) identify a versionable loop (body has speculative field-fed
FloatBinOps, all field writes after all field reads); (c) clone body → fast
(leaf guards -> slow; unboxed FloatBinOps; FP slots for float temps) + slow
(current); entry branch selects; (d) verify bit-identical + measure.
NOTE: the CFG `natural_loops` util (commit 271ee86) stays as reusable infra
(LICM/unrolling), but this flat-IR path doesn't depend on it.

### Stage 3 — premise VALIDATED on real nbody, soundness model revised (iter 20)

Wired flat_ir_loops + a heap read/write probe into unbox_floats (gated
BEAGLE_SSA_GUARDED_FLOAT, behaviour-neutral). Run on real tier-2 nbody/advance:
- flat_ir_loops correctly finds the (nested) loops: outer body=(23,1595),
  inners (47,1161)/(161,1139)/(1166,1573); the innermost has 6 speculative
  FloatBinOps, the bigger ones 28-34. So flat-IR loop detection WORKS on real
  code — premise confirmed.
- BUT `writes_after_reads=false` on ALL of them: first_write(591) < last_read
  (1133+). Field writes INTERLEAVE with reads (`bj.vx = bj.vx + ...` reads
  bj.vx after bi.vx was already written). So the simple bail model — "guard all
  field reads at body top, bail before any write" — is UNSOUND here: a late
  guard-miss bail would double the earlier writes.

REVISED soundness model for the versioning codegen: the FAST version must HOIST
every guarded field read above every field WRITE in the body (read all inputs
first, guard, then compute+write), AND this is only sound if no hoisted read
reads a location written earlier in the same iteration — i.e. needs alias
reasoning (two `object.field` accesses alias iff same object + same field). For
nbody each velocity field is read-before-its-single-write per j-iteration, so
hoisting is sound there, but the CHECK must prove it (conservatively: bail
unless every speculative-float-feeding read's (object,field) is never written
in the body, OR the read provably precedes all writes to that location). This
is the real remaining difficulty — an alias-aware hoist-safety check — not just
loop detection. The probe is the tool to validate any candidate check on nbody.

### Stage 3 — loop-body versioning RULED OUT for field-fed floats (iter 21, decisive)

Built `ir_loops::versionable_float_loop` (conservative, tested) and ran it on
real tier-2 IR: EVERY field-fed float loop in nbody (4) AND the probe (1) is
`WriteBeforeRead` — NOT soundly versionable. Root cause is a genuine
read-after-write FIELD dependency, not conservatism: `p.x = p.x + dt*p.vx`
reads the just-written `p.vx`; nbody adds `bi`/`bj` aliasing. Sound versioning
needs to hoist guarded field reads above all writes, but these reads MUST see
post-write values, so hoisting is semantically wrong. Loop-body versioning
therefore cannot deliver the field-fed float win.

WHY the 2.44x probe still measured a win: it kept the values in unboxed LOCALS,
where the read-after-write flows through registers (no field round-trip). Moving
them to struct FIELDS introduces the dependency that blocks unboxing-via-
versioning.

CORRECT remaining path = UNBOXED FIELD STORAGE (struct-layout specialization):
a provably-/guarded-float struct field stores raw f64 inline (not a boxed
pointer), so reads/writes are direct f64 with no allocation and the read-after-
write dependency is just a normal memory dep — no hoisting/versioning needed.
This is a different, large build (struct layout + GC must not scan the f64 field
+ all access sites + guard/box on non-float write). Not started. Loop-body
versioning (natural_loops 271ee86, flat_ir_loops 2d1deb5, versionable_float_loop)
is sound reusable analysis but is NOT the path to the field-fed float win.
