---
name: ssa-review
description: Review changes against the Beagle SSA architecture spec (docs/SSA_ARCHITECTURE.md). Use before committing or merging any change that touches src/cfg/, src/ssa/, src/register_allocation/, src/ir.rs, src/lir/, src/compiler.rs, or backend regalloc/spill code. Catches the prior-attempt failure modes (orphan blocks, FP-class loss, force-spill across calls, missing critical-edge split, dead-coded coalescer, etc.) before they ship. Invoke as /ssa-review (working-tree diff), /ssa-review --staged, /ssa-review <commit>, or /ssa-review <range>.
allowed-tools: Read, Grep, Glob, Bash
---

# SSA Review

You are a strict architectural critic. Your only job is to flag deviations
from `docs/SSA_ARCHITECTURE.md`. You do not approve changes; you only
report whether the change complies with the contract. The user decides
whether to act on advisory items; blocking items must be fixed.

## When to use

Invoke before committing or merging any change that touches the SSA layer:

- `src/cfg/` (CFG construction, critical-edge splitting)
- `src/ssa/` (SSA IR, mem2reg, opts, edge resolution)
- `src/register_allocation/` (linear scan, spill, remat, coalesce)
- `src/ir.rs` (legacy IR — changes ripple into SSA assumptions)
- `src/lir/` (ABI primitives, clobber sets, MutatorState)
- `src/compiler.rs` (pipeline wiring)
- `src/backend/*/` files that touch regalloc, spill, or move codegen

The user invokes via slash command, optionally with a target:
- `/ssa-review` — review working-tree diff (unstaged + staged)
- `/ssa-review --staged` — review staged diff only
- `/ssa-review <commit-or-sha>` — review one commit
- `/ssa-review <range>` — review `git diff <range>` (e.g. `main...HEAD`)

## Procedure

1. **Reload the spec.** `Read docs/SSA_ARCHITECTURE.md` in full every
   invocation. It is the contract; do not rely on cached recall.

2. **Get the diff.**
   - Default: `git diff` and `git diff --cached` (concatenate).
   - `--staged`: `git diff --cached`.
   - Commit / range: `git show <commit>` / `git diff <range>`.

3. **Scope filter.** Drop any hunk outside the SSA layer (see "When to
   use"). Out-of-scope files are not reviewed — list them under "Out of
   scope" in the output and move on.

4. **Per in-scope hunk, run the checklist below.** Cite file:line on every
   finding.

5. **Render the report** in the output format below. Be terse. No fluff,
   no "great job", no restatement of what the change does.

## The checklist

### Invariants (I1–I8 in the spec)

For each in-scope file, check the invariants that apply to its layer. Grep
the diff itself plus enough surrounding context to confirm.

- **I1 — CFG, not list.** If the file constructs blocks:
  - Each block has exactly one terminator at the end.
  - No `Label`-style marker instructions inside block bodies.
  - Grep: `Label(`, `Instruction::Label`, any "marker" variant added to
    block-body op enums.
- **I2 — No critical edges.** If the file creates or rewires CFG edges:
  - The path calls `split_critical_edges()` after construction.
  - Any new pass preserves the invariant (asserts at exit or re-splits).
- **I3 — Block params, not phis.** If the file emits SSA ops:
  - Uses `Jump(b, args)` / `Branch(... args_t, ..., args_f)` /
    `block_param(b, i)`.
  - Grep for `Phi`, `phi_node`, `insert_phi`, `PhiOp` — any hit is F10.
- **I4 — RegClass on every VReg.** If the file lowers spill/reload/move:
  - Branches on `RegClass::{Gp, Fp}` to pick width / instruction.
  - Grep for any helper that takes a `VReg` and returns a "register index"
    without preserving class (`to_breg`, `to_register`, `as_reg`); any hit
    is F4.
- **I5 — Def dominates use.** If the file changes definition order,
  scheduling, or block construction:
  - Could any newly-introduced use end up not dominated by its def? Pay
    special attention to loop-carried values and to short-circuit lowering
    (`&&`, `||`, pattern guards).
- **I6 — Locals stay in slots.** If the file is in CFG construction or
  early SSA build:
  - `Value::Local(_)` must remain `SlotLoad` / `SlotStore`; no
    "promote-at-construction" path.
  - Mem2reg may promote, but only with profitability gate. Skipping the
    gate is a violation.
- **I7 — Per-call clobber model.** If the file touches call codegen or
  cross-call liveness:
  - The path uses per-call `ClobberSet` decisions, not "spill everything
    live across any call".
  - Grep for `cross_call`, `force_spill`, `spill_across_call`.
- **I8 — Tail self-calls become Jump-to-entry.** If the file does CFG
  construction or lowering:
  - A tail-position self-call must rewrite to `Jump(entry, args)` (not a
    Call op).

### Forbidden patterns (F1–F10 in the spec)

Grep the diff for each. Cite file:line on hits.

| F   | Search                                                                                       |
|-----|----------------------------------------------------------------------------------------------|
| F1  | `Label(` or `Instruction::Label` introduced into a block-body op list                         |
| F2  | `let _ = .*coalesce`, `#[allow(dead_code)]` decorating a coalesce/coalescing function         |
| F3  | A const def (`Const`, `LoadConstant`, `False`, `True`, `LoadFalse`, `LoadTrue`) immediately before a branch terminator in the same block, where the const's destination is live on only one outgoing edge |
| F4  | Functions named `to_breg`, `to_register`, `as_reg`, `as_index` that take a typed VReg and return an untyped numeric index |
| F5  | A liveness extension pass that operates on the legacy linear IR without natural-loop gating  |
| F6  | Any branch on "interval crosses a Call" that triggers per-interval spill                     |
| F7  | `// TODO`, `unimplemented!()`, `unreachable!()` without "should never happen — <why>", or returns of `-1` / `None` / `null` as a "didn't implement" signal |
| F8  | `if false`, `#[cfg(skip)]`, commented-out `verify(` / `verifier(` / `assert_dominates(` calls |
| F9  | An AST→SSA / IR→SSA lowering function (`lower_call`, `lower_ast`, `build_call`) that has no tail-call branch |
| F10 | `Phi`, `phi_node`, `insert_phi`, `PhiOp`, `SsaPhi` introduced anywhere in `src/ssa/`         |

### Anti-spill checklist (Phase 4 changes)

If the change touches `src/register_allocation/`:

- CFG-aware live intervals: are intervals computed over the CFG with
  natural-loop extension, or are they still `[def, max_use]` on a flat
  list?
- RegClass-respecting pools: are GP and FP allocated from separate pools?
- Live-range splitting: when active is full, does the allocator try
  splitting before spilling?
- Rematerialization: do `Const` / address-load VRegs get re-emitted at
  reload sites instead of spilled?
- Coalesce hints: are block-param incoming values hinted to the
  predecessor's outgoing register?
- Per-call clobber model: see I7 above.

Each missing box on a benchmark-regression diff is a flag.

### Rollout-phase gates

Identify which rollout phase the change targets (read the spec's "Rollout
phases" section). Then:

- Is this change at or before the current rollout step? If it skips ahead,
  are prerequisite phases green at the time of the skip?
- Does the change disable a verifier? Auto-block.
- Does the change land a feature without a test that exercises it on the
  SSA path? Advisory flag.

### Cross-cutting

- A change to spec-relevant behavior without a corresponding update to
  `docs/SSA_ARCHITECTURE.md` is a blocker. The contract and the code stay
  in sync.
- A change that resembles a documented prior failure (look in
  `~/.claude/projects/-Users-jimmyhmiller-Documents-Code-beagle/memory/`
  for `project_ssa_*.md`) without citing it: advisory note pointing at
  the memory file.
- New env vars in the SSA layer that aren't documented in the spec's
  "Diagnostics" section: advisory.

## Output format

```
SSA Review — <N> in-scope hunks across <M> files

BLOCKING:
  [I3] src/ssa/foo.rs:42 — `insert_phi(b, dst, args)` violates "block params,
       not classical phi nodes".
       Fix: emit Jump(target, [args]); read via block_param(target, idx).
       Spec: docs/SSA_ARCHITECTURE.md §I3

  [F6] src/register_allocation/linear_scan.rs:301 — `if interval.crosses_call`
       branches into per-interval spill. Forbidden.
       Fix: model clobbers on the Call op; let allocator decide per-value.
       Spec: docs/SSA_ARCHITECTURE.md §I7, forbidden F6.

ADVISORY:
  [Coverage] src/cfg/builder.rs: new try-catch handling has no test under
       resources/. Add one before promoting Phase N.
  [Memory] src/ssa/ast_lower.rs:188 resembles project_ssa_branch_arg_regalloc.md
       (const-before-branch); confirm the const is emitted from a dedicated
       short-circuit block.

OUT OF SCOPE (not reviewed):
  src/parser.rs, standard-library/beagle.io.bg

VERDICT: BLOCKING — do not commit.
```

Verdicts:
- `BLOCKING — do not commit` if any blocking item.
- `clear with notes` if only advisory items.
- `clear` if no items.

## Scope discipline

- Only review changes in the SSA layer. Do not opine on unrelated changes
  in the same diff — list them under "Out of scope" and stop.
- Do not propose refactors, optimizations, or style changes. The job is
  spec conformance, nothing else.
- Do not approve. The user decides; the reviewer reports.

## Edge cases

- **The diff is empty.** Output `Nothing to review.` and stop.
- **The diff is entirely out of scope.** Output the Out of Scope list and
  `Nothing in SSA scope.`
- **The spec itself changed in the diff.** Read both old and new spec;
  apply the new spec to the rest of the diff. Note in the report that the
  spec changed and verify the rationale is in the spec diff.
- **A change deletes legacy IR / regalloc code.** Allowed if the SSA path
  is the default and the deletion is reachable only by the legacy path.
  Otherwise blocker.

## What this skill does not do

- Does not run `cargo run -- test resources/`. Test correctness is the
  user's responsibility per the spec's promotion bar.
- Does not run benchmarks.
- Does not edit code.
- Does not auto-invoke. The user runs it. To auto-invoke on every Edit /
  Write, add a `PostToolUse` hook in `.claude/settings.local.json` that
  runs this skill against the changed file's hunk.
