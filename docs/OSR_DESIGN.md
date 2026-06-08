# On-Stack Replacement (OSR) — Design

**Status:** design / not yet built. Companion to `docs/PERF_NEXT_WORK.md` (Priority 1.5)
and `docs/SSA_ARCHITECTURE.md` (deopt + state-map notes).

## STATUS: LANDED, OPT-IN (`BEAGLE_OSR=1`, default OFF) — 2026-06-04

The F_osr-vs-warm perf gap is **closed for integer loops** (multi 1.71→0.26s,
sumloop 0.58→0.19s, fannkuch 2.68→1.56s; 370/370 + bit-identical benchmarksgame).
See `docs/OSR_PERF_HANDOFF.md` (top) for the fix: F_osr is compiled with a deopt
context + the int live-ins are guarded at the entry so it reaches warm quality.

**Default-on was tried and REVERTED 2026-06-04** — it hung the beagle-zelda game on
a room-2 render Y-sort that drives `beagle.core/tim-binary-insertion-sort`, whose
SSA-compiled F_osr hits a latent SSA control-flow bug in that function (unrelated to
the deopt work — that function is deopt-ineligible). OSR exposes latent SSA bugs as
hangs in previously-working code, so it stays opt-in until F_osr SSA correctness is
hardened (suggested: a differential harness comparing F_osr vs tier-1 output over
real stdlib functions, especially nested-loop + closure-call loops). Float loops
(spectral/mandelbrot) remain the Phase-D perf follow-on.

Original Phase-3 status below.

## STATUS: LANDED (end-to-end, behind `BEAGLE_OSR`) — 2026-06-03

OSR works end-to-end and is validated: a once-entered hot loop transfers mid-flight
into an optimized continuation and returns the correct result. Evidence:
- **Correct:** `sumloop(100M)` → identical result, **0.63s → 0.34s (1.85×)**; an
  allocating loop (vector/iter, GC pressure) → identical result under OSR.
- **Suite:** full suite **370/370** with `BEAGLE_OSR=1` at default *and* aggressive
  thresholds (transfers exercised across diverse loop shapes); **370/370** default
  (OSR off → zero impact).
- **Benchmarks:** all benchmarksgame ports **bit-identical** with OSR on.

**Pipeline (all landed):** back-edge `OsrCheck` (counter + `osr_trampoline(key,
counter, sp, fp)` + sentinel-guarded conditional return) → trampoline requests an
async build on the compiler thread (`CompilerMessage::BuildOsrVariant` →
`Compiler::build_osr_variant`), re-arms the counter → build feedback-compiles F,
captures its specialized IR (`osr_capture`), deconstructs it
(`crate::osr::build_osr_variant_ir`: strip checks, prepend the live-ins-as-args
entry, jump header, rebuild label positions), compiles `F_osr`, publishes its address
in the OSR registry → a later trip reads live-ins from F's frame and calls `F_osr`,
returning its result from F.

**Live-ins via an OSR BUFFER (any count).** The trampoline packs the loop's live-ins
into a buffer and passes its pointer as `F_osr`'s single argument; `F_osr`'s entry
`HeapLoad`s each `buf[i]` into its slot (the unpack runs before any safepoint, so the
not-yet-rooted buffer can't be moved by GC). This lifts the old arg-register (>8)
limit — validated: a 10-live-in pure-int loop OSRs at **2.16× (1.60→0.74s)**, fannkuch
(12 live-ins) is bit-identical.

**Remaining tuning (not correctness):**
- **Loop selection** — the *first* loop to reach the threshold is OSR'd, and a function
  OSRs at most once per invocation. When that's a minor loop (fannkuch: 1.15×, vs the
  2.16× a dominant loop gives), most work stays in tier-1. Better: instrument all
  back-edges and OSR the hottest, or OSR the outermost loop.
- **Precise live-ins** — "all named locals in scope" over-approximates; precise
  liveness shrinks the buffer and the unpack.
- OSR mostly helps *self-contained* hot loops — when the hot work is in a callee, the
  callee tiers up on its own anyway.

## 0. Why

Tier-up in Beagle is triggered by a **function-entry** counter
(`Instruction::TierUpCheck`, emitted once in the prologue —
`src/ast.rs:1556-1577`, lowered `src/ir.rs:4363`). A function that is *entered
once* but runs a hot inner loop never trips it, so it runs the whole computation
as unoptimized tier-1. This is the single largest systemic perf gap.

**Measured ceiling (the prize).** Forcing tier-up with a warmup +
`runtime/specialize-all()` and re-running (so the loop runs tier-2 from entry):

| bench | tier-1 (never tiers up) | tier-2 (warmed) | win |
|---|---|---|---|
| fannkuch_redux n=10 | 2.67 s | **1.23 s** | **2.17×** |
| fasta n=2.5M | 1.40 s | 0.78 s | 1.80× |
| binary_trees n=16 | 0.32 s | 0.18 s | 1.78× |
| spectral_norm n=2000 | 6.34 s | 5.1 s | 1.24× |

OSR's job: deliver these numbers to the **first** (and only) call, by transferring
the running tier-1 loop into tier-2 code at a hot back-edge.

## 1. How production JITs do it (and the one fact that matters)

The PLDI'18 framing (D'Elia & Demetrescu, *OSR Distilled*): an OSR transition only
needs to reconstruct **live variables** at the target point, via a *compensation
code* fragment that recomputes the target's live values from the source's. The
cost of OSR is governed entirely by how different the two frames' layouts are.

| Engine | back-edge trigger | frame | enter mid-code | baseline→opt mapping |
|---|---|---|---|---|
| HotSpot C1/C2 | `backedge_count` vs threshold | **new**; `malloc` OSR buffer, copy in, free | recompile w/ entry at loop; pre-loop dead | interp oop-maps → buffer; compiler **Phi**s merge buffer-loads w/ normal-entry values |
| V8 Maglev/TurboFan | interrupt budget on `JumpLoop` | **new** | graph w/ `OsrValue` nodes + "OSR deconstruction" removes normal entry | `FrameState` deopt descriptors run inward; Smi/double guards on `OsrValue`s |
| JSC DFG/FTL | exec counter at `op_loop_hint` | **new**; scratch buffer via `prepareOSREntry` | `osrEntryThunkGenerator` thunk → DFG loop block | `OSREntryData` incl. `m_localsForcedDouble`; `Flush` nodes pin operands to slots |
| **V8 Sparkplug** | interrupt budget | **identical layout to Ignition** | resume in equivalent code | **none needed** — "almost zero frame translation overhead" |
| LuaJIT / PyPy (trace) | `hotcount[]` / interp counter | **in-place** | recording *starts* at the back-edge; trace *is* the loop | none for entry; snapshots only for *exit* |

**The load-bearing fact:** V8 deliberately gave Sparkplug the *same frame layout as
the interpreter* specifically so that "any OSR logic that works for the interpreter
will work for Sparkplug… we can swap between the interpreter and Sparkplug code with
almost zero frame translation overhead" (v8.dev/blog/sparkplug). Maglev/TurboFan use
different layouts and therefore need full deopt-data-driven translation.

**Beagle is in the Sparkplug regime — for the part that matters.** This is the
whole reason OSR is tractable here (see §3).

Trace JITs (LuaJIT/PyPy) are the other paradigm: no separate optimized frame —
compilation starts at the back-edge over the live frame. **Rejected for Beagle:** we
are a method JIT with an AST→IR→regalloc pipeline, not a tracing interpreter;
adopting tracing would be a rewrite, not a feature.

## 2. Beagle codegen realities (with citations)

1. **Slot layout — prefix is identical across tiers, total size is not.**
   Named locals (`find_or_insert_local`, `src/ast.rs:5961`) and intermediates
   (`push_to_stack`, `src/ir.rs:4873`) are numbered in **AST source order**,
   feedback-independent. Tier-1 (legacy linear-scan) and tier-2 (SSA) compile from
   the *same AST*, so slots `0..ir.num_locals` — **all named locals + eval-stack
   temps** — sit at **identical FP-relative offsets** in both. The CFG path maps
   `Value::Local(k)` → `SlotId::root(k)` 1:1 (`src/cfg/builder.rs:644`).
   *But* total frame size differs: each tier appends **spill/root slots** above the
   prefix via a *different* allocator (legacy `LinearScan::stack_slot` vs SSA
   `alloc_root_slot`, `src/cfg/mod.rs:803`). **Verdict: rely on identical prefix
   offsets; do NOT assume equal frame size.**

2. **Slots are FP-relative, downward.** Local `k` at `[FP - (k+3)*8]`
   (`src/backend/arm64/mod.rs:503`); header at `[FP-8]` (size = `num_slots`), GC
   prev-ptr at `[FP-16]`. FP-relative addressing means the prefix offsets are valid
   regardless of the differing SP/spill area — **OSR-favorable**.

3. **Tier-1 is slot-resident; tier-2 promotes to registers.** Legacy codegen stores
   every named-local update through to its slot and reloads each use (that's *why*
   it's slow). So at a loop back-edge **the live loop-carried locals are already in
   their slots** in tier-1 — no flush instrumentation needed to read them. Tier-2 is
   the tier that keeps them in registers; the OSR entry must *reload slots →
   registers*.

4. **GC scans all root slots by tag bits — no stack maps.** `header.size` slots
   scanned; any slot whose bits look like a heap pointer is a root
   (`src/gc/stack_walker.rs:97`). Sound non-pointer analysis
   (`src/cfg/pointer_class.rs`) lets ints stay unrooted; **boxed floats are
   maybe-pointer**; raw `f64` lives only in FP/unscanned slots. OSR target frame
   must keep this invariant across the transfer.

5. **Floats: boxed in tier-1, unboxed in tier-2.** A tier-1 float local is a
   *tagged pointer to a heap box* (f64 bits at word offset 1). Tier-2 wants a raw
   `f64` in an FP register. Conversion primitives exist: `coerce_to_fp` (untag →
   load@1 → `fmov` GP→FP, `src/ir.rs:1503`) and `box_fp` (`src/ir.rs:1533`, **note:
   allocates → safepoint**). `apply_float_param_versioning` (`src/ir.rs:1588`) is
   the existing "guard-then-unbox-or-fall-back" template.

6. **The mid-function-entry + frame-transfer primitive ALREADY EXISTS.**
   `load_label_address` (ARM64 `adr`, `src/arm.rs:2326`) materializes an interior
   code address; the hand-written **`return-jump` trampoline** (`src/main.rs:123`)
   restores SP/FP/LR (+ optional callee-saved) and `BR`s to an arbitrary target.
   Continuations already use this to resume in the middle of a frame. **OSR reuses
   this transfer mechanism.** What's missing: a loop-header→OSR-entry address table
   and the back-edge trigger.

7. **No back-edge safepoint / no back-edge trigger today.** Loops lower to a header
   `Label` + a latch `Instruction::Jump(header)` (`src/ast.rs:1884-1947`, codegen
   `src/ir.rs:3778`); the pause check + `TierUpCheck` are **entry-only**. OSR needs
   new latch instrumentation — but `TierUpCheck` (`src/ir.rs:4363`: load counter,
   `subs #1`, branch, `call_builtin trampoline`) is a complete, working template.

8. **Install redirects future *calls* only.** Tier-2 install swaps a jump-table
   slot at a stop-the-world (`modify_jump_table_entry`, `src/runtime.rs:8715`); it
   never touches a *running* activation. That running activation is exactly what OSR
   exists to migrate. Deopt (`apply_deopt_rewrite`) *re-invokes from entry* and does
   **no** frame transfer — it gives us two reusable assets (resident generic stays a
   valid target; identity state-map for the prefix) but not the transfer itself.

## 2b. Foundations VERIFIED (2026-06-03, Phase 2 prep)

A `BEAGLE_OSR_DEBUG` probe dumped, at every loop latch in both tiers, the
named-local→slot map + eval-stack depth (`AstCompiler::osr_debug_dump_latch`):

- **Slot identity — CONFIRMED.** `sumloop(n, base)` dumps *byte-identical* maps in
  tier-1 and tier-2: `n→0, base→1, s→2, i→3, acc→4`. The state map is the identity.
- **Eval-stack empty at the latch — CONFIRMED** for `sumloop` and **every stdlib
  loop exercised**, including match/closure/resumable-exception/continuation-heavy
  ones (`process-eval-request`, `handle-await-first-threaded`, `FlatMapSource_*`).
  Live state at a back-edge = named locals only, nothing transient.
- **Self-protecting guard added:** `maybe_emit_osr_backedge_check` now skips any loop
  whose eval stack is non-empty at the latch (`local_stack_indices()`), so a future
  lowering that breaks this can never silently produce an unrecoverable OSR point.

## 2d. CRITICAL correction: F_osr must come from a FEEDBACK-SPECIALIZED compile

Investigation (2026-06-03) established: tier-2 specialization (int/float type info,
`*_with_bail`, float unboxing) is decided at **AST→IR lowering** using recorded
feedback, and is **fully baked into the IR ops** before the CFG exists. Therefore:

- F's **tier-1** IR is *unspecialized* (polymorphic `*_any`). Cloning it would yield
  slow code — useless.
- F's **tier-2** IR is specialized, but tier-2 is normally triggered by the
  *entry counter*, which **never fires for a once-entered hot-loop function** — the
  exact case OSR exists for.

**Resolution (matches HotSpot/V8 "compile the OSR variant when the loop is hot"):**
`F_osr` is built **lazily, at trampoline time**. When the back-edge counter trips,
the loop's arithmetic feedback has *already been recorded* (the loop ran in tier-1),
so the trampoline drives a `specialize_function`-style **feedback compile of F**, and
during that compile transforms the specialized IR into `F_osr` (strip OSR checks,
prepend the OSR entry, re-root). This is why `F_osr` construction lives on the
specialized-compile path, not at F's first compile.

**Implementation increments (each independently validated):**
- **B** — `OsrCheck` codegen: counter + `osr_trampoline(key, sp, fp)` + sentinel-
  guarded conditional `return`. ✅ **DONE & validated inert.** New legacy-only
  `Instruction::OsrCheck` (`src/ir.rs`) clones the proven `TierUpCheck` counter
  sequence, then passes `sp`/`fp` to `osr_trampoline`, compares the return to
  `OSR_NO_OSR` (`usize::MAX`), and on a real result jumps to the shared `exit`
  (returns F_osr's result); on the sentinel falls through to the back-edge. Safe
  because loop-carried values sit in callee-saved `X19-X27`. Trampoline currently
  always returns the sentinel → behaviour identical to Phase 1: fannkuch output
  correct, **suite 370/370 with `BEAGLE_OSR=1 BEAGLE_OSR_THRESHOLD=1`** (every loop
  exercises the conditional-return path). Stripped from `F_osr` later; gated off
  under `BEAGLE_USE_SSA` (legacy-only codegen).
- **A** — `F_osr` IR construction (`build_osr_variant_ir`) hooked into the specialized
  compile; register as a new function. Validated by direct call.
  - ✅ **Transform DONE & unit-tested** (`src/osr.rs`): clones F's specialized IR,
    strips `OsrCheck`, prepends `StoreLocal(slot_i, arg_i)` per live-in + `Jump(header)`
    so the OSR entry is instruction 0 (CFG reachability DCEs the dead pre-loop).
    `OsrLoopInfo { header_label, live_in_slots }` is the captured per-loop data.
    Remaining for A: the capture hook (record `live_in_slots` = env locals at the
    latch, per-function loop index) + `build_osr_variant` (feedback-compile F →
    capture specialized IR → transform → compile `F_osr` → cache its code address).
- **C** — trampoline reads live-ins from F's frame, calls `F_osr`, returns result.
  End-to-end on a loop whose F tiers up; validate the win.
- **D** — lazy `build_osr_variant` from the trampoline for once-entered loops
  (fannkuch). The headline case; validate fannkuch ~2×.

## 2c. Transfer architecture — REVISED to a normal call (safer than §3)

The original §3 plan (jump the running tier-1 frame into tier-2 via the `return-jump`
trampoline, rewriting the frame header mid-flight) works but is GC-hairy. The cleaner
design that the foundations enable:

**Transfer = an ordinary Beagle call + return.** At a hot back-edge, tier-1 `F` calls
an optimized **loop-continuation** `F_osr(live_ins…)` — the rest of `F` from the loop
header H, with the loop-carried live-ins passed as arguments — and `F` returns that
call's result. Concretely the latch becomes:

```
counter--; if counter != 0 -> back-edge
r = osr_trampoline(key, sp, fp)     # reads live-ins from F's slots, calls F_osr
if r == NO_OSR_SENTINEL -> back-edge   # not ready / ineligible: keep running tier-1
return r                              # OSR happened: F returns the continuation's result
```

Why this is better:
- **GC-safe by construction** — it's a normal call building a fresh frame; no
  header rewrite, no mid-flight frame reinterpretation, no `return-jump` asm. The
  scariest hazard class disappears.
- **Reuses the already-specialized tier-2 code** (F_osr is/uses F's optimized loop).
- **Fallback is still free** — the trampoline returns a sentinel if F_osr isn't
  compiled yet or the loop is ineligible, and tier-1 just keeps looping.

This relies on **Foundation 1** (live-ins are at known, identical slots → the
trampoline reads them from F's frame) and **Foundation 2** (only named locals are
live, so the arg list is exactly live-in(H)).

**The one open implementation fork** (next investigation): how F_osr reuses F's
*specialized* code.
- **(A) Alternate entry into F's tier-2 body** — emit a second prologue that stores
  args→live-in slots and jumps to H. Reuses specialized code directly; needs SSA
  second-entry support (the prologue must feed H's phis).
- **(B) Separate specialized function** — synthesize F_osr's AST (live-ins as params
  + loop + post-loop) and specialize it. No SSA second-entry work, but must thread
  F's loop feedback into F_osr's compile (its arith sites are a suffix of F's).

Decision pending a feasibility probe of the SSA second-entry path (A). A is cleaner
if tractable; B is the fallback.

## 3. (Original plan, superseded by §2c for the transfer) "same-FP frame reinterpretation"

Because the live loop state is **named locals already sitting in prefix slots at
identical FP-relative offsets** (§2.1–2.3), Beagle can do something close to the
Sparkplug regime rather than HotSpot's malloc-buffer regime: **keep the same frame,
reinterpret it with tier-2 code.**

The transfer, at a hot back-edge in the tier-1 frame, with FP unchanged:

1. **Grow the frame** — lower SP to `FP - tier2_frame_size` (tier-2's spill area is
   larger; it lives *below* the untouched prefix). The prefix slots `[FP-24…]` stay
   exactly where they are — **no memcpy**.
2. **Fix the GC header** — write tier-2's `num_slots` into `[FP-8]` so the collector
   scans the right count. (Prev-ptr at `[FP-16]` is unchanged; same frame.)
3. **Enter tier-2's OSR-entry block** (per-loop, see §4) via the `return-jump`
   primitive: same FP, same saved-LR (returns to tier-1's caller on loop exit),
   `BR` to the OSR-entry address. The OSR-entry block **reloads each live-at-header
   local from its slot into the register tier-2 expects**, unboxing floats, then
   falls into the loop body.

Why this is the right shape for Beagle:
- The expensive part of every other engine's OSR — translating baseline-frame slots
  to optimized-frame slots — is the **identity** here (identical prefix offsets +
  same FP). We skip HotSpot's buffer, JSC's thunk+`OSREntryData`, TurboFan's
  deopt-data translation.
- The irreducible work that survives (per the external survey) is exactly four
  things, all bounded and addressed in §4: frame-size growth, register reload,
  float unboxing+guards, and safepoint-map consistency.

## 4. The OSR-entry block (the one new piece of compiled code)

Modeled on TurboFan's `OsrValue` + "OSR deconstruction" and HotSpot's loop-entry
nmethod, but trivial-state-map. When we tier-2-compile a function that has an
OSR-able loop, we additionally emit, per such loop header H, an **OSR-entry block**
`osr_H`:

- It is a *second entry* into the tier-2 body. It does **not** run the prologue or
  any pre-loop code.
- It **loads every value in `live-in(H)` from its prefix slot** into the vreg the
  loop body expects (`SlotLoad`), since at OSR time those values are slot-resident
  (tier-1 wrote them) but tier-2's body wants them in registers.
- For each **float** loop-carried local: `GuardFloat` the slot, then `coerce_to_fp`
  (untag→load@1→fmov). On guard miss → **abort OSR** (see fallback below).
- It **recomputes any loop-invariant values** the body needs that LICM hoisted above
  H (or, simpler for v1: compile the OSR variant with hoisting disabled across the
  OSR entry, so invariants are recomputed once inside `osr_H` — OSR fires once, so
  the cost is irrelevant).
- It then `Jump`s to H (the normal loop header), where tier-2's loop-header φ merges
  the OSR-loaded values with the back-edge values — exactly HotSpot's Phi role.

`live-in(H)` comes from `cfg/loops.rs::natural_loops` + liveness at H. The
loop-header→`osr_H`-address table is recorded at compile time (the address via
`load_label_address`, like continuation resume points).

**Fallback is free.** Tier-1 is fully general. If anything is unfavorable at OSR
time (a float slot doesn't hold a float, the loop isn't OSR-able, tier-2 isn't ready
yet), the OSR trampoline simply **returns without transferring** and tier-1 keeps
running. No mid-loop deopt machinery is needed — the "do nothing" path is always
correct. This makes OSR strictly best-effort and removes the scariest class of bugs.

## 5. Trigger + transfer plumbing

- **Back-edge counter** (`Instruction::OsrCheck`, new, modeled on `TierUpCheck`):
  emit at each OSR-able loop latch, *before* `Jump(header)`. Decrement a per-loop
  counter; on trip, spill live volatiles and `call_builtin(osr_trampoline)` passing
  SP, FP, and a `(function, loop-id)` key. **Caveat vs `TierUpCheck`:** the entry
  check runs where "args already moved to locals, clobbering volatiles is safe"
  (`src/ir.rs:4369`); the latch has live loop state, so the check must be placed
  where regalloc treats the call as a clobber/safepoint (reuse the existing `call`
  infrastructure that already does this for `__pause`).
- **`osr_trampoline` (Rust):** look up/compile the tier-2 OSR variant for
  `(function, loop-id)`; if not ready, kick the compile and return (tier-1
  continues — next time it'll be ready). If ready, run the representation guards in
  Rust over the prefix slots (read via FP); on any miss, return. On success: write
  tier-2 `num_slots` into the header, compute new SP, and tail into the `return-jump`
  primitive with `jump_target = osr_H`, `new_fp = FP`, `new_sp = FP -
  tier2_frame_size`, `new_lr =` tier-1's saved return address.
- **Compile-at-STW reuse:** the existing tier-2 staging/STW-install path
  (`stage_specialization_installs` → `stop_world_and_apply_installs`) compiles the
  OSR variant off the hot thread, same as ordinary tier-up.

## 6. Phased plan (each phase independently validatable, default-off behind a flag)

- **Phase 0 — ceiling confirmed.** Done (§0 table). The warmed `specialize-all`
  numbers *are* the OSR target.
- **Phase 1 — back-edge counter + no-op trampoline** (`BEAGLE_OSR=1`). ✅ **DONE.**
  - **No new IR instruction needed.** `Instruction::TierUpCheck(counter, key_ptr,
    trampoline_ptr)` is already a generic "decrement counter → on zero call
    `trampoline(key_ptr)`" primitive wired through every match site (legacy, SSA/CFG,
    regalloc, pretty-print). We emit it at the loop latch pointing at a new
    `osr_trampoline` (logging-only). Avoids all the codegen/regalloc/CFG surface a
    bespoke `OsrCheck` would have touched. Phase 3 will swap the trampoline body,
    not the instruction.
  - **Latch-call safety — PROVEN.** The linear-scan allocator's pool is
    `X19–X27`, **all callee-saved** (`src/abi/mod.rs:60`). So every loop-carried
    value the allocator keeps in a register is preserved across the `extern "C"`
    trampoline call; the call only clobbers caller-saved scratch, which holds no
    live loop state at the latch. This is the load-bearing fact that makes a
    mid-loop call safe (and will matter again for the Phase 3 transfer).
  - **Emission** (`AstCompiler::maybe_emit_osr_backedge_check`, `src/ast.rs`): gated
    on `BEAGLE_OSR` + first-compile only (`feedback_bits_input.is_empty()` — same
    tier-1 gate as the entry check), skips bail helpers; per-loop counter via
    `add_function_counter("<fn>#osr<id>")`, threshold `BEAGLE_OSR_THRESHOLD`
    (default 100k). Wired at both `Ast::Loop` and `Ast::While` latches; `for`
    desugars to `Loop`.
  - **Validated:** fires correctly (fannkuch: 6 hot loops logged, output 228 /
    Pfannkuchen(7)=16); flag-off is byte-identical (zero OSR lines, no codegen);
    **full suite 370/370 with `BEAGLE_OSR=1 BEAGLE_OSR_THRESHOLD=1`** (check fires on
    every loop). **Overhead ~2.3%** on fannkuch n=10 (load/dec/store/branch per
    back-edge) — paid only in tier-1 before transfer; tier-2 has no check.
    Reduce later if needed (check every Nth iter, or reserved counter register).
  - Trampoline takes only `key_ptr` today; Phase 3 will extend it to pass SP/FP for
    the frame transfer.
- **Phase 2 — OSR-entry compilation, integer loops only.** Emit `osr_H` reload
  blocks; build the loop-header→address table; compile the OSR variant. Still don't
  transfer — instead, unit-test that entering `osr_H` with hand-set slots produces
  correct results.
- **Phase 3 — the transfer, integer loops.** Wire `osr_trampoline` + `return-jump`.
  Target fannkuch / fasta / binary_trees (int/pointer loops, no float reload).
  Expect the §0 wins on the *first* call. Validate bit-identical output + full suite
  + gc-stress.
- **Phase 4 — floats.** Add `GuardFloat`+`coerce_to_fp` in `osr_H`; target
  spectral / mandelbrot-shaped loops. Hardest GC bit: the OSR-entry safepoint map
  must reflect **baseline** pointer-classes for not-yet-reloaded slots and
  **optimized** classes after. Validate against gc-always.
- **Phase 5 — TCO loops.** Tail-self-call loops are `Jump→entry` with block-params
  (`src/cfg/builder.rs:1241`), a different back-edge shape than explicit loops.
  Extend the trigger + entry to that shape. (Deferred: the headline benchmarks are
  explicit `while` loops.)

## 7. Open risks / things to nail before Phase 3

1. **Eval-stack emptiness at the latch.** ✅ **VERIFIED** (§2b) across `sumloop` +
   all stdlib loops; self-protecting guard added in `maybe_emit_osr_backedge_check`.
2. **Prefix-identity.** ✅ **VERIFIED** (§2b): tier-1 and tier-2 produce identical
   named-local→slot maps for the probe function. (For the call-based transfer of §2c
   the trampoline reads live-ins from F's own tier-1 slots, so identity is needed
   only between F's slots and the arg order — still the same source-order map.)
3. **Same-FP frame growth safety.** Lowering SP into the spill region must not clobber
   anything still read from the prefix (it won't — prefix is *above* spills) and must
   respect the 16-byte alignment + red-zone assumptions in
   `patch_prelude_and_epilogue` (`src/arm.rs:2056`). The OSR variant's epilogue must
   restore SP from FP (it already does — frames tear down via FP).
4. **GC during the transfer window.** Between "write new header size" and "registers
   reloaded," a collection must see a consistent frame. Either run the whole
   trampoline transfer with GC excluded (it's O(live-ins), microscopic), or order the
   header-size write so the scanned set is always a valid superset.
5. **LICM-hoisted invariants.** v1: disable cross-OSR-entry hoisting in the OSR
   variant (recompute in `osr_H`). Revisit only if a benchmark shows the recompute
   matters (it runs once).

## 8. Summary

Beagle is unusually well-positioned for OSR: the deterministic, source-ordered,
FP-relative slot prefix makes the baseline→optimized state map the **identity**
(the Sparkplug regime), and the `return-jump` trampoline + `load_label_address`
already provide mid-frame transfer. The new work is bounded: a back-edge counter
(clone of `TierUpCheck`), a per-loop **OSR-entry reload block** (clone the
`OsrValue`/Phi shape, trivial-state-map), float unbox guards (clone
`apply_float_param_versioning`), and a best-effort trampoline whose failure mode is
simply "keep running tier-1." Target: the §0 ceiling (fannkuch 2.17×, fasta 1.8×,
binary_trees 1.78×) delivered to the first call — the largest remaining systemic
perf lever.
