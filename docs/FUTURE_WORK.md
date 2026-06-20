# Beagle — Future Work

A consolidated, honest map of what stands between Beagle today and a
**production-ready dynamic language with full, robust live-coding support**.
This is the master tracker; it links out to the deep-dive design/handoff docs
that already exist (`docs/*.md`). It covers three things:

1. **Known bugs** — open, deferred, or partially-mitigated defects.
2. **Testing gaps** — where coverage is thin, skipped, or non-deterministic.
3. **The road to production** — language, stdlib, runtime, tooling, and
   especially the live-coding story, which is the headline goal.

Scope note: "production ready" here means a language someone could ship a real
program on and a live-coding session you could run for hours against a live
process (e.g. a running game/editor) without it crashing, leaking, or silently
corrupting state. That bar drives the priorities below.

Status legend: 🔴 open / broken · 🟡 mitigated but incomplete · 🟢 done (listed
only where it anchors related future work) · ⏸ deliberately deferred.

---

## 1. Known bugs

### 1.1 Concurrent structural redefinition (live coding's hardest corner) 🟡

This is the single most load-bearing area for live coding and the one with the
most scar tissue. The full history is in
`docs/CONCURRENT_REDEFINE_OPEN_ISSUES.md` and
`docs/GC_REDEFINE_RACE_INVESTIGATION.md`. Nine distinct bugs (#1–#9) were found
and fixed: binding zero-clobber, slow-path over-read, GC layout-migration
stale-root, unregistered-main-thread root scanning, compiler-thread deadlock,
new-field writes, stale allocation sites, the thread+`core/gc()` test-runner
deadlock, and 4-bit `layout_version` wraparound.

**What still needs work:**

- 🔴 **The whole mechanism rests on a forced stop-the-world migration** that the
  *mutator-reachable `eval`* triggers when a struct is structurally redefined
  (`structural_redefinition_needs_gc`). This works, but it's a heavyweight,
  somewhat fragile contract: every compiled allocation site carries a sticky
  "ever redefined" bit and permanently falls back to generic construction once
  set. We need (a) a fuzz/stress harness that cycles many structs through
  grow/shrink/reorder redefinitions under many reader threads and `--gc-always`
  for hours, and (b) a way to retire the sticky-bit penalty (re-specialize the
  new layout once migration settles) so a long live-coding session doesn't
  accumulate permanent slow paths.
- 🟡 **Layout-version width** is still the original 4-bit header field. The
  sticky-bit fallback removes the wraparound *hazard*, but the design would be
  more robust with a wider, non-aliasing version (documented as a known
  trade-off in `CONCURRENT_REDEFINE_OPEN_ISSUES.md` #9).
- 🔴 **`memory.threads.len()` vs `registered_thread_count` can disagree** — a
  latent inconsistency noted during the stale-root investigation
  (`docs/GC_STALE_ROOT_REDEFINE_BUG.md` §9). Tighten this into a single source
  of truth; the GC coordinator's "subtract one if registered" logic is currently
  the kind of special-case that hides bugs.

### 1.2 Protocol / function / let redefinition under churn 🟡

From `project_protocol_redefine_races` and `docs/PROTOCOL_REDEFINE_HANDOFF.md`:
the L1/L2 lock-free dispatch redesign landed (jump table append-only atomic
store; dispatch tables via `AtomicPtr` snapshot), and tier-2 installs are staged
and applied at a stop-the-world.

- 🔴 **~2% rare torn-read crash remains** on protocol-impl redefinition under
  threshold-1 churn. The zero-cost fix (atomic 16-byte `LDP`, LSE2, ARM-only)
  needs IR/backend work and is unbuilt; the current mitigation is
  fence + recompile. This must be closed for "robust" live coding.
- 🔴 **Mutator-driven STW-on-redefinition does not work** because redefine runs
  Beagle code that self-parks. Any future redesign here has to respect that.

### 1.3 GC safety across safepoints (the B-class bugs) 🟡

The B4 concurrent-GC corruption (live value surviving an allocation call *in a
callee-saved register*, invisible to the conservative frame-slot GC, stale after
a moving collection) is **fixed in both register allocators** — see
`docs/NEXT_STEPS.md`. The standing rule: any value live across a GC safepoint
that might be a heap pointer must be in a GC-scanned slot.

- ⏸ **B4 residual perf**: the SSA fix uses `spill_one` (reload at every use); a
  spill-*around*-safepoint variant would be faster for many-use values.
  Non-blocking.
- 🔴 **The invariant is only enforced by discipline + a safety-net bail**
  (`verify_clobber_safety`). We have no exhaustive checker that proves *every*
  maybe-pointer cross-safepoint value is slot-rooted. A standing `--gc-always`
  fuzzer over concurrent allocation patterns is the realistic guard; it should
  be CI-enforced, not run by hand.

### 1.4 bigint missing core operations ⏸

From `docs/NEXT_STEPS.md` §4: `bigint` has **no `div`/`mod`/`divmod`/
`quotient`/`remainder`** (needs limb long-division with a documented sign
convention) and **no `pow`**. These are real bignum batteries; deferred but
needed before claiming numeric completeness.

### 1.5 date/time epoch round-trip off-by-a-day near year 0 ⏸

`from-epoch`↔`to-epoch` is off by a day for ~61 days around year 0 AD (Hinnant
civil-from-days era division uses floor where truncation is needed). No effect
on dates ≥ year 1. Fix: truncating division for the `era` terms.

### 1.6 SSA backend — not the default, several open correctness items 🟡

`BEAGLE_USE_SSA=1` is **off by default**. Tier-2 SSA hot-path codegen *is*
default-on and proven faster, but the full SSA pipeline still has open issues
tracked across `docs/SSA_STATE_AND_POSTMORTEM.md`,
`docs/SSA_PRESSURE_EXPLOSION.md`, `docs/SSA_RUNTIME_PARITY_PLAN.md`, and the
`project_ssa_*` memories:

- 🔴 Spiller not fully wired on all paths (historical ~15% SpillOverflow
  fallback); "zero bails" needs live-range splitting (Phase 7).
- 🔴 Coalescing O(G²) blow-up fix and the `color.rs` group-color change were
  reverted during bisection and are **not currently re-applied**
  (`SSA_STATE_AND_POSTMORTEM.md` §3).
- 🔴 Orphan `Label`-marker blocks polluting the predecessor map; dominance
  verifier landed for diagnostics but the root structural fix is open
  (`project_ssa_orphan_blocks`).
- 🔴 **`HeapLoadPair` (the protocol-dispatch torn-read fix) has no SSA
  representation and currently forces a bail to legacy.** The atomic
  inline-cache pair-load is a *two-def* IR op (`HeapLoadPair(key, value, base,
  offset)`); the single-assignment SSA form has no multi-result op, so `to_op`
  in `src/cfg/builder.rs` leaves it untranslated and any protocol-dispatch
  function bails out of SSA to legacy (which lowers it to the atomic `LDP`, so
  it is *correct* — just not SSA-compiled). **Before SSA goes default-on**, this
  must be either (a) given a real SSA multi-result representation that lowers to
  `LDP`, or (b) made an explicit, asserted permanent-bail — otherwise flipping
  SSA on would silently route protocol dispatch through an SSA path that
  reintroduces the torn read. Do not let this rot. Regression:
  `resources/dispatch_torn_read_test.bg`.
- The architectural contract is `docs/SSA_ARCHITECTURE.md` (invariants I1–I10);
  any SSA diff that contradicts it is a blocking failure.

### 1.7 Fixed-but-fragile classes worth a permanent guard 🟢→🔴

These are *fixed* but represent recurring failure modes that need standing
regression coverage so they don't silently return:

- **Live-coding deref/throw crashes** (`project_livecode_crash_class_and_hunter`,
  `project_live_coding_crash_fixes`): runtime ops segfaulting on scalars/null;
  throw delivery must restore **all** per-thread dynamic state (GC context,
  marks, effect handlers, prompt tags). New per-thread dynamic state MUST get
  snapshot+restore or these come back — there is no compiler check enforcing it.
- **REPL client disconnect** (`docs/REPL_CLIENT_DISCONNECT_CRASH.md`): a
  broken-pipe throw must truncate the effect-handler/prompt-tag side stacks.
  Fixed; needs to stay covered by the 30-abrupt-disconnect stress repro.
- **CI 6h hang** (`project_ci_hang_gc_root_invariant`): GC self-deadlock from
  allocating under a lock inside `gc_impl`. Fixed structurally; the lesson
  (never allocate a heap value while holding a lock the collector re-enters) is
  not mechanically enforced.

---

## 2. Testing gaps

### 2.1 Skipped / non-deterministic tests 🔴

Currently skipped (`// Skip`) tests that represent real coverage holes:

- `resources/repl_server_test.bg` — 🟢 **DONE.** Now a deterministic
  `main()`+snapshot end-to-end live-coding test: it starts a real REPL server
  (made stoppable + readiness-signalling + quiet) and drives function / struct
  (add-field) / protocol redefinition over the socket, asserting the new
  behavior is observable on the next eval. Determinism comes from the
  synchronous request→response barrier + a readiness atom (no sleeps). Building
  it found and fixed the `update_binding` concurrent-redefinition race
  (§1.2-adjacent). It is `main()`+snapshot rather than a `test {}` block because
  of the async-handler gap below.
- `resources/repl_session_test.bg` — still skipped for the same scheduling/ordering
  reason; convert it the same way.

**Found while building the REPL test (two real infra gaps):**

- 🔴 **`test {}` blocks have no ambient async handler.** Only `main()` and
  spawned threads get the cooperative scheduler (via `beagle.async/__main__` →
  `run-cooperative`); a `test {}` body does not, so socket / file / `sleep`
  ops (which `perform` an async effect) can't run in a test body, and even
  threads spawned from a test body can't do I/O (no event loop). This is why
  async tests are all `main()`+snapshot. Attempted fix: run each test body
  under `run-cooperative` in `src/main.rs run_blocks` — non-async tests pass,
  but socket/`sleep` in a test body then *hangs* (the cooperative parking
  doesn't resume under the test-runner's nested invocation). Needs proper
  integration before `test {}` blocks can test async/live-coding paths
  directly.
- 🔴 **`main` does not force-exit while spawned threads run.** A background
  thread that never self-terminates blocks process exit (no force-exit
  primitive) — verified with a minimal repro. The REPL test had to explicitly
  `close` its session (to stop the session eval-loop thread) and stop the
  server before `main` could return. Servers/long-lived-thread programs need a
  force-exit or all threads must be cancellable.
- `resources/struct_structural_redefine_race_test.bg` — now enabled as a `test`
  block, but historically the hardest to keep green; keep it in the
  always-run-under-`--gc-always` set.

These are skipped because the test harness compares stdout exactly and can't
express "eventually" / "in any order" assertions. We need **non-deterministic
assertion primitives** (await-condition, unordered-set-equality, timeout-bounded
polling) so concurrency and server tests can run for real instead of being
skipped.

### 2.2 No standing fuzz/stress harness in CI 🔴

The roadmap (`docs/STDLIB_ROADMAP.md` Phase 0) specifies a GC regression harness
(split assertion, 6-thread string churn, missing-file read, `sqrt(2)`, nested
map literal, thread-per-connection echo, under both `--gc-always` and a
normal-pressure loop) that **gates** progress. This needs to actually exist as a
checked-in, CI-run harness rather than ad-hoc manual runs. Same for:

- A **concurrent-redefinition fuzzer** (§1.1) running for a long duration.
- A **live-coding panic hunter** beyond the existing `smoke/` harnesses —
  `smoke/live_coding_smoke.py` and `smoke/zelda_livecode_hunt.py` exist and are
  effective but are not part of the suite (by design); they should be wired into
  a nightly/CI job that fails on *any* server panic.

### 2.3 Property/invariant tests for the non-deterministic modules 🔴

`docs/NEXT_STEPS.md` §4 deliberately skipped channel/async/timer in the
gap-hunt because they have no Python ground truth. They need property/invariant
tests (e.g. channel FIFO ordering, no message loss, timer monotonicity) instead.

### 2.4 Disk-racing tests 🟡

`reflect/write-source` tests still race on disk and need a runner sandbox
(per-test temp dir) — noted in `project_tier2_dead_feedback`. fs/socket tests
were made parallel-safe with unique paths/ports, but the disk-write reflect path
is the remaining offender.

### 2.5 Backend / GC matrix coverage 🟡

The suite passes on ARM64 + the three GCs, but coverage across the full matrix
(x86-64 under Rosetta × {mark-sweep, compacting, generational} × {SSA on/off} ×
{tier-2 on/off} × `--gc-always`) is run by hand, not gated. A CI matrix would
catch backend-specific regressions (e.g. the `MoveFloat` integer-`mov` bug that
silently corrupted FP block-param transfers).

### 2.6 No coverage measurement 🔴

There is no line/branch coverage tooling for the Rust runtime or the `.bg`
stdlib. We're flying on "all tests pass" without knowing what fraction of the
runtime is exercised.

---

## 3. The road to production

### 3.1 Live coding — the headline goal 🔴

Robust live coding means: edit and reload code against a **running** process
(REPL server, or an embedded host like jim-editor / a game) for an extended
session with zero crashes and no state corruption. Where we are and what's left:

**Working today** (`project_cooperative_async_default`,
`16194f8`/`b83cc75` reflect/persist work): per-session namespaces, sticky
source-text anchoring for `reflect`/`persist`, `disk_location` accuracy across
tier-up, multi-def splices + failed-splice recovery, redefining live structs
with active readers, cooperative scheduler default.

**Open for "robust":**

- 🔴 **Deterministic REPL-server test coverage** (§2.1) — we cannot claim robust
  live coding while the server's own test is skipped.
- 🔴 **Long-session soak**: no evidence we can run a multi-hour live session.
  Need a soak test that continuously redefines structs/functions/protocols
  against worker threads and asserts no leak (RSS bound), no crash, no stale
  state. The sticky-bit slow-path accumulation (§1.1) is a concrete suspected
  long-session degradation.
- 🔴 **Protocol-redefine torn read** (§1.2) — the ~2% crash must go to 0%.
- 🟡 **Embedded-source `${...}` interpolation leak**
  (`project_embedded_source_interpolation_leak`): passing Beagle source
  containing `${...}` to `eval`/`persist` interpolates it in the REPL's scope.
  Mitigated by escaping at the call site; a real fix would make `eval` not
  re-interpolate string-literal source. Bit the live-code agent once already.
- 🔴 **State migration story**: structural redefinition fills new fields from
  literal defaults or `null`. Live coding often wants *custom* migration (carry
  over derived state). No hook exists for "when this struct's layout changes,
  run this migration fn."
- 🔴 **Error surfacing in the REPL**: a redefinition that fails to compile should
  give a precise, recoverable error to the session without disturbing other
  sessions; the failed-splice recovery path exists but its UX/guarantees aren't
  specified or tested across all failure modes.

### 3.2 Language feature gaps 🟡

- **SSA as default** (§1.6) — needed for the perf story to be the *default*
  experience, and to retire the dual legacy/SSA maintenance burden.
- **Float unboxing breadth** (`project_float_unboxing`,
  `project_type_aware_gc_safety`): field-fed floats stay boxed; the whole class
  of register-resident loop floats across calls is gated on **deoptimization**,
  which is built but narrow (pure-no-call int loops only, default-off behind
  `BEAGLE_SSA_DEOPT`). Broadening deopt unlocks a large perf class.
- **OSR** (`project_osr_landed`): landed behind `BEAGLE_OSR`, limited to ≤8
  live-ins; fannkuch-class loops need an OSR buffer. This is the biggest
  remaining perf lever (single-function hot loops never tier up on entry-counter
  triggering). Make it default-on and lift the live-in cap.
- **No whole-program analysis** (`project_no_whole_program_analysis`) is a hard
  constraint, not a gap — any optimization assuming "all callers/constructors
  seen" is unsound because code can load/redefine at runtime. Future opts must
  respect this.

### 3.3 Standard library — toward batteries-included 🟡

Catalog: `docs/STDLIB.md`; roadmap: `docs/STDLIB_ROADMAP.md`. ~35 modules
shipped. Constraints (from `docs/NEXT_STEPS.md`): implement **in Beagle**
wherever possible, FFI for system/C, minimal new Rust, no external Rust crates,
GC-correctness validated every step.

Known gaps:

- 🔴 **bigint** division/mod/pow (§1.4).
- 🔴 **HTTPS/TLS/wss**: plaintext HTTP only. TLS needs Security.framework Secure
  Transport via `SSLSetIOFuncs` C callbacks (the weakest FFI corner) or a `curl`
  shell-out stopgap. Until then there is no secure networking.
- ⏸ **Bigger Python-parity modules** not yet picked: decimal/fractions,
  zlib/gzip, zipfile/tarfile, sqlite, difflib, secrets/CSPRNG, datetime
  timezones, functools.lru_cache.
- 🟡 **FFI robustness**: TLS will exercise C-callback-into-Beagle and GC-managed
  buffer corners that are currently lightly tested. Needs a dedicated FFI stress
  suite.

### 3.4 Runtime & GC hardening 🟡

- 🔴 **GC-root invariant enforcement** beyond discipline (§1.3) — a static or
  dynamic checker that proves the maybe-pointer-cross-safepoint rule.
- 🔴 **Thread-count single source of truth** (§1.1).
- 🟡 **Leak/RSS bounding** under long live sessions (§3.1).
- 🟡 **Stop-the-world cost** of structural-redefinition migration — acceptable
  for a dev/live-coding loop, but it's a global pause; document the expected
  latency and whether it's tolerable for embedded hosts (games at 60fps).

### 3.5 Tooling & ergonomics 🔴

- **Debugger frontend**: a GUI debugger written in Beagle is a stated project
  goal; the runtime introspection exists, the frontend does not.
- **Diagnostics**: error messages, source spans, and stack traces across the
  eval/continuation/effect boundaries need an audit for a production-quality
  developer experience.
- **Packaging/distribution**: there is no package manager, dependency story, or
  versioned stdlib release. `beag` installs from source. Production use needs a
  distributable runtime and a way to ship libraries.
- **Platform**: macOS-only (ARM64 native, x86-64 via Rosetta). No Linux/Windows
  native backend; Linux is exercised only in CI for the GC matrix. A genuinely
  production language needs at least Linux.
- **CI completeness** (§2.5/§2.2): the matrix and the stress/fuzz harnesses need
  to be gating, not manual.

---

## 4. Priority ordering (recommended)

For the stated goal — **production-ready with full robust live coding** — the
critical path is roughly:

1. **Deterministic REPL-server test + non-deterministic assertion primitives**
   (§2.1) — unblocks honestly testing everything else about live coding.
2. **Long-session soak harness** for concurrent redefinition (§3.1, §2.2) —
   turns "we fixed nine bugs" into "it survives hours."
3. **Close the protocol-redefine torn read to 0%** (§1.2).
4. **Retire the sticky-bit permanent slow path** after migration settles (§1.1).
5. **Mechanical GC-root-invariant guard** + thread-count unification (§1.3, §1.1).
6. **SSA default-on + OSR default-on** (§1.6, §3.2) — the perf story.
7. **TLS/HTTPS + bigint division + the next stdlib tier** (§3.3).
8. **Linux native + packaging + debugger frontend** (§3.5) — true production
   distribution.

---

## Appendix — index of the deep-dive docs this consolidates

- Concurrent redefinition: `CONCURRENT_REDEFINE_OPEN_ISSUES.md`,
  `GC_REDEFINE_RACE_INVESTIGATION.md`, `GC_STALE_ROOT_REDEFINE_BUG.md`
- Protocol redefinition: `PROTOCOL_REDEFINE_HANDOFF.md`
- REPL/live coding: `REPL_CLIENT_DISCONNECT_CRASH.md`, `self-hosted-test-runner.md`
- SSA: `SSA_ARCHITECTURE.md`, `SSA_STATE_AND_POSTMORTEM.md`,
  `SSA_PRESSURE_EXPLOSION.md`, `SSA_RUNTIME_PARITY_PLAN.md`
- Perf: `PERF_NEXT_WORK.md`, `TIER2_OPTIMIZATION.md`, `OSR_DESIGN.md`,
  `OSR_PERF_HANDOFF.md`
- GC design: `GC_ROOT_SLOTS_PLAN.md`, `ROOT_SLOT_REUSE.md`,
  `fixed-root-area-gc-design.md`, `stack-segment-safety-margin.md`
- Stdlib: `STDLIB.md`, `STDLIB_ROADMAP.md`, `NEXT_STEPS.md`
- Language reference: `language-guide.md` (§28 = common mistakes)
