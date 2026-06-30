# Handoff: cooperative-scheduler continuation leak (HTTP server OOMs under load)

**Status: FIXED (2026-06-30).** The HTTP server now holds flat RSS (247 MB steady
over 50k+ `ab` requests, was OOM @30k); `leak_await_reuse` is flat (238 MB, was
OOM). Full suite 549/0/1-skip, smoke 10/10. The fix is stdlib-only
(`standard-library/beagle.async.bg`) — no runtime changes. See §FIX below; the
historical diagnosis (§1–§3b) is kept for context. The earlier "validated
mechanism breaks concurrency / corrupts memory" findings were resolved by making
the fix complete (every cooperative resume is a tail resume — see §FIX).

---

## FIX (what actually landed)

Three changes, all in `beagle.async.bg`, implementing **fully tail-resumptive
handling** (the textbook correct shape for a delimited-continuation scheduler —
cf. Koka's tail-resumption optimization):

1. **Every resume in the `CooperativeHandler` is now `effect/resume-tail(resume,
   v)`** instead of `resume(v)` — the parking ops already didn't resume; the
   *synchronous* ops (`Spawn`, `SpawnWithToken`, `TcpListen`, `TcpClose`,
   `TcpCloseListener`, `Await{null}`, `AwaitAll`, `AwaitFirst`, `Cancel`) now tail-
   resume. This is the load-bearing change: a plain `resume(v)` re-pushes a NESTED
   prompt, so the continuation of a *following* `perform` gets captured under that
   ephemeral inner prompt; when later tail-resumed it lands at the wrong stack
   depth → the corruption that sank every prior attempt. Tail resumes never nest,
   so **every continuation is captured under the single, stable scheduler prompt**,
   and `resume-tail` (which lands the body at its `original_base`) is always
   correct because `original_base + seg_size == the live scheduler-handle SP`.

2. **`scheduler-loop` resumes the parked task via `effect/resume-tail`** (was
   `item.resume(item.result)`), reclaiming the handler-body frame that held the
   previous continuation — the chain link that caused the leak.

3. **Ready-queue restructure**: `resume-tail` teleports (does not return to a `for`
   loop), so the scheduler now resumes AT MOST ONE ready task per iteration and
   stashes the rest of a poll's ready batch in `handler.ready` / `handler.ready_idx`
   (added to `CooperativeHandler`). The un-dequeued tasks are NOT re-polled — their
   event-loop results are already consumed and carried on each `ReadyTask`.

Why this is principled (answers "how would a proper delimited-continuation system
do it?"): a scheduler is the *tail-resumptive* case — every resume is the last
thing the handler does. Making every resume tail-resumptive means no handler frame
and no nested delimiter is retained across a resume, so the continuation chain
that caused the leak cannot form, and the placement fragility that caused the
corruption disappears (one stable prompt, never a re-homed/ephemeral one). Gate
used: the §3 `ab` load test (flat RSS), full suite, smoke.

---

**Severity:** #1 production blocker. The HTTP server (and any await-heavy async
program) leaks memory linearly under sustained traffic and **OOM-crashes**. There
is no stable error-free request rate.

---

## 1. The bug in one paragraph

Every async **park/resume of a future** (i.e. `await`) leaks exactly one
continuation. The leaked continuations form a **reference chain** — each captured
continuation closes over the previously-resumed one — rooted by the single live
parked continuation, so the GC marks them all live and the heap grows without
bound. `socket/on-connection` parks per connection and the keep-alive loop awaits
per request, so the server leaks ~0.4 MB/request + ~0.9 MB/connection until the GC
can't grow the heap and panics.

The exact source is the scheduler's resume in `scheduler-loop`
(`standard-library/beagle.async.bg`, ~line 3011):

```beagle
} else {
    handle effect/Handler(Async) with handler {
        item.resume(item.result)        // <-- `item` (the continuation) is live in
    }                                   //     this handle BODY frame, INSIDE the
}                                       //     delimited prompt
```

`item` holds the continuation being resumed. The handle **body** frame is *inside*
the delimited prompt, so when the resumed flow performs its next async op, its new
continuation captures that frame → captures `item` → the previous continuation →
an unbounded chain.

---

## 2. How it was diagnosed (so you can trust the conclusion)

Build the release binary (`cargo build --release`; binary is `target/release/beag`).
A debug binary (`cargo build`) additionally has `BEAGLE_DEBUG_CONT_LIVE` (counts
live continuations in the old gen after a full GC).

Isolation results (watch RSS with `ps -o rss= -p <pid>`):

| Repro | Result |
|---|---|
| `while i<N { async/sleep(0) }` (one task, Timer park) | **flat** — but only because the timer rate-limits iterations (a few k/s), masking the same leak |
| `while i<N { async/await(null) }` (immediate resume, no park) | **flat** — capture+immediate-resume doesn't leak |
| `while i<N { future(fn(){1}); async/sleep(0) }` (spawn, no await) | **flat** — spawning + completing a task is fine |
| `while i<N { async/await(future(fn(){1})) }` | **LEAKS** ~0.5 GB/s |
| `let f=make-future(Resolved{1}); while i<N { await(f) }` (reuse ONE resolved future) | **LEAKS** — so it's per-**park**, not per-future |

Root-set elimination (instrument and observe — all done during diagnosis, now
removed):
- `add_root`/`remove_root` net count (the `GlobalObjectBlock` handle-root table): **flat**.
- `tg.handle_stack_top` (the `HandleScope` shadow stack): **0** always.
- `count(deref(handler.tasks))` in `scheduler-loop`: **1** always.
- `BEAGLE_DEBUG_CONT_LIVE`: **3000 continuations** accumulate in the old gen.

→ Not held by any growing root → they **chain**, rooted by the one live parked
continuation. The `match r {...}` inside `await` is a **red herring** (removing it
did not change the leak; an earlier "it amplifies" reading was release-vs-debug
timing confusion).

---

## 3. The load-test gate (use this to verify any fix)

Only `ab` (ApacheBench) and `curl` are installed (no wrk/hey/oha/vegeta).

Hello-world server (plain HTTP, real forever-server):

```beagle
namespace hello_server
use beagle.http as http
fn main() {
    http/serve("127.0.0.1", 8421, fn(req) { http/ok("Hello, World!") })
}
```

Run it, then ramp load and watch the server's RSS:

```bash
nohup ./target/release/beag hello_server.bg > srv.log 2>&1 & SRVPID=$!
sleep 2; curl -s http://127.0.0.1:8421/    # -> Hello, World!
for batch in 1 2 3 4 5 6 7 8; do
  ab -n 5000 -c 10 -k -q http://127.0.0.1:8421/ >/dev/null 2>&1
  echo "requests=$((batch*5000))  rss=$(ps -o rss= -p $SRVPID | awk '{printf "%.0f MB",$1/1024}')"
done
```

**Before the fix (current main of this branch):** 79 MB → 2.2 GB @5k → 10.2 GB @30k,
then OOM-panics in `src/gc/mark_and_sweep.rs:130` (`OutOfMemory`).
**A correct fix:** RSS stays roughly flat across 50k+ requests; server never dies.

Fast deterministic repro (no sockets) for inner-loop iteration:

```beagle
namespace leak_await_reuse
use beagle.async as async
fn main() {
    let f = async/make-future(async/FutureState.Resolved { value: 1 })
    let mut i = 0
    while i < 3000000 { async/await(f) i = i + 1 }
    println("done")
}
```
Before fix (release): 2 GB in 2 s, climbing. After a correct fix: flat.

---

## 3b. The dead-end that revealed the real cause (history — RESOLVED by §FIX)

The first attempt at §4 (`resume-tail` only in `scheduler-loop`, leaving the
handler's synchronous `resume(...)` calls intact) **silently corrupted the heap
whenever the resumed flow's NEXT `perform` was not shallow.** This section
explains why — the same mechanism that §FIX neutralizes by making *every* handler
resume tail-resumptive. (The ready-queue / one-resume-per-iteration restructure
was always correct; only the un-tailed handler resumes were the problem.)

### How `resume-tail` actually works (measured, not theorized)

The trampoline (`continuation_trampoline`, `reset_shift.rs`) lands the resumed
body at `dst`. For a tail resume `dst = cont.original_base()`, which is the
**`perform` SP** recorded at capture (`set_original_base(stack_pointer)` in
`capture_continuation_*_runtime`). The captured segment spans `[perform_SP ..
capture_top]` where `capture_top` = the **prompt SP** of the handle the
continuation was captured under, so `dst + seg_size == capture_top`.

The leak is broken **only if the body OVERWRITES the trampoline frame** (which
holds the previous continuation object — that frame, not the `item` local, is
what chains; see below). That requires `dst + seg_size > trampoline_fp`, i.e.
**`capture_top` must be a live prompt ABOVE the current trampoline.**

- **Shallow perform** (`while { await(f) }` directly under the scheduler handle):
  `capture_top` = scheduler-handle SP, which is above the trampoline → body
  overwrites trampoline → chain broken → `leak_await_reuse` goes FLAT. This is
  the only case §4 "validated."
- **Deep / nested perform**: `capture_top` lands BELOW `trampoline_fp`, the body
  is placed entirely below the trampoline, the trampoline frame survives (still
  leaks) AND the mis-placed copy scribbles over unrelated stack → a live parked
  continuation's heap segment gets a corrupt saved-FP chain → SIGSEGV / "misaligned
  pointer dereference" later, usually inside GC (`segment_outermost_fp_offset` via
  `segment_gc_frame_info`).

### What makes a perform "deep" — the real trap

Two independent sources:
1. **Lexical nesting**: `main → with-timeout → await` puts the `perform` several
   frames below the scheduler handle.
2. **Immediate non-tail resumes in the cooperative handler create NESTED prompts.**
   `Async.Spawn` / `Async.SpawnWithToken` / `Async.IO{TcpClose,TcpListen}` /
   `Async.Await{null}` etc. do a synchronous `resume(value)`. A non-tail tagged
   resume **re-pushes a fresh prompt record** at `post_overlay_sp` (trampoline_fp+16,
   see `reset_shift.rs` ~line 2319). So after e.g. `perform SpawnWithToken`, the
   flow runs under that inner prompt, and its next `perform AwaitWithTimeout` is
   captured with `capture_top` = the INNER prompt SP (verified: two captures with
   the same tag at different SPs — `0x…b4f0` the handle vs `0x…6610` the inner).
   When that continuation is later tail-resumed, the inner prompt is long gone, so
   neither `original_base` (deep) nor the current top prompt matches.

`with-timeout` does BOTH (`perform SpawnWithToken` then `perform AwaitWithTimeout`),
which is why `with_timeout_test` and `ws_client_test` crash. **`http/serve`'s hot
path uses the same machinery** (read-deadline / Slowloris = `Async.IOTimeout` →
`Race`; per-connection spawn), so the hello-world server **SIGSEGVs on the very
first request** under resume-tail. A corrupting "fix" is strictly worse than the
leak.

### Dead ends (already tried, don't repeat)

- **Anchor `dst = top_prompt_sp - seg_size`** (use the current live prompt instead
  of `original_base`): fixes the deep case but BREAKS the others — the continuation
  was captured under a now-gone *inner* prompt, not the current top one, so the top
  prompt is the wrong anchor. Net: trades one set of crashes for another.
- **Make the handler's immediate resumes also `resume-tail`** (to stop creating
  nested prompts): causes a DIFFERENT corruption — the next capture's
  `count_marks_in_frame_range` walks a poisoned saved-FP link (`0xffff…ffff`).
  resume-tail of the spawn continuation leaves the frame chain unwalkable.
- **Null `item.resume` after resuming** (break the chain at the GC level by making
  the retained ReadyTask stop pointing at the old continuation): does NOT reduce the
  leak. Codegen keeps the continuation closure in the trampoline frame independently
  of the `ReadyTask` object, so clearing the field changes nothing. (Required making
  the field `mut`.) This also proves the chain is through the **trampoline frame**,
  not the `item` local as §1 implies.

### Two viable fix paths (both are real work, neither is a one-liner)

- **(A) A correct "re-homing" tail-resume trampoline mode.** Add a resume variant
  that lands the body flush under WHICHEVER prompt is currently live (discarding the
  trampoline+handler frames at any depth) AND rebuilds the outer saved-FP/LR + the
  GC-prev chain to return to that live handle, AND correctly re-establishes the
  continuation's captured *inner* prompts (the side-state machinery) relative to the
  new base. This is deep surgery in the most fragile function in the codebase
  (`continuation_trampoline` / `continuation_restore_on_scratch`); budget for the
  multi-attempt history the SSA/continuation memories warn about. Pair it with the
  ready-queue restructure (which is correct and can be reinstated wholesale).
- **(B) Coroutine-style scheduler redesign.** Persistent `handle { driver-loop }`
  where parking does NOT abort to the handle but instead resumes a *stored driver
  continuation* (captured once, multi-shot), and the driver resumes the next task.
  Avoids `resume-tail` entirely. The driver reads ready tasks from a handler field
  (not a captured local), so no per-await chain forms. Bigger structural change, but
  sidesteps every trampoline-fragility issue above.

### Gate any future attempt with the hello-world `ab` load test (below) FIRST —
it crashes in ~1 request if resume-tail is mis-placed, far faster than the suite.

---

## 4. The (ORIGINAL, now-known-incomplete) fix plan: kept for reference — see §3b before using

`effect/resume-tail(resume, value)` (stdlib `beagle.effect`, backed by
`builtin/resume-tail` → `resume_tail_runtime` in `src/builtins/continuations.rs`,
which sets the tail-resume flag read by the trampoline in
`src/builtins/reset_shift.rs`) **consumes** the continuation and tail-replaces the
body frame, so `item` is no longer captured.

**Validated:** replacing the scheduler resume with
`effect/resume-tail(item.resume, item.result)` made `leak_await_reuse` **FLAT
(241 MB steady, was OOM)** in release. So the mechanism is correct.

**Why that one-line change is NOT enough:** the scheduler resumes *multiple* ready
tasks in a `for item in ready { ... }` loop. `resume-tail` tail-replaces the frame
and does not return to the loop to process the remaining ready tasks, so under
concurrency it drops tasks. With the naive change, these FAILED:
`concurrent_socket_echo_test`, `http_keepalive_test`, `with_timeout_test`,
`http_pool_test`, `ws_client_test`. (Dropping the `handle` wrapper entirely breaks
even more — the wrapper IS needed for the resumed flow's effect dispatch.) The HTTP
server's hot path *is* multi-ready (many concurrent connections), so the fix MUST
handle multiple ready tasks.

### Plan: resume one ready task per scheduler iteration via `resume-tail`, with a ready queue

The current `scheduler-loop` (`beagle.async.bg` ~line 2976):

```beagle
fn scheduler-loop(handler) {
    loop {
        let tasks = deref(handler.tasks)
        if length(tasks) == 0 { break(null) }
        let loop_id = deref(__io_loop_atom)
        let result = poll-tasks(tasks, loop_id)   // -> [ready, remaining]
        let ready = result[0]
        let remaining = result[1]
        reset!(handler.tasks, remaining)
        if length(ready) == 0 {
            // idle: event-loop-run-once / sleep
        } else {
            for item in ready {                    // <-- multi-resume loop = the problem
                reset!(__current-cancel-token, item.token)
                if item.resume == null { handle ... { item.result() } }
                else                   { handle ... { item.resume(item.result) } }  // LEAK
            }
            reset!(__current-cancel-token, null)
        }
    }
}
```

Restructure so each loop iteration resumes **at most one** task as a tail resume:

1. Keep a `ready-queue` (a module/handler field, or thread it through): when
   `poll-tasks` produces N ready tasks, append all N to the queue, set tasks =
   remaining.
2. Each `scheduler-loop` iteration: if the ready-queue is non-empty, **dequeue one**
   and resume it. Resume the parked-continuation case via
   `effect/resume-tail(item.resume, item.result)` (and keep the new-task case
   `handle ... { item.result() }` — new tasks have no captured handler, so they need
   the fresh `handle`; verify whether new tasks also need tail treatment to avoid a
   chain when the thunk itself awaits).
3. When that single resume's flow re-parks (performs an async op), control returns
   to `scheduler-loop`; the loop re-iterates and dequeues the next ready task.
4. **Do NOT just re-poll the un-dequeued ready tasks** — `poll-pending-op` for
   `TcpIO` calls `core/tcp-result-pop-for-op-id`, which **consumes** the result.
   A ready task's already-derived result must be STASHED in the queue (the
   `ReadyTask { resume, result, token }` already carries it), not recomputed.

Open questions to resolve while implementing:
- Does `resume-tail` cleanly return to `scheduler-loop` when the resumed flow
  **re-parks** vs **completes**? (Single-flow worked; verify the multi-flow
  hand-back.) The tail-resume flag (`take_tail_resume`) is one-shot — make sure one
  resume's flag can't bleed into the next (the cooperative handler also does
  immediate `resume(...)` calls for synchronous ops like `Async.Spawn`).
- New-task branch: confirm whether a spawned thunk that performs async ops also
  chains (the same `item`-in-body-frame capture). If so, apply the same treatment.
- `__current-cancel-token` publishing must still wrap each resume correctly.

### Gate every attempt with BOTH:
1. The `ab` load test above — **RSS must stay flat over ≥50k requests** and the
   server must not die. Also run `leak_await_reuse.bg` (release) and confirm flat.
2. Full suite (`./target/release/beag test resources/` — must stay 549/0/1-skip)
   **and** smoke (`python3 smoke/live_coding_smoke.py` — 10/10). The async/socket
   tests (`concurrent_socket_echo_test`, `http_keepalive_test`, `with_timeout_test`,
   `http_pool_test`, `ws_client_test`, `async_*`) are the canaries.

---

## 5. Key files

- `standard-library/beagle.async.bg` — `scheduler-loop` (~2976), `poll-tasks`
  (~2950), `poll-pending-op` (~2860), cooperative handler arms (`Async.Await` ~2591,
  `Async.Sleep` ~2439), `await` (~1684), `make-future` (~51), `FutureState` enum (~30).
- `standard-library/beagle.effect.bg` — `resume-tail` (~62).
- `src/builtins/continuations.rs` — `resume_tail_runtime` (~86) / `set_tail_resume`.
- `src/builtins/reset_shift.rs` — continuation capture/invoke/trampoline; tail-resume
  flag (~155); side-state re-establishment on invoke (~930–1010); shift teardown
  (~1783).
- `src/runtime.rs` — `run_gc` extra-roots assembly (~4033); `add_root`/`remove_root`
  (~326/369); `add_handle_root`/`register_temporary_root` (~6776/6913);
  `handle_stack` shadow stack (`HandleScope` in `src/collections/handle_arena.rs`).
- `src/gc/generational.rs` — `full_gc` + the `BEAGLE_DEBUG_CONT_LIVE` counter (~1494).

## 6. Background / related memory

- `project_cooperative_spawn_leak.md` — the canonical memory for this bug (mirrors
  this doc).
- `project_cooperative_async_default.md` — the cooperative scheduler is the default
  async handler; this leak is almost certainly pre-existing, just never load-tested.
- `project_continuation_trampoline_overlap.md` — a *different* (crash, not leak)
  trampoline hazard; useful context on how delicate this code is.
- `project_live_coding_crash_fixes.md` / `project_livecode_crash_class_and_hunter.md`
  — the continuation subsystem's crash-class history; change carefully.

This bug is pre-existing and orthogonal to the (green, complete) networking
milestone — see `docs/NETWORKING_HANDOFF.md`.
