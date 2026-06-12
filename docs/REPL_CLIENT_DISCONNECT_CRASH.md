# Handoff: REPL server aborts when a client disconnects mid-eval

**Status:** FIXED. Root cause: `throw_exception` unwound the native stack but
never truncated the per-thread **effect-handler / prompt-tag side stacks**
(`PerThreadData::effect_handlers` / `prompt_tags`). The broken-pipe throw from
`socket/write` unwound past the async runtime's `handle` frames, leaving stale
records whose SP/FP pointed into abandoned stack. A later `perform` (next
socket op) matched a stale record, the tagged capture computed a capture
boundary below the current SP (`saturating_sub` → zero-byte segment), and
invoking that continuation hit the `segment_frame_info().expect(...)` abort.

Fix: `ExceptionHandler` snapshots both side-stack depths at push
(`saved_effect_handlers_len` / `saved_prompt_tags_len`); `throw_exception`
truncates both stacks back to the handler's level on delivery, releasing the
GC root slots of dropped handler entries (`unwind_effect_state_to` in
`src/builtins/exceptions.rs`). The same delivery path also rolls back
continuation marks (`binding`) — see the sibling fix in the same commit.

Verified: §2 repro (30 abrupt disconnects) — zero panics, server keeps serving.
**Severity:** High for the live-coding setup — a single misbehaving/disconnecting
client takes down the whole REPL server (process abort, all sessions lost).
**Pre-existing:** YES. Present on `main`. NOT introduced by `ssa-foundation`
(see §4 for proof). Surfaced while stress-testing the live-coding path.

## 1. Symptom

```
thread 'main' panicked at src/builtins/reset_shift.rs:1383:10:
continuation_trampoline: continuation has no segment data
...
thread caused non-unwinding panic. aborting.
```

`continuation_trampoline` is the closure body for an *invoked* continuation. It
fetches the captured stack segment via `ContinuationObject::segment_frame_info()`
and `.expect()`s it. The crash means a continuation object is being invoked whose
**segment pointer is null** (`segment_frame_info()` → `None`; the object passed the
`from_tagged` check, so it's a real `ContinuationObject`, just with no segment).

## 2. Minimal reproduction

Start the REPL server with the release `beag`:

```bash
beag run resources/examples/repl_server.bg     # listens on 127.0.0.1:7888
```

Then a client that sends an eval and disconnects **before reading the response**:

```python
import socket, json, time
for i in range(30):
    s = socket.create_connection(("127.0.0.1", 7888))
    req = {"op":"eval","id":str(i),"session":"x","code":f'println("work{i}"); {i}*2'}
    s.sendall((json.dumps(req)+"\n").encode())
    s.close()                 # abrupt: never read the response
    time.sleep(0.05)
```

The server aborts within a few iterations (subsequent `connect()` → connection
refused). Reproduces 100% reliably within ~30 attempts.

What does NOT crash (useful negative results):
- Clean clients that read the full response, even 200+ evals incl. heavy
  protocol/struct/function redefinition under `BEAGLE_SPECIALIZE_THRESHOLD=1`.
- An eval that `throw`s, with a *connected* client (see §5 — separate bug).
- 50× sequential connect → eval → **clean** close.

So the trigger is specifically: **the server writes to a socket the client has
already closed.**

## 3. Mechanism (best current understanding)

1. The REPL server is intentionally single-threaded (`start-repl-server` in
   `standard-library/beagle.repl.bg` ~378: accept one client, `handle-client`
   loops over its requests until disconnect, then accept the next).
2. The server's `main` runs **inside the async runtime's delimited-continuation
   reset** (`beagle.async/__main__` → `beagle.core/__reset__`).
3. An eval completes; the server calls `send-response` (`beagle.repl.bg` ~45),
   which does a `socket/write` to the now-closed socket. That fails with
   **`Broken pipe (os error 32)`**, raised as a **resumable runtime error** (per
   Beagle's "all runtime errors are resumable" design).
4. `handle-client`'s `catch (e)` (~365) *does* catch and print it
   (`Error handling client: Write failed: Broken pipe` is printed **before** the
   panic). So the throw itself is delivered. The abort comes when a continuation
   associated with that resumable error is **invoked** with no segment.

The leading hypothesis: a resumable error raised from a **Rust builtin**
(`socket/write`) inside a reset context captures a continuation whose Beagle stack
segment between the throw-site and the reset is empty/absent (there are no Beagle
frames to snapshot — the immediate frame is native). Invoking that continuation —
during delivery to the handler, or when the reset frame later unwinds/resumes —
hits the null segment. The exact invocation site has not been pinned (the
trampoline is entered directly from JIT code, so the Rust backtrace shows only the
trampoline entry, not the Beagle caller).

This needs confirmation — see §6.

## 4. Proof it is pre-existing (not from `ssa-foundation`)

- `git diff main..HEAD --stat -- src/builtins/reset_shift.rs src/builtins/continuations.rs src/gc/`
  is **empty** — the continuation/GC code is byte-identical to `main`.
- Reproduces with tier-2 churn **off** (default threshold) and with **no
  redefinition at all** — none of the branch's dispatch/tier-2/protocol work is on
  the path.
- Built the **clean committed HEAD** (no uncommitted changes) in a worktree; it
  aborts identically.

## 5. Secondary bug (also pre-existing, likely related)

An eval whose body `throw`s (e.g. `throw("boom")`, or a field error like
`(S{a:1}).nope`) never sends a `status: ["done"]` message back, so a well-behaved
client blocks forever. The server stays up; only that request hangs. This is in
the eval/response path (`session/session-eval` in
`standard-library/beagle.repl-session.bg`; exception responses set
`request.result_atom` ~118/191). Worth fixing alongside, since it's the same
"exception inside the eval/reset machinery" area.

## 6. Where to look / how to continue

Code locations:
- Crash site + segment accessors: `src/builtins/reset_shift.rs`
  `continuation_trampoline` (~1360), `segment_frame_info` (~406),
  `segment_base_and_size`.
- Continuation capture (where the empty segment is likely produced):
  `capture_continuation_runtime_inner` (~847), `capture_continuation_tagged_runtime`
  (~986); `set_segment_ptr_with_barrier` / `set_segment_frame_pointer_offset`.
- Resume-closure construction (trampoline as fn_ptr): `src/ast.rs` ~2356, ~4785;
  effect/handler delivery in `src/builtins/effects.rs`,
  `src/builtins/continuations.rs`.
- REPL server: `standard-library/beagle.repl.bg` (`send-response` ~45,
  `handle-client` ~332, `start-repl-server` ~378);
  `standard-library/beagle.repl-session.bg` (`session-eval` ~198,
  `process-eval-request` ~88).

Suggested next steps:
1. **Confirm the invocation site.** Instrument the moment a `ContinuationObject`
   with a null segment is *created* (log a backtrace/marker), and separately when
   `continuation_trampoline` is entered, to capture which capture produced the
   empty-segment continuation and what invokes it. Gate on an env var; the
   timing is not tight here (unlike the protocol-redefine bug), so `eprintln` is
   fine.
2. **Decide the correct fix at the capture site, not the trampoline.** Either
   capture a valid (possibly empty-but-well-formed) segment for a resumable error
   raised at a Rust-builtin boundary, or make such an error's continuation a
   no-segment "tail" continuation the trampoline handles explicitly. Do **not**
   "fix" it by making the error non-resumable — resumability is a design goal.
3. **Repro harness:** the Python snippet in §2. Loop it ~30× to catch the abort.

## 7. Dead ends / notes

- Not concurrency: the server is single-threaded; the original "concurrent" repro
  was really queued clients timing out and disconnecting mid-eval.
- Not the dispatch/jump-table/tier-2 work (§4).
- A Beagle-level `try/catch` around `send-response` does not help — the broken
  pipe is already caught by `handle-client`; the abort is in the continuation
  machinery, downstream of the catch.
