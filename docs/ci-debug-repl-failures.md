# CI Debug Build REPL Test Failures

## Failing tests

Three tests fail on **Linux x86-64 debug** CI builds:
- `test_repl_starts_socket_server`
- `test_repl_struct_hotreload_crash`
- `test_repl_shooter_hotreload_crash`

## What passes / what fails

| Platform / Mode | socket_server | struct_hotreload | shooter_hotreload |
|---|---|---|---|
| Linux debug | FAIL (flaky — passed on 3 of 10 runs) | FAIL (all 10 runs) | FAIL (all 10 runs) |
| Linux release | pass | pass | pass |
| macOS debug | pass (latest run) | pass (latest run) | pass (latest run) |
| macOS release | pass | pass | pass |
| Local debug (no trace) | pass | pass | pass |
| Local debug (with BEAGLE_TRACE) | FAIL | FAIL | FAIL |

## Symptom

All three failures have the same symptom: the test sends a message over TCP to the socket REPL, and gets **no response** (empty string). The 5-second `set_read_timeout` expires.

Specific panic messages:
- `test_repl_starts_socket_server`: "describe response should contain request id, got: " (empty)
- `test_repl_struct_hotreload_crash`: "Initial code eval should complete, got: " (empty)
- `test_repl_shooter_hotreload_crash`: "Initial code eval should complete, got: " (empty)

The REPL child process does NOT crash — there's no stderr output (no panic, no segfault). It just doesn't respond.

## What trace output shows (local repro with BEAGLE_TRACE)

When running locally with `BEAGLE_TRACE='event-loop,tcp'`, the trace shows:
1. Event loop created successfully
2. `tcp_listen` on the port succeeds
3. `tcp_accept` succeeds — client connection is accepted
4. `tcp_read_async` is initiated on the accepted socket
5. **Trace stops here** — no read result ever comes back, no write, no further event loop activity

The I/O thread accepted the connection and started a read, but the data the test wrote is never delivered as a result to the Beagle code.

## Timing observations

- Local debug **without trace**: test completes in ~1.1s (passes)
- Local debug **with BEAGLE_TRACE='*'**: test takes ~2.1s (fails — read timeout hit)
- CI Linux debug unit tests take 690-3160s depending on GC backend (vs ~600s local)
- CI integration tests take ~85s total

## What has been tried

1. **Removed BEAGLE_TRACE and trace feature from debug CI** — This was the original CI config change. Result: `test_repl_starts_socket_server` went from passing (3/5 runs) to failing (0/4 runs). The trace feature itself compiles `trace!()` to no-ops when `BEAGLE_TRACE` env var is unset, so the feature flag shouldn't affect runtime behavior. However, adding/removing any feature flag causes a full recompile with potentially different codegen.

2. **Increased read timeout from 5s to 30s** — Reverted. Didn't address root cause.

3. **Replaced thread::sleep(1ms) with thread::yield_now() in event_loop_run_once** — Made things worse. The busy-loop starved other threads (I/O thread, game thread). Release builds that previously passed started failing. `try_all_examples` also failed.

4. **Added c_call registration around thread::sleep in event_loop_run_once** — Current state. The idea: `event_loop_run_once` calls `thread::sleep(1ms)` on a managed Beagle thread. During sleep, the thread can't reach a GC safepoint. If GC tries to stop-the-world, it waits for this thread, which is blocked. Fix: register as `c_calling` during sleep so GC treats the thread as safe. Result: all macOS jobs pass, all release jobs pass, Linux debug still fails with same 3 tests.

## Open questions

1. **Why does the I/O thread's read never complete?** The trace shows `tcp_read_async` was initiated, but no result is ever produced. The test writes data to the socket after connecting. Is the I/O thread stuck? Is the data never arriving? Is there a mio/epoll issue specific to Linux debug?

2. **Why does the trace feature affect behavior?** Adding `--features trace` changes nothing at runtime when `BEAGLE_TRACE` is not set (the macro compiles to `{}`). But CI behavior changed when we removed it. Could be coincidence (flaky test), or could be codegen differences from the recompile.

3. **Why only Linux x86-64 debug?** macOS debug passes, all release builds pass, local debug passes. Something about the combination of Linux + debug + CI environment triggers this.

4. **What is the REPL thread doing while not responding?** We know it's not crashing (no stderr). Is it blocked on something? Is the event loop's `run_once` returning 0 forever because the I/O thread never produces results? Or is the REPL thread blocked before even calling `run_once`?

## Key code paths

- `event_loop_run_once`: `src/builtins.rs:9143` — polls for I/O results, sleeps 1ms if none
- `event_loop_thread_main`: `src/runtime.rs:2388` — I/O thread main loop, uses mio poll
- `handle_socket_read`: `src/runtime.rs:2260` — processes read results
- `beagle.repl/handle-client`: `standard-library/beagle.repl.bg:177` — REPL client handler, calls socket/read in loop
- `beagle.async` socket read: `standard-library/beagle.async.bg:2246` — loops calling `event-loop-run-once(loop_id, 0)`

## Relevant CI config

Debug CI runs: `RUST_BACKTRACE=1 cargo test --features $GC,$BACKEND -- --test-threads=1`

All tests run with `--test-threads=1` (sequential).

## New findings from local investigation

### 1. The child process is actually crashing

The empty socket read is not just a hang. In a standalone local repro using the same debug feature shape as CI, the REPL child exits with `SIGSEGV` (`returncode = -11`) about 1 second after the client sends the first socket message.

This means the tests are currently masking the real symptom:
- `read_until_done()` sees `""` because the peer closed the socket
- the tests then report a timeout/empty response
- but the underlying REPL process has already died

The current tests do not check the child exit status before killing it.

### 2. Minimal local reproduction

These reproduce locally for me on Linux:

```bash
cargo test --features generational,backend-x86-64,trace test_repl_starts_socket_server -- --test-threads=1 --nocapture
cargo test --features generational,backend-x86-64,trace test_repl_struct_hotreload_crash -- --test-threads=1 --nocapture
cargo test --features generational,backend-x86-64,trace test_repl_shooter_hotreload_crash -- --test-threads=1 --nocapture
```

Standalone repro without the cargo test harness:

1. Build a debug binary with the failing feature shape:

```bash
cargo build --features generational,backend-x86-64,trace
```

2. Start `target/debug/beag repl`, run:

```clojure
use beagle.repl as repl
thread(fn() { repl/start-repl-server("127.0.0.1", 23255) })
```

3. Connect to the socket REPL and send:

```json
{"op":"describe","id":"d1"}
```

Observed result:
- client receives `""`
- child exits with `-11`

### 3. The I/O thread does read the socket data

With `BEAGLE_TRACE=event-loop,tcp`, the trace gets further than the original notes suggested. The I/O thread does:

1. accept the connection
2. register the accepted socket
3. perform the immediate read
4. successfully produce `ReadOk` with the JSON request bytes

So the failure is **not** “epoll never delivered the read”.

The crash happens after the TCP read succeeds.

### 4. GDB backtrace points at continuation/effect-handler stack restoration

Running the standalone repro under `gdb` stops in:

- [`return_from_shift_runtime`](/home/jimmyhmiller/Documents/Code/beagle/src/builtins.rs#L11495)
- via [`return_from_shift_handler_runtime`](/home/jimmyhmiller/Documents/Code/beagle/src/builtins.rs#L11779)

Crash site:

- [`src/builtins.rs:11550`](/home/jimmyhmiller/Documents/Code/beagle/src/builtins.rs#L11550)

The failing copy had obviously invalid addresses:

- `src=0xdf`
- `dst=0xd8`
- `count=88`

That means the corrupted state is in the continuation/prompt/invocation-return-point machinery, not in TCP/mio itself.

### 5. Most likely root-cause area

The socket REPL path goes through async effect handling:

- [`beagle.socket/read`](/home/jimmyhmiller/Documents/Code/beagle/standard-library/beagle.socket.bg)
- `perform async/Async.IO`
- cooperative async handler in [`standard-library/beagle.async.bg`](/home/jimmyhmiller/Documents/Code/beagle/standard-library/beagle.async.bg)
- continuation capture / handler return path in [`src/ast.rs`](/home/jimmyhmiller/Documents/Code/beagle/src/ast.rs#L3900) and [`src/builtins.rs`](/home/jimmyhmiller/Documents/Code/beagle/src/builtins.rs#L11115)

Strong evidence points at a bug in the interaction between:

- `capture_continuation`
- `return_from_shift_handler_runtime`
- `return_from_shift_runtime`
- `invocation_return_points`
- prompt handler push/pop state

In particular, there is a suspicious mismatch between comment and implementation:

- comment: [`src/builtins.rs:11776`](/home/jimmyhmiller/Documents/Code/beagle/src/builtins.rs#L11776) says handler returns should avoid consuming invocation return points
- implementation: [`src/builtins.rs:11512`](/home/jimmyhmiller/Documents/Code/beagle/src/builtins.rs#L11512) unconditionally does `ptd.invocation_return_points.pop()`

I tested a narrow patch that stopped handler returns from popping return points. That changed the failure mode, but it also exposed broken prompt/handler state afterwards, so it is **not** the full fix. Still, it strongly suggests this is the right subsystem.

### 6. Practical next steps

1. Treat this as a continuation/effect-handler bug, not a TCP/event-loop bug.
2. Add a temporary assertion/log in `return_from_shift_runtime` for invalid `relocated_sp/original_sp` before the copy loop.
3. Instrument prompt stack depth and `invocation_return_points.len()` across:
   - `capture_continuation_runtime`
   - `return_from_shift_handler_runtime`
   - `return_from_shift_runtime`
   - `pop_prompt_runtime`
4. Update the REPL socket tests to report child exit status before killing the process. That will make future failures point at the real symptom immediately.

## Resolution

The failing REPL tests are now fixed locally.

### Runtime root cause

There were two continuation bookkeeping bugs in the async handler path:

1. Handler-return unwinds could still consume the wrong invocation return point and pop prompt state too aggressively.
2. A continuation invoked from inside an already-relocated continuation stack could still be marked as a "root" invocation.

That second bug was the direct cause of the `describe`/`eval` REPL crash:

- `accept` resumed the server on a relocated continuation stack
- `read` was then captured and invoked from inside that relocated stack
- the runtime incorrectly treated that `read` invocation as root
- `return_from_shift_runtime` copied mutable slots back into the wrong stack segment
- execution resumed with corrupted frames and crashed with `SIGSEGV`

The fix was to track whether a prompt was already relocated when the continuation was captured, and use that plus any enclosing relocated prompts to decide whether an invocation is truly root.

### REPL response fix

The REPL session layer was also changed so the eval worker thread no longer writes directly to the client socket. It now returns a response batch to the server thread, and the server thread performs the actual `send-response` calls.

That avoids cross-thread socket/event-loop access while still preserving the isolated eval session thread.

### Verified passing

These now pass locally with the debug feature shape that reproduced CI:

```bash
cargo test --features generational,backend-x86-64,trace test_repl_starts_socket_server -- --test-threads=1
cargo test --features generational,backend-x86-64,trace test_repl_struct_hotreload_crash -- --test-threads=1
cargo test --features generational,backend-x86-64,trace test_repl_shooter_hotreload_crash -- --test-threads=1
cargo run --features generational,backend-x86-64,trace -- test resources/handler_post_nested_test.bg
```

### Additional fixes

The original REPL crash fix uncovered two more continuation issues that showed up in longer async/socket flows:

1. In the final handler-return path, a valid stack-local `cont_ptr` could still refer to a nested relocated continuation. Using that continuation's prompt sent execution back to the relocated stack instead of the original root prompt, which produced a null return on unwind. Handler returns now prefer the saved root continuation over a still-valid nested stack local.
2. `async/with-implicit-async` resumed each async operation inline via `ImplicitAsyncHandler`, so long sequences like repeated `socket/read(conn, 1)` calls built an ever-deeper continuation chain. It now runs the thunk on an isolated `CooperativeHandler` + `scheduler-loop`, which keeps async sequencing bounded instead of nesting one invocation per operation.

### Verified after follow-up fixes

These now pass locally too:

```bash
cargo run --features generational,backend-x86-64,trace -- test resources/concurrent_socket_echo_test.bg
cargo run --features generational,backend-x86-64,trace -- test resources/async_implicit_handle_test.bg
cargo run --features generational,backend-x86-64,trace -- test resources/async_implicit_ops_test.bg
cargo run --features generational,backend-x86-64,trace -- test resources/async_parallel_test.bg
```

And a bounded direct run of the integration resource now completes its client workflow:

```bash
timeout 8s cargo run --features generational,backend-x86-64,trace -- run resources/repl_server_test.bg
```

That command times out only because the resource intentionally leaves the background REPL server thread running; the client-side test body reaches `All tests passed!` and `done`.
