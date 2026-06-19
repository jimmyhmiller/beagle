# Beagle async explainer (interactive)

An interactive page that explains Beagle's async model and lets you **run every
scenario for real** against the compiler.

Beagle is native code (macOS only) — there is no in-browser runtime — so a tiny
local server acts as the "hosted REPL": it compiles and runs each snippet with
`target/release/beag` and returns the output.

## Run it

```bash
cargo build --release           # if you haven't already
python3 docs/async-explainer/server.py
# open http://127.0.0.1:8787
```

Edit any snippet and press **Run** (or ⌘/Ctrl-Enter). The last card is a
free-form playground.

`PORT` and `RUN_TIMEOUT` (seconds) are configurable via env vars. The server
binds to localhost only and executes arbitrary Beagle code you type — keep it
local.

## What it demonstrates (each is a live, self-terminating program)

1. **Sequential by default** — plain top-to-bottom code, no callbacks.
2. **`future` = opt-in concurrency** — two 200 ms sleeps: ~400 ms sequential, ~200 ms wrapped in `spawn`.
3. **Many clients, one thread** — a sequential `loop { read; write }` handler serves two clients concurrently on a single thread (proven by equal thread ids).
4. **Error isolation** — a handler `try/catch` survives a continuation park; one client's thrown error doesn't kill the server.
5. **Why the default matters** — cooperative (default) overlaps; the explicit `run-blocking` handler serializes.

Server scenarios accept a *bounded* number of connections so they terminate;
`socket/on-connection` loops forever and would time out in the playground.
