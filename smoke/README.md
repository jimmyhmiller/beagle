# Live-coding smoke test

A panic hunter for the runtime. It starts real REPL servers (standalone
and `run-with-repl` with a hot game loop) and hammers them with the
operations live coding actually performs — evals, throws, aborts,
resumes, multi-def `reflect/persist` storms, failing + corrective
persists, abrupt client disconnects, effect unwinds, thread churn, GC
pressure, main-thread crash recovery, and a seeded random fuzz over all
of it. A scenario fails if the server panics, dies, stops responding,
or returns a wrong result.

**This is not part of the regular test suite.** Run it once after
runtime changes (exception delivery, continuations, GC context,
effects, persist/eval machinery, threads):

```bash
cargo build --release
python3 smoke/live_coding_smoke.py                  # full battery, ~15s
python3 smoke/live_coding_smoke.py --fuzz-ops 2000  # longer fuzz soak
python3 smoke/live_coding_smoke.py --only persist-churn
python3 smoke/live_coding_smoke.py --list           # scenario names
```

- Exit code 0 = no panics anywhere; nonzero = at least one failure.
- The fuzz seed is printed every run — reproduce a failure with
  `--seed <n>` (and the same `--fuzz-ops`).
- On failure the scenario's temp workspace and `server.log` are kept
  and the path + log tail are printed.
- `--beag /path/to/beag` (or `BEAG=` env) tests a different binary;
  default is `target/release/beag`.

Each scenario maps to a crash class that has actually shipped:

| scenario           | guards against                                              |
|--------------------|-------------------------------------------------------------|
| sanity             | basic eval/define/output plumbing                           |
| disconnect-storm   | broken-pipe throw leaking effect/prompt records → "continuation has no segment data" abort |
| throw-abort-resume | resumable-error delivery, nested suspensions, resume values |
| output-capture     | `binding(core/out)` leaking across throw unwinds            |
| persist-churn      | stale GC-context "shift without an enclosing reset" abort; multi-def byte-shift bookkeeping; failed-splice recovery |
| effect-churn       | throws unwinding past `handle` blocks                       |
| thread-churn       | concurrent throw/catch on spawned threads                   |
| gc-pressure        | allocation storms interleaved with unwinds                  |
| main-crash         | main-thread crash → `main-status`/`main-resume` recovery    |
| fuzz               | seeded random interleavings of all of the above             |

Validated: run against the pre-fix runtime (commit before the 2026-06
live-coding fixes), `disconnect-storm` and `persist-churn` catch the
historical aborts with their exact panic lines.

When you add a new per-thread runtime mechanism (a registry, a side
stack, a TLS cache), add a scenario — or extend the fuzz op table —
that churns it across throws and Rust→Beagle reentry. That's the
family every one of these crashes came from.
