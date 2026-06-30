# Beagle Networking Benchmarks

> **These are SINGLE-PROCESS micro-benchmarks.** In every case the server thread
> and the client run in the *same* `beag` process on the cooperative async
> scheduler, over loopback (`127.0.0.1`). They measure stack overhead and
> *relative* comparisons (pooled vs one-shot, plain vs TLS, etc.), **not**
> absolute production throughput. Treat the numbers as a way to compare code
> paths against each other, not as real-network performance figures.

All benchmarks were run with the release binary (`./target/release/beag`, ARM64,
default GC) and each exits 0. None carry a `@beagle.core.snapshot` marker or
`test` blocks, so the test suite skips them.

## Results

| Benchmark | Metric | Result | File |
|---|---|---|---|
| HTTP keep-alive throughput | requests/sec & us/request (one pooled keep-alive conn) | ~174–190 req/s, ~5274–5739 us/req (2000 requests, all 200) | `resources/bench_net_http_throughput.bg` |
| Connection-pooling speedup | total ms for N=400 each way + one-shot/pooled ratio | pooled ~1188 ms vs one-shot ~2436 ms → **2.05x** (range 2.05x–2.11x) | `resources/bench_net_pool_speedup.bg` |
| TLS overhead | avg ms/request over keep-alive + TLS overhead | HTTP ~2.55 ms/req, HTTPS ~2.70 ms/req → +0.1–0.3 ms/req (~3–12%) | `resources/bench_net_tls_overhead.bg` |
| WebSocket throughput | round-trips/sec & us/round-trip (one ws conn) | ~267–279 round-trips/s, ~3587–3744 us/round-trip (1000/1000 verified) | `resources/bench_net_ws_throughput.bg` |
| Router dispatch + serialization | dispatches/sec & us/dispatch (no sockets) | ~43,000–44,500 dispatches/s, 22–23 us/dispatch | `resources/bench_net_router_dispatch.bg` |

All five benchmarks ran cleanly.

## Methodology

### HTTP keep-alive throughput — `resources/bench_net_http_throughput.bg`
N=2000 timed requests (plus 1 untimed warmup). A server thread (`socket/listen`
+ ONE `socket/accept`) hands the single connection to `http/handle-connection`,
whose keep-alive loop serves every request; a pooled `http/client()` reuses that
one connection for all `client-get` calls to `http://127.0.0.1:${port}/bench`.
The server thread does finite work (serves N, then `socket/close-listener` and
returns) so the process exits cleanly. Wall-clock is timed with
`core/time-now()` (nanoseconds) around the client loop only. Single-process,
loopback only — not a real network benchmark.

**Result:** ~174–190 requests/sec, ~5274–5739 us/request, ~10.5–11.5 s total for
2000 requests. Three runs: (1) 189.6 req/s, 5274 us/req, 10549 ms; (2) 187.5
req/s, 5332 us/req, 10664 ms; (3) 174.2 req/s, 5739 us/req, 11478 ms. All 2000
requests returned 200; the handler served 2001 (2000 timed + 1 warmup).

### Connection-pooling speedup — `resources/bench_net_pool_speedup.bg`
Single-process micro-benchmark; one process runs both the HTTP server(s) on
background OS threads and the timing client on main under the cooperative async
scheduler. N=400 requests each way. Each phase has its OWN finite server thread
so the two measurements are independent and the process exits cleanly: the
pooled server accepts exactly 1 connection whose keep-alive loop
(`http/handle-connection`) serves all 400 requests (ends on client-close); the
one-shot server accepts exactly 400 connections (1 request each, client sends
`Connection: close` so the server closes its side immediately); then both close
their listeners and return. The handler returns a constant `http/ok("ok")` body.
The pooled phase runs FIRST, then one-shot. Timing via `core/time-now()`
(nanoseconds, `/1000000` for ms) around each request loop only.

*Reliability note:* doing the heavy one-shot churn before the pooled phase, or
having the server echo a longer/variable response body, intermittently wedged
the cooperative scheduler/socket layer (hang or rare abort); the final design
(pooled-first, constant short response body, a small `core/sleep(10)`
scheduler-pump between phases outside the timed regions) ran 10/10 + 3/3 clean.

**Result:** N=400 requests each way. Representative run: pooled (1 reused
connection) = 1188 ms; one-shot (400 fresh connections, `Connection: close`) =
2436 ms; speedup = 2.05x. Across 13 successful runs the numbers were stable:
pooled ~1180–1210 ms, one-shot ~2435–2505 ms, speedup 2.05x–2.11x. The one-shot
path pays a fresh TCP connect + teardown on every request, costing roughly 2x.

### TLS overhead — `resources/bench_net_tls_overhead.bg`
N=100 steady-state keep-alive round-trips per protocol (plus 1 warmup excluded;
the TLS handshake is excluded since the connection is established once). SINGLE
PROCESS: two server threads (a plain HTTP server via `http/handle-connection`,
and an HTTPS server via `tls/accept` + `http/handle-tls-connection` using the
`resources/test_tls_cert.pem` + `test_tls_key.pem` fixtures) and the client all
run in one process over loopback (random ports) under the cooperative async
scheduler. Both protocols use the IDENTICAL code path — one persistent
connection, the same raw `GET /ping HTTP/1.1 Connection: keep-alive` request
bytes, the same handler returning `http/ok("pong")`, and the same
`http/read-response` parser — so the only measured difference is the transport
(plain socket read/write vs `tls/read` / `tls/write`). HTTP client = raw socket +
`socket/reader`; HTTPS client = `tls/connect-insecure` +
`socket/reader-with(tls/read)`. Each server thread accepts exactly one
connection, serves the keep-alive loop until the client closes, then closes its
listener and returns. Timing via `core/time-now()` (ns) converted to ms.

**Result:** three clean runs (all exit 0, 100/100 ok on both protocols). Run 1:
HTTP 2.50469 ms/req, HTTPS 2.80525 ms/req, overhead +0.30057 ms/req (12.0%). Run
2: HTTP 2.58918, HTTPS 2.66337, overhead +0.07420 ms/req (2.9%). Run 3: HTTP
2.54906, HTTPS 2.70687, overhead +0.15780 ms/req (6.2%). Representative: HTTP
~2.55 ms/req, HTTPS ~2.70 ms/req, TLS overhead roughly +0.1–0.3 ms/req (~3–12%,
noisy since both are loopback round-trips dominated by per-request
scheduling/syscall cost rather than crypto). TLS is consistently the slower of
the two across all runs.

### WebSocket throughput — `resources/bench_net_ws_throughput.bg`
N=1000 round-trips; in-process echo server thread + client in ONE `beag` process
over ONE loopback ws connection. Server: a non-daemon thread does `socket/listen`,
sets a ready atom, `socket/accept`, then `ws/handle-ws-connection(conn,
fn(c,payload){payload}, 1048576)` (identity echo) for exactly ONE connection,
then `socket/close-listener` and returns. Client (in main, under the cooperative
async scheduler): `ws/connect`, one warmup send+receive, then a timed `while`
loop of N `ws/send` + `ws/receive`, verifying each reply equals the 38-byte
payload. Timing via `core/time-now()` (nanoseconds). Closes the ws connection
(`ws/close`) so the server thread ends and the process exits 0.

**Result:** N=1000 sequential `ws/send` + `ws/receive` round-trips over ONE
connection. Stable across 5 clean runs: ~267–279 round-trips/sec, ~3.59–3.74 ms
(3587–3744 us) per round-trip, elapsed ~3587–3744 ms, 1000/1000 echoes verified.
Representative: 267.0 round-trips/sec, 3744.75 us/round-trip, 3744 ms, 1000/1000
verified.

*Tuning note:* N=3000 (the originally suggested default) reliably OOM'd the GC
inside continuation capture (`mark_and_sweep` grow → ENOMEM) because each
`ws/receive` parks the cooperative scheduler and captures a full continuation, so
per-iteration retained memory grows; N was lowered to 1000 where it runs cleanly
and reproducibly. Latency is dominated by the cooperative-scheduler park +
continuation-capture on every receive, not by network (loopback, same process).
One transient base64/encode `TypeError` occurred on a single startup before a run
(a known intermittent runtime hiccup in `gen-ws-key`); it did not recur across
the subsequent 5 runs.

### Router dispatch + serialization — `resources/bench_net_router_dispatch.bg`
N=20000 dispatches per measured run, 3 measured runs after a 5000-dispatch
warmup. Single-process, NO sockets: builds an `http/router` with 10 routes
(static + `:param` + `*wildcard`), constructs 12 `http/Request` structs directly
(covering every route kind plus a 404 and a 405), then loops dispatching each
through the router closure and serializing the returned `Response` with
`http/serialize-response`, accumulating the wire-byte count. Times with
`core/time-now()` (nanoseconds). Server-side routing+serialization only; no
client, no network, no threads. Run with `./target/release/beag` (ARM64, default
GC).

**Result:** ~43,000–44,500 dispatches/sec, 22–23 us/dispatch, 2,286,610 bytes
serialized per 20000-dispatch run (deterministic byte count). Representative run:
20000 dispatches in ~455 ms = 43,883 dispatches/sec, 22 us/dispatch. Numbers were
stable across 4 separate process invocations (each printing 3 runs):
dispatches/sec ranged 43,089–44,582; per-dispatch 22–23 us.

## How to run

From the repo root, after building the release binary (`cargo build --release`):

```bash
timeout 40 ./target/release/beag resources/bench_net_http_throughput.bg
timeout 40 ./target/release/beag resources/bench_net_pool_speedup.bg
timeout 40 ./target/release/beag resources/bench_net_tls_overhead.bg
timeout 40 ./target/release/beag resources/bench_net_ws_throughput.bg
timeout 40 ./target/release/beag resources/bench_net_router_dispatch.bg
```

Each prints its own report and exits 0.
