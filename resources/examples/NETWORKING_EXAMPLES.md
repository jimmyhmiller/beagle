# Beagle Networking Examples

A small, self-contained tour of Beagle's networking stdlib: HTTP server/client,
connection pooling, HTTPS/TLS, WebSockets, and cooperative-concurrent fetches.

Every example runs in a single process: it stands up an in-process server on a
background thread (doing finite work — accept a fixed number of connections,
then close the listener and return, so the non-daemon thread doesn't hang
process exit) and drives it with a client in `main`, so each one self-verifies
and exits 0. None carry a `@beagle.core.snapshot` marker or `test` blocks — they
are illustrative examples, not part of the test suite.

All five examples ran cleanly on the first attempt (exit 0).

## How to run

From the repo root, after building the release binary (`cargo build --release`):

```bash
timeout 40 ./target/release/beag resources/examples/<file>.bg
```

## Examples

### Minimal REST API server — `resources/examples/net_rest_api.bg`
Builds a 3-route HTTP API (`GET /` welcome, `GET /users/:id` echoing a path
param, `GET /search` reading a query param), serves it via `http/router` +
`http/handle-connection` on a background thread that handles exactly 3
connections then closes, and drives it with 3 `http/get` client calls that print
status + body. Demonstrates `http/route` + `http/router(routes,
http/default-not-found)`, `:id` path params via `core/get(req.params, "id")`,
and `http/query-param` URL-decoding (`beagle%20lang` → `beagle lang`).

```
REST API demo on port 39422
/ -> 200 Welcome to the Beagle REST API!
/users/42 -> 200 user #42
/search?q=beagle%20lang -> 200 you searched for: beagle lang
done
```

### Pooled HTTP client — `resources/examples/net_pooled_client.bg`
A tiny local server accepts ONE connection whose keep-alive loop serves all
requests; a pooled `http/client()` makes three `client-get` calls to the same
origin (reusing that single connection), then hits a closed port to demonstrate
matching on `Result.Ok` / `Result.Err`. The single server accept serving all
three GETs is the proof the keep-alive connection was reused rather than
reopened; the closed-port request returns `Result.Err` and matches the
`Error.IO` branch.

```
GET /alpha -> 200: you asked for /alpha
GET /beta -> 200: you asked for /beta
GET /gamma -> 200: you asked for /gamma
closed port -> Error.IO: connection refused
done
```

### HTTPS server (TLS round-trip) — `resources/examples/net_https_server.bg`
A background thread runs a TLS server using the repo's self-signed cert fixtures
(`tls/server-config` + `tls/accept` + `http/handle-tls-connection`), accepting
exactly one connection then closing its listener; `main` is the client
(`tls/connect-insecure`), sends a raw HTTP GET over TLS, parses the response with
`http/read-response` over a TLS reader, and prints status/header/body. The
handler sets a custom response header via `http/with-header`, read back
client-side via `core/get(resp.headers, "x-served-by")` (headers are lowercased).
The example notes that `tls/connect-insecure` is dev/test only (self-signed, no
MITM protection); production should use `tls/connect`.

```
status: 200
header X-Served-By: beagle-tls-example
body: Hello over HTTPS! You asked for: /secure-hello
done
```

### WebSocket echo round-trip — `resources/examples/net_websocket_echo.bg`
Runs a one-connection WebSocket echo server (`ws/handle-ws-connection` with an
identity `on-message` handler) on a thread and a client (`ws/connect`, `ws/send`,
`ws/receive`, `ws/close`) in `main`, exchanging three messages over the RFC 6455
handshake with no external server or certificates. Demonstrates the full
client/server frame round-trip and a clean self-terminating exit (a random port +
ready-atom handshake avoid racing `socket/listen`).

```
sent: hello  ->  received: hello
sent: websocket world  ->  received: websocket world
sent: round-trip #3  ->  received: round-trip #3
done
```

### Concurrent HTTP fetches with futures — `resources/examples/net_concurrent_fetch.bg`
Fetches four HTTP endpoints concurrently with `future()` / `async/await` on a
single cooperative thread, against a local server whose handlers park
(`async/sleep`) so the per-request delays overlap — total elapsed (~206ms) is one
delay, not four (~800ms), proving the concurrency. `main` fires all four
`future(http/get(...))` at once, then `async/await` + matches each
`Result.Ok{value: resp}` / `Result.Err`.

> Correctness note baked into the example: to demonstrate cooperative
> concurrency you must park via an async op (`async/sleep` / socket I/O), **not**
> `core/sleep` — `core/sleep` blocks the OS thread and does NOT yield to the
> cooperative scheduler, which serializes the handlers (~800ms instead of
> ~206ms).

```
Fetched 4 endpoints concurrently:
  you asked for /alpha
  you asked for /beta
  you asked for /gamma
  you asked for /delta
elapsed: 206ms (concurrent, not 800ms sequential)
future()/await gave us concurrency on ONE thread — no OS threads, no locks.
```
