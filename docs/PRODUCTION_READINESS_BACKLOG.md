# Beagle Production-Readiness Backlog

> **Progress log.** Done so far (suite 472/0, regression-tested):
> - ✅ B4 — `json/stringify` throws on non-finite floats (was emitting invalid `NaN`/`infinity`).
> - ✅ B8 — `mutable-array/write-field` throws on OOB write (was a silent no-op); added explicit `write-field-or-ignore`.
> - ✅ B10 — `ansi` honors `NO_COLOR` + `set-enabled!`/`enabled?` (escapes were always emitted).
> - ✅ Quick win #1 — `om-has-key?` O(n)→O(log n) (native `contains?`), fixing the O(n²) ordered-map build.
> - ✅ Quick win #2 — `contains-key?` delegates to native `contains?` (O(n)→O(log n)).
> - ✅ Quick win #3 — `last` O(1) for vectors (was O(n) seq walk + per-node allocations).
> - ✅ B3 — `fs/exists?`/`is-file?`/`is-directory?` (+ blocking-* variants) return a plain `Bool` (were `Result.Ok{value:bool}`, so `if fs/exists?(p)` was *always taken*). Migrated `glob.bg` + `custom_handler_test` snapshot. Regression: `fs_predicate_bool_test.bg`. (NOTE: `match perform X { fields } { ... }` is ambiguous — bind the perform to a `let` first.)
> - ✅ Quick win #9 — deleted the dead duplicated `file-stream-sync*`/`FileChunkSourceSync` copy in stream.bg (~106 lines; kept the `core/`-qualified one).
> - ✅ Quick win #4 — `stats/percentile` floor is O(1) native `to-int` (was an O(n) count-up loop); deleted `float-floor-to-int`.
> - (Earlier: cross-module protocol-extension compiler fix; `Conj`/`into`/`reverse`/`vec`/`subvec`/`includes?`/set-algebra; container protocol lift.)
>
> - ✅ Quick win #10 — MD5 round-shift/constant tables (`MD5_S`/`MD5_K`) built once at module load instead of rebuilt per digest. (Time FFI buffer deliberately NOT hoisted — a shared mutable buffer would be a cross-thread data race.)
>
> **ALL 10 quick wins done** + deep blockers B9 + B2. Suite 476/0, smoke 10/10.
> - ✅ **B2 (deep)** — under the default cooperative scheduler, `await` on a rejected future now THROWS (was returning the error as a value, breaking try/catch) and on a cancelled future THROWS (was parking forever). Fix: `poll-tasks` resumes the awaiting continuation with `AwaitRejected`/`AwaitCancelled` sentinels; `await` re-raises them (blocking/threaded handlers throw internally, never produce sentinels, so they're unaffected). The throw correctly crosses the park into the surrounding try/catch. Regression: `async_await_reject_cancel_test.bg`.
> - ✅ **B9 (deep)** — `beagle.bigint` now has `divmod`/`div`/`mod`/`pow`/`abs` (schoolbook base-10000 long division, square-and-multiply pow; truncated-toward-zero with remainder-follows-dividend, Java/Rust semantics; throws on /0 and negative exp). Thoroughly tested incl. 2^100, signed cases, and the `a == q*b + r` identity: `resources/bigint_division_test.bg`.
>
> Deep work remaining: B1 (`vec-pop` — native persistent-vec drop-last; risky core-data-structure surgery, fixes 5 O(n²) drains), B2 (async await on rejected/cancelled futures under default scheduler), B5–B7 (HTTP/WS hardening — own milestone). Plus the broad completeness/perf backlog (§3/§5).

## 1. Verdict

The Beagle stdlib is **broad and correct at the core, but not yet 1.0-ready**. The collection/numeric/codec foundations are genuinely production-grade; the failures cluster in three places: **silent-wrong semantics** (commas on floats, fs predicates returning `Result` not bool, await-on-rejected-future swallowing errors), **quadratic perf on the headline data structures** (channels, deque, priority queue, streams — all defeated by `drop(1)`/`take(len-1)` rebuilds), and **untested whole modules** (`beagle.spawn`, `beagle.test`, `beagle.random`).

Per criterion:

| Criterion | Grade | Summary |
|---|---|---|
| **Standard design** | B | Strong protocol model, but cross-module name/arg-order collisions (`partition`, `repeat`, `min-of`, string-arg position in ansi) and `?`-suffix-returns-non-bool footguns. |
| **Sense** (does the right thing) | C+ | Several silent-wrong bugs that violate the project's own "no silent sentinels" rule: invalid-JSON for non-finite floats, OOB array write no-ops, `with-timeout` doesn't time out, await-rejected returns the error as a value. |
| **Complete** | C | Core ops solid; big gaps in bignum (no division), HTTP server (serial, no param routes, no chunked), WebSocket (no fragmentation), ansi (no NO_COLOR), struct-pack (no floats/repeats). |
| **Tested + performant** | C | Correctness coverage good where it exists, but entire modules are CI-invisible, and the most-used concurrency/collection primitives are O(n²). |

**Top blockers:** (1) the `drop(1)`/`take(len-1)` O(n²) family across channel/deque/pq/stream — needs a native `vec-pop`/drop-last primitive; (2) the silent-wrong correctness bugs (fs predicates, await-rejected, non-finite JSON, OOB write); (3) the HTTP server being serial + exact-match-only routing makes it unusable for real services; (4) three untested modules.

---

## 2. Blockers (must-fix before 1.0, ordered by impact)

### B1 — Native persistent-vector `drop-last`/`pop` (the keystone perf+correctness fix)  ✅ PRIMITIVE DONE

> **Status: DONE (whole O(n²) family closed).** native `rust-coll/vec-pop` landed (`src/collections/persistent_vec.rs`, Clojure popTail; correct across 1/32/1024 boundaries, GC-barrier-safe). LAST-removal sites wired to `butlast`: `priorityqueue/pq-pop`, `containers` deque `pop-front`/`pop-back`. FRONT-removal sites restructured: `channel` now uses a two-vector queue (recv side reversed → dequeue = `butlast`; flips amortize O(1)) — kills both the O(n²) drain AND redundant work on CAS retry; `stream/BufferedSource` now uses an emit-index cursor instead of `rest(buf)` per emit. Demos: 50k pq/channel/stream drains all ~0.1s (were O(n²)). Suite 524/0, smoke 10/10, gc-stress green.
**Files:** `src/collections/persistent_vec.rs` (add structural drop-last, expose `rust-coll/vec-pop`), then `beagle.channel/try-receive-loop`, `beagle.containers/pop-front`+`pop-back`, `beagle.priorityqueue/pq-pop`, `beagle.stream/BufferedSource.source-next`.
**Why:** Five separate modules fake "remove last/first element" via `drop(1,q)` or `take(len-1,v)`, each O(n), turning every drain into O(n²). This defeats the channel FIFO, the deque (whose header *falsely* advertises amortized O(1)), and the heap (pq becomes O(n²) instead of O(n log n)). Under CAS contention the channel redoes the full O(n) rebuild on every retry.
**Fix:** Add `O(log n)` structural drop-last to the persistent vector, expose as `rust-coll/vec-pop`, replace all five call sites. Single primitive, five wins.

### B2 — `beagle.async` await swallows/hangs on rejected & cancelled futures (default scheduler)
**File:** `beagle.async/poll-tasks` (`PendingOp.SpawnedTask`).
**Why:** Under the **default** cooperative scheduler, `await(rejected_future)` *returns the error as a normal value* (breaks every `try/catch` around await), and `await(cancelled_future)` **parks forever**. Diverges from `BlockingAsyncHandler`/`poll-until-resolved` which correctly throw. Latent because the reject/cancel paths are untested.
**Fix:** Add a `FutureState.Cancelled` arm that resumes with a thrown sentinel; make the `Rejected` arm re-raise the error inside the awaiting continuation (reuse the existing `TcpIoError` sentinel-then-throw trick) so await throws consistently across all handlers.

### B3 — `beagle.fs` boolean predicates return `Result`, not `bool`
**File:** `beagle.fs/exists?`, `is-file?`, `is-directory?` (+ `blocking-*` variants).
**Why:** `if fs/blocking-exists?(p) { ... }` is **always taken** — a `Result.Ok{value:false}` struct is truthy. Directly contradicts the `?`-means-bool convention (`beagle.os` predicates do it right).
**Fix:** Unwrap internally and return a plain `bool` (default missing/err → false), or rename the Result-returning forms to non-`?` names.

### B4 — Non-finite floats serialize to invalid JSON
**File:** `beagle.json/write-value` (Float branch).
**Why:** `json/stringify(1e400)` → bare `infinity`; `stringify(0.0/0.0)` → `NaN`. Both are invalid JSON; any consumer (including `json/parse`) rejects them. Violates "no silent-wrong."
**Fix:** Test finiteness (`value != value` for NaN, compare against infinity sentinel) and throw `"beagle.json/stringify: cannot serialize non-finite float"` (or coerce to `null` like JS via an option).

### B5 — HTTP server is serial + exact-match-only routing  ⚠️ ROUTING DONE

> **Routing done:** `match-path` supports `:name` params + trailing `*name` wildcard; `router` threads matched params into `req.params`; added Response combinators (`with-status`/`with-header`/`with-body`/`with-content-type`) + query helpers (`query-params`/`query-param`, URL-decoded). Pure + tested (`resources/http_routing_test.bg`). **Remaining:** cooperative concurrent `on-connection` (still serial).
**File:** `beagle.http/serve` + `handle-connection`, `beagle.http/router`.
**Why:** `serve()` fully handles one connection before `accept()`ing the next — one slow client blocks all others (head-of-line-blocking DoS). `socket/on-connection` (cooperative spawn-per-conn) already exists and is unused. Routing is exact-string equality only — no `/users/:id` params, no wildcards, no prefix mounts. Neither is usable for a real service.
**Fix:** Reimplement `serve` on `socket/on-connection` (cooperative task per connection, single-thread for GC-safety, bounded in-flight). Add a segment-split path matcher with `:name` capture into a `params` map on `Request` and `*` tail.

### B6 — HTTP server ignores chunked transfer-encoding  ✅ DONE (req + resp)

> **Done:** pure byte-exact `decode-chunked` (+ `parse-chunk-size`) and incremental `read-chunked-body` (read-line size + read-n data); `read-request` now reads a `Transfer-Encoding: chunked` body (precedence over Content-Length). Tested `resources/http_chunked_test.bg` (incl. CRLF-inside-data Wikipedia example). Client `read-response` also decodes chunked responses. Done req+resp.
**File:** `beagle.http/read-request`.
**Why:** Body is read **only** via `Content-Length`; a `Transfer-Encoding: chunked` upload is silently parsed as empty body with framing bytes left in the buffer, and no 411/400 is returned.
**Fix:** Decode chunked bodies (hex size line → bytes → 0-chunk), or explicitly reject with 411/501. Don't return a silently-empty body.

### B7 — WebSocket drops continuation frames (no fragmentation reassembly)  ✅ DONE

> **Done:** `frag-step` (pure fold) + `reassemble` + `continuation?`; the server frame loop now accumulates fragments (data frame fin=false → continuations → fin=true) into one message, delivering only on completion, with control frames (ping/pong/close) handled inline between fragments. Tested in `resources/ws_fragmentation_test.bg`. (Client-side masked frames + close-code extraction still open.)
**File:** `beagle.ws/handle-ws-connection`, `read-frame`.
**Why:** RFC 6455 §5.4 fragmentation is mandatory on receive. A continuation frame (opcode 0x0) is **silently dropped**; a fragmented text message delivers each fragment as a separate partial payload. Also no max-frame cap → an attacker's 127-length frame triggers unbounded allocation (DoS), and client frames aren't validated as masked.
**Fix:** Buffer frames until FIN and deliver the reassembled message; reject (close 1002) on unmasked client frames / orphan continuations; add a configurable max payload (close 1009 on exceed).

### B8 — OOB array write silently no-ops
**File:** `beagle.mutable-array/write-field`.
**Why:** Writing past the end vanishes and returns null. Per the project's own rule, a programmer error must surface, not disappear. (OOB *read*→null is defensible; a swallowed *write* is not.)
**Fix:** Throw resumable `SystemError.IndexError` with index/size in the message; keep an explicit `write-field-or-ignore` if the silent variant is genuinely wanted.

### B9 — `beagle.bigint` has no division
**File:** `beagle.bigint.bg` (add `divmod`/`div`/`mod`/`pow`/`abs`).
**Why:** An arbitrary-precision integer type with no division, modulo, or pow is unusable for its primary jobs (modular exponentiation, base conversion, formatting). Every peer (Java BigInteger, Python int, num-bigint) ships divmod + pow.
**Fix:** Schoolbook long division on limb vectors (base-10000 keeps trial division cheap) → `divmod(a,b)->[q,r]`, wrap as `div`/`mod`; `pow` via square-and-multiply over `mul`; `abs`. Throw (resumable) on divide-by-zero.

### B10 — `beagle.ansi` cannot be disabled (no NO_COLOR)
**File:** `beagle.ansi.bg` (module-level).
**Why:** Every fn unconditionally emits ESC sequences, so any program using `ansi/*` dumps raw escapes into pipes/logs/redirects. Production styling libs honor `NO_COLOR`/isatty.
**Fix:** Module-level `enabled?` atom + `set-enabled!`, defaulting from `NO_COLOR`; `wrap`/`style` short-circuit to return the string unchanged when disabled.

### B11 — `beagle.stream/retry` never retries (and dead duplicate code)
**File:** `beagle.stream/RetrySource.source-next`, plus the **duplicated** `file-stream-sync*` definitions (~lines 747–830 vs 843–926).
**Why:** `next()` calls `ensure-closed` on error, so by the time `RetrySource` loops the upstream is already closed → the retry returns `Done`, silently swallowing the first error and faking a clean end. Separately, `file-stream-sync`/`FileChunkSourceSync` are defined **twice** in the same file with divergent bodies (~80 lines of dead, shadowed code).
**Fix:** `retry` must take a thunk `() -> stream` and re-create the source per attempt (or operate below the close boundary). Delete one copy of the duplicated `file-stream-sync*` (keep the `core/`-qualified one).

---

## 3. Completeness gaps (by module)

**beagle.bigint** — `divmod`/`div`/`mod`, `pow`/`pow-mod`, `abs`, `to-int`, `gcd`, `sqrt`, bit ops, equality predicate. *(div = blocker B9.)*

**beagle.http** — param/wildcard routing (B5); chunked req+resp (B6); `with-header`/`with-status`/`with-body` Response combinators; query helpers on Request; cookie parsing; `application/x-www-form-urlencoded` + multipart body parsing; keep-alive; HEAD auto-handling; client redirect-following, request timeout, TLS.

**beagle.ws** — fragmentation reassembly (B7); client-side (masked client frames); close frames carrying status code (1000/1001/1009) + extraction; max-frame cap.

**beagle.socket** — `read-line`/`read-all` at socket layer (http+ws each reimplement buffering); timeouts (`SO_RCVTIMEO`), `SO_REUSEADDR`, half-close/shutdown, UDP, peer/local address, non-throwing `try-connect`.

**beagle.url / ip** — full URL parser struct (scheme/userinfo/host/port/path/query/fragment); `parse-query-multi`; `encode-uri` vs `encode-uri-component`. IP: ~~IPv6 entirely absent~~ ✅ IPv6 DONE (parse/format RFC 5952 + loopback/unspecified/link-local/unique-local/multicast predicates); `is-loopback?`/`is-multicast?`/`broadcast-address`/`host-count`.

**beagle.channel** — `close!`/`closed?` (no end-of-stream signal; receive on empty busy-spins forever); bounded/buffered channel with backpressure; `select`/`alts`; `receive-with-timeout`; `try-acquire!`; parking variant (built on existing `core/future-wait`/`future-notify`).

**beagle.async** — `gather`/`all-settled` (collect partial results — `await-all` throws on first rejection); public promise/`deliver`.

**beagle.stream** — `concat`(>2), `partition`/`chunk`, `distinct`, `scan`, `window`, `interpose`, `tap`, parallel/buffered map; `merge`/`zip` for >2 streams.

**std (core/collections)** — `remove` (filter complement), `keep`/`keep-indexed`/`map-indexed`, 2-arity `reduce` (seed from first), `split-at`/`split-with`, `partition-all`, `nth` with not-found/throwing variant, native `map-dissoc`/`set-disj`, `Range` `Length` extend, variadic `concat`/`into`, multi-pair `assoc`.

**beagle.iter** — `combinations-with-replacement`, windowed/sliding `partition(n, step, coll)`, N-way `zip`.

**beagle.containers** — `om-remove`/`om-dissoc`, `dm-remove` (no deletion today).

**beagle.mutable-array** — `fill`, `to-vec`/`from-vec` bridge (rust-coll has them, unexposed), `index-of`/`find`.

**beagle.struct-pack** — float codes `f`/`d`, 64-bit `q`/`Q`, pad `x`, fixed string `s`, repeat counts (`>4H`), `unpack-from(offset)`, `iter-unpack`.

**beagle.csv** — configurable delimiter (TSV/`;`) + quote char; lazy row iteration.

**beagle.semver** — compound ranges (AND/OR/hyphen/wildcard), `inc`/bump, `format(SemVer)->string`, `sort`, `max-satisfying`.

**beagle.date/time** — `before?`/`after?`/`compare`, `add-months`/`add-years`, timezone offset (UTC-only today), custom format patterns; monotonic clock, `sleep`, http-date parse.

**beagle.hash** — `hmac-sha1` (conspicuous: 3 other HMACs ship — needed for AWS SigV2, OAuth 1.0a, TOTP), sha224/384, incremental/streaming API, pbkdf2, constant-time compare, crc32.

**beagle.cli** — type coercion (`:int`/`:float`/`:bool`), `:choices`, `:multiple` (today repeated option silently overwrites), subcommands, variadic positionals, `--version`, `parse-or-exit`/`print-help-and-exit` driver.

**beagle.log** — named/hierarchical loggers, pluggable sink (file/in-memory capture), lazy/thunk message API, ISO-8601 timestamps.

**beagle.text** — float-aware `commas`, `center`, `chars`/`char-seq`, public `split-lines`, whitespace-tolerant `parse-float`; co-locate `parse-float` with `parse-int` in core.

**beagle.ansi** — disable toggle (B10); strip ALL ANSI (CSI/OSC), not just SGR.

---

## 4. Test gaps

**Whole modules invisible to CI (highest priority — they can silently rot):**
- **`beagle.spawn`** — *entire module* untested; no `resources/*.bg` imports it. The Spawn/SpawnWithToken enums and all handlers duplicate `beagle.async` and can drift independently. Either test it or **delete it** and fold into async (see design note below).
- **`beagle.test`** — *entire framework* never auto-discovered (no `test{}` blocks, no snapshot marker, no importer). `assert-eq`/`run-tests`/`print-summary`/`record-*` all unverified. Add `resources/test_framework_test.bg` asserting `run-tests([...])` returns the right `{passed, failed, failures}`, **or delete it** if the built-in `test{}` runner supersedes it.
- **`beagle.random`** — `next-float`, `int-range`, `choice`, `shuffle`, `sample`, `uuid-v4`, `normalize-seed`, `xorshift-step` have **zero** suite coverage (the in-module `main()` self-test isn't discovered). Add `resources/random_coverage_test.bg` (determinism by seed, ranges, multiset-permutation for shuffle/sample, UUID shape/version/variant, throw paths).

**Untested-and-broken (test would have caught the bug):**
- `beagle.text/commas` with a Float (broken — §findings), `parse-float` with surrounding whitespace (returns null).
- `beagle.async` reject/cancel-through-await paths (the B2 bugs are *unreached* by any test).
- `beagle.stream/retry` (broken — B11).

**Large untested surfaces:**
- `beagle.http` — nearly the entire module: `router`, `read-request`, `serialize-response`, `request`, `serve`, `make-response`/`json`/`html`/`redirect`/`not-found` (only `ok()` is touched). Needs a socket-free harness feeding canned request bytes.
- `beagle.ws` — `read-frame`, `handle-ws-connection`, masked parse path, frame-type predicates, handshake — only exercised in a non-discovered `main()`.
- `beagle.async` — `with-scope`/scope-* family, `await-first`/`await-timeout`/`with-timeout`/`async-with-timeout`, `BlockingAsyncHandler` future ops.
- `beagle.stream` — ~30 combinators untested (`flat-map`, `take-while`, `reduce`, `merge`, `zip`, `from-generator`, `retry`, etc.).
- **std public-but-unreferenced:** `sort-with`, `subvec`, `unwrap-or` (0 refs); `zipmap`/`not-every?`/`reduce-right`/`union`/`intersection`/`difference` (single ref, no edge cases).
- `beagle.io` low-level: `read-char`, `eof?`, `read-bytes`, `write-bytes`, `write-stdout*`.
- `beagle.fs` real `BlockingFsHandler` path (append/copy/rename/file-size only via mock handler).
- `beagle.log` sink-split (debug/info→stdout vs warn/error→stderr) and timestamped `build-emit-line` shape — never directly asserted; a regression sending error to stdout would pass.

---

## 5. Performance hotspots

**The O(n²) "fake pop" family (fixed by B1's `vec-pop`):**
- `beagle.channel/try-receive-loop` — `drop(1,q)` rebuilds the whole vector per dequeue → O(n²) drain, redone on every CAS retry. *(Or restructure as `{in, out}` two-vector amortized-O(1) queue.)*
- `beagle.containers/pop-front`+`pop-back` — `coll-subvec` rebuilds the side vector per pop; drain O(n²), contradicting the documented amortized O(1).
- `beagle.priorityqueue/pq-pop` — `take(len-1,v)` O(n) copy dominates the O(log n) sift; `pq-to-sorted-vec` becomes O(n²).
- `beagle.stream/BufferedSource.source-next` — per-emit `core/rest(buf)` = `drop(1)` = O(n); draining an N-batch is O(N²). Use an emit-index atom instead.

**`O(n)` linear scans where an `O(log n)` native lookup exists (one-line wins):**
- `beagle.containers/om-has-key?` — linear scan of the order vector while `om.table` holds the same keys; called per `om-assoc` → O(n²) build. Replace body with `contains?(om.table, key)`.
- `std/contains-key?` — `any?` scan; redundant with native `contains?` (O(log n)). A footgun: the discoverable map fn is the slow one. Delegate to `contains?`.
- `std/last` — O(n) seq walk + n allocations on a vector; fast-path `get(coll, count-1)` for `PersistentVector`.

**`O(n log n)` full-rebuilds (need native primitives):**
- `std/dissoc` (no native map remove — add `rust-coll/map-dissoc`), `std/set-remove` (no native disj — add `rust-coll/set-disj`).
- `std/vals` and the `invert`/`map-keys`/`map-vals`/`select-keys`/`merge`/`merge-with` family re-look-up each value via `get(m,k)` in a keys loop. Iterate `seq()` entries to avoid re-lookup.

**Re-parsing / re-scanning (recompute on every call):**
- `beagle.template/render` — re-tokenizes + re-parses the template on *every* render (the dominant per-row/per-request loop pays full lexing each time). **Split into `compile(template)` + `render-compiled(compiled, ctx)`** (Jinja/Handlebars design — biggest single perf win in the text slice). `tokenize`/`find-from` are also O(n·m) per-position substring scans.
- `beagle.semver/compare` — re-parses both version strings every call; sorting N versions = O(N log N) re-parses. Expose a parsed-struct compare.
- `beagle.ip/is-private-ipv4?` — re-parses the input IP ~6× and re-parses the 5 constant CIDR strings every call. Parse to int once; precompute the 5 `[network, mask]` pairs as module constants.
- `beagle.stream/SplitOnSource`+`BySizeSource` — growing `buf ++ value` re-scanned by `index-of` from offset 0 each pull → O(n²) on long lines; `lines()` inherits it. Track a scan-start offset.

**`O(n²)` string concatenation in loops (use string-builder, which the modules already have):**
- `beagle.text/commas` — `result = result ++ ...` per digit (O(n²)) instead of a string-builder like the rest of the module.
- `beagle.log/format-fields-helper` — `acc ++ entry` per field (O(n²)), plus unstable field order (iterates unsorted map keys). Accumulate into a vector and join; sort keys.

**Per-byte / per-char allocation:**
- `beagle.hash/to-bytes`+`pad-message*`+`word-be/le` — message held in a PersistentVector with per-word O(log n) indexed reads → O(n log n) hashing. Use a mutable `arr` buffer (the module already does for the W schedule). Also hoist the round-constant tables (`md5-k`/`sha256-k`/`sha512-k`) to module-level `let` (re-allocated per digest today).
- `beagle.base64`+`hex`+`struct-pack` — per-byte persistent-vector `push` to build byte sequences. Accumulate in `beagle.mutable-array`, freeze once.
- `beagle.url/encode-char-into`+`append-char-bytes` — fresh 4-byte scratch builder *per character*. Hoist one reusable scratch builder per call.
- `beagle.io/read-stdin-line` — one **unbuffered `read()` syscall per byte**. Read a block, scan for `\n`, retain remainder.

**Server-side DoS-shaped costs:**
- `beagle.http/byte-length` — materializes a full string-builder copy of the body just to count UTF-8 bytes; body then copied a *second* time when appended. Add a byte-count primitive.
- `beagle.http/read-line`+`cb-fill!` — O(n²) buffer concat + full re-scan from index 0 per chunk for an unterminated long header (Slowloris vector). Track scan offset; cap line length, return 431.
- `beagle.fs/handle-copy-file` — whole-file-into-String copy: O(filesize) memory, **binary corruption** (String decode), and dropped permissions. Stream over `io/read-bytes`/`write-bytes`, or add a `std::fs::copy` builtin.
- `beagle.glob/find-walk` — walks the entire tree with no prefix pruning + a redundant `is-directory?` stat per entry. Prune by leading literal segments; use readdir `d_type`.
- `beagle.stats/float-floor-to-int` — O(value) count-up float loop where `to-int(x)` is O(1). Replace the whole function with `to-int(rank)`.

---

## 6. Prioritized roadmap

### Next 10 quick wins (high impact, low effort)

| # | Item | Criterion | Impact | Effort | Quick-win? |
|---|---|---|---|---|---|
| 1 | `om-has-key?` → `contains?(om.table,key)` (O(n²)→O(n log n) build) | performant | High | Trivial (1 line) | ✅ |
| 2 | `std/contains-key?` → delegate to native `contains?` | performant | Med | Trivial | ✅ |
| 3 | `std/last` fast-path `get(coll, count-1)` for vectors | performant | Med | Tiny | ✅ |
| 4 | `stats/float-floor-to-int` → `to-int(rank)` | performant | High (percentile O(n) scan) | Tiny | ✅ |
| 5 | `json/write-value` throw on non-finite float (B4) | sense | High | Tiny | ✅ |
| 6 | `mutable-array/write-field` throw IndexError on OOB (B8) | sense | High | Tiny | ✅ |
| 7 | `fs` predicates return bool not Result (B3) | design | High | Small | ✅ |
| 8 | `ansi` NO_COLOR enable toggle (B10) | complete | High (CLI usability) | Small | ✅ |
| 9 | Delete duplicate `file-stream-sync*` in stream.bg (B11 part 2) | design | Med (dead/divergent code) | Tiny | ✅ |
| 10 | Hoist hash round-constant tables + `time/epoch-millis` buffer to module scope | performant | Med | Tiny | ✅ |

### Deep work (high impact, real effort)

| Item | Criterion | Impact | Effort | Quick-win? |
|---|---|---|---|---|
| Native `rust-coll/vec-pop` + rewire channel/deque/pq/stream (B1) | performant | **Critical** (5 O(n²)→O(log n)) | Med (Rust + 5 call sites) | ❌ |
| Async await reject/cancel correctness (B2) | sense | **Critical** | Med | ❌ |
| HTTP server: cooperative `on-connection` + param routing (B5) | complete | **Critical** (serial DoS, no real routes) | High | ❌ |
| HTTP chunked transfer-encoding (B6) | complete | High | Med | ❌ |
| WebSocket fragmentation reassembly + max-frame cap (B7) | complete | High | Med | ❌ |
| `bigint` divmod/div/mod/pow/abs (B9) | complete | High | High (long division) | ❌ |
| `template/compile` + `render-compiled` split | performant | High (per-render re-parse) | Med | ❌ |
| Native `map-dissoc` + `set-disj` (HAMT remove) → fix `std/dissoc`/`set-remove` | performant | Med | Med (Rust) | ✅ DONE |
| Channel `close!`/`closed?` + parking receive (kill busy-spin) | complete/perf | High | Med | ❌ |
| Resolve `beagle.spawn` vs `beagle.async` duplication (fold or test) | design/tested | Med | Med | ❌ |
| Test harnesses for http/ws/spawn/random/test/async-reject | tested | High | Med | ❌ |
| `cli` type coercion + `:choices` + `:multiple` + subcommands | complete | Med | Med | ❌ |
| Cross-module naming fixes: `partition`/`repeat`/`min-of` semantics, ansi arg-position, `Range` Length | sense/design | Med | Small-Med | ⚠️ partial |
| `hmac-sha1` + refactor HMAC to shared `hmac-bytes` | complete/design | Med | Small | ⚠️ |

**Sequencing advice:** Land the 10 quick wins first (one session — they erase the worst silent-wrong bugs and three trivial perf cliffs). Then do **B1 (`vec-pop`)** as the single highest-leverage deep item — it fixes four modules at once. **B2** next (correctness, latent in the default scheduler). The HTTP/WS work (B5–B7) is the largest effort and should be scoped as its own milestone since the networking stack is the least production-ready slice.

---

## 7. Already production-grade — preserve these

Do not refactor these without strong cause; they have correctness proofs and exhaustive coverage:

- **Core sequences/maps/sort** — `reduce`/`map`/`filter`/`take`/`drop-while` (function-first, protocol-driven); the map utilities with correct present-with-null handling (`contains?` not `v==null`); the stable adaptive **Timsort**; the `Result`/`Error` model.
- **`beagle.priorityqueue`** — clean comparator contract, Floyd O(n) heapify, fully immutable, 23 tests (only `pq-pop` perf is weak — B1).
- **`beagle.mutable-array`** (28 tests), **`beagle.iter` combinatorics** (itertools-order, lexicographic).
- **Text/codec correctness:** `beagle.string-builder` (every fn tested), `beagle.ansi` SGR sequences (exact bytes), `beagle.base64`/`hex` (RFC 4648 vectors, multibyte UTF-8), `beagle.json`/`csv`/`ini` parsers (state machines, surrogate pairs, error paths), `beagle.struct-pack` round-trips.
- **`beagle.path`** and **`beagle.glob`** — reference-quality, POSIX-correct, exhaustively tested.
- **`beagle.os`** and **`beagle.process`** — clean libc wrappers, real bools, correct popen/pclose/status handling.
- **Numeric/crypto correctness:** `beagle.bigint` arithmetic (most-negative-int, canonical zero), `beagle.time` (Hinnant algorithm), `beagle.date`, `beagle.mathx`, `beagle.semver` (spec-accurate precedence, 91 asserts), `beagle.hash` (bit-exact KATs — correctness only; perf needs work), `beagle.stats`.
- **`beagle.ip`** (CVE-2021-29921 octal guard), **`beagle.url`** codecs, **`beagle.ws/accept-key`**+frame round-trips (RFC 6455 vectors).
- **`beagle.cli`** spec/parse core (immutable spec, cluster/`=value`/`--`, 30+ tests), **`beagle.log`** pure formatting (throw-on-invalid), **`beagle.test-async`** deterministic concurrency assertions (happens-before design), **`beagle.effect/resume-tail`** constant-stack path.
- **`beagle.stream`** pull-based `StreamSource` protocol design (idempotent CAS close) and the core `map`/`filter`/`take`/`lines`/`collect` combinators — the *architecture* is sound; the perf and `retry` bug are localized.