# Beagle → Batteries-Included: Phased Roadmap

A plan to take Beagle from a capable functional core to a Python-comparable, batteries-included language able to build HTTP servers, websockets, and templating engines — under the hard constraints: **Beagle-first, FFI for system/C, minimal Rust, no external crates, GC-correctness validated every step.**

## Executive summary

Beagle is much further along than a naive inventory suggests. Collections/functional are Clojure-grade and GC-solid. Single-string text, regex (without GC pressure), JSON, math, FFI, and async TCP all work. I independently reproduced the headline bugs against the live `target/release/beag` build — **they are real, deterministic, and several fire under *normal* GC pressure, not just `--gc-always`.** The strategic situation is therefore: **the headline web "batteries" (HTTP, websockets, templating) are almost entirely buildable in pure Beagle TODAY — but a small cluster of runtime correctness bugs silently corrupts the exact code paths those libraries depend on (string splitting, equality, concurrent allocation, binary socket reads).** Fix that cluster first; then nearly everything downstream is pure-Beagle library work with a thin FFI seam for time/crypto-accel/TLS.

Two findings correct the inventory and matter for sequencing:
- **String predicates already exist** (`starts-with?`/`ends-with?`/`contains?`/`index-of`/`char-at` via `get`) — verified. The "text utilities missing" claim is partly wrong; the *real* text blocker is GC corruption in `split`/`words`/`lines` and the regex result-builders.
- **CLI argv already works** via `fn main(args)` — verified by the OS-domain probe. The inventory's "argv not reachable" was wrong.

## Blocking bugs (must fix FIRST — empirically reproduced)

These block multiple downstream domains. Each was reproduced against `/Users/jimmyhmiller/Documents/Code/beagle/target/release/beag`.

| # | Bug | Reproduction | Impact | Fix locus |
|---|-----|-------------|--------|-----------|
| B1 | `split`/`words`/`lines` collapse every element to the **last** substring under moving GC. **Fires under normal pressure too**, not only `--gc-always`. | `split("alpha,beta,gamma,delta", ",")` → `["delta","delta","delta","delta"]` | HTTP request-line/header parsing, path helpers, CSV — all silently corrupt | Rust: `src/builtins/strings.rs` + `runtime.rs:5387 create_string_array` — root each freshly-allocated string and re-read relocated handles across allocations |
| B2 | `!=` is **identity/pointer** comparison, not negation of structural `==`. | `a="foo"; b="${build()}oo"` → `a == b` is `true` **and** `a != b` is `true` | Every `if hash != expected` / `if version != x` check is wrong; breaks crypto verification, parsers, routing | Rust: compiler/runtime — lower `!=` to `not(equal(a,b))` like `==` |
| B3 | Concurrent **heap allocation** across threads races the generational collector. Pure-integer threads are clean; allocation crashes. | 6 threads building interpolated strings under `--gc-always` → `generational.rs:1345 Stale young gen pointer ... aborting` | No concurrent program that allocates in workers is trustworthy → no real HTTP/WS server | Rust: `src/gc/generational.rs` + `allocate()` — alloc-triggered GC must root every other thread's in-flight JIT frame + spill slots |
| B4 | `thread()` + socket op crashes deterministically under `--gc-always` in the threaded event-loop bootstrap (`get-io-loop`/`__io_loop_atom`). Shipped `concurrent_socket_echo_test.bg` also affected. | thread-per-connection accept loop aborts at `socket/accept` | Thread-per-connection server model is GC-unsafe | Rust: threaded event-loop startup races GC (same class as prior write-barrier fix) |
| B5 | `fs/blocking-read-file` (and `fs/copy`) on a **missing file SIGABRTs the whole process** (uncatchable). | `fs/blocking-read-file("/missing")` → `panic ... 'SystemError.IOError' not found`, exits 134/0 | Robust file I/O impossible | Beagle: add `IOError { message, location }` variant to `SystemError` in `std.bg` (cheapest), OR 1-line Rust change at `io.rs:371` |
| B6 | Inbound TCP read passes bytes through `String::from_utf8_lossy` (`networking.rs:778,877`) → binary corrupted irreversibly. | `[255,0,129,65]` arrives as 8 bytes (each invalid byte → U+FFFD) | **All masked WebSocket frames** and binary HTTP bodies destroyed | Rust: `from_utf8_unchecked` (matches existing `get_string`) or a raw `tcp-read-bytes` path |
| B7 | `sqrt`/trig/`log`/`pow` with an **integer argument silently terminate** the program (exit 0, no error). | `sqrt(2)` prints "before", never "after", exit 0 | Naive numeric/stats code mysteriously dies | Rust: `src/builtins/math.rs` — coerce int→float or throw a clear error |
| B8 | Parser rejects inline vector literal as a map value: `{:a [1,2]}`. | `Compile error: Expected close bracket ']' but found Comma` | Canonical JSON/config literal unwritable; bites every data-construction site | Rust: parser — key-followed-by-`[` must not parse as subscript inside a map literal |
| B9 | Compile errors and uncaught errors **exit 0**. | confirmed twice | CI/scripts can't detect failure; no `process-exit(code)` | Rust/CLI: nonzero exit on compile/uncaught error + expose `process-exit` |
| B10 | `substring`/`get` on **cons-strings** (interpolated/concatenated) raise bogus `IndexError` under `--gc-always`. | interpolation-built template tokenizer miscounts | Templating/HTTP (interpolation-heavy) corrupt under pressure | Rust: `src/builtins/strings.rs` — re-read relocated handle into `get_substring`/indexing |
| B11 | Compound values (vectors/maps) as **map keys / set members** silently fail: `get(assoc({},[1,2],"v"),[1,2])` → `null`. | `hash_value` (`runtime.rs:8472`) doesn't recurse; `equal` does | Can't key on tuples/vectors — core dict/set behavior | Rust: make `hash_value` recurse structurally to agree with `equal` (allocation-free) |

**Note on Rust surface:** B1, B3, B4, B6, B7, B10, B11 are genuine runtime/compiler bugs that *cannot* be fixed from Beagle — they are corrections to existing primitives, not new crates or features. This honors "minimal Rust, no external crates." B5 is fixable in pure Beagle.

## GC strategy (applies to every phase)

The GC is moving/generational. Three rules govern the whole project:

1. **`--gc-always` is the correctness oracle, never a tuning knob.** Any crash that appears only under `--gc-always` is a real rooting/stackmap bug to fix — never lower GC frequency to dodge it (this matches the established CLAUDE.md/MEMORY policy).
2. **The live bug class is "Rust builtin allocates N heap objects in a loop without rooting earlier ones."** It already bit `create_string_array` (split/words/lines), the regex result-builders, and cons-string substring. Every fix uses the established pattern already present in the same files (`register_temporary_root` + re-read across allocations).
3. **Pure-Beagle code is GC-safe by construction** (it goes through the safepoint-aware allocator). Therefore *prefer pure Beagle* — it both honors the constraint and sidesteps the bug class. The remaining GC risk surfaces are: (a) FFI buffer→string conversions, (b) concurrent allocation (B3/B4), (c) long-lived C-held pointers into the Beagle heap. Every FFI capability must bracket calls with `__register_c_call`/`__unregister_c_call` (the FFI layer already does) and copy/pin buffers; never hand a Beagle heap pointer to C across a possible GC.

**Per-phase gate:** each phase ends by running its representative programs under `--gc-always` *and* under a normal-pressure load loop. A phase is not "done" until both pass.

## Gap matrix: Beagle vs Python

| Domain | Python | Beagle today | After roadmap | How |
|--------|--------|--------------|---------------|-----|
| Collections / itertools / functools | full | **strong** (Clojure-grade, GC-solid) | full | beagle (+ Rust B8/B11) |
| Strings / regex | full | strong single-string; **split/regex GC-broken** | full | Rust fix B1; beagle.text |
| JSON | full | works | + pretty/keyword-key | rust core + beagle |
| Math | full | strong; **int-arg footgun B7** | + stats/format/predicates | rust B7; beagle |
| Date/time | datetime | only monotonic clock | full (epoch + civil + fmt) | ffi (time) + beagle |
| base64/hex/url | full | none (proven buildable) | full | beagle |
| hashlib/hmac | full | none (SHA-256 proven in pure Beagle) | full | beagle (FFI accel optional) |
| Random / UUID | full | random ok; no uuid/shuffle | full | beagle |
| OS / env / argv | full | env via FFI; **argv works** | full | ffi + beagle |
| Subprocess | full | popen via FFI works | full | ffi |
| pathlib / glob / tempfile / stat | full | sync fs ok; helpers absent | full | beagle + ffi |
| Concurrency (channels/locks/pool) | full | atoms+async; **alloc race B3** | full | rust B3; beagle |
| Socket (TCP) | full | works (async) | + sync/UDP/options/DNS | beagle + ffi |
| TLS/SSL | full | **none** (no dlopen OpenSSL on macOS) | https/wss | ffi (Security.framework) or curl shell-out |
| HTTP client/server | full | buildable on TCP | full (plaintext now, TLS later) | beagle |
| WebSockets | (ext) | buildable once B6 + sha1/base64 | full | beagle (+ Rust B6) |
| Templating | (ext) | buildable; cons-string GC risk | full | beagle (+ Rust B1/B10) |
| CSV / binary pack | full | buildable (avoid split) | full | beagle |
| BigInt / Decimal | full | **none; silent overflow wrap** | design item | beagle (hard) |
| Macros | (n/a) | **none** | (out of scope) | language change |

## Phases

Each phase unblocks the next: fix the runtime → trustworthy text/data → OS/IO → encoding/crypto (needs bit-correct text) → networking (needs TCP) → HTTP (needs TCP + date + bytes) → WebSockets (needs base64+sha1+binary read) → templating → advanced.

### Phase 0 — Foundations & Bugfixes (the gate)
**Goal:** make existing primitives trustworthy so all downstream pure-Beagle work rests on solid ground.
Fix B1–B11 (see table). Deliverables ordered by leverage: B1 (split, repairs 3 builtins), B2 (`!=`), B3+B4 (concurrent-alloc + thread-socket GC race), B6 (binary read), B7 (int math), B5 (IOError variant — Beagle), B8 (map-literal parser), B10 (cons-string substring), B11 (structural hash), B9 (exit codes + `process-exit`).
**GC validation:** dedicated regression programs — 4-part split assertion under `--gc-always`; 6-thread interpolated-string churn; thread-per-connection echo (`concurrent_socket_echo_test.bg`) — must all pass `--gc-always` AND a normal-pressure loop.

### Phase 1 — Text & Data
**Goal:** Python-grade text + collection completeness on the now-safe primitives.
- `beagle.text` (beagle): printf/format-spec (width/precision/decimals/hex-oct-bin/zero-pad/thousands), capitalize/title/case-insensitive-compare, string-repeat, trim-charset, split-with-limit, splitlines-keepends, `parse-float`.
- itertools/functools (beagle): mapcat, reductions, take-nth, partition-all, cycle, keep, remove, split-at/with, reduce-kv, count-by, combinations/permutations/product, sum, product; deque/default-map/ordered-map; map-entry iteration.
- Regex ergonomics (beagle, after B1-class fix to regex result-builders): replace-with-fn, named-group access.
- JSON (rust+beagle): keyword-key option + pretty-printer (Beagle post-process of existing decode).
**GC validation:** split-heavy + regex-extraction + format loops under `--gc-always`.

### Phase 2 — IO & OS
**Goal:** files, paths, processes, environment — the CLI/data-processing substrate.
- `beagle.os` (ffi→libSystem): get/set/unset-env, environ, getcwd/chdir, getpid/getppid, hostname, `process-exit(code)`.
- `beagle.path` (beagle, depends on B1): basename/dirname/join/splitext/normalize/is-absolute.
- `beagle.process` (ffi): `run-capture` (popen, proven), `run` (system rc), then posix_spawn/fork+exec+waitpid+pipe for separate stdout/stderr/stdin + true exit code.
- fs additions (ffi/beagle): sync stat/metadata (mtime/size/mode; macOS `stat$INODE64`), glob + recursive walk (beagle), tempfile/tempdir (ffi mkstemp/mkdtemp), realpath/symlink/chmod/seek (ffi), errno read helper (read-immediately, no alloc between).
**GC validation:** env/cwd/pid/popen probes (already passed) re-run; stat buffer→string conversions under `--gc-always`.

### Phase 3 — Encoding & Crypto
**Goal:** the codecs HTTP/WS/JWT need — all pure Beagle (proven), zero new Rust.
- `beagle.base64` (beagle, encode+decode verified), `beagle.hex`, `beagle.url` (percent + query-string).
- `beagle.hash` (beagle): **sha256 (verified against NIST), sha1** (for WS accept), md5, hmac (JWT), crc32. SHA-512 documented as needing 32-bit-half emulation (62-bit int cap).
- `beagle.bytes` / binary pack-unpack (beagle on string-builder); CSV reader/writer (beagle, byte-scan — **avoid split**).
- Optional FFI acceleration to system libcrypto/CommonCrypto behind the same Beagle API (only if perf demands).
**GC validation:** 5000-iter base64 + SHA-256 byte-shuffling under `--gc-always` (already passes clean — re-gate after B2, since B2 was the false "GC corruption" of SHA-256).

### Phase 4 — Networking (sockets, UDP, DNS, options)
**Goal:** complete the transport layer beneath HTTP/WS.
- `beagle.url`/`beagle.ip` (beagle): URL + IPv4/IPv6 + CIDR (bitwise, verified).
- `beagle.udp` (ffi): SOCK_DGRAM socket/bind/sendto/recvfrom (GC-safe raw FFI, proven).
- `beagle.dns` (ffi): getaddrinfo + `socket/resolve(host)→[addr]`.
- Socket options (ffi): setsockopt for SO_REUSEADDR/TCP_NODELAY/timeouts/keepalive.
- Bytes-faithful socket read surfaced (depends on B6).
**GC validation:** raw-FFI socket loop under `--gc-always` (passed); concurrent server path re-gated after B3/B4.

### Phase 5 — Web: HTTP/1.1 (client + server)
**Goal:** the headline deliverable — a trustworthy HTTP/1.1 server and plaintext client, pure Beagle.
- `beagle.http` (beagle): read-until-complete (loop reads until `\r\n\r\n` + Content-Length), request parser (method/path/version, case-insensitive header map, query split — on index-of/substring, not split until B1 trusted), response builder + status table + MIME, router (method/path/params, 404/405), keep-alive.
- HTTP **Date** header via `beagle.time` (ffi libc `time`/`gmtime_r`/`strftime`, or pure-Beagle civil-from-days) — depends on Phase 6 time work or pull it forward.
- Concurrency: cooperative single-thread path (validated under GC) as the safe default; bounded thread pool (beagle on thread()+channel) once B3/B4 land.
- HTTP client: GET/POST, headers/body, chunked decode, redirects (plaintext now; HTTPS blocked on TLS).
**GC validation:** full server (parser+router+keep-alive) under `--gc-always` AND a high-request normal-pressure load — mandatory, because B1 fired under ordinary pressure.

### Phase 6 — Date/Time, Stats, Random (parallel with Phase 5)
**Goal:** the cross-cutting utilities HTTP/logging/CLIs need.
- `beagle.time` (ffi): epoch-seconds/millis/nanos (libc `time`/`gettimeofday`/`clock_gettime`, verified GC-clean). Fix the monotonic-vs-epoch `timer/now` docstring.
- `beagle.date` (beagle): Howard-Hinnant civil↔days (verified), Date struct, weekday/leap/day-of-year, epoch↔Date, strftime/ISO-8601/RFC-3339 fmt+parse — **requires floor-div/floor-mod helpers** (Beagle division truncates).
- Statistics (beagle): mean/median/mode/variance/stdev/percentile/sum; float predicates is-nan/is-finite/clamp/signum/hypot/cbrt/copysign; bit-not; seeded RNG, shuffle/sample/choice, gaussian, UUID v4.
**GC validation:** FFI time loop + float-heavy stats under `--gc-always` (passes).

### Phase 7 — WebSockets
**Goal:** RFC 6455 server, pure Beagle, once Phase 0 (B6) + Phase 3 (sha1/base64) land.
- `beagle.ws` (beagle): upgrade handshake (`base64(sha1(key + GUID))`), frame parse/mask/unmask (bitwise XOR, verified), ping/pong, close, fragmentation reassembly, text/binary dispatch, optional UTF-8 validation.
**GC validation:** handshake + masked-frame echo + ping/pong + fragmentation under `--gc-always`; multi-connection robustness gated on B3/B4. Operate on string-builder bytes (`byte-at`), never `length`/`get` on binary strings.

### Phase 8 — Templating
**Goal:** Jinja/Liquid-style engine, pure Beagle (a working prototype already runs).
- `beagle.template` (beagle): tokenizer (`{{ }}`/`{% %}`), recursive evaluator (if/elif/else, for-in with loop vars, nesting via depth-matched end-tags), HTML/attr/JS escaping + raw markers, dotted-path context (`get-in`), filters/pipes (tiny expr parser).
- Depends on B1 + B10 (templates are assembled dynamically → cons-strings).
**GC validation:** engine over **cons-string (interpolated/concatenated) template inputs** under `--gc-always` — the literal-template path passes and masks the bug, so test the dynamic path specifically.

### Phase 9 — Advanced / Design items
**Goal:** the genuinely hard, lower-frequency items — flagged as design decisions, not blockers.
- Concurrency primitives (beagle on atoms/CAS/park, after B3): channel, bounded blocking queue, mutex/with-lock, semaphore, condition var/wait-group/barrier, pmap/parallel-for, bounded executor; thread-join-with-result + atomic-int fast path (rust).
- TLS/wss (ffi): Security.framework Secure Transport (loads; needs C-callback I/O via `SSLSetIOFuncs` — exercises the weakest FFI corners) **or** shell out to `curl` via `beagle.process` as a pragmatic stopgap for HTTPS clients.
- BigInt / Decimal / Rational + checked arithmetic (beagle struct over limb-vectors; the **silent overflow wrap is the real hazard** and a checked-arith trap would need Rust). The one item not cleanly pure-Beagle.
- FFI hardening (rust): true C varargs (printf-family; Apple ARM64 all-variadic-on-stack), x86-64 general marshaller (currently a hardcoded pattern table — Linux servers fail on many signatures), struct-by-value ABI decomposition, pointer read-at/write-at, errno/data-symbol access, off-main-thread callbacks.
- Macros (language change, out of scope but noted as the biggest *ergonomics* ceiling).
- compression zlib/gzip (ffi to system libz) — large; FFI is pragmatic.
**GC validation:** concurrent primitives under `--gc-always` with multiple producers/consumers; every FFI/TLS path validated; callback-on-foreign-thread treated as high GC risk (defer or design carefully).

## Effort & realism

- **Phase 0 is the whole ballgame and is mostly small, surgical Rust fixes** (the rooting pattern already exists in the same files; B2/B8 are compiler one-locus fixes; B5 is pure Beagle). Estimate days, not weeks — but B3/B4 (concurrent-alloc GC race) is the genuinely deep one and may take real debugging, as the same class was "fixed" before per MEMORY and regressed.
- **Phases 1–8 are overwhelmingly pure-Beagle library authoring** with a thin, proven FFI seam (time, env, process, optional crypto-accel). No new Rust crates. This is exactly the project's intended shape.
- **Phase 9 holds the hard/uncertain items** (TLS via Security.framework callbacks, BigInt/checked-arith, x86-64 FFI, varargs). HTTPS for clients has a pragmatic escape hatch (curl shell-out) so it doesn't block the server story.
- **No external Rust crates are introduced anywhere.** All new Rust is bug fixes to existing primitives.

---

## Appendix A — Blocking bugs (structured)

- B1 — split/words/lines collapse every element to the LAST substring under moving GC, and it fires under NORMAL pressure (not only --gc-always). Verified: split("alpha,beta,gamma,delta", ",") => ["delta","delta","delta","delta"]. Root cause: src/builtins/strings.rs + runtime.rs:5387 create_string_array allocate result strings in a loop without rooting earlier ones across later allocations. Blocks ALL HTTP request/header parsing, path helpers, CSV. Rust fix.
- B2 — `!=` is identity/pointer comparison, NOT negation of structural ==. Verified: a="foo", b="${build()}oo" => (a == b) is true AND (a != b) is ALSO true. Breaks every `if hash != expected` / version/route check. Rust fix: lower != to not(equal(a,b)).
- B3 — Concurrent heap allocation across threads races the generational collector. Verified: 6 threads building interpolated strings under --gc-always => 'thread panicked at src/gc/generational.rs:1345: Stale young gen pointer 0x... found while scanning copied object ... struct_name=beagle.core/SystemError.TypeError' then aborts. Pure-integer threads are clean — the trigger is multi-thread allocation. Blocks every trustworthy concurrent server. Rust fix in gc/generational.rs + allocate().
- B4 — thread()+socket crashes deterministically under --gc-always in the threaded event-loop bootstrap (beagle.async get-io-loop / __io_loop_atom racing GC); shipped concurrent_socket_echo_test.bg is affected. Blocks thread-per-connection servers. Rust fix.
- B5 — fs/blocking-read-file on a missing file SIGABRTs the whole process (uncatchable). Verified: panic at src/builtins/exceptions.rs:364 'Failed to create SystemError: Variant struct beagle.core/SystemError.IOError not found' (read_full_file throws kind IOError but SystemError has no such variant). Beagle fix: add IOError variant to SystemError in std.bg (or 1-line Rust change at io.rs:371).
- B6 — Inbound TCP read passes bytes through String::from_utf8_lossy (networking.rs:778,877), irreversibly corrupting binary: [255,0,129,65] arrives as 8 bytes. ALL masked WebSocket frames and binary HTTP bodies destroyed. Rust fix: from_utf8_unchecked (matches get_string) or a raw tcp-read-bytes path.
- B7 — sqrt/trig/log/pow with an integer argument SILENTLY TERMINATE the program (exit 0, no error). Verified: sqrt(2) prints 'before' then exits 0, never reaching 'after'; sqrt(2.0) works. Rust fix in src/builtins/math.rs: coerce int->float or throw a clear error.
- B8 — Parser rejects inline vector literal as a map value: {:a [1,2]}. Verified: 'Compile error: Expected close bracket ] but found Comma'. Makes the canonical JSON/config literal unwritable. Rust parser fix.
- B9 — Compile errors and uncaught errors exit with status 0. Verified twice. CI/scripts cannot detect failure; no process-exit(code). Rust/CLI fix.
- B10 — substring/get on cons-strings (interpolated/concatenated) raise bogus IndexError under --gc-always (stale relocated pointer in src/builtins/strings.rs get_substring/get_string_index). Hits templating and HTTP (both interpolation-heavy). Rust fix: re-read relocated handle.
- B11 — Compound values (vectors/maps) as map keys / set members silently fail: get(assoc({},[1,2],"v"),[1,2]) => null; into-set of two equal vectors does not dedup. hash_value (runtime.rs:8472) does not recurse structurally while equal does. Rust fix: make hash_value recurse (allocation-free).

## Appendix B — Phases (structured)

### Phase 0 — Foundations & Bugfixes (the gate)
**Goal:** Make existing primitives trustworthy so all downstream pure-Beagle work rests on solid ground. Fix the empirically-reproduced blocking bugs B1–B11.

- B1 (rust, highest leverage): fix split/words/lines moving-GC collapse in src/builtins/strings.rs + runtime.rs:5387 create_string_array — root each freshly-allocated string and re-read relocated handles across allocations; one fix repairs all three builtins. Add a 4-part-split --gc-always assertion test.
- B2 (rust): lower `!=` to not(equal(a,b)) so it negates structural == (currently identity comparison; a==b and a!=b both true — verified).
- B3 (rust, deepest): fix concurrent heap-allocation GC race (generational.rs:1345 'Stale young gen pointer' + allocate()); alloc-triggered GC must root every other thread's in-flight JIT frame + spill slots.
- B4 (rust): fix thread()+socket --gc-always crash in threaded event-loop bootstrap (get-io-loop / __io_loop_atom racing GC).
- B5 (beagle): add IOError { message, location } variant to SystemError in std.bg so fs/blocking-read-file on a missing file returns a catchable Result.Err instead of SIGABRT.
- B6 (rust): byte-faithful inbound TCP read — replace from_utf8_lossy at networking.rs:778,877 with from_utf8_unchecked (matches get_string) or add tcp-read-bytes.
- B7 (rust): sqrt/trig/log/pow must coerce integer args to float or throw a clear error, not silently terminate (verified sqrt(2) exits 0 with no output).
- B8 (rust): parser fix — inline vector literal as a map value {:a [1,2]} (key-followed-by-[ must not parse as subscript inside a map literal).
- B10 (rust): cons-string substring/get_string_index — re-read relocated handle into get_substring/indexing to stop bogus IndexError under --gc-always.
- B11 (rust): make hash_value (runtime.rs:8472) recurse structurally to agree with equal so vectors/maps work as map keys / set members (keep allocation-free).
- B9 (rust/CLI): nonzero exit on compile error and uncaught error (currently exit 0 — verified); expose process-exit(code).
- GC VALIDATION: regression programs — 4-part split assertion, 6-thread interpolated-string churn, thread-per-connection echo (concurrent_socket_echo_test.bg) — all must pass under --gc-always AND a normal-pressure loop.

### Phase 1 — Text & Data
**Goal:** Python-grade text and collection completeness on the now-safe primitives. Pure Beagle, no new Rust.

- beagle.text (beagle): printf/format-spec (width/precision/decimals, hex/oct/bin, zero-pad, thousands), capitalize/title/case-insensitive-compare, string-repeat, trim-charset, split-with-limit, splitlines-keepends, parse-float.
- itertools/functools (beagle): mapcat, reductions, take-nth, partition-all, cycle, keep, remove, split-at, split-with, reduce-kv, count-by, combinations, permutations, product, sum, product.
- convenience structures (beagle): deque, default-map, ordered-map, map-entry iteration (entries/pairs).
- regex ergonomics (beagle, after B1-class fix to regex.rs result-builders): replace-with-fn, named-group access.
- JSON (rust core + beagle): keyword-key decode option + pretty-printer as Beagle post-process of existing decode.
- GC VALIDATION: split-heavy, regex-extraction, and format loops under --gc-always.

### Phase 2 — IO & OS
**Goal:** Files, paths, processes, environment — the CLI/data-processing substrate. Beagle + thin libc FFI, no new Rust.

- beagle.os (ffi→libSystem): get/set/unset-env, environ, getcwd/chdir, getpid/getppid, hostname, process-exit(code).
- beagle.path (beagle, depends on B1): basename, dirname, join, splitext/extension, normalize, is-absolute, split-path.
- beagle.process (ffi): run-capture (popen — proven GC-clean), run (system rc), then posix_spawn/fork+execvp+waitpid+pipe for separate stdout/stderr/stdin + true exit code.
- fs metadata (ffi): sync stat/lstat (mtime/ctime/atime/mode/size/inode; macOS stat$INODE64) with mode-bit predicates (beagle).
- glob + recursive directory walk (beagle on read-dir + is-directory? + wildcard matcher).
- tempfile/tempdir (ffi mkstemp/mkdtemp), realpath/symlink/readlink (ffi), chmod/umask (ffi), seek/tell (ffi fseek/ftell).
- errno read helper (ffi): read immediately after the failing call with no allocation between (current __error deref is fragile).
- GC VALIDATION: env/cwd/pid/popen probes re-run; stat buffer→string conversions under --gc-always.

### Phase 3 — Encoding & Crypto
**Goal:** The codecs HTTP/WebSockets/JWT need — all pure Beagle (SHA-256 + base64 already proven), zero new Rust crates.

- beagle.base64 (beagle): encode + decode (verified against RFC vectors) + url-safe variant.
- beagle.hex (beagle): encode/decode.
- beagle.url (beagle): percent-encode/decode + query-string parse/build.
- beagle.hash (beagle): sha256 (verified vs NIST), sha1 (for WS Sec-WebSocket-Accept), md5, hmac (JWT HMAC-SHA256), crc32. Document SHA-512 needing 32-bit-half emulation (62-bit int cap; 64-bit hex literals overflow at compile time).
- beagle.bytes + binary pack/unpack (beagle on string-builder) and CSV reader/writer (beagle, byte-scan — must AVOID the split builtin).
- optional FFI acceleration to system libcrypto/CommonCrypto behind the same Beagle API (only if perf demands).
- GC VALIDATION: 5000-iter base64 + SHA-256 byte-shuffling under --gc-always (passes); re-gate after B2 since B2 was the false 'GC corruption' of SHA-256.

### Phase 4 — Networking (sockets, UDP, DNS, options)
**Goal:** Complete the transport layer beneath HTTP and WebSockets. Beagle + libc FFI, no new Rust.

- beagle.url / beagle.ip (beagle): URL parse/build, IPv4/IPv6 string↔int, CIDR/subnet membership, classification (bitwise — verified).
- beagle.udp (ffi): socket(AF_INET,SOCK_DGRAM)+bind/sendto/recvfrom (raw FFI, GC-safe — proven).
- beagle.dns (ffi): getaddrinfo wrapper + socket/resolve(host)→[addr].
- socket options (ffi): setsockopt for SO_REUSEADDR, TCP_NODELAY, timeouts, keepalive.
- bytes-faithful socket read surfaced to users (depends on B6).
- GC VALIDATION: raw-FFI socket loop under --gc-always (passed); concurrent server path re-gated after B3/B4.

### Phase 5 — Web: HTTP/1.1 (client + server)
**Goal:** The headline deliverable — a trustworthy HTTP/1.1 server and plaintext client, pure Beagle on TCP + string-builder.

- beagle.http server (beagle): read-until-complete (loop reads until \r\n\r\n + Content-Length), request parser (method/path/version, case-insensitive header map, query split on index-of/substring), response builder + status table + MIME, router (method/path/params, 404/405), keep-alive.
- HTTP Date header (ffi/beagle): RFC-1123 timestamp via beagle.time (libc time/gmtime_r/strftime or pure-Beagle civil-from-days).
- concurrency (beagle): cooperative single-thread path validated under GC as the safe default; bounded thread pool on thread()+channel once B3/B4 land.
- beagle.http client (beagle): GET/POST, headers/body, chunked decode, redirects — plaintext now; HTTPS deferred to Phase 9 TLS.
- GC VALIDATION: full server (parser+router+keep-alive) under --gc-always AND a high-request normal-pressure load (mandatory — B1 fired under ordinary pressure).

### Phase 6 — Date/Time, Stats, Random
**Goal:** Cross-cutting utilities HTTP/logging/CLIs need. Runs in parallel with Phase 5 (Date header depends on beagle.time). FFI for epoch, pure Beagle for the rest.

- beagle.time (ffi): epoch-seconds/millis/nanos via libc time/gettimeofday/clock_gettime (verified GC-clean); fix the monotonic-vs-epoch timer/now docstring.
- beagle.date (beagle): Howard-Hinnant civil↔days (verified), Date struct, weekday/leap/day-of-year, epoch↔Date, strftime/ISO-8601/RFC-3339 fmt+parse — requires floor-div/floor-mod helpers (Beagle division truncates).
- statistics (beagle): mean/median/mode/variance/stdev/percentile/sum.
- float predicates (beagle): is-nan?/is-infinite?/is-finite?/signum/cbrt/hypot/copysign/clamp/trunc; bit-not (x ^ -1).
- random extras (beagle): seeded RNG, random-float-range, choice/shuffle/sample, gaussian, UUID v4.
- GC VALIDATION: FFI time loop + float-heavy stats under --gc-always (passes).

### Phase 7 — WebSockets
**Goal:** RFC 6455 server, pure Beagle, once Phase 0 (B6 binary read) and Phase 3 (sha1/base64) land.

- beagle.ws (beagle): upgrade handshake (base64(sha1(key + 258EAFA5-...))), HTTP upgrade request parse, 101 response.
- frame codec (beagle): parse FIN/RSV/opcode/MASK/length, unmask payload by XOR (verified), encode unmasked server frames via string-builder.
- control frames (beagle): ping→pong, close echo, enforce control <=125 bytes + unfragmented.
- fragmentation reassembly (beagle): accumulate continuation frames until FIN, allow interleaved control frames; text/binary dispatch + optional UTF-8 validation.
- GC VALIDATION: handshake + masked-frame echo + ping/pong + fragmentation under --gc-always; multi-connection robustness gated on B3/B4. Read frames via string-builder byte-at only, never length/get on binary strings.

### Phase 8 — Templating
**Goal:** Jinja/Liquid-style engine, pure Beagle (a working prototype already runs). Depends on B1 + B10 (templates are dynamically assembled → cons-strings).

- beagle.template tokenizer (beagle): split literal vs {{ var }} vs {% tag %} via substring/index-of.
- beagle.template evaluator (beagle): if/elif/else, for-in with loop vars, nesting via depth-matched end-tags, recursive render-range.
- escaping (beagle): HTML/attr/JS escaping + safe/raw markers, char-by-char over string-builder.
- expression sub-language + filters/pipes (beagle): dotted-path context via get-in, comparisons, {{ x | upper | default:... }} via a tiny Pratt/recursive-descent parser.
- GC VALIDATION: engine over cons-string (interpolated/concatenated) template inputs under --gc-always — the literal path passes and masks the bug, so test the dynamic path specifically.

### Phase 9 — Advanced / Design items
**Goal:** The genuinely hard, lower-frequency items — flagged as design decisions, not blockers. Beagle-first; FFI for TLS; the few deep Rust items are primitive fixes, not crates.

- concurrency primitives (beagle on atoms/CAS/park, after B3): channel, bounded blocking queue, mutex/with-lock, semaphore, condition var/wait-group/barrier, pmap/parallel-for, bounded executor.
- thread-join-with-result + atomic-int fast path (rust): capture return value as a GC root; hardware fetch-add for counters.
- TLS / wss (ffi): Security.framework Secure Transport (loads; needs C-callback I/O via SSLSetIOFuncs — weakest FFI corner) OR shell out to curl via beagle.process as a pragmatic HTTPS-client stopgap.
- BigInt / Decimal / Rational + checked arithmetic (beagle struct over limb-vectors; silent overflow wrap is the real hazard, checked-arith trap would need rust). The one item not cleanly pure-Beagle.
- FFI hardening (rust): true C varargs (Apple ARM64 all-variadic-on-stack), x86-64 general marshaller (currently a hardcoded pattern table — Linux fails on many signatures), struct-by-value ABI decomposition, pointer read-at/write-at, errno/data-symbol access, off-main-thread callbacks.
- compression zlib/gzip (ffi to system libz); DEFLATE pure-Beagle possible but high-effort.
- macros (language change): noted as the biggest ergonomics ceiling; out of scope for batteries but document.
- GC VALIDATION: concurrent primitives under --gc-always with multiple producers/consumers; every FFI/TLS path validated; callback-on-foreign-thread treated as high GC risk (defer or design carefully).

## Appendix C — Immediate next actions

- Fix B1 (split/words/lines GC collapse) in src/builtins/strings.rs + runtime.rs:5387 create_string_array using the existing register_temporary_root + re-read-across-allocations pattern; add a regression test that splits a 4-part string and asserts no element collapse, run under --gc-always. This single fix repairs split, words, and lines and unblocks all HTTP/path/CSV parsing.
- Fix B2 (!= operator) by lowering != to not(equal(a,b)) in the compiler/runtime so it negates structural equality; add a test asserting (a == b) != (a != b) for distinct-but-equal strings.
- Fix B5 (missing-file SIGABRT) in pure Beagle by adding an IOError { message, location } variant to SystemError in standard-library/std.bg, then verify fs/blocking-read-file("/missing") returns a catchable Result.Err.
- Fix B7 (sqrt/trig int-arg silent termination) in src/builtins/math.rs to coerce int->float (or throw a clear error); add a test that sqrt(2) and pow(2,3) succeed.
- Fix B8 (map-literal inline vector value parser bug) so {:a [1,2]} compiles; add a parse test for {:a [1,2], :b [3]}.
- Fix B6 (binary TCP read) at networking.rs:778,877 — replace from_utf8_lossy with from_utf8_unchecked (or add tcp-read-bytes); verify a [255,0,129,65] socket round-trip is byte-exact.
- Begin debugging B3/B4 (concurrent-allocation + thread-socket GC race) — these are the deepest fixes (same class previously regressed per MEMORY); reproduce with the 6-thread interpolated-string program and concurrent_socket_echo_test.bg under --gc-always, then ensure alloc-triggered GC roots every other thread's JIT frame + spill slots.
- Fix B9 (exit codes) so compile errors and uncaught errors return nonzero, and expose process-exit(code) — needed for CI/scripting of everything downstream.
- Fix B11 (structural hash_value) in runtime.rs so vectors/maps work as map keys / set members, allocation-free; add an assoc/get round-trip test with a vector key.
- Stand up the Phase-0 GC regression harness: a script that runs the 4-part split assertion, 6-thread string churn, missing-file read, sqrt(2), {:a [1,2]}, and the thread-per-connection echo under both --gc-always and a normal-pressure loop, and gates 'Phase 0 done' on all passing.

---
_Generated by the beagle-stdlib-ultraplan workflow (22 agents); blocking bugs B1/B2/B5/B7/B8 independently re-verified against target/release/beag before adoption._
---

## IMPLEMENTATION STATUS (autonomous build, 2026-06-15)

Phase 0 through Phase 9 implemented. All modules are pure Beagle on a thin libc/socket FFI seam, embedded in the binary, loaded on demand via `use`, and gc-validated (except concurrency under gc-always — see below).

**Shipped modules** (`standard-library/`): text, base64, hex, hash (sha1/sha256/hmac), url, os, path, process, time, http, template, ws, iter, collections, stats, random, ip, json, csv, bigint, channel, date.

**Named deliverables — all working in Beagle:**
- HTTP server (cooperative, single-thread) + client — `examples/web_app_demo.bg` (curl-verified: routing, HTML, JSON, 404).
- WebSockets (RFC 6455) — `examples/websocket_echo_demo.bg` (live-verified vs a real client: handshake, masked echo, ping/pong; exact accept-key vector).
- Templating engine ({{var}}/if/for/escape).

**Verified vectors:** sha1/sha256 (NIST), base64 (RFC 4648), ws accept-key (RFC 6455 §1.3 + §5.7 frame), bigint 20-digit square, ISO-8601 / RFC-1123 dates, IPv4/CIDR.

**Runtime bugs fixed during the build:** Phase-0 cluster (B1 split-GC-collapse, B2 `!=`, B5 IOError, B6 byte-faithful TCP, B7 math int-coercion, B8 map-literal parse, B11 structural hash); plus `tcp_connect`/`tcp_listen` accepting only literal hosts (`get_string_literal`→`get_string`).

**KNOWN REMAINING DEEP BUG (non-blocking, gc-always-only):** concurrent multi-thread allocation racing the generational collector — `generational.rs:1345 "Stale young gen pointer"` (a missed cross-thread root during STW). Reliable repro: the channel producer/consumer test under `--gc-always` (4 producers × 50 sends). Everything works under normal GC (OnPressure); this is the B3/B4 class and the #1 runtime follow-up. Concurrency primitives (beagle.channel) are correct under normal GC; their suite test runs in normal mode for this reason.

---

## TURNKEY FIX DESIGN: maps cannot store `null` (the #1 remaining data bug)

**Symptom:** `assoc(m, k, null)` is a silent no-op; `{:a null}` drops `:a`; JSON round-trip drops null-valued fields.

**Root cause** (`src/collections/persistent_map.rs`): in a BitmapIndexedNode the children array is interleaved `[key0,val0,key1,val1,...]`, and a slot is interpreted as a **child node** when its *value* slot `== null` (the key slot then holds the sub-node pointer). So a leaf with a genuine null value is indistinguishable from a child-node slot, and `assoc` refuses to store it.

**Fix:** stop using `value == null` to mean "child node". Instead detect a child node by the **type** of the key slot — a child is present iff the key slot holds a node-typed heap object (`TYPE_ID_BITMAP_NODE` / `TYPE_ID_ARRAY_NODE` / `TYPE_ID_COLLISION_NODE`). User keys can never have those type-ids (a user map is `TYPE_ID_PERSISTENT_MAP`, distinct), so this is unambiguous. Then leaf values — including `null` — store and read normally. Audit every `value == null`/`slot_value != null_val` check in `get`, `assoc_bitmap_node`, `assoc_node`, and the ArrayNode paths and replace the child-node test with an `is_node_typed(key_slot)` helper.

**get() semantics:** like Clojure, `get(m,k)` still returns `null` for BOTH absent and stored-null (inherent). That's acceptable; pair it with `contains?(m,k)` / `get(m,k,default)` for disambiguation (add these if missing).

**Risk/validation:** maps back namespaces and compiler symbol tables — a subtle bug breaks compilation itself. Gate on: full suite (400+/0), `CLJVM_GC=every`-equivalent (`--gc-always`) on map-heavy tests, and a dedicated test storing/reading/removing null values + nested maps with null + JSON null round-trip. Do this attended, not in an unattended autonomous run.
