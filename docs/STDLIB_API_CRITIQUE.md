# Beagle Stdlib API Design Critique & Recommendations

To: the author of Beagle
From: synthesis of 8 area critiques (collections, strings/format, data-formats, IO, async, networking, numeric/time, cross-cutting)

---

## 1. Executive summary

Beagle has the right protocol *vocabulary* (Seqable/Seq/Indexed/Associable/Push/Length/Contains/Format/Writer) but has not finished *committing to it*. Across all eight areas the same failure recurs: **a protocol exists, but new types and operations route around it via concrete per-type functions or `instance-of` type-switches.** That single structural problem generates most of the individual findings. The five dominant themes:

1. **The protocols are under-extended, so polymorphism stops at the door.** `into` is a hand-rolled `if set?/instance-of(PersistentMap)/else push` switch (std.bg:1798); json serialization is a 7-way `instance-of` cascade (beagle.json:589); every custom container (Deque/OrderedMap/DefaultMap/PriorityQueue) ships `dm-get`/`om-assoc`/`pq-count` instead of extending Indexed/Associable/Length; stream/map reimplements Seq. The protocol surface is right; the *extensions* are missing.

2. **Missing foundational protocols force whole feature classes to be impossible or hand-rolled.** There is no `Conj`, no polymorphic `Comparable`/`compare`, no user-extensible `Eq`/`Hash` (the runtime's own `equal` carries a `// TODO: Make this pluggeable` at src/runtime.rs:8553), no `Closeable`, no numeric tower. Consequences: BigInt can't use `+` or `sort`; user structs can't be map keys or define value equality; files can't be `with-open`'d; SemVer/DateTime need bespoke comparators.

3. **Two parallel display engines that silently disagree.** `println(x)` dispatches through `Format`, but `${x}` interpolation and `to-string(x)` go through a *separate* Rust `get_repr_inner` (src/runtime.rs:7924) that never calls `Format`. Extend `Format` on your type and `println(x)` honors it while `"${x}"` does not — same value, two outputs. This quietly defeats the one display protocol you have.

4. **Argument-order and naming drift contradict your own freshly-set conventions.** You standardized seq HOFs as function-first (`map(f, coll)`, chain with `|>>`) — correctly, it's the single most important thing to get right and core gets it right — but `join(coll, sep)`, every `stream/*` combinator (`stream/map(stream, f)`), and `stream/reduce(stream, init, f)` are collection-first, breaking `|>>` and contradicting Clojure. `?`-predicates return wrapped Results (`fs/exists?` → `Result.Ok{value:true}`, so `if exists?(p)` is always true), and `!`-mutators are unmarked (`record-pass`, `st-set-option`).

5. **No coherent error policy.** Three disjoint channels — `Result.Ok/Err`, thrown `SystemError`/`Error` enums, and thrown bare strings — with no bridge, no `ex-info`/`ex-data` carrier, and per-namespace inconsistency (fs returns Result, os mixes throw/null/bool, glob swallows to `[]`). `unwrap(Err)` prints-and-returns-null (std.bg:3177), the exact silent-failure your own CLAUDE.md forbids.

The protocol story (themes 1–3) is the high-leverage work; it dissolves dozens of individual findings. Start there.

---

## 2. Protocol architecture — the highest-value work

The unifying principle, stated once: **for any operation that today is spelled as N concrete per-type functions or an `instance-of` ladder, there should be ONE protocol method that types `extend` into.** Clojure's entire core library is consumable over any `deftype` because that type implements `Counted`/`ILookup`/`Associative`/`Seqable`/`IFn`. Beagle should hold the same line.

### 2.1 `Conj` — the polymorphic add (dissolves the `into` type-switch)

The single highest-value change. Adding one element is currently split three ways (`push` vectors-only, `set-add`, `assoc` maps), and `Push` is extended only to PersistentVector. So `into` is a type-switch and every HOF hardcodes `push(acc, …)` → always returns a vector.

```
protocol Conj { fn conj(coll, x) }
extend PersistentVector with Conj { fn conj(v, x) { push(v, x) } }       // append
extend PersistentSet   with Conj { fn conj(s, x) { set-add(s, x) } }     // add
extend PersistentMap   with Conj { fn conj(m, kv) { assoc(m, kv[0], kv[1]) } }  // [k v]
extend Deque           with Conj { fn conj(d, x) { push-back(d, x) } }
```
Then `into` collapses to its definition: `fn into(to, from) { reduce(conj, to, from) }` — zero type-switch — and custom containers become first-class accumulation targets. **Grounding:** `clojure.core/conj` is THE polymorphic add and `(into to from)` is literally `(reduce conj to from)`. Rust models the same idea as the `Extend` trait; Elixir as the `Collectable` protocol.

### 2.2 Seq-coercing `first`/`rest`/`next` (delete `first-of`)

`Seq.first/next` operate only on a live seq, so `first([1,2,3])` fails and an undiscoverable `first-of` exists solely to call `seq` first (its docstring admits it). Demote the raw accessors and make the public trio coerce:

```
protocol Seq { fn -first(s)  fn -next(s) }     // internal, dashed
fn first(coll) { let s = seq(coll); if s == null { null } else { -first(s) } }
fn next(coll)  { ... }   fn rest(coll) { ... }
```
Delete `first-of`; `min-of`/`max-of`/`last` collapse onto the trio. **Grounding:** `clojure.core/first` calls `seq` internally; ClojureScript exposes the raw method as `ISeq.-first` and keeps `first` as the public seq-calling fn. Rust's `Iterator::next` is the trait method; `.nth(0)` is the convenience.

### 2.3 `Comparable` — one three-way `compare` (dissolves per-type comparators)

`sort`/`min`/`max` are hardwired to `<` (primitives only), so SemVer/BigInt/DateTime each ship a bespoke comparator and `semver` even hand-rolls `cmp-int`/`cmp-string` (= `compare` on primitives). Also unifies the two clashing comparator shapes — `sort-with(less)` (boolean) vs `pq-new-with(compare)` (3-way).

```
protocol Comparable { fn compare(a, b) }   // -> -1 | 0 | 1
// default impl over primitives folds in cmp-int/cmp-string
extend BigInt   with Comparable { ... }
extend SemVer   with Comparable { ... }
extend DateTime with Comparable { ... }
```
Route `sort`/`sort-by`/`min-of`/`max-of` through it; keep `sort-with-pred(less, coll)` as the boolean-predicate variant (Clojure's `sort` accepts both). Expose one core `compare`, delete `priorityqueue/default-compare`. **Grounding:** `clojure.core/compare` is the single 3-way comparator backing `sort`; Rust is `Ord`/`PartialOrd` (`#[derive(Ord)]`, `slice::sort` requires `Ord`).

### 2.4 `Eq` + `Hashable` — user-extensible value semantics

`Runtime::equal` is a fixed Rust match (with its own pluggability TODO) and there is no value `hash`, so user structs cannot define equality (a `Money` comparing only cents) nor be PersistentMap/Set keys with custom semantics. `date/dt-equal?` exists precisely because "structs don't compare field-wise with `==`."

```
protocol Eq       { fn =(a, b) }      // default impl = current field-wise compare
protocol Hashable { fn hash(x) }      // map/set key hashing dispatches here
```
The runtime `equal` calls `Eq` for HeapObjects; collection key-hashing calls `Hashable`; document the equals/hash invariant. `dt-equal?` becomes `=`. **Grounding:** ClojureScript's `IEquiv`/`IHash`; Rust's `Eq`+`Hash` with the explicit rule that they must agree — the exact invariant you currently can't let users uphold.

### 2.5 A Display/Show protocol unifying `to-string` / `format` / `${...}` / `print`

Make `to-string` and interpolation dispatch through `Format` (have the `to_string` builtin / `get_repr_inner` call back into the protocol for HeapObjects), so `${x}`, `to-string(x)`, and `println(x)` can never disagree. Then `extend BigInt/SemVer/DateTime/W64 with Format` and delete `bigint/to-string-any` (the protocol already handles the String-vs-other case). **Grounding:** Clojure has exactly one extension point, the `print-method` multimethod — `str`, `pr`, `println`, `format`'s `%s` all funnel through it. Rust: `Display`/`Debug` are traits and `format!`/`.to_string()` always go through the impl, never a hardcoded fallback that shadows it.

### 2.6 `ToJson` / `Codec` — extensible serialization (kills the json type-switch)

`json/write-value` is a closed 7-way `instance-of` cascade terminating in `throw("unsupported type")` — user structs/enums *cannot* be encoded and there is no hook.

```
protocol ToJson { fn write-json(value, out, indent, depth) }
extend String / Int / Float / PersistentVector / PersistentMap / Keyword with ToJson { ... }
// stringify(v) = write-json(v, out, -1, 0); users extend their own types
```
The catch-all `throw` becomes a fixable missing-impl error. **Grounding:** this is exactly `clojure.data.json`'s `JSONWriter` protocol (`-write [object out options]`, extended via `extend-protocol`); Rust serde's `Serialize` trait. More broadly, unify the four divergent verb pairs (`parse`/`stringify` vs `parse`/`write` vs `encode`/`decode` vs `pack`/`unpack`) — Clojure reuses one `read`/`write` across data.json/edn/transit.

### 2.7 `Closeable` + `with-open`, and wire files into `Writer`/`Reader`

Resource management is entirely manual (`io/close` everywhere; a throw between open and close leaks the FILE*). And `io/File` ignores the *existing* `Writer` protocol, so a file can't back `*out*` and you instead get `write-stdout`/`write-stderr`/`write-string` as concrete per-target fns.

```
protocol Closeable { fn close(self) }          // + with-open(f = open(...)) { ... } bracket
extend File with Writer { fn write(self, s) { ... } }
protocol Reader { fn read-bytes(self, n)  fn read-line(self) }   // File, Stdin
```
Collapses `write-stdout`/`write-stderr`/`write-string` into one `write`, and `read-stdin*`/`read-line` into one `read-line`; Stdin/Stdout/Stderr become *values*, not name prefixes. **Grounding:** `clojure.core/with-open` + `java.io.Closeable`; the IOFactory protocol (`make-reader`/`make-writer`) so the same `slurp`/`spit`/`copy` work over files, sockets, URLs. Rust unifies on `Read`/`Write` + `Drop`; Python on context managers.

### 2.8 A numeric/arithmetic protocol so BigInt joins the tower

Every BigInt op is a separate concrete fn accepting only BigInt; `+`/`<`/`sort` don't work, so `factorial` detects overflow and tells the user to *switch modules and rewrite the call site*.

```
protocol Numeric { fn add(a, b)  fn mul(a, b)  fn neg(a) }   // + promotion rule Int->BigInt
```
Ideal: make core `+`/`*` overflow-promote Int→BigInt automatically (Clojure's `+'`/`*'`) so `factorial`/`pow-int` never bail. Add `bigint/to-int` for narrowing. **Grounding:** Clojure's numeric tower (`clojure.lang.Numbers`) makes `+` uniform across Long/BigInt/BigDecimal; `+'`/`*'` auto-promote. Rust: `Add`/`Mul` traits + `num::BigInt`; Python: transparent int→arbitrary-precision.

### 2.9 Stream → Seqable, and a port protocol for channels

`StreamResult.Value/Done` + `source-next` *is* `Seq.first/next` with an Error arm, yet Stream implements neither Seqable nor Seq — giving three disjoint map/filter/reduce universes (core, stream/*, nothing for channels). `extend Stream with Seqable` so core HOFs work unchanged and the `stream/*` family (with its backwards arg order) disappears; `collect` becomes `into([], s)`. Separately, channels are a sidecar concurrency stack whose `receive` busy-spins and pins the cooperative scheduler — give them a port protocol (`take!`/`put!`) that **parks** the continuation like `Async.Await` does. **Grounding:** Clojure unifies lazy-seqs/reducers/core.async channels through transducers; `<!` parks a go-block on the same scheduler that drives I/O.

### 2.10 `Contains` for CIDR/network membership (networking)

`ipv4-in-cidr?(ip-str, cidr-str)` re-parses both strings every call. Introduce a `Cidr` struct and `extend Cidr with Contains { fn contains?(c, ip) { (ip & c.mask) == c.network } }` — membership becomes the same polymorphic `contains?` used for maps/sets, parse-once. **Grounding:** Rust's `ipnet::Ipv4Net::contains`.

---

## 3. Cross-cutting consistency fixes

Each gets one recommended rule.

- **`?` always returns `Bool`.** `fs/exists?`/`is-file?`/`is-directory?` return `Result.Ok{value:true}`, making `if exists?(p)` a latent always-true bug. **Rule:** predicates return a plain `Bool`; push the absent-vs-denied distinction into a separate `stat(p) -> Result`. Rename `is-file?`→`file?` (the `is-` is redundant with `?`).

- **`!` marks every mutation, nothing else.** `record-pass`, `st-set-option`, `st-add-positional`, `beagle.test`'s `assert-*` all `swap!` atoms but lack `!`. **Rule:** any atom/state mutation carries `!`; pure output fns (`println`, `info`) do not (matches Clojure — `println` has no `!`). Rename `record-pass!`/`st-set-option!`/etc.

- **Argument order: function-first, collection-last, everywhere.** You committed to this for core seq HOFs (correctly). **Rule:** apply it without exception. Flip `join(coll, sep)`→`join(sep, coll)` (Clojure order, enables `wrap(t,w) |>> join("\n")`); flip every `stream/*` combinator to function-first (or better, delete them via §2.9); fix `reduce` docstrings that show collection-first.

- **One name per concept.** Delete the duplicates: `contains-key?` (O(n)) is just `contains?` (O(1)); `text/replace-all` and `regex/replace-all` should be one polymorphic `replace(s, match, repl)` dispatching on `match`'s type; `decode-url` is byte-identical to `decode`; `async-ok`/`async-err`/`async-ok?` shadow core `Result` (and `async-err`'s `code` param is dead — a silent no-op your stub policy forbids). Pick one home for functions defined in both `std.bg` and `beagle.iter` (`partition`/`flatten`/`range`/`repeat` have *divergent semantics* across the two — a real bug, not just duplication).

- **Result vs throw: one documented axis.** **Rule:** throw resumable errors for genuinely exceptional failures (I/O faults, programmer errors); reserve `Result` for outcomes the caller *routinely branches on* (HTTP non-2xx, parse-maybe). Then: glob must stop swallowing to `[]`; os must stop mixing throw/null/bool; EOF must be a normal value (`null`/`Eof`), never `Result.Err`; `unwrap(Err)` must **throw** (Rust contract), never println+null. Add `ex-info(msg, data)`/`ex-data`/`ex-message` so thrown errors carry structure instead of bare strings, and a `protocol Errorish { fn error-message(e)  fn error-kind(e) }` spanning `SystemError` (24 variants) and `Error` (9 variants) so callers stop exhaustively matching.

- **Options maps over named-function explosion.** `parse`/`parse-with-string-keys`, base64's four `encode`/`encode-url`/`decode`/`decode-url`, csv's `parse`/`parse-with-header` → one fn + trailing opts (`parse(s, {:keyword-keys? false})`). **Grounding:** `clojure.data.json/read-str` takes `:key-fn`, not separate fns.

---

## 4. Per-area highlights

**Core collections & sequences**
- `Conj` protocol to kill the `into` type-switch and the always-vector HOFs (§2.1) — top priority.
- Seq-coercing `first`/`rest`/`next`, delete `first-of` (§2.2).
- `reverse` (std.bg:1820) is O(n²) via `concat([x], acc)`; the fast O(n) version is hidden in `beagle.containers/coll-reverse`. Reimplement core `reverse` via backward index walk. Add a public `vec(coll)` to replace the repeated `instance-of(PersistentVector)` materialize-switches.

**Strings, text & formatting**
- Two mutable accumulators: fast `beagle.string-builder` implements *no* protocol while `std/StringBuffer` is the `Writer`. `extend StringBuilder with Writer` + `Length`; collapse to one.
- `append-int!`/`append-float!`/`append-char!`/`append!` is a hand-unrolled type-switch → one polymorphic `append!(sb, x)` via `Writer.write` + a `print-method`-style renderer.
- `join(coll, sep)` is collection-first — flip it (§3). Three private copies of `ws-code?`/`code-at` across text/textwrap/template — extract one exported char layer.

**Data formats (json/csv/ini/encodings)**
- `json/write-value`'s 7-way `instance-of` cascade → `ToJson` protocol (§2.6).
- Duplicated private `to-bytes` in base64 (28) and hex (19), byte-for-byte → a shared `beagle.bytes/ToBytes` protocol. struct-pack's triple cond-chain (`code-width`/`pack-one`/`unpack-one`) → one data table keyed on the format char.
- json silently degrades int-overflow to lossy Float and accepts leading-zeros — violates the module's own fail-loud stance; throw or require an explicit opt.

**IO, filesystem & process**
- `Closeable` + `with-open`; wire `File` into `Writer` + a `Reader` protocol (§2.7) — biggest missing idiom.
- `blocking-` prefix doubles the fs API (16 ops × 2) and leaks the execution axis into names; make the bare name the default-blocking common case. Predicates return `Result` not `Bool` (§3).
- `process/system` returns raw `(exit<<8)` wait-status, shell-only (injection hazard). Provide `run(argv) -> {exit, out, err}` (vector form, no shell) + explicit `sh("...")`. **Grounding:** `clojure.java.shell/sh` → `{:exit :out :err}`; babashka prefers the vector form.

**Async, concurrency & streams**
- Stream → Seqable; channels → parking port protocol (§2.9).
- Channels are unbounded with no backpressure and no `close!`/closed-state — unusable for real producer/consumer; add buffer strategies (`chan(n)`) + `close!`. Dequeue is O(n) per element (`drop(1, q)` on a vector → O(n²) drain); back with a PersistentQueue/two-vector.
- Four overlapping timeout APIs with three return contracts; `spawn`'s docstring ("never an OS thread") is false under 2 of 3 handlers. Split into honestly-named `spawn`/`thread`/`future` (Clojure's `go`/`thread`/`future`).

**Networking (http/ws/url/ip)**
- HTTP client takes 6 positional args → request map (`request({:method :url :headers :body})`), clj-http style; reuse the `Request` struct on both client and server (it's currently server-only).
- `Cidr`/`Ipv4`/`Url` as opaque structs with `Contains`/`Format` instead of bare Int and positional `[host, port, path]` vectors (§2.10).
- WebSocket opcodes are 6 nullary fns + raw-Int dispatch ladder → `enum Opcode` + exhaustive `match`.

**Numeric, time & hashing**
- Numeric tower so BigInt composes with `+`/`sort` (§2.8); core `Comparable` (§2.3); `Format` on all value structs (§2.5); `Eq`/`Hashable` so they're map keys (§2.4).
- `weekday` returns a magic `0..6` int despite the language having enums → `enum Weekday`. `floor-div`/`floor-mod`/`min-of`/`max-of`/`PI` duplicated across time/mathx/stats/std with *divergent* empty-handling contracts.
- `mathx-float?` detects type via `equal(type-of(x), type-of(1.0))` — add core `float?`/`int?`/`number?` predicates. Hash algorithm isn't a value (selection by fn name) → `enum Algorithm` + `digest(algo, input)`.

**Cross-cutting**
- Display unification (§2.5) and `Eq`/`Hashable` (§2.4) are the two highest-value runtime-touching protocols.
- `unwrap` println+null (§3). Duplicate `extend PersistentVector with Format` defined twice ~460 lines apart (std.bg:254 and :716) — dead, shadowing code; delete one and add a duplicate-extension lint.

---

## 5. Prioritized roadmap

### Quick wins — low-risk renames / arg-order / dedup

| Change | Impact | Effort | Breaking? |
|---|---|---|---|
| `?` predicates return `Bool` (`fs/exists?` etc.) | High (fixes always-true bug) | Low | Yes (small) |
| Flip `join(sep, coll)` + `stream/*` to function-first | High (consistency, `|>>`) | Low | Yes |
| Add `!` to atom-mutators (`record-pass!` etc.) | Medium | Low | Yes (renames) |
| Delete `contains-key?`, `decode-url`, `async-ok/err`, `to-string-any` | Medium (surface ↓) | Low | Yes (small) |
| Resolve std.bg vs beagle.iter divergent dupes (`partition`/`flatten`/`repeat`) | High (silent wrong behavior) | Low | Maybe |
| `unwrap(Err)` throws instead of println+null | High (no silent fails) | Low | Yes |
| Delete duplicate `PersistentVector` Format impl | Low (dead code) | Trivial | No |
| Fix docstrings to Beagle syntax + correct arg order | Medium (teaches right) | Low | No |
| Public `vec(coll)`; O(n) `reverse` | Medium (perf+dedup) | Low | No |

### Deep changes — protocol refactors

| Change | Impact | Effort | Breaking? |
|---|---|---|---|
| `Conj` protocol; rewrite `into` as `reduce(conj, …)` | **Very high** | Medium | No (additive) |
| Seq-coercing `first`/`rest`/`next`; delete `first-of` | High | Medium | Yes (deprecate alias) |
| Display unification (`to-string`/`${}` → `Format`) | **Very high** | Medium (runtime) | No (fixes bug) |
| `Eq` + `Hashable` protocols; pluggable `equal` | **Very high** | High (runtime) | No (additive) |
| `Comparable`/`compare`; route sort/min/max | High | Medium | No (additive) |
| `Closeable` + `with-open`; File→`Writer`/`Reader` | High | Medium | No (additive) |
| `ToJson` + unified `Codec` verbs; `ToBytes` | High | Medium | Partial |
| Numeric tower (`Numeric`, `+'`/auto-promote) | High | High | No (additive) |
| Stream→Seqable; channel port protocol + `close!`/backpressure | High | High | Partial |
| `ex-info`/`ex-data` + `Errorish`; unify error policy | High | Medium | No (additive) |
| Extend custom containers (Deque/OrderedMap/DefaultMap/PQ) to core protocols | Medium | Medium | No (additive) |

**Suggested sequencing:** Quick-wins batch first (they're independent and unblock confidence). Then the three additive protocols that fix the most findings with least breakage — **`Conj`, Display-unification, `Comparable`** — followed by `Eq`/`Hashable` (runtime work but unblocks an entire value-type class), then `Closeable`/IO and `ToJson`/Codec, then the numeric tower and stream/channel unification as larger projects.

---

## 6. What's already good — preserve this

- **HOF argument order is function-first and consistent**, and `reduce(f, init, coll)` matches `clojure.core/reduce` exactly — the single most important thing to get right, and it's right. Chains cleanly with `|>>`.
- **`contains?` semantics are faithful and carefully documented** (key/index membership, true even for stored-null in maps, std.bg:177-186) — a classic trap handled correctly. The 3-arg `get` + `find-entry` story correctly distinguishes stored-null from missing.
- **`Format(self, depth)`** folding pr-vs-print into one method with a depth parameter is the genuinely idiomatic move — the abstraction is right (the bug is only that interpolation bypasses it).
- **`let dynamic out` + `Writer`** is a clean port of Clojure's `*out*`/`java.io.Writer`.
- **The Fs effect design** (operations-as-data in `enum Fs` → pluggable `Handler`) and the **parameterized `Handler(T)` effect protocol** are more principled than most languages expose; the *surface* needs work, not the core idea.
- **`StreamSource` with CAS-guarded idempotent `ensure-closed`**, **`with-scope` structured concurrency**, and **channel.bg's well-commented CAS discipline** are genuinely good rare primitives.
- **beagle.path** is excellent — pure, POSIX-correct on the hard edges, exhaustively self-tested. **beagle.socket** is well-designed (one-line `perform` verbs, clean `on-connection`). The **ring-style HTTP server** (handler = `Request -> Response`) is the strongest single design decision in networking.
- **Immutable value-struct discipline** (DateTime/SemVer/BigInt/W64 never mutate), **BigInt zero-canonicalization**, **consistent -1/0/1 compare convention**, and **honest hard-errors on undefined results** (overflow points users at beagle.bigint; empty `min-of` throws) — all correct, all worth keeping.
- **Suffix discipline (`?`/`!`) is right in the large majority of cases** and the data-format modules' **fail-loud throw-on-malformed** stance is exactly the project philosophy.

The throughline: your *abstractions* are sound and Clojure-faithful. The work is finishing the commitment — extend the protocols you already designed, add the four or five missing ones (`Conj`, `Comparable`, `Eq`/`Hashable`, `Closeable`, `Numeric`), and route the two display engines and three error channels into one each.