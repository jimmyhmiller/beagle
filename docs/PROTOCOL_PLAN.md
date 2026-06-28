# Beagle Protocol Plan — the complete protocol architecture & function-change ledger

This is the concrete, end-to-end plan for Beagle's protocol surface: every protocol
(existing and proposed), the exact types that extend each, **which functions change**
because of it, and the cleanup of the two worst offenders — `beagle.collections`
(the Rust-builtin primitive layer) and `beagle.containers` (Deque/DefaultMap/OrderedMap).

## Status (updated as phases land)

- ✅ **Cross-module protocol extension fixed (compiler).** `extend Foo with Length`
  (unqualified core protocol, from any module incl. user code) used to *silently*
  register a dead impl; it now resolves correctly (current ns → `beagle.core`) and
  **errors loudly** on an unknown protocol. This unblocks everything below.
- ✅ **Phase B keystone (additive, suite 470/0):** `Conj` protocol; `into` = `reduce(conj, …)`;
  O(n) `reverse`; `vec`/`subvec`/`includes?`; core `union`/`intersection`/`difference`.
- ✅ **Container lift:** Deque/DefaultMap/OrderedMap extend the core protocols — `count`,
  `get`, `assoc`, `contains?`, `keys`, `seq`, `conj`, `into`, `for`, `reduce` all work on them.
- ⏳ **Next:** Display unification (`to-string`/`${}` → `Format`), `Comparable`, `Eq`/`Hashable`,
  `Closeable`/`with-open`, `ToJson`/Codec, numeric tower, Stream→Seqable; then delete the now-redundant
  concrete container fns + `first-of`/`contains-key?`/duplicate extends.

---

The governing rule, stated once:

> **For any operation that today is spelled as N concrete per-type functions or an
> `instance-of` ladder, there must be ONE protocol method that types `extend` into.
> Concrete primitives (`vec-push`, `map-assoc`, `deque-count`, `dm-get`) become the
> *implementation behind* an `extend`, never the public surface.**

Clojure is the north star: its entire `core` is consumable over any `deftype` because
the type implements `Counted`/`ILookup`/`Associative`/`Seqable`/`IFn`. Beagle already
has the right protocol *vocabulary*; this plan finishes the *commitment*.

### Naming convention: `_` = internal, hidden from docs

> **Any internal function, protocol method, or namespace is prefixed with `_`.**
> The doc generators (`tools/docgen-html.bg`, `tools/docgen.bg`) hide them: a function
> whose name starts with `_`, and a namespace whose name (or any `.`-segment) starts
> with `_` (e.g. `beagle._collections`), never appears in `docs/api.html` / `docs/api.md`.

Throughout this plan, raw protocol accessors that exist only to be called *through* a
public seq-coercing wrapper are named with the `_` prefix (`_first`, `_next`, `_rseq`),
and the primitive collection layer is moved into the internal `beagle._collections`
namespace. This replaces Clojure's `-`-prefixed convention (`-first`) — Beagle uses `_`.

---

## 0. The layering problem (why collections/containers are a mess)

Today there are **three** parallel collection APIs:

| Layer | Namespace | Examples | Should be |
|---|---|---|---|
| Rust primitives | `beagle.collections` | `vec-push`, `vec-count`, `vec-get`, `map-assoc`, `map-get`, `set-add`, `set-contains?`, `mutable-map-put!` | **internal** — only called by `extend` blocks and the compiler (ast.rs) |
| Protocol methods | `beagle.core` (std.bg) | `push`, `length`, `get`, `assoc`, `contains?`, `keys`, `seq` | **the public surface** |
| Ad-hoc containers | `beagle.containers` | `deque-count`, `dm-get`, `om-assoc`, `coll-reverse`, `vec-member?`, `set-union-vec` | **extend the protocols**; delete the duplicates |

The fix is a single principle applied twice: **demote** `beagle.collections` to a private
primitive layer, and **lift** `beagle.containers`' types onto the existing protocols so
`count`/`get`/`assoc`/`seq`/`first`/`for`/`map`/`into` work on them for free.

---

## 1. Protocol catalog

Tiers reflect dependency + rollout order. Tier 0 exists today (needs fixes). Tiers 1–3 are new.
Each entry gives the Beagle signature, who extends it, the public functions routed through
it, what it deletes, and the Clojure analog.

### Tier 0 — Core abstractions that already exist (fix & finish)

#### `Seqable { fn seq(coll) }` + `Seq { fn _first(s)  fn _next(s) }`
- **Extended by today:** PersistentVector(Seq), PersistentMap(Seq), PersistentSet(Seq),
  Array, Range, String, ArraySeq/…Seq cursors. (std.bg:168–175, 555–710, 1037–1105, 2809–2885)
- **Change:** rename raw `Seq.first/next` → `_first`/`_next` (internal). Make the **public**
  `first/rest/next` seq-coerce:
  ```
  fn first(coll) { let s = seq(coll)  if s == null { null } else { _first(s) } }
  fn rest(coll)  { let s = seq(coll)  if s == null { [] } else { _rest_or_empty(s) } }
  fn next(coll)  { let s = seq(coll)  if s == null { null } else { _next(s) } }
  ```
- **Deletes:** `first-of` (std.bg:1438 — exists *only* to call seq-then-first; its docstring
  admits it). All `first-of` call sites → `first`.
- **Clojure:** `clojure.core/first` calls `seq` internally; CLJS exposes the raw method as
  `ISeq.-first`. Rust: `Iterator::next` is the trait method, `.nth(0)` the convenience.

#### `Length { fn length(coll) }`  → **rename to `Counted { fn count(coll) }`**
- **Extended by today:** PersistentVector, PersistentMap, PersistentSet, String, Array, …Seqs.
- **Change:** Beagle currently has BOTH `length` (the protocol method) and `count(coll){length(coll)}`
  (std.bg:1416) — a pure duplicate. Collapse to one. **Recommended:** make `count` the protocol
  method (Clojure's name + `Counted`), keep `length` as a thin deprecated alias for strings/arrays
  during migration, then remove. `empty?(coll)` (std.bg:1426) becomes `count(coll) == 0` and stays.
- **Clojure:** `clojure.core/count` + `Counted`. Rust: `len()`.

#### `Indexed { fn get(coll, index) ; fn get(coll, index, default) }`
- **Default:** the 3-arg form already defaults via 2-arg + null check (std.bg:132–143). Keep.
- **Extended by today:** PersistentVector, PersistentMap, String, Array.
- **Routed through it:** `get`; the existing `nth` (std.bg:1494, currently concrete) becomes the
  ordinal sibling routed through `Indexed`/`Seq`.
- **Clojure:** `ILookup` (`get`) + `Indexed` (`nth`). Beagle merges them; fine for a dynamic lang,
  but keep the *semantic* split in docs (`get` = associative/keyed, `nth` = ordinal).

#### `Associable { fn assoc(coll, key, value) }`
- **Extended by today:** PersistentVector (index), PersistentMap (key). (std.bg:278, 314)
- **Lift:** the existing `dissoc` (std.bg:2489, map-only) onto a sibling protocol
  `Dissociable { fn dissoc(coll, key) }` (maps + records) — today's remove is not polymorphic.
- **Routed through it:** `assoc`, `assoc-in`, `update`, `update-in`, `into` (maps).
- **Clojure:** `Associative`/`assoc` + `clojure.core/dissoc`.

#### `Keys { fn keys(coll) }` (+ add `Vals { fn vals(coll) }`)
- **Extended by today:** PersistentMap (std.bg:321).
- **Lift `vals`** onto a peer protocol `Vals { fn vals(coll) }` (today `vals` is map-only, std.bg:2463,
  and OrderedMap ships its own `om-vals-in-order`).
- **Clojure:** `clojure.core/keys` / `vals`.

#### `Push { fn push(coll, value) }`  → **subsumed by `Conj` (see 1.1)**
- Today `Push` is extended **only** to PersistentVector (std.bg:247, 624 — *duplicated*). It is the
  ancestor of the new `Conj`. Keep `push` as the vector-tail alias; make `conj` the polymorphic one.

#### `Contains { fn contains?(coll, x) }`
- **Extended by today:** String (substring), PersistentVector (**index-in-range**), PersistentSet
  (member), PersistentMap (key present — true even for stored-null). (std.bg:184–233)
- **Change:** this is *correct and carefully documented* — **do not** collapse `vec-member?` into it
  (see §3): `contains?` on a vector asks about the **index**, not the value. Add a separate
  `includes?(coll, x)` / `some-=?` for **value** membership (Clojure: `contains?` = key/index,
  value membership = `some`/`.contains`).
- **Deletes:** `contains-key?` (std.bg:2651 — O(n) `keys`+scan; identical to O(1) `contains?` on maps).

#### `Format { fn format(self, depth) }`  → **the Display/Show protocol (fix the bypass)**
- **Extended by today:** Struct, String, PersistentVector(×2 — *duplicated* at 254 & 716),
  PersistentMap, PersistentSet, Regex. (std.bg:32, 188, 254, 328, 716, 2773, 3070)
- **THE BUG:** `println(x)` dispatches through `Format`, but `${x}` interpolation and `to-string(x)`
  go through a *separate* Rust path (`get_repr_inner`, src/runtime.rs:7924) that never calls `Format`.
  Extend the type and `println` honors it while `"${x}"` does not.
- **Change:** route the `to-string` builtin and string-interpolation through `Format` for HeapObjects
  (Rust calls back into the protocol). Then **delete the duplicate** `PersistentVector with Format`
  (keep one of 254/716) and add a duplicate-extension lint.
- **Routed through it:** `to-string`, `${...}`, `print`/`println`, `format`, and `Format` impls on
  BigInt/SemVer/DateTime/W64 (so `bigint/to-string-any` is deleted).
- **Clojure:** the single `print-method` multimethod backs `str`/`pr`/`println`/`%s`. Rust:
  `Display`/`Debug` traits — `format!` always goes through the impl, never a shadowing fallback.

#### `Writer { fn write(self, s) }`
- **Extended by today:** Stdout, StringBuffer (std.bg:825–858).
- **Change:** extend **File** (IO), **StringBuilder** (the fast byte buffer currently implements *no*
  protocol), **Socket**, and the JSON/CSV/INI serializers write *to a Writer* instead of each
  privately accumulating a String. Collapses `write-stdout`/`write-stderr`/`write-string` → one
  `write`; Stdin/Stdout/Stderr become **values**, not name prefixes.
- **Clojure:** `java.io.Writer` + `*out*` (Beagle's `let dynamic out` is already a clean port).

### Tier 1 — New core abstractions

#### 1.1 `Conj { fn conj(coll, x) }` — the polymorphic add *(highest value)*
- **Signature & impls:**
  ```
  protocol Conj { fn conj(coll, x) }
  extend PersistentVector with Conj { fn conj(v, x)  { push(v, x) } }            // append
  extend PersistentSet    with Conj { fn conj(s, x)  { set-add(s, x) } }         // add
  extend PersistentMap    with Conj { fn conj(m, kv) { assoc(m, kv[0], kv[1]) } } // [k v]
  extend Deque            with Conj { fn conj(d, x)  { push-back(d, x) } }
  ```
- **Rewrites:** `into` (std.bg:1798) collapses from its `set?/instance-of(PersistentMap)/else`
  type-switch to **`fn into(to, from) { reduce(conj, to, from) }`**. Every HOF that hardcodes
  `push(acc, …)` (so they always return a vector) gains an optional accumulator target.
- **Enables:** custom containers become first-class accumulation targets.
- **Clojure:** `clojure.core/conj`; `(into to from)` *is literally* `(reduce conj to from)`. Rust:
  `Extend` trait. Elixir: `Collectable`.

#### 1.2 `Comparable { fn compare(a, b) }` → `-1 | 0 | 1`
- **Impls:** default over primitives (folds in semver's hand-rolled `cmp-int`/`cmp-string`);
  `extend BigInt/SemVer/DateTime/Version with Comparable`.
- **Rewrites:** `sort`/`sort-by`/`min-of`/`max-of`/`min-by`/`max-by` route through `compare` instead
  of hardwired `<` (today primitives-only, std.bg:2322). Unifies the **two clashing comparator shapes**:
  `sort-with(less, coll)` (boolean) vs priorityqueue's `pq-new-with(compare)` (3-way). Keep
  `sort-with(less?, coll)` as the boolean variant (Clojure's `sort` accepts both).
- **Deletes:** `semver/cmp-int`, `semver/cmp-string`, `priorityqueue/default-compare`,
  `date/dt-compare` (becomes `compare`).
- **Clojure:** `clojure.core/compare` (single 3-way comparator backing `sort`). Rust: `Ord`/`PartialOrd`.

#### 1.3 `Eq { fn =(a, b) }` + `Hashable { fn hash(x) }` — user-extensible value semantics
- **Why:** `Runtime::equal` is a fixed Rust match carrying its own `// TODO: Make this pluggeable`
  (src/runtime.rs:8553); there is no value `hash`. So user structs can't define equality (a `Money`
  comparing only cents) nor be PersistentMap/Set keys with custom semantics. `date/dt-equal?` exists
  *because* "structs don't compare field-wise with `==`."
- **Impls:** default `Eq` = current field-wise compare; default `Hashable` = structural hash. Types
  override (Money, normalized DateTime, etc.).
- **Runtime wiring:** `Runtime::equal` calls `Eq` for HeapObjects; PersistentMap/Set key hashing calls
  `Hashable`. Document the `=`/`hash` agreement invariant.
- **Deletes:** `date/dt-equal?` → `=`.
- **Clojure:** CLJS `IEquiv`/`IHash`. Rust: `Eq`+`Hash` with the explicit "must agree" rule.

#### 1.4 `Reversible { fn _rseq(coll) }` (optional) + fast `reverse`
- Core `reverse` (std.bg:1820) is **O(n²)** (`concat([x], acc)` each step). The fast O(n) version is
  hidden in `beagle.containers/coll-reverse`. Reimplement core `reverse` as the backward index walk;
  optionally back it with a `Reversible` protocol for vectors/deques that can reverse in O(1)/O(n).
- **Deletes:** `containers/coll-reverse` (its body *becomes* core `reverse`).
- **Clojure:** `rseq` (`Reversible`) + `reverse`.

### Tier 2 — IO & resource protocols

#### 2.1 `Closeable { fn close(self) }` + `with-open` form
- **Impls:** `extend File/Socket/Stream/Channel with Closeable`.
- **Adds:** a `with-open(r = open(...)) { ... }` bracket that closes on **every** exit path
  (including a throw). Replaces all manual `io/close` and the open/match/close dances in
  `async/handle-write-file`, and unifies `stream/source-close`.
- **Clojure:** `with-open` + `java.io.Closeable`. Rust: `Drop`. Python: context managers.

#### 2.2 `Reader { fn read-bytes(self, n)  fn read-line(self) }`
- **Impls:** File, Stdin.
- **Collapses:** `read-stdin*`/`read-line` → one polymorphic `read-line`; file reads + stdin reads
  share one surface. Pairs with `Writer` (1.Tier-0) extended to File.
- **Adds (coercion):** IOFactory-style `reader(x)`/`writer(x)`/`slurp(x)`/`spit(x, v)` polymorphic over
  path-string | File | Stdin, so the same `slurp`/`spit`/`copy` work across sources (today: path-only
  `fs/blocking-read-file` + per-target `write-stdout`/`write-string`).
- **Clojure:** `clojure.java.io` IOFactory (`make-reader`/`make-writer`), `slurp`/`spit`.

### Tier 3 — Domain protocols

#### 3.1 `ToJson { fn write-json(value, out, indent, depth) }` (+ unify codec verbs)
- **Why:** `json/write-value` is a closed **7-way `instance-of` cascade** ending in
  `throw("unsupported type")` (beagle.json:589) — user structs/enums *cannot* be encoded, no hook.
- **Impls:** String/Int/Float/PersistentVector/PersistentMap/Keyword; users `extend` their own types.
  The catch-all `throw` becomes a fixable missing-impl. Writes to a `Writer` (2.1) so JSON can stream
  to any sink.
- **Verb unification:** collapse the four divergent pairs — `parse`/`stringify` (json) vs
  `parse`/`write` (ini) vs `encode`/`decode` (base64/hex) vs `pack`/`unpack` (struct-pack) — onto a
  shared `decode(format, src)` / `encode(format, value)` vocabulary, optionally a `Codec` protocol
  dispatching on a format value.
- **Clojure:** `clojure.data.json`'s `JSONWriter` protocol (`-write [obj out opts]`, `extend-protocol`).
  Rust: serde `Serialize`.

#### 3.2 `ToBytes { fn to-bytes(x) }` / `FromBytes`
- **Why:** base64 (`to-bytes`, ~L28) and hex (`to-bytes`, ~L19) each ship a **byte-identical private
  copy**; struct-pack only accepts Vec input.
- **Impls:** String, PersistentVector(of ints), Array. Plus `bytes->string`/`string->bytes` inverses.
  Shared `beagle.bytes` module that base64/hex/struct-pack/hash/http `use`.
- **Clojure:** byte-streams `to-byte-array`; clojure.java.io coercions.

#### 3.3 `Numeric { fn add(a, b)  fn sub(a, b)  fn mul(a, b)  fn neg(a) }` — the numeric tower
- **Why:** every BigInt op is a concrete BigInt-only fn; `+`/`<`/`sort` don't work, so `factorial`
  detects overflow and tells the user to *switch modules and rewrite the call site*.
- **Impls:** Int, Float, BigInt with an Int→BigInt promotion rule.
- **Ideal:** core `+`/`*` overflow-promote Int→BigInt automatically (Clojure's `+'`/`*'`) so
  `factorial`/`pow-int` never bail. Add `bigint/to-int` narrowing.
- **Deletes/reroutes:** `bigint/add`/`sub`/`mul`/`bi-lt?` become `extend BigInt` impls of `Numeric`
  + `Comparable`; call sites use `+`/`*`/`compare`.
- **Clojure:** the numeric tower (`clojure.lang.Numbers`); `+'`/`*'` auto-promote. Rust: `Add`/`Mul`
  + `num::BigInt`.

#### 3.4 `Awaitable { fn await(x) }` + `Port { fn take!(p)  fn put!(p, v) }`
- **Awaitable:** `await` dispatches over Future, TaskScope, and a channel-read uniformly (today
  `await` is futures-only; channels have a disjoint `receive`).
- **Port:** Channel implements `take!`/`put!` that **park** the continuation under the cooperative
  scheduler (today `channel/receive` busy-spins and pins the scheduler). Adds `close!` + buffered
  `chan(n)` backpressure.
- **Stream → Seqable:** `StreamResult.Value/Done` + `source-next` *is* `Seq._first/_next` with an
  Error arm. `extend Stream with Seqable` so core `map`/`filter`/`reduce`/`take`/`into` work on
  streams unchanged — **deleting the entire `stream/*` combinator family** (which also has the wrong,
  collection-first arg order). `stream/collect` → `into([], s)`.
- **Clojure:** core.async `ReadPort`/`WritePort`, `<!`/`>!` parking; manifold `to-deferred`; lazy
  seqs/reducers/channels unified through transducers.

#### 3.5 `Errorish { fn error-message(e)  fn error-kind(e) }` + `ex-info`
- **Why:** three disjoint error channels (`Result.Ok/Err`, thrown `SystemError`/`Error` enums, thrown
  bare strings), no structured carrier. Callers exhaustively `match` 24 `SystemError` + 9 `Error`
  variants.
- **Adds:** `ex-info(msg, data)` / `ex-data(e)` / `ex-message(e)`; `Errorish` spanning the enum
  families so callers stop exhaustive-matching.
- **Clojure:** `ex-info`/`ex-data`/`ex-message`. Rust: `std::error::Error` trait.

---

## 2. The `beagle.collections` cleanup (the primitive layer)

`beagle.collections` is the **Rust-builtin primitive layer** the compiler emits directly
(ast.rs:3653–3735 calls `beagle.collections/vec`, `vec-push`, `map-assoc`, …). It must become
**private plumbing**, not public API.

| Current public fn | Fate | Public replacement |
|---|---|---|
| `vec`, `array-to-vec`, `vec-to-array` | keep, mark **internal** | `vec(coll)` materializer (new, see §4) wraps these |
| `vec-count`, `map-count`, `set-count` | internal | `count` (via `Counted`) |
| `vec-get`, `map-get` | internal | `get` (via `Indexed`) |
| `vec-push` | internal | `push`/`conj` (via `Conj`) |
| `vec-assoc`, `map-assoc` | internal | `assoc` (via `Associable`) |
| `map-keys` | internal | `keys` (via `Keys`) |
| `map-get-default` | internal | 3-arg `get` (via `Indexed`) |
| `map-contains?`, `set-contains?` | internal | `contains?` (via `Contains`) |
| `map-find` | internal | `find-entry` |
| `set`, `set-add`, `set-elements` | internal | `set(coll)` ctor + `conj` + `seq` |
| `mutable-map*` (`put!`/`get`/`increment!`/`count`/`entries`) | **keep public** as the explicitly-mutable API, but rename to a `MutableMap` type with `!`-suffixed ops and an `extend` of `Counted`/`Indexed` for read ops | `mm-put!`→`put!`, `mutable-map-get`→`get`, `mutable-map-count`→`count` |

**Action:** **rename the namespace `beagle.collections` → `beagle._collections`** so the whole primitive
layer is internal-by-convention and auto-hidden from docs (no per-function `_` prefixing needed). This
requires updating the compiler's hardcoded call sites in `src/ast.rs` (`beagle.collections/vec`,
`/vec-push`, `/map`, `/map-assoc`, …) and the `extend … with …` bodies in std.bg that call these
primitives. Net: the public collection surface becomes the ~12 protocol methods, not the ~27 `xxx-yyy`
primitives. (Individual still-public-but-internal helpers elsewhere just take a `_` prefix instead.)

---

## 3. The `beagle.containers` cleanup (Deque / DefaultMap / OrderedMap)

This is the flagged mess. Every type here reimplements operations the core protocols already define.
**Lift each onto the protocols; delete the concrete duplicates.** Constructors get clean names.

### Deque
| Current | Fate | Becomes |
|---|---|---|
| `deque-new()` | rename | `deque()` |
| `deque-count(d)` | **delete** → `extend Deque with Counted` | `count(d)` |
| `deque-empty?(d)` | **delete** | `empty?(d)` (via `Counted`) |
| `deque-to-vec(d)` | **delete** → `extend Deque with Seqable`/`Reversible` | `vec(d)` / `seq(d)` / `for x in d` |
| `push-back(d, x)` | keep (deque-specific) **+** `extend Deque with Conj` | `conj(d, x)` or `push-back` |
| `push-front`,`pop-front`,`pop-back` | keep — genuine double-ended ops (no core protocol) | optionally a `Deque`/`Stack` protocol `{ push-front pop-front push-back pop-back }` |
| `deque-rebalance` | keep **internal** (impl detail) | — |
| `coll-reverse(v)` | **delete** — body becomes core O(n) `reverse` (§1.4) | `reverse(coll)` |
| `coll-subvec(v,s,e)` | **promote to core** | `subvec(coll, start, end)` |

### DefaultMap
| Current | Fate | Becomes |
|---|---|---|
| `dm-new(default)` | rename | `default-map(default)` |
| `dm-get(dm, k)` | **delete** → `extend DefaultMap with Indexed` (impl returns stored default when absent) | `get(dm, k)` |
| `dm-assoc(dm,k,v)` | **delete** → `extend DefaultMap with Associable` | `assoc(dm, k, v)` |
| `dm-update(dm,k,f)` | **delete** → core generic `update` | `update(dm, k, f)` |
| `dm-count(dm)` | **delete** → `extend DefaultMap with Counted` | `count(dm)` |
| — | **add** | `extend DefaultMap with Contains/Keys/Seqable` so `contains?`/`keys`/`seq`/`for` work |

### OrderedMap
| Current | Fate | Becomes |
|---|---|---|
| `om-new()` | rename | `ordered-map()` |
| `om-get(om,k)` | **delete** → `extend OrderedMap with Indexed` | `get(om, k)` |
| `om-assoc(om,k,v)` | **delete** → `extend OrderedMap with Associable` | `assoc(om, k, v)` |
| `om-has-key?(om,k)` | **delete** → `extend OrderedMap with Contains` | `contains?(om, k)` |
| `om-keys-in-order(om)` | **delete** → `extend OrderedMap with Keys` (insertion order) | `keys(om)` |
| `om-vals-in-order(om)` | **delete** → core `vals` (new `Vals` peer of `Keys`) | `vals(om)` |
| `om-pairs(om)` | **delete** → `extend OrderedMap with Seqable` (yields `[k v]` in order) | `seq(om)` / `entries(om)` |
| `om-count(om)` | **delete** → `Counted` | `count(om)` |

> Note: once OrderedMap extends `Seqable` + `Associable` + `Keys`, it drops transparently into
> `map`/`filter`/`reduce`/`into`/`for`/`merge` with **zero** bespoke code — exactly the payoff.

### Vector-as-set helpers (the subtle one)
| Current | Fate | Becomes | Why |
|---|---|---|---|
| `vec-member?(v, x)` | **rename, do NOT fold into `contains?`** | `includes?(coll, x)` / `some-=?` | `contains?` on a vector is **index** membership (std.bg:177–186); value membership is a *different* op (Clojure: `some`/`.contains` vs `contains?`) |
| `set-union-vec(a,b)` | **promote to core set algebra** over `PersistentSet` (+ vector convenience) | `union(a, b)` | clojure.set/union |
| `set-intersection-vec` | promote | `intersection(a, b)` | clojure.set/intersection |
| `set-difference-vec` | promote | `difference(a, b)` | clojure.set/difference |

The set-algebra functions should operate on `PersistentSet` (Beagle has a real set type) and accept any
`Seqable` via `into-set`; the "vectors-as-sets" variants become thin wrappers or are dropped.

---

## 4. Global function-change ledger (by effect)

**Deleted (duplicates / dead):**
`first-of`→`first`; `contains-key?`→`contains?`; `count` keeps (the duplicate is `length`, demoted);
`coll-reverse`→core `reverse`; `bigint/to-string-any`→`Format`; `date/dt-equal?`→`=`;
`date/dt-compare`→`compare`; `semver/cmp-int`/`cmp-string`→`compare`; `priorityqueue/default-compare`→`compare`;
one of the duplicate `PersistentVector with Format` extends (254/716); the duplicate
`PersistentVector with Length`/`Push`/`Indexed` extends (240/605, 247/624, 210/612);
all `beagle.containers/{deque-count,deque-empty?,deque-to-vec,dm-*,om-*}` (→ protocol methods);
`stream/*` combinator family (→ Stream-as-Seqable);
`async-ok`/`async-err`/`async-ok?` (shadow core `Result`); `decode-url` (= `decode`).

**Rerouted (body changes, name kept):**
`into` → `reduce(conj, to, from)`; `reverse` → O(n) walk; `sort`/`sort-by`/`min-of`/`max-of` → `compare`;
`to-string` + `${...}` → `Format`; json `write-value` → `ToJson`; base64/hex `to-bytes` → `ToBytes`;
`Runtime::equal` → `Eq`; map/set key hashing → `Hashable`.

**Lifted from concrete/map-only to protocol-backed polymorphic (already exist):**
`nth` (std.bg:1494), `dissoc` (2489 → `Dissociable`), `vals` (2463 → `Vals`), `await` (futures-only → `Awaitable`),
`equal`/`=` (runtime → `Eq`).

**Genuinely new public fns:**
`conj`, `compare`, `hash`, `subvec`, `vec(coll)`, `includes?`,
`union`/`intersection`/`difference`, `with-open`, `reader`/`writer`/`slurp`/`spit`, `ex-info`/`ex-data`/`ex-message`,
`close`, `take!`/`put!`/`close!`/`chan(n)`.

**Renamed for convention (`?`→Bool, `!`→mutation, fn-first arg order):**
`fs/exists?`/`is-file?`/`is-directory?` return `Bool` (today `Result.Ok{value:true}` → `if` always true);
`is-file?`→`file?`; `record-pass`→`record-pass!`, `st-set-option`→`st-set-option!`, `assert-*` keep
(macros) but mutating helpers get `!`; `join(coll, sep)`→`join(sep, coll)`; every `stream/*` flips to
function-first (mostly moot post-Seqable).

---

## 5. The resulting canonical core API (target surface)

After the plan, *these* are the public collection/value operations — uniform across vectors, maps,
sets, strings, ranges, seqs, deques, default-maps, ordered-maps, streams, and user types:

```
seq, first, rest, next, count, empty?            // sequence/cardinality   (Seqable/Seq/Counted)
get, nth, assoc, dissoc, update, keys, vals      // associative/indexed    (Indexed/Associable/Keys/Vals)
conj, push, into, contains?, includes?           // build/membership       (Conj/Contains)
map, filter, reduce, remove, take, drop, …        // HOFs (function-first, accumulate via Conj)
reverse, sort, sort-by, sort-with, distinct      // ordering               (Reversible/Comparable)
union, intersection, difference                  // set algebra            (clojure.set)
compare, =, hash                                 // value semantics        (Comparable/Eq/Hashable)
to-string, format, print, println                // display (one engine)   (Format)
write, read-line, close, with-open, slurp, spit  // IO                     (Writer/Reader/Closeable)
+, -, *, <, > (promoting Int→BigInt)             // numeric tower          (Numeric/Comparable)
```

Custom container authors write only `extend MyType with Seqable/Counted/Indexed/Associable/Conj` and get
the entire top section for free — the Clojure `deftype` experience.

---

## 6. Rollout sequencing

**Phase A — Quick wins (independent, low risk):** delete `first-of`/`contains-key?`/duplicate extends;
fix O(n) `reverse`; add `vec`/`subvec`/`nth`/`vals`; `?`-predicates return Bool; flip `join` arg order;
`unwrap(Err)` throws. *(additive or tiny breaks)*

**Phase B — `Conj` + `into`-rewrite + custom-container lift:** add `Conj`; rewrite `into`; lift
Deque/DefaultMap/OrderedMap onto Counted/Indexed/Associable/Keys/Seqable/Conj; demote `beagle.collections`
primitives to internal. *(additive; deletes container duplicates)* **← fixes the flagged mess.**

**Phase C — Display unification + `Comparable`:** route `to-string`/`${}` through `Format`; add
`compare`; route sort/min/max. *(fixes the two-engines bug)*

**Phase D — `Eq`/`Hashable` (runtime):** pluggable `equal` + value `hash`; user types as map keys.

**Phase E — IO (`Closeable`/`Reader`/Writer-on-File/`with-open`/`slurp`/`spit`) and codecs
(`ToJson`/`ToBytes`/unified verbs).**

**Phase F — Numeric tower; Stream→Seqable + channel Port/`close!`; `ex-info`/`Errorish`.** *(largest)*

Phases A–C dissolve the majority of the 94 critique findings (and all of the collections/containers
mess) with mostly-additive change; D–F are the deeper, higher-effort projects.
```
