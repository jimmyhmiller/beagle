# Beagle — Next Steps (session handoff)

Handoff for starting a fresh session. Context for the big "make Beagle a
Python-comparable, batteries-included language" effort. Hard constraints (do not
violate): implement IN BEAGLE wherever possible, FFI for system/C, MINIMAL new
Rust (only true primitives), NO external Rust crates, and **GC-correctness is
validated every step** — `--gc-always` is the correctness oracle; a gc-always
crash is a real rooting/liveness bug to FIX, never a reason to lower GC
frequency.

Detailed memory lives in the assistant memory dir under
`project_beagle_*` (stdlib roadmap, B4 GC bug, string-slice hash, design
decisions). This doc is the actionable to-do list.

## Done this session (committed alongside this doc)

- ~35 stdlib modules shipped (text/json/csv/ini/url/base64/hex/hash/http/ws/
  os/path/glob/date/time/stats/mathx/bigint/iter/containers/regex/cli/log/ansi/
  struct-pack/semver/ip/…); catalog `docs/STDLIB.md`, roadmap `docs/STDLIB_ROADMAP.md`.
- ~25 bug fixes incl. string-slice/cons hash truncation, template string-keyed
  renders, mathx/struct-pack overflow→throw, path/semver/json-float/ip/textwrap/
  regex fixes, qualified-fn-refs-as-values (compiler), maps & lazy seqs Seqable.
- **Constant-stack tail resume** — `handle { loop { perform } }` now runs in 0
  net stack growth (continuation trampoline lands one-shot tail resumes at the
  captured base; `reset_shift.rs` + `continuations.rs`). Regression:
  `resources/effect_tail_resume_constant_stack_test.bg`.
- **Heap-frontier mismatch** — `GenerationalGC::can_allocate` now accounts for the
  JIT fast-path frontier (was `allocate_no_gc` panic at >150k tight iters).
- **B4 concurrent-GC corruption — FIXED in BOTH register allocators.** Root cause:
  the allocator pools are the callee-saved register set, so a live value survives
  an allocation call *in a register* — invisible to the conservative frame-slot GC,
  stale after a moving collection (nested-struct corruption `op.field == op` on
  spawned threads under gc-always; surfaced as the socket-echo crash). Fix: any
  value live across a GC safepoint that might be a heap pointer must be in a
  GC-scanned slot. Legacy: `linear_scan.rs::insert_safepoint_spills` (spill→slot
  before each safepoint, reload after). SSA: `cfg/regalloc/spill.rs` force-spills
  maybe-pointer cross-safepoint values (via `pointer_class`) so the function stays
  in SSA; `verify_clobber_safety` bail is a safety net only. Suite 412/0,
  perf-neutral. Regression: `resources/spawned_thread_nested_ctor_gc_test.bg`.

## Decided design items 1–3 — ALL IMPLEMENTED this session

These were explicit design decisions (see `project_beagle_stdlib_design_decisions`);
all three are now done, tested, and documented (see the per-item ✅ notes below).

### ✅ 1. Maps must store `null` values — the Clojure way (DONE)
Implemented. The fix was narrower than feared: Beagle's HAMT already stores
sub-nodes in the **key** slot and `assoc`/`bitmap_to_array_node` already
discriminate branch-vs-leaf by key node-ness — only lookups and `keys` keyed off
`value == null`. Changes:
- **Not-found sentinel** (`PersistentMap::NOT_FOUND = 0b1111`, Null-tagged with a
  nonzero payload ⇒ `is_heap_pointer == false`, GC never follows it, user code
  can't construct it). Returned only Rust-side from `get`/collision lookup; the
  builtin layer translates it to null / default / false. Never stored.
- `find_node` / `find_in_collision_node` / `get` return `NOT_FOUND` for misses
  but still return a real stored `null` on a key match (kept the fast path:
  non-null value slot ⇒ leaf; only `value == null` disambiguates via the KEY).
- `collect_keys` (the actual bug): null-valued leaves were dropped or misrecursed
  — now discriminates sub-node vs leaf by the key slot.
- `persistent_set.rs::contains` updated to compare against the sentinel.
- New builtins (`src/builtins/collections.rs` + `install.rs`):
  `map-get-default`, `map-contains?`, `map-find`. `map-get` now maps the
  sentinel → null (2-arg Clojure wart preserved).
- **std**: `contains?` is now a **protocol** (`Contains`), polymorphic over
  String (substring, backed by renamed `string-contains?` builtin) / PersistentMap
  (key present, incl. null value) / PersistentSet (member) / PersistentVector
  (index in range). 3-arg `get` routed to `map-get-default` for maps, and given
  explicit 3-arg impls for String/Array/Vector (the protocol's default 3-arg does
  NOT route through primitive-type dispatch). Map-entry lookup added as
  **`find-entry(m, key)`** — NOT `find`, which is already taken by the
  predicate-search `find(coll, pred)` (load-bearing in tests).
- **Null keys** work for free (null is never a node pointer ⇒ a `(null, v)` slot
  is just a leaf). Supported + tested.
- **null-aware std map fns:** `merge-with` and `select-keys` rewritten to use
  `contains?` instead of `v == null` (they were silently dropping present-with-null
  entries). `merge` / `dissoc` / `vals` were already correct (they rebuild via
  `keys` + `get`, which now preserve nulls).
- Regression: `resources/map_null_values_test.bg` (12 blocks) +
  `resources/map_null_values_gcalways_test.bg` (gc-always). Suite green,
  gc-always clean.

### ✅ 2. Seq functions → function-first arg order (`f, coll`) (DONE)
All function/predicate-taking core seq HOFs flipped to function-first, plus the
count/sep-arg ones flipped to Clojure n/sep-first. Suite 414/0.
- **Flipped to function-first** (std.bg defs + ALL callers): `map`, `filter`,
  `reduce` (`f, init, coll`), `reduce-right`, `any?`, `all?`, `none?`,
  `not-every?`, `find`, `find-index`, `take-while`, `drop-while`, `flat-map`,
  `zip-with` (`f, c1, c2`), `partition-by`, `group-by`, `sort-by`, `sort-with`
  (`less, coll`), `min-by`, `max-by`.
- **Flipped to count/sep-first** (Clojure-match): `take(n, coll)`,
  `drop(n, coll)`, `partition(n, coll)`, `interpose(sep, coll)`. `slice` kept
  coll-first (matches `subvec`); `remove-at(coll, idx)` kept coll-first (index,
  not a fn).
- **Left coll-first** (collection-update / not fn-taking): `assoc`, `get`,
  `conj`, `update`, `dissoc`, `nth`, `map-keys`, `map-vals`, `filter-keys`,
  `filter-vals` (Clojure `update-vals/keys` are coll-first), `zip`, `zipmap`,
  `enumerate`, `distinct`, `dedupe`, `flatten`, `frequencies`, `reverse`,
  `repeat`, `repeatedly`, `iterate`.
- **`beagle.iter`** was already f-first; **`beagle.stream`/`streamtest`** define a
  SEPARATE stream-first API (designed for `|>` first-arg pipes) and were left
  untouched.
- **Pipe operators:** f-first seq fns now chain with `|>>` (last-arg pipe), not
  `|>`. Updated `pipe_test.bg` and the guide examples. The stream API still uses
  `|>`.
- **Compiler:** array destructuring `[a, ...rest]` desugars to a `drop` call in
  `src/ast.rs` — flipped its emitted arg order to `drop(n, coll)`. (`nth(coll, i)`
  emission unchanged — `nth` stays coll-first.)
- Docs synced: `docs/language-guide.md` §16 + pipe section, all std.bg
  docstrings, and the `beagle-language` skill's `reference.md`.
- Caller updates were fanned out across files (std.bg + ~25 resources + 10 stdlib
  modules); verified by a global "fn-as-2nd-arg" scan + full suite.

### ✅ 3. Formatting rework — no printf (DONE)
Retired the printf machinery and replaced it with plain composable helpers in
`beagle.text`, meant to be used with `${...}` interpolation. Suite 415/0.
- **Removed** from `beagle.text`: `format` (printf), `format-one`,
  `maybe-force-sign`, `apply-pad`. No `%`-codes, no format mini-language.
- **Added / exposed:** `fixed(x, decimals)` (was the private `float->fixed`;
  renamed + made public — string with exactly N decimals, half-up, sign-aware,
  works on ints+floats), `round-to(x, n)` (returns a rounded *number*),
  `commas(n)` (thousands separators). `pad-left`/`pad-right` made multi-arity:
  2-arg form defaults to space padding, 3-arg keeps the custom pad string.
  `int->hex` kept (plainly-named, not printf).
- **Callers updated:** the only `text/format` users were two test files
  (`stdlib_ultracode_test`, `stdlib_phase1_3_test`) — rewritten to use the new
  fns / `${...}`.
- Regression: `resources/text_format_test.bg` (6 blocks). Docs synced:
  `docs/STDLIB.md` catalog, language-guide §2.4, skill `reference.md`.

## Outstanding — queued

### ◑ 4. Round-6 gap-hunt — RAN; 8/12 confirmed defects fixed, 4 deferred
Ran as a multi-agent workflow (probe vs Python ground truth → adversarial verify
→ triage+fix). Probed **bigint, containers, priorityqueue, glob, date, stream,
mathx, stats** (swapped the original list's channel/async/timer — concurrency/time
modules have no deterministic Python ground truth; probe them later with
property/invariant tests instead). priorityqueue + stats came back CLEAN.
12 defects confirmed by the verify stage.

**Fixed this session** (regression: `resources/gaphunt_round6_fixes_test.bg`):
- `bigint/from-int(MIN_INT)` silently returned 0 — `0 - n` overflows for -2^60.
  Now extracts limbs from `n` directly (abs per limb), never forming `0 - n`.
- `containers/dm-get` (DefaultMap) conflated stored-null with absent — now uses
  `contains?` (same class as the item-1 merge/select-keys fixes).
- `mathx/floor-div` + `floor-mod` were silently wrong for FLOAT args (int `/`
  truncates, float `/` is true division) — now branch on a local `mathx-float?`
  and `floor()` the float quotient. Fixes `is-even?`/`is-odd?` on floats too.
- `stream/buffered` CRASHED on any non-empty stream (`first` on a plain vector,
  which is Seqable not a Seq) — now uses indexed `get(buf, 0)`.
- `stream/merge` concatenated instead of interleaving — added a `turn` atom for
  true round-robin (`merge-pull` helper).
- `glob` treated any `**` as crossing `/` — now segment-only `**` crosses, per
  the module's own documented model.
- `date/format-iso8601` produced malformed years ≤ 0 (`0-44-…`) — now signed,
  fixed-width 4-digit years (`-0044-…`).

**Deferred (documented, not fixed):**
- `bigint`: no `div`/`mod`/`divmod`/`quotient`/`remainder` (feature gap — needs
  limb long-division with a documented sign convention, likely Python floor) and
  no `pow` (exponentiation-by-squaring on `mul`). Both are real bignum batteries
  worth adding; non-trivial, so left for a focused follow-up.
- `date`/`time` `from-epoch`↔`to-epoch` round-trip is off by a day for ~61 days
  around year 0 AD (Hinnant civil-from-days era division uses floor where
  truncation is needed). Real but extremely narrow (no effect on dates ≥ year 1);
  deferred. Fix: use truncating (toward-zero) division for the `era` terms.

### 5. B4 residual (non-blocking, optional perf)
The SSA fix uses `spill_one` (reloads at every use). A spill-AROUND-safepoint
variant (slot at the safepoint, register-resident between, one reload after)
would be marginally faster for many-use values — current perf is already neutral,
so this is optional. Finishing `pointer_class` Phases B/C/D (type-feedback-aware
non-pointer proof) would shrink the must-spill set further.

### 6. Bigger Python modules (not yet picked)
decimal/fractions, zlib/gzip, zipfile/tarfile, sqlite, difflib, secrets/CSPRNG,
datetime timezones, functools.lru_cache.

## Deferred deep bug
- B3/B4-class concurrent multi-thread alloc under gc-always: the specific
  socket-corruption instance is FIXED (this session). If new gc-always crashes
  appear in concurrent code, first check the GC-safepoint/root invariant
  (see `project_beagle_b4_concurrent_effect_gc`).

## Build / verify loop
```
cd ~/Documents/Code/beagle
cargo build --release            # ~13s
cargo install --path . --force   # install to ~/.cargo/bin/beag
BEAGLE_TEST_TIMEOUT_SECS=120 ./target/release/beag test resources   # full suite (412 tests; gc-always tests are slow)
```
Stdlib modules are registered in TWO places in `src/main.rs`
(`embedded_stdlib::get` match — REQUIRED, installed beag has no sibling files —
and the auto-load list). After editing a `.bg` stdlib file, rebuild (build.rs
syncs `target/release/standard-library/`). Run user programs from a fresh dir.
