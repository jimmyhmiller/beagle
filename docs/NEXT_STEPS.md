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

## Outstanding — decided this session, NOT yet implemented

These are explicit design decisions (see `project_beagle_stdlib_design_decisions`).

### 1. Maps must store `null` values — the Clojure way
Beagle's HAMT (`src/collections/persistent_map.rs`) overloads `null` as
empty-slot + child-node marker + not-found, so `assoc(m, k, null)` can't be
distinguished / stored. Fix exactly like Clojure's PersistentHashMap:
- occupancy via the node bitmap (not null);
- discriminate branch-vs-leaf by the **KEY slot** (null key / node marker), NOT
  by `value == null`, so the value slot may hold anything incl. null;
- keep a distinct internal not-found sentinel.
- Add `contains?`, `find` (returns the `[k v]` entry or null), and 3-arg
  `get(m, k, not-found)` to std.
CAUTION: maps back the compiler's symbol tables — validate hard under
`--gc-always` + full suite before trusting. Attended/careful work.

### 2. Seq functions → function-first arg order (`f, coll`)
Clojure convention: higher-order seq fns take the function/predicate FIRST:
`map(f, coll)`, `filter(pred, coll)`, `reduce(f, init, coll)`,
`take-while(pred, coll)`, `drop-while`, `remove`, `keep`, etc. Collection-update
fns stay collection-first (`assoc(m,k,v)`, `get(m,k)`, `conj(coll,x)`, `update`,
`dissoc`) — already correct. Beagle core currently has the seq fns backwards
(coll-first); `beagle.iter` is already f-first. FLIP core + update EVERY stdlib
caller + test in lockstep (mechanical but wide). std.bg + all `resources/*` that
call map/filter/reduce.

### 3. Formatting rework — no printf
Jimmy rejected printf-style `%f`/`text/format`. Use Beagle's `${...}` string
interpolation + plainly-named composable fns: `fixed(x, decimals)`,
`round-to(x, n)`, `pad-left/pad-right(s, width)`, `commas(n)`. No format
mini-language, no `%`-codes, no Rust `{:.2}` typed specs. Retire/deprecate the
printf-style `text/format`.

## Outstanding — queued

### 4. Round-6 gap-hunt
Parallel probe (vs Python ground truth, adversarial-verify stage) of the
still-uncovered modules: bigint, channel, containers, priorityqueue, glob, date
arithmetic, async (pure parts), stream, timer. Use the same workflow shape as
rounds 4/5 (probe clusters → verify → I triage+fix). Run probes from a FRESH
empty dir (stale `/tmp/beagle.*.bg` siblings shadow the embedded stdlib and
cause false results). Installed `beag` at `~/.cargo/bin/beag`.

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
