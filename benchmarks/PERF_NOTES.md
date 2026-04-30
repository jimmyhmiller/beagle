# Benchmarksgame perf notes

This branch (`benchmark-perf-clean`) is the result of applying the perf changes
from `benchmark-perf-improvements` one at a time, measuring each, and keeping
only the ones that produced a measurable, repeatable speedup. No FFI, no
unsafe primitives, no SSA pipeline, no benchmark restructuring that didn't
pay for itself.

All measurements are 3-iteration median wall time, release build (`cargo build
--release`), Apple Silicon, otherwise-idle machine.

## Summary

| benchmark      | n         | before (ms) | after (ms) | speedup |
|----------------|-----------|-------------|------------|---------|
| fannkuch_redux | 9         | 989         | 736        | **-26 %** |
| spectral_norm  | 700       | 1650        | 1240       | **-25 %** |
| nbody          | 500 000   | 1548        | 1394       | -10 %   |
| fasta          | 1 000 000 | 1627 (FFI)  | 928        | **-43 %** *(no longer uses FFI)* |
| fasta          | 5 000 000 | 7519 (FFI)  | 3995       | **-47 %** *(no longer uses FFI)* |
| knucleotide    | (fasta 250k) | 2107 (FFI) | 1964     | -7 % *(no longer uses FFI)* |
| revcomp        | (fasta 1M)   | broken on main | 1403 | works *(was broken; no longer uses FFI)* |
| binary_trees   | 16        | 429         | 434        | (noise) |
| mandelbrot     | 800       | 626         | 628        | (noise) |

The fasta/knucleotide/revcomp numbers are vs. the FFI implementations
that were on `main` — the new versions are pure Beagle (use the new
`beagle.string-builder` stdlib module). For revcomp, the FFI version on
`main` had a pre-existing arity mismatch with `ffi/copy-bytes_filter`
and didn't even compile; the new pure-Beagle version is the first
working revcomp on this branch. See section 3 below.

## What was changed

### 1. New stdlib helpers in `beagle.mutable-array` (the big win)

Two functions were added:

- `get(array, index)` — bounds-checked read that **skips the type check**.
  Caller asserts `array` is an array (came from `new-array`, never escaped to
  user code that could rebind it). Out-of-range still returns `null`, so we
  don't lose memory safety — we only lose the `is-object && type-id == 1`
  guard on every access.

- `swap(array, i, j)` — does two reads + two writes against one
  `panic-if-not-array` and one `primitive/size` load, instead of four of each
  if you wrote it as four separate `read-field`/`write-field` calls. Both
  indices are still bounds-checked.

Why it matters: `arr/read-field` does a runtime tag check + size load on
every call. In a tight inner loop those are the dominant cost. spectral_norm
does ~10 M array reads at n=700; fannkuch_redux at n=9 spends most of its
time inside the swap-pairs loop. Removing the per-call type check is what
buys back the 25 % on those benchmarks.

### 2. Benchmarks updated to use the new helpers

- `fannkuch_redux.bg`: 4-call swap pattern → `arr/swap`; permutation reads →
  `arr/get`. Inner loop reduces from 4 + 4 type checks per swap to 1.
- `spectral_norm.bg`: every `arr/read-field` → `arr/get`. The `Au` /
  `Atu` inner loops are O(n²) array reads per iteration, with 10 outer
  iterations. Type-check elision is the entire win here.
- `nbody.bg`: every `arr/read-field` → `arr/get`. Smaller win than the others
  because the inner loop only reads 5 bodies per pair, not n bodies. But
  still 5 M array reads at n=500 000 and they show up.

The structural rewrites that the WIP branch had (while-loops → tail-recursive
helpers, `Body { mut x, mut y, ... }` → immutable + `Bodies { b0..b4 }` bag,
etc.) were not applied here. See "Things tried and rejected" below.

### 3. New `beagle.string-builder` stdlib module + `fasta` ported off FFI

Added a Java-style mutable string builder backed by a packed-byte storage
heap object. fasta no longer touches `beagle.ffi` — and is **2× faster**
than the FFI version was.

#### Runtime layout

Two new heap types (`src/collections/type_ids.rs`):

- `TYPE_ID_BYTE_STORAGE = 38` — opaque heap object holding `capacity`
  packed `u8` bytes in its trailing storage. The header's `type_data`
  stores the byte capacity. `opaque: true` so the GC skips it during
  scanning (no traced fields).
- `TYPE_ID_STRING_BUILDER = 40` — non-opaque 2-field heap object:
  - field 0: tagged pointer to the current `ByteStorage` (traced)
  - field 1: tagged int — current length in bytes

Geometric growth: when `len + needed > storage.capacity`, allocate a fresh
`ByteStorage` at `max(2 * capacity, len + needed)`, memcpy old → new,
rewrite `sb.field[0]` with a write barrier. The `StringBuilder` itself
never moves *as an identity* — only its `storage` pointer changes — so
existing references stay valid even across grows.

#### API (`beagle.string-builder` namespace)

```
new(capacity)                — fresh builder, hint capacity
append!(sb, s)               — push a string (any kind)
append-byte!(sb, byte)       — push a u8 (Int 0..255)
append-char!(sb, ch)         — push a 1-char string
append-int!(sb, i)           — itoa straight into the buffer
append-float!(sb, f)         — formatted, no intermediate Beagle string
length(sb), capacity(sb)
byte-at(sb, i), set-byte-at!(sb, i, b)
reverse!(sb)                 — in-place byte reverse (ASCII-safe)
clear!(sb)                   — len = 0, keeps capacity
to-string(sb)                — one alloc + one memcpy → immutable String
```

Every mutator returns the (possibly relocated) builder so chained calls
work. The `!` suffix follows the existing Beagle convention for
mutating ops.

#### Why this is the win for fasta

The FFI version of fasta did per-byte work via `ffi/get-u8` /
`ffi/set-u8` over a buffer it `ffi/allocate`d, then dumped via
`io/write-stdout-buffer`. Two builtin calls per byte plus the FFI
trampoline overhead.

The new version does one `sb/append-byte!` per byte. The hot loop has
**fewer** builtin calls and no FFI trampoline. The byte storage grows
geometrically — average of < 1 reallocation per dozens of MBs. Final
`print(sb/to-string(buf))` is one alloc + one memcpy per flush
(`FLUSH_AT = 65536` bytes). Net result: 2× faster than the FFI version
with no unsafe code, plus a stdlib module that any future "build a
big string" workload can use.

#### GC safety

The non-obvious correctness issue: `ensure-capacity-for` allocates a
new `ByteStorage`, which can trigger a compacting/generational GC,
which moves the `StringBuilder` itself. Every mutator must therefore
*re-fetch* the sb pointer through a `HandleScope`-rooted `Handle`
after the grow returns, and propagate the new pointer back through
its return value (the caller's `sb` local is the GC root, but the
builtin's register copy of the arg is not). This is verified by
running the snapshot test and fasta under `--gc-always`, which
forces a collection on every allocation. Both produce byte-identical
output to the non-`gc-always` runs.

### 4. `knucleotide` ported off FFI

`knucleotide` used `beagle.ffi` only in `read_sequence`: it accumulated
input lines, then memcpy'd them all into a single `ffi/allocate`d byte
buffer one byte at a time, then converted that buffer back to a
String. The new version uses a `string-builder`: pre-sized to the
known total length, then one `sb/append!` per line (memcpy's the whole
line in one builtin call), `sb/to-string` at the end. ~7 % faster on
fasta-250k input. Most of knucleotide's time is in the k-mer hashmap,
not the byte assembly, so the speedup is small — but it's no longer
on FFI.

This port also surfaced and fixed a pre-existing bug: three
`substring(s, start, len)` calls were passing length where end was
expected, crashing on any non-trivial input.

### 5. `revcomp` ported off FFI

`revcomp` was the heaviest FFI user — it allocated its own sequence
buffer with `ffi/allocate` + `ffi/realloc`, used bulk byte ops
(`ffi/translate-bytes`, `ffi/reverse-bytes`, `ffi/copy-bytes_filter`),
and read raw stdin via `io/read-stdin`. The new version uses a
`string-builder` for the sequence buffer, walks input chunks
byte-by-byte to filter newlines and detect FASTA `>` headers, then
calls the new `sb/reverse!` builtin for the actual reverse-complement
(after a `translate` pass that walks bytes through a 256-entry
mutable-array lookup table).

The FFI version on `main` had a pre-existing arity mismatch
(`ffi/copy-bytes_filter` declared with 5 args, called with 6) and
didn't compile. So this is the first revcomp on this branch that
actually runs. Round-trip verified: revcomp(revcomp(input)) matches
input modulo case normalization (revcomp uppercases by design).

The per-byte cost in revcomp is higher than fasta's per-byte append,
because revcomp also does per-byte read+write+lookup during translate,
all going through stdlib helpers + builtins. This is the place where a
batch `string-builder-translate-bytes!(sb, table)` builtin would
likely win another ~20–30 %, but I haven't added one yet — the
current pure-Beagle version is fast enough at 1.4 s on 1 M bases of
input, and adding the batch op is a localized future optimization.

## Things tried and rejected

| change | benchmark | result |
|--------|-----------|--------|
| while-loops → tail-recursive helpers | spectral_norm | **+5 % slower** (1242 → 1310 ms) |
| `(byte_acc << 1) \| escape` → `byte_acc * 2 + escape` | mandelbrot | within noise |
| `math_pow` → tail-recursive `math_pow_helper` | binary_trees | within noise |

The first one is the interesting one: rewriting `while` as tail-recursive
inner loops *did not help* and was in fact slightly slower. The user
expectation is that `while` and tail-recursion compile to equivalent code, so
this measurement says they currently do (good — no runtime regression to
chase). It also says the perf branch's structural rewrites of these
benchmarks were not load-bearing for the speedups it claimed; the wins were
coming from `arr/get` and `arr/swap`, not from the while → recursion
conversion.

## Things that are slow and need runtime work, not benchmark tweaks

### 2.1 `arr/read-field`'s type check is paid per call

This is the dominant overhead in any array-loop benchmark, and the way we
worked around it (`arr/get` that skips the check) only works because the
caller can prove array-ness statically. A real fix is at the runtime / IR
level. Options to investigate:

- Inline cache / type guard: `arr/read-field` is a tiny function. After the
  first successful call from a given site, the type-id should be cached and
  subsequent calls jump straight to `read-field-unsafe` with a single guard
  branch.
- Specialised IR opcode for "read field of array we just allocated", emitted
  when the AST can see the array origin in scope.
- Function inlining: `read-field` itself is small enough that inlining it
  would let the optimiser see the type check as a redundancy check across
  successive calls.

### 2.2 Float boxing in tight accumulators (suspected, not measured here)

The WIP branch shipped an SSA pass to unbox floats. That pass is gated off
on this branch and isn't in scope for the user-level wins. But the
accumulator pattern in spectral_norm:

```
let mut t = 0.0
while j < len {
    t = t + (... * arr/get(u, j))   // every step likely allocates a Float box
    j = j + 1
}
```

is the canonical trigger. `t` is reassigned on every loop step; if each
addition produces a fresh boxed float, n²·10·2 ≈ 10 M float allocations
happen at n=700, all of which die immediately. A runtime that keeps `t` in
a fp register (NaN-tagging, or a typed-locals analysis cheap enough to run
unconditionally) would unlock another big chunk on top of `arr/get`.

Ways to confirm before doing the work: instrument the GC to count Float
allocations during a spectral_norm run, or sample the profiler at the
accumulator line and look for an alloc-fastpath fall-through.

### 2.3 Tree allocation in binary_trees

`binary_trees` is dominated by `TreeNode` allocation (one struct per node,
~16 M nodes at depth 18). No source-level change wins here. This is a pure
GC / allocator throughput problem. If we want this benchmark competitive
the work is in `src/gc/`: either a faster bump pointer for short-lived
allocations, or a generational nursery with a tighter fast path.

### 2.4 Remaining FFI: `mandelbrot`

`fasta`, `knucleotide`, and `revcomp` are all off FFI (sections 3–5
below). The only remaining benchmark that touches FFI is `mandelbrot`,
which uses `builtin/print-byte` for its output stream — a single
FFI-adjacent call rather than a buffer abstraction. Probably wants a
`print-bytes` builtin (or a `print` overload that accepts a byte
storage range from the string-builder), but it's a separate question
from the buffer-assembly problem we just solved.

## How to reproduce

```bash
cargo build --release

bench() {
    local prog="$1" arg="$2" iters="${3:-3}"
    local times=()
    for ((i=0; i<iters; i++)); do
        local start=$(python3 -c 'import time; print(time.time_ns())')
        ./target/release/beag "$prog" "$arg" >/dev/null 2>&1
        local end=$(python3 -c 'import time; print(time.time_ns())')
        times+=($(( (end - start) / 1000000 )))
    done
    printf '%s\n' "${times[@]}" | sort -n | awk -v m=$((iters/2)) 'NR==m+1{print}'
}

bench benchmarks/benchmarksgame/binary_trees.bg 16
bench benchmarks/benchmarksgame/fannkuch_redux.bg 9
bench benchmarks/benchmarksgame/mandelbrot.bg 800
bench benchmarks/benchmarksgame/nbody.bg 500000
bench benchmarks/benchmarksgame/spectral_norm.bg 700
```

## Note on existing test snapshots

`fannkuch_redux.bg`, `spectral_norm.bg`, `binary_trees.bg`, and the stdlib's
own `mutable-array.bg` test all fail `cargo run -- test` on `main` *before*
any of the changes on this branch — the snapshots all expect the function's
return value to appear at the end of the captured output, and it does not.
This is pre-existing on main; the changes on this branch don't make it
better or worse. Logging it here so it doesn't get blamed on the perf work.
