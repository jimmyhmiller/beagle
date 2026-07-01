# Adopting the `gc-rust` concurrent generational GC into Beagle

## Status

- **Vendored** ✓ — `src/gc/gcrust/` (std-only, zero deps), behind the `gcrust`
  cargo feature. Its own 108-test suite passes inside Beagle
  (`cargo test --features gcrust gc::gcrust::`).
- **Phase 1 (STW copying collector) ✓ DONE**.
- **Stage 5 (STW *generational* collector) ✓ DONE** — nursery + two tenured
  semi-spaces + card table; minor GC (promote young survivors, dirty-card +
  stack-old-gen old→young roots) and major GC (full evacuate + empty nursery).
  `cargo run --features gcrust -- test resources/` is 100% green (549/0/1).
  Tunables: `BEAGLE_GCRUST_NURSERY_MB` (default 64), `BEAGLE_GCRUST_SPACE_MB`
  (tenured per-space, default 256); `BEAGLE_GCRUST_MAJOR_ONLY=1` forces
  major-only (debug/bisect).
- **Phase 2 (concurrent) ✓ WORKING** (env-gated `BEAGLE_GCRUST_CONCURRENT=1`).
  A stop-the-world-free major collection: two short STW pauses with a
  concurrent trace between, mutators running. Validated: full suite green with
  concurrent on (549/0/1), live-coding smoke 10/10, and a bespoke
  multi-thread allocate+mutate stress (4 threads, tenured-pressure) 0 fails
  over repeated soak runs. The default (concurrent-off) generational path is
  unchanged. Stages below.
- **Phase 2 build log:**
  - **Stage A ✓** — phase machine (`gc_phase`/`barriers_active`) + a
    concurrent-dirty write barrier (during the copying phase, every mutation of
    a from-space tenured object marks its card, value-independent). Inert until
    a concurrent cycle runs; default path stays green (549/0/1).
  - **Stage B ✓** — out-of-band **side-table** copy/repoint (`major_gc_via_table`
    + `copy_via_table`/`trace_slot_via_table`/`repoint_slot`). Forwarding lives
    in a `HashMap<from,to>`, NOT in object headers — because Beagle's single
    word header (type_id+size+flags) can't hold a forwarding pointer without
    corrupting live mutator header reads (e.g. the `set!` inline-cache struct-id
    check), unlike gc-rust's separate `gc_word`. This is the mechanism a
    concurrent cycle needs. **Validated stop-the-world**: full suite green with
    every collection forced through it (`BEAGLE_GCRUST_MAJOR_ONLY=1
    BEAGLE_GCRUST_TABLE=1`, 549/0/1). A separate repoint pass replaces
    header-forwarding.
  - **Stage C ✓** — the live concurrent driver. `Runtime::gc_impl`'s
    multi-thread path now, when `begin_concurrent` fires (concurrent enabled,
    no migration pending, tenured pressure), resumes mutators → runs the
    concurrent trace on the triggering thread → re-pauses (STW#2) → finishes
    (drain dirty cards, re-scan roots + nursery-as-roots, repoint via table,
    flip). The trace holds the alloc mutex so mutators block only on
    *allocation*; it uses atomic word-copy (`copy_conc`) for objects a mutator
    may be writing. The concurrent major is **tenured-only** — the nursery is a
    root set (walked + repointed), not evacuated. Card marking is extended: in
    the copying phase every mutation of a from-space tenured object dirties its
    card so STW#2 can re-copy it. Post-major, old→young cards are rebuilt.
  - **Liveness fix (the crux)** — a thread whose nursery fills while a
    concurrent cycle holds `gc_lock` (with `is_paused==0`) would spin
    `WouldBlock`→`__pause` and exhaust its allocation retries (manifests as a
    spurious OOM). Fixed in `gc_impl`'s `WouldBlock` arm: if the world isn't
    stopped AND a concurrent cycle is active (`concurrent_gc_active`), register
    as scannable (`register_c_call`) and **block on `gc_lock`** until the cycle
    finishes, then retry. Gated on `concurrent_gc_active` so the ordinary
    stop-the-world path (concurrent off) is byte-for-byte unchanged (`__pause`)
    — broadening it to the non-concurrent STW-not-yet-armed window regressed a
    continuation stress test into a hang, so it stays gated.
  - **Stage D ✓** — validated: full suite (concurrent on) 549/0/1; smoke 10/10;
    the multi-thread stress soak 0 fails. NOTE the suite is mostly
    single-threaded (takes `gc_impl`'s no-STW fast path), so concurrent only
    engages with >1 thread + tenured pressure — deeper soak on real
    multi-thread workloads (HTTP server) is worthwhile before trusting it in
    production.

  Design details + risk below.

## Performance (integration + concurrent)

The gcrust GC is a **fully integrated, competitive** drop-in, and the concurrent
mode is a real latency win. Measured on Apple Silicon, release builds, best of
several runs; default GC = beagle's generational (mark-sweep old gen).

**Throughput — parity with the default GC.** The integration work (below)
took gcrust from ~2.2× slower + OOM to parity:
- `binary_trees` depth 21 (canonical GC benchmark): default **8.86s**, gcrust
  **9.03s** (~2%). Was 31s / OOM before the integration fixes.
- `fib` (compute-bound): gcrust ≈ default.

**Concurrent — ~4× shorter GC pauses** (latency), at a modest throughput cost.
On a 4-thread build-and-compute workload (`conc_bench3`, big long-lived data +
compute) forcing major collections:
- STW (concurrent off): **max GC pause ~6.6–7.1 ms**.
- Concurrent on: **max GC pause ~1.7 ms** (a ~4× reduction) — the copy runs
  while mutators execute; only two short pauses (snapshot ~0.5 ms, reconcile
  ~1.7 ms) stop the world.
- Cost: ~5–15% total throughput (the copy + STW reconcile is more total work,
  split across pauses). Classic latency/throughput tradeoff — a clear win for
  latency-sensitive multi-threaded workloads (e.g. servers).

Reproduce: `BEAGLE_GC_PAUSE=1 [BEAGLE_GCRUST_CONCURRENT=1] target/release/beag
run <bench>` prints each stop-the-world window.

### What made it competitive (integration fixes)
1. **Inline JIT allocation fast path** — `allocator_frontier`/
   `sync_allocator_frontier` expose the nursery bump window so single-threaded
   allocation is a bump, not a locked slow-path call. (Was `(0,0)` = every alloc
   locked → the initial throughput killer.)
2. **Growable heap** — `ensure_to_space_fits` grows the spare tenured space to
   fit the worst-case live set before each major, and `grow()` doubles it. (Was
   fixed-size → OOM/abort on `binary_trees`.)
3. **Finalizer side list** — `register_finalizable` tracks finalizable objects
   in a `Vec`, so a collection is O(finalizable count), NOT a full-heap walk
   with a per-object registry hash lookup (which was ~40% of runtime on
   allocation-heavy code).
4. **Migration check hoisted** — `copy_object` reads a per-collection
   `MIGRATE_ACTIVE` flag instead of taking the struct-registry RwLock per
   object.
5. **Barrier fast path** — the write barrier uses fixed-range `contains` (no
   cursor reads) and skips the concurrent branch entirely unless concurrent GC
   is on.
6. **Cheap minor GC** — the dirty-card scan is skipped when no cards are dirty
   (avoids an O(tenured) index build every minor).

### How the concurrent pauses stay short
- **Repoint during the trace, not at STW#2.** Because mutators keep the
  from-space invariant (they never read to-space until the flip), the trace
  copies AND repoints to-space fields as it goes, and marks young cards inline.
  So STW#2 is O(dirty + roots + nursery), not O(live) — no full repoint pass,
  no post-flip card rebuild.
- **All non-migration majors go concurrent** (`begin_concurrent` owns the
  periodic + pressure schedule when multi-threaded), so no long STW major slips
  through. Single-threaded programs keep the STW path (concurrency can't help
  one thread). A small nursery keeps the O(nursery) nursery-as-roots walk cheap.

## Phase 2 design (concurrent) — feasibility + plan

gc-rust's concurrent collector is a **SATB concurrent copying collector**: two
short STW pauses with a concurrent copy between them.
- **STW #1**: flip `gc_phase → Copying` (barriers on), snapshot + copy roots,
  resume threads.
- **Concurrent**: Cheney-copy reachable objects from→to (atomic word-copy + CAS
  forwarding), **without** repointing parent fields.
- **STW #2**: drain SATB, re-scan roots, **full to-space re-scan repoints every
  field**, swap.
- **Mutator obligations while `barriers_active()`**: safepoint-poll; SATB
  barrier (log the *old* pointer *before* a store); replication barrier (mirror
  the *new* value into the to-space copy *after* the store, at the field
  offset). No read barrier needed.

### What beagle already has (thread half — ready)
The STW rendezvous is solid and reusable: `Runtime::gc_impl`
(`src/runtime.rs:6320`), `is_paused` + `__pause` (`src/builtins/threads.rs:19`),
the `thread_state` condvar handshake, c-calling/FFI accounting, per-thread
`gc_frame_top` roots, safepoint polls at fn-entry/loop-headers
(`emit_gc_poll`, `src/ast.rs:1385`), and the `alloc_end`-disarm trick that
forces every allocation to a slow path where a safepoint can fire.

### The gap (barrier half — the real work)
Beagle does **not** intercept all the stores a concurrent copying collector
needs, and the ones it does carry the wrong data:
1. The emitted barrier is **post-store `(object, new_value)`** — no old value
   (SATB) and no field offset (replication).
2. It only fires for **mutations of existing objects**, filtered to old→young
   card marks; **initializing** writes (struct construction `ast.rs:3369`,
   closure capture `runtime.rs:7189`, JIT inline-alloc field init) write raw.
   *(Initializing writes to fresh post-snapshot objects don't actually need the
   concurrent barrier — such objects aren't in the snapshot, so aren't copied —
   so mutation-only coverage is plausibly sufficient, but soundness rests on
   that coverage being **complete**: one un-barriered mutation of a pre-existing
   object = silent heap corruption.)*

### Recommended approach
Use an **incremental-update (Dijkstra) + replication** barrier instead of SATB:
it works **post-store** with `(obj, field_addr, new_value)` — much closer to
beagle's current barrier (just add `field_addr`). `field_addr` gives the
replication offset; `new_value` is shaded for liveness; the STW #2 full
to-space re-scan is the backstop. This avoids the pre-store old-value capture
SATB requires.

### Stages
1. **Barrier rework (codegen)** — add `field_addr` to the ~20 mutation barrier
   sites (`ast.rs`, `primitives.rs`, `collections/persistent_*`, `runtime.rs`);
   route them to a runtime barrier that still does card-marking today. Keep the
   suite green STW (no behavior change). Audit that **every** mutation of a
   pre-existing pointer field is covered.
2. **Phase machine + buffers** — `gc_phase`/`barriers_active` + mark/replication
   log in `GcRustHeap`; inert until a concurrent cycle runs.
3. **Concurrent driver** — restructure `Runtime::gc_impl` (or a new path) to do
   the two-phase STW with a concurrent copy between, reusing `is_paused`/
   `__pause`. Implement the concurrent protocol (atomic copy, CAS bit-3
   forwarding, deferred repointing, drain, flip) in the adapter. Start with
   concurrent **major** only (minor is already short); env-gated.
4. **Validation** — stress under concurrency (this is where it's hard): the
   networking/thread soak tests, `smoke/`, gc-stress, plus new race stress.

### ⚠️ Risk
This is the highest-risk stage: it modifies write-barrier **codegen** and the
core **GC driver** — both load-bearing for the *working* generational GC — and
concurrent-GC bugs are races that manifest as intermittent, hard-to-reproduce
corruption. It is a multi-session effort, not a bounded patch.

## Generational design (Stage 5)

All in `src/gc/gcrust_adapter.rs`, reusing gc-rust's `BumpAllocator` +
`CardTable` primitives and the `ObjectModel`/`PtrPolicy` Cheney core. Driven by
beagle's existing stop-the-world (no gc-rust safepoint machinery).

- **Layout**: `nursery: BumpAllocator` (young) + `tenured: [BumpAllocator; 2]`
  (old semi-space, `from_idx`) + `card_tables: [CardTable; 2]` (one per tenured
  space). New objects → nursery; objects too big for the nursery → tenured.
- **Minor GC** promotes nursery survivors into tenured-from. Roots: (1) all
  beagle roots (frames + extra), (2) **dirty cards** (old→young via
  `build_object_start_index` + per-card object scan), (3) a **one-level scan of
  old-gen objects directly referenced from stack roots** — the second old→young
  source, needed because some construction-time stores aren't barriered (this
  was the `bug_b1`/`stdlib_*` cycle bug). Then a Cheney closure over the
  promoted region; then nursery reset + card clear.
- **Major GC** evacuates everything live (nursery ∪ tenured-from) into
  tenured-to, runs struct migration + `complete_pending_migrations`, sweeps
  finalizers in the reclaimed spaces, swaps tenured, and **empties the nursery**
  (beagle triggers GC on nursery-full, so a major that left the nursery
  populated would loop).
- **`gc()` policy**: major when a migration is pending, periodically (every
  `FULL_GC_FREQUENCY`), when the nursery is empty (pressure is tenured), or when
  worst-case promotion won't fit tenured-from; otherwise minor.
- **Write barrier**: `write_barrier`/`mark_card_unconditional` mark the tenured
  object's card on an old→young store (mirrors `generational::write_barrier`).

## What was done (Phase 1)

The engine was made object-model-agnostic and driven by Beagle's own header:

- **`ObjectModel` trait** (`src/gc/gcrust/semi_space.rs`) abstracts the two
  points the Cheney core was coupled to `TypeInfo` at — object size and traced
  slots — plus forwarding read/write, a plausibility predicate, and object copy
  (so migration can hook in). `TypeInfoModel` reproduces gc-rust's native
  behavior (keeps the 108 tests green); `BeagleObjectModel`
  (`src/gc/gcrust_adapter.rs`) drives from Beagle's 8-byte header.
- **`PtrPolicy::reencode_ptr`** added so a moving GC can restore Beagle's exact
  low-3-bit tag (Float/Closure/HeapObject) after relocation.
- **`alloc_raw(size, align)`** on the bump allocators — allocate N bytes with a
  caller-owned header.
- **`GcRustHeap: Allocator`** (`src/gc/gcrust_adapter.rs`) — the seam: raw
  allocation (stamps Beagle's header), the root bridge (`gc_frame_tops` via
  `StackWalker` + `extra_roots`), and a synchronous STW `gc()` (Beagle already
  stops the world).

Bridging pieces that real `.bg` programs required (each mirrors Beagle's
existing compacting GC):

1. **Continuation segments** — a captured continuation's detached stack-segment
   frames hold live roots; `BeagleObjectModel::scan_slots` walks them
   (`walk_segment_gc_roots`, absolute-FP rebuild + restore).
2. **Plausibility gate** — Beagle is *mostly*-precise: a traced slot can hold a
   non-pointer bit pattern with a heap tag. `process_slot` skips pointers that
   aren't aligned/in-bounds/valid objects (cf.
   `compacting::is_plausible_from_space_object`). gc-rust's native model stays
   truly precise (panics under the armed detector).
3. **Finalizers** — post-copy, pre-swap sweep of the old from-space enqueues
   finalizers for dead FFI objects (`copy_live` + `sweep_finalizers` +
   `swap_and_reset`).
4. **Struct migration** — `BeagleObjectModel::copy_object` migrates redefined
   structs to their new layout during copy; `gc()` calls
   `complete_pending_migrations(revision)` afterward (without it, stale plans
   accumulate and mis-migrate once the 4-bit layout version wraps).

### Known Phase-1 limitations (deferred)

- Non-generational (full copy every GC) and **stop-the-world** — the whole
  point (concurrency) is Phase 2.
- Fixed-size semi-space; `grow()` is a no-op (overflow surfaces as clean OOM).
  A live program with a >512 MB live set needs `BEAGLE_GCRUST_SPACE_MB` raised.
- Inline JIT bump fast path disabled (`allocator_frontier` returns `(0,0)`):
  every allocation takes the locked slow path. A TLAB frontier is a perf
  follow-up.

## Executive summary

The two GCs share the hard-to-get-right fundamentals, which is why this is
worth doing:

- Both are **precise** (no conservative stack scanning).
- Both discover roots via an **explicit shadow-stack frame chain**
  (Henderson-style), and both hand the collector **slot addresses** so a
  moving collector can update roots in place.
- Both are **moving/copying** with a **bit-flag forwarding marker** in a
  header word.
- Both have a **bump-allocator TLAB fast path** + Rust slow path.

But three things differ, and two of them are load-bearing:

| Concern | gc-rust | Beagle | Difficulty |
|---|---|---|---|
| Object header | 16-byte `Full` (forwarding word @0, `type_id` @8) | 8-byte packed (type_id, size-in-words, flags, bit-3 forwarding) | **HARD** |
| Layout / scan model | static per-type `TypeInfo[type_id]` table (monomorphized) | header-driven `num_traced_slots()` prefix convention (dynamic) | **HARD** |
| Pointer tagging | `IdentityPtrPolicy` (raw ptr; low-3-set = non-ptr) | 3-bit low tags; traced = Float/Closure/HeapObject | MEDIUM |
| Root feeding | own `Frame`/`FrameOrigin` + `RootSource` trait | `gc_frame_tops: &[usize]` + `extra_roots: &[(*mut usize, val)]` | MEDIUM |
| Allocation granularity | `alloc(type_id, site_id)` → typed, header stamped | `try_allocate(words, kind)` → raw, caller stamps header | MEDIUM |
| Write barrier | card table (gen) + SATB/replication (concurrent) | card table (gen), runtime-call not yet inlined | MEDIUM |
| Thread coordination | own `ThreadState` handshake, mutator-becomes-GC | own STW rendezvous (`is_paused`, `__pause`/park) | **HARDEST** |

### The key realization

gc-rust's collector is parameterized on exactly the two traits that differ —
`PtrPolicy` (tagging) and `ObjHeader` (header) — **and** on `TypeInfo` /
`scan_object` for layout. gc-rust's scanner is **type-table-driven**; Beagle
is **dynamically typed** and stores size/traced-slot info **in the header**.

So "adoption" is not "flip a feature." It is: **keep Beagle's header,
tagging, and header-based scanning; reuse gc-rust's collector *engine*** (the
nursery + tenured semi-spaces, Cheney copy loop, card table, SATB barrier,
and — the whole point — the concurrent safepoint handshake). The vendored
`scan_object` (TypeInfo-based) and the typed `alloc` path get replaced with
Beagle's header-based equivalents; the Cheney core, space management, and
concurrency machinery are what we actually want to reuse.

### Do NOT change

- **Do not adopt gc-rust's 16-byte header.** Beagle's entire codegen bakes in
  the 8-byte header offset, and doubling every object header is a real
  memory/perf regression. gc-rust's `ObjHeader` is a *trait* — implement it
  over Beagle's existing 8-byte `Header` (`src/types.rs:157`). This keeps
  codegen untouched. Its `Compact` (8-byte) header proves the engine already
  supports 8-byte headers; but ours packs size+flags differently, so we need
  our own impl, not `Compact`.

## The seam (recap)

- Beagle GC interface: `trait Allocator` (`src/gc/mod.rs:40`). The collection
  entry point is `fn gc(&mut self, gc_frame_tops: &[usize], extra_roots:
  &[(*mut usize, usize)])` — **synchronous, stop-the-world**.
- Roots: `StackWalker::walk_stack_roots` (`src/gc/stack_walker.rs:45`) yields
  `(slot_addr, value)`; extra roots assembled in `Memory::run_gc`
  (`src/runtime.rs:4130`).
- Object contract: `Header` (`src/types.rs:157`), `num_traced_slots()`
  (`src/types.rs:868`), bit-3 forwarding marker, 3-bit value tags
  (`BuiltInTypes`, `src/types.rs:10`).
- STW rendezvous: `Runtime::gc_impl` (`src/runtime.rs:6313`) + `__pause`
  (`src/builtins/threads.rs:19`).

## Phased plan

### Phase 1 — Generational copying, stop-the-world (get it running)

Goal: `--features gcrust` gives a working generational copying GC, driven by
Beagle's *existing* STW rendezvous. No concurrency yet. Proves the
header/tagging/root/scan bridge before touching thread coordination.

1. **`BeagleObjHeader`**: `impl gcrust::ObjHeader` over Beagle's 8-byte
   `Header` — 8-byte SIZE, type_id at byte 7, forwarding via bit 3.
2. **`BeaglePtrPolicy`**: `impl gcrust::PtrPolicy` using `BuiltInTypes`
   tagging. Must classify Float(001)/Closure(101)/HeapObject(110) as
   pointers (note: identity policy would wrongly skip Float — its low bits
   are set). Untag on read, re-tag on write-back.
3. **Header-based scan**: replace the vendored `scan_object`'s TypeInfo path
   with Beagle's `num_traced_slots()` prefix walk. This touches the Cheney
   loop in `semi_space.rs` and `heap.rs` (they call `scan_object` /
   `read_type_id` + `TypeInfo` size). Two sub-options:
   - (a) Modify the engine to size objects from Beagle's header
     (`fields_size()` + `header_size()`) and trace via `num_traced_slots()`.
     **Recommended** — no per-object TypeInfo synthesis.
   - (b) Synthesize a `TypeInfo` per object from the header each scan. Simpler
     to bolt on, slower.
4. **`GcRustHeap` adapter**: `impl Allocator`. `try_allocate(words, kind)`
   bumps gc-rust's nursery `AtomicBumpAllocator`; `allocator_frontier` /
   `sync_allocator_frontier` expose the nursery window for Beagle's inline
   bump fast path.
5. **Root bridge**: in `Allocator::gc`, wrap `gc_frame_tops` + `extra_roots`
   in a `RootSource` and run gc-rust's *minor*/*major* collection STW.
6. **Write barrier**: route `Allocator::write_barrier` to gc-rust's card
   table (already how Beagle's generational GC works — near drop-in).
7. Feature wiring: add the `MutatorState<GcRustHeap>` arm in `src/main.rs`
   (~`:2283`) and `type Alloc` alias.

**Exit criteria:** `cargo run --features gcrust -- test resources/` is
100% green.

### Phase 2 — Concurrency (the actual payoff)

Only after Phase 1 is green. This is where the two thread models collide and
the real work is.

1. **Safepoint model**: decide whether gc-rust's `ThreadState` handshake
   *replaces* Beagle's `is_paused`/`__pause` rendezvous, or whether the
   concurrent collector is driven *through* Beagle's existing safepoints.
   Replacing is cleaner long-term but touches `src/runtime.rs` thread
   coordination and `src/builtins/threads.rs`.
2. **Inline SATB write barrier** in codegen: today Beagle's write barrier is
   a runtime call; concurrent marking needs the SATB log on the fast path.
   `LirOp::InlineWriteBarrier` is already stubbed (`src/lir/mod.rs:179`) and
   `get_card_table_biased_ptr` exists to support inline barriers.
3. **Inline safepoint poll** at loop back-edges (Beagle already polls
   `is_paused`; gc-rust polls `thread.state` — unify).
4. Wire `barriers_active()` / SATB drain into the two short STW pauses of
   gc-rust's `concurrent_collect`.

**Exit criteria:** concurrent major GC runs with mutator threads live; smoke
test (`python3 smoke/live_coding_smoke.py`) + full suite green; measured
pause-time reduction vs Phase 1 STW.

## Risks / open questions

- **Continuations & segmented stacks**: Beagle has captured continuations
  with detached stack segments (`walk_segment_roots`) and special type ids
  (`TYPE_ID_CONTINUATION`, `TYPE_ID_FRAME`). The scan bridge (Phase 1.3) must
  handle these, not just plain structs.
- **Boxed floats**: Float is a heap object but tagged 001 (low bits set) —
  the single most likely source of a silent tracing bug. Cover it first in
  the `PtrPolicy` tests.
- **Namespace binding cells & handle scopes**: extra-root sources that are
  updated in place — must flow through the `RootSource` bridge.
- **Does concurrency need per-object mark state?** gc-rust's concurrent path
  uses replication + SATB, not a mark bit; confirm Beagle's header bit budget
  (only bit 3 is reserved for forwarding) is enough, or steal another flag
  bit.
- **Header size assumption is load-bearing** — Phase 1.1 must keep 8 bytes or
  every codegen offset breaks.

## Bottom line

The foundations line up unusually well (precise + shadow-stack + copying on
both sides), so this is a real, tractable port rather than a rewrite. The
generational copying collector (Phase 1) is a bounded adapter job. The
*concurrent* part (Phase 2) is the genuine lift, because it requires
replacing/merging thread-coordination and emitting inline barriers — you
cannot get concurrency purely behind the synchronous `Allocator::gc` seam.
