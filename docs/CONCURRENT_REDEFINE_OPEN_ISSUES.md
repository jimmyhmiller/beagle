# Concurrent redefinition: fixed bugs and open issues

This documents the cluster of bugs around redefining structs/enums while other
threads read them, found and fixed in June 2026. Companion to
`docs/GC_REDEFINE_RACE_INVESTIGATION.md`.

## TL;DR

| # | Bug | Status |
|---|-----|--------|
| 1 | Binding zero-clobber: redefine stored raw `0` in the type's binding cell → reader sees `Int` → "property X on Int" | **FIXED** |
| 2 | `property_access` slow-path read past an old instance's fields when its old layout was discarded → garbage value | **FIXED** |
| 3 | GC layout-migration stale-root: migration *moves* a live instance, a concurrent reader's root wasn't always updated → forwarded-corpse crash | **FIXED** (root cause was #5, plus #2) |
| 4 | Test runner ran `test {}` blocks on an **unregistered** main thread → a test that spawns `thread()` under gc-always corrupted GC root scanning | **FIXED** |
| 5 | `eval`/compile blocked on the compiler thread without registering `c_calling` → deadlock vs a concurrent stop-the-world | **FIXED** |
| 6 | Writing a newly-added field on a pre-existing instance | **FIXED** (was briefly broken; restored) |
| 7 | **Concurrent structural redefinition + migration required stale allocation-site invalidation plus a real STW migration before running redefined code** | **FIXED** |
| 8 | A `test {}` block that spawns threads *and* calls `core/gc()` deadlocked in the test runner | **FIXED** |
| 9 | 4-bit `layout_version` wraparound aliasing after stale old-layout allocations kept happening post-migration | **FIXED** |

## Fixed (1–6)

- **#1** `src/runtime.rs` `add_struct`/`add_enum`: reserve the binding slot with
  `BuiltInTypes::null_value()` (`0b111`), not raw `0` (a valid tagged `Int` that
  slipped past the "don't clobber a live value" guard). Regression:
  `resources/concurrent_redefine_race_test.bg`.
- **#2** `src/runtime.rs` `property_access`: while migration is pending, consult
  `previous_definitions`; after migration history is cleared, bounds-check the
  "no old definition" fallback so it can never read past an instance's fields.
- **#3/#6** GC layout-migration is **re-enabled** (`StructManager::insert` schedules
  it). With #2, #4, #5 in place, migrating (and thus *growing*) a live instance
  is safe for grow/shrink redefinitions, which restores writing a newly-added
  field on a pre-existing instance. Regressions:
  `resources/redefine_add_field_write_test.bg` (single-threaded) and
  `resources/struct_structural_redefine_race_test.bg` (concurrent grow/shrink).
- **#4** `src/main.rs` test runner: register the main thread as a GC mutator
  (same dance as `run_main_once`) around running `test {}` blocks. Without it a
  child-initiated stop-the-world didn't count/stop the main thread.
- **#5** `src/runtime.rs` `send_to_compiler_parked`: the mutator-reachable compile
  entry points (`compile_string`, `compile_string_in_namespace`,
  `compile_string_with_file_context`) register `c_calling` while blocked on the
  compiler thread, so a concurrent GC counts them as parked.

Focused redefine regressions pass on generational and mark-and-sweep.

## Fixed #7/#9 — stale allocation sites, then forced STW migration

The tempting symptom-level fix was to widen `layout_version` past the header's
4-bit `type_flags`. That avoided wraparound, but it did not address the real
contract violation: after a structural redefinition, already-compiled functions
could still allocate the old physical layout while stamping it with whatever
layout version was current at compile time. If migration history was retained
forever, those stale sites kept old definitions/plans alive until the 4-bit
version eventually aliased. If migration history was cleared, those same stale
sites could create old-layout objects that no longer had a valid migration or
old-definition entry.

The proper fix is:

- Every compiled struct allocation site, including enum variant constructors,
  embeds the struct id, field names, the layout version it was compiled against,
  and this struct's sticky "structurally redefined ever" guard: a stable boxed
  atomic word address plus a bit mask. If the bit is still false, the site takes
  the original inline allocation path. Once the bit is true, every allocation
  site for that struct permanently falls back to generic construction. The
  fallback maps the already-evaluated field values by name into the current
  layout and fills newly-added fields from literal defaults or `null`. Field
  values are kept in GC-visible local slots and passed to the fallback as a
  pointer/count pair, so this is not capped by a fixed builtin arity. This does
  not consult function source and does not install replacement code that throws.
- Struct spread uses the same bit for explicit override writes: the source object
  is copied/migrated to the current layout, then overrides are patched by field
  name once the struct has ever been redefined.
- Parsed named functions are required to retain source text. If source extraction
  fails for one, compilation fails with an internal compiler error; missing
  source is not a structural-redefinition runtime case. Compiler-synthetic
  functions are explicitly identified by an empty source context and are the
  only named functions that may have no original source slice.
- The compiler sets `Runtime::structural_redefinition_needs_gc`.
- Mutator-reachable `eval` consumes that flag and immediately runs a
  stop-the-world GC from the real Beagle frame before invoking the newly
  compiled top-level function. That migration rewrites all reachable old-layout
  objects while all mutators are stopped and roots are visible.
- After a full-live-set migration completes, `complete_pending_migrations`
  clears both migration plans and retained old definitions again. This is safe
  because allocation sites cannot create old-layout objects once the struct's
  sticky redefined bit has been set.

Regressions:
- `resources/struct_structural_redefine_race_test.bg` is enabled again as a
  `test {}` block.
- `resources/redefine_stale_allocation_no_throw_test.bg`
- `resources/redefine_enum_variant_stale_allocation_no_throw_test.bg`
- `resources/redefine_layout_version_wrap_sticky_bit_test.bg`
- `resources/redefine_struct_spread_sticky_bit_test.bg`
- `resources/redefine_large_struct_dynamic_fallback_test.bg`

The final concurrent failure was in GC coordination/root handling, not in
layout-version width:

- Function entry already polled the stop-the-world flag, but tight reader loops
  could run through atom loads and inline field-cache reads without calling back
  into the runtime. Loop headers now emit the same `__pause` poll, giving those
  loops a real safepoint where live registers are spilled and relocated roots can
  be reloaded after GC.
- `gc_impl` used to subtract one from `registered_thread_count` unconditionally.
  That was only correct when the GC coordinator was itself a registered mutator.
  The main/test thread is not counted there, so a main-thread `core/gc()` with
  two spawned readers waited for only one reader and let the other run during
  migration. The coordinator is now subtracted only when it is registered.
- Minor GC now validates young-generation roots against object starts during
  multi-mutator collections. The stack maps are conservative over frame slots, so
  a dead heap-looking slot must not be promoted from the middle of an object.
  This strict object-start index is rebuilt only for multi-frame STW collections;
  the single-thread path keeps the cheap plausibility check to avoid regressing
  allocation-heavy benchmarks.
- Field read/write slow paths follow forwarding pointers defensively before
  consulting struct metadata, caches, or the write barrier. In a correct STW this
  should be rare, but it prevents a stale from-space object header from being
  interpreted as a normal struct header.

`resources/struct_structural_redefine_race_test.bg` is the enabled regression.
It now cycles `{x,y}` → `{x,y,z}` → `{z,y,x}` → `{x,y}` while reader threads
continuously access `x` and `y`, and it passes in normal and forced-GC modes.

## Fixed #8 — `test {}` block that spawns threads AND calls `core/gc()`

The race regression now runs as a real `test {}` block:
`resources/struct_structural_redefine_race_test.bg`. The test runner registers
the main test thread as a GC mutator while test blocks execute, then unregisters
it with the same `c_calling` dance used by `main()`. The regression passes both
as `beag run --test ...` and through `beag test ...`; the forced-GC path passes
via `beag run --test ... --gc-always`, so the old `main()`+snapshot workaround
is no longer needed.

## Fixed #9 — no stale post-migration allocations

See #7/#9 above. `layout_version` remains the existing 4-bit header field; the
fix is that allocation sites route through their sticky-bit generic fallback
after a forced STW migration, and migration history is cleared once that
migration has completed. `get_old_definition` also searches retained definitions
from newest to oldest for defensive behavior while migration is pending.

## OPEN — `make_closure` `is_heap_pointer` abort under CPU starvation 🔴

Reliably reproduced by the soak harness (`smoke/soak_long.bg`) under
`--gc-always` while all cores are saturated — **2/2 crashes in ~8–12 s**:

```
thread '<unnamed>' panicked at src/types.rs:420:9:
assertion failed: BuiltInTypes::is_heap_pointer(pointer)
  ... beag::builtins::objects::make_closure ...
thread caused non-unwinding panic. aborting.   (exit 134 / SIGABRT)
```

`make_closure` (src/builtins/objects.rs:233) reads its `function` argument,
sees the `HeapObject` *tag*, then `HeapObject::from_tagged(function)` asserts
`is_heap_pointer` and fails — i.e. `function` carries the HeapObject tag bits but
a non-pointer payload: a **torn / clobbered tagged value**, the B4 GC-safety
class (a maybe-pointer racing a concurrent moving GC / non-atomic 8-byte read).
Pre-existing (independently observed during the §2.1 cross-GC review under two
competing GC-suites saturating all cores); vanishes under normal load. The same
starvation window also produced a `"Cannot access property 'resume' on null"` in
`beagle.async/scheduler-loop` (gc-always card-table path).

Repro: `for i in $(seq 1 $(sysctl -n hw.ncpu)); do yes >/dev/null & done;
./target/release/beag run --gc-always smoke/soak_long.bg`. The standing rule
(any maybe-pointer live across a GC safepoint must be in a GC-scanned slot)
needs auditing for the `make_closure` argument path. Tracked also in FUTURE_WORK
§1.3.
