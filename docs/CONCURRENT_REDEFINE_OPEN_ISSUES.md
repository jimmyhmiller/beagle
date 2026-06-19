# Concurrent redefinition: fixed bugs and open issues

This documents the cluster of bugs around redefining structs/enums while other
threads read them, found and (mostly) fixed in June 2026, plus the issues that
remain open. Companion to `docs/GC_REDEFINE_RACE_INVESTIGATION.md`.

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
| 8 | A `test {}` block that spawns threads *and* calls `core/gc()` deadlocks in the test runner | **OPEN** |
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
- `resources/struct_structural_redefine_race_test.bg` is enabled again.
- `resources/redefine_stale_allocation_no_throw_test.bg`
- `resources/redefine_enum_variant_stale_allocation_no_throw_test.bg`
- `resources/redefine_layout_version_wrap_sticky_bit_test.bg`
- `resources/redefine_struct_spread_sticky_bit_test.bg`
- `resources/redefine_large_struct_dynamic_fallback_test.bg`

Historical manifestations:

- **mark-and-sweep**: even a grow/shrink workload (`{x,y}`↔`{x,y,z}`) concurrent
  with readers hangs or crashes (signal 6 / forwarded-corpse). Generational runs
  the same workload clean (12/12).
- **generational**: grow/shrink is clean, but a field-**reorder** redefine
  (`{x,y}`→`{z,y,x}`) concurrent with readers hangs the collector.

Single-threaded migration (any of these) is fine; the hazard is concurrency.
`resources/struct_structural_redefine_race_test.bg` is the enabled regression.

### reorder repro (generational)

**Repro** (`run` it; hangs ~forever, exit 124 under timeout):

```
namespace r
use beagle.core as core
struct Point { x, y }
letonce running = atom(true)
letonce d1 = atom(0) letonce d2 = atom(0)
fn reader(p) { while deref(running) == true { p.x  p.y } }
fn wait(a,k){ if deref(a)>=1 {true} else if k>2000000000 {false} else {wait(a,k+1)} }
fn main() {
  let anchor = Point { x: 111, y: 222 }
  thread(fn(){ reader(anchor) swap!(d1, fn(x){1}) })
  thread(fn(){ reader(anchor) swap!(d2, fn(x){1}) })
  let mut r = 0
  while r < 40 {
    eval("namespace r
struct Point { z, y, x }")          // <-- REORDER: existing fields change index
    core/gc()
    eval("namespace r
struct Point { x, y }")
    core/gc()
    r = r + 1
  }
  reset!(running, false)
  wait(d1,0) wait(d2,0) println("done")
}
```

**What's known (from `sample` on the hung process):**
- The GC coordinator (the redefining thread, inside `core/gc()` →
  `gc_impl` → `full_gc` → `mark_and_sweep` migration) is stuck spinning in the
  migration walk (`Runtime::get_struct_by_id` + an internal `Mutex`/RwLock).
- Both reader threads are correctly **parked** at `__pause` (the world is
  stopped). So this is NOT a reader-vs-migration race; the coordinator itself
  fails to make progress inside the migration.
- It is **specific to reorder**: the identical harness with grow/shrink
  redefinitions (`{x,y}`↔`{x,y,z}`) completes in ~0.2 s (8/8). Adding a single
  reorder redefine (`{z,y,x}`) makes it hang (3/3).
- It is **concurrency-specific**: the same reorder redefinitions + `core/gc()`
  **single-threaded** complete fine (`reorder_single.bg`: prints `1 2 1 done`).

**Likely area:** the mark-and-sweep / generational `migrate_outdated_structs` /
`copy_with_migration` path when existing field *indices* change AND reader-thread
stacks are present as roots. Suspect a lock-ordering or non-terminating loop that
only manifests when (a) a field moves index and (b) there are other (parked)
mutator stacks/roots to process. Start by instrumenting the migration loop to log
per-object progress and which lock `get_struct_by_id` is blocked on, and check
whether the migrated reorder object is repeatedly re-classified as old-layout.

**Impact before the fix:** reordering struct fields in a running multi-threaded
program could hang once stale allocation sites and version aliasing lined up;
mark-and-sweep could also hang/crash on grow/shrink workloads.

## OPEN #8 — `test {}` block that spawns threads AND calls `core/gc()` deadlocks

A `test {}` block (run via the test runner's bare-trampoline invocation) that
both spawns `thread()` readers and calls `core/gc()` deadlocks, even though the
*identical* code as `main()` + snapshot runs fine. That's why
`struct_structural_redefine_race_test.bg` is a `main()`+snapshot test, not a
`test {}` block. The bare-trampoline test invocation differs from the
`beagle.async/__main__` entry `main()` uses; wrapping the test thunk in
`run-cooperative` was tried and did NOT fix it (and is currently removed because
plain registration (#4) fixes the gc-always corruption on its own). Root cause
not yet isolated — likely the same area as #7 since both involve `core/gc()` +
parked readers + migration.

## Fixed #9 — no stale post-migration allocations

See #7/#9 above. `layout_version` remains the existing 4-bit header field; the
fix is that allocation sites route through their sticky-bit generic fallback
after a forced STW migration, and migration history is cleared once that
migration has completed. `get_old_definition` also searches retained definitions
from newest to oldest for defensive behavior while migration is pending.
