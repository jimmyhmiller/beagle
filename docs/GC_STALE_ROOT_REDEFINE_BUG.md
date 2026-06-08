# Stale-root crash during live struct redefinition (a GC bug)

**Status:** **fixed 2026-06-06.** The forwarding crash came from
`MarkAndSweep::migrate_outdated_structs`: old-layout objects were moved after
sweep, but active Henderson-frame roots and captured continuation-frame roots
were not rewritten before the old copies were freed. The minor-GC root verifier
stayed silent because this was an old-generation migration, not a missed
young-generation promotion root.

The fix updates every root class during old-layout migration and enables
`resources/gc_redefine_stale_root_gcalways.bg` as a regression test. A separate
concurrent `StructManager` metadata race was also confirmed and fixed with a
synchronized registry and owned snapshots; property-cache publication is now
atomic on the Rust side. See `docs/GC_REDEFINE_RACE_INVESTIGATION.md` for the
full resolution and historical investigation.

---

## 9. Headless reproduction + refined diagnosis (2026-06)

The earlier claim that this needs the real beagle-zelda environment is **wrong**.
It reproduces deterministically headless. Canonical repro (kept as a `// Skip`
regression test):

`resources/gc_redefine_stale_root_gcalways.bg`

```
BEAGLE_GC_VERIFY_ROOTS=1 cargo run --features generational -- \
    test resources/gc_redefine_stale_root_gcalways.bg
```

It aborts in `property_access` with `is_forwarding=true` — bit-for-bit the
zelda crash — within seconds, every run.

### What the repro needs (each bisected and proven essential / non-essential)

- **Two threads + frequent GC** (`// gc-always`): required. A reader thread
  holding a struct instance, plus another thread driving GC.
- **Redefining the struct that is *in active use***: required. Redefining an
  *unrelated* struct does **not** crash (passes ∞). Pure heavy allocation on
  the second thread (no `eval`/redefine) does **not** crash. `eval`-ing a
  non-redefining expression (`"1 + 1"`) does **not** crash. So it is the
  **redefinition semantics of the live struct**, not the compile machinery and
  not multithreaded GC on its own.
- **Mechanism of "redefinition of the live struct":** bumping the struct's
  layout version makes existing instances *old-layout*, so the inline property
  cache rejects them and every field read on the held instance routes through
  the **`property_access` builtin slow path** instead of the inline fast path.
  The crash only happens once reads are on that slow path.
- **Tier-up / specialization:** *not* required (only modulates timing). With
  `--no-auto-specialize` the same corruption appears (sometimes as a garbage
  receiver / "field does not exist on <garbage>" instead of a clean
  `is_forwarding`); it is just much slower to hit.
- **The single-thread GC fast path** (`Runtime::gc_impl`,
  `if self.memory.threads.len() == 1`): *not* the cause. Forcing the
  stop-the-world path for every GC does **not** fix it. (Note, though, that
  `memory.threads.len()` and `registered_thread_count` were observed to
  disagree — a latent inconsistency worth tightening separately.)

### What is ruled out about the stale value's location

The end-of-GC root verification (§4, now implemented — see below) **never
fires**, including under forced stop-the-world where every thread's stack is
scanned. So the stale value is **not** in any GC-enumerated root slot at the end
of a collection: not a stack GC-slot, not a shadow-stack handle, not a binding
cell. Static analysis of the reader's machine code makes this sharper and, on a
single-thread timeline, *contradictory*:

- The held instance's stack slot (`[fp-0x18]` for the param) is written exactly
  once in the prologue and read fresh on each use; nothing overwrites it.
- The only GC safepoint reachable in the reader between that write and the use
  is the entry `__pause`; at that point the value is in the slot, so the
  collection updates it, and verification confirms the slot is correct
  afterward.
- Yet the value loaded from that slot at the use site is a forwarded young
  corpse.

A correctly-updated slot, never overwritten, that nonetheless yields a forwarded
pointer can only mean the slot (or the object) is being **corrupted by a
concurrent write from the redefiner thread** — i.e. a *torn-write / data race*
in the redefinition path racing the GC or the reader, **not** a clean
root-enumeration logic gap. This is consistent with the recent commit history
(a string of redefinition concurrency-race fixes: unlocked `dispatch_tables`
HashMap, unlocked `jump_table_pages` Vec, torn cache writes) and the
"~2% residual torn-read crash" noted in `docs/PROTOCOL_REDEFINE_HANDOFF.md`.

### Next steps (for whoever picks this up)

1. Treat it as a **redefinition/GC data race**, not a missing root. Run the
   repro under ThreadSanitizer (or lldb watchpoints on the held slot address)
   to catch the racing writer.
2. Audit the redefinition path (`Compiler::add_struct` →
   `revert_all_specializations` / `invalidate_all_property_caches`,
   `Structs::insert`, jump-table swap) for writes that race a concurrent GC or
   a concurrent reader's `property_access` slow path **without** holding
   `gc_lock` / stopping the world.
3. The end-of-GC verification (§4) is implemented and gated on
   `BEAGLE_GC_VERIFY_ROOTS`; keep it on while bisecting. It is a true negative
   here (the defect is not an unupdated enumerated root), which is itself a
   useful signal — do not chase root-enumeration fixes.

---

## 1. Symptom

Live-redefining a struct in the running beagle-zelda game (via the REPL's
`eval` while the raylib loop runs) intermittently aborts:

```
Struct not found! struct_pointer=0x3800000da06, struct_type_id=229377,
  header=Header { type_id:0, type_data:229377, size:7628, opaque:true,
                  marked:false, large:true, type_flags:8 },
  raw_header=0x380011dcc8e, is_forwarding=true
Object was forwarded to 0x380011dcc86

thread 'main' panicked at src/runtime.rs:7467:
  Struct not found by ID 229377 - this is a fatal error
    beag::runtime::Runtime::property_access
    beag::builtins::objects::property_access
thread caused non-unwinding panic. aborting.   → SIGABRT
```

### Decoding the log

- The panic is in `Runtime::property_access`, reached from the
  `property_access` builtin on an ordinary field read.
- `struct_type_id=229377` (`0x38001`) is **not** a real struct id — it is bits
  of a *pointer* misread as a header field.
- `is_forwarding=true` + `Object was forwarded to 0x380011dcc86`: the runtime's
  own diagnostic already determined the receiver's header is a **GC forwarding
  pointer**. The object was moved and a forwarding pointer was installed at the
  old address.
- Addresses: `struct_pointer` untags to a **young-generation** address; the
  forwarded-to address is in the **old generation**. So this is a young→old
  **promotion** forwarding (a minor-GC / promotion event), *not* the old-gen
  layout-migration path.

In short: `property_access` was handed a pointer to an object the GC had
already moved, dereferenced the corpse, read the forwarding word as a struct
id, and aborted.

Redefinition is the trigger only because compiling the new definition allocates
heavily on the REPL's `eval` thread (`standard-library/beagle.repl-session.bg`
spawns `thread(fn(){ session-eval-loop ... })`), which drives frequent GC while
the game thread holds struct references.

---

## 2. The invariant that is violated

Beagle's GC is **stop-the-world** (generational: a young `Space` evacuated into
an old `MarkAndSweep`, with mutator threads parked / counted-in-C-call during a
collection — see `src/gc/generational.rs`, `src/runtime.rs` `stack_pointers` /
`c_calling_stack_pointers`).

For a stop-the-world copying/promoting collector the contract is:

> When the collector moves an object it installs a forwarding pointer at the old
> location, then **updates every live reference** (roots + heap fields) to the
> new location **before mutators resume**. After the cycle completes, no live
> reference may point to a forwarded object.

Forwarding pointers are therefore **transient, GC-internal bookkeeping** that
exist only *during* a collection. A mutator (runtime code running between
collections, such as `property_access`) must **never** observe one.

The crash is a direct violation: a forwarding pointer reached mutator code.
**This is a GC / root-tracking bug**, not a runtime-logic or struct-layout bug.
(Confirmed: the only code that legitimately touches forwarding bits is in
`src/gc/*` while it scans; nothing on the mutator side resolves forwarding, and
should not need to.)

---

## 3. Proven vs. inferred

Keep these separate — only the first is established by evidence.

- **Proven (from the log):** a forwarding pointer reached `property_access`,
  i.e. a live reference to a moved object survived a collection. GC invariant
  violated.
- **Inferred (NOT reproduced):** the *specific* mechanism. The plausible story
  is that the receiver flows into the builtin **by value** and goes stale across
  a GC safepoint reached *inside* the builtin. This has not been reproduced or
  proven; treat it as the leading hypothesis, not fact.

Eight headless reproductions (single-thread, `--gc-always`, forced old-gen,
two-thread reader/redefiner, C-call/`sleep` interleaving) all failed to trigger
it — the receiver was always correctly maintained. The bug needs the game's
real concurrency shape (raylib FFI frames on the main thread + the eval thread
collecting), which the synthetic cases did not recreate.

---

## 4. Root-cause hypotheses (ranked) and how to confirm each

All three are "a root the GC should have updated wasn't updated." They differ in
*which* root and *when*, which determines the fix.

### H1 — A heap pointer held *by value* across an in-builtin GC safepoint
`property_access` (`src/builtins/objects.rs:179`) receives `struct_pointer` as a
plain `usize` argument and calls `save_gc_context!`
(`src/builtins/mod.rs:281`) then `get_runtime().get_mut()`. If acquiring the
runtime (contended by the `eval` thread) or any later step is a GC safepoint —
i.e. another thread's collection is allowed to proceed while this thread is
counted safe — then the collection promotes the receiver and updates the **Beagle
stack slot** it came from, but **not** the by-value copy already sitting in the
builtin's register/argument (that copy is not a scanned root). On return the
builtin uses the stale copy.

*Confirm:* determine whether a builtin call registers the thread as
parked/`c_calling` (`Runtime::register_c_call`, `src/runtime.rs:~917`) and
whether `get_mut()` can block while another thread collects. If yes, add a
targeted check: on entry vs. just-before-use, compare `struct_pointer`'s header
for the forwarding bit; if it flips mid-call, H1 is confirmed.

### H2 — Cross-thread stack scan misses a parked thread's register-resident root
When the `eval` thread collects, it must scan the game thread's stack from the
snapshot in `stack_pointers` / `c_calling_stack_pointers`. If the game thread is
parked in a C-call (raylib FFI is constant in the render loop) with the live
receiver in a **callee-saved register** that was not spilled into a scanned frame
slot, the GC cannot find or update it.

*Confirm:* reproduce with the game thread inside an FFI/C-call holding a struct
in a register across a collection driven by another thread; check whether that
register's slot is in the scanned set.

### H3 — Write-barrier / remembered-set gap during the redefinition recompile
Redefinition recompiles dependent functions and mutates long-lived (old-gen)
runtime tables. If a young pointer is stored into an old-gen structure without a
write barrier / card mark (`src/gc/generational.rs` remembered set + card
table), a later minor GC won't scan it, promotes the young object without
updating that reference, and leaves it stale.

*Confirm:* audit the redefinition/recompile path for old←young stores that
bypass the write barrier.

### A decisive, cheap diagnostic for all three
**IMPLEMENTED** (`GenerationalGC::verify_no_young_roots`, gated on
`BEAGLE_GC_VERIFY_ROOTS`): a verification pass at the **end** of every minor
collection walks all enumerated roots (every scanned thread's stack via the same
walker used for promotion, plus `extra_roots` = shadow-stack handles + binding
cells) and asserts none still points into young space (after promotion, a live
young root must have been rewritten to its old-gen copy). **Result on the §9
repro: it never fires** — see §9 for why that is itself decisive (the defect is
not an unupdated enumerated root). Original framing:

Walk all roots and assert none points to an object whose header has the
forwarding bit set.

- If it **fires**, the bug is GC root *enumeration/updating* (H2/H3 family): the
  collection itself left a live root unupdated.
- If it **never fires** yet the crash still occurs, the staleness happens
  *during* a builtin after the collection (H1): the by-value argument, not a
  scanned root.

This single assertion tells you which layer to fix and converts the
intermittent crash into a deterministic, early failure.

---

## 5. The correct fix (by hypothesis)

The fix lives in the GC/safepoint/root layer. Pick based on the diagnostic
above:

- **If H1:** a builtin must not hold a raw heap pointer across a GC safepoint.
  Either (a) keep the receiver in the handle scope (`GcHandle` / `extra_roots`,
  which the GC *does* update — see how shadow-stack handles are scanned in
  `minor_gc`) and re-read it after the safepoint, or (b) ensure the field-read
  path of `property_access` is genuinely safepoint-free (no contended lock that
  yields to a collecting thread, no allocation). The receiver value used after
  the safepoint must come from a GC-updated location, never a stale by-value
  copy.

- **If H2:** the safepoint/stack-map model must guarantee every live heap
  pointer is in a scanned slot at a safepoint (spill callee-saved roots to frame
  slots at C-call boundaries, or scan the saved register area). The fix is in
  how `gc_frame_tops` / the c-call snapshot enumerate a parked thread's roots.

- **If H3:** add the missing write barrier / card mark on the offending old←young
  store in the redefinition recompile path.

In every case forwarding stays **internal to the GC**. No mutator-side
forwarding resolution is added.

---

## 6. Rejected fix (and why) — do not reintroduce

A prototype made `property_access` and `write_field` call a
`resolve_forwarding(tagged)` helper that, on seeing a forwarding bit, followed
the forwarding pointer to the live object and continued. It was reverted.

Why it is wrong:

1. **Wrong layer.** Following forwarding pointers is a *concurrent*-collector
   read/load barrier (ZGC, Shenandoah). Beagle is stop-the-world and must never
   need one. It puts a GC concept into the runtime.
2. **Masks the bug.** It hides the real defect (an unupdated root). The same
   stale root still exists everywhere else — the next collection that scans that
   slot, or any other builtin/codegen path that dereferences it, can still
   corrupt or crash. It only papers over the two field-access entry points.
3. **Fragile.** Following works only while the old location's forwarding header
   is intact; once the young space is reused the header is overwritten and the
   "recovery" silently reads a wrong object.
4. **Hot-path-adjacent.** `property_access` is the inline-cache *miss* slow path,
   so the per-op cost is bounded — but it is still work on a path that, in a
   correct collector, can never encounter forwarding.

It did stop the observed crash and pass the suite, which is exactly why it is
dangerous: green tests over a hidden heap-safety bug.

---

## 7. Reproduction / validation plan

- **Reproduce** in the real environment first: beagle-zelda under the REPL, redo
  the room-2 / struct-redefine sequence while the render loop runs. The
  end-of-GC forwarding assertion (§4) should fire deterministically and point at
  the unupdated root. Headless synthetic repros did not trigger it (§3); the
  FFI-during-collection shape (H2) is the most promising synthetic angle.
- **Validate any fix** with:
  - full suite (`cargo run -- test resources/`) — must stay 100% (was 371/371),
  - the same suite under each GC feature (`generational`, `mark-and-sweep`,
    `compacting`) and `// gc-always` stress,
  - the end-of-GC forwarding assertion enabled, replaying the redefinition
    workload with no abort,
  - all `benchmarks/benchmarksgame/*` bit-identical.

---

## 8. Code map

- Crash site / panic: `src/runtime.rs` `Runtime::property_access` (the
  `is_forwarding` diagnostic + `panic!`), sibling write path
  `Runtime::write_field`.
- Builtin entry + GC-context save: `src/builtins/objects.rs:179`,
  `save_gc_context!` at `src/builtins/mod.rs:281`.
- Multi-thread root snapshots: `src/runtime.rs` `stack_pointers`,
  `c_calling_stack_pointers`, `register_c_call`.
- Generational GC + promotion/forwarding: `src/gc/generational.rs`
  (`copy` / `copy_object` install + follow forwarding, `minor_gc`, `full_gc`,
  remembered set + card table).
- Old-gen collection: `src/gc/mark_and_sweep.rs`.
- Forwarding bit helpers: `src/types.rs`
  (`set_forwarding_bit` / `clear_forwarding_bit` / `is_forwarding_bit_set`).
- REPL eval thread (the redefinition driver): `standard-library/beagle.repl-session.bg`.
