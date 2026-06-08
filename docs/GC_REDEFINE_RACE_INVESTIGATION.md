# Stale-root / forwarding crash on live struct redefinition — full investigation

## Resolution (2026-06-06)

**Fixed.** The inferred conclusion below was wrong in one important respect:
the forwarded receiver was not a young object missed by minor-GC root
processing. It was an **old-generation object moved by post-sweep struct-layout
migration** in `MarkAndSweep::migrate_outdated_structs`. That pass rewrote heap
fields and extra roots, but omitted active Henderson-frame slots (and captured
continuation-frame slots). The minor-GC verifier was therefore silent because
it checks young-space evacuation, while the stale pointer was created later by
old-space migration.

The fix passes `gc_frame_tops` into old-generation migration and rewrites active
stack roots, captured continuation roots, heap fields, and extra roots before
freeing the old copies. The canonical repro is now enabled and passes.

Stress testing also confirmed a **separate real race** from the original
inference: the compiler thread replaced `StructManager` `Vec`/`HashMap` contents
while mutators read borrowed `Struct` references in `property_access`. Struct
metadata is now behind an `RwLock`; lookups return owned snapshots, property
resolution takes one coherent snapshot, stale GC completion cannot clear plans
from a newer redefinition, and property-cache Rust writers/readers use atomic
key publication.

Validation performed: 20/20 canonical test runs and 20/20
`--no-auto-specialize` runs with `BEAGLE_GC_VERIFY_ROOTS=1`, plus focused unit
tests. The historical investigation remains below because its observations led
to the reproducer, but §§0, 1, 6–8, 10, and 13 are superseded by this resolution.

**Author's note:** this is a complete write-up of my current understanding after
deterministically reproducing the bug headless and bisecting it. It is meant to
be read top-to-bottom by the next person. It supersedes the hypothesis ranking
in `docs/GC_STALE_ROOT_REDEFINE_BUG.md §4` on the points called out below; that
file remains the "official" bug record and its §9 is a condensed version of this.

Throughout the historical text I separate **PROVEN** (established by evidence
gathered at the time) from **INFERRED**. See the resolution above for the final
diagnosis.

---

## 0. One-paragraph summary

Redefining a struct that is currently in use intermittently aborts a reader
thread in `property_access` with `is_forwarding=true`: the reader dereferences a
struct pointer that the GC has already moved (young→old promotion), reading the
forwarding word as a bogus header. I reproduced this **deterministically and
headlessly**, bisected the essential conditions, and implemented the
end-of-collection root verifier the original doc recommended. The verifier
**never fires** — even under forced stop-the-world — which, combined with a
machine-code analysis showing the reader's slot is written once, correctly
updated by the only relevant GC, and never overwritten, rules out the entire
"GC missed a root" family of explanations. The remaining consistent explanation
is a **concurrent torn write from the redefiner thread** corrupting the slot or
object after the GC validated it. The fix therefore belongs in the redefinition
write path's synchronization, not in the GC root layer.

---

## 1. The observed crash

```
Struct not found! struct_pointer=0x38000032e06, struct_type_id=229376,
  header=Header { type_id:0, type_data:229376, size:819, opaque:true,
                  marked:false, large:true, type_flags:4 },
  raw_header=0x3800003334e, is_forwarding=true
Object was forwarded to 0x38000033346
thread 'main' panicked ... Struct not found by ID 229376 - this is a fatal error
  beag::runtime::Runtime::property_access
  beag::builtins::objects::property_access
thread caused non-unwinding panic. aborting.   → SIGABRT
```

Decoding (PROVEN from the log + code at `src/runtime.rs:7449`):

- The receiver `struct_pointer` untags to a **young-generation** address.
- Its first word has the **forwarding bit set** (`is_forwarding=true`); the
  forwarded-to address `0x38000033346` is in the **old generation**.
- `struct_type_id=229376` is not a real id — it is bits of the forwarding
  pointer misread as a header field, so `get_struct_by_id` returns `None` and
  the function panics.

This is a **young→old promotion** (minor GC) in which a live reference to the
moved object was not pointed at the new copy. The receiver reached
`property_access` as a corpse.

---

## 2. Architecture you need in your head

### 2.1 GC: precise, stop-the-world, generational

- Young `Space` evacuated into an old `MarkAndSweep`
  (`src/gc/generational.rs`). Minor GC promotes survivors young→old, installs a
  forwarding pointer at the old young address, updates roots, then clears young.
- **Precise** roots via **Henderson frames** (`src/gc/stack_walker.rs`): every
  compiled function links a frame into a per-thread linked list
  (`gc_frame_link`/`gc_frame_unlink` in prologue/epilogue). The GC walks the
  `prev` chain from each thread's `gc_frame_top` and scans only the **declared
  GC slots** of each frame. Arbitrary registers are **not** scanned — the
  codegen is responsible for keeping live heap pointers in slots across
  safepoints.

Frame layout (from `stack_walker.rs`):
```
[FP+8]  saved return address
[FP+0]  saved caller FP
[FP-8]  frame header (TYPE_ID_FRAME, size = num_slots)   ← gc_frame_top points here
[FP-16] prev pointer (raw address of previous frame's header, or 0)
[FP-24] local[0]
[FP-32] local[1] ...
```

### 2.2 The complete root set

`Runtime::run_gc` (`src/runtime.rs:3561`) hands the collector exactly:
1. every **scanned thread's** GC-frame-chain slots (`gc_frame_tops`),
2. per-thread **shadow-stack handles** + `head_block`,
3. namespace **binding cells**.

There is nothing else. If a forwarding pointer reaches a mutator, either one of
these was left stale, or the value lives outside this set.

### 2.3 Threading and safepoints

- Mutator threads cooperate. A collection can only proceed once every *other*
  registered mutator is parked — either at a `__pause` safepoint
  (`stack_pointers`) or inside an FFI/`register_c_call` window
  (`c_calling_stack_pointers`). See the rendezvous loop in `gc_impl`
  (`src/runtime.rs:5579`).
- **`__pause` is the only cooperative safepoint, and it is emitted only at
  function entry** (`src/ast.rs:1611`), guarded by an atomic "should pause"
  check. There is **no loop back-edge `__pause`**. So within a single function
  call, after the entry pause, a thread reaches no further cooperative safepoint
  unless it allocates or makes a call that itself can GC.
- `property_access` (the slow-path builtin) does **not** `register_c_call` and
  does **not** allocate, so another thread cannot collect "inside" it, and it is
  not itself a safepoint. (PROVEN by reading `src/builtins/objects.rs:179` and
  `src/builtins/threads.rs:80`.)
- `get_runtime().get_mut()` is `SyncUnsafeCell::get_mut()` — no lock, no
  safepoint (PROVEN, `src/main.rs:2664`). So the original doc's H1 ("receiver
  goes stale across a safepoint *inside* `property_access` because acquiring the
  runtime is a safepoint") is **impossible as stated**.

### 2.4 Struct redefinition

`Compiler::add_struct` (`src/compiler.rs:1864`) on a redefinition:
- `runtime.add_struct` → `Structs::insert` bumps the struct's **layout version**
  and retains the old definition,
- `invalidate_all_property_caches` writes `usize::MAX` into every inline
  property-cache `struct_id` slot,
- `revert_all_specializations` (+ `invalidate_all_protocol_dispatch_caches`)
  reverts any tier-2 code and resets dispatch caches, taking
  `install_apply_lock` and bumping `install_generation`.

The recompile/`eval` itself allocates heavily, which is what drives frequent GC.

### 2.5 Property access: fast path vs slow path

A field read `p.x` compiles to an **inline cache check** followed by two arms
(PROVEN from disassembly, §6):
- **fast path:** untag `p`, load its header, mask to `struct_id+layout_version`,
  compare against the cached key; on a hit, load the field inline.
- **slow path:** call the `property_access` **builtin**, which consults the
  runtime, handles old-layout objects, and re-populates the cache.

Crucial consequence: after a redefinition bumps the layout version, an existing
instance is **old-layout**, its header no longer matches the cache key, so every
read of it takes the **slow path** (builtin call). This is the single behavioural
change that turns the bug on (see §5).

---

## 3. The reproduction (deterministic, headless)

`resources/gc_redefine_stale_root_gcalways.bg` (kept as `// Skip`). Shape:

- a **reader**: main thread holds a long-lived struct instance `anchor` and, in
  a loop, calls helpers that read its fields (the receiver is live across the
  helper's entry `__pause`);
- a **redefiner**: a spawned thread repeatedly `eval`s a redefinition of that
  same struct;
- `// gc-always`: every allocation triggers a full GC, maximizing the collision
  window.

```
BEAGLE_GC_VERIFY_ROOTS=1 cargo run --features generational -- \
    test resources/gc_redefine_stale_root_gcalways.bg
```

Aborts in `property_access` with `is_forwarding=true` within seconds, every run.
The original doc's claim that this needs the real beagle-zelda environment (after
8 failed synthetic repros) is **wrong**.

---

## 4. Bisection — what's essential (each toggled independently, PROVEN)

| Condition changed | Result |
|---|---|
| baseline: 2 threads, child redefines the **live** struct, gc-always | **crashes (forwarding)** |
| child does pure allocation, no `eval`/redefine | passes |
| child `eval`s `"1 + 1"` (compile machinery, no redefine) | passes |
| child redefines an **unrelated** struct | passes |
| tier-up disabled (`--no-auto-specialize`) | still corrupts (slower; sometimes garbage receiver instead of clean forwarding) |
| force full stop-the-world on every GC (`memory.threads.len()==1` bypassed) | **still crashes** |
| reader loop body has **no allocation** (only the entry `__pause` is reachable) | **still crashes** |

Readings:

- **Redefining the *live* struct is the essential trigger.** Not the compile
  machinery (eval of a non-redefinition is fine), not multithreaded GC on its
  own (pure-alloc child is fine), not redefining some other struct.
- **Tier-up/specialization is not required** — it only modulates timing. With it
  off the same corruption appears, just much slower to hit and sometimes as a
  garbage receiver ("field does not exist on `<garbage>`") rather than a clean
  `is_forwarding`.
- **The single-thread GC fast path is not the cause.** Forcing stop-the-world
  for every collection does not fix it.
- **No loop-body allocation is needed.** The only safepoint the reader passes is
  its entry `__pause`, and it still crashes — so the staleness is associated with
  the entry-pause window, not a mid-loop allocation.

Each user function is compiled **exactly once** during the run (PROVEN by logging
every compile): the reader functions are **not** recompiled by the redefinition.
So "redefinition produces buggy recompiled reader code" is ruled out.

---

## 5. Why "redefine the live struct" specifically

The chain (PROVEN mechanism, INFERRED as the relevance):

1. Redefining the struct bumps its layout version
   (`Structs::insert` / `invalidate_all_property_caches`).
2. The held `anchor` becomes **old-layout**; its header no longer matches the
   inline cache key.
3. Therefore every `anchor.field` read switches from the inline fast path to the
   **`property_access` builtin slow path** (a real call).
4. The crash only manifests once reads are on that slow path.

Redefining an *unrelated* struct also invalidates caches globally but does **not**
make `anchor` old-layout, so `anchor`'s reads quickly re-cache and return to the
fast path — and it does not crash. That contrast is what isolates "old-layout →
slow-path reads of the held instance" as the necessary condition.

---

## 6. The machine-code analysis and the contradiction

I disassembled the reader (`--dump asm`). For the held parameter `p` (the struct
whose field is read):

```
 164  stur  x0, [x29, #-0x18]     ; store arg p into its GC slot — the ONLY write to this slot
 236  blr   <__pause>             ; entry safepoint (guarded). Only GC point reachable in this fn.
 304  blr   <tier-up counter>     ; normally skipped; not a GC safepoint
 820  blr   <comparison builtin>  ; loop condition; no allocation, not a GC safepoint
 856  ldur  x26, [x29, #-0x18]    ; load p fresh from slot (first use)
 892  ldur  x20, [x23]            ; x23 = untag(p); dereference p's header for the cache check
                                  ;   <-- the forwarded value is observed HERE
2008  ldur  x23, [x29, #-0x18]    ; load p fresh from slot (second use)
```

Grepping the whole function, `[x29,#-0x18]` is **written exactly once** (offset
164, the prologue) and only ever **read** afterward (856, 2008). The reader does
**not** cache `p` in a register across the loop; it reloads from the slot each
use.

Now trace the single-thread timeline:

1. `p` is stored into its slot (164).
2. The entry `__pause` (236) runs a GC. `p` is in its slot, so the collector
   promotes the object and **updates the slot** — and the verifier (§7) confirms
   the slot is correct after every collection it sees.
3. Between 236 and the loads (856/2008) there is **no GC safepoint** — the two
   intervening calls (304, 820) cannot collect, and there is no loop back-edge
   pause.
4. Nothing ever overwrites the slot.
5. Yet the load at 856 → deref at 892 reads a **forwarded young corpse**.

**This is a contradiction for any "the GC mishandled a root" theory.** A slot
that is (a) written once, (b) correctly updated by the only relevant collection,
(c) never overwritten, cannot produce a forwarded pointer when read — *unless the
memory backing that slot, or the object it names, is mutated by some agent
outside this thread's control between the validating GC and the read.*

---

## 7. The verifier, and why its silence is decisive

I implemented the doc's recommended diagnostic:
`GenerationalGC::verify_no_young_roots` (`src/gc/generational.rs`), gated on
`BEAGLE_GC_VERIFY_ROOTS`. At the end of each minor GC, **before** `young.clear()`,
it re-walks every enumerated root (all scanned `gc_frame_tops` + `extra_roots` =
shadow-stack handles + binding cells) and asserts none still points into young
space. After promotion, a surviving young pointer in a root is, by definition, a
root the collector failed to update.

Observed (PROVEN):

- It **never fires** on the repro — 12/12 runs, and also under
  `BEAGLE_GC_FORCE_STW=1` (which forces the full stop-the-world path so *every*
  thread's stack is scanned, not just the collector's).
- It costs nothing when the env var is unset; the full suite is green with it on
  (371 passed, 1 skipped).

Interpretation: at the end of every collection, **every enumerated root is
correct**. The stale value is *not* in a stack GC slot, *not* in a shadow-stack
handle, *not* in a binding cell. The verifier is a **true negative** — which is
itself a strong, useful signal: do not chase root-enumeration bugs.

---

## 8. Conclusion (INFERRED, clearly marked)

Putting §6 and §7 together, the only explanation consistent with all the
evidence:

> The reader's slot (or the object it points to) is **corrupted by a concurrent
> write from the redefiner thread** — a torn write / data race — *after* the GC
> has validated it and *before* the reader loads it. It is **not** a root the GC
> failed to enumerate or update; the GC's bookkeeping is correct.

This explains every observation at once:
- verifier silent (the defect is outside the GC's enumerated roots and outside
  the GC's logic),
- forced stop-the-world doesn't help (the race is not in the GC rendezvous),
- redefinition essential (it is the *writer*),
- redefining the *live* struct essential (it forces the reader onto the slow
  path, where the timing windows align and the read actually dereferences the
  corrupted pointer),
- tier-up irrelevant (just changes timing),
- single-thread fast path irrelevant.

It is also consistent with the surrounding history: a string of recent
redefinition concurrency fixes (unlocked `dispatch_tables` HashMap, unlocked
`jump_table_pages` Vec, torn cache writes — see recent commits) and the
"~2% residual torn-read crash" already noted in
`docs/PROTOCOL_REDEFINE_HANDOFF.md`. This repro looks like a fatter,
deterministic instance of that same residual race.

**What I have NOT done:** identified the exact unsynchronized write. The
conclusion is an inference from the contradiction, not a caught racing writer.

---

## 9. A real but separate loose end

While instrumenting `gc_impl` I observed that `memory.threads.len()` and
`registered_thread_count` **disagree** during collections, and the single-thread
fast path keys off `memory.threads.len() == 1` (`src/runtime.rs:~5514`) without
holding `gc_lock`. That looked like a smoking gun (a GC skipping another live
thread entirely). But forcing the stop-the-world path did **not** fix the crash,
so it is **not** the cause of *this* bug. It is a genuine latent inconsistency
worth tightening on its own (prefer the authoritative `registered_thread_count`,
taken under the lock), just not here.

---

## 10. How to catch the racing writer (next steps)

1. **ThreadSanitizer build** of the repro: `RUSTFLAGS=-Zsanitizer=thread` on
   nightly (the JIT/`mprotect`/W^X dance may need care, but the Rust-side data
   structures are what we want TSan to watch). A reported race on the slot
   address or on a struct-definition/cache field is the answer.
2. **lldb hardware watchpoint** on the held slot's address (`fp-0x18` for the
   reader frame) under the repro: break when anything writes it, and look at
   which thread / which redefiner code path is the writer.
3. **Audit the redefinition write path** for writes that race a concurrent GC or
   a concurrent reader's slow-path `property_access` *without* `gc_lock` / a
   stop-the-world:
   - `Compiler::add_struct` → `Structs::insert` (layout-version bump + old-def
     retention),
   - `invalidate_all_property_caches` / `invalidate_all_protocol_dispatch_caches`
     (raw `*mut usize` writes into cache buffers),
   - `revert_all_specializations` (jump-table pointer swaps under
     `install_apply_lock`),
   - the `property_access` slow path's **cache write-back**
     (`src/builtins/objects.rs:259+`) — it writes into the inline cache buffer
     from the reader thread; consider whether that write can race the
     redefiner's invalidation or a GC.
4. Keep `BEAGLE_GC_VERIFY_ROOTS=1` on while bisecting — it stays a true negative
   here, confirming you are *not* looking at a root-enumeration regression.

---

## 11. What NOT to do

- **Do not** reintroduce the reverted mutator-side `resolve_forwarding`
  band-aid (`property_access`/`write_field` following forwarding pointers). It is
  the wrong layer (a concurrent-collector load barrier in a stop-the-world VM),
  masks the real race, is fragile once young space is reused, and produces green
  tests over a live heap-safety bug. See `docs/GC_STALE_ROOT_REDEFINE_BUG.md §6`.
- **Do not** add roots or "fix" the GC enumeration — the verifier proves the
  enumeration is complete and correct for this bug.
- **Do not** rely on the single-thread fast-path change as a fix — it doesn't
  fix this and would only mask timing.

---

## 12. Code map

- Crash site: `Runtime::property_access` (`src/runtime.rs:7426`), builtin entry
  `src/builtins/objects.rs:179`, cache write-back `src/builtins/objects.rs:259+`.
- GC: `src/gc/generational.rs` (minor/full GC, promotion, card table,
  `verify_no_young_roots`), `src/gc/stack_walker.rs` (frame walk),
  `src/gc/mark_and_sweep.rs` (old gen).
- Root collection / rendezvous: `Runtime::run_gc` (`src/runtime.rs:3561`),
  `Runtime::gc_impl` (`src/runtime.rs:5427`), `ThreadState`
  (`src/runtime.rs:874`), `register_c_call`/`__pause`
  (`src/builtins/threads.rs`).
- Entry safepoint emission: `src/ast.rs:1611`.
- Redefinition: `Compiler::add_struct` (`src/compiler.rs:1864`),
  `revert_all_specializations` (`src/compiler.rs:1882`),
  `invalidate_all_property_caches` (`src/compiler.rs:1924`),
  `Runtime::add_struct` (`src/runtime.rs:8991`).
- Repro: `resources/gc_redefine_stale_root_gcalways.bg`.
- Diagnostic flag: `BEAGLE_GC_VERIFY_ROOTS`.
- Prior records: `docs/GC_STALE_ROOT_REDEFINE_BUG.md`,
  `docs/PROTOCOL_REDEFINE_HANDOFF.md`.

---

## 13. Confidence summary

| Claim | Status |
|---|---|
| Receiver reaches `property_access` as a forwarded young corpse | PROVEN (log) |
| Reproduces deterministically headless | PROVEN |
| Requires redefining the *live* struct + 2 threads + frequent GC | PROVEN (bisection) |
| Mechanism = old-layout instance forced onto slow-path reads | PROVEN (mechanism) / INFERRED (that this is *why* it's essential) |
| Tier-up, compile machinery, single-thread fast path NOT required | PROVEN |
| Reader's slot written once, correctly GC-updated, never overwritten | PROVEN (disassembly + verifier) |
| Stale value is not in any enumerated GC root | PROVEN (verifier silent, incl. forced STW) |
| Therefore: concurrent torn write from the redefiner | **INFERRED** (best/only explanation; racing writer not yet caught) |
| `memory.threads.len()` vs `registered_thread_count` inconsistency | PROVEN observation; NOT the cause of this bug (PROVEN by forced-STW) |
