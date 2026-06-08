# Handoff: protocol-impl redefinition mis-dispatch under aggressive tier-up

**Status:** RESOLVED. Root-caused and fixed in `src/compiler.rs`.
**Branch:** `ssa-foundation`
**Test:** `resources/redefine_protocol_impl_test.bg`
**Repro knob:** `BEAGLE_SPECIALIZE_THRESHOLD=1` (env, read in `src/ast.rs` ~line 1540).
**Shipping config (default threshold 1000): fully clean — 372/372, this test passes.**

## 0. Resolution

The async/delimited-continuation hypothesis in §4–§5 below was a red herring
(the crash reproduced at the same rate when bypassing `beagle.async/__main__`).
**Those sections are stale — do not chase them.**

Actual root cause: protocol impl methods are compiled under generated internal
names like `Cat_speak` / `Dog_speak`, but their stored `source_text` is the
original source slice, e.g. `fn speak(self) { ... }`. Threshold-1 auto-tier-up of
`Cat_speak` reparsed that source; the parser produced a function named `speak`,
and the install keys on the *parsed* name — so the tier-2 install was staged for
plain `redefine_protocol_impl_test/speak`, the protocol dispatcher. If that stale
install landed after a later `eval`, the dispatcher jump-table slot was
overwritten with Cat's impl body, so the final `speak(dog)` ran Cat's
`self.lives` code on a Dog.

Fix (`rename_sole_top_level_function` in `src/compiler.rs`, called from
`specialize_function`): a tier-2 recompile must reinstall under the name it is
specializing (`full_name`), regardless of what the source's `fn` header parses
to. After reparsing the source, force the sole top-level `fn` to the unqualified
`full_name` (`Cat_speak`), so the install lands on the correct jump-table slot
and never touches the dispatcher. Protocol dispatchers themselves remain skipped
by `Runtime::is_protocol_dispatcher`.

This is preferred over the earlier source-suppression approach (suppress the
stored source so `specialize_function` bails): that would have **disabled tier-2
for every protocol impl method body**. The rename keeps impl bodies optimizable —
verified `Cat_speak`/`Dog_speak` tier-2 install correctly under threshold-1
churn.

Validation after the fix:

```bash
cargo build --release --bin beag
for i in $(seq 1 300); do
  BEAGLE_SPECIALIZE_THRESHOLD=1 ./target/release/beag resources/redefine_protocol_impl_test.bg 2>&1 | tail -1
done | sort | uniq -c
# 300 Rex knows 5 tricks!

for i in $(seq 1 150); do
  BEAGLE_SPECIALIZE_THRESHOLD=1 ./target/release/beag run --no-gc resources/redefine_protocol_impl_test.bg 2>&1 | tail -1
done | sort | uniq -c
# 150 Rex knows 5 tricks!

BEAGLE_SPECIALIZE_THRESHOLD=1 ./target/release/beag test resources/redefine_protocol_impl_test.bg
# 1 passed, 0 failed, 0 skipped
# also: redefine_protocol_add_test, struct_redefine_protocol_test pass @ thr1; full suite 372/372
```

The remaining sections are preserved as historical investigation notes — note
§4–§5's async-continuation theory is WRONG (see above).

---

## 1. TL;DR

Redefining a protocol impl at runtime (`extend Dog with Speak { ... }` via
`eval`) under threshold-1 tier-up churn used to return **stale/wrong results**
(25–80% of runs). Five fixes eliminated the staleness (wrong results) entirely.
What remains is a **rare (~2%) crash** at threshold 1 only: `speak(dog)`
occasionally dispatches to **Cat's** `speak` impl, which does
`self.lives` → `FieldError: Field 'lives' does not exist on Dog`.

The crash is **NOT** a dispatch-cache race, **NOT** GC, **NOT** thread
concurrency. Ground truth shows the failing call operates on a **Dog object at a
different address than the `dog` used elsewhere** — i.e. an **object-identity
problem**, downstream of which the dispatch mis-resolves. The prime suspect is
the **async/delimited-continuation runtime** (`beagle.async/__main__` →
`beagle.core/__reset__`) that wraps `main` in the test harness.

---

## 2. The test and expected output

`resources/redefine_protocol_impl_test.bg`: defines `Cat{name,lives}`,
`Dog{name,tricks}`, protocol `Speak{fn speak(self)}`, impls for both, then in
`main`:

```
println(speak(cat))                 // Meow from Whiskers
println(speak(dog))                 // Woof from Rex
eval("extend Cat with Speak {... Purr ... self.lives ...}")   // redefine Cat
println(speak(cat))                 // Purr from Whiskers (9 lives)
println(speak(dog))                 // Woof from Rex
eval("extend Dog with Speak {... self.tricks ... knows ...}") // redefine Dog
println(speak(cat))                 // Purr from Whiskers (9 lives)
println(speak(dog))                 // Rex knows 5 tricks!   <-- crashes here ~2%
```

The crash is on the final `speak(dog)`: it runs Cat's redefined `speak`
(`"Purr ... " ++ to-string(self.lives) ...`) on the Dog → `self.lives` fails.

### Reproduce

```bash
cargo build --release --bin beag
# plain run, ~2-9% crash:
for i in $(seq 1 150); do
  BEAGLE_SPECIALIZE_THRESHOLD=1 ./target/release/beag resources/redefine_protocol_impl_test.bg 2>&1 | tail -1
done | sort | uniq -c
# or via the harness:
BEAGLE_SPECIALIZE_THRESHOLD=1 ./target/release/beag test resources/redefine_protocol_impl_test.bg
```
The crash rate is load/timing dependent (~2–9%). **Any tracing (`eprintln`,
atomics in the dispatch path) MASKS it** — it's timing-tight. Use thread-local
plain-`Cell` recording dumped only at the crash if you must instrument.

---

## 3. What was fixed (committed on `ssa-foundation`)

These are real bugs; keep them. They took the test from 25–80% wrong → ~2% crash
and are validated default + generational + tier-2-harness = 370/370, perf intact
(mandelbrot ~1.04s, fannkuch ~2.56s).

| commit | fix |
|--------|-----|
| `7c4cace` | **Race-free tier-2 install at stop-the-world.** Compiler thread stages a `PendingInstall`; a coordinator applies it at an STW (`Runtime::stop_world_and_apply_installs`, `apply_pending_installs`). Non-registered coordinator (async tier-up spawn thread) vs registered (the calling mutator after `specialize-all`); GC rendezvous also drains. Built on `runtime/stop-the-world` (the `f821bca` observable primitive) + a thread-local `REGISTERED_MUTATOR` flag for correct wait-count accounting. Generation counter for de-spec soundness. |
| `0540a7d` | **Protocol-dispatch inline-cache invalidation** on redefine (`invalidate_all_protocol_dispatch_caches`, fired from `register_extension` AFTER the dispatch-table update, via the `InvalidateProtocolDispatchCaches` compiler message — only on a redefinition, via `add_protocol_info`'s new `bool` return). **Don't tier-2-specialize protocol dispatcher functions** (`Runtime::is_protocol_dispatcher`, gated in `Compiler::specialize_function`). |
| `7b3dc87` | **`jump_table_lock`** — `add_jump_table_entry`/`modify_jump_table_entry` mutate the `jump_table_pages` Vec + do the W^X mprotect with no lock; compiler-thread recompile raced spawn-thread installs, dropping a write → stale jump-table entry. **This was the biggest staleness fix.** Lock order: `gc_lock → install_apply_lock → jump_table_lock`. |
| `219425d` | Cache write **value-before-key + release fence** (`protocol_dispatch` slow path) for torn-read safety. Slow-path only; fast path untouched. |
| `5d9d46a` | **`dispatch_tables_lock`** — `add_protocol_info` (`&mut`) on the calling thread races `get_dispatch_table_ptr` (`&`) on the compiler thread baking call-site pointers; concurrent `&mut`+`&` on a `HashMap` is UB. Lock scoped inside the methods (released before `register_extension`'s blocking invalidate message → no deadlock). |

---

## 4. The residual crash — evidence chain (what's RULED OUT, with proof)

All from instrumenting the failing run (diagnostics removed after; see §6 to
re-add). Each conclusion is backed by captured data, not theory.

1. **It's a mis-dispatch.** `speak(dog)` runs Cat's impl. Stack: `redefine_protocol_impl_test/speak (repl:1)` (the recompiled dispatcher/impl) → `main:22` → `beagle.async/__main__` → `beagle.core/__reset__`.

2. **The object IS a Dog.** At the crash, the heap object's header is `0xc0000200`; `type_data`/`get_struct_id` = **192** (Dog). Not a corrupted/cat object by type.

3. **NOT GC.** `--no-gc` still crashes at the same rate. Not relocation, not forwarding-pointer-read.

4. **NOT thread concurrency.** A global (thread, dispatch_table) recorder shows **only `ThreadId(1)`** dispatches structs, even in a crashing run. So no concurrent cache writer → the "torn cache" theory is dead.

5. **The dispatcher uses the inline cache, never the if-chain fallback.** Logged `compile_protocol_method` decisions: `speak` is always `OPTIMIZED` (cache_off ~1400, far below the 4096 page limit where it would fall back to `build_method_if_chain`). So the if-chain is not involved.

6. **The fast-path `read_struct_id` codegen is correct.** Disassembled the dispatcher (`--dump asm --dump-filter redefine_protocol_impl_test/speak`): untag arg → load header → `and #0xffffffff000000` → `<<3` → `>>24` → `>>3` untag → compares to `cache[0]`, `b.ne` to slow. Computes 192 correctly for a 192 object and would branch to slow on mismatch.

7. **THE DECISIVE CLUE — two different Dog objects.** Instrumented `protocol_dispatch` to log `first_arg` pointer for each Cat/Dog (191/192) slow dispatch, and dumped the crashing object's pointer:
   - Cat object: `0x1800031c00` (struct 191)
   - Dog object (earlier `speak(dog)` calls): `0x1800031cc0` (struct 192)
   - **Crashing object (final `speak(dog)`): `0x300006398`** (struct 192) — a *different* Dog at a *different* address.
   - These addresses are **deterministic across runs** and **identical with `--no-gc`**.
   So `main`'s single `let dog = Dog{...}` is being dispatched as **two different
   Dog instances**. The final `speak(dog)` operates on `0x300006398`, which was
   never seen by the slow path (so its dispatch was a fast-path hit against a
   cache slot populated for the *other* dog / a stale Cat entry).

### Why every "dispatch cache" / "GC" explanation failed

The cache key is `type_id` (192), not address, so two Dog instances *should*
dispatch identically — unless the second instance's type read or the call's
binding is wrong. The contradiction ("single thread + correct 192 object should
never mis-dispatch") is resolved by #7: the problem is **upstream** of dispatch —
*which object* `speak(dog)` binds to — not the dispatch mechanism itself.

---

## 5. Current best hypothesis

The test's `main` runs inside the async runtime's delimited-continuation reset
(`beagle.async/__main__` → `beagle.core/__reset__`; see the stack). Under
threshold-1 churn, something in the **continuation capture/resume** (or the
`eval` path interacting with it) causes a **local binding (`dog`) to resolve to a
duplicate/relocated object** (`0x300006398`) distinct from the original
(`0x1800031cc0`). The mis-dispatch is a downstream symptom: the second object's
dispatch fast-hits a cache slot/entry that doesn't correspond to it.

This is an **object-identity / variable-binding bug in the async-continuation
subsystem**, NOT in protocol dispatch or GC. It needs a fresh investigation from
that angle.

### Concrete next steps

1. **Find where the second Dog comes from.** Instrument struct allocation
   (`allocate` for struct objects, or `Dog` construction) to log the pointer +
   a backtrace/frame, and confirm whether `let dog = Dog{...}` runs **twice**
   (continuation re-invocation / `main` re-entry) or whether `dog` is *copied*
   (continuation stack capture copying heap-referenced locals).
2. **Inspect `__reset__`/`__main__`** (`standard-library/std.bg` ~307,
   `standard-library/beagle.async.bg` ~2469) and `src/builtins/reset_shift.rs` /
   `src/builtins/continuations.rs`: does capturing/resuming the continuation that
   wraps `main` duplicate or relocate frame-local object references? Does `eval`
   (`src/builtins/threads.rs::eval`, runs the top-level via the `apply_call_0`
   shim) perform an effect that triggers a continuation hop?
3. **Check GC-root / continuation-frame handling of `dog`.** `0x18...` vs
   `0x30...` are different heap regions; even with `--no-gc` the second object
   exists, so it's allocated (not just moved). Why is a second Dog allocated?
4. **Bisect the threshold dependence.** It only appears at very low thresholds
   (many recompiles). Does the second-object appearance correlate with a tier-up
   / install-STW landing during the `eval`'s continuation window?

### Diagnostic that nailed #7 (re-add to continue)

Thread-local plain-`Cell`/`RefCell` (no atomics/eprintln on the hot path) ring in
`src/builtins/dispatch.rs::protocol_dispatch` recording `(type_id, fn_ptr,
cache_location, first_arg)` for `type_id == 191 || 192`; dump it (plus the
crashing object's `untagged()` pointer) at the `FieldError` throw site in
`src/runtime.rs` (the current-layout branch, ~line 7479, `"Field '{}' does not
exist on {}"`). Gate everything on an env var (e.g. `BEAGLE_DISPATCH_CLASSIFY`)
and make `pub mod dispatch` in `src/builtins/mod.rs` so the throw site can read
the thread-local. Catch a crash by looping the repro ~200×.

---

## 6. Key code locations

- Protocol dispatch fast/slow path codegen: `src/ast.rs` `Ast::ProtocolDispatch` (~4268–4407).
- Slow path + cache write: `src/builtins/dispatch.rs::protocol_dispatch`.
- `read_struct_id` IR: `src/types.rs` (~975).
- Dispatcher build (cache vs if-chain): `src/compiler.rs::compile_protocol_method`, `build_optimized_dispatch`, `build_method_if_chain` (~2321–2490).
- Impl registration + cache invalidation trigger: `src/builtins/keywords.rs::register_extension`; `Runtime::add_protocol_info` (returns `was_redefinition`), `get_dispatch_table_ptr`, `is_protocol_dispatcher`, `invalidate_protocol_dispatch_caches` in `src/runtime.rs`.
- Cache invalidation impl: `Compiler::invalidate_all_protocol_dispatch_caches`, `add_protocol_dispatch_cache` (page-full → if-chain fallback) in `src/compiler.rs`.
- Tier-2 install at STW: `Runtime::{stop_world_and_apply_installs, apply_pending_installs, stop_the_world_observe}`, `PendingInstall`, `REGISTERED_MUTATOR`, `install_generation`, `install_apply_lock`, `jump_table_lock`, `dispatch_tables_lock` in `src/runtime.rs`; `Compiler::stage_specialization_installs` + `SpecializeFunction` handler in `src/compiler.rs`.
- Eval entry: `src/builtins/threads.rs::eval` (compiles via compiler thread, runs top-level on the calling thread via `apply_call_0` shim).
- Async/continuation wrappers (suspect): `standard-library/beagle.async.bg` (~2469 `__main__`), `standard-library/std.bg` (~307 `__reset__`), `src/builtins/reset_shift.rs`, `src/builtins/continuations.rs`.

## 7. Dead ends (don't repeat)

- "Torn fast-path cache read needing read-side acquire" — ruled out by single-thread evidence (#4) and the correct-codegen disasm (#6). (Also: user vetoed fast-path changes.)
- Mutator-driven STW *around the eval* to make redefinition atomic — does NOT work: the redefine's dispatch-table update runs by *executing compiled Beagle code* (the `extend` top-level on the calling thread), which hits the entry safepoint and would self-park. Regressed the test to 0% when tried.
- The if-chain fallback — never triggers for this test (cache not full, #5).
- GC relocation / forwarding pointers — `--no-gc` still crashes (#3).
