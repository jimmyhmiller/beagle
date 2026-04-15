# Effect Handlers — Implementation Handoff

**Purpose:** Complete the effect-handler implementation on top of the prompt-tag infrastructure already in place. This doc is self-contained — read it, read the code it points to, and you should be able to pick up without prior context.

**Last updated:** 2026-04-14 (SIGSEGV resolved; E8 plumbing + Handle rewired)

## Current state snapshot (2026-04-14 PM)

- Commit `0c01350` — SIGSEGV fix (TLS hazard in `continuation_trampoline`).
- Commit `20600c8` — fresh-tag + find-handler-tag builtins, push-handler bumped to 3 args, `Ast::Handle` threads a fresh tag into a handler-scoped local.
- Commit `b8a7bfd` — `Ast::Handle` now compiles through `beagle.core/__reset__` instead of the stubbed push-prompt. Handle-without-perform works (previously threw "Refactor A" error).
- Baseline: 265/323, 0 regressions.

**What's still needed (in rough order):**

1. **Implement `perform_effect` itself (Step E6).** `Ast::Perform` still compiles to the stubbed `beagle.builtin/perform-effect` IR op. The trampoline/capture/return infrastructure is all in place — the missing piece is the handler-dispatch call.

   **Landmines discovered in-session — read before attempting:**
   - `handle` is a Beagle keyword. A Beagle-level helper that writes `handler.handle(op, resume)` does *not* parse as a protocol call — it parses as a field access ("Field 'handle' does not exist on Handler"). Writing `handle(handler, op, resume)` tries to start a `handle` block.
   - Qualified calls like `beagle.effect/handle(h, op, resume)` don't parse either (the parser doesn't treat the dotted prefix as a namespace-qualified call in this context).
   - **Conclusion:** a stdlib `dispatch-handler` helper is not viable. Two workable paths:
     - *(a)* Rust-side dispatch inside `perform_effect_runtime_with_saved_regs`: look up the protocol dispatch table for `beagle.effect/Handler<EnumType>` via `runtime.get_dispatch_table_ptr(...)`, find the handler's struct_id from its header, index into `DispatchTable.struct_dispatch`, and call the resulting function pointer via the save-volatile-registers trampolines (`call_fn_3`-style). No parser involvement.
     - *(b)* Rename the protocol method. Change `fn handle(...)` in `beagle.effect` to something like `fn handle-op(...)` so a Beagle-side helper can call `handler.handle-op(op, resume)`. Breaks every user handler and every stdlib handler that implements `Handler`.

     Path (a) is the cleaner one. Budget for: understanding how `beagle.effect/Handler<EnumType>` gets registered (see `add_protocol_info` in `src/runtime.rs`, called from `extend` compilation) and when the dispatch table is populated.

   - `beagle.builtin/panic` (called from `Ast::Perform`'s error paths for null enum_type and null handler) is **not actually registered**. This is a latent bug that masks the null-enum-type case: a `perform` on a non-enum falls through into `string-concat` and produces "TypeError: Expected string, got Null" instead of "perform requires an enum value". Worth fixing separately — either register the primitive or switch to `beagle.primitive/panic`.

   - `get_enum_type_builtin` returning null for a valid-looking variant (`MyEffect.GetValue` produced a heap pointer whose `struct_id` wasn't in `runtime.variant_to_enum`) indicates enum-variant registration isn't wired for the chez-demo case. Needs investigation before perform can succeed end-to-end.

2. **Tag-aware perform boundary (Step E6 + E8 continued).** Once perform_effect works untagged, swap to `capture-continuation-tagged` + `return-from-shift-tagged` using the tag stashed in `__handle_tag_<n>__` by `Ast::Handle`. The `find-handler-tag` builtin is already in place.

3. **Deep-handler wrapping for resume** (E7 / E6.4) — wrap the resume closure so a re-perform inside the resumed body installs the same handler. Template: `Ast::Try`'s resumable-exception resume closure construction in commit `d6019cf`.

---

## Background (read first)

- `docs/exceptions-effects-plan.md` — the overall design doc. Read this first.
- Resumable exceptions now work via reset+shift (commit `d6019cf`). That's the template pattern for effect handlers — each step below mirrors something that was done for exceptions.
- Test baseline: **265/323 passing, 58 failing.** Don't regress this. Effect-handler tests are mostly in the 58 failing — they're the target.

---

## What's already in place (Steps E1–E4)

### E1: Handler registry runtime (commit `e6b9505`, `src/builtins/effects.rs`)

```rust
pub struct HandlerRegistryEntry {
    pub protocol_key: String,
    pub handler_instance: usize,
    pub tag: u64,     // currently always 0 — Step E8 populates this
}
```

Implemented and working:
- `push_handler_builtin(protocol_key_ptr, handler_instance)` — appends to `ptd.effect_handlers`.
- `pop_handler_builtin(protocol_key_ptr)` — removes topmost matching entry.
- `find_handler_builtin(sp, protocol_key_ptr)` — returns tagged handler or null.
- `get_enum_type_builtin(sp, fp, value)` — returns enum name as Beagle string.

Still stubbed:
- `call_handler_builtin` — Step E6 implements this.
- `perform_effect_runtime_with_saved_regs` — Step E6 replaces this.

### E2: Prompt-tag side stack (commit `e6b9505`, `src/runtime.rs:3660-3688`)

```rust
pub struct PromptTagRecord {
    pub tag: u64,
    pub stack_pointer: usize,
    pub frame_pointer: usize,
    pub link_register: usize,
}
```

In `PerThreadData.prompt_tags: Vec<PromptTagRecord>`.

Runtime methods (`src/runtime.rs` around line 4790):
- `push_prompt_tag(tag, sp, fp, lr) -> u64`
- `pop_prompt_tag(expected_tag) -> PromptTagRecord`
- `find_prompt_tag(tag) -> Option<(index, PromptTagRecord)>`
- `truncate_prompt_tags(len)`

### E3: Tag builtins (commit `e6b9505`, `src/builtins/continuations.rs`)

- `push_prompt_tag_runtime(tag, sp, fp, lr)` — builtin wrapper around `Runtime::push_prompt_tag`.
- `pop_prompt_tag_runtime(tag)` — builtin wrapper around `Runtime::pop_prompt_tag`.

Both registered in `src/builtins/install.rs` (~line 378-396):
- `beagle.builtin/push-prompt-tag` (4 args: tag, sp, fp, lr)
- `beagle.builtin/pop-prompt-tag` (1 arg: tag)

### E4: Tag-aware capture + return (commit `0622cda`, `src/builtins/reset_shift.rs`)

- `capture_continuation_tagged_runtime` (line ~517) — **registered and tested.**
- `return_from_shift_tagged_runtime` (line ~770) — **registered** (commit `0c01350`).

Registered as `beagle.builtin/capture-continuation-tagged` (5 args) and
`beagle.builtin/return-from-shift-tagged` (4 args).

---

## The SIGSEGV Issue (RESOLVED in commit `0c01350`)

**Root cause.** `continuation_trampoline` ran `GC_FRAME_TOP.with(|cell| cell.set(...))`
*inside* the no-call critical section (after copying segment bytes over a `dst`
that overlaps its own stack frame). `LocalKey::with` normally inlines to a
direct TLS access on arm64-darwin, but when another Rust caller of
`GC_FRAME_TOP.with` exists (e.g., `return_from_shift_tagged_runtime`), the
inliner may emit it as an out-of-line call — which pushes a frame below SP
straight into `dst`, corrupting the just-copied continuation bytes.

**Fix.** Cache the cell's raw backing pointer (`cell.as_ptr()`) *before* the
critical section and use a plain store to update `GC_FRAME_TOP`. No
`LocalKey::with` call in the critical section regardless of inliner choices.

## Legacy notes (for reference)

**Symptom:** Adding `beagle.builtin/return-from-shift-tagged` to the builtin registry (via `add_builtin_function_with_fp` in `install.rs`) causes `resources/repl_main_resume_test.bg` to SIGSEGV reliably. The test runs threads and exercises the resumable-exception machinery; it doesn't call `return-from-shift-tagged` at all.

**Reproduces with:**
- The actual `return_from_shift_tagged_runtime` function pointer
- The known-good `return_from_shift_runtime` function pointer (under the new name)
- Both `needs_frame_pointer=true` and `needs_frame_pointer=false`
- Argument counts 3 and 4
- With and without `#[no_mangle]` / `#[inline(never)]`

**What worked:**
- Registering `capture-continuation-tagged` alone → test passes.
- Adding `return-from-shift-tagged` after it → test fails.

**Hypothesis:** Jump-table page allocation. `add_jump_table_entry` grows the jump table by adding pages (`src/runtime.rs:~7897`). The 4th additional registration might cross a page boundary that triggers a realloc/remap, and something cached before that remap (maybe `RESET_CODE_RANGE` or a stack-walk address check) becomes stale.

**Diagnostic suggestions:**
1. Compare `self.jump_table_pages.len()` before/after the registration. Does it grow by 1?
2. Check if `RESET_CODE_RANGE` (`src/builtins/reset_shift.rs:~80`) is initialized before or after `install_all`. If before, and `__reset__`'s code is moved by later registrations, the cached range is stale.
3. Run under lldb: `cargo build --release` then `lldb target/release/beag -- test resources/repl_main_resume_test.bg` and inspect the SIGSEGV site.
4. Try reordering: register `return-from-shift-tagged` BEFORE the other tagged builtins. If the test still fails → it's about the ADDITIONAL-builtin count, not the specific function.
5. Try adding a dummy `no-op` builtin before `return-from-shift-tagged` to see if it's purely position-sensitive.

**Must be resolved before E6.** E6 calls this function at runtime; an unregistered function can't be called via the Beagle calling convention.

---

## Step E5: Nested-tag tracking (optional, may defer)

**Only needed for nested handlers of DIFFERENT effects.** If `handle<A> { handle<B> { perform A.foo } }` works (A reached past the nested B), E5 is done-enough. If shifted continuations that contain nested resets don't restore correctly on invoke, E5 is needed.

**Design:** Extend `ContinuationObject` with one more field:

```rust
const FIELD_NESTED_TAGS_PTR: usize = 4;  // new
pub const NUM_FIELDS: usize = 5;          // was 4
```

Points to a heap-allocated array of 3-tuples: `(tag, segment_relative_fp_offset, link_register)`. One entry per nested-tag record that was popped during capture.

**In `capture_continuation_tagged_runtime`** (`reset_shift.rs:~517`), change:
```rust
runtime.truncate_prompt_tags(idx + 1);  // pops matched + above
```
to:
```rust
// Collect records above the match before truncating.
let nested: Vec<(u64, usize, usize)> = ptd.prompt_tags[idx+1..]
    .iter()
    .map(|r| (r.tag, r.frame_pointer - stack_pointer, r.link_register))
    .collect();
runtime.truncate_prompt_tags(idx + 1);  // same as before
// Store `nested` in ContinuationObject. Allocate heap array, fill, write ptr.
```

**In `continuation_trampoline`** (`reset_shift.rs:~786`), after splicing:
```rust
// Re-push each nested-tag record with SP/FP adjusted for dst.
for (tag, fp_offset, lr) in cont.nested_tags() {
    let fp_abs = dst + fp_offset;
    let sp_abs = ???; // need to also store SP offset, not just FP
    runtime.push_prompt_tag(tag, sp_abs, fp_abs, lr);
}
```

Also pop these on normal body completion (when the spliced frames return into the trampoline). This is handled by the overlay mechanism that already returns to the trampoline — just pop in the trampoline's return path.

**Acceptance test:** Write a `.bg` test with two nested `handle` blocks of different effect types, where the inner handle doesn't handle the outer's effect. `perform A.foo` from inside the inner handle must reach the outer handle correctly.

---

## Step E6: Implement perform_effect (THE BIG STEP)

This is where effect handlers become real. Four sub-pieces, do them in order:

### E6.1: Tag propagation through the registry

Modify `HandlerRegistryEntry.tag` to actually get populated:
- Change `push_handler_builtin` signature to take a tag parameter (3 args: `protocol_key_ptr, handler_instance, tag`).
- Update registration in `install.rs` (~line 1926) to take 3 args.
- Update `Ast::Handle` compilation in `ast.rs:4424` to generate a fresh tag and pass it.

Fresh tag generation — reuse `runtime.prompt_id_counter` (already exists, atomic):
```rust
let tag = runtime.prompt_id_counter.fetch_add(1, Ordering::SeqCst) as u64;
```
This needs to happen at compile time or be threaded through at runtime. Simplest: add a `fresh_tag` builtin that returns a fresh u64. `Ast::Handle` calls it, stashes the tag in a local, threads it through push_handler, push_prompt_tag, and the body.

### E6.2: find-handler returns both handler AND tag

Either:
- **(a)** Add a second builtin `beagle.builtin/find-handler-tag(protocol_key) -> tag` — scans the registry, returns the tag of the matching entry.
- **(b)** Change `find_handler_builtin` to return a 2-word heap object or a pair.

Option (a) is simpler. Implement it in `src/builtins/effects.rs` next to `find_handler_builtin`. Register in `install.rs`.

### E6.3: perform_effect does tagged shift

Rewrite the body of `perform_effect_runtime_with_saved_regs` (`src/builtins/effects.rs:~165`):

```rust
pub unsafe extern "C" fn perform_effect_runtime_with_saved_regs(
    stack_pointer: usize,
    frame_pointer: usize,
    enum_type_ptr: usize,
    op_value: usize,
    resume_address: usize,
    result_local_offset_raw: usize,
    _saved_regs_ptr: *const usize,
) -> usize {
    // Find the handler and its tag.
    let runtime = get_runtime().get();
    let protocol_key = compute_protocol_key(enum_type_ptr);  // "Handler<{enum_type_name}>"
    let (handler, tag) = ptd.effect_handlers.iter()
        .rev()
        .find(|e| e.protocol_key == protocol_key)
        .map(|e| (e.handler_instance, e.tag))
        .expect("no handler");

    // Capture with tagged variant.
    let cont_ptr = capture_continuation_tagged_runtime(
        stack_pointer, frame_pointer,
        resume_address, result_local_offset,
        tag as usize,
    );

    // Wrap cont_ptr in a closure, call handler.handle(op_value, resume_closure).
    let resume_closure = wrap_in_trampoline_closure(cont_ptr);
    let result = call_handler_protocol_dispatch(handler, enum_type, op_value, resume_closure);

    // Longjmp with result.
    return_from_shift_tagged_runtime(
        stack_pointer, frame_pointer, result, tag as usize,
    );
}
```

**call_handler_protocol_dispatch** is the hard part — you need to look up the handler struct's `handle` method by protocol dispatch. Look at how `Ast::CallExpr` compiles protocol method calls in `src/ast.rs`, or check if there's an existing builtin for protocol dispatch (search `dispatch`). If nothing exists, you'll need to build protocol dispatch in Rust, which involves reading the struct's type id, looking up the protocol implementation table, finding the method offset, and calling it.

**Alternative: do the dispatch at the Beagle level.** Instead of a single `perform_effect` builtin, lower `Ast::Perform` to a sequence of Beagle/IR operations:
```
tag = find-handler-tag(protocol_key)
handler = find-handler(protocol_key)
raw_k = capture-continuation-tagged(..., tag)
resume_closure = make_closure(trampoline, 1, [raw_k])
result = handler.handle(op_value, resume_closure)   // normal method call via IR
return-from-shift-tagged(result, tag)              // noreturn
after_shift_label:
...
```

This sidesteps `call_handler_builtin` entirely. Cleaner. Model it on `Ast::Shift`'s compilation (`ast.rs:~4047`).

### E6.4: Deep-handler wrapping (same as Step E7)

When perform_effect creates the resume closure, wrap it the same way Ast::Try does for resumable exceptions:

```
resume_closure(v) = fn(v') {
    reset {
        push_handler(key, handler, new_tag)
        push_prompt_tag(new_tag, ...)
        raw_k(v')
    }
}
```

This re-installs the handler so a second `perform Op` inside the resumed body dispatches correctly. Without it, the first `perform` works but subsequent performs in the resumed body fail.

See commit `d6019cf` in `src/ast.rs:~1790` (the resumable-try resume-closure construction) for the pattern — build an `Ast::Function` with the wrapping body, compile via `call_compile`.

---

## Step E7: Deep-handler wrapping for resume

Subsumed into Step E6.4 above. The deep-handler wrapping for effect handlers is the same shape as the one for resumable exceptions. Just apply the same `Ast::Function { body: [Ast::Reset { body: [Ast::CallExpr { ... }] }] }` pattern used in `Ast::Try`.

**Acceptance tests:** `handler_choice_multishot_test.bg`, `handler_amb_test.bg`, `handler_deep_call_test.bg`.

---

## Step E8: Wire Ast::Handle with tags

`Ast::Handle` (`src/ast.rs:~4424`) currently does:

```
push_handler(key, handler_instance)
reset { body }
pop_handler(key)
```

Change to:

```
tag = fresh_tag()   // new builtin, or use prompt_id_counter directly
push_handler(key, handler_instance, tag)   // now 3 args
push_prompt_tag(tag, sp, fp, after_handle_label)
<body>
pop_prompt_tag(tag)
pop_handler(key)
after_handle_label:
```

Note: the `reset { body }` wrap currently in `Ast::Handle` may need to be removed or kept depending on whether you use tag-only boundary or dual (reset + tag). Recommend: remove the reset wrapper for handle, use the prompt-tag as the sole boundary. The tagged capture doesn't need a `__reset__` frame.

**But** note that for the shift body (inside `perform_effect`), there may still be a need for a reset boundary for any nested shifts/perform inside the handler body itself. Test carefully.

**Acceptance:** All remaining effect-handler tests pass. Target: 310+/323.

---

## Files to read (in order)

1. `docs/exceptions-effects-plan.md` — overall design.
2. `src/runtime.rs:3645-3710` — `HandlerRegistryEntry`, `PromptTagRecord`, `PerThreadData`.
3. `src/runtime.rs:~4790` — registry helper methods.
4. `src/builtins/effects.rs` — handler registry builtins (stubs for call_handler / perform_effect).
5. `src/builtins/continuations.rs:~80-110` — push_prompt_tag / pop_prompt_tag.
6. `src/builtins/reset_shift.rs:~517` — `capture_continuation_tagged_runtime`.
7. `src/builtins/reset_shift.rs:~770` — `return_from_shift_tagged_runtime`.
8. `src/builtins/install.rs:~363-420` — builtin registrations (also where SIGSEGV triggers).
9. `src/ast.rs:~1645-1926` — `Ast::Try` resumable compilation (template for handler wrapping).
10. `src/ast.rs:~4282-4485` — `Ast::Perform` and `Ast::Handle` (the targets).
11. `src/builtins/reset_shift.rs:~786` — `continuation_trampoline` (for E5 nested-tag re-push).

## Commands

```bash
# Build
cargo build --release

# Full suite
cargo run --release -- test resources/

# Single test
cargo run --release -- test resources/chez_handle_demo.bg

# Debug run (not via test harness, shows output)
cargo run --release -- resources/chez_handle_demo.bg

# Effect-handler test files to target in order of simplicity
# (inspect each to understand what shape of handler it expects):
resources/chez_handle_demo.bg
resources/custom_handler_test.bg
resources/handler_exception_test.bg
resources/handler_choice_multishot_test.bg
resources/handler_amb_test.bg
resources/handler_deep_call_test.bg
resources/async_implicit_handle_test.bg
resources/async_implicit_ops_test.bg
```

## Test suite discipline

- Each step must leave **≥265 tests passing, 0 new regressions**.
- If you regress a test, revert that step immediately and investigate.
- Use `git stash` to compare baseline: `git stash && cargo run --release -- test resources/ | tail -3 && git stash pop`.

## Don't do these

- Don't modify reset/shift primitives (`capture_continuation_runtime_inner`, `return_from_shift_runtime_inner`, `continuation_trampoline`) — they work correctly and are load-bearing.
- Don't remove `__reset__` from `std.bg` — resumable exceptions depend on it.
- Don't inline `Ast::Reset` — the body-thunk outlining is required (frames need their own locals area). See the "Why the Earlier Plan Was Wrong" section in `exceptions-effects-plan.md`.
- Don't use `#[no_mangle]` on the tagged runtime functions — caused a different SIGSEGV in testing.

## If you get stuck

- The SIGSEGV is the most likely blocker. If it can't be resolved in ~2 hours, consider alternative approaches: register the function somewhere other than immediately after capture-continuation-tagged; or inline return_from_shift_tagged's logic into perform_effect_runtime to avoid the separate registration entirely.
- Protocol dispatch for `handler.handle()` is the second biggest unknown. If call_handler_builtin feels impossible to write, do the "dispatch at Beagle level" alternative in E6.3 — compile `Ast::Perform` to emit the method call via normal IR, not via a builtin.
- Effect-handler tests can have surprising semantic expectations. If a test's expected output looks weird, read the test carefully and make sure the deep/shallow semantics match what the test expects.
