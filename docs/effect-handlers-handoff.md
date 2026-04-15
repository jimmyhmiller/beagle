# Effect Handlers — Honest Handoff

**Last updated:** 2026-04-14

**Read this before writing any code.** The previous handoff overstated how close the implementation was. The scaffolding is real, but the foundation it sits on is a poor match for how the rest of the compiler dispatches protocols. A cleanup step almost certainly needs to precede Step E6.

---

## TL;DR

- Reset/shift primitives work. Exceptions ride on them. Handle-without-perform works.
- `perform` does not work end-to-end and has probably never worked in this branch.
- The effect-handler registry uses **runtime string-keyed dispatch** — every `perform` allocates two strings, concatenates them into a key like `"beagle.effect/Handler<MyEffect>"`, and linear-scans a `Vec<HandlerRegistryEntry>` comparing strings. The rest of the compiler dispatches protocols via integer type-id lookups into `DispatchTable`. Effect handlers have a parallel, slower, weirder path that exists nowhere else.
- Recommendation: replace the string-keyed registry with an enum-type-id-keyed registry *before* implementing `perform_effect_runtime`. See "The work" below.
- Baseline: **265/323 tests passing, 58 failing.** Don't regress this. Most of the 58 exercise `perform`.

---

## What actually works

| Piece | Status | File |
| --- | --- | --- |
| `reset { ... }` / `shift(k) { ... }` | Works | `src/builtins/reset_shift.rs` |
| Resumable `try/catch` (rides on reset+shift) | Works | `src/ast.rs` (Ast::Try) |
| `__reset__` as a prompt frame | Works | stdlib + `RESET_CODE_RANGE` |
| `continuation_trampoline` (resume path) | Works, fragile — see SIGSEGV notes | `src/builtins/reset_shift.rs:901` |
| `handle { body }` without `perform` | Works | `Ast::Handle` now compiles through `__reset__` |
| `capture-continuation-tagged` builtin | Registered, not exercised yet | `src/builtins/reset_shift.rs:~517` |
| `return-from-shift-tagged` builtin | Registered (as of SIGSEGV fix) | `src/builtins/reset_shift.rs:~770` |
| `push-prompt-tag` / `pop-prompt-tag` | Registered, not called from AST | `src/builtins/continuations.rs:~80` |
| `fresh-tag`, `find-handler-tag` | Registered | `src/builtins/effects.rs` |
| `push-handler` with tag (3 args) | Registered, tag threaded through `Ast::Handle` | `src/builtins/effects.rs` |

## What does not work

- `perform op`. Compiles to the stubbed `beagle.builtin/perform-effect`. Even reaching that stub fails first on string-concat because of the setup pipeline described below.
- `beagle.builtin/panic`, called from `Ast::Perform`'s error paths (null enum-type, null handler). **Not registered anywhere.** Latent.
- `get-enum-type` returns `None` for freshly-declared enum variants in simple test cases (`variant_to_enum[struct_id]` lookup misses). Needs investigation — might just be registration ordering.

---

## The pieces, in plain language

### Reset/shift

- **`__reset__(thunk)`** is a Beagle stdlib function whose entire body is `thunk()`. Its only purpose is to leave a *recognizable frame* on the stack. `RESET_CODE_RANGE` (a `OnceLock`) caches its compiled code's byte range. `is_pc_in_reset_function(pc)` asks "does this PC lie in `__reset__`'s code?"
- **`capture_continuation(sp, fp, resume_addr, result_local)`** — walks the FP chain upward looking for the frame whose *child's* saved LR points into `__reset__`'s code range. That's the prompt. Copies the stack bytes from `sp` up to (but not including) `__reset__`'s frame into a heap segment. Rewrites saved-FP slots inside the copy to **segment-relative offsets** so GC can relocate the segment. Returns a `ContinuationObject` pointer.
- **`return_from_shift(sp, fp, value, _cont)`** — same walk, then longjmps to simulate `__reset__` returning normally to its caller, with `value` as the return register.
- **`continuation_trampoline(closure, value)`** — the body of a continuation closure. Copies segment bytes back onto the stack (at `trampoline_fp - outermost_offset`), rewrites FP offsets back to absolute, rebuilds the GC prev chain, writes `value` into the resume point's result slot, updates `GC_FRAME_TOP`, then `return-jump`s to the resume address.
  - **Critical invariant:** between the first byte-copy and the final `return-jump`, NO function calls may be emitted — any call would push a frame below SP into the `dst` region we're writing, corrupting the just-copied frames. This is why the trampoline caches `GC_FRAME_TOP.with(|c| c.as_ptr())` *before* the copy: a plain store in the critical section emits no call regardless of inliner decisions. Violating this was the SIGSEGV resolved in commit `0c01350`.

### Handler registry (string-keyed)

```rust
pub struct HandlerRegistryEntry {
    pub protocol_key: String,        // "beagle.effect/Handler<MyEffect>"
    pub handler_instance: usize,
    pub tag: u64,
}
```

Stored on `PerThreadData.effect_handlers`. `push-handler` and `pop-handler` are the interface. `find-handler` is a linear `.iter().rev().find(|e| e.protocol_key == key)`.

### Prompt-tag side stack

```rust
pub struct PromptTagRecord {
    pub tag: u64,
    pub stack_pointer: usize,
    pub frame_pointer: usize,
    pub link_register: usize,
}
```

Stored on `PerThreadData.prompt_tags`. The "tagged" capture/return variants use this instead of FP-walking for `__reset__` — they match by `tag` rather than code range.

---

## How a `handle`/`perform` is supposed to flow

```beagle
handle beagle.effect/Handler(MyEffect) with h {
    perform MyEffect.Foo { x: 42 }
}
```

**At Handle:**
1. Evaluate `h`.
2. `tag = fresh-tag()` (atomic counter).
3. `push-handler("beagle.effect/Handler<MyEffect>", h, tag)`.
4. `result = __reset__(body_thunk)`.
5. `pop-handler(key)`.
6. Return `result`.

**At Perform:**
1. Evaluate the op value (a `MyEffect.Foo { x: 42 }` struct instance).
2. `enum_type = get-enum-type(op)` — reads the struct header's type_id, looks up `runtime.variant_to_enum[type_id]` to get `"MyEffect"`, **allocates a Beagle string** for it.
3. `partial = string-concat("beagle.effect/Handler<", enum_type)` — allocates.
4. `protocol_key = string-concat(partial, ">")` — allocates.
5. `handler = find-handler(protocol_key)` — linear scan, string-compare each entry.
6. Capture continuation, wrap in trampoline closure to get `resume`.
7. Call `handler.handle(op, resume)` — needs to go through protocol dispatch.
8. Longjmp back to after the handle with the handler's result.

Step 7 is what `perform_effect_runtime_with_saved_regs` in `src/builtins/effects.rs` is supposed to do. It's currently a stub.

---

## What's genuinely a hack (and not how the rest of the compiler works)

1. **String-keyed handler registry with runtime concat.** No other protocol in the compiler works this way. Ordinary protocol calls dispatch via `DispatchTable` (`src/runtime.rs:3563`) keyed by integer type_id — a dense `Vec<usize>` indexed by struct_id. Two pointer loads, one indirect call. Effect handlers allocate two strings and linear-scan a Vec of Strings at every `perform`.
2. **Runtime protocol-key building.** The key is deterministic from the enum type. It could be interned once, or better, avoided entirely by keying on the enum's type_id.
3. **`__reset__`-by-code-range.** The prompt is identified by walking FP and asking "does the saved LR lie in `__reset__`'s byte range?" Works, but `RESET_CODE_RANGE` is a cached singleton. Any future relocation/recompile of `__reset__` silently breaks it.
4. **Two parallel prompt mechanisms.** The FP-walker identifies prompts by code range; the prompt-tag side stack identifies them by tag. The tagged variants are written but not primary. Every bit of code has to reason about both worlds during the migration.
5. **Trampoline no-call critical section.** Correctness depends on an optimizer-level property the compiler doesn't know about. A single `thread_local!` call-site added elsewhere in the binary can change LLVM's inline decision and break it. (This is the SIGSEGV we already fixed — but the structural fragility remains.)
6. **`handle` is a Beagle keyword.** And the Handler protocol's method is also named `handle`. So `handler.handle(op, resume)` parses as a field access (which errors on "Field 'handle' does not exist"), and `handle(handler, op, resume)` starts a handle block. Writing a Beagle-side dispatch helper is not viable without renaming either the keyword or the method.
7. **`beagle.builtin/panic` is referenced but not registered.** The null checks in `Ast::Perform` call a function that doesn't exist. Probably masked by the fact that the null paths haven't been exercised.

## What's reasonable-but-complex (don't touch unless you're changing the design)

- Heap-allocated continuation segments with segment-relative FP offsets — the right shape for GC-safe multi-shot continuations.
- `save_volatile_registers_{0,1,2,3}` trampolines for Rust→Beagle reentry — normal JIT plumbing.
- Tags distinguishing nested handlers of different effects — standard for algebraic effects (Racket, Multicore OCaml do the same).

---

## The work

### Strong recommendation: clean up the registry first (pre-E6)

Replace the string-keyed registry with type-id-keyed:

```rust
pub struct HandlerRegistryEntry {
    pub enum_type_id: usize,        // the enum's canonical type_id
    pub handler_instance: usize,
    pub tag: u64,
}
```

- `push-handler(enum_type_id_int, handler, tag)` — compile-time: `Ast::Handle` looks up the enum type by name and passes its type_id. No strings.
- `find-handler(enum_type_id_int)` — integer compare.
- `get-enum-type(op)` returns the type_id as a tagged int, not a string.
- Null-enum case: returns a sentinel (e.g., 0), checked with an integer comparison in IR. No `panic` builtin needed.
- Delete the `get-enum-type` string allocation and both `string-concat` calls from `Ast::Perform`.

While here, also investigate why `variant_to_enum[struct_id]` was missing in my test case — that bug will bite the type-id approach too.

### Then Step E6: perform_effect itself

With the registry cleaned up:

- `Ast::Perform` compiles to (pseudocode):
  ```
  enum_id = get-enum-type(op)
  if enum_id == 0: panic "perform requires an enum value"
  handler = find-handler(enum_id)
  if handler == null: panic "no handler for ..."
  tag = find-handler-tag(enum_id)
  raw_k = capture-continuation-tagged(..., tag, after_perform, result_local)
  resume_closure = make-closure(continuation_trampoline, [raw_k])
  result = dispatch_handle(handler, op, resume_closure)    // see below
  return-from-shift-tagged(result, tag)
  after_perform:
    result_local holds the resumed value
  ```
- `dispatch_handle` is the hard bit. Two viable paths:
  - **Rust-side dispatch.** Implement `perform_effect_runtime_with_saved_regs` in `src/builtins/effects.rs`. Look up the Handler protocol's DispatchTable via `runtime.get_dispatch_table_ptr("beagle.effect/Handler<...>", "handle")` (integer key once you've done the cleanup), index by handler's struct_id, and call via `call_fn_3` (see `src/builtins/apply.rs:1320`). Then the builtin does the whole capture→dispatch→return sequence and the AST lowering is trivial.
  - **Beagle-level dispatch.** Emit IR that calls the protocol dispatcher function directly by pointer — each `extend S with Handler(E)` compiles a function at `"beagle.effect/handle"` that's the inline-cached dispatcher. Bypass the parser entirely: grab its compiled address at `Ast::Perform` compile time and emit a direct call.

The Rust-side path is probably cleaner and avoids dealing with the `handle`-keyword mess.

### Step E7 / E6.4: Deep-handler wrapping for resume

When `perform_effect` builds the resume closure, wrap it so a re-perform inside the resumed body installs the same handler again:

```
resume(v) = fn(v') {
    reset {
        push_handler(enum_id, handler, new_tag)
        push_prompt_tag(new_tag, ...)
        raw_k(v')
    }
}
```

Without this, `perform` works the first time but subsequent `perform`s inside a resumed body dispatch to the wrong (or no) handler. Template: the resume-closure construction in `Ast::Try` (commit `d6019cf`) does this for resumable exceptions.

---

## Files you'll touch

1. `src/runtime.rs:~3645-3710` — `HandlerRegistryEntry`, `PromptTagRecord`, `PerThreadData`. Change `protocol_key: String` → `enum_type_id: usize`.
2. `src/builtins/effects.rs` — all four handler-registry builtins and the stubbed `perform_effect_runtime_with_saved_regs`. The cleanup + Step E6 land here.
3. `src/builtins/install.rs:~1960` — builtin registrations (adjust arities).
4. `src/ast.rs:~4282` — `Ast::Perform` compilation. Rewrite to use the cleaned-up registry + capture/dispatch/return sequence.
5. `src/ast.rs:~4424` — `Ast::Handle`. Already mostly right; `push-handler` args change.
6. `src/builtins/reset_shift.rs:~517` and `~770` — the tagged capture/return functions. Already registered. Probably no changes needed.
7. `src/runtime.rs:8223` — `get_enum_name_for_variant` / `variant_to_enum`. Investigate the registration gap.

## Commands

```bash
cargo build --release
cargo run --release -- test resources/                   # full suite
cargo run --release -- test resources/chez_handle_demo.bg
cargo run --release -- resources/chez_handle_demo.bg     # direct (no test harness)
```

## Test suite discipline

- Baseline is **265/323**. Any step that regresses this is wrong — revert and rethink.
- The 58 failing tests are mostly `perform`-users. They're the target.
- Pick a single test to chase end-to-end first — `chez_handle_demo.bg` is the simplest (three cases: no perform, perform returning a value, perform resuming).

## Relevant commits

- `e6b9505` — handler registry + prompt-tag side stack scaffolding.
- `0622cda` — tagged capture/return written.
- `d6019cf` — resumable exceptions via reset+shift (pattern to follow for E7).
- `0c01350` — SIGSEGV fix in the trampoline (TLS hazard). Read this before touching the trampoline.
- `20600c8` — `fresh-tag` / `find-handler-tag` builtins + Handle threads tag.
- `b8a7bfd` — Handle compiles through `__reset__` instead of stubbed push-prompt.

## Don't do these

- Don't modify the reset/shift inner primitives (`capture_continuation_runtime_inner`, `return_from_shift_runtime_inner`, `continuation_trampoline`) — they work and are load-bearing. The trampoline especially is fragile; read the critical-section comments.
- Don't remove `__reset__` from `std.bg` — resumable exceptions depend on it.
- Don't add fallbacks or "temporary" stubs that return `-1` or silently succeed. Stubs must throw a clear error (`panic!` or `throw_runtime_error`).
- Don't carry the string-keyed registry forward "for now." Replacing it later is harder once more code depends on its shape.
