# Plan: Fix Resumable Exceptions and Enable Effect Handlers

**Status:** Draft. Supersedes the "Refactor B" plan in `src/builtins/reset_shift.rs` comments.

**Last updated:** 2026-04-13

## TL;DR

1. **Resumable exceptions are broken** because `throw_exception` reuses `capture_continuation_runtime`, which requires a `__reset__` frame on the stack — and `try/catch` doesn't install one.
2. **Fix:** compile `try/catch-resumable` to `reset { ... shift ... }`. This installs the `__reset__` frame. No changes to the capture primitive.
3. **Effect handlers** (future work) need one small extension: **per-handler prompt tags**. Single-prompt reset/shift is insufficient for nested handlers of different effects. Every production system (Koka, libhandler, Multicore OCaml, Eff, Effekt) uses prompt tags.
4. **No other changes** to reset/shift are needed. The earlier plan to replace the FP-walker with a prompt stack was aesthetic, not functional, and turned out to be harder than estimated. Abandoned.

---

## Background

### Current working mechanism: reset/shift

```
reset { body }   ==>   __reset__(body_thunk)
```

- Body is outlined into a zero-arg thunk closure.
- `__reset__` is a Beagle stdlib function defined as `fn __reset__(thunk) { thunk() }`.
- Its only purpose: leave a recognizable frame on the stack.
- At `shift` time, `capture_continuation_runtime` walks the FP chain upward looking for the topmost frame whose saved LR points into `__reset__`'s code range. That frame's FP (= body_thunk's FP) marks the prompt boundary.
- Capture copies `[current_SP, outermost_body_fp + 16)` to the heap, relativizes saved-FP links, and returns a `ContinuationObject`.
- `return_from_shift` walks the same FP chain to longjmp past `__reset__` back to its caller.
- All reset/shift tests currently pass. **This machinery stays unchanged.**

### Current broken mechanism: resumable exceptions

`try/catch-resumable` is compiled in `Ast::TryResumable` (`src/ast.rs` ~line 1700). It:

- Pushes an `ExceptionHandler` record onto `per_thread_data().exception_handlers`.
- Outlines the try body into a thunk.
- Calls the thunk.
- Pops the handler on normal completion.

When `throw(value)` fires inside the try body:

- `throw_exception` (`src/builtins/exceptions.rs:81`) pops the top `ExceptionHandler`.
- If the handler is resumable, it calls `capture_continuation_runtime(sp, fp, resume_address, resume_local_offset)`.
- **This panics** with *"shift without an enclosing reset"* because `capture_continuation_runtime`'s FP-chain walk finds no `__reset__` frame — try/catch didn't install one.

Failing tests today:
- `resources/resumable_exception_test.bg`
- `resources/resumable_multi_throw_test.bg`
- `resources/resumable_eval_test.bg`
- `resources/repl_resume_test.bg`
- `resources/repl_main_resume_test.bg`

---

## The Fix (one AST-level rewrite)

Compile `try { B } catch e resume k => H` to the moral equivalent of:

```beagle
reset {
    try-with-abortive-handler {
        B
    } catch-abortive e' => {
        // shift captures the continuation; bind k, run H
        shift(raw_k => {
            let k = wrap_as_closure(raw_k)
            let e = e'
            H
        })
    }
}
```

Mechanically in `Ast::TryResumable`:

1. Emit `__reset__(body_thunk)` wrapping the try body.
2. Inside `body_thunk`, install an **abortive** exception handler that, on match, executes a `shift` whose body binds `e` and `k` and runs `H`.
3. Non-resumable throw keeps the existing abortive path unchanged — only resumable throw gets routed through shift.

This installs the `__reset__` frame, so `throw_exception`'s call to `capture_continuation_runtime` finds a prompt boundary. No runtime changes needed.

### What gets deleted

- The resumable branch of `throw_exception` (currently broken anyway).
- `push_resumable_exception_handler_runtime` and `pop_exception_handler_by_id_runtime`.
- The `is_resumable` / `resume_local` / `handler_id` fields on `ExceptionHandler`.
- References to these builtins in `src/builtins/install.rs`.

### What stays

- `__reset__`, `find_enclosing_reset_frame`, `is_pc_in_reset_function`, `RESET_CODE_RANGE` — all untouched.
- `capture_continuation_runtime_inner`, `return_from_shift_runtime_inner`, `continuation_trampoline` — all untouched.
- Non-resumable try/catch (the abortive path) — keeps its own fast path, no capture.

---

## Task Breakdown

### Task A: Revert Step 1 dead code (prerequisite, ~15 min)

Remove from `src/runtime.rs`:
- `HandlerFilter` enum
- `PromptKind` enum
- `PromptRecord` struct
- `prompts: Vec<PromptRecord>` field on `PerThreadData`
- `push_prompt` / `pop_prompt` / `top_prompt` methods on `Runtime`

These were added in an earlier design iteration that is no longer needed. Cheap to revert.

**Acceptance:** `cargo build --release` clean, `cargo run --release -- test resources/` still passes 259/64.

### Task B: Compile resumable try/catch to reset+shift (the actual fix, ~2-4 hours)

Locate `Ast::TryResumable` in `src/ast.rs` (~line 1700). Rewrite its compilation:

1. **Wrap the whole thing in a reset.** Outline a new body thunk that contains everything the current code does, call it via `__reset__`.
2. **Inside that reset, install an abortive handler** (using the existing `push_exception_handler_runtime` path) whose `handler_address` points to a label that runs a `shift`.
3. **The shift body** binds `e` to the thrown exception (written by `throw_exception` to `result_local`), and binds `k` to a closure wrapping the raw continuation pointer (same wrapping pattern `Ast::Shift` uses today — `make_closure(continuation_trampoline, 1, [cont_ptr])`). Then compiles the user's `H` expression with `e` and `k` in scope.
4. **`throw_exception` stays abortive.** For resumable exceptions, it still pops the handler, writes exception into `result_local`, and jumps to the handler label. But now that label is inside a reset, and the label's code issues a shift — capture finds the reset, all is well.

**Key simplification:** `throw_exception`'s resumable branch goes away. It only ever does the abortive jump now. The "resumable" behavior emerges from the generated handler code calling shift.

**Tests to pass:**
- `resources/resumable_exception_test.bg`
- `resources/resumable_multi_throw_test.bg`
- `resources/resumable_eval_test.bg`
- `resources/repl_resume_test.bg`
- `resources/repl_main_resume_test.bg`

**Tests that must not regress:** all 259 currently-passing tests.

### Task C: Add prompt tags to reset/shift (DEFERRED)

Don't do this until re-enabling `Ast::Handle` / `Ast::Perform`. Single-prompt reset/shift is sufficient for resumable exceptions (nearest-catch semantics is correct).

When effect handlers get re-enabled:

1. Extend `ContinuationObject` (or the prompt-identification mechanism) to carry a **prompt tag** (fresh `u64` per `reset`).
2. Generate a fresh tag at each `reset` entry; store it in `__reset__`'s frame (or in a parallel side-stack).
3. `shift(tag)` takes a tag parameter; capture walks up past frames whose tag doesn't match.
4. Handler dispatch (via the existing `find-handler` registry) produces the tag for `perform`'s shift call.
5. Escape-to-wrong-handler (closure with `perform` invoked outside its handle) detected cleanly: registry lookup returns None → `UnhandledEffect`.

This mirrors libhandler's design (closest analog to Beagle's stack-copying runtime) and Koka's evidence-passing (after ICFP 2021).

---

## Why the Earlier Plan Was Wrong

An earlier version of this plan proposed:

- Replace the FP-walker with a per-thread prompt stack.
- Inline `reset { body }` instead of outlining body as a thunk.
- Push a `PromptRecord` at reset entry, pop at exit, peek at capture.

Problems discovered during implementation:

1. **Inline reset breaks continuations.** Inline body runs in the caller's frame, so body's locals live in caller's frame — which is ABOVE `capture_top` and therefore NOT captured. When the continuation is invoked elsewhere, its code tries to use locals that aren't in the segment. Result: 7 regressions, all panicking with "continuation has no segment data" because `stack_size == 0`.
2. **Body-thunk outlining is load-bearing.** The thunk exists precisely so body's locals have a frame to live in, and that frame is what gets captured. This is the same pattern libhandler uses. It cannot be inlined away.
3. **Reconciling `capture_top` vs `new_sp` for __reset__'s frame** is painful. In the __reset__-based design these differ by 16 bytes; in an inline design they're equal. The intermediate "keep __reset__ but push a prompt" state requires the prompt record to encode this offset, which couples the prompt to __reset__'s layout.

**Conclusion:** the frame that `__reset__` / body-thunk provides is essential, not incidental. Don't remove it. The FP-walker is fine.

The "prompt stack replaces FP walker" idea is still possible in principle, but it's an aesthetic refactor — the functional goal (fix resumable exceptions) doesn't need it.

---

## Research Corrections

Two claims in earlier writeups were wrong; corrected here:

**WRONG:** "Nobody walks the FP chain; every production system uses an explicit prompt stack."
**RIGHT:** Most production systems use a handler side-stack, but not as a replacement for a native stack walk — it carries prompt tags and handler identity. Beagle's FP-walker is unusual but not fundamentally broken.

**WRONG:** "Single-prompt reset/shift + a handler registry is enough for full effect handlers."
**RIGHT:** Only sufficient for resumable exceptions (nearest-handler semantics). General effect handlers with nested handles of different effects need prompt tags. Every production system uses them. Single-prompt + re-shift bubbling is theoretically equivalent (Dybvig/Peyton-Jones/Sabry 2007) but nobody ships it because: (a) O(depth) per perform, (b) silent wrong-handler bugs with escaped closures, (c) per-level stack copies in a copying runtime.

References:
- Leijen, *Type Directed Compilation of Row-Typed Algebraic Effects*, POPL 2017
- Xie, Brachthäuser, Hillerström, Schuster, Leijen, *Generalized Evidence Passing for Effect Handlers*, ICFP 2021
- Sivaramakrishnan et al., *Retrofitting Effect Handlers onto OCaml*, PLDI 2021
- Hillerström & Lindley, *Shallow Effect Handlers*, ICFP 2018
- Dybvig, Peyton-Jones, Sabry, *A Monadic Framework for Delimited Continuations*, JFP 2007
- libhandler source: `github.com/koka-lang/libhandler`

---

## Summary for the Impatient

1. Revert Task A (dead code cleanup).
2. Do Task B (compile resumable try to reset+shift). This fixes the 5 failing tests.
3. Defer Task C until effects are re-enabled.

No changes to `reset_shift.rs` or the continuation runtime. The fix is entirely in `Ast::TryResumable` compilation.
