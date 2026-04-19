# Continuation Stack Restoration — Approach Evaluation

See `docs/continuation-stack-restoration.md` for the problem description.

> Both backends (ARM64 and x86-64) now share the pure-Rust continuation
> trampoline in `src/builtins/reset_shift.rs` and differ only in the
> small JIT helpers installed by `compile_arm_continuation_return_stub`
> / `compile_x86_continuation_return_stub` in `src/main.rs`.

## Chosen Approach: Segmented Stack with Lazy Restoration (Chez Scheme style)

See the detailed implementation plan in the next section.

## All Approaches Evaluated

### A: Per-Prompt Virtual Memory Stacks (libmprompt style)
Each `handle` block mmaps a new stack. perform/resume = SP switch. Zero copy, zero descent. Large refactor (codegen + runtime + GC). ~100ns/op. Proven by libmprompt (10M ops/sec).

### B: Segmented Stack with Lazy Restoration (Chez Scheme style) ← CHOSEN
Stack is a linked list of segments. Capture = split the chain (O(1)). Resume = copy one segment + install underflow handler. Bounded copy per resume. Complex but most efficient for multi-shot. Challenge: Rust FFI interleaving.

### C: OCaml-Style Fibers
Small growable stacks per handler. Zero-copy one-shot capture/resume. Growth requires pointer fixup — showstopper with interleaved Rust frames.

### D: Guile-Style Recursive Growth
Recursively grow native stack below target, then memcpy. Doesn't solve the effect handler case (handler always overlaps captured region).

### E: Slot Reuse (Fix RP Lifecycle)
Reuse previous invocation's stack slot. Zero descent, minimal change. Unknown crash when attempted — RP removal breaks something in the continuation lifecycle.
