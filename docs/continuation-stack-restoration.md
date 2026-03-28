# Continuation Stack Restoration

## The Problem

When invoking a delimited continuation, we need to restore captured stack
frames to memory and then jump there.  On platforms where the language stack
and the runtime (Rust/C) stack are the same physical stack — which is the
case for Beagle's JIT on ARM64 and x86-64 — this creates an **overlap
problem**: the Rust code doing the restoration is itself running on the same
stack that the restored frames need to occupy.

### Why Beagle Hits This

Every `socket/read`, `socket/write`, `core/sleep`, and other IO operation
goes through the effect system:

```
perform Async.IO { action: ... }
  → shift (captures continuation)
  → handler receives continuation + operation
  → handler executes the IO
  → handler calls resume(result)
  → invoke_continuation_runtime restores captured frames + jumps
```

In a tight loop like reading a TCP response byte-by-byte, this means
hundreds of `invoke_continuation_runtime` calls.

### The Stack Descent Bug

`invoke_continuation_runtime` computes where to place the restored frames:

```rust
let new_sp = actual_rsp - stack_segment_size - safety_margin;
```

Each invocation places the restored segment BELOW the current stack
pointer.  When the restored code runs and does the next `perform`, the
handler calls `resume(result)` again — but `actual_rsp` is now at the
lower position.  The next `new_sp` goes below THAT.  Linear descent,
forever.

With the original 1 MB safety margin, 8 reads crashed the 8 MB thread
stack.

### Current Fix: 4 KB Margin + Scratch Stack

The safety margin was reduced from 1 MB to 4 KB.  Additionally, the actual
frame restoration (the `restore_to_stack` writes) was moved to a separate
heap-allocated scratch stack, so the margin only needs to cover the Rust
bookkeeping code in Phase 1-2, not the restoration itself.

With 4 KB margin, a 150-byte read uses ~750 KB of stack — well within
the 8 MB thread stack limit.  This is adequate for typical programs but
still technically O(n) in the number of sequential `perform` calls.

### Why Zero Descent Is Hard

The obvious fix — restore to `original_sp` (the address where frames were
originally captured) — doesn't work because the handler's Beagle frames
**reuse the same stack region** after the shift:

```
After shift, handler grows DOWN from prompt_sp into the captured region:

  prompt_fp ──── prompt frame (part of captured segment)
  prompt_sp ────
    handler .handle() frame        ← occupies captured region's space
    handle-io-action frame
    resume(result) call
    continuation_trampoline        ← beagle_fp (inside captured region!)
    invoke_continuation_runtime    ← actual_rsp (Rust frame)
```

The handler frame at `beagle_fp` sits INSIDE the `[original_sp,
original_sp+segment_size)` range.  Writing restored frames there corrupts
the handler's live state.  Even though the scratch stack does the writes
(avoiding Rust frame overlap), the restored continuation data clobbers the
handler's Beagle call chain, causing garbage register values when the next
`perform` calls `call_handler_builtin`.

InvocationReturnPoint cleanup (popping stale RPs before pushing new ones)
also fails because `pop_prompt_runtime` and the return path have complex
dependencies on RP ordering that aren't fully understood yet.

## How Real Runtimes Solve This

### Strategy 1: Separate Managed Stack (Chez Scheme, Racket, GHC)

The language stack is a heap-allocated buffer, completely separate from the
C/Rust stack.  Runtime code runs on the C stack and copies continuation
frames into the managed stack buffer.  No overlap is possible.

**Chez Scheme** uses segmented stack regions with underflow handlers for
lazy restoration.  Capture is O(1) — the stack segment is split at the
capture point.

### Strategy 2: Per-Prompt Stacks (OCaml 5, libmprompt)

Each delimited continuation context gets its own stack — a separate virtual
memory region.  Capture and resume are pure stack-pointer switches with zero
copying.

**libmprompt** reserves 8 MB of virtual address space per prompt but
initially commits only 4 KB of physical memory (pages committed on demand).
Guard gaps (64 KB) between stacks prevent overflow.

**OCaml 5** uses small fibers (starting at 32 words) that grow on demand.
Continuations are one-shot, so capture/resume is just pointer manipulation.

### Strategy 3: Recursive Stack Growth (Guile)

When sharing the C stack, Guile checks whether the restore target overlaps
the current execution.  If so, it recursively calls a function with a large
local array to push SP below the target, then does memcpy + longjmp.

This works because Guile's copying function is tiny — just memcpy + longjmp.
Beagle's `invoke_continuation_runtime` is too large (many Rust callees) for
this approach to work directly.

## Future Direction: Per-Prompt Stack Segments

The correct long-term fix for Beagle is per-prompt virtual memory stacks
(the libmprompt approach):

1. When `handle ... with` is entered, `mmap` a new stack segment (8 MB
   virtual, 4 KB physical, with guard page).
2. Switch SP to the new segment.
3. On `perform`/`shift`, switch SP back to the handler's segment.
4. On `resume`, switch SP to the captured segment.
5. When the handler returns, `munmap` the segment.

Benefits:
- Zero-copy capture and resume (just register saves + SP switch)
- Zero stack descent — each prompt has its own address space
- Natural overflow protection via guard pages

This requires changes to the code generator (stack segment switching in
prologues/epilogues) but would eliminate continuation overhead almost
entirely.  See libmprompt's `longjmp_arm64.S` for the ARM64 stack
switching primitives.
