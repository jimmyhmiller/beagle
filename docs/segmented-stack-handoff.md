# Segmented Stack Implementation — Context Handoff

## What You're Doing

Implementing Chez Scheme-style segmented stacks for Beagle's delimited continuations. This eliminates the stack descent bug where every `perform` (IO operation) pushes the stack down ~5KB, crashing after ~1600 operations.

## The Problem

Every `socket/read`, `socket/write`, `core/sleep` goes through the effect system:
```
perform Async.IO { action }
  → shift (capture continuation — copies frames to heap)
  → handler does IO
  → handler calls resume(result)
  → invoke_continuation_runtime (restores frames to stack, jumps)
  → next perform → cycle repeats
```

Currently, each `invoke_continuation_runtime` places restored frames BELOW the current SP with a 4KB safety margin. Linear descent. A byte-by-byte TCP read of 150 bytes = 150 continuation invocations = ~750KB descent.

## The Solution: Segmented Stacks

Instead of one contiguous stack, use a linked list of stack segments. Each segment is a fixed-size region (e.g., 64KB). When entering a `handle` block, start a new segment.

**Capture (shift/perform)**: Detach the current segment chain from the prompt boundary back to the current point. O(1) — just pointer manipulation. The detached segments ARE the captured continuation. No CapturedFrame heap copying needed.

**Resume**: Reattach the segment chain. Install an underflow handler at the segment boundary. When execution returns past the boundary, the underflow handler loads the next segment. Bounded copy per step.

## Key Architecture Facts About Beagle

Read these files to understand the current system:

### Stack Frame Layout (ARM64)
```
[FP + 8]   Saved LR (return address)
[FP + 0]   Saved FP (caller's frame pointer)
[FP - 8]   Frame header (type_id=37, num_slots for GC)
[FP - 16]  GC prev pointer (frame chain linkage)
[FP - 24]  Local 0
[FP - 32]  Local 1
...
[SP]       Callee-saved registers (x19-x28) at bottom
```

### Current Continuation Machinery (to be replaced)
- `src/builtins.rs`:
  - `capture_continuation_runtime` (~line 11674): Walks FP chain, copies each frame to heap CapturedFrame objects
  - `invoke_continuation_runtime` (~line 12668): Restores CapturedFrame objects back to stack, jumps
  - `continue_invoke_on_safe_stack` (~line 12174): Does the actual frame writes from a scratch stack
  - `return_from_shift_runtime` (~line 12345): Handles continuation return, copies mutable ranges back
  - `pop_prompt_runtime` (~line 11565): Handle block epilogue
  - `call_handler_builtin` (~line 16557): Dispatches to effect handler
- `src/runtime.rs`:
  - `CapturedFrame`: Heap object storing one stack frame's data
  - `ContinuationObject`: Heap object with segment chain + metadata (17 fields)
  - `InvocationReturnPoint`: Tracks relocation for multi-shot support
  - `PromptHandler`: Pushed on handle entry, popped on shift
  - `PerThreadData`: Thread-local prompt/exception/RP stacks

### Thread Stacks
- Each thread gets a 128MB mmap'd stack with a guard page at the bottom
- Created in `runtime.rs` (`create_stack_with_protected_page_after`)
- Stack grows DOWN (ARM64)

### Code Generation
- `src/arm.rs`: ARM64 instruction encoding
  - Function prologue: `STP x29,x30,[sp,-N]!` → `MOV x29,sp` → zero locals → save callee-saved
  - Function epilogue: Restore callee-saved → `LDP x29,x30,[sp],N` → `RET`
- `src/ir.rs`: IR instructions including `PushPromptHandler`, `CaptureContinuation`, `ReturnFromShift`
- `src/ast.rs`: `Ast::Reset`/`Ast::Handle` compilation, `Ast::Perform`/`Ast::Shift` compilation

### GC
- Walks `GC_FRAME_TOP` thread-local chain (linked via [FP-16] prev pointers)
- Each frame header at [FP-8] has `num_slots` telling GC how many locals to scan
- Callee-saved registers at frame bottom are NOT scanned (they're raw values)

### Effect System (Beagle-side)
- `standard-library/beagle.socket.bg`: Every socket op uses `perform async/Async.IO { ... }`
- `standard-library/beagle.async.bg`: `ImplicitAsyncHandler.handle(self, op, resume)` calls `resume(result)` after doing IO

## The Chez Scheme Approach — How It Works

Reference: Hieb, Dybvig, Bruggeman 1990 "Representing Control in the Presence of First-Class Continuations"

1. **Stack = linked list of segments**. Each segment is a contiguous region with a fixed max size.
2. **Segment overflow check in prologue**: If current frame won't fit, allocate new segment, link it, switch SP.
3. **Continuation capture at prompt**: Detach segments from prompt to current. The detached chain IS the continuation. No per-frame copying.
4. **Continuation invoke**: Attach the saved segment chain. Install underflow handler at the join point. When the last frame in the segment returns, the underflow handler triggers, loading the next segment.
5. **Multi-shot**: Each shot gets a fresh copy of the segment chain (one memcpy per segment per shot).

## The Hard Part: Rust FFI Interleaving

Beagle's JIT code and Rust builtins share the same native stack. A typical call chain looks like:
```
[Beagle frame: main()]
[Beagle frame: game-loop()]
[Rust frame: property_access builtin]
[Beagle frame: socket/read]
[Rust frame: capture_continuation_runtime]
```

Chez Scheme doesn't have this problem — Scheme code runs on a separate managed stack. For Beagle, segment boundaries must account for Rust frames embedded in the Beagle call chain.

**Possible solutions**:
1. Treat Rust builtin calls as opaque — the segment containing a Rust frame cannot be split at the Rust boundary. Segments are split only at Beagle frame boundaries.
2. Save/restore Rust frames as raw byte ranges (no GC scanning needed for Rust data).
3. Ensure Rust builtins are called via a trampoline that switches to a separate "C stack" for Rust execution (like OCaml's approach). This fully isolates Rust frames from Beagle segments.

Option 3 is cleanest but most invasive. Option 1 is most practical for a first implementation.

## Current State of the Code

The 4KB safety margin fix (reduced from 1MB) is committed and working. All 319 tests pass. The REPL server works for typical payloads. This is the baseline to build on.

Recent commits:
- `property_access` fix: Added `save_gc_context!` so FieldError resume works
- GC frame chain unwinding in `throw_exception`: Prevents GC crashes after exception
- 4KB margin + scratch stack: `invoke_continuation_runtime` does frame restoration from a separate heap-allocated scratch stack

## Test Files to Verify Against
- `resources/continuation_stack_stress_test.bg` — Byte-by-byte reads, multi-line reads, many writes, sleep loops
- `resources/repl_main_resume_test.bg` — Resume from FieldError
- `resources/repl_resume_test.bg` — Session-level resume/abort
- `resources/resumable_exception_test.bg` — Basic resumable exceptions
- `resources/resumable_multi_throw_test.bg` — Multi-throw after resume
- All 319 tests in `resources/` must continue to pass

## Suggested Implementation Order

1. **Design the segment data structure**: Size, header format, linking mechanism, overflow check
2. **Implement segment allocator**: Pool of segments, allocation, recycling
3. **Modify function prologue**: Add segment overflow check (conditional branch to slow path that allocates new segment)
4. **Modify handle block**: Start a new segment on handle entry
5. **Replace capture_continuation_runtime**: Instead of CapturedFrame copying, detach segment chain
6. **Replace invoke_continuation_runtime**: Reattach segment chain, install underflow handler
7. **Implement underflow handler**: When a segment's last frame returns, load next segment
8. **GC integration**: Walk segment chains during GC
9. **Multi-shot support**: Copy segment chain on non-first invocation
10. **Remove old machinery**: CapturedFrame, InvocationReturnPoint, relocation, scratch stack

Each step should be independently testable. Run the full 319-test suite after each step.
