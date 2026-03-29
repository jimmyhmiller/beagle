# Segmented Stack Implementation — What Went Wrong and What Needs to Change

## The Goal

Implement Chez Scheme-style segmented stacks so that the effect-handler loop (`perform` → `capture` → `handler` → `resume` → `perform` → ...) does **zero stack descent**. Currently every cycle pushes the stack down ~5KB, crashing after ~1600 IO operations. The real fix is:

- **Capture = detach a stack segment (O(1), no per-frame copying)**
- **Resume = reattach the segment (O(1), no per-frame restoration)**
- The segment's memory IS the continuation. No `CapturedFrame` heap objects.

## How Chez Scheme Does It

1. The stack is a **linked list of segments** (each a fixed-size mmap'd region).
2. Each function prologue checks if the current segment has enough space. If not, allocate a new segment and switch to it.
3. When a `reset`/`handle` block is entered, a new segment is started.
4. **Capture**: detach the segment chain from the prompt boundary to the current point. The detached segments ARE the captured continuation. No copying.
5. **Resume**: reattach the segment chain. Install an underflow handler at the boundary. When the last frame returns past the boundary, the underflow handler loads the next segment.
6. **Multi-shot**: copy the segment chain (one `memcpy` per segment).

The key property: **all execution happens on segments**. There is no "old stack" vs "new stack." Segments are the only way frames exist.

## What Beagle Does Today (the problem)

Beagle uses a single contiguous 128MB mmap'd stack per thread. The continuation machinery:

1. **Capture** (`capture_continuation_runtime`): walks the FP chain from the shift point to the prompt, allocates a `CapturedFrame` heap object for each frame, copies frame data to the heap.
2. **Invoke** (`invoke_continuation_runtime`): computes `new_sp = actual_rsp - segment_size - 4096`, writes `CapturedFrame` data back to the stack at that address, jumps to the resume point.

Every invoke descends by `segment_size + 4096` bytes. This is the linear descent that causes the crash.

## What I Tried and Why It Failed

### Attempt 1: Switch SP to a segment at handle-block entry

**Idea**: `push_prompt_runtime` allocates a segment (2MB mmap), returns its top address. JIT code does `MOV SP, X0` to switch to the segment. Body runs on the segment. `pop_prompt_runtime` switches back.

**Why it failed**: Handle blocks in Beagle are **inline code** within the enclosing function. They are not separate function calls. When we switch SP to the segment, FP still points to the enclosing function's frame on the **old stack**. If the body directly does `shift`/`perform` without an intermediate function call, `original_sp` ends up being the segment top (no frames were pushed on the segment), and the captured frames are on the old stack, not the segment.

Even when function calls DO happen within the body (the common case), the **FP chain crosses memory regions**: frames on the segment have saved-FP values pointing back to the old stack. The capture code's FP chain walk used `while fp < prompt_fp` — an address comparison that is meaningless across disjoint memory regions (a segment at 0x120000000 vs the old stack at 0x16F000000).

**What I changed**: I fixed the FP chain walk to use `fp != prompt_fp` (equality check instead of less-than). This fix is correct and committed. But it didn't solve the deeper problem.

### Attempt 2: Restore to `original_sp` (zero descent)

**Idea**: Instead of `new_sp = actual_rsp - size - 4096`, just use `new_sp = original_sp`. The captured frames go back to their original addresses. `relocation_offset = 0`. No descent.

**Why it failed**: The frames ABOVE `original_sp` (between `original_sp` and `prompt_fp`) are the captured region. But the handler was called FROM the shift body, which runs at the capture point — meaning the handler's Beagle frames and the Rust frames for `call_handler_builtin`, `continuation_trampoline`, and `invoke_continuation_runtime` are all **below `original_sp`** on the same stack. The scratch stack writes frames to `[original_sp, original_sp + size]`, which doesn't overlap with the handler (handler is below). So the write is safe.

However, when the continuation body completes, `return_from_shift_runtime` restores the `saved_frame` (the shift body's captured frame) to `beagle_fp`. `beagle_fp` is the Beagle frame that called `resume()`. With zero relocation, this frame is at its **original position** within the captured region — and the restored continuation frames are AT those same addresses. The `saved_frame` restoration **overwrites part of the restored continuation**.

Actually, on closer investigation with lldb, the specific crash was: after restoring to `original_sp` and jumping to the resume address, the resumed code **re-executes the perform** instead of continuing past it. `call_handler_builtin` is called a second time with the same arguments, and during its cleanup (String drop), it hits a corrupted pointer. The root cause appears to be that with `relocation_offset = 0`, something in the frame data causes the JIT code to take the wrong branch.

With `original_sp - 4096` (a small descent), everything works. With `original_sp - 16`, it crashes. The 4096-byte gap is needed to clear the Rust frames between the handler and the restoration region.

### Attempt 3: Pop stale InvocationReturnPoints

**Idea**: When a second invoke happens for the same `prompt_id`, pop the stale `InvocationReturnPoint` from the previous iteration and reuse its `relocated_sp`.

**Why it failed**: `InvocationReturnPoint`s are deeply coupled to the `return_from_shift_runtime` chain. That function chains through parent RPs to copy mutable ranges back to original locations. Popping the stale RP breaks this chain. Even just popping the stale RP (without reusing its location, falling back to normal descent) caused SIGBUS crashes during cleanup — the stale RP's `saved_gc_prev` field was needed by the GC frame chain, and losing it corrupted the chain.

**Key invariant discovered**: `InvocationReturnPoint`s must ONLY be popped through the normal `return_from_shift_runtime` path. Popping them anywhere else corrupts the GC chain.

## The Core Architectural Problem

Beagle's continuation system is built around **per-frame copying and relocation**. Every piece of the machinery assumes:

1. Capture creates heap objects (`CapturedFrame`) for each frame
2. Invoke writes those heap objects to a NEW stack location (different from original)
3. `relocation_offset` adjusts all FP-chain pointers for the new location
4. `InvocationReturnPoint` tracks the relocation so `return_from_shift` can copy mutable ranges back
5. The mutable-range chain links nested invocations through parent RPs

This is fundamentally incompatible with the Chez Scheme approach where segments are detached/reattached at their original addresses. To get true segmented stacks, we need to **replace** this machinery, not work around it.

## What Actually Needs to Change

### Phase 1: Make handle bodies run as segment-callable functions

The handle body must execute in its own frame on a segment. Two sub-approaches:

**Option A: Compile handle bodies as closures**
- The handle compilation emits a closure that captures the needed outer locals
- A trampoline allocates a segment, switches SP, calls the closure, switches back
- Pro: clean separation. Con: closure allocation, captured variable bookkeeping

**Option B: Emit an inline trampoline frame**
- After `push_prompt_runtime`, emit a function-call-like prologue ON the segment: `STP X29, X30, [SP, -frame_size]!; MOV X29, SP; ...`
- Save the outer FP in a callee-saved register (e.g., X28) so outer locals remain accessible
- Body code runs in this segment frame
- At `pop_prompt_runtime`, emit the matching epilogue and switch SP back
- Pro: no closure allocation. Con: need a dedicated register for outer FP

**Option C: Accept that the handle function's own frame is on the old stack**
- Don't create a trampoline. Just switch SP.
- All CALLS within the body go on the segment. The body's own inline code uses FP on the old stack.
- Capture only detaches segment frames (function calls within the body), not the handle function's frame.
- The handle function's frame is never captured (it's on the old stack, above the prompt).
- Pro: simplest. Con: the handle function's own locals (modified between performs) need special handling.

Option C is actually the Chez approach for the "current frame at prompt boundary" — it stays on the old segment. Only frames BELOW (called from the body) are on the new segment and get detached.

### Phase 2: Replace capture with segment detachment

Instead of walking frames and allocating `CapturedFrame`s:
1. Pop the segment from the active stack
2. Record the current SP/FP within the segment
3. Store the segment pointer in the `ContinuationObject`
4. Switch SP back to the parent (the handle function's old SP)

O(1), no allocation, no GC pressure.

### Phase 3: Replace invoke with segment reattachment

Instead of writing `CapturedFrame`s to a new stack location:
1. Push the segment back onto the active stack
2. Set SP to the saved position within the segment
3. Write the resumed value to the result local
4. Jump to the resume address

O(1), no frame copying.

### Phase 4: GC integration for captured segments

Captured segments contain live Beagle frames with heap pointers. GC must scan them:
- Walk the FP chain within each captured segment
- Trace locals in each frame (same as current GC frame scanning)
- Register captured segments in `PerThreadData` so GC can find them

### Phase 5: Multi-shot via segment copying

First invoke uses the original segment. Second+ invoke copies the segment (`memcpy` + FP chain relocation within the copy).

### Phase 6: Underflow handling

When the last frame on a segment returns, execution crosses back to the parent segment. The `continuation_return_trampoline` (set as LR during invoke) already handles this for the current system. For segmented stacks, we need to ensure SP is switched back to the parent segment when this happens.

### Phase 7: Remove old machinery

Delete `CapturedFrame`, `InvocationReturnPoint` relocation fields, `mutable_ranges`, scratch stack continuation code, `SafeInvokeContext`, `SafeReturnContext`.

## What's Already Done (Committed)

- `StackSegment` struct with mmap allocation and guard page (`src/runtime.rs`)
- Segment pool and `active_segments`/`captured_segments` in `PerThreadData`
- `stack_pointer_reg()` on `CodegenBackend` trait (returns SP on ARM64, RSP on x86-64)
- FP chain walk fixed to use equality boundary check (`fp != prompt_fp`)
- `push_prompt_runtime` and `pop_prompt_runtime` return SP (currently no-op — returns same SP)
- `effective_root` flag for future stale RP handling
- All 319 tests pass

## Int Tagging Detail (for future reference)

Stack addresses stored in `ContinuationObject` are tagged as Ints: `value << 3 | 0b000`. Untagging: `value >> 3`. This is lossless for 64-bit addresses (top 3 bits are always 0 for userspace addresses on ARM64). This means `continuation.original_sp()` correctly recovers the original address.
