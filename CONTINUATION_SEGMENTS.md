# Continuation Segment Redesign: Current Status

## Summary

Beagle continuations no longer keep captured stack segments in a Rust-side `HashMap` or clone a full mmap segment on every invoke.

The live design is:

- Capture copies only the occupied stack bytes into an **opaque heap object**.
- The `ContinuationObject` stores a tagged heap pointer to that segment object.
- Invoke copies those bytes back into a reusable execution mmap segment.
- GC traces and updates both the `ContinuationObject` and the heap pointers embedded inside the captured segment.

This is the active implementation in the code today.

## What changed

The original redesign fixed the big memory blow-up by replacing per-continuation 2MB mmaps with heap-backed captured segments.

A second bug existed after that redesign:

- If GC ran after the segment heap object was allocated but before the `ContinuationObject` was created, the pending segment could move.
- The captured frame-link pointers inside the segment still pointed at the old heap location.
- We then stored the moved base as `segment_original_data_base`, so later relocation became a no-op.
- Symptoms: `--gc-always` async/effect programs returned after the first resume instead of continuing through caller frames.

That is now fixed by:

- Recording the **capture-time segment data base** immediately when the segment heap object is created.
- Carrying that base through `pending_heap_segments`.
- Keeping the capture-time base until the `ContinuationObject` is created, so later relocation still has the correct source base.
- Making `segment_info()` relocate frame links by walking the **caller-FP chain**, which is more robust than advancing only through the GC-prev chain.

This session also fixed a separate GC-sensitive async/socket bug:

- TCP async operations were using raw heap atom addresses as result keys.
- Under moving GC, those addresses could change while the I/O op was pending.
- The socket/effect path now uses stable TCP op IDs instead.

And it tightened the pre-attachment GC window:

- Pending heap segments are now scanned conservatively as raw words during the short window before `ContinuationObject` creation.
- That avoids depending on detached-frame metadata before the continuation is fully attached and normalized.

## Verified State

These were re-checked in this session on Linux x86-64:

- `cargo test test_run_async_future_basic_gc_always -- --nocapture` passes.
- `cargo test test_run_event_loop_handler_gc_always -- --nocapture` passes.
- `target/debug/beag run --gc-always resources/async_multi_effect_test.bg` passes.
- `target/debug/beag run --gc-always resources/async_test.bg` passes.
- `target/debug/beag run --gc-always resources/gc_frame_chain_multishot_test.bg` passes.
- `target/debug/beag run resources/gc_continuation_stress_test.bg` passes.
- `cargo test continuation_segments_are_heap_objects_and_gc_reclaimable -- --nocapture` passes.
- A minimal socket continuation repro for the old `socket/accept` / `socket/read` `--gc-always` hang now passes.

Memory check:

- `/usr/bin/time -v target/release/beag run resources/gc_continuation_stress_test.bg`
- Max RSS: **46224 KB** (~45 MB)

That is consistent with the intended “no crazy memory use” behavior.

## What Is Now True

### 1. Continuations are not using crazy amounts of memory

Yes.

- Captured segments are sized to the occupied stack bytes only.
- The release stress run above stayed around 45 MB RSS.

### 2. Continuations are heap objects

Yes.

- `capture_continuation_runtime_inner` allocates the captured segment with `runtime.allocate(..., BuiltInTypes::HeapObject)`.
- `ContinuationObject::FIELD_SEGMENT_PTR` stores the tagged heap pointer directly.
- The regression test `continuation_segments_are_heap_objects_and_gc_reclaimable` verifies that live continuation segment pointers are heap pointers and that the segment objects are opaque heap objects.

### 3. Continuations are GC’d like normal objects

Yes, for the verified x86-64 generational path covered by the tests above.

- The same regression test roots a captured continuation in a global atom, confirms it is live, clears the root, forces GC twice, and verifies no continuation segment remains.
- GC backends also scan pending heap segments during the allocation window before `ContinuationObject` creation.
- There is still one broader `--gc-always` socket stress failure; see “Known Remaining Issue” below.

## Architecture Notes

### Capture path

`src/builtins.rs`

1. Pop the active prompt-owned mmap segment.
2. Compute the occupied stack size.
3. Allocate an opaque heap object for the captured bytes.
4. Copy the used stack bytes into the heap object.
5. Relocate interior frame links from mmap addresses to heap-object addresses.
6. Record the heap data base immediately.
7. Push `(segment_ptr, gc_frame_offset, segment_size, original_data_base)` into `pending_heap_segments`.
8. Allocate and initialize the `ContinuationObject`.
9. Store the capture-time `original_data_base` in the continuation.

### Invoke path

`src/builtins.rs`

1. Read normalized segment info from `ContinuationObject::segment_info()`.
2. Allocate an execution mmap segment.
3. Copy the captured bytes into that segment.
4. Relocate interior frame links from heap addresses to execution-segment addresses.
5. Patch the outermost resumed frame to return through `continuation-return-stub`.
6. Reinstall prompt/exception metadata and jump to the resume address.

### GC interaction

- `ContinuationObject` traces `segment_ptr` as a normal heap field.
- The segment object is opaque, so GC does not blindly scan its raw bytes.
- GC walks the captured segment explicitly through `walk_segment_gc_roots`.
- Pending segments are scanned conservatively before they are attached to a `ContinuationObject`.

## Code Pointers

- `src/builtins.rs`
  - `capture_continuation_runtime_inner`
  - `invoke_segmented_continuation`
- `src/runtime.rs`
  - `ContinuationObject`
  - `segment_info`
  - `relocate_segment_frame_links`
  - `pending_heap_segments`
- `src/gc/generational.rs`
  - pending-segment conservative scanning
  - regression test `continuation_segments_are_heap_objects_and_gc_reclaimable`
- `src/gc/mark_and_sweep.rs`
  - pending-segment conservative scanning
- `src/gc/compacting.rs`
  - pending-segment conservative scanning
- `src/types.rs`
  - `num_traced_slots` for `TYPE_ID_CONTINUATION`

## Known Remaining Issue

This session re-ran the old x86-64 `--gc-always` socket stress case and it is still not fully clean:

- `target/debug/beag run --gc-always resources/continuation_stack_stress_test.bg` still fails.
- The previously observed immediate `socket/accept` hang is fixed, but a later continuation/GC interaction in that stress workload still trips a mark-and-sweep allocation assertion.
- So the continuation segment redesign is verified for the targeted async/effect and heap/GC tests above, but the old stack/socket stress workload is not fully closed yet.

## Still Not Revalidated Here

These items were not re-run in this session:

- macOS ARM64 `resources/closure_protocol_test.bg`
- the old intermittent x86-64 mark-and-sweep debug multishot report

## Old Infrastructure Still Present

The legacy HashMap-based captured-segment machinery is still in the tree, but the active continuation path no longer uses it:

- `captured_segments`
- `prompt_captured_segments`
- `pending_captured_segment_handles`
- `record_captured_segment`
- `find_captured_segment_bounds`
- `find_captured_segment_info`
- `recycle_unreachable_captured_segments`

The new regression test asserts that `captured_segments` and `pending_captured_segment_handles` stay empty on the live path.
