# GC Continuation Segment Corruption Bug

## Summary

Root cause: effect handlers were stored in a Rust thread-local `HandlerStack` that the GC never traced. Active handlers could be collected and their memory reused (often by continuation segments), so the handler pointer in the stack pointed at a moved/overwritten object, producing the corrupted header.

Fix: register handler stack entries as GC handle roots (using GlobalObjectBlocks), remove those roots when popping/clearing handlers, and read handler pointers via the updated root slot so moving GCs rewrite them. With this change, the minimal/stress repro now runs cleanly.

## Reproduction

**Test Case**: `resources/gc_continuation_minimal_repro.bg` or `resources/gc_continuation_stress_test.bg`

**Configuration**:
- Features: `compacting,thread-safe,backend-x86-64`
- Build: Release mode
- Command: `./target/release/main resources/gc_continuation_stress_test.bg`

**Success Criteria**:
- < 60 iterations: Usually passes
- 60-500 iterations: May pass or fail (nondeterministic threshold)
- 500 iterations: Consistently fails

## Symptoms

The program crashes with the following error:

```
=====================
Header { type_id: 0, type_data: 61334132, size: 57377, opaque: true, marked: false, large: true } 0x...
=====================
Uncaught exception:
SystemError.RuntimeError { message: "Handler type does not implement beagle.effect/Handler<...> protocol. Handler: ErrorOpaque", location: null }
```

The corrupted header shows:
- `type_id: 0` (struct type, not the expected handler struct ID)
- `opaque: true` (should be false for structs)
- `large: true` and `size: ~57377 words` (matches continuation segment sizes)
- `type_data`: Contains a large number that doesn't match expected struct metadata

This header pattern exactly matches how continuation segments are allocated (see `src/builtins.rs:4905-4916`), but with `type_id` incorrectly set to 0 instead of `TYPE_ID_CONTINUATION_SEGMENT` (32).

## Analysis

### What We Know

1. **Timing**: Corruption occurs after multiple GC cycles (typically after processing ~500 handler invocations)

2. **Affected Objects**: The YieldHandler struct instance gets its header corrupted

3. **Corruption Pattern**: The corrupted header looks like a continuation segment header:
   - Large opaque object (~57377 words = ~459KB)
   - But `type_id` is 0 (struct) instead of 32 (continuation segment)

4. **GC Processing**: Debug output shows:
   ```
   [GC-DEBUG] Processing continuation object at 0x..., segment_ptr=0x...
   [GC-DEBUG] GC'ing continuation segment: len=320 bytes
   [GC-DEBUG] Found continuation segment in to_space: type_id=32, size=40 words, opaque=true, large=false, at 0x...
   ```

### Code Locations

**Continuation Segment Creation** (`src/builtins.rs:4897-4920`):
```rust
// Allocate an opaque heap buffer for the stack segment
let segment_words = stack_size.div_ceil(8);
let segment_ptr = runtime
    .allocate(segment_words, stack_pointer, BuiltInTypes::HeapObject)
    .expect("Failed to allocate continuation segment");

let mut segment_obj = HeapObject::from_tagged(segment_ptr);
let is_large = segment_words > Header::MAX_INLINE_SIZE;
segment_obj.writer_header_direct(Header {
    type_id: TYPE_ID_CONTINUATION_SEGMENT,  // = 32
    type_data: stack_size as u32,
    size: if is_large { 0xFFFF } else { segment_words as u16 },
    opaque: true,
    marked: false,
    large: is_large,
});
```

**GC Continuation Processing** (`src/gc/compacting.rs:286-302`):
```rust
if object.get_type_id() == TYPE_ID_CONTINUATION as usize {
    let obj_ptr = object.get_pointer();
    if let Some(cont) = ContinuationObject::from_heap_object(object) {
        cont.with_segment_bytes_mut(|segment| {
            if segment.is_empty() {
                return;
            }
            self.gc_continuation_segment(
                segment,
                cont.original_sp(),
                cont.original_fp(),
                cont.prompt_stack_pointer(),
                stack_map,
            );
        });
    }
}
```

**Segment GC Processing** (`src/gc/compacting.rs:330-367`):
- Walks the continuation segment bytes
- Identifies heap pointers within the captured stack
- Copies referenced objects using Cheney's algorithm
- Updates pointers in the segment

### Hypotheses

1. **Buffer Overflow**: The continuation segment GC walker might be writing beyond the segment bounds, corrupting adjacent heap objects

2. **Incorrect Size Calculation**: `get_full_object_data()` or `fields_size()` might return incorrect sizes for large opaque objects, causing too much data to be copied

3. **Alignment Issue**: Object allocation or copying might have alignment problems causing objects to overlap

4. **Header Overwrite During Copy**: The `copy_using_cheneys_algorithm` or `copy_data_to_offset` might incorrectly write headers when copying continuation segments

5. **Type Confusion**: The GC might be treating a continuation segment as a regular object (or vice versa) at some point

### What Doesn't Explain It

- **Not a simple use-after-free**: The corrupted header has a specific pattern (continuation segment-like)
- **Not random memory corruption**: The corruption is consistent and reproducible
- **Not a missing GC root**: The handler is properly rooted (it's in the prompt handler stack)

## Next Steps for Investigation

1. **Add bounds checking**: Instrument `gc_continuation_segment` to verify all writes stay within segment bounds

2. **Track object lifecycle**: Add logging for:
   - When the handler struct is allocated
   - When continuation segments are allocated
   - When each is copied during GC
   - The memory addresses and sizes involved

3. **Verify size calculations**: Double-check that `full_size()`, `fields_size()`, and `get_opaque_bytes_len()` return correct values for:
   - Large opaque objects (continuation segments)
   - Small opaque objects
   - Regular struct objects

4. **Check segment pointer updates**: Verify that when a continuation object is copied, its `segment_ptr` field is correctly updated to point to the copied segment

5. **Memory dump comparison**: Compare heap state before and after a GC that causes corruption

## Workarounds

None currently known. The test consistently fails with the compacting GC. Other GC strategies (mark-and-sweep) may not exhibit this bug but haven't been tested.

## Related Files

- `src/builtins.rs` - Continuation creation (`capture_continuation_runtime`)
- `src/gc/compacting.rs` - Compacting GC implementation
- `src/gc/continuation_walker.rs` - Continuation segment walking
- `src/runtime.rs` - `ContinuationObject` wrapper
- `src/types.rs` - `HeapObject` and `Header` structures
- `resources/gc_continuation_stress_test.bg` - Original failing test
- `resources/gc_continuation_minimal_repro.bg` - Minimal reproduction

## Status

**Resolved** - Handler stack entries are now GC roots, preventing handler objects from being collected or overwritten. Minimal/stress repros complete without corruption.
