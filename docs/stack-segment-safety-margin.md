# Stack Segment Safety Margin: A Necessary Hack

## The Problem

When invoking a captured continuation with a non-empty stack segment, we need to copy that segment to a new location in memory before jumping to it. The naive approach is to place this copy just below the current Rust stack pointer (RSP) with a small safety margin:

```rust
let new_sp = actual_rsp - stack_segment_size - 256; // 256 byte safety margin
```

This causes **silent data corruption** when the handle block contains multiple sequential `perform` calls.

## What Goes Wrong

Consider this Beagle code:

```
handle {
    let r1 = perform Ask.Get {}
    println("r1 = " ++ int_to_string(r1))
    let r2 = perform Ask.Get {}
    println("r2 = " ++ int_to_string(r2))
    r1 + r2  // Returns garbage!
}
```

The second `perform` captures a stack segment that includes the memory location where `r1` is stored. When we invoke that continuation:

1. We copy the stack segment (containing `r1`'s value) to `new_sp`
2. The value is correctly written - we verified this with debug output
3. We call `eprintln!` or do other Rust operations for debugging/runtime work
4. **Rust's stack grows downward from RSP, overwriting our copied data**
5. When Beagle code resumes and accesses `r1`, it reads garbage

The issue is that the copied stack segment extends from `new_sp` up to `new_sp + stack_segment_size`. If this range overlaps with where Rust is actively using the stack, we get corruption.

Example addresses from a debug session:
```
actual_rsp = 0x7c7dd36b37e0
new_sp     = 0x7c7dd36b2240
segment_end = new_sp + 5280 = 0x7c7dd36b36e0

r1 location in copy = 0x7c7dd36b36b8
```

Notice that `r1`'s location (0x7c7dd36b36b8) is only 296 bytes below `actual_rsp`. A single `eprintln!` call can easily use more stack than that, corrupting `r1`.

## The Fix

```rust
// IMPORTANT: Need a LARGE margin because Rust code (eprintln, etc.) uses the stack
// and the stack segment includes high addresses that could overlap with Rust's stack.
// The margin needs to be at least as large as the stack segment to avoid overlap.
let safety_margin = stack_segment_size.max(4096) + 4096; // At least segment size + 4KB extra
let new_sp = actual_rsp - stack_segment_size - safety_margin;
```

By making the safety margin at least as large as the stack segment itself (plus extra buffer), we ensure that even the highest addresses in our copied segment are well below where Rust might write.

## Why This Is A Hack

This is fundamentally a hack because:

1. **We're guessing at Rust's stack usage.** We don't actually know how much stack space Rust will use before we switch to the continuation. The 4KB buffer is arbitrary.

2. **We're fighting against the runtime.** The continuation machinery is implemented in Rust, but we need to place data in a location that Rust's stack won't touch. This is an impedance mismatch.

3. **It wastes address space.** We're leaving large gaps of unused memory between the Rust stack and our copied segments.

4. **It doesn't compose well.** Deeply nested continuations will push the stack ever lower, potentially causing issues with stack limits.

## Better Solutions (Future Work)

1. **Separate stack regions**: Allocate continuation stack segments from a different memory region entirely, not below RSP. This would require changes to how we set up RSP/RBP when jumping to continuations.

2. **Copy-on-switch**: Defer the stack copy until the very last moment before the assembly jump, minimizing the window where Rust code can corrupt the data.

3. **mmap'd stacks**: Use `mmap` to allocate dedicated stack regions for continuations, similar to how green threads or coroutines handle this.

4. **Compiler support**: Have the Beagle compiler track which variables are live across continuation boundaries and save/restore them explicitly rather than relying on stack copying.

## Conclusion

The safety margin hack works, but it's a band-aid over a fundamental architectural tension: we're implementing delimited continuations (which need precise control over stack memory) inside a Rust runtime (which assumes it owns the stack). A production-quality implementation would need a more principled solution.
