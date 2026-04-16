# Generational GC Bug: Zero-Field Structs + Continuations

**Filed:** 2026-04-15
**Affects:** 7 tests under `// gc-always` with the default generational GC
**Does NOT affect:** mark-and-sweep GC (`--features mark-and-sweep`)

---

## Minimal repro

```beagle
namespace repro
// gc-always

use beagle.effect as effect

enum C { X }

struct H {}

extend H with effect/Handler(C) {
    fn handle(self, op, resume) { resume(1) }
}

fn f(n) {
    if n <= 0 { 0 }
    else {
        let h = H {}
        let r = handle effect/Handler(C) with h {
            let x = perform C.X
            x * n
        }
        r + f(n - 1)
    }
}

fn main() {
    println(f(3))
}
```

**Expected:** prints `6` (3 + 2 + 1)
**Actual:** prints nothing, exits 0 (first iteration works, second iteration's `resume(1)` never returns — the trampoline teleports correctly but the return path exits the program instead of returning to the handler)

## What makes it trigger

ALL of these must be true:

| Condition | Change it and... |
|---|---|
| `struct H {}` (zero fields, 8-byte object) | Change to `struct H { _pad }` → **passes** |
| `// gc-always` (GC on every allocation) | Remove → **passes** |
| Handle block inlined in the recursive function | Move handle into a separate `fn do_handle(n)` called from the recursion → **passes** |
| Generational GC (default) | Switch to `--features mark-and-sweep` → **passes** |

Sequential calls to the same handle block (no recursion) also pass. The bug requires the *combination* of all four conditions.

## What the investigation found

### The crash site

lldb shows `EXC_BAD_ACCESS` deep in JIT code — `ldur x20, [x19, #0x8]` with `x19 = 1`. An integer value `1` is being treated as a heap pointer.

Protocol dispatch instrumentation confirmed: the handler pointer arriving at `perform_dispatch_and_return_runtime` has the correct `struct_id=188` for both iterations. The `save_volatile_registers3` call enters the handler correctly. But **it never returns** on the second iteration.

Inside the handler, `resume(1)` is called. The continuation trampoline fires (confirmed by logging). The resumed body runs. But the return path from the body back through the handler frame chain goes somewhere wrong, and the program silently exits with code 0.

### Why zero-field structs specifically

A zero-field struct is exactly 8 bytes (header only). When the generational GC promotes it from young gen to old gen:

1. The object's 8 bytes are copied to old gen
2. A forwarding pointer is written at the old young-gen location — this **overwrites the entire object** (since the forwarding pointer IS 8 bytes and the object IS 8 bytes)
3. Young gen is cleared (`allocation_offset = 0`, memory NOT zeroed)
4. Next allocation reuses the same young-gen address, writing a new header over the forwarding pointer

For 16-byte objects (1+ fields), the forwarding pointer overwrites only the header; the field data remains and there's more room for the GC to work correctly.

### What's NOT the cause

- **Stale virtual registers:** We added early-store-to-local + late-reload-from-local for handler/op/enum_id values across GC-triggering allocations. Correct fix for a real class of bugs, but doesn't fix this specific issue.
- **GC prev chain corruption:** The `GC_FRAME_TOP` truncation in `return_from_shift_runtime_inner` is correct and tested.
- **Continuation segment scanning:** `scan_continuation_segment` does run during promotion and updates young-gen pointers inside the segment. Segment heap pointers were validated at trampoline time — no forwarding bits set, all headers valid.
- **Deep-handler wrapper:** Same failure with `BEAGLE_NO_DEEP_WRAP=1` (raw trampoline closure, no wrapper).

### Proof that it's the generational copier

```bash
# Fails (generational GC, default):
cargo run --release -- run --gc-always /tmp/repro.bg
# exits 0, no output

# Passes (mark-and-sweep, non-moving GC):
cargo run --release --features mark-and-sweep -- run --gc-always /tmp/repro.bg
# prints: 6

# Passes (generational but struct has a field):
# change `struct H {}` to `struct H { _pad }` and `H {}` to `H { _pad: 0 }`
cargo run --release -- run --gc-always /tmp/repro_with_field.bg
# prints: 6
```

### Workaround that confirms the size theory

Adding `let size = size.max(1);` in `src/builtins/allocation.rs` (forcing all allocations to have at least 1 word of payload = 16 bytes minimum) makes the zero-field test pass. But it regresses ~40 other tests because the header's `size` field (0) no longer matches the actual allocation size (1), confusing `get_fields_mut`, `num_traced_slots`, and the GC scanner.

## Where to look

The bug is in the interaction between:

1. **`src/gc/generational.rs` — `copy()` (line ~1002):** Copies the object, writes forwarding pointer. For 8-byte objects, the forwarding pointer replaces the entire object contents.

2. **`src/gc/generational.rs` — `minor_gc()` / `full_gc()`:** Under `gc_always`, `full_gc` runs on every allocation. Young gen is cleared at line 830 (`self.young.clear()`), which just resets `allocation_offset` to 0 without zeroing memory.

3. **`src/builtins/reset_shift.rs` — `continuation_trampoline` (line ~916):** Reads the continuation segment and copies it back onto the stack. If any pointer in the segment or in a closure referenced by the segment is stale, the restored stack has bad data.

### Hypotheses to test

**H1: Young-gen address reuse after clear.** After minor GC promotes objects and clears young gen, the next allocation at offset 0 writes a new header at the same address where a forwarding pointer used to be. If any code path reads the old address expecting to find the forwarding pointer (to follow it to old gen), it instead finds the new object's header. The forwarding pointer is gone.

To test: zero the young gen memory in `clear()` and see if behavior changes (it should panic at a different point if something reads from a cleared address).

**H2: Inline cache staleness.** The protocol dispatcher's inline cache stores `(type_id, fn_ptr)` pairs. After GC moves an object, the cache entry might match the wrong object (one that happened to land at the same address). For 8-byte objects, this is more likely because they're more densely packed.

To test: disable inline caching (always fall through to `protocol_dispatch` slow path) and see if the test passes.

**H3: Frame header confusion.** A zero-field struct's header (`type_id=0, type_data=struct_id, size=0`) could be confused with a frame header or other internal object during GC scanning, especially if adjacent memory contains frame-like data.

To test: set `type_id` to a unique value (e.g., `TYPE_ID_EMPTY_STRUCT = 254`) for zero-field structs so they can't be confused with frames.

## Affected tests

```
resources/gc_frame_chain_continuation_test.bg   (signal 11)
resources/gc_frame_chain_multishot_test.bg      (signal 6)
resources/gc_frame_chain_nested_handler_test.bg (depends on prev-chain fix variant)
resources/gc_handler_minimal_test.bg            (was fixed by handler root-id change)
resources/gc_shift_reset_basic_test.bg          (pre-existing, different crash)
resources/continuation_stack_stress_test.bg     (signal 11)
resources/dynamic_var_continuation_test.bg      (semantic: wrong value, not crash)
```

The non-GC failures (`async_ambient_thread_test`, `concurrent_socket_echo_test`, `repl_namespace_integration_test`) are unrelated — they're about threading and socket setup.

## Files

- `src/gc/generational.rs` — copier, promotion, `copy()`, `minor_gc`, `full_gc`
- `src/gc/stack_walker.rs` — frame scanning, `walk_segment_gc_roots`
- `src/builtins/reset_shift.rs` — continuation capture/restore/trampoline
- `src/builtins/allocation.rs` — struct allocation entry point
- `src/types.rs` — `Header`, `HeapObject`, `fields_size`, `num_traced_slots`
