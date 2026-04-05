# Plan: Replace Save-Around-Calls with Permanent Root Slots

## Goal

Replace Beagle's current GC root strategy (save live registers to locals before calls, restore after, use stack maps to tell GC how many slots are live) with a simpler model: every virtual register gets a permanent frame slot ("root slot"). Values are always written to their root slot. After any GC safepoint, values are reloaded from root slots. The GC scans all root slots in a frame — no stack maps needed.

## Key Insight: Root Slot Reuse via Liveness

To avoid an explosion of root slots, virtual registers whose lifetimes don't overlap share a root slot. The linear scan allocator already computes `lifetimes: HashMap<VirtualRegister, (usize, usize)>` — the same data structure drives root slot assignment. When a virtual register's lifetime ends, its root slot returns to a free pool. The number of root slots per function = **max simultaneously live virtual registers**, not total virtual registers.

---

## Current Architecture (what exists today)

1. **Linear scan** allocates physical registers (callee-saved: X19-X28 on ARM, R12-R15+RBX on x86)
2. When registers spill, they go to stack slots starting at index `num_locals` — these are **not GC-traced**
3. Before each call, `replace_calls_with_call_with_save()` identifies live registers and emits `CallWithSaves` which stores them to freshly-allocated locals, calls, then restores
4. The GC uses **stack maps** (return address → `StackMapDetails`) to know how many slots to scan per frame at each safepoint
5. Frame header encodes `size` (number of scannable slots) in the header word

## New Architecture (what we're building)

1. **Every virtual register gets a root slot** (a frame local index), assigned during register allocation using liveness-based reuse
2. **Every write to a physical register also stores to the root slot**: `add x19, x20, x21; str x19, [fp, #root_slot_offset]`
3. **After any safepoint** (call, allocation), reload live registers from their root slots: `ldr x19, [fp, #root_slot_offset]`
4. **No more WithSaves variants** — plain `Call` handles everything because roots are always in the frame
5. **No more stack maps** — frame header encodes `num_roots` directly; GC scans that many slots unconditionally
6. **No more spill slots** — if we run out of physical registers, the "spilled" register just lives in its root slot and gets loaded into a temporary register on each use (which is what `Value::Spill` already does via `load_local`/`store_local`)

---

## Implementation Steps

### Step 1: Add root slot assignment to `LinearScan`

**File: `src/register_allocation/linear_scan.rs`**

Add a new field `root_slots: HashMap<VirtualRegister, usize>` and a free-list for root slots. During `allocate()`, when a virtual register's interval starts, assign it a root slot (reuse one from a dead register if available, otherwise bump a counter). When its interval expires, return the root slot to the free pool.

```rust
pub struct LinearScan {
    // ... existing fields ...
    pub root_slots: HashMap<VirtualRegister, usize>,
    free_root_slots: Vec<usize>,
    next_root_slot: usize, // starts at num_locals (to preserve existing locals)
}
```

In `expire_old_intervals()`, when freeing a register, also free its root slot:
```rust
// existing: self.free_register(register_to_free);
// add: 
if let Some(slot) = self.root_slots.get(&j.2) {
    self.free_root_slots.push(*slot);
}
```

In the main allocation loop, assign root slot alongside physical register:
```rust
let root_slot = self.free_root_slots.pop().unwrap_or_else(|| {
    let s = self.next_root_slot;
    self.next_root_slot += 1;
    s
});
self.root_slots.insert(register, root_slot);
```

**Spilled registers also get root slots** — in fact for spilled registers, the root slot IS their storage location. Change `spill_at_interval` to use the root slot as the spill location:
```rust
// Instead of: self.location.insert(spill.2, self.new_stack_location());
// Use: self.location.insert(spill.2, *self.root_slots.get(&spill.2).unwrap());
```

After allocation, `self.next_root_slot` is the total root slot count (= `num_roots` for the frame).

At the end, set `self.stack_slot = self.next_root_slot` so the existing `num_locals` flow picks it up.

### Step 2: Emit root slot stores after every write

**File: `src/ir.rs`** — in `compile_instructions()` (the big match on `Instruction`)

After every instruction that writes to a register, emit a `store_local` to the register's root slot. This requires passing the root slot map into the compilation.

Add a field to `Ir`:
```rust
pub root_slots: HashMap<VirtualRegister, usize>,
```

Populated from `linear_scan.root_slots` after allocation (mapping through `allocated_registers` to get physical register → root slot).

Then, after codegen for each instruction that has a destination register, emit:
```rust
// pseudo-code in the compile loop:
if let Some(root_slot) = self.root_slot_for(dest_register) {
    backend.store_local(dest_physical_reg, root_slot as i32);
}
```

**Important**: The `value_to_register()` function for `Value::Spill` already does `load_local(temp, index)`. With root slots, spills just use the root slot index, so this continues to work unchanged.

Similarly, `store_spill()` already does `store_local(dest, index)`. This becomes the root slot store for spilled registers. For non-spilled registers, you add an *additional* store to the root slot after the physical register is written.

### Step 3: Emit root slot reloads after safepoints

**File: `src/ir.rs`** — in `compile_instructions()`

After every safepoint (Call, Recurse, and any instruction that can trigger GC), reload all live registers from their root slots.

Use the lifetime information: at instruction index `i`, any virtual register with `start < i && end > i` that is in a physical register (not spilled) needs a reload:

```rust
// After emitting the call:
for (virt_reg, (start, end)) in &self.lifetimes {
    if *start <= i && *end > i {
        if let Some(phys_reg) = self.allocated_registers.get(virt_reg) {
            if let Some(root_slot) = self.root_slots.get(virt_reg) {
                let phys = backend.register_from_index(phys_reg.index);
                backend.load_local(phys, *root_slot as i32);
            }
        }
    }
}
```

**What counts as a safepoint**: `Call`, `Recurse`, `TailRecurse` (only for args before the jump), `CaptureContinuation`, `PerformEffect`. Also any explicit allocation builtins if those exist in the IR.

### Step 4: Remove WithSaves variants

**File: `src/ir.rs`**

Delete these instruction variants:
- `CallWithSaves`
- `RecurseWithSaves`
- `CaptureContinuationWithSaves`
- `PerformEffectWithSaves`

Delete `SavedValue` struct.

Update all match arms throughout `ir.rs` that handle these variants — the logic collapses into the plain variants (`Call`, `Recurse`, etc.) since saves/restores are now handled uniformly by Steps 2 and 3.

**File: `src/register_allocation/linear_scan.rs`**

Delete `replace_calls_with_call_with_save()` entirely. It's no longer called from `allocate()`.

### Step 5: Remove stack maps

**File: `src/gc/mod.rs`**

Delete `StackMap` and `StackMapDetails` structs. Remove the `stack_map` field from wherever it's stored in the runtime (likely in the memory/runtime struct).

**File: `src/gc/stack_walker.rs`**

Simplify `walk_stack_roots()`: instead of looking up `StackMapDetails` by return address, just read `num_roots` from the frame header's `size` field (which already exists — `header.size`). Remove `detached_frame_live_slots()` and its stack map lookup.

The scanning loop becomes simply:
```rust
let num_slots = header.size as usize;
for i in 0..num_slots {
    let slot_addr = header_addr - 16 - (i * 8);
    let slot_value = unsafe { *(slot_addr as *const usize) };
    if BuiltInTypes::is_heap_pointer(slot_value) { ... }
}
```

This is essentially what it already does on the main path (lines 133-176 of stack_walker.rs). The change is removing the `detached_frame_live_slots` path that does the stack map lookup for continuation frames.

**File: `src/backend/mod.rs`**

Remove from the `CodegenBackend` trait:
- `translate_stack_map()`
- `record_gc_safepoint()` (unless still needed for continuation resume points — check)
- `return_address_adjustment()`
- `increment_stack_size()` / `stack_size()` / `max_stack_size()`

**File: `src/arm.rs` (and `src/x86.rs`)**

Remove stack map recording storage and methods. Remove `max_stack_size` tracking.

**File: `src/compiler.rs`**

Remove the stack map construction (lines ~1152-1170) and registration. Functions no longer carry stack maps.

### Step 6: Simplify frame layout

**File: `src/arm.rs` — `patch_prelude_and_epilogue()`**

Frame size calculation becomes:
```
total_slots = 2 (header + prev) + num_roots + num_callee_saved
```

No more `max_stack_size` component. The header's `size` field = `num_roots` (just root slots, no eval stack).

Frame zeroing loop zeros `num_roots` slots instead of `max_locals + max_stack_size`.

The `type_data` encoding in the frame header should store `num_roots` in the upper 16 bits (it currently stores `num_slots = max_locals + max_stack_size`).

### Step 7: Handle continuations

**Files: `src/ir.rs`**

`CaptureContinuation` and `PerformEffect` currently have WithSaves variants that explicitly save callee-saved registers into a buffer for the continuation runtime. With root slots, all live values are already in the frame's root slots at all times. The continuation capture can just snapshot the frame.

However, when a continuation is *resumed*, the physical registers need to be reloaded from the root slots of the restored frame. This is similar to the post-safepoint reload in Step 3, but happens at the continuation resume point.

**Check**: The existing `RecordGcSafepoint` instruction marks continuation resume points. After resume, emit reloads for all registers that are live at that point.

The `emit_continuation_saved_regs_arg()` function (lines ~1674-1715 in ir.rs) that builds the callee-saved register array can potentially be simplified or removed if the continuation runtime reads root slots directly from the frame instead.

---

## What Stays the Same

- **Frame chain via prev pointers** — no change, `gc_frame_link`/`gc_frame_unlink` in prologue/epilogue stay
- **Frame zeroing in prologue** — stays, ensures root slots start as null
- **Tagged value scanning** — `BuiltInTypes::is_heap_pointer()` check during GC walk stays
- **Physical register allocation** — linear scan still maps virtual → physical registers; root slots are orthogonal
- **The `Value::Local` variant** — user-level locals (e.g., `let` bindings) continue to work; they occupy root slot indices 0..num_locals and are unaffected

## Testing Strategy

1. **Run the existing test suite** after each step — regressions will show as GC crashes or wrong results
2. **Binary Trees benchmark** is a good GC stress test (deep recursion, lots of allocation)
3. **Continuation tests** are critical — continuations capture/restore frames, so root slot layout must be consistent
4. **Enable `debug-gc` feature** to get verbose GC scanning output and verify root slots are populated correctly
5. **Compare frame sizes** before/after — root slot reuse should keep them comparable to current max_locals + max_stack_size

## Risk Areas

1. **Continuation capture/restore** is the most complex interaction. The current WithSaves logic explicitly manages which registers to save for continuations. Moving to "everything is in root slots" simplifies this *in theory*, but the continuation runtime code that restores registers needs to be updated to reload from root slots instead of from the saved register buffer.

2. **Performance**: Every register write now has an accompanying store to the frame. On ARM64, this is one extra `STR` per write. This is offset by removing the save/restore pairs around calls. Net effect depends on the ratio of register writes to calls — likely close to neutral, possibly better for call-heavy code.

3. **Root slot count**: With reuse, `num_roots` should be bounded by the max register pressure at any point in the function. But functions with many simultaneously-live values (e.g., large match arms with many bindings) could have more root slots than the current scheme. Monitor this.

4. **Argument registers**: Currently, argument registers (X0-X7) are handled specially — they're callee-saved by convention in Beagle (mapped via `register.argument`). They need root slots too, assigned at function entry, stored immediately in the prologue before any safepoint can occur.
