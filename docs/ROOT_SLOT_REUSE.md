# Root Slot Reuse in the Linear Allocator

## Problem

In the root-slot GC model, every virtual register needs a frame slot (root slot) so the GC can find and update heap pointers. Naively, this means `total_virtual_registers` frame slots per function — which can be far more than necessary, since many virtual registers have short, non-overlapping lifetimes.

## Solution: Liveness-Based Reuse

The linear scan allocator already computes `lifetimes: HashMap<VirtualRegister, (usize, usize)>` mapping each virtual register to its `(first_use, last_use)` instruction indices. Two virtual registers whose lifetimes don't overlap can share the same root slot.

### How It Works

Alongside the existing physical register free-pool, maintain a **root slot free-pool**:

```
root_slots: HashMap<VirtualRegister, usize>   // vreg → slot index
free_root_slots: Vec<usize>                    // returned slots available for reuse
next_root_slot: usize                          // bump counter (starts at num_locals)
```

**On interval start** (when a virtual register becomes live):
- Pop a slot from `free_root_slots`, or bump `next_root_slot` to allocate fresh.
- Record the mapping in `root_slots`.

**On interval expiry** (in `expire_old_intervals`, when a virtual register dies):
- Push its root slot back onto `free_root_slots`.

The result: `next_root_slot` at the end of allocation = the **maximum number of simultaneously live virtual registers**, not the total count. This is the `num_roots` encoded in the frame header.

### Worked Example

```
Instruction 0:  r0 = 42          ← r0 starts, gets root slot 0
Instruction 1:  r1 = r0 + 1      ← r1 starts, gets root slot 1 (r0 still live)
Instruction 2:  call(r1)          ← r0 dead after inst 1, r1 last used here
                                     2 root slots active simultaneously (slots 0, 1)
Instruction 3:  r2 = 10           ← r1 expired, slot 1 returned; r2 gets slot 1
                                     r0 expired, slot 0 returned; but r2 already got slot 1
Instruction 4:  r3 = r2 + 5       ← r3 starts, gets slot 0 (reused from r0)
Instruction 5:  ret r3             ← done

Total virtual registers: 4 (r0, r1, r2, r3)
Max simultaneously live: 2
Root slots needed: 2 (not 4)
```

### Why This Is Correct

The GC only needs to scan root slots for *currently live* values. A root slot that was freed and reassigned to a new virtual register will contain the new value (or zero, from frame zeroing). The old value is gone — but that's fine, because the old virtual register is dead and no code will ever reload it.

The key invariant: **at every safepoint, every live virtual register's current value is in its assigned root slot**. This is guaranteed by:
1. Every write to a physical register also stores to the root slot (Step 2 of the main plan).
2. Root slots are zeroed at frame entry (existing prologue behavior).
3. A freed root slot is only reused after its previous owner is dead.

### Spilled Registers

Spilled registers (ones that don't get a physical register) **use their root slot as their storage location**. Instead of allocating a separate spill slot, `spill_at_interval` uses the root slot the register already has. This unifies spill slots and root slots — there's only one kind of frame slot.

### Relationship to Physical Register Allocation

Root slot assignment is **orthogonal** to physical register allocation. Every virtual register gets a root slot regardless of whether it gets a physical register or is spilled. The root slot is where the GC looks; the physical register (if any) is where the CPU operates. They stay synchronized via stores (on write) and loads (after safepoints).

### Frame Size Impact

With reuse, `num_roots` = max register pressure at any point. This is comparable to the current scheme's `max_locals + max_stack_size` (which also reflects peak liveness, just measured differently). Functions with many short-lived temporaries benefit most — without reuse, each temporary would inflate the frame; with reuse, they share slots.
