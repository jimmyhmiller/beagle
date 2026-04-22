#![allow(dead_code)]
use std::collections::{BTreeMap, HashMap};

use crate::ir::{Instruction, SavedValue, Value, VirtualRegister};

pub struct LinearScan {
    pub lifetimes: HashMap<VirtualRegister, (usize, usize)>,
    pub instructions: Vec<Instruction>,
    // Needs to be a btreemap so I iterate in a defined order
    // so that things are deterministic
    pub allocated_registers: BTreeMap<VirtualRegister, VirtualRegister>,
    pub free_registers: Vec<VirtualRegister>,
    pub location: HashMap<VirtualRegister, usize>,
    pub stack_slot: usize,
    pub max_registers: usize,
    // Root slot assignment for GC root tracking
    pub root_slots: HashMap<VirtualRegister, usize>,
    free_root_slots: Vec<usize>,
    next_root_slot: usize,
    root_slot_base: usize,
}

fn physical(index: usize) -> VirtualRegister {
    VirtualRegister {
        argument: None,
        index,
        volatile: true,
        is_physical: true,
    }
}

impl LinearScan {
    pub fn new(instructions: Vec<Instruction>, num_locals: usize) -> Self {
        let lifetimes = Self::get_register_lifetime(&instructions);

        // Select physical registers based on backend
        cfg_if::cfg_if! {
            if #[cfg(any(feature = "backend-x86-64", all(target_arch = "x86_64", not(feature = "backend-arm64"))))] {
                // x86-64: callee-saved registers
                // We use R12-R15 (indices 12-15) and RBX (virtual index 16) because they
                // don't conflict with argument register indices (0-5 map to RDI, RSI, RDX, RCX, R8, R9).
                // RBX has raw x86 index 3 but we use virtual index 16 to avoid arg(3) conflict.
                let physical_registers: Vec<VirtualRegister> =
                    vec![12, 13, 14, 15, 16].into_iter().map(physical).collect();
            } else {
                // ARM64: callee-saved registers X19-X27.
                // X28 is reserved to hold a pointer to the current thread's
                // MutatorState (see runtime::MutatorState). Removing it from the
                // allocator pool ensures no generated code clobbers the slot.
                let physical_registers: Vec<VirtualRegister> = (19..=27).map(physical).collect();
            }
        }
        let max_registers = physical_registers.len();

        LinearScan {
            lifetimes,
            instructions,
            allocated_registers: BTreeMap::new(),
            free_registers: physical_registers,
            max_registers,
            location: HashMap::new(),
            stack_slot: num_locals,
            root_slots: HashMap::new(),
            free_root_slots: Vec::new(),
            next_root_slot: num_locals,
            root_slot_base: num_locals,
        }
    }

    fn get_register_lifetime(
        instructions: &[Instruction],
    ) -> HashMap<VirtualRegister, (usize, usize)> {
        let mut result: HashMap<VirtualRegister, (usize, usize)> = HashMap::new();
        for (index, instruction) in instructions.iter().enumerate().rev() {
            for register in instruction.get_registers() {
                if let Some((_start, end)) = result.get(&register) {
                    result.insert(register, (index, *end));
                } else {
                    result.insert(register, (index, index));
                }
            }
        }
        result
    }

    //  LinearScanRegisterAllocation
    //   active <- {}
    //   foreach live interval i, in order of increasing start point
    //     ExpireOldIntervals(i)
    //     if length(active) == R then
    //       SpillAtInterval(i)
    //     else
    //       register[i] <- a register removed from pool of free registers
    //       add i to active, sorted by increasing end point

    // ExpireOldIntervals(i)
    //   foreach interval j in active, in order of increasing end point
    //     if endpoint[j] >= startpoint[i] then  # > が正しい気がする。
    //       return
    //     remove j from active
    //     add register[j] to pool of free registers

    // SpillAtInterval(i)
    //   spill <- last interval in active
    //   if endpoint[spill] > endpoint[i] then
    //     register[i] <- register[spill]
    //     location[spill] <- new stack location
    //     remove spill from active
    //     add i to active, sorted by increasing end point
    //   else
    //     location[i] <- new stack location

    pub fn allocate(&mut self) {
        // Assign root slots first — every virtual register gets a frame slot
        // with reuse for non-overlapping lifetimes. Spills and call saves
        // use these slots instead of allocating fresh ones.
        self.assign_root_slots();

        let mut intervals = self
            .lifetimes
            .iter()
            .map(|(register, (start, end))| (*start, *end, *register))
            .collect::<Vec<_>>();

        intervals.sort_by_key(|(start, _, _)| *start);

        let mut active: Vec<(usize, usize, VirtualRegister)> = Vec::new();
        for i in intervals.iter() {
            let (start, end, register) = i;
            self.expire_old_intervals(*i, &mut active);
            if active.len() == self.max_registers {
                self.spill_at_interval(*start, *end, *register, &mut active);
            } else {
                if register.argument.is_some() {
                    // For argument registers, use the argument NUMBER as the index
                    // so that register_from_index() maps it to the correct physical register.
                    // This works because register_from_index(0) returns arg(0), etc.
                    let new_register = VirtualRegister {
                        argument: register.argument,
                        index: register.argument.unwrap(), // Use argument number, not virtual index
                        volatile: false,
                        is_physical: true,
                    };
                    self.allocated_registers.insert(*register, new_register);
                } else {
                    let physical_register = self.free_registers.pop().unwrap();
                    self.allocated_registers
                        .insert(*register, physical_register);
                }
                active.push(*i);
                active.sort_by_key(|(_, end, _)| *end);
            }
        }
        self.replace_spilled_registers_with_spill();
        self.replace_virtual_with_allocated();
        self.replace_calls_with_call_with_save();
        // Frame size is determined by root slots — all spills and saves use root slot indices
        self.stack_slot = self.next_root_slot;
    }

    fn expire_old_intervals(
        &mut self,
        i: (usize, usize, VirtualRegister),
        active: &mut Vec<(usize, usize, VirtualRegister)>,
    ) {
        let mut active_copy = active.clone();
        active_copy.sort_by_key(|(_, end, _)| *end);
        for j in active_copy.iter() {
            let (_, end, _) = j;
            if *end >= i.0 {
                return;
            }
            active.retain(|x| x != j);
            let register_to_free = *self.allocated_registers.get(&j.2).unwrap();
            self.free_register(register_to_free);
        }
    }

    fn spill_at_interval(
        &mut self,
        start: usize,
        end: usize,
        register: VirtualRegister,
        active: &mut Vec<(usize, usize, VirtualRegister)>,
    ) {
        let spill = *active.last().unwrap();
        if spill.1 > end {
            // The new interval (register) ends earlier than the spill victim,
            // so the new interval steals the physical register and the old one gets spilled.
            let physical_register = *self.allocated_registers.get(&spill.2).unwrap();
            self.allocated_registers.insert(register, physical_register);
            // Use the spilled register's root slot as its frame location
            let root_slot = *self.root_slots.get(&spill.2).unwrap();
            assert!(!self.location.contains_key(&spill.2));
            self.location.insert(spill.2, root_slot);
            active.retain(|x| *x != spill);
            // NOTE: Do NOT free the physical_register here - it's now in use by `register`.
            // The register will be freed when the new interval expires.
            active.push((start, end, register));
            active.sort_by_key(|(_, end, _)| *end);
        } else {
            // The new interval ends later, so it should be spilled directly.
            assert!(!self.location.contains_key(&register));
            let root_slot = *self.root_slots.get(&register).unwrap();
            self.location.insert(register, root_slot);
        }
    }

    /// Assign root slots to virtual registers, reusing slots when lifetimes don't overlap.
    /// Returns the total number of root slots needed (= max simultaneously live virtual registers).
    pub fn assign_root_slots(&mut self) -> usize {
        let mut intervals: Vec<(usize, usize, VirtualRegister)> = self
            .lifetimes
            .iter()
            .map(|(register, (start, end))| (*start, *end, *register))
            .collect();
        intervals.sort_by_key(|(start, _, _)| *start);

        // Track which root slots are currently in use, sorted by end point
        let mut active: Vec<(usize, usize, VirtualRegister)> = Vec::new();

        for (start, end, register) in intervals {
            // Expire intervals that ended before this one starts — free their root slots
            let mut expired = Vec::new();
            active.sort_by_key(|(_, end, _)| *end);
            for j in active.iter() {
                if j.1 < start {
                    expired.push(*j);
                }
            }
            for j in &expired {
                active.retain(|x| x != j);
                if let Some(slot) = self.root_slots.get(&j.2) {
                    self.free_root_slots.push(*slot);
                }
            }

            // Assign a root slot: reuse a free one or allocate fresh
            let root_slot = self.free_root_slots.pop().unwrap_or_else(|| {
                let s = self.next_root_slot;
                self.next_root_slot += 1;
                s
            });
            self.root_slots.insert(register, root_slot);
            active.push((start, end, register));
        }

        self.next_root_slot
    }

    /// Number of root slots needed (call after assign_root_slots).
    pub fn num_root_slots(&self) -> usize {
        self.next_root_slot
    }

    fn free_register(&mut self, register: VirtualRegister) {
        if register.argument.is_some() {
            return;
        }
        self.free_registers.push(register);
    }

    fn new_stack_location(&mut self) -> usize {
        let result = self.stack_slot;
        self.stack_slot += 1;
        result
    }

    fn replace_spilled_registers_with_spill(&mut self) {
        for instruction in self.instructions.iter_mut() {
            for register in instruction.get_registers() {
                if let Some(stack_offset) = self.location.get(&register) {
                    instruction.replace_register(register, Value::Spill(register, *stack_offset));
                }
            }
        }
    }

    fn replace_virtual_with_allocated(&mut self) {
        for instruction in self.instructions.iter_mut() {
            for register in instruction.get_registers() {
                if let Some(physical_register) = self.allocated_registers.get(&register) {
                    instruction.replace_register(register, Value::Register(*physical_register));
                }
            }
        }
    }

    fn replace_calls_with_call_with_save(&mut self) {
        for i in 0..self.instructions.len() {
            let instruction = self.instructions[i].clone();
            if let Instruction::Call(dest, f, args, builtin) = &instruction {
                // println!("{}", instruction.pretty_print());
                // We want to get all ranges that are valid at this point
                // if they are not spilled (meaning there isn't an entry in location)
                // we want to add them to the list of saves
                let mut saves = Vec::new();
                let mut live_root_slots = std::collections::HashSet::new();
                let live_ranges: Vec<(VirtualRegister, usize, usize)> = self
                    .lifetimes
                    .iter()
                    .map(|(register, (start, end))| (*register, *start, *end))
                    .collect();
                for (original_register, start, end) in live_ranges {
                    // Keep any root slot whose interval covers the call instruction
                    // itself. Spilled args/function values may be read while setting
                    // up the call even if they are dead immediately after it.
                    if start <= i && end >= i {
                        let root_slot = *self.root_slots.get(&original_register).unwrap();
                        live_root_slots.insert(root_slot);
                    }
                    // *end > i: register is used after the call (at instruction i+1 or later)
                    if start < i && end > i && !self.location.contains_key(&original_register) {
                        let register = self.allocated_registers.get(&original_register).unwrap();
                        // We save ALL registers that are live across calls, including callee-saved.
                        // While the ABI guarantees callee-saved registers are preserved by the callee,
                        // our GC needs to be able to find and update heap pointers during collection.
                        // If a register holds a heap pointer and GC runs during the call, the object
                        // may be moved. By saving to the stack, GC can scan and update the pointer.
                        if let Value::Register(dest) = dest
                            && dest == register
                        {
                            continue;
                        }
                        saves.push(SavedValue {
                            source: Value::Register(*register),
                            local: *self.root_slots.get(&original_register).unwrap(),
                        });
                    }
                }
                for dead_slot in self.root_slot_base..self.next_root_slot {
                    if !live_root_slots.contains(&dead_slot) {
                        saves.push(SavedValue {
                            source: Value::Null,
                            local: dead_slot,
                        });
                    }
                }
                self.instructions[i] =
                    Instruction::CallWithSaves(*dest, *f, args.clone(), *builtin, saves);
            } else if let Instruction::CaptureContinuation(dest, label, local_index, builtin) =
                &instruction
            {
                let mut saves = Vec::new();
                let live_ranges: Vec<(VirtualRegister, usize, usize)> = self
                    .lifetimes
                    .iter()
                    .map(|(register, (start, end))| (*register, *start, *end))
                    .collect();
                for (original_register, start, end) in live_ranges {
                    if start < i && end > i && !self.location.contains_key(&original_register) {
                        let register = self.allocated_registers.get(&original_register).unwrap();
                        if let Value::Register(dest) = dest
                            && dest == register
                        {
                            continue;
                        }
                        saves.push(SavedValue {
                            source: Value::Register(*register),
                            local: *self.root_slots.get(&original_register).unwrap(),
                        });
                    }
                }
                self.instructions[i] = Instruction::CaptureContinuationWithSaves(
                    *dest,
                    *label,
                    *local_index,
                    *builtin,
                    saves,
                );
            } else if let Instruction::CaptureContinuationTagged(
                dest,
                label,
                local_index,
                builtin,
                tag,
            ) = &instruction
            {
                let mut saves = Vec::new();
                let live_ranges: Vec<(VirtualRegister, usize, usize)> = self
                    .lifetimes
                    .iter()
                    .map(|(register, (start, end))| (*register, *start, *end))
                    .collect();
                for (original_register, start, end) in live_ranges {
                    if start < i && end > i && !self.location.contains_key(&original_register) {
                        let register = self.allocated_registers.get(&original_register).unwrap();
                        if let Value::Register(dest) = dest
                            && dest == register
                        {
                            continue;
                        }
                        saves.push(SavedValue {
                            source: Value::Register(*register),
                            local: *self.root_slots.get(&original_register).unwrap(),
                        });
                    }
                }
                self.instructions[i] = Instruction::CaptureContinuationTaggedWithSaves(
                    *dest,
                    *label,
                    *local_index,
                    *builtin,
                    *tag,
                    saves,
                );
            } else if let Instruction::PerformEffect(
                handler,
                enum_type,
                op_value,
                label,
                local_index,
                builtin,
            ) = &instruction
            {
                let mut saves = Vec::new();
                let live_ranges: Vec<(VirtualRegister, usize, usize)> = self
                    .lifetimes
                    .iter()
                    .map(|(register, (start, end))| (*register, *start, *end))
                    .collect();
                for (original_register, start, end) in live_ranges {
                    if start < i && end > i && !self.location.contains_key(&original_register) {
                        let register = self.allocated_registers.get(&original_register).unwrap();
                        saves.push(SavedValue {
                            source: Value::Register(*register),
                            local: *self.root_slots.get(&original_register).unwrap(),
                        });
                    }
                }
                self.instructions[i] = Instruction::PerformEffectWithSaves(
                    *handler,
                    *enum_type,
                    *op_value,
                    *label,
                    *local_index,
                    *builtin,
                    saves,
                );
            } else if let Instruction::Recurse(dest, args) = &instruction {
                let mut saves = Vec::new();
                for (original_register, (start, end)) in self.lifetimes.iter() {
                    // *end > i: register is used after the call (at instruction i+1 or later)
                    if *start < i && *end > i && !self.location.contains_key(original_register) {
                        let register = self.allocated_registers.get(original_register).unwrap();
                        // Save all live registers across recursive calls for GC safety
                        if let Value::Register(dest) = dest
                            && dest == register
                        {
                            continue;
                        }
                        saves.push(SavedValue {
                            source: Value::Register(*register),
                            local: *self.root_slots.get(original_register).unwrap(),
                        });
                    }
                }
                self.instructions[i] = Instruction::RecurseWithSaves(*dest, args.clone(), saves);
            }
        }

        // Note: ReloadRootSlots at resume points is handled in compile_instructions
        // (ir.rs) by looking up the resume label from CaptureContinuationWithSaves
        // and PerformEffectWithSaves instructions. We don't insert instructions here
        // because that would invalidate label_locations indices.
    }
}

// ============================================================
// Root slot reuse tests
// ============================================================

// Key lifetime semantics: `get_register_lifetime` computes inclusive ranges.
// For `Assign(r1, r0)`, both r0 and r1 are referenced at that instruction,
// so they overlap there. Two registers can share a root slot only when the
// first register's last-use instruction is strictly before the second's first-use.

#[test]
fn test_non_overlapping_locals_share_root_slot() {
    // r0 and r3 have non-overlapping lifetimes and should share a root slot.
    //
    // Instructions:
    //   0: Assign(r0, 42)        ← r0 live [0, 2]
    //   1: Assign(r1, 10)        ← r1 live [1, 2]
    //   2: AddInt(r2, r0, r1)    ← r2 live [2, 4]; peak 3 (r0, r1, r2)
    //   3: Assign(r3, 99)        ← r3 live [3, 4]; r0 dead (end=2 < start=3)
    //   4: AddInt(r4, r2, r3)    ← r4 live [4, 5]; peak 3 (r2, r3, r4)
    //   5: Ret(r4)
    //
    // Peak simultaneous: 3 (at inst 2 and inst 4).
    // But r0 [0,2] and r3 [3,4] don't overlap → share a slot.
    // Similarly r1 [1,2] and r4 [4,5] don't overlap → share a slot.
    // 5 vregs, 3 root slots.
    use crate::ir::Ir;

    let mut ir = Ir::new(0);
    let r0 = ir.assign_new(42);
    let r1 = ir.assign_new(10);
    let r2 = ir.add_int(r0, r1);
    let r3 = ir.assign_new(99);
    let r4 = ir.add_int(r2, r3);
    ir.ret(r4);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    ls.assign_root_slots();

    let slot_r0 = *ls.root_slots.get(&r0).unwrap();
    let slot_r1 = *ls.root_slots.get(&r1).unwrap();
    let slot_r3 = *ls.root_slots.get(&r3).unwrap();

    // r3 [3,4] doesn't overlap with r0 [0,2] or r1 [1,2] — must reuse one of their slots
    assert!(
        slot_r3 == slot_r0 || slot_r3 == slot_r1,
        "r3 (live [3,4]) should reuse slot from r0 ({}) or r1 ({}), got {}",
        slot_r0,
        slot_r1,
        slot_r3
    );

    // Peak liveness is 3 → 3 root slots (not 5)
    assert_eq!(
        ls.num_root_slots(),
        3,
        "peak liveness is 3, should need 3 root slots, got {}",
        ls.num_root_slots()
    );
}

#[test]
fn test_overlapping_locals_get_separate_root_slots() {
    // r0 and r1 are both live at instruction 2 (the AddInt) → must get separate slots.
    //
    //   0: Assign(r0, 42)        ← r0 live [0, 2]
    //   1: Assign(r1, 10)        ← r1 live [1, 2], r0 still live
    //   2: AddInt(r2, r0, r1)    ← 3 live: r0, r1, r2
    //   3: Ret(r2)
    use crate::ir::Ir;

    let mut ir = Ir::new(0);
    let r0 = ir.assign_new(42);
    let r1 = ir.assign_new(10);
    let r2 = ir.add_int(r0, r1);
    ir.ret(r2);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    ls.assign_root_slots();

    let slot_r0 = *ls.root_slots.get(&r0).unwrap();
    let slot_r1 = *ls.root_slots.get(&r1).unwrap();

    assert_ne!(
        slot_r0, slot_r1,
        "overlapping lifetimes must get separate root slots"
    );
}

#[test]
fn test_chain_reuses_with_two_slots() {
    // A chain `r0 → r1 → r2 → r3` via assign_new_force.
    // At each Assign(rN, rN-1), both registers are referenced → overlap.
    // So peak is always 2. With reuse: r0 and r2 share, r1 and r3 share.
    //
    //   0: Assign(r0, 1)       ← r0 live [0, 1]
    //   1: Assign(r1, r0)      ← r0 end=1, r1 start=1 → overlap at 1; 2 live
    //   2: Assign(r2, r1)      ← r1 end=2, r2 start=2 → overlap; r0 dead → 2 live
    //   3: Assign(r3, r2)      ← r2 end=3, r3 start=3 → overlap; r1 dead → 2 live
    //   4: Ret(r3)
    //
    // 4 vregs, peak 2, root slots needed: 2
    // r0 [0,1] and r2 [2,3]: no overlap (1 < 2) → share
    // r1 [1,2] and r3 [3,4]: no overlap (2 < 3) → share
    use crate::ir::Ir;

    let mut ir = Ir::new(0);
    let r0 = ir.assign_new(1);
    let r1 = ir.assign_new_force(Value::Register(r0));
    let r2 = ir.assign_new_force(Value::Register(r1));
    let r3 = ir.assign_new_force(Value::Register(r2));
    ir.ret(r3);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    ls.assign_root_slots();

    assert_eq!(
        ls.num_root_slots(),
        2,
        "chain with peak 2 should need 2 root slots, got {}",
        ls.num_root_slots()
    );

    // r0 and r2 should share a slot (non-overlapping)
    let slot_r0 = *ls.root_slots.get(&r0).unwrap();
    let slot_r2 = *ls.root_slots.get(&r2).unwrap();
    assert_eq!(
        slot_r0, slot_r2,
        "r0 [0,1] and r2 [2,3] should share a root slot"
    );

    // r1 and r3 should share a slot (non-overlapping)
    let slot_r1 = *ls.root_slots.get(&r1).unwrap();
    let slot_r3 = *ls.root_slots.get(&r3).unwrap();
    assert_eq!(
        slot_r1, slot_r3,
        "r1 [1,2] and r3 [3,4] should share a root slot"
    );
}

#[test]
fn test_root_slots_start_after_num_locals() {
    // When num_locals > 0, root slots should be numbered starting at num_locals.
    use crate::ir::Ir;

    let mut ir = Ir::new(0);
    let r0 = ir.assign_new(1);
    ir.ret(r0);

    let num_locals = 3;
    let mut ls = LinearScan::new(ir.instructions.clone(), num_locals);
    ls.assign_root_slots();

    let slot = *ls.root_slots.get(&r0).unwrap();
    assert!(
        slot >= num_locals,
        "root slot {} should be >= num_locals {}",
        slot,
        num_locals
    );
}

#[test]
fn test_diamond_pattern_reuses_root_slots() {
    // Three values built up, consumed pairwise, new values reuse freed slots.
    //
    //   0: Assign(r0, 1)         ← r0 live [0, 3]
    //   1: Assign(r1, 2)         ← r1 live [1, 3]
    //   2: Assign(r2, 3)         ← r2 live [2, 4]; peak: 3 (r0, r1, r2)
    //   3: AddInt(r3, r0, r1)    ← r0,r1 die (end=3); r3 live [3, 4]; 3 live (r2,r3 + dying r0,r1)
    //   4: AddInt(r4, r3, r2)    ← r2,r3 die; r4 live [4, 5]
    //   5: Ret(r4)
    //
    // Peak: 3 (at inst 2-3). With reuse, r3 takes a slot from dead r0 or r1.
    // 5 vregs → 3 root slots.
    use crate::ir::Ir;

    let mut ir = Ir::new(0);
    let r0 = ir.assign_new(1);
    let r1 = ir.assign_new(2);
    let r2 = ir.assign_new(3);
    let r3 = ir.add_int(r0, r1);
    let r4 = ir.add_int(r3, r2);
    ir.ret(r4);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    ls.assign_root_slots();

    // At inst 3, r0 and r1 end but r3 starts at the same instruction.
    // They overlap at inst 3, so r3 can't reuse r0/r1's slot yet.
    // At inst 4, r3 and r2 end, r4 starts. r0, r1 are long dead → r4 reuses.
    // Peak: 3 at inst 3 (r0, r1, r2 still live + r3 starting = but r0,r1 end here too)
    //
    // The exact count depends on whether the implementation considers
    // end == start as overlapping (our expiry uses strict <).
    // With strict <: r0 end=3, r3 start=3, so 3 < 3 is false → overlap → 4 slots
    // This is conservative but correct.
    let num_slots = ls.num_root_slots();
    assert!(
        num_slots < 5,
        "should reuse at least some slots (5 vregs but {} root slots)",
        num_slots
    );

    // r4 should definitely reuse a slot — it starts at 4, and r0 (end=3), r1 (end=3) are dead
    let r4_reg = match r4 {
        Value::Register(reg) => reg,
        _ => panic!("expected register"),
    };
    let slot_r4 = *ls.root_slots.get(&r4_reg).unwrap();
    let slot_r0 = *ls.root_slots.get(&r0).unwrap();
    let slot_r1 = *ls.root_slots.get(&r1).unwrap();
    assert!(
        slot_r4 == slot_r0 || slot_r4 == slot_r1,
        "r4 (start=4) should reuse slot from r0 or r1 (end=3)"
    );
}

#[test]
fn test_five_chain_reuses_two_slots() {
    // Longer chain to show the alternating reuse pattern.
    //   0: Assign(r0, 1)       [0, 1]
    //   1: Assign(r1, r0)      [1, 2]  — r0 overlaps at 1
    //   2: Assign(r2, r1)      [2, 3]  — r1 overlaps at 2, r0 expired (1 < 2)
    //   3: Assign(r3, r2)      [3, 4]  — r2 overlaps at 3, r1 expired (2 < 3)
    //   4: Assign(r4, r3)      [4, 5]  — r3 overlaps at 4, r2 expired (3 < 4)
    //   5: Ret(r4)
    //
    // Peak always 2. Slots: A={r0,r2,r4}, B={r1,r3}
    use crate::ir::Ir;

    let mut ir = Ir::new(0);
    let r0 = ir.assign_new(1);
    let r1 = ir.assign_new_force(Value::Register(r0));
    let r2 = ir.assign_new_force(Value::Register(r1));
    let r3 = ir.assign_new_force(Value::Register(r2));
    let r4 = ir.assign_new_force(Value::Register(r3));
    ir.ret(r4);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    ls.assign_root_slots();

    assert_eq!(
        ls.num_root_slots(),
        2,
        "5-element chain (peak 2) should need 2 root slots, got {}",
        ls.num_root_slots()
    );

    // r0, r2, r4 should all share one slot
    let slot_r0 = *ls.root_slots.get(&r0).unwrap();
    let slot_r2 = *ls.root_slots.get(&r2).unwrap();
    let slot_r4 = *ls.root_slots.get(&r4).unwrap();
    assert_eq!(slot_r0, slot_r2, "r0 and r2 should share a slot");
    assert_eq!(slot_r2, slot_r4, "r2 and r4 should share a slot");

    // r1, r3 should share the other slot
    let slot_r1 = *ls.root_slots.get(&r1).unwrap();
    let slot_r3 = *ls.root_slots.get(&r3).unwrap();
    assert_eq!(slot_r1, slot_r3, "r1 and r3 should share a slot");

    // The two groups use different slots
    assert_ne!(
        slot_r0, slot_r1,
        "the two alternating groups use different slots"
    );
}

#[test]
fn test_extreme_all_live_simultaneously_no_reuse() {
    // Worst case: N registers all live at the same time. No reuse possible.
    // Create 20 registers, then consume them all in a single reduction chain.
    //
    //   0:  Assign(r0, 0)
    //   1:  Assign(r1, 1)
    //   ...
    //   19: Assign(r19, 19)
    //   20: AddInt(a0, r0, r1)       ← all r0..r19 still live here (r2..r19 used later)
    //   21: AddInt(a1, a0, r2)
    //   ...
    //   38: AddInt(a18, a17, r19)    ← finally r19 dies
    //   39: Ret(a18)
    //
    // At instruction 20, registers r0..r19 + a0 are all referenced → 21 live.
    // Peak is 21 (20 source regs + 1 result at the first AddInt). No reuse until
    // registers start dying off one by one.
    use crate::ir::Ir;

    let mut ir = Ir::new(0);
    let regs: Vec<VirtualRegister> = (0..20).map(|i| ir.assign_new(i)).collect();

    // Fold them: a = r0 + r1, a = a + r2, a = a + r3, ...
    let mut acc = ir.add_int(regs[0], regs[1]);
    for r in &regs[2..] {
        acc = ir.add_int(acc, *r);
    }
    ir.ret(acc);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    let total_vregs = ls.lifetimes.len();
    ls.assign_root_slots();

    // All 20 source regs are live until they're consumed. Peak is 21 (20 sources + 1 accumulator).
    // The fold consumes one source per step, so reuse kicks in for later accumulators.
    // But the peak is still 21, so we need at least 21 slots.
    assert!(
        ls.num_root_slots() >= 21,
        "with 20 simultaneous source regs, need at least 21 root slots, got {}",
        ls.num_root_slots()
    );
    // But we should use fewer than total vregs (20 sources + 19 accumulators = 39)
    assert!(
        ls.num_root_slots() < total_vregs,
        "should reuse slots for later accumulators: {} root slots vs {} total vregs",
        ls.num_root_slots(),
        total_vregs
    );
}

#[test]
fn test_extreme_long_sequential_chain_max_reuse() {
    // Best case: 50-element chain where each register dies before the next starts.
    // Should need exactly 2 root slots no matter how long the chain.
    //
    //   0: Assign(r0, 1)        [0, 1]
    //   1: Assign(r1, r0)       [1, 2]   — overlaps r0 at inst 1
    //   2: Assign(r2, r1)       [2, 3]   — overlaps r1 at inst 2, r0 freed
    //   ...
    //   49: Assign(r49, r48)    [49, 50]
    //   50: Ret(r49)
    //
    // Peak always 2. 50 vregs → 2 root slots. 25x compression.
    use crate::ir::Ir;

    let n = 50;
    let mut ir = Ir::new(0);
    let mut prev = ir.assign_new(1);
    let mut all_regs = vec![prev];
    for _ in 1..n {
        let next = ir.assign_new_force(Value::Register(prev));
        all_regs.push(next);
        prev = next;
    }
    ir.ret(prev);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    ls.assign_root_slots();

    assert_eq!(
        ls.num_root_slots(),
        2,
        "{}-element chain should need exactly 2 root slots, got {}",
        n,
        ls.num_root_slots()
    );

    // Verify alternating pattern: even-indexed regs share slot A, odd share slot B
    let slot_a = *ls.root_slots.get(&all_regs[0]).unwrap();
    let slot_b = *ls.root_slots.get(&all_regs[1]).unwrap();
    assert_ne!(slot_a, slot_b);

    for (i, reg) in all_regs.iter().enumerate() {
        let slot = *ls.root_slots.get(reg).unwrap();
        let expected = if i % 2 == 0 { slot_a } else { slot_b };
        assert_eq!(
            slot,
            expected,
            "r{} should be in slot {} ({}), got {}",
            i,
            if i % 2 == 0 { "A" } else { "B" },
            expected,
            slot
        );
    }
}

#[test]
fn test_extreme_one_long_lived_many_short_lived() {
    // One register lives across the entire function while many short-lived
    // registers come and go. The short-lived ones should all reuse slots
    // among themselves, but never the long-lived register's slot.
    //
    //   0:  Assign(long, 999)               [0, 2*N]
    //   1:  Assign(s0, 0)                   [1, 2]
    //   2:  AddInt(t0, long, s0)            [2, 2]  — s0 dies
    //   3:  Assign(s1, 1)                   [3, 4]
    //   4:  AddInt(t1, long, s1)            [4, 4]  — s1 dies
    //   ...
    //   2N-1: Assign(sN-1, N-1)
    //   2N:   AddInt(tN-1, long, sN-1)      — long finally consumed
    //   2N+1: Ret(tN-1)
    //
    // At any point: long + 1 short + 1 temp = 3 live max.
    // Total vregs: 1 (long) + N (shorts) + N (temps) = 2N+1.
    // Root slots needed: 3. Massive reuse for the short-lived ones.
    use crate::ir::Ir;

    let n = 20;
    let mut ir = Ir::new(0);
    let long = ir.assign_new(999);

    let mut last_temp = Value::Register(long); // placeholder
    for i in 0..n {
        let short = ir.assign_new(i);
        last_temp = ir.add_int(long, short);
    }
    ir.ret(last_temp);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    let total_vregs = ls.lifetimes.len();
    ls.assign_root_slots();

    // Peak: long + short_i + temp_i = 3 live at each AddInt.
    // But at AddInt, long (still live), short_i (end here), temp_i (start here)
    // → 3 live. Next iteration: long, short_{i+1} (start), so short_i's slot is freed.
    assert_eq!(
        ls.num_root_slots(),
        3,
        "one long-lived + rotating short-lived should need 3 root slots, got {}",
        ls.num_root_slots()
    );

    assert!(
        total_vregs >= 2 * n + 1,
        "should have many vregs ({}) compressed into 3 root slots",
        total_vregs
    );
}

#[test]
fn test_extreme_pressure_spike_then_release() {
    // Pressure builds to a peak, then drops. Slots allocated at peak are
    // reused during the low-pressure tail.
    //
    // Phase 1 (high pressure): create 10 regs all live simultaneously
    //   0..9: Assign(r0..r9, 0..9)
    //
    // Phase 2 (consume all at once): fold into one value
    //   10: AddInt(a0, r0, r1)     — peak: 10 sources + 1 result = 11
    //   11: AddInt(a1, a0, r2)     — r0,r1 dead → freed slots
    //   ...
    //
    // Phase 3 (low pressure): 10 new registers, each sequential
    //   After fold completes, chain 10 more assigns through the result.
    //   These should ALL reuse slots freed from phase 1.
    use crate::ir::Ir;

    let mut ir = Ir::new(0);

    // Phase 1: 10 simultaneously live registers
    let phase1: Vec<VirtualRegister> = (0..10).map(|i| ir.assign_new(i)).collect();

    // Phase 2: fold them down
    let mut acc = ir.add_int(phase1[0], phase1[1]);
    for r in &phase1[2..] {
        acc = ir.add_int(acc, *r);
    }

    // Phase 3: chain of 10 sequential assigns through the accumulated value
    let mut prev = acc;
    for _ in 0..10 {
        prev = ir.add_int(prev, prev); // self-add just to create a new vreg
    }
    ir.ret(prev);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    let total_vregs = ls.lifetimes.len();
    ls.assign_root_slots();

    let num_slots = ls.num_root_slots();

    // Peak is 11 (10 sources + 1 acc at first AddInt).
    // Phase 3 adds 10 more vregs but should reuse freed slots from phase 1.
    // Total vregs = 10 (sources) + 9 (fold accs) + 10 (phase3) = 29
    // Root slots should be ~11 (the peak), NOT 29.
    assert!(
        num_slots <= 12,
        "spike-then-release: peak is ~11 but got {} root slots (total vregs: {})",
        num_slots,
        total_vregs
    );
    assert!(
        num_slots < total_vregs,
        "must reuse: {} root slots should be less than {} total vregs",
        num_slots,
        total_vregs
    );
}

#[test]
fn test_extreme_interleaved_lifetimes() {
    // Registers with deliberately interleaved lifetimes — each pair overlaps
    // but non-adjacent ones don't.
    //
    //   0: Assign(r0, 0)        [0, 2]
    //   1: Assign(r1, 1)        [1, 3]
    //   2: Assign(r2, r0)       [2, 4]   — r0 dies at 2, overlaps with r1 and r2
    //   3: Assign(r3, r1)       [3, 5]   — r1 dies at 3
    //   4: Assign(r4, r2)       [4, 6]   — r2 dies at 4
    //   5: Assign(r5, r3)       [5, 7]   — r3 dies at 5
    //   6: Assign(r6, r4)       [6, 8]   — r4 dies at 6
    //   7: Assign(r7, r5)       [7, 8]   — r5 dies at 7
    //   8: AddInt(r8, r6, r7)   [8, 9]
    //   9: Ret(r8)
    //
    // At any instruction, at most 3 registers are live (the staggered window).
    // Total 9 vregs → should need only 3 root slots.
    use crate::ir::Ir;

    let mut ir = Ir::new(0);
    let r0 = ir.assign_new(0);
    let r1 = ir.assign_new(1);
    let r2 = ir.assign_new_force(Value::Register(r0)); // r0 last used
    let r3 = ir.assign_new_force(Value::Register(r1)); // r1 last used
    let r4 = ir.assign_new_force(Value::Register(r2)); // r2 last used
    let r5 = ir.assign_new_force(Value::Register(r3)); // r3 last used
    let r6 = ir.assign_new_force(Value::Register(r4)); // r4 last used
    let r7 = ir.assign_new_force(Value::Register(r5)); // r5 last used
    let r8 = ir.add_int(r6, r7);
    ir.ret(r8);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    ls.assign_root_slots();

    // Staggered window of width 2 (each assign overlaps source + dest),
    // plus the final AddInt with 3 (r6, r7, r8). Peak is 3.
    assert!(
        ls.num_root_slots() <= 3,
        "interleaved staggered lifetimes should need at most 3 root slots, got {}",
        ls.num_root_slots()
    );

    // Verify reuse: 9 vregs compressed into at most 3 slots
    assert!(
        ls.root_slots
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len()
            <= 3,
        "9 vregs should fit in at most 3 distinct root slots"
    );
}

// ============================================================
// Root slot integration tests — verifying that allocate() uses
// root slots for spills and call saves, reducing frame size
// ============================================================

#[test]
fn test_call_saves_reuse_root_slots() {
    // A register live across multiple calls should use the SAME root slot
    // for all saves, not a fresh slot per call site.
    //
    // Old behavior: 3 calls × 1 live register = 3 save slots
    // New behavior: 1 root slot reused across all 3 calls
    //
    //   0: Assign(r0, 42)           — r0 live [0, 7]
    //   1: Assign(func, 99)         — func register for calls
    //   2: Call(c0, func, [])       — r0 live, saved to root slot
    //   3: Call(c1, func, [])       — r0 live, saved to SAME root slot
    //   4: Call(c2, func, [])       — r0 live, saved to SAME root slot
    //   5: AddInt(result, r0, c2)   — r0 finally consumed
    //   6: Ret(result)
    use crate::ir::Ir;

    let mut ir = Ir::new(0);
    let r0 = ir.assign_new(42);
    let func = ir.assign_new(99);
    let _c0 = ir.call(Value::Register(func), vec![]);
    let _c1 = ir.call(Value::Register(func), vec![]);
    let c2 = ir.call(Value::Register(func), vec![]);
    let result = ir.add_int(r0, c2);
    ir.ret(result);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    ls.allocate();

    // With root slots: each register gets one slot, reused across all call saves.
    // r0, func, c0, c1, c2, result — but not all live simultaneously.
    // The key metric: stack_slot should be much less than
    // what 3 calls × 2 live registers (r0 + func) = 6 separate save slots would need.
    //
    // Count the actual SavedValue locals in the WithSaves instructions to verify reuse.
    let mut save_locals: Vec<usize> = Vec::new();
    for inst in &ls.instructions {
        if let Instruction::CallWithSaves(_, _, _, _, saves) = inst {
            for save in saves {
                save_locals.push(save.local);
            }
        }
    }

    // Multiple calls should reuse the same locals (root slots)
    let unique_locals: std::collections::HashSet<usize> = save_locals.iter().copied().collect();
    assert!(
        unique_locals.len() < save_locals.len(),
        "root slots should be reused across calls: {} unique locals for {} total saves",
        unique_locals.len(),
        save_locals.len()
    );
}

#[test]
fn test_many_calls_same_live_set_constant_slots() {
    // 10 calls with the same 2 registers live across all of them.
    // Old: 10 calls × 2 saves = 20 save slots
    // New: 2 root slots reused across all 10 calls
    use crate::ir::Ir;

    let mut ir = Ir::new(0);
    let r0 = ir.assign_new(42);
    let r1 = ir.assign_new(99);
    let func = ir.assign_new_force(Value::TaggedConstant(1));
    for _ in 0..10 {
        let _ = ir.call(Value::Register(func), vec![]);
    }
    let sum = ir.add_int(r0, r1);
    ir.ret(sum);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    ls.allocate();

    // Collect all save locals across all WithSaves instructions
    let mut all_save_locals: Vec<usize> = Vec::new();
    for inst in &ls.instructions {
        if let Instruction::CallWithSaves(_, _, _, _, saves) = inst {
            for save in saves {
                all_save_locals.push(save.local);
            }
        }
    }

    let unique_locals: std::collections::HashSet<usize> = all_save_locals.iter().copied().collect();

    // Old behavior would have ~20 unique locals (one per save per call).
    // New behavior: only as many unique locals as there are simultaneously-live registers.
    assert!(
        unique_locals.len() <= 5,
        "10 calls with same live set should reuse a small number of root slots, got {} unique (from {} total saves)",
        unique_locals.len(),
        all_save_locals.len()
    );

    // Total saves should be much larger than unique slots (proving reuse)
    assert!(
        all_save_locals.len() > unique_locals.len() * 2,
        "should have many total saves ({}) reusing few slots ({})",
        all_save_locals.len(),
        unique_locals.len()
    );
}

#[test]
fn test_stack_slot_smaller_with_root_slots() {
    // Directly verify that stack_slot (frame size) is smaller than it would be
    // without root slot reuse. We do this by counting how many new_stack_location()
    // calls the old approach would have made.
    //
    // Setup: 3 registers live across 5 calls.
    // Old: 5 calls × 3 saves = 15 save slots → stack_slot >= 15
    // New: 3 root slots → stack_slot = max simultaneously live ≈ 6-8
    use crate::ir::Ir;

    let mut ir = Ir::new(0);
    let r0 = ir.assign_new(1);
    let r1 = ir.assign_new(2);
    let r2 = ir.assign_new(3);
    let func = ir.assign_new_force(Value::TaggedConstant(0));
    for _ in 0..5 {
        let _ = ir.call(Value::Register(func), vec![]);
    }
    let s1 = ir.add_int(r0, r1);
    let s2 = ir.add_int(s1, r2);
    ir.ret(s2);

    let mut ls = LinearScan::new(ir.instructions.clone(), 0);
    ls.allocate();

    // Count what the old approach would have allocated:
    // each call saves ~3-4 live registers, 5 calls = 15-20 save slots
    let mut total_saves = 0;
    for inst in &ls.instructions {
        if let Instruction::CallWithSaves(_, _, _, _, saves) = inst {
            total_saves += saves.len();
        }
    }

    // stack_slot should be much less than total_saves
    // (root slots reuse means we only need max-simultaneous-live slots)
    assert!(
        ls.stack_slot < total_saves,
        "frame size ({}) should be less than total saves without reuse ({})",
        ls.stack_slot,
        total_saves,
    );
}

#[test]
fn test_example() {
    use crate::{ir::Ir, pretty_print::PrettyPrint};
    let mut ir = Ir::new(0);
    let r0 = ir.assign_new(0);
    let r1 = ir.assign_new(0);
    let r2 = ir.assign_new(0);
    let r3 = ir.assign_new(0);
    let r4 = ir.assign_new(0);
    let r5 = ir.assign_new(0);
    let r6 = ir.assign_new(0);
    let r7 = ir.assign_new(0);
    let r8 = ir.assign_new(0);
    let r9 = ir.assign_new(0);
    let r10 = ir.assign_new(0);
    let add1 = ir.add_int(r1, r2);
    let add2 = ir.add_int(r3, r4);
    let add3 = ir.add_int(r5, r6);
    let add4 = ir.add_int(r7, r8);
    let add5 = ir.add_int(r9, r10);
    let add6 = ir.add_int(r0, r1);
    let add7 = ir.add_int(r2, r3);
    let add8 = ir.add_int(r4, r5);
    let add9 = ir.add_int(r6, r7);
    let add10 = ir.add_int(r8, r9);
    let add11 = ir.add_int(r10, r0);
    let add12 = ir.add_int(add1, add2);
    let add13 = ir.add_int(add3, add4);
    let add14 = ir.add_int(add5, add6);
    let add15 = ir.add_int(add7, add8);
    let add16 = ir.add_int(add9, add10);
    let add17 = ir.add_int(add11, add12);
    let add18 = ir.add_int(r0, r1);
    let add19 = ir.add_int(r2, r3);
    let add20 = ir.add_int(r4, r5);
    let add21 = ir.add_int(r6, r7);
    let add22 = ir.add_int(r8, r9);
    let add23 = ir.add_int(r10, r0);
    let add24 = ir.add_int(add1, add2);
    let add25 = ir.add_int(add24, add13);
    let add26 = ir.add_int(add25, add14);
    let add27 = ir.add_int(add26, add15);
    let add28 = ir.add_int(add27, add16);
    let add29 = ir.add_int(add28, add17);
    let add30 = ir.add_int(add29, add18);
    let add31 = ir.add_int(add30, add19);
    let add32 = ir.add_int(add31, add20);
    let add33 = ir.add_int(add32, add21);
    let add34 = ir.add_int(add33, add22);
    let add35 = ir.add_int(add34, add23);
    ir.ret(add35);

    let mut linear_scan = LinearScan::new(ir.instructions.clone(), 0);
    linear_scan.allocate();
    println!("{:#?}", linear_scan.allocated_registers);
    println!("=======");
    println!("{:#?}", linear_scan.location);

    println!("{}", linear_scan.instructions.pretty_print());
}
