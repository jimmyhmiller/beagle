//! GC-safety analysis for SSA passes that move slot-backed values
//! into registers.
//!
//! Per **I9** in `docs/SSA_ARCHITECTURE.md`, Beagle's GC scans the
//! frame's local slots — it does **not** scan registers or the
//! callee-saved spill area. Any GP-class value live across a
//! GC-safepoint op must therefore be in a slot at that op, or the
//! GC will miss it and (under compacting GC) the value's register
//! copy goes stale when the heap object is relocated.
//!
//! The legacy IR satisfies this by AST-compiler discipline (every
//! evaluated arg gets a `push_to_stack` slot before the next arg is
//! evaluated). The input CFG therefore has no GP VReg live across a
//! safepoint — until an optimization violates it.
//!
//! **`mem2reg` is the prime violator**: promoting a slot removes its
//! `SlotStore` ops, leaving the value only in SSA / registers. If
//! the value's live range crosses a safepoint, GC misses it.
//!
//! This module is the single source of truth for:
//!   1. *What counts as a GC-safepoint op* (`is_gc_safepoint_op`,
//!      `is_gc_safepoint_terminator`).
//!   2. *Whether a slot is safe to promote* under I9
//!      (`slot_is_gc_safe_to_promote`).
//!
//! Other passes that extend SSA-value lifetimes (trivial-param
//! elimination, copy coalesce) must call into the same module before
//! extending a value across a safepoint.
//!
//! # Algorithm — per-slot forward dataflow
//!
//! For each slot `S`, walk the CFG forward from the entry, tracking:
//!
//! ```text
//! enum State {
//!     NoStore,             // no SlotStore(S) seen yet on this path
//!     StorePending(false), // SlotStore happened, no safepoint yet
//!     StorePending(true),  // SlotStore happened, safepoint happened
//!                          //   → if a SlotLoad(S) appears now, UNSAFE
//! }
//! ```
//!
//! Transitions per op:
//!   - `SlotStore(S)`         → `StorePending(false)`     (new value)
//!   - `SlotLoad(S)`          → if state == StorePending(true), UNSAFE;
//!                              otherwise unchanged.
//!   - GC-safepoint op        → if state == StorePending(false),
//!                              advance to `StorePending(true)`.
//!   - Any other op           → unchanged.
//!
//! Block-level merge (least-upper-bound):
//!   - `NoStore ⊔ X            = X`
//!   - `StorePending(false) ⊔ StorePending(true)  = StorePending(true)`
//!   - same-state merge        = same state.
//!
//! Worklist iterates until fixed point or UNSAFE flag triggers early
//! exit. Bounded by `3 * num_blocks` state changes per slot.

#![allow(dead_code)]

use std::collections::{HashSet, VecDeque};

use crate::cfg::{BlockId, CfgFunction, InlineBranchOp, Op, SlotId, Terminator};

/// True if this op can transfer control to the GC (directly via
/// `RecordGcSafepoint` or indirectly via a runtime call that may
/// allocate / collect).
///
/// **Conservative set** — anything that crosses the Rust/JIT
/// boundary or may allocate. Missing one would silently miss roots;
/// false positives just prevent some `mem2reg` promotions.
pub fn is_gc_safepoint_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Call { .. }
            | Op::Recurse { .. }
            | Op::RecordGcSafepoint
            | Op::PushExceptionHandler { .. }
            | Op::PushResumableExceptionHandler { .. }
            | Op::PopExceptionHandler { .. }
            | Op::PopExceptionHandlerById { .. }
            | Op::PushPromptHandler { .. }
            | Op::PopPromptHandler { .. }
            | Op::PushPromptTag { .. }
            | Op::CaptureContinuation { .. }
            | Op::CaptureContinuationTagged { .. }
            | Op::PerformEffect { .. }
            | Op::ReturnFromShift { .. }
    )
}

/// True if this terminator can transfer control to the GC.
/// `InlineBranch::InlineBumpAllocate` bails to the allocator call on
/// overflow; `Throw` calls the exception runtime.
pub fn is_gc_safepoint_terminator(term: &Terminator) -> bool {
    match term {
        Terminator::InlineBranch { op, .. } => {
            matches!(op, InlineBranchOp::InlineBumpAllocate { .. })
        }
        Terminator::Throw { .. } => true,
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    NoStore,
    /// `StorePending(false)`: SlotStore happened, no safepoint between
    /// it and now. `StorePending(true)`: SlotStore happened AND a
    /// safepoint happened — a SlotLoad reached in this state makes
    /// the slot unsafe to promote.
    StorePending(bool),
}

impl State {
    fn lub(self, other: State) -> State {
        match (self, other) {
            (State::NoStore, x) | (x, State::NoStore) => x,
            (State::StorePending(a), State::StorePending(b)) => State::StorePending(a | b),
        }
    }
}

/// Per-slot CFG-wide dataflow: returns `true` iff promoting `slot`
/// would not violate I9. Promotion is safe iff no execution path
/// from any `SlotStore(slot)` to any `SlotLoad(slot)` passes through
/// a GC-safepoint.
///
/// Conservative on irreducible / unreachable structure: any block
/// the worklist doesn't visit stays at `NoStore`, which means we
/// can't reach an UNSAFE conclusion through it. Combined with the
/// monotone LUB, this keeps the analysis sound.
pub fn slot_is_gc_safe_to_promote(f: &CfgFunction, slot: SlotId) -> bool {
    if f.blocks.is_empty() {
        return true;
    }
    // Use `Option<State>` so the lattice has a true bottom: an
    // unvisited block has `None`, which LUB-ed with `Some(NoStore)`
    // becomes `Some(NoStore)` (a change). Without this, the
    // worklist would stall on the trivial case where every block's
    // initial state is `NoStore` and the entry's exit state is also
    // `NoStore` — no change → no propagation → unreachable blocks
    // never visited, including the ones holding the actual
    // SlotLoad/SlotStore for this slot.
    let mut in_state: Vec<Option<State>> = vec![None; f.num_blocks()];
    let mut worklist: VecDeque<BlockId> = VecDeque::new();
    let mut in_worklist: HashSet<BlockId> = HashSet::new();
    in_state[f.entry.0 as usize] = Some(State::NoStore);
    worklist.push_back(f.entry);
    in_worklist.insert(f.entry);

    while let Some(bid) = worklist.pop_front() {
        in_worklist.remove(&bid);
        let block = f.block(bid);
        let mut state = in_state[bid.0 as usize].unwrap_or(State::NoStore);

        // Process body ops.
        for op in &block.body {
            match op {
                Op::SlotStore { slot: s, .. } if *s == slot => {
                    state = State::StorePending(false);
                }
                Op::SlotLoad { slot: s, .. } if *s == slot => {
                    if let State::StorePending(true) = state {
                        // A safepoint passed since the store that
                        // produced this load's value — UNSAFE.
                        return false;
                    }
                }
                _ => {
                    if is_gc_safepoint_op(op)
                        && let State::StorePending(false) = state
                    {
                        state = State::StorePending(true);
                    }
                }
            }
        }
        // Process terminator-as-safepoint (InlineBumpAllocate, Throw).
        if is_gc_safepoint_terminator(&block.terminator)
            && let State::StorePending(false) = state
        {
            state = State::StorePending(true);
        }

        // Propagate to successors. First visit always propagates
        // (old == None → new == Some(state) → change). Later visits
        // propagate only when LUB shows a genuine increase.
        for succ in block.terminator.successors() {
            let old = in_state[succ.0 as usize];
            let new = match old {
                None => Some(state),
                Some(o) => Some(o.lub(state)),
            };
            if new != old {
                in_state[succ.0 as usize] = new;
                if in_worklist.insert(succ) {
                    worklist.push_back(succ);
                }
            }
        }
    }
    true
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CallTarget, CfgFunction, ClobberSet, Op, RegClass, SlotId, Terminator};

    /// Single block: SlotStore, then SlotLoad, no call between.
    /// Promotion is safe.
    #[test]
    fn store_then_load_no_safepoint_is_safe() {
        let mut f = CfgFunction::new(Some("safe".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        let dst = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: v,
        });
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: dst };
        assert!(slot_is_gc_safe_to_promote(&f, SlotId(0)));
    }

    /// SlotStore, Call, SlotLoad — UNSAFE. The promoted value would
    /// be in a register across the call; GC would miss it.
    #[test]
    fn store_call_load_is_unsafe() {
        let mut f = CfgFunction::new(Some("unsafe".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        let call_dst = f.new_vreg(RegClass::Gp);
        let dst = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: v,
        });
        f.block_mut(entry).body.push(Op::Call {
            dst: call_dst,
            target: CallTarget::Pointer(0x1000),
            args: vec![],
            is_builtin: true,
            clobbers: ClobberSet::AllCallerSaved,
        });
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: dst };
        assert!(!slot_is_gc_safe_to_promote(&f, SlotId(0)));
    }

    /// SlotStore, Call BEFORE the store, SlotLoad — safe. The call
    /// doesn't cross any live range of the promoted value.
    #[test]
    fn call_before_store_is_safe() {
        let mut f = CfgFunction::new(Some("call_first".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        let call_dst = f.new_vreg(RegClass::Gp);
        let dst = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::Call {
            dst: call_dst,
            target: CallTarget::Pointer(0x1000),
            args: vec![],
            is_builtin: true,
            clobbers: ClobberSet::AllCallerSaved,
        });
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: v,
        });
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: dst };
        assert!(slot_is_gc_safe_to_promote(&f, SlotId(0)));
    }

    /// Diamond: SlotStore in entry, SlotLoad in join, Call in only
    /// ONE of the branches. UNSAFE — there's a path that crosses
    /// the call between store and load.
    #[test]
    fn call_on_one_branch_is_unsafe() {
        let mut f = CfgFunction::new(Some("diamond".into()), 1);
        let entry = f.new_block();
        let then_b = f.new_block();
        let else_b = f.new_block();
        let join = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        let cond = f.new_vreg(RegClass::Gp);
        let call_dst = f.new_vreg(RegClass::Gp);
        let dst = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: v,
        });
        f.block_mut(entry).terminator = Terminator::Branch {
            cond: crate::ir::Condition::Equal,
            lhs: cond,
            rhs: cond,
            t_target: then_b,
            t_args: vec![],
            f_target: else_b,
            f_args: vec![],
        };
        f.block_mut(then_b).body.push(Op::Call {
            dst: call_dst,
            target: CallTarget::Pointer(0x1000),
            args: vec![],
            is_builtin: true,
            clobbers: ClobberSet::AllCallerSaved,
        });
        f.block_mut(then_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        f.block_mut(else_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        f.block_mut(join).body.push(Op::SlotLoad {
            dst,
            slot: SlotId(0),
        });
        f.block_mut(join).terminator = Terminator::Ret { value: dst };
        f.block_mut(then_b).predecessors.push(entry);
        f.block_mut(else_b).predecessors.push(entry);
        f.block_mut(join).predecessors.push(then_b);
        f.block_mut(join).predecessors.push(else_b);

        assert!(!slot_is_gc_safe_to_promote(&f, SlotId(0)));
    }

    /// Loop with a Call in the body; SlotStore before the loop,
    /// SlotLoad in the loop after the Call. The store reaches the
    /// load along the back-edge with a call in between — UNSAFE.
    #[test]
    fn loop_with_call_is_unsafe() {
        let mut f = CfgFunction::new(Some("loop_call".into()), 1);
        let entry = f.new_block();
        let header = f.new_block();
        let body = f.new_block();
        let exit = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        let cond = f.new_vreg(RegClass::Gp);
        let call_dst = f.new_vreg(RegClass::Gp);
        let dst = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: v,
        });
        f.block_mut(entry).terminator = Terminator::Jump {
            target: header,
            args: vec![],
        };
        f.block_mut(header).terminator = Terminator::Branch {
            cond: crate::ir::Condition::Equal,
            lhs: cond,
            rhs: cond,
            t_target: body,
            t_args: vec![],
            f_target: exit,
            f_args: vec![],
        };
        f.block_mut(body).body.push(Op::Call {
            dst: call_dst,
            target: CallTarget::Pointer(0x1000),
            args: vec![],
            is_builtin: true,
            clobbers: ClobberSet::AllCallerSaved,
        });
        f.block_mut(body).body.push(Op::SlotLoad {
            dst,
            slot: SlotId(0),
        });
        f.block_mut(body).terminator = Terminator::Jump {
            target: header,
            args: vec![],
        };
        f.block_mut(exit).terminator = Terminator::Ret { value: dst };
        f.block_mut(header).predecessors.push(entry);
        f.block_mut(header).predecessors.push(body);
        f.block_mut(body).predecessors.push(header);
        f.block_mut(exit).predecessors.push(header);

        assert!(!slot_is_gc_safe_to_promote(&f, SlotId(0)));
    }
}
