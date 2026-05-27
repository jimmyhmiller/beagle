//! Per-block live-in / live-out for the SSA pipeline.
//!
//! Standard backward dataflow with a worklist. The data:
//!
//! - `live_out[B]` = union over successors `S` of `live_in[S]`.
//! - `live_in[B]` = `gen_set[B] ∪ (live_out[B] - kill[B])`.
//!
//! Where:
//!
//! - `gen_set[B]` is every VReg used in B (body op uses + terminator uses,
//!   including outgoing block-param args) whose def is NOT inside B.
//! - `kill[B]` is every VReg defined in B (block params + body op defs
//!   + terminator def, if any).
//!
//! Block-param semantics:
//!
//! - A block param P of B is a def at B's entry. It's in `kill[B]`,
//!   not in `live_in[B]`.
//! - The corresponding outgoing arg A passed by a predecessor's
//!   terminator is a use at the terminator — it's in `gen_set[pred]`
//!   (unless A is locally killed before the terminator, which doesn't
//!   happen in valid SSA), so it propagates into `live_in[pred]`.
//! - This carries the "value must be available for the handoff"
//!   constraint backward through the predecessor chain.
//!
//! Convergen_setce: worklist-based, processes predecessors when a block's
//! live sets change. Linear in IR size for most programs; loops may
//! require O(loop-depth) iterations.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

use crate::cfg::{BlockId, CfgFunction, VReg};

/// Liveness result. Both maps are keyed by every block in the
/// function (including unreachable ones, which get empty sets).
#[derive(Debug, Clone)]
pub struct Liveness {
    pub live_in: HashMap<BlockId, HashSet<VReg>>,
    pub live_out: HashMap<BlockId, HashSet<VReg>>,
}

impl Liveness {
    pub fn live_in(&self, b: BlockId) -> &HashSet<VReg> {
        self.live_in.get(&b).expect("block must be in liveness")
    }
    pub fn live_out(&self, b: BlockId) -> &HashSet<VReg> {
        self.live_out.get(&b).expect("block must be in liveness")
    }
}

pub fn compute_liveness(f: &CfgFunction) -> Liveness {
    let num_blocks = f.blocks.len();
    let mut live_in: HashMap<BlockId, HashSet<VReg>> = HashMap::with_capacity(num_blocks);
    let mut live_out: HashMap<BlockId, HashSet<VReg>> = HashMap::with_capacity(num_blocks);
    for i in 0..num_blocks {
        live_in.insert(BlockId(i as u32), HashSet::new());
        live_out.insert(BlockId(i as u32), HashSet::new());
    }

    // Compute gen_set / kill per block once. They don't change as
    // dataflow iterates — only live_in / live_out do.
    let mut gen_set: HashMap<BlockId, HashSet<VReg>> = HashMap::with_capacity(num_blocks);
    let mut kill: HashMap<BlockId, HashSet<VReg>> = HashMap::with_capacity(num_blocks);
    for (idx, block) in f.blocks.iter().enumerate() {
        let bid = BlockId(idx as u32);
        let mut local_defs: HashSet<VReg> = HashSet::new();
        let mut block_gen_set: HashSet<VReg> = HashSet::new();
        // Block params are defined at the very top of the block.
        for &p in &block.params {
            local_defs.insert(p);
        }
        // Body ops: walk forward. Uses of VRegs not yet defined
        // locally are "gen_set"; defs add to local_defs (which "kills"
        // anything live coming in).
        for op in &block.body {
            for u in op.uses() {
                if !local_defs.contains(&u) {
                    block_gen_set.insert(u);
                }
            }
            for d in op.defs() {
                local_defs.insert(d);
            }
        }
        // Terminator: uses (including outgoing block-param args) are
        // at the end of the block — same treatment as body uses.
        for u in block.terminator.uses() {
            if !local_defs.contains(&u) {
                block_gen_set.insert(u);
            }
        }
        for d in block.terminator.defs() {
            local_defs.insert(d);
        }
        gen_set.insert(bid, block_gen_set);
        kill.insert(bid, local_defs);
    }

    // Worklist of blocks to (re-)process. Initialize with every block.
    let mut worklist: VecDeque<BlockId> = (0..num_blocks).map(|i| BlockId(i as u32)).collect();
    let mut in_worklist: HashSet<BlockId> = worklist.iter().copied().collect();

    while let Some(bid) = worklist.pop_front() {
        in_worklist.remove(&bid);
        let block = f.block(bid);

        // live_out[B] = union over successors of live_in[S].
        let mut new_live_out: HashSet<VReg> = HashSet::new();
        for s in block.terminator.successors() {
            for &v in live_in.get(&s).unwrap() {
                new_live_out.insert(v);
            }
        }

        // live_in[B] = gen_set[B] ∪ (live_out[B] - kill[B]).
        let mut new_live_in: HashSet<VReg> = gen_set[&bid].clone();
        for &v in &new_live_out {
            if !kill[&bid].contains(&v) {
                new_live_in.insert(v);
            }
        }

        let changed = new_live_in != live_in[&bid] || new_live_out != live_out[&bid];
        live_in.insert(bid, new_live_in);
        live_out.insert(bid, new_live_out);

        if changed {
            // Predecessors' live_out depends on our live_in — re-queue them.
            for pred in &block.predecessors {
                if in_worklist.insert(*pred) {
                    worklist.push_back(*pred);
                }
            }
        }
    }

    Liveness { live_in, live_out }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CfgFunction, Op, RegClass, Terminator};
    use crate::ir::Condition;

    /// `fn id(x) { x }` — entry has param `x`, returns it. live_in is
    /// empty (x is killed by being a block param); live_out is empty
    /// (no successors).
    #[test]
    fn identity_function_liveness() {
        let mut f = CfgFunction::new(Some("id".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let x = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(x);
        f.block_mut(entry).terminator = Terminator::Ret { value: x };

        let l = compute_liveness(&f);
        assert!(
            l.live_in(entry).is_empty(),
            "entry live_in empty (x is killed by param)"
        );
        assert!(
            l.live_out(entry).is_empty(),
            "entry live_out empty (no successors)"
        );
    }

    /// `fn use_after_def(x) { let y = x + x; y }` — entry has x as
    /// param, defines y, returns y. live_in / live_out both empty.
    #[test]
    fn use_after_def_liveness() {
        let mut f = CfgFunction::new(Some("uad".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let x = f.new_vreg(RegClass::Gp);
        let y = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(x);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: y,
            lhs: x,
            rhs: x,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: y };

        let l = compute_liveness(&f);
        assert!(l.live_in(entry).is_empty());
        assert!(l.live_out(entry).is_empty());
    }

    /// Cross-block value flow: entry computes a constant c, jumps to
    /// tail (no args), tail uses c. c must appear in live_out[entry]
    /// (it's needed by a successor) and in live_in[tail] (it's used
    /// there without being killed locally). This is the classical
    /// liveness-propagation case.
    #[test]
    fn cross_block_value_propagates() {
        let mut f = CfgFunction::new(Some("xblock".into()), 0);
        let entry = f.new_block();
        let tail = f.new_block();
        f.entry = entry;
        let c = f.new_vreg(RegClass::Gp);
        f.block_mut(entry)
            .body
            .push(Op::ConstTaggedInt { dst: c, value: 42 });
        f.block_mut(entry).terminator = Terminator::Jump {
            target: tail,
            args: vec![],
        };
        f.block_mut(tail).terminator = Terminator::Ret { value: c };
        f.block_mut(tail).predecessors.push(entry);

        let l = compute_liveness(&f);
        assert!(
            l.live_out(entry).contains(&c),
            "entry live_out carries c to tail"
        );
        assert!(l.live_in(tail).contains(&c), "tail live_in needs c");
    }

    /// Loop pattern: header reads loop var, body re-defines and back-
    /// jumps. The header's param is live across the loop body's
    /// outgoing edge; the body's new value (passed on the back-edge)
    /// must be in body's live_out.
    #[test]
    fn loop_carried_value_liveness() {
        // entry → header(phi: x)
        // header → body or exit
        // body → header(phi: y)
        let mut f = CfgFunction::new(Some("loop".into()), 0);
        let entry = f.new_block();
        let header = f.new_block();
        let body = f.new_block();
        let exit = f.new_block();
        f.entry = entry;

        let init = f.new_vreg(RegClass::Gp);
        let x = f.new_vreg(RegClass::Gp);
        let y = f.new_vreg(RegClass::Gp);
        let cmp = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: init,
            value: 0,
        });
        f.block_mut(entry).terminator = Terminator::Jump {
            target: header,
            args: vec![init],
        };
        f.block_mut(header).params.push(x);
        f.block_mut(header).body.push(Op::Compare {
            dst: cmp,
            lhs: x,
            rhs: x,
            cond: Condition::Equal,
        });
        f.block_mut(header).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: cmp,
            rhs: cmp,
            t_target: body,
            t_args: vec![],
            f_target: exit,
            f_args: vec![],
        };
        f.block_mut(body).body.push(Op::AddInt {
            dst: y,
            lhs: x,
            rhs: x,
        });
        f.block_mut(body).terminator = Terminator::Jump {
            target: header,
            args: vec![y],
        };
        f.block_mut(exit).terminator = Terminator::Ret { value: x };

        f.block_mut(header).predecessors.push(entry);
        f.block_mut(header).predecessors.push(body);
        f.block_mut(body).predecessors.push(header);
        f.block_mut(exit).predecessors.push(header);

        let l = compute_liveness(&f);
        // header's live_out carries x because both successors (body
        // and exit) need it. body reads x in AddInt; exit returns x.
        assert!(
            l.live_out(header).contains(&x),
            "x flows from header into both succs"
        );
        assert!(l.live_in(body).contains(&x), "body needs x at entry");
        assert!(l.live_in(exit).contains(&x), "exit needs x at entry");
        // body's live_out does NOT include x (header kills it via
        // param) or y (consumed by the back-edge handoff). Standard
        // live_out is post-terminator; per-instruction live sets
        // (computed by the interference walker in Phase 4b) will see
        // y as live AT body's terminator.
        assert!(!l.live_out(body).contains(&x), "x killed by header's param");
    }
}
