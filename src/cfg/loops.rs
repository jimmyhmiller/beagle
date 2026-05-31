//! Natural-loop detection over the CFG.
//!
//! A back-edge is a CFG edge `latch -> header` whose target (`header`)
//! dominates its source (`latch`). The natural loop of that back-edge is
//! the header plus every block that can reach the latch without passing
//! through the header. Back-edges that share a header are merged into one
//! loop with multiple latches.
//!
//! Pure analysis over `terminator.successors()` — normal CFG edges only;
//! soft handler/resume edges don't form loops. Reusable by any loop-aware
//! pass (the planned float loop-body versioning, future LICM / unrolling).

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::cfg::dom::{compute_idoms, dominates, reverse_postorder};
use crate::cfg::{BlockId, CfgFunction};

/// One natural loop, keyed by its header.
#[derive(Debug, Clone)]
pub struct NaturalLoop {
    /// The loop header — the back-edge target; dominates the whole body.
    pub header: BlockId,
    /// Every block in the loop, including the header and the latches.
    pub body: HashSet<BlockId>,
    /// Latch blocks: the sources of back-edges to `header` (sorted, deduped).
    pub latches: Vec<BlockId>,
}

impl NaturalLoop {
    pub fn contains(&self, b: BlockId) -> bool {
        self.body.contains(&b)
    }
}

/// Predecessor map built from terminator successors (normal CFG edges).
/// Self-contained so loop detection doesn't depend on `Block.predecessors`
/// being populated/current.
fn predecessor_map(f: &CfgFunction) -> HashMap<BlockId, Vec<BlockId>> {
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for (idx, block) in f.blocks.iter().enumerate() {
        let b = BlockId(idx as u32);
        for s in block.terminator.successors() {
            preds.entry(s).or_default().push(b);
        }
    }
    preds
}

/// Detect natural loops. Returns one `NaturalLoop` per loop header, ordered
/// by header id. Nested loops appear as separate entries; an inner loop's
/// `body` is a subset of the enclosing loop's `body`.
pub fn natural_loops(f: &CfgFunction) -> Vec<NaturalLoop> {
    if f.blocks.is_empty() {
        return Vec::new();
    }
    let rpo = reverse_postorder(f);
    let idom = compute_idoms(f, &rpo);
    let preds = predecessor_map(f);
    let reachable: HashSet<BlockId> = rpo.iter().copied().collect();

    // Group back-edge latches by header. `latch -> header` is a back-edge
    // iff `header` dominates `latch`.
    let mut header_latches: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for &b in &rpo {
        for s in f.block(b).terminator.successors() {
            if reachable.contains(&s) && dominates(&idom, s, b) {
                header_latches.entry(s).or_default().push(b);
            }
        }
    }

    let mut loops = Vec::new();
    for (header, mut latches) in header_latches {
        // Natural-loop body (Dragon book): start with {header, latches},
        // walk predecessors backward, stopping at the header (its preds are
        // never traversed), collecting every block that reaches a latch.
        let mut body: HashSet<BlockId> = HashSet::new();
        body.insert(header);
        let mut stack: Vec<BlockId> = Vec::new();
        for &latch in &latches {
            if body.insert(latch) {
                stack.push(latch);
            }
        }
        while let Some(d) = stack.pop() {
            if let Some(ps) = preds.get(&d) {
                for &p in ps {
                    if body.insert(p) {
                        stack.push(p);
                    }
                }
            }
        }
        latches.sort();
        latches.dedup();
        loops.push(NaturalLoop {
            header,
            body,
            latches,
        });
    }
    loops.sort_by_key(|l| l.header.0);
    loops
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CfgFunction, Op, RegClass, SlotId, Terminator};
    use crate::ir::Condition;

    /// Helper: a block that branches on a dummy condition to two targets.
    fn set_branch(
        f: &mut CfgFunction,
        b: BlockId,
        t: BlockId,
        e: BlockId,
        cond_v: crate::cfg::VReg,
    ) {
        f.block_mut(b).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: cond_v,
            rhs: cond_v,
            t_target: t,
            t_args: vec![],
            f_target: e,
            f_args: vec![],
        };
    }
    fn set_jump(f: &mut CfgFunction, b: BlockId, t: BlockId) {
        f.block_mut(b).terminator = Terminator::Jump {
            target: t,
            args: vec![],
        };
    }
    /// Populate every block's `predecessors` from terminator successors —
    /// the real CFG builder does this; hand-built test CFGs must too, since
    /// `compute_idoms` reads the field.
    fn finalize_preds(f: &mut CfgFunction) {
        let n = f.blocks.len();
        for i in 0..n {
            f.blocks[i].predecessors.clear();
        }
        for i in 0..n {
            let succs = f.blocks[i].terminator.successors();
            for s in succs {
                f.block_mut(s).predecessors.push(BlockId(i as u32));
            }
        }
    }

    fn headers(loops: &[NaturalLoop]) -> Vec<u32> {
        loops.iter().map(|l| l.header.0).collect()
    }

    #[test]
    fn no_loop_diamond_has_none() {
        // entry -> then/else -> join. No back-edge.
        let mut f = CfgFunction::new(Some("d".into()), 0);
        let entry = f.new_block();
        let then_b = f.new_block();
        let else_b = f.new_block();
        let join = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v,
            slot: SlotId(0),
        });
        set_branch(&mut f, entry, then_b, else_b, v);
        set_jump(&mut f, then_b, join);
        set_jump(&mut f, else_b, join);
        f.block_mut(join).terminator = Terminator::Ret { value: v };
        finalize_preds(&mut f);
        assert!(natural_loops(&f).is_empty());
    }

    #[test]
    fn simple_loop() {
        // entry(0) -> header(1); header -> body(2) | exit(3); body -> header.
        let mut f = CfgFunction::new(Some("s".into()), 0);
        let entry = f.new_block();
        let header = f.new_block();
        let body = f.new_block();
        let exit = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v,
            slot: SlotId(0),
        });
        set_jump(&mut f, entry, header);
        set_branch(&mut f, header, body, exit, v);
        set_jump(&mut f, body, header); // back-edge
        f.block_mut(exit).terminator = Terminator::Ret { value: v };

        finalize_preds(&mut f);
        let loops = natural_loops(&f);
        assert_eq!(headers(&loops), vec![1]);
        let l = &loops[0];
        assert_eq!(l.header, header);
        assert_eq!(l.latches, vec![body]);
        let mut got: Vec<u32> = l.body.iter().map(|b| b.0).collect();
        got.sort();
        assert_eq!(got, vec![1, 2]); // header + body, not entry/exit
    }

    #[test]
    fn self_loop() {
        // entry -> h; h -> h (self back-edge) | exit.
        let mut f = CfgFunction::new(Some("self".into()), 0);
        let entry = f.new_block();
        let h = f.new_block();
        let exit = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v,
            slot: SlotId(0),
        });
        set_jump(&mut f, entry, h);
        set_branch(&mut f, h, h, exit, v); // self back-edge
        f.block_mut(exit).terminator = Terminator::Ret { value: v };

        finalize_preds(&mut f);
        let loops = natural_loops(&f);
        assert_eq!(headers(&loops), vec![1]);
        assert_eq!(loops[0].latches, vec![h]);
        assert_eq!(
            loops[0].body.iter().map(|b| b.0).collect::<Vec<_>>(),
            vec![1]
        );
    }

    #[test]
    fn multiple_latches_one_header() {
        // entry -> h; h -> a | b; a -> h; b -> h. Two back-edges, one header.
        let mut f = CfgFunction::new(Some("m".into()), 0);
        let entry = f.new_block();
        let h = f.new_block();
        let a = f.new_block();
        let b = f.new_block();
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v,
            slot: SlotId(0),
        });
        set_jump(&mut f, entry, h);
        set_branch(&mut f, h, a, b, v);
        set_jump(&mut f, a, h); // back-edge 1
        set_jump(&mut f, b, h); // back-edge 2
        // h needs a real exit too, else it's an infinite loop with no Ret;
        // that's fine for loop detection, but give the function a ret block
        // reachable so reverse_postorder is well-formed. Re-point b's branch?
        // Simpler: leave as is — both a and b loop back; detection only needs
        // the back-edges. (No Ret is acceptable for this analysis-only test.)

        finalize_preds(&mut f);
        let loops = natural_loops(&f);
        assert_eq!(headers(&loops), vec![1]);
        assert_eq!(loops[0].latches, vec![a, b]);
        let mut got: Vec<u32> = loops[0].body.iter().map(|x| x.0).collect();
        got.sort();
        assert_eq!(got, vec![1, 2, 3]);
    }

    #[test]
    fn nested_loops() {
        // entry(0) -> outer(1); outer -> inner(2) | exit(4);
        // inner -> inner_body(3) | outer(back to 1);
        // inner_body -> inner (back to 2). Inner loop {2,3}, outer {1,2,3}.
        let mut f = CfgFunction::new(Some("n".into()), 0);
        let entry = f.new_block(); // 0
        let outer = f.new_block(); // 1
        let inner = f.new_block(); // 2
        let inner_body = f.new_block(); // 3
        let exit = f.new_block(); // 4
        f.entry = entry;
        let v = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: v,
            slot: SlotId(0),
        });
        set_jump(&mut f, entry, outer);
        set_branch(&mut f, outer, inner, exit, v);
        set_branch(&mut f, inner, inner_body, outer, v); // inner->body or back to outer
        set_jump(&mut f, inner_body, inner); // inner back-edge
        f.block_mut(exit).terminator = Terminator::Ret { value: v };

        finalize_preds(&mut f);
        let loops = natural_loops(&f);
        // Two headers: outer(1) and inner(2).
        assert_eq!(headers(&loops), vec![1, 2]);
        let outer_loop = loops.iter().find(|l| l.header == outer).unwrap();
        let inner_loop = loops.iter().find(|l| l.header == inner).unwrap();
        let mut ob: Vec<u32> = outer_loop.body.iter().map(|b| b.0).collect();
        ob.sort();
        assert_eq!(ob, vec![1, 2, 3]);
        let mut ib: Vec<u32> = inner_loop.body.iter().map(|b| b.0).collect();
        ib.sort();
        assert_eq!(ib, vec![2, 3]);
        // Inner body is a subset of outer body.
        assert!(inner_loop.body.is_subset(&outer_loop.body));
    }
}
