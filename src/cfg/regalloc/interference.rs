//! Build the SSA interference graph.
//!
//! Walks each block backward starting from `live_out[B]`. At each
//! instruction (terminator first, then body in reverse), every def
//! gets an interference edge to every currently-live VReg (except
//! itself); then `live` is updated to "before this instruction" by
//! removing defs and adding uses. Block params are defined at block
//! entry — they interfere with each other and with everything live at
//! block entry.
//!
//! Per Hack 2006, the interference graph of a program in SSA form is
//! chordal — every cycle of length ≥4 has a chord. Chordal graphs
//! admit a perfect elimination ordering (PEO); a reverse-dominator-
//! tree DFS produces one. Greedy graph-coloring in PEO order yields
//! an optimal coloring in polynomial time — that's what the next
//! sub-phase (4c, coloring) does. Spilling is only needed if the
//! maximum clique size at any program point exceeds the available
//! physical-register pool.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::cfg::regalloc::liveness::Liveness;
use crate::cfg::{BlockId, CfgFunction, VReg};

/// Symmetric interference graph: `adj[a]` contains every VReg that
/// interferes with `a`. Each edge is stored twice (once per endpoint).
#[derive(Debug, Clone, Default)]
pub struct InterferenceGraph {
    pub adj: HashMap<VReg, HashSet<VReg>>,
}

impl InterferenceGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a symmetric edge. No-op if `a == b` (a VReg doesn't
    /// interfere with itself).
    pub fn add_edge(&mut self, a: VReg, b: VReg) {
        if a == b {
            return;
        }
        self.adj.entry(a).or_default().insert(b);
        self.adj.entry(b).or_default().insert(a);
    }

    /// Ensure `v` is present as a node (with empty neighborhood) even
    /// if it has no interferences. Needed so the coloring pass sees
    /// every VReg.
    pub fn add_node(&mut self, v: VReg) {
        self.adj.entry(v).or_default();
    }

    pub fn interferes(&self, a: VReg, b: VReg) -> bool {
        self.adj.get(&a).map(|s| s.contains(&b)).unwrap_or(false)
    }

    pub fn neighbors(&self, v: VReg) -> impl Iterator<Item = VReg> + '_ {
        self.adj.get(&v).into_iter().flat_map(|s| s.iter().copied())
    }

    pub fn degree(&self, v: VReg) -> usize {
        self.adj.get(&v).map_or(0, |s| s.len())
    }

    pub fn nodes(&self) -> impl Iterator<Item = VReg> + '_ {
        self.adj.keys().copied()
    }

    pub fn num_nodes(&self) -> usize {
        self.adj.len()
    }

    /// Largest clique in the graph (= chromatic number for chordal
    /// graphs = minimum #colors needed). Naive O(N²); the coloring
    /// pass figures this out incrementally for free, so this method
    /// is for diagnostics only.
    pub fn max_clique_size_heuristic(&self) -> usize {
        // Quick lower bound: 1 + max degree. Tight for many real
        // graphs but not always.
        1 + self.adj.values().map(|s| s.len()).max().unwrap_or(0)
    }
}

/// Build the interference graph for `f` given precomputed liveness.
pub fn build_interference(f: &CfgFunction, liveness: &Liveness) -> InterferenceGraph {
    let mut g = InterferenceGraph::new();

    // Register every VReg as a node so the coloring pass doesn't miss
    // unconnected VRegs (e.g. a value defined and used in the same op
    // with nothing else live around it).
    for v in 0..f.num_vregs() {
        let class = f.vreg_classes[v];
        g.add_node(crate::cfg::VReg {
            index: v as u32,
            class,
        });
    }

    for (idx, block) in f.blocks.iter().enumerate() {
        let bid = BlockId(idx as u32);
        // Start at "live after the terminator" = live_out[B].
        let mut live: HashSet<VReg> = liveness.live_out(bid).clone();

        // Terminator: add edges from each def to everything currently
        // live, then "rewind" to before the terminator.
        for &d in &block.terminator.defs() {
            for &v in &live {
                g.add_edge(d, v);
            }
        }
        for &d in &block.terminator.defs() {
            live.remove(&d);
        }
        for u in block.terminator.uses() {
            live.insert(u);
        }

        // Body in reverse order.
        for op in block.body.iter().rev() {
            for &d in &op.defs() {
                for &v in &live {
                    g.add_edge(d, v);
                }
            }
            for &d in &op.defs() {
                live.remove(&d);
            }
            for u in op.uses() {
                live.insert(u);
            }
        }

        // Block params are all defined at block entry, simultaneously
        // with each other. Each param interferes with every other
        // param and with everything live at block entry (= the
        // current `live` set after the backward walk).
        for &p in &block.params {
            for &v in &live {
                g.add_edge(p, v);
            }
        }
        for i in 0..block.params.len() {
            for j in (i + 1)..block.params.len() {
                g.add_edge(block.params[i], block.params[j]);
            }
        }
    }

    g
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::regalloc::liveness::compute_liveness;
    use crate::cfg::{CfgFunction, Op, RegClass, Terminator};

    /// Two simultaneously-live params interfere (Chaitin convention A:
    /// values live at the same program point conflict for register
    /// assignment).
    #[test]
    fn simultaneously_live_params_interfere() {
        let mut f = CfgFunction::new(Some("two".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let b = f.new_vreg(RegClass::Gp);
        let r = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(a);
        f.block_mut(entry).params.push(b);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: r,
            lhs: a,
            rhs: b,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: r };

        let liveness = compute_liveness(&f);
        let g = build_interference(&f, &liveness);
        assert!(g.interferes(a, b), "a and b are both live at AddInt");
        // r does NOT interfere with a or b under convention A: a and b
        // are dead immediately after the AddInt (their last use was the
        // AddInt itself), so r can reuse one of their registers. This
        // is the whole point of three-operand SSA on a 2-operand ISA.
        assert!(
            !g.interferes(r, a),
            "r can reuse a's reg (a dead after use)"
        );
        assert!(
            !g.interferes(r, b),
            "r can reuse b's reg (b dead after use)"
        );
    }

    /// A def DOES interfere with anything live-across (live both before
    /// and after the def). Here `c` is live across `r = a + b` because
    /// it's used later by `Ret c`.
    #[test]
    fn def_interferes_with_live_across() {
        let mut f = CfgFunction::new(Some("across".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let b = f.new_vreg(RegClass::Gp);
        let c = f.new_vreg(RegClass::Gp);
        let r = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(a);
        f.block_mut(entry).params.push(b);
        f.block_mut(entry).params.push(c);
        // r = a + b consumes a, b; c is still live (used by Ret below).
        f.block_mut(entry).body.push(Op::AddInt {
            dst: r,
            lhs: a,
            rhs: b,
        });
        // Use both r and c in a single op to keep r live alongside c.
        let s = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: s,
            lhs: r,
            rhs: c,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: s };

        let liveness = compute_liveness(&f);
        let g = build_interference(&f, &liveness);
        // r is defined while c is live-across — must use a different
        // register than c.
        assert!(g.interferes(r, c), "r's def conflicts with c (live-across)");
        // All three params interfere pairwise (all live at block entry).
        assert!(g.interferes(a, b));
        assert!(g.interferes(a, c));
        assert!(g.interferes(b, c));
    }

    /// A value dead before another's def doesn't interfere.
    #[test]
    fn sequentially_dead_values_dont_interfere() {
        let mut f = CfgFunction::new(Some("seq".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let b = f.new_vreg(RegClass::Gp);
        let c = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(a);
        // b = a + a; a dead after.
        f.block_mut(entry).body.push(Op::AddInt {
            dst: b,
            lhs: a,
            rhs: a,
        });
        // c = b + b; b dead after.
        f.block_mut(entry).body.push(Op::AddInt {
            dst: c,
            lhs: b,
            rhs: b,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: c };

        let liveness = compute_liveness(&f);
        let g = build_interference(&f, &liveness);
        // a and c are never simultaneously live.
        assert!(!g.interferes(a, c), "a dead long before c is defined");
        // a and b: a is dead at b's def site (under convention A —
        // both are uses-then-defs at the same instruction).
        assert!(
            !g.interferes(a, b),
            "a's last use is the AddInt that defines b"
        );
        assert!(
            !g.interferes(b, c),
            "b's last use is the AddInt that defines c"
        );
    }
}
