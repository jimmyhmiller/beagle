//! Phase 4e — edge resolution for the chordal-coloring regalloc.
//!
//! After coloring assigns each VReg a color, block-param transfers on
//! each control-flow edge become "parallel copies" between physical
//! registers (colors). E.g. an edge `block_a → block_b(p1, p2, p3)`
//! with terminator args `[v1, v2, v3]` from `block_a` becomes three
//! simultaneous moves: `color(p1) ← color(v1)`, `color(p2) ← color(v2)`,
//! `color(p3) ← color(v3)`.
//!
//! "Simultaneously" matters: if two transfers form a cycle (e.g. r2
//! ← r3, r3 ← r2 — a swap), naively sequentialising clobbers the
//! source of the second move. The standard fix (Briggs/Sreedhar):
//!
//! 1. Greedily emit moves whose destination isn't anyone else's
//!    source (= "ready").
//! 2. When no ready move remains, the rest must form one or more
//!    cycles. Break one cycle by saving the "victim" location to a
//!    scratch register, renaming any pending source that matched that
//!    location to the scratch, and resuming step 1.
//!
//! Per **I2** + **I3** the spec already split critical edges, so
//! every edge here goes from a single-successor source to a single-
//! predecessor target (after Phase 1c). Parallel-copy semantics are
//! well-defined on these edges with no ambiguity.
//!
//! This module is pure analysis — it returns a `Vec<EdgeTransfers>`
//! that the emit step (Phase 4f) interleaves into the source block's
//! body before the terminator. The CFG itself is not mutated.

#![allow(dead_code)]

use crate::cfg::regalloc::color::Coloring;
use crate::cfg::{BlockId, CfgFunction, RegClass, Terminator, VReg};

/// One physical-register move to be emitted on an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysMove {
    pub src_color: u32,
    pub dst_color: u32,
    pub class: RegClass,
}

/// All the moves to emit on a single edge `from → to`.
#[derive(Debug, Clone)]
pub struct EdgeTransfers {
    pub from: BlockId,
    pub to: BlockId,
    pub moves: Vec<PhysMove>,
}

/// Colors reserved as scratch for cycle-breaking. Caller must
/// guarantee these are not assigned to any normal VReg (typically
/// reserved physical registers above the regalloc budget).
#[derive(Debug, Clone, Copy)]
pub struct Scratch {
    pub gp: u32,
    pub fp: u32,
}

impl Scratch {
    pub fn for_class(&self, c: RegClass) -> u32 {
        match c {
            RegClass::Gp => self.gp,
            RegClass::Fp => self.fp,
        }
    }
}

/// For every edge that has block-param transfers, compute the
/// serialized parallel-copy sequence. Edges with no transfers (no
/// block params, or all transfers are no-ops `r ← r`) are omitted.
pub fn resolve_edges(f: &CfgFunction, coloring: &Coloring, scratch: Scratch) -> Vec<EdgeTransfers> {
    let mut out = Vec::new();
    for (idx, block) in f.blocks.iter().enumerate() {
        let from = BlockId(idx as u32);
        for succ in block.terminator.successors() {
            let target_params = &f.block(succ).params;
            if target_params.is_empty() {
                continue;
            }
            let args = terminator_args_for_succ(&block.terminator, succ);
            // Pair args with target params; map to color transfers.
            let transfers: Vec<(u32, u32, RegClass)> = args
                .iter()
                .zip(target_params.iter())
                .filter_map(|(arg, param)| {
                    let src = coloring.color_of(*arg);
                    let dst = coloring.color_of(*param);
                    if src == dst {
                        None // no-op
                    } else {
                        Some((src, dst, param.class))
                    }
                })
                .collect();
            if transfers.is_empty() {
                continue;
            }
            let moves = serialize_parallel_copy(transfers, scratch);
            if !moves.is_empty() {
                out.push(EdgeTransfers {
                    from,
                    to: succ,
                    moves,
                });
            }
        }
    }
    out
}

/// Serialize parallel-copy transfers, breaking cycles via scratch
/// registers. Filters out self-moves before processing. Asserts a
/// successful exit (panics if the algorithm gets stuck, which would
/// indicate a bug rather than a real input).
pub fn serialize_parallel_copy(
    transfers: Vec<(u32, u32, RegClass)>,
    scratch: Scratch,
) -> Vec<PhysMove> {
    let mut pending: Vec<(u32, u32, RegClass)> =
        transfers.into_iter().filter(|(s, d, _)| s != d).collect();
    let mut emitted: Vec<PhysMove> = Vec::new();

    while !pending.is_empty() {
        // Try to find a "ready" move: dst isn't a src for any other
        // pending move.
        let ready = pending
            .iter()
            .position(|(_, dst, _)| !pending.iter().any(|(s, _, _)| s == dst));
        match ready {
            Some(i) => {
                let (s, d, c) = pending.remove(i);
                emitted.push(PhysMove {
                    src_color: s,
                    dst_color: d,
                    class: c,
                });
            }
            None => {
                // No ready move → at least one cycle exists. Break
                // it by saving the first pending move's destination
                // to scratch, then renaming any pending source that
                // matched that destination to the scratch.
                let (_s0, d0, c) = pending[0];
                let scratch_color = scratch.for_class(c);
                emitted.push(PhysMove {
                    src_color: d0,
                    dst_color: scratch_color,
                    class: c,
                });
                for (src, _, _) in pending.iter_mut() {
                    if *src == d0 {
                        *src = scratch_color;
                    }
                }
                // After renaming, (s0, d0) becomes ready on the next
                // iteration (d0 is no longer a source for anyone).
            }
        }
    }

    emitted
}

fn terminator_args_for_succ(term: &Terminator, succ: BlockId) -> Vec<VReg> {
    match term {
        Terminator::Jump { target, args } if *target == succ => args.clone(),
        Terminator::Branch {
            t_target,
            t_args,
            f_target,
            f_args,
            ..
        } => {
            if *t_target == succ {
                t_args.clone()
            } else if *f_target == succ {
                f_args.clone()
            } else {
                Vec::new()
            }
        }
        Terminator::InlineBranch {
            fall_through,
            fall_args,
            bail,
            bail_args,
            ..
        } => {
            if *fall_through == succ {
                fall_args.clone()
            } else if *bail == succ {
                bail_args.clone()
            } else {
                Vec::new()
            }
        }
        Terminator::Throw {
            resume,
            resume_args,
            ..
        } if *resume == succ => resume_args.clone(),
        _ => Vec::new(),
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: u32, d: u32) -> (u32, u32, RegClass) {
        (s, d, RegClass::Gp)
    }

    fn scratch() -> Scratch {
        Scratch { gp: 99, fp: 199 }
    }

    /// No transfers → no moves.
    #[test]
    fn empty_transfers_no_moves() {
        let moves = serialize_parallel_copy(vec![], scratch());
        assert!(moves.is_empty());
    }

    /// Self-moves are filtered out.
    #[test]
    fn self_moves_dropped() {
        let moves = serialize_parallel_copy(vec![t(1, 1), t(2, 2)], scratch());
        assert!(moves.is_empty());
    }

    /// Linear chain `1 → 2 → 3` serializes in reverse order (write
    /// the leaf first so the prior source isn't clobbered).
    #[test]
    fn linear_chain() {
        let moves = serialize_parallel_copy(vec![t(1, 2), t(2, 3)], scratch());
        assert_eq!(moves.len(), 2);
        // First emit: dst=3 (ready — no one uses 3 as src).
        assert_eq!(moves[0].dst_color, 3);
        assert_eq!(moves[0].src_color, 2);
        // Second: dst=2.
        assert_eq!(moves[1].dst_color, 2);
        assert_eq!(moves[1].src_color, 1);
    }

    /// Swap cycle `1 ↔ 2`: save one to scratch, swap, restore.
    #[test]
    fn swap_cycle_uses_scratch() {
        let moves = serialize_parallel_copy(vec![t(1, 2), t(2, 1)], scratch());
        assert_eq!(moves.len(), 3, "swap costs 3 moves");
        // Expected: save 2 → scratch, 2 ← 1, 1 ← scratch.
        assert_eq!(moves[0].dst_color, 99, "first move saves to scratch");
        assert_eq!(moves[0].src_color, 2);
        assert_eq!(moves[1].dst_color, 2);
        assert_eq!(moves[1].src_color, 1);
        assert_eq!(moves[2].dst_color, 1);
        assert_eq!(moves[2].src_color, 99, "last move restores from scratch");
    }

    /// 3-cycle a → b, b → c, c → a (transfer src → dst): save one,
    /// chain the rest, restore.
    #[test]
    fn three_cycle_uses_scratch() {
        // Transfers: (1→2), (2→3), (3→1) means
        //   dst=2 ← src=1; dst=3 ← src=2; dst=1 ← src=3.
        // Cycle in src→dst direction: 1→2→3→1.
        let moves = serialize_parallel_copy(vec![t(1, 2), t(2, 3), t(3, 1)], scratch());
        assert_eq!(moves.len(), 4, "3-cycle costs 4 moves");
        // First: save d0 (= 2) to scratch.
        assert_eq!(moves[0].dst_color, 99);
        assert_eq!(moves[0].src_color, 2);
        // Rest: write the chain, with the "consumer of 2's old value"
        // pulling from scratch instead.
        // After scratch save: pending becomes (1,2), (scratch,3), (3,1).
        // Ready: (1,2)? dst=2, src=2 used by no one (we renamed). YES.
        //   emit (1,2).
        // Then (scratch,3)? dst=3, src=3 used by (3,1). NOT ready.
        //   (3,1)? dst=1, src=1 used by no one. READY. emit.
        // Then (scratch,3): READY. emit.
        // So order: [save, 1→2, 3→1, scratch→3]. Verify just the
        // last: must write 3 from scratch.
        let last = moves.last().unwrap();
        assert_eq!(last.dst_color, 3);
        assert_eq!(last.src_color, 99);
    }

    /// Two disjoint chains process independently.
    #[test]
    fn disjoint_chains_serialize_in_order() {
        // (1→2), (3→4) — independent moves; each ready immediately.
        let moves = serialize_parallel_copy(vec![t(1, 2), t(3, 4)], scratch());
        assert_eq!(moves.len(), 2);
        // Both should be straight moves, no scratch involved.
        for m in &moves {
            assert_ne!(m.dst_color, 99);
            assert_ne!(m.src_color, 99);
        }
    }

    /// resolve_edges integration: a function with a Jump-with-arg gets
    /// a single move on the edge.
    #[test]
    fn resolve_edges_simple_jump() {
        use crate::cfg::regalloc::color::color as color_fn;
        use crate::cfg::regalloc::interference::build_interference;
        use crate::cfg::regalloc::liveness::compute_liveness;
        use crate::cfg::{CfgFunction, Op, Terminator};

        let mut f = CfgFunction::new(Some("edge".into()), 0);
        let entry = f.new_block();
        let tail = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let p = f.new_vreg(RegClass::Gp);
        let r = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(a);
        // Compute something so a is live until terminator.
        f.block_mut(entry).body.push(Op::AddInt {
            dst: r,
            lhs: a,
            rhs: a,
        });
        f.block_mut(entry).terminator = Terminator::Jump {
            target: tail,
            args: vec![r],
        };
        f.block_mut(tail).params.push(p);
        f.block_mut(tail).terminator = Terminator::Ret { value: p };
        f.block_mut(tail).predecessors.push(entry);

        let liveness = compute_liveness(&f);
        let ig = build_interference(&f, &liveness);
        let coloring = color_fn(&f, &ig);
        let edges = resolve_edges(&f, &coloring, scratch());
        // The single edge entry → tail has one block-param transfer
        // (r → p). If they happen to share a color, the transfer is a
        // no-op and no edge entry is emitted; otherwise one move.
        if coloring.color_of(r) == coloring.color_of(p) {
            assert!(edges.is_empty(), "same color → no-op");
        } else {
            assert_eq!(edges.len(), 1);
            assert_eq!(edges[0].from, entry);
            assert_eq!(edges[0].to, tail);
            assert_eq!(edges[0].moves.len(), 1);
            assert_eq!(edges[0].moves[0].dst_color, coloring.color_of(p));
            assert_eq!(edges[0].moves[0].src_color, coloring.color_of(r));
        }
    }
}
