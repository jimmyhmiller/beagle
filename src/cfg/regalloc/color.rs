//! Greedy graph coloring over the SSA interference graph.
//!
//! SSA interference graphs are chordal (Hack 2006). A perfect
//! elimination ordering (PEO) exists; greedy assignment in PEO order
//! is optimal. Reverse dominator-tree DFS gives a valid PEO — the
//! easiest way to compute one for an SSA function.
//!
//! Per **I4**, GP and FP form separate color pools and are allocated
//! independently. The coloring pass returns a `Coloring` mapping each
//! VReg to a non-negative color index within its class — colors below
//! `max_physical_regs[class]` map to physical registers; colors at or
//! above signal spills (handled by Phase 4d).

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::cfg::dom::{compute_idoms, dominator_tree_children, reverse_postorder};
use crate::cfg::regalloc::interference::InterferenceGraph;
use crate::cfg::{BlockId, CfgFunction, RegClass, Terminator, VReg};

/// The per-call clobber model (I7), supplied to the coloring pass.
///
/// A value live across a GC-safepoint must not be colored into a
/// caller-saved register — the call would clobber it. We model this
/// (per the plan: "every value live across a safepoint interferes with
/// the entire caller-saved sub-pool") by forbidding the caller-saved
/// GP color range `[callee_saved_gp, pool_gp)` for every cross-safepoint
/// GP value. If that leaves no callee-saved color free, the value's
/// color lands at `pool_gp` (overflow) and the driver spills/bails —
/// exactly as it did before caller-saved registers existed.
///
/// FP is unconstrained: the FP allocator pool is entirely callee-saved,
/// so any FP value already survives calls.
pub struct ClobberConstraints<'a> {
    /// VRegs live across a safepoint (from
    /// `interference::cross_safepoint_values`).
    pub cross_safepoint: &'a HashSet<VReg>,
    /// First caller-saved GP color (= number of callee-saved GP regs).
    pub callee_saved_gp: u32,
    /// Total GP allocator pool size; colors `>= pool_gp` overflow.
    pub pool_gp: u32,
}

#[derive(Debug, Clone)]
pub struct Coloring {
    /// Color (0-indexed) assigned to each VReg. Per RegClass — colors
    /// in different classes do not conflict (different physical pools).
    pub colors: HashMap<VReg, u32>,
    /// Max color used per class. `max_color_used[Gp]` + 1 is the
    /// number of distinct GP colors actually consumed. If this exceeds
    /// the physical-register pool size, Phase 4d's spilling pass
    /// rewrites the IR to bring it back under budget.
    pub max_color_used: HashMap<RegClass, u32>,
}

impl Coloring {
    pub fn color_of(&self, v: VReg) -> u32 {
        *self
            .colors
            .get(&v)
            .expect("VReg must have been assigned a color")
    }

    pub fn max_color(&self, class: RegClass) -> u32 {
        self.max_color_used.get(&class).copied().unwrap_or(0)
    }
}

/// Compute a chordal-graph coloring. Walks the dominator tree in
/// pre-order DFS (= reverse of a PEO suffix), processing VRegs in the
/// order they become "fresh" (block params first, then body op defs).
/// For each VReg, picks the lowest color not used by any already-
/// colored neighbor in the same RegClass.
pub fn color(f: &CfgFunction, ig: &InterferenceGraph) -> Coloring {
    color_with_constraints(f, ig, None)
}

/// Like [`color`], but applies the per-call clobber model (I7): every
/// cross-safepoint GP value is forbidden the caller-saved color range,
/// so it cannot be assigned a register the call would clobber. Passing
/// `None` is identical to [`color`] — the unconstrained chordal optimum.
pub fn color_with_constraints(
    f: &CfgFunction,
    ig: &InterferenceGraph,
    constraints: Option<&ClobberConstraints>,
) -> Coloring {
    let mut colors: HashMap<VReg, u32> = HashMap::new();
    let mut max_used: HashMap<RegClass, u32> = HashMap::new();
    let mut max_used_or_zero = |class: RegClass, c: u32| {
        let cur = max_used.get(&class).copied().unwrap_or(0);
        if c >= cur {
            max_used.insert(class, c);
        }
    };

    if f.blocks.is_empty() {
        return Coloring {
            colors,
            max_color_used: max_used,
        };
    }

    let hints = build_copy_hints(f);

    // Build dominator tree.
    let rpo = reverse_postorder(f);
    let idom = compute_idoms(f, &rpo);
    let dom_children = dominator_tree_children(&idom);

    // Pre-order dominator-tree DFS. For chordal graphs this visit
    // order is a valid PEO (reversed); greedy assignment in this
    // order yields optimal coloring.
    let mut stack: Vec<BlockId> = vec![f.entry];
    while let Some(bid) = stack.pop() {
        let block = f.block(bid);

        // Block params first (they become live at block entry).
        for &p in &block.params {
            assign_color(
                p,
                ig,
                &hints,
                constraints,
                &mut colors,
                &mut max_used_or_zero,
            );
        }
        // Body op defs in forward order.
        for op in &block.body {
            for d in op.defs() {
                assign_color(
                    d,
                    ig,
                    &hints,
                    constraints,
                    &mut colors,
                    &mut max_used_or_zero,
                );
            }
        }
        // Terminator def (if any — only InlineBranch defines).
        for d in block.terminator.defs() {
            assign_color(
                d,
                ig,
                &hints,
                constraints,
                &mut colors,
                &mut max_used_or_zero,
            );
        }

        // Recurse to dominator-tree children.
        if let Some(children) = dom_children.get(&bid) {
            for &c in children {
                stack.push(c);
            }
        }
    }

    // Fallback: color any block the dominator-tree walk didn't reach.
    // These are handler / continuation-resume / prompt-abort blocks
    // referenced by `Op::PushExceptionHandler`, `Op::CaptureContinuation`,
    // etc. — the runtime enters them via the handler stack, not via a
    // normal CFG terminator edge, so they have `preds=[]` and aren't in
    // the dominator tree from `entry`. Their VRegs still need colors or
    // `emit_legacy` emits raw SSA-VReg indices as physical registers
    // (e.g. index 28 → X28, the reserved mutator-state register → SIGILL).
    //
    // Coloring them after the main walk is sound: by **I9**, every value
    // crossing into a handler block goes through a stack slot (the AST
    // compiler spills before any safepoint, and mem2reg's I9 gate keeps
    // it that way), so a handler block has no SSA-value live-in from the
    // protected region. Its VRegs can be colored independently, greedily,
    // respecting the interference edges the interference pass already
    // recorded for them.
    for bid_idx in 0..f.num_blocks() {
        let bid = BlockId(bid_idx as u32);
        let block = f.block(bid);
        for &p in &block.params {
            assign_color(
                p,
                ig,
                &hints,
                constraints,
                &mut colors,
                &mut max_used_or_zero,
            );
        }
        for op in &block.body {
            for d in op.defs() {
                assign_color(
                    d,
                    ig,
                    &hints,
                    constraints,
                    &mut colors,
                    &mut max_used_or_zero,
                );
            }
        }
        for d in block.terminator.defs() {
            assign_color(
                d,
                ig,
                &hints,
                constraints,
                &mut colors,
                &mut max_used_or_zero,
            );
        }
    }

    Coloring {
        colors,
        max_color_used: max_used,
    }
}

fn assign_color(
    v: VReg,
    ig: &InterferenceGraph,
    hints: &HashMap<VReg, Vec<VReg>>,
    constraints: Option<&ClobberConstraints>,
    colors: &mut HashMap<VReg, u32>,
    max_used: &mut impl FnMut(RegClass, u32),
) {
    if colors.contains_key(&v) {
        return; // already colored (e.g. function-arg block param visited
        // twice via dom-tree subtlety — shouldn't happen but is
        // a safe no-op)
    }
    // Collect colors used by already-colored interferers in same class.
    let mut forbidden: Vec<u32> = ig
        .neighbors(v)
        .filter(|n| n.class == v.class)
        .filter_map(|n| colors.get(&n).copied())
        .collect();

    // Clobber model (I7): a GP value live across a safepoint cannot take
    // a caller-saved color — forbid the whole caller-saved range. If
    // every callee-saved color is also taken, the smallest free color
    // then lands at `pool_gp` (overflow) and the driver spills/bails,
    // exactly as before caller-saved regs existed.
    if let Some(c) = constraints {
        if v.class == RegClass::Gp && c.cross_safepoint.contains(&v) {
            forbidden.extend(c.callee_saved_gp..c.pool_gp);
        }
    }

    forbidden.sort_unstable();
    forbidden.dedup();

    // Coalesce hint: if `v` is copy-related to an already-colored VReg
    // (a block param and the arg flowing into it across an edge), prefer
    // that VReg's color when it's free. Giving both the same color makes
    // the edge's parallel-copy move a no-op, which edge resolution then
    // drops — eliminating the register shuffles at merge blocks (the
    // dominant SSA-vs-legacy runtime overhead on call-heavy code).
    //
    // This never picks a forbidden (interfering) color, so the result is
    // always a valid coloring — the hint only changes *which* free color
    // is chosen, never correctness.
    if let Some(related) = hints.get(&v) {
        for &r in related {
            if r.class != v.class {
                continue;
            }
            if let Some(&rc) = colors.get(&r) {
                if forbidden.binary_search(&rc).is_err() {
                    colors.insert(v, rc);
                    max_used(v.class, rc);
                    return;
                }
            }
        }
    }

    // Pick smallest non-forbidden color.
    let mut c = 0u32;
    for &f in &forbidden {
        if f == c {
            c += 1;
        } else if f > c {
            break;
        }
    }
    colors.insert(v, c);
    max_used(v.class, c);
}

/// Build the copy-relation: each block param is copy-related to the arg
/// flowing into it from every predecessor's terminator. Coalescing a
/// copy-related pair onto the same color removes the edge's move.
fn build_copy_hints(f: &CfgFunction) -> HashMap<VReg, Vec<VReg>> {
    let mut hints: HashMap<VReg, Vec<VReg>> = HashMap::new();
    for block in &f.blocks {
        let edges: Vec<(BlockId, &Vec<VReg>)> = match &block.terminator {
            Terminator::Jump { target, args } => vec![(*target, args)],
            Terminator::Branch {
                t_target,
                t_args,
                f_target,
                f_args,
                ..
            } => vec![(*t_target, t_args), (*f_target, f_args)],
            Terminator::InlineBranch {
                fall_through,
                fall_args,
                bail,
                bail_args,
                ..
            } => vec![(*fall_through, fall_args), (*bail, bail_args)],
            Terminator::Throw {
                resume,
                resume_args,
                ..
            } => vec![(*resume, resume_args)],
            Terminator::Ret { .. } | Terminator::Unreachable => vec![],
        };
        for (target, args) in edges {
            let params = &f.block(target).params;
            for (i, &arg) in args.iter().enumerate() {
                if let Some(&param) = params.get(i) {
                    if arg != param {
                        hints.entry(arg).or_default().push(param);
                        hints.entry(param).or_default().push(arg);
                    }
                }
            }
        }
    }
    hints
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::regalloc::interference::build_interference;
    use crate::cfg::regalloc::liveness::compute_liveness;
    use crate::cfg::{CfgFunction, Op, RegClass, Terminator};
    use crate::ir::Condition;

    /// Two mutually-interfering params get different colors.
    #[test]
    fn distinct_colors_for_interferers() {
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
        let ig = build_interference(&f, &liveness);
        let coloring = color(&f, &ig);

        assert_ne!(
            coloring.color_of(a),
            coloring.color_of(b),
            "interfering params must have different colors"
        );
        // r can share a color with a (a is dead after AddInt).
        // The greedy assignment might choose anything; just verify
        // total colors used is 2 (a, b) since r can reuse a's.
        assert_eq!(
            coloring.max_color(RegClass::Gp),
            1,
            "GP colors 0 and 1 suffice (max color = 1)"
        );
    }

    /// GP and FP pools are independent.
    #[test]
    fn gp_and_fp_pools_are_separate() {
        let mut f = CfgFunction::new(Some("mixed".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let a_gp = f.new_vreg(RegClass::Gp);
        let b_fp = f.new_vreg(RegClass::Fp);
        let r_fp = f.new_vreg(RegClass::Fp);
        f.block_mut(entry).params.push(a_gp);
        f.block_mut(entry).params.push(b_fp);
        f.block_mut(entry).body.push(Op::AddFloat {
            dst: r_fp,
            lhs: b_fp,
            rhs: b_fp,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: a_gp };

        let liveness = compute_liveness(&f);
        let ig = build_interference(&f, &liveness);
        let coloring = color(&f, &ig);

        // a_gp gets a GP color (0). b_fp gets FP color 0. They don't
        // collide because they're in different RegClass pools.
        assert_eq!(coloring.color_of(a_gp), 0);
        assert_eq!(coloring.color_of(b_fp), 0);
        assert_eq!(coloring.max_color(RegClass::Gp), 0);
        assert_eq!(coloring.max_color(RegClass::Fp), 0);
    }

    /// Three mutually-interfering values need three colors.
    #[test]
    fn three_clique_needs_three_colors() {
        let mut f = CfgFunction::new(Some("three".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let b = f.new_vreg(RegClass::Gp);
        let c = f.new_vreg(RegClass::Gp);
        let s = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(a);
        f.block_mut(entry).params.push(b);
        f.block_mut(entry).params.push(c);
        // r and s use all three across two ops, keeping them all live.
        let r = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: r,
            lhs: a,
            rhs: b,
        });
        f.block_mut(entry).body.push(Op::AddInt {
            dst: s,
            lhs: r,
            rhs: c,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: s };

        let liveness = compute_liveness(&f);
        let ig = build_interference(&f, &liveness);
        let coloring = color(&f, &ig);

        // a, b, c are all simultaneously live at block entry: need 3
        // distinct colors.
        let ca = coloring.color_of(a);
        let cb = coloring.color_of(b);
        let cc = coloring.color_of(c);
        assert_ne!(ca, cb);
        assert_ne!(ca, cc);
        assert_ne!(cb, cc);
        assert_eq!(
            coloring.max_color(RegClass::Gp),
            2,
            "max color = 2 (three colors used)"
        );
    }

    /// Loop pattern with phi: header's phi-param and the body's def
    /// should both get colored without conflict in the loop.
    #[test]
    fn loop_phi_and_back_edge_arg_color_consistently() {
        let mut f = CfgFunction::new(Some("loop_color".into()), 0);
        let entry = f.new_block();
        let header = f.new_block();
        let body = f.new_block();
        let exit = f.new_block();
        f.entry = entry;

        let init = f.new_vreg(RegClass::Gp);
        let x = f.new_vreg(RegClass::Gp);
        let y = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: init,
            value: 0,
        });
        f.block_mut(entry).terminator = Terminator::Jump {
            target: header,
            args: vec![init],
        };
        f.block_mut(header).params.push(x);
        f.block_mut(header).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: x,
            rhs: x,
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

        let liveness = compute_liveness(&f);
        let ig = build_interference(&f, &liveness);
        let coloring = color(&f, &ig);

        // Every VReg must be colored.
        for v in [init, x, y] {
            assert!(
                coloring.colors.contains_key(&v),
                "{:?} should be colored",
                v
            );
        }
    }

    /// Clobber model: a cross-safepoint value cannot take a caller-saved
    /// color. With two interfering values, a tiny callee-saved pool
    /// (`callee_saved_gp = 1`), and BOTH marked cross-safepoint, the
    /// second can't use the caller-saved color 1 — it overflows to
    /// `pool_gp`, signalling a spill/bail.
    #[test]
    fn cross_safepoint_value_avoids_caller_saved() {
        let mut f = CfgFunction::new(Some("clobber".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let b = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(a);
        f.block_mut(entry).params.push(b);
        // Use both so they're simultaneously live at entry → interfere.
        let r = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: r,
            lhs: a,
            rhs: b,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: r };

        let liveness = compute_liveness(&f);
        let ig = build_interference(&f, &liveness);
        let mut cross = HashSet::new();
        cross.insert(a);
        cross.insert(b);
        let constraints = ClobberConstraints {
            cross_safepoint: &cross,
            callee_saved_gp: 1, // only color 0 is callee-saved
            pool_gp: 2,         // color 1 is caller-saved; >=2 overflows
        };
        let coloring = color_with_constraints(&f, &ig, Some(&constraints));

        let ca = coloring.color_of(a);
        let cb = coloring.color_of(b);
        // One gets the lone callee-saved color 0; the other can't use
        // caller-saved color 1, so it overflows to >= pool_gp (2).
        assert!(
            ca == 0 || cb == 0,
            "one cross-call value uses callee-saved 0"
        );
        let other = if ca == 0 { cb } else { ca };
        assert!(
            other >= 2,
            "the other cross-call value overflowed past caller-saved, got {}",
            other
        );
    }

    /// Without the cross-safepoint mark, the same two interfering values
    /// happily use color 0 (callee-saved) and color 1 (caller-saved) —
    /// the relief the grown pool provides for short-lived values.
    #[test]
    fn short_lived_value_uses_caller_saved() {
        let mut f = CfgFunction::new(Some("short".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let b = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(a);
        f.block_mut(entry).params.push(b);
        let r = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).body.push(Op::AddInt {
            dst: r,
            lhs: a,
            rhs: b,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: r };

        let liveness = compute_liveness(&f);
        let ig = build_interference(&f, &liveness);
        let empty = HashSet::new(); // nothing cross-safepoint
        let constraints = ClobberConstraints {
            cross_safepoint: &empty,
            callee_saved_gp: 1,
            pool_gp: 2,
        };
        let coloring = color_with_constraints(&f, &ig, Some(&constraints));
        let ca = coloring.color_of(a);
        let cb = coloring.color_of(b);
        assert_ne!(ca, cb, "interfering values differ");
        assert!(ca < 2 && cb < 2, "both fit in the 2-reg pool (0 and 1)");
    }
}
