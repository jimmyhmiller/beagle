//! Dominator computations shared across CFG passes.
//!
//! Cooper / Harvey / Kennedy iterative idom + dominance-frontier
//! computation. Used by the verifier (I5 dominance check), mem2reg
//! (Cytron-style phi placement), and any future SSA-aware pass that
//! needs the dominator tree.
//!
//! Reachable blocks only: unreachable blocks (no path from `entry`) are
//! omitted from every returned map. Callers that need them must DCE
//! first.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

use crate::cfg::{BlockId, CfgFunction};

/// Set of blocks reachable from `entry` via the terminator-successor
/// graph.
pub fn compute_reachable(f: &CfgFunction) -> HashSet<BlockId> {
    let mut reached = HashSet::new();
    if f.blocks.is_empty() {
        return reached;
    }
    let mut queue = VecDeque::new();
    queue.push_back(f.entry);
    reached.insert(f.entry);
    while let Some(b) = queue.pop_front() {
        // Normal CFG successors via the block's terminator.
        for s in f.block(b).terminator.successors() {
            if reached.insert(s) {
                queue.push_back(s);
            }
        }
        // "Soft" successors: handler/resume/abort blocks referenced by
        // exception, continuation, prompt, and effect ops. The runtime
        // jumps to these via the handler stack — they're not normal
        // CFG successors, but DCE must not wipe them or `emit_legacy`
        // will produce unbound labels for the runtime to jump to.
        for op in &f.block(b).body {
            for s in op.block_refs() {
                if reached.insert(s) {
                    queue.push_back(s);
                }
            }
        }
    }
    reached
}

/// Reverse postorder traversal from `entry`. Only reachable blocks
/// appear in the result. Used by `compute_idoms` and by any worklist
/// algorithm that wants the "good" iteration order for forward
/// dataflow.
pub fn reverse_postorder(f: &CfgFunction) -> Vec<BlockId> {
    let mut visited = HashSet::new();
    let mut post = Vec::new();
    fn dfs(f: &CfgFunction, b: BlockId, visited: &mut HashSet<BlockId>, post: &mut Vec<BlockId>) {
        if !visited.insert(b) {
            return;
        }
        for s in f.block(b).terminator.successors() {
            dfs(f, s, visited, post);
        }
        post.push(b);
    }
    dfs(f, f.entry, &mut visited, &mut post);
    post.reverse();
    post
}

/// Cooper/Harvey/Kennedy iterative idom. Returns a map from each
/// reachable block (other than entry) to its immediate dominator.
/// Entry has no entry in the map (it dominates itself but has no idom).
pub fn compute_idoms(f: &CfgFunction, rpo: &[BlockId]) -> HashMap<BlockId, BlockId> {
    let mut rpo_index: HashMap<BlockId, usize> = HashMap::new();
    for (i, &b) in rpo.iter().enumerate() {
        rpo_index.insert(b, i);
    }
    let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
    let entry = f.entry;
    idom.insert(entry, entry);

    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let preds: Vec<BlockId> = f
                .block(b)
                .predecessors
                .iter()
                .copied()
                .filter(|p| idom.contains_key(p))
                .collect();
            if preds.is_empty() {
                continue;
            }
            let mut new_idom = preds[0];
            for &p in &preds[1..] {
                new_idom = intersect(&idom, &rpo_index, p, new_idom);
            }
            if idom.get(&b) != Some(&new_idom) {
                idom.insert(b, new_idom);
                changed = true;
            }
        }
    }

    idom.remove(&entry);
    idom
}

fn intersect(
    idom: &HashMap<BlockId, BlockId>,
    rpo_index: &HashMap<BlockId, usize>,
    mut b1: BlockId,
    mut b2: BlockId,
) -> BlockId {
    while b1 != b2 {
        while rpo_index[&b1] > rpo_index[&b2] {
            b1 = idom[&b1];
        }
        while rpo_index[&b2] > rpo_index[&b1] {
            b2 = idom[&b2];
        }
    }
    b1
}

/// True if `a` dominates `b` (`a == b` counts as dominating).
pub fn dominates(idom: &HashMap<BlockId, BlockId>, a: BlockId, mut b: BlockId) -> bool {
    if a == b {
        return true;
    }
    while let Some(&parent) = idom.get(&b) {
        if parent == a {
            return true;
        }
        if parent == b {
            return false;
        }
        b = parent;
    }
    false
}

/// Children of each block in the dominator tree (the inverse of the
/// idom relation). Entry isn't a key (it has no idom); reachable
/// non-entry blocks with no children may also be absent. Used by
/// dominator-tree DFS passes (e.g. the mem2reg rename phase).
pub fn dominator_tree_children(idom: &HashMap<BlockId, BlockId>) -> HashMap<BlockId, Vec<BlockId>> {
    let mut children: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for (&child, &parent) in idom {
        children.entry(parent).or_default().push(child);
    }
    for kids in children.values_mut() {
        kids.sort();
    }
    children
}

/// Cytron-style dominance frontiers. `DF(b)` is the set of blocks `y`
/// such that `b` dominates a predecessor of `y` but does NOT strictly
/// dominate `y`. Phi placement (= block-param placement in our model)
/// for a variable defined in `b` happens at every block in the
/// iterated dominance frontier of `b`'s def-sites.
///
/// Returns a map from every reachable block to its DF (which may be
/// empty). The standard "two-finger" walk: for each join point Y, walk
/// upward from each predecessor until reaching `idom(Y)`; add Y to
/// DF of every block on that walk.
pub fn compute_dominance_frontiers(
    f: &CfgFunction,
    idom: &HashMap<BlockId, BlockId>,
) -> HashMap<BlockId, HashSet<BlockId>> {
    let mut df: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
    let reachable = compute_reachable(f);
    for &b in &reachable {
        df.insert(b, HashSet::new());
    }
    for (idx, block) in f.blocks.iter().enumerate() {
        let b = BlockId(idx as u32);
        if !reachable.contains(&b) {
            continue;
        }
        if block.predecessors.len() < 2 {
            continue;
        }
        let idom_b = match idom.get(&b) {
            Some(&i) => i,
            None => continue, // entry; no idom
        };
        for &pred in &block.predecessors {
            let mut runner = pred;
            while runner != idom_b {
                df.entry(runner).or_default().insert(b);
                runner = match idom.get(&runner) {
                    Some(&parent) => parent,
                    None => break, // reached entry
                };
            }
        }
    }
    df
}

/// Iterated dominance frontier of a set of "def" blocks. Standard fixed-
/// point: keep unioning DF of the current frontier until stable. This is
/// where phi nodes (block params) for a variable get placed.
pub fn iterated_dominance_frontier(
    def_blocks: &HashSet<BlockId>,
    df: &HashMap<BlockId, HashSet<BlockId>>,
) -> HashSet<BlockId> {
    let mut worklist: Vec<BlockId> = def_blocks.iter().copied().collect();
    let mut result: HashSet<BlockId> = HashSet::new();
    while let Some(b) = worklist.pop() {
        if let Some(df_b) = df.get(&b) {
            for &y in df_b {
                if result.insert(y) {
                    worklist.push(y);
                }
            }
        }
    }
    result
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CfgFunction, Op, RegClass, SlotId, Terminator};
    use crate::ir::Condition;

    fn make_diamond() -> CfgFunction {
        // entry → then → join
        // entry → else → join
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
        f.block_mut(entry).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: v,
            rhs: v,
            t_target: then_b,
            t_args: vec![],
            f_target: else_b,
            f_args: vec![],
        };
        f.block_mut(then_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        f.block_mut(else_b).terminator = Terminator::Jump {
            target: join,
            args: vec![],
        };
        f.block_mut(join).terminator = Terminator::Ret { value: v };
        f.block_mut(then_b).predecessors.push(entry);
        f.block_mut(else_b).predecessors.push(entry);
        f.block_mut(join).predecessors.push(then_b);
        f.block_mut(join).predecessors.push(else_b);
        f
    }

    #[test]
    fn idom_diamond() {
        let f = make_diamond();
        let rpo = reverse_postorder(&f);
        let idom = compute_idoms(&f, &rpo);
        // entry idoms then, else, join
        let entry = f.entry;
        let then_b = BlockId(1);
        let else_b = BlockId(2);
        let join = BlockId(3);
        assert_eq!(idom.get(&then_b), Some(&entry));
        assert_eq!(idom.get(&else_b), Some(&entry));
        assert_eq!(idom.get(&join), Some(&entry));
    }

    #[test]
    fn dom_frontiers_diamond() {
        let f = make_diamond();
        let rpo = reverse_postorder(&f);
        let idom = compute_idoms(&f, &rpo);
        let df = compute_dominance_frontiers(&f, &idom);
        let entry = f.entry;
        let then_b = BlockId(1);
        let else_b = BlockId(2);
        let join = BlockId(3);
        // then and else both have join in their DF (they're predecessors
        // of a join point and don't strictly dominate it).
        assert_eq!(df[&then_b].iter().copied().collect::<Vec<_>>(), vec![join]);
        assert_eq!(df[&else_b].iter().copied().collect::<Vec<_>>(), vec![join]);
        // entry dominates everything so its DF is empty.
        assert!(df[&entry].is_empty());
        // join has no successors → empty DF.
        assert!(df[&join].is_empty());
    }

    #[test]
    fn iterated_df_of_singleton_is_df() {
        let f = make_diamond();
        let rpo = reverse_postorder(&f);
        let idom = compute_idoms(&f, &rpo);
        let df = compute_dominance_frontiers(&f, &idom);
        let mut defs = HashSet::new();
        defs.insert(BlockId(1)); // then
        let idf = iterated_dominance_frontier(&defs, &df);
        // join is in DF(then), and DF(join) is empty, so IDF = {join}.
        assert_eq!(idf.iter().copied().collect::<Vec<_>>(), vec![BlockId(3)]);
    }

    #[test]
    fn dominator_tree_children_diamond() {
        let f = make_diamond();
        let rpo = reverse_postorder(&f);
        let idom = compute_idoms(&f, &rpo);
        let tree = dominator_tree_children(&idom);
        // entry's children: then, else, join (all dominated by entry).
        let mut entry_children = tree.get(&f.entry).cloned().unwrap_or_default();
        entry_children.sort();
        assert_eq!(entry_children, vec![BlockId(1), BlockId(2), BlockId(3)]);
    }
}
