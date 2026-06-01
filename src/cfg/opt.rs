//! Phase 3 SSA optimizations — runs after lift + mem2reg, before
//! regalloc would consume the CFG.
//!
//! Three interacting passes, run to fixpoint:
//!
//! 1. **Trivial-block-param elimination.** A block param P is trivial
//!    when every predecessor's outgoing arg-at-this-position is either
//!    a single common VReg V or P itself (a self-loop with V as the
//!    only external value). Trivial params are removed; uses of P
//!    rewritten to V; every predecessor's outgoing arg at the
//!    corresponding index dropped. Cleans up mem2reg's "phi-where-
//!    all-incoming-are-the-same" output and the recursive
//!    self-loop-but-also-V cases.
//!
//! 2. **Slot store forwarding.** A same-block same-register `SlotLoad(S)`
//!    after `SlotStore(S, V)` and before any side-effecting op becomes
//!    `Move(V, V)`. The store remains as the GC root; coalescing removes the
//!    reload. Forwarding is deliberately block-local, does not introduce new
//!    VReg aliases, and stops at side effects so it never reuses a
//!    pre-safepoint register value after the GC may have relocated the object.
//!
//! 3. **Copy coalesce.** Every `Op::Move { dst, src }` becomes a rename
//!    `dst → src`. The rename is resolved transitively then applied to
//!    every use across the function (body ops + terminators). The Move
//!    ops are then removed; their old `dst` VRegs become dead.
//!
//! 4. **Dead non-pointer slot stores.** A `SlotStore(S, const-immediate)`
//!    is removed when slot liveness proves no future `SlotLoad(S)` can
//!    observe it before another store. Pointer-valued stores are kept even
//!    when slot-dead because they may be intentional GC roots across a
//!    safepoint for a value still used through a VReg.
//!
//! 5. **DCE.** Mark-and-sweep over pure ops: side-effecting ops, block
//!    params, and terminator uses are roots; an op whose def is in the
//!    live set propagates use-liveness; sweep removes pure ops whose
//!    defs are all dead.
//!
//! Each pass returns `bool` (whether it changed anything); the
//! orchestrator loops until none of them change.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::cfg::{BlockId, CfgFunction, InlineBranchOp, Op, SlotId, Terminator, VReg};

/// Run all Phase 3 passes to fixpoint. Wired into `build_cfg` after
/// mem2reg.
pub fn optimize(f: &mut CfgFunction) {
    let no_trivial = std::env::var("BEAGLE_SSA_NO_TRIVIAL").is_ok();
    let no_slot_forward = std::env::var("BEAGLE_SSA_NO_SLOT_FORWARD").is_ok();
    let no_coalesce = std::env::var("BEAGLE_SSA_NO_COALESCE").is_ok();
    let no_dead_slot_stores = std::env::var("BEAGLE_SSA_NO_DEAD_SLOT_STORES").is_ok();
    let no_dce = std::env::var("BEAGLE_SSA_NO_DCE").is_ok();
    let no_licm = std::env::var("BEAGLE_SSA_NO_LICM").is_ok();
    let no_guards = std::env::var("BEAGLE_SSA_NO_GUARD_ELIM").is_ok();
    let mut changed = true;
    while changed {
        changed = false;
        if !no_guards {
            changed |= eliminate_redundant_guards(f);
        }
        if !no_trivial {
            changed |= eliminate_trivial_block_params(f);
        }
        if !no_slot_forward {
            changed |= forward_slot_stores(f);
        }
        if !no_coalesce {
            changed |= coalesce_copies(f);
        }
        if !no_dead_slot_stores {
            changed |= eliminate_dead_non_pointer_slot_stores(f);
        }
        if !no_dce {
            changed |= dead_code_elimination(f);
        }
        if !no_licm {
            changed |= crate::cfg::licm::loop_invariant_code_motion(f);
        }
    }
}

// =========================================================================
// Slot store forwarding
// =========================================================================

/// Forward a same-block slot store into later same-register slot loads:
///
/// ```text
/// SlotStore slot(S) <- v
/// ...
/// v = SlotLoad slot(S)
/// ```
///
/// becomes `v = Move v` when the `...` region contains only pure ops that
/// cannot trigger GC or mutate local slots. The `SlotStore` is intentionally
/// kept: for GP values it may be the frame root that keeps `v` visible to the
/// collector. The same-register restriction is important: a broader
/// `dst = Move v` form creates a new global alias that copy coalescing may
/// extend past places where the original slot reload was the intended value
/// boundary.
pub fn forward_slot_stores(f: &mut CfgFunction) -> bool {
    let mut changed = false;

    for block in f.blocks.iter_mut() {
        let mut available: HashMap<SlotId, VReg> = HashMap::new();

        for op in block.body.iter_mut() {
            match op {
                Op::SlotStore { slot, src } => {
                    available.insert(*slot, *src);
                    continue;
                }
                Op::SlotLoad { dst, slot } => {
                    if let Some(&src) = available.get(slot) {
                        if src == *dst {
                            *op = Op::Move { dst: *dst, src };
                            changed = true;
                        }
                    }
                    continue;
                }
                _ => {}
            }

            // A later def of an available source register would make the
            // cached value ambiguous in malformed/non-SSA input. Valid CFGs
            // should not hit this, but invalidating keeps the pass local and
            // conservative.
            let defs = op.defs();
            if !defs.is_empty() {
                available.retain(|_, src| !defs.contains(src));
            }

            // Calls, heap writes, handler/prompt operations, stack mutation,
            // and instrumentation may either observe/mutate frame state or
            // run the GC. Stop forwarding across them; the store remains as
            // the root, and a post-safepoint load still reloads the possibly
            // relocated value from the frame.
            if op.has_side_effect() {
                available.clear();
            }
        }
    }

    changed
}

// =========================================================================
// Dead non-pointer slot store elimination
// =========================================================================

/// Remove stores that cannot be observed by any later `SlotLoad` before
/// another store to the same slot, but only when the stored value is a known
/// non-pointer immediate. This deliberately avoids pointer-valued dead-store
/// removal: a root slot store may exist solely to keep a VReg-held heap value
/// visible to GC across a safepoint.
pub fn eliminate_dead_non_pointer_slot_stores(f: &mut CfgFunction) -> bool {
    if f.blocks.is_empty() {
        return false;
    }

    let (_, live_out) = compute_slot_liveness(f);
    let non_pointer_defs = known_non_pointer_defs(f);
    let mut changed = false;

    for (idx, block) in f.blocks.iter_mut().enumerate() {
        let mut live = live_out[idx].clone();
        let old_body = std::mem::take(&mut block.body);
        let mut new_rev = Vec::with_capacity(old_body.len());

        for op in old_body.into_iter().rev() {
            match &op {
                Op::SlotLoad { slot, .. } => {
                    live.insert(*slot);
                    new_rev.push(op);
                }
                Op::SlotStore { slot, src } => {
                    if !live.contains(slot) && non_pointer_defs.contains(src) {
                        changed = true;
                        continue;
                    }
                    live.remove(slot);
                    new_rev.push(op);
                }
                _ => new_rev.push(op),
            }
        }

        new_rev.reverse();
        block.body = new_rev;
    }

    changed
}

fn compute_slot_liveness(f: &CfgFunction) -> (Vec<HashSet<SlotId>>, Vec<HashSet<SlotId>>) {
    let n = f.num_blocks();
    let mut gen_set = vec![HashSet::new(); n];
    let mut kill = vec![HashSet::new(); n];

    for (idx, block) in f.blocks.iter().enumerate() {
        for op in &block.body {
            match op {
                Op::SlotLoad { slot, .. } => {
                    if !kill[idx].contains(slot) {
                        gen_set[idx].insert(*slot);
                    }
                }
                Op::SlotStore { slot, .. } => {
                    kill[idx].insert(*slot);
                }
                _ => {}
            }
        }
    }

    let mut live_in = vec![HashSet::new(); n];
    let mut live_out = vec![HashSet::new(); n];
    let mut changed = true;
    while changed {
        changed = false;
        for idx in (0..n).rev() {
            let mut new_out = HashSet::new();
            for succ in f.blocks[idx].terminator.successors() {
                new_out.extend(live_in[succ.0 as usize].iter().copied());
            }

            let mut new_in = gen_set[idx].clone();
            for slot in &new_out {
                if !kill[idx].contains(slot) {
                    new_in.insert(*slot);
                }
            }

            if new_in != live_in[idx] || new_out != live_out[idx] {
                live_in[idx] = new_in;
                live_out[idx] = new_out;
                changed = true;
            }
        }
    }

    (live_in, live_out)
}

fn known_non_pointer_defs(f: &CfgFunction) -> HashSet<VReg> {
    let mut defs = HashSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for block in &f.blocks {
            for op in &block.body {
                if let Some(dst) = known_non_pointer_def(op, &defs)
                    && defs.insert(dst)
                {
                    changed = true;
                }
            }
        }
    }
    defs
}

fn known_non_pointer_def(op: &Op, known: &HashSet<VReg>) -> Option<VReg> {
    match op {
        Op::ConstTaggedInt { dst, .. }
        | Op::ConstTrue { dst }
        | Op::ConstFalse { dst }
        | Op::ConstNull { dst } => Some(*dst),
        Op::ConstRawValue { dst, value } if *value == 0 => Some(*dst),
        Op::Move { dst, src } if known.contains(src) => Some(*dst),
        _ => None,
    }
}

// =========================================================================
// Trivial block param elimination
// =========================================================================

/// Remove every block param whose incoming arg from every predecessor
/// is the same VReg (ignoring self-loops). Returns true if any param
/// was eliminated.
pub fn eliminate_trivial_block_params(f: &mut CfgFunction) -> bool {
    let mut any_removed = false;
    let mut changed_pass = true;
    while changed_pass {
        changed_pass = false;
        // Snapshot block count; we don't add or remove blocks here.
        let num_blocks = f.blocks.len();
        for bid_idx in 0..num_blocks {
            let bid = BlockId(bid_idx as u32);
            let num_params = f.block(bid).params.len();
            // Iterate from highest index down so per-block param
            // removal doesn't shift the indices of params we haven't
            // checked yet.
            for param_idx in (0..num_params).rev() {
                let Some(replacement) = try_trivial_param(f, bid, param_idx) else {
                    continue;
                };
                let param_vreg = f.block(bid).params[param_idx];

                // 1. Remove the param from B.
                f.block_mut(bid).params.remove(param_idx);

                // 2. Drop the corresponding arg from each predecessor's
                //    outgoing terminator-args list.
                let preds: Vec<BlockId> = f.block(bid).predecessors.clone();
                for pred in preds {
                    remove_terminator_arg(&mut f.block_mut(pred).terminator, bid, param_idx);
                }

                // 3. Rewrite every use of the removed param to the
                //    replacement VReg.
                let mut rename: HashMap<VReg, VReg> = HashMap::new();
                rename.insert(param_vreg, replacement);
                apply_rename_function_wide(f, &rename);

                any_removed = true;
                changed_pass = true;
            }
        }
    }
    any_removed
}

/// Returns Some(V) if the param at `param_idx` of block `bid` is
/// trivial — i.e. every predecessor's contribution is either V or
/// the param itself (self-loop). Returns None if any predecessor
/// passes a different value or there are no predecessors (entry args).
fn try_trivial_param(f: &CfgFunction, bid: BlockId, param_idx: usize) -> Option<VReg> {
    // Entry block params are function arguments — sourced from the
    // calling convention, not from any CFG predecessor. Folding them
    // would lose the implicit "caller" arg. Skip even when the
    // visible predecessors (e.g. tail-recurse back-edge) all pass the
    // same value, because the initial call still needs the param.
    if bid == f.entry {
        return None;
    }
    let preds = &f.block(bid).predecessors;
    if preds.is_empty() {
        return None;
    }
    let param_vreg = f.block(bid).params[param_idx];
    let mut unique_external: Option<VReg> = None;
    for pred in preds {
        let arg = extract_terminator_arg(&f.block(*pred).terminator, bid, param_idx)?;
        if arg == param_vreg {
            continue; // self-loop — ignore for trivial check
        }
        match unique_external {
            None => unique_external = Some(arg),
            Some(u) if u == arg => continue,
            Some(_) => return None, // multiple distinct external values → not trivial
        }
    }
    unique_external
}

// =========================================================================
// Copy coalesce
// =========================================================================

/// Replace `Op::Move { dst, src }` with a rename `dst → src` applied
/// globally; delete the Move ops. Returns true if any Move was
/// removed.
///
/// **Exception 1: `Op::CompareAndSwap`'s in/out operand.** The legacy
/// `Instruction::CompareAndSwap`'s first operand serves as both the
/// "expected" input and the "old value at addr" output — ARM `CAS`
/// and x86 `LOCK CMPXCHG` mutate that register in place. The IR
/// builder marks this by emitting `v_temp = Move v_expected` right
/// before the CAS, so the CAS mutates `v_temp` rather than
/// `v_expected`. The subsequent `branch Equal v_temp v_expected`
/// (== "did CAS succeed?") relies on `v_temp != v_expected` as
/// distinct VRegs. Coalescing the Move would replace `v_temp` with
/// `v_expected` everywhere, turning the branch into the trivially-
/// true `Equal v_expected v_expected` and making every CAS appear
/// to succeed. Caught by `atom.bg`'s second `compare-and-swap!`
/// returning `true` (expected `false`).
///
/// **Exception 2: Moves whose src is an entry block param.** Entry
/// block params are pre-colored to arg-register positions (X0..X7
/// on ARM64) per the calling convention. The AST compiler emits a
/// prologue copy `new_v = Move arg_0` whose `new_v` gets a callee-
/// saved color, so the long-lived value lives in X19..X27 instead.
/// Coalescing that Move would replace `new_v` with `arg_0` everywhere
/// — and `arg_0`'s color is X0, a caller-saved register that any
/// subsequent call clobbers. Result: SIGSEGV when a function arg
/// is used after a call (e.g. `sort_timsort_test.bg`).
pub fn coalesce_copies(f: &mut CfgFunction) -> bool {
    // Build the set of VRegs that are used as a CAS in/out operand
    // (the `addr` field per Op::CompareAndSwap's confused naming —
    // semantically it's the in/out "expected_then_old" slot).
    let mut cas_in_out: HashSet<VReg> = HashSet::new();
    for block in &f.blocks {
        for op in &block.body {
            if let Op::CompareAndSwap { addr, .. } = op {
                cas_in_out.insert(*addr);
            }
        }
    }

    // Build the set of entry block param VRegs. Moves whose `src`
    // is one of these must stay so the bridge to callee-saved is
    // preserved (see "Exception 2" above).
    let entry_params: HashSet<VReg> = f.block(f.entry).params.iter().copied().collect();

    // Build raw rename map from every Move EXCEPT the two exception
    // categories above.
    let mut rename: HashMap<VReg, VReg> = HashMap::new();
    for block in &f.blocks {
        for op in &block.body {
            if let Op::Move { dst, src } = op
                && !cas_in_out.contains(dst)
                && !entry_params.contains(src)
            {
                rename.insert(*dst, *src);
            }
        }
    }
    if rename.is_empty() {
        return false;
    }

    // Resolve transitively: if rename[a] = b and rename[b] = c, then
    // rename[a] = c. Each lookup walks the chain to the final value.
    let resolved = resolve_transitive_rename(&rename);

    // Apply the rename function-wide.
    apply_rename_function_wide(f, &resolved);

    // Delete only the Moves we actually coalesced — i.e., those
    // whose dst is in the rename map. Preserves both exception
    // categories (CAS in/out, entry-param-bridge).
    let mut removed_any = false;
    for block in f.blocks.iter_mut() {
        let before = block.body.len();
        block.body.retain(|op| match op {
            Op::Move { dst, .. } => !rename.contains_key(dst),
            _ => true,
        });
        if block.body.len() != before {
            removed_any = true;
        }
    }
    removed_any
}

// =========================================================================
// Redundant guard elimination (typed-SSA cleanup)
// =========================================================================

/// Remove `GuardInt`/`GuardFloat` checks whose input is *already* known to be
/// an int / float, replacing the two-successor `InlineBranch` with a plain
/// `Jump` to its fall-through.
///
/// Beagle speculates with guard-and-inline-bail (no deopt), so a monomorphic
/// op like `a + b` emits `guard_int a; guard_int b; add` — and every later use
/// of `a` re-guards it. Once a value has passed a guard (or is produced by an
/// int/float-typed op or constant), all *dominated* re-guards are provably
/// redundant: the guard can never bail, so it becomes an unconditional jump and
/// its bail block goes dead (removed by later DCE / unreachable cleanup).
///
/// **Soundness.** A guard's `dst` equals its `src` on the (always-taken)
/// fall-through, so we rename `dst → src` function-wide and drop the guard. The
/// "known int/float" facts are def-based and therefore hold at every use by SSA
/// dominance (a def dominates all its uses): constants, int/float arithmetic
/// results, and the `dst` of any guard are typed; `Move` and block params
/// propagate to a fixpoint. We never *introduce* a type assumption — we only
/// drop a check that a dominating fact already established, so a value that
/// could be non-int still hits its first guard.
pub fn eliminate_redundant_guards(f: &mut CfgFunction) -> bool {
    let (known_int, known_float) = compute_known_types(f);

    let mut rename: HashMap<VReg, VReg> = HashMap::new();
    let mut new_terms: Vec<(usize, Terminator)> = Vec::new();
    for (i, block) in f.blocks.iter().enumerate() {
        if let Terminator::InlineBranch {
            op,
            fall_through,
            fall_args,
            ..
        } = &block.terminator
        {
            let redundant = match op {
                InlineBranchOp::GuardInt { dst, src } if known_int.contains(src) => {
                    Some((*dst, *src))
                }
                InlineBranchOp::GuardFloat { dst, src } if known_float.contains(src) => {
                    Some((*dst, *src))
                }
                _ => None,
            };
            if let Some((dst, src)) = redundant {
                rename.insert(dst, src);
                new_terms.push((
                    i,
                    Terminator::Jump {
                        target: *fall_through,
                        args: fall_args.clone(),
                    },
                ));
            }
        }
    }
    if new_terms.is_empty() {
        return false;
    }
    for (i, term) in new_terms {
        f.blocks[i].terminator = term;
    }
    let resolved = resolve_transitive_rename(&rename);
    let full: HashMap<VReg, VReg> = rename
        .into_iter()
        .map(|(k, v)| (k, *resolved.get(&k).unwrap_or(&v)))
        .collect();
    apply_rename_function_wide(f, &full);
    crate::cfg::builder::rebuild_predecessors(f);
    true
}

/// Def-based int/float typing of every VReg, to a fixpoint over `Move` and
/// block-param propagation. A VReg is "int" if it can only ever hold a tagged
/// int at runtime, "float" if it can only ever hold a tagged/boxed float.
fn compute_known_types(f: &CfgFunction) -> (HashSet<VReg>, HashSet<VReg>) {
    let mut known_int: HashSet<VReg> = HashSet::new();
    let mut known_float: HashSet<VReg> = HashSet::new();

    // Seed from type-determined definitions.
    for block in &f.blocks {
        for op in &block.body {
            match op {
                Op::ConstTaggedInt { dst, .. } | Op::AddInt { dst, .. } => {
                    known_int.insert(*dst);
                }
                Op::Compare { dst, .. } | Op::CompareFloat { dst, .. } => {
                    // A comparison yields a tagged boolean, which the int guard
                    // accepts as a small tagged value; conservatively NOT marked
                    // (bools are not ints for guard purposes) — left out.
                    let _ = dst;
                }
                Op::AddFloat { dst, .. }
                | Op::SubFloat { dst, .. }
                | Op::MulFloat { dst, .. }
                | Op::DivFloat { dst, .. }
                | Op::IntToFloat { dst, .. }
                | Op::FRoundToZero { dst, .. } => {
                    known_float.insert(*dst);
                }
                _ => {}
            }
        }
        // A guard's dst (and the checked-arithmetic dsts) are typed on the
        // fall-through path, where they're defined.
        if let Terminator::InlineBranch { op, .. } = &block.terminator {
            match op {
                InlineBranchOp::GuardInt { dst, .. }
                | InlineBranchOp::SubChecked { dst, .. }
                | InlineBranchOp::MulChecked { dst, .. }
                | InlineBranchOp::DivChecked { dst, .. }
                | InlineBranchOp::ModuloChecked { dst, .. }
                | InlineBranchOp::ShiftLeftChecked { dst, .. }
                | InlineBranchOp::ShiftRightChecked { dst, .. }
                | InlineBranchOp::ShiftRightZeroChecked { dst, .. }
                | InlineBranchOp::ShiftRightImmChecked { dst, .. } => {
                    known_int.insert(*dst);
                }
                InlineBranchOp::GuardFloat { dst, .. } => {
                    known_float.insert(*dst);
                }
                InlineBranchOp::InlineBumpAllocate { .. } => {}
            }
        }
    }

    // Propagate through Moves and block params to a fixpoint.
    let preds = predecessor_map(f);
    let mut changed = true;
    while changed {
        changed = false;
        for block in &f.blocks {
            for op in &block.body {
                if let Op::Move { dst, src } = op {
                    if known_int.contains(src) && known_int.insert(*dst) {
                        changed = true;
                    }
                    if known_float.contains(src) && known_float.insert(*dst) {
                        changed = true;
                    }
                }
            }
        }
        // A block param is int/float iff every predecessor passes an int/float
        // at that position.
        for (bi, block) in f.blocks.iter().enumerate() {
            let bid = BlockId(bi as u32);
            if block.params.is_empty() {
                continue;
            }
            let Some(pred_list) = preds.get(&bid) else {
                continue;
            };
            if pred_list.is_empty() {
                continue;
            }
            for (idx, &param) in block.params.iter().enumerate() {
                let all_int = pred_list
                    .iter()
                    .all(|&p| incoming_arg(f, p, bid, idx).is_some_and(|v| known_int.contains(&v)));
                if all_int && known_int.insert(param) {
                    changed = true;
                }
                let all_float = pred_list.iter().all(|&p| {
                    incoming_arg(f, p, bid, idx).is_some_and(|v| known_float.contains(&v))
                });
                if all_float && known_float.insert(param) {
                    changed = true;
                }
            }
        }
    }
    (known_int, known_float)
}

/// Predecessor map (block -> its predecessor blocks), from terminators.
fn predecessor_map(f: &CfgFunction) -> HashMap<BlockId, Vec<BlockId>> {
    let mut m: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for (i, block) in f.blocks.iter().enumerate() {
        let from = BlockId(i as u32);
        for succ in block.terminator.successors() {
            m.entry(succ).or_default().push(from);
        }
    }
    m
}

/// The arg `pred` passes to `succ`'s block param at position `idx`.
fn incoming_arg(f: &CfgFunction, pred: BlockId, succ: BlockId, idx: usize) -> Option<VReg> {
    match &f.blocks[pred.0 as usize].terminator {
        Terminator::Jump { target, args } if *target == succ => args.get(idx).copied(),
        Terminator::Branch {
            t_target,
            t_args,
            f_target,
            f_args,
            ..
        } => {
            // A single edge to `succ`; if both go to succ they must agree by
            // CFG construction, take the true side first.
            if *t_target == succ {
                t_args.get(idx).copied()
            } else if *f_target == succ {
                f_args.get(idx).copied()
            } else {
                None
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
                fall_args.get(idx).copied()
            } else if *bail == succ {
                bail_args.get(idx).copied()
            } else {
                None
            }
        }
        _ => None,
    }
}

fn resolve_transitive_rename(raw: &HashMap<VReg, VReg>) -> HashMap<VReg, VReg> {
    let mut resolved: HashMap<VReg, VReg> = HashMap::new();
    for (&start, _) in raw {
        let mut cur = start;
        let mut seen: HashSet<VReg> = HashSet::new();
        seen.insert(cur);
        while let Some(&next) = raw.get(&cur) {
            if !seen.insert(next) {
                // Cycle (shouldn't happen for Moves in valid SSA, but
                // bail out gracefully if it does).
                break;
            }
            cur = next;
        }
        if cur != start {
            resolved.insert(start, cur);
        }
    }
    resolved
}

// =========================================================================
// DCE — mark-and-sweep over pure ops
// =========================================================================

/// Remove pure ops whose defs are unused. Side-effecting ops stay
/// regardless of whether their dst (if any) is used. Block params and
/// terminator-used VRegs are roots. Returns true if any op was
/// removed.
pub fn dead_code_elimination(f: &mut CfgFunction) -> bool {
    // Initial live set: every VReg used by a side-effecting op or a
    // terminator, plus every block param (they're "imported" and may be
    // observed via successor terminators in ways we can't trace
    // perfectly here; trivial-param elimination handles redundant ones
    // separately).
    let mut live: HashSet<VReg> = HashSet::new();
    for block in &f.blocks {
        for &p in &block.params {
            live.insert(p);
        }
        for op in &block.body {
            if op.has_side_effect() {
                for u in op.uses() {
                    live.insert(u);
                }
            }
        }
        for u in block.terminator.uses() {
            live.insert(u);
        }
    }

    // Fixpoint: an op whose def is live makes its uses live.
    let mut changed = true;
    while changed {
        changed = false;
        for block in &f.blocks {
            for op in &block.body {
                let defs = op.defs();
                if defs.iter().any(|d| live.contains(d)) {
                    for u in op.uses() {
                        if live.insert(u) {
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // Sweep: remove side-effect-free ops whose defs are all dead.
    let mut removed_any = false;
    for block in f.blocks.iter_mut() {
        let before = block.body.len();
        block.body.retain(|op| {
            if op.has_side_effect() {
                return true;
            }
            let defs = op.defs();
            if defs.is_empty() {
                // Side-effect-free op with no defs is pointless but
                // shouldn't exist in our enum — keep defensively.
                return true;
            }
            defs.iter().any(|d| live.contains(d))
        });
        if block.body.len() != before {
            removed_any = true;
        }
    }
    removed_any
}

// =========================================================================
// Shared helpers
// =========================================================================

/// Walk every op (body + terminators in every block) and apply the
/// rename map via `Op::rename_uses` / `Terminator::rename_uses`. Defs
/// are NOT touched (that would change SSA identity).
fn apply_rename_function_wide(f: &mut CfgFunction, rename: &HashMap<VReg, VReg>) {
    for block in f.blocks.iter_mut() {
        for op in block.body.iter_mut() {
            op.rename_uses(rename);
        }
        block.terminator.rename_uses(rename);
    }
}

/// Read the outgoing arg at `idx` for the edge `term → succ`. Returns
/// None if `term` has no matching edge.
fn extract_terminator_arg(term: &Terminator, succ: BlockId, idx: usize) -> Option<VReg> {
    match term {
        Terminator::Jump { target, args } if *target == succ => args.get(idx).copied(),
        Terminator::Branch {
            t_target,
            t_args,
            f_target,
            f_args,
            ..
        } => {
            if *t_target == succ {
                t_args.get(idx).copied()
            } else if *f_target == succ {
                f_args.get(idx).copied()
            } else {
                None
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
                fall_args.get(idx).copied()
            } else if *bail == succ {
                bail_args.get(idx).copied()
            } else {
                None
            }
        }
        Terminator::Throw {
            resume,
            resume_args,
            ..
        } if *resume == succ => resume_args.get(idx).copied(),
        _ => None,
    }
}

/// Drop the outgoing arg at `idx` for the edge `term → succ` (if any).
/// If `term` has multiple edges to `succ` (e.g. `Branch t==f`), the
/// arg is dropped from each matching list — they must stay in lockstep.
fn remove_terminator_arg(term: &mut Terminator, succ: BlockId, idx: usize) {
    match term {
        Terminator::Jump { target, args } if *target == succ => {
            if idx < args.len() {
                args.remove(idx);
            }
        }
        Terminator::Branch {
            t_target,
            t_args,
            f_target,
            f_args,
            ..
        } => {
            if *t_target == succ && idx < t_args.len() {
                t_args.remove(idx);
            }
            if *f_target == succ && idx < f_args.len() {
                f_args.remove(idx);
            }
        }
        Terminator::InlineBranch {
            fall_through,
            fall_args,
            bail,
            bail_args,
            ..
        } => {
            if *fall_through == succ && idx < fall_args.len() {
                fall_args.remove(idx);
            }
            if *bail == succ && idx < bail_args.len() {
                bail_args.remove(idx);
            }
        }
        Terminator::Throw {
            resume,
            resume_args,
            ..
        } if *resume == succ => {
            if idx < resume_args.len() {
                resume_args.remove(idx);
            }
        }
        _ => {}
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{CfgFunction, InlineBranchOp, Op, RegClass, SlotId, Terminator};
    use crate::ir::Condition;

    #[test]
    fn redundant_guard_on_known_int_is_removed() {
        // A: x=1; y=2; t = x + y;  guard_int(g, t) -> B else Cbail
        // B: ret g
        // t is an AddInt result => known int => the guard is redundant.
        let mut f = CfgFunction::new(Some("guard".into()), 0);
        let a = f.new_block();
        let b = f.new_block();
        let cbail = f.new_block();
        f.entry = a;

        let x = f.new_vreg(RegClass::Gp);
        let y = f.new_vreg(RegClass::Gp);
        let t = f.new_vreg(RegClass::Gp);
        let g = f.new_vreg(RegClass::Gp);

        f.block_mut(a)
            .body
            .push(Op::ConstTaggedInt { dst: x, value: 1 });
        f.block_mut(a)
            .body
            .push(Op::ConstTaggedInt { dst: y, value: 2 });
        f.block_mut(a).body.push(Op::AddInt {
            dst: t,
            lhs: x,
            rhs: y,
        });
        f.block_mut(a).terminator = Terminator::InlineBranch {
            op: InlineBranchOp::GuardInt { dst: g, src: t },
            fall_through: b,
            fall_args: vec![],
            bail: cbail,
            bail_args: vec![],
        };
        f.block_mut(b).terminator = Terminator::Ret { value: g };
        f.block_mut(cbail).terminator = Terminator::Unreachable;
        f.block_mut(b).predecessors = vec![a];
        f.block_mut(cbail).predecessors = vec![a];

        let changed = eliminate_redundant_guards(&mut f);
        assert!(changed, "guard on a known int should be eliminated");
        // The guard terminator becomes a plain Jump to the fall-through.
        assert!(
            matches!(&f.block(a).terminator, Terminator::Jump { target, .. } if *target == b),
            "guard replaced by Jump to fall-through, got {:?}",
            f.block(a).terminator
        );
        // g was renamed to t (the guard's input).
        assert!(
            matches!(&f.block(b).terminator, Terminator::Ret { value } if *value == t),
            "use of guard dst renamed to its src"
        );
    }

    #[test]
    fn guard_on_unknown_value_is_kept() {
        // Guard on a SlotLoad result (unknown type) must NOT be eliminated.
        let mut f = CfgFunction::new(Some("guard_keep".into()), 1);
        let a = f.new_block();
        let b = f.new_block();
        let cbail = f.new_block();
        f.entry = a;
        let v = f.new_vreg(RegClass::Gp);
        let g = f.new_vreg(RegClass::Gp);
        f.block_mut(a).body.push(Op::SlotLoad {
            dst: v,
            slot: SlotId(0),
        });
        f.block_mut(a).terminator = Terminator::InlineBranch {
            op: InlineBranchOp::GuardInt { dst: g, src: v },
            fall_through: b,
            fall_args: vec![],
            bail: cbail,
            bail_args: vec![],
        };
        f.block_mut(b).terminator = Terminator::Ret { value: g };
        f.block_mut(cbail).terminator = Terminator::Unreachable;
        f.block_mut(b).predecessors = vec![a];
        f.block_mut(cbail).predecessors = vec![a];

        assert!(
            !eliminate_redundant_guards(&mut f),
            "guard on an untyped SlotLoad must be kept"
        );
    }

    /// `c = const; v = Move c; ret v`: coalesce removes the Move,
    /// folds v → c. Uses a non-entry-param Move src so the "preserve
    /// entry-param bridge" exception doesn't apply.
    #[test]
    fn coalesce_removes_move_op() {
        let mut f = CfgFunction::new(Some("coalesce".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let c = f.new_vreg(RegClass::Gp);
        let v_copy = f.new_vreg(RegClass::Gp);
        f.block_mut(entry)
            .body
            .push(Op::ConstTaggedInt { dst: c, value: 42 });
        f.block_mut(entry).body.push(Op::Move {
            dst: v_copy,
            src: c,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: v_copy };

        let changed = coalesce_copies(&mut f);
        assert!(changed, "coalesce should have removed a Move");
        assert_eq!(
            f.block(entry).body.len(),
            1,
            "Move op should be gone; only ConstTaggedInt left"
        );
        match &f.block(entry).terminator {
            Terminator::Ret { value } => assert_eq!(*value, c, "Ret should now use c directly"),
            _ => panic!("expected Ret"),
        }
    }

    /// Entry-param Move is preserved: `block(arg) { v = Move arg; ret v }`.
    /// The Move bridges the arg-reg color (X0..X7) to a callee-saved
    /// vreg; coalescing it would extend `arg`'s lifetime into regions
    /// where its arg register is clobbered by intervening calls.
    /// See the "Exception 2" comment in `coalesce_copies`.
    #[test]
    fn coalesce_preserves_entry_param_move() {
        let mut f = CfgFunction::new(Some("arg_bridge".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let arg = f.new_vreg(RegClass::Gp);
        let v_copy = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(arg);
        f.block_mut(entry).body.push(Op::Move {
            dst: v_copy,
            src: arg,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: v_copy };

        let changed = coalesce_copies(&mut f);
        assert!(!changed, "Move from entry param must stay");
        assert_eq!(
            f.block(entry).body.len(),
            1,
            "Move op should still be present"
        );
    }

    /// Store-to-load forwarding keeps the root store but removes a redundant
    /// same-register reload when no side effect can run the GC or mutate the
    /// frame between the store and load.
    #[test]
    fn slot_store_forwarding_removes_same_block_reload() {
        let mut f = CfgFunction::new(Some("slot_forward".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let value = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: value,
            value: 42,
        });
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: value,
        });
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: value,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Ret { value };

        let changed = forward_slot_stores(&mut f);
        assert!(changed, "same-register reload should be forwarded");

        assert!(
            !f.block(entry)
                .body
                .iter()
                .any(|op| matches!(op, Op::SlotLoad { .. })),
            "the same-block reload should be forwarded away: {:?}",
            f.block(entry).body
        );
        assert!(
            f.block(entry)
                .body
                .iter()
                .any(|op| matches!(op, Op::SlotStore { slot, src } if *slot == SlotId(0) && *src == value)),
            "the SlotStore must remain as the frame root"
        );
        match &f.block(entry).terminator {
            Terminator::Ret { value: ret } => assert_eq!(*ret, value),
            _ => panic!("expected Ret"),
        }
    }

    /// A side-effecting op is a forwarding barrier. In particular, a GC
    /// safepoint may relocate a heap object, so a later load must read the
    /// frame slot instead of reusing the pre-safepoint register value.
    #[test]
    fn slot_store_forwarding_stops_at_side_effect() {
        let mut f = CfgFunction::new(Some("slot_forward_barrier".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let value = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: value,
            value: 42,
        });
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: value,
        });
        f.block_mut(entry).body.push(Op::RecordGcSafepoint);
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: value,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Ret { value };

        optimize(&mut f);

        assert!(
            f.block(entry).body.iter().any(
                |op| matches!(op, Op::SlotLoad { dst, slot } if *dst == value && *slot == SlotId(0))
            ),
            "the post-safepoint load must remain: {:?}",
            f.block(entry).body
        );
    }

    /// Forwarding intentionally does not turn `load S -> other` into
    /// `Move other, value`: that broader form creates aliases that the global
    /// coalescer can move past the intended slot boundary.
    #[test]
    fn slot_store_forwarding_does_not_alias_distinct_vregs() {
        let mut f = CfgFunction::new(Some("slot_forward_no_alias".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let value = f.new_vreg(RegClass::Gp);
        let loaded = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: value,
            value: 42,
        });
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: value,
        });
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: loaded,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: loaded };

        let changed = forward_slot_stores(&mut f);
        assert!(!changed, "distinct destination should not be forwarded");
        assert!(f.block(entry).body.iter().any(
            |op| matches!(op, Op::SlotLoad { dst, slot } if *dst == loaded && *slot == SlotId(0))
        ));
    }

    /// A non-pointer immediate store is dead when no later SlotLoad can
    /// observe it before another store. The subsequent DCE pass removes the
    /// now-unused constant too.
    #[test]
    fn dead_non_pointer_slot_store_is_removed() {
        let mut f = CfgFunction::new(Some("dead_slot_store".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let zero = f.new_vreg(RegClass::Gp);
        let ret = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: zero,
            value: 0,
        });
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: zero,
        });
        f.block_mut(entry)
            .body
            .push(Op::ConstTaggedInt { dst: ret, value: 1 });
        f.block_mut(entry).terminator = Terminator::Ret { value: ret };

        optimize(&mut f);

        assert!(
            !f.block(entry)
                .body
                .iter()
                .any(|op| matches!(op, Op::SlotStore { .. })),
            "dead non-pointer slot store should be removed: {:?}",
            f.block(entry).body
        );
        assert!(
            !f.block(entry)
                .body
                .iter()
                .any(|op| matches!(op, Op::ConstTaggedInt { dst, .. } if *dst == zero)),
            "unused zero should be DCE'd after store removal"
        );
    }

    /// If a later load can observe the store, slot-liveness keeps it.
    #[test]
    fn live_non_pointer_slot_store_is_kept() {
        let mut f = CfgFunction::new(Some("live_slot_store".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let zero = f.new_vreg(RegClass::Gp);
        let loaded = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: zero,
            value: 0,
        });
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: zero,
        });
        f.block_mut(entry).body.push(Op::SlotLoad {
            dst: loaded,
            slot: SlotId(0),
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: loaded };

        eliminate_dead_non_pointer_slot_stores(&mut f);

        assert!(f.block(entry).body.iter().any(
            |op| matches!(op, Op::SlotStore { slot, src } if *slot == SlotId(0) && *src == zero)
        ));
    }

    /// A pointer-like value may be a GC root even when no later SlotLoad reads
    /// the slot; this pass only removes known non-pointer stores.
    #[test]
    fn dead_pointer_slot_store_is_kept() {
        let mut f = CfgFunction::new(Some("dead_pointer_slot_store".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let ptr = f.new_vreg(RegClass::Gp);
        let ret = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).body.push(Op::ConstPointer {
            dst: ptr,
            ptr: 0x1000,
        });
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: ptr,
        });
        f.block_mut(entry)
            .body
            .push(Op::ConstTaggedInt { dst: ret, value: 1 });
        f.block_mut(entry).terminator = Terminator::Ret { value: ret };

        optimize(&mut f);

        assert!(f.block(entry).body.iter().any(
            |op| matches!(op, Op::SlotStore { slot, src } if *slot == SlotId(0) && *src == ptr)
        ));
    }

    /// A trivial block param: join block where both predecessors pass
    /// the same value. Trivial-param elim removes the param, drops the
    /// arg from each pred, rewrites uses.
    #[test]
    fn trivial_block_param_eliminated() {
        let mut f = CfgFunction::new(Some("trivial".into()), 0);
        let entry = f.new_block();
        let then_b = f.new_block();
        let else_b = f.new_block();
        let join = f.new_block();
        f.entry = entry;

        let arg = f.new_vreg(RegClass::Gp);
        let common = f.new_vreg(RegClass::Gp);
        let phi = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).params.push(arg);
        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: common,
            value: 7,
        });
        f.block_mut(entry).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: arg,
            rhs: arg,
            t_target: then_b,
            t_args: vec![],
            f_target: else_b,
            f_args: vec![],
        };
        f.block_mut(then_b).terminator = Terminator::Jump {
            target: join,
            args: vec![common],
        };
        f.block_mut(else_b).terminator = Terminator::Jump {
            target: join,
            args: vec![common],
        };
        f.block_mut(join).params.push(phi);
        f.block_mut(join).terminator = Terminator::Ret { value: phi };
        f.block_mut(then_b).predecessors.push(entry);
        f.block_mut(else_b).predecessors.push(entry);
        f.block_mut(join).predecessors.push(then_b);
        f.block_mut(join).predecessors.push(else_b);

        let changed = eliminate_trivial_block_params(&mut f);
        assert!(changed, "expected trivial-param elimination to fire");
        assert!(
            f.block(join).params.is_empty(),
            "join's trivial param removed"
        );
        match &f.block(then_b).terminator {
            Terminator::Jump { args, .. } => {
                assert!(args.is_empty(), "then's outgoing arg dropped")
            }
            _ => panic!("expected Jump"),
        }
        match &f.block(else_b).terminator {
            Terminator::Jump { args, .. } => {
                assert!(args.is_empty(), "else's outgoing arg dropped")
            }
            _ => panic!("expected Jump"),
        }
        match &f.block(join).terminator {
            Terminator::Ret { value } => assert_eq!(*value, common, "Ret rewritten to use common"),
            _ => panic!("expected Ret"),
        }
    }

    /// A self-loop trivial param (loop-header where the body re-passes
    /// the same value): trivial-param elim still folds it because only
    /// one distinct external value exists.
    #[test]
    fn self_loop_trivial_param_eliminated() {
        let mut f = CfgFunction::new(Some("self_loop".into()), 0);
        let entry = f.new_block();
        let header = f.new_block();
        let exit = f.new_block();
        f.entry = entry;

        let arg = f.new_vreg(RegClass::Gp);
        let phi = f.new_vreg(RegClass::Gp);

        f.block_mut(entry).params.push(arg);
        f.block_mut(entry).terminator = Terminator::Jump {
            target: header,
            args: vec![arg],
        };
        f.block_mut(header).params.push(phi);
        // header branches to itself or exits; back-edge passes phi.
        f.block_mut(header).terminator = Terminator::Branch {
            cond: Condition::Equal,
            lhs: phi,
            rhs: phi,
            t_target: header,
            t_args: vec![phi], // self-loop: pass phi unchanged
            f_target: exit,
            f_args: vec![],
        };
        f.block_mut(exit).terminator = Terminator::Ret { value: phi };
        f.block_mut(header).predecessors.push(entry);
        f.block_mut(header).predecessors.push(header);
        f.block_mut(exit).predecessors.push(header);

        let changed = eliminate_trivial_block_params(&mut f);
        assert!(
            changed,
            "self-loop trivial param should fold to entry's arg"
        );
        assert!(f.block(header).params.is_empty(), "phi removed");
        match &f.block(exit).terminator {
            Terminator::Ret { value } => assert_eq!(*value, arg, "exit Ret now uses entry's arg"),
            _ => panic!("expected Ret"),
        }
    }

    /// DCE removes a pure op whose def is never used.
    #[test]
    fn dce_removes_dead_const() {
        let mut f = CfgFunction::new(Some("dce".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let arg = f.new_vreg(RegClass::Gp);
        let dead = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(arg);
        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: dead,
            value: 42,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: arg };

        let changed = dead_code_elimination(&mut f);
        assert!(changed, "DCE should remove the unused ConstTaggedInt");
        assert!(
            f.block(entry).body.is_empty(),
            "body should be empty after DCE"
        );
    }

    /// DCE keeps side-effecting ops even when their result is unused
    /// (e.g. SlotStore — no def at all, only side effect).
    #[test]
    fn dce_keeps_side_effecting_ops() {
        let mut f = CfgFunction::new(Some("dce_keep".into()), 1);
        let entry = f.new_block();
        f.entry = entry;
        let arg = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(arg);
        f.block_mut(entry).body.push(Op::SlotStore {
            slot: SlotId(0),
            src: arg,
        });
        f.block_mut(entry).terminator = Terminator::Ret { value: arg };

        let changed = dead_code_elimination(&mut f);
        assert!(!changed, "no DCE — SlotStore is side-effecting");
        assert_eq!(f.block(entry).body.len(), 1);
    }

    /// Fixpoint: trivial-param elim creates dead Moves, coalesce kills
    /// them, DCE cleans up the const that fed them. Uses a non-entry-
    /// param Move src so the "preserve entry-param bridge" exception
    /// doesn't gate coalescing.
    #[test]
    fn optimize_fixpoint() {
        let mut f = CfgFunction::new(Some("fixpoint".into()), 0);
        let entry = f.new_block();
        let join = f.new_block();
        f.entry = entry;
        let dead_const = f.new_vreg(RegClass::Gp);
        let live_const = f.new_vreg(RegClass::Gp);
        let copy = f.new_vreg(RegClass::Gp);
        let phi = f.new_vreg(RegClass::Gp);

        // A dead constant + a live constant + a Move of the live const.
        // (No entry params; coalesce is free to fold the Move.)
        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: dead_const,
            value: 99,
        });
        f.block_mut(entry).body.push(Op::ConstTaggedInt {
            dst: live_const,
            value: 7,
        });
        f.block_mut(entry).body.push(Op::Move {
            dst: copy,
            src: live_const,
        });
        // Jump to join, passing arg via copy.
        f.block_mut(entry).terminator = Terminator::Jump {
            target: join,
            args: vec![copy],
        };
        f.block_mut(join).params.push(phi);
        f.block_mut(join).terminator = Terminator::Ret { value: phi };
        f.block_mut(join).predecessors.push(entry);

        optimize(&mut f);

        // After fixpoint:
        // - Move coalesced → copy gone, Jump args = [live_const].
        // - Trivial param: join has 1 pred passing live_const; phi → live_const.
        // - DCE: dead_const removed. live_const survives because Ret uses it.
        assert_eq!(
            f.block(entry).body.len(),
            1,
            "entry body has only live_const left",
        );
        assert!(
            matches!(f.block(entry).body[0], Op::ConstTaggedInt { value: 7, .. }),
            "the surviving op is the live_const",
        );
        assert!(f.block(join).params.is_empty(), "join param folded");
        match &f.block(entry).terminator {
            Terminator::Jump { args, .. } => assert!(args.is_empty(), "Jump args folded"),
            _ => panic!("expected Jump"),
        }
        match &f.block(join).terminator {
            Terminator::Ret { value } => assert_eq!(*value, live_const),
            _ => panic!("expected Ret"),
        }
    }
}
