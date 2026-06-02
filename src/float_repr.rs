//! Float representation analysis for the tier-2 SSA float-unboxing pass.
//!
//! Floats are heap-boxed; keeping them unboxed in FP registers/slots
//! across chained float ops is the bulk of the win (see
//! `docs/SSA_ARCHITECTURE.md` "Float unboxing"). The first thing the pass
//! needs is to know which local slots provably hold *only* float values,
//! so they can be turned into unscanned FP slots.
//!
//! **Soundness.** A slot is float iff every store to it is a float-typed
//! value and it has at least one store. The seeds that make a value
//! provably float without any runtime check:
//! - an `Instruction::FloatBinOp` result — a float-specialized op always
//!   yields a float (fast path) or its bail helper coerces-or-throws, so
//!   the result is a float on every path where control continues;
//! - a float-tagged constant — a `RawValue` whose low 3 bits are the
//!   `Float` type tag is a tagged pointer to a boxed float (only floats
//!   carry that tag).
//!
//! These propagate through `Assign`/`LoadLocal` of float values and float
//! locals to a least-fixed-point. Args and other untyped values are never
//! float, so they keep their boxed representation.

use std::collections::{HashMap, HashSet};

use crate::ir::{Instruction, Value};
use crate::pretty_print::PrettyPrint;
use crate::types::BuiltInTypes;

fn reg_index(v: &Value) -> Option<usize> {
    match v {
        Value::Register(r) => Some(r.index),
        _ => None,
    }
}

fn local_index(v: &Value) -> Option<usize> {
    match v {
        Value::Local(n) => Some(*n),
        _ => None,
    }
}

/// A `RawValue` whose low 3 bits are the Float tag is a tagged pointer to
/// a boxed float constant. Only floats carry that tag, so this is a
/// reliable float witness.
pub fn is_float_const(v: &Value) -> bool {
    match v {
        Value::RawValue(n) => (*n as isize) & 0b111 == BuiltInTypes::Float.get_tag(),
        _ => false,
    }
}

fn is_float_value(v: &Value, float_regs: &HashSet<usize>, float_locals: &HashSet<usize>) -> bool {
    if is_float_const(v) {
        return true;
    }
    if let Some(r) = reg_index(v) {
        if float_regs.contains(&r) {
            return true;
        }
    }
    if let Some(l) = local_index(v) {
        if float_locals.contains(&l) {
            return true;
        }
    }
    false
}

/// If `ins` stores to a local slot, return `(slot, source_value)`.
fn local_store(ins: &Instruction) -> Option<(usize, &Value)> {
    match ins {
        Instruction::StoreLocal(dst, src) => local_index(dst).map(|l| (l, src)),
        Instruction::Assign(dst, src) if matches!(dst, Value::Local(_)) => {
            local_index(dst).map(|l| (l, src))
        }
        _ => None,
    }
}

/// The result of float-type analysis: the local slots and registers that
/// hold a value that is **definitely** a float at runtime — not merely a
/// float-specialized op's result (which bails to a non-float on a type
/// miss). Only definitely-float values are safe to keep unboxed without a
/// runtime guard.
#[derive(Debug, Default, Clone)]
pub struct FloatTypes {
    pub locals: HashSet<usize>,
    pub regs: HashSet<usize>,
}

/// Compute the definitely-float locals and registers.
///
/// **Soundness.** A `FloatBinOp` is float-specialized speculation: given
/// non-float operands at runtime it bails to the polymorphic helper, whose
/// result may be a non-float. So its result is definitely-float **only if
/// both operands are definitely-float** (then no type miss is possible).
/// The only unconditional float witnesses are float-tagged constants. All
/// other sources — args, `HeapLoad` (struct fields), call results, int ops
/// — are treated as non-float. Loads of definitely-float locals propagate.
///
/// Loop-carried accumulators (`s = s + e` where `s` is float-initialised)
/// need an optimistic fixpoint: candidate locals start float and are
/// removed when a store is shown non-float. The inner register fixpoint is
/// recomputed against the shrinking candidate set until both stabilise.
pub fn analyze_float_types(instructions: &[Instruction]) -> FloatTypes {
    analyze_core(instructions, AnalyzeOpts::default())
}

/// Options that relax the *sound-by-default* float analysis for the
/// float-parameter-versioning planner (see `plan_float_param_version`). With
/// the defaults, `analyze_core` is the sound analysis used by the normal unbox
/// pass. With `float_arg_indices` set, it computes what *would* be provably
/// float if exactly those parameters were guaranteed float at entry — the
/// optimistic set a guarded fast version could rely on.
#[derive(Clone, Default)]
struct AnalyzeOpts {
    /// Treat an argument register as a float witness iff its argument index
    /// is in this set. `None` = guard nothing via args (the sound default).
    /// Used by the planner to model "what is provably float if exactly these
    /// params are guarded float at entry".
    float_arg_indices: Option<std::collections::HashSet<usize>>,
    /// Model the `TailRecurse` back-edge as float stores to the parameter
    /// slots `local0..localN` (sound dataflow — the values do flow there —
    /// but only useful in conjunction with knowing the entry is float).
    model_tailrecurse: bool,
}

impl AnalyzeOpts {
    /// Whether argument register `v` is an assumed-float witness under these
    /// options.
    fn arg_is_float(&self, v: &Value) -> bool {
        let Value::Register(r) = v else { return false };
        let Some(arg) = r.argument else { return false };
        self.float_arg_indices
            .as_ref()
            .is_some_and(|s| s.contains(&arg))
    }
}

fn is_arg_reg(v: &Value) -> bool {
    matches!(v, Value::Register(r) if r.argument.is_some())
}

/// All `(slot, src)` stores of an instruction. With `model_tailrecurse`, a
/// `TailRecurse(_, args)` is modelled as storing `args[i]` into `local{i}`
/// (parameters are the first locals, in order).
fn collect_stores(ins: &Instruction, model_tailrecurse: bool) -> Vec<(usize, Value)> {
    if let Some((slot, src)) = local_store(ins) {
        return vec![(slot, src.clone())];
    }
    if model_tailrecurse {
        if let Instruction::TailRecurse(_, args) = ins {
            return args
                .iter()
                .enumerate()
                .map(|(i, v)| (i, v.clone()))
                .collect();
        }
    }
    vec![]
}

fn analyze_core(instructions: &[Instruction], opts: AnalyzeOpts) -> FloatTypes {
    let wit = |v: &Value, fr: &HashSet<usize>, cand: &HashSet<usize>| {
        is_float_value(v, fr, cand) || opts.arg_is_float(v)
    };

    // Optimistic: every slot with a store starts as a float candidate.
    let mut candidate: HashSet<usize> = HashSet::new();
    for ins in instructions {
        for (slot, _) in collect_stores(ins, opts.model_tailrecurse) {
            candidate.insert(slot);
        }
    }

    let mut float_regs: HashSet<usize> = HashSet::new();
    loop {
        // Inner fixpoint: grow float_regs given the current candidate set.
        float_regs.clear();
        loop {
            let mut changed = false;
            for ins in instructions {
                let def: Option<usize> = match ins {
                    // Sound: float result only if both operands are float.
                    Instruction::FloatBinOp { dst, a, b, .. } => {
                        if wit(a, &float_regs, &candidate) && wit(b, &float_regs, &candidate) {
                            reg_index(dst)
                        } else {
                            None
                        }
                    }
                    Instruction::Assign(dst, src) | Instruction::LoadLocal(dst, src)
                        if wit(src, &float_regs, &candidate) =>
                    {
                        reg_index(dst)
                    }
                    _ => None,
                };
                if let Some(r) = def {
                    if float_regs.insert(r) {
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Outer: a candidate stays float iff every store is float.
        let mut all_float: HashMap<usize, bool> = HashMap::new();
        for ins in instructions {
            for (slot, src) in collect_stores(ins, opts.model_tailrecurse) {
                let f = wit(&src, &float_regs, &candidate);
                let entry = all_float.entry(slot).or_insert(true);
                *entry = *entry && f;
            }
        }
        let next: HashSet<usize> = candidate
            .iter()
            .copied()
            .filter(|s| all_float.get(s).copied().unwrap_or(false))
            .collect();
        if next == candidate {
            break;
        }
        candidate = next;
    }

    FloatTypes {
        locals: candidate,
        regs: float_regs,
    }
}

/// A plan to compile a guarded, float-unboxed *fast version* of a function
/// whose float parameters become provably float once guarded at entry.
///
/// The codegen (next step) will: guard each `guard_params` argument is a
/// float at entry; if all pass, run a body whose float locals/regs are
/// `float_types` (driving the existing `unbox_floats`); if any fails, run the
/// ordinary boxed body. Sound because the guard is checked once at entry,
/// before the loop — no per-iteration speculation, no hoist hazard.
#[derive(Debug, Clone)]
pub struct FloatParamVersionPlan {
    /// Parameter slot indices (`local{i}`) to guard-float at entry.
    pub guard_params: Vec<usize>,
    /// The float locals/regs provable once the guarded params are float.
    pub float_types: FloatTypes,
}

/// Parameter slots: locals whose value is stored directly from an argument
/// register in the prologue (`store_local local_i, argN`).
fn param_slots(instructions: &[Instruction]) -> HashSet<usize> {
    let mut out = HashSet::new();
    for ins in instructions {
        if let Some((slot, src)) = local_store(ins) {
            if is_arg_reg(src) {
                out.insert(slot);
            }
        }
    }
    out
}

/// The float types provable if exactly the parameters whose slot index is in
/// `guarded` are known float at entry (the rest of the args stay untyped).
/// Models the `TailRecurse` back-edge too (harmless when absent). Slot index ==
/// argument index for parameters (the prologue stores `argK` into `localK`).
fn analyze_with_guarded(instructions: &[Instruction], guarded: &HashSet<usize>) -> FloatTypes {
    analyze_core(
        instructions,
        AnalyzeOpts {
            float_arg_indices: Some(guarded.clone()),
            model_tailrecurse: true,
        },
    )
}

/// True if some value in `types` is an operand of a `FloatBinOp` — i.e. the
/// unboxed set actually feeds float *arithmetic*, so eliminating the box/unbox
/// round-trips is a real win.
///
/// This is the gate that makes versioning pay off **and** keeps it sound. A
/// guarded float param that never feeds a `FloatBinOp` (it is only copied and
/// passed to a call — e.g. mandelbrot's `draw_row`, which guards `ci` only to
/// hand it to `mandel(cr, ci)`) gains nothing from unboxing: the value is
/// unboxed and immediately re-boxed at the call boundary. Such "unbox a value
/// that only flows into calls" cases are exactly where an unboxed FP value
/// crosses a call (which the regalloc doesn't yet preserve across), so they are
/// both pointless and unsafe. mandelbrot's `mandel`, by contrast, feeds its
/// guarded `cr`/`ci` into real `FloatBinOp` arithmetic — the prize.
/// True if any `FloatBinOp` in the body combines a value the versioning can
/// NOT prove unboxed — a register absent from `types.regs` that isn't a float
/// constant (a heap/field load or global-cell read). Such ops stay boxed even
/// in the fast version, so their presence means versioning would duplicate the
/// body without removing the dominant box/unbox traffic (the `nbody/advance`
/// shape). Used as a profitability gate, not a soundness gate.
fn has_uncovered_float_arith(instructions: &[Instruction], types: &FloatTypes) -> bool {
    let covered = |v: &Value| match v {
        Value::Register(r) => types.regs.contains(&r.index),
        _ => is_float_const(v),
    };
    instructions
        .iter()
        .any(|i| matches!(i, Instruction::FloatBinOp { a, b, .. } if !covered(a) || !covered(b)))
}

fn float_types_feed_arithmetic(
    instructions: &[Instruction],
    types: &FloatTypes,
    baseline: &FloatTypes,
) -> bool {
    // A FloatBinOp operand must be a register that is float *because of the
    // guards* — in `types.regs` but NOT already float in the sound baseline.
    // This excludes float-*constant* registers (e.g. the `1.5` in `cr - 1.5`,
    // which is float regardless of any guard) so draw_row — whose only float
    // ops combine a const with a non-guarded value, never a guarded one — is
    // correctly rejected, while `mandel`'s guarded `cr`/`ci` (which genuinely
    // feed the arithmetic) is accepted.
    let unlocked = |v: &Value| {
        matches!(v, Value::Register(r)
            if types.regs.contains(&r.index) && !baseline.regs.contains(&r.index))
    };
    instructions
        .iter()
        .any(|i| matches!(i, Instruction::FloatBinOp { a, b, .. } if unlocked(a) || unlocked(b)))
}

/// Decide whether `instructions` is a float-parameter-versioning candidate:
/// a function with a loop-carried float chain that is boxed today only because
/// its float values originate in arguments. Returns the params to guard and the
/// float set a guarded fast version could unbox.
///
/// Handles both the TCO'd (`TailRecurse`) and the `while`/`loop` (carried via
/// `StoreLocal` back-edge) shapes — the only requirement is a loop, so a
/// straight-line function (no carried state) is never versioned.
///
/// Conservative on two axes:
/// - **Necessity-minimised guards.** Starting from all parameters, a guard is
///   dropped whenever assuming it float does not enlarge the *derived* (non-
///   parameter) float set. This excludes integer params (e.g. a loop bound `n`)
///   whose float assumption unlocks nothing — guarding them would only produce a
///   dead fast path that always misses.
/// - **Strict win.** Returns `None` unless the guarded set proves *strictly*
///   more float locals than the sound baseline (there is a real boxing win).
///
/// **Soundness of the returned `float_types`:** it is computed assuming float
/// for *exactly* the guarded params, so an unguarded param never enters the
/// unbox set. The entry guards the codegen inserts establish that assumption at
/// runtime; a guard miss takes the slow (re-invoke generic) path.
///
/// **Arithmetic-benefit restriction.** Only versions when some unboxed value
/// actually feeds a `FloatBinOp` (see `float_types_feed_arithmetic`). A guarded
/// param that is merely copied and passed to a call (e.g. mandelbrot's
/// `draw_row`, which guards `ci` only to hand it to `mandel(cr, ci)`) gains
/// nothing from unboxing and would push an unboxed FP value across a call — both
/// pointless and unsafe (the regalloc doesn't preserve FP across calls yet). The
/// prize, `mandel`, feeds its guarded params into real float arithmetic.
pub fn plan_float_param_version(instructions: &[Instruction]) -> Option<FloatParamVersionPlan> {
    // Require a loop carrying state (TCO back-edge or a flat back-edge jump).
    let has_loop = instructions
        .iter()
        .any(|i| matches!(i, Instruction::TailRecurse(..)))
        || !crate::ir_loops::flat_ir_loops(instructions).is_empty();
    if !has_loop {
        return None;
    }

    let params = param_slots(instructions);
    if params.is_empty() {
        return None;
    }
    let baseline = analyze_core(instructions, AnalyzeOpts::default());

    // Derived (non-guarded-param) float locals unlocked by guarding `g`.
    let derived = |g: &HashSet<usize>| -> HashSet<usize> {
        analyze_with_guarded(instructions, g)
            .locals
            .difference(g)
            .copied()
            .collect()
    };

    // Minimise: drop any guard whose removal preserves the derived float set.
    let mut guarded: HashSet<usize> = params;
    loop {
        let cur = derived(&guarded);
        let mut order: Vec<usize> = guarded.iter().copied().collect();
        order.sort_unstable();
        let mut removed = false;
        for p in order {
            let mut without = guarded.clone();
            without.remove(&p);
            if derived(&without).is_superset(&cur) {
                guarded = without;
                removed = true;
                break;
            }
        }
        if !removed {
            break;
        }
    }
    if guarded.is_empty() {
        return None;
    }

    let float_types = analyze_with_guarded(instructions, &guarded);
    // The fast version must prove strictly more float locals than today.
    if !float_types.locals.is_superset(&baseline.locals) || float_types.locals == baseline.locals {
        return None;
    }
    // The unboxed set must feed real float arithmetic — otherwise versioning is
    // pointless (unbox-then-rebox) and risks pushing an unboxed FP value across
    // a call (see `float_types_feed_arithmetic`).
    if !float_types_feed_arithmetic(instructions, &float_types, &baseline) {
        return None;
    }
    // ...and the unboxed set must cover essentially ALL the loop's float
    // arithmetic. If the body also does float ops over values versioning can't
    // prove unboxed (heap/field loads, global cells — `nbody/advance`'s body
    // floats), those stay boxed regardless, so duplicating the whole body
    // behind a param guard adds guard + icache cost without removing their
    // box/unbox traffic — a measured net loss. Versioning only pays when the
    // guarded-derived set is the loop's float work (mandel: all-local accums).
    if has_uncovered_float_arith(instructions, &float_types) {
        return None;
    }
    let mut guard_params: Vec<usize> = guarded.into_iter().collect();
    guard_params.sort_unstable();
    Some(FloatParamVersionPlan {
        guard_params,
        float_types,
    })
}

/// Opportunity stats for IR-level guarded-float *region versioning* (SSA
/// spec stage 3). Counts deferred `FloatBinOp`s that are *speculative* (not
/// provably float — operands come from field `HeapLoad`s etc.) and, among
/// those, how many are *chained* (an operand is another speculative op's
/// result). Chained speculative ops are exactly the box→unbox round-trips
/// that region versioning eliminates: guard the region's leaves once, run
/// it unboxed, box only the escapes. Returns
/// `(total_float_binops, speculative, chained_speculative)`.
///
/// Behaviour-neutral analysis — used only to validate the opportunity and
/// (later) to drive the versioning codegen. A high `chained` count on a
/// benchmark means the prize is reachable at this layer.
pub fn speculative_chain_stats(
    instructions: &[Instruction],
    types: &FloatTypes,
) -> (usize, usize, usize) {
    let def_float = |v: &Value| match v {
        Value::Register(r) => types.regs.contains(&r.index),
        _ => is_float_const(v),
    };
    let is_speculative = |a: &Value, b: &Value| !(def_float(a) && def_float(b));

    // Registers defined by a speculative FloatBinOp.
    let mut spec_dsts: HashSet<usize> = HashSet::new();
    let mut total = 0usize;
    let mut speculative = 0usize;
    for ins in instructions {
        if let Instruction::FloatBinOp { dst, a, b, .. } = ins {
            total += 1;
            if is_speculative(a, b) {
                speculative += 1;
                if let Value::Register(r) = dst {
                    spec_dsts.insert(r.index);
                }
            }
        }
    }

    // Chained: a speculative op fed by another speculative op's result.
    let mut chained = 0usize;
    for ins in instructions {
        if let Instruction::FloatBinOp { a, b, .. } = ins {
            if is_speculative(a, b) {
                let fed_by_spec = [a, b]
                    .iter()
                    .any(|v| matches!(v, Value::Register(r) if spec_dsts.contains(&r.index)));
                if fed_by_spec {
                    chained += 1;
                }
            }
        }
    }
    (total, speculative, chained)
}

/// Local-slot indices referenced anywhere in `inst`, recovered from its
/// rendered form (`Value::Local(n)` always prints as `localN`). Used only
/// by the conservative safety check below, so the string scan is fine.
fn rendered_local_indices(inst: &Instruction) -> Vec<usize> {
    let s = inst.pretty_print();
    let mut out = Vec::new();
    let mut rest = s.as_str();
    while let Some(pos) = rest.find("local") {
        let after = &rest[pos + "local".len()..];
        let digits: String = after
            .chars()
            .take_while(|c: &char| c.is_ascii_digit())
            .collect();
        if !digits.is_empty() {
            if let Ok(n) = digits.parse::<usize>() {
                out.push(n);
            }
        }
        rest = &after[digits.len()..];
    }
    out
}

/// True if `inst` reads/writes a local only in a form the unboxing rewrite
/// knows how to convert (`Assign`/`LoadLocal` load, `StoreLocal`/`Assign`
/// store, `Ret`). Any other shape that touches a float local would read the
/// wrong slot region, so the rewrite must bail.
fn is_handled_local_form(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Assign(Value::Register(_), Value::Local(_))
            | Instruction::Assign(Value::Local(_), _)
            | Instruction::LoadLocal(Value::Register(_), Value::Local(_))
            | Instruction::StoreLocal(Value::Local(_), _)
            | Instruction::Ret(Value::Local(_))
    )
}

/// Conservative safety gate: the unboxing rewrite is safe only if every
/// reference to a float local is in a handled form. Otherwise the function
/// must stay fully boxed (clear the float sets). This guarantees no
/// unhandled instruction ever reads a float local's slot as the wrong
/// region.
pub fn unbox_safe(instructions: &[Instruction], float_locals: &HashSet<usize>) -> bool {
    for inst in instructions {
        if is_handled_local_form(inst) {
            continue;
        }
        for n in rendered_local_indices(inst) {
            if float_locals.contains(&n) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::FloatOp;

    fn reg(index: usize) -> Value {
        Value::Register(crate::ir::VirtualRegister {
            argument: None,
            index,
            volatile: false,
            is_physical: false,
        })
    }

    // A boxed-float constant: any 8-byte-aligned address with the Float
    // tag (0b001) in the low bits.
    fn float_const() -> Value {
        Value::RawValue(0x4000 | 0b001)
    }

    fn float_binop(dst: usize, a: Value, b: Value) -> Instruction {
        Instruction::FloatBinOp {
            op: FloatOp::Add,
            dst: reg(dst),
            a,
            b,
            feedback_slot: 0,
            bail_table: 0,
        }
    }

    #[test]
    fn accumulator_loop_locals_are_float() {
        // local1 = 0.0; local2 = 0.0; loop { local1 = local1 + local2 }
        let prog = vec![
            // init: store float consts
            Instruction::StoreLocal(Value::Local(1), float_const()),
            Instruction::StoreLocal(Value::Local(2), float_const()),
            // s + i
            Instruction::Assign(reg(30), Value::Local(1)),
            Instruction::Assign(reg(29), Value::Local(2)),
            float_binop(31, reg(30), reg(29)),
            Instruction::StoreLocal(Value::Local(1), reg(31)),
            // i + 1.0
            Instruction::Assign(reg(34), Value::Local(2)),
            float_binop(35, reg(34), float_const()),
            Instruction::StoreLocal(Value::Local(2), reg(35)),
        ];
        let floats = analyze_float_types(&prog).locals;
        assert!(floats.contains(&1), "local1 (s) should be float");
        assert!(floats.contains(&2), "local2 (i) should be float");
    }

    #[test]
    fn arg_local_is_not_float() {
        // local0 = arg0 (not provably float); used only in a non-float way.
        let prog = vec![
            Instruction::StoreLocal(Value::Local(0), reg(0)),
            Instruction::Assign(reg(5), Value::Local(0)),
        ];
        let floats = analyze_float_types(&prog).locals;
        assert!(!floats.contains(&0), "arg local must not be typed float");
    }

    #[test]
    fn mixed_store_local_is_not_float() {
        // A slot with one float store and one non-float store is not float.
        let prog = vec![
            Instruction::StoreLocal(Value::Local(3), float_const()),
            Instruction::StoreLocal(Value::Local(3), reg(7)), // r7 untyped → non-float
        ];
        let floats = analyze_float_types(&prog).locals;
        assert!(!floats.contains(&3), "mixed-typed slot must not be float");
    }

    fn arg_reg(index: usize, argument: usize) -> Value {
        Value::Register(crate::ir::VirtualRegister {
            argument: Some(argument),
            index,
            volatile: false,
            is_physical: false,
        })
    }

    fn tail_recurse(args: Vec<Value>) -> Instruction {
        Instruction::TailRecurse(reg(99), args)
    }

    fn label(index: usize) -> crate::common::Label {
        crate::common::Label { index }
    }

    /// A minimal `mandel`-shaped TCO'd float loop: a float param `cr` (local0)
    /// arrives as an arg and feeds a *derived* float `t` (local5) recomputed
    /// each iteration; the param is carried back unchanged via `TailRecurse`.
    /// Boxed today (the arg-store poisons local0, so `cr + 1.0` is speculative);
    /// a guarded fast version proves both float.
    fn mandel_like_tco() -> Vec<Instruction> {
        vec![
            Instruction::StoreLocal(Value::Local(0), arg_reg(0, 0)), // cr = arg0
            Instruction::Assign(reg(9), Value::Local(0)),            // cr
            float_binop(11, reg(9), float_const()),                  // t = cr + 1.0
            Instruction::StoreLocal(Value::Local(5), reg(11)),       // local5 = t (derived)
            tail_recurse(vec![reg(9)]),                              // carry cr unchanged
        ]
    }

    /// The same shape as a `while`/`loop` (flat back-edge `Jump`, no
    /// `TailRecurse`): `acc` (local1) is carried via `StoreLocal` across the
    /// back-edge. This is the mandelbrot `mandel` shape.
    fn mandel_like_while() -> Vec<Instruction> {
        vec![
            Instruction::StoreLocal(Value::Local(0), arg_reg(0, 0)), // cr = arg0
            Instruction::StoreLocal(Value::Local(1), float_const()), // acc = 0.0
            Instruction::Label(label(0)),                            // loop header
            Instruction::Assign(reg(9), Value::Local(0)),            // cr
            Instruction::Assign(reg(10), Value::Local(1)),           // acc
            float_binop(12, reg(10), reg(9)),                        // acc + cr
            Instruction::StoreLocal(Value::Local(1), reg(12)),       // acc = acc + cr
            Instruction::Jump(label(0)),                             // back-edge
        ]
    }

    #[test]
    fn baseline_does_not_prove_float_param() {
        // Without guarding, the arg-store poisons local0 → not float, and the
        // speculative `acc + cr` poisons the derived accumulator too.
        let floats = analyze_float_types(&mandel_like_tco()).locals;
        assert!(!floats.contains(&0), "param must be boxed without a guard");
        assert!(
            !floats.contains(&5),
            "derived local must be boxed without a guard"
        );
    }

    #[test]
    fn planner_detects_tco_candidate() {
        let plan = plan_float_param_version(&mandel_like_tco())
            .expect("mandel-like TCO float loop should be a candidate");
        assert_eq!(
            plan.guard_params,
            vec![0],
            "only the float param is guarded"
        );
        assert!(
            plan.float_types.locals.contains(&0) && plan.float_types.locals.contains(&5),
            "guarded fast version proves param + derived local float"
        );
    }

    #[test]
    fn planner_detects_while_loop_candidate() {
        let plan = plan_float_param_version(&mandel_like_while())
            .expect("mandel-like while-loop float loop should be a candidate");
        assert_eq!(
            plan.guard_params,
            vec![0],
            "only the float param is guarded"
        );
        assert!(
            plan.float_types.locals.contains(&1),
            "guarded fast version proves the derived acc float"
        );
    }

    #[test]
    fn planner_does_not_guard_int_param() {
        // Two params: an int `n` (local0, used only in an int add) and a float
        // `c` (local1) that feeds a derived float acc (local2). Guarding `n`
        // unlocks nothing float, so the planner must guard only `c`.
        let prog = vec![
            Instruction::StoreLocal(Value::Local(0), arg_reg(0, 0)), // n = arg0
            Instruction::StoreLocal(Value::Local(1), arg_reg(1, 1)), // c = arg1
            Instruction::StoreLocal(Value::Local(2), float_const()), // acc = 0.0
            Instruction::Label(label(0)),
            Instruction::Assign(reg(8), Value::Local(0)), // n
            Instruction::AddInt(reg(9), reg(8), reg(8)),  // int use of n
            Instruction::Assign(reg(10), Value::Local(1)), // c
            Instruction::Assign(reg(11), Value::Local(2)), // acc
            float_binop(12, reg(11), reg(10)),            // acc + c
            Instruction::StoreLocal(Value::Local(2), reg(12)),
            Instruction::Jump(label(0)),
        ];
        let plan =
            plan_float_param_version(&prog).expect("float param c should make this a candidate");
        assert_eq!(
            plan.guard_params,
            vec![1],
            "guard only the float param c, not int n"
        );
        assert!(
            !plan.float_types.locals.contains(&0),
            "int param n must never enter the unbox set"
        );
        assert!(plan.float_types.locals.contains(&2), "derived acc is float");
    }

    #[test]
    fn planner_rejects_field_fed_float_arith() {
        // nbody/advance shape: a guarded float param `c` feeds a derived float
        // acc (a real versioning opportunity in isolation), BUT the loop ALSO
        // does a FloatBinOp combining a value versioning can't prove unboxed —
        // here an int-add result standing in for a heap/field-loaded float.
        // Those ops stay boxed regardless, so duplicating the body behind the
        // `c` guard would add guard + icache cost for no net win → reject (the
        // `has_uncovered_float_arith` profitability gate). Without the extra op
        // this is exactly `planner_does_not_guard_int_param`, which IS accepted.
        let prog = vec![
            Instruction::StoreLocal(Value::Local(0), arg_reg(0, 0)), // n = arg0 (int)
            Instruction::StoreLocal(Value::Local(1), arg_reg(1, 1)), // c = arg1 (float)
            Instruction::StoreLocal(Value::Local(2), float_const()), // acc = 0.0
            Instruction::Label(label(0)),
            Instruction::Assign(reg(8), Value::Local(0)), // n
            Instruction::AddInt(reg(9), reg(8), reg(8)),  // int value (uncovered operand)
            Instruction::Assign(reg(10), Value::Local(1)), // c
            Instruction::Assign(reg(11), Value::Local(2)), // acc
            float_binop(12, reg(11), reg(10)),            // acc + c (both covered)
            float_binop(13, reg(12), reg(9)),             // (acc+c) + <uncovered field-fed>
            Instruction::StoreLocal(Value::Local(2), reg(13)),
            Instruction::Jump(label(0)),
        ];
        assert!(
            plan_float_param_version(&prog).is_none(),
            "field-fed float arithmetic must suppress versioning"
        );
    }

    #[test]
    fn planner_rejects_non_loop_function() {
        // No loop (no TailRecurse, no back-edge) → not the carried-state case.
        let prog = vec![
            Instruction::StoreLocal(Value::Local(0), arg_reg(0, 0)),
            Instruction::Assign(reg(10), Value::Local(0)),
            float_binop(11, reg(10), float_const()),
        ];
        assert!(plan_float_param_version(&prog).is_none());
    }

    #[test]
    fn planner_rejects_when_no_extra_floats_unlocked() {
        // A loop whose carried state is plainly an int counter: guarding args
        // float unlocks nothing → no plan.
        let prog = vec![
            Instruction::StoreLocal(Value::Local(0), arg_reg(0, 0)),
            Instruction::Assign(reg(10), Value::Local(0)),
            // integer add (not a FloatBinOp) — nothing float to unlock
            Instruction::AddInt(reg(11), reg(10), reg(10)),
            tail_recurse(vec![reg(11)]),
        ];
        assert!(plan_float_param_version(&prog).is_none());
    }
}
