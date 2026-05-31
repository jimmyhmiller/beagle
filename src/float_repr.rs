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
#[derive(Debug, Default)]
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
    // Optimistic: every slot with a store starts as a float candidate.
    let mut candidate: HashSet<usize> = HashSet::new();
    for ins in instructions {
        if let Some((slot, _)) = local_store(ins) {
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
                        if is_float_value(a, &float_regs, &candidate)
                            && is_float_value(b, &float_regs, &candidate)
                        {
                            reg_index(dst)
                        } else {
                            None
                        }
                    }
                    Instruction::Assign(dst, src) | Instruction::LoadLocal(dst, src)
                        if is_float_value(src, &float_regs, &candidate) =>
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
            if let Some((slot, src)) = local_store(ins) {
                let f = is_float_value(src, &float_regs, &candidate);
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
}
