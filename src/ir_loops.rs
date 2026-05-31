//! Flat-IR loop detection (pre-CFG).
//!
//! At `unbox_floats` time the IR is a flat instruction list with `Label`s
//! and `Jump`/`JumpIf`-to-label. A **back-edge** is a (conditional or
//! unconditional) jump whose target label sits EARLIER in the stream than
//! the jump itself. The loop **header** is the target-label position; the
//! **latch** is the jump position; the **body** is the contiguous range
//! `[header_pos, latch_pos]`. This is sound for the structured loops the
//! Beagle front-end emits (`while`/`loop`/`for`), and nested loops nest by
//! range containment.
//!
//! Used by the field-fed float loop-body versioning pass (SSA spec stage
//! 3): the body range is what gets cloned into fast/slow versions.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::ir::Instruction;

/// A loop discovered in the flat IR, keyed by header position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatLoop {
    /// `Label::index` of the loop header.
    pub header_label: usize,
    /// Instruction position of the header `Label`.
    pub header_pos: usize,
    /// Instruction position of the (furthest) back-edge jump.
    pub latch_pos: usize,
    /// Inclusive instruction range `[header_pos, latch_pos]` — the loop body.
    pub body: (usize, usize),
}

impl FlatLoop {
    /// Whether instruction position `p` is inside this loop's body range.
    pub fn contains_pos(&self, p: usize) -> bool {
        p >= self.body.0 && p <= self.body.1
    }
}

/// Position of each label index in the instruction stream.
fn label_positions(instructions: &[Instruction]) -> HashMap<usize, usize> {
    let mut m = HashMap::new();
    for (i, ins) in instructions.iter().enumerate() {
        if let Instruction::Label(l) = ins {
            m.insert(l.index, i);
        }
    }
    m
}

/// The target label index of a jump instruction, if any.
fn jump_target(ins: &Instruction) -> Option<usize> {
    match ins {
        Instruction::Jump(l) => Some(l.index),
        Instruction::JumpIf(l, ..) => Some(l.index),
        _ => None,
    }
}

/// Why a loop is (not) soundly versionable for guarded float unboxing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Versionable {
    /// Sound to version: has speculative field-fed float ops AND every heap
    /// read precedes every heap write in the body (so guarded reads can be
    /// hoisted above all writes — a guard-miss bail re-runs the slow body
    /// with no doubled side effects, and no hoisted read crosses a write to
    /// a possibly-aliasing location).
    Yes,
    /// No speculative (field-fed) float op — nothing to unbox.
    NoSpeculativeFloat,
    /// A heap WRITE precedes a heap READ in the body. Hoisting reads above
    /// writes would be unsound (the read may alias the written location —
    /// e.g. `p.x = p.x + dt*p.vx` reads `p.vx` after writing it). Sound
    /// versioning of this shape needs alias analysis we don't have.
    WriteBeforeRead { first_write: usize, last_read: usize },
}

/// Conservative hoist-safety / versionability check (SSA spec stage 3).
/// `is_float` decides whether a `Value` is *definitely* float (so a
/// `FloatBinOp` is speculative iff NOT both operands are definitely-float).
/// Positions in the returned reason are absolute instruction indices.
pub fn versionable_float_loop(
    instructions: &[Instruction],
    lp: &FlatLoop,
    is_float: &dyn Fn(&crate::ir::Value) -> bool,
) -> Versionable {
    let mut has_spec = false;
    let mut last_read: Option<usize> = None;
    let mut first_write: Option<usize> = None;
    for p in lp.body.0..=lp.body.1 {
        match &instructions[p] {
            Instruction::FloatBinOp { a, b, .. } => {
                if !(is_float(a) && is_float(b)) {
                    has_spec = true;
                }
            }
            Instruction::HeapLoad(..) | Instruction::HeapLoadReg(..) => last_read = Some(p),
            Instruction::HeapStore(..)
            | Instruction::HeapStoreOffset(..)
            | Instruction::HeapStoreOffsetReg(..) => {
                first_write.get_or_insert(p);
            }
            _ => {}
        }
    }
    if !has_spec {
        return Versionable::NoSpeculativeFloat;
    }
    // Hoist-safe iff every read precedes every write (or there are no writes).
    match (last_read, first_write) {
        (Some(lr), Some(fw)) if fw <= lr => Versionable::WriteBeforeRead {
            first_write: fw,
            last_read: lr,
        },
        _ => Versionable::Yes,
    }
}

/// Detect loops in the flat IR via backward jumps. One entry per loop
/// header, merging multiple back-edges to the same header (the furthest
/// latch becomes the body end). Sorted by header position. Inner loops of
/// a nest appear as separate entries whose body range is contained in the
/// enclosing loop's range.
pub fn flat_ir_loops(instructions: &[Instruction]) -> Vec<FlatLoop> {
    let pos = label_positions(instructions);
    // header_pos -> (header_label, furthest latch_pos)
    let mut by_header: HashMap<usize, (usize, usize)> = HashMap::new();
    for (p, ins) in instructions.iter().enumerate() {
        if let Some(tgt) = jump_target(ins) {
            if let Some(&hpos) = pos.get(&tgt) {
                if hpos < p {
                    let e = by_header.entry(hpos).or_insert((tgt, p));
                    if p > e.1 {
                        e.1 = p;
                    }
                }
            }
        }
    }
    let mut loops: Vec<FlatLoop> = by_header
        .into_iter()
        .map(|(hpos, (label, latch))| FlatLoop {
            header_label: label,
            header_pos: hpos,
            latch_pos: latch,
            body: (hpos, latch),
        })
        .collect();
    loops.sort_by_key(|l| l.header_pos);
    loops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Label;
    use crate::ir::{Condition, Value};

    fn lbl(i: usize) -> Label {
        Label { index: i }
    }
    fn jump_if(i: usize) -> Instruction {
        Instruction::JumpIf(lbl(i), Condition::Equal, Value::Null, Value::Null)
    }
    fn reg(index: usize) -> Value {
        Value::Register(crate::ir::VirtualRegister {
            argument: None,
            index,
            volatile: false,
            is_physical: false,
        })
    }
    fn spec_floatbinop() -> Instruction {
        Instruction::FloatBinOp {
            op: crate::ir::FloatOp::Add,
            dst: reg(1),
            a: reg(2),
            b: reg(3),
            feedback_slot: 0,
            bail_table: 0,
        }
    }

    #[test]
    fn versionable_reads_before_writes() {
        let prog = vec![
            Instruction::Label(lbl(0)),
            spec_floatbinop(),
            Instruction::HeapLoad(reg(4), reg(5), 0),
            Instruction::HeapStore(reg(6), reg(7)),
            Instruction::Jump(lbl(0)),
        ];
        let lp = FlatLoop {
            header_label: 0,
            header_pos: 0,
            latch_pos: 4,
            body: (0, 4),
        };
        assert_eq!(
            versionable_float_loop(&prog, &lp, &|_| false),
            Versionable::Yes
        );
    }

    #[test]
    fn not_versionable_write_before_read() {
        // The nbody/probe shape: a write precedes a later read (read-after-
        // write through a field) — hoisting the read would be unsound.
        let prog = vec![
            Instruction::Label(lbl(0)),
            spec_floatbinop(),
            Instruction::HeapStore(reg(6), reg(7)), // write first
            Instruction::HeapLoad(reg(4), reg(5), 0), // read after
            Instruction::Jump(lbl(0)),
        ];
        let lp = FlatLoop {
            header_label: 0,
            header_pos: 0,
            latch_pos: 4,
            body: (0, 4),
        };
        assert!(matches!(
            versionable_float_loop(&prog, &lp, &|_| false),
            Versionable::WriteBeforeRead { .. }
        ));
    }

    #[test]
    fn no_speculative_float_op() {
        // FloatBinOp whose operands are definitely-float (is_float == true) is
        // NOT speculative → nothing to unbox.
        let prog = vec![
            Instruction::Label(lbl(0)),
            spec_floatbinop(),
            Instruction::Jump(lbl(0)),
        ];
        let lp = FlatLoop {
            header_label: 0,
            header_pos: 0,
            latch_pos: 2,
            body: (0, 2),
        };
        assert_eq!(
            versionable_float_loop(&prog, &lp, &|_| true),
            Versionable::NoSpeculativeFloat
        );
    }

    #[test]
    fn no_loop_forward_jumps_only() {
        // jump forward to a label that appears LATER → not a back-edge.
        let prog = vec![
            jump_if(0),                 // 0: forward jump to label 0 (below)
            Instruction::Breakpoint,    // 1
            Instruction::Label(lbl(0)), // 2 (after the jump)
            Instruction::Breakpoint,    // 3
        ];
        assert!(flat_ir_loops(&prog).is_empty());
    }

    #[test]
    fn simple_loop() {
        // header label 0, body, conditional back-edge to 0.
        let prog = vec![
            Instruction::Breakpoint,    // 0
            Instruction::Label(lbl(0)), // 1  header
            Instruction::Breakpoint,    // 2  body
            jump_if(0),                 // 3  latch (back-edge)
            Instruction::Breakpoint,    // 4  after loop
        ];
        let loops = flat_ir_loops(&prog);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].header_label, 0);
        assert_eq!(loops[0].header_pos, 1);
        assert_eq!(loops[0].latch_pos, 3);
        assert_eq!(loops[0].body, (1, 3));
    }

    #[test]
    fn unconditional_back_edge() {
        let prog = vec![
            Instruction::Label(lbl(5)), // 0 header
            Instruction::Breakpoint,    // 1
            Instruction::Jump(lbl(5)),  // 2 latch
        ];
        let loops = flat_ir_loops(&prog);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].body, (0, 2));
    }

    #[test]
    fn nested_loops() {
        // outer header 0 ... inner header 1 ... inner back-edge to 1 ...
        // outer back-edge to 0.
        let prog = vec![
            Instruction::Label(lbl(0)), // 0 outer header
            Instruction::Breakpoint,    // 1
            Instruction::Label(lbl(1)), // 2 inner header
            Instruction::Breakpoint,    // 3 inner body
            jump_if(1),                 // 4 inner latch
            Instruction::Breakpoint,    // 5
            jump_if(0),                 // 6 outer latch
        ];
        let loops = flat_ir_loops(&prog);
        assert_eq!(loops.len(), 2);
        // sorted by header_pos: outer (0) then inner (2)
        let outer = &loops[0];
        let inner = &loops[1];
        assert_eq!(outer.body, (0, 6));
        assert_eq!(inner.body, (2, 4));
        // inner range contained in outer range
        assert!(inner.body.0 >= outer.body.0 && inner.body.1 <= outer.body.1);
    }

    #[test]
    fn multiple_back_edges_same_header_merge() {
        // two back-edges to header 0 → one loop, body ends at the furthest.
        let prog = vec![
            Instruction::Label(lbl(0)), // 0 header
            Instruction::Breakpoint,    // 1
            jump_if(0),                 // 2 back-edge A
            Instruction::Breakpoint,    // 3
            jump_if(0),                 // 4 back-edge B (furthest)
        ];
        let loops = flat_ir_loops(&prog);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].body, (0, 4));
    }
}
