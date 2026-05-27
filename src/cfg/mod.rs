//! Control-flow-graph layer for the SSA pipeline.
//!
//! This module defines the data types for a CFG-shaped IR with explicit
//! block parameters and stack-slot locals. It is the input to mem2reg
//! (Phase 2) and the output of CFG construction from the legacy linear IR
//! (Phase 1).
//!
//! The full architectural contract is in `docs/SSA_ARCHITECTURE.md`; the
//! invariants enforced here are listed as **I1–I8** in that document.
//! Changes that loosen any invariant must update the spec in the same
//! commit and pass the `/ssa-review` skill.
//!
//! Phase 0 (this file plus `verify.rs`) lands only the data types and the
//! verifier. There is no construction from legacy IR yet; that lands in
//! Phase 1. The types intentionally omit a `Legacy(_)` passthrough op —
//! the rule "implement properly, no silent stubs" applies, so op variants
//! are added when their construction lands rather than as catch-alls.

#![allow(dead_code)]

pub mod builder;
pub mod verify;

use crate::ir::Condition;

/// Register class. Tracked on every `VReg` so spill / reload picks slot
/// width and instruction from the class, never from the surrounding op
/// (Invariant **I4**).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegClass {
    /// General-purpose / integer / pointer register class.
    Gp,
    /// Floating-point register class.
    Fp,
}

/// SSA virtual register. Carries its class so the verifier and the
/// allocator can never lose it (I4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg {
    pub index: u32,
    pub class: RegClass,
}

/// Stack-slot index. `Value::Local(n)` from the legacy IR becomes a
/// `SlotId(n)` here. Slots stay materialized as `SlotLoad` / `SlotStore`
/// at construction time (I6); mem2reg promotes hot slots to SSA values
/// only when profitable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId(pub u32);

/// Block identifier within a single function's CFG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// A basic block in the CFG.
///
/// - `params` is empty at construction; mem2reg fills it when promoting a
///   slot (I3 / I6).
/// - `body` is straight-line: no labels, no embedded terminators
///   (Invariant **I1**).
/// - `terminator` is exactly one. The type system forbids zero-or-many.
/// - `predecessors` is maintained by whatever pass mutates terminators;
///   `verify::check_predecessors` cross-checks the two.
#[derive(Debug, Clone)]
pub struct Block {
    pub params: Vec<VReg>,
    pub body: Vec<Op>,
    pub terminator: Terminator,
    pub predecessors: Vec<BlockId>,
}

/// CFG-layer op. Variants are added as Phase 1b lowers their legacy IR
/// counterparts. Per the project rule, every variant corresponds to a
/// real, fully-translated legacy op — no catch-all `Legacy(_)` stub.
#[derive(Debug, Clone)]
pub enum Op {
    /// Read a stack-slot local into a fresh VReg.
    SlotLoad { dst: VReg, slot: SlotId },
    /// Write a VReg into a stack-slot local.
    SlotStore { slot: SlotId, src: VReg },
    /// Tagged-integer addition. Two-input, one-output. Both operands and
    /// the destination are GP-class.
    AddInt { dst: VReg, lhs: VReg, rhs: VReg },
}

/// Block terminator. Exactly one per block (**I1**). All control transfer
/// carries block-param args (**I3**); fall-through is not permitted.
#[derive(Debug, Clone)]
pub enum Terminator {
    Jump {
        target: BlockId,
        args: Vec<VReg>,
    },
    Branch {
        cond: Condition,
        lhs: VReg,
        rhs: VReg,
        t_target: BlockId,
        t_args: Vec<VReg>,
        f_target: BlockId,
        f_args: Vec<VReg>,
    },
    Ret {
        value: VReg,
    },
    /// Exception path. The handler edge is explicit so the verifier can
    /// dominance-check both the normal and the exception path (**I5**).
    Throw {
        value: VReg,
        handler: BlockId,
        handler_args: Vec<VReg>,
    },
    /// Block must not execute. Used for newly created empty blocks before
    /// their terminator is filled in, and as a poison value the verifier
    /// reports rather than silently passing.
    Unreachable,
}

/// A function in the SSA pipeline. Built from a legacy `Ir` in Phase 1.
#[derive(Debug, Clone)]
pub struct CfgFunction {
    pub debug_name: Option<String>,
    pub entry: BlockId,
    pub blocks: Vec<Block>,
    /// Per-VReg class. Indexed by `VReg::index`. The verifier asserts
    /// every used VReg appears here and matches the class on the VReg
    /// itself (**I4**).
    pub vreg_classes: Vec<RegClass>,
    /// Number of stack slots reserved for `Value::Local` storage. Stays
    /// stable across mem2reg so GC-root machinery and debug info stay
    /// valid (**I6**).
    pub num_slots: u32,
}

impl CfgFunction {
    pub fn new(debug_name: Option<String>, num_slots: u32) -> Self {
        Self {
            debug_name,
            entry: BlockId(0),
            blocks: Vec::new(),
            vreg_classes: Vec::new(),
            num_slots,
        }
    }

    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }

    pub fn block_mut(&mut self, id: BlockId) -> &mut Block {
        &mut self.blocks[id.0 as usize]
    }

    /// Create a new block with no params, an empty body, an `Unreachable`
    /// terminator, and no predecessors. Caller is responsible for filling
    /// in the terminator before the verifier runs (the `Unreachable`
    /// placeholder is intentional and detectable).
    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(Block {
            params: Vec::new(),
            body: Vec::new(),
            terminator: Terminator::Unreachable,
            predecessors: Vec::new(),
        });
        id
    }

    /// Allocate a fresh VReg of the given class.
    pub fn new_vreg(&mut self, class: RegClass) -> VReg {
        let index = self.vreg_classes.len() as u32;
        self.vreg_classes.push(class);
        VReg { index, class }
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn num_vregs(&self) -> usize {
        self.vreg_classes.len()
    }
}

// ---- traversal helpers --------------------------------------------------
//
// Public because the verifier, mem2reg, regalloc, and edge resolution all
// need to walk terminator successors / collect VReg uses uniformly.

impl Terminator {
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Jump { target, .. } => vec![*target],
            Terminator::Branch {
                t_target, f_target, ..
            } => vec![*t_target, *f_target],
            Terminator::Throw { handler, .. } => vec![*handler],
            Terminator::Ret { .. } | Terminator::Unreachable => vec![],
        }
    }

    /// All VRegs read or passed-as-arg by this terminator.
    pub fn uses(&self) -> Vec<VReg> {
        match self {
            Terminator::Jump { args, .. } => args.clone(),
            Terminator::Branch {
                lhs,
                rhs,
                t_args,
                f_args,
                ..
            } => {
                let mut v = Vec::with_capacity(2 + t_args.len() + f_args.len());
                v.push(*lhs);
                v.push(*rhs);
                v.extend_from_slice(t_args);
                v.extend_from_slice(f_args);
                v
            }
            Terminator::Ret { value } => vec![*value],
            Terminator::Throw {
                value,
                handler_args,
                ..
            } => {
                let mut v = Vec::with_capacity(1 + handler_args.len());
                v.push(*value);
                v.extend_from_slice(handler_args);
                v
            }
            Terminator::Unreachable => vec![],
        }
    }
}

impl Op {
    pub fn defs(&self) -> Vec<VReg> {
        match self {
            Op::SlotLoad { dst, .. } => vec![*dst],
            Op::SlotStore { .. } => vec![],
            Op::AddInt { dst, .. } => vec![*dst],
        }
    }

    pub fn uses(&self) -> Vec<VReg> {
        match self {
            Op::SlotLoad { .. } => vec![],
            Op::SlotStore { src, .. } => vec![*src],
            Op::AddInt { lhs, rhs, .. } => vec![*lhs, *rhs],
        }
    }
}
