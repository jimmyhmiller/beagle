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
pub mod dom;
pub mod dump;
pub mod lift_vregs;
pub mod mem2reg;
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
///
/// Bail-out arithmetic (Sub, Mul, Div, Modulo, Shifts, GuardInt, GuardFloat)
/// is not in this enum — they need a two-successor terminator variant and
/// land in Phase 1b-3. Calls, exception/continuation ops, and inline bump
/// allocation are deferred to 1b-4 / 1b-5.
#[derive(Debug, Clone)]
pub enum Op {
    // ---- Slots (I6 — locals stay materialized until mem2reg) ----
    SlotLoad {
        dst: VReg,
        slot: SlotId,
    },
    SlotStore {
        slot: SlotId,
        src: VReg,
    },

    // ---- Moves & constants ----
    /// Register-to-register copy. SSA copy-coalescing may eliminate.
    Move {
        dst: VReg,
        src: VReg,
    },
    /// Tagged integer constant materialization.
    ConstTaggedInt {
        dst: VReg,
        value: i64,
    },
    /// Pointer to a string in the string-constants region.
    ConstStringPtr {
        dst: VReg,
        ptr: usize,
    },
    /// Pointer to a keyword in the keyword-constants region.
    ConstKeywordPtr {
        dst: VReg,
        ptr: usize,
    },
    /// Function id, materialized as a tagged function pointer at runtime.
    ConstFunctionId {
        dst: VReg,
        function_id: usize,
    },
    /// Raw pointer constant (untagged).
    ConstPointer {
        dst: VReg,
        ptr: usize,
    },
    /// Raw 64-bit constant — no tagging applied.
    ConstRawValue {
        dst: VReg,
        value: u64,
    },
    ConstTrue {
        dst: VReg,
    },
    ConstFalse {
        dst: VReg,
    },
    /// Tagged null (legacy uses bit pattern 0b111).
    ConstNull {
        dst: VReg,
    },
    /// Address of a CFG block, used by handler / continuation setup.
    ConstLabelAddress {
        dst: VReg,
        target: BlockId,
    },

    // ---- Integer arithmetic (no bail) ----
    AddInt {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },

    // ---- Comparisons (produce GP 0/1) ----
    Compare {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
        cond: Condition,
    },
    CompareFloat {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
        cond: Condition,
    },

    // ---- Floating-point arithmetic ----
    AddFloat {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    SubFloat {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    MulFloat {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    DivFloat {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },

    // ---- Conversions / bit-moves between classes ----
    IntToFloat {
        dst: VReg,
        src: VReg,
    },
    FRoundToZero {
        dst: VReg,
        src: VReg,
    },
    /// GP→FP bit move (bits of an int reg become bits of an FP reg).
    FmovGpToFp {
        dst: VReg,
        src: VReg,
    },
    /// FP→GP bit move.
    FmovFpToGp {
        dst: VReg,
        src: VReg,
    },

    // ---- Tag bit manipulation (low 3 bits on tagged values) ----
    /// `dst = src OR tag_source` — install tag bits in the low bits of
    /// src. The tag may be a compile-time immediate (the BuiltInType for
    /// the common case) or a runtime-computed register (when the tag is
    /// derived from a struct id or other dynamic source).
    Tag {
        dst: VReg,
        src: VReg,
        tag_source: TagSource,
    },
    Untag {
        dst: VReg,
        src: VReg,
    },
    GetTag {
        dst: VReg,
        src: VReg,
    },

    // ---- Bit ops ----
    And {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    Or {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    Xor {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    AndImm {
        dst: VReg,
        src: VReg,
        imm: u64,
    },
    ShiftRightImmRaw {
        dst: VReg,
        src: VReg,
        imm: i32,
    },

    // ---- Heap memory ----
    HeapLoad {
        dst: VReg,
        base: VReg,
        offset: i32,
    },
    HeapLoadReg {
        dst: VReg,
        base: VReg,
        offset: VReg,
    },
    HeapLoadByteReg {
        dst: VReg,
        base: VReg,
        offset: VReg,
    },
    HeapStore {
        addr: VReg,
        src: VReg,
    },
    HeapStoreOffset {
        base: VReg,
        src: VReg,
        offset: usize,
    },
    HeapStoreOffsetReg {
        base: VReg,
        src: VReg,
        offset: VReg,
    },
    HeapStoreByteOffsetReg {
        base: VReg,
        src: VReg,
        offset: VReg,
    },
    /// Read-modify-write of a single byte at `ptr + offset + byte_offset`,
    /// using `mask` to clear the target byte before ORing `val` in.
    /// `temp1` / `temp2` are scratch defs (the original IR pre-allocates
    /// them as VRegs; we treat them as defs so the verifier sees their
    /// dataflow correctly).
    HeapStoreByteOffsetMasked {
        ptr: VReg,
        val: VReg,
        temp1: VReg,
        temp2: VReg,
        offset: usize,
        byte_offset: usize,
        mask: usize,
    },

    // ---- Atomic memory ----
    AtomicLoad {
        dst: VReg,
        src: VReg,
    },
    AtomicStore {
        addr: VReg,
        src: VReg,
    },
    /// 3-operand CAS — no result, just side-effect.
    CompareAndSwap {
        addr: VReg,
        expected: VReg,
        new: VReg,
    },

    /// Parse a float constant from a string at compile time and store its
    /// bits at offset 1 of `dest`. `temp` is a GP scratch used to hold
    /// the bit pattern between mov-imm and store-on-heap.
    StoreFloatConstant {
        dest: VReg,
        temp: VReg,
        value_text: String,
    },

    // ---- Stack pointer manipulation ----
    PushStack {
        src: VReg,
    },
    PopStack {
        dst: VReg,
    },
    GetStackPointer {
        dst: VReg,
        offset: VReg,
    },
    GetStackPointerImm {
        dst: VReg,
        offset: isize,
    },
    GetFramePointer {
        dst: VReg,
    },
    CurrentStackPosition {
        dst: VReg,
    },

    // ---- Variadic plumbing ----
    ReadArgCount {
        dst: VReg,
    },

    // ---- Misc no-VReg ops & lifetime markers ----
    Breakpoint,
    /// Keeps a VReg live for regalloc bookkeeping; no semantic effect.
    ExtendLifetime {
        src: VReg,
    },
    RecordGcSafepoint,
    /// Inline-feedback bit-mask OR into a runtime feedback word at a
    /// known absolute address. All arguments are compile-time immediates.
    FeedbackOr {
        slot_addr: usize,
        bits: u64,
    },
    /// Per-function entry tier-up check. All immediates.
    TierUpCheck {
        counter_addr: usize,
        name_c_str_ptr: usize,
        trampoline_fn_ptr: usize,
    },

    // ---- Calls (I7 — carry an explicit clobber set) ----
    /// General function call (direct or builtin). The callee target may
    /// be a register (computed function pointer) or a compile-time
    /// constant (function id, raw pointer, etc.).
    Call {
        dst: VReg,
        target: CallTarget,
        args: Vec<VReg>,
        is_builtin: bool,
        clobbers: ClobberSet,
    },
    /// Self-recursive call that returns (NOT tail). Tail self-calls are
    /// rewritten to `Terminator::Jump(entry, args)` per **I8** during
    /// construction.
    Recurse {
        dst: VReg,
        args: Vec<VReg>,
        clobbers: ClobberSet,
    },

    // ---- Exception handling ----
    PushExceptionHandler {
        handler: BlockId,
        result_slot: SlotId,
        builtin_fn_ptr: usize,
    },
    PushResumableExceptionHandler {
        dst: VReg,
        catch_block: BlockId,
        exception_slot: SlotId,
        resume_slot: SlotId,
        builtin_fn_ptr: usize,
    },
    PopExceptionHandler {
        builtin_fn_ptr: usize,
    },
    PopExceptionHandlerById {
        handler_id: VReg,
        builtin_fn_ptr: usize,
    },

    // ---- Delimited continuations & prompts ----
    PushPromptHandler {
        handler: BlockId,
        result_slot: SlotId,
        builtin_fn_ptr: usize,
    },
    PopPromptHandler {
        result: VReg,
        builtin_fn_ptr: usize,
    },
    PushPromptTag {
        tag: VReg,
        abort_block: BlockId,
        result_slot: SlotId,
        builtin_fn_ptr: usize,
    },
    CaptureContinuation {
        dst: VReg,
        resume_block: BlockId,
        result_slot: SlotId,
        builtin_fn_ptr: usize,
    },
    CaptureContinuationTagged {
        dst: VReg,
        resume_block: BlockId,
        result_slot: SlotId,
        builtin_fn_ptr: usize,
        tag: VReg,
    },

    // ---- Algebraic effects ----
    PerformEffect {
        handler: VReg,
        enum_type: VReg,
        op_value: VReg,
        resume_block: BlockId,
        result_slot: SlotId,
        builtin_fn_ptr: usize,
    },
    /// Return from a `shift` — calls return_from_shift with current SP/FP.
    /// No def — execution does not continue normally.
    ReturnFromShift {
        value: VReg,
        cont_ptr: VReg,
        builtin_fn_ptr: usize,
    },
}

/// Set of physical registers that a Call op may clobber, used by the
/// Phase 4 regalloc to decide per live-value how to survive the call
/// (callee-saved placement, spill to a root slot, or rematerialize) per
/// **I7**. Phase 1b uses the conservative default for every call;
/// specific clobber sets are added later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClobberSet {
    /// All caller-saved registers in both pools may be clobbered. This is
    /// Beagle's default ABI assumption for any call across the Rust/JIT
    /// boundary or to another Beagle function.
    AllCallerSaved,
}

/// Source of tag bits for `Op::Tag`. The IR builder emits Tag with either
/// a constant tag (the common case — BuiltInType-derived) or a
/// runtime-computed tag (e.g. derived from a struct id).
#[derive(Debug, Clone)]
pub enum TagSource {
    Register(VReg),
    Bits(u64),
}

/// Callee target for `Op::Call`. The legacy IR's Call op accepts any
/// Value as the function position; we split the common kinds into named
/// variants so SSA passes (mem2reg, regalloc, codegen) can pattern-match
/// without sniffing through a Value wrapper.
#[derive(Debug, Clone)]
pub enum CallTarget {
    /// Computed function pointer in a register (e.g. closure dispatch,
    /// method resolution).
    Register(VReg),
    /// Function-id constant; the codegen resolves this to the tagged
    /// function pointer at emit time.
    FunctionId(usize),
    /// Raw pointer constant — typically a Rust runtime builtin's address
    /// pre-resolved at IR build time.
    Pointer(usize),
    /// Raw 64-bit constant. Rare; covers cases where the IR builder
    /// pre-tagged or pre-shifted a function pointer.
    Raw(u64),
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
    /// Two-successor terminator for ops that may bail to a slow path.
    /// On success the op writes `op`'s `dst` and execution flows to
    /// `fall_through`. On failure (overflow, type mismatch, bump
    /// overflow) execution jumps to `bail` without writing `dst`. Covers
    /// the bail-out arithmetic family (Sub/Mul/Div/Modulo/Shifts), the
    /// type guards (GuardInt, GuardFloat), and InlineBumpAllocate.
    InlineBranch {
        op: InlineBranchOp,
        fall_through: BlockId,
        fall_args: Vec<VReg>,
        bail: BlockId,
        bail_args: Vec<VReg>,
    },
    Ret {
        value: VReg,
    },
    /// Exception throw. `resume` is the resume label that the runtime
    /// returns control to after a handler catches and resumes; if no
    /// handler catches, control never returns there.
    Throw {
        value: VReg,
        resume: BlockId,
        resume_args: Vec<VReg>,
        resume_local: SlotId,
        builtin_fn_ptr: usize,
    },
    /// Block must not execute. Used for newly created empty blocks before
    /// their terminator is filled in, and as a poison value the verifier
    /// reports rather than silently passing.
    Unreachable,
}

/// Inner op-data for `Terminator::InlineBranch`. Each variant has its
/// own def (`dst`) which is written only on the fall-through path.
#[derive(Debug, Clone)]
pub enum InlineBranchOp {
    SubChecked {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    MulChecked {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    DivChecked {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    ModuloChecked {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    ShiftLeftChecked {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    ShiftRightChecked {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    ShiftRightZeroChecked {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    ShiftRightImmChecked {
        dst: VReg,
        src: VReg,
        imm: i32,
    },
    GuardInt {
        dst: VReg,
        src: VReg,
    },
    GuardFloat {
        dst: VReg,
        src: VReg,
    },
    InlineBumpAllocate {
        dst: VReg,
        size_bytes: u32,
        header: u64,
    },
}

impl InlineBranchOp {
    pub fn dst(&self) -> VReg {
        match self {
            InlineBranchOp::SubChecked { dst, .. }
            | InlineBranchOp::MulChecked { dst, .. }
            | InlineBranchOp::DivChecked { dst, .. }
            | InlineBranchOp::ModuloChecked { dst, .. }
            | InlineBranchOp::ShiftLeftChecked { dst, .. }
            | InlineBranchOp::ShiftRightChecked { dst, .. }
            | InlineBranchOp::ShiftRightZeroChecked { dst, .. }
            | InlineBranchOp::ShiftRightImmChecked { dst, .. }
            | InlineBranchOp::GuardInt { dst, .. }
            | InlineBranchOp::GuardFloat { dst, .. }
            | InlineBranchOp::InlineBumpAllocate { dst, .. } => *dst,
        }
    }

    pub fn uses(&self) -> Vec<VReg> {
        match self {
            InlineBranchOp::SubChecked { lhs, rhs, .. }
            | InlineBranchOp::MulChecked { lhs, rhs, .. }
            | InlineBranchOp::DivChecked { lhs, rhs, .. }
            | InlineBranchOp::ModuloChecked { lhs, rhs, .. }
            | InlineBranchOp::ShiftLeftChecked { lhs, rhs, .. }
            | InlineBranchOp::ShiftRightChecked { lhs, rhs, .. }
            | InlineBranchOp::ShiftRightZeroChecked { lhs, rhs, .. } => vec![*lhs, *rhs],
            InlineBranchOp::ShiftRightImmChecked { src, .. }
            | InlineBranchOp::GuardInt { src, .. }
            | InlineBranchOp::GuardFloat { src, .. } => vec![*src],
            InlineBranchOp::InlineBumpAllocate { .. } => vec![],
        }
    }

    pub fn rename_uses(&mut self, rename: &std::collections::HashMap<VReg, VReg>) {
        let apply = |v: &mut VReg| {
            if let Some(&new) = rename.get(v) {
                *v = new;
            }
        };
        match self {
            InlineBranchOp::SubChecked { lhs, rhs, .. }
            | InlineBranchOp::MulChecked { lhs, rhs, .. }
            | InlineBranchOp::DivChecked { lhs, rhs, .. }
            | InlineBranchOp::ModuloChecked { lhs, rhs, .. }
            | InlineBranchOp::ShiftLeftChecked { lhs, rhs, .. }
            | InlineBranchOp::ShiftRightChecked { lhs, rhs, .. }
            | InlineBranchOp::ShiftRightZeroChecked { lhs, rhs, .. } => {
                apply(lhs);
                apply(rhs);
            }
            InlineBranchOp::ShiftRightImmChecked { src, .. }
            | InlineBranchOp::GuardInt { src, .. }
            | InlineBranchOp::GuardFloat { src, .. } => {
                apply(src);
            }
            InlineBranchOp::InlineBumpAllocate { .. } => {}
        }
    }

    /// Mirror of `rename_uses` but for the `dst` def. Used by the lift
    /// pass to give each multi-def site its own unique VReg name so the
    /// result is true SSA.
    pub fn rename_defs(&mut self, rename: &std::collections::HashMap<VReg, VReg>) {
        let apply = |v: &mut VReg| {
            if let Some(&new) = rename.get(v) {
                *v = new;
            }
        };
        match self {
            InlineBranchOp::SubChecked { dst, .. }
            | InlineBranchOp::MulChecked { dst, .. }
            | InlineBranchOp::DivChecked { dst, .. }
            | InlineBranchOp::ModuloChecked { dst, .. }
            | InlineBranchOp::ShiftLeftChecked { dst, .. }
            | InlineBranchOp::ShiftRightChecked { dst, .. }
            | InlineBranchOp::ShiftRightZeroChecked { dst, .. }
            | InlineBranchOp::ShiftRightImmChecked { dst, .. }
            | InlineBranchOp::GuardInt { dst, .. }
            | InlineBranchOp::GuardFloat { dst, .. }
            | InlineBranchOp::InlineBumpAllocate { dst, .. } => apply(dst),
        }
    }
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
            Terminator::InlineBranch {
                fall_through, bail, ..
            } => vec![*fall_through, *bail],
            Terminator::Throw { resume, .. } => vec![*resume],
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
            Terminator::InlineBranch {
                op,
                fall_args,
                bail_args,
                ..
            } => {
                let inner = op.uses();
                let mut v = Vec::with_capacity(inner.len() + fall_args.len() + bail_args.len());
                v.extend(inner);
                v.extend_from_slice(fall_args);
                v.extend_from_slice(bail_args);
                v
            }
            Terminator::Ret { value } => vec![*value],
            Terminator::Throw {
                value, resume_args, ..
            } => {
                let mut v = Vec::with_capacity(1 + resume_args.len());
                v.push(*value);
                v.extend_from_slice(resume_args);
                v
            }
            Terminator::Unreachable => vec![],
        }
    }

    /// Rewrite every use of any VReg in `rename` to the renamed VReg.
    /// Defs (e.g. fall-through-edge-only `InlineBranchOp::dst`) are left
    /// alone — only use positions are touched. Used by the lift-VRegs
    /// pass to redirect cross-block uses through a slot.
    pub fn rename_uses(&mut self, rename: &std::collections::HashMap<VReg, VReg>) {
        let apply = |v: &mut VReg, rename: &std::collections::HashMap<VReg, VReg>| {
            if let Some(&new) = rename.get(v) {
                *v = new;
            }
        };
        match self {
            Terminator::Jump { args, .. } => {
                for a in args.iter_mut() {
                    apply(a, rename);
                }
            }
            Terminator::Branch {
                lhs,
                rhs,
                t_args,
                f_args,
                ..
            } => {
                apply(lhs, rename);
                apply(rhs, rename);
                for a in t_args.iter_mut() {
                    apply(a, rename);
                }
                for a in f_args.iter_mut() {
                    apply(a, rename);
                }
            }
            Terminator::InlineBranch {
                op,
                fall_args,
                bail_args,
                ..
            } => {
                op.rename_uses(rename);
                for a in fall_args.iter_mut() {
                    apply(a, rename);
                }
                for a in bail_args.iter_mut() {
                    apply(a, rename);
                }
            }
            Terminator::Ret { value } => apply(value, rename),
            Terminator::Throw {
                value, resume_args, ..
            } => {
                apply(value, rename);
                for a in resume_args.iter_mut() {
                    apply(a, rename);
                }
            }
            Terminator::Unreachable => {}
        }
    }

    /// VRegs DEFINED by this terminator. Only `InlineBranch` defines a
    /// VReg from a terminator position; the result is live on the
    /// `fall_through` path only. The verifier treats this def as
    /// happening at `body.len() + 1` of the source block.
    pub fn defs(&self) -> Vec<VReg> {
        match self {
            Terminator::InlineBranch { op, .. } => vec![op.dst()],
            _ => vec![],
        }
    }

    /// Rename `dst` defs (only `InlineBranch` has any). Mirror of
    /// `rename_uses` for def positions.
    pub fn rename_defs(&mut self, rename: &std::collections::HashMap<VReg, VReg>) {
        if let Terminator::InlineBranch { op, .. } = self {
            op.rename_defs(rename);
        }
    }
}

impl Op {
    pub fn defs(&self) -> Vec<VReg> {
        match self {
            Op::SlotLoad { dst, .. } => vec![*dst],
            Op::SlotStore { .. } => vec![],

            Op::Move { dst, .. } => vec![*dst],
            Op::ConstTaggedInt { dst, .. } => vec![*dst],
            Op::ConstStringPtr { dst, .. } => vec![*dst],
            Op::ConstKeywordPtr { dst, .. } => vec![*dst],
            Op::ConstFunctionId { dst, .. } => vec![*dst],
            Op::ConstPointer { dst, .. } => vec![*dst],
            Op::ConstRawValue { dst, .. } => vec![*dst],
            Op::ConstTrue { dst } => vec![*dst],
            Op::ConstFalse { dst } => vec![*dst],
            Op::ConstNull { dst } => vec![*dst],
            Op::ConstLabelAddress { dst, .. } => vec![*dst],

            Op::AddInt { dst, .. } => vec![*dst],
            Op::Compare { dst, .. } => vec![*dst],
            Op::CompareFloat { dst, .. } => vec![*dst],

            Op::AddFloat { dst, .. } => vec![*dst],
            Op::SubFloat { dst, .. } => vec![*dst],
            Op::MulFloat { dst, .. } => vec![*dst],
            Op::DivFloat { dst, .. } => vec![*dst],

            Op::IntToFloat { dst, .. } => vec![*dst],
            Op::FRoundToZero { dst, .. } => vec![*dst],
            Op::FmovGpToFp { dst, .. } => vec![*dst],
            Op::FmovFpToGp { dst, .. } => vec![*dst],

            Op::Tag { dst, .. } => vec![*dst],
            Op::Untag { dst, .. } => vec![*dst],
            Op::GetTag { dst, .. } => vec![*dst],

            Op::And { dst, .. } => vec![*dst],
            Op::Or { dst, .. } => vec![*dst],
            Op::Xor { dst, .. } => vec![*dst],
            Op::AndImm { dst, .. } => vec![*dst],
            Op::ShiftRightImmRaw { dst, .. } => vec![*dst],

            Op::HeapLoad { dst, .. } => vec![*dst],
            Op::HeapLoadReg { dst, .. } => vec![*dst],
            Op::HeapLoadByteReg { dst, .. } => vec![*dst],
            Op::HeapStore { .. } => vec![],
            Op::HeapStoreOffset { .. } => vec![],
            Op::HeapStoreOffsetReg { .. } => vec![],
            Op::HeapStoreByteOffsetReg { .. } => vec![],
            Op::HeapStoreByteOffsetMasked { temp1, temp2, .. } => vec![*temp1, *temp2],

            Op::AtomicLoad { dst, .. } => vec![*dst],
            Op::AtomicStore { .. } => vec![],
            Op::CompareAndSwap { .. } => vec![],

            Op::StoreFloatConstant { temp, .. } => vec![*temp],

            Op::PushStack { .. } => vec![],
            Op::PopStack { dst } => vec![*dst],
            Op::GetStackPointer { dst, .. } => vec![*dst],
            Op::GetStackPointerImm { dst, .. } => vec![*dst],
            Op::GetFramePointer { dst } => vec![*dst],
            Op::CurrentStackPosition { dst } => vec![*dst],

            Op::ReadArgCount { dst } => vec![*dst],

            Op::Breakpoint => vec![],
            Op::ExtendLifetime { .. } => vec![],
            Op::RecordGcSafepoint => vec![],
            Op::FeedbackOr { .. } => vec![],
            Op::TierUpCheck { .. } => vec![],

            Op::Call { dst, .. } => vec![*dst],
            Op::Recurse { dst, .. } => vec![*dst],

            Op::PushExceptionHandler { .. } => vec![],
            Op::PushResumableExceptionHandler { dst, .. } => vec![*dst],
            Op::PopExceptionHandler { .. } => vec![],
            Op::PopExceptionHandlerById { .. } => vec![],

            Op::PushPromptHandler { .. } => vec![],
            Op::PopPromptHandler { .. } => vec![],
            Op::PushPromptTag { .. } => vec![],
            Op::CaptureContinuation { dst, .. } => vec![*dst],
            Op::CaptureContinuationTagged { dst, .. } => vec![*dst],

            Op::PerformEffect { .. } => vec![],
            Op::ReturnFromShift { .. } => vec![],
        }
    }

    /// Rewrite every DEF of any VReg in `rename` to its renamed value.
    /// Uses are NOT touched. Mirror of `rename_uses` for def positions.
    /// Used by the lift pass to give each multi-def site its own unique
    /// VReg name (so the resulting IR is true SSA — one def per VReg).
    pub fn rename_defs(&mut self, rename: &std::collections::HashMap<VReg, VReg>) {
        let apply = |v: &mut VReg| {
            if let Some(&new) = rename.get(v) {
                *v = new;
            }
        };
        match self {
            Op::SlotLoad { dst, .. } => apply(dst),
            Op::SlotStore { .. } => {}

            Op::Move { dst, .. } => apply(dst),
            Op::ConstTaggedInt { dst, .. } => apply(dst),
            Op::ConstStringPtr { dst, .. } => apply(dst),
            Op::ConstKeywordPtr { dst, .. } => apply(dst),
            Op::ConstFunctionId { dst, .. } => apply(dst),
            Op::ConstPointer { dst, .. } => apply(dst),
            Op::ConstRawValue { dst, .. } => apply(dst),
            Op::ConstTrue { dst } => apply(dst),
            Op::ConstFalse { dst } => apply(dst),
            Op::ConstNull { dst } => apply(dst),
            Op::ConstLabelAddress { dst, .. } => apply(dst),

            Op::AddInt { dst, .. }
            | Op::Compare { dst, .. }
            | Op::CompareFloat { dst, .. }
            | Op::AddFloat { dst, .. }
            | Op::SubFloat { dst, .. }
            | Op::MulFloat { dst, .. }
            | Op::DivFloat { dst, .. }
            | Op::IntToFloat { dst, .. }
            | Op::FRoundToZero { dst, .. }
            | Op::FmovGpToFp { dst, .. }
            | Op::FmovFpToGp { dst, .. }
            | Op::Tag { dst, .. }
            | Op::Untag { dst, .. }
            | Op::GetTag { dst, .. }
            | Op::And { dst, .. }
            | Op::Or { dst, .. }
            | Op::Xor { dst, .. }
            | Op::AndImm { dst, .. }
            | Op::ShiftRightImmRaw { dst, .. } => apply(dst),

            Op::HeapLoad { dst, .. } => apply(dst),
            Op::HeapLoadReg { dst, .. } => apply(dst),
            Op::HeapLoadByteReg { dst, .. } => apply(dst),
            Op::HeapStore { .. }
            | Op::HeapStoreOffset { .. }
            | Op::HeapStoreOffsetReg { .. }
            | Op::HeapStoreByteOffsetReg { .. } => {}
            Op::HeapStoreByteOffsetMasked { temp1, temp2, .. } => {
                apply(temp1);
                apply(temp2);
            }

            Op::AtomicLoad { dst, .. } => apply(dst),
            Op::AtomicStore { .. } | Op::CompareAndSwap { .. } => {}

            Op::StoreFloatConstant { temp, .. } => apply(temp),

            Op::PushStack { .. } => {}
            Op::PopStack { dst } => apply(dst),
            Op::GetStackPointer { dst, .. } => apply(dst),
            Op::GetStackPointerImm { dst, .. } => apply(dst),
            Op::GetFramePointer { dst } => apply(dst),
            Op::CurrentStackPosition { dst } => apply(dst),

            Op::ReadArgCount { dst } => apply(dst),

            Op::Breakpoint
            | Op::ExtendLifetime { .. }
            | Op::RecordGcSafepoint
            | Op::FeedbackOr { .. }
            | Op::TierUpCheck { .. } => {}

            Op::Call { dst, .. } => apply(dst),
            Op::Recurse { dst, .. } => apply(dst),

            Op::PushExceptionHandler { .. } => {}
            Op::PushResumableExceptionHandler { dst, .. } => apply(dst),
            Op::PopExceptionHandler { .. } | Op::PopExceptionHandlerById { .. } => {}

            Op::PushPromptHandler { .. }
            | Op::PopPromptHandler { .. }
            | Op::PushPromptTag { .. } => {}
            Op::CaptureContinuation { dst, .. } => apply(dst),
            Op::CaptureContinuationTagged { dst, .. } => apply(dst),

            Op::PerformEffect { .. } | Op::ReturnFromShift { .. } => {}
        }
    }

    pub fn uses(&self) -> Vec<VReg> {
        match self {
            Op::SlotLoad { .. } => vec![],
            Op::SlotStore { src, .. } => vec![*src],

            Op::Move { src, .. } => vec![*src],
            Op::ConstTaggedInt { .. }
            | Op::ConstStringPtr { .. }
            | Op::ConstKeywordPtr { .. }
            | Op::ConstFunctionId { .. }
            | Op::ConstPointer { .. }
            | Op::ConstRawValue { .. }
            | Op::ConstTrue { .. }
            | Op::ConstFalse { .. }
            | Op::ConstNull { .. }
            | Op::ConstLabelAddress { .. } => vec![],

            Op::AddInt { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::Compare { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::CompareFloat { lhs, rhs, .. } => vec![*lhs, *rhs],

            Op::AddFloat { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::SubFloat { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::MulFloat { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::DivFloat { lhs, rhs, .. } => vec![*lhs, *rhs],

            Op::IntToFloat { src, .. } => vec![*src],
            Op::FRoundToZero { src, .. } => vec![*src],
            Op::FmovGpToFp { src, .. } => vec![*src],
            Op::FmovFpToGp { src, .. } => vec![*src],

            Op::Tag {
                src, tag_source, ..
            } => match tag_source {
                TagSource::Register(t) => vec![*src, *t],
                TagSource::Bits(_) => vec![*src],
            },
            Op::Untag { src, .. } => vec![*src],
            Op::GetTag { src, .. } => vec![*src],

            Op::And { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::Or { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::Xor { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::AndImm { src, .. } => vec![*src],
            Op::ShiftRightImmRaw { src, .. } => vec![*src],

            Op::HeapLoad { base, .. } => vec![*base],
            Op::HeapLoadReg { base, offset, .. } => vec![*base, *offset],
            Op::HeapLoadByteReg { base, offset, .. } => vec![*base, *offset],
            Op::HeapStore { addr, src } => vec![*addr, *src],
            Op::HeapStoreOffset { base, src, .. } => vec![*base, *src],
            Op::HeapStoreOffsetReg { base, src, offset } => vec![*base, *src, *offset],
            Op::HeapStoreByteOffsetReg { base, src, offset } => vec![*base, *src, *offset],
            Op::HeapStoreByteOffsetMasked { ptr, val, .. } => vec![*ptr, *val],

            Op::AtomicLoad { src, .. } => vec![*src],
            Op::AtomicStore { addr, src } => vec![*addr, *src],
            Op::CompareAndSwap {
                addr,
                expected,
                new,
            } => vec![*addr, *expected, *new],

            Op::StoreFloatConstant { dest, .. } => vec![*dest],

            Op::PushStack { src } => vec![*src],
            Op::PopStack { .. } => vec![],
            Op::GetStackPointer { offset, .. } => vec![*offset],
            Op::GetStackPointerImm { .. } => vec![],
            Op::GetFramePointer { .. } => vec![],
            Op::CurrentStackPosition { .. } => vec![],

            Op::ReadArgCount { .. } => vec![],

            Op::Breakpoint => vec![],
            Op::ExtendLifetime { src } => vec![*src],
            Op::RecordGcSafepoint => vec![],
            Op::FeedbackOr { .. } => vec![],
            Op::TierUpCheck { .. } => vec![],

            Op::Call { target, args, .. } => {
                let mut v = Vec::with_capacity(1 + args.len());
                if let CallTarget::Register(fp) = target {
                    v.push(*fp);
                }
                v.extend_from_slice(args);
                v
            }
            Op::Recurse { args, .. } => args.clone(),

            Op::PushExceptionHandler { .. } => vec![],
            Op::PushResumableExceptionHandler { .. } => vec![],
            Op::PopExceptionHandler { .. } => vec![],
            Op::PopExceptionHandlerById { handler_id, .. } => vec![*handler_id],

            Op::PushPromptHandler { .. } => vec![],
            Op::PopPromptHandler { result, .. } => vec![*result],
            Op::PushPromptTag { tag, .. } => vec![*tag],
            Op::CaptureContinuation { .. } => vec![],
            Op::CaptureContinuationTagged { tag, .. } => vec![*tag],

            Op::PerformEffect {
                handler,
                enum_type,
                op_value,
                ..
            } => vec![*handler, *enum_type, *op_value],
            Op::ReturnFromShift {
                value, cont_ptr, ..
            } => vec![*value, *cont_ptr],
        }
    }

    /// Rewrite every use of any VReg listed in `rename` to its renamed
    /// counterpart. Defs are NOT touched — this is the inverse rewriting
    /// the lift-VRegs pass needs to redirect uses through a slot without
    /// disturbing the original def.
    pub fn rename_uses(&mut self, rename: &std::collections::HashMap<VReg, VReg>) {
        let apply = |v: &mut VReg| {
            if let Some(&new) = rename.get(v) {
                *v = new;
            }
        };
        match self {
            Op::SlotLoad { .. } => {}
            Op::SlotStore { src, .. } => apply(src),

            Op::Move { src, .. } => apply(src),
            Op::ConstTaggedInt { .. }
            | Op::ConstStringPtr { .. }
            | Op::ConstKeywordPtr { .. }
            | Op::ConstFunctionId { .. }
            | Op::ConstPointer { .. }
            | Op::ConstRawValue { .. }
            | Op::ConstTrue { .. }
            | Op::ConstFalse { .. }
            | Op::ConstNull { .. }
            | Op::ConstLabelAddress { .. } => {}

            Op::AddInt { lhs, rhs, .. }
            | Op::Compare { lhs, rhs, .. }
            | Op::CompareFloat { lhs, rhs, .. }
            | Op::AddFloat { lhs, rhs, .. }
            | Op::SubFloat { lhs, rhs, .. }
            | Op::MulFloat { lhs, rhs, .. }
            | Op::DivFloat { lhs, rhs, .. }
            | Op::And { lhs, rhs, .. }
            | Op::Or { lhs, rhs, .. }
            | Op::Xor { lhs, rhs, .. } => {
                apply(lhs);
                apply(rhs);
            }

            Op::IntToFloat { src, .. }
            | Op::FRoundToZero { src, .. }
            | Op::FmovGpToFp { src, .. }
            | Op::FmovFpToGp { src, .. }
            | Op::Untag { src, .. }
            | Op::GetTag { src, .. }
            | Op::AndImm { src, .. }
            | Op::ShiftRightImmRaw { src, .. } => apply(src),

            Op::Tag {
                src, tag_source, ..
            } => {
                apply(src);
                if let TagSource::Register(t) = tag_source {
                    apply(t);
                }
            }

            Op::HeapLoad { base, .. } => apply(base),
            Op::HeapLoadReg { base, offset, .. } | Op::HeapLoadByteReg { base, offset, .. } => {
                apply(base);
                apply(offset);
            }
            Op::HeapStore { addr, src } => {
                apply(addr);
                apply(src);
            }
            Op::HeapStoreOffset { base, src, .. } => {
                apply(base);
                apply(src);
            }
            Op::HeapStoreOffsetReg { base, src, offset }
            | Op::HeapStoreByteOffsetReg { base, src, offset } => {
                apply(base);
                apply(src);
                apply(offset);
            }
            Op::HeapStoreByteOffsetMasked { ptr, val, .. } => {
                apply(ptr);
                apply(val);
            }

            Op::AtomicLoad { src, .. } => apply(src),
            Op::AtomicStore { addr, src } => {
                apply(addr);
                apply(src);
            }
            Op::CompareAndSwap {
                addr,
                expected,
                new,
            } => {
                apply(addr);
                apply(expected);
                apply(new);
            }

            Op::StoreFloatConstant { dest, .. } => apply(dest),

            Op::PushStack { src } => apply(src),
            Op::PopStack { .. } => {}
            Op::GetStackPointer { offset, .. } => apply(offset),
            Op::GetStackPointerImm { .. }
            | Op::GetFramePointer { .. }
            | Op::CurrentStackPosition { .. }
            | Op::ReadArgCount { .. } => {}

            Op::Breakpoint
            | Op::RecordGcSafepoint
            | Op::FeedbackOr { .. }
            | Op::TierUpCheck { .. } => {}
            Op::ExtendLifetime { src } => apply(src),

            Op::Call { target, args, .. } => {
                if let CallTarget::Register(fp) = target {
                    apply(fp);
                }
                for a in args.iter_mut() {
                    apply(a);
                }
            }
            Op::Recurse { args, .. } => {
                for a in args.iter_mut() {
                    apply(a);
                }
            }

            Op::PushExceptionHandler { .. }
            | Op::PushResumableExceptionHandler { .. }
            | Op::PopExceptionHandler { .. }
            | Op::PushPromptHandler { .. }
            | Op::CaptureContinuation { .. } => {}
            Op::PopExceptionHandlerById { handler_id, .. } => apply(handler_id),
            Op::PopPromptHandler { result, .. } => apply(result),
            Op::PushPromptTag { tag, .. } => apply(tag),
            Op::CaptureContinuationTagged { tag, .. } => apply(tag),

            Op::PerformEffect {
                handler,
                enum_type,
                op_value,
                ..
            } => {
                apply(handler);
                apply(enum_type);
                apply(op_value);
            }
            Op::ReturnFromShift {
                value, cont_ptr, ..
            } => {
                apply(value);
                apply(cont_ptr);
            }
        }
    }
}
