//! Low-level IR (Lir) — ABI-aware codegen primitives.
//!
//! Lir sits between the user-facing `ir::Instruction` and the per-backend
//! machine-code emission. Each `LirOp` variant is a *named primitive* with
//! a single definition and per-backend lowering rule. Primitives whose
//! instruction-sequence shape depends on the Beagle ABI — MutatorState
//! field access, Rust runtime calls, inline allocation, write barriers —
//! belong here rather than being spelled out ad-hoc at each call site.
//!
//! # Why a separate layer
//!
//! Before Lir, these primitives were open-coded across arm.rs and main.rs.
//! The `gc_frame_top` offset (`[x28, #16]`) appeared as the magic number
//! `imm12: 2` in three places; `jit_load_current_mutator_state` was called
//! via hand-rolled `save-call-restore` sequences in four trampolines; and
//! adding a new primitive like inline bump allocation would have meant
//! touching both backends with bespoke code. Lir collapses that to: one
//! enum variant, one lowering rule per backend.
//!
//! # Extension pattern
//!
//! Adding a new ABI primitive is:
//! 1. Add a variant to `LirOp<R>`.
//! 2. Add a lowering arm to each backend's `lower_lir` implementation.
//! 3. Emit the op from wherever produces the primitive (e.g., a Beagle
//!    function prologue, or a JIT-side allocation site).
//!
//! The convention for "documented but not yet lowered" variants is to
//! leave the ARM lowering as `todo!("<primitive>: <why waiting>")` — this
//! way the variant name, its parameters, and the constraint on its
//! introduction are all expressed in code rather than in a separate plan.

use crate::runtime::{
    MUTATOR_STATE_ALLOC_END_OFFSET, MUTATOR_STATE_ALLOC_PTR_OFFSET,
    MUTATOR_STATE_GC_FRAME_TOP_OFFSET, MUTATOR_STATE_SAVED_FP_OFFSET,
    MUTATOR_STATE_SAVED_RET_OFFSET, MUTATOR_STATE_SAVED_SP_OFFSET,
};

/// Named fields on `MutatorState` accessible from JIT'd code via the
/// reserved mutator-state register. Offsets come from `src/runtime.rs`
/// constants; changing the `MutatorState` layout there is the single
/// edit that updates all JIT sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutatorField {
    /// TLAB bump pointer. JIT allocation fast path: bump this, compare
    /// against `AllocEnd`, slow-path if overflow.
    AllocPtr,
    /// One past the last usable byte in the current TLAB. When
    /// `AllocPtr == AllocEnd`, the thread is disarmed and must take the
    /// slow path (used to force a safepoint before GC).
    AllocEnd,
    /// Top of the GC frame linked list. Inlined `gc_frame_link` writes
    /// this on every Beagle function entry.
    GcFrameTop,
    /// Beagle frame pointer saved at the last Rust-callable builtin
    /// entry. Populated by `save_gc_context!`.
    SavedFramePointer,
    /// Beagle stack pointer saved at the last Rust-callable builtin
    /// entry.
    SavedStackPointer,
    /// Return address saved at the last Rust-callable builtin entry.
    /// Used by GC safepoint walking and by `throw_runtime_error` unwind.
    SavedGcReturnAddr,
}

impl MutatorField {
    /// Byte offset of this field within `MutatorState`.
    pub const fn byte_offset(self) -> i32 {
        match self {
            MutatorField::AllocPtr => MUTATOR_STATE_ALLOC_PTR_OFFSET,
            MutatorField::AllocEnd => MUTATOR_STATE_ALLOC_END_OFFSET,
            MutatorField::GcFrameTop => MUTATOR_STATE_GC_FRAME_TOP_OFFSET,
            MutatorField::SavedFramePointer => MUTATOR_STATE_SAVED_FP_OFFSET,
            MutatorField::SavedStackPointer => MUTATOR_STATE_SAVED_SP_OFFSET,
            MutatorField::SavedGcReturnAddr => MUTATOR_STATE_SAVED_RET_OFFSET,
        }
    }
}

/// A Rust function exposed to JIT'd code as a stable `extern "C"`
/// symbol. Values on this enum are what `LirOp::CallRuntime` targets;
/// each has a corresponding `#[unsafe(no_mangle)] #[inline(never)]`
/// Rust definition whose address the backend resolves at codegen time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeSym {
    /// Returns the current thread's `*mut MutatorState`. The backend
    /// emits code that calls this and copies the result into the
    /// reserved mutator-state register.
    LoadMutatorState,
}

impl RuntimeSym {
    /// The concrete function pointer the JIT should `blr` to.
    pub fn addr(self) -> usize {
        match self {
            RuntimeSym::LoadMutatorState => crate::runtime::jit_load_current_mutator_state as usize,
        }
    }
}

/// An ABI-aware codegen primitive. Generic over the backend's register
/// type so ARM and x86-64 can share the enum shape without leaking
/// architecture-specific register types into the Lir.
#[derive(Clone, Debug)]
pub enum LirOp<R: 'static> {
    /// `dst = MutatorState.<field>`. Lowers to a single load through the
    /// reserved mutator-state register (ARM: `ldr dst, [x28, #offset]`).
    LoadMutatorField { field: MutatorField, dst: R },

    /// `MutatorState.<field> = src`. Lowers to a single store (ARM:
    /// `str src, [x28, #offset]`).
    StoreMutatorField { field: MutatorField, src: R },

    /// Call a Rust runtime function from JIT'd code, preserving the
    /// registers in `preserve` across the call. On return the reserved
    /// mutator-state register holds the result (only applicable when
    /// `target` is `RuntimeSym::LoadMutatorState` — future call targets
    /// will need a more general protocol).
    ///
    /// Used at every Rust→Beagle boundary (main trampoline,
    /// `save_volatile_registers_N`, `apply_call_N`, `return-jump`) to
    /// guarantee a valid mutator-state pointer regardless of what Rust
    /// left in the reserved register.
    ///
    /// `preserve` is owned (rather than a borrowed slice) so this op can
    /// be constructed from a pattern match with arity-varying arms
    /// without the match fighting Rust's lifetime rules.
    CallRuntime {
        target: RuntimeSym,
        preserve: Vec<R>,
    },
    // ----- Future perf primitives --------------------------------------
    //
    // The following variants are listed but not yet lowered. Adding them
    // is:
    //   1. Uncomment / add the variant.
    //   2. Add a lowering arm in each backend's `lower_lir`.
    //   3. Emit the op from the allocation / store / load site.
    //
    // They are commented rather than made `todo!()` panics so that the
    // enum isn't artificially exhaustive before there's an implementation
    // plan — see docs/beagle-abi-cleanup.md for the sequencing.
    //
    // InlineBumpAllocate {
    //     size: usize,
    //     dst: R,
    //     slow_path: crate::common::Label,
    // },
    // InlineWriteBarrier {
    //     obj: R,
    //     field_offset: u32,
    //     value: R,
    // },
    // InlineReadBarrier {
    //     dst: R,
    //     src: R,
    //     offset: i32,
    // },
}
