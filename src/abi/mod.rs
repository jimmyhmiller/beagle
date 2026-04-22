//! Single source of truth for the Beagle calling convention.
//!
//! Historically the ABI was encoded implicitly across four places:
//! `canonical_volatile_registers` in `arm.rs`, two separate
//! `(19..=27)` ranges in the register allocators, and a hardcoded
//! bitmask test. A single change to any ABI-relevant decision (like
//! "reserve x28 for MutatorState") meant touching all four in sync,
//! with the one you forgot surfacing as a SIGBUS at runtime.
//!
//! `BeagleAbi` consolidates those roles into one per-backend constant
//! that the rest of the compiler reads through the `CodegenBackend`
//! trait. Adding a new role is one field; changing an existing role
//! is one edit.

/// Register roles for a target architecture. Generic over the backend's
/// physical register type so the ARM and x86-64 backends can share the
/// same shape without leaking architecture-specific types upward.
pub struct BeagleAbi<R: 'static> {
    /// Holds the current thread's `MutatorState` pointer. Reserved: not
    /// in the allocator pool. Must be preserved across every call into
    /// Beagle; every Rust→Beagle boundary reloads it via
    /// `jit_load_current_mutator_state`.
    pub mutator_state_reg: R,

    /// Holds the argument count for variadic / apply-style calls.
    pub arg_count_reg: R,

    /// Registers the register allocator may hand out. Callee-saved under
    /// AAPCS / SysV ABI, minus any reserved registers.
    pub allocator_pool: &'static [R],

    /// Full callee-saved set that shim trampolines save/restore across a
    /// Rust→Beagle call, regardless of whether the register allocator
    /// uses it. Includes `mutator_state_reg` so shims can preserve the
    /// caller's value for AAPCS compliance. Length is used for
    /// stack-alignment math, so this set must stay an even count.
    pub callee_saved: &'static [R],

    /// Argument registers in order (X0-X7 on ARM, RDI/RSI/RDX/RCX/R8/R9
    /// on x86-64).
    pub arg_regs: &'static [R],
}

#[cfg(not(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
)))]
pub mod arm64 {
    use super::BeagleAbi;
    use crate::machine_code::arm_codegen::{
        Register, X0, X1, X2, X3, X4, X5, X6, X7, X9, X19, X20, X21, X22, X23, X24, X25, X26, X27,
        X28,
    };

    /// AAPCS-compatible register plan with x28 carved out for the
    /// per-thread `MutatorState*`.
    pub static ABI: BeagleAbi<Register> = BeagleAbi {
        mutator_state_reg: X28,
        arg_count_reg: X9,
        allocator_pool: &[X19, X20, X21, X22, X23, X24, X25, X26, X27],
        callee_saved: &[X19, X20, X21, X22, X23, X24, X25, X26, X27, X28],
        arg_regs: &[X0, X1, X2, X3, X4, X5, X6, X7],
    };
}

#[cfg(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
))]
pub mod x86_64 {
    use super::BeagleAbi;
    use crate::machine_code::x86_codegen::{
        R8, R9, R10, R12, R13, R14, R15, RBX, RCX, RDI, RDX, RSI, X86Register,
    };

    /// x86-64 System V register plan. Callee-saved set here does not
    /// include a reserved MutatorState register yet — x86-64 isn't part
    /// of the x28-reserved ABI refactor and still uses the thread-local
    /// slow path. `mutator_state_reg` is placeholder R15; the
    /// Rust→Beagle boundary on x86-64 doesn't address it directly.
    pub static ABI: BeagleAbi<X86Register> = BeagleAbi {
        mutator_state_reg: R15,
        arg_count_reg: R10,
        allocator_pool: &[R12, R13, R14, R15, RBX],
        callee_saved: &[R12, R13, R14, R15, RBX],
        arg_regs: &[RDI, RSI, RDX, RCX, R8, R9],
    };
}
