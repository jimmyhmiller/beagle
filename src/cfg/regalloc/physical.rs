//! Phase 4f-2c — remap greedy chordal colors to backend physical
//! register indices.
//!
//! Phase 4c (`color.rs`) assigns dense colors `{0, 1, 2, ...}` per
//! `RegClass`. Those colors are abstract — the legacy backend's
//! `register_from_index(n)` expects `n` to be a physical register
//! index (e.g. X19=19 on ARM64, R12=12 on x86-64). Two pieces are
//! also needed beyond a naive remap:
//!
//! 1. **Entry-block params hold function arguments** on entry,
//!    placed by the calling convention at `arg_regs[0..N]` (X0..X7 on
//!    ARM64). Their physical assignment must match the arg position.
//! 2. **Non-arg VRegs must come from the allocator pool**
//!    (X19..X27 on ARM64) so they survive across cross-Beagle calls
//!    by AAPCS callee-save discipline.
//!
//! The greedy assignment in `color.rs` already gives entry params
//! the colors 0..N-1 in arg order (they all interfere at block
//! entry and are visited first by the dom-tree DFS). So:
//!
//! - For entry params: physical = `arg_regs[param_position]`.
//! - For everything else: physical = `allocator_pool[color]`. Two
//!   non-interfering non-arg VRegs sharing color `c` remap to the
//!   same `allocator_pool[c]` physical register, which is correct
//!   by construction (they don't interfere).
//!
//! Caveats this commit does **not** address:
//!
//! - **Cross-call clobber.** A VReg live across `Op::Call` colored
//!   into an arg register (e.g. an entry-param-X0 surviving across
//!   a call) will be wiped. Honoring `ClobberSet::AllCallerSaved`
//!   requires per-call interference edges (live-set ∩ caller-saved
//!   regs); that lands in a follow-up.
//! - **Pool overflow.** If the greedy color exceeds the allocator
//!   pool size, this returns a sentinel (`PHYSICAL_OVERFLOW`). The
//!   driver should detect this and either spill (Phase 4d's
//!   iterative re-color) or refuse the function and fall back to
//!   the legacy pipeline.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::cfg::regalloc::color::Coloring;
use crate::cfg::{CfgFunction, RegClass, VReg};

/// Returned in place of a physical index when the greedy color
/// exceeds the allocator pool size. The driver must check for this
/// and either invoke spilling or bail.
pub const PHYSICAL_OVERFLOW: u32 = u32::MAX;

/// ABI register pools for one target architecture.
pub struct PhysicalLayout {
    /// Arg-register physical indices, in arg-position order. Entry
    /// block params map 1:1 to these.
    pub arg_regs_gp: &'static [u32],
    pub arg_regs_fp: &'static [u32],
    /// Allocator pool — physical indices available for non-arg VRegs.
    pub allocator_gp: &'static [u32],
    pub allocator_fp: &'static [u32],
}

#[cfg(not(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
)))]
pub static ARM64_LAYOUT: PhysicalLayout = PhysicalLayout {
    // X0..X7
    arg_regs_gp: &[0, 1, 2, 3, 4, 5, 6, 7],
    // D0..D7
    arg_regs_fp: &[0, 1, 2, 3, 4, 5, 6, 7],
    // X19..X27 (allocator pool per `abi::arm64::ABI`).
    allocator_gp: &[19, 20, 21, 22, 23, 24, 25, 26, 27],
    // D8..D15 (callee-saved low-64 of V8..V15). Legacy doesn't have
    // a distinct FP pool; this matches the de-facto "ARM FP is safe
    // here" usage. If a function actually needs FP across calls,
    // spilling kicks in.
    allocator_fp: &[8, 9, 10, 11, 12, 13, 14, 15],
};

#[cfg(any(
    feature = "backend-x86-64",
    all(target_arch = "x86_64", not(feature = "backend-arm64"))
))]
pub static X86_64_LAYOUT: PhysicalLayout = PhysicalLayout {
    // RDI=7, RSI=6, RDX=2, RCX=1, R8=8, R9=9 — System V arg regs
    arg_regs_gp: &[7, 6, 2, 1, 8, 9],
    // XMM0..XMM7 caller-saved arg regs
    arg_regs_fp: &[0, 1, 2, 3, 4, 5, 6, 7],
    // R12=12, R13=13, R14=14, R15=15, RBX virtual=16 — legacy pool
    allocator_gp: &[12, 13, 14, 15, 16],
    // No XMM callee-saved on SysV; allocator must spill FP across calls
    allocator_fp: &[],
};

/// Get the layout for the current build target.
pub fn current_layout() -> &'static PhysicalLayout {
    #[cfg(not(any(
        feature = "backend-x86-64",
        all(target_arch = "x86_64", not(feature = "backend-arm64"))
    )))]
    {
        &ARM64_LAYOUT
    }
    #[cfg(any(
        feature = "backend-x86-64",
        all(target_arch = "x86_64", not(feature = "backend-arm64"))
    ))]
    {
        &X86_64_LAYOUT
    }
}

/// Map every colored VReg to its physical register index. Entry
/// block params get `arg_regs[position]`; all other VRegs get
/// `allocator_pool[color]`.
///
/// Returns a new `Coloring` where `colors[v]` is a physical index
/// (or [`PHYSICAL_OVERFLOW`] if the greedy color exceeded the
/// allocator pool size). The `max_color_used` field is repurposed
/// to track the maximum physical index used per class.
pub fn assign_physical_registers(
    cfg: &CfgFunction,
    coloring: &Coloring,
    layout: &PhysicalLayout,
) -> Coloring {
    let mut physical: HashMap<VReg, u32> = HashMap::new();
    let mut max_used: HashMap<RegClass, u32> = HashMap::new();

    // 1. Entry block params → arg regs by position.
    let entry_params: HashMap<VReg, usize> = cfg
        .block(cfg.entry)
        .params
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();

    for (&vreg, &color) in &coloring.colors {
        let phys = if let Some(&pos) = entry_params.get(&vreg) {
            let pool = match vreg.class {
                RegClass::Gp => layout.arg_regs_gp,
                RegClass::Fp => layout.arg_regs_fp,
            };
            if pos < pool.len() {
                pool[pos]
            } else {
                // More entry params than arg regs — caller passes
                // overflow on the stack. The MVP doesn't handle this;
                // mark as overflow so the driver can bail.
                PHYSICAL_OVERFLOW
            }
        } else {
            let pool = match vreg.class {
                RegClass::Gp => layout.allocator_gp,
                RegClass::Fp => layout.allocator_fp,
            };
            if (color as usize) < pool.len() {
                pool[color as usize]
            } else {
                PHYSICAL_OVERFLOW
            }
        };
        physical.insert(vreg, phys);
        if phys != PHYSICAL_OVERFLOW {
            let cur = max_used.get(&vreg.class).copied().unwrap_or(0);
            if phys > cur {
                max_used.insert(vreg.class, phys);
            }
        }
    }

    Coloring {
        colors: physical,
        max_color_used: max_used,
    }
}

/// True if any VReg in `coloring` got `PHYSICAL_OVERFLOW` —
/// i.e., the function needs spilling or doesn't fit the layout.
pub fn has_overflow(coloring: &Coloring) -> bool {
    coloring.colors.values().any(|&c| c == PHYSICAL_OVERFLOW)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::regalloc::color::color;
    use crate::cfg::regalloc::interference::build_interference;
    use crate::cfg::regalloc::liveness::compute_liveness;
    use crate::cfg::{CfgFunction, Op, Terminator};

    fn test_layout() -> PhysicalLayout {
        PhysicalLayout {
            arg_regs_gp: &[0, 1, 2, 3, 4, 5, 6, 7],
            arg_regs_fp: &[0, 1, 2, 3, 4, 5, 6, 7],
            allocator_gp: &[19, 20, 21, 22, 23, 24, 25, 26, 27],
            allocator_fp: &[8, 9, 10, 11, 12, 13, 14, 15],
        }
    }

    /// Identity function with a single arg → physical X0.
    #[test]
    fn entry_param_maps_to_arg_reg() {
        let mut f = CfgFunction::new(Some("id".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let x = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(x);
        f.block_mut(entry).terminator = Terminator::Ret { value: x };

        let liveness = compute_liveness(&f);
        let ig = build_interference(&f, &liveness);
        let coloring = color(&f, &ig);
        let layout = test_layout();
        let phys = assign_physical_registers(&f, &coloring, &layout);

        assert_eq!(phys.colors[&x], 0, "x is arg(0) → X0 (index 0)");
        assert!(!has_overflow(&phys));
    }

    /// Two entry params + a body VReg: args at X0,X1; body at X19.
    #[test]
    fn body_vreg_maps_to_allocator_pool() {
        let mut f = CfgFunction::new(Some("add".into()), 0);
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
        let layout = test_layout();
        let phys = assign_physical_registers(&f, &coloring, &layout);

        assert_eq!(phys.colors[&a], 0, "a is arg(0) → X0");
        assert_eq!(phys.colors[&b], 1, "b is arg(1) → X1");
        // r is non-arg; greedy gave it color 0 (a, b dead after AddInt).
        // Remap: allocator_gp[0] = 19.
        assert_eq!(phys.colors[&r], 19, "r is body VReg → X19");
        assert!(!has_overflow(&phys));
    }

    /// More entry params than arg regs marks overflow.
    #[test]
    fn too_many_entry_params_overflows() {
        // Make a layout with 1 arg reg, then function with 2 entry
        // params — second one overflows.
        let layout = PhysicalLayout {
            arg_regs_gp: &[0],
            arg_regs_fp: &[],
            allocator_gp: &[19, 20],
            allocator_fp: &[],
        };
        let mut f = CfgFunction::new(Some("over".into()), 0);
        let entry = f.new_block();
        f.entry = entry;
        let a = f.new_vreg(RegClass::Gp);
        let b = f.new_vreg(RegClass::Gp);
        f.block_mut(entry).params.push(a);
        f.block_mut(entry).params.push(b);
        f.block_mut(entry).terminator = Terminator::Ret { value: a };

        let liveness = compute_liveness(&f);
        let ig = build_interference(&f, &liveness);
        let coloring = color(&f, &ig);
        let phys = assign_physical_registers(&f, &coloring, &layout);

        assert_eq!(phys.colors[&a], 0, "first param fits");
        assert_eq!(phys.colors[&b], PHYSICAL_OVERFLOW, "second param overflows");
        assert!(has_overflow(&phys));
    }
}
