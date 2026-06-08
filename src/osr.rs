//! On-stack replacement (OSR) support.
//!
//! OSR lets a long-running loop in tier-1 (unoptimized) code transfer into
//! an optimized continuation mid-execution, instead of waiting for the
//! whole function to be re-entered. See `docs/OSR_DESIGN.md`.
//!
//! The transfer is a **normal call**: at a hot loop back-edge, tier-1 `F`
//! calls an optimized continuation `F_osr(buf)` — the rest of `F` starting at
//! the loop header — passing a pointer to a buffer of the loop-carried live-in
//! values, and returns that call's result.
//!
//! `F_osr` is produced by **OSR-deconstruction** of F's already-specialized
//! IR (`build_osr_variant_ir`): strip the OSR checks, then prepend an entry
//! that loads each live-in from the buffer into its local slot and jumps to
//! the loop header. The original pre-loop code (prologue, arg-loads, pre-loop
//! computation) becomes unreachable from the new entry and is dead-code
//! eliminated when the CFG is built (reachability from instruction 0). The
//! post-loop continuation is reached naturally through the loop's exit, so
//! `F_osr` returns F's result.

use crate::common::Label;
use crate::ir::{Instruction, Ir, Value};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

/// Runtime state of an OSR continuation, keyed by `"<fn>#L<idx>"`. Shared
/// between the mutator (the back-edge trampoline) and the compiler thread
/// (which builds the variant), so it lives behind a global mutex.
#[derive(Debug, Clone)]
pub enum OsrState {
    /// A build has been requested and is in flight on the compiler thread.
    Requested,
    /// The continuation is compiled and callable at `code_addr`; `live_in_slots`
    /// are the frame slots to read out of F's running frame, in argument order.
    Ready {
        code_addr: usize,
        live_in_slots: Vec<usize>,
    },
    /// The loop is ineligible or the build failed — never retry.
    Failed,
}

fn registry() -> &'static Mutex<HashMap<String, OsrState>> {
    static R: OnceLock<Mutex<HashMap<String, OsrState>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn osr_lookup(key: &str) -> Option<OsrState> {
    registry().lock().ok()?.get(key).cloned()
}

/// Reserve `key` for building. Returns true iff it was previously absent (so
/// the caller should send exactly one build request).
pub fn osr_try_reserve(key: &str) -> bool {
    let mut g = registry().lock().unwrap();
    if g.contains_key(key) {
        return false;
    }
    g.insert(key.to_string(), OsrState::Requested);
    true
}

pub fn osr_set_ready(key: &str, code_addr: usize, live_in_slots: Vec<usize>) {
    registry().lock().unwrap().insert(
        key.to_string(),
        OsrState::Ready {
            code_addr,
            live_in_slots,
        },
    );
}

pub fn osr_set_failed(key: &str) {
    registry()
        .lock()
        .unwrap()
        .insert(key.to_string(), OsrState::Failed);
}

thread_local! {
    /// Keys whose `F_osr` is currently executing on this thread. A deopt inside
    /// `F_osr` re-invokes generic tier-1 F, which runs the *same* loop and would
    /// hit its OSR back-edge again — re-entering the trampoline for the same key
    /// and nesting another transfer, unbounded, if a guard keeps failing. While
    /// a key is in this set, the trampoline declines to transfer so the
    /// re-invoked generic F runs to completion instead.
    static IN_TRANSFER: std::cell::RefCell<HashSet<String>> =
        std::cell::RefCell::new(HashSet::new());
}

/// Mark `key` as transferring on this thread. Returns `true` if it was not
/// already transferring (caller may proceed); `false` if a transfer for this
/// key is already on the stack (caller must decline to avoid re-entrant nesting).
pub fn osr_transfer_begin(key: &str) -> bool {
    IN_TRANSFER.with(|s| s.borrow_mut().insert(key.to_string()))
}

pub fn osr_transfer_end(key: &str) {
    IN_TRANSFER.with(|s| {
        s.borrow_mut().remove(key);
    });
}

/// Per-loop information captured during compilation, indexed by source
/// order within the function (the "loop index" baked into the OSR check's
/// key string `"<fn>#L<idx>"`).
#[derive(Debug, Clone)]
pub struct OsrLoopInfo {
    /// The loop header label (back-edge target) in the function's IR.
    pub header_label: Label,
    /// Local-slot indices holding the loop's live-in state — every named
    /// local in scope at the latch. The trampoline reads these slots out of
    /// F's running frame (in this order) into a buffer; the OSR entry loads
    /// `buf[i]` back into slot `live_in_slots[i]`. Identical slot layout
    /// across tiers (verified) makes this sound.
    pub live_in_slots: Vec<usize>,
}

/// Build `F_osr`'s IR from F's already-specialized IR and one loop's info.
///
/// Precondition: `base` is the *specialized* (feedback-driven, tier-2) IR —
/// cloning tier-1 IR would yield unspecialized, slow code (see
/// `docs/OSR_DESIGN.md` §2d).
pub fn build_osr_variant_ir(base: &Ir, info: &OsrLoopInfo, int_slots: &HashSet<usize>) -> Ir {
    let mut ir = base.clone();
    // `F_osr` is the optimized target, not an OSR source — drop any OSR
    // checks the clone carried (tier-2 IR has none, but be defensive).
    ir.instructions
        .retain(|i| !matches!(i, Instruction::OsrCheck(..)));

    // Build the OSR entry. `F_osr` takes a SINGLE argument: a pointer to a
    // buffer of the loop's live-in values, packed by the trampoline (in
    // `live_in_slots` order). Load each `buf[i]` into its slot, then jump to
    // the loop header; the original pre-loop code becomes unreachable and is
    // DCE'd by CFG build. (Note: enclosing-loop headers sit *before* the OSR'd
    // header but stay reachable via their back-edges, so DON'T truncate.) The
    // unpack runs before any safepoint, so the not-yet-rooted buffer can't be
    // invalidated by GC mid-unpack.
    //
    // The buffer pointer is arg 0. Reuse the function's CANONICAL arg-0 vreg
    // (the lowest-index one — what `collect_argument_vregs` makes the entry
    // param). Minting a fresh `ir.arg(0)` would create a second arg-0 vreg the
    // CFG builder leaves undefined (a latent "use of undefined vreg" that bails
    // SSA → legacy). If F has no arg-0 use, mint one.
    //
    // PERF (Phase C, 2026-06-04): two coupled pieces make F_osr ≈ a warm tier-2
    // compile, instead of writing every loop-carried local back to its slot and
    // re-guarding it each iteration:
    //
    //  1. F_osr is compiled with a deopt context
    //     (`Compiler::build_osr_variant_inner` enters a `DeoptContextGuard`).
    //     `apply_deopt_rewrite` then redirects the loop's guard bails to
    //     terminal deopt edges (re-invoke generic F with the original args,
    //     reloaded from the param slots) instead of a polymorphic merge-back —
    //     so the hot loop is safepoint-free and mem2reg promotes the
    //     loop-carried locals to registers.
    //
    //  2. Each int live-in (`slot ∈ int_slots`) is `GuardInt`-ed here at the
    //     entry, and the proven-int dst is stored — so the loop-header φ merges
    //     a *known-int* entry value with the loop's known-int body value, which
    //     lets `eliminate_redundant_guards` drop the now-redundant loop guards
    //     (matching warm's ~1 guard, down from ~18). A guard miss bails to
    //     `abort`; with deopt on, `apply_deopt_rewrite` redirects that bail to a
    //     generic re-invoke; with deopt off it returns the `OSR_NO_OSR` sentinel
    //     so tier-1 keeps running (sound: no loop iteration has executed yet).
    //     Param slots are NOT guarded (their stores must stay in the entry
    //     block, else the deopt eligibility check trips on a param-slot store
    //     outside entry) — they match warm, which also leaves params slot-resident.
    let buf = ir
        .instructions
        .iter()
        .flat_map(|ins| ins.get_registers())
        .filter(|r| r.argument == Some(0))
        .min_by_key(|r| r.index)
        .map(Value::Register)
        .unwrap_or_else(|| ir.arg(0));

    let abort = if int_slots.is_empty() {
        None
    } else {
        Some(ir.label("osr_entry_abort"))
    };

    let mut prologue: Vec<Instruction> = Vec::with_capacity(info.live_in_slots.len() * 2 + 4);
    // Pass 1: the un-guarded live-ins (param slots and non-int locals), stored
    // FIRST so every param slot (`0..f_arity`) is populated before any guard
    // below can bail. The deopt re-invoke a guard bail triggers reloads exactly
    // those param slots, so they must already hold F's original args.
    for (i, &slot) in info.live_in_slots.iter().enumerate() {
        if int_slots.contains(&slot) {
            continue;
        }
        let dest = ir.volatile_register();
        // HeapLoad's offset is a WORD index (the backend multiplies by 8).
        prologue.push(Instruction::HeapLoad(dest.into(), buf, i as i32));
        prologue.push(Instruction::StoreLocal(Value::Local(slot), dest.into()));
    }
    // Pass 2: the int live-ins, each guarded so the proven-int dst (which is
    // `known_int` for guard-elim) is what gets stored.
    for (i, &slot) in info.live_in_slots.iter().enumerate() {
        let Some(abort) = abort else { break };
        if !int_slots.contains(&slot) {
            continue;
        }
        let dest = ir.volatile_register();
        prologue.push(Instruction::HeapLoad(dest.into(), buf, i as i32));
        let guarded = ir.volatile_register();
        prologue.push(Instruction::GuardInt(guarded.into(), dest.into(), abort));
        prologue.push(Instruction::StoreLocal(Value::Local(slot), guarded.into()));
    }
    prologue.push(Instruction::Jump(info.header_label));
    // Abort block: only reachable via the entry-guard bails (the unconditional
    // `Jump(header)` above keeps it off the fast path). Returns the OSR_NO_OSR
    // sentinel so tier-1 keeps running. With deopt on, the deopt rewrite
    // replaces these bail edges and DCE removes this block.
    if let Some(abort) = abort {
        let sentinel = ir.volatile_register();
        prologue.push(Instruction::Label(abort));
        prologue.push(Instruction::Assign(
            sentinel.into(),
            Value::RawValue(crate::builtins::OSR_NO_OSR),
        ));
        prologue.push(Instruction::Ret(sentinel.into()));
    }
    ir.instructions.splice(0..0, prologue);
    // Prepending shifted every instruction position, so the cached
    // label→position map is stale; rebuild it or jumps mis-resolve.
    ir.rebuild_label_locations();
    ir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn osr_variant_prepends_entry_and_strips_osr_checks() {
        let mut base = Ir::new(0);
        // Simulate a function body with a loop:
        //   <prologue/pre-loop>           (will become dead)
        //   header:  ...body...
        //            osr_check            (must be stripped)
        //            jump header          (back-edge)
        let pre = base.assign_new(Value::Null); // some pre-loop instruction
        let _ = pre;
        let header = base.label("loop_start");
        base.write_label(header);
        base.instructions
            .push(Instruction::OsrCheck(0x10, 0x20, 0x30));
        base.instructions.push(Instruction::Jump(header));

        let info = OsrLoopInfo {
            header_label: header,
            live_in_slots: vec![0, 2, 5],
        };
        // Empty int_slots => no entry guards: plain HeapLoad+StoreLocal layout.
        let variant = build_osr_variant_ir(&base, &info, &HashSet::new());

        // No OSR checks survive.
        assert!(
            !variant
                .instructions
                .iter()
                .any(|i| matches!(i, Instruction::OsrCheck(..))),
            "OsrCheck must be stripped from F_osr"
        );

        // The first instructions are the OSR entry: per live-in, a HeapLoad
        // of buf[i] (from arg0) followed by a StoreLocal into its slot; then
        // a Jump to the header.
        for (i, &slot) in info.live_in_slots.iter().enumerate() {
            match &variant.instructions[i * 2] {
                Instruction::HeapLoad(Value::Register(_), Value::Register(b), off) => {
                    assert_eq!(b.argument, Some(0), "buffer base must be arg0");
                    assert_eq!(
                        *off, i as i32,
                        "live-in {} loads buf[{}] (word offset)",
                        i, i
                    );
                }
                other => panic!("expected HeapLoad(_, arg0, {}), got {other:?}", i * 8),
            }
            match &variant.instructions[i * 2 + 1] {
                Instruction::StoreLocal(Value::Local(s), Value::Register(_)) => {
                    assert_eq!(*s, slot, "live-in {} should store into slot {}", i, slot);
                }
                other => panic!("expected StoreLocal(Local({slot}), _), got {other:?}"),
            }
        }
        match &variant.instructions[info.live_in_slots.len() * 2] {
            Instruction::Jump(l) => {
                assert_eq!(*l, header, "OSR entry must jump to the loop header")
            }
            other => panic!("expected Jump(header), got {other:?}"),
        }
    }
}
