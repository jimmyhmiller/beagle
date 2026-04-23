#[cfg(debug_assertions)]
use std::cell::RefCell;
#[cfg(debug_assertions)]
use std::collections::VecDeque;
use std::{
    cell::Cell,
    error::Error,
    ffi::{CStr, c_void},
    mem::{self, transmute},
    slice::{from_raw_parts, from_raw_parts_mut},
    thread,
};

#[cfg(debug_assertions)]
use crate::collections::TYPE_ID_FRAME;
#[cfg(debug_assertions)]
use crate::types::Header;
use crate::{
    Message,
    collections::{
        GcHandle, PersistentVec, TYPE_ID_CONS_STRING, TYPE_ID_KEYWORD,
        TYPE_ID_MULTI_ARITY_FUNCTION, TYPE_ID_STRING, TYPE_ID_STRING_SLICE,
    },
    gc::STACK_SIZE,
    get_runtime,
    runtime::{DiskLocation, DispatchTable, FFIInfo, FFIType, RawPtr, Runtime},
    types::{BuiltInTypes, HeapObject},
};

// FileResultData is used in local imports where needed

use rand::Rng;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

// Thread-local storage for the frame pointer, stack pointer, and gc return
// address at builtin entry. Set by builtins that receive frame_pointer from
// Beagle code; read by `gc()` when triggered internally (e.g., during
// allocation).
//
// Grouped into a single compound thread-local so `save_gc_context!` goes
// through exactly one `_tlv_get_addr` thunk call instead of three. Per
// allocation this alone is ~2×–3× cheaper on the TLS path, and allocation
// is hot enough (≈600M calls on the binary_tree benchmark) that the
// difference shows up in wall time.
pub struct GcContextSlots {
    pub saved_frame_pointer: Cell<usize>,
    pub saved_stack_pointer: Cell<usize>,
    pub saved_gc_return_addr: Cell<usize>,
}

thread_local! {
    pub static GC_CONTEXT: GcContextSlots = const {
        GcContextSlots {
            saved_frame_pointer: Cell::new(0),
            saved_stack_pointer: Cell::new(0),
            saved_gc_return_addr: Cell::new(0),
        }
    };
}

// Top of the GC frame linked list. Points to the frame header address
// of the most recently entered Beagle function. Each frame's prev pointer
// (at [header_addr - 8], i.e. [FP-16]) links to the previous frame's header.
// GC walks this chain instead of the FP chain, so captured/restored
// continuation frames are naturally excluded/included.
//
// Storage lives in the per-thread MutatorState (see runtime::MutatorState).
// The thread_local below is only for the debug GC chain trace buffer, which
// is never touched on the hot path.
#[cfg(debug_assertions)]
thread_local! {
    static GC_CHAIN_TRACE: RefCell<VecDeque<String>> = const { RefCell::new(VecDeque::new()) };
}

// ============================================================================
// Submodules
// ============================================================================

mod allocation;
mod apply;
mod assertions;
mod async_file;
mod bindings;
pub mod collections;
mod continuations;
mod debugger;
mod diagnostics;
mod dispatch;
mod effects;
mod exceptions;
mod ffi;
mod filesystem;
mod gc_frame;
mod install;
mod io;
mod json;
mod keywords;
mod math;
mod networking;
mod objects;
mod reflect;
pub mod regex;
pub mod reset_shift;
mod strings;
mod threads;

// ============================================================================
// Re-exports
// ============================================================================

pub use allocation::*;
pub use apply::*;
pub use assertions::*;
pub use async_file::*;
pub use bindings::*;
pub use continuations::*;
pub use debugger::*;
pub use diagnostics::*;
pub use dispatch::*;
pub use effects::*;
pub use exceptions::*;
pub use ffi::*;
pub use filesystem::*;
pub use gc_frame::*;
pub use io::*;
pub use json::*;
pub use keywords::*;
pub use math::*;
pub use networking::*;
pub use objects::*;
pub use reflect::*;
pub use reset_shift::*;
pub use strings::*;
pub use threads::*;

// ============================================================================
// Shared infrastructure (thread-local accessors, helpers, gc function)
// ============================================================================

/// Save the frame pointer for later use by gc().
/// Called by builtins that receive frame_pointer from Beagle.
pub fn save_frame_pointer(fp: usize) {
    GC_CONTEXT.with(|ctx| ctx.saved_frame_pointer.set(fp));
}

/// Get the saved frame pointer.
/// Returns 0 if none has been saved (shouldn't happen in normal operation).
pub fn get_saved_frame_pointer() -> usize {
    GC_CONTEXT.with(|ctx| ctx.saved_frame_pointer.get())
}

/// Save the stack pointer for later use by throw_runtime_error.
/// Called by builtins that receive stack_pointer from Beagle.
pub fn save_stack_pointer(sp: usize) {
    GC_CONTEXT.with(|ctx| ctx.saved_stack_pointer.set(sp));
}

/// Get the saved stack pointer.
/// Returns 0 if none has been saved (shouldn't happen in normal operation).
pub fn get_saved_stack_pointer() -> usize {
    GC_CONTEXT.with(|ctx| ctx.saved_stack_pointer.get())
}

/// Save the gc return address for later use by gc().
/// Called by builtins that receive frame_pointer/stack_pointer from Beagle.
pub fn save_gc_return_addr(addr: usize) {
    GC_CONTEXT.with(|ctx| ctx.saved_gc_return_addr.set(addr));
}

/// Get the saved gc return address.
/// Returns 0 if none has been saved.
pub fn get_saved_gc_return_addr() -> usize {
    GC_CONTEXT.with(|ctx| ctx.saved_gc_return_addr.get())
}

/// Get the current top of the GC frame linked list.
/// Used by the stack walker to start walking frames.
pub fn get_gc_frame_top() -> usize {
    unsafe { (*crate::runtime::current_mutator_state()).gc_frame_top }
}

/// Set the GC frame top directly.
/// Used when restoring continuation frames into the GC chain.
pub fn set_gc_frame_top(v: usize) {
    #[cfg(debug_assertions)]
    {
        if std::env::var("BEAGLE_DEBUG_GC_CHAIN_TRACE").is_ok() {
            if v == 0 {
                gc_frame::record_gc_chain_event("set_gc_frame_top header=0x0".to_string());
            } else {
                let header = Header::from_usize(unsafe { *(v as *const usize) });
                let fp = v + 8;
                let saved_fp = unsafe { *(fp as *const usize) };
                let return_addr = unsafe { *((fp + 8) as *const usize) };
                let function = crate::get_runtime()
                    .get()
                    .get_function_containing_pointer(return_addr as *const u8)
                    .map(|(function, offset)| format!(" {}+{:#x}", function.name, offset))
                    .unwrap_or_default();
                gc_frame::record_gc_chain_event(format!(
                    "set_gc_frame_top header={:#x} fp={:#x} type_id={} size={} saved_fp={:#x} ret={:#x}{}",
                    v, fp, header.type_id, header.size, saved_fp, return_addr, function
                ));
            }
        }
        if std::env::var("BEAGLE_DEBUG_GC_CHAIN_WRITES").is_ok() && v != 0 {
            let header = Header::from_usize(unsafe { *(v as *const usize) });
            let fp = v + 8;
            let saved_fp = unsafe { *(fp as *const usize) };
            let return_addr = unsafe { *((fp + 8) as *const usize) };
            if header.type_id != TYPE_ID_FRAME || saved_fp == 0 || return_addr < 0x1000 {
                if let Some((function, offset)) = crate::get_runtime()
                    .get()
                    .get_function_containing_pointer(return_addr as *const u8)
                {
                    eprintln!(
                        "[gc-top-write] via=set_gc_frame_top header={:#x} fp={:#x} type_id={} size={} saved_fp={:#x} ret={:#x} fn={}+{:#x}",
                        v,
                        fp,
                        header.type_id,
                        header.size,
                        saved_fp,
                        return_addr,
                        function.name,
                        offset
                    );
                } else {
                    eprintln!(
                        "[gc-top-write] via=set_gc_frame_top header={:#x} fp={:#x} type_id={} size={} saved_fp={:#x} ret={:#x}",
                        v, fp, header.type_id, header.size, saved_fp, return_addr
                    );
                }
            }
        }
    }
    unsafe { (*crate::runtime::current_mutator_state()).gc_frame_top = v };
}

/// Saved GC context - used to save/restore around calls back into Beagle
pub struct SavedGcContext {
    pub frame_pointer: usize,
    pub stack_pointer: usize,
    pub gc_return_addr: usize,
    pub gc_frame_top: usize,
}

/// Save the current GC context. Call this BEFORE calling back into Beagle.
/// The Beagle code may call builtins that update the saved GC context.
/// After the Beagle code returns, call restore_gc_context to restore it.
pub fn save_current_gc_context() -> SavedGcContext {
    SavedGcContext {
        frame_pointer: get_saved_frame_pointer(),
        stack_pointer: get_saved_stack_pointer(),
        gc_return_addr: get_saved_gc_return_addr(),
        gc_frame_top: get_gc_frame_top(),
    }
}

/// Restore a previously saved GC context. Call this AFTER calling back into Beagle.
/// This ensures that if GC runs after the Beagle call returns, it uses the
/// correct (non-stale) frame pointer.
pub fn restore_gc_context(ctx: SavedGcContext) {
    save_frame_pointer(ctx.frame_pointer);
    save_stack_pointer(ctx.stack_pointer);
    save_gc_return_addr(ctx.gc_return_addr);
    set_gc_frame_top(ctx.gc_frame_top);
}

/// Macro to save frame pointer, stack pointer, and gc return address for GC stack walking.
/// The gc_return_addr is the return address from the Beagle code that called
/// this builtin. On ARM64, this was in LR and is now saved at [Rust_FP + 8]
/// by Rust's prologue. On x86-64, it's also at [Rust_FP + 8].
///
/// The saved frame_pointer (Beagle FP) is used by __pause to check if the
/// saved gc_return_addr is still valid for the current call.
/// The saved stack_pointer is used by throw_runtime_error for error handling.
#[macro_export]
macro_rules! save_gc_context {
    ($stack_pointer:expr, $frame_pointer:expr) => {{
        // Single TLS thunk (`_tlv_get_addr` on macOS arm64) reaches all three
        // slots at once. See [`GcContextSlots`] for why these were
        // consolidated.
        let rust_fp = $crate::builtins::get_current_rust_frame_pointer();
        // SAFETY: rust_fp + 8 points to the saved return address in the Rust stack frame
        #[allow(unused_unsafe)]
        let gc_return_addr = unsafe { *((rust_fp + 8) as *const usize) };
        $crate::builtins::GC_CONTEXT.with(|ctx| {
            ctx.saved_frame_pointer.set($frame_pointer);
            ctx.saved_stack_pointer.set($stack_pointer);
            ctx.saved_gc_return_addr.set(gc_return_addr);
        });
    }};
}

/// Cached function pointer to the `beagle.builtin/read-fp` JIT trampoline.
/// Used by GC to walk the stack starting from the Rust caller's frame.
/// Resolving by name was doing a linear scan over every registered function
/// on every allocation; caching the pointer once at first use removes that
/// cost without changing the semantics (the trampoline has no prologue, so it
/// reads the *caller's* FP, which is the contract we need and which works
/// uniformly across ARM64 and x86-64 debug/release). Replacing it with inline
/// asm reading x29/rbp directly is tempting but fragile on x86-64 debug —
/// Rust's codegen there doesn't always produce a prologue that leaves rbp
/// pointing at our own saved FP, so a dereference returns a value one level
/// too deep (produces SIGSEGVs in continuation-capture tests).
static READ_FP_TRAMPOLINE: AtomicUsize = AtomicUsize::new(0);

#[inline(always)]
pub fn get_current_rust_frame_pointer() -> usize {
    let cached = READ_FP_TRAMPOLINE.load(Ordering::Relaxed);
    let ptr = if cached != 0 {
        cached
    } else {
        let fn_entry = get_runtime()
            .get()
            .get_function_by_name("beagle.builtin/read-fp")
            .expect("read-fp trampoline not found");
        let p: usize = fn_entry.pointer.into();
        READ_FP_TRAMPOLINE.store(p, Ordering::Relaxed);
        p
    };
    let read_fp: extern "C" fn() -> usize = unsafe { std::mem::transmute::<_, _>(ptr) };
    read_fp()
}

/// Check if a value is a string-like type (string literal, heap string, string slice, or cons string)
#[inline]
fn is_string_like(value: usize) -> bool {
    use crate::collections::{TYPE_ID_CONS_STRING, TYPE_ID_STRING, TYPE_ID_STRING_SLICE};
    let tag = BuiltInTypes::get_kind(value);
    if tag == BuiltInTypes::String {
        return true;
    }
    if !BuiltInTypes::is_heap_pointer(value) {
        return false;
    }
    let heap_obj = HeapObject::from_tagged(value);
    let type_id = heap_obj.get_header().type_id;
    type_id == TYPE_ID_STRING || type_id == TYPE_ID_STRING_SLICE || type_id == TYPE_ID_CONS_STRING
}

#[inline(always)]
fn print_call_builtin(_runtime: &Runtime, _name: &str) {
    #[cfg(debug_assertions)]
    {
        if _runtime.get_command_line_args().print_builtin_calls {
            println!("Calling: {}", _name);
        }
    }
}

pub fn get_current_stack_pointer() -> usize {
    let runtime = get_runtime().get();
    let fn_entry = runtime
        .get_function_by_name("beagle.builtin/read-sp")
        .expect("read-sp trampoline not found");
    let read_sp: extern "C" fn() -> usize =
        unsafe { std::mem::transmute::<_, _>(fn_entry.pointer) };
    read_sp()
}

/// Get the current frame pointer (RBP on x86-64, X29 on ARM64).
/// Note: This is currently unused because Rust functions may not preserve
/// frame pointers, making FP-chain traversal unreliable for GC.
#[allow(dead_code)]
pub fn get_current_frame_pointer() -> usize {
    let runtime = get_runtime().get();
    let fn_entry = runtime
        .get_function_by_name("beagle.builtin/read-fp")
        .expect("read-fp trampoline not found");
    let read_fp: extern "C" fn() -> usize =
        unsafe { std::mem::transmute::<_, _>(fn_entry.pointer) };
    read_fp()
}

fn print_stack(_stack_pointer: usize) {
    let runtime = get_runtime().get_mut();
    let stack_base = runtime.get_stack_base();
    let stack_begin = stack_base - STACK_SIZE;

    // Get the current frame pointer via JIT trampoline
    let mut current_frame_ptr = {
        let fn_entry = runtime
            .get_function_by_name("beagle.builtin/read-fp")
            .expect("read-fp trampoline not found");
        let read_fp: extern "C" fn() -> usize =
            unsafe { std::mem::transmute::<_, _>(fn_entry.pointer) };
        read_fp()
    };

    println!("Stack trace:");

    let mut frame_count = 0;
    const MAX_FRAMES: usize = 100; // Prevent infinite loops

    while frame_count < MAX_FRAMES && current_frame_ptr != 0 {
        // Validate frame pointer is within stack bounds
        if current_frame_ptr < stack_begin || current_frame_ptr >= stack_base {
            break;
        }

        // Frame layout in our calling convention:
        // [previous_X29] [X30/return_addr] <- X29 points here
        // [zero] [zero] [locals...]
        let frame = current_frame_ptr as *const usize;
        let previous_frame_ptr = unsafe { *frame.offset(0) }; // Previous X29
        let return_address = unsafe { *frame.offset(1) }; // X30 (return address)

        // Look up the function containing this return address
        for function in runtime.functions.iter() {
            let function_size = function.size;
            let function_start = usize::from(function.pointer);
            let range = function_start..function_start + function_size;
            if range.contains(&return_address) {
                match (&function.source_file, &function.source_line) {
                    (Some(file), Some(line)) => {
                        println!("  at {} ({}:{})", function.name, file, line);
                    }
                    (Some(file), None) => {
                        println!("  at {} ({})", function.name, file);
                    }
                    _ => {
                        println!("  at {}", function.name);
                    }
                }
                break;
            }
        }

        // Move to the previous frame
        current_frame_ptr = previous_frame_ptr;
        frame_count += 1;
    }
}

pub unsafe extern "C" fn gc(stack_pointer: usize, frame_pointer: usize) -> usize {
    // Save the GC context including the return address
    save_gc_context!(stack_pointer, frame_pointer);
    #[cfg(feature = "debug-gc")]
    {
        eprintln!(
            "DEBUG gc: stack_pointer={:#x}, frame_pointer={:#x}",
            stack_pointer, frame_pointer
        );
    }
    let runtime = get_runtime().get_mut();
    runtime.gc_impl(frame_pointer);
    BuiltInTypes::null_value() as usize
}
