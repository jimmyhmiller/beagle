//! Delimited continuation primitives: `reset` and `shift`.
//!
//! This module is the entire runtime implementation of Beagle's
//! first-class delimited continuations. Everything specific to
//! reset/shift lives here — the heap-object type, the stack-walking
//! helpers, the runtime entry points called from compiled code, and
//! the tiny pure-Rust continuation_trampoline that resumes a captured
//! continuation. Generic runtime primitives (heap allocation, Beagle
//! frame headers, thread-local GC context) live in their usual homes;
//! this file imports them but adds nothing to them.
//!
//! # Design in one paragraph
//!
//! A prompt is a frame on the stack whose code is `beagle.core/__reset__`
//! (defined in `standard-library/std.bg`). The runtime knows that
//! function's code range and uses it as a sentinel: to find the
//! enclosing prompt of a `shift`, walk the FP chain from the current
//! frame upward and look for the topmost frame whose saved LR points
//! into `__reset__`. That frame — the outermost captured body frame —
//! is the prompt boundary. Nothing is stored in a side-table; nothing
//! is snapshotted at `reset` entry; the prompt's SP/FP/return-address
//! are all re-derived from the live stack at `shift` time.
//!
//! Capturing a continuation is a byte copy: copy
//! `[current_sp, outermost_body_fp + 16)` into a heap-allocated
//! `ContinuationObject`, relocating saved-FP links from stack
//! addresses to the heap copy's addresses. Invoking a captured
//! continuation is the reverse byte copy, plus an FP-chain relocation
//! and a GC-prev-chain rebuild, plus a jump to the saved resume
//! address — all done in `continuation_trampoline`, which is ~140
//! lines of inline Rust + one call to the `return-jump` JIT trampoline
//! to perform the final SP/FP/PC switch. Multi-shot falls out
//! naturally because the captured bytes are never mutated: each
//! invocation copies a fresh snapshot onto a fresh region of the
//! stack above the invoker's frame.
//!
//! # Layout of this file
//!
//! 1. `is_pc_in_reset_function` — lazily-cached lookup of `__reset__`'s
//!    code range. Used by `find_enclosing_reset_frame`.
//! 2. Stack-segment byte utilities (`relocate_segment_caller_fp_links`
//!    and friends) — walk a region of copied frame bytes and patch
//!    saved-FP links by a delta. Shared between capture and invoke.
//! 3. `ContinuationObject` — heap-object wrapper. A captured
//!    continuation is a `TYPE_ID_CONTINUATION` heap object whose
//!    fields describe how to resume it. Also owns the
//!    `normalized_segment_base` helper that transparently re-patches
//!    FP links when GC moves the backing heap object.
//! 4. `find_enclosing_reset_frame` — the FP-chain walker that locates
//!    the prompt boundary.
//! 5. `capture_continuation_runtime_inner` — called from `shift`.
//!    Builds a `ContinuationObject` from the current execution state.
//! 6. `return_from_shift_runtime_inner` — called when a `shift` body
//!    completes normally. Longjmps back to `__reset__`'s caller with
//!    the shift body's result as `__reset__`'s return value.
//! 7. `continuation_trampoline` — the closure body for invoked
//!    continuations. Copies bytes, patches chains, jumps.
//!
//! # What's not here
//!
//! - `Ast::Reset` / `Ast::Shift` IR compilation — see `src/ast.rs`.
//! - `__reset__`'s Beagle source — see `standard-library/std.bg`.
//! - The `return-jump` / `read-fp` JIT trampolines — see `src/main.rs`.
//! - `handle` / `perform` / resumable `try/catch` — Refactor A
//!   disabled these; Refactor B will rebuild them on top of
//!   reset/shift. Their error-throwing stubs live in
//!   `src/builtins/continuations.rs`.

use std::sync::OnceLock;

use crate::builtins::GC_FRAME_TOP;
use crate::collections::{TYPE_ID_CONTINUATION, TYPE_ID_FRAME};
use crate::runtime::Runtime;
use crate::types::{BuiltInTypes, Header, HeapObject};

// ============================================================================
// §1. Reset frame identity
// ============================================================================

/// Cached code range of `beagle.core/__reset__`. Populated the first
/// time `is_pc_in_reset_function` is called with a non-zero PC. Set
/// once and never changed — `__reset__`'s compiled code address is
/// stable for the life of the runtime.
static RESET_CODE_RANGE: OnceLock<(usize, usize)> = OnceLock::new();

/// Returns `true` if `pc` lies within the code range of
/// `beagle.core/__reset__`. Used by `find_enclosing_reset_frame` to
/// identify the prompt boundary during FP-chain walks.
pub fn is_pc_in_reset_function(pc: usize) -> bool {
    let (start, end) = *RESET_CODE_RANGE.get_or_init(|| {
        let runtime = crate::get_runtime().get();
        match runtime.get_function_by_name("beagle.core/__reset__") {
            Some(f) => {
                let start: usize = f.pointer.into();
                (start, start + f.size)
            }
            // Before stdlib compiles there's nothing to match. Return an
            // empty range; the OnceLock will be re-populated on the next
            // call after compilation completes... actually no, OnceLock
            // locks once. Use a sentinel value so future calls re-check.
            None => (0, 0),
        }
    });
    // Empty sentinel: try once more to populate (OnceLock only runs the
    // closure once, so this only helps for the specific case where the
    // first call happens before __reset__ exists; fall back to a direct
    // name lookup). Practically this only matters during early bring-up.
    if end == 0 {
        let runtime = crate::get_runtime().get();
        if let Some(f) = runtime.get_function_by_name("beagle.core/__reset__") {
            let fstart: usize = f.pointer.into();
            let fend = fstart + f.size;
            return pc >= fstart && pc < fend;
        }
        return false;
    }
    pc >= start && pc < end
}

// ============================================================================
// §2. Byte-region helpers for segment relocation
// ============================================================================
//
// A captured continuation's bytes are stored in a heap object whose
// payload is a verbatim snapshot of the original stack region. The
// saved-FP links inside those bytes are absolute addresses, so when
// the bytes are copied (from stack to heap at capture time, from heap
// to stack at invoke time, or from old heap location to new heap
// location by a compacting GC), every saved-FP that fell within the
// old range needs to be shifted by `new_base - old_base`.
//
// `relocate_segment_caller_fp_links` walks the caller-FP chain starting
// at the innermost frame and rewrites each in-range saved-FP by the
// delta. It stops when the next fp to follow is outside the range,
// which is how it detects the "top" of the captured region.
//
// `relocate_segment_gc_prev_links` is the GC-prev-chain analog for
// cases where there's no valid FP chain entry point (e.g. the
// innermost frame was stored directly by a GC chain traversal rather
// than by the normal FP link scheme).
//
// `rebuild_segment_gc_prev_links_from_caller_chain` is the inverse:
// given an FP chain in the segment, write each frame's GC-prev slot
// (at `[header - 8]`) to point to its parent frame's header. Used
// both at capture (to initialize the GC prev chain in the copy) and
// at invoke (to rebuild it in the destination stack copy).

pub fn relocate_segment_caller_fp_links(
    data_base: usize,
    size: usize,
    fp_offset: usize,
    original_base: usize,
) -> usize {
    if size == 0 || original_base == 0 {
        return original_base;
    }
    if original_base == data_base {
        return data_base;
    }

    let delta = data_base as isize - original_base as isize;

    if fp_offset < size {
        let mut fp = data_base + fp_offset;
        while fp >= data_base && fp < data_base + size && fp != 0 {
            let caller_fp_slot = fp as *mut usize;
            let caller_fp = unsafe { *caller_fp_slot };
            let next_fp = if caller_fp >= original_base && caller_fp < original_base + size {
                let relocated = (caller_fp as isize + delta) as usize;
                unsafe {
                    *caller_fp_slot = relocated;
                }
                relocated
            } else {
                caller_fp
            };

            if next_fp == 0 || next_fp < data_base || next_fp >= data_base + size {
                break;
            }
            fp = next_fp;
        }
    }

    data_base
}

pub fn relocate_segment_gc_prev_links(
    data_base: usize,
    size: usize,
    gc_frame_offset: usize,
    original_base: usize,
) -> usize {
    if size == 0 || gc_frame_offset >= size || original_base == 0 {
        return original_base;
    }
    if original_base == data_base {
        return data_base;
    }

    let delta = data_base as isize - original_base as isize;
    let mut header_addr = data_base + gc_frame_offset;
    while header_addr >= data_base && header_addr < data_base + size && header_addr != 0 {
        let header = Header::from_usize(unsafe { *(header_addr as *const usize) });
        if header.type_id != TYPE_ID_FRAME {
            return original_base;
        }

        let prev_slot = (header_addr - 8) as *mut usize;
        let prev_val = unsafe { *prev_slot };
        let next_header = if prev_val >= original_base && prev_val < original_base + size {
            let relocated = (prev_val as isize + delta) as usize;
            unsafe {
                *prev_slot = relocated;
            }
            relocated
        } else {
            prev_val
        };

        if next_header == 0 || next_header < data_base || next_header >= data_base + size {
            break;
        }
        header_addr = next_header;
    }

    data_base
}

pub fn rebuild_segment_gc_prev_links_from_caller_chain(
    data_base: usize,
    size: usize,
    fp_offset: usize,
    outer_prev: usize,
) -> Option<usize> {
    if size == 0 || fp_offset >= size {
        return None;
    }

    let segment_top = data_base + size;
    let mut fp = data_base + fp_offset;
    let gc_frame_top = fp.checked_sub(8)?;

    while fp >= data_base + 8 && fp < segment_top {
        let header_addr = fp - 8;
        let header = Header::from_usize(unsafe { *(header_addr as *const usize) });
        if header.type_id != TYPE_ID_FRAME {
            return None;
        }

        let caller_fp = unsafe { *(fp as *const usize) };
        let prev_value = if caller_fp >= data_base + 8 && caller_fp < segment_top {
            caller_fp - 8
        } else {
            outer_prev
        };
        unsafe {
            *((fp - 16) as *mut usize) = prev_value;
        }

        if caller_fp < data_base + 8 || caller_fp >= segment_top {
            break;
        }
        fp = caller_fp;
    }

    Some(gc_frame_top)
}

// ============================================================================
// §3. ContinuationObject
// ============================================================================
//
// A captured continuation is a heap object with `type_id =
// TYPE_ID_CONTINUATION` and the field layout defined below. The
// object itself holds metadata; the actual saved stack bytes live in
// a separate opaque-bytes heap object pointed to by `FIELD_SEGMENT_PTR`.
//
// Many of the `FIELD_PROMPT_*` / `FIELD_EXC_*` slots are holdovers
/// Captured continuation: a snapshot of the stack between a shift
/// point and its enclosing reset. Stored as a heap object with
/// `type_id = TYPE_ID_CONTINUATION`.
pub struct ContinuationObject {
    heap_obj: HeapObject,
}

impl ContinuationObject {
    const FIELD_RESUME_ADDRESS: usize = 0;
    const FIELD_RESULT_LOCAL: usize = 1;
    const FIELD_SEGMENT_PTR: usize = 2;
    /// Offset from the segment base to the **innermost** captured
    /// frame's FP. This is the "resume FP" — the FP the resumed body
    /// starts executing with — and also the entry point for the
    /// saved-FP-chain relocation walk.
    const FIELD_SEGMENT_FRAME_POINTER_OFFSET: usize = 3;
    /// Offset from the segment base to the **outermost** captured
    /// frame's FP. This is where the bottom of the captured region
    /// lands in the destination stack during invocation; the
    /// invocation trampoline uses it to place the copy so the
    /// outermost frame's saved FP/LR slots overlay the trampoline's
    /// own saved caller FP/LR.
    const FIELD_SEGMENT_OUTERMOST_FP_OFFSET: usize = 4;
    const FIELD_SEGMENT_SIZE: usize = 5;
    /// The data base address at capture time. Used by compacting GC
    /// to compute the relocation delta when the segment heap object
    /// moves.
    const FIELD_SEGMENT_ORIGINAL_DATA_BASE: usize = 6;

    /// Total number of fields; used when allocating a fresh
    /// continuation object.
    pub const NUM_FIELDS: usize = 7;

    pub fn from_tagged(tagged: usize) -> Option<Self> {
        let heap_obj = HeapObject::try_from_tagged(tagged)?;
        Self::from_heap_object(heap_obj)
    }

    pub fn from_heap_object(heap_obj: HeapObject) -> Option<Self> {
        if heap_obj.get_type_id() == TYPE_ID_CONTINUATION as usize {
            Some(Self { heap_obj })
        } else {
            None
        }
    }

    pub fn tagged_ptr(&self) -> usize {
        self.heap_obj.tagged_pointer()
    }

    pub fn resume_address(&self) -> usize {
        BuiltInTypes::untag(self.heap_obj.get_field(Self::FIELD_RESUME_ADDRESS))
    }

    pub fn result_local(&self) -> isize {
        BuiltInTypes::untag_isize(self.heap_obj.get_field(Self::FIELD_RESULT_LOCAL) as isize)
    }

    /// Returns the tagged heap pointer to the opaque bytes object containing captured stack frames.
    pub fn segment_ptr(&self) -> usize {
        self.heap_obj.get_field(Self::FIELD_SEGMENT_PTR)
    }

    /// Returns the offset of the innermost frame pointer within the captured segment data.
    pub fn segment_frame_pointer_offset(&self) -> usize {
        BuiltInTypes::untag(
            self.heap_obj
                .get_field(Self::FIELD_SEGMENT_FRAME_POINTER_OFFSET),
        )
    }

    /// Returns the offset of the outermost frame pointer within the captured segment data.
    pub fn segment_gc_frame_offset(&self) -> usize {
        BuiltInTypes::untag(
            self.heap_obj
                .get_field(Self::FIELD_SEGMENT_OUTERMOST_FP_OFFSET),
        )
    }

    /// Returns the size of the captured segment data in bytes.
    pub fn segment_size(&self) -> usize {
        BuiltInTypes::untag(self.heap_obj.get_field(Self::FIELD_SEGMENT_SIZE))
    }

    pub fn set_segment_frame_pointer_offset(&self, offset: usize) {
        self.heap_obj.write_field(
            Self::FIELD_SEGMENT_FRAME_POINTER_OFFSET as i32,
            BuiltInTypes::Int.tag(offset as isize) as usize,
        );
    }

    pub fn set_segment_gc_frame_offset(&self, offset: usize) {
        self.heap_obj.write_field(
            Self::FIELD_SEGMENT_OUTERMOST_FP_OFFSET as i32,
            BuiltInTypes::Int.tag(offset as isize) as usize,
        );
    }

    pub fn set_segment_size(&self, size: usize) {
        self.heap_obj.write_field(
            Self::FIELD_SEGMENT_SIZE as i32,
            BuiltInTypes::Int.tag(size as isize) as usize,
        );
    }

    /// Returns the data base address at the time the segment was captured.
    pub fn segment_original_data_base(&self) -> usize {
        BuiltInTypes::untag(
            self.heap_obj
                .get_field(Self::FIELD_SEGMENT_ORIGINAL_DATA_BASE),
        )
    }

    /// Sets the original data base address for compacting GC relocation detection.
    pub fn set_segment_original_data_base(&mut self, data_base: usize) {
        self.heap_obj.write_field(
            Self::FIELD_SEGMENT_ORIGINAL_DATA_BASE as i32,
            BuiltInTypes::Int.tag(data_base as isize) as usize,
        );
    }

    pub fn set_segment_ptr_with_barrier(&mut self, runtime: &mut Runtime, segment_ptr: usize) {
        runtime.set_field_with_barrier(self.tagged_ptr(), Self::FIELD_SEGMENT_PTR, segment_ptr);
    }

    /// Internally, the captured segment's saved-FP links are absolute
    /// pointers that were valid when the bytes were last copied. If a
    /// compacting GC has moved the heap object since then, those
    /// pointers are stale. This method detects that case (by
    /// comparing `FIELD_SEGMENT_ORIGINAL_DATA_BASE` against the
    /// current data base) and rewrites the chain in place, updating
    /// the recorded base so subsequent calls are no-ops.
    fn normalized_segment_base(&self) -> Option<(usize, usize)> {
        let seg_tagged = self.segment_ptr();
        if seg_tagged == 0
            || seg_tagged == BuiltInTypes::null_value() as usize
            || !BuiltInTypes::is_heap_pointer(seg_tagged)
        {
            return None;
        }
        let seg_obj = HeapObject::from_tagged(seg_tagged);
        let data_base = seg_obj.untagged() + seg_obj.header_size();
        let size = self.segment_size();
        let original_base = self.segment_original_data_base();
        if size == 0 {
            return None;
        }

        let fp_offset = self.segment_frame_pointer_offset();
        let gc_offset = self.segment_gc_frame_offset();
        let caller_moved = fp_offset < size
            && relocate_segment_caller_fp_links(data_base, size, fp_offset, original_base)
                != original_base;
        if fp_offset < size {
            let _ = rebuild_segment_gc_prev_links_from_caller_chain(data_base, size, fp_offset, 0);
        } else if gc_offset < size {
            let _ = relocate_segment_gc_prev_links(data_base, size, gc_offset, original_base);
        }
        if caller_moved {
            self.heap_obj.write_field(
                Self::FIELD_SEGMENT_ORIGINAL_DATA_BASE as i32,
                BuiltInTypes::Int.tag(data_base as isize) as usize,
            );
        }

        Some((data_base, size))
    }

    /// Returns (segment_data_base, segment_data_top, innermost_fp) for
    /// this continuation's captured segment, after normalizing for any
    /// GC move that may have happened since capture. Returns None if
    /// the continuation has no captured segment.
    pub fn segment_frame_info(&self) -> Option<(usize, usize, usize)> {
        let (data_base, size) = self.normalized_segment_base()?;
        let fp_offset = self.segment_frame_pointer_offset();
        if fp_offset >= size {
            return None;
        }
        Some((data_base, data_base + size, data_base + fp_offset))
    }

    pub fn segment_gc_frame_info(&self) -> Option<(usize, usize, usize)> {
        let (data_base, size) = self.normalized_segment_base()?;
        let gc_offset = self.segment_gc_frame_offset();
        if gc_offset >= size {
            return None;
        }
        Some((data_base, data_base + size, data_base + gc_offset))
    }

    /// Initialize a freshly-allocated continuation heap object.
    /// Segment metadata is set to zero and expected to be filled in
    /// by the caller after the backing segment object is allocated.
    pub fn initialize(heap_obj: &mut HeapObject, resume_address: usize, result_local: isize) {
        heap_obj.write_type_id(TYPE_ID_CONTINUATION as usize);
        heap_obj.write_field(
            Self::FIELD_RESUME_ADDRESS as i32,
            BuiltInTypes::Int.tag(resume_address as isize) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_RESULT_LOCAL as i32,
            BuiltInTypes::Int.tag(result_local) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_SEGMENT_PTR as i32,
            BuiltInTypes::null_value() as usize,
        );
        heap_obj.write_field(
            Self::FIELD_SEGMENT_FRAME_POINTER_OFFSET as i32,
            BuiltInTypes::Int.tag(0) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_SEGMENT_OUTERMOST_FP_OFFSET as i32,
            BuiltInTypes::Int.tag(0) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_SEGMENT_SIZE as i32,
            BuiltInTypes::Int.tag(0) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_SEGMENT_ORIGINAL_DATA_BASE as i32,
            BuiltInTypes::Int.tag(0) as usize,
        );
    }
}

// ============================================================================
// §4. FP-chain walk for prompt lookup
// ============================================================================

/// Walks the FP chain from `from_fp` upward, looking for the topmost
/// body frame — the frame whose saved return address lies inside the
/// code range of `beagle.core/__reset__`. That frame is immediately
/// above the reset frame in the call chain; its FP is the last live
/// frame of the reset body, and its saved FP points to the reset
/// frame itself.
///
/// Returns `Some((outermost_body_fp, reset_fp))` on success. Returns
/// None if the walk terminates (reaches the stack base, hits a 0 FP,
/// or detects a cycle / out-of-order chain) without finding a reset
/// frame.
unsafe fn find_enclosing_reset_frame(from_fp: usize) -> Option<(usize, usize)> {
    let mut fp = from_fp;
    // Defensive iteration cap — no realistic stack has this many frames.
    for _ in 0..100_000 {
        if fp == 0 {
            return None;
        }
        let saved_lr = unsafe { *((fp + 8) as *const usize) };
        if is_pc_in_reset_function(saved_lr) {
            let reset_fp = unsafe { *(fp as *const usize) };
            return Some((fp, reset_fp));
        }
        let saved_fp = unsafe { *(fp as *const usize) };
        if saved_fp == 0 || saved_fp <= fp {
            return None;
        }
        fp = saved_fp;
    }
    None
}

// ============================================================================
// §5. Capture
// ============================================================================

/// Runtime entry point called by `shift`'s generated IR. Captures the
/// frames between the current SP and the enclosing `__reset__` frame
/// into a freshly-allocated ContinuationObject and returns the tagged
/// heap pointer.
pub unsafe extern "C" fn capture_continuation_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    resume_address: usize,
    result_local_offset: isize,
) -> usize {
    unsafe {
        capture_continuation_runtime_inner(
            stack_pointer,
            frame_pointer,
            resume_address,
            result_local_offset,
        )
    }
}

/// Variant with a trailing `saved_regs_ptr` argument for the
/// `beagle.builtin/capture-continuation` builtin calling convention.
/// The saved-regs pointer is no longer used — callee-saved registers
/// are spilled to root slots by the codegen and reloaded at the
/// resume point — but the signature is preserved because the
/// generated IR still passes a (null) pointer.
pub unsafe extern "C" fn capture_continuation_runtime_with_saved_regs(
    stack_pointer: usize,
    frame_pointer: usize,
    resume_address: usize,
    result_local_offset: isize,
    _saved_regs_ptr: *const usize,
) -> usize {
    unsafe {
        capture_continuation_runtime_inner(
            stack_pointer,
            frame_pointer,
            resume_address,
            result_local_offset,
        )
    }
}

pub unsafe fn capture_continuation_runtime_inner(
    stack_pointer: usize,
    frame_pointer: usize,
    resume_address: usize,
    result_local_offset: isize,
) -> usize {
    crate::save_gc_context!(stack_pointer, frame_pointer);

    // Walk the FP chain to find the prompt boundary — the outermost
    // captured body frame is the topmost frame whose saved return
    // address points into `__reset__`. That frame was called directly
    // by `__reset__` (i.e., body_thunk). The captured bytes run from
    // the current SP (innermost) up through the outermost body
    // frame's saved FP+LR pair.
    let (outermost_body_fp, _reset_fp) = unsafe { find_enclosing_reset_frame(frame_pointer) }
        .unwrap_or_else(|| {
            panic!(
                "capture_continuation: shift without an enclosing reset. Walked FP chain \
                 from fp={:#x} and found no __reset__ frame. This is a compiler bug — \
                 shift must be inside reset.",
                frame_pointer
            )
        });

    // Captured byte range: from the current SP up to and including
    // the saved FP+LR pair of the outermost body frame.
    let capture_top = outermost_body_fp + 16;
    let stack_size = capture_top.saturating_sub(stack_pointer);

    // `segment_frame_pointer_offset` = offset from capture base to the
    // INNERMOST frame's FP. This is the canonical "resume FP" — where
    // execution picks up when the continuation is invoked — and also
    // where the saved-FP-chain relocation walk starts, since the
    // chain goes from innermost upward to outermost.
    //
    // `segment_gc_frame_offset` is repurposed to store the OUTERMOST
    // frame offset. Invoke needs this to place the copy at exactly
    // the right destination so that the outermost frame's saved
    // FP/LR slots overlay the trampoline's own saved caller FP/LR
    // slots — making normal body return flow back to the invoker
    // automatic with no slot patching.
    let innermost_fp_offset = frame_pointer - stack_pointer;
    let outermost_fp_offset = outermost_body_fp - stack_pointer;

    let runtime = crate::get_runtime().get_mut();

    // Allocate the continuation heap object.
    let cont_ptr = match runtime.allocate(
        ContinuationObject::NUM_FIELDS + 3, // +3 word header slack, matches historical allocation
        stack_pointer,
        BuiltInTypes::HeapObject,
    ) {
        Ok(ptr) => ptr,
        Err(_) => unsafe {
            crate::builtins::throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate continuation object - out of memory".to_string(),
            );
        },
    };

    let mut cont_obj = HeapObject::from_tagged(cont_ptr);
    ContinuationObject::initialize(&mut cont_obj, resume_address, result_local_offset);

    // Root the continuation across the segment allocation so GC
    // running during the segment allocate doesn't collect it.
    crate::runtime::per_thread_data()
        .saved_continuation_ptrs
        .push(cont_ptr);

    let segment_words = (stack_size + 7) / 8;
    let mut segment_data_base_at_capture = 0usize;
    let segment_heap_ptr = if segment_words > 0 {
        match runtime.allocate(segment_words, stack_pointer, BuiltInTypes::HeapObject) {
            Ok(ptr) => {
                let seg_obj = HeapObject::from_tagged(ptr);
                let header_ptr = seg_obj.untagged() as *mut usize;
                let mut header_val = unsafe { *header_ptr };
                header_val |= 0x2; // opaque bit — GC should not scan raw bytes as ptrs
                unsafe {
                    *header_ptr = header_val;
                }

                let data_ptr = seg_obj.untagged() + seg_obj.header_size();
                segment_data_base_at_capture = data_ptr;

                // Copy the live frames [stack_pointer, capture_top)
                // into the heap object. The live frames remain in
                // place on the main stack; this is a snapshot for
                // later resume.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        stack_pointer as *const u8,
                        data_ptr as *mut u8,
                        stack_size,
                    );
                }

                // Relocate saved-FP links inside the copied bytes
                // from their original stack addresses to the heap
                // object addresses.
                relocate_segment_caller_fp_links(
                    data_ptr,
                    stack_size,
                    innermost_fp_offset,
                    stack_pointer,
                );
                let _ = rebuild_segment_gc_prev_links_from_caller_chain(
                    data_ptr,
                    stack_size,
                    innermost_fp_offset,
                    0,
                );

                ptr
            }
            Err(_) => unsafe {
                crate::builtins::throw_runtime_error(
                    stack_pointer,
                    "AllocationError",
                    "Failed to allocate captured segment - out of memory".to_string(),
                );
            },
        }
    } else {
        BuiltInTypes::null_value() as usize
    };

    // Re-read the cont pointer in case GC moved it during segment allocation.
    let cont_ptr = crate::runtime::per_thread_data()
        .saved_continuation_ptrs
        .last()
        .copied()
        .unwrap_or(cont_ptr);

    let mut cont = ContinuationObject::from_tagged(cont_ptr).unwrap();
    cont.set_segment_ptr_with_barrier(runtime, segment_heap_ptr);
    cont.set_segment_frame_pointer_offset(innermost_fp_offset);
    cont.set_segment_gc_frame_offset(outermost_fp_offset);
    cont.set_segment_size(stack_size);
    if segment_heap_ptr != BuiltInTypes::null_value() as usize {
        cont.set_segment_original_data_base(segment_data_base_at_capture);
    }

    let cont_ptr = crate::runtime::per_thread_data()
        .saved_continuation_ptrs
        .last()
        .copied()
        .unwrap_or(cont_ptr);

    // Drop the temporary root.
    let _ = crate::runtime::per_thread_data()
        .saved_continuation_ptrs
        .pop();

    cont_ptr
}

// ============================================================================
// §6. Return from shift
// ============================================================================

/// Runtime entry point called by a shift body's generated IR at its
/// completion point. Longjmps back to `__reset__`'s caller with
/// `value` as `__reset__`'s return value.
pub unsafe extern "C" fn return_from_shift_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
    _cont_ptr: usize,
) -> ! {
    unsafe { return_from_shift_runtime_inner(stack_pointer, frame_pointer, value) }
}

pub unsafe fn return_from_shift_runtime_inner(
    _stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> ! {
    // Walk the FP chain to locate the enclosing `__reset__` frame.
    // The topmost body frame is the one whose saved LR points into
    // `__reset__`; `__reset__`'s own FP is topmost_body_fp's saved FP.
    let (_topmost_body_fp, reset_fp) = unsafe { find_enclosing_reset_frame(frame_pointer) }
        .unwrap_or_else(|| {
            panic!(
                "return_from_shift: no enclosing reset found walking FP chain from fp={:#x}",
                frame_pointer
            )
        });

    // Derive the longjmp target: simulate `__reset__` returning
    // normally to its caller with `value` as the return register.
    //
    //   new_fp = __reset__'s saved caller FP       = [reset_fp + 0]
    //   new_ra = __reset__'s saved return address  = [reset_fp + 8]
    //   new_sp = post-__reset__-return SP          = reset_fp + 16
    //
    // Nothing is snapshotted in advance. These reads happen now
    // because `__reset__`'s frame is still live below shift body's
    // execution until the jump fires.
    let new_fp = unsafe { *(reset_fp as *const usize) };
    let new_ra = unsafe { *((reset_fp + 8) as *const usize) };
    let new_sp = reset_fp + 16;

    // ARM64: use the return-jump JIT trampoline. It takes
    // (new_sp, new_fp, new_lr, jump_target, callee_saved_ptr, value).
    // `value` is written into X0 before the branch, making it the
    // return value of `__reset__`. We want control to land AT
    // `new_ra` with sp/fp restored, as if `__reset__` had executed
    // its epilogue, so we pass `new_ra` as the jump target. The
    // `new_lr` arg is unused — the handler doesn't need its own
    // caller's LR because it's just resuming a normal return.
    let runtime = crate::get_runtime().get();
    let return_jump_fn = runtime
        .get_function_by_name("beagle.builtin/return-jump")
        .expect("return-jump function not found");
    let ptr: *const u8 = return_jump_fn.pointer.into();
    let return_jump: extern "C" fn(usize, usize, usize, usize, *const usize, usize) -> ! =
        unsafe { std::mem::transmute(ptr) };
    return_jump(new_sp, new_fp, 0, new_ra, std::ptr::null(), value);
}

// ============================================================================
// §7. Invocation trampoline
// ============================================================================

/// Invoke a captured continuation.
///
/// Called as the body of a cont-closure: `k(v)` where `k` is the
/// closure returned by shift. The closure's single free variable is
/// the ContinuationObject; `value` is the value being plugged into
/// the shift point's result slot.
///
/// The design is:
/// 1. Compute a destination address `dst` on this trampoline's own
///    stack such that the outermost captured frame's FP slot lands
///    exactly at the trampoline's own FP. The trampoline's prologue
///    already wrote the invoker's FP at `[trampoline_fp + 0]` and
///    the invoker's LR at `[trampoline_fp + 8]`. By overlaying the
///    outermost captured frame on those slots, the resumed body's
///    eventual "return past bottom" unwinds directly into the
///    invoker — no return stubs, no prompt pushes, no side tables.
/// 2. Copy the captured bytes (minus the bottom 16) from the heap
///    segment into `[dst, dst + copy_size)`.
/// 3. Walk the copy and relocate saved-FP links from heap-addresses
///    to stack-addresses by a constant delta.
/// 4. Rebuild the GC prev chain in the copy so GC can walk it as
///    normal stack frames.
/// 5. Write `value` into the resume point's result slot.
/// 6. Use `return-jump` to set SP/FP and branch to resume_address.
///
/// Multi-shot is automatic: the heap segment's bytes are never
/// mutated, so each invocation copies fresh bytes to a fresh
/// destination.
///
/// Safety: All writes to `dst` happen while Rust is mid-function.
/// The `dst` region overlaps Rust's own local area but Rust has
/// already packed every value it needs into local variables (which
/// the compiler keeps in registers for the remaining straight-line
/// code). Between the first dst write and the final `return_jump`
/// call, no helper function is invoked — so no frame gets pushed
/// below Rust's SP into a region that might overlap dst. Once
/// `return_jump` runs, SP is set to `dst` and control transfers to
/// the resumed body; the old Rust frame is abandoned.
#[allow(unused_variables)]
pub unsafe extern "C" fn continuation_trampoline(closure_ptr: usize, value: usize) -> ! {
    // Extract the continuation from the closure's free variables.
    // Closure layout: header(8) + fn_ptr(8) + num_free(8) + num_locals(8) + free_var[0]...
    let untagged_closure = BuiltInTypes::untag(closure_ptr);
    let cont_ptr = unsafe { *((untagged_closure + 32) as *const usize) };
    let cont = ContinuationObject::from_tagged(cont_ptr)
        .expect("continuation_trampoline: closure free var is not a ContinuationObject");

    // Normalize segment — this handles GC moves by re-relocating the
    // in-segment saved-FP chain against the current heap data base.
    let (seg_base, _seg_top, _innermost_fp_heap) = cont
        .segment_frame_info()
        .expect("continuation_trampoline: continuation has no segment data");
    let seg_size = cont.segment_size();
    let innermost_offset = cont.segment_frame_pointer_offset();
    let outermost_offset = cont.segment_gc_frame_offset();
    let resume_address = cont.resume_address();
    let result_local_offset = cont.result_local();

    // Read the trampoline's own FP via `read-fp` — a 2-instruction
    // Beagle-side JIT trampoline that returns x29. This function call
    // pushes a frame below trampoline's SP; that frame is popped
    // before we return here, so it doesn't touch any memory we care
    // about.
    let trampoline_fp = {
        let runtime = crate::get_runtime().get();
        let fn_entry = runtime
            .get_function_by_name("beagle.builtin/read-fp")
            .expect("read-fp trampoline not found");
        let read_fp: extern "C" fn() -> usize =
            unsafe { std::mem::transmute::<_, _>(fn_entry.pointer) };
        read_fp()
    };

    // Read return-jump's pointer here, before we start writing dst,
    // so that the final call doesn't need any function lookups.
    let return_jump_ptr: *const u8 = {
        let runtime = crate::get_runtime().get();
        let fn_entry = runtime
            .get_function_by_name("beagle.builtin/return-jump")
            .expect("return-jump trampoline not found");
        fn_entry.pointer.into()
    };

    // Destination placement: outermost_fp_in_dst must equal trampoline_fp.
    //   outermost_fp_in_dst = dst + outermost_offset
    //   => dst = trampoline_fp - outermost_offset
    //
    // copy_size = outermost_offset. We do NOT copy the bottom 16
    // bytes (the outermost frame's saved FP+LR pair). Those slots at
    // [trampoline_fp, trampoline_fp+16) already hold the invoker's
    // FP and LR (written by the trampoline's own prologue), which is
    // exactly what the outermost frame's saved FP/LR need to be.
    let dst = trampoline_fp - outermost_offset;
    let copy_size = outermost_offset;

    // ------------------------------------------------------------------
    // From here until the final `return_jump` call, NO function calls
    // may happen. All writes go to `dst` (which may overlap Rust's
    // own frame); a function call would push a frame below
    // trampoline's SP, into a region of `dst` we're actively writing.
    // ------------------------------------------------------------------

    // 1. Copy bytes: [seg_base, seg_base + copy_size) → [dst, dst + copy_size).
    let mut i = 0usize;
    while i < copy_size {
        let src_word = unsafe { *((seg_base + i) as *const usize) };
        unsafe { *((dst + i) as *mut usize) = src_word };
        i += 8;
    }

    // 2. Relocate saved-FP chain in dst. Walk from innermost upward
    //    via saved_fp, adjusting each by delta until a saved_fp falls
    //    outside the source range.
    let delta = dst.wrapping_sub(seg_base);
    let mut fp = dst + innermost_offset;
    loop {
        let saved_slot = fp as *mut usize;
        let saved = unsafe { *saved_slot };
        if saved < seg_base || saved >= seg_base + seg_size {
            break;
        }
        let relocated = saved.wrapping_add(delta);
        unsafe { *saved_slot = relocated };
        fp = relocated;
    }

    // 3. Rebuild GC prev chain in dst. For each frame header (at
    //    [fp-8]), write to [header-8] (the GC prev slot) the address
    //    of the parent frame's header, or 0 if the parent is outside
    //    the copy range (i.e., this is the outermost frame, which
    //    logically chains into the invoker — GC will pick that up
    //    naturally via GC_FRAME_TOP once we update it below).
    //
    //    Range check uses the header address (fp - 8), not fp,
    //    because the outermost frame's fp equals dst + copy_size and
    //    is therefore not strictly less than dst + copy_size. Its
    //    header at dst + copy_size - 8 IS within range.
    let copy_end = dst + copy_size;
    let mut fp = dst + innermost_offset;
    loop {
        let header_addr = fp.wrapping_sub(8);
        if header_addr < dst || header_addr >= copy_end {
            break;
        }
        let saved_fp = unsafe { *(fp as *const usize) };
        let parent_header = saved_fp.wrapping_sub(8);
        let parent_in_range = parent_header >= dst && parent_header < copy_end;
        let prev_val = if parent_in_range { parent_header } else { 0 };
        unsafe { *((header_addr - 8) as *mut usize) = prev_val };
        if !parent_in_range {
            break;
        }
        fp = saved_fp;
    }

    // 4. Write value to the resume point's result slot.
    let innermost_fp_in_dst = dst + innermost_offset;
    if result_local_offset != 0 {
        let result_ptr =
            (innermost_fp_in_dst as isize).wrapping_add(result_local_offset) as *mut usize;
        unsafe { *result_ptr = value };
    }

    // 5. Update GC_FRAME_TOP to the innermost frame's header in dst.
    //    This is a simple thread-local cell write — no function
    //    call. (We can't call set_gc_frame_top because that's a Rust
    //    function.)
    GC_FRAME_TOP.with(|cell| cell.set(innermost_fp_in_dst - 8));

    // 6. Final jump via return-jump. Signature:
    //    return_jump(new_sp, new_fp, new_lr, jump_target, callee_saved_ptr, value)
    //    It sets sp/fp/lr/x0 and branches. noreturn.
    let return_jump: extern "C" fn(usize, usize, usize, usize, *const usize, usize) -> ! =
        unsafe { std::mem::transmute(return_jump_ptr) };
    return_jump(
        dst,
        innermost_fp_in_dst,
        0,
        resume_address,
        std::ptr::null(),
        0,
    );
}
