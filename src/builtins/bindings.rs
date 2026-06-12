use super::*;
use crate::save_gc_context;

// Thread-local counter of active continuation marks.
// When zero, get_dynamic_var skips the frame walk entirely (fast path).
thread_local! {
    static ACTIVE_MARKS_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Latched once: BEAGLE_DEBUG_MARKS=1 traces mark install/uninstall and
/// reads that resolve through a mark. Diagnostic for dynamic-binding
/// leaks across unwinds (the `binding` + throw class of bug).
fn debug_marks() -> bool {
    static DEBUG_MARKS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DEBUG_MARKS.get_or_init(|| std::env::var("BEAGLE_DEBUG_MARKS").is_ok())
}

/// Current value of the thread's active-marks counter. Snapshotted by
/// `push_exception_handler` so a throw can roll marks back to the
/// handler's level.
pub fn current_marks_count() -> usize {
    ACTIVE_MARKS_COUNT.with(|c| c.get())
}

/// Reset the active-marks counter to a snapshotted value. Used by
/// `throw_exception` when unwinding: marks installed between handler
/// push and the throw belong to abandoned extents.
pub fn set_marks_count(count: usize) {
    ACTIVE_MARKS_COUNT.with(|c| c.set(count));
}

/// Add to the active-marks counter. Called by `continuation_trampoline`
/// when a restored segment re-introduces frames that carry continuation
/// marks: the marks are physically back on the stack (the segment bytes
/// include the mark locals and frame-header flags), so the fast-path
/// gate in `get_dynamic_var` must see a non-zero count or the restored
/// marks would be invisible.
pub fn marks_count_add(n: usize) {
    if n > 0 {
        ACTIVE_MARKS_COUNT.with(|c| c.set(c.get() + n));
    }
}

/// Subtract from the active-marks counter. Called by continuation
/// capture: the captured frames are about to be abandoned (longjmp to
/// the prompt / jump to the catch), taking their marks off the live
/// stack. The exception path additionally resets the count absolutely
/// to the handler's push-time snapshot, which subsumes this.
pub fn marks_count_sub(n: usize) {
    if n > 0 {
        ACTIVE_MARKS_COUNT.with(|c| c.set(c.get().saturating_sub(n)));
    }
}

/// Count the continuation-mark entries carried by the live frames whose
/// frame pointers lie in `[innermost_fp, capture_top)` — exactly the
/// frames a continuation capture is about to snapshot.
///
/// Walks the GC FRAME CHAIN (like `get_dynamic_var`), NOT the raw
/// saved-FP chain: the FP chain passes through Rust builtin frames,
/// whose `[fp - 8]` is arbitrary stack data — a random bit pattern there
/// can masquerade as a marks flag and a junk "mark pointer" can pass the
/// heap-tag check while being misaligned (observed as a flaky
/// `Misaligned heap pointer` abort during throw-captures under the
/// smoke fuzz). The GC chain links only real Beagle frame headers.
///
/// # Safety
/// Must run on the capturing thread, before the GC frame chain is
/// unwound for the throw/shift, with `innermost_fp`/`capture_top`
/// bounding the capture region on this thread's stack.
pub unsafe fn count_marks_in_frame_range(innermost_fp: usize, capture_top: usize) -> usize {
    let mut total = 0;
    let mut header_addr = get_gc_frame_top();
    // Defensive cap mirrors get_dynamic_var's implicit trust with a bound.
    for _ in 0..100_000 {
        if header_addr == 0 {
            break;
        }
        let fp = header_addr.wrapping_add(8);
        if fp >= capture_top {
            // GC chain is innermost→outermost ascending; everything past
            // the capture top is outside the captured segment.
            break;
        }
        unsafe {
            if fp >= innermost_fp {
                let header_word = *(header_addr as *const usize);
                let header = crate::types::Header::from_usize(header_word);
                if header.type_flags & crate::collections::FRAME_HAS_MARKS_FLAG != 0 {
                    let idx = (header.type_data & 0xFFFF) as usize;
                    let mut entry_ptr =
                        *((fp.wrapping_sub(24).wrapping_sub(idx * 8)) as *const usize);
                    while entry_ptr != BuiltInTypes::null_value() as usize
                        && entry_ptr != 0
                        && BuiltInTypes::is_heap_pointer(entry_ptr)
                    {
                        total += 1;
                        let entry = crate::types::HeapObject::from_tagged(entry_ptr);
                        entry_ptr = entry.get_field(2);
                    }
                }
            }
            header_addr = *((header_addr.wrapping_sub(8)) as *const usize);
        }
    }
    total
}

/// Snapshot the continuation-mark chain of a single frame: returns
/// `Some(chain_head)` if the frame's FRAME_HAS_MARKS_FLAG is set,
/// `None` otherwise. Called at exception-handler push.
///
/// # Safety
/// `frame_pointer` must be the frame pointer of a live Beagle frame.
pub unsafe fn snapshot_frame_marks(frame_pointer: usize) -> Option<usize> {
    unsafe {
        let header_addr = frame_pointer.wrapping_sub(8);
        let header = crate::types::Header::from_usize(*(header_addr as *const usize));
        if header.type_flags & crate::collections::FRAME_HAS_MARKS_FLAG == 0 {
            return None;
        }
        let mark_local_index = (header.type_data & 0xFFFF) as usize;
        let mark_ptr_addr = frame_pointer
            .wrapping_sub(24)
            .wrapping_sub(mark_local_index * 8);
        Some(*(mark_ptr_addr as *const usize))
    }
}

/// Restore a frame's continuation-mark chain to a snapshot taken by
/// `snapshot_frame_marks`. Called by `throw_exception` on the handler's
/// frame before jumping to the catch block: marks installed inside the
/// try body live in a hoisted local of this same frame (it stays live
/// across the unwind), so without this they would remain visible to
/// `get_dynamic_var` forever.
///
/// Install/uninstall pairs are lexically nested, so at throw time the
/// frame's chain is exactly the snapshot chain plus zero or more newer
/// entries — truncating back to the snapshot is always sound.
///
/// # Safety
/// `frame_pointer` must be the frame pointer of a live Beagle frame —
/// the handler's own frame, which the throw is about to jump into.
pub unsafe fn restore_frame_marks(frame_pointer: usize, snapshot: Option<usize>) {
    unsafe {
        let header_addr = frame_pointer.wrapping_sub(8);
        let header_word = *(header_addr as *const usize);
        let mut header = crate::types::Header::from_usize(header_word);
        if header.type_flags & crate::collections::FRAME_HAS_MARKS_FLAG == 0 {
            // No marks now — nothing was installed (or everything was
            // balanced); the snapshot must have been None.
            return;
        }
        let mark_local_index = (header.type_data & 0xFFFF) as usize;
        let mark_ptr_addr = frame_pointer
            .wrapping_sub(24)
            .wrapping_sub(mark_local_index * 8);
        match snapshot {
            Some(head) => {
                *(mark_ptr_addr as *mut usize) = head;
            }
            None => {
                *(mark_ptr_addr as *mut usize) = BuiltInTypes::null_value() as usize;
                header.type_flags &= !crate::collections::FRAME_HAS_MARKS_FLAG;
                *(header_addr as *mut usize) = header.to_usize();
            }
        }
    }
}

/// Get the current value of a dynamic variable.
/// If any continuation marks are active, walks the GC frame chain looking for marks.
/// Otherwise falls back directly to the root namespace value.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe extern "C" fn get_dynamic_var(namespace_id: usize, slot: usize) -> usize {
    let ns_id = BuiltInTypes::untag(namespace_id);
    let slot_val = BuiltInTypes::untag(slot);

    // Fast path: no marks active anywhere, skip frame walk
    let marks_active = ACTIVE_MARKS_COUNT.with(|c| c.get());
    if marks_active > 0 {
        let key = (ns_id << 16) | slot_val;

        // Walk the GC frame chain looking for continuation marks.
        let mut header_addr = get_gc_frame_top();
        #[cfg(feature = "debug-gc")]
        let mut fast = header_addr;
        while header_addr != 0 {
            let header_word = unsafe { *(header_addr as *const usize) };
            let header = crate::types::Header::from_usize(header_word);

            if header.type_flags & crate::collections::FRAME_HAS_MARKS_FLAG != 0 {
                let mark_local_index = (header.type_data & 0xFFFF) as usize;
                let mark_ptr_addr = header_addr
                    .wrapping_sub(16)
                    .wrapping_sub(mark_local_index * 8);
                let mut entry_ptr = unsafe { *(mark_ptr_addr as *const usize) };

                while entry_ptr != BuiltInTypes::null_value() as usize && entry_ptr != 0 {
                    if BuiltInTypes::is_heap_pointer(entry_ptr) {
                        let entry = crate::types::HeapObject::from_tagged(entry_ptr);
                        if BuiltInTypes::untag(entry.get_field(0)) == key {
                            if debug_marks() {
                                eprintln!(
                                    "[marks] key={:#x} FOUND at frame header {:#x} (gc_frame_top={:#x}, active={})",
                                    key,
                                    header_addr,
                                    get_gc_frame_top(),
                                    marks_active
                                );
                            }
                            return entry.get_field(1);
                        }
                        entry_ptr = entry.get_field(2);
                    } else {
                        break;
                    }
                }
            }

            header_addr = unsafe { *((header_addr.wrapping_sub(8)) as *const usize) };

            #[cfg(feature = "debug-gc")]
            {
                for _ in 0..2 {
                    if fast != 0 {
                        fast = unsafe { *((fast.wrapping_sub(8)) as *const usize) };
                    }
                }
                if header_addr != 0 && header_addr == fast {
                    panic!(
                        "BUG: cycle in GC frame chain at {:#x} during get_dynamic_var — saved_gc_prev fix didn't cover this case",
                        header_addr
                    );
                }
            }
        }
    }

    // Fall back to root binding in namespace
    let runtime = get_runtime().get();
    runtime.get_namespace_binding(ns_id, slot_val)
}

/// Install a continuation mark in the caller's frame.
/// Allocates a MarkEntry heap object and chains it to the mark pointer local.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe extern "C" fn install_continuation_mark(
    stack_pointer: usize,
    frame_pointer: usize,
    namespace_id: usize,
    slot: usize,
    value: usize,
    mark_local_index: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let ns_id = BuiltInTypes::untag(namespace_id);
        let slot_val = BuiltInTypes::untag(slot);
        let mark_local_idx = BuiltInTypes::untag(mark_local_index);
        let key = BuiltInTypes::Int.tag(((ns_id << 16) | slot_val) as isize) as usize;

        // Read current mark pointer from the caller's frame
        let mark_ptr_addr = frame_pointer
            .wrapping_sub(24)
            .wrapping_sub(mark_local_idx * 8);
        let old_mark_ptr = *(mark_ptr_addr as *const usize);

        // Allocate a MarkEntry heap object (3 fields: key, value, next)
        let runtime = get_runtime().get_mut();
        let entry_ptr = match runtime.allocate(3, stack_pointer, BuiltInTypes::HeapObject) {
            Ok(ptr) => ptr,
            Err(_) => panic!("Failed to allocate MarkEntry"),
        };
        let mut entry = crate::types::HeapObject::from_tagged(entry_ptr);
        entry.writer_header_direct(crate::types::Header {
            type_id: crate::collections::TYPE_ID_MARK_ENTRY,
            type_data: 0,
            size: 3,
            opaque: false,
            marked: false,
            large: false,
            type_flags: 0,
        });
        entry.write_field(0, key);
        entry.write_field(1, value);
        entry.write_field(2, old_mark_ptr);

        // Store new entry as the mark pointer
        *(mark_ptr_addr as *mut usize) = entry_ptr;

        if debug_marks() {
            eprintln!(
                "[marks] INSTALL key={:#x} fp={:#x} header={:#x} local_idx={} old_mark={:#x}",
                key,
                frame_pointer,
                frame_pointer.wrapping_sub(8),
                mark_local_idx,
                old_mark_ptr
            );
        }

        // Increment active marks counter so get_dynamic_var knows to walk frames
        ACTIVE_MARKS_COUNT.with(|c| c.set(c.get() + 1));

        // Set the mark flag in the frame header if not already set
        let header_addr = frame_pointer.wrapping_sub(8);
        let header_word = *(header_addr as *const usize);
        let mut header = crate::types::Header::from_usize(header_word);
        if header.type_flags & crate::collections::FRAME_HAS_MARKS_FLAG == 0 {
            header.type_flags |= crate::collections::FRAME_HAS_MARKS_FLAG;
            header.type_data = (header.type_data & 0xFFFF0000) | (mark_local_idx as u32 & 0xFFFF);
            *(header_addr as *mut usize) = header.to_usize();
        }

        0b111 // null
    }
}

/// Uninstall the top continuation mark from the caller's frame.
pub unsafe extern "C" fn uninstall_continuation_mark(
    _stack_pointer: usize,
    frame_pointer: usize,
    mark_local_index: usize,
) -> usize {
    unsafe {
        let mark_local_idx = BuiltInTypes::untag(mark_local_index);

        // Read current mark pointer
        let mark_ptr_addr = frame_pointer
            .wrapping_sub(24)
            .wrapping_sub(mark_local_idx * 8);
        let mark_ptr = *(mark_ptr_addr as *const usize);

        if debug_marks() {
            eprintln!(
                "[marks] UNINSTALL fp={:#x} header={:#x} local_idx={} mark_ptr={:#x}",
                frame_pointer,
                frame_pointer.wrapping_sub(8),
                mark_local_idx,
                mark_ptr
            );
        }
        if mark_ptr != BuiltInTypes::null_value() as usize
            && mark_ptr != 0
            && BuiltInTypes::is_heap_pointer(mark_ptr)
        {
            let entry = crate::types::HeapObject::from_tagged(mark_ptr);
            let next = entry.get_field(2);
            *(mark_ptr_addr as *mut usize) = next;

            // Decrement active marks counter
            ACTIVE_MARKS_COUNT.with(|c| c.set(c.get().saturating_sub(1)));

            // If no more marks in this frame, clear the flag
            if next == BuiltInTypes::null_value() as usize || next == 0 {
                let header_addr = frame_pointer.wrapping_sub(8);
                let header_word = *(header_addr as *const usize);
                let mut header = crate::types::Header::from_usize(header_word);
                header.type_flags &= !crate::collections::FRAME_HAS_MARKS_FLAG;
                *(header_addr as *mut usize) = header.to_usize();
            }
        }

        0b111 // null
    }
}

pub unsafe extern "C" fn update_binding(
    _stack_pointer: usize,
    _frame_pointer: usize,
    namespace_slot: usize,
    value: usize,
) -> usize {
    print_call_builtin(get_runtime().get(), "update_binding");
    let runtime = get_runtime().get_mut();
    let namespace_slot = BuiltInTypes::untag(namespace_slot);
    let namespace_id = runtime.current_namespace_id();

    // Cells are the authoritative store. Writing one is a single 8-byte
    // store into a stable mmap region — no allocation, no GC trigger,
    // so we don't need to save the GC context or root the value.
    runtime.update_binding(namespace_id, namespace_slot, value);

    BuiltInTypes::null_value() as usize
}

/// Store a function object in a namespace binding.
/// Takes a namespace slot and a Function-tagged pointer.
/// Creates a proper function object (with name, arity) and stores it.
pub unsafe extern "C" fn store_function_binding(
    stack_pointer: usize,
    frame_pointer: usize,
    namespace_slot: usize,
    function: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let namespace_slot = BuiltInTypes::untag(namespace_slot);
    let namespace_id = runtime.current_namespace_id();

    // Extract the raw function pointer
    let fn_ptr = BuiltInTypes::untag(function) as *const u8;

    // Look up function metadata for name and arity
    let (name, arity) = if let Some(func) = runtime.get_function_by_pointer(fn_ptr) {
        (func.name.clone(), func.number_of_args)
    } else {
        ("<unknown>".to_string(), 0)
    };

    // `create_function_value` allocates; the cell scan during a GC
    // triggered by that allocation will see the slot's previous value
    // (whatever is currently there), which is fine — we just haven't
    // installed the new function yet. Once the value is back, a single
    // cell store publishes it.
    let fn_obj = runtime
        .create_function_value(stack_pointer, fn_ptr, &name, arity)
        .expect("Failed to create function value");

    runtime.update_binding(namespace_id, namespace_slot, fn_obj);

    BuiltInTypes::null_value() as usize
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_binding(namespace: usize, slot: usize) -> usize {
    // The JIT compiles namespace-variable reads to a direct load from
    // the binding's cell address, so this builtin is no longer on the
    // hot path. It still serves a few internal callers (reflection,
    // type lookups), so it remains for compatibility — but it just
    // forwards to the Runtime's cell read.
    let runtime = get_runtime().get();
    let namespace = BuiltInTypes::untag(namespace);
    let slot = BuiltInTypes::untag(slot);
    runtime.get_binding(namespace, slot)
}

pub unsafe extern "C" fn set_current_namespace(namespace: usize) -> usize {
    let runtime = get_runtime().get_mut();
    runtime.set_current_namespace(namespace);
    BuiltInTypes::null_value() as usize
}

pub extern "C" fn get_current_namespace(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let name = runtime.current_namespace_name();
    match runtime.allocate_string(stack_pointer, name) {
        Ok(ptr) => ptr.into(),
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate namespace name string".to_string(),
            );
        },
    }
}
