use super::*;
use crate::save_gc_context;

// Thread-local counter of active continuation marks.
// When zero, get_dynamic_var skips the frame walk entirely (fast path).
thread_local! {
    static ACTIVE_MARKS_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
    stack_pointer: usize,
    frame_pointer: usize,
    namespace_slot: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "update_binding");
    let runtime = get_runtime().get_mut();
    let namespace_slot = BuiltInTypes::untag(namespace_slot);
    let namespace_id = runtime.current_namespace_id();

    // Root the value so GC can update it if objects move during allocation
    let value_root_id = runtime.register_temporary_root(value);

    // Re-read from the root before the map update so we never pass a stale
    // pre-GC pointer into the heap binding machinery.
    let value = runtime.peek_temporary_root(value_root_id);

    // Store binding in heap-based PersistentMap (no namespace_roots tracking needed!)
    if let Err(e) = runtime.set_heap_binding(stack_pointer, namespace_id, namespace_slot, value) {
        eprintln!("Error in update_binding: {}", e);
    }

    // Re-read value from root in case GC moved it
    let value = runtime.unregister_temporary_root(value_root_id);

    // Also update the Rust-side HashMap for backwards compatibility during migration
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

    // Create a function object
    let fn_obj = runtime
        .create_function_value(stack_pointer, fn_ptr, &name, arity)
        .expect("Failed to create function value");

    // Root the value so GC can update it if objects move during allocation
    let value_root_id = runtime.register_temporary_root(fn_obj);

    // Re-read from the root before the map update so we never pass a stale
    // pre-GC pointer into the heap binding machinery.
    let fn_obj = runtime.peek_temporary_root(value_root_id);

    // Store binding in heap-based PersistentMap
    if let Err(e) = runtime.set_heap_binding(stack_pointer, namespace_id, namespace_slot, fn_obj) {
        eprintln!("Error in store_function_binding: {}", e);
    }

    // Re-read value from root in case GC moved it
    let fn_obj = runtime.unregister_temporary_root(value_root_id);

    // Also update the Rust-side HashMap
    runtime.update_binding(namespace_id, namespace_slot, fn_obj);

    BuiltInTypes::null_value() as usize
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_binding(namespace: usize, slot: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let namespace = BuiltInTypes::untag(namespace);
    let slot = BuiltInTypes::untag(slot);

    // TODO: Flush pending bindings at a safer point (not during get_binding)
    // For now, rely on HashMap fallback for bindings added by compiler thread
    // let stack_pointer = get_current_stack_pointer();
    // runtime.flush_pending_heap_bindings(stack_pointer);

    // Try heap-based PersistentMap first
    let result = runtime.get_heap_binding(namespace, slot);
    if result != BuiltInTypes::null_value() as usize {
        return result;
    }

    // Fall back to Rust-side HashMap during migration
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
