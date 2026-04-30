//! Mutable string builder backed by a packed byte storage.
//!
//! # Layout
//!
//! `StringBuilder` (TYPE_ID_STRING_BUILDER) is a 2-field heap object:
//! - field 0: tagged pointer to a TYPE_ID_BYTE_STORAGE (traced by GC)
//! - field 1: tagged int — current length in bytes
//!
//! `ByteStorage` (TYPE_ID_BYTE_STORAGE) is an opaque heap object:
//! - header.type_data = capacity in bytes
//! - body: capacity bytes packed into ceil(capacity/8) words after the header
//!
//! Two-object dance: when the buffer fills, we allocate a new `ByteStorage`
//! at `max(2 * capacity, len + needed)`, memcpy old → new, and rewrite
//! `sb.field[0]` with a write barrier. The `StringBuilder` itself never
//! moves (its identity is stable), only its storage pointer changes.

use std::error::Error;

use crate::collections::{HandleScope, TYPE_ID_BYTE_STORAGE, TYPE_ID_STRING, TYPE_ID_STRING_BUILDER};
use crate::types::{BuiltInTypes, Header, HeapObject};

use super::*;
use crate::save_gc_context;

const FIELD_STORAGE: usize = 0;
const FIELD_LEN: usize = 1;

const DEFAULT_CAPACITY: usize = 16;

/// Pointer to the first byte of a ByteStorage's packed payload.
/// For non-large storages: header(8) + payload.
/// For large storages: header(8) + extended_size(8) + payload.
#[inline]
fn storage_bytes_ptr(storage_tagged: usize) -> *mut u8 {
    let storage = HeapObject::from_tagged(storage_tagged);
    let untagged = storage.untagged() as *mut u8;
    let offset = storage.header_size();
    unsafe { untagged.add(offset) }
}

#[inline]
fn storage_capacity(storage_tagged: usize) -> usize {
    HeapObject::from_tagged(storage_tagged).get_header().type_data as usize
}

#[inline]
fn sb_storage(sb_tagged: usize) -> usize {
    HeapObject::from_tagged(sb_tagged).get_field(FIELD_STORAGE)
}

#[inline]
fn sb_len(sb_tagged: usize) -> usize {
    BuiltInTypes::untag(HeapObject::from_tagged(sb_tagged).get_field(FIELD_LEN))
}

#[inline]
fn write_len(sb_tagged: usize, len: usize) {
    let sb = HeapObject::from_tagged(sb_tagged);
    sb.write_field(
        FIELD_LEN as i32,
        BuiltInTypes::construct_int(len as isize) as usize,
    );
}

/// Allocate a new ByteStorage with the given byte capacity, zeroed.
/// Caller is responsible for rooting any other live values across this
/// call (this function does not allocate any other objects).
fn allocate_byte_storage(
    runtime: &mut crate::runtime::Runtime,
    stack_pointer: usize,
    byte_capacity: usize,
) -> Result<usize, Box<dyn Error>> {
    let words = byte_capacity.div_ceil(8).max(1);
    let pointer = runtime.allocate_zeroed(words, stack_pointer, BuiltInTypes::HeapObject)?;
    let mut heap_object = HeapObject::from_tagged(pointer);
    let is_large = words > Header::MAX_INLINE_SIZE;
    heap_object.writer_header_direct(Header {
        type_id: TYPE_ID_BYTE_STORAGE,
        type_data: byte_capacity as u32,
        size: if is_large { 0xFFFF } else { words as u16 },
        opaque: true,
        marked: false,
        large: is_large,
        type_flags: 0,
    });
    if is_large {
        let size_ptr = (heap_object.untagged() + 8) as *mut usize;
        unsafe { *size_ptr = words };
    }
    Ok(pointer)
}

/// Allocate a fresh StringBuilder with the given initial capacity.
fn allocate_string_builder(
    runtime: &mut crate::runtime::Runtime,
    stack_pointer: usize,
    capacity: usize,
) -> Result<usize, Box<dyn Error>> {
    let cap = capacity.max(1);

    let mut scope = HandleScope::new(runtime, stack_pointer);
    let storage_tagged = allocate_byte_storage(scope.runtime(), stack_pointer, cap)?;
    let storage_h = scope.alloc(storage_tagged);

    // Allocate the StringBuilder. Two non-opaque fields, GC traces field 0.
    let sb_h = scope.allocate_typed_zeroed(2, TYPE_ID_STRING_BUILDER)?;
    let sb = sb_h.to_gc_handle();

    sb.set_field_with_barrier(scope.runtime(), FIELD_STORAGE, storage_h.get());
    sb.set_field(FIELD_LEN, BuiltInTypes::construct_int(0) as usize);

    Ok(sb.as_tagged())
}

/// Make sure storage has room for `needed` total bytes (i.e. len after the
/// pending append). If a grow is necessary the StringBuilder may be moved
/// by GC — the returned tagged pointer is the (possibly new) sb. Callers
/// must use the returned value, not the original `sb_tagged`.
fn ensure_capacity_for(
    runtime: &mut crate::runtime::Runtime,
    stack_pointer: usize,
    sb_tagged: usize,
    needed: usize,
) -> Result<usize, Box<dyn Error>> {
    let storage_tagged = sb_storage(sb_tagged);
    let cap = storage_capacity(storage_tagged);
    if needed <= cap {
        return Ok(sb_tagged);
    }

    let new_cap = (cap.saturating_mul(2)).max(needed);
    let len = sb_len(sb_tagged);

    // Snapshot old bytes — survives the allocation that follows even if
    // a compacting GC moves the storage object.
    let mut snapshot = Vec::with_capacity(len);
    {
        let bytes = storage_bytes_ptr(storage_tagged);
        unsafe { snapshot.set_len(len) };
        unsafe { std::ptr::copy_nonoverlapping(bytes, snapshot.as_mut_ptr(), len) };
    }

    let mut scope = HandleScope::new(runtime, stack_pointer);
    let sb_h = scope.alloc(sb_tagged);

    let new_storage_tagged = allocate_byte_storage(scope.runtime(), stack_pointer, new_cap)?;

    // Re-read sb via handle (may have moved) and copy snapshot into the new storage.
    {
        let dst = storage_bytes_ptr(new_storage_tagged);
        unsafe { std::ptr::copy_nonoverlapping(snapshot.as_ptr(), dst, len) };
    }

    // Swap the storage pointer with a write barrier.
    let sb_gc = sb_h.to_gc_handle();
    sb_gc.set_field_with_barrier(scope.runtime(), FIELD_STORAGE, new_storage_tagged);

    // Return the (possibly relocated) sb pointer.
    Ok(sb_h.get())
}

// ---------------- Builtin entry points ----------------

pub extern "C" fn string_builder_new(
    stack_pointer: usize,
    frame_pointer: usize,
    capacity_tagged: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "string_builder_new");
    let runtime = get_runtime().get_mut();
    let cap = if BuiltInTypes::get_kind(capacity_tagged) == BuiltInTypes::Int {
        let raw = BuiltInTypes::untag_isize(capacity_tagged as isize);
        if raw < 0 {
            DEFAULT_CAPACITY
        } else {
            raw as usize
        }
    } else {
        DEFAULT_CAPACITY
    };
    match allocate_string_builder(runtime, stack_pointer, cap) {
        Ok(t) => t,
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate StringBuilder".to_string(),
            );
        },
    }
}

pub extern "C" fn string_builder_length(sb_tagged: usize) -> usize {
    HeapObject::from_tagged(sb_tagged).get_field(FIELD_LEN)
}

pub extern "C" fn string_builder_capacity(sb_tagged: usize) -> usize {
    let storage_tagged = sb_storage(sb_tagged);
    BuiltInTypes::construct_int(storage_capacity(storage_tagged) as isize) as usize
}

pub extern "C" fn string_builder_clear(sb_tagged: usize) -> usize {
    write_len(sb_tagged, 0);
    sb_tagged
}

pub extern "C" fn string_builder_push_byte(
    stack_pointer: usize,
    frame_pointer: usize,
    sb_tagged: usize,
    byte_tagged: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "string_builder_push_byte");
    let runtime = get_runtime().get_mut();

    let byte = BuiltInTypes::untag_isize(byte_tagged as isize) as i64;
    let byte = (byte & 0xFF) as u8;

    let len = sb_len(sb_tagged);
    let cap = storage_capacity(sb_storage(sb_tagged));

    let sb_tagged = if len >= cap {
        match ensure_capacity_for(runtime, stack_pointer, sb_tagged, len + 1) {
            Ok(new_sb) => new_sb,
            Err(e) => unsafe {
                throw_runtime_error(stack_pointer, "AllocationError", e.to_string());
            },
        }
    } else {
        sb_tagged
    };

    let storage_tagged = sb_storage(sb_tagged);
    unsafe { *storage_bytes_ptr(storage_tagged).add(len) = byte };
    write_len(sb_tagged, len + 1);
    sb_tagged
}

pub extern "C" fn string_builder_push_string(
    stack_pointer: usize,
    frame_pointer: usize,
    sb_tagged: usize,
    s_tagged: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "string_builder_push_string");
    let runtime = get_runtime().get_mut();

    // Collect source bytes into an owned Vec — survives allocation.
    let src_bytes: Vec<u8> = if BuiltInTypes::get_kind(s_tagged) == BuiltInTypes::String
        || BuiltInTypes::is_heap_pointer(s_tagged)
    {
        runtime.get_string_bytes_vec(s_tagged)
    } else {
        let s = runtime.get_string(stack_pointer, s_tagged);
        s.into_bytes()
    };

    if src_bytes.is_empty() {
        return sb_tagged;
    }

    let len = sb_len(sb_tagged);
    let needed = len + src_bytes.len();
    let sb_tagged = if needed > storage_capacity(sb_storage(sb_tagged)) {
        match ensure_capacity_for(runtime, stack_pointer, sb_tagged, needed) {
            Ok(new_sb) => new_sb,
            Err(e) => unsafe {
                throw_runtime_error(stack_pointer, "AllocationError", e.to_string());
            },
        }
    } else {
        sb_tagged
    };

    let storage_tagged = sb_storage(sb_tagged);
    unsafe {
        std::ptr::copy_nonoverlapping(
            src_bytes.as_ptr(),
            storage_bytes_ptr(storage_tagged).add(len),
            src_bytes.len(),
        );
    }
    write_len(sb_tagged, needed);
    sb_tagged
}

pub extern "C" fn string_builder_push_int(
    stack_pointer: usize,
    frame_pointer: usize,
    sb_tagged: usize,
    int_tagged: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "string_builder_push_int");
    let runtime = get_runtime().get_mut();

    let value = BuiltInTypes::untag_isize(int_tagged as isize);
    let s = value.to_string();
    let bytes = s.as_bytes();

    let len = sb_len(sb_tagged);
    let needed = len + bytes.len();
    let sb_tagged = if needed > storage_capacity(sb_storage(sb_tagged)) {
        match ensure_capacity_for(runtime, stack_pointer, sb_tagged, needed) {
            Ok(new_sb) => new_sb,
            Err(e) => unsafe {
                throw_runtime_error(stack_pointer, "AllocationError", e.to_string());
            },
        }
    } else {
        sb_tagged
    };

    let storage_tagged = sb_storage(sb_tagged);
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            storage_bytes_ptr(storage_tagged).add(len),
            bytes.len(),
        );
    }
    write_len(sb_tagged, needed);
    sb_tagged
}

pub extern "C" fn string_builder_push_float(
    stack_pointer: usize,
    frame_pointer: usize,
    sb_tagged: usize,
    float_tagged: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "string_builder_push_float");
    let runtime = get_runtime().get_mut();

    let raw = BuiltInTypes::untag(float_tagged) as *const f64;
    let value = unsafe { *raw.add(1) };
    let s = if value.is_nan() {
        "NaN".to_string()
    } else if value.is_infinite() {
        if value.is_sign_positive() {
            "infinity".to_string()
        } else {
            "-infinity".to_string()
        }
    } else {
        let s = value.to_string();
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{}.0", s)
        }
    };
    let bytes = s.as_bytes();

    let len = sb_len(sb_tagged);
    let needed = len + bytes.len();
    let sb_tagged = if needed > storage_capacity(sb_storage(sb_tagged)) {
        match ensure_capacity_for(runtime, stack_pointer, sb_tagged, needed) {
            Ok(new_sb) => new_sb,
            Err(e) => unsafe {
                throw_runtime_error(stack_pointer, "AllocationError", e.to_string());
            },
        }
    } else {
        sb_tagged
    };

    let storage_tagged = sb_storage(sb_tagged);
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            storage_bytes_ptr(storage_tagged).add(len),
            bytes.len(),
        );
    }
    write_len(sb_tagged, needed);
    sb_tagged
}

pub extern "C" fn string_builder_byte_at(sb_tagged: usize, idx_tagged: usize) -> usize {
    let len = sb_len(sb_tagged);
    let idx = BuiltInTypes::untag_isize(idx_tagged as isize);
    if idx < 0 || (idx as usize) >= len {
        return BuiltInTypes::null_value() as usize;
    }
    let storage_tagged = sb_storage(sb_tagged);
    let byte = unsafe { *storage_bytes_ptr(storage_tagged).add(idx as usize) } as isize;
    BuiltInTypes::construct_int(byte) as usize
}

pub extern "C" fn string_builder_set_byte_at(
    sb_tagged: usize,
    idx_tagged: usize,
    byte_tagged: usize,
) -> usize {
    let len = sb_len(sb_tagged);
    let idx = BuiltInTypes::untag_isize(idx_tagged as isize);
    if idx < 0 || (idx as usize) >= len {
        return BuiltInTypes::null_value() as usize;
    }
    let byte = (BuiltInTypes::untag_isize(byte_tagged as isize) & 0xFF) as u8;
    let storage_tagged = sb_storage(sb_tagged);
    unsafe { *storage_bytes_ptr(storage_tagged).add(idx as usize) = byte };
    sb_tagged
}

pub extern "C" fn string_builder_reverse(sb_tagged: usize) -> usize {
    let len = sb_len(sb_tagged);
    if len < 2 {
        return sb_tagged;
    }
    let storage_tagged = sb_storage(sb_tagged);
    let bytes = storage_bytes_ptr(storage_tagged);
    let mut i = 0;
    let mut j = len - 1;
    while i < j {
        unsafe {
            let a = *bytes.add(i);
            let b = *bytes.add(j);
            *bytes.add(i) = b;
            *bytes.add(j) = a;
        }
        i += 1;
        j -= 1;
    }
    sb_tagged
}

pub extern "C" fn string_builder_to_string(
    stack_pointer: usize,
    frame_pointer: usize,
    sb_tagged: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "string_builder_to_string");
    let runtime = get_runtime().get_mut();

    let len = sb_len(sb_tagged);
    let storage_tagged = sb_storage(sb_tagged);

    // Snapshot bytes — allocation may move storage.
    let mut snapshot = Vec::with_capacity(len);
    {
        let bytes = storage_bytes_ptr(storage_tagged);
        unsafe { snapshot.set_len(len) };
        unsafe { std::ptr::copy_nonoverlapping(bytes, snapshot.as_mut_ptr(), len) };
    }

    match runtime.allocate_string_from_bytes(stack_pointer, &snapshot) {
        Ok(t) => t.into(),
        Err(_) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "AllocationError",
                "Failed to allocate string from StringBuilder".to_string(),
            );
        },
    }
}

/// Used by `repr` / `print` to render a StringBuilder cheaply.
pub fn string_builder_debug_repr(value: usize) -> String {
    let len = sb_len(value);
    let storage_tagged = sb_storage(value);
    let cap = storage_capacity(storage_tagged);
    format!("StringBuilder {{ len: {}, capacity: {} }}", len, cap)
}

const _: () = {
    if TYPE_ID_STRING != 2 {
        panic!("TYPE_ID_STRING moved; revisit string-builder layout assumptions");
    }
};
