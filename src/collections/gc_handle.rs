//! GcHandle - Safe wrapper around GC-managed heap pointers.
//!
//! GcHandle provides a safe abstraction for working with objects on Beagle's heap.
//! It wraps a tagged pointer and provides methods for field access.
//!
//! # GC Safety
//!
//! When allocations occur, GC may run and move objects (in compacting mode).
//! Any GcHandle held across an allocation must be protected as a temporary root
//! using AllocationContext. After allocation, re-read the handle to get the
//! potentially-updated pointer.

use crate::types::{BuiltInTypes, HeapObject};

/// A handle to a GC-managed object on Beagle's heap.
///
/// The internal pointer is tagged (lower 3 bits = type tag).
/// This handle should be registered as a temporary root before any
/// allocation that might trigger GC.
#[derive(Debug, Clone, Copy)]
pub struct GcHandle {
    tagged_ptr: usize,
}

impl GcHandle {
    /// Create a handle from a tagged heap pointer.
    ///
    /// # Panics
    /// Panics if the pointer is not a valid heap pointer (tag != 0b110).
    pub fn from_tagged(ptr: usize) -> Self {
        debug_assert!(
            BuiltInTypes::is_heap_pointer(ptr),
            "GcHandle::from_tagged called with non-heap pointer: {:#x}",
            ptr
        );
        Self { tagged_ptr: ptr }
    }

    /// Try to create a handle from a tagged pointer.
    /// Returns None if the pointer is not a heap pointer.
    pub fn try_from_tagged(ptr: usize) -> Option<Self> {
        if BuiltInTypes::is_heap_pointer(ptr) {
            Some(Self { tagged_ptr: ptr })
        } else {
            None
        }
    }

    /// Get the raw tagged pointer value.
    /// This is what should be stored in heap object fields.
    pub fn as_tagged(&self) -> usize {
        self.tagged_ptr
    }

    /// Get the underlying HeapObject for lower-level operations.
    pub fn as_heap_object(&self) -> HeapObject {
        HeapObject::from_tagged(self.tagged_ptr)
    }

    /// Read a field at the given index.
    /// Returns the raw value (may be tagged int, tagged pointer, etc).
    pub fn get_field(&self, index: usize) -> usize {
        self.as_heap_object().get_field(index)
    }

    /// Write a value to a field at the given index.
    pub fn set_field(&self, index: usize, value: usize) {
        self.as_heap_object().write_field(index as i32, value);
    }

    /// Get the number of fields in this object.
    ///
    /// Note: We don't provide a `fields()` method that returns a slice because
    /// the HeapObject is a temporary and we can't return references to it.
    /// Use `get_field(i)` in a loop instead.
    pub fn field_count(&self) -> usize {
        self.as_heap_object().get_fields().len()
    }

    /// Copy all fields from this handle to another.
    /// Used for copy-on-write operations.
    pub fn copy_fields_to(&self, dest: &GcHandle, count: usize) {
        if count == 0 {
            return;
        }
        // Use bulk memcpy instead of field-by-field copy
        // Need to keep HeapObject alive for the slice borrow
        let src_obj = self.as_heap_object();
        let src = src_obj.get_fields();
        let mut dest_obj = dest.as_heap_object();
        let dst = dest_obj.get_fields_mut();
        dst[..count].copy_from_slice(&src[..count]);
    }

    /// Copy a range of fields from this handle to another at a specific offset.
    /// Used for inserting into the middle of arrays.
    ///
    /// Copies `count` fields from `self[src_start..]` to `dest[dst_start..]`.
    pub fn copy_fields_range_to(
        &self,
        dest: &GcHandle,
        src_start: usize,
        dst_start: usize,
        count: usize,
    ) {
        if count == 0 {
            return;
        }
        // Need to keep HeapObject alive for the slice borrow
        let src_obj = self.as_heap_object();
        let src = src_obj.get_fields();
        let mut dest_obj = dest.as_heap_object();
        let dst = dest_obj.get_fields_mut();
        dst[dst_start..dst_start + count].copy_from_slice(&src[src_start..src_start + count]);
    }

    /// Write the type_id to the object header.
    pub fn set_type_id(&self, type_id: u8) {
        let mut heap_obj = self.as_heap_object();
        heap_obj.write_type_id(type_id as usize);
    }

    /// Read the type_id from the object header.
    pub fn get_type_id(&self) -> u8 {
        self.as_heap_object().get_type_id() as u8
    }

    /// Create a null handle (represents Beagle's null value).
    pub fn null() -> Self {
        Self {
            tagged_ptr: BuiltInTypes::null_value() as usize,
        }
    }

    /// Check if this handle represents null.
    pub fn is_null(&self) -> bool {
        self.tagged_ptr == BuiltInTypes::null_value() as usize
    }
}

impl Default for GcHandle {
    fn default() -> Self {
        Self::null()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_handle() {
        let null = GcHandle::null();
        assert!(null.is_null());
        assert_eq!(null.as_tagged(), BuiltInTypes::null_value() as usize);
    }

    #[test]
    fn test_try_from_non_heap() {
        // A tagged integer should not create a handle
        let int_val = BuiltInTypes::construct_int(42) as usize;
        assert!(GcHandle::try_from_tagged(int_val).is_none());
    }
}
