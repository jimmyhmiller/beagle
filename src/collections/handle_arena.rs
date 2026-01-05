//! HandleScope - GC-safe handle allocation using GlobalObject slots.
//!
//! Provides stable handles (slot indices) into the thread's GlobalObject.
//! GC updates GlobalObject entries directly, so handles remain valid across allocations.
//!
//! # Performance
//!
//! - Scope enter/exit: O(n) where n = handles in scope
//! - Handle alloc: O(1) via GlobalObject.add_root
//! - Handle get: O(1) via GlobalObject.get_root
//! - No registration needed - GlobalObject is already traced by GC

use std::error::Error;

use crate::runtime::Runtime;
use crate::types::BuiltInTypes;

use super::gc_handle::GcHandle;

/// A handle to a value stored in GlobalObject.
///
/// Handles are slot indices - the underlying value may be updated by GC,
/// but the handle itself remains valid until its scope exits.
#[derive(Clone, Copy, Debug)]
pub struct Handle {
    slot: usize,
    /// Pointer to runtime for accessing the value (valid for scope lifetime)
    runtime_ptr: *const Runtime,
}

impl Handle {
    /// Get the current value of this handle.
    ///
    /// The value may have changed since allocation if GC ran and moved objects.
    #[inline]
    pub fn get(self) -> usize {
        // Safety: runtime_ptr is valid for the scope lifetime
        let runtime = unsafe { &*self.runtime_ptr };
        runtime.get_handle_root(self.slot)
    }

    /// Get as a GcHandle if this is a heap pointer.
    #[inline]
    pub fn as_gc_handle(self) -> Option<GcHandle> {
        let val = self.get();
        GcHandle::try_from_tagged(val)
    }

    /// Get as a GcHandle, panicking if not a heap pointer.
    #[inline]
    pub fn to_gc_handle(self) -> GcHandle {
        GcHandle::from_tagged(self.get())
    }

    /// Check if this handle holds null.
    #[inline]
    pub fn is_null(self) -> bool {
        self.get() == BuiltInTypes::null_value() as usize
    }
}

/// RAII scope guard for handle allocation.
///
/// Handles allocated within this scope are automatically released when
/// the scope is dropped. Scopes can be nested.
pub struct HandleScope<'a> {
    runtime: &'a mut Runtime,
    stack_pointer: usize,
    /// Slot indices allocated in this scope (dynamically sized)
    slots: Vec<usize>,
}

impl<'a> HandleScope<'a> {
    /// Create a new handle scope.
    ///
    /// No registration needed - handles are stored in GlobalObject which
    /// is automatically traced by GC.
    pub fn new(runtime: &'a mut Runtime, stack_pointer: usize) -> Self {
        Self {
            runtime,
            stack_pointer,
            slots: Vec::with_capacity(16), // Start with reasonable capacity
        }
    }

    /// Allocate a handle for a raw value (may or may not be a heap pointer).
    #[inline]
    pub fn alloc(&mut self, value: usize) -> Handle {
        let slot = self
            .runtime
            .add_handle_root(value)
            .expect("HandleScope: GlobalObject full - allocation failed");

        self.slots.push(slot);

        Handle {
            slot,
            runtime_ptr: self.runtime as *const Runtime,
        }
    }

    /// Allocate a handle for a GcHandle.
    #[inline]
    pub fn alloc_handle(&mut self, handle: GcHandle) -> Handle {
        self.alloc(handle.as_tagged())
    }

    /// Allocate a new heap object and return a handle to it.
    ///
    /// This may trigger GC, which will update all handles in GlobalObject.
    pub fn allocate(&mut self, num_fields: usize) -> Result<Handle, Box<dyn Error>> {
        // Check if GC is in progress before allocating
        if self.runtime.is_paused() {
            let frame_pointer = crate::builtins::get_saved_frame_pointer();
            let gc_return_addr = crate::builtins::get_saved_gc_return_addr();
            self.runtime
                .gc_impl(self.stack_pointer, frame_pointer, gc_return_addr);
        }

        let ptr =
            self.runtime
                .allocate(num_fields, self.stack_pointer, BuiltInTypes::HeapObject)?;
        Ok(self.alloc(ptr))
    }

    /// Allocate a new heap object with a type_id and return a handle to it.
    pub fn allocate_typed(
        &mut self,
        num_fields: usize,
        type_id: u8,
    ) -> Result<Handle, Box<dyn Error>> {
        let handle = self.allocate(num_fields)?;
        handle.to_gc_handle().set_type_id(type_id);
        Ok(handle)
    }

    /// Allocate a new zeroed heap object and return a handle to it.
    ///
    /// This is important for arrays that don't initialize all fields immediately,
    /// as GC (especially with gc_always) may run before fields are set.
    pub fn allocate_zeroed(&mut self, num_fields: usize) -> Result<Handle, Box<dyn Error>> {
        // Check if GC is in progress before allocating
        if self.runtime.is_paused() {
            let frame_pointer = crate::builtins::get_saved_frame_pointer();
            let gc_return_addr = crate::builtins::get_saved_gc_return_addr();
            self.runtime
                .gc_impl(self.stack_pointer, frame_pointer, gc_return_addr);
        }

        let ptr = self.runtime.allocate_zeroed(
            num_fields,
            self.stack_pointer,
            BuiltInTypes::HeapObject,
        )?;
        Ok(self.alloc(ptr))
    }

    /// Allocate a new zeroed heap object with a type_id and return a handle to it.
    pub fn allocate_typed_zeroed(
        &mut self,
        num_fields: usize,
        type_id: u8,
    ) -> Result<Handle, Box<dyn Error>> {
        let handle = self.allocate_zeroed(num_fields)?;
        handle.to_gc_handle().set_type_id(type_id);
        Ok(handle)
    }

    /// Set a field on a Handle's object with proper write barrier.
    /// This is required for generational GC to track old-to-young references.
    ///
    /// Parameters:
    /// - `handle`: The object to write to (will be read fresh from the handle)
    /// - `index`: Field index to write to
    /// - `value`: The value to write (may be a tagged int or heap pointer)
    #[inline]
    pub fn set_field(&mut self, handle: Handle, index: usize, value: usize) {
        let object_ptr = handle.get();
        self.runtime
            .set_field_with_barrier(object_ptr, index, value);
    }

    /// Set a field on a Handle's object with proper write barrier, using another Handle as value.
    /// This is the common case when writing a heap pointer to an object field.
    #[inline]
    pub fn set_field_handle(&mut self, handle: Handle, index: usize, value_handle: Handle) {
        let object_ptr = handle.get();
        let value = value_handle.get();
        self.runtime
            .set_field_with_barrier(object_ptr, index, value);
    }

    /// Get the stack pointer for this scope.
    #[inline]
    pub fn stack_pointer(&self) -> usize {
        self.stack_pointer
    }

    /// Get access to the runtime.
    #[inline]
    pub fn runtime(&mut self) -> &mut Runtime {
        self.runtime
    }
}

impl Drop for HandleScope<'_> {
    fn drop(&mut self) {
        // Free all slots allocated in this scope
        for &slot in &self.slots {
            self.runtime.remove_handle_root(slot);
        }
    }
}

// Note: Tests for HandleScope require a full runtime environment.
// Integration tests are in the resources/*.bg test files.
