//! HandleArena - Thread-local handle storage for GC-safe allocations.
//!
//! Provides stable handles (indices) into a thread-local arena. GC updates
//! the arena slots directly, so handles remain valid across allocations.
//!
//! # Performance
//!
//! - Scope enter/exit: O(1) integer save/restore
//! - Handle alloc: O(1) array write
//! - Handle get: O(1) array index
//! - GC registration: Once per thread (not per scope)

use std::cell::{Cell, UnsafeCell};
use std::error::Error;

use crate::runtime::Runtime;
use crate::types::BuiltInTypes;

use super::gc_handle::GcHandle;

/// Maximum number of handles per thread.
const MAX_HANDLES: usize = 256;

/// Maximum scope nesting depth.
const MAX_SCOPE_DEPTH: usize = 32;

/// Thread-local handle storage.
///
/// Stores values that need to be protected from GC. During collection,
/// the GC updates `slots[0..top]` in place when objects move.
pub struct HandleArena {
    slots: [usize; MAX_HANDLES],
    top: usize,
    scope_stack: [usize; MAX_SCOPE_DEPTH],
    scope_depth: usize,
}

impl HandleArena {
    /// Create a new empty arena.
    pub const fn new() -> Self {
        Self {
            slots: [0; MAX_HANDLES],
            top: 0,
            scope_stack: [0; MAX_SCOPE_DEPTH],
            scope_depth: 0,
        }
    }

    /// Enter a new scope. Saves the current top for later restoration.
    #[inline]
    pub fn enter_scope(&mut self) {
        debug_assert!(
            self.scope_depth < MAX_SCOPE_DEPTH,
            "HandleArena: scope depth overflow (max {})",
            MAX_SCOPE_DEPTH
        );
        self.scope_stack[self.scope_depth] = self.top;
        self.scope_depth += 1;
    }

    /// Exit the current scope. Restores top to discard handles from this scope.
    #[inline]
    pub fn exit_scope(&mut self) {
        debug_assert!(self.scope_depth > 0, "HandleArena: scope underflow");
        self.scope_depth -= 1;
        self.top = self.scope_stack[self.scope_depth];
    }

    /// Allocate a handle for a value.
    #[inline]
    pub fn alloc(&mut self, value: usize) -> Handle {
        assert!(
            self.top < MAX_HANDLES,
            "HandleArena: handle overflow (max {})",
            MAX_HANDLES
        );
        let idx = self.top;
        self.slots[idx] = value;
        self.top += 1;
        Handle(idx as u16)
    }

    /// Get the current value of a handle.
    #[inline]
    pub fn get(&self, h: Handle) -> usize {
        debug_assert!(
            (h.0 as usize) < self.top,
            "HandleArena: invalid handle {}",
            h.0
        );
        self.slots[h.0 as usize]
    }

    /// Get mutable access to all active slots for GC to update.
    #[inline]
    pub fn roots_mut(&mut self) -> &mut [usize] {
        &mut self.slots[..self.top]
    }

    /// Get immutable access to all active slots for GC marking.
    #[inline]
    pub fn roots(&self) -> &[usize] {
        &self.slots[..self.top]
    }

    /// Get the number of active handles.
    #[inline]
    pub fn len(&self) -> usize {
        self.top
    }

    /// Check if arena is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.top == 0
    }
}

impl Default for HandleArena {
    fn default() -> Self {
        Self::new()
    }
}

/// A handle to a value in the thread-local arena.
///
/// Handles are stable indices - the underlying value may be updated by GC,
/// but the handle itself remains valid until its scope exits.
#[derive(Clone, Copy, Debug)]
pub struct Handle(u16);

impl Handle {
    /// Get the current value of this handle.
    ///
    /// The value may have changed since allocation if GC ran and moved objects.
    #[inline]
    pub fn get(self) -> usize {
        with_arena(|arena| arena.get(self))
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

// Thread-local arena storage
thread_local! {
    static HANDLE_ARENA: UnsafeCell<HandleArena> = const { UnsafeCell::new(HandleArena::new()) };
    static ARENA_REGISTERED: Cell<bool> = const { Cell::new(false) };
    static ARENA_REG_ID: Cell<usize> = const { Cell::new(usize::MAX) };
    /// Tracks scope nesting depth. When 1, we're in outermost scope.
    static SCOPE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Pointer to a HandleArena, wrapped for Send+Sync.
///
/// Safety: Arena is only mutated by its owning thread or by GC during STW.
#[derive(Clone, Copy)]
pub struct HandleArenaPtr(pub *mut HandleArena);

unsafe impl Send for HandleArenaPtr {}
unsafe impl Sync for HandleArenaPtr {}

/// Access the thread-local arena.
#[inline]
fn with_arena<R>(f: impl FnOnce(&mut HandleArena) -> R) -> R {
    HANDLE_ARENA.with(|arena| {
        // Safety: We have exclusive access via thread-local, or GC has STW
        let arena = unsafe { &mut *arena.get() };
        f(arena)
    })
}

/// Ensure the thread-local arena is registered with the GC.
fn ensure_registered(runtime: &mut Runtime) {
    ARENA_REGISTERED.with(|registered| {
        if !registered.get() {
            HANDLE_ARENA.with(|arena| {
                let ptr = HandleArenaPtr(arena.get());
                let thread_id = std::thread::current().id();
                let id = runtime.register_handle_arena(ptr, thread_id);
                ARENA_REG_ID.with(|id_cell| id_cell.set(id));
            });
            registered.set(true);
        }
    });
}

/// RAII scope guard for handle allocation.
///
/// Handles allocated within this scope are automatically released when
/// the scope is dropped. Scopes can be nested.
pub struct HandleScope<'a> {
    runtime: &'a mut Runtime,
    stack_pointer: usize,
}

impl<'a> HandleScope<'a> {
    /// Create a new handle scope.
    ///
    /// Registers the thread-local arena with GC if not already registered.
    /// GC will wait for this thread when scanning roots - the thread pauses
    /// on allocation attempts when GC is in progress.
    pub fn new(runtime: &'a mut Runtime, stack_pointer: usize) -> Self {
        ensure_registered(runtime);
        with_arena(|arena| arena.enter_scope());

        // Track scope depth
        SCOPE_DEPTH.with(|d| {
            let current = d.get();
            d.set(current + 1);
        });

        // Note: We do NOT register as c_calling here. This is intentional.
        // If we registered as c_calling, GC would proceed while we're still
        // running, causing a race condition when GC updates our arena.
        // Instead, GC will wait for us, and we'll pause when we next allocate.

        Self {
            runtime,
            stack_pointer,
        }
    }

    /// Allocate a handle for a raw value (may or may not be a heap pointer).
    #[inline]
    pub fn alloc(&mut self, value: usize) -> Handle {
        with_arena(|arena| arena.alloc(value))
    }

    /// Allocate a handle for a GcHandle.
    #[inline]
    pub fn alloc_handle(&mut self, handle: GcHandle) -> Handle {
        self.alloc(handle.as_tagged())
    }

    /// Allocate a new heap object and return a handle to it.
    ///
    /// This may trigger GC, which will update all handles in the arena.
    pub fn allocate(&mut self, num_fields: usize) -> Result<Handle, Box<dyn Error>> {
        // Check if GC is in progress before allocating
        // This ensures we participate in GC even if allocation succeeds
        // (allocation might succeed even while another thread is doing GC)
        if self.runtime.is_paused() {
            // Trigger gc_impl which will call __pause
            // gc_return_addr = 0 tells stack walker to use FP+8 lookup
            // (not that it matters - our roots are in HandleArena, not on Beagle stack)
            self.runtime.gc_impl(
                self.stack_pointer,
                crate::builtins::get_saved_frame_pointer(),
                0,
            );
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

    /// Get the stack pointer for this scope.
    #[inline]
    pub fn stack_pointer(&self) -> usize {
        self.stack_pointer
    }

    /// Get access to the runtime (use with care - allocations through
    /// runtime directly won't automatically protect handles).
    #[inline]
    pub fn runtime(&mut self) -> &mut Runtime {
        self.runtime
    }
}

impl Drop for HandleScope<'_> {
    fn drop(&mut self) {
        with_arena(|arena| arena.exit_scope());

        // Track scope depth
        SCOPE_DEPTH.with(|d| {
            let current = d.get();
            d.set(current - 1);
        });

        // Note: We don't unregister from c_calling because we never registered.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let mut arena = HandleArena::new();
        arena.enter_scope();

        let h1 = arena.alloc(100);
        let h2 = arena.alloc(200);

        assert_eq!(arena.get(h1), 100);
        assert_eq!(arena.get(h2), 200);
        assert_eq!(arena.len(), 2);

        arena.exit_scope();
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn test_arena_nested_scopes() {
        let mut arena = HandleArena::new();
        arena.enter_scope();
        let h1 = arena.alloc(100);

        arena.enter_scope();
        let h2 = arena.alloc(200);
        assert_eq!(arena.len(), 2);

        arena.exit_scope();
        assert_eq!(arena.len(), 1);
        assert_eq!(arena.get(h1), 100);

        arena.exit_scope();
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn test_roots_mut() {
        let mut arena = HandleArena::new();
        arena.enter_scope();
        arena.alloc(100);
        arena.alloc(200);

        // Simulate GC updating roots
        let roots = arena.roots_mut();
        roots[0] = 999;
        roots[1] = 888;

        assert_eq!(arena.get(Handle(0)), 999);
        assert_eq!(arena.get(Handle(1)), 888);

        arena.exit_scope();
    }
}
