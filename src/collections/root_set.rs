//! RootSet - Simple GC root storage for AllocationContext.
//!
//! RootSet is a simple Vec of values that need to be protected from GC.
//! When GC runs, it receives `&mut RootSet` and updates any moved pointers
//! in-place. This eliminates the need for per-root registration with the GC.

/// A set of GC roots stored in AllocationContext.
///
/// Values are stored in a Vec and can be looked up by RootIdx.
/// GC updates the Vec in-place when objects move.
pub struct RootSet {
    roots: Vec<usize>,
}

/// Handle to a protected root - just an index into the RootSet.
///
/// Use `RootSet::get(idx)` to read the current value.
#[derive(Clone, Copy, Debug)]
pub struct RootIdx(pub(crate) usize);

impl RootSet {
    /// Create a new empty RootSet with pre-allocated capacity.
    pub fn new() -> Self {
        RootSet {
            roots: Vec::with_capacity(16),
        }
    }

    /// Protect a value from GC. Returns an index to retrieve it later.
    pub fn protect(&mut self, value: usize) -> RootIdx {
        let idx = self.roots.len();
        self.roots.push(value);
        RootIdx(idx)
    }

    /// Get the current value at the given index.
    ///
    /// This may return a different pointer than was originally protected
    /// if GC has moved the object.
    pub fn get(&self, idx: RootIdx) -> usize {
        self.roots[idx.0]
    }

    /// Get a mutable slice of all roots for GC to update.
    pub fn roots_mut(&mut self) -> &mut [usize] {
        &mut self.roots
    }

    /// Get an immutable slice of all roots for GC marking.
    pub fn roots(&self) -> &[usize] {
        &self.roots
    }

    /// Get the number of protected roots.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Clear all roots (used when context is finished).
    pub fn clear(&mut self) {
        self.roots.clear();
    }
}

impl Default for RootSet {
    fn default() -> Self {
        Self::new()
    }
}

/// A Send+Sync wrapper for *mut RootSet.
///
/// This is safe because:
/// 1. RootSet is only accessed during GC, which is single-threaded
/// 2. The pointer is only dereferenced by the thread that registered it
/// 3. Registration/unregistration is protected by GC synchronization
#[derive(Clone, Copy)]
pub struct RootSetPtr(pub *mut RootSet);

// Safety: See above - GC synchronization protects access
unsafe impl Send for RootSetPtr {}
unsafe impl Sync for RootSetPtr {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protect_and_get() {
        let mut roots = RootSet::new();
        let idx1 = roots.protect(100);
        let idx2 = roots.protect(200);

        assert_eq!(roots.get(idx1), 100);
        assert_eq!(roots.get(idx2), 200);
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_roots_mut_update() {
        let mut roots = RootSet::new();
        let idx = roots.protect(100);

        // Simulate GC updating the root
        roots.roots_mut()[0] = 999;

        assert_eq!(roots.get(idx), 999);
    }
}
