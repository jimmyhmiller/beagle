//! Persistent Vector implementation.
//!
//! A 32-way trie with tail optimization, inspired by Clojure's PersistentVector.
//! All nodes live on Beagle's heap and participate in garbage collection.
//!
//! # Structure
//!
//! ```text
//! PersistentVector (4 fields):
//!   [0] count - number of elements (tagged int)
//!   [1] shift - tree depth indicator (tagged int, starts at 5)
//!   [2] root  - pointer to root node (GcHandle or null)
//!   [3] tail  - pointer to tail array (GcHandle)
//! ```
//!
//! # Performance
//!
//! - count: O(1)
//! - get: O(log32 n) - effectively O(1) for practical sizes
//! - push: O(log32 n) amortized - O(1) when tail has room
//! - assoc: O(log32 n)

use std::error::Error;

use crate::runtime::Runtime;
use crate::types::BuiltInTypes;

use super::gc_handle::GcHandle;
use super::handle_arena::{Handle, HandleScope};
use super::type_ids::{TYPE_ID_PERSISTENT_VEC, TYPE_ID_PERSISTENT_VEC_NODE};

/// Field indices for PersistentVector struct
const FIELD_COUNT: usize = 0;
const FIELD_SHIFT: usize = 1;
const FIELD_ROOT: usize = 2;
const FIELD_TAIL: usize = 3;

/// Branching factor (32 = 2^5, so we use 5 bits per level)
const BRANCH_FACTOR: usize = 32;
const BRANCH_MASK: usize = 31; // 0b11111

/// Rust-native persistent vector living on Beagle's heap.
pub struct PersistentVec;

impl PersistentVec {
    /// Create a new empty persistent vector.
    ///
    /// Allocates the vector struct and two empty node arrays (root and tail).
    pub fn empty(runtime: &mut Runtime, stack_pointer: usize) -> Result<GcHandle, Box<dyn Error>> {
        let mut scope = HandleScope::new(runtime, stack_pointer);

        // Allocate empty root array (0 elements)
        let root_h = scope.allocate_typed_zeroed(0, TYPE_ID_PERSISTENT_VEC_NODE)?;

        // Allocate empty tail array (0 elements)
        let tail_h = scope.allocate_typed_zeroed(0, TYPE_ID_PERSISTENT_VEC_NODE)?;

        // Allocate the vector struct (4 fields)
        let vec_h = scope.allocate_typed_zeroed(4, TYPE_ID_PERSISTENT_VEC)?;

        // No re-reading needed! Handles are stable.
        let root = root_h.to_gc_handle();
        let tail = tail_h.to_gc_handle();
        let vec = vec_h.to_gc_handle();

        // Initialize fields
        vec.set_field(FIELD_COUNT, BuiltInTypes::construct_int(0) as usize);
        vec.set_field(FIELD_SHIFT, BuiltInTypes::construct_int(5) as usize);
        // Use write barriers for heap pointer writes
        vec.set_field_with_barrier(scope.runtime(), FIELD_ROOT, root.as_tagged());
        vec.set_field_with_barrier(scope.runtime(), FIELD_TAIL, tail.as_tagged());

        Ok(vec_h.to_gc_handle())
    }

    /// Get the count of elements in the vector.
    ///
    /// O(1) - just reads and untags the count field.
    pub fn count(vec: GcHandle) -> usize {
        let count_tagged = vec.get_field(FIELD_COUNT);
        BuiltInTypes::untag(count_tagged)
    }

    /// Get a value at the given index.
    ///
    /// Returns the raw tagged value (may be int, pointer, etc).
    /// Returns null value if index is out of bounds.
    ///
    /// O(log32 n) - navigates the trie.
    pub fn get(vec: GcHandle, index: usize) -> usize {
        let count = Self::count(vec);

        if index >= count {
            return BuiltInTypes::null_value() as usize;
        }

        let tail_offset = Self::tail_offset(count);

        if index >= tail_offset {
            // Value is in the tail
            let tail = GcHandle::from_tagged(vec.get_field(FIELD_TAIL));
            tail.get_field(index & BRANCH_MASK)
        } else {
            // Value is in the tree
            let shift = BuiltInTypes::untag(vec.get_field(FIELD_SHIFT));
            let root = GcHandle::from_tagged(vec.get_field(FIELD_ROOT));

            let node = Self::get_node_for_level(root, shift, index);
            node.get_field(index & BRANCH_MASK)
        }
    }

    /// Push a value onto the vector, returning a new vector.
    ///
    /// This is the primary way to add elements. The original vector
    /// is unchanged (persistent/immutable).
    ///
    /// O(log32 n) amortized, O(1) when tail has room.
    pub fn push(
        runtime: &mut Runtime,
        stack_pointer: usize,
        vec: GcHandle,
        value: usize,
    ) -> Result<GcHandle, Box<dyn Error>> {
        let count = Self::count(vec);
        let shift = BuiltInTypes::untag(vec.get_field(FIELD_SHIFT));
        let tail = GcHandle::from_tagged(vec.get_field(FIELD_TAIL));
        let tail_len = tail.field_count();

        if tail_len < BRANCH_FACTOR {
            // Fast path: room in tail
            Self::push_fast_path(
                runtime,
                stack_pointer,
                vec,
                value,
                count,
                shift,
                tail,
                tail_len,
            )
        } else {
            // Slow path: tail is full, need to push into tree
            Self::push_slow_path(runtime, stack_pointer, vec, value, count, shift, tail)
        }
    }

    /// Fast path for push when tail has room.
    #[allow(clippy::too_many_arguments)]
    fn push_fast_path(
        runtime: &mut Runtime,
        stack_pointer: usize,
        vec: GcHandle,
        value: usize,
        count: usize,
        shift: usize,
        tail: GcHandle,
        tail_len: usize,
    ) -> Result<GcHandle, Box<dyn Error>> {
        let mut scope = HandleScope::new(runtime, stack_pointer);

        // Protect inputs
        let vec_h = scope.alloc_handle(vec);
        let tail_h = scope.alloc_handle(tail);
        let value_h = scope.alloc(value);

        // Allocate new tail with one more slot
        let new_tail_h = scope.allocate_typed_zeroed(tail_len + 1, TYPE_ID_PERSISTENT_VEC_NODE)?;

        // Allocate new vector struct
        let new_vec_h = scope.allocate_typed_zeroed(4, TYPE_ID_PERSISTENT_VEC)?;

        // Bulk copy old tail to new tail
        let old_tail = tail_h.to_gc_handle();
        let new_tail = new_tail_h.to_gc_handle();
        old_tail.copy_fields_to(&new_tail, tail_len);

        // Call write barriers for all copied heap pointers
        for i in 0..tail_len {
            let slot = new_tail.get_field(i);
            scope.runtime().write_barrier(new_tail.as_tagged(), slot);
        }

        // Add new value with barrier (value could be a heap pointer)
        let new_tail = new_tail_h.to_gc_handle();
        new_tail.set_field_with_barrier(scope.runtime(), tail_len, value_h.get());

        // Set up new vector - reuse root from old vector
        let vec = vec_h.to_gc_handle();
        let root = vec.get_field(FIELD_ROOT);
        let new_vec = new_vec_h.to_gc_handle();
        new_vec.set_field(
            FIELD_COUNT,
            BuiltInTypes::construct_int((count + 1) as isize) as usize,
        );
        new_vec.set_field(
            FIELD_SHIFT,
            BuiltInTypes::construct_int(shift as isize) as usize,
        );
        // Use write barriers for heap pointer writes
        new_vec.set_field_with_barrier(scope.runtime(), FIELD_ROOT, root);
        let new_tail = new_tail_h.to_gc_handle();
        new_vec.set_field_with_barrier(scope.runtime(), FIELD_TAIL, new_tail.as_tagged());

        Ok(new_vec_h.to_gc_handle())
    }

    /// Slow path for push when tail is full.
    fn push_slow_path(
        runtime: &mut Runtime,
        stack_pointer: usize,
        vec: GcHandle,
        value: usize,
        count: usize,
        shift: usize,
        old_tail: GcHandle,
    ) -> Result<GcHandle, Box<dyn Error>> {
        let mut scope = HandleScope::new(runtime, stack_pointer);

        // Protect inputs
        let vec_h = scope.alloc_handle(vec);
        let old_tail_h = scope.alloc_handle(old_tail);
        let value_h = scope.alloc(value);

        // Create new single-element tail with the new value
        let new_tail_h = scope.allocate_typed_zeroed(1, TYPE_ID_PERSISTENT_VEC_NODE)?;

        let new_tail = new_tail_h.to_gc_handle();
        // Use write barrier - value could be a heap pointer
        new_tail.set_field_with_barrier(scope.runtime(), 0, value_h.get());

        // Determine if tree needs to grow
        let cnt_shifted = count >> 5;
        let shifted_one = 1usize << shift;

        let vec = vec_h.to_gc_handle();
        let root = GcHandle::from_tagged(vec.get_field(FIELD_ROOT));
        let root_h = scope.alloc_handle(root);

        let (new_root_h, new_shift) = if cnt_shifted > shifted_one {
            // Need new root level - tree grows taller
            Self::grow_tree(&mut scope, vec_h, old_tail_h, shift)?
        } else {
            // Insert old tail into existing tree
            Self::push_tail_into_tree(&mut scope, vec_h, root_h, old_tail_h, count, shift)?
        };

        // Allocate final vector
        let result_vec_h = scope.allocate_typed_zeroed(4, TYPE_ID_PERSISTENT_VEC)?;

        let new_tail = new_tail_h.to_gc_handle();
        let new_root = new_root_h.to_gc_handle();
        let result_vec = result_vec_h.to_gc_handle();

        result_vec.set_field(
            FIELD_COUNT,
            BuiltInTypes::construct_int((count + 1) as isize) as usize,
        );
        result_vec.set_field(
            FIELD_SHIFT,
            BuiltInTypes::construct_int(new_shift as isize) as usize,
        );
        // Use write barriers for heap pointer writes
        result_vec.set_field_with_barrier(scope.runtime(), FIELD_ROOT, new_root.as_tagged());
        result_vec.set_field_with_barrier(scope.runtime(), FIELD_TAIL, new_tail.as_tagged());

        Ok(result_vec_h.to_gc_handle())
    }

    /// Grow the tree by adding a new root level.
    fn grow_tree(
        scope: &mut HandleScope<'_>,
        vec_h: Handle,
        old_tail_h: Handle,
        shift: usize,
    ) -> Result<(Handle, usize), Box<dyn Error>> {
        let vec = vec_h.to_gc_handle();
        let current_root = GcHandle::from_tagged(vec.get_field(FIELD_ROOT));
        let current_root_h = scope.alloc_handle(current_root);

        let old_tail = old_tail_h.to_gc_handle();

        // Create path from old tail down to leaf level
        let new_path_h = Self::new_path(scope, shift, old_tail)?;

        // Allocate new root array
        let new_root_h = scope.allocate_typed_zeroed(BRANCH_FACTOR, TYPE_ID_PERSISTENT_VEC_NODE)?;

        let current_root = current_root_h.to_gc_handle();
        let new_path = new_path_h.to_gc_handle();
        let new_root = new_root_h.to_gc_handle();

        // New root: [old_root, new_path, null, null, ...]
        // Use write barriers for heap pointer writes
        new_root.set_field_with_barrier(scope.runtime(), 0, current_root.as_tagged());
        new_root.set_field_with_barrier(scope.runtime(), 1, new_path.as_tagged());
        // Fill rest with null (no barrier needed for null)
        for i in 2..BRANCH_FACTOR {
            new_root.set_field(i, BuiltInTypes::null_value() as usize);
        }

        Ok((new_root_h, shift + 5))
    }

    /// Create a path from the given level down to level 0, containing the node.
    fn new_path(
        scope: &mut HandleScope<'_>,
        level: usize,
        node: GcHandle,
    ) -> Result<Handle, Box<dyn Error>> {
        if level == 0 {
            Ok(scope.alloc_handle(node))
        } else {
            let node_h = scope.alloc_handle(node);

            // Recursively create the inner path first
            let inner_h = Self::new_path(scope, level - 5, node_h.to_gc_handle())?;

            // Allocate wrapper node
            let path_node_h = scope.allocate_typed_zeroed(BRANCH_FACTOR, TYPE_ID_PERSISTENT_VEC_NODE)?;

            let inner = inner_h.to_gc_handle();
            let path_node = path_node_h.to_gc_handle();

            // Only first slot is used - use write barrier for heap pointer
            path_node.set_field_with_barrier(scope.runtime(), 0, inner.as_tagged());
            // Fill rest with null (no barrier needed for null)
            for i in 1..BRANCH_FACTOR {
                path_node.set_field(i, BuiltInTypes::null_value() as usize);
            }

            Ok(path_node_h)
        }
    }

    /// Push the old tail into the existing tree structure.
    fn push_tail_into_tree(
        scope: &mut HandleScope<'_>,
        _vec_h: Handle,
        root_h: Handle,
        tail_h: Handle,
        count: usize,
        level: usize,
    ) -> Result<(Handle, usize), Box<dyn Error>> {
        let root = root_h.to_gc_handle();
        let tail = tail_h.to_gc_handle();

        let new_root_h = Self::do_push_tail(scope, level, root, tail, count)?;
        Ok((new_root_h, level))
    }

    /// Recursive helper for pushing tail into tree.
    fn do_push_tail(
        scope: &mut HandleScope<'_>,
        level: usize,
        parent: GcHandle,
        tail_node: GcHandle,
        count: usize,
    ) -> Result<Handle, Box<dyn Error>> {
        let parent_h = scope.alloc_handle(parent);
        let tail_h = scope.alloc_handle(tail_node);

        // Allocate new parent (copy-on-write)
        let new_parent_h = scope.allocate_typed_zeroed(BRANCH_FACTOR, TYPE_ID_PERSISTENT_VEC_NODE)?;

        let parent = parent_h.to_gc_handle();
        let new_parent = new_parent_h.to_gc_handle();

        // Bulk copy all fields from old parent
        let parent_len = parent.field_count();
        parent.copy_fields_to(&new_parent, parent_len);

        // Call write barriers for all copied heap pointers
        for i in 0..parent_len {
            let slot = new_parent.get_field(i);
            if slot != BuiltInTypes::null_value() as usize {
                scope.runtime().write_barrier(new_parent.as_tagged(), slot);
            }
        }

        // Fill rest with null if parent was smaller (no barrier needed for null)
        let new_parent = new_parent_h.to_gc_handle();
        for i in parent_len..BRANCH_FACTOR {
            new_parent.set_field(i, BuiltInTypes::null_value() as usize);
        }

        let sub_index = ((count - 1) >> level) & BRANCH_MASK;

        if level == 5 {
            // Bottom level: insert tail directly
            let tail = tail_h.to_gc_handle();
            let new_parent = new_parent_h.to_gc_handle();
            // Use write barrier for heap pointer
            new_parent.set_field_with_barrier(scope.runtime(), sub_index, tail.as_tagged());
            Ok(new_parent_h)
        } else {
            // Recurse into child
            let parent = parent_h.to_gc_handle();
            let child_ptr = parent.get_field(sub_index);

            let tail = tail_h.to_gc_handle();

            let new_child_h = if child_ptr == BuiltInTypes::null_value() as usize {
                // No child exists, create new path
                Self::new_path(scope, level - 5, tail)?
            } else {
                // Child exists, recurse
                let child = GcHandle::from_tagged(child_ptr);
                Self::do_push_tail(scope, level - 5, child, tail, count)?
            };

            let new_parent = new_parent_h.to_gc_handle();
            let new_child = new_child_h.to_gc_handle();

            // Use write barrier for heap pointer
            new_parent.set_field_with_barrier(scope.runtime(), sub_index, new_child.as_tagged());
            Ok(new_parent_h)
        }
    }

    /// Calculate the tail offset for a given count.
    fn tail_offset(count: usize) -> usize {
        if count < BRANCH_FACTOR {
            0
        } else {
            ((count - 1) >> 5) << 5
        }
    }

    /// Navigate the trie to find the node at the given level for the index.
    fn get_node_for_level(node: GcHandle, level: usize, index: usize) -> GcHandle {
        if level == 0 {
            node
        } else {
            let sub_index = (index >> level) & BRANCH_MASK;
            let child_ptr = node.get_field(sub_index);
            let child = GcHandle::from_tagged(child_ptr);
            Self::get_node_for_level(child, level - 5, index)
        }
    }

    /// Update a value at the given index, returning a new vector.
    ///
    /// O(log32 n) - creates new nodes along the path.
    pub fn assoc(
        runtime: &mut Runtime,
        stack_pointer: usize,
        vec: GcHandle,
        index: usize,
        value: usize,
    ) -> Result<GcHandle, Box<dyn Error>> {
        let count = Self::count(vec);

        if index >= count {
            // For now, panic on out of bounds. Could return Result instead.
            panic!("Index out of bounds: {} >= {}", index, count);
        }

        let tail_offset = Self::tail_offset(count);
        let shift = BuiltInTypes::untag(vec.get_field(FIELD_SHIFT));

        if index >= tail_offset {
            // Update in tail
            Self::assoc_tail(runtime, stack_pointer, vec, index, value, count, shift)
        } else {
            // Update in tree
            Self::assoc_tree(runtime, stack_pointer, vec, index, value, count, shift)
        }
    }

    /// Update a value in the tail.
    fn assoc_tail(
        runtime: &mut Runtime,
        stack_pointer: usize,
        vec: GcHandle,
        index: usize,
        value: usize,
        count: usize,
        shift: usize,
    ) -> Result<GcHandle, Box<dyn Error>> {
        let mut scope = HandleScope::new(runtime, stack_pointer);

        let vec_h = scope.alloc_handle(vec);
        let value_h = scope.alloc(value);

        let vec = vec_h.to_gc_handle();
        let tail = GcHandle::from_tagged(vec.get_field(FIELD_TAIL));
        let tail_h = scope.alloc_handle(tail);

        let tail_len = tail.field_count();

        // Allocate new tail (copy)
        let new_tail_h = scope.allocate_typed_zeroed(tail_len, TYPE_ID_PERSISTENT_VEC_NODE)?;

        // Allocate new vector
        let new_vec_h = scope.allocate_typed_zeroed(4, TYPE_ID_PERSISTENT_VEC)?;

        let old_tail = tail_h.to_gc_handle();
        let new_tail = new_tail_h.to_gc_handle();
        let value = value_h.get();

        // Copy tail and update the target index - use barriers for all values
        // since they could be heap pointers
        for i in 0..tail_len {
            if i == (index & BRANCH_MASK) {
                new_tail.set_field_with_barrier(scope.runtime(), i, value);
            } else {
                let slot = old_tail.get_field(i);
                let new_tail = new_tail_h.to_gc_handle();
                new_tail.set_field_with_barrier(scope.runtime(), i, slot);
            }
        }

        // Set up new vector
        let vec = vec_h.to_gc_handle();
        let new_vec = new_vec_h.to_gc_handle();
        new_vec.set_field(
            FIELD_COUNT,
            BuiltInTypes::construct_int(count as isize) as usize,
        );
        new_vec.set_field(
            FIELD_SHIFT,
            BuiltInTypes::construct_int(shift as isize) as usize,
        );
        // Use write barriers for heap pointer writes
        let root = vec.get_field(FIELD_ROOT);
        new_vec.set_field_with_barrier(scope.runtime(), FIELD_ROOT, root);
        let new_tail = new_tail_h.to_gc_handle();
        new_vec.set_field_with_barrier(scope.runtime(), FIELD_TAIL, new_tail.as_tagged());

        Ok(new_vec_h.to_gc_handle())
    }

    /// Update a value in the tree.
    fn assoc_tree(
        runtime: &mut Runtime,
        stack_pointer: usize,
        vec: GcHandle,
        index: usize,
        value: usize,
        count: usize,
        shift: usize,
    ) -> Result<GcHandle, Box<dyn Error>> {
        let mut scope = HandleScope::new(runtime, stack_pointer);

        let vec_h = scope.alloc_handle(vec);
        let value_h = scope.alloc(value);

        let vec = vec_h.to_gc_handle();
        let root = GcHandle::from_tagged(vec.get_field(FIELD_ROOT));
        let root_h = scope.alloc_handle(root);

        let new_root_h = Self::do_assoc(&mut scope, shift, root_h, index, value_h)?;

        // Allocate new vector
        let new_vec_h = scope.allocate_typed_zeroed(4, TYPE_ID_PERSISTENT_VEC)?;

        let vec = vec_h.to_gc_handle();
        let new_root = new_root_h.to_gc_handle();
        let new_vec = new_vec_h.to_gc_handle();

        new_vec.set_field(
            FIELD_COUNT,
            BuiltInTypes::construct_int(count as isize) as usize,
        );
        new_vec.set_field(
            FIELD_SHIFT,
            BuiltInTypes::construct_int(shift as isize) as usize,
        );
        // Use write barriers for heap pointer writes
        new_vec.set_field_with_barrier(scope.runtime(), FIELD_ROOT, new_root.as_tagged());
        let tail = vec.get_field(FIELD_TAIL);
        new_vec.set_field_with_barrier(scope.runtime(), FIELD_TAIL, tail);

        Ok(new_vec_h.to_gc_handle())
    }

    /// Recursive helper for tree association.
    fn do_assoc(
        scope: &mut HandleScope<'_>,
        level: usize,
        node_h: Handle,
        index: usize,
        value_h: Handle,
    ) -> Result<Handle, Box<dyn Error>> {
        let node = node_h.to_gc_handle();
        let node_len = node.field_count();

        // Allocate new node (copy-on-write)
        let new_node_h = scope.allocate_typed_zeroed(node_len, TYPE_ID_PERSISTENT_VEC_NODE)?;

        let node = node_h.to_gc_handle();
        let new_node = new_node_h.to_gc_handle();

        // Bulk copy all fields
        node.copy_fields_to(&new_node, node_len);

        // Call write barriers for all copied heap pointers
        for i in 0..node_len {
            let slot = new_node.get_field(i);
            scope.runtime().write_barrier(new_node.as_tagged(), slot);
        }

        if level == 0 {
            // Leaf level: update the value directly
            let value = value_h.get();
            let new_node = new_node_h.to_gc_handle();
            // Use write barrier - value could be a heap pointer
            new_node.set_field_with_barrier(scope.runtime(), index & BRANCH_MASK, value);
            Ok(new_node_h)
        } else {
            // Recurse into child
            let sub_index = (index >> level) & BRANCH_MASK;
            let node = node_h.to_gc_handle();
            let child_ptr = node.get_field(sub_index);
            let child = GcHandle::from_tagged(child_ptr);
            let child_h = scope.alloc_handle(child);

            let new_child_h = Self::do_assoc(scope, level - 5, child_h, index, value_h)?;

            let new_node = new_node_h.to_gc_handle();
            let new_child = new_child_h.to_gc_handle();

            // Use write barrier for heap pointer
            new_node.set_field_with_barrier(scope.runtime(), sub_index, new_child.as_tagged());
            Ok(new_node_h)
        }
    }
}

#[cfg(test)]
mod tests {
    // Integration tests in resources/*.bg will cover this
}
