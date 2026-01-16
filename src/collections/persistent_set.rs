//! Persistent Set implementation backed by PersistentMap.
//!
//! A hash-based persistent set that stores elements as keys in a HAMT
//! with `true` as the value. All nodes live on Beagle's heap and participate in GC.
//!
//! # Performance
//!
//! - count: O(1)
//! - contains: O(log32 n) - effectively O(1) for practical sizes
//! - add: O(log32 n)

use std::error::Error;

use crate::runtime::Runtime;
use crate::types::BuiltInTypes;

use super::gc_handle::GcHandle;
use super::handle_arena::HandleScope;
use super::persistent_map::PersistentMap;
use super::persistent_vec::PersistentVec;
use super::type_ids::{TYPE_ID_PERSISTENT_MAP, TYPE_ID_PERSISTENT_SET};

/// Field indices for PersistentSet struct
const FIELD_COUNT: usize = 0;
const FIELD_ROOT: usize = 1;

/// Rust-native persistent set living on Beagle's heap.
/// Internally backed by a PersistentMap where elements are keys and values are `true`.
pub struct PersistentSet;

impl PersistentSet {
    /// Create a new empty persistent set.
    pub fn empty(runtime: &mut Runtime, stack_pointer: usize) -> Result<GcHandle, Box<dyn Error>> {
        let mut scope = HandleScope::new(runtime, stack_pointer);

        // Allocate the set struct (2 fields: count, root)
        let set = scope.allocate_typed(2, TYPE_ID_PERSISTENT_SET)?;

        // Initialize fields
        let set_gc = set.to_gc_handle();
        set_gc.set_field(FIELD_COUNT, BuiltInTypes::construct_int(0) as usize);
        set_gc.set_field(FIELD_ROOT, BuiltInTypes::null_value() as usize);

        Ok(set_gc)
    }

    /// Get the count of elements in the set.
    pub fn count(set: GcHandle) -> usize {
        let count_tagged = set.get_field(FIELD_COUNT);
        BuiltInTypes::untag(count_tagged)
    }

    /// Check if an element exists in the set.
    ///
    /// Returns true if the element is in the set, false otherwise.
    pub fn contains(runtime: &Runtime, set: GcHandle, element: usize) -> bool {
        let root_ptr = set.get_field(FIELD_ROOT);

        if root_ptr == BuiltInTypes::null_value() as usize {
            return false;
        }

        // We reuse PersistentMap's get logic by creating a temporary map handle
        // The set's root is the same structure as a map's root
        let result = Self::map_get(runtime, set, element);
        result != BuiltInTypes::null_value() as usize
    }

    /// Internal helper to get a value from the underlying map structure.
    /// This reuses PersistentMap's get logic without creating a full map wrapper.
    fn map_get(runtime: &Runtime, set: GcHandle, key: usize) -> usize {
        // Create a temporary "view" of the set as a map
        // Since set has same layout as map (count, root), we can use map's get
        PersistentMap::get(runtime, set, key)
    }

    /// Add an element to the set, returning a new set.
    ///
    /// If the element already exists, returns a set with the same count.
    pub fn add(
        runtime: &mut Runtime,
        stack_pointer: usize,
        set_ptr: usize,
        element: usize,
    ) -> Result<GcHandle, Box<dyn Error>> {
        let mut scope = HandleScope::new(runtime, stack_pointer);

        let set_h = scope.alloc(set_ptr);
        let element_h = scope.alloc(element);

        // Use `true` as the value for set elements
        let true_value = BuiltInTypes::true_value() as usize;
        let true_h = scope.alloc(true_value);

        let old_count = Self::count(set_h.to_gc_handle());
        let root_ptr = set_h.to_gc_handle().get_field(FIELD_ROOT);

        // Create a temporary map with the set's root to use PersistentMap::assoc
        // First, we need to create a map struct pointing to the same root
        let temp_map = scope.allocate_typed(2, TYPE_ID_PERSISTENT_MAP)?;
        temp_map.to_gc_handle().set_field(
            FIELD_COUNT,
            BuiltInTypes::construct_int(old_count as isize) as usize,
        );
        temp_map
            .to_gc_handle()
            .set_field_with_barrier(scope.runtime(), FIELD_ROOT, root_ptr);

        // Now use map assoc
        let new_map = PersistentMap::assoc(
            scope.runtime(),
            stack_pointer,
            temp_map.get(),
            element_h.get(),
            true_h.get(),
        )?;

        // Create a new set struct with the map's new root
        let new_set = scope.allocate_typed(2, TYPE_ID_PERSISTENT_SET)?;
        let new_count = PersistentMap::count(new_map);

        let new_set_gc = new_set.to_gc_handle();
        new_set_gc.set_field(
            FIELD_COUNT,
            BuiltInTypes::construct_int(new_count as isize) as usize,
        );
        new_set_gc.set_field_with_barrier(
            scope.runtime(),
            FIELD_ROOT,
            new_map.get_field(FIELD_ROOT),
        );

        Ok(new_set_gc)
    }

    /// Get all elements from the set as a PersistentVec.
    pub fn elements(
        runtime: &mut Runtime,
        stack_pointer: usize,
        set: GcHandle,
    ) -> Result<GcHandle, Box<dyn Error>> {
        // Since set elements are stored as map keys, we use map's keys function
        // We need to create a temporary map view of the set
        let mut scope = HandleScope::new(runtime, stack_pointer);

        let set_h = scope.alloc(set.as_tagged());
        let set = set_h.to_gc_handle();

        let count = Self::count(set);
        let root_ptr = set.get_field(FIELD_ROOT);

        if root_ptr == BuiltInTypes::null_value() as usize {
            return PersistentVec::empty(scope.runtime(), stack_pointer);
        }

        // Create temporary map with set's root
        let temp_map = scope.allocate_typed(2, TYPE_ID_PERSISTENT_MAP)?;
        temp_map.to_gc_handle().set_field(
            FIELD_COUNT,
            BuiltInTypes::construct_int(count as isize) as usize,
        );
        let set = set_h.to_gc_handle();
        temp_map.to_gc_handle().set_field_with_barrier(
            scope.runtime(),
            FIELD_ROOT,
            set.get_field(FIELD_ROOT),
        );

        // Use map's keys function
        PersistentMap::keys(scope.runtime(), stack_pointer, temp_map.to_gc_handle())
    }
}

#[cfg(test)]
mod tests {
    // Integration tests in resources/*.bg will cover this
}
