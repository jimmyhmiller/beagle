//! Mutable HashMap implementation using open-addressing with linear probing.
//!
//! A high-performance mutable hash map backed by Beagle heap arrays.
//! Keys and values are stored in parallel arrays, so the GC traces them
//! automatically — no special GC handling needed.
//!
//! # Structure (3-field heap object, TYPE_ID_MUTABLE_MAP)
//!
//! - Field 0: keys array (Beagle heap array, null = empty slot)
//! - Field 1: values array (Beagle heap array, parallel to keys)
//! - Field 2: size (tagged int)
//!
//! # Performance
//!
//! - get: O(1) amortized
//! - put: O(1) amortized
//! - increment: O(1) amortized (optimized get-or-0 + 1)

use std::error::Error;

use crate::runtime::Runtime;
use crate::types::BuiltInTypes;

use super::gc_handle::GcHandle;
use super::handle_arena::{Handle, HandleScope};
use super::type_ids::{TYPE_ID_MUTABLE_MAP, TYPE_ID_RAW_ARRAY};

/// Field indices for MutableMap struct
const FIELD_KEYS: usize = 0;
const FIELD_VALUES: usize = 1;
const FIELD_SIZE: usize = 2;

/// Load factor threshold (75%) — resize when size > capacity * 3 / 4
const LOAD_FACTOR_NUM: usize = 3;
const LOAD_FACTOR_DEN: usize = 4;

pub struct MutableMap;

impl MutableMap {
    /// Create a new empty mutable map with the given capacity.
    pub fn empty(
        runtime: &mut Runtime,
        stack_pointer: usize,
        capacity: usize,
    ) -> Result<GcHandle, Box<dyn Error>> {
        let cap = if capacity < 4 { 4 } else { capacity };

        let mut scope = HandleScope::new(runtime, stack_pointer);

        // Allocate keys array (all null = empty)
        let keys_h = scope.allocate_typed_zeroed(cap, TYPE_ID_RAW_ARRAY)?;
        // Initialize all slots to null
        let keys = keys_h.to_gc_handle();
        let null_val = BuiltInTypes::null_value() as usize;
        for i in 0..cap {
            keys.set_field(i, null_val);
        }

        // Allocate values array (all null)
        let values_h = scope.allocate_typed_zeroed(cap, TYPE_ID_RAW_ARRAY)?;
        let values = values_h.to_gc_handle();
        for i in 0..cap {
            values.set_field(i, null_val);
        }

        // Allocate the map struct (3 fields)
        let map_h = scope.allocate_typed_zeroed(3, TYPE_ID_MUTABLE_MAP)?;
        let map = map_h.to_gc_handle();
        let keys = keys_h.to_gc_handle();
        let values = values_h.to_gc_handle();

        map.set_field_with_barrier(scope.runtime(), FIELD_KEYS, keys.as_tagged());
        map.set_field_with_barrier(scope.runtime(), FIELD_VALUES, values.as_tagged());
        map.set_field(FIELD_SIZE, BuiltInTypes::construct_int(0) as usize);

        Ok(map)
    }

    /// Get the number of entries in the map.
    #[inline]
    pub fn count(map: GcHandle) -> usize {
        BuiltInTypes::untag(map.get_field(FIELD_SIZE))
    }

    /// Get the capacity (number of slots) from the keys array.
    #[inline]
    fn capacity(map: GcHandle) -> usize {
        let keys = GcHandle::from_tagged(map.get_field(FIELD_KEYS));
        keys.field_count()
    }

    /// Get a value by key. Returns null if not found.
    pub fn get(runtime: &Runtime, map: GcHandle, key: usize) -> usize {
        let null_val = BuiltInTypes::null_value() as usize;
        let keys = GcHandle::from_tagged(map.get_field(FIELD_KEYS));
        let values = GcHandle::from_tagged(map.get_field(FIELD_VALUES));
        let cap = keys.field_count();

        let hash = runtime.hash_value(key);
        let mut idx = hash % cap;

        for _ in 0..cap {
            let slot_key = keys.get_field(idx);
            if slot_key == null_val {
                return null_val;
            }
            if runtime.equal(slot_key, key) {
                return values.get_field(idx);
            }
            idx += 1;
            if idx >= cap {
                idx = 0;
            }
        }

        null_val
    }

    /// Insert or update a key-value pair. Mutates in place.
    ///
    /// Uses a fast path that avoids HandleScope overhead when no resize is needed
    /// (99.99% of calls). HandleScope is only needed during resize, which allocates
    /// new arrays and may trigger GC. Without allocation, GC cannot run, so raw
    /// pointers are safe.
    pub fn put(
        runtime: &mut Runtime,
        stack_pointer: usize,
        map_ptr: usize,
        key: usize,
        value: usize,
    ) -> Result<(), Box<dyn Error>> {
        let map = GcHandle::from_tagged(map_ptr);
        let size = Self::count(map);
        let cap = Self::capacity(map);

        if size * LOAD_FACTOR_DEN >= cap * LOAD_FACTOR_NUM {
            // Slow path: need resize, which allocates → need HandleScope for GC safety
            return Self::put_slow(runtime, stack_pointer, map_ptr, key, value);
        }

        // Fast path: no resize needed, no allocation, GC cannot trigger.
        // Safe to use raw pointers directly.
        let keys = GcHandle::from_tagged(map.get_field(FIELD_KEYS));
        let values = GcHandle::from_tagged(map.get_field(FIELD_VALUES));
        let cap = keys.field_count();
        let null_val = BuiltInTypes::null_value() as usize;

        let hash = runtime.hash_value(key);
        let mut idx = hash % cap;

        for _ in 0..cap {
            let slot_key = keys.get_field(idx);
            if slot_key == null_val {
                // Empty slot — insert new entry
                keys.set_field_with_barrier(runtime, idx, key);
                values.set_field_with_barrier(runtime, idx, value);

                let new_size = size + 1;
                map.set_field(
                    FIELD_SIZE,
                    BuiltInTypes::construct_int(new_size as isize) as usize,
                );
                return Ok(());
            }
            if runtime.equal(slot_key, key) {
                // Key exists — update value
                values.set_field_with_barrier(runtime, idx, value);
                return Ok(());
            }
            idx += 1;
            if idx >= cap {
                idx = 0;
            }
        }

        Err("MutableMap::put: table full (should not happen)".into())
    }

    /// Slow path for put: resize is needed, so we use HandleScope.
    fn put_slow(
        runtime: &mut Runtime,
        stack_pointer: usize,
        map_ptr: usize,
        key: usize,
        value: usize,
    ) -> Result<(), Box<dyn Error>> {
        let mut scope = HandleScope::new(runtime, stack_pointer);
        let map_h = scope.alloc(map_ptr);
        let key_h = scope.alloc(key);
        let value_h = scope.alloc(value);

        Self::resize(&mut scope, map_h)?;

        // After resize, do the insert (no further resize possible)
        let map = map_h.to_gc_handle();
        let keys = GcHandle::from_tagged(map.get_field(FIELD_KEYS));
        let cap = keys.field_count();
        let null_val = BuiltInTypes::null_value() as usize;

        let key_val = key_h.get();
        let hash = scope.runtime().hash_value(key_val);
        let mut idx = hash % cap;

        for _ in 0..cap {
            let slot_key = keys.get_field(idx);
            if slot_key == null_val {
                let k = key_h.get();
                let v = value_h.get();
                let map = map_h.to_gc_handle();
                let keys = GcHandle::from_tagged(map.get_field(FIELD_KEYS));
                let values = GcHandle::from_tagged(map.get_field(FIELD_VALUES));

                keys.set_field_with_barrier(scope.runtime(), idx, k);
                values.set_field_with_barrier(scope.runtime(), idx, v);

                let map = map_h.to_gc_handle();
                let new_size = Self::count(map) + 1;
                map.set_field(
                    FIELD_SIZE,
                    BuiltInTypes::construct_int(new_size as isize) as usize,
                );
                return Ok(());
            }
            if scope.runtime().equal(slot_key, key_val) {
                let v = value_h.get();
                let map = map_h.to_gc_handle();
                let values = GcHandle::from_tagged(map.get_field(FIELD_VALUES));
                values.set_field_with_barrier(scope.runtime(), idx, v);
                return Ok(());
            }
            idx += 1;
            if idx >= cap {
                idx = 0;
            }
        }

        Err("MutableMap::put_slow: table full (should not happen after resize)".into())
    }

    /// Increment the integer value for a key by 1, inserting 1 if absent.
    /// Combines get + put into a single probe for the common count_kmers pattern.
    pub fn increment(
        runtime: &mut Runtime,
        stack_pointer: usize,
        map_ptr: usize,
        key: usize,
    ) -> Result<(), Box<dyn Error>> {
        let map = GcHandle::from_tagged(map_ptr);
        let size = Self::count(map);
        let cap = Self::capacity(map);

        if size * LOAD_FACTOR_DEN >= cap * LOAD_FACTOR_NUM {
            // Need resize first, then do the increment
            return Self::increment_slow(runtime, stack_pointer, map_ptr, key);
        }

        // Fast path: no resize needed
        let keys = GcHandle::from_tagged(map.get_field(FIELD_KEYS));
        let values = GcHandle::from_tagged(map.get_field(FIELD_VALUES));
        let cap = keys.field_count();
        let null_val = BuiltInTypes::null_value() as usize;

        let hash = runtime.hash_value(key);
        let mut idx = hash % cap;

        for _ in 0..cap {
            let slot_key = keys.get_field(idx);
            if slot_key == null_val {
                // Empty slot — insert with count = 1
                keys.set_field_with_barrier(runtime, idx, key);
                values.set_field_with_barrier(
                    runtime,
                    idx,
                    BuiltInTypes::construct_int(1) as usize,
                );
                let new_size = size + 1;
                map.set_field(
                    FIELD_SIZE,
                    BuiltInTypes::construct_int(new_size as isize) as usize,
                );
                return Ok(());
            }
            if runtime.equal(slot_key, key) {
                // Key exists — increment value
                let old_val = values.get_field(idx);
                let old_int = BuiltInTypes::untag(old_val) as isize;
                values.set_field(idx, BuiltInTypes::construct_int(old_int + 1) as usize);
                return Ok(());
            }
            idx += 1;
            if idx >= cap {
                idx = 0;
            }
        }

        Err("MutableMap::increment: table full (should not happen)".into())
    }

    fn increment_slow(
        runtime: &mut Runtime,
        stack_pointer: usize,
        map_ptr: usize,
        key: usize,
    ) -> Result<(), Box<dyn Error>> {
        let mut scope = HandleScope::new(runtime, stack_pointer);
        let map_h = scope.alloc(map_ptr);
        let key_h = scope.alloc(key);

        Self::resize(&mut scope, map_h)?;

        let map = map_h.to_gc_handle();
        let keys = GcHandle::from_tagged(map.get_field(FIELD_KEYS));
        let cap = keys.field_count();
        let null_val = BuiltInTypes::null_value() as usize;

        let key_val = key_h.get();
        let hash = scope.runtime().hash_value(key_val);
        let mut idx = hash % cap;

        for _ in 0..cap {
            let slot_key = keys.get_field(idx);
            if slot_key == null_val {
                let k = key_h.get();
                let map = map_h.to_gc_handle();
                let keys = GcHandle::from_tagged(map.get_field(FIELD_KEYS));
                let values = GcHandle::from_tagged(map.get_field(FIELD_VALUES));

                keys.set_field_with_barrier(scope.runtime(), idx, k);
                values.set_field_with_barrier(
                    scope.runtime(),
                    idx,
                    BuiltInTypes::construct_int(1) as usize,
                );

                let map = map_h.to_gc_handle();
                let new_size = Self::count(map) + 1;
                map.set_field(
                    FIELD_SIZE,
                    BuiltInTypes::construct_int(new_size as isize) as usize,
                );
                return Ok(());
            }
            if scope.runtime().equal(slot_key, key_val) {
                let map = map_h.to_gc_handle();
                let values = GcHandle::from_tagged(map.get_field(FIELD_VALUES));
                let old_val = values.get_field(idx);
                let old_int = BuiltInTypes::untag(old_val) as isize;
                values.set_field(idx, BuiltInTypes::construct_int(old_int + 1) as usize);
                return Ok(());
            }
            idx += 1;
            if idx >= cap {
                idx = 0;
            }
        }

        Err("MutableMap::increment_slow: table full after resize".into())
    }

    /// Collect all entries as an array of [key, value] pair arrays.
    pub fn entries(
        runtime: &mut Runtime,
        stack_pointer: usize,
        map: GcHandle,
    ) -> Result<GcHandle, Box<dyn Error>> {
        let size = Self::count(map);
        let keys = GcHandle::from_tagged(map.get_field(FIELD_KEYS));
        let cap = keys.field_count();
        let null_val = BuiltInTypes::null_value() as usize;

        let mut scope = HandleScope::new(runtime, stack_pointer);

        // Allocate result array
        let result_h = scope.allocate_typed_zeroed(size, TYPE_ID_RAW_ARRAY)?;

        // We need to re-read map fields after allocation (GC may have run)
        // But map is not protected by the scope — we need to protect it.
        // Actually, the caller passes map as GcHandle, but map_ptr was already
        // on the Beagle stack. We'll read keys/values from the map handle.
        // Since map is a GcHandle (not Handle), we need to be careful.
        // Let's protect the map's constituent parts.
        let map_tagged = map.as_tagged();
        let map_handle = scope.alloc(map_tagged);

        let mut dest_idx = 0;
        for src_idx in 0..cap {
            let map_gc = map_handle.to_gc_handle();
            let keys = GcHandle::from_tagged(map_gc.get_field(FIELD_KEYS));
            let slot_key = keys.get_field(src_idx);

            if slot_key != null_val {
                let values = GcHandle::from_tagged(map_gc.get_field(FIELD_VALUES));
                let slot_value = values.get_field(src_idx);

                let key_h = scope.alloc(slot_key);
                let val_h = scope.alloc(slot_value);

                // Allocate a 2-element pair array
                let pair_h = scope.allocate_typed_zeroed(2, TYPE_ID_RAW_ARRAY)?;
                let pair = pair_h.to_gc_handle();
                pair.set_field_with_barrier(scope.runtime(), 0, key_h.get());
                pair.set_field_with_barrier(scope.runtime(), 1, val_h.get());

                // Store pair in result
                let result = result_h.to_gc_handle();
                let pair = pair_h.to_gc_handle();
                result.set_field_with_barrier(scope.runtime(), dest_idx, pair.as_tagged());

                dest_idx += 1;
            }
        }

        Ok(result_h.to_gc_handle())
    }

    /// Resize the hash table by doubling capacity and rehashing all entries.
    fn resize(scope: &mut HandleScope<'_>, map_h: Handle) -> Result<(), Box<dyn Error>> {
        let map = map_h.to_gc_handle();
        let old_keys_ptr = map.get_field(FIELD_KEYS);
        let old_values_ptr = map.get_field(FIELD_VALUES);
        let old_keys = GcHandle::from_tagged(old_keys_ptr);
        let old_cap = old_keys.field_count();
        let new_cap = old_cap * 2;
        let null_val = BuiltInTypes::null_value() as usize;

        let old_keys_h = scope.alloc(old_keys_ptr);
        let old_values_h = scope.alloc(old_values_ptr);

        // Allocate new keys array
        let new_keys_h = scope.allocate_typed_zeroed(new_cap, TYPE_ID_RAW_ARRAY)?;
        let new_keys = new_keys_h.to_gc_handle();
        for i in 0..new_cap {
            new_keys.set_field(i, null_val);
        }

        // Allocate new values array
        let new_values_h = scope.allocate_typed_zeroed(new_cap, TYPE_ID_RAW_ARRAY)?;
        let new_values = new_values_h.to_gc_handle();
        for i in 0..new_cap {
            new_values.set_field(i, null_val);
        }

        // Rehash all entries from old arrays into new arrays
        for i in 0..old_cap {
            let old_keys = GcHandle::from_tagged(old_keys_h.get());
            let slot_key = old_keys.get_field(i);

            if slot_key != null_val {
                let old_values = GcHandle::from_tagged(old_values_h.get());
                let slot_value = old_values.get_field(i);

                let hash = scope.runtime().hash_value(slot_key);
                let new_keys = new_keys_h.to_gc_handle();
                let new_values = new_values_h.to_gc_handle();
                let mut idx = hash % new_cap;

                // Find empty slot in new table (guaranteed to exist since we doubled)
                loop {
                    if new_keys.get_field(idx) == null_val {
                        new_keys.set_field_with_barrier(scope.runtime(), idx, slot_key);
                        new_values.set_field_with_barrier(scope.runtime(), idx, slot_value);
                        break;
                    }
                    idx += 1;
                    if idx >= new_cap {
                        idx = 0;
                    }
                }
            }
        }

        // Update map to point to new arrays
        let map = map_h.to_gc_handle();
        let new_keys = new_keys_h.to_gc_handle();
        let new_values = new_values_h.to_gc_handle();
        map.set_field_with_barrier(scope.runtime(), FIELD_KEYS, new_keys.as_tagged());
        map.set_field_with_barrier(scope.runtime(), FIELD_VALUES, new_values.as_tagged());

        Ok(())
    }
}
