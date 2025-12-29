//! Persistent Map implementation using HAMT (Hash Array Mapped Trie).
//!
//! A hash-based persistent map that uses bitmap-indexed nodes for efficient
//! sparse storage. All nodes live on Beagle's heap and participate in GC.
//!
//! # Node Types (matching Clojure's PersistentHashMap layout)
//!
//! - **PersistentMap** (2 fields): count, root
//! - **BitmapIndexedNode** (2 fields): bitmap, children array (keys/values interleaved)
//!   - Even indexes are keys, odd are values
//!   - If key slot contains a node pointer and value is null, it's a child node
//! - **ArrayNode** (2 fields): count, children array of 32 node slots
//!   - Stores only child node pointers (no inline key/value)
//!   - Empty slots are null
//! - **CollisionNode** (3 fields): hash, count, kv_array (alternating keys/values)
//!
//! # Performance
//!
//! - count: O(1)
//! - get: O(log32 n) - effectively O(1) for practical sizes
//! - assoc: O(log32 n)

use std::error::Error;

use crate::runtime::Runtime;
use crate::types::BuiltInTypes;

use super::gc_handle::GcHandle;
use super::handle_arena::{Handle, HandleScope};
use super::type_ids::{
    TYPE_ID_ARRAY_NODE, TYPE_ID_BITMAP_NODE, TYPE_ID_COLLISION_NODE, TYPE_ID_PERSISTENT_MAP,
};

/// Field indices for PersistentMap struct
const FIELD_COUNT: usize = 0;
const FIELD_ROOT: usize = 1;

/// Field indices for BitmapIndexedNode
const BN_FIELD_BITMAP: usize = 0;
const BN_FIELD_CHILDREN: usize = 1;

/// Field indices for CollisionNode
const CN_FIELD_HASH: usize = 0;
const CN_FIELD_COUNT: usize = 1;
const CN_FIELD_KV_ARRAY: usize = 2;

/// Branching factor (32 = 2^5, so we use 5 bits per level)
const BRANCH_MASK: usize = 31;

/// Field indices for ArrayNode
/// ArrayNode stores only child nodes (no inline key/value), matching Clojure's layout
const AN_FIELD_COUNT: usize = 0;
const AN_FIELD_CHILDREN: usize = 1;

/// Threshold for promoting BitmapNode to ArrayNode
const BITMAP_TO_ARRAY_THRESHOLD: usize = 16;

/// Threshold for demoting ArrayNode back to BitmapNode
const ARRAY_TO_BITMAP_THRESHOLD: usize = 8;

/// Rust-native persistent map living on Beagle's heap.
pub struct PersistentMap;

impl PersistentMap {
    /// Create a new empty persistent map.
    pub fn empty(runtime: &mut Runtime, stack_pointer: usize) -> Result<GcHandle, Box<dyn Error>> {
        let mut scope = HandleScope::new(runtime, stack_pointer);

        // Allocate the map struct (2 fields: count, root)
        let map = scope.allocate_typed(2, TYPE_ID_PERSISTENT_MAP)?;

        // Initialize fields
        let map_gc = map.to_gc_handle();
        map_gc.set_field(FIELD_COUNT, BuiltInTypes::construct_int(0) as usize);
        map_gc.set_field(FIELD_ROOT, BuiltInTypes::null_value() as usize);

        Ok(map_gc)
    }

    /// Get the count of key-value pairs in the map.
    pub fn count(map: GcHandle) -> usize {
        let count_tagged = map.get_field(FIELD_COUNT);
        BuiltInTypes::untag(count_tagged)
    }

    /// Get a value by key.
    ///
    /// Returns the value if found, or null if not found.
    pub fn get(runtime: &Runtime, map: GcHandle, key: usize) -> usize {
        let root_ptr = map.get_field(FIELD_ROOT);

        if root_ptr == BuiltInTypes::null_value() as usize {
            return BuiltInTypes::null_value() as usize;
        }

        let hash = runtime.hash_value(key);
        let root = GcHandle::from_tagged(root_ptr);

        Self::find_node(runtime, root, key, hash, 0)
    }

    /// Find a value in the tree.
    /// Optimized iterative version to avoid recursion overhead.
    #[inline]
    fn find_node(
        runtime: &Runtime,
        node: GcHandle,
        key: usize,
        hash: usize,
        shift: usize,
    ) -> usize {
        let null_val = BuiltInTypes::null_value() as usize;
        let mut current_node = node;
        let mut current_shift = shift;

        loop {
            let type_id = current_node.get_type_id();

            if type_id == TYPE_ID_BITMAP_NODE {
                let bitmap = BuiltInTypes::untag(current_node.get_field(BN_FIELD_BITMAP));
                let bit = 1usize << ((hash >> current_shift) & BRANCH_MASK);

                if (bitmap & bit) == 0 {
                    return null_val;
                }

                let children_ptr = current_node.get_field(BN_FIELD_CHILDREN);
                let children = GcHandle::from_tagged(children_ptr);
                let index = (bitmap & (bit - 1)).count_ones() as usize;

                let slot_key = children.get_field(index * 2);
                let slot_value = children.get_field(index * 2 + 1);

                // Fast path: if value is not null, it's a leaf entry
                if slot_value != null_val {
                    return if runtime.equal(slot_key, key) {
                        slot_value
                    } else {
                        null_val
                    };
                }

                // Value is null - check if it's a sub-node
                if BuiltInTypes::is_heap_pointer(slot_key) {
                    let child = GcHandle::from_tagged(slot_key);
                    let child_type = child.get_type_id();
                    if child_type == TYPE_ID_BITMAP_NODE || child_type == TYPE_ID_ARRAY_NODE {
                        current_node = child;
                        current_shift += 5;
                        continue;
                    } else if child_type == TYPE_ID_COLLISION_NODE {
                        return Self::find_in_collision_node(runtime, child, key);
                    }
                }

                // Leaf with null value
                return if runtime.equal(slot_key, key) {
                    slot_value
                } else {
                    null_val
                };
            } else if type_id == TYPE_ID_ARRAY_NODE {
                // ArrayNode stores only child nodes (no inline key/value)
                let children_ptr = current_node.get_field(AN_FIELD_CHILDREN);
                let children = GcHandle::from_tagged(children_ptr);
                let index = (hash >> current_shift) & BRANCH_MASK;

                let child_ptr = children.get_field(index);

                if child_ptr == null_val {
                    return null_val;
                }

                // Child is always a node (BitmapNode, ArrayNode, or CollisionNode)
                let child = GcHandle::from_tagged(child_ptr);
                let child_type = child.get_type_id();

                if child_type == TYPE_ID_BITMAP_NODE || child_type == TYPE_ID_ARRAY_NODE {
                    current_node = child;
                    current_shift += 5;
                    continue;
                } else if child_type == TYPE_ID_COLLISION_NODE {
                    return Self::find_in_collision_node(runtime, child, key);
                }

                // Should not reach here - invalid child type
                return null_val;
            } else if type_id == TYPE_ID_COLLISION_NODE {
                return Self::find_in_collision_node(runtime, current_node, key);
            } else {
                return null_val;
            }
        }
    }

    /// Find a value in a collision node.
    fn find_in_collision_node(runtime: &Runtime, node: GcHandle, key: usize) -> usize {
        let count = BuiltInTypes::untag(node.get_field(CN_FIELD_COUNT));
        let kv_array_ptr = node.get_field(CN_FIELD_KV_ARRAY);
        let kv_array = GcHandle::from_tagged(kv_array_ptr);

        for i in 0..count {
            let stored_key = kv_array.get_field(i * 2);
            if runtime.equal(stored_key, key) {
                return kv_array.get_field(i * 2 + 1);
            }
        }

        BuiltInTypes::null_value() as usize
    }

    /// Associate a key with a value, returning a new map.
    ///
    /// Takes raw tagged pointer instead of GcHandle to ensure the pointer
    /// is protected by HandleScope BEFORE any access. This prevents races
    /// where GC could move the object between GcHandle creation and protection.
    pub fn assoc(
        runtime: &mut Runtime,
        stack_pointer: usize,
        map_ptr: usize,
        key: usize,
        value: usize,
    ) -> Result<GcHandle, Box<dyn Error>> {
        // CRITICAL: Create HandleScope and protect all pointers BEFORE any access.
        // This ensures GC cannot move objects while we hold unprotected GcHandles.
        let mut scope = HandleScope::new(runtime, stack_pointer);

        let map_h = scope.alloc(map_ptr);
        let key_h = scope.alloc(key);
        let value_h = scope.alloc(value);

        // Now safe to access - map_h is protected and will be updated if GC runs
        let map = map_h.to_gc_handle();
        let hash = scope.runtime().hash_value(key_h.get());
        let count = Self::count(map);
        let root_ptr = map.get_field(FIELD_ROOT);

        let (new_root, added) = if root_ptr == BuiltInTypes::null_value() as usize {
            // Empty map - create initial leaf node
            let node = Self::create_leaf_node(&mut scope, key_h, value_h, hash)?;
            (node, true)
        } else {
            let root = GcHandle::from_tagged(root_ptr);
            let root_h = scope.alloc_handle(root);

            Self::assoc_node(&mut scope, root_h, 0, hash, key_h, value_h)?
        };

        // Allocate new map
        let new_map = scope.allocate_typed(2, TYPE_ID_PERSISTENT_MAP)?;
        let new_count = if added { count + 1 } else { count };

        // No re-reading needed! Handles are stable.
        let new_map_gc = new_map.to_gc_handle();
        new_map_gc.set_field(
            FIELD_COUNT,
            BuiltInTypes::construct_int(new_count as isize) as usize,
        );
        new_map_gc.set_field(FIELD_ROOT, new_root.get());

        Ok(new_map_gc)
    }

    /// Create a leaf bitmap node with a single key-value pair.
    fn create_leaf_node(
        scope: &mut HandleScope<'_>,
        key_h: Handle,
        value_h: Handle,
        hash: usize,
    ) -> Result<Handle, Box<dyn Error>> {
        // Create children array with 2 elements (key, value)
        let children = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;

        // Create bitmap node
        let node = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;

        // No re-reading needed - handles are stable
        children.to_gc_handle().set_field(0, key_h.get());
        children.to_gc_handle().set_field(1, value_h.get());

        // Set bitmap with single bit for hash at level 0
        let bit = 1usize << (hash & BRANCH_MASK);
        node.to_gc_handle().set_field(
            BN_FIELD_BITMAP,
            BuiltInTypes::construct_int(bit as isize) as usize,
        );
        node.to_gc_handle()
            .set_field(BN_FIELD_CHILDREN, children.get());

        Ok(node)
    }

    /// Create a singleton bitmap node with a single key-value pair at the given shift level.
    /// Used when wrapping leaf entries for ArrayNode children.
    fn create_singleton_bitmap_node(
        scope: &mut HandleScope<'_>,
        key_h: Handle,
        value_h: Handle,
        hash: usize,
        shift: usize,
    ) -> Result<Handle, Box<dyn Error>> {
        // Create children array with 2 elements (key, value)
        let children = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;

        // Create bitmap node
        let node = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;

        children.to_gc_handle().set_field(0, key_h.get());
        children.to_gc_handle().set_field(1, value_h.get());

        // Set bitmap with single bit for hash at this shift level
        let bit = 1usize << ((hash >> shift) & BRANCH_MASK);
        node.to_gc_handle().set_field(
            BN_FIELD_BITMAP,
            BuiltInTypes::construct_int(bit as isize) as usize,
        );
        node.to_gc_handle()
            .set_field(BN_FIELD_CHILDREN, children.get());

        Ok(node)
    }

    /// Associate into an existing node.
    /// Returns (new_node, was_added).
    fn assoc_node(
        scope: &mut HandleScope<'_>,
        node_h: Handle,
        shift: usize,
        hash: usize,
        key_h: Handle,
        value_h: Handle,
    ) -> Result<(Handle, bool), Box<dyn Error>> {
        let node = node_h.to_gc_handle();
        let type_id = node.get_type_id();

        match type_id {
            TYPE_ID_BITMAP_NODE => {
                Self::assoc_bitmap_node(scope, node_h, shift, hash, key_h, value_h)
            }
            TYPE_ID_ARRAY_NODE => {
                Self::assoc_array_node(scope, node_h, shift, hash, key_h, value_h)
            }
            TYPE_ID_COLLISION_NODE => {
                Self::assoc_collision_node(scope, node_h, shift, hash, key_h, value_h)
            }
            _ => {
                // Unknown node type
                Err("Unknown node type in assoc_node".into())
            }
        }
    }

    /// Associate into a bitmap-indexed node.
    fn assoc_bitmap_node(
        scope: &mut HandleScope<'_>,
        node_h: Handle,
        shift: usize,
        hash: usize,
        key_h: Handle,
        value_h: Handle,
    ) -> Result<(Handle, bool), Box<dyn Error>> {
        let node = node_h.to_gc_handle();
        let bitmap = BuiltInTypes::untag(node.get_field(BN_FIELD_BITMAP));
        let bit = 1usize << ((hash >> shift) & BRANCH_MASK);
        let index = (bitmap & (bit - 1)).count_ones() as usize;

        let children_ptr = node.get_field(BN_FIELD_CHILDREN);
        let children = GcHandle::from_tagged(children_ptr);
        let children_h = scope.alloc_handle(children);

        if (bitmap & bit) == 0 {
            // No entry at this position - add new entry
            let children = children_h.to_gc_handle();
            let old_len = children.field_count();

            // Create new children array with 2 more slots
            let new_children_h = scope.allocate_typed(old_len + 2, TYPE_ID_BITMAP_NODE)?;

            // Copy entries before insertion point (bulk memcpy)
            let children = children_h.to_gc_handle();
            let new_children = new_children_h.to_gc_handle();
            children.copy_fields_to(&new_children, index * 2);

            // Insert new key-value
            let key = key_h.get();
            let value = value_h.get();
            let new_children = new_children_h.to_gc_handle();
            new_children.set_field(index * 2, key);
            new_children.set_field(index * 2 + 1, value);

            // Copy entries after insertion point (bulk memcpy)
            let children = children_h.to_gc_handle();
            let new_children = new_children_h.to_gc_handle();
            children.copy_fields_range_to(
                &new_children,
                index * 2,
                index * 2 + 2,
                old_len - index * 2,
            );

            // Create new bitmap node
            let new_bitmap = bitmap | bit;
            let new_node_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;

            let new_children = new_children_h.to_gc_handle();
            let new_node = new_node_h.to_gc_handle();
            new_node.set_field(
                BN_FIELD_BITMAP,
                BuiltInTypes::construct_int(new_bitmap as isize) as usize,
            );
            new_node.set_field(BN_FIELD_CHILDREN, new_children.as_tagged());

            // Check if we should promote to ArrayNode
            let popcount = new_bitmap.count_ones() as usize;
            if popcount >= BITMAP_TO_ARRAY_THRESHOLD {
                let array_node = Self::bitmap_to_array_node(scope, new_node_h, shift)?;
                Ok((array_node, true))
            } else {
                Ok((new_node_h, true))
            }
        } else {
            // Entry exists at this position
            let children = children_h.to_gc_handle();
            let stored_key = children.get_field(index * 2);
            let stored_value = children.get_field(index * 2 + 1);

            // Check if this is a sub-node or a leaf
            if BuiltInTypes::is_heap_pointer(stored_key) {
                let maybe_node = GcHandle::from_tagged(stored_key);
                let maybe_type = maybe_node.get_type_id();

                if maybe_type == TYPE_ID_BITMAP_NODE
                    || maybe_type == TYPE_ID_ARRAY_NODE
                    || maybe_type == TYPE_ID_COLLISION_NODE
                {
                    // Recurse into sub-node
                    let sub_node_h = scope.alloc_handle(maybe_node);
                    let (new_sub_h, added) =
                        Self::assoc_node(scope, sub_node_h, shift + 5, hash, key_h, value_h)?;

                    // Create new children array with updated sub-node
                    let children = children_h.to_gc_handle();
                    let old_len = children.field_count();
                    let new_children_h = scope.allocate_typed(old_len, TYPE_ID_BITMAP_NODE)?;

                    // Bulk copy all fields
                    let children = children_h.to_gc_handle();
                    let new_children = new_children_h.to_gc_handle();
                    children.copy_fields_to(&new_children, old_len);

                    // Update the sub-node reference
                    let new_sub = new_sub_h.to_gc_handle();
                    let new_children = new_children_h.to_gc_handle();
                    // For sub-nodes, we store them directly, not as key-value pairs
                    new_children.set_field(index * 2, new_sub.as_tagged());
                    new_children.set_field(index * 2 + 1, BuiltInTypes::null_value() as usize);

                    // Create new bitmap node
                    let node = node_h.to_gc_handle();
                    let new_node_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;
                    let new_children = new_children_h.to_gc_handle();
                    let new_node = new_node_h.to_gc_handle();

                    new_node.set_field(BN_FIELD_BITMAP, node.get_field(BN_FIELD_BITMAP));
                    new_node.set_field(BN_FIELD_CHILDREN, new_children.as_tagged());

                    return Ok((new_node_h, added));
                }
            }

            // Leaf entry - check if same key
            let key = key_h.get();
            let stored_key_h = scope.alloc(stored_key);
            let stored_value_h = scope.alloc(stored_value);

            // We need to compare keys - read values first, then compare
            let stored_key_val = stored_key_h.get();
            let keys_equal = scope.runtime().equal(stored_key_val, key);

            if keys_equal {
                // Same key - update value
                let children = children_h.to_gc_handle();
                let old_len = children.field_count();
                let new_children_h = scope.allocate_typed(old_len, TYPE_ID_BITMAP_NODE)?;

                // Bulk copy all fields
                let children = children_h.to_gc_handle();
                let new_children = new_children_h.to_gc_handle();
                children.copy_fields_to(&new_children, old_len);

                // Update value
                let value = value_h.get();
                let new_children = new_children_h.to_gc_handle();
                new_children.set_field(index * 2 + 1, value);

                // Create new bitmap node
                let node = node_h.to_gc_handle();
                let new_node_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;
                let new_children = new_children_h.to_gc_handle();
                let new_node = new_node_h.to_gc_handle();

                new_node.set_field(BN_FIELD_BITMAP, node.get_field(BN_FIELD_BITMAP));
                new_node.set_field(BN_FIELD_CHILDREN, new_children.as_tagged());

                Ok((new_node_h, false))
            } else {
                // Different key - need to create sub-node or collision node
                let stored_key = stored_key_h.get();
                // Note: stored_value is accessed via stored_value_h in create_* functions

                let stored_hash = scope.runtime().hash_value(stored_key);

                if stored_hash == hash {
                    // Hash collision - create collision node
                    let collision_h = Self::create_collision_node(
                        scope,
                        hash,
                        stored_key_h,
                        stored_value_h,
                        key_h,
                        value_h,
                    )?;

                    // Update children to point to collision node
                    let children = children_h.to_gc_handle();
                    let old_len = children.field_count();
                    let new_children_h = scope.allocate_typed(old_len, TYPE_ID_BITMAP_NODE)?;

                    // Bulk copy all fields
                    let children = children_h.to_gc_handle();
                    let new_children = new_children_h.to_gc_handle();
                    children.copy_fields_to(&new_children, old_len);

                    let collision = collision_h.to_gc_handle();
                    let new_children = new_children_h.to_gc_handle();
                    new_children.set_field(index * 2, collision.as_tagged());
                    new_children.set_field(index * 2 + 1, BuiltInTypes::null_value() as usize);

                    let node = node_h.to_gc_handle();
                    let new_node_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;
                    let new_children = new_children_h.to_gc_handle();
                    let new_node = new_node_h.to_gc_handle();

                    new_node.set_field(BN_FIELD_BITMAP, node.get_field(BN_FIELD_BITMAP));
                    new_node.set_field(BN_FIELD_CHILDREN, new_children.as_tagged());

                    Ok((new_node_h, true))
                } else {
                    // Different hashes - create intermediate bitmap node
                    let sub_node_h = Self::create_two_entry_node(
                        scope,
                        shift + 5,
                        stored_key_h,
                        stored_value_h,
                        stored_hash,
                        key_h,
                        value_h,
                        hash,
                    )?;

                    // Update children to point to sub-node
                    let children = children_h.to_gc_handle();
                    let old_len = children.field_count();
                    let new_children_h = scope.allocate_typed(old_len, TYPE_ID_BITMAP_NODE)?;

                    // Bulk copy all fields
                    let children = children_h.to_gc_handle();
                    let new_children = new_children_h.to_gc_handle();
                    children.copy_fields_to(&new_children, old_len);

                    let sub_node = sub_node_h.to_gc_handle();
                    let new_children = new_children_h.to_gc_handle();
                    new_children.set_field(index * 2, sub_node.as_tagged());
                    new_children.set_field(index * 2 + 1, BuiltInTypes::null_value() as usize);

                    let node = node_h.to_gc_handle();
                    let new_node_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;
                    let new_children = new_children_h.to_gc_handle();
                    let new_node = new_node_h.to_gc_handle();

                    new_node.set_field(BN_FIELD_BITMAP, node.get_field(BN_FIELD_BITMAP));
                    new_node.set_field(BN_FIELD_CHILDREN, new_children.as_tagged());

                    Ok((new_node_h, true))
                }
            }
        }
    }

    /// Associate into an array node.
    ///
    /// In the Clojure-style layout, ArrayNode has 32 fixed slots containing only
    /// child nodes (no inline key/value). When inserting into an empty slot, we
    /// create a singleton BitmapNode to hold the key/value pair.
    fn assoc_array_node(
        scope: &mut HandleScope<'_>,
        node_h: Handle,
        shift: usize,
        hash: usize,
        key_h: Handle,
        value_h: Handle,
    ) -> Result<(Handle, bool), Box<dyn Error>> {
        let node = node_h.to_gc_handle();
        let old_count = BuiltInTypes::untag(node.get_field(AN_FIELD_COUNT));
        let children_ptr = node.get_field(AN_FIELD_CHILDREN);
        let children = GcHandle::from_tagged(children_ptr);
        let children_h = scope.alloc_handle(children);

        let index = (hash >> shift) & BRANCH_MASK;
        let children = children_h.to_gc_handle();
        let child_ptr = children.get_field(index);

        if child_ptr == BuiltInTypes::null_value() as usize {
            // Empty slot - create singleton BitmapNode with key/value
            let singleton_h =
                Self::create_singleton_bitmap_node(scope, key_h, value_h, hash, shift + 5)?;

            // Create new children array with the singleton
            let new_children_h = scope.allocate_typed(32, TYPE_ID_ARRAY_NODE)?;

            let children = children_h.to_gc_handle();
            let new_children = new_children_h.to_gc_handle();
            children.copy_fields_to(&new_children, 32);

            let singleton = singleton_h.to_gc_handle();
            let new_children = new_children_h.to_gc_handle();
            new_children.set_field(index, singleton.as_tagged());

            // Create new array node with incremented count
            let new_node_h = scope.allocate_typed(2, TYPE_ID_ARRAY_NODE)?;
            let new_children = new_children_h.to_gc_handle();
            let new_node = new_node_h.to_gc_handle();
            new_node.set_field(
                AN_FIELD_COUNT,
                BuiltInTypes::construct_int((old_count + 1) as isize) as usize,
            );
            new_node.set_field(AN_FIELD_CHILDREN, new_children.as_tagged());

            Ok((new_node_h, true))
        } else {
            // Non-empty slot - recurse into child node
            let child = GcHandle::from_tagged(child_ptr);
            let sub_node_h = scope.alloc_handle(child);
            let (new_sub_h, added) =
                Self::assoc_node(scope, sub_node_h, shift + 5, hash, key_h, value_h)?;

            // Create new children array with updated child
            let new_children_h = scope.allocate_typed(32, TYPE_ID_ARRAY_NODE)?;

            let children = children_h.to_gc_handle();
            let new_children = new_children_h.to_gc_handle();
            children.copy_fields_to(&new_children, 32);

            let new_sub = new_sub_h.to_gc_handle();
            let new_children = new_children_h.to_gc_handle();
            new_children.set_field(index, new_sub.as_tagged());

            // Create new array node (count unchanged since we're updating existing child)
            let new_node_h = scope.allocate_typed(2, TYPE_ID_ARRAY_NODE)?;
            let new_children = new_children_h.to_gc_handle();
            let new_node = new_node_h.to_gc_handle();
            new_node.set_field(
                AN_FIELD_COUNT,
                BuiltInTypes::construct_int(old_count as isize) as usize,
            );
            new_node.set_field(AN_FIELD_CHILDREN, new_children.as_tagged());

            Ok((new_node_h, added))
        }
    }

    /// Convert a BitmapNode to an ArrayNode.
    /// Called when bitmap node density exceeds BITMAP_TO_ARRAY_THRESHOLD.
    ///
    /// In the Clojure-style layout, ArrayNode stores only child nodes (no inline key/value).
    /// Leaf entries from the BitmapNode are wrapped in singleton BitmapNodes at shift + 5.
    fn bitmap_to_array_node(
        scope: &mut HandleScope<'_>,
        bitmap_node_h: Handle,
        shift: usize,
    ) -> Result<Handle, Box<dyn Error>> {
        let bitmap_node = bitmap_node_h.to_gc_handle();
        let bitmap = BuiltInTypes::untag(bitmap_node.get_field(BN_FIELD_BITMAP));
        let old_children_ptr = bitmap_node.get_field(BN_FIELD_CHILDREN);
        let old_children = GcHandle::from_tagged(old_children_ptr);
        let old_children_h = scope.alloc_handle(old_children);

        // Create new 32-slot children array (node slots only, no key-value pairs)
        let new_children_h = scope.allocate_typed(32, TYPE_ID_ARRAY_NODE)?;

        // Initialize all slots to null
        let new_children = new_children_h.to_gc_handle();
        for i in 0..32 {
            new_children.set_field(i, BuiltInTypes::null_value() as usize);
        }

        // Convert entries from bitmap node to array node
        // Count non-null children as we go
        let old_children = old_children_h.to_gc_handle();
        let mut src_index = 0;
        let mut child_count = 0usize;

        for bit_pos in 0..32 {
            if (bitmap & (1usize << bit_pos)) != 0 {
                let key = old_children.get_field(src_index * 2);
                let value = old_children.get_field(src_index * 2 + 1);

                // Check if this is a sub-node or a leaf entry
                let is_subnode = value == BuiltInTypes::null_value() as usize
                    && BuiltInTypes::is_heap_pointer(key)
                    && {
                        let maybe_node = GcHandle::from_tagged(key);
                        let t = maybe_node.get_type_id();
                        t == TYPE_ID_BITMAP_NODE
                            || t == TYPE_ID_ARRAY_NODE
                            || t == TYPE_ID_COLLISION_NODE
                    };

                if is_subnode {
                    // Sub-node: store directly in array slot
                    let new_children = new_children_h.to_gc_handle();
                    new_children.set_field(bit_pos, key);
                } else {
                    // Leaf entry: wrap in singleton BitmapNode at shift + 5
                    let key_h = scope.alloc(key);
                    let value_h = scope.alloc(value);

                    // Need to compute the hash to create the singleton at correct position
                    let key_val = key_h.get();
                    let hash = scope.runtime().hash_value(key_val);

                    let singleton_h =
                        Self::create_singleton_bitmap_node(scope, key_h, value_h, hash, shift + 5)?;
                    let singleton = singleton_h.to_gc_handle();
                    let new_children = new_children_h.to_gc_handle();
                    new_children.set_field(bit_pos, singleton.as_tagged());
                }

                child_count += 1;
                src_index += 1;
            }
        }

        // Create array node with 2 fields: count and children
        let array_node_h = scope.allocate_typed(2, TYPE_ID_ARRAY_NODE)?;
        let new_children = new_children_h.to_gc_handle();
        let array_node = array_node_h.to_gc_handle();
        array_node.set_field(
            AN_FIELD_COUNT,
            BuiltInTypes::construct_int(child_count as isize) as usize,
        );
        array_node.set_field(AN_FIELD_CHILDREN, new_children.as_tagged());

        Ok(array_node_h)
    }

    /// Convert an ArrayNode back to a BitmapNode.
    /// Called when array node density drops below ARRAY_TO_BITMAP_THRESHOLD.
    ///
    /// In the Clojure-style layout, ArrayNode stores only child nodes.
    /// Each child node becomes a sub-node entry (node_ptr, null) in the BitmapNode.
    #[allow(dead_code)]
    fn array_to_bitmap_node(
        scope: &mut HandleScope<'_>,
        array_node_h: Handle,
    ) -> Result<Handle, Box<dyn Error>> {
        let array_node = array_node_h.to_gc_handle();
        let old_children_ptr = array_node.get_field(AN_FIELD_CHILDREN);
        let old_children = GcHandle::from_tagged(old_children_ptr);
        let old_children_h = scope.alloc_handle(old_children);

        // Count non-null children and build bitmap
        let old_children = old_children_h.to_gc_handle();
        let mut bitmap = 0usize;
        let mut count = 0usize;
        for bit_pos in 0..32 {
            let child = old_children.get_field(bit_pos);
            if child != BuiltInTypes::null_value() as usize {
                bitmap |= 1usize << bit_pos;
                count += 1;
            }
        }

        // Create new children array with count * 2 slots (for key/value pairs)
        // Each child node becomes (node_ptr, null)
        let new_children_h = scope.allocate_typed(count * 2, TYPE_ID_BITMAP_NODE)?;

        // Copy non-null child nodes
        let old_children = old_children_h.to_gc_handle();
        let new_children = new_children_h.to_gc_handle();
        let mut dst_index = 0;
        for bit_pos in 0..32 {
            let child = old_children.get_field(bit_pos);
            if child != BuiltInTypes::null_value() as usize {
                // Store child node as sub-node entry (node_ptr, null)
                new_children.set_field(dst_index * 2, child);
                new_children.set_field(dst_index * 2 + 1, BuiltInTypes::null_value() as usize);
                dst_index += 1;
            }
        }

        // Create bitmap node
        let bitmap_node_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;
        let new_children = new_children_h.to_gc_handle();
        let bitmap_node = bitmap_node_h.to_gc_handle();
        bitmap_node.set_field(
            BN_FIELD_BITMAP,
            BuiltInTypes::construct_int(bitmap as isize) as usize,
        );
        bitmap_node.set_field(BN_FIELD_CHILDREN, new_children.as_tagged());

        Ok(bitmap_node_h)
    }

    /// Create a collision node with two key-value pairs.
    fn create_collision_node(
        scope: &mut HandleScope<'_>,
        hash: usize,
        key1_h: Handle,
        value1_h: Handle,
        key2_h: Handle,
        value2_h: Handle,
    ) -> Result<Handle, Box<dyn Error>> {
        // Create kv_array with 4 elements
        let kv_array_h = scope.allocate_typed(4, TYPE_ID_COLLISION_NODE)?;

        let key1 = key1_h.get();
        let value1 = value1_h.get();
        let key2 = key2_h.get();
        let value2 = value2_h.get();
        let kv_array = kv_array_h.to_gc_handle();

        kv_array.set_field(0, key1);
        kv_array.set_field(1, value1);
        kv_array.set_field(2, key2);
        kv_array.set_field(3, value2);

        // Create collision node
        let node_h = scope.allocate_typed(3, TYPE_ID_COLLISION_NODE)?;
        let kv_array = kv_array_h.to_gc_handle();
        let node = node_h.to_gc_handle();

        node.set_field(
            CN_FIELD_HASH,
            BuiltInTypes::construct_int(hash as isize) as usize,
        );
        node.set_field(CN_FIELD_COUNT, BuiltInTypes::construct_int(2) as usize);
        node.set_field(CN_FIELD_KV_ARRAY, kv_array.as_tagged());

        Ok(node_h)
    }

    /// Create a bitmap node with two entries at different hash positions.
    fn create_two_entry_node(
        scope: &mut HandleScope<'_>,
        shift: usize,
        key1_h: Handle,
        value1_h: Handle,
        hash1: usize,
        key2_h: Handle,
        value2_h: Handle,
        hash2: usize,
    ) -> Result<Handle, Box<dyn Error>> {
        let bit1 = 1usize << ((hash1 >> shift) & BRANCH_MASK);
        let bit2 = 1usize << ((hash2 >> shift) & BRANCH_MASK);

        if bit1 == bit2 {
            // Same position at this level - need to go deeper
            let sub_node_h = Self::create_two_entry_node(
                scope,
                shift + 5,
                key1_h,
                value1_h,
                hash1,
                key2_h,
                value2_h,
                hash2,
            )?;

            // Create children array with just the sub-node
            let children_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;

            let sub_node = sub_node_h.to_gc_handle();
            let children = children_h.to_gc_handle();
            children.set_field(0, sub_node.as_tagged());
            children.set_field(1, BuiltInTypes::null_value() as usize);

            // Create bitmap node
            let node_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;
            let children = children_h.to_gc_handle();
            let node = node_h.to_gc_handle();

            node.set_field(
                BN_FIELD_BITMAP,
                BuiltInTypes::construct_int(bit1 as isize) as usize,
            );
            node.set_field(BN_FIELD_CHILDREN, children.as_tagged());

            Ok(node_h)
        } else {
            // Different positions - create node with both entries
            let bitmap = bit1 | bit2;

            // Create children array with 4 elements (2 key-value pairs)
            let children_h = scope.allocate_typed(4, TYPE_ID_BITMAP_NODE)?;

            let key1 = key1_h.get();
            let value1 = value1_h.get();
            let key2 = key2_h.get();
            let value2 = value2_h.get();
            let children = children_h.to_gc_handle();

            if bit1 < bit2 {
                children.set_field(0, key1);
                children.set_field(1, value1);
                children.set_field(2, key2);
                children.set_field(3, value2);
            } else {
                children.set_field(0, key2);
                children.set_field(1, value2);
                children.set_field(2, key1);
                children.set_field(3, value1);
            }

            // Create bitmap node
            let node_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;
            let children = children_h.to_gc_handle();
            let node = node_h.to_gc_handle();

            node.set_field(
                BN_FIELD_BITMAP,
                BuiltInTypes::construct_int(bitmap as isize) as usize,
            );
            node.set_field(BN_FIELD_CHILDREN, children.as_tagged());

            Ok(node_h)
        }
    }

    /// Promote a collision node to a bitmap node when a different hash is inserted.
    ///
    /// This handles the case where:
    /// 1. A collision node exists with entries sharing hash H1
    /// 2. A new key with hash H2 (H1 != H2) needs to be inserted
    ///
    /// At the current shift level, H1 and H2 may map to:
    /// - Different bit positions: create bitmap with collision node + new entry
    /// - Same bit position: recurse deeper until they diverge
    fn promote_collision_to_bitmap(
        scope: &mut HandleScope<'_>,
        collision_node_h: Handle,
        shift: usize,
        new_hash: usize,
        new_key_h: Handle,
        new_value_h: Handle,
    ) -> Result<(Handle, bool), Box<dyn Error>> {
        let collision_node = collision_node_h.to_gc_handle();
        let collision_hash = BuiltInTypes::untag(collision_node.get_field(CN_FIELD_HASH));

        let bit1 = 1usize << ((collision_hash >> shift) & BRANCH_MASK);
        let bit2 = 1usize << ((new_hash >> shift) & BRANCH_MASK);

        if bit1 == bit2 {
            // Same position at this level - need to go deeper
            // Recursively create a sub-node that contains both the collision node
            // and the new entry at a deeper level
            let sub_node_h = Self::promote_collision_to_bitmap_recursive(
                scope,
                collision_node_h,
                collision_hash,
                shift + 5,
                new_hash,
                new_key_h,
                new_value_h,
            )?;

            // Create children array with just the sub-node
            let children_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;

            let sub_node = sub_node_h.to_gc_handle();
            let children = children_h.to_gc_handle();
            children.set_field(0, sub_node.as_tagged());
            children.set_field(1, BuiltInTypes::null_value() as usize);

            // Create bitmap node with single bit
            let node_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;
            let children = children_h.to_gc_handle();
            let node = node_h.to_gc_handle();

            node.set_field(
                BN_FIELD_BITMAP,
                BuiltInTypes::construct_int(bit1 as isize) as usize,
            );
            node.set_field(BN_FIELD_CHILDREN, children.as_tagged());

            Ok((node_h, true))
        } else {
            // Different positions - create bitmap node with collision node and new entry
            let bitmap = bit1 | bit2;

            // Create children array with 4 elements (2 slots)
            let children_h = scope.allocate_typed(4, TYPE_ID_BITMAP_NODE)?;

            let collision_node = collision_node_h.to_gc_handle();
            let new_key = new_key_h.get();
            let new_value = new_value_h.get();
            let children = children_h.to_gc_handle();

            if bit1 < bit2 {
                // Collision node comes first
                children.set_field(0, collision_node.as_tagged());
                children.set_field(1, BuiltInTypes::null_value() as usize);
                children.set_field(2, new_key);
                children.set_field(3, new_value);
            } else {
                // New entry comes first
                children.set_field(0, new_key);
                children.set_field(1, new_value);
                children.set_field(2, collision_node.as_tagged());
                children.set_field(3, BuiltInTypes::null_value() as usize);
            }

            // Create bitmap node
            let node_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;
            let children = children_h.to_gc_handle();
            let node = node_h.to_gc_handle();

            node.set_field(
                BN_FIELD_BITMAP,
                BuiltInTypes::construct_int(bitmap as isize) as usize,
            );
            node.set_field(BN_FIELD_CHILDREN, children.as_tagged());

            Ok((node_h, true))
        }
    }

    /// Recursive helper for promote_collision_to_bitmap.
    /// Handles the case where collision hash and new hash share bits at current level.
    fn promote_collision_to_bitmap_recursive(
        scope: &mut HandleScope<'_>,
        collision_node_h: Handle,
        collision_hash: usize,
        shift: usize,
        new_hash: usize,
        new_key_h: Handle,
        new_value_h: Handle,
    ) -> Result<Handle, Box<dyn Error>> {
        let bit1 = 1usize << ((collision_hash >> shift) & BRANCH_MASK);
        let bit2 = 1usize << ((new_hash >> shift) & BRANCH_MASK);

        if bit1 == bit2 {
            // Still same position - go deeper
            let sub_node_h = Self::promote_collision_to_bitmap_recursive(
                scope,
                collision_node_h,
                collision_hash,
                shift + 5,
                new_hash,
                new_key_h,
                new_value_h,
            )?;

            // Create children array with just the sub-node
            let children_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;

            let sub_node = sub_node_h.to_gc_handle();
            let children = children_h.to_gc_handle();
            children.set_field(0, sub_node.as_tagged());
            children.set_field(1, BuiltInTypes::null_value() as usize);

            // Create bitmap node with single bit
            let node_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;
            let children = children_h.to_gc_handle();
            let node = node_h.to_gc_handle();

            node.set_field(
                BN_FIELD_BITMAP,
                BuiltInTypes::construct_int(bit1 as isize) as usize,
            );
            node.set_field(BN_FIELD_CHILDREN, children.as_tagged());

            Ok(node_h)
        } else {
            // Different positions - create bitmap with both
            let bitmap = bit1 | bit2;

            let children_h = scope.allocate_typed(4, TYPE_ID_BITMAP_NODE)?;

            let collision_node = collision_node_h.to_gc_handle();
            let new_key = new_key_h.get();
            let new_value = new_value_h.get();
            let children = children_h.to_gc_handle();

            if bit1 < bit2 {
                children.set_field(0, collision_node.as_tagged());
                children.set_field(1, BuiltInTypes::null_value() as usize);
                children.set_field(2, new_key);
                children.set_field(3, new_value);
            } else {
                children.set_field(0, new_key);
                children.set_field(1, new_value);
                children.set_field(2, collision_node.as_tagged());
                children.set_field(3, BuiltInTypes::null_value() as usize);
            }

            let node_h = scope.allocate_typed(2, TYPE_ID_BITMAP_NODE)?;
            let children = children_h.to_gc_handle();
            let node = node_h.to_gc_handle();

            node.set_field(
                BN_FIELD_BITMAP,
                BuiltInTypes::construct_int(bitmap as isize) as usize,
            );
            node.set_field(BN_FIELD_CHILDREN, children.as_tagged());

            Ok(node_h)
        }
    }

    /// Associate into a collision node.
    fn assoc_collision_node(
        scope: &mut HandleScope<'_>,
        node_h: Handle,
        shift: usize,
        hash: usize,
        key_h: Handle,
        value_h: Handle,
    ) -> Result<(Handle, bool), Box<dyn Error>> {
        let node = node_h.to_gc_handle();
        let stored_hash = BuiltInTypes::untag(node.get_field(CN_FIELD_HASH));

        if stored_hash != hash {
            // Different hash - promote collision node to bitmap node and insert new key
            Self::promote_collision_to_bitmap(scope, node_h, shift, hash, key_h, value_h)
        } else {
            // Same hash - update or add entry
            let count = BuiltInTypes::untag(node.get_field(CN_FIELD_COUNT));
            let kv_array_ptr = node.get_field(CN_FIELD_KV_ARRAY);
            let kv_array = GcHandle::from_tagged(kv_array_ptr);
            let kv_array_h = scope.alloc_handle(kv_array);

            // Check if key already exists
            let key = key_h.get();
            let kv_array = kv_array_h.to_gc_handle();

            for i in 0..count {
                let stored_key = kv_array.get_field(i * 2);
                let keys_equal = scope.runtime().equal(stored_key, key);

                if keys_equal {
                    // Update existing entry
                    let new_kv_array_h = scope.allocate_typed(count * 2, TYPE_ID_COLLISION_NODE)?;

                    let kv_array = kv_array_h.to_gc_handle();
                    let new_kv_array = new_kv_array_h.to_gc_handle();

                    for j in 0..count {
                        new_kv_array.set_field(j * 2, kv_array.get_field(j * 2));
                        if j == i {
                            let value = value_h.get();
                            let new_kv_array = new_kv_array_h.to_gc_handle();
                            new_kv_array.set_field(j * 2 + 1, value);
                        } else {
                            let new_kv_array = new_kv_array_h.to_gc_handle();
                            new_kv_array.set_field(j * 2 + 1, kv_array.get_field(j * 2 + 1));
                        }
                    }

                    // Create new collision node
                    let new_node_h = scope.allocate_typed(3, TYPE_ID_COLLISION_NODE)?;
                    let new_kv_array = new_kv_array_h.to_gc_handle();
                    let new_node = new_node_h.to_gc_handle();

                    new_node.set_field(
                        CN_FIELD_HASH,
                        BuiltInTypes::construct_int(hash as isize) as usize,
                    );
                    new_node.set_field(
                        CN_FIELD_COUNT,
                        BuiltInTypes::construct_int(count as isize) as usize,
                    );
                    new_node.set_field(CN_FIELD_KV_ARRAY, new_kv_array.as_tagged());

                    return Ok((new_node_h, false));
                }
            }

            // Key not found - add new entry
            let new_kv_array_h = scope.allocate_typed((count + 1) * 2, TYPE_ID_COLLISION_NODE)?;

            // Bulk copy existing entries
            let kv_array = kv_array_h.to_gc_handle();
            let new_kv_array = new_kv_array_h.to_gc_handle();
            kv_array.copy_fields_to(&new_kv_array, count * 2);

            // Add new entry
            let key = key_h.get();
            let value = value_h.get();
            let new_kv_array = new_kv_array_h.to_gc_handle();
            new_kv_array.set_field(count * 2, key);
            new_kv_array.set_field(count * 2 + 1, value);

            // Create new collision node
            let new_node_h = scope.allocate_typed(3, TYPE_ID_COLLISION_NODE)?;
            let new_kv_array = new_kv_array_h.to_gc_handle();
            let new_node = new_node_h.to_gc_handle();

            new_node.set_field(
                CN_FIELD_HASH,
                BuiltInTypes::construct_int(hash as isize) as usize,
            );
            new_node.set_field(
                CN_FIELD_COUNT,
                BuiltInTypes::construct_int((count + 1) as isize) as usize,
            );
            new_node.set_field(CN_FIELD_KV_ARRAY, new_kv_array.as_tagged());

            Ok((new_node_h, true))
        }
    }
}

#[cfg(test)]
mod tests {
    // Integration tests in resources/*.bg will cover this
}
