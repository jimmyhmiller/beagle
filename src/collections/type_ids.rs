//! Type ID constants for Rust-native persistent collections.
//!
//! These type IDs are used in the heap object headers to identify the type of
//! collection node. They are chosen to not conflict with existing type IDs:
//! - 1 = raw array
//! - 2 = string
//! - 3 = keyword
//! - User-defined structs use higher IDs assigned at runtime

/// Type ID for PersistentVec struct (4 fields: count, shift, root, tail)
pub const TYPE_ID_PERSISTENT_VEC: u8 = 20;

/// Type ID for persistent vector internal nodes (up to 32 slots)
pub const TYPE_ID_PERSISTENT_VEC_NODE: u8 = 21;

/// Type ID for PersistentMap struct (2 fields: count, root)
pub const TYPE_ID_PERSISTENT_MAP: u8 = 22;

/// Type ID for HAMT bitmap-indexed nodes (2 fields: bitmap, children)
pub const TYPE_ID_BITMAP_NODE: u8 = 23;

/// Type ID for HAMT collision nodes (3 fields: hash, count, kv_array)
pub const TYPE_ID_COLLISION_NODE: u8 = 24;

/// Type ID for HAMT array nodes (1 field: children array with 64 slots)
/// Used when BitmapNode density exceeds threshold (16+ entries)
pub const TYPE_ID_ARRAY_NODE: u8 = 25;
