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

/// Type ID for HAMT ArrayNode struct (2 fields: count, children_ptr)
/// Used when BitmapNode density exceeds threshold (16+ entries)
pub const TYPE_ID_ARRAY_NODE: u8 = 25;

/// Type ID for ArrayNode's 32-slot children array
/// Separate from TYPE_ID_ARRAY_NODE to catch bugs where children array
/// is confused with ArrayNode struct (both have same field semantics but different layout)
pub const TYPE_ID_ARRAY_NODE_CHILDREN: u8 = 27;

/// Type ID for Atom (1 field: value)
/// Used for atomic reference cells (thread-safe mutable cells)
pub const TYPE_ID_ATOM: u8 = 26;

/// Type ID for FunctionObject (1 field: function_pointer)
/// A callable heap object wrapping a function pointer.
/// Unlike closures, FunctionObjects have no free variables and don't receive self as arg0.
pub const TYPE_ID_FUNCTION_OBJECT: u8 = 10;

/// Type ID for PersistentSet struct (2 fields: count, root)
/// Backed by PersistentMap internally (keys are elements, values are true)
pub const TYPE_ID_PERSISTENT_SET: u8 = 28;

/// Type ID for MultiArityFunction (dispatch structure for multi-arity functions)
/// Layout: [num_arities] [entry0: arity, fn_ptr, is_variadic] [entry1: ...] ...
pub const TYPE_ID_MULTI_ARITY_FUNCTION: u8 = 29;

/// Type ID for Regex (compiled regular expression)
/// The heap object stores an index into Runtime::compiled_regexes
pub const TYPE_ID_REGEX: u8 = 30;
