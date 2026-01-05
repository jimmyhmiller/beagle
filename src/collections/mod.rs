//! Persistent collections for Beagle.
//!
//! This module provides persistent vector and map implementations that live on
//! Beagle's heap and fully participate in garbage collection. The API is designed
//! to be GC-safe, handling the fact that any allocation can trigger collection.
//!
//! # Usage
//!
//! ```beagle
//! import "beagle.collections" as c
//! let v = c/vec()
//! let v2 = c/vec-push(v, 42)
//! ```

mod gc_handle;
mod handle_arena;
mod persistent_map;
mod persistent_vec;
mod type_ids;

pub use gc_handle::GcHandle;
pub use handle_arena::{Handle, HandleScope};
pub use persistent_map::PersistentMap;
pub use persistent_vec::PersistentVec;
pub use type_ids::*;
