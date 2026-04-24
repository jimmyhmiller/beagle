//! Eternal allocation region for compile-time constants.
//!
//! Objects allocated here never move and are never collected. Each GC's
//! "is this pointer in my heap?" guard returns false for eternal pointers,
//! so marking, tracing, and moving all skip them naturally.
//!
//! Only used for pointerless immutable values (currently: float literals).
//! Composite objects would require teaching the field-scanning code not to
//! recurse into eternal objects; not needed yet.

use std::collections::HashMap;
use std::sync::Mutex;

use mmap_rs::{MmapMut, MmapOptions};

use crate::types::{BuiltInTypes, Header};

/// Fixed-size region for immortal constants. Backed by a single mmap so the
/// address range is contiguous and stable for its lifetime.
pub struct EternalSpace {
    mmap: MmapMut,
    base: usize,
    size: usize,
    state: Mutex<State>,
}

struct State {
    offset: usize,
    float_dedup: HashMap<u64, usize>,
}

impl EternalSpace {
    pub fn new(size: usize) -> Self {
        let mmap = MmapOptions::new(size)
            .expect("Failed to create mmap options for eternal space")
            .map_mut()
            .expect("Failed to map eternal space");
        let base = mmap.as_ptr() as usize;
        Self {
            mmap,
            base,
            size,
            state: Mutex::new(State {
                offset: 0,
                float_dedup: HashMap::new(),
            }),
        }
    }

    /// Intern a float by its bit pattern. Returns a Float-tagged heap pointer
    /// suitable for use anywhere a Float value is expected, or `None` if the
    /// region is exhausted — callers should then emit an inline heap
    /// allocation as a fallback.
    pub fn intern_float(&self, bits: u64) -> Option<usize> {
        let mut state = self.state.lock().unwrap();
        if let Some(&tagged) = state.float_dedup.get(&bits) {
            return Some(tagged);
        }
        let untagged = self.alloc_locked(&mut state, 16)?;
        // Header matches `IR::write_small_object_header`: opaque single-word
        // payload, no children. Must round-trip through the normal header
        // layout so any debug path that inspects the object sees a valid
        // shape.
        let header = Header {
            type_id: 0,
            type_data: 0,
            size: 1,
            opaque: true,
            marked: false,
            large: false,
            type_flags: 0,
        };
        unsafe {
            *(untagged as *mut usize) = header.to_usize();
            *((untagged + 8) as *mut u64) = bits;
        }
        let tagged = BuiltInTypes::Float.tag(untagged as isize) as usize;
        state.float_dedup.insert(bits, tagged);
        Some(tagged)
    }

    fn alloc_locked(&self, state: &mut State, bytes: usize) -> Option<usize> {
        let aligned = (bytes + 7) & !7;
        if state.offset + aligned > self.size {
            return None;
        }
        let addr = self.base + state.offset;
        state.offset += aligned;
        Some(addr)
    }

    #[allow(dead_code)]
    pub fn contains(&self, ptr: usize) -> bool {
        ptr >= self.base && ptr < self.base + self.size
    }
}

// SAFETY: the mmap is owned by a single struct with all mutation gated by a
// Mutex; reads after publish are naturally ordered by the mutex and by the
// icache flush at JIT-code publish time.
unsafe impl Send for EternalSpace {}
unsafe impl Sync for EternalSpace {}

// Silence "mmap is never read" — the field exists solely to keep the mapping
// alive; the memory is accessed through `base`.
#[allow(dead_code)]
fn _assert_mmap_field_used(s: &EternalSpace) -> &MmapMut {
    &s.mmap
}
