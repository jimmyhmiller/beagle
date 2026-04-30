//! Stable, resizable storage for namespace binding cells.
//!
//! Each top-level `let` / `fn` reserves an 8-byte cell here. The cell address
//! is stable for the lifetime of the runtime, which lets the JIT bake it in
//! as an immediate and read/write the binding with a single load or store —
//! no function call, no map lookup, no mutex.
//!
//! The space is chunked so it can grow without invalidating any previously
//! handed-out address: each chunk is its own mmap, and once allocated, a
//! chunk is never freed or moved. New chunks are appended on demand when
//! the current chunk fills up.
//!
//! Cells live outside the moving GC heap (the GC's "is this in my heap?"
//! predicate returns false for cell addresses). The GC instead treats each
//! cell as an external root: it reads the cell, traces through it, and on
//! a copying collector writes the forwarded pointer back into the cell
//! in place. Iteration is done via `for_each_cell`.
//!
//! Concurrency: reservation and chunk growth go through a `Mutex`. The hot
//! path — reading or writing a cell from JIT code — never touches the mutex.
//! Writes during normal execution and rewrites during GC are coordinated by
//! the existing stop-the-world pause.
//!
//! See `EternalSpace` for the same pattern applied to immutable float
//! constants.

use std::sync::Mutex;

use mmap_rs::{MmapMut, MmapOptions};

use crate::types::BuiltInTypes;

/// Default chunk size: 1 MiB = 128K cells per chunk. Chosen large enough
/// that reservation contention and chunk-table growth are negligible, and
/// small enough that even minimal programs don't waste a lot of address
/// space.
pub const DEFAULT_CHUNK_BYTES: usize = 1 << 20;

/// Size of one binding cell in bytes.
pub const CELL_BYTES: usize = 8;

struct Chunk {
    /// Owns the mapping; kept alive for the life of the chunk.
    _mmap: MmapMut,
    base: usize,
    size: usize,
}

impl Chunk {
    fn new(size: usize) -> Self {
        let mmap = MmapOptions::new(size)
            .expect("Failed to create mmap options for binding space chunk")
            .map_mut()
            .expect("Failed to map binding space chunk");
        let base = mmap.as_ptr() as usize;
        Self {
            _mmap: mmap,
            base,
            size,
        }
    }
}

struct State {
    chunks: Vec<Chunk>,
    /// Bytes used in the most recent chunk. Always a multiple of `CELL_BYTES`.
    used_in_last: usize,
}

pub struct BindingSpace {
    chunk_size: usize,
    state: Mutex<State>,
}

impl BindingSpace {
    pub fn new() -> Self {
        Self::with_chunk_size(DEFAULT_CHUNK_BYTES)
    }

    pub fn with_chunk_size(chunk_size: usize) -> Self {
        assert!(
            chunk_size % CELL_BYTES == 0,
            "chunk size must be a multiple of cell size",
        );
        assert!(chunk_size >= CELL_BYTES, "chunk size must hold at least one cell");
        Self {
            chunk_size,
            state: Mutex::new(State {
                chunks: Vec::new(),
                used_in_last: 0,
            }),
        }
    }

    /// Reserve a fresh cell, initialise it to the tagged null value, and
    /// return its absolute address. The address is stable for the lifetime
    /// of this `BindingSpace`.
    pub fn reserve(&self) -> usize {
        let mut state = self.state.lock().expect("BindingSpace mutex poisoned");

        if state.chunks.is_empty() || state.used_in_last + CELL_BYTES > self.chunk_size {
            state.chunks.push(Chunk::new(self.chunk_size));
            state.used_in_last = 0;
        }

        let chunk = state
            .chunks
            .last()
            .expect("BindingSpace just pushed a chunk but it's gone");
        let addr = chunk.base + state.used_in_last;
        state.used_in_last += CELL_BYTES;

        // Initialise to the language's null value so reads of an
        // unassigned slot don't observe uninitialised memory.
        unsafe {
            *(addr as *mut usize) = BuiltInTypes::null_value() as usize;
        }

        addr
    }

    /// Whether `ptr` falls within any chunk owned by this space.
    /// Used by the GC's "is this in my heap?" predicate so cells aren't
    /// confused for moving heap objects.
    pub fn contains(&self, ptr: usize) -> bool {
        let state = self.state.lock().expect("BindingSpace mutex poisoned");
        for chunk in &state.chunks {
            if ptr >= chunk.base && ptr < chunk.base + chunk.size {
                return true;
            }
        }
        false
    }

    /// Visit every reserved cell exactly once. The callback receives the
    /// cell's address as `*mut usize`; it may read and write through it.
    /// Intended for the GC root-scan path.
    pub fn for_each_cell<F: FnMut(*mut usize)>(&self, mut f: F) {
        let state = self.state.lock().expect("BindingSpace mutex poisoned");
        let last_idx = state.chunks.len().saturating_sub(1);
        for (i, chunk) in state.chunks.iter().enumerate() {
            let used = if i == last_idx {
                state.used_in_last
            } else {
                chunk.size
            };
            let mut p = chunk.base;
            let end = chunk.base + used;
            while p < end {
                f(p as *mut usize);
                p += CELL_BYTES;
            }
        }
    }

    /// Total number of cells currently reserved. Useful for diagnostics
    /// and for sizing the GC's extra-roots vector.
    pub fn cell_count(&self) -> usize {
        let state = self.state.lock().expect("BindingSpace mutex poisoned");
        let last_idx = state.chunks.len().saturating_sub(1);
        let mut total = 0usize;
        for (i, chunk) in state.chunks.iter().enumerate() {
            let used = if i == last_idx {
                state.used_in_last
            } else {
                chunk.size
            };
            total += used / CELL_BYTES;
        }
        total
    }
}

impl Default for BindingSpace {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: every chunk's `MmapMut` is owned by exactly one `BindingSpace`,
// and structural mutation (chunk creation, cursor bump) is gated by the
// inner `Mutex`. Cell-content reads and writes go through raw pointers and
// are coordinated by the runtime's stop-the-world GC pause and by the
// fact that JIT-published code paths flush the icache before becoming
// reachable.
unsafe impl Send for BindingSpace {}
unsafe impl Sync for BindingSpace {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_are_unique_and_initialised_to_null() {
        let space = BindingSpace::with_chunk_size(64);
        let mut addrs = Vec::new();
        for _ in 0..4 {
            let a = space.reserve();
            addrs.push(a);
            unsafe {
                assert_eq!(*(a as *const usize), BuiltInTypes::null_value() as usize);
            }
        }
        addrs.sort();
        addrs.dedup();
        assert_eq!(addrs.len(), 4);
    }

    #[test]
    fn grows_across_chunks_with_stable_addresses() {
        // 16 bytes / chunk = exactly two cells per chunk.
        let space = BindingSpace::with_chunk_size(16);
        let a = space.reserve();
        let b = space.reserve();
        let c = space.reserve(); // forces a new chunk
        let d = space.reserve();

        // a/b sit in the first chunk, c/d in the second; the addresses
        // of a and b must NOT have moved.
        unsafe {
            *(a as *mut usize) = 0xAAAA;
            *(b as *mut usize) = 0xBBBB;
            *(c as *mut usize) = 0xCCCC;
            *(d as *mut usize) = 0xDDDD;
        }
        let _ = space.reserve();
        let _ = space.reserve();
        unsafe {
            assert_eq!(*(a as *const usize), 0xAAAA);
            assert_eq!(*(b as *const usize), 0xBBBB);
            assert_eq!(*(c as *const usize), 0xCCCC);
            assert_eq!(*(d as *const usize), 0xDDDD);
        }
    }

    #[test]
    fn for_each_cell_visits_everything_in_order() {
        let space = BindingSpace::with_chunk_size(16);
        let addrs: Vec<usize> = (0..5).map(|_| space.reserve()).collect();

        let mut visited = Vec::new();
        space.for_each_cell(|p| visited.push(p as usize));

        assert_eq!(visited, addrs);
        assert_eq!(space.cell_count(), 5);
    }

    #[test]
    fn contains_only_returns_true_for_owned_addresses() {
        let space = BindingSpace::with_chunk_size(64);
        let a = space.reserve();
        assert!(space.contains(a));
        assert!(!space.contains(0xDEAD_BEEF));
        let stack_local = 0u64;
        let stack_addr = &stack_local as *const _ as usize;
        assert!(!space.contains(stack_addr));
    }
}
