//! Append-only, never-reallocating vector for the function table.
//!
//! WHY: `Runtime::functions` was a `Vec<Function>` read lock-free by
//! `get_function_by_pointer` (closure dispatch / `make_closure`) on mutator
//! threads while the compiler thread appended to it. A `Vec` push that crosses a
//! capacity boundary REALLOCATES — moving the backing buffer and dangling any
//! concurrent reader (a rare-but-real use-after-free; the "mode-1" latent race,
//! see `docs/CONCURRENT_REDEFINE_OPEN_ISSUES.md`). The fix is a container whose
//! already-published elements NEVER move.
//!
//! DESIGN: a fixed array of `AtomicPtr` chunk slots; each chunk is a heap block
//! of `CHUNK_SIZE` elements that, once allocated, is never moved or freed (the
//! table only ever grows — same invariant `get_function_by_pointer` already
//! relied on). Growth allocates a NEW chunk and atomically publishes its pointer;
//! it never touches existing chunks. So a reader holding `&element` (or computing
//! one from a stable index) is never invalidated by a concurrent grow.
//!
//! CONCURRENCY CONTRACT:
//! - WRITES (`push`, `get_mut`, `iter_mut`) take `&mut self` — serialized by the
//!   existing `&mut Runtime` discipline (one writer at a time).
//! - READS (`get`, `index`, `iter`, `len`) take `&self` and are lock-free; they
//!   may run on other threads concurrently with a `&mut` writer (the runtime's
//!   pre-existing unsafe-alias pattern). Soundness comes from the atomics:
//!   `push` fully initializes the element in its chunk, then RELEASE-stores the
//!   new chunk pointer (if any) and RELEASE-increments `len`; readers ACQUIRE-load
//!   `len` and the chunk pointer, so they never observe a half-initialized
//!   element or a not-yet-published chunk. Reads beyond `len` return `None`.
//! - There is NO reclamation (and so no leak/epoch/hazard complexity): chunks
//!   live for the process, exactly as the old `Vec`'s elements did.

use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

const CHUNK_SIZE: usize = 1024;
/// CHUNK_SIZE * MAX_CHUNKS = 8M elements — far beyond any real function count;
/// `push` panics with a clear message rather than silently wrapping if exceeded.
const MAX_CHUNKS: usize = 8192;

pub struct AppendOnlyChunked<T> {
    /// One slot per potential chunk; `null` until the chunk is allocated. The
    /// array itself is heap-allocated ONCE at construction and never moves.
    chunks: Box<[AtomicPtr<T>]>,
    /// Number of published elements. The single source of truth for bounds.
    len: AtomicUsize,
}

unsafe impl<T: Send> Send for AppendOnlyChunked<T> {}
unsafe impl<T: Sync> Sync for AppendOnlyChunked<T> {}

impl<T> Default for AppendOnlyChunked<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AppendOnlyChunked<T> {
    pub fn new() -> Self {
        let mut v = Vec::with_capacity(MAX_CHUNKS);
        for _ in 0..MAX_CHUNKS {
            v.push(AtomicPtr::new(std::ptr::null_mut()));
        }
        Self {
            chunks: v.into_boxed_slice(),
            len: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Allocate a fresh chunk of `CHUNK_SIZE` default-less slots. We allocate a
    /// `Box<[T]>`-shaped block via a `Vec` of uninitialized capacity is unsafe;
    /// instead we require the chunk be filled lazily. To keep it simple and safe
    /// we allocate the chunk as a raw array of `T` written element-by-element by
    /// `push` (only the slot at the live frontier is ever written before being
    /// published), so a chunk only ever holds initialized elements below `len`.
    fn chunk_ptr(&self, chunk_idx: usize, order: Ordering) -> *mut T {
        self.chunks[chunk_idx].load(order)
    }

    /// Append `value`. `&mut self` => exclusive writer. Initializes the element
    /// in its (possibly newly-allocated) chunk, then publishes it by RELEASE-
    /// incrementing `len`, so concurrent lock-free readers see a fully-formed
    /// element or nothing.
    pub fn push(&mut self, value: T) -> usize {
        let idx = self.len.load(Ordering::Relaxed); // exclusive writer: Relaxed ok
        let chunk_idx = idx / CHUNK_SIZE;
        let in_chunk = idx % CHUNK_SIZE;
        assert!(
            chunk_idx < MAX_CHUNKS,
            "AppendOnlyChunked: exceeded {} elements (raise MAX_CHUNKS)",
            CHUNK_SIZE * MAX_CHUNKS
        );

        // Ensure the chunk exists. Only the exclusive writer allocates, so no CAS
        // race; publish with Release so a reader that later sees `len` past this
        // index also sees the chunk pointer.
        let mut chunk = self.chunks[chunk_idx].load(Ordering::Acquire);
        if chunk.is_null() {
            // Allocate an uninitialized chunk of CHUNK_SIZE Ts. We only ever read
            // slots strictly below `len`, all of which have been written, so the
            // uninitialized tail is never observed.
            let layout = std::alloc::Layout::array::<T>(CHUNK_SIZE).unwrap();
            // SAFETY: layout is non-zero (CHUNK_SIZE>0); we publish before any
            // reader can index into this chunk (gated on `len`).
            chunk = unsafe { std::alloc::alloc(layout) as *mut T };
            assert!(!chunk.is_null(), "AppendOnlyChunked: chunk alloc failed");
            self.chunks[chunk_idx].store(chunk, Ordering::Release);
        }

        // SAFETY: `chunk` is valid for CHUNK_SIZE Ts; `in_chunk < CHUNK_SIZE`;
        // this slot is at the frontier (== current len) so no reader can be
        // reading it yet (readers are bounded by `len`, which we bump AFTER).
        unsafe {
            std::ptr::write(chunk.add(in_chunk), value);
        }

        // Publish: a reader that Acquire-loads this new `len` is guaranteed to
        // see the element write above and the chunk Release-store.
        self.len.store(idx + 1, Ordering::Release);
        idx
    }

    #[inline]
    pub fn get(&self, idx: usize) -> Option<&T> {
        if idx >= self.len.load(Ordering::Acquire) {
            return None;
        }
        let chunk = self.chunk_ptr(idx / CHUNK_SIZE, Ordering::Acquire);
        if chunk.is_null() {
            return None;
        }
        // SAFETY: idx < len => the element was fully written and published; the
        // chunk pointer is non-null and the element is initialized and immortal.
        Some(unsafe { &*chunk.add(idx % CHUNK_SIZE) })
    }

    /// Mutable access. `&mut self` => exclusive; safe to hand out `&mut T` because
    /// no concurrent reader exists under the `&mut`.
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        if idx >= self.len.load(Ordering::Relaxed) {
            return None;
        }
        let chunk = self.chunk_ptr(idx / CHUNK_SIZE, Ordering::Relaxed);
        if chunk.is_null() {
            return None;
        }
        // SAFETY: exclusive `&mut self`; idx < len; element initialized.
        Some(unsafe { &mut *chunk.add(idx % CHUNK_SIZE) })
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let len = self.len.load(Ordering::Acquire);
        (0..len).map(move |i| self.get(i).expect("index < len is always present"))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        let len = self.len.load(Ordering::Relaxed);
        // Build the list of raw element pointers up front (exclusive &mut, so no
        // aliasing concern), then hand out &mut — avoids borrowck fighting the
        // per-element chunk lookups.
        let mut ptrs: Vec<*mut T> = Vec::with_capacity(len);
        for i in 0..len {
            let chunk = self.chunk_ptr(i / CHUNK_SIZE, Ordering::Relaxed);
            debug_assert!(!chunk.is_null());
            ptrs.push(unsafe { chunk.add(i % CHUNK_SIZE) });
        }
        ptrs.into_iter().map(|p| unsafe { &mut *p })
    }
}

impl<T> std::ops::Index<usize> for AppendOnlyChunked<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: usize) -> &T {
        self.get(idx)
            .expect("AppendOnlyChunked: index out of bounds")
    }
}

impl<T> Drop for AppendOnlyChunked<T> {
    fn drop(&mut self) {
        let len = self.len.load(Ordering::Relaxed);
        for chunk_idx in 0..MAX_CHUNKS {
            let chunk = self.chunks[chunk_idx].load(Ordering::Relaxed);
            if chunk.is_null() {
                break; // chunks are allocated in order; first null => no more
            }
            // Drop the initialized elements in this chunk (those below len).
            let base = chunk_idx * CHUNK_SIZE;
            let count = (len.saturating_sub(base)).min(CHUNK_SIZE);
            for in_chunk in 0..count {
                unsafe { std::ptr::drop_in_place(chunk.add(in_chunk)) };
            }
            let layout = std::alloc::Layout::array::<T>(CHUNK_SIZE).unwrap();
            unsafe { std::alloc::dealloc(chunk as *mut u8, layout) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_get_index_len_across_chunks() {
        let mut v: AppendOnlyChunked<usize> = AppendOnlyChunked::new();
        assert!(v.is_empty());
        // Cross several chunk boundaries.
        let n = CHUNK_SIZE * 3 + 7;
        for i in 0..n {
            assert_eq!(v.push(i * 2), i);
        }
        assert_eq!(v.len(), n);
        for i in 0..n {
            assert_eq!(*v.get(i).unwrap(), i * 2);
            assert_eq!(v[i], i * 2);
        }
        assert!(v.get(n).is_none());
    }

    #[test]
    fn iter_and_iter_mut() {
        let mut v: AppendOnlyChunked<i64> = AppendOnlyChunked::new();
        for i in 0..(CHUNK_SIZE + 5) {
            v.push(i as i64);
        }
        let sum: i64 = v.iter().sum();
        let expected: i64 = (0..(CHUNK_SIZE as i64 + 5)).sum();
        assert_eq!(sum, expected);
        for x in v.iter_mut() {
            *x += 1;
        }
        assert_eq!(*v.get(0).unwrap(), 1);
        assert_eq!(*v.get(CHUNK_SIZE).unwrap(), CHUNK_SIZE as i64 + 1);
    }

    #[test]
    fn stable_addresses_across_grow() {
        // The whole point: an element's address never changes as the table grows.
        let mut v: AppendOnlyChunked<u64> = AppendOnlyChunked::new();
        v.push(42);
        let addr0 = v.get(0).unwrap() as *const u64 as usize;
        for i in 1..(CHUNK_SIZE * 4) {
            v.push(i as u64);
        }
        let addr0_after = v.get(0).unwrap() as *const u64 as usize;
        assert_eq!(addr0, addr0_after, "element 0 must not move as table grows");
    }
}
