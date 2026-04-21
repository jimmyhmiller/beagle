//! Single chokepoint for off-heap memory owned by Beagle but held outside the GC heap.
//!
//! Every persistent FFI-adjacent allocation (Buffer, Cell, TypedArray, C string arrays
//! constructed for FFI calls) goes through this module. The GC's finalizer for any of
//! those objects calls `free` here, closing the loop.
//!
//! Backed by `std::alloc` — the same allocator `Vec` uses — so there's no behavioral
//! change from the previous `Vec + mem::forget` pattern, just a single place that owns
//! the bookkeeping.
//!
//! In debug builds we track outstanding bytes and live allocation count so tests can
//! assert the GC actually reclaims what it's supposed to.
use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

// These counters are always compiled in — the release overhead is a single
// Relaxed atomic op per allocation, and tests run under --release.
static OUTSTANDING_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_FREES: AtomicUsize = AtomicUsize::new(0);

/// Minimum alignment for native allocations. Covers all primitive types
/// Beagle passes to C, including f64 and u64.
const MIN_ALIGN: usize = 8;

fn layout_for(size: usize) -> Layout {
    // size == 0 allocations are still legal; we bump to 1 so std::alloc doesn't
    // see a zero-sized layout (which has special rules we don't want to deal with).
    let size = size.max(1);
    Layout::from_size_align(size, MIN_ALIGN).expect("native_memory: invalid layout")
}

/// Allocate `size` bytes of zeroed off-heap memory.
///
/// Returns a non-null pointer. Aborts on OOM — matches `Vec`'s behavior on
/// allocator failure, and callers upstream have no recovery path anyway.
pub fn alloc_zeroed(size: usize) -> NonNull<u8> {
    let layout = layout_for(size);
    let ptr = unsafe { alloc::alloc_zeroed(layout) };
    let ptr = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));

    OUTSTANDING_BYTES.fetch_add(size, Ordering::Relaxed);
    LIVE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);

    ptr
}

/// Free a pointer previously returned by `alloc_zeroed` or `realloc`.
///
/// `size` must match the size originally requested. Beagle's finalizable FFI
/// structs store size in field 1 for this reason.
///
/// Safety: `ptr` must have come from this module's allocator and must not be
/// used after this call.
pub unsafe fn free(ptr: NonNull<u8>, size: usize) {
    let layout = layout_for(size);
    unsafe { alloc::dealloc(ptr.as_ptr(), layout) };

    OUTSTANDING_BYTES.fetch_sub(size, Ordering::Relaxed);
    LIVE_ALLOCATIONS.fetch_sub(1, Ordering::Relaxed);
    TOTAL_FREES.fetch_add(1, Ordering::Relaxed);
}

/// Reallocate `ptr` from `old_size` to `new_size`. New bytes are uninitialized.
///
/// Safety: `ptr` must have come from this module's allocator with size
/// `old_size` and must not be used after this call — use the returned pointer.
pub unsafe fn realloc(ptr: NonNull<u8>, old_size: usize, new_size: usize) -> NonNull<u8> {
    let old_layout = layout_for(old_size);
    let new_size_padded = new_size.max(1);
    let new_ptr = unsafe { alloc::realloc(ptr.as_ptr(), old_layout, new_size_padded) };
    let new_ptr =
        NonNull::new(new_ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout_for(new_size)));

    // Update by net delta; realloc is one allocation, not two.
    if new_size > old_size {
        OUTSTANDING_BYTES.fetch_add(new_size - old_size, Ordering::Relaxed);
    } else if old_size > new_size {
        OUTSTANDING_BYTES.fetch_sub(old_size - new_size, Ordering::Relaxed);
    }

    new_ptr
}

/// Total bytes currently held in off-heap allocations.
pub fn outstanding_bytes() -> usize {
    OUTSTANDING_BYTES.load(Ordering::Relaxed)
}

/// Number of live off-heap allocations.
pub fn live_allocations() -> usize {
    LIVE_ALLOCATIONS.load(Ordering::Relaxed)
}

/// Cumulative count of frees since startup. Used by tests to assert the
/// finalizer actually ran.
pub fn total_frees() -> usize {
    TOTAL_FREES.load(Ordering::Relaxed)
}
