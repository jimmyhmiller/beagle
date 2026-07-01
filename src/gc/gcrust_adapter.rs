//! Adapter that drives the vendored `gc-rust` collector (`src/gc/gcrust/`)
//! as a Beagle [`Allocator`].
//!
//! **Stage 5 — stop-the-world *generational* copying collector.** Layout:
//! a bump-allocated nursery (young gen) + two tenured semi-spaces (old gen)
//! + one card table per tenured space (old→young remembered set). New objects
//! go to the nursery; a **minor GC** promotes nursery survivors into the
//! tenured from-space (roots = Beagle roots + dirty cards); a **major GC**
//! evacuates everything live (nursery ∪ tenured-from) into tenured-to and
//! empties the nursery. Beagle already stops the world before calling
//! [`Allocator::gc`], so collection runs synchronously (no gc-rust safepoint
//! machinery — that is Phase 2). Algorithm mirrors gc-rust's `heap.rs`.
//!
//! The bridge pieces:
//! - [`BeaglePtrPolicy`] — Beagle's low-3-bit tag scheme (a heap reference is
//!   `(addr << 3) | tag`, tag ∈ {Float, Closure, HeapObject}) + tag-preserving
//!   relocation.
//! - [`BeagleObjectModel`] — Beagle's self-describing 8-byte header: size
//!   (`full_size`), traced slots (`num_traced_slots` + continuation segments),
//!   bit-3 forwarding, struct-migration-aware copy.
//! - [`GcRustHeap`] — the [`Allocator`] impl: nursery/tenured allocation, the
//!   root bridge, the card write barrier, and the minor/major collectors.
//!
//! See `docs/GCRUST_ADOPTION.md` for the roadmap (concurrent = Phase 2).

use std::cell::Cell;
use std::collections::HashMap;
use std::error::Error;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

use crate::builtins::reset_shift::ContinuationObject;
use crate::collections::TYPE_ID_CONTINUATION;
use crate::gc::gcrust::{BumpAllocator, CardTable, Compact, ObjectModel, PtrPolicy, RootSource};
use crate::gc::stack_walker::StackWalker;
use crate::gc::{AllocateAction, Allocator, AllocatorOptions};
use crate::types::{BuiltInTypes, Header, HeapObject, Word};

/// Low-3-bit tag mask.
const TAG_MASK: u64 = 0b111;

/// Per-collection flag: are struct-layout migrations pending? Set by `gc()`
/// before collecting so `copy_object` can gate migration on a cheap relaxed
/// load instead of the struct-registry RwLock per object.
static MIGRATE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// RAII timer for GC pause instrumentation. When `BEAGLE_GC_PAUSE` is set,
/// prints the stop-the-world work window for each collection phase. Used to
/// compare STW-major pauses against the two short concurrent pauses.
struct PauseGuard {
    label: &'static str,
    start: Option<std::time::Instant>,
}
impl PauseGuard {
    fn new(label: &'static str) -> Self {
        let start = std::env::var("BEAGLE_GC_PAUSE")
            .is_ok()
            .then(std::time::Instant::now);
        PauseGuard { label, start }
    }
}
impl Drop for PauseGuard {
    fn drop(&mut self) {
        if let Some(t) = self.start {
            eprintln!("[pause] {} {}us", self.label, t.elapsed().as_micros());
        }
    }
}

// ─── PtrPolicy ───────────────────────────────────────────────────────

/// Beagle's tagged-pointer scheme for the collector.
///
/// A slot's `u64` is a heap reference iff its tag is `Float`, `Closure`, or
/// `HeapObject`; the underlying address is `bits >> 3`. After the object moves,
/// the slot must be rewritten with the SAME tag, which [`reencode_ptr`] does.
///
/// [`reencode_ptr`]: PtrPolicy::reencode_ptr
pub struct BeaglePtrPolicy;

impl PtrPolicy for BeaglePtrPolicy {
    #[inline]
    fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
        if BuiltInTypes::is_heap_pointer(bits as usize) {
            // Real address = tagged >> 3 (Beagle stores addr << 3 | tag).
            Some((bits >> 3) as usize as *mut u8)
        } else {
            None
        }
    }

    #[inline]
    fn encode_ptr(ptr: *mut u8) -> u64 {
        // Fallback only; the hot path uses reencode_ptr to preserve the tag.
        (((ptr as usize) << 3) as u64) | BuiltInTypes::HeapObject.get_tag() as u64
    }

    #[inline]
    fn reencode_ptr(old_bits: u64, new_ptr: *mut u8) -> u64 {
        // new_ptr is the raw (8-byte-aligned) to-space address; restore the
        // original slot's tag.
        (((new_ptr as usize) << 3) as u64) | (old_bits & TAG_MASK)
    }
}

// ─── ObjectModel ─────────────────────────────────────────────────────

/// Beagle's self-describing header as an [`ObjectModel`]. No type table:
/// every object carries its size and (implicitly) its traced-slot count in
/// its 8-byte (or 16-byte, for large objects) header.
pub struct BeagleObjectModel;

impl ObjectModel for BeagleObjectModel {
    #[inline]
    unsafe fn object_layout(&self, obj: *const u8) -> (usize, usize) {
        let ho = HeapObject::from_untagged(obj);
        // full_size = header + fields; always an 8-byte multiple, 8-aligned.
        (ho.full_size(), 8)
    }

    #[inline]
    unsafe fn scan_slots(&self, obj: *mut u8, visitor: &mut dyn FnMut(*mut u64)) {
        let ho = HeapObject::from_untagged(obj as *const u8);
        let n = ho.num_traced_slots();
        for i in 0..n {
            visitor(ho.get_field_ptr(i) as *mut u64);
        }
        // A captured continuation additionally owns a detached stack segment
        // whose frames hold live heap pointers. Those slots aren't reachable
        // through the object's fields, so trace them explicitly — otherwise a
        // resumable continuation loses its roots. Mirrors the compacting GC.
        if ho.get_header().type_id == TYPE_ID_CONTINUATION {
            unsafe { scan_continuation_segment(obj, visitor) };
        }
    }

    unsafe fn copy_object(
        &self,
        old: *mut u8,
        tag: u8,
        alloc: &mut dyn FnMut(usize, usize) -> *mut u8,
    ) -> *mut u8 {
        let ho = HeapObject::from_untagged(old as *const u8);

        // Live struct migration: when a struct type has been redefined, its
        // pre-existing instances (type_id 0, not opaque, HeapObject-tagged —
        // NOT closures, which are also type_id 0 but Closure-tagged) are copied
        // into the new layout, remapping old fields and defaulting added ones.
        // MIGRATE_ACTIVE is a cheap per-collection flag set by `gc()` from
        // `has_pending_migrations()` — checking it (a relaxed load) per object
        // instead of taking the struct-registry RwLock per object is a big win
        // on allocation-heavy workloads with no pending migration.
        let closure_tag = BuiltInTypes::Closure.get_tag() as u8;
        if MIGRATE_ACTIVE.load(Ordering::Relaxed)
            && ho.get_type_id() == 0
            && tag != closure_tag
            && !ho.is_opaque_object()
        {
            let struct_id = ho.get_struct_id();
            let layout_version = ho.get_layout_version();
            let runtime = crate::get_runtime().get_mut();
            if let Some(plan) = runtime
                .structs
                .migration_plan_for(struct_id, layout_version)
            {
                // Resolve a GC-stable default (immediate or eternal-region,
                // never a fresh allocation) for each field new to this layout.
                let def = runtime.get_struct_by_id(struct_id);
                let new_field_defaults: Vec<usize> = plan
                    .field_map
                    .iter()
                    .enumerate()
                    .map(|(new_idx, mapping)| match mapping {
                        Some(_) => BuiltInTypes::null_value() as usize,
                        None => def
                            .as_ref()
                            .map(|d| runtime.field_default_value_at(d, new_idx))
                            .unwrap_or(BuiltInTypes::null_value() as usize),
                    })
                    .collect();

                let old_header = ho.get_header();
                let new_header = Header {
                    type_id: old_header.type_id,
                    type_data: old_header.type_data, // struct_id unchanged
                    size: plan.new_field_count as u16,
                    opaque: old_header.opaque,
                    marked: false,
                    large: false,
                    type_flags: plan.new_layout_version,
                };
                let total_bytes = 8 + plan.new_field_count * 8;
                let new = alloc(total_bytes, 8);
                unsafe {
                    *(new as *mut usize) = new_header.to_usize();
                    for (new_idx, mapping) in plan.field_map.iter().enumerate() {
                        let value = match mapping {
                            Some(old_idx) => ho.get_field(*old_idx),
                            None => new_field_defaults
                                .get(new_idx)
                                .copied()
                                .unwrap_or(BuiltInTypes::null_value() as usize),
                        };
                        *((new as *mut usize).add(1 + new_idx)) = value;
                    }
                }
                return new;
            }
        }

        // Default: verbatim copy (header + fields).
        let size = ho.full_size();
        let new = alloc(size, 8);
        unsafe { core::ptr::copy_nonoverlapping(old, new, size) };
        new
    }

    #[inline]
    unsafe fn read_forwarding(&self, obj: *const u8) -> Option<*mut u8> {
        let word = unsafe { *(obj as *const usize) };
        if Header::is_forwarding_bit_set(word) {
            // Stored as HeapObject-tagged (addr << 3 | tag) with bit 3 set.
            Some((Header::clear_forwarding_bit(word) >> 3) as *mut u8)
        } else {
            None
        }
    }

    #[inline]
    unsafe fn write_forwarding(&self, obj: *mut u8, new: *mut u8) {
        // Beagle's standard forwarding encoding: the new address in tagged form
        // with the bit-3 forwarding marker. Consistent with `Header` helpers and
        // `HeapObject::try_from_tagged`, so the finalizer sweep and any other
        // runtime code that inspects forwarded headers sees the expected shape.
        let tagged = BuiltInTypes::HeapObject.tag(new as isize) as usize;
        unsafe { *(obj as *mut usize) = Header::set_forwarding_bit(tagged) };
    }

    #[inline]
    unsafe fn is_valid(&self, obj: *const u8, bytes_available: usize) -> bool {
        // Beagle is mostly-precise: a traced slot may hold a non-pointer bit
        // pattern that happens to carry a heap tag. Validate that `obj` names a
        // real object whose full extent fits in `bytes_available` before we
        // treat it as one. Mirrors compacting::is_plausible_from_space_object.
        if bytes_available < 8 {
            return false;
        }
        let header_word = unsafe { *(obj as *const usize) };
        // A forwarding header (bit 3) marks an already-copied live object.
        if Header::is_forwarding_bit_set(header_word) {
            return true;
        }
        let header = Header::from_usize(header_word);
        let (header_size, field_words) = if header.large {
            if bytes_available < 16 {
                return false;
            }
            let field_words = unsafe { *((obj as *const usize).add(1)) };
            (16usize, field_words)
        } else {
            (8usize, header.size as usize)
        };
        let Some(fields_size) = field_words.checked_mul(8) else {
            return false;
        };
        let Some(full_size) = header_size.checked_add(fields_size) else {
            return false;
        };
        full_size <= bytes_available && full_size.is_multiple_of(8)
    }
}

/// Walk the GC roots living in a captured continuation's detached stack
/// segment, feeding each slot's ADDRESS to `visitor` (the collector rewrites
/// it in place — segment memory is stable across GC). `segment_gc_frame_info`
/// temporarily rebuilds absolute FP links for the walk; we always restore the
/// relative representation afterward. Mirrors `compacting::scan_continuation_segment`.
///
/// # Safety
/// `obj` must point at a live continuation object of this model.
unsafe fn scan_continuation_segment(obj: *mut u8, visitor: &mut dyn FnMut(*mut u64)) {
    let ho = HeapObject::from_untagged(obj as *const u8);
    let Some(cont) = ContinuationObject::from_heap_object(ho) else {
        return;
    };
    let Some((base, top, gc_frame_top)) = cont.segment_gc_frame_info() else {
        return;
    };
    if gc_frame_top >= base && gc_frame_top < top {
        StackWalker::walk_segment_gc_roots(gc_frame_top, base, top, |slot_addr, _val| {
            visitor(slot_addr as *mut u64);
        });
    }
    // Restore position-independent (relative) FP links now that scanning is done.
    cont.make_fp_links_relative_again();
}

// ─── Root bridge ─────────────────────────────────────────────────────

/// Bridges Beagle's two root sets into a single [`RootSource`] the collector
/// consumes:
/// - `gc_frame_tops`: one GC-frame-chain head per thread, walked precisely by
///   [`StackWalker`] (yields each live heap-pointer slot's ADDRESS).
/// - `extra_roots`: `(slot_address, value)` pairs — per-thread handle/shadow
///   stacks, head-block chains, and namespace binding cells. The collector
///   rewrites the slot in place, so moving works transparently.
struct BeagleRoots<'a> {
    gc_frame_tops: &'a [usize],
    extra_roots: &'a [(*mut usize, usize)],
}

impl RootSource for BeagleRoots<'_> {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for &top in self.gc_frame_tops {
            StackWalker::walk_stack_roots(top, |slot_addr, _slot_value| {
                visitor(slot_addr as *mut u64);
            });
        }
        for &(slot, _value) in self.extra_roots {
            visitor(slot as *mut u64);
        }
    }
}

// ─── GcRustHeap (generational) ───────────────────────────────────────

/// Nursery (young gen) size in bytes. Override with `BEAGLE_GCRUST_NURSERY_MB`.
const DEFAULT_NURSERY_BYTES: usize = 64 * 1024 * 1024;
/// Size (bytes) of EACH tenured semi-space. Override with `BEAGLE_GCRUST_SPACE_MB`.
const DEFAULT_TENURED_BYTES: usize = 256 * 1024 * 1024;
/// Run a major (full) collection every Nth `gc()` to reclaim tenured garbage.
const FULL_GC_FREQUENCY: usize = 64;
/// Default thread-local allocation buffer size. Each mutator thread bumps a
/// private slice this big lock-free, taking the alloc mutex only to refill.
/// Override with `BEAGLE_GCRUST_TLAB_KB`.
const DEFAULT_TLAB_BYTES: usize = 256 * 1024;

fn tlab_bytes() -> usize {
    static TLAB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TLAB.get_or_init(|| {
        std::env::var("BEAGLE_GCRUST_TLAB_KB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|kb| kb * 1024)
            .unwrap_or(DEFAULT_TLAB_BYTES)
    })
}

fn env_bytes(var: &str, default_bytes: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(default_bytes)
}

#[inline]
fn header_bytes(words: usize) -> usize {
    if words > Header::MAX_INLINE_SIZE {
        16
    } else {
        8
    }
}

/// Beagle [`Allocator`] backed by the vendored gc-rust primitives, arranged as
/// a stop-the-world generational copying collector (see the module docs).
pub struct GcRustHeap {
    /// Young generation — new allocations land here.
    nursery: BumpAllocator,
    /// Old generation: two semi-spaces; `tenured[from_idx]` is live.
    tenured: [BumpAllocator; 2],
    /// One card table per tenured space (`card_tables[i]` covers `tenured[i]`),
    /// recording old→young pointer writes.
    card_tables: [CardTable; 2],
    from_idx: Cell<usize>,
    /// Collection phase for concurrent GC (Phase 2): `PHASE_IDLE` (barriers
    /// off, the STW generational collector) or `PHASE_COPYING` (a concurrent
    /// major cycle is in flight — mutators must dirty-track their writes so the
    /// final pause can reconcile the copies). Idle everywhere until a
    /// concurrent cycle is driven; the STW generational path never sets it.
    gc_phase: AtomicU8,
    /// When set (`BEAGLE_GCRUST_TABLE=1`), major collections use the
    /// out-of-band side-table copy/repoint path (validated STW; the foundation
    /// for concurrent GC) instead of header-forwarding.
    use_table: bool,
    /// When set (`BEAGLE_GCRUST_CONCURRENT=1`), tenured-pressure major
    /// collections run concurrently (mutators keep running during the trace).
    concurrent_enabled: bool,
    /// Side forwarding table (`from_addr → to_addr`) for the in-flight
    /// concurrent cycle; `Some` between `begin_concurrent` and `finish_concurrent`.
    conc_table: Option<HashMap<usize, usize>>,
    /// Collection counter for the concurrent path's periodic-major schedule
    /// (separate from `gc_count`, which counts the minor/STW path).
    conc_count: usize,
    /// Untagged addresses of live objects whose type has a finalizer (FFI
    /// buffers/cells/typed-arrays). Tracked in a side list so a collection finds
    /// dead finalizables in O(finalizable count) — NOT by walking the whole heap
    /// (which was O(all objects) × a per-object registry hash lookup, the
    /// dominant cost on allocation-heavy workloads). `RefCell` for interior
    /// mutability from the `&self` collection paths (access is serialized by the
    /// runtime — the alloc mutex for registration, stop-the-world for sweeps).
    finalizable: std::cell::RefCell<Vec<usize>>,
    options: AllocatorOptions,
    gc_count: usize,
}

const PHASE_IDLE: u8 = 0;
const PHASE_COPYING: u8 = 1;

// Safe to share across threads: like Beagle's other GCs, all mutation is
// serialized by the runtime — the `MutexAllocator` wrapper for allocation and
// stop-the-world for collection.
unsafe impl Send for GcRustHeap {}
unsafe impl Sync for GcRustHeap {}

impl GcRustHeap {
    #[inline]
    fn from_idx(&self) -> usize {
        self.from_idx.get()
    }
    #[inline]
    fn tenured_from(&self) -> &BumpAllocator {
        &self.tenured[self.from_idx()]
    }
    #[inline]
    fn tenured_to(&self) -> &BumpAllocator {
        &self.tenured[1 - self.from_idx()]
    }

    /// Grow the spare (to) space if it can't hold the worst-case live set a
    /// major collection would copy into it (everything in the nursery + the
    /// tenured from-space). Reallocated to 2× the need so growth amortizes.
    /// The spare is empty between collections, so replacing it is safe. This is
    /// how the copying heap grows — call before any major/concurrent copy.
    fn ensure_to_space_fits(&mut self) {
        let need = self.nursery.used() + self.tenured_from().used();
        let to = 1 - self.from_idx();
        if self.tenured[to].size() < need {
            let new_size = need
                .saturating_mul(2)
                .max(self.tenured[to].size().saturating_mul(2));
            self.tenured[to] = BumpAllocator::new::<Compact>(new_size);
            self.card_tables[to] = CardTable::new(self.tenured[to].base(), self.tenured[to].size());
        }
    }

    /// True while a concurrent major cycle is copying — mutators must
    /// dirty-track pointer writes so the final pause can reconcile the copies.
    /// A single relaxed load; the STW happens-before around phase flips makes
    /// this sufficient (matches gc-rust's `barriers_active`).
    #[inline]
    fn barriers_active(&self) -> bool {
        self.gc_phase.load(Ordering::Relaxed) != PHASE_IDLE
    }

    /// True if `addr` is 8-aligned and inside the ALLOCATED region of `region`;
    /// if so also return the bytes remaining from `addr` to the region's used
    /// frontier (for the object-fits plausibility check).
    #[inline]
    fn within(region: &BumpAllocator, addr: usize) -> Option<usize> {
        let base = region.base() as usize;
        let end = base + region.used();
        if addr >= base && addr < end && addr.is_multiple_of(8) {
            Some(end - addr)
        } else {
            None
        }
    }

    /// Allocate `words` fields (+ header) in `region`, zeroed, with the header
    /// written and Float marked opaque. Returns the raw untagged pointer, or
    /// None if the region is exhausted.
    fn alloc_in(region: &BumpAllocator, words: usize, kind: BuiltInTypes) -> Option<*const u8> {
        let total = words * 8 + header_bytes(words);
        let ptr = region.alloc_raw(total, 8);
        if ptr.is_null() {
            return None;
        }
        unsafe {
            std::ptr::write_bytes(ptr, 0, total);
            let mut ho = HeapObject::from_untagged(ptr as *const u8);
            ho.write_header(Word::from_word(words));
            if kind == BuiltInTypes::Float {
                *(ptr as *mut usize) |= Header::OPAQUE_BIT_MASK;
            }
        }
        Some(ptr as *const u8)
    }

    // ── Minor GC ─────────────────────────────────────────────────────

    /// Promote a nursery object referenced by `slot` into the tenured
    /// from-space (leaving tenured/immediate references untouched), then rewrite
    /// the slot in place (preserving its tag).
    unsafe fn promote_slot(&self, model: &BeagleObjectModel, slot: *mut u64) {
        let bits = unsafe { *slot };
        let Some(ptr) = BeaglePtrPolicy::try_decode_ptr(bits) else {
            return;
        };
        let Some(avail) = Self::within(&self.nursery, ptr as usize) else {
            return; // not a nursery object (tenured, or a non-pointer bit pattern)
        };
        if !unsafe { model.is_valid(ptr, avail) } {
            return;
        }
        let new = unsafe { self.promote_or_forward(model, ptr, (bits & 0b111) as u8) };
        unsafe { *slot = BeaglePtrPolicy::reencode_ptr(bits, new) };
    }

    /// If `slot` names a pre-existing tenured (old-gen) object, scan that
    /// object's fields ONE level for young references and promote them
    /// (updating the old object's slots in place). This is the second old→young
    /// root source alongside dirty cards: an old object directly on the stack
    /// may hold a young pointer whose card was never dirtied (e.g. a store not
    /// routed through the write barrier during construction). Mirrors
    /// `generational::process_old_gen_object`. The promoted young objects land
    /// in tenured-from and get transitively scanned by the Cheney phase.
    unsafe fn scan_old_gen_root(&self, model: &BeagleObjectModel, slot: *mut u64) {
        let bits = unsafe { *slot };
        let Some(ptr) = BeaglePtrPolicy::try_decode_ptr(bits) else {
            return;
        };
        let Some(avail) = Self::within(self.tenured_from(), ptr as usize) else {
            return; // not an old-gen object
        };
        if !unsafe { model.is_valid(ptr, avail) } {
            return;
        }
        unsafe { model.scan_slots(ptr, &mut |s| self.promote_slot(model, s)) };
    }

    unsafe fn promote_or_forward(
        &self,
        model: &BeagleObjectModel,
        old: *mut u8,
        tag: u8,
    ) -> *mut u8 {
        if let Some(fwd) = unsafe { model.read_forwarding(old) } {
            return fwd;
        }
        let dest = self.tenured_from();
        let new = unsafe {
            model.copy_object(old, tag, &mut |size, align| {
                let p = dest.alloc_raw(size, align);
                assert!(!p.is_null(), "tenured exhausted during minor-GC promotion");
                p
            })
        };
        unsafe { model.write_forwarding(old, new) };
        new
    }

    /// Scan pre-existing tenured objects reachable via dirty cards for old→young
    /// pointers, promoting the young targets. `tenured_used_limit` bounds the
    /// pre-existing region (objects promoted this cycle are handled by the
    /// Cheney phase instead).
    unsafe fn scan_dirty_cards(&self, model: &BeagleObjectModel, tenured_used_limit: usize) {
        let card_table = &self.card_tables[self.from_idx()];
        let tenured = self.tenured_from();
        // Fast out: with no dirty cards there are no old→young pointers to
        // find, so skip building the object-start index entirely. That index
        // is O(tenured), so building it every minor when nothing is dirty would
        // make minor GC O(tenured) — defeating the generational point.
        let dirty: Vec<(usize, *const u8)> = card_table.iter_dirty().collect();
        if dirty.is_empty() {
            return;
        }
        let obj_starts = unsafe {
            self.build_object_start_index(model, tenured, tenured_used_limit, card_table)
        };
        let card_size = card_table.card_size();
        for (card_idx, card_addr) in dirty {
            let Some(&start) = obj_starts.get(card_idx) else {
                continue;
            };
            if start == usize::MAX {
                continue;
            }
            unsafe {
                self.scan_card(
                    model,
                    tenured,
                    tenured_used_limit,
                    card_addr as usize,
                    card_size,
                    start,
                );
            }
        }
    }

    /// Build a per-card index of the first object offset at/before each card, so
    /// a dirty-card scan knows where to start walking (forward-filled so a card
    /// with no object start inherits the previous card's, catching objects that
    /// span into it). Mirrors gc-rust's `build_object_start_index`.
    unsafe fn build_object_start_index(
        &self,
        model: &BeagleObjectModel,
        tenured: &BumpAllocator,
        used: usize,
        card_table: &CardTable,
    ) -> Vec<usize> {
        let mut starts = vec![usize::MAX; card_table.card_count()];
        let base = tenured.base() as usize;
        let card_size = card_table.card_size();
        let mut off = 0usize;
        while off < used {
            let obj = unsafe { tenured.base().add(off) };
            let card_idx = ((obj as usize) - base) / card_size;
            if let Some(slot) = starts.get_mut(card_idx)
                && *slot == usize::MAX
            {
                *slot = off;
            }
            let (size, align) = unsafe { model.object_layout(obj) };
            if size == 0 {
                break;
            }
            off = (off + size + align - 1) & !(align - 1);
        }
        // Forward-fill: a card with no start inherits the previous card's.
        if !starts.is_empty() && starts[0] == usize::MAX {
            starts[0] = 0;
        }
        for i in 1..starts.len() {
            if starts[i] == usize::MAX {
                starts[i] = starts[i - 1];
            }
        }
        starts
    }

    /// Walk objects overlapping a single dirty card, promoting any young
    /// pointers they hold. Mirrors gc-rust's `scan_card_from_offset`.
    unsafe fn scan_card(
        &self,
        model: &BeagleObjectModel,
        tenured: &BumpAllocator,
        used: usize,
        card_start: usize,
        card_size: usize,
        start_offset: usize,
    ) {
        let card_end = card_start + card_size;
        let mut off = start_offset;
        while off < used {
            let obj = unsafe { tenured.base().add(off) };
            let (size, align) = unsafe { model.object_layout(obj) };
            if size == 0 {
                break;
            }
            let obj_addr = obj as usize;
            let obj_end = obj_addr + size;
            if obj_end > card_start && obj_addr < card_end {
                unsafe { model.scan_slots(obj, &mut |slot| self.promote_slot(model, slot)) };
            }
            if obj_addr >= card_end {
                break;
            }
            off = (off + size + align - 1) & !(align - 1);
        }
    }

    /// Minor collection: scavenge the nursery, promoting survivors to tenured.
    unsafe fn minor_gc(&self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]) {
        let _pause = PauseGuard::new("minor-stw");
        let model = BeagleObjectModel;
        let promotion_start = self.tenured_from().used();

        // Phase 1: promote nursery objects reachable from Beagle's roots.
        // (Beagle passes ALL roots — stack frames + shadow/handle stacks +
        // namespace cells — every collection, so unlike gc-rust we include the
        // per-call extra_roots in the minor root set.)
        let roots = BeagleRoots {
            gc_frame_tops,
            extra_roots,
        };
        roots.scan_roots(&mut |slot| unsafe {
            // Nursery root → promote it; old-gen root → scan one level for young
            // refs. (A slot can only be one of these; the calls are disjoint.)
            self.promote_slot(&model, slot);
            self.scan_old_gen_root(&model, slot);
        });

        // Phase 2: dirty-card scan for old→young roots (pre-existing tenured).
        unsafe { self.scan_dirty_cards(&model, promotion_start) };

        // Phase 3: Cheney-scan the just-promoted objects for THEIR young
        // references, promoting transitively. `used()` grows as we promote.
        let tf = self.tenured_from();
        let mut off = promotion_start;
        while off < tf.used() {
            let obj = unsafe { tf.base().add(off) };
            unsafe { model.scan_slots(obj, &mut |slot| self.promote_slot(&model, slot)) };
            let (size, align) = unsafe { model.object_layout(obj) };
            if size == 0 {
                break;
            }
            off = (off + size + align - 1) & !(align - 1);
        }

        // Phase 4: dead nursery objects' finalizers, then reset + clear cards.
        // Minor collects the nursery only.
        let nbase = self.nursery.base() as usize;
        let nend = nbase + self.nursery.used();
        unsafe { self.sweep_finalizables_header(|a| a >= nbase && a < nend) };
        self.nursery.reset();
        self.card_tables[self.from_idx()].clear_all();
    }

    // ── Major GC ─────────────────────────────────────────────────────

    /// Evacuate an object referenced by `slot` (from the nursery OR the tenured
    /// from-space) into the tenured to-space, then rewrite the slot.
    unsafe fn major_process_slot(&self, model: &BeagleObjectModel, slot: *mut u64) {
        let bits = unsafe { *slot };
        let Some(ptr) = BeaglePtrPolicy::try_decode_ptr(bits) else {
            return;
        };
        let addr = ptr as usize;
        let avail = match Self::within(&self.nursery, addr) {
            Some(a) => a,
            None => match Self::within(self.tenured_from(), addr) {
                Some(a) => a,
                None => return, // already in to-space, or a non-pointer pattern
            },
        };
        if !unsafe { model.is_valid(ptr, avail) } {
            return;
        }
        let new = unsafe { self.copy_or_forward_major(model, ptr, (bits & 0b111) as u8) };
        unsafe { *slot = BeaglePtrPolicy::reencode_ptr(bits, new) };
    }

    unsafe fn copy_or_forward_major(
        &self,
        model: &BeagleObjectModel,
        old: *mut u8,
        tag: u8,
    ) -> *mut u8 {
        if let Some(fwd) = unsafe { model.read_forwarding(old) } {
            return fwd;
        }
        let dest = self.tenured_to();
        let new = unsafe {
            model.copy_object(old, tag, &mut |size, align| {
                let p = dest.alloc_raw(size, align);
                assert!(!p.is_null(), "tenured to-space exhausted during major GC");
                p
            })
        };
        unsafe { model.write_forwarding(old, new) };
        new
    }

    /// Major (full) collection: evacuate everything live (nursery ∪ tenured
    /// from-space) into the tenured to-space, then swap and empty the nursery.
    unsafe fn major_gc(&self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]) {
        let _pause = PauseGuard::new("major-stw");
        let model = BeagleObjectModel;
        // Migrate redefined structs during this pass; clear the satisfied plans
        // afterward (see the field-migration note in the module docs).
        let migration_revision = crate::get_runtime()
            .get()
            .structs
            .pending_migration_revision();

        // Phase 1: roots.
        let roots = BeagleRoots {
            gc_frame_tops,
            extra_roots,
        };
        roots.scan_roots(&mut |slot| unsafe { self.major_process_slot(&model, slot) });

        // Phase 2: Cheney-scan the tenured to-space (grows as we copy).
        let to = self.tenured_to();
        let mut off = 0usize;
        while off < to.used() {
            let obj = unsafe { to.base().add(off) };
            unsafe { model.scan_slots(obj, &mut |slot| self.major_process_slot(&model, slot)) };
            let (size, align) = unsafe { model.object_layout(obj) };
            if size == 0 {
                break;
            }
            off = (off + size + align - 1) & !(align - 1);
        }

        if let Some(revision) = migration_revision {
            crate::get_runtime()
                .get_mut()
                .structs
                .complete_pending_migrations(revision);
        }

        // Finalizers for dead objects in the spaces we are about to reclaim
        // (nursery ∪ tenured from-space — both are evacuated by a major).
        let nbase = self.nursery.base() as usize;
        let nend = nbase + self.nursery.used();
        let tbase = self.tenured_from().base() as usize;
        let tend = tbase + self.tenured_from().used();
        unsafe {
            self.sweep_finalizables_header(|a| (a >= nbase && a < nend) || (a >= tbase && a < tend))
        };

        // Phase 3: swap tenured, reset the old from-space + the nursery, and
        // clear the (new) live tenured card table — every old→young edge was
        // just rebuilt.
        let old_from = self.from_idx();
        self.tenured[old_from].reset();
        self.nursery.reset();
        self.from_idx.set(1 - old_from);
        self.card_tables[self.from_idx()].clear_all();
    }

    // ── Side-table major GC (concurrent foundation) ─────────────────
    //
    // A major collection that records forwarding OUT OF BAND in a HashMap
    // (`from_addr → to_addr`) instead of overwriting object headers. This is
    // the copy/repoint mechanism the concurrent collector needs: during a
    // concurrent cycle the collector must NOT clobber a live object's header
    // (Beagle packs type_id+size+flags into one word; a live mutator reading it
    // — e.g. a `set!` inline-cache struct-id check — would see garbage). Here it
    // is exercised STOP-THE-WORLD (env `BEAGLE_GCRUST_TABLE=1`) so the algorithm
    // is validated deterministically before Stage C makes it actually
    // concurrent. Produces the same result as `major_gc`.

    /// Bytes available if `addr` names an object in a currently-collected space
    /// (nursery ∪ tenured-from); else None.
    #[inline]
    fn in_collected(&self, addr: usize) -> Option<usize> {
        Self::within(&self.nursery, addr).or_else(|| Self::within(self.tenured_from(), addr))
    }

    /// Copy `from` into to-space (unless already copied), recording the mapping
    /// in `table`. Does NOT touch `from`'s header (out-of-band forwarding).
    unsafe fn copy_via_table(
        &self,
        model: &BeagleObjectModel,
        table: &mut HashMap<usize, usize>,
        from: *mut u8,
        tag: u8,
    ) -> *mut u8 {
        if let Some(&to) = table.get(&(from as usize)) {
            return to as *mut u8;
        }
        let dest = self.tenured_to();
        let new = unsafe {
            model.copy_object(from, tag, &mut |size, align| {
                let p = dest.alloc_raw(size, align);
                assert!(!p.is_null(), "tenured to-space exhausted during major GC");
                p
            })
        };
        table.insert(from as usize, new as usize);
        new
    }

    /// Trace a slot: ensure its target is copied (recorded in `table`), WITHOUT
    /// repointing the slot — repointing is a separate pass so the trace can read
    /// original from-space field values.
    unsafe fn trace_slot_via_table(
        &self,
        model: &BeagleObjectModel,
        table: &mut HashMap<usize, usize>,
        slot: *mut u64,
    ) {
        let bits = unsafe { *slot };
        let Some(ptr) = BeaglePtrPolicy::try_decode_ptr(bits) else {
            return;
        };
        let Some(avail) = self.in_collected(ptr as usize) else {
            return;
        };
        if !unsafe { model.is_valid(ptr, avail) } {
            return;
        }
        unsafe { self.copy_via_table(model, table, ptr, (bits & 0b111) as u8) };
    }

    /// Rewrite `slot` to the to-space copy of its target, if the target was
    /// copied (present in `table`), preserving the tag.
    #[inline]
    unsafe fn repoint_slot(table: &HashMap<usize, usize>, slot: *mut u64) {
        let bits = unsafe { *slot };
        let Some(ptr) = BeaglePtrPolicy::try_decode_ptr(bits) else {
            return;
        };
        if let Some(&to) = table.get(&(ptr as usize)) {
            unsafe { *slot = BeaglePtrPolicy::reencode_ptr(bits, to as *mut u8) };
        }
    }

    /// Full collection via the out-of-band side table, driven stop-the-world.
    unsafe fn major_gc_via_table(
        &self,
        gc_frame_tops: &[usize],
        extra_roots: &[(*mut usize, usize)],
    ) {
        let model = BeagleObjectModel;
        let migration_revision = crate::get_runtime()
            .get()
            .structs
            .pending_migration_revision();
        let mut table: HashMap<usize, usize> = HashMap::new();
        let roots = BeagleRoots {
            gc_frame_tops,
            extra_roots,
        };

        // Phase 1: copy everything reachable (roots, then Cheney over to-space).
        // Fields are NOT repointed yet — to-space copies still hold from-space
        // addresses, which is what the trace reads to find children.
        roots
            .scan_roots(&mut |slot| unsafe { self.trace_slot_via_table(&model, &mut table, slot) });
        {
            let to = self.tenured_to();
            let mut off = 0usize;
            while off < to.used() {
                let obj = unsafe { to.base().add(off) };
                unsafe {
                    model.scan_slots(obj, &mut |slot| {
                        self.trace_slot_via_table(&model, &mut table, slot)
                    })
                };
                let (size, align) = unsafe { model.object_layout(obj) };
                if size == 0 {
                    break;
                }
                off = (off + size + align - 1) & !(align - 1);
            }
        }

        if let Some(revision) = migration_revision {
            crate::get_runtime()
                .get_mut()
                .structs
                .complete_pending_migrations(revision);
        }
        // Finalizers (table major evacuates nursery ∪ tenured from-space).
        let nbase = self.nursery.base() as usize;
        let nend = nbase + self.nursery.used();
        let tbase = self.tenured_from().base() as usize;
        let tend = tbase + self.tenured_from().used();
        unsafe {
            self.sweep_finalizables_table(&table, |a| {
                (a >= nbase && a < nend) || (a >= tbase && a < tend)
            })
        };

        // Phase 2: repoint every to-space field and every root via the table.
        {
            let to = self.tenured_to();
            let mut off = 0usize;
            while off < to.used() {
                let obj = unsafe { to.base().add(off) };
                unsafe { model.scan_slots(obj, &mut |slot| Self::repoint_slot(&table, slot)) };
                let (size, align) = unsafe { model.object_layout(obj) };
                if size == 0 {
                    break;
                }
                off = (off + size + align - 1) & !(align - 1);
            }
        }
        roots.scan_roots(&mut |slot| unsafe { Self::repoint_slot(&table, slot) });

        // Phase 3: swap, reset old from-space + nursery, clear new card table.
        let old_from = self.from_idx();
        self.tenured[old_from].reset();
        self.nursery.reset();
        self.from_idx.set(1 - old_from);
        self.card_tables[self.from_idx()].clear_all();
    }

    // ── Concurrent major GC (Stage C) ───────────────────────────────
    //
    // A concurrent major collects the OLD generation (tenured) only; the
    // nursery is treated as a root set (walked, its tenured pointers copied +
    // repointed) but is NOT evacuated — running mutators keep allocating into
    // it. Forwarding is out-of-band (`conc_table`), so live mutators reading
    // object headers during the trace stay correct. The trace runs while
    // mutators execute (the driver holds the alloc mutex, so mutators block
    // only on allocation); it copies with ATOMIC word reads because a mutator
    // may be writing a from-space object as it's copied.

    /// Copy `from` (a tenured-from object) into to-space via atomic word reads,
    /// recording the mapping in `table`. No migration (concurrent runs only when
    /// no migration is pending). Header untouched (out-of-band forwarding).
    unsafe fn copy_conc(&self, table: &mut HashMap<usize, usize>, from: *mut u8) -> *mut u8 {
        if let Some(&to) = table.get(&(from as usize)) {
            return to as *mut u8;
        }
        let ho = HeapObject::from_untagged(from as *const u8);
        let size = ho.full_size();
        let dest = self.tenured_to();
        let new = dest.alloc_raw(size, 8);
        assert!(
            !new.is_null(),
            "tenured to-space exhausted during concurrent trace"
        );
        // Atomic word-by-word copy: a mutator may be writing `from`'s fields
        // concurrently. An atomic (Relaxed) load yields a whole old-or-new
        // word, never a torn one, so any pointer we then trace is valid.
        let words = size / 8;
        let src = from as *const AtomicU64;
        let dst = new as *mut u64;
        for i in 0..words {
            let v = unsafe { (*src.add(i)).load(Ordering::Relaxed) };
            unsafe { *dst.add(i) = v };
        }
        table.insert(from as usize, new as usize);
        new
    }

    /// Process a slot for the concurrent major: if it points at a tenured-from
    /// object, copy it (recorded in `table`) AND repoint the slot to the
    /// to-space copy — safe because to-space is GC-private during the trace
    /// (mutators keep the from-space invariant). Repointing here (rather than in
    /// a separate STW#2 pass) is what keeps the final pause short. Returns
    /// `true` if, after processing, the slot holds a NURSERY (young) pointer —
    /// the caller marks the owning object's card so the next minor scans it.
    unsafe fn copy_repoint_slot_conc(
        &self,
        model: &BeagleObjectModel,
        table: &mut HashMap<usize, usize>,
        slot: *mut u64,
        repoint: bool,
    ) -> bool {
        let bits = unsafe { *slot };
        let Some(ptr) = BeaglePtrPolicy::try_decode_ptr(bits) else {
            return false;
        };
        if let Some(avail) = Self::within(self.tenured_from(), ptr as usize) {
            if unsafe { model.is_valid(ptr, avail) } {
                let new = unsafe { self.copy_conc(table, ptr) };
                // Repoint ONLY when the slot is not visible to a running mutator:
                // during the trace (to-space is GC-private) and at STW#2. At
                // STW#1 mutators are about to resume on from-space, so root and
                // nursery slots must keep pointing there (copy-only).
                if repoint {
                    unsafe { *slot = BeaglePtrPolicy::reencode_ptr(bits, new) };
                }
            }
            false
        } else {
            // Nursery pointer (nursery isn't evacuated by a concurrent major) —
            // signals an old→young edge the next minor must see.
            self.nursery.contains(ptr)
        }
    }

    /// Walk every nursery object (nursery-as-roots): copy+repoint each of its
    /// tenured references. Nursery objects aren't evacuated or card-marked here
    /// (the minor GC finds them via roots), so the young flag is ignored.
    unsafe fn nursery_walk_conc(
        &self,
        model: &BeagleObjectModel,
        table: &mut HashMap<usize, usize>,
        repoint: bool,
    ) {
        let base = self.nursery.base() as usize;
        let end = base + self.nursery.used();
        let mut addr = base;
        while addr < end {
            let obj = addr as *mut u8;
            unsafe {
                model.scan_slots(obj, &mut |slot| {
                    self.copy_repoint_slot_conc(model, table, slot, repoint);
                })
            };
            let (size, align) = unsafe { model.object_layout(obj as *const u8) };
            if size == 0 {
                break;
            }
            addr += (size + align - 1) & !(align - 1);
        }
    }

    /// Drain dirty cards: a tenured object that was mutated during the
    /// concurrent trace (its card is dirty) has a STALE to-space copy — re-copy
    /// its current from-space contents so the copy reflects the mutation.
    unsafe fn recopy_dirty_cards(
        &self,
        model: &BeagleObjectModel,
        table: &mut HashMap<usize, usize>,
    ) {
        let card_table = &self.card_tables[self.from_idx()];
        let tenured = self.tenured_from();
        let used = tenured.used();
        let dirty: Vec<(usize, *const u8)> = card_table.iter_dirty().collect();
        if dirty.is_empty() {
            return;
        }
        let obj_starts = unsafe { self.build_object_start_index(model, tenured, used, card_table) };
        let card_size = card_table.card_size();
        // Collect dirty (from,to) pairs first, then re-copy (avoids reborrow).
        let mut dirty_objs: Vec<(usize, usize)> = Vec::new();
        for (card_idx, card_addr) in dirty {
            let Some(&start) = obj_starts.get(card_idx) else {
                continue;
            };
            if start == usize::MAX {
                continue;
            }
            let card_start = card_addr as usize;
            let card_end = card_start + card_size;
            let mut off = start;
            while off < used {
                let obj = unsafe { tenured.base().add(off) };
                let (size, align) = unsafe { model.object_layout(obj as *const u8) };
                if size == 0 {
                    break;
                }
                let obj_addr = obj as usize;
                if obj_addr + size > card_start
                    && obj_addr < card_end
                    && let Some(&to) = table.get(&obj_addr)
                {
                    dirty_objs.push((obj_addr, to));
                }
                if obj_addr >= card_end {
                    break;
                }
                off += (size + align - 1) & !(align - 1);
            }
        }
        let to_idx = 1 - self.from_idx();
        for (from, to) in dirty_objs {
            let size = unsafe { HeapObject::from_untagged(from as *const u8).full_size() };
            // Refresh the to-space copy with the mutated from-space contents...
            unsafe { core::ptr::copy_nonoverlapping(from as *const u8, to as *mut u8, size) };
            // ...then repoint its fields (they now hold from-space addresses
            // again) and re-mark its young card if it points at the nursery.
            let mut has_young = false;
            unsafe {
                model.scan_slots(to as *mut u8, &mut |slot| {
                    if self.copy_repoint_slot_conc(model, table, slot, true) {
                        has_young = true;
                    }
                })
            };
            if has_young {
                self.card_tables[to_idx].mark_dirty(to as *const u8);
            }
        }
    }

    // ── Finalizers (side-list, O(finalizable count)) ─────────────────

    /// Sweep the finalizable side list after an STW/header-forwarding
    /// collection: an entry in a collected space that did NOT survive (no
    /// forwarding pointer in its header) is dead — enqueue its finalizer and
    /// drop it; a survivor is updated to its new address. Entries outside the
    /// collected space(s) are untouched. Must run before the spaces are reset.
    unsafe fn sweep_finalizables_header(&self, in_collected: impl Fn(usize) -> bool) {
        let mut list = self.finalizable.borrow_mut();
        if list.is_empty() {
            return;
        }
        list.retain_mut(|old| {
            if !in_collected(*old) {
                return true; // not collected → survives, unchanged
            }
            let word = unsafe { *(*old as *const usize) };
            if Header::is_forwarding_bit_set(word) {
                *old = Header::clear_forwarding_bit(word) >> 3; // survived, relocated
                true
            } else {
                let hobj = HeapObject::from_untagged(*old as *const u8);
                unsafe { crate::gc::finalizers::maybe_enqueue_finalizer(&hobj) };
                false
            }
        });
    }

    /// Sweep the finalizable side list after a table-forwarding collection
    /// (table major / concurrent): an entry in a collected space absent from
    /// `table` is dead; a present one survived (updated to its to-space address).
    unsafe fn sweep_finalizables_table(
        &self,
        table: &HashMap<usize, usize>,
        in_collected: impl Fn(usize) -> bool,
    ) {
        let mut list = self.finalizable.borrow_mut();
        if list.is_empty() {
            return;
        }
        list.retain_mut(|old| {
            if !in_collected(*old) {
                return true;
            }
            if let Some(&new) = table.get(old) {
                *old = new;
                true
            } else {
                let hobj = HeapObject::from_untagged(*old as *const u8);
                unsafe { crate::gc::finalizers::maybe_enqueue_finalizer(&hobj) };
                false
            }
        });
    }
}

impl Allocator for GcRustHeap {
    fn new(options: AllocatorOptions) -> Self {
        let nursery_bytes = env_bytes("BEAGLE_GCRUST_NURSERY_MB", DEFAULT_NURSERY_BYTES);
        let tenured_bytes = env_bytes("BEAGLE_GCRUST_SPACE_MB", DEFAULT_TENURED_BYTES);
        // The header type only fixes the (unused here) TypeInfo `type_id_offset`;
        // collection is driven by BeagleObjectModel.
        let tenured = [
            BumpAllocator::new::<crate::gc::gcrust::Compact>(tenured_bytes),
            BumpAllocator::new::<crate::gc::gcrust::Compact>(tenured_bytes),
        ];
        let card_tables = [
            CardTable::new(tenured[0].base(), tenured[0].size()),
            CardTable::new(tenured[1].base(), tenured[1].size()),
        ];
        GcRustHeap {
            nursery: BumpAllocator::new::<crate::gc::gcrust::Compact>(nursery_bytes),
            tenured,
            card_tables,
            from_idx: Cell::new(0),
            gc_phase: AtomicU8::new(PHASE_IDLE),
            use_table: std::env::var("BEAGLE_GCRUST_TABLE").is_ok(),
            concurrent_enabled: std::env::var("BEAGLE_GCRUST_CONCURRENT").is_ok(),
            conc_table: None,
            conc_count: 0,
            finalizable: std::cell::RefCell::new(Vec::new()),
            options,
            gc_count: 0,
        }
    }

    fn try_allocate(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let total = words * 8 + header_bytes(words);
        // Objects that can never fit the nursery go straight to tenured.
        let region = if total > self.nursery.size() {
            self.tenured_from()
        } else {
            &self.nursery
        };
        match Self::alloc_in(region, words, kind) {
            Some(ptr) => Ok(AllocateAction::Allocated(ptr)),
            None => Ok(AllocateAction::Gc),
        }
    }

    fn supports_tlab(&self) -> bool {
        true
    }

    fn grab_tlab(&mut self, words: usize) -> Option<(usize, usize)> {
        // Debug switch: BEAGLE_GCRUST_NOTLAB=1 disables TLABs (falls back to the
        // direct per-object locked path) to isolate TLAB-specific bugs. Cached so
        // the refill path doesn't do a per-call env lookup.
        static NOTLAB: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *NOTLAB.get_or_init(|| std::env::var("BEAGLE_GCRUST_NOTLAB").is_ok()) {
            return None;
        }
        let tlab = tlab_bytes();
        let min_bytes = words * 8 + header_bytes(words);
        // Too big for a TLAB → caller falls back to a direct/tenured alloc.
        if min_bytes > tlab {
            return None;
        }
        // Grant a TLAB (or the nursery's remaining tail, if smaller). alloc_raw
        // advances the nursery cursor and returns null if it can't fit — that's
        // the young-gen-exhausted signal (caller GCs).
        let grant = (tlab.min(self.nursery.remaining())) & !7;
        if grant < min_bytes {
            return None;
        }
        let ptr = self.nursery.alloc_raw(grant, 8);
        if ptr.is_null() {
            return None;
        }
        let start = ptr as usize;
        Some((start, start + grant))
    }

    fn grab_tlab_sized(&mut self, min_bytes: usize) -> Option<(usize, usize)> {
        // Reserve at least `min_bytes` (rounded up) — a bigger-than-default TLAB
        // for a no-GC allocation sequence. `None` if the nursery can't fit it.
        let want = (min_bytes + 7) & !7;
        if want > self.nursery.remaining() {
            return None;
        }
        let grant = want.max(tlab_bytes()).min(self.nursery.remaining()) & !7;
        let grant = grant.max(want);
        let ptr = self.nursery.alloc_raw(grant, 8);
        if ptr.is_null() {
            return None;
        }
        let start = ptr as usize;
        Some((start, start + grant))
    }

    fn tlab_bump(
        &self,
        alloc_ptr: usize,
        alloc_end: usize,
        words: usize,
        kind: BuiltInTypes,
        zeroed: bool,
    ) -> Option<(usize, usize)> {
        let obj = (alloc_ptr + 7) & !7;
        let total = words * 8 + header_bytes(words);
        if obj == 0 || obj + total > alloc_end {
            return None;
        }
        // Zero the whole object (not just when `zeroed`): TLAB memory is recycled
        // from-space and holds garbage. `alloc_in` (the path this replaces) always
        // zeroed, and the GC relies on it — a not-yet-fully-initialized object
        // that becomes a root (e.g. an intervening GC during nested construction)
        // must trace as nulls, not garbage. `_zeroed` is kept for API symmetry.
        let _ = zeroed;
        unsafe {
            std::ptr::write_bytes(obj as *mut u8, 0, total);
            let mut ho = HeapObject::from_untagged(obj as *const u8);
            ho.write_header(Word::from_word(words));
            if kind == BuiltInTypes::Float {
                *(obj as *mut usize) |= Header::OPAQUE_BIT_MASK;
            }
        }
        Some((obj, obj + total))
    }

    fn allocate_for_runtime(&mut self, words: usize) -> Result<usize, Box<dyn Error>> {
        // Long-lived runtime infrastructure goes directly to the old generation.
        match Self::alloc_in(self.tenured_from(), words, BuiltInTypes::HeapObject) {
            Some(ptr) => Ok(BuiltInTypes::HeapObject.tag(ptr as isize) as usize),
            None => Err("Need GC to allocate runtime object".into()),
        }
    }

    fn gc(&mut self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]) {
        if !self.options.gc {
            return;
        }
        self.gc_count += 1;
        let migration_pending = crate::get_runtime().get().structs.has_pending_migrations();
        // Publish for copy_object (avoids a per-object registry lock).
        MIGRATE_ACTIVE.store(migration_pending, Ordering::Relaxed);
        let periodic = self.gc_count.is_multiple_of(FULL_GC_FREQUENCY);
        // A minor GC only reclaims the nursery. Fall back to a major when: a
        // struct migration is pending (migration happens during a full pass);
        // periodically (to reclaim tenured garbage); there is nothing young to
        // scavenge (so the pressure is tenured); or a worst-case full promotion
        // wouldn't fit the tenured from-space.
        // The concurrent path (`begin_concurrent`, in gc_impl) only runs
        // multi-threaded, and only then handles non-migration majors. So this
        // STW path still owns majors when concurrent is off OR when single-
        // threaded (registered thread count 0 = main only). Migration majors
        // are always STW.
        let single_threaded = crate::get_runtime()
            .get()
            .registered_thread_count
            .load(Ordering::Acquire)
            == 0;
        let do_major = migration_pending
            || std::env::var("BEAGLE_GCRUST_MAJOR_ONLY").is_ok()
            || ((!self.concurrent_enabled || single_threaded)
                && (periodic
                    || self.nursery.used() == 0
                    || self.nursery.used() > self.tenured_from().remaining()));
        let dbg = std::env::var("BEAGLE_GCRUST_DEBUG").is_ok();
        if dbg {
            eprintln!(
                "[gcrust] gc: {} nursery.used={} tenured.remaining={}",
                if do_major { "major-stw" } else { "minor" },
                self.nursery.used(),
                self.tenured_from().remaining(),
            );
        }
        unsafe {
            if do_major {
                // Ensure the to-space can hold the worst-case live set before we
                // copy into it (the collector is a copying collector, so a
                // too-small to-space would exhaust mid-copy). This is how the
                // heap grows.
                self.ensure_to_space_fits();
                if self.use_table {
                    self.major_gc_via_table(gc_frame_tops, extra_roots);
                } else {
                    self.major_gc(gc_frame_tops, extra_roots);
                }
            } else {
                self.minor_gc(gc_frame_tops, extra_roots);
            }
        }
        if dbg {
            eprintln!(
                "[gcrust]   -> nursery.used={} tenured.remaining={}",
                self.nursery.used(),
                self.tenured_from().remaining(),
            );
        }
    }

    fn grow(&mut self) {
        // Called by allocate-with-retries when an allocation still fails after a
        // collection. Double the spare (to) space; the next major collection
        // copies the live set into it (and `ensure_to_space_fits` guarantees it
        // is large enough). The active from-space is never reallocated here — it
        // holds live data — so growth is realized on the next flip.
        let to = 1 - self.from_idx();
        let new_size = self.tenured[to].size().saturating_mul(2);
        self.tenured[to] = BumpAllocator::new::<Compact>(new_size);
        self.card_tables[to] = CardTable::new(self.tenured[to].base(), self.tenured[to].size());
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }

    fn can_allocate(&self, words: usize, _kind: BuiltInTypes) -> bool {
        let total = words * 8 + header_bytes(words);
        if total > self.nursery.size() {
            self.tenured_from().remaining() >= total
        } else {
            self.nursery.remaining() >= total
        }
    }

    fn bytes_in_use(&self) -> usize {
        self.nursery.used() + self.tenured_from().used()
    }

    fn begin_concurrent(
        &mut self,
        gc_frame_tops: &[usize],
        extra_roots: &[(*mut usize, usize)],
    ) -> bool {
        let dbg = std::env::var("BEAGLE_GCRUST_DEBUG").is_ok();
        if !self.concurrent_enabled {
            return false;
        }
        // Migration must run in a full STW pass; go concurrent only otherwise.
        if crate::get_runtime().get().structs.has_pending_migrations() {
            if dbg {
                eprintln!("[gcrust] begin_concurrent: skip (migration pending)");
            }
            return false;
        }
        // Route EVERY non-migration major through the concurrent path (not just
        // tenured-pressure ones) — otherwise gc()'s periodic majors would run
        // stop-the-world and dominate the pause profile. Minor situations fall
        // back to the normal path.
        self.conc_count += 1;
        let periodic = self.conc_count.is_multiple_of(FULL_GC_FREQUENCY);
        let tenured_pressure = periodic
            || self.nursery.used() == 0
            || self.nursery.used() > self.tenured_from().remaining();
        if dbg {
            eprintln!(
                "[gcrust] begin_concurrent: nursery.used={} tenured.remaining={} pressure={}",
                self.nursery.used(),
                self.tenured_from().remaining(),
                tenured_pressure
            );
        }
        if !tenured_pressure {
            return false;
        }
        let _pause = PauseGuard::new("concurrent-STW1");

        if std::env::var("BEAGLE_GCRUST_DEBUG").is_ok() {
            eprintln!("[gcrust] concurrent major cycle begin");
        }
        // Grow the to-space if needed BEFORE the concurrent trace copies into it
        // (this runs at STW#1, so reallocating the spare is safe).
        self.ensure_to_space_fits();
        // Turn the concurrent write barrier on (published under STW here).
        self.gc_phase.store(PHASE_COPYING, Ordering::Release);
        let model = BeagleObjectModel;
        let mut table: HashMap<usize, usize> = HashMap::new();
        // Phase-1 copy: roots + nursery-as-roots → copy their tenured targets
        // into to-space (out-of-band forwarding; fields not yet repointed).
        let roots = BeagleRoots {
            gc_frame_tops,
            extra_roots,
        };
        roots.scan_roots(&mut |slot| unsafe {
            self.copy_repoint_slot_conc(&model, &mut table, slot, false);
        });
        unsafe { self.nursery_walk_conc(&model, &mut table, false) };
        self.conc_table = Some(table);
        true
    }

    fn concurrent_trace(&mut self) {
        let model = BeagleObjectModel;
        let mut table = self
            .conc_table
            .take()
            .expect("concurrent_trace called without begin_concurrent");
        // Cheney over to-space: copy the reachable tenured graph AND repoint
        // fields as we go (to-space is GC-private during the trace, so this is
        // safe) — doing the repoint here rather than in a separate STW#2 pass is
        // what keeps the final pause short. Also mark the young card for any
        // to-space object left pointing at the nursery, so the next minor GC
        // scans it (this folds in what a post-flip card rebuild would do).
        // Mutators run concurrently (blocked only on allocation); atomic
        // word-copy (`copy_conc`) keeps reads of concurrently-mutated from-space
        // objects untorn. to-space objects become the NEW tenured after the
        // flip, so their cards live in `card_tables[1 - from_idx]`.
        let to_idx = 1 - self.from_idx();
        let to = self.tenured_to();
        let mut off = 0usize;
        while off < to.used() {
            let obj = unsafe { to.base().add(off) };
            let mut has_young = false;
            unsafe {
                model.scan_slots(obj, &mut |slot| {
                    if self.copy_repoint_slot_conc(&model, &mut table, slot, true) {
                        has_young = true;
                    }
                })
            };
            if has_young {
                self.card_tables[to_idx].mark_dirty(obj as *const u8);
            }
            let (size, align) = unsafe { model.object_layout(obj as *const u8) };
            if size == 0 {
                break;
            }
            off = (off + size + align - 1) & !(align - 1);
        }
        self.conc_table = Some(table);
    }

    fn finish_concurrent(&mut self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]) {
        let _pause = PauseGuard::new("concurrent-STW2");
        let model = BeagleObjectModel;
        let mut table = self
            .conc_table
            .take()
            .expect("finish_concurrent called without begin_concurrent");
        let roots = BeagleRoots {
            gc_frame_tops,
            extra_roots,
        };
        // The concurrent trace already copied AND repointed the whole reachable
        // graph that existed at STW#1, and marked young cards as it went. This
        // pause only has to reconcile what changed since: fresh roots, mutated
        // (dirty-carded) tenured objects, and any newly-reachable objects.
        // Everything is copy+repointed, so there is NO O(live) repoint pass and
        // NO O(tenured) card rebuild here — that is what keeps STW#2 short.

        // Objects the trace already produced end at this watermark; the closure
        // below can only append ABOVE it, so the fix-point scan need only cover
        // the tail.
        let watermark = self.tenured_to().used();

        // 1. Fresh roots + nursery-as-roots: copy+repoint (roots/nursery slots
        //    are updated in place here).
        roots.scan_roots(&mut |slot| unsafe {
            self.copy_repoint_slot_conc(&model, &mut table, slot, true);
        });
        unsafe { self.nursery_walk_conc(&model, &mut table, true) };
        // 2. Dirty cards: refresh + repoint to-space copies of objects a mutator
        //    changed during the trace (also re-marks their young cards).
        unsafe { self.recopy_dirty_cards(&model, &mut table) };
        // 3. Cheney fix-point over ONLY the newly-appended tail (copy+repoint +
        //    young-card mark) — O(objects new since the trace), not O(live).
        let to_idx = 1 - self.from_idx();
        {
            let to = self.tenured_to();
            let mut off = watermark;
            while off < to.used() {
                let obj = unsafe { to.base().add(off) };
                let mut has_young = false;
                unsafe {
                    model.scan_slots(obj, &mut |slot| {
                        if self.copy_repoint_slot_conc(&model, &mut table, slot, true) {
                            has_young = true;
                        }
                    })
                };
                if has_young {
                    self.card_tables[to_idx].mark_dirty(obj as *const u8);
                }
                let (size, align) = unsafe { model.object_layout(obj as *const u8) };
                if size == 0 {
                    break;
                }
                off = (off + size + align - 1) & !(align - 1);
            }
        }
        // 4. Finalizers for dead tenured-from objects (absent from the table).
        //    A concurrent major evacuates ONLY tenured; nursery finalizables are
        //    not collected here (nursery kept), so bound to tenured-from.
        let tbase = self.tenured_from().base() as usize;
        let tend = tbase + self.tenured_from().used();
        unsafe { self.sweep_finalizables_table(&table, |a| a >= tbase && a < tend) };
        // 5. Flip tenured; the nursery is KEPT (not evacuated). The new
        //    from-space's young cards were marked during the trace/finish, so
        //    only the old from-space's stale marks need clearing.
        let old_from = self.from_idx();
        self.tenured[old_from].reset();
        self.from_idx.set(1 - old_from);
        self.card_tables[old_from].clear_all();
        // 6. Barriers off; concurrent state cleared.
        self.gc_phase.store(PHASE_IDLE, Ordering::Release);
        if std::env::var("BEAGLE_GCRUST_DEBUG").is_ok() {
            eprintln!(
                "[gcrust] concurrent major done: live(to)={} nursery={} table={}",
                self.tenured_from().used(),
                self.nursery.used(),
                table.len(),
            );
        }
    }

    fn write_barrier(&mut self, object_ptr: usize, new_value: usize) {
        // HOT PATH — called after every heap pointer-field store. Keep it cheap:
        // fixed-range `contains` checks (no cursor reads), and the concurrent
        // branch is skipped entirely unless concurrent GC is enabled (so the
        // ordinary generational path pays no atomic load).
        if !BuiltInTypes::is_heap_pointer(object_ptr) {
            return;
        }
        let obj = (object_ptr >> 3) as *const u8;
        let from_idx = self.from_idx();

        // Concurrent-phase dirty tracking (Phase 2, only while a concurrent
        // major is copying): ANY mutation of a from-space tenured object must
        // dirty its card so the final pause re-copies it. Value-independent.
        if self.concurrent_enabled && self.barriers_active() && self.tenured[from_idx].contains(obj)
        {
            self.card_tables[from_idx].mark_dirty(obj);
        }

        // Generational old→young edge: mark the card for a tenured object made
        // to point at a nursery object, so the next minor GC scans it.
        if !BuiltInTypes::is_heap_pointer(new_value) {
            return;
        }
        let nv = (new_value >> 3) as *const u8;
        if self.nursery.contains(nv) && self.tenured[from_idx].contains(obj) {
            self.card_tables[from_idx].mark_dirty(obj);
        }
    }

    fn mark_card_unconditional(&mut self, object_ptr: usize) {
        if !BuiltInTypes::is_heap_pointer(object_ptr) {
            return;
        }
        let obj = (object_ptr >> 3) as *const u8;
        let from_idx = self.from_idx();
        if self.tenured[from_idx].contains(obj) {
            self.card_tables[from_idx].mark_dirty(obj);
        }
    }

    fn register_finalizable(&mut self, tagged_ptr: usize) {
        // Track it in the side list so collections find it in O(finalizable
        // count) rather than by walking the whole heap.
        if !BuiltInTypes::is_heap_pointer(tagged_ptr) {
            return;
        }
        self.finalizable.borrow_mut().push(tagged_ptr >> 3);
    }

    fn allocator_frontier(&self) -> (usize, usize) {
        // Expose the nursery bump window so the JIT inline fast path can bump
        // `MutatorState.alloc_ptr` directly (no locked slow-path call per
        // allocation). `MutexAllocator` returns (0,0) once >1 thread is
        // registered, disarming the fast path under multithreading.
        let base = self.nursery.base() as usize;
        (base + self.nursery.used(), base + self.nursery.size())
    }

    fn sync_allocator_frontier(&mut self, alloc_ptr: usize) {
        // Absorb objects the JIT fast path placed by bumping `alloc_ptr` without
        // telling us: advance the nursery cursor to match, so the next slow-path
        // allocation and the next GC walk see them.
        //
        // ONLY ADVANCE, never rewind. In single-thread mode `alloc_ptr` is the
        // whole-nursery frontier and is always >= the cursor. In multi-thread
        // TLAB mode the global cursor is authoritative (advanced by `grab_tlab`
        // per whole slice) and is AHEAD of any one thread's private `alloc_ptr`;
        // setting the cursor back to a single thread's TLAB position would
        // discard the other threads' granted TLABs and orphan the live objects
        // in them (they'd fall "past the cursor", invisible to the GC and reused
        // out from under their roots).
        let base = self.nursery.base() as usize;
        if alloc_ptr >= base && alloc_ptr <= base + self.nursery.size() {
            let off = alloc_ptr - base;
            if off > self.nursery.used() {
                self.nursery.set_used(off);
            }
        }
    }
}
