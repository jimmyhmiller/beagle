//! Post-mortem callbacks for GC'd objects that own off-heap resources.
//!
//! A finalizer is a `fn(&[usize])` keyed by struct_id. When a GC identifies
//! a dead object whose struct_id has a registered finalizer, it snapshots
//! the object's field values and enqueues the work. A dedicated finalizer
//! thread drains the queue and runs the callbacks — never inline with
//! collection.
//!
//! Rules for finalizer functions:
//!  * Must not allocate beagle heap memory.
//!  * Must not call into beagle code.
//!  * Must not hold the GC lock.
//!  * Must tolerate being called with a null pointer in field 0 (means the
//!    resource was already released explicitly or disowned via ffi/forget).
//!
//! The queue is an unbounded mpsc channel. The finalizer thread exits when
//! the sender is dropped — which only happens at process exit.
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{OnceLock, RwLock};
use std::thread;

pub type FinalizerFn = fn(&[usize]);

struct FinalizerWork {
    finalizer: FinalizerFn,
    fields: Vec<usize>,
}

static REGISTRY: OnceLock<RwLock<HashMap<usize, FinalizerFn>>> = OnceLock::new();
static SENDER: OnceLock<Sender<FinalizerWork>> = OnceLock::new();
// Paired counters so tests (and anything needing synchronization) can tell
// when the finalizer thread has caught up with everything the GC enqueued.
static ENQUEUED: AtomicUsize = AtomicUsize::new(0);
static COMPLETED: AtomicUsize = AtomicUsize::new(0);

fn registry() -> &'static RwLock<HashMap<usize, FinalizerFn>> {
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

fn sender() -> &'static Sender<FinalizerWork> {
    SENDER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<FinalizerWork>();
        thread::Builder::new()
            .name("beagle-finalizer".into())
            .spawn(move || {
                while let Ok(work) = rx.recv() {
                    // A panicking finalizer must not kill the thread — other
                    // resources still need freeing. Swallow the panic; the
                    // free counter will reveal missed frees in tests.
                    let _ = catch_unwind(AssertUnwindSafe(|| (work.finalizer)(&work.fields)));
                    COMPLETED.fetch_add(1, Ordering::Release);
                }
            })
            .expect("failed to spawn beagle-finalizer thread");
        tx
    })
}

/// Register a finalizer callback for a struct_id. Called at runtime
/// initialization once per finalizable type. Later registrations for the
/// same struct_id overwrite — typically harmless; indicates a re-init.
pub fn register_finalizer(struct_id: usize, finalizer: FinalizerFn) {
    registry().write().unwrap().insert(struct_id, finalizer);
}

/// Look up the finalizer for a struct_id, if any. GC calls this during
/// sweep/evacuation to decide whether a dead object needs post-mortem work.
pub fn finalizer_for(struct_id: usize) -> Option<FinalizerFn> {
    registry().read().unwrap().get(&struct_id).copied()
}

/// Enqueue a finalizer + field snapshot. Non-blocking; the dedicated
/// finalizer thread will pick it up. Safe to call from GC sweep context.
pub fn enqueue(finalizer: FinalizerFn, fields: Vec<usize>) {
    ENQUEUED.fetch_add(1, Ordering::Release);
    // send() only fails if the receiver has been dropped, which only happens
    // when the process is exiting. In that case leaking is acceptable.
    let _ = sender().send(FinalizerWork { finalizer, fields });
}

/// Block until the finalizer thread has processed everything enqueued so far.
/// Used by tests that need to observe post-GC state deterministically.
pub fn drain() {
    // Snapshot enqueued first. Finalizer thread only increments COMPLETED
    // after running a callback, so once COMPLETED >= our snapshot, every
    // pre-drain-call finalizer has finished.
    let target = ENQUEUED.load(Ordering::Acquire);
    while COMPLETED.load(Ordering::Acquire) < target {
        std::thread::yield_now();
    }
}

/// Check a dead heap object and, if its struct has a registered finalizer,
/// snapshot its first two fields (ptr, size by convention) and enqueue work.
///
/// Called by each GC's sweep/evacuation pass. No-op for non-struct objects
/// (strings, keywords, collections, etc.) and for user structs without a
/// registered finalizer.
///
/// User structs in Beagle have `type_id == 0` with the struct_id stored in
/// `type_data`, and `opaque == false`. Raw byte allocations share `type_id == 0`
/// but are `opaque == true`, so both checks are required to avoid aliasing a
/// string's byte length onto an unrelated struct_id.
pub unsafe fn maybe_enqueue_finalizer(heap_object: &crate::types::HeapObject) {
    if heap_object.get_type_id() != 0 {
        return;
    }
    if heap_object.is_opaque_object() {
        return;
    }
    let struct_id = heap_object.get_struct_id();
    let Some(finalizer) = finalizer_for(struct_id) else {
        return;
    };
    // Snapshot only what the finalizer needs. By convention: field 0 = ptr,
    // field 1 = size. We snapshot at GC time; the finalizer thread never
    // touches beagle heap memory (the object may be reclaimed by then).
    let fields = vec![heap_object.get_field(0), heap_object.get_field(1)];
    enqueue(finalizer, fields);
}
