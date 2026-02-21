use libloading::Library;
use mmap_rs::{Mmap, MmapMut, MmapOptions, Reserved};
use nanoserde::SerJson;
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    ffi::{CString, c_void},
    io::Write,
    slice::{self},
    sync::{
        Arc, Condvar, Mutex, TryLockError,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle, Thread, ThreadId},
    time::Duration,
    vec,
};

use crate::{
    Alloc, CommandLineArguments, Data, Message,
    builtins::{__pause, debugger},
    compiler::{
        BlockingSender, CompilerMessage, CompilerResponse, CompilerThread, blocking_channel,
    },
    gc::{
        AllocateAction, Allocator, AllocatorOptions, STACK_SIZE, StackMap, StackMapDetails,
        usdt_probes,
    },
    ir::StringValue,
    types::{BuiltInTypes, Header, HeapObject, Tagged},
};

use crate::collections::{
    GcHandle, HandleScope, PersistentMap, PersistentVec, TYPE_ID_ATOM, TYPE_ID_CONS_STRING,
    TYPE_ID_CONTINUATION, TYPE_ID_CONTINUATION_SEGMENT, TYPE_ID_FUNCTION_OBJECT, TYPE_ID_KEYWORD,
    TYPE_ID_MULTI_ARITY_FUNCTION, TYPE_ID_PERSISTENT_MAP, TYPE_ID_PERSISTENT_SET,
    TYPE_ID_PERSISTENT_VEC, TYPE_ID_RAW_ARRAY, TYPE_ID_STRING, TYPE_ID_STRING_SLICE,
};

use std::cell::{Cell, RefCell};
use std::sync::mpsc;
use std::time::Instant;

// mio imports for async I/O
use mio::net::TcpStream;
use mio::{Events, Interest, Poll, Token, Waker};

// ============================================================================
// GlobalObject: Unified GC roots system
// ============================================================================

/// Number of root entries per GlobalObjectBlock (fixed size)
pub const GLOBAL_BLOCK_SIZE: usize = 64;

/// Header fields in GlobalObjectBlock: next_block and count
pub const GLOBAL_BLOCK_HEADER_FIELDS: usize = 2;

/// Total fields per block: header + entries
pub const GLOBAL_BLOCK_TOTAL_FIELDS: usize = GLOBAL_BLOCK_HEADER_FIELDS + GLOBAL_BLOCK_SIZE;

/// Marker value for free slots in GlobalObjectBlock (null value)
pub const GLOBAL_BLOCK_FREE_SLOT: usize = 0b111; // Same as BuiltInTypes::null_value()

// ============================================================================
// Reserved GlobalObject slots for runtime infrastructure
// ============================================================================

/// Reserved slot index for the namespaces atom.
/// This slot holds an Atom containing a PersistentMap of namespace bindings.
pub const GLOBAL_SLOT_NAMESPACES: usize = 0;

/// Reserved slot index for the current thread's Thread object.
/// Each thread stores its own Thread object here so GC can keep it alive.
pub const GLOBAL_SLOT_THREAD: usize = 1;

/// Number of reserved slots at the start of the first GlobalObjectBlock.
/// These slots are not available for general-purpose roots via add_root().
pub const GLOBAL_RESERVED_SLOTS: usize = 2;

/// Wrapper for accessing a GlobalObjectBlock on the heap.
///
/// Layout:
/// - Field 0: next_block (tagged pointer to next GlobalObjectBlock, or null)
/// - Field 1: count (tagged integer - number of entries used, NOT including freed slots)
/// - Fields 2..66: entries (tagged pointers to roots, or GLOBAL_BLOCK_FREE_SLOT if freed)
#[derive(Clone, Copy)]
pub struct GlobalObjectBlock {
    /// Tagged pointer to the heap object
    ptr: usize,
}

impl GlobalObjectBlock {
    /// Create from a tagged pointer (must be a valid heap pointer)
    #[inline]
    pub fn from_tagged(ptr: usize) -> Self {
        debug_assert!(BuiltInTypes::is_heap_pointer(ptr));
        GlobalObjectBlock { ptr }
    }

    /// Get the tagged pointer
    pub fn tagged_ptr(&self) -> usize {
        self.ptr
    }

    /// Get the next block in the linked list, if any
    pub fn next_block(&self) -> Option<GlobalObjectBlock> {
        let heap_obj = HeapObject::from_tagged(self.ptr);
        let next = heap_obj.get_field(0);
        if next == GLOBAL_BLOCK_FREE_SLOT || next == 0 {
            None
        } else if BuiltInTypes::is_heap_pointer(next) {
            Some(GlobalObjectBlock::from_tagged(next))
        } else {
            None
        }
    }

    /// Set the next block pointer
    pub fn set_next_block(&self, next: Option<GlobalObjectBlock>) {
        let heap_obj = HeapObject::from_tagged(self.ptr);
        let next_ptr = next.map_or(GLOBAL_BLOCK_FREE_SLOT, |b| b.ptr);
        heap_obj.write_field(0, next_ptr);
    }

    /// Get the count of active entries (note: doesn't account for freed slots)
    pub fn count(&self) -> usize {
        let heap_obj = HeapObject::from_tagged(self.ptr);
        let count_tagged = heap_obj.get_field(1);
        BuiltInTypes::untag(count_tagged)
    }

    /// Set the count of active entries
    pub fn set_count(&self, count: usize) {
        let heap_obj = HeapObject::from_tagged(self.ptr);
        let count_tagged = BuiltInTypes::construct_int(count as isize) as usize;
        heap_obj.write_field(1, count_tagged);
    }

    /// Get an entry at a given index (0-based, within this block only)
    #[inline]
    pub fn get_entry(&self, index: usize) -> usize {
        debug_assert!(index < GLOBAL_BLOCK_SIZE);
        let heap_obj = HeapObject::from_tagged(self.ptr);
        heap_obj.get_field(GLOBAL_BLOCK_HEADER_FIELDS + index)
    }

    /// Set an entry at a given index
    #[inline]
    pub fn set_entry(&self, index: usize, value: usize) {
        debug_assert!(index < GLOBAL_BLOCK_SIZE);
        let heap_obj = HeapObject::from_tagged(self.ptr);
        heap_obj.write_field((GLOBAL_BLOCK_HEADER_FIELDS + index) as i32, value);
    }

    /// Check if a slot is free
    #[inline]
    pub fn is_entry_free(&self, index: usize) -> bool {
        self.get_entry(index) == GLOBAL_BLOCK_FREE_SLOT
    }

    /// Get the HeapObject for this block
    pub fn as_heap_object(&self) -> HeapObject {
        HeapObject::from_tagged(self.ptr)
    }

    /// Initialize all entries to free slots and set count to 0
    pub fn initialize(&self) {
        for i in 0..GLOBAL_BLOCK_SIZE {
            self.set_entry(i, GLOBAL_BLOCK_FREE_SLOT);
        }
        self.set_count(0);
        self.set_next_block(None);
    }
}

/// Rust-side anchor for a thread's GlobalObject roots.
///
/// This struct lives in Rust memory (NOT heap-allocated), so GC can update
/// head_block when the GlobalObjectBlock moves during compaction.
///
/// This is the key to making temporary roots work with a moving GC:
/// - Rust code calls add_root/get_root via this struct
/// - GC updates head_block when it moves the GlobalObjectBlock
/// - Rust reads via (updated) head_block, so it always gets current values
pub struct ThreadGlobal {
    /// Tagged pointer to the first GlobalObjectBlock (GC updates this when block moves!)
    /// Must be atomic since it's read without holding the mutex lock
    pub head_block: AtomicUsize,
    /// Hint for fast slot allocation (global slot index across all blocks)
    next_free_slot: usize,
    /// Thread ID for debugging/validation
    pub thread_id: ThreadId,
    /// Stack base address for this thread (to update stack slot when head_block changes)
    pub stack_base: usize,
    /// Shadow stack for HandleScope: pre-allocated contiguous buffer of GC root values.
    /// HandleScope saves/restores `handle_stack_top` for zero-cost scope enter/exit.
    pub handle_stack: Vec<usize>,
    /// Current top index into handle_stack (next free slot).
    pub handle_stack_top: usize,
}

impl ThreadGlobal {
    /// Create a new ThreadGlobal with an already-allocated head block
    pub fn new(head_block: usize, thread_id: ThreadId, stack_base: usize) -> Self {
        debug_assert!(BuiltInTypes::is_heap_pointer(head_block));
        ThreadGlobal {
            head_block: AtomicUsize::new(head_block),
            next_free_slot: 0,
            thread_id,
            stack_base,
            handle_stack: vec![0; 1024],
            handle_stack_top: 0,
        }
    }

    /// Link a new block at the END of the chain.
    /// This preserves existing slot indices (unlike prepending which invalidates them).
    pub fn link_new_block(&mut self, new_block: usize) {
        debug_assert!(BuiltInTypes::is_heap_pointer(new_block));

        // Find the last block in the chain
        let mut current = GlobalObjectBlock::from_tagged(self.head_block.load(Ordering::SeqCst));
        let mut block_count = 1;
        while let Some(next) = current.next_block() {
            current = next;
            block_count += 1;
        }

        // Link the new block at the end
        let new_block_obj = GlobalObjectBlock::from_tagged(new_block);
        new_block_obj.set_next_block(None);
        current.set_next_block(Some(new_block_obj));

        // Update free slot hint to the first slot in the new block
        self.next_free_slot = block_count * GLOBAL_BLOCK_SIZE;
    }

    /// Add a root value. Returns a global slot index that can be used with get_root.
    ///
    /// If all slots are full, returns None - caller must allocate a new block first.
    /// Note: Slots 0..GLOBAL_RESERVED_SLOTS in the first block are reserved for
    /// runtime infrastructure (like the namespaces atom) and are skipped.
    pub fn add_root(&mut self, value: usize) -> Option<usize> {
        // Search for a free slot starting from next_free_slot hint
        let mut block_num = 0;
        let mut current = Some(GlobalObjectBlock::from_tagged(
            self.head_block.load(Ordering::SeqCst),
        ));

        while let Some(block) = current {
            // In the first block (block_num == 0), skip reserved slots
            let start_index = if block_num == 0 {
                GLOBAL_RESERVED_SLOTS
            } else {
                0
            };

            for i in start_index..GLOBAL_BLOCK_SIZE {
                if block.is_entry_free(i) {
                    block.set_entry(i, value);
                    let slot = block_num * GLOBAL_BLOCK_SIZE + i;
                    self.next_free_slot = slot + 1;
                    return Some(slot);
                }
            }
            current = block.next_block();
            block_num += 1;
        }

        // No free slot found
        None
    }

    /// Remove a root by slot index. Returns the value that was in the slot.
    pub fn remove_root(&mut self, slot: usize) -> usize {
        let (block, index) = self.find_slot(slot);
        let value = block.get_entry(index);
        block.set_entry(index, GLOBAL_BLOCK_FREE_SLOT);
        // Update hint if this slot is earlier
        if slot < self.next_free_slot {
            self.next_free_slot = slot;
        }
        value
    }

    /// Get a root value by slot index.
    ///
    /// This is the critical method for temporary roots: it follows head_block
    /// (which GC updates when the block moves) to read the current value.
    #[inline]
    pub fn get_root(&self, slot: usize) -> usize {
        let (block, index) = self.find_slot(slot);
        block.get_entry(index)
    }

    /// Set a root value by slot index.
    #[inline]
    pub fn set_root(&self, slot: usize, value: usize) {
        let (block, index) = self.find_slot(slot);
        block.set_entry(index, value);
    }

    /// Get this thread's Thread object from the reserved slot.
    #[inline]
    pub fn get_thread_object(&self) -> usize {
        let block = GlobalObjectBlock::from_tagged(self.head_block.load(Ordering::SeqCst));
        block.get_entry(GLOBAL_SLOT_THREAD)
    }

    /// Set this thread's Thread object in the reserved slot.
    #[inline]
    pub fn set_thread_object(&self, thread_obj: usize) {
        let block = GlobalObjectBlock::from_tagged(self.head_block.load(Ordering::SeqCst));
        block.set_entry(GLOBAL_SLOT_THREAD, thread_obj)
    }

    /// Find the block and local index for a global slot index
    #[inline]
    fn find_slot(&self, slot: usize) -> (GlobalObjectBlock, usize) {
        let block_num = slot / GLOBAL_BLOCK_SIZE;
        let local_index = slot % GLOBAL_BLOCK_SIZE;

        let mut current = GlobalObjectBlock::from_tagged(self.head_block.load(Ordering::SeqCst));
        for _ in 0..block_num {
            current = current.next_block().expect("slot index out of bounds");
        }

        (current, local_index)
    }

    /// Iterate over all blocks in this ThreadGlobal
    pub fn iter_blocks(&self) -> GlobalBlockIter {
        GlobalBlockIter {
            current: Some(GlobalObjectBlock::from_tagged(
                self.head_block.load(Ordering::SeqCst),
            )),
        }
    }

    /// Update the head block pointer (called by GC after copying)
    pub fn update_head_block(&mut self, new_ptr: usize) {
        debug_assert!(BuiltInTypes::is_heap_pointer(new_ptr));
        self.head_block.store(new_ptr, Ordering::SeqCst);
    }

    /// Check if the head block is full (all slots used).
    /// Returns true if we need to allocate a new block before adding a root.
    pub fn is_head_block_full(&self) -> bool {
        let block = GlobalObjectBlock::from_tagged(self.head_block.load(Ordering::SeqCst));
        for i in 0..GLOBAL_BLOCK_SIZE {
            if block.is_entry_free(i) {
                return false;
            }
        }
        true
    }

    /// Get the namespaces atom from slot 0.
    /// Returns the tagged pointer to the Atom, or null if not initialized.
    #[inline]
    pub fn get_namespaces_atom(&self) -> usize {
        self.get_root(GLOBAL_SLOT_NAMESPACES)
    }

    /// Set the namespaces atom in slot 0.
    /// The value should be a tagged pointer to an Atom containing a PersistentMap.
    #[inline]
    pub fn set_namespaces_atom(&self, atom_ptr: usize) {
        self.set_root(GLOBAL_SLOT_NAMESPACES, atom_ptr);
    }
}

/// Iterator over GlobalObjectBlocks in a ThreadGlobal
pub struct GlobalBlockIter {
    current: Option<GlobalObjectBlock>,
}

impl Iterator for GlobalBlockIter {
    type Item = GlobalObjectBlock;

    fn next(&mut self) -> Option<Self::Item> {
        let block = self.current?;
        self.current = block.next_block();
        Some(block)
    }
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub name: String,
    pub fields: Vec<String>,
    pub mutable_fields: Vec<bool>,
    /// Docstring for the struct (from /// comments in source)
    pub docstring: Option<String>,
    /// Docstrings for each field (in same order as fields)
    pub field_docstrings: Vec<Option<String>>,
}

impl Struct {
    pub fn size(&self) -> usize {
        self.fields.len()
    }

    pub fn is_field_mutable(&self, index: usize) -> bool {
        self.mutable_fields.get(index).copied().unwrap_or(false)
    }
}

/// Computed by StructManager, consumed by any GC for migrating objects to new layouts.
pub struct MigrationPlan {
    pub new_field_count: usize,
    pub new_shape_id: usize,
    /// For each field in the new layout: Some(old_index) or None (fill with null)
    pub field_map: Vec<Option<usize>>,
}

pub struct StructManager {
    name_to_id: HashMap<String, usize>,
    structs: Vec<Struct>,
    name_to_family_id: HashMap<String, usize>,
    shape_id_to_family_id: Vec<usize>,
    family_id_to_latest_shape: Vec<usize>,
    family_id_counter: usize,
}

impl Default for StructManager {
    fn default() -> Self {
        Self::new()
    }
}

impl StructManager {
    pub fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            structs: Vec::new(),
            name_to_family_id: HashMap::new(),
            shape_id_to_family_id: Vec::new(),
            family_id_to_latest_shape: Vec::new(),
            family_id_counter: 0,
        }
    }

    /// Insert a struct definition. Returns (shape_id, is_redefinition).
    pub fn insert(&mut self, name: String, s: Struct) -> (usize, bool) {
        let new_shape_id = self.structs.len();
        let is_redefinition = self.name_to_id.contains_key(&name);

        let family_id = if let Some(&existing_family) = self.name_to_family_id.get(&name) {
            // Redefinition: reuse family_id, update latest shape
            self.family_id_to_latest_shape[existing_family] = new_shape_id;
            existing_family
        } else {
            // New struct: allocate family_id
            let fid = self.family_id_counter;
            self.family_id_counter += 1;
            self.name_to_family_id.insert(name.clone(), fid);
            self.family_id_to_latest_shape.push(new_shape_id);
            fid
        };

        self.name_to_id.insert(name, new_shape_id);
        self.structs.push(s);
        self.shape_id_to_family_id.push(family_id);

        (new_shape_id, is_redefinition)
    }

    pub fn get(&self, name: &str) -> Option<(usize, &Struct)> {
        let id = self.name_to_id.get(name)?;
        self.structs.get(*id).map(|x| (*id, x))
    }

    pub fn get_by_id(&self, type_id: usize) -> Option<&Struct> {
        self.structs.get(type_id)
    }

    /// Check if a shape_id is the latest version for its family
    pub fn is_latest_shape(&self, shape_id: usize) -> bool {
        if let Some(&family_id) = self.shape_id_to_family_id.get(shape_id) {
            self.family_id_to_latest_shape[family_id] == shape_id
        } else {
            true
        }
    }

    /// Get the latest shape_id for the same family as the given shape_id
    pub fn get_latest_shape_for(&self, shape_id: usize) -> usize {
        if let Some(&family_id) = self.shape_id_to_family_id.get(shape_id) {
            self.family_id_to_latest_shape[family_id]
        } else {
            shape_id
        }
    }

    /// Get the family_id for a given shape_id
    pub fn get_family_id(&self, shape_id: usize) -> Option<usize> {
        self.shape_id_to_family_id.get(shape_id).copied()
    }

    /// Check if two shape_ids belong to the same family
    pub fn same_family(&self, a: usize, b: usize) -> bool {
        match (self.get_family_id(a), self.get_family_id(b)) {
            (Some(fa), Some(fb)) => fa == fb,
            _ => false,
        }
    }

    /// Build a migration map: for each field in new_shape, the index in old_shape (or None)
    fn build_migration_map(&self, old_shape: usize, new_shape: usize) -> Vec<Option<usize>> {
        let old_def = &self.structs[old_shape];
        let new_def = &self.structs[new_shape];
        new_def
            .fields
            .iter()
            .map(|new_field| old_def.fields.iter().position(|f| f == new_field))
            .collect()
    }

    /// Returns None if object doesn't need migration (already latest shape)
    pub fn migration_plan_for(&self, shape_id: usize) -> Option<MigrationPlan> {
        if self.is_latest_shape(shape_id) {
            return None;
        }
        let latest = self.get_latest_shape_for(shape_id);
        Some(MigrationPlan {
            new_field_count: self.structs[latest].fields.len(),
            new_shape_id: latest,
            field_map: self.build_migration_map(shape_id, latest),
        })
    }

    /// Iterate over all structs
    pub fn iter(&self) -> impl Iterator<Item = &Struct> {
        self.structs.iter()
    }
}

pub trait Printer: Send + Sync {
    fn print(&mut self, value: String);
    fn println(&mut self, value: String);
    fn print_byte(&mut self, byte: u8);
    // Gross just for testing. I'll need to do better;
    fn get_output(&self) -> Vec<String>;
    fn reset(&mut self);
}

pub struct DefaultPrinter;

impl Printer for DefaultPrinter {
    fn print(&mut self, value: String) {
        use std::io::Write;
        print!("{}", value);
        let _ = std::io::stdout().flush();
    }

    fn println(&mut self, value: String) {
        println!("{}", value);
    }

    fn print_byte(&mut self, byte: u8) {
        use std::io::Write;
        let _ = std::io::stdout().write_all(&[byte]);
    }

    fn get_output(&self) -> Vec<String> {
        vec![]
    }

    fn reset(&mut self) {
        // no-op: DefaultPrinter doesn't buffer output
    }
}

pub struct TestPrinter {
    pub output: Vec<String>,
    pub other_printer: Box<dyn Printer>,
}

impl TestPrinter {
    pub fn new(other_printer: Box<dyn Printer>) -> Self {
        Self {
            output: vec![],
            other_printer,
        }
    }
}

impl Printer for TestPrinter {
    fn print(&mut self, value: String) {
        self.output.push(value.clone());
        // self.other_printer.print(value);
    }

    fn println(&mut self, value: String) {
        self.output.push(value.clone() + "\n");
        // self.other_printer.println(value);
    }

    fn print_byte(&mut self, byte: u8) {
        // For testing, convert byte to a string representation
        self.output.push(format!("{}", byte as char));
    }

    fn get_output(&self) -> Vec<String> {
        self.output.clone()
    }

    fn reset(&mut self) {
        self.output.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FFIType {
    U8,
    U32,
    I32,
    F32,
    F64,
    Pointer,
    MutablePointer,
    String,
    Void,
    U16,
    U64,
    Structure(Vec<FFIType>),
}

#[derive(Debug, Clone)]
pub struct FFIInfo {
    pub name: String,
    pub function: RawPtr<u8>,
    pub number_of_arguments: usize,
    pub argument_types: Vec<FFIType>,
    pub return_type: FFIType,
}

/// Information about a callback (C → Beagle function pointer).
/// The trampoline uses `callback_index` to look up this info at call time,
/// so the Beagle function can be relocated by GC without invalidating the trampoline.
#[derive(Debug)]
pub struct CallbackInfo {
    pub trampoline_ptr: *const u8,
    pub beagle_fn: usize,
    pub arg_types: Vec<FFIType>,
    pub return_type: FFIType,
    pub gc_root_id: usize,
}

// Safety: CallbackInfo contains raw pointers that are only accessed from the main thread.
unsafe impl Send for CallbackInfo {}
unsafe impl Sync for CallbackInfo {}

pub struct ThreadState {
    /// (stack_base, frame_pointer, gc_return_addr) for each paused thread, keyed by ThreadId.
    /// Using HashMap ensures each thread has exactly one entry and unpause removes the correct one.
    pub stack_pointers: HashMap<ThreadId, (usize, usize, usize)>,
    // TODO: I probably don't want to do this here. This requires taking a mutex
    // not really ideal for c calls.
    pub c_calling_stack_pointers: HashMap<ThreadId, (usize, usize, usize)>,
}

impl ThreadState {
    pub fn paused_threads(&self) -> usize {
        self.stack_pointers.len()
    }

    pub fn pause(&mut self, stack_pointer: (usize, usize, usize)) {
        let thread_id = thread::current().id();
        let prev = self.stack_pointers.insert(thread_id, stack_pointer);
        debug_assert!(prev.is_none(), "Thread {:?} double-paused", thread_id);
    }

    pub fn unpause(&mut self) {
        let thread_id = thread::current().id();
        let removed = self.stack_pointers.remove(&thread_id);
        debug_assert!(removed.is_some(), "Thread {:?} not paused", thread_id);
    }

    pub fn register_c_call(&mut self, stack_pointer: (usize, usize, usize)) {
        let thread_id = thread::current().id();
        self.c_calling_stack_pointers
            .insert(thread_id, stack_pointer);
    }

    pub fn unregister_c_call(&mut self) {
        let thread_id = thread::current().id();
        self.c_calling_stack_pointers.remove(&thread_id);
    }

    pub fn clear(&mut self) {
        self.stack_pointers.clear();
    }
}

/// FutureWaitSet enables efficient waiting for future completion.
/// Instead of polling, threads can block on a condition variable until
/// any future completes and notifies waiters.
pub struct FutureWaitSet {
    /// Condition variable that all waiters block on
    pub cond: Condvar,
    /// Mutex protecting the wait count
    pub mutex: Mutex<usize>,
}

impl FutureWaitSet {
    pub fn new() -> Self {
        Self {
            cond: Condvar::new(),
            mutex: Mutex::new(0),
        }
    }

    /// Wait for a notification with timeout (in milliseconds).
    /// Returns true if notified, false if timed out.
    pub fn wait_timeout(&self, timeout_ms: u64) -> bool {
        let guard = self.mutex.lock().unwrap();
        let result = self
            .cond
            .wait_timeout(guard, Duration::from_millis(timeout_ms))
            .unwrap();
        !result.1.timed_out()
    }

    /// Wait for a notification without timeout.
    pub fn wait(&self) {
        let guard = self.mutex.lock().unwrap();
        let _guard = self.cond.wait(guard).unwrap();
    }

    /// Notify all waiting threads that a future has completed.
    pub fn notify_all(&self) {
        self.cond.notify_all();
    }
}

impl Default for FutureWaitSet {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// EventLoop - mio-based async I/O event loop
// ============================================================================

/// Token for the waker (used to wake the event loop from other threads)
const WAKER_TOKEN: Token = Token(0);

/// Represents a pending async operation registered with the event loop
#[derive(Debug)]
pub enum PendingOperation {
    /// TCP connection in progress
    TcpConnect {
        future_atom: usize,
        socket_id: usize,
        op_id: usize,
    },
    /// TCP accept operation (waiting for incoming connection)
    TcpAccept {
        future_atom: usize,
        listener_id: usize,
        op_id: usize,
    },
    /// Timer (deadline-based)
    Timer {
        future_atom: usize,
        deadline: Instant,
    },
    /// File operation (handled by thread pool, wakes event loop on completion)
    FileOp { future_atom: usize },
}

/// Result of a completed TCP operation
#[derive(Debug, Clone)]
pub enum TcpResult {
    /// Connection completed successfully
    ConnectOk {
        future_atom: usize,
        socket_id: usize,
        op_id: usize,
    },
    /// Connection failed
    ConnectErr {
        future_atom: usize,
        error: String,
        op_id: usize,
    },
    /// Accept completed successfully
    AcceptOk {
        future_atom: usize,
        socket_id: usize,
        listener_id: usize,
        op_id: usize,
    },
    /// Accept failed
    AcceptErr {
        future_atom: usize,
        error: String,
        op_id: usize,
    },
    /// Read completed successfully
    ReadOk {
        future_atom: usize,
        data: Vec<u8>,
        op_id: usize,
    },
    /// Read failed
    ReadErr {
        future_atom: usize,
        error: String,
        op_id: usize,
    },
    /// Write completed successfully
    WriteOk {
        future_atom: usize,
        bytes_written: usize,
        op_id: usize,
    },
    /// Write failed
    WriteErr {
        future_atom: usize,
        error: String,
        op_id: usize,
    },
}

/// Opaque handle for async file operations
/// This is a thread-safe identifier that does NOT contain any Beagle heap references
pub type OperationHandle = u64;

/// Result of a completed file operation from the thread pool
/// NOTE: This enum does NOT contain any Beagle heap references (future_atom removed)
/// Results are identified by their OperationHandle
#[derive(Debug)]
pub enum FileResultData {
    ReadOk { content: String },
    ReadErr { error: String },
    WriteOk { bytes_written: usize },
    WriteErr { error: String },
    DeleteOk,
    DeleteErr { error: String },
    StatOk { size: u64 },
    StatErr { error: String },
    ReadDirOk { entries: Vec<String> },
    ReadDirErr { error: String },
    BoolOk { value: bool },
    BoolErr { error: String },
    HandleOk { handle: u64 },
    HandleErr { error: String },
}

/// Completed result with its handle for lookup
#[derive(Debug)]
pub struct CompletedResult {
    pub handle: OperationHandle,
    pub data: FileResultData,
}

/// File operation to be executed by the thread pool
#[derive(Debug)]
pub enum FileOperation {
    Read { path: String },
    Write { path: String, content: Vec<u8> },
    Delete { path: String },
    Stat { path: String },
    ReadDir { path: String },
    Append { path: String, content: Vec<u8> },
    Exists { path: String },
    Rename { old_path: String, new_path: String },
    Copy { src_path: String, dest_path: String },
    Mkdir { path: String },
    MkdirAll { path: String },
    Rmdir { path: String },
    RmdirAll { path: String },
    IsDir { path: String },
    IsFile { path: String },
    FileOpen { path: String, mode: String },
    FileClose { handle_key: u64 },
    FileHandleRead { handle_key: u64, count: usize },
    FileHandleWrite { handle_key: u64, content: Vec<u8> },
    FileHandleReadLine { handle_key: u64 },
    FileHandleFlush { handle_key: u64 },
}

/// Task for the file I/O thread pool
/// NOTE: Uses OperationHandle instead of raw future_atom pointer
pub struct FileTask {
    pub op: FileOperation,
    pub handle: OperationHandle,
}

/// TCP operation to be submitted to a threaded event loop via channel
#[derive(Debug)]
pub enum TcpOperation {
    Connect {
        addr: std::net::SocketAddr,
        future_atom: usize,
    },
    Accept {
        listener_id: usize,
        future_atom: usize,
    },
    Read {
        socket_id: usize,
        buffer_size: usize,
        future_atom: usize,
    },
    Write {
        socket_id: usize,
        data: Vec<u8>,
        future_atom: usize,
    },
    Listen {
        addr: std::net::SocketAddr,
        backlog: u32,
        response_tx: mpsc::SyncSender<Result<usize, String>>,
    },
    Close {
        socket_id: usize,
    },
    CloseListener {
        listener_id: usize,
    },
}

/// Task submitted to a threaded event loop
pub struct TcpTask {
    pub op: TcpOperation,
}

/// Event loop for async I/O operations using mio
/// Always runs a dedicated I/O thread that handles polling and TCP operations.
/// State is shared via Arc<Mutex<EventLoopState>> for access from both the
/// I/O thread and calling threads.
pub struct EventLoop {
    /// Shared I/O state — accessible from both I/O thread and callers via lock
    state: Arc<Mutex<EventLoopState>>,
    /// Waker to wake the event loop from other threads
    waker: Arc<Waker>,
    /// Current result being inspected (after pop), per-thread for concurrent safety.
    current_results: Mutex<HashMap<ThreadId, TcpResult>>,
    /// Channel to send file operations to thread pool
    file_task_tx: mpsc::Sender<FileTask>,
    /// Channel sender for submitting TCP operations to the event loop thread
    tcp_task_tx: mpsc::Sender<TcpTask>,
    /// Shutdown flag for the event loop thread
    shutdown: Arc<AtomicBool>,
    /// Handle to the event loop thread
    event_loop_thread: Mutex<Option<JoinHandle<()>>>,
    /// Condvar to notify consumers when results are available
    results_notify: Arc<(Mutex<bool>, Condvar)>,
}

impl EventLoop {
    /// Create a new event loop with a file I/O thread pool and a dedicated I/O thread
    pub fn new(pool_size: usize) -> std::io::Result<Self> {
        let poll = Poll::new()?;
        let waker = Arc::new(Waker::new(poll.registry(), WAKER_TOKEN)?);

        // Create channels for file I/O thread pool
        let (file_task_tx, file_task_rx) = mpsc::channel::<FileTask>();
        let (file_result_tx, file_result_rx) = mpsc::channel::<CompletedResult>();

        // Wrap receiver in Arc<Mutex> for sharing among worker threads
        let shared_rx = Arc::new(Mutex::new(file_task_rx));

        // Clone waker for thread pool workers to wake event loop
        let waker_clone = waker.clone();

        // Spawn file I/O worker threads
        for _ in 0..pool_size {
            let rx = shared_rx.clone();
            let tx = file_result_tx.clone();
            let waker = waker_clone.clone();

            std::thread::spawn(move || {
                file_worker_loop_shared(rx, tx, waker);
            });
        }

        let state = EventLoopState {
            poll: Some(poll),
            events: Some(Events::with_capacity(128)),
            pending_ops: HashMap::new(),
            next_token: 1, // Start at 1, 0 is reserved for waker
            tcp_streams: HashMap::new(),
            tcp_listeners: HashMap::new(),
            next_socket_id: 1,
            token_to_socket: HashMap::new(),
            token_to_listener: HashMap::new(),
            completed_tcp_results: Vec::new(),
            timers: HashMap::new(),
            next_timer_id: 1,
            completed_timers: Vec::new(),
            next_handle: AtomicU64::new(1),
            completed_file_results: HashMap::new(),
            next_op_id: 1,
            file_result_rx: Mutex::new(file_result_rx),
            socket_tokens: HashMap::new(),
            pending_reads: HashMap::new(),
            pending_writes: HashMap::new(),
        };

        let state = Arc::new(Mutex::new(state));
        let shutdown = Arc::new(AtomicBool::new(false));
        let results_notify = Arc::new((Mutex::new(false), Condvar::new()));

        // Create TCP task channel
        let (tcp_task_tx, tcp_task_rx) = mpsc::channel::<TcpTask>();

        // Spawn I/O thread immediately
        let thread_state = state.clone();
        let thread_shutdown = shutdown.clone();
        let thread_notify = results_notify.clone();
        let thread_waker = waker.clone();

        let handle = std::thread::Builder::new()
            .name("event-loop-thread".to_string())
            .spawn(move || {
                event_loop_thread_main(
                    thread_state,
                    tcp_task_rx,
                    thread_waker,
                    thread_shutdown,
                    thread_notify,
                );
            })
            .expect("Failed to spawn event loop thread");

        Ok(Self {
            state,
            waker,
            current_results: Mutex::new(HashMap::new()),
            file_task_tx,
            tcp_task_tx,
            shutdown,
            event_loop_thread: Mutex::new(Some(handle)),
            results_notify,
        })
    }

    /// Submit a TCP operation to the event loop thread
    pub fn submit_tcp_op(&self, op: TcpOperation) -> Result<(), String> {
        self.tcp_task_tx
            .send(TcpTask { op })
            .map_err(|e| format!("Failed to submit TCP operation: {}", e))?;
        // Wake the event loop thread so it drains the queue promptly
        let _ = self.waker.wake();
        Ok(())
    }

    /// Submit a TCP listen operation synchronously (blocks until listener_id is returned)
    pub fn submit_tcp_listen(
        &self,
        addr: std::net::SocketAddr,
        backlog: u32,
    ) -> Result<usize, String> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        self.submit_tcp_op(TcpOperation::Listen {
            addr,
            backlog,
            response_tx,
        })?;
        response_rx
            .recv()
            .map_err(|e| format!("Failed to receive listen response: {}", e))?
    }

    /// Get the count of completed TCP results
    pub fn tcp_results_count(&self) -> usize {
        self.state
            .lock()
            .map(|s| s.completed_tcp_results.len())
            .unwrap_or(0)
    }

    /// Pop the next completed TCP result
    pub fn pop_tcp_result(&self) -> Option<TcpResult> {
        let mut state = self.state.lock().ok()?;
        if state.completed_tcp_results.is_empty() {
            None
        } else {
            Some(state.completed_tcp_results.remove(0))
        }
    }

    /// Pop a completed TCP result matching a specific future_atom
    /// This ensures each thread only retrieves its own results when multiple
    /// threads share the same event loop.
    pub fn pop_tcp_result_for_atom(&self, future_atom: usize) -> Option<TcpResult> {
        let mut state = self.state.lock().ok()?;
        let pos = state.completed_tcp_results.iter().position(|r| match r {
            TcpResult::ConnectOk {
                future_atom: fa, ..
            } => *fa == future_atom,
            TcpResult::ConnectErr {
                future_atom: fa, ..
            } => *fa == future_atom,
            TcpResult::AcceptOk {
                future_atom: fa, ..
            } => *fa == future_atom,
            TcpResult::AcceptErr {
                future_atom: fa, ..
            } => *fa == future_atom,
            TcpResult::ReadOk {
                future_atom: fa, ..
            } => *fa == future_atom,
            TcpResult::ReadErr {
                future_atom: fa, ..
            } => *fa == future_atom,
            TcpResult::WriteOk {
                future_atom: fa, ..
            } => *fa == future_atom,
            TcpResult::WriteErr {
                future_atom: fa, ..
            } => *fa == future_atom,
        });
        pos.map(|i| state.completed_tcp_results.remove(i))
    }

    /// Shutdown the event loop thread gracefully
    pub fn shutdown_thread(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Wake the thread so it sees the shutdown flag
        let _ = self.waker.wake();
        if let Ok(mut guard) = self.event_loop_thread.lock() {
            if let Some(handle) = guard.take() {
                let _ = handle.join();
            }
        }
    }

    /// Submit a file operation to the thread pool
    /// Returns the operation handle that can be used to poll for results
    pub fn submit_file_op(&self, op: FileOperation) -> Result<OperationHandle, String> {
        let handle = self.state.lock().unwrap().next_operation_handle();
        self.file_task_tx
            .send(FileTask { op, handle })
            .map_err(|e| format!("Failed to submit file operation: {}", e))?;
        Ok(handle)
    }

    /// Wake the event loop from another thread
    pub fn wake(&self) -> std::io::Result<()> {
        self.waker.wake()
    }

    /// Get a clone of the waker for external use
    pub fn get_waker(&self) -> Arc<Waker> {
        self.waker.clone()
    }

    /// Get a reference to the results_notify Condvar
    pub fn results_notify(&self) -> &Arc<(Mutex<bool>, Condvar)> {
        &self.results_notify
    }

    /// Set a timer that fires after delay_ms milliseconds
    pub fn timer_set(&self, delay_ms: u64, future_atom: usize) -> usize {
        self.state.lock().unwrap().timer_set(delay_ms, future_atom)
    }

    /// Cancel a timer by ID
    pub fn timer_cancel(&self, timer_id: usize) -> bool {
        self.state.lock().unwrap().timer_cancel(timer_id)
    }

    /// Get the number of pending completed timers
    pub fn completed_timers_len(&self) -> usize {
        self.state.lock().unwrap().completed_timers_len()
    }

    /// Pop the next completed timer's future_atom
    pub fn pop_completed_timer(&self) -> Option<usize> {
        self.state.lock().unwrap().pop_completed_timer()
    }

    /// Get the number of completed file results waiting
    pub fn file_results_count(&self) -> usize {
        self.state.lock().unwrap().file_results_count()
    }

    /// Check if a result is ready for the given handle
    pub fn file_result_ready(&self, handle: OperationHandle) -> bool {
        self.state.lock().unwrap().file_result_ready(handle)
    }

    /// Poll for a file result by handle
    pub fn file_result_poll(&self, handle: OperationHandle) -> Option<FileResultData> {
        self.state.lock().unwrap().file_result_poll(handle)
    }

    /// Peek at a file result's type code without removing it
    pub fn file_result_peek_type(&self, handle: OperationHandle) -> Option<usize> {
        self.state.lock().unwrap().file_result_peek_type(handle)
    }

    /// Get the type code for a FileResultData
    pub fn file_result_type_code(data: &FileResultData) -> usize {
        match data {
            FileResultData::ReadOk { .. } => 1,
            FileResultData::ReadErr { .. } => 2,
            FileResultData::WriteOk { .. } => 3,
            FileResultData::WriteErr { .. } => 4,
            FileResultData::DeleteOk => 5,
            FileResultData::DeleteErr { .. } => 6,
            FileResultData::StatOk { .. } => 7,
            FileResultData::StatErr { .. } => 8,
            FileResultData::ReadDirOk { .. } => 9,
            FileResultData::ReadDirErr { .. } => 10,
            FileResultData::BoolOk { .. } => 11,
            FileResultData::BoolErr { .. } => 12,
            FileResultData::HandleOk { .. } => 13,
            FileResultData::HandleErr { .. } => 14,
        }
    }

    /// Get the string data from a FileResultData
    pub fn file_result_string_data(data: &FileResultData) -> Option<&str> {
        match data {
            FileResultData::ReadOk { content } => Some(content),
            FileResultData::ReadErr { error } => Some(error),
            FileResultData::WriteErr { error } => Some(error),
            FileResultData::DeleteErr { error } => Some(error),
            FileResultData::StatErr { error } => Some(error),
            FileResultData::ReadDirErr { error } => Some(error),
            FileResultData::BoolErr { error } => Some(error),
            FileResultData::HandleErr { error } => Some(error),
            _ => None,
        }
    }

    /// Get the numeric value from a FileResultData
    pub fn file_result_value(data: &FileResultData) -> usize {
        match data {
            FileResultData::WriteOk { bytes_written } => *bytes_written,
            FileResultData::StatOk { size } => *size as usize,
            FileResultData::BoolOk { value } => {
                if *value {
                    1
                } else {
                    0
                }
            }
            FileResultData::HandleOk { handle } => *handle as usize,
            _ => 0,
        }
    }

    /// Get the directory entries from a ReadDirOk result
    pub fn file_result_entries(data: &FileResultData) -> Option<&Vec<String>> {
        match data {
            FileResultData::ReadDirOk { entries } => Some(entries),
            _ => None,
        }
    }

    /// Set the current result for the calling thread (for inspection after pop)
    pub fn set_current_result(&self, result: TcpResult) {
        let tid = std::thread::current().id();
        self.current_results.lock().unwrap().insert(tid, result);
    }

    /// Get the current result for the calling thread (cloned)
    pub fn current_result(&self) -> Option<TcpResult> {
        let tid = std::thread::current().id();
        self.current_results.lock().unwrap().get(&tid).cloned()
    }
}

// ============================================================================
// File Handle Registry - global registry for handle-based file operations
// ============================================================================
// Stores open file handles accessible from worker threads.
// Keys are u64 identifiers, values are BufReader<File> for buffered I/O.

static FILE_HANDLE_COUNTER: AtomicU64 = AtomicU64::new(1);

use std::io::{BufRead, BufReader};
use std::sync::LazyLock;

static FILE_REGISTRY: LazyLock<Mutex<HashMap<u64, BufReader<std::fs::File>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Worker loop for file I/O threads with shared receiver
fn file_worker_loop_shared(
    rx: Arc<Mutex<mpsc::Receiver<FileTask>>>,
    tx: mpsc::Sender<CompletedResult>,
    waker: Arc<Waker>,
) {
    // Note: In a full implementation, we'd register this thread with the GC
    // For now, file operations are short-lived and don't allocate on the Beagle heap

    loop {
        // Lock the receiver to get a task
        let task = {
            let guard = rx.lock().unwrap();
            guard.recv()
        };

        match task {
            Ok(task) => {
                let data = execute_file_op(task.op);
                let result = CompletedResult {
                    handle: task.handle,
                    data,
                };
                if tx.send(result).is_ok() {
                    // Wake the event loop to process the result
                    let _ = waker.wake();
                }
            }
            Err(_) => {
                // Channel closed, exit the loop
                break;
            }
        }
    }
}

/// Execute a file operation and return the result data
/// NOTE: This function does NOT receive any Beagle heap references
fn execute_file_op(op: FileOperation) -> FileResultData {
    match op {
        FileOperation::Read { path } => match std::fs::read_to_string(&path) {
            Ok(content) => FileResultData::ReadOk { content },
            Err(e) => FileResultData::ReadErr {
                error: e.to_string(),
            },
        },
        FileOperation::Write { path, content } => match std::fs::write(&path, &content) {
            Ok(()) => FileResultData::WriteOk {
                bytes_written: content.len(),
            },
            Err(e) => FileResultData::WriteErr {
                error: e.to_string(),
            },
        },
        FileOperation::Delete { path } => match std::fs::remove_file(&path) {
            Ok(()) => FileResultData::DeleteOk,
            Err(e) => FileResultData::DeleteErr {
                error: e.to_string(),
            },
        },
        FileOperation::Stat { path } => match std::fs::metadata(&path) {
            Ok(meta) => FileResultData::StatOk { size: meta.len() },
            Err(e) => FileResultData::StatErr {
                error: e.to_string(),
            },
        },
        FileOperation::ReadDir { path } => match std::fs::read_dir(&path) {
            Ok(entries) => {
                let names: Vec<String> = entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                    .collect();
                FileResultData::ReadDirOk { entries: names }
            }
            Err(e) => FileResultData::ReadDirErr {
                error: e.to_string(),
            },
        },
        FileOperation::Append { path, content } => {
            use std::io::Write as IoWrite;
            match std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&path)
            {
                Ok(mut file) => match file.write_all(&content) {
                    Ok(()) => FileResultData::WriteOk {
                        bytes_written: content.len(),
                    },
                    Err(e) => FileResultData::WriteErr {
                        error: e.to_string(),
                    },
                },
                Err(e) => FileResultData::WriteErr {
                    error: e.to_string(),
                },
            }
        }
        FileOperation::Exists { path } => FileResultData::BoolOk {
            value: std::path::Path::new(&path).exists(),
        },
        FileOperation::Rename { old_path, new_path } => match std::fs::rename(&old_path, &new_path)
        {
            Ok(()) => FileResultData::DeleteOk,
            Err(e) => FileResultData::DeleteErr {
                error: e.to_string(),
            },
        },
        FileOperation::Copy {
            src_path,
            dest_path,
        } => match std::fs::copy(&src_path, &dest_path) {
            Ok(bytes) => FileResultData::WriteOk {
                bytes_written: bytes as usize,
            },
            Err(e) => FileResultData::WriteErr {
                error: e.to_string(),
            },
        },
        FileOperation::Mkdir { path } => match std::fs::create_dir(&path) {
            Ok(()) => FileResultData::DeleteOk,
            Err(e) => FileResultData::DeleteErr {
                error: e.to_string(),
            },
        },
        FileOperation::MkdirAll { path } => match std::fs::create_dir_all(&path) {
            Ok(()) => FileResultData::DeleteOk,
            Err(e) => FileResultData::DeleteErr {
                error: e.to_string(),
            },
        },
        FileOperation::Rmdir { path } => match std::fs::remove_dir(&path) {
            Ok(()) => FileResultData::DeleteOk,
            Err(e) => FileResultData::DeleteErr {
                error: e.to_string(),
            },
        },
        FileOperation::RmdirAll { path } => match std::fs::remove_dir_all(&path) {
            Ok(()) => FileResultData::DeleteOk,
            Err(e) => FileResultData::DeleteErr {
                error: e.to_string(),
            },
        },
        FileOperation::IsDir { path } => FileResultData::BoolOk {
            value: std::path::Path::new(&path).is_dir(),
        },
        FileOperation::IsFile { path } => FileResultData::BoolOk {
            value: std::path::Path::new(&path).is_file(),
        },
        FileOperation::FileOpen { path, mode } => {
            let file_result = match mode.as_str() {
                "r" => std::fs::File::open(&path),
                "w" => std::fs::File::create(&path),
                "a" => std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(&path),
                _ => {
                    return FileResultData::HandleErr {
                        error: format!("Unknown file mode: {}", mode),
                    };
                }
            };
            match file_result {
                Ok(file) => {
                    let key = FILE_HANDLE_COUNTER.fetch_add(1, Ordering::Relaxed);
                    let reader = BufReader::new(file);
                    FILE_REGISTRY.lock().unwrap().insert(key, reader);
                    FileResultData::HandleOk { handle: key }
                }
                Err(e) => FileResultData::HandleErr {
                    error: e.to_string(),
                },
            }
        }
        FileOperation::FileClose { handle_key } => {
            match FILE_REGISTRY.lock().unwrap().remove(&handle_key) {
                Some(_) => FileResultData::DeleteOk,
                None => FileResultData::DeleteErr {
                    error: format!("Invalid file handle: {}", handle_key),
                },
            }
        }
        FileOperation::FileHandleRead { handle_key, count } => {
            use std::io::Read as IoRead;
            let mut registry = FILE_REGISTRY.lock().unwrap();
            match registry.get_mut(&handle_key) {
                Some(reader) => {
                    let mut buf = vec![0u8; count];
                    match reader.read(&mut buf) {
                        Ok(n) => {
                            buf.truncate(n);
                            match String::from_utf8(buf) {
                                Ok(s) => FileResultData::ReadOk { content: s },
                                Err(e) => FileResultData::ReadErr {
                                    error: e.to_string(),
                                },
                            }
                        }
                        Err(e) => FileResultData::ReadErr {
                            error: e.to_string(),
                        },
                    }
                }
                None => FileResultData::ReadErr {
                    error: format!("Invalid file handle: {}", handle_key),
                },
            }
        }
        FileOperation::FileHandleWrite {
            handle_key,
            content,
        } => {
            use std::io::Write as IoWrite;
            let mut registry = FILE_REGISTRY.lock().unwrap();
            match registry.get_mut(&handle_key) {
                Some(reader) => {
                    // Get the underlying file from BufReader to write
                    let file = reader.get_mut();
                    match file.write_all(&content) {
                        Ok(()) => FileResultData::WriteOk {
                            bytes_written: content.len(),
                        },
                        Err(e) => FileResultData::WriteErr {
                            error: e.to_string(),
                        },
                    }
                }
                None => FileResultData::WriteErr {
                    error: format!("Invalid file handle: {}", handle_key),
                },
            }
        }
        FileOperation::FileHandleReadLine { handle_key } => {
            let mut registry = FILE_REGISTRY.lock().unwrap();
            match registry.get_mut(&handle_key) {
                Some(reader) => {
                    let mut line = String::new();
                    match reader.read_line(&mut line) {
                        Ok(0) => FileResultData::ReadOk {
                            content: String::new(),
                        },
                        Ok(_) => FileResultData::ReadOk { content: line },
                        Err(e) => FileResultData::ReadErr {
                            error: e.to_string(),
                        },
                    }
                }
                None => FileResultData::ReadErr {
                    error: format!("Invalid file handle: {}", handle_key),
                },
            }
        }
        FileOperation::FileHandleFlush { handle_key } => {
            use std::io::Write as IoWrite;
            let mut registry = FILE_REGISTRY.lock().unwrap();
            match registry.get_mut(&handle_key) {
                Some(reader) => {
                    let file = reader.get_mut();
                    match file.flush() {
                        Ok(()) => FileResultData::DeleteOk,
                        Err(e) => FileResultData::DeleteErr {
                            error: e.to_string(),
                        },
                    }
                }
                None => FileResultData::DeleteErr {
                    error: format!("Invalid file handle: {}", handle_key),
                },
            }
        }
    }
}

// ============================================================================
// EventLoopState - Unified mutable I/O state for both modes
// ============================================================================

/// Pending read operation on a socket
#[allow(dead_code)]
struct PendingReadOp {
    future_atom: usize,
    buffer_size: usize,
    op_id: usize,
}

/// Pending write operation on a socket
#[allow(dead_code)]
struct PendingWriteOp {
    future_atom: usize,
    data: Vec<u8>,
    bytes_written: usize,
    op_id: usize,
}

/// Mutable I/O state used by both InProcess and Threaded event loop modes.
/// In InProcess mode, this lives inside EventLoop via Option<EventLoopState>.
/// In Threaded mode, this is moved to the dedicated event loop thread.
pub struct EventLoopState {
    /// Poll and Events are Option so the I/O thread can temporarily take them
    /// out during the blocking poll() call, allowing other threads to access
    /// the rest of the state without contention.
    poll: Option<Poll>,
    events: Option<Events>,
    pending_ops: HashMap<usize, PendingOperation>,
    next_token: usize,
    tcp_streams: HashMap<usize, TcpStream>,
    tcp_listeners: HashMap<usize, mio::net::TcpListener>,
    next_socket_id: usize,
    token_to_socket: HashMap<usize, usize>,
    token_to_listener: HashMap<usize, usize>,
    completed_tcp_results: Vec<TcpResult>,
    timers: HashMap<usize, (Instant, usize)>,
    next_timer_id: usize,
    completed_timers: Vec<usize>,
    next_op_id: usize,
    file_result_rx: Mutex<mpsc::Receiver<CompletedResult>>,
    completed_file_results: HashMap<OperationHandle, FileResultData>,
    next_handle: AtomicU64,
    // Per-socket tracking for concurrent read/write
    socket_tokens: HashMap<usize, Token>, // socket_id → stable Token
    pending_reads: HashMap<usize, PendingReadOp>, // socket_id → pending read
    pending_writes: HashMap<usize, PendingWriteOp>, // socket_id → pending write
}

impl EventLoopState {
    /// Get a mutable reference to the poll (panics if taken by I/O thread during poll)
    fn poll_mut(&mut self) -> &mut Poll {
        self.poll
            .as_mut()
            .expect("Poll temporarily taken by I/O thread")
    }

    fn next_token(&mut self) -> Token {
        let token = Token(self.next_token);
        self.next_token += 1;
        token
    }

    fn next_socket_id(&mut self) -> usize {
        let id = self.next_socket_id;
        self.next_socket_id += 1;
        id
    }

    fn next_op_id(&mut self) -> usize {
        let op_id = self.next_op_id;
        self.next_op_id += 1;
        op_id
    }

    /// Get or create a stable token for a socket
    fn get_or_create_socket_token(&mut self, socket_id: usize) -> Token {
        if let Some(&token) = self.socket_tokens.get(&socket_id) {
            token
        } else {
            let token = self.next_token();
            self.socket_tokens.insert(socket_id, token);
            self.token_to_socket.insert(token.0, socket_id);
            token
        }
    }

    /// Compute the combined interest for a socket based on pending operations
    fn socket_interest(&self, socket_id: usize) -> Option<Interest> {
        let has_read = self.pending_reads.contains_key(&socket_id);
        let has_write = self.pending_writes.contains_key(&socket_id);
        match (has_read, has_write) {
            (true, true) => Some(Interest::READABLE | Interest::WRITABLE),
            (true, false) => Some(Interest::READABLE),
            (false, true) => Some(Interest::WRITABLE),
            (false, false) => None,
        }
    }

    /// Register/reregister a socket with its current combined interest
    fn update_socket_registration(&mut self, socket_id: usize) -> Result<(), String> {
        let interest = match self.socket_interest(socket_id) {
            Some(i) => i,
            None => return Ok(()), // No pending ops, nothing to register
        };
        let token = self.get_or_create_socket_token(socket_id);
        if let Some(mut stream) = self.tcp_streams.remove(&socket_id) {
            let registry = self.poll_mut().registry();
            let result = registry
                .register(&mut stream, token, interest)
                .or_else(|_| registry.reregister(&mut stream, token, interest));
            self.tcp_streams.insert(socket_id, stream);
            result.map_err(|e| e.to_string())
        } else {
            Err("Socket not found".to_string())
        }
    }

    /// Handle a TCP task submitted via channel
    fn handle_tcp_task(&mut self, task: TcpTask) {
        match task.op {
            TcpOperation::Connect { addr, future_atom } => match TcpStream::connect(addr) {
                Ok(mut stream) => {
                    let socket_id = self.next_socket_id();
                    let token = self.next_token();
                    let op_id = self.next_op_id();

                    if let Err(e) =
                        self.poll_mut()
                            .registry()
                            .register(&mut stream, token, Interest::WRITABLE)
                    {
                        self.completed_tcp_results.push(TcpResult::ConnectErr {
                            future_atom,
                            error: e.to_string(),
                            op_id,
                        });
                        return;
                    }

                    self.tcp_streams.insert(socket_id, stream);
                    self.token_to_socket.insert(token.0, socket_id);
                    self.pending_ops.insert(
                        token.0,
                        PendingOperation::TcpConnect {
                            future_atom,
                            socket_id,
                            op_id,
                        },
                    );
                }
                Err(e) => {
                    let op_id = self.next_op_id();
                    self.completed_tcp_results.push(TcpResult::ConnectErr {
                        future_atom,
                        error: e.to_string(),
                        op_id,
                    });
                }
            },
            TcpOperation::Listen {
                addr,
                backlog: _,
                response_tx,
            } => match mio::net::TcpListener::bind(addr) {
                Ok(listener) => {
                    let listener_id = self.next_socket_id();
                    self.tcp_listeners.insert(listener_id, listener);
                    let _ = response_tx.send(Ok(listener_id));
                }
                Err(e) => {
                    let _ = response_tx.send(Err(e.to_string()));
                }
            },
            TcpOperation::Accept {
                listener_id,
                future_atom,
            } => {
                if let Some(mut listener) = self.tcp_listeners.remove(&listener_id) {
                    let token = self.next_token();
                    let op_id = self.next_op_id();

                    let registry = self.poll_mut().registry();
                    let result = registry
                        .register(&mut listener, token, Interest::READABLE)
                        .or_else(|_| registry.reregister(&mut listener, token, Interest::READABLE));

                    self.tcp_listeners.insert(listener_id, listener);

                    if let Err(e) = result {
                        self.completed_tcp_results.push(TcpResult::AcceptErr {
                            future_atom,
                            error: e.to_string(),
                            op_id,
                        });
                        return;
                    }

                    self.token_to_listener.insert(token.0, listener_id);
                    self.pending_ops.insert(
                        token.0,
                        PendingOperation::TcpAccept {
                            future_atom,
                            listener_id,
                            op_id,
                        },
                    );
                } else {
                    let op_id = self.next_op_id();
                    self.completed_tcp_results.push(TcpResult::AcceptErr {
                        future_atom,
                        error: "Listener not found".to_string(),
                        op_id,
                    });
                }
            }
            TcpOperation::Read {
                socket_id,
                buffer_size,
                future_atom,
            } => {
                if self.tcp_streams.contains_key(&socket_id) {
                    let op_id = self.next_op_id();
                    self.pending_reads.insert(
                        socket_id,
                        PendingReadOp {
                            future_atom,
                            buffer_size,
                            op_id,
                        },
                    );
                    if let Err(e) = self.update_socket_registration(socket_id) {
                        self.pending_reads.remove(&socket_id);
                        self.completed_tcp_results.push(TcpResult::ReadErr {
                            future_atom,
                            error: e,
                            op_id,
                        });
                    }
                } else {
                    let op_id = self.next_op_id();
                    self.completed_tcp_results.push(TcpResult::ReadErr {
                        future_atom,
                        error: "Socket not found".to_string(),
                        op_id,
                    });
                }
            }
            TcpOperation::Write {
                socket_id,
                data,
                future_atom,
            } => {
                if self.tcp_streams.contains_key(&socket_id) {
                    let op_id = self.next_op_id();
                    self.pending_writes.insert(
                        socket_id,
                        PendingWriteOp {
                            future_atom,
                            data,
                            bytes_written: 0,
                            op_id,
                        },
                    );
                    if let Err(e) = self.update_socket_registration(socket_id) {
                        self.pending_writes.remove(&socket_id);
                        self.completed_tcp_results.push(TcpResult::WriteErr {
                            future_atom,
                            error: e,
                            op_id,
                        });
                    }
                } else {
                    let op_id = self.next_op_id();
                    self.completed_tcp_results.push(TcpResult::WriteErr {
                        future_atom,
                        error: "Socket not found".to_string(),
                        op_id,
                    });
                }
            }
            TcpOperation::Close { socket_id } => {
                if let Some(mut stream) = self.tcp_streams.remove(&socket_id) {
                    let _ = self.poll_mut().registry().deregister(&mut stream);
                }
                self.pending_reads.remove(&socket_id);
                self.pending_writes.remove(&socket_id);
                if let Some(token) = self.socket_tokens.remove(&socket_id) {
                    self.token_to_socket.remove(&token.0);
                }
            }
            TcpOperation::CloseListener { listener_id } => {
                if let Some(mut listener) = self.tcp_listeners.remove(&listener_id) {
                    let _ = self.poll_mut().registry().deregister(&mut listener);
                }
            }
        }
    }

    /// Compute the effective poll timeout based on pending timers
    fn compute_poll_timeout(&self, max_ms: u64) -> Duration {
        if self.timers.is_empty() {
            Duration::from_millis(max_ms)
        } else {
            let now = Instant::now();
            let nearest_deadline = self
                .timers
                .values()
                .map(|(deadline, _)| *deadline)
                .min()
                .unwrap();
            let time_to_timer = nearest_deadline.saturating_duration_since(now);
            std::cmp::min(Duration::from_millis(max_ms), time_to_timer)
        }
    }

    /// Process polled events, file results, and expired timers.
    /// Called by the I/O thread after poll() returns with the events.
    fn process_events_and_timers(&mut self, events: &Events) -> usize {
        let mut processed = 0;
        let mut tokens_to_process: Vec<(usize, bool, bool)> = Vec::new();

        for event in events.iter() {
            if event.token() == WAKER_TOKEN {
                processed += 1;
            } else {
                tokens_to_process.push((event.token().0, event.is_readable(), event.is_writable()));
            }
        }

        for (token_id, is_readable, is_writable) in tokens_to_process {
            // First check pending_ops for Connect/Accept operations
            if let Some(op) = self.pending_ops.remove(&token_id) {
                self.handle_tcp_event(token_id, op);
                processed += 1;
                continue;
            }

            // Check per-socket read/write operations
            if let Some(&socket_id) = self.token_to_socket.get(&token_id) {
                if is_readable {
                    if let Some(read_op) = self.pending_reads.remove(&socket_id) {
                        self.handle_socket_read(socket_id, read_op);
                        processed += 1;
                    }
                }
                if is_writable {
                    if let Some(write_op) = self.pending_writes.remove(&socket_id) {
                        self.handle_socket_write(socket_id, write_op);
                        processed += 1;
                    }
                }
                // Re-register with remaining interest if any ops still pending
                let _ = self.update_socket_registration(socket_id);
            }
        }

        // Process completed file operations
        let results: Vec<CompletedResult> = {
            if let Ok(rx) = self.file_result_rx.lock() {
                let mut results = Vec::new();
                while let Ok(result) = rx.try_recv() {
                    results.push(result);
                }
                results
            } else {
                Vec::new()
            }
        };
        for result in results {
            self.completed_file_results
                .insert(result.handle, result.data);
            processed += 1;
        }

        // Process expired timers
        let now = Instant::now();
        let expired_timer_ids: Vec<usize> = self
            .timers
            .iter()
            .filter(|(_, (deadline, _))| *deadline <= now)
            .map(|(id, _)| *id)
            .collect();

        for timer_id in expired_timer_ids {
            if let Some((_deadline, future_atom)) = self.timers.remove(&timer_id) {
                self.completed_timers.push(future_atom);
                processed += 1;
            }
        }

        processed
    }

    /// Set a timer that fires after delay_ms milliseconds
    /// Returns the timer ID
    pub fn timer_set(&mut self, delay_ms: u64, future_atom: usize) -> usize {
        let timer_id = self.next_timer_id;
        self.next_timer_id += 1;
        let deadline = Instant::now() + Duration::from_millis(delay_ms);
        self.timers.insert(timer_id, (deadline, future_atom));
        timer_id
    }

    /// Cancel a timer by ID
    pub fn timer_cancel(&mut self, timer_id: usize) -> bool {
        self.timers.remove(&timer_id).is_some()
    }

    /// Get the number of pending completed timers
    pub fn completed_timers_len(&self) -> usize {
        self.completed_timers.len()
    }

    /// Pop the next completed timer's future_atom
    pub fn pop_completed_timer(&mut self) -> Option<usize> {
        if self.completed_timers.is_empty() {
            None
        } else {
            Some(self.completed_timers.remove(0))
        }
    }

    /// Get the number of completed file results waiting
    pub fn file_results_count(&self) -> usize {
        self.completed_file_results.len()
    }

    /// Check if a result is ready for the given handle
    pub fn file_result_ready(&self, handle: OperationHandle) -> bool {
        self.completed_file_results.contains_key(&handle)
    }

    /// Poll for a result by handle
    pub fn file_result_poll(&mut self, handle: OperationHandle) -> Option<FileResultData> {
        self.completed_file_results.remove(&handle)
    }

    /// Peek at a result's type code without removing it
    pub fn file_result_peek_type(&self, handle: OperationHandle) -> Option<usize> {
        self.completed_file_results
            .get(&handle)
            .map(EventLoop::file_result_type_code)
    }

    /// Generate a new unique operation handle
    pub fn next_operation_handle(&self) -> OperationHandle {
        self.next_handle.fetch_add(1, Ordering::SeqCst)
    }

    /// Handle a TCP event for Connect/Accept (Read/Write handled by per-socket methods)
    fn handle_tcp_event(&mut self, _token_id: usize, op: PendingOperation) {
        match op {
            PendingOperation::TcpConnect {
                future_atom,
                socket_id,
                op_id,
            } => {
                if let Some(stream) = self.tcp_streams.get(&socket_id) {
                    match stream.peer_addr() {
                        Ok(_) => {
                            self.completed_tcp_results.push(TcpResult::ConnectOk {
                                future_atom,
                                socket_id,
                                op_id,
                            });
                        }
                        Err(e) => {
                            self.completed_tcp_results.push(TcpResult::ConnectErr {
                                future_atom,
                                error: e.to_string(),
                                op_id,
                            });
                        }
                    }
                } else {
                    self.completed_tcp_results.push(TcpResult::ConnectErr {
                        future_atom,
                        error: "Socket not found".to_string(),
                        op_id,
                    });
                }
            }
            PendingOperation::TcpAccept {
                future_atom,
                listener_id,
                op_id,
            } => {
                if let Some(listener) = self.tcp_listeners.get(&listener_id) {
                    match listener.accept() {
                        Ok((stream, _addr)) => {
                            let socket_id = self.next_socket_id();
                            self.tcp_streams.insert(socket_id, stream);
                            self.completed_tcp_results.push(TcpResult::AcceptOk {
                                future_atom,
                                socket_id,
                                listener_id,
                                op_id,
                            });
                        }
                        Err(e) => {
                            self.completed_tcp_results.push(TcpResult::AcceptErr {
                                future_atom,
                                error: e.to_string(),
                                op_id,
                            });
                        }
                    }
                } else {
                    self.completed_tcp_results.push(TcpResult::AcceptErr {
                        future_atom,
                        error: "Listener not found".to_string(),
                        op_id,
                    });
                }
            }
            // Read and Write are handled by handle_socket_read/handle_socket_write
            _ => {}
        }
    }

    /// Handle a readable event for a socket
    fn handle_socket_read(&mut self, socket_id: usize, read_op: PendingReadOp) {
        use std::io::Read;
        let PendingReadOp {
            future_atom,
            buffer_size,
            op_id,
        } = read_op;

        if let Some(mut stream) = self.tcp_streams.remove(&socket_id) {
            let mut buffer = vec![0u8; buffer_size];
            let result = stream.read(&mut buffer);
            self.tcp_streams.insert(socket_id, stream);

            match result {
                Ok(n) => {
                    buffer.truncate(n);
                    self.completed_tcp_results.push(TcpResult::ReadOk {
                        future_atom,
                        data: buffer,
                        op_id,
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // Put the read op back - it will be re-registered by the caller
                    self.pending_reads.insert(
                        socket_id,
                        PendingReadOp {
                            future_atom,
                            buffer_size,
                            op_id,
                        },
                    );
                }
                Err(e) => {
                    self.completed_tcp_results.push(TcpResult::ReadErr {
                        future_atom,
                        error: e.to_string(),
                        op_id,
                    });
                }
            }
        } else {
            self.completed_tcp_results.push(TcpResult::ReadErr {
                future_atom,
                error: "Socket not found".to_string(),
                op_id,
            });
        }
    }

    /// Handle a writable event for a socket
    fn handle_socket_write(&mut self, socket_id: usize, write_op: PendingWriteOp) {
        use std::io::Write;
        let PendingWriteOp {
            future_atom,
            data,
            bytes_written,
            op_id,
        } = write_op;

        if let Some(mut stream) = self.tcp_streams.remove(&socket_id) {
            let result = stream.write(&data[bytes_written..]);
            self.tcp_streams.insert(socket_id, stream);

            match result {
                Ok(n) => {
                    let total_written = bytes_written + n;
                    if total_written >= data.len() {
                        self.completed_tcp_results.push(TcpResult::WriteOk {
                            future_atom,
                            bytes_written: total_written,
                            op_id,
                        });
                    } else {
                        // Partial write - put the write op back for more
                        self.pending_writes.insert(
                            socket_id,
                            PendingWriteOp {
                                future_atom,
                                data,
                                bytes_written: total_written,
                                op_id,
                            },
                        );
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // Put the write op back - it will be re-registered by the caller
                    self.pending_writes.insert(
                        socket_id,
                        PendingWriteOp {
                            future_atom,
                            data,
                            bytes_written,
                            op_id,
                        },
                    );
                }
                Err(e) => {
                    self.completed_tcp_results.push(TcpResult::WriteErr {
                        future_atom,
                        error: e.to_string(),
                        op_id,
                    });
                }
            }
        } else {
            self.completed_tcp_results.push(TcpResult::WriteErr {
                future_atom,
                error: "Socket not found".to_string(),
                op_id,
            });
        }
    }
}

/// Main loop for the dedicated event loop thread.
/// The lock is only held briefly for task draining and event processing.
/// The blocking poll() call happens WITHOUT the lock, preventing contention.
fn event_loop_thread_main(
    state: Arc<Mutex<EventLoopState>>,
    rx: mpsc::Receiver<TcpTask>,
    _waker: Arc<Waker>,
    shutdown: Arc<AtomicBool>,
    results_notify: Arc<(Mutex<bool>, Condvar)>,
) {
    loop {
        // Phase 1: Lock state briefly — drain tasks and take poll/events out
        let (mut poll, mut events, effective_timeout) = {
            let mut s = state.lock().unwrap();
            // Drain operation queue (non-blocking)
            while let Ok(task) = rx.try_recv() {
                s.handle_tcp_task(task);
            }
            let timeout = s.compute_poll_timeout(50);
            let poll = s.poll.take().expect("Poll must be present");
            let events = s.events.take().expect("Events must be present");
            (poll, events, timeout)
        }; // lock released — other threads can now access state freely

        // Phase 2: Poll I/O events WITHOUT holding the lock
        let _ = poll.poll(&mut events, Some(effective_timeout));

        // Phase 3: Lock state briefly — put poll/events back and process results
        let should_notify = {
            let mut s = state.lock().unwrap();
            // IMPORTANT: Restore poll BEFORE processing events, because
            // process_events_and_timers may call update_socket_registration
            // which needs poll_mut() to re-register sockets with new interest.
            s.poll = Some(poll);
            let initial_file_count = s.completed_file_results.len();
            s.process_events_and_timers(&events);
            s.events = Some(events);
            // Notify if TCP results, timers, or new file results arrived
            !s.completed_tcp_results.is_empty()
                || !s.completed_timers.is_empty()
                || s.completed_file_results.len() > initial_file_count
        }; // lock released

        if should_notify {
            if let Ok(mut ready) = results_notify.0.lock() {
                *ready = true;
                results_notify.1.notify_all();
            }
        }

        // Phase 4: Check shutdown
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
    }
}

// ============================================================================
// EventLoopHandle - Stored in Runtime for access from builtins
// ============================================================================

/// Storage for event loops in the runtime
/// Each event loop is stored with a unique ID
pub struct EventLoopRegistry {
    loops: HashMap<usize, EventLoop>,
    next_id: usize,
}

impl EventLoopRegistry {
    pub fn new() -> Self {
        Self {
            loops: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create a new event loop and return its ID
    /// Always spawns a dedicated I/O thread.
    pub fn create(&mut self, pool_size: usize) -> Result<usize, String> {
        let event_loop =
            EventLoop::new(pool_size).map_err(|e| format!("Failed to create event loop: {}", e))?;
        let id = self.next_id;
        self.next_id += 1;
        self.loops.insert(id, event_loop);
        Ok(id)
    }

    /// Get an immutable reference to an event loop by ID
    pub fn get(&self, id: usize) -> Option<&EventLoop> {
        self.loops.get(&id)
    }

    /// Register an event loop and return its ID
    pub fn register(&mut self, event_loop: EventLoop) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.loops.insert(id, event_loop);
        id
    }

    /// Unregister (remove) an event loop by ID
    /// Automatically shuts down the I/O thread.
    pub fn unregister(&mut self, id: usize) -> Option<EventLoop> {
        if let Some(el) = self.loops.get(&id) {
            el.shutdown_thread();
        }
        self.loops.remove(&id)
    }

    /// Remove an event loop (alias for unregister)
    pub fn remove(&mut self, id: usize) -> Option<EventLoop> {
        self.unregister(id)
    }

    /// Shutdown all event loops and clear the registry.
    /// This must be called during runtime reset to properly cleanup threads.
    pub fn shutdown_all(&mut self) {
        for (_id, el) in self.loops.drain() {
            el.shutdown_thread();
        }
        self.next_id = 1;
    }
}

impl Default for EventLoopRegistry {
    fn default() -> Self {
        Self::new()
    }
}

struct Namespace {
    name: String,
    ids: Vec<String>,
    bindings: HashMap<String, usize>,
    #[allow(unused)]
    aliases: HashMap<String, usize>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RawPtr<T> {
    pub ptr: *const T,
}

unsafe impl<T> Sync for RawPtr<T> {}
unsafe impl<T> Send for RawPtr<T> {}

impl<T> RawPtr<T> {
    pub fn new(ptr: *const T) -> Self {
        Self { ptr }
    }

    pub fn get(&self) -> *const T {
        self.ptr
    }
}

impl<T> From<RawPtr<T>> for usize {
    fn from(raw_ptr: RawPtr<T>) -> Self {
        raw_ptr.ptr as usize
    }
}

impl<T> From<RawPtr<T>> for u64 {
    fn from(raw_ptr: RawPtr<T>) -> Self {
        raw_ptr.ptr as u64
    }
}

impl<T, R> From<*const R> for RawPtr<T> {
    fn from(ptr: *const R) -> Self {
        Self {
            ptr: ptr as *const T,
        }
    }
}

impl<T, R> From<RawPtr<T>> for *const R {
    fn from(ptr: RawPtr<T>) -> Self {
        ptr.ptr as *const R
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub pointer: RawPtr<u8>,
    pub jump_table_offset: usize,
    pub is_foreign: bool,
    pub is_builtin: bool,
    pub needs_stack_pointer: bool,
    pub needs_frame_pointer: bool,
    pub is_defined: bool,
    pub number_of_locals: usize,
    pub size: usize,
    pub number_of_args: usize,
    pub is_variadic: bool,
    pub min_args: usize,
    /// Docstring for the function (from /// comments in source)
    pub docstring: Option<String>,
    /// Argument names for reflection
    pub arg_names: Vec<String>,
    /// Source file where the function was defined
    pub source_file: Option<String>,
    /// Source line number where the function was defined
    pub source_line: Option<usize>,
    /// Source column number where the function was defined
    pub source_column: Option<usize>,
}

// ============================================================================
// Documentation Export Types
// ============================================================================

/// Documentation for the entire codebase, exported as JSON
#[derive(nanoserde::SerJson)]
pub struct Documentation {
    pub namespaces: Vec<NamespaceDoc>,
    pub structs: Vec<StructDoc>,
    pub enums: Vec<EnumDoc>,
}

/// Documentation for a namespace
#[derive(nanoserde::SerJson)]
pub struct NamespaceDoc {
    pub name: String,
    pub functions: Vec<FunctionDoc>,
}

/// Documentation for a function
#[derive(nanoserde::SerJson)]
pub struct FunctionDoc {
    pub name: String,
    pub full_name: String,
    pub docstring: Option<String>,
    pub arg_names: Vec<String>,
    pub arity: usize,
    pub is_variadic: bool,
    pub min_args: usize,
    pub is_builtin: bool,
    pub source_file: Option<String>,
    pub source_line: Option<usize>,
}

/// Documentation for a struct
#[derive(nanoserde::SerJson)]
pub struct StructDoc {
    pub name: String,
    pub docstring: Option<String>,
    pub fields: Vec<FieldDoc>,
    pub source_file: Option<String>,
    pub source_line: Option<usize>,
}

/// Documentation for a struct field
#[derive(nanoserde::SerJson)]
pub struct FieldDoc {
    pub name: String,
    pub docstring: Option<String>,
    pub mutable: bool,
}

/// Documentation for an enum
#[derive(nanoserde::SerJson)]
pub struct EnumDoc {
    pub name: String,
    pub docstring: Option<String>,
    pub variants: Vec<VariantDoc>,
    pub source_file: Option<String>,
    pub source_line: Option<usize>,
}

/// Documentation for an enum variant
#[derive(nanoserde::SerJson)]
pub struct VariantDoc {
    pub name: String,
    pub fields: Vec<String>,
}

pub struct MMapMutWithOffset {
    mmap: MmapMut,
    offset: usize,
}

impl MMapMutWithOffset {
    fn new() -> Self {
        Self {
            mmap: MmapOptions::new(MmapOptions::page_size() * 10)
                .expect("Failed to create mmap for CStringStorage - out of memory")
                .map_mut()
                .expect("Failed to map CStringStorage memory - this is a fatal error"),
            offset: 0,
        }
    }

    pub fn write_c_string(&mut self, string: String) -> *mut i8 {
        let string = string.clone();
        let start = self.offset;
        let c_string =
            CString::new(string).expect("Failed to create CString - string contains null byte");
        let bytes = c_string.as_bytes_with_nul();
        for byte in bytes {
            self.mmap[self.offset] = *byte;
            self.offset += 1;
        }
        unsafe { self.mmap.as_ptr().add(start) as *mut i8 }
    }

    pub fn write_u16(&mut self, value: u16) -> &u16 {
        let start = self.offset;
        let bytes = value.to_le_bytes();
        for byte in bytes {
            self.mmap[self.offset] = byte;
            self.offset += 1;
        }
        unsafe { &*(self.mmap.as_ptr().add(start) as *const u16) }
    }

    pub fn write_u32(&mut self, value: u32) -> *const u32 {
        // Align to 4 bytes
        self.offset = (self.offset + 3) & !3;
        let start = self.offset;
        let bytes = value.to_le_bytes();
        for byte in bytes {
            self.mmap[self.offset] = byte;
            self.offset += 1;
        }
        unsafe { self.mmap.as_ptr().add(start) as *const u32 }
    }

    pub fn write_u64(&mut self, value: u64) -> *const u64 {
        // Align to 8 bytes
        self.offset = (self.offset + 7) & !7;
        let start = self.offset;
        let bytes = value.to_le_bytes();
        for byte in bytes {
            self.mmap[self.offset] = byte;
            self.offset += 1;
        }
        unsafe { self.mmap.as_ptr().add(start) as *const u64 }
    }

    pub fn write_i32(&mut self, value: i32) -> *const i32 {
        // Align to 4 bytes
        self.offset = (self.offset + 3) & !3;
        let start = self.offset;
        let bytes = value.to_le_bytes();
        for byte in bytes {
            self.mmap[self.offset] = byte;
            self.offset += 1;
        }

        (unsafe { self.mmap.as_ptr().add(start) as *const i32 }) as _
    }

    pub fn write_u8(&mut self, argument: u8) -> *const u8 {
        let start = self.offset;
        self.mmap[start] = argument;
        // We need to make sure we keep correct alignment
        self.offset += 2;
        unsafe { self.mmap.as_ptr().add(start) }
    }

    pub fn write_pointer(&mut self, value: usize) -> *const *mut c_void {
        // Align to 8 bytes for pointer types
        self.offset = (self.offset + 7) & !7;
        let start = self.offset;
        let bytes = value.to_le_bytes();
        for byte in bytes {
            self.mmap[self.offset] = byte;
            self.offset += 1;
        }
        unsafe { self.mmap.as_ptr().add(start) as *const *mut c_void }
    }

    pub fn write_buffer(&mut self, ptr: usize, size: usize) -> *const u8 {
        // we are going to get the buffer located at ptr with a size of size
        // and copy it into our mmap
        let start = self.offset;
        let buffer = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
        for byte in buffer {
            self.mmap[self.offset] = *byte;
            self.offset += 1;
        }
        unsafe { self.mmap.as_ptr().add(start) }
    }

    pub fn clear(&mut self) {
        self.offset = 0;
    }
}

thread_local! {
    pub static NATIVE_ARGUMENTS : RefCell<MMapMutWithOffset> = RefCell::new(MMapMutWithOffset::new());

    /// Cached pointer to the current thread's ThreadGlobal.
    /// Avoids locking the thread_globals mutex on every add_handle_root/remove_handle_root.
    /// Safety: The ThreadGlobal is Box'd in the HashMap, so its address is stable.
    /// GC updates head_block via AtomicUsize, which is visible when the thread resumes.
    static CACHED_THREAD_GLOBAL: Cell<*mut ThreadGlobal> = const { Cell::new(std::ptr::null_mut()) };
}

/// Get the cached ThreadGlobal pointer for the current thread.
/// Returns null if the thread hasn't been initialized yet.
pub(crate) fn cached_thread_global_ptr() -> *mut ThreadGlobal {
    CACHED_THREAD_GLOBAL.with(|c| c.get())
}

// ============================================================================
// Effect Handler Stack (Thread-Local)
// ============================================================================

/// An entry in the handler stack.
/// Each entry maps a protocol key (e.g., "Handler<Async>") to a handler instance.
#[derive(Clone, Debug)]
pub struct HandlerEntry {
    /// Protocol key including type args, e.g., "Handler<Async>" or "Handler<Log>"
    pub protocol_key: String,
    /// Slot in the GlobalObjectBlock that stores the handler pointer.
    /// GC updates this slot when the handler object moves.
    pub root_slot: usize,
}

/// Thread-local handler stack for effect handlers.
/// Handlers are scoped - inner handlers shadow outer ones for the same protocol key.
#[derive(Default)]
pub struct HandlerStack {
    entries: Vec<HandlerEntry>,
}

impl HandlerStack {
    pub fn new() -> Self {
        HandlerStack {
            entries: Vec::new(),
        }
    }

    /// Push a handler onto the stack
    pub fn push(&mut self, protocol_key: String, root_slot: usize) {
        self.entries.push(HandlerEntry {
            protocol_key,
            root_slot,
        });
    }

    /// Pop the most recent handler with the given protocol key
    pub fn pop(&mut self, protocol_key: &str) -> Option<HandlerEntry> {
        // Find the last entry with matching protocol key
        if let Some(idx) = self
            .entries
            .iter()
            .rposition(|e| e.protocol_key == protocol_key)
        {
            Some(self.entries.remove(idx))
        } else {
            None
        }
    }

    /// Find the most recent handler for the given protocol key
    pub fn find(&self, protocol_key: &str) -> Option<&HandlerEntry> {
        self.entries
            .iter()
            .rfind(|e| e.protocol_key == protocol_key)
    }

    /// Clear all handlers (used on reset)
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

thread_local! {
    /// Per-thread handler stack for effect handlers
    pub static HANDLER_STACK: RefCell<HandlerStack> = RefCell::new(HandlerStack::new());
}

/// Push a handler onto the current thread's handler stack
pub fn push_handler(protocol_key: String, handler_instance: usize) {
    // Register the handler as a handle root so GC can update it when moved.
    // This also keeps the handler alive even though the stack entry lives in Rust memory.
    let runtime = crate::get_runtime().get_mut();
    let root_slot = runtime.register_temporary_root(handler_instance);
    assert!(root_slot != 0, "Failed to register handler as GC root");

    HANDLER_STACK.with(|stack| stack.borrow_mut().push(protocol_key, root_slot));
}

/// Pop a handler from the current thread's handler stack
pub fn pop_handler(protocol_key: &str) -> Option<usize> {
    let runtime = crate::get_runtime().get_mut();
    HANDLER_STACK.with(|stack| {
        stack.borrow_mut().pop(protocol_key).map(|entry| {
            let value = runtime.get_handle_root(entry.root_slot);
            runtime.remove_handle_root(entry.root_slot);
            value
        })
    })
}

/// Find a handler in the current thread's handler stack
pub fn find_handler(protocol_key: &str) -> Option<usize> {
    HANDLER_STACK.with(|stack| {
        stack
            .borrow()
            .find(protocol_key)
            .map(|e| crate::get_runtime().get().get_handle_root(e.root_slot))
    })
}

/// Clear the handler stack (called on reset)
pub fn clear_handler_stack() {
    let runtime = crate::get_runtime().get_mut();
    HANDLER_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        for entry in stack.entries.drain(..) {
            runtime.remove_handle_root(entry.root_slot);
        }
    });
}

// ============================================================================
// Dynamic Variable Stack (Thread-Local Rebinding)
// ============================================================================

/// An entry in the dynamic variable stack.
/// Each entry maps a (namespace_id, slot) pair to a thread-local value.
#[derive(Clone, Debug)]
pub struct DynamicVarBinding {
    /// Namespace ID where the dynamic var is defined
    pub namespace_id: usize,
    /// Slot within the namespace
    pub slot: usize,
    /// Slot in the GlobalObjectBlock that stores the bound value.
    /// GC updates this slot when the value moves.
    pub root_slot: usize,
}

/// Thread-local dynamic variable stack.
/// Dynamic vars support thread-local rebinding with stack-like scoping.
#[derive(Default)]
pub struct DynamicVarStack {
    bindings: Vec<DynamicVarBinding>,
}

impl DynamicVarStack {
    pub fn new() -> Self {
        DynamicVarStack {
            bindings: Vec::new(),
        }
    }

    /// Push a binding onto the stack
    pub fn push(&mut self, namespace_id: usize, slot: usize, root_slot: usize) {
        self.bindings.push(DynamicVarBinding {
            namespace_id,
            slot,
            root_slot,
        });
    }

    /// Pop the most recent binding for the given (namespace_id, slot)
    pub fn pop(&mut self, namespace_id: usize, slot: usize) -> Option<DynamicVarBinding> {
        // Find the last binding with matching namespace_id and slot
        if let Some(idx) = self
            .bindings
            .iter()
            .rposition(|b| b.namespace_id == namespace_id && b.slot == slot)
        {
            Some(self.bindings.remove(idx))
        } else {
            None
        }
    }

    /// Find the most recent binding for the given (namespace_id, slot)
    pub fn find(&self, namespace_id: usize, slot: usize) -> Option<&DynamicVarBinding> {
        self.bindings
            .iter()
            .rfind(|b| b.namespace_id == namespace_id && b.slot == slot)
    }

    /// Clear all bindings (used on reset)
    pub fn clear(&mut self) {
        self.bindings.clear();
    }
}

thread_local! {
    /// Per-thread dynamic variable stack
    pub static DYNAMIC_VAR_STACK: RefCell<DynamicVarStack> =
        RefCell::new(DynamicVarStack::new());
}

/// Push a dynamic variable binding onto the current thread's stack
pub fn push_dynamic_binding(namespace_id: usize, slot: usize, value: usize) {
    // Register the value as a handle root so GC can update it when moved.
    // This also keeps the value alive even though the stack entry lives in Rust memory.
    let runtime = crate::get_runtime().get_mut();
    let root_slot = runtime.register_temporary_root(value);
    assert!(
        root_slot != 0,
        "Failed to register dynamic var binding as GC root"
    );

    DYNAMIC_VAR_STACK.with(|stack| stack.borrow_mut().push(namespace_id, slot, root_slot));
}

/// Pop a dynamic variable binding from the current thread's stack
pub fn pop_dynamic_binding(namespace_id: usize, slot: usize) -> Option<usize> {
    let runtime = crate::get_runtime().get_mut();
    DYNAMIC_VAR_STACK.with(|stack| {
        stack.borrow_mut().pop(namespace_id, slot).map(|binding| {
            let value = runtime.get_handle_root(binding.root_slot);
            runtime.remove_handle_root(binding.root_slot);
            value
        })
    })
}

/// Find a dynamic variable binding in the current thread's stack
pub fn find_dynamic_binding(namespace_id: usize, slot: usize) -> Option<usize> {
    DYNAMIC_VAR_STACK.with(|stack| {
        stack
            .borrow()
            .find(namespace_id, slot)
            .map(|b| crate::get_runtime().get().get_handle_root(b.root_slot))
    })
}

/// Clear the dynamic variable stack (called on reset)
pub fn clear_dynamic_var_stack() {
    let runtime = crate::get_runtime().get_mut();
    DYNAMIC_VAR_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        for binding in stack.bindings.drain(..) {
            runtime.remove_handle_root(binding.root_slot);
        }
    });
}

pub struct Memory {
    heap: Alloc,
    stacks: Mutex<Vec<(ThreadId, MmapMut)>>,
    pub join_handles: Vec<JoinHandle<u64>>,
    pub threads: Vec<Thread>,
    pub stack_map: StackMap,
    /// Per-thread GlobalObject management (now owned by Memory, not GC)
    /// Protected by Mutex for thread-safe access from multiple threads.
    /// Box<ThreadGlobal> provides stable addresses so thread_locals can cache raw pointers.
    pub thread_globals: Mutex<HashMap<ThreadId, Box<ThreadGlobal>>>,
    #[allow(unused)]
    command_line_arguments: CommandLineArguments,
    /// Temporary storage for namespace binding values during GC.
    /// Provides stable pointers for copying GCs (like generational GC).
    /// Values are copied here before GC, GC updates them in place, then they're
    /// copied back to namespace HashMaps after GC.
    namespace_root_storage: Vec<usize>,
}

impl Memory {
    fn reset(&mut self) {
        // Full memory barrier before reset to ensure all pending stores are visible
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        let options = self.heap.get_allocation_options();
        self.heap = Alloc::new(options);
        let mut stacks = self
            .stacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *stacks = vec![(
            std::thread::current().id(),
            create_stack_with_protected_page_after(STACK_SIZE),
        )];
        self.join_handles = vec![];
        self.threads = vec![std::thread::current()];
        self.stack_map = StackMap::new();
        self.thread_globals.lock().unwrap().clear();
        // Invalidate cached thread_local pointer since ThreadGlobals were cleared
        CACHED_THREAD_GLOBAL.with(|c| c.set(std::ptr::null_mut()));

        // Clear the effect handler stack
        clear_handler_stack();

        // Clear the dynamic variable stack
        clear_dynamic_var_stack();

        // Full memory barrier after reset to ensure all stores are visible
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }

    fn active_threads(&mut self) -> usize {
        let mut completed_threads = vec![];
        for (index, thread) in self.join_handles.iter().enumerate() {
            if thread.is_finished() {
                completed_threads.push(index);
            }
        }
        for index in completed_threads.iter().rev() {
            if let Some(thread) = self.join_handles.get(*index) {
                let thread_id = thread.thread().id();
                let mut stacks = self
                    .stacks
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                stacks.retain(|(id, _)| *id != thread_id);
                self.threads.retain(|t| t.id() != thread_id);
                self.join_handles.remove(*index);
                self.heap.remove_thread(thread_id);
            } else {
                println!("Inconsistent join_handles state {:?}", self.join_handles);
            }
        }

        self.join_handles.len()
    }

    fn allocate_string(&mut self, bytes: &[u8], pointer: usize) -> Result<Tagged, Box<dyn Error>> {
        let mut heap_object = HeapObject::from_tagged(pointer);
        // Layout: [header][cached_hash: 8 bytes][string bytes]
        let text_words = bytes.len().div_ceil(8);
        let total_words = 1 + text_words; // 1 for hash + text
        let is_large = total_words > Header::MAX_INLINE_SIZE;
        heap_object.writer_header_direct(Header {
            type_id: 2,
            type_data: bytes.len() as u32,
            size: if is_large { 0xFFFF } else { total_words as u16 },
            opaque: true,
            marked: false,
            large: is_large,
            type_flags: if bytes.is_ascii() { 1 } else { 0 },
        });
        // For large objects, write the actual size in the next word
        if is_large {
            let size_ptr = (heap_object.untagged() + 8) as *mut usize;
            unsafe { *size_ptr = total_words };
        }
        // Write hash=0 (not yet computed) then string bytes
        let mut data = Vec::with_capacity(total_words * 8);
        data.extend_from_slice(&0u64.to_le_bytes()); // cached hash = 0
        data.extend_from_slice(bytes);
        // Pad to word boundary
        while data.len() < total_words * 8 {
            data.push(0);
        }
        heap_object.write_fields(&data);
        Ok(BuiltInTypes::HeapObject.tagged(pointer))
    }

    fn allocate_keyword(&mut self, bytes: &[u8], pointer: usize) -> Result<Tagged, Box<dyn Error>> {
        let mut heap_object = HeapObject::from_tagged(pointer);

        // Compute stable hash based on keyword text
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        let hash = hasher.finish();

        // Layout: [hash (8 bytes)][keyword text bytes]
        // Size includes hash word + text words
        let text_words = bytes.len().div_ceil(8); // Round up
        let total_words = 1 + text_words; // 1 for hash + text

        let is_large = total_words > Header::MAX_INLINE_SIZE;
        heap_object.writer_header_direct(Header {
            type_id: 3,
            type_data: bytes.len() as u32, // Store text length
            size: if is_large { 0xFFFF } else { total_words as u16 },
            opaque: true,
            marked: false,
            large: is_large,
            type_flags: 0,
        });
        // For large objects, write the actual size in the next word
        if is_large {
            let size_ptr = (heap_object.untagged() + 8) as *mut usize;
            unsafe { *size_ptr = total_words };
        }

        // Write hash as first 8 bytes, then the text
        let mut data = Vec::with_capacity(total_words * 8);
        data.extend_from_slice(&hash.to_le_bytes());
        data.extend_from_slice(bytes);
        // Pad to word boundary
        while data.len() < total_words * 8 {
            data.push(0);
        }

        heap_object.write_fields(&data);
        Ok(BuiltInTypes::HeapObject.tagged(pointer))
    }

    pub fn write_c_string(&mut self, string: String) -> *mut i8 {
        let mut result: *mut i8 = std::ptr::null_mut();
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_c_string(string));
        result
    }

    pub fn write_pointer(&mut self, value: usize) -> &*mut c_void {
        let mut result: *const *mut c_void = std::ptr::null_mut();
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_pointer(value));
        unsafe { &*result }
    }

    pub fn write_u32(&mut self, value: u32) -> &u32 {
        let mut result: *const u32 = &0;
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_u32(value));
        unsafe { &*result }
    }

    pub fn write_u64(&mut self, value: u64) -> &u64 {
        let mut result: *const u64 = &0;
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_u64(value));
        unsafe { &*result }
    }

    pub fn write_i32(&mut self, value: i32) -> &i32 {
        let mut result: *const i32 = &0;
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_i32(value));
        unsafe { &*result }
    }

    pub fn write_u8(&mut self, value: u8) -> &u8 {
        let mut result: *const u8 = &0;
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_u8(value));
        unsafe { &*result }
    }

    pub fn write_u16(&mut self, value: u16) -> &u16 {
        let mut result: *const u16 = &0;
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_u16(value));
        unsafe { &*result }
    }

    pub fn write_buffer(&mut self, ptr: usize, size: usize) -> &u8 {
        let mut result: *const u8 = &0;
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_buffer(ptr, size));
        unsafe { &*result }
    }

    pub fn clear_native_arguments(&self) {
        NATIVE_ARGUMENTS.with(|memory| memory.borrow_mut().clear());
    }

    /// Convenience method for Runtime to trigger GC using Memory's own stack_map and thread_globals.
    /// Collects shadow stack roots from all threads and passes them to the GC.
    /// namespace_roots: Additional GC roots from namespace bindings (passed from Runtime)
    /// Returns updated namespace root values after GC (for copying collectors)
    pub fn run_gc(
        &mut self,
        stack_pointers: &[(usize, usize, usize)],
        namespace_roots: &[usize],
    ) -> Vec<usize> {
        // Collect shadow stack entries from all threads as extra GC roots.
        // Safety: world is stopped during GC, so no thread modifies handle_stack.
        let mut extra_roots: Vec<(*mut usize, usize)> = {
            let thread_globals = self.thread_globals.lock().unwrap();
            let mut roots = Vec::new();
            for tg in thread_globals.values() {
                for i in 0..tg.handle_stack_top {
                    let value = tg.handle_stack[i];
                    if BuiltInTypes::is_heap_pointer(value) {
                        let slot_addr = &tg.handle_stack[i] as *const usize as *mut usize;
                        roots.push((slot_addr, value));
                    }
                }
            }
            roots
        };

        // Store namespace roots in a Vec to provide stable pointers during GC
        // This is required for copying collectors (like generational GC) that need to
        // update pointers when objects move.
        self.namespace_root_storage.clear();
        self.namespace_root_storage
            .extend_from_slice(namespace_roots);

        // Add pointers to namespace roots in our storage
        for i in 0..self.namespace_root_storage.len() {
            let value = self.namespace_root_storage[i];
            if BuiltInTypes::is_heap_pointer(value) {
                let slot_addr = &mut self.namespace_root_storage[i] as *mut usize;
                extra_roots.push((slot_addr, value));
            }
        }

        self.heap.gc(&self.stack_map, stack_pointers, &extra_roots);

        // Sync ThreadGlobal.head_block from stack slots (updated by GC)
        // Hold lock for entire operation to prevent concurrent modifications
        let mut thread_globals = self.thread_globals.lock().unwrap();
        for (stack_base, _, _) in stack_pointers.iter() {
            let global_block_slot = (stack_base - 8) as *const usize;
            let new_head_block = unsafe { *global_block_slot };

            for tg in thread_globals.values_mut() {
                if tg.stack_base == *stack_base {
                    tg.head_block.store(new_head_block, Ordering::SeqCst);
                    break;
                }
            }
        }

        // Return updated values (GC may have moved objects for copying collectors)
        self.namespace_root_storage.clone()
    }
}

impl Allocator for Memory {
    fn new(_options: AllocatorOptions) -> Self {
        unimplemented!("Not going to use this");
    }

    fn try_allocate(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        self.heap.try_allocate(words, kind)
    }

    fn gc(
        &mut self,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize, usize)],
        extra_roots: &[(*mut usize, usize)],
    ) {
        self.heap.gc(stack_map, stack_pointers, extra_roots);
    }

    fn grow(&mut self) {
        self.heap.grow()
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.heap.get_allocation_options()
    }
}

pub enum EnumVariant {
    StructVariant { name: String, fields: Vec<String> },
    StaticVariant { name: String },
}

pub struct Enum {
    pub name: String,
    pub variants: Vec<EnumVariant>,
    /// Docstring for the enum (from /// comments in source)
    pub docstring: Option<String>,
}

pub struct EnumManager {
    name_to_id: HashMap<String, usize>,
    enums: Vec<Enum>,
}

impl Default for EnumManager {
    fn default() -> Self {
        Self::new()
    }
}

impl EnumManager {
    pub fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            enums: Vec::new(),
        }
    }

    pub fn insert(&mut self, e: Enum) {
        let id = self.enums.len();
        self.name_to_id.insert(e.name.clone(), id);
        self.enums.push(e);
    }

    pub fn get(&self, name: &str) -> Option<&Enum> {
        let id = self.name_to_id.get(name)?;
        self.enums.get(*id)
    }

    /// Iterate over all enums
    pub fn iter(&self) -> impl Iterator<Item = &Enum> {
        self.enums.iter()
    }
}

struct NamespaceManager {
    namespaces: Vec<Mutex<Namespace>>,
    namespace_names: HashMap<String, usize>,
    id_to_name: HashMap<usize, String>,
    current_namespace: usize,
}

impl NamespaceManager {
    fn new() -> Self {
        let mut s = Self {
            namespaces: vec![Mutex::new(Namespace {
                name: "global".to_string(),
                ids: vec![],
                bindings: HashMap::new(),
                aliases: HashMap::new(),
            })],
            namespace_names: HashMap::new(),
            id_to_name: HashMap::new(),
            current_namespace: 0,
        };
        s.add_namespace("beagle.primitive");
        s.add_namespace("beagle.builtin");
        s.add_namespace("beagle.__internal_test__");
        s.add_namespace("beagle.debug");
        s.add_namespace("beagle.reflect");
        s
    }

    fn add_namespace(&mut self, name: &str) -> usize {
        let position = self.namespaces.iter().position(|n| {
            n.lock()
                .expect("Failed to lock namespace in add_namespace - this is a fatal error")
                .name
                == name
        });
        if let Some(position) = position {
            return position;
        }

        self.namespaces.push(Mutex::new(Namespace {
            name: name.to_string(),
            ids: vec![],
            bindings: HashMap::new(),
            aliases: HashMap::new(),
        }));
        let id = self.namespaces.len() - 1;
        self.namespace_names.insert(name.to_string(), id);
        self.id_to_name.insert(id, name.to_string());
        self.namespaces.len() - 1
    }

    #[allow(unused)]
    fn get_namespace(&self, name: &str) -> Option<&Mutex<Namespace>> {
        let position = self.namespace_names.get(name)?;
        self.namespaces.get(*position)
    }

    fn get_namespace_id(&self, name: &str) -> Option<usize> {
        self.namespace_names.get(name).cloned()
    }

    fn get_current_namespace(&self) -> &Mutex<Namespace> {
        self.namespaces
            .get(self.current_namespace)
            .expect("Current namespace not found - this is a fatal error")
    }

    fn get_namespace_by_id(&self, id: usize) -> Option<&Mutex<Namespace>> {
        self.namespaces.get(id)
    }

    fn set_current_namespace(&mut self, id: usize) {
        self.current_namespace = id;
    }

    fn add_binding(&self, name: &str, pointer: usize) -> usize {
        let mut namespace = self
            .get_current_namespace()
            .lock()
            .expect("Failed to lock namespace in add_binding - this is a fatal error");
        if namespace.bindings.contains_key(name) {
            // Only overwrite if the new value is non-null OR the existing value is null
            // This prevents reserve_namespace_slot from overwriting real values with null
            let existing = *namespace.bindings.get(name).unwrap();
            let is_null = pointer == BuiltInTypes::null_value() as usize;
            let existing_is_null = existing == BuiltInTypes::null_value() as usize;
            if !is_null || existing_is_null {
                namespace.bindings.insert(name.to_string(), pointer);
            }
            return namespace.ids.iter().position(|n| n == name).expect(
                "Binding exists in map but not in ids vec - this is a fatal internal error",
            );
        }
        namespace.bindings.insert(name.to_string(), pointer);
        namespace.ids.push(name.to_string());
        namespace.ids.len() - 1
    }

    #[allow(unused)]
    fn find_binding_id(&self, name: &str) -> Option<usize> {
        let namespace = self
            .get_current_namespace()
            .lock()
            .expect("Failed to lock namespace in find_binding_id - this is a fatal error");
        namespace.ids.iter().position(|n| n == name)
    }

    #[allow(unused)]
    fn get_namespace_name(&self, id: usize) -> Option<String> {
        self.id_to_name.get(&id).cloned()
    }
}

#[allow(unused)]
#[derive(Debug)]
pub struct StackValue {
    function: Option<String>,
    details: StackMapDetails,
    stack: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ProtocolMethodInfo {
    pub method_name: String,
    pub _type: String,
    pub fn_pointer: usize,
}

/// Dispatch table for O(1) protocol method lookup
/// Maps type_id -> function pointer for fast protocol dispatch
#[derive(Debug, Clone)]
pub struct DispatchTable {
    /// Dense array for struct IDs (positive IDs, index = struct_id)
    /// Value = function pointer (0 if not implemented for this type)
    pub struct_dispatch: Vec<usize>,
    /// Sparse array for primitive types (negative IDs mapped to indices)
    /// Index mapping: -1 -> 0, -2 -> 1, etc.
    pub primitive_dispatch: Vec<usize>,
    /// Default method pointer (if protocol has default implementation)
    pub default_method: Option<usize>,
}

impl DispatchTable {
    pub fn new(default_method: Option<usize>) -> Self {
        Self {
            struct_dispatch: Vec::new(),
            primitive_dispatch: Vec::new(),
            default_method,
        }
    }

    /// Register an implementation for a type
    /// type_id: positive for structs, negative for primitives
    pub fn register(&mut self, type_id: isize, fn_pointer: usize) {
        if type_id < 0 {
            // Primitive type - map negative ID to index
            let index = (-type_id - 1) as usize;
            if index >= self.primitive_dispatch.len() {
                self.primitive_dispatch.resize(index + 1, 0);
            }
            self.primitive_dispatch[index] = fn_pointer;
        } else {
            // Struct type - use ID as index
            let index = type_id as usize;
            if index >= self.struct_dispatch.len() {
                self.struct_dispatch.resize(index + 1, 0);
            }
            self.struct_dispatch[index] = fn_pointer;
        }
    }

    /// Lookup function pointer for a struct type ID
    pub fn lookup_struct(&self, struct_id: usize) -> usize {
        if struct_id < self.struct_dispatch.len() {
            let ptr = self.struct_dispatch[struct_id];
            if ptr != 0 {
                return ptr;
            }
        }
        self.default_method.unwrap_or(0)
    }

    /// Lookup function pointer for a primitive type
    /// primitive_index: 0 for Int (-1), 1 for Float (-2), etc.
    pub fn lookup_primitive(&self, primitive_index: usize) -> usize {
        if primitive_index < self.primitive_dispatch.len() {
            let ptr = self.primitive_dispatch[primitive_index];
            if ptr != 0 {
                return ptr;
            }
        }
        self.default_method.unwrap_or(0)
    }

    /// Set the default method pointer for this dispatch table
    pub fn set_default_method(&mut self, fn_ptr: usize) {
        self.default_method = Some(fn_ptr);
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct ExceptionHandler {
    pub handler_address: usize,
    pub stack_pointer: usize,
    pub frame_pointer: usize,
    pub link_register: usize,
    pub result_local: isize,
}

/// Prompt handler for delimited continuations.
/// Marks a point on the stack that serves as a delimiter for continuation capture.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct PromptHandler {
    /// Address to jump to when the shift body completes normally
    pub handler_address: usize,
    /// Stack pointer at the time of prompt
    pub stack_pointer: usize,
    /// Frame pointer at the time of prompt
    pub frame_pointer: usize,
    /// Link register (return address) at the time of prompt
    pub link_register: usize,
    /// Local variable offset to store the result value
    pub result_local: isize,
    /// Unique ID for this prompt (to distinguish sequential handlers at same depth)
    pub prompt_id: usize,
}

/// Invocation return point for multi-shot continuations.
/// When a continuation k is invoked via k(value), we save where to return after the
/// continuation body completes. This enables multiple invocations like [k(1), k(2), k(3)].
#[repr(C)]
#[derive(Debug, Clone)]
pub struct InvocationReturnPoint {
    /// Stack pointer at the point where k(value) was called
    pub stack_pointer: usize,
    /// Frame pointer at the point where k(value) was called
    pub frame_pointer: usize,
    /// Address to return to after continuation body completes
    pub return_address: usize,
    /// Callee-saved registers (x19-x28 on ARM64) that must be preserved
    /// These are saved when k() is called and restored when returning
    pub callee_saved_regs: [usize; 10],
    /// Saved stack frame contents from the shift body.
    /// For multi-shot continuations, we need to preserve the shift body's locals
    /// (including the continuation k) when the continuation body runs, because
    /// the continuation body may write to stack locations that overlap with
    /// the shift body's frame. We save the frame from FP to SP.
    pub saved_stack_frame: Vec<u8>,
    /// Unique ID of the prompt handler that this return point belongs to.
    /// Used to match return points with their corresponding handle blocks.
    /// This distinguishes sequential handlers at the same stack depth.
    pub prompt_id: usize,
    /// Where the continuation's stack segment was relocated to.
    /// Used to copy modifications back to the original location.
    pub relocated_sp: usize,
    /// Original SP where the stack segment was captured from.
    /// The modifications need to be copied back here.
    pub original_sp: usize,
    /// Length of the stack segment in bytes.
    pub segment_len: usize,
    /// Whether this is a "root" invocation from original code (true) or a nested
    /// invocation from within a continuation body (false).
    /// Only root invocations should copy modifications back, because nested invocations
    /// have an "original_sp" that points to a relocated stack, not the true original.
    pub is_root_invocation: bool,
}

/// Captured continuation: a snapshot of the stack between a shift point and its enclosing reset.
/// This is stored as a heap object with type_id = TYPE_ID_CONTINUATION.
pub struct ContinuationObject {
    heap_obj: HeapObject,
}

impl ContinuationObject {
    const FIELD_SEGMENT: usize = 0;
    const FIELD_ORIGINAL_SP: usize = 1;
    const FIELD_ORIGINAL_FP: usize = 2;
    const FIELD_RESUME_ADDRESS: usize = 3;
    const FIELD_RESULT_LOCAL: usize = 4;
    const FIELD_HANDLER_ADDRESS: usize = 5;
    const FIELD_PROMPT_SP: usize = 6;
    const FIELD_PROMPT_FP: usize = 7;
    const FIELD_PROMPT_LR: usize = 8;
    const FIELD_PROMPT_RESULT_LOCAL: usize = 9;
    const FIELD_PROMPT_ID: usize = 10;

    pub fn from_tagged(tagged: usize) -> Option<Self> {
        let heap_obj = HeapObject::try_from_tagged(tagged)?;
        Self::from_heap_object(heap_obj)
    }

    pub fn from_heap_object(heap_obj: HeapObject) -> Option<Self> {
        if heap_obj.get_type_id() == TYPE_ID_CONTINUATION as usize {
            Some(Self { heap_obj })
        } else {
            None
        }
    }

    pub fn tagged_ptr(&self) -> usize {
        self.heap_obj.tagged_pointer()
    }

    pub fn segment_ptr(&self) -> usize {
        self.heap_obj.get_field(Self::FIELD_SEGMENT)
    }

    pub fn segment_len(&self) -> usize {
        let segment_obj = HeapObject::from_tagged(self.segment_ptr());
        segment_obj.get_opaque_bytes_len()
    }

    pub fn with_segment_bytes<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[u8]) -> R,
    {
        let segment_obj = HeapObject::from_tagged(self.segment_ptr());
        let bytes = segment_obj.get_opaque_bytes();
        f(bytes)
    }

    pub fn with_segment_bytes_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        let mut segment_obj = HeapObject::from_tagged(self.segment_ptr());
        let bytes = segment_obj.get_opaque_bytes_mut();
        f(bytes)
    }

    pub fn original_sp(&self) -> usize {
        BuiltInTypes::untag(self.heap_obj.get_field(Self::FIELD_ORIGINAL_SP))
    }

    pub fn original_fp(&self) -> usize {
        BuiltInTypes::untag(self.heap_obj.get_field(Self::FIELD_ORIGINAL_FP))
    }

    pub fn resume_address(&self) -> usize {
        BuiltInTypes::untag(self.heap_obj.get_field(Self::FIELD_RESUME_ADDRESS))
    }

    pub fn result_local(&self) -> isize {
        BuiltInTypes::untag_isize(self.heap_obj.get_field(Self::FIELD_RESULT_LOCAL) as isize)
    }

    pub fn handler_address(&self) -> usize {
        BuiltInTypes::untag(self.heap_obj.get_field(Self::FIELD_HANDLER_ADDRESS))
    }

    pub fn prompt_stack_pointer(&self) -> usize {
        BuiltInTypes::untag(self.heap_obj.get_field(Self::FIELD_PROMPT_SP))
    }

    pub fn prompt_frame_pointer(&self) -> usize {
        BuiltInTypes::untag(self.heap_obj.get_field(Self::FIELD_PROMPT_FP))
    }

    pub fn prompt_link_register(&self) -> usize {
        BuiltInTypes::untag(self.heap_obj.get_field(Self::FIELD_PROMPT_LR))
    }

    pub fn prompt_result_local(&self) -> isize {
        BuiltInTypes::untag_isize(self.heap_obj.get_field(Self::FIELD_PROMPT_RESULT_LOCAL) as isize)
    }

    pub fn prompt_id(&self) -> usize {
        BuiltInTypes::untag(self.heap_obj.get_field(Self::FIELD_PROMPT_ID))
    }

    pub fn prompt_handler(&self) -> PromptHandler {
        PromptHandler {
            handler_address: self.handler_address(),
            stack_pointer: self.prompt_stack_pointer(),
            frame_pointer: self.prompt_frame_pointer(),
            link_register: self.prompt_link_register(),
            result_local: self.prompt_result_local(),
            prompt_id: self.prompt_id(),
        }
    }

    pub fn initialize(
        heap_obj: &mut HeapObject,
        segment_ptr: usize,
        original_sp: usize,
        original_fp: usize,
        resume_address: usize,
        result_local: isize,
        prompt: &PromptHandler,
    ) {
        heap_obj.write_type_id(TYPE_ID_CONTINUATION as usize);
        heap_obj.write_field(Self::FIELD_SEGMENT as i32, segment_ptr);
        heap_obj.write_field(
            Self::FIELD_ORIGINAL_SP as i32,
            BuiltInTypes::Int.tag(original_sp as isize) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_ORIGINAL_FP as i32,
            BuiltInTypes::Int.tag(original_fp as isize) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_RESUME_ADDRESS as i32,
            BuiltInTypes::Int.tag(resume_address as isize) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_RESULT_LOCAL as i32,
            BuiltInTypes::Int.tag(result_local) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_HANDLER_ADDRESS as i32,
            BuiltInTypes::Int.tag(prompt.handler_address as isize) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_PROMPT_SP as i32,
            BuiltInTypes::Int.tag(prompt.stack_pointer as isize) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_PROMPT_FP as i32,
            BuiltInTypes::Int.tag(prompt.frame_pointer as isize) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_PROMPT_LR as i32,
            BuiltInTypes::Int.tag(prompt.link_register as isize) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_PROMPT_RESULT_LOCAL as i32,
            BuiltInTypes::Int.tag(prompt.result_local) as usize,
        );
        heap_obj.write_field(
            Self::FIELD_PROMPT_ID as i32,
            BuiltInTypes::Int.tag(prompt.prompt_id as isize) as usize,
        );
    }
}

pub struct Runtime {
    pub memory: Memory,
    pub libraries: Vec<Library>,
    #[allow(unused)]
    command_line_arguments: CommandLineArguments,
    pub printer: Box<dyn Printer>,
    // TODO: I don't have any code that looks at u8, just always u64
    // so that's why I need usize
    pub is_paused: AtomicUsize,
    /// Count of threads that are registered and ready to respond to GC.
    /// Threads increment this while holding gc_lock when starting,
    /// and decrement it while holding gc_lock when exiting.
    /// GC uses this instead of join_handles.len() to know how many threads to wait for.
    pub registered_thread_count: AtomicUsize,
    pub thread_state: Arc<(Mutex<ThreadState>, Condvar)>,
    pub gc_lock: Mutex<()>,
    pub ffi_function_info: Vec<FFIInfo>,
    pub ffi_info_by_name: HashMap<String, usize>,
    namespaces: NamespaceManager,
    pub structs: StructManager,
    pub enums: EnumManager,
    pub string_constants: Vec<StringValue>,
    /// Pre-interned tagged string literal values for ASCII chars 0..128.
    /// Indexed by byte value. Avoids heap allocation for single-char string returns.
    pub ascii_char_cache: [usize; 128],
    pub keyword_constants: Vec<StringValue>,
    pub keyword_heap_ptrs: Vec<Option<usize>>,
    pub diagnostic_store: Arc<Mutex<crate::compiler::DiagnosticStore>>,
    // TODO: Do I need anything more than
    // namespace manager? Shouldn't these functions
    // and things be under that?
    pub functions: Vec<Function>,
    jump_table_reserved: Reserved,
    jump_table_pages: Vec<Mmap>,
    jump_table_base_ptr: usize,
    pub jump_table_offset: usize,
    compiler_channel: Option<BlockingSender<CompilerMessage, CompilerResponse>>,
    compiler_thread: Option<JoinHandle<()>>,
    protocol_info: HashMap<String, Vec<ProtocolMethodInfo>>,
    /// Dispatch tables for O(1) protocol method lookup
    /// Key = "protocol_name/method_name"
    /// Boxed to keep pointers stable when HashMap reallocates
    dispatch_tables: HashMap<String, Box<DispatchTable>>,
    stacks_for_continuation_swapping: Vec<ContinuationStack>,
    // Per-thread try-catch handler stacks
    pub exception_handlers: HashMap<ThreadId, Vec<ExceptionHandler>>,
    // Per-thread prompt handler stacks for delimited continuations
    pub prompt_handlers: HashMap<ThreadId, Vec<PromptHandler>>,
    // Per-thread invocation return points for multi-shot continuations.
    // When a continuation k is invoked via k(value), we push where to return to here.
    // When the continuation body completes, return_from_shift pops from here.
    pub invocation_return_points: HashMap<ThreadId, Vec<InvocationReturnPoint>>,
    // Per-thread counter for how many return_from_shift calls to skip.
    // This is incremented when continuations are fully invoked, to skip the
    // spurious return_from_shift calls that happen when handlers return.
    pub skip_return_from_shift: HashMap<ThreadId, usize>,
    // Per-thread dynamic variable binding stacks
    // Key = (namespace_id, slot), Value = stack of previous values
    // Each thread has its own binding stacks for dynamic variable rebinding
    pub dynamic_binding_stacks: HashMap<ThreadId, HashMap<(usize, usize), Vec<usize>>>,
    // Per-thread flag indicating pop_prompt is calling return_from_shift.
    // When set, return_from_shift should use InvocationReturnPoints.
    // When not set (handler return case), it should use the passed continuation pointer.
    pub return_from_shift_via_pop_prompt: HashMap<ThreadId, bool>,
    // Per-thread flag indicating return_from_shift is being called from handler return
    // (after `call-handler` in perform). When set, return_from_shift should skip
    // popping InvocationReturnPoints and use the passed continuation pointer instead.
    pub is_handler_return: HashMap<ThreadId, bool>,
    // Per-thread saved continuation pointer from when via_return_point was used.
    // When a continuation is invoked and returns via via_return_point, we save
    // the continuation pointer here. Then when return_from_shift_handler is called
    // with a null cont_ptr (because the stack local was overwritten), we use this
    // saved pointer instead.
    pub saved_continuation_ptr: HashMap<ThreadId, usize>,
    /// Depth of relocated stacks for each thread. When > 0, we're running on a
    /// relocated stack and should not copy modifications back to "original_sp"
    /// because the "original" is itself a relocated location.
    pub relocation_depth: HashMap<ThreadId, usize>,
    // Counter for generating unique prompt IDs to distinguish sequential handlers
    pub prompt_id_counter: AtomicUsize,
    // Per-thread uncaught exception handlers (Beagle function pointers)
    pub thread_exception_handler_fns: HashMap<ThreadId, usize>,
    // Global default uncaught exception handler (Beagle function pointer)
    pub default_exception_handler_fn: Option<usize>,
    // Namespace ID for keyword GC roots
    keyword_namespace: usize,
    /// Queue of pending binding updates for PersistentMap.
    /// Used when bindings are added from threads without a Beagle stack (e.g., compiler thread).
    /// Format: (namespace_id, slot, value)
    pending_heap_bindings: Mutex<Vec<(usize, usize, usize)>>,
    /// Mapping from enum variant struct_id to enum name
    /// Used by effect handlers to determine which handler to call for a `perform` value
    pub variant_to_enum: HashMap<usize, String>,
    /// Storage for compiled Regex objects.
    /// Regexes are stored here and accessed by index.
    /// The index is tagged as a special "Regex" type for Beagle.
    pub compiled_regexes: Vec<Regex>,
    /// Wait set for efficient future waiting.
    /// Maps atom addresses to condition variables that threads wait on.
    /// When an atom is updated (via reset!), waiters should be notified.
    pub future_wait_set: FutureWaitSet,
    /// Registry for mio-based event loops.
    /// Allows creating multiple event loops for different async contexts.
    pub event_loops: EventLoopRegistry,
    /// Registered FFI callbacks (C → Beagle).
    /// Each entry's `beagle_fn` is kept alive via GC root.
    /// The trampoline code references callbacks by index.
    pub callbacks: Vec<CallbackInfo>,
}

pub fn create_stack_with_protected_page_after(stack_size: usize) -> MmapMut {
    let page_size = MmapOptions::page_size();
    let stack_size = stack_size + page_size;
    let stack = MmapOptions::new(stack_size)
        .expect("Failed to create mmap for stack - out of memory")
        .map_mut()
        .expect("Failed to map stack memory - this is a fatal error");
    // because the stack grows down we will protect the first page
    // so that if we go over the stack we will get a segfault
    let protected_area = &stack[0..page_size];
    unsafe {
        libc::mprotect(
            protected_area.as_ptr() as *mut c_void,
            page_size,
            libc::PROT_NONE,
        );
    }
    stack
}

type ContinuationStackBuffer = [usize; 512];

fn new_continuation_stack_buffer() -> ContinuationStackBuffer {
    [0; 512]
}

struct ContinuationStack {
    is_used: AtomicBool,
    stack: ContinuationStackBuffer,
}

impl Runtime {
    pub fn new(
        command_line_arguments: CommandLineArguments,
        allocator: Alloc,
        printer: Box<dyn Printer>,
    ) -> Self {
        let jump_table_reserved = MmapOptions::new(MmapOptions::page_size() * 10000)
            .expect("Failed to create mmap options for jump table")
            .reserve()
            .expect("Failed to reserve address space for jump table");
        let jump_table_base_ptr = jump_table_reserved.as_ptr() as usize;

        Self {
            printer,
            command_line_arguments: command_line_arguments.clone(),
            memory: Memory {
                heap: allocator,
                stacks: Mutex::new(vec![(
                    std::thread::current().id(),
                    create_stack_with_protected_page_after(STACK_SIZE),
                )]),
                join_handles: vec![],
                threads: vec![std::thread::current()],
                command_line_arguments,
                stack_map: StackMap::new(),
                thread_globals: Mutex::new(HashMap::new()),
                namespace_root_storage: Vec::new(),
            },
            libraries: vec![],
            is_paused: AtomicUsize::new(0),
            registered_thread_count: AtomicUsize::new(0),
            gc_lock: Mutex::new(()),
            thread_state: Arc::new((
                Mutex::new(ThreadState {
                    stack_pointers: HashMap::new(),
                    c_calling_stack_pointers: HashMap::new(),
                }),
                Condvar::new(),
            )),
            ffi_function_info: vec![],
            ffi_info_by_name: HashMap::new(),
            namespaces: NamespaceManager::new(),
            structs: StructManager::new(),
            enums: EnumManager::new(),
            string_constants: vec![],
            ascii_char_cache: [0; 128],
            keyword_constants: vec![],
            keyword_heap_ptrs: vec![],
            jump_table_reserved,
            jump_table_pages: vec![],
            jump_table_base_ptr,
            jump_table_offset: 0,
            functions: vec![],
            compiler_channel: None,
            compiler_thread: None,
            protocol_info: HashMap::new(),
            dispatch_tables: HashMap::new(),
            diagnostic_store: Arc::new(Mutex::new(crate::compiler::DiagnosticStore::new())),
            stacks_for_continuation_swapping: vec![ContinuationStack {
                is_used: AtomicBool::new(false),
                stack: [0; 512],
            }],
            exception_handlers: {
                let mut map = HashMap::new();
                map.insert(std::thread::current().id(), Vec::new());
                map
            },
            prompt_handlers: {
                let mut map = HashMap::new();
                map.insert(std::thread::current().id(), Vec::new());
                map
            },
            invocation_return_points: {
                let mut map = HashMap::new();
                map.insert(std::thread::current().id(), Vec::new());
                map
            },
            skip_return_from_shift: HashMap::new(),
            dynamic_binding_stacks: {
                let mut map = HashMap::new();
                map.insert(std::thread::current().id(), HashMap::new());
                map
            },
            return_from_shift_via_pop_prompt: HashMap::new(),
            is_handler_return: HashMap::new(),
            saved_continuation_ptr: HashMap::new(),
            relocation_depth: HashMap::new(),
            prompt_id_counter: AtomicUsize::new(1),
            thread_exception_handler_fns: HashMap::new(),
            default_exception_handler_fn: None,
            keyword_namespace: 0, // Will be set when first keyword is allocated
            pending_heap_bindings: Mutex::new(Vec::new()),
            variant_to_enum: HashMap::new(),
            compiled_regexes: Vec::new(),
            future_wait_set: FutureWaitSet::new(),
            event_loops: EventLoopRegistry::new(),
            callbacks: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.wait_for_other_threads();

        // Reset saved frame pointer and gc return address to prevent
        // stale values from being used during gc-always mode
        crate::builtins::reset_saved_gc_context();

        self.memory.reset();
        self.namespaces = NamespaceManager::new();
        self.structs = StructManager::new();
        self.enums = EnumManager::new();
        self.string_constants.clear();
        self.ascii_char_cache = [0; 128];
        self.keyword_constants.clear();
        self.keyword_heap_ptrs.clear();
        self.functions.clear();
        self.jump_table_reserved = MmapOptions::new(MmapOptions::page_size() * 10000)
            .expect("Failed to create mmap options for jump table")
            .reserve()
            .expect("Failed to reserve address space for jump table");
        self.jump_table_base_ptr = self.jump_table_reserved.as_ptr() as usize;
        self.jump_table_pages.clear();
        self.jump_table_offset = 0;
        self.printer.reset();
        self.ffi_function_info.clear();
        self.ffi_info_by_name.clear();
        self.compiler_channel
            .as_mut()
            .expect("Compiler channel not initialized - this is a fatal error")
            .send(CompilerMessage::Reset);
        self.protocol_info.clear();
        self.dispatch_tables.clear();

        // Clear continuation and exception handler state to prevent stale pointers
        self.exception_handlers = {
            let mut map = HashMap::new();
            map.insert(std::thread::current().id(), Vec::new());
            map
        };
        self.prompt_handlers = {
            let mut map = HashMap::new();
            map.insert(std::thread::current().id(), Vec::new());
            map
        };
        self.invocation_return_points = {
            let mut map = HashMap::new();
            map.insert(std::thread::current().id(), Vec::new());
            map
        };
        self.skip_return_from_shift.clear();
        self.return_from_shift_via_pop_prompt.clear();
        self.is_handler_return.clear();
        self.thread_exception_handler_fns.clear();
        self.default_exception_handler_fn = None;
        self.stacks_for_continuation_swapping.clear();
        self.saved_continuation_ptr.clear();
        self.relocation_depth.clear();
        self.variant_to_enum.clear();
        self.compiled_regexes.clear();

        // Shutdown all event loops and their I/O threads to prevent stale references
        self.event_loops.shutdown_all();
    }

    /// Export all documentation as JSON for the documentation generator.
    /// Returns a JSON string containing all namespaces, functions, structs, and enums.
    pub fn export_documentation(&self) -> String {
        use std::collections::BTreeMap;

        // Collect all namespaces
        let mut namespaces: BTreeMap<String, Vec<FunctionDoc>> = BTreeMap::new();

        for function in &self.functions {
            // Skip internal functions
            if function.name.starts_with("__") || function.name.contains("/__") {
                continue;
            }

            // Parse namespace from function name
            let (namespace, fn_name) = if let Some(pos) = function.name.rfind('/') {
                (
                    function.name[..pos].to_string(),
                    function.name[pos + 1..].to_string(),
                )
            } else {
                ("global".to_string(), function.name.clone())
            };

            let doc = FunctionDoc {
                name: fn_name,
                full_name: function.name.clone(),
                docstring: function.docstring.clone(),
                arg_names: function.arg_names.clone(),
                arity: function.number_of_args,
                is_variadic: function.is_variadic,
                min_args: function.min_args,
                is_builtin: function.is_builtin,
                source_file: function.source_file.clone(),
                source_line: function.source_line,
            };

            namespaces.entry(namespace).or_default().push(doc);
        }

        // Collect structs
        let mut struct_docs: Vec<StructDoc> = Vec::new();
        for struct_info in self.structs.iter() {
            let fields: Vec<FieldDoc> = struct_info
                .fields
                .iter()
                .enumerate()
                .map(|(i, field_name)| FieldDoc {
                    name: field_name.clone(),
                    docstring: struct_info.field_docstrings.get(i).cloned().flatten(),
                    mutable: struct_info.mutable_fields.get(i).copied().unwrap_or(false),
                })
                .collect();

            struct_docs.push(StructDoc {
                name: struct_info.name.clone(),
                docstring: struct_info.docstring.clone(),
                fields,
                source_file: None, // Struct doesn't track source file yet
                source_line: None,
            });
        }

        // Collect enums
        let mut enum_docs: Vec<EnumDoc> = Vec::new();
        for enum_info in self.enums.iter() {
            let variants: Vec<VariantDoc> = enum_info
                .variants
                .iter()
                .map(|v| match v {
                    crate::runtime::EnumVariant::StructVariant { name, fields } => VariantDoc {
                        name: name.clone(),
                        fields: fields.clone(),
                    },
                    crate::runtime::EnumVariant::StaticVariant { name } => VariantDoc {
                        name: name.clone(),
                        fields: vec![],
                    },
                })
                .collect();

            enum_docs.push(EnumDoc {
                name: enum_info.name.clone(),
                docstring: enum_info.docstring.clone(),
                variants,
                source_file: None, // Enum doesn't track source file yet
                source_line: None,
            });
        }

        // Build the documentation structure
        let doc = Documentation {
            namespaces: namespaces
                .into_iter()
                .map(|(name, functions)| NamespaceDoc { name, functions })
                .collect(),
            structs: struct_docs,
            enums: enum_docs,
        };

        // Serialize to JSON
        nanoserde::SerJson::serialize_json(&doc)
    }

    pub fn start_compiler_thread(&mut self) {
        if self.compiler_channel.is_none() {
            let (sender, receiver) = blocking_channel();
            let args_clone = self.command_line_arguments.clone();
            let diagnostic_store_clone = Arc::clone(&self.diagnostic_store);
            let compiler_thread = thread::Builder::new()
                .name("Beagle Compiler".to_string())
                .stack_size(16 * 1024 * 1024) // 16MB stack for compiler thread
                .spawn(move || {
                    CompilerThread::new(receiver, args_clone, diagnostic_store_clone)
                        .expect("Failed to create compiler thread - this is a fatal error")
                        .run();
                })
                .expect("Failed to spawn compiler thread - this is a fatal error");
            self.compiler_channel = Some(sender);
            self.compiler_thread = Some(compiler_thread);
        }
    }

    pub fn print(&mut self, result: usize) {
        let result = self
            .get_repr(result, 0)
            .expect("Failed to get representation for print - this is a fatal error");
        self.printer.print(result);
    }

    pub fn println(&mut self, pointer: usize) -> Result<(), Box<dyn Error>> {
        let result = self.get_repr(pointer, 0).ok_or("Error printing")?;
        self.printer.println(result);
        Ok(())
    }

    pub fn is_paused(&self) -> bool {
        // Use Acquire ordering to synchronize-with the Release store in gc_impl.
        // This ensures that when a thread sees is_paused = 1, it also sees all
        // GC setup writes that happened before the store.
        self.is_paused.load(std::sync::atomic::Ordering::Acquire) == 1
    }

    pub fn pause_atom_ptr(&self) -> usize {
        self.is_paused.as_ptr() as usize
    }

    pub fn compile(&mut self, file_name: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let response = self
            .compiler_channel
            .as_ref()
            .expect("Compiler channel not initialized - this is a fatal error")
            .send(CompilerMessage::CompileFile(file_name.to_string()));
        match response {
            CompilerResponse::FunctionsToRun(functions) => Ok(functions),
            CompilerResponse::CompileError(msg) => {
                eprintln!("Compile error: {}", msg);
                Err(format!("Error compiling: {}", msg).into())
            }
            _ => {
                eprintln!("Unexpected compiler response");
                Err("Error compiling".into())
            }
        }
    }

    pub fn compile_source(
        &mut self,
        name: &str,
        source: &str,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let response = self
            .compiler_channel
            .as_ref()
            .expect("Compiler channel not initialized - this is a fatal error")
            .send(CompilerMessage::CompileSource(
                name.to_string(),
                source.to_string(),
            ));
        match response {
            CompilerResponse::FunctionsToRun(functions) => Ok(functions),
            CompilerResponse::CompileError(msg) => {
                eprintln!("Compile error: {}", msg);
                Err(format!("Error compiling: {}", msg).into())
            }
            _ => {
                eprintln!("Unexpected compiler response");
                Err("Error compiling".into())
            }
        }
    }

    pub fn compile_string(&mut self, _string: &str) -> Result<usize, Box<dyn Error>> {
        let response = self
            .compiler_channel
            .as_ref()
            .expect("Compiler channel not initialized - this is a fatal error")
            .send(CompilerMessage::CompileString(_string.to_string()));
        match response {
            CompilerResponse::FunctionPointer(pointer) => Ok(pointer),
            CompilerResponse::CompileError(msg) => Err(msg.into()),
            _ => Err("Unexpected compiler response".into()),
        }
    }

    pub fn allocate_string(
        &mut self,
        stack_pointer: usize,
        string: String,
    ) -> Result<Tagged, Box<dyn Error>> {
        self.allocate_string_from_bytes(stack_pointer, string.as_bytes())
    }

    /// Allocate a string directly from a byte slice, avoiding an intermediate Rust String.
    pub fn allocate_string_from_bytes(
        &mut self,
        stack_pointer: usize,
        bytes: &[u8],
    ) -> Result<Tagged, Box<dyn Error>> {
        let text_words = bytes.len().div_ceil(8);
        let total_words = 1 + text_words; // 1 for cached hash + text
        let pointer = self.allocate(total_words, stack_pointer, BuiltInTypes::HeapObject)?;
        let pointer = self.memory.allocate_string(bytes, pointer)?;
        Ok(pointer)
    }

    /// Allocate a string slice: a lightweight view into a parent string.
    /// Layout: [header][parent_ptr][offset_tagged]
    /// The parent pointer is traced by GC (non-opaque object).
    /// `parent` must be a tagged pointer to a flat string or string slice.
    /// If parent is itself a slice, we resolve to the original flat string.
    pub fn allocate_string_slice(
        &mut self,
        stack_pointer: usize,
        parent: usize,
        byte_offset: usize,
        byte_length: usize,
    ) -> Result<Tagged, Box<dyn Error>> {
        // If parent is a slice, resolve to the original flat string
        let (real_parent, real_offset) = if BuiltInTypes::is_heap_pointer(parent) {
            let parent_obj = HeapObject::from_tagged(parent);
            if parent_obj.is_string_slice() {
                let grandparent = parent_obj.get_field(0);
                let parent_offset = BuiltInTypes::untag(parent_obj.get_field(1));
                (grandparent, parent_offset + byte_offset)
            } else {
                (parent, byte_offset)
            }
        } else {
            (parent, byte_offset)
        };

        // Root the parent so GC doesn't collect it during allocation
        let parent_root = self.register_temporary_root(real_parent);

        // Allocate 3-field non-opaque object (field 2 = cached hash)
        let pointer = self.allocate(3, stack_pointer, BuiltInTypes::HeapObject)?;

        // Read back parent (may have moved during GC)
        let real_parent = self.get_handle_root(parent_root);
        self.unregister_temporary_root(parent_root);

        let mut heap_object = HeapObject::from_tagged(pointer);
        heap_object.writer_header_direct(Header {
            type_id: TYPE_ID_STRING_SLICE,
            type_data: byte_length as u32,
            size: 3,
            opaque: false, // GC must trace parent pointer
            marked: false,
            large: false,
            type_flags: 0,
        });
        // Field 0: parent pointer (tagged heap pointer)
        heap_object.write_field(0, real_parent);
        // Field 1: byte offset (tagged int)
        heap_object.write_field(
            1,
            BuiltInTypes::construct_int(real_offset as isize) as usize,
        );
        // Field 2: cached hash (0 = not yet computed, stored as tagged int for GC safety)
        heap_object.write_field(2, 0);

        // Write barrier: the slice points to the parent
        self.write_barrier(pointer, real_parent);

        Ok(BuiltInTypes::HeapObject.tagged(pointer))
    }

    /// Allocate a cons string: a lazy concatenation node.
    /// Layout: [header][left_ptr][right_ptr][cached_hash=0], 3 fields, opaque=false.
    /// `type_data` = total byte length (sum of children).
    /// `type_flags` bit 0 = is_ascii (AND of children's ascii flags).
    pub fn allocate_cons_string(
        &mut self,
        stack_pointer: usize,
        left: usize,
        right: usize,
    ) -> Result<Tagged, Box<dyn Error>> {
        let left_root = self.register_temporary_root(left);
        let right_root = self.register_temporary_root(right);

        let left_len = self.get_string_byte_length(left);
        let right_len = self.get_string_byte_length(right);
        let total_len = left_len + right_len;

        let left_ascii = self.is_string_ascii(left);
        let right_ascii = self.is_string_ascii(right);
        let is_ascii = left_ascii && right_ascii;

        let pointer = self.allocate(3, stack_pointer, BuiltInTypes::HeapObject)?;

        let left = self.get_handle_root(left_root);
        let right = self.get_handle_root(right_root);
        self.unregister_temporary_root(left_root);
        self.unregister_temporary_root(right_root);

        let mut heap_object = HeapObject::from_tagged(pointer);
        heap_object.writer_header_direct(Header {
            type_id: TYPE_ID_CONS_STRING,
            type_data: total_len as u32,
            size: 3,
            opaque: false,
            marked: false,
            large: false,
            type_flags: if is_ascii { 1 } else { 0 },
        });
        heap_object.write_field(0, left);
        heap_object.write_field(1, right);
        heap_object.write_field(2, 0);

        self.write_barrier(pointer, left);
        self.write_barrier(pointer, right);

        Ok(BuiltInTypes::HeapObject.tagged(pointer))
    }

    /// Get the byte length of any string value (constant, flat, slice, or cons).
    pub fn get_string_byte_length(&self, value: usize) -> usize {
        let tag = BuiltInTypes::get_kind(value);
        if tag == BuiltInTypes::String {
            let s = self.get_str_literal(value);
            return s.len();
        }
        let heap_object = HeapObject::from_tagged(value);
        heap_object.get_header().type_data as usize
    }

    /// Check if a string value is ASCII (constant, flat, slice, or cons).
    pub fn is_string_ascii(&self, value: usize) -> bool {
        let tag = BuiltInTypes::get_kind(value);
        if tag == BuiltInTypes::String {
            let s = self.get_str_literal(value);
            return s.is_ascii();
        }
        let heap_object = HeapObject::from_tagged(value);
        let header = heap_object.get_header();
        if header.type_id == TYPE_ID_STRING_SLICE {
            let parent = HeapObject::from_tagged(heap_object.get_field(0));
            parent.get_header().type_flags & 1 != 0
        } else {
            header.type_flags & 1 != 0
        }
    }

    /// Collect all bytes from any string value into a Vec.
    /// Handles constants, flat strings, slices, and cons strings.
    /// Uses an explicit stack for cons trees to avoid stack overflow.
    pub fn get_string_bytes_vec(&self, value: usize) -> Vec<u8> {
        let tag = BuiltInTypes::get_kind(value);
        if tag == BuiltInTypes::String {
            let s = self.get_str_literal(value);
            return s.as_bytes().to_vec();
        }
        let heap_object = HeapObject::from_tagged(value);
        if !heap_object.is_cons_string() {
            return heap_object.get_string_bytes().to_vec();
        }
        let total_len = heap_object.get_header().type_data as usize;
        let mut buf = Vec::with_capacity(total_len);
        let mut work_stack: Vec<usize> = vec![value];
        while let Some(val) = work_stack.pop() {
            let val_tag = BuiltInTypes::get_kind(val);
            if val_tag == BuiltInTypes::String {
                let s = self.get_str_literal(val);
                buf.extend_from_slice(s.as_bytes());
            } else {
                let obj = HeapObject::from_tagged(val);
                if obj.is_cons_string() {
                    work_stack.push(obj.get_field(1));
                    work_stack.push(obj.get_field(0));
                } else {
                    buf.extend_from_slice(obj.get_string_bytes());
                }
            }
        }
        buf
    }

    /// Creates a Beagle array of strings from Rust strings.
    /// Returns a tagged pointer to the array.
    pub fn create_string_array(
        &mut self,
        stack_pointer: usize,
        strings: &[String],
    ) -> Result<usize, Box<dyn Error>> {
        let num_elements = strings.len();
        let array_ptr =
            self.allocate_zeroed(num_elements, stack_pointer, BuiltInTypes::HeapObject)?;

        let mut heap_obj = HeapObject::from_tagged(array_ptr);
        heap_obj.write_type_id(1); // type_id=1 marks raw array

        // Root the array so GC can update its pointer during string allocations.
        // Without this, GC could move the array and we'd write to stale memory.
        let array_root_slot = self
            .add_handle_root(array_ptr)
            .ok_or("Failed to add root")?;

        for (i, s) in strings.iter().enumerate() {
            let string_ptr = self.allocate_string(stack_pointer, s.clone())?;
            // Get the current array pointer from the root (GC updates this when array moves)
            let current_array_ptr = self.get_handle_root(array_root_slot);
            heap_obj = HeapObject::from_tagged(current_array_ptr);
            heap_obj.write_field(i as i32, string_ptr.into());
        }

        // Get final pointer and remove root
        let final_array_ptr = self.get_handle_root(array_root_slot);
        self.remove_handle_root(array_root_slot);

        Ok(final_array_ptr)
    }

    /// Low-level keyword allocation. Private because keywords must be interned
    /// for equality to work correctly (keywords compare by pointer identity).
    /// Use `intern_keyword` instead.
    fn allocate_keyword_raw(
        &mut self,
        stack_pointer: usize,
        keyword_text: String,
    ) -> Result<Tagged, Box<dyn Error>> {
        let bytes = keyword_text.as_bytes();
        // Need space for: 1 word for hash + words for text
        let text_words = bytes.len().div_ceil(8); // Round up
        let words = 1 + text_words;
        let pointer = self.allocate(words, stack_pointer, BuiltInTypes::HeapObject)?;
        let pointer = self.memory.allocate_keyword(bytes, pointer)?;
        Ok(pointer)
    }

    /// Intern a keyword: check if it exists, otherwise allocate and register as GC root
    pub fn intern_keyword(
        &mut self,
        stack_pointer: usize,
        keyword_text: String,
    ) -> Result<usize, Box<dyn Error>> {
        // First check if this keyword text already has an index
        let index = if let Some(idx) = self
            .keyword_constants
            .iter()
            .position(|k| k.str == keyword_text)
        {
            idx
        } else {
            // Add new keyword to the constants table
            self.keyword_constants.push(StringValue {
                str: keyword_text.clone(),
            });
            self.keyword_heap_ptrs.push(None);
            self.keyword_constants.len() - 1
        };

        // Ensure keyword namespace exists
        if self.keyword_namespace == 0 {
            self.keyword_namespace = self.namespaces.add_namespace("beagle.internal.keywords");
        }

        // Check heap-based PersistentMap first (survives GC relocation)
        let heap_ptr = self.get_heap_binding(self.keyword_namespace, index);
        if heap_ptr != BuiltInTypes::null_value() as usize {
            return Ok(heap_ptr);
        }

        // Check Rust-side cache (may be stale after GC but try it as fallback)
        if let Some(ptr) = self.keyword_heap_ptrs[index] {
            // Store in heap-based map for future lookups
            if let Err(e) = self.set_heap_binding(stack_pointer, self.keyword_namespace, index, ptr)
            {
                eprintln!("Warning: failed to set heap binding for keyword: {}", e);
            }
            return Ok(ptr);
        }

        // Allocate the keyword
        let ptr = self
            .allocate_keyword_raw(stack_pointer, keyword_text)?
            .into();

        // Store in heap-based PersistentMap (survives GC)
        // NOTE: set_heap_binding may trigger GC during PersistentMap::assoc,
        // which could move the keyword. We must re-read the correct pointer afterward.
        if let Err(e) = self.set_heap_binding(stack_pointer, self.keyword_namespace, index, ptr) {
            eprintln!("Warning: failed to set heap binding for keyword: {}", e);
        }

        // Re-read the keyword pointer from heap binding.
        // This is critical because GC may have moved the keyword during set_heap_binding.
        let ptr = self.get_heap_binding(self.keyword_namespace, index);

        // Also cache in Rust-side vec (may become stale but faster first lookup)
        self.keyword_heap_ptrs[index] = Some(ptr);

        Ok(ptr)
    }

    pub fn allocate(
        &mut self,
        words: usize,
        stack_pointer: usize,
        kind: BuiltInTypes,
    ) -> Result<usize, Box<dyn Error>> {
        self.allocate_with_retries(words, stack_pointer, kind, 0)
    }

    fn allocate_with_retries(
        &mut self,
        words: usize,
        stack_pointer: usize,
        kind: BuiltInTypes,
        retries: usize,
    ) -> Result<usize, Box<dyn Error>> {
        const MAX_GROW_RETRIES: usize = 20;

        let options = self.memory.heap.get_allocation_options();

        // Get frame pointer from thread-local storage (set by builtin entry)
        let frame_pointer = crate::builtins::get_saved_frame_pointer();

        if options.gc_always {
            self.run_gc(stack_pointer, frame_pointer);
        }

        let result = self.memory.heap.try_allocate(words, kind);

        match result {
            Ok(AllocateAction::Allocated(value)) => {
                assert!(value.is_aligned());
                let value = kind.tag(value as isize);
                Ok(value as usize)
            }
            Ok(AllocateAction::Gc) => {
                self.run_gc(stack_pointer, frame_pointer);
                let result = self.memory.heap.try_allocate(words, kind);
                if let Ok(AllocateAction::Allocated(result)) = result {
                    // tag
                    assert!(result.is_aligned());
                    let result = kind.tag(result as isize);
                    Ok(result as usize)
                } else {
                    if retries >= MAX_GROW_RETRIES {
                        return Err(format!(
                            "Out of memory: failed to allocate {} words ({} bytes) after {} grow attempts",
                            words, words * 8, retries
                        ).into());
                    }
                    self.memory.heap.grow();
                    let pointer =
                        self.allocate_with_retries(words, stack_pointer, kind, retries + 1)?;
                    // If we went down this path, our pointer is already tagged
                    Ok(pointer)
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Allocate with zeroed memory (for arrays that don't initialize all fields)
    pub fn allocate_zeroed(
        &mut self,
        words: usize,
        stack_pointer: usize,
        kind: BuiltInTypes,
    ) -> Result<usize, Box<dyn Error>> {
        self.allocate_zeroed_with_retries(words, stack_pointer, kind, 0)
    }

    fn allocate_zeroed_with_retries(
        &mut self,
        words: usize,
        stack_pointer: usize,
        kind: BuiltInTypes,
        retries: usize,
    ) -> Result<usize, Box<dyn Error>> {
        const MAX_GROW_RETRIES: usize = 20;

        let options = self.memory.heap.get_allocation_options();
        let frame_pointer = crate::builtins::get_saved_frame_pointer();

        if options.gc_always {
            self.run_gc(stack_pointer, frame_pointer);
        }

        let result = self.memory.heap.try_allocate_zeroed(words, kind);

        match result {
            Ok(AllocateAction::Allocated(value)) => {
                assert!(value.is_aligned());
                let value = kind.tag(value as isize);
                Ok(value as usize)
            }
            Ok(AllocateAction::Gc) => {
                self.run_gc(stack_pointer, frame_pointer);
                let result = self.memory.heap.try_allocate_zeroed(words, kind);
                if let Ok(AllocateAction::Allocated(result)) = result {
                    assert!(result.is_aligned());
                    let result = kind.tag(result as isize);
                    Ok(result as usize)
                } else {
                    if retries >= MAX_GROW_RETRIES {
                        return Err(format!(
                            "Out of memory: failed to allocate {} words ({} bytes) after {} grow attempts",
                            words, words * 8, retries
                        ).into());
                    }
                    self.memory.heap.grow();
                    let pointer =
                        self.allocate_zeroed_with_retries(words, stack_pointer, kind, retries + 1)?;
                    Ok(pointer)
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Write barrier for generational GC.
    ///
    /// Call this after writing a pointer value into a heap object's field.
    /// For generational GC, this records old-to-young pointers so they can
    /// be traced during minor GC.
    ///
    /// Parameters:
    /// - `object_ptr`: Tagged pointer to the object being written to
    /// - `new_value`: The value that was written (may or may not be a heap pointer)
    #[inline]
    pub fn write_barrier(&mut self, object_ptr: usize, new_value: usize) {
        self.memory.heap.write_barrier(object_ptr, new_value);
    }

    /// Get the card table biased pointer for generated code write barriers.
    ///
    /// For generational GC, returns a biased pointer such that:
    /// `biased_ptr[addr >> 9] = 1` marks the card containing `addr` as dirty.
    ///
    /// For non-generational GCs, returns null.
    #[inline]
    pub fn get_card_table_biased_ptr(&self) -> *mut u8 {
        self.memory.heap.get_card_table_biased_ptr()
    }

    /// Mark the card for an object if it's in old gen.
    /// This is used by generated code write barriers.
    /// The object_ptr should be a tagged HeapObject pointer.
    #[inline]
    pub fn mark_card_for_object(&mut self, object_ptr: usize) {
        // For generational GC: mark the card if object is in old gen
        // This is used by codegen write barriers where we don't know the value being written
        self.memory.heap.mark_card_unconditional(object_ptr);
    }

    /// Set a field on a heap object and call the write barrier.
    /// This is the safe way to write heap pointers to object fields in Rust code.
    ///
    /// Parameters:
    /// - `object_ptr`: Tagged pointer to the object being written to
    /// - `index`: Field index to write to
    /// - `value`: The value to write (may be a tagged int or heap pointer)
    #[inline]
    pub fn set_field_with_barrier(&mut self, object_ptr: usize, index: usize, value: usize) {
        let obj = HeapObject::from_tagged(object_ptr);
        obj.write_field(index as i32, value);
        self.write_barrier(object_ptr, value);
    }

    /// Create a Beagle struct or enum from Rust code
    ///
    /// # Arguments
    /// * `struct_name` - Fully qualified struct name (e.g., "beagle.core/SystemError")
    /// * `variant_name` - For enums, the variant name (e.g., "StructError"). None for regular structs
    /// * `fields` - Field values in the order they appear in the struct/variant definition
    /// * `stack_pointer` - Current stack pointer for GC
    ///
    /// # Example
    /// ```rust
    /// let fields = vec![message_str, location_str];
    /// let error = runtime.create_struct(
    ///     "beagle.core/SystemError",
    ///     Some("StructError"),
    ///     &fields,
    ///     stack_pointer
    /// )?;
    /// ```
    pub fn create_struct(
        &mut self,
        struct_name: &str,
        variant_name: Option<&str>,
        fields: &[usize],
        stack_pointer: usize,
    ) -> Result<usize, Box<dyn Error>> {
        // Look up the struct definition
        let actual_struct_id = if let Some(variant) = variant_name {
            // For enum variants, look up the variant struct (e.g., "Enum.Variant")
            let variant_struct_name = format!("{}.{}", struct_name, variant);
            let (variant_id, _variant_def) = self
                .get_struct(&variant_struct_name)
                .ok_or_else(|| format!("Variant struct '{}' not found", variant_struct_name))?;

            variant_id
        } else {
            // For regular structs, look up the struct directly
            let (id, _struct_def) = self
                .get_struct(struct_name)
                .ok_or_else(|| format!("Struct '{}' not found", struct_name))?;
            id
        };

        // Allocate heap object
        let obj_ptr = self.allocate(fields.len(), stack_pointer, BuiltInTypes::HeapObject)?;
        let heap_obj = HeapObject::from_tagged(obj_ptr);

        // Write struct_id to header using Header API (no manual bit manipulation!)
        let untagged = heap_obj.untagged();
        let header_ptr = untagged as *mut usize;
        unsafe {
            let mut header = Header::from_usize(*header_ptr);
            // struct_id is stored as raw value (not tagged)
            header.type_data = actual_struct_id as u32;
            *header_ptr = header.to_usize();
        }

        // Write fields directly (both structs and enum variants)
        // For enum variants, the variant is identified by the struct_id, not a separate field
        for (i, &field_value) in fields.iter().enumerate() {
            heap_obj.write_field(i as i32, field_value);
        }

        Ok(obj_ptr)
    }

    unsafe fn buffer_between<T>(start: *const T, end: *const T) -> &'static [T] {
        unsafe {
            let len = end.offset_from(start);
            slice::from_raw_parts(start, len as usize)
        }
    }

    fn find_function_for_pc(&self, pc: usize) -> Option<&Function> {
        self.functions.iter().find(|f| {
            let start = f.pointer.into();
            let end = start + f.size;
            pc >= start && pc < end
        })
    }

    #[allow(unused)]
    pub fn parse_stack_frames(&self, stack_pointer: usize) -> Vec<StackValue> {
        let stack_base = self.get_stack_base();
        let stack = unsafe {
            Self::buffer_between(
                (stack_pointer as *const usize).sub(1),
                stack_base as *const usize,
            )
        };
        println!("Stack: {:?}", stack);
        let mut stack_frames = vec![];
        let mut i = 0;
        while i < stack.len() {
            let value = stack[i];
            if let Some(details) = self.memory.stack_map.find_stack_data(value) {
                let mut frame_size = details.max_stack_size + details.number_of_locals;
                if frame_size % 2 != 0 {
                    frame_size += 1
                }
                let current_frame = &stack[i..i + frame_size + 1];
                let stack_value = StackValue {
                    function: self.find_function_for_pc(value).map(|f| f.name.clone()),
                    details: details.clone(),
                    stack: current_frame.to_vec(),
                };
                println!("Stack value: {:#?}", stack_value);
                stack_frames.push(stack_value);
                i += details.current_stack_size + details.number_of_locals + 1;
                continue;
            }
            i += 1;
        }
        stack_frames
    }

    // Exception handling methods
    pub fn push_exception_handler(&mut self, handler: ExceptionHandler) {
        let thread_id = std::thread::current().id();
        self.exception_handlers
            .entry(thread_id)
            .or_default()
            .push(handler);
    }

    pub fn pop_exception_handler(&mut self) -> Option<ExceptionHandler> {
        let thread_id = std::thread::current().id();
        self.exception_handlers.get_mut(&thread_id)?.pop()
    }

    pub fn set_thread_exception_handler(&mut self, handler_fn: usize) {
        let thread_id = std::thread::current().id();
        self.thread_exception_handler_fns
            .insert(thread_id, handler_fn);
    }

    pub fn set_default_exception_handler(&mut self, handler_fn: usize) {
        self.default_exception_handler_fn = Some(handler_fn);
    }

    pub fn get_thread_exception_handler(&self) -> Option<usize> {
        let thread_id = std::thread::current().id();
        self.thread_exception_handler_fns.get(&thread_id).copied()
    }

    pub fn update_exception_handlers_after_gc(&mut self) {
        // No-op: Exception handlers and keywords are now stored in heap-based PersistentMap
        // which is automatically updated by GC during tracing
    }

    // Delimited continuation (prompt handler) methods
    pub fn push_prompt_handler(&mut self, handler: PromptHandler) {
        let thread_id = std::thread::current().id();
        self.prompt_handlers
            .entry(thread_id)
            .or_default()
            .push(handler);
    }

    pub fn pop_prompt_handler(&mut self) -> Option<PromptHandler> {
        let thread_id = std::thread::current().id();
        self.prompt_handlers.get_mut(&thread_id)?.pop()
    }

    pub fn prompt_handler_count(&self) -> usize {
        let thread_id = std::thread::current().id();
        self.prompt_handlers
            .get(&thread_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    pub fn cleanup_finished_thread_handlers(&mut self) {
        // Get list of active thread IDs
        let active_thread_ids: std::collections::HashSet<ThreadId> = self
            .memory
            .threads
            .iter()
            .map(|t| t.id())
            .chain(std::iter::once(std::thread::current().id()))
            .collect();

        // Remove exception handlers for threads that are no longer active
        let dead_threads: Vec<ThreadId> = self
            .exception_handlers
            .keys()
            .filter(|id| !active_thread_ids.contains(id))
            .copied()
            .collect();

        for thread_id in dead_threads {
            self.exception_handlers.remove(&thread_id);
            self.thread_exception_handler_fns.remove(&thread_id);
        }
    }

    pub fn create_exception(
        &mut self,
        _stack_pointer: usize,
        value: usize,
    ) -> Result<usize, Box<dyn Error>> {
        // Create a simple exception object - for now just return the value
        // In the future, this could be a struct with stack trace information
        Ok(value)
    }

    /// GC entry point called when allocation fails or gc_always is set.
    /// With frame pointers enabled, we just read the current FP and walk from there.
    fn run_gc(&mut self, stack_pointer: usize, frame_pointer: usize) {
        let gc_return_addr = crate::builtins::get_saved_gc_return_addr();
        self.gc_impl(stack_pointer, frame_pointer, gc_return_addr);
    }

    pub fn gc_impl(&mut self, stack_pointer: usize, frame_pointer: usize, gc_return_addr: usize) {
        // Save the gc context so that any nested __pause calls use the correct address.
        // This is critical: if we call __pause from within gc_impl, __pause must use
        // the gc_return_addr we received (which points to Beagle code), not its own
        // return address (which would point to gc_impl, Rust code).
        crate::builtins::save_gc_return_addr(gc_return_addr);
        crate::builtins::save_frame_pointer(frame_pointer);
        // With frame pointers enabled everywhere (via -C force-frame-pointers=yes),
        // we just start walking from the current Rust function's FP.
        // The stack walker checks each [FP+8] against the stack map:
        // - If in stack map, it's a Beagle frame → scan it
        // - If not in stack map, it's a Rust frame → skip it
        // This naturally handles the mixed Rust/Beagle call stack.

        if self.memory.threads.len() == 1 {
            // If there is only one thread, that is us
            // that means nothing else could spin up a thread in the mean time
            // so there is no need to lock anything
            // The tuple is (stack_base, frame_pointer, gc_return_addr)
            // We pass the saved gc_return_addr to ensure the first Beagle frame is scanned
            let mut all_stack_pointers =
                vec![(self.get_stack_base(), frame_pointer, gc_return_addr)];

            // Also include all ThreadGlobal stack bases explicitly. This ensures the
            // GlobalObjectBlock head pointers are always traced even if a thread's
            // pause bookkeeping misses adding its stack entry (e.g., platform quirks).
            let thread_globals = self.memory.thread_globals.lock().unwrap();
            for tg in thread_globals.values() {
                if tg.stack_base != 0 {
                    all_stack_pointers.push((tg.stack_base, 0, 0));
                }
            }
            drop(thread_globals);

            // Collect namespace bindings as GC roots
            // IMPORTANT: Namespace bindings are NOT in a heap-based PersistentMap despite
            // what old comments said. They're in HashMap<String, usize> and must be
            // explicitly passed as GC roots to prevent dynamic variables from being collected.
            let mut namespace_roots: Vec<usize> = Vec::new();
            let mut namespace_keys: Vec<Vec<String>> = Vec::new();
            for ns_mutex in &self.namespaces.namespaces {
                let ns = ns_mutex.lock().expect("Failed to lock namespace for GC");
                let keys: Vec<String> = ns.bindings.keys().cloned().collect();
                let values: Vec<usize> = ns.bindings.values().copied().collect();
                namespace_keys.push(keys);
                namespace_roots.extend(values);
            }

            let updated_roots = self.memory.run_gc(&all_stack_pointers, &namespace_roots);

            // Write updated namespace roots back to namespaces (for copying GC)
            let mut root_index = 0;
            for (ns_mutex, keys) in self.namespaces.namespaces.iter().zip(namespace_keys.iter()) {
                let mut ns = ns_mutex.lock().expect("Failed to lock namespace after GC");
                for key in keys {
                    if root_index < updated_roots.len() {
                        ns.bindings.insert(key.clone(), updated_roots[root_index]);
                        root_index += 1;
                    }
                }
            }

            return;
        }

        let locked = self.gc_lock.try_lock();

        if let Err(e) = &locked {
            match e {
                TryLockError::WouldBlock => {
                    drop(locked);
                    unsafe { __pause(stack_pointer, frame_pointer) };
                }
                TryLockError::Poisoned(e) => {
                    eprintln!("Warning: Poisoned lock in GC: {:?}", e);
                    // Try to recover by using the poisoned data anyway
                    // The lock is poisoned but the data might still be usable
                    drop(locked);
                    unsafe { __pause(stack_pointer, frame_pointer) };
                }
            }

            return;
        }
        // Use Release ordering when setting is_paused = 1 to ensure all prior
        // GC setup writes are visible to threads that see is_paused = 1.
        // Threads read with Acquire ordering, creating a synchronizes-with relationship.
        let result = self.is_paused.compare_exchange(
            0,
            1,
            std::sync::atomic::Ordering::Release,
            std::sync::atomic::Ordering::Relaxed,
        );
        if result != Ok(0) {
            drop(locked);
            unsafe { __pause(stack_pointer, frame_pointer) };
            return;
        }

        // Fire USDT probe - we're initiating stop-the-world
        usdt_probes::fire_stw_begin();

        let locked = locked.expect("Failed to lock GC - this is a fatal error");

        let (lock, cvar) = &*self.thread_state;
        let mut thread_state = lock
            .lock()
            .expect("Failed to lock thread state - this is a fatal error");

        // Count threads we need to wait for.
        // Use registered_thread_count which tracks threads that are actually ready to respond to GC.
        // Both main thread and child threads are registered, so we always subtract 1 because
        // the current thread (doing GC) is implicitly "paused" since it's doing GC.
        let registered_count = self
            .registered_thread_count
            .load(std::sync::atomic::Ordering::Acquire);
        let mut threads_to_wait_for = registered_count.saturating_sub(1);

        while thread_state.paused_threads() + thread_state.c_calling_stack_pointers.len()
            < threads_to_wait_for
        {
            // Use wait_timeout to avoid infinite blocking
            let result = cvar
                .wait_timeout(thread_state, std::time::Duration::from_millis(10))
                .expect("Failed waiting on condition variable - this is a fatal error");
            thread_state = result.0;

            // Recalculate in case new threads registered
            let registered_count = self
                .registered_thread_count
                .load(std::sync::atomic::Ordering::Acquire);
            threads_to_wait_for = registered_count.saturating_sub(1);
        }

        // Fire USDT probe - all threads are now paused
        let num_paused =
            thread_state.paused_threads() + thread_state.c_calling_stack_pointers.len();
        let total_threads = self
            .registered_thread_count
            .load(std::sync::atomic::Ordering::Acquire)
            + thread_state.c_calling_stack_pointers.len();
        usdt_probes::fire_stw_all_paused(num_paused, total_threads);

        // Paused threads already have (base, fp, gc_return_addr) triples
        // Filter out null stacks - threads that pause with (0, 0, 0) are exiting
        // and don't have a valid stack to scan.
        let mut stack_pointers: Vec<(usize, usize, usize)> = thread_state
            .stack_pointers
            .values()
            .copied()
            .filter(|(base, _fp, _gc_ret)| *base != 0)
            .collect();
        // Also include threads that are in C calls (FFI) - their stacks must be scanned too!
        // Filter out null stacks - threads that are exiting may register with (0, 0, 0)
        // to maintain thread count, but we can't scan a null stack.
        let c_calling_stacks: Vec<_> = thread_state
            .c_calling_stack_pointers
            .iter()
            .map(|(tid, s)| (*tid, *s))
            .collect();

        stack_pointers.extend(
            c_calling_stacks
                .iter()
                .map(|(_, s)| *s)
                .filter(|(base, _fp, _gc_ret)| *base != 0),
        );

        // Add an explicit entry for each ThreadGlobal's stack base. This keeps the
        // GlobalObjectBlock chain rooted even if a thread fails to register a pause
        // record (observed on arm64 where handler roots went missing).
        let thread_globals = self.memory.thread_globals.lock().unwrap();
        for tg in thread_globals.values() {
            if tg.stack_base != 0 {
                stack_pointers.push((tg.stack_base, 0, 0));
            }
        }
        drop(thread_globals);

        // Main thread uses the saved frame_pointer and gc_return_addr
        stack_pointers.push((self.get_stack_base(), frame_pointer, gc_return_addr));

        drop(thread_state);

        // Collect namespace bindings as GC roots
        // IMPORTANT: Namespace bindings are NOT in a heap-based PersistentMap despite
        // what old comments said. They're in HashMap<String, usize> and must be
        // explicitly passed as GC roots to prevent dynamic variables from being collected.
        let mut namespace_roots: Vec<usize> = Vec::new();
        let mut namespace_keys: Vec<Vec<String>> = Vec::new();
        for ns_mutex in &self.namespaces.namespaces {
            let ns = ns_mutex.lock().expect("Failed to lock namespace for GC");
            let keys: Vec<String> = ns.bindings.keys().cloned().collect();
            let values: Vec<usize> = ns.bindings.values().copied().collect();
            namespace_keys.push(keys);
            namespace_roots.extend(values);
        }

        let updated_roots = self.memory.run_gc(&stack_pointers, &namespace_roots);

        // Write updated namespace roots back to namespaces (for copying GC)
        let mut root_index = 0;
        for (ns_mutex, keys) in self.namespaces.namespaces.iter().zip(namespace_keys.iter()) {
            let mut ns = ns_mutex.lock().expect("Failed to lock namespace after GC");
            for key in keys {
                if root_index < updated_roots.len() {
                    ns.bindings.insert(key.clone(), updated_roots[root_index]);
                    root_index += 1;
                }
            }
        }

        // Memory barrier to ensure all GC writes are visible before continuing
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        // Fire USDT probe - stop-the-world is ending
        usdt_probes::fire_stw_end();

        self.is_paused
            .store(0, std::sync::atomic::Ordering::Release);

        self.memory.active_threads();
        for thread in self.memory.threads.iter() {
            thread.unpark();
        }

        let mut thread_state = lock
            .lock()
            .expect("Failed to lock thread state after GC - this is a fatal error");
        while thread_state.paused_threads() > 0 {
            let (state, timeout) = cvar
                .wait_timeout(thread_state, Duration::from_millis(1))
                .expect(
                    "Failed waiting on condition variable with timeout - this is a fatal error",
                );
            thread_state = state;

            if timeout.timed_out() {
                self.memory.active_threads();
                for thread in self.memory.threads.iter() {
                    // println!("Unparking thread {:?}", thread.thread().id());
                    thread.unpark();
                }
            }
        }
        thread_state.clear();

        drop(locked);
    }

    pub fn register_temporary_root(&mut self, root: usize) -> usize {
        // Use the handle root mechanism which is backed by Memory.thread_globals
        self.add_handle_root(root).unwrap_or(0)
    }

    pub fn unregister_temporary_root(&mut self, id: usize) -> usize {
        // Fast path: use cached thread_local pointer (no mutex)
        let cached = CACHED_THREAD_GLOBAL.with(|c| c.get());
        if !cached.is_null() {
            let tg = unsafe { &mut *cached };
            return tg.remove_root(id);
        }

        let thread_id = std::thread::current().id();
        self.memory
            .thread_globals
            .lock()
            .unwrap()
            .get_mut(&thread_id)
            .map(|tg| tg.remove_root(id))
            .unwrap_or(0)
    }

    pub fn peek_temporary_root(&self, id: usize) -> usize {
        self.get_handle_root(id)
    }

    /// Initialize the GlobalObject for the current thread.
    /// This allocates a GlobalObjectBlock and creates a ThreadGlobal.
    /// Should be called once per thread at thread start.
    pub fn initialize_thread_global(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let thread_id = std::thread::current().id();
        let stack_base = self.get_stack_base();
        self.initialize_thread_global_for(thread_id, stack_base)
    }

    /// Initialize the GlobalObject for a specific thread.
    /// This allocates a GlobalObjectBlock and creates a ThreadGlobal.
    /// Can be called by parent thread to initialize child thread's GlobalObject.
    pub fn initialize_thread_global_for(
        &mut self,
        thread_id: std::thread::ThreadId,
        stack_base: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Allocate GlobalObjectBlock first (before checking/locking)
        // This avoids holding the lock during allocation
        let head_block = self
            .memory
            .heap
            .allocate_for_runtime(GLOBAL_BLOCK_TOTAL_FIELDS)?;

        // Initialize the block
        let block = GlobalObjectBlock::from_tagged(head_block);
        block.initialize();

        // Create the ThreadGlobal (boxed for stable address)
        let thread_global = Box::new(ThreadGlobal::new(head_block, thread_id, stack_base));

        // Atomically insert if not already present (entry API ensures atomicity)
        let mut thread_globals = self.memory.thread_globals.lock().unwrap();
        use std::collections::hash_map::Entry;
        let actual_head_block = match thread_globals.entry(thread_id) {
            Entry::Vacant(e) => {
                let tg_ptr: *mut ThreadGlobal = &mut **e.insert(thread_global);
                // Cache in thread_local for lock-free access
                if thread_id == std::thread::current().id() {
                    CACHED_THREAD_GLOBAL.with(|c| c.set(tg_ptr));
                }
                head_block // Use the newly allocated block
            }
            Entry::Occupied(e) => {
                // Already initialized by another thread, use existing head_block
                let tg_ptr: *mut ThreadGlobal = &**e.get() as *const _ as *mut _;
                if thread_id == std::thread::current().id() {
                    CACHED_THREAD_GLOBAL.with(|c| c.set(tg_ptr));
                }
                e.get().head_block.load(Ordering::SeqCst)
            }
        };
        drop(thread_globals);

        // Write the GlobalObjectBlock pointer to the stack at stack_base - 8.
        // This is critical for GC: the stack walker reads from this slot, and after GC,
        // Memory::run_gc syncs ThreadGlobal.head_block from the updated stack slot.
        // For the current thread, we write immediately. For child threads, run_thread
        // writes separately after the child's stack is set up.
        if thread_id == std::thread::current().id() {
            unsafe {
                *((stack_base - 8) as *mut usize) = actual_head_block;
            }
        }

        Ok(())
    }

    /// Check if the current thread has a GlobalObject initialized.
    pub fn has_thread_global(&self) -> bool {
        let thread_id = std::thread::current().id();
        self.memory
            .thread_globals
            .lock()
            .unwrap()
            .contains_key(&thread_id)
    }

    /// Ensure the CACHED_THREAD_GLOBAL thread-local is populated for the current thread.
    /// If not cached, looks up via mutex and caches. Returns the pointer.
    pub fn ensure_cached_thread_global(&self) -> *mut ThreadGlobal {
        let thread_id = std::thread::current().id();
        let thread_globals = self.memory.thread_globals.lock().unwrap();
        if let Some(tg) = thread_globals.get(&thread_id) {
            let tg_ptr: *mut ThreadGlobal = &**tg as *const _ as *mut _;
            CACHED_THREAD_GLOBAL.with(|c| c.set(tg_ptr));
            tg_ptr
        } else {
            std::ptr::null_mut()
        }
    }

    /// Initialize the namespaces atom in GlobalObject slot 0.
    ///
    /// Creates an Atom containing an empty PersistentMap and stores it in the
    /// reserved slot. This should be called once during runtime initialization,
    /// after `initialize_thread_global()`.
    ///
    /// The namespaces atom stores: symbol → PersistentMap of bindings
    /// Each namespace's bindings are: symbol → value
    pub fn initialize_namespaces(&mut self) -> Result<(), Box<dyn Error>> {
        let thread_id = std::thread::current().id();

        // Check if already initialized (slot 0 should be null/free initially)
        let current = self
            .memory
            .thread_globals
            .lock()
            .unwrap()
            .get(&thread_id)
            .map(|tg| tg.get_namespaces_atom())
            .unwrap_or(GLOBAL_BLOCK_FREE_SLOT);
        if current != GLOBAL_BLOCK_FREE_SLOT && current != BuiltInTypes::null_value() as usize {
            // Already initialized
            return Ok(());
        }

        // Use stack base as stack pointer during initialization
        // (we're at the top of the stack during init, no Beagle frames yet)
        let stack_pointer = self.get_stack_base();

        // Create empty PersistentMap for namespace storage
        let mut scope = HandleScope::new(self, stack_pointer);
        let empty_map = PersistentMap::empty(scope.runtime(), stack_pointer)?;
        let empty_map_h = scope.alloc_handle(empty_map);

        // Create Atom with 1 field (the map)
        let atom = scope.allocate_typed(1, TYPE_ID_ATOM)?;
        atom.to_gc_handle()
            .set_field_with_barrier(scope.runtime(), 0, empty_map_h.get());

        // Store atom in GlobalObject slot 0
        let atom_ptr = atom.get();
        drop(scope);

        let thread_globals = self.memory.thread_globals.lock().unwrap();
        if let Some(tg) = thread_globals.get(&thread_id) {
            tg.set_namespaces_atom(atom_ptr);
        }
        Ok(())
    }

    /// Get the namespaces atom from GlobalObject slot 0.
    /// Returns the tagged pointer to the Atom, or null if not initialized.
    pub fn get_namespaces_atom(&self) -> usize {
        let thread_id = std::thread::current().id();
        self.memory
            .thread_globals
            .lock()
            .unwrap()
            .get(&thread_id)
            .map(|tg| tg.get_namespaces_atom())
            .unwrap_or(GLOBAL_BLOCK_FREE_SLOT)
    }

    /// Create a composite key for binding lookup.
    /// Encodes (namespace_id, slot) as a tagged integer.
    #[inline]
    fn make_binding_key(namespace_id: usize, slot: usize) -> usize {
        // Use 20 bits for slot (supports up to 1M bindings per namespace)
        // and remaining bits for namespace_id
        let key = (namespace_id << 20) | (slot & 0xFFFFF);
        BuiltInTypes::construct_int(key as isize) as usize
    }

    /// Get a binding value from the heap-based PersistentMap.
    ///
    /// Returns the value if found, or null if not found.
    /// This is the core lookup for the new heap-based binding storage.
    pub fn get_heap_binding(&self, namespace_id: usize, slot: usize) -> usize {
        let atom_ptr = self.get_namespaces_atom();
        if !BuiltInTypes::is_heap_pointer(atom_ptr) {
            return BuiltInTypes::null_value() as usize;
        }

        // Additional validation: check pointer is properly aligned
        let untagged = BuiltInTypes::untag(atom_ptr);
        if !untagged.is_multiple_of(8) {
            // Pointer is corrupted (possibly stale after GC)
            return BuiltInTypes::null_value() as usize;
        }

        // Deref atom to get the map
        let atom = HeapObject::from_tagged(atom_ptr);
        let map_ptr = atom.get_field(0);
        if !BuiltInTypes::is_heap_pointer(map_ptr) {
            return BuiltInTypes::null_value() as usize;
        }

        // Additional validation for map pointer
        let map_untagged = BuiltInTypes::untag(map_ptr);
        if !map_untagged.is_multiple_of(8) {
            return BuiltInTypes::null_value() as usize;
        }

        let key = Self::make_binding_key(namespace_id, slot);
        let map = crate::collections::GcHandle::from_tagged(map_ptr);
        PersistentMap::get(self, map, key)
    }

    /// Set a binding value in the heap-based PersistentMap.
    ///
    /// Creates a new map with the binding added/updated and swaps it into the atom.
    /// This is the core update for the new heap-based binding storage.
    pub fn set_heap_binding(
        &mut self,
        stack_pointer: usize,
        namespace_id: usize,
        slot: usize,
        value: usize,
    ) -> Result<(), Box<dyn Error>> {
        let thread_id = std::thread::current().id();
        let atom_ptr = self
            .memory
            .thread_globals
            .lock()
            .unwrap()
            .get(&thread_id)
            .map(|tg| tg.get_namespaces_atom())
            .unwrap_or(GLOBAL_BLOCK_FREE_SLOT);
        if !BuiltInTypes::is_heap_pointer(atom_ptr) {
            return Err("Namespaces atom not initialized".into());
        }

        // Deref atom to get the current map
        let atom = HeapObject::from_tagged(atom_ptr);
        let map_ptr = atom.get_field(0);

        let key = Self::make_binding_key(namespace_id, slot);

        // Create new map with the binding
        let new_map = PersistentMap::assoc(self, stack_pointer, map_ptr, key, value)?;

        // Write new map to atom (this is atomic at the word level)
        // CRITICAL: Re-read atom_ptr from GlobalObjectBlock because GC may have moved it
        // during PersistentMap::assoc(). The GlobalObjectBlock entry is updated by GC's
        // stack walker, but our local atom_ptr variable is stale.
        let atom_ptr = self
            .memory
            .thread_globals
            .lock()
            .unwrap()
            .get(&thread_id)
            .map(|tg| tg.get_namespaces_atom())
            .unwrap_or(GLOBAL_BLOCK_FREE_SLOT);
        let atom = HeapObject::from_tagged(atom_ptr);
        let new_map_tagged = new_map.as_tagged();

        atom.write_field(0, new_map_tagged);

        // Write barrier: notify GC that an old gen object (atom) may now point to young gen (new_map)
        self.write_barrier(atom_ptr, new_map_tagged);

        Ok(())
    }

    /// Flush any pending heap binding updates.
    ///
    /// Called from threads that have a stack (e.g., main thread) to process
    /// bindings that were queued by threads without a stack (e.g., compiler thread).
    pub fn flush_pending_heap_bindings(&mut self, stack_pointer: usize) {
        // Take all pending bindings (swap with empty vec to minimize lock time)
        let pending = {
            let mut guard = self
                .pending_heap_bindings
                .lock()
                .expect("Failed to lock pending_heap_bindings");
            std::mem::take(&mut *guard)
        };

        // Process each pending binding
        for (namespace_id, slot, value) in pending {
            if let Err(e) = self.set_heap_binding(stack_pointer, namespace_id, slot, value) {
                eprintln!(
                    "Warning: failed to flush heap binding (ns={}, slot={}): {}",
                    namespace_id, slot, e
                );
            }
        }
    }

    /// Get the GlobalObjectBlock pointer for the current thread.
    /// Returns 0 if no ThreadGlobal is initialized for this thread.
    pub fn get_global_block_ptr(&self) -> usize {
        let thread_id = std::thread::current().id();
        self.memory
            .thread_globals
            .lock()
            .unwrap()
            .get(&thread_id)
            .map(|tg| tg.head_block.load(Ordering::SeqCst))
            .unwrap_or(0)
    }

    /// Add a handle root to the current thread's GlobalObject.
    /// Automatically allocates new GlobalObjectBlocks when needed.
    /// Returns the slot index, or None if allocation fails.
    pub fn add_handle_root(&mut self, value: usize) -> Option<usize> {
        // Fast path: use cached thread_local pointer (no mutex)
        let cached = CACHED_THREAD_GLOBAL.with(|c| c.get());
        if !cached.is_null() {
            let tg = unsafe { &mut *cached };
            if let Some(slot) = tg.add_root(value) {
                return Some(slot);
            }
            // Block is full, fall through to slow path to allocate a new block
            return self.add_handle_root_slow(value, cached);
        }

        // No cached pointer — fall back to mutex lookup
        let thread_id = std::thread::current().id();
        let mut thread_globals = self.memory.thread_globals.lock().unwrap();
        if let Some(tg) = thread_globals.get_mut(&thread_id) {
            // Cache for future calls
            let tg_ptr: *mut ThreadGlobal = &mut **tg;
            CACHED_THREAD_GLOBAL.with(|c| c.set(tg_ptr));
            if let Some(slot) = tg.add_root(value) {
                return Some(slot);
            }
            drop(thread_globals);
            return self.add_handle_root_slow(value, tg_ptr);
        }
        None
    }

    /// Slow path for add_handle_root: allocate a new GlobalObjectBlock and link it.
    fn add_handle_root_slow(&mut self, value: usize, tg_ptr: *mut ThreadGlobal) -> Option<usize> {
        // Allocate new block using allocate_for_runtime (long-lived)
        let new_block = self
            .memory
            .heap
            .allocate_for_runtime(GLOBAL_BLOCK_TOTAL_FIELDS)
            .ok()?;

        // Initialize the new block
        let block = GlobalObjectBlock::from_tagged(new_block);
        block.initialize();

        // Link it and add the root (no mutex needed — only this thread touches its own ThreadGlobal)
        let tg = unsafe { &mut *tg_ptr };
        tg.link_new_block(new_block);
        tg.add_root(value)
    }

    /// Remove a handle root from the current thread's GlobalObject.
    pub fn remove_handle_root(&mut self, slot: usize) {
        // Fast path: use cached thread_local pointer (no mutex)
        let cached = CACHED_THREAD_GLOBAL.with(|c| c.get());
        if !cached.is_null() {
            let tg = unsafe { &mut *cached };
            tg.remove_root(slot);
            return;
        }

        // Fallback: mutex lookup
        let thread_id = std::thread::current().id();
        let mut thread_globals = self.memory.thread_globals.lock().unwrap();
        if let Some(tg) = thread_globals.get_mut(&thread_id) {
            tg.remove_root(slot);
        }
    }

    /// Get the value of a handle root from the current thread's GlobalObject.
    pub fn get_handle_root(&self, slot: usize) -> usize {
        // Fast path: use cached thread_local pointer (no mutex)
        let cached = CACHED_THREAD_GLOBAL.with(|c| c.get());
        if !cached.is_null() {
            let tg = unsafe { &*cached };
            return tg.get_root(slot);
        }

        let thread_id = std::thread::current().id();
        self.memory
            .thread_globals
            .lock()
            .unwrap()
            .get(&thread_id)
            .map(|tg| tg.get_root(slot))
            .unwrap_or(0)
    }

    pub fn register_parked_thread(&mut self, stack_pointer: usize) {
        self.memory
            .heap
            .register_parked_thread(std::thread::current().id(), stack_pointer);
    }

    pub fn get_stack_base(&self) -> usize {
        self.try_get_stack_base()
            .expect("Current thread stack not found - this is a fatal error")
    }

    /// Try to get the stack base for the current thread.
    /// Returns None if the current thread doesn't have a registered stack
    /// (e.g., compiler thread).
    pub fn try_get_stack_base(&self) -> Option<usize> {
        let current_thread = std::thread::current().id();
        let stacks = self
            .memory
            .stacks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stacks
            .iter()
            .find(|(thread_id, _)| *thread_id == current_thread)
            .map(|(_, stack)| stack.as_ptr() as usize + STACK_SIZE)
    }

    pub fn make_closure(
        &mut self,
        stack_pointer: usize,
        function: usize,
        free_variables: &[usize],
    ) -> Result<usize, Box<dyn Error>> {
        let len = 8 + 8 + 8 + free_variables.len() * 8;
        let heap_pointer = self.allocate(len / 8, stack_pointer, BuiltInTypes::Closure)?;
        let heap_object = HeapObject::from_tagged(heap_pointer);
        let num_free = free_variables.len();
        let function_definition =
            self.get_function_by_pointer(BuiltInTypes::untag(function) as *const u8);
        let function_definition = function_definition.unwrap_or_else(|| unsafe {
            crate::builtins::throw_runtime_error(
                stack_pointer,
                "FunctionError",
                "Function not found when creating closure".to_string(),
            );
        });
        let num_locals = function_definition.number_of_locals;

        heap_object.write_field(0, function);
        heap_object.write_field(1, BuiltInTypes::Int.tag(num_free as isize) as usize);
        heap_object.write_field(2, BuiltInTypes::Int.tag(num_locals as isize) as usize);
        for (index, value) in free_variables.iter().enumerate() {
            heap_object.write_field((index + 3) as i32, *value);
        }

        // Debug: Verify closure was created correctly
        if std::env::var("BEAGLE_CLOSURE_DEBUG").is_ok() {
            let verify_fn_ptr = heap_object.get_field(0);
            if verify_fn_ptr != function {
                eprintln!(
                    "[CLOSURE_DEBUG] make_closure: MISMATCH! wrote fn={:#x} but read back {:#x}",
                    function, verify_fn_ptr
                );
            }
            if verify_fn_ptr == 0x7 {
                eprintln!(
                    "[CLOSURE_DEBUG] make_closure: CREATED WITH NULL fn_ptr! closure={:#x}",
                    heap_pointer
                );
            }
        }

        Ok(heap_pointer)
    }

    /// Create a FunctionObject - a callable wrapper around a function pointer.
    ///
    /// Unlike closures, FunctionObjects:
    /// - Have no free variables
    /// - Don't receive `self` as arg0 when called
    ///
    /// This allows top-level functions to be passed around as first-class values.
    ///
    /// Layout:
    /// - Header (8 bytes) with type_id = TYPE_ID_FUNCTION_OBJECT
    /// - Field 0: tagged function pointer
    pub fn make_function_object(
        &mut self,
        stack_pointer: usize,
        function: usize,
    ) -> Result<usize, Box<dyn Error>> {
        // Allocate heap object with 1 field for the function pointer
        let heap_pointer = self.allocate(1, stack_pointer, BuiltInTypes::HeapObject)?;
        let mut heap_object = HeapObject::from_tagged(heap_pointer);

        // Set type_id to identify this as a FunctionObject
        heap_object.write_type_id(TYPE_ID_FUNCTION_OBJECT as usize);

        // Store the tagged function pointer
        heap_object.write_field(0, function);

        Ok(heap_pointer)
    }

    #[allow(clippy::type_complexity)]
    pub fn get_function_base(&self, name: &str) -> Option<(u64, u64, fn(u64, u64) -> u64)> {
        let function = self.functions.iter().find(|f| f.name == name)?;

        let trampoline = self.get_trampoline();
        let stack_pointer = self.get_stack_base();

        Some((stack_pointer as u64, function.pointer.into(), trampoline))
    }

    pub fn get_function0(&self, name: &str) -> Option<Box<dyn Fn() -> u64>> {
        let (stack_pointer, start, trampoline) = self.get_function_base(name)?;
        let global_block_ptr = self.get_global_block_ptr();
        Some(Box::new(move || {
            // Write GlobalObjectBlock pointer to stack_base - 8 before calling trampoline
            // GC will find this via stack walking and trace it naturally
            unsafe {
                *((stack_pointer - 8) as *mut usize) = global_block_ptr;
            }
            trampoline(stack_pointer, start)
        }))
    }

    /// Returns the number of arguments a function takes, or None if the function doesn't exist.
    pub fn get_function_arity(&self, name: &str) -> Option<usize> {
        self.functions
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.number_of_args)
    }

    pub fn get_function1(&self, name: &str) -> Option<Box<dyn Fn(u64) -> u64>> {
        let (stack_pointer, start, trampoline_start) = self.get_function_base(name)?;
        let global_block_ptr = self.get_global_block_ptr();
        let f: fn(u64, u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline_start) };
        Some(Box::new(move |arg1| {
            unsafe {
                *((stack_pointer - 8) as *mut usize) = global_block_ptr;
            }
            f(stack_pointer, start, arg1)
        }))
    }

    #[allow(unused)]
    pub fn get_function2(&self, name: &str) -> Option<Box<dyn Fn(u64, u64) -> u64>> {
        let (stack_pointer, start, trampoline_start) = self.get_function_base(name)?;
        let global_block_ptr = self.get_global_block_ptr();
        let f: fn(u64, u64, u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline_start) };
        Some(Box::new(move |arg1, arg2| {
            unsafe {
                *((stack_pointer - 8) as *mut usize) = global_block_ptr;
            }
            f(stack_pointer, start, arg1, arg2)
        }))
    }

    /// Call a Beagle function by name with arguments from Rust code
    /// This is a general-purpose method for builtins to call user-defined functions
    pub fn call_function_by_name(
        &self,
        name: &str,
        args: &[usize],
    ) -> Result<usize, Box<dyn Error>> {
        let function = self
            .get_function_by_name(name)
            .ok_or_else(|| format!("Function '{}' not found", name))?;

        let function_pointer = function.pointer;
        let trampoline = self.get_trampoline();
        let stack_pointer = self.get_stack_base();
        let global_block_ptr = self.get_global_block_ptr();

        // Write GlobalObjectBlock pointer to stack_base - 8
        unsafe {
            *((stack_pointer - 8) as *mut usize) = global_block_ptr;
        }

        // Call the function with the appropriate number of arguments
        let result = match args.len() {
            0 => {
                let trampoline: fn(u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline) };
                trampoline(stack_pointer as u64, function_pointer.into())
            }
            1 => {
                let trampoline: fn(u64, u64, u64) -> u64 =
                    unsafe { std::mem::transmute(trampoline) };
                trampoline(
                    stack_pointer as u64,
                    function_pointer.into(),
                    args[0] as u64,
                )
            }
            2 => {
                let trampoline: fn(u64, u64, u64, u64) -> u64 =
                    unsafe { std::mem::transmute(trampoline) };
                trampoline(
                    stack_pointer as u64,
                    function_pointer.into(),
                    args[0] as u64,
                    args[1] as u64,
                )
            }
            3 => {
                let trampoline: fn(u64, u64, u64, u64, u64) -> u64 =
                    unsafe { std::mem::transmute(trampoline) };
                trampoline(
                    stack_pointer as u64,
                    function_pointer.into(),
                    args[0] as u64,
                    args[1] as u64,
                    args[2] as u64,
                )
            }
            4 => {
                let trampoline: fn(u64, u64, u64, u64, u64, u64) -> u64 =
                    unsafe { std::mem::transmute(trampoline) };
                trampoline(
                    stack_pointer as u64,
                    function_pointer.into(),
                    args[0] as u64,
                    args[1] as u64,
                    args[2] as u64,
                    args[3] as u64,
                )
            }
            _ => {
                return Err(format!(
                    "call_function_by_name: unsupported arity {} for function '{}'",
                    args.len(),
                    name
                )
                .into());
            }
        };

        Ok(result as usize)
    }

    pub fn new_thread(&mut self, f: usize, stack_pointer: usize, frame_pointer: usize) {
        let trampoline = self.get_trampoline();
        let trampoline: fn(u64, u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline) };

        // Get the __run_thread builtin which will:
        // 1. Unregister from C-call
        // 2. Get Thread object from thread root (using current thread's ID)
        // 3. Call __run_thread_closure to extract and run the closure
        // 4. Remove thread root when done
        let thread_run_fn = self
            .get_function_by_name("beagle.builtin/__run_thread")
            .expect("beagle.builtin/__run_thread not found - this is a fatal error");
        let function_pointer = self
            .get_pointer(thread_run_fn)
            .expect("Failed to get pointer for __run_thread - this is a fatal error")
            as usize;

        // Temporarily protect the closure while we allocate the Thread struct
        let closure_temp_id = self.register_temporary_root(f);

        // Create Thread struct containing the closure.
        // Allocate + initialize + register temp root while holding the allocator lock
        // to prevent a GC between these steps.
        let (_thread_obj, thread_temp_id) = {
            let actual_struct_id = self
                .get_struct("beagle.core/Thread")
                .expect("Struct 'beagle.core/Thread' not found")
                .0;
            let options = self.memory.heap.get_allocation_options();
            let frame_pointer = crate::builtins::get_saved_frame_pointer();

            if options.gc_always {
                self.run_gc(stack_pointer, frame_pointer);
            }

            loop {
                let result = self
                    .memory
                    .heap
                    .with_locked_alloc(|alloc| {
                        match alloc.try_allocate(1, BuiltInTypes::HeapObject) {
                            Ok(AllocateAction::Allocated(ptr)) => {
                                let tagged = BuiltInTypes::HeapObject.tag(ptr as isize) as usize;
                                let heap_obj = HeapObject::from_tagged(tagged);

                                // Write struct_id to header using Header API (as raw value).
                                let untagged = heap_obj.untagged();
                                let header_ptr = untagged as *mut usize;
                                unsafe {
                                    let mut header = Header::from_usize(*header_ptr);
                                    header.type_data = actual_struct_id as u32;
                                    *header_ptr = header.to_usize();
                                }

                                heap_obj.write_field(0, f);
                                // Return the tagged pointer - we'll register temp root outside
                                // the allocator lock but while still holding gc_lock
                                Ok(Some(tagged))
                            }
                            Ok(AllocateAction::Gc) => Ok(None),
                            Err(err) => Err(err),
                        }
                    })
                    .expect("Failed to allocate Thread struct");

                if let Some(tagged) = result {
                    // Register temp root outside allocator lock but still holding gc_lock
                    // This is safe because no GC can happen while we hold gc_lock
                    let temp_id = self.register_temporary_root(tagged);
                    break (tagged, temp_id);
                }

                self.run_gc(stack_pointer, frame_pointer);
            }
        };

        // Get the possibly-relocated closure from temp root and update Thread struct
        let relocated_closure = self.peek_temporary_root(closure_temp_id);
        if relocated_closure != f {
            // Closure was moved by GC during allocation, update Thread struct
            // Get current Thread location (may have moved)
            let current_thread_obj = self.peek_temporary_root(thread_temp_id);
            let mut heap_obj = HeapObject::from_tagged(current_thread_obj);
            heap_obj.get_fields_mut()[0] = relocated_closure;
        }

        // Unregister closure temp root - Thread struct now holds the closure
        self.unregister_temporary_root(closure_temp_id);

        if std::env::var("BEAGLE_THREAD_DEBUG").is_ok() {
            let current_thread_obj = self.peek_temporary_root(thread_temp_id);
            let heap_obj = HeapObject::from_tagged(current_thread_obj);
            let closure_field = heap_obj.get_field(0);
            eprintln!(
                "[THREAD_DEBUG] new_thread: thread_obj={:#x} closure_field={:#x}",
                current_thread_obj, closure_field
            );
            if BuiltInTypes::is_heap_pointer(closure_field)
                && matches!(BuiltInTypes::get_kind(closure_field), BuiltInTypes::Closure)
            {
                let closure_obj = HeapObject::from_tagged(closure_field);
                let fn_ptr = closure_obj.get_field(0);
                let fn_ptr_untagged = BuiltInTypes::untag(fn_ptr);
                if let Some(function) = self.get_function_by_pointer(fn_ptr_untagged as *const u8) {
                    eprintln!(
                        "[THREAD_DEBUG] new_thread: closure fn={} args={}",
                        function.name, function.number_of_args
                    );
                } else {
                    eprintln!(
                        "[THREAD_DEBUG] new_thread: closure fn ptr not found: {:#x}",
                        fn_ptr
                    );
                }
            }
        }

        let new_stack = MmapOptions::new(STACK_SIZE)
            .expect("Failed to create mmap for thread stack - out of memory")
            .map_mut()
            .expect("Failed to map thread stack memory - this is a fatal error");
        // IMPORTANT: Use a different name to avoid shadowing the function parameter.
        // The function's stack_pointer is the CURRENT thread's stack, which is what
        // __pause needs when the main thread pauses during thread creation.
        let new_thread_stack_pointer = new_stack.as_ptr() as usize + STACK_SIZE;
        // Save for later use (after spawn, for writing GlobalObjectBlock pointer)
        let child_stack_base = new_thread_stack_pointer;
        let thread_state = self.thread_state.clone();

        // Create a barrier to ensure the spawned thread waits until it's fully registered.
        // This prevents a race where GC could run while the thread is active but not counted.
        use std::sync::Barrier;
        let barrier = Arc::new(Barrier::new(2));
        let barrier_clone = barrier.clone();

        // Hold gc_lock during spawn and until child is registered.
        // This prevents GC from running during the critical startup window.
        // The child will acquire gc_lock in run_thread before transitioning to Beagle code.
        // We use try_lock because if another thread is doing GC, it holds gc_lock and
        // is waiting for us to pause. Blocking would deadlock.
        let gc_lock = loop {
            // Check if GC needs us to pause
            if self.is_paused() {
                unsafe { __pause(stack_pointer, frame_pointer) };
            }

            match self.gc_lock.try_lock() {
                Ok(guard) => break guard,
                Err(_) => {
                    // Couldn't get lock - GC might be starting
                    if self.is_paused() {
                        unsafe { __pause(stack_pointer, frame_pointer) };
                    }
                    std::thread::yield_now();
                }
            }
        };

        let thread = thread::spawn(move || {
            // Wait for main thread to finish registering us
            barrier_clone.wait();

            // No c_calling registration here - run_thread handles GC coordination
            // by waiting for GC, acquiring gc_lock, and calling __pause as first instruction

            // Call trampoline. The builtin run_thread will:
            // 1. Wait for any ongoing GC
            // 2. Acquire gc_lock briefly to prevent new GC during transition
            // 3. Enter Beagle code where __pause is the first instruction
            // 4. On cleanup: wait for GC, acquire gc_lock, remove thread root
            let result = trampoline(new_thread_stack_pointer as u64, function_pointer as u64, 0);

            // If we end while another thread is waiting for us to pause
            // we need to notify that waiter so they can see we are dead.
            let (_lock, cvar) = &*thread_state;
            cvar.notify_one();
            result
        });

        // Fire USDT probe - OS thread now exists but not yet registered
        usdt_probes::fire_thread_spawn();

        {
            let mut stacks = self.memory.stacks.lock().unwrap();
            stacks.push((thread.thread().id(), new_stack));
        }
        self.memory.heap.register_thread(thread.thread().id());

        let thread_id = thread.thread().id();
        let main_thread_id = std::thread::current().id();

        // Initialize exception handler stack for new thread
        self.exception_handlers.insert(thread_id, Vec::new());

        self.memory.threads.push(thread.thread().clone());
        self.memory.join_handles.push(thread);

        // Register the child thread with GC BEFORE releasing gc_lock.
        // This prevents a race where threads.len() > 1 but registered_thread_count
        // hasn't been incremented yet. In that window, GC would use the multi-thread
        // path but calculate threads_to_wait_for = 0, proceeding without scanning
        // the child's stack. By incrementing registered_thread_count and registering
        // the child as c_calling (with null stack), GC will correctly count it.
        // The child's run_thread will transition from c_calling to active Beagle thread.
        self.registered_thread_count
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        {
            let (lock, condvar) = &*self.thread_state.clone();
            let mut state = lock.lock().unwrap();
            state.c_calling_stack_pointers.insert(thread_id, (0, 0, 0));
            condvar.notify_one();
        }
        // Release gc_lock before allocating GlobalObject
        drop(gc_lock);

        // Initialize GlobalObject for the new thread
        // This allocates a GlobalObjectBlock so the child can use temporary roots
        // CRITICAL: If this fails, we must not release the barrier - the thread cannot run without this
        self.initialize_thread_global_for(thread_id, child_stack_base)
            .expect("Failed to initialize ThreadGlobal for new thread - this is a fatal error");

        // Get the Thread object from main thread's temp root (may have been relocated by GC)
        // and store it in the child's reserved GLOBAL_SLOT_THREAD
        let thread_globals = self.memory.thread_globals.lock().unwrap();
        let current_thread_obj = thread_globals
            .get(&main_thread_id)
            .map(|tg| tg.get_root(thread_temp_id))
            .unwrap_or(0);

        // CRITICAL: Share the main thread's namespaces atom with the child thread.
        // Namespaces, keywords, and function bindings are global and must be shared
        // across all threads. Only temporary roots (local variables) are per-thread.
        let main_namespaces_atom = thread_globals
            .get(&main_thread_id)
            .map(|tg| tg.get_namespaces_atom())
            .unwrap_or(GLOBAL_BLOCK_FREE_SLOT);

        if let Some(child_tg) = thread_globals.get(&thread_id) {
            child_tg.set_thread_object(current_thread_obj);
            // Share the namespaces atom (don't create a new one!)
            if main_namespaces_atom != GLOBAL_BLOCK_FREE_SLOT {
                child_tg.set_namespaces_atom(main_namespaces_atom);
            }
        }
        drop(thread_globals);

        // Now we can unregister the temp root - child's GlobalObjectBlock keeps it alive
        self.memory
            .thread_globals
            .lock()
            .unwrap()
            .get_mut(&main_thread_id)
            .map(|tg| tg.remove_root(thread_temp_id));

        // Write the child's GlobalObjectBlock pointer to its stack (at stack_base - 8)
        let thread_globals = self.memory.thread_globals.lock().unwrap();
        if let Some(thread_global) = thread_globals.get(&thread_id) {
            unsafe {
                *((child_stack_base - 8) as *mut usize) =
                    thread_global.head_block.load(Ordering::SeqCst);
            }
        }

        // Signal the spawned thread that registration is complete.
        // The thread was waiting on this barrier before starting execution.
        barrier.wait();
    }

    pub fn wait_for_other_threads(&mut self) {
        let mut panicked_threads = Vec::new();

        loop {
            if self.memory.join_handles.is_empty() {
                break;
            }

            let handles: Vec<_> = self.memory.join_handles.drain(..).collect();
            for thread in handles {
                if let Err(panic_payload) = thread.join() {
                    // Extract panic message if possible
                    let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Unknown panic payload".to_string()
                    };
                    eprintln!("ERROR: Spawned thread panicked: {}", msg);
                    panicked_threads.push(msg);
                }
            }

            // Clean up exception handlers for finished threads
            self.cleanup_finished_thread_handlers();
        }

        // After all threads are joined, if any panicked, propagate the error
        if !panicked_threads.is_empty() {
            panic!(
                "One or more spawned threads panicked during execution: {:?}",
                panicked_threads
            );
        }
    }

    pub fn get_pause_atom(&self) -> usize {
        self.memory.heap.get_pause_pointer()
    }

    pub fn add_library(&mut self, lib: libloading::Library) -> usize {
        self.libraries.push(lib);
        self.libraries.len() - 1
    }

    pub fn copy_object(
        &mut self,
        from_object: HeapObject,
        to_object: &mut HeapObject,
    ) -> Result<usize, Box<dyn Error>> {
        from_object.copy_full_object(to_object);
        Ok(to_object.tagged_pointer())
    }

    pub fn copy_object_except_header(
        &mut self,
        from_object: HeapObject,
        to_object: &mut HeapObject,
    ) -> Result<usize, Box<dyn Error>> {
        from_object.copy_object_except_header(to_object);
        Ok(to_object.tagged_pointer())
    }

    pub fn write_functions_to_pid_map(&self) {
        // https://github.com/torvalds/linux/blob/6485cf5ea253d40d507cd71253c9568c5470cd27/tools/perf/Documentation/jit-interface.txt
        let pid = std::process::id();
        let mut file = match std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(format!("/tmp/perf-{}.map", pid))
        {
            Ok(f) => f,
            Err(_) => return, // Silently skip if we can't create the perf map
        };
        // Each line has the following format, fields separated with spaces:
        // START SIZE symbolname
        // START and SIZE are hex numbers without 0x.
        // symbolname is the rest of the line, so it could contain special characters.
        // Perf map format uses absolute virtual addresses.
        // IMPORTANT: Entries must be sorted by address for samply's symbol table lookup.

        // Collect and sort functions by address
        // Include builtins that have JIT code (size > 0), exclude foreign functions
        let mut entries: Vec<_> = self
            .functions
            .iter()
            .filter(|f| !f.is_foreign && f.size > 0)
            .map(|f| {
                let start: usize = f.pointer.into();
                (start, f.size, f.name.clone())
            })
            .collect();
        entries.sort_by_key(|(start, _, _)| *start);

        for (start, size, name) in entries {
            let line = format!("{:x} {:x} {}\n", start, size, name);
            file.write_all(line.as_bytes())
                .expect("Failed to write to pid map file - this is a fatal error");
        }
    }

    pub fn get_library(&self, library_id: usize) -> &libloading::Library {
        let library_object = HeapObject::from_tagged(library_id);
        let library_id = BuiltInTypes::untag(library_object.get_field(0));
        &self.libraries[library_id]
    }

    pub fn add_ffi_function_info(&mut self, ffi_function_info: FFIInfo) -> usize {
        self.ffi_function_info.push(ffi_function_info);
        self.ffi_function_info.len() - 1
    }

    pub fn get_ffi_info(&self, ffi_info_id: usize) -> &FFIInfo {
        self.ffi_function_info
            .get(ffi_info_id)
            .expect("FFI function info not found - this is a fatal error")
    }

    pub fn add_ffi_info_by_name(&mut self, function_name: String, ffi_info_id: usize) {
        self.ffi_info_by_name.insert(function_name, ffi_info_id);
    }

    pub fn add_callback(&mut self, info: CallbackInfo) -> usize {
        let index = self.callbacks.len();
        self.callbacks.push(info);
        index
    }

    pub fn get_callback(&self, index: usize) -> &CallbackInfo {
        &self.callbacks[index]
    }

    pub fn find_ffi_info_by_name(&self, function_name: &str) -> Option<usize> {
        self.ffi_info_by_name.get(function_name).cloned()
    }

    pub fn reserve_namespace_slot(&self, name: &str) -> usize {
        self.namespaces
            .add_binding(name, BuiltInTypes::null_value() as usize)
    }

    pub fn current_namespace_id(&self) -> usize {
        self.namespaces.current_namespace
    }

    pub fn update_binding(&mut self, namespace_id: usize, namespace_slot: usize, value: usize) {
        let mut namespace = self
            .namespaces
            .get_namespace_by_id(namespace_id)
            .expect("Namespace not found in update_binding - this is a fatal error")
            .lock()
            .expect("Failed to lock namespace in update_binding - this is a fatal error");
        let name = namespace
            .ids
            .get(namespace_slot)
            .expect("Namespace slot not found in update_binding - this is a fatal error")
            .clone();
        namespace.bindings.insert(name, value);
    }

    pub fn get_binding(&self, namespace: usize, slot: usize) -> usize {
        let ns_obj = self
            .namespaces
            .namespaces
            .get(namespace)
            .expect("Namespace not found in get_binding - this is a fatal error");
        let ns = ns_obj
            .lock()
            .expect("Failed to lock namespace in get_binding - this is a fatal error");
        let name = ns
            .ids
            .get(slot)
            .expect("Namespace slot not found in get_binding - this is a fatal error");
        *ns.bindings
            .get(name)
            .expect("Binding not found in namespace - this is a fatal error")
    }

    /// Get namespace binding - alias for get_binding (used by dynamic vars)
    pub fn get_namespace_binding(&self, namespace: usize, slot: usize) -> usize {
        self.get_binding(namespace, slot)
    }

    /// Push a new dynamic binding (temporarily rebind a namespace variable)
    /// Saves the current value on a per-thread stack and sets the new value
    pub fn push_dynamic_binding(&mut self, namespace: usize, slot: usize, new_value: usize) {
        let thread_id = std::thread::current().id();

        // Get the current value before we change it
        let old_value = self.get_binding(namespace, slot);

        // Push the old value onto the binding stack for this thread
        let thread_stacks = self
            .dynamic_binding_stacks
            .entry(thread_id)
            .or_insert_with(HashMap::new);
        let stack = thread_stacks
            .entry((namespace, slot))
            .or_insert_with(Vec::new);
        stack.push(old_value);

        // Update the namespace binding with the new value
        let ns_obj = self
            .namespaces
            .namespaces
            .get(namespace)
            .expect("Namespace not found in push_dynamic_binding");
        let mut ns = ns_obj
            .lock()
            .expect("Failed to lock namespace in push_dynamic_binding");
        let name = ns
            .ids
            .get(slot)
            .expect("Namespace slot not found in push_dynamic_binding")
            .clone();
        ns.bindings.insert(name, new_value);
    }

    /// Pop a dynamic binding (restore previous value)
    /// Restores the value from the per-thread binding stack
    pub fn pop_dynamic_binding(&mut self, namespace: usize, slot: usize) {
        let thread_id = std::thread::current().id();

        // Pop the old value from the binding stack
        let old_value = self
            .dynamic_binding_stacks
            .get_mut(&thread_id)
            .and_then(|thread_stacks| thread_stacks.get_mut(&(namespace, slot)))
            .and_then(|stack| stack.pop())
            .expect("Dynamic binding stack underflow - pop without matching push");

        // Restore the old value to the namespace binding
        let ns_obj = self
            .namespaces
            .namespaces
            .get(namespace)
            .expect("Namespace not found in pop_dynamic_binding");
        let mut ns = ns_obj
            .lock()
            .expect("Failed to lock namespace in pop_dynamic_binding");
        let name = ns
            .ids
            .get(slot)
            .expect("Namespace slot not found in pop_dynamic_binding")
            .clone();
        ns.bindings.insert(name, old_value);
    }

    pub fn reserve_namespace(&mut self, name: String) -> usize {
        self.namespaces.add_namespace(name.as_str())
    }

    pub fn set_current_namespace(&mut self, namespace: usize) {
        self.namespaces.set_current_namespace(namespace);
    }

    pub fn find_binding(&self, namespace_id: usize, name: &str) -> Option<usize> {
        let namespace = self
            .namespaces
            .namespaces
            .get(namespace_id)
            .expect("Namespace not found in find_binding - this is a fatal error");
        let namespace = namespace
            .lock()
            .expect("Failed to lock namespace in find_binding - this is a fatal error");
        namespace.ids.iter().position(|n| n == name)
    }

    pub fn global_namespace_id(&self) -> usize {
        0
    }

    pub fn keyword_namespace_id(&self) -> usize {
        self.keyword_namespace
    }

    /// Returns the names of all test functions (matching `__test_*__` pattern)
    pub fn get_test_function_names(&self) -> Vec<String> {
        self.functions
            .iter()
            .filter(|f| {
                let short_name = f.name.rsplit('/').next().unwrap_or(&f.name);
                short_name.starts_with("__test_") && short_name.ends_with("__")
            })
            .map(|f| f.name.clone())
            .collect()
    }

    pub fn current_namespace_name(&self) -> String {
        self.namespaces
            .namespaces
            .get(self.namespaces.current_namespace)
            .expect("Current namespace not found - this is a fatal error")
            .lock()
            .expect("Failed to lock current namespace - this is a fatal error")
            .name
            .clone()
    }

    fn get_current_namespace(&self) -> &Mutex<Namespace> {
        self.namespaces
            .namespaces
            .get(self.namespaces.current_namespace)
            .expect("Current namespace not found - this is a fatal error")
    }

    pub fn add_alias(&self, namespace_name: String, alias: String) {
        // TODO: I really need to get rid of this mutex business
        let namespace_id = match self.namespaces.get_namespace_id(namespace_name.as_str()) {
            Some(id) => id,
            None => {
                eprintln!(
                    "Warning: Could not find namespace {} when adding alias {}",
                    namespace_name, alias
                );
                return;
            }
        };

        let current_namespace = self.get_current_namespace();
        if let Ok(mut namespace) = current_namespace.lock() {
            namespace.aliases.insert(alias, namespace_id);
        } else {
            eprintln!("Warning: Could not lock namespace to add alias");
        }
    }

    pub fn get_namespace_id(&self, name: &str) -> Option<usize> {
        self.namespaces.get_namespace_id(name)
    }

    /// Return all namespace names (for REPL completion).
    pub fn all_namespace_names(&self) -> Vec<String> {
        self.namespaces.namespace_names.keys().cloned().collect()
    }

    /// Return all function names visible from the current namespace (for REPL completion).
    /// Includes: current namespace bindings + imported aliases.
    pub fn visible_function_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        let current_ns = self.current_namespace_name();

        for f in &self.functions {
            if !f.is_defined {
                continue;
            }
            if let Some(short) = f
                .name
                .strip_prefix(&current_ns)
                .and_then(|s| s.strip_prefix('/'))
            {
                names.push(short.to_string());
            }
        }

        // Also add names from aliased namespaces
        let current_ns_id = self.namespaces.current_namespace;
        if let Some(ns_mutex) = self.namespaces.namespaces.get(current_ns_id) {
            if let Ok(ns) = ns_mutex.lock() {
                for (alias, &ns_id) in &ns.aliases {
                    if let Some(aliased_name) = self.namespaces.id_to_name.get(&ns_id) {
                        let prefix = format!("{}/", aliased_name);
                        for f in &self.functions {
                            if !f.is_defined {
                                continue;
                            }
                            if let Some(short) = f.name.strip_prefix(&prefix) {
                                names.push(format!("{}/{}", alias, short));
                            }
                        }
                    }
                }
            }
        }

        names
    }

    /// Return all struct names (for REPL completion).
    pub fn all_struct_names(&self) -> Vec<String> {
        self.structs.iter().map(|s| s.name.clone()).collect()
    }

    /// Return all enum names and their variants (for REPL completion).
    pub fn all_enum_names_and_variants(&self) -> Vec<(String, Vec<String>)> {
        self.enums
            .iter()
            .map(|e| {
                let variants = e
                    .variants
                    .iter()
                    .map(|v| match v {
                        EnumVariant::StructVariant { name, .. } => name.clone(),
                        EnumVariant::StaticVariant { name } => name.clone(),
                    })
                    .collect();
                (e.name.clone(), variants)
            })
            .collect()
    }

    fn escape_string_for_repr(s: &str) -> String {
        let mut escaped = String::with_capacity(s.len() + 2);
        escaped.push('"');
        for c in s.chars() {
            match c {
                '\\' => escaped.push_str("\\\\"),
                '"' => escaped.push_str("\\\""),
                '\n' => escaped.push_str("\\n"),
                '\r' => escaped.push_str("\\r"),
                '\t' => escaped.push_str("\\t"),
                '\0' => escaped.push_str("\\0"),
                c => escaped.push(c),
            }
        }
        escaped.push('"');
        escaped
    }

    fn quote_string(s: &str, escape: bool) -> String {
        if escape {
            Self::escape_string_for_repr(s)
        } else {
            format!("\"{}\"", s)
        }
    }

    pub fn get_repr(&self, value: usize, depth: usize) -> Option<String> {
        self.get_repr_inner(value, depth, false)
    }

    pub fn get_eval_repr(&self, value: usize, depth: usize) -> Option<String> {
        self.get_repr_inner(value, depth, true)
    }

    fn get_repr_inner(&self, value: usize, depth: usize, escape: bool) -> Option<String> {
        if depth > 10 {
            return Some("...".to_string());
        }
        let tag = BuiltInTypes::get_kind(value);
        match tag {
            BuiltInTypes::Null => Some("null".to_string()),
            BuiltInTypes::Int => Some(BuiltInTypes::untag_isize(value as isize).to_string()),
            BuiltInTypes::Float => {
                let value = BuiltInTypes::untag(value);
                let value = value as *const f64;
                let value = unsafe { *value.add(1) };
                if value.is_nan() {
                    Some("NaN".to_string())
                } else if value.is_infinite() {
                    if value.is_sign_positive() {
                        Some("infinity".to_string())
                    } else {
                        Some("-infinity".to_string())
                    }
                } else {
                    // Ensure whole number floats display with decimal point (4.0 not 4)
                    let s = value.to_string();
                    if s.contains('.') || s.contains('e') || s.contains('E') {
                        Some(s)
                    } else {
                        Some(format!("{}.0", s))
                    }
                }
            }
            BuiltInTypes::String => {
                let value = BuiltInTypes::untag(value);
                let string = &self.string_constants[value];
                if depth > 0 {
                    return Some(Self::quote_string(&string.str, escape));
                }
                Some(string.str.clone())
            }
            BuiltInTypes::Bool => {
                let value = BuiltInTypes::untag(value);
                if value == 0 {
                    Some("false".to_string())
                } else {
                    Some("true".to_string())
                }
            }
            BuiltInTypes::Function => Some("function".to_string()),
            BuiltInTypes::Closure => {
                let heap_object = HeapObject::from_tagged(value);
                let function_pointer = heap_object.get_field(0);
                let num_free = heap_object.get_field(1);
                let num_locals = heap_object.get_field(2);
                let free_variables = heap_object.get_fields()[3..].to_vec();
                let mut repr = "Closure { ".to_string();
                repr.push_str(&self.get_repr_inner(function_pointer, depth + 1, escape)?);
                repr.push_str(", ");
                repr.push_str(&num_free.to_string());
                repr.push_str(", ");
                repr.push_str(&num_locals.to_string());
                repr.push_str(", [");
                for value in free_variables {
                    repr.push_str(&self.get_repr_inner(value, depth + 1, escape)?);
                    repr.push_str(", ");
                }
                repr.push_str("] }");
                Some(repr)
            }
            BuiltInTypes::HeapObject => {
                // TODO: Once I change the setup for heap objects
                // I need to figure out what kind of heap object I have
                let object = HeapObject::from_tagged(value);
                let header = object.get_header();

                // TODO: Make this documented and good
                match header.type_id {
                    0 => {
                        if header.opaque {
                            println!("=====================");
                            println!("{:?} {:?}", header, BuiltInTypes::untag(value) as *const u8);
                            println!("=====================");
                            return Some("ErrorOpaque".to_string());
                        }
                        let struct_id = object.get_struct_id();
                        let struct_value = self.get_struct_by_id(struct_id);
                        Some(self.get_struct_repr_inner(
                            struct_value?,
                            object.get_fields(),
                            depth + 1,
                            escape,
                        )?)
                    }
                    1 => {
                        let fields = object.get_fields();
                        let mut repr = "[ ".to_string();
                        for (index, field) in fields.iter().enumerate() {
                            repr.push_str(&self.get_repr_inner(*field, depth + 1, escape)?);
                            if index != fields.len() - 1 {
                                repr.push_str(", ");
                            }
                        }
                        repr.push_str(" ]");
                        Some(repr)
                    }
                    2 | 34 | 35 => {
                        let bytes = self.get_string_bytes_vec(value);
                        let string = unsafe { std::str::from_utf8_unchecked(&bytes) };
                        if depth > 0 {
                            return Some(Self::quote_string(string, escape));
                        }
                        Some(string.to_string())
                    }
                    3 => {
                        let bytes = object.get_keyword_bytes();
                        let keyword_text = unsafe { std::str::from_utf8_unchecked(bytes) };
                        Some(format!(":{}", keyword_text))
                    }
                    20 => {
                        // PersistentVector
                        let vec_handle = GcHandle::from_tagged(value);
                        let count = PersistentVec::count(vec_handle);
                        let mut repr = "[".to_string();
                        for i in 0..count {
                            if i > 0 {
                                repr.push_str(", ");
                            }
                            let elem = PersistentVec::get(vec_handle, i);
                            repr.push_str(&self.get_repr_inner(elem, depth + 1, escape)?);
                        }
                        repr.push(']');
                        Some(repr)
                    }
                    22 => {
                        // PersistentMap - iterate over all entries
                        let map_handle = GcHandle::from_tagged(value);
                        let count = PersistentMap::count(map_handle);
                        if count == 0 {
                            Some("{}".to_string())
                        } else {
                            // Collect all entries by traversing the map structure
                            let entries = self.collect_map_entries(map_handle);
                            let mut repr = "{".to_string();
                            for (i, (k, v)) in entries.iter().enumerate() {
                                if i > 0 {
                                    repr.push_str(", ");
                                }
                                repr.push_str(&self.get_repr_inner(*k, depth + 1, escape)?);
                                repr.push(' ');
                                repr.push_str(&self.get_repr_inner(*v, depth + 1, escape)?);
                            }
                            repr.push('}');
                            Some(repr)
                        }
                    }
                    _ => {
                        // This is an unknown object. Meaning it is invalid.
                        // We are going to print everything we can to debug this

                        println!("=====================");
                        println!("{:?} {:?}", header, BuiltInTypes::untag(value));
                        println!("=====================");
                        Some("ErrorUnknown".to_string())
                    }
                }
            }
        }
    }

    /// Collect all key-value entries from a PersistentMap for formatting.
    /// Traverses the HAMT structure recursively.
    fn collect_map_entries(&self, map: GcHandle) -> Vec<(usize, usize)> {
        let mut entries = Vec::new();
        let root_ptr = map.get_field(1); // FIELD_ROOT = 1
        let null_val = BuiltInTypes::null_value() as usize;

        if root_ptr == null_val {
            return entries;
        }

        let root = GcHandle::from_tagged(root_ptr);
        self.collect_node_entries(root, &mut entries, null_val);
        entries
    }

    fn collect_node_entries(
        &self,
        node: GcHandle,
        entries: &mut Vec<(usize, usize)>,
        null_val: usize,
    ) {
        use crate::collections::{TYPE_ID_ARRAY_NODE, TYPE_ID_BITMAP_NODE, TYPE_ID_COLLISION_NODE};

        let type_id = node.get_type_id();

        if type_id == TYPE_ID_BITMAP_NODE {
            // BitmapNode: children array has interleaved key/value pairs
            let children_ptr = node.get_field(1); // BN_FIELD_CHILDREN = 1
            let children = GcHandle::from_tagged(children_ptr);
            let children_len = children.field_count();

            let mut i = 0;
            while i < children_len {
                let key = children.get_field(i);
                let value = children.get_field(i + 1);

                if value != null_val {
                    // It's a leaf entry (key, value)
                    entries.push((key, value));
                } else if BuiltInTypes::is_heap_pointer(key) {
                    // Value is null and key is a pointer - check if it's a sub-node
                    let child = GcHandle::from_tagged(key);
                    let child_type = child.get_type_id();
                    if child_type == TYPE_ID_BITMAP_NODE
                        || child_type == TYPE_ID_ARRAY_NODE
                        || child_type == TYPE_ID_COLLISION_NODE
                    {
                        self.collect_node_entries(child, entries, null_val);
                    }
                }
                // If value is null and key is not a node pointer, it's a leaf with null value
                i += 2;
            }
        } else if type_id == TYPE_ID_ARRAY_NODE {
            // ArrayNode: 32 slots, each slot is either null or a child node
            let children_ptr = node.get_field(1); // AN_FIELD_CHILDREN = 1
            let children = GcHandle::from_tagged(children_ptr);

            for i in 0..32 {
                let child_ptr = children.get_field(i);
                if child_ptr != null_val {
                    let child = GcHandle::from_tagged(child_ptr);
                    self.collect_node_entries(child, entries, null_val);
                }
            }
        } else if type_id == TYPE_ID_COLLISION_NODE {
            // CollisionNode: kv_array has alternating keys/values
            let count = BuiltInTypes::untag(node.get_field(1)); // CN_FIELD_COUNT = 1
            let kv_array_ptr = node.get_field(2); // CN_FIELD_KV_ARRAY = 2
            let kv_array = GcHandle::from_tagged(kv_array_ptr);

            for i in 0..count {
                let key = kv_array.get_field(i * 2);
                let value = kv_array.get_field(i * 2 + 1);
                entries.push((key, value));
            }
        }
    }

    /// Public method to collect all key-value entries from a PersistentMap.
    /// Used by json-encode builtin for serialization.
    pub fn get_map_entries_for_json(&self, map: GcHandle) -> Vec<(usize, usize)> {
        self.collect_map_entries(map)
    }

    pub fn get_struct_by_id(&self, struct_id: usize) -> Option<&Struct> {
        self.structs.get_by_id(struct_id)
    }

    pub fn property_access(
        &self,
        struct_pointer: usize,
        str_constant_ptr: usize,
    ) -> Result<(usize, usize), Box<dyn Error>> {
        if !BuiltInTypes::untag(struct_pointer).is_multiple_of(8) {
            return Err("Not aligned".into());
        }
        let heap_object = HeapObject::from_tagged(struct_pointer);
        let str_constant_ptr: usize = BuiltInTypes::untag(str_constant_ptr);
        let string_value = &self.string_constants[str_constant_ptr];
        let string = &string_value.str;
        let struct_type_id = heap_object.get_struct_id();
        let struct_value = self.get_struct_by_id(struct_type_id);
        if struct_value.is_none() {
            let untagged = heap_object.untagged();
            let raw_header = unsafe { *(untagged as *const usize) };
            let is_forwarding = Header::is_forwarding_bit_set(raw_header);
            eprintln!(
                "Struct not found! struct_pointer={:#x}, struct_type_id={}, header={:?}, raw_header={:#x}, is_forwarding={}",
                struct_pointer,
                struct_type_id,
                heap_object.get_header(),
                raw_header,
                is_forwarding
            );
            if is_forwarding {
                let forwarded_to = Header::clear_forwarding_bit(raw_header);
                eprintln!("Object was forwarded to {:#x}", forwarded_to);
            }
            panic!(
                "Struct not found by ID {} - this is a fatal error",
                struct_type_id
            );
        }
        let struct_value = struct_value.unwrap();

        // Check the latest definition for field validity (removed fields error immediately)
        let latest_shape_id = self.structs.get_latest_shape_for(struct_type_id);
        if latest_shape_id != struct_type_id {
            let latest_def = self
                .get_struct_by_id(latest_shape_id)
                .expect("Latest shape must exist");
            if !latest_def.fields.iter().any(|f| f == string) {
                let simple_name = latest_def
                    .name
                    .split_once("/")
                    .map(|(_, n)| n)
                    .unwrap_or(&latest_def.name);
                return Err(format!("Field '{}' does not exist on {}", string, simple_name).into());
            }
        }

        // Find field in object's actual layout
        if let Some(field_index) = struct_value.fields.iter().position(|f| f == string) {
            Ok((heap_object.get_field(field_index), field_index))
        } else {
            // Field was added after this object was created — return null
            Ok((BuiltInTypes::null_value() as usize, usize::MAX))
        }
    }

    pub fn type_of(
        &mut self,
        _stack_pointer: usize,
        value: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let tag = BuiltInTypes::get_kind(value);
        let beagle_core_id = self
            .get_namespace_id("beagle.core")
            .ok_or("beagle.core namespace must exist for type-of")?;

        // Map BuiltInTypes to type descriptor binding names
        let type_name = match tag {
            BuiltInTypes::Null => "Null",
            BuiltInTypes::Int => "Int",
            BuiltInTypes::Float => "Float",
            BuiltInTypes::String => "String",
            BuiltInTypes::Bool => "Bool",
            BuiltInTypes::Function => "Function",
            BuiltInTypes::Closure => "Closure",
            BuiltInTypes::HeapObject => {
                // Check HeapObject type_id to distinguish heap-based types
                let heap_object = HeapObject::from_tagged(value);
                let type_id = heap_object.get_type_id();

                // Check if this is a type descriptor (negative id_field at index 1)
                // Type descriptors have struct_id=1 (Struct base type) and id_field < 0
                let fields_size = heap_object.fields_size() / 8;
                if fields_size >= 2 {
                    let id_field = heap_object.get_field(1);
                    if BuiltInTypes::get_kind(id_field) == BuiltInTypes::Int {
                        let id_value = BuiltInTypes::untag_isize(id_field as isize);
                        if id_value < 0 {
                            // This is a type descriptor - return it as-is
                            return Ok(value);
                        }
                    }
                }

                match type_id {
                    val if val == TYPE_ID_RAW_ARRAY as usize => "Array",
                    val if val == TYPE_ID_STRING as usize
                        || val == TYPE_ID_STRING_SLICE as usize
                        || val == TYPE_ID_CONS_STRING as usize =>
                    {
                        "String"
                    }
                    val if val == TYPE_ID_KEYWORD as usize => "Keyword",
                    val if val == TYPE_ID_PERSISTENT_VEC as usize => "PersistentVector",
                    val if val == TYPE_ID_PERSISTENT_MAP as usize => "PersistentMap",
                    val if val == TYPE_ID_PERSISTENT_SET as usize => "PersistentSet",
                    val if val == TYPE_ID_MULTI_ARITY_FUNCTION as usize => "MultiArityFunction",
                    val if val == TYPE_ID_CONTINUATION as usize => "Continuation",
                    val if val == TYPE_ID_CONTINUATION_SEGMENT as usize => "ContinuationSegment",
                    _ => {
                        // Custom struct (type_id == 0 or other) - use struct_id
                        let struct_type_id = heap_object.get_struct_id();
                        let struct_value = self
                            .get_struct_by_id(struct_type_id)
                            .ok_or("Struct type not found")?;
                        let struct_name = struct_value.name.clone();
                        let (namespace_name, struct_name) = struct_name
                            .split_once("/")
                            .ok_or("Struct name must be namespace-qualified")?;
                        let namespace_id = self
                            .get_namespace_id(namespace_name)
                            .ok_or("Namespace for struct not found")?;
                        let slot = self
                            .find_binding(namespace_id, struct_name)
                            .ok_or("Struct binding not found in namespace")?;
                        return Ok(self.get_binding(namespace_id, slot));
                    }
                }
            }
        };

        // Look up the type descriptor binding from beagle.core
        let slot = self
            .find_binding(beagle_core_id, type_name)
            .ok_or_else(|| format!("Type descriptor '{}' not found in beagle.core", type_name))?;
        Ok(self.get_binding(beagle_core_id, slot))
    }

    pub fn equal(&self, a: usize, b: usize) -> bool {
        let mut a = a;
        let mut b = b;
        if a == b {
            return true;
        }
        let mut a_tag = BuiltInTypes::get_kind(a);
        let mut b_tag = BuiltInTypes::get_kind(b);
        // TODO: Make this pluggeable by having a protocol for it
        // so that we can have equality for hashmaps and vectors
        // without custom handling. I can probably do this by just calling
        // this as the default implementation for the protocol
        // I don't have that concept right now.

        if a_tag == BuiltInTypes::String && b_tag == BuiltInTypes::HeapObject {
            (a_tag, b_tag) = (b_tag, a_tag);
            (a, b) = (b, a);
        }

        if a_tag == BuiltInTypes::HeapObject && b_tag == BuiltInTypes::String {
            let a_object = HeapObject::from_tagged(a);
            let a_type = a_object.get_type_id();
            if a_type != TYPE_ID_STRING as usize
                && a_type != TYPE_ID_STRING_SLICE as usize
                && a_type != TYPE_ID_CONS_STRING as usize
            {
                return false;
            }
            let b_string = self.get_str_literal(b);
            let a_bytes = self.get_string_bytes_vec(a);
            let a_string = unsafe { std::str::from_utf8_unchecked(&a_bytes) };
            return a_string == b_string;
        }
        // Handle int/float mixed comparison: convert int to float and compare values
        if (a_tag == BuiltInTypes::Int && b_tag == BuiltInTypes::Float)
            || (a_tag == BuiltInTypes::Float && b_tag == BuiltInTypes::Int)
        {
            let (int_val, float_tagged) = if a_tag == BuiltInTypes::Int {
                (a, b)
            } else {
                (b, a)
            };
            let int_as_f64 = BuiltInTypes::untag(int_val) as i64 as f64;
            let float_ptr = BuiltInTypes::untag(float_tagged) as *const f64;
            let float_val = unsafe { *float_ptr.add(1) };
            return int_as_f64 == float_val;
        }
        if a_tag != b_tag {
            return false;
        }
        match a_tag {
            BuiltInTypes::Null => true,
            BuiltInTypes::Int => a == b,
            BuiltInTypes::Float => {
                // Floats are heap-allocated, so we need to compare the actual values
                let a_ptr = BuiltInTypes::untag(a) as *const f64;
                let b_ptr = BuiltInTypes::untag(b) as *const f64;
                let a_val = unsafe { *a_ptr.add(1) };
                let b_val = unsafe { *b_ptr.add(1) };
                a_val == b_val
            }
            BuiltInTypes::String => a == b,
            BuiltInTypes::Bool => a == b,
            BuiltInTypes::Function => a == b,
            BuiltInTypes::Closure => a == b,
            BuiltInTypes::HeapObject => {
                let a_object = HeapObject::from_tagged(a);
                let b_object = HeapObject::from_tagged(b);

                // Keywords are interned, so compare by pointer identity
                if a_object.get_type_id() == TYPE_ID_KEYWORD as usize
                    && b_object.get_type_id() == TYPE_ID_KEYWORD as usize
                {
                    return a == b;
                }

                // Strings: compare cached hashes first, then byte content
                let a_is_string = matches!(
                    a_object.get_type_id(),
                    x if x == TYPE_ID_STRING as usize
                        || x == TYPE_ID_STRING_SLICE as usize
                        || x == TYPE_ID_CONS_STRING as usize
                );
                let b_is_string = matches!(
                    b_object.get_type_id(),
                    x if x == TYPE_ID_STRING as usize
                        || x == TYPE_ID_STRING_SLICE as usize
                        || x == TYPE_ID_CONS_STRING as usize
                );
                if a_is_string && b_is_string {
                    // If both have cached hashes and they differ, strings are definitely not equal
                    let a_hash = a_object.get_string_hash();
                    let b_hash = b_object.get_string_hash();
                    if a_hash != 0 && b_hash != 0 && a_hash != b_hash {
                        return false;
                    }
                    let a_bytes = self.get_string_bytes_vec(a);
                    let b_bytes = self.get_string_bytes_vec(b);
                    return a_bytes == b_bytes;
                }

                // Different types are not equal
                if a_object.get_type_id() != b_object.get_type_id() {
                    return false;
                }

                // For other HeapObjects (structs, arrays, etc.), compare struct_id and fields
                if a_object.get_struct_id() != b_object.get_struct_id() {
                    return false;
                }
                let a_fields = a_object.get_fields();
                let b_fields = b_object.get_fields();
                if a_fields.len() != b_fields.len() {
                    return false;
                }
                for (a, b) in a_fields.iter().zip(b_fields.iter()) {
                    if !self.equal(*a, *b) {
                        return false;
                    }
                }
                true
            }
        }
    }

    pub fn write_field(
        &self,
        stack_pointer: usize,
        struct_pointer: usize,
        str_constant_ptr: usize,
        value: usize,
    ) -> usize {
        if !BuiltInTypes::untag(struct_pointer).is_multiple_of(8) {
            unsafe {
                crate::builtins::throw_runtime_error(
                    stack_pointer,
                    "TypeError",
                    "Struct pointer not aligned".to_string(),
                );
            }
        }
        let heap_object = HeapObject::from_tagged(struct_pointer);
        let str_constant_ptr: usize = BuiltInTypes::untag(str_constant_ptr);
        let string_value = &self.string_constants[str_constant_ptr];
        let string = &string_value.str;
        let struct_type_id = heap_object.get_struct_id();
        let struct_value = self
            .get_struct_by_id(struct_type_id)
            .expect("Struct not found by ID - this is a fatal error");

        // Check the latest definition for field validity (removed fields error immediately)
        let latest_shape_id = self.structs.get_latest_shape_for(struct_type_id);
        if latest_shape_id != struct_type_id {
            let latest_def = self
                .get_struct_by_id(latest_shape_id)
                .expect("Latest shape must exist");
            if !latest_def.fields.iter().any(|f| f == string) {
                let simple_name = latest_def
                    .name
                    .split_once("/")
                    .map(|(_, n)| n)
                    .unwrap_or(&latest_def.name);
                unsafe {
                    crate::builtins::throw_runtime_error(
                        stack_pointer,
                        "FieldError",
                        format!("Field '{}' does not exist on {}", string, simple_name),
                    );
                }
            }
        }

        // Find field in object's actual layout
        let field_index = struct_value.fields.iter().position(|f| f == string);

        let field_index = match field_index {
            Some(idx) => idx,
            None => {
                // Field was added after this object was created — can't write to it
                unsafe {
                    crate::builtins::throw_runtime_error(
                        stack_pointer,
                        "FieldError",
                        format!(
                            "Cannot write field '{}' on old-layout object (field was added in a later redefinition)",
                            string
                        ),
                    );
                }
            }
        };

        // Check if field is mutable (use latest definition for mutability)
        let mutable_check_def = if latest_shape_id != struct_type_id {
            self.get_struct_by_id(latest_shape_id).unwrap()
        } else {
            struct_value
        };
        let latest_field_index = mutable_check_def.fields.iter().position(|f| f == string);
        if let Some(li) = latest_field_index {
            if !mutable_check_def.is_field_mutable(li) {
                unsafe {
                    crate::builtins::throw_runtime_error(
                        stack_pointer,
                        "MutabilityError",
                        format!(
                            "Cannot mutate immutable field '{}' in struct '{}'",
                            string, mutable_check_def.name
                        ),
                    );
                }
            }
        }

        heap_object.write_field(field_index as i32, value);
        field_index
    }

    pub fn get_struct_repr(
        &self,
        struct_value: &Struct,
        fields: &[usize],
        depth: usize,
    ) -> Option<String> {
        self.get_struct_repr_inner(struct_value, fields, depth, false)
    }

    fn get_struct_repr_inner(
        &self,
        struct_value: &Struct,
        fields: &[usize],
        depth: usize,
        escape: bool,
    ) -> Option<String> {
        // It should look like this
        // struct_name { field1: value1, field2: value2 }
        let simple_name = struct_value
            .name
            .split_once("/")
            .map(|(_, name)| name)
            .unwrap_or(&struct_value.name);
        let mut repr = simple_name.to_string();
        if struct_value.fields.is_empty() {
            return Some(repr);
        }
        repr.push_str(" { ");
        for (index, field) in struct_value.fields.iter().enumerate() {
            repr.push_str(field);
            repr.push_str(": ");
            let value = fields[index];
            repr.push_str(&self.get_repr_inner(value, depth + 1, escape)?);
            if index != struct_value.fields.len() - 1 {
                repr.push_str(", ");
            }
        }
        repr.push_str(" }");
        Some(repr)
    }

    pub fn get_string_literal(&self, value: usize) -> String {
        let value = BuiltInTypes::untag(value);
        self.string_constants[value].str.clone()
    }

    pub fn get_str_literal(&self, value: usize) -> &str {
        let value = BuiltInTypes::untag(value);
        &self.string_constants[value].str
    }

    /// Hash a Beagle value. Returns raw hash (not tagged).
    pub fn hash_value(&self, value: usize) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let tag = BuiltInTypes::get_kind(value);
        match tag {
            BuiltInTypes::Int => {
                let mut s = DefaultHasher::new();
                value.hash(&mut s);
                s.finish() as usize
            }
            BuiltInTypes::HeapObject => {
                let heap_object = HeapObject::from_tagged(value);
                let type_id = heap_object.get_header().type_id;
                if type_id == TYPE_ID_STRING
                    || type_id == TYPE_ID_STRING_SLICE
                    || type_id == TYPE_ID_CONS_STRING
                {
                    let cached = heap_object.get_string_hash();
                    if cached != 0 {
                        return cached as usize;
                    }
                    let bytes = self.get_string_bytes_vec(value);
                    let string = unsafe { std::str::from_utf8_unchecked(&bytes) };
                    let mut s = DefaultHasher::new();
                    string.hash(&mut s);
                    let hash = s.finish();
                    let hash = if hash == 0 { 1 } else { hash };
                    heap_object.set_string_hash(hash);
                    hash as usize
                } else if type_id == TYPE_ID_KEYWORD {
                    // Keyword - use cached hash
                    heap_object.get_keyword_hash() as usize
                } else {
                    // Generic struct - hash all fields
                    let fields = heap_object.get_fields();
                    let mut s = DefaultHasher::new();
                    for field in fields {
                        field.hash(&mut s);
                    }
                    s.finish() as usize
                }
            }
            BuiltInTypes::String => {
                let string = self.get_string_literal(value);
                let mut s = DefaultHasher::new();
                string.hash(&mut s);
                s.finish() as usize
            }
            _ => {
                // For other types, just hash the raw value
                let mut s = DefaultHasher::new();
                value.hash(&mut s);
                s.finish() as usize
            }
        }
    }

    pub fn get_string(&self, stack_pointer: usize, value: usize) -> String {
        let tag = BuiltInTypes::get_kind(value);
        if tag == BuiltInTypes::String {
            self.get_string_literal(value)
        } else if tag == BuiltInTypes::HeapObject {
            let heap_object = HeapObject::from_tagged(value);
            let tid = heap_object.get_type_id();
            if tid != TYPE_ID_STRING as usize
                && tid != TYPE_ID_STRING_SLICE as usize
                && tid != TYPE_ID_CONS_STRING as usize
            {
                unsafe {
                    crate::builtins::throw_runtime_error(
                        stack_pointer,
                        "TypeError",
                        format!("Expected string, got heap object with type_id {}", tid),
                    );
                }
            }
            let bytes = self.get_string_bytes_vec(value);
            unsafe { String::from_utf8_unchecked(bytes) }
        } else {
            unsafe {
                crate::builtins::throw_runtime_error(
                    stack_pointer,
                    "TypeError",
                    format!("Expected string, got {:?}", tag),
                );
            }
        }
    }

    /// Extract a substring efficiently from a `&str` without copying the source.
    /// ASCII fast path: O(length). Unicode path: O(start + length).
    fn compute_substring(
        stack_pointer: usize,
        s: &str,
        start: usize,
        length: usize,
        is_ascii: bool,
    ) -> String {
        if length == 0 {
            return String::new();
        }

        let end = start + length;

        // Fast path for ASCII strings: byte index == char index
        if is_ascii {
            if end > s.len() {
                unsafe {
                    crate::builtins::throw_runtime_error(
                        stack_pointer,
                        "IndexError",
                        format!(
                            "substring index out of bounds: start={}, length={}, but string length is {}",
                            start,
                            length,
                            s.len()
                        ),
                    );
                }
            }
            return s[start..end].to_string();
        }

        // Unicode path: scan to start, then continue for length chars
        let mut char_iter = s.char_indices();

        let byte_start = match char_iter.nth(start) {
            Some((i, _)) => i,
            None => unsafe {
                crate::builtins::throw_runtime_error(
                    stack_pointer,
                    "IndexError",
                    format!(
                        "substring index out of bounds: start={}, length={}, but string is shorter",
                        start, length,
                    ),
                );
            },
        };

        // Continue from current position for length-1 more chars to find end
        // (nth(start) already consumed the char at `start`, so we need length-1 more)
        let byte_end = if length == 1 {
            byte_start + s[byte_start..].chars().next().unwrap().len_utf8()
        } else {
            match char_iter.nth(length - 2) {
                Some((i, c)) => i + c.len_utf8(),
                None => unsafe {
                    crate::builtins::throw_runtime_error(
                        stack_pointer,
                        "IndexError",
                        format!(
                            "substring index out of bounds: start={}, length={}, but string is shorter",
                            start, length,
                        ),
                    );
                },
            }
        };

        s[byte_start..byte_end].to_string()
    }

    pub fn get_substring(
        &mut self,
        stack_pointer: usize,
        string: usize,
        start: usize,
        length: usize,
    ) -> Result<Tagged, Box<dyn Error>> {
        let tag = BuiltInTypes::get_kind(string);

        if tag == BuiltInTypes::HeapObject {
            let heap_object = HeapObject::from_tagged(string);
            let type_id = heap_object.get_type_id();
            if type_id != TYPE_ID_STRING as usize
                && type_id != TYPE_ID_STRING_SLICE as usize
                && type_id != TYPE_ID_CONS_STRING as usize
            {
                unsafe {
                    crate::builtins::throw_runtime_error(
                        stack_pointer,
                        "TypeError",
                        format!(
                            "Expected string for substring, got heap object with type_id {}",
                            type_id
                        ),
                    );
                }
            }

            // Cons strings: flatten first, then take substring
            if type_id == TYPE_ID_CONS_STRING as usize {
                let bytes = self.get_string_bytes_vec(string);
                let s = unsafe { std::str::from_utf8_unchecked(&bytes) };
                let is_ascii = heap_object.get_header().type_flags & 1 != 0;
                let result = Self::compute_substring(stack_pointer, s, start, length, is_ascii);
                return self.allocate_string(stack_pointer, result);
            }

            let is_ascii = if type_id == TYPE_ID_STRING_SLICE as usize {
                let parent = HeapObject::from_tagged(heap_object.get_field(0));
                parent.get_header().type_flags & 1 != 0
            } else {
                heap_object.get_header().type_flags & 1 != 0
            };

            if is_ascii {
                let str_bytes = heap_object.get_string_bytes();
                let end = start + length;
                if end > str_bytes.len() {
                    unsafe {
                        crate::builtins::throw_runtime_error(
                            stack_pointer,
                            "IndexError",
                            format!(
                                "substring index out of bounds: start={}, length={}, but string length is {}",
                                start,
                                length,
                                str_bytes.len()
                            ),
                        );
                    }
                }

                let (parent_ptr, byte_offset) = if type_id == TYPE_ID_STRING_SLICE as usize {
                    let parent_ptr = heap_object.get_field(0);
                    let parent_offset = BuiltInTypes::untag(heap_object.get_field(1));
                    (parent_ptr, parent_offset + start)
                } else {
                    (string, start)
                };

                return self.allocate_string_slice(stack_pointer, parent_ptr, byte_offset, length);
            }
            let bytes = heap_object.get_string_bytes();
            let s = unsafe { std::str::from_utf8_unchecked(bytes) };
            let result = Self::compute_substring(stack_pointer, s, start, length, false);
            return self.allocate_string(stack_pointer, result);
        }

        if tag == BuiltInTypes::String {
            // String literals can't be sliced (not on GC heap), so copy
            let s = self.get_str_literal(string);
            let is_ascii = s.is_ascii();
            if is_ascii {
                let end = start + length;
                if end > s.len() {
                    unsafe {
                        crate::builtins::throw_runtime_error(
                            stack_pointer,
                            "IndexError",
                            format!(
                                "substring index out of bounds: start={}, length={}, but string length is {}",
                                start,
                                length,
                                s.len()
                            ),
                        );
                    }
                }
                let slice = s.as_bytes()[start..end].to_vec();
                return self.allocate_string_from_bytes(stack_pointer, &slice);
            }
            let result = Self::compute_substring(stack_pointer, s, start, length, false);
            return self.allocate_string(stack_pointer, result);
        }

        unsafe {
            crate::builtins::throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!("Expected string for substring, got {:?}", tag),
            );
        }
    }

    pub fn add_foreign_function(
        &mut self,
        name: Option<&str>,
        function: *const u8,
        number_of_args: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let index = self.functions.len();
        let pointer = function;
        let jump_table_offset = self.add_jump_table_entry(index, pointer)?;
        self.functions.push(Function {
            name: name.unwrap_or("<Anonymous>").to_string(),
            pointer: pointer.into(),
            jump_table_offset,
            is_foreign: true,
            is_builtin: false,
            needs_stack_pointer: false,
            needs_frame_pointer: false,
            is_defined: true,
            number_of_locals: 0,
            size: 0,
            number_of_args,
            is_variadic: false,
            min_args: number_of_args,
            docstring: None,
            arg_names: vec![],
            source_file: None,
            source_line: None,
            source_column: None,
        });
        debugger(Message {
            kind: "foreign_function".to_string(),
            data: Data::ForeignFunction {
                name: name.unwrap_or("<Anonymous>").to_string(),
                pointer: Self::get_function_pointer(
                    self,
                    self.functions
                        .last()
                        .expect("No functions in function table - this is a fatal error")
                        .clone(),
                )
                .expect("Failed to get function pointer - this is a fatal error")
                    as usize,
            },
        });
        let function_pointer = Self::get_function_pointer(
            self,
            self.functions
                .last()
                .expect("No functions in function table - this is a fatal error")
                .clone(),
        )
        .expect("Failed to get function pointer - this is a fatal error");
        Ok(function_pointer as usize)
    }

    pub fn add_builtin_function(
        &mut self,
        name: &str,
        function: *const u8,
        needs_stack_pointer: bool,
        number_of_args: usize,
    ) -> Result<usize, Box<dyn Error>> {
        // When needs_stack_pointer is true, we also need frame_pointer for GC stack walking.
        // The frame_pointer is passed as the second implicit argument (after stack_pointer).
        // We need to add 1 to number_of_args to account for the frame_pointer.
        let adjusted_args = if needs_stack_pointer {
            number_of_args + 1
        } else {
            number_of_args
        };
        self.add_builtin_function_with_fp(
            name,
            function,
            needs_stack_pointer,
            needs_stack_pointer,
            adjusted_args,
        )
    }

    pub fn add_builtin_function_with_fp(
        &mut self,
        name: &str,
        function: *const u8,
        needs_stack_pointer: bool,
        needs_frame_pointer: bool,
        number_of_args: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let index = self.functions.len();
        let pointer = function;

        let jump_table_offset = self.add_jump_table_entry(index, pointer)?;
        self.functions.push(Function {
            name: name.to_string(),
            pointer: pointer.into(),
            jump_table_offset,
            is_foreign: true,
            is_builtin: true,
            needs_stack_pointer,
            needs_frame_pointer,
            is_defined: true,
            number_of_locals: 0,
            size: 0,
            number_of_args,
            is_variadic: false,
            min_args: number_of_args,
            docstring: None,
            arg_names: vec![],
            source_file: None,
            source_line: None,
            source_column: None,
        });
        let pointer = Self::get_function_pointer(
            self,
            self.functions
                .last()
                .expect("No functions in function table - this is a fatal error")
                .clone(),
        )
        .expect("Failed to get function pointer - this is a fatal error");
        // self.namespaces.add_binding(name, pointer as usize);
        debugger(Message {
            kind: "builtin_function".to_string(),
            data: Data::BuiltinFunction {
                name: name.to_string(),
                pointer: pointer as usize,
            },
        });
        Ok(self.functions.len() - 1)
    }

    /// Add a builtin function with documentation.
    /// This is the preferred way to register builtins as it includes docstrings and argument names.
    pub fn add_builtin_with_doc(
        &mut self,
        name: &str,
        function: *const u8,
        needs_stack_pointer: bool,
        arg_names: &[&str],
        docstring: &str,
    ) -> Result<usize, Box<dyn Error>> {
        // When needs_stack_pointer is true, we also need frame_pointer for GC stack walking.
        let adjusted_args = if needs_stack_pointer {
            arg_names.len() + 2 // +2 for stack_pointer and frame_pointer
        } else {
            arg_names.len()
        };
        let index = self.functions.len();
        let pointer = function;

        let jump_table_offset = self.add_jump_table_entry(index, pointer)?;
        self.functions.push(Function {
            name: name.to_string(),
            pointer: pointer.into(),
            jump_table_offset,
            is_foreign: true,
            is_builtin: true,
            needs_stack_pointer,
            needs_frame_pointer: needs_stack_pointer,
            is_defined: true,
            number_of_locals: 0,
            size: 0,
            number_of_args: adjusted_args,
            is_variadic: false,
            min_args: adjusted_args,
            docstring: Some(docstring.to_string()),
            arg_names: arg_names.iter().map(|s| s.to_string()).collect(),
            source_file: None,
            source_line: None,
            source_column: None,
        });
        let pointer = Self::get_function_pointer(
            self,
            self.functions
                .last()
                .expect("No functions in function table - this is a fatal error")
                .clone(),
        )
        .expect("Failed to get function pointer - this is a fatal error");
        debugger(Message {
            kind: "builtin_function".to_string(),
            data: Data::BuiltinFunction {
                name: name.to_string(),
                pointer: pointer as usize,
            },
        });
        Ok(self.functions.len() - 1)
    }

    pub fn update_stack_map_information(
        &mut self,
        stack_map: Vec<(usize, StackMapDetails)>,
        function_pointer: usize,
        name: Option<&str>,
    ) {
        debugger(Message {
            kind: "stack_map".to_string(),
            data: Data::StackMap {
                pc: function_pointer,
                name: name.unwrap_or("<Anonymous>").to_string(),
                stack_map: stack_map.clone(),
            },
        });
        self.memory.stack_map.extend(stack_map);
    }

    pub fn replace_function_binding(
        &mut self,
        name: String,
        pointer: usize,
    ) -> Result<(), Box<dyn Error>> {
        let untagged_pointer = BuiltInTypes::untag(pointer) as *const u8;
        let existing_function = self
            .get_function_by_pointer(untagged_pointer)
            .ok_or_else(|| format!("Function not found for pointer {:p}", untagged_pointer))?;
        let existing_function = existing_function.clone();
        for (index, function) in self.functions.iter_mut().enumerate() {
            if function.name == name {
                self.overwrite_function(index, untagged_pointer, existing_function.size)?;
                return Ok(());
            }
        }
        Err(format!("Function {} not found in function table", name).into())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsert_function(
        &mut self,
        name: Option<&str>,
        pointer: *const u8,
        size: usize,
        number_of_locals: usize,
        stack_map: Vec<(usize, StackMapDetails)>,
        number_of_args: usize,
        is_variadic: bool,
        min_args: usize,
        docstring: Option<String>,
        arg_names: Vec<String>,
        source_file: Option<String>,
        source_line: Option<usize>,
    ) -> Result<usize, Box<dyn Error>> {
        let mut already_defined = false;
        let mut function_pointer = 0;
        if let Some(n) = name {
            for (index, function) in self.functions.iter_mut().enumerate() {
                if function.name == n {
                    function_pointer = self.overwrite_function(index, pointer, size)?;
                    // Update variadic info on existing function
                    self.functions[index].is_variadic = is_variadic;
                    self.functions[index].min_args = min_args;
                    // Update docstring and arg_names if provided
                    if docstring.is_some() {
                        self.functions[index].docstring = docstring.clone();
                    }
                    if !arg_names.is_empty() {
                        self.functions[index].arg_names = arg_names.clone();
                    }
                    if source_file.is_some() {
                        self.functions[index].source_file = source_file.clone();
                    }
                    if source_line.is_some() {
                        self.functions[index].source_line = source_line;
                    }
                    // Tag and add the binding for reserved functions that are now being defined
                    let tagged_fn = BuiltInTypes::Function.tag(function_pointer as isize) as usize;
                    self.add_binding(n, tagged_fn);
                    already_defined = true;
                    break;
                }
            }
        }
        if !already_defined {
            function_pointer = self.add_function(
                name,
                pointer,
                size,
                number_of_locals,
                number_of_args,
                is_variadic,
                min_args,
                docstring,
                arg_names,
                source_file,
                source_line,
            )?;
        }
        assert!(function_pointer != 0);

        self.update_stack_map_information(stack_map, function_pointer, name);

        debugger(Message {
            kind: "user_function".to_string(),
            data: Data::UserFunction {
                name: name.unwrap_or("<Anonymous>").to_string(),
                pointer: function_pointer,
                len: size,
                number_of_arguments: number_of_args,
            },
        });
        Ok(function_pointer)
    }

    pub fn reserve_function(
        &mut self,
        name: &str,
        number_of_args: usize,
        is_variadic: bool,
        min_args: usize,
    ) -> Result<Function, Box<dyn Error>> {
        for function in self.functions.iter_mut() {
            if function.name == name {
                function.number_of_args = number_of_args;
                function.is_variadic = is_variadic;
                function.min_args = min_args;
                return Ok(function.clone());
            }
        }
        let index = self.functions.len();
        let jump_table_offset = self.add_jump_table_entry(index, std::ptr::null())?;
        let function = Function {
            name: name.to_string(),
            pointer: RawPtr::new(std::ptr::null()),
            jump_table_offset,
            is_foreign: false,
            is_builtin: false,
            needs_stack_pointer: false,
            needs_frame_pointer: false,
            is_defined: false,
            number_of_locals: 0,
            size: 0,
            number_of_args,
            is_variadic,
            min_args,
            docstring: None,
            arg_names: vec![],
            source_file: None,
            source_line: None,
            source_column: None,
        };
        self.functions.push(function.clone());
        Ok(function)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_function(
        &mut self,
        name: Option<&str>,
        pointer: *const u8,
        size: usize,
        number_of_locals: usize,
        number_of_args: usize,
        is_variadic: bool,
        min_args: usize,
        docstring: Option<String>,
        arg_names: Vec<String>,
        source_file: Option<String>,
        source_line: Option<usize>,
    ) -> Result<usize, Box<dyn Error>> {
        let index = self.functions.len();
        self.functions.push(Function {
            name: name.unwrap_or("<Anonymous>").to_string(),
            pointer: pointer.into(),
            jump_table_offset: 0,
            is_foreign: false,
            is_builtin: false,
            needs_stack_pointer: false,
            needs_frame_pointer: false,
            is_defined: true,
            number_of_locals,
            size,
            number_of_args,
            is_variadic,
            min_args,
            docstring,
            arg_names,
            source_file,
            source_line,
            source_column: None,
        });
        let function_pointer = Self::get_function_pointer(
            self,
            self.functions
                .last()
                .expect("No functions in function table - this is a fatal error")
                .clone(),
        )
        .expect("Failed to get function pointer - this is a fatal error");
        let jump_table_offset = self.add_jump_table_entry(index, function_pointer)?;

        self.functions[index].jump_table_offset = jump_table_offset;
        if let Some(name) = name {
            // Tag the function pointer so dynamic calls can identify it as a Function
            let tagged_fn = BuiltInTypes::Function.tag(function_pointer as isize) as usize;
            self.add_binding(name, tagged_fn);
        }
        Ok(function_pointer as usize)
    }

    pub fn overwrite_function(
        &mut self,
        index: usize,
        pointer: *const u8,
        size: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let function = &mut self.functions[index];
        function.pointer = pointer.into();
        let jump_table_offset = function.jump_table_offset;
        let function_clone = function.clone();
        let function_pointer = self
            .get_function_pointer(function_clone)
            .expect("Failed to get function pointer - this is a fatal error");
        self.modify_jump_table_entry(jump_table_offset, function_pointer as usize)?;
        let function = &mut self.functions[index];
        function.size = size;
        function.is_defined = true;
        Ok(function_pointer as usize)
    }

    pub fn get_pointer(&self, function: &Function) -> Result<*const u8, Box<dyn Error>> {
        Ok(function.pointer.into())
    }

    pub fn get_function_pointer(&self, function: Function) -> Result<*const u8, Box<dyn Error>> {
        // Gets the absolute pointer to a function
        // if it is a foreign function, return the offset
        // if it is a local function, return the offset + the start of code_memory
        Ok(function.pointer.into())
    }

    pub fn get_jump_table_pointer(&self, function: Function) -> Result<usize, Box<dyn Error>> {
        if self.jump_table_base_ptr == 0 {
            return Err("Jump table not initialized".into());
        }
        Ok(function.jump_table_offset * 8 + self.jump_table_base_ptr)
    }

    pub fn add_jump_table_entry(
        &mut self,
        _index: usize,
        pointer: *const u8,
    ) -> Result<usize, Box<dyn Error>> {
        let jump_table_offset = self.jump_table_offset;
        let page_size = MmapOptions::page_size();
        let byte_offset = jump_table_offset * 8;
        let page_index = byte_offset / page_size;
        let offset_in_page = byte_offset % page_size;

        // Ensure we have enough pages committed
        while self.jump_table_pages.len() <= page_index {
            // Split a new page from reserved space and commit it
            let reserved = self
                .jump_table_reserved
                .split_to(page_size)
                .map_err(|e| format!("Failed to split jump table page: {:?}", e))?;

            let mut new_page = reserved.make_mut().map_err(|(_reserved, e)| {
                format!("Failed to make jump table page mutable: {:?}", e)
            })?;

            // Zero out the new page
            unsafe {
                std::ptr::write_bytes(new_page.as_mut_ptr(), 0, page_size);
            }

            let page_reserved = new_page.make_read_only().map_err(|(_page, e)| {
                format!("Failed to make jump table page read-only: {:?}", e)
            })?;

            let page: Mmap = page_reserved
                .try_into()
                .map_err(|e| format!("Failed to convert Reserved to Mmap: {:?}", e))?;

            self.jump_table_pages.push(page);
        }

        // Get the page, make it mutable, write to it, make it read-only again
        let page = self.jump_table_pages.remove(page_index);
        let mut page_mut = page.make_mut().map_err(|(_, e)| {
            format!(
                "Failed to make jump table page mutable for writing: {:?}",
                e
            )
        })?;

        let tagged_pointer = BuiltInTypes::Function.tag(pointer as isize) as usize;
        let bytes = tagged_pointer.to_le_bytes();

        page_mut[offset_in_page..offset_in_page + 8].copy_from_slice(&bytes);

        let page_reserved = page_mut.make_read_only().map_err(|(_, e)| {
            format!(
                "Failed to make jump table page read-only after writing: {:?}",
                e
            )
        })?;

        let page: Mmap = page_reserved;

        self.jump_table_pages.insert(page_index, page);

        self.jump_table_offset += 1;
        Ok(jump_table_offset)
    }

    pub fn modify_jump_table_entry(
        &mut self,
        jump_table_offset: usize,
        function_pointer: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let page_size = MmapOptions::page_size();
        let byte_offset = jump_table_offset * 8;
        let page_index = byte_offset / page_size;
        let offset_in_page = byte_offset % page_size;

        if page_index >= self.jump_table_pages.len() {
            return Err("Jump table entry does not exist".into());
        }

        // Get the page, make it mutable, write to it, make it read-only again
        let page = self.jump_table_pages.remove(page_index);
        let mut page_mut = page.make_mut().map_err(|(_, e)| {
            format!(
                "Failed to make jump table page mutable for modification: {}",
                e
            )
        })?;

        let tagged_pointer = BuiltInTypes::Function.tag(function_pointer as isize) as usize;
        let bytes = tagged_pointer.to_le_bytes();

        page_mut[offset_in_page..offset_in_page + 8].copy_from_slice(&bytes);

        let page_reserved = page_mut.make_read_only().map_err(|(_, e)| {
            format!(
                "Failed to make jump table page read-only after modification: {:?}",
                e
            )
        })?;

        let page: Mmap = page_reserved;

        self.jump_table_pages.insert(page_index, page);

        Ok(jump_table_offset)
    }

    pub fn find_function(&self, name: &str) -> Option<Function> {
        assert!(
            name.contains("/"),
            "Function name should contain /: {:?}",
            name
        );
        self.functions.iter().find(|f| f.name == name).cloned()
    }

    pub fn get_function_by_pointer(&self, value: *const u8) -> Option<&Function> {
        self.functions.iter().find(|f| f.pointer == value.into())
    }

    pub fn get_function_by_pointer_mut(&mut self, value: *const u8) -> Option<&mut Function> {
        self.functions
            .iter_mut()
            .find(|f| f.pointer == value.into())
    }

    pub fn check_functions(&self) -> Result<(), Box<dyn Error>> {
        let undefined_functions: Vec<&Function> =
            self.functions.iter().filter(|f| !f.is_defined).collect();
        if !undefined_functions.is_empty() {
            return Err(format!(
                "Undefined functions: {:?} only have functions {:#?}",
                undefined_functions
                    .iter()
                    .map(|f| f.name.clone())
                    .collect::<Vec<String>>(),
                self.functions
                    .iter()
                    .map(|f| f.name.clone())
                    .collect::<Vec<String>>()
            )
            .into());
        }
        Ok(())
    }

    pub fn get_function_by_name(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }

    pub fn get_function_by_name_mut(&mut self, name: &str) -> Option<&mut Function> {
        self.functions.iter_mut().find(|f| f.name == name)
    }

    pub fn add_function_mark_executable(
        &self,
        name: String,
        code: &[u8],
        number_of_locals: i32,
        number_of_args: usize,
    ) -> Result<usize, Box<dyn Error>> {
        self.compiler_channel
            .as_ref()
            .expect("Compiler channel not initialized - this is a fatal error")
            .send(CompilerMessage::AddFunctionMarkExecutable(
                name.clone(),
                code.to_vec(),
                number_of_locals as usize,
                number_of_args,
            ));
        Ok(self
            .get_function_by_name(&name)
            .expect("Function not found after compilation - this is a fatal error")
            .pointer
            .into())
    }

    fn add_binding(&mut self, name: &str, function_pointer: usize) -> usize {
        // Add to Rust-side namespace first to get the slot
        let slot = self.namespaces.add_binding(name, function_pointer);
        let namespace_id = self.namespaces.current_namespace;

        // Store in heap-based PersistentMap if we have a stack,
        // otherwise queue for later processing
        if let Some(stack_pointer) = self.try_get_stack_base() {
            if let Err(e) =
                self.set_heap_binding(stack_pointer, namespace_id, slot, function_pointer)
            {
                eprintln!("Warning: failed to set heap binding for {}: {}", name, e);
            }
        } else {
            // Queue for later - will be flushed when a thread with a stack accesses bindings
            self.pending_heap_bindings
                .lock()
                .expect("Failed to lock pending_heap_bindings")
                .push((namespace_id, slot, function_pointer));
        }

        slot
    }

    pub fn get_trampoline(&self) -> fn(u64, u64) -> u64 {
        let trampoline = self
            .get_function_by_name("trampoline")
            .expect("Trampoline function not found - this is a fatal error");

        unsafe { std::mem::transmute(trampoline.pointer) }
    }

    /// Call a JIT-compiled function via the trampoline.
    /// This switches to the Beagle stack and saves/restores callee-saved registers,
    /// which is required for correctness when calling JIT code from Rust.
    pub fn call_via_trampoline(&self, fn_ptr: usize) -> usize {
        let trampoline = self.get_trampoline();
        let stack_pointer = self.get_stack_base();
        let global_block_ptr = self.get_global_block_ptr();
        unsafe {
            *((stack_pointer - 8) as *mut usize) = global_block_ptr;
        }
        trampoline(stack_pointer as u64, fn_ptr as u64) as usize
    }

    /// Return a tagged string literal for a single ASCII byte.
    /// Lazily interns the character on first use. O(1), no heap allocation.
    #[inline]
    pub fn get_ascii_char_literal(&mut self, byte: u8) -> usize {
        debug_assert!(byte < 128);
        let cached = self.ascii_char_cache[byte as usize];
        if cached != 0 {
            return cached;
        }
        let ch = byte as char;
        let index = self.add_string(StringValue {
            str: ch.to_string(),
        });
        let tagged = BuiltInTypes::String.tag(index as isize) as usize;
        self.ascii_char_cache[byte as usize] = tagged;
        tagged
    }

    pub fn add_string(&mut self, string_value: StringValue) -> usize {
        if let Some(index) = self
            .string_constants
            .iter()
            .position(|s| s.str == string_value.str)
        {
            return index;
        }
        self.string_constants.push(string_value);
        self.string_constants.len() - 1
    }

    pub fn add_keyword(&mut self, keyword_text: String) -> usize {
        if let Some(index) = self
            .keyword_constants
            .iter()
            .position(|k| k.str == keyword_text)
        {
            return index;
        }
        self.keyword_constants
            .push(StringValue { str: keyword_text });
        self.keyword_heap_ptrs.push(None);
        self.keyword_constants.len() - 1
    }

    pub fn add_struct(&mut self, s: Struct) -> bool {
        let name = s.name.clone();
        // Grab the old shape_id before insert (if this is a redefinition)
        let old_shape_id = self.structs.get(&name).map(|(id, _)| id);

        let (new_shape_id, is_redefinition) = self.structs.insert(name.clone(), s);

        if is_redefinition {
            if let Some(old_id) = old_shape_id {
                // Copy protocol dispatch entries from old shape to new shape
                for table in self.dispatch_tables.values_mut() {
                    if old_id < table.struct_dispatch.len() && table.struct_dispatch[old_id] != 0 {
                        let fn_ptr = table.struct_dispatch[old_id];
                        table.register(new_shape_id as isize, fn_ptr);
                    }
                }
            }
        }

        // grab the simple name for the binding
        let (_, simple_name) = name
            .split_once("/")
            .expect("Struct name must contain / separator - this is a fatal error");
        self.add_binding(simple_name, 0);

        is_redefinition
    }

    pub fn add_enum(&mut self, e: Enum) {
        let name = e.name.clone();
        self.enums.insert(e);
        // See comment about structs above
        self.add_binding(&name, 0);
    }

    pub fn get_enum(&self, name: &str) -> Option<&Enum> {
        self.enums.get(name)
    }

    pub fn get_struct(&self, name: &str) -> Option<(usize, &Struct)> {
        self.structs.get(name)
    }

    /// Register a mapping from enum variant struct_id to enum name
    /// Called when enum variants are compiled
    pub fn register_enum_variant(&mut self, struct_id: usize, enum_name: String) {
        self.variant_to_enum.insert(struct_id, enum_name);
    }

    /// Get the enum name for a given struct_id (type_id from header)
    /// Returns None if this isn't an enum variant
    pub fn get_enum_name_for_variant(&self, struct_id: usize) -> Option<&String> {
        self.variant_to_enum.get(&struct_id)
    }

    pub fn get_namespace_from_alias(&self, alias: &str) -> Option<String> {
        let current_namespace = self.current_namespace_name();
        let current_namespace = self
            .get_namespace_id(current_namespace.as_str())
            .expect("Current namespace ID not found - this is a fatal error");
        let current_namespace = self
            .namespaces
            .namespaces
            .get(current_namespace)
            .expect("Current namespace not found in namespaces - this is a fatal error");
        let current_namespace = current_namespace
            .lock()
            .expect("Failed to lock current namespace - this is a fatal error");
        let namespace_id = current_namespace.aliases.get(alias)?;
        let namespace = self
            .namespaces
            .namespaces
            .get(*namespace_id)
            .expect("Alias target namespace not found - this is a fatal error");
        let namespace = namespace
            .lock()
            .expect("Failed to lock alias target namespace - this is a fatal error");
        Some(namespace.name.clone())
    }

    pub fn get_pointer_for_function(&self, function: &Function) -> Option<usize> {
        Some(function.pointer.into())
    }

    pub fn set_compiler_channel(
        &mut self,
        compiler_channel: BlockingSender<CompilerMessage, CompilerResponse>,
    ) {
        self.compiler_channel = Some(compiler_channel);
    }

    pub fn set_pause_atom_ptr(&self, pause_atom_ptr: usize) {
        self.compiler_channel
            .as_ref()
            .expect("Compiler channel not initialized - this is a fatal error")
            .send(CompilerMessage::SetPauseAtomPointer(pause_atom_ptr));
    }

    pub fn add_protocol_info(
        &mut self,
        protocol_name: &str,
        struct_name: &str,
        method_name: &str,
        f: usize,
    ) {
        // TODO: When do I deal with duplicates?
        self.protocol_info
            .entry(protocol_name.to_string())
            .or_default()
            .push(ProtocolMethodInfo {
                _type: struct_name.to_string(),
                method_name: method_name.to_string(),
                fn_pointer: f,
            });

        // Also update the dispatch table for O(1) lookup
        // Key format: "protocol_namespace/method_name" (e.g., "myapp/dispatch")
        let (protocol_namespace, _protocol_name) =
            protocol_name.split_once("/").unwrap_or(("", protocol_name));
        let dispatch_key = format!("{}/{}", protocol_namespace, method_name);

        // f is already a tagged function pointer from Beagle code
        // (tagged with Function tag 0b100 = 4)

        // Check for built-in types first
        // Built-in heap types use primitive dispatch with their type_id
        // Tagged primitives use primitive dispatch with tag + 16
        let builtin_primitive_id = match struct_name {
            "beagle.core/Array" => Some(1),             // type_id for Array
            "beagle.core/String" => Some(2),            // type_id for String
            "beagle.core/Keyword" => Some(3),           // type_id for Keyword
            "beagle.core/Int" => Some(16),              // tag 0 + 16
            "beagle.core/Float" => Some(17),            // tag 1 + 16
            "beagle.core/Bool" => Some(19),             // tag 3 + 16
            "beagle.core/PersistentVector" => Some(20), // TYPE_ID_PERSISTENT_VEC
            "beagle.core/PersistentMap" => Some(22),    // TYPE_ID_PERSISTENT_MAP
            "beagle.core/PersistentSet" => Some(28),    // TYPE_ID_PERSISTENT_SET
            "beagle.core/Regex" => Some(30),            // TYPE_ID_REGEX
            _ => None,
        };

        if let Some(primitive_id) = builtin_primitive_id {
            // Built-in type - register in primitive dispatch
            let dispatch_table = self
                .dispatch_tables
                .entry(dispatch_key)
                .or_insert_with(|| Box::new(DispatchTable::new(None)));
            // Use negative ID for primitive types (DispatchTable converts to index)
            dispatch_table.register(-(primitive_id as isize + 1), f);
        } else if let Some((struct_id, _)) = self.structs.get(struct_name) {
            // Custom struct - register with struct_id
            let dispatch_table = self
                .dispatch_tables
                .entry(dispatch_key)
                .or_insert_with(|| Box::new(DispatchTable::new(None)));
            dispatch_table.register(struct_id as isize, f);
        }
        // Note: silently ignore unknown types for now (could be forward references)
    }

    /// Get the dispatch table pointer for a protocol method
    /// Returns the raw pointer to the DispatchTable struct for use in generated code
    pub fn get_dispatch_table_ptr(&self, protocol_name: &str, method_name: &str) -> Option<usize> {
        let (protocol_namespace, _protocol_name) =
            protocol_name.split_once("/").unwrap_or(("", protocol_name));
        let dispatch_key = format!("{}/{}", protocol_namespace, method_name);

        self.dispatch_tables
            .get(&dispatch_key)
            .map(|table| &**table as *const DispatchTable as usize)
    }

    /// Get a reference to a dispatch table
    pub fn get_dispatch_table(
        &self,
        protocol_name: &str,
        method_name: &str,
    ) -> Option<&DispatchTable> {
        let (protocol_namespace, _protocol_name) =
            protocol_name.split_once("/").unwrap_or(("", protocol_name));
        let dispatch_key = format!("{}/{}", protocol_namespace, method_name);

        self.dispatch_tables.get(&dispatch_key).map(|b| &**b)
    }

    /// Set the default method for a dispatch table
    pub fn set_dispatch_table_default(
        &mut self,
        protocol_name: &str,
        method_name: &str,
        default_fn_ptr: usize,
    ) {
        let (protocol_namespace, _) = protocol_name.split_once("/").unwrap_or(("", protocol_name));
        let dispatch_key = format!("{}/{}", protocol_namespace, method_name);

        if let Some(table) = self.dispatch_tables.get_mut(&dispatch_key) {
            table.set_default_method(default_fn_ptr);
        }
    }

    pub fn compile_protocol_method(&self, protocol_name: &str, method_name: &str) -> usize {
        let protocol_info = self
            .protocol_info
            .get(protocol_name)
            .expect("Protocol not found - this is a fatal error");
        let method_info: Vec<ProtocolMethodInfo> = protocol_info
            .iter()
            .filter(|m| m.method_name == method_name)
            .cloned()
            .collect();
        self.compiler_channel
            .as_ref()
            .expect("Compiler channel not initialized - this is a fatal error")
            .send(CompilerMessage::CompileProtocolMethod(
                protocol_name.to_string(),
                method_name.to_string(),
                method_info,
            ));

        0
    }

    pub fn resolve(&self, struct_name: String) -> String {
        if !struct_name.contains("/") {
            let current_namespace_name = self.current_namespace_name();
            let current_namespace_id = self
                .get_namespace_id(current_namespace_name.as_str())
                .expect("Current namespace does not exist - this is a fatal error");
            let find_binding = self.find_binding(current_namespace_id, struct_name.as_str());
            if find_binding.is_some() {
                return format!("{}/{}", current_namespace_name, struct_name);
            }
            if let Some(beagle_core) = self.get_namespace_id("beagle.core") {
                let find_binding = self.find_binding(beagle_core, struct_name.as_str());
                if find_binding.is_some() {
                    return format!("beagle.core/{}", struct_name);
                }
            }
            eprintln!(
                "Warning: Cannot resolve struct '{}', current_ns={}, using as-is",
                struct_name, current_namespace_name
            );
            return struct_name.to_string();
        }

        let (namespace_or_alias, struct_name) = match struct_name.split_once("/") {
            Some(parts) => parts,
            None => {
                eprintln!("Warning: Invalid struct name format {}", struct_name);
                return struct_name.to_string();
            }
        };

        let namespace_from_alias = self.get_namespace_from_alias(namespace_or_alias);
        if let Some(namespace) = namespace_from_alias {
            return format!("{}/{}", namespace, struct_name);
        }
        let namespace_id = self.get_namespace_id(namespace_or_alias);
        if namespace_id.is_some() {
            return format!("{}/{}", namespace_or_alias, struct_name);
        }
        eprintln!(
            "Warning: Cannot resolve struct {}, using as-is",
            struct_name
        );
        format!("{}/{}", namespace_or_alias, struct_name)
    }

    pub fn get_command_line_args(&self) -> &CommandLineArguments {
        &self.command_line_arguments
    }

    pub fn get_stack_for_continuiation_swapping(&mut self) -> (*const u8, usize) {
        for (i, stack) in self.stacks_for_continuation_swapping.iter().enumerate() {
            if stack
                .is_used
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return (
                    unsafe {
                        stack.stack.as_ptr().offset(stack.stack.len() as isize - 64) as *const u8
                    },
                    i,
                );
            }
        }
        self.stacks_for_continuation_swapping
            .push(ContinuationStack {
                is_used: AtomicBool::new(true),
                stack: new_continuation_stack_buffer(),
            });
        self.get_stack_for_continuiation_swapping()
    }

    pub fn release_stack_for_continuation_swapping(&mut self, index: usize) {
        if let Some(stack) = self.stacks_for_continuation_swapping.get(index) {
            stack.is_used.store(false, Ordering::SeqCst);
        }
    }
}
