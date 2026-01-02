//! Debug tracing for GC and allocation events
//!
//! This module provides detailed tracing of:
//! - Every allocation
//! - Thread state changes (running, paused, waiting, c-call)
//! - GC events (start, root processing, copy operations, end)
//!
//! Enable with BEAGLE_GC_TRACE=1 environment variable

// Many tracing functions and variants are intentionally kept for debugging
// even when not actively used in production code.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::ThreadId;
use std::time::Instant;

/// Global flag to enable/disable tracing
static TRACING_ENABLED: AtomicBool = AtomicBool::new(false);
/// Incremental mode - write events immediately to file
static INCREMENTAL_MODE: AtomicBool = AtomicBool::new(false);
/// Global event counter for ordering
static EVENT_COUNTER: AtomicU64 = AtomicU64::new(0);
/// Start time for relative timestamps
static START_TIME: Mutex<Option<Instant>> = Mutex::new(None);
/// File for incremental output
static INCREMENTAL_FILE: Mutex<Option<File>> = Mutex::new(None);

// Thread-local buffer to reduce lock contention
thread_local! {
    static LOCAL_EVENTS: RefCell<Vec<TraceEvent>> = const { RefCell::new(Vec::new()) };
    static THREAD_NAME: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Global event log
static GLOBAL_EVENTS: Mutex<Vec<TraceEvent>> = Mutex::new(Vec::new());
/// Thread state tracking
static THREAD_STATES: Mutex<Option<HashMap<ThreadId, ThreadState>>> = Mutex::new(None);

/// Initialize tracing based on environment variable
pub fn init() {
    if std::env::var("BEAGLE_GC_TRACE").is_ok() {
        TRACING_ENABLED.store(true, Ordering::SeqCst);
        let mut start = START_TIME.lock().unwrap();
        *start = Some(Instant::now());
        let mut states = THREAD_STATES.lock().unwrap();
        *states = Some(HashMap::new());

        // Check for incremental mode (enabled by default if trace file is set, or explicitly)
        if std::env::var("BEAGLE_GC_TRACE_INCREMENTAL").is_ok()
            || std::env::var("BEAGLE_GC_TRACE_FILE").is_ok()
        {
            INCREMENTAL_MODE.store(true, Ordering::SeqCst);

            // Open file immediately
            let trace_file = std::env::var("BEAGLE_GC_TRACE_FILE")
                .unwrap_or_else(|_| "/tmp/beagle_gc_trace.txt".to_string());

            match File::create(&trace_file) {
                Ok(mut file) => {
                    // Write header
                    let _ = writeln!(file, "# Beagle GC Trace (incremental mode)");
                    let _ = writeln!(
                        file,
                        "# Format: seq timestamp_us thread_id event_type details"
                    );
                    let _ = writeln!(file, "#");
                    let mut incremental = INCREMENTAL_FILE.lock().unwrap();
                    *incremental = Some(file);
                    eprintln!("[GC_TRACE] Incremental tracing to {}", trace_file);
                }
                Err(e) => {
                    eprintln!("[GC_TRACE] Failed to create trace file: {}", e);
                    INCREMENTAL_MODE.store(false, Ordering::SeqCst);
                }
            }
        }

        eprintln!("[GC_TRACE] Tracing enabled");
    }
}

/// Check if tracing is enabled
#[inline]
pub fn is_enabled() -> bool {
    TRACING_ENABLED.load(Ordering::Relaxed)
}

/// Get relative timestamp in microseconds
fn timestamp_us() -> u64 {
    if let Ok(start) = START_TIME.lock() {
        if let Some(start) = *start {
            return start.elapsed().as_micros() as u64;
        }
    }
    0
}

/// Thread state for tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadState {
    Running,
    PausedForGc,
    WaitingOnLock,
    InCCall,
    Starting,
    Exiting,
}

impl std::fmt::Display for ThreadState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThreadState::Running => write!(f, "RUNNING"),
            ThreadState::PausedForGc => write!(f, "PAUSED_FOR_GC"),
            ThreadState::WaitingOnLock => write!(f, "WAITING_ON_LOCK"),
            ThreadState::InCCall => write!(f, "IN_C_CALL"),
            ThreadState::Starting => write!(f, "STARTING"),
            ThreadState::Exiting => write!(f, "EXITING"),
        }
    }
}

/// A single trace event
#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub seq: u64,
    pub timestamp_us: u64,
    pub thread_id: ThreadId,
    pub event: EventKind,
}

/// Types of events we track
#[derive(Debug, Clone)]
pub enum EventKind {
    /// Allocation happened
    Alloc {
        address: usize,
        size_words: usize,
        generation: Generation,
    },
    /// Thread state changed
    ThreadStateChange {
        old_state: Option<ThreadState>,
        new_state: ThreadState,
        reason: String,
    },
    /// GC started
    GcStart {
        gc_number: usize,
        young_alloc_offset: usize,
    },
    /// GC ended
    GcEnd {
        gc_number: usize,
        young_alloc_offset: usize,
        objects_copied: usize,
    },
    /// Root gathered from stack
    RootGathered {
        slot_addr: usize,
        value: usize,
        function_name: Option<String>,
    },
    /// Object copied
    ObjectCopied {
        old_addr: usize,
        new_addr: usize,
        size_bytes: usize,
    },
    /// Stack registered for GC
    StackRegistered {
        stack_base: usize,
        frame_pointer: usize,
        gc_return_addr: usize,
    },
    /// Closure relocated during thread creation
    ClosureRelocated { old_ptr: usize, new_ptr: usize },
    /// Custom message for debugging
    Debug { message: String },
}

#[derive(Debug, Clone, Copy)]
pub enum Generation {
    Young,
    Old,
}

impl std::fmt::Display for Generation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Generation::Young => write!(f, "young"),
            Generation::Old => write!(f, "old"),
        }
    }
}

/// Record an event
fn record_event(event: EventKind) {
    if !is_enabled() {
        return;
    }

    let trace_event = TraceEvent {
        seq: EVENT_COUNTER.fetch_add(1, Ordering::SeqCst),
        timestamp_us: timestamp_us(),
        thread_id: std::thread::current().id(),
        event,
    };

    // In incremental mode, write directly to file
    if INCREMENTAL_MODE.load(Ordering::Relaxed) {
        write_event_to_file(&trace_event);
    }

    // Add to thread-local buffer
    LOCAL_EVENTS.with(|events| {
        let mut events = events.borrow_mut();
        events.push(trace_event.clone());

        // Flush if buffer is getting large
        if events.len() >= 100 {
            if let Ok(mut global) = GLOBAL_EVENTS.lock() {
                global.extend(events.drain(..));
            }
        }
    });
}

/// Write a single event to the incremental file
fn write_event_to_file(event: &TraceEvent) {
    if let Ok(mut file_guard) = INCREMENTAL_FILE.lock() {
        if let Some(ref mut file) = *file_guard {
            let thread_str = format!("{:?}", event.thread_id);
            let thread_short = thread_str
                .trim_start_matches("ThreadId(")
                .trim_end_matches(')');

            let line = match &event.event {
                EventKind::Alloc {
                    address,
                    size_words,
                    generation,
                } => {
                    format!(
                        "{:08} {:012} T{:>3} ALLOC      addr={:#x} size={} gen={}",
                        event.seq,
                        event.timestamp_us,
                        thread_short,
                        address,
                        size_words,
                        generation
                    )
                }
                EventKind::ThreadStateChange {
                    old_state,
                    new_state,
                    reason,
                } => {
                    let old_str = old_state.map(|s| s.to_string()).unwrap_or("?".to_string());
                    format!(
                        "{:08} {:012} T{:>3} THREAD     {} -> {} ({})",
                        event.seq, event.timestamp_us, thread_short, old_str, new_state, reason
                    )
                }
                EventKind::GcStart {
                    gc_number,
                    young_alloc_offset,
                } => {
                    format!(
                        "{:08} {:012} T{:>3} GC_START   gc_num={} young_offset={:#x}",
                        event.seq, event.timestamp_us, thread_short, gc_number, young_alloc_offset
                    )
                }
                EventKind::GcEnd {
                    gc_number,
                    young_alloc_offset,
                    objects_copied,
                } => {
                    format!(
                        "{:08} {:012} T{:>3} GC_END     gc_num={} young_offset={:#x} copied={}",
                        event.seq,
                        event.timestamp_us,
                        thread_short,
                        gc_number,
                        young_alloc_offset,
                        objects_copied
                    )
                }
                EventKind::RootGathered {
                    slot_addr,
                    value,
                    function_name,
                } => {
                    let fn_str = function_name.as_deref().unwrap_or("?");
                    format!(
                        "{:08} {:012} T{:>3} ROOT       slot={:#x} val={:#x} fn={}",
                        event.seq, event.timestamp_us, thread_short, slot_addr, value, fn_str
                    )
                }
                EventKind::ObjectCopied {
                    old_addr,
                    new_addr,
                    size_bytes,
                } => {
                    format!(
                        "{:08} {:012} T{:>3} COPY       {:#x} -> {:#x} size={}",
                        event.seq, event.timestamp_us, thread_short, old_addr, new_addr, size_bytes
                    )
                }
                EventKind::StackRegistered {
                    stack_base,
                    frame_pointer,
                    gc_return_addr,
                } => {
                    format!(
                        "{:08} {:012} T{:>3} STACK_REG  base={:#x} fp={:#x} gc_ret={:#x}",
                        event.seq,
                        event.timestamp_us,
                        thread_short,
                        stack_base,
                        frame_pointer,
                        gc_return_addr
                    )
                }
                EventKind::ClosureRelocated { old_ptr, new_ptr } => {
                    format!(
                        "{:08} {:012} T{:>3} CLOSURE    {:#x} -> {:#x}",
                        event.seq, event.timestamp_us, thread_short, old_ptr, new_ptr
                    )
                }
                EventKind::Debug { message } => {
                    format!(
                        "{:08} {:012} T{:>3} DEBUG      {}",
                        event.seq, event.timestamp_us, thread_short, message
                    )
                }
            };

            let _ = writeln!(file, "{}", line);
            // Flush to ensure data is written even on crash
            let _ = file.flush();
        }
    }
}

/// Record an allocation
pub fn trace_alloc(address: usize, size_words: usize, generation: Generation) {
    record_event(EventKind::Alloc {
        address,
        size_words,
        generation,
    });
}

/// Record a thread state change
pub fn trace_thread_state(new_state: ThreadState, reason: &str) {
    if !is_enabled() {
        return;
    }

    let thread_id = std::thread::current().id();
    let old_state = if let Ok(mut states) = THREAD_STATES.lock() {
        if let Some(ref mut states) = *states {
            let old = states.get(&thread_id).copied();
            states.insert(thread_id, new_state);
            old
        } else {
            None
        }
    } else {
        None
    };

    record_event(EventKind::ThreadStateChange {
        old_state,
        new_state,
        reason: reason.to_string(),
    });
}

/// Record GC start
pub fn trace_gc_start(gc_number: usize, young_alloc_offset: usize) {
    record_event(EventKind::GcStart {
        gc_number,
        young_alloc_offset,
    });
}

/// Record GC end
pub fn trace_gc_end(gc_number: usize, young_alloc_offset: usize, objects_copied: usize) {
    record_event(EventKind::GcEnd {
        gc_number,
        young_alloc_offset,
        objects_copied,
    });
}

/// Record a root gathered from stack
pub fn trace_root_gathered(slot_addr: usize, value: usize, function_name: Option<&str>) {
    record_event(EventKind::RootGathered {
        slot_addr,
        value,
        function_name: function_name.map(|s| s.to_string()),
    });
}

/// Record an object copy
pub fn trace_object_copied(old_addr: usize, new_addr: usize, size_bytes: usize) {
    record_event(EventKind::ObjectCopied {
        old_addr,
        new_addr,
        size_bytes,
    });
}

/// Record stack registration
pub fn trace_stack_registered(stack_base: usize, frame_pointer: usize, gc_return_addr: usize) {
    record_event(EventKind::StackRegistered {
        stack_base,
        frame_pointer,
        gc_return_addr,
    });
}

/// Record closure relocation during thread creation
pub fn trace_closure_relocated(old_ptr: usize, new_ptr: usize) {
    record_event(EventKind::ClosureRelocated { old_ptr, new_ptr });
}

/// Record a debug message
pub fn trace_debug(message: &str) {
    record_event(EventKind::Debug {
        message: message.to_string(),
    });
}

/// Flush thread-local events to global log
pub fn flush() {
    if !is_enabled() {
        return;
    }

    LOCAL_EVENTS.with(|events| {
        let mut events = events.borrow_mut();
        if let Ok(mut global) = GLOBAL_EVENTS.lock() {
            global.extend(events.drain(..));
        }
    });
}

/// Get all thread states
pub fn get_thread_states() -> HashMap<ThreadId, ThreadState> {
    if let Ok(states) = THREAD_STATES.lock() {
        states.clone().unwrap_or_default()
    } else {
        HashMap::new()
    }
}

/// Dump all events to a file
pub fn dump_to_file(path: &str) -> std::io::Result<()> {
    flush();

    let events = if let Ok(global) = GLOBAL_EVENTS.lock() {
        global.clone()
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to lock global events",
        ));
    };

    let mut file = File::create(path)?;

    writeln!(file, "# Beagle GC Trace")?;
    writeln!(file, "# Total events: {}", events.len())?;
    writeln!(file, "#")?;
    writeln!(
        file,
        "# Format: seq timestamp_us thread_id event_type details"
    )?;
    writeln!(file, "#")?;

    for event in &events {
        let thread_str = format!("{:?}", event.thread_id);
        let thread_short = thread_str
            .trim_start_matches("ThreadId(")
            .trim_end_matches(')');

        match &event.event {
            EventKind::Alloc {
                address,
                size_words,
                generation,
            } => {
                writeln!(
                    file,
                    "{:08} {:012} T{:>3} ALLOC      addr={:#x} size={} gen={}",
                    event.seq, event.timestamp_us, thread_short, address, size_words, generation
                )?;
            }
            EventKind::ThreadStateChange {
                old_state,
                new_state,
                reason,
            } => {
                let old_str = old_state.map(|s| s.to_string()).unwrap_or("?".to_string());
                writeln!(
                    file,
                    "{:08} {:012} T{:>3} THREAD     {} -> {} ({})",
                    event.seq, event.timestamp_us, thread_short, old_str, new_state, reason
                )?;
            }
            EventKind::GcStart {
                gc_number,
                young_alloc_offset,
            } => {
                writeln!(
                    file,
                    "{:08} {:012} T{:>3} GC_START   gc_num={} young_offset={:#x}",
                    event.seq, event.timestamp_us, thread_short, gc_number, young_alloc_offset
                )?;
            }
            EventKind::GcEnd {
                gc_number,
                young_alloc_offset,
                objects_copied,
            } => {
                writeln!(
                    file,
                    "{:08} {:012} T{:>3} GC_END     gc_num={} young_offset={:#x} copied={}",
                    event.seq,
                    event.timestamp_us,
                    thread_short,
                    gc_number,
                    young_alloc_offset,
                    objects_copied
                )?;
            }
            EventKind::RootGathered {
                slot_addr,
                value,
                function_name,
            } => {
                let fn_str = function_name.as_deref().unwrap_or("?");
                writeln!(
                    file,
                    "{:08} {:012} T{:>3} ROOT       slot={:#x} val={:#x} fn={}",
                    event.seq, event.timestamp_us, thread_short, slot_addr, value, fn_str
                )?;
            }
            EventKind::ObjectCopied {
                old_addr,
                new_addr,
                size_bytes,
            } => {
                writeln!(
                    file,
                    "{:08} {:012} T{:>3} COPY       {:#x} -> {:#x} size={}",
                    event.seq, event.timestamp_us, thread_short, old_addr, new_addr, size_bytes
                )?;
            }
            EventKind::StackRegistered {
                stack_base,
                frame_pointer,
                gc_return_addr,
            } => {
                writeln!(
                    file,
                    "{:08} {:012} T{:>3} STACK_REG  base={:#x} fp={:#x} gc_ret={:#x}",
                    event.seq,
                    event.timestamp_us,
                    thread_short,
                    stack_base,
                    frame_pointer,
                    gc_return_addr
                )?;
            }
            EventKind::ClosureRelocated { old_ptr, new_ptr } => {
                writeln!(
                    file,
                    "{:08} {:012} T{:>3} CLOSURE    {:#x} -> {:#x}",
                    event.seq, event.timestamp_us, thread_short, old_ptr, new_ptr
                )?;
            }
            EventKind::Debug { message } => {
                writeln!(
                    file,
                    "{:08} {:012} T{:>3} DEBUG      {}",
                    event.seq, event.timestamp_us, thread_short, message
                )?;
            }
        }
    }

    eprintln!("[GC_TRACE] Dumped {} events to {}", events.len(), path);
    Ok(())
}

/// Generate a summary of events
pub fn generate_summary() -> String {
    flush();

    let events = if let Ok(global) = GLOBAL_EVENTS.lock() {
        global.clone()
    } else {
        return "Failed to lock events".to_string();
    };

    let mut summary = String::new();
    writeln!(summary, "=== GC Trace Summary ===").unwrap();
    writeln!(summary, "Total events: {}", events.len()).unwrap();

    // Count by type
    let mut alloc_count = 0;
    let mut gc_count = 0;
    let mut copy_count = 0;
    let mut thread_changes = 0;
    let mut root_count = 0;

    // Track threads seen
    let mut threads_seen: std::collections::HashSet<ThreadId> = std::collections::HashSet::new();

    for event in &events {
        threads_seen.insert(event.thread_id);
        match &event.event {
            EventKind::Alloc { .. } => alloc_count += 1,
            EventKind::GcStart { .. } => gc_count += 1,
            EventKind::ObjectCopied { .. } => copy_count += 1,
            EventKind::ThreadStateChange { .. } => thread_changes += 1,
            EventKind::RootGathered { .. } => root_count += 1,
            _ => {}
        }
    }

    writeln!(summary, "Allocations: {}", alloc_count).unwrap();
    writeln!(summary, "GC cycles: {}", gc_count).unwrap();
    writeln!(summary, "Objects copied: {}", copy_count).unwrap();
    writeln!(summary, "Thread state changes: {}", thread_changes).unwrap();
    writeln!(summary, "Roots gathered: {}", root_count).unwrap();
    writeln!(summary, "Unique threads: {}", threads_seen.len()).unwrap();

    summary
}

/// Print current thread states
pub fn print_thread_states() {
    let states = get_thread_states();
    eprintln!("=== Thread States ===");
    for (tid, state) in &states {
        eprintln!("  {:?}: {}", tid, state);
    }
}
