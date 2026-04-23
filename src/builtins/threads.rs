use super::*;
use crate::save_gc_context;

#[allow(unused)]
pub unsafe extern "C" fn new_thread(
    stack_pointer: usize,
    frame_pointer: usize,
    function: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    trace!("scheduler", "new_thread: function={:#x}", function);
    let runtime = get_runtime().get_mut();
    runtime.new_thread(function, stack_pointer, frame_pointer);
    BuiltInTypes::null_value() as usize
}

// I don't know what the deal is here

pub unsafe extern "C" fn __pause(_stack_pointer: usize, frame_pointer: usize) -> usize {
    use crate::gc::usdt_probes::{self, ThreadStateCode};

    let runtime = get_runtime().get_mut();

    // Fire USDT probe for thread state change
    usdt_probes::fire_thread_pause_enter();
    usdt_probes::fire_thread_state(ThreadStateCode::PausedForGc);

    trace!("scheduler", "pause: entering GC safepoint");
    let pause_start = std::time::Instant::now();

    // Use frame_pointer passed from Beagle code for FP-chain stack walking
    pause_current_thread(frame_pointer, runtime);

    // Memory barrier to ensure all writes are visible before parking
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    while runtime.is_paused() {
        // Park can unpark itself even if I haven't called unpark
        thread::park();
    }

    // Apparently, I can't count on this not unparking
    // I need some other mechanism to know that things are ready
    unpause_current_thread(runtime);

    let pause_duration_ns = pause_start.elapsed().as_nanos() as u64;
    trace!(
        "scheduler",
        "pause: resumed after {}us",
        pause_duration_ns / 1000
    );

    // Fire USDT probe for thread resuming
    usdt_probes::fire_thread_pause_exit(pause_duration_ns);
    usdt_probes::fire_thread_state(ThreadStateCode::Running);

    // Memory barrier to ensure all GC updates are visible before continuing
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    BuiltInTypes::null_value() as usize
}

pub fn pause_current_thread(frame_pointer: usize, runtime: &mut Runtime) {
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    state.pause(frame_pointer);
    condvar.notify_one();
    drop(state);
}

pub fn unpause_current_thread(runtime: &mut Runtime) {
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    state.unpause();
    condvar.notify_one();
}

pub extern "C" fn register_c_call(_stack_pointer: usize, frame_pointer: usize) -> usize {
    // Use frame_pointer passed from Beagle code for FP-chain stack walking
    let runtime = get_runtime().get_mut();
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    state.register_c_call(frame_pointer);
    condvar.notify_one();
    BuiltInTypes::null_value() as usize
}

pub extern "C" fn unregister_c_call() -> usize {
    let runtime = get_runtime().get_mut();
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    state.unregister_c_call();
    condvar.notify_one();
    while runtime.is_paused() {
        // Park can unpark itself even if I haven't called unpark
        thread::park();
    }
    BuiltInTypes::null_value() as usize
}

/// Called from a newly spawned thread to safely get a closure from a temporary root and call it.
/// This function:
/// 1. Unregisters from C-call (so this thread can participate in GC safepoints)
/// 2. Peeks the current closure value from the temporary root (which may have been updated by GC)
/// 3. Unregisters the temporary root
/// 4. Calls the closure via __call_fn
///
/// The temporary_root_id is passed as a tagged integer.
/// Get a value from a temporary root and unregister it.
/// This is called from Beagle code after entering a managed context.
pub extern "C" fn get_and_unregister_temp_root(temporary_root_id: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let root_id = BuiltInTypes::untag(temporary_root_id);
    // Read and unregister in one operation
    runtime.unregister_temporary_root(root_id)
}

/// Run a thread by calling the no-argument __run_thread_start function.
/// The Thread object is accessed via the get_my_thread_obj builtin inside Beagle code,
/// ensuring GC can update stack slots properly.
///
/// Key invariant: We are NOT counted in registered_thread_count until we hold gc_lock
/// and increment it. This prevents GC from waiting for us before we're ready.
///
/// Flow:
/// 1. Acquire gc_lock (GC can't start while we hold it)
/// 2. Increment registered_thread_count (now GC will count us)
/// 3. Release gc_lock
/// 4. Run Beagle code (first instruction is __pause, handles any pending GC)
/// 5. On cleanup: acquire gc_lock, decrement count, remove thread root, release
pub extern "C" fn run_thread(_unused: usize) -> usize {
    // Initialize per-thread data for this child thread
    let _ptd_ptr = crate::runtime::init_per_thread_data();
    let runtime = get_runtime().get_mut();
    let my_thread_id = thread::current().id();

    // === STARTUP ===
    // The parent's new_thread already incremented registered_thread_count and
    // registered us as c_calling with a null stack. We need to transition from
    // c_calling to active Beagle thread. The tricky part: between unregistering
    // from c_calling and hitting the first __pause safepoint, GC could run and
    // not account for us properly.
    //
    // Solution: Stay c_calling until the first safepoint (__pause or allocate).
    // We set a flag that __pause checks to atomically transition from c_calling
    // to paused (under the thread_state lock, ensuring no gap).
    //
    // But we also need to handle the case where is_paused=0 (no GC in progress)
    // at the first __pause check. In that case, __pause is skipped entirely.
    // We handle this by checking is_paused ourselves first: if GC is active, we
    // wait for it. Then we unregister from c_calling while holding gc_lock
    // (preventing any new GC from starting during the transition).
    loop {
        // If GC is in progress, we're c_calling so GC will count us and proceed.
        // Wait for it to finish.
        while runtime.is_paused() {
            thread::yield_now();
        }

        // Try to get gc_lock. While we hold it, no GC can start.
        match runtime.gc_lock.try_lock() {
            Ok(_guard) => {
                // Unregister from c_calling while holding gc_lock.
                // No GC can start during this window.
                {
                    let (lock, condvar) = &*runtime.thread_state.clone();
                    let mut state = lock.lock().unwrap();
                    state.c_calling_stack_pointers.remove(&my_thread_id);
                    condvar.notify_one();
                }
                let current_count = runtime
                    .registered_thread_count
                    .load(std::sync::atomic::Ordering::Acquire);
                // Fire USDT probes for thread start and registration
                crate::gc::usdt_probes::fire_thread_register(current_count);
                crate::gc::usdt_probes::fire_thread_start();
                // gc_lock released here - but we immediately enter Beagle code
                // where __pause is the first instruction
                break;
            }
            Err(_) => {
                // gc_lock is held by a GC in progress
                thread::yield_now();
            }
        }
    }

    // Enter Beagle code - __run_thread_start calls __pause as first instruction!
    // If GC started right after we released gc_lock, __pause will handle it.
    // Check if beagle.async/__run_thread_start exists — if so, use it to wrap
    // the thread function with the same async handler that __main__ provides.
    let thread_start_fn = if runtime
        .get_function_arity("beagle.async/__run_thread_start")
        .is_some()
    {
        "beagle.async/__run_thread_start"
    } else {
        "beagle.core/__run_thread_start"
    };
    let result = unsafe { call_fn_0(runtime, thread_start_fn) };

    // === CLEANUP ===
    // We're still registered but we're in C code now, not Beagle code.
    // If GC starts, it will wait for us to pause, but we can't pause from C.
    // Solution: register as c_calling so GC counts us and proceeds.
    {
        let (lock, condvar) = &*runtime.thread_state.clone();
        let mut state = lock.lock().unwrap();
        state.register_c_call(0); // No stack to scan
        condvar.notify_one();
    }

    // Now any GC will count us as c_calling and proceed.
    // Wait for any in-progress GC to finish, then unregister everything.
    loop {
        while runtime.is_paused() {
            thread::yield_now();
        }

        match runtime.gc_lock.try_lock() {
            Ok(_guard) => {
                // While holding lock: unregister from c_calling, decrement count, remove root
                {
                    let (lock, condvar) = &*runtime.thread_state.clone();
                    let mut state = lock.lock().unwrap();
                    state.unregister_c_call();
                    condvar.notify_one();
                }
                let new_count = runtime
                    .registered_thread_count
                    .fetch_sub(1, std::sync::atomic::Ordering::Release)
                    - 1;
                // Thread object is in our GlobalObjectBlock - cleanup happens
                // when thread_globals entry is removed. MutatorState lives in
                // per-OS-thread `thread_local!` storage that outlives this
                // removal, so the JIT epilogue's gc_frame_unlink still sees
                // a valid pointer after we drop the ThreadGlobal.
                runtime
                    .memory
                    .thread_globals
                    .lock()
                    .unwrap()
                    .remove(&my_thread_id);
                // Clean up per-thread data
                crate::runtime::cleanup_per_thread_data();

                // Fire USDT probes for thread unregistration and exit
                crate::gc::usdt_probes::fire_thread_unregister(new_count);
                crate::gc::usdt_probes::fire_thread_exit();
                break;
            }
            Err(_) => {
                thread::yield_now();
            }
        }
    }

    result
}

/// Get the current thread's Thread object from its GlobalObjectBlock.
/// Called from Beagle code in __run_thread_start.
/// Takes stack_pointer and frame_pointer so we can call __pause if needed.
///
/// CRITICAL: This function must return a pointer that won't become stale before
/// the caller can use it. We check is_paused() after releasing the lock to ensure
/// that if GC is about to run, we pause and get the updated pointer.
pub extern "C" fn get_my_thread_obj(stack_pointer: usize, frame_pointer: usize) -> usize {
    // CRITICAL: Save the gc context here so that if we call __pause,
    // the GC will have the correct return address pointing to Beagle code.
    save_gc_context!(stack_pointer, frame_pointer);

    let runtime = get_runtime().get();
    let thread_id = thread::current().id();

    // Read the Thread object from our GlobalObjectBlock's reserved slot.
    // We need the gc_lock to ensure the value isn't stale during GC.
    let thread_obj = loop {
        // Check if GC needs us to pause first
        if runtime.is_paused() {
            unsafe { __pause(stack_pointer, frame_pointer) };
            continue;
        }

        // Try to get the lock
        let obj = match runtime.gc_lock.try_lock() {
            Ok(_guard) => {
                let thread_globals = runtime.memory.thread_globals.lock().unwrap();
                thread_globals
                    .get(&thread_id)
                    .map(|tg| tg.get_thread_object())
                    .expect("ThreadGlobal not found in get_my_thread_obj")
            }
            Err(_) => {
                thread::yield_now();
                continue;
            }
        };

        // After releasing the lock, check if GC is pending.
        if runtime.is_paused() {
            unsafe { __pause(stack_pointer, frame_pointer) };
            continue;
        }

        break obj;
    };

    #[cfg(debug_assertions)]
    {
        if std::env::var("BEAGLE_THREAD_DEBUG").is_ok() {
            eprintln!(
                "[THREAD_DEBUG] get_my_thread_obj: thread_id={:?} thread_obj={:#x}",
                thread_id, thread_obj
            );
            if BuiltInTypes::is_heap_pointer(thread_obj) {
                let heap_obj = HeapObject::from_tagged(thread_obj);
                let closure_field = heap_obj.get_field(0);
                eprintln!(
                    "[THREAD_DEBUG]   closure_field={:#x} is_heap_ptr={}",
                    closure_field,
                    BuiltInTypes::is_heap_pointer(closure_field)
                );
                if BuiltInTypes::is_heap_pointer(closure_field) {
                    let closure_tag = BuiltInTypes::get_kind(closure_field);
                    eprintln!("[THREAD_DEBUG]   closure_tag={:?}", closure_tag);
                    if matches!(closure_tag, BuiltInTypes::Closure) {
                        let closure_obj = HeapObject::from_tagged(closure_field);
                        let fn_ptr = closure_obj.get_field(0);
                        let fn_ptr_untagged = BuiltInTypes::untag(fn_ptr);
                        if let Some(function) =
                            runtime.get_function_by_pointer(fn_ptr_untagged as *const u8)
                        {
                            eprintln!(
                                "[THREAD_DEBUG]   closure fn={} args={}",
                                function.name, function.number_of_args
                            );
                        } else {
                            eprintln!("[THREAD_DEBUG]   closure fn ptr not found: {:#x}", fn_ptr);
                            panic!("Closure function pointer not found in runtime");
                        }
                    }
                }
            }
        }
    }

    thread_obj
}

/// Returns true if notified (a future completed), false if timed out.
pub extern "C" fn future_wait(timeout_ms: usize) -> usize {
    let timeout = BuiltInTypes::untag(timeout_ms) as u64;
    trace!("scheduler", "future_wait: timeout={}ms", timeout);
    let runtime = get_runtime().get();
    let notified = runtime.future_wait_set.wait_timeout(timeout);
    trace!("scheduler", "future_wait: returned notified={}", notified);
    BuiltInTypes::construct_boolean(notified) as usize
}

/// future_notify builtin - Notify all threads waiting on futures.
/// Should be called after a future completes (after reset! on the future atom).
pub extern "C" fn future_notify() -> usize {
    trace!("scheduler", "future_notify: waking all waiters");
    let runtime = get_runtime().get();
    runtime.future_wait_set.notify_all();
    BuiltInTypes::null_value() as usize
}

pub extern "C" fn eval(stack_pointer: usize, frame_pointer: usize, code: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let code = match BuiltInTypes::get_kind(code) {
        BuiltInTypes::String => runtime.get_string_literal(code),
        BuiltInTypes::HeapObject => {
            let code_obj = HeapObject::from_tagged(code);
            if code_obj.get_header().type_id != TYPE_ID_STRING
                && code_obj.get_header().type_id != TYPE_ID_STRING_SLICE
                && code_obj.get_header().type_id != TYPE_ID_CONS_STRING
            {
                unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "TypeError",
                        format!(
                            "Expected string, got heap object with type_id {}",
                            code_obj.get_header().type_id
                        ),
                    );
                }
            }
            let bytes = runtime.get_string_bytes_vec(code);
            match std::str::from_utf8(&bytes) {
                Ok(s) => s.to_string(),
                Err(e) => unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "EncodingError",
                        format!("String contains invalid UTF-8: {}", e),
                    );
                },
            }
        }
        _ => unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!("Expected string, got {:?}", BuiltInTypes::get_kind(code)),
            );
        },
    };
    let result = match runtime.compile_string(&code) {
        Ok(result) => result,
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "CompileError", format!("{}", e));
        },
    };
    mem::forget(code);
    if result == 0 {
        return BuiltInTypes::null_value() as usize;
    }
    // Route the call through the apply_call_0 shim so X28 (MutatorState)
    // is guaranteed valid on Beagle-side entry. Direct `transmute + call`
    // relies on Rust preserving X28, but Rust is free to use callee-saved
    // registers as scratch while `eval` executes — in debug builds it
    // often does, and the Beagle prologue's inline gc_frame_link then
    // dereferences garbage via [x28, #16] and faults. Same bug class as
    // the apply_call_N fix. Surfaced as SIGBUS in resumable_eval_test.
    let runtime = get_runtime().get();
    let shim_entry = runtime
        .get_function_by_name("beagle.builtin/apply_call_0")
        .expect("apply_call_0 trampoline not compiled");
    let shim_ptr = runtime
        .get_pointer(shim_entry)
        .expect("apply_call_0 has no code pointer") as usize;
    let shim: extern "C" fn(usize) -> usize = unsafe { transmute(shim_ptr) };
    shim(result as usize)
}

pub extern "C" fn eval_in_ns(
    stack_pointer: usize,
    frame_pointer: usize,
    code: usize,
    namespace: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Extract namespace string
    let ns_str = match BuiltInTypes::get_kind(namespace) {
        BuiltInTypes::String => runtime.get_string_literal(namespace),
        BuiltInTypes::HeapObject => {
            let bytes = runtime.get_string_bytes_vec(namespace);
            match std::str::from_utf8(&bytes) {
                Ok(s) => s.to_string(),
                Err(e) => unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "EncodingError",
                        format!("Namespace string contains invalid UTF-8: {}", e),
                    );
                },
            }
        }
        _ => unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!(
                    "Expected string for namespace, got {:?}",
                    BuiltInTypes::get_kind(namespace)
                ),
            );
        },
    };

    // Extract code string
    let code = match BuiltInTypes::get_kind(code) {
        BuiltInTypes::String => runtime.get_string_literal(code),
        BuiltInTypes::HeapObject => {
            let bytes = runtime.get_string_bytes_vec(code);
            match std::str::from_utf8(&bytes) {
                Ok(s) => s.to_string(),
                Err(e) => unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "EncodingError",
                        format!("String contains invalid UTF-8: {}", e),
                    );
                },
            }
        }
        _ => unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!("Expected string, got {:?}", BuiltInTypes::get_kind(code)),
            );
        },
    };

    let result = match runtime.compile_string_in_namespace(&code, &ns_str) {
        Ok(result) => result,
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "CompileError", format!("{}", e));
        },
    };
    mem::forget(code);
    mem::forget(ns_str);
    if result == 0 {
        return BuiltInTypes::null_value() as usize;
    }
    let f: fn() -> usize = unsafe { transmute(result) };
    f()
}

pub extern "C" fn sleep(_stack_pointer: usize, frame_pointer: usize, time: usize) -> usize {
    let time = BuiltInTypes::untag(time);
    trace!("scheduler", "sleep: {}ms", time);

    // Register as c_calling before sleeping. Without this, GC would wait
    // forever for this thread to hit a safepoint while it's blocked in sleep.
    {
        let runtime = get_runtime().get();
        let thread_state = runtime.thread_state.clone();
        let (lock, condvar) = &*thread_state;
        let mut state = lock.lock().unwrap();
        state.register_c_call(frame_pointer);
        condvar.notify_one();
    }

    std::thread::sleep(std::time::Duration::from_millis(time as u64));

    // Unregister from c_calling under gc_lock
    {
        let runtime = get_runtime().get();
        while runtime.is_paused() {
            thread::yield_now();
        }
        loop {
            match runtime.gc_lock.try_lock() {
                Ok(_guard) => {
                    let thread_state = runtime.thread_state.clone();
                    let (lock, condvar) = &*thread_state;
                    let mut state = lock.lock().unwrap();
                    state.unregister_c_call();
                    condvar.notify_one();
                    break;
                }
                Err(_) => thread::yield_now(),
            }
        }
    }

    BuiltInTypes::null_value() as usize
}

/// High-precision timer returning nanoseconds since an arbitrary epoch
/// Useful for benchmarking - subtract two values to get elapsed time
pub extern "C" fn time_now() -> usize {
    use std::time::Instant;
    // Use a static instant as the epoch to avoid overflow
    // This gives us relative timing which is what we need for benchmarks
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    let elapsed = start.elapsed().as_nanos() as isize;
    BuiltInTypes::Int.tag(elapsed) as usize
}

/// Returns a unique identifier for the current thread
/// Useful for verifying that code is running on different threads
pub extern "C" fn thread_id() -> usize {
    use std::sync::atomic::{AtomicIsize, Ordering};
    static NEXT_ID: AtomicIsize = AtomicIsize::new(1);
    thread_local! {
        static THREAD_ID: isize = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    }
    let id = THREAD_ID.with(|id| *id);
    BuiltInTypes::Int.tag(id) as usize
}

/// Returns the number of available CPU cores
/// Useful for sizing thread pools
pub extern "C" fn get_cpu_count() -> usize {
    let count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1) as isize;
    BuiltInTypes::Int.tag(count) as usize
}
