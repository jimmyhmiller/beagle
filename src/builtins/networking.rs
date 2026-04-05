use super::*;

/// Always spawns a dedicated I/O thread.
pub extern "C" fn event_loop_create(pool_size: usize) -> usize {
    let pool_size = BuiltInTypes::untag(pool_size);
    let runtime = get_runtime().get();

    trace!("event-loop", "event_loop_create: pool_size={}", pool_size);
    match runtime.event_loops.create(pool_size) {
        Ok(id) => {
            trace!("event-loop", "event_loop_create: ok id={}", id);
            BuiltInTypes::Int.tag(id as isize) as usize
        }
        Err(e) => {
            eprintln!("Failed to create event loop: {}", e);
            BuiltInTypes::Int.tag(-1) as usize
        }
    }
}

/// Create a new event loop (same as event_loop_create — kept for backward compatibility)
pub extern "C" fn event_loop_create_threaded(pool_size: usize) -> usize {
    event_loop_create(pool_size)
}

/// Run the event loop once, waiting for results with given timeout in milliseconds
/// The dedicated I/O thread handles polling; this waits on Condvar for results.
/// Returns the number of available results.
pub extern "C" fn event_loop_run_once(
    _stack_pointer: usize,
    frame_pointer: usize,
    loop_id: usize,
    timeout_ms: usize,
) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let timeout_ms = BuiltInTypes::untag(timeout_ms) as u64;
    let runtime = get_runtime().get_mut();

    // Simple polling: check for results, and if none, sleep briefly.
    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    let count = event_loop.tcp_results_count();
    if count > 0 {
        trace!(
            "event-loop",
            "run_once: loop={} has {} results ready", loop_id, count
        );
        return BuiltInTypes::Int.tag(count as isize) as usize;
    }

    // No results yet — sleep briefly to avoid busy-looping.
    // Register as c_calling during sleep so GC doesn't wait for this thread
    // to reach a safepoint (it can't while blocked in sleep).
    {
        let thread_state = runtime.thread_state.clone();
        let (lock, condvar) = &*thread_state;
        let mut state = lock.lock().unwrap();
        state.register_c_call(frame_pointer);
        condvar.notify_one();
    }

    let sleep_ms = if timeout_ms == 0 {
        1
    } else {
        timeout_ms.min(1)
    };
    thread::sleep(std::time::Duration::from_millis(sleep_ms));

    // Unregister from c_calling, waiting for any in-progress GC to finish
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

    let runtime = get_runtime().get_mut();
    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };
    let count = event_loop.tcp_results_count();
    trace!(
        "event-loop",
        "run_once: loop={} timeout={}ms results={}", loop_id, timeout_ms, count
    );
    BuiltInTypes::Int.tag(count as isize) as usize
}

/// Wake the event loop from another thread
pub extern "C" fn event_loop_wake(loop_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();

    trace!("event-loop", "event_loop_wake: loop={}", loop_id);
    if let Some(event_loop) = runtime.event_loops.get(loop_id) {
        match event_loop.wake() {
            Ok(()) => BuiltInTypes::Int.tag(1) as usize,
            Err(e) => {
                eprintln!("Event loop wake error: {}", e);
                BuiltInTypes::Int.tag(-1) as usize
            }
        }
    } else {
        trace!("event-loop", "event_loop_wake: loop {} not found", loop_id);
        BuiltInTypes::Int.tag(-1) as usize
    }
}

/// Destroy an event loop and shut down its I/O thread
pub extern "C" fn event_loop_destroy(loop_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();

    if runtime.event_loops.unregister(loop_id).is_some() {
        BuiltInTypes::Int.tag(1) as usize
    } else {
        BuiltInTypes::Int.tag(-1) as usize
    }
}

// =============================================================================
// TCP Networking Builtins
// =============================================================================

/// Start a non-blocking TCP connection
/// In InProcess mode: returns socket_id on success, -1 on error
/// In Threaded mode: submits operation, returns 0 on success, -1 on error
pub extern "C" fn tcp_connect_async(
    _stack_pointer: usize,
    _frame_pointer: usize,
    loop_id: usize,
    host: usize,
    port: usize,
    future_atom: usize,
) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let port = BuiltInTypes::untag(port) as u16;
    let future_atom = BuiltInTypes::untag(future_atom);
    let runtime = get_runtime().get_mut();

    let host_str = runtime.get_string_literal(host);
    trace!(
        "tcp",
        "tcp_connect_async: loop={} host={} port={} future_atom={}",
        loop_id,
        host_str,
        port,
        future_atom
    );

    // Parse the address
    let addr = match format!("{}:{}", host_str, port).parse::<std::net::SocketAddr>() {
        Ok(addr) => addr,
        Err(_) => {
            // Try resolving hostname
            match std::net::ToSocketAddrs::to_socket_addrs(&(host_str, port)) {
                Ok(mut addrs) => match addrs.next() {
                    Some(addr) => addr,
                    None => return BuiltInTypes::Int.tag(-1) as usize,
                },
                Err(_) => return BuiltInTypes::Int.tag(-1) as usize,
            }
        }
    };

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };
    let op_id = event_loop.next_tcp_op_id();
    match event_loop.submit_tcp_op(crate::runtime::TcpOperation::Connect {
        addr,
        future_atom,
        op_id,
    }) {
        Ok(()) => BuiltInTypes::Int.tag(op_id as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Create a TCP listener
/// Returns listener_id on success, -1 on error
pub extern "C" fn tcp_listen(
    _stack_pointer: usize,
    _frame_pointer: usize,
    loop_id: usize,
    host: usize,
    port: usize,
    backlog: usize,
) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let port = BuiltInTypes::untag(port) as u16;
    let backlog = BuiltInTypes::untag(backlog) as u32;
    let runtime = get_runtime().get_mut();

    let host_str = runtime.get_string_literal(host);
    trace!(
        "tcp",
        "tcp_listen: loop={} host={} port={} backlog={}", loop_id, host_str, port, backlog
    );

    // Parse the address
    let addr = match format!("{}:{}", host_str, port).parse::<std::net::SocketAddr>() {
        Ok(addr) => addr,
        Err(_) => return BuiltInTypes::Int.tag(-1) as usize,
    };

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };
    match event_loop.submit_tcp_listen(addr, backlog) {
        Ok(listener_id) => {
            trace!("tcp", "tcp_listen: ok listener_id={}", listener_id);
            BuiltInTypes::Int.tag(listener_id as isize) as usize
        }
        Err(_e) => {
            trace!("tcp", "tcp_listen: failed: {}", _e);
            BuiltInTypes::Int.tag(-1) as usize
        }
    }
}

/// Start accepting a connection on a listener
/// Returns op_id on success (operation started), -1 on error
/// In Threaded mode: submits operation, returns 0 on success
pub extern "C" fn tcp_accept_async(
    loop_id: usize,
    listener_id: usize,
    future_atom: usize,
) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let listener_id = BuiltInTypes::untag(listener_id);
    let future_atom = BuiltInTypes::untag(future_atom);
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };
    trace!(
        "tcp",
        "tcp_accept_async: loop={} listener={} future_atom={}", loop_id, listener_id, future_atom
    );
    let op_id = event_loop.next_tcp_op_id();
    match event_loop.submit_tcp_op(crate::runtime::TcpOperation::Accept {
        listener_id,
        future_atom,
        op_id,
    }) {
        Ok(()) => BuiltInTypes::Int.tag(op_id as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Start a read operation on a TCP socket
/// Returns op_id on success (operation started), -1 on error
/// In Threaded mode: submits operation, returns 0 on success
pub extern "C" fn tcp_read_async(
    loop_id: usize,
    socket_id: usize,
    buffer_size: usize,
    future_atom: usize,
) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let socket_id = BuiltInTypes::untag(socket_id);
    let buffer_size = BuiltInTypes::untag(buffer_size);
    let future_atom = BuiltInTypes::untag(future_atom);
    let runtime = get_runtime().get_mut();

    trace!(
        "tcp",
        "tcp_read_async: loop={} socket={} buf_size={} future_atom={}",
        loop_id,
        socket_id,
        buffer_size,
        future_atom
    );
    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };
    let op_id = event_loop.next_tcp_op_id();
    match event_loop.submit_tcp_op(crate::runtime::TcpOperation::Read {
        socket_id,
        buffer_size,
        future_atom,
        op_id,
    }) {
        Ok(()) => BuiltInTypes::Int.tag(op_id as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Start a write operation on a TCP socket
/// Returns op_id on success (operation started), -1 on error
/// In Threaded mode: submits operation, returns 0 on success
pub extern "C" fn tcp_write_async(
    stack_pointer: usize,
    _frame_pointer: usize,
    loop_id: usize,
    socket_id: usize,
    data: usize,
    future_atom: usize,
) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let socket_id = BuiltInTypes::untag(socket_id);
    let future_atom = BuiltInTypes::untag(future_atom);
    let runtime = get_runtime().get_mut();

    // Get the data as bytes - handle both string literals and heap strings
    let data_str = runtime.get_string(stack_pointer, data);
    let data_bytes = data_str.as_bytes().to_vec();

    trace!(
        "tcp",
        "tcp_write_async: loop={} socket={} data_len={} future_atom={}",
        loop_id,
        socket_id,
        data_bytes.len(),
        future_atom
    );
    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };
    let op_id = event_loop.next_tcp_op_id();
    match event_loop.submit_tcp_op(crate::runtime::TcpOperation::Write {
        socket_id,
        data: data_bytes,
        future_atom,
        op_id,
    }) {
        Ok(()) => BuiltInTypes::Int.tag(op_id as isize) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Close a TCP socket
pub extern "C" fn tcp_close(loop_id: usize, socket_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let socket_id = BuiltInTypes::untag(socket_id);
    trace!("tcp", "tcp_close: loop={} socket={}", loop_id, socket_id);
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };
    match event_loop.submit_tcp_op(crate::runtime::TcpOperation::Close { socket_id }) {
        Ok(()) => BuiltInTypes::Int.tag(1) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Close a TCP listener
pub extern "C" fn tcp_close_listener(loop_id: usize, listener_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let listener_id = BuiltInTypes::untag(listener_id);
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };
    match event_loop.submit_tcp_op(crate::runtime::TcpOperation::CloseListener { listener_id }) {
        Ok(()) => BuiltInTypes::Int.tag(1) as usize,
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// Check if there are any completed TCP results
/// Returns the count of pending results
pub extern "C" fn tcp_results_count(loop_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };
    let count = event_loop.tcp_results_count();
    if count > 0 {
        trace!("tcp", "tcp_results_count: loop={} count={}", loop_id, count);
    }
    BuiltInTypes::Int.tag(count as isize) as usize
}

/// Pop and get the next completed TCP result type
/// Returns 0 if no results, otherwise:
/// 1=connect-ok, 2=connect-err, 3=accept-ok, 4=accept-err,
/// 5=read-ok, 6=read-err, 7=write-ok, 8=write-err
/// The result stays "current" until the next call
pub extern "C" fn tcp_result_pop(loop_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();

    use crate::runtime::TcpResult;

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };
    let maybe_result = event_loop.pop_tcp_result();

    match maybe_result {
        None => BuiltInTypes::Int.tag(0) as usize,
        #[allow(unused_variables)]
        Some(result) => {
            let type_code = match &result {
                TcpResult::ConnectOk {
                    socket_id,
                    future_atom,
                    ..
                } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop: ConnectOk socket={} future_atom={}",
                        socket_id,
                        future_atom
                    );
                    1
                }
                TcpResult::ConnectErr {
                    error, future_atom, ..
                } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop: ConnectErr error={} future_atom={}", error, future_atom
                    );
                    2
                }
                TcpResult::AcceptOk {
                    socket_id,
                    listener_id,
                    future_atom,
                    ..
                } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop: AcceptOk socket={} listener={} future_atom={}",
                        socket_id,
                        listener_id,
                        future_atom
                    );
                    3
                }
                TcpResult::AcceptErr {
                    error, future_atom, ..
                } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop: AcceptErr error={} future_atom={}", error, future_atom
                    );
                    4
                }
                TcpResult::ReadOk {
                    data, future_atom, ..
                } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop: ReadOk data_len={} future_atom={}",
                        data.len(),
                        future_atom
                    );
                    5
                }
                TcpResult::ReadErr {
                    error, future_atom, ..
                } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop: ReadErr error={} future_atom={}", error, future_atom
                    );
                    6
                }
                TcpResult::WriteOk {
                    bytes_written,
                    future_atom,
                    ..
                } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop: WriteOk bytes={} future_atom={}",
                        bytes_written,
                        future_atom
                    );
                    7
                }
                TcpResult::WriteErr {
                    error, future_atom, ..
                } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop: WriteErr error={} future_atom={}", error, future_atom
                    );
                    8
                }
            };
            event_loop.set_current_result(result);
            BuiltInTypes::Int.tag(type_code) as usize
        }
    }
}

/// Pop and get the next completed TCP result matching a specific future_atom
/// This is used in threaded mode to ensure each thread only gets its own results.
/// Returns 0 if no matching result, otherwise the type code (same as tcp_result_pop)
pub extern "C" fn tcp_result_pop_for_atom(loop_id: usize, future_atom: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let future_atom = BuiltInTypes::untag(future_atom);
    let runtime = get_runtime().get_mut();

    use crate::runtime::TcpResult;

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };
    let maybe_result = event_loop.pop_tcp_result_for_atom(future_atom);

    match maybe_result {
        None => BuiltInTypes::Int.tag(0) as usize,
        #[allow(unused_variables)]
        Some(result) => {
            let type_code = match &result {
                TcpResult::ConnectOk { socket_id, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_atom: ConnectOk socket={} future_atom={}",
                        socket_id,
                        future_atom
                    );
                    1
                }
                TcpResult::ConnectErr { error, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_atom: ConnectErr error={} future_atom={}",
                        error,
                        future_atom
                    );
                    2
                }
                TcpResult::AcceptOk {
                    socket_id,
                    listener_id,
                    ..
                } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_atom: AcceptOk socket={} listener={} future_atom={}",
                        socket_id,
                        listener_id,
                        future_atom
                    );
                    3
                }
                TcpResult::AcceptErr { error, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_atom: AcceptErr error={} future_atom={}",
                        error,
                        future_atom
                    );
                    4
                }
                TcpResult::ReadOk { data, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_atom: ReadOk data_len={} future_atom={} data={:?}",
                        data.len(),
                        future_atom,
                        std::str::from_utf8(&data[..data.len().min(120)]).unwrap_or("<binary>")
                    );
                    5
                }
                TcpResult::ReadErr { error, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_atom: ReadErr error={} future_atom={}",
                        error,
                        future_atom
                    );
                    6
                }
                TcpResult::WriteOk { bytes_written, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_atom: WriteOk bytes={} future_atom={}",
                        bytes_written,
                        future_atom
                    );
                    7
                }
                TcpResult::WriteErr { error, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_atom: WriteErr error={} future_atom={}",
                        error,
                        future_atom
                    );
                    8
                }
            };
            event_loop.set_current_result(result);
            BuiltInTypes::Int.tag(type_code) as usize
        }
    }
}

pub extern "C" fn tcp_result_pop_for_op_id(loop_id: usize, op_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let op_id = BuiltInTypes::untag(op_id);
    let runtime = get_runtime().get_mut();

    use crate::runtime::TcpResult;

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };
    let maybe_result = event_loop.pop_tcp_result_for_op_id(op_id);

    match maybe_result {
        None => BuiltInTypes::Int.tag(0) as usize,
        Some(result) => {
            let type_code = match &result {
                TcpResult::ConnectOk { socket_id, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_op_id: ConnectOk socket={} op_id={}", socket_id, op_id
                    );
                    1
                }
                TcpResult::ConnectErr { error, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_op_id: ConnectErr error={} op_id={}", error, op_id
                    );
                    2
                }
                TcpResult::AcceptOk {
                    socket_id,
                    listener_id,
                    ..
                } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_op_id: AcceptOk socket={} listener={} op_id={}",
                        socket_id,
                        listener_id,
                        op_id
                    );
                    3
                }
                TcpResult::AcceptErr { error, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_op_id: AcceptErr error={} op_id={}", error, op_id
                    );
                    4
                }
                TcpResult::ReadOk { data, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_op_id: ReadOk data_len={} op_id={} data={:?}",
                        data.len(),
                        op_id,
                        std::str::from_utf8(&data[..data.len().min(120)]).unwrap_or("<binary>")
                    );
                    5
                }
                TcpResult::ReadErr { error, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_op_id: ReadErr error={} op_id={}", error, op_id
                    );
                    6
                }
                TcpResult::WriteOk { bytes_written, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_op_id: WriteOk bytes={} op_id={}", bytes_written, op_id
                    );
                    7
                }
                TcpResult::WriteErr { error, .. } => {
                    trace!(
                        "tcp",
                        "tcp_result_pop_for_op_id: WriteErr error={} op_id={}", error, op_id
                    );
                    8
                }
            };
            event_loop.set_current_result_for_op_id(op_id, result);
            BuiltInTypes::Int.tag(type_code) as usize
        }
    }
}

/// Get the value (socket_id or bytes_written) from the current op_id-specific TCP result.
/// Consumes the stored inspected result for `op_id`.
pub extern "C" fn tcp_result_value_for_op_id(loop_id: usize, op_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let op_id = BuiltInTypes::untag(op_id);
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };

    use crate::runtime::TcpResult;

    let value = match event_loop.take_current_result_for_op_id(op_id) {
        None => 0,
        Some(result) => match result {
            TcpResult::ConnectOk { socket_id, .. } => socket_id,
            TcpResult::AcceptOk { socket_id, .. } => socket_id,
            TcpResult::WriteOk { bytes_written, .. } => bytes_written,
            _ => 0,
        },
    };
    BuiltInTypes::Int.tag(value as isize) as usize
}

/// Get the data/error string from the current op_id-specific TCP result.
/// Consumes the stored inspected result for `op_id`.
pub extern "C" fn tcp_result_data_for_op_id(
    stack_pointer: usize,
    loop_id: usize,
    op_id: usize,
) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let op_id = BuiltInTypes::untag(op_id);
    let runtime = get_runtime().get_mut();

    let data = {
        let event_loop = match runtime.event_loops.get(loop_id) {
            Some(el) => el,
            None => {
                return runtime
                    .allocate_string(stack_pointer, String::new())
                    .map(|t| t.into())
                    .unwrap_or(0);
            }
        };

        use crate::runtime::TcpResult;

        match event_loop.take_current_result_for_op_id(op_id) {
            None => String::new(),
            Some(result) => match result {
                TcpResult::ReadOk { data, .. } => String::from_utf8_lossy(&data).to_string(),
                TcpResult::ConnectErr { error, .. } => error,
                TcpResult::AcceptErr { error, .. } => error,
                TcpResult::ReadErr { error, .. } => error,
                TcpResult::WriteErr { error, .. } => error,
                _ => String::new(),
            },
        }
    };

    trace!(
        "tcp",
        "tcp_result_data_for_op_id: op_id={} data_len={} data={:?}",
        op_id,
        data.len(),
        &data[..data.len().min(80)]
    );
    let runtime = get_runtime().get_mut();
    runtime
        .allocate_string(stack_pointer, data)
        .map(|t| t.into())
        .unwrap_or(0)
}

/// Get the future_atom from the current TCP result
pub extern "C" fn tcp_result_future_atom(loop_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };

    use crate::runtime::TcpResult;

    let atom = match event_loop.current_result() {
        None => 0,
        Some(result) => match result {
            TcpResult::ConnectOk { future_atom, .. } => future_atom,
            TcpResult::ConnectErr { future_atom, .. } => future_atom,
            TcpResult::AcceptOk { future_atom, .. } => future_atom,
            TcpResult::AcceptErr { future_atom, .. } => future_atom,
            TcpResult::ReadOk { future_atom, .. } => future_atom,
            TcpResult::ReadErr { future_atom, .. } => future_atom,
            TcpResult::WriteOk { future_atom, .. } => future_atom,
            TcpResult::WriteErr { future_atom, .. } => future_atom,
        },
    };
    BuiltInTypes::Int.tag(atom as isize) as usize
}

/// Get the value (socket_id or bytes_written) from the current TCP result
/// For error results, returns 0
pub extern "C" fn tcp_result_value(loop_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };

    use crate::runtime::TcpResult;

    let value = match event_loop.current_result() {
        None => 0,
        Some(result) => match result {
            TcpResult::ConnectOk { socket_id, .. } => socket_id,
            TcpResult::AcceptOk { socket_id, .. } => socket_id,
            TcpResult::WriteOk { bytes_written, .. } => bytes_written,
            _ => 0,
        },
    };
    BuiltInTypes::Int.tag(value as isize) as usize
}

/// Get the data/error string from the current TCP result
/// Returns the data for ReadOk, error message for error results, empty string otherwise
pub extern "C" fn tcp_result_data(stack_pointer: usize, loop_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();

    let data = {
        let event_loop = match runtime.event_loops.get(loop_id) {
            Some(el) => el,
            None => {
                return runtime
                    .allocate_string(stack_pointer, String::new())
                    .map(|t| t.into())
                    .unwrap_or(0);
            }
        };

        use crate::runtime::TcpResult;

        match event_loop.current_result() {
            None => String::new(),
            Some(result) => match result {
                TcpResult::ReadOk { data, .. } => String::from_utf8_lossy(&data).to_string(),
                TcpResult::ConnectErr { error, .. } => error,
                TcpResult::AcceptErr { error, .. } => error,
                TcpResult::ReadErr { error, .. } => error,
                TcpResult::WriteErr { error, .. } => error,
                _ => String::new(),
            },
        }
    };

    trace!(
        "tcp",
        "tcp_result_data: data_len={} data={:?}",
        data.len(),
        &data[..data.len().min(80)]
    );
    let runtime = get_runtime().get_mut();
    runtime
        .allocate_string(stack_pointer, data)
        .map(|t| t.into())
        .unwrap_or(0)
}

/// Get the op_id from the current TCP result
/// Returns 0 if no result is current
pub extern "C" fn tcp_result_op_id(loop_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };

    use crate::runtime::TcpResult;

    let op_id = match event_loop.current_result() {
        None => 0,
        Some(result) => match result {
            TcpResult::ConnectOk { op_id, .. } => op_id,
            TcpResult::ConnectErr { op_id, .. } => op_id,
            TcpResult::AcceptOk { op_id, .. } => op_id,
            TcpResult::AcceptErr { op_id, .. } => op_id,
            TcpResult::ReadOk { op_id, .. } => op_id,
            TcpResult::ReadErr { op_id, .. } => op_id,
            TcpResult::WriteOk { op_id, .. } => op_id,
            TcpResult::WriteErr { op_id, .. } => op_id,
        },
    };
    BuiltInTypes::Int.tag(op_id as isize) as usize
}

/// Get the listener_id from the current TCP result (for AcceptOk results)
/// Returns 0 if not an AcceptOk result or no result is current
pub extern "C" fn tcp_result_listener_id(loop_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get_mut();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };

    use crate::runtime::TcpResult;

    let listener_id = match event_loop.current_result() {
        Some(TcpResult::AcceptOk { listener_id, .. }) => listener_id,
        _ => 0,
    };
    BuiltInTypes::Int.tag(listener_id as isize) as usize
}

// =============================================================================
// Timer Builtins
// =============================================================================

/// Set a timer that fires after delay_ms milliseconds
/// Returns timer_id on success, -1 on error
pub extern "C" fn timer_set(loop_id: usize, delay_ms: usize, future_atom: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let delay_ms = BuiltInTypes::untag(delay_ms) as u64;
    let future_atom = BuiltInTypes::untag(future_atom);
    let runtime = get_runtime().get();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };

    let timer_id = event_loop.timer_set(delay_ms, future_atom);
    trace!(
        "event-loop",
        "timer_set: loop={} delay={}ms future_atom={} timer_id={}",
        loop_id,
        delay_ms,
        future_atom,
        timer_id
    );
    BuiltInTypes::Int.tag(timer_id as isize) as usize
}

/// Cancel a timer by ID
/// Returns 1 if cancelled, 0 if not found
pub extern "C" fn timer_cancel(loop_id: usize, timer_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let timer_id = BuiltInTypes::untag(timer_id);
    let runtime = get_runtime().get();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };

    let cancelled = event_loop.timer_cancel(timer_id);
    BuiltInTypes::Int.tag(if cancelled { 1 } else { 0 }) as usize
}

/// Get the count of completed timers
pub extern "C" fn timer_completed_count(loop_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };

    let count = event_loop.completed_timers_len();
    BuiltInTypes::Int.tag(count as isize) as usize
}

/// Pop the next completed timer's future_atom
/// Returns 0 if no completed timers
pub extern "C" fn timer_pop_completed(loop_id: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let runtime = get_runtime().get();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };

    match event_loop.pop_completed_timer() {
        Some(future_atom) => BuiltInTypes::Int.tag(future_atom as isize) as usize,
        None => BuiltInTypes::Int.tag(0) as usize,
    }
}

/// Remove a completed timer entry matching `future_atom`.
/// Returns 1 if found and removed, 0 otherwise.
pub extern "C" fn timer_take_completed(loop_id: usize, future_atom: usize) -> usize {
    let loop_id = BuiltInTypes::untag(loop_id);
    let future_atom = BuiltInTypes::untag(future_atom);
    let runtime = get_runtime().get();

    let event_loop = match runtime.event_loops.get(loop_id) {
        Some(el) => el,
        None => return BuiltInTypes::Int.tag(0) as usize,
    };

    let removed = event_loop.take_completed_timer(future_atom);
    BuiltInTypes::Int.tag(if removed { 1 } else { 0 }) as usize
}
