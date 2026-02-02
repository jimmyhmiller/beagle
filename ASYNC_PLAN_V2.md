# Production-Ready Async Plan for Beagle (v2 - with mio)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Beagle Async API                      │
│  async/await, spawn, file-read, tcp-connect, etc.       │
├─────────────────────────────────────────────────────────┤
│                   Effect Handlers                        │
│  BlockingHandler, ThreadedHandler, EventLoopHandler     │
├─────────────────────────────────────────────────────────┤
│                    Event Loop (mio)                      │
│  Poll for socket readiness, timer expiry, task completion│
├──────────────────────┬──────────────────────────────────┤
│   Thread Pool        │         mio Registry             │
│   (File I/O)         │    (Network I/O, Timers)         │
│   Blocking ops       │    Non-blocking sockets          │
└──────────────────────┴──────────────────────────────────┘
```

**Key insight**: File I/O uses thread pool (standard practice - tokio does this too), Network I/O uses mio for true async.

---

## Phase 1: Foundation ✅ COMPLETE

- [x] Filesystem builtins (unlink, access, mkdir, rmdir, readdir, file_size)
- [x] Efficient await with condition variables
- [x] Basic future waiting infrastructure

---

## Phase 2: Event Loop Foundation with mio

### 2.1 Add mio dependency

```toml
# Cargo.toml
[dependencies]
mio = { version = "0.8", features = ["os-poll", "net"] }
```

### 2.2 EventLoop struct in runtime.rs

```rust
use mio::{Events, Poll, Interest, Token, Waker};
use std::collections::VecDeque;

pub struct EventLoop {
    poll: Poll,
    waker: Arc<Waker>,
    // Token -> callback mapping
    pending_ops: HashMap<Token, PendingOperation>,
    next_token: usize,
    // Thread pool for blocking file I/O
    file_io_pool: ThreadPool,
    // Completed operations ready to resume
    completed: VecDeque<CompletedOperation>,
}

pub enum PendingOperation {
    TcpConnect { future_atom: usize },
    TcpRead { socket: TcpStream, future_atom: usize, buffer: Vec<u8> },
    TcpWrite { socket: TcpStream, future_atom: usize, data: Vec<u8> },
    Timer { deadline: Instant, future_atom: usize },
    FileOp { future_atom: usize }, // Completed by thread pool
}
```

### 2.3 Thread Pool for File I/O

```rust
pub struct ThreadPool {
    workers: Vec<JoinHandle<()>>,
    sender: mpsc::Sender<FileTask>,
    waker: Arc<Waker>, // To wake event loop when file op completes
}

pub struct FileTask {
    op: FileOperation,
    future_atom: usize,
    result_sender: mpsc::Sender<FileResult>,
}

pub enum FileOperation {
    Read { path: String },
    Write { path: String, content: Vec<u8> },
    Delete { path: String },
    Stat { path: String },
    ReadDir { path: String },
}
```

### 2.4 New Builtins

- `event_loop_create()` → EventLoop handle
- `event_loop_run_once(loop, timeout_ms)` → Process events
- `event_loop_wake(loop)` → Wake from another thread
- `get_cpu_count()` → Number of CPU cores

### 2.5 Files to Modify
- `Cargo.toml` - Add mio dependency
- `src/runtime.rs` - Add EventLoop and ThreadPool
- `src/builtins.rs` - Add event loop builtins

---

## Phase 3: Network I/O with mio

### 3.1 TCP Builtins (non-blocking via mio)

```rust
// All return immediately, register with event loop
fn tcp_connect_async(loop, host, port, future_atom) → Token
fn tcp_listen(loop, host, port, backlog) → Listener handle
fn tcp_accept_async(loop, listener, future_atom) → Token
fn tcp_read_async(loop, socket, n, future_atom) → Token
fn tcp_write_async(loop, socket, data, future_atom) → Token
fn tcp_close(socket)
```

### 3.2 UDP Builtins

```rust
fn udp_bind(loop, host, port) → Socket handle
fn udp_send_to_async(loop, socket, data, host, port, future_atom) → Token
fn udp_recv_from_async(loop, socket, n, future_atom) → Token
```

### 3.3 Beagle API in beagle.async.bg

```beagle
// High-level API wraps the builtins
fn tcp-connect(host, port) {
    perform Async.TcpConnect { host: host, port: port }
}

fn tcp-read(socket, n) {
    perform Async.TcpRead { socket: socket, n: n }
}
```

---

## Phase 4: EventLoopHandler

### 4.1 New Handler in beagle.async.bg

```beagle
struct EventLoopHandler {
    event_loop,
    pending_futures  // Map of Token -> Future
}

fn create-event-loop-handler() {
    EventLoopHandler {
        event_loop: core/event-loop-create(),
        pending_futures: map()
    }
}

extend EventLoopHandler with effect/Handler(Async) {
    fn handle(self, op, resume) {
        match op {
            // File ops go to thread pool
            Async.ReadFile { path } => {
                let future = make-future(FutureState.Running {})
                core/file-read-async(self.event_loop, path, future.state_atom)
                resume(future)
            },

            // Network ops use mio directly
            Async.TcpConnect { host, port } => {
                let future = make-future(FutureState.Running {})
                core/tcp-connect-async(self.event_loop, host, port, future.state_atom)
                resume(future)
            },

            // Await processes events until future resolves
            Async.Await { future } => {
                run-until-resolved(self.event_loop, future)
            },

            // ... other operations
        }
    }
}

fn run-until-resolved(event_loop, future) {
    loop {
        let state = future-state(future)
        match state {
            FutureState.Resolved { value } => break(value),
            FutureState.Rejected { error } => throw(error),
            _ => {
                // Process events (network, timers, file completions)
                core/event-loop-run-once(event_loop, 50)
            }
        }
    }
}
```

---

## Phase 5: Structured Concurrency

### 5.1 Cancellation Support

```beagle
enum FutureState {
    Pending {},
    Running {},
    Resolved { value },
    Rejected { error },
    Cancelled {}  // NEW
}

struct CancellationToken { cancelled_atom }

fn cancel!(token) {
    reset!(token.cancelled_atom, true)
}

fn cancelled?(token) {
    deref(token.cancelled_atom)
}
```

### 5.2 Task Scopes

```beagle
fn with-scope(body) {
    let scope = TaskScope {
        token: CancellationToken { cancelled_atom: atom(false) },
        children: atom([])
    }
    try {
        body(scope)
    } finally {
        // Cancel all children and wait for them
        cancel!(scope.token)
        await-all-ignore-errors(deref(scope.children))
    }
}

fn spawn-in-scope(scope, thunk) {
    let future = async(fn() {
        if cancelled?(scope.token) {
            throw("Cancelled")
        }
        thunk()
    })
    reset!(scope.children, push(deref(scope.children), future))
    future
}
```

---

## Phase 6: Timers

### 6.1 Timer Builtins

```rust
fn timer_set(loop, delay_ms, future_atom) → Token
fn timer_cancel(loop, token)
```

### 6.2 Beagle API

```beagle
fn sleep-async(ms) {
    perform Async.Sleep { ms: ms }
}

fn with-timeout(ms, thunk) {
    let timer_future = async-sleep(ms)
    let work_future = async(thunk)

    let result = await-first([timer_future, work_future])
    match result {
        RaceResult.Ok { value, index } => {
            if index == 0 {
                TimeoutResult.TimedOut {}
            } else {
                TimeoutResult.Ok { value: value }
            }
        },
        RaceResult.AllFailed { errors } => throw(first(errors))
    }
}
```

---

## Phase 7: Process Spawning

### 7.1 Builtins

```rust
fn spawn_process(cmd, args) → Process handle
fn process_wait_async(loop, process, future_atom) → Token
fn process_kill(process)
fn process_stdin(process) → writable pipe
fn process_stdout(process) → readable pipe (registered with mio)
fn process_stderr(process) → readable pipe (registered with mio)
```

---

## Implementation Order

1. **Phase 2** (This PR): Event loop + thread pool foundation
   - Add mio dependency
   - Implement EventLoop and ThreadPool in Rust
   - Basic builtins for event loop management

2. **Phase 3**: Network I/O
   - TCP connect/read/write/close
   - UDP send/recv
   - Test with echo server

3. **Phase 4**: EventLoopHandler in Beagle
   - Wire up the effect handler to use event loop
   - Migrate file I/O to use thread pool

4. **Phase 5**: Structured concurrency
   - Cancellation tokens
   - Task scopes

5. **Phase 6**: Timers

6. **Phase 7**: Process spawning

---

## Testing Strategy

### Unit Tests
- Event loop processes events correctly
- Thread pool executes file ops
- Sockets connect and transfer data

### Integration Tests
- `async_event_loop_test.bg` - Basic event loop usage
- `async_tcp_echo_test.bg` - TCP client/server
- `async_file_pool_test.bg` - File ops via thread pool
- `async_timeout_test.bg` - Timeouts work correctly

### Benchmarks
- Compare EventLoopHandler vs ThreadedHandler
- Measure latency and throughput for network ops
- File I/O performance with different pool sizes

---

## Verification After Each Phase

1. `cargo run -- --all-tests` - All existing tests pass
2. New phase-specific tests pass
3. `cargo clippy` - No warnings
4. `cargo fmt` - Code formatted
