//! TLS client builtins (rustls + ring + the Mozilla webpki root store).
//!
//! These are SYNCHRONOUS, blocking builtins: `tls_connect` opens a blocking
//! std `TcpStream`, completes the TLS handshake, and stashes the rustls
//! `StreamOwned` in a global registry keyed by an integer handle. `tls_read`/
//! `tls_write` then do blocking plaintext I/O on that handle (rustls handles
//! record framing + encryption). They block the calling OS thread for the
//! duration of the network I/O — fine for a client request on the cooperative
//! scheduler (the program is waiting on the response anyway); run on a dedicated
//! `thread()` if you need concurrency. Full event-loop integration (parking the
//! continuation like the socket layer) is a later step.

use super::*;
use crate::save_gc_context;

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use rustls::pki_types::ServerName;
use rustls::{ClientConfig, ClientConnection, RootCertStore, StreamOwned};

type TlsStream = StreamOwned<ClientConnection, TcpStream>;

static TLS_HANDLE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Registry of live TLS connections. Each is behind its own Mutex so concurrent
/// I/O on different connections doesn't serialize on the map lock.
static TLS_REGISTRY: LazyLock<Mutex<HashMap<u64, Arc<Mutex<TlsStream>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Shared client config: the Mozilla webpki root store, ring crypto, no client
/// auth. Built once on first use.
static TLS_CONFIG: LazyLock<Arc<ClientConfig>> = LazyLock::new(|| {
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = ClientConfig::builder_with_provider(
        rustls::crypto::ring::default_provider().into(),
    )
    .with_safe_default_protocol_versions()
    .expect("ring provides the default protocol versions")
    .with_root_certificates(roots)
    .with_no_client_auth();
    Arc::new(config)
});

/// Open a TCP connection and complete the TLS handshake (fails fast on
/// connect/cert/handshake errors).
fn tls_open(host: &str, port: u16) -> Result<TlsStream, String> {
    let sock = TcpStream::connect((host, port)).map_err(|e| e.to_string())?;
    sock.set_nodelay(true).ok();
    let server_name = ServerName::try_from(host.to_string())
        .map_err(|_| format!("invalid TLS server name: {}", host))?;
    let conn =
        ClientConnection::new(TLS_CONFIG.clone(), server_name).map_err(|e| e.to_string())?;
    let mut stream = StreamOwned::new(conn, sock);
    // Drive the handshake to completion now so a bad cert surfaces at connect.
    stream
        .conn
        .complete_io(&mut stream.sock)
        .map_err(|e| e.to_string())?;
    Ok(stream)
}

/// Look up a live connection by handle.
fn lookup(handle: usize) -> Option<Arc<Mutex<TlsStream>>> {
    let key = BuiltInTypes::untag(handle) as u64;
    TLS_REGISTRY.lock().unwrap().get(&key).cloned()
}

/// tls-connect(host, port) -> an integer connection handle, or -1 on failure.
pub extern "C" fn tls_connect(
    stack_pointer: usize,
    frame_pointer: usize,
    host: usize,
    port: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let host_str = runtime.get_string(stack_pointer, host);
    let port_n = BuiltInTypes::untag(port);

    match tls_open(&host_str, port_n as u16) {
        Ok(stream) => {
            let key = TLS_HANDLE_COUNTER.fetch_add(1, Ordering::Relaxed);
            TLS_REGISTRY
                .lock()
                .unwrap()
                .insert(key, Arc::new(Mutex::new(stream)));
            BuiltInTypes::Int.tag(key as isize) as usize
        }
        Err(_) => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// tls-write(handle, data) -> bytes written, or -1 on failure/invalid handle.
pub extern "C" fn tls_write(
    stack_pointer: usize,
    frame_pointer: usize,
    handle: usize,
    data: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let bytes = runtime.get_string_bytes_vec(data);

    match lookup(handle) {
        Some(stream) => {
            let mut s = stream.lock().unwrap();
            match s.write_all(&bytes).and_then(|_| s.flush()) {
                Ok(()) => BuiltInTypes::Int.tag(bytes.len() as isize) as usize,
                Err(_) => BuiltInTypes::Int.tag(-1) as usize,
            }
        }
        None => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// tls-read(handle, n) -> up to n plaintext bytes as a byte-faithful string;
/// "" at end-of-stream (clean close or error), null on an invalid handle.
pub extern "C" fn tls_read(
    stack_pointer: usize,
    frame_pointer: usize,
    handle: usize,
    n: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let count = BuiltInTypes::untag(n) as usize;

    match lookup(handle) {
        Some(stream) => {
            let mut buf = vec![0u8; count];
            // Read with the per-connection lock held, then release before we
            // allocate (allocation may GC).
            let nread = {
                let mut s = stream.lock().unwrap();
                s.read(&mut buf)
            };
            let slice: &[u8] = match nread {
                Ok(k) => &buf[..k],
                Err(_) => &[],
            };
            runtime
                .allocate_string_from_bytes(stack_pointer, slice)
                .map(|t| t.into())
                .unwrap_or(BuiltInTypes::Null.tag(0) as usize)
        }
        None => BuiltInTypes::Null.tag(0) as usize,
    }
}

/// tls-close(handle) -> null. Drops the connection (sends close_notify on drop).
pub extern "C" fn tls_close(
    stack_pointer: usize,
    frame_pointer: usize,
    handle: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let key = BuiltInTypes::untag(handle) as u64;
    TLS_REGISTRY.lock().unwrap().remove(&key);
    BuiltInTypes::Null.tag(0) as usize
}
