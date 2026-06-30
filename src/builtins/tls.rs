//! TLS builtins (rustls + ring) — CLIENT and SERVER, both SANS-I/O.
//!
//! rustls runs purely on in-memory byte buffers and never touches a socket. The
//! Beagle `beagle.tls` layer owns the (async, non-blocking) socket and shuttles
//! bytes between it and these builtins:
//!   - `tls_outgoing(h)`      -> ciphertext rustls wants to send (write to socket)
//!   - `tls_feed(h, bytes)`   -> hand received ciphertext to rustls + process it
//!   - `tls_read_plain(h, n)` -> decrypted application plaintext ("" if none yet)
//!   - `tls_write_plain(h, d)`-> queue plaintext to encrypt (drained via tls_outgoing)
//!   - `tls_is_handshaking`/`tls_close_notify`/`tls_destroy`
//! A handle is a `rustls::Connection` (the enum unifying Client and Server), so
//! every byte-pumping op above is identical for both sides. `tls_client_create`
//! (webpki roots), `tls_client_create_insecure` (dev: skip cert verification),
//! `tls_server_config_create` (cert+key DER) and `tls_server_create` make handles.
//! Because all socket I/O is the cooperative async socket layer, TLS connect/
//! read/write PARK like plain TCP and inherit timeouts/cancellation/concurrency.

use super::*;
use crate::save_gc_context;

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{CryptoProvider, verify_tls12_signature, verify_tls13_signature};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{
    ClientConfig, ClientConnection, Connection, DigitallySignedStruct, RootCertStore, ServerConfig,
    ServerConnection, SignatureScheme,
};

static TLS_HANDLE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Registry of live rustls connections (Client OR Server — crypto state only; the
/// socket lives in Beagle as a TcpSocket). Each behind its own Mutex so concurrent
/// work on different connections doesn't serialize on the map lock.
static TLS_REGISTRY: LazyLock<Mutex<HashMap<u64, Arc<Mutex<Connection>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Registry of built server configs (cert chain + key), reusable across accepts.
static TLS_SERVER_CONFIGS: LazyLock<Mutex<HashMap<u64, Arc<ServerConfig>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Shared client config: the Mozilla webpki root store, ring crypto, no client
/// auth. Built once on first use.
static TLS_CONFIG: LazyLock<Arc<ClientConfig>> = LazyLock::new(|| {
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config =
        ClientConfig::builder_with_provider(rustls::crypto::ring::default_provider().into())
            .with_safe_default_protocol_versions()
            .expect("ring provides the default protocol versions")
            .with_root_certificates(roots)
            .with_no_client_auth();
    Arc::new(config)
});

/// A certificate verifier that skips TRUST-CHAIN validation but still verifies the
/// handshake signature against the presented cert's key (like `curl -k`). DEV/TEST
/// ONLY — exposed via `tls_client_create_insecure`. It defeats MITM protection.
#[derive(Debug)]
struct NoCertVerification(Arc<CryptoProvider>);

impl ServerCertVerifier for NoCertVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

/// Register a connection and return its tagged-Int handle.
fn register_connection(conn: Connection) -> usize {
    let key = TLS_HANDLE_COUNTER.fetch_add(1, Ordering::Relaxed);
    TLS_REGISTRY
        .lock()
        .unwrap()
        .insert(key, Arc::new(Mutex::new(conn)));
    BuiltInTypes::Int.tag(key as isize) as usize
}

/// Look up a live connection by handle.
fn lookup(handle: usize) -> Option<Arc<Mutex<Connection>>> {
    let key = BuiltInTypes::untag(handle) as u64;
    TLS_REGISTRY.lock().unwrap().get(&key).cloned()
}

/// tls-client-create(host) -> a connection handle (rustls client state for the
/// given SNI host), or -1 if the server name is invalid.
pub extern "C" fn tls_client_create(
    stack_pointer: usize,
    frame_pointer: usize,
    host: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let host_str = runtime.get_string(stack_pointer, host);

    let server_name = match ServerName::try_from(host_str.to_string()) {
        Ok(sn) => sn,
        Err(_) => return BuiltInTypes::Int.tag(-1) as usize,
    };
    let conn = match ClientConnection::new(TLS_CONFIG.clone(), server_name) {
        Ok(c) => c,
        Err(_) => return BuiltInTypes::Int.tag(-1) as usize,
    };
    register_connection(Connection::Client(conn))
}

/// tls-client-create-insecure(host) -> a client connection handle that SKIPS
/// certificate-chain verification (still does the handshake + signature checks).
/// DEV/TEST ONLY — defeats MITM protection. Returns -1 on an invalid server name.
pub extern "C" fn tls_client_create_insecure(
    stack_pointer: usize,
    frame_pointer: usize,
    host: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let host_str = runtime.get_string(stack_pointer, host);

    let server_name = match ServerName::try_from(host_str.to_string()) {
        Ok(sn) => sn,
        Err(_) => return BuiltInTypes::Int.tag(-1) as usize,
    };
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let config = ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .expect("ring provides the default protocol versions")
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(NoCertVerification(provider)))
        .with_no_client_auth();
    let conn = match ClientConnection::new(Arc::new(config), server_name) {
        Ok(c) => c,
        Err(_) => return BuiltInTypes::Int.tag(-1) as usize,
    };
    register_connection(Connection::Client(conn))
}

/// tls-server-config-create(cert_der, key_der) -> a reusable server config handle
/// (cert chain = the single DER cert; key = PKCS#8 DER), or -1 on a bad cert/key.
pub extern "C" fn tls_server_config_create(
    stack_pointer: usize,
    frame_pointer: usize,
    cert_der: usize,
    key_der: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let cert_bytes = runtime.get_string_bytes_vec(cert_der);
    let key_bytes = runtime.get_string_bytes_vec(key_der);

    let certs = vec![CertificateDer::from(cert_bytes)];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_bytes));
    let config =
        match ServerConfig::builder_with_provider(rustls::crypto::ring::default_provider().into())
            .with_safe_default_protocol_versions()
            .expect("ring provides the default protocol versions")
            .with_no_client_auth()
            .with_single_cert(certs, key)
        {
            Ok(c) => c,
            Err(_) => return BuiltInTypes::Int.tag(-1) as usize,
        };
    let key_id = TLS_HANDLE_COUNTER.fetch_add(1, Ordering::Relaxed);
    TLS_SERVER_CONFIGS
        .lock()
        .unwrap()
        .insert(key_id, Arc::new(config));
    BuiltInTypes::Int.tag(key_id as isize) as usize
}

/// tls-server-create(config_handle) -> a server connection handle for a freshly
/// accepted socket, or -1 if the config handle is unknown.
pub extern "C" fn tls_server_create(config_handle: usize) -> usize {
    let config_key = BuiltInTypes::untag(config_handle) as u64;
    let config = match TLS_SERVER_CONFIGS.lock().unwrap().get(&config_key).cloned() {
        Some(c) => c,
        None => return BuiltInTypes::Int.tag(-1) as usize,
    };
    let conn = match ServerConnection::new(config) {
        Ok(c) => c,
        Err(_) => return BuiltInTypes::Int.tag(-1) as usize,
    };
    register_connection(Connection::Server(conn))
}

/// tls-outgoing(handle) -> all ciphertext rustls currently wants to send, as a
/// byte-faithful string ("" if none). The caller writes these bytes to the socket.
pub extern "C" fn tls_outgoing(stack_pointer: usize, frame_pointer: usize, handle: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    match lookup(handle) {
        Some(arc) => {
            let mut out: Vec<u8> = Vec::new();
            {
                let mut conn = arc.lock().unwrap();
                while conn.wants_write() {
                    match conn.write_tls(&mut out) {
                        Ok(0) => break,
                        Ok(_) => {}
                        Err(_) => break,
                    }
                }
            }
            runtime
                .allocate_string_from_bytes(stack_pointer, &out)
                .map(|t| t.into())
                .unwrap_or(BuiltInTypes::Null.tag(0) as usize)
        }
        None => BuiltInTypes::Null.tag(0) as usize,
    }
}

/// tls-feed(handle, ciphertext) -> 0 on success, -1 on a TLS error (bad cert,
/// alert, malformed record). Hands received ciphertext to rustls and processes it.
pub extern "C" fn tls_feed(
    stack_pointer: usize,
    frame_pointer: usize,
    handle: usize,
    data: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let bytes = runtime.get_string_bytes_vec(data);

    match lookup(handle) {
        Some(arc) => {
            let mut conn = arc.lock().unwrap();
            let mut cursor: &[u8] = &bytes;
            while !cursor.is_empty() {
                match conn.read_tls(&mut cursor) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(_) => return BuiltInTypes::Int.tag(-1) as usize,
                }
                // Process after each read_tls so rustls's input buffer can't
                // overflow on a large feed.
                if conn.process_new_packets().is_err() {
                    return BuiltInTypes::Int.tag(-1) as usize;
                }
            }
            if conn.process_new_packets().is_err() {
                return BuiltInTypes::Int.tag(-1) as usize;
            }
            BuiltInTypes::Int.tag(0) as usize
        }
        None => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// tls-read-plain(handle, n) -> up to n bytes of decrypted application plaintext
/// as a byte-faithful string; "" when no plaintext is buffered yet (caller should
/// feed more ciphertext) or at clean end-of-stream; null on an invalid handle.
pub extern "C" fn tls_read_plain(
    stack_pointer: usize,
    frame_pointer: usize,
    handle: usize,
    n: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let count = BuiltInTypes::untag(n) as usize;

    match lookup(handle) {
        Some(arc) => {
            let mut buf = vec![0u8; count];
            let nread = {
                let mut conn = arc.lock().unwrap();
                conn.reader().read(&mut buf)
            };
            // Ok(k) -> k plaintext bytes; WouldBlock / other errors -> none yet.
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

/// tls-write-plain(handle, data) -> bytes queued, or -1. Queues plaintext to be
/// encrypted; the resulting ciphertext is drained via tls-outgoing.
pub extern "C" fn tls_write_plain(
    stack_pointer: usize,
    frame_pointer: usize,
    handle: usize,
    data: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let bytes = runtime.get_string_bytes_vec(data);

    match lookup(handle) {
        Some(arc) => {
            let mut conn = arc.lock().unwrap();
            match conn.writer().write_all(&bytes) {
                Ok(()) => BuiltInTypes::Int.tag(bytes.len() as isize) as usize,
                Err(_) => BuiltInTypes::Int.tag(-1) as usize,
            }
        }
        None => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// tls-is-handshaking(handle) -> 1 while the TLS handshake is in progress, 0 once
/// it is complete, -1 on an invalid handle.
pub extern "C" fn tls_is_handshaking(handle: usize) -> usize {
    match lookup(handle) {
        Some(arc) => {
            let conn = arc.lock().unwrap();
            let v = if conn.is_handshaking() { 1 } else { 0 };
            BuiltInTypes::Int.tag(v) as usize
        }
        None => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// tls-close-notify(handle) -> 0. Queues a close_notify alert (drain via
/// tls-outgoing and send before closing the socket).
pub extern "C" fn tls_close_notify(handle: usize) -> usize {
    match lookup(handle) {
        Some(arc) => {
            arc.lock().unwrap().send_close_notify();
            BuiltInTypes::Int.tag(0) as usize
        }
        None => BuiltInTypes::Int.tag(-1) as usize,
    }
}

/// tls-destroy(handle) -> null. Drops the rustls connection state.
pub extern "C" fn tls_destroy(handle: usize) -> usize {
    let key = BuiltInTypes::untag(handle) as u64;
    TLS_REGISTRY.lock().unwrap().remove(&key);
    BuiltInTypes::Null.tag(0) as usize
}
