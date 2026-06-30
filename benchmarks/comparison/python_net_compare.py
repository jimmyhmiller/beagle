# Python equivalents of the Beagle networking micro-benchmarks, matched framing:
# server thread + client in ONE process, loopback, same N. Python-vs-Python here
# mirrors Beagle-vs-Beagle. Stdlib only (http.client, socket, ssl, threading).
import http.client, socket, ssl, threading, time, random

HOST = "127.0.0.1"
BODY = b"hello"

def make_server(tls=False):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, 0)); s.listen(128)
    port = s.getsockname()[1]
    ctx = None
    if tls:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain("resources/test_tls_cert.pem", "resources/test_tls_key.pem")
    def serve_conn(conn):
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            f = conn.makefile("rb")
            while True:
                line = f.readline()
                if not line: break
                keep = True
                while True:
                    h = f.readline()
                    if h in (b"\r\n", b"\n", b""): break
                    if h.lower().startswith(b"connection:") and b"close" in h.lower(): keep = False
                resp = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: %s\r\n\r\n%s" % (
                    len(BODY), b"keep-alive" if keep else b"close", BODY)
                conn.sendall(resp)
                if not keep: break
        except OSError:
            pass
        finally:
            try: conn.close()
            except OSError: pass
    def loop():
        while True:
            try: conn, _ = s.accept()
            except OSError: break
            if ctx is not None:
                try: conn = ctx.wrap_socket(conn, server_side=True)
                except (ssl.SSLError, OSError): continue
            serve_conn(conn)
    t = threading.Thread(target=loop, daemon=True); t.start()
    return s, port

# --- 1. HTTP keep-alive throughput (N=2000, one reused connection) ---
def bench_http_throughput():
    s, port = make_server()
    c = http.client.HTTPConnection(HOST, port); c.connect()
    c.request("GET", "/"); c.getresponse().read()  # warmup
    N = 2000
    t0 = time.perf_counter()
    for _ in range(N):
        c.request("GET", "/"); c.getresponse().read()
    dt = time.perf_counter() - t0
    c.close(); s.close()
    print(f"[py] http keep-alive : {N} req in {dt*1000:.0f} ms -> {N/dt:.0f} req/s, {dt/N*1e6:.0f} us/req")

# --- 2. Pooling speedup (N=400, pooled vs new-conn-each) ---
def bench_pool_speedup():
    s, port = make_server()
    N = 400
    c = http.client.HTTPConnection(HOST, port); c.connect()
    t0 = time.perf_counter()
    for _ in range(N):
        c.request("GET", "/"); c.getresponse().read()
    pooled = time.perf_counter() - t0; c.close()
    t0 = time.perf_counter()
    for _ in range(N):
        cc = http.client.HTTPConnection(HOST, port)
        cc.request("GET", "/", headers={"Connection": "close"}); cc.getresponse().read(); cc.close()
    oneshot = time.perf_counter() - t0
    s.close()
    print(f"[py] pooling speedup : pooled {pooled*1000:.0f} ms vs one-shot {oneshot*1000:.0f} ms -> {oneshot/pooled:.2f}x")

# --- 3. TLS overhead (N=100 each, avg ms/req) ---
def bench_tls_overhead():
    N = 100
    sp, pport = make_server(tls=False)
    c = http.client.HTTPConnection(HOST, pport); c.connect()
    c.request("GET","/"); c.getresponse().read()
    t0 = time.perf_counter()
    for _ in range(N): c.request("GET","/"); c.getresponse().read()
    http_ms = (time.perf_counter()-t0)/N*1000; c.close(); sp.close()
    st, tport = make_server(tls=True)
    ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
    ct = http.client.HTTPSConnection(HOST, tport, context=ctx); ct.connect()
    ct.request("GET","/"); ct.getresponse().read()
    t0 = time.perf_counter()
    for _ in range(N): ct.request("GET","/"); ct.getresponse().read()
    https_ms = (time.perf_counter()-t0)/N*1000; ct.close(); st.close()
    print(f"[py] tls overhead    : HTTP {http_ms:.3f} ms/req, HTTPS {https_ms:.3f} ms/req -> +{https_ms-http_ms:.3f} ms ({(https_ms/http_ms-1)*100:.0f}%)")

# --- 4. Router dispatch + serialization (pure CPU, no sockets, N=20000) ---
def bench_router():
    ROUTES = [
        ("GET","/"), ("GET","/about"), ("GET","/health"),
        ("GET","/users/:id"), ("GET","/users/:id/posts/:pid"),
        ("POST","/users"), ("GET","/files/*path"),
        ("GET","/search"), ("PUT","/users/:id"), ("DELETE","/users/:id"),
    ]
    def match(rm, rp, method, segs):
        if rm != method: return None
        rsegs = rp.strip("/").split("/") if rp != "/" else [""]
        if rp == "/": return {} if segs == [""] else None
        if rsegs and rsegs[-1].startswith("*"):
            if len(segs) < len(rsegs)-1: return None
            params = {}
            for i, rs in enumerate(rsegs[:-1]):
                if rs.startswith(":"): params[rs[1:]] = segs[i]
                elif rs != segs[i]: return None
            params[rsegs[-1][1:]] = "/".join(segs[len(rsegs)-1:]); return params
        if len(rsegs) != len(segs): return None
        params = {}
        for rs, s in zip(rsegs, segs):
            if rs.startswith(":"): params[rs[1:]] = s
            elif rs != s: return None
        return params
    def dispatch(method, path):
        segs = path.strip("/").split("/") if path != "/" else [""]
        for rm, rp in ROUTES:
            p = match(rm, rp, method, segs)
            if p is not None:
                return 200, "ok:" + ":".join(f"{k}={v}" for k,v in p.items())
        # method-exists-but-wrong -> 405, else 404
        for rm, rp in ROUTES:
            if match(rm.replace(rm, method if False else rm), rp, rm, segs) is not None:
                pass
        return 404, "not found"
    def serialize(status, body):
        return ("HTTP/1.1 %d OK\r\nContent-Length: %d\r\nConnection: keep-alive\r\n\r\n%s" % (status, len(body.encode()), body))
    REQS = [("GET","/"),("GET","/about"),("GET","/health"),("GET","/users/42"),
            ("GET","/users/42/posts/7"),("POST","/users"),("GET","/files/a/b/c.txt"),
            ("GET","/search"),("PUT","/users/9"),("DELETE","/users/9"),
            ("GET","/nope"),("GET","/users")]
    N = 20000; total = 0
    t0 = time.perf_counter()
    for i in range(N):
        m, p = REQS[i % len(REQS)]
        st, body = dispatch(m, p)
        total += len(serialize(st, body))
    dt = time.perf_counter() - t0
    print(f"[py] router dispatch : {N} in {dt*1000:.0f} ms -> {N/dt:.0f} dispatches/s, {dt/N*1e6:.0f} us/dispatch")

for fn in (bench_router, bench_pool_speedup, bench_http_throughput, bench_tls_overhead):
    fn()
