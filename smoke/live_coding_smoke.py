#!/usr/bin/env python3
"""Live-coding smoke test: hammer a running Beagle REPL server with the
operations live coding actually performs and verify the process NEVER
panics or aborts.

Not part of the regular test suite — run it once after runtime changes:

    python3 smoke/live_coding_smoke.py                 # full battery
    python3 smoke/live_coding_smoke.py --only fuzz --fuzz-ops 500
    python3 smoke/live_coding_smoke.py --list

Every scenario gets a fresh server process and a fresh temp workspace.
A scenario fails if:
  - the server's stderr contains a panic marker,
  - the server process dies while the scenario is still using it,
  - the server stops responding (protocol timeout), or
  - a scenario-specific assertion fails.

Failures keep the workspace + log on disk and print the path.

Scenario map (each tracks a crash class that has actually happened):
  sanity             basic evals, definitions, output capture
  disconnect-storm   clients vanish mid-eval (broken-pipe → stale
                     effect/prompt records → "no segment data" abort)
  throw-abort-resume resumable errors, nested suspensions, resume w/ code
  output-capture     binding(core/out) must be restored after throws
  persist-churn      multi-def reflect/persist storms against a hot game
                     loop, incl. failing + corrective persists (stale
                     GC-context "shift without enclosing reset" abort)
  effect-churn       throws unwinding past `handle` blocks, then perform
  thread-churn       threads that throw/catch concurrently with evals
  gc-pressure        allocation storms interleaved with throwing evals
  main-crash         main-thread crash → main-status/main-resume recovery
  fuzz               seeded random op stream over everything above
"""

import argparse
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BEAG = os.path.join(REPO_ROOT, "target", "release", "beag")

PANIC_MARKERS = [
    "panicked at",
    "non-unwinding panic",
    "fatal runtime error",
    "Segmentation fault",
]

# ---------------------------------------------------------------------------
# Fixtures (written into each scenario's temp workspace)
# ---------------------------------------------------------------------------

SMOKEGAME_BG = """namespace smokegame

use beagle.core as core
use beagle.effect as effect

struct Stats {
    hp
    mana
}

fn make_stats() {
    Stats { hp: 100, mana: 50 }
}

fn tick_value(s) {
    s.hp + s.mana
}

// -- effect machinery for the effect-churn scenario --------------------

enum Ping {
    Ask {}
}

struct OuterH {}
struct InnerH {}

extend OuterH with effect/Handler(Ping) {
    fn handle(self, op, resume) {
        match op {
            Ping.Ask {} => resume("outer")
        }
    }
}

extend InnerH with effect/Handler(Ping) {
    fn handle(self, op, resume) {
        match op {
            Ping.Ask {} => resume("inner")
        }
    }
}

/// Throw past an inner handle, then perform: must reach OuterH.
fn effect_roundtrip() {
    handle effect/Handler(Ping) with OuterH {} {
        let r = try {
            handle effect/Handler(Ping) with InnerH {} {
                throw("boom")
            }
        } catch (e, resume) {
            "caught"
        }
        perform Ping.Ask {}
    }
}

/// Spawn n threads that each throw + catch, join via atom.
fn thread_churn(n) {
    let done = atom(0)
    let mut i = 0
    while i < n {
        thread(fn() {
            let r = try { throw("t") } catch (e, resume) { 1 }
            swap!(done, fn(x) { x + r })
        })
        i = i + 1
    }
    while deref(done) < n {
        core/sleep(1)
    }
    deref(done)
}

/// Build a big vector to force GC activity.
fn alloc_churn(n) {
    let mut v = []
    let mut i = 0
    while i < n {
        v = push(v, "item ${i}")
        i = i + 1
    }
    length(v)
}

/// Game loop: constantly constructs + reads the structs the persist
/// scenarios redefine. Per-tick errors (transient shape mismatches
/// mid-persist) are caught and counted, never fatal.
fn main() {
    let mut errs = 0
    let mut i = 0
    while true {
        let r = try {
            tick_value(make_stats())
        } catch (e, resume) {
            errs = errs + 1
            0
        }
        core/sleep(2)
        i = i + 1
    }
    0
}
"""

CRASHMAIN_BG = """namespace crashmain

use beagle.core as core

fn main() {
    let mut i = 0
    while i < 1000000 {
        if i == 100 {
            throw("main crashed at tick ${i}")
        }
        core/sleep(2)
        i = i + 1
    }
    0
}
"""

GAME_RUNNER_TEMPLATE = """namespace __smoke_runner

use beagle.repl-main as repl-main
use {target_ns} as target

fn main() {{
    eval("namespace {target_ns}")
    repl-main/run-with-repl("127.0.0.1", {port}, fn() {{
        target/main()
    }})
}}
"""

STANDALONE_TEMPLATE = """namespace __smoke_standalone

use beagle.repl as repl

fn main() {{
    repl/start-repl-server("127.0.0.1", {port})
}}
"""

# Full-file persist payloads for namespace `smokegame`. Each contains ALL
# three Stats defs so runtime + disk stay mutually consistent.
PERSIST_A = (
    "struct Stats {\n    hp\n    mana\n}\n\n"
    "fn make_stats() {\n    Stats { hp: 100, mana: 50 }\n}\n\n"
    "fn tick_value(s) {\n    s.hp + s.mana\n}"
)
PERSIST_B = (
    "struct Stats {\n    hp\n    mana\n    stamina\n}\n\n"
    "fn make_stats() {\n    Stats { hp: 100, mana: 50, stamina: 25 }\n}\n\n"
    "fn tick_value(s) {\n    s.hp + s.mana + s.stamina\n}"
)
# First def splices cleanly (struct shape change!), second fails to
# compile — exercises write-then-throw and corrective-persist recovery.
PERSIST_BAD = (
    "struct Stats {\n    hp\n    mana\n    broken_field\n}\n\n"
    "fn make_stats() { undefined_var_smoke_qq }"
)

EXPECT_A = "150"
EXPECT_B = "175"


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

class SmokeFail(Exception):
    pass


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Server:
    """A `beag run` child whose combined output is captured to a log."""

    def __init__(self, beag, bg_file, include_dir=None, env=None, log_path=None):
        cmd = [beag, "run"]
        if include_dir:
            cmd += ["-I", include_dir]
        cmd.append(bg_file)
        self.log_path = log_path
        self.log_file = open(log_path, "wb")
        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        self.proc = subprocess.Popen(
            cmd, stdout=self.log_file, stderr=subprocess.STDOUT, env=full_env
        )

    def alive(self):
        return self.proc.poll() is None

    def stop(self):
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.log_file.close()

    def log_text(self):
        with open(self.log_path, "rb") as f:
            return f.read().decode("utf-8", errors="replace")

    def panic_lines(self):
        text = self.log_text()
        return [
            line
            for line in text.splitlines()
            if any(m in line for m in PANIC_MARKERS)
        ]


def wait_for_port(port, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=2)
            s.close()
            return
        except OSError:
            time.sleep(0.2)
    raise SmokeFail(f"server never opened port {port}")


class Client:
    """Line-JSON REPL client. Requests on one connection are processed
    sequentially by the server, so reading until a terminal status is
    enough."""

    def __init__(self, port):
        self.port = port
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.buf = b""

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def send_raw(self, msg):
        self.sock.sendall((json.dumps(msg) + "\n").encode())

    def _read_line(self, timeout):
        self.sock.settimeout(timeout)
        deadline = time.time() + timeout
        while b"\n" not in self.buf:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise SmokeFail("protocol timeout waiting for response line")
            self.sock.settimeout(remaining)
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                raise SmokeFail("protocol timeout waiting for response line")
            if not chunk:
                raise SmokeFail("server closed the connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line

    def request(self, op, timeout=30, **fields):
        """Send one request and collect messages until a terminal status
        (["done"] or ["error", ...]). Returns the list of JSON messages."""
        msg = {"op": op}
        msg.update(fields)
        self.send_raw(msg)
        out = []
        while True:
            line = self._read_line(timeout)
            if not line.strip():
                continue
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(resp)
            status = resp.get("status")
            if status and ("done" in status or "error" in status):
                return out

    # -- convenience -----------------------------------------------------

    def eval(self, code, session="smoke", req_id=None, timeout=30):
        rid = req_id or f"e{random.randrange(1 << 30)}"
        return self.request(
            "eval", id=rid, session=session, code=code, timeout=timeout
        )

    def abort(self, session="smoke", timeout=30):
        return self.request(
            "abort", id=f"a{random.randrange(1 << 30)}", session=session,
            timeout=timeout,
        )

    def resume(self, code, session="smoke", timeout=30):
        return self.request(
            "resume", id=f"r{random.randrange(1 << 30)}", session=session,
            code=code, timeout=timeout,
        )

    def drain_suspensions(self, session="smoke", limit=64):
        """Abort until the session reports nothing left to abort."""
        for _ in range(limit):
            msgs = self.abort(session=session)
            if any(
                "error" in (m.get("status") or []) for m in msgs
            ):
                return


def status_of(msgs):
    for m in msgs:
        if m.get("status"):
            return m["status"]
    return None


def value_of(msgs):
    for m in msgs:
        if "value" in m:
            return m["value"]
    return None


def out_of(msgs):
    return "\n".join(m["out"] for m in msgs if "out" in m)


def ex_of(msgs):
    for m in msgs:
        if "ex" in m:
            return m["ex"]
    return None


def expect(cond, why):
    if not cond:
        raise SmokeFail(why)


# ---------------------------------------------------------------------------
# Workspace / scenario plumbing
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, beag, workspace, fuzz_ops, seed):
        self.beag = beag
        self.workspace = workspace
        self.fuzz_ops = fuzz_ops
        self.seed = seed
        self.port = None
        self.server = None

    def write(self, rel, content):
        path = os.path.join(self.workspace, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return path

    def start_standalone(self):
        self.port = free_port()
        bg = self.write(
            "standalone.bg", STANDALONE_TEMPLATE.format(port=self.port)
        )
        self.server = Server(
            self.beag, bg, log_path=os.path.join(self.workspace, "server.log")
        )
        wait_for_port(self.port)

    def start_game(self, target_ns="smokegame"):
        self.port = free_port()
        self.write("src/smokegame.bg", SMOKEGAME_BG)
        self.write("src/crashmain.bg", CRASHMAIN_BG)
        runner = self.write(
            "runner.bg",
            GAME_RUNNER_TEMPLATE.format(target_ns=target_ns, port=self.port),
        )
        self.server = Server(
            self.beag,
            runner,
            include_dir=os.path.join(self.workspace, "src"),
            log_path=os.path.join(self.workspace, "server.log"),
        )
        wait_for_port(self.port)

    def client(self):
        return Client(self.port)

    def check_alive(self):
        if not self.server.alive():
            raise SmokeFail(
                f"server process died (exit {self.server.proc.returncode})"
            )


def persist_code(text, ns="smokegame"):
    # Self-contained: the eval namespace can drift under churn (persist
    # fragment compiles switch the compiler's current namespace), so
    # never rely on an alias registered by an earlier eval.
    return (
        "use beagle.reflect as reflect\n"
        f'reflect/persist("{ns}", {json.dumps(text)})'
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_sanity(ctx):
    ctx.start_standalone()
    c = ctx.client()
    msgs = c.eval("1 + 2", timeout=60)
    expect(value_of(msgs) == "3", f"1+2 != 3: {msgs}")
    msgs = c.eval('fn smoke_double(x) { x * 2 }\nsmoke_double(21)')
    expect(value_of(msgs) == "42", f"fn define+call failed: {msgs}")
    msgs = c.eval('println("hello smoke")\n7')
    expect("hello smoke" in out_of(msgs), f"output capture failed: {msgs}")
    expect(value_of(msgs) == "7", f"value after println failed: {msgs}")
    msgs = c.eval('"a" ++ "b" ++ "${1 + 1}"')
    expect(value_of(msgs) == "ab2", f"string interp failed: {msgs}")
    c.close()


def scenario_disconnect_storm(ctx):
    ctx.start_standalone()
    # Warm up the session (first eval compiles slowly).
    c = ctx.client()
    c.eval("0", session="x", timeout=60)
    c.close()
    for i in range(40):
        s = socket.create_connection(("127.0.0.1", ctx.port), timeout=5)
        req = {
            "op": "eval", "id": str(i), "session": "x",
            "code": f'println("work{i}"); {i} * 2',
        }
        s.sendall((json.dumps(req) + "\n").encode())
        s.close()  # abrupt: never read the response
        time.sleep(0.05)
        ctx.check_alive()
    # Server must still serve a clean client correctly.
    time.sleep(0.5)
    c = ctx.client()
    msgs = c.eval("40 + 2", session="x")
    expect(value_of(msgs) == "42", f"post-storm eval failed: {msgs}")
    c.close()


def scenario_throw_abort_resume(ctx):
    ctx.start_standalone()
    c = ctx.client()
    c.eval("0", timeout=60)

    # Throw → resumable error → abort.
    msgs = c.eval('throw("smoke boom")')
    expect(
        status_of(msgs) == ["error", "resumable"],
        f"throw not resumable: {msgs}",
    )
    expect("smoke boom" in (ex_of(msgs) or ""), f"missing ex: {msgs}")
    c.abort()

    # Throw → resume with a replacement value; body continues.
    msgs = c.eval('let x = throw("need a value")\nx + 1')
    expect(status_of(msgs) == ["error", "resumable"], f"not suspended: {msgs}")
    msgs = c.resume("41")
    expect(value_of(msgs) == "42", f"resume value wrong: {msgs}")

    # Nested suspensions: throw while suspended, then unwind both.
    msgs = c.eval('throw("outer")')
    expect(status_of(msgs) == ["error", "resumable"], f"no outer: {msgs}")
    msgs = c.eval('throw("inner")')
    expect(status_of(msgs) == ["error", "resumable"], f"no inner: {msgs}")
    depth = next(
        (m.get("suspend-depth") for m in msgs if "suspend-depth" in m), None
    )
    expect(depth == 2, f"expected suspend-depth 2, got {depth}: {msgs}")
    c.drain_suspensions()

    # Errors from runtime (not just explicit throw) are also resumable.
    msgs = c.eval('struct SmokeS { a }\n(SmokeS { a: 1 }).nope')
    expect(
        status_of(msgs) == ["error", "resumable"],
        f"field error not resumable: {msgs}",
    )
    c.abort()

    msgs = c.eval("1 + 1")
    expect(value_of(msgs) == "2", f"session wedged after throws: {msgs}")
    c.close()


def scenario_output_capture(ctx):
    ctx.start_standalone()
    c = ctx.client()
    c.eval("0", timeout=60)
    for i in range(10):
        # A throwing eval whose body printed first. The session wraps
        # evals in binding(core/out = buf); the unwind must restore it.
        msgs = c.eval(f'println("pre-throw {i}")\nthrow("boom {i}")')
        expect(
            status_of(msgs) == ["error", "resumable"],
            f"round {i}: not suspended: {msgs}",
        )
        expect(
            f"pre-throw {i}" in out_of(msgs),
            f"round {i}: lost pre-throw output: {msgs}",
        )
        c.abort()
        # Output of the NEXT eval must still be captured and delivered.
        msgs = c.eval(f'println("after {i}")\n{i}')
        expect(
            f"after {i}" in out_of(msgs),
            f"round {i}: output swallowed after throw (binding leak): {msgs}",
        )
        expect(value_of(msgs) == str(i), f"round {i}: bad value: {msgs}")
    c.close()


def scenario_persist_churn(ctx):
    ctx.start_game()
    c = ctx.client()
    c.eval("0", timeout=60)

    def do_persist(text, expect_ok):
        msgs = c.eval(persist_code(text), timeout=60)
        st = status_of(msgs)
        if expect_ok:
            expect(st == ["done"], f"persist failed unexpectedly: {msgs}")
        else:
            expect(
                st == ["error", "resumable"],
                f"bad persist did not error: {msgs}",
            )
            c.abort()

    rounds = 30
    for i in range(rounds):
        ctx.check_alive()
        if i % 5 == 4:
            # Failing persist (writes broken text to disk), then a
            # corrective persist must recover.
            do_persist(PERSIST_BAD, expect_ok=False)
            do_persist(PERSIST_A, expect_ok=True)
            want = EXPECT_A
        elif i % 2 == 0:
            do_persist(PERSIST_B, expect_ok=True)
            want = EXPECT_B
        else:
            do_persist(PERSIST_A, expect_ok=True)
            want = EXPECT_A
        msgs = c.eval("smokegame/tick_value(smokegame/make_stats())")
        expect(
            value_of(msgs) == want,
            f"round {i}: tick_value {value_of(msgs)} != {want}: {msgs}",
        )
    # Other defs in the file (shifted by every splice) must still work.
    msgs = c.eval("smokegame/effect_roundtrip()")
    expect(value_of(msgs) == "outer", f"shifted defs broken: {msgs}")
    c.close()


def scenario_effect_churn(ctx):
    ctx.start_game()
    c = ctx.client()
    c.eval("0", timeout=60)
    for i in range(25):
        msgs = c.eval("smokegame/effect_roundtrip()")
        expect(
            value_of(msgs) == "outer",
            f"round {i}: effect roundtrip got {value_of(msgs)}: {msgs}",
        )
        ctx.check_alive()
    c.close()


def scenario_thread_churn(ctx):
    ctx.start_game()
    c = ctx.client()
    c.eval("0", timeout=60)
    for i in range(10):
        msgs = c.eval("smokegame/thread_churn(8)", timeout=60)
        expect(
            value_of(msgs) == "8",
            f"round {i}: thread_churn got {value_of(msgs)}: {msgs}",
        )
        ctx.check_alive()
    c.close()


def scenario_gc_pressure(ctx):
    ctx.start_game()
    c = ctx.client()
    c.eval("0", timeout=60)
    for i in range(8):
        msgs = c.eval("smokegame/alloc_churn(20000)", timeout=120)
        expect(
            value_of(msgs) == "20000",
            f"round {i}: alloc_churn got {value_of(msgs)}: {msgs}",
        )
        msgs = c.eval(f'throw("gc round {i}")')
        expect(
            status_of(msgs) == ["error", "resumable"],
            f"round {i}: throw under pressure: {msgs}",
        )
        c.abort()
        ctx.check_alive()
    c.close()


def scenario_main_crash(ctx):
    ctx.start_game(target_ns="crashmain")
    c = ctx.client()
    c.eval("0", timeout=60)
    # crashmain throws at tick 100 (~200ms in). Poll main-status.
    deadline = time.time() + 30
    suspended = False
    while time.time() < deadline:
        msgs = c.request("main-status", id="ms")
        state = next(
            (m.get("main-thread") for m in msgs if "main-thread" in m), None
        )
        if state == "suspended":
            suspended = True
            break
        time.sleep(0.2)
    expect(suspended, "main thread never reported suspended")
    msgs = c.request("main-resume", id="mr", code="42")
    expect(status_of(msgs) == ["done"], f"main-resume failed: {msgs}")
    time.sleep(0.5)
    msgs = c.request("main-status", id="ms2")
    state = next(
        (m.get("main-thread") for m in msgs if "main-thread" in m), None
    )
    expect(state == "running", f"main thread not running after resume: {msgs}")
    # Server still evals fine.
    msgs = c.eval("2 + 2")
    expect(value_of(msgs) == "4", f"eval after main recovery: {msgs}")
    c.close()


def scenario_fuzz(ctx):
    rng = random.Random(ctx.seed)
    ctx.start_game()
    c = ctx.client()
    c.eval("0", session="fz", timeout=60)

    suspended = 0
    last_persist_good = PERSIST_A

    def op_good():
        k = rng.randrange(1000)
        msgs = c.eval(f"{k} + {k}", session="fz")
        expect(value_of(msgs) == str(2 * k), f"fuzz eval wrong: {msgs}")

    def op_bad_compile():
        nonlocal suspended
        msgs = c.eval("this is not ((( valid", session="fz")
        if status_of(msgs) == ["error", "resumable"]:
            suspended += 1

    def op_throw():
        nonlocal suspended
        msgs = c.eval(f'throw("fz {rng.randrange(1000)}")', session="fz")
        if status_of(msgs) == ["error", "resumable"]:
            suspended += 1

    def op_abort():
        nonlocal suspended
        if suspended > 0:
            c.abort(session="fz")
            suspended -= 1

    def op_resume():
        nonlocal suspended
        if suspended > 0:
            c.resume("0", session="fz")
            suspended -= 1

    def op_persist_good():
        nonlocal last_persist_good
        text = rng.choice([PERSIST_A, PERSIST_B])
        msgs = c.eval(persist_code(text), session="fz", timeout=60)
        expect(
            status_of(msgs) == ["done"], f"fuzz good persist failed: {msgs}"
        )
        last_persist_good = text

    def op_persist_bad():
        nonlocal suspended
        msgs = c.eval(persist_code(PERSIST_BAD), session="fz", timeout=60)
        if status_of(msgs) == ["error", "resumable"]:
            c.abort(session="fz")
        # Corrective persist so the workspace stays usable.
        msgs = c.eval(
            persist_code(last_persist_good), session="fz", timeout=60
        )
        expect(
            status_of(msgs) == ["done"],
            f"fuzz corrective persist failed: {msgs}",
        )

    def op_disconnect():
        nonlocal c, suspended
        # Fire an eval and vanish without reading the response.
        try:
            c.send_raw({
                "op": "eval", "id": "gone", "session": "fz",
                "code": 'println("zap"); 1',
            })
        except OSError:
            pass
        c.close()
        time.sleep(0.05)
        c = ctx.client()
        # The vanished eval may or may not have suspended anything; drain
        # to a known state.
        c.drain_suspensions(session="fz")
        suspended = 0

    def op_output():
        n = rng.randrange(1, 30)
        # `let mut` is not allowed at eval top level — wrap in a fn.
        msgs = c.eval(
            "fn __fz_out(n) {\n"
            "    let mut i = 0\n"
            "    while i < n {\n"
            '        println("line ${i}")\n'
            "        i = i + 1\n"
            "    }\n"
            "    i\n"
            "}\n"
            f"__fz_out({n})",
            session="fz",
        )
        expect(value_of(msgs) == str(n), f"fuzz output eval wrong: {msgs}")

    def op_effects():
        msgs = c.eval("smokegame/effect_roundtrip()", session="fz")
        expect(value_of(msgs) == "outer", f"fuzz effects wrong: {msgs}")

    def op_threads():
        msgs = c.eval("smokegame/thread_churn(4)", session="fz", timeout=60)
        expect(value_of(msgs) == "4", f"fuzz threads wrong: {msgs}")

    def op_misc():
        c.request("describe", id="d")
        c.request("ls-sessions", id="l")

    weighted = [
        (op_good, 20),
        (op_bad_compile, 6),
        (op_throw, 10),
        (op_abort, 10),
        (op_resume, 4),
        (op_persist_good, 8),
        (op_persist_bad, 4),
        (op_disconnect, 5),
        (op_output, 8),
        (op_effects, 6),
        (op_threads, 3),
        (op_misc, 2),
    ]
    ops = [f for f, w in weighted for _ in range(w)]

    for i in range(ctx.fuzz_ops):
        op = rng.choice(ops)
        try:
            op()
        except SmokeFail as e:
            raise SmokeFail(f"op #{i} ({op.__name__}): {e}")
        ctx.check_alive()
    c.drain_suspensions(session="fz")
    msgs = c.eval("1 + 1", session="fz")
    expect(value_of(msgs) == "2", f"post-fuzz eval failed: {msgs}")
    c.close()


SCENARIOS = [
    ("sanity", scenario_sanity),
    ("disconnect-storm", scenario_disconnect_storm),
    ("throw-abort-resume", scenario_throw_abort_resume),
    ("output-capture", scenario_output_capture),
    ("persist-churn", scenario_persist_churn),
    ("effect-churn", scenario_effect_churn),
    ("thread-churn", scenario_thread_churn),
    ("gc-pressure", scenario_gc_pressure),
    ("main-crash", scenario_main_crash),
    ("fuzz", scenario_fuzz),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_scenario(name, fn, beag, fuzz_ops, seed, keep_logs):
    workspace = tempfile.mkdtemp(prefix=f"beagle-smoke-{name}-")
    ctx = Ctx(beag, workspace, fuzz_ops, seed)
    start = time.time()
    failure = None
    try:
        fn(ctx)
    except SmokeFail as e:
        failure = str(e)
    except Exception as e:  # harness bug or unexpected I/O error
        failure = f"{type(e).__name__}: {e}"
    finally:
        if ctx.server:
            died_early = not ctx.server.alive()
            exit_code = ctx.server.proc.poll()
            ctx.server.stop()
            panics = ctx.server.panic_lines()
            if panics and failure is None:
                failure = f"panic in server log: {panics[0]}"
            elif died_early and failure is None:
                failure = f"server died mid-scenario (exit {exit_code})"
            elif panics and failure is not None:
                failure += f" | panic in server log: {panics[0]}"
    elapsed = time.time() - start

    if failure is None:
        print(f"  PASS  {name}  ({elapsed:.1f}s)")
        if not keep_logs:
            shutil.rmtree(workspace, ignore_errors=True)
        return True
    print(f"  FAIL  {name}  ({elapsed:.1f}s)")
    print(f"        {failure}")
    print(f"        workspace + log kept at: {workspace}")
    log = os.path.join(workspace, "server.log")
    if os.path.exists(log):
        with open(log, "rb") as f:
            tail = f.read().decode("utf-8", errors="replace").splitlines()[-15:]
        for line in tail:
            print(f"        | {line}")
    return False


def main():
    ap = argparse.ArgumentParser(
        description="Beagle live-coding smoke test (panic hunter)."
    )
    ap.add_argument("--beag", default=os.environ.get("BEAG", DEFAULT_BEAG),
                    help="path to the beag binary (default: target/release/beag)")
    ap.add_argument("--only", action="append",
                    help="run only this scenario (repeatable)")
    ap.add_argument("--fuzz-ops", type=int, default=150,
                    help="number of fuzz operations (default 150)")
    ap.add_argument("--seed", type=int, default=None,
                    help="fuzz seed (default: random, printed for repro)")
    ap.add_argument("--keep-logs", action="store_true",
                    help="keep workspaces/logs even on pass")
    ap.add_argument("--list", action="store_true", help="list scenarios")
    args = ap.parse_args()

    if args.list:
        for name, _ in SCENARIOS:
            print(name)
        return 0

    if not os.path.exists(args.beag):
        print(f"beag binary not found at {args.beag}", file=sys.stderr)
        print("build it first: cargo build --release", file=sys.stderr)
        return 2

    seed = args.seed if args.seed is not None else random.randrange(1 << 32)

    chosen = SCENARIOS
    if args.only:
        unknown = set(args.only) - {n for n, _ in SCENARIOS}
        if unknown:
            print(f"unknown scenario(s): {', '.join(sorted(unknown))}",
                  file=sys.stderr)
            return 2
        chosen = [(n, f) for n, f in SCENARIOS if n in set(args.only)]

    print(f"beag: {args.beag}")
    print(f"fuzz seed: {seed}  (rerun with --seed {seed})")
    print()

    passed = failed = 0
    for name, fn in chosen:
        ok = run_scenario(
            name, fn, args.beag, args.fuzz_ops, seed, args.keep_logs
        )
        if ok:
            passed += 1
        else:
            failed += 1

    print()
    print(f"{passed} passed, {failed} failed ({len(chosen)} total)")
    return 1 if failed else 0


if __name__ == "__main__":
    # Don't let a dying server's broken pipe kill the harness.
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    sys.exit(main())
