#!/usr/bin/env python3
"""
Live-coding bug hunter for beagle-zelda.

Stands up beagle-zelda with an embedded REPL server (the run-with-repl pattern
the agent uses) and drives the LIVE game the way real live-coding does:

  1. hazard battery  — feed null / wrong types into every builtin & operator
     (the "stale struct field after a redefinition" class) and call/access
     scalars. A clean Beagle error is FINE; a process death/hang is a BUG.

  2. main-thread recovery battery — redefine a HOT game-loop function so it
     throws (in many different ways), confirm the main thread SUSPENDS (not
     crashes), then recover via fix+resume / resume-with-value / abort, and
     confirm the game is RUNNING again. Repeated churn stresses the resumable
     continuation machinery. The whole point: resume from exceptions, never crash.

    python3 smoke/zelda_livecode_hunt.py
"""
import json, os, socket, subprocess, sys, time, tempfile
from pathlib import Path

# Headless harness (reliable for repeated runs — no graphics). Same live-coding
# machinery as a real game: run-with-repl + a loop calling a hot function.
SMOKE_DIR = Path(__file__).resolve().parent
HARNESS_NS = "livecode_loop"
HOT_FN = "hot_fn"
GOOD_FN = "fn hot_fn(state) { state.acc + 1 }"
PORT = 7891
PANIC = ("panicked at", "SIGSEGV", "SIGABRT", "Segmentation fault", "abort()",
         "non-unwinding", "EXC_BAD_ACCESS", "fatal runtime")

WRAPPER = f"""namespace __lc_hunt_runner
use beagle.repl-main as repl-main
use {HARNESS_NS} as target
fn main() {{
    eval("namespace {HARNESS_NS}")
    repl-main/run-with-repl("127.0.0.1", {PORT}, fn() {{ target/main() }})
}}
"""

# reflect/persist WRITES redefinitions back to the source file, so the hunter
# loads from a fresh disposable copy each launch — persists can't corrupt the
# committed harness, and every relaunch starts clean.
HARNESS_SRC = """namespace livecode_loop
use beagle.core as core

struct State { tick, acc }

fn hot_fn(state) { state.acc + 1 }

fn loop_body(state) {
    let v = hot_fn(state)
    core/sleep(16)
    loop_body(State { tick: state.tick + 1, acc: v })
}

fn main() {
    loop_body(State { tick: 0, acc: 0 })
}
"""
HARNESS_DIR = Path(tempfile.gettempdir()) / "lc_hunt"


class Game:
    def __init__(self):
        self.proc = None
        self.log = None

    def start(self):
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        HARNESS_DIR.mkdir(exist_ok=True)
        (HARNESS_DIR / f"{HARNESS_NS}.bg").write_text(HARNESS_SRC)  # fresh copy each launch
        wrapper = HARNESS_DIR / "__lc_hunt_runner.bg"
        wrapper.write_text(WRAPPER)
        self.proc = subprocess.Popen(["beag", "run", "-I", str(HARNESS_DIR), str(wrapper)],
                                     stdout=self.log, stderr=subprocess.STDOUT, env=dict(os.environ))
        for _ in range(40):
            if self.proc.poll() is not None:
                return False
            try:
                socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
                time.sleep(0.4)
                return True
            except OSError:
                time.sleep(0.5)
        return False

    def alive(self):
        return self.proc and self.proc.poll() is None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired: self.proc.kill()

    def panic_lines(self):
        self.log.flush()
        return [l for l in Path(self.log.name).read_text(errors="ignore").splitlines()
                if any(m in l for m in PANIC)]


def req(op, timeout=12, **fields):
    """One request on a fresh connection -> (status, parsed-or-detail)."""
    try:
        s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
    except OSError as e:
        return "no-connection", str(e)
    try:
        msg = {"op": op, "id": "h", "session": "hunt"}
        msg.update(fields)
        s.sendall((json.dumps(msg) + "\n").encode())
        s.settimeout(timeout)
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(8192)
            if not chunk:
                return "closed", "server closed connection"
            buf += chunk
        return "ok", json.loads(buf.split(b"\n", 1)[0].decode())
    except socket.timeout:
        return "timeout", f"no response in {timeout}s"
    finally:
        s.close()


def ev(code, timeout=12, session="hunt"):
    return req("eval", timeout=timeout, code=code, session=session)


# Reflect evals need the alias compiled in-line (sessions share one global
# current-namespace, so a bare `reflect/...` fails with "alias not found"). A
# failed reflect eval leaves the session resumable-suspended — abort to clear.
REFLECT_PRELUDE = "use beagle.reflect as reflect\n"
PERSIST_SESSION = "persist"


def _abort_if_suspended(data, session):
    if isinstance(data, dict):
        status = data.get("status") or []
        if any("resumable" in str(s) or s == "suspended" for s in status) or "ex" in data:
            req("abort", session=session)
            return True
    return False


def persist(ns, code):
    """Redefine `code` in namespace `ns`. Returns True on success."""
    st, data = ev(f'{REFLECT_PRELUDE}reflect/persist({json.dumps(ns)}, {json.dumps(code)})',
                  session=PERSIST_SESSION)
    if st != "ok":
        return False
    suspended = _abort_if_suspended(data, PERSIST_SESSION)
    # success = a normal value response with no exception/suspension
    return (not suspended) and isinstance(data, dict) and "ex" not in data


def main_thread(game):
    """Return 'running' / 'suspended' / None(can't tell)."""
    st, data = req("main-status")
    if st != "ok" or not isinstance(data, dict):
        return None
    return data.get("main-thread")


def wait_status(game, target, timeout=12):
    end = time.time() + timeout
    while time.time() < end:
        if not game.alive():
            return "dead"
        if main_thread(game) == target:
            return "ok"
        time.sleep(0.2)
    return "timeout"


# ------------------------------------------------------------------ batteries

def hazard_battery():
    out = []
    for fn in ["sqrt", "floor", "ceil", "abs", "round", "trunc", "sin", "cos",
               "tan", "exp", "log", "to-float", "to-int"]:
        for v in ["null", '"x"', "true"]:
            out.append((f"{fn}({v})", f"{fn}({v})"))
    for v in ["null", '"x"', "true"]:
        out += [(f"clamp({v},0.0,9.0)", f"clamp({v}, 0.0, 9.0)"),
                (f"clamp(1.0,0.0,{v})", f"clamp(1.0, 0.0, {v})"),
                (f"max(1.0,{v})", f"max(1.0, {v})"), (f"min({v},1.0)", f"min({v}, 1.0)")]
    for op in ["+", "-", "*", "/", "%", "<", ">", "<=", ">=", "=="]:
        out += [(f"null {op} 1", f"null {op} 1"), (f"1 {op} null", f"1 {op} null"),
                (f'"a" {op} 1', f'"a" {op} 1'), (f"{{}} {op} 1", f"{{}} {op} 1")]
    out += [("1/0", "1 / 0"), ("1.0/0.0", "1.0 / 0.0"), ("5%0", "5 % 0")]
    for fn in ["length", "first", "rest", "seq", "reverse", "count", "last"]:
        for v in ["null", "42", "3.5", "true", '"s"']:
            out.append((f"{fn}({v})", f"{fn}({v})"))
    for v in ["null", "42", "3.5", "true", '"s"', "[1,2]", "{}"]:
        out += [(f"{v}[0]", f"let __v = {v}; __v[0]"),
                (f"{v}.foo", f"let __v = {v}; __v.foo"),
                (f"{v}.a.b", f"let __v = {v}; __v.a.b")]   # nested field chain
    out += [("get(null,:k)", "get(null, :k)"), ("push(null,1)", "push(null, 1)"),
            ("nth(null,0)", "nth(null, 0)"), ("get(42,:k)", "get(42, :k)"),
            ("nth([],9)", "nth([], 9)"), ("nth([1],9)", "nth([1], 9)")]
    out += [('"x"++null', '"x" ++ null'), ('null++"x"', 'null ++ "x"'),
            ("split(null,c)", 'split(null, ",")'), ("to-string(null)", "to-string(null)")]
    out += [("call-int", "let __f = 42; __f()"), ("call-null", "let __f = null; __f()"),
            ("call-str", 'let __f = "x"; __f()'), ("call-arr", "let __f = [1]; __f()"),
            ("call-map", "let __f = {}; __f()")]
    out += [("match-int-struct", "struct __S { a } match 7 { __S { a } => a, _ => 0 }"),
            ("match-null-struct", "struct __S2 { a } match null { __S2 { a } => a, _ => 0 }")]
    out += [("make-array-neg", "make-array(0 - 1)"),
            ("make-array-huge", "make-array(1000000000)"),
            ("deep-spread-null", "struct __P { a, b } let base = null; __P { ...base, a: 1 }"),
            ("spread-scalar", "struct __P2 { a, b } let base = 42; __P2 { ...base, a: 1 }")]
    # higher-order fns fed a NON-function where a function is expected (function-first)
    for hof in ["map", "filter", "reduce", "flat-map", "take-while", "drop-while",
                "group-by", "sort-by", "find", "each", "any?", "all?"]:
        out.append((f"{hof}-nonfn", f"{hof}(42, [1, 2, 3])"))
        out.append((f"{hof}-nullfn", f"{hof}(null, [1, 2, 3])"))
    out += [("sort-with-nonfn", "sort-with(42, [3, 1, 2])"),
            ("reduce-nonfn", "reduce(42, 0, [1, 2])"),
            ("map-noncoll", "map(fn(x) { x }, 42)"),
            ("map-nullcoll", "map(fn(x) { x }, null)")]
    # atoms / mutation
    out += [("swap-nonfn", "let a = atom(1); swap!(a, 42)"),
            ("swap-nonatom", "swap!(42, fn(x) { x })"),
            ("deref-nonatom", "deref(42)"), ("deref-null", "deref(null)"),
            ("reset-nonatom", "reset!(42, 1)")]
    # string ops with wrong types / bounds
    out += [("substring-null", "substring(null, 0, 1)"),
            ("substring-oob", 'substring("ab", 0, 99)'),
            ("split-nonstr", "split(42, \",\")"),
            ("replace-null", 'replace(null, "a", "b")'),
            ("string-index-oob", 'let s = "ab"; s[99]'),
            ("char-at-null", "char-at(null, 0)")]
    # bitwise / numeric on wrong types
    out += [("bitand-str", '42 & "x"'), ("shift-null", "1 << null"),
            ("bitand-float", "1.5 & 2"), ("xor-null", "null ^ 1")]
    # conversions
    out += [("to-int-str", 'to-int("abc")'), ("to-float-str", 'to-float("xyz")'),
            ("to-int-huge", "to-int(1e300)"), ("to-int-null", "to-int(null)"),
            ("parse-int-bad", 'to-int("12.x.3")')]
    # destructuring on wrong shapes
    out += [("destr-short", "match [1] { [a, b, c] => a, _ => 0 }"),
            ("destr-null", "match null { [a] => a, _ => 0 }"),
            ("let-destr-null", "let [a, b] = null; a"),
            ("let-destr-struct-null", "struct __D { x, y } let __D { x, y } = null; x")]
    # effects with no handler
    out += [("perform-no-handler", "perform UndefinedEffect.Op {}")]
    # deep recursion -> stack growth (must not segfault the process)
    out += [("deep-recursion", "fn __r(n) { if n <= 0 { 0 } else { __r(n - 1) + 1 } } __r(500000)")]
    # array write out of bounds
    out += [("arr-write-oob", "let a = make-array(2); arr/write-field(a, 999, 7)")]
    # array/vector ops on non-arrays
    out += [("arr-read-nonarr", "arr/read-field(42, 0)"),
            ("arr-read-null", "arr/read-field(null, 0)"),
            ("arr-write-nonarr", "arr/write-field(42, 0, 1)"),
            ("arr-len-nonarr", "arr/length(42)"),
            ("vec-get-nonvec", "vec-get(42, 0)"),
            ("vec-count-null", "vec-count(null)")]
    # struct field SET on wrong types / chained
    out += [("set-field-on-int", "let v = 42; v.foo = 1"),
            ("set-field-on-null", "let v = null; v.foo = 1"),
            ("struct-as-callable", "struct __C { a } let c = __C { a: 1 }; c()"),
            ("struct-index", "struct __I { a } let c = __I { a: 1 }; c[0]"),
            ("chained-null-call", "let v = null; v.foo()"),
            ("chained-int-method", "let v = 42; v.foo.bar()")]
    # keyword / map edge cases
    out += [("get-nonmap-key", "get(42, :k)"), ("get-null-default", "get(null, :k, 0)"),
            ("map-int-key", "{42 1}")]
    return out


def concurrency_battery():
    # (label, code) — spawn/await/future hazards. spawn is cooperative now.
    return [
        ("spawn-nonfn", "async/spawn(42)"),
        ("spawn-null", "async/spawn(null)"),
        ("await-nonfuture", "async/await(42)"),
        ("await-null", "async/await(null)"),
        ("await-all-nonlist", "async/await-all(42)"),
        ("spawn-throwing", "let f = async/spawn(fn() { throw(\"in future\") }); async/await(f)"),
        ("spawn-bad-body", "let f = async/spawn(fn() { let x = null x + 1 }); async/await(f)"),
        ("spawn-then-await-twice", "let f = async/spawn(fn() { 5 }); async/await(f) + async/await(f)"),
        ("nested-spawn-await", "async/await(async/spawn(fn() { async/await(async/spawn(fn() { 1 })) }))"),
        ("thread-throwing", "thread(fn() { throw(\"thread boom\") }) 1 + 1"),
        ("sleep-negative", "async/sleep(0 - 5) 1"),
    ]


def redefine_battery():
    # (label, namespace, code) — applied via persist() against the live loop.
    ns = HARNESS_NS
    return [
        ("add-field", ns, "struct State { tick, acc, extra }"),     # stale instances read null for extra
        ("remove-field", ns, "struct State { tick }"),              # acc now missing on reads
        ("reorder-fields", ns, "struct State { acc, tick }"),
        ("restore-struct", ns, "struct State { tick, acc }"),
        ("redefine-enum", ns, "enum Color { Red, Green, Blue, NewVariant { x } }"),
        ("redefine-cold-fn", ns, "fn unused_helper() { 42 }"),
        ("bad-syntax", ns, "fn broken( { "),
        ("undefined-ref", ns, "fn uses_undef() { totally_undefined_thing() }"),
        ("redefine-hot-fn-ok", ns, "fn hot_fn(state) { state.acc + 2 }"),
        ("restore-hot-fn", ns, GOOD_FN),
    ]


# hot_fn breakages: each makes the hot loop throw a different runtime error.
BREAKAGES = [
    ("throw-string",   'fn hot_fn(state) { throw("boom from live edit") }'),
    ("arith-on-null",  'fn hot_fn(state) { state.nope_field + 1.0 }'),
    ("clamp-on-null",  'fn hot_fn(state) { clamp(state.nope_field, 0.0, 1.0) }'),
    ("call-non-fn",    'fn hot_fn(state) { let f = 42 f() }'),
    ("prop-on-scalar", 'fn hot_fn(state) { state.nope_field.deeper }'),
    ("div-by-zero",    'fn hot_fn(state) { 1 / 0 }'),
    ("index-oob",      'fn hot_fn(state) { nth([], 9) }'),
    ("bad-arg-builtin",'fn hot_fn(state) { sqrt(null) }'),
    ("match-scalar",   'fn hot_fn(state) { match 7 { State { tick, acc } => acc, _ => 0 } }'),
    ("seq-on-int",     'fn hot_fn(state) { length(42) }'),
]


def recover_cycle(game, label, break_code, recovery, results):
    """break the hot fn -> expect suspend -> fix+recover -> expect running, no crash."""
    if not game.alive():
        results.append(("DIED-BEFORE", label, "process already dead")); return False
    persist(HARNESS_NS, break_code)
    s = wait_status(game, "suspended", timeout=10)
    if s == "dead":
        results.append(("CRASH", label, f"crashed instead of suspending; panics={game.panic_lines()[:2]}")); return False
    if s == "timeout":
        # never suspended — maybe didn't throw this frame; restore and move on
        persist(HARNESS_NS, GOOD_FN)
        results.append(("NO-SUSPEND", label, "hot fn redefinition didn't suspend main thread")); return True
    # main thread is suspended (recovered cleanly, no crash) — now recover it
    persist(HARNESS_NS, GOOD_FN)  # fix future frames
    if recovery == "resume":
        req("main-resume", code="1.0")
    elif recovery == "resume-nocode":
        req("main-resume")
        persist(HARNESS_NS, GOOD_FN)
    elif recovery == "abort":
        req("main-abort")
        time.sleep(0.4)
        if not game.alive():
            results.append(("CRASH", label, "crashed during abort")); return False
        return "aborted"
    s2 = wait_status(game, "running", timeout=10)
    if s2 == "dead":
        results.append(("CRASH", label, f"crashed during resume; panics={game.panic_lines()[:2]}")); return False
    if s2 == "timeout":
        results.append(("STUCK", label, "stayed suspended after resume")); return True
    if not game.alive():
        results.append(("CRASH", label, "died after recover")); return False
    return True


def run_simple(game, battery, results, name):
    print(f"=== battery: {name} ({len(battery)} ops) ===", flush=True)
    i = 0
    while i < len(battery):
        if not game.alive():
            return False, game
        label, code = battery[i]
        st, _ = req("eval", timeout=12, code=code)
        time.sleep(0.12)
        if not game.alive():
            results.append(("CRASH", label, f"code={code!r} status={st} panics={game.panic_lines()[:2]}"))
            print(f"  !! CRASH at [{label}] — relaunching", flush=True)
            game.stop(); game = Game()
            if not game.start():
                results.append(("FATAL", "relaunch", "won't restart")); return True, game
            i += 1; continue
        if st in ("timeout", "closed", "no-connection"):
            results.append(("BAD", label, f"code={code!r} -> {st}"))
        i += 1
    return True, game


def main():
    results = []
    game = Game()
    print("launching beagle-zelda with REPL server...", flush=True)
    if not game.start():
        print("FAILED to start game"); return
    print(f"up on :{PORT}\n", flush=True)

    ok, game = run_simple(game, hazard_battery(), results, "hazards")

    # concurrency hazards need the async alias compiled in-line
    conc = [(label, "use beagle.async as async\n" + code) for label, code in concurrency_battery()]
    ok, game = run_simple(game, conc, results, "concurrency")

    print(f"=== battery: redefine ({len(redefine_battery())} live struct/fn/enum changes) ===", flush=True)
    for label, ns, code in redefine_battery():
        if not game.alive():
            results.append(("DIED-BEFORE", label, "process already dead")); break
        persist(ns, code)
        time.sleep(0.4)  # let the game loop run a few frames on the changed defs
        if not game.alive():
            results.append(("CRASH", label, f"redefine crashed; panics={game.panic_lines()[:2]}"))
            print(f"  !! CRASH at [{label}] — relaunching", flush=True)
            game.stop(); game = Game(); game.start()

    # --- main-thread recovery battery ---
    print("=== battery: main-thread recovery (resume from exceptions) ===", flush=True)
    recoveries = ["resume", "resume-nocode"]
    for ri, recovery in enumerate(recoveries):
        for label, brk in BREAKAGES:
            tag = f"{label}/{recovery}"
            r = recover_cycle(game, tag, brk, recovery, results)
            if r is False:  # crashed -> relaunch
                print(f"  !! CRASH at [{tag}] — relaunching", flush=True)
                game.stop(); game = Game()
                if not game.start():
                    results.append(("FATAL", "relaunch", "won't restart")); break
    # abort path + churn
    print("=== battery: abort + churn ===", flush=True)
    for label, brk in BREAKAGES[:3]:
        r = recover_cycle(game, f"{label}/abort", brk, "abort", results)
        # after an abort the game loop has exited; relaunch for the next test
        game.stop(); game = Game()
        if not game.start():
            results.append(("FATAL", "relaunch-after-abort", "won't restart")); break
    # rapid churn: same break/fix/resume many times to stress continuations
    print("  churn x12...", flush=True)
    for n in range(12):
        if not recover_cycle(game, f"churn-{n}", BREAKAGES[n % len(BREAKAGES)][1], "resume", results):
            game.stop(); game = Game(); game.start()

    game.stop()
    print("\n========== RESULTS ==========")
    crash = [r for r in results if r[0] in ("CRASH", "FATAL")]
    weird = [r for r in results if r[0] in ("BAD", "STUCK", "DIED-BEFORE", "NO-SUSPEND")]
    if not crash and not weird:
        print("No crashes/hangs. All live changes + exception resume/abort recovered cleanly. ✅")
    for k, label, detail in crash: print(f"[{k}] {label}\n      {detail}")
    for k, label, detail in weird: print(f"[{k}] {label}\n      {detail}")
    print(f"\n{len(crash)} hard crashes, {len(weird)} stuck/weird.")


if __name__ == "__main__":
    main()
