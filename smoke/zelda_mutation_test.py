#!/usr/bin/env python3
"""
Mutation testing for the Beagle runtime, driven through beagle-zelda.

Instead of synthetic hazards, this mutates the REAL beagle-zelda source — the
bad edits a developer actually makes while live-coding — and live-redefines each
mutant into the running game. The invariant under test is the RUNTIME's, not the
game's: **no mutant may ever hard-crash the process.** Every broken edit must
either run, or raise a clean error that suspends the main thread recoverably.

For each hot function we fetch its source via reflect, apply one mutation at a
time (operator swaps, literal->null, field->nonexistent, identifier->null, ...),
persist it, and classify the result:

  CRASH  — process died (segfault/panic). A RUNTIME BUG.
  HANG   — main thread stopped responding. A RUNTIME BUG.
  ok-suspended — clean error, main thread suspended (then we restore+resume).
  ok-ran       — mutant ran without error (valid-but-different behavior).
  skip   — mutant didn't compile (syntax broken by the mutation) — not applied.

    python3 smoke/zelda_mutation_test.py
"""
import json, os, re, socket, subprocess, time, itertools
from pathlib import Path

ZD = Path(os.environ["HOME"]) / "Documents/Code/PlayGround/beagle-zelda"
PORT = 7912
PRE = "use beagle.reflect as reflect\n"
PERSIST_SESSION = "persist"
PANIC = ("panicked at", "SIGSEGV", "SIGABRT", "Segmentation fault", "abort()",
         "non-unwinding", "EXC_BAD_ACCESS", "fatal runtime")
MAX_MUTANTS_PER_FN = 24

# Hot functions the game loop calls every frame, so mutations trigger promptly.
TARGETS = [
    "gameplay/read_speed_mult", "gameplay/advance_player", "gameplay/update_camera",
    "gameplay/update_aim", "gameplay/approach_velocity",
    "main/playing_tick",
    "vec/scale", "vec/add", "vec/sub", "vec/length_sq",
    "rendering/drawable_ground_y", "rendering/ysort_drawables",
    "enemies/advance_all_monsters",
]

WRAPPER = f"""namespace __mz_runner
use beagle.repl-main as repl-main
use main as target
fn main() {{ eval("namespace main") repl-main/run-with-repl("127.0.0.1", {PORT}, fn() {{ target/main() }}) }}
"""


class Game:
    def __init__(self):
        self.proc = None; self.log = None
    def start(self):
        import tempfile
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        Path("/tmp/__mz_runner.bg").write_text(WRAPPER)
        self.proc = subprocess.Popen(["beag", "run", "-I", str(ZD / "src"), "/tmp/__mz_runner.bg"],
                                     stdout=self.log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, BEAGLE_ZELDA_NOFOCUS="1"))
        for _ in range(50):
            if self.proc.poll() is not None: return False
            try:
                socket.create_connection(("127.0.0.1", PORT), timeout=1).close(); time.sleep(0.6); return True
            except OSError: time.sleep(0.5)
        return False
    def alive(self): return self.proc and self.proc.poll() is None
    def stop(self):
        if self.alive():
            self.proc.terminate()
            try: self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired: self.proc.kill()
    def panic_lines(self):
        self.log.flush()
        return [l for l in Path(self.log.name).read_text(errors="ignore").splitlines()
                if any(m in l for m in PANIC)]


def req(op, timeout=10, **f):
    try: s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
    except OSError as e: return None
    try:
        m = {"op": op, "id": "m", "session": f.pop("session", "mut")}; m.update(f)
        s.sendall((json.dumps(m) + "\n").encode()); s.settimeout(timeout); b = b""
        while b"\n" not in b:
            c = s.recv(8192)
            if not c: return None
            b += c
        return json.loads(b.split(b"\n", 1)[0])
    except socket.timeout: return "TIMEOUT"
    finally: s.close()


def source(fn):
    r = req("eval", code=PRE + f'reflect/source("{fn}")', session=PERSIST_SESSION)
    if isinstance(r, dict) and r.get("value"):
        v = r["value"]
        return v[1:-1] if v.startswith('"') else v   # strip json-ish quotes if any
    return None


def persist(ns, code):
    r = req("eval", code=PRE + f"reflect/persist({json.dumps(ns)}, {json.dumps(code)})", session=PERSIST_SESSION)
    if not isinstance(r, dict): return False
    if "ex" in r or (r.get("status") and any("resumable" in str(s) or s == "suspended" for s in r["status"])):
        req("abort", session=PERSIST_SESSION); return False
    return True


def main_state():
    r = req("main-status")
    if not isinstance(r, dict): return None if r != "TIMEOUT" else "hang"
    return r.get("main-thread")


# ---- mutation operators -------------------------------------------------
def gen_mutants(src):
    """Yield (description, mutated_src). One mutation per mutant."""
    out = []
    def swaps(pairs):
        for a, b in pairs:
            for m in re.finditer(re.escape(a), src):
                i = m.start()
                out.append((f"{a.strip()!r}->{b.strip()!r}@{i}", src[:i] + b + src[i + len(a):]))
    swaps([(" + ", " - "), (" - ", " + "), (" * ", " / "), (" / ", " * "),
           (" < ", " > "), (" > ", " < "), (" <= ", " >= "), (" >= ", " <= "),
           (" == ", " != "), (" && ", " || "), (" || ", " && ")])
    # field access -> nonexistent field
    for m in re.finditer(r"\.([a-z_][a-zA-Z0-9_]*)", src):
        s, e = m.span()
        out.append((f"field .{m.group(1)}->.zzz_nx@{s}", src[:s] + ".zzz_nx" + src[e:]))
    # numeric literal -> null / 0
    for m in re.finditer(r"(?<![\w.])\d+\.?\d*", src):
        s, e = m.span()
        out.append((f"lit {m.group()}->null@{s}", src[:s] + "null" + src[e:]))
    # a bound identifier usage -> null (simple: replace `name.` reads' base)
    for m in re.finditer(r"\b([a-z_][a-zA-Z0-9_]*)\.", src):
        name = m.group(1)
        if name in ("r", "v", "g", "e", "core"):  # skip namespace aliases
            continue
        s = m.start(1)
        out.append((f"ident {name}->null@{s}", src[:s] + "null" + src[m.end(1):]))
    return out


def recover(game, ns, original):
    """Restore the original fn and resume the (possibly suspended) main thread."""
    persist(ns, original)
    for _ in range(4):
        st = main_state()
        if st == "running": return True
        if st in (None, "hang"): return game.alive()
        # suspended — resume past the failing expression with a benign value
        req("main-resume", code="1.0")
        time.sleep(0.3)
    return main_state() == "running"


def test_function(game, fn, results):
    ns = fn.split("/")[0]
    original = source(fn)
    if not original:
        results.append(("no-source", fn, "")); return game
    mutants = gen_mutants(original)[:MAX_MUTANTS_PER_FN]
    print(f"  {fn}: {len(mutants)} mutants", flush=True)
    for desc, mut in mutants:
        if not game.alive():
            game = relaunch(game, results, fn)
            if not game: return None
        applied = persist(ns, mut)
        if not applied:
            results.append(("skip", f"{fn} {desc}", "didn't compile")); continue
        time.sleep(0.7)  # let a few frames run the mutant
        if not game.alive():
            results.append(("CRASH", f"{fn} {desc}", f"mut={mut[:80]!r} panics={game.panic_lines()[:1]}"))
            print(f"    !! CRASH [{fn} {desc}]", flush=True)
            game = relaunch(game, results, fn)
            if not game: return None
            continue
        st = main_state()
        if st == "hang":
            results.append(("HANG", f"{fn} {desc}", f"mut={mut[:80]!r}"))
            print(f"    !! HANG [{fn} {desc}]", flush=True)
            game.stop(); game = relaunch(game, results, fn)
            if not game: return None
            continue
        # ok (suspended or ran) — restore and continue
        if not recover(game, ns, original):
            game.stop(); game = relaunch(game, results, fn)
            if not game: return None
    persist(ns, original)
    return game


def relaunch(game, results, ctx):
    if game: game.stop()
    g = Game()
    if not g.start():
        results.append(("FATAL", "relaunch", f"won't restart (at {ctx})")); return None
    return g


def main():
    results = []
    game = Game()
    print("launching beagle-zelda...", flush=True)
    if not game.start():
        print("FAILED to start:", Path(game.log.name).read_text()[-400:]); return
    print("up. running mutation testing...\n", flush=True)
    for fn in TARGETS:
        if game is None: break
        game = test_function(game, fn, results)
    if game: game.stop()

    print("\n========== MUTATION RESULTS ==========")
    crash = [r for r in results if r[0] in ("CRASH", "HANG", "FATAL")]
    counts = {}
    for k, *_ in results: counts[k] = counts.get(k, 0) + 1
    if not crash:
        print("No crashes/hangs. Every mutant ran or suspended recoverably. ✅")
    for k, label, detail in crash:
        print(f"[{k}] {label}\n      {detail}")
    print(f"\nsummary: {counts}")


if __name__ == "__main__":
    main()
