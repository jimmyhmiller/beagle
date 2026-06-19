#!/usr/bin/env python3
"""
Local host for the Beagle async explainer.

Beagle compiles to native machine code (macOS only) — there's no in-browser
runtime — so this tiny server *is* the "hosted REPL": it compiles and runs each
scenario with the real `target/release/beag` binary and streams the output back
to the page.

    python3 docs/async-explainer/server.py
    # then open http://127.0.0.1:8787

Binds to localhost only. It executes arbitrary Beagle code you type, so don't
expose it to the network.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BEAG = REPO_ROOT / "target" / "release" / "beag"
INDEX = HERE / "index.html"
PORT = int(os.environ.get("PORT", "8787"))
RUN_TIMEOUT = float(os.environ.get("RUN_TIMEOUT", "30"))


def run_beagle(code: str):
    """Write `code` to a temp .bg file, run it, return (stdout+stderr, exit, ms)."""
    if not BEAG.exists():
        return (
            f"beag binary not found at {BEAG}\n"
            "Build it first:  cargo build --release",
            127,
            0,
        )
    with tempfile.TemporaryDirectory() as d:
        bg = Path(d) / "playground.bg"
        bg.write_text(code)
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [str(BEAG), str(bg)],
                capture_output=True,
                text=True,
                timeout=RUN_TIMEOUT,
                cwd=str(REPO_ROOT),
            )
            elapsed = int((time.monotonic() - start) * 1000)
            out = proc.stdout
            if proc.stderr.strip():
                out += ("\n" if out else "") + "[stderr]\n" + proc.stderr
            return out.rstrip("\n"), proc.returncode, elapsed
        except subprocess.TimeoutExpired as e:
            elapsed = int((time.monotonic() - start) * 1000)
            partial = (e.stdout or "") if isinstance(e.stdout, str) else ""
            return (
                (partial.rstrip("\n") + "\n" if partial else "")
                + f"[timed out after {RUN_TIMEOUT:.0f}s — a server scenario may not have terminated]",
                124,
                elapsed,
            )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            if INDEX.exists():
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(500, "index.html missing", "text/plain")
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True, "beag": str(BEAG), "exists": BEAG.exists()}))
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/run":
            self._send(404, "not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "bad json"}))
            return
        code = payload.get("code", "")
        if not code.strip():
            self._send(400, json.dumps({"error": "no code"}))
            return
        out, rc, ms = run_beagle(code)
        self._send(200, json.dumps({"output": out, "exit": rc, "ms": ms}))


def main():
    if not BEAG.exists():
        print(f"WARNING: {BEAG} not found — run `cargo build --release` first.", file=sys.stderr)
    print(f"Beagle async explainer → http://127.0.0.1:{PORT}")
    print(f"  beag: {BEAG}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
