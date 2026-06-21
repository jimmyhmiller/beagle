#!/bin/bash
# ============================================================================
# Starvation probe for the long-session soak — the reliable repro vehicle for
# GC/scheduler fragility under EXTREME CPU contention (e.g. the make_closure
# is_heap_pointer abort, fixed in commit bc96a10; keep this as the standing
# regression). Saturates every core, then runs smoke/soak_long.bg under
# --gc-always so every allocation collects while the cores are oversubscribed.
#
#   smoke/soak_starvation.sh [WINDOW_SECONDS] [REPEATS]
#   smoke/soak_starvation.sh 90 5            # 5 runs, 90s each
#
# Exit 0 = no crash in any run. Nonzero = a run crashed (the probe's whole point)
# — the offending output is printed.
#
# SAFE CLEANUP (do not regress this): the CPU saturators are reaped THREE ways so
# a panicked / timed-out / SIGKILL'd run never orphans them and saturates the
# host (the bug that slowed the fleet):
#   1. each saturator is `timeout`-wrapped, so it SELF-terminates after the
#      window even if this script is hard-killed (-9, which can't be trapped) —
#      the RAII backstop;
#   2. a `trap cleanup EXIT INT TERM` reaps them on any normal/signalled exit;
#   3. an explicit cleanup after each run.
# ============================================================================
set -u
WINDOW="${1:-90}"
REPEATS="${2:-5}"
NCORES="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
BEAG="${BEAG:-./target/release/beag}"
SOAK="$(dirname "$0")/soak_long.bg"

SATS=()
cleanup() {
    local p
    # Kill the `yes` CHILD first (pkill -P), then the `timeout` parent. Killing
    # only the `timeout` wrapper with -9 ORPHANS its `yes` child — SIGKILL can't
    # be forwarded — so the child must be reaped directly.
    for p in "${SATS[@]:-}"; do
        [ -n "$p" ] || continue
        pkill -9 -P "$p" 2>/dev/null
        kill -9 "$p" 2>/dev/null
    done
    SATS=()
}
trap cleanup EXIT INT TERM

crashes=0
for run in $(seq 1 "$REPEATS"); do
    SATS=()
    # (1) self-limiting saturators: `timeout` guarantees they die after the
    # window + slack even if THIS script is SIGKILL'd before the trap can fire.
    for _ in $(seq 1 "$NCORES"); do
        timeout "$((WINDOW + 30))" yes >/dev/null 2>&1 &
        SATS+=("$!")
    done
    out="$(mktemp)"
    timeout "$WINDOW" "$BEAG" run --gc-always "$SOAK" >"$out" 2>&1
    cleanup   # (3) reap immediately after the run
    if grep -qiE "is_heap_pointer|from_tagged: not|non-unwinding panic|panicked at|aborting|SIGSEGV|SIGBUS" "$out"; then
        crashes=$((crashes + 1))
        echo "run $run: CRASH"
        grep -iE "is_heap_pointer|from_tagged: not|panicked at" "$out" | head -3
    else
        echo "run $run: clean ($(grep -c 'SOAK OK' "$out") completions in the ${WINDOW}s window)"
    fi
    rm -f "$out"
done

echo "=== starvation probe: $crashes/$REPEATS crashed ==="
[ "$crashes" -eq 0 ]
