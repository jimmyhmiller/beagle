#!/usr/bin/env bash
#
# Phase 0 differential harness for docs/SSA_RUNTIME_PARITY_PLAN.md.
#
# The oracle that gates every later phase. For each benchmark it runs
# the LEGACY allocator and the SSA allocator (BEAGLE_USE_SSA=1) in
# release and records:
#
#   - runtime_ns : the benchmark's own `core/time-now()` measurement of
#                  the hot loop (the LAST `(N ns)` it prints, i.e. the
#                  post-`specialize-all` steady state). This isolates
#                  *runtime* from compile-time, which is what the plan
#                  scopes. Min over --reps repetitions (least noisy).
#   - bails      : # functions the SSA path kicked back to legacy
#                  (`[ssa-compile] BAIL`). The pre-Phase-3 spill proxy:
#                  every bail is a function that didn't fit the 9-GP pool.
#   - edge_moves : Σ phi/edge-resolution moves across all functions
#                  (the move cost Phase 5 coalescing must drive down).
#   - root_slots : Σ frame slots reserved (the GC-scanned region).
#   - maxlive_gp : peak GP register pressure across all functions.
#
# Usage:
#   scripts/ssa_diff.sh                  # measure + compare to baseline
#   scripts/ssa_diff.sh --update-baseline  # (re)write the baseline
#   scripts/ssa_diff.sh --reps 3         # more repetitions per mode
#   scripts/ssa_diff.sh --heavy          # also run bench_btrees_full (~11s/run)
#   scripts/ssa_diff.sh --tol 8          # regression tolerance, percent (default 5)
#
# Exit status: 0 = no regression; 1 = a gated metric (runtime, bails, or
# edge_moves) regressed vs the saved baseline beyond tolerance, or a
# benchmark failed to run. This is what makes it a gate: a phase that
# regresses the oracle fails CI.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BASELINE="scripts/ssa_diff_baseline.tsv"
REPS=2
TOL=5
UPDATE=0
HEAVY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --update-baseline) UPDATE=1; shift ;;
        --reps) REPS="$2"; shift 2 ;;
        --tol) TOL="$2"; shift 2 ;;
        --heavy) HEAVY=1; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

BENCHES=(bench_fib_specialize bench_nbody_full bench_nbody_specialize bench_btrees_specialize)
if [[ $HEAVY -eq 1 ]]; then
    BENCHES+=(bench_btrees_full)
fi

echo ">> building release binary ..." >&2
if ! cargo build --release 2>/tmp/ssa_diff_build.log; then
    echo "!! release build failed:" >&2
    tail -20 /tmp/ssa_diff_build.log >&2
    exit 2
fi
BIN="target/release/beag"

# Extract the LAST "(N ns)" value the program printed (post-specialize
# steady state). Prints nothing if the program produced no timing line.
last_ns() {
    grep -oE '\(([0-9]+) ns\)' | grep -oE '[0-9]+' | tail -1
}

# Min runtime over REPS runs of one mode. $1=label(legacy|ssa) $2=file.
min_runtime() {
    local mode="$1" file="$2" best="" v
    for ((i = 0; i < REPS; i++)); do
        if [[ "$mode" == "ssa" ]]; then
            v=$(BEAGLE_USE_SSA=1 "$BIN" "resources/$file.bg" 2>/dev/null | last_ns)
        else
            v=$("$BIN" "resources/$file.bg" 2>/dev/null | last_ns)
        fi
        [[ -z "$v" ]] && continue
        if [[ -z "$best" || "$v" -lt "$best" ]]; then best="$v"; fi
    done
    echo "${best:-0}"
}

# SSA pipeline counts for one benchmark (single instrumented run; its
# runtime is ignored). Echoes "bails edge_moves root_slots maxlive_gp".
ssa_counts() {
    local file="$1" log
    log=$(BEAGLE_USE_SSA=1 BEAGLE_SSA_REGALLOC_STATS=1 BEAGLE_SSA_LOG_BAIL=1 \
        "$BIN" "resources/$file.bg" 2>&1 1>/dev/null)
    local bails edge_moves root_slots maxlive
    bails=$(grep -c '\[ssa-compile\] BAIL' <<<"$log")
    edge_moves=$(grep -oE 'edge_moves=[0-9]+' <<<"$log" | grep -oE '[0-9]+' \
        | awk '{s+=$1} END{print s+0}')
    root_slots=$(grep -oE 'root_slots=[0-9]+' <<<"$log" | grep -oE '[0-9]+' \
        | awk '{s+=$1} END{print s+0}')
    maxlive=$(grep -oE 'maxlive_gp=[0-9]+' <<<"$log" | grep -oE '[0-9]+' \
        | awk 'BEGIN{m=0} {if($1>m)m=$1} END{print m+0}')
    echo "$bails $edge_moves $root_slots $maxlive"
}

# Parallel indexed arrays (macOS ships bash 3.2 -- no associative
# arrays). Index i corresponds to BENCHES[i].
CUR_LEG=(); CUR_SSA=(); CUR_BAIL=(); CUR_EM=(); CUR_RS=(); CUR_ML=()

echo ">> measuring ${#BENCHES[@]} benchmarks (reps=$REPS) ..." >&2
for i in "${!BENCHES[@]}"; do
    b="${BENCHES[$i]}"
    echo "   - $b" >&2
    leg=$(min_runtime legacy "$b")
    ssa=$(min_runtime ssa "$b")
    read -r bails em rs ml < <(ssa_counts "$b")
    if [[ "$leg" == "0" || "$ssa" == "0" ]]; then
        echo "!! benchmark $b produced no timing (legacy=$leg ssa=$ssa) -- did it crash?" >&2
    fi
    CUR_LEG[$i]=$leg; CUR_SSA[$i]=$ssa
    CUR_BAIL[$i]=$bails; CUR_EM[$i]=$em; CUR_RS[$i]=$rs; CUR_ML[$i]=$ml
done

# ---- table ----------------------------------------------------------
printf '\n%-26s %14s %14s %7s %6s %6s %5s %5s\n' \
    BENCH LEGACY_ns SSA_ns SSA/LEG BAILS MOVES RSLOT MLgp
printf '%s\n' "-------------------------------------------------------------------------------------------"
for i in "${!BENCHES[@]}"; do
    b="${BENCHES[$i]}"; leg=${CUR_LEG[$i]}; ssa=${CUR_SSA[$i]}
    if [[ "$leg" != "0" ]]; then
        ratio=$(awk -v a="$ssa" -v l="$leg" 'BEGIN{printf "%.3f", a/l}')
    else
        ratio="n/a"
    fi
    printf '%-26s %14s %14s %7s %6s %6s %5s %5s\n' \
        "$b" "$leg" "$ssa" "$ratio" \
        "${CUR_BAIL[$i]}" "${CUR_EM[$i]}" "${CUR_RS[$i]}" "${CUR_ML[$i]}"
done
echo ""

# ---- update baseline ------------------------------------------------
if [[ $UPDATE -eq 1 ]]; then
    {
        printf '# bench\tlegacy_ns\tssa_ns\tbails\tedge_moves\troot_slots\tmaxlive_gp\n'
        for i in "${!BENCHES[@]}"; do
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${BENCHES[$i]}" \
                "${CUR_LEG[$i]}" "${CUR_SSA[$i]}" "${CUR_BAIL[$i]}" \
                "${CUR_EM[$i]}" "${CUR_RS[$i]}" "${CUR_ML[$i]}"
        done
    } > "$BASELINE"
    echo ">> wrote baseline -> $BASELINE" >&2
    exit 0
fi

# ---- compare to baseline (the gate) ---------------------------------
if [[ ! -f "$BASELINE" ]]; then
    echo "!! no baseline at $BASELINE; run with --update-baseline first." >&2
    exit 2
fi

# Find this run's index for a benchmark name (-1 if not run this time).
idx_of() {
    local name="$1" j
    for j in "${!BENCHES[@]}"; do
        [[ "${BENCHES[$j]}" == "$name" ]] && { echo "$j"; return; }
    done
    echo "-1"
}

regressed=0
hardfail=0
while IFS=$'\t' read -r b base_leg base_ssa base_bails base_em base_rs base_ml; do
    [[ "$b" == \#* || -z "$b" ]] && continue
    i=$(idx_of "$b")
    [[ "$i" == "-1" ]] && continue  # bench not in this run

    cur_ssa=${CUR_SSA[$i]}; cur_bails=${CUR_BAIL[$i]}; cur_em=${CUR_EM[$i]}

    if [[ "$cur_ssa" == "0" ]]; then
        echo "REGRESSION $b: failed to produce a runtime this run" >&2
        hardfail=1; continue
    fi
    # runtime: fail if slower than baseline SSA by more than TOL%.
    if awk -v c="$cur_ssa" -v base="$base_ssa" -v t="$TOL" \
        'BEGIN{exit !(base>0 && c > base*(1+t/100))}'; then
        pct=$(awk -v c="$cur_ssa" -v base="$base_ssa" 'BEGIN{printf "%+.1f", (c-base)*100/base}')
        echo "REGRESSION $b: SSA runtime ${cur_ssa}ns vs baseline ${base_ssa}ns (${pct}%, tol ${TOL}%)" >&2
        regressed=1
    fi
    # bails (spill proxy): any increase is a regression.
    if [[ "$cur_bails" -gt "$base_bails" ]]; then
        echo "REGRESSION $b: SSA bails $cur_bails vs baseline $base_bails" >&2
        regressed=1
    fi
    # edge moves: any increase is a regression.
    if [[ "$cur_em" -gt "$base_em" ]]; then
        echo "REGRESSION $b: edge_moves $cur_em vs baseline $base_em" >&2
        regressed=1
    fi
done < "$BASELINE"

if [[ $hardfail -ne 0 || $regressed -ne 0 ]]; then
    echo "" >&2
    echo "!! differential gate FAILED" >&2
    exit 1
fi
echo ">> differential gate passed (no SSA regression vs baseline)" >&2
exit 0
