#!/usr/bin/env bash

# Benchmark comparison script for Beagle vs Node.js vs Ruby vs Python
#
# Usage:
#   run_benchmarks.sh [options] [--] [benchmark ...]
#
# Options:
#   -l, --languages LANGS   Comma-separated languages in display order (default: all)
#                            Available: beagle, node, ruby, python
#   -n, --iterations N      Iterations per benchmark (default: 3)
#       --fast              Use small inputs for quick smoke-test runs (default: 1 iteration)
#       --no-build          Skip building Beagle (use existing binary)
#   -v, --verbose           Show output from each benchmark run
#       --verify            Verify Beagle output matches Node.js (then exit)
#       --beagle-only       Shorthand for --languages beagle
#   -h, --help              Show this help message
#
# Benchmarks:
#   revcomp  binary_trees  fasta  spectral_norm
#   knucleotide  nbody  mandelbrot  fannkuch_redux
#
# If no benchmarks are specified, all are run in default order.
# When specified, benchmarks run in the order given.
# Languages run and display in the order given to --languages.
#
# Examples:
#   run_benchmarks.sh mandelbrot
#   run_benchmarks.sh -l node,beagle,ruby mandelbrot nbody
#   run_benchmarks.sh -n 5 --no-build fasta nbody mandelbrot

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
SKIP_BUILD=false
VERBOSE=false
VERIFY=false
FAST=false
ITERATIONS=3
ITERATIONS_SET=false
LANGUAGES=""
SELECTED_BENCHMARKS=()

usage() {
    sed -n '3,/^$/s/^# \?//p' "$0"
    exit 0
}

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        --no-build)
            SKIP_BUILD=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --fast)
            FAST=true
            shift
            ;;
        --beagle-only)
            LANGUAGES="beagle"
            shift
            ;;
        -l|--languages)
            if [[ -z "${2:-}" || "$2" == -* ]]; then
                echo "Error: --languages requires an argument" >&2
                exit 1
            fi
            LANGUAGES="$2"
            shift 2
            ;;
        -n|--iterations)
            if [[ -z "${2:-}" || ! "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: --iterations requires a positive integer" >&2
                exit 1
            fi
            ITERATIONS="$2"
            ITERATIONS_SET=true
            shift 2
            ;;
        --)
            shift
            SELECTED_BENCHMARKS+=("$@")
            break
            ;;
        -*)
            echo "Error: unknown option: $1" >&2
            echo "Try --help for usage." >&2
            exit 1
            ;;
        *)
            SELECTED_BENCHMARKS+=("$1")
            shift
            ;;
    esac
done

# Ordered list of languages to run (preserves user-specified order)
LANG_ORDER=()

normalize_lang() {
    local lang
    lang=$(echo "$1" | tr '[:upper:]' '[:lower:]' | xargs)
    case $lang in
        beagle) echo "beagle" ;;
        node|nodejs|node.js) echo "node" ;;
        ruby) echo "ruby" ;;
        python|python3) echo "python" ;;
        *)
            echo "Error: unknown language: $1" >&2
            echo "Available: beagle, node, ruby, python" >&2
            exit 1
            ;;
    esac
}

if [ -n "$LANGUAGES" ]; then
    IFS=',' read -ra LANG_ARRAY <<< "$LANGUAGES"
    for lang in "${LANG_ARRAY[@]}"; do
        LANG_ORDER+=("$(normalize_lang "$lang")")
    done
else
    LANG_ORDER=(beagle node ruby python)
fi

lang_enabled() {
    local target="$1"
    for lang in "${LANG_ORDER[@]}"; do
        [ "$lang" = "$target" ] && return 0
    done
    return 1
}

lang_display() {
    case "$1" in
        beagle) echo "Beagle" ;;
        node)   echo "Node.js" ;;
        ruby)   echo "Ruby" ;;
        python) echo "Python" ;;
    esac
}

lang_pad() {
    case "$1" in
        beagle) echo "Beagle...  " ;;
        node)   echo "Node.js... " ;;
        ruby)   echo "Ruby...    " ;;
        python) echo "Python...  " ;;
    esac
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Official benchmarksgame arguments (single source of truth)
# Format: "name:verify_arg:fast_arg:perf_arg"
# Default order: Ruby benchmark times (fastest to slowest)
# Note: revcomp and knucleotide args are fasta size (they read from stdin)
ALL_BENCHMARKS=(
    "revcomp:1000:10000:25000000"
    "binary_trees:10:12:21"
    "fasta:1000:10000:25000000"
    "spectral_norm:100:500:5500"
    "knucleotide:1000:10000:25000000"
    "nbody:1000:500000:50000000"
    "mandelbrot:200:800:16000"
    "fannkuch_redux:7:9:12"
)

# --fast defaults to 1 iteration unless -n was explicitly set
if [ "$FAST" = true ] && [ "$ITERATIONS_SET" = false ]; then
    ITERATIONS=1
fi

valid_benchmark_names() {
    for config in "${ALL_BENCHMARKS[@]}"; do
        IFS=':' read -r bname _ _ <<< "$config"
        echo "  $bname"
    done
}

# Filter and reorder benchmarks based on command-line selection
if [ ${#SELECTED_BENCHMARKS[@]} -gt 0 ]; then
    BENCHMARKS=()
    for selected in "${SELECTED_BENCHMARKS[@]}"; do
        found=false
        for config in "${ALL_BENCHMARKS[@]}"; do
            IFS=':' read -r bname _ _ <<< "$config"
            if [ "$bname" = "$selected" ]; then
                BENCHMARKS+=("$config")
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            echo -e "${RED}Error: unknown benchmark: $selected${NC}" >&2
            echo "Available benchmarks:" >&2
            valid_benchmark_names >&2
            exit 1
        fi
    done
else
    BENCHMARKS=("${ALL_BENCHMARKS[@]}")
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Beagle Benchmark Comparison Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Iterations per benchmark: $ITERATIONS"

# Show which languages are being compared (in order)
LANG_LIST=""
for lang in "${LANG_ORDER[@]}"; do
    LANG_LIST="${LANG_LIST}$(lang_display "$lang"), "
done
LANG_LIST="${LANG_LIST%, }"
echo "Languages: $LANG_LIST"
echo ""

BEAGLE="$PROJECT_DIR/target/release/main"

# Build Beagle in release mode (unless skipped or not running Beagle)
if lang_enabled beagle; then
    if [ "$SKIP_BUILD" = false ]; then
        echo -e "${YELLOW}Building Beagle (release mode)...${NC}"
        cd "$PROJECT_DIR"
        cargo build --release 2>&1 | tail -5
        echo -e "${GREEN}Build complete.${NC}"
        echo ""
    else
        echo -e "${YELLOW}Skipping build (using existing binary)${NC}"
        if [ ! -f "$BEAGLE" ]; then
            echo -e "${RED}Error: $BEAGLE not found. Run without --no-build first.${NC}"
            exit 1
        fi
        echo ""
    fi
fi

cd "$PROJECT_DIR"

# Check if benchmark needs stdin input (uses fasta output)
needs_stdin_input() {
    local name="$1"
    case "$name" in
        revcomp|knucleotide) return 0 ;;
        *) return 1 ;;
    esac
}

# Generate fasta input file for stdin-based benchmarks
generate_fasta_input() {
    local size="$1"
    local input_file="$2"

    local fasta_js="$SCRIPT_DIR/comparison/node/fasta.js"
    if [ -f "$fasta_js" ]; then
        node "$fasta_js" "$size" > "$input_file" 2>/dev/null
    else
        local fasta_bg="$SCRIPT_DIR/benchmarksgame/fasta.bg"
        "$BEAGLE" "$fasta_bg" "$size" > "$input_file" 2>/dev/null
    fi
}

# Get the command to run a benchmark for a given language
# Returns empty string (and exit 1) if file not found
get_benchmark_cmd() {
    local lang="$1" name="$2" arg="$3" input_file="$4"
    local file cmd_prefix

    case "$lang" in
        beagle)
            file="$SCRIPT_DIR/benchmarksgame/$name.bg"
            cmd_prefix="$BEAGLE $file"
            ;;
        node)
            file="$SCRIPT_DIR/comparison/node/${name}.js"
            cmd_prefix="node $file"
            ;;
        ruby)
            file="$SCRIPT_DIR/comparison/ruby/${name}.rb"
            cmd_prefix="ruby $file"
            ;;
        python)
            file="$SCRIPT_DIR/comparison/python/${name}.py"
            cmd_prefix="python3 $file"
            ;;
    esac

    [ ! -f "$file" ] && return 1

    if [ -n "$input_file" ]; then
        echo "$cmd_prefix < $input_file"
    else
        echo "$cmd_prefix $arg"
    fi
}

# Verify mode: check Beagle output against Node.js
if [ "$VERIFY" = true ]; then
    echo -e "${BLUE}Verifying Beagle output against Node.js...${NC}"
    echo ""

    ALL_PASSED=true

    for benchmark_config in "${BENCHMARKS[@]}"; do
        IFS=':' read -r name verify_arg fast_arg perf_arg <<< "$benchmark_config"

        beagle_file="$SCRIPT_DIR/benchmarksgame/$name.bg"
        node_file="$SCRIPT_DIR/comparison/node/${name}.js"

        echo -n "  $name (arg=$verify_arg)... "

        if [ ! -f "$beagle_file" ]; then
            echo -e "${RED}SKIP (no Beagle file)${NC}"
            continue
        fi
        if [ ! -f "$node_file" ]; then
            echo -e "${RED}SKIP (no Node file)${NC}"
            continue
        fi

        if needs_stdin_input "$name"; then
            input_file=$(mktemp)
            generate_fasta_input "$verify_arg" "$input_file"
            beagle_out=$("$BEAGLE" "$beagle_file" < "$input_file" 2>&1) || true
            node_out=$(node "$node_file" < "$input_file" 2>&1) || true
            rm -f "$input_file"
        else
            beagle_out=$("$BEAGLE" "$beagle_file" "$verify_arg" 2>&1) || true
            node_out=$(node "$node_file" "$verify_arg" 2>&1) || true
        fi

        if [ "$beagle_out" = "$node_out" ]; then
            echo -e "${GREEN}PASS${NC}"
        else
            echo -e "${RED}FAIL${NC}"
            ALL_PASSED=false
            echo "    Expected (Node.js):"
            echo "$node_out" | head -10 | sed 's/^/      /'
            echo "    Got (Beagle):"
            echo "$beagle_out" | head -10 | sed 's/^/      /'
        fi
    done

    echo ""
    if [ "$ALL_PASSED" = true ]; then
        echo -e "${GREEN}All verifications passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some verifications failed.${NC}"
        exit 1
    fi
fi

# Function to run a command and return average time in milliseconds
run_benchmark() {
    local cmd="$1"
    local iterations="$2"
    local total=0

    for ((i=1; i<=iterations; i++)); do
        local start=$(python3 -c 'import time; print(int(time.time() * 1000))')
        if [ "$VERBOSE" = true ]; then
            eval "$cmd"
            local exit_code=$?
        else
            eval "$cmd" > /dev/null 2>&1
            local exit_code=$?
        fi
        local end=$(python3 -c 'import time; print(int(time.time() * 1000))')

        if [ $exit_code -ne 0 ]; then
            echo "-1"
            return
        fi

        local elapsed=$((end - start))
        total=$((total + elapsed))
    done

    echo $((total / iterations))
}

format_time() {
    local ms=$1
    if [ "$ms" -eq -1 ]; then
        echo "FAILED"
    elif [ "$ms" -ge 1000 ]; then
        printf "%.2fs" "$(echo "$ms / 1000" | bc -l)"
    else
        echo "${ms}ms"
    fi
}

# Results stored as "benchmark_name:lang:time_ms" lines
RESULTS_FILE=$(mktemp)

get_result() {
    local name="$1" lang="$2"
    local line
    line=$(grep "^${name}:${lang}:" "$RESULTS_FILE" 2>/dev/null) || true
    if [ -n "$line" ]; then
        echo "$line" | cut -d: -f3
    else
        echo "0"
    fi
}

run_single_benchmark() {
    local name="$1"
    local arg="$2"

    echo -e "${CYAN}Running: $name (arg=$arg)${NC}"

    local input_file=""

    # For stdin-based benchmarks, generate input first
    if needs_stdin_input "$name"; then
        input_file=$(mktemp)
        echo -n "  Generating input... "
        generate_fasta_input "$arg" "$input_file"
        echo "done"
    fi

    for lang in "${LANG_ORDER[@]}"; do
        local cmd
        if cmd=$(get_benchmark_cmd "$lang" "$name" "$arg" "$input_file"); then
            echo -n "  $(lang_pad "$lang")"
            local time_ms
            time_ms=$(run_benchmark "$cmd" "$ITERATIONS")
            echo "$(format_time $time_ms)"
            echo "$name:$lang:$time_ms" >> "$RESULTS_FILE"
        else
            echo "  $(lang_pad "$lang")N/A"
            echo "$name:$lang:0" >> "$RESULTS_FILE"
        fi
    done

    # Clean up input file
    if [ -n "$input_file" ]; then
        rm -f "$input_file"
    fi

    echo ""
}

# Run benchmarks
for benchmark_config in "${BENCHMARKS[@]}"; do
    IFS=':' read -r name verify_arg fast_arg perf_arg <<< "$benchmark_config"
    if [ "$FAST" = true ]; then
        run_single_benchmark "$name" "$fast_arg"
    else
        run_single_benchmark "$name" "$perf_arg"
    fi
done

# Print summary table
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}              RESULTS${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

print_header() {
    printf "%-18s" "Benchmark"
    for lang in "${LANG_ORDER[@]}"; do
        printf " %12s" "$(lang_display "$lang")"
    done
    printf "\n"

    printf "%-18s" "-----------------"
    for lang in "${LANG_ORDER[@]}"; do
        printf " %12s" "------------"
    done
    printf "\n"
}

print_header

for benchmark_config in "${BENCHMARKS[@]}"; do
    IFS=':' read -r name _ _ <<< "$benchmark_config"
    printf "%-18s" "$name"
    for lang in "${LANG_ORDER[@]}"; do
        time_ms=$(get_result "$name" "$lang")
        if [ "$time_ms" -gt 0 ]; then
            printf " %12s" "$(format_time $time_ms)"
        else
            printf " %12s" "N/A"
        fi
    done
    printf "\n"
done

# Print relative performance (only if Beagle is included and there are other languages)
if lang_enabled beagle && [ ${#LANG_ORDER[@]} -gt 1 ]; then
    echo ""
    echo -e "${BLUE}Relative Performance (Beagle time / Other time):${NC}"
    print_header

    for benchmark_config in "${BENCHMARKS[@]}"; do
        IFS=':' read -r name _ _ <<< "$benchmark_config"
        beagle_time=$(get_result "$name" "beagle")

        printf "%-18s" "$name"

        for lang in "${LANG_ORDER[@]}"; do
            if [ "$beagle_time" -gt 0 ]; then
                if [ "$lang" = "beagle" ]; then
                    printf " %12s" "1.00x"
                else
                    other_time=$(get_result "$name" "$lang")
                    if [ "$other_time" -gt 0 ]; then
                        ratio=$(echo "scale=2; $beagle_time / $other_time" | bc)
                        printf " %12s" "${ratio}x"
                    else
                        printf " %12s" "N/A"
                    fi
                fi
            else
                printf " %12s" "N/A"
            fi
        done

        printf "\n"
    done
fi

rm -f "$RESULTS_FILE"

echo ""
echo -e "${GREEN}Benchmark comparison complete.${NC}"
echo ""
echo "Notes:"
echo "  - Times are averages of $ITERATIONS runs"
echo "  - Relative: <1.0x means Beagle is faster, >1.0x means Beagle is slower"
