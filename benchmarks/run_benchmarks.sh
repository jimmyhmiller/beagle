#!/usr/bin/env bash

# Benchmark comparison script for Beagle vs Node.js vs Ruby vs Python
# Usage: ./benchmarks/run_benchmarks.sh [--no-build] [--verbose|-v] [--verify] [--beagle-only] [iterations]
#
# Options:
#   --no-build     Skip building Beagle (use existing binary)
#   --verbose,-v   Show output from each benchmark run
#   --verify       Verify Beagle output matches Node.js output (then exit)
#   --beagle-only  Only run Beagle benchmarks (skip Node.js, Ruby, Python)
#   iterations     Number of iterations per benchmark (default: 3)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
SKIP_BUILD=false
VERBOSE=false
VERIFY=false
BEAGLE_ONLY=false
ITERATIONS=3

for arg in "$@"; do
    case $arg in
        --no-build)
            SKIP_BUILD=true
            ;;
        --verbose|-v)
            VERBOSE=true
            ;;
        --verify)
            VERIFY=true
            ;;
        --beagle-only)
            BEAGLE_ONLY=true
            ;;
        *)
            if [[ "$arg" =~ ^[0-9]+$ ]]; then
                ITERATIONS=$arg
            fi
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Official benchmarksgame arguments (single source of truth)
# Format: "name:verify_arg:perf_arg"
BENCHMARKS=(
    "binary_trees:10:21"
    "fannkuch_redux:7:12"
    "fasta:1000:25000000"
    "mandelbrot:200:16000"
    "nbody:1000:50000000"
    "spectral_norm:100:5500"
)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Beagle Benchmark Comparison Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Iterations per benchmark: $ITERATIONS"
echo ""

BEAGLE="$PROJECT_DIR/target/release/main"

# Build Beagle in release mode (unless skipped)
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

cd "$PROJECT_DIR"

# Verify mode: check Beagle output against Node.js
if [ "$VERIFY" = true ]; then
    echo -e "${BLUE}Verifying Beagle output against Node.js...${NC}"
    echo ""

    ALL_PASSED=true

    for benchmark_config in "${BENCHMARKS[@]}"; do
        IFS=':' read -r name verify_arg perf_arg <<< "$benchmark_config"

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

        beagle_out=$("$BEAGLE" "$beagle_file" "$verify_arg" 2>&1) || true
        node_out=$(node "$node_file" "$verify_arg" 2>&1) || true

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

# Function to format time
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

# Temporary file to store results for summary
RESULTS_FILE=$(mktemp)

# Benchmark configurations
run_single_benchmark() {
    local name="$1"
    local arg="$2"

    echo -e "${CYAN}Running: $name (arg=$arg)${NC}"

    local beagle_time=0 node_time=0 ruby_time=0 python_time=0

    # Beagle
    beagle_file="$SCRIPT_DIR/benchmarksgame/$name.bg"
    if [ -f "$beagle_file" ]; then
        echo -n "  Beagle...  "
        beagle_time=$(run_benchmark "$BEAGLE $beagle_file $arg" "$ITERATIONS")
        echo "$(format_time $beagle_time)"
    else
        echo "  Beagle...  N/A (file not found)"
    fi

    # Node.js
    if [ "$BEAGLE_ONLY" = false ]; then
        node_file="$SCRIPT_DIR/comparison/node/${name}.js"
        if [ -f "$node_file" ]; then
            echo -n "  Node.js... "
            node_time=$(run_benchmark "node $node_file $arg" "$ITERATIONS")
            echo "$(format_time $node_time)"
        else
            echo "  Node.js... N/A"
        fi
    fi

    # Ruby
    if [ "$BEAGLE_ONLY" = false ]; then
        ruby_file="$SCRIPT_DIR/comparison/ruby/${name}.rb"
        if [ -f "$ruby_file" ]; then
            echo -n "  Ruby...    "
            ruby_time=$(run_benchmark "ruby $ruby_file $arg" "$ITERATIONS")
            echo "$(format_time $ruby_time)"
        else
            echo "  Ruby...    N/A"
        fi
    fi

    # Python
    if [ "$BEAGLE_ONLY" = false ]; then
        python_file="$SCRIPT_DIR/comparison/python/${name}.py"
        if [ -f "$python_file" ]; then
            echo -n "  Python...  "
            python_time=$(run_benchmark "python3 $python_file $arg" "$ITERATIONS")
            echo "$(format_time $python_time)"
        else
            echo "  Python...  N/A"
        fi
    fi

    echo ""

    # Store results
    echo "$name $beagle_time $node_time $ruby_time $python_time" >> "$RESULTS_FILE"
}

# Run all benchmarks (using official benchmarksgame performance arguments)
for benchmark_config in "${BENCHMARKS[@]}"; do
    IFS=':' read -r name verify_arg perf_arg <<< "$benchmark_config"
    run_single_benchmark "$name" "$perf_arg"
done

# Print summary table
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}              RESULTS${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$BEAGLE_ONLY" = true ]; then
    # Beagle-only summary
    printf "%-18s %12s\n" "Benchmark" "Beagle"
    printf "%-18s %12s\n" "-----------------" "------------"

    while read -r name beagle_time node_time ruby_time python_time; do
        printf "%-18s " "$name"
        [ "$beagle_time" -gt 0 ] && printf "%12s" "$(format_time $beagle_time)" || printf "%12s" "N/A"
        printf "\n"
    done < "$RESULTS_FILE"
else
    # Full comparison summary
    printf "%-18s %12s %12s %12s %12s\n" "Benchmark" "Beagle" "Node.js" "Ruby" "Python"
    printf "%-18s %12s %12s %12s %12s\n" "-----------------" "------------" "------------" "------------" "------------"

    while read -r name beagle_time node_time ruby_time python_time; do
        printf "%-18s " "$name"
        [ "$beagle_time" -gt 0 ] && printf "%12s " "$(format_time $beagle_time)" || printf "%12s " "N/A"
        [ "$node_time" -gt 0 ] && printf "%12s " "$(format_time $node_time)" || printf "%12s " "N/A"
        [ "$ruby_time" -gt 0 ] && printf "%12s " "$(format_time $ruby_time)" || printf "%12s " "N/A"
        [ "$python_time" -gt 0 ] && printf "%12s" "$(format_time $python_time)" || printf "%12s" "N/A"
        printf "\n"
    done < "$RESULTS_FILE"

    # Print relative performance (only for full comparison)
    echo ""
    echo -e "${BLUE}Relative Performance (Beagle time / Other time):${NC}"
    printf "%-18s %12s %12s %12s %12s\n" "Benchmark" "Beagle" "Node.js" "Ruby" "Python"
    printf "%-18s %12s %12s %12s %12s\n" "-----------------" "------------" "------------" "------------" "------------"

    while read -r name beagle_time node_time ruby_time python_time; do
        printf "%-18s " "$name"

        if [ "$beagle_time" -gt 0 ]; then
            printf "%12s " "1.00x"

            if [ "$node_time" -gt 0 ]; then
                ratio=$(echo "scale=2; $beagle_time / $node_time" | bc)
                printf "%12s " "${ratio}x"
            else
                printf "%12s " "N/A"
            fi

            if [ "$ruby_time" -gt 0 ]; then
                ratio=$(echo "scale=2; $beagle_time / $ruby_time" | bc)
                printf "%12s " "${ratio}x"
            else
                printf "%12s " "N/A"
            fi

            if [ "$python_time" -gt 0 ]; then
                ratio=$(echo "scale=2; $beagle_time / $python_time" | bc)
                printf "%12s" "${ratio}x"
            else
                printf "%12s" "N/A"
            fi
        else
            printf "%12s %12s %12s %12s" "N/A" "N/A" "N/A" "N/A"
        fi

        printf "\n"
    done < "$RESULTS_FILE"
fi

rm -f "$RESULTS_FILE"

echo ""
echo -e "${GREEN}Benchmark comparison complete.${NC}"
echo ""
echo "Notes:"
echo "  - Times are averages of $ITERATIONS runs"
echo "  - Relative: <1.0x means Beagle is faster, >1.0x means Beagle is slower"
