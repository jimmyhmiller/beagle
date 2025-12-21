#!/usr/bin/env bash

# Benchmark comparison script for Beagle vs Node.js vs Ruby vs Python
# Usage: ./benchmarks/run_benchmarks.sh [--no-build] [--verbose|-v] [--verify] [--beagle-only] [--languages LANGS] [iterations]
#
# Options:
#   --no-build        Skip building Beagle (use existing binary)
#   --verbose,-v      Show output from each benchmark run
#   --verify          Verify Beagle output matches Node.js output (then exit)
#   --beagle-only     Only run Beagle benchmarks (skip Node.js, Ruby, Python)
#   --languages LANGS Comma-separated list of languages to compare (e.g., beagle,ruby,node)
#                     Available: beagle, node, ruby, python
#   iterations        Number of iterations per benchmark (default: 3)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
SKIP_BUILD=false
VERBOSE=false
VERIFY=false
BEAGLE_ONLY=false
ITERATIONS=3
LANGUAGES=""

# Parse arguments (need to handle --languages with its value)
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-build)
            SKIP_BUILD=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --beagle-only)
            BEAGLE_ONLY=true
            shift
            ;;
        --languages|-l)
            LANGUAGES="$2"
            shift 2
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                ITERATIONS=$1
            fi
            shift
            ;;
    esac
done

# Determine which languages to run
RUN_BEAGLE=false
RUN_NODE=false
RUN_RUBY=false
RUN_PYTHON=false

if [ -n "$LANGUAGES" ]; then
    # Parse comma-separated languages
    IFS=',' read -ra LANG_ARRAY <<< "$LANGUAGES"
    for lang in "${LANG_ARRAY[@]}"; do
        lang=$(echo "$lang" | tr '[:upper:]' '[:lower:]' | xargs)  # lowercase and trim
        case $lang in
            beagle) RUN_BEAGLE=true ;;
            node|nodejs|node.js) RUN_NODE=true ;;
            ruby) RUN_RUBY=true ;;
            python|python3) RUN_PYTHON=true ;;
            *)
                echo "Unknown language: $lang"
                echo "Available: beagle, node, ruby, python"
                exit 1
                ;;
        esac
    done
elif [ "$BEAGLE_ONLY" = true ]; then
    RUN_BEAGLE=true
else
    # Default: run all
    RUN_BEAGLE=true
    RUN_NODE=true
    RUN_RUBY=true
    RUN_PYTHON=true
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Official benchmarksgame arguments (single source of truth)
# Format: "name:verify_arg:perf_arg"
# Ordered by Ruby benchmark times (fastest to slowest)
# Note: revcomp and knucleotide args are fasta size (they read from stdin)
BENCHMARKS=(
    "revcomp:1000:25000000"
    "binary_trees:10:21"
    "fasta:1000:25000000"
    "spectral_norm:100:5500"
    "knucleotide:1000:25000000"
    "nbody:1000:50000000"
    "mandelbrot:200:16000"
    "fannkuch_redux:7:12"
)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Beagle Benchmark Comparison Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Iterations per benchmark: $ITERATIONS"

# Show which languages are being compared
LANG_LIST=""
[ "$RUN_BEAGLE" = true ] && LANG_LIST="${LANG_LIST}Beagle, "
[ "$RUN_NODE" = true ] && LANG_LIST="${LANG_LIST}Node.js, "
[ "$RUN_RUBY" = true ] && LANG_LIST="${LANG_LIST}Ruby, "
[ "$RUN_PYTHON" = true ] && LANG_LIST="${LANG_LIST}Python, "
LANG_LIST="${LANG_LIST%, }"  # Remove trailing comma
echo "Languages: $LANG_LIST"
echo ""

BEAGLE="$PROJECT_DIR/target/release/main"

# Build Beagle in release mode (unless skipped or not running Beagle)
if [ "$RUN_BEAGLE" = true ]; then
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

    # Use Node.js fasta to generate input (fastest option)
    local fasta_js="$SCRIPT_DIR/comparison/node/fasta.js"
    if [ -f "$fasta_js" ]; then
        node "$fasta_js" "$size" > "$input_file" 2>/dev/null
    else
        # Fallback to Beagle fasta
        local fasta_bg="$SCRIPT_DIR/benchmarksgame/fasta.bg"
        "$BEAGLE" "$fasta_bg" "$size" > "$input_file" 2>/dev/null
    fi
}

# Benchmark configurations
run_single_benchmark() {
    local name="$1"
    local arg="$2"

    echo -e "${CYAN}Running: $name (arg=$arg)${NC}"

    local beagle_time=0 node_time=0 ruby_time=0 python_time=0
    local input_file=""

    # For stdin-based benchmarks, generate input first
    if needs_stdin_input "$name"; then
        input_file=$(mktemp)
        echo -n "  Generating input... "
        generate_fasta_input "$arg" "$input_file"
        echo "done"
    fi

    # Beagle
    if [ "$RUN_BEAGLE" = true ]; then
        beagle_file="$SCRIPT_DIR/benchmarksgame/$name.bg"
        if [ -f "$beagle_file" ]; then
            echo -n "  Beagle...  "
            if [ -n "$input_file" ]; then
                beagle_time=$(run_benchmark "$BEAGLE $beagle_file < $input_file" "$ITERATIONS")
            else
                beagle_time=$(run_benchmark "$BEAGLE $beagle_file $arg" "$ITERATIONS")
            fi
            echo "$(format_time $beagle_time)"
        else
            echo "  Beagle...  N/A (file not found)"
        fi
    fi

    # Node.js
    if [ "$RUN_NODE" = true ]; then
        node_file="$SCRIPT_DIR/comparison/node/${name}.js"
        if [ -f "$node_file" ]; then
            echo -n "  Node.js... "
            if [ -n "$input_file" ]; then
                node_time=$(run_benchmark "node $node_file < $input_file" "$ITERATIONS")
            else
                node_time=$(run_benchmark "node $node_file $arg" "$ITERATIONS")
            fi
            echo "$(format_time $node_time)"
        else
            echo "  Node.js... N/A"
        fi
    fi

    # Ruby
    if [ "$RUN_RUBY" = true ]; then
        ruby_file="$SCRIPT_DIR/comparison/ruby/${name}.rb"
        if [ -f "$ruby_file" ]; then
            echo -n "  Ruby...    "
            if [ -n "$input_file" ]; then
                ruby_time=$(run_benchmark "ruby $ruby_file < $input_file" "$ITERATIONS")
            else
                ruby_time=$(run_benchmark "ruby $ruby_file $arg" "$ITERATIONS")
            fi
            echo "$(format_time $ruby_time)"
        else
            echo "  Ruby...    N/A"
        fi
    fi

    # Python
    if [ "$RUN_PYTHON" = true ]; then
        python_file="$SCRIPT_DIR/comparison/python/${name}.py"
        if [ -f "$python_file" ]; then
            echo -n "  Python...  "
            if [ -n "$input_file" ]; then
                python_time=$(run_benchmark "python3 $python_file < $input_file" "$ITERATIONS")
            else
                python_time=$(run_benchmark "python3 $python_file $arg" "$ITERATIONS")
            fi
            echo "$(format_time $python_time)"
        else
            echo "  Python...  N/A"
        fi
    fi

    # Clean up input file
    if [ -n "$input_file" ]; then
        rm -f "$input_file"
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

# Build dynamic header based on selected languages
print_header() {
    printf "%-18s" "Benchmark"
    [ "$RUN_BEAGLE" = true ] && printf " %12s" "Beagle"
    [ "$RUN_NODE" = true ] && printf " %12s" "Node.js"
    [ "$RUN_RUBY" = true ] && printf " %12s" "Ruby"
    [ "$RUN_PYTHON" = true ] && printf " %12s" "Python"
    printf "\n"

    printf "%-18s" "-----------------"
    [ "$RUN_BEAGLE" = true ] && printf " %12s" "------------"
    [ "$RUN_NODE" = true ] && printf " %12s" "------------"
    [ "$RUN_RUBY" = true ] && printf " %12s" "------------"
    [ "$RUN_PYTHON" = true ] && printf " %12s" "------------"
    printf "\n"
}

print_header

while read -r name beagle_time node_time ruby_time python_time; do
    printf "%-18s" "$name"
    [ "$RUN_BEAGLE" = true ] && { [ "$beagle_time" -gt 0 ] && printf " %12s" "$(format_time $beagle_time)" || printf " %12s" "N/A"; }
    [ "$RUN_NODE" = true ] && { [ "$node_time" -gt 0 ] && printf " %12s" "$(format_time $node_time)" || printf " %12s" "N/A"; }
    [ "$RUN_RUBY" = true ] && { [ "$ruby_time" -gt 0 ] && printf " %12s" "$(format_time $ruby_time)" || printf " %12s" "N/A"; }
    [ "$RUN_PYTHON" = true ] && { [ "$python_time" -gt 0 ] && printf " %12s" "$(format_time $python_time)" || printf " %12s" "N/A"; }
    printf "\n"
done < "$RESULTS_FILE"

# Count how many languages are enabled
lang_count=0
[ "$RUN_BEAGLE" = true ] && lang_count=$((lang_count + 1))
[ "$RUN_NODE" = true ] && lang_count=$((lang_count + 1))
[ "$RUN_RUBY" = true ] && lang_count=$((lang_count + 1))
[ "$RUN_PYTHON" = true ] && lang_count=$((lang_count + 1))

# Print relative performance (only if Beagle is included and there are other languages)
if [ "$RUN_BEAGLE" = true ] && [ "$lang_count" -gt 1 ]; then
    echo ""
    echo -e "${BLUE}Relative Performance (Beagle time / Other time):${NC}"
    print_header

    while read -r name beagle_time node_time ruby_time python_time; do
        printf "%-18s" "$name"

        if [ "$beagle_time" -gt 0 ]; then
            printf " %12s" "1.00x"

            if [ "$RUN_NODE" = true ]; then
                if [ "$node_time" -gt 0 ]; then
                    ratio=$(echo "scale=2; $beagle_time / $node_time" | bc)
                    printf " %12s" "${ratio}x"
                else
                    printf " %12s" "N/A"
                fi
            fi

            if [ "$RUN_RUBY" = true ]; then
                if [ "$ruby_time" -gt 0 ]; then
                    ratio=$(echo "scale=2; $beagle_time / $ruby_time" | bc)
                    printf " %12s" "${ratio}x"
                else
                    printf " %12s" "N/A"
                fi
            fi

            if [ "$RUN_PYTHON" = true ]; then
                if [ "$python_time" -gt 0 ]; then
                    ratio=$(echo "scale=2; $beagle_time / $python_time" | bc)
                    printf " %12s" "${ratio}x"
                else
                    printf " %12s" "N/A"
                fi
            fi
        else
            [ "$RUN_BEAGLE" = true ] && printf " %12s" "N/A"
            [ "$RUN_NODE" = true ] && printf " %12s" "N/A"
            [ "$RUN_RUBY" = true ] && printf " %12s" "N/A"
            [ "$RUN_PYTHON" = true ] && printf " %12s" "N/A"
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
