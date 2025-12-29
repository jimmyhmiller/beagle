#!/bin/bash
# Compare Beagle vs Clojure benchmark results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAGLE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Running Beagle benchmark..."
BEAGLE_OUTPUT=$(cd "$BEAGLE_ROOT" && cargo run --release -- resources/benchmarks/clojure_comparison.bg 2>/dev/null | grep -E "^[A-Za-z].*: [0-9]+ ms$")

echo "Running Clojure benchmark..."
CLOJURE_OUTPUT=$(cd "$SCRIPT_DIR" && clj -M clojure_comparison.clj 2>/dev/null | grep -E "^[A-Za-z].*: [0-9]+ ms$")

echo ""
echo "=============================================="
echo "        Beagle vs Clojure Comparison"
echo "=============================================="
echo ""
printf "%-35s %10s %10s %12s\n" "Benchmark" "Beagle" "Clojure" "Comparison"
printf "%-35s %10s %10s %12s\n" "---------" "------" "-------" "----------"

# Parse and compare results
while IFS= read -r beagle_line; do
    # Extract benchmark name and time
    name=$(echo "$beagle_line" | sed 's/: [0-9]* ms$//')
    beagle_ms=$(echo "$beagle_line" | grep -oE '[0-9]+' | tail -1)

    # Find matching Clojure result
    clojure_line=$(echo "$CLOJURE_OUTPUT" | grep "^$name:")
    if [ -n "$clojure_line" ]; then
        clojure_ms=$(echo "$clojure_line" | grep -oE '[0-9]+' | tail -1)

        # Calculate comparison
        if [ "$beagle_ms" -eq 0 ] && [ "$clojure_ms" -eq 0 ]; then
            comparison="~same"
        elif [ "$beagle_ms" -eq 0 ]; then
            comparison="Beagle faster"
        elif [ "$clojure_ms" -eq 0 ]; then
            comparison="Clojure faster"
        elif [ "$beagle_ms" -lt "$clojure_ms" ]; then
            ratio=$(echo "scale=1; $clojure_ms / $beagle_ms" | bc)
            comparison="${ratio}x faster"
        elif [ "$beagle_ms" -gt "$clojure_ms" ]; then
            ratio=$(echo "scale=1; $beagle_ms / $clojure_ms" | bc)
            comparison="${ratio}x slower"
        else
            comparison="~same"
        fi

        printf "%-35s %8s ms %8s ms %12s\n" "$name" "$beagle_ms" "$clojure_ms" "$comparison"
    fi
done <<< "$BEAGLE_OUTPUT"

echo ""
echo "=============================================="
