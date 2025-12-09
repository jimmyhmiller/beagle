#!/bin/bash

# Compare ARM64 vs x86-64 backend outputs
# Categories: pass, mismatch, crash, timeout
# Uses thread-safe GC feature for both backends

TIMEOUT_SECS=10
LONG_TIMEOUT_SECS=180  # For computationally intensive tests like binary_tree (~100s on x86-64)

PASSED=()
MISMATCH=()
CRASHED=()
TIMEOUT=()

# Pre-build both binaries
echo "Building x86-64 binary..."
cargo build --target x86_64-apple-darwin --features "backend-x86-64 thread-safe" --release 2>/dev/null
X86_BIN="target/x86_64-apple-darwin/release/main"

echo "Building ARM64 binary..."
cargo build --features "thread-safe" --release 2>/dev/null
ARM_BIN="target/release/main"

if [ ! -f "$X86_BIN" ] || [ ! -f "$ARM_BIN" ]; then
    echo "Error: Failed to build binaries"
    exit 1
fi

echo "Running tests..."
echo ""

for f in resources/*.bg; do
    # Skip files without // Expect
    if ! grep -q "// Expect" "$f" 2>/dev/null; then
        continue
    fi

    name=$(basename "$f")

    # Use longer timeout for known slow tests
    if [[ "$name" == "binary_tree.bg" ]]; then
        test_timeout=$LONG_TIMEOUT_SECS
    else
        test_timeout=$TIMEOUT_SECS
    fi

    # Run x86-64 binary directly
    x86_out=$(timeout $test_timeout "$X86_BIN" "$f" 2>/dev/null)
    x86_exit=$?

    if [ $x86_exit -eq 124 ]; then
        TIMEOUT+=("$name")
        echo "TIMEOUT: $name"
        continue
    fi

    if [ $x86_exit -ne 0 ]; then
        CRASHED+=("$name:exit=$x86_exit")
        echo "CRASH: $name (exit $x86_exit)"
        continue
    fi

    # Run ARM64 binary directly
    arm_out=$(timeout $test_timeout "$ARM_BIN" "$f" 2>/dev/null)

    if [ "$x86_out" = "$arm_out" ]; then
        PASSED+=("$name")
        echo "PASS: $name"
    else
        MISMATCH+=("$name")
        echo "MISMATCH: $name"
    fi
done

echo ""
echo "========== SUMMARY =========="
echo "Passed: ${#PASSED[@]}"
echo "Mismatch: ${#MISMATCH[@]}"
echo "Crashed: ${#CRASHED[@]}"
echo "Timeout: ${#TIMEOUT[@]}"
echo ""

if [ ${#MISMATCH[@]} -gt 0 ]; then
    echo "=== Mismatched files ==="
    printf '%s\n' "${MISMATCH[@]}"
    echo ""
fi

if [ ${#CRASHED[@]} -gt 0 ]; then
    echo "=== Crashed files ==="
    printf '%s\n' "${CRASHED[@]}"
    echo ""
fi

if [ ${#TIMEOUT[@]} -gt 0 ]; then
    echo "=== Timeout files ==="
    printf '%s\n' "${TIMEOUT[@]}"
    echo ""
fi
