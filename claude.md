# Beagle Language Project

## Overview
Beagle is a dynamically-typed, functional programming language that compiles directly to ARM64 machine code (macOS only). Inspired by Clojure, it aims to bring the best of dynamic languages to a modern, high-performance runtime without JVM overhead.

## Key Architecture
- **Compilation Pipeline**: AST → IR → ARM64 machine code (no VM)
- **Runtime**: Rust-based with hand-written parser and ARM64 code generator
- **Memory Management**: 3 modern GC implementations (mark-sweep, compacting, generational)
- **Performance**: Already outperforms Ruby 2x, runs 30% slower than Node.js

## Language Features
- Dynamic typing with structs, enums, closures, atoms, threads
- Namespace system for code organization
- Tail call optimization (currently only way to do loops)
- Multi-threading with thread-safe GC options
- FFI for C interop
- Built-in debugger with runtime introspection

## Development Commands
```bash
# Build and run
cargo run -- resources/example.bg

# Run tests
cargo run -- --all-tests

# Run with debugging
cargo run -- --debug resources/example.bg

# Show compilation times
cargo run -- --show-times resources/example.bg

# Different GC backends via features
cargo run --features compacting -- resources/example.bg
cargo run --features mark-and-sweep -- resources/example.bg
cargo run --features generational -- resources/example.bg

# Code formatting
cargo fmt
cargo clippy --fix --allow-dirty --allow-staged

# Testing after changes
# ALWAYS run cargo fmt and tests after making code changes
cargo fmt
cargo run -- --all-tests
# For specific GC implementations, use the appropriate feature flag:
cargo run --features generation-v2 -- --all-tests

# Trailing whitespace cleanup
# For non-Rust files (*.bg, *.md, *.toml, etc.), remove trailing whitespace:
sed -i '' 's/[[:space:]]*$//' path/to/file
# Note: cargo fmt handles this automatically for Rust files
```

## Project Structure
- `src/main.rs` - Entry point and runtime initialization
- `src/parser.rs` - Hand-written parser
- `src/ast.rs` - AST definitions
- `src/compiler.rs` - Compilation orchestration
- `src/ir.rs` - Intermediate representation
- `src/machine_code/` - ARM64 code generation
- `src/gc/` - Multiple garbage collector implementations
- `src/runtime.rs` - Runtime system and memory management
- `resources/` - Example Beagle programs and tests

## Example Beagle Code
```beagle
namespace example

fn fib(n) {
    if n <= 1 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn main() {
    println(fib(30))
}
```

## Testing System
Beagle uses a simple but effective testing system:

**Test Requirements:**
- Tests must be `.bg` files in the `resources/` directory
- **Must contain the exact string `"// Expect"`** to be included in `--all-tests`
- Expected output follows immediately after `// Expect` on lines starting with `//`

**Test Format:**
```beagle
namespace example_test

fn main() {
    println("Hello")
    println(42)
    "done"  // Return value also appears in output
}

// Expect
// Hello
// 42
// done
```

**Special Annotations:**
- `// thread-safe` - Only runs with thread-safe GC features enabled
- `// no-std` - Sets no_std flag automatically

**Test Execution:**
- `cargo run -- --all-tests` runs all tests with `// Expect`
- Output comparison is exact - whitespace and formatting must match
- Tests use `TestPrinter` to capture all `println()` and `print()` calls
- Each test runs in a fresh runtime environment

## Current Status
Early proof-of-concept with solid foundations. Missing many language features but demonstrates promising performance characteristics. Active development focuses on GC optimization and language feature completion.

## Goals
- Build production-ready dynamic language
- GUI debugger frontend written in Beagle itself
- Full standard library and better FFI
- Make dynamic languages popular again with modern performance