# Beagle Language Project

## Overview
Beagle is a dynamically-typed, functional programming language that compiles directly to native machine code (macOS only). Inspired by Clojure, it aims to bring the best of dynamic languages to a modern, high-performance runtime without JVM overhead.

## Key Architecture
- **Compilation Pipeline**: AST → IR → machine code (no VM)
- **Runtime**: Rust-based with hand-written parser and pluggable code generator
- **Code Generation**: Pluggable backend system (ARM64 default, x86-64 available, LLVM/Cranelift planned)
- **Memory Management**: 3 modern GC implementations (mark-sweep, compacting, generational)
- **Performance**: Already outperforms Ruby 2x, runs 30% slower than Node.js

## Language Features
- Dynamic typing with structs, enums, protocols, closures, atoms, keywords
- Persistent collections (vectors, maps, sets) and dedicated mutable-array API
- Namespace system with aliased imports (`use beagle.fs as fs`)
- Control flow: `if`/`match` expressions, `while`/`for`/`loop` expressions, `break(value)`/`continue()` as call syntax
- Multi-arity and variadic (`...rest`) functions, destructuring in `let`/params/match
- String interpolation with `${...}`, two pipe operators (`|>` first-arg, `|>>` last-arg)
- Resumable exceptions (`catch (e, resume)`) — runtime errors included
- Algebraic effects with handlers (`handle ... with ...`, `perform`); delimited continuations (`reset`/`shift`)
- Async I/O, futures, structured concurrency (`with-scope`); OS threads
- Tail call optimization
- FFI for C interop with GC-managed buffers/cells
- Built-in debugger with runtime introspection

**Authoritative language reference:** `docs/language-guide.md`. Read it before writing or reviewing `.bg` code — it's the source of truth for syntax and semantics, and lists the most common mistakes (§28). The `beagle-language` skill loads it on demand.

## Development Commands
```bash
# Build and run
cargo run -- resources/example.bg

# Run tests
cargo run -- test resources/
# Run a single test file
cargo run -- test resources/some_test.bg

# Run with debugging
cargo run -- --debug resources/example.bg

# Show compilation times
cargo run -- --show-times resources/example.bg

# Code generation backends (ARM64 is default)
cargo run --features backend-arm64 -- resources/example.bg
cargo run --target x86_64-apple-darwin --features backend-x86-64 -- resources/example.bg
# Future backends (not yet implemented):
# cargo run --features backend-llvm -- resources/example.bg
# cargo run --features backend-cranelift -- resources/example.bg

# Different GC backends via features
cargo run --features compacting -- resources/example.bg
cargo run --features mark-and-sweep -- resources/example.bg
cargo run --features generational -- resources/example.bg

# Combine backend and GC selection
cargo run --features "backend-arm64 generational" -- resources/example.bg

# Code formatting
cargo fmt
cargo clippy --fix --allow-dirty --allow-staged

# Testing after changes
# ALWAYS run cargo fmt and tests after making code changes
cargo fmt
cargo run -- test resources/
# For specific GC implementations, use the appropriate feature flag:
cargo run --features generation-v2 -- test resources/

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
- `src/backend/` - Pluggable code generation backends
  - `mod.rs` - `CodegenBackend` trait and `Backend` type alias
  - `arm64/` - ARM64 backend implementation (default)
  - `x86_64/` - x86-64 backend implementation (for Rosetta 2)
- `src/x86.rs` - Low-level x86-64 code generation
- `src/machine_code/` - Machine code instruction encoding
  - `mod.rs` - ARM64 instruction encoding
  - `x86_codegen.rs` - x86-64 instruction encoding
- `src/gc/` - Multiple garbage collector implementations
- `src/runtime.rs` - Runtime system and memory management
- `resources/` - Example Beagle programs and tests
- `.cargo/config.toml` - Cross-compilation configuration and cargo aliases

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
    println("fib(30) = ${fib(30)}")
}
```

For idiomatic examples covering effects, protocols, async, persistent collections, etc., see `docs/language-guide.md` and the `.bg` files in `resources/` and `standard-library/`.

## Testing System
Beagle has two complementary test styles; files can mix both. See `docs/language-guide.md` §27 for full details.

**CRITICAL: ALL TESTS MUST PASS**
- **100% test pass rate is required** - partial success is not acceptable
- Any change that breaks existing tests must be fixed or reverted
- "Most tests passing" is considered a failure state
- New features should not break existing functionality

**Discovery:** `cargo run -- test` finds any `.bg` file containing either:
- the marker `// @beagle.core.snapshot`, **or**
- one or more `test "name" { ... }` blocks.

### Snapshot tests
Compare `main()`'s stdout to `//` lines following `// @beagle.core.snapshot`. Comparison is exact (whitespace + formatting).

```beagle
namespace example_test

fn main() {
    println("Hello")
    println(42)
    "done"  // return value of main is also printed
}

// @beagle.core.snapshot
// Hello
// 42
// done
```

### `test "name" { ... }` blocks
Top-level test forms with assertion macros (globally available, no `use`):

```beagle
namespace math_test

fn add(a, b) { a + b }

test "addition" {
    assert-eq!(add(1, 2), 3)
    assert-ne!(add(1, 2), 4)
    assert!(true)
    assert-throws!(fn() { throw("boom") })
}
```

A test passes if its body completes without a thrown exception. The `!` is part of the macro name.

**Special Annotations:**
- `// no-std` - Sets no_std flag automatically
- `// Skip` or `// Skip: reason` - Skip this test (use for flaky/timing-dependent tests)
- `// gc-always` - Run GC on every allocation (for GC stress testing)

**Test Execution:**
- `cargo run -- test resources/` runs all discovered tests in a directory
- `cargo run -- test resources/some_test.bg` runs a single test file
- `cargo run -- test --update-snapshots resources/some_test.bg` updates snapshot to match actual output
- With no path, `test/` or `tests/` is used if present
- Files are dependency-sorted so imports work
- Tests use `TestPrinter` to capture all `println()` / `print()` calls
- Each test runs in a fresh runtime environment

## SSA work — read the spec

Beagle has tried SSA several times and each attempt blew up on the same
architectural mistakes (orphan blocks, FP-class loss, force-spill across
calls, missing critical-edge split, dead-coded coalescer, missing TCO).

The next attempt has a written contract:

- **`docs/SSA_ARCHITECTURE.md`** — the eight invariants, the pipeline, the
  forbidden-patterns table, and the rollout phases. Read it before
  touching `src/cfg/`, `src/ssa/`, `src/register_allocation/`, `src/ir.rs`,
  `src/lir/`, `src/compiler.rs`, or backend regalloc/spill code. A diff
  that contradicts the spec is a blocking review failure; change the spec
  in the same commit if there's a real reason.

## Current Status
Early proof-of-concept with solid foundations. Missing many language features but demonstrates promising performance characteristics. Active development focuses on GC optimization and language feature completion.

## Cross-Compilation and Rosetta 2

Beagle supports cross-compilation to x86-64 on Apple Silicon Macs, allowing the compiler to run under Rosetta 2.

### How It Works
- **Native ARM64 (default)**: Beagle compiles to native ARM64 machine code on Apple Silicon
- **x86-64 via Rosetta 2**: Build the compiler as an x86-64 binary, which runs under Rosetta 2 and emits x86-64 machine code

### Rosetta 2 and JIT
Rosetta 2 fully supports JIT compilation:
- The `MAP_JIT` mmap flag is supported for executable memory
- W^X (write-xor-execute) memory transitions work correctly
- The emitted x86-64 code is translated by Rosetta at execution time

### Setup (One-Time)
```bash
rustup target add x86_64-apple-darwin
```

### Building for x86-64
```bash
# Using cargo aliases (defined in .cargo/config.toml)
cargo build-x86       # Debug build
cargo release-x86     # Release build
cargo test-x86        # Run tests
cargo run-x86 -- resources/example.bg

# Or explicitly
cargo build --target x86_64-apple-darwin --features backend-x86-64
```

### Performance Notes
- Expect ~2x overhead when running under Rosetta 2 (translation cost)
- Both the Beagle compiler and the generated code are translated
- Useful for testing x86-64 code generation without native hardware
- Native ARM64 builds are recommended for production use on Apple Silicon

### Architecture Constraint
You cannot mix architectures in a single process. When running under Rosetta 2:
- The host process must be x86-64
- The JIT-generated code must be x86-64
- You cannot emit native ARM64 code from an x86-64 process

## Goals
- Build production-ready dynamic language
- GUI debugger frontend written in Beagle itself
- Full standard library and better FFI
- Make dynamic languages popular again with modern performance