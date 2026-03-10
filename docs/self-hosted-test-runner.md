# Self-Hosted Test Runner: Problems Found

We attempted to write the test runner in Beagle itself — a Beagle program that
discovers `.bg` test files, `eval`s their source, and calls `main()`. This
documents what broke and what would need to change to make it work.

## What worked

- Reading the filesystem (`core/fs-readdir`, `fs/blocking-read-file`)
- Parsing test metadata (skip annotations, namespace extraction)
- `eval` of simple test files that don't use imports or async effects
- `try/catch` around test execution to report failures gracefully

## Problem 1: Imports don't resolve

Test files use `use other_namespace as alias` to import dependencies. When you
`eval` a file's source as a string, the compiler doesn't know where to find
those namespaces on disk. The normal compilation path (`runtime.compile(file)`)
resolves files relative to the source file's path, but `eval` compiles a bare
string with no file context.

Example: `import_test.bg` has `use fib as fib` — eval produces:

```
Namespace alias not found: fib
```

### What would fix it

Expose a `load-file` or `compile-file` builtin that goes through the full file
compilation path (with path resolution), rather than just `eval` of a string.

## Problem 2: Async effects crash without a handler

Many tests define `fn main()` and rely on the implicit `beagle.async/__main__`
wrapper that the normal entry path provides. This wrapper installs an async
effect handler so that `perform` calls inside main (for file I/O, timers,
sockets, etc.) have something to handle them.

When we `eval` a test file and then call `main()` directly via
`eval-in-ns("main()", ns)`, there is no enclosing async handler. Any test that
uses async I/O hits:

```
shift/perform without enclosing reset/handle. This is a compiler bug -
shift/perform must be inside a reset/handle block.
```

This is a hard abort (panic in an `extern "C"` function), not a catchable
exception.

### What would fix it

Either:
- Wrap the `main()` call in `beagle.async/__main__` from the test runner
- Make the missing-handler case a catchable exception instead of a panic
- Expose a `run-with-async-handler(thunk)` utility

## Problem 3: No output capture

Snapshot tests need to capture what `println` produces and compare it to
expected output. There's no Beagle-level mechanism to redirect or capture
printed output. The `TestPrinter` that handles this is a Rust-side concept
configured at runtime initialization.

### What would fix it

A `with-captured-output(thunk)` form that returns the printed output as a
string, or a dynamic variable for stdout redirection.

## Current solution

Each test runs as a subprocess (`beag run --test <file>`), giving it a
completely fresh runtime. This is simple and correct — no shared state, no
reset needed — but slower than in-process execution.

## Path forward

To get a self-hosted test runner, we'd need at minimum:
1. A `load-file` builtin (or make `eval` path-aware)
2. Output capture at the Beagle level
3. Either automatic handler wrapping or a way to install handlers from the runner
