# Beagle Friction Log

A log of issues, confusing errors, and missing features encountered while building a real-world application.

## Application: File Sync Status Checker
A tool that monitors a directory and reports file changes to an HTTP endpoint.

---

## Issues Encountered

### Issue 1: Unhelpful Error Messages - Position Only
**Task**: Running initial version of file_sync_checker.bg
**Error**: `Expected expression but found Colon at position 505`
**Problem**: The error gives only a byte position, not a line number or surrounding context. Impossible to know where the error is in a multi-file project or even in a single file.
**Workaround**: Had to manually count characters or bisect the file to find the error.
**Severity**: High - very common to encounter during development.
**Suggestion**: Include line number, column, and a snippet of the surrounding code.

### Issue 2: Confusing Error for JavaScript-style Map Syntax
**Task**: Creating simple data structures for change tracking
**Code Tried**: `{ type: "added", path: path }`
**Error**: `Expected expression but found Colon at position 505`
**Problem**: Beagle uses Clojure/Ruby-style map literals `{:key value}`, not JavaScript-style `{key: value}`. The error message doesn't hint at this.
**Correct Syntax**: `{:type "added", :path path}`
**Suggestion**: Parser should detect `{ identifier:` pattern and emit a helpful error like:
  `"Invalid map syntax. Did you mean {:type "added"} instead of {type: "added"}? Beagle uses Clojure-style map literals."`
**Severity**: Medium - confusing for newcomers from JS/Python backgrounds.

### Issue 3: No Octal Literal Support
**Task**: Setting file permissions with `fs-mkdir`
**Code Tried**: `fs-mkdir(path, 0o755)`
**Error**: `Expected comma ',' but found o755 at position 15`
**Problem**: Octal literals (`0o755`, `0o644`) are not supported. Must convert manually to decimal.
**Workaround**: Use decimal equivalents (`493` for `0o755`, `420` for `0o644`)
**Impact**: Error-prone when working with Unix permissions. Easy to get wrong.
**Suggestion**: Add octal literal support or at least provide a clearer error message.

### Issue 4: No REPL or -e Flag for Quick Testing
**Task**: Quick syntax testing
**Problem**: No way to quickly test expressions without creating a file. No `-e 'expr'` flag or REPL mode.
**Impact**: Slows down development iteration when testing small things.
**Workaround**: Create temporary test files.

### Issue 5: No TCP Client Examples
**Task**: Send file change notifications via TCP
**Problem**: The `tcp-connect-async`, `tcp-write-async` functions exist but there are zero examples of how to use them.
**Impact**: Had to reverse-engineer the API from builtin function signatures in Rust code.
**Workaround**: Read the builtins.rs source and experiment.
**Note**: The async TCP API actually works well once you figure it out. The event loop + result polling pattern is reasonable.

### Issue 6: Misleading Examples Use Unnecessary Prefixes
**Task**: Using stdlib functions
**Problem**: Examples like `tcp_basic_test.bg` do `use beagle.core as core` then `core/fs-readdir()`. This suggests the prefix is required.
**Reality**: Builtins are globally available. Just `fs-readdir()` works fine.
**Impact**: Copied verbose pattern unnecessarily.
**Suggestion**: Update examples to not use unnecessary imports/prefixes.

### Issue 7: Result Types Are Magic Numbers
**Task**: Interpreting TCP operation results
**Problem**: `tcp-result-pop` returns numeric type codes (1 = success, 2 = error?). No constants or documentation.
**Workaround**: Trial and error to understand what codes mean.
**Suggestion**: Export constants like `TCP_RESULT_CONNECTED`, `TCP_RESULT_ERROR` or document the codes.

---

## Missing Features

### MF1: No TCP Client in Effect System - FIXED
**Need**: Connect to remote servers (webhooks, APIs, etc.)
**Problem**: `beagle.socket` only exposed server-side operations. No `connect` function existed.
**Fix Applied**: Added `IOAction.TcpConnect` to beagle.async.bg and `socket/connect(host, port)` to beagle.socket.bg.
**Result**: TCP client code went from ~30 lines of event loop management to 3 lines:
```
let conn = socket/connect(host, port)
socket/write(conn, data)
socket/close(conn)
```

### MF2: No HTTP Client
**Need**: Send webhook notifications about file changes
**Status**: No built-in HTTP client. Would need to implement via FFI to libcurl or build on raw TCP.
**Update**: Raw TCP works at low level, so could build minimal HTTP/1.1 client on top.

### MF2: No JSON Serialization
**Need**: Format change data for HTTP POST
**Status**: No built-in JSON encoder/decoder. Found `slow_json_parser.bg` in examples but no encoder.
**Workaround**: Manual string concatenation works for simple cases but is error-prone.

### MF3: No String Interpolation
**Need**: Building strings with embedded values
**Impact**: Lots of `++` concatenation: `"size: " ++ to-string(size) ++ " bytes"`
**Suggestion**: String interpolation like `"size: ${size} bytes"` would reduce noise.

---

## Positive Observations

### P1: File System APIs Work Well
The `fs-readdir`, `fs-is-file?`, `fs-file-size`, `fs-mkdir`, `fs-unlink`, `fs-rmdir` functions all work reliably. The synchronous APIs are straightforward.

### P2: TCP Networking Is Functional
Once you figure out the async event loop pattern, TCP networking works well. Was able to build a working HTTP client with connection, send, and close.

### P3: Map Operations Are Solid
`assoc`, `get`, `keys`, `length` all work as expected. The `{:key value}` literal syntax is clean.

### P4: String Operations Are Comprehensive
`++` concatenation, `replace`, `to-string`, `length` cover most needs. The stdlib has good string functions.

---

## Summary

**Application Built**: File Sync Status Checker
- Scans directories for files
- Tracks file sizes and detects changes (add/modify/delete)
- Writes human-readable reports to files
- Sends JSON change notifications via HTTP POST to webhook endpoints

**Time to Build**: ~30 minutes once syntax was understood

**Biggest Friction Points**:
1. Error messages with byte positions instead of line numbers
2. No helpful error for common syntax mistakes (JS-style `{key: value}`)
3. No examples for TCP client usage
4. Magic numbers for TCP result types

**Recommendations for Language Authors**:
1. Add line/column numbers to all error messages
2. Detect common syntax mistakes and suggest fixes
3. Add more networking examples
4. Document or export constants for API result codes
5. Consider string interpolation syntax
6. Consider octal literal support for Unix permissions

**Overall**: Beagle is usable for real applications. The core primitives (files, networking, data structures) work. The main friction is documentation and error messages, not missing functionality.

