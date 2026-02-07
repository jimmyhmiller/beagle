# 7GUIs Friction Log

Bugs and language limitations encountered while implementing the 7GUIs benchmark in Beagle.

## Language Issues

### 1. `nil` vs `null` naming confusion
- **Severity:** Low (compile error is clear)
- **Details:** Coming from Clojure, I expected `nil` but Beagle uses `null`. The error message "Undefined variable: nil" is clear enough.

### 2. `!function()` does not work — must use `not(function())`
- **Severity:** Medium
- **Details:** The `!` prefix operator works on variables (`!hover`) but not directly before function calls (`!starts-with(x, y)` parses as calling a function named `!starts-with`). Workaround: use `not(starts-with(x, y))`.
- **Where:** cells.bg formula parser

### 3. No anonymous struct / record literals for quick multi-return
- **Severity:** Medium
- **Details:** JavaScript-style `{ key: value }` is not valid for creating anonymous records. Must define a named `struct` for every shape of data, even for internal helper returns. Maps (`{:key value}`) exist but require `get(map, :key)` to access fields — no dot syntax. This leads to struct proliferation (e.g., `TextInputResult`, `CellRef`, `EditingInfo`) where a simple tuple or anonymous record would suffice.
- **Where:** raylib_gui.bg (TextInputResult), cells.bg (CellRef, EditingInfo)

### 4. `continue` requires parentheses: `continue()`
- **Severity:** Low
- **Details:** Unlike `break(value)` which naturally takes an argument, `continue()` with empty parens feels inconsistent with other languages. Easy to forget the `()`.
- **Where:** circle_drawer.bg

### 5. No float formatting built-in
- **Severity:** Medium
- **Details:** `to-string()` on floats produces full precision output. Had to write manual `format-temp()` and `format-number()` helpers using `truncate()` and string concatenation. A `format(value, decimals)` or `to-string(value, precision)` would be very helpful.
- **Where:** temperature.bg, cells.bg, timer.bg

### 6. `to-number()` throws on invalid input instead of returning null
- **Severity:** Medium
- **Details:** For GUI text input parsing, you almost always want a "try parse" that returns null on failure. Had to wrap `to-number()` in `try { ... } catch (e) { null }` everywhere. A `parse-number(s)` that returns null on failure would be much more ergonomic.
- **Where:** temperature.bg, cells.bg

## Runtime Issues

### 7. Intermittent segfault in GC during GUI loops
- **Severity:** High (but non-deterministic)
- **Details:** When running temperature converter or other programs with struct-heavy game loops, occasionally get `EXC_BAD_ACCESS` in JIT code. The crash shows a tagged integer (e.g., value 30) being dereferenced as a struct pointer. This suggests GC is moving/collecting a struct that's still referenced from the stack. The crash is intermittent — runs fine most of the time, but occasionally triggers on startup. Possibly a GC root scanning issue with deeply nested struct values in function parameters.
- **Where:** temperature.bg, possibly any program allocating many short-lived structs per frame

## FFI / Raylib Issues

### 8. No friction with FFI itself
- **Positive note:** The FFI system worked flawlessly for all raylib bindings needed. The variadic argument support, F32 type, and string passing all worked on the first try. Very impressive.

## GUI Architecture Notes

### 9. Immediate-mode GUI works well in Beagle
- **Positive note:** The functional, immutable-state-per-frame approach maps naturally to Beagle's semantics. Each frame is a pure function: `state -> (draw side effects, new state)`. Tail-call optimized `game-loop(new_state)` is a clean pattern.

### 10. Array-heavy state is slow for large collections
- **Severity:** Low (for this benchmark)
- **Details:** The cells spreadsheet (26*100 = 2600 cells) uses a flat array. Operations like `set-cell` that rebuild the entire array on each edit are O(n). Persistent vectors (`vec-push`, `vec-assoc`) would be more efficient but require different access patterns.
- **Where:** cells.bg

### 11. No string `trim`, `starts-with`, `ends-with` in std
- **Severity:** Low
- **Details:** Had to implement `trim()`, `starts-with()`, and `ends-with()` manually in cells.bg. These are common enough to belong in the standard library. (Note: `contains?` and `index-of` do exist.)
- **Where:** cells.bg, crud.bg
