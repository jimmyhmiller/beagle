# Beagle Language Friction Log

## Overview
Building a Todo List Manager application to test Beagle language features.
Documenting all issues, confusions, and pain points encountered.

---

## Issues Found

### Issue #1: String interpolation with `task.id` fails - "Expected string, got Int"

**Location**: `format_task` function
**Code**:
```beagle
"${status_str} #${task.id} [${priority_str}] ${task.title}${tags_str}"
```

**Error**:
```
Uncaught exception:
SystemError.TypeError { message: "Expected string, got Int", location: null }
```

**Stack trace showed**: The error occurred in `join` being called from `format_task`.

**Analysis**: It seems the `join` function expects all elements to be strings, but when I'm joining tags, if tags is empty this shouldn't be an issue. OR the string interpolation `${task.id}` might be the problem since `task.id` is an integer.

**Friction**:
1. The error message doesn't point to which specific variable caused the issue
2. String interpolation should auto-convert integers to strings
3. Stack trace shows internal Rust functions which isn't helpful for users

**Workaround attempted**: Need to use `to-string()` explicitly

---

### Issue #2: `join()` fails on array literals containing strings

**Location**: Any code using `join` with an array literal
**Code**:
```beagle
let tags = ["learning", "programming"]
let result = join(tags, ", ")  // FAILS!
```

**Error**:
```
SystemError.TypeError { message: "Expected string, got Int", location: null }
```

**Analysis**:
- `println(get(tags, 0))` works fine and prints "learning"
- `length(tags)` correctly returns 2
- But `join(tags, ", ")` fails with a type error claiming it got an Int
- The existing test file `join_test.bg` only uses `split()` to create arrays, not array literals
- This suggests array literals with strings have a different internal representation than arrays created by `split()`

**Friction**:
1. Array literal syntax `["a", "b"]` doesn't work with `join()`
2. Error message is misleading ("got Int" when values are clearly strings)
3. No way to easily create an array of strings that works with `join()`

**Workaround**: Must implement manual string concatenation loop

---

### Issue #3: String concatenation fails mysteriously in while loops with "Expected string, got Null"

**Location**: While loop inside a function called from nested structs/arrays context
**Code**:
```beagle
fn manual_join(arr, sep) {
    let len = length(arr)
    if len == 0 {
        ""
    } else {
        let mut result = get(arr, 0)
        let mut i = 1
        while i < len {
            let elem = get(arr, i)
            result = result ++ sep ++ elem  // FAILS HERE
            i = i + 1
        }
        result
    }
}
```

**Error**:
```
SystemError.TypeError { message: "Expected string, got Null", location: null }
```

**Debugging revealed**:
- `println(result)` works and shows correct string
- `println(sep)` works and shows correct string
- `println(elem)` works and shows correct string
- `type-of()` for all three returns "String"
- Even re-getting from array works
- But `result ++ sep ++ elem` fails
- Even `temp = result ++ sep; temp ++ elem` fails (first works, second fails)

**Analysis**: This appears to be a GC or compiler bug where:
1. Variables appear valid when printed
2. But become null when used in concatenation
3. The issue only manifests in complex call stacks (nested function calls with structs)
4. Simple test cases work fine

**Friction**:
1. Extremely difficult to debug - all visible state looks correct
2. Error occurs between println and actual operation
3. Workarounds unclear - might need to restructure entire program

**Severity**: BLOCKING - Cannot implement string operations in while loops reliably

**Workaround Found**: Use recursion instead of while loops!
```beagle
fn join_recursive(arr, sep, i, accum) {
    let len = length(arr)
    if i >= len {
        accum
    } else {
        let elem = get(arr, i)
        let new_accum = accum ++ sep ++ elem
        join_recursive(arr, sep, i + 1, new_accum)
    }
}
```
This recursive version works perfectly where the while loop fails.

---

### Issue #4: `throw()` does not support expressions, only simple values

**Location**: Any `throw` with concatenation or function call
**Code**:
```beagle
throw("outer from " ++ to-string(e))  // COMPILE ERROR
```

**Error**:
```
Compile error: Missing ')' after throw value at position 1817
```

**Analysis**: The `throw` statement parser only accepts simple values (strings, variables), not expressions.

**Friction**:
1. Error message is confusing - says "Missing ')'" when the real issue is unsupported expression
2. Must assign to a variable first, then throw the variable
3. Inconsistent with other language constructs that accept expressions

**Workaround**:
```beagle
let msg = "outer from " ++ to-string(e)
throw(msg)
```

---

### Issue #5: Structs cannot be defined inside functions

**Location**: Struct definition inside a function body
**Code**:
```beagle
fn test_map_struct() {
    struct Person {  // COMPILE ERROR
        name
        metadata
    }
    // ...
}
```

**Error**:
```
Compile error: Cannot resolve struct: Person
```

**Analysis**: Struct definitions must be at the top level of a namespace, not inside functions.

**Friction**:
1. Error message doesn't explain that structs must be top-level
2. Would be nice to have local struct definitions for test cases
3. No scoping for struct definitions

**Workaround**: Move struct definitions to top level

---

### Issue #6: Cannot mix integer and float in arithmetic operations

**Location**: Any arithmetic expression mixing int and float
**Code**:
```beagle
3.14 * 2    // ERROR - must be 3.14 * 2.0
```

**Error**:
```
SystemError.TypeError { message: "Type mismatch in arithmetic operation", location: null }
```

**Analysis**: Beagle requires explicit type matching for arithmetic - no automatic int-to-float coercion.

**Friction**:
1. Easy mistake to make, especially for developers from Python/JS background
2. Error message doesn't suggest the fix
3. Must remember to use `.0` suffix on all numeric literals when working with floats

**Workaround**: Always use explicit float literals (2.0, not 2) when mixing with floats

---

### Issue #7: `abs()` crashes with misaligned pointer on negative integers

**Location**: Calling `abs(-42)` or any negative integer
**Code**:
```beagle
abs(-42)
```

**Error**:
```
thread 'main' panicked at src/builtins.rs:2413:27:
misaligned pointer dereference: address must be a multiple of 0x8 but is 0x1fffffffffffffde
```

**Analysis**: This is a critical memory safety bug in the `abs` builtin function when given negative integers. The function likely has incorrect pointer arithmetic or type handling.

**Severity**: CRITICAL - causes hard crash, potential security issue

**Workaround**: Implement manual abs:
```beagle
fn safe_abs(n) {
    if n < 0 { 0 - n } else { n }
}
```

---

## Summary

### Bugs Found (7 total)

| # | Issue | Severity | Workaround |
|---|-------|----------|------------|
| 1 | String interpolation doesn't auto-convert types | Medium | Use `to-string()` and `++` |
| 2 | `join()` fails on array literals | High | Use recursion to build strings |
| 3 | String concat in while loops corrupts memory | Critical | Use recursion instead of while |
| 4 | `throw()` only accepts simple values | Low | Assign to variable first |
| 5 | Structs must be top-level | Low | Move structs to top level |
| 6 | No int-to-float coercion | Medium | Use explicit `.0` suffixes |
| 7 | `abs()` crashes on negative ints | Critical | Implement manual abs function |

### What Works Well

1. **Pattern matching** - Enums and match expressions work great
2. **Higher-order functions** - map, filter, reduce all work well
3. **Closures** - Capturing mutable variables works correctly
4. **Tail Call Optimization** - Recursion with 10,000+ calls works
5. **Structs** - Deeply nested structs work fine
6. **For loops** - Iteration over ranges, arrays, and strings works
7. **Try/catch** - Exception handling works as expected
8. **Maps with keywords** - `{:key value}` syntax works well

### Recommendations for Language Improvement

1. **Better error messages** - Include line numbers and suggest fixes
2. **Fix GC/memory issues** - Issues #3 and #7 indicate memory safety problems
3. **Allow expressions in throw** - Parser should accept any expression
4. **Add type coercion** - Auto-convert int to float in mixed arithmetic
5. **Fix `join()` for array literals** - Should work the same as split-created arrays
6. **Support local struct definitions** - Or at least give a clearer error

### Test Coverage Created

- `todo_app.bg` - Full todo list manager with CRUD operations
- `todo_app_v2.bg` - Extended version with grouping, batch ops, complex filters
- `edge_cases.bg` - Comprehensive edge case test suite
- `debug_test.bg` - Debugging helper file

