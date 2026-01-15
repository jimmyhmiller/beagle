# Bug: Calling Top-Level Functions Through Variables

## Summary

Calling a top-level named function through a variable causes a segfault. The issue is isolated to the dynamic function call mechanism in `compile_closure_call`, not related to closures or capturing.

## What Works ✅

```beagle
// Direct call to top-level function
fn add1(x) { x + 1 }
let result = add1(5)  // Works: 6

// Anonymous function through variable
let f = fn(x) { x + 1 }
let result = f(5)  // Works: 6

// Nested named function
fn main() {
    fn helper(x) { x + 1 }
    let result = helper(5)  // Works: 6
}
```

## What Fails ❌

```beagle
// Top-level function through variable
fn add1(x) { x + 1 }
let f = add1
let result = f(5)  // SEGFAULT
```

## The Problem

### Code Flow

1. **Function Definition** (ast.rs:858-879)
   - Top-level function `add1` is compiled
   - Function pointer is tagged with `BuiltInTypes::Function.tag(ptr)`
   - Stored in namespace binding via `update-binding` builtin

2. **Variable Assignment** (`let f = add1`)
   - Identifier `add1` resolved as `VariableLocation::NamespaceVariable`
   - `resolve_variable` calls `get-binding` builtin (ast.rs:3298-3304)
   - Returns the tagged function pointer from namespace storage
   - Stored in local variable `f`

3. **Function Call** (`f(5)`)
   - Identifier `f` is non-qualified, so uses `compile_closure_call` (ast.rs:2091-2092)
   - Runtime check determines it's a bare Function, not Closure (ast.rs:2767-2768)
   - Takes non-closure path (ast.rs:2770-2774)
   - Calls: `ir.call(function_register.into(), args)`
   - **SEGFAULT**

### Current Code (ast.rs:2770-2774)

```rust
// Non-closure function call path
// Top-level functions don't expect arg0 to be the closure
// So just call with the original args (no closure prepended)
let result = self.ir.call(function_register.into(), args.clone());
self.ir.assign(ret_register, result);
self.ir.jump(exit_closure_call);
```

### IR Call Implementation (ir.rs:1771-1779)

```rust
// TODO: I am not actually checking any tags here
// or unmasking or anything. Just straight up calling it
let function = self.value_to_register(function, backend);
backend.shift_right_imm(function, function, BuiltInTypes::tag_size());
if *builtin {
    backend.call_builtin(function);
} else {
    backend.call(function);
}
```

## Root Cause Analysis

The segfault indicates the function pointer is either:
1. **Corrupted** - The tagged pointer is malformed or incorrectly untagged
2. **Wrong** - We're jumping to an invalid address
3. **Calling Convention Mismatch** - The function expects different arguments than we're passing

### Investigation Results

- ✅ The function pointer is correctly tagged when stored
- ✅ `get-binding` returns a valid tagged pointer
- ✅ The tag check correctly identifies it as a Function (not Closure)
- ✅ `ir.call()` correctly untags the pointer (shift right by 3 bits)
- ❌ Something between untagging and calling causes the crash

### Why Anonymous Functions Work

Anonymous functions (`fn(x) { x }`) are compiled as closures even with zero captured variables:
- Wrapped in closure object at definition time (ast.rs:840-855)
- Stored as Closure-tagged heap objects
- Take the closure calling path (ast.rs:2775-2797)
- Closure arg0 is passed, function pointer extracted from closure structure
- **This path works correctly**

### Why Top-Level Functions Fail

Top-level functions are stored as bare tagged function pointers:
- No closure wrapper
- Stored directly in namespace bindings
- Take the non-closure calling path (ast.rs:2770-2774)
- **This path segfaults**

## Potential Fixes

### Option 1: Wrap All Top-Level Functions in Closures

**Change:** ast.rs:858

```rust
// Instead of:
let function = self.ir.function(Value::Function(function_pointer));

// Do:
let function = self.compile_closure(
    BuiltInTypes::Function.tag(function_pointer as isize) as usize,
);
```

**Pros:**
- Unified calling convention - everything is a closure
- Leverages existing working closure calling path
- Minimal code changes

**Cons:**
- Performance overhead - extra heap allocation and indirection for every function
- Memory overhead - closure object for every top-level function
- Breaks assumptions about function representation

**Blocker:**
- This was tried but failed with "Function not found when creating closure"
- The function hasn't been registered with the runtime yet when `compile_closure` is called

### Option 2: Fix the Non-Closure Calling Path

**Investigate:**
1. **Verify function pointer integrity**
   - Add logging to trace the pointer value from storage through call
   - Check if `get-binding` preserves the tag correctly
   - Verify the untagged address points to valid code

2. **Check calling convention**
   - Top-level functions: `fn(arg1, arg2, ...)`
   - Closures: `fn(closure_ptr, arg1, arg2, ...)`
   - Ensure we're not accidentally passing closure_ptr to non-closures

3. **Debug the IR Call instruction**
   - The TODO comment suggests tag handling is incomplete (ir.rs:1771)
   - Might need special handling for dynamically loaded function pointers
   - Current implementation assumes statically known functions

**Specific Investigation Points:**

```rust
// In compile_closure_call, before calling:
// 1. Log the function_register value (should be tagged with 0b100)
self.ir.breakpoint();  // Add before line 2772
let logged_value = function_register.into();
// Log: logged_value should be (function_ptr << 3) | 0b100

// 2. In ir.rs Call instruction compilation:
// Verify the untagged address is valid executable code
// Check: Is (tagged_value >> 3) a valid function entry point?

// 3. Compare with direct call compilation (ast.rs:2445-2517):
// How does `self.call("namespace/name", args)` differ?
// It loads from jump table, not from namespace bindings
```

### Option 3: Use Jump Table for All Functions

**Change:** Make `get-binding` return jump table indices instead of function pointers

**Pros:**
- Consistent with how direct calls work
- Jump table already handles function pointer indirection correctly

**Cons:**
- Major refactoring of namespace binding system
- Functions would need dual storage (namespace bindings + jump table)
- Unclear how to handle closures (which can't be in jump table)

### Option 4: Special Builtin for Function Calls

**Add:** `beagle.builtin/call-function` that handles the dynamic dispatch

```rust
// Instead of ir.call() in compile_closure_call:
self.call_builtin(
    "beagle.builtin/call-function",
    vec![function_register.into(), args_array],
)
```

**Pros:**
- Centralized logic for runtime function dispatch
- Can handle all edge cases (variadic, arity checking, etc.)
- Easier to debug

**Cons:**
- Performance overhead of builtin call
- Need to pack/unpack arguments into array

## Recommended Next Steps

1. **Add extensive logging** to trace the function pointer value:
   - When stored by `update-binding`
   - When retrieved by `get-binding`
   - Before and after untagging in `ir.call()`
   - At the actual call site

2. **Compare assembly** generated for:
   - Direct call: `add1(5)`
   - Variable call: `let f = add1; f(5)`
   - Anonymous function call: `let f = fn(x) { x }; f(5)`

3. **Test with minimal IR**:
   - Manually construct the simplest possible dynamic function call
   - Verify each step: tag, store, load, untag, call

4. **Examine runtime function registration**:
   - Check if `get_function_by_pointer` can find the function
   - Verify the function metadata is correct
   - Ensure the pointer in the runtime matches the pointer in the namespace

## Related Code Locations

- **compile_closure_call**: src/ast.rs:2752-2802
- **IR Call instruction**: src/ir.rs:1755-1795, 2323-2328
- **resolve_variable for NamespaceVariable**: src/ast.rs:3298-3304
- **get-binding builtin**: src/builtins.rs:2366-2384
- **Top-level function compilation**: src/ast.rs:858-879

## Test Cases

```beagle
// Minimal failing case
namespace test
fn add1(x) { x + 1 }
fn main() {
    let f = add1
    let result = f(5)  // SEGFAULT
    println(to-string(result))
}

// Should work after fix
namespace test
fn identity(x) { x }
fn main() {
    let f = identity
    println(to-string(f(42)))  // Should print: 42
}
```
