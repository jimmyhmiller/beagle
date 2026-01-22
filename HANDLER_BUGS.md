# Algebraic Effect Handler Bugs Analysis

## Original CI Failure

The CI was failing on `handler_sequential_blocks_test.bg` which caused an infinite loop on macOS ARM64.

## Root Cause Investigation

### The Sequential Handler Bug

**File:** `resources/handler_sequential_blocks_test.bg`

**Symptom:** Two sequential handle blocks - the second one loops infinitely, printing output from the first handler.

**Root Cause:** In `return_from_shift_runtime` (src/builtins.rs:4656), the fallback path uses `conts.first()` to find the continuation to return to. For sequential handlers:
1. First handler captures continuation at index 0
2. First handler completes, but continuation at index 0 is never removed
3. Second handler captures continuation at index 1
4. Second handler completes, `conts.first()` returns index 0 (first handler's stale continuation)
5. Jumps to first handler's `handler_address` → infinite loop

**Attempted Fix:** Change `conts.first()` to `conts.last()` so each handler uses its own (most recent) continuation.

**Result:** Fixes sequential handlers but exposes/causes other issues.

---

## Related Bugs Discovered

### 1. Multi-shot Continuation Behavior Mismatch

**File:** `resources/handler_multishot_test.bg`

**Expected output (in test file):**
```
main started
chose: 1
final result: 10
main done
```

**Actual output (both with AND without my fix):**
```
main started
chose: 1
first branch returned: 10
chose: 2
second branch returned: 20
final result: 30
main done
```

**Analysis:** The test expects single-shot behavior but the runtime already implements multi-shot. The expected output is outdated/wrong. Both `resume(1)` and `resume(2)` execute successfully.

---

### 2. Closure Protocol Test Mismatch

**File:** `resources/closure_protocol_test.bg`

**Expected output:**
```
in shift, about to call k
11
```

**Actual output (both with AND without my fix):**
```
in shift, about to call k
k returned
11
```

**Analysis:** The shift body continues after `k()` returns, printing "k returned". The expected output is outdated.

---

### 3. Cross-File Handler Crash

**File:** `resources/handler_cross_file_test.bg`

**Symptom:** Test output is correct, but process crashes (SIGSEGV) during cleanup/exit.

**Status:** This crash occurs **with AND without my fix**. It's a pre-existing bug.

**Analysis:** The test has two `perform` calls in the same handler (via `get-state()` function called twice). Something in the cleanup path corrupts memory.

---

### 4. Nested Handler Bugs (Pre-existing)

**Files:**
- `resources/handler_nested_test.bg`
- `resources/handler_post_nested_test.bg`
- `resources/handler_stateful_test.bg`

**Symptom:** All fail with "ContinuationError" - continuation lookup fails.

**Status:** These were **already failing before my fix**. They document known bugs with:
- Tunneled effects (outer effect performed from inside inner handler's body)
- Effects after nested handlers complete
- Multiple performs in the same handler

---

## Summary of Current State

| Test | Without Fix | With Fix (`last()` instead of `first()`) |
|------|-------------|------------------------------------------|
| `handler_sequential_blocks_test.bg` | **INFINITE LOOP** | **PASSES** |
| `handler_multishot_test.bg` | Output mismatch (test expects single-shot, runtime does multi-shot) | Same |
| `closure_protocol_test.bg` | Output mismatch (missing "k returned") | Same |
| `handler_cross_file_test.bg` | Crash during cleanup | Crash during cleanup |
| `handler_nested_test.bg` | ContinuationError | ContinuationError |
| `handler_post_nested_test.bg` | ContinuationError | ContinuationError |
| `handler_stateful_test.bg` | ContinuationError | ContinuationError |

## The Core Problem

The continuation management in `return_from_shift_runtime` uses a simple list (`captured_continuations`) but the semantics are complex:

1. **Sequential handlers:** Each handler should use its own continuation (need `last()`)
2. **Nested/tunneled handlers:** Inner handler's continuation captured with outer's prompt info (complex)
3. **Multi-shot:** Same continuation invoked multiple times (need to keep it in list)
4. **Cleanup:** Continuations must be removed at the right time to avoid stale data

The `conts.first()` approach worked for some cases but fails for sequential handlers.
The `conts.last()` approach fixes sequential handlers but the test expectations for multi-shot and closure_protocol are outdated.

## Recommended Actions

1. **Fix the sequential handler bug** by using `conts.last()` instead of `conts.first()`
2. **Update test expectations** for `handler_multishot_test.bg` and `closure_protocol_test.bg` to match actual (correct) behavior
3. **Investigate the cross-file crash** separately - it's a pre-existing memory corruption bug
4. **Keep nested handler tests disabled** until those bugs are fixed (separate issue)
