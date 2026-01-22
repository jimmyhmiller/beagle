# Algebraic Effects Handler Bugs

This document describes bugs discovered during comprehensive testing of the algebraic effects system.

## Bug 1: Sequential Handler Blocks Cause Infinite Loops

**Test file:** `handler_sequential_blocks_test.bg`

**Description:** When there are multiple `handle` blocks at the same level in `main()`, the second block loops infinitely instead of executing once.

**Reproduction:**
```beagle
let result_a = handle effect/Handler(EffectA) with handler_a {
    perform EffectA.DoA {}
}
// After first handle completes...

let result_b = handle effect/Handler(EffectB) with handler_b {
    perform EffectB.DoB {}  // This loops forever
}
```

**Expected:** Both handle blocks execute once and complete.

**Actual:** The second handle block repeats indefinitely.

---

## Bug 2: Effects After Nested Handler Don't Execute

**Test file:** `handler_post_nested_test.bg`

**Description:** When performing an effect after a nested handler block completes, the effect is not executed. The outer handler block appears to return early.

**Reproduction:**
```beagle
handle effect/Handler(Outer) with outer_h {
    perform Outer.Log { message: "Before" }  // Works

    let inner = handle effect/Handler(Inner) with inner_h {
        perform Inner.Compute { value: 21 }
    }

    perform Outer.Log { message: "After" }  // NEVER EXECUTES
    inner
}
```

**Expected:** "After" log message should appear.

**Actual:** The outer handler block returns immediately after the nested handler completes. The program exits with code 0 but skips:
- `perform Outer.Log { message: "After nested handler" }`
- `println("Inner result was:", inner_result)`

---

## Bug 3: Effects Through Deep Call Stacks Crash

**Test file:** `handler_deep_call_test.bg`

**Description:** When effects are performed through multiple levels of nested function calls (3+ levels), the runtime crashes with a segfault.

**Reproduction:**
```beagle
fn level1() {
    trace(1, "In level1")
    let r = level2()
    trace(1, "Back in level1")  // Crash after this
    r
}

fn level2() {
    trace(2, "In level2")
    let r = level3()
    trace(2, "Back in level2")
    r
}
// ... more levels ...
```

**Expected:** Effects work correctly through any call depth.

**Actual:** Segfault occurs when returning through multiple call levels with effects.

---

## Bug 4: Effects in Loops Through Nested Functions Crash

**Test files:** `handler_generator_test.bg`, `handler_writer_test.bg`

**Description:** When effects are performed in a loop inside a nested function, the runtime crashes.

**Reproduction:**
```beagle
fn range-gen(start, end) {
    let mut i = start
    while i < end {
        yield-val(i)  // Effect through nested call
        i = i + 1
    }
}

handle effect/Handler(Gen) with gen_h {
    range-gen(10, 15)  // Crashes
}
```

**Expected:** Effects work correctly in loops.

**Actual:** Runtime crash (core dump).

---

## Bug 5: Field Assignment in Handlers Causes Compiler Panic

**Test file:** `handler_stateful_test.bg`

**Description:** Using `self.field = value` inside a handler's `handle` method causes the compiler to panic with "Expected string".

**Reproduction:**
```beagle
extend CounterHandler with effect/Handler(Counter) {
    fn handle(self, op, resume) {
        match op {
            Counter.Inc {} => {
                self.count = self.count + 1  // COMPILER PANIC
                resume(self.count)
            }
        }
    }
}
```

**Expected:** Field assignment should work normally.

**Actual:** Compiler panics at `src/ast.rs:622` with "Expected string".

**Workaround:** The existing `sequential_test.bg` uses `let new_count = self.count + 1` instead of mutating the field.

---

## Summary of Test Results

| Test File | Status | Bug |
|-----------|--------|-----|
| `handler_test.bg` | ✅ PASS | - |
| `handler_test2.bg` | ✅ PASS | - |
| `handler_inline_test.bg` | ✅ PASS | - |
| `handler_multishot_test.bg` | ✅ PASS | - |
| `handler_nested_test.bg` | ✅ PASS | - |
| `handler_multi_effect_type_test.bg` | ✅ PASS | - |
| `handler_cross_file_test.bg` | ✅ PASS | - |
| `handler_reader_test.bg` | ✅ PASS | - |
| `handler_exception_test.bg` | ✅ PASS | - |
| `handler_choice_multishot_test.bg` | ✅ PASS | - |
| `handler_amb_test.bg` | ✅ PASS | - |
| `handler_writer_test.bg` | ❌ CRASH | Bug 4 |
| `handler_deep_call_test.bg` | ❌ CRASH | Bug 3 |
| `handler_generator_test.bg` | ❌ CRASH | Bug 4 |
| `handler_sequential_blocks_test.bg` | ❌ INFINITE LOOP | Bug 1 |
| `handler_post_nested_test.bg` | ❌ WRONG OUTPUT (exits early) | Bug 2 |
| `handler_stateful_test.bg` | ❌ COMPILER PANIC | Bug 5 |

## Root Cause Hypothesis

These bugs appear to be related to **continuation management**:

1. **Stale continuation pointers:** The commit `fef16ef` mentions "clearing stale continuation pointers on reset" - similar issues may exist elsewhere.

2. **Stack frame corruption:** Crashes during return from nested calls suggests the continuation's captured stack state becomes invalid.

3. **Handler scope tracking:** The infinite loop and early-return bugs suggest the handler installation/uninstallation isn't correctly tracking scope boundaries.

## Recommendations

1. Add assertions to verify continuation validity before invocation
2. Review stack pointer management during effect capture/resume
3. Add tests for handler scope boundary conditions
4. Consider adding debug tracing for continuation lifecycle
