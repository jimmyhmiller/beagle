# Friction Log Fixes

This document summarizes the fixes made in response to the friction log from building a Contact Book Manager in Beagle.

## Summary of Changes

### 1. Improved Error Message for Struct Field Commas

**Friction Point:** When users accidentally use commas between struct fields, the error message was cryptic: "Expected field name but found NewLine".

**Fix:** Updated `src/parser.rs` to detect when a newline appears where a field name is expected (which typically happens after a comma was consumed). The error now provides a helpful message:

```
Expected field name but found newline. Note: struct fields should be separated by newlines, not commas. Use:
struct Foo {
    field1
    field2
}
```

**File:** `src/parser.rs:2182-2186`

### 2. Added `enumerate` Function

**Friction Point:** For loops don't provide an index variable, requiring users to maintain a mutable counter.

**Fix:** Added `enumerate(coll)` function to the standard library that returns pairs of `[index, element]` for each element in the collection.

**Usage:**
```beagle
for [idx, contact] in enumerate(contacts) {
    println("[" ++ to-string(idx + 1) ++ "]", contact.name)
}
```

**File:** `standard-library/std.bg:910-920`

### 3. Added `remove-at` Function

**Friction Point:** Users had to implement array removal manually.

**Fix:** Added `remove-at(coll, idx)` function to the standard library that removes the element at the specified index and returns a new collection.

**Usage:**
```beagle
let new-list = remove-at([1, 2, 3, 4, 5], 2)  // Returns [1, 2, 4, 5]
```

**File:** `standard-library/std.bg:922-930`

### 4. Added `parse-int` Function

**Friction Point:** Users had to implement integer parsing from scratch.

**Fix:** Added `parse-int(s)` function to the standard library that parses a string into an integer. Returns `null` if the string is not a valid integer. Supports negative numbers.

**Usage:**
```beagle
let num = parse-int("42")        // 42
let neg = parse-int("-123")      // -123
let bad = parse-int("abc")       // null
```

**File:** `standard-library/std.bg:515-555`

## Items Not Addressed (Future Work)

The following friction points from the log were noted but not addressed in this change:

1. **Function Discovery** - Hard to know what functions are available and where they come from. This would require documentation or a REPL with introspection capabilities.

2. **Struct Field Shorthand** - Rust-style shorthand `{ name, phone }` when variable names match field names. This would require parser changes.

3. **Global Availability of `char-code`** - Currently requires importing `beagle.builtin`. This is a design decision about what should be globally available.

4. **Match on Imported Enum** - Requires full path for enum variants from imported modules. This is consistent with the module system design.

## Test File

A test file was added to verify the new functions work correctly:

**File:** `resources/friction_log_fixes_test.bg`

This test covers:
- `enumerate` with a string array
- `remove-at` at various positions (first, middle, last)
- `parse-int` with positive numbers, negative numbers, zero, and invalid input
