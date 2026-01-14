# Missing Features for a Functional Dynamic Language

This document outlines features that Beagle is missing compared to mature functional dynamic languages like Clojure, Elixir, or Racket.

## Critical Runtime Limitations

### Closures Capturing Closures

**Status:** Broken (see `standard-library/std.bg:1074-1076`)

This is the most significant limitation. Higher-order functions that return closures capturing other functions don't work. This blocks idiomatic functional patterns:

```beagle
// These CANNOT be implemented currently:
fn compose(f, g) { fn(x) { f(g(x)) } }      // Function composition
fn partial(f, ...args) { fn(...more) { ... } }  // Partial application
fn complement(pred) { fn(x) { not(pred(x)) } }  // Predicate negation
fn memoize(f) { ... }                        // Caching wrapper
```

**Impact:** High - this is a core functional programming pattern.

**Workaround:** Users must inline these patterns rather than abstracting them.

## Missing Language Features

### Destructuring

No destructuring in let bindings or function parameters.

```beagle
// Not supported:
let [a, b, c] = some_array
let {x, y} = some_map
fn process([head, ...tail]) { ... }
```

**Impact:** High - destructuring is essential for clean pattern-based code.

### String Interpolation

No template string syntax.

```beagle
// Not supported:
let msg = "Hello ${name}, you have ${count} messages"

// Must use:
let msg = "Hello " ++ name ++ ", you have " ++ to-string(count) ++ " messages"
```

**Impact:** Medium - verbose but workable.

### Regular Expressions

No native regex support.

```beagle
// Not supported:
let pattern = /\d{3}-\d{4}/
if matches?(phone, pattern) { ... }
```

**Impact:** Medium - many string processing tasks require regex.

### Macros

No macro system for metaprogramming.

```beagle
// Not supported:
macro unless(condition, body) {
    if not(condition) { body }
}
```

**Impact:** High - macros enable DSLs and syntax extension. This is a defining feature of Lisps.

### Lazy Sequences

All sequences are eager. No infinite sequences or deferred computation.

```beagle
// Not supported:
let naturals = iterate(fn(x) { x + 1 }, 0)  // Would be infinite
let first-ten = take(naturals, 10)
```

**Impact:** High - lazy evaluation enables memory-efficient stream processing.

### Multi-Arity Functions

Cannot define multiple implementations based on argument count.

```beagle
// Not supported:
fn greet() { "Hello" }
fn greet(name) { "Hello " ++ name }
fn greet(name, greeting) { greeting ++ " " ++ name }
```

**Impact:** Medium - common pattern in Clojure.

### Default Parameter Values

Cannot specify defaults for optional parameters.

```beagle
// Not supported:
fn connect(host, port = 8080, timeout = 5000) { ... }
```

**Impact:** Medium - reduces need for multiple function variants.

### Named/Keyword Arguments

Cannot call functions with named parameters.

```beagle
// Not supported:
create-user(name: "Alice", admin: true, age: 30)
```

**Impact:** Low-Medium - improves readability for functions with many parameters.

### Spread Operator in Calls

Can collect rest arguments but cannot spread arrays into function calls.

```beagle
// Collecting works:
fn sum(...nums) { reduce(nums, 0, fn(a, x) { a + x }) }

// Spreading does NOT work:
let args = [1, 2, 3]
sum(...args)  // Not supported
```

**Impact:** Medium - useful for dynamic function application.

### Cond Expression

No multi-branch conditional without nested if/else.

```beagle
// Not supported:
cond {
    x < 0  => "negative"
    x == 0 => "zero"
    x > 0  => "positive"
}

// Must use nested if/else:
if x < 0 {
    "negative"
} else {
    if x == 0 {
        "zero"
    } else {
        "positive"
    }
}
```

**Impact:** Low - match expressions cover some use cases.

## Missing Standard Library

### Data Structures

| Structure | Status |
|-----------|--------|
| PersistentVector | Implemented |
| PersistentMap | Implemented |
| PersistentSet | **Missing** |
| Lazy Sequence | **Missing** |
| Queue | **Missing** |
| Sorted Map/Set | **Missing** |

### String Functions

Currently missing (per STDLIB_PLAN.md):

```
split(str, delimiter)      // Split into vector
join(coll, separator)      // Join with separator
trim(str)                  // Remove whitespace
trim-left(str)
trim-right(str)
starts-with?(str, prefix)
ends-with?(str, suffix)
contains?(str, substr)
index-of(str, substr)
last-index-of(str, substr)
replace(str, from, to)
replace-first(str, from, to)
pad-left(str, width, char)
pad-right(str, width, char)
lowercase(str)             // uppercase exists
blank?(str)
lines(str)
words(str)
```

### Math Functions

Currently missing:

```
tan(x), asin(x), acos(x), atan(x), atan2(y, x)
exp(x), log(x), log10(x), log2(x), pow(base, exp)
round(x), truncate(x)
max(a, b), min(a, b), clamp(x, low, high)
gcd(a, b), lcm(a, b)
even?(n), odd?(n), positive?(n), negative?(n), zero?(n)
random(), random-int(max), random-range(min, max)

// Constants
PI, E, TAU
```

### Map Utilities

Currently missing:

```
vals(m)                    // Get all values
dissoc(m, key)             // Remove key
merge(m1, m2)              // Combine maps
merge-with(f, m1, m2)      // Combine with conflict resolver
select-keys(m, ks)         // Keep only specified keys
update(m, key, f)          // Update value with function
get-in(m, path)            // Nested access: get-in(m, [:a, :b, :c])
assoc-in(m, path, val)     // Nested set
update-in(m, path, f)      // Nested update
contains-key?(m, key)
invert(m)                  // Swap keys and values
map-keys(m, f)
map-vals(m, f)
filter-keys(m, pred)
filter-vals(m, pred)
```

### Option/Result Types

These exist in `beagle.io` but should be in core:

```beagle
enum Option {
    Some { value }
    None
}

enum Result {
    Ok { value }
    Err { error }
}

// With helper functions:
some(value), none()
ok(value), err(error)
unwrap(opt), unwrap-or(opt, default)
map-option(opt, f), flat-map-option(opt, f)
map-result(result, f), map-err(result, f)
and-then(result, f), or-else(result, f)
```

### Collection Utilities

Missing:

```
min-of(coll), max-of(coll)
min-by(coll, f), max-by(coll, f)
reduce-right(coll, init, f)
```

## What Beagle DOES Have

For reference, these features are already well-implemented:

- Pipe operators (`|>` and `|>>`)
- Protocols with inline caching
- Comprehensive collection functions (map, filter, reduce, zip, group-by, frequencies, sort, etc.)
- Persistent vectors and maps (Rust-backed HAMT)
- Pattern matching on enums with guards
- Atoms for thread-safe mutable state
- Variadic functions (`...args`)
- Tail call optimization
- try/catch/throw exception handling
- for/while/loop control flow
- FFI for C interop
- Multiple GC backends (mark-sweep, compacting, generational)
- Namespace and import system
- Mutable struct fields
- eval() for runtime code execution

## Prioritized Recommendations

### High Priority

1. **Fix closure-capturing-closure** - Unblocks idiomatic FP patterns
2. **Add destructuring** - Essential for clean code
3. **Add lazy sequences** - Enables efficient stream processing
4. **Add PersistentSet** - Common data structure

### Medium Priority

5. **Add string functions** - split, join, trim, etc.
6. **Add math functions** - Especially random numbers
7. **Add map utilities** - get-in, assoc-in, merge, etc.
8. **Add string interpolation** - Developer experience
9. **Move Option/Result to core** - Standardize error handling

### Lower Priority

10. **Add macros** - Powerful but complex to implement
11. **Add regex** - Can use FFI as workaround
12. **Add multi-arity functions** - Nice to have
13. **Add default parameters** - Nice to have
14. **Add cond expression** - Syntactic sugar
