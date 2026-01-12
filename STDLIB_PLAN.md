# Beagle Standard Library Enhancement Plan

## Current State

The Beagle stdlib has:
- Basic protocols: `Indexed`, `Push`, `Length`, `Format`, `Associable`, `Keys`, `Seqable`, `Seq`
- Persistent collections: `PersistentVector`, `PersistentMap`
- Atoms for mutable state
- Range iteration
- Basic string operations
- File I/O
- FFI bindings

## What's Missing

### 1. Higher-Order Collection Functions (PRIORITY: HIGH)

These are the most glaring omissions. Every functional language needs these.

**File: `standard-library/std.bg` (add to core)**

```
// Transformations
fn map(coll, f)           // Apply f to each element, return new collection
fn filter(coll, pred)     // Keep elements where pred returns true
fn reduce(coll, init, f)  // Fold collection with accumulator
fn reduce-right(coll, init, f)  // Fold from right

// Predicates
fn any?(coll, pred)       // True if any element matches
fn all?(coll, pred)       // True if all elements match
fn none?(coll, pred)      // True if no elements match
fn find(coll, pred)       // First element matching predicate
fn find-index(coll, pred) // Index of first match

// Slicing
fn take(coll, n)          // First n elements
fn drop(coll, n)          // Skip first n elements
fn take-while(coll, pred) // Take while predicate true
fn drop-while(coll, pred) // Drop while predicate true
fn slice(coll, start, end) // Sub-collection

// Grouping
fn partition(coll, n)     // Split into groups of n
fn group-by(coll, f)      // Group by key function result
fn frequencies(coll)      // Count occurrences of each value

// Combining
fn concat(coll1, coll2)   // Combine two collections
fn flatten(coll)          // Flatten nested collections one level
fn flat-map(coll, f)      // Map then flatten
fn zip(coll1, coll2)      // Pair up elements
fn zip-with(coll1, coll2, f) // Combine elements with function

// Ordering
fn reverse(coll)          // Reverse order
fn sort(coll)             // Sort naturally
fn sort-by(coll, f)       // Sort by key function

// Element access
fn first(coll)            // First element (already in Seq)
fn last(coll)             // Last element
fn rest(coll)             // All but first
fn butlast(coll)          // All but last
fn nth(coll, n)           // Element at index n with bounds checking

// Collection creation
fn repeat(value, n)       // Repeat value n times
fn repeatedly(f, n)       // Call f n times, collect results
fn iterate(f, init, n)    // Repeatedly apply f, collect n values

// Misc
fn count(coll)            // Alias for length
fn empty?(coll)           // True if length == 0
fn distinct(coll)         // Remove duplicates
fn interleave(coll1, coll2) // Alternate elements
fn interpose(coll, sep)   // Insert separator between elements
```

### 2. String Functions (PRIORITY: HIGH)

**Add to `standard-library/std.bg` or new `standard-library/beagle.string.bg`**

```
// Predicates
fn starts-with?(str, prefix)
fn ends-with?(str, suffix)
fn contains?(str, substr)
fn blank?(str)            // null, empty, or only whitespace
fn empty-string?(str)     // exactly ""

// Transformation
fn trim(str)              // Remove leading/trailing whitespace
fn trim-left(str)
fn trim-right(str)
fn lowercase(str)         // uppercase already exists
fn replace(str, from, to) // Replace all occurrences
fn replace-first(str, from, to)
fn reverse-string(str)

// Splitting/Joining
fn split(str, delimiter)  // Split into vector
fn join(coll, separator)  // Join with separator
fn lines(str)             // Split by newlines
fn words(str)             // Split by whitespace

// Searching
fn index-of(str, substr)  // First index of substr, or -1
fn last-index-of(str, substr)

// Padding
fn pad-left(str, width, char)
fn pad-right(str, width, char)

// Parsing
fn parse-int(str)         // String to int, returns Result
fn parse-float(str)       // String to float, returns Result
```

### 3. Math Functions (PRIORITY: MEDIUM)

**Add to `standard-library/std.bg` or new `standard-library/beagle.math.bg`**

Already have: `sqrt`, `floor`, `ceil`, `abs`, `sin`, `cos`, `to-float`

```
// Trigonometry
fn tan(x)
fn asin(x)
fn acos(x)
fn atan(x)
fn atan2(y, x)

// Exponential/Logarithmic
fn exp(x)
fn log(x)                 // Natural log
fn log10(x)
fn log2(x)
fn pow(base, exp)

// Rounding
fn round(x)
fn truncate(x)

// Comparison
fn max(a, b)
fn min(a, b)
fn clamp(x, low, high)

// Integer operations
fn mod(a, b)              // Modulo (handles negatives correctly)
fn quot(a, b)             // Integer division
fn rem(a, b)              // Remainder
fn gcd(a, b)
fn lcm(a, b)

// Predicates
fn even?(n)
fn odd?(n)
fn positive?(n)
fn negative?(n)
fn zero?(n)

// Constants (if possible)
let PI = 3.14159265358979323846
let E = 2.71828182845904523536
let TAU = 6.28318530717958647692

// Random (requires runtime support)
fn random()               // Random float 0-1
fn random-int(max)        // Random int 0 to max-1
fn random-range(min, max) // Random in range
```

### 4. Comparison and Equality (PRIORITY: MEDIUM)

**Protocol additions to `std.bg`**

```
protocol Comparable {
    fn compare(a, b)      // Returns -1, 0, or 1
}

protocol Hashable {
    fn hash(value)        // Return hash code
}

// Comparison helpers
fn <(a, b)
fn >(a, b)
fn <=(a, b)
fn >=(a, b)
fn min-of(coll)
fn max-of(coll)
fn min-by(coll, f)
fn max-by(coll, f)
```

### 5. Result/Option Types (PRIORITY: HIGH)

**Add to `std.bg`**

```
enum Option {
    Some { value }
    None {}
}

fn some(value) { Option::Some { value: value } }
fn none() { Option::None {} }
fn some?(opt)
fn none?(opt)
fn unwrap(opt)            // Panic if None
fn unwrap-or(opt, default)
fn map-option(opt, f)     // Apply f if Some
fn flat-map-option(opt, f)
fn option-to-result(opt, err)

// Result already exists in beagle.io, but should be in core
enum Result {
    Ok { value }
    Err { error }
}

fn ok(value) { Result::Ok { value: value } }
fn err(error) { Result::Err { error: error } }
fn ok?(result)
fn err?(result)
fn unwrap-result(result)  // Panic if Err
fn unwrap-err(result)
fn map-result(result, f)
fn map-err(result, f)
fn and-then(result, f)    // Flat map for Result
fn or-else(result, f)
fn result-to-option(result)
```

### 6. Function Utilities (PRIORITY: MEDIUM)

```
fn identity(x)            // Return x unchanged
fn constantly(value)      // Return a fn that always returns value
fn complement(pred)       // Negate a predicate
fn partial(f, ...args)    // Partial application
fn compose(f, g)          // Function composition f(g(x))
fn pipe(f, g)             // Reverse composition g(f(x))
fn juxt(...fns)           // Apply multiple fns, return vector of results
fn apply(f, args)         // Apply fn to vector of args
fn memoize(f)             // Cache function results
fn once(f)                // Only call f once, return cached result after
fn throttle(f, ms)        // Limit call frequency
fn debounce(f, ms)        // Delay until calls stop
```

### 7. Boolean/Logic (PRIORITY: LOW)

```
fn not(x)                 // Boolean negation
fn and(a, b)              // Logical and (not short-circuit)
fn or(a, b)               // Logical or (not short-circuit)
fn xor(a, b)
fn bool(x)                // Coerce to boolean
fn truthy?(x)             // Check if truthy
fn falsy?(x)              // Check if falsy (null or false)
```

### 8. Map/Dictionary Operations (PRIORITY: MEDIUM)

```
// Already have: assoc, get, keys

fn vals(m)                // Get all values
fn entries(m)             // Get key-value pairs as vectors
fn merge(m1, m2)          // Combine maps (m2 wins conflicts)
fn merge-with(f, m1, m2)  // Combine with conflict resolver
fn select-keys(m, ks)     // Keep only specified keys
fn dissoc(m, key)         // Remove key
fn update(m, key, f)      // Update value at key with f
fn update-in(m, path, f)  // Update nested value
fn get-in(m, path)        // Get nested value
fn assoc-in(m, path, val) // Set nested value
fn contains-key?(m, key)
fn invert(m)              // Swap keys and values
fn map-keys(m, f)         // Transform all keys
fn map-vals(m, f)         // Transform all values
fn filter-keys(m, pred)   // Keep entries where key matches
fn filter-vals(m, pred)   // Keep entries where value matches
```

### 9. Set Operations (PRIORITY: MEDIUM)

**Need to add PersistentSet first, then:**

```
struct PersistentSet (using PersistentMap internally)

fn set(...items)          // Create set
fn set?(x)                // Check if set
fn contains?(set, value)
fn add(set, value)        // Add element
fn remove(set, value)     // Remove element
fn union(s1, s2)
fn intersection(s1, s2)
fn difference(s1, s2)
fn symmetric-difference(s1, s2)
fn subset?(s1, s2)
fn superset?(s1, s2)
fn disjoint?(s1, s2)
```

### 10. Type Checking Predicates (PRIORITY: LOW)

```
fn nil?(x)               // Check if null
fn string?(x)
fn number?(x)            // Int or Float
fn int?(x)
fn float?(x)
fn bool?(x)
fn function?(x)
fn vector?(x)
fn map?(x)
fn array?(x)
fn atom?(x)
fn range?(x)
fn seq?(x)
```

### 11. Assertion/Testing (PRIORITY: LOW)

```
fn assert!(condition)
fn assert!(condition, message)
fn assert-eq!(expected, actual)
fn assert-eq!(expected, actual, message)
fn assert-ne!(a, b)
fn assert-throws!(f)
fn assert-throws!(f, error-type)
```

### 12. Debug/Development (PRIORITY: LOW)

```
fn dbg(value)             // Print and return value
fn tap(value, f)          // Apply f for side effect, return value
fn spy(value)             // Detailed debug output
fn type-name(value)       // Get type as string
fn inspect(value)         // Pretty-print structure
```

## Implementation Order

### Phase 1: Core Collection Functions
1. `map`, `filter`, `reduce` - Most essential
2. `any?`, `all?`, `none?`, `find`
3. `take`, `drop`, `concat`
4. `first`, `last`, `rest`, `empty?`, `count`

### Phase 2: String Operations
1. `split`, `join`
2. `trim`, `starts-with?`, `ends-with?`, `contains?`
3. `replace`, `index-of`
4. `pad-left`, `pad-right`

### Phase 3: Result/Option
1. Define `Option` enum
2. Helper functions for Option
3. Move `Result` to core, add helpers

### Phase 4: Map Operations
1. `vals`, `entries`, `merge`
2. `dissoc`, `update`
3. `get-in`, `assoc-in`, `update-in`

### Phase 5: Function Utilities
1. `identity`, `constantly`, `complement`
2. `partial`, `compose`, `pipe`
3. `apply`

### Phase 6: Additional Math
1. Trig functions
2. `pow`, `log`, `exp`
3. `max`, `min`, `clamp`
4. Number predicates

### Phase 7: Sorting and Advanced Collections
1. `sort`, `sort-by`
2. `group-by`, `frequencies`
3. `distinct`, `partition`

### Phase 8: Set Type
1. Implement PersistentSet
2. Set operations

## Notes

- All collection functions should work with any Seqable type
- Consider lazy evaluation for infinite sequences later
- Many of these can be built on `reduce`
- Need to ensure proper tail-call optimization for recursive implementations
