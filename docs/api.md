# Beagle API Reference

> Auto-generated documentation for the Beagle programming language.

**Overall documentation coverage:** 1560/1863 functions (83%)

## Quick Start

```beagle
// Hello World
fn main() {
    println("Hello, Beagle!")
}

// Using collections
let numbers = [1, 2, 3, 4, 5]
let doubled = map(numbers, fn(x) { x * 2 })
let sum = reduce(numbers, 0, fn(acc, x) { acc + x })

// File I/O
use beagle.fs as fs
let content = fs/blocking-read-file("/tmp/test.txt")
```

## Table of Contents

### Namespaces

- [beagle.core](#beagle-core) (240/410 documented)
- [beagle.fs](#beagle-fs) (58/59 documented)
- [beagle.async](#beagle-async) (149/152 documented)
- [beagle.io](#beagle-io) (20/21 documented)
- [beagle.timer](#beagle-timer) (8/10 documented)
- [beagle.regex](#beagle-regex) (9/9 documented)
- [beagle.reflect](#beagle-reflect) (22/22 documented)
- [beagle.ansi](#beagle-ansi) (54/54 documented)
- [beagle.bail](#beagle-bail) (11/11 documented)
- [beagle.base64](#beagle-base64) (11/11 documented)
- [beagle.bigint](#beagle-bigint) (22/26 documented)
- [beagle.channel](#beagle-channel) (21/21 documented)
- [beagle.cli](#beagle-cli) (32/32 documented)
- [beagle.containers](#beagle-containers) (34/51 documented)
- [beagle.csv](#beagle-csv) (13/13 documented)
- [beagle.date](#beagle-date) (19/20 documented)
- [beagle.effect](#beagle-effect) (1/2 documented)
- [beagle.ffi](#beagle-ffi) (47/50 documented)
- [beagle.glob](#beagle-glob) (15/15 documented)
- [beagle.hash](#beagle-hash) (45/45 documented)
- [beagle.hex](#beagle-hex) (5/5 documented)
- [beagle.http](#beagle-http) (56/70 documented)
- [beagle.ini](#beagle-ini) (16/16 documented)
- [beagle.ip](#beagle-ip) (20/31 documented)
- [beagle.iter](#beagle-iter) (34/38 documented)
- [beagle.json](#beagle-json) (37/39 documented)
- [beagle.log](#beagle-log) (22/22 documented)
- [beagle.mathx](#beagle-mathx) (21/21 documented)
- [beagle.mutable-array](#beagle-mutable-array) (19/21 documented)
- [beagle.os](#beagle-os) (15/15 documented)
- [beagle.path](#beagle-path) (23/23 documented)
- [beagle.priorityqueue](#beagle-priorityqueue) (18/18 documented)
- [beagle.process](#beagle-process) (8/8 documented)
- [beagle.random](#beagle-random) (14/16 documented)
- [beagle.regex-wrapper](#beagle-regex-wrapper) (14/14 documented)
- [beagle.repl](#beagle-repl) (20/20 documented)
- [beagle.repl-interactive](#beagle-repl-interactive) (4/4 documented)
- [beagle.repl-main](#beagle-repl-main) (15/15 documented)
- [beagle.repl-session](#beagle-repl-session) (13/13 documented)
- [beagle.runtime](#beagle-runtime) (2/2 documented)
- [beagle.semver](#beagle-semver) (28/29 documented)
- [beagle.socket](#beagle-socket) (8/8 documented)
- [beagle.spawn](#beagle-spawn) (9/11 documented)
- [beagle.stats](#beagle-stats) (18/18 documented)
- [beagle.stream](#beagle-stream) (41/86 documented)
- [beagle.string-builder](#beagle-string-builder) (20/20 documented)
- [beagle.struct-pack](#beagle-struct-pack) (60/60 documented)
- [beagle.template](#beagle-template) (25/25 documented)
- [beagle.test](#beagle-test) (11/11 documented)
- [beagle.test-async](#beagle-test-async) (12/20 documented)
- [beagle.text](#beagle-text) (30/36 documented)
- [beagle.textwrap](#beagle-textwrap) (13/15 documented)
- [beagle.time](#beagle-time) (17/17 documented)
- [beagle.tls](#beagle-tls) (6/6 documented)
- [beagle.url](#beagle-url) (20/20 documented)
- [beagle.ws](#beagle-ws) (35/35 documented)
- [global](#global) (0/1 documented)

### [Types](#types)

### [Enums](#enums-1)


---

## beagle.core

> **Documentation coverage:** 240/410 functions (58%)

#### `repr(value)` `builtin`

Return a string representation of a value that could be evaluated back.
Strings are quoted, special characters are escaped.

Examples:
  (repr 42)        ; => "42"
  (repr "hello")   ; => "\"hello\""
  (repr [1 "a"])   ; => "[1, \"a\"]"

#### `current-namespace()` `builtin`

Return the name of the current namespace as a string.

#### `to-number(string)` `builtin`

Parse a string into a number. Throws an error if the string is not a valid number.

Examples:
  (to-number "42")   ; => 42
  (to-number "-5")   ; => -5

#### `type-of(value)` `builtin`

Return a type descriptor for the given value.

Returns a Struct instance representing the type (e.g., Int, String, Array).

Examples:
  (type-of 42)        ; => Int
  (type-of "hello")   ; => String
  (type-of [1 2 3])   ; => Array

#### `get-os()` `builtin`

Return the name of the current operating system.

Returns one of: "macos", "linux", "windows", or "unknown".

#### `atom-address(value)` `builtin`

Return the raw memory address of a value as an integer.

Useful for identity comparisons and async coordination.

#### `equal(a, b)` `builtin`

Compare two values for deep equality.

Returns true if the values are structurally equal, false otherwise.

Examples:
  (equal 1 1)           ; => true
  (equal [1 2] [1 2])   ; => true
  (equal {:a 1} {:a 1}) ; => true

#### `apply(f, args)` `builtin`

Apply a function to an array of arguments.

Examples:
  (apply + [1 2 3])      ; => 6
  (apply max [3 7 2])    ; => 7
  (apply my-fn [a b c])  ; equivalent to (my-fn a b c)

#### `set-thread-exception-handler!(handler)` `builtin`

Set an exception handler for the current thread.

The handler function receives the exception value when an uncaught exception occurs.

#### `set-default-exception-handler!(handler)` `builtin`

Set the default exception handler for all threads.

This handler is used when a thread doesn't have its own handler set.

#### `assert!(value)` `builtin`

Assert that a value is truthy. Throws AssertError if the value is false, null, or 0.

#### `assert-eq!(actual, expected)` `builtin`

Assert that two values are equal. Throws AssertError showing both values on failure.

#### `assert-ne!(a, b)` `builtin`

Assert that two values are not equal. Throws AssertError on failure.

#### `gc()` `builtin`

Trigger garbage collection manually.

Normally GC runs automatically, but this can be useful for testing or freeing memory at specific points.

#### `heap-bytes()` `builtin`

Live/used bytes in the managed heap (the GC's own accounting). Read right AFTER gc() for the live set — a deterministic leak metric (a per-round leak grows it monotonically across checkpoints). Use RELATIVE deltas within ONE GC; absolute values are NOT comparable across GCs (mark-and-sweep reports committed minus free = live + fragmentation, not pure live; compacting/generational report the compacted/promoted live set). Excludes JIT code memory, which grows monotonically with redefinition and is not reclaimed.

#### `sqrt(x)` `builtin`

Return the square root of x.

Examples:
  (sqrt 4)    ; => 2.0
  (sqrt 2)    ; => 1.414...

#### `floor(x)` `builtin`

Return the largest integer less than or equal to x.

Examples:
  (floor 3.7)   ; => 3
  (floor -2.3)  ; => -3

#### `ceil(x)` `builtin`

Return the smallest integer greater than or equal to x.

Examples:
  (ceil 3.2)   ; => 4
  (ceil -2.7)  ; => -2

#### `abs(x)` `builtin`

Return the absolute value of x.

Examples:
  (abs -5)   ; => 5
  (abs 3.2)  ; => 3.2

#### `round(x)` `builtin`

Round x to the nearest integer.

Examples:
  (round 3.4)  ; => 3
  (round 3.6)  ; => 4

#### `truncate(x)` `builtin`

Truncate x toward zero (remove the fractional part).

Examples:
  (truncate 3.7)   ; => 3
  (truncate -3.7)  ; => -3

#### `max(a, b)` `builtin`

Return the larger of two numbers.

Examples:
  (max 3 7)  ; => 7
  (max -1 -5)  ; => -1

#### `min(a, b)` `builtin`

Return the smaller of two numbers.

Examples:
  (min 3 7)  ; => 3
  (min -1 -5)  ; => -5

#### `even?(n)` `builtin`

Return true if n is even.

Examples:
  (even? 4)  ; => true
  (even? 3)  ; => false

#### `odd?(n)` `builtin`

Return true if n is odd.

Examples:
  (odd? 3)  ; => true
  (odd? 4)  ; => false

#### `positive?(n)` `builtin`

Return true if n is positive (greater than zero).

Examples:
  (positive? 5)   ; => true
  (positive? -3)  ; => false
  (positive? 0)   ; => false

#### `negative?(n)` `builtin`

Return true if n is negative (less than zero).

Examples:
  (negative? -3)  ; => true
  (negative? 5)   ; => false
  (negative? 0)   ; => false

#### `zero?(n)` `builtin`

Return true if n is zero.

Examples:
  (zero? 0)  ; => true
  (zero? 5)  ; => false

#### `clamp(x, min_val, max_val)` `builtin`

Clamp x to be within the range [min_val, max_val].

Examples:
  (clamp 5 0 10)   ; => 5
  (clamp -5 0 10)  ; => 0
  (clamp 15 0 10)  ; => 10

#### `gcd(a, b)` `builtin`

Return the greatest common divisor of a and b.

Examples:
  (gcd 12 8)  ; => 4
  (gcd 17 5)  ; => 1

#### `lcm(a, b)` `builtin`

Return the least common multiple of a and b.

Examples:
  (lcm 4 6)  ; => 12
  (lcm 3 5)  ; => 15

#### `random()` `builtin`

Return a random floating-point number between 0.0 (inclusive) and 1.0 (exclusive).

Examples:
  (random)  ; => 0.7234... (varies)

#### `random-int(max)` `builtin`

Return a random integer from 0 (inclusive) to max (exclusive).

Examples:
  (random-int 10)  ; => 7 (varies, 0-9)

#### `random-range(min, max)` `builtin`

Return a random integer from min (inclusive) to max (exclusive).

Examples:
  (random-range 5 10)  ; => 7 (varies, 5-9)

#### `sin(x)` `builtin`

Return the sine of x (in radians).

Examples:
  (sin 0)       ; => 0.0
  (sin (/ PI 2))  ; => 1.0

#### `cos(x)` `builtin`

Return the cosine of x (in radians).

Examples:
  (cos 0)   ; => 1.0
  (cos PI)  ; => -1.0

#### `tan(x)` `builtin`

Return the tangent of x (in radians).

Examples:
  (tan 0)  ; => 0.0

#### `asin(x)` `builtin`

Return the arc sine (inverse sine) of x in radians.

The result is in the range [-PI/2, PI/2]. x must be in [-1, 1].

#### `acos(x)` `builtin`

Return the arc cosine (inverse cosine) of x in radians.

The result is in the range [0, PI]. x must be in [-1, 1].

#### `atan(x)` `builtin`

Return the arc tangent (inverse tangent) of x in radians.

The result is in the range [-PI/2, PI/2].

#### `atan2(y, x)` `builtin`

Return the arc tangent of y/x in radians, using signs to determine the quadrant.

The result is in the range [-PI, PI].

#### `exp(x)` `builtin`

Return e raised to the power x (e^x).

Examples:
  (exp 0)  ; => 1.0
  (exp 1)  ; => 2.718...

#### `log(x)` `builtin`

Return the natural logarithm (base e) of x.

Examples:
  (log 1)  ; => 0.0
  (log E)  ; => 1.0

#### `log10(x)` `builtin`

Return the base-10 logarithm of x.

Examples:
  (log10 10)   ; => 1.0
  (log10 100)  ; => 2.0

#### `log2(x)` `builtin`

Return the base-2 logarithm of x.

Examples:
  (log2 2)  ; => 1.0
  (log2 8)  ; => 3.0

#### `pow(base, exponent)` `builtin`

Return base raised to the power exponent.

Examples:
  (pow 2 3)  ; => 8.0
  (pow 10 2) ; => 100.0

#### `to-float(x)` `builtin`

Convert an integer to a floating-point number.

Examples:
  (to-float 42)  ; => 42.0

#### `to-int(x)` `builtin`

Convert a float to an integer by truncating towards zero. If already an integer, returns unchanged.

Examples:
  (to-int 3.7)   ; => 3
  (to-int -3.7)  ; => -3
  (to-int 42)    ; => 42

#### `parse-float-string(s)` `builtin`

Parse a string into a Float using correctly-rounded IEEE-754 parsing. Returns null if the string is not a valid float.

Examples:
  (parse-float-string "6.022e23")  ; => 6.022e23
  (parse-float-string "nope")      ; => null

#### `thread(f)` `builtin`

Spawn a new thread to execute function f.

Returns a Thread object that can be used with thread-join.

Examples:
  (let t (thread fn() { expensive-computation() }))
  (thread-join t)  ; wait for result

#### `eval(code)` `builtin`

Evaluate a string as Beagle code at runtime.

Returns the result of the evaluated expression.

Examples:
  eval("1 + 2")  ; => 3

#### `eval-in-ns(code, namespace)` `builtin`

Evaluate a string as Beagle code in a specific namespace.

The code is compiled with access to all bindings and imports of the given namespace.

Examples:
  eval-in-ns("my-fn(42)", "my_app")

#### `last-eval-namespace()` `builtin`

Return the namespace the most recent eval-in-ns on the calling thread ended in (after any `namespace X` directive), or null. Lets a REPL session track its own current namespace without reading the shared global current-namespace.

#### `sleep(ms)` `builtin`

Pause execution for the specified number of milliseconds.

Examples:
  (sleep 1000)  ; sleep for 1 second

#### `time-now()` `builtin`

Return the current time in milliseconds since the Unix epoch.

Useful for timing operations or generating timestamps.

#### `thread-id()` `builtin`

Return the ID of the current thread as an integer.

#### `get-cpu-count()` `builtin`

Return the number of CPU cores available on the system.

#### `substring(string, start, end)` `builtin`

Extract a substring from a string.

Arguments:
  string - The source string
  start  - Starting index (0-based, inclusive)
  end    - Ending index (exclusive)

Examples:
  (substring "hello" 1 4)  ; => "ell"

#### `uppercase(string)` `builtin`

Convert a string to uppercase.

Examples:
  (uppercase "hello")  ; => "HELLO"

#### `lowercase(string)` `builtin`

Convert a string to lowercase.

Examples:
  (lowercase "HELLO")  ; => "hello"

#### `split(string, delimiter)` `builtin`

Split a string into an array of substrings.

Examples:
  (split "a,b,c" ",")  ; => ["a", "b", "c"]

#### `trim(string)` `builtin`

Remove leading and trailing whitespace from a string.

Examples:
  (trim "  hello  ")  ; => "hello"

#### `trim-left(string)` `builtin`

Remove leading whitespace from a string.

Examples:
  (trim-left "  hello  ")  ; => "hello  "

#### `trim-right(string)` `builtin`

Remove trailing whitespace from a string.

Examples:
  (trim-right "  hello  ")  ; => "  hello"

#### `starts-with?(string, prefix)` `builtin`

Check if a string starts with a given prefix.

Examples:
  (starts-with? "hello" "he")  ; => true

#### `ends-with?(string, suffix)` `builtin`

Check if a string ends with a given suffix.

Examples:
  (ends-with? "hello" "lo")  ; => true

#### `string-contains?(string, substr)` `builtin`

Check if a string contains a substring.

Examples:
  (string-contains? "hello" "ell")  ; => true

#### `index-of(string, substr)` `builtin`

Find the first index of a substring in a string.

Returns -1 if not found.

Examples:
  (index-of "hello" "l")  ; => 2

#### `last-index-of(string, substr)` `builtin`

Find the last index of a substring in a string.

Returns -1 if not found.

Examples:
  (last-index-of "hello" "l")  ; => 3

#### `replace(string, from, to)` `builtin`

Replace all occurrences of a substring.

Examples:
  (replace "hello" "l" "L")  ; => "heLLo"

#### `blank?(string)` `builtin`

Check if a string is empty or contains only whitespace.

Examples:
  (blank? "")       ; => true
  (blank? "  ")     ; => true
  (blank? "hello")  ; => false

#### `replace-first(string, from, to)` `builtin`

Replace the first occurrence of a substring.

Examples:
  (replace-first "hello" "l" "L")  ; => "heLlo"

#### `pad-left(string, width, pad_char)` `builtin`

Pad a string on the left to a given width.

Examples:
  (pad-left "42" 5 "0")  ; => "00042"

#### `pad-right(string, width, pad_char)` `builtin`

Pad a string on the right to a given width.

Examples:
  (pad-right "42" 5 "0")  ; => "42000"

#### `lines(string)` `builtin`

Split a string into an array of lines.

Examples:
  (lines "a\nb\nc")  ; => ["a", "b", "c"]

#### `words(string)` `builtin`

Split a string into an array of words (whitespace-separated).

Examples:
  (words "hello world")  ; => ["hello", "world"]

#### `json-encode(value)` `builtin`

Serialize a Beagle value to a JSON string.

Supports primitives, vectors, maps, and nested structures.

Examples:
  (json-encode {:name "alice" :age 30})
  ; => "{\"name\":\"alice\",\"age\":30}"

#### `json-decode(json-string)` `builtin`

Parse a JSON string to a Beagle value.

JSON objects become maps with string keys (use `get` to access).
JSON arrays become vectors.

Examples:
  (let data (json-decode "{\"name\":\"alice\"}"))
  (get data "name")  ; => "alice"

#### `to-string(x)`

Convert any value to its display string. Routes through the `Format` protocol,
so a type's `extend T with Format` impl is honored — `to-string(x)`, `${x}`
interpolation, and `println(x)` all produce the SAME output. Equivalent to
`format(x, 0)`.

Examples:
  to-string(42)     // => "42"
  to-string([1, 2])  // => "[1, 2]"

#### `format-rust-vec-helper(vec, idx, len, acc)`

Helper: recursively format a Rust-backed vector's elements into acc.

#### `format-rust-map-entries(map, keys-vec, idx, len, acc)`

Helper: recursively format a Rust-backed map's key/value entries into acc.

#### `deref(atom)`

Dereference an atom to get its current value.

Atoms provide thread-safe mutable state. Use `deref` to read the value.

Examples:
  let a = atom(0)
  deref(a)  // => 0

#### `swap!(atom, f)`

Atomically update an atom's value by applying a function.

Takes the current value, applies f to it, and attempts to set the new value.
Retries if another thread changed the value concurrently.

Examples:
  let counter = atom(0)
  swap!(counter, fn(x) { x + 1 })  // => 1

#### `reset!(atom, value)`

Reset an atom to a new value unconditionally.

Unlike `swap!`, this does not read the current value first.

Examples:
  let a = atom(0)
  reset!(a, 42)
  deref(a)  // => 42

#### `compare-and-swap!(atom, old, new)`

Atomically compare and swap an atom's value.

If the atom's current value equals `old`, set it to `new` and return true.
Otherwise return false without changing the value.

Examples:
  let a = atom(0)
  compare-and-swap!(a, 0, 1)  // => true, a is now 1
  compare-and-swap!(a, 0, 2)  // => false, a is still 1

#### `atom(value)`

Create a new atom with the given initial value.

Atoms provide thread-safe mutable state.

Examples:
  let counter = atom(0)
  swap!(counter, fn(x) { x + 1 })
  deref(counter)  // => 1

#### `instance-of(value, type)`

Test whether a value is an instance of the given type descriptor.
Handles both primitive types (negative id) and custom structs (by struct-id).

#### `struct?(value)`

True if `value` is an instance of a user-defined struct (e.g. a record, or a
value type like BigInt/SemVer/DateTime). False for every built-in type —
numbers, strings, keywords, booleans, vectors, maps, sets, functions, null.

Built-in types are assigned negative type ids; user structs get non-negative
ones, which is exactly this test. Used by `compare`/`sort` to decide between
the native `<` ordering and the `Comparable` protocol.

#### `format-persistent-vector-helper(vec, idx, acc, depth)`

Helper: recursively format a PersistentVector's elements into acc.

#### `join(coll, sep)`

Join elements of a collection into a string with a separator.

Works with any collection that implements Indexed and Length protocols.

Examples:
  join([1, 2, 3], ", ")  // => "1, 2, 3"
  join(["a", "b"], "-") // => "a-b"

#### `join-helper(coll, sep, idx, acc)`

Helper: recursively join a collection's elements into acc, separated by sep.

#### `parse-int(s)`

Parse a string into an integer.

Returns null if the string is not a valid integer.
Supports negative numbers with leading '-'.

Examples:
  parse-int("42")   // => 42
  parse-int("-5")   // => -5
  parse-int("abc")  // => null

#### `parse-int-helper(s, idx, acc)`

Helper: accumulate digits of s from idx into acc; null on a non-digit or overflow.

#### `new-string-buffer()`

Create a new empty StringBuffer.

#### `print(...args)`

Print arguments separated by spaces without a trailing newline.

Accepts any number of arguments. Each is formatted and printed.
Returns the last argument.

Examples:
  print("Hello", "World")  // prints: Hello World

#### `println(...args)`

Print arguments separated by spaces with a trailing newline.

Accepts any number of arguments. Each is formatted and printed.
Returns the last argument.

Examples:
  println("Hello", "World")  // prints: Hello World\n
  println(1, 2, 3)            // prints: 1 2 3\n

#### `keyword?(value)`

Check if a value is a keyword.

Examples:
  keyword?(:foo)  // => true
  keyword?("foo") // => false

#### `keyword->string(kw)`

Convert a keyword to a string.

Examples:
  keyword->string(:foo)  // => "foo"

#### `string->keyword(str)`

Convert a string to a keyword.

Examples:
  string->keyword("foo")  // => :foo

#### `range(start, end)`

Create a range from start (inclusive) to end (exclusive).

Ranges are iterable and can be used with for loops and seq functions.

Examples:
  for i in range(0, 5) { println(i) }  // prints 0, 1, 2, 3, 4
  reduce(fn(a, x) { a + x }, 0, range(1, 5))  // => 10

#### `range-step(start, end, step)`

Create a range with a custom step value.

Examples:
  range-step(0, 10, 2)  // => 0, 2, 4, 6, 8

#### `reduce(f, init, coll)`

Fold a collection with an accumulator function.

Calls (f acc elem) for each element, threading the accumulator through.
Works with any Seqable type.

Examples:
  reduce(fn(acc, x) { acc + x }, 0, [1, 2, 3])  // => 6
  reduce(fn(acc, x) { push(acc, x * 2) }, [], [1, 2, 3])  // => [2, 4, 6]

#### `map(f, coll)`

Apply a function to each element of a collection.

Returns a new vector with the transformed elements.

Examples:
  map(fn(x) { x * 2 }, [1, 2, 3])  // => [2, 4, 6]
  map(uppercase, ["a", "b"])      // => ["A", "B"]

#### `map-indexed(f, coll)`

Map `f`, which receives (index, element), over `coll`.

Examples:
  map-indexed(fn(i, x) { [i, x] }, ["a", "b"])  // => [[0, "a"], [1, "b"]]

#### `remove(pred, coll)`

Remove the elements for which `pred` is true — the complement of `filter`.

Examples:
  remove(fn(x) { x > 2 }, [1, 2, 3, 4])  // => [1, 2]

#### `keep(f, coll)`

Map `f` over `coll`, keeping only the non-null results (map + drop-nulls).

Examples:
  keep(fn(x) { if x > 2 { x * 10 } else { null } }, [1, 2, 3, 4])  // => [30, 40]

#### `keep-indexed(f, coll)`

Like `keep`, but `f` receives (index, element); keeps non-null results.

#### `split-at(n, coll)`

Split `coll` at index `n`: returns [take(n, coll), drop(n, coll)].

Examples:
  split-at(2, [1, 2, 3, 4])  // => [[1, 2], [3, 4]]

#### `split-with(pred, coll)`

Split `coll` at the first element failing `pred`:
returns [take-while(pred, coll), drop-while(pred, coll)].

Examples:
  split-with(fn(x) { x < 3 }, [1, 2, 3, 4, 1])  // => [[1, 2], [3, 4, 1]]

#### `filter(pred, coll)`

Keep elements where the predicate returns true.

Returns a new vector containing only matching elements.

Examples:
  filter(even?, [1, 2, 3, 4])  // => [2, 4]
  filter(fn(s) { length(s) > 0 }, ["", "a", ""])  // => ["a"]

#### `any?(pred, coll)`

Check if any element matches the predicate.

Returns true if pred returns true for at least one element.

Examples:
  any?(even?, [1, 2, 3])  // => true
  any?(even?, [1, 3, 5])  // => false

#### `all?(pred, coll)`

Check if all elements match the predicate.

Returns true if pred returns true for every element.

Examples:
  all?(even?, [2, 4, 6])  // => true
  all?(even?, [1, 2, 3])  // => false

#### `none?(pred, coll)`

Check if no elements match the predicate.

Returns true if pred returns false for every element.

Examples:
  none?(even?, [1, 3, 5])  // => true
  none?(even?, [1, 2, 3])  // => false

#### `not-every?(pred, coll)`

Check if at least one element does NOT match the predicate.

Equivalent to (not (all? coll pred)).

Examples:
  not-every?(even?, [1, 2, 3])  // => true
  not-every?(even?, [2, 4, 6])  // => false

#### `find(pred, coll)`

Find the first element matching the predicate.

Returns the element or null if no match.

Examples:
  find(even?, [1, 2, 3, 4])  // => 2
  find(even?, [1, 3, 5])    // => null

#### `find-index(pred, coll)`

Find the index of the first element matching the predicate.

Returns the index or -1 if no match.

Examples:
  find-index(even?, [1, 2, 3, 4])  // => 1
  find-index(even?, [1, 3, 5])    // => -1

#### `take(n, coll)`

Take the first n elements from a collection.

Returns a new vector with at most n elements.

Examples:
  take(3, [1, 2, 3, 4, 5])  // => [1, 2, 3]
  take(5, [1, 2])        // => [1, 2]

#### `drop(n, coll)`

Drop the first n elements from a collection.

Returns a new vector with the remaining elements.

Examples:
  drop(2, [1, 2, 3, 4, 5])  // => [3, 4, 5]
  drop(5, [1, 2])        // => []

#### `take-while(pred, coll)`

Take elements while the predicate returns true.

Stops at the first element where pred returns false.

Examples:
  take-while(fn(x) { x < 4 }, [1, 2, 3, 4, 1])  // => [1, 2, 3]

#### `drop-while(pred, coll)`

Drop elements while the predicate returns true.

Returns elements starting from the first where pred returns false.

Examples:
  drop-while(fn(x) { x < 3 }, [1, 2, 3, 4, 1])  // => [3, 4, 1]

#### `slice(coll, start, end)`

Get a sub-collection from start (inclusive) to end (exclusive).

Examples:
  slice([0, 1, 2, 3, 4], 1, 4)  // => [1, 2, 3]

#### `enumerate(coll)`

Return pairs of [index, element] for each element.

Examples:
  enumerate(["a", "b", "c"])  // => [[0, "a"], [1, "b"], [2, "c"]]

#### `remove-at(coll, idx)`

Remove element at the specified index.

Returns the collection unchanged if index is out of bounds.

Examples:
  remove-at([1, 2, 3, 4], 1)  // => [1, 3, 4]

#### `count(coll)`

Return the number of elements in a collection.

Alias for length, more idiomatic for collections.

Examples:
  count([1, 2, 3])  // => 3
  count("hello")  // => 5

#### `empty?(coll)`

Check if a collection has no elements.

Examples:
  empty?([])    // => true
  empty?([1])   // => false
  empty?("")    // => true

#### `first-of(coll)`

Get the first element of a collection.

Returns null if the collection is empty.
Named differently to avoid conflict with Seq protocol's first.

Examples:
  first-of([1, 2, 3])  // => 1
  first-of([])       // => null

#### `last(coll)`

Get the last element of a collection.

Returns null if the collection is empty.

Examples:
  last([1, 2, 3])  // => 3
  last([])       // => null

#### `rest(coll)`

Get all elements except the first.

Examples:
  rest([1, 2, 3])  // => [2, 3]
  rest([1])      // => []

#### `butlast(coll)`

Get all elements except the last one.

Examples:
  butlast([1, 2, 3])  // => [1, 2]
  butlast([1])      // => []

#### `nth(coll, n)`

Get the element at index n with bounds checking.

Returns null if index is out of bounds.

Examples:
  nth([1, 2, 3], 1)   // => 2
  nth([1, 2, 3], 10)  // => null

#### `second(coll)`

Get the second element of a collection.

Examples:
  second([1, 2, 3])  // => 2

#### `third(coll)`

Get the third element of a collection.

Examples:
  third([1, 2, 3])  // => 3

#### `compare(a, b)`

Three-way comparison: returns -1 if a < b, 0 if equal, 1 if a > b.

Works directly on primitives (numbers, strings, booleans) and on any heap
value type implementing the `Comparable` protocol (BigInt, SemVer, DateTime,
user types via `extend T with Comparable`). Mixed-type comparisons are
undefined. Backs `sort`/`sort-by`/`min-of`/`max-of`.

Examples:
  compare(3, 5)      // => -1
  compare("b", "a")  // => 1

#### `min-of(coll)`

Return the minimum element in a collection (by `compare`, so value types work).

Examples:
  min-of([3, 1, 4, 1, 5])  // => 1
  min-of([])           // => null

#### `max-of(coll)`

Return the maximum element in a collection.

Elements must be comparable with >.

Examples:
  max-of([3, 1, 4, 1, 5])  // => 5
  max-of([])           // => null

#### `min-by(f, coll)`

Return the element with the minimum key value.

Compares elements by applying f to each and using < on the results.

Examples:
  min-by(length, ["abc", "a", "ab"])  // => "a"

#### `max-by(f, coll)`

Return the element with the maximum key value.

Compares elements by applying f to each and using > on the results.

Examples:
  max-by(length, ["a", "abc", "ab"])  // => "abc"

#### `reduce-right(f, init, coll)`

Reduce a collection from right to left.

Like reduce, but processes elements in reverse order.

Examples:
  reduce-right(fn(acc, x) { push(acc, x) }, [], [1, 2, 3])  // => [3, 2, 1]

#### `concat(coll1, coll2)`

Concatenate two collections into one.

Returns a new vector with elements from both collections.

Examples:
  concat([1, 2], [3, 4])  // => [1, 2, 3, 4]

#### `flatten(coll)`

Flatten one level of nesting in a collection.

Examples:
  flatten([[1, 2], [3, 4]])  // => [1, 2, 3, 4]

#### `flat-map(f, coll)`

Map a function over a collection then flatten the result.

Examples:
  flat-map(fn(x) { [x, x] }, [1, 2, 3])  // => [1, 1, 2, 2, 3, 3]

#### `zip(coll1, coll2)`

Pair up elements from two collections into a vector of pairs.

Examples:
  zip([1, 2, 3], ["a", "b", "c"])  // => [[1, "a"], [2, "b"], [3, "c"]]

#### `zip-with(f, coll1, coll2)`

Combine elements from two collections using a function.

Like zip, but applies f to each pair instead of creating tuples.

Examples:
  zip-with(fn(a, b) { a + b }, [1, 2, 3], [10, 20, 30])  // => [11, 22, 33]

#### `zipmap(keys-coll, vals-coll)`

Create a map from parallel key and value collections.

Pairs up keys[i] with vals[i] to form map entries.

Examples:
  zipmap([:a, :b, :c], [1, 2, 3])  // => {:a 1, :b 2, :c 3}

#### `interleave(coll1, coll2)`

Interleave elements from two collections.

Returns a vector alternating between elements of coll1 and coll2.

Examples:
  interleave([1, 2, 3], [:a, :b, :c])  // => [1, :a, 2, :b, 3, :c]

#### `interpose(sep, coll)`

Insert a separator between each element of a collection.

Examples:
  interpose(0, [1, 2, 3])  // => [1, 0, 2, 0, 3]

#### `into(target, source)`

Pour elements from source collection into target collection.

For vectors: appends elements. For maps: source should be [key, value] pairs.
For sets: adds elements.

Examples:
  into([1, 2], [3, 4])          // => [1, 2, 3, 4]
  into({}, [[:a, 1], [:b, 2]])   // => {:a 1, :b 2}
  into(#{1, 2}, [2, 3, 4])       // => #{1, 2, 3, 4}

#### `reverse(coll)`

Reverse the order of elements in a collection.

Returns a new vector with elements in reverse order.

Examples:
  reverse([1, 2, 3])  // => [3, 2, 1]

#### `repeat(value, n)`

Create a vector with a value repeated n times.

Examples:
  repeat("x", 3)  // => ["x", "x", "x"]

#### `repeatedly(f, n)`

Call a function n times and collect the results.

Examples:
  repeatedly(fn() { random() }, 3)  // => [0.1, 0.7, 0.4] (random values)

#### `iterate(f, init, n)`

Generate a sequence by repeatedly applying a function.

Starts with init, then f(init), f(f(init)), etc. for n values.

Examples:
  iterate(fn(x) { x * 2 }, 1, 5)  // => [1, 2, 4, 8, 16]

#### `partition(n, coll)`

Split a collection into groups of n elements.

Examples:
  partition(2, [1, 2, 3, 4, 5, 6])  // => [[1, 2], [3, 4], [5, 6]]

#### `partition-by(f, coll)`

Split a collection when the key function's return value changes.

Groups consecutive elements with the same key value.

Examples:
  partition-by(identity, [1, 1, 2, 2, 1])  // => [[1, 1], [2, 2], [1]]

#### `group-by(f, coll)`

Group elements by the result of a key function.

Returns a map from keys to vectors of elements with that key.

Examples:
  group-by(even?, [1, 2, 3, 4, 5])  // => {true: [2, 4], false: [1, 3, 5]}

#### `frequencies(coll)`

Count occurrences of each value in a collection.

Returns a map from values to their counts.

Examples:
  frequencies([1, 2, 1, 3, 1, 2])  // => {1: 3, 2: 2, 3: 1}

#### `distinct(coll)`

Remove duplicate values from a collection.

Keeps the first occurrence of each value.

Examples:
  distinct([1, 2, 1, 3, 2, 1])  // => [1, 2, 3]

#### `dedupe(coll)`

Remove consecutive duplicate elements.

Keeps the first occurrence of each run of duplicates.

Examples:
  dedupe([1, 1, 2, 2, 2, 1, 1])  // => [1, 2, 1]

#### `tim-allocate-array(size)`

Allocate a raw mutable array of the given size, filled with null (write-barrier safe).

#### `tim-reverse(arr, lo, hi)`

Reverse arr[lo..=hi] in place.

#### `tim-count-run-and-make-ascending(arr, start, n, less)`

Scan a natural run at start (reversing it if strictly descending); return its length.

#### `tim-binary-insertion-sort(arr, lo, hi, start, less)`

Binary-insertion-sort arr[lo..hi], with arr[lo..start] already sorted (stable).

#### `tim-calc-min-run(n)`

Compute Timsort's minrun in [32,64) so n/minrun is just below a power of two.

#### `tim-merge(arr, temp, base1, len1, base2, len2, less)`

Merge two adjacent runs, copying the smaller side into temp.

#### `tim-merge-lo(arr, temp, base1, len1, base2, len2, less)`

Merge two runs with the left run buffered in temp, scanning forward (stable on ties).

#### `tim-merge-hi(arr, temp, base1, len1, base2, len2, less)`

Merge two runs with the right run buffered in temp, scanning backward (stable on ties).

#### `tim-merge-at(arr, temp, stack-base, stack-len, stack-size, i, less)`

Merge the pending runs at stack[i] and stack[i+1]; return the new stack size.

#### `tim-merge-collapse(arr, temp, stack-base, stack-len, stack-size, less)`

Merge pending runs until Timsort's stack invariants hold; return the new stack size.

#### `tim-merge-force-collapse(arr, temp, stack-base, stack-len, stack-size, less)`

Merge all remaining pending runs at the end of the sort.

#### `tim-sort-array(arr, n, less)`

Sort a raw array of n elements in place using Timsort and the given less comparator.

#### `sort-with(less, coll)`

Sort a collection with a user-provided `less` comparator.

`less(a, b)` must return true when `a` should come before `b`. Stable and
O(n log n) via Timsort; O(n) on already-sorted or reverse-sorted input.

#### `sort(coll)`

Sort a collection in ascending order.

Stable O(n log n) Timsort; runs in O(n) on already-sorted or reverse-sorted
input. Elements must be comparable with `<`.

Examples:
  sort([3, 1, 4, 1, 5])  // => [1, 1, 3, 4, 5]

#### `sort-by(f, coll)`

Sort a collection by a key function.

Elements are ordered by comparing `f(elem)` values. `f` is invoked on every
comparison — wrap in `memoize` if it's expensive and you have many ties.

Examples:
  sort-by(fn(x) { x.age }, [{ name: "bob", age: 30 }, { name: "alice", age: 25 }])
  // => [{ name: "alice", age: 25 }, { name: "bob", age: 30 }]

#### `with-open(resource, body)`

Run `body(resource)` and guarantee `close(resource)` runs afterwards on EVERY
exit path — normal return OR a thrown exception — then return body's value (or
re-raise). `resource` must implement the `Closeable` protocol (e.g. an io/File).
This is Beagle's RAII / try-with-resources: no more leaked handles on a throw.

Examples:
  with-open(unwrap(io/open("/tmp/x", "r")), fn(f) {
      io/read-line(f)
  })

#### `identity(x)`

Return the input unchanged.

Useful as a default function argument.

Examples:
  identity(42)  // => 42
  map(identity, [1, 2, 3])  // => [1, 2, 3]

#### `constantly(value)`

Return a function that always returns the given value.

Ignores any arguments passed to the returned function.

Examples:
  let always-42 = constantly(42)
  always-42("ignored")  // => 42

#### `complement(pred)`

Return a function that negates a predicate.

Examples:
  let not-even? = complement(even?)
  not-even?(3)  // => true

#### `compose(f, g)`

Compose two functions: (compose f g) returns fn(x) { f(g(x)) }.

The second function is applied first, then the first.

Examples:
  let add1-then-double = compose(fn(x) { x * 2 }, fn(x) { x + 1 })
  add1-then-double(3)  // => 8  (3+1=4, 4*2=8)

#### `partial(f, a)`

Partially apply a function by fixing its first argument.

Examples:
  let add10 = partial(fn(a, b) { a + b }, 10)
  add10(5)  // => 15

#### `not(x)`

Boolean negation.

Returns true if x is falsy, false if x is truthy.

Examples:
  not(true)   // => false
  not(false)  // => true
  not(null)   // => true

#### `truthy?(x)`

Check if a value is truthy (not null and not false).

Examples:
  truthy?(1)      // => true
  truthy?(null)   // => false
  truthy?(false)  // => false

#### `falsy?(x)`

Check if a value is falsy (null or false).

Examples:
  falsy?(null)   // => true
  falsy?(false)  // => true
  falsy?(0)      // => false

#### `nil?(x)`

Check if a value is null.

Examples:
  nil?(null)  // => true
  nil?(0)     // => false

#### `some?(x)`

Check if a value is not null.

Examples:
  some?(0)     // => true
  some?(null)  // => false

#### `vals(m)`

Get all values from a map as a vector.

Examples:
  vals({:a 1, :b 2})  // => [1, 2]

#### `find-entry(m, key)`

Find the [key value] entry for a key in a map, or null if the key is absent.

This is Clojure's `find`, renamed because Beagle's `find` already means
"first element matching a predicate" (see `find`/`find-index`). Unlike
`get`, it distinguishes a missing key (null) from a key whose stored value
is null (returns `[key null]`).

Examples:
  find-entry({:a 1}, :a)     // => [:a, 1]
  find-entry({:a 1}, :b)     // => null
  find-entry({:a null}, :a)  // => [:a, null]

#### `dissoc(m, key)`

Remove a key from a map.

Returns a new map without the specified key.

Examples:
  dissoc({:a 1, :b 2}, :a)  // => {:b 2}

#### `merge(m1, m2)`

Merge two maps together.

m2's values take precedence for duplicate keys.

Examples:
  merge({:a 1, :b 2}, {:b 3, :c 4})  // => {:a 1, :b 3, :c 4}

#### `merge-with(f, m1, m2)`

Merge two maps using a function to resolve key conflicts.

When a key exists in both maps, calls f(v1, v2) to get the merged value.

Examples:
  merge-with(fn(a, b) { a + b }, {:a 1}, {:a 2, :b 3})  // => {:a 3, :b 3}

#### `merge-with-helper(f, m1, m2, ks, idx, acc)`

Helper: fold m2's keys into acc, resolving conflicts with f.

#### `select-keys(m, ks)`

Keep only the specified keys from a map.

Returns a new map with only the keys in ks.

Examples:
  select-keys({:a 1, :b 2, :c 3}, [:a, :c])  // => {:a 1, :c 3}

#### `select-keys-helper(m, ks, idx, acc)`

Helper: accumulate the entries of m whose keys are in ks.

#### `update(m, key, f)`

Update a value at a key using a function.

Applies f to the current value at key and associates the result.

Examples:
  update({:a 1}, :a, fn(x) { x + 1 })  // => {:a 2}

#### `get-in(m, path)`

Get a nested value using a path of keys.

Returns null if any key in the path is not found.

Examples:
  get-in({:a {:b {:c 42}}}, [:a, :b, :c])  // => 42
  get-in({:a {:b 1}}, [:a, :x])           // => null

#### `assoc-in(m, path, val)`

Set a nested value using a path of keys.

Creates intermediate maps as needed.

Examples:
  assoc-in({:a {:b 1}}, [:a, :b], 42)  // => {:a {:b 42}}
  assoc-in({}, [:a, :b, :c], 1)         // => {:a {:b {:c 1}}}

#### `update-in(m, path, f)`

Update a nested value using a path of keys and a function.

Applies f to the current value at the path.

Examples:
  update-in({:a {:b 1}}, [:a, :b], fn(x) { x + 1 })  // => {:a {:b 2}}

#### `contains-key?(m, key)`

Check if a map contains a key.

Examples:
  contains-key?({:a 1, :b 2}, :a)  // => true
  contains-key?({:a 1, :b 2}, :c)  // => false

#### `invert(m)`

Swap keys and values in a map.

Values become keys and keys become values.

Examples:
  invert({:a 1, :b 2})  // => {1: :a, 2: :b}

#### `map-keys(m, f)`

Apply a function to all keys in a map.

Returns a new map with transformed keys, same values.

Examples:
  map-keys({:a 1, :b 2}, keyword->string)  // => {"a": 1, "b": 2}

#### `map-keys-helper(m, f, ks, idx, acc)`

Helper: rebuild a map applying f to each key.

#### `map-vals(m, f)`

Apply a function to all values in a map.

Returns a new map with same keys, transformed values.

Examples:
  map-vals({:a 1, :b 2}, fn(x) { x * 2 })  // => {:a 2, :b 4}

#### `map-vals-helper(m, f, ks, idx, acc)`

Helper: rebuild a map applying f to each value.

#### `filter-keys(m, pred)`

Keep only entries where the key satisfies the predicate.

Examples:
  filter-keys({:a 1, :ab 2, :abc 3}, fn(k) { length(keyword->string(k)) > 1 })
  ; => {:ab 2, :abc 3}

#### `filter-keys-helper(m, pred, ks, idx, acc)`

Helper: accumulate entries of m whose keys satisfy pred.

#### `filter-vals(m, pred)`

Keep only entries where the value satisfies the predicate.

Examples:
  filter-vals({:a 1, :b 2, :c 3}, fn(v) { v > 1 })  // => {:b 2, :c 3}

#### `filter-vals-helper(m, pred, ks, idx, acc)`

Helper: accumulate entries of m whose values satisfy pred.

#### `format-rust-set-elements(elems-vec, idx, len, acc)`

Helper: recursively format a Rust-backed set's elements into acc.

#### `set?(x)`

Check if a value is a PersistentSet.

Examples:
  set?(#{1, 2, 3})  // => true
  set?([1, 2, 3])   // => false

#### `set-contains?(set, elem)`

Check if an element is in a set.

Examples:
  set-contains?(#{1, 2, 3}, 2)  // => true
  set-contains?(#{1, 2, 3}, 5)  // => false

#### `set-add(set, elem)`

Add an element to a set.

Returns a new set with the element added (no-op if already present).

Examples:
  set-add(#{1, 2}, 3)  // => #{1, 2, 3}
  set-add(#{1, 2}, 2)  // => #{1, 2}

#### `set-remove(set, elem)`

Remove an element from a set.

Returns a new set without the element.

Examples:
  set-remove(#{1, 2, 3}, 2)  // => #{1, 3}

#### `set-union(s1, s2)`

Return the union of two sets.

Contains all elements from both sets.

Examples:
  set-union(#{1, 2}, #{2, 3})  // => #{1, 2, 3}

#### `set-union-helper(s1, elems2, idx, len)`

Helper: add each element of elems2 into s1.

#### `set-intersection(s1, s2)`

Return the intersection of two sets.

Contains only elements present in both sets.

Examples:
  set-intersection(#{1, 2, 3}, #{2, 3, 4})  // => #{2, 3}

#### `set-intersection-helper(elems1, s2, idx, len, acc)`

Helper: accumulate elements of elems1 that are also in s2.

#### `set-difference(s1, s2)`

Return the difference of two sets.

Contains elements in s1 but not in s2.

Examples:
  set-difference(#{1, 2, 3}, #{2, 3, 4})  // => #{1}

#### `set-difference-helper(elems1, s2, idx, len, acc)`

Helper: accumulate elements of elems1 that are not in s2.

#### `set-subset?(s1, s2)`

Check if s1 is a subset of s2.

Returns true if all elements of s1 are also in s2.

Examples:
  set-subset?(#{1, 2}, #{1, 2, 3})  // => true
  set-subset?(#{1, 4}, #{1, 2, 3})  // => false

#### `set-subset-helper(elems1, s2, idx, len)`

Helper: return true if every element of elems1 is in s2.

#### `into-set(coll)`

Convert a collection to a set.

Removes duplicates since sets only contain unique elements.

Examples:
  into-set([1, 2, 2, 3, 3, 3])  // => #{1, 2, 3}

#### `vec(coll)`

Materialize any collection into a persistent vector (identity on vectors).

Examples:
  vec(#{1, 2, 3})  // => [1, 2, 3]

#### `subvec(coll, start, end)`

Return the sub-vector coll[start:end] (end exclusive) as a new vector.

Examples:
  subvec([10, 20, 30, 40], 1, 3)  // => [20, 30]

#### `includes?(coll, x)`

Value membership: true if `x` is structurally equal to some element of `coll`.

Distinct from `contains?`, which tests keys/indices: `contains?([10,20], 1)`
asks "is index 1 valid", whereas `includes?([10,20], 20)` asks "is 20 an element".

Examples:
  includes?([10, 20, 30], 20)  // => true
  includes?([10, 20, 30], 1)   // => false

#### `union(a, b)`

Set union: a set containing every element of `a` and `b` (inputs may be any
collections; the result is always a set).

Examples:
  union(#{1, 2}, [2, 3])  // => #{1, 2, 3}

#### `intersection(a, b)`

Set intersection: a set of the elements present in both `a` and `b`.

Examples:
  intersection([1, 2, 3], [2, 3, 4])  // => #{2, 3}

#### `difference(a, b)`

Set difference: a set of the elements of `a` not present in `b`.

Examples:
  difference([1, 2, 3], [2])  // => #{1, 3}

#### `char-code(s)`

Get the Unicode code point of the first character in a string.

Examples:
  char-code("A")  // => 65
  char-code("a")  // => 97

#### `char-from-code(code)`

Create a single-character string from a Unicode code point.

Examples:
  char-from-code(65)  // => "A"
  char-from-code(97)  // => "a"

#### `ok(value)`

Create a successful Result containing a value.

Examples:
  ok(42)  // => Result.Ok { value: 42 }

#### `err(error)`

Create an error Result containing an Error.

Examples:
  err(Error.NotFound { path: "/missing" })

#### `err-io(message)`

Create an Err Result wrapping an Error.IO with the given message.

#### `err-code(code, message)`

Create an Err Result wrapping an Error.Other with code and message.

#### `ok?(result)`

Check if a Result is successful (Ok variant).

Examples:
  ok?(ok(42))                      // => true
  ok?(err(Error.IO { message: "failed" }))  // => false

#### `unwrap(result)`

Unwrap a Result, returning the value or null on error.

Prints an error message if the result is an Err.

Examples:
  unwrap(ok(42))  // => 42
  unwrap(err(Error.NotFound { path: "/x" }))  // prints error, returns null

#### `unwrap-or(result, default)`

Unwrap a Result, returning the value or a default on error.

Examples:
  unwrap-or(ok(42), 0)                        // => 42
  unwrap-or(err(Error.IO { message: "" }), 0)  // => 0

#### `timeout?(result)`

Check whether a Result is an Err carrying a Timeout error.

#### `not-found?(result)`

Check whether a Result is an Err carrying a NotFound error.

#### `get-error(result)`

Return the error from an Err Result, or null if it is Ok.

#### `doc(value)`

Get documentation for a function or type.

Returns the docstring associated with the value, or null if none exists.
Works with functions, structs, enums, and other documented values.

Examples:
  doc(map)       ; => "Apply a function to each element..."
  doc(Result)    ; => documentation for Result enum

#### `apropos(query)`

Search for functions by name or docstring.

Returns a list of fully-qualified function names that match the query.
The query is matched against both function names and their docstrings.

Examples:
  apropos("file")    ; => ["beagle.fs/read-file", "beagle.fs/write-file", ...]
  apropos("reduce")  ; => ["beagle.core/reduce", "beagle.core/reduce-right"]

#### `assert-throws!(thunk)`

Assert that calling the given function throws an exception.
Throws AssertionError if no exception is thrown.

Examples:
  assert-throws!(fn() { throw "boom" })   ; passes
  assert-throws!(fn() { 42 })              ; fails - no exception thrown

<details>
<summary><strong>Undocumented functions (170)</strong></summary>

- `event-loop-create()`
- `event-loop-create-threaded()`
- `event-loop-run-once()`
- `event-loop-wake()`
- `event-loop-destroy()`
- `tcp-connect-async()`
- `tcp-listen()`
- `tcp-accept-async()`
- `tcp-read-async()`
- `tcp-write-async()`
- `tcp-close()`
- `tcp-close-listener()`
- `tcp-results-count()`
- `tcp-result-pop()`
- `tcp-result-pop-for-atom()`
- `tcp-result-pop-for-op-id()`
- `tcp-result-future-atom()`
- `tcp-result-value()`
- `tcp-result-value-for-op-id()`
- `tcp-result-data()`
- `tcp-result-data-for-op-id()`
- `tcp-result-op-id()`
- `tcp-result-listener-id()`
- `timer-set()`
- `timer-cancel()`
- `timer-completed-count()`
- `timer-pop-completed()`
- `timer-take-completed()`
- `file-read-submit()`
- `file-write-submit()`
- `file-delete-submit()`
- `file-stat-submit()`
- `file-readdir-submit()`
- `file-append-submit()`
- `file-exists-submit()`
- `file-rename-submit()`
- `file-copy-submit()`
- `file-mkdir-submit()`
- `file-mkdir-all-submit()`
- `file-rmdir-submit()`
- `file-rmdir-all-submit()`
- `file-is-dir-submit()`
- `file-is-file-submit()`
- `file-open-submit()`
- `file-close-submit()`
- `file-handle-read-submit()`
- `file-handle-write-submit()`
- `file-handle-readline-submit()`
- `file-handle-flush-submit()`
- `file-results-count()`
- `file-result-ready()`
- `file-result-poll-type()`
- `file-result-get-string()`
- `file-result-get-value()`
- `file-result-consume()`
- `file-result-get-entries()`
- `string-concat()`
- `string-builder-new()`
- `string-builder-length()`
- `string-builder-capacity()`
- `string-builder-clear!()`
- `string-builder-push-byte!()`
- `string-builder-push-string!()`
- `string-builder-push-string-range!()`
- `string-builder-push-string-range-filter-byte!()`
- `string-builder-push-string-range-uppercase!()`
- `string-builder-push-builder-range!()`
- `string-index-byte()`
- `string-builder-push-int!()`
- `string-builder-push-float!()`
- `string-builder-byte-at()`
- `string-builder-set-byte-at!()`
- `string-builder-reverse!()`
- `string-builder-to-string()`
- `read-full-file()`
- `write-full-file()`
- `fs-unlink()`
- `fs-access()`
- `fs-mkdir()`
- `fs-rmdir()`
- `fs-rename()`
- `fs-is-directory?()`
- `fs-is-file?()`
- `fs-readdir()`
- `fs-file-size()`
- `future-wait()`
- `future-notify()`
- `diagnostics()`
- `diagnostics-for-file()`
- `files-with-diagnostics()`
- `clear-diagnostics()`
- `get$2(coll, index)`
- `get$3(coll, index, default)`
- `push(coll, value)`
- `conj(coll, value)`
- `compare-to(self, other)`
- `close(self)`
- `length(coll)`
- `format(self, depth)`
- `assoc(coll, key, value)`
- `keys(coll)`
- `seq(coll)`
- `first(seq)`
- `next(seq)`
- `contains?(coll, x)`
- `write(self, s)`
- `Struct_format(self, depth)`
- `$Indexed_get$2(coll, index)`
- `$Indexed_get$3(coll, index, default)`
- `Format_format(self, depth)`
- `String_format(self, depth)`
- `String_contains?(s, substr)`
- `PersistentVector_get$2(vec, i)`
- `PersistentVector_get$3(vec, i, default)`
- `PersistentVector_contains?(vec, index)`
- `PersistentSet_contains?(set, element)`
- `PersistentVector_length(vec)`
- `PersistentVector_push(vec, value)`
- `PersistentVector_format(self, depth)`
- `PersistentVector_assoc(vec, index, value)`
- `PersistentMap_get$2(m, key)`
- `PersistentMap_get$3(m, key, default)`
- `PersistentMap_contains?(map, key)`
- `PersistentMap_length(m)`
- `PersistentMap_assoc(m, key, value)`
- `PersistentMap_keys(map)`
- `PersistentMap_format(map, depth)`
- `PersistentVector_conj(vec, value)`
- `PersistentSet_conj(set, value)`
- `PersistentMap_conj(map, kv)`
- `String_get$2(str, i)`
- `String_get$3(str, i, default)`
- `String_length(str)`
- `Array_get$2(arr, i)`
- `Array_get$3(arr, i, default)`
- `Array_length(arr)`
- `Array_seq(arr)`
- `ArraySeq_first(seq)`
- `ArraySeq_next(seq)`
- `ArraySeq_seq(s)`
- `ArraySeq_length(s)`
- `PersistentVector_seq(vec)`
- `PersistentVectorSeq_first(seq)`
- `PersistentVectorSeq_next(seq)`
- `PersistentVectorSeq_seq(s)`
- `PersistentVectorSeq_length(s)`
- `Stdout_write(self, s)`
- `StringBuffer_write(self, s)`
- `Range_seq(r)`
- `Range_first(r)`
- `Range_next(r)`
- `Range_length(r)`
- `String_seq(str)`
- `StringSeq_first(seq)`
- `StringSeq_next(seq)`
- `StringSeq_seq(s)`
- `StringSeq_length(s)`
- `PersistentSet_length(set)`
- `PersistentSet_format(set, depth)`
- `PersistentSetSeq_first(seq)`
- `PersistentSetSeq_next(seq)`
- `PersistentSetSeq_seq(s)`
- `PersistentSetSeq_length(s)`
- `PersistentSet_seq(set)`
- `PersistentMapSeq_first(seq)`
- `PersistentMapSeq_next(seq)`
- `PersistentMapSeq_seq(s)`
- `PersistentMapSeq_length(s)`
- `PersistentMap_seq(map)`
- `Regex_format(regex, depth)`

</details>

---

## beagle.fs

> **Documentation coverage:** 58/59 functions (98%)

#### `read-file(path)`

Read the entire contents of a file as a string.

Must be called within an Fs effect handler block.
Returns Result.Ok with content or Result.Err on failure.

Examples:
  handle effect/Handler(Fs) with handler {
      match fs/read-file("/tmp/test.txt") {
          Result.Ok { value } => println(value),
          Result.Err { error } => println("Error:", error)
      }
  }

#### `write-file(path, content)`

Write content to a file (creates or overwrites).

Returns Result.Ok with bytes written, or Result.Err on failure.

Examples:
  fs/write-file("/tmp/test.txt", "Hello, World!")

#### `append-file(path, content)`

Append content to a file (creates if doesn't exist).

Returns Result.Ok with bytes written, or Result.Err on failure.

Examples:
  fs/append-file("/tmp/log.txt", "New log entry\n")

#### `delete-file(path)`

Delete a file.

Returns Result.Ok on success, or Result.Err if file doesn't exist.

Examples:
  fs/delete-file("/tmp/test.txt")

#### `exists?(path)`

Check if a file or directory exists. Returns a plain `Bool` (a `?`-predicate
must be directly usable in `if`); an error determining existence reads as false.

Examples:
  fs/exists?("/tmp/test.txt")  // => true

#### `file-size(path)`

Get file size in bytes.

Returns Result.Ok with the size, or Result.Err if file doesn't exist.

Examples:
  fs/file-size("/tmp/test.txt")  // => Result.Ok { value: 1024 }

#### `is-file?(path)`

Check if path is a regular file (not a directory). Returns a plain `Bool`;
a missing path or stat error reads as false.

Examples:
  fs/is-file?("/tmp/test.txt")  // => true

#### `rename(old-path, new-path)`

Rename or move a file or directory.

Returns Result.Ok on success, or Result.Err on failure.

Examples:
  fs/rename("/tmp/old.txt", "/tmp/new.txt")

#### `copy(src-path, dest-path)`

Copy a file from source to destination.

Returns Result.Ok on success, or Result.Err on failure.

Examples:
  fs/copy("/tmp/source.txt", "/tmp/dest.txt")

#### `read-dir(path)`

List directory contents.

Returns Result.Ok with vector of entry names, or Result.Err on failure.

Examples:
  fs/read-dir("/tmp")  // => Result.Ok { value: ["file1.txt", "file2.txt"] }

#### `create-dir(path)`

Create a directory.

Parent directories must exist. Use create-dir-all for recursive creation.

Examples:
  fs/create-dir("/tmp/mydir")

#### `create-dir-all(path)`

Create a directory and all parent directories.

Like mkdir -p. Creates any missing parent directories.

Examples:
  fs/create-dir-all("/tmp/a/b/c")

#### `remove-dir(path)`

Remove an empty directory.

Fails if directory is not empty. Use remove-dir-all for recursive removal.

Examples:
  fs/remove-dir("/tmp/empty-dir")

#### `remove-dir-all(path)`

Remove a directory and all its contents recursively.

Like rm -rf. Removes files and subdirectories.

Examples:
  fs/remove-dir-all("/tmp/mydir")

#### `is-directory?(path)`

Check if path is a directory. Returns a plain `Bool`; a missing path or stat
error reads as false.

Examples:
  fs/is-directory?("/tmp")  // => true

#### `open(path, mode)`

Open a file with a mode string.

Modes: "r" (read), "w" (write/create), "a" (append), "r+" (read/write).
Returns a File handle for low-level operations.

Examples:
  let file = unwrap(fs/open("/tmp/test.txt", "r"))

#### `close(file)`

Close a file handle.

Always close files when done to release resources.

Examples:
  fs/close(file)

#### `read(file, n)`

Read up to n bytes from a file handle.

Returns the data read as a string.

Examples:
  fs/read(file, 1024)

#### `write(file, content)`

Write content to a file handle.

Returns the number of bytes written.

Examples:
  fs/write(file, "Hello, World!")

#### `read-line(file)`

Read a line from a file handle.

Reads until newline or end of file.

Examples:
  fs/read-line(file)  // => "First line\n"

#### `flush(file)`

Flush a file's buffers to disk.

Ensures all buffered writes are actually written.

Examples:
  fs/flush(file)

#### `handle-read-file(path)`

Read a file's full contents, returning ok(content) or err(NotFound)
if reading fails. Backing implementation for the ReadFile operation.

#### `handle-write-file(path, content)`

Open the path in write mode, write content, and close it. Returns
ok(bytes-written) or err on open/write failure. Backs WriteFile.

#### `handle-append-file(path, content)`

Open the path in append mode, write content, and close it. Returns
ok(bytes-written) or err on open/write failure. Backs AppendFile.

#### `handle-delete-file(path)`

Unlink the file at path. Returns ok(null) on success or err(NotFound)
if the unlink syscall fails. Backs DeleteFile.

#### `handle-file-exists(path)`

Test path existence via access(F_OK). Always returns ok(true/false).
Backs FileExists.

#### `handle-file-size(path)`

Return ok(size-in-bytes), or err(NotFound) when the size query
returns a negative value. Backs FileSize.

#### `handle-is-file(path)`

Return ok(true) when path is a regular file, ok(false) otherwise.
Backs IsFile.

#### `handle-is-directory(path)`

Return ok(true) when path is a directory, ok(false) otherwise.
Backs IsDirectory.

#### `handle-rename-file(old-path, new-path)`

Rename old-path to new-path. Returns ok(null) on success or err(IO)
with a descriptive message on failure. Backs RenameFile.

#### `handle-copy-file(src-path, dest-path)`

Copy src-path to dest-path by reading the source fully then writing
it out. Returns the write Result, or err if the read fails. Backs
CopyFile.

#### `handle-read-dir(path)`

List directory entries. Returns ok(vector-of-names), or err(NotFound)
when the path cannot be read. Backs ReadDir.

#### `handle-create-dir(path)`

Create a single directory with mode 0755. Returns ok(null), or
err(AlreadyExists) on EEXIST (-17), or err(IO) otherwise. Backs
CreateDir.

#### `handle-create-dir-all(path)`

Create path and all missing parent directories (like mkdir -p),
splitting on "/" and creating each component with mode 0755. Treats
EEXIST as success; returns ok(null) or the first err(IO). Backs
CreateDirAll.

#### `handle-remove-dir(path)`

Remove an empty directory. Returns ok(null) on success or err(IO) on
failure (e.g. when the directory is not empty). Backs RemoveDir.

#### `handle-remove-dir-all(path)`

Recursively remove a directory and its contents (like rm -rf). If
path is not a directory it is unlinked as a file; otherwise each
entry is unlinked or recursively removed before the now-empty
directory is removed. Returns ok(null) or the first err(IO). Backs
RemoveDirAll.

#### `handle-open(path, mode)`

Open path with the given mode and return ok(file-handle) or err.
Backs the Open operation.

#### `handle-close(file)`

Close a file handle. Returns ok(null) on success or err. Backs Close.

#### `handle-read(file, n)`

Read up to n bytes from a file handle, decode them into a string and
free the temporary buffer. Returns ok(string) or err. Backs Read.

#### `handle-write(file, content)`

Write content to a file handle. Returns ok(bytes-written) or err.
Backs Write.

#### `handle-read-line(file)`

Read a line (up to a 4096-byte buffer) from a file handle. Returns
ok(line) or err. Backs ReadLine.

#### `handle-flush(file)`

Flush a file handle's buffered writes to disk. Returns ok(null) or
err. Backs Flush.

#### `run-blocking(thunk)`

Run thunk with a fresh BlockingFsHandler installed, so any Fs effects
it performs execute synchronously. Returns the thunk's value. Used by
all the blocking-* convenience wrappers.

#### `blocking-read-file(path)`

Read the entire contents of a file synchronously.

This is the simplest way to read a file. No handler required.

Examples:
  match fs/blocking-read-file("/tmp/test.txt") {
      Result.Ok { value } => println(value),
      Result.Err { error } => println("Error:", error)
  }

#### `blocking-write-file(path, content)`

Write content to a file synchronously (creates or overwrites).

This is the simplest way to write a file. No handler required.

Examples:
  fs/blocking-write-file("/tmp/test.txt", "Hello, World!")

#### `blocking-append-file(path, content)`

Append content to a file synchronously (creates it if missing).
Returns Result.Ok with bytes written. No handler required.

#### `blocking-delete-file(path)`

Delete a file synchronously.

Examples:
  fs/blocking-delete-file("/tmp/test.txt")

#### `blocking-exists?(path)`

Check if a file or directory exists synchronously. Returns a plain `Bool`.

Examples:
  fs/blocking-exists?("/tmp/test.txt")  // => true

#### `blocking-file-size(path)`

Get a file's size in bytes synchronously. Returns Result.Ok with the
size or Result.Err if it doesn't exist. No handler required.

#### `blocking-is-file?(path)`

Check synchronously whether path is a regular file. Returns a plain `Bool`.
No handler required.

#### `blocking-is-directory?(path)`

Check synchronously whether path is a directory. Returns a plain `Bool`.
No handler required.

#### `blocking-rename(old-path, new-path)`

Rename or move a file/directory synchronously. Returns Result.Ok on
success or Result.Err on failure. No handler required.

#### `blocking-copy(src-path, dest-path)`

Copy a file from source to destination synchronously. Returns
Result.Ok on success or Result.Err on failure. No handler required.

#### `blocking-read-dir(path)`

List directory contents synchronously.

Returns a Result containing a vector of filenames.

Examples:
  fs/blocking-read-dir("/tmp")  // => Result.Ok { value: ["file1.txt", "file2.txt"] }

#### `blocking-create-dir(path)`

Create a directory synchronously.

Examples:
  fs/blocking-create-dir("/tmp/mydir")

#### `blocking-create-dir-all(path)`

Create a directory and all missing parents synchronously (mkdir -p).
No handler required.

#### `blocking-remove-dir(path)`

Remove an empty directory synchronously. Fails if not empty. No
handler required.

#### `blocking-remove-dir-all(path)`

Remove a directory and all its contents recursively, synchronously
(rm -rf). No handler required.

<details>
<summary><strong>Undocumented functions (1)</strong></summary>

- `BlockingFsHandler_handle(self, op, resume)`

</details>

---

## beagle.async

> **Documentation coverage:** 149/152 functions (98%)

#### `make-future(initial_state)`

Create a new future with an initial state.

Futures represent values that may not be available yet.

Examples:
  let f = make-future(FutureState.Pending {})

#### `future-state(future)`

Get the current state of a future.

Returns a FutureState: Pending, Running, Resolved, Rejected, or Cancelled.

#### `resolve-future!(future, value)`

Resolve a future with a successful value.

Notifies any threads waiting on this future.

#### `reject-future!(future, error)`

Reject a future with an error.

Notifies any threads waiting on this future.

#### `cancel-future!(future)`

Cancel a future.

The future's state becomes Cancelled. Notifies any waiting threads.

#### `future-cancelled?(future)`

Check if a future has been cancelled.

#### `make-cancellation-token()`

Create a new cancellation token.

Cancellation tokens provide cooperative cancellation for async tasks.

#### `cancel!(token)`

Cancel a cancellation token.

All tasks watching this token should stop.

#### `cancelled?(token)`

Check if a cancellation token has been cancelled.

#### `make-scope()`

Create a new task scope with a fresh cancellation token and no children.

#### `scope-add-child!(scope, future)`

Register a child future with a scope so the scope tracks and can cancel it.

#### `scope-children(scope)`

Return the vector of child futures registered with a scope.

#### `scope-cancel-all!(scope)`

Cancel a scope's token and every child future registered with it.

#### `scope-await-all-children(scope)`

Block until every child future of a scope reaches a terminal state (resolved, rejected, or cancelled).

#### `with-scope(body)`

Run a block with a task scope for structured concurrency.

Ensures all spawned tasks complete before the scope exits.
On error or cancellation, all child tasks are cancelled.

Examples:
  with-scope(fn(scope) {
      let f1 = spawn-in-scope(scope, fn() { compute1() })
      let f2 = spawn-in-scope(scope, fn() { compute2() })
      [await(f1), await(f2)]
  })

#### `async-ok(value)`

Wrap a value as a successful Result (beagle.core/ok).

#### `async-err(code, message)`

Wrap a message as a failed Result carrying Error.IO. The code argument is currently ignored.

#### `async-ok?(result)`

Return true if a Result is Ok, false if it is an Err.

#### `async-unwrap(result)`

Return the value of an Ok Result, or throw the contained error if it is an Err.

#### `async-unwrap-or(result, default)`

Return the value of an Ok Result, or the given default if it is an Err.

#### `handle-read-file(path)`

Read a file's full contents synchronously; returns ok(content) or err(Error.NotFound) on failure.

#### `handle-write-file(path, content)`

Open path for writing, write content, and close; returns ok(bytes) or err(error).

#### `handle-append-file(path, content)`

Open path in append mode, write content, and close; returns ok(bytes) or err(error).

#### `handle-delete-file(path)`

Unlink a file synchronously; returns ok(null) or err(Error.NotFound).

#### `handle-file-exists(path)`

Return ok(true/false) for whether path exists (via an F_OK access check).

#### `handle-file-size(path)`

Return the file size in bytes as an ok Result, or an err if the stat fails.

#### `handle-is-file(path)`

Return ok(true/false) for whether path is a regular file.

#### `handle-is-directory(path)`

Return ok(true/false) for whether path is a directory.

#### `handle-rename-file(old-path, new-path)`

Rename/move old-path to new-path; returns ok(null) or an err Result on failure.

#### `handle-copy-file(src-path, dest-path)`

Copy src-path to dest-path by reading then writing; returns the write Result or the read err.

#### `handle-read-dir(path)`

List directory entries; returns ok(entries) or err(Error.NotFound).

#### `handle-create-dir(path)`

Create a single directory (mode 0755); returns ok(null), err(AlreadyExists), or err(IO).

#### `handle-create-dir-all(path)`

Create path and all missing parent directories; returns ok(null) or the first err encountered.

#### `handle-remove-dir(path)`

Remove an empty directory; returns ok(null) or an err Result.

#### `handle-remove-dir-all(path)`

Recursively remove a file or directory tree; returns ok(null) or the first err encountered.

#### `handle-open(path, mode)`

Open a file with the given mode; returns ok(file handle) or err(error).

#### `handle-close(file)`

Close a file handle; returns ok(null) or err(error).

#### `handle-read(file, n)`

Read n bytes from a file, decode as a string, and free the buffer; returns ok(string) or err(error).

#### `handle-write(file, content)`

Write a string to a file; returns ok(bytes written) or err(error).

#### `handle-read-line(file)`

Read a line (up to 4096 bytes) from a file; returns ok(line) or err(error).

#### `handle-flush(file)`

Flush a file's buffers; returns ok(null) or err(error).

#### `handle-sleep(ms)`

Sleep synchronously for ms milliseconds via core/sleep; returns ok(null).

#### `handle-await-blocking(future)`

Await a future in blocking mode: returns its value, throws its error or cancellation, or throws if it is still pending/running (futures must already be resolved). A null future returns null.

#### `handle-await-all-blocking(futures)`

Await every future in order (blocking) and return their values as a vector.

#### `handle-await-first-blocking(futures)`

Return the first resolved future's value as a RaceResult.Ok, or AllFailed with the collected errors. Assumes futures are already resolved.

#### `handle-cancel(future)`

Cancel a future and return null.

#### `handle-spawn-blocking(thunk)`

Run thunk synchronously, returning a future already resolved with its result (or rejected on throw).

#### `handle-spawn-with-token-blocking(thunk, token)`

Run thunk synchronously unless the token is cancelled, returning a resolved, rejected, or cancelled future.

#### `handle-spawn-threaded(thunk)`

Spawn an OS thread to run thunk, returning a future the thread resolves or rejects on completion.

#### `handle-spawn-with-token-threaded(thunk, token)`

Spawn an OS thread to run thunk, honoring the cancellation token before and after; returns its future.

#### `handle-await-threaded(future)`

Await a future by polling until it resolves (or return null for a null future).

#### `poll-until-resolved(future)`

Block on a future, waiting on the condition variable (up to 50ms per round), until it resolves (returns the value), rejects, or is cancelled (throws).

#### `handle-await-all-threaded(futures)`

Await every future in order and return their values as a vector.

#### `handle-await-first-threaded(futures)`

Poll all futures until one resolves (RaceResult.Ok) or all fail (RaceResult.AllFailed with errors).

#### `handle-sleep-event-loop(loop_id, ms)`

Sleep using an event-loop timer: set a timer and drive the loop until this sleeper's marker completes. Returns ok(null).

#### `poll-file-result(loop_id, handle)`

Poll an async file operation by handle, decoding its result-type code into an ok/err Result, or null if it is not ready yet.

#### `wait-for-file-result(loop_id, handle)`

Drive the event loop, blocking, until the file operation for handle completes, then return its Result.

#### `handle-read-file-async(loop_id, path)`

Submit an async file read and block (via the event loop) until it completes. Throws if submission fails.

#### `handle-write-file-async(loop_id, path, content)`

Submit an async file write and block until it completes. Throws if submission fails.

#### `handle-delete-file-async(loop_id, path)`

Submit an async file delete and block until it completes. Throws if submission fails.

#### `handle-file-size-async(loop_id, path)`

Submit an async stat and block until it completes, returning the file size Result. Throws if submission fails.

#### `handle-read-dir-async(loop_id, path)`

Submit an async directory listing and block until it completes. Throws if submission fails.

#### `handle-append-file-async(loop_id, path, content)`

Submit an async file append and block until it completes. Throws if submission fails.

#### `handle-file-exists-async(loop_id, path)`

Submit an async existence check and block until it completes. Throws if submission fails.

#### `handle-rename-file-async(loop_id, old_path, new_path)`

Submit an async rename and block until it completes. Throws if submission fails.

#### `handle-copy-file-async(loop_id, src_path, dest_path)`

Submit an async file copy and block until it completes. Throws if submission fails.

#### `handle-create-dir-async(loop_id, path)`

Submit an async mkdir and block until it completes. Throws if submission fails.

#### `handle-create-dir-all-async(loop_id, path)`

Submit an async recursive mkdir and block until it completes. Throws if submission fails.

#### `handle-remove-dir-async(loop_id, path)`

Submit an async rmdir and block until it completes. Throws if submission fails.

#### `handle-remove-dir-all-async(loop_id, path)`

Submit an async recursive rmdir and block until it completes. Throws if submission fails.

#### `handle-is-directory-async(loop_id, path)`

Submit an async is-directory check and block until it completes. Throws if submission fails.

#### `handle-is-file-async(loop_id, path)`

Submit an async is-file check and block until it completes. Throws if submission fails.

#### `handle-open-async(loop_id, path, mode)`

Submit an async file open and block until it completes, returning a handle-key Result. Throws if submission fails.

#### `handle-close-async(loop_id, handle_key)`

Submit an async close for a file handle-key and block until it completes. Throws if submission fails.

#### `handle-handle-read-async(loop_id, handle_key, n)`

Submit an async read of n bytes on a file handle-key and block until it completes. Throws if submission fails.

#### `handle-handle-write-async(loop_id, handle_key, content)`

Submit an async write of content on a file handle-key and block until it completes. Throws if submission fails.

#### `handle-handle-readline-async(loop_id, handle_key)`

Submit an async read-line on a file handle-key and block until it completes. Throws if submission fails.

#### `handle-handle-flush-async(loop_id, handle_key)`

Submit an async flush on a file handle-key and block until it completes. Throws if submission fails.

#### `read-file(path)`

Read the entire contents of a file as a string. Performs Async.ReadFile; returns a Result.

#### `write-file(path, content)`

Write content to a file, creating or overwriting it. Performs Async.WriteFile; returns a Result.

#### `append-file(path, content)`

Append content to a file, creating it if absent. Performs Async.AppendFile; returns a Result.

#### `delete-file(path)`

Delete a file. Performs Async.DeleteFile; returns a Result.

#### `file-exists?(path)`

Check whether a file exists. Performs Async.FileExists; returns ok(true/false).

#### `read-dir(path)`

List a directory's entries. Performs Async.ReadDir; returns a Result.

#### `create-dir(path)`

Create a directory. Performs Async.CreateDir; returns a Result.

#### `create-dir-all(path)`

Create a directory and all missing parents. Performs Async.CreateDirAll; returns a Result.

#### `remove-dir(path)`

Remove an empty directory. Performs Async.RemoveDir; returns a Result.

#### `remove-dir-all(path)`

Remove a directory and all its contents. Performs Async.RemoveDirAll; returns a Result.

#### `file-size(path)`

Get a file's size in bytes. Performs Async.FileSize; returns a Result.

#### `is-directory?(path)`

Check whether a path is a directory. Performs Async.IsDirectory; returns ok(true/false).

#### `is-file?(path)`

Check whether a path is a regular file. Performs Async.IsFile; returns ok(true/false).

#### `rename-file(old-path, new-path)`

Rename or move a file or directory. Performs Async.RenameFile; returns a Result.

#### `copy-file(src-path, dest-path)`

Copy a file. Performs Async.CopyFile; returns a Result.

#### `open(path, mode)`

Open a file with the given mode ("r", "w", "a", ...). Performs Async.Open; returns a Result with a file handle.

#### `close(file)`

Close a file handle. Performs Async.Close; returns a Result.

#### `read(file, n)`

Read n bytes from a file. Performs Async.Read; returns a Result with the data.

#### `write(file, content)`

Write content to a file. Performs Async.Write; returns a Result.

#### `read-line(file)`

Read a single line from a file. Performs Async.ReadLine; returns a Result.

#### `flush(file)`

Flush a file's buffers. Performs Async.Flush; returns a Result.

#### `sleep(ms)`

Sleep for a number of milliseconds.

Must be called within an Async effect handler.

Examples:
  sleep(1000)  // Sleep for 1 second

#### `await(future)`

Wait for a future to complete and return its value.

Blocks until the future resolves, rejects, or is cancelled.

Examples:
  let f = spawn(fn() { compute() })
  await(f)  // => result of compute()

#### `await-all(futures)`

Wait for all futures to complete and return their values as a vector.

Cooperative: awaits each future in turn. The futures were all spawned up
front, so they make progress concurrently on the scheduler while we collect
their results in order.

Examples:
  let fs = [spawn(fn() { 1 }), spawn(fn() { 2 })]
  await-all(fs)  // => [1, 2]

#### `await-first(futures)`

Wait for the first future to complete and return its result as a RaceResult.

Cooperative: polls the futures, yielding to the scheduler (via a 0ms sleep)
between rounds so the spawned tasks can run, until one resolves or all fail.

#### `await-timeout(ms, future)`

Wait for a future with a timeout in milliseconds.

Returns TimeoutResult.Ok if completed, TimeoutResult.TimedOut if timeout.

#### `async-with-token(token, thunk)`

Spawn thunk as a task governed by a cancellation token, returning its future. Performs Async.SpawnWithToken.

#### `cancel(future)`

Request cooperative cancellation of a future. Performs Async.Cancel.

#### `spawn-in-scope(scope, thunk)`

Spawn thunk as a child of a scope (using the scope's token) and register its future with the scope.

#### `with-timeout(ms, thunk)`

Run thunk synchronously and report whether it finished within ms milliseconds. Does NOT interrupt thunk; it only measures elapsed time, returning TimeoutResult.Ok or TimedOut.

#### `async-with-timeout(ms, thunk)`

Spawn thunk with a fresh cancellation token and await it up to ms milliseconds, cancelling and returning TimedOut on timeout, else Ok with the value.

#### `async-sleep(ms)`

Return a future that resolves to null after ms milliseconds (spawns a task that sleeps).

#### `await-sleep(ms)`

Sleep for ms milliseconds by spawning and awaiting an async sleep.

#### `time-now()`

Return the current time in nanoseconds (thin wrapper over core/time-now).

#### `spawn(thunk)`

Spawn a cooperative task and return a future.

The task runs concurrently with the caller — but on the SAME thread, via
the cooperative scheduler. It makes progress whenever the caller (or any
other task) parks on an async op. `spawn` never starts an OS thread; use
`thread()` for real parallelism. `await` the future to get the result.

Examples:
  let f = spawn(fn() { compute() })
  await(f)  // => result of compute()

#### `read-file!(path)`

Read a file and unwrap the Result, throwing on error. Returns the contents string.

#### `write-file!(path, content)`

Write a file and unwrap the Result, throwing on error.

#### `ok?(result)`

Return true if a Result is Ok (alias for async-ok?).

#### `unwrap-or(result, default)`

Return the Ok value of a Result, or default on error (alias for async-unwrap-or).

#### `default-handler()`

Create the default async handler instance (a BlockingAsyncHandler).

#### `run-blocking(thunk)`

Run thunk with the BlockingAsyncHandler installed for the Async effect.

#### `blocking-read-file(path)`

Run read-file under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-write-file(path, content)`

Run write-file under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-append-file(path, content)`

Run append-file under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-delete-file(path)`

Run delete-file under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-file-exists?(path)`

Run file-exists? under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-read-dir(path)`

Run read-dir under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-create-dir(path)`

Run create-dir under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-remove-dir(path)`

Run remove-dir under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-file-size(path)`

Run file-size under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-is-directory?(path)`

Run is-directory? under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-is-file?(path)`

Run is-file? under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-rename-file(old-path, new-path)`

Run rename-file under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-copy-file(src-path, dest-path)`

Run copy-file under the blocking handler. Convenience for a single synchronous operation.

#### `blocking-sleep(ms)`

Run sleep under the blocking handler. Convenience for a single synchronous operation.

#### `run-all-thunks(thunks)`

Call each thunk in order and collect their return values into a vector.

#### `await-any(thunks)`

Call each thunk in order, returning the first whose Result is Ok, or null if none succeed.

#### `map-results(results, f)`

Map f over the Ok values of a sequence of Results, leaving Err results unchanged.

#### `filter-ok(results)`

Return only the Ok Results from a sequence.

#### `unwrap-all(results)`

Return the unwrapped values of all Ok Results in a sequence (Err results are dropped).

#### `decode-tcp-result(result_type, loop_id, op_id)`

Decode a completed TCP result (by its numeric type) into a value: a TcpSocket, read data, bytes written, or a TcpIoError sentinel for failures.

#### `tcp-result-or-throw(result)`

Re-raise a TcpIoError sentinel as a thrown error, or pass any other value through unchanged.

#### `get-io-loop()`

Return the shared global event loop, lazily creating a 4-worker threaded loop on first use (CAS-guarded against races).

#### `get-io-loop-threaded()`

Return the shared global event loop (thin alias for get-io-loop).

#### `stash-task(tasks_atom, op, resume)`

Park a cooperative task: push a SchedulerTask (its pending op and continuation) onto the handler's task atom. Returns null.

#### `spawn-cooperative-task(tasks_atom, thunk)`

Spawn thunk as a cooperative task (no OS thread) on the current scheduler and return its future; the scheduler interleaves it with other work.

#### `handle-io-action(action)`

Execute a TCP IOAction synchronously, blocking the current thread on the event loop until the operation completes. Throws on failure.

#### `create-implicit-handler()`

Create an ImplicitAsyncHandler instance (the default production handler).

#### `poll-tasks(tasks, loop_id)`

Poll every parked scheduler task once, returning [ready, remaining]: tasks whose op completed (paired with their result) and those still pending.

#### `scheduler-loop(handler)`

Run the cooperative scheduler: repeatedly poll parked tasks, resume ready ones (and start new tasks), driving the event loop until no tasks remain.

#### `run-cooperative(thunk)`

Run thunk under a fresh cooperative scheduler and return its value. The thunk is the root task; async ops park their continuation so other tasks progress on the same thread.

<details>
<summary><strong>Undocumented functions (3)</strong></summary>

- `BlockingAsyncHandler_handle(self, op, resume)`
- `CooperativeHandler_handle(self, op, resume)`
- `ImplicitAsyncHandler_handle(self, op, resume)`

</details>

---

## beagle.io

> **Documentation coverage:** 20/21 functions (95%)

#### `get-libc-path()`

Return the platform-specific path to the libc shared library.

Resolves "macos" and "linux"; falls back to "libc" on unknown platforms.

#### `io-err(msg)`

Wrap a message in a Result.Err carrying an Error.IO with that message.

#### `is-null(pointer)`

Return true if the given FFI pointer is null (its low word is 0).

#### `string-to-buffer(s)`

Allocate a native buffer and copy the bytes of string `s` into it.

Returns the buffer (caller is responsible for deallocating it).

#### `open(path, mode)`

Open a file with the specified mode.

Modes: "r" (read), "w" (write), "a" (append), "r+" (read/write).
Returns Result with a File handle or error.

Examples:
  match io/open("/tmp/test.txt", "r") {
      Result.Ok { value } => value,
      Result.Err { error } => println("Failed to open")
  }

#### `close(file)`

Close an open file handle.

#### `read-bytes(file, n)`

Read up to n bytes from a file.

Returns a BufferResult with buffer and actual length read.
Read up to n bytes from a file into a buffer.

Returns a BufferResult with the buffer and bytes read.

#### `read-char(file)`

Read a single character from a file.

Returns Result.Ok with the character code, or Result.Err at EOF.

#### `write-bytes(file, buf, len)`

Write bytes from a buffer to a file.

Returns the number of bytes written.

#### `write-string(file, s)`

Write a string to a file.

Returns the number of bytes written.

#### `flush(file)`

Flush a file's buffers to disk.

#### `eof?(file)`

Check if a file has reached end-of-file.

#### `read-line(file, max-len)`

Read a line from a file (up to newline or max-len bytes).

Returns Result with the line as a string.

#### `read-stdin(n)`

Read up to n bytes from standard input.

Returns a BufferResult with the buffer and bytes read.

#### `read-stdin-string(n)`

Read up to n bytes from stdin and return as a string.

Convenience wrapper around read-stdin.

#### `write-stdout(s)`

Write a string to standard output.

Returns the number of bytes written.

#### `write-stdout-buffer(buf, len)`

Write raw buffer bytes to standard output.

More efficient than write-stdout for binary data.

#### `write-stderr(s)`

Write a string to standard error.

Returns the number of bytes written.

#### `write-stdout-buffer-offset(buf, offset, len)`

Write part of a buffer to stdout starting at an offset.

Useful for efficiently writing portions of a buffer.

#### `read-stdin-line(max-len)`

Read a line from standard input (up to max-len bytes).

Reads until newline or max-len is reached.

<details>
<summary><strong>Undocumented functions (1)</strong></summary>

- `File_close(file)`

</details>

---

## beagle.timer

> **Documentation coverage:** 8/10 functions (80%)

#### `sleep(ms)`

Sleep for a number of milliseconds.

Must be called within a Timer effect handler block.

Examples:
  timer/sleep(100)  // => null

#### `now()`

Get current time in nanoseconds since epoch.

Examples:
  let start = timer/now()
  // ... do work ...
  let elapsed = timer/now() - start

#### `timeout(ms, future)`

Wrap a future with a timeout.

Returns Result.Ok if completed before timeout, or Result.Err with Timeout error.
Requires both Timer and Async handlers to be installed.

Examples:
  match timer/timeout(1000, my-future) {
      Result.Ok { value } => value,
      Result.Err { error } => "timed out"
  }

#### `deadline(ms)`

Create a deadline (absolute time in nanoseconds from now).

Useful for setting a single deadline across multiple operations.

Examples:
  let d = timer/deadline(5000)  // => 5000000000

#### `deadline-passed?(deadline-time)`

Check if a deadline has passed.

#### `deadline-remaining(deadline-time)`

Get remaining milliseconds until a deadline (0 if passed).

#### `blocking-sleep(ms)`

Sleep for a number of milliseconds synchronously.

No handler required. Blocks the current thread.

Examples:
  timer/blocking-sleep(100)  // => null

#### `blocking-now()`

Get current time in nanoseconds synchronously.

No handler required.

<details>
<summary><strong>Undocumented functions (2)</strong></summary>

- `run-blocking(thunk)`
- `BlockingTimerHandler_handle(self, op, resume)`

</details>

---

## beagle.regex

> **Documentation coverage:** 9/9 functions (100%)

#### `compile(pattern)` `builtin`

Compile a regular expression pattern.

Returns a Regex object that can be used with other regex functions.

Examples:
  (let re (regex/compile "[0-9]+"))

#### `matches?(regex, string)` `builtin`

Check if the entire string matches the regex.

Examples:
  (regex/matches? (regex/compile "[0-9]+") "123")  ; => true
  (regex/matches? (regex/compile "[0-9]+") "abc")  ; => false

#### `find(regex, string)` `builtin`

Find the first match in the string.

Returns a map with :start, :end, and :match keys, or null if no match.

Examples:
  (regex/find (regex/compile "[0-9]+") "abc123def")
  ; => {:start 3 :end 6 :match "123"}

#### `find-all(regex, string)` `builtin`

Find all matches in the string.

Returns a vector of match maps.

Examples:
  (regex/find-all (regex/compile "[0-9]+") "a1b2c3")
  ; => [{:start 1 :end 2 :match "1"} ...]

#### `replace(regex, string, replacement)` `builtin`

Replace the first match in the string with the replacement.

Examples:
  (regex/replace (regex/compile "[0-9]+") "a1b2c3" "X")
  ; => "aXb2c3"

#### `replace-all(regex, string, replacement)` `builtin`

Replace all matches in the string with the replacement.

Examples:
  (regex/replace-all (regex/compile "[0-9]+") "a1b2c3" "X")
  ; => "aXbXcX"

#### `split(regex, string)` `builtin`

Split a string by the regex pattern.

Returns a vector of strings.

Examples:
  (regex/split (regex/compile ",\\s*") "a, b, c")
  ; => ["a" "b" "c"]

#### `captures(regex, string)` `builtin`

Get capture groups from the first match.

Returns a vector of captured strings (index 0 is the full match), or null if no match.

Examples:
  (regex/captures (regex/compile "(\\w+)@(\\w+)") "user@host")
  ; => ["user@host" "user" "host"]

#### `is-regex?(value)` `builtin`

Check if a value is a compiled regex.

Examples:
  (regex/is-regex? (regex/compile "test"))  ; => true
  (regex/is-regex? "test")  ; => false

---

## beagle.reflect

> **Documentation coverage:** 22/22 functions (100%)

#### `type-of(value)` `builtin`

Get a type descriptor for any value.

Returns a type descriptor that can be used with other reflect functions.

Examples:
  (reflect/type-of 42)        ; => <type Int>
  (reflect/type-of "hello")   ; => <type String>

#### `kind(descriptor)` `builtin`

Get the type kind from a type descriptor.

Returns a keyword: :struct, :enum, :function, or :primitive.

Examples:
  (reflect/kind (reflect/type-of some-struct))  ; => :struct

#### `name(descriptor)` `builtin`

Get the type name from a type descriptor.

Returns the name as a string.

Examples:
  (reflect/name (reflect/type-of 42))  ; => "Int"

#### `doc(descriptor)` `builtin`

Get the docstring for a type or function.

Returns the documentation string or null if none available.

Examples:
  (reflect/doc (reflect/type-of some-fn))  ; => "Documentation..."
  (reflect/doc (reflect/type-of 42))  ; => null

#### `fields(descriptor)` `builtin`

Get the field names for a struct type.

Returns a vector of field names, or null for non-struct types.

Examples:
  (reflect/fields (reflect/type-of my-struct))  ; => ["field1" "field2"]

#### `variants(descriptor)` `builtin`

Get the variant names for an enum type.

Returns a vector of variant names, or null for non-enum types.

Examples:
  (reflect/variants (reflect/type-of Result.Ok))  ; => ["Ok" "Err"]

#### `args(descriptor)` `builtin`

Get the argument names for a function.

Returns a vector of argument names, or null for non-function types.

Examples:
  (reflect/args (reflect/type-of println))  ; => ["value"]

#### `variadic?(descriptor)` `builtin`

Check if a function accepts variable arguments.

Returns true if the function is variadic, false otherwise.

Examples:
  (reflect/variadic? (reflect/type-of +))  ; => true

#### `info(descriptor)` `builtin`

Get complete type information as a map.

Returns a map containing all available metadata about the type,
including kind, name, docstring, fields/variants/args as appropriate.

Examples:
  (reflect/info (reflect/type-of my-fn))
  ; => {:kind :function :name "my-fn" :args [...] ...}

#### `struct?(value)` `builtin`

Check if a value is a struct type or instance.

Examples:
  (reflect/struct? my-struct-instance)  ; => true
  (reflect/struct? 42)  ; => false

#### `enum?(value)` `builtin`

Check if a value is an enum type or variant.

Examples:
  (reflect/enum? Result.Ok)  ; => true
  (reflect/enum? 42)  ; => false

#### `function?(value)` `builtin`

Check if a value is a function.

Examples:
  (reflect/function? println)  ; => true
  (reflect/function? 42)  ; => false

#### `primitive?(value)` `builtin`

Check if a value is a primitive type (Int, Float, String, Bool, Null).

Examples:
  (reflect/primitive? 42)  ; => true
  (reflect/primitive? [1 2 3])  ; => false

#### `namespace-members(namespace-name)` `builtin`

List all members defined in a namespace.

Returns a vector of member names.

Examples:
  (reflect/namespace-members "beagle.core")  ; => ["println" "map" ...]

#### `all-namespaces()` `builtin`

List all namespace names in the runtime.

Returns a vector of namespace name strings.

Examples:
  (reflect/all-namespaces)  ; => ["beagle.core" "beagle.fs" ...]

#### `apropos(query)` `builtin`

Search for functions by name or docstring substring.

Returns a vector of matching function names.

Examples:
  (reflect/apropos "print")  ; => ["println" "print" ...]

#### `namespace-info(namespace-name)` `builtin`

Get detailed information about a namespace.

Returns a map with the namespace's functions, structs, and enums.

Examples:
  (reflect/namespace-info "beagle.core")
  ; => {:functions [...] :structs [...] :enums [...]}

#### `source(value)` `builtin`

Return the original source text for a definition.

Accepts a function value, struct/enum value, or type descriptor. Returns null if no source text is stored (builtins, foreign functions, or anonymous closures).

This is the round-trippable counterpart to `eval`: read source, edit it, and re-eval the new text to redefine the function in place.

Examples:
  (reflect/source greet)
  ; => "fn greet(name) {\n    println(\"Hello, \" + name)\n}"

#### `namespace-source(namespace-name)` `builtin`

Return the concatenated source text of every definition in a namespace.

Members without stored source (builtins, foreign fns) are skipped. Definitions are emitted in registration order: structs first, then enums, then functions. Returns null if the namespace has no members with source.

Examples:
  (reflect/namespace-source "my.module")
  ; => "struct Point { x, y }\n\nfn distance(a, b) { ... }"

#### `location(value)` `builtin`

Return the on-disk location of a definition as a map `{:file, :byte-start, :byte-end, :line-start, :line-end}`, or null.

Accepts the same kinds of values as `reflect/source`: function, struct/enum value or type descriptor. The byte range covers the full block as it lives on disk, including any preceding `///` doc comment lines. REPL/eval definitions, builtins, and foreign functions return null.

Pair with `reflect/write-source` to persist edits back to disk.

Examples:
  (reflect/location greet)
  ; => {:file "my.bg" :byte-start 120 :byte-end 200 :line-start 5 :line-end 8}

#### `write-source(value, new-text)` `builtin`

Persist an edited definition back to its source file and re-register it in the runtime.

Reads the file at the definition's on-disk location, verifies the bytes still match what was loaded (aborting if the file drifted), splices `new-text` in at the recorded byte range, writes the file, and re-compiles the new text in the definition's namespace so subsequent `reflect/source` returns it. Byte ranges of other definitions in the same file are shifted to account for the length change.

Throws a runtime error (kind `write-source`) when the value has no on-disk origin, the file has drifted, I/O fails, or the new text doesn't parse. Returns `true` on success.

Examples:
  (reflect/write-source greet "fn greet(name) { println(\"hi \" ++ name) }")

#### `persist(namespace, text)` `builtin`

Persist one or more top-level definitions to disk, dispatching each as an update or an append.

Parses `text` to discover top-level fn/struct/enum definitions. For each def, if `namespace/<name>` already has an on-disk location, it is spliced in place (drift-checked against the stored source text first); otherwise the fragment is appended to the namespace's source file. All drift checks run up-front, so either every update succeeds or nothing is written.

Returns a vector of maps `[{:name, :action}]` where `:action` is `"updated"` or `"appended"`. Throws on parse failure, drift, I/O error, or compile error.

Examples:
  (reflect/persist "my.module" "fn greet(name) { println(\"hi \" ++ name) }")
  ; => [{:name "my.module/greet" :action "appended"}]

---

## beagle.ansi

> **Documentation coverage:** 54/54 functions (100%)

#### `enabled?()`

True if ANSI escapes are currently emitted (false when NO_COLOR disabled them
or set-enabled!(false) was called).

#### `set-enabled!(on)`

Enable or disable ANSI escape emission at runtime. When disabled, every
styling function returns its text unchanged.

#### `wrap(code, s)`

Wrap `s` in a single SGR escape: ESC[<code>m s ESC[0m. Returns a new string.

#### `style(s, codes)`

Apply a vector of SGR codes as one combined escape; an empty `codes` returns `s` unchanged.

#### `bold(s)`

Render `s` bold (SGR 1).

#### `dim(s)`

Render `s` dim/faint (SGR 2).

#### `italic(s)`

Render `s` italic (SGR 3).

#### `underline(s)`

Underline `s` (SGR 4).

#### `blink(s)`

Render `s` blinking (SGR 5).

#### `reverse(s)`

Swap foreground/background for `s` (reverse video, SGR 7).

#### `hidden(s)`

Render `s` hidden/invisible (SGR 8).

#### `strikethrough(s)`

Render `s` with a strikethrough line (SGR 9).

#### `black(s)`

Black foreground (SGR 30).

#### `red(s)`

Red foreground (SGR 31).

#### `green(s)`

Green foreground (SGR 32).

#### `yellow(s)`

Yellow foreground (SGR 33).

#### `blue(s)`

Blue foreground (SGR 34).

#### `magenta(s)`

Magenta foreground (SGR 35).

#### `cyan(s)`

Cyan foreground (SGR 36).

#### `white(s)`

White foreground (SGR 37).

#### `bright-black(s)`

Bright black foreground (SGR 90).

#### `bright-red(s)`

Bright red foreground (SGR 91).

#### `bright-green(s)`

Bright green foreground (SGR 92).

#### `bright-yellow(s)`

Bright yellow foreground (SGR 93).

#### `bright-blue(s)`

Bright blue foreground (SGR 94).

#### `bright-magenta(s)`

Bright magenta foreground (SGR 95).

#### `bright-cyan(s)`

Bright cyan foreground (SGR 96).

#### `bright-white(s)`

Bright white foreground (SGR 97).

#### `gray(s)`

Alias for bright-black (SGR 90).

#### `grey(s)`

Alias for bright-black (SGR 90); British spelling of `gray`.

#### `bg-black(s)`

Black background (SGR 40).

#### `bg-red(s)`

Red background (SGR 41).

#### `bg-green(s)`

Green background (SGR 42).

#### `bg-yellow(s)`

Yellow background (SGR 43).

#### `bg-blue(s)`

Blue background (SGR 44).

#### `bg-magenta(s)`

Magenta background (SGR 45).

#### `bg-cyan(s)`

Cyan background (SGR 46).

#### `bg-white(s)`

White background (SGR 47).

#### `bg-bright-black(s)`

Bright black background (SGR 100).

#### `bg-bright-red(s)`

Bright red background (SGR 101).

#### `bg-bright-green(s)`

Bright green background (SGR 102).

#### `bg-bright-yellow(s)`

Bright yellow background (SGR 103).

#### `bg-bright-blue(s)`

Bright blue background (SGR 104).

#### `bg-bright-magenta(s)`

Bright magenta background (SGR 105).

#### `bg-bright-cyan(s)`

Bright cyan background (SGR 106).

#### `bg-bright-white(s)`

Bright white background (SGR 107).

#### `fg-256(n, s)`

256-color palette foreground; `n` is a palette index 0..255: ESC[38;5;n m.

#### `bg-256(n, s)`

256-color palette background; `n` is a palette index 0..255: ESC[48;5;n m.

#### `rgb(r, g, b, s)`

Truecolor foreground from `r`,`g`,`b` (each 0..255): ESC[38;2;r;g;b m.

#### `bg-rgb(r, g, b, s)`

Truecolor background from `r`,`g`,`b` (each 0..255): ESC[48;2;r;g;b m.

#### `hyperlink(url, label)`

Wrap `label` in an OSC 8 terminal hyperlink pointing at `url`:
ESC]8;;<url>ESC\<label>ESC]8;;ESC\. Returns `label` unchanged when disabled.

#### `combine(fns)`

Build one styling function from `fns` (each string->string), applied left-to-right so combine([bold, red])("x") == bold(red("x")).

#### `sgr-param-code?(c)`

True when byte code `c` is an SGR parameter character: a digit '0'..'9' or ';'.

#### `strip-ansi(s)`

Remove all SGR escape sequences (ESC[<params>m) from `s`, returning the plain text; malformed/non-SGR escapes are kept verbatim.

---

## beagle.bail

> **Documentation coverage:** 11/11 functions (100%)

#### `add(a, b)`

Tier-2 bail-out for `+`: adds `a` and `b` via the polymorphic path.

#### `sub(a, b)`

Tier-2 bail-out for `-`: subtracts `b` from `a` via the polymorphic path.

#### `mul(a, b)`

Tier-2 bail-out for `*`: multiplies `a` and `b` via the polymorphic path.

#### `div(a, b)`

Tier-2 bail-out for `/`: divides `a` by `b` via the polymorphic path.

#### `modulo(a, b)`

Tier-2 bail-out for `%`: returns `a` modulo `b` via the polymorphic path.

#### `lt(a, b)`

Tier-2 bail-out for `<`: returns whether `a` is less than `b`.

#### `lte(a, b)`

Tier-2 bail-out for `<=`: returns whether `a` is less than or equal to `b`.

#### `gt(a, b)`

Tier-2 bail-out for `>`: returns whether `a` is greater than `b`.

#### `gte(a, b)`

Tier-2 bail-out for `>=`: returns whether `a` is greater than or equal to `b`.

#### `eq(a, b)`

Tier-2 bail-out for `==`: returns whether `a` and `b` are equal.

#### `ne(a, b)`

Tier-2 bail-out for `!=`: returns whether `a` and `b` are not equal.

---

## beagle.base64

> **Documentation coverage:** 11/11 functions (100%)

#### `to-bytes(input)`

Normalize `input` to a Vec<Int> of byte values. A String is converted to
its raw UTF-8 bytes (multibyte-correct via a string-builder); a Vec<Int> is
assumed to already be byte values and returned unchanged.

#### `encode-with(input, alphabet, pad?)`

Core base64 encoder. Encodes `input` (String or Vec<Int>) using the 64-char
`alphabet`, emitting '=' padding for the trailing 1- or 2-byte group only
when `pad?` is true. Returns the encoded String.

#### `decode-char-value(code)`

Map a single character code to its 0..63 base64 value, accepting both the
standard ('+', '/') and URL-safe ('-', '_') symbols. Returns -1 for any
non-alphabet character (the caller handles padding/whitespace).

#### `decode-any(text)`

Core base64 decoder. Decodes `text` (standard or URL-safe alphabet) into a
Vec<Int> of byte values, ignoring '=' padding and ASCII whitespace. Throws
if any other invalid base64 character is encountered.

#### `encode(input)`

Encode `input` (String or Vec<Int>) to a standard-alphabet base64 String
with '=' padding. Thin wrapper over encode-with using STD_ALPHABET.

#### `decode(text)`

Decode a standard base64 String to a Vec<Int> of byte values. Thin wrapper
over decode-any (which also tolerates URL-safe symbols and whitespace).

#### `encode-url(input)`

Encode `input` (String or Vec<Int>) to a URL-safe base64 String using '-'
and '_' and emitting NO padding. Thin wrapper over encode-with using
URL_ALPHABET.

#### `decode-url(text)`

Decode a URL-safe base64 String to a Vec<Int> of byte values. Also accepts
standard-alphabet input and '=' padding. Thin wrapper over decode-any.

#### `decode-strict(text)`

Strictly decode a canonical base64 String to a Vec<Int> of byte values.
Throws on malformed input: length not a multiple of 4, any non-alphabet
character, or '=' padding that is misplaced or longer than two characters.

#### `decode-to-string(text)`

Decode a base64 String and return the decoded bytes as a UTF-8 String.
Convenience over decode + bytes-to-string; lenient like `decode` (tolerates
missing padding, ASCII whitespace, and both alphabets).

#### `bytes-to-string(bytes)`

Build a String from a Vec<Int> of byte values, appending each as a raw
UTF-8 byte. Useful for round-tripping decoded textual data back to a String.

---

## beagle.bigint

> **Documentation coverage:** 22/26 functions (84%)

#### `trim-limbs(limbs)`

Return a new limb vector with the most-significant (trailing) zero limbs
removed, so the result has no leading-zero limbs.

#### `make(sign, limbs)`

Construct a canonical BigInt from `sign` and a possibly-untrimmed limb
vector; if the trimmed limbs are empty the result is canonical zero (sign 0).

#### `zero()`

Return the canonical zero BigInt (sign 0, empty limbs).

#### `from-int(n)`

Build a BigInt from a native Int, handling negatives and the full 62-bit
range (including the most-negative value, whose magnitude is never formed).

#### `from-string(s)`

Parse a decimal string (optional leading '+'/'-', then one or more digits;
leading zeros allowed) into a BigInt. Throws on empty input, a sign with no
digits, or any non-digit character.

#### `is-zero?(b)`

Return true if `b` is the BigInt zero (sign 0).

#### `cmp-mag(a, b)`

Compare two magnitudes (limb vectors), ignoring sign. Returns -1, 0, or 1.

#### `add-mag(a, b)`

Add two magnitudes (limb vectors) and return a trimmed limb vector.

#### `sub-mag(a, b)`

Subtract magnitude `b` from magnitude `a` (caller must ensure a >= b) and
return a trimmed limb vector.

#### `mul-mag(a, b)`

Multiply two magnitudes with schoolbook long multiplication, returning a
trimmed limb vector (empty when either input is empty).

#### `compare(a, b)`

Compare two BigInts by value (sign then magnitude). Returns -1, 0, or 1.

#### `negate(b)`

Return `b` with its sign flipped; zero (sign 0) is unchanged.

#### `add(a, b)`

Add two BigInts, returning a canonical BigInt.

#### `sub(a, b)`

Subtract `b` from `a`; thin wrapper for `add(a, negate(b))`.

#### `mul(a, b)`

Multiply two BigInts, returning a canonical BigInt (zero if either is zero).

#### `abs(b)`

Absolute value of a BigInt: its magnitude as a non-negative BigInt.

#### `divmod(a, b)`

Truncated division with remainder: returns [quotient, remainder] with
a == quotient*b + remainder, the quotient truncated toward zero, and the
remainder carrying the sign of `a` (Java/Rust BigInteger semantics).
Throws on division by zero.

#### `div(a, b)`

Truncated quotient a / b (toward zero). Throws on division by zero.

#### `mod(a, b)`

Remainder of a / b, carrying the sign of `a` (a == div(a,b)*b + mod(a,b)).
Throws on division by zero.

#### `pow(base, exp)`

Raise `base` (a BigInt) to a non-negative integer `exp` (an Int) via
square-and-multiply. Throws on a negative exponent.

#### `append-padded-limb!(builder, limb)`

Append `limb` (0..9999) to `builder` as a 4-digit zero-padded group; used
for every limb except the most-significant one.

#### `to-string(b)`

Render a BigInt as a decimal string ("0" for zero, leading '-' for negatives).

<details>
<summary><strong>Undocumented functions (4)</strong></summary>

- `limb-shift-add(rem, digit)`
- `find-quotient-digit(rem, b)`
- `divmod-mag(a, b)`
- `BigInt_compare-to(a, b)`

</details>

---

## beagle.channel

> **Documentation coverage:** 21/21 functions (100%)

#### `make-channel()`

Create a new empty, open, unbounded channel.

#### `close!(ch)`

Mark `ch` closed (permanent, idempotent). Returns null. Buffered values can still be drained.

#### `closed?(ch)`

True iff `ch` has been closed via close!. A snapshot; may be stale under concurrency.

#### `send!(ch, v)`

Enqueue `v` at the tail of `ch` and return `v`. Lock-free (inlined CAS retry). Throws if `ch` is closed.

#### `send-loop(ch, v)`

Internal CAS retry helper for send!: append `v`, install, retry on contention. Returns null.

#### `try-receive(ch)`

Non-blocking dequeue. Returns {:some value} if an element was removed, {:closed true} if closed and drained, or {:none true} if empty and open.

#### `try-receive-loop(ch)`

Internal CAS retry helper for try-receive: read head, install tail (drop index 0), retry on contention.

#### `receive(ch)`

Blocking dequeue: spins with growing backoff until a value is available. Returns the value, or the `:closed` keyword if the channel is closed and drained.

#### `receive-spin(ch, backoff)`

Internal spin loop for receive: polls try-receive, backing off (capped at 256) while empty; returns `:closed` once closed and drained.

#### `spin-backoff(n)`

Busy-burn `n` iterations of pure work (no allocation) to throttle re-polling. Returns null.

#### `channel-count(ch)`

Number of buffered (unreceived) elements. A snapshot; may be stale under concurrency.

#### `channel-empty?(ch)`

True iff the channel had no buffered elements at the moment we looked.

#### `make-mutex()`

Create a new unlocked mutex.

#### `acquire!(mutex)`

Acquire the lock, spinning with growing backoff until the CAS false -> true wins. Returns null.

#### `acquire-spin(mutex, backoff)`

Internal spin loop for acquire!: retries the CAS, backing off (capped at 256) until acquired.

#### `release!(mutex)`

Release the lock by resetting it to false. Only the holder should call this. Returns null.

#### `with-lock(mutex, thunk)`

Run `thunk` while holding `mutex`, releasing on both normal return and exception (re-thrown). Returns thunk's value.

#### `make-counter()`

Create a new counter starting at 0.

#### `incr!(counter)`

Atomically add 1 and return the new value. Thin wrapper over add!(counter, 1).

#### `add!(counter, delta)`

Atomically add `delta` and return the new value. Inlined CAS retry loop.

#### `counter-value(counter)`

Read the current counter value (snapshot).

---

## beagle.cli

> **Documentation coverage:** 32/32 functions (100%)

#### `make-spec(prog, desc)`

Create an empty spec with a program name and description.

#### `add-flag(spec, name, short, help)`

Add a boolean flag to the spec. `short` may be null for no short form.

#### `add-option(spec, name, short, help, required, default)`

Add a value-taking option to the spec.
`short` may be null. `required` is a bool. `default` is used when absent.

#### `add-positional(spec, name, help, required)`

Add a positional argument to the spec.

#### `find-flag-by-name(spec, name)`

Find a flag entry by its long `:name`. Returns the entry map or null.

#### `find-option-by-name(spec, name)`

Find an option entry by its long `:name`. Returns the entry map or null.

#### `find-flag-by-short(spec, short)`

Find a flag entry by its single-character `:short` name. Returns entry or null.

#### `find-option-by-short(spec, short)`

Find an option entry by its single-character `:short` name. Returns entry or null.

#### `seed-defaults(spec)`

Build the initial options map: each flag and `:help` default to false, each
option to its declared `:default` (or null). Returns the seeded map.

#### `st-new(spec)`

Create a fresh parse-state atom holding {:options seeded :positionals [] :errors []}.

#### `st-set-option(state, name, value)`

Set `name` to `value` in the state atom's `:options` map (mutates via swap!).

#### `st-add-positional(state, value)`

Append `value` to the state atom's `:positionals` vector (mutates via swap!).

#### `st-add-error(state, msg)`

Append error string `msg` to the state atom's `:errors` vector (mutates via swap!).

#### `parse(argv, spec)`

Parse argv (a vector of strings) against a spec.
Returns {:options map :positionals vector :errors vector}.

#### `is-long-arg(arg)`

True if `arg` is a long option: begins with "--" and is longer than 2 chars.

#### `is-short-arg(arg)`

True if `arg` is a short option: starts with a single "-", longer than 1,
not "--", and not a negative number (char after "-" is not a digit).

#### `parse-long(state, spec, argv, arg, i, n)`

Handle a "--..." token (flag, --help, --name=value, or --name value option),
updating state and returning the next argv index to read.

#### `apply-long-option(state, spec, name, value)`

Apply a "--name=value": set the option if `name` is an option; if it is a
flag, record an error (flags take no value) but still mark it true; else
record an unknown-option error.

#### `parse-short(state, spec, argv, arg, i, n)`

Handle a "-..." token (single short flag/option or a cluster), updating
state and returning the next argv index to read.

#### `parse-short-cluster(state, spec, argv, cluster, i, n)`

Walk each char of a short cluster: flag chars set their flag true; an option
char consumes the rest of the cluster (-ovalue) or the next argv token
(-o value). Records errors for unknown chars / missing values. Returns next index.

#### `finalize(state, spec)`

Finish parsing: if --help was given, skip validation; otherwise check
required options and positionals. Returns the dereferenced result map.

#### `check-required-options(state, spec)`

Record a "missing required option" error for each required option left null. Returns null.

#### `check-required-positionals(state, spec)`

Record a "missing required positional" error for each required positional
not supplied (by position). Returns null.

#### `get-int(parsed, name)`

Read option `name` from a parsed result, coercing its value to an Int.
Returns null when the option is absent (its stored value is null). Throws a
clear error when the value is present but is not a valid integer.

#### `get-float(parsed, name)`

Read option `name` from a parsed result, coercing its value to a Float.
Returns null when the option is absent (its stored value is null). Throws a
clear error when the value is present but is not a valid float.

#### `get-bool(parsed, name)`

Read option `name` from a parsed result, coercing its value to a Bool.
A flag stored as true/false is returned as-is; a string option value must be
"true" or "false". Returns null when the option is absent (its stored value
is null). Throws a clear error on any other present value.

#### `get-or(parsed, name, default)`

Read option `name` from a parsed result, returning `default` when the option
is absent (its stored value is null) and the stored value otherwise.

#### `help-text(spec)`

Build a usage / help string from a spec.

#### `build-usage-line(spec)`

Build the "Usage: prog [options] <required> [optional]" line from the spec.

#### `build-positionals-section(spec)`

Build the "Arguments:" help section listing positionals, or "" if none.

#### `build-options-section(spec)`

Build the "Options:" help section listing each option (short/long, help,
and a "(required)" marker), or "" if there are no options.

#### `build-flags-section(spec)`

Build the "Flags:" help section listing each flag plus the always-present
"-h, --help" line.

---

## beagle.containers

> **Documentation coverage:** 34/51 functions (66%)

#### `deque-new()`

Create a new empty deque.

#### `deque-count(d)`

Number of elements in the deque.

#### `deque-empty?(d)`

Is the deque empty?

#### `coll-reverse(v)`

Return a new vector containing the elements of `v` in reverse order.

#### `coll-subvec(v, start, end)`

Return a new vector with the elements of `v` from index `start` (inclusive)
to `end` (exclusive).

#### `push-front(d, value)`

Push a value onto the FRONT of the deque, returning a new deque.

#### `push-back(d, value)`

Push a value onto the BACK of the deque, returning a new deque.

#### `deque-rebalance(d)`

Rebalance a deque when one side is empty: move half of the non-empty side
across (reversing as needed to preserve logical order), returning a new
deque. Returns `d` unchanged when neither side needs rebalancing.

#### `pop-front(d)`

Pop from the FRONT. Returns [value, new-deque].
Throws on an empty deque.

#### `pop-back(d)`

Pop from the BACK. Returns [value, new-deque].
Throws on an empty deque.

#### `deque-peek-front(d)`

Peek at the FRONT (logical head) element WITHOUT removing it.
Throws on an empty deque (matching pop-front's error style).

#### `deque-peek-back(d)`

Peek at the BACK (logical tail) element WITHOUT removing it.
Throws on an empty deque (matching pop-back's error style).

#### `deque-to-vec(d)`

Materialize the deque into a plain vector in logical (head->tail) order.

#### `dm-new(default)`

Create a new default-map with the given default value.

#### `dm-get(dm, key)`

Look up a key, returning the default if the key is absent.

Uses contains? (not v == null) so a key explicitly stored with a null value
reads back as null, distinct from an absent key returning the default.

#### `dm-assoc(dm, key, value)`

Associate key->value, returning a new default-map.

#### `dm-update(dm, key, f)`

Update key by applying f to its current value (or the default if absent),
returning a new default-map.

#### `dm-count(dm)`

Number of explicitly-stored keys.

#### `dm-remove(dm, key)`

Remove `key` from a default-map, returning a new default-map whose table no
longer contains `key`. The default value is preserved. Removing a key that
is absent yields an equivalent default-map (no error).

#### `om-new()`

Create a new empty ordered map.

#### `om-get(om, key)`

Look up a key (null if absent).

#### `om-has-key?(om, key)`

Return true if `key` is present in the ordered map's insertion-order vector
(linear scan using structural equality).

#### `om-assoc(om, key, value)`

Associate key->value, returning a new ordered map. New keys are appended
to the insertion order; existing keys keep their position.

#### `om-update(om, key, f)`

Update `key` by applying `f` to its current value (or null if the key is
absent), returning a new ordered map. Insertion order is preserved: an
existing key keeps its position, a new key is appended.

#### `om-keys-in-order(om)`

The keys in insertion order (a vector).

#### `om-pairs(om)`

The [key, value] pairs in insertion order (a vector of 2-vectors).

#### `om-vals-in-order(om)`

The values in insertion order (a vector).

#### `om-count(om)`

Number of entries.

#### `om-remove(om, key)`

Remove `key` from an ordered map, returning a new ordered map with `key`
removed from BOTH the insertion-order vector and the backing table. The
relative order of the remaining keys is preserved. Removing a key that is
absent yields an equivalent ordered map (no error).

#### `om-dissoc(om, key)`

Alias for `om-remove`: remove `key` from an ordered map (Clojure-style
`dissoc` name). Returns a new ordered map with `key` gone from both the
order vector and the table.

#### `vec-member?(v, x)`

Return true if `x` appears in vector `v` (linear scan using structural
equality).

#### `set-union-vec(a, b)`

Union of two vectors-as-sets: all distinct elements from both.

#### `set-intersection-vec(a, b)`

Intersection: distinct elements of `a` that also appear in `b`.

#### `set-difference-vec(a, b)`

Difference: distinct elements of `a` that do NOT appear in `b`.

<details>
<summary><strong>Undocumented functions (17)</strong></summary>

- `Deque_length(d)`
- `Deque_seq(d)`
- `Deque_conj(d, x)`
- `DefaultMap_get$2(dm, key)`
- `DefaultMap_get$3(dm, key, default)`
- `DefaultMap_assoc(dm, key, value)`
- `DefaultMap_length(dm)`
- `DefaultMap_contains?(dm, key)`
- `DefaultMap_keys(dm)`
- `DefaultMap_seq(dm)`
- `OrderedMap_get$2(om, key)`
- `OrderedMap_get$3(om, key, default)`
- `OrderedMap_assoc(om, key, value)`
- `OrderedMap_length(om)`
- `OrderedMap_contains?(om, key)`
- `OrderedMap_keys(om)`
- `OrderedMap_seq(om)`

</details>

---

## beagle.csv

> **Documentation coverage:** 13/13 functions (100%)

#### `parse-delim(text, delim, quote)`

Parse delimiter-separated text into a Vec of rows of string fields, using
the byte codes `delim` and `quote` (e.g. from char-code) for the field
separator and the quote character. The shared core of parse/parse-with.

#### `parse(text)`

Parse RFC 4180 CSV text into a Vec of rows, each a Vec of string fields.
Handles quoted fields containing commas, embedded newlines, and doubled
quotes. Empty input yields []; a trailing terminator adds no empty row.

#### `opt-char(opts, key, default-str)`

Read a one-character option string from map `opts` at `key`, defaulting to
`default-str`; throws if the resolved value is not exactly one character.

#### `parse-with(text, opts)`

Parse delimiter-separated text using an options map: `:delimiter` and
`:quote` are one-character strings (defaulting to "," and "\""). Returns a
Vec of rows of string fields. Throws if an option is not one character or
if the delimiter and quote characters are equal.

#### `parse-tsv(text)`

Parse tab-separated (TSV) text into a Vec of rows of string fields, using
the tab character as the delimiter and the double-quote for quoting.

#### `parse-with-header(text)`

Parse CSV whose first row is a header into a Vec of maps keyed by the
header names. Extra fields are dropped; missing fields default to "".
Returns [] when the input has no rows.

#### `needs-quoting?(s)`

Return true if string `s` must be quoted per RFC 4180, i.e. it contains
a comma, double-quote, CR, or LF.

#### `write-field!(builder, s)`

Append field string `s` to string-builder `builder`, wrapping it in
quotes and doubling embedded quotes only when needs-quoting? is true.
Mutates the builder; returns null.

#### `write-row!(builder, row)`

Append `row` (a Vec of fields) to `builder`, comma-separating fields
(each via to-string + write-field!) and terminating with CRLF.
Mutates the builder; returns null.

#### `write(rows)`

Serialize a Vec of rows (each a Vec of fields) into a CSV string with
CRLF line endings, including a trailing CRLF.

#### `write-with-header(header, rows)`

Serialize `header` as the first CSV row followed by `rows` (each a Vec
of fields), returning a CSV string with CRLF line endings.

#### `write-row(fields)`

Serialize one row (`fields`, a Vec of fields) into a single CSV line
string, comma-separating fields and quoting/escaping per RFC 4180. Adds
no trailing terminator. An empty Vec yields "".

#### `write-rows(rows)`

Serialize `rows` (each a Vec of fields) into a CSV string, joining rows
with CRLF and adding no trailing terminator. Each row is rendered via
write-row (RFC 4180 quoting). An empty Vec yields "".

---

## beagle.date

> **Documentation coverage:** 19/20 functions (95%)

#### `from-epoch(secs)`

Convert epoch seconds (UTC) into a DateTime using beagle.time/civil-from-epoch.

#### `now()`

Return the current wall-clock time as a UTC DateTime.

#### `to-epoch(dt)`

Convert a DateTime back to epoch seconds (UTC); inverse of from-epoch.

#### `is-leap-year?(y)`

Return true if year `y` is a leap year under the proleptic Gregorian rule.

#### `day-of-year(dt)`

Ordinal day-of-year for a DateTime: 1 for January 1st, up to 365 (or 366 in
a leap year) for December 31st. Computed purely from the year/month/day
fields by differencing day counts since the civil epoch, so it inherits the
module's leap-year handling and ignores time-of-day.

#### `weekday(dt)`

Day of week for a DateTime as an integer, 0 = Sunday .. 6 = Saturday.

#### `weekday-name(dt)`

Full English weekday name for a DateTime, e.g. "Monday". Built on top of
`weekday` (0 = Sunday .. 6 = Saturday), so the name always matches that
numbering.

#### `is-weekend?(dt)`

True when a DateTime falls on a Saturday or Sunday. Built on top of
`weekday` (0 = Sunday .. 6 = Saturday), so it matches that numbering.

#### `add-seconds(dt, n)`

Return a new DateTime `n` seconds after `dt` (negative `n` moves backward).

#### `add-days(dt, n)`

Return a new DateTime `n` days after `dt`; thin wrapper over add-seconds.

#### `diff-seconds(a, b)`

Whole seconds between two DateTimes, computed as b - a (signed).

#### `diff-days(a, b)`

Whole calendar days from `a` to `b` (signed), comparing only the UTC date
fields (year/month/day) — time-of-day is ignored. Counts day numbers since
the epoch, so it handles month and year rollover and is negative when `b`
precedes `a`. For example, 2024-01-15 -> 2024-01-20 is 5, the reverse is -5.

#### `pad-num(n, width)`

Format integer `n` as a zero-padded decimal string at least `width` digits
wide. Assumes `n >= 0`; does not truncate values wider than `width`.

#### `format-iso8601(dt)`

Render a DateTime as an ISO-8601 UTC string "YYYY-MM-DDTHH:MM:SSZ" (always
with the trailing 'Z'). Negative years emit a leading '-' with 4-digit magnitude.

#### `days-in-month(y, month)`

Number of days in `month` (1-12) for year `y`, accounting for leap Februaries.

#### `parse-digits(s, start, width)`

Parse `width` consecutive ASCII digits of `s` starting at `start` into a
non-negative integer; throws a parse-iso8601 error on any non-digit.

#### `expect-sep(s, idx, expected)`

Assert the character of `s` at `idx` equals the single-char `expected`,
throwing a parse-iso8601 separator error otherwise; returns null on success.

#### `parse-iso8601(s)`

Strictly parse an ISO-8601 UTC string "YYYY-MM-DDTHH:MM:SS" with an optional
trailing 'Z' into a DateTime. Validates separator positions and field ranges,
throwing a "beagle.date/parse-iso8601: ..." error on any malformed input.

#### `dt-equal?(a, b)`

True when two DateTimes have identical year/month/day/hour/minute/second
(field-wise comparison, since structs don't compare with ==).

<details>
<summary><strong>Undocumented functions (1)</strong></summary>

- `DateTime_compare-to(a, b)`

</details>

---

## beagle.effect

> **Documentation coverage:** 1/2 functions (50%)

#### `resume-tail(resume, value)`

Resume a captured effect continuation in tail position with `value`,
avoiding stack growth from nested resumes. Delegates to builtin/resume-tail.

<details>
<summary><strong>Undocumented functions (1)</strong></summary>

- `handle(self, op, resume)`

</details>

---

## beagle.ffi

> **Documentation coverage:** 47/50 functions (94%)

#### `load-library(path)` `builtin`

Load a dynamic library (shared object) from the given path.

Returns a Library struct that can be used with get-function.

Examples:
  (let lib (ffi/load-library "libm.dylib"))

#### `get-function(library, name, arg_types, return_type)` `builtin`

Get a function from a loaded library.

arg_types is an array of Type values, return_type is a Type.

Examples:
  (let sqrt-fn (ffi/get-function lib "sqrt" [Type.F64] Type.F64))

#### `get-symbol(library, name)` `builtin`

Get a raw function pointer (symbol) from a loaded library.

Returns a Pointer struct. Used with call-variadic for per-call type specification.

#### `create-callback(fn, arg_types, return_type)` `builtin`

Create an FFI callback from a Beagle function.

Returns a Pointer that can be passed to C functions expecting function pointers.
arg_types is an array of Type values for the C callback parameters.
return_type is the C return type.

Callbacks must be called from the main thread.

#### `allocate(size)` `builtin`

Allocate size bytes of unmanaged memory.

Returns a Pointer. Must be freed with deallocate.

#### `deallocate(ptr)` `builtin`

Free memory allocated with allocate.

#### `get-u32(ptr, offset)` `builtin`

Read an unsigned 32-bit integer from memory at ptr + offset.

#### `set-i16(ptr, offset, value)` `builtin`

Write a signed 16-bit integer to memory at ptr + offset.

#### `set-i32(ptr, offset, value)` `builtin`

Write a signed 32-bit integer to memory at ptr + offset.

#### `set-u8(ptr, offset, value)` `builtin`

Write an unsigned 8-bit integer (byte) to memory at ptr + offset.

#### `get-u8(ptr, offset)` `builtin`

Read an unsigned 8-bit integer (byte) from memory at ptr + offset.

#### `get-i32(ptr, offset)` `builtin`

Read a signed 32-bit integer from memory at ptr + offset.

#### `set-i8(ptr, offset, value)` `builtin`

Write a signed 8-bit integer to memory at ptr + offset.

#### `get-i8(ptr, offset)` `builtin`

Read a signed 8-bit integer from memory at ptr + offset.

#### `set-u16(ptr, offset, value)` `builtin`

Write an unsigned 16-bit integer to memory at ptr + offset.

#### `get-u16(ptr, offset)` `builtin`

Read an unsigned 16-bit integer from memory at ptr + offset.

#### `get-i16(ptr, offset)` `builtin`

Read a signed 16-bit integer from memory at ptr + offset.

#### `set-u32(ptr, offset, value)` `builtin`

Write an unsigned 32-bit integer to memory at ptr + offset.

#### `set-i64(ptr, offset, value)` `builtin`

Write a signed 64-bit integer to memory at ptr + offset.

#### `get-i64(ptr, offset)` `builtin`

Read a signed 64-bit integer from memory at ptr + offset.

#### `set-u64(ptr, offset, value)` `builtin`

Write an unsigned 64-bit integer to memory at ptr + offset.

#### `get-u64(ptr, offset)` `builtin`

Read an unsigned 64-bit integer from memory at ptr + offset.

#### `set-f32(ptr, offset, value)` `builtin`

Write a 32-bit float to memory at ptr + offset. Accepts Int or Float.

#### `set-f64(ptr, offset, value)` `builtin`

Write a 64-bit float to memory at ptr + offset. Accepts Int or Float.

#### `get-f32(ptr, offset)` `builtin`

Read a 32-bit float from memory at ptr + offset. Returns a Beagle Float.

#### `get-f64(ptr, offset)` `builtin`

Read a 64-bit float from memory at ptr + offset. Returns a Beagle Float.

#### `get-string(buffer, offset, length)` `builtin`

Read a string from a buffer at the given offset with the specified length.

Examples:
  (ffi/get-string buf 0 10)  ; Read 10 bytes starting at offset 0

#### `get-string-and-free(buffer, offset, length)` `builtin`

Read a string from a buffer and free the native buffer.
Combines get-string + deallocate to avoid holding Buffer reference across GC.

#### `create-array(size)` `builtin`

Create a new FFI buffer/array of the given size in bytes.

#### `copy-bytes(src, src_offset, dest, dest_offset, length)` `builtin`

Copy length bytes from src+src_offset to dest+dest_offset.

#### `realloc(ptr, new_size)` `builtin`

Reallocate memory to a new size, preserving contents.

#### `buffer-size(buffer)` `builtin`

Get the size of a buffer in bytes.

#### `write-buffer-offset(buffer, offset, value, size)` `builtin`

Write a value to a buffer at a given offset.

#### `translate-bytes(buffer, offset, length, table)` `builtin`

Translate bytes using a lookup table.

#### `reverse-bytes(buffer, offset, length)` `builtin`

Reverse bytes in place in a buffer.

#### `find-byte(buffer, offset, length, byte)` `builtin`

Find the first occurrence of a byte in a buffer.

Returns the offset or -1 if not found.

#### `call-variadic(func_ptr, args, types, return_type)`

Call raw function pointer `func_ptr` with `args` whose per-argument FFI
types are given in `types`, returning a value of `return_type`. Registers
the C call with the GC for the duration.

#### `size-of(ty)`

Return the byte size of a primitive FFI `ty`. Throws for String, Void, and
Structure, which have no fixed primitive size.

#### `read-at(ptr, offset, ty)`

Read a value of type `ty` from `ptr` at byte `offset`, dispatching to the
matching get-* builtin. Throws for Pointer/MutablePointer/String/Void/
Structure, which are not supported here.

#### `write-at(ptr, offset, ty, value)`

Write `value` of type `ty` to `ptr` at byte `offset`, dispatching to the
matching set-* builtin. Throws for Pointer/MutablePointer/String/Void/
Structure, which are not supported here.

#### `cell(ty, initial)`

Allocate a GC-managed typed `Cell` of type `ty` initialized to `initial`.
Its raw address can be passed directly to C; the backing memory is freed
when the Cell is collected.

#### `cell-get(c)`

Read the current value out of cell `c`, using the cell's stored type.

#### `cell-set(c, value)`

Store `value` into cell `c`, using the cell's stored type.

#### `array(ty, length)`

Allocate a GC-managed `TypedArray` of `length` elements of type `ty`. The
raw address can be passed directly to C; the backing memory is freed when
the array is collected.

#### `array-get(a, i)`

Read element `i` from typed array `a` (offset = `i` * element stride).

#### `array-set(a, i, value)`

Write `value` to element `i` of typed array `a` (offset = `i` * stride).

#### `forget(x)`

Disown the off-heap memory held by finalizable FFI struct `x` (Buffer,
Cell, or TypedArray), handing ownership to C. Afterward the GC finalizer
no-ops and beagle-side reads/writes raise a resumable FFIError.

<details>
<summary><strong>Undocumented functions (3)</strong></summary>

- `call-ffi-info()`
- `call-variadic-raw()`
- `copy-bytes_filter()`

</details>

---

## beagle.glob

> **Documentation coverage:** 15/15 functions (100%)

#### `code-at(s, i)`

Return the integer character code of `s[i]`. Thin wrapper over `char-code`.

#### `is-slash-code(c)`

True if char code `c` is the path separator '/' (code 47).

#### `class-match(pattern, open-idx, target-code)`

Match a character class beginning just past '[' (at `open-idx`) against the
char code `target-code`. Handles ranges (`a-z`) and negation (`!`/`^`).
Returns `[matched?, next-index]` where next-index is just past the ']';
an unterminated class returns `[false, -1]`. Classes never match '/'.

#### `is-double-star(pattern, pi)`

True if the '*' run at index `pi` is a cross-slash `**` wildcard, i.e. a
whole path segment (preceded by start-of-pattern or '/', followed by
end-of-pattern or '/'). Otherwise the run collapses to a single '*'.

#### `match-from(pattern, pi, name, ni)`

Core recursive glob matcher: does `pattern` from index `pi` match `name`
from index `ni`? Handles `*`, `**`, `?`, `[...]` classes, and `\` escapes
with explicit star backtracking. Returns Bool.

#### `match?(pattern, name)`

Match a glob pattern against a name (or path).

Examples:
  match?("*.txt", "foo.txt")        // => true
  match?("*.txt", "foo.md")         // => false
  match?("a?c", "abc")              // => true
  match?("[a-c]x", "bx")            // => true
  match?("src/**/*.bg", "src/a/b/c.bg") // => true
  match?("*", "a/b")               // => false

#### `translate(pattern)`

Translate a glob pattern into a space-separated token description.

Useful for verifying how a pattern parses.

Examples:
  translate("a*c")        // => "LIT(a) STAR LIT(c)"
  translate("src/**/*.bg")
    // => "LIT(s) LIT(r) LIT(c) LIT(/) DOUBLESTAR LIT(/) STAR LIT(.) LIT(b) LIT(g)"

#### `translate-class(pattern, pi)`

Render the character-class token starting at '[' (`pattern[pi]`) for
`translate`. Returns `[token-string, next-index]`; an unterminated class
yields `["LIT([)", pi + 1]`.

#### `match-ci?(pattern, name)`

Case-insensitive variant of `match?`: matches `pattern` against `name`
ignoring ASCII letter case. Lowercases both sides, then runs the same
matcher (so `*`, `?`, classes, ranges, and escapes behave identically).

Examples:
  match-ci?("*.TXT", "foo.txt")  // => true
  match-ci?("[A-C]x", "BX")      // => true

#### `brace-first-open(pattern)`

Index of the first unescaped '{' in `pattern`, or -1 if none.

#### `brace-matching-close(pattern, open)`

Index of the '}' matching the '{' at `open` in `pattern`, accounting for
nested braces and '\' escapes. Returns -1 if the group is never closed.

#### `brace-split-commas(inner)`

Split `inner` (the text between a group's braces) on its depth-0 commas,
honoring nested `{...}` and '\' escapes. Returns a vector of option strings;
text with no top-level comma yields a single-element vector.

#### `expand-braces(pattern)`

Expand shell-style brace groups in `pattern` into the list of concrete
strings they denote, e.g. "a{b,c}d" => ["abd", "acd"]. Multiple groups
produce the cartesian product; nested groups expand out-in. A group with no
top-level comma, and any unbalanced '{', are left literal. Never throws on a
well-formed string; always returns a non-empty vector.

Examples:
  expand-braces("a{b,c}d")      // => ["abd", "acd"]
  expand-braces("{a,b}{1,2}")   // => ["a1", "a2", "b1", "b2"]
  expand-braces("plain")        // => ["plain"]

#### `find-walk(root, start-rel, pattern, acc)`

Iteratively walk the directory tree under `root` (starting at relative path
`start-rel`), accumulating into `acc` every entry whose path relative to
`root` matches `pattern`. Uses an explicit worklist (no Beagle recursion
across the fs perform/resume boundary); read/stat errors are skipped.
Returns the vector of matching relative paths.

#### `find(pattern, dir)`

Recursively find paths under `dir` matching `pattern`.

Paths are returned relative to `dir`. Wraps the directory walk in a
blocking fs handler, so no handler setup is required by the caller.
Pure matching is done by match?; this only walks the directory tree.

Examples:
  find("src/**/*.bg", ".")  // => ["src/a/b.bg", ...]

---

## beagle.hash

> **Documentation coverage:** 45/45 functions (100%)

#### `to-bytes(input)`

Normalizes hash input into a Vec<Int> of byte values (0..255). A String is
converted to its UTF-8 bytes; a Vec<Int> has each element masked to a byte.

#### `add32(a, b)`

Adds two 32-bit words modulo 2^32 (result masked to 32 bits).

#### `rotl32(x, n)`

Rotates the 32-bit word `x` left by `n` bits.

#### `rotr32(x, n)`

Rotates the 32-bit word `x` right by `n` bits.

#### `append-hex32!(builder, word)`

Appends `word` to `builder` as 8 lowercase big-endian hex digits. Returns the builder.

#### `append-hex8!(builder, byte)`

Appends `byte` to `builder` as 2 lowercase hex digits. Returns the builder.

#### `pad-message(bytes)`

Applies SHA-1/SHA-256 (big-endian) message padding to `bytes`, returning a
new Vec<Int> whose length is a multiple of 64.

#### `word-be(bytes, i)`

Reads a big-endian 32-bit word from `bytes` at index `i`.

#### `word-le(bytes, i)`

Reads a little-endian 32-bit word from `bytes` at index `i` (used by MD5).

#### `pad-message-le(bytes)`

MD5 message padding: like pad-message but with a little-endian 64-bit bit-length field.

#### `md5-s()`

Returns the 64 per-round left-rotate amounts for MD5.

#### `md5-k()`

Returns the 64 MD5 round constants K[i] = floor(abs(sin(i+1)) * 2^32).

#### `md5-digest-words(raw-bytes)`

Runs the MD5 compression over `raw-bytes`, returning the digest as a Vec of 4 words.

#### `append-hex32-le!(builder, word)`

Appends `word` to `builder` byte-reversed (little-endian) as 8 hex digits. Returns the builder.

#### `md5(input)`

Computes the MD5 hash of `input` (String or Vec<Int>), returning a 32-char lowercase hex String.

#### `md5-digest-bytes(raw-bytes)`

MD5 digest of `raw-bytes` as a Vec of 16 bytes (little-endian per word), for HMAC.

#### `sha1(input)`

Computes the SHA-1 hash of `input` (String or Vec<Int>), returning a 40-char lowercase hex String.

#### `sha256-digest-words(raw-bytes)`

Runs SHA-256 over a Vec<Int> of bytes, returning the digest as a Vec of 8 words.

#### `words-to-hex(words)`

Concatenates a Vec of 32-bit words into a big-endian lowercase hex String.

#### `sha256(input)`

Computes the SHA-256 hash of `input` (String or Vec<Int>), returning a 64-char lowercase hex String.

#### `sha256-k()`

Returns the 64 SHA-256 round constants.

#### `w64(hi, lo)`

Builds a W64 from `hi` and `lo`, masking each to 32 bits.

#### `xor64(a, b)`

Bitwise XOR of two W64 values.

#### `and64(a, b)`

Bitwise AND of two W64 values.

#### `not64(a)`

Bitwise NOT (complement) of a W64 value.

#### `add64(a, b)`

Adds two W64 values modulo 2^64, carrying the low-half overflow into the high half.

#### `shr64(x, n)`

Logical right shift of a W64 by `n` bits (0 <= n < 64).

#### `shl64(x, n)`

Left shift of a W64 by `n` bits (0 <= n < 64).

#### `rotr64(x, n)`

Rotates a W64 right by `n` bits (0 <= n < 64).

#### `word-be64(bytes, i)`

Reads a big-endian 64-bit word from `bytes` at index `i` as a W64.

#### `pad-message-128(bytes)`

SHA-512 message padding: block size 128 with a 128-bit big-endian length
field (the high 64 bits are always zero for our inputs).

#### `sha512-k()`

Returns the 80 SHA-512 round constants as W64 values.

#### `sha512-digest-words(raw-bytes)`

Runs SHA-512 over a Vec<Int> of bytes, returning the digest as a Vec of 8 W64.

#### `words64-to-hex(words)`

Concatenates a Vec of W64 into a big-endian lowercase hex String.

#### `sha512(input)`

Computes the SHA-512 hash of `input` (String or Vec<Int>), returning a 128-char lowercase hex String.

#### `sha512-digest-bytes(raw-bytes)`

SHA-512 digest of `raw-bytes` as a Vec of 64 bytes (big-endian), for HMAC.

#### `words-to-bytes(words)`

Converts 8 SHA-256 words into a Vec of 32 bytes (big-endian).

#### `concat-bytes(a, b)`

Returns a new byte Vec with the bytes of `b` appended to `a`.

#### `bytes-to-hex(bytes)`

Returns the lowercase hex String of an arbitrary byte Vec.

#### `hmac-bytes(digest-bytes, block-size, key, msg)`

Generic HMAC over a raw-bytes digest function. `digest-bytes` maps
Vec<Int> -> Vec<Int> (the raw hash output); `block-size` is the hash's
internal block size in bytes. Returns the raw HMAC bytes.

#### `hmac-sha256(key, msg)`

Computes HMAC-SHA-256 of `msg` under `key`, returning a 64-char lowercase hex String.

#### `hmac-md5(key, msg)`

Computes HMAC-MD5 of `msg` under `key`, returning a 32-char lowercase hex String.

#### `hmac-sha512(key, msg)`

Computes HMAC-SHA-512 of `msg` under `key`, returning a 128-char lowercase hex String.

#### `crc32(input)`

Computes the standard CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320)
of `input` (a String, hashed as its UTF-8 bytes, or a Vec<Int> of byte
values), returning the unsigned 32-bit checksum as an Int. crc32("") == 0.

#### `fnv1a-32(input)`

Computes the 32-bit FNV-1a hash of `input` (a String, hashed as its UTF-8
bytes, or a Vec<Int> of byte values), returning the unsigned 32-bit hash as
an Int. Offset basis 2166136261, prime 16777619: for each byte, XOR into the
hash then multiply by the prime (masked to 32 bits). fnv1a-32("") == 2166136261.

---

## beagle.hex

> **Documentation coverage:** 5/5 functions (100%)

#### `to-bytes(input)`

Coerce `input` to a Vec<Int> of bytes: a String becomes its UTF-8 bytes
(via a string-builder), any other value (a Vec<Int>) is returned unchanged.

#### `nibble-to-code(n)`

Map a nibble `n` (0..15) to its lowercase hex character code ('0'-'9', 'a'-'f').

#### `code-to-nibble(c)`

Map a hex digit character code `c` to its value (0..15), case-insensitive;
returns -1 if `c` is not a hex digit.

#### `encode(input)`

Encode `input` (a Vec<Int> of byte values, or a String by its UTF-8 bytes)
as a lowercase hex string with exactly two characters per byte.

#### `decode(s)`

Decode hex string `s` into a Vec<Int> of byte values (case-insensitive).
Throws on an odd-length string or any non-hex character.

---

## beagle.http

> **Documentation coverage:** 56/70 functions (80%)

#### `reason-phrase(status)`

Return the reason phrase for an HTTP status code, or "Unknown" if the code
is not in the table.

#### `header(req-or-resp, name)`

Read a header from a Request or Response by name, case-insensitively;
returns null if absent.

#### `set-header(headers, name, value)`

Return a new headers map with name (lowercased) set to value.

#### `has-header?(headers, name)`

True if the headers map contains the given header name (case-insensitive).

#### `make-response(status, headers, body)`

Build a Response from a status code, headers map, and body; the reason
phrase is filled in from the status table.

#### `ok(body)`

200 OK response with a text/plain UTF-8 body.

#### `ok-with(content-type, body)`

200 OK response with an explicit content type and body.

#### `html(body)`

200 OK response with a text/html UTF-8 body.

#### `json(json-string)`

200 OK response with an application/json body; json-string must already be
a serialized JSON string.

#### `not-found(body)`

404 Not Found response with a text/plain body.

#### `status-response(status, body)`

Response with an arbitrary status code and a text/plain body.

#### `redirect(location)`

302 Found redirect response to the given location.

#### `with-status(resp, status)`

Return a copy of `resp` with its status code (and matching reason phrase) set.

Examples:
  with-status(ok("x"), 201)  // => 201 Created

#### `with-header(resp, name, value)`

Return a copy of `resp` with the header set (case-insensitive; replaces any
existing value).

Examples:
  with-header(ok("x"), "X-Request-Id", "abc")

#### `with-body(resp, body)`

Return a copy of `resp` with its body replaced.

#### `with-content-type(resp, content-type)`

Return a copy of `resp` with its Content-Type header set.

#### `conn-buf(conn)`

Wrap a raw socket connection in a fresh ConnBuf with an empty leftover
buffer.

#### `conn-buf-tls(conn)`

Wrap a TLS connection handle in a fresh ConnBuf for HTTPS reads.

#### `byte-slice(s, start, end)`

Return the substring of s between BYTE indices [start, end); "" if
end <= start.

#### `cb-fill!(cb)`

Pull one more socket chunk into the buffer's leftover; returns true if any
bytes were appended, false at EOF.

#### `read-line(cb)`

Read one CRLF/LF-terminated line (terminator stripped); returns null at EOF
before any bytes. Over-read bytes stay buffered for the next read.

#### `strip-cr(line)`

Drop a single trailing CR (the "\r" of a CRLF terminator) from a line.

#### `read-n(cb, n)`

Read exactly n bytes (or fewer at EOF) from the connection as a string,
measured in BYTES; over-read bytes stay buffered.

#### `parse-request-line(line)`

Parse a request line "METHOD PATH?QUERY VERSION" into
[method, path, query, version], or null if malformed.

#### `parse-header-line(line)`

Parse a "Name: value" header line into [lowercase-name, trimmed-value], or
null if there is no colon.

#### `read-headers(cb)`

Read header lines into a PersistentMap (keys lowercased) until the blank
line that ends the header block, or EOF.

#### `parse-chunk-size(line)`

Parse a chunked-encoding chunk-size line (hex, optionally with ";extension")
into an Int.

#### `decode-chunked(s)`

Decode a complete chunked-transfer-encoding body string into the actual body.

#### `decode-chunked-into(s, pos, out)`

Internal recursive worker for decode-chunked.

#### `read-chunked-body(cb)`

Read and decode a chunked-transfer-encoding body from the connection buffer.

#### `read-chunked-loop(cb, out)`

Internal read loop for read-chunked-body.

#### `read-request(cb)`

Read and parse a full HTTP request (request line, headers, and a
Content-Length or chunked body if present) into a Request; null if no
request line.

#### `serialize-response(resp)`

Serialize a Response to a full HTTP/1.1 string, always setting
Content-Length and Connection: close and emitting the user headers.

#### `byte-length(s)`

Return the UTF-8 byte length of a string (codepoint count != bytes for
non-ASCII), computed via a string-builder.

#### `route(method, path, handler)`

Build a Route, uppercasing the method.

#### `split-path-segments(p)`

Split a path into non-empty "/"-separated segments.

#### `match-path(pattern, path)`

Match a route pattern against a path. Returns a params map (string->string)
on match, or null. Supports `:name` params and a trailing `*name` wildcard.

#### `match-segs(pat, pth, i, params)`

Internal recursive segment matcher for match-path.

#### `query-params(req)`

Parse the request's query string into a decoded map (e.g. "a=1&b=2" ->
{"a" "1", "b" "2"}). Empty/absent query yields {}.

#### `query-param(req, name)`

Return the decoded value of query parameter `name`, or null if absent.

#### `parse-cookie-header(h)`

Parse a Cookie header value ("a=1; b=2") into a map of cookie name -> value.

#### `parse-cookie-pairs(h, start, acc)`

Internal: fold the "; "-separated pairs of a Cookie header into a map.

#### `cookies(req)`

Parse the request's Cookie header into a map of cookie name -> value ({} if none).

#### `cookie(req, name)`

Return the value of cookie `name`, or null if absent.

#### `form-params(req)`

Parse the request body as application/x-www-form-urlencoded into a decoded
map (string->string). Empty/absent body yields {}.

#### `form-param(req, name)`

Return the decoded form field `name`, or null if absent.

#### `with-params(req, params)`

Return a copy of `req` with `params` attached (used by the router).

#### `router(routes, not-found-handler)`

Build a handler dispatching a vector of Route on method + path (with `:name`
params and a trailing `*name` wildcard). 405 with an Allow header on a
path-but-not-method match, otherwise the not-found-handler.

#### `default-not-found(req)`

Default 404 handler returning "Not Found: <path>".

#### `handle-connection(conn, handler)`

Serve one connection: read the request, run the handler (handler errors
become a 500 so the loop survives), write the response, and close.

#### `serve(host, port, handler)`

Run a cooperative single-thread HTTP server with the given handler
(Request -> Response), serving one connection at a time on the calling
thread (GC-safe). Blocks forever.

#### `serve-routes(host, port, routes)`

Serve a vector of Route with the default 404 handler.

#### `read-response(cb)`

Read and parse a full HTTP response (status line, headers, then body by
Content-Length if present else until EOF) into a Response.

#### `read-until-eof(cb)`

Read and return everything remaining on the connection until EOF, draining
any already-buffered bytes first.

#### `build-request(method, host, path, headers, body)`

Build the HTTP/1.1 request bytes for a method/host/path/headers/body.

#### `request(method, url, opts)`

Make an HTTP or HTTPS request to a full `url` and return the parsed Response.
The URL scheme selects the transport (plain TCP for http, TLS for https) and
the default port (80 / 443). `opts` is a map of options:
  :headers  a map of request headers (default {})
  :body     a request body string (default "")

Examples:
  request("GET", "https://example.com/", {})
  request("POST", "http://api.local/x", {:body "hi", :headers {"x-tok" "v"}})

<details>
<summary><strong>Undocumented functions (14)</strong></summary>

- `hex-val(c)`
- `trim-spaces(s)`
- `get$1(url)`
- `get$2(url, opts)`
- `post$1(url)`
- `post$2(url, opts)`
- `put$1(url)`
- `put$2(url, opts)`
- `patch$1(url)`
- `patch$2(url, opts)`
- `delete$1(url)`
- `delete$2(url, opts)`
- `head$1(url)`
- `head$2(url, opts)`

</details>

---

## beagle.ini

> **Documentation coverage:** 16/16 functions (100%)

#### `ws?(c)`

True if byte `c` is ASCII whitespace (space, tab, CR, or LF).

#### `skip-ws-forward(s, start, end)`

Index of the first non-whitespace byte in s[start:end], or `end` if none.

#### `trim-ws-backward(s, start, end)`

Index one past the last non-whitespace byte in s[start:end]; `start` if the range is all whitespace.

#### `trim-range(s, start, end)`

Trim ASCII whitespace from s[start:end] and return that substring.

#### `unquote-value(v)`

Strip one pair of surrounding double quotes from an already-trimmed value, returning the inner text; returns the value unchanged if it is not quoted.

#### `find-separator(s, start, end)`

Index of the first '=' or ':' in s[start:end], or -1 if neither occurs.

#### `parse(text)`

Parse INI/.conf `text` into a map of section-keyword -> property map (string keys and string values). Keys before any [section] live under :default. Accepts both '\n' and '\r\n' line endings.

#### `parse-line(text, start, end, result, current)`

Process one raw line (text[start:end], no terminator) against the running parse state. Handles blank lines, ';'/'#' comments, [section] headers, and key=value / key:value properties; returns [updated-result, updated-current-section].

#### `value-needs-quoting?(s)`

True if writing `s` needs surrounding quotes to survive a parse round-trip — i.e. it has leading/trailing whitespace or contains a double quote. Empty strings never need quoting.

#### `write-value!(builder, s)`

Append value `s` to the string builder, wrapping it in double quotes only when value-needs-quoting? is true. Returns null.

#### `write-section-body!(builder, section)`

Append one "key=value" line (LF-terminated) per entry of a section's property map to the builder, quoting values as needed. Returns null.

#### `stringify(data)`

Serialize a parsed-shape map back into INI text. The :default section, if present, is emitted first without a header; every other section gets a [name] header. Uses '\n' line endings and '=' separators, and re-quotes values only when needed so stringify(parse(x)) re-parses equal.

#### `get-raw(ini, section, key)`

Look up the raw string value for `section`/`key` in a parsed INI map, or null when the section or key is absent. `section` is the bare header string (e.g. "db").

#### `get-bool(ini, section, key)`

Read `section`/`key` from a parsed INI map as a Bool. Accepts (case-insensitively) true/false, yes/no, on/off, and 1/0. Returns null when absent; throws when the value is present but is not one of those.

#### `get-int(ini, section, key)`

Read `section`/`key` from a parsed INI map as an Int (optional leading '-'). Returns null when absent; throws when the value is present but is not a valid integer.

#### `get-float(ini, section, key)`

Read `section`/`key` from a parsed INI map as a Float (decimal/exponent forms accepted; an integer like "10" becomes 10.0). Returns null when absent; throws when the value is present but is not a valid number.

---

## beagle.ip

> **Documentation coverage:** 20/31 functions (64%)

#### `parse-octet(s)`

Parse a single decimal IPv4 octet string into an Int in [0,255].

Returns null for empty input, non-digit characters, more than 3 digits,
values above 255, or a leading zero (octal-confusion guard); "0" is valid.

#### `parse-ipv4(s)`

Parse a dotted-quad IPv4 string into a 32-bit Int.

Returns null if the string is not a valid IPv4 address.

Examples:
  parse-ipv4("192.168.1.1")   // => 3232235777
  parse-ipv4("999.1.1.1")     // => null

#### `ipv4-to-string(n)`

Convert a 32-bit Int into a dotted-quad IPv4 string.

Examples:
  ipv4-to-string(3232235777)   // => "192.168.1.1"

#### `prefix-to-mask(prefix)`

Build a 32-bit IPv4 netmask Int from a prefix length in [0,32].

prefix 0 => 0; prefix 32 => 0xFFFFFFFF; values are clamped to that range.

#### `cidr-network(cidr)`

Parse a CIDR string ("a.b.c.d/prefix") into [network-int, mask-int].

Returns null if the CIDR is malformed.

Examples:
  cidr-network("192.168.1.0/24")   // => [3232235776, 4294967040]

#### `ipv4-in-cidr?(ip-str, cidr-str)`

Check whether an IPv4 address string lies within a CIDR block.

Examples:
  ipv4-in-cidr?("192.168.1.5", "192.168.1.0/24")   // => true
  ipv4-in-cidr?("192.168.2.5", "192.168.1.0/24")   // => false

#### `is-private-ipv4?(s)`

Check whether an IPv4 address string is in a private range.

Covers RFC 1918 (10/8, 172.16/12, 192.168/16), loopback (127/8),
and link-local (169.254/16). Returns false for invalid input.

Examples:
  is-private-ipv4?("10.0.0.1")      // => true
  is-private-ipv4?("8.8.8.8")       // => false

#### `is-loopback?(ip-str)`

Check whether an IPv4 address string is a loopback address (127.0.0.0/8).

Returns false for invalid input.

Examples:
  is-loopback?("127.0.0.1")         // => true
  is-loopback?("127.255.255.254")   // => true
  is-loopback?("128.0.0.1")         // => false

#### `is-private?(ip-str)`

Check whether an IPv4 address string is in an RFC 1918 private range.

Covers only the three RFC 1918 blocks: 10.0.0.0/8, 172.16.0.0/12, and
192.168.0.0/16. Loopback and link-local are NOT considered private here
(see is-private-ipv4? for the broader check). Returns false for invalid input.

Examples:
  is-private?("10.1.2.3")         // => true
  is-private?("172.16.0.1")       // => true
  is-private?("172.15.255.255")   // => false
  is-private?("192.168.1.1")      // => true
  is-private?("8.8.8.8")          // => false

#### `is-multicast?(ip-str)`

Check whether an IPv4 address string is a multicast address (224.0.0.0/4).

The multicast range is 224.0.0.0 through 239.255.255.255. Returns false
for invalid input.

Examples:
  is-multicast?("224.0.0.1")         // => true
  is-multicast?("239.255.255.255")   // => true
  is-multicast?("223.255.255.255")   // => false
  is-multicast?("240.0.0.0")         // => false

#### `broadcast-address(cidr-str)`

Compute the broadcast address string for an IPv4 CIDR block.

The broadcast address sets every host bit of the network. For /32 the
broadcast equals the single host address; for /31 it is the upper of the
two addresses. Throws on a malformed CIDR string.

Examples:
  broadcast-address("192.168.1.0/24")     // => "192.168.1.255"
  broadcast-address("192.168.1.130/26")   // => "192.168.1.191"
  broadcast-address("0.0.0.0/0")          // => "255.255.255.255"

#### `host-count(cidr-str)`

Count the number of usable host addresses in an IPv4 CIDR block.

For prefixes /0 through /30 this is the total address count minus the
network and broadcast addresses (2^(32-prefix) - 2). Following the
standard small-subnet conventions, a /31 returns 2 (RFC 3021
point-to-point) and a /32 returns 1 (single host). Throws on a malformed
CIDR string.

Examples:
  host-count("192.168.1.0/24")   // => 254
  host-count("192.168.1.0/30")   // => 2
  host-count("192.168.1.0/31")   // => 2
  host-count("192.168.1.1/32")   // => 1

#### `parse-ipv6(s)`

Parse an IPv6 address string into a vector of 8 16-bit group ints, or null if
invalid. Handles "::" zero-compression. Embedded-IPv4 forms are unsupported.

Examples:
  parse-ipv6("::1")           // => [0, 0, 0, 0, 0, 0, 0, 1]
  parse-ipv6("2001:db8::1")   // => [8193, 3512, 0, 0, 0, 0, 0, 1]

#### `ipv6?(s)`

True if `s` is a valid IPv6 address string.

#### `ipv6-to-string(groups)`

Format a vector of 8 16-bit group ints as a canonical IPv6 string (RFC 5952:
lowercase, the longest run of >=2 zero groups compressed to "::", leftmost on
a tie).

Examples:
  ipv6-to-string([0, 0, 0, 0, 0, 0, 0, 1])       // => "::1"
  ipv6-to-string([8193, 3512, 0, 0, 0, 0, 0, 1])  // => "2001:db8::1"

#### `is-ipv6-loopback?(s)`

True if `s` is the IPv6 loopback address (::1).

#### `is-ipv6-unspecified?(s)`

True if `s` is the IPv6 unspecified address (::).

#### `is-ipv6-link-local?(s)`

True if `s` is an IPv6 link-local address (fe80::/10).

#### `is-ipv6-unique-local?(s)`

True if `s` is an IPv6 unique-local address (fc00::/7).

#### `is-ipv6-multicast?(s)`

True if `s` is an IPv6 multicast address (ff00::/8).

<details>
<summary><strong>Undocumented functions (11)</strong></summary>

- `hex-digit-val(c)`
- `parse-hex-group(s)`
- `split-on-char(s, sep)`
- `parse-hex-run(s)`
- `parse-hex-run-helper(parts, i, acc)`
- `vec-append(a, b)`
- `to-hex-lower(n)`
- `to-hex-lower-helper(n, acc)`
- `longest-zero-run(groups)`
- `format-groups-range(groups, start, end)`
- `ipv6-all-zero?(g)`

</details>

---

## beagle.iter

> **Documentation coverage:** 34/38 functions (89%)

#### `iter-to-vec(coll)`

Walk coll via seq/first/next and collect every element into a fresh
persistent vector. Used internally to materialize Seqable inputs.

#### `mapcat(f, coll)`

Map f over coll and concatenate the resulting collections.
mapcat(fn(x) { [x, x] }, [1, 2]) // => [1, 1, 2, 2]

#### `reductions(f, init, coll)`

Successive reductions of f over coll, starting from init.
reductions(fn(a, x) { a + x }, 0, [1, 2, 3]) // => [0, 1, 3, 6]

#### `take-while(pred, coll)`

Take leading elements while pred is true.
take-while(fn(x) { x < 3 }, [1, 2, 3, 1]) // => [1, 2]

#### `drop-while(pred, coll)`

Drop leading elements while pred is true; return the rest.
drop-while(fn(x) { x < 3 }, [1, 2, 3, 1]) // => [3, 1]

#### `partition(n, coll)`

Partition coll into consecutive vectors of exactly n elements.
A trailing group with fewer than n elements is dropped.
partition(2, [1, 2, 3, 4, 5]) // => [[1, 2], [3, 4]]
n must be >= 1; for n <= 0 there is no valid chunk size, so [] is returned.

#### `partition-all(n, coll)`

Like partition but keep the trailing partial group.
partition-all(2, [1, 2, 3, 4, 5]) // => [[1, 2], [3, 4], [5]]
n must be >= 1; for n <= 0 there is no valid chunk size, so [] is returned.

#### `windowed(n, step, coll)`

Sliding windows of size n over coll, advancing the start by step each time.
Returns a vector of vectors. A trailing window with fewer than n elements is
dropped (only full windows are kept). n and step must each be >= 1.
windowed(2, 1, [1, 2, 3, 4]) // => [[1, 2], [2, 3], [3, 4]]
windowed(2, 2, [1, 2, 3, 4, 5]) // => [[1, 2], [3, 4]]
windowed(3, 1, [1, 2, 3, 4, 5]) // => [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

#### `chunk-by(f, coll)`

Split coll into runs of consecutive elements that share f's return value.
chunk-by(fn(x) { x % 2 }, [1, 3, 2, 4, 5]) // => [[1, 3], [2, 4], [5]]

#### `zip(a, b)`

Pair up elements from a and b. Stops at the shorter input.
zip([1, 2, 3], ["a", "b"]) // => [[1, "a"], [2, "b"]]

#### `zip-with(f, a, b)`

Combine a and b elementwise with f. Stops at the shorter input.
zip-with(fn(x, y) { x + y }, [1, 2, 3], [10, 20, 30]) // => [11, 22, 33]

#### `interpose(sep, coll)`

Insert sep between each element of coll.
interpose(0, [1, 2, 3]) // => [1, 0, 2, 0, 3]

#### `flatten(coll)`

Fully flatten an arbitrarily-nested vector of vectors into a flat vector.
Non-vector elements (including strings) are treated as leaves.
flatten([1, [2, [3, 4]], 5]) // => [1, 2, 3, 4, 5]

#### `iter-vector?(x)`

True when x is a PersistentVector. Used by flatten to decide whether to
recurse into a value (vectors are branches; everything else is a leaf).

#### `distinct(coll)`

Remove duplicate elements, keeping first occurrence order.
distinct([1, 2, 1, 3, 2]) // => [1, 2, 3]

#### `frequencies(coll)`

Count occurrences of each element. Returns a map element -> count.
frequencies([1, 1, 2, 3, 3, 3]) // => {1 2, 2 1, 3 3}

#### `group-by(f, coll)`

Group elements by f's return value. Returns a map key -> vector of elements.
group-by(fn(x) { x % 2 }, [1, 2, 3, 4]) // => {1 [1, 3], 0 [2, 4]}

#### `count-by(f, coll)`

Count elements by f's return value. Returns a map key -> count.
count-by(fn(x) { x % 2 }, [1, 2, 3, 4, 5]) // => {1 3, 0 2}

#### `repeat(n, x)`

A vector containing x repeated n times.
repeat(3, "a") // => ["a", "a", "a"]

#### `cycle-take(n, coll)`

Take n elements by cycling through coll repeatedly.
cycle-take(7, [1, 2, 3]) // => [1, 2, 3, 1, 2, 3, 1]

#### `enumerate(coll)`

Pair each element with its 0-based index.
enumerate(["a", "b", "c"]) // => [[0, "a"], [1, "b"], [2, "c"]]

#### `sum(coll)`

Sum of all elements (0 for empty).
sum([1, 2, 3, 4]) // => 10

#### `product(coll)`

Product of all elements (1 for empty).
product([1, 2, 3, 4]) // => 24

#### `min-by(f, coll)`

Element with the smallest f value (first such on ties); null if empty.
min-by(fn(s) { length(s) }, ["abc", "a", "ab"]) // => "a"

#### `max-by(f, coll)`

Element with the largest f value (first such on ties); null if empty.
max-by(fn(s) { length(s) }, ["a", "abc", "ab"]) // => "abc"

#### `accumulate(f, init, coll)`

Running fold prefixes of f over coll, starting from init. Alias of
`reductions` (the itertools.accumulate / reductions concept). The output
includes the initial value as its first element, then each successive
accumulation.
accumulate(fn(a, x) { a + x }, 0, [1, 2, 3]) // => [0, 1, 3, 6]

#### `cartesian-product(vec-of-vecs)`

Cartesian product of a vector of vectors. Returns a vector of tuples
(each tuple itself a vector), with the rightmost input varying fastest,
matching python's itertools.product.

NOTE ON NAMING: this is named `cartesian-product` rather than `product`
because `beagle.iter/product` already exists as the numeric product of a
collection (product([1,2,3,4]) => 24). Keeping both names preserves the
existing public API.

cartesian-product([[1, 2], [3, 4]]) // => [[1, 3], [1, 4], [2, 3], [2, 4]]
cartesian-product([]) // => [[]]            (empty product is one empty tuple)
cartesian-product([[1, 2], []]) // => []    (any empty factor => no tuples)

#### `combinations(coll, k)`

All k-element combinations of vec, in lexicographic index order, no repeats,
preserving the input order within each combination. Matches
itertools.combinations.
combinations([1, 2, 3, 4], 2) // => [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
combinations(x, 0) // => [[]]
combinations(x, k) where k > length(x) // => []

#### `combinations-from(v, n, k, start)`

Recursive helper for `combinations`: build all k-element combinations drawn
from indices start..n of v, in lexicographic index order, as element vectors.

#### `push-front-elem(x, vec)`

Prepend element x to vector vec, returning a new vector with x at index 0.

#### `combinations-with-replacement(coll, k)`

All k-element combinations of coll allowing repeated elements, in
lexicographic index order, preserving input order within each combination.
Matches itertools.combinations_with_replacement.
combinations-with-replacement([1, 2, 3], 2) // => [[1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3]]
combinations-with-replacement(x, 0) // => [[]]
combinations-with-replacement([], k) where k > 0 // => []

#### `combinations-wr-from(v, n, k, start)`

Recursive helper for `combinations-with-replacement`: build all k-element
combinations drawn from indices start..n of v, allowing each index to repeat,
in lexicographic index order, as element vectors.

#### `permutations-of(v, k)`

Helper for `permutations`: all k-length permutations of vector v in
itertools order. Throws if k < 0; returns [] when k > length(v).

#### `permutations-rec(v, n, k, used)`

Recursive helper for `permutations-of`: place each still-unused index in
turn (tracked by the boolean `used` vector) to build k-length element tuples.

<details>
<summary><strong>Undocumented functions (4)</strong></summary>

- `range$2(start, end)`
- `range$3(start, end, step)`
- `permutations$1(coll)`
- `permutations$2(coll, k)`

</details>

---

## beagle.json

> **Documentation coverage:** 37/39 functions (94%)

#### `cur-code(c)`

Char code at the cursor's current position, or -1 at end of input.

#### `at-end?(c)`

True when the cursor has consumed the entire source string.

#### `ws-code?(code)`

True when `code` is a JSON whitespace byte (space, tab, LF, or CR).

#### `skip-ws(c)`

Advance the cursor past any run of JSON whitespace.

#### `parse-error(c, msg)`

Throw a `beagle.json/parse` error tagged with `msg` and the cursor position.

#### `parse-document(c)`

Parse a complete JSON document (one value plus surrounding whitespace).
Throws if any non-whitespace remains after the value.

#### `parse-value(c)`

Parse a single JSON value at the cursor, dispatching on the leading byte
(object, array, string, true/false/null literal, or number).

#### `parse-literal(c, word, value)`

Match the exact keyword `word` (true/false/null) at the cursor and return
`value`; throws if the text does not match.

#### `parse-object(c)`

Parse a JSON object (cursor at the opening `{`) into a map. Keys become
keywords or stay strings depending on the cursor's `keyword-keys` flag.

#### `parse-array(c)`

Parse a JSON array (cursor at the opening `[`) into a vector.

#### `parse-string(c)`

Parse a JSON string (cursor at the opening quote), decoding escapes, into
a String. Throws on an unterminated string or bad escape.

#### `parse-escape(c, out)`

Decode the backslash escape at the cursor (`\"`, `\\`, `\/`, `\n`, `\t`,
`\r`, `\b`, `\f`, or `\uXXXX`) and append its byte(s) to `out`.

#### `parse-unicode-escape(c, out)`

Parse a `\uXXXX` escape (cursor at the `u`), combining a high/low surrogate
pair when present, and append the code point as UTF-8 to `out`.

#### `read-hex4(c)`

Read exactly 4 hex digits at the cursor and return their integer value;
throws on a short or non-hex sequence.

#### `hex-digit-value(code)`

Value (0-15) of a hex-digit char code (0-9, a-f, A-F), or -1 if not hex.

#### `encode-utf8(out, cp)`

Encode Unicode code point `cp` as 1-4 UTF-8 bytes appended to `out`.

#### `digit-code?(code)`

True when `code` is an ASCII digit '0'-'9'.

#### `parse-number(c)`

Scan a JSON number at the cursor, returning an Int when it has no fraction
or exponent and a Float otherwise. Throws on malformed numbers.

#### `parse-int-text(c, text)`

Parse an integer numeric substring into a tagged Int, degrading to a Float
approximation when the value overflows the 2^60-1 tagged-Int range.

#### `to-float-checked(c, text)`

Parse an already-scanner-validated float substring into a Float (thin
wrapper over parse-float-text).

#### `parse-float-text(s)`

Parse a validated numeric substring `s` into a Float via the native
correctly-rounded parser. Throws if the native parser unexpectedly rejects it.

#### `parse(s)`

Parse a JSON string into a Beagle value. Object keys become keywords.

Examples:
  parse("{\"a\":[1,2,3],\"b\":true,\"c\":null}")
    // => {:a [1, 2, 3], :b true, :c null}

#### `parse-with-string-keys(s)`

Parse a JSON string into a Beagle value, keeping object keys as Strings.

Examples:
  parse-with-string-keys("{\"a\":1}")  // => {"a" 1}

#### `parse-result(s)`

Parse a JSON string without throwing: returns Result.Ok { value } on
success, or Result.Err { error } carrying the parse error message instead
of throwing. Object keys become keywords, like `parse`.

Examples:
  parse-result("{\"a\":1}")  // => Result.Ok { value: {:a 1} }
  parse-result("{")          // => Result.Err { error: "beagle.json/parse: ..." }

#### `stringify(value)`

Serialize a Beagle value to compact JSON.

Examples:
  stringify({:a 1, :b [1, 2]})  // => {"a":1,"b":[1,2]}

#### `stringify-sorted(value)`

Serialize a Beagle value to compact JSON with object keys emitted in sorted
(canonical) order at every nesting level. Useful for stable, comparable
output such as hashing or diffing two equivalent values.

Examples:
  stringify-sorted({:b 1, :a 2})  // => {"a":2,"b":1}

#### `write-value-sorted(out, value, indent, depth)`

Write `value` as JSON into `out` like `write-value`, but recursing through
the sorted writers so object keys are emitted in sorted order at every depth.

#### `write-array-sorted(out, vec, indent, depth)`

Write vector `vec` as a JSON array into `out` like `write-array`, but
recursing through `write-value-sorted` so nested object keys stay sorted.

#### `write-object-sorted(out, m, indent, depth)`

Write map `m` as a JSON object into `out` like `write-object`, but with keys
emitted in sorted order (by their JSON string form) and values recursing
through `write-value-sorted`.

#### `write-value(out, value, indent, depth)`

Write `value` as JSON into builder `out`. `indent` < 0 means compact;
`depth` is the current nesting level. Throws on unserializable value types.

#### `write-newline-indent(out, indent, depth)`

In pretty mode (`indent` >= 0) write a newline plus `indent * depth` spaces;
a no-op in compact mode.

#### `write-array(out, vec, indent, depth)`

Write vector `vec` as a JSON array into `out`, honoring indent/depth.

#### `write-object(out, m, indent, depth)`

Write map `m` as a JSON object into `out`, honoring indent/depth; keys are
stringified via key->string.

#### `key->string(k)`

Convert a map key (keyword, string, or other) to its JSON string form.

#### `write-string(out, s)`

Write string `s` into `out` as a JSON string literal, including the
surrounding quotes and escaping of special and control characters.

#### `write-control-escape(out, code)`

Write control char `code` as a `\u00XX` escape sequence into `out`.

#### `hex-char(d)`

ASCII char code of the lowercase hex digit for `d` (0-15).

<details>
<summary><strong>Undocumented functions (2)</strong></summary>

- `stringify-pretty$1(value)`
- `stringify-pretty$2(value, indent)`

</details>

---

## beagle.log

> **Documentation coverage:** 22/22 functions (100%)

#### `level-name(level)`

Map a level int to its uppercase name ("DEBUG"/"INFO"/"WARN"/"ERROR").
Throws on an unknown level int.

#### `normalize-level(level)`

Normalize a level to its int, accepting either the LEVEL-* int constants
or the keywords :debug/:info/:warn/:error. Throws on anything else.

#### `set-level!(level)`

Set the minimum level that will be emitted. Accepts a keyword
(:debug/:info/:warn/:error) or one of the LEVEL-* int constants.
Messages below this level are suppressed.

#### `current-level()`

Return the current minimum level as an int.

#### `enabled?(level)`

True if a message at `level` would be emitted under the current min-level.

#### `format-fields(fields)`

Render a fields map as " key=value key2=value2 ..." (one leading space per
entry). Returns "" when `fields` is null or empty. Keyword keys are
rendered without the leading colon.

#### `format-fields-helper(fields, ks, idx, len, acc)`

Tail-recursive accumulator for `format-fields`: appends " key=value" for
each key in `ks` from `idx` up to `len`. Internal helper.

#### `field-key-string(k)`

Render a field key to the exact string used in output: keyword keys lose
their leading colon (via keyword->string), everything else goes through
to-string. Matches the rendering in `format-fields-helper`.

#### `format-fields-sorted(fields)`

Render a fields map as " key=value key2=value2 ..." with keys in ASCENDING
order (lexicographic by their rendered key string). Returns "" when
`fields` is null or empty. Unlike `format-fields`, the entry order is
deterministic regardless of map insertion order.

#### `format-line(level, msg)`

Build a log line WITHOUT a timestamp, for exact-assertion testing:
    "LEVEL msg"
`level` may be a keyword or int constant.

#### `format-line-fields(level, msg, fields)`

Build a structured log line WITHOUT a timestamp:
    "LEVEL msg key=value ..."

#### `format-line-fields-sorted(level, msg, fields)`

Build a structured log line WITHOUT a timestamp, with fields in ASCENDING
key order (deterministic regardless of map insertion order):
    "LEVEL msg key=value ..."
Used by `log-fields` for its (timestamped) output and for exact-assertion
testing of the sorted-field rendering.

#### `build-emit-line(level, msg, fields)`

Build the full line that gets written, including a live epoch-millis
timestamp: "LEVEL [<ts>] msg<fields>". `level` may be a keyword or int.

#### `build-emit-line-sorted(level, msg, fields)`

Build the full line that gets written, including a live epoch-millis
timestamp, with fields sorted in ascending key order:
"LEVEL [<ts>] msg key=value ...". `level` may be a keyword or int.

#### `emit-line!(lvl, line)`

Write a fully-built line to its sink based on `lvl`: debug/info to stdout
(println), warn/error to stderr (io/write-stderr with a trailing newline).
Returns null.

#### `log-at(level, msg, fields)`

Core logging entry point: emits the message (with timestamp and fields)
only if `level` is at or above the current min-level. Returns true if
emitted, false if suppressed.

#### `log(level, msg, fields)`

Structured log: log(level, msg, fields-map). Returns true if emitted.

#### `log-fields(level, msg, fields)`

Structured log at `level` whose `fields-map` is emitted as space-separated
key=value pairs in ASCENDING key order (deterministic, unlike `log`/`log-at`
which use map insertion order). The message is emitted as
"LEVEL [<ts>] msg key=value ...", respecting the module's current min-level:
suppressed (and returns false) when `level` is below it, otherwise emitted
(and returns true). debug/info go to stdout, warn/error to stderr.
`level` may be a keyword (:debug/:info/:warn/:error) or a LEVEL-* int.
A null or empty `fields-map` emits just "LEVEL [<ts>] msg".

#### `debug(msg)`

Log at debug level. Returns true if emitted (false if suppressed).

#### `info(msg)`

Log at info level. Returns true if emitted (false if suppressed).

#### `warn(msg)`

Log at warn level (stderr). Returns true if emitted.

#### `error(msg)`

Log at error level (stderr). Returns true if emitted.

---

## beagle.mathx

> **Documentation coverage:** 21/21 functions (100%)

#### `pi()`

Pi as a Float.

#### `e()`

Euler's number e as a Float.

#### `abs-int(n)`

Absolute value of an Int.

  abs-int(-5)  // => 5
  abs-int(5)   // => 5

#### `sign(n)`

Sign of a number: -1, 0, or 1.

  sign(-3)  // => -1
  sign(0)   // => 0
  sign(42)  // => 1

#### `gcd(a, b)`

Greatest common divisor of two integers (always non-negative).
Uses the Euclidean algorithm on absolute values.

  gcd(12, 18)  // => 6
  gcd(0, 5)    // => 5

#### `lcm(a, b)`

Least common multiple of two integers (always non-negative).
lcm(0, n) == 0.

  lcm(4, 6)  // => 12
  lcm(0, 5)  // => 0

#### `clamp(x, lo, hi)`

Clamp x into the inclusive range [lo, hi].

  clamp(5, 0, 3)   // => 3
  clamp(-1, 0, 3)  // => 0
  clamp(2, 0, 3)   // => 2

#### `lerp(a, b, t)`

Linear interpolation between a and b by fraction t (Float result).
t == 0 gives a, t == 1 gives b.

  lerp(0, 10, 0.5)  // => 5.0

#### `min-of(coll)`

Minimum element of a non-empty collection.
Empty input is a hard error (the minimum is undefined).

  min-of([3, 1, 4, 1, 5])  // => 1

#### `max-of(coll)`

Maximum element of a non-empty collection.
Empty input is a hard error (the maximum is undefined).

  max-of([3, 1, 4, 1, 5])  // => 5

#### `is-even?(n)`

True if n is even.

  is-even?(4)  // => true
  is-even?(3)  // => false

#### `is-odd?(n)`

True if n is odd.

  is-odd?(3)  // => true
  is-odd?(4)  // => false

#### `factorial(n)`

Factorial of a non-negative Int. Negative input is a hard error.

  factorial(0)  // => 1
  factorial(5)  // => 120

#### `pow-int(base, exp)`

Integer exponentiation: base raised to a non-negative integer exp.
Returns an Int. Negative exponents are a hard error (would not be an Int).
Uses exponentiation by squaring.

  pow-int(2, 10)  // => 1024
  pow-int(3, 0)   // => 1
  pow-int(5, 3)   // => 125

#### `deg->rad(d)`

Convert degrees to radians (Float).

  deg->rad(180)  // => 3.141592653589793

#### `rad->deg(r)`

Convert radians to degrees (Float).

  rad->deg(pi())  // => 180.0

#### `floor-div(a, b)`

True floor division (rounds toward negative infinity).

  floor-div(7, 3)    // => 2
  floor-div(-1, 3)   // => -1
  floor-div(-7, 3)   // => -3
  floor-div(7, -3)   // => -3

#### `mathx-float?(x)`

True if x is a Float (type-compared against a float literal).

#### `floor-mod(a, b)`

True floor modulo: result has the same sign as the divisor b.

  floor-mod(7, 3)    // => 1
  floor-mod(-1, 3)   // => 2
  floor-mod(-7, 3)   // => 2
  floor-mod(7, -3)   // => -2

#### `log-base(x, base)`

Logarithm of x in the given base, as a Float.
Computed as the natural log of x divided by the natural log of base.
x must be strictly positive; base must be strictly positive and not 1
(a base of 1 has no logarithm — the natural log of 1 is 0, so the ratio
would divide by zero). All three are hard errors.

  log-base(8, 2)     // => 3.0
  log-base(100, 10)  // => 2.0
  log-base(1, 10)    // => 0.0

#### `round-to(x, places)`

Round a Float to `places` decimal places, returning a Float.
`places` must be a non-negative Int (a hard error otherwise). Rounding is
half-away-from-zero, matching beagle.core's `round`.

  round-to(3.14159, 2)  // => 3.14
  round-to(2.5, 0)      // => 3.0
  round-to(-2.5, 0)     // => -3.0

---

## beagle.mutable-array

> **Documentation coverage:** 19/21 functions (90%)

#### `allocate-array-unsafe(size)`

Allocate a raw object of `size` fields with every field set to `null`.
Internal helper: the result has no array type-id yet, so it is not a
valid array until `new-array` tags it. Returns the allocated object.

#### `read-field-unsafe(array, index)`

Read field `index` directly via `primitive/read-field` with no type or
bounds checking. Caller must guarantee `array` is an array and `index`
is in range.

#### `write-field-unsafe(array, index, value)`

Write `value` to field `index` directly via `primitive/write-field` with
no type or bounds checking. Caller must guarantee `array` is an array and
`index` is in range.

#### `is-array(array)`

Return true when `array` is a heap object carrying the array type-id (1),
i.e. one produced by `new-array`. Returns false for any other value.

#### `panic-if-not-array(message, array)`

Guard that `array` is a real array; otherwise print `message` and the
offending value, then throw "Not an array". Returns null when the check
passes.

#### `write-field(array, index, value)`

Bounds- and type-checked write of `value` at `index`. Throws if `array`
is not an array, and throws a (resumable) IndexError when `index` is out of
range — a silently-dropped write is a programmer error, not a no-op.

#### `write-field-or-ignore(array, index, value)`

Like `write-field` but silently ignores an out-of-range `index` (returns
null). Use only when an out-of-bounds write is genuinely expected/benign.

#### `read-field(array, index)`

Bounds- and type-checked read at `index`. Throws if `array` is not an
array; returns null when `index` is out of range, otherwise the element.

#### `new-array(size)`

Create a new array of `size` null-initialized elements and tag it with
the array type-id. Throws "Negative size" when `size < 0`.

#### `copy-array(array)`

Return a new array that is a shallow copy of `array`. Throws if `array`
is not an array (and, redundantly, if it is null).

#### `unsafe-copy-from-array-to(from, to)`

Copy the contents of `from` into `to` via `builtin/copy-from-to-object`
with no type or size validation. Internal helper for `copy-from-array-to`.

#### `copy-from-array-to(from, to)`

Copy all of `from`'s elements into `to`. Both must be arrays; throws if
`from` is larger than `to`.

#### `copy-range(from, to, start, count)`

Copy `count` elements starting at index `start` from `from` into `to`
via `builtin/copy-array-range`. Both arguments must be arrays.

#### `allocate-array-and-return()`

Demo helper: build a length-1 array holding a sample `ExampleStruct` and
return it. Used by `main`.

#### `count(array)`

Return the number of elements in `array`, or 0 when `array` is null.
Throws if a non-null `array` is not actually an array.

#### `mod(a, b)`

Integer remainder of `a` divided by `b`, computed as `a - (a / b) * b`.

#### `fill(array, value)`

Set every slot of `array` to `value` in place and return `array`.
Throws if `array` is not an array.

#### `index-of(array, value)`

Return the index of the first element structurally equal to `value`
(compared with `equal`), or -1 when no element matches.
Throws if `array` is not an array.

#### `to-vec(array)`

Materialize `array` into a new persistent vector holding its elements in
order (nulls preserved). Throws if `array` is not an array.

<details>
<summary><strong>Undocumented functions (2)</strong></summary>

- `get(array, index)`
- `swap(array, i, j)`

</details>

---

## beagle.os

> **Documentation coverage:** 15/15 functions (100%)

#### `getpid()`

Return the current process id.

Examples:
  getpid()  // => 90718

#### `getppid()`

Return the parent process id.

Examples:
  getppid()  // => 90715

#### `getuid()`

Return the real user id of the calling process.

Examples:
  getuid()  // => 501

#### `getcwd()`

Return the current working directory as a string.

Allocates a buffer, calls getcwd, then reads back the NUL-terminated path.
Throws if getcwd fails (returns NULL — e.g. path longer than the buffer).

Examples:
  getcwd()  // => "/Users/me/project"

#### `getenv(name)`

Look up an environment variable by name.

Returns the value string, or null if the variable is not set.

Examples:
  getenv("HOME")          // => "/Users/me"
  getenv("NOPE_NOT_SET")  // => null

#### `setenv(name, value)`

Set an environment variable, overwriting any existing value.

Returns true on success, false on failure.

Examples:
  setenv("FOO", "bar")  // => true

#### `unsetenv(name)`

Remove an environment variable.

Returns true on success, false on failure. Removing an unset variable
still succeeds.

Examples:
  unsetenv("FOO")  // => true

#### `hostname()`

Return the system hostname as a string.

Allocates a buffer, calls gethostname, then reads back the NUL-terminated
name. Throws if gethostname reports an error.

Examples:
  hostname()  // => "mymachine.local"

#### `process-exit(code)`

Terminate the process immediately with the given exit code.

Calls C exit(3): runs atexit handlers and flushes stdio, then never
returns. Do not call this in a test you expect to continue.

Examples:
  process-exit(0)  // clean exit
  process-exit(1)  // failure exit

#### `temp-dir()`

Return the directory used for temporary files.

Reads the TMPDIR environment variable (via getenv); when it is unset,
falls back to the POSIX default "/tmp". The returned value is whatever
TMPDIR holds, verbatim (it may or may not have a trailing slash).

Examples:
  temp-dir()  // => "/var/folders/xy/.../T/"  (TMPDIR set)
  temp-dir()  // => "/tmp"                     (TMPDIR unset)

#### `path-separator()`

Return the POSIX path component separator, "/".

Beagle's supported platforms (macOS, Linux) are all POSIX, so this is a
constant. It does not consult the running platform.

Examples:
  path-separator()  // => "/"

#### `line-separator()`

Return the POSIX line separator, "\n".

Beagle's supported platforms (macOS, Linux) use a single line-feed, so
this is a constant. It does not consult the running platform.

Examples:
  line-separator()  // => "\n"

#### `is-macos?()`

Return true when running on macOS.

Built on the runtime's platform name from get-os(), which reports
"macos" on Apple platforms.

Examples:
  is-macos?()  // => true on macOS

#### `is-linux?()`

Return true when running on Linux.

Built on the runtime's platform name from get-os(), which reports
"linux" on Linux platforms.

Examples:
  is-linux?()  // => true on Linux

#### `is-windows?()`

Return true when running on Windows.

Built on the runtime's platform name from get-os(), which reports
"windows" on Windows. Note: Beagle only supports macOS and Linux, so in
practice this predicate is always false; it exists for completeness.

Examples:
  is-windows?()  // => false on macOS/Linux

---

## beagle.path

> **Documentation coverage:** 23/23 functions (100%)

#### `slash-at?(s, i)`

True if the byte at index i of string s is a '/' (slash) character.

#### `is-absolute?(p)`

True if path p is absolute, i.e. begins with '/'. The empty path is not absolute.

#### `strip-trailing-slashes(p)`

Remove trailing slashes from p, but keep a single "/" if p was all slashes.

#### `last-slash-index(p, end)`

Index of the last '/' in p within the range [0, end), or -1 if none is found.

#### `basename(p)`

Final path component of p, ignoring trailing slashes. Returns "/" for "/" and "" for the empty path.

#### `dirname(p)`

Directory portion of p (everything before the final component), trailing slashes ignored. Returns "." when p has no directory and "/" for the root.

#### `join(a, b)`

Join two segments with exactly one '/'. An absolute b (leading '/') overrides a; an empty segment falls back to the other.

#### `join-all(parts)`

Fold a vector of segments left-to-right with join. Returns "" for an empty vector.

#### `all-dots-before?(s, end)`

True if every char of s in [0, end) is '.'; used to reject all-dot names ("..", "...") as having an extension.

#### `ext-dot-index(p)`

Index *within basename(p)* of the '.' that separates the extension, or -1 if
none (leading-dot dotfiles and all-dot names excluded). Note: the returned
offset is relative to the basename, not to the full path p.

#### `extension(p)`

File extension of p including the leading dot, or "" if none. Dotfiles like ".bashrc" and all-dot names have no extension.

#### `strip-extension(p)`

p with its final extension removed; returns p unchanged when there is no extension.

#### `split-path(p)`

Split p into its non-empty components, dropping leading, duplicate, and trailing slashes. Use is-absolute? to recover absoluteness.

#### `rebuild-path(components, absolute)`

Join components back into a path, prefixing "/" when absolute. Empty components yield "/" (absolute) or "." (relative).

#### `normalize(p)`

Resolve '.' and '..' and collapse duplicate slashes. Drops '..' above an absolute root, keeps a leading '..' for relative paths; the empty path normalizes to ".".

#### `with-extension(p, ext)`

p with its final extension replaced by ext (given as ".md" or "md"); an empty
ext removes the extension. Throws if ext contains '/' or is all dots.

#### `is-relative?(p)`

True if path p is relative, i.e. does not begin with '/'. The empty path is relative.

#### `parts-of-normalized(n)`

Component vector of an already-normalized path string, treating "." as having no components.

#### `common-prefix-len(a, b)`

Number of leading components a and b share (by value equality).

#### `relative-to(p, base)`

p expressed relative to base (POSIX, using ".."), after normalizing both.
Throws if one path is absolute and the other relative, or if base escapes above its root.

#### `components(p)`

Vector of p's components; an absolute path gets a leading "/" element so
absoluteness is preserved. Duplicate and trailing slashes are dropped.

#### `is-hidden?(p)`

True when basename(p) begins with a '.' (a POSIX dotfile). The empty path is
not hidden; note "." and ".." count as hidden by this rule.

#### `strip-prefix(p, prefix)`

p with the leading directory prefix removed, as a relative path ("." when p
equals prefix). Throws if p is not under prefix or their absoluteness differs.

---

## beagle.priorityqueue

> **Documentation coverage:** 18/18 functions (100%)

#### `default-compare(a, b)`

Default min-heap comparator. Returns -1 when a < b, 1 when a > b, 0 when
equal. Works for numbers and (lexicographically) strings.

#### `pq-new()`

Create an empty min-heap priority queue using the default comparator.

Examples:
  pq-new()  // => empty min-heap

#### `pq-new-with(compare-fn)`

Create an empty priority queue ordered by a custom comparator.

`compare(a, b)` must return a negative number when `a` should be popped
before `b`, zero when equivalent, positive otherwise.

Examples:
  pq-new-with(fn(a, b) { b - a })  // => empty max-heap (numbers)

#### `pq-new-max()`

Create an empty MAX-heap priority queue (largest element pops first).

Uses the default ordering, inverted.

Examples:
  pq-new-max()  // => empty max-heap

#### `pq-count(pq)`

Return the number of items in the queue.

Examples:
  pq-count(pq-push(pq-new(), 5))  // => 1

#### `pq-empty?(pq)`

Return true when the queue has no items.

Examples:
  pq-empty?(pq-new())  // => true

#### `pq-swap(v, i, j)`

Swap the elements at indices `i` and `j` of vector `v`, returning a new
vector (the input is unchanged). Internal heap helper.

#### `pq-sift-up(v, idx, compare)`

Restore the heap upward: bubble the element at `idx` toward the root while
it compares before its parent under `compare`. Returns a new heap vector.
Internal helper.

#### `pq-sift-down(v, idx, compare)`

Restore the heap downward: push the element at `idx` down, swapping with
the smaller (compare-before) child while that child compares before it.
Returns a new heap vector. Internal helper.

#### `pq-push(pq, item)`

Insert an item into the queue, returning a NEW priority queue.

The original queue is unchanged.

Examples:
  pq-peek(pq-push(pq-push(pq-new(), 5), 3))  // => 3

#### `pq-peek(pq)`

Return the minimum item (under the comparator) WITHOUT removing it.

Returns null when the queue is empty. (Use pq-empty? to distinguish an
empty queue from a queue whose root is the value null — though null
elements are not recommended since maps/ordering treat them poorly.)

Examples:
  pq-peek(pq-new())  // => null

#### `pq-pop(pq)`

Remove and return the minimum item.

Returns a two-element vector [min-item, new-pq]. The new queue has the
root removed and the heap invariant restored. Throws when the queue is
empty (peek + empty? let you check first).

Examples:
  let [m, rest] = pq-pop(pq-push(pq-new(), 7))  // m => 7, rest empty

#### `pq-from-vec(v)`

Build a priority queue from a vector (min-heap, default comparator).

Equivalent to pushing every element, but uses Floyd's bottom-up heapify
(O(n) instead of O(n log n)).

Examples:
  pq-peek(pq-from-vec([5, 3, 8, 1, 9, 2]))  // => 1

#### `pq-from-vec-with(v, compare)`

Build a priority queue from a vector using a custom comparator.

Uses Floyd's bottom-up heapify.

#### `peek(pq)`

Return the minimum item (under the comparator) WITHOUT removing it.

Convenience alias for pq-peek. Returns null when the queue is empty.

Examples:
  peek(pq-from-vec([5, 3, 8, 1, 9, 2]))  // => 1
  peek(pq-new())                     // => null

#### `pq-size(pq)`

Return the number of items in the queue.

Convenience alias for pq-count.

Examples:
  pq-size(pq-push(pq-new(), 5))  // => 1
  pq-size(pq-new())              // => 0

#### `from-vec(vec)`

Build a priority queue from a vector (min-heap, default comparator).

Convenience alias for pq-from-vec. Inserts every element of `vec` using
Floyd's bottom-up heapify (O(n)).

Examples:
  peek(from-vec([5, 3, 8, 1, 9, 2]))  // => 1

#### `pq-to-sorted-vec(pq)`

Drain a priority queue into a sorted vector (smallest first under the
comparator). Does not mutate the input.

Examples:
  pq-to-sorted-vec(pq-from-vec([5, 3, 8, 1, 9, 2]))  // => [1, 2, 3, 5, 8, 9]

---

## beagle.process

> **Documentation coverage:** 8/8 functions (100%)

#### `get-libc-path()`

Return the platform's libc shared-library name to load via FFI.

"libSystem.B.dylib" on macos, "libc.so.6" on linux, else a bare "libc".

#### `is-null-ptr(p)`

True if the FFI pointer `p` is null (its low word is 0).

#### `run-capture(cmd)`

Run `cmd` through the shell and return everything it writes to stdout.

The command runs via popen(cmd, "r"); stdout is read in full and returned
as a String. stderr is not captured (append "2>&1" if you want it).
The child's exit status is discarded — see `system` if you need it.

Examples:
  run-capture("echo hello")        // => "hello\n"
  run-capture("printf 'a\nb'")     // => "a\nb"

#### `read-all(handle)`

Read the entire FILE* stream `handle` into a String via chunked fread.

Loops reading CHUNK-SIZE bytes into an FFI buffer until fread returns 0
(the only reliable pipe EOF), copying each chunk length-based so all byte
values round-trip. Frees the buffer before returning.

#### `run-capture-trim(cmd)`

Like run-capture but trims leading/trailing whitespace from the result.

Handy for commands whose output ends in a newline you do not want, e.g.
`pwd`, `whoami`, `git rev-parse HEAD`.

Examples:
  run-capture-trim("echo hello")   // => "hello"

#### `system(cmd)`

Run `cmd` through the shell and return the raw status from C system().

This is the wait4-style status value, NOT a clean exit code. For a normal
exit the low byte is the signal/termination info and the next byte is the
exit code (status >> 8 & 0xFF). A command that exits 0 yields a status of 0,
so `(system "true") == 0` while `(system "false")` is non-zero.

Use `exit-status` to decode the actual exit code.

Examples:
  system("true")           // => 0
  system("exit 3")         // => 768   (3 << 8)
  exit-status(system("exit 3"))  // => 3

#### `exit-status(status)`

Decode the exit code from a raw status returned by `system` or pclose.

On POSIX, WEXITSTATUS(status) == (status >> 8) & 0xFF for a normally-exited
child. Returns -1 if the status indicates the child did not exit normally
(was killed by a signal — low 7 bits non-zero).

Examples:
  exit-status(system("exit 5"))  // => 5
  exit-status(0)                  // => 0

#### `success?(cmd)`

Convenience: true if `system(cmd)` reported a clean zero exit.

Examples:
  success?("true")   // => true
  success?("false")  // => false

---

## beagle.random

> **Documentation coverage:** 14/16 functions (87%)

#### `normalize-seed(seed)`

Normalize an integer seed to a non-zero 32-bit state, mapping 0 to a fixed
constant so the xorshift generator does not degenerate to all-zeros.

#### `xorshift-step(x)`

Advance the 32-bit xorshift state once and return the new state.

#### `next-int(rng)`

Return the next raw 32-bit value (0 .. 2^32-1) and advance the generator.

#### `next-float(rng)`

Return a uniform Float in [0, 1) and advance the generator.

#### `int-range(rng, lo, hi)`

Return a uniform Int in [lo, hi) and advance the generator. Throws if hi <= lo.

#### `choice(rng, v)`

Return a uniformly-random element of non-empty vector v. Throws if v is empty.

#### `shuffle(rng, v)`

Return a new immutable vector that is a uniformly-random permutation of v
(Fisher-Yates). Vectors of length <= 1 are returned unchanged.

#### `sample(rng, v, k)`

Return a new vector of k distinct elements drawn from v without replacement
(partial Fisher-Yates). Throws if k < 0 or k > length(v).

#### `builtin-allocate-filled(size)`

Allocate a raw mutable array (type_id 1) of the given size, filled with null.
Internal scratch-array helper.

#### `primitive-read(arr, i)`

Read element i from a raw mutable array. Thin wrapper over primitive/read-field.

#### `primitive-write(arr, i, val)`

Write val to element i of a raw mutable array. Thin wrapper over primitive/write-field.

#### `nibble-to-code(n)`

Map a nibble (0..15) to its lowercase hex character code.

#### `append-random-nibbles!(b, rng, count)`

Append `count` random lowercase hex nibbles to string-builder b, drawing
fresh 32-bit words as needed (8 nibbles per word). Returns b.

#### `uuid-v4(rng)`

Generate an RFC 4122 version-4 UUID string in 8-4-4-4-12 form, with the
version nibble forced to '4' and the variant nibble to one of 8/9/a/b.

<details>
<summary><strong>Undocumented functions (2)</strong></summary>

- `make-rng$0()`
- `make-rng$1(seed)`

</details>

---

## beagle.regex-wrapper

> **Documentation coverage:** 14/14 functions (100%)

#### `as-regex(pattern-or-regex)`

Normalize a pattern-or-regex argument into a compiled regex: returns it
unchanged if already a regex, otherwise compiles the pattern string.

#### `compile(pattern)`

Compile a pattern string into a reusable regex value. Thin wrapper over
regex/compile; throws (RegexError) on an invalid pattern.

#### `regex?(value)`

True if `value` is a compiled regex.

#### `match?(pattern, s)`

True if `pattern` matches anywhere in `s` (unanchored).

#### `matches-full?(pattern, s)`

True only if `pattern` matches the ENTIRE string `s` (anchored with
^(?:...)$). Requires a pattern STRING; throws if given a compiled regex,
which cannot be re-anchored.

#### `find(pattern, s)`

First match in `s` as the matched string, or null if there is no match.

#### `find-all(pattern, s)`

Vector of every non-overlapping match string in `s`, in order (empty if
none). Includes empty matches (Python re.findall semantics) by mapping
find-all-positions to its :match strings.

#### `find-all-positions(pattern, s)`

Vector of match descriptors {:match <string> :start <int> :end <int>}
(:end exclusive) for every non-overlapping match; offsets index into the
original string. Empty matches are emitted and advance one position (RE2
semantics). Marker-byte-free, so any input bytes are safe.

#### `split(pattern, s)`

Split `s` on `pattern`, returning the vector of between-match segments.
Thin wrapper over the native builtin (leading/trailing matches yield empty
segments at the ends).

#### `captures(pattern, s)`

Capture groups of the FIRST match as a positional vector (index 0 = whole
match, 1.. = parenthesized groups), or null if no match. A group that did
not participate is null at its position. Thin wrapper over the native builtin.

#### `replace-all(pattern, s, replacement)`

Replace ALL non-overlapping matches in `s` with `replacement` (supports
$1 / ${name} group references). Thin wrapper over the native builtin.

#### `replace-first(pattern, s, replacement)`

Replace only the FIRST match in `s` with `replacement`. Thin wrapper over
the native builtin.

#### `replace-with(pattern, s, f)`

Replace every match in `s` with the result of calling `f` on the matched
substring. Built on find-all-positions, so it is exact for empty/adjacent
matches and safe on any input bytes.

#### `replace-all-re(pattern, s, replacement)`

Replace every non-overlapping match in `s` with the LITERAL `replacement`
string (no $1 / ${name} group-reference interpretation — `$` is kept
verbatim). Built on find-all-positions via replace-with, so it is exact for
empty/adjacent matches and safe on any input bytes.

---

## beagle.repl

> **Documentation coverage:** 20/20 functions (100%)

#### `register-main-crash-atom(a)`

Register the atom that repl-main uses to publish a main-thread crash.
`a` becomes the value held by `main_crash_atom` (a holder atom of {:error, :resume_atom}).

#### `get-or-create-session(session_id)`

Return the existing session for `session_id`, or create, register, and return a new one.

#### `remove-session(session_id)`

Remove `session_id` from the global session registry.

#### `send-response(client, response)`

JSON-encode `response` and write it as a newline-terminated line to the client socket.

#### `parse-json-line(line)`

Decode a single JSON `line`, returning the parsed value or null if it fails to parse.

#### `handle-eval(client, msg)`

Handle an "eval" op: evaluate `code` in the session named by `msg`, blocking until the
result atom is filled, then stream each response back to the client. Sends an error
response if the session, id, or code fields are missing.

#### `handle-resume(client, msg)`

Handle a "resume" op: resume a suspended computation in the named session with `code`
(defaulting to "null"), streaming responses back. Errors if session/id are missing or
the session is not found.

#### `handle-abort(client, msg)`

Handle an "abort" op: abort the in-flight computation in the named session and stream
responses back. Errors if the session field is missing or the session is not found.

#### `handle-interrupt(client, msg)`

Handle an "interrupt" op: signal the named session to interrupt and reply with a "done"
status. Errors if the session field is missing or the session is not found.

#### `handle-main-status(client, msg)`

Handle a "main-status" op: report whether the main thread is "running" or "suspended"
(after a crash), including the crash error when suspended.

#### `handle-main-resume(client, msg)`

Handle a "main-resume" op: resume a crashed main thread by writing {:op "resume", :code}
to its resume atom. Errors if there is no main-thread crash to resume.

#### `handle-main-abort(client, msg)`

Handle a "main-abort" op: abort a crashed main thread by writing {:op "abort"} to its
resume atom. Errors if there is no main-thread crash to abort.

#### `handle-describe(client, msg)`

Handle a "describe" op: reply with the supported ops and beagle/protocol version info.

#### `handle-ls-sessions(client, msg)`

Handle an "ls-sessions" op: reply with the list of currently registered session ids.

#### `handle-close(client, msg)`

Handle a "close" op: close and unregister the named session (if present) and reply with
a "done" status. Errors if the session field is missing.

#### `handle-message(client, msg)`

Dispatch a decoded request `msg` to the matching op handler based on its "op" field.
Sends an error response if "op" is missing or unrecognized.

#### `extract-lines(data, buffer)`

Split `buffer ++ data` on newlines, returning {:lines complete-lines, :buffer leftover}
where `leftover` is the trailing partial line to carry into the next read.

#### `handle-client(client)`

Read/parse loop for one connected client: accumulate socket data, decode each JSON line,
dispatch via handle-message, and close the socket when the peer disconnects or errors.

#### `start-repl-server(host, port)`

Start the REPL server on `host`:`port` and run its accept loop forever (production entry).

#### `start-repl-server-stoppable(host, port, stop-atom, ready-atom, quiet)`

Start the REPL server with cooperative shutdown for tests/embedders: listen on
`host`:`port`, set `ready-atom` true once listening, serve clients sequentially, and exit
the accept loop after a session when `stop-atom` is true. `quiet` suppresses stdout logging.

---

## beagle.repl-interactive

> **Documentation coverage:** 4/4 functions (100%)

#### `main()`

Entry point for the interactive REPL: switches to the "user" namespace,
prints the banner, runs the read-eval-print loop until `:repl-exit` (or EOF),
then saves the input history.

#### `repl-loop()`

Reads one line at the namespace prompt and dispatches it: EOF exits, Ctrl-C
re-prompts, `:`-prefixed lines run commands, otherwise the input is `eval`d
(printing `=> result` or, on error, entering the resume prompt). Recurses
until EOF or a `:repl-exit` throw.

#### `resume-prompt(resume)`

Prompts the user for a value to resume a thrown exception with. `:abort`/`:a`
or EOF aborts (returns null); an empty line re-prompts; otherwise the input
is `eval`d and passed to `resume`, re-prompting if that evaluation errors.

#### `handle-command(input)`

Parses and executes a `:`-prefixed REPL command from `input` (command plus
optional argument): `:quit`/`:q`/`:exit` throws `:repl-exit`, `:help` lists
commands, `:clear` clears the screen, `:ns [name]` shows or switches the
namespace; unknown commands print a hint.

---

## beagle.repl-main

> **Documentation coverage:** 15/15 functions (100%)

#### `process-eval-queue()`

Process pending eval requests on the main thread.

Call this in your game/render loop. It drains the eval queue
and executes each request on the calling thread (which should
be the main thread for GUI framework compatibility).

This is safe to call when the queue is empty (no-op).

#### `process-requests(requests, i)`

Recursively process the drained request list starting at index `i`,
calling process-single-request on each.

#### `process-single-request(request)`

Evaluate one request's code with output redirected to a string buffer,
then set its result_atom to an EvalResult capturing the output and either
the stringified value or a caught error.

#### `send-response(client, response)`

JSON-encode `response` and write it to the client socket followed by a newline.

#### `handle-eval(client, msg)`

Handle an "eval" op: validate the session/id/code fields, queue the code
for the main thread, poll until it finishes, then stream out, value or
exception responses back over the client socket.

#### `handle-interrupt(client, msg)`

Handle an "interrupt" op by replying with a "done" status (no-op interrupt).

#### `handle-describe(client, msg)`

Handle a "describe" op by replying with the supported ops and version info.

#### `handle-ls-sessions(client, msg)`

Handle an "ls-sessions" op by replying with an empty session list and "done".

#### `handle-close(client, msg)`

Handle a "close" op by replying with a "done" status.

#### `handle-message(client, msg)`

Dispatch a decoded message to the matching op handler based on its "op"
field; replies with an error response for a missing or unknown op.

#### `extract-lines(data, buffer)`

Split buffered socket data on newlines into complete lines, returning
{:lines, :buffer} where :buffer holds the trailing incomplete fragment
to carry over to the next read.

#### `handle-client(client)`

Read-loop for a connected client: accumulate data, decode each newline-
delimited JSON message and dispatch it, until the connection closes or
errors; closes the socket on exit.

#### `start(host, port)`

Start the REPL server on a background thread.

The server accepts connections and handles the JSON protocol,
but all eval requests are queued for main thread processing.
Call process-eval-queue() in your main loop to execute them.

#### `run-with-repl(host, port, main-fn)`

Register the crash-recovery atom, start the regular REPL server on a
background thread, then run main-fn on the main thread under crash recovery.

#### `run-with-recovery(main-fn)`

Run main-fn, and if it throws, publish the error to main_thread_crash and
block waiting for a REPL "resume" (optionally evaluating fix-up code and
resuming the continuation) or "abort" (returning to let main-fn exit).

---

## beagle.repl-session

> **Documentation coverage:** 13/13 functions (100%)

#### `create-session(id)`

Create and start a new REPL `Session` with the given `id`, spawning its
eval thread. The session inherits the namespace current at creation time
and isolates it thereafter. Returns the `Session`.

#### `rest-or-empty(items)`

Return all but the first of `items`, or `[]` when there is no tail.

#### `pop-or-empty(items)`

Return `items` without its last element, or `[]` when nothing remains.

#### `session-eval-loop(queue, running, cancel, resume_stack, ns)`

The eval thread's main loop: until `cancel` is set, dequeue and process one
request at a time from `queue`, toggling the `running` flag and sleeping
when the queue is empty.

#### `build-responses(req_id, buf, result)`

Build response list for a successful eval result.

#### `process-eval-request(request, cancel, queue, running, resume_stack, ns)`

Process eval with resumable exception support.

#### `session-eval(session, request_id, code)`

Enqueue `code` for evaluation on `session` under `request_id`. Returns the
result atom that will be filled with the response list once processed.

#### `session-resume(session, request_id, code)`

Resume the topmost suspended exception on `session` with the value of
`code`, under `request_id`. Returns a result atom holding an error response
when there is no suspended exception to resume.

#### `session-abort(session, request_id)`

Abort the topmost suspended exception on `session` under `request_id`,
discarding it without resuming. Returns a result atom holding an error
response when there is no suspended exception to abort.

#### `session-interrupt(session)`

Interrupt `session`'s in-flight evaluation by briefly raising and clearing
its cancel token.

#### `session-close(session)`

Permanently stop `session` by setting its cancel token, ending the eval loop.

#### `session-busy?(session)`

Return whether `session` is currently evaluating a request.

#### `session-suspend-depth(session)`

Return the number of suspended (resumable) exceptions currently stacked on
`session`, or 0 when none.

---

## beagle.runtime

> **Documentation coverage:** 2/2 functions (100%)

#### `specialize-all()` `builtin`

Walk arithmetic-feedback and recompile every fully-monomorphic function with specialized variants. Returns the count of functions specialized.

Call after a warmup phase to populate feedback, before the measured phase. Subsequent calls re-evaluate in case feedback evolved.

#### `stop-the-world()` `builtin`

Drive a stop-the-world rendezvous from the calling thread: park every other mutator at its entry safepoint, observe the world is stopped, then resume. Returns the number of other threads that were parked. Diagnostic scaffolding for race-free tier-2 install application; set BEAGLE_STW_LOG=1 to trace it.

---

## beagle.semver

> **Documentation coverage:** 28/29 functions (96%)

#### `digit?(c)`

True if `c` is an ASCII digit (0-9).

#### `alpha?(c)`

True if `c` is an ASCII letter (A-Z or a-z).

#### `ident-char?(c)`

True if `c` is a legal prerelease/build identifier char ([0-9A-Za-z-]).

#### `all-digits?(s)`

True if `s` is non-empty and every character is an ASCII digit.

#### `all-ident-chars?(s)`

True if `s` is non-empty and every character is a legal identifier char
([0-9A-Za-z-]).

#### `valid-num-part?(s)`

True if `s` is a valid numeric version part: all digits with no leading
zero (unless the part is exactly "0").

#### `valid-prerelease?(s)`

True if `s` is a valid prerelease string: a non-empty, dot-separated series
of non-empty identifiers, each [0-9A-Za-z-], with no leading zeros on
numeric identifiers.

#### `valid-build?(s)`

True if `s` is valid build metadata: a non-empty, dot-separated series of
non-empty [0-9A-Za-z-] identifiers (leading zeros allowed).

#### `parse(s)`

Parse a SemVer 2.0.0 string into a SemVer struct, or null if invalid.

Examples:
  parse("1.2.3")               // => SemVer { 1, 2, 3, null, null }
  parse("1.2.3-alpha.1+build") // => SemVer { 1, 2, 3, "alpha.1", "build" }
  parse("1.2")                 // => null

#### `parse-validated(core, prerelease, build)`

Validate the already-split `core` ("major.minor.patch"), `prerelease`, and
`build` pieces and build a SemVer struct, or null if any part is invalid.

#### `valid?(s)`

True if `s` parses as a valid SemVer string.

#### `cmp-int(a, b)`

Compare two ints -> -1 if a<b, 0 if equal, 1 if a>b.

#### `cmp-string(a, b)`

Lexicographically compare two ASCII strings -> -1/0/1.

#### `cmp-pre-identifier(a, b)`

Compare two prerelease identifiers -> -1/0/1: numeric ones compare
numerically, alphanumeric ones lexically, and numeric ranks below
alphanumeric.

#### `cmp-prerelease(a, b)`

Compare two prerelease strings -> -1/0/1. Either may be null (a release),
which ranks HIGHER than any prerelease.

#### `cmp-pre-parts(a-parts, b-parts)`

Compare two lists of prerelease identifiers field-by-field -> -1/0/1; when
all shared fields are equal, the list with fewer fields ranks lower.

#### `compare-semver(a, b)`

Compare two SemVer structs by precedence -> -1/0/1 (major, then minor,
patch, then prerelease; `build` is ignored).

#### `compare(a, b)`

Compare two version strings -> -1/0/1 (precedence per SemVer spec).
Throws if either argument is not a valid SemVer string.

Examples:
  compare("1.0.0", "2.0.0")        // => -1
  compare("1.0.0-alpha", "1.0.0")  // => -1
  compare("1.0.0", "1.0.0")        // => 0

#### `caret-upper(base)`

Exclusive upper bound for a caret range over `base`: bumps the left-most
non-zero of major/minor/patch and zeroes the rest.

#### `tilde-upper(base)`

Exclusive upper bound for a tilde range over `base`: bumps minor and zeroes
patch (allows patch-level changes within the given minor).

#### `ge?(v, lower)`

True if SemVer `v` is >= `lower` by precedence.

#### `lt?(v, upper)`

True if SemVer `v` is < `upper` by precedence.

#### `same-tuple?(a, b)`

True if `a` and `b` share the same [major, minor, patch] tuple.

#### `prerelease-ok?(v, base)`

Apply the node-semver prerelease rule: a prerelease version `v` may match
only if the comparator `base` shares its tuple and also carries a
prerelease; stable versions always pass.

#### `satisfies?(version, range)`

Does `version` satisfy `range`?

Supported range forms:
  "1.2.3"   exact          (also "=1.2.3")
  "^1.2.0"  caret          (compatible-with)
  "~1.2.0"  tilde          (approximately-equivalent)
  ">=1.2.0" ">1.2.0" "<=1.2.0" "<1.2.0"  comparators

Throws if the version or range cannot be parsed.

Examples:
  satisfies?("1.2.5", "^1.2.0")  // => true
  satisfies?("2.0.0", "^1.2.0")  // => false

#### `require-parse(s, range)`

Parse `s` as a SemVer or throw, citing `range` in the error. Used to parse
the comparator operand inside a range expression.

#### `satisfies-parsed?(v, range)`

Test parsed version `v` against the (trimmed) `range` string, dispatching on
its leading operator (^, ~, >=, <=, >, <, =, or bare = exact match).

#### `range-and(version, ranges)`

True if `version` satisfies EVERY comparator string in the vector `ranges`
(logical AND). Each comparator uses the same forms as `satisfies?` (^, ~,
>=, <=, >, <, =, or a bare exact version). An empty `ranges` vector is
vacuously true. Throws if `version` or any comparator cannot be parsed.

Examples:
  range-and("1.2.5", [">=1.2.0", "<2.0.0"])  // => true
  range-and("2.0.0", [">=1.2.0", "<2.0.0"])  // => false

<details>
<summary><strong>Undocumented functions (1)</strong></summary>

- `SemVer_compare-to(a, b)`

</details>

---

## beagle.socket

> **Documentation coverage:** 8/8 functions (100%)

#### `listen(host, port)`

Create a TCP listener bound to `host` and `port`, returning a listener handle.

#### `connect(host, port)`

Open a TCP connection to `host`:`port`, returning a socket. Throws on connection failure.

#### `accept(listener)`

Block until a client connects to `listener`, returning the new socket. Throws on error.

#### `read(socket, n)`

Block reading up to `n` bytes from `socket`, returning the data read. Throws on error.

#### `write(socket, data)`

Block writing `data` to `socket`. Throws on error.

#### `close(socket)`

Close `socket`, releasing its underlying connection.

#### `close-listener(listener)`

Close `listener`, stopping it from accepting further connections.

#### `on-connection(listener, callback)`

Accept connections on `listener` forever, spawning a cooperative task that
runs `callback(conn)` for each one. Loops indefinitely; never returns normally.

---

## beagle.spawn

> **Documentation coverage:** 9/11 functions (81%)

#### `spawn(thunk)`

Perform the `Spawn` effect to run `thunk` as a task; returns a Future
resolving to its result. Must run inside a `handle Handler(Spawn)` block.

#### `spawn-with-token(thunk, token)`

Perform the `SpawnWithToken` effect to run `thunk` as a cancellable task
guarded by `token`; returns a Future. Must run inside a `Handler(Spawn)` block.

#### `handle-spawn-blocking(thunk)`

Run `thunk` synchronously, returning a Future already resolved with its
result or rejected with any thrown exception.

#### `handle-spawn-with-token-blocking(thunk, token)`

Run `thunk` synchronously unless `token` is cancelled before or after it
completes; returns a Future that is resolved, cancelled, or rejected.

#### `handle-spawn-threaded(thunk)`

Run `thunk` on a new thread, updating the returned Future's state atom to
Resolved with its result or Rejected with any thrown exception.

#### `handle-spawn-with-token-threaded(thunk, token)`

Run `thunk` on a new thread that checks `token` before and after execution,
settling the Future as Cancelled, Resolved, or Rejected and notifying waiters.

#### `run-blocking(thunk)`

Run `thunk` with a `BlockingSpawnHandler` installed so any `Spawn` effects
performed inside execute synchronously; returns the thunk's value.

#### `blocking-spawn(thunk)`

Spawn `thunk` under a blocking handler; returns its already-settled Future.

#### `blocking-spawn-with-token(thunk, token)`

Spawn `thunk` with cancellation `token` under a blocking handler; returns
its already-settled Future (Resolved, Cancelled, or Rejected).

<details>
<summary><strong>Undocumented functions (2)</strong></summary>

- `BlockingSpawnHandler_handle(self, op, resume)`
- `ThreadedSpawnHandler_handle(self, op, resume)`

</details>

---

## beagle.stats

> **Documentation coverage:** 18/18 functions (100%)

#### `as-vector(coll)`

Materialize any Seqable into a plain vector for indexing and reuse.
A PersistentVector is returned unchanged; anything else is reduced into a
new vector.

#### `to-float(x)`

Promote a number into the Float domain by multiplying by 1.0.
An Int becomes a Float; an existing Float is returned unchanged.

#### `require-non-empty(v, who)`

Throw a labeled "empty input" error if v is empty, otherwise return null.
`who` is the statistic name embedded in the message (e.g. "mean").

#### `sum(coll)`

Sum of all elements. Returns 0 for an empty collection (the additive
identity — summation is well-defined on the empty set).

  sum([1, 2, 3, 4])  // => 10

#### `mean(coll)`

Arithmetic mean (average) as a Float.
Throws on empty input.

  mean([1, 2, 3, 4])  // => 2.5

#### `min(coll)`

Smallest element. Throws on empty input.

  min([3, 1, 4, 1, 5])  // => 1

#### `max(coll)`

Largest element. Throws on empty input.

  max([3, 1, 4, 1, 5])  // => 5

#### `range-of(coll)`

Range: the spread between the largest and smallest element (max - min).
Throws on empty input.

  range-of([3, 1, 4, 1, 5])  // => 4

#### `median(coll)`

Median (middle value of the sorted data).
  * odd count  -> the single middle element (preserves its type)
  * even count -> the Float average of the two middle elements
Throws on empty input.

  median([1, 2, 3])     // => 2
  median([1, 2, 3, 4])  // => 2.5

#### `mode(coll)`

Mode: the most frequently occurring value. Ties are broken by the smallest
value (the data is sorted first, then the first value to reach the max run
length wins), so the result is deterministic. Throws on empty input.

  mode([1, 2, 2, 3, 3, 3])  // => 3
  mode([4, 4, 1, 1])        // => 1  (tie -> smallest)

#### `sum-squared-deviations(v)`

Sum of squared deviations of each element from the mean, as a Float.
Shared core used by the variance functions. Throws on empty input (via mean).

#### `variance(coll)`

Population variance: average squared deviation from the mean (divides by n).
Throws on empty input.

  variance([2, 4, 4, 4, 5, 5, 7, 9])  // => 4.0

#### `variance-sample(coll)`

Sample variance: divides by (n - 1) (Bessel's correction). Requires at
least two data points; throws otherwise.

  variance-sample([2, 4, 4, 4, 5, 5, 7, 9])  // => 4.571428571428571

#### `stdev(coll)`

Population standard deviation: sqrt of the population variance.
Throws on empty input.

  stdev([2, 4, 4, 4, 5, 5, 7, 9])  // => 2.0

#### `stdev-sample(coll)`

Sample standard deviation: sqrt of the sample variance. Needs >= 2 points.

  stdev-sample([2, 4, 4, 4, 5, 5, 7, 9])  // => 2.138...

#### `percentile(coll, p)`

Percentile via linear interpolation between closest ranks (the same
"linear" / type-7 method used by NumPy's default and Excel PERCENTILE).
  * p is in 0..100 (inclusive).
  * p == 0   -> minimum, p == 100 -> maximum.
Throws on empty input or when p is outside [0, 100].

  percentile([1,2,3,4,5,6,7,8,9,10], 50)  // => 5.5
  percentile([1,2,3,4],            0)      // => 1.0
  percentile([1,2,3,4],            100)    // => 4.0

#### `quantile(coll, q)`

Quantile via linear interpolation (NumPy type-7), with q expressed on a
0..1 scale instead of percentile's 0..100. Equivalent to
`percentile(coll, q * 100)`:
  * q == 0   -> minimum, q == 1 -> maximum.
  * q == 0.5 -> median (for the interpolating definition).
Throws on empty input or when q is outside [0, 1].

  quantile([1,2,3,4,5,6,7,8,9,10], 0.5)   // => 5.5
  quantile([1,2,3,4],              0)      // => 1.0
  quantile([1,2,3,4],              1)      // => 4.0
  quantile([1,2,3,4],              0.25)   // => 1.75

#### `stddev(coll)`

Population standard deviation (the `stddev` spelling of `stdev`): the square
root of the population variance. Throws on empty input.

  stddev([2, 4, 4, 4, 5, 5, 7, 9])  // => 2.0

---

## beagle.stream

> **Documentation coverage:** 41/86 functions (47%)

#### `from-source(source)`

Create a stream from any StreamSource implementation

This is the base constructor for all streams. Combinators wrap and
extend streams created by this function.

#### `ensure-closed(stream)`

Ensure cleanup happens exactly once, even if called multiple times

Uses atomic compare-and-swap to guarantee idempotency. Multiple threads
can safely call this; only one will actually close the resource.

#### `next(stream)`

Pull the next value from the stream

Returns StreamResult.Value { value }, StreamResult.Done {}, or
StreamResult.Error { error }.

Automatically calls close() if stream ends or errors occur.

#### `close(stream)`

Close the stream early, releasing resources

Safe to call multiple times. If already closed, does nothing.

#### `map(stream, f)`

Transform stream values with a function

The function is called for each value the source produces.
If the function throws, the error propagates to the consumer.

#### `filter(stream, pred)`

Keep only stream values that satisfy a predicate

The predicate function is called for each value. If it returns true,
the value is yielded; if false, the next value is fetched.
If the predicate throws, the error propagates.

#### `take(stream, n)`

Take the first n values from the stream

After yielding n values, returns Done and closes the stream.

#### `take-while(stream, pred)`

Take values while a predicate is true

Once the predicate returns false for a value, the stream stops
and the upstream is closed.

#### `skip(stream, n)`

Skip the first n values

Consumes and discards the first n values, then yields from the stream.

#### `drop-while(stream, pred)`

Drop the leading prefix of values for which the predicate holds

Discards values while pred(value) is true. The FIRST value for which pred
returns false — and every value after it — is yielded unchanged; the
predicate is never consulted again once dropping stops. This is the mirror
of take-while: drop-while keeps the tail that take-while discards.
If the predicate throws while dropping, the error becomes a
StreamResult.Error and the stream terminates.

#### `flat-map(stream, f)`

Map each value to a stream and flatten the results

For each value from the source, calls f to get a stream.
Yields all values from that stream before moving to the next source value.

#### `collect(stream)`

Collect all stream values into a vector

WARNING: This loads the entire stream into memory. Use with caution
on large or infinite streams. Consider take() to limit size.

#### `reduce(stream, init, f)`

Reduce stream to a single value using an accumulator function

Calls f(accumulator, value) for each stream value.
Returns the final accumulator value.

#### `fold(stream, init, f)`

Fold with early termination support

Calls f(accumulator, value) for each value. f should return:
- Result.Ok { value: new_accumulator } to continue
- Result.Err to stop and return the current accumulator

#### `for-each(stream, f)`

Execute a side effect for each stream value

The return values from f are discarded. If f throws, the error
propagates and the stream is closed.

#### `find(stream, pred)`

Find the first value matching a predicate

Returns the value if found, null if stream ends without match.

#### `any?(stream, pred)`

Check if any value satisfies a predicate

#### `all?(stream, pred)`

Check if all values satisfy a predicate

#### `count(stream)`

Count the number of values in the stream

#### `file-stream(path)`

Stream file as raw byte chunks

Opens the file on first pull, reads in fixed-size chunks.
Just emits raw bytes - the consumer decides how to interpret.
Last chunk may be smaller if EOF is reached.
Automatically closes the file when done or on error.

Typically used with decoders like lines() or by-size():
  file-stream("/var/log/app.log")
    |> stream/lines()           // Split on newlines
    |> stream/map(parse)
    |> stream/for-each(process)

#### `file-stream-sized(path, chunk_size)`

Stream file with custom chunk size

#### `file-stream-sync(path)`

Stream file using synchronous (blocking) I/O

Much faster than async/file-stream for local files since it avoids
thread pool and event loop overhead. Use for files already in OS cache.
Use async/file-stream for network-based file access or when you need
concurrent I/O with other operations.

Example:
  file-stream-sync("/var/log/app.log")
    |> stream/lines()
    |> stream/count()

#### `file-stream-sync-sized(path, chunk_size)`

Stream file with custom chunk size (synchronous)

#### `read-dir-stream(path)`

Stream directory entries (lazy iteration)

Loads the directory contents once on first pull, then iterates lazily.
Returns just the filenames, not full paths.

Example:
  stream/read-dir-stream("/data")
    |> stream/filter(fn(name) { core/ends-with?(name, ".txt") })
    |> stream/collect()

#### `split-on(stream, delimiter)`

Split incoming byte stream on a delimiter

Buffers incoming bytes and yields complete messages separated by delimiter.
Works with file-stream, socket-stream, or any byte stream.

Example:
  file-stream("/data.csv")
    |> stream/split-on(",")  // Split on comma
    |> stream/collect()

  socket-stream(conn)
    |> stream/split-on("\r\n\r\n")  // Split on double CRLF
    |> stream/map(parse-headers)

#### `lines(stream)`

Convenience: split on newlines

Equivalent to split-on(stream, "\n")

Example:
  file-stream("/var/log/app.log")
    |> stream/lines()
    |> stream/filter(has-error)
    |> stream/map(parse)

#### `by-size(stream, size)`

Batch stream into fixed-size chunks

Groups data into chunks of specified size.
Useful when you want fixed message sizes rather than delimiters.

Example:
  socket-stream(conn)
    |> stream/by-size(1024)  // 1KB chunks
    |> stream/map(parse-packet)

#### `from-generator(gen)`

Create a stream from a generator function

Generator is called repeatedly until it returns null or throws StopIteration.
Any other exception is propagated as a stream error.

#### `range(start, end)`

Create a stream from a range [start, end)

Yields integers starting from start, incrementing by 1, up to but not
including end.

Example:
  stream/range(0, 5) |> stream/collect()  ; [0, 1, 2, 3, 4]

#### `repeat(f)`

Create an infinite stream by repeatedly calling a function

WARNING: Infinite stream. Must use take() or find() to limit results.

#### `from-vector(vec)`

Create a stream from a vector

Yields each element of the vector in order.

#### `merge(stream1, stream2)`

Merge two streams (interleaved)

Pulls from both streams, alternating. If one stream ends, continues
with the other. If one errors, propagates the error.

#### `merge-pull(self, from-left)`

Pull one side of a merge, advancing to the other side on Done and propagating Error.

#### `zip(stream1, stream2)`

Zip two streams into [value1, value2] pairs

Yields pairs from both streams until one ends.
If one stream ends before the other, the zip ends too.

#### `buffered(stream, size)`

Buffer stream values to enable batching

Collects up to size values from the source before yielding them.
Useful for reducing context switches or enabling bulk processing.

Note: This buffers eagerly to reach the specified size, so it may
block longer than direct pulls from the source.

#### `distinct(stream)`

Yield only the first occurrence of each value, dropping later duplicates

Tracks previously-seen values in an atom-held set, so equality (==) decides
duplication. Lazy: pulls upstream until a not-yet-seen value appears.

#### `scan(stream, init, f)`

Yield each intermediate accumulator of a left fold, starting with init

Like reduce, but emits every step: first init, then f(acc, value) for each
upstream value. So scan(s, 0, +) over [1, 2, 3] yields 0, 1, 3, 6.
If f throws, the error becomes a StreamResult.Error and the stream ends.

#### `catch-default(stream, default_value)`

Catch stream errors and substitute a default value

If an error occurs, yields the default value and continues with Done.
Useful for graceful degradation in the face of I/O errors.

#### `retry(stream, max_retries)`

Retry on error up to n times before propagating

If a stream operation fails, automatically retries the same operation
up to max_retries times. After all retries are exhausted, the error
is propagated.

#### `socket-stream(sock)`

Stream TCP socket as raw byte chunks

Reads from the socket and yields chunks as they arrive.
Just emits raw bytes - the consumer decides how to interpret.
Last chunk may be smaller if EOF is reached.

Typically used with decoders like lines() or split-on():
  socket-stream(conn)
    |> stream/lines()              // Split on newlines
    |> stream/map(json-decode)
    |> stream/for-each(handle-msg)

#### `socket-stream-sized(sock, chunk_size)`

Stream socket with custom chunk size

<details>
<summary><strong>Undocumented functions (45)</strong></summary>

- `source-next(self)`
- `source-close(self)`
- `Stream_close(stream)`
- `MapSource_source-next(self)`
- `MapSource_source-close(self)`
- `FilterSource_source-next(self)`
- `FilterSource_source-close(self)`
- `TakeSource_source-next(self)`
- `TakeSource_source-close(self)`
- `TakeWhileSource_source-next(self)`
- `TakeWhileSource_source-close(self)`
- `SkipSource_source-next(self)`
- `SkipSource_source-close(self)`
- `DropWhileSource_source-next(self)`
- `DropWhileSource_source-close(self)`
- `FlatMapSource_source-next(self)`
- `FlatMapSource_source-close(self)`
- `FileChunkSource_source-next(self)`
- `FileChunkSource_source-close(self)`
- `FileChunkSourceSync_source-next(self)`
- `FileChunkSourceSync_source-close(self)`
- `DirSource_source-next(self)`
- `DirSource_source-close(self)`
- `SplitOnSource_source-next(self)`
- `SplitOnSource_source-close(self)`
- `BySizeSource_source-next(self)`
- `BySizeSource_source-close(self)`
- `GeneratorSource_source-next(self)`
- `GeneratorSource_source-close(self)`
- `MergeSource_source-next(self)`
- `MergeSource_source-close(self)`
- `ZipSource_source-next(self)`
- `ZipSource_source-close(self)`
- `BufferedSource_source-next(self)`
- `BufferedSource_source-close(self)`
- `DistinctSource_source-next(self)`
- `DistinctSource_source-close(self)`
- `ScanSource_source-next(self)`
- `ScanSource_source-close(self)`
- `CatchSource_source-next(self)`
- `CatchSource_source-close(self)`
- `RetrySource_source-next(self)`
- `RetrySource_source-close(self)`
- `SocketChunkSource_source-next(self)`
- `SocketChunkSource_source-close(self)`

</details>

---

## beagle.string-builder

> **Documentation coverage:** 20/20 functions (100%)

#### `new(capacity)`

Construct an empty builder. `capacity` is a hint — the buffer grows
automatically. Pass a large value if you know roughly how many bytes
you'll write to skip intermediate growth.

#### `append-byte!(sb, byte)`

Append one byte (an Int 0..255). Out-of-range values are masked to a u8.

#### `append-char!(sb, ch)`

Append a single-character string (or any string really; the name
just signals that this is a per-char hot path).

#### `append!(sb, s)`

Append a string. Works on flat strings, slices, and cons strings.

#### `append-range!(sb, s, start, end)`

Append bytes from s[start:end].

#### `append-range-filter-byte!(sb, s, start, end, skip-byte)`

Append bytes from s[start:end], skipping every occurrence of skip-byte.

#### `append-range-uppercase!(sb, s, start, end)`

Append bytes from s[start:end], converting ASCII letters to uppercase.

#### `append-builder-range!(dst, src, start, end)`

Append bytes from src[start:end] into dst.

#### `index-byte(s, byte, start)`

Return the byte index of byte in s at or after start, or -1.

#### `append-int!(sb, i)`

Append the decimal representation of an int.

#### `append-float!(sb, f)`

Append a float, formatted the same way `println` formats floats.

#### `length(sb)`

Number of bytes written so far.

#### `capacity(sb)`

Total backing-buffer capacity in bytes.

#### `clear!(sb)`

Reset length to 0. Keeps the backing buffer (no re-alloc).

#### `byte-at(sb, i)`

Read a byte by index (returns Int, or null if out of range).

#### `set-byte-at!(sb, i, byte)`

Overwrite the byte at index `i`. Returns sb. No-op if out of range.

#### `reverse!(sb)`

Reverse the bytes in place. ASCII-safe.

#### `to-string(sb)`

Materialize an immutable String containing the bytes appended so far.
One allocation, one memcpy. Repeated calls produce independent strings.

#### `append-line!(sb, s)`

Append the string `s` followed by a newline ("\n"). Returns the builder
so calls can be chained.

#### `is-empty?(sb)`

True when nothing has been appended yet (current byte length is 0).

---

## beagle.struct-pack

> **Documentation coverage:** 60/60 functions (100%)

#### `require-unsigned(value, hi, code)`

Return `value` unchanged if it fits in unsigned [0, hi]; otherwise throw a
range error naming the format `code`.

#### `require-signed(value, lo, hi, code)`

Return `value` unchanged if it fits in signed [lo, hi]; otherwise throw a
range error naming the format `code`.

#### `mask-bits(value, bits)`

Mask `value` to its low `bits` bits (unsigned), yielding the two's-complement
form of a negative value. Returns `value` unchanged when `bits` >= 62.

#### `pack-u8(value)`

Pack an unsigned 8-bit value into a single-byte Vec<Int>.

#### `pack-i8(value)`

Pack a signed 8-bit value (two's complement) into a single-byte Vec<Int>.

#### `unpack-u8(bytes, offset)`

Read an unsigned byte at `offset`.

#### `unpack-i8(bytes, offset)`

Read a signed byte (two's complement) at `offset`.

#### `pack-u16-be(value)`

Pack an unsigned 16-bit value as big-endian (high byte first).

#### `pack-u16-le(value)`

Pack an unsigned 16-bit value as little-endian (low byte first).

#### `pack-u16(value)`

Pack an unsigned 16-bit value (big-endian default).

#### `pack-i16-be(value)`

Pack a signed 16-bit value (two's complement) as big-endian.

#### `pack-i16-le(value)`

Pack a signed 16-bit value (two's complement) as little-endian.

#### `pack-i16(value)`

Pack a signed 16-bit value (big-endian default).

#### `unpack-u16-be(bytes, offset)`

Read an unsigned big-endian 16-bit value at `offset`.

#### `unpack-u16-le(bytes, offset)`

Read an unsigned little-endian 16-bit value at `offset`.

#### `unpack-u16(bytes, offset)`

Read an unsigned 16-bit value (big-endian default).

#### `sign16(v)`

Sign-extend an unsigned 16-bit value `v` into a signed Int.

#### `unpack-i16-be(bytes, offset)`

Read a signed big-endian 16-bit value at `offset`.

#### `unpack-i16-le(bytes, offset)`

Read a signed little-endian 16-bit value at `offset`.

#### `unpack-i16(bytes, offset)`

Read a signed 16-bit value (big-endian default).

#### `pack-u32-be(value)`

Pack an unsigned 32-bit value as big-endian (high byte first).

#### `pack-u32-le(value)`

Pack an unsigned 32-bit value as little-endian (low byte first).

#### `pack-u32(value)`

Pack an unsigned 32-bit value (big-endian default).

#### `pack-i32-be(value)`

Pack a signed 32-bit value (two's complement) as big-endian.

#### `pack-i32-le(value)`

Pack a signed 32-bit value (two's complement) as little-endian.

#### `pack-i32(value)`

Pack a signed 32-bit value (big-endian default).

#### `unpack-u32-be(bytes, offset)`

Read an unsigned big-endian 32-bit value at `offset`.

#### `unpack-u32-le(bytes, offset)`

Read an unsigned little-endian 32-bit value at `offset`.

#### `unpack-u32(bytes, offset)`

Read an unsigned 32-bit value (big-endian default).

#### `sign32(v)`

Sign-extend an unsigned 32-bit value `v` into a signed Int.

#### `unpack-i32-be(bytes, offset)`

Read a signed big-endian 32-bit value at `offset`.

#### `unpack-i32-le(bytes, offset)`

Read a signed little-endian 32-bit value at `offset`.

#### `unpack-i32(bytes, offset)`

Read a signed 32-bit value (big-endian default).

#### `require-unsigned-64(value, code)`

Return `value` unchanged if it is non-negative (it then necessarily fits in
unsigned 64 bits given Beagle's 62-bit ints); otherwise throw, naming `code`.

#### `bytes-u64-be(v)`

Return the 8 big-endian bytes (high byte first) of `v`'s low 64 bits as a
Vec<Int>; negative `v` yields its two's-complement bytes.

#### `bytes-u64-le(v)`

Return the 8 little-endian bytes (low byte first) of `v`'s low 64 bits as a
Vec<Int>; negative `v` yields its two's-complement bytes.

#### `pack-u64-be(value)`

Pack an unsigned 64-bit value as big-endian (high byte first). Input must be
non-negative (Beagle's 62-bit ints cannot express bit-63-set u64 values).

#### `pack-u64-le(value)`

Pack an unsigned 64-bit value as little-endian (low byte first). Input must
be non-negative.

#### `pack-u64(value)`

Pack an unsigned 64-bit value (big-endian default).

#### `pack-i64-be(value)`

Pack a signed 64-bit value (two's complement) as big-endian.

#### `pack-i64-le(value)`

Pack a signed 64-bit value (two's complement) as little-endian.

#### `pack-i64(value)`

Pack a signed 64-bit value (big-endian default).

#### `unpack-u64-be(bytes, offset)`

Read an unsigned big-endian 64-bit value at `offset`. Only round-trips
values within Beagle's 62-bit int range (larger u64s are unrepresentable).

#### `unpack-u64-le(bytes, offset)`

Read an unsigned little-endian 64-bit value at `offset`. Only round-trips
values within Beagle's 62-bit int range.

#### `unpack-u64(bytes, offset)`

Read an unsigned 64-bit value (big-endian default).

#### `unpack-i64-be(bytes, offset)`

Read a signed big-endian 64-bit value at `offset`. Sign-extends from the
most-significant byte, so negatives round-trip without forming the
unrepresentable unsigned 64-bit value.

#### `unpack-i64-le(bytes, offset)`

Read a signed little-endian 64-bit value at `offset`. Sign-extends from the
most-significant (last) byte so negatives round-trip correctly.

#### `unpack-i64(bytes, offset)`

Read a signed 64-bit value (big-endian default).

#### `append-all(dst, src)`

Append every element of `src` onto `dst`, returning the resulting Vec<Int>.

#### `order-is-le(c)`

True when order char `c` selects little-endian (`<` or `=`); false otherwise.

#### `code-width(c)`

Return the byte width of field code `c` (1 for b/B/x, 2 for h/H, 4 for i/I,
8 for q/Q); throws on an unknown code.

#### `pack-one(c, value, le)`

Pack a single `value` for field code `c`, little-endian when `le` is true,
returning its Vec<Int> bytes; throws on an unknown code.

#### `unpack-one(c, bytes, offset, le)`

Read a single value for field code `c` from `bytes` at `offset`,
little-endian when `le` is true; throws on an unknown code.

#### `is-order-char(c)`

True when `c` is an endianness/order marker (`>`, `<`, `!`, or `=`).

#### `is-skip-char(c)`

True when `c` is whitespace (space, tab, newline, or carriage return) to be
ignored within a format string.

#### `format-body-start(format)`

Return the index where field codes begin in `format`: 1 when a leading order
char is present, otherwise 0.

#### `format-is-le(format)`

True when `format`'s optional leading order char selects little-endian;
defaults to false (big-endian) when no order char is present.

#### `pack(format, values)`

Pack a vector of `values` according to `format`, returning a Vec<Int> of
bytes. Each value-bearing field code consumes one value in order; the pad
code `x` writes a zero byte and consumes no value.

  pack(">HHB", [0x0102, 0x0304, 0x05]) == [1, 2, 3, 4, 5]

#### `unpack(format, bytes)`

Unpack `bytes` according to `format`, returning a Vec<Int> of values in
field order. The pad code `x` skips a byte and produces no value.

  unpack(">HHB", [1,2,3,4,5]) == [258, 772, 5]

#### `calcsize(format)`

Compute the total byte size of a `format` string (like Python's calcsize).

---

## beagle.template

> **Documentation coverage:** 25/25 functions (100%)

#### `ws-code?(c)`

True if char-code `c` is an ASCII whitespace character (space, tab,
newline, CR, form-feed, or vertical-tab).

#### `strip(s)`

Trim ASCII whitespace from both ends of string `s` and return the
substring. Self-contained (does not rely on core/trim).

#### `matches-at?(s, i, pat)`

True if the literal `pat` occurs in `s` starting exactly at index `i`.

#### `find-from(s, pat, start)`

Return the first index >= `start` where `pat` occurs in `s`, or -1 if not
found.

#### `tokenize(src)`

Scan `src` left-to-right into a vector of `Token`s, splitting on the
`{{{ }}}`, `{{ }}`, and `{% %}` delimiters (triple-brace is tested first).
Throws a clear error on an unterminated block.

#### `flush-literal(tokens, src, start, end)`

Append a "literal" `Token` for `src[start:end]` to `tokens` if the range is
non-empty; otherwise return `tokens` unchanged.

#### `parse-path(expr)`

Split a dotted path string like "user.name" into a vector of keyword
segments `[:user :name]`. Throws on an empty or malformed path
(e.g. "a..b" or a leading/trailing dot).

#### `parse-expr(text, default-escape)`

Parse the inside of an interpolation into a `VarNode`. An optional " | raw"
filter forces `escape` false; `default-escape` sets the base (true for
`{{ }}`, false for `{{{ }}}`). Throws on any unknown filter.

#### `tag-keyword(tag)`

Return the first whitespace-delimited word of a tag body, e.g.
"if x" -> "if".

#### `tag-rest(tag)`

Return everything after the first word of a tag body, stripped, e.g.
"for x in items" -> "x in items".

#### `parse-body(tokens, start, stops)`

Recursive-descent parse of `tokens` from index `start`, building a node
vector until it hits a tag whose keyword is in `stops` (which it does NOT
consume). Returns `[nodes, next-index]`.

#### `is-stop?(kw, stops)`

True if tag keyword `kw` is one of the terminating keywords in the `stops`
vector.

#### `parse-if(tokens, start)`

Parse an `{% if %}` block beginning at `tokens[start]`, consuming an
optional `{% else %}` and the closing `{% endif %}`. Returns
`[IfNode, next-index]`. Throws if the condition is empty or the block is
unterminated.

#### `parse-for(tokens, start)`

Parse a `{% for VAR in ITEMS %}` block beginning at `tokens[start]`,
consuming the closing `{% endfor %}`. Returns `[ForNode, next-index]`.
Throws on malformed syntax or a missing `{% endfor %}`.

#### `parse(tokens)`

Parse the full token vector into a top-level node vector. Throws if a stray
`endif`/`else`/`endfor` tag is found with no open block.

#### `lookup(ctx, path)`

Resolve a keyword `path` against context map `ctx` via chained `get`,
trying the keyword key first then falling back to its string form. Returns
null if any segment is missing.

#### `truthy?(v)`

Jinja-style truthiness for `{% if %}`: null, false, the empty string, the
empty vector, and numeric zero (Int or Float) are falsy; everything else is
truthy.

#### `html-escape(s)`

Return `s` with the five significant HTML characters (& < > " ') replaced
by their entities, processed character by character.

#### `render-value(v)`

Render a value to its string form for output: null becomes "" (so missing
vars vanish), strings pass through, and everything else goes through
`to-string`.

#### `render-body(builder, nodes, ctx)`

Render each node in `nodes` into the string-`builder` against context
`ctx`. Returns null.

#### `render-node(builder, node, ctx)`

Render a single AST node (TextNode/VarNode/IfNode/ForNode) into `builder`
against context `ctx`. Throws on an unknown node type.

#### `render-for(builder, node, ctx)`

Render a `ForNode` by iterating its vector and rendering the body once per
element with the loop var bound in a child context. A null collection
renders nothing; a non-vector value throws a clear error.

#### `compile(template)`

Tokenize and parse `template` ONCE into a reusable `CompiledTemplate`.
Render it against many contexts with `render-compiled` to avoid re-parsing
the same string repeatedly. Throws on malformed template syntax (the same
errors `render` raises).

#### `render-compiled(compiled, ctx)`

Render an already-`compile`d template against context map `ctx`. Equivalent
to `render` but skips the tokenize+parse step, so it is cheap to call
repeatedly with the same compiled template. Throws if `compiled` is not a
`CompiledTemplate`.

#### `render(template, ctx)`

Render a template string against a context map.

Context keys are keywords; template variable paths are dotted strings
resolved as nested keyword lookups. Interpolated values are HTML-escaped
unless the `| raw` filter or triple-brace `{{{ }}}` is used.

Equivalent to `render-compiled(compile(template), ctx)`; use `compile` +
`render-compiled` directly when rendering one template many times.

Examples:
  render("Hello {{ name }}!", {:name "World"})       // => "Hello World!"
  render("{{ user.name }}", {:user {:name "Ann"}})   // => "Ann"

---

## beagle.test

> **Documentation coverage:** 11/11 functions (100%)

#### `new-results()`

Create a fresh, empty results accumulator (an atom).

#### `results-snapshot(acc)`

Snapshot the current results map out of the accumulator atom.

#### `record-pass(acc)`

Increment the `:passed` count in the accumulator atom `acc`. Returns null.

#### `record-fail(acc, failure)`

Increment the `:failed` count and append `failure` (a map describing what
went wrong) to the `:failures` list in accumulator atom `acc`. Returns null.

#### `assert-eq(acc, actual, expected, name)`

Assert that `actual` equals `expected` (structural equality).
Records a pass or a failure into `acc`. Returns true on pass.

#### `assert-true(acc, cond, name)`

Assert that `cond` is truthy (not null, not false).
Records a pass or a failure into `acc`. Returns true on pass.

#### `assert-throws(acc, thunk, name)`

Assert that calling `thunk` (a zero-arg function) throws.
Passes if the thunk raises; fails if it returns normally. Returns true on pass.

#### `print-failures(failures, idx, len)`

Recursively print each failure in `failures` from index `idx` up to `len`,
one `FAIL [name] (kind): message` line per failure. Returns null.

#### `print-summary(results)`

Print a human-readable summary of a results map.

#### `run-tests-loop(test-fns, idx, len, acc)`

Recursively invoke each test function in `test-fns` from index `idx` up to
`len`, passing accumulator `acc`. A test that throws is caught and recorded
as a "test-crash" failure rather than aborting the run. Returns null.

#### `run-tests(test-fns)`

Run a vector of one-arg test functions `fn(acc) { ... }`. Each test function
calls the assert helpers (assert-eq / assert-true / assert-throws) with the
accumulator it is given. A test function that itself throws is caught and
recorded as a failure (so one broken test never aborts the run).

Returns {:passed n :failed n :failures [..]} and prints a summary.

---

## beagle.test-async

> **Documentation coverage:** 12/20 functions (60%)

#### `default-watchdog-ms()`

Default watchdog budget in milliseconds (30000) used by every assertion; tunable per call.

#### `deadline-after(watchdog_ms)`

Convert a millisecond watchdog budget into an absolute monotonic-nanosecond deadline.

#### `past-deadline?(deadline)`

True once the monotonic clock has passed the given nanosecond deadline.

#### `spin-burn(n)`

Burn n iterations of pure CPU (no allocation, no GC) to back off between empty polls; returns null.

#### `receive-within(ch, watchdog_ms, what)`

Busy-spin receive one value from ch, returning it as soon as present; throws (hard fail) naming `what` if the watchdog fires first.

#### `receive-within-spin(ch, deadline, watchdog_ms, what, backoff)`

Recursive spin helper for receive-within: polls try-receive with growing backoff until a value arrives, else throws once past `deadline`.

#### `count-equal(coll, x)`

Count elements of coll structurally equal to x.

#### `multiset-eq?(a, b)`

True iff a and b are equal as multisets (same elements with the same multiplicities, order irrelevant).

#### `assert-set-eq!(actual, expected)`

Assert actual and expected are equal as multisets (order-insensitive); throws on mismatch, returns null.

#### `collect-n(ch, n, watchdog_ms)`

Receive exactly n values from ch (each watchdog-bounded), returning them in arrival order.

#### `assert-channel-drained!(ch)`

Assert ch holds no unreceived values (catches over-production); throws if any remain. Call only after a completion barrier proves all producers are done.

#### `await-condition-spin!(pred, deadline, watchdog_ms, backoff)`

Recursive spin helper for await-condition!: polls pred with growing backoff until truthy, else throws once past `deadline`.

<details>
<summary><strong>Undocumented functions (8)</strong></summary>

- `assert-receives!$2(ch, expected)`
- `assert-receives!$3(ch, expected, watchdog_ms)`
- `assert-receives-all!$3(ch, n, expected-set)`
- `assert-receives-all!$4(ch, n, expected-set, watchdog_ms)`
- `await-condition!$1(pred)`
- `await-condition!$2(pred, watchdog_ms)`
- `assert-eventually!$1(pred)`
- `assert-eventually!$2(pred, watchdog_ms)`

</details>

---

## beagle.text

> **Documentation coverage:** 30/36 functions (83%)

#### `code-at(s, i)`

Return the char-code (codepoint) of the character at index i in s.
Thin wrapper over `char-code(s[i])`.

#### `ws-code?(c)`

True if code point c is ASCII whitespace (space, tab, newline,
carriage-return, form-feed, or vertical-tab).

#### `digit-code?(c)`

True if code point c is an ASCII decimal digit ('0'..'9').

#### `repeat-string(s, n)`

Repeat a string n times, concatenated together.

Returns "" when n <= 0.

Examples:
  repeat-string("ab", 3)  // => "ababab"
  repeat-string("x", 0)   // => ""

#### `clip-fill(pad, n)`

Build a fill string of exactly `n` characters by repeating `pad` and
clipping the result to length n. Helper for pad-left/pad-right.

#### `chars(s)`

Return the characters of `s` as a vector of single-character strings.

Examples:
  chars("abc")  // => ["a", "b", "c"]

#### `lines(s)`

Split `s` into a vector of lines on "\n" (a trailing newline yields a final
empty string). Carriage returns are left intact.

Examples:
  lines("a\nb\nc")  // => ["a", "b", "c"]

#### `capitalize(s)`

Uppercase the first character, lowercase the rest.

Examples:
  capitalize("hELLO")  // => "Hello"
  capitalize("")       // => ""

#### `title-case(s)`

Capitalize the first letter of each whitespace-separated word.

Collapses runs of whitespace to a single space in the output.

Examples:
  title-case("the quick brown fox")  // => "The Quick Brown Fox"
  title-case("  hELLO   wORLD ")      // => "Hello World"

#### `char-set-contains?(chars, ch)`

True if the single-character string `ch` appears in `chars` (treated as a
set of characters). Thin wrapper over `index-of`.

#### `trim-chars(s, chars)`

Trim every leading and trailing character that appears in `chars`.

`chars` is a string treated as a set of characters to strip. Unlike
core's trim (whitespace only), this strips an arbitrary set.

Examples:
  trim-chars("xxhelloxx", "x")     // => "hello"
  trim-chars("[[data]]", "[]")     // => "data"
  trim-chars("--a-b--", "-")       // => "a-b"

#### `trim-chars-left(s, chars)`

Trim only leading characters in `chars`.

#### `trim-chars-right(s, chars)`

Trim only trailing characters in `chars`.

#### `replace-all(s, old, new)`

Replace every non-overlapping occurrence of `old` with `new`.

Matches core's `replace` semantics (replaces ALL occurrences). Provided
here under the conventional name. Throws on an empty `old` (which would
otherwise loop forever).

Examples:
  replace-all("a.b.c", ".", "/")   // => "a/b/c"
  replace-all("aaa", "a", "bb")    // => "bbbbbb"

#### `replace-first(s, old, new)`

Replace only the FIRST occurrence of `old` with `new`.

If `old` does not occur in `s`, returns `s` unchanged. Companion to
`replace-all`. Throws on an empty `old` (an empty needle has no
well-defined single occurrence).

Examples:
  replace-first("a.b.c", ".", "/")   // => "a/b.c"
  replace-first("hello", "l", "L")   // => "heLlo"
  replace-first("abc", "x", "y")     // => "abc"

#### `split-with-limit(s, delim, limit)`

Split s on `delim`, producing at most `limit` pieces.

The final piece holds the unsplit remainder (delimiters and all). A
`limit` <= 0 means "no limit" (split fully). Returns a vector of strings.

Examples:
  split-with-limit("a,b,c,d", ",", 2)  // => ["a", "b,c,d"]
  split-with-limit("a,b,c,d", ",", 3)  // => ["a", "b", "c,d"]
  split-with-limit("a,b,c", ",", 0)    // => ["a", "b", "c"]
  split-with-limit("abc", ",", 2)      // => ["abc"]

#### `starts-with-any?(s, prefixes)`

True if s starts with ANY of the given prefix strings.

`prefixes` is any seqable collection of strings.

Examples:
  starts-with-any?("http://x", ["http://", "https://"])  // => true
  starts-with-any?("ftp://x",  ["http://", "https://"])  // => false

#### `ends-with-any?(s, suffixes)`

True if s ends with ANY of the given suffix strings.

Examples:
  ends-with-any?("a.png", [".png", ".jpg"])  // => true

#### `contains-any?(s, subs)`

True if s contains ANY of the given substrings.

Examples:
  contains-any?("hello world", ["xyz", "wor"])  // => true
  contains-any?("hello", ["a", "b"])            // => false

#### `parse-float(s)`

Parse a string into a Float. Returns null on any malformed input.

Accepts an optional leading sign, an integer part, an optional fractional
part, and an optional exponent (e/E with optional sign). At least one
digit must be present overall.

Examples:
  parse-float("3.14")    // => 3.14
  parse-float("-0.5")    // => -0.5
  parse-float("10")      // => 10.0
  parse-float("1e3")     // => 1000.0
  parse-float("2.5e-2")  // => 0.025
  parse-float("abc")     // => null
  parse-float("")        // => null

#### `hex-digit(d, upper)`

Return the hex digit character for value d (0..15) as a single-char string.
When `upper` is true, letters are uppercase ('A'..'F'), otherwise lowercase.

#### `int->hex(n, upper)`

Render a non-negative int as a hexadecimal string.

Negative inputs are rendered with a leading '-'.

Examples:
  int->hex(255, false)  // => "ff"
  int->hex(255, true)   // => "FF"
  int->hex(0, false)    // => "0"

#### `fixed(x, decimals)`

Render a number with EXACTLY `decimals` fractional digits, rounded HALF-UP.

Sign is split off and reapplied; the fraction is zero-padded on the left to
exactly `decimals` digits; rounding that reaches a whole unit carries into
the integer part (e.g. 9.999 -> "10.00"). Works on ints and floats.

Examples:
  fixed(3.14159, 2)        // => "3.14"
  fixed(3.14159, 6)        // => "3.141590"
  fixed(9.999, 2)          // => "10.00"
  fixed(0.0 - 9.999, 2)  // => "-10.00"
  fixed(3.0, 0)            // => "3"

#### `round-to(x, n)`

Round a number to `n` decimal places, returning a NUMBER (not a string).

Use this when you need the rounded value for further math; use `fixed` when
you want a string with a guaranteed number of decimals.

Examples:
  round-to(3.14159, 2)  // => 3.14
  round-to(2.5, 0)      // => 3.0

#### `commas(n)`

Insert thousands separators into an integer, returning a string.

Examples:
  commas(1234567)  // => "1,234,567"
  commas(-1000)    // => "-1,000"
  commas(42)       // => "42"

#### `count-occurrences(s, sub)`

Count non-overlapping occurrences of `sub` in s.

Examples:
  count-occurrences("banana", "a")   // => 3
  count-occurrences("aaaa", "aa")    // => 2

#### `reverse-string(s)`

Reverse the characters of a string.

Examples:
  reverse-string("abc")  // => "cba"

#### `string->bytes(s)`

Return the UTF-8 byte values (each 0..255) of a string as a Vec<Int>.

Multi-byte codepoints expand to their full UTF-8 encoding — e.g. "é" is
[195, 169], not [233].

Examples:
  string->bytes("A")   // => [65]
  string->bytes("é")   // => [195, 169]
  string->bytes("")    // => []

#### `bytes->string(vec)`

Build a String from a Vec of UTF-8 byte values (each an Int 0..255).

Inverse of `string->bytes`. The bytes must form valid UTF-8.

Examples:
  bytes->string([65])         // => "A"
  bytes->string([195, 169])   // => "é"
  bytes->string([])           // => ""

#### `ellipsize(s, n, ellipsis)`

Truncate s to at most `n` characters, appending `ellipsis` if truncated.

The result (including the ellipsis) is at most `n` characters. If `n` is
smaller than the ellipsis, the ellipsis itself is truncated.

Examples:
  ellipsize("hello world", 8, "...")  // => "hello..."
  ellipsize("hi", 8, "...")            // => "hi"

<details>
<summary><strong>Undocumented functions (6)</strong></summary>

- `pad-left$2(s, width)`
- `pad-left$3(s, width, pad)`
- `pad-right$2(s, width)`
- `pad-right$3(s, width, pad)`
- `center$2(s, width)`
- `center$3(s, width, pad)`

</details>

---

## beagle.textwrap

> **Documentation coverage:** 13/15 functions (86%)

#### `code-at(s, i)`

Return the char-code (code point) of the character at index `i` in `s`.

#### `ws-code?(c)`

True if code point `c` is ASCII whitespace (space, tab, LF, CR, FF, or VT).

#### `ws-char?(ch)`

True if the single-character string `ch` is ASCII whitespace.

#### `split-words(text)`

Split `text` into whitespace-delimited words. Whitespace runs are
collapsed and leading/trailing whitespace is dropped. Returns a vector of
non-empty word strings (empty vector for empty/whitespace-only input).

#### `wrap(text, width)`

Greedy word-wrap `text` to lines of at most `width` characters.

Words are kept whole and never split. A line packs as many words as fit
(joined with single spaces). A single word longer than `width` gets its
own line and is allowed to overflow (it is NOT broken). Leading/trailing
whitespace is stripped and internal whitespace runs are collapsed.

Returns a vector of line strings (empty vector for empty/whitespace-only
input). Throws on width <= 0.

Examples:
  wrap("the quick brown fox", 9)  // => ["the quick", "brown fox"]
  wrap("supercalifragilistic", 9) // => ["supercalifragilistic"]
  wrap("", 9)                      // => []

#### `fill(text, width)`

Wrap `text` to `width` and join the resulting lines with newlines.

Equivalent to `join(wrap(text, width), "\n")`. Returns "" for empty input.

Examples:
  fill("the quick brown fox", 9)  // => "the quick\nbrown fox"

#### `split-lines-keep(text)`

Split `text` into lines on "\n", preserving empty lines and any trailing
empty segment (so "a\n" => ["a", ""]). Returns a vector of line strings.

#### `leading-ws-len(line)`

Return the number of leading whitespace characters at the start of `line`.

#### `blank-line?(line)`

True if `line` is empty or consists entirely of whitespace.

#### `common-prefix-len(a, b)`

Return the length of the longest common prefix of strings `a` and `b`,
compared character by character.

#### `dedent(text)`

Remove the longest common leading-whitespace prefix from every line.

Lines that consist solely of whitespace are ignored when computing the
common prefix (matching Python's textwrap.dedent) and are normalized to
empty lines in the output. Mixed tabs/spaces are compared literally.

Examples:
  dedent("    a\n    b")    // => "a\nb"
  dedent("  a\n   b\n  c")  // => "a\n b\nc"
  dedent("")                // => ""

#### `indent(text, prefix)`

Add `prefix` to the start of every non-empty line in `text`.

Lines that are empty or whitespace-only are left untouched (matching
Python's textwrap.indent default predicate). A trailing newline in the
input is preserved.

Examples:
  indent("a\nb", "> ")     // => "> a\n> b"
  indent("a\n\nb", "> ")   // => "> a\n\n> b"
  indent("a\nb\n", "> ")   // => "> a\n> b\n"

#### `trim-left-ws(s)`

Return `s` with any leading whitespace removed (trailing whitespace kept).

<details>
<summary><strong>Undocumented functions (2)</strong></summary>

- `shorten$2(text, width)`
- `shorten$3(text, width, placeholder)`

</details>

---

## beagle.time

> **Documentation coverage:** 17/17 functions (100%)

#### `libc-path()`

Returns the platform-specific libc shared-library to load via FFI:
libSystem on macOS, the glibc .so on Linux, falling back to "libc.so.6".

#### `epoch-seconds()`

Returns the current Unix time in whole seconds (UTC) via libc `time(NULL)`.

#### `epoch-millis()`

Returns the current Unix time in milliseconds (UTC), reading a struct
timeval via libc `gettimeofday` and combining tv_sec*1000 + tv_usec/1000.

#### `floor-div(a, b)`

Euclidean/floored integer division (rounds toward negative infinity),
unlike Beagle's `/` which truncates toward zero. Needed so the calendar
math stays correct for negative (pre-1970) inputs.

#### `floor-mod(a, b)`

Floored modulo paired with `floor-div`: the result has the same sign as
the divisor `b`. Thin wrapper computing `a - floor-div(a, b) * b`.

#### `days-from-civil(y, m, d)`

Returns the number of days from 1970-01-01 to the civil date `y`-`m`-`d`
(month 1..12, day 1..31), negative for pre-epoch dates. Uses Hinnant's
proleptic-Gregorian algorithm.

#### `civil-from-days(z0)`

Inverse of `days-from-civil`: converts a day count since 1970-01-01 (`z0`,
may be negative) into a `[year, month, day]` civil date.

#### `days-since-epoch(epoch)`

Returns the floored count of whole days since 1970-01-01 for an
epoch-seconds value `epoch`.

#### `weekday(epoch)`

Returns the weekday for an epoch-seconds value `epoch`, where 0 = Sunday
.. 6 = Saturday.

#### `civil-from-epoch(epoch)`

Decomposes an epoch-seconds value `epoch` into the UTC tuple
`[year, month, day, hour, minute, second, weekday]`.

#### `append-2!(b, n)`

Appends `n` to string-builder `b` as a zero-padded two-digit decimal.

#### `append-4!(b, n)`

Appends `n` to string-builder `b` as a zero-padded four-digit decimal
(assumes `n` >= 0).

#### `http-date(epoch)`

Formats an epoch-seconds value `epoch` as an RFC 1123 / HTTP Date string
in GMT, e.g. "Sun, 06 Nov 1994 08:49:37 GMT".

#### `ms-of-seconds(s)`

Converts a whole-second count `s` into milliseconds (multiplies by 1000).
Use this to make the unit explicit when building a millisecond duration.

#### `ms-of-minutes(m)`

Converts a whole-minute count `m` into milliseconds (multiplies by 60000).
Use this to make the unit explicit when building a millisecond duration.

#### `ms-of-hours(h)`

Converts a whole-hour count `h` into milliseconds (multiplies by 3600000).
Use this to make the unit explicit when building a millisecond duration.

#### `format-duration(ms)`

Formats a non-negative millisecond count `ms` as a compact human-readable
duration string. The output concatenates the non-zero hour/minute/second/
millisecond components with single-letter units (`h`, `m`, `s`, `ms`),
omitting any component that is zero. Examples: `0` -> "0ms",
`500` -> "500ms", `1500` -> "1s500ms", `3661000` -> "1h1m1s". A `ms` of 0
yields the literal "0ms". Throws on a negative input (durations are
non-negative; use a sign elsewhere if you need a delta).

---

## beagle.tls

> **Documentation coverage:** 6/6 functions (100%)

#### `connect(host, port)`

Open a TLS connection to `host`:`port`, returning an opaque connection
handle. Throws if the TCP connect, TLS handshake, or certificate
verification fails.

Examples:
  let c = connect("example.com", 443)

#### `write(conn, data)`

Write `data` (a string, byte-faithful) over the TLS connection. Returns the
number of bytes written. Throws on a write error.

#### `read(conn, n)`

Read up to `n` bytes of plaintext from the TLS connection, returning them as
a byte-faithful string. Returns "" at end-of-stream.

#### `read-all(conn)`

Read everything until end-of-stream, returning the full body as a string.
(Suitable for HTTP responses sent with `Connection: close`.)

#### `read-all-into(conn, builder)`

Internal: append successive reads into `builder` until end-of-stream.

#### `close(conn)`

Close the TLS connection (sends a close_notify and releases the handle).

---

## beagle.url

> **Documentation coverage:** 20/20 functions (100%)

#### `hex-digit(n)`

Map a nibble value 0..15 to its uppercase hex character ("0".."9", "A".."F").
Throws if `n` is outside 0..15 so codec bugs surface immediately.

#### `hex-value(code)`

Convert a hex-digit char code into its value 0..15 (upper or lower case),
or -1 if `code` is not a hex digit.

#### `unreserved-byte?(b)`

True if byte `b` is in the RFC 3986 unreserved set (A-Z a-z 0-9 - _ . ~),
i.e. a byte that need not be percent-encoded.

#### `encode-char-into(result, ch)`

Percent-encode a single character `ch` into the `result` byte-builder:
unreserved bytes are copied verbatim, all others emitted as %XX (uppercase
hex) over the character's UTF-8 bytes. Returns null.

#### `encode-component(s)`

Percent-encode string `s` as a URI component: every byte outside the
unreserved set A-Za-z0-9-_.~ becomes %XX (uppercase hex). Returns a new string.

#### `decode-component(s)`

Decode a percent-encoded component string: %XX -> byte, '+' kept literal,
malformed '%' left verbatim. For form decoding (where '+' is a space) use
decode-form/parse-query instead. Returns a new string.

#### `decode-form(s)`

Decode `s` with application/x-www-form-urlencoded semantics: '+' becomes a
space and %XX becomes its byte. Returns a new string.

#### `decode-into(s, plus-is-space)`

Shared decode driver for decode-component/decode-form. When `plus-is-space`
is true, '+' decodes to a space; otherwise it stays literal. Decoded bytes
are reassembled as UTF-8. Returns a new string.

#### `append-char-bytes(result, ch)`

Append the raw UTF-8 bytes of a single character `ch` onto the `result`
byte-builder. Returns null.

#### `parse-query(s)`

Parse an x-www-form-urlencoded query string into a map of string keys to
string values (both form-decoded). A leading '?' is stripped, empty pairs
from "&&" are skipped, a key without '=' maps to "", and on duplicate keys
the last value wins.

#### `parse-pair-into(m, pair)`

Form-decode one "key=value" pair (or bare "key", which yields "") and assoc
it into map `m`, returning the updated map.

#### `index-of-char(s, code)`

Return the index of the first character in `s` whose char-code equals `code`,
or -1 if none is found.

#### `build-query(m)`

Build an x-www-form-urlencoded query string from map `m`. Keys and values are
coerced to strings and percent-encoded with component rules (space -> %20),
joined by '&' in the map's key-iteration order. Round-trips through parse-query.

#### `coerce-string(x)`

Coerce a query key/value to a string: keywords become their name, existing
strings pass through, everything else goes through to-string.

#### `parse-query-multi(s)`

Parse an x-www-form-urlencoded query string into a map of string keys to
vectors of string values, preserving duplicate keys in encounter order
(the complement of parse-query, which is last-wins). A leading '?' is
stripped, empty pairs from "&&" are skipped, and a bare key (no '=') yields
a single empty-string value. "a=1&a=2" => {"a" ["1", "2"]}.

#### `parse-pair-multi-into(m, pair)`

Form-decode one "key=value" pair (or bare "key", which yields "") and append
its value onto the existing vector for that key in map `m`, creating a new
one-element vector for a first-seen key. Returns the updated map.

#### `last-index-of-char(s, code)`

Return the index of the last character in `s` whose char-code equals `code`,
or -1 if none is found.

#### `all-digits?(s)`

True when `s` is non-empty and every character is an ASCII digit 0-9.

#### `parse-url(s)`

Parse an absolute URL "scheme://host[:port][/path][?query][#fragment]" into
a Url struct. Throws if the "://" separator or the host is missing. port is
null unless a trailing all-digit port is present; query/fragment are the raw
strings (null when their delimiter is absent); path is "" when none is given.

#### `format-url(u)`

Reconstruct a URL string from a Url struct (the inverse of parse-url). Emits
":port" only when port is non-null, "?query" only when query is non-null,
and "#fragment" only when fragment is non-null; an empty-string query or
fragment still re-emits its leading '?' or '#'.

---

## beagle.ws

> **Documentation coverage:** 35/35 functions (100%)

#### `opcode-continuation()`

RFC 6455 opcode value for a continuation frame (0x0).

#### `opcode-text()`

RFC 6455 opcode value for a text frame (0x1).

#### `opcode-binary()`

RFC 6455 opcode value for a binary frame (0x2).

#### `opcode-close()`

RFC 6455 opcode value for a close control frame (0x8).

#### `opcode-ping()`

RFC 6455 opcode value for a ping control frame (0x9).

#### `opcode-pong()`

RFC 6455 opcode value for a pong control frame (0xA).

#### `to-buf(s)`

Wrap a byte-faithful string in a string-builder so its raw bytes can be
read by index regardless of UTF-8 validity.

#### `length-or-zero(s)`

Return length(s), or 0 if length() throws; used only as a builder size hint
for possibly-invalid-UTF-8 wire strings.

#### `buf-byte(buf, i)`

Read byte i (0..255) from a builder produced by to-buf.

#### `hex-to-bytes(hex)`

Convert a lowercase hex string into a Vec<Int> of byte values, one byte per
pair of hex digits.

#### `hex-digit-value(code)`

Map a single hex-digit char-code (0-9, a-f, A-F) to its 0..15 value; throws
on any non-hex code.

#### `accept-key(client-key)`

Compute the Sec-WebSocket-Accept value (base64(sha1(key ++ GUID))) for a
client's Sec-WebSocket-Key.

#### `is-websocket-upgrade?(request)`

True if the http/Request is a valid RFC 6455 upgrade: Upgrade: websocket,
a Connection header containing "upgrade", and a Sec-WebSocket-Key present.

#### `handshake-response(request)`

Build the 101 Switching Protocols http/Response that completes the
handshake; throws if the request has no Sec-WebSocket-Key header.

#### `parse-frame(bytes)`

Parse one complete WebSocket frame from a byte-faithful string starting at
index 0, handling 126/127 extended lengths and XOR-unmasking; returns a Frame.

#### `encode-frame(opcode, payload)`

Encode an unmasked, FIN=1 server->client frame for the given opcode and
payload String; returns a byte-faithful String suitable for socket/write.

#### `text-frame(payload)`

Encode payload as a text frame (thin encode-frame wrapper).

#### `binary-frame(payload)`

Encode payload as a binary frame (thin encode-frame wrapper).

#### `ping-frame(payload)`

Encode payload as a ping control frame (thin encode-frame wrapper).

#### `pong-frame(payload)`

Encode payload as a pong control frame (thin encode-frame wrapper).

#### `close-frame()`

Encode an empty (codeless, clean) close frame.

#### `text?(frame)`

True if the parsed frame is a text frame.

#### `binary?(frame)`

True if the parsed frame is a binary frame.

#### `ping?(frame)`

True if the parsed frame is a ping frame.

#### `pong?(frame)`

True if the parsed frame is a pong frame.

#### `close?(frame)`

True if the parsed frame is a close frame.

#### `continuation?(frame)`

True if the parsed frame is a continuation frame (opcode 0x0).

#### `frag-step(cur-opcode, cur-payload, frame)`

Pure fragmentation step: fold one data/continuation frame into the current
(opcode, payload) state, emitting a completed {:opcode, :payload} on a fin frame.

#### `reassemble(frames)`

Reassemble a list of data/continuation frames into complete messages
({:opcode, :payload}); single (fin=true) frames pass through as one message each.

#### `reassemble-helper(frames, i, cur-opcode, cur-payload, out)`

Internal fold for reassemble.

#### `read-exact-buf(conn, n)`

Read up to n bytes from conn into a byte-faithful string-builder, looping
over short socket reads and stopping at EOF.

#### `read-frame(conn)`

Read and parse one complete WebSocket frame off the socket; returns a Frame,
or null at EOF (connection closed before a full header arrived).

#### `do-handshake(conn)`

Perform the HTTP upgrade handshake on conn: on a valid WebSocket upgrade,
write the 101 response and return true; otherwise write a 400 and return false.

#### `serve-ws(host, port, on-message)`

Cooperative single-thread WebSocket server: accept connections one at a time,
upgrade each, and run its frame loop via on-message. GC-safe; blocks forever.

#### `handle-ws-connection(conn, on-message)`

Run one upgraded connection's frame loop (text/binary -> on-message reply,
ping -> pong, close -> close), isolated in a try so one bad client can't kill
the accept loop; closes the connection on exit.

---

## global

> **Documentation coverage:** 0/1 functions (0%)

<details>
<summary><strong>Undocumented functions (1)</strong></summary>

- `trampoline()`

</details>

---

## Types

These are the built-in struct types available in Beagle.

#### beagle.core/Function

A first-class function object.

**Fields:**

- `pointer`
- `name`
- `arity`

#### beagle.core/NEVER

Internal sentinel struct (workaround for id 0 activating the fast-path cache).

#### beagle.core/Struct

Runtime descriptor for a struct type: its name and numeric id.

**Fields:**

- `name`
- `id`

#### beagle.core/Protocol

Runtime descriptor for a protocol, identified by name.

**Fields:**

- `name`

#### beagle.core/Thread

Holds a thread's closure and acts as a GC root while the thread runs.

**Fields:**

- `closure`

#### beagle.core/Atom

Thread-safe mutable reference cell holding a single value.

**Fields:**

- `value`

#### beagle.core/ArraySeq

Seq view over a raw Array, tracking the current index.

**Fields:**

- `arr`
- `index`

#### beagle.core/PersistentVectorSeq

Seq view over a PersistentVector, tracking the current index.

**Fields:**

- `vec`
- `index`

#### beagle.core/Stdout

Default Writer that sends output to standard out.

#### beagle.core/StringBuffer

Mutable Writer that accumulates written output into a string.

**Fields:**

- `content` `mutable`

#### beagle.core/__Box__

**Fields:**

- `value`

#### beagle.core/SystemError

**Fields:**

- `StructError`
- `ParseError`
- `CompileError`
- `TypeError`
- `ArityError`
- `RuntimeError`
- `ThreadError`
- `FFIError`
- `FunctionError`
- `FieldError`
- `IndexError`
- `MutabilityError`
- `JsonError`
- `AssertError`
- `IOError`
- `AllocationError`
- `ArgumentError`
- `InvalidArgument`
- `StringError`
- `RegexError`
- `EncodingError`
- `CopyError`

#### beagle.core/SystemError.StructError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.ParseError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.CompileError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.TypeError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.ArityError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.RuntimeError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.ThreadError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.FFIError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.FunctionError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.FieldError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.IndexError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.MutabilityError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.JsonError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.AssertError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.IOError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.AllocationError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.ArgumentError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.InvalidArgument

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.StringError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.RegexError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.EncodingError

**Fields:**

- `message`
- `location`

#### beagle.core/SystemError.CopyError

**Fields:**

- `message`
- `location`

#### beagle.core/DiagnosticSeverity

**Fields:**

- `error`
- `warning`
- `info`
- `hint`

#### beagle.core/DiagnosticSeverity.error

#### beagle.core/DiagnosticSeverity.warning

#### beagle.core/DiagnosticSeverity.info

#### beagle.core/DiagnosticSeverity.hint

#### beagle.core/Diagnostic

A compiler diagnostic (severity, kind, location, message) for programmatic inspection.

**Fields:**

- `severity`
- `kind`
- `file-name`
- `line`
- `column`
- `message`
- `enum-name`
- `missing-variants`

#### beagle.core/Range

Lazy numeric range with current value, exclusive end, and step.

**Fields:**

- `current`
- `end`
- `step`

#### beagle.core/StringSeq

Seq view over a String's characters, tracking the current index.

**Fields:**

- `str`
- `index`

#### beagle.core/PersistentSetSeq

Seq view over a PersistentSet's elements, tracking the current index.

**Fields:**

- `elements`
- `index`

#### beagle.core/PersistentMapSeq

Seq view over a PersistentMap, yielding [key value] entry pairs.

**Fields:**

- `map`
- `keys`
- `index`

#### beagle.core/Error

**Fields:**

- `NotFound`
- `PermissionDenied`
- `AlreadyExists`
- `IsDirectory`
- `NotDirectory`
- `Timeout`
- `Cancelled`
- `IO`
- `Other`

#### beagle.core/Error.NotFound

**Fields:**

- `path`

#### beagle.core/Error.PermissionDenied

**Fields:**

- `path`

#### beagle.core/Error.AlreadyExists

**Fields:**

- `path`

#### beagle.core/Error.IsDirectory

**Fields:**

- `path`

#### beagle.core/Error.NotDirectory

**Fields:**

- `path`

#### beagle.core/Error.Timeout

**Fields:**

- `operation`

#### beagle.core/Error.Cancelled

#### beagle.core/Error.IO

**Fields:**

- `message`

#### beagle.core/Error.Other

**Fields:**

- `code`
- `message`

#### beagle.core/Result

**Fields:**

- `Ok`
- `Err`

#### beagle.core/Result.Ok

**Fields:**

- `value`

#### beagle.core/Result.Err

**Fields:**

- `error`

#### beagle.ffi/Library

Handle to a dynamically loaded shared library; `id` is the runtime's
internal library identifier.

**Fields:**

- `id`

#### beagle.ffi/Pointer

A raw C pointer split into low and high 32-bit halves (`lo`, `hi`).

**Fields:**

- `lo`
- `hi`

#### beagle.ffi/Buffer

An off-heap byte buffer: `ptr` is the raw address, `size` its byte length.
Registered with the GC finalizer so the backing allocation is freed when
the Buffer becomes unreachable.

**Fields:**

- `ptr`
- `size`

#### beagle.ffi/Cell

A typed one-element off-heap cell: `ptr` raw address, `size` byte length,
`ty` the FFI element Type. GC-finalized like Buffer.

**Fields:**

- `ptr`
- `size`
- `ty`

#### beagle.ffi/TypedArray

A typed fixed-length off-heap array: `ptr` raw address, `size` total bytes,
`ty` the element Type, `length` the element count. GC-finalized like Buffer.

**Fields:**

- `ptr`
- `size`
- `ty`
- `length`

#### beagle.ffi/StructReturn

Holds a by-value C struct return of up to 16 bytes: `low` is the first
8 bytes, `high` the second 8 bytes.

**Fields:**

- `low`
- `high`

#### beagle.ffi/Type

**Fields:**

- `U8`
- `U16`
- `U32`
- `U64`
- `I8`
- `I16`
- `I32`
- `I64`
- `F32`
- `F64`
- `Pointer`
- `MutablePointer`
- `String`
- `Void`
- `Structure`

#### beagle.ffi/Type.U8

#### beagle.ffi/Type.U16

#### beagle.ffi/Type.U32

#### beagle.ffi/Type.U64

#### beagle.ffi/Type.I8

#### beagle.ffi/Type.I16

#### beagle.ffi/Type.I32

#### beagle.ffi/Type.I64

#### beagle.ffi/Type.F32

#### beagle.ffi/Type.F64

#### beagle.ffi/Type.Pointer

#### beagle.ffi/Type.MutablePointer

#### beagle.ffi/Type.String

#### beagle.ffi/Type.Void

#### beagle.ffi/Type.Structure

**Fields:**

- `types`

#### beagle.io/File

Wraps an open FILE* handle returned from libc fopen.

**Fields:**

- `handle`

#### beagle.io/BufferResult

Result of a read: a native buffer and the number of bytes read into it.

**Fields:**

- `buffer`
- `length`

#### beagle.async/FutureState

**Fields:**

- `Pending`
- `Running`
- `Resolved`
- `Rejected`
- `Cancelled`

#### beagle.async/FutureState.Pending

#### beagle.async/FutureState.Running

#### beagle.async/FutureState.Resolved

**Fields:**

- `value`

#### beagle.async/FutureState.Rejected

**Fields:**

- `error`

#### beagle.async/FutureState.Cancelled

#### beagle.async/Future

A future, holding its FutureState in an atom for thread-safe access.

**Fields:**

- `state_atom`

#### beagle.async/CancellationToken

A cooperative cancellation token wrapping a boolean atom; tasks poll it and exit early when it is set.

**Fields:**

- `cancelled_atom`

#### beagle.async/TaskScope

A structured-concurrency scope: a cancellation token plus an atom holding its child futures.

**Fields:**

- `token`
- `children_atom`

#### beagle.async/RaceResult

**Fields:**

- `Ok`
- `AllFailed`

#### beagle.async/RaceResult.Ok

**Fields:**

- `value`
- `index`

#### beagle.async/RaceResult.AllFailed

**Fields:**

- `errors`

#### beagle.async/TimeoutResult

**Fields:**

- `Ok`
- `TimedOut`

#### beagle.async/TimeoutResult.Ok

**Fields:**

- `value`

#### beagle.async/TimeoutResult.TimedOut

#### beagle.async/TcpListener

A TCP listener, wrapping its owning event-loop id and listener id.

**Fields:**

- `loop_id`
- `listener_id`

#### beagle.async/TcpSocket

A TCP socket, wrapping its owning event-loop id and socket id.

**Fields:**

- `loop_id`
- `socket_id`

#### beagle.async/TcpIoError

Sentinel value returned (not thrown) for a failed async TCP op, so tcp-result-or-throw can re-raise it in the caller's stack. Holds the error message.

**Fields:**

- `message`

#### beagle.async/IOAction

**Fields:**

- `TcpListen`
- `TcpConnect`
- `TcpAccept`
- `TcpRead`
- `TcpWrite`
- `TcpClose`
- `TcpCloseListener`

#### beagle.async/IOAction.TcpListen

**Fields:**

- `host`
- `port`

#### beagle.async/IOAction.TcpConnect

**Fields:**

- `host`
- `port`

#### beagle.async/IOAction.TcpAccept

**Fields:**

- `listener`

#### beagle.async/IOAction.TcpRead

**Fields:**

- `socket`
- `n`

#### beagle.async/IOAction.TcpWrite

**Fields:**

- `socket`
- `data`

#### beagle.async/IOAction.TcpClose

**Fields:**

- `socket`

#### beagle.async/IOAction.TcpCloseListener

**Fields:**

- `listener`

#### beagle.async/PendingOp

**Fields:**

- `TcpIO`
- `FileIO`
- `Timer`
- `SpawnedTask`
- `NewTask`

#### beagle.async/PendingOp.TcpIO

**Fields:**

- `op_id`
- `loop_id`

#### beagle.async/PendingOp.FileIO

**Fields:**

- `handle`
- `loop_id`

#### beagle.async/PendingOp.Timer

**Fields:**

- `marker`
- `loop_id`

#### beagle.async/PendingOp.SpawnedTask

**Fields:**

- `future`

#### beagle.async/PendingOp.NewTask

**Fields:**

- `thunk`

#### beagle.async/ReadyTask

A scheduler task ready to run, pairing a continuation (resume) with the result to resume it with.

**Fields:**

- `resume`
- `result`

#### beagle.async/AwaitRejected

Sentinel: the awaited future rejected; `await` re-raises `error`.

**Fields:**

- `error`

#### beagle.async/AwaitCancelled

Sentinel: the awaited future was cancelled; `await` throws.

#### beagle.async/SchedulerTask

A parked cooperative task: the PendingOp it waits on plus the continuation to resume when it completes.

**Fields:**

- `op`
- `resume`

#### beagle.async/CooperativeHandler

Cooperative single-threaded async handler state: an atom of parked tasks plus the event-loop id.

**Fields:**

- `tasks`
- `loop_id`

#### beagle.async/Async

**Fields:**

- `Await`
- `AwaitAll`
- `AwaitFirst`
- `Cancel`
- `IO`
- `ReadFile`
- `WriteFile`
- `AppendFile`
- `DeleteFile`
- `FileExists`
- `RenameFile`
- `CopyFile`
- `ReadDir`
- `CreateDir`
- `CreateDirAll`
- `RemoveDir`
- `RemoveDirAll`
- `FileSize`
- `IsDirectory`
- `IsFile`
- `Open`
- `Close`
- `Read`
- `Write`
- `ReadLine`
- `Flush`
- `Sleep`
- `Spawn`
- `SpawnWithToken`

#### beagle.async/Async.Await

**Fields:**

- `future`

#### beagle.async/Async.AwaitAll

**Fields:**

- `futures`

#### beagle.async/Async.AwaitFirst

**Fields:**

- `futures`

#### beagle.async/Async.Cancel

**Fields:**

- `future`

#### beagle.async/Async.IO

**Fields:**

- `action`

#### beagle.async/Async.ReadFile

**Fields:**

- `path`

#### beagle.async/Async.WriteFile

**Fields:**

- `path`
- `content`

#### beagle.async/Async.AppendFile

**Fields:**

- `path`
- `content`

#### beagle.async/Async.DeleteFile

**Fields:**

- `path`

#### beagle.async/Async.FileExists

**Fields:**

- `path`

#### beagle.async/Async.RenameFile

**Fields:**

- `old_path`
- `new_path`

#### beagle.async/Async.CopyFile

**Fields:**

- `src_path`
- `dest_path`

#### beagle.async/Async.ReadDir

**Fields:**

- `path`

#### beagle.async/Async.CreateDir

**Fields:**

- `path`

#### beagle.async/Async.CreateDirAll

**Fields:**

- `path`

#### beagle.async/Async.RemoveDir

**Fields:**

- `path`

#### beagle.async/Async.RemoveDirAll

**Fields:**

- `path`

#### beagle.async/Async.FileSize

**Fields:**

- `path`

#### beagle.async/Async.IsDirectory

**Fields:**

- `path`

#### beagle.async/Async.IsFile

**Fields:**

- `path`

#### beagle.async/Async.Open

**Fields:**

- `path`
- `mode`

#### beagle.async/Async.Close

**Fields:**

- `file`

#### beagle.async/Async.Read

**Fields:**

- `file`
- `n`

#### beagle.async/Async.Write

**Fields:**

- `file`
- `content`

#### beagle.async/Async.ReadLine

**Fields:**

- `file`

#### beagle.async/Async.Flush

**Fields:**

- `file`

#### beagle.async/Async.Sleep

**Fields:**

- `ms`

#### beagle.async/Async.Spawn

**Fields:**

- `thunk`

#### beagle.async/Async.SpawnWithToken

**Fields:**

- `thunk`
- `token`

#### beagle.async/BlockingAsyncHandler

Async handler that runs every operation synchronously on the current thread. Used for deterministic testing.

#### beagle.async/ImplicitAsyncHandler

The primary production async handler: threaded spawn/await combined with event-loop sleep and async file I/O.

#### beagle.fs/Fs

**Fields:**

- `ReadFile`
- `WriteFile`
- `AppendFile`
- `DeleteFile`
- `FileExists`
- `FileSize`
- `IsFile`
- `RenameFile`
- `CopyFile`
- `ReadDir`
- `CreateDir`
- `CreateDirAll`
- `RemoveDir`
- `RemoveDirAll`
- `IsDirectory`
- `Open`
- `Close`
- `Read`
- `Write`
- `ReadLine`
- `Flush`

#### beagle.fs/Fs.ReadFile

**Fields:**

- `path`

#### beagle.fs/Fs.WriteFile

**Fields:**

- `path`
- `content`

#### beagle.fs/Fs.AppendFile

**Fields:**

- `path`
- `content`

#### beagle.fs/Fs.DeleteFile

**Fields:**

- `path`

#### beagle.fs/Fs.FileExists

**Fields:**

- `path`

#### beagle.fs/Fs.FileSize

**Fields:**

- `path`

#### beagle.fs/Fs.IsFile

**Fields:**

- `path`

#### beagle.fs/Fs.RenameFile

**Fields:**

- `old_path`
- `new_path`

#### beagle.fs/Fs.CopyFile

**Fields:**

- `src_path`
- `dest_path`

#### beagle.fs/Fs.ReadDir

**Fields:**

- `path`

#### beagle.fs/Fs.CreateDir

**Fields:**

- `path`

#### beagle.fs/Fs.CreateDirAll

**Fields:**

- `path`

#### beagle.fs/Fs.RemoveDir

**Fields:**

- `path`

#### beagle.fs/Fs.RemoveDirAll

**Fields:**

- `path`

#### beagle.fs/Fs.IsDirectory

**Fields:**

- `path`

#### beagle.fs/Fs.Open

**Fields:**

- `path`
- `mode`

#### beagle.fs/Fs.Close

**Fields:**

- `file`

#### beagle.fs/Fs.Read

**Fields:**

- `file`
- `n`

#### beagle.fs/Fs.Write

**Fields:**

- `file`
- `content`

#### beagle.fs/Fs.ReadLine

**Fields:**

- `file`

#### beagle.fs/Fs.Flush

**Fields:**

- `file`

#### beagle.fs/BlockingFsHandler

Synchronous (blocking) handler for the Fs effect. Its `handle`
implementation executes each Fs operation immediately using the
underlying beagle.io/beagle.core syscalls and resumes with the Result.

#### beagle.timer/Timer

**Fields:**

- `Sleep`
- `Now`

#### beagle.timer/Timer.Sleep

**Fields:**

- `ms`

#### beagle.timer/Timer.Now

#### beagle.timer/BlockingTimerHandler

#### beagle.stream/StreamResult

**Fields:**

- `Value`
- `Done`
- `Error`

#### beagle.stream/StreamResult.Value

**Fields:**

- `value`

#### beagle.stream/StreamResult.Done

#### beagle.stream/StreamResult.Error

**Fields:**

- `error`

#### beagle.stream/Stream

Stream wrapper - provides functional combinators over a StreamSource

Never instantiate directly; use from-source() or combinator functions.

**Fields:**

- `source`
- `closed`

#### beagle.stream/MapSource

StreamSource that applies a mapper function to each upstream value.

**Fields:**

- `upstream`
- `mapper`

#### beagle.stream/FilterSource

StreamSource that yields only upstream values matching the predicate.

**Fields:**

- `upstream`
- `predicate`

#### beagle.stream/TakeSource

StreamSource that yields at most a remaining number of upstream values.

**Fields:**

- `upstream`
- `remaining`

#### beagle.stream/TakeWhileSource

StreamSource that yields upstream values while the predicate holds, then stops.

**Fields:**

- `upstream`
- `predicate`
- `done`

#### beagle.stream/SkipSource

StreamSource that discards the first to_skip upstream values, then passes through.

**Fields:**

- `upstream`
- `to_skip`

#### beagle.stream/DropWhileSource

StreamSource that drops the leading prefix matching the predicate, then passes the rest through.

**Fields:**

- `upstream`
- `predicate`
- `dropping`

#### beagle.stream/FlatMapSource

StreamSource that maps each upstream value to a stream and flattens the results.

**Fields:**

- `upstream`
- `mapper`
- `current_inner`

#### beagle.stream/FileChunkSource

StreamSource that reads a file as raw byte chunks using async I/O.

**Fields:**

- `path`
- `chunk_size`
- `file_handle`
- `eof`

#### beagle.stream/FileChunkSourceSync

StreamSource that reads a file as raw byte chunks using synchronous I/O.

**Fields:**

- `path`
- `chunk_size`
- `file_handle`
- `eof`

#### beagle.stream/DirSource

StreamSource that lazily iterates directory entry names.

**Fields:**

- `path`
- `entries`
- `index`
- `loaded`

#### beagle.stream/SplitOnSource

StreamSource that buffers bytes and yields messages split on a delimiter.

**Fields:**

- `upstream`
- `delimiter`
- `buffer`

#### beagle.stream/BySizeSource

StreamSource that buffers bytes and yields fixed-size chunks.

**Fields:**

- `upstream`
- `size`
- `buffer`

#### beagle.stream/GeneratorSource

StreamSource that pulls values from a generator function.

**Fields:**

- `generator`
- `stopped`

#### beagle.stream/MergeSource

StreamSource that interleaves two upstream streams round-robin.

**Fields:**

- `left`
- `right`
- `left_done`
- `right_done`
- `turn`

#### beagle.stream/ZipSource

StreamSource that pairs values from two upstream streams.

**Fields:**

- `left`
- `right`

#### beagle.stream/BufferedSource

StreamSource that batches upstream values into an internal buffer.

**Fields:**

- `upstream`
- `buffer_size`
- `buffer`
- `pos`

#### beagle.stream/DistinctSource

StreamSource that yields each upstream value only the first time it is seen.

**Fields:**

- `upstream`
- `seen`

#### beagle.stream/ScanSource

StreamSource that emits init then each running accumulator of a left fold.

**Fields:**

- `upstream`
- `accumulator`
- `reducer`
- `emitted_init`

#### beagle.stream/CatchSource

StreamSource that replaces an upstream error with a default value.

**Fields:**

- `upstream`
- `default`

#### beagle.stream/RetrySource

StreamSource that retries the upstream on error up to max_retries times.

**Fields:**

- `upstream`
- `max_retries`
- `retries`

#### beagle.stream/SocketChunkSource

StreamSource that reads a TCP socket as raw byte chunks.

**Fields:**

- `socket`
- `chunk_size`
- `eof`

#### beagle.repl-session/EvalRequest

A queued evaluation: the request `id`, the `code` string to eval, and
`result_atom` where the resulting response list is published.

**Fields:**

- `id`
- `code`
- `result_atom`

#### beagle.repl-session/Session

One REPL connection's evaluation state: its `id`, the dedicated
`eval_thread`, the `message_queue` of pending requests, the `is_running`
busy flag, the `cancel_token`, the `resume_stack` of suspended exceptions,
and `ns`, the session's pinned current namespace.

**Fields:**

- `id`
- `eval_thread`
- `message_queue`
- `is_running`
- `cancel_token`
- `resume_stack`
- `ns`

#### beagle.repl-main/MainEvalRequest

A queued eval request handed from a server thread to the main thread.
`result_atom` starts null; the main thread sets it to an EvalResult once done.

**Fields:**

- `id`
- `code`
- `result_atom`

#### beagle.repl-main/EvalResult

The outcome of evaluating one request: captured output, the result value
as a string, and an error message (null when evaluation succeeded).

**Fields:**

- `output`
- `value`
- `error`

#### beagle.spawn/Spawn

**Fields:**

- `Spawn`
- `SpawnWithToken`

#### beagle.spawn/Spawn.Spawn

**Fields:**

- `thunk`

#### beagle.spawn/Spawn.SpawnWithToken

**Fields:**

- `thunk`
- `token`

#### beagle.spawn/BlockingSpawnHandler

Spawn handler that runs tasks immediately and synchronously on the current
thread, so spawned futures are already settled when returned.

#### beagle.spawn/ThreadedSpawnHandler

Spawn handler that runs each task on its own OS thread for parallel
execution, returning a Future that settles when the thread finishes.

#### beagle.mutable-array/ExampleStruct

A trivial single-field struct used only by this module's demos/tests
(`allocate-array-and-return` and `main`).

**Fields:**

- `value`

#### beagle.hash/W64

A 64-bit word emulated as two 32-bit halves (`hi` = upper 32 bits, `lo` = lower 32 bits).

**Fields:**

- `hi`
- `lo`

#### beagle.url/Url

A parsed URL with fields scheme, host, port, path, query and fragment, as
returned by parse-url. port is an Int or null; query/fragment are the raw
strings after '?'/'#' or null when absent; path is "" when none is present.

**Fields:**

- `scheme`
- `host`
- `port`
- `path`
- `query`
- `fragment`

#### beagle.http/Request

A parsed HTTP request: method, path, query, version, a lowercased-key
headers map, and the body string.

**Fields:**

- `method`
- `path`
- `query`
- `version`
- `headers`
- `body`
- `params`

#### beagle.http/Response

An HTTP response to write (server) or parsed from a call (client): status
code, reason phrase, a lowercased-key headers map, and the body string.

**Fields:**

- `status`
- `reason`
- `headers`
- `body`

#### beagle.http/Route

A single router entry pairing a method + path with its handler.

**Fields:**

- `method`
- `path`
- `handler`

#### beagle.http/ConnBuf

A buffered wrapper over a raw socket connection, holding an atom of
over-read leftover bytes carried across read calls.

**Fields:**

- `conn`
- `leftover`
- `tls`

#### beagle.template/Token

A lexer token produced by `tokenize`: `kind` is "literal", "expr",
"raw-expr", or "tag"; `text` is the verbatim run or the inner
expression/directive.

**Fields:**

- `kind`
- `text`

#### beagle.template/TextNode

AST node for literal output: `text` is copied verbatim on render.

**Fields:**

- `text`

#### beagle.template/VarNode

AST node for an interpolation: `path` is the keyword lookup path; `escape`
is true to HTML-escape the looked-up value.

**Fields:**

- `path`
- `escape`

#### beagle.template/IfNode

AST node for `{% if %}`: `path` is the condition lookup; `then-body` and
`else-body` are node vectors (`else-body` is empty when there is no
`{% else %}`).

**Fields:**

- `path`
- `then-body`
- `else-body`

#### beagle.template/ForNode

AST node for `{% for %}`: `var` is the loop-binding keyword, `path` locates
the vector to iterate, and `body` is the node vector rendered per element.

**Fields:**

- `var`
- `path`
- `body`

#### beagle.template/CompiledTemplate

A pre-parsed template produced by `compile`: `nodes` is the parsed AST node
vector and `size-hint` seeds the output string-builder. Render it any number
of times with `render-compiled` without re-tokenizing or re-parsing.

**Fields:**

- `nodes`
- `size-hint`

#### beagle.ws/Frame

A parsed WebSocket frame: fin (final-fragment Bool), opcode (Int), masked
(Bool: was the payload masked), payload (unmasked byte-faithful String).

**Fields:**

- `fin`
- `opcode`
- `masked`
- `payload`

#### beagle.containers/Deque

**Fields:**

- `front`
- `back`

#### beagle.containers/DefaultMap

**Fields:**

- `default`
- `table`

#### beagle.containers/OrderedMap

**Fields:**

- `order`
- `table`

#### beagle.json/Cursor

Mutable parse cursor: source string `s`, in-place byte index `pos`, and
`keyword-keys` flag controlling whether object keys parse as keywords.

**Fields:**

- `s`
- `pos` `mutable`
- `keyword-keys`

#### beagle.bigint/BigInt

Arbitrary-precision integer. `sign` is -1, 0, or 1; `limbs` is a
least-significant-first Vec<Int> of base-10000 groups with no leading-zero
limbs. Zero is canonical: sign 0 with an empty limbs vector.

**Fields:**

- `sign`
- `limbs`

#### beagle.channel/Channel

Unbounded FIFO channel: an atom holding a persistent-vector queue plus an
atom holding a `closed` flag (false until close! flips it true).

**Fields:**

- `queue`
- `closed`

#### beagle.channel/_ChannelQueue

**Fields:**

- `sendq`
- `recvq`

#### beagle.channel/Mutex

CAS-acquired lock: wraps an atom holding a bool (false = unlocked, true = locked).

**Fields:**

- `locked`

#### beagle.channel/Counter

Atomic integer counter: wraps an atom holding an int.

**Fields:**

- `value`

#### beagle.date/DateTime

A broken-down UTC calendar date-time: year, month (1-12), day (1-31),
hour (0-23), minute (0-59), second (0-59). All fields are plain integers.

**Fields:**

- `year`
- `month`
- `day`
- `hour`
- `minute`
- `second`

#### beagle.priorityqueue/PriorityQueue

A binary min-heap priority queue. `items` is a persistent vector holding
the heap (children of index i at 2i+1/2i+2); `compare` is the ordering
function (negative => first arg pops before second). Immutable: operations
return a new PriorityQueue.

**Fields:**

- `items`
- `compare`

#### beagle.semver/SemVer

A parsed Semantic Version. `major`/`minor`/`patch` are ints; `prerelease`
and `build` are the raw strings after the `-`/`+` markers (null when absent).

**Fields:**

- `major`
- `minor`
- `patch`
- `prerelease`
- `build`

---

## Enums

These are the built-in enum types available in Beagle.

#### beagle.core/SystemError

Built-in error variants raised by the runtime; each carries a message and location.

**Variants:**

- `StructError { message, location }`
- `ParseError { message, location }`
- `CompileError { message, location }`
- `TypeError { message, location }`
- `ArityError { message, location }`
- `RuntimeError { message, location }`
- `ThreadError { message, location }`
- `FFIError { message, location }`
- `FunctionError { message, location }`
- `FieldError { message, location }`
- `IndexError { message, location }`
- `MutabilityError { message, location }`
- `JsonError { message, location }`
- `AssertError { message, location }`
- `IOError { message, location }`
- `AllocationError { message, location }`
- `ArgumentError { message, location }`
- `InvalidArgument { message, location }`
- `StringError { message, location }`
- `RegexError { message, location }`
- `EncodingError { message, location }`
- `CopyError { message, location }`

#### beagle.core/DiagnosticSeverity

Severity levels for compiler diagnostics: error, warning, info, hint.

**Variants:**

- `error`
- `warning`
- `info`
- `hint`

#### beagle.core/Error

Canonical error type for async and I/O operations; pattern-match to handle each case.

**Variants:**

- `NotFound { path }`
- `PermissionDenied { path }`
- `AlreadyExists { path }`
- `IsDirectory { path }`
- `NotDirectory { path }`
- `Timeout { operation }`
- `Cancelled`
- `IO { message }`
- `Other { code, message }`

#### beagle.core/Result

Canonical Ok/Err result type used across async and I/O operations.

**Variants:**

- `Ok { value }`
- `Err { error }`

#### beagle.ffi/Type

Enumeration of FFI value types: the fixed-width integers (U8..U64, I8..I64),
floats (F32/F64), Pointer/MutablePointer, String, Void, and a nested
`Structure { types }` for aggregate layouts.

**Variants:**

- `U8`
- `U16`
- `U32`
- `U64`
- `I8`
- `I16`
- `I32`
- `I64`
- `F32`
- `F64`
- `Pointer`
- `MutablePointer`
- `String`
- `Void`
- `Structure { types }`

#### beagle.async/FutureState

A future's lifecycle state: Pending, Running, Resolved (with value), Rejected (with error), or Cancelled.

**Variants:**

- `Pending`
- `Running`
- `Resolved { value }`
- `Rejected { error }`
- `Cancelled`

#### beagle.async/RaceResult

Result of await-first: Ok with the winning value and its index, or AllFailed with the collected errors.

**Variants:**

- `Ok { value, index }`
- `AllFailed { errors }`

#### beagle.async/TimeoutResult

Result of a timed await: Ok with the value, or TimedOut if the deadline passed.

**Variants:**

- `Ok { value }`
- `TimedOut`

#### beagle.async/IOAction

Describes a TCP I/O operation for the async handler to perform (listen, connect, accept, read, write, close).

**Variants:**

- `TcpListen { host, port }`
- `TcpConnect { host, port }`
- `TcpAccept { listener }`
- `TcpRead { socket, n }`
- `TcpWrite { socket, data }`
- `TcpClose { socket }`
- `TcpCloseListener { listener }`

#### beagle.async/PendingOp

A pending cooperative-scheduler operation: TCP I/O, file I/O, a timer, a spawned-task wait, or a new task to start.

**Variants:**

- `TcpIO { op_id, loop_id }`
- `FileIO { handle, loop_id }`
- `Timer { marker, loop_id }`
- `SpawnedTask { future }`
- `NewTask { thunk }`

#### beagle.async/Async

The Async effect: futures (await/cancel), generic TCP I/O, file and directory operations, low-level file handles, sleep, and spawning.

**Variants:**

- `Await { future }`
- `AwaitAll { futures }`
- `AwaitFirst { futures }`
- `Cancel { future }`
- `IO { action }`
- `ReadFile { path }`
- `WriteFile { path, content }`
- `AppendFile { path, content }`
- `DeleteFile { path }`
- `FileExists { path }`
- `RenameFile { old_path, new_path }`
- `CopyFile { src_path, dest_path }`
- `ReadDir { path }`
- `CreateDir { path }`
- `CreateDirAll { path }`
- `RemoveDir { path }`
- `RemoveDirAll { path }`
- `FileSize { path }`
- `IsDirectory { path }`
- `IsFile { path }`
- `Open { path, mode }`
- `Close { file }`
- `Read { file, n }`
- `Write { file, content }`
- `ReadLine { file }`
- `Flush { file }`
- `Sleep { ms }`
- `Spawn { thunk }`
- `SpawnWithToken { thunk, token }`

#### beagle.fs/Fs

The filesystem effect. Each variant is a distinct file or directory
operation that is dispatched to whatever Handler(Fs) is installed.
Performed indirectly by the wrapper functions below (read-file, etc.).

**Variants:**

- `ReadFile { path }`
- `WriteFile { path, content }`
- `AppendFile { path, content }`
- `DeleteFile { path }`
- `FileExists { path }`
- `FileSize { path }`
- `IsFile { path }`
- `RenameFile { old_path, new_path }`
- `CopyFile { src_path, dest_path }`
- `ReadDir { path }`
- `CreateDir { path }`
- `CreateDirAll { path }`
- `RemoveDir { path }`
- `RemoveDirAll { path }`
- `IsDirectory { path }`
- `Open { path, mode }`
- `Close { file }`
- `Read { file, n }`
- `Write { file, content }`
- `ReadLine { file }`
- `Flush { file }`

#### beagle.timer/Timer

**Variants:**

- `Sleep { ms }`
- `Now`

#### beagle.stream/StreamResult

Stream result - outcome of pulling the next value

**Variants:**

- `Value { value }`
- `Done`
- `Error { error }`

#### beagle.spawn/Spawn

Effect operations for spawning tasks. `Spawn` carries a thunk to run;
`SpawnWithToken` also carries a cancellation token. Performed via the
`spawn`/`spawn-with-token` helpers and interpreted by an installed handler.

**Variants:**

- `Spawn { thunk }`
- `SpawnWithToken { thunk, token }`

