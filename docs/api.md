# Beagle API Reference

> Auto-generated documentation for the Beagle programming language.

**Overall documentation coverage:** 304/643 functions (47%)

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

- [beagle.core](#beagle-core) (177/295 documented)
- [beagle.fs](#beagle-fs) (27/59 documented)
- [beagle.async](#beagle-async) (16/139 documented)
- [beagle.io](#beagle-io) (16/20 documented)
- [beagle.timer](#beagle-timer) (8/10 documented)
- [beagle.regex](#beagle-regex) (9/9 documented)
- [beagle.reflect](#beagle-reflect) (17/17 documented)
- [beagle.collections](#beagle-collections) (15/15 documented)
- [beagle.effect](#beagle-effect) (0/1 documented)
- [beagle.ffi](#beagle-ffi) (19/21 documented)
- [beagle.simple-socket](#beagle-simple-socket) (0/8 documented)
- [beagle.socket](#beagle-socket) (0/7 documented)
- [global](#global) (0/42 documented)

### [Types](#types)

### [Enums](#enums-1)


---

## beagle.core

> **Documentation coverage:** 177/295 functions (60%)

#### `_println(value)` `builtin`

Print a value followed by a newline to standard output.

#### `_print(value)` `builtin`

Print a value to standard output without a trailing newline.

#### `to-string(value)` `builtin`

Convert any value to its string representation.

Examples:
  (to-string 42)     ; => "42"
  (to-string true)   ; => "true"
  (to-string [1 2])  ; => "[1, 2]"

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

#### `gc()` `builtin`

Trigger garbage collection manually.

Normally GC runs automatically, but this can be useful for testing or freeing memory at specific points.

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
  (eval "(+ 1 2)")  ; => 3

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

#### `substring(string, start, length)` `builtin`

Extract a substring from a string.

Arguments:
  string - The source string
  start  - Starting index (0-based)
  length - Number of characters to extract

Examples:
  (substring "hello" 1 3)  ; => "ell"

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

#### `contains?(string, substr)` `builtin`

Check if a string contains a substring.

Examples:
  (contains? "hello" "ell")  ; => true

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

#### `deref(atom)`

Dereference an atom to get its current value.

Atoms provide thread-safe mutable state. Use `deref` to read the value.

Examples:
  (let a (atom 0))
  (deref a)  ; => 0

#### `swap!(atom, f)`

Atomically update an atom's value by applying a function.

Takes the current value, applies f to it, and attempts to set the new value.
Retries if another thread changed the value concurrently.

Examples:
  (let counter (atom 0))
  (swap! counter fn(x) { x + 1 })  ; => 1

#### `reset!(atom, value)`

Reset an atom to a new value unconditionally.

Unlike `swap!`, this does not read the current value first.

Examples:
  (let a (atom 0))
  (reset! a 42)
  (deref a)  ; => 42

#### `compare-and-swap!(atom, old, new)`

Atomically compare and swap an atom's value.

If the atom's current value equals `old`, set it to `new` and return true.
Otherwise return false without changing the value.

Examples:
  (let a (atom 0))
  (compare-and-swap! a 0 1)  ; => true, a is now 1
  (compare-and-swap! a 0 2)  ; => false, a is still 1

#### `atom(value)`

Create a new atom with the given initial value.

Atoms provide thread-safe mutable state.

Examples:
  (let counter (atom 0))
  (swap! counter fn(x) { x + 1 })
  (deref counter)  ; => 1

#### `join(coll, sep)`

Join elements of a collection into a string with a separator.

Works with any collection that implements Indexed and Length protocols.

Examples:
  (join [1 2 3] ", ")  ; => "1, 2, 3"
  (join ["a" "b"] "-") ; => "a-b"

#### `parse-int(s)`

Parse a string into an integer.

Returns null if the string is not a valid integer.
Supports negative numbers with leading '-'.

Examples:
  (parse-int "42")   ; => 42
  (parse-int "-5")   ; => -5
  (parse-int "abc")  ; => null

#### `print(...args)`

Print arguments separated by spaces without a trailing newline.

Accepts any number of arguments. Each is formatted and printed.
Returns the last argument.

Examples:
  (print "Hello" "World")  ; prints: Hello World

#### `println(...args)`

Print arguments separated by spaces with a trailing newline.

Accepts any number of arguments. Each is formatted and printed.
Returns the last argument.

Examples:
  (println "Hello" "World")  ; prints: Hello World\n
  (println 1 2 3)            ; prints: 1 2 3\n

#### `keyword?(value)`

Check if a value is a keyword.

Examples:
  (keyword? :foo)  ; => true
  (keyword? "foo") ; => false

#### `keyword->string(kw)`

Convert a keyword to a string.

Examples:
  (keyword->string :foo)  ; => "foo"

#### `string->keyword(str)`

Convert a string to a keyword.

Examples:
  (string->keyword "foo")  ; => :foo

#### `range(start, end)`

Create a range from start (inclusive) to end (exclusive).

Ranges are iterable and can be used with for loops and seq functions.

Examples:
  (for i in (range 0 5) { println(i) })  ; prints 0, 1, 2, 3, 4
  (reduce (range 1 5) 0 fn(a x) { a + x })  ; => 10

#### `range-step(start, end, step)`

Create a range with a custom step value.

Examples:
  (range-step 0 10 2)  ; => 0, 2, 4, 6, 8

#### `reduce(coll, init, f)`

Fold a collection with an accumulator function.

Calls (f acc elem) for each element, threading the accumulator through.
Works with any Seqable type.

Examples:
  (reduce [1 2 3] 0 fn(acc x) { acc + x })  ; => 6
  (reduce [1 2 3] [] fn(acc x) { push acc (* x 2) })  ; => [2 4 6]

#### `map(coll, f)`

Apply a function to each element of a collection.

Returns a new vector with the transformed elements.

Examples:
  (map [1 2 3] fn(x) { x * 2 })  ; => [2 4 6]
  (map ["a" "b"] uppercase)      ; => ["A" "B"]

#### `filter(coll, pred)`

Keep elements where the predicate returns true.

Returns a new vector containing only matching elements.

Examples:
  (filter [1 2 3 4] even?)  ; => [2 4]
  (filter ["" "a" ""] fn(s) { length(s) > 0 })  ; => ["a"]

#### `any?(coll, pred)`

Check if any element matches the predicate.

Returns true if pred returns true for at least one element.

Examples:
  (any? [1 2 3] even?)  ; => true
  (any? [1 3 5] even?)  ; => false

#### `all?(coll, pred)`

Check if all elements match the predicate.

Returns true if pred returns true for every element.

Examples:
  (all? [2 4 6] even?)  ; => true
  (all? [1 2 3] even?)  ; => false

#### `none?(coll, pred)`

Check if no elements match the predicate.

Returns true if pred returns false for every element.

Examples:
  (none? [1 3 5] even?)  ; => true
  (none? [1 2 3] even?)  ; => false

#### `not-every?(coll, pred)`

Check if at least one element does NOT match the predicate.

Equivalent to (not (all? coll pred)).

Examples:
  (not-every? [1 2 3] even?)  ; => true
  (not-every? [2 4 6] even?)  ; => false

#### `find(coll, pred)`

Find the first element matching the predicate.

Returns the element or null if no match.

Examples:
  (find [1 2 3 4] even?)  ; => 2
  (find [1 3 5] even?)    ; => null

#### `find-index(coll, pred)`

Find the index of the first element matching the predicate.

Returns the index or -1 if no match.

Examples:
  (find-index [1 2 3 4] even?)  ; => 1
  (find-index [1 3 5] even?)    ; => -1

#### `take(coll, n)`

Take the first n elements from a collection.

Returns a new vector with at most n elements.

Examples:
  (take [1 2 3 4 5] 3)  ; => [1 2 3]
  (take [1 2] 5)        ; => [1 2]

#### `drop(coll, n)`

Drop the first n elements from a collection.

Returns a new vector with the remaining elements.

Examples:
  (drop [1 2 3 4 5] 2)  ; => [3 4 5]
  (drop [1 2] 5)        ; => []

#### `take-while(coll, pred)`

Take elements while the predicate returns true.

Stops at the first element where pred returns false.

Examples:
  (take-while [1 2 3 4 1] fn(x) { x < 4 })  ; => [1 2 3]

#### `drop-while(coll, pred)`

Drop elements while the predicate returns true.

Returns elements starting from the first where pred returns false.

Examples:
  (drop-while [1 2 3 4 1] fn(x) { x < 3 })  ; => [3 4 1]

#### `slice(coll, start, end)`

Get a sub-collection from start (inclusive) to end (exclusive).

Examples:
  (slice [0 1 2 3 4] 1 4)  ; => [1 2 3]

#### `enumerate(coll)`

Return pairs of [index, element] for each element.

Examples:
  (enumerate ["a" "b" "c"])  ; => [[0 "a"] [1 "b"] [2 "c"]]

#### `remove-at(coll, idx)`

Remove element at the specified index.

Returns the collection unchanged if index is out of bounds.

Examples:
  (remove-at [1 2 3 4] 1)  ; => [1 3 4]

#### `count(coll)`

Return the number of elements in a collection.

Alias for length, more idiomatic for collections.

Examples:
  (count [1 2 3])  ; => 3
  (count "hello")  ; => 5

#### `empty?(coll)`

Check if a collection has no elements.

Examples:
  (empty? [])    ; => true
  (empty? [1])   ; => false
  (empty? "")    ; => true

#### `first-of(coll)`

Get the first element of a collection.

Returns null if the collection is empty.
Named differently to avoid conflict with Seq protocol's first.

Examples:
  (first-of [1 2 3])  ; => 1
  (first-of [])       ; => null

#### `last(coll)`

Get the last element of a collection.

Returns null if the collection is empty.

Examples:
  (last [1 2 3])  ; => 3
  (last [])       ; => null

#### `rest(coll)`

Get all elements except the first.

Examples:
  (rest [1 2 3])  ; => [2 3]
  (rest [1])      ; => []

#### `butlast(coll)`

Get all elements except the last one.

Examples:
  (butlast [1 2 3])  ; => [1 2]
  (butlast [1])      ; => []

#### `nth(coll, n)`

Get the element at index n with bounds checking.

Returns null if index is out of bounds.

Examples:
  (nth [1 2 3] 1)   ; => 2
  (nth [1 2 3] 10)  ; => null

#### `second(coll)`

Get the second element of a collection.

Examples:
  (second [1 2 3])  ; => 2

#### `third(coll)`

Get the third element of a collection.

Examples:
  (third [1 2 3])  ; => 3

#### `min-of(coll)`

Return the minimum element in a collection.

Elements must be comparable with <.

Examples:
  (min-of [3 1 4 1 5])  ; => 1
  (min-of [])           ; => null

#### `max-of(coll)`

Return the maximum element in a collection.

Elements must be comparable with >.

Examples:
  (max-of [3 1 4 1 5])  ; => 5
  (max-of [])           ; => null

#### `min-by(coll, f)`

Return the element with the minimum key value.

Compares elements by applying f to each and using < on the results.

Examples:
  (min-by ["abc" "a" "ab"] length)  ; => "a"

#### `max-by(coll, f)`

Return the element with the maximum key value.

Compares elements by applying f to each and using > on the results.

Examples:
  (max-by ["a" "abc" "ab"] length)  ; => "abc"

#### `reduce-right(coll, init, f)`

Reduce a collection from right to left.

Like reduce, but processes elements in reverse order.

Examples:
  (reduce-right [1 2 3] [] fn(acc x) { push(acc, x) })  ; => [3 2 1]

#### `concat(coll1, coll2)`

Concatenate two collections into one.

Returns a new vector with elements from both collections.

Examples:
  (concat [1 2] [3 4])  ; => [1 2 3 4]

#### `flatten(coll)`

Flatten one level of nesting in a collection.

Examples:
  (flatten [[1 2] [3 4]])  ; => [1 2 3 4]

#### `flat-map(coll, f)`

Map a function over a collection then flatten the result.

Examples:
  (flat-map [1 2 3] fn(x) { [x, x] })  ; => [1 1 2 2 3 3]

#### `zip(coll1, coll2)`

Pair up elements from two collections into a vector of pairs.

Examples:
  (zip [1 2 3] ["a" "b" "c"])  ; => [[1 "a"] [2 "b"] [3 "c"]]

#### `zip-with(coll1, coll2, f)`

Combine elements from two collections using a function.

Like zip, but applies f to each pair instead of creating tuples.

Examples:
  (zip-with [1 2 3] [10 20 30] fn(a b) { a + b })  ; => [11 22 33]

#### `zipmap(keys-coll, vals-coll)`

Create a map from parallel key and value collections.

Pairs up keys[i] with vals[i] to form map entries.

Examples:
  (zipmap [:a :b :c] [1 2 3])  ; => {:a 1, :b 2, :c 3}

#### `interleave(coll1, coll2)`

Interleave elements from two collections.

Returns a vector alternating between elements of coll1 and coll2.

Examples:
  (interleave [1 2 3] [:a :b :c])  ; => [1 :a 2 :b 3 :c]

#### `interpose(coll, sep)`

Insert a separator between each element of a collection.

Examples:
  (interpose [1 2 3] 0)  ; => [1 0 2 0 3]

#### `into(target, source)`

Pour elements from source collection into target collection.

For vectors: appends elements. For maps: source should be [key, value] pairs.
For sets: adds elements.

Examples:
  (into [1 2] [3 4])          ; => [1 2 3 4]
  (into {} [[:a 1] [:b 2]])   ; => {:a 1, :b 2}
  (into #{1 2} [2 3 4])       ; => #{1 2 3 4}

#### `reverse(coll)`

Reverse the order of elements in a collection.

Returns a new vector with elements in reverse order.

Examples:
  (reverse [1 2 3])  ; => [3 2 1]

#### `repeat(value, n)`

Create a vector with a value repeated n times.

Examples:
  (repeat "x" 3)  ; => ["x" "x" "x"]

#### `repeatedly(f, n)`

Call a function n times and collect the results.

Examples:
  (repeatedly fn() { random() } 3)  ; => [0.1 0.7 0.4] (random values)

#### `iterate(f, init, n)`

Generate a sequence by repeatedly applying a function.

Starts with init, then f(init), f(f(init)), etc. for n values.

Examples:
  (iterate fn(x) { x * 2 } 1 5)  ; => [1 2 4 8 16]

#### `partition(coll, n)`

Split a collection into groups of n elements.

Examples:
  (partition [1 2 3 4 5 6] 2)  ; => [[1 2] [3 4] [5 6]]

#### `partition-by(coll, f)`

Split a collection when the key function's return value changes.

Groups consecutive elements with the same key value.

Examples:
  (partition-by [1 1 2 2 1] identity)  ; => [[1 1] [2 2] [1]]

#### `group-by(coll, f)`

Group elements by the result of a key function.

Returns a map from keys to vectors of elements with that key.

Examples:
  (group-by [1 2 3 4 5] even?)  ; => {true: [2 4], false: [1 3 5]}

#### `frequencies(coll)`

Count occurrences of each value in a collection.

Returns a map from values to their counts.

Examples:
  (frequencies [1 2 1 3 1 2])  ; => {1: 3, 2: 2, 3: 1}

#### `distinct(coll)`

Remove duplicate values from a collection.

Keeps the first occurrence of each value.

Examples:
  (distinct [1 2 1 3 2 1])  ; => [1 2 3]

#### `dedupe(coll)`

Remove consecutive duplicate elements.

Keeps the first occurrence of each run of duplicates.

Examples:
  (dedupe [1 1 2 2 2 1 1])  ; => [1 2 1]

#### `sort(coll)`

Sort a collection in ascending order.

Elements must be comparable. Uses insertion sort.

Examples:
  (sort [3 1 4 1 5])  ; => [1 1 3 4 5]

#### `sort-by(coll, f)`

Sort a collection by a key function.

Elements are ordered by comparing (f elem) values.

Examples:
  (sort-by [{:name "bob" :age 30} {:name "alice" :age 25}] fn(x) { x.age })
  ; => [{:name "alice" :age 25} {:name "bob" :age 30}]

#### `identity(x)`

Return the input unchanged.

Useful as a default function argument.

Examples:
  (identity 42)  ; => 42
  (map [1 2 3] identity)  ; => [1 2 3]

#### `constantly(value)`

Return a function that always returns the given value.

Ignores any arguments passed to the returned function.

Examples:
  (let always-42 (constantly 42))
  (always-42 "ignored")  ; => 42

#### `complement(pred)`

Return a function that negates a predicate.

Examples:
  (let not-even? (complement even?))
  (not-even? 3)  ; => true

#### `compose(f, g)`

Compose two functions: (compose f g) returns fn(x) { f(g(x)) }.

The second function is applied first, then the first.

Examples:
  (let add1-then-double (compose fn(x) { x * 2 } fn(x) { x + 1 }))
  (add1-then-double 3)  ; => 8  (3+1=4, 4*2=8)

#### `partial(f, a)`

Partially apply a function by fixing its first argument.

Examples:
  (let add10 (partial fn(a b) { a + b } 10))
  (add10 5)  ; => 15

#### `not(x)`

Boolean negation.

Returns true if x is falsy, false if x is truthy.

Examples:
  (not true)   ; => false
  (not false)  ; => true
  (not null)   ; => true

#### `truthy?(x)`

Check if a value is truthy (not null and not false).

Examples:
  (truthy? 1)      ; => true
  (truthy? null)   ; => false
  (truthy? false)  ; => false

#### `falsy?(x)`

Check if a value is falsy (null or false).

Examples:
  (falsy? null)   ; => true
  (falsy? false)  ; => true
  (falsy? 0)      ; => false

#### `nil?(x)`

Check if a value is null.

Examples:
  (nil? null)  ; => true
  (nil? 0)     ; => false

#### `some?(x)`

Check if a value is not null.

Examples:
  (some? 0)     ; => true
  (some? null)  ; => false

#### `vals(m)`

Get all values from a map as a vector.

Examples:
  (vals {:a 1 :b 2})  ; => [1 2]

#### `dissoc(m, key)`

Remove a key from a map.

Returns a new map without the specified key.

Examples:
  (dissoc {:a 1 :b 2} :a)  ; => {:b 2}

#### `merge(m1, m2)`

Merge two maps together.

m2's values take precedence for duplicate keys.

Examples:
  (merge {:a 1 :b 2} {:b 3 :c 4})  ; => {:a 1 :b 3 :c 4}

#### `merge-with(f, m1, m2)`

Merge two maps using a function to resolve key conflicts.

When a key exists in both maps, calls f(v1, v2) to get the merged value.

Examples:
  (merge-with fn(a b) { a + b } {:a 1} {:a 2 :b 3})  ; => {:a 3, :b 3}

#### `select-keys(m, ks)`

Keep only the specified keys from a map.

Returns a new map with only the keys in ks.

Examples:
  (select-keys {:a 1 :b 2 :c 3} [:a :c])  ; => {:a 1, :c 3}

#### `update(m, key, f)`

Update a value at a key using a function.

Applies f to the current value at key and associates the result.

Examples:
  (update {:a 1} :a fn(x) { x + 1 })  ; => {:a 2}

#### `get-in(m, path)`

Get a nested value using a path of keys.

Returns null if any key in the path is not found.

Examples:
  (get-in {:a {:b {:c 42}}} [:a :b :c])  ; => 42
  (get-in {:a {:b 1}} [:a :x])           ; => null

#### `assoc-in(m, path, val)`

Set a nested value using a path of keys.

Creates intermediate maps as needed.

Examples:
  (assoc-in {:a {:b 1}} [:a :b] 42)  ; => {:a {:b 42}}
  (assoc-in {} [:a :b :c] 1)         ; => {:a {:b {:c 1}}}

#### `update-in(m, path, f)`

Update a nested value using a path of keys and a function.

Applies f to the current value at the path.

Examples:
  (update-in {:a {:b 1}} [:a :b] fn(x) { x + 1 })  ; => {:a {:b 2}}

#### `contains-key?(m, key)`

Check if a map contains a key.

Examples:
  (contains-key? {:a 1 :b 2} :a)  ; => true
  (contains-key? {:a 1 :b 2} :c)  ; => false

#### `invert(m)`

Swap keys and values in a map.

Values become keys and keys become values.

Examples:
  (invert {:a 1 :b 2})  ; => {1: :a, 2: :b}

#### `map-keys(m, f)`

Apply a function to all keys in a map.

Returns a new map with transformed keys, same values.

Examples:
  (map-keys {:a 1 :b 2} keyword->string)  ; => {"a": 1, "b": 2}

#### `map-vals(m, f)`

Apply a function to all values in a map.

Returns a new map with same keys, transformed values.

Examples:
  (map-vals {:a 1 :b 2} fn(x) { x * 2 })  ; => {:a 2, :b 4}

#### `filter-keys(m, pred)`

Keep only entries where the key satisfies the predicate.

Examples:
  (filter-keys {:a 1 :ab 2 :abc 3} fn(k) { length(keyword->string(k)) > 1 })
  ; => {:ab 2, :abc 3}

#### `filter-vals(m, pred)`

Keep only entries where the value satisfies the predicate.

Examples:
  (filter-vals {:a 1 :b 2 :c 3} fn(v) { v > 1 })  ; => {:b 2, :c 3}

#### `set?(x)`

Check if a value is a PersistentSet.

Examples:
  (set? #{1 2 3})  ; => true
  (set? [1 2 3])   ; => false

#### `set-contains?(set, elem)`

Check if an element is in a set.

Examples:
  (set-contains? #{1 2 3} 2)  ; => true
  (set-contains? #{1 2 3} 5)  ; => false

#### `set-add(set, elem)`

Add an element to a set.

Returns a new set with the element added (no-op if already present).

Examples:
  (set-add #{1 2} 3)  ; => #{1 2 3}
  (set-add #{1 2} 2)  ; => #{1 2}

#### `set-remove(set, elem)`

Remove an element from a set.

Returns a new set without the element.

Examples:
  (set-remove #{1 2 3} 2)  ; => #{1 3}

#### `set-union(s1, s2)`

Return the union of two sets.

Contains all elements from both sets.

Examples:
  (set-union #{1 2} #{2 3})  ; => #{1 2 3}

#### `set-intersection(s1, s2)`

Return the intersection of two sets.

Contains only elements present in both sets.

Examples:
  (set-intersection #{1 2 3} #{2 3 4})  ; => #{2 3}

#### `set-difference(s1, s2)`

Return the difference of two sets.

Contains elements in s1 but not in s2.

Examples:
  (set-difference #{1 2 3} #{2 3 4})  ; => #{1}

#### `set-subset?(s1, s2)`

Check if s1 is a subset of s2.

Returns true if all elements of s1 are also in s2.

Examples:
  (set-subset? #{1 2} #{1 2 3})  ; => true
  (set-subset? #{1 4} #{1 2 3})  ; => false

#### `into-set(coll)`

Convert a collection to a set.

Removes duplicates since sets only contain unique elements.

Examples:
  (into-set [1 2 2 3 3 3])  ; => #{1 2 3}

#### `char-code(s)`

Get the Unicode code point of the first character in a string.

Examples:
  (char-code "A")  ; => 65
  (char-code "a")  ; => 97

#### `char-from-code(code)`

Create a single-character string from a Unicode code point.

Examples:
  (char-from-code 65)  ; => "A"
  (char-from-code 97)  ; => "a"

#### `ok(value)`

Create a successful Result containing a value.

Examples:
  (ok 42)  ; => Result.Ok { value: 42 }

#### `err(error)`

Create an error Result containing an Error.

Examples:
  (err (Error.NotFound { path: "/missing" }))

#### `ok?(result)`

Check if a Result is successful (Ok variant).

Examples:
  (ok? (ok 42))                      ; => true
  (ok? (err (Error.IO { message: "failed" })))  ; => false

#### `unwrap(result)`

Unwrap a Result, returning the value or null on error.

Prints an error message if the result is an Err.

Examples:
  (unwrap (ok 42))  ; => 42
  (unwrap (err (Error.NotFound { path: "/x" })))  ; prints error, returns null

#### `unwrap-or(result, default)`

Unwrap a Result, returning the value or a default on error.

Examples:
  (unwrap-or (ok 42) 0)                        ; => 42
  (unwrap-or (err (Error.IO { message: "" })) 0)  ; => 0

<details>
<summary><strong>Undocumented functions (118)</strong></summary>

- `event-loop-create()`
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
- `tcp-result-future-atom()`
- `tcp-result-value()`
- `tcp-result-data()`
- `tcp-result-op-id()`
- `tcp-result-listener-id()`
- `timer-set()`
- `timer-cancel()`
- `timer-completed-count()`
- `timer-pop-completed()`
- `file-read-submit()`
- `file-write-submit()`
- `file-delete-submit()`
- `file-stat-submit()`
- `file-readdir-submit()`
- `file-results-count()`
- `file-result-ready()`
- `file-result-poll-type()`
- `file-result-get-string()`
- `file-result-get-value()`
- `file-result-consume()`
- `file-result-get-entries()`
- `string-concat()`
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
- `get(coll, index)`
- `push(coll, value)`
- `length(coll)`
- `format(self, depth)`
- `assoc(coll, key, value)`
- `keys(coll)`
- `seq(coll)`
- `first(seq)`
- `next(seq)`
- `format-rust-vec-helper(vec, idx, len, acc)`
- `format-rust-map-entries(map, keys-vec, idx, len, acc)`
- `instance-of(value, type)`
- `format-persistent-vector-helper(vec, idx, acc, depth)`
- `join-helper(coll, sep, idx, acc)`
- `parse-int-helper(s, idx, acc)`
- `insert-sorted(sorted, elem)`
- `insert-sorted-by(sorted, elem, elem-key, key-fn)`
- `merge-with-helper(f, m1, m2, ks, idx, acc)`
- `select-keys-helper(m, ks, idx, acc)`
- `map-keys-helper(m, f, ks, idx, acc)`
- `map-vals-helper(m, f, ks, idx, acc)`
- `filter-keys-helper(m, pred, ks, idx, acc)`
- `filter-vals-helper(m, pred, ks, idx, acc)`
- `format-rust-set-elements(elems-vec, idx, len, acc)`
- `set-remove-helper(elems, to-remove, idx, len, acc)`
- `set-union-helper(s1, elems2, idx, len)`
- `set-intersection-helper(elems1, s2, idx, len, acc)`
- `set-difference-helper(elems1, s2, idx, len, acc)`
- `set-subset-helper(elems1, s2, idx, len)`
- `err-io(message)`
- `err-code(code, message)`
- `timeout?(result)`
- `not-found?(result)`
- `get-error(result)`
- `Struct_format(self, depth)`
- `Format_format(self, depth)`
- `String_format(self, depth)`
- `PersistentVector_get(vec, i)`
- `PersistentVector_length(vec)`
- `PersistentVector_push(vec, value)`
- `PersistentVector_format(self, depth)`
- `PersistentVector_assoc(vec, index, value)`
- `PersistentMap_get(m, key)`
- `PersistentMap_length(m)`
- `PersistentMap_assoc(m, key, value)`
- `PersistentMap_keys(map)`
- `PersistentMap_format(map, depth)`
- `String_get(str, i)`
- `String_length(str)`
- `Array_get(arr, i)`
- `Array_length(arr)`
- `PersistentVector_seq(vec)`
- `PersistentVectorSeq_first(seq)`
- `PersistentVectorSeq_next(seq)`
- `Range_seq(r)`
- `Range_first(r)`
- `Range_next(r)`
- `String_seq(str)`
- `StringSeq_first(seq)`
- `StringSeq_next(seq)`
- `PersistentSet_length(set)`
- `PersistentSet_format(set, depth)`
- `PersistentSetSeq_first(seq)`
- `PersistentSetSeq_next(seq)`
- `PersistentSet_seq(set)`
- `Regex_format(regex, depth)`

</details>

---

## beagle.fs

> **Documentation coverage:** 27/59 functions (45%)

#### `read-file(path)`

Read the entire contents of a file as a string.

Must be called within an Fs effect handler block.
Returns Result.Ok with content or Result.Err on failure.

Examples:
  handle effect/Handler(Fs) with handler {
      match (fs/read-file "/tmp/test.txt") {
          Result.Ok { value } => println(value),
          Result.Err { error } => println("Error:", error)
      }
  }

#### `write-file(path, content)`

Write content to a file (creates or overwrites).

Returns Result.Ok with bytes written, or Result.Err on failure.

Examples:
  (fs/write-file "/tmp/test.txt" "Hello, World!")

#### `append-file(path, content)`

Append content to a file (creates if doesn't exist).

Returns Result.Ok with bytes written, or Result.Err on failure.

Examples:
  (fs/append-file "/tmp/log.txt" "New log entry\n")

#### `delete-file(path)`

Delete a file.

Returns Result.Ok on success, or Result.Err if file doesn't exist.

Examples:
  (fs/delete-file "/tmp/test.txt")

#### `exists?(path)`

Check if a file or directory exists.

Returns Result.Ok { value: true/false }.

Examples:
  (fs/exists? "/tmp/test.txt")  ; => Result.Ok { value: true }

#### `file-size(path)`

Get file size in bytes.

Returns Result.Ok with the size, or Result.Err if file doesn't exist.

Examples:
  (fs/file-size "/tmp/test.txt")  ; => Result.Ok { value: 1024 }

#### `is-file?(path)`

Check if path is a regular file (not a directory).

Returns Result.Ok { value: true/false }.

Examples:
  (fs/is-file? "/tmp/test.txt")  ; => Result.Ok { value: true }

#### `rename(old-path, new-path)`

Rename or move a file or directory.

Returns Result.Ok on success, or Result.Err on failure.

Examples:
  (fs/rename "/tmp/old.txt" "/tmp/new.txt")

#### `copy(src-path, dest-path)`

Copy a file from source to destination.

Returns Result.Ok on success, or Result.Err on failure.

Examples:
  (fs/copy "/tmp/source.txt" "/tmp/dest.txt")

#### `read-dir(path)`

List directory contents.

Returns Result.Ok with vector of entry names, or Result.Err on failure.

Examples:
  (fs/read-dir "/tmp")  ; => Result.Ok { value: ["file1.txt", "file2.txt"] }

#### `create-dir(path)`

Create a directory.

Parent directories must exist. Use create-dir-all for recursive creation.

Examples:
  (fs/create-dir "/tmp/mydir")

#### `create-dir-all(path)`

Create a directory and all parent directories.

Like mkdir -p. Creates any missing parent directories.

Examples:
  (fs/create-dir-all "/tmp/a/b/c")

#### `remove-dir(path)`

Remove an empty directory.

Fails if directory is not empty. Use remove-dir-all for recursive removal.

Examples:
  (fs/remove-dir "/tmp/empty-dir")

#### `remove-dir-all(path)`

Remove a directory and all its contents recursively.

Like rm -rf. Removes files and subdirectories.

Examples:
  (fs/remove-dir-all "/tmp/mydir")

#### `is-directory?(path)`

Check if path is a directory.

Returns Result.Ok { value: true/false }.

Examples:
  (fs/is-directory? "/tmp")  ; => Result.Ok { value: true }

#### `open(path, mode)`

Open a file with a mode string.

Modes: "r" (read), "w" (write/create), "a" (append), "r+" (read/write).
Returns a File handle for low-level operations.

Examples:
  (let file (unwrap (fs/open "/tmp/test.txt" "r")))

#### `close(file)`

Close a file handle.

Always close files when done to release resources.

Examples:
  (fs/close file)

#### `read(file, n)`

Read up to n bytes from a file handle.

Returns the data read as a string.

Examples:
  (fs/read file 1024)

#### `write(file, content)`

Write content to a file handle.

Returns the number of bytes written.

Examples:
  (fs/write file "Hello, World!")

#### `read-line(file)`

Read a line from a file handle.

Reads until newline or end of file.

Examples:
  (fs/read-line file)  ; => "First line\n"

#### `flush(file)`

Flush a file's buffers to disk.

Ensures all buffered writes are actually written.

Examples:
  (fs/flush file)

#### `blocking-read-file(path)`

Read the entire contents of a file synchronously.

This is the simplest way to read a file. No handler required.

Examples:
  match (fs/blocking-read-file "/tmp/test.txt") {
      Result.Ok { value } => println(value),
      Result.Err { error } => println("Error:", error)
  }

#### `blocking-write-file(path, content)`

Write content to a file synchronously (creates or overwrites).

This is the simplest way to write a file. No handler required.

Examples:
  (fs/blocking-write-file "/tmp/test.txt" "Hello, World!")

#### `blocking-delete-file(path)`

Delete a file synchronously.

Examples:
  (fs/blocking-delete-file "/tmp/test.txt")

#### `blocking-exists?(path)`

Check if a file or directory exists synchronously.

Examples:
  (fs/blocking-exists? "/tmp/test.txt")  ; => Result.Ok { value: true }

#### `blocking-read-dir(path)`

List directory contents synchronously.

Returns a Result containing a vector of filenames.

Examples:
  (fs/blocking-read-dir "/tmp")  ; => Result.Ok { value: ["file1.txt", "file2.txt"] }

#### `blocking-create-dir(path)`

Create a directory synchronously.

Examples:
  (fs/blocking-create-dir "/tmp/mydir")

<details>
<summary><strong>Undocumented functions (32)</strong></summary>

- `handle-read-file(path)`
- `handle-write-file(path, content)`
- `handle-append-file(path, content)`
- `handle-delete-file(path)`
- `handle-file-exists(path)`
- `handle-file-size(path)`
- `handle-is-file(path)`
- `handle-is-directory(path)`
- `handle-rename-file(old-path, new-path)`
- `handle-copy-file(src-path, dest-path)`
- `handle-read-dir(path)`
- `handle-create-dir(path)`
- `handle-create-dir-all(path)`
- `handle-remove-dir(path)`
- `handle-remove-dir-all(path)`
- `handle-open(path, mode)`
- `handle-close(file)`
- `handle-read(file, n)`
- `handle-write(file, content)`
- `handle-read-line(file)`
- `handle-flush(file)`
- `run-blocking(thunk)`
- `blocking-append-file(path, content)`
- `blocking-file-size(path)`
- `blocking-is-file?(path)`
- `blocking-is-directory?(path)`
- `blocking-rename(old-path, new-path)`
- `blocking-copy(src-path, dest-path)`
- `blocking-create-dir-all(path)`
- `blocking-remove-dir(path)`
- `blocking-remove-dir-all(path)`
- `BlockingFsHandler_handle(self, op, resume)`

</details>

---

## beagle.async

> **Documentation coverage:** 16/139 functions (11%)

#### `make-future(initial_state)`

Create a new future with an initial state.

Futures represent values that may not be available yet.

Examples:
  (let f (make-future (FutureState.Pending {})))

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

#### `with-scope(body)`

Run a block with a task scope for structured concurrency.

Ensures all spawned tasks complete before the scope exits.
On error or cancellation, all child tasks are cancelled.

Examples:
  (with-scope fn(scope) {
      let f1 = spawn(scope, fn() { compute1() })
      let f2 = spawn(scope, fn() { compute2() })
      [await(f1), await(f2)]
  })

#### `sleep(ms)`

Sleep for a number of milliseconds.

Must be called within an Async effect handler.

Examples:
  (sleep 1000)  ; Sleep for 1 second

#### `await(future)`

Wait for a future to complete and return its value.

Blocks until the future resolves, rejects, or is cancelled.

Examples:
  (let f (spawn fn() { compute() }))
  (await f)  ; => result of compute()

#### `await-all(futures)`

Wait for all futures to complete and return their values as a vector.

Examples:
  (let fs [(spawn fn() { 1 }) (spawn fn() { 2 })])
  (await-all fs)  ; => [1 2]

#### `await-first(futures)`

Wait for the first future to complete and return its result.

Returns a RaceResult indicating which future completed first.

#### `await-timeout(ms, future)`

Wait for a future with a timeout in milliseconds.

Returns TimeoutResult.Ok if completed, TimeoutResult.TimedOut if timeout.

#### `spawn(thunk)`

Spawn an async task and return a future.

The task runs concurrently. Use await to get the result.

Examples:
  (let f (spawn fn() { expensive-computation() }))
  (await f)  ; => result of computation

<details>
<summary><strong>Undocumented functions (123)</strong></summary>

- `make-scope()`
- `scope-add-child!(scope, future)`
- `scope-children(scope)`
- `scope-cancel-all!(scope)`
- `scope-await-all-children(scope)`
- `async-ok(value)`
- `async-err(code, message)`
- `async-ok?(result)`
- `async-unwrap(result)`
- `async-unwrap-or(result, default)`
- `handle-read-file(path)`
- `handle-write-file(path, content)`
- `handle-append-file(path, content)`
- `handle-delete-file(path)`
- `handle-file-exists(path)`
- `handle-file-size(path)`
- `handle-is-file(path)`
- `handle-is-directory(path)`
- `handle-rename-file(old-path, new-path)`
- `handle-copy-file(src-path, dest-path)`
- `handle-read-dir(path)`
- `handle-create-dir(path)`
- `handle-create-dir-all(path)`
- `handle-remove-dir(path)`
- `handle-remove-dir-all(path)`
- `handle-open(path, mode)`
- `handle-close(file)`
- `handle-read(file, n)`
- `handle-write(file, content)`
- `handle-read-line(file)`
- `handle-flush(file)`
- `handle-sleep(ms)`
- `handle-await-blocking(future)`
- `handle-await-all-blocking(futures)`
- `handle-await-first-blocking(futures)`
- `handle-cancel(future)`
- `handle-spawn-blocking(thunk)`
- `handle-spawn-with-token-blocking(thunk, token)`
- `handle-spawn-threaded(thunk)`
- `handle-spawn-with-token-threaded(thunk, token)`
- `handle-await-threaded(future)`
- `poll-until-resolved(future)`
- `handle-await-all-threaded(futures)`
- `handle-await-first-threaded(futures)`
- `create-event-loop-handler(pool_size)`
- `create-event-loop-handler-default()`
- `handle-sleep-event-loop(loop_id, ms)`
- `handle-spawn-event-loop(loop_id, thunk)`
- `handle-await-event-loop(loop_id, future)`
- `process-tcp-results(loop_id)`
- `poll-file-result(loop_id, handle)`
- `wait-for-file-result(loop_id, handle)`
- `handle-read-file-async(loop_id, path)`
- `handle-write-file-async(loop_id, path, content)`
- `handle-delete-file-async(loop_id, path)`
- `handle-file-size-async(loop_id, path)`
- `handle-read-dir-async(loop_id, path)`
- `process-timer-results(loop_id)`
- `handle-await-all-event-loop(loop_id, futures)`
- `handle-await-first-event-loop(loop_id, futures)`
- `read-file(path)`
- `write-file(path, content)`
- `append-file(path, content)`
- `delete-file(path)`
- `file-exists?(path)`
- `read-dir(path)`
- `create-dir(path)`
- `create-dir-all(path)`
- `remove-dir(path)`
- `remove-dir-all(path)`
- `file-size(path)`
- `is-directory?(path)`
- `is-file?(path)`
- `rename-file(old-path, new-path)`
- `copy-file(src-path, dest-path)`
- `open(path, mode)`
- `close(file)`
- `read(file, n)`
- `write(file, content)`
- `read-line(file)`
- `flush(file)`
- `async(thunk)`
- `async-with-token(token, thunk)`
- `cancel(future)`
- `spawn-in-scope(scope, thunk)`
- `with-timeout(ms, thunk)`
- `async-with-timeout(ms, thunk)`
- `async-sleep(ms)`
- `await-sleep(ms)`
- `time-now()`
- `read-file!(path)`
- `write-file!(path, content)`
- `ok?(result)`
- `unwrap-or(result, default)`
- `default-handler()`
- `run-blocking(thunk)`
- `blocking-read-file(path)`
- `blocking-write-file(path, content)`
- `blocking-append-file(path, content)`
- `blocking-delete-file(path)`
- `blocking-file-exists?(path)`
- `blocking-read-dir(path)`
- `blocking-create-dir(path)`
- `blocking-remove-dir(path)`
- `blocking-file-size(path)`
- `blocking-is-directory?(path)`
- `blocking-is-file?(path)`
- `blocking-rename-file(old-path, new-path)`
- `blocking-copy-file(src-path, dest-path)`
- `blocking-sleep(ms)`
- `run-all-thunks(thunks)`
- `await-any(thunks)`
- `map-results(results, f)`
- `filter-ok(results)`
- `unwrap-all(results)`
- `get-io-loop()`
- `handle-io-action(action)`
- `create-implicit-handler()`
- `with-implicit-async(thunk)`
- `BlockingAsyncHandler_handle(self, op, resume)`
- `ThreadedAsyncHandler_handle(self, op, resume)`
- `EventLoopHandler_handle(self, op, resume)`
- `ImplicitAsyncHandler_handle(self, op, resume)`

</details>

---

## beagle.io

> **Documentation coverage:** 16/20 functions (80%)

#### `open(path, mode)`

Open a file with the specified mode.

Modes: "r" (read), "w" (write), "a" (append), "r+" (read/write).
Returns Result with a File handle or error.

Examples:
  (match (io/open "/tmp/test.txt" "r") {
      Result.Ok { value } => value,
      Result.Err { error } => println("Failed to open")
  })

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
<summary><strong>Undocumented functions (4)</strong></summary>

- `get-libc-path()`
- `io-err(msg)`
- `is-null(pointer)`
- `string-to-buffer(s)`

</details>

---

## beagle.timer

> **Documentation coverage:** 8/10 functions (80%)

#### `sleep(ms)`

Sleep for a number of milliseconds.

Must be called within a Timer effect handler block.

Examples:
  (timer/sleep 100)  ; Sleep for 100ms

#### `now()`

Get current time in nanoseconds since epoch.

Examples:
  (let start (timer/now))
  ; ... do work ...
  (let elapsed (- (timer/now) start))

#### `timeout(ms, future)`

Wrap a future with a timeout.

Returns Result.Ok if completed before timeout, or Result.Err with Timeout error.
Requires both Timer and Async handlers to be installed.

Examples:
  (match (timer/timeout 1000 my-future) {
      Result.Ok { value } => value,
      Result.Err { error } => "timed out"
  })

#### `deadline(ms)`

Create a deadline (absolute time in nanoseconds from now).

Useful for setting a single deadline across multiple operations.

Examples:
  (let d (timer/deadline 5000))  ; 5 second deadline

#### `deadline-passed?(deadline-time)`

Check if a deadline has passed.

#### `deadline-remaining(deadline-time)`

Get remaining milliseconds until a deadline (0 if passed).

#### `blocking-sleep(ms)`

Sleep for a number of milliseconds synchronously.

No handler required. Blocks the current thread.

Examples:
  (timer/blocking-sleep 100)  ; Sleep for 100ms

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

> **Documentation coverage:** 17/17 functions (100%)

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

---

## beagle.collections

> **Documentation coverage:** 15/15 functions (100%)

#### `vec()` `builtin`

Create a new empty persistent vector.

Persistent vectors are immutable - all operations return new vectors.

Examples:
  (let v (collections/vec))

#### `vec-count(vec)` `builtin`

Return the number of elements in the vector.

Examples:
  (vec-count (push [] 1 2 3))  ; => 3

#### `vec-get(vec, index)` `builtin`

Get the element at index. Returns null if out of bounds.

Examples:
  (vec-get [1 2 3] 1)  ; => 2

#### `vec-push(vec, value)` `builtin`

Return a new vector with value appended.

Examples:
  (vec-push [1 2] 3)  ; => [1 2 3]

#### `vec-assoc(vec, index, value)` `builtin`

Return a new vector with the value at index replaced.

Examples:
  (vec-assoc [1 2 3] 1 99)  ; => [1 99 3]

#### `map()` `builtin`

Create a new empty persistent map.

Persistent maps are immutable - all operations return new maps.

Examples:
  (let m (collections/map))

#### `map-count(m)` `builtin`

Return the number of key-value pairs in the map.

Examples:
  (map-count {:a 1 :b 2})  ; => 2

#### `map-get(m, key)` `builtin`

Get the value for key. Returns null if not found.

Examples:
  (map-get {:a 1} :a)  ; => 1
  (map-get {:a 1} :b)  ; => null

#### `map-assoc(m, key, value)` `builtin`

Return a new map with the key-value pair added or updated.

Examples:
  (map-assoc {:a 1} :b 2)  ; => {:a 1 :b 2}

#### `map-keys(m)` `builtin`

Return a vector of all keys in the map.

Examples:
  (map-keys {:a 1 :b 2})  ; => [:a :b]

#### `set()` `builtin`

Create a new empty persistent set.

Persistent sets are immutable - all operations return new sets.

Examples:
  (let s (collections/set))

#### `set-count(s)` `builtin`

Return the number of elements in the set.

Examples:
  (set-count #{1 2 3})  ; => 3

#### `set-contains?(s, element)` `builtin`

Return true if the set contains the element.

Examples:
  (set-contains? #{1 2 3} 2)  ; => true

#### `set-add(s, element)` `builtin`

Return a new set with the element added.

Examples:
  (set-add #{1 2} 3)  ; => #{1 2 3}

#### `set-elements(s)` `builtin`

Return a vector of all elements in the set.

Examples:
  (set-elements #{1 2 3})  ; => [1 2 3]

---

## beagle.effect

> **Documentation coverage:** 0/1 functions (0%)

<details>
<summary><strong>Undocumented functions (1)</strong></summary>

- `handle(self, op, resume)`

</details>

---

## beagle.ffi

> **Documentation coverage:** 19/21 functions (90%)

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

#### `get-string(buffer, offset, length)` `builtin`

Read a string from a buffer at the given offset with the specified length.

Examples:
  (ffi/get-string buf 0 10)  ; Read 10 bytes starting at offset 0

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

<details>
<summary><strong>Undocumented functions (2)</strong></summary>

- `call-ffi-info()`
- `copy-bytes_filter()`

</details>

---

## beagle.simple-socket

> **Documentation coverage:** 0/8 functions (0%)

<details>
<summary><strong>Undocumented functions (8)</strong></summary>

- `create-continuation-handler(loop_id)`
- `drive-continuation-loop(handler)`
- `listen(host, port)`
- `accept(listener)`
- `read(client, n)`
- `write(client, data)`
- `close(client)`
- `ContinuationHandler_handle(self, op, resume)`

</details>

---

## beagle.socket

> **Documentation coverage:** 0/7 functions (0%)

<details>
<summary><strong>Undocumented functions (7)</strong></summary>

- `listen(host, port)`
- `connect(host, port)`
- `accept(listener)`
- `read(socket, n)`
- `write(socket, data)`
- `close(socket)`
- `close-listener(listener)`

</details>

---

## global

> **Documentation coverage:** 0/42 functions (0%)

<details>
<summary><strong>Undocumented functions (42)</strong></summary>

- `trampoline()`
- `<Anonymous>(acc, x)`
- `<Anonymous>(a, x)`
- `<Anonymous>(acc, elem)`
- `<Anonymous>(acc, pair)`
- `<Anonymous>(acc, elem)`
- `<Anonymous>(acc, x)`
- `<Anonymous>(ignored)`
- `<Anonymous>(x)`
- `<Anonymous>(x)`
- `<Anonymous>(b)`
- `<Anonymous>(k)`
- `<Anonymous>(k)`
- `<Anonymous>(acc, k)`
- `<Anonymous>(acc, k)`
- `<Anonymous>(k)`
- `<Anonymous>(acc, k)`
- `<Anonymous>(acc, elem)`
- `<Anonymous>(a1, a2, a3, a4, a5, a6)`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>(r)`
- `<Anonymous>(r)`
- `<Anonymous>(r)`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`
- `<Anonymous>()`

</details>

---

## Types

These are the built-in struct types available in Beagle.

#### beagle.core/NEVER

#### beagle.core/Struct

**Fields:**

- `name`
- `id`

#### beagle.core/Protocol

**Fields:**

- `name`

#### beagle.core/Thread

**Fields:**

- `closure`

#### beagle.core/Atom

**Fields:**

- `value`

#### beagle.core/PersistentVectorSeq

**Fields:**

- `vec`
- `index`

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

**Fields:**

- `current`
- `end`
- `step`

#### beagle.core/StringSeq

**Fields:**

- `str`
- `index`

#### beagle.core/PersistentSetSeq

**Fields:**

- `elements`
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

**Fields:**

- `id`

#### beagle.ffi/Pointer

**Fields:**

- `ptr`

#### beagle.ffi/Buffer

**Fields:**

- `ptr`
- `size`

#### beagle.ffi/StructReturn

**Fields:**

- `low`
- `high`

#### beagle.ffi/Type

**Fields:**

- `U8`
- `U16`
- `U32`
- `U64`
- `I32`
- `F32`
- `Pointer`
- `MutablePointer`
- `String`
- `Void`
- `Structure`

#### beagle.ffi/Type.U8

#### beagle.ffi/Type.U16

#### beagle.ffi/Type.U32

#### beagle.ffi/Type.U64

#### beagle.ffi/Type.I32

#### beagle.ffi/Type.F32

#### beagle.ffi/Type.Pointer

#### beagle.ffi/Type.MutablePointer

#### beagle.ffi/Type.String

#### beagle.ffi/Type.Void

#### beagle.ffi/Type.Structure

**Fields:**

- `types`

#### beagle.io/File

**Fields:**

- `handle`

#### beagle.io/BufferResult

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

**Fields:**

- `state_atom`

#### beagle.async/CancellationToken

**Fields:**

- `cancelled_atom`

#### beagle.async/TaskScope

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

**Fields:**

- `loop_id`
- `listener_id`

#### beagle.async/TcpSocket

**Fields:**

- `loop_id`
- `socket_id`

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

#### beagle.async/ThreadedAsyncHandler

#### beagle.async/EventLoopHandler

**Fields:**

- `event_loop_id`

#### beagle.async/ImplicitAsyncHandler

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

#### beagle.timer/Timer

**Fields:**

- `Sleep`
- `Now`

#### beagle.timer/Timer.Sleep

**Fields:**

- `ms`

#### beagle.timer/Timer.Now

#### beagle.timer/BlockingTimerHandler

#### beagle.simple-socket/ContinuationHandler

**Fields:**

- `loop_id`
- `accept_continuations`
- `pending_ops`

---

## Enums

These are the built-in enum types available in Beagle.

#### beagle.core/SystemError

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

#### beagle.core/DiagnosticSeverity

**Variants:**

- `error`
- `warning`
- `info`
- `hint`

#### beagle.core/Error

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

**Variants:**

- `Ok { value }`
- `Err { error }`

#### beagle.ffi/Type

**Variants:**

- `U8`
- `U16`
- `U32`
- `U64`
- `I32`
- `F32`
- `Pointer`
- `MutablePointer`
- `String`
- `Void`
- `Structure { types }`

#### beagle.async/FutureState

**Variants:**

- `Pending`
- `Running`
- `Resolved { value }`
- `Rejected { error }`
- `Cancelled`

#### beagle.async/RaceResult

**Variants:**

- `Ok { value, index }`
- `AllFailed { errors }`

#### beagle.async/TimeoutResult

**Variants:**

- `Ok { value }`
- `TimedOut`

#### beagle.async/IOAction

**Variants:**

- `TcpListen { host, port }`
- `TcpConnect { host, port }`
- `TcpAccept { listener }`
- `TcpRead { socket, n }`
- `TcpWrite { socket, data }`
- `TcpClose { socket }`
- `TcpCloseListener { listener }`

#### beagle.async/Async

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

