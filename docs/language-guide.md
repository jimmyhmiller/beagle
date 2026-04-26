# Beagle Language Guide

This is a comprehensive reference for the Beagle programming language. It's intended as a hand-off document — hand it to an AI or a new contributor and they should be able to write correct Beagle code from it.

Beagle is a dynamically-typed functional language. It is inspired by Clojure but uses a braces-and-commas surface syntax. It has structs, enums, protocols, algebraic effects, continuations, closures, atoms, persistent collections, threads, and FFI.

All examples in this guide are taken from real working Beagle code. If something in this guide conflicts with actual runtime behavior, treat the runtime as the source of truth.

---

## 1. Program Structure

Every file starts with a `namespace` declaration.

```beagle
namespace example

fn main() {
    println("hello")
}
```

There is no separate module/header. Files are single-file modules.

### 1.1 What's allowed at top level

Top-level (module-level) forms:

- `use` imports
- `let` bindings (immutable only — `let mut` is **NOT** allowed at top level)
- `struct` / `enum` / `protocol` / `extend` definitions
- `fn` definitions
- `let dynamic` declarations
- `test "name" { ... }` blocks (see §27)
- Top-level expressions (including function calls with side-effects like `println`)

Top-level code runs once in order when the namespace is loaded:

```beagle
namespace top_level

let x = 2

println(x)          // runs at load time

fn main() {
    println("Hello, world!")
    println(x)
    "done"
}
```

### 1.2 `main`

Programs usually define `fn main()` as the entry point. `main` runs after top-level code.

### 1.3 Comments

- `//` — line comment
- `///` — docstring (attaches to the following `fn` or `struct`; retrievable via `reflect/doc`)

```beagle
/// Adds two numbers.
/// Can span multiple lines.
fn add(x, y) { x + y }
```

---

## 2. Values and Literals

### 2.1 Integers

```beagle
42
-5
0xFF        // hex        = 255
0o755       // octal      = 493
0b1010      // binary     = 10
0XFF 0O755 0B1010  // uppercase prefixes also work
```

### 2.2 Floats

```beagle
3.14159
1.0 + 2.5
```

### 2.3 Strings

Strings use double quotes. Escape sequences: `\t`, `\n`, `\r`, `\0`, `\\`, `\"`, `\$`.

```beagle
"hello"
"tab: [\t]"
"backslash: [\\]"
"quote: [\"]"
```

### 2.4 String interpolation — `${...}`

**This is the single most-missed feature by new users / AIs.** String interpolation uses `${expression}` inside a normal double-quoted string. There is no separate "template string" syntax, no backticks, no `#{}`. To embed a literal dollar sign, escape it as `\$`.

```beagle
let name = "World"
let greeting = "Hello, ${name}!"
let sum      = "Sum: ${a + b}"
let nested   = "Double of ${x} is ${double(x)}"
let cond     = "x is ${if x > 5 { "big" } else { "small" }}"
let literal  = "Price: \$100"     // prints: Price: $100
```

Arbitrary expressions are allowed inside `${...}` — including function calls, arithmetic, `if`, `match`, struct field access, etc.

### 2.5 Booleans and null

```beagle
true
false
null
```

### 2.6 Arrays (persistent vectors)

Array literals use `[...]` and produce a persistent vector.

```beagle
let a = [1, 2, 3]
let empty = []
let mixed = [1, "two", p]
a[0]                  // indexing with []
get(a, 0)             // indexing via function
length(a)             // size
push(a, 4)            // returns NEW array; a is unchanged
```

### 2.7 Maps

Map literals use `{...}` with **space (or newline) separating key and value**, and optional commas. Keyword keys are most idiomatic, but strings and numbers also work; computed keys use parens.

```beagle
let empty   = {}
let simple  = {:name "Alice" :age 30}
let commas  = {:name "Alice", :age 30}          // commas allowed
let mixed   = {"string-key" 1 123 "numeric-key"}
let nested  = {:outer {:inner 42}}
let computed = {(1 + 2) (10 * 10)}              // key 3 -> value 100

get(simple, :name)     // "Alice"
```

### 2.8 Sets

Set literals use `#{...}`.

```beagle
let s = #{1, 2, 3}
set-contains?(s, 2)
set-add(s, 4)
set-union(#{1,2}, #{2,3})
```

### 2.9 Keywords

Keywords are interned symbolic values prefixed with `:`. Equality is identity-cheap.

```beagle
:hello
:age
keyword->string(:foo)     // "foo"
string->keyword("foo")    // :foo
```

### 2.10 Atoms

Atoms are mutable reference cells (the only sanctioned way to mutate shared state).

```beagle
let a = atom(42)
deref(a)                              // 42 — read current value
reset!(a, 100)                        // overwrite
swap!(a, fn(x) { x + 1 })             // atomically update via function
compare-and-swap!(a, old, new)        // CAS, returns true/false
```

---

## 3. Variables and Mutability

### 3.1 Immutable `let`

```beagle
let x = 10
let [a, b] = [1, 2]         // destructuring
let Point { x, y } = p
```

### 3.2 `let mut` — local mutable bindings

`let mut` creates a mutable binding. **This is only legal inside a function body — never at top level.** Reassignment uses plain `=`.

```beagle
fn main() {
    let mut x = 0
    let mut y = 0
    x = 1
    y = 2

    let mut counter = 0
    let inc = fn() {
        counter = counter + 1     // closures can mutate captured mut bindings
    }
}
```

At top level, `let` is always immutable. If you need shared mutable state at top level, use an `atom`:

```beagle
let counter = atom(0)   // top-level-safe mutable cell
```

### 3.3 `let dynamic` — dynamically-scoped variables

Dynamic variables are declared at top level. Their value can be rebound within a lexical region using `binding (...) { ... }`; the rebinding is undone on exit.

```beagle
let dynamic dyn_x = 42

fn main() {
    println(dyn_x)                   // 42
    binding (dyn_x = 100) {
        println(dyn_x)               // 100
    }
    println(dyn_x)                   // 42 (restored)
}
```

---

## 4. Operators

### 4.1 Arithmetic

```beagle
1 + 2
5 - 3
2 * 3
10 / 2
5 % 2
```

### 4.2 Comparison

```beagle
a == b    a != b
a < b     a > b
a <= b    a >= b
equal(a, b)      // deep/structural equality
```

### 4.3 Bitwise

```beagle
5 & 3       // AND
5 | 3       // OR
5 ^ 3       // XOR
16 << 2     // left shift
16 >> 2     // arithmetic right shift (sign-extending)
16 >>> 2    // logical right shift (zero-fill)
```

### 4.4 String concatenation

```beagle
"hello" ++ " " ++ "world"
```

### 4.5 Pipe operators

Beagle has **two** pipe operators. Do not confuse them.

- `|>` — **pipe-first**: `x |> f(a, b)` ≡ `f(x, a, b)` (x becomes the *first* arg)
- `|>>` — **pipe-last**:  `x |>> f(a, b)` ≡ `f(a, b, x)` (x becomes the *last* arg)

```beagle
5 |> add_one                       // add_one(5)
10 |> add(5)                       // add(10, 5) = 15
3  |>> subtract(10)                // subtract(10, 3) = 7
20 |>> subtract(5)                 // subtract(5, 20) = -15

[1,2,3,4,5]
    |> map(fn(x) { x * 2 })
    |> filter(fn(x) { x > 4 })
    |> reduce(0, fn(acc, x) { acc + x })
```

Most stdlib collection functions put the collection first, so `|>` chains naturally.

---

## 5. Control Flow

### 5.1 `if` / `else` (expression)

```beagle
let label = if x > 5 { "big" } else { "small" }
```

`if` with no `else` yields `null` on the false branch.

### 5.2 `while` (expression)

`while` returns a value — the last body expression evaluated before the condition went false.

```beagle
let mut i = 0
let result = while i < 5 {
    i = i + 1
    i * 10
}
// result is 50
```

### 5.3 `for ... in ...` (expression)

Iterates over ranges, arrays, strings, sets, etc.

```beagle
for i in range(0, 5) { println(i) }       // 0..4
for c in "abc"       { println(c) }
for x in [1, 2, 3]   { println(x) }
```

`for` is also an expression and supports `break(value)`:

```beagle
let result = for i in range(0, 10) {
    if i == 5 { break(i * 100) }
    i
}
// result is 500
```

### 5.4 `loop { ... }` + `break(...)`

`loop` is an unbounded loop. Exit with `break()` or `break(value)`.

```beagle
let x = loop {
    break(42)
}
```

### 5.5 `break` / `continue`

Both are **function-call syntax**: `break()`, `break(value)`, `continue()`. Not bare keywords.

### 5.6 `match`

`match` is an expression. Arms are separated by commas; bodies can be blocks or expressions; use `_` for wildcard.

```beagle
match value {
    0  => "zero",
    1  => "one",
    42 => "forty-two",
    _  => "other",
}
```

Matchable patterns:

- Literals: numbers, strings, booleans, negatives
- Identifier — binds the whole value (no type check)
- Wildcard `_`
- Array patterns — fixed arity (`[]`, `[x]`, `[x, y]`, `[1, b, c]`, `[_, b, _]`, `[[x, y], [z, w]]`)
- Struct patterns — `Point { x, y }` (destructures fields)
- Enum patterns — `MyOption.some { value: v }`, `MyOption.none`

```beagle
match arr {
    []              => "empty",
    [x]             => "single",
    [x, y]          => "pair",
    [1, b, c]       => b + c,               // literal first element
    [a, 0, c]       => a + c,               // literal middle
    [_, b, _]       => b,                   // wildcards
    [[x, y], [z, w]] => x + y + z + w,     // nested
    _               => "other",
}

match p {
    Point { x, y } => x + y,
}

match opt {
    MyOption.some { value: v } => v,
    MyOption.none              => 0,
}
```

---

## 6. Functions

### 6.1 Named functions

```beagle
fn add(x, y) {
    x + y
}
```

The last expression of the body is the return value. There is no early-`return` statement in the imperative sense — control flow out of a function is always by falling off the last expression or by `throw`.

### 6.2 Anonymous functions / closures

```beagle
let f = fn(x) { x + 1 }
f(5)

let x = 42
let g = fn() { x }        // closes over x
```

### 6.3 Multi-arity functions

Define multiple dispatch arms by arity.

```beagle
fn greet {
    () => greet("World")
    (name) => "Hello, " ++ name
}

fn sum {
    () => 0
    (x) => x
    (x, y) => x + y
}
```

### 6.4 Variadic functions (rest args)

Use `...name` for rest args. Rest is delivered as an array.

```beagle
fn sum(...nums) {
    let mut total = 0
    let mut i = 0
    while i < length(nums) {
        total = total + get(nums, i)
        i = i + 1
    }
    total
}
sum(1, 2, 3)                // 6

fn sum_with_base(base, ...rest) { /* base + everything in rest */ }
```

### 6.5 Destructuring in parameters

```beagle
fn first_two([a, b, ...rest]) { a + b }
fn mag(Point { x, y })        { sqrt(x*x + y*y) }
```

### 6.6 Functions as values

Functions (including variadic and multi-arity) can be passed and stored as first-class values.

```beagle
let p = println
p("hi", 1, 2)

fn apply(f, a, b, c) { f(a, b, c) }
apply(println, 1, 2, 3)
```

---

## 7. Destructuring

Works uniformly in `let`, function parameters, and `match` arms.

### 7.1 Arrays

```beagle
let [a, b, c]       = [1, 2, 3]
let [first, _, last] = [1, 2, 3]
let [[x, y], [z, w]] = [[1, 2], [3, 4]]
let [head, ...tail]  = [1, 2, 3, 4]      // rest pattern
```

### 7.2 Maps

```beagle
let m = {:name "Alice", :age 30}
let { name, age }        = m         // shorthand — binds name and age from :name / :age
let { name: n, age: a }  = m         // rename
```

### 7.3 Structs

```beagle
let p = Point { x: 1, y: 2 }
let Point { x, y }             = p
let Point { x, y: my_y }       = p
```

---

## 8. Structs

### 8.1 Definition

```beagle
struct Point { x, y }

struct Config {
    name
    debug: false
    verbose: false
    retries: 3
}

struct WithExprDefaults {
    a: 1 + 2
    b: 10 * 5
}
```

Fields can optionally have default-value expressions. Defaults are evaluated per-instance.

### 8.2 Construction

```beagle
let p  = Point { x: 10, y: 20 }
let p2 = Point { x, y }            // shorthand: field name matches local var
let p3 = Point { x: 5, y }         // mix shorthand and explicit
let c  = Config { name: "prod" }   // unspecified fields use defaults
let pZ = Point {}                  // all defaults (fails if any required field missing)
```

### 8.3 Field access

```beagle
p.x
p.y
```

### 8.4 Immutable update — struct spread `...`

This is how you "update" a struct: create a new one from an existing instance with selected fields overridden. This is the other feature AIs commonly miss.

```beagle
let p  = Point { x: 10, y: 20 }
let p2 = Point { ...p, x: 15 }     // p2.x = 15, p2.y = 20 (inherited)
let p3 = Point { ...p }            // full copy
```

The spread must be written as `TypeName { ...instance, field: value, ... }`. The type name is required.

---

## 9. Enums

### 9.1 Definition

Enum variants may be unit-like or may carry fields.

```beagle
enum Action {
    run { speed },
    walk,
    stop,
}

enum MyOption {
    some { value },
    none,
}
```

### 9.2 Construction

```beagle
let a = Action.run { speed: 5 }
let w = Action.walk
let s = Action.stop

let opt = MyOption.some { value: 42 }
let none = MyOption.none
```

### 9.3 Field access

```beagle
a.speed     // 5
```

### 9.4 Matching

```beagle
match a {
    Action.run { speed } => if speed > 3 { "fast" } else { "slow" },
    Action.walk          => "walking",
    Action.stop          => "stopped",
}

match opt {
    MyOption.some { value: v } => v,
    MyOption.none              => 0,
}
```

### 9.5 The built-in `Result` enum

The standard library defines a `Result` enum used pervasively in I/O, FFI, async:

```beagle
enum Result {
    Ok  { value },
    Err { error },
}

match result {
    Result.Ok  { value } => use_value(value),
    Result.Err { error } => handle(error),
}

// Helpers
ok(42)                     // Result.Ok { value: 42 }
err("bad")                 // Result.Err { error: "bad" }
ok?(r)                     // true/false
unwrap(r)                  // value, or throws
unwrap-or(r, default)
```

---

## 10. Protocols

Protocols are like typeclasses / traits — a named set of methods that types can implement. Dispatch is based on the first argument's type.

### 10.1 Definition

```beagle
protocol Axis {
    fn get-x(self)
    fn get-y(self)
}

protocol Format {
    fn format(self) {           // default implementation
        "no format"
    }
}
```

### 10.2 Implementation

```beagle
extend Point with Axis {
    fn get-x(self) { self.x }
    fn get-y(self) { self.y }
}

extend String with Format {
    fn format(self) { self }
}
```

After extension, call protocol methods as ordinary functions:

```beagle
get-x(point)
format("hi")
format(2)          // falls back to default: "no format"
```

Protocols can be parameterized (used by the effect system — see §13):

```beagle
extend ConsoleLogger with effect/Handler(Log) { ... }
```

---

## 11. Namespaces and Imports

### 11.1 Namespace declaration

Every file begins with `namespace name`. Names can use dots and dashes:

```beagle
namespace fib
namespace my.sub.module
namespace my-kebab-namespace
```

### 11.2 Importing other namespaces — `use ... as ...`

```beagle
use fib as f
use beagle.fs as fs
use beagle.async as async
use beagle.effect as effect
use beagle.ffi as ffi
use beagle.core as core

fn main() {
    f/fib(10)                  // call f's fib
    fs/read-file("/tmp/x")
}
```

Qualified reference is `alias/name`. The alias is required — there's no "flat" import.

### 11.3 Standard-library namespaces

All shipped under `standard-library/`:

| Namespace | Purpose |
|---|---|
| `beagle.core` | Built-in type descriptors (`PersistentVector`, `Array`, `Int`, etc.), core helpers, `eval` |
| `beagle.fs` | File system: `read-file`, `write-file`, `read-dir`, etc. (synchronous + handler-based) |
| `beagle.io` | Low-level file handles: `open`, `read`, `write`, `close`, stdin/stdout/stderr |
| `beagle.async` | Async file I/O, futures, spawning, `await`, `sleep`, scopes |
| `beagle.effect` | Effect handler infrastructure (`Handler(E)` protocol) |
| `beagle.ffi` | Foreign function interface |
| `beagle.spawn` | Thread spawning / spawn handlers |
| `beagle.socket` | TCP sockets |
| `beagle.timer` | Timers, sleep, `now`, deadlines, timeouts |
| `beagle.stream` | Lazy streams |
| `beagle.mutable-array` | Mutable array ops (when you need in-place mutation) |
| `beagle.primitive` | Low-level primitives (rarely needed) |
| `beagle.reflect` | `doc`, `apropos`, `namespace-members`, `all-namespaces` |

Things in `std.bg` (the implicit core) are available without `use`: `println`, `print`, `format`, `atom`, `deref`, `reset!`, `swap!`, `compare-and-swap!`, `map`, `filter`, `reduce`, `range`, most collection and math utilities, `ok`, `err`, `ok?`, `unwrap`.

---

## 12. Errors: `throw` / `try` / `catch`

### 12.1 Throwing

```beagle
throw("my error")
throw(Error.IO { message: "file gone" })
```

You can throw any value.

### 12.2 Catching

```beagle
try {
    println("In try block")
    throw("my error")
    println("never printed")
} catch (e) {
    println("caught:", e)
}
```

### 12.3 Resumable exceptions — `catch (e, resume)`

A `catch` clause can bind a **second parameter**, a `resume` function. Calling `resume(value)` re-enters the computation at the `throw` site, and the throw expression evaluates to `value`. Not calling `resume` unwinds as usual.

```beagle
// Resume with a value: the throw acts as if it returned 42
let result = try {
    let x = throw("need value") + 1
    x
} catch (e, resume) {
    resume(42)
}
// result is 43  (42 + 1)

// Don't call resume: behaves like a normal catch
let r2 = try {
    throw("error")
    println("never printed")
} catch (e, resume) {
    println("caught without resume")
    99
}
// r2 is 99

// Single-arg catch (no resume) still works
let r3 = try {
    throw("old style")
} catch (e) {
    77
}
```

**Runtime errors are also resumable.** Type errors, divide-by-zero, bad field access, compile errors from `eval`, etc. all go through the same mechanism:

```beagle
let r = try {
    2 + "hello"              // TypeError thrown by the runtime
} catch (e, resume) {
    resume(42)               // supply a "replacement" result
}
// r is 42

let r2 = try {
    eval("this is invalid syntax !!!")
} catch (e, resume) {
    resume(100)
}
// r2 is 100
```

This means every runtime error can, in principle, be patched and continued from — useful for REPLs, long-running services, and debugger-style tools.

### 12.4 Result-based error handling

For most I/O, prefer `Result`-returning APIs (see §9.5). Pattern-match on `Result.Ok { value }` / `Result.Err { error }` rather than using `try`.

---

## 13. Algebraic Effects and Handlers

Beagle has a first-class effect system built on top of protocols. Effects are used for async I/O, dependency injection (e.g. writer/reader patterns), non-determinism, generators, etc.

### 13.1 Declaring an effect

An effect is just an enum of operations:

```beagle
enum Log {
    Info { message }
}
```

### 13.2 Implementing a handler

Implement `effect/Handler(EffectName)` on a struct. The `handle` method receives the operation and a `resume` continuation.

```beagle
use beagle.effect as effect

struct ConsoleLogger {}

extend ConsoleLogger with effect/Handler(Log) {
    fn handle(self, op, resume) {
        println("in handler")
        resume(null)             // resume the computation with a value
    }
}
```

### 13.3 Using a handler — `handle ... with ... { ... }`

```beagle
let logger = ConsoleLogger {}
let result = handle effect/Handler(Log) with logger {
    perform Log.Info { message: "Hello!" }
    println("after perform")
    42
}
println(result)        // 42
```

### 13.4 Multi-shot continuations

Handlers may call `resume` more than once, or not at all.

```beagle
extend Chooser with effect/Handler(Choice) {
    fn handle(self, op, resume) {
        let a = resume(1)      // run the body as if Choose returned 1
        let b = resume(2)      // run the body again as if it returned 2
        a + b
    }
}
```

### 13.5 Low-level continuations — `reset` / `shift`

Delimited continuations are available directly:

```beagle
let result = reset {
    1 + shift(fn(k) {
        42                    // ignore k, just return 42 out of the reset
    })
}
// result is 42
```

`k` is a first-class function you can call to resume the captured continuation.

---

## 14. Async, Futures, Scopes

Most async operations live in `beagle.async` and require an implicit handler to be installed. When running with Beagle's default entry point, the handler is already active, so top-level async calls Just Work.

### 14.1 File I/O

`beagle.async` (preferred for most code — handler-based, cancellable):

```beagle
use beagle.async as async

let write_result = async/write-file("/tmp/x", "hello")
match write_result {
    Result.Ok  { value } => println("wrote", value, "bytes"),
    Result.Err { error } => println("error:", error),
}

let read = async/read-file("/tmp/x")   // Result.Ok { value: "hello" }
async/file-size("/tmp/x")
async/file-exists?("/tmp/x")
async/is-file?("/tmp/x")
async/is-directory?("/tmp/x")
async/delete-file("/tmp/x")
async/rename-file(old, new)
async/copy-file(src, dst)
async/append-file(path, more)
async/read-dir("/tmp")
async/create-dir(path)
async/create-dir-all(path)
async/remove-dir(path)
async/remove-dir-all(path)
```

`beagle.fs` provides the same shape of API in a direct (non-async, non-cancellable) form — use it when you don't need the async handler machinery.

### 14.2 Low-level file handles (`beagle.io`)

```beagle
use beagle.io as io

match io/open("/tmp/x", "w") {      // modes: "r", "w", "a", "r+"
    Result.Ok { value } => {
        let f = value
        io/write-string(f, "hello\n")
        io/close(f)
    },
    Result.Err { error } => println("err:", error),
}

io/read-line(file, 256)    // Result.Ok { value: "..." }
io/write-stderr("oops\n")
```

### 14.3 Futures, spawn, await

```beagle
use beagle.async as async

let fut = async/spawn(fn() {
    expensive_work()
})
let v = async/await(fut)

// many at once
let [a, b] = async/await-all([f1, f2])
let first  = async/await-first([f1, f2])

// timeouts
let r = async/await-timeout(1000, fut)   // Result
async/with-timeout(1000, fn() { expensive_work() })

// cooperative cancellation
let tok = async/make-cancellation-token()
let f = async/async-with-token(tok, fn() { ... })
async/cancel(f)
```

### 14.4 Structured concurrency — `with-scope`

```beagle
async/with-scope(fn(scope) {
    let f1 = async/spawn-in-scope(scope, fn() { a() })
    let f2 = async/spawn-in-scope(scope, fn() { b() })
    [async/await(f1), async/await(f2)]
})
```

All children are awaited (or cancelled) before the scope returns.

### 14.5 Sleep / time

```beagle
async/sleep(500)           // ms, cooperative (handler-based)
async/time-now()           // nanoseconds since epoch

use beagle.timer as timer
timer/blocking-sleep(500)  // blocking version
timer/now()
let d = timer/deadline(5000)
timer/deadline-passed?(d)
timer/deadline-remaining(d)
timer/timeout(1000, fut)
```

---

## 15. Threads

OS-thread spawning is available via the global `thread` function; no import is needed. The GC is thread-safe.

```beagle
thread(fn() {
    println("running in thread")
})
```

For structured, async-aware spawning use `async/spawn` (§14.3). By default `async/spawn` uses a blocking handler; installing `ThreadedSpawnHandler` (from `beagle.spawn`) makes `async/spawn` actually run on OS threads.

Atoms (§2.10) are the preferred cross-thread communication primitive.

---

## 16. Collections — Complete Reference

All of these come from `standard-library/std.bg` and are globally available (no `use` needed).

### 16.1 Size / access

```beagle
length(coll)   count(coll)
get(coll, idx)                 get(map, key)     get(coll, k, default)
nth(coll, n)   second(coll)    third(coll)
first-of(coll) last(coll)      rest(coll)        butlast(coll)
empty?(coll)
```

### 16.2 Transformation / filtering

```beagle
map([1,2,3], fn(x) { x * 2 })                   // [2,4,6]
filter([1,2,3,4], fn(x) { x > 2 })              // [3,4]
reduce([1,2,3], 0, fn(acc, x) { acc + x })      // 6
reduce-right(coll, init, f)
flat-map(coll, f)
take(coll, n)          drop(coll, n)
take-while(coll, pred) drop-while(coll, pred)
slice(coll, start, end)                          // end exclusive
remove-at(coll, idx)
```

### 16.3 Predicates / search

```beagle
any?(coll, pred)        all?(coll, pred)
none?(coll, pred)       not-every?(coll, pred)
find(coll, pred)        find-index(coll, pred)
```

### 16.4 Aggregation / ordering

```beagle
min-of(coll)    max-of(coll)
min-by(coll, keyfn)     max-by(coll, keyfn)
sort(coll)              sort-by(coll, keyfn)
reverse(coll)
distinct(coll)          dedupe(coll)              // consecutive dups
frequencies(coll)       group-by(coll, keyfn)
```

### 16.5 Combining

```beagle
zip(a, b)        zip-with(a, b, f)
zipmap(keys, vals)
interleave(a, b) interpose(coll, sep)
concat(a, b)     flatten(coll)
into(target, source)       // "pour" source into target
```

### 16.6 Generators

```beagle
range(start, end)                  // end exclusive
range-step(start, end, step)
repeat(value, n)
repeatedly(fn, n)
iterate(fn, init, n)               // [init, f(init), f(f(init)), ...]
enumerate(coll)                    // [[0, v0], [1, v1], ...]
partition(coll, n)
partition-by(coll, keyfn)
```

### 16.7 Higher-order helpers

```beagle
identity(x)
constantly(v)
complement(pred)
compose(f, g)                      // (f ∘ g)
partial(f, a)
not(x)
truthy?(x)  falsy?(x)
nil?(x)     some?(x)
```

---

## 17. Maps — Dedicated API

```beagle
get(m, key)        get(m, key, default)
keys(m)            vals(m)
assoc(m, k, v)     dissoc(m, k)
merge(m1, m2)      merge-with(f, m1, m2)
select-keys(m, [:a :b])
update(m, k, fn)
contains-key?(m, k)
invert(m)
map-keys(m, fn)    map-vals(m, fn)
filter-keys(m, pred)    filter-vals(m, pred)

// Nested
get-in(m, [:a :b :c])
assoc-in(m, [:a :b :c], value)
update-in(m, [:a :b :c], fn)
```

---

## 18. Sets — Dedicated API

```beagle
let s = #{1, 2, 3}

set?(x)
set-contains?(s, e)
set-add(s, e)        set-remove(s, e)
set-union(a, b)      set-intersection(a, b)
set-difference(a, b)
set-subset?(a, b)
into-set(coll)
```

---

## 19. Strings — Operations

These are global functions in the core:

```beagle
length(s)
split(s, sep)                     // ["a", "b"]
join(coll, sep)                   // "a,b,c"
lines(s)                          // split on \n
words(s)                          // split on whitespace
uppercase(s)   lowercase(s)
trim(s)   trim-left(s)   trim-right(s)
starts-with?(s, prefix)
ends-with?(s, suffix)
contains?(s, sub)
index-of(s, sub)        last-index-of(s, sub)
replace(s, old, new)
replace-first(s, old, new)
replace-blank(s, new)
substring(s, start, end)
pad-left(s, width, char)
pad-right(s, width, char)
char-code("A")          // 65
char-from-code(65)      // "A"
to-string(value)
format(value, depth)    // 0 = natural, 1 = quoted repr
parse-int("42")         // 42; returns null on failure
```

For heap-y concatenation inside loops, the `++` operator is fine; for more efficient accumulation use `new-string-buffer()`.

---

## 20. Math

All available as globals; trig funcs are in radians.

```beagle
+ - * / %
abs(x)  sqrt(x)  pow(x, y)
exp(x)  log(x)   log10(x)
sin(x)  cos(x)   tan(x)
asin(x) acos(x)  atan(x)
floor(x) ceil(x) round(x) truncate(x)
min(a, b) max(a, b)
clamp(x, lo, hi)
gcd(a, b)  lcm(a, b)

even?(n)  odd?(n)  zero?(n)
positive?(n)  negative?(n)

random()        // 0.0 .. 1.0
random-int(max) // 0 .. max-1

// Constants — globally available (defined as top-level lets in the prelude)
PI   // 3.141592653589793
E    // 2.718281828459045
TAU  // 6.283185307179586

to-float(x)           // int -> float
parse-int("42")       // string -> int or null
```

---

## 21. I/O and Printing

```beagle
println(...args)        // args printed with spaces between, trailing newline
print(...args)          // same, no newline
format(value, 0)        // render value as a string, natural form
format(value, 1)        // quoted form (strings get "..." wrappers)
to-string(value)
```

Output is routed through a dynamic `*out*` variable, so you can redirect with `binding`.

---

## 22. JSON

```beagle
json-encode(42)                  // "42"
json-encode("hello")             // "\"hello\""
json-encode([1, 2, 3])           // "[1,2,3]"
json-encode({:name "Alice"})     // "{\"name\":\"Alice\"}"

json-decode("42")                // 42
json-decode("[1,2,3]")           // [1, 2, 3]
json-decode("{\"name\":\"Bob\"}") // {:name "Bob"}   (object keys become keywords)
```

---

## 23. FFI

```beagle
use beagle.ffi as ffi

// Load a library and resolve functions
let libc = ffi/load-library("libSystem.B.dylib")    // macOS path
let c_strlen = ffi/get-function(libc, "strlen",
                                [ffi/Type.String], ffi/Type.U64)
c_strlen("hello")   // 5
```

### Type enum

```beagle
ffi/Type.U8  ffi/Type.U16  ffi/Type.U32  ffi/Type.U64
ffi/Type.I8  ffi/Type.I16  ffi/Type.I32  ffi/Type.I64
ffi/Type.F32 ffi/Type.F64
ffi/Type.Pointer  ffi/Type.MutablePointer
ffi/Type.String   ffi/Type.Void
ffi/Type.Structure { types: [...] }
```

### Raw buffers

```beagle
let buf = ffi/allocate(64)        // GC-managed, auto-freed
ffi/set-u8(buf, 0, 255)
ffi/get-u8(buf, 0)
ffi/set-i32(buf, 4, -12345)
ffi/get-i32(buf, 4)
// Also: set-u16/u32/u64, set-i16/i64, set-f32/f64 + matching getters
ffi/copy-bytes(src, src_off, dst, dst_off, len)
let size = ffi/buffer-size(buf)
let big  = ffi/realloc(buf, 128)
ffi/deallocate(buf)               // explicit free
ffi/forget(buf)                   // hand ownership to C
ffi/size-of(ffi/Type.I32)         // 4
```

### Typed cells / arrays

```beagle
let c = ffi/cell(ffi/Type.I32, 42)
ffi/cell-get(c)                   // 42
ffi/cell-set(c, 7)

let arr = ffi/array(ffi/Type.F64, 10)
ffi/array-set(arr, 0, 3.14)
ffi/array-get(arr, 0)
```

### Getting OS info

```beagle
let os = get-os()        // "macos" | "linux" | ...
```

---

## 24. Sockets (TCP)

```beagle
use beagle.socket as socket

let listener = socket/listen("127.0.0.1", 8080)
socket/on-connection(listener, fn(conn) {
    let msg = socket/read(conn, 1024)
    socket/write(conn, "Echo: " ++ msg)
    socket/close(conn)
})

// Client
let sock = socket/connect("127.0.0.1", 8080)
socket/write(sock, "hi\n")
let reply = socket/read(sock, 1024)
socket/close(sock)
```

---

## 25. Streams (Lazy)

```beagle
use beagle.stream as stream

let s = stream/stream-range(1, 100)
    |> stream/stream-map(fn(x) { x * 2 })
    |> stream/stream-filter(fn(x) { x > 50 })
    |> stream/stream-take(5)
    |> stream/stream-to-vec()
```

Terminal ops: `stream-to-vec`, `stream-reduce`, `stream-first`, `stream-find`, `stream-any?`, `stream-all?`.

---

## 26. Reflection / Metaprogramming

```beagle
use beagle.reflect as reflect

type-of(42)               // Int
type-of("hi")             // String
instance-of(x, core/PersistentVector)   // true/false

reflect/doc(fn_value)                   // docstring if any
reflect/apropos("add")                  // search
reflect/namespace-members("my.ns")
reflect/all-namespaces()

eval("(+ 1 2)")                         // runtime compile+run
```

---

## 27. Testing

Beagle has two complementary styles of tests, and files can mix both freely.

### 27.1 Snapshot tests

A snapshot test runs a program and compares its stdout against an expected block written directly in the file. The marker is the exact string `// @beagle.core.snapshot`; every `//` line that follows it is expected output (a bare `//` line matches a blank line).

```beagle
namespace my_test

fn main() {
    println("hello")
    println(1 + 2)
    "done"              // return value of main is also printed
}

// @beagle.core.snapshot
// hello
// 3
// done
```

Output comparison is exact — whitespace and formatting must match.

### 27.2 Test blocks — `test "name" { ... }`

A top-level `test` form declares a named test. The body runs as part of the test suite, independent of `main`. Use these for assertion-style tests that don't need a full stdout comparison.

```beagle
namespace math_test

fn add(a, b) { a + b }
fn divide(a, b) {
    if b == 0 { throw("division by zero") }
    a / b
}

test "addition" {
    assert-eq!(add(1, 2), 3)
    assert-eq!(add(0, 0), 0)
    assert-eq!(add(-1, 1), 0)
}

test "assert truthy" {
    assert!(true)
    assert!(1)
    assert!("hello")
    assert!([1, 2, 3])       // non-empty collections are truthy
}

test "assert not equal" {
    assert-ne!(1, 2)
    assert-ne!("hello", "world")
}

test "assert throws" {
    assert-throws!(fn() { throw("boom") })
    assert-throws!(fn() { divide(1, 0) })
}
```

A test passes if its body completes without a thrown exception. Any thrown error (including from an assertion) fails the test with its message.

### 27.3 Assertion macros

Available globally (no `use` needed):

| Macro | Fails when |
|---|---|
| `assert!(x)` | `x` is `null` or `false` |
| `assert-eq!(a, b)` | `a` and `b` are not structurally equal |
| `assert-ne!(a, b)` | `a` and `b` *are* structurally equal |
| `assert-throws!(thunk)` | calling `thunk()` returns normally (no exception) |

Failure messages include a rendered form of both sides for `assert-eq!` / `assert-ne!`. The `!` suffix is part of the name — these are real calls, not macros you can omit the `!` on.

### 27.4 Discovery and running

`cargo run -- test` discovers any `.bg` file that contains either:

- the marker `// @beagle.core.snapshot`, **or**
- one or more `test "..." { ... }` blocks

Within a file both styles are honored: test blocks run, and if a snapshot marker is present the stdout of `main()` is compared against it.

```bash
cargo run -- test resources/                              # run a directory
cargo run -- test resources/my_test.bg                    # single file
cargo run -- test --update-snapshots resources/my_test.bg # accept current stdout
```

With no path, `test/` or `tests/` is used if present. Files are dependency-sorted so imports work.

### 27.5 Skipping a test file

Add a `// Skip` comment line to skip an entire file — optionally with a reason:

```beagle
// Skip: flaky on CI, investigating

namespace slow_test
// ...
```

The runner treats the file as skipped (not failed).

---

## 28. Things AIs Commonly Get Wrong

Quick hit-list — skim before writing code.

1. **String interpolation is `${...}`**, not `#{}`, not backticks. Escape `$` with `\$`.
2. **`let mut` is function-local only.** At top level use `atom(...)` for mutable state.
3. **Struct "update" uses spread**: `Point { ...p, x: 15 }`. Not `p.with(x=15)`.
4. **`break(value)` and `continue()` are function-call syntax**, not bare keywords.
5. **Two pipes, opposite directions**: `|>` = pipe-first (x becomes first arg); `|>>` = pipe-last (x becomes last arg).
6. **Map literals use spaces between key and value**: `{:a 1 :b 2}` (commas optional).
7. **Set literals are `#{...}`**, not `{...}`.
8. **Multi-arity functions use `fn name { () => ..., (x) => ..., ... }`** — no `body` keyword after the name.
9. **Variadic is `...name`** in the param list, not `&name` or `*name`. Rest comes in as an array.
10. **`use` requires an alias**: `use beagle.fs as fs`. You then call `fs/read-file(...)`.
11. **Enum variants are namespaced on the enum**: `Action.run { speed: 5 }`, and matched the same way.
12. **Protocols dispatch on the first arg**: extend a type with `extend Foo with Bar { fn m(self) { ... } }`.
13. **Most I/O returns `Result`**: pattern-match on `Result.Ok { value }` / `Result.Err { error }`.
14. **Tail calls are optimized.** Deep recursion is fine, but idiomatic code uses `for`, `while`, or `loop` for iteration.
15. **The GC is real and precise.** Don't hold raw FFI pointers in Beagle values across arbitrary operations — prefer `ffi/Cell`, `ffi/Buffer` which are GC-managed.
16. **Runtime errors are catchable AND resumable.** Prefer `Result` at API boundaries; use `try`/`catch` for true exceptional paths.

---

## 29. A Complete Small Example

```beagle
namespace wordcount

use beagle.async as async

fn word-count(path) {
    match async/read-file(path) {
        Result.Ok { value } => {
            value
                |> lowercase
                |> words
                |> frequencies
        },
        Result.Err { error } => {
            println("couldn't read ${path}: ${error}")
            {}
        },
    }
}

fn main() {
    let counts = word-count("/tmp/example.txt")
    let top = counts
        |> into([])              // {kw -> n} -> [[kw, n], ...]
        |> sort-by(fn([_, n]) { 0 - n })
        |> take(10)

    for [word, n] in top {
        println("${word}: ${n}")
    }
}
```

