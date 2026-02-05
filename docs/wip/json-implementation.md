# JSON Implementation in Beagle

## Overview

Beagle provides `json-encode` and `json-decode` builtins for JSON serialization. This document covers the implementation, design decisions, and known limitations.

## Usage

```beagle
// Encoding
json-encode(42)              // "42"
json-encode("hello")         // "\"hello\""
json-encode({:name "Alice"}) // "{\"name\":\"Alice\"}"
json-encode([1, 2, 3])       // "[1,2,3]"

// Decoding
json-decode("42")                    // 42
json-decode("{\"x\":1}")             // {:x 1} (map with string key "x")
json-decode("[1,2,3]")               // [1, 2, 3]
json-decode("invalid")               // throws JsonError
```

## The Depth Limit Question

The current implementation has a `JSON_MAX_DEPTH` of 512. Why?

### The Problem: Recursive Descent Parsing

Both our encoder and decoder use recursive functions:

```rust
fn value_to_json(runtime: &Runtime, value: usize, depth: usize) -> Result<String, JsonError> {
    // ...
    for elem in array {
        parts.push(value_to_json(runtime, elem, depth + 1)?);  // Recursive call
    }
    // ...
}
```

Each level of JSON nesting adds a stack frame. Deeply nested JSON like:

```json
[[[[[[[[[[[[...1000 levels...]]]]]]]]]]]
```

Would require 1000+ stack frames, potentially causing a stack overflow.

### What Other Languages Do

| Language | Behavior |
|----------|----------|
| **Python** | Default recursion limit ~1000 (configurable via `sys.setrecursionlimit`) |
| **JavaScript/V8** | No explicit limit, but hits stack overflow around ~10,000-15,000 levels |
| **Go** | No explicit limit, uses recursion, will stack overflow |
| **Java (Jackson)** | Configurable `StreamReadConstraints.maxNestingDepth`, default 1000 |
| **Rust (serde_json)** | Optional `recursion_limit` feature, default unlimited (will overflow) |
| **Ruby** | No explicit limit, uses C extension, very deep nesting can crash |

Most languages either:
1. Use recursion and accept the stack overflow risk
2. Have a configurable depth limit
3. Use an iterative approach with an explicit stack

### Is 512 Too Low?

**No.** Real-world JSON rarely exceeds 20-30 levels of nesting. Here's why:

- **API responses**: Typically 3-10 levels (user -> posts -> comments -> author)
- **Config files**: Usually 2-5 levels
- **GeoJSON**: Complex geometries might reach 10-15 levels
- **Deeply nested data**: Even pathological cases rarely exceed 50 levels

512 is generous enough for any legitimate use case while protecting against:
- Maliciously crafted JSON designed to crash parsers
- Accidentally malformed JSON with runaway nesting
- Circular reference bugs that produce infinite nesting

### Could We Remove the Limit?

**Yes**, with an iterative (non-recursive) implementation:

```rust
// Instead of recursion, use an explicit stack
fn parse_json_iterative(input: &str) -> Result<Value, JsonError> {
    let mut stack: Vec<ParseState> = vec![];
    let mut current = ParseState::new();

    for token in tokenize(input) {
        match token {
            Token::ArrayStart => {
                stack.push(current);
                current = ParseState::Array(vec![]);
            }
            Token::ArrayEnd => {
                let array = current.finish_array();
                current = stack.pop().unwrap();
                current.add_value(array);
            }
            // ... etc
        }
    }

    Ok(current.finish())
}
```

This approach:
- Uses heap memory (the `stack` Vec) instead of call stack
- Can handle millions of nesting levels (limited only by RAM)
- Is more complex to implement and maintain
- Is used by high-performance JSON libraries like simdjson

### Current Decision

We chose the recursive approach with a depth limit because:

1. **Simplicity**: Recursive descent is easier to read, write, and debug
2. **Performance**: For typical JSON, recursion is fast (no heap allocation for the parse stack)
3. **Safety**: The 512 limit prevents crashes while being invisible to normal use
4. **Pragmatism**: No real user will hit 512 levels; if they do, they can restructure their data

## Type Mapping

### Beagle → JSON

| Beagle Type | JSON Type | Notes |
|-------------|-----------|-------|
| `null` | `null` | |
| `true`/`false` | `true`/`false` | |
| Int | number | |
| Float | number | NaN/Infinity throw `JsonError` |
| String | string | Properly escaped |
| Keyword | string | `:foo` → `"foo"` (loses keyword-ness) |
| Array/Vector | array | |
| Map | object | Keys converted to strings |
| Struct | object | Field names become keys |
| Function/Closure | — | Throws `JsonError` |
| Atom | — | Throws `JsonError` |

### JSON → Beagle

| JSON Type | Beagle Type |
|-----------|-------------|
| `null` | `null` |
| `true`/`false` | Bool |
| integer number | Int |
| decimal/exp number | Float |
| string | String |
| array | PersistentVector |
| object | PersistentMap (string keys) |

## Error Handling

Both functions throw `JsonError` on failure:

```beagle
try {
    json-decode("{invalid json}")
} catch (e) {
    // e is SystemError.JsonError { message: "...", location: "..." }
    println(e.message)  // "JSON parse error at position 1: Object key must be a string"
}

try {
    json-encode(fn() { 1 })
} catch (e) {
    println(e.message)  // "Cannot encode function to JSON"
}
```

### Error Types

- **Parse errors**: Invalid syntax, unexpected characters, unterminated strings
- **Encode errors**: Unencodable types (functions, closures, atoms, NaN, Infinity)
- **Depth exceeded**: Nesting deeper than 512 levels

## Float Formatting

Floats are formatted to preserve precision while being minimal:

```beagle
json-encode(3.14159)      // "3.14159"
json-encode(1.0)          // "1.0" (not "1" - keeps float identity)
json-encode(1e10)         // "10000000000.0"
json-encode(0.0000001)    // "1e-7" (scientific notation when shorter)
```

The implementation uses Rust's built-in float formatting, which implements a hybrid Grisu3/Dragon4 algorithm for optimal output.

## Limitations

1. **No streaming**: Entire JSON must fit in memory
2. **No custom serialization**: Can't define how custom types serialize
3. **Keyword keys lose identity**: `{:foo 1}` → `{"foo":1}` → `{"foo": 1}` (string key, not keyword)
4. **No BigInt support**: Large integers may lose precision as Float
5. **No duplicate key handling**: Last value wins for duplicate keys

## Future Improvements

### Short Term
- [ ] Add `json-encode-pretty` for formatted output
- [ ] Support encoding enums as tagged objects

### Medium Term
- [ ] Custom serialization via protocol (like Rust's Serialize trait)
- [ ] Streaming parser for large files

### Long Term
- [ ] Consider iterative implementation to remove depth limit
- [ ] SIMD-accelerated parsing (simdjson-style)

## References

- [RFC 8259 - The JavaScript Object Notation (JSON) Data Interchange Format](https://tools.ietf.org/html/rfc8259)
- [simdjson - Parsing Gigabytes of JSON per Second](https://github.com/simdjson/simdjson)
- [Grisu3 - Printing Floating-Point Numbers](https://www.cs.tufts.edu/~nr/cs257/archive/florian-loitsch/printf.pdf)
