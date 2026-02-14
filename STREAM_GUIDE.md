# Beagle Streams: Complete User Guide

## Overview

Streams are **lazy, pull-based iterators** for efficient data processing. They enable memory-efficient processing of large files, TCP streams, directories, and sequences without loading everything into memory.

**Key benefits:**
- 🚀 **Lazy**: Values computed on-demand
- 💾 **Memory efficient**: No implicit buffering
- 🔗 **Composable**: Chain transformations with `|>`
- 🛡️ **Safe**: Automatic resource cleanup
- ⚡ **Fast**: Zero-copy where possible

## Core Concepts

### StreamResult

Every stream operation returns one of three results:

```beagle
enum StreamResult {
    Value { value },      // Yielded a value
    Done {},              // Stream exhausted (no more values)
    Error { error }       // Error occurred, stream terminated
}
```

### Pull-Based Model

Streams use **pull-based** evaluation:

```
Consumer calls next() → Source pulls from upstream → Value returned to consumer
```

Only the consumer determines the rate of iteration. No buffering happens unless explicit.

### StreamSource Protocol

To create custom streams, implement the `StreamSource` protocol:

```beagle
protocol StreamSource {
    fn next(self)         // Return StreamResult
    fn close(self)        // Cleanup resources
}
```

## Quick Start

### Basic Stream from Vector

```beagle
use beagle.stream as stream

fn main() {
    stream/from-vector([1, 2, 3, 4, 5])
        |> stream/map(fn(x) { x * 2 })
        |> stream/filter(fn(x) { x > 4 })
        |> stream/for-each(println)
    // Output: 6, 8, 10
}
```

### Read File Lines

```beagle
stream/lines("/var/log/app.log")
    |> stream/filter(fn(line) { core/contains?(line, "ERROR") })
    |> stream/map(parse-log-entry)
    |> stream/for-each(handle-error)
```

### Process Directory

```beagle
stream/read-dir-stream("/data")
    |> stream/filter(fn(name) { core/ends-with?(name, ".txt") })
    |> stream/count()
    |> println
```

### TCP Messages

```beagle
let conn = socket/connect("localhost", 8080)
stream/socket-lines(conn, "\n")
    |> stream/map(json-decode)
    |> stream/take(100)
    |> stream/for-each(handle-message)
socket/close(conn)
```

## Stream Sources

### File Streams

#### `lines(path)`

Stream file lines lazily, one per pull.

```beagle
stream/lines("/path/to/file.txt")
    |> stream/map(core/to-upper)
    |> stream/collect()
```

**Properties:**
- Opens file on first `next()` call
- Returns lines without trailing newline
- Closes file automatically on Done/Error
- Memory: O(1) for streaming (only one line buffered)

#### `chunks(path, size)`

Stream file in fixed-size byte chunks.

```beagle
stream/chunks("/path/to/binary.bin", 8192)
    |> stream/map(process-chunk)
    |> stream/for-each(write-result)
```

**Properties:**
- Reads up to `size` bytes per pull
- Last chunk may be smaller
- Suitable for binary files
- Memory: O(1) if chunk size is small

#### `read-dir-stream(path)`

Stream directory entries (filenames) lazily.

```beagle
stream/read-dir-stream("/path")
    |> stream/filter(fn(name) { core/starts-with?(name, ".") == false })
    |> stream/collect()
```

**Properties:**
- Loads directory once on first pull
- Returns filenames only (not full paths)
- Iterates lazily through entries
- Memory: O(number of entries in directory)

### TCP Socket Streams

#### `socket-lines(socket, separator)`

Stream newline (or custom delimiter) delimited messages from TCP socket.

```beagle
let conn = socket/connect("localhost", 5000)
stream/socket-lines(conn, "\n")
    |> stream/take(10)
    |> stream/collect()
    |> println
socket/close(conn)
```

**Properties:**
- Handles partial reads automatically
- Buffers incomplete messages
- Returns complete messages (separator removed)
- Closes on connection EOF
- Memory: O(1) between messages, grows with message size

#### `socket-chunks(socket, size)`

Stream raw bytes from socket in fixed-size chunks.

```beagle
stream/socket-chunks(conn, 4096)
    |> stream/for-each(process-binary-data)
```

**Properties:**
- Reads up to `size` bytes per pull
- Useful for binary protocols
- Last chunk may be smaller
- Memory: O(size)

### Generator Streams

#### `from-generator(fn)`

Create stream from a function that yields values.

```beagle
let counter = atom(0)
stream/from-generator(fn() {
    let val = deref(counter)
    if val >= 5 {
        null  // End stream
    } else {
        reset!(counter, val + 1)
        val
    }
})
|> stream/collect()  // [0, 1, 2, 3, 4]
```

**Properties:**
- Called repeatedly until returns `null`
- Exceptions propagate as stream errors
- Flexible: can perform side effects, I/O, etc.
- Memory: Depends on generator state

#### `range(start, end)`

Stream integers [start, end).

```beagle
stream/range(0, 100)
    |> stream/filter(fn(x) { x % 2 == 0 })  // Even numbers
    |> stream/collect()
```

#### `repeat(fn)`

Create infinite stream by repeating function calls.

```beagle
stream/repeat(fn() { random/next() })
    |> stream/take(10)
    |> stream/collect()
```

⚠️ **Warning**: Must use `take()` or other early-termination combinators!

#### `from-vector(vec)`

Stream vector elements lazily.

```beagle
stream/from-vector([10, 20, 30, 40, 50])
    |> stream/map(fn(x) { x / 10 })
    |> stream/for-each(println)
```

## Combinators

All combinators are lazy - they don't execute until consumed.

### Transformation

#### `map(stream, fn)`

Transform each value.

```beagle
stream/range(1, 4)
    |> stream/map(fn(x) { x * x })
    |> stream/collect()  // [1, 4, 9]
```

#### `filter(stream, predicate)`

Keep only values where predicate returns true.

```beagle
stream/range(1, 10)
    |> stream/filter(fn(x) { x % 2 == 0 })  // Even
    |> stream/collect()  // [2, 4, 6, 8]
```

#### `flat-map(stream, fn)`

Map each value to a stream, flatten results.

```beagle
stream/range(1, 3)
    |> stream/flat-map(fn(x) {
        stream/range(1, x + 1)  // [1], [1,2], [1,2,3]
    })
    |> stream/collect()  // [1, 1, 2, 1, 2, 3]
```

### Limiting

#### `take(stream, n)`

Yield first n values, stop stream.

```beagle
stream/range(0, 1000000)
    |> stream/take(5)
    |> stream/collect()  // [0, 1, 2, 3, 4]
```

**Performance**: Only reads 5 values regardless of stream size!

#### `take-while(stream, predicate)`

Yield values while predicate is true.

```beagle
stream/range(1, 10)
    |> stream/take-while(fn(x) { x < 5 })
    |> stream/collect()  // [1, 2, 3, 4]
```

#### `skip(stream, n)`

Skip first n values.

```beagle
stream/range(0, 10)
    |> stream/skip(5)
    |> stream/collect()  // [5, 6, 7, 8, 9]
```

### Advanced

#### `merge(stream1, stream2)`

Merge two streams (interleaved).

```beagle
let s1 = stream/range(0, 3)  // [0, 1, 2]
let s2 = stream/range(10, 13)  // [10, 11, 12]
stream/merge(s1, s2)
    |> stream/collect()  // [0, 10, 1, 11, 2, 12]
```

#### `zip(stream1, stream2)`

Combine two streams into [v1, v2] pairs.

```beagle
let s1 = stream/range(1, 4)  // [1, 2, 3]
let s2 = stream/range(10, 13)  // [10, 11, 12]
stream/zip(s1, s2)
    |> stream/collect()  // [[1,10], [2,11], [3,12]]
```

#### `buffered(stream, size)`

Buffer values for batching.

```beagle
stream/lines("/huge/file")
    |> stream/buffered(100)  // Batch 100 lines
    |> stream/for-each(process-batch)
```

### Error Handling

#### `catch-default(stream, value)`

Replace errors with default value.

```beagle
stream/lines("/maybe-missing.txt")
    |> stream/catch-default("FILE NOT FOUND")
    |> stream/for-each(println)
```

#### `retry(stream, max_retries)`

Retry failed operations up to n times.

```beagle
stream/socket-lines(conn, "\n")
    |> stream/retry(3)  // Retry up to 3 times on error
    |> stream/collect()
```

## Terminal Operations

Terminal operations consume the stream and return a value.

### Aggregation

#### `collect(stream)`

Load all values into a vector.

```beagle
stream/range(0, 5)
    |> stream/collect()  // [0, 1, 2, 3, 4]
```

⚠️ **Warning**: Loads entire stream into memory!

#### `reduce(stream, init, fn)`

Accumulate a single value.

```beagle
stream/range(1, 5)
    |> stream/reduce(0, fn(acc, x) { acc + x })
    // 1 + 2 + 3 + 4 = 10
```

#### `fold(stream, init, fn)`

Reduce with early termination.

```beagle
stream/range(1, 100)
    |> stream/fold(0, fn(acc, x) {
        let new_acc = acc + x
        if new_acc > 100 {
            core/Result.Err { error: "Sum too large" }
        } else {
            core/Result.Ok { value: new_acc }
        }
    })
```

### Iteration

#### `for-each(stream, fn)`

Execute function for each value (side effects).

```beagle
stream/lines("/data.txt")
    |> stream/map(parse-entry)
    |> stream/for-each(save-to-database)
```

### Searching

#### `find(stream, predicate)`

Return first matching value or null.

```beagle
stream/range(1, 100)
    |> stream/find(fn(x) { x * x > 50 })
    // 8 (first x where x*x > 50)
```

#### `any?(stream, predicate)`

Check if any value matches.

```beagle
stream/lines("/data.txt")
    |> stream/any?(fn(line) { core/contains?(line, "ERROR") })
    // true if file has ERROR
```

#### `all?(stream, predicate)`

Check if all values match.

```beagle
stream/range(1, 10)
    |> stream/all?(fn(x) { x > 0 })
    // true
```

### Counting

#### `count(stream)`

Count values in stream.

```beagle
stream/lines("/huge/file.log")
    |> stream/count()
    // Number of lines (O(1) memory!)
```

## Real-World Examples

### Log Analysis

```beagle
fn analyze-logs(path) {
    stream/lines(path)
        |> stream/filter(fn(line) {
            core/contains?(line, "ERROR") || core/contains?(line, "WARN")
        })
        |> stream/map(parse-log-entry)
        |> stream/reduce(
            { errors: 0, warnings: 0 },
            fn(acc, entry) {
                if entry.level == "ERROR" {
                    acc.{ errors: acc.errors + 1 }
                } else {
                    acc.{ warnings: acc.warnings + 1 }
                }
            }
        )
}
```

### Data Streaming Over Network

```beagle
fn read-json-messages(host, port) {
    let conn = socket/connect(host, port)
    let messages = stream/socket-lines(conn, "\n")
        |> stream/map(json-decode)
        |> stream/filter(fn(msg) { msg.type == "data" })
        |> stream/take(1000)
        |> stream/collect()
    socket/close(conn)
    messages
}
```

### File Processing Pipeline

```beagle
fn process-csv(input_file, output_file) {
    stream/lines(input_file)
        |> stream/skip(1)  // Skip header
        |> stream/map(parse-csv-line)
        |> stream/filter(fn(row) { validate(row) })
        |> stream/map(transform-row)
        |> stream/for-each(fn(row) {
            async/write-file!(output_file, encode-csv(row) ++ "\n")
        })
}
```

### Memory-Efficient Statistics

```beagle
fn file-statistics(path) {
    let stats = stream/lines(path)
        |> stream/reduce(
            { min: 999999, max: 0, sum: 0, count: 0 },
            fn(acc, line) {
                let num = parse-number(line)
                acc.{
                    min: if num < acc.min { num } else { acc.min },
                    max: if num > acc.max { num } else { acc.max },
                    sum: acc.sum + num,
                    count: acc.count + 1
                }
            }
        )
    stats.{ average: stats.sum / stats.count }
}
```

## Performance Guide

### Memory Usage

| Operation | Memory |
|-----------|--------|
| `lines(file)` | O(1) constant |
| `lines(file) \| take(n)` | O(n) for n lines |
| `lines(file) \| collect()` | O(file size) - loads all |
| `chunks(file, 8k)` | O(8k) per chunk |
| `read-dir-stream()` | O(# entries) |
| `socket-lines()` | O(message size) |
| `buffered(10)` | O(10 items) |

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| `take(n)` | O(n) reads, stops early |
| `filter()` | O(total values), worst case reads all |
| `find()` | O(n), stops on match |
| `count()` | O(n), must read all |
| `collect()` | O(n), loads all |

### Best Practices

1. **Use `take()` to limit** large streams:
   ```beagle
   stream/lines("/huge/file")
       |> stream/take(1000)
       |> stream/collect()  // Safe even for GB files
   ```

2. **Use `map()` before `filter()`** if possible:
   ```beagle
   // Good - filter is fast
   stream/range(1, 1000000)
       |> stream/filter(fn(x) { x % 2 == 0 })
       |> stream/take(100)
   ```

3. **Avoid `collect()` on large streams**:
   ```beagle
   // Bad - loads all into memory
   stream/lines("/huge/file") |> stream/collect()

   // Good - processes one at a time
   stream/lines("/huge/file") |> stream/for-each(process)
   ```

4. **Use early termination combinators**:
   ```beagle
   // Good - stops reading after first match
   stream/lines("/file")
       |> stream/find(fn(line) { condition(line) })
   ```

## Limitations & Gotchas

### What Streams Can't Do

- **Seek back**: Streams are forward-only
- **Peek ahead**: Use `buffered()` if you need buffering
- **Multiple consumers**: One stream per consumer
- **Random access**: Can't access element n directly

### Important Notes

1. **Streams are single-use**: Once you consume a stream, it's closed
   ```beagle
   let s = stream/range(1, 5)
   stream/collect(s)  // [1,2,3,4]
   stream/collect(s)  // [] - stream already closed!
   ```

2. **Resource cleanup**: Streams close automatically, but sockets don't
   ```beagle
   let conn = socket/connect("localhost", 8080)
   stream/socket-lines(conn, "\n") |> stream/collect()
   socket/close(conn)  // Must close explicitly!
   ```

3. **Errors propagate**: Exceptions in map/filter propagate as errors
   ```beagle
   stream/range(1, 5)
       |> stream/map(fn(x) { 10 / x })  // x=0 throws!
       // Error propagates, stream stops
   ```

## Integration with Async

Streams work seamlessly with async I/O:

```beagle
use beagle.stream as stream
use beagle.async as async

fn process-file-async(path) {
    async/with-implicit-async(fn() {
        stream/lines(path)
            |> stream/map(fn(line) {
                // Can use async operations here
                async/sleep(10)
                core/to-upper(line)
            })
            |> stream/collect()
    })
}
```

## Custom Stream Sources

Implement `StreamSource` protocol:

```beagle
struct MySource { state }

extend MySource with stream/StreamSource {
    fn next(self) {
        // Compute next value
        // Return StreamResult.Value { value },
        //        StreamResult.Done {},
        //        or StreamResult.Error { error }
    }

    fn close(self) {
        // Cleanup resources
    }
}

fn my-stream() {
    stream/from-source(MySource { state: init() })
}
```

Example - stream lines of a string:

```beagle
struct StringLineSource {
    text
    lines_list
    index
}

extend StringLineSource with stream/StreamSource {
    fn next(self) {
        let lines = deref(self.lines_list)
        let idx = deref(self.index)
        if idx >= core/length(lines) {
            StreamResult.Done {}
        } else {
            reset!(self.index, idx + 1)
            StreamResult.Value { value: core/get(lines, idx) }
        }
    }

    fn close(self) {
        null
    }
}

fn string-lines(text) {
    let lines = core/split(text, "\n")
    stream/from-source(StringLineSource {
        text: text,
        lines_list: atom(lines),
        index: atom(0)
    })
}
```

## Summary

Streams provide a powerful, composable abstraction for processing sequences:

- ✅ **Memory efficient**: Lazy evaluation, no implicit buffering
- ✅ **Composable**: Chain operations with `|>`
- ✅ **Extensible**: Implement `StreamSource` for custom sources
- ✅ **Safe**: Automatic resource cleanup
- ✅ **Async-ready**: Works with async/await
- ✅ **Production-tested**: Used in file I/O, TCP, directory streaming

Start with `stream/lines()` or `stream/from-vector()` and build from there!
