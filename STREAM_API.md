# Beagle Streams API Reference

## Module: `beagle.stream`

Complete API documentation for the stream library.

---

## Core Types

### `enum StreamResult`

Result of pulling from a stream.

```beagle
enum StreamResult {
    Value { value }       // Yielded a value
    Done {}               // Stream exhausted
    Error { error }       // Error occurred, stream terminated
}
```

### `protocol StreamSource`

Interface for stream implementations.

```beagle
protocol StreamSource {
    fn next(self)         // Pull next value, return StreamResult
    fn close(self)        // Cleanup resources
}
```

### `struct Stream`

Wrapper around a StreamSource with combinators.

```beagle
struct Stream {
    source                // The underlying StreamSource
    closed                // Atom tracking if closed
}
```

---

## Basic Functions

### `fn from-source(source) -> Stream`

Create a stream from a StreamSource.

```beagle
struct MySource { /* ... */ }
extend MySource with StreamSource { /* ... */ }

let s = stream/from-source(MySource { /* ... */ })
```

### `fn next(stream) -> StreamResult`

Pull the next value from a stream.

```beagle
let result = stream/next(my_stream)
match result {
    StreamResult.Value { value } => println(value),
    StreamResult.Done {} => println("Done"),
    StreamResult.Error { error } => println("Error:", error)
}
```

### `fn close(stream) -> null`

Close stream and release resources. Safe to call multiple times.

```beagle
stream/close(my_stream)
```

---

## Stream Sources

### File I/O

#### `fn lines(path) -> Stream`

Stream file lines.

**Parameters:**
- `path`: String - File path

**Returns:** Stream yielding String lines (without newline)

**Properties:**
- Lazy: Opens file on first pull
- Memory: O(1)
- Auto-closes on Done/Error

**Example:**
```beagle
stream/lines("/var/log/app.log")
    |> stream/map(parse-entry)
    |> stream/collect()
```

#### `fn chunks(path, chunk_size) -> Stream`

Stream file in byte chunks.

**Parameters:**
- `path`: String - File path
- `chunk_size`: Number - Bytes per chunk

**Returns:** Stream yielding String chunks

**Properties:**
- Last chunk may be smaller than chunk_size
- Memory: O(chunk_size)
- Auto-closes on Done/Error

**Example:**
```beagle
stream/chunks("/data.bin", 4096)
    |> stream/for-each(process-chunk)
```

#### `fn read-dir-stream(path) -> Stream`

Stream directory entries.

**Parameters:**
- `path`: String - Directory path

**Returns:** Stream yielding String filenames

**Properties:**
- Loads directory once on first pull
- Returns filenames only (not full paths)
- Memory: O(# entries)

**Example:**
```beagle
stream/read-dir-stream("/data")
    |> stream/filter(is-text-file)
    |> stream/collect()
```

### TCP Sockets

#### `fn socket-lines(socket, separator) -> Stream`

Stream newline-delimited messages from TCP socket.

**Parameters:**
- `socket`: TcpSocket - Connected socket
- `separator`: String - Message delimiter (e.g., "\n")

**Returns:** Stream yielding String messages

**Properties:**
- Buffers partial reads internally
- Returns complete messages (separator removed)
- Memory: O(1) between messages
- Caller must close socket

**Example:**
```beagle
let conn = socket/connect("localhost", 8080)
stream/socket-lines(conn, "\n")
    |> stream/map(json-decode)
    |> stream/take(100)
    |> stream/for-each(handle-message)
socket/close(conn)
```

#### `fn socket-chunks(socket, chunk_size) -> Stream`

Stream raw bytes from socket.

**Parameters:**
- `socket`: TcpSocket - Connected socket
- `chunk_size`: Number - Max bytes per pull

**Returns:** Stream yielding String chunks

**Properties:**
- Last chunk may be smaller
- Memory: O(chunk_size)
- Caller must close socket

**Example:**
```beagle
stream/socket-chunks(conn, 8192)
    |> stream/for-each(write-to-disk)
```

### Generators

#### `fn from-generator(fn) -> Stream`

Create stream from generator function.

**Parameters:**
- `fn`: Function - Yields values, returns null to stop

**Returns:** Stream yielding whatever the generator produces

**Properties:**
- Generator called repeatedly until returns null
- Exceptions propagate as stream errors
- Memory: Depends on generator

**Example:**
```beagle
stream/from-generator(fn() {
    let x = random/next()
    if x > 0.9 { null } else { x }
})
|> stream/collect()
```

#### `fn range(start, end) -> Stream`

Stream integers [start, end).

**Parameters:**
- `start`: Number - Starting value (inclusive)
- `end`: Number - Ending value (exclusive)

**Returns:** Stream yielding Number values

**Example:**
```beagle
stream/range(0, 100)
    |> stream/filter(fn(x) { x % 2 == 0 })
    |> stream/collect()
```

#### `fn repeat(fn) -> Stream`

Create infinite stream by repeating function calls.

**Parameters:**
- `fn`: Function - Called repeatedly to generate values

**Returns:** Stream yielding function results

**⚠️ Warning:** Must use `take()` or `find()` to stop!

**Example:**
```beagle
stream/repeat(fn() { random/next() })
    |> stream/take(10)
    |> stream/collect()
```

#### `fn from-vector(vec) -> Stream`

Stream vector elements lazily.

**Parameters:**
- `vec`: Vector - Source vector

**Returns:** Stream yielding vector elements

**Example:**
```beagle
stream/from-vector([1, 2, 3, 4, 5])
    |> stream/map(fn(x) { x * 2 })
    |> stream/collect()
```

---

## Combinators

### Transformation

#### `fn map(stream, fn) -> Stream`

Transform each value.

**Parameters:**
- `stream`: Stream - Input stream
- `fn`: Function - Transform function

**Returns:** Stream with transformed values

**Properties:**
- Lazy: Transform only as values are pulled
- Exceptions in fn propagate as stream errors
- Memory: O(1)

**Example:**
```beagle
stream/range(1, 5)
    |> stream/map(fn(x) { x * x })
    |> stream/collect()  // [1, 4, 9, 16]
```

#### `fn filter(stream, predicate) -> Stream`

Keep only matching values.

**Parameters:**
- `stream`: Stream - Input stream
- `predicate`: Function - Returns true to keep value

**Returns:** Stream with filtered values

**Properties:**
- Lazy: Only processes values as pulled
- Calls predicate for each value
- Exceptions in predicate propagate as errors
- Memory: O(1)

**Example:**
```beagle
stream/range(1, 10)
    |> stream/filter(fn(x) { x % 2 == 0 })
    |> stream/collect()  // [2, 4, 6, 8]
```

#### `fn flat-map(stream, fn) -> Stream`

Map each value to a stream and flatten.

**Parameters:**
- `stream`: Stream - Input stream
- `fn`: Function - Maps value to Stream

**Returns:** Stream with flattened values

**Properties:**
- Lazy: Only pulls from inner streams as needed
- Inner stream must be from StreamSource
- Memory: O(1) unless inner streams buffer

**Example:**
```beagle
stream/range(1, 4)
    |> stream/flat-map(fn(x) {
        stream/range(1, x + 1)
    })
    |> stream/collect()  // [1, 1, 2, 1, 2, 3]
```

### Limiting

#### `fn take(stream, n) -> Stream`

Yield first n values.

**Parameters:**
- `stream`: Stream - Input stream
- `n`: Number - Count of values to take

**Returns:** Stream with first n values

**Properties:**
- Stops stream after n values
- Lazy: Only reads n values
- Memory: O(1)

**⚡ Performance:** Only reads requested values!

**Example:**
```beagle
stream/range(0, 1000000)
    |> stream/take(5)
    |> stream/collect()  // [0, 1, 2, 3, 4]
```

#### `fn take-while(stream, predicate) -> Stream`

Yield values while predicate is true.

**Parameters:**
- `stream`: Stream - Input stream
- `predicate`: Function - Stop when returns false

**Returns:** Stream while predicate is true

**Properties:**
- Stops on first false from predicate
- Lazy evaluation
- Memory: O(1)

**Example:**
```beagle
stream/range(1, 10)
    |> stream/take-while(fn(x) { x < 5 })
    |> stream/collect()  // [1, 2, 3, 4]
```

#### `fn skip(stream, n) -> Stream`

Skip first n values.

**Parameters:**
- `stream`: Stream - Input stream
- `n`: Number - Count of values to skip

**Returns:** Stream without first n values

**Properties:**
- Consumes first n values internally
- Memory: O(1)

**Example:**
```beagle
stream/range(0, 10)
    |> stream/skip(5)
    |> stream/collect()  // [5, 6, 7, 8, 9]
```

### Advanced

#### `fn merge(stream1, stream2) -> Stream`

Merge two streams.

**Parameters:**
- `stream1`, `stream2`: Stream - Input streams

**Returns:** Stream alternating between inputs

**Properties:**
- Interleaves values from both streams
- If one ends, continues with other
- If one errors, propagates error
- Memory: O(1)

**Example:**
```beagle
stream/merge(
    stream/range(1, 3),
    stream/range(10, 12)
)
|> stream/collect()  // [1, 10, 2, 11]
```

#### `fn zip(stream1, stream2) -> Stream`

Combine two streams into pairs.

**Parameters:**
- `stream1`, `stream2`: Stream - Input streams

**Returns:** Stream of [v1, v2] pairs

**Properties:**
- Stops when either stream ends
- Memory: O(1)

**Example:**
```beagle
stream/zip(
    stream/range(1, 4),
    stream/range(10, 13)
)
|> stream/collect()  // [[1,10], [2,11], [3,12]]
```

#### `fn buffered(stream, size) -> Stream`

Buffer stream values.

**Parameters:**
- `stream`: Stream - Input stream
- `size`: Number - Buffer size

**Returns:** Stream that batches values

**Properties:**
- Eagerly fills buffer to size
- Returns one value per pull (from buffer)
- Useful for batching operations
- Memory: O(size)

**Example:**
```beagle
stream/lines("/huge/file")
    |> stream/buffered(1000)
    |> stream/for-each(process-batch)
```

### Error Handling

#### `fn catch-default(stream, value) -> Stream`

Replace errors with default value.

**Parameters:**
- `stream`: Stream - Input stream
- `value` - Default value on error

**Returns:** Stream with errors replaced

**Properties:**
- Converts error to single value, ends stream
- Memory: O(1)

**Example:**
```beagle
stream/lines("/maybe-missing.txt")
    |> stream/catch-default("NOT FOUND")
    |> stream/collect()
```

#### `fn retry(stream, max_retries) -> Stream`

Retry operations on error.

**Parameters:**
- `stream`: Stream - Input stream
- `max_retries`: Number - Max retry attempts

**Returns:** Stream with retry logic

**Properties:**
- Retries failed operations up to max_retries
- After retries exhausted, propagates error
- Memory: O(1)

**Example:**
```beagle
stream/socket-lines(conn, "\n")
    |> stream/retry(3)
    |> stream/collect()
```

---

## Terminal Operations

Terminal operations consume the stream and return a result.

### Aggregation

#### `fn collect(stream) -> Vector`

Load all values into vector.

**Parameters:**
- `stream`: Stream

**Returns:** Vector of all stream values

**⚠️ Warning:** Loads entire stream into memory!

**Example:**
```beagle
stream/range(0, 5)
    |> stream/collect()  // [0, 1, 2, 3, 4]
```

#### `fn reduce(stream, init, fn) -> Any`

Accumulate to single value.

**Parameters:**
- `stream`: Stream
- `init` - Initial accumulator value
- `fn`: Function(acc, value) -> new_acc

**Returns:** Final accumulator value

**Example:**
```beagle
stream/range(1, 5)
    |> stream/reduce(0, fn(acc, x) { acc + x })
    // 10 (1+2+3+4)
```

#### `fn fold(stream, init, fn) -> Any`

Reduce with early termination.

**Parameters:**
- `stream`: Stream
- `init` - Initial accumulator
- `fn`: Function -> Result.Ok or Result.Err

**Returns:** Accumulator value

**Example:**
```beagle
stream/range(1, 100)
    |> stream/fold(0, fn(acc, x) {
        let sum = acc + x
        if sum > 1000 {
            core/Result.Err { error: "Too large" }
        } else {
            core/Result.Ok { value: sum }
        }
    })
```

### Iteration

#### `fn for-each(stream, fn) -> null`

Execute function for each value.

**Parameters:**
- `stream`: Stream
- `fn`: Function(value) -> Any (return ignored)

**Returns:** null

**Example:**
```beagle
stream/lines("/data.txt")
    |> stream/map(parse-entry)
    |> stream/for-each(save-to-database)
```

### Searching

#### `fn find(stream, predicate) -> Any`

Find first matching value.

**Parameters:**
- `stream`: Stream
- `predicate`: Function -> boolean

**Returns:** First matching value, or null if none

**Properties:**
- Stops immediately on match
- Returns null if no match

**Example:**
```beagle
stream/range(1, 100)
    |> stream/find(fn(x) { x * x > 50 })
    // 8
```

#### `fn any?(stream, predicate) -> boolean`

Check if any value matches.

**Parameters:**
- `stream`: Stream
- `predicate`: Function -> boolean

**Returns:** true if any value matches, false otherwise

**Example:**
```beagle
stream/lines("/data.txt")
    |> stream/any?(fn(line) { core/contains?(line, "ERROR") })
```

#### `fn all?(stream, predicate) -> boolean`

Check if all values match.

**Parameters:**
- `stream`: Stream
- `predicate`: Function -> boolean

**Returns:** true if all match, false otherwise

**Example:**
```beagle
stream/range(1, 10)
    |> stream/all?(fn(x) { x > 0 })
    // true
```

### Counting

#### `fn count(stream) -> Number`

Count values in stream.

**Parameters:**
- `stream`: Stream

**Returns:** Number of values

**Properties:**
- Reads entire stream
- Memory: O(1) - just counts

**Example:**
```beagle
stream/lines("/huge/file.log")
    |> stream/count()
    // Returns number of lines without loading all into memory
```

---

## Type Summary

| Function | Input | Output | Memory |
|----------|-------|--------|--------|
| `from-vector` | Vector | Stream | O(1) |
| `range` | (int, int) | Stream | O(1) |
| `lines` | String | Stream | O(1) |
| `chunks` | (String, int) | Stream | O(chunk_size) |
| `read-dir-stream` | String | Stream | O(# entries) |
| `socket-lines` | (Socket, String) | Stream | O(msg size) |
| `socket-chunks` | (Socket, int) | Stream | O(chunk_size) |
| `map` | (Stream, Fn) | Stream | O(1) |
| `filter` | (Stream, Fn) | Stream | O(1) |
| `flat-map` | (Stream, Fn) | Stream | O(1) |
| `take` | (Stream, int) | Stream | O(1) |
| `skip` | (Stream, int) | Stream | O(1) |
| `collect` | Stream | Vector | O(n) |
| `reduce` | (Stream, Any, Fn) | Any | O(1) |
| `for-each` | (Stream, Fn) | null | O(1) |
| `find` | (Stream, Fn) | Any | O(n) |
| `count` | Stream | int | O(1) |

---

## Error Handling

Streams use standard Beagle error types:

```beagle
enum Error {
    NotFound { path },
    PermissionDenied { path },
    AlreadyExists { path },
    IsDirectory { path },
    NotDirectory { path },
    Timeout { operation },
    Cancelled {},
    IO { message },
    Other { code, message }
}
```

Errors propagate through the stream pipeline and are available as `StreamResult.Error { error }`.

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `take(n)` | O(n) | Stops after n elements |
| `filter()` | O(n) worst | Reads until predicate false |
| `find()` | O(n) worst | Stops on match |
| `count()` | O(n) | Must read all |
| `collect()` | O(n) | Loads all |

### Space Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `map()` | O(1) | No buffering |
| `filter()` | O(1) | No buffering |
| `take(n)` | O(1) | Constant space |
| `buffered(n)` | O(n) | Buffers n items |
| `collect()` | O(n) | Loads all items |

---

## Notes

- All combinators are lazy (no computation until consumed)
- Streams are single-use (cannot iterate twice)
- Resource cleanup is automatic (files, sockets handled by source)
- TCP sockets must be closed by caller (stream doesn't own socket)
- Errors propagate immediately through the pipeline
