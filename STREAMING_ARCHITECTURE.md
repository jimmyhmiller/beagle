# Beagle Streaming Architecture

## Design Principle: Raw Sources + Composable Decoders

This is how production streaming systems work (Node.js streams, Rust iterators, etc.).

### The Pattern

```
Raw Byte Source → Decoder/Splitter → Consumer
                                   ↓
                            (interpretation happens here)
```

**Raw sources** don't care what the data means - they just emit bytes.
**Decoders** decide how to interpret those bytes.

### Raw Byte Sources

These just read and emit, nothing more:

```beagle
file-stream(path)           // Read file, emit 4KB chunks
file-stream-sized(path, n)  // Read file, emit n-byte chunks
socket-stream(socket)       // Read socket, emit chunks
socket-stream-sized(s, n)   // Read socket, emit n-byte chunks
```

They don't care about delimiters, message boundaries, or encoding. Just raw bytes.

### Composable Decoders

These transform byte streams by interpreting the data:

```beagle
split-on(stream, delimiter)  // Buffer until delimiter found
lines(stream)                // Convenience: split-on(stream, "\n")
by-size(stream, n)           // Buffer until n bytes, emit
```

These work with **any** byte stream.

### Why This is Better

| Scenario | Old Design | New Design |
|----------|-----------|-----------|
| Read file as lines | `stream/lines(path)` | `file-stream(path) \|> lines()` |
| Read file as CSV | Not possible cleanly | `file-stream(path) \|> split-on(",")` |
| Read TCP as lines | `stream/socket-lines(s, "\n")` | `socket-stream(s) \|> lines()` |
| Read TCP as 1KB packets | `stream/socket-chunks(s, 1024)` | `socket-stream(s) \|> by-size(1024)` |
| Read TCP as JSON (custom delim) | Not possible | `socket-stream(s) \|> split-on("\n\n")` |

### Examples

**Log file analysis:**
```beagle
file-stream("/var/log/app.log")
  |> stream/lines()
  |> stream/filter(fn(line) { contains(line, "ERROR") })
  |> stream/count()
```

**CSV parsing:**
```beagle
file-stream("data.csv")
  |> stream/lines()
  |> stream/skip(1)  // Skip header
  |> stream/map(fn(row) {
    row |> split-on(",") |> collect()  // Split row on commas
  })
  |> stream/for-each(process-record)
```

**Binary protocol (fixed-size packets):**
```beagle
socket-stream(conn)
  |> stream/by-size(1024)  // 1KB packets
  |> stream/map(parse-packet)
  |> stream/for-each(handle)
```

**HTTP headers (double CRLF delimiter):**
```beagle
socket-stream(conn)
  |> stream/split-on("\r\n\r\n")
  |> stream/take(1)  // Just headers
  |> stream/map(parse-http-headers)
```

**JSON lines (newline-delimited JSON):**
```beagle
file-stream("/data/ndjson")
  |> stream/lines()
  |> stream/map(json-decode)
  |> stream/for-each(process-json)
```

## Implementation Details

### Raw Sources (FileChunkSource, SocketChunkSource)

```beagle
struct FileChunkSource {
    path
    chunk_size
    file_handle
    eof
}

extend FileChunkSource with StreamSource {
    fn source-next(self) {
        // Lazy open, read chunk_size bytes, emit
        // That's it - no interpretation
    }

    fn source-close(self) {
        // Close file handle
    }
}
```

Just reads and emits. No special logic.

### Decoders (SplitOnSource, BySizeSource)

```beagle
struct SplitOnSource {
    upstream
    delimiter
    buffer  // Atom holding incomplete data
}

extend SplitOnSource with StreamSource {
    fn source-next(self) {
        // Loop:
        //   - Check if buffer has complete message (contains delimiter)
        //   - If yes: extract message, return it
        //   - If no: pull more data from upstream, append to buffer
        //   - On upstream EOF: return buffered data, then Done
    }

    fn source-close(self) {
        close(self.upstream)
    }
}
```

Buffers and interprets the data stream. Works with any upstream source.

## Why This Design

### Flexibility
- One decoder works for many sources
- Easy to add new decoders without touching sources
- Arbitrary delimiters, not just newlines

### Composability
- Chain sources and decoders freely
- `file-stream(p) |> lines() |> map(f) |> filter(p) |> collect()`
- Each layer is independent

### Reusability
- `split-on` works for files, sockets, generators, anything
- No duplicate buffering logic
- Test decoders separately from sources

### Memory Efficiency
- Raw sources emit on-demand
- Decoders buffer only what's necessary
- No implicit buffering surprises

### Scalability
- Easy to add: `split-on-regex`, `decompress-gzip`, `decode-json`, `rate-limit`, etc.
- Each decoder is small and focused
- Composable optimization (fusion, etc.)

## Key Insight

**The stream doesn't need to know what the bytes mean.**

The source's job: "Here are some bytes."
The decoder's job: "Here's what those bytes mean."
The combinator's job: "Here's what to do with those interpretations."

This separation of concerns is fundamental to scalable streaming systems.

## Comparison to Other Languages

### Node.js Streams
```javascript
fs.createReadStream('/path')
  .pipe(split('\n'))  // Decoder
  .pipe(map(parse))   // Combinator
```

### Rust Iterators
```rust
std::fs::read_to_string('/path')?
  .lines()            // Decoder
  .filter(|l| l.contains("ERROR"))  // Combinator
  .collect()
```

### Beagle Streams
```beagle
file-stream('/path')
  |> stream/lines()             // Decoder
  |> stream/filter(has-error)   // Combinator
  |> stream/collect()
```

Same pattern across all three: raw source → decoder → combinators.

## Future Decoders

The beauty of this design is adding new decoders is trivial:

```beagle
// JSON lines decoder
fn json-lines(stream) {
    split-on(stream, "\n")
      |> stream/map(json-decode)
}

// Fixed-size line buffering
fn buffered-lines(stream, size) {
    lines(stream)
      |> stream/buffered(size)
}

// Regex-based splitting
fn split-on-regex(stream, pattern) {
    // ... implementation ...
}

// Gzip decompression
fn gunzip(stream) {
    // ... implementation ...
}

// Rate limiting
fn rate-limit(stream, items-per-sec) {
    // ... implementation ...
}
```

All of these compose naturally with existing streams and combinators.

## Conclusion

Beagle's streaming system now follows the **proven pattern** used by production streaming libraries worldwide:

- ✅ **Raw sources** that emit uninterpreted data
- ✅ **Composable decoders** that interpret the data
- ✅ **Combinators** that transform interpreted values
- ✅ **Clean separation** between transport and interpretation
- ✅ **Maximum flexibility** and extensibility

This is the right way to build streaming systems.
