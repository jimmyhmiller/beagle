# Beagle

![Cargo Tests](https://github.com/jimmyhmiller/beagle/actions/workflows/main.yml/badge.svg)

> **Pre-alpha software.** Not finished, not fully baked.

Beagle is a dynamically-typed, mostly functional programming language that compiles directly to native machine code — no VM, no bytecode, no JIT warmup.

## Code examples


#### Enums and pattern matching

```rust
enum Cat {
    tabby,
    calico,
}

enum Dog {
    beagle,
    poodle,
}

fn feed(pet) {
    match pet {
        Cat.tabby => "lasagna",
        Cat.calico => "salmon",
        Dog.beagle => "everything in sight",
        Dog.poodle => "kibble",
    }
}
```

#### Protocols

```rust
protocol Area {
    fn area(self)
}

struct Circle { radius }
struct Rect { width, height }

extend Circle with Area {
    fn area(self) {
        3.14 * self.radius * self.radius
    }
}

extend Rect with Area {
    fn area(self) {
        self.width * self.height
    }
}

area(Circle { radius: 5 })       // 78.5
area(Rect { width: 3, height: 4 }) // 12
```

#### Threads and atoms

```rust
let counter = atom(0)

for i in range(0, 10) {
    thread(fn() {
        swap!(counter, fn(x) { x + 1 })
    })
}
```


#### Effect handlers

```rust
use beagle.effect as effect

enum Logger {
    Log { message }
}

fn log(message) {
    perform Logger.Log { message: message }
}

struct ConsoleLogger {}

extend ConsoleLogger with effect/Handler(Logger) {
    fn handle(self, op, resume) {
        println(op.message)
        resume(null)
    }
}

handle effect/Handler(Logger) with ConsoleLogger {} {
    log("hello from effects")
}
```

#### FFI

```rust
use beagle.ffi as ffi

let libc = ffi/load-library("libSystem.B.dylib")
let c_strlen = ffi/get-function(libc, "strlen", [ffi/Type.String], ffi/Type.U64)
println(ffi/call-function(c_strlen, ["hello"])) // 5
```


## Examples

The [`examples/`](examples/) directory has self-contained programs you can run:

- **[`appkit_counter.bg`](examples/appkit_counter.bg)** — A native macOS counter app using AppKit via Objective-C FFI. No bindings generator, just raw `objc_msgSend` calls.
- **[`raylib_game.bg`](examples/raylib_game.bg)** — A falling-squares game using raylib. Arrow keys to move, catch the squares. Requires `brew install raylib`.
- **[`echo_server.bg`](examples/echo_server.bg)** — A multi-client TCP echo server. Each connection gets its own thread. Test with `nc localhost 8080`.

```bash
cargo run -- examples/echo_server.bg
cargo run -- examples/appkit_counter.bg
cargo run -- examples/raylib_game.bg
```

## Vision

Beagle has the goal of being a fast dynamically typed language with deep support for live coding. The popular dynamic languages (javascript, ruby, python, etc) have not explored the true power of their dynamicness. With beagle I hope to show just how far we can take this.

## AI Usage

I have used both Claude and Codex quite extensively in developing beagle. It is not a vibe coded project by any stretch. It started well before these AI agents and does get as much time and attention as I can give it as a side project.

## Implementation Details

Beagle has its own x86 and arm backends. It does not rely on any external libraries for codegen (no llvm, cranelift etc). I have implemented an interface for gc to be pluggable (at compile time) and implemented three different gc's for it. One is a compacting collector using Cheney's algorithm, the other a simple mark and sweep, and finally a very simple generational collector. None of these are currently concurrent. But they do have a stop the world thread-safe mechanism.

Beagle also supports a decent amount of c ffi. Originally this was via libffi, but I have since replaced this with a custom implementation that follows the c abi. I am certain there are rough edges here. But it has been enough both for some raylib applications and some appkit applications.

## In Progress Areas

* Beagle supports async without function coloring via delimited continuations. But currently this system is a bit of a mess. Not as performant as it should be but also has some gc issues.
* Beagle is able to utilize proper system threads, but has some gc issues around this as well.
* We need much better tooling for understanding the running system.
* Language features have not been finalized. One thing I have considered is a nice macro system. It is definitely on the long term roadmap, but something I want to design carefully.
* Many error messages today are awful and there are even things that should be recoverable errors that panic.
