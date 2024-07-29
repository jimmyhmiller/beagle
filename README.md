# Beagle

This is the very beginnings of a dynamically typed, functional programming language I am building. It aims to be a modern, fast, multithreaded, dynamically typed langauge, compiling straight to machine code on the fly rather than using a VM. Currently it is in a very Proof of Concept stage with many things not working. The things that do work, only work on macos arm64.

Currently it has the following things, all at the very beginnings and chock full of bugs. But with some solid foundations to continue on.

1. A few basic types (Strings, Numbers, Booleans, etc)
2. Structs
3. Atoms
4. Closures
5. Theads
6. Garbage collection (three very basic GC implementations)
7. Tail Call Optimization (Only way to do loops)
8. Hand written parser (with basically no error handling)

While this is incredibly early days. There are some promising things I can already see. A very simple binary trees benchmark from the benchmark games out performs ruby (with yjit) on my machine by almost 2x. In some ways, this is cheating as there are some checks the code currently doesn't do. But in other ways, beagle is at a disadvantage here as our struct property lookup code is incredibly naive and slow. Even without any of these optimizations, that same benchmark is only about 30% slower than node!

## Goals

The goal is to make Beagle a language people would want to use for real development. In other words, over the long run, I want it to be popular. People seem to believe that dynamic languages are dying and will probably stay dead. But the best aspects of dynamic langauges haven't been seen by most people. This is my attempt at bringing the best of dynamically typed langauges to one place. I've taken a lot of inspiration from Clojure. But without its borrowing lisp syntax, staying away from the jvm. But, ultimately, the goal will be the interactive, repl driven development that Clojure enables. But we should also have all the multi-threading and observability benefits of the jvm without relying on that large platform. 

Getting to where I want to be will take years and years of work. But I think its doable. In the mean time my goal is to get beagle to the point where I can build a GUI debugger frontend for itself in itself. I have a rust debugger frontend I've used for it that is fairly hacked together. You can see some debugger code scattered through out. For this, we will need some C interop.

## History

This codebase originally existed [here](https://github.com/jimmyhmiller/PlayGround/tree/master/rust/asm-arm2). It outgrew that junk drawer and moved it here as it matures into proper project.


### Personal TODO list (incomplete)

* Check argument count
* Need to properly handle parse errors
* Do I want clap?
* Guard on arithemtic (Started)
* import/export
* Namespaces
* Data structures
* Decide if I want to support kabab-case
* Continuations?
* Make closures better