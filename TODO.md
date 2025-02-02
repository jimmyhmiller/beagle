* Need to add !
* Not a fan of how I'm dealing with dots in namespace names. I'm thinking I shouldn't have them at all
* I think I should probably make namespaces not be a rust construct, but actually defined in the language. Meaning that namespaces would be normal heap objects. This would make gc much easier, as they wouldn't need to be so much of a special case. We might want to be able to say that they are objects that should go in the old generation or permanent or whatever. But having to have them be all separate is a bit awkward.
* I've got some problem with enum and registers...
* Need else if
* Need || and && operators. Or should they be `or` and `and`?
* I feel like I have a bug with moving floats. Maybe I don't. But because they are small objects, I can't do the thing where I write in the first field a pointer. And if I can't do that, I'm guessing I duplicate them when I copy them. I need some way to deal with that.
* Struct Ids should be stored untagged
* My quick and dirty thread-safe gc is slower even without multiple threads
* Do I zero on allocation?
* Code memory needs to be dynamic
* If I try to recursion multiple times in the same function (same code path), it fails. I think because of tail position calcalations
* Implement https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/jitdump-specification.txt
* Technically speaking, my arrays are now mutable. So they should have write barriers for old generation.
* I fixed a bunch of stack map code in compacting. The errors have to also exist in mark and sweep. (Maybe also generational) Cleanup and unify this stuff.
* I should probably have some gc-always runs
* Mulit Arity and Var Args
* I need the ability to re-export functions without a level of indirection
* Figure out a print system for types like the vector
* I need to do indexing with [] I just need to figure out how to distinguish a literal and indexing. How do other languages do that?
* Make namespacing reference work for structs and enums better
* Builtins need better type checking setup
* Need to make the tests start a process so if one crashes, we still know the total results
* I've got some stuff going on with malloc and free that is making rust not happy. In the case I saw, it looks like namsepace bindings are causing issues. I can find it by running the sdl example.
* I need to manage chunks of code better. I think right now I am marking things as mutable, but then code is running. If I just manage writing code in chunks rather than one big region that I keep mutating and then re-execing, this shouldn't be a problem
* Get rid of mmap now that I have libc
* I need a way to associate functions with a struct or I need to say all functions can be so associated
    * I like the latter but it has issues.
    * The former is more structured and probably faster as I can specialize
* I need a way to do iterator/seqs/whatever
* I need to move current_namespace from runtime to compiler
* I really need to solve this whole namespace vs function thing and fix that problem
* How could I make what I'm doing with protocols a general feature? I would want the ability to do codegen in reaction to registering extensions. But I also want the ability for the optimizer to figure out that we can specialize on type.


```
thread '<unnamed>' panicked at src/runtime.rs:430:59:
called `Option::unwrap()` on a `None` value
stack backtrace:
   0: rust_begin_unwind
             at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/std/src/panicking.rs:665:5
   1: core::panicking::panic_fmt
             at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/core/src/panicking.rs:76:14
   2: core::panicking::panic
             at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/core/src/panicking.rs:148:5
   3: core::option::unwrap_failed
             at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/core/src/option.rs:2009:5
   4: core::option::Option<T>::unwrap
             at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/core/src/option.rs:972:21
   5: main::runtime::Memory::active_threads
             at ./src/runtime.rs:430:29
   6: main::runtime::Runtime::gc
             at ./src/runtime.rs:1005:15
   7: main::runtime::Runtime::allocate
             at ./src/runtime.rs:871:17
   8: main::builtins::allocate
             at ./src/builtins.rs:78:18
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
   1:        0x102c0c09c - std::backtrace_rs::backtrace::trace_unsynchronized::hf4fa2da75bbd5d09
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/std/src/../../backtrace/src/backtrace/mod.rs:66:5
   2:        0x102c0c09c - std::sys::backtrace::_print_fmt::h75773692a17404c8
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/std/src/sys/backtrace.rs:66:9
   3:        0x102c0c09c - <std::sys::backtrace::BacktraceLock::print::DisplayBacktrace as core::fmt::Display>::fmt::h39ba3129e355bb22
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/std/src/sys/backtrace.rs:39:26
   4:        0x102c28268 - core::fmt::rt::Argument::fmt::h34f25d464889fcc7
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/core/src/fmt/rt.rs:177:76
   5:        0x102c28268 - core::fmt::write::h8b50d3a0f616451a
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/core/src/fmt/mod.rs:1189:21
   6:        0x102c09128 - std::io::Write::write_fmt::h4b3bbae7048e35f8
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/std/src/io/mod.rs:1884:15
   7:        0x102c0bf50 - std::sys::backtrace::BacktraceLock::print::h7934b1e389160086
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/std/src/sys/backtrace.rs:42:9
   8:        0x102c0d2b8 - std::panicking::default_hook::{{closure}}::hbcd636b20f603d1e
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/std/src/panicking.rs:268:22
   9:        0x102c0d0ec - std::panicking::default_hook::ha9081970ba26bc6c
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/std/src/panicking.rs:295:9
  10:        0x102baa350 - <alloc::boxed::Box<F,A> as core::ops::function::Fn<Args>>::call::h24231d3c986eef43
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/alloc/src/boxed.rs:1986:9
  11:        0x102baa350 - test::test_main::{{closure}}::h5e1e543293f867e4
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/test/src/lib.rs:134:21
  12:        0x102c0dae8 - <alloc::boxed::Box<F,A> as core::ops::function::Fn<Args>>::call::h3ea003f283d2c744
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/alloc/src/boxed.rs:1986:9
  13:        0x102c0dae8 - std::panicking::rust_panic_with_hook::h9a5dc30b684e2ff4
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/std/src/panicking.rs:809:13
  14:        0x102c0d6f8 - std::panicking::begin_panic_handler::{{closure}}::hbcb5de8b840ae91c
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/std/src/panicking.rs:667:13
  15:        0x102c0c560 - std::sys::backtrace::__rust_end_short_backtrace::ha657d4b4d65dc993
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/std/src/sys/backtrace.rs:170:18
  16:        0x102c0d3d8 - rust_begin_unwind
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/std/src/panicking.rs:665:5
  17:        0x102c38d38 - core::panicking::panic_nounwind_fmt::runtime::h13e8a6e8075ea543
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/core/src/panicking.rs:119:22
  18:        0x102c38d38 - core::panicking::panic_nounwind_fmt::h4a10ecea0e21f67a
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/core/src/intrinsics/mod.rs:3535:9
  19:        0x102c38db0 - core::panicking::panic_nounwind::ha9a59379b5f3f41a
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/core/src/panicking.rs:223:5
  20:        0x102c38ee0 - core::panicking::panic_cannot_unwind::h1bb1158913507f0a
                               at /rustc/e71f9a9a98b0faf423844bf0ba7438f29dc27d58/library/core/src/panicking.rs:315:5
  21:        0x102b3d8e4 - main::builtins::allocate::h2a7f9c4eebd6ee94
                               at /Users/runner/work/beagle/beagle/src/builtins.rs:74:1
thread caused non-unwinding panic. aborting.
error: test failed, to rerun pass `--bin main`
```


* My janky after_return error stuff is actually calling the start of the function
* I'm guessing this is a side-effect of messing up labels after register allocation
* I am going to either 1. get rid of string constants all together or 2. return a pointer to an object that looks identical whether it is a string constant or not. I could have a buffer of memory that stores my string constants. I'm not sure the correct answer here
* I need to resolve this problem where persistent_vector isn't loaded by std and can't be statically imported because of Struct
    * I could add dynamic imports after top level of beagle.core runs
    * I could explicitly add multiple standard things that get loaded