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