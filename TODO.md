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