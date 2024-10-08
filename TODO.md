* Negative Numbers
* I think I should probably make namespaces not be a rust construct, but actually defined in the language. Meaning that namespaces would be normal heap objects. This would make gc much easier, as they wouldn't need to be so much of a special case. We might want to be able to say that they are objects that should go in the old generation or permanent or whatever. But having to have them be all separate is a bit awkward.
* I've got some problem with enum and registers...
* Need else if
* Need || and && operators. Or should they be `or` and `and`?
* I feel like I have a bug with moving floats. Maybe I don't. But because they are small objects, I can't do the thing where I write in the first field a pointer. And if I can't do that, I'm guessing I duplicate them when I copy them. I need some way to deal with that.
* Struct Ids should be stored untagged
* Sizes should be stored unmultiplied
* My quick and dirty thread-safe gc is slower even without multiple threads