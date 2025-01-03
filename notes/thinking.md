What is the proper split between runtime and compiler?

I need my runtime to be thread-safe. So maybe I should have the compiler have a reference to the runtime? Then the compiler could be single threaded and to get it to do work you pass it on the queue. But to lookup information about the runtime or to add features, it just directly accesses it. I could maybe refactor to this point by making a shared data structure at first?

I'm not reall sure as this whole thing is a bit of a mess. Especially the namespace vs function/structs/enums business.