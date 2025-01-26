What would protocols/traits look like?

```rust
protocol Indexed {
    fn get(coll, index)
}

specialize fn Indexed.get(coll, index) for PersistentVector {
    
}

extend PersistentVector with Indexed {
    fn get(coll, index) {
        vector/get(coll, index)
    }
}

extend PersistentVector {
    fn get(coll, index) {

    }
}

fn get(coll, index) {
    throw Unimplemented(typeof(coll))
}

specialize fn get(coll, index) for PersistentVector {

}

let x = [1]
x.get(0)

let type = get_type_of(x)
let f = lookup(type, "get")
call(f)

Indexed.get(x, 0)

// Can I specialize fns in other modules?
// That becomes a form of monkey patching in some ways
// Or its a way of protocol based development

```

I'm also interested in this idea that any function can be called with .

```rust
fn fib(n) {
    ...
}

10.fib()

struct Point {
    x
    y
}

fn distance(x, y) {
    ..
}

let x = Point { x: 2, y: 3 }
let y = Point { x: 1, y: 5 }
x.distance(y)


fn println(x) {
    system/println(x)
}

specialize fn println(x) for Point {
    
}

x.println()
println(x)
println(2)


import { get as coll_get } from "collection"
import { get } from "other"

x.get()
x.coll_get()


```

I need to think about dynamic contexts

```rust

struct Thing {
    get
}

let x = Thing {
    get: fn() {}
}

x.get()

// Then a evaluate this

import { get } from "utils"

x.get()

```
Should I resolve the field? Should I resolve the function?
The good thing about the function is I know at compile time the
function exists. So if I go with that options, I can optimize.
And this way you can't duck type your way into breaking code.
But it does mean any code that any cache of property with the name
get is going to need to be cleared
I don't think that's the end of the world though.




On thing I do think I want to uphold is that only things you explcitly make dynamic are dynamic. So I don't want to make it so you can customize a random function. Nor do I think the standard way you do things is to pass an object with methods you call. I want predictable execution.