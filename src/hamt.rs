#[allow(unused)]
pub struct PersistentVector {
    count: usize,
    shift: usize,
    root: *mut usize,
    tail: *mut usize,
}

// TODO: as I've been thinking about this, it becomes difficult
// For example, how do I make a namespace level binding?
// If I try to just do an allocation and save a pointer,
// that pointer can move. Do I make a pointer type that will
// be updated when the pointer moves?
// Do I make some way of interacting with this thing so that
// there is a nice way to define these data types and things in rust?
// I think I might need to whole runtime, not just an allocator.
// But the runtime is quite large and not super clear what things I need
// I could make annotations and stuff for more easily doing this.
// But I worry that is a bit much for right now
// I would imagine I will get a good performance boost from this though
