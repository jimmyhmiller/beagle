# Design: Protocol-Based FFI Struct Marshalling

A design for letting beagle users pass structs to and receive structs from
C functions through the FFI, using beagle's existing `protocol` / `extend`
machinery.

Context from a real port (raylib/raygui via beagle-zelda): C APIs that
accept or return structs currently force users to hand-expand structs into
their constituent scalars and to know the AArch64 register layout. The
`Shader` handle that `LoadShaderFromMemory` returns has to be accessed as
`.low`/`.high` because there's no way to say "this is a struct of
`(u32 id, void* locs)` and I want field access." This design fixes that.

## Beagle's protocol system — what we have to work with

- Declaration: `protocol Name(TypeParams?) { fn method(self, ...) }`
  - Methods can be stubs or carry default bodies.
- Implementation: `extend Type with Protocol(TypeArgs?) { fn method(self, ...) { body } }`
- Call syntax: `method(obj, args)` — function-call form, not dot access.
- Dispatch: dynamic, single-dispatch on `self` (the first argument).
- Can extend primitives (`U32`, `F32`, `Pointer`) as well as user structs.
- No derive macros. No type constraints ("any type implementing X").
- Protocol type parameters exist (e.g. `Handler(T)`) but are name
  substitutions, not full generics.

The most important constraint: **dispatch needs a `self`**. A `decode`
method called on bytes has no value to dispatch on — it's producing the
value. That shapes the design.

## The key design choice: protocol lives on the *descriptor*

Rather than having each beagle struct type implement `FFIMarshal` (which
would need some non-existent type-constraint mechanism to locate the right
impl at decode time), the protocol attaches to a **marshaler descriptor**
value. The descriptor is the thing with an identity at both encode and
decode time — the `self` to dispatch on.

- A beagle struct stays a plain beagle struct. No FFI awareness baked in.
- A descriptor value describes how to marshal a given beagle type to/from
  bytes.
- `extend SomeDescriptor with FFIMarshal` supplies encode/decode for that
  descriptor.
- The FFI runtime is handed descriptors (in function signatures) and calls
  protocol methods to do the work.

This also means the set of "things beagle knows how to marshal" is
open-ended — users extend it with new descriptors.

## The protocol

```beagle
protocol FFIMarshal {
    fn ffi-size(self)                    // bytes this marshaler occupies
    fn ffi-align(self)                   // alignment in bytes
    fn ffi-encode(self, value, buf, off) // write `value` into buf at off
    fn ffi-decode(self, buf, off)        // read from buf at off, return a value
}
```

Every method dispatches on `self` (the descriptor). `ffi-encode` takes the
value as an ordinary argument; `ffi-decode` has no value to take — it
produces one.

Size and alignment are queried by the FFI runtime when laying out nested
structs and when allocating scratch buffers.

## Built-in primitive marshalers

Primitives get singleton descriptor values. They're cheap to construct once
at module load:

```beagle
struct FFIPrim { tag }

let U32    = FFIPrim { tag: "u32" }
let I32    = FFIPrim { tag: "i32" }
let U64    = FFIPrim { tag: "u64" }
let F32    = FFIPrim { tag: "f32" }
let F64    = FFIPrim { tag: "f64" }
let Ptr    = FFIPrim { tag: "ptr" }

extend FFIPrim with FFIMarshal {
    fn ffi-size(self) {
        match self.tag {
            "u32" => 4, "i32" => 4, "f32" => 4,
            "u64" => 8, "f64" => 8, "ptr" => 8
        }
    }

    fn ffi-align(self) { ffi-size(self) }    // natural alignment for primitives

    fn ffi-encode(self, value, buf, off) {
        match self.tag {
            "u32" => ffi/set-u32(buf, off, value),
            "i32" => ffi/set-i32(buf, off, value),
            "f32" => ffi/set-f32(buf, off, value),
            "u64" => ffi/set-u64(buf, off, value),
            "f64" => ffi/set-f64(buf, off, value),
            "ptr" => ffi/set-ptr(buf, off, value)
        }
    }

    fn ffi-decode(self, buf, off) {
        match self.tag {
            "u32" => ffi/get-u32(buf, off),
            "i32" => ffi/get-i32(buf, off),
            "f32" => ffi/get-f32(buf, off),
            "u64" => ffi/get-u64(buf, off),
            "f64" => ffi/get-f64(buf, off),
            "ptr" => ffi/get-ptr(buf, off)
        }
    }
}
```

The tag-based dispatch inside each method is unfortunate but unavoidable
without type constraints — a single `FFIPrim` struct covers every
primitive through a runtime switch. The alternative (`FFIU32`, `FFII32`,
`FFIF32`, ... as separate types with an `extend` each) is more idiomatic
but more boilerplate. Tags are fine for something this closed-set.

## Struct marshaler

The struct marshaler holds the beagle type handle plus the list of fields
with their marshalers and computed offsets:

```beagle
struct FFIStruct {
    beagle_type       // handle used to construct instances during decode
    fields            // list of FFIField
    size              // total bytes, padded to alignment
    alignment         // max of field alignments
}

struct FFIField {
    name              // string; the field name in the beagle struct
    marshaler         // any value implementing FFIMarshal
    offset            // bytes from start of struct
}

fn ffi/struct(beagle_type, field_specs) {
    // field_specs is a list of (name, marshaler) pairs.
    let mut fields = []
    let mut running = 0
    let mut struct_align = 1
    let mut i = 0
    while i < length(field_specs) {
        let name = field_specs[i][0]
        let m = field_specs[i][1]
        let a = ffi-align(m)
        // Round running up to field alignment.
        let off = if running % a == 0 { running } else { running + (a - running % a) }
        fields = push(fields, FFIField { name: name, marshaler: m, offset: off })
        running = off + ffi-size(m)
        if a > struct_align { struct_align = a }
        i = i + 1
    }
    let total = if running % struct_align == 0 { running }
                else { running + (struct_align - running % struct_align) }
    FFIStruct {
        beagle_type: beagle_type,
        fields: fields,
        size: total,
        alignment: struct_align
    }
}

extend FFIStruct with FFIMarshal {
    fn ffi-size(self)  { self.size }
    fn ffi-align(self) { self.alignment }

    fn ffi-encode(self, value, buf, off) {
        let mut i = 0
        while i < length(self.fields) {
            let f = self.fields[i]
            let field_value = ffi/get-field(value, f.name)
            ffi-encode(f.marshaler, field_value, buf, off + f.offset)
            i = i + 1
        }
    }

    fn ffi-decode(self, buf, off) {
        let mut field_values = []
        let mut i = 0
        while i < length(self.fields) {
            let f = self.fields[i]
            let v = ffi-decode(f.marshaler, buf, off + f.offset)
            field_values = push(field_values, (f.name, v))
            i = i + 1
        }
        ffi/construct-struct(self.beagle_type, field_values)
    }
}
```

The interesting line is `ffi-encode(f.marshaler, ...)` — a protocol call
whose receiver is whatever marshaler happens to live in that field. If
it's `U32`, you get the primitive impl; if it's a nested `FFIStruct`, you
get the recursive impl. **This is exactly the composition I couldn't get
cleanly out of the descriptor-switch design.** Protocol dispatch replaces
a match-over-type-tags at each level.

The two primitives the runtime needs to expose:

- `ffi/get-field(value, "name")` — read a named field out of a beagle
  struct.
- `ffi/construct-struct(type_handle, [(name, value), ...])` — build a
  beagle struct instance of the given type from a name/value list.

Both are reflection operations on the beagle struct system. If beagle
already has them, great; if not, they're the only new primitives the
runtime needs for this whole design.

## Putting it together: user code

```beagle
// Plain beagle structs — nothing FFI-aware.
struct Shader   { id, locs }
struct Vector2  { x, y }
struct Rectangle { x, y, w, h }
struct Color    { r, g, b, a }
struct Camera2D { offset, target, rotation, zoom }

// Layout descriptors — tell the FFI how each maps to C memory.
let ShaderL = ffi/struct(Shader, [
    ("id",   U32),
    ("locs", Ptr)
])

let Vector2L = ffi/struct(Vector2, [
    ("x", F32),
    ("y", F32)
])

let RectangleL = ffi/struct(Rectangle, [
    ("x", F32), ("y", F32), ("w", F32), ("h", F32)
])

// Color is 4 bytes. If we want each field to be a U8, we need a U8 primitive;
// if we want to stay with the current U32-packed approach, the whole type can
// be marshaled as a single U32. Shown here as four U8s:
let ColorL = ffi/struct(Color, [
    ("r", U8), ("g", U8), ("b", U8), ("a", U8)
])

// Nested marshalers — Camera2DL references Vector2L.
let Camera2DL = ffi/struct(Camera2D, [
    ("offset",   Vector2L),
    ("target",   Vector2L),
    ("rotation", F32),
    ("zoom",     F32)
])

// FFI function bindings take marshalers as types.
let rl_load_shader = ffi/get-function(
    raylib, "LoadShaderFromMemory",
    [ffi/Type.String, ffi/Type.String],
    ShaderL
)

let rl_draw_rectangle_rec = ffi/get-function(
    raylib, "DrawRectangleRec",
    [RectangleL, ColorL],
    ffi/Type.Void
)
```

At call sites the user works only with plain beagle structs:

```beagle
let s = rl_load_shader(null, frag_src)    // s is a Shader
s.id                                       // ordinary field access
s.locs
let Shader { id, locs } = s                // pattern-match
Shader { ...s, id: new_id }                // spread update

rl_draw_rectangle_rec(
    Rectangle { x: 10.0, y: 20.0, w: 100.0, h: 50.0 },
    Color { r: 255, g: 0, b: 0, a: 255 }
)
```

No `.low` / `.high`. No packed U32 for colors. No hand-expanded arguments.
The beagle struct is the API.

## How the FFI runtime uses the protocol

At `ffi/get-function` time, the runtime inspects each argument type and
the return type:

- If it's a raw primitive type (`ffi/Type.I32`, `ffi/Type.String`, etc.):
  existing path.
- If it's anything implementing `FFIMarshal`: treat it as a marshaled
  value. Cache `ffi-size(m)` and `ffi-align(m)` for planning.

At call time, for each struct-typed argument:

1. Allocate a scratch buffer of `ffi-size(marshaler)` bytes.
2. `ffi-encode(marshaler, value, buf, 0)` — protocol dispatches to either
   the primitive or struct impl, which handles nested composition.
3. Hand the buffer to the ABI-lowering layer, which decides whether those
   bytes go into registers (HFA rules, small-aggregate rules), into a hidden
   pointer, or onto the stack — per the target platform.

For a struct return:

1. Either pre-allocate a return buffer (large-aggregate case, passed as
   hidden pointer), or let the ABI layer stage registers into a scratch
   buffer (HFA / small-aggregate cases).
2. `ffi-decode(marshaler, buf, 0)` — protocol dispatches; primitive impls
   read bytes, struct impls recurse and construct beagle values.
3. Return the constructed beagle value to the caller.

The protocol handles **bytes ↔ beagle value**. The ABI layer handles
**bytes ↔ registers + stack + hidden pointers**. Clean split.

## Extensibility

Because `FFIMarshal` is an open protocol, users can add new marshaler
types the core FFI doesn't know about:

```beagle
// A fixed-size C array, e.g. the `int[MAX_MATERIAL_MAPS]` in raylib's Material.
struct FFIFixedArray { element, count }

extend FFIFixedArray with FFIMarshal {
    fn ffi-size(self)  { ffi-size(self.element) * self.count }
    fn ffi-align(self) { ffi-align(self.element) }
    fn ffi-encode(self, value, buf, off) {
        let stride = ffi-size(self.element)
        let mut i = 0
        while i < self.count {
            ffi-encode(self.element, value[i], buf, off + i * stride)
            i = i + 1
        }
    }
    fn ffi-decode(self, buf, off) {
        let stride = ffi-size(self.element)
        let mut out = []
        let mut i = 0
        while i < self.count {
            out = push(out, ffi-decode(self.element, buf, off + i * stride))
            i = i + 1
        }
        out
    }
}
```

Used as a field marshaler just like any built-in:

```beagle
struct Material { maps, params }
let MaterialL = ffi/struct(Material, [
    ("maps",   FFIFixedArray { element: MaterialMapL, count: MAX_MATERIAL_MAPS }),
    ("params", FFIFixedArray { element: F32, count: 4 })
])
```

Other obvious user-defined marshalers:

- **Nullable pointer → `Option<T>`:** encode null as `Option.None`, non-null
  as `Option.Some { value: decode_thing(ptr) }`.
- **`{char*, size_t}` string slice:** encode a beagle string by
  allocating a C buffer and copying; decode by reading length bytes.
- **Tagged union:** encode/decode a C `{int tag; union { ... }}` struct
  based on a user-supplied tag-to-variant mapping.

None of these require FFI runtime changes. They're just types extending
`FFIMarshal`.

## The boilerplate question

The heaviest cost of this design is writing the `ffi/struct(T, [...])`
line for each struct. For `Shader`, two fields, it's trivial. For a
raylib-scale binding (dozens of structs, some with 8+ fields), it
accumulates. Three mitigation paths, in order of increasing beagle work:

1. **Accept it.** raylib's non-obvious structs are <30; one line per
   struct totaling ~30 extra lines is a rounding error vs. the current
   `ray.bg` boilerplate for scalar expansion.
2. **Runtime macro/helper.** If beagle has a way to derive a list of
   `(name, type)` pairs from a struct declaration at runtime, a helper
   `ffi/struct-from-decl(BeagleStruct, [("id", U32), ("locs", Ptr)])` could
   cross-check the names and fail at load time if they don't match the
   struct's field list. Catches typos without any compile-time machinery.
3. **Compile-time derive.** A future `derive FFIMarshal for T where { id:
   U32, locs: Ptr }` at the declaration site, generating the descriptor
   directly. Requires compiler work; isn't strictly necessary to get the
   design off the ground.

## What this design deliberately doesn't do

- **ABI lowering:** HFA detection, eightbyte classification, hidden-pointer
  insertion. That's the FFI runtime's job; protocols don't help. Use
  libffi or emit per-target stubs.
- **Type constraints:** we can't say "this function takes any T that
  implements FFIMarshal" because beagle doesn't have that syntax. We work
  around it: the function signature mentions a specific marshaler value
  (like `ShaderL`), and the runtime just assumes marshaler values support
  the protocol. If a mismatched thing is passed, dispatch will fail at
  call time with a missing-method error. Coarser than static checking but
  workable.
- **Compile-time layout validation:** no way to check at compile time that
  the declared fields match the beagle struct. `ffi/struct` can do this
  at construction time (module load) by asking the runtime for the
  struct's field list and comparing — runtime failure, but early enough
  to catch in smoke tests.
- **Mutation propagation:** returned structs decode into plain beagle
  values; mutating them doesn't write back to any C-side memory. That's
  correct for value-semantics APIs (Shader, Rectangle, Color, Camera2D)
  and insufficient for handle-semantics APIs. Handle-semantics is a
  separate design (probably a different protocol — `FFIHandle` — with
  pointer-backed field access).

## Implementation checklist

Mandatory:

1. Reflection primitives: `ffi/get-field(value, name)` and
   `ffi/construct-struct(type_handle, fields)`. Everything else builds on
   these.
2. The `FFIMarshal` protocol and stdlib extensions for primitives and
   `FFIStruct`.
3. FFI runtime change: when a function signature references a value
   implementing `FFIMarshal` (instead of a raw `ffi/Type.X`), route
   encode/decode through the protocol and hand the bytes to ABI lowering.
4. ABI lowering for struct-by-value — this is the non-trivial part, but
   it's an orthogonal concern that has to happen regardless of the
   marshalling layer. See any reference on AAPCS64 HFA / SysV
   classification / Win64 by-reference. libffi handles all three if you'd
   rather not implement from scratch.

Nice-to-have:

5. `ffi/struct-from-decl` that validates field names against the beagle
   struct at module load.
6. User-defined marshalers shipping in stdlib: `FFIFixedArray`,
   `FFINullable`, `FFIStringSlice`.
7. Eventually: compile-time derive for the common case.

## Summary

The split is:

- **Beagle struct** — plain language value, normal field access, pattern
  match, spread update. FFI-oblivious.
- **Marshaler descriptor** — a first-class value (primitive singleton or
  `FFIStruct` built via `ffi/struct`) that implements `FFIMarshal`. Knows
  how to convert between beagle values and byte buffers.
- **FFI runtime** — calls the protocol to bridge beagle ↔ bytes, and
  handles per-ABI lowering for bytes ↔ registers.

Using protocols instead of a hand-rolled descriptor interpreter is a clean
win for composition (nested structs Just Work), extensibility (user
marshalers plug in), and alignment with the rest of beagle's design. The
one real cost is that every type needing FFI marshalling gets an
`ffi/struct(T, [...])` line — and even that goes away with a future
derive.
