//! Type-feedback bitfield encoding for arithmetic and comparison sites.
//!
//! Each `+`, `-`, `*`, `/`, `%`, and comparison site owns one 8-byte slot in
//! `Compiler::arith_feedback_cache`. The slot starts zero. The fast and slow
//! paths each `OR` a bit corresponding to the operand-type combination they
//! observed. Bits are monotonic — once set they stay set — so a tier-2
//! consumer just reads the slot, counts bits, and decides whether the site
//! is monomorphic, polymorphic, or unused.

/// Both operands were tagged ints. Recorded on the inline fast path.
pub const FB_INT_INT: u64 = 1 << 0;

/// Both operands were heap-allocated floats.
pub const FB_FLOAT_FLOAT: u64 = 1 << 1;

/// Left operand was int, right was float.
pub const FB_INT_FLOAT: u64 = 1 << 2;

/// Left operand was float, right was int.
pub const FB_FLOAT_INT: u64 = 1 << 3;

/// Site reached a non-numeric path: either a runtime type-error throw
/// (arithmetic on a non-number) or a non-numeric equality comparison
/// (e.g. two structs). For tier-2 these are equivalent — "do not
/// speculate on numeric specialization" — so they share one bit.
pub const FB_OTHER: u64 = 1 << 4;

/// Mask covering every defined feedback bit. Useful for sanity asserts.
pub const FB_ALL: u64 = FB_INT_INT | FB_FLOAT_FLOAT | FB_INT_FLOAT | FB_FLOAT_INT | FB_OTHER;

/// Decode a slot value into a human-readable list of observed shapes.
/// Used by `--dump-arith-feedback` and tests.
pub fn decode_shapes(slot: u64) -> Vec<&'static str> {
    let mut shapes = Vec::new();
    if slot & FB_INT_INT != 0 {
        shapes.push("int+int");
    }
    if slot & FB_FLOAT_FLOAT != 0 {
        shapes.push("float+float");
    }
    if slot & FB_INT_FLOAT != 0 {
        shapes.push("int+float");
    }
    if slot & FB_FLOAT_INT != 0 {
        shapes.push("float+int");
    }
    if slot & FB_OTHER != 0 {
        shapes.push("other");
    }
    shapes
}

/// True if the slot recorded exactly one observed shape (or none — never
/// executed). Polymorphic sites have at least two distinct bits set.
pub fn is_monomorphic(slot: u64) -> bool {
    (slot & FB_ALL).count_ones() <= 1
}
