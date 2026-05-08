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

/// What a single feedback slot tells us about its arithmetic site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteShape {
    /// No observations — site never executed.
    Cold,
    /// One numeric shape observed; the site can be specialized to it.
    Monomorphic(u64),
    /// Multiple distinct numeric shapes — int+int and float+float, etc.
    /// Specializing isn't safe without a polymorphic IC.
    Polymorphic,
    /// Site reached a non-numeric path (type-error throw or non-numeric
    /// equality). Specializing on numeric assumptions is unsafe.
    NotNumeric,
}

/// Classify a single slot.
pub fn classify_slot(slot: u64) -> SiteShape {
    if slot & FB_OTHER != 0 {
        return SiteShape::NotNumeric;
    }
    let numeric = slot & (FB_INT_INT | FB_FLOAT_FLOAT | FB_INT_FLOAT | FB_FLOAT_INT);
    match numeric.count_ones() {
        0 => SiteShape::Cold,
        1 => SiteShape::Monomorphic(numeric),
        _ => SiteShape::Polymorphic,
    }
}

/// What we'd do with a function if we ran tier-up right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecializationVerdict {
    /// Function never ran (or ran but never reached any arithmetic) — no
    /// specialization signal yet.
    Cold,
    /// Every executed site is monomorphic. The function can be rebuilt
    /// with guard-free specialized arithmetic. Best case.
    FullySpecializable,
    /// At least one site is monomorphic but at least one is polymorphic
    /// or not-numeric. Tier-up could still specialize the monomorphic
    /// sites and leave the rest generic.
    PartiallySpecializable,
    /// No monomorphic numeric sites — nothing to specialize.
    NotSpecializable,
}

/// Per-function summary used by `--dump-specializable` and (eventually)
/// by tier-up to decide whether to recompile a function.
#[derive(Debug, Clone)]
pub struct FunctionFeedbackSummary {
    pub code_address: usize,
    pub debug_name: String,
    /// Total slots allocated for the function (live + cold).
    pub total: usize,
    pub cold: usize,
    /// Slot count broken down by what specialization would target. Sums
    /// across `Monomorphic` + `Polymorphic` + `NotNumeric` + `cold` ==
    /// `total`.
    pub monomorphic_int_int: usize,
    pub monomorphic_float_float: usize,
    pub monomorphic_int_float: usize,
    pub monomorphic_float_int: usize,
    pub polymorphic: usize,
    pub not_numeric: usize,
    pub verdict: SpecializationVerdict,
}

impl FunctionFeedbackSummary {
    /// Build a summary from this function's slot bits in source order.
    pub fn from_bits(code_address: usize, debug_name: String, bits: &[u64]) -> Self {
        let mut summary = FunctionFeedbackSummary {
            code_address,
            debug_name,
            total: bits.len(),
            cold: 0,
            monomorphic_int_int: 0,
            monomorphic_float_float: 0,
            monomorphic_int_float: 0,
            monomorphic_float_int: 0,
            polymorphic: 0,
            not_numeric: 0,
            verdict: SpecializationVerdict::Cold,
        };
        for &slot in bits {
            match classify_slot(slot) {
                SiteShape::Cold => summary.cold += 1,
                SiteShape::Monomorphic(FB_INT_INT) => summary.monomorphic_int_int += 1,
                SiteShape::Monomorphic(FB_FLOAT_FLOAT) => summary.monomorphic_float_float += 1,
                SiteShape::Monomorphic(FB_INT_FLOAT) => summary.monomorphic_int_float += 1,
                SiteShape::Monomorphic(FB_FLOAT_INT) => summary.monomorphic_float_int += 1,
                SiteShape::Monomorphic(_) => {
                    // Defensive: classify_slot returned a single bit not in
                    // our known set. Treat as polymorphic so we don't
                    // accidentally specialize on something unexpected.
                    summary.polymorphic += 1;
                }
                SiteShape::Polymorphic => summary.polymorphic += 1,
                SiteShape::NotNumeric => summary.not_numeric += 1,
            }
        }
        let mono_total = summary.monomorphic_int_int
            + summary.monomorphic_float_float
            + summary.monomorphic_int_float
            + summary.monomorphic_float_int;
        let nonmono = summary.polymorphic + summary.not_numeric;
        summary.verdict = if mono_total == 0 && nonmono == 0 {
            SpecializationVerdict::Cold
        } else if mono_total > 0 && nonmono == 0 {
            SpecializationVerdict::FullySpecializable
        } else if mono_total > 0 {
            SpecializationVerdict::PartiallySpecializable
        } else {
            SpecializationVerdict::NotSpecializable
        };
        summary
    }

    /// Active slots = anything that ran (anything not Cold).
    pub fn active(&self) -> usize {
        self.total - self.cold
    }
}
