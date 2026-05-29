//! Real-corpus stats probe for the SSA regalloc.
//!
//! Runs liveness → interference → chordal coloring on a `CfgFunction`
//! and writes one log line per function under
//! `BEAGLE_SSA_REGALLOC_STATS=1`. Output format:
//!
//! ```text
//! [regalloc-stats] <fn_name> blocks=N vregs=N maxlive_gp=N maxlive_fp=N \
//!   colors_gp=N colors_fp=N edges=N edge_moves=N root_slots=N
//! ```
//!
//! Where:
//! - `maxlive_gp` / `maxlive_fp` is the maximum register pressure per
//!   `RegClass` (largest #values simultaneously live at any point). For
//!   SSA this equals the chordal max-clique = the optimal #colors.
//! - `colors_gp` / `colors_fp` is the number of distinct colors the
//!   chordal coloring actually assigned per class (the minimum #physical
//!   registers needed to avoid spilling). It must equal `maxlive_*`;
//!   a divergence is a coloring/interference bug.
//! - `edges` is the number of undirected interference edges.
//! - `edge_moves` is the number of physical moves edge resolution would
//!   materialize on block-param transfers (the phi-copy cost). Phase 5
//!   coalescing drives this toward zero.
//! - `root_slots` is the number of frame slots the function reserves
//!   (the GC-scanned root region once Phase 1 splits the slot space).
//!
//! This is the per-function half of the Phase 0 measurement harness;
//! the differential runner (`scripts/ssa_diff.sh`) aggregates these
//! across the benchmark set and fails on regression.
//!
//! Aggregate by piping through `grep [regalloc-stats] | awk` to get a
//! histogram of how wide functions get.

#![allow(dead_code)]

use crate::cfg::CfgFunction;
use crate::cfg::regalloc::spill::{Budget, allocate_with_spilling, fits_budget};
use crate::cfg::regalloc::{color, interference, liveness};

/// Run the Phase 4 analysis pipeline on `f` and emit one log line to
/// stderr if `BEAGLE_SSA_REGALLOC_STATS=1`. No-op otherwise.
///
/// Always reports the no-spill coloring (`gp_max`, `fp_max`, `edges`).
/// If `BEAGLE_SSA_REGALLOC_SPILL_STATS=1` is ALSO set, additionally
/// runs the iterative spiller under an ARM64-ish budget (24 GP /
/// 32 FP) and reports `spills_24_32` and `fits_24_32`. Spill
/// simulation is opt-in because it's quadratic per re-color and slow
/// on the largest functions (5000+ vregs).
pub fn record(f: &CfgFunction) {
    let enabled = std::env::var("BEAGLE_SSA_REGALLOC_STATS")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);
    if !enabled {
        return;
    }
    if f.blocks.is_empty() {
        return;
    }

    use crate::cfg::RegClass;
    use crate::cfg::regalloc::edge::{Scratch, resolve_edges};

    let liveness = liveness::compute_liveness(f);
    let ig = interference::build_interference(f, &liveness);
    let coloring = color::color(f, &ig);
    let (maxlive_gp, maxlive_fp) = liveness::max_live(f, &liveness);
    let colors_gp = distinct_colors(&coloring, RegClass::Gp);
    let colors_fp = distinct_colors(&coloring, RegClass::Fp);
    let edges: usize = ig.adj.values().map(|s| s.len()).sum::<usize>() / 2;

    // Count the moves edge resolution would emit on block-param
    // transfers. Scratch colors must sit outside any real color; pick
    // indices above the widest pressure so they can never collide.
    let scratch = Scratch {
        gp: (maxlive_gp as u32) + 1000,
        fp: (maxlive_fp as u32) + 1000,
    };
    let edge_moves: usize = resolve_edges(f, &coloring, scratch)
        .iter()
        .map(|e| e.moves.len())
        .sum();

    let root_slots = f.num_slots;

    let name = f.debug_name.as_deref().unwrap_or("<anonymous>");

    let spill_enabled = std::env::var("BEAGLE_SSA_REGALLOC_SPILL_STATS")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);

    if !spill_enabled {
        eprintln!(
            "[regalloc-stats] {} blocks={} vregs={} maxlive_gp={} maxlive_fp={} \
             colors_gp={} colors_fp={} edges={} edge_moves={} root_slots={}",
            name,
            f.num_blocks(),
            f.num_vregs(),
            maxlive_gp,
            maxlive_fp,
            colors_gp,
            colors_fp,
            edges,
            edge_moves,
            root_slots,
        );
        return;
    }

    let mut spill_target = f.clone();
    let budget = Budget { gp: 24, fp: 32 };
    let result = allocate_with_spilling(&mut spill_target, budget);
    let fits = fits_budget(&result.coloring, budget);
    eprintln!(
        "[regalloc-stats] {} blocks={} vregs={} maxlive_gp={} maxlive_fp={} \
         colors_gp={} colors_fp={} edges={} edge_moves={} root_slots={} \
         spills_24_32={} fits_24_32={}",
        name,
        f.num_blocks(),
        f.num_vregs(),
        maxlive_gp,
        maxlive_fp,
        colors_gp,
        colors_fp,
        edges,
        edge_moves,
        root_slots,
        result.spilled.len(),
        fits,
    );
}

/// Number of distinct colors assigned to VRegs of `class`. Unlike
/// `Coloring::max_color` (which returns the highest index and is 0 both
/// for "one color at index 0" and "no colors at all"), this is the true
/// count of physical registers the class consumes.
fn distinct_colors(
    coloring: &crate::cfg::regalloc::color::Coloring,
    class: crate::cfg::RegClass,
) -> usize {
    use std::collections::HashSet;
    coloring
        .colors
        .iter()
        .filter(|(v, _)| v.class == class)
        .map(|(_, c)| *c)
        .collect::<HashSet<_>>()
        .len()
}
