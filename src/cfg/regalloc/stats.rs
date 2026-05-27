//! Real-corpus stats probe for the SSA regalloc.
//!
//! Runs liveness → interference → chordal coloring on a `CfgFunction`
//! and writes one log line per function under
//! `BEAGLE_SSA_REGALLOC_STATS=1`. Output format:
//!
//! ```text
//! [regalloc-stats] <fn_name> blocks=N vregs=N gp_max=N fp_max=N edges=N
//! ```
//!
//! Where:
//! - `gp_max` / `fp_max` is the highest color index assigned per
//!   `RegClass` (so `gp_max+1` is the minimum #physical GP registers
//!   needed to avoid spilling).
//! - `edges` is the number of undirected interference edges.
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

    let liveness = liveness::compute_liveness(f);
    let ig = interference::build_interference(f, &liveness);
    let coloring = color::color(f, &ig);
    let gp_max = coloring.max_color(crate::cfg::RegClass::Gp);
    let fp_max = coloring.max_color(crate::cfg::RegClass::Fp);
    let edges: usize = ig.adj.values().map(|s| s.len()).sum::<usize>() / 2;

    let name = f.debug_name.as_deref().unwrap_or("<anonymous>");

    let spill_enabled = std::env::var("BEAGLE_SSA_REGALLOC_SPILL_STATS")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);

    if !spill_enabled {
        eprintln!(
            "[regalloc-stats] {} blocks={} vregs={} gp_max={} fp_max={} edges={}",
            name,
            f.num_blocks(),
            f.num_vregs(),
            gp_max,
            fp_max,
            edges,
        );
        return;
    }

    let mut spill_target = f.clone();
    let budget = Budget { gp: 24, fp: 32 };
    let result = allocate_with_spilling(&mut spill_target, budget);
    let fits = fits_budget(&result.coloring, budget);
    eprintln!(
        "[regalloc-stats] {} blocks={} vregs={} gp_max={} fp_max={} edges={} \
         spills_24_32={} fits_24_32={}",
        name,
        f.num_blocks(),
        f.num_vregs(),
        gp_max,
        fp_max,
        edges,
        result.spilled.len(),
        fits,
    );
}
