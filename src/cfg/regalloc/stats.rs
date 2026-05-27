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
use crate::cfg::regalloc::{color, interference, liveness};

/// Run the full Phase 4 analysis pipeline on `f` and emit one log
/// line to stderr if `BEAGLE_SSA_REGALLOC_STATS=1`. No-op otherwise.
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

    let name = f.debug_name.as_deref().unwrap_or("<anonymous>");
    // Each edge is stored twice (once per endpoint), so divide by 2.
    let edges: usize = ig.adj.values().map(|s| s.len()).sum::<usize>() / 2;
    eprintln!(
        "[regalloc-stats] {} blocks={} vregs={} gp_max={} fp_max={} edges={}",
        name,
        f.num_blocks(),
        f.num_vregs(),
        coloring.max_color(crate::cfg::RegClass::Gp),
        coloring.max_color(crate::cfg::RegClass::Fp),
        edges,
    );
}
