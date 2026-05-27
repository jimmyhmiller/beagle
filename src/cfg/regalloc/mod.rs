//! SSA-aware register allocation via chordal-graph coloring.
//!
//! This is the spec's **Phase 4 Option B** (Hack / Goos / Grund 2006).
//! In SSA form, the interference graph is chordal — meaning optimal
//! coloring is polynomial-time via greedy assignment in any perfect
//! elimination ordering (reverse dominator-tree DFS works). At any
//! program point, the set of live VRegs forms a clique in the
//! interference graph; the maximum clique size equals the chromatic
//! number, which equals the minimum register requirement before
//! spilling. No NP-hard heuristics needed.
//!
//! Pipeline (rolled out one sub-phase at a time):
//!
//! - **4a — Liveness** (`liveness.rs`, this commit). Per-block
//!   live-in / live-out via backward dataflow. Block params count as
//!   defs at block entry; terminator args count as uses at block exit.
//! - **4b — Interference graph.** Per-instruction live sets walked
//!   forward through each block; pairs of simultaneously-live VRegs
//!   form edges. SSA guarantees the resulting graph is chordal.
//! - **4c — Coloring.** Reverse-dominator-tree DFS gives a PEO;
//!   greedy color in PEO order yields optimal coloring per RegClass
//!   (GP / FP pools allocated independently per **I4**).
//! - **4d — Spilling.** When the live set at a program point exceeds
//!   the physical register pool, choose a spill candidate (e.g.
//!   highest-degree / lowest-use-density), insert SlotStore / SlotLoad
//!   to route the value through a stack slot, then re-color.
//! - **4e — Edge resolution.** (= the spec's Phase 5.) Block-param
//!   transfers become parallel copies on the edges before emit.

#![allow(dead_code)]

pub mod color;
pub mod interference;
pub mod liveness;
pub mod spill;
pub mod stats;
