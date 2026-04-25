//! Structured compiler dumps.
//!
//! Emits one NDJSON record per (function, stage) pair to a file
//! (`beagle-dump.jsonl` by default). Stages are AST, IR pre/post
//! register allocation, register allocation results, and final
//! disassembled machine code.
//!
//! The same `serde_json::Value` records produced here are intended
//! to be reused by a future runtime-callable API that returns
//! Beagle data structures instead of writing to disk.

use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use regex::Regex;
use serde_json::{Value, json};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Ast,
    IrPre,
    IrPost,
    Regalloc,
    Asm,
}

impl Stage {
    pub const ALL: [Stage; 5] = [
        Stage::Ast,
        Stage::IrPre,
        Stage::IrPost,
        Stage::Regalloc,
        Stage::Asm,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Ast => "ast",
            Stage::IrPre => "ir-pre",
            Stage::IrPost => "ir-post",
            Stage::Regalloc => "regalloc",
            Stage::Asm => "asm",
        }
    }

    pub fn parse(s: &str) -> Option<Stage> {
        match s {
            "ast" => Some(Stage::Ast),
            "ir-pre" => Some(Stage::IrPre),
            "ir-post" => Some(Stage::IrPost),
            "regalloc" => Some(Stage::Regalloc),
            "asm" => Some(Stage::Asm),
            _ => None,
        }
    }

    fn bit(self) -> u8 {
        1 << (self as u8)
    }
}

/// Parse a comma-separated stage list (e.g. `ast,ir-post,asm` or `all`).
pub fn parse_stage_list(spec: &str) -> Result<u8, String> {
    let mut bits = 0u8;
    for token in spec.split(',') {
        let t = token.trim();
        if t.is_empty() {
            continue;
        }
        if t == "all" {
            for s in Stage::ALL {
                bits |= s.bit();
            }
            continue;
        }
        match Stage::parse(t) {
            Some(stage) => bits |= stage.bit(),
            None => {
                return Err(format!(
                    "unknown dump stage '{t}'. expected one of: ast, ir-pre, ir-post, regalloc, asm, all"
                ));
            }
        }
    }
    Ok(bits)
}

pub struct DumpConfig {
    enabled_stages: u8,
    name_filter: Option<Regex>,
    sink: Option<Mutex<BufWriter<File>>>,
    /// Path that will be reported to the user once we know writes have happened.
    output_path: Option<PathBuf>,
}

impl std::fmt::Debug for DumpConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DumpConfig")
            .field("enabled_stages", &self.enabled_stages)
            .field(
                "name_filter",
                &self.name_filter.as_ref().map(|r| r.as_str()),
            )
            .field("output_path", &self.output_path)
            .finish()
    }
}

impl DumpConfig {
    /// A no-op config with all stages disabled. Used when `--dump` is absent.
    pub fn disabled() -> Arc<Self> {
        Arc::new(DumpConfig {
            enabled_stages: 0,
            name_filter: None,
            sink: None,
            output_path: None,
        })
    }

    /// Build a config from already-parsed CLI inputs. `stages_spec` is the raw
    /// `--dump` value (comma-separated), `filter` is an optional regex matched
    /// against the fully-qualified function name (`namespace/name`), and
    /// `output` overrides the default `beagle-dump.jsonl` location.
    pub fn from_args(
        stages_spec: &str,
        filter: Option<&str>,
        output: Option<&Path>,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let enabled_stages = parse_stage_list(stages_spec)?;
        if enabled_stages == 0 {
            return Ok(Self::disabled());
        }
        let path: PathBuf = match output {
            Some(p) => p.to_path_buf(),
            None => PathBuf::from("beagle-dump.jsonl"),
        };
        let file = File::create(&path)
            .map_err(|e| format!("failed to open dump output {}: {e}", path.display()))?;
        let name_filter = match filter {
            Some(pat) => Some(
                Regex::new(pat).map_err(|e| format!("invalid --dump-filter regex {pat:?}: {e}"))?,
            ),
            None => None,
        };
        Ok(Arc::new(DumpConfig {
            enabled_stages,
            name_filter,
            sink: Some(Mutex::new(BufWriter::new(file))),
            output_path: Some(path),
        }))
    }

    pub fn is_stage_enabled(&self, stage: Stage) -> bool {
        self.enabled_stages & stage.bit() != 0
    }

    pub fn matches_function(&self, fully_qualified: &str) -> bool {
        match &self.name_filter {
            Some(re) => re.is_match(fully_qualified),
            None => true,
        }
    }

    pub fn should_emit(&self, stage: Stage, fully_qualified: &str) -> bool {
        self.sink.is_some()
            && self.is_stage_enabled(stage)
            && self.matches_function(fully_qualified)
    }

    pub fn output_path(&self) -> Option<&Path> {
        self.output_path.as_deref()
    }

    /// Emit a single record. Failures to write are logged once and otherwise
    /// swallowed — a broken dump pipe must never fail the compilation.
    pub fn emit(&self, stage: Stage, function: &str, mut payload: Value) {
        let Some(sink) = &self.sink else { return };
        let obj = payload
            .as_object_mut()
            .expect("dump payload must be a JSON object");
        obj.insert("schema".to_string(), json!(SCHEMA_VERSION));
        obj.insert("stage".to_string(), json!(stage.name()));
        obj.insert("function".to_string(), json!(function));
        let line = serde_json::to_string(&payload).unwrap();
        let mut guard = match sink.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Err(e) = guard
            .write_all(line.as_bytes())
            .and_then(|_| guard.write_all(b"\n"))
        {
            eprintln!(
                "[dump] write failed (stage={}, fn={}): {}",
                stage.name(),
                function,
                e
            );
        }
    }

    /// Flush the buffered writer. Called at process shutdown by the runtime.
    pub fn flush(&self) {
        if let Some(sink) = &self.sink {
            if let Ok(mut g) = sink.lock() {
                let _ = g.flush();
            }
        }
    }
}

/// Disassemble a slice of native machine code starting at `address`. The
/// architecture is picked at compile time to match the active backend.
/// On disasm failure each instruction falls back to a `bytes`-only entry.
pub fn disassemble_machine_code(bytes: &[u8], address: u64) -> Vec<Value> {
    use capstone::prelude::*;

    cfg_if::cfg_if! {
        if #[cfg(any(feature = "backend-x86-64", all(target_arch = "x86_64", not(feature = "backend-arm64"))))] {
            let cs = Capstone::new()
                .x86()
                .mode(arch::x86::ArchMode::Mode64)
                .syntax(arch::x86::ArchSyntax::Intel)
                .build();
        } else {
            let cs = Capstone::new()
                .arm64()
                .mode(arch::arm64::ArchMode::Arm)
                .build();
        }
    }
    let cs = match cs {
        Ok(cs) => cs,
        Err(e) => {
            eprintln!("[dump] capstone init failed: {e}");
            return raw_bytes_only(bytes, address);
        }
    };
    match cs.disasm_all(bytes, address) {
        Ok(insns) => insns
            .iter()
            .map(|i| {
                let bytes_hex: String = i.bytes().iter().map(|b| format!("{b:02x}")).collect();
                json!({
                    "address": format!("0x{:x}", i.address()),
                    "offset": (i.address() as i64) - (address as i64),
                    "size": i.bytes().len(),
                    "bytes": bytes_hex,
                    "mnemonic": i.mnemonic().unwrap_or(""),
                    "operands": i.op_str().unwrap_or(""),
                })
            })
            .collect(),
        Err(e) => {
            eprintln!("[dump] disassembly failed at {address:#x}: {e}");
            raw_bytes_only(bytes, address)
        }
    }
}

fn raw_bytes_only(bytes: &[u8], address: u64) -> Vec<Value> {
    let bytes_hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    vec![json!({
        "address": format!("0x{:x}", address),
        "offset": 0,
        "size": bytes.len(),
        "bytes": bytes_hex,
    })]
}
