use crate::{
    CommandLineArguments,
    ast::{Ast, Pattern, TokenRange},
    backend::{Backend, CodegenBackend},
    code_memory::CodeAllocator,
    debug_only, get_runtime,
    ir::{StringValue, Value},
    parser::Parser,
    runtime::{Enum, Function, ProtocolMethodInfo, Struct},
    types::BuiltInTypes,
};

// These imports are only used in debug_only! blocks
#[cfg(debug_assertions)]
use crate::{Data, Message, builtins::debugger, pretty_print::PrettyPrint};

use mmap_rs::{MmapMut, MmapOptions};
use std::{
    collections::{HashMap, HashSet},
    env,
    error::Error,
    fmt,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Receiver, SyncSender},
    },
};

#[derive(Debug, Clone)]
pub enum CompileError {
    RegisterAllocation(String),
    LabelLookup {
        label: String,
    },
    StructResolution {
        struct_name: String,
    },
    PropertyCacheFull,
    MemoryMapping(String),
    ParseError(crate::parser::ParseError),
    FunctionNotFound {
        function_name: String,
    },
    InvalidFunctionPointer {
        function_name: String,
    },
    PathConversion {
        path: String,
    },
    // New semantic errors
    UndefinedVariable {
        name: String,
    },
    GlobalMutableVariable,
    BreakOutsideLoop,
    ContinueOutsideLoop,
    BindingNotFound {
        name: String,
    },
    EnumVariantNotFound {
        name: String,
    },
    StructFieldNotDefined {
        struct_name: String,
        field: String,
    },
    ExpectedIdentifier {
        got: String,
    },
    InvalidAssignment {
        reason: String,
    },
    InternalError {
        message: String,
    },
    NamespaceAliasNotFound {
        alias: String,
    },
    ArityMismatch {
        function_name: String,
        expected: usize,
        got: usize,
        is_variadic: bool,
    },
    MultiArityNoMatch {
        function_name: String,
        arg_count: usize,
        available_arities: Vec<usize>,
    },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::RegisterAllocation(msg) => {
                write!(f, "Register allocation error: {}", msg)
            }
            CompileError::LabelLookup { label } => write!(f, "Could not find label: {}", label),
            CompileError::StructResolution { struct_name } => {
                write!(
                    f,
                    "Cannot resolve struct '{}'. Note: struct definitions must be at the top level of a namespace, not inside functions.",
                    struct_name
                )
            }
            CompileError::PropertyCacheFull => write!(f, "Property look up cache is full"),
            CompileError::MemoryMapping(msg) => write!(f, "Memory mapping error: {}", msg),
            CompileError::ParseError(e) => write!(f, "Parse error: {}", e),
            CompileError::FunctionNotFound { function_name } => {
                write!(f, "Function not found: {}", function_name)?;
                // Check if this looks like subtraction written without spaces
                if let Some(dash_pos) = function_name.find('-') {
                    let left = &function_name[..dash_pos];
                    let right = &function_name[dash_pos + 1..];
                    if !left.is_empty()
                        && !right.is_empty()
                        && left.chars().next().is_some_and(|c| c.is_alphanumeric())
                        && right
                            .chars()
                            .next()
                            .is_some_and(|c| c.is_alphanumeric() || c == '-')
                    {
                        write!(
                            f,
                            "\n  hint: did you mean `{} - {}`? Subtraction requires spaces around `-`",
                            left,
                            &function_name[dash_pos + 1..]
                        )?;
                    }
                }
                Ok(())
            }
            CompileError::InvalidFunctionPointer { function_name } => {
                write!(f, "Invalid function pointer for: {}", function_name)
            }
            CompileError::PathConversion { path } => {
                write!(f, "Failed to convert path to string: {}", path)
            }
            CompileError::UndefinedVariable { name } => {
                write!(f, "Undefined variable: {}", name)?;
                // Check if this looks like subtraction written without spaces
                if let Some(dash_pos) = name.find('-') {
                    let left = &name[..dash_pos];
                    let right = &name[dash_pos + 1..];
                    if !left.is_empty()
                        && !right.is_empty()
                        && left.chars().next().is_some_and(|c| c.is_alphanumeric())
                        && right
                            .chars()
                            .next()
                            .is_some_and(|c| c.is_alphanumeric() || c == '-')
                    {
                        write!(
                            f,
                            "\n  hint: did you mean `{} - {}`? Subtraction requires spaces around `-`",
                            left,
                            &name[dash_pos + 1..]
                        )?;
                    }
                }
                Ok(())
            }
            CompileError::GlobalMutableVariable => {
                write!(f, "Cannot create mutable variable in global scope")
            }
            CompileError::BreakOutsideLoop => {
                write!(f, "break statement outside of loop")
            }
            CompileError::ContinueOutsideLoop => {
                write!(f, "continue statement outside of loop")
            }
            CompileError::BindingNotFound { name } => {
                write!(f, "Binding not found: {}", name)
            }
            CompileError::EnumVariantNotFound { name } => {
                write!(f, "Enum variant not found: {}", name)
            }
            CompileError::StructFieldNotDefined { struct_name, field } => {
                write!(
                    f,
                    "Struct field '{}' not defined on '{}'",
                    field, struct_name
                )
            }
            CompileError::ExpectedIdentifier { got } => {
                write!(f, "Expected identifier, got: {}", got)
            }
            CompileError::InvalidAssignment { reason } => {
                write!(f, "Invalid assignment: {}", reason)
            }
            CompileError::InternalError { message } => {
                write!(f, "Internal compiler error: {}", message)
            }
            CompileError::NamespaceAliasNotFound { alias } => {
                write!(f, "Namespace alias not found: {}", alias)
            }
            CompileError::ArityMismatch {
                function_name,
                expected,
                got,
                is_variadic,
            } => {
                if *is_variadic {
                    write!(
                        f,
                        "Function '{}' requires at least {} argument(s), but {} were provided",
                        function_name, expected, got
                    )
                } else {
                    write!(
                        f,
                        "Function '{}' expects {} argument(s), but {} were provided",
                        function_name, expected, got
                    )
                }
            }
            CompileError::MultiArityNoMatch {
                function_name,
                arg_count,
                available_arities,
            } => {
                let arities_str: Vec<String> =
                    available_arities.iter().map(|a| a.to_string()).collect();
                write!(
                    f,
                    "No arity of '{}' accepts {} argument(s). Available arities: {}",
                    function_name,
                    arg_count,
                    arities_str.join(", ")
                )
            }
        }
    }
}

impl Error for CompileError {}

impl From<crate::parser::ParseError> for CompileError {
    fn from(err: crate::parser::ParseError) -> Self {
        CompileError::ParseError(err)
    }
}

// ============================================================================
// Diagnostic System
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error = 0,
    Warning = 1,
    Info = 2,
    Hint = 3,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub kind: String,
    pub file_name: String,
    pub line: usize,
    pub column: usize,
    pub message: String,
    // Optional fields for specific diagnostic kinds
    pub enum_name: Option<String>,
    pub missing_variants: Option<Vec<String>>,
}

#[derive(Debug, Default)]
pub struct DiagnosticStore {
    diagnostics: std::collections::HashMap<String, Vec<Diagnostic>>,
}

impl DiagnosticStore {
    pub fn new() -> Self {
        Self {
            diagnostics: std::collections::HashMap::new(),
        }
    }

    /// Replace all diagnostics for a file (called after file compilation)
    pub fn set_file_diagnostics(&mut self, file: String, diags: Vec<Diagnostic>) {
        if diags.is_empty() {
            self.diagnostics.remove(&file);
        } else {
            self.diagnostics.insert(file, diags);
        }
    }

    /// Get all diagnostics across all files
    pub fn all(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics.values().flatten()
    }

    /// Get diagnostics for a specific file
    pub fn for_file(&self, file: &str) -> Option<&Vec<Diagnostic>> {
        self.diagnostics.get(file)
    }

    /// Get list of files that have diagnostics
    pub fn files(&self) -> impl Iterator<Item = &String> {
        self.diagnostics.keys()
    }

    /// Clear diagnostics for a specific file
    pub fn clear_file(&mut self, file: &str) {
        self.diagnostics.remove(file);
    }

    /// Clear all diagnostics
    pub fn clear_all(&mut self) {
        self.diagnostics.clear();
    }
}

/// Metadata about a multi-arity function
#[derive(Debug, Clone)]
pub struct MultiArityInfo {
    /// List of (fixed_arity, is_variadic) for each case
    pub arities: Vec<(usize, bool)>,
}

/// Holds compiled function data to be installed into the jump table later.
/// Used for batch installation: all functions in an eval are compiled first,
/// then their jump table entries are updated atomically, preventing partial
/// states where some functions are new but others are still old.
struct DeferredFunctionUpdate {
    name: Option<String>,
    pointer: *const u8,
    code_size: usize,
    max_locals: usize,
    number_of_args: usize,
    is_variadic: bool,
    min_args: usize,
    docstring: Option<String>,
    arg_names: Vec<String>,
    source_file: Option<String>,
    source_line: Option<usize>,
    source_text: Option<String>,
    disk_location: Option<crate::runtime::DiskLocation>,
}

/// Transient capture state for `build_osr_variant`. The per-function compile
/// fills `captured` with the target's specialized IR + loop info.
pub struct OsrCapture {
    pub target: String,
    pub captured: Option<(crate::ir::Ir, Vec<crate::osr::OsrLoopInfo>)>,
}

pub struct Compiler {
    pub code_allocator: CodeAllocator,
    pub property_look_up_cache: MmapMut,
    pub command_line_arguments: CommandLineArguments,
    pub pause_atom_ptr: Option<usize>,
    pub property_look_up_cache_offset: usize,
    /// Code-entry address of the function owning each property-cache
    /// entry, parallel to allocation order. 0 = pending (set later by
    /// `bind_property_cache` after `upsert_function` returns the
    /// function pointer). Used by tier-up to read a function's
    /// observed struct shapes and bake them into the recompile.
    pub property_cache_owners: Vec<usize>,
    /// Reverse index: code address → property-cache entry addresses
    /// (in source order). Each entry is the start address of the
    /// 24-byte slot whose layout is documented on
    /// `add_property_lookup`.
    pub property_cache_by_address: HashMap<usize, Vec<usize>>,
    /// Code address → contiguous `(start_byte_offset, end_byte_offset)`
    /// range in `property_look_up_cache` covering every slot allocated
    /// while compiling this function's body — including slots from any
    /// nested fns (closures / lambdas). Tier-up reads this transitive
    /// range as the feedback array so the recompile can bake offsets
    /// for nested-fn property accesses too. Per-function ownership for
    /// per-fn tier-up triggering still lives in
    /// `property_cache_by_address`.
    pub property_cache_range_by_address: HashMap<usize, (usize, usize)>,
    pub compiled_file_cache: HashSet<String>,
    pub diagnostic_store: Arc<Mutex<DiagnosticStore>>,
    /// Cache for protocol dispatch inline caching
    /// Layout per entry: [type_id (8 bytes), fn_ptr (8 bytes)]
    pub protocol_dispatch_cache: MmapMut,
    pub protocol_dispatch_cache_offset: usize,
    /// Type-feedback slots for arithmetic and comparison sites.
    /// Layout per entry: [bitfield (8 bytes)] using `crate::feedback` constants.
    /// Each `+ - * / % < > <= >= == !=` site owns one slot. The fast and slow
    /// paths OR observed-shape bits into their slot; bits are monotonic.
    pub arith_feedback_cache: MmapMut,
    pub arith_feedback_cache_offset: usize,
    /// Code-entry address of the compiled function owning each slot,
    /// parallel to the allocation order. 0 means "not yet bound" — slots
    /// are allocated during AST→IR before the entry pointer is known, then
    /// stamped via `bind_arith_feedback` once `upsert_function` returns.
    /// A slot whose owning function is later recompiled keeps its old
    /// address; the new compile allocates fresh slots tied to the new
    /// address. Tier-up reads slots filtered by the current entry pointer.
    pub arith_feedback_owners: Vec<usize>,
    /// Reverse index: code address → slot addresses (in source order).
    /// Populated by `bind_arith_feedback`.
    pub arith_feedback_by_address: HashMap<usize, Vec<usize>>,
    /// Code address → contiguous `(start_byte_offset, end_byte_offset)`
    /// range in `arith_feedback_cache` covering every slot allocated
    /// while compiling this function's body — including slots from
    /// nested fns. Symmetric to `property_cache_range_by_address` and
    /// used by tier-up so per-site specialization decisions stay in
    /// lockstep with allocation order even when nested fns appear
    /// between sites.
    pub arith_feedback_range_by_address: HashMap<usize, (usize, usize)>,
    /// Debug-only fully-qualified function name parallel to slots. Used
    /// by `--dump-arith-feedback` to give the dump a human-readable owner
    /// label. Tier-up logic should use the address, not the name.
    pub arith_feedback_debug_names: Vec<String>,
    /// Names of functions that have been recompiled to a specialized
    /// version at least once. Used by `specialize_function` to skip
    /// repeated work — the per-function entry counter trampoline can
    /// fire multiple times in races, and we don't want to recompile
    /// over and over.
    pub specialized_names: HashSet<String>,
    /// Fully-qualified names of functions whose stored source fragment
    /// came from an `extend ... with ...` method body. Beagle's grammar
    /// is context-sensitive there (reserved words like `perform` are
    /// valid method names inside extend blocks), so tier-up/OSR must
    /// re-parse these fragments with `FragmentContext::ExtendMethod` —
    /// a top-level re-parse would reject the name. Grows only; a stale
    /// entry merely re-parses a fragment permissively, which cannot
    /// change the meaning of valid code.
    pub extend_method_fragments: HashSet<String>,
    /// Set transiently by `build_osr_variant` so the per-function compile can
    /// snapshot the target function's specialized IR + loop info for OSR
    /// continuation construction. `None` outside an OSR build.
    pub osr_capture: Option<OsrCapture>,
    /// For each specialized function, the (pointer, size) of its tier-1 code
    /// *before* the tier-2 swap. On a runtime redefinition (eval/REPL/reload),
    /// `revert_all_specializations` restores these into the jump table so no
    /// specialized code keeps a stale snapshot of redefined state (struct
    /// layouts, name resolution, etc.). The generic code is retained (deopt
    /// relies on it), so the revert is just a jump-table-slot swap-back.
    pub specialization_originals: HashMap<String, (usize, usize)>,
    /// Per-function entry counters. Each compiled (named, non-bail,
    /// first-time) function gets one 8-byte slot. The slot is
    /// initialized to the negative of the threshold and incremented on
    /// every call; when it reaches zero the entry trampoline fires.
    /// Layout: parallel mmap, slots are 8-byte signed integers.
    pub function_counter_cache: MmapMut,
    pub function_counter_cache_offset: usize,
    /// Leaked C-strings, one per function with an entry counter,
    /// holding the function's fully-qualified name. The pointer is
    /// baked into the function's prologue; the trampoline reads it
    /// to know which name to specialize.
    pub function_counter_names: Vec<std::ffi::CString>,
    /// Multi-arity function metadata for static dispatch
    pub multi_arity_functions: HashMap<String, MultiArityInfo>,
    /// Dynamic variables: name -> (namespace_id, slot)
    pub dynamic_vars: HashMap<String, (usize, usize)>,
    /// When true, upsert_function defers jump table updates
    defer_function_installs: bool,
    /// Buffer of function updates to be applied atomically
    deferred_updates: Vec<DeferredFunctionUpdate>,
    /// Default value expressions for struct fields: fully_qualified_name -> [(field_index, default_ast)]
    pub struct_defaults: HashMap<String, Vec<(usize, Ast)>>,
    /// Structured-dump sink. Always present; a no-op when `--dump` is absent.
    pub dump: std::sync::Arc<crate::dump::DumpConfig>,
}

/// Force the single top-level `fn` definition in a freshly-parsed `program`
/// to be named `target`. Used by tier-up: a function's stored source slice may
/// parse to a different name than the (possibly compiler-generated) name it is
/// installed under — e.g. a protocol impl method stored as `Dog_speak` whose
/// source reads `fn speak(...)` — and the install keys on the parsed name. No-op
/// unless the program contains exactly one top-level `Ast::Function` (the common
/// single-arity case; multi-arity definitions are left untouched).
fn rename_sole_top_level_function(program: &mut crate::ast::Ast, target: &str) {
    use crate::ast::Ast;
    let Ast::Program { elements, .. } = program else {
        return;
    };
    // Exactly one top-level function, or we can't tell which one is `target`.
    let fn_count = elements
        .iter()
        .filter(|e| matches!(e, Ast::Function { .. }))
        .count();
    if fn_count != 1 {
        return;
    }
    if let Some(Ast::Function { name, .. }) = elements
        .iter_mut()
        .find(|e| matches!(e, Ast::Function { .. }))
    {
        if name.as_deref() != Some(target) {
            *name = Some(target.to_string());
        }
    }
}

/// Parse a stored source fragment under the syntactic context it was
/// originally defined in (see [`crate::parser::FragmentContext`]),
/// returning the parser (for its token maps) alongside the AST. Shared
/// by the tier-up and OSR recompile paths.
fn parse_fragment(
    source_file: &str,
    source_text: &str,
    context: crate::parser::FragmentContext,
) -> Result<(Parser, crate::ast::Ast), Box<dyn Error>> {
    let mut parser = Parser::new(source_file.to_string(), source_text.to_string())?;
    parser.set_fragment_context(context);
    let ast = parser.parse()?;
    Ok((parser, ast))
}

impl Compiler {
    pub fn get_pause_atom(&self) -> usize {
        self.pause_atom_ptr.unwrap_or(0)
    }

    pub fn set_pause_atom_ptr(&mut self, pointer: usize) {
        self.pause_atom_ptr = Some(pointer);
    }

    /// Run `f` with the current namespace switched to `ns_id`, restoring
    /// the previous namespace on EVERY exit — including `?` early returns
    /// inside `f`, which historically leaked the switch and poisoned all
    /// later evals on the (shared) compiler thread. Any code that needs a
    /// temporary namespace MUST go through this; never pair a bare
    /// `set_current_namespace` with a manual restore.
    fn with_namespace<R>(&mut self, ns_id: usize, f: impl FnOnce(&mut Self) -> R) -> R {
        let saved = self.current_namespace_id();
        self.set_current_namespace(ns_id);
        let result = f(self);
        self.set_current_namespace(saved);
        result
    }

    /// `with_namespace`, but `None` runs `f` in the current namespace.
    fn with_namespace_opt<R>(&mut self, ns_id: Option<usize>, f: impl FnOnce(&mut Self) -> R) -> R {
        match ns_id {
            Some(id) => self.with_namespace(id, f),
            None => f(self),
        }
    }

    /// Record that `full_name`'s stored source fragment is an
    /// extend-block method body, so later re-parses (tier-up, OSR) use
    /// `FragmentContext::ExtendMethod`. Called during `Ast::Extend`
    /// compilation for every method it registers.
    pub fn mark_extend_method_fragment(&mut self, full_name: String) {
        self.extend_method_fragments.insert(full_name);
    }

    /// The syntactic context `full_name`'s stored source fragment was
    /// originally defined in.
    fn fragment_context_for(&self, full_name: &str) -> crate::parser::FragmentContext {
        if self.extend_method_fragments.contains(full_name) {
            crate::parser::FragmentContext::ExtendMethod
        } else {
            crate::parser::FragmentContext::TopLevel
        }
    }

    pub fn register_dynamic_var(&mut self, name: String, namespace_id: usize, slot: usize) {
        self.dynamic_vars.insert(name, (namespace_id, slot));
    }

    pub fn lookup_dynamic_var(&self, name: &str) -> Option<(usize, usize)> {
        // First try direct lookup (for simple names like "out")
        if let Some(result) = self.dynamic_vars.get(name).copied() {
            return Some(result);
        }

        // If name contains "/", it's a qualified name like "core/out"
        if name.contains("/") {
            let parts: Vec<&str> = name.splitn(2, '/').collect();
            if parts.len() == 2 {
                let (namespace_alias, var_name) = (parts[0], parts[1]);

                // Resolve alias to actual namespace name (e.g., "core" -> "beagle.core")
                let namespace_name = self
                    .get_namespace_from_alias(namespace_alias)
                    .unwrap_or_else(|| namespace_alias.to_string());

                // Resolve namespace name to ID
                if let Some(namespace_id) = self.get_namespace_id(&namespace_name) {
                    // Look for dynamic var with simple name that belongs to this namespace
                    for (dyn_var_name, (ns_id, slot)) in &self.dynamic_vars {
                        if dyn_var_name == var_name && *ns_id == namespace_id {
                            return Some((*ns_id, *slot));
                        }
                    }
                }
            }
        }

        None
    }

    pub fn allocate_fn_pointer(&mut self) -> Result<usize, CompileError> {
        let allocate_fn = self
            .find_function("beagle.builtin/allocate")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.builtin/allocate".to_string(),
            })?;
        self.get_function_pointer(allocate_fn)
            .ok_or_else(|| CompileError::InvalidFunctionPointer {
                function_name: "beagle.builtin/allocate".to_string(),
            })
    }

    pub fn add_code(&mut self, code: &[u8]) -> Result<*const u8, Box<dyn Error>> {
        let new_pointer = self.code_allocator.write_bytes(code);
        Ok(new_pointer)
    }

    pub fn compile_string(&mut self, string: &str) -> Result<usize, Box<dyn Error>> {
        // The ending namespace is only of interest to the namespace-tracked
        // path (CompileStringInNamespace); plain string compiles discard it.
        self.compile_string_in_namespace(string, None)
            .map(|(pointer, _ending_namespace)| pointer)
    }

    /// Compile `string` as if it were a slice of an on-disk file at
    /// `byte_offset` / `line_offset`. The offsets are added to the byte
    /// spans and line numbers the parser produces, so the resulting
    /// `Function`/`Struct`/`Enum` records point into the real file rather
    /// than a standalone "repl" origin.
    ///
    /// Used by `beagle.reflect/write-source`: after splicing new text into
    /// a file and writing it, we re-eval just the edited fragment with
    /// the file's byte offset so the updated definition's `disk_location`
    /// lines up with its new position in the file on disk.
    pub fn compile_string_with_file_context(
        &mut self,
        string: &str,
        namespace: Option<&str>,
        file_name: &str,
        byte_offset: usize,
        line_offset: usize,
    ) -> Result<usize, Box<dyn Error>> {
        // Runtime recompile may redefine functions/structs/lets — drop any
        // tier-2 specializations so none keep a stale snapshot of the world.
        self.revert_all_specializations();

        // Parse BEFORE switching namespaces. Parsing is namespace-independent,
        // and a parse error propagating with `?` after the switch would leave
        // the compiler's current namespace permanently switched — poisoning
        // every later eval on the (shared) compiler thread.
        let mut parser = Parser::new(file_name.to_string(), string.to_string())?;
        let ast = parser.parse()?;

        let ns_id = match namespace {
            Some(ns_name) => match self.get_namespace_id(ns_name) {
                Some(id) => Some(id),
                None => return Err(format!("Namespace '{}' not found", ns_name).into()),
            },
            None => None,
        };

        let token_line_column_map: Vec<(usize, usize)> = parser
            .get_token_line_column_map()
            .into_iter()
            .map(|(line, col)| (line + line_offset, col))
            .collect();
        let token_byte_spans: Vec<(usize, usize)> = parser
            .get_token_byte_spans()
            .into_iter()
            .map(|(s, e)| (s + byte_offset, e + byte_offset))
            .collect();
        let definition_byte_ranges: HashMap<(usize, usize), (usize, usize)> = parser
            .get_definition_byte_ranges()
            .into_iter()
            .map(|(k, (s, e))| (k, (s + byte_offset, e + byte_offset)))
            .collect();
        // The source string we hand to the AST compiler needs a leading
        // padding of `byte_offset` bytes so that the shifted byte ranges
        // resolve correctly when we slice. The padding bytes are never
        // read for source text — `extract_source_text` only touches bytes
        // inside recorded definition ranges.
        let mut padded_source = String::with_capacity(byte_offset + string.len());
        padded_source.push_str(&" ".repeat(byte_offset));
        padded_source.push_str(string);

        let functions_before = {
            let runtime = get_runtime().get_mut();
            runtime.functions.len()
        };
        self.defer_function_installs = true;

        let compile_result = self.with_namespace_opt(ns_id, |c| {
            c.compile_ast(
                ast,
                None,
                file_name,
                token_line_column_map,
                padded_source,
                token_byte_spans,
                definition_byte_ranges,
            )
        });

        let (top_level, diagnostics) = match compile_result {
            Ok(result) => result,
            Err(e) => {
                self.deferred_updates.clear();
                self.defer_function_installs = false;
                let runtime = get_runtime().get_mut();
                runtime.functions.truncate(functions_before);
                return Err(e);
            }
        };

        if let Ok(mut store) = self.diagnostic_store.lock() {
            store.set_file_diagnostics(file_name.to_string(), diagnostics);
        }

        self.defer_function_installs = false;
        self.flush_deferred_functions();
        self.code_allocator.make_executable();
        if let Some(top_level) = top_level {
            let function = self.get_function_by_name(&top_level).ok_or_else(|| {
                CompileError::FunctionNotFound {
                    function_name: top_level.clone(),
                }
            })?;
            let function_pointer = self.get_pointer_for_function(function).ok_or({
                CompileError::InvalidFunctionPointer {
                    function_name: top_level,
                }
            })?;
            Ok(function_pointer)
        } else {
            Ok(0)
        }
    }

    /// Recompile a single function with prior arithmetic-feedback bits
    /// threaded into the AST→IR pass. Used by tier-up
    /// (`runtime/specialize-all`) to install a specialized version of a
    /// function whose feedback indicates monomorphic shapes — the
    /// recompile emits `*_with_bail` IR for those sites instead of
    /// `*_any`. The new code is registered via the same deferred-flush
    /// path used by source redefinition, so the jump-table swap is
    /// atomic with respect to other threads.
    ///
    /// Returns `Ok(true)` if the function was recompiled and swapped,
    /// `Ok(false)` if there's nothing to do (no source, no feedback, or
    /// not eligible — e.g. it's a `beagle.bail/...` helper, which we
    /// must not specialize to avoid recursion through the bail path).
    pub fn specialize_function(&mut self, full_name: &str) -> Result<bool, Box<dyn Error>> {
        // Don't specialize the bail helpers themselves — their slow path
        // is the bail call, and a specialized bail-of-a-bail would call
        // itself.
        if full_name.starts_with("beagle.bail/") {
            return Ok(false);
        }
        // Already specialized in a prior tick. Skip — re-specializing
        // produces identical code and just churns the jump table. A
        // future enhancement could compare current feedback to the
        // last-specialization snapshot and recompile when shapes have
        // genuinely changed.
        if self.specialized_names.contains(full_name) {
            return Ok(false);
        }

        // Resolve current pointer + source text.
        let runtime = get_runtime().get_mut();
        // Don't tier-2 specialize protocol dispatcher functions: they are
        // recompiled in place by `compile_protocol_method` on every impl
        // (re)definition, so an async tier-2 install of one races that recompile
        // for the same jump-table slot — letting a stale dispatcher version win
        // (observed as protocol-redefinition staleness under aggressive tier-up).
        // The dispatch happens through the (always-current) dispatch table at
        // runtime regardless, so specializing the glue gains nothing.
        if runtime.is_protocol_dispatcher(full_name) {
            return Ok(false);
        }
        let function = match runtime.find_function(full_name) {
            Some(f) => f,
            None => return Ok(false),
        };
        let source_text = match function.source_text.as_ref() {
            Some(s) if !s.is_empty() => s.clone(),
            _ => return Ok(false),
        };
        let source_file = function
            .source_file
            .clone()
            .unwrap_or_else(|| "<specialized>".to_string());
        let current_address: usize = function.pointer.into();
        // Tier-1 (pre-swap) code, retained so a later runtime redefinition can
        // revert this function out of its specialized snapshot.
        let tier1_size = function.size;
        // Use the transitive ranges so we also capture feedback for
        // sites inside nested fns (closures / lambdas). The cursor in
        // the recompile advances on every alloc, so the feedback array
        // must include every slot the original compile allocated — not
        // just those bound to this fn's pointer. Falls back to the
        // per-fn list when no range was recorded.
        let mut feedback_bits = self.arith_feedback_range_for_address(current_address);
        if feedback_bits.is_empty() {
            feedback_bits = self.arith_feedback_for_address(current_address);
        }
        let mut property_feedback = self.property_cache_range_for_address(current_address);
        if property_feedback.is_empty() {
            property_feedback = self.property_cache_for_address(current_address);
        }
        if feedback_bits.is_empty() && property_feedback.is_empty() {
            // No instrumented sites for the current compile — nothing to
            // specialize on.
            return Ok(false);
        }

        let (namespace, _) = match full_name.rsplit_once('/') {
            Some(parts) => parts,
            None => return Ok(false),
        };

        // Parse BEFORE switching namespaces — a parse error propagating with
        // `?` after the switch would leave the compiler's current namespace
        // permanently switched, poisoning every later eval on the (shared)
        // compiler thread.
        //
        // The fragment is re-parsed under the syntactic context it was
        // originally defined in: extend-block method bodies allow reserved
        // words (e.g. `perform`) as method names, which a top-level parse
        // would reject.
        let context = self.fragment_context_for(full_name);
        let (parser, mut ast) = parse_fragment(&source_file, &source_text, context)?;
        // The recompile MUST reinstall under `full_name`, but the install keys
        // on the name the source's `fn` header parses to — and for
        // compiler-generated functions those differ. A protocol impl method is
        // stored as `Dog_speak`, yet its source slice still reads
        // `fn speak(...)`, which reparses as the public dispatcher `speak`; a
        // tier-2 install of it would clobber the dispatcher's jump-table slot
        // with one type's impl body (observed as protocol-redefinition
        // mis-dispatch). Force the sole top-level `fn` to the name we're
        // specializing so the install lands on the right slot — and so impl
        // bodies still tier up (suppressing their source would skip tier-2).
        if let Some((_, unqualified)) = full_name.rsplit_once('/') {
            rename_sole_top_level_function(&mut ast, unqualified);
        }
        let token_line_column_map = parser.get_token_line_column_map();
        let token_byte_spans = parser.get_token_byte_spans();
        let definition_byte_ranges = parser.get_definition_byte_ranges();

        // Resolve the function's namespace so qualified-name resolution
        // works during the recompile; the switch goes through
        // `with_namespace` so the restore is unskippable.
        let ns_id = match self.get_namespace_id(namespace) {
            Some(id) => id,
            None => return Ok(false),
        };

        let functions_before = {
            let runtime = get_runtime().get_mut();
            runtime.functions.len()
        };
        self.defer_function_installs = true;

        // Mark this whole recompile (and every nested function compile it
        // drives) as a tier-up compile, so `Ir::compile` routes it through
        // SSA when `BEAGLE_SSA_TIER2` is set. First-compiles never enter
        // this guard, so they stay on legacy.
        let result = self.with_namespace(ns_id, |this| {
            let _tier_up = crate::ir::TierUpCompileGuard::enter();
            // Deopt: record the resident generic code address so the
            // specialized build can re-invoke it on a guard miss instead of
            // rejoining the bail result. Also pass __pause's address so the
            // rewrite can ignore the transparent entry safepoint call. Only
            // active under BEAGLE_SSA_DEOPT.
            let pause_addr = {
                let rt = get_runtime().get_mut();
                rt.get_function_by_name("beagle.builtin/__pause")
                    .cloned()
                    .and_then(|f| rt.get_pointer(&f).ok())
                    .map(|p| p as usize)
                    .unwrap_or(0)
            };
            let _deopt = crate::ir::DeoptContextGuard::enter(
                full_name.to_string(),
                current_address,
                pause_addr,
            );
            this.compile_ast_with_feedback(
                ast,
                None,
                &source_file,
                token_line_column_map,
                source_text,
                token_byte_spans,
                definition_byte_ranges,
                feedback_bits,
                property_feedback,
            )
        });

        match result {
            Ok(_) => {
                self.defer_function_installs = false;
                // Stage the install instead of swapping the jump table here:
                // this runs on the compiler thread while mutators read the
                // function table locklessly. A coordinator applies the staged
                // install inside a stop-the-world (see
                // `Runtime::stop_world_and_apply_installs`), driven right after
                // this returns.
                self.stage_specialization_installs();
                self.specialized_names.insert(full_name.to_string());
                // Remember the tier-1 code so a runtime redefinition can revert
                // this function (only the first specialization records it — the
                // `specialized_names` guard above prevents re-entry until revert).
                self.specialization_originals
                    .entry(full_name.to_string())
                    .or_insert((current_address, tier1_size));
                Ok(true)
            }
            Err(e) => {
                self.deferred_updates.clear();
                self.defer_function_installs = false;
                let runtime = get_runtime().get_mut();
                runtime.functions.truncate(functions_before);
                Err(e)
            }
        }
    }

    /// Build the optimized OSR continuation `F_osr` for the `loop_idx`-th loop
    /// of `full_name` and publish its callable address in the OSR registry.
    /// Runs on the compiler thread (driven by a `BuildOsrVariant` message).
    ///
    /// Strategy: feedback-recompile `full_name` (so its IR is *specialized*),
    /// snapshotting that IR + loop info via `osr_capture`; deconstruct it into
    /// `F_osr` (`crate::osr::build_osr_variant_ir`); compile `F_osr` through the
    /// SSA tier-2 path; register its raw code address. F's own tier-2 install is
    /// discarded — we only want the continuation. On any failure the loop is
    /// marked `Failed` so the back-edge never retries.
    pub fn build_osr_variant(&mut self, full_name: &str, loop_idx: usize) {
        let key = format!("{}#L{}", full_name, loop_idx);
        let dbg = std::env::var("BEAGLE_OSR_DEBUG").is_ok();
        if dbg {
            eprintln!("[OSR-build] start {}", key);
        }
        if self
            .build_osr_variant_inner(full_name, loop_idx, &key)
            .is_none()
        {
            if dbg {
                eprintln!("[OSR-build] FAILED {}", key);
            }
            crate::osr::osr_set_failed(&key);
        } else if dbg {
            eprintln!("[OSR-build] ready {}", key);
        }
    }

    fn build_osr_variant_inner(
        &mut self,
        full_name: &str,
        loop_idx: usize,
        key: &str,
    ) -> Option<()> {
        // --- Resolve source + feedback (mirrors `specialize_function`). ---
        let (source_text, source_file, current_address, f_arity, feedback_bits, property_feedback) = {
            let runtime = get_runtime().get_mut();
            let function = runtime.find_function(full_name)?;
            let source_text = match function.source_text.as_ref() {
                Some(s) if !s.is_empty() => s.clone(),
                _ => return None,
            };
            let source_file = function
                .source_file
                .clone()
                .unwrap_or_else(|| "<osr>".to_string());
            let current_address: usize = function.pointer.into();
            // F's real arity — the deopt re-invoke (built into F_osr below)
            // must call generic F with this many args, reloaded from slots
            // `0..f_arity`. F_osr's own entry param is the live-in buffer.
            let f_arity = function.number_of_args;
            let mut feedback_bits = self.arith_feedback_range_for_address(current_address);
            if feedback_bits.is_empty() {
                feedback_bits = self.arith_feedback_for_address(current_address);
            }
            let mut property_feedback = self.property_cache_range_for_address(current_address);
            if property_feedback.is_empty() {
                property_feedback = self.property_cache_for_address(current_address);
            }
            (
                source_text,
                source_file,
                current_address,
                f_arity,
                feedback_bits,
                property_feedback,
            )
        };

        // --- Resolve the function's namespace for name resolution. The
        // switch goes through `with_namespace` so the restore is
        // unskippable on every exit path. ---
        let (namespace, _) = full_name.rsplit_once('/')?;
        let ns_id = self.get_namespace_id(namespace)?;

        // --- Feedback-compile F, capturing its specialized IR + loop info. ---
        let source_file_for_compile = source_file.clone();
        let fragment_context = self.fragment_context_for(full_name);
        let capture = self.with_namespace(ns_id, |this| {
            this.osr_capture = Some(OsrCapture {
                target: full_name.to_string(),
                captured: None,
            });
            this.defer_function_installs = true;

            // Re-parse the fragment under its original syntactic context
            // (extend-block methods allow reserved words as names).
            let (parser, ast) =
                match parse_fragment(&source_file_for_compile, &source_text, fragment_context) {
                    Ok(pair) => pair,
                    Err(_) => {
                        this.osr_capture = None;
                        this.cleanup_after_osr_compile(None);
                        return None;
                    }
                };
            let token_line_column_map = parser.get_token_line_column_map();
            let token_byte_spans = parser.get_token_byte_spans();
            let definition_byte_ranges = parser.get_definition_byte_ranges();

            let result = {
                let _tier_up = crate::ir::TierUpCompileGuard::enter();
                let pause_addr = {
                    let rt = get_runtime().get_mut();
                    rt.get_function_by_name("beagle.builtin/__pause")
                        .cloned()
                        .and_then(|f| rt.get_pointer(&f).ok())
                        .map(|p| p as usize)
                        .unwrap_or(0)
                };
                let _deopt = crate::ir::DeoptContextGuard::enter(
                    full_name.to_string(),
                    current_address,
                    pause_addr,
                );
                this.compile_ast_with_feedback(
                    ast,
                    None,
                    &source_file_for_compile,
                    token_line_column_map,
                    source_text,
                    token_byte_spans,
                    definition_byte_ranges,
                    feedback_bits,
                    property_feedback,
                )
            };

            // Discard F's tier-2 install entirely — we only want the continuation.
            let capture = this.osr_capture.take();
            this.cleanup_after_osr_compile(Some(full_name));
            if result.is_err() {
                return None;
            }
            capture
        })?;

        let (base_ir, loops) = capture.captured?;
        let info = loops.get(loop_idx)?.clone();
        if std::env::var("BEAGLE_OSR_DEBUG").is_ok() {
            eprintln!(
                "[OSR-build] captured ir ({} insns), {} loops; loop {} header={:?} live_ins={:?}",
                base_ir.instructions.len(),
                loops.len(),
                loop_idx,
                info.header_label,
                info.live_in_slots
            );
        }

        // --- Deconstruct + compile F_osr. ---
        let osr_deopt = std::env::var("BEAGLE_OSR_DEOPT")
            .map(|v| v != "0")
            .unwrap_or(true);
        let pause_addr = {
            let rt = get_runtime().get_mut();
            rt.get_function_by_name("beagle.builtin/__pause")
                .cloned()
                .and_then(|f| rt.get_pointer(&f).ok())
                .map(|p| p as usize)
                .unwrap_or(0)
        };
        // Classify which non-param live-in slots F treats as ints. Analyzing a
        // *deopt-applied* CFG of F (merge-back removed) makes `pointer_class`
        // report the loop-carried int accumulators as non-pointer. We guard
        // exactly those at the OSR entry so the loop-header φ is known-int and
        // guard-elim drops the loop's redundant int guards. Param slots
        // (`0..f_arity`) are excluded: their stores must stay in the entry block
        // for deopt eligibility, and warm leaves params slot-resident anyway.
        let int_slots: std::collections::HashSet<usize> = if osr_deopt {
            crate::cfg::builder::build_cfg_for_int_analysis(
                &base_ir,
                current_address,
                pause_addr,
                f_arity,
            )
            .filter(|(_, deopt_applied)| *deopt_applied)
            .map(|(cfg, _)| {
                let pc = crate::cfg::pointer_class::analyze(&cfg);
                info.live_in_slots
                    .iter()
                    .copied()
                    .filter(|s| {
                        *s >= f_arity
                            && pc
                                .non_pointer_slots
                                .contains(&crate::cfg::SlotId(*s as u32))
                    })
                    .collect()
            })
            .unwrap_or_default()
        } else {
            std::collections::HashSet::new()
        };
        if std::env::var("BEAGLE_OSR_DEBUG").is_ok() {
            eprintln!("[OSR-build] int live-in slots to guard: {:?}", int_slots);
        }
        let mut f_osr_ir = crate::osr::build_osr_variant_ir(&base_ir, &info, &int_slots);
        let f_osr_name = format!("{}$osr{}", full_name, loop_idx);
        f_osr_ir.debug_name = Some(f_osr_name.clone());
        let error_fn_pointer = f_osr_ir.error_fn_pointer;
        let dump = self.dump.clone();
        // Compile F_osr under a deopt context keyed on *its* name, re-invoking
        // generic F (`current_address`, the resident tier-1 code) with F's real
        // arity (see the int-slot analysis above for the rationale).
        let mut backend = {
            let _tier_up = crate::ir::TierUpCompileGuard::enter();
            let _deopt = osr_deopt.then(|| {
                crate::ir::DeoptContextGuard::enter_with_nargs(
                    f_osr_name.clone(),
                    current_address,
                    pause_addr,
                    Some(f_arity),
                )
            });
            let backend = Backend::new();
            f_osr_ir.compile(backend, error_fn_pointer, &dump)
        };
        let code = backend.compile_to_bytes();
        let addr = self.add_code(&code).ok()?;
        self.code_allocator.make_executable();

        crate::osr::osr_set_ready(key, addr as usize, info.live_in_slots);
        Some(())
    }

    /// Tear down the transient state of an OSR capture-compile.
    ///
    /// `target` = `Some(full_name)` on the success path, `None` on an
    /// error/abort path.
    ///
    /// The capture-compile runs with `defer_function_installs = true`, so it
    /// never pushes to `runtime.functions` directly — every compiled function
    /// (the OSR target F's recompile *and* any inner closures / nested
    /// functions in F's body) lands in `deferred_updates` instead. F's body
    /// closures are compiled to FRESH code addresses and those addresses are
    /// baked into the captured IR (hence into F_osr's `make_closure` ops). So
    /// on success we MUST register those inner functions — otherwise F_osr
    /// calls `make_closure(<unregistered address>)` and throws "Function not
    /// found when creating closure" (the bug that hung beagle-zelda: any OSR'd
    /// loop that creates a closure). We discard only F's own recompile (we
    /// publish F_osr, not F's tier-2). Registration goes through the staged +
    /// stop-the-world install path so it can't race mutator table reads, and
    /// happens before F_osr is built/published.
    fn cleanup_after_osr_compile(&mut self, target: Option<&str>) {
        self.defer_function_installs = false;
        match target {
            Some(full_name) => {
                self.deferred_updates
                    .retain(|u| u.name.as_deref() != Some(full_name));
                // Register the retained inner functions directly (append-only,
                // preserving the "functions only ever grows" invariant
                // `get_function_by_pointer` relies on). NOT via the
                // stop-the-world install path: this runs on the compiler
                // thread, and a STW there deadlocks (the compiler thread must
                // stay free to service the park messages of the mutators it is
                // stopping — which is why `tier_up_trampoline` runs its STW on a
                // spawn thread). `flush_deferred_functions` runs *before* F_osr
                // is built/published below, so F_osr's `make_closure` ops always
                // resolve to a registered function.
                self.flush_deferred_functions();
            }
            None => {
                self.deferred_updates.clear();
            }
        }
    }

    /// Compile `string` in `namespace`. Returns `(function_pointer,
    /// ending_namespace)`, where `ending_namespace` is the namespace the
    /// compiler is in AFTER processing the code — i.e. the namespace any
    /// trailing `namespace X` directive switched into (the first pass sets the
    /// current namespace as it sees those directives). Returning it lets a
    /// caller (e.g. a REPL session) track its own current namespace WITHOUT
    /// reading the shared global current-namespace, which other sessions/threads
    /// mutate. This is the readback that makes per-session namespaces correct.
    pub fn compile_string_in_namespace(
        &mut self,
        string: &str,
        namespace: Option<&str>,
    ) -> Result<(usize, String), Box<dyn Error>> {
        // Runtime recompile may redefine functions/structs/lets — drop any
        // tier-2 specializations so none keep a stale snapshot of the world.
        self.revert_all_specializations();

        // Parse BEFORE switching namespaces. Parsing is namespace-independent,
        // and a parse error propagating with `?` after the switch would leave
        // the compiler's current namespace permanently switched — poisoning
        // every later eval on the (shared) compiler thread.
        //
        // For REPL/eval, we use "repl" as the file name for diagnostics.
        // Each eval replaces the previous "repl" diagnostics.
        let mut parser = Parser::new("".to_string(), string.to_string())?;
        let ast = parser.parse()?;

        // Resolve the target namespace; the switch itself happens through
        // `with_namespace_opt` below so the restore is unskippable.
        let ns_id = match namespace {
            Some(ns_name) => match self.get_namespace_id(ns_name) {
                Some(id) => Some(id),
                None => return Err(format!("Namespace '{}' not found", ns_name).into()),
            },
            None => None,
        };
        let token_line_column_map = parser.get_token_line_column_map();
        let token_byte_spans = parser.get_token_byte_spans();
        let source_text = parser.source().to_string();
        let definition_byte_ranges = parser.get_definition_byte_ranges();

        // Track function count before compilation so we can clean up on failure.
        // The first pass reserves functions (with null pointers) before compiling
        // bodies. If body compilation fails, those reserved functions would remain
        // with null code pointers, causing segfaults if called later.
        let functions_before = {
            let runtime = get_runtime().get_mut();
            runtime.functions.len()
        };

        // Enable deferred function installation: compile all functions in this
        // eval, but don't update their jump table entries until the end.
        // This prevents partial states where some functions are new (expecting
        // new struct layout) while others are still old (producing old-layout objects).
        self.defer_function_installs = true;

        let (compile_result, ending_namespace) = self.with_namespace_opt(ns_id, |c| {
            let result = c.compile_ast(
                ast,
                Some("REPL_FUNCTION".to_string()),
                "repl",
                token_line_column_map,
                source_text,
                token_byte_spans,
                definition_byte_ranges,
            );
            // Capture the namespace the first pass switched into (a trailing
            // `namespace X` directive) BEFORE `with_namespace_opt` restores the
            // caller's namespace. With no directive this is just the namespace
            // we compiled in.
            (result, c.current_namespace_name())
        });

        let (top_level, diagnostics) = match compile_result {
            Ok(result) => result,
            Err(e) => {
                // Discard any deferred updates from the failed compilation
                self.deferred_updates.clear();
                self.defer_function_installs = false;
                // Remove functions that were reserved during this failed compilation
                let runtime = get_runtime().get_mut();
                runtime.functions.truncate(functions_before);
                return Err(e);
            }
        };

        // Store diagnostics for "repl" (replaces previous REPL diagnostics)
        if let Ok(mut store) = self.diagnostic_store.lock() {
            store.set_file_diagnostics("repl".to_string(), diagnostics);
        }

        // Flush all deferred function updates: make code executable and
        // update all jump table entries atomically.
        self.defer_function_installs = false;
        self.flush_deferred_functions();
        self.code_allocator.make_executable();
        // Reset protocol-dispatch inline caches AGAIN now that the new impls
        // and their dispatch-table entries are installed. The reset inside
        // `revert_all_specializations` ran before this compile, so a dispatch
        // in the gap could have re-cached the OLD target; this final reset
        // guarantees the next dispatch re-resolves through the updated table.
        self.invalidate_all_protocol_dispatch_caches();
        if let Some(top_level) = top_level {
            let function = self.get_function_by_name(&top_level).ok_or_else(|| {
                CompileError::FunctionNotFound {
                    function_name: top_level.clone(),
                }
            })?;
            let function_pointer = self.get_pointer_for_function(function).ok_or({
                CompileError::InvalidFunctionPointer {
                    function_name: top_level,
                }
            })?;
            Ok((function_pointer, ending_namespace))
        } else {
            Ok((0, ending_namespace))
        }
    }

    // TODO: I'm going to need to change how this works at some point.
    // I want to be able to dynamically run these namespaces
    // not have this awkward compile returns top level names thing
    pub fn compile(&mut self, file_name: &str) -> Result<Vec<String>, Box<dyn Error>> {
        // Canonicalize path for cache consistency
        let canonical_path = std::fs::canonicalize(file_name)
            .unwrap_or_else(|_| std::path::PathBuf::from(file_name));
        let canonical_str = canonical_path
            .to_str()
            .ok_or_else(|| CompileError::PathConversion {
                path: format!("{:?}", canonical_path),
            })?
            .to_string();

        if self.compiled_file_cache.contains(&canonical_str) {
            if self.command_line_arguments.verbose {
                println!("Already compiled {:?}", file_name);
            }
            return Ok(vec![]);
        }

        // Note: We no longer clear diagnostics here. The new diagnostic system
        // stores diagnostics per-file and replaces them when a file is recompiled.
        if self.command_line_arguments.verbose {
            println!("Compiling {:?}", file_name);
        }

        let parse_time = std::time::Instant::now();
        let code = std::fs::read_to_string(file_name)?;
        let mut parser = Parser::new(file_name.to_string(), code.to_string())?;
        let ast = parser.parse()?;
        let token_line_column_map = parser.get_token_line_column_map();
        let token_byte_spans = parser.get_token_byte_spans();
        let source_text = parser.source().to_string();
        let definition_byte_ranges = parser.get_definition_byte_ranges();

        if self.command_line_arguments.print_parse {
            println!("{:#?}", ast);
        }

        if self.command_line_arguments.print_ast {
            println!("{:#?}", ast);
        }

        if self.command_line_arguments.show_times {
            println!("Parse time: {:?}", parse_time.elapsed());
        }

        let mut top_levels_to_run = self.compile_dependencies(&ast, file_name)?;

        let (top_level, diagnostics) = self.compile_ast(
            ast,
            None,
            file_name,
            token_line_column_map,
            source_text,
            token_byte_spans,
            definition_byte_ranges,
        )?;
        if let Some(top_level) = top_level {
            top_levels_to_run.push(top_level);
        }

        // Store diagnostics for this file (replaces any existing diagnostics for this file)
        if let Ok(mut store) = self.diagnostic_store.lock() {
            store.set_file_diagnostics(canonical_str.clone(), diagnostics);
        }

        if self.command_line_arguments.verbose {
            println!("Done compiling {:?}", file_name);
        }
        self.code_allocator.make_executable();
        self.compiled_file_cache.insert(canonical_str);
        Ok(top_levels_to_run)
    }

    pub fn compile_source(
        &mut self,
        name: &str,
        source: &str,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        if self.compiled_file_cache.contains(name) {
            if self.command_line_arguments.verbose {
                println!("Already compiled {:?}", name);
            }
            return Ok(vec![]);
        }

        if self.command_line_arguments.verbose {
            println!("Compiling {:?} (from embedded source)", name);
        }

        let parse_time = std::time::Instant::now();
        let mut parser = Parser::new(name.to_string(), source.to_string())?;
        let ast = parser.parse()?;
        let token_line_column_map = parser.get_token_line_column_map();
        let token_byte_spans = parser.get_token_byte_spans();
        let source_text = parser.source().to_string();
        let definition_byte_ranges = parser.get_definition_byte_ranges();

        if self.command_line_arguments.print_parse {
            println!("{:#?}", ast);
        }

        if self.command_line_arguments.print_ast {
            println!("{:#?}", ast);
        }

        if self.command_line_arguments.show_times {
            println!("Parse time: {:?}", parse_time.elapsed());
        }

        let mut top_levels_to_run = self.compile_dependencies(&ast, name)?;

        let (top_level, diagnostics) = self.compile_ast(
            ast,
            None,
            name,
            token_line_column_map,
            source_text,
            token_byte_spans,
            definition_byte_ranges,
        )?;
        if let Some(top_level) = top_level {
            top_levels_to_run.push(top_level);
        }

        if let Ok(mut store) = self.diagnostic_store.lock() {
            store.set_file_diagnostics(name.to_string(), diagnostics);
        }

        if self.command_line_arguments.verbose {
            println!("Done compiling {:?}", name);
        }
        self.code_allocator.make_executable();
        self.compiled_file_cache.insert(name.to_string());
        Ok(top_levels_to_run)
    }

    pub fn compile_dependencies(
        &mut self,
        ast: &crate::ast::Ast,
        source_file: &str,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let mut top_levels_to_run = vec![];

        for use_stmt in ast.uses() {
            let (namespace_name, _alias) = self.extract_use(&use_stmt)?;
            // Skip namespaces already registered in the runtime (e.g. beagle.primitive,
            // beagle.builtin, beagle.core, etc.) that were loaded by runtime init or
            // load_default_files before we get here. This also catches namespaces compiled
            // via compile_source (which caches under "name.bg" rather than canonical path),
            // since any compiled source registers its namespace in the runtime.
            if self.namespace_exists(&namespace_name) {
                continue;
            }
            match self.resolve_namespace_to_file(&namespace_name, source_file)? {
                Some(file_name) => {
                    let top_level = self.compile(&file_name)?;
                    top_levels_to_run.extend(top_level);
                }
                None => {
                    // Not found on disk — try embedded stdlib.
                    let embedded_name = format!("{}.bg", namespace_name);
                    if let Some(source) = crate::embedded_stdlib::get(&embedded_name) {
                        let top_level = self.compile_source(&embedded_name, source)?;
                        top_levels_to_run.extend(top_level);
                    } else {
                        return Err(format!(
                            "Cannot resolve namespace '{}': not found as a file, in embedded stdlib, or as a runtime namespace",
                            namespace_name
                        ).into());
                    }
                }
            }
        }

        Ok(top_levels_to_run)
    }

    pub fn compile_ast(
        &mut self,
        ast: crate::ast::Ast,
        fn_name: Option<String>,
        file_name: &str,
        token_line_column_map: Vec<(usize, usize)>,
        source_text: String,
        token_byte_spans: Vec<(usize, usize)>,
        definition_byte_ranges: HashMap<(usize, usize), (usize, usize)>,
    ) -> Result<(Option<String>, Vec<Diagnostic>), Box<dyn Error>> {
        self.compile_ast_with_feedback(
            ast,
            fn_name,
            file_name,
            token_line_column_map,
            source_text,
            token_byte_spans,
            definition_byte_ranges,
            Vec::new(),
            Vec::new(),
        )
    }

    /// Like `compile_ast`, but threads `feedback_bits` (in source-visit
    /// order) into the AST→IR pass so monomorphic arithmetic sites get
    /// emitted with `*_with_bail` specialization. `property_feedback`
    /// plays the analogous role for `Ast::PropertyAccess`: each entry
    /// is the prior `(struct_id_versioned, field_offset_bytes,
    /// is_mutable)` triple from the IC slot, in source order. Used by
    /// tier-up.
    // Mirrors the parameter list of `compile_ast` plus the two
    // feedback inputs; bundling them into a struct would just push
    // the noise to the call sites.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_ast_with_feedback(
        &mut self,
        ast: crate::ast::Ast,
        fn_name: Option<String>,
        file_name: &str,
        token_line_column_map: Vec<(usize, usize)>,
        source_text: String,
        token_byte_spans: Vec<(usize, usize)>,
        definition_byte_ranges: HashMap<(usize, usize), (usize, usize)>,
        feedback_bits: Vec<u64>,
        property_feedback: Vec<(u64, u64, u64)>,
    ) -> Result<(Option<String>, Vec<Diagnostic>), Box<dyn Error>> {
        let (mut ir, token_map, diagnostics, top_level_arith_slots, top_level_property_slots) = ast
            .compile_with_feedback(
                self,
                file_name,
                token_line_column_map,
                source_text,
                token_byte_spans,
                definition_byte_ranges,
                feedback_bits,
                property_feedback,
            )?;
        let top_level_name =
            fn_name.unwrap_or_else(|| format!("{}/__top_level", self.current_namespace_name()));
        if ast.has_top_level() {
            let backend = Backend::new();
            let error_fn = self
                .find_function("beagle.builtin/throw-type-error")
                .ok_or_else(|| CompileError::FunctionNotFound {
                    function_name: "beagle.builtin/throw-type-error".to_string(),
                })?;
            let error_fn_pointer = self.get_function_pointer(error_fn).ok_or_else(|| {
                CompileError::InvalidFunctionPointer {
                    function_name: "beagle.builtin/throw-type-error".to_string(),
                }
            })?;

            ir.ir_range_to_token_range = token_map.clone();
            ir.debug_name = Some(top_level_name.clone());
            // SSA-pipeline coverage probe: build a CfgFunction from the
            // finalized legacy IR and run the verifier. No-op unless
            // BEAGLE_SSA_VERIFY=1; failures go to stderr and do not
            // affect the legacy lowering below.
            let _ = crate::cfg::builder::try_build_and_verify(&ir);
            let dump = self.dump.clone();
            let mut backend = ir.compile(backend, error_fn_pointer, &dump);
            let _token_map = ir.ir_range_to_token_range.clone();
            let max_locals = backend.max_locals() as usize;
            let _function_pointer = self.upsert_function(
                Some(&top_level_name),
                &mut backend,
                max_locals,
                0,
                false,
                0,
                None,
                vec![],
                Some(file_name.to_string()),
                None,
                None,
                None,
            )?;
            // Stamp the top-level body's feedback slots with its entry
            // address now that we have it. Nested fn bodies were bound
            // during AST compilation.
            self.bind_arith_feedback(&top_level_arith_slots, _function_pointer);
            self.bind_property_cache(&top_level_property_slots, _function_pointer);
            debug_only! {
                debugger(Message {
                    kind: "ir".to_string(),
                    data: Data::Ir {
                        function_pointer: _function_pointer,
                        file_name: file_name.to_string(),
                        instructions: ir.instructions.iter().map(|x| x.pretty_print()).collect(),
                        token_range_to_ir_range: _token_map
                            .iter()
                            .map(|(token, ir)| ((token.start, token.end), (ir.start, ir.end)))
                            .collect(),
                    },
                });

                let pretty_arm_instructions =
                    backend.instructions_mut().iter().map(|x| x.pretty_print()).collect();
                let ir_to_machine_code_range = ir
                    .ir_to_machine_code_range
                    .iter()
                    .map(|(ir, machine_range)| (*ir, (machine_range.start, machine_range.end)))
                    .collect();

                debugger(crate::Message {
                    kind: "asm".to_string(),
                    data: Data::Arm {
                        function_pointer: _function_pointer,
                        file_name: file_name.to_string(),
                        instructions: pretty_arm_instructions,
                        ir_to_machine_code_range,
                    },
                });
            }

            return Ok((Some(top_level_name), diagnostics));
        }
        Ok((None, diagnostics))
    }

    pub fn add_string(&mut self, string_value: StringValue) -> Value {
        let runtime = get_runtime().get_mut();
        let offset = runtime.add_string(string_value.clone());
        Value::StringConstantPtr(offset)
    }

    pub fn add_keyword(&mut self, keyword_text: String) -> Value {
        let runtime = get_runtime().get_mut();
        let offset = runtime.add_keyword(keyword_text);
        Value::KeywordConstantPtr(offset)
    }

    pub fn add_property_lookup(&mut self) -> Result<usize, CompileError> {
        if self.property_look_up_cache_offset >= self.property_look_up_cache.len() {
            return Err(CompileError::PropertyCacheFull);
        }
        let location = unsafe {
            self.property_look_up_cache
                .as_ptr()
                .add(self.property_look_up_cache_offset) as usize
        };
        // Cache layout: [struct_id (8 bytes), field_offset (8 bytes), is_mutable (8 bytes)]
        // Initialize struct_id with sentinel value that will never match a real struct_id.
        // This ensures the first access always goes to the slow path.
        // (mmap is zero-initialized, so struct_id=0 would falsely match the first registered struct.)
        unsafe {
            (*(location as *const AtomicUsize)).store(usize::MAX, Ordering::Release);
        }
        self.property_look_up_cache_offset += 3 * 8;
        // Owner address is unknown at allocation time; tier-up binds it
        // when `upsert_function` returns, via `bind_property_cache`.
        self.property_cache_owners.push(0);
        Ok(location)
    }

    /// Allocate a 16-byte inline cache entry for protocol dispatch
    /// Layout: [type_id (8 bytes), fn_ptr (8 bytes)]
    pub fn add_protocol_dispatch_cache(&mut self) -> Result<usize, CompileError> {
        if self.protocol_dispatch_cache_offset >= self.protocol_dispatch_cache.len() {
            return Err(CompileError::PropertyCacheFull); // Reuse error type for now
        }
        let location = unsafe {
            self.protocol_dispatch_cache
                .as_ptr()
                .add(self.protocol_dispatch_cache_offset) as usize
        };
        // Initialize type_id with sentinel value that will never match a real type_id
        // This ensures the first call always goes to slow path (struct_id=0 would otherwise match)
        unsafe {
            let cache_ptr = location as *mut usize;
            *cache_ptr = usize::MAX; // Sentinel: impossible type_id
        }
        // 16 bytes: type_id (8) + fn_ptr (8)
        self.protocol_dispatch_cache_offset += 2 * 8;
        Ok(location)
    }

    /// Allocate an 8-byte type-feedback slot for an arithmetic or comparison
    /// site. The slot is zero-initialized (mmap default), which represents
    /// "no observations yet". Returns the absolute address of the slot.
    ///
    /// The owning code-address is left as 0 ("pending"); the AST compiler
    /// is expected to call `bind_arith_feedback` once `upsert_function`
    /// returns the function's entry pointer. `debug_name` is recorded as a
    /// human-readable label for `--dump-arith-feedback`; tier-up logic
    /// should look up by address, not by name.
    pub fn add_arith_feedback(&mut self, debug_name: String) -> Result<usize, CompileError> {
        if self.arith_feedback_cache_offset + 8 > self.arith_feedback_cache.len() {
            return Err(CompileError::PropertyCacheFull);
        }
        let location = unsafe {
            self.arith_feedback_cache
                .as_ptr()
                .add(self.arith_feedback_cache_offset) as usize
        };
        self.arith_feedback_cache_offset += 8;
        self.arith_feedback_owners.push(0);
        self.arith_feedback_debug_names.push(debug_name);
        Ok(location)
    }

    /// Stamp a code address onto every slot in `slots`, and add them to
    /// the per-address index. Called by the AST compiler after
    /// `upsert_function` returns the entry pointer for the function whose
    /// body just allocated those slots. Slots already bound to a previous
    /// address are left alone (defensive — should never happen).
    pub fn bind_arith_feedback(&mut self, slots: &[usize], code_address: usize) {
        if code_address == 0 || slots.is_empty() {
            return;
        }
        let base = self.arith_feedback_cache.as_ptr() as usize;
        for &slot_addr in slots {
            let index = (slot_addr - base) / 8;
            if let Some(owner) = self.arith_feedback_owners.get_mut(index)
                && *owner == 0
            {
                *owner = code_address;
            }
        }
        self.arith_feedback_by_address
            .entry(code_address)
            .or_default()
            .extend_from_slice(slots);
    }

    /// Allocate an 8-byte entry-counter slot for a freshly compiled
    /// function. Initializes the slot to `-threshold` so an
    /// inline `subs ; cbnz` decrement-and-branch fires when the
    /// function has been called `threshold` times. Returns
    /// `(slot_address, name_c_str_ptr)` — both meant to be baked into
    /// the function prologue's tier-up check. The C-string is leaked
    /// into `function_counter_names` so its pointer stays valid for
    /// the program's lifetime.
    pub fn add_function_counter(
        &mut self,
        function_name: &str,
        threshold: i64,
    ) -> Result<(usize, usize), CompileError> {
        if self.function_counter_cache_offset + 8 > self.function_counter_cache.len() {
            return Err(CompileError::PropertyCacheFull);
        }
        let slot = unsafe {
            self.function_counter_cache
                .as_ptr()
                .add(self.function_counter_cache_offset) as usize
        };
        unsafe {
            // Counter starts at +threshold and is decremented on each
            // call. The trampoline fires when it reaches zero.
            *(slot as *mut i64) = threshold;
        }
        self.function_counter_cache_offset += 8;
        let cstring = std::ffi::CString::new(function_name).unwrap_or_default();
        let name_ptr = cstring.as_ptr() as usize;
        self.function_counter_names.push(cstring);
        Ok((slot, name_ptr))
    }

    /// Iterate over all allocated arithmetic-feedback slots, yielding
    /// `(slot_address, slot_value, owning_code_address, debug_name)`
    /// tuples in allocation order. Used by `--dump-arith-feedback`.
    /// `owning_code_address` is 0 for slots whose function compile never
    /// reached the bind step (should be rare and indicates a bug).
    pub fn iter_arith_feedback(&self) -> impl Iterator<Item = (usize, u64, usize, &str)> + '_ {
        let base = self.arith_feedback_cache.as_ptr() as usize;
        let count = self.arith_feedback_cache_offset / 8;
        (0..count).map(move |i| {
            let addr = base + i * 8;
            let value = unsafe { *(addr as *const u64) };
            let owner_addr = self.arith_feedback_owners.get(i).copied().unwrap_or(0);
            let name = self
                .arith_feedback_debug_names
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or("<unknown>");
            (addr, value, owner_addr, name)
        })
    }

    /// Read every feedback slot bound to the given code entry address.
    /// Returns slot values in source order. Tier-up logic asks
    /// `runtime.get_function(name).pointer` and passes it here to get the
    /// feedback for the *currently installed* compile; older compiles'
    /// slots stay associated with their original (now-stale) addresses.
    pub fn arith_feedback_for_address(&self, code_address: usize) -> Vec<u64> {
        self.arith_feedback_by_address
            .get(&code_address)
            .map(|slots| {
                slots
                    .iter()
                    .map(|&addr| unsafe { *(addr as *const u64) })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Snapshot every arith-feedback slot allocated while compiling the
    /// function at `code_address` — INCLUDING nested fns. Symmetric to
    /// `property_cache_range_for_address`. Tier-up needs this so per-
    /// site specialization stays aligned with allocation order across
    /// nested-fn boundaries.
    pub fn arith_feedback_range_for_address(&self, code_address: usize) -> Vec<u64> {
        let Some(&(start, end)) = self.arith_feedback_range_by_address.get(&code_address) else {
            return Vec::new();
        };
        if end <= start {
            return Vec::new();
        }
        let base = self.arith_feedback_cache.as_ptr() as usize;
        let stride = 8;
        let count = (end - start) / stride;
        (0..count)
            .map(|i| unsafe { *((base + start + i * stride) as *const u64) })
            .collect()
    }

    /// Record the contiguous slot range covered by `bind_arith_feedback`
    /// and any nested-fn binds for the function compiled at
    /// `code_address`. Called once per `Ast::Function` exit.
    pub fn record_arith_feedback_range(
        &mut self,
        code_address: usize,
        start_byte_offset: usize,
        end_byte_offset: usize,
    ) {
        if code_address == 0 || end_byte_offset <= start_byte_offset {
            return;
        }
        self.arith_feedback_range_by_address
            .entry(code_address)
            .or_insert((start_byte_offset, end_byte_offset));
    }

    /// Stamp `code_address` onto every property-cache entry in `slots`,
    /// and add them to the per-address index. Mirrors
    /// `bind_arith_feedback`. Called by AST-side bookkeeping after
    /// `upsert_function` returns.
    pub fn bind_property_cache(&mut self, slots: &[usize], code_address: usize) {
        if code_address == 0 || slots.is_empty() {
            return;
        }
        let base = self.property_look_up_cache.as_ptr() as usize;
        for &slot_addr in slots {
            let index = (slot_addr - base) / (3 * 8);
            if let Some(owner) = self.property_cache_owners.get_mut(index)
                && *owner == 0
            {
                *owner = code_address;
            }
        }
        self.property_cache_by_address
            .entry(code_address)
            .or_default()
            .extend_from_slice(slots);
    }

    /// Snapshot every property-cache entry owned by the function compiled
    /// at `code_address`. Returns `(struct_id_versioned, field_offset_bytes,
    /// is_mutable)` triples in source order. Sentinel entries (where the
    /// site never executed) appear with `struct_id_versioned == usize::MAX`.
    pub fn property_cache_for_address(&self, code_address: usize) -> Vec<(u64, u64, u64)> {
        self.property_cache_by_address
            .get(&code_address)
            .map(|slots| {
                slots
                    .iter()
                    .map(|&addr| unsafe {
                        let p = addr as *const AtomicUsize;
                        let key_before = (*p).load(Ordering::Acquire);
                        let offset = (*p.add(1)).load(Ordering::Relaxed) as u64;
                        let is_mutable = (*p.add(2)).load(Ordering::Relaxed) as u64;
                        let key_after = (*p).load(Ordering::Acquire);
                        if key_before == key_after {
                            (key_after as u64, offset, is_mutable)
                        } else {
                            (usize::MAX as u64, 0, 0)
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Snapshot every property-cache slot allocated while compiling the
    /// function at `code_address` — INCLUDING nested fns (closures and
    /// lambdas). Returns slots in cache-allocation order, which matches
    /// AST source-visit order. Tier-up uses this so the recompile can
    /// also bake offsets for property accesses inside the function's
    /// nested fns (their slots aren't in `property_cache_by_address`,
    /// which only holds slots directly bound to each fn's pointer for
    /// per-fn tier-up triggering).
    ///
    /// Returns the empty vec when no range was recorded (anonymous
    /// top-level bodies, etc.) — caller falls back to the per-fn list.
    pub fn property_cache_range_for_address(&self, code_address: usize) -> Vec<(u64, u64, u64)> {
        let Some(&(start, end)) = self.property_cache_range_by_address.get(&code_address) else {
            return Vec::new();
        };
        if end <= start {
            return Vec::new();
        }
        let base = self.property_look_up_cache.as_ptr() as usize;
        let stride = 3 * 8;
        let count = (end - start) / stride;
        (0..count)
            .map(|i| unsafe {
                let p = (base + start + i * stride) as *const AtomicUsize;
                let key_before = (*p).load(Ordering::Acquire);
                let offset = (*p.add(1)).load(Ordering::Relaxed) as u64;
                let is_mutable = (*p.add(2)).load(Ordering::Relaxed) as u64;
                let key_after = (*p).load(Ordering::Acquire);
                if key_before == key_after {
                    (key_after as u64, offset, is_mutable)
                } else {
                    (usize::MAX as u64, 0, 0)
                }
            })
            .collect()
    }

    /// Record the contiguous slot range that `bind_property_cache` and
    /// nested-fn binds covered for the function compiled at
    /// `code_address`. Called once per `Ast::Function` exit, after
    /// `upsert_function` returns the pointer.
    pub fn record_property_cache_range(
        &mut self,
        code_address: usize,
        start_byte_offset: usize,
        end_byte_offset: usize,
    ) {
        if code_address == 0 || end_byte_offset <= start_byte_offset {
            return;
        }
        // First write wins. A function whose AST is recompiled (tier-up)
        // gets a NEW pointer from `overwrite_function`, so it lands in a
        // fresh entry — no collision with the prior compile's range.
        self.property_cache_range_by_address
            .entry(code_address)
            .or_insert((start_byte_offset, end_byte_offset));
    }

    /// Look up the debug name we recorded for the function compiled at
    /// `code_address`. Returns `None` if no slot is bound there (function
    /// has no instrumented arithmetic, or never had its slots bound).
    pub fn debug_name_for_code_address(&self, code_address: usize) -> Option<&str> {
        let slots = self.arith_feedback_by_address.get(&code_address)?;
        let first_slot_addr = *slots.first()?;
        let base = self.arith_feedback_cache.as_ptr() as usize;
        let index = (first_slot_addr - base) / 8;
        self.arith_feedback_debug_names
            .get(index)
            .map(|s| s.as_str())
    }

    /// Walk every bound function and produce a per-function feedback
    /// summary. Sorted with the most-active functions first
    /// (active = slots with any observed shape) so consumers see
    /// specialization candidates at the top. Functions whose slots are
    /// all cold are still included so the dump can show "X functions are
    /// instrumented but never ran".
    pub fn specialization_report(&self) -> Vec<crate::feedback::FunctionFeedbackSummary> {
        let mut report: Vec<crate::feedback::FunctionFeedbackSummary> = self
            .arith_feedback_by_address
            .iter()
            .map(|(&code_addr, slots)| {
                let bits: Vec<u64> = slots
                    .iter()
                    .map(|&addr| unsafe { *(addr as *const u64) })
                    .collect();
                let name = self
                    .debug_name_for_code_address(code_addr)
                    .unwrap_or("<unknown>")
                    .to_string();
                crate::feedback::FunctionFeedbackSummary::from_bits(code_addr, name, &bits)
            })
            .collect();
        report.sort_by(|a, b| {
            b.active()
                .cmp(&a.active())
                .then_with(|| a.debug_name.cmp(&b.debug_name))
        });
        report
    }

    // TODO: All of this seems bad
    pub fn add_struct(&mut self, s: Struct) {
        let is_redefinition = {
            let runtime = get_runtime().get_mut();
            runtime.add_struct(s)
        };
        if is_redefinition {
            self.invalidate_all_property_caches();
            self.revert_all_specializations();
        }
    }

    /// Revert every specialized function to its retained tier-1 code (swap the
    /// jump-table slots back) and forget the specializations. Called on a
    /// runtime redefinition (eval/REPL/reload) so no tier-2 code keeps a stale
    /// snapshot of the world it was specialized against — struct layouts, name
    /// resolution, baked feedback. Tier-1 is fully dynamic, so reverting is
    /// always correct; hot functions re-tier-up afterward. No-op when nothing
    /// is specialized (so it's free at startup / cold runs).
    pub fn revert_all_specializations(&mut self) {
        // A redefinition may have changed protocol dispatch; stale inline-cache
        // entries (`[type_id -> old_fn_ptr]`) would keep calling old code, so
        // reset them regardless of whether any tier-2 code needs reverting.
        self.invalidate_all_protocol_dispatch_caches();
        if self.specialization_originals.is_empty() {
            return;
        }
        let runtime = get_runtime().get_mut();
        // Raw pointer so we can call `&mut self` reverts while holding the
        // `install_apply_lock` guard (which borrows a field of runtime).
        let rt_ptr: *mut crate::runtime::Runtime = runtime;
        // Hold the apply lock for the whole revert so a concurrent
        // `apply_pending_installs` (driven by an async tier-up) cannot interleave
        // its upserts with our pointer-reverts. Bumping the generation first
        // means any install staged against the old world is skipped when applied.
        let _apply_guard = runtime
            .install_apply_lock
            .lock()
            .expect("install_apply_lock poisoned");
        let g = runtime
            .install_generation
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if std::env::var("BEAGLE_STW_LOG").is_ok() {
            eprintln!("[revert] bumped generation {} -> {}", g, g + 1);
        }
        // Drop any tier-2 installs staged but not yet applied — they were
        // compiled against the pre-redefinition world and must not activate.
        runtime
            .installs_ready
            .store(false, std::sync::atomic::Ordering::Release);
        if let Ok(mut pending) = runtime.pending_installs.lock() {
            pending.clear();
        }
        for (name, (pointer, size)) in self.specialization_originals.drain() {
            unsafe { (*rt_ptr).revert_function_pointer(&name, pointer, size) };
        }
        self.specialized_names.clear();
    }

    /// Invalidate all property access inline caches by writing usize::MAX to the
    /// struct_id slot of every entry. This forces all accesses through the slow path.
    fn invalidate_all_property_caches(&mut self) {
        // Each cache entry is 24 bytes (3 words): [struct_id, field_offset, is_mutable]
        // Write usize::MAX to the struct_id slot (word 0) of each entry.
        let entry_size = 3 * 8; // 24 bytes per entry
        let mut offset = 0;
        while offset + entry_size <= self.property_look_up_cache_offset {
            unsafe {
                let cache_ptr =
                    self.property_look_up_cache.as_mut_ptr().add(offset) as *const AtomicUsize;
                (*cache_ptr).store(usize::MAX, Ordering::Release);
            }
            offset += entry_size;
        }
    }

    /// Reset every protocol-dispatch inline cache to its sentinel so the next
    /// call re-resolves through the (updated) dispatch table. Each entry is
    /// 16 bytes: [type_id, fn_ptr]; writing `usize::MAX` to the type_id slot
    /// forces a slow-path miss. Needed after a protocol impl is redefined —
    /// otherwise a cached `[type_id -> old_fn_ptr]` keeps calling stale code.
    pub fn invalidate_all_protocol_dispatch_caches(&mut self) {
        let entry_size = 2 * 8; // 16 bytes per entry: type_id + fn_ptr
        let mut offset = 0;
        while offset + entry_size <= self.protocol_dispatch_cache_offset {
            unsafe {
                let cache_ptr = self.protocol_dispatch_cache.as_mut_ptr().add(offset) as *mut usize;
                *cache_ptr = usize::MAX;
            }
            offset += entry_size;
        }
    }

    pub fn add_enum(&mut self, e: Enum) {
        let runtime = get_runtime().get_mut();
        runtime.add_enum(e);
    }

    /// Register reflection metadata for a top-level `let`-binding. Called
    /// from the AST compiler when a `let name = ...` is compiled at
    /// namespace scope. Sticky: a later REPL re-eval that passes `None`
    /// for `disk_location` will not clobber the binding's original
    /// on-disk origin.
    pub fn upsert_binding_metadata(
        &mut self,
        full_name: &str,
        source_text: Option<String>,
        disk_location: Option<crate::runtime::DiskLocation>,
    ) {
        let runtime = get_runtime().get_mut();
        runtime.upsert_binding_metadata(full_name, source_text, disk_location);
    }

    pub fn get_enum(&self, name: &str) -> Option<&Enum> {
        let runtime = get_runtime().get_mut();
        runtime.get_enum(name)
    }

    pub fn get_struct(&self, name: &str) -> Option<(usize, Struct)> {
        let runtime = get_runtime().get_mut();
        runtime.get_struct(name)
    }

    pub fn get_struct_family_id(&self, name: &str) -> Option<usize> {
        // With stable IDs, family_id == struct_id
        let runtime = get_runtime().get_mut();
        let (struct_id, _) = runtime.get_struct(name)?;
        Some(struct_id)
    }

    pub fn get_struct_layout_version(&self, struct_id: usize) -> u8 {
        let runtime = get_runtime().get_mut();
        runtime.structs.get_current_layout_version(struct_id)
    }

    /// Register a mapping from enum variant struct_id to enum name
    /// Used by effect handlers to determine which handler to call for a `perform` value
    pub fn register_enum_variant(&mut self, struct_id: usize, enum_name: String) {
        let runtime = get_runtime().get_mut();
        runtime.register_enum_variant(struct_id, enum_name);
    }

    pub fn is_inline_primitive_function(&self, name: &str) -> bool {
        name.starts_with("beagle.primitive/")
            || matches!(
                name,
                "beagle.string-builder/byte-at"
                    | "beagle.string-builder/set-byte-at!"
                    | "beagle.mutable-array/get"
                    | "beagle.mutable-array/write-field"
                    | "beagle.mutable-array/swap"
                    | "beagle.mutable-array/read-field-unsafe"
            )
    }

    pub fn extract_use(
        &self,
        use_stmt: &crate::ast::Ast,
    ) -> Result<(String, String), CompileError> {
        match use_stmt {
            crate::ast::Ast::Use {
                namespace_name,
                alias,
                ..
            } => {
                let alias = alias.as_ref().get_string();
                Ok((namespace_name.clone(), alias))
            }
            _ => Err(CompileError::ParseError(
                crate::parser::ParseError::InvalidDeclaration {
                    message: "Expected use AST node".to_string(),
                    location: crate::parser::SourceLocation::from_position(0),
                },
            )),
        }
    }

    /// Resolve a dotted namespace name to a file path.
    ///
    /// For `com.foo.bar`, tries in order:
    /// 1. `com/foo/bar.bg` (dots → slashes)
    /// 2. `foo/bar.bg` (drop first segment)
    /// 3. `bar.bg` (just last segment)
    ///
    /// Each path is tried:
    /// 1. Relative to source file
    /// 2. In standard library paths
    ///
    /// If a `beagle.toml` is found by walking up from the source file and
    /// declares `name = "thing"`, then for namespaces whose first segment
    /// matches that name, both the folder layout (`thing/foo/bar.bg`) and
    /// the stripped layout (`foo/bar.bg`) are accepted; an error is
    /// returned if both exist as different files.
    pub fn resolve_namespace_to_file(
        &self,
        namespace_name: &str,
        source_file: &str,
    ) -> Result<Option<String>, Box<dyn Error>> {
        let source_dir = Path::new(source_file)
            .parent()
            .ok_or("Invalid source file path")?;

        let parts: Vec<&str> = namespace_name.split('.').collect();

        // Candidate paths in order of preference.
        let mut candidates: Vec<String> = vec![];
        candidates.push(format!("{}.bg", parts.join("/"))); // com/foo/bar.bg
        candidates.push(format!("{}.bg", namespace_name)); // com.foo.bar.bg
        if parts.len() >= 2 {
            candidates.push(format!("{}.bg", parts[1..].join("/"))); // foo/bar.bg
        }
        if let Some(last) = parts.last() {
            candidates.push(format!("{}.bg", last)); // bar.bg
        }

        let mut winner: Option<String> = None;
        for candidate in &candidates {
            if let Some(p) = self.find_candidate_file(candidate, source_dir)? {
                winner = Some(p);
                break;
            }
        }

        // If beagle.toml declares a project name and the namespace's first
        // segment matches it, both the folder and stripped layouts are valid.
        // Disallow the ambiguous case where both exist as different files.
        if parts.len() >= 2
            && let Some(project_name) = Self::find_project_name(source_file)
            && parts[0] == project_name
        {
            let folder_form = format!("{}.bg", parts.join("/"));
            let stripped_form = format!("{}.bg", parts[1..].join("/"));
            let folder_resolved = self.find_candidate_file(&folder_form, source_dir)?;
            let stripped_resolved = self.find_candidate_file(&stripped_form, source_dir)?;
            if let (Some(a), Some(b)) = (folder_resolved, stripped_resolved)
                && a != b
            {
                return Err(format!(
                    "Ambiguous namespace '{}': resolves to both '{}' and '{}'. Remove one.",
                    namespace_name, a, b
                )
                .into());
            }
        }

        Ok(winner)
    }

    /// Search every location used by `resolve_namespace_to_file` for a
    /// single candidate filename (e.g. `foo/bar.bg`). Returns the first
    /// existing path, or `None`.
    fn find_candidate_file(
        &self,
        candidate: &str,
        source_dir: &Path,
    ) -> Result<Option<String>, Box<dyn Error>> {
        let path_to_string = |p: &Path| -> Result<String, Box<dyn Error>> {
            p.to_str().map(|s| s.to_string()).ok_or_else(|| {
                CompileError::PathConversion {
                    path: format!("{:?}", p),
                }
                .into()
            })
        };

        let relative_path = source_dir.join(candidate);
        if relative_path.exists() {
            return Ok(Some(path_to_string(&relative_path)?));
        }

        for include_dir in &self.command_line_arguments.include_paths {
            let include_path = Path::new(include_dir).join(candidate);
            if include_path.exists() {
                return Ok(Some(path_to_string(&include_path)?));
            }
        }

        let exe_path = env::current_exe()?;
        let exe_dir = exe_path
            .parent()
            .ok_or("Cannot get parent of executable path")?
            .to_path_buf();

        let stdlib_path = exe_dir.join(format!("standard-library/{}", candidate));
        if stdlib_path.exists() {
            return Ok(Some(path_to_string(&stdlib_path)?));
        }

        // Walk up from the executable's directory looking for `standard-library/`,
        // covering layouts like `target/debug/...` (cargo run, cargo test).
        let mut walk = exe_dir.parent();
        for _ in 0..3 {
            let Some(dir) = walk else { break };
            let candidate_path = dir.join(format!("standard-library/{}", candidate));
            if candidate_path.exists() {
                return Ok(Some(path_to_string(&candidate_path)?));
            }
            walk = dir.parent();
        }

        Ok(None)
    }

    /// Walk up from the source file's directory looking for a `beagle.toml`,
    /// and return the value of its `name = "..."` field if present.
    fn find_project_name(source_file: &str) -> Option<String> {
        let mut dir = Path::new(source_file).parent()?.to_path_buf();
        if dir.as_os_str().is_empty() {
            dir = std::env::current_dir().ok()?;
        }
        loop {
            let toml_path = dir.join("beagle.toml");
            if toml_path.exists() {
                let content = std::fs::read_to_string(&toml_path).ok()?;
                return Self::parse_toml_name(&content);
            }
            dir = match dir.parent() {
                Some(p) => p.to_path_buf(),
                None => return None,
            };
        }
    }

    /// Minimal hand parser for the single `name = "..."` field we care about
    /// today. A full toml parser can replace this when the manifest grows.
    fn parse_toml_name(content: &str) -> Option<String> {
        for raw_line in content.lines() {
            // Strip comments, ignoring `#` inside strings (we only look at top-level
            // `name = "..."` so this simple stripping is fine for the field we read).
            let line = match raw_line.find('#') {
                Some(i) => &raw_line[..i],
                None => raw_line,
            };
            let line = line.trim();
            if line.is_empty() || line.starts_with('[') {
                continue;
            }
            let Some(rest) = line.strip_prefix("name") else {
                continue;
            };
            let rest = rest.trim_start();
            let Some(rest) = rest.strip_prefix('=') else {
                continue;
            };
            let rest = rest.trim();
            let value = rest
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .or_else(|| rest.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))?;
            if value.is_empty() {
                return None;
            }
            return Some(value.to_string());
        }
        None
    }

    pub fn get_struct_by_id(&self, struct_id: usize) -> Option<Struct> {
        let runtime = get_runtime().get_mut();
        runtime.get_struct_by_id(struct_id)
    }

    pub fn current_namespace_name(&self) -> String {
        let runtime = get_runtime().get_mut();
        runtime.current_namespace_name()
    }

    pub fn reserve_namespace_slot(&mut self, name: &str) -> usize {
        let runtime = get_runtime().get_mut();
        runtime.reserve_namespace_slot(name)
    }

    /// Stable absolute address of the binding cell for `(namespace_id, slot)`.
    /// The compiler bakes this into emitted code so a namespace-variable
    /// read is a single load instruction instead of a runtime call.
    pub fn binding_cell_address(&self, namespace_id: usize, slot: usize) -> Option<usize> {
        let runtime = get_runtime().get_mut();
        runtime.binding_cell_address(namespace_id, slot)
    }

    pub fn find_binding(&self, namespace_id: usize, name: &str) -> Option<usize> {
        let runtime = get_runtime().get_mut();
        runtime.find_binding(namespace_id, name)
    }

    pub fn reserve_namespace(&mut self, name: String) -> usize {
        let runtime = get_runtime().get_mut();
        runtime.reserve_namespace(name)
    }

    pub fn add_alias(&mut self, library_name: String, alias: String) {
        let runtime = get_runtime().get_mut();
        runtime.add_alias(library_name, alias);
    }

    /// Check if a namespace is already registered in the runtime
    pub fn namespace_exists(&self, name: &str) -> bool {
        let runtime = get_runtime().get();
        runtime.get_namespace_id(name).is_some()
    }

    pub fn current_namespace_id(&self) -> usize {
        let runtime = get_runtime().get();
        runtime.current_namespace_id()
    }

    pub fn get_namespace_from_alias(&self, alias: &str) -> Option<String> {
        let runtime = get_runtime().get();
        runtime.get_namespace_from_alias(alias).or_else(|| {
            if runtime.get_namespace_id(alias).is_some() {
                Some(alias.to_string())
            } else {
                None
            }
        })
    }

    pub fn global_namespace_id(&self) -> usize {
        let runtime = get_runtime().get();
        runtime.global_namespace_id()
    }

    pub fn get_namespace_id(&self, namespace_name: &str) -> Option<usize> {
        let runtime = get_runtime().get();
        runtime.get_namespace_id(namespace_name)
    }

    pub fn set_current_namespace(&mut self, namespace_id: usize) {
        let runtime = get_runtime().get_mut();
        runtime.set_current_namespace(namespace_id);
    }

    pub fn get_function_pointer(&self, f: Function) -> Option<usize> {
        let runtime = get_runtime().get();
        runtime.get_function_pointer(f).ok().map(|x| x as usize)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsert_function<B: CodegenBackend>(
        &mut self,
        function_name: Option<&str>,
        backend: &mut B,
        max_locals: usize,
        number_of_args: usize,
        is_variadic: bool,
        min_args: usize,
        docstring: Option<String>,
        arg_names: Vec<String>,
        source_file: Option<String>,
        source_line: Option<usize>,
        source_text: Option<String>,
        disk_location: Option<crate::runtime::DiskLocation>,
    ) -> Result<usize, Box<dyn Error>> {
        if let Some(name) = function_name {
            backend.set_function_name(name);
        }
        #[cfg(feature = "debug-gc")]
        eprintln!(
            "[GC DEBUG] upsert_function: {:?} max_locals_param={}, backend.max_locals={}",
            function_name,
            max_locals,
            backend.max_locals()
        );
        let code = backend.compile_to_bytes();
        let pointer = self.add_code(&code)?;

        if let Some(name) = function_name {
            if self.dump.should_emit(crate::dump::Stage::Asm, name) {
                let disasm = crate::dump::disassemble_machine_code(&code, pointer as u64);
                self.dump.emit(
                    crate::dump::Stage::Asm,
                    name,
                    serde_json::json!({
                        "address": format!("0x{:x}", pointer as u64),
                        "size": code.len(),
                        "instructions": disasm,
                    }),
                );
            }
        }

        if self.defer_function_installs {
            // Batch mode: compile code but defer the jump table update.
            // Code is written to executable pages but the jump table still
            // points to the old function until flush_deferred_functions().
            // make_executable is called later in flush_deferred_functions.
            self.deferred_updates.push(DeferredFunctionUpdate {
                name: function_name.map(|s| s.to_string()),
                pointer,
                code_size: code.len(),
                max_locals,
                number_of_args,
                is_variadic,
                min_args,
                docstring,
                arg_names,
                source_file,
                source_line,
                source_text,
                disk_location,
            });
            // Return a placeholder — the real pointer will be set during flush.
            // For now, return the code pointer (it's valid, just not in the jump table yet).
            Ok(pointer as usize)
        } else {
            // Immediate mode: make executable and update jump table now.
            // Make the new code executable BEFORE updating the jump table.
            // Without this, there's a race: another thread following the jump table
            // could try to execute the new code while the page is still writable
            // (not executable), causing SIGBUS on ARM64 due to W^X enforcement.
            self.code_allocator.make_executable();
            let runtime = get_runtime().get_mut();
            runtime.upsert_function(
                function_name,
                pointer,
                code.len(),
                max_locals,
                number_of_args,
                is_variadic,
                min_args,
                docstring,
                arg_names,
                source_file,
                source_line,
                source_text,
                disk_location,
            )
        }
    }

    /// Install all deferred function updates into the jump table atomically.
    /// This ensures that other threads see either all old functions or all new
    /// functions, never a mix that could lead to new functions being called
    /// with old-layout struct values.
    /// Make the just-compiled tier-2 code executable and QUEUE its installs on
    /// the runtime without touching the jump table or function metadata. A
    /// coordinator activates them inside a stop-the-world
    /// (`Runtime::apply_pending_installs`); `installs_ready` flags that work is
    /// pending. Used by `specialize_function` so the compiler thread never
    /// mutates the function table while mutators read it.
    fn stage_specialization_installs(&mut self) {
        if self.deferred_updates.is_empty() {
            return;
        }
        self.code_allocator.make_executable();
        let updates = std::mem::take(&mut self.deferred_updates);
        let runtime = get_runtime().get_mut();
        // Tag with the current world generation so a redefinition that bumps
        // the generation before this is applied marks it stale.
        let generation = runtime
            .install_generation
            .load(std::sync::atomic::Ordering::Acquire);
        {
            let mut pending = runtime
                .pending_installs
                .lock()
                .expect("pending_installs poisoned");
            for u in updates {
                pending.push(crate::runtime::PendingInstall {
                    generation,
                    name: u.name,
                    pointer: u.pointer as usize,
                    code_size: u.code_size,
                    max_locals: u.max_locals,
                    number_of_args: u.number_of_args,
                    is_variadic: u.is_variadic,
                    min_args: u.min_args,
                    docstring: u.docstring,
                    arg_names: u.arg_names,
                    source_file: u.source_file,
                    source_line: u.source_line,
                    source_text: u.source_text,
                    disk_location: u.disk_location,
                });
            }
        }
        runtime
            .installs_ready
            .store(true, std::sync::atomic::Ordering::Release);
    }

    fn flush_deferred_functions(&mut self) {
        if self.deferred_updates.is_empty() {
            return;
        }
        // First make all new code pages executable
        self.code_allocator.make_executable();
        // Then install all functions into the jump table
        let updates = std::mem::take(&mut self.deferred_updates);
        let runtime = get_runtime().get_mut();
        for update in updates {
            let _ = runtime.upsert_function(
                update.name.as_deref(),
                update.pointer,
                update.code_size,
                update.max_locals,
                update.number_of_args,
                update.is_variadic,
                update.min_args,
                update.docstring,
                update.arg_names,
                update.source_file,
                update.source_line,
                update.source_text,
                update.disk_location,
            );
        }
    }

    pub fn upsert_function_bytes(
        &mut self,
        function_name: Option<&str>,
        code: Vec<u8>,
        max_locals: usize,
        number_of_args: usize,
        is_variadic: bool,
        min_args: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let pointer = self.add_code(&code)?;
        self.code_allocator.make_executable();
        let runtime = get_runtime().get_mut();
        runtime.upsert_function(
            function_name,
            pointer,
            code.len(),
            max_locals,
            number_of_args,
            is_variadic,
            min_args,
            None, // No docstring for this path
            vec![],
            None,
            None,
            None,
            None,
        )
    }

    pub fn add_function_alias(
        &self,
        alias: &str,
        function: &Function,
    ) -> Result<(), Box<dyn Error>> {
        let runtime = get_runtime().get_mut();
        runtime.add_function(
            Some(alias),
            usize::from(function.pointer) as *const u8,
            function.size,
            function.number_of_locals,
            function.number_of_args,
            function.is_variadic,
            function.min_args,
            function.docstring.clone(),
            function.arg_names.clone(),
            function.source_file.clone(),
            function.source_line,
            function.source_text.clone(),
            function.disk_location.clone(),
        )?;
        Ok(())
    }

    pub fn get_pointer_for_function(&self, function: &Function) -> Option<usize> {
        let runtime = get_runtime().get();
        runtime.get_pointer_for_function(function)
    }

    pub fn find_function(&self, name: &str) -> Option<Function> {
        let runtime = get_runtime().get();
        runtime.find_function(name)
    }

    pub fn get_jump_table_pointer(&self, function: Function) -> Option<usize> {
        let runtime = get_runtime().get();
        runtime.get_jump_table_pointer(function).ok()
    }

    pub fn get_function_by_name(&self, name: &str) -> Option<&Function> {
        let runtime = get_runtime().get();
        runtime.get_function_by_name(name)
    }

    pub fn get_function_by_pointer(&self, pointer: usize) -> Option<&Function> {
        let runtime = get_runtime().get();
        runtime.get_function_by_pointer(pointer as *const u8)
    }

    pub fn reserve_function(
        &mut self,
        full_function_name: &str,
        number_of_args: usize,
        is_variadic: bool,
        min_args: usize,
    ) -> Option<Function> {
        let runtime = get_runtime().get_mut();
        runtime
            .reserve_function(full_function_name, number_of_args, is_variadic, min_args)
            .ok()
    }

    /// Register a multi-arity function for static dispatch optimization.
    /// The arities list contains (fixed_arity, is_variadic) pairs.
    pub fn register_multi_arity_function(&mut self, name: &str, arities: Vec<(usize, bool)>) {
        self.multi_arity_functions
            .insert(name.to_string(), MultiArityInfo { arities });
    }

    /// Look up multi-arity function info for static dispatch.
    pub fn get_multi_arity_info(&self, name: &str) -> Option<&MultiArityInfo> {
        self.multi_arity_functions.get(name)
    }

    /// Check if a function is registered as multi-arity.
    pub fn is_multi_arity_function(&self, name: &str) -> bool {
        self.multi_arity_functions.contains_key(name)
    }

    /// Find the appropriate arity variant for a multi-arity function call.
    /// Returns the name of the specific arity function to call.
    pub fn resolve_multi_arity_call(&self, name: &str, arg_count: usize) -> Option<String> {
        let info = self.multi_arity_functions.get(name)?;

        // First try exact match
        for (arity, is_variadic) in &info.arities {
            if !is_variadic && *arity == arg_count {
                return Some(format!("{}${}", name, arity));
            }
        }

        // Then try variadic match (arg_count >= min_args for variadic)
        for (min_arity, is_variadic) in &info.arities {
            if *is_variadic && arg_count >= *min_arity {
                return Some(format!("{}${}", name, min_arity));
            }
        }

        None
    }

    fn build_method_if_chain(
        &mut self,
        default_method: Option<&Function>,
        protocol_methods: Vec<ProtocolMethodInfo>,
        args: Vec<String>,
    ) -> Ast {
        if protocol_methods.is_empty() {
            if let Some(default_method) = default_method {
                let function_name = default_method.name.clone();
                return Ast::Call {
                    name: function_name,
                    args: args
                        .iter()
                        .map(|x| Ast::Identifier(x.to_string(), 0))
                        .collect(),
                    token_range: TokenRange::new(0, 0),
                };
            }
            return Ast::Call {
                name: "beagle.builtin/throw-error".to_string(),
                args: vec![],
                token_range: TokenRange::new(0, 0),
            };
        }

        // Use .first() which can't fail for non-empty vec (already checked above)
        let first_method = &protocol_methods[0];
        let untagged_pointer = BuiltInTypes::untag(first_method.fn_pointer);
        let function_name = self
            .get_function_by_pointer(untagged_pointer)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| {
                // Fall back to error function if we can't find the method
                "beagle.builtin/throw-error".to_string()
            });

        // Use instance_of for type checking
        // TODO: Future optimization - inline the struct_id comparison for struct types
        // with proper guard for heap objects
        Ast::If {
            condition: Box::new(Ast::Call {
                name: "beagle.core/instance-of".to_string(),
                args: vec![
                    Ast::Identifier(args[0].to_string(), 0),
                    Ast::Identifier(first_method._type.to_string(), 0),
                ],
                token_range: TokenRange::new(0, 0),
            }),
            then: vec![Ast::Call {
                name: function_name,
                args: args
                    .iter()
                    .map(|x| Ast::Identifier(x.to_string(), 0))
                    .collect(),
                token_range: TokenRange::new(0, 0),
            }],
            else_: vec![self.build_method_if_chain(
                default_method,
                protocol_methods[1..].to_vec(),
                args,
            )],
            token_range: TokenRange::new(0, 0),
        }
    }

    /// Build an optimized dispatch using inline cache and dispatch table
    /// This replaces the if-chain with O(1) lookup
    fn build_optimized_dispatch(
        &mut self,
        protocol_name: &str,
        method_name: &str,
        default_method: Option<&Function>,
        args: Vec<String>,
    ) -> Option<Ast> {
        let runtime = get_runtime().get();

        // Get dispatch table pointer - if none exists, fall back to if-chain
        let dispatch_table_ptr = runtime.get_dispatch_table_ptr(protocol_name, method_name)?;

        // Allocate inline cache entry
        let cache_location = self.add_protocol_dispatch_cache().ok()?;

        // Get default method pointer (0 if no default)
        let default_fn_ptr = default_method.map(|f| usize::from(f.pointer)).unwrap_or(0);

        Some(Ast::ProtocolDispatch {
            args,
            cache_location,
            dispatch_table_ptr,
            default_fn_ptr,
            num_args: default_method.map(|f| f.number_of_args).unwrap_or(1),
            token_range: TokenRange::new(0, 0),
        })
    }

    fn compile_protocol_method(
        &mut self,
        protocol_name: String,
        method_name: String,
        protocol_methods: Vec<ProtocolMethodInfo>,
    ) -> Result<(), Box<dyn Error>> {
        let runtime = get_runtime().get_mut();
        let (protocol_namespace, protocol_name_only) = protocol_name
            .split_once("/")
            .ok_or_else(|| format!("Invalid protocol name format: {}", protocol_name))?;
        let full_method_name = format!("{}/{}", protocol_namespace, method_name);
        let fully_qualified_name = format!(
            "{}/{}_{}",
            protocol_namespace, protocol_name_only, method_name
        );
        // Clone the function to release the borrow on runtime
        let default_method = runtime.get_function_by_name(&fully_qualified_name).cloned();
        // Set the default method on the dispatch table for fallback when no specific impl exists
        if let Some(ref default_fn) = default_method {
            // Tag the raw pointer as a function (tag = 0b100 = 4)
            let raw_ptr = usize::from(default_fn.pointer);
            let default_ptr = (raw_ptr << 3) | 4;
            runtime.set_dispatch_table_default(&protocol_name, &method_name, default_ptr);
        }
        let function = self.find_function(&full_method_name).ok_or_else(|| {
            CompileError::FunctionNotFound {
                function_name: full_method_name.clone(),
            }
        })?;
        let args: Vec<Pattern> = (0..function.number_of_args)
            .map(|x| Pattern::Identifier {
                name: format!("arg{}", x),
                token_range: TokenRange::new(0, 0),
            })
            .collect();
        let args_as_strings: Vec<String> = (0..function.number_of_args)
            .map(|x| format!("arg{}", x))
            .collect();

        let protocol_namespace_id = runtime
            .get_namespace_id(protocol_namespace)
            .ok_or_else(|| format!("Protocol namespace not found: {}", protocol_namespace))?;

        // The whole build + compile runs inside the protocol's namespace;
        // `with_namespace` makes the restore unskippable (a leaked switch
        // would poison every later compile on this thread).
        let compile_result = self.with_namespace(protocol_namespace_id, |this| {
            // Try optimized dispatch first
            let body = if let Some(optimized) = this.build_optimized_dispatch(
                &protocol_name,
                &method_name,
                default_method.as_ref(),
                args_as_strings.clone(),
            ) {
                vec![optimized]
            } else {
                // Fall back to if-chain
                vec![this.build_method_if_chain(
                    default_method.as_ref(),
                    protocol_methods,
                    args_as_strings.clone(),
                )]
            };

            let ast = Ast::Program {
                elements: vec![Ast::Function {
                    name: Some(method_name.clone()),
                    args: args.clone(),
                    rest_param: None,
                    body,
                    token_range: TokenRange::new(0, 0),
                    docstring: None,
                }],
                token_range: TokenRange::new(0, 0),
            };
            // Ignore diagnostics from reify method compilation - these are internal
            this.compile_ast(
                ast,
                None,
                "test",
                vec![],
                String::new(),
                vec![],
                HashMap::new(),
            )
        });
        let _ = compile_result?;
        self.code_allocator.make_executable();
        Ok(())
    }
}

impl fmt::Debug for Compiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: Make this better
        f.debug_struct("Compiler").finish()
    }
}

pub enum CompilerMessage {
    CompileString(String),
    CompileStringInNamespace(String, String), // (code, namespace_name)
    CompileFile(String),
    CompileSource(String, String),
    /// (code, namespace_name, file_name, byte_offset, line_offset) — used
    /// by `reflect/write-source` to re-register a function after splicing
    /// edited text into its original file.
    CompileStringWithFileContext(String, String, String, usize, usize),
    AddFunctionMarkExecutable(String, Vec<u8>, usize, usize),
    CompileProtocolMethod(String, String, Vec<ProtocolMethodInfo>),
    SetPauseAtomPointer(usize),
    GetCodeBaseAddress,
    /// Snapshot every allocated arithmetic-feedback slot. Used by
    /// `--dump-arith-feedback` after the program exits.
    GetArithFeedback,
    /// Walk feedback and produce per-function specialization verdicts.
    /// Used by `--dump-specializable`.
    GetSpecializationReport,
    /// Specialize every FullySpecializable function in the report.
    /// Returns the count of functions that were actually swapped.
    SpecializeAll,
    /// Tier-up trigger fired by the per-function entry counter.
    /// Specializes only the named function; idempotent (handler skips
    /// already-specialized).
    SpecializeFunction(String),
    /// OSR: build the optimized continuation for `(function, loop_index)`.
    /// Sent by the back-edge trampoline when a loop gets hot; the result is
    /// published in the OSR registry (`crate::osr`).
    BuildOsrVariant(String, usize),
    /// Reset all protocol-dispatch inline caches to their sentinel. Sent after
    /// a protocol impl is REDEFINED (so the dispatch table changed) — clears
    /// `[type_id -> old_fn_ptr]` entries so the next dispatch re-resolves.
    InvalidateProtocolDispatchCaches,
}

pub enum CompilerResponse {
    Done,
    FunctionsToRun(Vec<String>),
    FunctionPointer(usize),
    /// Like `FunctionPointer`, but also reports the namespace the compile
    /// ended in (after any `namespace X` directive). Used by
    /// `CompileStringInNamespace` so a REPL session can track its own current
    /// namespace without reading the shared global.
    FunctionPointerInNamespace(usize, String),
    CompileError(String),
    CodeBaseAddress(usize),
    /// Per slot: (slot_address, slot_value, owning_code_address, debug_name).
    /// `owning_code_address` is 0 for slots whose owning compile failed
    /// or was never bound. `debug_name` is the function's fully-qualified
    /// name at compile time and is for display only.
    ArithFeedback(Vec<(usize, u64, usize, String)>),
    SpecializationReport(Vec<crate::feedback::FunctionFeedbackSummary>),
    SpecializeCount(usize),
}

pub struct CompilerThread {
    compiler: Compiler,
    channel: BlockingReceiver<CompilerMessage, CompilerResponse>,
}

impl CompilerThread {
    pub fn new(
        channel: BlockingReceiver<CompilerMessage, CompilerResponse>,
        command_line_arguments: CommandLineArguments,
        diagnostic_store: Arc<Mutex<DiagnosticStore>>,
    ) -> Result<Self, CompileError> {
        Ok(CompilerThread {
            compiler: Compiler {
                code_allocator: CodeAllocator::new(),
                property_look_up_cache: MmapOptions::new(MmapOptions::page_size() * 256) // 1MB cache for property lookups
                    .map_err(|e| {
                        CompileError::MemoryMapping(format!("Failed to create mmap: {}", e))
                    })?
                    .map_mut()
                    .map_err(|e| CompileError::MemoryMapping(format!("Failed to map mmap: {}", e)))?
                    .make_mut()
                    .map_err(|(_map, e)| {
                        CompileError::MemoryMapping(format!("Failed to make mmap mutable: {}", e))
                    })?,
                property_look_up_cache_offset: 0,
                property_cache_owners: Vec::new(),
                property_cache_by_address: HashMap::new(),
                property_cache_range_by_address: HashMap::new(),
                protocol_dispatch_cache: MmapOptions::new(MmapOptions::page_size())
                    .map_err(|e| {
                        CompileError::MemoryMapping(format!("Failed to create mmap: {}", e))
                    })?
                    .map_mut()
                    .map_err(|e| CompileError::MemoryMapping(format!("Failed to map mmap: {}", e)))?
                    .make_mut()
                    .map_err(|(_map, e)| {
                        CompileError::MemoryMapping(format!("Failed to make mmap mutable: {}", e))
                    })?,
                protocol_dispatch_cache_offset: 0,
                arith_feedback_cache: MmapOptions::new(MmapOptions::page_size() * 256) // 1MB = ~131k slots
                    .map_err(|e| {
                        CompileError::MemoryMapping(format!("Failed to create mmap: {}", e))
                    })?
                    .map_mut()
                    .map_err(|e| CompileError::MemoryMapping(format!("Failed to map mmap: {}", e)))?
                    .make_mut()
                    .map_err(|(_map, e)| {
                        CompileError::MemoryMapping(format!("Failed to make mmap mutable: {}", e))
                    })?,
                arith_feedback_cache_offset: 0,
                arith_feedback_owners: Vec::new(),
                arith_feedback_by_address: HashMap::new(),
                arith_feedback_range_by_address: HashMap::new(),
                arith_feedback_debug_names: Vec::new(),
                specialized_names: HashSet::new(),
                extend_method_fragments: HashSet::new(),
                osr_capture: None,
                specialization_originals: HashMap::new(),
                function_counter_cache: MmapOptions::new(MmapOptions::page_size() * 64) // 256KB ≈ 32k functions
                    .map_err(|e| {
                        CompileError::MemoryMapping(format!("Failed to create mmap: {}", e))
                    })?
                    .map_mut()
                    .map_err(|e| CompileError::MemoryMapping(format!("Failed to map mmap: {}", e)))?
                    .make_mut()
                    .map_err(|(_map, e)| {
                        CompileError::MemoryMapping(format!("Failed to make mmap mutable: {}", e))
                    })?,
                function_counter_cache_offset: 0,
                function_counter_names: Vec::new(),
                command_line_arguments: command_line_arguments.clone(),
                pause_atom_ptr: None,
                compiled_file_cache: HashSet::new(),
                diagnostic_store,
                multi_arity_functions: HashMap::new(),
                dynamic_vars: HashMap::new(),
                defer_function_installs: false,
                deferred_updates: Vec::new(),
                struct_defaults: HashMap::new(),
                dump: command_line_arguments.dump.clone(),
            },
            channel,
        })
    }

    pub fn run(&mut self) {
        loop {
            let result = self.channel.receive();
            match result {
                Ok((message, work_done)) => match message {
                    CompilerMessage::CompileString(string) => {
                        match self.compiler.compile_string(&string) {
                            Ok(pointer) => {
                                work_done.mark_done(CompilerResponse::FunctionPointer(pointer));
                            }
                            Err(e) => {
                                work_done
                                    .mark_done(CompilerResponse::CompileError(format!("{}", e)));
                            }
                        }
                    }
                    CompilerMessage::CompileStringInNamespace(string, namespace) => {
                        match self
                            .compiler
                            .compile_string_in_namespace(&string, Some(&namespace))
                        {
                            Ok((pointer, ending_namespace)) => {
                                work_done.mark_done(CompilerResponse::FunctionPointerInNamespace(
                                    pointer,
                                    ending_namespace,
                                ));
                            }
                            Err(e) => {
                                work_done
                                    .mark_done(CompilerResponse::CompileError(format!("{}", e)));
                            }
                        }
                    }
                    CompilerMessage::CompileFile(file_name) => {
                        match self.compiler.compile(&file_name) {
                            Ok(top_levels) => {
                                work_done.mark_done(CompilerResponse::FunctionsToRun(top_levels));
                            }
                            Err(e) => {
                                work_done
                                    .mark_done(CompilerResponse::CompileError(format!("{}", e)));
                            }
                        }
                    }
                    CompilerMessage::CompileSource(name, source) => {
                        match self.compiler.compile_source(&name, &source) {
                            Ok(top_levels) => {
                                work_done.mark_done(CompilerResponse::FunctionsToRun(top_levels));
                            }
                            Err(e) => {
                                work_done
                                    .mark_done(CompilerResponse::CompileError(format!("{}", e)));
                            }
                        }
                    }
                    CompilerMessage::CompileStringWithFileContext(
                        code,
                        namespace,
                        file_name,
                        byte_offset,
                        line_offset,
                    ) => {
                        let ns = if namespace.is_empty() {
                            None
                        } else {
                            Some(namespace.as_str())
                        };
                        match self.compiler.compile_string_with_file_context(
                            &code,
                            ns,
                            &file_name,
                            byte_offset,
                            line_offset,
                        ) {
                            Ok(pointer) => {
                                work_done.mark_done(CompilerResponse::FunctionPointer(pointer));
                            }
                            Err(e) => {
                                work_done
                                    .mark_done(CompilerResponse::CompileError(format!("{}", e)));
                            }
                        }
                    }
                    CompilerMessage::SetPauseAtomPointer(pointer) => {
                        self.compiler.set_pause_atom_ptr(pointer);
                        work_done.mark_done(CompilerResponse::Done);
                    }
                    CompilerMessage::AddFunctionMarkExecutable(
                        name,
                        code,
                        max_locals,
                        number_of_args,
                    ) => {
                        match self.compiler.upsert_function_bytes(
                            Some(&name),
                            code,
                            max_locals,
                            number_of_args,
                            false,
                            number_of_args,
                        ) {
                            Ok(_) => {
                                self.compiler.code_allocator.make_executable();
                                work_done.mark_done(CompilerResponse::Done);
                            }
                            Err(e) => {
                                work_done
                                    .mark_done(CompilerResponse::CompileError(format!("{}", e)));
                            }
                        }
                    }
                    CompilerMessage::CompileProtocolMethod(
                        protocol_name,
                        method_name,
                        protocol_methods,
                    ) => {
                        match self.compiler.compile_protocol_method(
                            protocol_name,
                            method_name,
                            protocol_methods,
                        ) {
                            Ok(_) => {
                                work_done.mark_done(CompilerResponse::Done);
                            }
                            Err(e) => {
                                work_done
                                    .mark_done(CompilerResponse::CompileError(format!("{}", e)));
                            }
                        }
                    }
                    CompilerMessage::GetCodeBaseAddress => {
                        let base = self.compiler.code_allocator.base_address();
                        work_done.mark_done(CompilerResponse::CodeBaseAddress(base));
                    }
                    CompilerMessage::GetArithFeedback => {
                        let snapshot: Vec<(usize, u64, usize, String)> = self
                            .compiler
                            .iter_arith_feedback()
                            .map(|(addr, value, owner_addr, name)| {
                                (addr, value, owner_addr, name.to_string())
                            })
                            .collect();
                        work_done.mark_done(CompilerResponse::ArithFeedback(snapshot));
                    }
                    CompilerMessage::GetSpecializationReport => {
                        let report = self.compiler.specialization_report();
                        work_done.mark_done(CompilerResponse::SpecializationReport(report));
                    }
                    CompilerMessage::InvalidateProtocolDispatchCaches => {
                        self.compiler.invalidate_all_protocol_dispatch_caches();
                        work_done.mark_done(CompilerResponse::Done);
                    }
                    CompilerMessage::SpecializeFunction(name) => {
                        // Only STAGE the install here (no jump-table swap): the
                        // caller (the async tier-up spawn thread) drives the
                        // stop-the-world that applies it. We must NOT coordinate
                        // the STW from the compiler thread — a mutator blocked
                        // sending us another message could never park, hanging
                        // the rendezvous. Keeping this thread free to service
                        // messages lets such mutators unblock and park.
                        if let Err(e) = self.compiler.specialize_function(&name) {
                            eprintln!("specialize_function({}) failed: {}", name, e);
                        }
                        work_done.mark_done(CompilerResponse::Done);
                    }
                    CompilerMessage::BuildOsrVariant(name, loop_idx) => {
                        self.compiler.build_osr_variant(&name, loop_idx);
                        work_done.mark_done(CompilerResponse::Done);
                    }
                    CompilerMessage::SpecializeAll => {
                        let report = self.compiler.specialization_report();
                        let mut count = 0usize;
                        for summary in report {
                            if matches!(
                                summary.verdict,
                                crate::feedback::SpecializationVerdict::FullySpecializable
                            ) {
                                match self.compiler.specialize_function(&summary.debug_name) {
                                    Ok(true) => count += 1,
                                    Ok(false) => {}
                                    Err(e) => {
                                        eprintln!(
                                            "specialize_function({}) failed: {}",
                                            summary.debug_name, e
                                        );
                                    }
                                }
                            }
                        }
                        work_done.mark_done(CompilerResponse::SpecializeCount(count));
                    }
                },
                Err(_) => {
                    break;
                }
            }
        }
    }
}

pub struct BlockingSender<T, R> {
    inner: SyncSender<(T, SyncSender<R>)>,
}

// Manual Clone — the auto-derive adds `T: Clone, R: Clone` bounds we
// don't need (SyncSender is Clone for any inner type). Done by hand so
// the auto-specialize background thread can hold its own sender
// without forcing Clone onto CompilerMessage / CompilerResponse, which
// carry response payloads we never want to deep-copy.
impl<T, R> Clone for BlockingSender<T, R> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T, R> BlockingSender<T, R> {
    pub fn send(&self, message: T) -> R {
        let (done_tx, done_rx) = mpsc::sync_channel(0);
        self.inner
            .send((message, done_tx))
            .expect("Compiler thread has disconnected - this is a fatal error");
        done_rx
            .recv()
            .expect("Compiler thread failed to send response - this is a fatal error")
    }
}

pub struct BlockingReceiver<T, R> {
    inner: Receiver<(T, SyncSender<R>)>,
}

impl<T, R> BlockingReceiver<T, R> {
    pub fn receive(&self) -> Result<(T, WorkDone<R>), mpsc::RecvError> {
        let (message, done_tx) = self.inner.recv()?;
        Ok((message, WorkDone { done_tx }))
    }
}

pub fn blocking_channel<T, R>() -> (BlockingSender<T, R>, BlockingReceiver<T, R>) {
    let (sender, receiver) = mpsc::sync_channel(0);
    (
        BlockingSender { inner: sender },
        BlockingReceiver { inner: receiver },
    )
}
pub struct WorkDone<R> {
    done_tx: SyncSender<R>,
}

impl<R> WorkDone<R> {
    pub fn mark_done(self, result: R) {
        self.done_tx
            .send(result)
            .expect("Failed to send result back to main thread - this is a fatal error");
    }
}

#[cfg(test)]
mod fragment_context_tests {
    use super::parse_fragment;
    use crate::parser::FragmentContext;

    #[test]
    fn extend_method_context_allows_reserved_names() {
        // `perform` is a reserved keyword at the top level but a valid
        // method name inside extend blocks; an extend-method fragment
        // must re-parse under those rules.
        let src = "fn perform(self, op) {\n    1\n}";
        assert!(parse_fragment("test", src, FragmentContext::TopLevel).is_err());
        assert!(parse_fragment("test", src, FragmentContext::ExtendMethod).is_ok());
    }

    #[test]
    fn top_level_context_stays_strict() {
        // Ordinary fragments are unaffected by the context machinery.
        let src = "fn add(a, b) {\n    a + b\n}";
        assert!(parse_fragment("test", src, FragmentContext::TopLevel).is_ok());
        // And reserved names stay rejected for genuinely top-level code.
        let bad = "fn future(x) { x }";
        assert!(parse_fragment("test", bad, FragmentContext::TopLevel).is_err());
    }

    #[test]
    fn handle_is_contextual_everywhere() {
        // `handle` is no longer reserved as a fn name in ANY context —
        // the block form requires the full `handle <T> ... with` shape.
        let src = "fn handle(self, op, resume) {\n    resume(1)\n}";
        assert!(parse_fragment("test", src, FragmentContext::TopLevel).is_ok());
        assert!(parse_fragment("test", src, FragmentContext::ExtendMethod).is_ok());
    }
}
