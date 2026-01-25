use crate::{
    CommandLineArguments,
    ast::{Ast, Pattern, TokenRange},
    backend::{Backend, CodegenBackend},
    code_memory::CodeAllocator,
    debug_only,
    gc::{StackMap, StackMapDetails},
    get_runtime,
    ir::{StringValue, Value},
    macro_expander::MacroExpander,
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
    sync::{Arc, Mutex},
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
    Internal(String),
    UnquoteOutsideQuote {
        position: usize,
    },
    UnquoteSpliceOutsideQuote {
        position: usize,
    },
    MacroNotExpanded {
        name: String,
    },
    /// Macro expansion failed
    MacroExpansionError {
        name: String,
        message: String,
    },
    /// Macro returned invalid type
    MacroInvalidReturn {
        name: String,
        expected: String,
        got: String,
    },
    /// Error compiling macro body
    MacroCompilationError {
        name: String,
        inner: String,
    },
    /// Runtime error during macro execution
    MacroRuntimeError {
        name: String,
        message: String,
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
                write!(f, "Cannot resolve struct: {}", struct_name)
            }
            CompileError::PropertyCacheFull => write!(f, "Property look up cache is full"),
            CompileError::MemoryMapping(msg) => write!(f, "Memory mapping error: {}", msg),
            CompileError::ParseError(e) => write!(f, "Parse error: {}", e),
            CompileError::FunctionNotFound { function_name } => {
                write!(f, "Function not found: {}", function_name)
            }
            CompileError::InvalidFunctionPointer { function_name } => {
                write!(f, "Invalid function pointer for: {}", function_name)
            }
            CompileError::PathConversion { path } => {
                write!(f, "Failed to convert path to string: {}", path)
            }
            CompileError::UndefinedVariable { name } => {
                write!(f, "Undefined variable: {}", name)
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
            CompileError::Internal(msg) => {
                write!(f, "Internal error: {}", msg)
            }
            CompileError::UnquoteOutsideQuote { position } => {
                write!(
                    f,
                    "Unquote (~) used outside of quote at position {}",
                    position
                )
            }
            CompileError::UnquoteSpliceOutsideQuote { position } => {
                write!(
                    f,
                    "Unquote-splice (~@) used outside of quote at position {}",
                    position
                )
            }
            CompileError::MacroNotExpanded { name } => {
                write!(f, "Macro '{}' was not expanded before compilation", name)
            }
            CompileError::MacroExpansionError { name, message } => {
                write!(f, "Macro expansion error in '{}': {}", name, message)
            }
            CompileError::MacroInvalidReturn {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Macro '{}' returned invalid type: expected {}, got {}",
                    name, expected, got
                )
            }
            CompileError::MacroCompilationError { name, inner } => {
                write!(f, "Error compiling macro '{}': {}", name, inner)
            }
            CompileError::MacroRuntimeError { name, message } => {
                write!(f, "Runtime error in macro '{}': {}", name, message)
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

pub struct Compiler {
    pub code_allocator: CodeAllocator,
    pub property_look_up_cache: MmapMut,
    pub command_line_arguments: CommandLineArguments,
    pub stack_map: StackMap,
    pub pause_atom_ptr: Option<usize>,
    pub property_look_up_cache_offset: usize,
    pub compiled_file_cache: HashSet<String>,
    pub diagnostic_store: Arc<Mutex<DiagnosticStore>>,
    /// Cache for protocol dispatch inline caching
    /// Layout per entry: [type_id (8 bytes), fn_ptr (8 bytes)]
    pub protocol_dispatch_cache: MmapMut,
    pub protocol_dispatch_cache_offset: usize,
    /// Multi-arity function metadata for static dispatch
    pub multi_arity_functions: HashMap<String, MultiArityInfo>,
}

impl Compiler {
    /// Create a new Compiler instance for the current thread.
    pub fn new(
        command_line_arguments: CommandLineArguments,
        diagnostic_store: Arc<Mutex<DiagnosticStore>>,
    ) -> Result<Self, CompileError> {
        Ok(Compiler {
            code_allocator: CodeAllocator::new(),
            property_look_up_cache: MmapOptions::new(MmapOptions::page_size())
                .map_err(|e| CompileError::MemoryMapping(format!("Failed to create mmap: {}", e)))?
                .map_mut()
                .map_err(|e| CompileError::MemoryMapping(format!("Failed to map mmap: {}", e)))?
                .make_mut()
                .map_err(|(_map, e)| {
                    CompileError::MemoryMapping(format!("Failed to make mutable: {}", e))
                })?,
            property_look_up_cache_offset: 0,
            protocol_dispatch_cache: MmapOptions::new(MmapOptions::page_size())
                .map_err(|e| CompileError::MemoryMapping(format!("Failed to create mmap: {}", e)))?
                .map_mut()
                .map_err(|e| CompileError::MemoryMapping(format!("Failed to map mmap: {}", e)))?
                .make_mut()
                .map_err(|(_map, e)| {
                    CompileError::MemoryMapping(format!("Failed to make mutable: {}", e))
                })?,
            protocol_dispatch_cache_offset: 0,
            command_line_arguments,
            stack_map: StackMap::new(),
            pause_atom_ptr: None,
            compiled_file_cache: HashSet::new(),
            diagnostic_store,
            multi_arity_functions: HashMap::new(),
        })
    }

    pub fn reset(&mut self) -> Result<(), CompileError> {
        self.code_allocator = CodeAllocator::new();
        self.property_look_up_cache = MmapOptions::new(MmapOptions::page_size())
            .map_err(|e| CompileError::MemoryMapping(format!("Failed to create mmap: {}", e)))?
            .map_mut()
            .map_err(|e| CompileError::MemoryMapping(format!("Failed to map mmap: {}", e)))?
            .make_mut()
            .map_err(|(_map, e)| {
                CompileError::MemoryMapping(format!("Failed to make mmap mutable: {}", e))
            })?;
        self.property_look_up_cache_offset = 0;
        self.protocol_dispatch_cache = MmapOptions::new(MmapOptions::page_size())
            .map_err(|e| CompileError::MemoryMapping(format!("Failed to create mmap: {}", e)))?
            .map_mut()
            .map_err(|e| CompileError::MemoryMapping(format!("Failed to map mmap: {}", e)))?
            .make_mut()
            .map_err(|(_map, e)| {
                CompileError::MemoryMapping(format!("Failed to make mmap mutable: {}", e))
            })?;
        self.protocol_dispatch_cache_offset = 0;
        self.stack_map = StackMap::new();
        self.pause_atom_ptr = None;
        self.compiled_file_cache.clear();
        self.multi_arity_functions.clear();
        // If lock is poisoned, we can still clear by ignoring the error
        if let Ok(mut store) = self.diagnostic_store.lock() {
            store.clear_all();
        }
        Ok(())
    }

    pub fn get_pause_atom(&self) -> usize {
        self.pause_atom_ptr.unwrap_or(0)
    }

    pub fn set_pause_atom_ptr(&mut self, pointer: usize) {
        self.pause_atom_ptr = Some(pointer);
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
        // For REPL/eval, we use "repl" as the file name for diagnostics.
        // Each eval replaces the previous "repl" diagnostics.
        let mut parser = Parser::new("".to_string(), string.to_string())?;
        let ast = parser.parse()?;
        let token_line_column_map = parser.get_token_line_column_map();

        // Macro expansion phase
        let mut expander = MacroExpander::new();
        let ast = expander.collect_macros(&ast);
        // Compile procedural macros before expansion
        expander
            .compile_procedural_macros(self)
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;
        let ast = expander
            .expand(&ast)
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        let (top_level, diagnostics) = self.compile_ast(
            ast,
            Some("REPL_FUNCTION".to_string()),
            "repl",
            token_line_column_map,
        )?;

        // Store diagnostics for "repl" (replaces previous REPL diagnostics)
        if let Ok(mut store) = self.diagnostic_store.lock() {
            store.set_file_diagnostics("repl".to_string(), diagnostics);
        }

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

        if self.command_line_arguments.print_parse {
            println!("{:#?}", ast);
        }

        if self.command_line_arguments.print_ast {
            println!("{:#?}", ast);
        }

        if self.command_line_arguments.show_times {
            println!("Parse time: {:?}", parse_time.elapsed());
        }

        // Macro expansion phase: collect and expand macros before compilation
        let mut expander = MacroExpander::new();
        let ast = expander.collect_macros(&ast);
        // Compile procedural macros before expansion
        expander
            .compile_procedural_macros(self)
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;
        let ast = expander
            .expand(&ast)
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        let mut top_levels_to_run = self.compile_dependencies(&ast, file_name)?;

        let (top_level, diagnostics) =
            self.compile_ast(ast, None, file_name, token_line_column_map)?;
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

    pub fn compile_dependencies(
        &mut self,
        ast: &crate::ast::Ast,
        source_file: &str,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let mut top_levels_to_run = vec![];
        for import in ast.imports() {
            let (name, _alias) = self.extract_import(&import)?;
            if name.starts_with("beagle.") {
                continue;
            }
            let file_name = self.get_file_name_from_import(name, source_file)?;
            let top_level = self.compile(&file_name)?;
            top_levels_to_run.extend(top_level);
        }
        Ok(top_levels_to_run)
    }

    pub fn compile_ast(
        &mut self,
        ast: crate::ast::Ast,
        fn_name: Option<String>,
        file_name: &str,
        token_line_column_map: Vec<(usize, usize)>,
    ) -> Result<(Option<String>, Vec<Diagnostic>), Box<dyn Error>> {
        let (mut ir, token_map, diagnostics) =
            ast.compile(self, file_name, token_line_column_map)?;
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
            let mut backend = ir.compile(backend, error_fn_pointer);
            let _token_map = ir.ir_range_to_token_range.clone();
            let max_locals = backend.max_locals() as usize;
            let _function_pointer =
                self.upsert_function(Some(&top_level_name), &mut backend, max_locals, 0, false, 0)?;
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
        self.property_look_up_cache_offset += 3 * 8;
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

    // TODO: All of this seems bad
    pub fn add_struct(&mut self, s: Struct) {
        let runtime = get_runtime().get_mut();
        runtime.add_struct(s);
    }

    pub fn add_enum(&mut self, e: Enum) {
        let runtime = get_runtime().get_mut();
        runtime.add_enum(e);
    }

    pub fn get_enum(&self, name: &str) -> Option<&Enum> {
        let runtime = get_runtime().get_mut();
        runtime.get_enum(name)
    }

    pub fn get_struct(&self, name: &str) -> Option<(usize, &Struct)> {
        let runtime = get_runtime().get_mut();
        runtime.get_struct(name)
    }

    /// Register a mapping from enum variant struct_id to enum name
    /// Used by effect handlers to determine which handler to call for a `perform` value
    pub fn register_enum_variant(&mut self, struct_id: usize, enum_name: String) {
        let runtime = get_runtime().get_mut();
        runtime.register_enum_variant(struct_id, enum_name);
    }

    pub fn is_inline_primitive_function(&self, name: &str) -> bool {
        name.starts_with("beagle.primitive/")
    }

    pub fn get_file_name_from_import(
        &self,
        import_name: String,
        source_file: &str,
    ) -> Result<String, Box<dyn Error>> {
        // 1. Try relative to source file (PRIORITY 1)
        let source_dir = Path::new(source_file)
            .parent()
            .ok_or("Invalid source file path")?;

        let relative_path = source_dir.join(format!("{}.bg", import_name));
        if relative_path.exists() {
            return relative_path
                .to_str()
                .ok_or_else(|| CompileError::PathConversion {
                    path: format!("{:?}", relative_path),
                })
                .map(|s| s.to_string())
                .map_err(|e| e.into());
        }

        // 2. Try standard library (PRIORITY 2)
        let mut exe_path = env::current_exe()?;
        exe_path = exe_path
            .parent()
            .ok_or("Cannot get parent of executable path")?
            .to_path_buf();

        let stdlib_path = exe_path.join(format!("standard-library/{}.bg", import_name));
        if stdlib_path.exists() {
            return stdlib_path
                .to_str()
                .ok_or_else(|| CompileError::PathConversion {
                    path: format!("{:?}", stdlib_path),
                })
                .map(|s| s.to_string())
                .map_err(|e| e.into());
        }

        // Try one level up (for development - cargo run from project root)
        if let Some(parent) = exe_path.parent() {
            let parent_stdlib_path = parent.join(format!("standard-library/{}.bg", import_name));
            if parent_stdlib_path.exists() {
                return parent_stdlib_path
                    .to_str()
                    .ok_or_else(|| CompileError::PathConversion {
                        path: format!("{:?}", parent_stdlib_path),
                    })
                    .map(|s| s.to_string())
                    .map_err(|e| e.into());
            }
        }

        // Try two levels up (for cargo run - target/debug -> target -> root)
        if let Some(parent) = exe_path.parent()
            && let Some(grandparent) = parent.parent()
        {
            let grandparent_stdlib_path =
                grandparent.join(format!("standard-library/{}.bg", import_name));
            if grandparent_stdlib_path.exists() {
                return grandparent_stdlib_path
                    .to_str()
                    .ok_or_else(|| CompileError::PathConversion {
                        path: format!("{:?}", grandparent_stdlib_path),
                    })
                    .map(|s| s.to_string())
                    .map_err(|e| e.into());
            }
        }

        // Try three levels up (for cargo test - target/debug/deps -> target/debug -> target -> root)
        if let Some(parent) = exe_path.parent()
            && let Some(grandparent) = parent.parent()
            && let Some(great_grandparent) = grandparent.parent()
        {
            let great_grandparent_stdlib_path =
                great_grandparent.join(format!("standard-library/{}.bg", import_name));
            if great_grandparent_stdlib_path.exists() {
                return great_grandparent_stdlib_path
                    .to_str()
                    .ok_or_else(|| CompileError::PathConversion {
                        path: format!("{:?}", great_grandparent_stdlib_path),
                    })
                    .map(|s| s.to_string())
                    .map_err(|e| e.into());
            }
        }

        Err(format!(
            "Could not find module '{}' (searched: relative to {}, standard-library/)",
            import_name, source_file
        )
        .into())
    }

    pub fn extract_import(
        &self,
        import: &crate::ast::Ast,
    ) -> Result<(String, String), CompileError> {
        match import {
            crate::ast::Ast::Import {
                library_name,
                alias,
                ..
            } => {
                let library_name = library_name.to_string().replace("\"", "");
                let alias = alias.as_ref().get_string();
                Ok((library_name, alias))
            }
            _ => Err(CompileError::ParseError(
                crate::parser::ParseError::InvalidDeclaration {
                    message: "Expected import AST node".to_string(),
                    position: 0,
                },
            )),
        }
    }

    pub fn get_struct_by_id(&self, struct_id: usize) -> Option<&Struct> {
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

    pub fn upsert_function<B: CodegenBackend>(
        &mut self,
        function_name: Option<&str>,
        backend: &mut B,
        max_locals: usize,
        number_of_args: usize,
        is_variadic: bool,
        min_args: usize,
    ) -> Result<usize, Box<dyn Error>> {
        if let Some(name) = function_name {
            backend.set_function_name(name);
        }
        let code = backend.compile_to_bytes();
        let pointer = self.add_code(&code)?;
        let runtime = get_runtime().get_mut();
        // TODO: Before this we did some weird stuff of mapping over the stack_map details
        // and I don't remember why
        let stack_map = backend
            .translate_stack_map(pointer as usize)
            .iter()
            .map(|(key, value)| {
                (
                    *key,
                    StackMapDetails {
                        function_name: function_name.map(|x| x.to_string()),
                        current_stack_size: *value,
                        number_of_locals: backend.max_locals() as usize,
                        max_stack_size: backend.max_stack_size() as usize,
                    },
                )
            })
            .collect();
        runtime.upsert_function(
            function_name,
            pointer,
            code.len(),
            max_locals,
            stack_map,
            number_of_args,
            is_variadic,
            min_args,
        )
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
        let runtime = get_runtime().get_mut();
        let stack_map = vec![];
        runtime.upsert_function(
            function_name,
            pointer,
            code.len(),
            max_locals,
            stack_map,
            number_of_args,
            is_variadic,
            min_args,
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

    pub fn compile_protocol_method(
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

        let current_namespace_id = self.current_namespace_id();
        let protocol_namespace_id = runtime
            .get_namespace_id(protocol_namespace)
            .ok_or_else(|| format!("Protocol namespace not found: {}", protocol_namespace))?;
        self.set_current_namespace(protocol_namespace_id);

        // Try optimized dispatch first
        let body = if let Some(optimized) = self.build_optimized_dispatch(
            &protocol_name,
            &method_name,
            default_method.as_ref(),
            args_as_strings.clone(),
        ) {
            vec![optimized]
        } else {
            // Fall back to if-chain
            vec![self.build_method_if_chain(
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
            }],
            token_range: TokenRange::new(0, 0),
        };
        // Ignore diagnostics from reify method compilation - these are internal
        let _ = self.compile_ast(ast, None, "test", vec![])?;
        self.code_allocator.make_executable();
        self.set_current_namespace(current_namespace_id);
        Ok(())
    }
}

impl fmt::Debug for Compiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: Make this better
        f.debug_struct("Compiler").finish()
    }
}

// Thread-local compiler architecture:
// Each thread now has its own Compiler instance stored in thread-local storage (see runtime.rs).
// This follows Chez Scheme's model where multiple threads can compile concurrently.
// The old CompilerThread, BlockingSender, BlockingReceiver types have been removed.
