use crate::{
    ast::{Ast, FieldPattern, MatchArm, Pattern, StringInterpolationPart, TokenRange},
    compiler::{CompileError, Compiler},
    get_runtime,
    ir::Condition,
    types::{BuiltInTypes, HeapObject},
};
use std::collections::{HashMap, HashSet};

/// A registered macro definition
#[derive(Debug, Clone)]
pub struct MacroDef {
    pub name: String,
    pub args: Vec<Pattern>,
    pub rest_param: Option<String>,
    pub body: Vec<Ast>,
}

/// The macro expander handles compile-time macro expansion.
///
/// Workflow:
/// 1. First pass: collect all macro definitions from the AST
/// 2. Second pass: expand macro calls, replacing them with their expanded forms
///
/// Macros receive unevaluated AST and return transformed AST.
///
/// Current status: Template-based macros (quote/unquote) are fully supported.
/// Full procedural macros (executing arbitrary Beagle code at compile time)
/// are now supported - macro bodies are compiled as functions and called
/// with AST values as arguments.
#[allow(dead_code)]
pub struct MacroExpander {
    /// Registry of macro definitions by name
    macros: HashMap<String, MacroDef>,
    /// Counter for generating unique symbols (hygiene)
    gensym_counter: usize,
    /// Current namespace for resolving macro names
    current_namespace: String,
    /// Cache mapping struct names to their struct IDs
    struct_id_cache: HashMap<String, usize>,
    /// Compiled macro function names (for procedural macros)
    compiled_macros: HashMap<String, String>,
    /// Namespace aliases (e.g., "ast" -> "beagle.ast")
    namespace_aliases: HashMap<String, String>,
}

impl MacroExpander {
    pub fn new() -> Self {
        MacroExpander {
            macros: HashMap::new(),
            gensym_counter: 0,
            current_namespace: String::new(),
            struct_id_cache: HashMap::new(),
            compiled_macros: HashMap::new(),
            namespace_aliases: HashMap::new(),
        }
    }

    /// Check if a macro is "simple" (body is just `quote { ... }`) or procedural
    fn is_simple_macro(macro_def: &MacroDef) -> bool {
        // A macro is simple if its body is a single Quote expression
        if macro_def.body.len() == 1 {
            matches!(&macro_def.body[0], Ast::Quote { .. })
        } else {
            false
        }
    }

    /// Resolve namespace aliases in a function name
    /// e.g., "ast/make-identifier" with alias ast->beagle.ast becomes "beagle.ast/make-identifier"
    /// Note: This infrastructure is for future procedural macro support.
    #[allow(dead_code)]
    fn resolve_namespace_alias(&self, name: &str) -> String {
        if let Some(slash_pos) = name.find('/') {
            let prefix = &name[..slash_pos];
            let suffix = &name[slash_pos..];
            if let Some(full_ns) = self.namespace_aliases.get(prefix) {
                return format!("{}{}", full_ns, suffix);
            }
        }
        name.to_string()
    }

    /// Generate Beagle code for a macro function
    fn generate_macro_function_code(&self, macro_def: &MacroDef) -> String {
        let func_name = format!("__macro__{}", macro_def.name.replace('-', "_"));

        // Generate argument list
        let args: Vec<String> = macro_def
            .args
            .iter()
            .filter_map(|p| {
                if let Pattern::Identifier { name, .. } = p {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        let args_str = args.join(", ");

        // Generate the function body from the macro body AST
        // Don't resolve aliases - instead we'll include the necessary imports
        let body_str = self.ast_to_string_no_resolve(&macro_def.body);

        // Generate import statements for all used aliases
        let mut imports = String::new();
        for (alias, full_name) in &self.namespace_aliases {
            imports.push_str(&format!("import \"{}\" as {}\n", full_name, alias));
        }

        // Include namespace declaration and imports for the generated code
        format!(
            "namespace __macros\n\n{}\nfn {}({}) {{ {} }}",
            imports, func_name, args_str, body_str
        )
    }

    /// Convert AST to string without resolving namespace aliases
    fn ast_to_string_no_resolve(&self, body: &[Ast]) -> String {
        body.iter()
            .map(|ast| self.single_ast_to_string_no_resolve(ast))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Convert a single AST to string without resolving namespace aliases
    /// This version keeps namespace aliases as-is (e.g., "ast/" instead of "beagle.ast/")
    #[allow(dead_code)]
    fn single_ast_to_string_no_resolve(&self, ast: &Ast) -> String {
        match ast {
            Ast::Call { name, args, .. } => {
                let args_str: Vec<String> = args
                    .iter()
                    .map(|a| self.single_ast_to_string_no_resolve(a))
                    .collect();
                // Don't resolve the alias - keep the original name
                format!("{}({})", name, args_str.join(", "))
            }
            Ast::Let { pattern, value, .. } => {
                format!(
                    "let {} = {}",
                    self.pattern_to_string(pattern),
                    self.single_ast_to_string_no_resolve(value)
                )
            }
            Ast::LetMut { pattern, value, .. } => {
                format!(
                    "let mut {} = {}",
                    self.pattern_to_string(pattern),
                    self.single_ast_to_string_no_resolve(value)
                )
            }
            Ast::Add { left, right, .. } => {
                format!(
                    "({} + {})",
                    self.single_ast_to_string_no_resolve(left),
                    self.single_ast_to_string_no_resolve(right)
                )
            }
            Ast::Sub { left, right, .. } => {
                format!(
                    "({} - {})",
                    self.single_ast_to_string_no_resolve(left),
                    self.single_ast_to_string_no_resolve(right)
                )
            }
            Ast::Mul { left, right, .. } => {
                format!(
                    "({} * {})",
                    self.single_ast_to_string_no_resolve(left),
                    self.single_ast_to_string_no_resolve(right)
                )
            }
            Ast::Div { left, right, .. } => {
                format!(
                    "({} / {})",
                    self.single_ast_to_string_no_resolve(left),
                    self.single_ast_to_string_no_resolve(right)
                )
            }
            Ast::Array { array, .. } => {
                let elems: Vec<String> = array
                    .iter()
                    .map(|e| self.single_ast_to_string_no_resolve(e))
                    .collect();
                format!("[{}]", elems.join(", "))
            }
            Ast::If {
                condition,
                then,
                else_,
                ..
            } => {
                let then_str = self.ast_to_string_no_resolve(then);
                let else_str = self.ast_to_string_no_resolve(else_);
                format!(
                    "if {} {{ {} }} else {{ {} }}",
                    self.single_ast_to_string_no_resolve(condition),
                    then_str,
                    else_str
                )
            }
            Ast::Block { statements, .. } => {
                let stmts_str: Vec<String> = statements
                    .iter()
                    .map(|s| self.single_ast_to_string_no_resolve(s))
                    .collect();
                format!("{{ {} }}", stmts_str.join("\n"))
            }
            Ast::While {
                condition, body, ..
            } => {
                format!(
                    "while {} {{ {} }}",
                    self.single_ast_to_string_no_resolve(condition),
                    self.ast_to_string_no_resolve(body)
                )
            }
            Ast::Quote { body, .. } => {
                format!("quote {{ {} }}", self.single_ast_to_string_no_resolve(body))
            }
            Ast::Unquote { body, .. } => {
                format!("~{}", self.single_ast_to_string_no_resolve(body))
            }
            // For simple literals and other cases, use the regular method
            _ => self.single_ast_to_string(ast),
        }
    }

    /// Convert AST back to Beagle source code (for macro function generation)
    #[allow(dead_code)]
    fn ast_to_string(&self, body: &[Ast]) -> String {
        body.iter()
            .map(|ast| self.single_ast_to_string(ast))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Convert a single AST node to Beagle source code
    #[allow(dead_code)]
    fn single_ast_to_string(&self, ast: &Ast) -> String {
        match ast {
            Ast::IntegerLiteral(n, _) => n.to_string(),
            Ast::FloatLiteral(f, _) => f.to_string(),
            Ast::String(s, _) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
            Ast::Identifier(name, _) => name.clone(),
            Ast::True(_) => "true".to_string(),
            Ast::False(_) => "false".to_string(),
            Ast::Null(_) => "null".to_string(),
            Ast::Keyword(kw, _) => format!(":{}", kw),

            Ast::Add { left, right, .. } => {
                format!(
                    "({} + {})",
                    self.single_ast_to_string(left),
                    self.single_ast_to_string(right)
                )
            }
            Ast::Sub { left, right, .. } => {
                format!(
                    "({} - {})",
                    self.single_ast_to_string(left),
                    self.single_ast_to_string(right)
                )
            }
            Ast::Mul { left, right, .. } => {
                format!(
                    "({} * {})",
                    self.single_ast_to_string(left),
                    self.single_ast_to_string(right)
                )
            }
            Ast::Div { left, right, .. } => {
                format!(
                    "({} / {})",
                    self.single_ast_to_string(left),
                    self.single_ast_to_string(right)
                )
            }
            Ast::Modulo { left, right, .. } => {
                format!(
                    "({} % {})",
                    self.single_ast_to_string(left),
                    self.single_ast_to_string(right)
                )
            }

            Ast::Condition {
                operator,
                left,
                right,
                ..
            } => {
                let op_str = match operator {
                    Condition::LessThan => "<",
                    Condition::LessThanOrEqual => "<=",
                    Condition::GreaterThan => ">",
                    Condition::GreaterThanOrEqual => ">=",
                    Condition::Equal => "==",
                    Condition::NotEqual => "!=",
                };
                format!(
                    "({} {} {})",
                    self.single_ast_to_string(left),
                    op_str,
                    self.single_ast_to_string(right)
                )
            }

            Ast::And { left, right, .. } => {
                format!(
                    "({} && {})",
                    self.single_ast_to_string(left),
                    self.single_ast_to_string(right)
                )
            }
            Ast::Or { left, right, .. } => {
                format!(
                    "({} || {})",
                    self.single_ast_to_string(left),
                    self.single_ast_to_string(right)
                )
            }

            Ast::Let { pattern, value, .. } => {
                format!(
                    "let {} = {}",
                    self.pattern_to_string(pattern),
                    self.single_ast_to_string(value)
                )
            }
            Ast::LetMut { pattern, value, .. } => {
                format!(
                    "let mut {} = {}",
                    self.pattern_to_string(pattern),
                    self.single_ast_to_string(value)
                )
            }

            Ast::Assignment { name, value, .. } => {
                format!(
                    "{} = {}",
                    self.single_ast_to_string(name),
                    self.single_ast_to_string(value)
                )
            }

            Ast::If {
                condition,
                then,
                else_,
                ..
            } => {
                let then_str = self.ast_to_string(then);
                let else_str = self.ast_to_string(else_);
                format!(
                    "if {} {{ {} }} else {{ {} }}",
                    self.single_ast_to_string(condition),
                    then_str,
                    else_str
                )
            }

            Ast::Block { statements, .. } => {
                let stmts_str: Vec<String> = statements
                    .iter()
                    .map(|s| self.single_ast_to_string(s))
                    .collect();
                format!("{{ {} }}", stmts_str.join("\n"))
            }

            Ast::While {
                condition, body, ..
            } => {
                format!(
                    "while {} {{ {} }}",
                    self.single_ast_to_string(condition),
                    self.ast_to_string(body)
                )
            }

            Ast::Loop { body, .. } => {
                format!("loop {{ {} }}", self.ast_to_string(body))
            }

            Ast::For {
                binding,
                collection,
                body,
                ..
            } => {
                format!(
                    "for {} in {} {{ {} }}",
                    binding,
                    self.single_ast_to_string(collection),
                    self.ast_to_string(body)
                )
            }

            Ast::Break { value, .. } => {
                format!("break {}", self.single_ast_to_string(value))
            }
            Ast::Continue { .. } => "continue".to_string(),

            Ast::Call { name, args, .. } => {
                let args_str: Vec<String> =
                    args.iter().map(|a| self.single_ast_to_string(a)).collect();
                let resolved_name = self.resolve_namespace_alias(name);
                format!("{}({})", resolved_name, args_str.join(", "))
            }

            Ast::Array { array, .. } => {
                let elems: Vec<String> =
                    array.iter().map(|e| self.single_ast_to_string(e)).collect();
                format!("[{}]", elems.join(", "))
            }

            Ast::PropertyAccess {
                object, property, ..
            } => {
                format!(
                    "{}.{}",
                    self.single_ast_to_string(object),
                    self.single_ast_to_string(property)
                )
            }

            Ast::IndexOperator { array, index, .. } => {
                format!(
                    "{}[{}]",
                    self.single_ast_to_string(array),
                    self.single_ast_to_string(index)
                )
            }

            Ast::Quote { body, .. } => {
                format!("quote {{ {} }}", self.single_ast_to_string(body))
            }

            Ast::Unquote { body, .. } => {
                format!("~{}", self.single_ast_to_string(body))
            }

            Ast::Match { value, arms, .. } => {
                let arms_str: Vec<String> = arms
                    .iter()
                    .map(|arm| {
                        let pattern_str = self.pattern_to_string(&arm.pattern);
                        let body_str = self.ast_to_string(&arm.body);
                        format!("{} => {{ {} }}", pattern_str, body_str)
                    })
                    .collect();
                format!(
                    "match {} {{ {} }}",
                    self.single_ast_to_string(value),
                    arms_str.join(", ")
                )
            }

            // For other nodes, return a placeholder
            _ => format!(
                "/* unsupported AST node: {:?} */",
                std::mem::discriminant(ast)
            ),
        }
    }

    /// Convert a pattern to string
    #[allow(dead_code)]
    fn pattern_to_string(&self, pattern: &Pattern) -> String {
        match pattern {
            Pattern::Identifier { name, .. } => name.clone(),
            Pattern::Wildcard { .. } => "_".to_string(),
            Pattern::Literal { value, .. } => self.single_ast_to_string(value),
            Pattern::Array { elements, rest, .. } => {
                let elems: Vec<String> =
                    elements.iter().map(|p| self.pattern_to_string(p)).collect();
                if let Some(r) = rest {
                    format!("[{}, ...{}]", elems.join(", "), self.pattern_to_string(r))
                } else {
                    format!("[{}]", elems.join(", "))
                }
            }
            Pattern::EnumVariant {
                enum_name,
                variant_name,
                fields,
                ..
            } => {
                if fields.is_empty() {
                    format!("{}.{}", enum_name, variant_name)
                } else {
                    let fields_str: Vec<String> = fields
                        .iter()
                        .map(|f| {
                            if let Some(binding) = &f.binding_name {
                                format!("{}: {}", f.field_name, binding)
                            } else {
                                f.field_name.clone()
                            }
                        })
                        .collect();
                    format!(
                        "{}.{} {{ {} }}",
                        enum_name,
                        variant_name,
                        fields_str.join(", ")
                    )
                }
            }
            Pattern::Struct { name, fields, .. } => {
                let fields_str: Vec<String> = fields
                    .iter()
                    .map(|f| {
                        if let Some(binding) = &f.binding_name {
                            format!("{}: {}", f.field_name, binding)
                        } else {
                            f.field_name.clone()
                        }
                    })
                    .collect();
                format!("{} {{ {} }}", name, fields_str.join(", "))
            }
            Pattern::Map { .. } => "/* map pattern */".to_string(),
        }
    }

    /// Set the current namespace for macro resolution
    #[allow(dead_code)]
    pub fn set_namespace(&mut self, namespace: &str) {
        self.current_namespace = namespace.to_string();
    }

    /// Generate a unique symbol for hygiene
    pub fn gensym(&mut self, base: &str) -> String {
        let id = self.gensym_counter;
        self.gensym_counter += 1;
        format!("{}__gensym__{}", base, id)
    }

    // ========================================================================
    // Hygiene support
    // ========================================================================

    /// Collect all free identifiers from an AST (identifiers that are referenced but not bound)
    fn collect_free_identifiers(&self, ast: &Ast) -> HashSet<String> {
        let mut free = HashSet::new();
        self.collect_free_identifiers_inner(ast, &mut HashSet::new(), &mut free);
        free
    }

    /// Helper to collect free identifiers, tracking bound names
    fn collect_free_identifiers_inner(
        &self,
        ast: &Ast,
        bound: &mut HashSet<String>,
        free: &mut HashSet<String>,
    ) {
        match ast {
            Ast::Identifier(name, _) => {
                if !bound.contains(name) {
                    free.insert(name.clone());
                }
            }
            Ast::Let { pattern, value, .. } => {
                // First collect from value (before the binding is in scope)
                self.collect_free_identifiers_inner(value, bound, free);
                // Then add the bound name
                if let Pattern::Identifier { name, .. } = pattern {
                    bound.insert(name.clone());
                }
            }
            Ast::LetMut { pattern, value, .. } => {
                self.collect_free_identifiers_inner(value, bound, free);
                if let Pattern::Identifier { name, .. } = pattern {
                    bound.insert(name.clone());
                }
            }
            Ast::Function { args, body, .. } => {
                // Function args are bound in the body
                let mut inner_bound = bound.clone();
                for arg in args {
                    if let Pattern::Identifier { name, .. } = arg {
                        inner_bound.insert(name.clone());
                    }
                }
                for stmt in body {
                    self.collect_free_identifiers_inner(stmt, &mut inner_bound, free);
                }
            }
            Ast::If {
                condition,
                then,
                else_,
                ..
            } => {
                self.collect_free_identifiers_inner(condition, bound, free);
                for stmt in then {
                    self.collect_free_identifiers_inner(stmt, bound, free);
                }
                for stmt in else_ {
                    self.collect_free_identifiers_inner(stmt, bound, free);
                }
            }
            Ast::Block { statements, .. } => {
                for stmt in statements {
                    self.collect_free_identifiers_inner(stmt, bound, free);
                }
            }
            Ast::Call { args, .. } => {
                for arg in args {
                    self.collect_free_identifiers_inner(arg, bound, free);
                }
            }
            Ast::Add { left, right, .. }
            | Ast::Sub { left, right, .. }
            | Ast::Mul { left, right, .. }
            | Ast::Div { left, right, .. }
            | Ast::Modulo { left, right, .. }
            | Ast::And { left, right, .. }
            | Ast::Or { left, right, .. }
            | Ast::Condition { left, right, .. } => {
                self.collect_free_identifiers_inner(left, bound, free);
                self.collect_free_identifiers_inner(right, bound, free);
            }
            Ast::Array { array, .. } => {
                for elem in array {
                    self.collect_free_identifiers_inner(elem, bound, free);
                }
            }
            Ast::Match { value, arms, .. } => {
                self.collect_free_identifiers_inner(value, bound, free);
                for arm in arms {
                    let mut arm_bound = bound.clone();
                    self.add_pattern_bindings(&arm.pattern, &mut arm_bound);
                    for stmt in &arm.body {
                        self.collect_free_identifiers_inner(stmt, &mut arm_bound, free);
                    }
                }
            }
            Ast::While {
                condition, body, ..
            } => {
                self.collect_free_identifiers_inner(condition, bound, free);
                for stmt in body {
                    self.collect_free_identifiers_inner(stmt, bound, free);
                }
            }
            Ast::Loop { body, .. } => {
                for stmt in body {
                    self.collect_free_identifiers_inner(stmt, bound, free);
                }
            }
            Ast::For {
                binding,
                collection,
                body,
                ..
            } => {
                self.collect_free_identifiers_inner(collection, bound, free);
                let mut inner_bound = bound.clone();
                inner_bound.insert(binding.clone());
                for stmt in body {
                    self.collect_free_identifiers_inner(stmt, &mut inner_bound, free);
                }
            }
            Ast::PropertyAccess { object, .. } => {
                self.collect_free_identifiers_inner(object, bound, free);
            }
            Ast::IndexOperator { array, index, .. } => {
                self.collect_free_identifiers_inner(array, bound, free);
                self.collect_free_identifiers_inner(index, bound, free);
            }
            Ast::Assignment { name, value, .. } => {
                self.collect_free_identifiers_inner(name, bound, free);
                self.collect_free_identifiers_inner(value, bound, free);
            }
            // Literals and other nodes without identifiers
            _ => {}
        }
    }

    /// Add pattern bindings to the bound set
    fn add_pattern_bindings(&self, pattern: &Pattern, bound: &mut HashSet<String>) {
        match pattern {
            Pattern::Identifier { name, .. } => {
                bound.insert(name.clone());
            }
            Pattern::Array { elements, rest, .. } => {
                for elem in elements {
                    self.add_pattern_bindings(elem, bound);
                }
                if let Some(rest_pat) = rest {
                    self.add_pattern_bindings(rest_pat, bound);
                }
            }
            Pattern::EnumVariant { fields, .. } | Pattern::Struct { fields, .. } => {
                for field in fields {
                    if let Some(binding) = &field.binding_name {
                        bound.insert(binding.clone());
                    } else {
                        bound.insert(field.field_name.clone());
                    }
                }
            }
            Pattern::Map { fields, .. } => {
                for field in fields {
                    bound.insert(field.binding_name.clone());
                }
            }
            _ => {}
        }
    }

    /// Apply hygiene to the expanded AST.
    /// Renames macro-introduced bindings to avoid capturing user variables.
    fn apply_hygiene(
        &mut self,
        ast: Ast,
        user_identifiers: &HashSet<String>,
    ) -> Result<Ast, CompileError> {
        // First pass: collect all macro-introduced bindings that need renaming
        let mut renames: HashMap<String, String> = HashMap::new();
        self.collect_bindings_to_rename(&ast, user_identifiers, &mut renames);

        // Second pass: apply the renames
        if renames.is_empty() {
            Ok(ast)
        } else {
            self.apply_renames(ast, &renames)
        }
    }

    /// Collect bindings that need to be renamed for hygiene
    fn collect_bindings_to_rename(
        &mut self,
        ast: &Ast,
        user_identifiers: &HashSet<String>,
        renames: &mut HashMap<String, String>,
    ) {
        match ast {
            Ast::Let { pattern, value, .. } | Ast::LetMut { pattern, value, .. } => {
                // Check if this binding should be renamed
                if let Pattern::Identifier { name, .. } = pattern {
                    // Rename if the name conflicts with a user identifier (to avoid capture)
                    // Exception: identifiers starting with __ are "anaphoric" - deliberately
                    // exposed by macros for users to reference, so don't rename them
                    if user_identifiers.contains(name)
                        && !renames.contains_key(name)
                        && !name.starts_with("__")
                    {
                        let new_name = self.gensym(name);
                        renames.insert(name.clone(), new_name);
                    }
                }
                self.collect_bindings_to_rename(value, user_identifiers, renames);
            }
            Ast::Function { args, body, .. } => {
                for arg in args {
                    if let Pattern::Identifier { name, .. } = arg
                        && user_identifiers.contains(name)
                        && !renames.contains_key(name)
                        && !name.starts_with("__")
                    {
                        let new_name = self.gensym(name);
                        renames.insert(name.clone(), new_name);
                    }
                }
                for stmt in body {
                    self.collect_bindings_to_rename(stmt, user_identifiers, renames);
                }
            }
            Ast::If {
                condition,
                then,
                else_,
                ..
            } => {
                self.collect_bindings_to_rename(condition, user_identifiers, renames);
                for stmt in then {
                    self.collect_bindings_to_rename(stmt, user_identifiers, renames);
                }
                for stmt in else_ {
                    self.collect_bindings_to_rename(stmt, user_identifiers, renames);
                }
            }
            Ast::Block { statements, .. } => {
                for stmt in statements {
                    self.collect_bindings_to_rename(stmt, user_identifiers, renames);
                }
            }
            Ast::Call { args, .. } => {
                for arg in args {
                    self.collect_bindings_to_rename(arg, user_identifiers, renames);
                }
            }
            Ast::Add { left, right, .. }
            | Ast::Sub { left, right, .. }
            | Ast::Mul { left, right, .. }
            | Ast::Div { left, right, .. } => {
                self.collect_bindings_to_rename(left, user_identifiers, renames);
                self.collect_bindings_to_rename(right, user_identifiers, renames);
            }
            Ast::Array { array, .. } => {
                for elem in array {
                    self.collect_bindings_to_rename(elem, user_identifiers, renames);
                }
            }
            Ast::Match { value, arms, .. } => {
                self.collect_bindings_to_rename(value, user_identifiers, renames);
                for arm in arms {
                    // Match arm patterns introduce bindings
                    self.collect_pattern_bindings_to_rename(
                        &arm.pattern,
                        user_identifiers,
                        renames,
                    );
                    for stmt in &arm.body {
                        self.collect_bindings_to_rename(stmt, user_identifiers, renames);
                    }
                }
            }
            Ast::While {
                condition, body, ..
            } => {
                self.collect_bindings_to_rename(condition, user_identifiers, renames);
                for stmt in body {
                    self.collect_bindings_to_rename(stmt, user_identifiers, renames);
                }
            }
            Ast::Loop { body, .. } => {
                for stmt in body {
                    self.collect_bindings_to_rename(stmt, user_identifiers, renames);
                }
            }
            Ast::For {
                binding,
                collection,
                body,
                ..
            } => {
                if user_identifiers.contains(binding) && !renames.contains_key(binding) {
                    let new_name = self.gensym(binding);
                    renames.insert(binding.clone(), new_name);
                }
                self.collect_bindings_to_rename(collection, user_identifiers, renames);
                for stmt in body {
                    self.collect_bindings_to_rename(stmt, user_identifiers, renames);
                }
            }
            _ => {}
        }
    }

    /// Collect pattern bindings that need renaming
    fn collect_pattern_bindings_to_rename(
        &mut self,
        pattern: &Pattern,
        user_identifiers: &HashSet<String>,
        renames: &mut HashMap<String, String>,
    ) {
        match pattern {
            Pattern::Identifier { name, .. } => {
                // Rename if the name conflicts with a user identifier
                // Exception: identifiers starting with __ are anaphoric (deliberately exposed)
                if user_identifiers.contains(name)
                    && !renames.contains_key(name)
                    && !name.starts_with("__")
                {
                    let new_name = self.gensym(name);
                    renames.insert(name.clone(), new_name);
                }
            }
            Pattern::Array { elements, rest, .. } => {
                for elem in elements {
                    self.collect_pattern_bindings_to_rename(elem, user_identifiers, renames);
                }
                if let Some(rest_pat) = rest {
                    self.collect_pattern_bindings_to_rename(rest_pat, user_identifiers, renames);
                }
            }
            Pattern::EnumVariant { fields, .. } | Pattern::Struct { fields, .. } => {
                for field in fields {
                    let name = field.binding_name.as_ref().unwrap_or(&field.field_name);
                    // Rename if the name conflicts with a user identifier
                    // Exception: identifiers starting with __ are anaphoric
                    if user_identifiers.contains(name)
                        && !renames.contains_key(name)
                        && !name.starts_with("__")
                    {
                        let new_name = self.gensym(name);
                        renames.insert(name.clone(), new_name);
                    }
                }
            }
            _ => {}
        }
    }

    /// Apply renames to the AST
    /// NOTE: We only rename BINDINGS (patterns in let/function/match), NOT references.
    /// This way, user references (from unquote) keep pointing to user's variables,
    /// while macro-introduced bindings get unique names to avoid capture.
    fn apply_renames(
        &self,
        ast: Ast,
        renames: &HashMap<String, String>,
    ) -> Result<Ast, CompileError> {
        match ast {
            // Don't rename plain identifiers - only rename bindings (patterns)
            // This ensures user code references aren't accidentally renamed
            Ast::Identifier(_, _) => Ok(ast),
            Ast::Let {
                pattern,
                value,
                token_range,
            } => {
                let new_pattern = self.apply_renames_to_pattern(pattern, renames);
                let new_value = self.apply_renames(*value, renames)?;
                Ok(Ast::Let {
                    pattern: new_pattern,
                    value: Box::new(new_value),
                    token_range,
                })
            }
            Ast::LetMut {
                pattern,
                value,
                token_range,
            } => {
                let new_pattern = self.apply_renames_to_pattern(pattern, renames);
                let new_value = self.apply_renames(*value, renames)?;
                Ok(Ast::LetMut {
                    pattern: new_pattern,
                    value: Box::new(new_value),
                    token_range,
                })
            }
            Ast::Function {
                name,
                args,
                rest_param,
                body,
                token_range,
            } => {
                let new_args: Vec<Pattern> = args
                    .into_iter()
                    .map(|p| self.apply_renames_to_pattern(p, renames))
                    .collect();
                let new_rest = rest_param.map(|r| renames.get(&r).cloned().unwrap_or(r));
                let new_body: Result<Vec<Ast>, CompileError> = body
                    .into_iter()
                    .map(|s| self.apply_renames(s, renames))
                    .collect();
                Ok(Ast::Function {
                    name,
                    args: new_args,
                    rest_param: new_rest,
                    body: new_body?,
                    token_range,
                })
            }
            Ast::If {
                condition,
                then,
                else_,
                token_range,
            } => {
                let new_cond = self.apply_renames(*condition, renames)?;
                let new_then: Result<Vec<Ast>, CompileError> = then
                    .into_iter()
                    .map(|s| self.apply_renames(s, renames))
                    .collect();
                let new_else: Result<Vec<Ast>, CompileError> = else_
                    .into_iter()
                    .map(|s| self.apply_renames(s, renames))
                    .collect();
                Ok(Ast::If {
                    condition: Box::new(new_cond),
                    then: new_then?,
                    else_: new_else?,
                    token_range,
                })
            }
            Ast::Block {
                statements,
                token_range,
            } => {
                let new_statements: Result<Vec<Ast>, CompileError> = statements
                    .into_iter()
                    .map(|s| self.apply_renames(s, renames))
                    .collect();
                Ok(Ast::Block {
                    statements: new_statements?,
                    token_range,
                })
            }
            Ast::Call {
                name,
                args,
                token_range,
            } => {
                let new_args: Result<Vec<Ast>, CompileError> = args
                    .into_iter()
                    .map(|a| self.apply_renames(a, renames))
                    .collect();
                Ok(Ast::Call {
                    name,
                    args: new_args?,
                    token_range,
                })
            }
            Ast::Add {
                left,
                right,
                token_range,
            } => Ok(Ast::Add {
                left: Box::new(self.apply_renames(*left, renames)?),
                right: Box::new(self.apply_renames(*right, renames)?),
                token_range,
            }),
            Ast::Sub {
                left,
                right,
                token_range,
            } => Ok(Ast::Sub {
                left: Box::new(self.apply_renames(*left, renames)?),
                right: Box::new(self.apply_renames(*right, renames)?),
                token_range,
            }),
            Ast::Mul {
                left,
                right,
                token_range,
            } => Ok(Ast::Mul {
                left: Box::new(self.apply_renames(*left, renames)?),
                right: Box::new(self.apply_renames(*right, renames)?),
                token_range,
            }),
            Ast::Div {
                left,
                right,
                token_range,
            } => Ok(Ast::Div {
                left: Box::new(self.apply_renames(*left, renames)?),
                right: Box::new(self.apply_renames(*right, renames)?),
                token_range,
            }),
            Ast::Modulo {
                left,
                right,
                token_range,
            } => Ok(Ast::Modulo {
                left: Box::new(self.apply_renames(*left, renames)?),
                right: Box::new(self.apply_renames(*right, renames)?),
                token_range,
            }),
            Ast::And {
                left,
                right,
                token_range,
            } => Ok(Ast::And {
                left: Box::new(self.apply_renames(*left, renames)?),
                right: Box::new(self.apply_renames(*right, renames)?),
                token_range,
            }),
            Ast::Or {
                left,
                right,
                token_range,
            } => Ok(Ast::Or {
                left: Box::new(self.apply_renames(*left, renames)?),
                right: Box::new(self.apply_renames(*right, renames)?),
                token_range,
            }),
            Ast::Condition {
                operator,
                left,
                right,
                token_range,
            } => Ok(Ast::Condition {
                operator,
                left: Box::new(self.apply_renames(*left, renames)?),
                right: Box::new(self.apply_renames(*right, renames)?),
                token_range,
            }),
            Ast::Array { array, token_range } => {
                let new_array: Result<Vec<Ast>, CompileError> = array
                    .into_iter()
                    .map(|e| self.apply_renames(e, renames))
                    .collect();
                Ok(Ast::Array {
                    array: new_array?,
                    token_range,
                })
            }
            Ast::Match {
                value,
                arms,
                token_range,
            } => {
                let new_value = self.apply_renames(*value, renames)?;
                let new_arms: Result<Vec<MatchArm>, CompileError> = arms
                    .into_iter()
                    .map(|arm| {
                        let new_pattern = self.apply_renames_to_pattern(arm.pattern, renames);
                        let new_guard = arm
                            .guard
                            .map(|g| self.apply_renames(*g, renames))
                            .transpose()?;
                        let new_body: Result<Vec<Ast>, CompileError> = arm
                            .body
                            .into_iter()
                            .map(|s| self.apply_renames(s, renames))
                            .collect();
                        Ok(MatchArm {
                            pattern: new_pattern,
                            guard: new_guard.map(Box::new),
                            body: new_body?,
                            token_range: arm.token_range,
                        })
                    })
                    .collect();
                Ok(Ast::Match {
                    value: Box::new(new_value),
                    arms: new_arms?,
                    token_range,
                })
            }
            Ast::While {
                condition,
                body,
                token_range,
            } => {
                let new_cond = self.apply_renames(*condition, renames)?;
                let new_body: Result<Vec<Ast>, CompileError> = body
                    .into_iter()
                    .map(|s| self.apply_renames(s, renames))
                    .collect();
                Ok(Ast::While {
                    condition: Box::new(new_cond),
                    body: new_body?,
                    token_range,
                })
            }
            Ast::Loop { body, token_range } => {
                let new_body: Result<Vec<Ast>, CompileError> = body
                    .into_iter()
                    .map(|s| self.apply_renames(s, renames))
                    .collect();
                Ok(Ast::Loop {
                    body: new_body?,
                    token_range,
                })
            }
            Ast::For {
                binding,
                collection,
                body,
                token_range,
            } => {
                let new_binding = renames.get(&binding).cloned().unwrap_or(binding);
                let new_collection = self.apply_renames(*collection, renames)?;
                let new_body: Result<Vec<Ast>, CompileError> = body
                    .into_iter()
                    .map(|s| self.apply_renames(s, renames))
                    .collect();
                Ok(Ast::For {
                    binding: new_binding,
                    collection: Box::new(new_collection),
                    body: new_body?,
                    token_range,
                })
            }
            Ast::PropertyAccess {
                object,
                property,
                token_range,
            } => Ok(Ast::PropertyAccess {
                object: Box::new(self.apply_renames(*object, renames)?),
                property: Box::new(self.apply_renames(*property, renames)?),
                token_range,
            }),
            Ast::IndexOperator {
                array,
                index,
                token_range,
            } => Ok(Ast::IndexOperator {
                array: Box::new(self.apply_renames(*array, renames)?),
                index: Box::new(self.apply_renames(*index, renames)?),
                token_range,
            }),
            Ast::Assignment {
                name,
                value,
                token_range,
            } => Ok(Ast::Assignment {
                name: Box::new(self.apply_renames(*name, renames)?),
                value: Box::new(self.apply_renames(*value, renames)?),
                token_range,
            }),
            Ast::Break { value, token_range } => Ok(Ast::Break {
                value: Box::new(self.apply_renames(*value, renames)?),
                token_range,
            }),
            // Pass through other nodes unchanged
            other => Ok(other),
        }
    }

    /// Apply renames to a pattern
    fn apply_renames_to_pattern(
        &self,
        pattern: Pattern,
        renames: &HashMap<String, String>,
    ) -> Pattern {
        match pattern {
            Pattern::Identifier { name, token_range } => {
                if let Some(new_name) = renames.get(&name) {
                    Pattern::Identifier {
                        name: new_name.clone(),
                        token_range,
                    }
                } else {
                    Pattern::Identifier { name, token_range }
                }
            }
            Pattern::Array {
                elements,
                rest,
                token_range,
            } => {
                let new_elements: Vec<Pattern> = elements
                    .into_iter()
                    .map(|p| self.apply_renames_to_pattern(p, renames))
                    .collect();
                let new_rest = rest.map(|r| Box::new(self.apply_renames_to_pattern(*r, renames)));
                Pattern::Array {
                    elements: new_elements,
                    rest: new_rest,
                    token_range,
                }
            }
            Pattern::EnumVariant {
                enum_name,
                variant_name,
                fields,
                token_range,
            } => {
                let new_fields: Vec<FieldPattern> = fields
                    .into_iter()
                    .map(|f| {
                        let new_binding = f
                            .binding_name
                            .map(|b| renames.get(&b).cloned().unwrap_or(b));
                        FieldPattern {
                            field_name: f.field_name,
                            binding_name: new_binding,
                            token_range: f.token_range,
                        }
                    })
                    .collect();
                Pattern::EnumVariant {
                    enum_name,
                    variant_name,
                    fields: new_fields,
                    token_range,
                }
            }
            Pattern::Struct {
                name,
                fields,
                token_range,
            } => {
                let new_fields: Vec<FieldPattern> = fields
                    .into_iter()
                    .map(|f| {
                        let new_binding = f
                            .binding_name
                            .map(|b| renames.get(&b).cloned().unwrap_or(b));
                        FieldPattern {
                            field_name: f.field_name,
                            binding_name: new_binding,
                            token_range: f.token_range,
                        }
                    })
                    .collect();
                Pattern::Struct {
                    name,
                    fields: new_fields,
                    token_range,
                }
            }
            other => other,
        }
    }

    // ========================================================================
    // AST to Value conversion (for future procedural macro support)
    // ========================================================================

    /// Convert a Rust Ast node to a Beagle runtime value.
    /// This calls the beagle.ast/make-* functions at runtime.
    /// Note: This infrastructure is for future procedural macro support.
    #[allow(dead_code)]
    pub fn ast_to_value(&mut self, ast: &Ast) -> Result<usize, CompileError> {
        let _runtime = get_runtime().get();

        match ast {
            Ast::IntegerLiteral(n, _) => {
                let value = BuiltInTypes::construct_int(*n as isize) as usize;
                self.call_make_fn("beagle.ast/make-integer-literal", &[value])
            }
            Ast::FloatLiteral(s, _) => {
                let str_value = self.allocate_string(s)?;
                self.call_make_fn("beagle.ast/make-float-literal", &[str_value])
            }
            Ast::String(s, _) => {
                let str_value = self.allocate_string(s)?;
                self.call_make_fn("beagle.ast/make-string-literal", &[str_value])
            }
            Ast::Identifier(name, _) => {
                let name_value = self.allocate_string(name)?;
                self.call_make_fn("beagle.ast/make-identifier", &[name_value])
            }
            Ast::Keyword(k, _) => {
                let k_value = self.allocate_string(k)?;
                self.call_make_fn("beagle.ast/make-keyword", &[k_value])
            }
            Ast::True(_) => self.call_make_fn("beagle.ast/make-true", &[]),
            Ast::False(_) => self.call_make_fn("beagle.ast/make-false", &[]),
            Ast::Null(_) => self.call_make_fn("beagle.ast/make-null", &[]),

            // Binary operations
            Ast::Add { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-add", &[left_val, right_val])
            }
            Ast::Sub { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-sub", &[left_val, right_val])
            }
            Ast::Mul { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-mul", &[left_val, right_val])
            }
            Ast::Div { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-div", &[left_val, right_val])
            }
            Ast::Modulo { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-modulo", &[left_val, right_val])
            }

            // Logical operations
            Ast::And { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-and", &[left_val, right_val])
            }
            Ast::Or { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-or", &[left_val, right_val])
            }

            // Comparison
            Ast::Condition {
                operator,
                left,
                right,
                ..
            } => {
                let op_val = self.condition_to_value(operator)?;
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-condition", &[op_val, left_val, right_val])
            }

            // Control flow
            Ast::If {
                condition,
                then,
                else_,
                ..
            } => {
                let cond_val = self.ast_to_value(condition)?;
                let then_val = self.ast_vec_to_value(then)?;
                let else_val = self.ast_vec_to_value(else_)?;
                self.call_make_fn("beagle.ast/make-if", &[cond_val, then_val, else_val])
            }

            // Block expression
            Ast::Block { statements, .. } => {
                let stmts_val = self.ast_vec_to_value(statements)?;
                self.call_make_fn("beagle.ast/make-block", &[stmts_val])
            }

            // Function calls
            Ast::Call { name, args, .. } => {
                let name_val = self.allocate_string(name)?;
                let args_val = self.ast_vec_to_value(args)?;
                self.call_make_fn("beagle.ast/make-call", &[name_val, args_val])
            }

            // Let bindings
            Ast::Let { pattern, value, .. } => {
                let pattern_val = self.pattern_to_value(pattern)?;
                let value_val = self.ast_to_value(value)?;
                self.call_make_fn("beagle.ast/make-let", &[pattern_val, value_val])
            }
            Ast::LetMut { pattern, value, .. } => {
                let pattern_val = self.pattern_to_value(pattern)?;
                let value_val = self.ast_to_value(value)?;
                self.call_make_fn("beagle.ast/make-let-mut", &[pattern_val, value_val])
            }

            // Array
            Ast::Array { array, .. } => {
                let elements_val = self.ast_vec_to_value(array)?;
                self.call_make_fn("beagle.ast/make-array", &[elements_val])
            }

            // Arrow pair (for macro-friendly syntax)
            Ast::ArrowPair { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-arrow-pair", &[left_val, right_val])
            }

            // Quote/Unquote
            Ast::Quote { body, .. } => {
                let body_val = self.ast_to_value(body)?;
                self.call_make_fn("beagle.ast/make-quote", &[body_val])
            }
            Ast::Unquote { body, .. } => {
                let body_val = self.ast_to_value(body)?;
                self.call_make_fn("beagle.ast/make-unquote", &[body_val])
            }
            Ast::UnquoteSplice { body, .. } => {
                let body_val = self.ast_to_value(body)?;
                self.call_make_fn("beagle.ast/make-unquote-splice", &[body_val])
            }

            // Loop constructs
            Ast::Loop { body, .. } => {
                let body_val = self.ast_vec_to_value(body)?;
                self.call_make_fn("beagle.ast/make-loop", &[body_val])
            }
            Ast::While {
                condition, body, ..
            } => {
                let cond_val = self.ast_to_value(condition)?;
                let body_val = self.ast_vec_to_value(body)?;
                self.call_make_fn("beagle.ast/make-while", &[cond_val, body_val])
            }
            Ast::For {
                binding,
                collection,
                body,
                ..
            } => {
                let binding_val = self.allocate_string(binding)?;
                let collection_val = self.ast_to_value(collection)?;
                let body_val = self.ast_vec_to_value(body)?;
                self.call_make_fn(
                    "beagle.ast/make-for",
                    &[binding_val, collection_val, body_val],
                )
            }
            Ast::Break { value, .. } => {
                let value_val = self.ast_to_value(value)?;
                self.call_make_fn("beagle.ast/make-break", &[value_val])
            }
            Ast::Continue { .. } => self.call_make_fn("beagle.ast/make-continue", &[]),

            // Assignment
            Ast::Assignment { name, value, .. } => {
                let name_val = self.ast_to_value(name)?;
                let value_val = self.ast_to_value(value)?;
                self.call_make_fn("beagle.ast/make-assignment", &[name_val, value_val])
            }

            // Property access
            Ast::PropertyAccess {
                object, property, ..
            } => {
                let obj_val = self.ast_to_value(object)?;
                let prop_val = self.ast_to_value(property)?;
                self.call_make_fn("beagle.ast/make-property-access", &[obj_val, prop_val])
            }

            // Index operator
            Ast::IndexOperator { array, index, .. } => {
                let array_val = self.ast_to_value(array)?;
                let index_val = self.ast_to_value(index)?;
                self.call_make_fn("beagle.ast/make-index-operator", &[array_val, index_val])
            }

            // Match
            Ast::Match { value, arms, .. } => {
                let value_val = self.ast_to_value(value)?;
                let arms_val = self.match_arms_to_value(arms)?;
                self.call_make_fn("beagle.ast/make-match", &[value_val, arms_val])
            }

            // Error handling
            Ast::Throw { value, .. } => {
                let value_val = self.ast_to_value(value)?;
                self.call_make_fn("beagle.ast/make-throw", &[value_val])
            }
            Ast::Try {
                body,
                exception_binding,
                catch_body,
                ..
            } => {
                let body_val = self.ast_vec_to_value(body)?;
                let binding_val = self.allocate_string(exception_binding)?;
                let catch_val = self.ast_vec_to_value(catch_body)?;
                self.call_make_fn("beagle.ast/make-try", &[body_val, binding_val, catch_val])
            }

            // Bitwise operations
            Ast::ShiftLeft { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-shift-left", &[left_val, right_val])
            }
            Ast::ShiftRight { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-shift-right", &[left_val, right_val])
            }
            Ast::ShiftRightZero { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-shift-right-zero", &[left_val, right_val])
            }
            Ast::BitWiseAnd { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-bitwise-and", &[left_val, right_val])
            }
            Ast::BitWiseOr { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-bitwise-or", &[left_val, right_val])
            }
            Ast::BitWiseXor { left, right, .. } => {
                let left_val = self.ast_to_value(left)?;
                let right_val = self.ast_to_value(right)?;
                self.call_make_fn("beagle.ast/make-bitwise-xor", &[left_val, right_val])
            }

            // Namespace/Import
            Ast::Namespace { name, .. } => {
                let name_val = self.allocate_string(name)?;
                self.call_make_fn("beagle.ast/make-namespace", &[name_val])
            }
            Ast::Import {
                library_name,
                alias,
                ..
            } => {
                let lib_val = self.allocate_string(library_name)?;
                let alias_val = self.ast_to_value(alias)?;
                self.call_make_fn("beagle.ast/make-import", &[lib_val, alias_val])
            }

            // Recursion
            Ast::Recurse { args, .. } => {
                let args_val = self.ast_vec_to_value(args)?;
                self.call_make_fn("beagle.ast/make-recurse", &[args_val])
            }
            Ast::TailRecurse { args, .. } => {
                let args_val = self.ast_vec_to_value(args)?;
                self.call_make_fn("beagle.ast/make-tail-recurse", &[args_val])
            }

            // Continuations
            Ast::Reset { body, .. } => {
                let body_val = self.ast_vec_to_value(body)?;
                self.call_make_fn("beagle.ast/make-reset", &[body_val])
            }
            Ast::Shift {
                continuation_param,
                body,
                ..
            } => {
                let param_val = self.allocate_string(continuation_param)?;
                let body_val = self.ast_vec_to_value(body)?;
                self.call_make_fn("beagle.ast/make-shift", &[param_val, body_val])
            }

            // Effects
            Ast::Perform { value, .. } => {
                let value_val = self.ast_to_value(value)?;
                self.call_make_fn("beagle.ast/make-perform", &[value_val])
            }
            Ast::Handle {
                protocol,
                protocol_type_args,
                handler_instance,
                body,
                ..
            } => {
                let protocol_val = self.allocate_string(protocol)?;
                let type_args_val = self.string_vec_to_value(protocol_type_args)?;
                let handler_val = self.ast_to_value(handler_instance)?;
                let body_val = self.ast_vec_to_value(body)?;
                self.call_make_fn(
                    "beagle.ast/make-handle",
                    &[protocol_val, type_args_val, handler_val, body_val],
                )
            }

            // Fallback for unhandled nodes
            _ => {
                let kind = format!("{:?}", std::mem::discriminant(ast));
                let kind_val = self.allocate_string(&kind)?;
                self.call_make_fn("beagle.ast/make-unknown", &[kind_val])
            }
        }
    }

    /// Convert a condition operator to a Beagle enum value
    #[allow(dead_code)]
    fn condition_to_value(&mut self, op: &Condition) -> Result<usize, CompileError> {
        // For now, represent the operator as a string.
        // When beagle.ast is loaded, we can call the actual constructor.
        let op_str = match op {
            Condition::LessThan => "LessThan",
            Condition::LessThanOrEqual => "LessThanOrEqual",
            Condition::GreaterThan => "GreaterThan",
            Condition::GreaterThanOrEqual => "GreaterThanOrEqual",
            Condition::Equal => "Equal",
            Condition::NotEqual => "NotEqual",
        };
        self.allocate_string(op_str)
    }

    /// Convert a vector of AST nodes to a Beagle array
    #[allow(dead_code)]
    fn ast_vec_to_value(&mut self, asts: &[Ast]) -> Result<usize, CompileError> {
        let mut vec = self.call_make_fn("beagle.collections/vec", &[])?;
        for ast in asts {
            let val = self.ast_to_value(ast)?;
            vec = self.call_make_fn("beagle.collections/vec-push", &[vec, val])?;
        }
        Ok(vec)
    }

    /// Convert a vector of strings to a Beagle array
    #[allow(dead_code)]
    fn string_vec_to_value(&mut self, strings: &[String]) -> Result<usize, CompileError> {
        let mut vec = self.call_make_fn("beagle.collections/vec", &[])?;
        for s in strings {
            let val = self.allocate_string(s)?;
            vec = self.call_make_fn("beagle.collections/vec-push", &[vec, val])?;
        }
        Ok(vec)
    }

    /// Convert match arms to a Beagle array
    #[allow(dead_code)]
    fn match_arms_to_value(&mut self, arms: &[MatchArm]) -> Result<usize, CompileError> {
        let mut vec = self.call_make_fn("beagle.collections/vec", &[])?;
        for arm in arms {
            let pattern_val = self.pattern_to_value(&arm.pattern)?;
            let guard_val = if let Some(guard) = &arm.guard {
                self.ast_to_value(guard)?
            } else {
                BuiltInTypes::null_value() as usize
            };
            let body_val = self.ast_vec_to_value(&arm.body)?;
            let arm_val = self.call_make_fn(
                "beagle.ast/make-match-arm",
                &[pattern_val, guard_val, body_val],
            )?;
            vec = self.call_make_fn("beagle.collections/vec-push", &[vec, arm_val])?;
        }
        Ok(vec)
    }

    /// Convert a Pattern to a Beagle value
    #[allow(dead_code)]
    fn pattern_to_value(&mut self, pattern: &Pattern) -> Result<usize, CompileError> {
        match pattern {
            Pattern::Identifier { name, .. } => {
                let name_val = self.allocate_string(name)?;
                self.call_make_fn("beagle.ast/make-pattern-identifier", &[name_val])
            }
            Pattern::Wildcard { .. } => self.call_make_fn("beagle.ast/make-pattern-wildcard", &[]),
            Pattern::Literal { value, .. } => {
                let value_val = self.ast_to_value(value)?;
                self.call_make_fn("beagle.ast/make-pattern-literal", &[value_val])
            }
            Pattern::EnumVariant {
                enum_name,
                variant_name,
                fields,
                ..
            } => {
                let enum_val = self.allocate_string(enum_name)?;
                let variant_val = self.allocate_string(variant_name)?;
                let fields_val = self.field_patterns_to_value(fields)?;
                self.call_make_fn(
                    "beagle.ast/make-pattern-enum-variant",
                    &[enum_val, variant_val, fields_val],
                )
            }
            Pattern::Struct { name, fields, .. } => {
                let name_val = self.allocate_string(name)?;
                let fields_val = self.field_patterns_to_value(fields)?;
                self.call_make_fn("beagle.ast/make-pattern-struct", &[name_val, fields_val])
            }
            Pattern::Array { elements, rest, .. } => {
                let elements_val = self.patterns_to_value(elements)?;
                let rest_val = if let Some(r) = rest {
                    self.pattern_to_value(r)?
                } else {
                    BuiltInTypes::null_value() as usize
                };
                self.call_make_fn("beagle.ast/make-pattern-array", &[elements_val, rest_val])
            }
            Pattern::Map { fields: _, .. } => {
                // Simplified: just convert to unknown for now
                let kind = "Map".to_string();
                let kind_val = self.allocate_string(&kind)?;
                self.call_make_fn("beagle.ast/make-pattern-unknown", &[kind_val])
            }
        }
    }

    /// Convert a vector of Patterns to a Beagle array
    #[allow(dead_code)]
    fn patterns_to_value(&mut self, patterns: &[Pattern]) -> Result<usize, CompileError> {
        let mut vec = self.call_make_fn("beagle.collections/vec", &[])?;
        for pattern in patterns {
            let val = self.pattern_to_value(pattern)?;
            vec = self.call_make_fn("beagle.collections/vec-push", &[vec, val])?;
        }
        Ok(vec)
    }

    /// Convert FieldPatterns to a Beagle array
    #[allow(dead_code)]
    fn field_patterns_to_value(&mut self, fields: &[FieldPattern]) -> Result<usize, CompileError> {
        let mut vec = self.call_make_fn("beagle.collections/vec", &[])?;
        for field in fields {
            let field_name_val = self.allocate_string(&field.field_name)?;
            let binding_val = if let Some(b) = &field.binding_name {
                self.allocate_string(b)?
            } else {
                BuiltInTypes::null_value() as usize
            };
            let field_val = self.call_make_fn(
                "beagle.ast/make-field-pattern",
                &[field_name_val, binding_val],
            )?;
            vec = self.call_make_fn("beagle.collections/vec-push", &[vec, field_val])?;
        }
        Ok(vec)
    }

    /// Allocate a string in the runtime and return its tagged pointer
    #[allow(dead_code)]
    fn allocate_string(&self, s: &str) -> Result<usize, CompileError> {
        let runtime = get_runtime().get_mut();
        let str_ptr = runtime.add_string_constant(s);
        Ok(str_ptr)
    }

    /// Call a Beagle function at compile time
    ///
    /// This properly handles:
    /// - Regular Beagle functions (via trampoline)
    /// - Builtins that need sp/fp (called directly with sp/fp prepended)
    #[allow(dead_code)]
    fn call_make_fn(&self, fn_name: &str, args: &[usize]) -> Result<usize, CompileError> {
        let runtime = get_runtime().get();

        // Check if this is a builtin that needs sp/fp
        if let Some((needs_sp, _needs_fp)) = runtime.function_needs_sp_fp(fn_name) {
            if needs_sp {
                // Call builtin directly with sp/fp
                return match args.len() {
                    0 => runtime.call_builtin_sp_fp_0(fn_name).map(|r| r as usize),
                    1 => runtime
                        .call_builtin_sp_fp_1(fn_name, args[0] as u64)
                        .map(|r| r as usize),
                    2 => runtime
                        .call_builtin_sp_fp_2(fn_name, args[0] as u64, args[1] as u64)
                        .map(|r| r as usize),
                    3 => runtime
                        .call_builtin_sp_fp_3(
                            fn_name,
                            args[0] as u64,
                            args[1] as u64,
                            args[2] as u64,
                        )
                        .map(|r| r as usize),
                    _ => {
                        return Err(CompileError::Internal(format!(
                            "call_make_fn: builtin {} with sp/fp needs {} args, max supported is 3",
                            fn_name,
                            args.len()
                        )));
                    }
                }
                .ok_or_else(|| CompileError::FunctionNotFound {
                    function_name: fn_name.to_string(),
                });
            }
        }

        // Regular function call via trampoline
        match args.len() {
            0 => {
                if let Some(f) = runtime.get_function0(fn_name) {
                    Ok(f() as usize)
                } else {
                    Err(CompileError::FunctionNotFound {
                        function_name: fn_name.to_string(),
                    })
                }
            }
            1 => {
                if let Some(f) = runtime.get_function1(fn_name) {
                    Ok(f(args[0] as u64) as usize)
                } else {
                    Err(CompileError::FunctionNotFound {
                        function_name: fn_name.to_string(),
                    })
                }
            }
            2 => {
                if let Some(f) = runtime.get_function2(fn_name) {
                    Ok(f(args[0] as u64, args[1] as u64) as usize)
                } else {
                    Err(CompileError::FunctionNotFound {
                        function_name: fn_name.to_string(),
                    })
                }
            }
            3 => {
                if let Some(f) = runtime.get_function3(fn_name) {
                    Ok(f(args[0] as u64, args[1] as u64, args[2] as u64) as usize)
                } else {
                    Err(CompileError::FunctionNotFound {
                        function_name: fn_name.to_string(),
                    })
                }
            }
            4 => {
                if let Some(f) = runtime.get_function4(fn_name) {
                    Ok(f(
                        args[0] as u64,
                        args[1] as u64,
                        args[2] as u64,
                        args[3] as u64,
                    ) as usize)
                } else {
                    Err(CompileError::FunctionNotFound {
                        function_name: fn_name.to_string(),
                    })
                }
            }
            5 => {
                if let Some(f) = runtime.get_function5(fn_name) {
                    Ok(f(
                        args[0] as u64,
                        args[1] as u64,
                        args[2] as u64,
                        args[3] as u64,
                        args[4] as u64,
                    ) as usize)
                } else {
                    Err(CompileError::FunctionNotFound {
                        function_name: fn_name.to_string(),
                    })
                }
            }
            6 => {
                if let Some(f) = runtime.get_function6(fn_name) {
                    Ok(f(
                        args[0] as u64,
                        args[1] as u64,
                        args[2] as u64,
                        args[3] as u64,
                        args[4] as u64,
                        args[5] as u64,
                    ) as usize)
                } else {
                    Err(CompileError::FunctionNotFound {
                        function_name: fn_name.to_string(),
                    })
                }
            }
            7 => {
                if let Some(f) = runtime.get_function7(fn_name) {
                    Ok(f(
                        args[0] as u64,
                        args[1] as u64,
                        args[2] as u64,
                        args[3] as u64,
                        args[4] as u64,
                        args[5] as u64,
                        args[6] as u64,
                    ) as usize)
                } else {
                    Err(CompileError::FunctionNotFound {
                        function_name: fn_name.to_string(),
                    })
                }
            }
            8 => {
                if let Some(f) = runtime.get_function8(fn_name) {
                    Ok(f(
                        args[0] as u64,
                        args[1] as u64,
                        args[2] as u64,
                        args[3] as u64,
                        args[4] as u64,
                        args[5] as u64,
                        args[6] as u64,
                        args[7] as u64,
                    ) as usize)
                } else {
                    Err(CompileError::FunctionNotFound {
                        function_name: fn_name.to_string(),
                    })
                }
            }
            _ => Err(CompileError::Internal(format!(
                "call_make_fn: unsupported arity {}",
                args.len()
            ))),
        }
    }

    // ========================================================================
    // Value to AST conversion (for future procedural macro support)
    // ========================================================================

    /// Convert a Beagle runtime value back to a Rust Ast.
    /// The value should be a beagle.ast/Ast enum instance.
    /// Note: This infrastructure is for future procedural macro support.
    #[allow(dead_code)]
    pub fn value_to_ast(&self, value: usize) -> Result<Ast, CompileError> {
        // Check if it's a heap pointer
        if !BuiltInTypes::is_heap_pointer(value) {
            return Err(CompileError::MacroInvalidReturn {
                name: "macro".to_string(),
                expected: "AST node".to_string(),
                got: format!("{:?}", BuiltInTypes::get_kind(value)),
            });
        }

        let heap_obj = HeapObject::from_tagged(value);
        let header = heap_obj.get_header();
        let struct_id = BuiltInTypes::untag(header.type_data as usize);

        // Get the struct name from the runtime
        let runtime = get_runtime().get();
        let struct_info = runtime
            .get_struct_by_id(struct_id)
            .ok_or_else(|| CompileError::Internal(format!("Unknown struct id: {}", struct_id)))?;

        let variant_name = struct_info
            .name
            .split('/')
            .next_back()
            .unwrap_or(&struct_info.name);

        // Dispatch based on variant name
        self.value_to_ast_by_variant(variant_name, &heap_obj)
    }

    /// Convert a heap object to an AST based on its variant name
    #[allow(dead_code)]
    fn value_to_ast_by_variant(
        &self,
        variant_name: &str,
        obj: &HeapObject,
    ) -> Result<Ast, CompileError> {
        let token_range = TokenRange::new(0, 0);

        match variant_name {
            // Literals
            "Ast.IntegerLiteral" => {
                let value = obj.get_field(0);
                let n = BuiltInTypes::untag(value) as i64;
                Ok(Ast::IntegerLiteral(n, 0))
            }
            "Ast.FloatLiteral" => {
                let value = obj.get_field(0);
                let s = self.value_to_string(value)?;
                Ok(Ast::FloatLiteral(s, 0))
            }
            "Ast.StringLiteral" => {
                let value = obj.get_field(0);
                let s = self.value_to_string(value)?;
                Ok(Ast::String(s, 0))
            }
            "Ast.Identifier" => {
                let name = obj.get_field(0);
                let s = self.value_to_string(name)?;
                Ok(Ast::Identifier(s, 0))
            }
            "Ast.Keyword" => {
                let name = obj.get_field(0);
                let s = self.value_to_string(name)?;
                Ok(Ast::Keyword(s, 0))
            }
            "Ast.True" => Ok(Ast::True(0)),
            "Ast.False" => Ok(Ast::False(0)),
            "Ast.Null" => Ok(Ast::Null(0)),

            // Arithmetic operations
            "Ast.Add" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::Add {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }
            "Ast.Sub" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::Sub {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }
            "Ast.Mul" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::Mul {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }
            "Ast.Div" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::Div {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }
            "Ast.Modulo" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::Modulo {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }

            // Comparison
            "Ast.Condition" => {
                let op_value = obj.get_field(0);
                let operator = self.value_to_condition_operator(op_value)?;
                let left = self.value_to_ast(obj.get_field(1))?;
                let right = self.value_to_ast(obj.get_field(2))?;
                Ok(Ast::Condition {
                    operator,
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }

            // Logical
            "Ast.AndExpr" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::And {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }
            "Ast.OrExpr" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::Or {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }

            // Bitwise operations
            "Ast.ShiftLeftExpr" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::ShiftLeft {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }
            "Ast.ShiftRightExpr" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::ShiftRight {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }
            "Ast.ShiftRightZeroExpr" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::ShiftRightZero {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }
            "Ast.BitWiseAndExpr" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::BitWiseAnd {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }
            "Ast.BitWiseOrExpr" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::BitWiseOr {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }
            "Ast.BitWiseXorExpr" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::BitWiseXor {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }

            // Control flow
            "Ast.IfExpr" => {
                let cond = self.value_to_ast(obj.get_field(0))?;
                let then = self.value_to_ast_vec(obj.get_field(1))?;
                let else_ = self.value_to_ast_vec(obj.get_field(2))?;
                Ok(Ast::If {
                    condition: Box::new(cond),
                    then,
                    else_,
                    token_range,
                })
            }
            "Ast.MatchExpr" => {
                let value = self.value_to_ast(obj.get_field(0))?;
                let arms = self.value_to_match_arms(obj.get_field(1))?;
                Ok(Ast::Match {
                    value: Box::new(value),
                    arms,
                    token_range,
                })
            }

            // Block expression
            "Ast.BlockExpr" => {
                let statements = self.value_to_ast_vec(obj.get_field(0))?;
                Ok(Ast::Block {
                    statements,
                    token_range,
                })
            }

            // Call
            "Ast.Call" => {
                let name = self.value_to_string(obj.get_field(0))?;
                let args = self.value_to_ast_vec(obj.get_field(1))?;
                Ok(Ast::Call {
                    name,
                    args,
                    token_range,
                })
            }

            // Recursion
            "Ast.RecurseExpr" => {
                let args = self.value_to_ast_vec(obj.get_field(0))?;
                Ok(Ast::Recurse { args, token_range })
            }
            "Ast.TailRecurseExpr" => {
                let args = self.value_to_ast_vec(obj.get_field(0))?;
                Ok(Ast::TailRecurse { args, token_range })
            }

            // Let bindings
            "Ast.LetExpr" => {
                let pattern = self.value_to_pattern(obj.get_field(0))?;
                let value = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::Let {
                    pattern,
                    value: Box::new(value),
                    token_range,
                })
            }
            "Ast.LetMutExpr" => {
                let pattern = self.value_to_pattern(obj.get_field(0))?;
                let value = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::LetMut {
                    pattern,
                    value: Box::new(value),
                    token_range,
                })
            }
            "Ast.Assignment" => {
                let name = self.value_to_ast(obj.get_field(0))?;
                let value = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::Assignment {
                    name: Box::new(name),
                    value: Box::new(value),
                    token_range,
                })
            }

            // Collections
            "Ast.ArrayExpr" => {
                let elements = self.value_to_ast_vec(obj.get_field(0))?;
                Ok(Ast::Array {
                    array: elements,
                    token_range,
                })
            }
            "Ast.MapLiteral" => {
                let pairs = self.value_to_ast_pairs(obj.get_field(0))?;
                Ok(Ast::MapLiteral { pairs, token_range })
            }
            "Ast.SetLiteral" => {
                let elements = self.value_to_ast_vec(obj.get_field(0))?;
                Ok(Ast::SetLiteral {
                    elements,
                    token_range,
                })
            }
            "Ast.IndexOperator" => {
                let array = self.value_to_ast(obj.get_field(0))?;
                let index = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::IndexOperator {
                    array: Box::new(array),
                    index: Box::new(index),
                    token_range,
                })
            }
            "Ast.PropertyAccess" => {
                let object = self.value_to_ast(obj.get_field(0))?;
                let property = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::PropertyAccess {
                    object: Box::new(object),
                    property: Box::new(property),
                    token_range,
                })
            }

            // Arrow pair
            "Ast.ArrowPair" => {
                let left = self.value_to_ast(obj.get_field(0))?;
                let right = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::ArrowPair {
                    left: Box::new(left),
                    right: Box::new(right),
                    token_range,
                })
            }

            // Quote/Unquote
            "Ast.QuoteExpr" => {
                let body = self.value_to_ast(obj.get_field(0))?;
                Ok(Ast::Quote {
                    body: Box::new(body),
                    token_range,
                })
            }
            "Ast.UnquoteExpr" => {
                let body = self.value_to_ast(obj.get_field(0))?;
                Ok(Ast::Unquote {
                    body: Box::new(body),
                    token_range,
                })
            }
            "Ast.UnquoteSplice" => {
                let body = self.value_to_ast(obj.get_field(0))?;
                Ok(Ast::UnquoteSplice {
                    body: Box::new(body),
                    token_range,
                })
            }

            // Loop constructs
            "Ast.LoopExpr" => {
                let body = self.value_to_ast_vec(obj.get_field(0))?;
                Ok(Ast::Loop { body, token_range })
            }
            "Ast.WhileExpr" => {
                let cond = self.value_to_ast(obj.get_field(0))?;
                let body = self.value_to_ast_vec(obj.get_field(1))?;
                Ok(Ast::While {
                    condition: Box::new(cond),
                    body,
                    token_range,
                })
            }
            "Ast.ForExpr" => {
                let binding = self.value_to_string(obj.get_field(0))?;
                let collection = self.value_to_ast(obj.get_field(1))?;
                let body = self.value_to_ast_vec(obj.get_field(2))?;
                Ok(Ast::For {
                    binding,
                    collection: Box::new(collection),
                    body,
                    token_range,
                })
            }
            "Ast.BreakExpr" => {
                let value = self.value_to_ast(obj.get_field(0))?;
                Ok(Ast::Break {
                    value: Box::new(value),
                    token_range,
                })
            }
            "Ast.ContinueExpr" => Ok(Ast::Continue { token_range }),

            // Error handling
            "Ast.TryExpr" => {
                let body = self.value_to_ast_vec(obj.get_field(0))?;
                let exception_binding = self.value_to_string(obj.get_field(1))?;
                let catch_body = self.value_to_ast_vec(obj.get_field(2))?;
                Ok(Ast::Try {
                    body,
                    exception_binding,
                    catch_body,
                    token_range,
                })
            }
            "Ast.ThrowExpr" => {
                let value = self.value_to_ast(obj.get_field(0))?;
                Ok(Ast::Throw {
                    value: Box::new(value),
                    token_range,
                })
            }

            // Definitions
            "Ast.FunctionDef" => {
                let name = self.value_to_optional_string(obj.get_field(0))?;
                let args = self.value_to_patterns(obj.get_field(1))?;
                let rest_param = self.value_to_optional_string(obj.get_field(2))?;
                let body = self.value_to_ast_vec(obj.get_field(3))?;
                Ok(Ast::Function {
                    name,
                    args,
                    rest_param,
                    body,
                    token_range,
                })
            }
            "Ast.FunctionStub" => {
                let name = self.value_to_string(obj.get_field(0))?;
                let args = self.value_to_patterns(obj.get_field(1))?;
                let rest_param = self.value_to_optional_string(obj.get_field(2))?;
                Ok(Ast::FunctionStub {
                    name,
                    args,
                    rest_param,
                    token_range,
                })
            }
            "Ast.StructDef" => {
                let name = self.value_to_string(obj.get_field(0))?;
                let fields = self.value_to_ast_vec(obj.get_field(1))?;
                Ok(Ast::Struct {
                    name,
                    fields,
                    token_range,
                })
            }
            "Ast.StructField" => {
                let name = self.value_to_string(obj.get_field(0))?;
                let mutable = self.value_to_bool(obj.get_field(1));
                Ok(Ast::StructField {
                    name,
                    mutable,
                    token_range,
                })
            }
            "Ast.StructCreation" => {
                let name = self.value_to_string(obj.get_field(0))?;
                let fields = self.value_to_named_fields(obj.get_field(1))?;
                Ok(Ast::StructCreation {
                    name,
                    fields,
                    token_range,
                })
            }
            "Ast.EnumDef" => {
                let name = self.value_to_string(obj.get_field(0))?;
                let variants = self.value_to_ast_vec(obj.get_field(1))?;
                Ok(Ast::Enum {
                    name,
                    variants,
                    token_range,
                })
            }
            "Ast.EnumVariantDef" => {
                let name = self.value_to_string(obj.get_field(0))?;
                let fields = self.value_to_ast_vec(obj.get_field(1))?;
                Ok(Ast::EnumVariant {
                    name,
                    fields,
                    token_range,
                })
            }
            "Ast.EnumStaticVariantDef" => {
                let name = self.value_to_string(obj.get_field(0))?;
                Ok(Ast::EnumStaticVariant { name, token_range })
            }
            "Ast.EnumCreation" => {
                let name = self.value_to_string(obj.get_field(0))?;
                let variant = self.value_to_string(obj.get_field(1))?;
                let fields = self.value_to_named_fields(obj.get_field(2))?;
                Ok(Ast::EnumCreation {
                    name,
                    variant,
                    fields,
                    token_range,
                })
            }
            "Ast.ProtocolDef" => {
                let name = self.value_to_string(obj.get_field(0))?;
                let type_params = self.value_to_string_vec(obj.get_field(1))?;
                let body = self.value_to_ast_vec(obj.get_field(2))?;
                Ok(Ast::Protocol {
                    name,
                    type_params,
                    body,
                    token_range,
                })
            }
            "Ast.ExtendDef" => {
                let target_type = self.value_to_string(obj.get_field(0))?;
                let protocol = self.value_to_string(obj.get_field(1))?;
                let protocol_type_args = self.value_to_string_vec(obj.get_field(2))?;
                let body = self.value_to_ast_vec(obj.get_field(3))?;
                Ok(Ast::Extend {
                    target_type,
                    protocol,
                    protocol_type_args,
                    body,
                    token_range,
                })
            }
            "Ast.MacroDef" => {
                let name = self.value_to_string(obj.get_field(0))?;
                let args = self.value_to_patterns(obj.get_field(1))?;
                let rest_param = self.value_to_optional_string(obj.get_field(2))?;
                let body = self.value_to_ast_vec(obj.get_field(3))?;
                Ok(Ast::Macro {
                    name,
                    args,
                    rest_param,
                    body,
                    token_range,
                })
            }

            // Namespace
            "Ast.NamespaceDecl" => {
                let name = self.value_to_string(obj.get_field(0))?;
                Ok(Ast::Namespace { name, token_range })
            }
            "Ast.ImportDecl" => {
                let library_name = self.value_to_string(obj.get_field(0))?;
                let alias = self.value_to_ast(obj.get_field(1))?;
                Ok(Ast::Import {
                    library_name,
                    alias: Box::new(alias),
                    token_range,
                })
            }

            // Continuations / Effects
            "Ast.ResetExpr" => {
                let body = self.value_to_ast_vec(obj.get_field(0))?;
                Ok(Ast::Reset { body, token_range })
            }
            "Ast.ShiftExpr" => {
                let continuation_param = self.value_to_string(obj.get_field(0))?;
                let body = self.value_to_ast_vec(obj.get_field(1))?;
                Ok(Ast::Shift {
                    continuation_param,
                    body,
                    token_range,
                })
            }
            "Ast.PerformExpr" => {
                let value = self.value_to_ast(obj.get_field(0))?;
                Ok(Ast::Perform {
                    value: Box::new(value),
                    token_range,
                })
            }
            "Ast.HandleExpr" => {
                let protocol = self.value_to_string(obj.get_field(0))?;
                let protocol_type_args = self.value_to_string_vec(obj.get_field(1))?;
                let handler_instance = self.value_to_ast(obj.get_field(2))?;
                let body = self.value_to_ast_vec(obj.get_field(3))?;
                Ok(Ast::Handle {
                    protocol,
                    protocol_type_args,
                    handler_instance: Box::new(handler_instance),
                    body,
                    token_range,
                })
            }

            // Program
            "Ast.Program" => {
                let elements = self.value_to_ast_vec(obj.get_field(0))?;
                Ok(Ast::Program {
                    elements,
                    token_range,
                })
            }

            // Explicit unknown - allowed to return Null
            "Ast.Unknown" => Ok(Ast::Null(0)),

            // Unrecognized variant - ERROR
            other => Err(CompileError::Internal(format!(
                "value_to_ast: unrecognized AST variant '{}'",
                other
            ))),
        }
    }

    /// Convert a Beagle vec to a Vec<Ast>
    #[allow(dead_code)]
    fn value_to_ast_vec(&self, value: usize) -> Result<Vec<Ast>, CompileError> {
        let runtime = get_runtime().get();

        // Get vec count
        let count_fn = runtime
            .get_function1("beagle.collections/vec-count")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.collections/vec-count".to_string(),
            })?;
        let count = BuiltInTypes::untag(count_fn(value as u64) as usize);

        // Get each element
        let nth_fn = runtime
            .get_function2("beagle.collections/vec-get")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.collections/vec-get".to_string(),
            })?;

        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let idx = BuiltInTypes::construct_int(i as isize) as usize;
            let elem = nth_fn(value as u64, idx as u64) as usize;
            result.push(self.value_to_ast(elem)?);
        }
        Ok(result)
    }

    /// Convert a Beagle string value to a Rust String
    #[allow(dead_code)]
    fn value_to_string(&self, value: usize) -> Result<String, CompileError> {
        if BuiltInTypes::get_kind(value) == BuiltInTypes::String {
            // It's a tagged string pointer (index into string_constants)
            let runtime = get_runtime().get();
            let index = BuiltInTypes::untag(value);
            if let Some(s) = runtime.get_string_constant(index) {
                return Ok(s.to_string());
            }
        } else if BuiltInTypes::is_heap_pointer(value) {
            // It might be a heap-allocated string
            let heap_obj = HeapObject::from_tagged(value);
            // Try to read it as a string using get_str_unchecked
            let s = heap_obj.get_str_unchecked();
            return Ok(s.to_string());
        }
        Err(CompileError::Internal(format!(
            "Expected string, got {:?}",
            BuiltInTypes::get_kind(value)
        )))
    }

    /// Convert a Beagle Pattern value to a Rust Pattern
    #[allow(dead_code)]
    fn value_to_pattern(&self, value: usize) -> Result<Pattern, CompileError> {
        if !BuiltInTypes::is_heap_pointer(value) {
            return Err(CompileError::Internal("Expected pattern value".to_string()));
        }

        let heap_obj = HeapObject::from_tagged(value);
        let header = heap_obj.get_header();
        let struct_id = BuiltInTypes::untag(header.type_data as usize);

        let runtime = get_runtime().get();
        let struct_info = runtime
            .get_struct_by_id(struct_id)
            .ok_or_else(|| CompileError::Internal(format!("Unknown struct id: {}", struct_id)))?;

        let variant_name = struct_info
            .name
            .split('/')
            .next_back()
            .unwrap_or(&struct_info.name);
        let token_range = TokenRange::new(0, 0);

        match variant_name {
            "Pattern.Identifier" => {
                let name = self.value_to_string(heap_obj.get_field(0))?;
                Ok(Pattern::Identifier { name, token_range })
            }
            "Pattern.Wildcard" => Ok(Pattern::Wildcard { token_range }),
            "Pattern.Literal" => {
                let value_ast = self.value_to_ast(heap_obj.get_field(0))?;
                Ok(Pattern::Literal {
                    value: Box::new(value_ast),
                    token_range,
                })
            }
            _ => Ok(Pattern::Wildcard { token_range }),
        }
    }

    /// Convert a Beagle value to a Vec<Pattern>
    fn value_to_patterns(&self, value: usize) -> Result<Vec<Pattern>, CompileError> {
        let runtime = get_runtime().get();

        let count_fn = runtime
            .get_function1("beagle.collections/vec-count")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.collections/vec-count".to_string(),
            })?;
        let count = BuiltInTypes::untag(count_fn(value as u64) as usize);

        let nth_fn = runtime
            .get_function2("beagle.collections/vec-get")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.collections/vec-get".to_string(),
            })?;

        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let idx = BuiltInTypes::construct_int(i as isize) as usize;
            let elem = nth_fn(value as u64, idx as u64) as usize;
            result.push(self.value_to_pattern(elem)?);
        }
        Ok(result)
    }

    /// Convert a Beagle value to a Vec<String>
    fn value_to_string_vec(&self, value: usize) -> Result<Vec<String>, CompileError> {
        let runtime = get_runtime().get();

        let count_fn = runtime
            .get_function1("beagle.collections/vec-count")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.collections/vec-count".to_string(),
            })?;
        let count = BuiltInTypes::untag(count_fn(value as u64) as usize);

        let nth_fn = runtime
            .get_function2("beagle.collections/vec-get")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.collections/vec-get".to_string(),
            })?;

        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let idx = BuiltInTypes::construct_int(i as isize) as usize;
            let elem = nth_fn(value as u64, idx as u64) as usize;
            result.push(self.value_to_string(elem)?);
        }
        Ok(result)
    }

    /// Convert a Beagle value to Option<String> (null -> None)
    fn value_to_optional_string(&self, value: usize) -> Result<Option<String>, CompileError> {
        if value == BuiltInTypes::null_value() as usize {
            Ok(None)
        } else {
            Ok(Some(self.value_to_string(value)?))
        }
    }

    /// Convert a Beagle value to bool
    fn value_to_bool(&self, value: usize) -> bool {
        value == BuiltInTypes::true_value() as usize
    }

    /// Convert a Beagle value to Vec<(Ast, Ast)> for map pairs
    fn value_to_ast_pairs(&self, value: usize) -> Result<Vec<(Ast, Ast)>, CompileError> {
        let runtime = get_runtime().get();

        let count_fn = runtime
            .get_function1("beagle.collections/vec-count")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.collections/vec-count".to_string(),
            })?;
        let count = BuiltInTypes::untag(count_fn(value as u64) as usize);

        let nth_fn = runtime
            .get_function2("beagle.collections/vec-get")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.collections/vec-get".to_string(),
            })?;

        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let idx = BuiltInTypes::construct_int(i as isize) as usize;
            let pair = nth_fn(value as u64, idx as u64) as usize;
            // Each pair is expected to be a 2-element structure or ArrowPair
            let key = self.value_to_ast(HeapObject::from_tagged(pair).get_field(0))?;
            let val = self.value_to_ast(HeapObject::from_tagged(pair).get_field(1))?;
            result.push((key, val));
        }
        Ok(result)
    }

    /// Convert a Beagle value to Vec<(String, Ast)> for named fields
    fn value_to_named_fields(&self, value: usize) -> Result<Vec<(String, Ast)>, CompileError> {
        let runtime = get_runtime().get();

        let count_fn = runtime
            .get_function1("beagle.collections/vec-count")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.collections/vec-count".to_string(),
            })?;
        let count = BuiltInTypes::untag(count_fn(value as u64) as usize);

        let nth_fn = runtime
            .get_function2("beagle.collections/vec-get")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.collections/vec-get".to_string(),
            })?;

        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let idx = BuiltInTypes::construct_int(i as isize) as usize;
            let field = nth_fn(value as u64, idx as u64) as usize;
            let heap_obj = HeapObject::from_tagged(field);
            let name = self.value_to_string(heap_obj.get_field(0))?;
            let val = self.value_to_ast(heap_obj.get_field(1))?;
            result.push((name, val));
        }
        Ok(result)
    }

    /// Convert a Beagle ConditionOperator value to Rust Condition enum
    fn value_to_condition_operator(&self, value: usize) -> Result<Condition, CompileError> {
        if !BuiltInTypes::is_heap_pointer(value) {
            return Err(CompileError::Internal(
                "Expected condition operator value".to_string(),
            ));
        }

        let heap_obj = HeapObject::from_tagged(value);
        let header = heap_obj.get_header();
        let struct_id = BuiltInTypes::untag(header.type_data as usize);

        let runtime = get_runtime().get();
        let struct_info = runtime
            .get_struct_by_id(struct_id)
            .ok_or_else(|| CompileError::Internal(format!("Unknown struct id: {}", struct_id)))?;

        let variant_name = struct_info
            .name
            .split('/')
            .next_back()
            .unwrap_or(&struct_info.name);

        match variant_name {
            "ConditionOperator.EqualOp" => Ok(Condition::Equal),
            "ConditionOperator.NotEqualOp" => Ok(Condition::NotEqual),
            "ConditionOperator.LessThanOp" => Ok(Condition::LessThan),
            "ConditionOperator.LessThanOrEqualOp" => Ok(Condition::LessThanOrEqual),
            "ConditionOperator.GreaterThanOp" => Ok(Condition::GreaterThan),
            "ConditionOperator.GreaterThanOrEqualOp" => Ok(Condition::GreaterThanOrEqual),
            other => Err(CompileError::Internal(format!(
                "Unknown condition operator variant: {}",
                other
            ))),
        }
    }

    /// Convert a Beagle value to Vec<MatchArm>
    fn value_to_match_arms(&self, value: usize) -> Result<Vec<MatchArm>, CompileError> {
        let runtime = get_runtime().get();

        let count_fn = runtime
            .get_function1("beagle.collections/vec-count")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.collections/vec-count".to_string(),
            })?;
        let count = BuiltInTypes::untag(count_fn(value as u64) as usize);

        let nth_fn = runtime
            .get_function2("beagle.collections/vec-get")
            .ok_or_else(|| CompileError::FunctionNotFound {
                function_name: "beagle.collections/vec-get".to_string(),
            })?;

        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let idx = BuiltInTypes::construct_int(i as isize) as usize;
            let arm = nth_fn(value as u64, idx as u64) as usize;
            result.push(self.value_to_match_arm(arm)?);
        }
        Ok(result)
    }

    /// Convert a single Beagle MatchArmDef to Rust MatchArm
    fn value_to_match_arm(&self, value: usize) -> Result<MatchArm, CompileError> {
        let heap_obj = HeapObject::from_tagged(value);
        let token_range = TokenRange::new(0, 0);

        // MatchArmDef { arm_pat, arm_grd, arm_bdy }
        let pattern = self.value_to_pattern(heap_obj.get_field(0))?;
        let guard_value = heap_obj.get_field(1);
        let guard = if guard_value == BuiltInTypes::null_value() as usize {
            None
        } else {
            Some(Box::new(self.value_to_ast(guard_value)?))
        };
        let body = self.value_to_ast_vec(heap_obj.get_field(2))?;

        Ok(MatchArm {
            pattern,
            guard,
            body,
            token_range,
        })
    }

    // ========================================================================
    // Macro collection and expansion
    // ========================================================================

    /// First pass: collect macro definitions from the AST
    ///
    /// This removes macro definitions from the AST and registers them.
    pub fn collect_macros(&mut self, ast: &Ast) -> Ast {
        match ast {
            Ast::Program {
                elements,
                token_range,
            } => {
                let mut new_elements = Vec::new();
                for element in elements {
                    match element {
                        Ast::Macro {
                            name,
                            args,
                            rest_param,
                            body,
                            ..
                        } => {
                            // Register the macro
                            let full_name = if self.current_namespace.is_empty() {
                                name.clone()
                            } else {
                                format!("{}/{}", self.current_namespace, name)
                            };
                            self.macros.insert(
                                full_name.clone(),
                                MacroDef {
                                    name: name.clone(),
                                    args: args.clone(),
                                    rest_param: rest_param.clone(),
                                    body: body.clone(),
                                },
                            );
                            // Also register without namespace for local calls
                            self.macros.insert(
                                name.clone(),
                                MacroDef {
                                    name: name.clone(),
                                    args: args.clone(),
                                    rest_param: rest_param.clone(),
                                    body: body.clone(),
                                },
                            );
                        }
                        Ast::Namespace { name, token_range } => {
                            self.current_namespace = name.clone();
                            new_elements.push(Ast::Namespace {
                                name: name.clone(),
                                token_range: *token_range,
                            });
                        }
                        Ast::Import {
                            library_name,
                            alias,
                            token_range,
                        } => {
                            // Track namespace alias
                            // The alias is an Ast (typically an Identifier)
                            if let Ast::Identifier(alias_name, _) = alias.as_ref() {
                                self.namespace_aliases
                                    .insert(alias_name.clone(), library_name.clone());
                            }
                            new_elements.push(Ast::Import {
                                library_name: library_name.clone(),
                                alias: alias.clone(),
                                token_range: *token_range,
                            });
                        }
                        _ => {
                            new_elements.push(self.collect_macros(element));
                        }
                    }
                }
                Ast::Program {
                    elements: new_elements,
                    token_range: *token_range,
                }
            }
            // For other nodes, recursively process children
            _ => self.collect_macros_in_node(ast),
        }
    }

    /// Recursively collect macros in a single AST node
    fn collect_macros_in_node(&mut self, ast: &Ast) -> Ast {
        match ast {
            Ast::Function {
                name,
                args,
                rest_param,
                body,
                token_range,
            } => Ast::Function {
                name: name.clone(),
                args: args.clone(),
                rest_param: rest_param.clone(),
                body: body.iter().map(|a| self.collect_macros(a)).collect(),
                token_range: *token_range,
            },
            // Pass through other nodes unchanged for now
            _ => ast.clone(),
        }
    }

    /// Compile procedural macros (macros that aren't simple quote-based)
    ///
    /// This generates and compiles wrapper functions for each procedural macro.
    /// Must be called after collect_macros and before expand.
    pub fn compile_procedural_macros(
        &mut self,
        compiler: &mut Compiler,
    ) -> Result<(), CompileError> {
        let macros_to_compile: Vec<(String, MacroDef)> = self
            .macros
            .iter()
            .filter(|(_, m)| !Self::is_simple_macro(m))
            .map(|(name, m)| (name.clone(), m.clone()))
            .collect();

        for (name, macro_def) in macros_to_compile {
            // Generate wrapper function code
            let func_code = self.generate_macro_function_code(&macro_def);
            let func_name = format!("__macros/__macro__{}", macro_def.name.replace('-', "_"));

            // Compile the generated function using the compiler
            // This creates a new MacroExpander internally, so no recursion issues
            let result = compiler.compile_string(&func_code);

            match result {
                Ok(_) => {
                    // Register the compiled function
                    self.compiled_macros.insert(name.clone(), func_name);
                }
                Err(e) => {
                    return Err(CompileError::MacroExpansionError {
                        name: name.clone(),
                        message: format!("Failed to compile procedural macro: {:?}", e),
                    });
                }
            }
        }

        Ok(())
    }

    /// Second pass: expand macro calls in the AST
    ///
    /// This replaces macro calls with their expanded forms.
    pub fn expand(&mut self, ast: &Ast) -> Result<Ast, CompileError> {
        match ast {
            Ast::Program {
                elements,
                token_range,
            } => {
                let mut new_elements = Vec::new();
                for element in elements {
                    new_elements.push(self.expand(element)?);
                }
                Ok(Ast::Program {
                    elements: new_elements,
                    token_range: *token_range,
                })
            }
            Ast::Call {
                name,
                args,
                token_range,
            } => {
                // Special handling for macroexpand debugging
                if name == "macroexpand" {
                    if args.len() != 1 {
                        return Err(CompileError::MacroExpansionError {
                            name: "macroexpand".to_string(),
                            message: "macroexpand takes exactly one argument".to_string(),
                        });
                    }

                    // Expand the argument
                    let expanded = self.expand(&args[0])?;

                    // Print during compilation
                    eprintln!("=== macroexpand ===");
                    eprintln!("{:#?}", expanded);
                    eprintln!("===================");

                    // Return expanded AST to be compiled
                    return Ok(expanded);
                }

                // Check if this is a macro call
                if let Some(macro_def) = self.macros.get(name).cloned() {
                    self.expand_macro_call(&macro_def, args, *token_range)
                } else {
                    // Regular function call - expand args recursively
                    let expanded_args: Result<Vec<Ast>, CompileError> =
                        args.iter().map(|a| self.expand(a)).collect();
                    Ok(Ast::Call {
                        name: name.clone(),
                        args: expanded_args?,
                        token_range: *token_range,
                    })
                }
            }
            // Recursively expand other nodes
            Ast::Function {
                name,
                args,
                rest_param,
                body,
                token_range,
            } => {
                let expanded_body: Result<Vec<Ast>, CompileError> =
                    body.iter().map(|a| self.expand(a)).collect();
                Ok(Ast::Function {
                    name: name.clone(),
                    args: args.clone(),
                    rest_param: rest_param.clone(),
                    body: expanded_body?,
                    token_range: *token_range,
                })
            }
            Ast::If {
                condition,
                then,
                else_,
                token_range,
            } => {
                let expanded_then: Result<Vec<Ast>, CompileError> =
                    then.iter().map(|a| self.expand(a)).collect();
                let expanded_else: Result<Vec<Ast>, CompileError> =
                    else_.iter().map(|a| self.expand(a)).collect();
                Ok(Ast::If {
                    condition: Box::new(self.expand(condition)?),
                    then: expanded_then?,
                    else_: expanded_else?,
                    token_range: *token_range,
                })
            }
            Ast::Block {
                statements,
                token_range,
            } => {
                let expanded_statements: Result<Vec<Ast>, CompileError> =
                    statements.iter().map(|a| self.expand(a)).collect();
                Ok(Ast::Block {
                    statements: expanded_statements?,
                    token_range: *token_range,
                })
            }
            Ast::Let {
                pattern,
                value,
                token_range,
            } => Ok(Ast::Let {
                pattern: pattern.clone(),
                value: Box::new(self.expand(value)?),
                token_range: *token_range,
            }),
            Ast::Add {
                left,
                right,
                token_range,
            } => Ok(Ast::Add {
                left: Box::new(self.expand(left)?),
                right: Box::new(self.expand(right)?),
                token_range: *token_range,
            }),
            Ast::Sub {
                left,
                right,
                token_range,
            } => Ok(Ast::Sub {
                left: Box::new(self.expand(left)?),
                right: Box::new(self.expand(right)?),
                token_range: *token_range,
            }),
            Ast::Mul {
                left,
                right,
                token_range,
            } => Ok(Ast::Mul {
                left: Box::new(self.expand(left)?),
                right: Box::new(self.expand(right)?),
                token_range: *token_range,
            }),
            Ast::Div {
                left,
                right,
                token_range,
            } => Ok(Ast::Div {
                left: Box::new(self.expand(left)?),
                right: Box::new(self.expand(right)?),
                token_range: *token_range,
            }),
            Ast::Array { array, token_range } => {
                let expanded: Result<Vec<Ast>, CompileError> =
                    array.iter().map(|a| self.expand(a)).collect();
                Ok(Ast::Array {
                    array: expanded?,
                    token_range: *token_range,
                })
            }
            Ast::Match {
                value,
                arms,
                token_range,
            } => {
                let expanded_arms: Result<Vec<MatchArm>, CompileError> = arms
                    .iter()
                    .map(|arm| {
                        let expanded_guard =
                            arm.guard.as_ref().map(|g| self.expand(g)).transpose()?;
                        let expanded_body: Result<Vec<Ast>, CompileError> =
                            arm.body.iter().map(|a| self.expand(a)).collect();
                        Ok(MatchArm {
                            pattern: arm.pattern.clone(),
                            guard: expanded_guard.map(Box::new),
                            body: expanded_body?,
                            token_range: arm.token_range,
                        })
                    })
                    .collect();
                Ok(Ast::Match {
                    value: Box::new(self.expand(value)?),
                    arms: expanded_arms?,
                    token_range: *token_range,
                })
            }
            // Pass through literals and other simple nodes unchanged
            Ast::IntegerLiteral(_, _)
            | Ast::FloatLiteral(_, _)
            | Ast::String(_, _)
            | Ast::Identifier(_, _)
            | Ast::True(_)
            | Ast::False(_)
            | Ast::Null(_)
            | Ast::Keyword(_, _)
            | Ast::Namespace { .. } => Ok(ast.clone()),
            // For now, pass through other complex nodes unchanged
            // TODO: Implement expansion for all node types
            _ => Ok(ast.clone()),
        }
    }

    /// Expand a single macro call
    fn expand_macro_call(
        &mut self,
        macro_def: &MacroDef,
        args: &[Ast],
        _call_site: TokenRange,
    ) -> Result<Ast, CompileError> {
        // Check arity
        let expected = macro_def.args.len();
        let got = args.len();

        if macro_def.rest_param.is_none() && got != expected {
            return Err(CompileError::ArityMismatch {
                function_name: macro_def.name.clone(),
                expected,
                got,
                is_variadic: false,
            });
        }

        if macro_def.rest_param.is_some() && got < expected {
            return Err(CompileError::ArityMismatch {
                function_name: macro_def.name.clone(),
                expected,
                got,
                is_variadic: true,
            });
        }

        // Collect free identifiers from user-provided arguments for hygiene
        // These are identifiers that should NOT be renamed by the macro
        let mut user_identifiers: HashSet<String> = HashSet::new();
        for arg in args {
            let free_ids = self.collect_free_identifiers(arg);
            user_identifiers.extend(free_ids);
        }

        // Check if this is a compiled procedural macro
        if let Some(func_name) = self.compiled_macros.get(&macro_def.name).cloned() {
            // Procedural macro: call the compiled function with AST values
            return self.expand_procedural_macro(
                &func_name,
                &macro_def.name,
                args,
                &user_identifiers,
            );
        }

        // Simple macro: use template substitution
        self.expand_simple_macro(macro_def, args, &user_identifiers)
    }

    /// Expand a procedural macro by calling its compiled function
    fn expand_procedural_macro(
        &mut self,
        func_name: &str,
        macro_name: &str,
        args: &[Ast],
        user_identifiers: &HashSet<String>,
    ) -> Result<Ast, CompileError> {
        // Convert each argument to a Beagle AST value
        let mut arg_values: Vec<usize> = Vec::with_capacity(args.len());
        for arg in args.iter() {
            let value = self.ast_to_value(arg)?;
            arg_values.push(value);
        }

        // Call the compiled macro function
        let result = self.call_make_fn(func_name, &arg_values).map_err(|e| {
            CompileError::MacroRuntimeError {
                name: macro_name.to_string(),
                message: format!("Failed to call macro function: {:?}", e),
            }
        })?;

        // Convert the result back to an AST
        let expanded = self
            .value_to_ast(result)
            .map_err(|e| CompileError::MacroRuntimeError {
                name: macro_name.to_string(),
                message: format!("Failed to convert macro result to AST: {:?}", e),
            })?;

        // Apply hygiene to rename macro-introduced bindings
        let hygienic = self.apply_hygiene(expanded, user_identifiers)?;

        // Recursively expand in case the result contains more macro calls
        self.expand(&hygienic)
    }

    /// Expand a simple (template-based) macro using substitution
    fn expand_simple_macro(
        &mut self,
        macro_def: &MacroDef,
        args: &[Ast],
        user_identifiers: &HashSet<String>,
    ) -> Result<Ast, CompileError> {
        let expected = macro_def.args.len();

        // Build substitution map from pattern args to AST values
        let mut substitutions: HashMap<String, Ast> = HashMap::new();

        for (i, pattern) in macro_def.args.iter().enumerate() {
            if let Pattern::Identifier { name, .. } = pattern
                && i < args.len()
            {
                substitutions.insert(name.clone(), args[i].clone());
            }
        }

        // Handle rest param
        if let Some(rest_name) = &macro_def.rest_param {
            let rest_args: Vec<Ast> = args[expected..].to_vec();
            substitutions.insert(
                rest_name.clone(),
                Ast::Array {
                    array: rest_args,
                    token_range: TokenRange::new(0, 0),
                },
            );
        }

        // For simple macros, evaluate the body with substitutions
        if macro_def.body.len() == 1 {
            let expanded = self.expand_macro_body(&macro_def.body[0], &substitutions)?;
            // Apply hygiene to rename macro-introduced bindings
            let hygienic = self.apply_hygiene(expanded, user_identifiers)?;
            // Recursively expand in case the result contains more macro calls
            self.expand(&hygienic)
        } else {
            // Multi-statement macro body - return first expression (simplified)
            // TODO: Handle multi-statement bodies properly
            if let Some(first) = macro_def.body.first() {
                let expanded = self.expand_macro_body(first, &substitutions)?;
                // Apply hygiene to rename macro-introduced bindings
                let hygienic = self.apply_hygiene(expanded, user_identifiers)?;
                self.expand(&hygienic)
            } else {
                Ok(Ast::Null(0))
            }
        }
    }

    /// Expand a macro body expression, unwrapping Quote to return actual AST
    fn expand_macro_body(
        &self,
        ast: &Ast,
        subs: &HashMap<String, Ast>,
    ) -> Result<Ast, CompileError> {
        match ast {
            Ast::Quote { body, .. } => {
                // Macro body is quote { ... }, unwrap and substitute unquotes
                self.unwrap_quote(body, subs)
            }
            _ => {
                // Not a quote, just substitute normally
                self.substitute(ast, subs)
            }
        }
    }

    /// Unwrap a quoted expression, replacing unquotes with their substituted values
    fn unwrap_quote(&self, ast: &Ast, subs: &HashMap<String, Ast>) -> Result<Ast, CompileError> {
        match ast {
            Ast::Unquote { body, .. } => {
                // Unquote: replace with the substituted value
                if let Ast::Identifier(name, _) = body.as_ref()
                    && let Some(replacement) = subs.get(name)
                {
                    return Ok(replacement.clone());
                }
                // If not a simple identifier, recursively unwrap
                self.unwrap_quote(body, subs)
            }
            // Recursively unwrap children
            Ast::If {
                condition,
                then,
                else_,
                token_range,
            } => {
                let unwrapped_then: Result<Vec<Ast>, CompileError> =
                    then.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                let unwrapped_else: Result<Vec<Ast>, CompileError> =
                    else_.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                Ok(Ast::If {
                    condition: Box::new(self.unwrap_quote(condition, subs)?),
                    then: unwrapped_then?,
                    else_: unwrapped_else?,
                    token_range: *token_range,
                })
            }
            Ast::Block {
                statements,
                token_range,
            } => {
                let unwrapped_statements: Result<Vec<Ast>, CompileError> = statements
                    .iter()
                    .map(|a| self.unwrap_quote(a, subs))
                    .collect();
                Ok(Ast::Block {
                    statements: unwrapped_statements?,
                    token_range: *token_range,
                })
            }
            Ast::Add {
                left,
                right,
                token_range,
            } => Ok(Ast::Add {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Sub {
                left,
                right,
                token_range,
            } => Ok(Ast::Sub {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Mul {
                left,
                right,
                token_range,
            } => Ok(Ast::Mul {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Div {
                left,
                right,
                token_range,
            } => Ok(Ast::Div {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Condition {
                operator,
                left,
                right,
                token_range,
            } => Ok(Ast::Condition {
                operator: *operator,
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Call {
                name,
                args,
                token_range,
            } => {
                let unwrapped_args: Result<Vec<Ast>, CompileError> =
                    args.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                Ok(Ast::Call {
                    name: name.clone(),
                    args: unwrapped_args?,
                    token_range: *token_range,
                })
            }
            Ast::Array { array, token_range } => {
                let unwrapped = self.unwrap_quote_array_with_splice(array, subs)?;
                Ok(Ast::Array {
                    array: unwrapped,
                    token_range: *token_range,
                })
            }
            Ast::Let {
                pattern,
                value,
                token_range,
            } => Ok(Ast::Let {
                pattern: pattern.clone(),
                value: Box::new(self.unwrap_quote(value, subs)?),
                token_range: *token_range,
            }),
            Ast::LetMut {
                pattern,
                value,
                token_range,
            } => Ok(Ast::LetMut {
                pattern: pattern.clone(),
                value: Box::new(self.unwrap_quote(value, subs)?),
                token_range: *token_range,
            }),
            // More binary operators
            Ast::Modulo {
                left,
                right,
                token_range,
            } => Ok(Ast::Modulo {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::ShiftLeft {
                left,
                right,
                token_range,
            } => Ok(Ast::ShiftLeft {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::ShiftRight {
                left,
                right,
                token_range,
            } => Ok(Ast::ShiftRight {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::ShiftRightZero {
                left,
                right,
                token_range,
            } => Ok(Ast::ShiftRightZero {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::BitWiseAnd {
                left,
                right,
                token_range,
            } => Ok(Ast::BitWiseAnd {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::BitWiseOr {
                left,
                right,
                token_range,
            } => Ok(Ast::BitWiseOr {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::BitWiseXor {
                left,
                right,
                token_range,
            } => Ok(Ast::BitWiseXor {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::And {
                left,
                right,
                token_range,
            } => Ok(Ast::And {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Or {
                left,
                right,
                token_range,
            } => Ok(Ast::Or {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            // Property and index access
            Ast::PropertyAccess {
                object,
                property,
                token_range,
            } => Ok(Ast::PropertyAccess {
                object: Box::new(self.unwrap_quote(object, subs)?),
                property: Box::new(self.unwrap_quote(property, subs)?),
                token_range: *token_range,
            }),
            Ast::IndexOperator {
                array,
                index,
                token_range,
            } => Ok(Ast::IndexOperator {
                array: Box::new(self.unwrap_quote(array, subs)?),
                index: Box::new(self.unwrap_quote(index, subs)?),
                token_range: *token_range,
            }),
            // Control flow
            Ast::Loop { body, token_range } => {
                let unwrapped_body: Result<Vec<Ast>, CompileError> =
                    body.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                Ok(Ast::Loop {
                    body: unwrapped_body?,
                    token_range: *token_range,
                })
            }
            Ast::While {
                condition,
                body,
                token_range,
            } => {
                let unwrapped_body: Result<Vec<Ast>, CompileError> =
                    body.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                Ok(Ast::While {
                    condition: Box::new(self.unwrap_quote(condition, subs)?),
                    body: unwrapped_body?,
                    token_range: *token_range,
                })
            }
            Ast::For {
                binding,
                collection,
                body,
                token_range,
            } => {
                let unwrapped_body: Result<Vec<Ast>, CompileError> =
                    body.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                Ok(Ast::For {
                    binding: binding.clone(),
                    collection: Box::new(self.unwrap_quote(collection, subs)?),
                    body: unwrapped_body?,
                    token_range: *token_range,
                })
            }
            Ast::Break { value, token_range } => Ok(Ast::Break {
                value: Box::new(self.unwrap_quote(value, subs)?),
                token_range: *token_range,
            }),
            Ast::Assignment {
                name,
                value,
                token_range,
            } => Ok(Ast::Assignment {
                name: Box::new(self.unwrap_quote(name, subs)?),
                value: Box::new(self.unwrap_quote(value, subs)?),
                token_range: *token_range,
            }),
            // Exception handling
            Ast::Throw { value, token_range } => Ok(Ast::Throw {
                value: Box::new(self.unwrap_quote(value, subs)?),
                token_range: *token_range,
            }),
            Ast::Try {
                body,
                exception_binding,
                catch_body,
                token_range,
            } => {
                let unwrapped_body: Result<Vec<Ast>, CompileError> =
                    body.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                let unwrapped_catch: Result<Vec<Ast>, CompileError> = catch_body
                    .iter()
                    .map(|a| self.unwrap_quote(a, subs))
                    .collect();
                Ok(Ast::Try {
                    body: unwrapped_body?,
                    exception_binding: exception_binding.clone(),
                    catch_body: unwrapped_catch?,
                    token_range: *token_range,
                })
            }
            // Match
            Ast::Match {
                value,
                arms,
                token_range,
            } => {
                let unwrapped_arms: Result<Vec<MatchArm>, CompileError> = arms
                    .iter()
                    .map(|arm| {
                        let unwrapped_guard = arm
                            .guard
                            .as_ref()
                            .map(|g| self.unwrap_quote(g, subs).map(Box::new))
                            .transpose()?;
                        let unwrapped_body: Result<Vec<Ast>, CompileError> = arm
                            .body
                            .iter()
                            .map(|a| self.unwrap_quote(a, subs))
                            .collect();
                        Ok(MatchArm {
                            pattern: arm.pattern.clone(),
                            guard: unwrapped_guard,
                            body: unwrapped_body?,
                            token_range: arm.token_range,
                        })
                    })
                    .collect();
                Ok(Ast::Match {
                    value: Box::new(self.unwrap_quote(value, subs)?),
                    arms: unwrapped_arms?,
                    token_range: *token_range,
                })
            }
            // Anonymous functions
            Ast::Function {
                name,
                args,
                rest_param,
                body,
                token_range,
            } => {
                let unwrapped_body: Result<Vec<Ast>, CompileError> =
                    body.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                Ok(Ast::Function {
                    name: name.clone(),
                    args: args.clone(),
                    rest_param: rest_param.clone(),
                    body: unwrapped_body?,
                    token_range: *token_range,
                })
            }
            // Recurse/TailRecurse
            Ast::Recurse { args, token_range } => {
                let unwrapped_args: Result<Vec<Ast>, CompileError> =
                    args.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                Ok(Ast::Recurse {
                    args: unwrapped_args?,
                    token_range: *token_range,
                })
            }
            Ast::TailRecurse { args, token_range } => {
                let unwrapped_args: Result<Vec<Ast>, CompileError> =
                    args.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                Ok(Ast::TailRecurse {
                    args: unwrapped_args?,
                    token_range: *token_range,
                })
            }
            // Collections
            Ast::MapLiteral { pairs, token_range } => {
                let unwrapped_pairs: Result<Vec<(Ast, Ast)>, CompileError> = pairs
                    .iter()
                    .map(|(k, v)| Ok((self.unwrap_quote(k, subs)?, self.unwrap_quote(v, subs)?)))
                    .collect();
                Ok(Ast::MapLiteral {
                    pairs: unwrapped_pairs?,
                    token_range: *token_range,
                })
            }
            Ast::SetLiteral {
                elements,
                token_range,
            } => {
                let unwrapped: Result<Vec<Ast>, CompileError> = elements
                    .iter()
                    .map(|a| self.unwrap_quote(a, subs))
                    .collect();
                Ok(Ast::SetLiteral {
                    elements: unwrapped?,
                    token_range: *token_range,
                })
            }
            // Struct and enum creation
            Ast::StructCreation {
                name,
                fields,
                token_range,
            } => {
                let unwrapped_fields: Result<Vec<(String, Ast)>, CompileError> = fields
                    .iter()
                    .map(|(field_name, field_value)| {
                        Ok((field_name.clone(), self.unwrap_quote(field_value, subs)?))
                    })
                    .collect();
                Ok(Ast::StructCreation {
                    name: name.clone(),
                    fields: unwrapped_fields?,
                    token_range: *token_range,
                })
            }
            Ast::EnumCreation {
                name,
                variant,
                fields,
                token_range,
            } => {
                let unwrapped_fields: Result<Vec<(String, Ast)>, CompileError> = fields
                    .iter()
                    .map(|(field_name, field_value)| {
                        Ok((field_name.clone(), self.unwrap_quote(field_value, subs)?))
                    })
                    .collect();
                Ok(Ast::EnumCreation {
                    name: name.clone(),
                    variant: variant.clone(),
                    fields: unwrapped_fields?,
                    token_range: *token_range,
                })
            }
            // Effect handling
            Ast::Reset { body, token_range } => {
                let unwrapped_body: Result<Vec<Ast>, CompileError> =
                    body.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                Ok(Ast::Reset {
                    body: unwrapped_body?,
                    token_range: *token_range,
                })
            }
            Ast::Shift {
                continuation_param,
                body,
                token_range,
            } => {
                let unwrapped_body: Result<Vec<Ast>, CompileError> =
                    body.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                Ok(Ast::Shift {
                    continuation_param: continuation_param.clone(),
                    body: unwrapped_body?,
                    token_range: *token_range,
                })
            }
            Ast::Perform { value, token_range } => Ok(Ast::Perform {
                value: Box::new(self.unwrap_quote(value, subs)?),
                token_range: *token_range,
            }),
            Ast::Handle {
                protocol,
                protocol_type_args,
                handler_instance,
                body,
                token_range,
            } => {
                let unwrapped_body: Result<Vec<Ast>, CompileError> =
                    body.iter().map(|a| self.unwrap_quote(a, subs)).collect();
                Ok(Ast::Handle {
                    protocol: protocol.clone(),
                    protocol_type_args: protocol_type_args.clone(),
                    handler_instance: Box::new(self.unwrap_quote(handler_instance, subs)?),
                    body: unwrapped_body?,
                    token_range: *token_range,
                })
            }
            // String interpolation
            Ast::StringInterpolation { parts, token_range } => {
                let unwrapped_parts: Result<Vec<StringInterpolationPart>, CompileError> = parts
                    .iter()
                    .map(|part| match part {
                        StringInterpolationPart::Literal(s) => {
                            Ok(StringInterpolationPart::Literal(s.clone()))
                        }
                        StringInterpolationPart::Expression(expr) => {
                            Ok(StringInterpolationPart::Expression(Box::new(
                                self.unwrap_quote(expr, subs)?,
                            )))
                        }
                    })
                    .collect();
                Ok(Ast::StringInterpolation {
                    parts: unwrapped_parts?,
                    token_range: *token_range,
                })
            }
            // ArrowPair (used in macro-friendly syntax)
            Ast::ArrowPair {
                left,
                right,
                token_range,
            } => Ok(Ast::ArrowPair {
                left: Box::new(self.unwrap_quote(left, subs)?),
                right: Box::new(self.unwrap_quote(right, subs)?),
                token_range: *token_range,
            }),
            // Nested quote (don't descend into it, but unwrap the outer level)
            Ast::Quote { body, token_range } => Ok(Ast::Quote {
                body: Box::new(self.unwrap_quote(body, subs)?),
                token_range: *token_range,
            }),
            // Pass through literals and other nodes unchanged
            _ => Ok(ast.clone()),
        }
    }

    /// Unwrap an array with support for unquote-splice
    fn unwrap_quote_array_with_splice(
        &self,
        elements: &[Ast],
        subs: &HashMap<String, Ast>,
    ) -> Result<Vec<Ast>, CompileError> {
        let mut result = Vec::new();
        for elem in elements {
            match elem {
                Ast::UnquoteSplice { body, .. } => {
                    // Get the array value from substitution
                    if let Ast::Identifier(name, _) = body.as_ref()
                        && let Some(Ast::Array { array, .. }) = subs.get(name)
                    {
                        // Splice all elements
                        for spliced in array {
                            result.push(self.unwrap_quote(spliced, subs)?);
                        }
                        continue;
                    }
                    // If not a simple identifier or not an array, just unwrap
                    result.push(self.unwrap_quote(elem, subs)?);
                }
                _ => result.push(self.unwrap_quote(elem, subs)?),
            }
        }
        Ok(result)
    }

    /// Substitute identifiers in the AST with their values from the substitution map
    fn substitute(&self, ast: &Ast, subs: &HashMap<String, Ast>) -> Result<Ast, CompileError> {
        match ast {
            Ast::Quote { body, token_range } => {
                // Inside quote, substitute unquoted expressions
                Ok(Ast::Quote {
                    body: Box::new(self.substitute_in_quote(body, subs)?),
                    token_range: *token_range,
                })
            }
            Ast::Unquote { body, token_range } => {
                // Unquote: substitute and evaluate the identifier
                if let Ast::Identifier(name, _) = body.as_ref()
                    && let Some(replacement) = subs.get(name)
                {
                    return Ok(replacement.clone());
                }
                // If not a simple identifier or no substitution, return as-is
                Ok(Ast::Unquote {
                    body: Box::new(self.substitute(body, subs)?),
                    token_range: *token_range,
                })
            }
            Ast::Identifier(name, pos) => {
                // Outside quote, don't substitute identifiers
                // (They are code, not AST data)
                Ok(Ast::Identifier(name.clone(), *pos))
            }
            Ast::If {
                condition,
                then,
                else_,
                token_range,
            } => {
                let expanded_then: Result<Vec<Ast>, CompileError> =
                    then.iter().map(|a| self.substitute(a, subs)).collect();
                let expanded_else: Result<Vec<Ast>, CompileError> =
                    else_.iter().map(|a| self.substitute(a, subs)).collect();
                Ok(Ast::If {
                    condition: Box::new(self.substitute(condition, subs)?),
                    then: expanded_then?,
                    else_: expanded_else?,
                    token_range: *token_range,
                })
            }
            Ast::Block {
                statements,
                token_range,
            } => {
                let expanded_statements: Result<Vec<Ast>, CompileError> = statements
                    .iter()
                    .map(|a| self.substitute(a, subs))
                    .collect();
                Ok(Ast::Block {
                    statements: expanded_statements?,
                    token_range: *token_range,
                })
            }
            Ast::Call {
                name,
                args,
                token_range,
            } => {
                let new_args: Result<Vec<Ast>, CompileError> =
                    args.iter().map(|a| self.substitute(a, subs)).collect();
                Ok(Ast::Call {
                    name: name.clone(),
                    args: new_args?,
                    token_range: *token_range,
                })
            }
            Ast::Add {
                left,
                right,
                token_range,
            } => Ok(Ast::Add {
                left: Box::new(self.substitute(left, subs)?),
                right: Box::new(self.substitute(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Sub {
                left,
                right,
                token_range,
            } => Ok(Ast::Sub {
                left: Box::new(self.substitute(left, subs)?),
                right: Box::new(self.substitute(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Mul {
                left,
                right,
                token_range,
            } => Ok(Ast::Mul {
                left: Box::new(self.substitute(left, subs)?),
                right: Box::new(self.substitute(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Div {
                left,
                right,
                token_range,
            } => Ok(Ast::Div {
                left: Box::new(self.substitute(left, subs)?),
                right: Box::new(self.substitute(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Array { array, token_range } => {
                let new_array: Result<Vec<Ast>, CompileError> =
                    array.iter().map(|a| self.substitute(a, subs)).collect();
                Ok(Ast::Array {
                    array: new_array?,
                    token_range: *token_range,
                })
            }
            // Pass through literals unchanged
            _ => Ok(ast.clone()),
        }
    }

    /// Substitute inside a quoted expression (only process unquotes)
    fn substitute_in_quote(
        &self,
        ast: &Ast,
        subs: &HashMap<String, Ast>,
    ) -> Result<Ast, CompileError> {
        match ast {
            Ast::Unquote { body, token_range } => {
                // Inside quote, unquote means splice the substituted value
                if let Ast::Identifier(name, _) = body.as_ref()
                    && let Some(replacement) = subs.get(name)
                {
                    // Wrap the replacement in an Unquote to convert it back to AST
                    // This is the key: ~x in quote { ~x + 1 } becomes the AST for x's value
                    return Ok(Ast::Unquote {
                        body: Box::new(replacement.clone()),
                        token_range: *token_range,
                    });
                }
                Ok(ast.clone())
            }
            Ast::UnquoteSplice { body, token_range } => {
                // Handle ~@expr for splicing arrays
                if let Ast::Identifier(name, _) = body.as_ref()
                    && let Some(replacement) = subs.get(name)
                {
                    return Ok(Ast::UnquoteSplice {
                        body: Box::new(replacement.clone()),
                        token_range: *token_range,
                    });
                }
                Ok(ast.clone())
            }
            // Recursively process children for other nodes
            Ast::Add {
                left,
                right,
                token_range,
            } => Ok(Ast::Add {
                left: Box::new(self.substitute_in_quote(left, subs)?),
                right: Box::new(self.substitute_in_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Sub {
                left,
                right,
                token_range,
            } => Ok(Ast::Sub {
                left: Box::new(self.substitute_in_quote(left, subs)?),
                right: Box::new(self.substitute_in_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Mul {
                left,
                right,
                token_range,
            } => Ok(Ast::Mul {
                left: Box::new(self.substitute_in_quote(left, subs)?),
                right: Box::new(self.substitute_in_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::Div {
                left,
                right,
                token_range,
            } => Ok(Ast::Div {
                left: Box::new(self.substitute_in_quote(left, subs)?),
                right: Box::new(self.substitute_in_quote(right, subs)?),
                token_range: *token_range,
            }),
            Ast::If {
                condition,
                then,
                else_,
                token_range,
            } => {
                let expanded_then: Result<Vec<Ast>, CompileError> = then
                    .iter()
                    .map(|a| self.substitute_in_quote(a, subs))
                    .collect();
                let expanded_else: Result<Vec<Ast>, CompileError> = else_
                    .iter()
                    .map(|a| self.substitute_in_quote(a, subs))
                    .collect();
                Ok(Ast::If {
                    condition: Box::new(self.substitute_in_quote(condition, subs)?),
                    then: expanded_then?,
                    else_: expanded_else?,
                    token_range: *token_range,
                })
            }
            Ast::Block {
                statements,
                token_range,
            } => {
                let expanded_statements: Result<Vec<Ast>, CompileError> = statements
                    .iter()
                    .map(|a| self.substitute_in_quote(a, subs))
                    .collect();
                Ok(Ast::Block {
                    statements: expanded_statements?,
                    token_range: *token_range,
                })
            }
            Ast::Call {
                name,
                args,
                token_range,
            } => {
                let new_args: Result<Vec<Ast>, CompileError> = args
                    .iter()
                    .map(|a| self.substitute_in_quote(a, subs))
                    .collect();
                Ok(Ast::Call {
                    name: name.clone(),
                    args: new_args?,
                    token_range: *token_range,
                })
            }
            Ast::Array { array, token_range } => {
                let new_array: Result<Vec<Ast>, CompileError> = array
                    .iter()
                    .map(|a| self.substitute_in_quote(a, subs))
                    .collect();
                Ok(Ast::Array {
                    array: new_array?,
                    token_range: *token_range,
                })
            }
            // Pass through other nodes unchanged
            _ => Ok(ast.clone()),
        }
    }

    /// Check if a name refers to a macro
    #[allow(dead_code)]
    pub fn is_macro(&self, name: &str) -> bool {
        self.macros.contains_key(name)
    }

    /// Get all registered macro names
    #[allow(dead_code)]
    pub fn macro_names(&self) -> Vec<String> {
        self.macros.keys().cloned().collect()
    }
}

impl Default for MacroExpander {
    fn default() -> Self {
        Self::new()
    }
}
