use core::panic;
use ir::{Ir, Value, VirtualRegister};
use std::{collections::HashMap, hash::Hash};

use crate::{
    Data, Message,
    backend::{Backend, CodegenBackend},
    builtins::debugger,
    common::Label,
    compiler::Compiler,
    ir::{self, Condition},
    pretty_print::PrettyPrint,
    runtime::{Enum, EnumVariant, Struct},
    types::BuiltInTypes,
};

type TokenPosition = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenRange {
    pub start: usize,
    pub end: usize,
}

impl TokenRange {
    pub fn new(start: usize, end: usize) -> Self {
        TokenRange { start, end }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ast {
    Program {
        elements: Vec<Ast>,
        token_range: TokenRange,
    },
    Function {
        name: Option<String>,
        // TODO: Change this to a Vec<Ast>
        args: Vec<String>,
        rest_param: Option<String>,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    Struct {
        name: String,
        fields: Vec<Ast>,
        token_range: TokenRange,
    },
    Enum {
        name: String,
        variants: Vec<Ast>,
        token_range: TokenRange,
    },
    EnumVariant {
        name: String,
        fields: Vec<Ast>,
        token_range: TokenRange,
    },
    EnumStaticVariant {
        name: String,
        token_range: TokenRange,
    },
    Protocol {
        name: String,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    Extend {
        target_type: String,
        protocol: String,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    FunctionStub {
        name: String,
        args: Vec<String>,
        rest_param: Option<String>,
        token_range: TokenRange,
    },
    If {
        condition: Box<Ast>,
        then: Vec<Ast>,
        else_: Vec<Ast>,
        token_range: TokenRange,
    },
    Condition {
        operator: Condition,
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Add {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Sub {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Mul {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Div {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Recurse {
        args: Vec<Ast>,
        token_range: TokenRange,
    },
    TailRecurse {
        args: Vec<Ast>,
        token_range: TokenRange,
    },
    Call {
        name: String,
        args: Vec<Ast>,
        token_range: TokenRange,
    },
    Let {
        name: Box<Ast>,
        value: Box<Ast>,
        token_range: TokenRange,
    },
    LetMut {
        name: Box<Ast>,
        value: Box<Ast>,
        token_range: TokenRange,
    },
    IntegerLiteral(i64, TokenPosition),
    FloatLiteral(String, TokenPosition),
    Identifier(String, TokenPosition),
    String(String, TokenPosition),
    Keyword(String, TokenPosition),
    True(TokenPosition),
    False(TokenPosition),
    StructCreation {
        name: String,
        fields: Vec<(String, Ast)>,
        token_range: TokenRange,
    },
    PropertyAccess {
        object: Box<Ast>,
        property: Box<Ast>,
        token_range: TokenRange,
    },
    Null(TokenPosition),
    EnumCreation {
        name: String,
        variant: String,
        fields: Vec<(String, Ast)>,
        token_range: TokenRange,
    },
    Namespace {
        name: String,
        token_range: TokenRange,
    },
    Import {
        library_name: String,
        alias: Box<Ast>,
        token_range: TokenRange,
    },
    ShiftLeft {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    ShiftRight {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    ShiftRightZero {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    BitWiseAnd {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    BitWiseOr {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    BitWiseXor {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    And {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Or {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Array {
        array: Vec<Ast>,
        token_range: TokenRange,
    },
    MapLiteral {
        pairs: Vec<(Ast, Ast)>,
        token_range: TokenRange,
    },
    IndexOperator {
        array: Box<Ast>,
        index: Box<Ast>,
        token_range: TokenRange,
    },
    Loop {
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    While {
        condition: Box<Ast>,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    Break {
        value: Box<Ast>,
        token_range: TokenRange,
    },
    Continue {
        token_range: TokenRange,
    },
    For {
        binding: String,
        collection: Box<Ast>,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    Assignment {
        name: Box<Ast>,
        value: Box<Ast>,
        token_range: TokenRange,
    },
    Try {
        body: Vec<Ast>,
        exception_binding: String,
        catch_body: Vec<Ast>,
        token_range: TokenRange,
    },
    Throw {
        value: Box<Ast>,
        token_range: TokenRange,
    },
    Match {
        value: Box<Ast>,
        arms: Vec<MatchArm>,
        token_range: TokenRange,
    },
    /// Protocol dispatch with inline caching
    /// Generates optimized dispatch code using a dispatch table and inline cache
    ProtocolDispatch {
        /// Arguments to the protocol method (first arg is the dispatch target)
        args: Vec<String>,
        /// Address of the 16-byte inline cache [type_id, fn_ptr]
        cache_location: usize,
        /// Address of the DispatchTable struct
        dispatch_table_ptr: usize,
        /// Default method function pointer (0 if none)
        default_fn_ptr: usize,
        /// Number of arguments the protocol method expects
        num_args: usize,
        token_range: TokenRange,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Box<Ast>>,
    pub body: Vec<Ast>,
    pub token_range: TokenRange,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pattern {
    EnumVariant {
        enum_name: String,
        variant_name: String,
        fields: Vec<FieldPattern>,
        token_range: TokenRange,
    },
    Literal {
        value: Box<Ast>,
        token_range: TokenRange,
    },
    Wildcard {
        token_range: TokenRange,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldPattern {
    pub field_name: String,
    pub binding_name: Option<String>,
    pub token_range: TokenRange,
}

impl Ast {
    pub fn token_range(&self) -> TokenRange {
        match self {
            Ast::Program { token_range, .. }
            | Ast::Function { token_range, .. }
            | Ast::Struct { token_range, .. }
            | Ast::Enum { token_range, .. }
            | Ast::EnumVariant { token_range, .. }
            | Ast::EnumStaticVariant { token_range, .. }
            | Ast::Protocol { token_range, .. }
            | Ast::Extend { token_range, .. }
            | Ast::FunctionStub { token_range, .. }
            | Ast::If { token_range, .. }
            | Ast::Condition { token_range, .. }
            | Ast::Add { token_range, .. }
            | Ast::Sub { token_range, .. }
            | Ast::Mul { token_range, .. }
            | Ast::Div { token_range, .. }
            | Ast::Recurse { token_range, .. }
            | Ast::TailRecurse { token_range, .. }
            | Ast::Call { token_range, .. }
            | Ast::Let { token_range, .. }
            | Ast::LetMut { token_range, .. }
            | Ast::Assignment { token_range, .. }
            | Ast::Namespace { token_range, .. }
            | Ast::Import { token_range, .. }
            | Ast::ShiftLeft { token_range, .. }
            | Ast::ShiftRight { token_range, .. }
            | Ast::ShiftRightZero { token_range, .. }
            | Ast::BitWiseAnd { token_range, .. }
            | Ast::BitWiseOr { token_range, .. }
            | Ast::BitWiseXor { token_range, .. }
            | Ast::And { token_range, .. }
            | Ast::Or { token_range, .. }
            | Ast::Array { token_range, .. }
            | Ast::MapLiteral { token_range, .. }
            | Ast::IndexOperator { token_range, .. }
            | Ast::Loop { token_range, .. }
            | Ast::While { token_range, .. }
            | Ast::Break { token_range, .. }
            | Ast::Continue { token_range, .. }
            | Ast::For { token_range, .. }
            | Ast::StructCreation { token_range, .. }
            | Ast::PropertyAccess { token_range, .. }
            | Ast::EnumCreation { token_range, .. }
            | Ast::Try { token_range, .. }
            | Ast::Throw { token_range, .. }
            | Ast::Match { token_range, .. }
            | Ast::ProtocolDispatch { token_range, .. } => *token_range,
            Ast::IntegerLiteral(_, position)
            | Ast::FloatLiteral(_, position)
            | Ast::Identifier(_, position)
            | Ast::String(_, position)
            | Ast::Keyword(_, position)
            | Ast::True(position)
            | Ast::False(position)
            | Ast::Null(position) => TokenRange::new(*position, *position),
        }
    }
    pub fn compile(
        &self,
        compiler: &mut Compiler,
        file_name: &str,
        token_line_column_map: Vec<(usize, usize)>,
    ) -> (Ir, Vec<(TokenRange, IRRange)>) {
        let allocate_fn_pointer = compiler
            .allocate_fn_pointer()
            .expect("Failed to get allocate function pointer - this is a fatal compiler error");
        let mut ast_compiler = AstCompiler {
            ast: self.clone(),
            file_name: file_name.to_string(),
            ir: Ir::new(allocate_fn_pointer),
            ir_range_to_token_range: vec![Vec::new()],
            name: None,
            compiler,
            context: vec![],
            current_context: Context {
                tail_position: true,
                in_function: false,
            },
            next_context: Context {
                tail_position: true,
                in_function: false,
            },
            environment_stack: vec![Environment::new()],
            current_token_info: vec![],
            last_accounted_for_ir: 0,
            metadata: HashMap::new(),
            mutable_pass_env_stack: vec![HashMap::new()],
            loop_exit_stack: vec![],
            current_function_arity: 0,
            current_function_is_variadic: false,
            current_function_min_args: 0,
            token_line_column_map,
        };

        // println!("{:#?}", compiler);
        (
            ast_compiler.compile(),
            ast_compiler.ir_range_to_token_range.pop().unwrap(),
        )
    }

    pub fn has_top_level(&self) -> bool {
        self.nodes().iter().any(|node| {
            !matches!(
                node,
                Ast::Function { .. }
                    | Ast::Struct { .. }
                    // | Ast::Enum { .. }
                    | Ast::Namespace { .. }
            )
        })
    }

    pub fn nodes(&self) -> &Vec<Ast> {
        match self {
            Ast::Program { elements, .. } => elements,
            _ => panic!("Only works on program"),
        }
    }

    pub fn name(&self) -> Option<String> {
        match self {
            Ast::Function { name, .. } => name.clone(),
            _ => panic!("Only works on function"),
        }
    }

    pub fn get_string(&self) -> String {
        match self {
            Ast::String(str, _) => str.replace("\"", ""),
            Ast::Keyword(str, _) => str.to_string(),
            Ast::Identifier(str, _) => str.to_string(),
            _ => panic!("Expected string"),
        }
    }

    pub fn imports(&self) -> Vec<Ast> {
        self.nodes()
            .iter()
            .filter(|node| matches!(node, Ast::Import { .. }))
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone)]
pub enum VariableLocation {
    Register(VirtualRegister),
    Local(usize),
    MutableLocal(usize),
    FreeVariable(usize),
    NamespaceVariable(usize, usize),
    BoxedMutableLocal(usize),
    BoxedFreeVariable(usize),
    MutableFreeVariable(usize),
}
impl VariableLocation {
    fn is_free(&self) -> bool {
        match self {
            VariableLocation::FreeVariable(_)
            | VariableLocation::MutableFreeVariable(_)
            | VariableLocation::BoxedFreeVariable(_) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Context {
    pub tail_position: bool,
    pub in_function: bool,
}

#[derive(Debug, Clone)]
pub struct FreeVariable {
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Environment {
    pub local_variables: Vec<String>,
    pub variables: HashMap<String, VariableLocation>,
    pub free_variables: Vec<FreeVariable>,
    pub argument_locations: HashMap<usize, VariableLocation>,
}

impl Environment {
    fn new() -> Self {
        Environment {
            local_variables: vec![],
            variables: HashMap::new(),
            free_variables: vec![],
            argument_locations: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IRRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct Metadata {
    needs_to_be_boxed: bool,
}

#[derive(Debug, Clone)]
pub struct MutablePassInfo {
    // If it is mutable, this is Some
    mutable_definition: Option<Ast>,
}
impl MutablePassInfo {
    fn default() -> MutablePassInfo {
        MutablePassInfo {
            mutable_definition: None,
        }
    }
}

#[derive(Debug)]
pub struct AstCompiler<'a> {
    pub ast: Ast,
    pub ir: Ir,
    pub name: Option<String>,
    pub compiler: &'a mut Compiler,
    pub file_name: String,
    // This feels dumb and complicated. But my brain
    // won't let me think of a better way
    // I know there is one.
    pub context: Vec<Context>,
    pub current_context: Context,
    pub next_context: Context,
    pub environment_stack: Vec<Environment>,
    pub current_token_info: Vec<(TokenRange, usize)>,
    pub ir_range_to_token_range: Vec<Vec<(TokenRange, IRRange)>>,
    pub last_accounted_for_ir: usize,
    pub metadata: HashMap<Ast, Metadata>,
    pub mutable_pass_env_stack: Vec<HashMap<String, MutablePassInfo>>,
    pub loop_exit_stack: Vec<(Label, VirtualRegister, Label)>,
    pub current_function_arity: usize,
    pub current_function_is_variadic: bool,
    pub current_function_min_args: usize,
    pub token_line_column_map: Vec<(usize, usize)>,
}

impl AstCompiler<'_> {
    pub fn tail_position(&mut self) {
        self.next_context.tail_position = true;
    }

    pub fn not_tail_position(&mut self) {
        self.next_context.tail_position = false;
    }

    pub fn in_function(&mut self) {
        self.next_context.in_function = true;
    }

    pub fn is_tail_position(&self) -> bool {
        self.current_context.tail_position
    }

    pub fn call_compile(&mut self, ast: &Ast) -> Value {
        self.context.push(self.current_context.clone());
        self.current_context = self.next_context.clone();
        self.push_current_token(ast.token_range());
        let result = self.compile_to_ir(ast);
        self.pop_current_token();
        self.next_context = self.current_context.clone();
        self.current_context = self.context.pop().unwrap();

        result
    }

    pub fn compile(&mut self) -> Ir {
        // TODO: Get rid of clone
        self.find_mutable_vars_that_need_boxing(&self.ast.clone());

        // TODO: Get rid of clone
        self.first_pass(&self.ast.clone());

        self.tail_position();
        self.call_compile(&Box::new(self.ast.clone()));

        let allocate_fn_pointer = self
            .compiler
            .allocate_fn_pointer()
            .expect("Failed to get allocate function pointer - this is a fatal compiler error");
        let mut ir = Ir::new(allocate_fn_pointer);
        std::mem::swap(&mut ir, &mut self.ir);
        ir
    }

    pub fn compile_to_ir(&mut self, ast: &Ast) -> Value {
        match ast.clone() {
            Ast::Program { elements, .. } => {
                let mut last = Value::TaggedConstant(0);
                for ast in elements.iter() {
                    self.tail_position();
                    last = self.call_compile(ast);
                }
                self.ir.ret(last)
            }
            Ast::Function {
                name,
                args,
                rest_param,
                body,
                ..
            } => {
                self.create_new_environment();

                let is_not_top_level = self.environment_stack.len() > 2;
                let is_variadic = rest_param.is_some();
                let min_args = args.len();
                // For variadic functions, the actual arity is min_args + 1 (for the rest vector)
                let actual_arity = if is_variadic { min_args + 1 } else { min_args };

                let variable_locaton = if is_not_top_level {
                    // We are inside a function already, so this isn't a top level function
                    // We are going to create a location for it, so that we can make sure recursion will work
                    // for it
                    if let Some(name) = &name {
                        let new_local = self.find_or_insert_local(name);
                        self.insert_variable(name.to_string(), VariableLocation::Local(new_local));
                        Some(VariableLocation::Local(new_local))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let allocate_fn_pointer = self.compiler.allocate_fn_pointer().expect(
                    "Failed to get allocate function pointer - this is a fatal compiler error",
                );
                let old_ir = std::mem::replace(&mut self.ir, Ir::new(allocate_fn_pointer));
                let old_name = self.name.clone();
                let old_arity = self.current_function_arity;
                let old_is_variadic = self.current_function_is_variadic;
                let old_min_args = self.current_function_min_args;
                self.ir_range_to_token_range.push(Vec::new());

                self.name = name.clone();
                self.current_function_arity = actual_arity;
                self.current_function_is_variadic = is_variadic;
                self.current_function_min_args = min_args;
                if is_not_top_level {
                    let context_name = "<closure_context>";
                    let reg = self.ir.arg(0);
                    let local = self.find_or_insert_local(context_name);
                    let reg = self.ir.assign_new(reg);
                    self.ir.store_local(local, reg.into());
                    let local = VariableLocation::Local(local);
                    self.register_arg_location(0, local.clone());
                    self.insert_variable(context_name.to_string(), local);
                }
                for (index, arg) in args.iter().enumerate() {
                    let mut index = index;
                    if is_not_top_level {
                        index += 1;
                    }
                    let reg = self.ir.arg(index);
                    let local = self.find_or_insert_local(&arg.clone());
                    let reg = self.ir.assign_new(reg);
                    self.ir.store_local(local, reg.into());
                    let local = VariableLocation::Local(local);
                    self.register_arg_location(index, local.clone());
                    self.insert_variable(arg.clone(), local);
                }

                // Handle rest parameter - it comes as a PersistentVector
                if let Some(rest_name) = rest_param {
                    let mut rest_index = args.len();
                    if is_not_top_level {
                        rest_index += 1;
                    }
                    let reg = self.ir.arg(rest_index);
                    let local = self.find_or_insert_local(&rest_name);
                    let reg = self.ir.assign_new(reg);
                    self.ir.store_local(local, reg.into());
                    let local = VariableLocation::Local(local);
                    self.register_arg_location(rest_index, local.clone());
                    self.insert_variable(rest_name.clone(), local);
                }

                let should_pause_atom = self.compiler.get_pause_atom();

                // TODO: This isn't working with atomicbool. Need to figure it out
                if should_pause_atom != 0 {
                    let should_pause_atom = self.ir.assign_new(Value::RawValue(should_pause_atom));
                    let atomic_value = self.ir.volatile_register();
                    let should_pause_atom = self
                        .ir
                        .atomic_load(atomic_value.into(), should_pause_atom.into());
                    let skip_pause = self.ir.label("pause");
                    self.ir.jump_if(
                        skip_pause,
                        Condition::Equal,
                        should_pause_atom,
                        Value::RawValue(0),
                    );
                    let stack_pointer = self.ir.get_stack_pointer_imm(0);
                    let pause_function = self
                        .compiler
                        .get_function_by_name("beagle.builtin/__pause")
                        .unwrap();
                    let pause_function = self
                        .compiler
                        .get_function_pointer(pause_function.clone())
                        .unwrap();
                    let pause_function = self.ir.assign_new(pause_function);
                    self.ir
                        .call_builtin(pause_function.into(), vec![stack_pointer]);
                    self.ir.write_label(skip_pause);
                }

                for ast in body[..body.len().saturating_sub(1)].iter() {
                    self.call_compile(&Box::new(ast));
                }
                let last = body.last().unwrap_or(&Ast::Null(0));
                let return_value = self.call_compile(&Box::new(last));
                self.ir.ret(return_value);

                self.name = old_name;
                self.current_function_arity = old_arity;
                self.current_function_is_variadic = old_is_variadic;
                self.current_function_min_args = old_min_args;

                let backend = Backend::new();

                let error_fn_pointer = self
                    .compiler
                    .find_function("beagle.builtin/throw_error")
                    .unwrap();
                let error_fn_pointer = self
                    .compiler
                    .get_function_pointer(error_fn_pointer)
                    .unwrap();

                let token_map = self.ir_range_to_token_range.pop().unwrap();
                self.ir.ir_range_to_token_range = token_map.clone();

                let mut backend = self.ir.compile(backend, error_fn_pointer);
                let token_map = self.ir.ir_range_to_token_range.clone();

                let full_function_name = name
                    .clone()
                    .map(|name| self.compiler.current_namespace_name() + "/" + &name);

                let function_pointer = self
                    .compiler
                    .upsert_function(
                        full_function_name.as_deref(),
                        &mut backend,
                        self.ir.num_locals,
                        actual_arity,
                        is_variadic,
                        min_args,
                    )
                    .unwrap();

                let _ = backend.share_label_info_debug(function_pointer);

                debugger(Message {
                    kind: "ir".to_string(),
                    data: Data::Ir {
                        function_pointer,
                        file_name: self.file_name.clone(),
                        instructions: self
                            .ir
                            .instructions
                            .iter()
                            .map(|x| x.pretty_print())
                            .collect(),
                        token_range_to_ir_range: token_map
                            .iter()
                            .map(|(token_range, ir_range)| {
                                (
                                    (token_range.start, token_range.end),
                                    (ir_range.start, ir_range.end),
                                )
                            })
                            .collect(),
                    },
                });

                let pretty_arm_instructions = backend
                    .instructions_mut()
                    .iter()
                    .map(|x| x.pretty_print())
                    .collect();
                let ir_to_machine_code_range = self
                    .ir
                    .ir_to_machine_code_range
                    .iter()
                    .map(|(ir, machine_range)| (*ir, (machine_range.start, machine_range.end)))
                    .collect();

                debugger(crate::Message {
                    kind: "asm".to_string(),
                    data: Data::Arm {
                        function_pointer,
                        file_name: self.file_name.clone(),
                        instructions: pretty_arm_instructions,
                        ir_to_machine_code_range,
                    },
                });

                self.ir = old_ir;

                if is_not_top_level {
                    let function = self.compile_closure(
                        BuiltInTypes::Function.tag(function_pointer as isize) as usize,
                    );
                    if let Some(VariableLocation::Local(index)) = variable_locaton {
                        let reg = self.ir.assign_new(function);
                        self.ir.store_local(index, reg.into());
                    }
                    self.pop_environment();
                    // Re-insert the variable in the parent environment after popping
                    if let Some(VariableLocation::Local(index)) = variable_locaton
                        && let Some(name) = &name
                    {
                        self.insert_variable(name.to_string(), VariableLocation::Local(index));
                    }
                    return function;
                }

                let function = self.ir.function(Value::Function(function_pointer));
                if let Some(name) = name {
                    let function_reg = self.ir.assign_new(function);
                    let namespace_id = self.current_namespace_id();
                    let reserved_namespace_slot = self.compiler.reserve_namespace_slot(&name);
                    self.store_namespaced_variable(
                        Value::TaggedConstant(reserved_namespace_slot as isize),
                        function_reg,
                    );
                    self.insert_variable(
                        name.to_string(),
                        VariableLocation::NamespaceVariable(namespace_id, reserved_namespace_slot),
                    );
                }

                if let Some(VariableLocation::Local(index)) = variable_locaton {
                    let reg = self.ir.assign_new(function);
                    self.ir.store_local(index, reg.into());
                }
                self.pop_environment();

                function
            }

            Ast::Loop { body, .. } => {
                let loop_start = self.ir.label("loop_start");
                let loop_exit = self.ir.label("loop_exit");
                let result_reg = self.ir.assign_new(Value::Null);

                // Push loop context for break/continue statements
                self.loop_exit_stack
                    .push((loop_exit, result_reg, loop_start));

                self.ir.write_label(loop_start);
                for ast in body.iter() {
                    self.call_compile(ast);
                }
                self.ir.jump(loop_start);

                // Pop loop context
                self.loop_exit_stack.pop();

                // Exit label (only reached via break)
                self.ir.write_label(loop_exit);

                result_reg.into() // Return the break value
            }
            Ast::While {
                condition, body, ..
            } => {
                let loop_start = self.ir.label("while_start");
                let loop_exit = self.ir.label("while_exit");
                let result_reg = self.ir.assign_new(Value::Null);

                // Push loop context for break/continue statements
                self.loop_exit_stack
                    .push((loop_exit, result_reg, loop_start));

                self.ir.write_label(loop_start);

                // Check condition
                let cond_value = self.call_compile(&condition);
                self.ir
                    .jump_if(loop_exit, Condition::NotEqual, cond_value, Value::True);

                // Execute body and track last expression value
                let mut last_value = Value::Null;
                for ast in body.iter() {
                    last_value = self.call_compile(ast);
                }

                // Store last expression value in result_reg
                self.ir.assign(result_reg, last_value);

                // Jump back to check condition
                self.ir.jump(loop_start);

                // Pop loop context
                self.loop_exit_stack.pop();

                // Exit label
                self.ir.write_label(loop_exit);

                result_reg.into() // Return last expression or break value
            }
            Ast::Break { value, .. } => {
                if self.loop_exit_stack.is_empty() {
                    panic!("break statement outside of loop");
                }

                let (exit_label, result_reg, _) = *self.loop_exit_stack.last().unwrap();
                let break_value = self.call_compile(&value);
                self.ir.assign(result_reg, break_value);
                self.ir.jump(exit_label);

                Value::Null // Unreachable after jump
            }
            Ast::Continue { .. } => {
                if self.loop_exit_stack.is_empty() {
                    panic!("continue statement outside of loop");
                }

                let (_, _, loop_start) = *self.loop_exit_stack.last().unwrap();
                self.ir.jump(loop_start);

                Value::Null // Unreachable after jump
            }
            Ast::For {
                binding,
                collection,
                body,
                token_range,
            } => {
                // Generate unique variable names to avoid shadowing
                let seq_var = format!("__for_seq_{}", token_range.start);
                let first_var = format!("__for_first_{}", token_range.start);
                let result_var = format!("__for_result_{}", token_range.start);

                // Use loop instead of while to handle continue correctly
                // Equivalent to:
                //     let mut __seq = seq(coll)
                //     let mut __first = true
                //     let mut __result = null
                //     loop {
                //         if !__first {
                //             __seq = next(__seq)
                //         }
                //         __first = false
                //         if __seq == null { break(__result) }
                //         let binding = first(__seq)
                //         __result = { ...body... }
                //     }

                // First: let mut __seq = seq(coll)
                self.call_compile(&Ast::LetMut {
                    name: Box::new(Ast::Identifier(seq_var.clone(), token_range.start)),
                    value: Box::new(Ast::Call {
                        name: "beagle.core/seq".to_string(),
                        args: vec![*collection.clone()],
                        token_range,
                    }),
                    token_range,
                });

                // Second: let mut __first = true
                self.call_compile(&Ast::LetMut {
                    name: Box::new(Ast::Identifier(first_var.clone(), token_range.start)),
                    value: Box::new(Ast::True(token_range.start)),
                    token_range,
                });

                // Third: let mut __result = null
                self.call_compile(&Ast::LetMut {
                    name: Box::new(Ast::Identifier(result_var.clone(), token_range.start)),
                    value: Box::new(Ast::Null(token_range.start)),
                    token_range,
                });

                // Third: loop { ... }
                let mut loop_body = vec![
                    // if __first == false { __seq = next(__seq) }
                    Ast::If {
                        condition: Box::new(Ast::Condition {
                            operator: Condition::Equal,
                            left: Box::new(Ast::Identifier(first_var.clone(), token_range.start)),
                            right: Box::new(Ast::False(token_range.start)),
                            token_range,
                        }),
                        then: vec![Ast::Assignment {
                            name: Box::new(Ast::Identifier(seq_var.clone(), token_range.start)),
                            value: Box::new(Ast::Call {
                                name: "beagle.core/next".to_string(),
                                args: vec![Ast::Identifier(seq_var.clone(), token_range.start)],
                                token_range,
                            }),
                            token_range,
                        }],
                        else_: vec![],
                        token_range,
                    },
                    // __first = false
                    Ast::Assignment {
                        name: Box::new(Ast::Identifier(first_var.clone(), token_range.start)),
                        value: Box::new(Ast::False(token_range.start)),
                        token_range,
                    },
                    // if __seq == null { break(__result) }
                    Ast::If {
                        condition: Box::new(Ast::Condition {
                            operator: Condition::Equal,
                            left: Box::new(Ast::Identifier(seq_var.clone(), token_range.start)),
                            right: Box::new(Ast::Null(token_range.start)),
                            token_range,
                        }),
                        then: vec![Ast::Break {
                            value: Box::new(Ast::Identifier(result_var.clone(), token_range.start)),
                            token_range,
                        }],
                        else_: vec![],
                        token_range,
                    },
                    // let binding = first(__seq)
                    Ast::Let {
                        name: Box::new(Ast::Identifier(binding.clone(), token_range.start)),
                        value: Box::new(Ast::Call {
                            name: "beagle.core/first".to_string(),
                            args: vec![Ast::Identifier(seq_var.clone(), token_range.start)],
                            token_range,
                        }),
                        token_range,
                    },
                ];

                // Add body statements  - we need to inline them, not wrap in Program
                // We'll compile each body statement, and assign the last one to __result
                for (i, stmt) in body.iter().enumerate() {
                    if i == body.len() - 1 {
                        // Last statement: assign its value to __result
                        loop_body.push(Ast::Assignment {
                            name: Box::new(Ast::Identifier(result_var.clone(), token_range.start)),
                            value: Box::new(stmt.clone()),
                            token_range,
                        });
                    } else {
                        // Not last: just execute it
                        loop_body.push(stmt.clone());
                    }
                }

                self.call_compile(&Ast::Loop {
                    body: loop_body,
                    token_range,
                })
            }
            Ast::Try {
                body,
                exception_binding,
                catch_body,
                ..
            } => {
                // Create labels for catch block and continuation
                let catch_label = self.ir.label("catch_block");
                let after_catch = self.ir.label("after_catch");

                // Create a result register that both try and catch will write to
                let result_reg = self.ir.assign_new(Value::Null);

                // Save any existing binding with this name to restore later
                let saved_binding = self.get_variable(&exception_binding).clone();

                // Allocate local for exception object with a unique internal name
                // Use the label index to ensure uniqueness across multiple try-catch blocks
                let unique_exception_name =
                    format!("__exception_{}_{}__", exception_binding, catch_label.index);
                let exception_local = self.find_or_insert_local(&unique_exception_name);
                // Initialize it to null and register it with IR so num_locals includes it
                let null_reg = self.ir.assign_new(Value::Null);
                self.ir.store_local(exception_local, null_reg.into());

                // Get builtin function pointers
                let push_handler_fn = self
                    .compiler
                    .find_function("beagle.builtin/push_exception_handler")
                    .expect("push_exception_handler builtin not found");
                let push_handler_fn_ptr = usize::from(push_handler_fn.pointer);

                let pop_handler_fn = self
                    .compiler
                    .find_function("beagle.builtin/pop_exception_handler")
                    .expect("pop_exception_handler builtin not found");
                let pop_handler_fn_ptr = usize::from(pop_handler_fn.pointer);

                // Push exception handler
                self.ir.push_exception_handler(
                    catch_label,
                    Value::Local(exception_local),
                    push_handler_fn_ptr,
                );

                // Compile try body - disable tail call optimization
                // TCO would prevent proper stack frame setup, breaking exception handling
                self.not_tail_position();
                let mut try_result = Value::Null;
                for ast in body.iter() {
                    self.not_tail_position();
                    try_result = self.call_compile(ast);
                }

                // Normal path: store result, pop handler and skip catch
                self.ir.assign(result_reg, try_result);
                self.ir.pop_exception_handler(pop_handler_fn_ptr);
                self.ir.jump(after_catch);

                // Catch block (target of throw)
                self.ir.write_label(catch_label);

                // Exception already stored in exception_local by throw
                self.insert_variable(
                    exception_binding.clone(),
                    VariableLocation::Local(exception_local),
                );

                // Compile catch body - also disable tail call optimization
                let mut catch_result = Value::Null;
                for ast in catch_body.iter() {
                    self.not_tail_position();
                    catch_result = self.call_compile(ast);
                }

                // Store catch result in the same result register
                self.ir.assign(result_reg, catch_result);

                self.ir.write_label(after_catch);

                // Restore the saved binding so subsequent code sees the original variable
                if let Some(saved) = saved_binding.clone() {
                    self.insert_variable(exception_binding.clone(), saved);
                } else {
                    // Remove the exception binding if it didn't exist before
                    let current_env = self.environment_stack.last_mut().unwrap();
                    current_env.variables.remove(&exception_binding);
                }

                result_reg.into()
            }
            Ast::Throw { value, .. } => {
                let exception_value = self.call_compile(&value);
                // Ensure the exception value is in a register
                let exception_value = match exception_value {
                    Value::Register(_) => exception_value,
                    _ => {
                        let reg = self.ir.volatile_register();
                        self.ir.assign(reg, exception_value);
                        reg.into()
                    }
                };
                let throw_fn = self
                    .compiler
                    .find_function("beagle.builtin/throw_exception")
                    .expect("throw_exception builtin not found");
                let throw_fn_ptr = usize::from(throw_fn.pointer);
                self.ir.throw_value(exception_value, throw_fn_ptr);
                // Throw never returns, but we need to return something for type checking
                Value::Null
            }
            Ast::Match {
                value,
                arms,
                token_range,
            } => {
                // Compile the value to match on
                let value_reg = self.call_compile(&value);

                // Ensure value is in a register
                let value_reg = match value_reg {
                    Value::Register(_) => value_reg,
                    _ => {
                        let reg = self.ir.volatile_register();
                        self.ir.assign(reg, value_reg);
                        reg.into()
                    }
                };

                // Create labels
                let end_label = self.ir.label("match_end");
                let no_match_label = self.ir.label("match_no_match");
                let result_reg = self.ir.assign_new(Value::Null);

                // Compile each arm
                for (i, arm) in arms.iter().enumerate() {
                    let next_arm_label = if i < arms.len() - 1 {
                        self.ir.label(&format!("match_arm_{}", i + 1))
                    } else {
                        no_match_label
                    };

                    // Test pattern
                    self.compile_pattern_test(&arm.pattern, value_reg, next_arm_label);

                    // Pattern matched - save current environment before binding pattern variables
                    // We need to save the actual VariableLocation values, not just keys,
                    // to properly restore shadowed variables
                    let saved_env: HashMap<String, VariableLocation> =
                        self.environment_stack.last().unwrap().variables.clone();

                    self.bind_pattern_variables(&arm.pattern, value_reg);

                    // Compile arm body
                    let mut arm_result = Value::Null;
                    for ast in arm.body.iter() {
                        arm_result = self.call_compile(ast);
                    }
                    self.ir.assign(result_reg, arm_result);

                    // Restore environment - put back the exact state from before the match arm
                    let current_env = self.environment_stack.last_mut().unwrap();
                    current_env.variables = saved_env;

                    self.ir.jump(end_label);

                    // Write next arm label
                    if i < arms.len() - 1 {
                        self.ir.write_label(next_arm_label);
                    }
                }

                // Check exhaustiveness and generate warnings
                self.check_match_exhaustiveness(&arms, token_range);

                // No pattern matched - throw error
                self.ir.write_label(no_match_label);
                self.call_builtin("beagle.builtin/throw_error", vec![]);

                self.ir.write_label(end_label);
                result_reg.into()
            }
            Ast::Struct {
                name, fields: _, ..
            } => {
                // TODO: I need a store the fields, but I'm too lazy to make the
                // array right now
                let fully_qualified_name =
                    format!("{}/{}", self.compiler.current_namespace_name(), name);
                let (struct_id, _) = self
                    .compiler
                    .get_struct(&fully_qualified_name)
                    .unwrap_or_else(|| panic!("Struct not found {}", name));

                let value = self.call_compile(&Ast::StructCreation {
                    name: "beagle.core/Struct".to_string(),
                    fields: vec![
                        ("name".to_string(), Ast::String(fully_qualified_name, 0)),
                        ("id".to_string(), Ast::IntegerLiteral(struct_id as i64, 0)),
                    ],
                    token_range: ast.token_range(),
                });
                let namespace_id = self
                    .compiler
                    .find_binding(self.current_namespace_id(), &name)
                    .unwrap_or_else(|| panic!("binding not found {}", name));
                let value_reg = self.ir.assign_new(value);
                self.store_namespaced_variable(
                    Value::TaggedConstant(namespace_id as isize),
                    value_reg,
                );
                value
            }
            Ast::Enum {
                name,
                variants,
                token_range,
            } => {
                let mut struct_fields: Vec<(String, Ast)> = vec![];
                for variant in variants.iter() {
                    match variant {
                        Ast::EnumVariant {
                            name, fields: _, ..
                        } => {
                            // TODO: These should be functions??
                            // Maybe I should have a concept of a struct/data creator
                            // that gets called with named arguments like that?
                            // I'm not sure
                            // I think my whole setup here is janky. But I want things working first
                            struct_fields.push((name.clone(), Ast::Null(0)));
                        }
                        Ast::EnumStaticVariant {
                            name: variant_name,
                            token_range,
                        } => {
                            struct_fields.push((
                                variant_name.clone(),
                                Ast::StructCreation {
                                    name: format!("{}.{}", name, variant_name),
                                    fields: vec![],
                                    token_range: *token_range,
                                },
                            ));
                        }
                        _ => panic!("Shouldn't get here"),
                    }
                }
                let value = self.call_compile(&Ast::StructCreation {
                    name: name.clone(),
                    fields: struct_fields,
                    token_range,
                });
                let namespace_id = self
                    .compiler
                    .find_binding(self.current_namespace_id(), &name)
                    .unwrap_or_else(|| panic!("binding not found {}", name));
                let value_reg = self.ir.assign_new(value);
                self.store_namespaced_variable(
                    Value::TaggedConstant(namespace_id as isize),
                    value_reg,
                );
                // TODO: This should probably return the enum value
                // A concept I don't yet have
                Value::Null
            }
            Ast::EnumVariant {
                name: _, fields: _, ..
            } => {
                panic!("Shouldn't get here")
            }
            Ast::EnumStaticVariant { name: _, .. } => {
                panic!("Shouldn't get here")
            }
            Ast::Protocol {
                name,
                body,
                token_range: _,
            } => {
                for ast in body.iter() {
                    if matches!(ast, Ast::Function { .. }) {
                        self.call_compile(ast);
                        // TODO: This is not great, but I am just trying to get things working
                        let Ast::Function {
                            name: function_name,
                            ..
                        } = ast
                        else {
                            panic!("Expected function")
                        };
                        let function = self
                            .compiler
                            .get_function_by_name(&format!(
                                "{}/{}",
                                self.compiler.current_namespace_name(),
                                function_name.clone().unwrap()
                            ))
                            .unwrap();
                        let fully_qualified_name = format!(
                            "{}/{}_{}",
                            self.compiler.current_namespace_name(),
                            name,
                            function_name.clone().unwrap()
                        );

                        // Ignore error - if alias can't be added, the protocol method won't work but won't crash
                        let _ = self
                            .compiler
                            .add_function_alias(&fully_qualified_name, function);
                    } else {
                        self.call_compile(ast);
                    }
                }
                let fully_qualified_name =
                    format!("{}/{}", self.compiler.current_namespace_name(), name);
                let value = self.call_compile(&Ast::StructCreation {
                    name: "beagle.core/Protocol".to_string(),
                    fields: vec![("name".to_string(), Ast::String(fully_qualified_name, 0))],
                    token_range: ast.token_range(),
                });
                let reserved_namespace_slot = self.compiler.reserve_namespace_slot(&name);
                let value_reg = self.ir.assign_new(value);
                self.store_namespaced_variable(
                    Value::TaggedConstant(reserved_namespace_slot as isize),
                    value_reg,
                );
                value
            }
            Ast::FunctionStub {
                name,
                args,
                rest_param,
                token_range,
            } => {
                // TODO: I should store funcitons in slots instead of a big global list
                // let namespace_slot = self.compiler.find_binding(self.compiler.current_namespace_id(), &name).unwrap();
                // let namespace_slot = Value::TaggedConstant(namespace_slot as isize);

                self.call_compile(&Ast::Function {
                    name: Some(name),
                    args,
                    rest_param,
                    body: vec![Ast::Call {
                        name: "beagle.builtin/throw_error".to_string(),
                        args: vec![],
                        token_range,
                    }],
                    token_range,
                });
                Value::Null
            }
            Ast::Extend {
                target_type,
                protocol,
                body,
                token_range: _,
            } => {
                for ast in body.iter() {
                    if let Ast::Function {
                        name,
                        args,
                        rest_param,
                        body,
                        ..
                    } = ast
                    {
                        let name = name.clone().unwrap();
                        // TODO: Hygiene
                        let new_name = format!("{}_{}", target_type, name);
                        self.call_compile(&Ast::Function {
                            name: Some(new_name.clone()),
                            args: args.clone(),
                            rest_param: rest_param.clone(),
                            body: body.clone(),
                            token_range: ast.token_range(),
                        });
                        let fully_qualified_name =
                            format!("{}/{}", self.compiler.current_namespace_name(), new_name);
                        let function = self
                            .compiler
                            .get_function_by_name(&fully_qualified_name)
                            .unwrap();
                        let function_pointer = self
                            .compiler
                            .get_function_pointer(function.clone())
                            .unwrap();
                        let function_pointer = Value::RawValue(
                            BuiltInTypes::Function.tag(function_pointer as isize) as usize,
                        );
                        // builtin/register_extension(PersistentVector.name, Indexed.name, "get", get)
                        // TODO: I need to fully resolve the target_type and protocol if they are aliased
                        let target_type = self.string_constant(target_type.clone());
                        let target_type = self.ir.assign_new(target_type);
                        let protocol = self.string_constant(protocol.clone());
                        let protocol = self.ir.assign_new(protocol);
                        let name = self.string_constant(name.clone());
                        let name = self.ir.assign_new(name);
                        let function_pointer = self.ir.assign_new(function_pointer);
                        // self.ir.breakpoint();
                        self.call_builtin(
                            "beagle.builtin/register_extension",
                            vec![
                                target_type.into(),
                                protocol.into(),
                                name.into(),
                                function_pointer.into(),
                            ],
                        );
                    } else {
                        panic!("Expected function");
                    }
                }

                Value::Null
            }
            Ast::EnumCreation {
                name,
                variant,
                fields,
                ..
            } => {
                let (namespace, name) = self.get_namespace_name_and_name(&name);

                let full_struct_name = format!("{}/{}.{}", namespace, name, variant);
                let (struct_id, struct_type) = self
                    .compiler
                    .get_struct(&full_struct_name)
                    .unwrap_or_else(|| panic!("Struct not found {}", name));

                // Clone the field names to avoid borrow checker issues
                let defined_fields = struct_type.fields.clone();
                let size = struct_type.size();

                // Compile field values
                let field_results: Vec<_> = fields
                    .iter()
                    .map(|field| {
                        self.not_tail_position();
                        self.call_compile(&field.1)
                    })
                    .collect();

                // Build a mapping from field name to its compiled value
                let field_map: std::collections::HashMap<String, Value> = fields
                    .iter()
                    .zip(field_results.iter())
                    .map(|(field, result)| (field.0.clone(), *result))
                    .collect();

                // Validate that all user-provided fields exist in the struct definition
                for (field_name, _) in fields.iter() {
                    if !defined_fields.contains(field_name) {
                        panic!(
                            "Struct field not defined: '{}' in {}.{}",
                            field_name, name, variant
                        );
                    }
                }

                // Reorder fields to match the struct definition
                let mut ordered_field_results = Vec::new();
                for defined_field in defined_fields.iter() {
                    match field_map.get(defined_field) {
                        Some(&result) => ordered_field_results.push(result),
                        None => panic!(
                            "Field '{}' not provided in enum variant creation",
                            defined_field
                        ),
                    }
                }

                let allocate = self
                    .compiler
                    .find_function("beagle.builtin/allocate")
                    .unwrap();
                let allocate = self.compiler.get_function_pointer(allocate).unwrap();
                let allocate = self.ir.assign_new(allocate);

                let size_reg = self.ir.assign_new(size);
                let stack_pointer = self.ir.get_stack_pointer_imm(0);

                let struct_ptr = self
                    .ir
                    .call_builtin(allocate.into(), vec![stack_pointer, size_reg.into()]);

                let struct_pointer = self.ir.untag(struct_ptr);
                self.ir.write_struct_id(struct_pointer, struct_id);
                self.ir.write_fields(struct_pointer, &ordered_field_results);

                self.ir
                    .tag(struct_pointer, BuiltInTypes::HeapObject.get_tag())
            }
            Ast::StructCreation { name, fields, .. } => {
                let (namespace, name) = self.get_namespace_name_and_name(&name);
                for field in fields.iter() {
                    self.not_tail_position();
                    let value = self.call_compile(&field.1);
                    let reg = self.ir.assign_new(value);
                    self.ir.push_to_stack(reg.into());
                }

                let (struct_id, struct_type) = match self
                    .compiler
                    .get_struct(&format!("{}/{}", namespace, name))
                    .or_else(|| self.compiler.get_struct(&format!("beagle.core/{}", name)))
                {
                    Some(s) => s,
                    None => {
                        // Generate code to throw a runtime error that can be caught with try/catch

                        // Create error kind string
                        let kind_constant = self.string_constant("StructError".to_string());
                        let kind_str = self.ir.load_string_constant(kind_constant);

                        // Create error message string
                        let message =
                            format!("Struct '{}' not found in namespace '{}'", name, namespace);
                        let message_constant = self.string_constant(message);
                        let message_str = self.ir.load_string_constant(message_constant);

                        // Create null for location and assign to register
                        let null_reg = self.ir.assign_new(Value::Null);

                        // Call create_error builtin (stack_pointer is added automatically by call_builtin)
                        let error = self.call_builtin(
                            "beagle.builtin/create_error",
                            vec![kind_str, message_str, null_reg.into()],
                        );

                        // Throw the exception (this doesn't return)
                        self.call_builtin("beagle.builtin/throw_exception", vec![error]);

                        // Return null (unreachable but needed for type checking)
                        return Value::Null;
                    }
                };

                let mut field_order: Vec<usize> = vec![];
                for field in fields.iter() {
                    let mut found = false;
                    for (i, defined_field) in struct_type.fields.iter().enumerate() {
                        if &field.0 == defined_field {
                            field_order.push(i);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        panic!("Struct field not defined {}", field.0);
                    }
                }

                let allocate = self
                    .compiler
                    .find_function("beagle.builtin/allocate")
                    .unwrap();
                let allocate = self.compiler.get_function_pointer(allocate).unwrap();
                let allocate = self.ir.assign_new(allocate);

                let size_reg = self.ir.assign_new(struct_type.size());
                let stack_pointer = self.ir.get_stack_pointer_imm(0);

                let struct_ptr = self
                    .ir
                    .call_builtin(allocate.into(), vec![stack_pointer, size_reg.into()]);

                let struct_pointer = self.ir.untag(struct_ptr);
                self.ir.write_struct_id(struct_pointer, struct_id);

                for field in field_order.iter().rev() {
                    let reg = self.ir.pop_from_stack();
                    self.ir.write_field(struct_pointer, *field, reg);
                }

                self.ir
                    .tag(struct_pointer, BuiltInTypes::HeapObject.get_tag())
            }
            Ast::Array {
                array: elements, ..
            } => {
                // Let's stary by just adding a popping for simplicity
                for element in elements.iter() {
                    self.not_tail_position();
                    let value = self.call_compile(element);
                    let reg = self.ir.assign_new(value);
                    self.ir.push_to_stack(reg.into());
                }

                let vector_pointer = self.call("persistent_vector/vec", vec![]);

                let push = self.get_function("persistent_vector/push");
                let vector_register = self.ir.assign_new(vector_pointer);
                // the elements are on the stack in reverse, so I need to grab them by index in reverse
                // and then shift the stack pointer
                let stack_pointer = self.ir.get_current_stack_position();
                for i in (0..elements.len()).rev() {
                    let value = self.ir.load_from_memory(stack_pointer, (i as i32) + 1);
                    let push_result = self.ir.call(push, vec![vector_register.into(), value]);
                    self.ir.assign(vector_register, push_result);
                }
                for _ in 0..elements.len() {
                    // TODO: Hacky since we aren't using this. I think it is an efficiency waste
                    // I should probably do something better
                    self.ir.pop_from_stack();
                }

                vector_register.into()
            }
            Ast::MapLiteral { pairs, .. } => {
                // Special case: empty map
                if pairs.is_empty() {
                    return self.call("persistent_map/map", vec![]);
                }

                // Check for duplicate literal keys at compile time
                let mut seen_keys: HashMap<String, &Ast> = HashMap::new();
                for (key, _value) in pairs.iter() {
                    // Extract a string representation for literal keys
                    let key_str = match key {
                        Ast::String(s, _) => Some(format!("string:{}", s)),
                        Ast::IntegerLiteral(n, _) => Some(format!("int:{}", n)),
                        Ast::FloatLiteral(f, _) => Some(format!("float:{}", f)),
                        Ast::Keyword(k, _) => Some(format!("keyword:{}", k)),
                        Ast::True(_) => Some("bool:true".to_string()),
                        Ast::False(_) => Some("bool:false".to_string()),
                        Ast::Null(_) => Some("null".to_string()),
                        // For non-literal keys, we can't check at compile time
                        _ => None,
                    };

                    if let Some(key_repr) = key_str {
                        if let Some(first_key) = seen_keys.get(&key_repr) {
                            panic!(
                                "Duplicate key in map literal: {:?}. First occurrence: {:?}, Second occurrence: {:?}",
                                key_repr, first_key, key
                            );
                        }
                        seen_keys.insert(key_repr, key);
                    }
                }

                // Push all keys and values to stack
                for (key, value) in pairs.iter() {
                    self.not_tail_position();
                    let key_val = self.call_compile(key);
                    let key_reg = self.ir.assign_new(key_val);
                    self.ir.push_to_stack(key_reg.into());

                    self.not_tail_position();
                    let val_val = self.call_compile(value);
                    let val_reg = self.ir.assign_new(val_val);
                    self.ir.push_to_stack(val_reg.into());
                }

                // Create empty map
                let map_pointer = self.call("persistent_map/map", vec![]);
                let map_register = self.ir.assign_new(map_pointer);

                // Get assoc function
                let assoc = self.get_function("persistent_map/assoc");

                // Load pairs from stack and assoc them
                let stack_pointer = self.ir.get_current_stack_position();
                for i in 0..pairs.len() {
                    // Stack layout: [key0, val0, key1, val1, key2, val2, ...]
                    // Pair i (0-indexed):
                    // - key is at stack offset (2*i + 1)
                    // - value is at stack offset (2*i + 2)
                    let key_offset = (2 * i + 1) as i32;
                    let val_offset = (2 * i + 2) as i32;

                    // Note: val_offset is actually where the key is stored, key_offset has the value
                    // This is because of how the stack grows
                    let key = self.ir.load_from_memory(stack_pointer, val_offset);
                    let value = self.ir.load_from_memory(stack_pointer, key_offset);

                    let assoc_result = self.ir.call(assoc, vec![map_register.into(), key, value]);
                    self.ir.assign(map_register, assoc_result);
                }

                // Clean up stack
                for _ in 0..(pairs.len() * 2) {
                    self.ir.pop_from_stack();
                }

                map_register.into()
            }
            Ast::Namespace { name, .. } => {
                let namespace_id = self.compiler.reserve_namespace(name);
                let namespace_id = Value::RawValue(namespace_id);
                let namespace_id = self.ir.assign_new(namespace_id);
                self.call_builtin(
                    "beagle.builtin/set_current_namespace",
                    vec![namespace_id.into()],
                )
            }
            Ast::Import {
                library_name,
                alias,
                ..
            } => {
                self.compiler
                    .add_alias(library_name, (*alias).get_string().to_string());
                Value::Null
            }
            Ast::PropertyAccess {
                object, property, ..
            } => {
                let object = self.call_compile(object.as_ref());
                let object = self.ir.assign_new(object);
                let untagged_object = self.ir.untag(object.into());
                // self.ir.breakpoint();
                let struct_id = self.ir.read_struct_id(untagged_object);
                let property_location =
                    Value::RawValue(self.compiler.add_property_lookup().unwrap());
                let property_location = self.ir.assign_new(property_location);
                let property_value = self.ir.load_from_heap(property_location.into(), 0);
                let result = self.ir.assign_new(0);

                let exit_property_access = self.ir.label("exit_property_access");
                let slow_property_path = self.ir.label("slow_property_path");
                self.ir.jump_if(
                    slow_property_path,
                    Condition::NotEqual,
                    struct_id,
                    property_value,
                );

                let property_offset = self.ir.load_from_heap(property_location.into(), 1);
                let property_result = self.ir.read_field(untagged_object, property_offset);

                self.ir.assign(result, property_result);
                self.ir.jump(exit_property_access);

                self.ir.write_label(slow_property_path);
                let property = if let Ast::Identifier(name, _) = property.as_ref() {
                    name.clone()
                } else {
                    panic!("Expected identifier got {:?}", property)
                };

                let constant_ptr = self.string_constant(property.clone());
                let constant_ptr = self.ir.assign_new(constant_ptr);
                let call_result = self.call_builtin(
                    "beagle.builtin/property_access",
                    vec![object.into(), constant_ptr.into(), property_location.into()],
                );

                self.ir.assign(result, call_result);

                self.ir.write_label(exit_property_access);

                result.into()
            }
            Ast::IndexOperator { array, index, .. } => {
                let array = self.call_compile(array.as_ref());
                let index = self.call_compile(index.as_ref());
                let array = self.ir.assign_new(array);
                let index = self.ir.assign_new(index);
                self.call("beagle.core/get", vec![array.into(), index.into()])
            }
            Ast::If {
                condition,
                then,
                else_,
                ..
            } => {
                let condition = self.call_compile(&condition);

                let end_if_label = self.ir.label("end_if");

                let result_reg = self.ir.assign_new(Value::TaggedConstant(0));

                let then_label = self.ir.label("then");
                self.ir
                    .jump_if(then_label, Condition::Equal, condition, Value::True);

                let mut else_result = Value::TaggedConstant(0);
                for ast in else_.iter() {
                    else_result = self.call_compile(&Box::new(ast));
                }
                self.ir.assign(result_reg, else_result);
                self.ir.jump(end_if_label);

                self.ir.write_label(then_label);

                let mut then_result = Value::TaggedConstant(0);
                for ast in then.iter() {
                    then_result = self.call_compile(&Box::new(ast));
                }
                self.ir.assign(result_reg, then_result);

                self.ir.write_label(end_if_label);

                result_reg.into()
            }
            Ast::And { left, right, .. } => {
                let result_reg = self.ir.volatile_register();
                self.ir.assign(result_reg, Value::False);
                let short_circuit = self.ir.label("short_circuit_and");
                let left = self.call_compile(&left);
                self.ir
                    .jump_if(short_circuit, Condition::Equal, left, Value::False);
                let right = self.call_compile(&right);
                self.ir.assign(result_reg, right);
                self.ir.write_label(short_circuit);
                result_reg.into()
            }
            Ast::Or { left, right, .. } => {
                let result_reg = self.ir.volatile_register();
                self.ir.assign(result_reg, Value::True);
                let short_circuit = self.ir.label("short_circuit_or");
                let left = self.call_compile(&left);
                self.ir
                    .jump_if(short_circuit, Condition::Equal, left, Value::True);
                let right = self.call_compile(&right);
                self.ir.assign(result_reg, right);
                self.ir.write_label(short_circuit);
                result_reg.into()
            }
            Ast::Add { left, right, .. } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.add_any(left, right)
            }
            Ast::Sub { left, right, .. } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.sub_any(left, right)
            }
            Ast::Mul { left, right, .. } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.mul_any(left, right)
            }
            Ast::Div { left, right, .. } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.div_any(left, right)
            }
            Ast::ShiftLeft { left, right, .. } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.shift_left(left, right)
            }
            Ast::ShiftRight { left, right, .. } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.shift_right(left, right)
            }
            Ast::ShiftRightZero { left, right, .. } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.shift_right_zero(left, right)
            }
            Ast::BitWiseAnd { left, right, .. } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.bitwise_and(left, right)
            }
            Ast::BitWiseOr { left, right, .. } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.bitwise_or(left, right)
            }
            Ast::BitWiseXor { left, right, .. } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.bitwise_xor(left, right)
            }
            Ast::Recurse { args, .. } | Ast::TailRecurse { args, .. } => {
                let mut compiled_args: Vec<Value> = args
                    .iter()
                    .map(|arg| {
                        self.not_tail_position();
                        self.call_compile(&Box::new(arg.clone()))
                    })
                    .collect();

                // For variadic recursive calls, pack extra arguments
                if self.current_function_is_variadic {
                    compiled_args =
                        self.pack_variadic_args(compiled_args, self.current_function_min_args);
                }

                if matches!(ast, Ast::TailRecurse { .. }) {
                    self.ir.tail_recurse(compiled_args)
                } else {
                    self.ir.recurse(compiled_args)
                }
            }
            Ast::Call {
                name,
                args,
                token_range,
            } => {
                let name = self.get_qualified_function_name(&name);

                if self.name_matches(&name) {
                    // Check arity for recursive calls
                    if self.current_function_is_variadic {
                        // For variadic functions, we need at least min_args arguments
                        if args.len() < self.current_function_min_args {
                            panic!(
                                "Recursive call to variadic '{}' requires at least {} argument(s), but {} were provided",
                                self.name.as_ref().unwrap_or(&"<unknown>".to_string()),
                                self.current_function_min_args,
                                args.len()
                            );
                        }
                    } else if args.len() != self.current_function_arity {
                        panic!(
                            "Recursive call to '{}' expects {} argument(s), but {} were provided",
                            self.name.as_ref().unwrap_or(&"<unknown>".to_string()),
                            self.current_function_arity,
                            args.len()
                        );
                    }
                    if self.is_tail_position() {
                        return self.call_compile(&Ast::TailRecurse { args, token_range });
                    } else {
                        return self.call_compile(&Ast::Recurse { args, token_range });
                    }
                }

                if self.should_not_evaluate_arguments(&name) {
                    return self.compile_macro_like_primitive(name, args);
                }

                let args: Vec<Value> = args
                    .iter()
                    .map(|arg| {
                        self.not_tail_position();
                        let value = self.call_compile(&Box::new(arg.clone()));
                        match value {
                            Value::Register(_) => value,
                            _ => {
                                let reg = self.ir.volatile_register();
                                self.ir.assign(reg, value);
                                reg.into()
                            }
                        }
                    })
                    .collect();

                // TODO: Should the arguments be evaluated first?
                // I think so, this will matter once I think about macros
                // though
                if self.compiler.is_inline_primitive_function(&name) {
                    let expected = crate::primitives::get_inline_primitive_arity(&name);
                    if args.len() != expected {
                        panic!(
                            "Function '{}' expects {} argument(s), but {} were provided",
                            name,
                            expected,
                            args.len()
                        );
                    }
                    return self.compile_inline_primitive_function(&name, args);
                }

                // TODO: This isn't they way to handle this
                // I am acting as if all closures are assign to a variable when they aren't.
                // Need to have negative test cases for this
                if !self.is_qualifed_function_name(&name) {
                    if let Some(function) = self.get_variable(&name) {
                        self.compile_closure_call(function, args)
                    } else {
                        panic!("Not qualified and didn't find it {}", name);
                    }
                } else {
                    self.call(&name, args)
                }
            }
            Ast::IntegerLiteral(n, _) => Value::TaggedConstant(n as isize),
            Ast::FloatLiteral(n, _) => {
                // floats are heap allocated
                // Sadly I have to do this to avoid loss of percision
                let allocate = self
                    .compiler
                    .find_function("beagle.builtin/allocate_float")
                    .unwrap();
                let allocate = self.compiler.get_function_pointer(allocate).unwrap();
                let allocate = self.ir.assign_new(allocate);

                let size_reg = self.ir.assign_new(1);
                let stack_pointer = self.ir.get_stack_pointer_imm(0);

                let float_pointer = self
                    .ir
                    .call_builtin(allocate.into(), vec![stack_pointer, size_reg.into()]);

                let float_pointer = self.ir.untag(float_pointer);
                self.ir.write_small_object_header(float_pointer);
                self.ir.write_float_literal(float_pointer, n);

                self.ir.tag(float_pointer, BuiltInTypes::Float.get_tag())
            }
            Ast::Identifier(name, _) => {
                let reg = &self.get_variable_alloc_free_variable(&name);
                self.resolve_variable(reg)
                    .unwrap_or_else(|_| panic!("Could not resolve variable {}", name))
            }
            Ast::Let { name, value, .. } | Ast::LetMut { name, value, .. } => {
                let needs_boxing = self
                    .metadata
                    .get(ast)
                    .map(|m| m.needs_to_be_boxed)
                    .unwrap_or(false);
                if let Ast::Identifier(name, _) = name.as_ref() {
                    if self.environment_stack.len() == 1 {
                        if matches!(ast, Ast::LetMut { .. }) {
                            panic!("Can't create mutable variable in global scope");
                        }

                        self.not_tail_position();
                        let value = self.call_compile(&value);
                        self.not_tail_position();
                        let reg = self.ir.assign_new(value);
                        let namespace_id = self.compiler.current_namespace_id();
                        let reserved_namespace_slot = self.compiler.reserve_namespace_slot(name);
                        self.store_namespaced_variable(
                            Value::TaggedConstant(reserved_namespace_slot as isize),
                            reg,
                        );
                        self.insert_variable(
                            name.to_string(),
                            VariableLocation::NamespaceVariable(
                                namespace_id,
                                reserved_namespace_slot,
                            ),
                        );
                        reg.into()
                    } else {
                        let reg = self.ir.volatile_register();
                        if needs_boxing {
                            let boxed = self.call_compile(&Ast::StructCreation {
                                name: "beagle.core/__Box__".to_string(),
                                fields: vec![("value".to_string(), *value)],
                                token_range: ast.token_range(),
                            });
                            self.ir.assign(reg, boxed);
                        } else {
                            self.not_tail_position();
                            let value = self.call_compile(&value);
                            self.not_tail_position();
                            self.ir.assign(reg, value);
                        }
                        let local_index = self.find_or_insert_local(name);
                        self.ir.store_local(local_index, reg.into());

                        if matches!(ast, Ast::Let { .. }) {
                            self.insert_variable(
                                name.to_string(),
                                VariableLocation::Local(local_index),
                            );
                        } else if matches!(ast, Ast::LetMut { .. }) && !needs_boxing {
                            self.insert_variable(
                                name.to_string(),
                                VariableLocation::MutableLocal(local_index),
                            );
                        } else if matches!(ast, Ast::LetMut { .. }) && needs_boxing {
                            self.insert_variable(
                                name.to_string(),
                                VariableLocation::BoxedMutableLocal(local_index),
                            );
                        } else {
                            panic!("Expected let or mutlet")
                        }
                        reg.into()
                    }
                } else {
                    panic!("Expected variable")
                }
            }
            Ast::Assignment { name, value, .. } => {
                // TODO: if not marked as mut error
                // I will need to make it so that this gets heap allocated
                // if we access from a closure
                let name = if let Ast::Identifier(name, _) = name.as_ref() {
                    name
                } else {
                    panic!("Expected identifier")
                };
                let value = self.call_compile(&value);
                let value = self.ir.assign_new(value);
                let variable = self.get_variable_alloc_free_variable(name);
                match variable {
                    // TODO: Do I have mutable namespace variables?
                    VariableLocation::NamespaceVariable(_namespace_id, _slott) => {
                        panic!("Can't assign to a namespace variable {}", name);
                    }
                    VariableLocation::Local(_local_index) => {
                        panic!("You can only assign to mutable variables");
                    }
                    VariableLocation::MutableLocal(local_index) => {
                        self.ir.store_local(local_index, value.into());
                    }
                    VariableLocation::BoxedMutableLocal(local_index) => {
                        let local = self.ir.load_local(local_index);
                        // I thought I needed a write barrier, but I believe that isn't the case
                        // because these are only heap allocated if captured.
                        // self.call_builtin("beagle.builtin/gc_add_root", vec![local]);
                        let local = self.ir.untag(local);
                        self.ir.write_field(local, 0, value.into());
                    }
                    VariableLocation::FreeVariable(_free_variable) => {
                        panic!("Can't assign to a non-mutable free variable {}", name);
                    }
                    VariableLocation::BoxedFreeVariable(index) => {
                        let arg0_location = self
                            .get_argument_location(0)
                            .ok_or("Variable not found")
                            .unwrap();
                        let arg0 = self.resolve_variable(&arg0_location).unwrap();
                        let arg0: VirtualRegister = self.ir.assign_new(arg0);
                        let arg0 = self.ir.untag(arg0.into());
                        let index = self
                            .ir
                            .assign_new(Value::TaggedConstant((index + 3) as isize));
                        let slot = self.ir.read_field(arg0, index.into());
                        // I thought I needed a write barrier, but I believe that isn't the case
                        // because these are only heap allocated if captured.
                        // self.call_builtin("beagle.builtin/gc_add_root", vec![slot]);
                        let slot = self.ir.untag(slot);
                        self.ir.write_field(slot, 0, value.into());
                    }
                    VariableLocation::MutableFreeVariable(index) => {
                        let arg0_location = self
                            .get_argument_location(0)
                            .ok_or("Variable not found")
                            .unwrap();
                        let arg0 = self.resolve_variable(&arg0_location).unwrap();
                        let arg0: VirtualRegister = self.ir.assign_new(arg0);
                        // I thought I needed a write barrier, but I believe that isn't the case
                        // because these are only heap allocated if captured.
                        // self.call_builtin("beagle.builtin/gc_add_root", vec![arg0.into()]);
                        let arg0 = self.ir.untag(arg0.into());
                        self.ir.write_field(arg0, index + 3, value.into());
                    }
                    VariableLocation::Register(_virtual_register) => {
                        panic!("Can't assign to a register {}", name);
                    }
                }
                Value::Null
            }
            Ast::Condition {
                operator,
                left,
                right,
                ..
            } => {
                self.not_tail_position();
                let a = self.call_compile(&left);
                self.not_tail_position();
                let b = self.call_compile(&right);
                self.ir.compare_any(a, b, operator)
            }
            Ast::String(str, _) => {
                let constant_ptr = self.string_constant(str);
                self.ir.load_string_constant(constant_ptr)
            }
            Ast::Keyword(keyword_text, _) => {
                let constant_ptr = self.keyword_constant(keyword_text);
                let constant_ptr = self.ir.assign_new(constant_ptr);
                self.call_builtin(
                    "beagle.builtin/load_keyword_constant_runtime",
                    vec![constant_ptr.into()],
                )
            }
            Ast::ProtocolDispatch {
                args,
                cache_location,
                dispatch_table_ptr,
                default_fn_ptr: _,
                num_args: _,
                ..
            } => {
                // Protocol dispatch with inline caching
                // cache: [type_id (8 bytes), fn_ptr (8 bytes)]

                // 1. Load arg0 (the dispatch target) into a register
                let arg0_loc = self.get_variable_alloc_free_variable(&args[0]);
                let arg0_val = self
                    .resolve_variable(&arg0_loc)
                    .expect("Could not resolve first argument");
                let arg0_reg = self.ir.assign_new(arg0_val);

                // 2. Get the tag to determine if heap object or primitive
                let tag = self.ir.get_tag(arg0_reg.into());

                // 3. Set up labels
                let slow_path_label = self.ir.label("protocol_slow_path");
                let exit_label = self.ir.label("protocol_exit");
                let compute_primitive_type_id = self.ir.label("compute_primitive_type_id");
                let type_id_computed = self.ir.label("type_id_computed");

                let result_reg = self.ir.assign_new(Value::Null);
                let type_id_reg = self.ir.volatile_register();

                // Assign cache and dispatch_table pointers BEFORE any jumps
                let cache_ptr = Value::Pointer(cache_location);
                let cache_ptr_reg = self.ir.assign_new(cache_ptr);
                let dispatch_table_ptr_val = Value::Pointer(dispatch_table_ptr);
                let dispatch_table_ptr_reg = self.ir.assign_new(dispatch_table_ptr_val);

                // Get all args for the call (needed in both paths)
                let mut call_args = Vec::new();
                for arg_name in args.iter() {
                    let arg_loc = self.get_variable_alloc_free_variable(arg_name);
                    let arg_val = self
                        .resolve_variable(&arg_loc)
                        .expect("Could not resolve argument");
                    call_args.push(arg_val);
                }

                // 4. Check if HeapObject (tag == 0b110)
                let heap_object_tag = Value::RawValue(BuiltInTypes::HeapObject.get_tag() as usize);
                self.ir.jump_if(
                    compute_primitive_type_id,
                    Condition::NotEqual,
                    tag,
                    heap_object_tag,
                );

                // 5. HeapObject path: read struct_id
                let untagged = self.ir.untag(arg0_reg.into());
                let struct_id = self.ir.read_struct_id(untagged);
                self.ir.assign(type_id_reg, struct_id);
                self.ir.jump(type_id_computed);

                // 6. Primitive path: compute type_id to match slow path
                // Slow path uses: tag==2 -> index 2, else -> tag+16
                // So: Int(0)->16, Float(1)->17, String(2)->2, Bool(3)->19
                self.ir.write_label(compute_primitive_type_id);
                let high_bit = Value::Pointer(0x8000_0000_0000_0000);
                let high_bit_reg = self.ir.assign_new(high_bit);

                // Check if tag == 2 (String constant)
                let string_tag = Value::RawValue(2);
                let use_tag_plus_16 = self.ir.label("use_tag_plus_16");
                let primitive_type_id_done = self.ir.label("primitive_type_id_done");
                let primitive_index_reg = self.ir.volatile_register();

                self.ir.jump_if(use_tag_plus_16, Condition::NotEqual, tag, string_tag);

                // tag == 2: use index 2
                let index_2 = Value::RawValue(2);
                self.ir.assign(primitive_index_reg, index_2);
                self.ir.jump(primitive_type_id_done);

                // tag != 2: use tag + 16
                self.ir.write_label(use_tag_plus_16);
                let sixteen = Value::RawValue(16);
                let sixteen_reg = self.ir.assign_new(sixteen);
                let tag_plus_16 = self.ir.add_int(tag, sixteen_reg);
                self.ir.assign(primitive_index_reg, tag_plus_16);

                self.ir.write_label(primitive_type_id_done);
                let primitive_type_id =
                    self.ir.bitwise_or(high_bit_reg.into(), primitive_index_reg.into());
                self.ir.assign(type_id_reg, primitive_type_id);

                // 7. Compare with cached type_id
                self.ir.write_label(type_id_computed);
                let cached_type_id = self.ir.load_from_memory(cache_ptr_reg.into(), 0);

                self.ir.jump_if(
                    slow_path_label,
                    Condition::NotEqual,
                    type_id_reg,
                    cached_type_id,
                );

                // 8. FAST PATH: load fn_ptr and call
                let fn_ptr = self.ir.load_from_memory(cache_ptr_reg.into(), 1);
                let fn_val = self.ir.function(fn_ptr);

                let call_result = self.ir.call(fn_val, call_args.clone());
                self.ir.assign(result_reg, call_result);
                self.ir.jump(exit_label);

                // 9. SLOW PATH: call builtin to get fn_ptr, then call it
                self.ir.write_label(slow_path_label);

                // Call protocol_dispatch(first_arg, cache_location, dispatch_table_ptr) -> fn_ptr
                let slow_fn_ptr = self.call_builtin(
                    "beagle.builtin/protocol_dispatch",
                    vec![
                        arg0_reg.into(),
                        cache_ptr_reg.into(),
                        dispatch_table_ptr_reg.into(),
                    ],
                );

                // Call the returned fn_ptr with all args
                let fn_val = self.ir.function(slow_fn_ptr);
                let slow_call_result = self.ir.call(fn_val, call_args);
                self.ir.assign(result_reg, slow_call_result);

                // 10. Exit
                self.ir.write_label(exit_label);
                result_reg.into()
            }
            Ast::True(_) => Value::True,
            Ast::False(_) => Value::False,
            Ast::Null(_) => Value::Null,
        }
    }

    fn get_function(&mut self, function_name: &str) -> Value {
        let f = self.compiler.find_function(function_name).unwrap();
        let f = self.compiler.get_function_pointer(f).unwrap();
        let f = self.ir.assign_new(f);
        f.into()
    }

    fn call(&mut self, name: &str, mut args: Vec<Value>) -> Value {
        assert!(
            name.contains("/"),
            "Function name should be fully qualified {}",
            name
        );

        // TODO: I shouldn't just assume the function will exist
        // unless I have a good plan for dealing with when it doesn't
        let function = self.compiler.find_function(name);

        let function = function.unwrap_or_else(|| panic!("Could not find function {}", name));

        let builtin = function.is_builtin;
        let needs_stack_pointer = function.needs_stack_pointer;
        let is_variadic = function.is_variadic;
        let min_args = function.min_args;

        // Arity check - for functions that need stack pointer, number_of_args includes it
        let expected_user_args = if needs_stack_pointer {
            function.number_of_args.saturating_sub(1)
        } else {
            function.number_of_args
        };

        if is_variadic {
            // For variadic functions, we need at least min_args arguments
            if args.len() < min_args {
                panic!(
                    "Variadic function '{}' requires at least {} argument(s), but {} were provided",
                    name,
                    min_args,
                    args.len()
                );
            }
            // Pack extra arguments into a PersistentVector
            args = self.pack_variadic_args(args, min_args);
        } else if args.len() != expected_user_args {
            panic!(
                "Function '{}' expects {} argument(s), but {} were provided\n\
                 (function.number_of_args={}, needs_stack_pointer={}, is_builtin={})",
                name,
                expected_user_args,
                args.len(),
                function.number_of_args,
                needs_stack_pointer,
                builtin
            );
        }
        if needs_stack_pointer {
            let stack_pointer_reg = self.ir.volatile_register();
            let stack_pointer = self.ir.get_stack_pointer_imm(0);
            self.ir.assign(stack_pointer_reg, stack_pointer);
            args.insert(0, stack_pointer);
        }

        let jump_table_pointer = self.compiler.get_jump_table_pointer(function).unwrap();
        let jump_table_point_reg = self.ir.assign_new(Value::Pointer(jump_table_pointer));
        let function_pointer = self.ir.load_from_memory(jump_table_point_reg.into(), 0);

        let function = self.ir.function(function_pointer);
        if builtin {
            // self.ir.breakpoint();
            self.ir.call_builtin(function, args)
        } else {
            self.ir.call(function, args)
        }
    }

    /// Create an empty heap-allocated array (type_id=1, size=0)
    fn create_empty_array(&mut self) -> Value {
        let allocate = self
            .compiler
            .find_function("beagle.builtin/allocate")
            .unwrap();
        let allocate = self.compiler.get_function_pointer(allocate).unwrap();
        let allocate = self.ir.assign_new(allocate);

        let size = Value::TaggedConstant(0);
        let size_reg = self.ir.assign_new(size);
        let stack_pointer = self.ir.get_stack_pointer_imm(0);

        let array_ptr = self
            .ir
            .call_builtin(allocate.into(), vec![stack_pointer, size_reg.into()]);

        let array_pointer = self.ir.untag(array_ptr);
        let type_id_value = Value::RawValue(1);
        self.ir.write_type_id(array_pointer, type_id_value);

        self.ir
            .tag(array_pointer, BuiltInTypes::HeapObject.get_tag())
    }

    /// Pack a slice of arguments into a heap-allocated array
    /// The args should already be evaluated Values
    fn pack_args_into_array(&mut self, args: &[Value]) -> Value {
        let num_args = args.len();

        if num_args == 0 {
            return self.create_empty_array();
        }

        // Push args to stack so they survive GC during allocation
        for arg in args {
            let reg = self.ir.assign_new(*arg);
            self.ir.push_to_stack(reg.into());
        }

        // Allocate array
        let allocate = self
            .compiler
            .find_function("beagle.builtin/allocate")
            .unwrap();
        let allocate = self.compiler.get_function_pointer(allocate).unwrap();
        let allocate = self.ir.assign_new(allocate);

        let size = Value::TaggedConstant(num_args as isize);
        let size_reg = self.ir.assign_new(size);
        let stack_pointer = self.ir.get_stack_pointer_imm(0);

        let array_ptr = self
            .ir
            .call_builtin(allocate.into(), vec![stack_pointer, size_reg.into()]);

        let array_pointer = self.ir.untag(array_ptr);

        // Set type_id to 1 (raw array type)
        let type_id_value = Value::RawValue(1);
        self.ir.write_type_id(array_pointer, type_id_value);

        // Pop args from stack (in reverse order) and write to array
        for i in (0..num_args).rev() {
            let value = self.ir.pop_from_stack();
            let offset = i + 1;
            self.ir.heap_store_offset(array_pointer, value, offset);
        }

        // Tag the array pointer
        self.ir
            .tag(array_pointer, BuiltInTypes::HeapObject.get_tag())
    }

    /// Call a variadic function through a function value, handling min_args dynamically.
    /// This uses a runtime trampoline to avoid complex IR branching that confuses
    /// the register allocator.
    ///
    /// The trampoline (call_variadic_function_value in builtins.rs) handles:
    /// 1. Reading min_args from function metadata at runtime
    /// 2. Packing extra args into a rest array
    /// 3. Dispatching the call with the correct number of arguments
    ///
    /// Returns a VirtualRegister containing the call result.
    fn call_variadic_function_dynamic(
        &mut self,
        function_value: Value,
        args: Vec<Value>,
        _min_args_reg: VirtualRegister, // No longer used - trampoline reads from function metadata
        is_closure: bool,
        closure_reg: Option<VirtualRegister>,
    ) -> VirtualRegister {
        // Pack all args into a single array for the trampoline
        let args_array = self.pack_args_into_array(&args);

        // Build trampoline call arguments - must assign to registers, not pass raw Values
        let is_closure_val = if is_closure {
            self.ir.assign_new(Value::True)
        } else {
            self.ir.assign_new(Value::False)
        };
        let closure_val = match closure_reg {
            Some(r) => r,
            None => self.ir.assign_new(Value::Null),
        };

        // Single call to trampoline - fixed 4 args (stack_pointer added automatically)
        let result = self.call_builtin(
            "beagle.builtin/call_variadic_function_value",
            vec![
                function_value,
                args_array,
                is_closure_val.into(),
                closure_val.into(),
            ],
        );
        self.ir.assign_new(result)
    }

    /// Pack variadic arguments into a raw array
    /// Takes the first min_args as regular arguments, packs the rest into an array
    fn pack_variadic_args(&mut self, mut args: Vec<Value>, min_args: usize) -> Vec<Value> {
        // Split args: keep first min_args, pack the rest
        let extra_args: Vec<Value> = args.drain(min_args..).collect();
        let num_extra = extra_args.len();

        // Push extra args to stack so they survive GC during allocation
        for arg in &extra_args {
            let reg = self.ir.assign_new(*arg);
            self.ir.push_to_stack(reg.into());
        }

        // Allocate a raw array with type_id = 1
        // This uses the same pattern as struct allocation but with type_id instead of struct_id
        let allocate = self
            .compiler
            .find_function("beagle.builtin/allocate")
            .unwrap();
        let allocate = self.compiler.get_function_pointer(allocate).unwrap();
        let allocate = self.ir.assign_new(allocate);

        let size = Value::TaggedConstant(num_extra as isize);
        let size_reg = self.ir.assign_new(size);
        let stack_pointer = self.ir.get_stack_pointer_imm(0);

        let array_ptr = self
            .ir
            .call_builtin(allocate.into(), vec![stack_pointer, size_reg.into()]);

        let array_pointer = self.ir.untag(array_ptr);

        // Set type_id to 1 (raw array type)
        let type_id_value = Value::RawValue(1);
        self.ir.write_type_id(array_pointer, type_id_value);

        // Pop the extra args from stack (in reverse order) and write to array
        for i in (0..num_extra).rev() {
            let value = self.ir.pop_from_stack();
            // Write field at index i (offset in words, +1 to skip header)
            let offset = i + 1;
            self.ir.heap_store_offset(array_pointer, value, offset);
        }

        // Tag the array pointer and append as final argument
        let tagged_array = self
            .ir
            .tag(array_pointer, BuiltInTypes::HeapObject.get_tag());
        args.push(tagged_array);
        args
    }

    fn create_free_if_closable(&mut self, name: &String) -> Option<VariableLocation> {
        let mut location = None;
        for environment in self.environment_stack.iter_mut().rev().skip(1) {
            if let Some(loc) = environment.variables.get(name) {
                location = Some(loc.clone());
                break;
            }
        }
        if let Some(location) = location {
            let free_variable = self.find_or_insert_free_variable(name);
            match location {
                VariableLocation::BoxedMutableLocal(_) => {
                    return Some(VariableLocation::BoxedFreeVariable(free_variable));
                }
                VariableLocation::MutableLocal(_) => {
                    return Some(VariableLocation::MutableFreeVariable(free_variable));
                }
                _ => {
                    return Some(VariableLocation::FreeVariable(free_variable));
                }
            }
        }
        None
    }

    fn get_qualified_function_name(&mut self, name: &String) -> String {
        if name.contains("/") {
            let parts: Vec<&str> = name.split("/").collect();
            let alias = parts[0];
            let name = parts[1];
            let namespace = self
                .compiler
                .get_namespace_from_alias(alias)
                .unwrap_or_else(|| panic!("Can't find alias {}", alias));
            namespace + "/" + name
        } else if self.get_variable(name).is_some() {
            name.clone()
        } else if self
            .compiler
            .find_function(&(self.compiler.current_namespace_name() + "/" + name))
            .is_some()
        {
            self.compiler.current_namespace_name() + "/" + name
        } else if self
            .compiler
            .find_function(&("beagle.core/".to_owned() + name))
            .is_some()
        {
            "beagle.core/".to_string() + name
        } else if self.create_free_if_closable(name).is_some() {
            name.clone()
        } else {
            panic!("Can't find function {}", name);
        }
    }

    fn compile_closure_call(&mut self, function: VariableLocation, args: Vec<Value>) -> Value {
        // self.ir.breakpoint();
        let ret_register = self.ir.assign_new(Value::TaggedConstant(0));
        let function = self.resolve_variable(&function).unwrap();
        let function_register = self.ir.assign_new(function);
        let closure_register = self.ir.assign_new(function_register);
        // Check if the tag is a closure
        let tag = self.ir.get_tag(closure_register.into());
        let closure_tag = BuiltInTypes::Closure.get_tag();
        let closure_tag = Value::RawValue(closure_tag as usize);
        let call_closure = self.ir.label("call_closure");
        let exit_closure_call = self.ir.label("exit_closure_call");
        // TODO: It might be better to change the layout of these jumps
        // so that the non-closure case is the fall through
        // I just have to think about the correct way to do that
        self.ir
            .jump_if(call_closure, Condition::Equal, tag, closure_tag);

        // Non-closure function call path
        // Runtime arity check (handles both variadic and non-variadic)
        let expected_count = self
            .ir
            .assign_new(Value::TaggedConstant(args.len() as isize));
        self.call(
            "beagle.builtin/check_arity",
            vec![function_register.into(), expected_count.into()],
        );

        // Check if function is variadic and pack args if needed
        let is_variadic = self.call_builtin(
            "beagle.builtin/is_function_variadic",
            vec![function_register.into()],
        );
        let is_variadic_reg = self.ir.assign_new(is_variadic);

        let call_variadic = self.ir.label("call_variadic_fn");
        let after_fn_call = self.ir.label("after_fn_call");

        self.ir.jump_if(
            call_variadic,
            Condition::Equal,
            is_variadic_reg,
            Value::True,
        );

        // Non-variadic: call directly
        let result = self.ir.call(function_register.into(), args.clone());
        self.ir.assign(ret_register, result);
        self.ir.jump(after_fn_call);

        // Variadic: use dispatch table to handle min_args correctly
        self.ir.write_label(call_variadic);
        let min_args = self.call_builtin(
            "beagle.builtin/get_function_min_args",
            vec![function_register.into()],
        );
        let min_args_reg = self.ir.assign_new(min_args);
        let variadic_result = self.call_variadic_function_dynamic(
            function_register.into(),
            args.clone(),
            min_args_reg,
            false, // not a closure
            None,
        );
        self.ir.assign(ret_register, variadic_result);

        self.ir.write_label(after_fn_call);
        self.ir.jump(exit_closure_call);
        self.ir.write_label(call_closure);

        // Closure call path
        // I need to grab the function pointer
        // Closures are a pointer to a structure like this
        // struct Closure {
        //     function_pointer: *const u8,
        //     num_free_variables: usize,
        //     free_variables: *const Value,
        // }
        let untagged_closure_register = self.ir.untag(closure_register.into());
        // self.ir.breakpoint();
        // Load function pointer from closure structure (field 0 = offset 1, after 8-byte header)
        let function_pointer = self.ir.load_from_memory(untagged_closure_register, 1);

        // Runtime arity check for closure call (handles both variadic and non-variadic)
        let expected_count = self
            .ir
            .assign_new(Value::TaggedConstant(args.len() as isize));
        self.call(
            "beagle.builtin/check_arity",
            vec![function_pointer, expected_count.into()],
        );

        // Check if closure's function is variadic
        let is_variadic = self.call_builtin(
            "beagle.builtin/is_function_variadic",
            vec![function_pointer],
        );
        let is_variadic_reg = self.ir.assign_new(is_variadic);

        let call_variadic_closure = self.ir.label("call_variadic_closure");
        let after_closure_call = self.ir.label("after_closure_call");

        self.ir.jump_if(
            call_variadic_closure,
            Condition::Equal,
            is_variadic_reg,
            Value::True,
        );

        // Non-variadic closure: call directly
        let mut closure_args = args.clone();
        closure_args.insert(0, closure_register.into());
        let result = self.ir.call(function_pointer, closure_args);
        self.ir.assign(ret_register, result);
        self.ir.jump(after_closure_call);

        // Variadic closure: use dispatch table to handle min_args correctly
        self.ir.write_label(call_variadic_closure);
        let min_args = self.call_builtin(
            "beagle.builtin/get_function_min_args",
            vec![function_pointer],
        );
        let min_args_reg = self.ir.assign_new(min_args);
        let variadic_result = self.call_variadic_function_dynamic(
            function_pointer,
            args.clone(),
            min_args_reg,
            true, // is a closure
            Some(closure_register),
        );
        self.ir.assign(ret_register, variadic_result);

        self.ir.write_label(after_closure_call);
        self.ir.write_label(exit_closure_call);
        ret_register.into()
        // self.ir.breakpoint();
    }

    fn compile_closure(&mut self, function_pointer: usize) -> Value {
        // When I get those free variables, I'd need to
        // make sure that the variables they refer to are
        // heap allocated. How am I going to do that?
        // I actually probably need to think about this more
        // If they are already heap allocated, then I just
        // store the pointer. If they are immutable variables,
        // I just take the value
        // If they are mutable, then I'd need to heap allocate
        // but maybe I just heap allocate all mutable variables?
        // What about functions that change overtime?
        // Not 100% sure about all of this
        let label = self.ir.label("closure");

        // self.ir.breakpoint();
        // get a pointer to the start of the free variables on the stack
        let free_variable_pointer = self.ir.get_current_stack_position();

        self.ir.write_label(label);
        for free_variable in self.get_current_env().free_variables.clone().iter().rev() {
            let variable = self
                .get_variable(&free_variable.name)
                .unwrap_or_else(|| panic!("Can't find variable {:?}", free_variable));
            // we are now going to push these variables onto the stack

            match variable {
                VariableLocation::Register(reg) => {
                    self.ir.push_to_stack(reg.into());
                }
                VariableLocation::Local(index)
                | VariableLocation::MutableLocal(index)
                | VariableLocation::BoxedMutableLocal(index) => {
                    let reg = self.ir.load_local(index);
                    self.ir.push_to_stack(reg);
                }
                VariableLocation::NamespaceVariable(namespace, slot) => {
                    self.resolve_variable(&VariableLocation::NamespaceVariable(namespace, slot))
                        .unwrap();
                }
                VariableLocation::FreeVariable(_)
                | VariableLocation::BoxedFreeVariable(_)
                | VariableLocation::MutableFreeVariable(_) => {
                    panic!(
                        "We are trying to find this variable concretely and found a free variable"
                    )
                }
            }
        }
        // load count of free variables
        let num_free = self.get_current_env().free_variables.len();

        let num_free = Value::TaggedConstant(num_free as isize);
        let num_free_reg = self.ir.volatile_register();
        self.ir.assign(num_free_reg, num_free);
        // Call make_closure
        let make_closure = self
            .compiler
            .find_function("beagle.builtin/make_closure")
            .unwrap();
        let make_closure = self.compiler.get_function_pointer(make_closure).unwrap();
        let make_closure_reg = self.ir.volatile_register();
        self.ir.assign(make_closure_reg, make_closure);
        let function_pointer_reg = self.ir.volatile_register();
        self.ir
            .assign(function_pointer_reg, Value::RawValue(function_pointer));

        let stack_pointer = self.ir.get_stack_pointer_imm(0);

        self.ir.call(
            make_closure_reg.into(),
            vec![
                stack_pointer,
                function_pointer_reg.into(),
                num_free_reg.into(),
                free_variable_pointer,
            ],
        )
    }

    fn find_or_insert_local(&mut self, name: &str) -> usize {
        let current_env = self.environment_stack.last_mut().unwrap();
        if let Some(index) = current_env.local_variables.iter().position(|n| n == name) {
            index
        } else {
            current_env.local_variables.push(name.to_string());
            current_env.local_variables.len() - 1
        }
    }

    fn insert_variable(&mut self, name: String, location: VariableLocation) {
        let current_env = self.environment_stack.last_mut().unwrap();
        current_env.variables.insert(name, location);
    }

    fn get_accessible_variable(&self, name: &str) -> Option<VariableLocation> {
        // Let's look in the current environment
        // Then we will look in the current namespace
        // then we will look in the global namespace
        if let Some(variable) = self.environment_stack.last().unwrap().variables.get(name) {
            Some(variable.clone())
        } else if let Some(slot) = self
            .compiler
            .find_binding(self.compiler.current_namespace_id(), name)
        {
            Some(VariableLocation::NamespaceVariable(
                self.compiler.current_namespace_id(),
                slot,
            ))
        } else {
            // TODO: The global vs beagle.core is ugly
            let global_id = self.compiler.global_namespace_id();
            self.compiler
                .find_binding(self.compiler.global_namespace_id(), name)
                .map(|slot| (global_id, slot))
                .or_else(|| {
                    let beagle_core = self.compiler.get_namespace_id("beagle.core")?;
                    let slot = self.compiler.find_binding(beagle_core, name)?;
                    Some((beagle_core, slot))
                })
                .map(|(id, slot)| VariableLocation::NamespaceVariable(id, slot))
        }
    }

    fn find_or_insert_free_variable(&mut self, name: &str) -> usize {
        let current_env = self.environment_stack.last_mut().unwrap();
        if let Some(index) = current_env
            .free_variables
            .iter()
            .position(|n| n.name == name)
        {
            index
        } else {
            current_env.free_variables.push(FreeVariable {
                name: name.to_string(),
            });
            current_env.free_variables.len() - 1
        }
    }

    fn get_variable_alloc_free_variable(&mut self, name: &str) -> VariableLocation {
        let existing_location = self.get_variable(name);
        if let Some(variable) = self.get_accessible_variable(name) {
            variable.clone()
        } else if name.contains("/") {
            let (namespace_name, name) = self.get_namespace_name_and_name(name);
            let namespace_id = self.compiler.get_namespace_id(&namespace_name).unwrap();

            let slot = self
                .compiler
                .find_binding(namespace_id, &name)
                .unwrap_or_else(|| panic!("Can't find variable {}", name));

            VariableLocation::NamespaceVariable(namespace_id, slot)
        } else if let Some(free_var) = self.create_free_if_closable(&name.to_string()) {
            free_var
        } else {
            let current_env = self.environment_stack.last_mut().unwrap();
            current_env.free_variables.push(FreeVariable {
                name: name.to_string(),
            });
            let index = current_env.free_variables.len() - 1;

            let free = match existing_location {
                Some(VariableLocation::BoxedMutableLocal(_)) => {
                    VariableLocation::BoxedFreeVariable(index)
                }
                Some(VariableLocation::MutableLocal(_)) => {
                    VariableLocation::MutableFreeVariable(index)
                }
                _ => VariableLocation::FreeVariable(index),
            };
            current_env.variables.insert(name.to_string(), free);
            let current_env = self.environment_stack.last().unwrap();
            current_env.variables.get(name).unwrap().clone()
        }
    }

    fn get_variable(&self, name: &str) -> Option<VariableLocation> {
        for env in self.environment_stack.iter().rev() {
            if let Some(variable) = env.variables.get(name)
                && !variable.is_free()
            {
                return Some(variable.clone());
            }
        }
        None
    }

    pub fn string_constant(&mut self, str: String) -> Value {
        self.compiler.add_string(ir::StringValue { str })
    }

    pub fn keyword_constant(&mut self, keyword_text: String) -> Value {
        self.compiler.add_keyword(keyword_text)
    }

    fn create_new_environment(&mut self) {
        self.environment_stack.push(Environment::new());
    }

    fn pop_environment(&mut self) {
        self.environment_stack.pop();
    }

    fn get_current_env(&self) -> &Environment {
        self.environment_stack.last().unwrap()
    }

    fn get_current_env_mut(&mut self) -> &mut Environment {
        self.environment_stack.last_mut().unwrap()
    }

    pub fn call_builtin(&mut self, name: &str, args: Vec<Value>) -> Value {
        self.call(name, args)
    }

    fn first_pass(&mut self, ast: &Ast) {
        match ast {
            Ast::Program { elements, .. } => {
                for ast in elements.iter() {
                    self.first_pass(ast);
                }
            }
            Ast::Function {
                name,
                args,
                rest_param,
                body: _,
                ..
            } => {
                if let Some(name) = name {
                    let full_function_name = self.compiler.current_namespace_name() + "/" + name;
                    let min_args = args.len();
                    let is_variadic = rest_param.is_some();
                    let arity = min_args + if is_variadic { 1 } else { 0 };
                    self.compiler
                        .reserve_function(&full_function_name, arity, is_variadic, min_args)
                        .unwrap();
                } else {
                    panic!("Why do we have a top level function without a name? Is that allowed?");
                }
            }
            Ast::Struct { name, fields, .. } => {
                let (namespace, name) = self.get_namespace_name_and_name(name);
                self.compiler.add_struct(Struct {
                    name: format!("{}/{}", namespace, name),
                    fields: fields
                        .iter()
                        .map(|field| {
                            if let Ast::Identifier(name, _) = field {
                                name.clone()
                            } else {
                                panic!("Expected identifier got {:?}", field)
                            }
                        })
                        .collect(),
                });
            }
            Ast::Enum { name, variants, .. } => {
                let (namespace, name) = self.get_namespace_name_and_name(name);
                let enum_repr = Enum {
                    name: format!("{}/{}", namespace, name),
                    variants: variants
                        .iter()
                        .map(|variant| match variant {
                            Ast::EnumVariant { name, fields, .. } => {
                                let fields = fields
                                    .iter()
                                    .map(|field| {
                                        if let Ast::Identifier(name, _) = field {
                                            name.clone()
                                        } else {
                                            panic!("Expected identifier got {:?}", field)
                                        }
                                    })
                                    .collect();
                                EnumVariant::StructVariant {
                                    name: name.clone(),
                                    fields,
                                }
                            }
                            Ast::EnumStaticVariant { name, .. } => {
                                EnumVariant::StaticVariant { name: name.clone() }
                            }
                            _ => panic!("Expected enum variant got {:?}", variant),
                        })
                        .collect(),
                };

                let (namespace, name) = self.get_namespace_name_and_name(&name);
                self.compiler.add_struct(Struct {
                    name: format!("{}/{}", namespace, name),
                    fields: variants
                        .iter()
                        .map(|variant| match variant {
                            Ast::EnumVariant {
                                name, fields: _, ..
                            } => name.clone(),
                            Ast::EnumStaticVariant { name, .. } => name.clone(),
                            _ => panic!("Expected enum variant got {:?}", variant),
                        })
                        .collect(),
                });

                self.compiler.add_enum(enum_repr);
                for variant in variants.iter() {
                    match variant {
                        Ast::EnumVariant {
                            name: variant_name,
                            fields,
                            ..
                        } => {
                            let (namespace, name) = self.get_namespace_name_and_name(&name);
                            self.compiler.add_struct(Struct {
                                name: format!("{}/{}.{}", namespace, name, variant_name),
                                fields: fields
                                    .iter()
                                    .map(|field| {
                                        if let Ast::Identifier(name, _) = field {
                                            name.clone()
                                        } else {
                                            panic!("Expected identifier got {:?}", field)
                                        }
                                    })
                                    .collect(),
                            })
                        }
                        Ast::EnumStaticVariant {
                            name: variant_name, ..
                        } => {
                            let (namespace, name) = self.get_namespace_name_and_name(&name);
                            self.compiler.add_struct(Struct {
                                name: format!("{}/{}.{}", namespace, name, variant_name),
                                fields: vec![],
                            });
                        }
                        _ => panic!("Expected enum variant got {:?}", variant),
                    }
                }
            }
            Ast::Let {
                name: _, value: _, ..
            } => {}
            Ast::Namespace { name, .. } => {
                let namespace_id = self.compiler.reserve_namespace(name.clone());
                self.compiler.set_current_namespace(namespace_id);
            }
            Ast::Protocol {
                name,
                body,
                token_range: _,
            } => {
                self.compiler.reserve_namespace_slot(name);
                for ast in body.iter() {
                    self.first_pass(ast);
                }
            }
            Ast::FunctionStub {
                name,
                args,
                rest_param,
                token_range: _,
            } => {
                // TODO: Functions should just be stored in the namespace slot
                // self.compiler.reserve_namespace_slot(&name);
                let full_function_name = self.compiler.current_namespace_name() + "/" + name;
                let min_args = args.len();
                let is_variadic = rest_param.is_some();
                let arity = min_args + if is_variadic { 1 } else { 0 };
                self.compiler
                    .reserve_function(&full_function_name, arity, is_variadic, min_args)
                    .unwrap();
            }
            _ => {}
        }
    }

    fn store_namespaced_variable(&mut self, slot: Value, reg: VirtualRegister) {
        let slot: VirtualRegister = self.ir.assign_new(slot);
        self.call_builtin(
            "beagle.builtin/update_binding",
            vec![slot.into(), reg.into()],
        );
    }

    fn resolve_variable(&mut self, reg: &VariableLocation) -> Result<Value, String> {
        match reg {
            VariableLocation::Register(reg) => Ok(Value::Register(*reg)),
            VariableLocation::Local(index) | VariableLocation::MutableLocal(index) => {
                Ok(Value::Local(*index))
            }
            VariableLocation::BoxedMutableLocal(index) => {
                // TODO: I need to deref the box here I think
                let reg = self.ir.load_local(*index);
                let reg = self.ir.untag(reg);
                let value = self.ir.read_field(reg, Value::TaggedConstant(0));
                Ok(value)
            }
            VariableLocation::FreeVariable(index)
            | VariableLocation::MutableFreeVariable(index) => {
                let arg0_location = self.get_argument_location(0).ok_or("Variable not found")?;
                let arg0 = self.resolve_variable(&arg0_location)?;
                let arg0: VirtualRegister = self.ir.assign_new(arg0);
                let arg0 = self.ir.untag(arg0.into());
                let index = self
                    .ir
                    .assign_new(Value::TaggedConstant((*index + 3) as isize));
                Ok(self.ir.read_field(arg0, index.into()))
            }
            VariableLocation::BoxedFreeVariable(index) => {
                let arg0_location = self.get_argument_location(0).ok_or("Variable not found")?;
                let arg0 = self.resolve_variable(&arg0_location)?;
                let arg0: VirtualRegister = self.ir.assign_new(arg0);
                let arg0 = self.ir.untag(arg0.into());
                let index = self
                    .ir
                    .assign_new(Value::TaggedConstant((*index + 3) as isize));
                // TODO: Fix
                let slot = self.ir.read_field(arg0, index.into());
                // TODO: Is it tagged constant or raw?
                let slot = self.ir.untag(slot);
                let value = self.ir.read_field(slot, Value::TaggedConstant(0));
                Ok(value)
            }
            VariableLocation::NamespaceVariable(namespace, slot) => {
                let slot = self.ir.assign_new(*slot);
                let namespace = self.ir.assign_new(*namespace);
                Ok(self.call_builtin(
                    "beagle.builtin/get_binding",
                    vec![namespace.into(), slot.into()],
                ))
            }
        }
    }

    fn name_matches(&self, name: &String) -> bool {
        if self.name.is_none() {
            return false;
        }
        if name.contains("/") {
            *name == self.own_fully_qualified_name().unwrap()
        } else {
            name == self.name.as_ref().unwrap()
        }
    }

    fn own_fully_qualified_name(&self) -> Option<String> {
        let name = self.name.as_ref()?;
        Some(self.compiler.current_namespace_name() + "/" + name)
    }

    fn get_namespace_name_and_name(&self, name: &str) -> (String, String) {
        if name.contains("/") {
            let parts: Vec<&str> = name.split("/").collect();
            let alias = parts[0];
            let name = parts[1];
            let namespace = self
                .compiler
                .get_namespace_from_alias(alias)
                .unwrap_or_else(|| {
                    panic!(
                        "Namespace alias '{}' not found - did you forget to import it?",
                        alias
                    )
                });
            (namespace, name.to_string())
        } else {
            (self.compiler.current_namespace_name(), name.to_string())
        }
    }

    fn register_arg_location(&mut self, index: usize, local: VariableLocation) {
        self.get_current_env_mut()
            .argument_locations
            .insert(index, local);
    }

    fn get_argument_location(&self, index: usize) -> Option<VariableLocation> {
        self.get_current_env()
            .argument_locations
            .get(&index)
            .cloned()
    }

    fn is_qualifed_function_name(&self, name: &str) -> bool {
        name.contains("/")
    }

    fn current_namespace_id(&self) -> usize {
        self.compiler.current_namespace_id()
    }

    fn push_current_token(&mut self, token_range: TokenRange) {
        self.current_token_info
            .push((token_range, self.ir.instructions.len()));
    }

    fn pop_current_token(&mut self) {
        let (token_position, starting_ir_position) = self.current_token_info.pop().unwrap();
        let ending_ir = self.ir.instructions.len();
        self.ir_range_to_token_range.last_mut().unwrap().push((
            TokenRange {
                start: token_position.start,
                end: token_position.end,
            },
            IRRange {
                start: starting_ir_position,
                end: ending_ir,
            },
        ));
    }

    // I'm going to make this an entirely separate pass for simplicities sake
    // I could combine this and be more efficient, but I just want to make things work first
    fn find_mutable_vars_that_need_boxing(&mut self, ast: &Ast) {
        match ast {
            Ast::LetMut {
                name,
                value,
                token_range: _,
            } => {
                self.add_mutable_variable(name, ast);

                self.find_mutable_vars_that_need_boxing(value);
            }

            Ast::Assignment {
                name,
                value,
                token_range: _,
            } => {
                self.update_mutable_variable_meta(name);
                self.find_mutable_vars_that_need_boxing(value);
            }

            Ast::Function {
                name,
                args: _,
                rest_param: _,
                body,
                token_range: _,
            } => {
                if let Some(name) = name {
                    self.add_variable_for_mutable_pass(name);
                }
                self.track_function_mutable_variable_pass();
                for expression in body.iter() {
                    self.find_mutable_vars_that_need_boxing(expression);
                }
                self.pop_function_mutable_variable_pass();
            }

            Ast::Let {
                name,
                value,
                token_range: _,
            } => {
                self.add_variable_for_mutable_pass(&name.get_string());
                self.find_mutable_vars_that_need_boxing(value);
            }

            Ast::Program {
                elements,
                token_range: _,
            } => {
                for ast in elements.iter() {
                    self.find_mutable_vars_that_need_boxing(ast);
                }
            }

            Ast::Struct { .. }
            | Ast::Enum { .. }
            | Ast::EnumVariant { .. }
            | Ast::EnumStaticVariant { .. }
            | Ast::Protocol { .. }
            | Ast::FunctionStub { .. }
            | Ast::IntegerLiteral(_, _)
            | Ast::FloatLiteral(_, _)
            | Ast::Identifier(_, _)
            | Ast::String(_, _)
            | Ast::Keyword(_, _)
            | Ast::True(_)
            | Ast::False(_)
            | Ast::Null(_)
            | Ast::Namespace { .. }
            | Ast::Import { .. } => {
                // I shouldn't need to do anything here
            }

            // TODO: Should I allow these to be closures?
            // Probably makes sense
            Ast::Extend {
                target_type: _,
                protocol: _,
                body,
                token_range: _,
            } => {
                for ast in body.iter() {
                    self.find_mutable_vars_that_need_boxing(ast);
                }
            }
            Ast::If {
                condition,
                then,
                else_,
                token_range: _,
            } => {
                self.find_mutable_vars_that_need_boxing(condition);
                for ast in then.iter() {
                    self.find_mutable_vars_that_need_boxing(ast);
                }
                for ast in else_.iter() {
                    self.find_mutable_vars_that_need_boxing(ast);
                }
            }
            Ast::Condition { left, right, .. }
            | Ast::Add { left, right, .. }
            | Ast::Sub { left, right, .. }
            | Ast::Mul { left, right, .. }
            | Ast::Div { left, right, .. }
            | Ast::ShiftLeft { left, right, .. }
            | Ast::ShiftRight { left, right, .. }
            | Ast::ShiftRightZero { left, right, .. }
            | Ast::BitWiseAnd { left, right, .. }
            | Ast::BitWiseOr { left, right, .. }
            | Ast::BitWiseXor { left, right, .. }
            | Ast::And { left, right, .. }
            | Ast::Or { left, right, .. } => {
                self.find_mutable_vars_that_need_boxing(left);
                self.find_mutable_vars_that_need_boxing(right);
            }

            Ast::Recurse {
                args,
                token_range: _,
            } => {
                for arg in args.iter() {
                    self.find_mutable_vars_that_need_boxing(arg);
                }
            }
            Ast::TailRecurse {
                args,
                token_range: _,
            } => {
                for arg in args.iter() {
                    self.find_mutable_vars_that_need_boxing(arg);
                }
            }
            Ast::Call {
                name: _,
                args,
                token_range: _,
            } => {
                for arg in args.iter() {
                    self.find_mutable_vars_that_need_boxing(arg);
                }
            }

            Ast::StructCreation {
                name: _,
                fields,
                token_range: _,
            } => {
                for (_, field) in fields.iter() {
                    self.find_mutable_vars_that_need_boxing(field);
                }
            }
            Ast::PropertyAccess {
                object,
                property,
                token_range: _,
            } => {
                self.find_mutable_vars_that_need_boxing(object);
                self.find_mutable_vars_that_need_boxing(property);
            }
            Ast::EnumCreation {
                name: _,
                variant: _,
                fields,
                token_range: _,
            } => {
                for (_, field) in fields.iter() {
                    self.find_mutable_vars_that_need_boxing(field);
                }
            }

            Ast::Array {
                array,
                token_range: _,
            } => {
                for element in array.iter() {
                    self.find_mutable_vars_that_need_boxing(element);
                }
            }
            Ast::MapLiteral {
                pairs,
                token_range: _,
            } => {
                for (key, value) in pairs.iter() {
                    self.find_mutable_vars_that_need_boxing(key);
                    self.find_mutable_vars_that_need_boxing(value);
                }
            }
            Ast::IndexOperator {
                array,
                index,
                token_range: _,
            } => {
                self.find_mutable_vars_that_need_boxing(array);
                self.find_mutable_vars_that_need_boxing(index);
            }
            Ast::Loop {
                body,
                token_range: _,
            } => {
                for ast in body.iter() {
                    self.find_mutable_vars_that_need_boxing(ast);
                }
            }
            Ast::While {
                condition,
                body,
                token_range: _,
            } => {
                self.find_mutable_vars_that_need_boxing(condition);
                for ast in body.iter() {
                    self.find_mutable_vars_that_need_boxing(ast);
                }
            }
            Ast::Break {
                value,
                token_range: _,
            } => {
                self.find_mutable_vars_that_need_boxing(value);
            }
            Ast::Continue { token_range: _ } => {
                // No sub-expressions to check
            }
            Ast::Try {
                body,
                exception_binding: _,
                catch_body,
                token_range: _,
            } => {
                for ast in body.iter() {
                    self.find_mutable_vars_that_need_boxing(ast);
                }
                for ast in catch_body.iter() {
                    self.find_mutable_vars_that_need_boxing(ast);
                }
            }
            Ast::Throw {
                value,
                token_range: _,
            } => {
                self.find_mutable_vars_that_need_boxing(value);
            }
            Ast::Match {
                value,
                arms,
                token_range: _,
            } => {
                self.find_mutable_vars_that_need_boxing(value);
                for arm in arms.iter() {
                    for ast in arm.body.iter() {
                        self.find_mutable_vars_that_need_boxing(ast);
                    }
                }
            }
            Ast::For {
                collection,
                body,
                binding: _,
                token_range: _,
            } => {
                self.find_mutable_vars_that_need_boxing(collection);
                for ast in body.iter() {
                    self.find_mutable_vars_that_need_boxing(ast);
                }
            }
            Ast::ProtocolDispatch { .. } => {
                // Protocol dispatch has no sub-expressions, just argument names
            }
        }
    }

    fn add_mutable_variable(&mut self, name: &Ast, ast: &Ast) {
        self.mutable_pass_env_stack.last_mut().unwrap().insert(
            name.get_string(),
            MutablePassInfo {
                mutable_definition: Some(ast.clone()),
            },
        );
    }

    fn track_function_mutable_variable_pass(&mut self) {
        self.mutable_pass_env_stack.push(HashMap::new());
    }

    fn pop_function_mutable_variable_pass(&mut self) {
        self.mutable_pass_env_stack.pop();
    }

    fn update_mutable_variable_meta(&mut self, name: &Ast) {
        let variable_info = self.find_mutable_info(name);
        if let Some((in_current, variable_info)) = variable_info {
            if in_current {
                // It's in the current environment, we don't
                // need to do anything
                return;
            }
            if let Some(ast) = variable_info.mutable_definition {
                self.metadata.insert(
                    ast,
                    Metadata {
                        needs_to_be_boxed: true,
                    },
                );
            }
        } else {
            panic!("Can't find variable {:?}", name);
        }
    }

    fn add_variable_for_mutable_pass(&mut self, name: &str) {
        self.mutable_pass_env_stack
            .last_mut()
            .unwrap()
            .insert(name.to_string(), MutablePassInfo::default());
    }

    fn find_mutable_info(&self, name: &Ast) -> Option<(bool, MutablePassInfo)> {
        let mut in_current = true;
        for env in self.mutable_pass_env_stack.iter().rev() {
            if let Some(variable) = env.get(&name.get_string()) {
                return Some((in_current, variable.clone()));
            }
            in_current = false;
        }
        None
    }

    /// Build the full enum name with namespace.
    /// If enum_name already contains a '/', resolve the alias to get the actual namespace.
    /// Otherwise, prepend the current namespace.
    fn get_full_enum_name(&self, enum_name: &str) -> String {
        if enum_name.contains('/') {
            // Namespace-qualified with alias (e.g., "other/Color")
            // Need to resolve the alias to the actual namespace
            let parts: Vec<&str> = enum_name.split("/").collect();
            let alias = parts[0];
            let name = parts[1];
            // Try to resolve alias, but if it fails, assume alias is already a namespace
            let namespace = self
                .compiler
                .get_namespace_from_alias(alias)
                .unwrap_or_else(|| alias.to_string());
            format!("{}/{}", namespace, name)
        } else {
            // Add current namespace (e.g., "Color" -> "current_namespace/Color")
            format!("{}/{}", self.compiler.current_namespace_name(), enum_name)
        }
    }

    fn compile_pattern_test(&mut self, pattern: &Pattern, value_reg: Value, no_match_label: Label) {
        match pattern {
            Pattern::EnumVariant {
                enum_name,
                variant_name,
                ..
            } => {
                // Get the struct ID for this enum variant
                let full_enum_name = self.get_full_enum_name(enum_name);
                let full_name = format!("{}.{}", full_enum_name, variant_name);
                let (expected_struct_id, _) = self
                    .compiler
                    .get_struct(&full_name)
                    .unwrap_or_else(|| panic!("Enum variant not found: {}", full_name));

                // Untag the value before reading struct ID
                let untagged_value = self.ir.untag(value_reg);

                // Read the struct ID from the value
                let struct_id_reg = self.ir.read_struct_id(untagged_value);

                // Tag the struct ID before comparison (as Int)
                let tagged_struct_id = self.ir.tag(struct_id_reg, BuiltInTypes::Int.get_tag());

                // Create expected value as tagged int
                let expected_value = BuiltInTypes::Int.tag(expected_struct_id as isize);
                let expected_reg = self.ir.assign_new(Value::TaggedConstant(expected_value));

                // Compare and jump if not equal (both values are tagged ints)
                self.ir.jump_if(
                    no_match_label,
                    Condition::NotEqual,
                    tagged_struct_id,
                    expected_reg,
                );
            }
            Pattern::Literal { value, .. } => {
                // Compile the literal value
                let literal_value = self.call_compile(value);

                // Ensure literal is in a register
                let literal_reg = match literal_value {
                    Value::Register(_) => literal_value,
                    _ => {
                        let reg = self.ir.assign_new(literal_value);
                        reg.into()
                    }
                };

                // Use the equal builtin to compare
                let result_value =
                    self.call_builtin("beagle.core/equal", vec![value_reg, literal_reg]);

                // Jump if not equal (false)
                self.ir.jump_if(
                    no_match_label,
                    Condition::NotEqual,
                    result_value,
                    Value::True,
                );
            }
            Pattern::Wildcard { .. } => {
                // Wildcard always matches - no test needed
            }
        }
    }

    fn bind_pattern_variables(&mut self, pattern: &Pattern, value_reg: Value) {
        match pattern {
            Pattern::EnumVariant {
                enum_name,
                variant_name,
                fields,
                ..
            } => {
                // Get the variant's struct definition to look up field indices by name
                let full_enum_name = self.get_full_enum_name(enum_name);
                let full_name = format!("{}.{}", full_enum_name, variant_name);
                let (_, variant_struct) = self
                    .compiler
                    .get_struct(&full_name)
                    .unwrap_or_else(|| panic!("Enum variant not found: {}", full_name));

                // Clone the field names to avoid borrow checker issues
                let variant_field_names = variant_struct.fields.clone();

                // Untag the value once before reading fields
                let untagged_value = self.ir.untag(value_reg);

                // Bind each field to a local variable
                for field_pattern in fields.iter() {
                    // Look up the actual field index in the variant definition
                    let actual_field_idx = variant_field_names
                        .iter()
                        .position(|f| f == &field_pattern.field_name)
                        .unwrap_or_else(|| {
                            panic!(
                                "Field '{}' not found in variant {}.{}",
                                field_pattern.field_name, enum_name, variant_name
                            )
                        });

                    // Read the field from the value using the ACTUAL index, not iteration order
                    let field_value = self.ir.read_field(
                        untagged_value,
                        Value::TaggedConstant(actual_field_idx as isize),
                    );

                    // Determine the variable name to bind to
                    let binding_name = field_pattern
                        .binding_name
                        .as_ref()
                        .unwrap_or(&field_pattern.field_name);

                    // Convert Value to VariableLocation
                    let location = match field_value {
                        Value::Register(reg) => VariableLocation::Register(reg),
                        _ => {
                            // Assign to a register if not already
                            let reg = self.ir.assign_new(field_value);
                            VariableLocation::Register(reg)
                        }
                    };

                    // Add to environment
                    self.insert_variable(binding_name.clone(), location);
                }
            }
            Pattern::Literal { .. } | Pattern::Wildcard { .. } => {
                // Literals and wildcards don't bind variables
            }
        }
    }

    fn check_match_exhaustiveness(&mut self, arms: &[MatchArm], token_range: TokenRange) {
        use std::collections::{HashMap, HashSet};

        // Group patterns by enum type
        let mut enum_coverage: HashMap<String, HashSet<String>> = HashMap::new();
        let mut has_wildcard = false;

        for arm in arms {
            match &arm.pattern {
                Pattern::EnumVariant {
                    enum_name,
                    variant_name,
                    ..
                } => {
                    enum_coverage
                        .entry(enum_name.clone())
                        .or_default()
                        .insert(variant_name.clone());
                }
                Pattern::Wildcard { .. } => {
                    has_wildcard = true;
                }
                Pattern::Literal { .. } => {}
            }
        }

        if has_wildcard {
            return; // Wildcard suppresses all warnings
        }

        // Check each enum independently
        for (enum_name, covered_variants) in enum_coverage {
            // Get the full enum name with namespace
            let full_enum_name = self.get_full_enum_name(&enum_name);

            if let Some(enum_def) = self.compiler.get_enum(&full_enum_name) {
                let all_variants: HashSet<_> = enum_def
                    .variants
                    .iter()
                    .map(|v| match v {
                        crate::runtime::EnumVariant::StructVariant { name, .. } => name,
                        crate::runtime::EnumVariant::StaticVariant { name } => name,
                    })
                    .cloned()
                    .collect();

                let missing: Vec<_> = all_variants
                    .difference(&covered_variants)
                    .cloned()
                    .collect();

                if !missing.is_empty() {
                    // Get line and column from token_line_column_map
                    let (line, column) = if token_range.start < self.token_line_column_map.len() {
                        self.token_line_column_map[token_range.start]
                    } else {
                        (1, 1) // Default if out of bounds
                    };

                    let warning = crate::compiler::CompilerWarning {
                        kind: crate::compiler::WarningKind::NonExhaustiveMatch {
                            enum_name: enum_name.clone(),
                            missing_variants: missing.clone(),
                        },
                        file_name: self.file_name.clone(),
                        token_range,
                        line,
                        column,
                        message: format!(
                            "Non-exhaustive match on enum '{}'. Missing variants: {}",
                            enum_name,
                            missing.join(", ")
                        ),
                    };
                    self.compiler.warnings.lock().unwrap().push(warning);
                }
            }
        }
    }
}
