use ir::{Ir, Value, VirtualRegister};
use std::collections::HashMap;

use crate::{
    Data, Message,
    arm::LowLevelArm,
    builtins::debugger,
    compiler::Compiler,
    ir::{self, Condition},
    pretty_print::PrettyPrint,
    runtime::{Enum, EnumVariant, Struct},
    types::BuiltInTypes,
};

type TokenPosition = usize;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenRange {
    pub start: usize,
    pub end: usize,
}

impl TokenRange {
    pub fn new(start: usize, end: usize) -> Self {
        TokenRange { start, end }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Ast {
    Program {
        elements: Vec<Ast>,
        token_range: TokenRange,
    },
    Function {
        name: Option<String>,
        // TODO: Change this to a Vec<Ast>
        args: Vec<String>,
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
    IntegerLiteral(i64, TokenPosition),
    FloatLiteral(f64, TokenPosition),
    Identifier(String, TokenPosition),
    String(String, TokenPosition),
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
    IndexOperator {
        array: Box<Ast>,
        index: Box<Ast>,
        token_range: TokenRange,
    },
    Loop {
        body: Vec<Ast>,
        token_range: TokenRange,
    },
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
            | Ast::IndexOperator { token_range, .. }
            | Ast::Loop { token_range, .. }
            | Ast::StructCreation { token_range, .. }
            | Ast::PropertyAccess { token_range, .. }
            | Ast::EnumCreation { token_range, .. } => *token_range,
            Ast::IntegerLiteral(_, position)
            | Ast::FloatLiteral(_, position)
            | Ast::Identifier(_, position)
            | Ast::String(_, position)
            | Ast::True(position)
            | Ast::False(position)
            | Ast::Null(position) => TokenRange::new(*position, *position),
        }
    }
    pub fn compile(
        &self,
        compiler: &mut Compiler,
        file_name: &str,
    ) -> (Ir, Vec<(TokenRange, IRRange)>) {
        let mut ast_compiler = AstCompiler {
            ast: self.clone(),
            file_name: file_name.to_string(),
            ir: Ir::new(compiler.allocate_fn_pointer()),
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
    FreeVariable(usize),
    NamespaceVariable(usize, usize),
}

#[derive(Debug, Clone)]
pub struct Context {
    pub tail_position: bool,
    pub in_function: bool,
}

#[derive(Debug, Clone)]
pub struct Environment {
    pub local_variables: Vec<String>,
    pub variables: HashMap<String, VariableLocation>,
    pub free_variables: Vec<String>,
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
        self.first_pass(&self.ast.clone());

        self.tail_position();
        self.call_compile(&Box::new(self.ast.clone()));

        let mut ir = Ir::new(self.compiler.allocate_fn_pointer());
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
                name, args, body, ..
            } => {
                self.create_new_environment();

                let is_not_top_level = self.environment_stack.len() > 2;

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
                let old_ir =
                    std::mem::replace(&mut self.ir, Ir::new(self.compiler.allocate_fn_pointer()));
                let old_name = self.name.clone();
                self.ir_range_to_token_range.push(Vec::new());

                self.name = name.clone();
                if is_not_top_level {
                    let name = "<closure_context>";
                    let reg = self.ir.arg(0);
                    let local = self.find_or_insert_local(name);
                    let reg = self.ir.assign_new(reg);
                    self.ir.store_local(local, reg.into());
                    let local = VariableLocation::Local(local);
                    self.register_arg_location(0, local.clone());
                    self.insert_variable(name.to_string(), local);
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

                let lang = LowLevelArm::new();

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

                let mut code = self.ir.compile(lang, error_fn_pointer);
                let token_map = self.ir.ir_range_to_token_range.clone();

                let full_function_name = name
                    .clone()
                    .map(|name| self.compiler.current_namespace_name() + "/" + &name);

                let function_pointer = self
                    .compiler
                    .upsert_function(
                        full_function_name.as_deref(),
                        &mut code,
                        self.ir.num_locals,
                        args.len(),
                    )
                    .unwrap();

                code.share_label_info_debug(function_pointer);

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

                let pretty_arm_instructions =
                    code.instructions.iter().map(|x| x.pretty_print()).collect();
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
                    if variable_locaton.is_some() && name.is_some() {
                        let local = self.find_or_insert_local(name.as_ref().unwrap());
                        self.insert_variable(
                            name.as_ref().unwrap().to_string(),
                            VariableLocation::Local(local),
                        );
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
                self.ir.write_label(loop_start);
                for ast in body.iter() {
                    self.call_compile(ast);
                }
                self.ir.jump(loop_start);
                Value::Null
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
                    self.call_compile(ast);
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
                token_range,
            } => {
                // TODO: I should store funcitons in slots instead of a big global list
                // let namespace_slot = self.compiler.find_binding(self.compiler.current_namespace_id(), &name).unwrap();
                // let namespace_slot = Value::TaggedConstant(namespace_slot as isize);

                self.call_compile(&Ast::Function {
                    name: Some(name),
                    args,
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
                        name, args, body, ..
                    } = ast
                    {
                        let name = name.clone().unwrap();
                        // TODO: Hygiene
                        let new_name = format!("{}_{}", target_type, name);
                        self.call_compile(&Ast::Function {
                            name: Some(new_name.clone()),
                            args: args.clone(),
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
                let field_results = fields
                    .iter()
                    .map(|field| {
                        self.not_tail_position();
                        self.call_compile(&field.1)
                    })
                    .collect::<Vec<_>>();

                let (struct_id, struct_type) = self
                    .compiler
                    .get_struct(&format!("{}/{}.{}", namespace, name, variant))
                    .unwrap_or_else(|| panic!("Struct not found {}", name));

                for field in fields.iter() {
                    let mut found = false;
                    for defined_field in struct_type.fields.iter() {
                        if &field.0 == defined_field {
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
                self.ir.write_fields(struct_pointer, &field_results);

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

                let (struct_id, struct_type) = self
                    .compiler
                    .get_struct(&format!("{}/{}", namespace, name))
                    .or_else(|| self.compiler.get_struct(&format!("beagle.core/{}", name)))
                    .unwrap_or_else(|| panic!("Struct not found {}/{}", namespace, name));

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
                let property_location = Value::RawValue(self.compiler.add_property_lookup());
                let property_location = self.ir.assign_new(property_location);
                let property_value = self.ir.load_from_heap(property_location.into(), 0);
                let result = self.ir.assign_new(0);

                let exit_property_access = self.ir.label("exit_property_access");
                let slow_property_path = self.ir.label("slow_property_path");
                // self.ir.jump(slow_property_path);
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
                let args = args
                    .iter()
                    .map(|arg| {
                        self.not_tail_position();
                        self.call_compile(&Box::new(arg.clone()))
                    })
                    .collect();
                if matches!(ast, Ast::TailRecurse { .. }) {
                    self.ir.tail_recurse(args)
                } else {
                    self.ir.recurse(args)
                }
            }
            Ast::Call {
                name,
                args,
                token_range,
            } => {
                let name = self.get_qualified_function_name(&name);

                if self.name_matches(&name) {
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
            Ast::Let { name, value, .. } => {
                if let Ast::Identifier(name, _) = name.as_ref() {
                    if self.environment_stack.len() == 1 {
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
                        self.not_tail_position();
                        let value = self.call_compile(&value);
                        self.not_tail_position();
                        let reg = self.ir.volatile_register();
                        self.ir.assign(reg, value);
                        let local_index = self.find_or_insert_local(name);
                        self.ir.store_local(local_index, reg.into());
                        self.insert_variable(
                            name.to_string(),
                            VariableLocation::Local(local_index),
                        );
                        reg.into()
                    }
                } else {
                    panic!("Expected variable")
                }
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
                self.ir.compare(a, b, operator)
            }
            Ast::String(str, _) => {
                let constant_ptr = self.string_constant(str);
                self.ir.load_string_constant(constant_ptr)
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

    fn create_free_if_closable(&mut self, name: &String) -> Option<VariableLocation> {
        let mut can_be_closed = false;
        for environment in self.environment_stack.iter_mut().rev().skip(1) {
            if environment.variables.contains_key(name) {
                can_be_closed = true;
                break;
            }
        }
        if can_be_closed {
            let free_variable = self.find_or_insert_free_variable(name);
            Some(VariableLocation::FreeVariable(free_variable))
        } else {
            None
        }
    }

    fn get_qualified_function_name(&mut self, name: &String) -> String {
        let full_function_name = if name.contains("/") {
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
        };
        full_function_name
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
        let result = self.ir.call(function_register.into(), args.clone());
        self.ir.assign(ret_register, result);
        self.ir.jump(exit_closure_call);
        self.ir.write_label(call_closure);

        // I need to grab the function pointer
        // Closures are a pointer to a structure like this
        // struct Closure {
        //     function_pointer: *const u8,
        //     num_free_variables: usize,
        //     ree_variables: *const Value,
        // }
        let untagged_closure_register = self.ir.untag(closure_register.into());
        // self.ir.breakpoint();
        let function_pointer = self.ir.load_from_memory(untagged_closure_register, 1);
        // make the first argument to the closure be the values pointer
        // the rest of the arguments are the arguments passed to the closure
        let mut args = args;
        args.insert(0, closure_register.into());
        let result = self.ir.call(function_pointer, args);
        self.ir.assign(ret_register, result);

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
                .get_variable(free_variable)
                .unwrap_or_else(|| panic!("Can't find variable {}", free_variable));
            // we are now going to push these variables onto the stack

            match variable {
                VariableLocation::Register(reg) => {
                    self.ir.push_to_stack(reg.into());
                }
                VariableLocation::Local(index) => {
                    let reg = self.ir.volatile_register();
                    self.ir.load_local(reg, index);
                    self.ir.push_to_stack(reg.into());
                }
                VariableLocation::NamespaceVariable(namespace, slot) => {
                    self.resolve_variable(&VariableLocation::NamespaceVariable(namespace, slot))
                        .unwrap();
                }
                VariableLocation::FreeVariable(_) => {
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
        if let Some(index) = current_env.free_variables.iter().position(|n| n == name) {
            index
        } else {
            current_env.free_variables.push(name.to_string());
            current_env.free_variables.len() - 1
        }
    }

    fn get_variable_alloc_free_variable(&mut self, name: &str) -> VariableLocation {
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
        } else {
            let current_env = self.environment_stack.last_mut().unwrap();
            current_env.free_variables.push(name.to_string());
            let index = current_env.free_variables.len() - 1;
            current_env
                .variables
                .insert(name.to_string(), VariableLocation::FreeVariable(index));
            let current_env = self.environment_stack.last().unwrap();
            current_env.variables.get(name).unwrap().clone()
        }
    }

    fn get_variable(&self, name: &str) -> Option<VariableLocation> {
        for env in self.environment_stack.iter().rev() {
            if let Some(variable) = env.variables.get(name) {
                if !matches!(&variable, VariableLocation::FreeVariable(_)) {
                    return Some(variable.clone());
                }
            }
        }
        None
    }

    pub fn string_constant(&mut self, str: String) -> Value {
        self.compiler.add_string(ir::StringValue { str })
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
                body: _,
                ..
            } => {
                if let Some(name) = name {
                    let full_function_name = self.compiler.current_namespace_name() + "/" + name;
                    self.compiler
                        .reserve_function(&full_function_name, args.len())
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
                token_range: _,
            } => {
                // TODO: Functions should just be stored in the namespace slot
                // self.compiler.reserve_namespace_slot(&name);
                let full_function_name = self.compiler.current_namespace_name() + "/" + name;
                self.compiler
                    .reserve_function(&full_function_name, args.len())
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
            VariableLocation::Local(index) => Ok(Value::Local(*index)),
            VariableLocation::FreeVariable(index) => {
                let arg0_location = self.get_argument_location(0).ok_or("Variable not found")?;
                let arg0 = self.resolve_variable(&arg0_location)?;
                let arg0: VirtualRegister = self.ir.assign_new(arg0);
                let arg0 = self.ir.untag(arg0.into());
                let index = self
                    .ir
                    .assign_new(Value::TaggedConstant((*index + 3) as isize));
                Ok(self.ir.read_field(arg0, index.into()))
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
                .unwrap_or_else(|| panic!("Can't find alias {}", alias));
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
}
