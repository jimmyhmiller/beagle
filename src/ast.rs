use ir::{Ir, Value, VirtualRegister};
use std::collections::HashMap;

use crate::{
    arm::LowLevelArm,
    ir::{self, Condition},
    runtime::{Compiler, Enum, EnumVariant, Struct},
    types::BuiltInTypes,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Ast {
    Program {
        elements: Vec<Ast>,
    },
    Function {
        name: Option<String>,
        // TODO: Change this to a Vec<Ast>
        args: Vec<String>,
        body: Vec<Ast>,
    },
    Struct {
        name: String,
        fields: Vec<Ast>,
    },
    Enum {
        name: String,
        variants: Vec<Ast>,
    },
    EnumVariant {
        name: String,
        fields: Vec<Ast>,
    },
    EnumStaticVariant {
        name: String,
    },
    If {
        condition: Box<Ast>,
        then: Vec<Ast>,
        else_: Vec<Ast>,
    },
    Condition {
        operator: Condition,
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Add {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Sub {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Mul {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Div {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Recurse {
        args: Vec<Ast>,
    },
    TailRecurse {
        args: Vec<Ast>,
    },
    Call {
        name: String,
        args: Vec<Ast>,
    },
    Let(Box<Ast>, Box<Ast>),
    IntegerLiteral(i64),
    FloatLiteral(f64),
    Identifier(String),
    String(String),
    True,
    False,
    StructCreation {
        name: String,
        fields: Vec<(String, Ast)>,
    },
    PropertyAccess {
        object: Box<Ast>,
        property: Box<Ast>,
    },
    Null,
    EnumCreation {
        name: String,
        variant: String,
        fields: Vec<(String, Ast)>,
    },
    Namespace {
        name: String,
    },
    Import {
        library_name: String,
        alias: Box<Ast>,
    },
    ShiftLeft {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    ShiftRight {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    ShiftRightZero {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    BitWiseAnd {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    BitWiseOr {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    BitWiseXor {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    And {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Or {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Array(Vec<Ast>),
    IndexOperator {
        array: Box<Ast>,
        index: Box<Ast>,
    },
}

impl Ast {
    pub fn compile(&self, compiler: &mut Compiler) -> Ir {
        let mut ast_compiler = AstCompiler {
            ast: self.clone(),
            ir: Ir::new(
                compiler.get_compiler_ptr() as usize,
                compiler.allocate_fn_pointer(),
            ),
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
        };

        // println!("{:#?}", compiler);
        ast_compiler.compile()
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
            Ast::Program { elements } => elements,
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
            Ast::String(str) => str.replace("\"", ""),
            Ast::Identifier(str) => str.to_string(),
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
}

impl Environment {
    fn new() -> Self {
        Environment {
            local_variables: vec![],
            variables: HashMap::new(),
            free_variables: vec![],
        }
    }
}

#[derive(Debug)]
pub struct AstCompiler<'a> {
    pub ast: Ast,
    pub ir: Ir,
    pub name: Option<String>,
    pub compiler: &'a mut Compiler,
    // This feels dumb and complicated. But my brain
    // won't let me think of a better way
    // I know there is one.
    pub context: Vec<Context>,
    pub current_context: Context,
    pub next_context: Context,
    pub environment_stack: Vec<Environment>,
}

impl<'a> AstCompiler<'a> {
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
        let result = self.compile_to_ir(ast);
        self.next_context = self.current_context.clone();
        self.current_context = self.context.pop().unwrap();
        result
    }

    pub fn compile(&mut self) -> Ir {
        // TODO: Get rid of clone
        self.first_pass(&self.ast.clone());

        self.tail_position();
        self.call_compile(&Box::new(self.ast.clone()));
        let mut ir = Ir::new(
            self.compiler.get_compiler_ptr() as usize,
            self.compiler.allocate_fn_pointer(),
        );
        std::mem::swap(&mut ir, &mut self.ir);
        ir
    }
    pub fn compile_to_ir(&mut self, ast: &Ast) -> Value {
        match ast.clone() {
            Ast::Program { elements } => {
                let mut last = Value::TaggedConstant(0);
                for ast in elements.iter() {
                    self.tail_position();
                    last = self.call_compile(ast);
                }
                last
            }
            Ast::Function { name, args, body } => {
                self.create_new_environment();
                let old_ir = std::mem::replace(
                    &mut self.ir,
                    Ir::new(
                        self.compiler.get_compiler_ptr() as usize,
                        self.compiler.allocate_fn_pointer(),
                    ),
                );
                self.name = name.clone();
                for (index, arg) in args.iter().enumerate() {
                    let reg = self.ir.arg(index);
                    self.ir.register_argument(reg);
                    self.insert_variable(arg.clone(), VariableLocation::Register(reg));
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
                    let compiler_pointer_reg = self.ir.assign_new(self.compiler.get_compiler_ptr());
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
                    self.ir.call_builtin(
                        pause_function.into(),
                        vec![compiler_pointer_reg.into(), stack_pointer],
                    );
                    self.ir.write_label(skip_pause);
                }

                for ast in body[..body.len().saturating_sub(1)].iter() {
                    self.call_compile(&Box::new(ast));
                }
                let last = body.last().unwrap_or(&Ast::Null);
                let return_value = self.call_compile(&Box::new(last));
                self.ir.ret(return_value);

                let lang = LowLevelArm::new();

                let error_fn_pointer = self
                    .compiler
                    .find_function("beagle.builtin/throw_error")
                    .unwrap();
                let error_fn_pointer = self
                    .compiler
                    .get_function_pointer(error_fn_pointer)
                    .unwrap();

                let compiler_ptr = self.compiler.get_compiler_ptr() as usize;

                let mut code = self.ir.compile(lang, error_fn_pointer, compiler_ptr);

                let full_function_name =
                    name.map(|name| self.compiler.current_namespace_name() + "/" + &name);
                let function_pointer = self
                    .compiler
                    .upsert_function(full_function_name.as_deref(), &mut code, self.ir.num_locals)
                    .unwrap();

                code.share_label_info_debug(function_pointer);

                self.ir = old_ir;

                if self.has_free_variables() {
                    return self.compile_closure(function_pointer);
                }

                let function = self.ir.function(Value::Function(function_pointer));

                self.pop_environment();
                function
            }

            Ast::Struct { name: _, fields: _ } => {
                // TODO: This should probably return the struct value
                // A concept I don't yet have
                Value::Null
            }
            Ast::Enum { name, variants } => {
                let mut struct_fields: Vec<(String, Ast)> = vec![];
                for variant in variants.iter() {
                    match variant {
                        Ast::EnumVariant { name, fields: _ } => {
                            // TODO: These should be functions??
                            // Maybe I should have a concept of a struct/data creator
                            // that gets called with named arguments like that?
                            // I'm not sure
                            // I think my whole setup here is janky. But I want things working first
                            struct_fields.push((name.clone(), Ast::Null));
                        }
                        Ast::EnumStaticVariant { name: variant_name } => {
                            struct_fields.push((
                                variant_name.clone(),
                                Ast::StructCreation {
                                    name: format!("{}.{}", name, variant_name),
                                    fields: vec![],
                                },
                            ));
                        }
                        _ => panic!("Shouldn't get here"),
                    }
                }
                let value = self.call_compile(&Ast::StructCreation {
                    name: name.clone(),
                    fields: struct_fields,
                });
                let namespace_id = self
                    .compiler
                    .find_binding(self.compiler.current_namespace_id(), &name)
                    .unwrap();
                let value_reg = self.ir.assign_new(value);
                self.store_namespaced_variable(Value::RawValue(namespace_id), value_reg);
                // TODO: This should probably return the enum value
                // A concept I don't yet have
                Value::Null
            }
            Ast::EnumVariant { name: _, fields: _ } => {
                panic!("Shouldn't get here")
            }
            Ast::EnumStaticVariant { name: _ } => {
                panic!("Shouldn't get here")
            }
            Ast::EnumCreation {
                name,
                variant,
                fields,
            } => {
                let field_results = fields
                    .iter()
                    .map(|field| {
                        self.not_tail_position();
                        self.call_compile(&field.1)
                    })
                    .collect::<Vec<_>>();

                let (struct_id, struct_type) = self
                    .compiler
                    .get_struct(&format!("{}.{}", name, variant))
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

                let compiler_pointer_reg = self.ir.assign_new(self.compiler.get_compiler_ptr());

                let allocate = self
                    .compiler
                    .find_function("beagle.builtin/allocate")
                    .unwrap();
                let allocate = self.compiler.get_function_pointer(allocate).unwrap();
                let allocate = self.ir.assign_new(allocate);

                let size_reg = self.ir.assign_new(struct_type.size());
                let stack_pointer = self.ir.get_stack_pointer_imm(0);

                let struct_ptr = self.ir.call_builtin(
                    allocate.into(),
                    vec![compiler_pointer_reg.into(), stack_pointer, size_reg.into()],
                );

                let struct_pointer = self.ir.untag(struct_ptr);
                self.ir.write_struct_id(struct_pointer, struct_id);
                self.ir.write_fields(struct_pointer, &field_results);

                self.ir
                    .tag(struct_pointer, BuiltInTypes::HeapObject.get_tag())
            }
            Ast::StructCreation { name, fields } => {
                for field in fields.iter() {
                    self.not_tail_position();
                    let value = self.call_compile(&field.1);
                    let reg = self.ir.assign_new(value);
                    self.ir.push_to_stack(reg.into());
                }

                let (struct_id, struct_type) = self
                    .compiler
                    .get_struct(&name)
                    .unwrap_or_else(|| panic!("Struct not found {}", name));

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

                let compiler_pointer_reg = self.ir.assign_new(self.compiler.get_compiler_ptr());

                let allocate = self
                    .compiler
                    .find_function("beagle.builtin/allocate")
                    .unwrap();
                let allocate = self.compiler.get_function_pointer(allocate).unwrap();
                let allocate = self.ir.assign_new(allocate);

                let size_reg = self.ir.assign_new(struct_type.size());
                let stack_pointer = self.ir.get_stack_pointer_imm(0);

                let struct_ptr = self.ir.call_builtin(
                    allocate.into(),
                    vec![compiler_pointer_reg.into(), stack_pointer, size_reg.into()],
                );

                let struct_pointer = self.ir.untag(struct_ptr);
                self.ir.write_struct_id(struct_pointer, struct_id);

                for field in field_order.iter().rev() {
                    let reg = self.ir.pop_from_stack();
                    self.ir.write_field(struct_pointer, *field, reg);
                }

                self.ir
                    .tag(struct_pointer, BuiltInTypes::HeapObject.get_tag())
            }
            Ast::Array(elements) => {
                // Let's stary by just adding a popping for simplicity
                for element in elements.iter() {
                    self.not_tail_position();
                    let value = self.call_compile(element);
                    let reg = self.ir.assign_new(value);
                    self.ir.push_to_stack(reg.into());
                }

                let vec = self.get_function("persistent_vector/vec");

                let vector_pointer = self.ir.call(vec, vec![]);

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
            Ast::Namespace { name } => {
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
            } => {
                self.compiler
                    .add_alias(library_name, (*alias).get_string().to_string());
                Value::Null
            }
            Ast::PropertyAccess { object, property } => {
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
                let property = if let Ast::Identifier(name) = property.as_ref() {
                    name.clone()
                } else {
                    panic!("Expected identifier")
                };
                let constant_ptr = self.string_constant(property);
                let constant_ptr = self.ir.assign_new(constant_ptr);
                let call_result = self.call_builtin(
                    "beagle.builtin/property_access",
                    vec![object.into(), constant_ptr.into(), property_location.into()],
                );

                self.ir.assign(result, call_result);

                self.ir.write_label(exit_property_access);

                result.into()
            }
            Ast::IndexOperator { array, index } => {
                let get = self.get_function("persistent_vector/get");
                let array = self.call_compile(array.as_ref());
                let index = self.call_compile(index.as_ref());
                let array = self.ir.assign_new(array);
                let index = self.ir.assign_new(index);
                self.ir.call(get, vec![array.into(), index.into()])
            }
            Ast::If {
                condition,
                then,
                else_,
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
            Ast::And { left, right } => {
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
            Ast::Or { left, right } => {
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
            Ast::Add { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.add_any(left, right)
            }
            Ast::Sub { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.sub_any(left, right)
            }
            Ast::Mul { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.mul_any(left, right)
            }
            Ast::Div { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.div_any(left, right)
            }
            Ast::ShiftLeft { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.shift_left(left, right)
            }
            Ast::ShiftRight { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.shift_right(left, right)
            }
            Ast::ShiftRightZero { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.shift_right_zero(left, right)
            }
            Ast::BitWiseAnd { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.bitwise_and(left, right)
            }
            Ast::BitWiseOr { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.bitwise_or(left, right)
            }
            Ast::BitWiseXor { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.bitwise_xor(left, right)
            }
            Ast::Recurse { args } | Ast::TailRecurse { args } => {
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
            Ast::Call { name, args } => {
                let name = self.get_qualified_function_name(&name);
                if Some(name.clone()) == self.own_fully_qualified_name() {
                    if self.is_tail_position() {
                        return self.call_compile(&Ast::TailRecurse { args });
                    } else {
                        return self.call_compile(&Ast::Recurse { args });
                    }
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
                if let Some(function) = self.get_variable_current_env(&name) {
                    self.compile_closure_call(function, args)
                } else {
                    self.compile_standard_function_call(name, args)
                }
            }
            Ast::IntegerLiteral(n) => Value::TaggedConstant(n as isize),
            Ast::FloatLiteral(n) => {
                // floats are heap allocated
                // Sadly I have to do this to avoid loss of percision
                let allocate = self
                    .compiler
                    .find_function("beagle.builtin/allocate_float")
                    .unwrap();
                let allocate = self.compiler.get_function_pointer(allocate).unwrap();
                let allocate = self.ir.assign_new(allocate);

                let compiler_pointer_reg = self.ir.assign_new(self.compiler.get_compiler_ptr());

                let size_reg = self.ir.assign_new(1);
                let stack_pointer = self.ir.get_stack_pointer_imm(0);

                let float_pointer = self.ir.call_builtin(
                    allocate.into(),
                    vec![compiler_pointer_reg.into(), stack_pointer, size_reg.into()],
                );

                let float_pointer = self.ir.untag(float_pointer);
                self.ir.write_small_object_header(float_pointer);
                self.ir.write_float_literal(float_pointer, n);

                self.ir.tag(float_pointer, BuiltInTypes::Float.get_tag())
            }
            Ast::Identifier(name) => {
                let reg = &self.get_variable_alloc_free_variable(&name);
                self.resolve_variable(reg)
            }
            Ast::Let(name, value) => {
                if let Ast::Identifier(name) = name.as_ref() {
                    if self.environment_stack.len() == 1 {
                        self.not_tail_position();
                        let value = self.call_compile(&value);
                        self.not_tail_position();
                        let reg = self.ir.volatile_register();
                        self.ir.assign(reg, value);
                        let namespace_id = self.compiler.current_namespace_id();
                        let reserved_namespace_slot = self.compiler.reserve_namespace_slot(name);
                        self.store_namespaced_variable(
                            Value::RawValue(reserved_namespace_slot),
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
                        self.ir.store_local(local_index, reg);
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
            } => {
                self.not_tail_position();
                let a = self.call_compile(&left);
                self.not_tail_position();
                let b = self.call_compile(&right);
                self.ir.compare(a, b, operator)
            }
            Ast::String(str) => {
                let constant_ptr = self.string_constant(str);
                self.ir.load_string_constant(constant_ptr)
            }
            Ast::True => Value::True,
            Ast::False => Value::False,
            Ast::Null => Value::Null,
        }
    }

    fn get_function(&mut self, function_name: &str) -> Value {
        let f = self.compiler.find_function(function_name).unwrap();
        let f = self.compiler.get_function_pointer(f).unwrap();
        let f = self.ir.assign_new(f);
        f.into()
    }

    fn compile_standard_function_call(&mut self, name: String, mut args: Vec<Value>) -> Value {
        assert!(name.contains("/"));

        // TODO: I shouldn't just assume the function will exist
        // unless I have a good plan for dealing with when it doesn't
        let function = self.compiler.find_function(&name);

        let function = function.unwrap_or_else(|| panic!("Could not find function {}", name));

        let builtin = function.is_builtin;
        let needs_stack_pointer = function.needs_stack_pointer;
        if builtin {
            let pointer_reg = self.ir.volatile_register();
            let pointer: Value = self.compiler.get_compiler_ptr().into();
            self.ir.assign(pointer_reg, pointer);
            args.insert(0, pointer_reg.into());
        }
        if needs_stack_pointer {
            let stack_pointer_reg = self.ir.volatile_register();
            let stack_pointer = self.ir.get_stack_pointer_imm(0);
            self.ir.assign(stack_pointer_reg, stack_pointer);
            args.insert(1, stack_pointer);
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
        } else if self.get_variable_in_stack(name).is_some() {
            name.clone()
        } else if self
            .compiler
            .find_function(&(self.compiler.current_namespace_name() + "/" + name))
            .is_some()
        {
            self.compiler.current_namespace_name() + "/" + name
        } else {
            self.compiler
                .find_function(&("beagle.core/".to_owned() + name))
                .unwrap();
            "beagle.core/".to_string() + name
        };
        full_function_name
    }

    fn compile_closure_call(&mut self, function: VariableLocation, args: Vec<Value>) -> Value {
        let function_register = self.ir.volatile_register();

        let closure_register = self.ir.volatile_register();
        let function = self.resolve_variable(&function);
        self.ir.assign(closure_register, function);
        // Check if the tag is a closure
        let tag = self.ir.get_tag(closure_register.into());
        let closure_tag = BuiltInTypes::Closure.get_tag();
        let closure_tag = Value::RawValue(closure_tag as usize);
        let call_function = self.ir.label("call_function");
        let skip_load_function = self.ir.label("skip_load_function");
        // TODO: It might be better to change the layout of these jumps
        // so that the non-closure case is the fall through
        // I just have to think about the correct way to do that
        self.ir
            .jump_if(call_function, Condition::NotEqual, tag, closure_tag);
        // I need to grab the function pointer
        // Closures are a pointer to a structure like this
        // struct Closure {
        //     function_pointer: *const u8,
        //     num_free_variables: usize,
        //     ree_variables: *const Value,
        // }
        let closure_register = self.ir.untag(closure_register.into());
        let function_pointer = self.ir.load_from_memory(closure_register, 1);

        self.ir.assign(function_register, function_pointer);

        // TODO: I need to fix how these are stored on the stack

        let num_free_variables = self.ir.load_from_memory(closure_register, 2);
        let num_free_variables = self.ir.tag(num_free_variables, BuiltInTypes::Int.get_tag());
        // for each variable I need to push them onto the stack after the prelude
        let loop_start = self.ir.label("loop_start");
        let counter = self.ir.volatile_register();
        // self.ir.breakpoint();
        self.ir.assign(counter, Value::TaggedConstant(0));
        self.ir.write_label(loop_start);
        self.ir.jump_if(
            skip_load_function,
            Condition::GreaterThanOrEqual,
            counter,
            num_free_variables,
        );
        let free_variable_offset = self.ir.add_int(counter, Value::TaggedConstant(4));
        let free_variable_offset = self.ir.mul(free_variable_offset, Value::TaggedConstant(8));
        let free_variable_offset = self.ir.untag(free_variable_offset);
        let free_variable = self
            .ir
            .heap_load_with_reg_offset(closure_register, free_variable_offset);

        let free_variable_offset = self.ir.sub_int(num_free_variables, counter);
        let num_local = self.ir.load_from_memory(closure_register, 3);
        let num_local = self.ir.tag(num_local, BuiltInTypes::Int.get_tag());
        let free_variable_offset = self.ir.add_int(free_variable_offset, num_local);
        // TODO: Make this better
        let free_variable_offset = self.ir.mul(free_variable_offset, Value::TaggedConstant(-8));
        let free_variable_offset = self.ir.untag(free_variable_offset);
        let free_variable_slot_pointer = self.ir.get_stack_pointer_imm(2);
        self.ir.heap_store_with_reg_offset(
            free_variable_slot_pointer,
            free_variable,
            free_variable_offset,
        );

        let label = self.ir.label("increment_counter");
        self.ir.write_label(label);
        let counter_increment = self.ir.add_int(Value::TaggedConstant(1), counter);
        self.ir.assign(counter, counter_increment);

        self.ir.jump(loop_start);
        self.ir.extend_register_life(num_free_variables);
        self.ir.extend_register_life(counter.into());
        self.ir.extend_register_life(closure_register);
        self.ir.write_label(call_function);
        self.ir.assign(function_register, function);
        self.ir.write_label(skip_load_function);
        self.ir.call(function_register.into(), args)
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
        for free_variable in self.get_current_env().free_variables.clone().iter() {
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
                    self.resolve_variable(&VariableLocation::NamespaceVariable(namespace, slot));
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
        self.ir.assign(function_pointer_reg, function_pointer);

        let compiler_pointer_reg = self.ir.assign_new(self.compiler.get_compiler_ptr());

        let stack_pointer = self.ir.get_stack_pointer_imm(0);

        let closure = self.ir.call(
            make_closure_reg.into(),
            vec![
                compiler_pointer_reg.into(),
                stack_pointer,
                function_pointer_reg.into(),
                num_free_reg.into(),
                free_variable_pointer,
            ],
        );
        self.pop_environment();
        closure
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

    fn insert_variable(&mut self, clone: String, reg: VariableLocation) {
        let current_env = self.environment_stack.last_mut().unwrap();
        current_env.variables.insert(clone, reg);
    }

    // TODO: Need to walk the environment stack
    fn get_variable_current_env(&self, name: &str) -> Option<VariableLocation> {
        self.environment_stack
            .last()
            .unwrap()
            .variables
            .get(name)
            .cloned()
    }

    fn get_variable_in_stack(&self, name: &str) -> Option<VariableLocation> {
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
            self.compiler
                .find_binding(self.compiler.global_namespace_id(), name)
                .map(|slot| {
                    VariableLocation::NamespaceVariable(self.compiler.global_namespace_id(), slot)
                })
        }
    }

    fn get_variable_alloc_free_variable(&mut self, name: &str) -> VariableLocation {
        // TODO: Should walk the environment stack
        if let Some(variable) = self.get_variable_in_stack(name) {
            variable.clone()
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

    fn string_constant(&mut self, str: String) -> Value {
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

    fn has_free_variables(&self) -> bool {
        let current_env = self.get_current_env();
        !current_env.free_variables.is_empty()
    }

    pub fn call_builtin(&mut self, arg: &str, args: Vec<Value>) -> Value {
        let mut args = args;
        let function = self
            .compiler
            .find_function(arg)
            .unwrap_or_else(|| panic!("could not find function {}", arg));
        assert!(function.is_builtin);
        let function = self.compiler.get_function_pointer(function).unwrap();
        let function = self.ir.assign_new(function);
        let pointer_reg = self.ir.volatile_register();
        let pointer: Value = self.compiler.get_compiler_ptr().into();
        self.ir.assign(pointer_reg, pointer);
        args.insert(0, pointer_reg.into());
        self.ir.call(function.into(), args)
    }

    fn first_pass(&mut self, ast: &Ast) {
        match ast {
            Ast::Program { elements } => {
                for ast in elements.iter() {
                    self.first_pass(ast);
                }
            }
            Ast::Function {
                name,
                args: _,
                body: _,
            } => {
                if let Some(name) = name {
                    let full_function_name = self.compiler.current_namespace_name() + "/" + name;
                    self.compiler.reserve_function(&full_function_name).unwrap();
                } else {
                    panic!("Why do we have a top level function without a name? Is that allowed?");
                }
            }
            Ast::Struct { name, fields } => {
                self.compiler.add_struct(Struct {
                    name: name.clone(),
                    fields: fields
                        .iter()
                        .map(|field| {
                            if let Ast::Identifier(name) = field {
                                name.clone()
                            } else {
                                panic!("Expected identifier got {:?}", field)
                            }
                        })
                        .collect(),
                });
            }
            Ast::Enum { name, variants } => {
                let enum_repr = Enum {
                    name: name.clone(),
                    variants: variants
                        .iter()
                        .map(|variant| match variant {
                            Ast::EnumVariant { name, fields } => {
                                let fields = fields
                                    .iter()
                                    .map(|field| {
                                        if let Ast::Identifier(name) = field {
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
                            Ast::EnumStaticVariant { name } => {
                                EnumVariant::StaticVariant { name: name.clone() }
                            }
                            _ => panic!("Expected enum variant got {:?}", variant),
                        })
                        .collect(),
                };

                self.compiler.add_struct(Struct {
                    name: name.to_string(),
                    fields: variants
                        .iter()
                        .map(|variant| match variant {
                            Ast::EnumVariant { name, fields: _ } => name.clone(),
                            Ast::EnumStaticVariant { name } => name.clone(),
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
                        } => self.compiler.add_struct(Struct {
                            name: format!("{}.{}", name, variant_name),
                            fields: fields
                                .iter()
                                .map(|field| {
                                    if let Ast::Identifier(name) = field {
                                        name.clone()
                                    } else {
                                        panic!("Expected identifier got {:?}", field)
                                    }
                                })
                                .collect(),
                        }),
                        Ast::EnumStaticVariant { name: variant_name } => {
                            self.compiler.add_struct(Struct {
                                name: format!("{}.{}", name, variant_name),
                                fields: vec![],
                            });
                        }
                        _ => panic!("Expected enum variant got {:?}", variant),
                    }
                }
            }
            Ast::Let(_, _) => {}
            Ast::Namespace { name } => {
                let namespace_id = self.compiler.reserve_namespace(name.clone());
                self.compiler.set_current_namespace(namespace_id);
            }
            _ => {}
        }
    }

    fn store_namespaced_variable(&mut self, slot: Value, reg: VirtualRegister) {
        let slot = self.ir.assign_new(slot);
        self.call_builtin(
            "beagle.builtin/update_binding",
            vec![slot.into(), reg.into()],
        );
    }

    fn resolve_variable(&mut self, reg: &VariableLocation) -> Value {
        match reg {
            VariableLocation::Register(reg) => Value::Register(*reg),
            VariableLocation::Local(index) => Value::Local(*index),
            VariableLocation::FreeVariable(index) => Value::FreeVariable(*index),
            VariableLocation::NamespaceVariable(namespace, slot) => {
                let slot = self.ir.assign_new(Value::RawValue(*slot));
                let namespace = self.ir.assign_new(Value::RawValue(*namespace));
                self.call_builtin(
                    "beagle.builtin/get_binding",
                    vec![namespace.into(), slot.into()],
                )
            }
        }
    }

    fn own_fully_qualified_name(&self) -> Option<String> {
        let name = self.name.as_ref()?;
        Some(self.compiler.current_namespace_name() + "/" + name)
    }
}

impl From<i64> for Ast {
    fn from(val: i64) -> Self {
        Ast::IntegerLiteral(val)
    }
}

impl From<&'static str> for Ast {
    fn from(val: &'static str) -> Self {
        Ast::String(val.to_string())
    }
}
