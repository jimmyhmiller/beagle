use crate::{
    CommandLineArguments, Data, Message,
    arm::LowLevelArm,
    ast::{Ast, TokenRange},
    builtins::debugger,
    code_memory::CodeAllocator,
    debug_only,
    gc::{StackMap, StackMapDetails},
    get_runtime,
    ir::{StringValue, Value},
    parser::Parser,
    pretty_print::PrettyPrint,
    runtime::{Enum, Function, ProtocolMethodInfo, Struct},
    types::BuiltInTypes,
};

use mmap_rs::{MmapMut, MmapOptions};
use std::{
    collections::HashSet,
    env,
    error::Error,
    fmt,
    sync::mpsc::{self, Receiver, SyncSender},
};

pub struct Compiler {
    pub code_allocator: CodeAllocator,
    pub property_look_up_cache: MmapMut,
    pub command_line_arguments: CommandLineArguments,
    pub stack_map: StackMap,
    pub pause_atom_ptr: Option<usize>,
    pub property_look_up_cache_offset: usize,
    pub compiled_file_cache: HashSet<String>,
}

impl Compiler {
    pub fn reset(&mut self) {
        self.code_allocator = CodeAllocator::new();
        self.property_look_up_cache = MmapOptions::new(MmapOptions::page_size())
            .unwrap()
            .map_mut()
            .unwrap()
            .make_mut()
            .unwrap_or_else(|(_map, e)| {
                panic!("Failed to make mmap mutable: {}", e);
            });
        self.property_look_up_cache_offset = 0;
        self.stack_map = StackMap::new();
        self.pause_atom_ptr = None;
        self.compiled_file_cache.clear();
    }

    pub fn get_pause_atom(&self) -> usize {
        self.pause_atom_ptr.unwrap_or(0)
    }

    pub fn set_pause_atom_ptr(&mut self, pointer: usize) {
        self.pause_atom_ptr = Some(pointer);
    }

    pub fn allocate_fn_pointer(&mut self) -> usize {
        let allocate_fn_pointer = self.find_function("beagle.builtin/allocate").unwrap();
        self.get_function_pointer(allocate_fn_pointer).unwrap()
    }

    pub fn add_code(&mut self, code: &[u8]) -> Result<*const u8, Box<dyn Error>> {
        let new_pointer = self.code_allocator.write_bytes(code);
        Ok(new_pointer)
    }

    pub fn compile_string(&mut self, string: &str) -> Result<usize, Box<dyn Error>> {
        let mut parser = Parser::new("".to_string(), string.to_string());
        let ast = parser.parse();
        let top_level = self.compile_ast(ast, Some("REPL_FUNCTION".to_string()), "repl")?;
        self.code_allocator.make_executable();
        if let Some(top_level) = top_level {
            let function = self.get_function_by_name(&top_level).unwrap();
            let function_pointer = self.get_pointer_for_function(function).unwrap();
            Ok(function_pointer)
        } else {
            Ok(0)
        }
    }

    // TODO: I'm going to need to change how this works at some point.
    // I want to be able to dynamically run these namespaces
    // not have this awkward compile returns top level names thing
    pub fn compile(&mut self, file_name: &str) -> Result<Vec<String>, Box<dyn Error>> {
        if self.compiled_file_cache.contains(file_name) {
            if self.command_line_arguments.verbose {
                println!("Already compiled {:?}", file_name);
            }
            return Ok(vec![]);
        }
        if self.command_line_arguments.verbose {
            println!("Compiling {:?}", file_name);
        }

        let parse_time = std::time::Instant::now();
        let code = std::fs::read_to_string(file_name)
            .unwrap_or_else(|_| panic!("Could not find file {:?}", file_name));
        let mut parser = Parser::new(file_name.to_string(), code.to_string());
        let ast = parser.parse();

        if self.command_line_arguments.print_parse {
            println!("{:#?}", ast);
        }

        if self.command_line_arguments.print_ast {
            println!("{:#?}", ast);
        }

        if self.command_line_arguments.show_times {
            println!("Parse time: {:?}", parse_time.elapsed());
        }

        let mut top_levels_to_run = self.compile_dependencies(&ast)?;

        let top_level = self.compile_ast(ast, None, file_name)?;
        if let Some(top_level) = top_level {
            top_levels_to_run.push(top_level);
        }

        if self.command_line_arguments.verbose {
            println!("Done compiling {:?}", file_name);
        }
        self.code_allocator.make_executable();
        self.compiled_file_cache.insert(file_name.to_string());
        Ok(top_levels_to_run)
    }

    pub fn compile_dependencies(
        &mut self,
        ast: &crate::ast::Ast,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let mut top_levels_to_run = vec![];
        for import in ast.imports() {
            let (name, _alias) = self.extract_import(&import);
            if name == "beagle.core"
                || name == "beagle.primitive"
                || name == "beagle.builtin"
                || name == "beagle.__internal_test__"
            {
                continue;
            }
            let file_name = self.get_file_name_from_import(name);
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
    ) -> Result<Option<String>, Box<dyn Error>> {
        let (mut ir, token_map) = ast.compile(self, file_name);
        let top_level_name =
            fn_name.unwrap_or_else(|| format!("{}/__top_level", self.current_namespace_name()));
        if ast.has_top_level() {
            let arm = LowLevelArm::new();
            let error_fn_pointer = self.find_function("beagle.builtin/throw_error").unwrap();
            let error_fn_pointer = self.get_function_pointer(error_fn_pointer).unwrap();

            ir.ir_range_to_token_range = token_map.clone();
            let mut arm = ir.compile(arm, error_fn_pointer);
            let token_map = ir.ir_range_to_token_range.clone();
            let max_locals = arm.max_locals as usize;
            let function_pointer =
                self.upsert_function(Some(&top_level_name), &mut arm, max_locals, 0)?;
            debug_only! {
                debugger(Message {
                    kind: "ir".to_string(),
                    data: Data::Ir {
                        function_pointer,
                        file_name: file_name.to_string(),
                        instructions: ir.instructions.iter().map(|x| x.pretty_print()).collect(),
                        token_range_to_ir_range: token_map
                            .iter()
                            .map(|(token, ir)| ((token.start, token.end), (ir.start, ir.end)))
                            .collect(),
                    },
                });

                let pretty_arm_instructions =
                    arm.instructions.iter().map(|x| x.pretty_print()).collect();
                let ir_to_machine_code_range = ir
                    .ir_to_machine_code_range
                    .iter()
                    .map(|(ir, machine_range)| (*ir, (machine_range.start, machine_range.end)))
                    .collect();

                debugger(crate::Message {
                    kind: "asm".to_string(),
                    data: Data::Arm {
                        function_pointer,
                        file_name: file_name.to_string(),
                        instructions: pretty_arm_instructions,
                        ir_to_machine_code_range,
                    },
                });
            }

            return Ok(Some(top_level_name));
        }
        Ok(None)
    }

    pub fn add_string(&mut self, string_value: StringValue) -> Value {
        let runtime = get_runtime().get_mut();
        let offset = runtime.add_string(string_value);
        Value::StringConstantPtr(offset)
    }

    pub fn add_property_lookup(&mut self) -> usize {
        if self.property_look_up_cache_offset >= self.property_look_up_cache.len() {
            panic!("Property look up cache is full");
        }
        let location = unsafe {
            self.property_look_up_cache
                .as_ptr()
                .add(self.property_look_up_cache_offset) as usize
        };
        self.property_look_up_cache_offset += 2 * 8;
        location
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

    pub fn is_inline_primitive_function(&self, name: &str) -> bool {
        name.starts_with("beagle.primitive/")
    }

    /// TODO: Temporary please fix
    pub fn get_file_name_from_import(&self, import_name: String) -> String {
        let mut exe_path = env::current_exe().unwrap();
        exe_path = exe_path.parent().unwrap().to_path_buf();
        if !exe_path
            .join(format!("resources/{}.bg", import_name))
            .exists()
        {
            exe_path = exe_path.parent().unwrap().to_path_buf();
        }
        exe_path
            .join(format!("resources/{}.bg", import_name))
            .to_str()
            .unwrap()
            .to_string()
    }

    pub fn extract_import(&self, import: &crate::ast::Ast) -> (String, String) {
        match import {
            crate::ast::Ast::Import {
                library_name,
                alias,
                ..
            } => {
                let library_name = library_name.to_string().replace("\"", "");
                let alias = alias.as_ref().get_string();
                (library_name, alias)
            }
            _ => panic!("Not an import"),
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

    pub fn upsert_function(
        &mut self,
        function_name: Option<&str>,
        arm: &mut LowLevelArm,
        max_locals: usize,
        number_of_args: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let code = arm.compile_to_bytes();
        let pointer = self.add_code(&code)?;
        let runtime = get_runtime().get_mut();
        // TODO: Before this we did some weird stuff of mapping over the stack_map details
        // and I don't remember why
        let stack_map = arm
            .translate_stack_map(pointer as usize)
            .iter()
            .map(|(key, value)| {
                (
                    *key,
                    StackMapDetails {
                        function_name: function_name.map(|x| x.to_string()),
                        current_stack_size: *value,
                        number_of_locals: arm.max_locals as usize,
                        max_stack_size: arm.max_stack_size as usize,
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
        )
    }

    pub fn upsert_function_bytes(
        &mut self,
        function_name: Option<&str>,
        code: Vec<u8>,
        max_locals: usize,
        number_of_args: usize,
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
        )
    }

    pub fn add_function_alias(&self, alias: &str, function: &Function) {
        let runtime = get_runtime().get_mut();
        runtime
            .add_function(
                Some(alias),
                usize::from(function.pointer) as *const u8,
                function.size,
                function.number_of_locals,
                function.number_of_args,
            )
            .unwrap();
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
    ) -> Option<Function> {
        let runtime = get_runtime().get_mut();
        runtime
            .reserve_function(full_function_name, number_of_args)
            .ok()
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
                name: "beagle.builtin/throw_error".to_string(),
                args: vec![],
                token_range: TokenRange::new(0, 0),
            };
        }

        let first_method = protocol_methods.first().unwrap();
        let untagged_pointer = BuiltInTypes::untag(first_method.fn_pointer);
        let function_name = self
            .get_function_by_pointer(untagged_pointer)
            .unwrap()
            .name
            .clone();

        Ast::If {
            condition: Box::new(Ast::Call {
                name: "beagle.core/instance_of".to_string(),
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

    fn compile_protocol_method(
        &mut self,
        protocol_name: String,
        method_name: String,
        protocol_methods: Vec<ProtocolMethodInfo>,
    ) {
        let runtime = get_runtime().get_mut();
        let (protocol_namespace, protocol_name) = protocol_name.split_once("/").unwrap();
        let full_method_name = format!("{}/{}", protocol_namespace, method_name);
        let fully_qualified_name =
            format!("{}/{}_{}", protocol_namespace, protocol_name, method_name);
        let default_method = runtime.get_function_by_name(&fully_qualified_name);
        let function = self.find_function(&full_method_name).unwrap();
        let args: Vec<String> = (0..function.number_of_args)
            .map(|x| format!("arg{}", x))
            .collect();

        let current_namespace_id = self.current_namespace_id();
        let protocol_namespace_id = runtime.get_namespace_id(protocol_namespace).unwrap();
        self.set_current_namespace(protocol_namespace_id);

        let ast = Ast::Program {
            elements: vec![Ast::Function {
                name: Some(method_name.clone()),
                args: args.clone(),
                body: vec![self.build_method_if_chain(default_method, protocol_methods, args)],
                token_range: TokenRange::new(0, 0),
            }],
            token_range: TokenRange::new(0, 0),
        };
        self.compile_ast(ast, None, "test").unwrap();
        self.code_allocator.make_executable();
        self.set_current_namespace(current_namespace_id);
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
    CompileFile(String),
    AddFunctionMarkExecutable(String, Vec<u8>, usize, usize),
    CompileProtocolMethod(String, String, Vec<ProtocolMethodInfo>),
    SetPauseAtomPointer(usize),
    Reset,
}

pub enum CompilerResponse {
    Done,
    FunctionsToRun(Vec<String>),
    FunctionPointer(usize),
}

pub struct CompilerThread {
    compiler: Compiler,
    channel: BlockingReceiver<CompilerMessage, CompilerResponse>,
}

impl CompilerThread {
    pub fn new(
        channel: BlockingReceiver<CompilerMessage, CompilerResponse>,
        command_line_arguments: CommandLineArguments,
    ) -> Self {
        CompilerThread {
            compiler: Compiler {
                code_allocator: CodeAllocator::new(),
                property_look_up_cache: MmapOptions::new(MmapOptions::page_size())
                    .unwrap()
                    .map_mut()
                    .unwrap()
                    .make_mut()
                    .unwrap_or_else(|(_map, e)| {
                        panic!("Failed to make mmap mutable: {}", e);
                    }),
                property_look_up_cache_offset: 0,
                command_line_arguments: command_line_arguments.clone(),
                stack_map: StackMap::new(),
                pause_atom_ptr: None,
                compiled_file_cache: HashSet::new(),
            },
            channel,
        }
    }

    pub fn run(&mut self) {
        loop {
            let result = self.channel.receive();
            match result {
                Ok((message, work_done)) => match message {
                    CompilerMessage::CompileString(string) => {
                        let pointer = self.compiler.compile_string(&string).unwrap();
                        work_done.mark_done(CompilerResponse::FunctionPointer(pointer));
                    }
                    CompilerMessage::CompileFile(file_name) => {
                        let top_levels = self.compiler.compile(&file_name).unwrap();
                        work_done.mark_done(CompilerResponse::FunctionsToRun(top_levels));
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
                        self.compiler
                            .upsert_function_bytes(Some(&name), code, max_locals, number_of_args)
                            .unwrap();
                        self.compiler.code_allocator.make_executable();
                        work_done.mark_done(CompilerResponse::Done);
                    }
                    CompilerMessage::Reset => {
                        self.compiler.reset();
                        work_done.mark_done(CompilerResponse::Done);
                    }
                    CompilerMessage::CompileProtocolMethod(
                        protocol_name,
                        method_name,
                        protocol_methods,
                    ) => {
                        self.compiler.compile_protocol_method(
                            protocol_name,
                            method_name,
                            protocol_methods,
                        );
                        work_done.mark_done(CompilerResponse::Done);
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

impl<T, R> BlockingSender<T, R> {
    pub fn send(&self, message: T) -> R {
        let (done_tx, done_rx) = mpsc::sync_channel(0);
        self.inner.send((message, done_tx)).unwrap();
        done_rx.recv().unwrap()
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
        self.done_tx.send(result).unwrap();
    }
}
