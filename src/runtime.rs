use core::fmt;
use std::{
    collections::HashMap,
    error::Error,
    slice::from_raw_parts_mut,
    sync::{atomic::AtomicUsize, Arc, Condvar, Mutex, TryLockError},
    thread::{self, JoinHandle, Thread, ThreadId},
    time::Duration,
    vec,
};

use mmap_rs::{Mmap, MmapMut, MmapOptions};

use crate::{
    arm::LowLevelArm,
    debugger,
    ir::{StringValue, Value},
    CommandLineArguments, Data, Message, __pause,
    gc::{AllocateAction, Allocator, AllocatorOptions, StackMap, StackMapDetails, STACK_SIZE},
    parser::Parser,
    types::{BuiltInTypes, HeapObject},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    name: String,
    offset_or_pointer: usize,
    jump_table_offset: usize,
    is_foreign: bool,
    pub is_builtin: bool,
    pub needs_stack_pointer: bool,
    is_defined: bool,
    number_of_locals: usize,
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub name: String,
    pub fields: Vec<String>,
}

impl Struct {
    pub fn size(&self) -> usize {
        self.fields.len()
    }
}

struct StructManager {
    name_to_id: HashMap<String, usize>,
    structs: Vec<Struct>,
}

impl StructManager {
    fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            structs: Vec::new(),
        }
    }

    fn insert(&mut self, name: String, s: Struct) {
        let id = self.structs.len();
        self.name_to_id.insert(name.clone(), id);
        self.structs.push(s);
    }

    fn get(&self, name: &str) -> Option<(usize, &Struct)> {
        let id = self.name_to_id.get(name)?;
        self.structs.get(*id).map(|x| (*id, x))
    }

    pub fn get_by_id(&self, type_id: usize) -> Option<&Struct> {
        self.structs.get(type_id)
    }
}

pub trait Printer {
    fn print(&mut self, value: String);
    fn println(&mut self, value: String);
    // Gross just for testing. I'll need to do better;
    fn get_output(&self) -> Vec<String>;
}

pub struct DefaultPrinter;

impl Printer for DefaultPrinter {
    fn print(&mut self, value: String) {
        print!("{}", value);
    }

    fn println(&mut self, value: String) {
        println!("{}", value);
    }

    fn get_output(&self) -> Vec<String> {
        unimplemented!("We don't store this in the default")
    }
}

pub struct TestPrinter {
    pub output: Vec<String>,
    pub other_printer: Box<dyn Printer>,
}

impl TestPrinter {
    pub fn new(other_printer: Box<dyn Printer>) -> Self {
        Self {
            output: vec![],
            other_printer,
        }
    }
}

impl Printer for TestPrinter {
    fn print(&mut self, value: String) {
        self.output.push(value.clone());
        // self.other_printer.print(value);
    }

    fn println(&mut self, value: String) {
        self.output.push(value.clone() + "\n");
        // self.other_printer.println(value);
    }

    fn get_output(&self) -> Vec<String> {
        self.output.clone()
    }
}

pub struct ThreadState {
    pub paused_threads: usize,
    pub stack_pointers: Vec<(usize, usize)>,
}

impl ThreadState {
    pub fn pause(&mut self, stack_pointer: (usize, usize)) {
        self.paused_threads += 1;
        self.stack_pointers.push(stack_pointer);
    }

    pub fn unpause(&mut self) {
        self.paused_threads -= 1;
    }

    pub fn clear(&mut self) {
        self.stack_pointers.clear();
    }
}

struct Namespace {
    name: String,
    ids: Vec<String>,
    bindings: HashMap<String, usize>,
    #[allow(unused)]
    aliases: HashMap<String, String>,
}

pub struct Runtime<Alloc: Allocator> {
    pub compiler: Compiler,
    pub memory: Memory<Alloc>,
    command_line_arguments: CommandLineArguments,
    pub printer: Box<dyn Printer>,
    // TODO: I don't have any code that looks at u8, just always u64
    // so that's why I need usize
    pub is_paused: AtomicUsize,
    pub thread_state: Arc<(Mutex<ThreadState>, Condvar)>,
    pub gc_lock: Mutex<()>,
}

pub struct Memory<Alloc: Allocator> {
    heap: Alloc,
    stacks: Vec<(ThreadId, MmapMut)>,
    pub join_handles: Vec<JoinHandle<u64>>,
    pub threads: Vec<Thread>,
    pub stack_map: StackMap,
    #[allow(unused)]
    command_line_arguments: CommandLineArguments,
}
impl<Alloc: Allocator> Memory<Alloc> {
    fn active_threads(&mut self) -> usize {
        let mut completed_threads = vec![];
        for (index, thread) in self.join_handles.iter().enumerate() {
            if thread.is_finished() {
                completed_threads.push(index);
            }
        }
        for index in completed_threads.iter().rev() {
            let thread_id = self.join_handles.get(*index).unwrap().thread().id();
            self.stacks.retain(|(id, _)| *id != thread_id);
            self.threads.retain(|t| t.id() != thread_id);
            self.join_handles.remove(*index);
        }

        self.join_handles.len()
    }
}

pub enum EnumVariant {
    StructVariant { name: String, fields: Vec<String> },
    StaticVariant { name: String },
}

pub struct Enum {
    pub name: String,
    pub variants: Vec<EnumVariant>,
}

struct EnumManager {
    name_to_id: HashMap<String, usize>,
    enums: Vec<Enum>,
}

impl EnumManager {
    fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            enums: Vec::new(),
        }
    }

    fn insert(&mut self, e: Enum) {
        let id = self.enums.len();
        self.name_to_id.insert(e.name.clone(), id);
        self.enums.push(e);
    }

    fn get(&self, name: &str) -> Option<&Enum> {
        let id = self.name_to_id.get(name)?;
        self.enums.get(*id)
    }
}

struct NamespaceManager {
    namespaces: Vec<Mutex<Namespace>>,
    current_namespace: usize,
}

impl NamespaceManager {
    fn new() -> Self {
        Self {
            namespaces: vec![Mutex::new(Namespace {
                name: "global".to_string(),
                ids: vec![],
                bindings: HashMap::new(),
                aliases: HashMap::new(),
            })],
            current_namespace: 0,
        }
    }

    fn add_namespace(&mut self, name: &str) -> usize {
        let position = self
            .namespaces
            .iter()
            .position(|n| n.lock().unwrap().name == name);
        if let Some(position) = position {
            return position;
        }

        self.namespaces.push(Mutex::new(Namespace {
            name: name.to_string(),
            ids: vec![],
            bindings: HashMap::new(),
            aliases: HashMap::new(),
        }));
        self.namespaces.len() - 1
    }

    #[allow(unused)]
    fn get_namespace(&self, name: &str) -> Option<&Mutex<Namespace>> {
        self.namespaces
            .iter()
            .find(|n| n.lock().unwrap().name == name)
    }

    fn get_current_namespace(&self) -> &Mutex<Namespace> {
        self.namespaces.get(self.current_namespace).unwrap()
    }

    #[allow(unused)]
    fn set_current_namespace(&mut self, name: &str) {
        let index = self
            .namespaces
            .iter()
            .position(|n| n.lock().unwrap().name == name)
            .unwrap();
        self.current_namespace = index;
    }

    fn add_binding(&self, name: &str, pointer: usize) -> usize {
        let mut namespace = self.get_current_namespace().lock().unwrap();
        if namespace.bindings.contains_key(name) {
            namespace.bindings.insert(name.to_string(), pointer);
            return namespace.ids.iter().position(|n| n == name).unwrap();
        }
        namespace.bindings.insert(name.to_string(), pointer);
        namespace.ids.push(name.to_string());
        namespace.ids.len() - 1
    }

    #[allow(unused)]
    fn find_binding_id(&self, name: &str) -> Option<usize> {
        let namespace = self.get_current_namespace().lock().unwrap();
        namespace.ids.iter().position(|n| n == name)
    }
}

pub struct Compiler {
    code_offset: usize,
    code_memory: Option<Mmap>,
    // TODO: Think about the jump table more
    jump_table: Option<Mmap>,
    // Do I need this offset?
    jump_table_offset: usize,
    structs: StructManager,
    enums: EnumManager,
    functions: Vec<Function>,
    string_constants: Vec<StringValue>,
    namespaces: NamespaceManager,
    #[allow(unused)]
    command_line_arguments: CommandLineArguments,
    pub compiler_pointer: Option<*const Compiler>,
    // TODO: Need to transfer these after I compiler to memory
    // is there a better way?
    // Should I pass runtime to all the compiler stuff?
    pub stack_map: StackMap,
    pub pause_atom_ptr: Option<usize>,
}

impl Compiler {
    pub fn get_pause_atom(&self) -> usize {
        self.pause_atom_ptr.unwrap_or(0)
    }

    pub fn set_pause_atom_ptr(&mut self, pointer: usize) {
        self.pause_atom_ptr = Some(pointer);
    }

    pub fn get_compiler_ptr(&self) -> *const Compiler {
        self.compiler_pointer.unwrap()
    }

    pub fn set_compiler_lock_pointer(&mut self, pointer: *const Compiler) {
        self.compiler_pointer = Some(pointer);
    }

    pub fn add_foreign_function(
        &mut self,
        name: Option<&str>,
        function: *const u8,
    ) -> Result<usize, Box<dyn Error>> {
        let index = self.functions.len();
        let offset = function as usize;
        let jump_table_offset = self.add_jump_table_entry(index, offset)?;
        self.functions.push(Function {
            name: name.unwrap_or("<Anonymous>").to_string(),
            offset_or_pointer: offset,
            jump_table_offset,
            is_foreign: true,
            is_builtin: false,
            needs_stack_pointer: false,
            is_defined: true,
            number_of_locals: 0,
        });
        debugger(Message {
            kind: "foreign_function".to_string(),
            data: Data::ForeignFunction {
                name: name.unwrap_or("<Anonymous>").to_string(),
                pointer: Self::get_function_pointer(self, self.functions.last().unwrap().clone())
                    .unwrap(),
            },
        });
        Ok(self.functions.len() - 1)
    }
    pub fn add_builtin_function(
        &mut self,
        name: &str,
        function: *const u8,
        needs_stack_pointer: bool,
    ) -> Result<usize, Box<dyn Error>> {
        let index = self.functions.len();
        let offset = function as usize;

        let jump_table_offset = self.add_jump_table_entry(index, offset)?;
        self.functions.push(Function {
            name: name.to_string(),
            offset_or_pointer: offset,
            jump_table_offset,
            is_foreign: true,
            is_builtin: true,
            needs_stack_pointer,
            is_defined: true,
            number_of_locals: 0,
        });
        let pointer =
            Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap();
        self.namespaces.add_binding(name, pointer);
        debugger(Message {
            kind: "builtin_function".to_string(),
            data: Data::BuiltinFunction {
                name: name.to_string(),
                pointer,
            },
        });
        Ok(self.functions.len() - 1)
    }

    pub fn upsert_function(
        &mut self,
        name: Option<&str>,
        code: &mut LowLevelArm,
        number_of_locals: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let bytes = &(code.compile_to_bytes());
        let mut already_defined = false;
        let mut function_pointer = 0;
        if name.is_some() {
            for (index, function) in self.functions.iter_mut().enumerate() {
                if function.name == name.unwrap() {
                    function_pointer = self.overwrite_function(index, bytes)?;
                    self.namespaces.add_binding(name.unwrap(), function_pointer);
                    already_defined = true;
                    break;
                }
            }
        }
        if !already_defined {
            function_pointer = self.add_function(name, bytes, number_of_locals)?;
        }
        assert!(function_pointer != 0);

        let translated_stack_map = code.translate_stack_map(function_pointer);
        let translated_stack_map: Vec<(usize, StackMapDetails)> = translated_stack_map
            .iter()
            .map(|(key, value)| {
                (
                    *key,
                    StackMapDetails {
                        current_stack_size: *value,
                        number_of_locals: code.max_locals as usize,
                        max_stack_size: code.max_stack_size as usize,
                    },
                )
            })
            .collect();

        debugger(Message {
            kind: "stack_map".to_string(),
            data: Data::StackMap {
                pc: function_pointer,
                name: name.unwrap_or("<Anonymous>").to_string(),
                stack_map: translated_stack_map.clone(),
            },
        });
        self.stack_map.extend(translated_stack_map);

        debugger(Message {
            kind: "user_function".to_string(),
            data: Data::UserFunction {
                name: name.unwrap_or("<Anonymous>").to_string(),
                pointer: function_pointer,
                len: bytes.len(),
            },
        });
        Ok(function_pointer)
    }

    pub fn reserve_function(&mut self, name: &str) -> Result<Function, Box<dyn Error>> {
        for function in self.functions.iter_mut() {
            if function.name == name {
                return Ok(function.clone());
            }
        }
        let index = self.functions.len();
        let jump_table_offset = self.add_jump_table_entry(index, 0)?;
        let function = Function {
            name: name.to_string(),
            offset_or_pointer: 0,
            jump_table_offset,
            is_foreign: false,
            is_builtin: false,
            needs_stack_pointer: false,
            is_defined: false,
            number_of_locals: 0,
        };
        self.functions.push(function.clone());
        Ok(function)
    }

    pub fn add_function(
        &mut self,
        name: Option<&str>,
        code: &[u8],
        number_of_locals: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let offset = self.add_code(code)?;
        let index = self.functions.len();
        self.functions.push(Function {
            name: name.unwrap_or("<Anonymous>").to_string(),
            offset_or_pointer: offset,
            jump_table_offset: 0,
            is_foreign: false,
            is_builtin: false,
            needs_stack_pointer: false,
            is_defined: true,
            number_of_locals,
        });
        let function_pointer =
            Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap();
        let jump_table_offset = self.add_jump_table_entry(index, function_pointer)?;

        self.functions[index].jump_table_offset = jump_table_offset;
        if let Some(name) = name {
            self.namespaces.add_binding(name, function_pointer);
        }
        Ok(function_pointer)
    }

    pub fn overwrite_function(
        &mut self,
        index: usize,
        code: &[u8],
    ) -> Result<usize, Box<dyn Error>> {
        let offset = self.add_code(code)?;
        let function = &mut self.functions[index];
        function.offset_or_pointer = offset;
        let jump_table_offset = function.jump_table_offset;
        let function_clone = function.clone();
        let function_pointer = self.get_function_pointer(function_clone).unwrap();
        self.modify_jump_table_entry(jump_table_offset, function_pointer)?;
        let function = &mut self.functions[index];
        function.is_defined = true;
        Ok(function_pointer)
    }

    pub fn get_pointer(&self, function: &Function) -> Result<usize, Box<dyn Error>> {
        if function.is_foreign {
            Ok(function.offset_or_pointer)
        } else {
            Ok(function.offset_or_pointer + self.code_memory.as_ref().unwrap().as_ptr() as usize)
        }
    }

    pub fn get_function_pointer(&self, function: Function) -> Result<usize, Box<dyn Error>> {
        // Gets the absolute pointer to a function
        // if it is a foreign function, return the offset
        // if it is a local function, return the offset + the start of code_memory
        if function.is_foreign {
            Ok(function.offset_or_pointer)
        } else {
            Ok(function.offset_or_pointer + self.code_memory.as_ref().unwrap().as_ptr() as usize)
        }
    }

    pub fn get_jump_table_pointer(&self, function: Function) -> Result<usize, Box<dyn Error>> {
        Ok(function.jump_table_offset * 8 + self.jump_table.as_ref().unwrap().as_ptr() as usize)
    }

    pub fn add_jump_table_entry(
        &mut self,
        _index: usize,
        pointer: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let jump_table_offset = self.jump_table_offset;
        let memory = self.jump_table.take();
        let mut memory = memory.unwrap().make_mut().map_err(|(_, e)| e)?;
        let buffer = &mut memory[jump_table_offset * 8..];
        let pointer = BuiltInTypes::Function.tag(pointer as isize) as usize;
        // Write full usize to buffer
        for (index, byte) in pointer.to_le_bytes().iter().enumerate() {
            buffer[index] = *byte;
        }
        let mem = memory.make_read_only().unwrap_or_else(|(_map, e)| {
            panic!("Failed to make mmap read_only: {}", e);
        });
        self.jump_table_offset += 1;
        self.jump_table = Some(mem);
        Ok(jump_table_offset)
    }

    fn modify_jump_table_entry(
        &mut self,
        jump_table_offset: usize,
        function_pointer: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let memory = self.jump_table.take();
        let mut memory = memory.unwrap().make_mut().map_err(|(_, e)| e)?;
        let buffer = &mut memory[jump_table_offset * 8..];

        let function_pointer = BuiltInTypes::Function.tag(function_pointer as isize) as usize;
        // Write full usize to buffer
        for (index, byte) in function_pointer.to_le_bytes().iter().enumerate() {
            buffer[index] = *byte;
        }
        let mem = memory.make_read_only().unwrap_or_else(|(_map, e)| {
            panic!("Failed to make mmap read_only: {}", e);
        });
        self.jump_table = Some(mem);
        Ok(jump_table_offset)
    }

    pub fn add_code(&mut self, code: &[u8]) -> Result<usize, Box<dyn Error>> {
        let start = self.code_offset;
        let memory = self.code_memory.take();
        let mut memory = memory.unwrap().make_mut().map_err(|(_, e)| e)?;
        let buffer = &mut memory[self.code_offset..];

        for (index, byte) in code.iter().enumerate() {
            buffer[index] = *byte;
        }

        let size: usize = memory.size();
        memory.flush(0..size)?;
        memory.flush_icache()?;
        self.code_offset += code.len();

        let exec = memory.make_exec().unwrap_or_else(|(_map, e)| {
            panic!("Failed to make mmap executable: {}", e);
        });

        self.code_memory = Some(exec);

        Ok(start)
    }

    pub fn get_repr(&self, value: usize, depth: usize) -> Option<String> {
        if depth > 1000 {
            return Some("...".to_string());
        }
        let tag = BuiltInTypes::get_kind(value);
        match tag {
            BuiltInTypes::Null => Some("null".to_string()),
            BuiltInTypes::Int => {
                let value = BuiltInTypes::untag(value);
                Some(value.to_string())
            }
            BuiltInTypes::Float => todo!(),
            BuiltInTypes::String => {
                let value = BuiltInTypes::untag(value);
                let string = &self.string_constants[value];
                Some(string.str.clone())
            }
            BuiltInTypes::Bool => {
                let value = BuiltInTypes::untag(value);
                if value == 0 {
                    Some("false".to_string())
                } else {
                    Some("true".to_string())
                }
            }
            BuiltInTypes::Function => Some("function".to_string()),
            BuiltInTypes::Closure => {
                let value = BuiltInTypes::untag(value);
                unsafe {
                    let pointer = value as *const u8;
                    if pointer as usize % 8 != 0 {
                        panic!("Not aligned");
                    }
                    let function_pointer = *(pointer as *const usize);
                    let num_free = *(pointer.add(8) as *const usize);
                    let num_locals = *(pointer.add(16) as *const usize);
                    let free_variables = std::slice::from_raw_parts(pointer.add(24), num_free * 8);
                    let mut repr = "Closure { ".to_string();
                    repr.push_str(&self.get_repr(function_pointer, depth + 1)?);
                    repr.push_str(", ");
                    repr.push_str(&num_free.to_string());
                    repr.push_str(", ");
                    repr.push_str(&num_locals.to_string());
                    repr.push_str(", [");
                    for i in 0..num_free {
                        let value = &free_variables[i * 8..i * 8 + 8];
                        let mut bytes = [0u8; 8];
                        bytes.copy_from_slice(value);
                        let value = usize::from_le_bytes(bytes);
                        repr.push_str(&self.get_repr(value, depth + 1)?);
                        if i != num_free - 1 {
                            repr.push_str(", ");
                        }
                    }
                    repr.push_str("] }");
                    Some(repr)
                }
            }
            BuiltInTypes::HeapObject => {
                // TODO: Once I change the setup for heap objects
                // I need to figure out what kind of heap object I have
                let object = HeapObject::from_tagged(value);
                // TODO: abstract over this (memory-layout)
                let type_id = BuiltInTypes::untag(object.get_field(0));
                let struct_value = self.structs.get_by_id(type_id as usize);
                Some(self.get_struct_repr(struct_value?, &object.get_fields()[1..], depth + 1)?)
            }
        }
    }

    pub fn compile(&mut self, file_name: &str, code: &str) -> Result<(), Box<dyn Error>> {
        let mut parser = Parser::new(code.to_string());
        let ast = parser.parse();
        let top_level = self.compile_ast(file_name, ast)?;
        assert!(top_level.is_none());
        Ok(())
    }

    pub fn compile_ast(
        &mut self,
        file_name: &str,
        ast: crate::ast::Ast,
    ) -> Result<Option<String>, Box<dyn Error>> {
        let mut ir = ast.compile(self);
        if ast.has_top_level() {
            let arm = LowLevelArm::new();
            let error_fn_pointer = self.find_function("throw_error").unwrap();
            let error_fn_pointer = self.get_function_pointer(error_fn_pointer).unwrap();

            let compiler_ptr = self.get_compiler_ptr() as usize;
            let mut arm = ir.compile(arm, error_fn_pointer, compiler_ptr);
            let max_locals = arm.max_locals as usize;
            let function_name = format!("__{}_top_level", file_name);
            self.upsert_function(Some(&function_name), &mut arm, max_locals)?;
            return Ok(Some(function_name));
        }
        Ok(None)
    }

    pub fn get_trampoline(&self) -> fn(u64, u64) -> u64 {
        let trampoline = self
            .functions
            .iter()
            .find(|f| f.name == "trampoline")
            .unwrap();
        let trampoline_jump_table_offset = trampoline.jump_table_offset;
        let trampoline_offset = &self.jump_table.as_ref().unwrap()
            [trampoline_jump_table_offset * 8..trampoline_jump_table_offset * 8 + 8];

        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(trampoline_offset);
        let trampoline_start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;
        unsafe { std::mem::transmute(trampoline_start) }
    }

    pub fn add_string(&mut self, string_value: StringValue) -> Value {
        self.string_constants.push(string_value);
        let offset = self.string_constants.len() - 1;
        Value::StringConstantPtr(offset)
    }

    pub fn find_function(&self, name: &str) -> Option<Function> {
        self.functions.iter().find(|f| f.name == name).cloned()
    }

    pub fn get_function_by_pointer(&self, value: usize) -> Option<&Function> {
        let offset = value.saturating_sub(self.code_memory.as_ref().unwrap().as_ptr() as usize);
        self.functions
            .iter()
            .find(|f| f.offset_or_pointer == offset)
    }

    pub fn get_function_by_pointer_mut(&mut self, value: usize) -> Option<&mut Function> {
        let offset = value - self.code_memory.as_ref().unwrap().as_ptr() as usize;
        self.functions
            .iter_mut()
            .find(|f| f.offset_or_pointer == offset)
    }

    pub fn property_access(&self, struct_pointer: usize, str_constant_ptr: usize) -> usize {
        unsafe {
            if BuiltInTypes::untag(struct_pointer) % 8 != 0 {
                panic!("Not aligned");
            }
            let struct_pointer = BuiltInTypes::untag(struct_pointer);
            let struct_pointer = struct_pointer as *const u8;
            let size = *(struct_pointer as *const usize) >> 1;
            let str_constant_ptr: usize = BuiltInTypes::untag(str_constant_ptr);
            let string_value = &self.string_constants[str_constant_ptr];
            let string = &string_value.str;
            let struct_pointer = struct_pointer.add(8);
            let data = std::slice::from_raw_parts(struct_pointer, size);
            // type id is the first 8 bytes of data
            let type_id = usize::from_le_bytes(data[0..8].try_into().unwrap());
            let type_id = BuiltInTypes::untag(type_id);
            let struct_value = self.structs.get_by_id(type_id as usize).unwrap();
            let field_index = struct_value
                .fields
                .iter()
                .position(|f| f == string)
                .unwrap();
            let field_index = (field_index + 1) * 8;
            let field = &data[field_index..field_index + 8];
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(field);

            usize::from_le_bytes(bytes)
        }
    }

    pub fn add_struct(&mut self, s: Struct) {
        let name = s.name.clone();
        self.structs.insert(name.clone(), s);
        // TODO: I need a "Struct" object that I can add here
        // What I mean by this is a kind of meta object
        // if you define a struct like this:
        // struct Point { x y }
        // and then you say let x = Point
        // x is a struct object that describes the struct Point
        self.namespaces.add_binding(&name, 0);
    }

    pub fn add_enum(&mut self, e: Enum) {
        let name = e.name.clone();
        self.enums.insert(e);
        // See comment about structs above
        self.namespaces.add_binding(&name, 0);
    }

    pub fn get_enum(&self, name: &str) -> Option<&Enum> {
        self.enums.get(name)
    }

    pub fn get_struct(&self, name: &str) -> Option<(usize, &Struct)> {
        self.structs.get(name)
    }

    fn get_struct_repr(
        &self,
        struct_value: &Struct,
        fields: &[usize],
        depth: usize,
    ) -> Option<String> {
        // It should look like this
        // struct_name { field1: value1, field2: value2 }
        let mut repr = struct_value.name.clone();
        repr.push_str(" { ");
        for (index, field) in struct_value.fields.iter().enumerate() {
            repr.push_str(field);
            repr.push_str(": ");
            let value = fields[index];
            repr.push_str(&self.get_repr(value, depth + 1)?);
            if index != struct_value.fields.len() - 1 {
                repr.push_str(", ");
            }
        }
        repr.push_str(" }");
        Some(repr)
    }

    pub fn check_functions(&self) {
        let undefined_functions: Vec<&Function> =
            self.functions.iter().filter(|f| !f.is_defined).collect();
        if !undefined_functions.is_empty() {
            panic!(
                "Undefined functions: {:?} only have functions {:#?}",
                undefined_functions
                    .iter()
                    .map(|f| f.name.clone())
                    .collect::<Vec<String>>(),
                self.functions
                    .iter()
                    .map(|f| f.name.clone())
                    .collect::<Vec<String>>()
            );
        }
    }

    pub fn get_function_by_name(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }

    pub fn get_function_by_name_mut(&mut self, name: &str) -> Option<&mut Function> {
        self.functions.iter_mut().find(|f| f.name == name)
    }

    pub fn is_inline_primitive_function(&self, name: &str) -> bool {
        match name {
            "primitive_deref"
            | "primitive_swap!"
            | "primitive_reset!"
            | "primitive_compare_and_swap!"
            | "primitive_breakpoint!" => true,
            _ => false,
        }
    }

    pub fn reserve_namespace_slot(&self, name: &str) -> usize {
        self.namespaces
            .add_binding(name, BuiltInTypes::null_value() as usize)
    }

    pub fn current_namespace_id(&self) -> usize {
        self.namespaces.current_namespace
    }

    pub fn update_binding(&mut self, namespace_slot: usize, value: usize) {
        let mut namespace = self.namespaces.get_current_namespace().lock().unwrap();
        let name = namespace.ids.get(namespace_slot).unwrap().clone();
        namespace.bindings.insert(name, value);
    }

    pub fn get_binding(&self, namespace: usize, slot: usize) -> usize {
        let namespace = self.namespaces.namespaces.get(namespace).unwrap();
        let namespace = namespace.lock().unwrap();
        let name = namespace.ids.get(slot).unwrap();
        *namespace.bindings.get(name).unwrap()
    }

    pub fn reserve_namespace(&mut self, name: String) -> usize {
        self.namespaces.add_namespace(name.as_str())
    }

    pub fn set_current_namespace(&mut self, namespace: usize) {
        self.namespaces.current_namespace = namespace;
    }

    pub fn find_binding(&self, current_namespace_id: usize, name: &str) -> Option<usize> {
        let namespace = self
            .namespaces
            .namespaces
            .get(current_namespace_id)
            .unwrap();
        let namespace = namespace.lock().unwrap();
        namespace.ids.iter().position(|n| n == name)
    }

    pub fn global_namespace_id(&self) -> usize {
        0
    }

    pub fn current_namespace_name(&self) -> String {
        self.namespaces
            .namespaces
            .get(self.namespaces.current_namespace)
            .unwrap()
            .lock()
            .unwrap()
            .name
            .clone()
    }
}

impl fmt::Debug for Compiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: Make this better
        f.debug_struct("Compiler")
            .field("code_offset", &self.code_offset)
            .field("jump_table_offset", &self.jump_table_offset)
            .field("functions", &self.functions)
            .finish()
    }
}

impl<Alloc: Allocator> Runtime<Alloc> {
    pub fn new(
        command_line_arguments: CommandLineArguments,
        allocator: Alloc,
        printer: Box<dyn Printer>,
    ) -> Self {
        Self {
            printer,
            command_line_arguments: command_line_arguments.clone(),
            compiler: Compiler {
                code_memory: Some(
                    MmapOptions::new(MmapOptions::page_size())
                        .unwrap()
                        .map()
                        .unwrap(),
                ),
                code_offset: 0,
                jump_table: Some(
                    MmapOptions::new(MmapOptions::page_size())
                        .unwrap()
                        .map()
                        .unwrap(),
                ),
                jump_table_offset: 0,
                structs: StructManager::new(),
                enums: EnumManager::new(),
                namespaces: NamespaceManager::new(),
                functions: Vec::new(),
                string_constants: vec![],
                command_line_arguments: command_line_arguments.clone(),
                compiler_pointer: None,
                stack_map: StackMap::new(),
                pause_atom_ptr: None,
            },
            memory: Memory {
                heap: allocator,
                stacks: vec![(
                    std::thread::current().id(),
                    MmapOptions::new(STACK_SIZE).unwrap().map_mut().unwrap(),
                )],
                join_handles: vec![],
                threads: vec![std::thread::current()],
                command_line_arguments,
                stack_map: StackMap::new(),
            },
            is_paused: AtomicUsize::new(0),
            gc_lock: Mutex::new(()),
            thread_state: Arc::new((
                Mutex::new(ThreadState {
                    paused_threads: 0,
                    stack_pointers: vec![],
                }),
                Condvar::new(),
            )),
        }
    }

    fn get_allocate_options(&self) -> AllocatorOptions {
        AllocatorOptions {
            gc: !self.command_line_arguments.no_gc,
            print_stats: self.command_line_arguments.show_gc_times,
            gc_always: self.command_line_arguments.gc_always,
        }
    }

    pub fn print(&mut self, result: usize) {
        let result = self.compiler.get_repr(result, 0).unwrap();
        self.printer.print(result);
    }

    pub fn println(&mut self, result: usize) {
        let result = self.compiler.get_repr(result, 0).unwrap();
        self.printer.println(result);
    }

    pub fn is_paused(&self) -> bool {
        self.is_paused.load(std::sync::atomic::Ordering::Relaxed) == 1
    }

    pub fn pause_atom_ptr(&self) -> usize {
        self.is_paused.as_ptr() as usize
    }

    pub fn allocate(
        &mut self,
        bytes: usize,
        stack_pointer: usize,
        kind: BuiltInTypes,
    ) -> Result<usize, Box<dyn Error>> {
        // TODO: I need to make it so that I can pass the ability to pause
        // to the allocator. Maybe the allocator should have the pause atom?
        // I need to think about how to abstract all these details out.
        let options = self.get_allocate_options();

        if options.gc_always {
            self.gc(stack_pointer);
        }

        let result = self.memory.heap.allocate(bytes, kind, options);

        match result {
            Ok(AllocateAction::Allocated(value)) => Ok(value as usize),
            Ok(AllocateAction::Gc) => {
                self.gc(stack_pointer);
                let result = self.memory.heap.allocate(bytes, kind, options);
                if let Ok(AllocateAction::Allocated(result)) = result {
                    Ok(result as usize)
                } else {
                    self.memory.heap.grow(options);
                    // TODO: Detect loop here
                    self.allocate(bytes, stack_pointer, kind)
                }
            }
            Err(e) => Err(e),
        }
    }

    pub fn gc(&mut self, stack_pointer: usize) {
        if self.memory.threads.len() == 1 {
            // If there is only one thread, that is us
            // that means nothing else could spin up a thread in the mean time
            // so there is no need to lock anything
            self.memory.heap.gc(
                &self.memory.stack_map,
                &[(self.get_stack_base(), stack_pointer)],
                self.get_allocate_options(),
            );
            return;
        }

        let locked = self.gc_lock.try_lock();

        if locked.is_err() {
            match locked.as_ref().unwrap_err() {
                TryLockError::WouldBlock => {
                    drop(locked);
                    unsafe { __pause(self as *mut Runtime<Alloc>, stack_pointer) };
                }
                TryLockError::Poisoned(e) => {
                    panic!("Poisoned lock {:?}", e);
                }
            }

            return;
        }
        let result = self.is_paused.compare_exchange(
            0,
            1,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        );
        if result != Ok(0) {
            drop(locked);
            unsafe { __pause(self as *mut Runtime<Alloc>, stack_pointer) };
            return;
        }

        let locked = locked.unwrap();

        let (lock, cvar) = &*self.thread_state;
        let mut thread_state = lock.lock().unwrap();
        while thread_state.paused_threads < self.memory.active_threads() {
            thread_state = cvar.wait(thread_state).unwrap();
        }

        let mut stack_pointers = thread_state.stack_pointers.clone();
        stack_pointers.push((self.get_stack_base(), stack_pointer));

        drop(thread_state);

        let options = self.get_allocate_options();
        self.memory
            .heap
            .gc(&self.memory.stack_map, &stack_pointers, options);

        self.is_paused
            .store(0, std::sync::atomic::Ordering::Release);

        self.memory.active_threads();
        for thread in self.memory.threads.iter() {
            thread.unpark();
        }

        let mut thread_state = lock.lock().unwrap();
        while thread_state.paused_threads > 0 {
            let (state, timeout) = cvar
                .wait_timeout(thread_state, Duration::from_millis(1))
                .unwrap();
            thread_state = state;

            if timeout.timed_out() {
                self.memory.active_threads();
                for thread in self.memory.threads.iter() {
                    // println!("Unparking thread {:?}", thread.thread().id());
                    thread.unpark();
                }
            }
        }
        thread_state.clear();

        drop(locked);
    }

    pub fn gc_add_root(&mut self, old: usize, young: usize) {
        if BuiltInTypes::is_heap_pointer(young) {
            self.memory.heap.gc_add_root(old, young);
        }
    }

    pub fn register_parked_thread(&mut self, stack_pointer: usize) {
        self.memory
            .heap
            .register_parked_thread(std::thread::current().id(), stack_pointer);
    }

    pub fn get_stack_base(&self) -> usize {
        let current_thread = std::thread::current().id();
        self.memory
            .stacks
            .iter()
            .find(|(thread_id, _)| *thread_id == current_thread)
            .map(|(_, stack)| stack.as_ptr() as usize + STACK_SIZE)
            .unwrap()
    }

    pub fn make_closure(
        &mut self,
        function: usize,
        free_variables: &[usize],
    ) -> Result<usize, Box<dyn Error>> {
        let len = 8 + 8 + 8 + free_variables.len() * 8;
        // TODO: Stack pointer should be passed in
        let heap_pointer = self.allocate(len / 8, 0, BuiltInTypes::Closure)?;
        let pointer = heap_pointer as *mut u8;
        let num_free = free_variables.len();
        let function_definition = self
            .compiler
            .get_function_by_pointer(BuiltInTypes::untag(function));
        if function_definition.is_none() {
            panic!("Function not found");
        }
        let function_definition = function_definition.unwrap();
        let num_locals = function_definition.number_of_locals;

        let function = function.to_le_bytes();

        let free_variables = free_variables.iter().flat_map(|v| v.to_le_bytes());
        let buffer = unsafe { from_raw_parts_mut(pointer.add(8), len) };
        // write function pointer
        for (index, byte) in function.iter().enumerate() {
            buffer[index] = *byte;
        }
        let num_free = num_free.to_le_bytes();
        // Write number of free variables
        for (index, byte) in num_free.iter().enumerate() {
            buffer[8 + index] = *byte;
        }

        let num_locals = num_locals.to_le_bytes();
        // Write number of locals
        for (index, byte) in num_locals.iter().enumerate() {
            buffer[16 + index] = *byte;
        }
        // write free variables
        for (index, byte) in free_variables.enumerate() {
            buffer[24 + index] = byte;
        }
        Ok(BuiltInTypes::Closure.tag(heap_pointer as isize) as usize)
    }

    // TODO: Make this good
    pub fn run(&self, jump_table_offset: usize) -> Result<u64, Box<dyn Error>> {
        // get offset stored in jump table as a usize
        let offset = &self.compiler.jump_table.as_ref().unwrap()
            [jump_table_offset * 8..jump_table_offset * 8 + 8];
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(offset);
        let start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let trampoline = self
            .compiler
            .functions
            .iter()
            .find(|f| f.name == "trampoline")
            .unwrap();
        let trampoline_jump_table_offset = trampoline.jump_table_offset;
        let trampoline_offset = &self.compiler.jump_table.as_ref().unwrap()
            [trampoline_jump_table_offset * 8..trampoline_jump_table_offset * 8 + 8];

        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(trampoline_offset);
        let trampoline_start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let f: fn(u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline_start) };
        let stack_pointer = self.get_stack_base();
        let result = f(stack_pointer as u64, start as u64);
        Ok(result)
    }

    pub fn run1(&self, jump_table_offset: usize, arg: u64) -> Result<u64, Box<dyn Error>> {
        // get offset stored in jump table as a usize
        let offset = &self.compiler.jump_table.as_ref().unwrap()
            [jump_table_offset * 8..jump_table_offset * 8 + 8];
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(offset);
        let start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let trampoline = self
            .compiler
            .functions
            .iter()
            .find(|f| f.name == "trampoline")
            .unwrap();
        let trampoline_jump_table_offset = trampoline.jump_table_offset;
        let trampoline_offset = &self.compiler.jump_table.as_ref().unwrap()
            [trampoline_jump_table_offset * 8..trampoline_jump_table_offset * 8 + 8];

        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(trampoline_offset);
        let trampoline_start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let f: fn(u64, u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline_start) };
        let arg = BuiltInTypes::Int.tag(arg as isize) as u64;
        let stack_pointer = self.get_stack_base();
        let result = f(stack_pointer as u64, start as u64, arg);
        Ok(result)
    }

    pub fn run2(
        &self,
        _jump_table_offset: usize,
        _arg1: u64,
        _arg2: u64,
    ) -> Result<u64, Box<dyn Error>> {
        unimplemented!("Add trampoline");
    }

    // TODO: Make less ugly
    pub fn run_function(&self, name: &str, vec: Vec<i32>) -> u64 {
        let function = self
            .compiler
            .functions
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("Can't find function named {}", name));
        match vec.len() {
            0 => self.run(function.jump_table_offset).unwrap(),
            1 => self
                .run1(function.jump_table_offset, vec[0] as u64)
                .unwrap(),
            2 => self
                .run2(function.jump_table_offset, vec[0] as u64, vec[1] as u64)
                .unwrap(),
            _ => panic!("Too many arguments"),
        }
    }

    pub fn get_function_base(&self, name: &str) -> Option<(u64, u64, fn(u64, u64) -> u64)> {
        let function = self.compiler.functions.iter().find(|f| f.name == name)?;
        let jump_table_offset = function.jump_table_offset;
        let offset = &self.compiler.jump_table.as_ref().unwrap()
            [jump_table_offset * 8..jump_table_offset * 8 + 8];
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(offset);
        let start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let trampoline = self.compiler.get_trampoline();
        let stack_pointer = self.get_stack_base();

        Some((stack_pointer as u64, start as u64, trampoline))
    }

    pub fn get_function0(&self, name: &str) -> Option<Box<dyn Fn() -> u64>> {
        let (stack_pointer, start, trampoline) = self.get_function_base(name)?;
        Some(Box::new(move || trampoline(stack_pointer, start)))
    }

    #[allow(unused)]
    fn get_function1(&self, name: &str) -> Option<Box<dyn Fn(u64) -> u64>> {
        let (stack_pointer, start, trampoline_start) = self.get_function_base(name)?;
        let f: fn(u64, u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline_start) };
        Some(Box::new(move |arg1| f(stack_pointer, start, arg1)))
    }

    #[allow(unused)]
    fn get_function2(&self, name: &str) -> Option<Box<dyn Fn(u64, u64) -> u64>> {
        let (stack_pointer, start, trampoline_start) = self.get_function_base(name)?;
        let f: fn(u64, u64, u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline_start) };
        Some(Box::new(move |arg1, arg2| {
            f(stack_pointer, start, arg1, arg2)
        }))
    }

    // TODO: Allocate/gc need to change to work with this
    pub fn new_thread(&mut self, f: usize) {
        let trampoline = self.compiler.get_trampoline();
        let trampoline: fn(u64, u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline) };
        let call_fn = self
            .compiler
            .get_function_by_name("beagle.core/__call_fn")
            .unwrap();
        let function_pointer = self.compiler.get_pointer(call_fn).unwrap();

        let new_stack = MmapOptions::new(STACK_SIZE).unwrap().map_mut().unwrap();
        let stack_pointer = new_stack.as_ptr() as usize + STACK_SIZE;
        let thread_state = self.thread_state.clone();
        let thread = thread::spawn(move || {
            let result = trampoline(stack_pointer as u64, function_pointer as u64, f as u64);
            // If we end while another thread is waiting for us to pause
            // we need to notify that waiter so they can see we are dead.
            let (_lock, cvar) = &*thread_state;
            cvar.notify_one();
            result
        });

        self.memory.stacks.push((thread.thread().id(), new_stack));
        self.memory.heap.register_thread(thread.thread().id());
        self.memory.threads.push(thread.thread().clone());
        self.memory.join_handles.push(thread);
    }

    pub fn wait_for_other_threads(&mut self) {
        if self.memory.join_handles.is_empty() {
            return;
        }
        for thread in self.memory.join_handles.drain(..) {
            thread.join().unwrap();
        }
        self.wait_for_other_threads();
    }

    pub fn get_pause_atom(&self) -> usize {
        self.memory.heap.get_pause_pointer()
    }
}
