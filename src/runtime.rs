use core::fmt;
use std::{
    collections::HashMap,
    env,
    error::Error,
    io::Write,
    slice::{self},
    sync::{atomic::AtomicUsize, Arc, Condvar, Mutex, TryLockError},
    thread::{self, JoinHandle, Thread, ThreadId},
    time::Duration,
    vec,
};

use libffi::{low::CodePtr, middle::Cif};
use libloading::Library;
use mmap_rs::{Mmap, MmapMut, MmapOptions};

use crate::{
    arm::LowLevelArm,
    builtins::{__pause, debugger},
    gc::{AllocateAction, Allocator, AllocatorOptions, StackMap, StackMapDetails, STACK_SIZE},
    ir::{StringValue, Value},
    parser::Parser,
    types::{BuiltInTypes, HeapObject},
    CommandLineArguments, Data, Message,
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
    size: usize,
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

pub struct FFIInfo {
    pub function: CodePtr,
    pub cif: Cif,
    pub number_of_arguments: usize,
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
    aliases: HashMap<String, usize>,
}

pub struct Runtime<Alloc: Allocator> {
    pub compiler: Compiler,
    pub memory: Memory<Alloc>,
    pub libraries: Vec<Library>,
    #[allow(unused)]
    command_line_arguments: CommandLineArguments,
    pub printer: Box<dyn Printer>,
    // TODO: I don't have any code that looks at u8, just always u64
    // so that's why I need usize
    pub is_paused: AtomicUsize,
    pub thread_state: Arc<(Mutex<ThreadState>, Condvar)>,
    pub gc_lock: Mutex<()>,
    pub ffi_function_info: Vec<FFIInfo>,
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
            self.heap.remove_thread(thread_id);
        }

        self.join_handles.len()
    }
}

impl<Alloc: Allocator> Allocator for Memory<Alloc> {
    fn new(_options: AllocatorOptions) -> Self {
        unimplemented!("Not going to use this");
    }

    fn try_allocate(
        &mut self,
        bytes: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        self.heap.try_allocate(bytes, kind)
    }

    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]) {
        self.heap.gc(stack_map, stack_pointers);
    }

    fn grow(&mut self) {
        self.heap.grow()
    }

    fn gc_add_root(&mut self, old: usize) {
        self.heap.gc_add_root(old)
    }

    fn add_namespace_root(&mut self, namespace_id: usize, root: usize) {
        self.heap.add_namespace_root(namespace_id, root)
    }

    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)> {
        self.heap.get_namespace_relocations()
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.heap.get_allocation_options()
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
    namespace_names: HashMap<String, usize>,
    id_to_name: HashMap<usize, String>,
    current_namespace: usize,
}

impl NamespaceManager {
    fn new() -> Self {
        let mut s = Self {
            namespaces: vec![Mutex::new(Namespace {
                name: "global".to_string(),
                ids: vec![],
                bindings: HashMap::new(),
                aliases: HashMap::new(),
            })],
            namespace_names: HashMap::new(),
            id_to_name: HashMap::new(),
            current_namespace: 0,
        };
        s.add_namespace("beagle.primitive");
        s.add_namespace("beagle.builtin");
        s
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
        let id = self.namespaces.len() - 1;
        self.namespace_names.insert(name.to_string(), id);
        self.id_to_name.insert(id, name.to_string());
        self.namespaces.len() - 1
    }

    #[allow(unused)]
    fn get_namespace(&self, name: &str) -> Option<&Mutex<Namespace>> {
        let position = self.namespace_names.get(name)?;
        self.namespaces.get(*position)
    }

    fn get_namespace_id(&self, name: &str) -> Option<usize> {
        self.namespace_names.get(name).cloned()
    }

    fn get_current_namespace(&self) -> &Mutex<Namespace> {
        self.namespaces.get(self.current_namespace).unwrap()
    }

    fn set_current_namespace(&mut self, id: usize) {
        self.current_namespace = id;
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

    fn get_namespace_name(&self, id: usize) -> Option<String> {
        self.id_to_name.get(&id).cloned()
    }
}

pub struct Compiler {
    code_offset: usize,
    code_memory: Option<Mmap>,
    // TODO: Think about the jump table more
    jump_table: Option<Mmap>,
    // Do I need this offset?
    jump_table_offset: usize,

    property_look_up_cache: MmapMut,

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
    property_look_up_cache_offset: usize,
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

    pub fn allocate_fn_pointer(&mut self) -> usize {
        let allocate_fn_pointer = self.find_function("beagle.builtin/allocate").unwrap();
        self.get_function_pointer(allocate_fn_pointer).unwrap()
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
        let pointer = function as usize;
        let jump_table_offset = self.add_jump_table_entry(index, pointer)?;
        self.functions.push(Function {
            name: name.unwrap_or("<Anonymous>").to_string(),
            offset_or_pointer: pointer,
            jump_table_offset,
            is_foreign: true,
            is_builtin: false,
            needs_stack_pointer: false,
            is_defined: true,
            number_of_locals: 0,
            size: 0,
        });
        debugger(Message {
            kind: "foreign_function".to_string(),
            data: Data::ForeignFunction {
                name: name.unwrap_or("<Anonymous>").to_string(),
                pointer: Self::get_function_pointer(self, self.functions.last().unwrap().clone())
                    .unwrap(),
            },
        });
        let function_pointer =
            Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap();
        Ok(function_pointer)
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
            size: 0,
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

        self.update_stack_map_information(code, function_pointer, name);

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

    fn update_stack_map_information(
        &mut self,
        code: &mut LowLevelArm,
        function_pointer: usize,
        name: Option<&str>,
    ) {
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
            size: 0,
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
            size: code.len(),
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
        function.size = code.len();
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
        if depth > 10 {
            return Some("...".to_string());
        }
        let tag = BuiltInTypes::get_kind(value);
        match tag {
            BuiltInTypes::Null => Some("null".to_string()),
            BuiltInTypes::Int => Some(BuiltInTypes::untag_isize(value as isize).to_string()),
            BuiltInTypes::Float => {
                let value = BuiltInTypes::untag(value);
                let value = value as *const f64;
                let value = unsafe { *value.add(1) };
                Some(value.to_string())
            }
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
                let heap_object = HeapObject::from_tagged(value);
                let function_pointer = heap_object.get_field(0);
                let num_free = heap_object.get_field(1);
                let num_locals = heap_object.get_field(2);
                let free_variables = heap_object.get_fields()[3..].to_vec();
                let mut repr = "Closure { ".to_string();
                repr.push_str(&self.get_repr(function_pointer, depth + 1)?);
                repr.push_str(", ");
                repr.push_str(&num_free.to_string());
                repr.push_str(", ");
                repr.push_str(&num_locals.to_string());
                repr.push_str(", [");
                for value in free_variables {
                    repr.push_str(&self.get_repr(value, depth + 1)?);
                    repr.push_str(", ");
                }
                repr.push_str("] }");
                Some(repr)
            }
            BuiltInTypes::HeapObject => {
                // TODO: Once I change the setup for heap objects
                // I need to figure out what kind of heap object I have
                let object = HeapObject::from_tagged(value);
                let header = object.get_header();
                if header.type_id != 0 {
                    // This isn't a struct and right now
                    // we don't have a print system for other types
                    // so we are just going to print it like an array
                    let fields = object.get_fields();
                    let mut repr = "[ ".to_string();
                    for (index, field) in fields.iter().enumerate() {
                        repr.push_str(&self.get_repr(*field, depth + 1)?);
                        if index != fields.len() - 1 {
                            repr.push_str(", ");
                        }
                    }
                    repr.push_str(" ]");
                    return Some(repr);
                }
                // TODO: abstract over this (memory-layout)
                let struct_id = BuiltInTypes::untag(object.get_struct_id());
                let struct_value = self.structs.get_by_id(struct_id);
                Some(self.get_struct_repr(struct_value?, object.get_fields(), depth + 1)?)
            }
        }
    }

    // TODO: I'm going to need to change how this works at some point.
    // I want to be able to dynamically run these namespaces
    // not have this awkward compile returns top level names thing
    pub fn compile(&mut self, file_name: &str) -> Result<Vec<String>, Box<dyn Error>> {
        if self.command_line_arguments.verbose {
            println!("Compiling {:?}", file_name);
        }
        let parse_time = std::time::Instant::now();
        let code = std::fs::read_to_string(file_name)
            .unwrap_or_else(|_| panic!("Could not find file {:?}", file_name));
        let mut parser = Parser::new(file_name.to_string(), code.to_string());
        let ast = parser.parse();

        if self.command_line_arguments.print_ast {
            println!("{:#?}", ast);
        }

        if self.command_line_arguments.show_times {
            println!("Parse time: {:?}", parse_time.elapsed());
        }

        let mut top_levels_to_run = self.compile_dependencies(&ast)?;

        let top_level = self.compile_ast(ast)?;
        if let Some(top_level) = top_level {
            top_levels_to_run.push(top_level);
        }

        if self.command_line_arguments.verbose {
            println!("Done compiling {:?}", file_name);
        }
        Ok(top_levels_to_run)
    }

    fn compile_dependencies(
        &mut self,
        ast: &crate::ast::Ast,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let mut top_levels_to_run = vec![];
        for import in ast.imports() {
            let (name, _alias) = self.extract_import(&import);
            if name == "beagle.core" || name == "beagle.primitive" || name == "beagle.builtin" {
                continue;
            }
            let file_name = self.get_file_name_from_import(name);
            let top_level = self.compile(&file_name)?;
            top_levels_to_run.extend(top_level);
        }
        Ok(top_levels_to_run)
    }

    fn compile_ast(&mut self, ast: crate::ast::Ast) -> Result<Option<String>, Box<dyn Error>> {
        let mut ir = ast.compile(self);
        if ast.has_top_level() {
            let arm = LowLevelArm::new();
            let error_fn_pointer = self.find_function("beagle.builtin/throw_error").unwrap();
            let error_fn_pointer = self.get_function_pointer(error_fn_pointer).unwrap();

            let compiler_ptr = self.get_compiler_ptr() as usize;
            let mut arm = ir.compile(arm, error_fn_pointer, compiler_ptr);
            let max_locals = arm.max_locals as usize;
            let function_name = format!("{}/__top_level", self.current_namespace_name());
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

    pub fn find_function(&self, name: &str) -> Option<Function> {
        assert!(
            name.contains("/"),
            "Function name should contain /: {:?}",
            name
        );
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

    pub fn property_access(
        &self,
        struct_pointer: usize,
        str_constant_ptr: usize,
    ) -> (usize, usize) {
        if BuiltInTypes::untag(struct_pointer) % 8 != 0 {
            panic!("Not aligned");
        }
        let heap_object = HeapObject::from_tagged(struct_pointer);
        let str_constant_ptr: usize = BuiltInTypes::untag(str_constant_ptr);
        let string_value = &self.string_constants[str_constant_ptr];
        let string = &string_value.str;
        let struct_type_id = heap_object.get_struct_id();
        let struct_type_id = BuiltInTypes::untag(struct_type_id);
        let struct_value = self.structs.get_by_id(struct_type_id).unwrap();
        let field_index = struct_value
            .fields
            .iter()
            .position(|f| f == string)
            .unwrap();
        // Temporary +1 because I was writing size as the first field
        // and I haven't changed that
        (heap_object.get_field(field_index), field_index)
    }

    pub fn add_struct(&mut self, s: Struct) {
        let name = s.name.clone();
        // TODO: Namespace these
        self.structs.insert(name.clone(), s);
        // TODO: I need a "Struct" object that I can add here
        // What I mean by this is a kind of meta object
        // if you define a struct like this:
        // struct Point { x y }
        // and then you say let x = Point
        // x is a struct object that describes the struct Point

        // grab the simple name for the binding
        let (_, simple_name) = name.split_once("/").unwrap();
        self.namespaces.add_binding(simple_name, 0);
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
        let simple_name = struct_value.name.split_once("/").unwrap().1;
        let mut repr = simple_name.to_string();
        if struct_value.fields.is_empty() {
            return Some(repr);
        }
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
        name.starts_with("beagle.primitive/")
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
        self.namespaces.set_current_namespace(namespace);
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

    fn get_current_namespace(&self) -> &Mutex<Namespace> {
        self.namespaces
            .namespaces
            .get(self.namespaces.current_namespace)
            .unwrap()
    }

    pub fn add_alias(&self, namespace_name: String, alias: String) {
        // TODO: I really need to get rid of this mutex business
        let namespace_id = self
            .namespaces
            .get_namespace_id(namespace_name.as_str())
            .unwrap_or_else(|| panic!("Could not find namespace {}", namespace_name));

        let current_namespace = self.get_current_namespace();
        let mut namespace = current_namespace.lock().unwrap();
        namespace.aliases.insert(alias, namespace_id);
    }

    pub fn get_namespace_id(&self, name: &str) -> Option<usize> {
        self.namespaces.get_namespace_id(name)
    }

    /// TODO: Temporary please fix
    fn get_file_name_from_import(&self, import_name: String) -> String {
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

    pub fn get_namespace_from_alias(&self, alias: &str) -> Option<String> {
        let current_namespace = self.get_current_namespace();
        let namespace = current_namespace.lock().unwrap();
        let id = namespace.aliases.get(alias)?;
        let namespace_name = self.namespaces.get_namespace_name(*id)?;
        Some(namespace_name)
    }

    pub fn get_string(&self, value: usize) -> String {
        let value = BuiltInTypes::untag(value);
        self.string_constants[value].str.clone()
    }

    fn extract_import(&self, import: &crate::ast::Ast) -> (String, String) {
        match import {
            crate::ast::Ast::Import {
                library_name,
                alias,
            } => {
                let library_name = library_name.to_string().replace("\"", "");
                let alias = alias.as_ref().get_string();
                (library_name, alias)
            }
            _ => panic!("Not an import"),
        }
    }

    pub fn get_struct_by_id(&self, struct_id: usize) -> Option<&Struct> {
        self.structs.get_by_id(struct_id)
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

#[allow(unused)]
#[derive(Debug)]
pub struct StackValue {
    function: Option<String>,
    details: StackMapDetails,
    stack: Vec<usize>,
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
                    MmapOptions::new(MmapOptions::page_size() * 10)
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
                property_look_up_cache: MmapOptions::new(MmapOptions::page_size())
                    .unwrap()
                    .map_mut()
                    .unwrap()
                    .make_mut()
                    .unwrap_or_else(|(_map, e)| {
                        panic!("Failed to make mmap executable: {}", e);
                    }),
                property_look_up_cache_offset: 0,
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
            libraries: vec![],
            is_paused: AtomicUsize::new(0),
            gc_lock: Mutex::new(()),
            thread_state: Arc::new((
                Mutex::new(ThreadState {
                    paused_threads: 0,
                    stack_pointers: vec![],
                }),
                Condvar::new(),
            )),
            ffi_function_info: vec![],
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
        let options = self.memory.heap.get_allocation_options();

        if options.gc_always {
            self.gc(stack_pointer);
        }

        let result = self.memory.heap.try_allocate(bytes, kind);

        match result {
            Ok(AllocateAction::Allocated(value)) => {
                assert!(value.is_aligned());
                let value = kind.tag(value as isize);
                Ok(value as usize)
            }
            Ok(AllocateAction::Gc) => {
                self.gc(stack_pointer);
                let result = self.memory.heap.try_allocate(bytes, kind);
                if let Ok(AllocateAction::Allocated(result)) = result {
                    // tag
                    assert!(result.is_aligned());
                    let result = kind.tag(result as isize);
                    Ok(result as usize)
                } else {
                    self.memory.heap.grow();
                    // TODO: Detect loop here
                    let pointer = self.allocate(bytes, stack_pointer, kind)?;
                    // If we went down this path, our pointer is already tagged
                    Ok(pointer)
                }
            }
            Err(e) => Err(e),
        }
    }

    unsafe fn buffer_between<T>(start: *const T, end: *const T) -> &'static [T] {
        let len = end.offset_from(start);
        slice::from_raw_parts(start, len as usize)
    }

    fn find_function_for_pc(&self, pc: usize) -> Option<&Function> {
        self.compiler.functions.iter().find(|f| {
            let start =
                f.offset_or_pointer + self.compiler.code_memory.as_ref().unwrap().as_ptr() as usize;
            let end = start + f.size;
            pc >= start && pc < end
        })
    }

    #[allow(unused)]
    pub fn parse_stack_frames(&self, stack_pointer: usize) -> Vec<StackValue> {
        let stack_base = self.get_stack_base();
        let stack = unsafe {
            Self::buffer_between(
                (stack_pointer as *const usize).sub(1),
                stack_base as *const usize,
            )
        };
        println!("Stack: {:?}", stack);
        let mut stack_frames = vec![];
        let mut i = 0;
        while i < stack.len() {
            let value = stack[i];
            if let Some(details) = self.memory.stack_map.find_stack_data(value) {
                let mut frame_size = details.max_stack_size + details.number_of_locals;
                if frame_size % 2 != 0 {
                    frame_size += 1
                }
                let current_frame = &stack[i..i + frame_size + 1];
                let stack_value = StackValue {
                    function: self.find_function_for_pc(value).map(|f| f.name.clone()),
                    details: details.clone(),
                    stack: current_frame.to_vec(),
                };
                println!("Stack value: {:#?}", stack_value);
                stack_frames.push(stack_value);
                i += details.current_stack_size + details.number_of_locals + 1;
                continue;
            }
            i += 1;
        }
        stack_frames
    }

    pub fn gc(&mut self, stack_pointer: usize) {
        if self.memory.threads.len() == 1 {
            // If there is only one thread, that is us
            // that means nothing else could spin up a thread in the mean time
            // so there is no need to lock anything
            self.memory.heap.gc(
                &self.memory.stack_map,
                &[(self.get_stack_base(), stack_pointer)],
            );

            // TODO: This whole thing is awful.
            // I should be passing around the slot so I can just update the binding directly.
            let relocations = self.memory.get_namespace_relocations();
            for (namespace, values) in relocations {
                for (old, new) in values {
                    for (_, value) in self
                        .compiler
                        .namespaces
                        .namespaces
                        .get_mut(namespace)
                        .unwrap()
                        .get_mut()
                        .unwrap()
                        .bindings
                        .iter_mut()
                    {
                        if *value == old {
                            *value = new;
                        }
                    }
                }
            }
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

        self.memory.heap.gc(&self.memory.stack_map, &stack_pointers);

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

    pub fn gc_add_root(&mut self, old: usize) {
        if BuiltInTypes::is_heap_pointer(old) {
            self.memory.heap.gc_add_root(old);
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
        stack_pointer: usize,
        function: usize,
        free_variables: &[usize],
    ) -> Result<usize, Box<dyn Error>> {
        let len = 8 + 8 + 8 + free_variables.len() * 8;
        let heap_pointer = self.allocate(len / 8, stack_pointer, BuiltInTypes::Closure)?;
        let heap_object = HeapObject::from_tagged(heap_pointer);
        let num_free = free_variables.len();
        let function_definition = self
            .compiler
            .get_function_by_pointer(BuiltInTypes::untag(function));
        if function_definition.is_none() {
            panic!("Function not found");
        }
        let function_definition = function_definition.unwrap();
        let num_locals = function_definition.number_of_locals;

        heap_object.write_field(0, function);
        heap_object.write_field(1, num_free);
        heap_object.write_field(2, num_locals);
        for (index, value) in free_variables.iter().enumerate() {
            heap_object.write_field((index + 3) as i32, *value);
        }
        Ok(heap_pointer)
    }

    #[allow(clippy::type_complexity)]
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

    pub fn add_library(&mut self, lib: libloading::Library) -> usize {
        self.libraries.push(lib);
        self.libraries.len() - 1
    }

    pub fn copy_object(
        &mut self,
        from_object: HeapObject,
        to_object: &mut HeapObject,
    ) -> Result<usize, Box<dyn Error>> {
        from_object.copy_full_object(to_object);
        Ok(to_object.tagged_pointer())
    }

    pub fn copy_object_except_header(
        &mut self,
        from_object: HeapObject,
        to_object: &mut HeapObject,
    ) -> Result<usize, Box<dyn Error>> {
        from_object.copy_object_except_header(to_object);
        Ok(to_object.tagged_pointer())
    }

    pub fn write_functions_to_pid_map(&self) {
        // https://github.com/torvalds/linux/blob/6485cf5ea253d40d507cd71253c9568c5470cd27/tools/perf/Documentation/jit-interface.txt
        let pid = std::process::id();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(format!("/tmp/perf-{}.map", pid))
            .unwrap();
        // Each line has the following format, fields separated with spaces:
        // START SIZE symbolname
        // START and SIZE are hex numbers without 0x.
        // symbolname is the rest of the line, so it could contain special characters.

        let memory_offset = self.compiler.code_memory.as_ref().unwrap().as_ptr() as usize;
        for function in self.compiler.functions.iter() {
            if function.is_foreign || function.is_builtin {
                continue;
            }
            let start = memory_offset + function.offset_or_pointer;
            let size = function.size;
            let name = function.name.clone();
            let line = format!("{:x} {:x} {}\n", start, size, name);
            file.write_all(line.as_bytes()).unwrap();
        }
    }

    pub fn get_library(&self, library_id: usize) -> &libloading::Library {
        let library_object = HeapObject::from_tagged(library_id);
        let library_id = BuiltInTypes::untag(library_object.get_field(0));
        &self.libraries[library_id]
    }

    pub fn add_ffi_function_info(
        &mut self,
        ffi_function_info: FFIInfo,
    ) -> usize {
        self.ffi_function_info.push(ffi_function_info);
        self.ffi_function_info.len() - 1
    }

    pub fn get_ffi_info(&self, ffi_info_id: usize) -> &FFIInfo {
        self.ffi_function_info.get(ffi_info_id).unwrap()
    }
}


