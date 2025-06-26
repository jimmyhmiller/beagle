use libffi::middle::Cif;
use libloading::Library;
use mmap_rs::{Mmap, MmapMut, MmapOptions};
use std::{
    cell::UnsafeCell,
    collections::HashMap,
    error::Error,
    ffi::{CString, c_void},
    io::Write,
    slice::{self},
    sync::{
        Arc, Condvar, Mutex, TryLockError,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle, Thread, ThreadId},
    time::Duration,
    vec,
};

use crate::{
    Alloc, CommandLineArguments, Data, Message,
    builtins::{__pause, debugger},
    compiler::{
        BlockingSender, CompilerMessage, CompilerResponse, CompilerThread, blocking_channel,
    },
    gc::{
        AllocateAction, Allocator, AllocatorOptions, STACK_SIZE, StackMap, StackMapDetails,
        stack_segments::StackSegmentAllocator,
    },
    ir::StringValue,
    types::{BuiltInTypes, Header, HeapObject, Tagged},
};

use std::cell::RefCell;

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

pub struct StructManager {
    name_to_id: HashMap<String, usize>,
    structs: Vec<Struct>,
}

impl Default for StructManager {
    fn default() -> Self {
        Self::new()
    }
}

impl StructManager {
    pub fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            structs: Vec::new(),
        }
    }

    pub fn insert(&mut self, name: String, s: Struct) {
        let id = self.structs.len();
        self.name_to_id.insert(name.clone(), id);
        self.structs.push(s);
    }

    pub fn get(&self, name: &str) -> Option<(usize, &Struct)> {
        let id = self.name_to_id.get(name)?;
        self.structs.get(*id).map(|x| (*id, x))
    }

    pub fn get_by_id(&self, type_id: usize) -> Option<&Struct> {
        self.structs.get(type_id)
    }
}

pub trait Printer: Send + Sync {
    fn print(&mut self, value: String);
    fn println(&mut self, value: String);
    // Gross just for testing. I'll need to do better;
    fn get_output(&self) -> Vec<String>;
    fn reset(&mut self);
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

    fn reset(&mut self) {
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

    fn reset(&mut self) {
        self.output.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FFIType {
    U8,
    U32,
    I32,
    Pointer,
    MutablePointer,
    String,
    Void,
    U16,
    U64,
    Structure(Vec<FFIType>),
}

#[derive(Debug, Clone)]
pub struct FFIInfo {
    pub name: String,
    pub function: RawPtr<u8>,
    pub cif: SyncWrapper<Cif>,
    pub number_of_arguments: usize,
    pub argument_types: Vec<FFIType>,
    pub return_type: FFIType,
}

pub struct ThreadState {
    pub paused_threads: usize,
    pub stack_pointers: Vec<(usize, usize)>,
    // TODO: I probably don't want to do this here. This requires taking a mutex
    // not really ideal for c calls.
    pub c_calling_stack_pointers: HashMap<ThreadId, (usize, usize)>,
}

impl ThreadState {
    pub fn pause(&mut self, stack_pointer: (usize, usize)) {
        self.paused_threads += 1;
        self.stack_pointers.push(stack_pointer);
    }

    pub fn unpause(&mut self) {
        self.paused_threads -= 1;
    }

    pub fn register_c_call(&mut self, stack_pointer: (usize, usize)) {
        let thread_id = thread::current().id();
        self.c_calling_stack_pointers
            .insert(thread_id, stack_pointer);
    }

    pub fn unregister_c_call(&mut self) {
        let thread_id = thread::current().id();
        self.c_calling_stack_pointers.remove(&thread_id);
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RawPtr<T> {
    pub ptr: *const T,
}

unsafe impl<T> Sync for RawPtr<T> {}
unsafe impl<T> Send for RawPtr<T> {}

impl<T> RawPtr<T> {
    pub fn new(ptr: *const T) -> Self {
        Self { ptr }
    }

    pub fn get(&self) -> *const T {
        self.ptr
    }
}

impl<T> From<RawPtr<T>> for usize {
    fn from(raw_ptr: RawPtr<T>) -> Self {
        raw_ptr.ptr as usize
    }
}

impl<T> From<RawPtr<T>> for u64 {
    fn from(raw_ptr: RawPtr<T>) -> Self {
        raw_ptr.ptr as u64
    }
}

impl<T, R> From<*const R> for RawPtr<T> {
    fn from(ptr: *const R) -> Self {
        Self {
            ptr: ptr as *const T,
        }
    }
}

impl<T, R> From<RawPtr<T>> for *const R {
    fn from(ptr: RawPtr<T>) -> Self {
        ptr.ptr as *const R
    }
}

#[derive(Debug)]
pub struct SyncWrapper<T> {
    value: UnsafeCell<T>,
}

impl<T> Clone for SyncWrapper<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self::new(self.get().clone())
    }
}

impl<T> SyncWrapper<T> {
    /// Create a new `SyncWrapper`.
    pub const fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
        }
    }

    /// Get a reference to the wrapped value.
    /// Safety: You must ensure proper synchronization when accessing this value.
    pub fn get(&self) -> &T {
        unsafe { &*self.value.get() }
    }

    #[allow(clippy::mut_from_ref)]
    /// Get a mutable reference to the wrapped value.
    /// Safety: You must ensure exclusive access when mutating this value.
    pub fn get_mut(&self) -> &mut T {
        unsafe { &mut *self.value.get() }
    }
}

/// Unsafe implementation of `Sync` because `UnsafeCell` is inherently not `Sync`.
unsafe impl<T> Sync for SyncWrapper<T> {}
unsafe impl<T> Send for SyncWrapper<T> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub pointer: RawPtr<u8>,
    pub jump_table_offset: usize,
    pub is_foreign: bool,
    pub is_builtin: bool,
    pub needs_stack_pointer: bool,
    pub is_defined: bool,
    pub number_of_locals: usize,
    pub size: usize,
    pub number_of_args: usize,
}

pub struct MMapMutWithOffset {
    mmap: MmapMut,
    offset: usize,
}

impl MMapMutWithOffset {
    fn new() -> Self {
        Self {
            mmap: MmapOptions::new(MmapOptions::page_size() * 10)
                .unwrap()
                .map_mut()
                .unwrap(),
            offset: 0,
        }
    }

    pub fn write_c_string(&mut self, string: String) -> *mut i8 {
        let string = string.clone();
        let start = self.offset;
        let c_string = CString::new(string).unwrap();
        let bytes = c_string.as_bytes_with_nul();
        for byte in bytes {
            self.mmap[self.offset] = *byte;
            self.offset += 1;
        }
        unsafe { self.mmap.as_ptr().add(start) as *mut i8 }
    }

    pub fn write_u16(&mut self, value: u16) -> &u16 {
        let start = self.offset;
        let bytes = value.to_le_bytes();
        for byte in bytes {
            self.mmap[self.offset] = byte;
            self.offset += 1;
        }
        unsafe { &*(self.mmap.as_ptr().add(start) as *const u16) }
    }

    pub fn write_u32(&mut self, value: u32) -> *const u32 {
        let start = self.offset;
        let bytes = value.to_le_bytes();
        for byte in bytes {
            self.mmap[self.offset] = byte;
            self.offset += 1;
        }
        unsafe { self.mmap.as_ptr().add(start) as *const u32 }
    }

    pub fn write_u64(&mut self, value: u64) -> *const u64 {
        let start = self.offset;
        let bytes = value.to_le_bytes();
        for byte in bytes {
            self.mmap[self.offset] = byte;
            self.offset += 1;
        }
        unsafe { self.mmap.as_ptr().add(start) as *const u64 }
    }

    pub fn write_i32(&mut self, value: i32) -> *const i32 {
        let start = self.offset;
        let bytes = value.to_le_bytes();
        for byte in bytes {
            self.mmap[self.offset] = byte;
            self.offset += 1;
        }

        (unsafe { self.mmap.as_ptr().add(start) as *const i32 }) as _
    }

    pub fn write_u8(&mut self, argument: u8) -> *const u8 {
        let start = self.offset;
        self.mmap[start] = argument;
        // We need to make sure we keep correct alignment
        self.offset += 2;
        unsafe { self.mmap.as_ptr().add(start) }
    }

    pub fn write_pointer(&mut self, value: usize) -> *const *mut c_void {
        let start = self.offset;
        let bytes = value.to_le_bytes();
        for byte in bytes {
            self.mmap[self.offset] = byte;
            self.offset += 1;
        }
        unsafe { self.mmap.as_ptr().add(start) as *const *mut c_void }
    }

    pub fn write_buffer(&mut self, ptr: usize, size: usize) -> *const u8 {
        // we are going to get the buffer located at ptr with a size of size
        // and copy it into our mmap
        let start = self.offset;
        let buffer = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
        for byte in buffer {
            self.mmap[self.offset] = *byte;
            self.offset += 1;
        }
        unsafe { self.mmap.as_ptr().add(start) }
    }

    pub fn clear(&mut self) {
        self.offset = 0;
    }
}

thread_local! {
    pub static NATIVE_ARGUMENTS : RefCell<MMapMutWithOffset> = RefCell::new(MMapMutWithOffset::new());
}

pub struct Memory {
    heap: Alloc,
    stacks: Vec<(ThreadId, MmapMut)>,
    pub join_handles: Vec<JoinHandle<u64>>,
    pub threads: Vec<Thread>,
    pub stack_map: StackMap,
    #[allow(unused)]
    command_line_arguments: CommandLineArguments,
}

impl Memory {
    fn reset(&mut self) {
        let options = self.heap.get_allocation_options();
        self.heap = Alloc::new(options);
        self.stacks = vec![(
            std::thread::current().id(),
            create_stack_with_protected_page_after(STACK_SIZE),
        )];
        self.join_handles = vec![];
        self.threads = vec![std::thread::current()];
        self.stack_map = StackMap::new();
    }

    fn active_threads(&mut self) -> usize {
        let mut completed_threads = vec![];
        for (index, thread) in self.join_handles.iter().enumerate() {
            if thread.is_finished() {
                completed_threads.push(index);
            }
        }
        for index in completed_threads.iter().rev() {
            if let Some(thread) = self.join_handles.get(*index) {
                let thread_id = thread.thread().id();
                self.stacks.retain(|(id, _)| *id != thread_id);
                self.threads.retain(|t| t.id() != thread_id);
                self.join_handles.remove(*index);
                self.heap.remove_thread(thread_id);
            } else {
                println!("Inconsistent join_handles state {:?}", self.join_handles);
            }
        }

        self.join_handles.len()
    }

    fn allocate_string(&mut self, bytes: &[u8], pointer: usize) -> Result<Tagged, Box<dyn Error>> {
        let mut heap_object = HeapObject::from_tagged(pointer);
        let words = bytes.len() / 8 + 1;
        heap_object.writer_header_direct(Header {
            type_id: 2,
            type_data: bytes.len() as u32,
            size: words as u8,
            opaque: true,
            marked: false,
        });
        heap_object.write_fields(bytes);
        Ok(BuiltInTypes::HeapObject.tagged(pointer))
    }

    pub fn write_c_string(&mut self, string: String) -> *mut i8 {
        let mut result: *mut i8 = std::ptr::null_mut();
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_c_string(string));
        result
    }

    pub fn write_pointer(&mut self, value: usize) -> &*mut c_void {
        let mut result: *const *mut c_void = std::ptr::null_mut();
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_pointer(value));
        unsafe { &*result }
    }

    pub fn write_u32(&mut self, value: u32) -> &u32 {
        let mut result: *const u32 = &0;
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_u32(value));
        unsafe { &*result }
    }

    pub fn write_u64(&mut self, value: u64) -> &u64 {
        let mut result: *const u64 = &0;
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_u64(value));
        unsafe { &*result }
    }

    pub fn write_i32(&mut self, value: i32) -> &i32 {
        let mut result: *const i32 = &0;
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_i32(value));
        unsafe { &*result }
    }

    pub fn write_u8(&mut self, value: u8) -> &u8 {
        let mut result: *const u8 = &0;
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_u8(value));
        unsafe { &*result }
    }

    pub fn write_u16(&mut self, value: u16) -> &u16 {
        let mut result: *const u16 = &0;
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_u16(value));
        unsafe { &*result }
    }

    pub fn write_buffer(&mut self, ptr: usize, size: usize) -> &u8 {
        let mut result: *const u8 = &0;
        NATIVE_ARGUMENTS.with(|memory| result = memory.borrow_mut().write_buffer(ptr, size));
        unsafe { &*result }
    }

    pub fn clear_native_arguments(&self) {
        NATIVE_ARGUMENTS.with(|memory| memory.borrow_mut().clear());
    }
}

impl Allocator for Memory {
    fn new(_options: AllocatorOptions) -> Self {
        unimplemented!("Not going to use this");
    }

    fn try_allocate(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        self.heap.try_allocate(words, kind)
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

    fn register_temporary_root(&mut self, root: usize) -> usize {
        self.heap.register_temporary_root(root)
    }

    fn unregister_temporary_root(&mut self, id: usize) -> usize {
        self.heap.unregister_temporary_root(id)
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

pub struct EnumManager {
    name_to_id: HashMap<String, usize>,
    enums: Vec<Enum>,
}

impl Default for EnumManager {
    fn default() -> Self {
        Self::new()
    }
}

impl EnumManager {
    pub fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            enums: Vec::new(),
        }
    }

    pub fn insert(&mut self, e: Enum) {
        let id = self.enums.len();
        self.name_to_id.insert(e.name.clone(), id);
        self.enums.push(e);
    }

    pub fn get(&self, name: &str) -> Option<&Enum> {
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
        s.add_namespace("beagle.__internal_test__");
        s.add_namespace("beagle.debug");
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

    fn get_namespace_by_id(&self, id: usize) -> Option<&Mutex<Namespace>> {
        self.namespaces.get(id)
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

    #[allow(unused)]
    fn get_namespace_name(&self, id: usize) -> Option<String> {
        self.id_to_name.get(&id).cloned()
    }
}

#[allow(unused)]
#[derive(Debug)]
pub struct StackValue {
    function: Option<String>,
    details: StackMapDetails,
    stack: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ProtocolMethodInfo {
    pub method_name: String,
    pub _type: String,
    pub fn_pointer: usize,
}

pub struct Runtime {
    pub memory: Memory,
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
    pub ffi_info_by_name: HashMap<String, usize>,
    namespaces: NamespaceManager,
    pub structs: StructManager,
    pub enums: EnumManager,
    pub string_constants: Vec<StringValue>,
    // TODO: Do I need anything more than
    // namespace manager? Shouldn't these functions
    // and things be under that?
    pub functions: Vec<Function>,
    pub jump_table: Option<Mmap>,
    pub jump_table_offset: usize,
    compiler_channel: Option<BlockingSender<CompilerMessage, CompilerResponse>>,
    compiler_thread: Option<JoinHandle<()>>,
    protocol_info: HashMap<String, Vec<ProtocolMethodInfo>>,
    stack_segments: StackSegmentAllocator,
    stacks_for_continuation_swapping: Vec<ContinuationStack>,
}

pub fn create_stack_with_protected_page_after(stack_size: usize) -> MmapMut {
    let page_size = MmapOptions::page_size();
    let stack_size = stack_size + page_size;
    let stack = MmapOptions::new(stack_size).unwrap().map_mut().unwrap();
    // because the stack grows down we will protect the first page
    // so that if we go over the stack we will get a segfault
    let protected_area = &stack[0..page_size];
    unsafe {
        libc::mprotect(
            protected_area.as_ptr() as *mut c_void,
            page_size,
            libc::PROT_NONE,
        );
    }
    stack
}

type ContinuationStackBuffer = [usize; 512];

fn new_continuation_stack_buffer() -> ContinuationStackBuffer {
    [0; 512]
}

struct ContinuationStack {
    is_used: AtomicBool,
    stack: ContinuationStackBuffer,
}

impl Runtime {
    pub fn new(
        command_line_arguments: CommandLineArguments,
        allocator: Alloc,
        printer: Box<dyn Printer>,
    ) -> Self {
        Self {
            printer,
            command_line_arguments: command_line_arguments.clone(),
            memory: Memory {
                heap: allocator,
                stacks: vec![(
                    std::thread::current().id(),
                    create_stack_with_protected_page_after(STACK_SIZE),
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
                    c_calling_stack_pointers: HashMap::new(),
                }),
                Condvar::new(),
            )),
            ffi_function_info: vec![],
            ffi_info_by_name: HashMap::new(),
            namespaces: NamespaceManager::new(),
            structs: StructManager::new(),
            enums: EnumManager::new(),
            string_constants: vec![],
            jump_table: Some(
                MmapOptions::new(MmapOptions::page_size())
                    .unwrap()
                    .map()
                    .unwrap(),
            ),
            jump_table_offset: 0,
            functions: vec![],
            compiler_channel: None,
            compiler_thread: None,
            protocol_info: HashMap::new(),
            stack_segments: StackSegmentAllocator::new(),
            stacks_for_continuation_swapping: vec![ContinuationStack {
                is_used: AtomicBool::new(false),
                stack: [0; 512],
            }],
        }
    }

    pub fn reset(&mut self) {
        self.memory.reset();
        self.namespaces = NamespaceManager::new();
        self.structs = StructManager::new();
        self.enums = EnumManager::new();
        self.string_constants.clear();
        self.functions.clear();
        self.jump_table = Some(
            MmapOptions::new(MmapOptions::page_size())
                .unwrap()
                .map()
                .unwrap(),
        );
        self.jump_table_offset = 0;
        self.printer.reset();
        self.ffi_function_info.clear();
        self.ffi_info_by_name.clear();
        self.stack_segments.clear_all_segments();
        self.compiler_channel
            .as_mut()
            .unwrap()
            .send(CompilerMessage::Reset);
        self.protocol_info.clear();
    }

    pub fn start_compiler_thread(&mut self) {
        if self.compiler_channel.is_none() {
            let (sender, receiver) = blocking_channel();
            let args_clone = self.command_line_arguments.clone();
            let compiler_thread = thread::Builder::new()
                .name("Beagle Compiler".to_string())
                .spawn(move || {
                    CompilerThread::new(receiver, args_clone).run();
                })
                .unwrap();
            self.compiler_channel = Some(sender);
            self.compiler_thread = Some(compiler_thread);
        }
    }

    pub fn print(&mut self, result: usize) {
        let result = self.get_repr(result, 0).unwrap();
        self.printer.print(result);
    }

    pub fn println(&mut self, pointer: usize) -> Result<(), Box<dyn Error>> {
        let result = self.get_repr(pointer, 0).ok_or("Error printing")?;
        self.printer.println(result);
        Ok(())
    }

    pub fn is_paused(&self) -> bool {
        self.is_paused.load(std::sync::atomic::Ordering::Relaxed) == 1
    }

    pub fn pause_atom_ptr(&self) -> usize {
        self.is_paused.as_ptr() as usize
    }

    pub fn compile(&mut self, file_name: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let response = self
            .compiler_channel
            .as_ref()
            .unwrap()
            .send(CompilerMessage::CompileFile(file_name.to_string()));
        if let CompilerResponse::FunctionsToRun(functions) = response {
            Ok(functions)
        } else {
            Err("Error compiling".into())
        }
    }

    pub fn compile_string(&mut self, _string: &str) -> Result<usize, Box<dyn Error>> {
        let response = self
            .compiler_channel
            .as_ref()
            .unwrap()
            .send(CompilerMessage::CompileString(_string.to_string()));
        if let CompilerResponse::FunctionPointer(pointer) = response {
            Ok(pointer)
        } else {
            Err("Error compiling".into())
        }
    }

    pub fn allocate_string(
        &mut self,
        stack_pointer: usize,
        string: String,
    ) -> Result<Tagged, Box<dyn Error>> {
        let bytes = string.as_bytes();
        let words = bytes.len() / 8 + 1;
        let pointer = self.allocate(words, stack_pointer, BuiltInTypes::HeapObject)?;
        let pointer = self.memory.allocate_string(bytes, pointer)?;
        Ok(pointer)
    }

    pub fn allocate(
        &mut self,
        words: usize,
        stack_pointer: usize,
        kind: BuiltInTypes,
    ) -> Result<usize, Box<dyn Error>> {
        let options = self.memory.heap.get_allocation_options();

        if options.gc_always {
            self.gc(stack_pointer);
        }

        let result = self.memory.heap.try_allocate(words, kind);

        match result {
            Ok(AllocateAction::Allocated(value)) => {
                assert!(value.is_aligned());
                let value = kind.tag(value as isize);
                Ok(value as usize)
            }
            Ok(AllocateAction::Gc) => {
                self.gc(stack_pointer);
                let result = self.memory.heap.try_allocate(words, kind);
                if let Ok(AllocateAction::Allocated(result)) = result {
                    // tag
                    assert!(result.is_aligned());
                    let result = kind.tag(result as isize);
                    Ok(result as usize)
                } else {
                    self.memory.heap.grow();
                    // TODO: Detect loop here
                    let pointer = self.allocate(words, stack_pointer, kind)?;
                    // If we went down this path, our pointer is already tagged
                    Ok(pointer)
                }
            }
            Err(e) => Err(e),
        }
    }

    unsafe fn buffer_between<T>(start: *const T, end: *const T) -> &'static [T] {
        unsafe {
            let len = end.offset_from(start);
            slice::from_raw_parts(start, len as usize)
        }
    }

    fn find_function_for_pc(&self, pc: usize) -> Option<&Function> {
        self.functions.iter().find(|f| {
            let start = f.pointer.into();
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

    /// Save a stack segment for later restoration (for yield functionality)
    pub fn save_stack_segment(&mut self, stack_data: &[u8]) -> Result<usize, Box<dyn Error>> {
        self.stack_segments.add_segment(stack_data)
    }

    /// Restore a stack segment to the given pointer location
    pub fn restore_stack_segment(
        &self,
        id: usize,
        target_ptr: *mut u8,
    ) -> Result<usize, Box<dyn Error>> {
        self.stack_segments.restore_segment(id, target_ptr)
    }

    /// Remove a stack segment when it's no longer needed
    pub fn remove_stack_segment(&mut self, id: usize) -> Result<(), Box<dyn Error>> {
        self.stack_segments.remove_segment(id)
    }

    pub fn get_stack_segment_count(&self) -> usize {
        self.stack_segments.segment_count()
    }

    pub fn get_stack_segment(&self, id: usize) -> Option<&crate::gc::stack_segments::StackSegment> {
        self.stack_segments.get_segment(id)
    }

    pub fn gc(&mut self, stack_pointer: usize) {
        if self.memory.threads.len() == 1 {
            // If there is only one thread, that is us
            // that means nothing else could spin up a thread in the mean time
            // so there is no need to lock anything
            // Collect all stack pointers: regular stacks + saved stack segments
            let mut all_stack_pointers = vec![(self.get_stack_base(), stack_pointer)];
            all_stack_pointers.extend(self.stack_segments.get_all_stack_pointers());

            self.memory
                .heap
                .gc(&self.memory.stack_map, &all_stack_pointers);

            // duplicated below
            // TODO: This whole thing is awful.
            // I should be passing around the slot so I can just update the binding directly.
            let relocations = self.memory.get_namespace_relocations();
            for (namespace, values) in relocations {
                for (old, new) in values {
                    for (_, value) in self
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
                    unsafe { __pause(stack_pointer) };
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
            unsafe { __pause(stack_pointer) };
            return;
        }

        let locked = locked.unwrap();

        let (lock, cvar) = &*self.thread_state;
        let mut thread_state = lock.lock().unwrap();
        while thread_state.paused_threads + thread_state.c_calling_stack_pointers.len()
            < self.memory.active_threads()
        {
            thread_state = cvar.wait(thread_state).unwrap();
        }

        let mut stack_pointers = thread_state.stack_pointers.clone();
        stack_pointers.push((self.get_stack_base(), stack_pointer));
        // Add saved stack segments to the GC scan
        stack_pointers.extend(self.stack_segments.get_all_stack_pointers());

        drop(thread_state);

        self.memory.heap.gc(&self.memory.stack_map, &stack_pointers);

        // Duplicated from above becauase I can't borrow self twice
        // TODO: This whole thing is awful.
        // I should be passing around the slot so I can just update the binding directly.
        let relocations = self.memory.get_namespace_relocations();
        for (namespace, values) in relocations {
            for (old, new) in values {
                for (_, value) in self
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

    pub fn register_temporary_root(&mut self, root: usize) -> usize {
        self.memory.heap.register_temporary_root(root)
    }

    pub fn unregister_temporary_root(&mut self, id: usize) -> usize {
        self.memory.heap.unregister_temporary_root(id)
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
        let function_definition =
            self.get_function_by_pointer(BuiltInTypes::untag(function) as *const u8);
        if function_definition.is_none() {
            panic!("Function not found");
        }
        let function_definition = function_definition.unwrap();
        let num_locals = function_definition.number_of_locals;

        heap_object.write_field(0, function);
        heap_object.write_field(1, BuiltInTypes::Int.tag(num_free as isize) as usize);
        heap_object.write_field(2, BuiltInTypes::Int.tag(num_locals as isize) as usize);
        for (index, value) in free_variables.iter().enumerate() {
            heap_object.write_field((index + 3) as i32, *value);
        }
        Ok(heap_pointer)
    }

    #[allow(clippy::type_complexity)]
    pub fn get_function_base(&self, name: &str) -> Option<(u64, u64, fn(u64, u64) -> u64)> {
        let function = self.functions.iter().find(|f| f.name == name)?;

        let trampoline = self.get_trampoline();
        let stack_pointer = self.get_stack_base();

        Some((stack_pointer as u64, function.pointer.into(), trampoline))
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
        let trampoline = self.get_trampoline();
        let trampoline: fn(u64, u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline) };
        let call_fn = self.get_function_by_name("beagle.core/__call_fn").unwrap();
        let function_pointer = self.get_pointer(call_fn).unwrap() as usize;

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

        for function in self.functions.iter() {
            if function.is_foreign || function.is_builtin {
                continue;
            }
            let start: usize = function.pointer.into();
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

    pub fn add_ffi_function_info(&mut self, ffi_function_info: FFIInfo) -> usize {
        self.ffi_function_info.push(ffi_function_info);
        self.ffi_function_info.len() - 1
    }

    pub fn get_ffi_info(&self, ffi_info_id: usize) -> &FFIInfo {
        self.ffi_function_info.get(ffi_info_id).unwrap()
    }

    pub fn add_ffi_info_by_name(&mut self, function_name: String, ffi_info_id: usize) {
        self.ffi_info_by_name.insert(function_name, ffi_info_id);
    }

    pub fn find_ffi_info_by_name(&self, function_name: &str) -> Option<usize> {
        self.ffi_info_by_name.get(function_name).cloned()
    }

    pub fn reserve_namespace_slot(&self, name: &str) -> usize {
        self.namespaces
            .add_binding(name, BuiltInTypes::null_value() as usize)
    }

    pub fn current_namespace_id(&self) -> usize {
        self.namespaces.current_namespace
    }

    pub fn update_binding(&mut self, namespace_id: usize, namespace_slot: usize, value: usize) {
        let mut namespace = self
            .namespaces
            .get_namespace_by_id(namespace_id)
            .unwrap()
            .lock()
            .unwrap();
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

    pub fn find_binding(&self, namespace_id: usize, name: &str) -> Option<usize> {
        let namespace = self.namespaces.namespaces.get(namespace_id).unwrap();
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
                if depth > 0 {
                    return Some(format!("\"{}\"", string.str));
                }
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

                // TODO: Make this documented and good
                match header.type_id {
                    0 => {
                        if header.opaque {
                            println!("=====================");
                            println!("{:?} {:?}", header, BuiltInTypes::untag(value) as *const u8);
                            println!("=====================");
                            return Some("ErrorOpaque".to_string());
                        }
                        let struct_id = BuiltInTypes::untag(object.get_struct_id());
                        let struct_value = self.get_struct_by_id(struct_id);
                        Some(self.get_struct_repr(struct_value?, object.get_fields(), depth + 1)?)
                    }
                    1 => {
                        let fields = object.get_fields();
                        let mut repr = "[ ".to_string();
                        for (index, field) in fields.iter().enumerate() {
                            repr.push_str(&self.get_repr(*field, depth + 1)?);
                            if index != fields.len() - 1 {
                                repr.push_str(", ");
                            }
                        }
                        repr.push_str(" ]");
                        Some(repr)
                    }
                    2 => {
                        let bytes = object.get_string_bytes();
                        let string = unsafe { std::str::from_utf8_unchecked(bytes) };
                        if depth > 0 {
                            return Some(format!("\"{}\"", string));
                        }
                        Some(string.to_string())
                    }
                    _ => {
                        // This is an unknown object. Meaning it is invalid.
                        // We are going to print everything we can to debug this

                        println!("=====================");
                        println!("{:?} {:?}", header, BuiltInTypes::untag(value));
                        println!("=====================");
                        Some("ErrorUnknown".to_string())
                    }
                }
            }
        }
    }

    pub fn get_struct_by_id(&self, struct_id: usize) -> Option<&Struct> {
        self.structs.get_by_id(struct_id)
    }

    pub fn property_access(
        &self,
        struct_pointer: usize,
        str_constant_ptr: usize,
    ) -> Result<(usize, usize), Box<dyn Error>> {
        if BuiltInTypes::untag(struct_pointer) % 8 != 0 {
            return Err("Not aligned".into());
        }
        let heap_object = HeapObject::from_tagged(struct_pointer);
        let str_constant_ptr: usize = BuiltInTypes::untag(str_constant_ptr);
        let string_value = &self.string_constants[str_constant_ptr];
        let string = &string_value.str;
        let struct_type_id = heap_object.get_struct_id();
        let struct_type_id = BuiltInTypes::untag(struct_type_id);
        let struct_value = self.get_struct_by_id(struct_type_id).unwrap();
        let field_index = struct_value
            .fields
            .iter()
            .position(|f| f == string)
            .ok_or(format!(
                "Field not found {} for struct {:?}",
                string, struct_value
            ))?;
        Ok((heap_object.get_field(field_index), field_index))
    }

    pub fn type_of(&self, value: usize) -> usize {
        let tag = BuiltInTypes::get_kind(value);
        match tag {
            BuiltInTypes::Null => todo!(),
            BuiltInTypes::Int => todo!(),
            BuiltInTypes::Float => todo!(),
            BuiltInTypes::String => todo!(),
            BuiltInTypes::Bool => todo!(),
            BuiltInTypes::Function => todo!(),
            BuiltInTypes::Closure => todo!(),
            BuiltInTypes::HeapObject => {
                let heap_object = HeapObject::from_tagged(value);
                let struct_type_id = heap_object.get_struct_id();
                let struct_type_id = BuiltInTypes::untag(struct_type_id);
                let struct_value = self.get_struct_by_id(struct_type_id).unwrap();
                let struct_name = struct_value.name.clone();
                let (namespace_name, struct_name) = struct_name.split_once("/").unwrap();
                let namespace_id = self.get_namespace_id(namespace_name).unwrap();
                let slot = self.find_binding(namespace_id, struct_name).unwrap();

                self.get_binding(namespace_id, slot)
            }
        }
    }

    pub fn equal(&self, a: usize, b: usize) -> bool {
        let mut a = a;
        let mut b = b;
        if a == b {
            return true;
        }
        let mut a_tag = BuiltInTypes::get_kind(a);
        let mut b_tag = BuiltInTypes::get_kind(b);
        // TODO: Make this pluggeable by having a protocol for it
        // so that we can have equality for hashmaps and vectors
        // without custom handling. I can probably do this by just calling
        // this as the default implementation for the protocol
        // I don't have that concept right now.

        if a_tag == BuiltInTypes::String && b_tag == BuiltInTypes::HeapObject {
            (a_tag, b_tag) = (b_tag, a_tag);
            (a, b) = (b, a);
        }

        if a_tag == BuiltInTypes::HeapObject && b_tag == BuiltInTypes::String {
            let a_object = HeapObject::from_tagged(a);
            if a_object.get_type_id() != 2 {
                return false;
            }
            let b_string = self.get_str_literal(b);
            let a_string = a_object.get_string_bytes();
            let a_string = unsafe { std::str::from_utf8_unchecked(a_string) };
            return a_string == b_string;
        }
        if a_tag != b_tag {
            return false;
        }
        match a_tag {
            BuiltInTypes::Null => true,
            BuiltInTypes::Int => a == b,
            BuiltInTypes::Float => a == b,
            BuiltInTypes::String => a == b,
            BuiltInTypes::Bool => a == b,
            BuiltInTypes::Function => a == b,
            BuiltInTypes::Closure => a == b,
            BuiltInTypes::HeapObject => {
                let a_object = HeapObject::from_tagged(a);
                let b_object = HeapObject::from_tagged(b);
                if a_object.get_struct_id() != b_object.get_struct_id() {
                    return false;
                }
                let a_fields = a_object.get_fields();
                let b_fields = b_object.get_fields();
                if a_fields.len() != b_fields.len() {
                    return false;
                }
                for (a, b) in a_fields.iter().zip(b_fields.iter()) {
                    if !self.equal(*a, *b) {
                        return false;
                    }
                }
                true
            }
        }
    }

    pub fn write_field(
        &self,
        struct_pointer: usize,
        str_constant_ptr: usize,
        value: usize,
    ) -> usize {
        if BuiltInTypes::untag(struct_pointer) % 8 != 0 {
            panic!("Not aligned");
        }
        let heap_object = HeapObject::from_tagged(struct_pointer);
        let str_constant_ptr: usize = BuiltInTypes::untag(str_constant_ptr);
        let string_value = &self.string_constants[str_constant_ptr];
        let string = &string_value.str;
        let struct_type_id = heap_object.get_struct_id();
        let struct_type_id = BuiltInTypes::untag(struct_type_id);
        let struct_value = self.get_struct_by_id(struct_type_id).unwrap();
        let field_index = struct_value
            .fields
            .iter()
            .position(|f| f == string)
            .unwrap_or_else(|| panic!("Field {} not found in struct", string));
        // Temporary +1 because I was writing size as the first field
        // and I haven't changed that

        heap_object.write_field(field_index as i32, value);
        field_index
    }

    pub fn get_struct_repr(
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

    pub fn get_string_literal(&self, value: usize) -> String {
        let value = BuiltInTypes::untag(value);
        self.string_constants[value].str.clone()
    }

    pub fn get_str_literal(&self, value: usize) -> &str {
        let value = BuiltInTypes::untag(value);
        &self.string_constants[value].str
    }

    pub fn get_string(&self, value: usize) -> String {
        let tag = BuiltInTypes::get_kind(value);
        if tag == BuiltInTypes::String {
            self.get_string_literal(value)
        } else if tag == BuiltInTypes::HeapObject {
            let heap_object = HeapObject::from_tagged(value);
            if heap_object.get_type_id() != 2 {
                panic!("Not a string");
            }
            let bytes = heap_object.get_string_bytes();
            unsafe { std::str::from_utf8_unchecked(bytes).to_string() }
        } else {
            panic!("Not a string");
        }
    }

    pub fn get_substring(
        &mut self,
        stack_pointer: usize,
        string: usize,
        start: usize,
        length: usize,
    ) -> Result<Tagged, Box<dyn Error>> {
        let tag = BuiltInTypes::get_kind(string);
        if tag == BuiltInTypes::String {
            let string = self.get_str_literal(string);
            let string = string[start..start + length].to_string();
            self.allocate_string(stack_pointer, string)
        } else if tag == BuiltInTypes::HeapObject {
            let heap_object = HeapObject::from_tagged(string);
            if heap_object.get_type_id() != 2 {
                panic!("Not a string");
            }
            let bytes = heap_object.get_string_bytes();
            let string = unsafe { std::str::from_utf8_unchecked(bytes) };
            let string = string[start..start + length].to_string();
            self.allocate_string(stack_pointer, string)
        } else {
            panic!("Not a string");
        }
    }

    pub fn add_foreign_function(
        &mut self,
        name: Option<&str>,
        function: *const u8,
        number_of_args: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let index = self.functions.len();
        let pointer = function;
        let jump_table_offset = self.add_jump_table_entry(index, pointer)?;
        self.functions.push(Function {
            name: name.unwrap_or("<Anonymous>").to_string(),
            pointer: pointer.into(),
            jump_table_offset,
            is_foreign: true,
            is_builtin: false,
            needs_stack_pointer: false,
            is_defined: true,
            number_of_locals: 0,
            size: 0,
            number_of_args,
        });
        debugger(Message {
            kind: "foreign_function".to_string(),
            data: Data::ForeignFunction {
                name: name.unwrap_or("<Anonymous>").to_string(),
                pointer: Self::get_function_pointer(self, self.functions.last().unwrap().clone())
                    .unwrap() as usize,
            },
        });
        let function_pointer =
            Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap();
        Ok(function_pointer as usize)
    }

    pub fn add_builtin_function(
        &mut self,
        name: &str,
        function: *const u8,
        needs_stack_pointer: bool,
        number_of_args: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let index = self.functions.len();
        let pointer = function;

        let jump_table_offset = self.add_jump_table_entry(index, pointer)?;
        self.functions.push(Function {
            name: name.to_string(),
            pointer: pointer.into(),
            jump_table_offset,
            is_foreign: true,
            is_builtin: true,
            needs_stack_pointer,
            is_defined: true,
            number_of_locals: 0,
            size: 0,
            number_of_args,
        });
        let pointer =
            Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap();
        // self.namespaces.add_binding(name, pointer as usize);
        debugger(Message {
            kind: "builtin_function".to_string(),
            data: Data::BuiltinFunction {
                name: name.to_string(),
                pointer: pointer as usize,
            },
        });
        Ok(self.functions.len() - 1)
    }

    pub fn update_stack_map_information(
        &mut self,
        stack_map: Vec<(usize, StackMapDetails)>,
        function_pointer: usize,
        name: Option<&str>,
    ) {
        debugger(Message {
            kind: "stack_map".to_string(),
            data: Data::StackMap {
                pc: function_pointer,
                name: name.unwrap_or("<Anonymous>").to_string(),
                stack_map: stack_map.clone(),
            },
        });
        self.memory.stack_map.extend(stack_map);
    }

    pub fn replace_function_binding(&mut self, name: String, pointer: usize) {
        let untagged_pointer = BuiltInTypes::untag(pointer) as *const u8;
        let existing_function = self.get_function_by_pointer(untagged_pointer);
        if existing_function.is_none() {
            panic!("Function not found");
        }
        let existing_function = existing_function.unwrap().clone();
        for (index, function) in self.functions.iter_mut().enumerate() {
            if function.name == name {
                self.overwrite_function(index, untagged_pointer, existing_function.size)
                    .unwrap();
                break;
            }
        }
    }

    pub fn upsert_function(
        &mut self,
        name: Option<&str>,
        pointer: *const u8,
        size: usize,
        number_of_locals: usize,
        stack_map: Vec<(usize, StackMapDetails)>,
        number_of_args: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let mut already_defined = false;
        let mut function_pointer = 0;
        if name.is_some() {
            for (index, function) in self.functions.iter_mut().enumerate() {
                if function.name == name.unwrap() {
                    function_pointer = self.overwrite_function(index, pointer, size)?;
                    // self.namespaces.add_binding(name.unwrap(), function_pointer);
                    already_defined = true;
                    break;
                }
            }
        }
        if !already_defined {
            function_pointer =
                self.add_function(name, pointer, size, number_of_locals, number_of_args)?;
        }
        assert!(function_pointer != 0);

        self.update_stack_map_information(stack_map, function_pointer, name);

        debugger(Message {
            kind: "user_function".to_string(),
            data: Data::UserFunction {
                name: name.unwrap_or("<Anonymous>").to_string(),
                pointer: function_pointer,
                len: size,
                number_of_arguments: number_of_args,
            },
        });
        Ok(function_pointer)
    }

    pub fn reserve_function(
        &mut self,
        name: &str,
        number_of_args: usize,
    ) -> Result<Function, Box<dyn Error>> {
        for function in self.functions.iter_mut() {
            if function.name == name {
                return Ok(function.clone());
            }
        }
        let index = self.functions.len();
        let jump_table_offset = self.add_jump_table_entry(index, std::ptr::null())?;
        let function = Function {
            name: name.to_string(),
            pointer: RawPtr::new(std::ptr::null()),
            jump_table_offset,
            is_foreign: false,
            is_builtin: false,
            needs_stack_pointer: false,
            is_defined: false,
            number_of_locals: 0,
            size: 0,
            number_of_args,
        };
        self.functions.push(function.clone());
        Ok(function)
    }

    pub fn add_function(
        &mut self,
        name: Option<&str>,
        pointer: *const u8,
        size: usize,
        number_of_locals: usize,
        number_of_args: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let index = self.functions.len();
        self.functions.push(Function {
            name: name.unwrap_or("<Anonymous>").to_string(),
            pointer: pointer.into(),
            jump_table_offset: 0,
            is_foreign: false,
            is_builtin: false,
            needs_stack_pointer: false,
            is_defined: true,
            number_of_locals,
            size,
            number_of_args,
        });
        let function_pointer =
            Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap();
        let jump_table_offset = self.add_jump_table_entry(index, function_pointer)?;

        self.functions[index].jump_table_offset = jump_table_offset;
        if let Some(name) = name {
            self.add_binding(name, function_pointer as usize);
        }
        Ok(function_pointer as usize)
    }

    pub fn overwrite_function(
        &mut self,
        index: usize,
        pointer: *const u8,
        size: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let function = &mut self.functions[index];
        function.pointer = pointer.into();
        let jump_table_offset = function.jump_table_offset;
        let function_clone = function.clone();
        let function_pointer = self.get_function_pointer(function_clone).unwrap();
        self.modify_jump_table_entry(jump_table_offset, function_pointer as usize)?;
        let function = &mut self.functions[index];
        function.size = size;
        function.is_defined = true;
        Ok(function_pointer as usize)
    }

    pub fn get_pointer(&self, function: &Function) -> Result<*const u8, Box<dyn Error>> {
        Ok(function.pointer.into())
    }

    pub fn get_function_pointer(&self, function: Function) -> Result<*const u8, Box<dyn Error>> {
        // Gets the absolute pointer to a function
        // if it is a foreign function, return the offset
        // if it is a local function, return the offset + the start of code_memory
        Ok(function.pointer.into())
    }

    pub fn get_jump_table_pointer(&self, function: Function) -> Result<usize, Box<dyn Error>> {
        Ok(function.jump_table_offset * 8 + self.jump_table.as_ref().unwrap().as_ptr() as usize)
    }

    pub fn add_jump_table_entry(
        &mut self,
        _index: usize,
        pointer: *const u8,
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

    pub fn modify_jump_table_entry(
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

    pub fn find_function(&self, name: &str) -> Option<Function> {
        assert!(
            name.contains("/"),
            "Function name should contain /: {:?}",
            name
        );
        self.functions.iter().find(|f| f.name == name).cloned()
    }

    pub fn get_function_by_pointer(&self, value: *const u8) -> Option<&Function> {
        self.functions.iter().find(|f| f.pointer == value.into())
    }

    pub fn get_function_by_pointer_mut(&mut self, value: *const u8) -> Option<&mut Function> {
        self.functions
            .iter_mut()
            .find(|f| f.pointer == value.into())
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

    pub fn add_function_mark_executable(
        &self,
        name: String,
        code: &[u8],
        number_of_locals: i32,
        number_of_args: usize,
    ) -> Result<usize, Box<dyn Error>> {
        self.compiler_channel
            .as_ref()
            .unwrap()
            .send(CompilerMessage::AddFunctionMarkExecutable(
                name.clone(),
                code.to_vec(),
                number_of_locals as usize,
                number_of_args,
            ));
        Ok(self.get_function_by_name(&name).unwrap().pointer.into())
    }

    fn add_binding(&mut self, name: &str, function_pointer: usize) -> usize {
        self.namespaces.add_binding(name, function_pointer)
    }

    pub fn get_trampoline(&self) -> fn(u64, u64) -> u64 {
        let trampoline = self.get_function_by_name("trampoline").unwrap();

        unsafe { std::mem::transmute(trampoline.pointer) }
    }

    pub fn add_string(&mut self, string_value: StringValue) -> usize {
        if let Some(index) = self
            .string_constants
            .iter()
            .position(|s| s.str == string_value.str)
        {
            return index;
        }
        self.string_constants.push(string_value);
        self.string_constants.len() - 1
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
        self.add_binding(simple_name, 0);
    }

    pub fn add_enum(&mut self, e: Enum) {
        let name = e.name.clone();
        self.enums.insert(e);
        // See comment about structs above
        self.add_binding(&name, 0);
    }

    pub fn get_enum(&self, name: &str) -> Option<&Enum> {
        self.enums.get(name)
    }

    pub fn get_struct(&self, name: &str) -> Option<(usize, &Struct)> {
        self.structs.get(name)
    }

    pub fn get_namespace_from_alias(&self, alias: &str) -> Option<String> {
        let current_namespace = self.current_namespace_name();
        let current_namespace = self.get_namespace_id(current_namespace.as_str()).unwrap();
        let current_namespace = self.namespaces.namespaces.get(current_namespace).unwrap();
        let current_namespace = current_namespace.lock().unwrap();
        let namespace_id = current_namespace.aliases.get(alias)?;
        let namespace = self.namespaces.namespaces.get(*namespace_id).unwrap();
        let namespace = namespace.lock().unwrap();
        Some(namespace.name.clone())
    }

    pub fn get_pointer_for_function(&self, function: &Function) -> Option<usize> {
        Some(function.pointer.into())
    }

    pub fn set_compiler_channel(
        &mut self,
        compiler_channel: BlockingSender<CompilerMessage, CompilerResponse>,
    ) {
        self.compiler_channel = Some(compiler_channel);
    }

    pub fn set_pause_atom_ptr(&self, pause_atom_ptr: usize) {
        self.compiler_channel
            .as_ref()
            .unwrap()
            .send(CompilerMessage::SetPauseAtomPointer(pause_atom_ptr));
    }

    pub fn add_protocol_info(
        &mut self,
        protocol_name: &str,
        struct_name: &str,
        method_name: &str,
        f: usize,
    ) {
        // TODO: When do I deal with duplicates?
        self.protocol_info
            .entry(protocol_name.to_string())
            .or_default()
            .push(ProtocolMethodInfo {
                _type: struct_name.to_string(),
                method_name: method_name.to_string(),
                fn_pointer: f,
            });
    }

    pub fn compile_protocol_method(&self, protocol_name: &str, method_name: &str) -> usize {
        let protocol_info = self.protocol_info.get(protocol_name).unwrap();
        let method_info: Vec<ProtocolMethodInfo> = protocol_info
            .iter()
            .filter(|m| m.method_name == method_name)
            .cloned()
            .collect();
        self.compiler_channel
            .as_ref()
            .unwrap()
            .send(CompilerMessage::CompileProtocolMethod(
                protocol_name.to_string(),
                method_name.to_string(),
                method_info,
            ));

        0
    }

    pub fn resolve(&self, struct_name: String) -> String {
        if !struct_name.contains("/") {
            let current_namespace_name = self.current_namespace_name();
            let current_namespace_id = self
                .get_namespace_id(current_namespace_name.as_str())
                .unwrap();
            let find_binding = self.find_binding(current_namespace_id, struct_name.as_str());
            if find_binding.is_some() {
                return format!("{}/{}", current_namespace_name, struct_name);
            }
            let beagle_core = self.get_namespace_id("beagle.core").unwrap();
            let find_binding = self.find_binding(beagle_core, struct_name.as_str());
            if find_binding.is_some() {
                return format!("beagle.core/{}", struct_name);
            }
            panic!("Cannot resolve {}", struct_name);
        }
        let (namespace_or_alias, struct_name) = struct_name.split_once("/").unwrap();

        let namespace_from_alias = self.get_namespace_from_alias(namespace_or_alias);
        if namespace_from_alias.is_some() {
            return format!("{}/{}", namespace_from_alias.unwrap(), struct_name);
        }
        let namespace_id = self.get_namespace_id(namespace_or_alias);
        if namespace_id.is_some() {
            return format!("{}/{}", namespace_or_alias, struct_name);
        }
        panic!("Cannot resolve {}", struct_name);
    }

    pub fn get_command_line_args(&self) -> &CommandLineArguments {
        &self.command_line_arguments
    }

    pub fn get_stack_for_continuiation_swapping(&mut self) -> (*const u8, usize) {
        for (i, stack) in self.stacks_for_continuation_swapping.iter().enumerate() {
            if stack
                .is_used
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return (
                    unsafe {
                        stack.stack.as_ptr().offset(stack.stack.len() as isize - 64) as *const u8
                    },
                    i,
                );
            }
        }
        self.stacks_for_continuation_swapping
            .push(ContinuationStack {
                is_used: AtomicBool::new(true),
                stack: new_continuation_stack_buffer(),
            });
        self.get_stack_for_continuiation_swapping()
    }

    pub fn release_stack_for_continuation_swapping(&mut self, index: usize) {
        if let Some(stack) = self.stacks_for_continuation_swapping.get(index) {
            stack.is_used.store(false, Ordering::SeqCst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc::get_allocate_options;

    #[test]
    fn test_stack_segment_integration() {
        let args = CommandLineArguments {
            program: None,
            show_times: false,
            show_gc_times: false,
            print_ast: false,
            no_gc: false,
            gc_always: false,
            all_tests: false,
            test: false,
            debug: false,
            verbose: false,
            no_std: false,
            print_parse: false,
            print_builtin_calls: false,
        };
        let allocator_options = get_allocate_options(&args);
        let allocator = crate::Alloc::new(allocator_options);
        let printer = Box::new(crate::runtime::DefaultPrinter);
        let mut runtime = Runtime::new(args, allocator, printer);

        // Test saving a stack segment
        let test_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let segment_id = runtime.save_stack_segment(&test_data).unwrap();

        // Test restoring the segment
        let mut restore_buffer = vec![0u8; 10];
        let bytes_copied = runtime
            .restore_stack_segment(segment_id, restore_buffer.as_mut_ptr())
            .unwrap();

        assert_eq!(bytes_copied, 8);
        assert_eq!(&restore_buffer[0..8], &test_data[..]);

        // Test removing the segment
        runtime.remove_stack_segment(segment_id).unwrap();

        // Verify it's removed by trying to restore again (should fail)
        let result = runtime.restore_stack_segment(segment_id, restore_buffer.as_mut_ptr());
        assert!(result.is_err());
    }
}
