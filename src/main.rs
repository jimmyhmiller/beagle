#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::missing_safety_doc)]
use crate::machine_code::arm_codegen::{SP, X0, X1, X10, X2, X3, X4};
use arm::LowLevelArm;
use bincode::{config::standard, Decode, Encode};
use clap::{command, Parser as ClapParser};
#[allow(unused)]
use gc::{
    compacting::CompactingHeap, mutex_allocator::MutexAllocator,
    simple_generation::SimpleGeneration, simple_mark_and_sweep::SimpleMarkSweepHeap,
};
use gc::{get_allocate_options, Allocator, StackMapDetails};
use runtime::{DefaultPrinter, Printer, Runtime, TestPrinter};

use std::{cell::UnsafeCell, env, error::Error, sync::OnceLock, time::Instant};

mod arm;
pub mod ast;
mod builtins;
mod code_memory;
pub mod common;
mod compiler;
mod gc;
pub mod ir;
pub mod machine_code;
pub mod parser;
mod pretty_print;
mod primitives;
mod register_allocation;
pub mod runtime;
mod types;

#[derive(Debug, Encode, Decode, Clone)]
pub struct Message {
    kind: String,
    data: Data,
}

// TODO: This should really live on the debugger side of things
#[derive(Debug, Encode, Decode, Clone)]
enum Data {
    ForeignFunction {
        name: String,
        pointer: usize,
    },
    BuiltinFunction {
        name: String,
        pointer: usize,
    },
    HeapSegmentPointer {
        pointer: usize,
    },
    UserFunction {
        name: String,
        pointer: usize,
        len: usize,
        number_of_arguments: usize,
    },
    Label {
        label: String,
        function_pointer: usize,
        label_index: usize,
        label_location: usize,
    },
    StackMap {
        pc: usize,
        name: String,
        stack_map: Vec<(usize, StackMapDetails)>,
    },
    Allocate {
        bytes: usize,
        stack_pointer: usize,
        kind: String,
    },
    Tokens {
        file_name: String,
        tokens: Vec<String>,
        token_line_column_map: Vec<(usize, usize)>,
    },
    Ir {
        function_pointer: usize,
        file_name: String,
        instructions: Vec<String>,
        token_range_to_ir_range: Vec<((usize, usize), (usize, usize))>,
    },
    Arm {
        function_pointer: usize,
        file_name: String,
        instructions: Vec<String>,
        ir_to_machine_code_range: Vec<(usize, (usize, usize))>,
    },
}

trait Serialize {
    fn to_binary(&self) -> Vec<u8>;
    #[allow(unused)]
    fn from_binary(data: &[u8]) -> Self;
}

impl<T: Encode + Decode> Serialize for T {
    fn to_binary(&self) -> Vec<u8> {
        bincode::encode_to_vec(self, standard()).unwrap()
    }
    fn from_binary(data: &[u8]) -> T {
        let (data, _) = bincode::decode_from_slice(data, standard()).unwrap();
        data
    }
}

fn compile_trampoline(runtime: &mut Runtime) {
    let mut lang = LowLevelArm::new();
    // lang.breakpoint();
    lang.prelude(-2);

    // Should I store or push?
    for (i, reg) in lang.canonical_volatile_registers.clone().iter().enumerate() {
        lang.store_on_stack(*reg, -((i + 2) as i32));
    }

    lang.mov_reg(X10, SP);
    lang.mov_reg(SP, X0);
    lang.push_to_stack(X10);

    lang.mov_reg(X10, X1);
    lang.mov_reg(X0, X2);
    lang.mov_reg(X1, X3);
    lang.mov_reg(X2, X4);

    lang.call(X10);
    // lang.breakpoint();
    lang.pop_from_stack_indexed(X10, 0);
    lang.mov_reg(SP, X10);
    for (i, reg) in lang
        .canonical_volatile_registers
        .clone()
        .iter()
        .enumerate()
        .rev()
    {
        lang.load_from_stack(*reg, -((i + 2) as i32));
    }
    lang.epilogue(2);
    lang.ret();

    runtime
        .add_function_mark_executable("trampoline".to_string(), &lang.compile_directly(), 0, 3)
        .unwrap();

    let function = runtime.get_function_by_name_mut("trampoline").unwrap();
    function.is_builtin = true;
}

fn compile_save_volatile_registers(runtime: &mut Runtime) {
    let mut lang = LowLevelArm::new();
    // lang.breakpoint();
    lang.prelude(-2);

    lang.sub_stack_pointer((lang.canonical_volatile_registers.len() + 2) as i32);

    for (i, reg) in lang.canonical_volatile_registers.clone().iter().enumerate() {
        lang.store_on_stack(*reg, -((i + 3) as i32));
    }

    lang.call(X1);

    for (i, reg) in lang.canonical_volatile_registers.clone().iter().enumerate() {
        lang.load_from_stack(*reg, -((i + 3) as i32));
    }

    lang.add_stack_pointer((lang.canonical_volatile_registers.len() + 2) as i32);

    lang.epilogue(2);
    lang.ret();

    runtime
        .add_function_mark_executable(
            "beagle.builtin/save_volatile_registers".to_string(),
            &lang.compile_directly(),
            0,
            2,
        )
        .unwrap();
    let function = runtime
        .get_function_by_name_mut("beagle.builtin/save_volatile_registers")
        .unwrap();
    function.is_builtin = true;
}

#[derive(ClapParser, Debug, Clone)]
#[command(version, about, long_about = None)]
#[command(name = "beag")]
#[command(bin_name = "beag")]
pub struct CommandLineArguments {
    program: Option<String>,
    #[clap(long, default_value = "false")]
    show_times: bool,
    #[clap(long, default_value = "false")]
    show_gc_times: bool,
    #[clap(long, default_value = "false")]
    print_ast: bool,
    #[clap(long, default_value = "false")]
    no_gc: bool,
    #[clap(long, default_value = "false")]
    gc_always: bool,
    #[clap(long, default_value = "false")]
    all_tests: bool,
    #[clap(long, default_value = "false")]
    test: bool,
    #[clap(long, default_value = "false")]
    debug: bool,
    #[clap(long, default_value = "false")]
    verbose: bool,
    #[clap(long, default_value = "false")]
    no_std: bool,
    #[clap(long, default_value = "false")]
    print_parse: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = CommandLineArguments::parse();
    if args.all_tests {
        run_all_tests(args)
    } else {
        main_inner(args)
    }
}

fn run_all_tests(args: CommandLineArguments) -> Result<(), Box<dyn Error>> {
    for entry in std::fs::read_dir("resources")? {
        let entry = entry?;
        let mut path = entry.path();
        if !path.exists() {
            path = path
                .parent()
                .unwrap()
                .join("resources")
                .join(path.file_name().unwrap());
        }
        let path = path.to_str().unwrap();
        let source: String = std::fs::read_to_string(path)?;
        if !source.contains("// Expect") {
            continue;
        }
        #[cfg(not(feature = "thread-safe"))]
        {
            if source.contains("// thread-safe") {
                continue;
            }
        }

        println!("Running test: {}", path);
        let args = CommandLineArguments {
            program: Some(path.to_string()),
            show_times: args.show_times,
            show_gc_times: args.show_gc_times,
            print_ast: args.print_ast,
            no_gc: args.no_gc,
            gc_always: args.gc_always,
            all_tests: false,
            test: true,
            debug: args.debug,
            verbose: args.verbose,
            no_std: args.no_std,
            print_parse: args.print_parse,
        };
        main_inner(args).ok();
        RUNTIME.get().unwrap().get_mut().reset();
    }
    Ok(())
}

cfg_if::cfg_if! {
    if #[cfg(feature = "compacting")] {
        cfg_if::cfg_if! {
            if #[cfg(feature = "thread-safe")] {
                pub type Alloc = MutexAllocator<CompactingHeap>;
            } else {
                pub type Alloc = CompactingHeap;
            }
        }
    } else if #[cfg(feature = "simple-mark-and-sweep")] {
        cfg_if::cfg_if! {
            if #[cfg(feature = "thread-safe")] {
                pub type Alloc = MutexAllocator<SimpleMarkSweepHeap>;
            } else {
                pub type Alloc = SimpleMarkSweepHeap;
            }
        }
    } else if #[cfg(feature = "simple-generation")] {
        cfg_if::cfg_if! {
            if #[cfg(feature = "thread-safe")] {
                pub type Alloc = MutexAllocator<SimpleGeneration>;
            } else {
                pub type Alloc = SimpleGeneration;
            }
        }
    } else if #[cfg(feature = "thread-safe")] {
        pub type Alloc = MutexAllocator<SimpleGeneration>;
    } else {
        pub type Alloc = SimpleGeneration;
    }
}

fn main_inner(mut args: CommandLineArguments) -> Result<(), Box<dyn Error>> {
    if args.program.is_none() {
        println!("No program provided");
        return Ok(());
    }
    let program = args.program.clone().unwrap();
    let source = std::fs::read_to_string(program.clone())?;
    // TODO: This is very ad-hoc
    // I should make it real functionality later
    // but right now I just want something working
    let has_expect = args.test && source.contains("// Expect");

    let args_clone = args.clone();

    if source.contains("// no-std") {
        args.no_std = true;
    }

    RUNTIME.get_or_init(|| {
        let allocator = Alloc::new(get_allocate_options(&args_clone.clone()));
        let printer: Box<dyn Printer> = if has_expect {
            Box::new(TestPrinter::new(Box::new(DefaultPrinter)))
        } else {
            Box::new(DefaultPrinter)
        };

        let runtime = Runtime::new(args_clone, allocator, printer);
        SyncUnsafeCell::new(runtime)
    });

    // let allocator = Alloc::new(get_allocate_options(&args));
    // let printer: Box<dyn Printer> = if has_expect {
    //     Box::new(TestPrinter::new(Box::new(DefaultPrinter)))
    // } else {
    //     Box::new(DefaultPrinter)
    // };

    let runtime = RUNTIME.get().unwrap().get_mut();

    runtime.start_compiler_thread();

    compile_trampoline(runtime);
    compile_save_volatile_registers(runtime);

    let pause_atom_ptr = runtime.pause_atom_ptr();
    runtime.set_pause_atom_ptr(pause_atom_ptr);

    runtime.install_builtins()?;
    let compile_time = Instant::now();

    let mut top_levels = vec![];
    if !args.no_std {
        let mut exe_path = env::current_exe()?;
        exe_path = exe_path.parent().unwrap().to_path_buf();
        if !exe_path.join("resources/std.bg").exists() {
            exe_path = exe_path.parent().unwrap().to_path_buf();
        }
        top_levels = runtime.compile(exe_path.join("resources/std.bg").to_str().unwrap())?;
    }

    // TODO: Need better name for top_level
    // It should really be the top level of a namespace
    let new_top_levels = runtime.compile(&program)?;
    let current_namespace = runtime.current_namespace_id();
    top_levels.extend(new_top_levels);

    runtime.write_functions_to_pid_map();

    runtime.check_functions();
    if args.show_times {
        println!("Compile time {:?}", compile_time.elapsed());
    }

    let time = Instant::now();

    if args.program.unwrap() == "resources/array_literal.bg" {
        println!("Running {:?}", top_levels);
    }

    for top_level in top_levels {
        if let Some(f) = runtime.get_function0(&top_level) {
            f();
        } else {
            panic!(
                "We are supposed to have top level, but didn't find the function {}",
                top_level
            );
        }
    }
    runtime.set_current_namespace(current_namespace);
    let fully_qualified_main = runtime.current_namespace_name() + "/main";
    if let Some(f) = runtime.get_function0(&fully_qualified_main) {
        let result = f();
        runtime.println(result as usize).unwrap();
    } else if args.debug {
        println!("No main function");
    }

    if args.show_times {
        println!("Time {:?}", time.elapsed());
    }

    if has_expect {
        let source = std::fs::read_to_string(program)?;
        let expected = get_expect(&source);
        let expected = expected.trim();
        let printed = runtime.printer.get_output().join("").trim().to_string();
        if printed != expected {
            println!("Expected: \n{}\n", expected);
            println!("Got: \n{}\n", printed);
            panic!("Test failed");
        }
        println!("Test passed");
    }

    loop {
        // take the list of threads so we are not holding a borrow on the compiler
        // use mem::replace to swap out the threads with an empty vec
        let threads = std::mem::take(&mut runtime.memory.join_handles);
        if threads.is_empty() {
            break;
        }
        for thread in threads {
            thread.join().unwrap();
        }
    }

    Ok(())
}

fn get_expect(source: &str) -> String {
    let start = source.find("// Expect").unwrap();
    // get each line as long as they start with //
    let lines = source[start..]
        .lines()
        .skip(1)
        .take_while(|line| line.starts_with("//"))
        .map(|line| line.trim_start_matches("//").trim())
        .collect::<Vec<_>>()
        .join("\n");
    lines
}

#[test]
fn try_all_examples() -> Result<(), Box<dyn Error>> {
    let args = CommandLineArguments {
        program: None,
        show_times: false,
        show_gc_times: false,
        print_ast: false,
        no_gc: false,
        gc_always: false,
        all_tests: true,
        test: false,
        debug: false,
        verbose: false,
        no_std: false,
        print_parse: false,
    };
    run_all_tests(args)?;
    Ok(())
}
#[repr(transparent)]
pub struct SyncUnsafeCell<T: ?Sized> {
    value: UnsafeCell<T>,
}
impl<T> SyncUnsafeCell<T> {
    pub fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
        }
    }

    pub fn get(&self) -> &T {
        unsafe { &*self.value.get() }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn get_mut(&self) -> &mut T {
        unsafe { &mut *self.value.get() }
    }

    pub fn reset(&self, value: T) {
        unsafe {
            *self.value.get() = value;
        }
    }
}
unsafe impl<T: ?Sized + Sync> Sync for SyncUnsafeCell<T> {}

pub static RUNTIME: OnceLock<SyncUnsafeCell<Runtime>> = OnceLock::new();

pub fn get_runtime() -> &'static SyncUnsafeCell<Runtime> {
    RUNTIME.get().unwrap()
}
