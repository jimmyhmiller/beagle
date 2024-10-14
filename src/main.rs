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
use gc::{Allocator, StackMapDetails};
use runtime::{DefaultPrinter, Printer, Runtime, TestPrinter};

use std::{error::Error, time::Instant};

mod arm;
pub mod ast;
mod builtins;
pub mod common;
mod gc;
pub mod ir;
pub mod machine_code;
pub mod parser;
mod primitives;
pub mod runtime;
mod types;

#[derive(Debug, Encode, Decode)]
pub struct Message {
    kind: String,
    data: Data,
}

// TODO: This should really live on the debugger side of things
#[derive(Debug, Encode, Decode)]
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

fn compile_trampoline<Alloc: Allocator>(runtime: &mut Runtime<Alloc>) {
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
        .compiler
        .add_function(Some("trampoline"), &lang.compile_directly(), 0)
        .unwrap();
    let function = runtime
        .compiler
        .get_function_by_name_mut("trampoline")
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
        let path = entry.path();
        let path = path.to_str().unwrap();
        let source = std::fs::read_to_string(path)?;
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
        };
        main_inner(args)?;
    }
    Ok(())
}

fn main_inner(args: CommandLineArguments) -> Result<(), Box<dyn Error>> {
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

    cfg_if::cfg_if! {
        if #[cfg(feature = "compacting")] {
            cfg_if::cfg_if! {
                if #[cfg(feature = "thread-safe")] {
                    type Alloc = MutexAllocator<CompactingHeap>;
                } else {
                    type Alloc = CompactingHeap;
                }
            }
        } else if #[cfg(feature = "simple-mark-and-sweep")] {
            cfg_if::cfg_if! {
                if #[cfg(feature = "thread-safe")] {
                    type Alloc = MutexAllocator<SimpleMarkSweepHeap>;
                } else {
                    type Alloc = SimpleMarkSweepHeap;
                }
            }
        } else if #[cfg(feature = "simple-generation")] {
            cfg_if::cfg_if! {
                if #[cfg(feature = "thread-safe")] {
                    type Alloc = MutexAllocator<SimpleGeneration>;
                } else {
                    type Alloc = SimpleGeneration;
                }
            }
        } else if #[cfg(feature = "thread-safe")] {
            type Alloc = MutexAllocator<SimpleGeneration>;
        } else {
            type Alloc = SimpleGeneration;
        }
    }

    let allocator = Alloc::new();
    let printer: Box<dyn Printer> = if has_expect {
        Box::new(TestPrinter::new(Box::new(DefaultPrinter)))
    } else {
        Box::new(DefaultPrinter)
    };

    let mut runtime = Runtime::new(args.clone(), allocator, printer);

    compile_trampoline(&mut runtime);

    runtime
        .compiler
        .set_compiler_lock_pointer(&runtime.compiler as *const _);

    runtime
        .compiler
        .set_pause_atom_ptr(runtime.pause_atom_ptr());

    runtime.install_builtins()?;

    runtime.compiler.compile("resources/std.bg")?;

    let compile_time = Instant::now();

    // TODO: Need better name for top_level
    // It should really be the top level of a namespace
    let top_level = runtime.compiler.compile(&program)?;

    runtime.compiler.check_functions();
    if args.show_times {
        println!("Compile time {:?}", compile_time.elapsed());
    }

    // TODO: Do better
    // If I'm compiling on the fly I need this to happen when I compile
    // not just here
    runtime.memory.stack_map = runtime.compiler.stack_map.clone();

    runtime.write_functions_to_pid_map();

    let time = Instant::now();
    if let Some(top_level) = top_level {
        if let Some(f) = runtime.get_function0(&top_level) {
            f();
        } else {
            panic!(
                "We are supposed to have top level, but didn't find the function {}",
                top_level
            );
        }
    }
    let fully_qualified_main = runtime.compiler.current_namespace_name() + "/main";
    if let Some(f) = runtime.get_function0(&fully_qualified_main) {
        let result = f();
        runtime.println(result as usize);
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
    };
    run_all_tests(args)?;
    Ok(())
}

// TODO:
// Runtime?
//     Function in our language names and calling them.
//     Built-in functions
//     Stack
//     Heap
// Parser
// Debugging

// TODO: Make variables
// Should we allow reassignment?
// Need to add guards against type errors
// Need to do some looping construct
// Need to do load and store in the actual
// compiler instead of cheating
// Think about protocols
// Think about how to implementing interesting
// data structures in the language itself
// Consider checked and uncheck stuff

// Bugs:
