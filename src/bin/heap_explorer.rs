//! Simple CLI tool to explore heap dumps
//!
//! Usage:
//!   cargo run --bin heap_explorer --features heap-dump -- <dump.json>
//!   cargo run --bin heap_explorer --features heap-dump -- --binary <dump.bin>

use std::env;
use std::fs::File;
use std::io::{self, BufRead, Read, Write};

use serde::{Deserialize, Serialize};

// Duplicate types from heap_dump.rs to avoid crate issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectSnapshot {
    pub tagged_ptr: String,
    pub address: String,
    pub offset: usize,
    pub tag_type: String,
    pub header_raw: String,
    pub header: HeaderSnapshot,
    pub full_size: usize,
    pub fields: Vec<FieldSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderSnapshot {
    pub type_id: u8,
    pub type_data: u32,
    pub size: u16,
    pub opaque: bool,
    pub marked: bool,
    pub large: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSnapshot {
    pub index: usize,
    pub value: String,
    pub tag: String,
    pub is_heap_ptr: bool,
    pub points_to: Option<PointerTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointerTarget {
    pub in_young: bool,
    pub in_old: bool,
    pub in_young_allocated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackSlotSnapshot {
    pub slot_index: usize,
    pub address: String,
    pub value: String,
    pub tag: String,
    pub is_heap_ptr: bool,
    pub points_to: Option<PointerTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameSnapshot {
    pub frame_pointer: String,
    pub return_address: String,
    pub stack_map_found: bool,
    pub function_name: Option<String>,
    pub number_of_locals: usize,
    pub current_stack_size: usize,
    pub max_stack_size: usize,
    pub slots: Vec<StackSlotSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadStackSnapshot {
    pub thread_index: usize,
    pub stack_base: String,
    pub frame_pointer: String,
    pub gc_return_addr: String,
    pub frames: Vec<FrameSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceSnapshot {
    pub name: String,
    pub start: String,
    pub byte_count: usize,
    pub allocation_offset: usize,
    pub objects: Vec<ObjectSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootSnapshot {
    pub source: String,
    pub index: String,
    pub value: String,
    pub tag: String,
    pub is_heap_ptr: bool,
    pub points_to: Option<PointerTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeapDump {
    pub timestamp: String,
    pub label: String,
    pub young_gen: SpaceSnapshot,
    pub old_gen: SpaceSnapshot,
    pub stacks: Vec<ThreadStackSnapshot>,
    pub roots: RootsSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootsSnapshot {
    pub temporary_roots: Vec<RootSnapshot>,
    pub namespace_roots: Vec<RootSnapshot>,
    pub thread_roots: Vec<RootSnapshot>,
    pub additional_roots: Vec<RootSnapshot>,
}

impl HeapDump {
    pub fn load(path: &str) -> io::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        serde_json::from_str(&contents).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    pub fn print_summary(&self) {
        eprintln!("=== Heap Dump: {} ===", self.label);
        eprintln!("Timestamp: {}", self.timestamp);
        eprintln!();
        eprintln!(
            "Young Gen: {} ({} bytes, {} objects)",
            self.young_gen.start,
            self.young_gen.allocation_offset,
            self.young_gen.objects.len()
        );
        eprintln!(
            "Old Gen: {} ({} objects)",
            self.old_gen.start,
            self.old_gen.objects.len()
        );
        eprintln!();
        eprintln!("Stacks: {}", self.stacks.len());
        for stack in &self.stacks {
            eprintln!(
                "  Stack[{}]: {} frames",
                stack.thread_index,
                stack.frames.len()
            );
        }
        eprintln!();
        eprintln!("Roots:");
        eprintln!("  Temporary: {}", self.roots.temporary_roots.len());
        eprintln!("  Namespace: {}", self.roots.namespace_roots.len());
        eprintln!("  Thread: {}", self.roots.thread_roots.len());
        eprintln!("  Additional: {}", self.roots.additional_roots.len());
    }

    pub fn find_young_pointers(&self) -> Vec<String> {
        let mut results = Vec::new();

        for obj in &self.old_gen.objects {
            for field in &obj.fields {
                if let Some(target) = &field.points_to {
                    if target.in_young {
                        results.push(format!(
                            "OLD_HEAP[{}].field[{}] = {} -> young",
                            obj.address, field.index, field.value
                        ));
                    }
                }
            }
        }

        for stack in &self.stacks {
            for frame in &stack.frames {
                for slot in &frame.slots {
                    if let Some(target) = &slot.points_to {
                        if target.in_young {
                            results.push(format!(
                                "STACK[{}].frame[{}].slot[{}] = {} -> young (fn={:?})",
                                stack.thread_index,
                                frame.frame_pointer,
                                slot.slot_index,
                                slot.value,
                                frame.function_name
                            ));
                        }
                    }
                }
            }
        }

        for root in &self.roots.temporary_roots {
            if let Some(target) = &root.points_to {
                if target.in_young {
                    results.push(format!(
                        "TEMP_ROOT[{}] = {} -> young",
                        root.index, root.value
                    ));
                }
            }
        }

        for root in &self.roots.namespace_roots {
            if let Some(target) = &root.points_to {
                if target.in_young {
                    results.push(format!(
                        "NAMESPACE_ROOT[{}] = {} -> young",
                        root.index, root.value
                    ));
                }
            }
        }

        for root in &self.roots.thread_roots {
            if let Some(target) = &root.points_to {
                if target.in_young {
                    results.push(format!(
                        "THREAD_ROOT[{}] = {} -> young",
                        root.index, root.value
                    ));
                }
            }
        }

        results
    }

    pub fn find_object_at(&self, addr_hex: &str) -> Option<&ObjectSnapshot> {
        let addr = addr_hex.trim_start_matches("0x");
        for obj in &self.young_gen.objects {
            if obj.address.trim_start_matches("0x") == addr {
                return Some(obj);
            }
        }
        for obj in &self.old_gen.objects {
            if obj.address.trim_start_matches("0x") == addr {
                return Some(obj);
            }
        }
        None
    }

    pub fn find_containing_object(&self, addr_hex: &str) -> Option<(&ObjectSnapshot, usize)> {
        let addr = usize::from_str_radix(addr_hex.trim_start_matches("0x"), 16).ok()?;
        for obj in &self.young_gen.objects {
            let obj_addr = usize::from_str_radix(obj.address.trim_start_matches("0x"), 16).ok()?;
            if addr >= obj_addr && addr < obj_addr + obj.full_size {
                return Some((obj, addr - obj_addr));
            }
        }
        for obj in &self.old_gen.objects {
            let obj_addr = usize::from_str_radix(obj.address.trim_start_matches("0x"), 16).ok()?;
            if addr >= obj_addr && addr < obj_addr + obj.full_size {
                return Some((obj, addr - obj_addr));
            }
        }
        None
    }
}

#[derive(Debug)]
pub struct RawMemoryDump {
    pub label: String,
    pub start_addr: usize,
    pub data: Vec<u8>,
}

impl RawMemoryDump {
    pub fn load(path: &str) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;

        if &magic != b"BGLD" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid dump file magic",
            ));
        }

        let mut addr_bytes = [0u8; 8];
        file.read_exact(&mut addr_bytes)?;
        let start_addr = usize::from_le_bytes(addr_bytes);

        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        let len = usize::from_le_bytes(len_bytes);

        let mut data = vec![0u8; len];
        file.read_exact(&mut data)?;

        Ok(RawMemoryDump {
            label: path.to_string(),
            start_addr,
            data,
        })
    }

    pub fn hexdump(&self, offset: usize, len: usize) {
        let end = (offset + len).min(self.data.len());
        for i in (offset..end).step_by(16) {
            let addr = self.start_addr + i;
            print!("{:016x}  ", addr);

            for j in 0..16 {
                if i + j < end {
                    print!("{:02x} ", self.data[i + j]);
                } else {
                    print!("   ");
                }
                if j == 7 {
                    print!(" ");
                }
            }

            print!(" |");
            for j in 0..16 {
                if i + j < end {
                    let byte = self.data[i + j];
                    if byte.is_ascii_graphic() || byte == b' ' {
                        print!("{}", byte as char);
                    } else {
                        print!(".");
                    }
                }
            }
            println!("|");
        }
    }

    pub fn read_usize(&self, offset: usize) -> Option<usize> {
        if offset + 8 <= self.data.len() {
            Some(usize::from_le_bytes(
                self.data[offset..offset + 8].try_into().unwrap(),
            ))
        } else {
            None
        }
    }

    pub fn contains_addr(&self, addr: usize) -> bool {
        addr >= self.start_addr && addr < self.start_addr + self.data.len()
    }

    pub fn addr_to_offset(&self, addr: usize) -> Option<usize> {
        if self.contains_addr(addr) {
            Some(addr - self.start_addr)
        } else {
            None
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Beagle Heap Explorer");
        eprintln!();
        eprintln!("Usage:");
        eprintln!(
            "  {} <dump.json>                      - Explore JSON dump",
            args[0]
        );
        eprintln!(
            "  {} --binary <dump.bin>              - Explore binary dump",
            args[0]
        );
        eprintln!(
            "  {} --compare <before.json> <after.json>  - Compare dumps",
            args[0]
        );
        return;
    }

    if args[1] == "--binary" {
        if args.len() < 3 {
            eprintln!("Missing binary dump file");
            return;
        }
        explore_binary(&args[2]);
    } else if args[1] == "--compare" {
        if args.len() < 4 {
            eprintln!("Need two dump files to compare");
            return;
        }
        compare_dumps(&args[2], &args[3]);
    } else {
        explore_json(&args[1]);
    }
}

fn explore_json(path: &str) {
    let dump = match HeapDump::load(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load {}: {}", path, e);
            return;
        }
    };

    dump.print_summary();

    println!("\nCommands:");
    println!("  summary              - Print summary");
    println!("  young                - List young gen objects");
    println!("  old                  - List old gen objects");
    println!("  stacks               - List all stacks and frames");
    println!("  roots                - List all roots");
    println!("  find-young           - Find pointers still pointing to young gen");
    println!("  object <addr>        - Show object at address");
    println!("  containing <addr>    - Find object containing address");
    println!("  stack <n>            - Show stack N in detail");
    println!("  frame <stack> <n>    - Show frame N of stack");
    println!("  quit                 - Exit");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break;
        }

        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "quit" | "q" | "exit" => break,
            "summary" => dump.print_summary(),
            "young" => {
                println!("Young gen objects ({}):", dump.young_gen.objects.len());
                for obj in &dump.young_gen.objects {
                    println!(
                        "  {} offset={} size={} fields={}",
                        obj.address,
                        obj.offset,
                        obj.full_size,
                        obj.fields.len()
                    );
                }
            }
            "old" => {
                println!("Old gen objects ({}):", dump.old_gen.objects.len());
                for obj in &dump.old_gen.objects {
                    println!(
                        "  {} offset={} size={} fields={}",
                        obj.address,
                        obj.offset,
                        obj.full_size,
                        obj.fields.len()
                    );
                }
            }
            "stacks" => {
                for stack in &dump.stacks {
                    println!(
                        "Stack {}: base={} fp={} ({} frames)",
                        stack.thread_index,
                        stack.stack_base,
                        stack.frame_pointer,
                        stack.frames.len()
                    );
                    for (i, frame) in stack.frames.iter().enumerate() {
                        println!(
                            "  Frame {}: {:?} ({} slots)",
                            i,
                            frame.function_name,
                            frame.slots.len()
                        );
                    }
                }
            }
            "roots" => {
                println!("Temporary roots:");
                for r in &dump.roots.temporary_roots {
                    println!("  [{}] {} ({})", r.index, r.value, r.tag);
                }
                println!("Namespace roots:");
                for r in &dump.roots.namespace_roots {
                    println!("  [{}] {} ({})", r.index, r.value, r.tag);
                }
                println!("Thread roots:");
                for r in &dump.roots.thread_roots {
                    println!("  [{}] {} ({})", r.index, r.value, r.tag);
                }
            }
            "find-young" => {
                let results = dump.find_young_pointers();
                if results.is_empty() {
                    println!("No pointers to young generation found.");
                } else {
                    println!("Found {} pointers to young gen:", results.len());
                    for r in results {
                        println!("  {}", r);
                    }
                }
            }
            "object" => {
                if parts.len() < 2 {
                    println!("Usage: object <address>");
                    continue;
                }
                if let Some(obj) = dump.find_object_at(parts[1]) {
                    println!("Object at {}:", obj.address);
                    println!("  Tagged: {}", obj.tagged_ptr);
                    println!("  Offset: {}", obj.offset);
                    println!("  Size: {} bytes", obj.full_size);
                    println!("  Header: {:?}", obj.header);
                    println!("  Fields:");
                    for f in &obj.fields {
                        let target = if let Some(t) = &f.points_to {
                            format!(
                                " -> {}",
                                if t.in_young {
                                    "YOUNG"
                                } else if t.in_old {
                                    "old"
                                } else {
                                    "?"
                                }
                            )
                        } else {
                            String::new()
                        };
                        println!("    [{}] {} ({}){}", f.index, f.value, f.tag, target);
                    }
                } else {
                    println!("No object found at {}", parts[1]);
                }
            }
            "containing" => {
                if parts.len() < 2 {
                    println!("Usage: containing <address>");
                    continue;
                }
                if let Some((obj, offset)) = dump.find_containing_object(parts[1]) {
                    println!(
                        "Address {} is at offset {} within object at {}",
                        parts[1], offset, obj.address
                    );
                    println!("  Size: {} bytes", obj.full_size);
                    println!("  Header: {:?}", obj.header);
                } else {
                    println!("No object contains address {}", parts[1]);
                }
            }
            "stack" => {
                if parts.len() < 2 {
                    println!("Usage: stack <n>");
                    continue;
                }
                if let Ok(n) = parts[1].parse::<usize>() {
                    if n < dump.stacks.len() {
                        let stack = &dump.stacks[n];
                        println!("Stack {}:", n);
                        println!("  Base: {}", stack.stack_base);
                        println!("  Frame pointer: {}", stack.frame_pointer);
                        println!("  GC return addr: {}", stack.gc_return_addr);
                        for (i, frame) in stack.frames.iter().enumerate() {
                            println!("  Frame {} ({:?}):", i, frame.function_name);
                            println!("    FP: {}", frame.frame_pointer);
                            println!("    Return addr: {}", frame.return_address);
                            println!(
                                "    Locals: {}, Stack: {}/{}",
                                frame.number_of_locals,
                                frame.current_stack_size,
                                frame.max_stack_size
                            );
                            for slot in &frame.slots {
                                let target = if let Some(t) = &slot.points_to {
                                    format!(
                                        " -> {}",
                                        if t.in_young {
                                            "YOUNG"
                                        } else if t.in_old {
                                            "old"
                                        } else {
                                            "?"
                                        }
                                    )
                                } else {
                                    String::new()
                                };
                                println!(
                                    "      [{}] {} = {} ({}){}",
                                    slot.slot_index, slot.address, slot.value, slot.tag, target
                                );
                            }
                        }
                    } else {
                        println!("Stack {} not found (have {})", n, dump.stacks.len());
                    }
                }
            }
            "frame" => {
                if parts.len() < 3 {
                    println!("Usage: frame <stack> <frame>");
                    continue;
                }
                if let (Ok(s), Ok(f)) = (parts[1].parse::<usize>(), parts[2].parse::<usize>()) {
                    if s < dump.stacks.len() && f < dump.stacks[s].frames.len() {
                        let frame = &dump.stacks[s].frames[f];
                        println!("Frame {} of stack {} ({:?}):", f, s, frame.function_name);
                        println!("  FP: {}", frame.frame_pointer);
                        println!("  Return addr: {}", frame.return_address);
                        println!("  Stack map found: {}", frame.stack_map_found);
                        println!("  Locals: {}", frame.number_of_locals);
                        println!(
                            "  Stack size: {}/{}",
                            frame.current_stack_size, frame.max_stack_size
                        );
                        println!("  Slots:");
                        for slot in &frame.slots {
                            let target = if let Some(t) = &slot.points_to {
                                format!(
                                    " -> {}",
                                    if t.in_young {
                                        "YOUNG"
                                    } else if t.in_old {
                                        "old"
                                    } else {
                                        "?"
                                    }
                                )
                            } else {
                                String::new()
                            };
                            println!(
                                "    [{}] {} = {} ({}){}",
                                slot.slot_index, slot.address, slot.value, slot.tag, target
                            );
                        }
                    } else {
                        println!("Frame not found");
                    }
                }
            }
            _ => println!("Unknown command: {}", parts[0]),
        }
    }
}

fn explore_binary(path: &str) {
    let dump = match RawMemoryDump::load(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load {}: {}", path, e);
            return;
        }
    };

    println!("Binary dump: {}", dump.label);
    println!("Start address: {:#x}", dump.start_addr);
    println!("Size: {} bytes", dump.data.len());

    println!("\nCommands:");
    println!("  hexdump <offset> <len>  - Show hexdump");
    println!("  read <offset>           - Read usize at offset");
    println!("  addr <address>          - Show data at address");
    println!("  quit                    - Exit");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break;
        }

        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "quit" | "q" | "exit" => break,
            "hexdump" => {
                let offset = parts.get(1).and_then(|s| parse_number(s)).unwrap_or(0);
                let len = parts.get(2).and_then(|s| parse_number(s)).unwrap_or(256);
                dump.hexdump(offset, len);
            }
            "read" => {
                if parts.len() < 2 {
                    println!("Usage: read <offset>");
                    continue;
                }
                if let Some(offset) = parse_number(parts[1]) {
                    if let Some(val) = dump.read_usize(offset) {
                        println!("{:#x}", val);
                    } else {
                        println!("Out of bounds");
                    }
                }
            }
            "addr" => {
                if parts.len() < 2 {
                    println!("Usage: addr <address>");
                    continue;
                }
                if let Some(addr) = parse_number(parts[1]) {
                    if let Some(offset) = dump.addr_to_offset(addr) {
                        println!("Address {:#x} is at offset {:#x}", addr, offset);
                        dump.hexdump(offset, 64);
                    } else {
                        println!("Address not in this dump");
                    }
                }
            }
            _ => println!("Unknown command: {}", parts[0]),
        }
    }
}

fn compare_dumps(before_path: &str, after_path: &str) {
    let before = match HeapDump::load(before_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load {}: {}", before_path, e);
            return;
        }
    };

    let after = match HeapDump::load(after_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load {}: {}", after_path, e);
            return;
        }
    };

    println!("=== Comparing {} vs {} ===\n", before.label, after.label);

    println!("Young Gen:");
    println!(
        "  Before: {} objects, {} bytes allocated",
        before.young_gen.objects.len(),
        before.young_gen.allocation_offset
    );
    println!(
        "  After:  {} objects, {} bytes allocated",
        after.young_gen.objects.len(),
        after.young_gen.allocation_offset
    );

    println!("\nOld Gen:");
    println!("  Before: {} objects", before.old_gen.objects.len());
    println!("  After:  {} objects", after.old_gen.objects.len());

    println!("\nStale young pointers in 'after':");
    let stale = after.find_young_pointers();
    if stale.is_empty() {
        println!("  None found");
    } else {
        for s in &stale {
            println!("  {}", s);
        }
    }

    println!("\nStack slot changes:");
    for (bi, before_stack) in before.stacks.iter().enumerate() {
        if bi >= after.stacks.len() {
            continue;
        }
        let after_stack = &after.stacks[bi];

        for (fi, before_frame) in before_stack.frames.iter().enumerate() {
            if fi >= after_stack.frames.len() {
                continue;
            }
            let after_frame = &after_stack.frames[fi];

            for (si, before_slot) in before_frame.slots.iter().enumerate() {
                if si >= after_frame.slots.len() {
                    continue;
                }
                let after_slot = &after_frame.slots[si];

                if before_slot.value != after_slot.value {
                    println!(
                        "  Stack[{}].Frame[{}({:?})].Slot[{}]: {} -> {}",
                        bi, fi, before_frame.function_name, si, before_slot.value, after_slot.value
                    );
                }
            }
        }
    }
}

fn parse_number(s: &str) -> Option<usize> {
    if s.starts_with("0x") || s.starts_with("0X") {
        usize::from_str_radix(&s[2..], 16).ok()
    } else {
        s.parse().ok()
    }
}
