//! Heap dump mechanism for GC debugging
//!
//! This module provides comprehensive snapshots of:
//! - Young generation heap
//! - Old generation heap
//! - All thread stacks
//! - All root sources (temporary roots, namespace roots, thread roots, etc.)
//!
//! Dumps are saved as JSON for easy exploration and diffing.

use std::fs::File;
use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::types::{BuiltInTypes, Header, HeapObject};

use super::StackMap;

/// A snapshot of a single heap object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectSnapshot {
    /// Tagged pointer to this object
    pub tagged_ptr: String,
    /// Untagged address
    pub address: String,
    /// Offset from space start
    pub offset: usize,
    /// The tag type (Closure, HeapObject, Float)
    pub tag_type: String,
    /// Raw header value
    pub header_raw: String,
    /// Parsed header
    pub header: HeaderSnapshot,
    /// Size in bytes (including header)
    pub full_size: usize,
    /// Field values (as hex strings)
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
    /// If this is a heap pointer, where does it point?
    pub points_to: Option<PointerTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointerTarget {
    pub in_young: bool,
    pub in_old: bool,
    pub in_young_allocated: bool,
}

/// A snapshot of a stack slot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackSlotSnapshot {
    pub slot_index: usize,
    pub address: String,
    pub value: String,
    pub tag: String,
    pub is_heap_ptr: bool,
    pub points_to: Option<PointerTarget>,
}

/// A snapshot of a stack frame
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

/// A snapshot of a thread's stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadStackSnapshot {
    pub thread_index: usize,
    pub stack_base: String,
    pub frame_pointer: String,
    pub gc_return_addr: String,
    pub frames: Vec<FrameSnapshot>,
}

/// A snapshot of a memory space (young or old gen)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceSnapshot {
    pub name: String,
    pub start: String,
    pub byte_count: usize,
    pub allocation_offset: usize,
    pub objects: Vec<ObjectSnapshot>,
}

/// Complete heap dump
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeapDump {
    pub timestamp: String,
    pub label: String,
    pub young_gen: SpaceSnapshot,
    pub old_gen: SpaceSnapshot,
    pub stacks: Vec<ThreadStackSnapshot>,
}

/// Helper to determine where a pointer points
pub struct PointerClassifier {
    young_start: usize,
    young_end: usize,
    young_allocated_end: usize,
    old_start: usize,
    old_end: usize,
}

impl PointerClassifier {
    pub fn new(
        young_start: usize,
        young_byte_count: usize,
        young_allocation_offset: usize,
        old_start: usize,
        old_byte_count: usize,
    ) -> Self {
        Self {
            young_start,
            young_end: young_start + young_byte_count,
            young_allocated_end: young_start + young_allocation_offset,
            old_start,
            old_end: old_start + old_byte_count,
        }
    }

    pub fn classify(&self, tagged_value: usize) -> Option<PointerTarget> {
        if !BuiltInTypes::is_heap_pointer(tagged_value) {
            return None;
        }
        let untagged = BuiltInTypes::untag(tagged_value);
        Some(PointerTarget {
            in_young: untagged >= self.young_start && untagged < self.young_end,
            in_old: untagged >= self.old_start && untagged < self.old_end,
            in_young_allocated: untagged >= self.young_start && untagged < self.young_allocated_end,
        })
    }
}

fn tag_name(value: usize) -> String {
    match BuiltInTypes::get_kind(value) {
        BuiltInTypes::Int => "Int".to_string(),
        BuiltInTypes::Float => "Float".to_string(),
        BuiltInTypes::String => "String".to_string(),
        BuiltInTypes::Bool => "Bool".to_string(),
        BuiltInTypes::Function => "Function".to_string(),
        BuiltInTypes::Closure => "Closure".to_string(),
        BuiltInTypes::HeapObject => "HeapObject".to_string(),
        BuiltInTypes::Null => "Null".to_string(),
    }
}

/// Walk objects in a memory region
pub fn walk_space_objects(
    start: *const u8,
    allocation_offset: usize,
    classifier: &PointerClassifier,
) -> Vec<ObjectSnapshot> {
    let mut objects = Vec::new();
    let mut offset = 0;

    while offset < allocation_offset {
        let ptr = unsafe { start.add(offset) };

        // Read header to determine object size
        let header_raw = unsafe { *(ptr as *const usize) };
        let header = Header::from_usize(header_raw);

        // Calculate full size
        let header_size = if header.large { 16 } else { 8 };
        let fields_size = if header.large {
            let size_ptr = unsafe { (ptr as *const usize).add(1) };
            unsafe { *size_ptr * 8 }
        } else {
            header.size as usize * 8
        };
        let full_size = header_size + fields_size;

        // Skip if size is 0 or unreasonable
        if full_size == 0 || full_size > allocation_offset - offset {
            offset += 8; // Move forward
            continue;
        }

        let heap_obj = HeapObject::from_untagged(ptr);

        // Collect field snapshots
        let mut field_snapshots = Vec::new();
        if !header.opaque {
            let fields = heap_obj.get_fields();
            for (i, &field_value) in fields.iter().enumerate() {
                field_snapshots.push(FieldSnapshot {
                    index: i,
                    value: format!("{:#x}", field_value),
                    tag: tag_name(field_value),
                    is_heap_ptr: BuiltInTypes::is_heap_pointer(field_value),
                    points_to: classifier.classify(field_value),
                });
            }
        }

        objects.push(ObjectSnapshot {
            tagged_ptr: format!("{:#x}", BuiltInTypes::HeapObject.tag(ptr as isize)),
            address: format!("{:#x}", ptr as usize),
            offset,
            tag_type: "HeapObject".to_string(),
            header_raw: format!("{:#x}", header_raw),
            header: HeaderSnapshot {
                type_id: header.type_id,
                type_data: header.type_data,
                size: header.size,
                opaque: header.opaque,
                marked: header.marked,
                large: header.large,
            },
            full_size,
            fields: field_snapshots,
        });

        offset += full_size;
        // Align to 8 bytes
        offset = (offset + 7) & !7;
    }

    objects
}

/// Snapshot a thread's stack
pub fn snapshot_stack(
    thread_index: usize,
    stack_base: usize,
    frame_pointer: usize,
    gc_return_addr: usize,
    stack_map: &StackMap,
    classifier: &PointerClassifier,
) -> ThreadStackSnapshot {
    let mut frames = Vec::new();
    let mut fp = frame_pointer;
    let mut pending_return_addr = gc_return_addr;

    while fp != 0 && fp < stack_base {
        let caller_fp = unsafe { *(fp as *const usize) };
        let return_addr_for_caller = unsafe { *((fp + 8) as *const usize) };

        let stack_map_result = if pending_return_addr != 0 {
            stack_map.find_stack_data_no_debug(pending_return_addr)
        } else {
            None
        };

        let mut slots = Vec::new();

        if let Some(details) = &stack_map_result {
            let active_slots = details.number_of_locals + details.current_stack_size;
            for i in 0..active_slots {
                let slot_addr = fp - 8 - (i * 8);
                let slot_value = unsafe { *(slot_addr as *const usize) };

                slots.push(StackSlotSnapshot {
                    slot_index: i,
                    address: format!("{:#x}", slot_addr),
                    value: format!("{:#x}", slot_value),
                    tag: tag_name(slot_value),
                    is_heap_ptr: BuiltInTypes::is_heap_pointer(slot_value),
                    points_to: classifier.classify(slot_value),
                });
            }
        }

        frames.push(FrameSnapshot {
            frame_pointer: format!("{:#x}", fp),
            return_address: format!("{:#x}", pending_return_addr),
            stack_map_found: stack_map_result.is_some(),
            function_name: stack_map_result
                .as_ref()
                .and_then(|d| d.function_name.clone()),
            number_of_locals: stack_map_result
                .as_ref()
                .map(|d| d.number_of_locals)
                .unwrap_or(0),
            current_stack_size: stack_map_result
                .as_ref()
                .map(|d| d.current_stack_size)
                .unwrap_or(0),
            max_stack_size: stack_map_result
                .as_ref()
                .map(|d| d.max_stack_size)
                .unwrap_or(0),
            slots,
        });

        if caller_fp != 0 && caller_fp <= fp {
            break;
        }

        fp = caller_fp;
        pending_return_addr = return_addr_for_caller;
    }

    ThreadStackSnapshot {
        thread_index,
        stack_base: format!("{:#x}", stack_base),
        frame_pointer: format!("{:#x}", frame_pointer),
        gc_return_addr: format!("{:#x}", gc_return_addr),
        frames,
    }
}

/// Raw binary dump of a memory region
#[derive(Debug)]
pub struct RawMemoryDump {
    pub label: String,
    pub start_addr: usize,
    pub data: Vec<u8>,
}

impl RawMemoryDump {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::BufWriter;

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header: magic, start address, length
        writer.write_all(b"BGLD")?; // Beagle Dump magic
        writer.write_all(&self.start_addr.to_le_bytes())?;
        writer.write_all(&self.data.len().to_le_bytes())?;
        writer.write_all(&self.data)?;

        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;

        if &magic != b"BGLD" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
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

    /// Print hexdump of a range
    pub fn hexdump(&self, offset: usize, len: usize) {
        let end = (offset + len).min(self.data.len());
        for i in (offset..end).step_by(16) {
            let addr = self.start_addr + i;
            print!("{:016x}  ", addr);

            // Hex bytes
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

            // ASCII
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

    /// Read a usize at offset
    pub fn read_usize(&self, offset: usize) -> Option<usize> {
        if offset + 8 <= self.data.len() {
            Some(usize::from_le_bytes(
                self.data[offset..offset + 8].try_into().unwrap(),
            ))
        } else {
            None
        }
    }

    /// Find an address within this dump
    pub fn contains_addr(&self, addr: usize) -> bool {
        addr >= self.start_addr && addr < self.start_addr + self.data.len()
    }

    /// Get offset for an address
    pub fn addr_to_offset(&self, addr: usize) -> Option<usize> {
        if self.contains_addr(addr) {
            Some(addr - self.start_addr)
        } else {
            None
        }
    }
}

/// Complete binary dump set (young gen, old gen, stacks)
pub struct BinaryDumpSet {
    pub young_gen: RawMemoryDump,
    pub old_gen: RawMemoryDump,
    pub stacks: Vec<RawMemoryDump>,
}

impl BinaryDumpSet {
    pub fn save_all(&self, dir: &str, prefix: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;

        self.young_gen
            .save(&format!("{}/{}_young.bin", dir, prefix))?;
        self.old_gen.save(&format!("{}/{}_old.bin", dir, prefix))?;

        for (i, stack) in self.stacks.iter().enumerate() {
            stack.save(&format!("{}/{}_stack_{}.bin", dir, prefix, i))?;
        }

        Ok(())
    }
}

impl HeapDump {
    /// Save the dump to a JSON file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load a dump from a JSON file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let dump: HeapDump = serde_json::from_str(&contents)?;
        Ok(dump)
    }

    /// Print a summary of the dump
    pub fn print_summary(&self) {
        eprintln!("=== Heap Dump: {} ===", self.label);
        eprintln!("Timestamp: {}", self.timestamp);
        eprintln!();

        eprintln!(
            "Young Gen: {} - {}",
            self.young_gen.start,
            format!(
                "{:#x}",
                usize::from_str_radix(&self.young_gen.start[2..], 16).unwrap()
                    + self.young_gen.byte_count
            )
        );
        eprintln!(
            "  Allocated: {} bytes ({} objects)",
            self.young_gen.allocation_offset,
            self.young_gen.objects.len()
        );

        eprintln!(
            "Old Gen: {} (allocated portion not tracked here)",
            self.old_gen.start
        );
        eprintln!("  Objects: {}", self.old_gen.objects.len());

        eprintln!();
        eprintln!("Stacks: {}", self.stacks.len());
        for stack in &self.stacks {
            eprintln!(
                "  Stack[{}]: {} frames",
                stack.thread_index,
                stack.frames.len()
            );
        }
    }

    /// Find all pointers that point to young gen
    pub fn find_young_pointers(&self) -> Vec<String> {
        let mut results = Vec::new();

        // Check old gen objects
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

        // Check stacks
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

        results
    }

    /// Find objects at a specific address
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

    /// Check if an address is a valid object start
    pub fn is_valid_object_start(&self, addr_hex: &str) -> bool {
        self.find_object_at(addr_hex).is_some()
    }

    /// Find what object contains an address (if pointing to middle of object)
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
