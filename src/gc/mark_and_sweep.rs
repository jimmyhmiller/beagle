use std::{error::Error, ffi::c_void, io, io::Write};

use libc::mprotect;

use super::get_page_size;

use crate::builtins::reset_shift::ContinuationObject;
use crate::types::{BuiltInTypes, Header, HeapObject, Word};

use super::{AllocateAction, Allocator, AllocatorOptions, stack_walker::StackWalker};
use crate::collections::{TYPE_ID_PERSISTENT_VEC, TYPE_ID_PERSISTENT_VEC_NODE};

const DEFAULT_PAGE_COUNT: usize = 1024;
// Aribtary number that should be changed when I have
// better options for gc
const MAX_PAGE_COUNT: usize = 1000000;

struct Space {
    start: *const u8,
    page_count: usize,
    highmark: usize,
    #[allow(unused)]
    protected: bool,
}

unsafe impl Send for Space {}
unsafe impl Sync for Space {}

impl Space {
    #[allow(unused)]
    fn word_count(&self) -> usize {
        (self.page_count * get_page_size()) / 8
    }

    fn byte_count(&self) -> usize {
        self.page_count * get_page_size()
    }

    fn contains(&self, pointer: *const u8) -> bool {
        let start = self.start as usize;
        let end = start + self.byte_count();
        let pointer = pointer as usize;
        pointer >= start && pointer < end
    }

    fn copy_data_to_offset(&mut self, offset: usize, data: &[u8]) -> isize {
        unsafe {
            let start = self.start.add(offset);
            let new_pointer = start as isize;
            std::ptr::copy_nonoverlapping(data.as_ptr(), start as *mut u8, data.len());
            new_pointer
        }
    }

    fn write_object(&mut self, offset: usize, size: Word) -> *const u8 {
        let mut heap_object = HeapObject::from_untagged(unsafe { self.start.add(offset) });

        assert!(self.contains(heap_object.get_pointer()));

        heap_object.write_header(size);

        heap_object.get_pointer()
    }

    #[allow(unused)]
    fn protect(&mut self) {
        unsafe {
            mprotect(
                self.start as *mut _,
                self.byte_count() - 1024,
                libc::PROT_NONE,
            )
        };
        self.protected = true;
    }

    #[allow(unused)]
    fn unprotect(&mut self) {
        unsafe {
            mprotect(
                self.start as *mut _,
                self.byte_count() - 1024,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        };

        self.protected = false;
    }

    fn new(default_page_count: usize) -> Self {
        let pre_allocated_space = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                get_page_size() * MAX_PAGE_COUNT,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        Self::commit_memory(pre_allocated_space, default_page_count * get_page_size()).unwrap();

        Self {
            start: pre_allocated_space as *const u8,
            page_count: default_page_count,
            highmark: 0,
            protected: false,
        }
    }

    fn commit_memory(addr: *mut c_void, size: usize) -> Result<(), io::Error> {
        unsafe {
            if mprotect(addr, size, libc::PROT_READ | libc::PROT_WRITE) != 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        }
    }

    fn double_committed_memory(&mut self) {
        let new_page_count = self.page_count * 2;
        Self::commit_memory(self.start as *mut c_void, new_page_count * get_page_size()).unwrap();
        self.page_count = new_page_count;
    }

    fn update_highmark(&mut self, highmark: usize) {
        if highmark > self.highmark {
            self.highmark = highmark;
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct FreeListEntry {
    offset: usize,
    size: usize,
}

impl FreeListEntry {
    pub fn end(&self) -> usize {
        self.offset + self.size
    }

    pub fn can_hold(&self, size: usize) -> bool {
        self.size >= size
    }

    pub fn contains(&self, offset: usize) -> bool {
        self.offset <= offset && offset < self.end()
    }
}

pub struct FreeList {
    ranges: Vec<FreeListEntry>, // always sorted by start
    next_fit_index: usize,
}

impl FreeList {
    fn new(starting_range: FreeListEntry) -> Self {
        FreeList {
            ranges: vec![starting_range],
            next_fit_index: 0,
        }
    }

    fn empty_with_capacity(capacity: usize) -> Self {
        FreeList {
            ranges: Vec::with_capacity(capacity),
            next_fit_index: 0,
        }
    }

    fn insert(&mut self, range: FreeListEntry) {
        let mut i = match self
            .ranges
            .binary_search_by_key(&range.offset, |r| r.offset)
        {
            Ok(i) | Err(i) => i,
        };

        // Coalesce with previous if adjacent
        if i > 0 && self.ranges[i - 1].end() == range.offset {
            i -= 1;
            self.ranges[i].size += range.size;
        } else {
            self.ranges.insert(i, range);
        }

        // Coalesce with next if adjacent
        if i + 1 < self.ranges.len() && self.ranges[i].end() == self.ranges[i + 1].offset {
            self.ranges[i].size += self.ranges[i + 1].size;
            self.ranges.remove(i + 1);
        }

        if !self.ranges.is_empty() {
            self.next_fit_index = self.next_fit_index.min(self.ranges.len() - 1);
        } else {
            self.next_fit_index = 0;
        }
    }

    fn allocate(&mut self, size: usize) -> Option<usize> {
        if self.ranges.is_empty() {
            self.next_fit_index = 0;
            return None;
        }

        let start = self.next_fit_index.min(self.ranges.len() - 1);
        for step in 0..self.ranges.len() {
            let i = (start + step) % self.ranges.len();
            let r = &mut self.ranges[i];
            if r.can_hold(size) {
                let addr = r.offset;
                if !addr.is_multiple_of(8) {
                    panic!(
                        "GC internal error: free list entry at {:#x} is not 8-byte aligned",
                        addr
                    );
                }

                r.offset += size;
                r.size -= size;

                if r.size == 0 {
                    self.ranges.remove(i);
                }

                if self.ranges.is_empty() {
                    self.next_fit_index = 0;
                } else {
                    self.next_fit_index = i.min(self.ranges.len() - 1);
                }

                return Some(addr);
            }
        }
        None
    }

    fn find_entry_contains(&self, offset: usize) -> Option<&FreeListEntry> {
        self.ranges.iter().find(|&entry| entry.contains(offset))
    }

    fn trailing_free_entry(&self) -> Option<&FreeListEntry> {
        self.ranges.last()
    }

    fn append_sorted_coalesced(&mut self, range: FreeListEntry) {
        if range.size == 0 {
            return;
        }

        if let Some(last) = self.ranges.last_mut() {
            if last.end() == range.offset {
                last.size += range.size;
                return;
            }

            assert!(
                last.end() < range.offset,
                "GC internal error: free list append out of order ({:#x}..{:#x}) then ({:#x}..{:#x})",
                last.offset,
                last.end(),
                range.offset,
                range.end()
            );
        }

        self.ranges.push(range);
    }

    fn prefer_trailing_range(&mut self) {
        if !self.ranges.is_empty() {
            self.next_fit_index = self.ranges.len() - 1;
        }
    }

    fn total_free_bytes(&self) -> usize {
        self.ranges.iter().map(|range| range.size).sum()
    }
}

pub struct MarkAndSweep {
    space: Space,
    free_list: FreeList,
    options: AllocatorOptions,
}

struct PendingMark {
    object_ptr: usize,
    source_addr: usize,
    source_kind: &'static str,
}

fn should_log_suspicious_mark(object: &HeapObject) -> bool {
    if std::env::var("BEAGLE_DEBUG_SUSPECT_MARK").is_err() {
        return false;
    }
    let header = object.get_header();
    header.type_id == 0 && header.type_data == 0 && header.size == 0
}

fn log_suspicious_mark(pending: &PendingMark) {
    let object = HeapObject::from_tagged(pending.object_ptr);
    if !should_log_suspicious_mark(&object) {
        return;
    }

    let raw_header = unsafe { *(object.untagged() as *const usize) };
    let header = object.get_header();
    eprintln!(
        "[suspect-mark] object={:#x} raw_header={:#x} header=(type_id={} type_data={} size={} flags={} marked={} opaque={} large={}) source_kind={} source_addr={:#x}",
        pending.object_ptr,
        raw_header,
        header.type_id,
        header.type_data,
        header.size,
        header.type_flags,
        header.marked,
        header.opaque,
        header.large,
        pending.source_kind,
        pending.source_addr
    );
}

fn vec_gc_log_line(line: impl AsRef<str>) {
    let Ok(path) = std::env::var("BEAGLE_DEBUG_VEC_GC_FILE") else {
        return;
    };
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    else {
        return;
    };
    let _ = writeln!(file, "{}", line.as_ref());
}

// TODO: I got an issue with my freelist
impl MarkAndSweep {
    fn log_bad_continuation_closures(&mut self, phase: &str) {
        let mut bad_count = 0usize;
        self.walk_objects_mut(|_, heap_obj| {
            let object = HeapObject::from_untagged(heap_obj.untagged() as *const u8);
            if object.get_type_id() != 0 {
                return;
            }
            let function_tagged = object.get_field(0);
            if BuiltInTypes::get_kind(function_tagged) != BuiltInTypes::Function {
                return;
            }
            let function_ptr = BuiltInTypes::untag(function_tagged) as *const u8;
            let is_continuation_trampoline = crate::get_runtime()
                .get()
                .get_function_by_pointer(function_ptr)
                .map(|f| f.name == "beagle.builtin/continuation-trampoline")
                .unwrap_or(false);
            if !is_continuation_trampoline {
                return;
            }
            let field3 = object.get_field(3);
            if BuiltInTypes::get_kind(field3) != BuiltInTypes::HeapObject {
                bad_count += 1;
                eprintln!(
                    "[ms-bad-closure] phase={} closure={:#x} field3={:#x} tag={:?}",
                    phase,
                    BuiltInTypes::Closure.tag(object.untagged() as isize) as usize,
                    field3,
                    BuiltInTypes::get_kind(field3)
                );
            }
        });
        if bad_count != 0 {
            eprintln!(
                "[ms-bad-closure-summary] phase={} count={}",
                phase, bad_count
            );
        }
    }

    fn push_continuation_segment_children(
        &self,
        object: &HeapObject,
        to_mark: &mut Vec<PendingMark>,
    ) {
        let object = HeapObject::from_untagged(object.untagged() as *const u8);
        let cont_tagged = BuiltInTypes::HeapObject.tag(object.untagged() as isize) as usize;
        let Some(cont) = ContinuationObject::from_heap_object(object) else {
            return;
        };
        let Some((segment_base, segment_top, gc_frame_top)) = cont.segment_gc_frame_info() else {
            return;
        };
        eprintln!(
            "[ms-cont-segment] cont={:#x} segment={:#x}..{:#x} gc_top={:#x}",
            cont_tagged,
            segment_base,
            segment_top,
            gc_frame_top
        );
        let mut push_slot = |slot_addr, pointer| {
            let untagged = BuiltInTypes::untag(pointer);
            if untagged != 0 && untagged.is_multiple_of(8) {
                to_mark.push(PendingMark {
                    object_ptr: pointer,
                    source_addr: slot_addr,
                    source_kind: "captured-segment-root",
                });
            }
        };
        if gc_frame_top >= segment_base && gc_frame_top < segment_top {
            StackWalker::walk_segment_gc_roots(
                gc_frame_top,
                segment_base,
                segment_top,
                &mut push_slot,
            );
        }
        // Restore relative offsets after GC scanning
        cont.make_fp_links_relative_again();
    }

    /// Check if a pointer is within this allocator's space
    pub fn contains(&self, pointer: *const u8) -> bool {
        self.space.contains(pointer)
    }

    /// Get the start address of this heap space
    pub fn heap_start(&self) -> usize {
        self.space.start as usize
    }

    /// Get the size of this heap space in bytes
    pub fn heap_size(&self) -> usize {
        self.space.byte_count()
    }

    pub fn free_bytes(&self) -> usize {
        self.free_list.total_free_bytes()
    }

    fn shrink_highmark_if_tail_is_free(&mut self) {
        let Some(trailing_free) = self.free_list.trailing_free_entry() else {
            return;
        };

        if trailing_free.end() != self.space.byte_count() {
            return;
        }

        // `highmark` only needs to be somewhere after the start of the highest live object
        // and before the start of the trailing free range. Using `offset - 8` keeps all
        // existing `offset > highmark` scans correct while dropping dead tail space.
        self.space.highmark = trailing_free.offset.saturating_sub(8);
    }

    fn allocate_inner(
        &mut self,
        words: Word,
        data: Option<&[u8]>,
        kind: crate::types::BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        // Large objects need 16-byte header, small objects need 8-byte header
        let header_size = if words.to_words() > Header::MAX_INLINE_SIZE {
            16
        } else {
            8
        };
        let size_bytes = words.to_bytes() + header_size;

        let offset = self.free_list.allocate(size_bytes);
        if let Some(offset) = offset {
            let start = self.space.start as usize;
            let ptr_addr = start + offset;
            let ptr_end = ptr_addr + size_bytes;
            vec_gc_log_line(format!(
                "[ms-alloc] kind={:?} offset={:#x} ptr={:#x}..{:#x} size_bytes={:#x}",
                kind, offset, ptr_addr, ptr_end, size_bytes
            ));
            let watch_start = 0x7000060340usize;
            let watch_end = watch_start + 0x40;
            if ptr_addr < watch_end && ptr_end > watch_start {
                eprintln!(
                    "[ms-alloc-overlap] kind={:?} offset={:#x} size_bytes={:#x} ptr={:#x}..{:#x} watch={:#x}..{:#x}",
                    kind, offset, size_bytes, ptr_addr, ptr_end, watch_start, watch_end
                );
            }
            self.space.update_highmark(offset);
            if let Some(data) = data {
                // When data is provided, copy it directly without first writing
                // a temporary non-opaque header via write_object. The data already
                // contains the correct header (e.g., opaque for floats).
                assert_eq!(
                    data.len(),
                    size_bytes,
                    "data.len()={} != size_bytes={} (words={}, header_size={})",
                    data.len(),
                    size_bytes,
                    words.to_words(),
                    header_size
                );
                let pointer = unsafe { self.space.start.add(offset) };
                assert!(self.space.contains(pointer));
                self.space.copy_data_to_offset(offset, data);
                let obj = HeapObject::from_untagged(pointer);
                vec_gc_log_line(format!(
                    "[ms-alloc-data] ptr={:#x} type_id={} header={:#x} full_size={:#x}",
                    pointer as usize,
                    obj.get_header().type_id,
                    unsafe { *(pointer as *const usize) },
                    obj.full_size()
                ));
                return Ok(AllocateAction::Allocated(pointer));
            }
            let pointer = self.space.write_object(offset, words);
            // Float objects are opaque (their field is a raw f64, not a pointer).
            // Set the opaque bit immediately so GC never sees a non-opaque float.
            if kind == crate::types::BuiltInTypes::Float {
                unsafe {
                    *(pointer as *mut usize) |= 0x2; // Set opaque bit (bit 1)
                }
            }
            return Ok(AllocateAction::Allocated(pointer));
        }

        Ok(AllocateAction::Gc)
    }

    #[allow(unused)]
    pub fn copy_data_to_offset(&mut self, data: &[u8]) -> *const u8 {
        // TODO: I could amortize this by copying lazily and coalescing
        // the copies together if they are continuous

        let Some(offset) = self.free_list.allocate(data.len()) else {
            self.grow();
            return self.copy_data_to_offset(data);
        };
        let start = self.space.start as usize;
        let ptr_addr = start + offset;
        let ptr_end = ptr_addr + data.len();
        let watch_start = 0x7000060340usize;
        let watch_end = watch_start + 0x40;
        if ptr_addr < watch_end && ptr_end > watch_start {
            eprintln!(
                "[ms-copy-overlap] offset={:#x} len={:#x} ptr={:#x}..{:#x} watch={:#x}..{:#x}",
                offset,
                data.len(),
                ptr_addr,
                ptr_end,
                watch_start,
                watch_end
            );
        }
        self.space.update_highmark(offset);
        let pointer = self.space.copy_data_to_offset(offset, data) as *const u8;
        assert!(self.space.contains(pointer));
        let obj = HeapObject::from_untagged(pointer);
        vec_gc_log_line(format!(
            "[ms-copy-alloc] ptr={:#x} type_id={} header={:#x} full_size={:#x}",
            pointer as usize,
            obj.get_header().type_id,
            unsafe { *(pointer as *const usize) },
            obj.full_size()
        ));
        pointer
    }

    fn mark_from_chain(&self, gc_frame_top: usize) {
        let mut to_mark: Vec<PendingMark> = Vec::with_capacity(128);
        let push_root = |to_mark: &mut Vec<PendingMark>,
                         value: usize,
                         source_addr: usize,
                         source_kind: &'static str| {
            if let Some(object) = HeapObject::try_from_tagged(value) {
                to_mark.push(PendingMark {
                    object_ptr: object.tagged_pointer(),
                    source_addr,
                    source_kind,
                });
            }
        };

        StackWalker::walk_stack_roots(gc_frame_top, |slot_addr, pointer| {
            push_root(&mut to_mark, pointer, slot_addr, "stack-root");
        });

        while let Some(pending) = to_mark.pop() {
            log_suspicious_mark(&pending);
            let untagged = BuiltInTypes::untag(pending.object_ptr);
            if untagged == 0 || !self.contains(untagged as *const u8) {
                if pending.source_kind == "heap-child"
                    && BuiltInTypes::is_heap_pointer(pending.source_addr)
                {
                    let source = HeapObject::from_tagged(pending.source_addr);
                    let header = source.get_header();
                    let preview = source
                        .get_fields()
                        .iter()
                        .take(8)
                        .map(|field| format!("{:#x}", field))
                        .collect::<Vec<_>>()
                        .join(" ");
                    eprintln!(
                        "[ms-invalid-source] source={:#x} type={:?} type_id={} size={} header={:#x} fields={}",
                        pending.source_addr,
                        source.get_object_type(),
                        header.type_id,
                        header.size,
                        unsafe { *(source.untagged() as *const usize) },
                        preview
                    );
                }
                eprintln!(
                    "[ms-invalid-pending] object={:#x} untagged={:#x} source_kind={} source_addr={:#x}",
                    pending.object_ptr,
                    untagged,
                    pending.source_kind,
                    pending.source_addr
                );
                panic!("mark_from_chain queued pointer outside old space");
            }
            let object = HeapObject::from_tagged(pending.object_ptr);
            if object.get_object_type() == Some(BuiltInTypes::Closure)
                && crate::get_runtime()
                    .get()
                    .get_function_by_pointer(BuiltInTypes::untag(object.get_field(0)) as *const u8)
                    .map(|f| f.name == "beagle.builtin/continuation-trampoline")
                    .unwrap_or(false)
            {
                eprintln!(
                    "[ms-mark-closure] object={:#x} source_kind={} source_addr={:#x} header={:#x} size={}",
                    pending.object_ptr,
                    pending.source_kind,
                    pending.source_addr,
                    unsafe { *(object.untagged() as *const usize) },
                    object.get_header().size
                );
                if pending.source_kind == "stack-root" {
                    let mut header_addr = gc_frame_top;
                    while header_addr != 0 {
                        let header_value = unsafe { *(header_addr as *const usize) };
                        let header = Header::from_usize(header_value);
                        if header.type_id != crate::collections::TYPE_ID_FRAME {
                            break;
                        }
                        let num_slots = header.size as usize;
                        let low = header_addr.saturating_sub(24 + num_slots.saturating_sub(1) * 8);
                        let high = header_addr.saturating_sub(24);
                        if pending.source_addr >= low && pending.source_addr <= high {
                            let fp = header_addr + 8;
                            let ret = unsafe { *((fp + 8) as *const usize) };
                            let fn_name = crate::get_runtime()
                                .get()
                                .get_function_containing_pointer(ret as *const u8)
                                .map(|(function, offset)| format!("{}+{:#x}", function.name, offset))
                                .unwrap_or_else(|| "unknown".to_string());
                            let slot_index = (high - pending.source_addr) / 8;
                            eprintln!(
                                "[ms-mark-closure-stack-owner] object={:#x} slot={:#x} header={:#x} fp={:#x} slot_index={} fn={}",
                                pending.object_ptr,
                                pending.source_addr,
                                header_addr,
                                fp,
                                slot_index,
                                fn_name
                            );
                            break;
                        }
                        let prev = unsafe { *((header_addr - 8) as *const usize) };
                        if prev == 0 {
                            break;
                        }
                        header_addr = prev;
                    }
                } else if pending.source_kind == "heap-child"
                    && BuiltInTypes::is_heap_pointer(pending.source_addr)
                {
                    let parent = HeapObject::from_tagged(pending.source_addr);
                    let parent_fn = if parent.get_object_type() == Some(BuiltInTypes::Closure) {
                        crate::get_runtime()
                            .get()
                            .get_function_by_pointer(
                                BuiltInTypes::untag(parent.get_field(0)) as *const u8,
                            )
                            .map(|f| f.name.clone())
                    } else {
                        None
                    };
                    let parent_fields = parent
                        .get_fields()
                        .iter()
                        .take(8)
                        .map(|field| format!("{:#x}", field))
                        .collect::<Vec<_>>()
                        .join(" ");
                    eprintln!(
                        "[ms-mark-closure-parent] object={:#x} parent={:#x} parent_type={:?} parent_fn={:?} parent_header={:#x} parent_fields={}",
                        pending.object_ptr,
                        pending.source_addr,
                        parent.get_object_type(),
                        parent_fn,
                        unsafe { *(parent.untagged() as *const usize) },
                        parent_fields
                    );
                }
            }
            if object.marked() {
                continue;
            }

            object.mark();
            if std::env::var("BEAGLE_DEBUG_VEC_GC").is_ok() {
                let header = object.get_header();
                if header.type_id == TYPE_ID_PERSISTENT_VEC {
                    eprintln!(
                        "[ms-mark-vec] object={:#x} count={:#x} shift={:#x} root={:#x} tail={:#x}",
                        pending.object_ptr,
                        object.get_field(0),
                        object.get_field(1),
                        object.get_field(2),
                        object.get_field(3)
                    );
                } else if header.type_id == TYPE_ID_PERSISTENT_VEC_NODE {
                    let fields = object
                        .get_fields()
                        .iter()
                        .take(4)
                        .map(|field| format!("{:#x}", field))
                        .collect::<Vec<_>>()
                        .join(" ");
                    eprintln!(
                        "[ms-mark-vec-node] object={:#x} header={:#x} fields={}",
                        pending.object_ptr,
                        unsafe { *(object.untagged() as *const usize) },
                        fields
                    );
                }
            }
            let header = object.get_header();
            if header.type_id == TYPE_ID_PERSISTENT_VEC {
                vec_gc_log_line(format!(
                    "[ms-mark-vec] object={:#x} count={:#x} shift={:#x} root={:#x} tail={:#x}",
                    pending.object_ptr,
                    object.get_field(0),
                    object.get_field(1),
                    object.get_field(2),
                    object.get_field(3)
                ));
            } else if header.type_id == TYPE_ID_PERSISTENT_VEC_NODE {
                let fields = object.get_fields();
                vec_gc_log_line(format!(
                    "[ms-mark-vec-node] object={:#x} header={:#x} f0={:#x} f1={:#x} f2={:#x} f3={:#x}",
                    pending.object_ptr,
                    unsafe { *(object.untagged() as *const usize) },
                    fields.first().copied().unwrap_or(0),
                    fields.get(1).copied().unwrap_or(0),
                    fields.get(2).copied().unwrap_or(0),
                    fields.get(3).copied().unwrap_or(0)
                ));
            }
            for (field_index, child) in object.get_heap_references().into_iter().enumerate() {
                let child_untagged = BuiltInTypes::untag(child.tagged_pointer());
                let watch_start = 0x7000060340usize;
                let watch_end = watch_start + 0x40;
                if child_untagged >= watch_start && child_untagged < watch_end {
                    eprintln!(
                        "[ms-watch-child] parent={:#x} parent_type={:?} field_index={} child={:#x} child_untagged={:#x}",
                        pending.object_ptr,
                        object.get_object_type(),
                        field_index,
                        child.tagged_pointer(),
                        child_untagged
                    );
                }
                to_mark.push(PendingMark {
                    object_ptr: child.tagged_pointer(),
                    source_addr: pending.object_ptr,
                    source_kind: "heap-child",
                });
            }
            self.push_continuation_segment_children(&object, &mut to_mark);
        }
    }

    /// Mark extra roots from shadow stacks (HandleScope handles).
    /// These are heap pointers stored in Rust-side Vec buffers.
    fn mark_extra_roots(&self, extra_roots: &[(*mut usize, usize)]) {
        let mut to_mark: Vec<PendingMark> = Vec::new();
        for &(slot_addr, _cached_value) in extra_roots {
            let value = unsafe { *slot_addr };
            if BuiltInTypes::is_heap_pointer(value) && BuiltInTypes::untag(value) != 0 {
                to_mark.push(PendingMark {
                    object_ptr: value,
                    source_addr: slot_addr as usize,
                    source_kind: "extra-root",
                });
            }
        }
        while let Some(pending) = to_mark.pop() {
            log_suspicious_mark(&pending);
            let untagged = BuiltInTypes::untag(pending.object_ptr);
            if untagged == 0 || !self.contains(untagged as *const u8) {
                if pending.source_kind == "heap-child"
                    && BuiltInTypes::is_heap_pointer(pending.source_addr)
                {
                    let source = HeapObject::from_tagged(pending.source_addr);
                    let header = source.get_header();
                    let preview = source
                        .get_fields()
                        .iter()
                        .take(8)
                        .map(|field| format!("{:#x}", field))
                        .collect::<Vec<_>>()
                        .join(" ");
                    eprintln!(
                        "[ms-invalid-source] source={:#x} type={:?} type_id={} size={} header={:#x} fields={}",
                        pending.source_addr,
                        source.get_object_type(),
                        header.type_id,
                        header.size,
                        unsafe { *(source.untagged() as *const usize) },
                        preview
                    );
                }
                eprintln!(
                    "[ms-invalid-pending] object={:#x} untagged={:#x} source_kind={} source_addr={:#x}",
                    pending.object_ptr,
                    untagged,
                    pending.source_kind,
                    pending.source_addr
                );
                panic!("mark_extra_roots queued pointer outside old space");
            }
            let object = HeapObject::from_tagged(pending.object_ptr);
            if object.marked() {
                continue;
            }

            object.mark();
            if std::env::var("BEAGLE_DEBUG_VEC_GC").is_ok() {
                let header = object.get_header();
                if header.type_id == TYPE_ID_PERSISTENT_VEC {
                    eprintln!(
                        "[ms-mark-extra-vec] object={:#x} count={:#x} shift={:#x} root={:#x} tail={:#x}",
                        pending.object_ptr,
                        object.get_field(0),
                        object.get_field(1),
                        object.get_field(2),
                        object.get_field(3)
                    );
                } else if header.type_id == TYPE_ID_PERSISTENT_VEC_NODE {
                    let fields = object
                        .get_fields()
                        .iter()
                        .take(4)
                        .map(|field| format!("{:#x}", field))
                        .collect::<Vec<_>>()
                        .join(" ");
                    eprintln!(
                        "[ms-mark-extra-vec-node] object={:#x} header={:#x} fields={}",
                        pending.object_ptr,
                        unsafe { *(object.untagged() as *const usize) },
                        fields
                    );
                }
            }
            let header = object.get_header();
            if header.type_id == TYPE_ID_PERSISTENT_VEC {
                vec_gc_log_line(format!(
                    "[ms-mark-extra-vec] object={:#x} count={:#x} shift={:#x} root={:#x} tail={:#x}",
                    pending.object_ptr,
                    object.get_field(0),
                    object.get_field(1),
                    object.get_field(2),
                    object.get_field(3)
                ));
            } else if header.type_id == TYPE_ID_PERSISTENT_VEC_NODE {
                let fields = object.get_fields();
                vec_gc_log_line(format!(
                    "[ms-mark-extra-vec-node] object={:#x} header={:#x} f0={:#x} f1={:#x} f2={:#x} f3={:#x}",
                    pending.object_ptr,
                    unsafe { *(object.untagged() as *const usize) },
                    fields.first().copied().unwrap_or(0),
                    fields.get(1).copied().unwrap_or(0),
                    fields.get(2).copied().unwrap_or(0),
                    fields.get(3).copied().unwrap_or(0)
                ));
            }
            for child in object.get_heap_references() {
                to_mark.push(PendingMark {
                    object_ptr: child.tagged_pointer(),
                    source_addr: pending.object_ptr,
                    source_kind: "heap-child",
                });
            }
            self.push_continuation_segment_children(&object, &mut to_mark);
        }
    }

    fn sweep(&mut self) {
        let existing_range_count = self.free_list.ranges.len();
        let mut old_ranges = std::mem::take(&mut self.free_list.ranges)
            .into_iter()
            .peekable();
        let mut rebuilt_free_list = FreeList::empty_with_capacity(existing_range_count + 8);
        let mut offset = 0;

        while offset <= self.space.highmark {
            if let Some(entry) = old_ranges.peek().copied() {
                assert!(
                    entry.offset >= offset,
                    "GC internal error: free list entry starts before sweep cursor ({:#x} < {:#x})",
                    entry.offset,
                    offset
                );
            }

            let heap_object = HeapObject::from_untagged(unsafe { self.space.start.add(offset) });
            if std::env::var("BEAGLE_DEBUG_VEC_GC").is_ok() {
                let header = heap_object.get_header();
                if header.type_id == TYPE_ID_PERSISTENT_VEC || header.type_id == TYPE_ID_PERSISTENT_VEC_NODE {
                    eprintln!(
                        "[ms-sweep-check] addr={:#x} header={:#x} type_id={} marked={} full_size={:#x}",
                        self.space.start as usize + offset,
                        unsafe { *(heap_object.untagged() as *const usize) },
                        header.type_id,
                        heap_object.marked(),
                        heap_object.full_size()
                    );
                }
            }
            let header = heap_object.get_header();
            if header.type_id == TYPE_ID_PERSISTENT_VEC || header.type_id == TYPE_ID_PERSISTENT_VEC_NODE {
                vec_gc_log_line(format!(
                    "[ms-sweep-check] addr={:#x} header={:#x} type_id={} marked={} full_size={:#x}",
                    self.space.start as usize + offset,
                    unsafe { *(heap_object.untagged() as *const usize) },
                    header.type_id,
                    heap_object.marked(),
                    heap_object.full_size()
                ));
            }
            if let Some(entry) = old_ranges
                .peek()
                .copied()
                .filter(|entry| entry.offset == offset)
            {
                // Old free-list state is supposed to describe only dead space, but if it
                // ever goes stale we must not blindly preserve it over a newly marked live
                // object. That leaves the object allocatable and causes later overlap.
                if !heap_object.marked() {
                    if header.type_id == TYPE_ID_PERSISTENT_VEC || header.type_id == TYPE_ID_PERSISTENT_VEC_NODE {
                        vec_gc_log_line(format!(
                            "[ms-sweep-keep-free-entry] addr={:#x} type_id={} entry={:#x}..{:#x}",
                            self.space.start as usize + offset,
                            header.type_id,
                            self.space.start as usize + entry.offset,
                            self.space.start as usize + entry.end()
                        ));
                    }
                    rebuilt_free_list.append_sorted_coalesced(entry);
                    offset = entry.end();
                    old_ranges.next();
                    continue;
                }
            }

            let full_size = heap_object.full_size();

            if heap_object.marked() {
                let obj_addr = self.space.start as usize + offset;
                let obj_end = obj_addr + full_size;
                let watch_start = 0x7000060340usize;
                let watch_end = watch_start + 0x40;
                if obj_addr < watch_end && obj_end > watch_start {
                    eprintln!(
                        "[ms-sweep-live-watch] obj={:#x}..{:#x} header={:#x} size={} type_id={}",
                        obj_addr,
                        obj_end,
                        unsafe { *(heap_object.untagged() as *const usize) },
                        heap_object.get_header().size,
                        heap_object.get_header().type_id
                    );
                }
                heap_object.unmark();
                offset += full_size;
                offset = (offset + 7) & !7;
                continue;
            }
            let size = full_size;
            let free_addr = self.space.start as usize + offset;
            let free_end = free_addr + size;
            if header.type_id == TYPE_ID_PERSISTENT_VEC || header.type_id == TYPE_ID_PERSISTENT_VEC_NODE {
                vec_gc_log_line(format!(
                    "[ms-sweep-free] addr={:#x}..{:#x} type_id={} header={:#x} size={:#x}",
                    free_addr,
                    free_end,
                    header.type_id,
                    unsafe { *(heap_object.untagged() as *const usize) },
                    size
                ));
            }
            let watch_start = 0x7000060340usize;
            let watch_end = watch_start + 0x40;
            if free_addr < watch_end && free_end > watch_start {
                eprintln!(
                    "[ms-sweep-free-watch] free={:#x}..{:#x} size={:#x}",
                    free_addr, free_end, size
                );
            }
            rebuilt_free_list.append_sorted_coalesced(FreeListEntry { offset, size });
            offset += size;
            offset = (offset + 7) & !7;
            if offset % 8 != 0 {
                panic!(
                    "GC internal error: heap offset {:#x} is not 8-byte aligned after sweep",
                    offset
                );
            }

            if offset > self.space.byte_count() {
                panic!(
                    "GC internal error: heap offset {:#x} exceeds heap size {:#x}",
                    offset,
                    self.space.byte_count()
                );
            }
        }

        for entry in old_ranges {
            rebuilt_free_list.append_sorted_coalesced(entry);
        }

        rebuilt_free_list.prefer_trailing_range();
        self.free_list = rebuilt_free_list;
        self.shrink_highmark_if_tail_is_free();
    }

    #[allow(unused)]
    pub fn new_with_page_count(page_count: usize, options: AllocatorOptions) -> Self {
        let space = Space::new(page_count);
        let size = space.byte_count();
        Self {
            space,
            free_list: FreeList::new(FreeListEntry { offset: 0, size }),
            options,
        }
    }

    /// Walk all live objects in the heap, calling the provided function for each one.
    /// Returns the object's address and a mutable HeapObject reference.
    pub fn walk_objects_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut HeapObject),
    {
        let mut offset = 0;
        loop {
            if offset > self.space.highmark {
                break;
            }
            if let Some(entry) = self.free_list.find_entry_contains(offset) {
                offset = entry.end();
                continue;
            }
            let ptr = unsafe { self.space.start.add(offset) };
            let mut heap_object = HeapObject::from_untagged(ptr);
            f(ptr as usize, &mut heap_object);
            offset += heap_object.full_size();
            offset = (offset + 7) & !7;
        }
    }

    /// Walk all live objects in the heap, calling the provided function for each one.
    #[cfg(feature = "debug-gc")]
    #[allow(unused)]
    pub fn walk_objects<F>(&self, mut f: F)
    where
        F: FnMut(&HeapObject),
    {
        let mut offset = 0;
        loop {
            if offset > self.space.highmark {
                break;
            }
            if let Some(entry) = self.free_list.find_entry_contains(offset) {
                offset = entry.end();
                continue;
            }
            let heap_object = HeapObject::from_untagged(unsafe { self.space.start.add(offset) });
            f(&heap_object);
            offset += heap_object.full_size();
            offset = (offset + 7) & !7;
        }
    }

    /// Post-sweep migration: migrate live objects with outdated struct shapes to the latest layout.
    /// Also updates extra_roots (namespace bindings, handle stack) to point to new locations.
    fn migrate_outdated_structs(&mut self, extra_roots: &[(*mut usize, usize)]) {
        let runtime = crate::get_runtime().get_mut();
        if !runtime.structs.has_pending_migrations() {
            return;
        }

        // Phase 1: Find all objects that need migration, allocate new copies
        let mut forwarding_map: Vec<(usize, usize, usize)> = Vec::new();

        let mut offset = 0;
        loop {
            if offset > self.space.highmark {
                break;
            }
            if let Some(entry) = self.free_list.find_entry_contains(offset) {
                offset = entry.end();
                continue;
            }
            let ptr = unsafe { self.space.start.add(offset) };
            let heap_object = HeapObject::from_untagged(ptr);
            let full_size = heap_object.full_size();

            if heap_object.get_type_id() == 0 && !heap_object.is_opaque_object() {
                // Closures also have type_id=0 but must NOT be migrated as structs
                let is_closure = heap_object.get_object_type() == Some(BuiltInTypes::Closure);
                if !is_closure {
                    let struct_id = heap_object.get_struct_id();
                    let layout_version = heap_object.get_layout_version();
                    if let Some(plan) = runtime
                        .structs
                        .migration_plan_for(struct_id, layout_version)
                    {
                        let old_header = heap_object.get_header();
                        let new_header = Header {
                            type_id: old_header.type_id,
                            type_data: old_header.type_data, // struct_id unchanged
                            size: plan.new_field_count as u16,
                            opaque: old_header.opaque,
                            marked: false,
                            large: false,
                            type_flags: plan.new_layout_version,
                        };

                        let total_bytes = 8 + plan.new_field_count * 8;
                        let mut data = vec![0u8; total_bytes];
                        data[0..8].copy_from_slice(&new_header.to_usize().to_ne_bytes());

                        let null_val = BuiltInTypes::null_value() as usize;
                        for (new_idx, mapping) in plan.field_map.iter().enumerate() {
                            let value = match mapping {
                                Some(old_idx) => heap_object.get_field(*old_idx),
                                None => null_val,
                            };
                            let field_offset = (1 + new_idx) * 8;
                            data[field_offset..field_offset + 8]
                                .copy_from_slice(&value.to_ne_bytes());
                        }

                        let new_ptr = self.copy_data_to_offset(&data);
                        forwarding_map.push((ptr as usize, full_size, new_ptr as usize));

                        // Set forwarding pointer in old object's header
                        let tagged_new = BuiltInTypes::HeapObject.tag(new_ptr as isize) as usize;
                        unsafe {
                            *(ptr as *mut usize) = Header::set_forwarding_bit(tagged_new);
                        }
                    }
                }
            }

            offset += full_size;
            offset = (offset + 7) & !7;
        }

        if forwarding_map.is_empty() {
            runtime.structs.complete_pending_migrations();
            return;
        }

        // Phase 2: Walk all live objects and update references to forwarded objects
        let mut offset = 0;
        loop {
            if offset > self.space.highmark {
                break;
            }
            if let Some(entry) = self.free_list.find_entry_contains(offset) {
                offset = entry.end();
                continue;
            }
            let ptr = unsafe { self.space.start.add(offset) };
            let raw_header = unsafe { *(ptr as *const usize) };

            // Skip forwarded objects (old copies waiting to be freed)
            if Header::is_forwarding_bit_set(raw_header) {
                if let Some((_, old_size, _)) = forwarding_map
                    .iter()
                    .find(|(old_ptr, _, _)| *old_ptr == ptr as usize)
                {
                    offset += old_size;
                    offset = (offset + 7) & !7;
                    continue;
                }
                offset += 8;
                continue;
            }

            let mut heap_object = HeapObject::from_untagged(ptr);
            let full_size = heap_object.full_size();
            let is_cont_resume_closure =
                heap_object.get_object_type() == Some(BuiltInTypes::Closure)
                    && crate::get_runtime()
                        .get()
                        .get_function_by_pointer(
                            BuiltInTypes::untag(heap_object.get_field(0)) as *const u8
                        )
                        .map(|f| f.name == "beagle.builtin/continuation-trampoline")
                        .unwrap_or(false);

            if !heap_object.is_opaque_object() {
                for (field_index, field) in heap_object.get_fields_mut().iter_mut().enumerate() {
                    if let Some(field_obj) = HeapObject::try_from_tagged(*field) {
                        let field_untagged = field_obj.untagged();
                        let field_tag = *field & 7;
                        if self.space.contains(field_untagged as *const u8) {
                            let field_header = unsafe { *(field_untagged as *const usize) };
                            if Header::is_forwarding_bit_set(field_header) {
                                if is_cont_resume_closure && field_index == 3 {
                                    eprintln!(
                                        "[ms-update-before] closure={:#x} field3={:#x} tag={:#x}",
                                        BuiltInTypes::Closure.tag(ptr as isize) as usize,
                                        *field,
                                        field_tag
                                    );
                                }
                                let new_tagged = Header::clear_forwarding_bit(field_header);
                                *field = (new_tagged & !7) | field_tag;
                                if is_cont_resume_closure && field_index == 3 {
                                    eprintln!(
                                        "[ms-update-after] closure={:#x} field3={:#x}",
                                        BuiltInTypes::Closure.tag(ptr as isize) as usize,
                                        *field
                                    );
                                }
                            }
                        }
                    }
                }
            }

            offset += full_size;
            offset = (offset + 7) & !7;
        }

        // Phase 2b: Update extra roots (namespace bindings, handle stack entries)
        // These are stored in Rust-side Vecs, not on the heap, so Phase 2's heap walk misses them.
        for &(slot_addr, _) in extra_roots {
            let value = unsafe { *slot_addr };
            if BuiltInTypes::is_heap_pointer(value) {
                let untagged = BuiltInTypes::untag(value);
                let tag = value & 7;
                if self.space.contains(untagged as *const u8) {
                    let header_at_old = unsafe { *(untagged as *const usize) };
                    if Header::is_forwarding_bit_set(header_at_old) {
                        let new_tagged = Header::clear_forwarding_bit(header_at_old);
                        unsafe {
                            *slot_addr = (new_tagged & !7) | tag;
                        }
                    }
                }
            }
        }

        // Phase 3: Free old objects
        for (old_ptr, old_size, _) in forwarding_map {
            let heap_offset = old_ptr - self.space.start as usize;
            self.free_list.insert(FreeListEntry {
                offset: heap_offset,
                size: old_size,
            });
        }

        self.shrink_highmark_if_tail_is_free();
        runtime.structs.complete_pending_migrations();
    }
}

impl Allocator for MarkAndSweep {
    fn new(options: AllocatorOptions) -> Self {
        let page_count = DEFAULT_PAGE_COUNT;
        Self::new_with_page_count(page_count, options)
    }

    fn try_allocate(
        &mut self,
        words: usize,
        kind: crate::types::BuiltInTypes,
    ) -> Result<super::AllocateAction, Box<dyn std::error::Error>> {
        self.allocate_inner(Word::from_word(words), None, kind)
    }

    fn try_allocate_zeroed(
        &mut self,
        words: usize,
        kind: crate::types::BuiltInTypes,
    ) -> Result<super::AllocateAction, Box<dyn std::error::Error>> {
        let result = self.try_allocate(words, kind)?;
        if let AllocateAction::Allocated(ptr) = result {
            // Zero the field area (skip header) so GC doesn't trace garbage data
            let heap_object = HeapObject::from_untagged(ptr);
            let header_size = heap_object.header_size();
            let field_bytes = words * 8;
            unsafe {
                std::ptr::write_bytes((ptr as *mut u8).add(header_size), 0, field_bytes);
            }
            Ok(AllocateAction::Allocated(ptr))
        } else {
            Ok(result)
        }
    }

    fn gc(&mut self, gc_frame_tops: &[usize], extra_roots: &[(*mut usize, usize)]) {
        if !self.options.gc {
            return;
        }
        let start = std::time::Instant::now();
        for &gc_frame_top in gc_frame_tops {
            self.mark_from_chain(gc_frame_top);
        }
        self.log_bad_continuation_closures("after-mark-from-chain");

        // Mark extra roots from shadow stacks
        self.mark_extra_roots(extra_roots);
        self.log_bad_continuation_closures("after-mark-extra-roots");

        self.log_bad_continuation_closures("before-sweep");
        self.sweep();
        self.log_bad_continuation_closures("after-sweep");

        self.migrate_outdated_structs(extra_roots);
        self.log_bad_continuation_closures("after-migrate");
        if self.options.print_stats {
            println!("Mark and sweep took {:?}", start.elapsed());
        }
    }

    fn grow(&mut self) {
        let current_max_offset = self.space.byte_count();
        self.space.double_committed_memory();
        let after_max_offset = self.space.byte_count();
        self.free_list.insert(FreeListEntry {
            offset: current_max_offset,
            size: after_max_offset - current_max_offset,
        });
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }

    fn can_allocate(&self, words: usize, _kind: BuiltInTypes) -> bool {
        // Before GC: returning `false` is harmless — it just makes the
        // runtime call `gc()` first, which is free if we really don't
        // have space.
        //
        // After GC: returning `false` here tells the runtime to call
        // `grow()`, which doubles the committed heap. If the free list
        // actually has a block large enough to satisfy `words`, we must
        // return `true` so the caller uses that block instead of
        // growing. The old unconditional-`false` implementation grew
        // the heap on every `ensure_space_for` even when GC had just
        // reclaimed plenty of space — continuation-heavy workloads hit
        // the OS commit ceiling well before the heap was actually full.
        let header_size = if words > Header::MAX_INLINE_SIZE { 16 } else { 8 };
        let needed = words * 8 + header_size;
        self.free_list.ranges.iter().any(|r| r.size >= needed)
    }
}

#[cfg(test)]
mod tests {
    use super::{FreeList, FreeListEntry};

    #[test]
    fn append_sorted_coalesced_merges_adjacent_ranges() {
        let mut free_list = FreeList::empty_with_capacity(4);
        free_list.append_sorted_coalesced(FreeListEntry { offset: 8, size: 8 });
        free_list.append_sorted_coalesced(FreeListEntry {
            offset: 16,
            size: 16,
        });
        free_list.append_sorted_coalesced(FreeListEntry {
            offset: 32,
            size: 24,
        });

        assert_eq!(free_list.ranges.len(), 1);
        assert_eq!(
            free_list.ranges[0],
            FreeListEntry {
                offset: 8,
                size: 48,
            }
        );
    }

    #[test]
    fn append_sorted_coalesced_keeps_gaps_separate() {
        let mut free_list = FreeList::empty_with_capacity(4);
        free_list.append_sorted_coalesced(FreeListEntry { offset: 0, size: 8 });
        free_list.append_sorted_coalesced(FreeListEntry {
            offset: 24,
            size: 8,
        });

        assert_eq!(free_list.ranges.len(), 2);
        assert_eq!(free_list.ranges[0], FreeListEntry { offset: 0, size: 8 });
        assert_eq!(
            free_list.ranges[1],
            FreeListEntry {
                offset: 24,
                size: 8
            }
        );
    }
}

// Helper methods for heap dump
impl MarkAndSweep {
    /// Collect all objects for heap dump
    #[cfg(feature = "heap-dump")]
    pub fn collect_objects_for_dump(
        &self,
        classifier: &super::heap_dump::PointerClassifier,
    ) -> Vec<super::heap_dump::ObjectSnapshot> {
        use super::heap_dump::*;
        use crate::types::{BuiltInTypes, Header, HeapObject};

        let mut objects = Vec::new();
        let mut offset = 0;

        loop {
            if offset > self.space.highmark {
                break;
            }

            // Check if this offset is in the free list
            if let Some(entry) = self.free_list.find_entry_contains(offset) {
                offset = entry.end();
                continue;
            }

            let ptr = unsafe { self.space.start.add(offset) };
            let header_raw = unsafe { *(ptr as *const usize) };
            let header = Header::from_usize(header_raw);

            let header_size = if header.large { 16 } else { 8 };
            let fields_size = if header.large {
                let size_ptr = unsafe { (ptr as *const usize).add(1) };
                unsafe { *size_ptr * 8 }
            } else {
                header.size as usize * 8
            };
            let full_size = header_size + fields_size;

            if full_size == 0 {
                offset += 8;
                continue;
            }

            let heap_obj = HeapObject::from_untagged(ptr);

            let mut field_snapshots = Vec::new();
            if !header.opaque {
                let fields = heap_obj.get_fields();
                for (i, &field_value) in fields.iter().enumerate() {
                    field_snapshots.push(FieldSnapshot {
                        index: i,
                        value: format!("{:#x}", field_value),
                        tag: tag_name_local(field_value),
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
            offset = (offset + 7) & !7;
        }

        objects
    }
}

#[cfg(feature = "heap-dump")]
fn tag_name_local(value: usize) -> String {
    use crate::types::BuiltInTypes;
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
