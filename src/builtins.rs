use core::panic;
use std::{
    arch::asm,
    cell::Cell,
    error::Error,
    ffi::{CStr, c_void},
    mem::{self, transmute},
    slice::{from_raw_parts, from_raw_parts_mut},
    thread,
};

use crate::{
    Message,
    collections::TYPE_ID_CONTINUATION_SEGMENT,
    gc::STACK_SIZE,
    get_runtime,
    runtime::{ContinuationObject, DispatchTable, FFIInfo, FFIType, RawPtr, Runtime},
    types::{BuiltInTypes, Header, HeapObject},
};

use rand::Rng;
use std::hint::black_box;

// Thread-local storage for the frame pointer and gc return address at builtin entry.
// This is set by builtins that receive frame_pointer from Beagle code.
// Used by gc() when triggered internally (e.g., during allocation).
thread_local! {
    static SAVED_FRAME_POINTER: Cell<usize> = const { Cell::new(0) };
    static SAVED_GC_RETURN_ADDR: Cell<usize> = const { Cell::new(0) };
    // Used for struct returns - stores the high 64 bits of a 16-byte struct return
    static STRUCT_RETURN_HIGH: Cell<u64> = const { Cell::new(0) };
}

/// Save the frame pointer for later use by gc().
/// Called by builtins that receive frame_pointer from Beagle.
pub fn save_frame_pointer(fp: usize) {
    SAVED_FRAME_POINTER.with(|cell| cell.set(fp));
}

/// Get the saved frame pointer.
/// Returns 0 if none has been saved (shouldn't happen in normal operation).
pub fn get_saved_frame_pointer() -> usize {
    SAVED_FRAME_POINTER.with(|cell| cell.get())
}

/// Save the gc return address for later use by gc().
/// Called by builtins that receive frame_pointer/stack_pointer from Beagle.
pub fn save_gc_return_addr(addr: usize) {
    SAVED_GC_RETURN_ADDR.with(|cell| cell.set(addr));
}

/// Get the saved gc return address.
/// Returns 0 if none has been saved.
pub fn get_saved_gc_return_addr() -> usize {
    SAVED_GC_RETURN_ADDR.with(|cell| cell.get())
}

/// Reset the saved frame pointer and gc return address.
/// Called by Runtime::reset() to clear stale values between test runs.
pub fn reset_saved_gc_context() {
    SAVED_FRAME_POINTER.with(|cell| cell.set(0));
    SAVED_GC_RETURN_ADDR.with(|cell| cell.set(0));
}

/// Saved GC context - used to save/restore around calls back into Beagle
pub struct SavedGcContext {
    pub frame_pointer: usize,
    pub gc_return_addr: usize,
}

/// Save the current GC context. Call this BEFORE calling back into Beagle.
/// The Beagle code may call builtins that update the saved GC context.
/// After the Beagle code returns, call restore_gc_context to restore it.
pub fn save_current_gc_context() -> SavedGcContext {
    SavedGcContext {
        frame_pointer: get_saved_frame_pointer(),
        gc_return_addr: get_saved_gc_return_addr(),
    }
}

/// Restore a previously saved GC context. Call this AFTER calling back into Beagle.
/// This ensures that if GC runs after the Beagle call returns, it uses the
/// correct (non-stale) frame pointer.
pub fn restore_gc_context(ctx: SavedGcContext) {
    save_frame_pointer(ctx.frame_pointer);
    save_gc_return_addr(ctx.gc_return_addr);
}

/// Macro to save frame pointer and gc return address for GC stack walking.
/// The gc_return_addr is the return address from the Beagle code that called
/// this builtin. On ARM64, this was in LR and is now saved at [Rust_FP + 8]
/// by Rust's prologue. On x86-64, it's also at [Rust_FP + 8].
///
/// The saved frame_pointer (Beagle FP) is used by __pause to check if the
/// saved gc_return_addr is still valid for the current call.
#[macro_export]
macro_rules! save_gc_context {
    ($stack_pointer:expr, $frame_pointer:expr) => {{
        $crate::builtins::save_frame_pointer($frame_pointer);
        // Get the return address from where Rust's prologue saved it
        // This is the address in Beagle code right after the builtin call
        let rust_fp = $crate::builtins::get_current_rust_frame_pointer();
        // SAFETY: rust_fp + 8 points to the saved return address in the Rust stack frame
        #[allow(unused_unsafe)]
        let gc_return_addr = unsafe { *((rust_fp + 8) as *const usize) };
        $crate::builtins::save_gc_return_addr(gc_return_addr);
    }};
}

/// Read the current frame pointer register.
/// Used by GC to walk the stack starting from the current Rust function's frame.
/// MUST be inlined so we read the caller's frame pointer, not this function's.
#[inline(always)]
pub fn get_current_rust_frame_pointer() -> usize {
    let fp: usize;
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            unsafe { core::arch::asm!("mov {}, rbp", out(reg) fp) };
        } else if #[cfg(target_arch = "aarch64")] {
            unsafe { core::arch::asm!("mov {}, x29", out(reg) fp) };
        } else {
            compile_error!("Unsupported architecture");
        }
    }
    fp
}

#[allow(unused)]
#[unsafe(no_mangle)]
#[inline(never)]
/// # Safety
///
/// This does nothing
pub unsafe extern "C" fn debugger_info(buffer: *const u8, length: usize) {
    // Hack to make sure this isn't inlined
    black_box(buffer);
    black_box(length);
}

#[macro_export]
macro_rules! debug_only {
    ($($code:tt)*) => {
        #[cfg(debug_assertions)]
        {
            $($code)*
        }
    };
}

#[macro_export]
macro_rules! debug_flag_only {
    ($($code:tt)*) => {
        {
            let runtime = get_runtime().get();
            if runtime.get_command_line_args().debug {
                $($code)*
            }
        }
    };
}

pub fn debugger(_message: Message) {
    debug_only! {
        let serialized_message : Vec<u8>;
        #[cfg(feature="json")] {
        use nanoserde::SerJson;
            let serialized : String = SerJson::serialize_json(&_message);
            serialized_message = serialized.into_bytes();
        }
        #[cfg(not(feature="json"))] {
            use crate::Serialize;
            serialized_message = _message.to_binary();
        }
        let ptr = serialized_message.as_ptr();
        let length = serialized_message.len();
        mem::forget(serialized_message);
        unsafe {
            debugger_info(ptr, length);
        }
        // TODO: Should clean up this memory
    }
}

pub unsafe extern "C" fn println_value(value: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let result = runtime.println(value);
    if let Err(error) = result {
        let stack_pointer = get_current_stack_pointer();
        let frame_pointer = get_saved_frame_pointer();
        println!("Error: {:?}", error);
        unsafe { throw_error(stack_pointer, frame_pointer) };
    }
    0b111
}

pub unsafe extern "C" fn to_string(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let result = runtime.get_repr(value, 0);
    if result.is_none() {
        let stack_pointer = get_current_stack_pointer();
        unsafe { throw_error(stack_pointer, frame_pointer) };
    }
    let result = result.unwrap();
    runtime
        .allocate_string(stack_pointer, result)
        .unwrap()
        .into()
}

pub unsafe extern "C" fn to_number(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let string = runtime.get_string(stack_pointer, value);
    if string.contains(".") {
        todo!()
    } else {
        let result = string.parse::<isize>().unwrap();
        BuiltInTypes::Int.tag(result) as usize
    }
}

#[inline(always)]
fn print_call_builtin(_runtime: &Runtime, _name: &str) {
    debug_only!(if _runtime.get_command_line_args().print_builtin_calls {
        println!("Calling: {}", _name);
    });
}

pub unsafe extern "C" fn print_value(value: usize) -> usize {
    let runtime = get_runtime().get_mut();
    runtime.print(value);
    0b111
}

pub extern "C" fn print_byte(value: usize) -> usize {
    let byte_value = BuiltInTypes::untag(value) as u8;
    let runtime = get_runtime().get_mut();
    runtime.printer.print_byte(byte_value);
    0b111
}

/// Mark the card containing an address as dirty for write barrier.
/// Called from generated code after heap stores to old gen objects.
/// Takes the untagged address of the object being written to.
///
/// This is a no-op for non-generational GCs (card_table_ptr will be null).
/// For generational GC, it only marks cards for addresses in old gen.
pub extern "C" fn mark_card(untagged_addr: usize) -> usize {
    let runtime = get_runtime().get_mut();
    // Tag the address as HeapObject for the mark_card_for_object call
    let tagged_addr = BuiltInTypes::HeapObject.tag(untagged_addr as isize) as usize;
    // Mark the card if object is in old gen (no-op for non-generational GCs)
    runtime.mark_card_for_object(tagged_addr);
    0b111 // Return null
}

extern "C" fn allocate(stack_pointer: usize, frame_pointer: usize, size: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let size = BuiltInTypes::untag(size);
    let runtime = get_runtime().get_mut();

    let result = runtime
        .allocate(size, stack_pointer, BuiltInTypes::HeapObject)
        .unwrap();

    debug_assert!(BuiltInTypes::is_heap_pointer(result));
    debug_assert!(BuiltInTypes::untag(result).is_multiple_of(8));
    result
}

/// Allocate with zeroed memory (for arrays that don't initialize all fields)
extern "C" fn allocate_zeroed(stack_pointer: usize, frame_pointer: usize, size: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let size = BuiltInTypes::untag(size);
    let runtime = get_runtime().get_mut();

    let result = runtime
        .allocate_zeroed(size, stack_pointer, BuiltInTypes::HeapObject)
        .unwrap();

    debug_assert!(BuiltInTypes::is_heap_pointer(result));
    debug_assert!(BuiltInTypes::untag(result).is_multiple_of(8));
    result
}

extern "C" fn allocate_float(stack_pointer: usize, frame_pointer: usize, size: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let value = BuiltInTypes::untag(size);

    let result = runtime
        .allocate(value, stack_pointer, BuiltInTypes::Float)
        .unwrap();

    debug_assert!(BuiltInTypes::get_kind(result) == BuiltInTypes::Float);
    debug_assert!(BuiltInTypes::untag(result).is_multiple_of(8));
    result
}

extern "C" fn get_string_index(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    index: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "get_string_index");
    let runtime = get_runtime().get_mut();
    if BuiltInTypes::get_kind(string) == BuiltInTypes::String {
        let string = runtime.get_string_literal(string);
        let index = BuiltInTypes::untag(index);
        let result = string.chars().nth(index).unwrap();
        let result = result.to_string();
        runtime
            .allocate_string(stack_pointer, result)
            .unwrap()
            .into()
    } else {
        let object_pointer_id = runtime.register_temporary_root(string);
        // we have a heap allocated string
        let string = HeapObject::from_tagged(string);
        // TODO: Type safety
        // We are just going to assert that the type_id == 2
        assert!(string.get_type_id() == 2);
        let index = BuiltInTypes::untag(index);
        let string = string.get_string_bytes();
        // TODO: This will break with unicode
        let result = string[index] as char;
        let result = result.to_string();
        let result = runtime
            .allocate_string(stack_pointer, result)
            .unwrap()
            .into();
        runtime.unregister_temporary_root(object_pointer_id);
        result
    }
}

extern "C" fn get_string_length(string: usize) -> usize {
    print_call_builtin(get_runtime().get(), "get_string_length");
    let runtime = get_runtime().get_mut();
    if BuiltInTypes::get_kind(string) == BuiltInTypes::String {
        // TODO: Make faster
        let string = runtime.get_string_literal(string);
        BuiltInTypes::Int.tag(string.len() as isize) as usize
    } else {
        // we have a heap allocated string
        let string = HeapObject::from_tagged(string);
        let length = string.get_type_data();
        BuiltInTypes::Int.tag(length as isize) as usize
    }
}

extern "C" fn string_concat(
    stack_pointer: usize,
    frame_pointer: usize,
    a: usize,
    b: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "string_concat");
    let runtime = get_runtime().get_mut();
    let a = runtime.get_string(stack_pointer, a);
    let b = runtime.get_string(stack_pointer, b);
    let result = a + &b;
    runtime
        .allocate_string(stack_pointer, result)
        .unwrap()
        .into()
}

extern "C" fn substring(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    start: usize,
    length: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "substring");
    let runtime = get_runtime().get_mut();
    let string_pointer = runtime.register_temporary_root(string);
    let start = BuiltInTypes::untag(start);
    let length = BuiltInTypes::untag(length);
    let string = runtime
        .get_substring(stack_pointer, string, start, length)
        .unwrap();
    runtime.unregister_temporary_root(string_pointer);
    string.into()
}

extern "C" fn uppercase(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "uppercase");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let uppercased = string_value.to_uppercase();
    runtime
        .allocate_string(stack_pointer, uppercased)
        .unwrap()
        .into()
}

extern "C" fn lowercase(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "lowercase");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let lowercased = string_value.to_lowercase();
    runtime
        .allocate_string(stack_pointer, lowercased)
        .unwrap()
        .into()
}

extern "C" fn split(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    delimiter: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "split");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let delimiter_value = runtime.get_string(stack_pointer, delimiter);

    let parts: Vec<String> = string_value
        .split(&delimiter_value)
        .map(|s| s.to_string())
        .collect();

    runtime.create_string_array(stack_pointer, &parts).unwrap()
}

extern "C" fn join(
    stack_pointer: usize,
    frame_pointer: usize,
    array: usize,
    separator: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "join");
    let runtime = get_runtime().get_mut();
    let separator_value = runtime.get_string(stack_pointer, separator);

    let array_obj = HeapObject::from_tagged(array);
    let fields = array_obj.get_fields();

    let strings: Vec<String> = fields
        .iter()
        .map(|&field| runtime.get_string(stack_pointer, field))
        .collect();

    let joined = strings.join(&separator_value);

    runtime
        .allocate_string(stack_pointer, joined)
        .unwrap()
        .into()
}

extern "C" fn trim(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "trim");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let trimmed = string_value.trim();
    runtime
        .allocate_string(stack_pointer, trimmed.to_string())
        .unwrap()
        .into()
}

extern "C" fn trim_left(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "trim-left");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let trimmed = string_value.trim_start();
    runtime
        .allocate_string(stack_pointer, trimmed.to_string())
        .unwrap()
        .into()
}

extern "C" fn trim_right(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "trim-right");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let trimmed = string_value.trim_end();
    runtime
        .allocate_string(stack_pointer, trimmed.to_string())
        .unwrap()
        .into()
}

extern "C" fn starts_with(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    prefix: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "starts-with?");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let prefix_value = runtime.get_string(stack_pointer, prefix);
    let result = string_value.starts_with(&prefix_value);
    BuiltInTypes::construct_boolean(result) as usize
}

extern "C" fn ends_with(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    suffix: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "ends-with?");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let suffix_value = runtime.get_string(stack_pointer, suffix);
    let result = string_value.ends_with(&suffix_value);
    BuiltInTypes::construct_boolean(result) as usize
}

extern "C" fn string_contains(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    substr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "contains?");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let substr_value = runtime.get_string(stack_pointer, substr);
    let result = string_value.contains(&substr_value);
    BuiltInTypes::construct_boolean(result) as usize
}

extern "C" fn index_of(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    substr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "index-of");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let substr_value = runtime.get_string(stack_pointer, substr);

    match string_value.find(&substr_value) {
        Some(index) => BuiltInTypes::construct_int(index as isize) as usize,
        None => BuiltInTypes::construct_int(-1) as usize,
    }
}

extern "C" fn last_index_of(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    substr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "last-index-of");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let substr_value = runtime.get_string(stack_pointer, substr);

    match string_value.rfind(&substr_value) {
        Some(index) => BuiltInTypes::construct_int(index as isize) as usize,
        None => BuiltInTypes::construct_int(-1) as usize,
    }
}

extern "C" fn replace_string(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    from: usize,
    to: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "replace");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let from_value = runtime.get_string(stack_pointer, from);
    let to_value = runtime.get_string(stack_pointer, to);

    let replaced = string_value.replace(&from_value, &to_value);

    runtime
        .allocate_string(stack_pointer, replaced)
        .unwrap()
        .into()
}

extern "C" fn blank_string(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "blank?");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);

    // A string is blank if it's empty or contains only whitespace
    let is_blank = string_value.trim().is_empty();

    BuiltInTypes::construct_boolean(is_blank) as usize
}

extern "C" fn replace_first_string(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    from: usize,
    to: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "replace-first");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let from_value = runtime.get_string(stack_pointer, from);
    let to_value = runtime.get_string(stack_pointer, to);

    let replaced = string_value.replacen(&from_value, &to_value, 1);

    runtime
        .allocate_string(stack_pointer, replaced)
        .unwrap()
        .into()
}

extern "C" fn pad_left_string(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    width: usize,
    pad_char: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "pad-left");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let width_value = BuiltInTypes::untag_isize(width as isize) as usize;
    let pad_char_value = runtime.get_string(stack_pointer, pad_char);

    // Get the first character from the pad string, or use space if empty
    let pad_ch = pad_char_value.chars().next().unwrap_or(' ');

    let current_len = string_value.chars().count();
    if current_len >= width_value {
        // Already at or exceeds desired width, return as-is
        return string;
    }

    let padding_needed = width_value - current_len;
    let padded = format!(
        "{}{}",
        pad_ch.to_string().repeat(padding_needed),
        string_value
    );

    runtime
        .allocate_string(stack_pointer, padded)
        .unwrap()
        .into()
}

extern "C" fn pad_right_string(
    stack_pointer: usize,
    frame_pointer: usize,
    string: usize,
    width: usize,
    pad_char: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "pad-right");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);
    let width_value = BuiltInTypes::untag_isize(width as isize) as usize;
    let pad_char_value = runtime.get_string(stack_pointer, pad_char);

    // Get the first character from the pad string, or use space if empty
    let pad_ch = pad_char_value.chars().next().unwrap_or(' ');

    let current_len = string_value.chars().count();
    if current_len >= width_value {
        // Already at or exceeds desired width, return as-is
        return string;
    }

    let padding_needed = width_value - current_len;
    let padded = format!(
        "{}{}",
        string_value,
        pad_ch.to_string().repeat(padding_needed)
    );

    runtime
        .allocate_string(stack_pointer, padded)
        .unwrap()
        .into()
}

extern "C" fn lines_string(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "lines");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);

    let line_vec: Vec<String> = string_value.lines().map(|s| s.to_string()).collect();

    runtime
        .create_string_array(stack_pointer, &line_vec)
        .unwrap()
}

extern "C" fn words_string(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "words");
    let runtime = get_runtime().get_mut();
    let string_value = runtime.get_string(stack_pointer, string);

    let word_vec: Vec<String> = string_value
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();

    runtime
        .create_string_array(stack_pointer, &word_vec)
        .unwrap()
}

extern "C" fn fill_object_fields(object_pointer: usize, value: usize) -> usize {
    print_call_builtin(get_runtime().get(), "fill_object_fields");
    let mut object = HeapObject::from_tagged(object_pointer);
    let raw_slice = object.get_fields_mut();
    raw_slice.fill(value);
    object_pointer
}

extern "C" fn make_closure(
    stack_pointer: usize,
    frame_pointer: usize,
    function: usize,
    num_free: usize,
    free_variable_pointer: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "make_closure");
    let runtime = get_runtime().get_mut();
    if BuiltInTypes::get_kind(function) != BuiltInTypes::Function {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!(
                    "Expected function, got {:?}",
                    BuiltInTypes::get_kind(function)
                ),
            );
        }
    }

    assert!(matches!(
        BuiltInTypes::get_kind(function),
        BuiltInTypes::Function
    ));

    let num_free = BuiltInTypes::untag(num_free);
    let free_variable_pointer = free_variable_pointer as *const usize;
    let start = unsafe { free_variable_pointer.sub(num_free.saturating_sub(1)) };
    let free_variables = unsafe { from_raw_parts(start, num_free) };
    runtime
        .make_closure(stack_pointer, function, free_variables)
        .unwrap()
}

/// Create a FunctionObject from a function pointer.
/// Unlike closures, FunctionObjects have no free variables and don't pass self as arg0.
extern "C" fn make_function_object(
    stack_pointer: usize,
    frame_pointer: usize,
    function: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "make_function_object");
    let runtime = get_runtime().get_mut();

    if BuiltInTypes::get_kind(function) != BuiltInTypes::Function {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!(
                    "make-function-object: Expected function, got {:?}",
                    BuiltInTypes::get_kind(function)
                ),
            );
        }
    }

    runtime
        .make_function_object(stack_pointer, function)
        .unwrap()
}

pub fn get_current_stack_pointer() -> usize {
    use core::arch::asm;
    let sp: usize;
    unsafe {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                asm!(
                    "mov {0}, rsp",
                    out(reg) sp
                );
            } else {
                asm!(
                    "mov {0}, sp",
                    out(reg) sp
                );
            }
        }
    }
    sp
}

/// Get the current frame pointer (RBP on x86-64, X29 on ARM64).
/// Note: This is currently unused because Rust functions may not preserve
/// frame pointers, making FP-chain traversal unreliable for GC.
#[allow(dead_code)]
pub fn get_current_frame_pointer() -> usize {
    use core::arch::asm;
    let fp: usize;
    unsafe {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                asm!(
                    "mov {0}, rbp",
                    out(reg) fp
                );
            } else {
                asm!(
                    "mov {0}, x29",
                    out(reg) fp
                );
            }
        }
    }
    fp
}

extern "C" fn property_access(
    struct_pointer: usize,
    str_constant_ptr: usize,
    property_cache_location: usize,
) -> usize {
    let runtime = get_runtime().get_mut();
    let (result, index) = runtime
        .property_access(struct_pointer, str_constant_ptr)
        .unwrap_or_else(|error| {
            let stack_pointer = get_current_stack_pointer();
            let frame_pointer = get_saved_frame_pointer();
            let heap_object = HeapObject::from_tagged(struct_pointer);
            println!("Heap object: {:?}", heap_object.get_header());
            println!("Error: {:?}", error);
            unsafe {
                throw_error(stack_pointer, frame_pointer);
            };
        });
    let type_id = HeapObject::from_tagged(struct_pointer).get_struct_id();
    let buffer = unsafe { from_raw_parts_mut(property_cache_location as *mut usize, 2) };
    buffer[0] = type_id;
    buffer[1] = index * 8;
    result
}

/// Slow path for protocol dispatch - looks up function in dispatch table and updates cache
/// Returns the function pointer to call
///
/// Type identification for dispatch:
/// - Tagged primitives (Int, Float, Bool): high bit marker + (tag + 16)
/// - Heap objects with type_id > 0 (Array=1, String=2, Keyword=3): high bit marker + type_id
/// - Heap objects with type_id = 0 (structs): struct_id from type_data (tagged)
extern "C" fn protocol_dispatch(
    stack_pointer: usize,
    frame_pointer: usize,
    first_arg: usize,
    cache_location: usize,
    dispatch_table_ptr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let type_id = if BuiltInTypes::is_heap_pointer(first_arg) {
        let heap_obj = HeapObject::from_tagged(first_arg);
        let header_type_id = heap_obj.get_type_id();

        if header_type_id == 0 {
            // Custom struct - use struct_id from type_data (tagged)
            heap_obj.get_struct_id()
        } else {
            // Built-in heap type (Array=1, String=2, Keyword=3)
            // Use primitive dispatch with header_type_id
            0x8000_0000_0000_0000 | header_type_id
        }
    } else {
        // Tagged primitive (Int, Float, Bool, String constant, etc.)
        let kind = BuiltInTypes::get_kind(first_arg);
        let tag = kind.get_tag() as usize;
        // Map tags to primitive indices:
        // - String constant (tag 2) -> index 2 (same as heap String)
        // - Int (tag 0) -> index 16
        // - Float (tag 1) -> index 17
        // - Bool (tag 3) -> index 19
        let primitive_index = if tag == 2 {
            2 // String constant uses same index as heap String
        } else {
            tag + 16
        };
        0x8000_0000_0000_0000 | primitive_index
    };

    // Look up function pointer in dispatch table
    let dispatch_table = unsafe { &*(dispatch_table_ptr as *const DispatchTable) };
    let fn_ptr = if type_id & 0x8000_0000_0000_0000 != 0 {
        // Primitive/built-in type
        let primitive_index = type_id & 0x7FFF_FFFF_FFFF_FFFF;
        dispatch_table.lookup_primitive(primitive_index)
    } else {
        // Struct type - struct_id is tagged in header, need to untag
        let struct_id = BuiltInTypes::untag(type_id);
        dispatch_table.lookup_struct(struct_id)
    };

    if fn_ptr == 0 {
        // Get a string representation of the value for the error message
        let runtime = get_runtime().get_mut();
        let value_repr = runtime
            .get_repr(first_arg, 0)
            .unwrap_or_else(|| "<unknown>".to_string());

        // Get type name for better error message
        let type_name = if type_id & 0x8000_0000_0000_0000 != 0 {
            let primitive_index = type_id & 0x7FFF_FFFF_FFFF_FFFF;
            match primitive_index {
                1 => "Array",
                2 => "String",
                3 => "Keyword",
                16 => "Int",
                17 => "Float",
                19 => "Bool",
                _ => "Unknown",
            }
        } else {
            "Struct"
        };
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!(
                    "Function not implemented for {} value: {}",
                    type_name, value_repr
                ),
            );
        }
    }

    // Update cache
    let buffer = unsafe { from_raw_parts_mut(cache_location as *mut usize, 2) };
    buffer[0] = type_id;
    buffer[1] = fn_ptr;

    fn_ptr
}

extern "C" fn type_of(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "type_of");
    let runtime = get_runtime().get_mut();
    runtime.type_of(stack_pointer, value).unwrap()
}

extern "C" fn get_os(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "get_os");
    let runtime = get_runtime().get_mut();
    let os_name = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "unknown"
    };
    runtime
        .allocate_string(stack_pointer, os_name.to_string())
        .unwrap()
        .into()
}

extern "C" fn equal(a: usize, b: usize) -> usize {
    print_call_builtin(get_runtime().get(), "equal");
    let runtime = get_runtime().get_mut();
    if runtime.equal(a, b) {
        BuiltInTypes::true_value() as usize
    } else {
        BuiltInTypes::false_value() as usize
    }
}

extern "C" fn write_field(
    stack_pointer: usize,
    frame_pointer: usize,
    struct_pointer: usize,
    str_constant_ptr: usize,
    property_cache_location: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    // write_field checks mutability and throws if not mutable
    let index = runtime.write_field(stack_pointer, struct_pointer, str_constant_ptr, value);
    // Write barrier for generational GC - mark the card containing this object
    runtime.mark_card_for_object(struct_pointer);
    let type_id = HeapObject::from_tagged(struct_pointer).get_struct_id();
    // Cache layout: [struct_id, field_offset, is_mutable]
    // We only reach here if field is mutable (otherwise write_field would have thrown)
    let buffer = unsafe { from_raw_parts_mut(property_cache_location as *mut usize, 3) };
    buffer[0] = type_id;
    buffer[1] = index * 8;
    buffer[2] = 1; // is_mutable = true
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn throw_error(stack_pointer: usize, frame_pointer: usize) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);
    print_stack(stack_pointer);
    panic!("Error!");
}

pub unsafe extern "C" fn throw_type_error(stack_pointer: usize, frame_pointer: usize) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);

    let (kind_str, message_str) = {
        let runtime = get_runtime().get_mut();
        let kind = runtime
            .allocate_string(stack_pointer, "TypeError".to_string())
            .expect("Failed to allocate kind string")
            .into();
        let msg = runtime
            .allocate_string(
                stack_pointer,
                "Type mismatch in arithmetic operation".to_string(),
            )
            .expect("Failed to allocate message string")
            .into();
        (kind, msg)
    };

    let null_location = BuiltInTypes::Null.tag(0) as usize;
    unsafe {
        let error = create_error(
            stack_pointer,
            frame_pointer,
            kind_str,
            message_str,
            null_location,
        );
        throw_exception(stack_pointer, frame_pointer, error);
    }
}

pub unsafe extern "C" fn check_arity(
    stack_pointer: usize,
    frame_pointer: usize,
    function_pointer: usize,
    expected_args: isize,
) -> isize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get();

    // Function pointer is tagged, need to untag
    let untagged_ptr = (function_pointer >> BuiltInTypes::tag_size()) as *const u8;

    if let Some(function) = runtime.get_function_by_pointer(untagged_ptr) {
        // expected_args is a tagged integer, need to untag it
        let expected_args_untagged = BuiltInTypes::untag(expected_args as usize);

        if function.is_variadic {
            // For variadic functions, we need at least min_args arguments
            if expected_args_untagged < function.min_args {
                println!(
                    "Arity mismatch for variadic '{}': expected at least {} args, got {}",
                    function.name, function.min_args, expected_args_untagged
                );
                unsafe {
                    throw_error(stack_pointer, frame_pointer);
                }
            }
        } else {
            // For non-variadic functions, exact match required
            if function.number_of_args != expected_args_untagged {
                println!(
                    "Arity mismatch for '{}': expected {} args, got {}",
                    function.name, function.number_of_args, expected_args_untagged
                );
                unsafe {
                    throw_error(stack_pointer, frame_pointer);
                }
            }
        }
    }

    0 // Return value unused
}

/// Check if a function is variadic. Returns tagged boolean.
pub unsafe extern "C" fn is_function_variadic(function_pointer: usize) -> usize {
    let runtime = get_runtime().get();

    // Function pointer is tagged, need to untag
    let untagged_ptr = (function_pointer >> BuiltInTypes::tag_size()) as *const u8;

    if let Some(function) = runtime.get_function_by_pointer(untagged_ptr) {
        if function.is_variadic {
            BuiltInTypes::true_value() as usize
        } else {
            BuiltInTypes::false_value() as usize
        }
    } else {
        BuiltInTypes::false_value() as usize
    }
}

/// Get the min_args for a variadic function. Returns tagged int.
pub unsafe extern "C" fn get_function_min_args(function_pointer: usize) -> usize {
    let runtime = get_runtime().get();

    // Function pointer is tagged, need to untag
    let untagged_ptr = (function_pointer >> BuiltInTypes::tag_size()) as *const u8;

    if let Some(function) = runtime.get_function_by_pointer(untagged_ptr) {
        BuiltInTypes::Int.tag(function.min_args as isize) as usize
    } else {
        BuiltInTypes::Int.tag(0) as usize
    }
}

/// Build rest array from saved locals.
/// Used by variadic function prologues to build the rest parameter array
/// from arguments that were saved to dedicated local slots.
///
/// Arguments:
/// - stack_pointer: for GC
/// - frame_pointer: used to compute local addresses
/// - arg_count: number of args passed (tagged int, from X9)
/// - min_args: minimum args before rest param (tagged int, compile-time constant)
/// - first_local_index: index of first saved arg local (untagged raw value)
///
/// Returns: tagged array pointer containing args[min_args..arg_count]
pub unsafe extern "C" fn build_rest_array_from_locals(
    stack_pointer: usize,
    frame_pointer: usize,
    arg_count: usize,
    min_args: usize,
    first_local_index: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let total = BuiltInTypes::untag(arg_count);
    let min = BuiltInTypes::untag(min_args);

    // Number of extra args to pack
    let num_extra = total.saturating_sub(min);

    if num_extra == 0 {
        // Return empty array
        let array_ptr = runtime
            .allocate_zeroed(0, stack_pointer, BuiltInTypes::HeapObject)
            .unwrap();
        let mut heap_obj = HeapObject::from_tagged(array_ptr);
        heap_obj.write_type_id(1);
        return array_ptr;
    }

    // Allocate array for extra args (zeroed)
    let array_ptr = runtime
        .allocate_zeroed(num_extra, stack_pointer, BuiltInTypes::HeapObject)
        .unwrap();

    // Set type_id to 1 (raw array)
    let mut heap_obj = HeapObject::from_tagged(array_ptr);
    heap_obj.write_type_id(1);

    let array_data = heap_obj.untagged() as *mut usize;

    // Read args from locals and write to array
    // Local N is at FP - (N + 1) * 8
    // We saved args starting at local first_local_index
    // The saved args are: arg[first_arg_index], arg[first_arg_index+1], ...
    // We want args[min..total], which are at local indices:
    //   first_local_index + (min - first_arg_index), first_local_index + (min - first_arg_index) + 1, ...
    // But since first_arg_index is typically 0 (or 1 for closures), and the saved args
    // already account for this offset, we can simplify:
    // Saved local at index i corresponds to arg (first_arg_index + i)
    // We want arg indices min..total
    // So we want saved local indices (min - first_arg_index)..(total - first_arg_index)
    // Since first_arg_index is typically 0 for top-level functions, this simplifies to min..total
    // But for closures, first_arg_index = 1, so we want (min - 1)..(total - 1)

    // Actually, let's simplify: the caller passes min_args correctly adjusted
    // So we just read from saved_local[min]..saved_local[total]
    for i in 0..num_extra {
        let local_index = first_local_index + min + i;
        // Local address = FP - (local_index + 1) * 8
        let local_addr = frame_pointer.wrapping_sub((local_index + 1) * 8);
        // SAFETY: local_addr points to a valid stack slot that was written by the caller
        unsafe {
            let arg = *(local_addr as *const usize);
            // Write to array field i (offset 1 to skip header)
            *array_data.add(i + 1) = arg;
        }
    }

    array_ptr
}

/// Pack variadic arguments from stack into an array.
/// stack_pointer: current stack pointer
/// args_base: pointer to first arg on stack (args are contiguous)
/// total_args: total number of args (tagged int)
/// min_args: minimum args before rest param (tagged int)
/// Returns: tagged array pointer containing args[min_args..total_args]
pub unsafe extern "C" fn pack_variadic_args_from_stack(
    stack_pointer: usize,
    frame_pointer: usize,
    args_base: usize,
    total_args: usize,
    min_args: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    let total = BuiltInTypes::untag(total_args);
    let min = BuiltInTypes::untag(min_args);

    // Number of extra args to pack
    let num_extra = total.saturating_sub(min);

    // Allocate array for extra args (zeroed)
    let array_ptr = runtime
        .allocate_zeroed(num_extra, stack_pointer, BuiltInTypes::HeapObject)
        .unwrap();

    // Set type_id to 1 (raw array)
    let mut heap_obj = HeapObject::from_tagged(array_ptr);
    heap_obj.write_type_id(1);

    // Copy extra args from stack to array
    // Args on stack are at args_base, args_base+8, args_base+16, etc.
    let args_ptr = args_base as *const usize;
    let array_data = heap_obj.untagged() as *mut usize;

    for i in 0..num_extra {
        // Read arg at index (min + i) from args on stack
        // SAFETY: args_ptr and array_data are valid pointers set up by the caller,
        // and indices are bounds-checked via num_extra calculation
        unsafe {
            let arg = *args_ptr.add(min + i);
            // Write to array field i (offset 1 to skip header)
            *array_data.add(i + 1) = arg;
        }
    }

    array_ptr
}

/// Call a variadic function through a function value.
/// This handles the min_args dispatch at runtime, avoiding complex IR branching
/// that confuses the register allocator.
///
/// Arguments:
/// - stack_pointer: For GC safety during allocation
/// - frame_pointer: For GC stack walking
/// - function_ptr: Tagged function pointer
/// - args_array_ptr: Tagged pointer to array containing all call arguments
/// - is_closure: Tagged boolean - true if calling through a closure
/// - closure_ptr: Tagged closure pointer (or null if not a closure)
///
/// Returns: The function's return value
pub unsafe extern "C" fn call_variadic_function_value(
    stack_pointer: usize,
    frame_pointer: usize,
    function_ptr: usize,
    args_array_ptr: usize,
    is_closure: usize,
    closure_ptr: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get();

    // Get function metadata
    let untagged_fn = (function_ptr >> BuiltInTypes::tag_size()) as *const u8;
    let function = runtime
        .get_function_by_pointer(untagged_fn)
        .expect("Invalid function pointer in call_variadic_function_value");
    let min_args = function.min_args;
    let is_closure_bool = is_closure == BuiltInTypes::true_value() as usize;

    // Get arg count before allocation (allocation can trigger GC)
    let args_heap = HeapObject::from_tagged(args_array_ptr);
    let total_args = args_heap.get_fields().len();

    // Build rest array for args[min_args..]
    let num_extra = total_args.saturating_sub(min_args);

    // Register args_array as a temporary root before allocation
    let runtime_mut = get_runtime().get_mut();
    let args_root_id = runtime_mut.register_temporary_root(args_array_ptr);

    // Allocate array for extra args (can trigger GC, zeroed)
    let rest_array = runtime_mut
        .allocate_zeroed(num_extra, stack_pointer, BuiltInTypes::HeapObject)
        .unwrap();

    // Get the updated args_array_ptr from the root (GC may have moved it)
    // then unregister the root
    let args_array_ptr = runtime_mut.unregister_temporary_root(args_root_id);

    // Set type_id to 1 (raw array)
    let mut rest_heap = HeapObject::from_tagged(rest_array);
    rest_heap.write_type_id(1);

    // Now get the args fields - using the updated pointer after potential GC
    let args_heap = HeapObject::from_tagged(args_array_ptr);
    let all_args = args_heap.get_fields();

    // Copy extra args into rest array
    let rest_fields = rest_heap.get_fields_mut();
    rest_fields[..num_extra].copy_from_slice(&all_args[min_args..(num_extra + min_args)]);

    // Dispatch based on (min_args, is_closure)
    // We support min_args 0-7 which covers typical use cases
    // SAFETY: We're transmuting function pointers to call them with the correct number of arguments.
    // The function pointer comes from get_function_by_pointer which validates it's a known function.
    unsafe {
        match (min_args, is_closure_bool) {
            (0, false) => {
                let f: fn(usize) -> usize = transmute(untagged_fn);
                f(rest_array)
            }
            (0, true) => {
                let f: fn(usize, usize) -> usize = transmute(untagged_fn);
                f(closure_ptr, rest_array)
            }
            (1, false) => {
                let f: fn(usize, usize) -> usize = transmute(untagged_fn);
                f(all_args[0], rest_array)
            }
            (1, true) => {
                let f: fn(usize, usize, usize) -> usize = transmute(untagged_fn);
                f(closure_ptr, all_args[0], rest_array)
            }
            (2, false) => {
                let f: fn(usize, usize, usize) -> usize = transmute(untagged_fn);
                f(all_args[0], all_args[1], rest_array)
            }
            (2, true) => {
                let f: fn(usize, usize, usize, usize) -> usize = transmute(untagged_fn);
                f(closure_ptr, all_args[0], all_args[1], rest_array)
            }
            (3, false) => {
                let f: fn(usize, usize, usize, usize) -> usize = transmute(untagged_fn);
                f(all_args[0], all_args[1], all_args[2], rest_array)
            }
            (3, true) => {
                let f: fn(usize, usize, usize, usize, usize) -> usize = transmute(untagged_fn);
                f(
                    closure_ptr,
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    rest_array,
                )
            }
            (4, false) => {
                let f: fn(usize, usize, usize, usize, usize) -> usize = transmute(untagged_fn);
                f(
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    rest_array,
                )
            }
            (4, true) => {
                let f: fn(usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    closure_ptr,
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    rest_array,
                )
            }
            (5, false) => {
                let f: fn(usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    all_args[4],
                    rest_array,
                )
            }
            (5, true) => {
                let f: fn(usize, usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    closure_ptr,
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    all_args[4],
                    rest_array,
                )
            }
            (6, false) => {
                let f: fn(usize, usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    all_args[4],
                    all_args[5],
                    rest_array,
                )
            }
            (6, true) => {
                let f: fn(usize, usize, usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    closure_ptr,
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    all_args[4],
                    all_args[5],
                    rest_array,
                )
            }
            (7, false) => {
                let f: fn(usize, usize, usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    all_args[4],
                    all_args[5],
                    all_args[6],
                    rest_array,
                )
            }
            (7, true) => {
                let f: fn(usize, usize, usize, usize, usize, usize, usize, usize, usize) -> usize =
                    transmute(untagged_fn);
                f(
                    closure_ptr,
                    all_args[0],
                    all_args[1],
                    all_args[2],
                    all_args[3],
                    all_args[4],
                    all_args[5],
                    all_args[6],
                    rest_array,
                )
            }
            _ => panic!(
                "Unsupported min_args value {} for variadic function call through function value",
                min_args
            ),
        }
    }
}

fn print_stack(_stack_pointer: usize) {
    let runtime = get_runtime().get_mut();
    let stack_base = runtime.get_stack_base();
    let stack_begin = stack_base - STACK_SIZE;

    // Get the current frame pointer directly from the register
    let fp: usize;
    let mut current_frame_ptr = unsafe {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                asm!(
                    "mov {0}, rbp",
                    out(reg) fp
                );
            } else {
                asm!(
                    "mov {0}, x29",
                    out(reg) fp
                );
            }
        }
        fp
    };

    println!("Walking stack frames:");

    let mut frame_count = 0;
    const MAX_FRAMES: usize = 100; // Prevent infinite loops

    while frame_count < MAX_FRAMES && current_frame_ptr != 0 {
        // Validate frame pointer is within stack bounds
        if current_frame_ptr < stack_begin || current_frame_ptr >= stack_base {
            break;
        }

        // Frame layout in our calling convention:
        // [previous_X29] [X30/return_addr] <- X29 points here
        // [zero] [zero] [locals...]
        let frame = current_frame_ptr as *const usize;
        let previous_frame_ptr = unsafe { *frame.offset(0) }; // Previous X29
        let return_address = unsafe { *frame.offset(1) }; // X30 (return address)

        // Look up the function containing this return address
        for function in runtime.functions.iter() {
            let function_size = function.size;
            let function_start = usize::from(function.pointer);
            let range = function_start..function_start + function_size;
            if range.contains(&return_address) {
                println!("Function: {:?}", function.name);
                break;
            }
        }

        // Move to the previous frame
        current_frame_ptr = previous_frame_ptr;
        frame_count += 1;
    }
}

pub unsafe extern "C" fn gc(stack_pointer: usize, frame_pointer: usize) -> usize {
    // Save the GC context including the return address
    save_gc_context!(stack_pointer, frame_pointer);
    let gc_return_addr = get_saved_gc_return_addr();
    #[cfg(feature = "debug-gc")]
    {
        eprintln!(
            "DEBUG gc: stack_pointer={:#x}, frame_pointer={:#x}, gc_return_addr={:#x}",
            stack_pointer, frame_pointer, gc_return_addr
        );
    }
    let runtime = get_runtime().get_mut();
    runtime.gc_impl(stack_pointer, frame_pointer, gc_return_addr);
    BuiltInTypes::null_value() as usize
}

/// sqrt builtin - computes square root of a float
/// Takes a tagged float pointer, returns a new tagged float pointer with the result
pub unsafe extern "C" fn sqrt_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        // Read the float value from the heap object
        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1); // Float value is at offset 1 (after header)

        // Compute sqrt
        let result = float_value.sqrt();

        // Allocate a new float object for the result
        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        // Write the result
        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// floor builtin - computes floor of a float
pub unsafe extern "C" fn floor_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.floor();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// ceil builtin - computes ceiling of a float
pub unsafe extern "C" fn ceil_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.ceil();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// abs builtin - computes absolute value of a float
pub unsafe extern "C" fn abs_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.abs();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// round builtin - rounds a float to nearest integer
pub unsafe extern "C" fn round_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.round();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// truncate builtin - truncates a float towards zero
pub unsafe extern "C" fn truncate_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.trunc();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// max builtin - returns the maximum of two numbers
extern "C" fn max_builtin(stack_pointer: usize, frame_pointer: usize, a: usize, b: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);

    let a_kind = BuiltInTypes::get_kind(a);
    let b_kind = BuiltInTypes::get_kind(b);

    // Both are ints
    if a_kind == BuiltInTypes::Int && b_kind == BuiltInTypes::Int {
        let a_val = BuiltInTypes::untag_isize(a as isize);
        let b_val = BuiltInTypes::untag_isize(b as isize);
        return BuiltInTypes::construct_int(a_val.max(b_val)) as usize;
    }

    // At least one is a float - convert both to float and compare
    unsafe {
        let a_float = if a_kind == BuiltInTypes::Float {
            let untagged = BuiltInTypes::untag(a);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        } else {
            BuiltInTypes::untag_isize(a as isize) as f64
        };

        let b_float = if b_kind == BuiltInTypes::Float {
            let untagged = BuiltInTypes::untag(b);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        } else {
            BuiltInTypes::untag_isize(b as isize) as f64
        };

        let result = a_float.max(b_float);

        let runtime = get_runtime().get_mut();
        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// min builtin - returns the minimum of two numbers
extern "C" fn min_builtin(stack_pointer: usize, frame_pointer: usize, a: usize, b: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);

    let a_kind = BuiltInTypes::get_kind(a);
    let b_kind = BuiltInTypes::get_kind(b);

    // Both are ints
    if a_kind == BuiltInTypes::Int && b_kind == BuiltInTypes::Int {
        let a_val = BuiltInTypes::untag_isize(a as isize);
        let b_val = BuiltInTypes::untag_isize(b as isize);
        return BuiltInTypes::construct_int(a_val.min(b_val)) as usize;
    }

    // At least one is a float - convert both to float and compare
    unsafe {
        let a_float = if a_kind == BuiltInTypes::Float {
            let untagged = BuiltInTypes::untag(a);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        } else {
            BuiltInTypes::untag_isize(a as isize) as f64
        };

        let b_float = if b_kind == BuiltInTypes::Float {
            let untagged = BuiltInTypes::untag(b);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        } else {
            BuiltInTypes::untag_isize(b as isize) as f64
        };

        let result = a_float.min(b_float);

        let runtime = get_runtime().get_mut();
        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// clamp builtin - clamps a value between low and high bounds
extern "C" fn clamp_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
    low: usize,
    high: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let value_kind = BuiltInTypes::get_kind(value);
        let low_kind = BuiltInTypes::get_kind(low);
        let high_kind = BuiltInTypes::get_kind(high);

        // If all are integers, do integer clamping
        if value_kind == BuiltInTypes::Int
            && low_kind == BuiltInTypes::Int
            && high_kind == BuiltInTypes::Int
        {
            let value_int = BuiltInTypes::untag_isize(value as isize);
            let low_int = BuiltInTypes::untag_isize(low as isize);
            let high_int = BuiltInTypes::untag_isize(high as isize);

            let clamped = value_int.max(low_int).min(high_int);
            return BuiltInTypes::construct_int(clamped) as usize;
        }

        // Otherwise, convert to float and do float clamping
        let value_float = if value_kind == BuiltInTypes::Int {
            BuiltInTypes::untag_isize(value as isize) as f64
        } else {
            let untagged = BuiltInTypes::untag(value);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        };

        let low_float = if low_kind == BuiltInTypes::Int {
            BuiltInTypes::untag_isize(low as isize) as f64
        } else {
            let untagged = BuiltInTypes::untag(low);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        };

        let high_float = if high_kind == BuiltInTypes::Int {
            BuiltInTypes::untag_isize(high as isize) as f64
        } else {
            let untagged = BuiltInTypes::untag(high);
            let float_ptr = untagged as *const f64;
            *float_ptr.add(1)
        };

        let result = value_float.max(low_float).min(high_float);

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// gcd builtin - computes greatest common divisor of two integers
extern "C" fn gcd_builtin(a: usize, b: usize) -> usize {
    let mut a_val = BuiltInTypes::untag_isize(a as isize).abs();
    let mut b_val = BuiltInTypes::untag_isize(b as isize).abs();

    // Euclidean algorithm
    while b_val != 0 {
        let temp = b_val;
        b_val = a_val % b_val;
        a_val = temp;
    }

    BuiltInTypes::construct_int(a_val) as usize
}

/// lcm builtin - computes least common multiple of two integers
extern "C" fn lcm_builtin(a: usize, b: usize) -> usize {
    let a_val = BuiltInTypes::untag_isize(a as isize).abs();
    let b_val = BuiltInTypes::untag_isize(b as isize).abs();

    if a_val == 0 || b_val == 0 {
        return BuiltInTypes::construct_int(0) as usize;
    }

    // LCM(a,b) = |a*b| / GCD(a,b)
    let gcd_result = gcd_builtin(a, b);
    let gcd_val = BuiltInTypes::untag_isize(gcd_result as isize);

    let lcm = (a_val * b_val) / gcd_val;

    BuiltInTypes::construct_int(lcm) as usize
}

/// random builtin - returns a random float between 0.0 and 1.0
pub unsafe extern "C" fn random_builtin(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();
        let mut rng = rand::thread_rng();
        let random_value: f64 = rng.gen_range(0.0..1.0);

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = random_value;

        new_float_ptr
    }
}

/// random-int builtin - returns a random integer from 0 to max-1
extern "C" fn random_int_builtin(max: usize) -> usize {
    let max_val = BuiltInTypes::untag_isize(max as isize);

    if max_val <= 0 {
        return BuiltInTypes::construct_int(0) as usize;
    }

    let mut rng = rand::thread_rng();
    let random_value: isize = rng.gen_range(0..max_val);

    BuiltInTypes::construct_int(random_value) as usize
}

/// random-range builtin - returns a random integer from min to max-1
extern "C" fn random_range_builtin(min: usize, max: usize) -> usize {
    let min_val = BuiltInTypes::untag_isize(min as isize);
    let max_val = BuiltInTypes::untag_isize(max as isize);

    if min_val >= max_val {
        return BuiltInTypes::construct_int(min_val) as usize;
    }

    let mut rng = rand::thread_rng();
    let random_value: isize = rng.gen_range(min_val..max_val);

    BuiltInTypes::construct_int(random_value) as usize
}

/// even? predicate - checks if an integer is even
extern "C" fn is_even(value: usize) -> usize {
    let value_kind = BuiltInTypes::get_kind(value);
    if value_kind == BuiltInTypes::Int {
        let int_val = BuiltInTypes::untag_isize(value as isize);
        BuiltInTypes::construct_boolean(int_val % 2 == 0) as usize
    } else {
        BuiltInTypes::construct_boolean(false) as usize
    }
}

/// odd? predicate - checks if an integer is odd
extern "C" fn is_odd(value: usize) -> usize {
    let value_kind = BuiltInTypes::get_kind(value);
    if value_kind == BuiltInTypes::Int {
        let int_val = BuiltInTypes::untag_isize(value as isize);
        BuiltInTypes::construct_boolean(int_val % 2 != 0) as usize
    } else {
        BuiltInTypes::construct_boolean(false) as usize
    }
}

/// positive? predicate - checks if a number is positive
extern "C" fn is_positive(value: usize) -> usize {
    let value_kind = BuiltInTypes::get_kind(value);

    if value_kind == BuiltInTypes::Int {
        let int_val = BuiltInTypes::untag_isize(value as isize);
        BuiltInTypes::construct_boolean(int_val > 0) as usize
    } else if value_kind == BuiltInTypes::Float {
        unsafe {
            let untagged = BuiltInTypes::untag(value);
            let float_ptr = untagged as *const f64;
            let float_value = *float_ptr.add(1);
            BuiltInTypes::construct_boolean(float_value > 0.0) as usize
        }
    } else {
        BuiltInTypes::construct_boolean(false) as usize
    }
}

/// negative? predicate - checks if a number is negative
extern "C" fn is_negative(value: usize) -> usize {
    let value_kind = BuiltInTypes::get_kind(value);

    if value_kind == BuiltInTypes::Int {
        let int_val = BuiltInTypes::untag_isize(value as isize);
        BuiltInTypes::construct_boolean(int_val < 0) as usize
    } else if value_kind == BuiltInTypes::Float {
        unsafe {
            let untagged = BuiltInTypes::untag(value);
            let float_ptr = untagged as *const f64;
            let float_value = *float_ptr.add(1);
            BuiltInTypes::construct_boolean(float_value < 0.0) as usize
        }
    } else {
        BuiltInTypes::construct_boolean(false) as usize
    }
}

/// zero? predicate - checks if a number is zero
extern "C" fn is_zero(value: usize) -> usize {
    let value_kind = BuiltInTypes::get_kind(value);

    if value_kind == BuiltInTypes::Int {
        let int_val = BuiltInTypes::untag_isize(value as isize);
        BuiltInTypes::construct_boolean(int_val == 0) as usize
    } else if value_kind == BuiltInTypes::Float {
        unsafe {
            let untagged = BuiltInTypes::untag(value);
            let float_ptr = untagged as *const f64;
            let float_value = *float_ptr.add(1);
            BuiltInTypes::construct_boolean(float_value == 0.0) as usize
        }
    } else {
        BuiltInTypes::construct_boolean(false) as usize
    }
}

/// sin builtin - computes sine of a float (in radians)
pub unsafe extern "C" fn sin_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.sin();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// cos builtin - computes cosine of a float (in radians)
pub unsafe extern "C" fn cos_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.cos();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// tan builtin - computes tangent of a float (in radians)
pub unsafe extern "C" fn tan_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.tan();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// asin builtin - computes arcsine of a float (returns radians)
pub unsafe extern "C" fn asin_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.asin();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// acos builtin - computes arccosine of a float (returns radians)
pub unsafe extern "C" fn acos_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.acos();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// atan builtin - computes arctangent of a float (returns radians)
pub unsafe extern "C" fn atan_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.atan();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// atan2 builtin - computes arctangent of y/x (returns radians)
pub unsafe extern "C" fn atan2_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    y: usize,
    x: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let y_untagged = BuiltInTypes::untag(y);
        let y_float_ptr = y_untagged as *const f64;
        let y_value = *y_float_ptr.add(1);

        let x_untagged = BuiltInTypes::untag(x);
        let x_float_ptr = x_untagged as *const f64;
        let x_value = *x_float_ptr.add(1);

        let result = y_value.atan2(x_value);

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// exp builtin - computes e^x
pub unsafe extern "C" fn exp_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.exp();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// log builtin - computes natural logarithm (base e)
pub unsafe extern "C" fn log_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.ln();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// log10 builtin - computes base-10 logarithm
pub unsafe extern "C" fn log10_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.log10();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// log2 builtin - computes base-2 logarithm
pub unsafe extern "C" fn log2_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let untagged = BuiltInTypes::untag(value);
        let float_ptr = untagged as *const f64;
        let float_value = *float_ptr.add(1);

        let result = float_value.log2();

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// pow builtin - computes base^exponent
pub unsafe extern "C" fn pow_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    base: usize,
    exponent: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        let base_untagged = BuiltInTypes::untag(base);
        let base_float_ptr = base_untagged as *const f64;
        let base_value = *base_float_ptr.add(1);

        let exp_untagged = BuiltInTypes::untag(exponent);
        let exp_float_ptr = exp_untagged as *const f64;
        let exp_value = *exp_float_ptr.add(1);

        let result = base_value.powf(exp_value);

        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = result;

        new_float_ptr
    }
}

/// to_float builtin - converts an integer to a float
pub unsafe extern "C" fn to_float_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    unsafe {
        let runtime = get_runtime().get_mut();

        // Get the integer value (tagged integers have 0b000 tag)
        // Use signed right shift to preserve negative numbers
        let int_value = BuiltInTypes::untag_isize(value as isize) as i64;

        // Convert to f64
        let float_value = int_value as f64;

        // Allocate a new float object for the result
        let new_float_ptr = runtime
            .allocate(1, stack_pointer, BuiltInTypes::Float)
            .unwrap();

        let untagged_result = BuiltInTypes::untag(new_float_ptr);
        let result_ptr = untagged_result as *mut f64;
        *result_ptr.add(1) = float_value;

        new_float_ptr
    }
}

#[allow(unused)]
pub unsafe extern "C" fn new_thread(
    stack_pointer: usize,
    frame_pointer: usize,
    function: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    #[cfg(feature = "thread-safe")]
    {
        let runtime = get_runtime().get_mut();
        runtime.new_thread(function, stack_pointer, frame_pointer);
        BuiltInTypes::null_value() as usize
    }
    #[cfg(not(feature = "thread-safe"))]
    {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "ThreadError",
                "Threads are not supported in this build".to_string(),
            );
        }
    }
}

// I don't know what the deal is here

#[unsafe(no_mangle)]
pub unsafe extern "C" fn update_binding(namespace_slot: usize, value: usize) -> usize {
    print_call_builtin(get_runtime().get(), "update_binding");
    let runtime = get_runtime().get_mut();
    let namespace_slot = BuiltInTypes::untag(namespace_slot);
    let namespace_id = runtime.current_namespace_id();

    // Store binding in heap-based PersistentMap (no namespace_roots tracking needed!)
    let stack_pointer = get_current_stack_pointer();
    if let Err(e) = runtime.set_heap_binding(stack_pointer, namespace_id, namespace_slot, value) {
        eprintln!("Error in update_binding: {}", e);
    }

    // Also update the Rust-side HashMap for backwards compatibility during migration
    runtime.update_binding(namespace_id, namespace_slot, value);

    BuiltInTypes::null_value() as usize
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_binding(namespace: usize, slot: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let namespace = BuiltInTypes::untag(namespace);
    let slot = BuiltInTypes::untag(slot);

    // TODO: Flush pending bindings at a safer point (not during get_binding)
    // For now, rely on HashMap fallback for bindings added by compiler thread
    // let stack_pointer = get_current_stack_pointer();
    // runtime.flush_pending_heap_bindings(stack_pointer);

    // Try heap-based PersistentMap first
    let result = runtime.get_heap_binding(namespace, slot);
    if result != BuiltInTypes::null_value() as usize {
        return result;
    }

    // Fall back to Rust-side HashMap during migration
    runtime.get_binding(namespace, slot)
}

pub unsafe extern "C" fn set_current_namespace(namespace: usize) -> usize {
    let runtime = get_runtime().get_mut();
    runtime.set_current_namespace(namespace);
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn __pause(_stack_pointer: usize, frame_pointer: usize) -> usize {
    use crate::gc::usdt_probes::{self, ThreadStateCode};

    // Get the gc_return_addr. When called from Rust code (get_my_thread_obj, gc_impl),
    // save_gc_context! was already called with the correct Beagle return address.
    // When called directly from Beagle (via primitive/__pause), we capture it ourselves.
    //
    // To distinguish these cases, we compare the Beagle frame_pointer we received
    // with the saved frame_pointer from save_gc_context!. If they match, we're in
    // the same Beagle call chain and the saved gc_return_addr is valid.
    let saved_beagle_fp = get_saved_frame_pointer();
    let my_rust_fp = get_current_rust_frame_pointer();
    let captured_addr = unsafe { *((my_rust_fp + 8) as *const usize) };
    let saved_addr = get_saved_gc_return_addr();

    let gc_return_addr = if saved_beagle_fp != 0 && saved_beagle_fp == frame_pointer {
        // Same Beagle frame as when save_gc_context! was called - use saved addr
        if std::env::var("BEAGLE_PAUSE_DEBUG").is_ok() {
            eprintln!(
                "[PAUSE_DEBUG] Using saved gc_return_addr={:#x} (saved_fp={:#x} == frame_pointer={:#x})",
                saved_addr, saved_beagle_fp, frame_pointer
            );
        }
        saved_addr
    } else {
        // Called directly from Beagle with a different/no saved frame - capture our own
        if std::env::var("BEAGLE_PAUSE_DEBUG").is_ok() {
            eprintln!(
                "[PAUSE_DEBUG] Using captured gc_return_addr={:#x} (saved_fp={:#x} != frame_pointer={:#x})",
                captured_addr, saved_beagle_fp, frame_pointer
            );
        }
        captured_addr
    };

    let runtime = get_runtime().get_mut();

    // Fire USDT probe for thread state change
    usdt_probes::fire_thread_pause_enter();
    usdt_probes::fire_thread_state(ThreadStateCode::PausedForGc);

    let pause_start = std::time::Instant::now();

    // Use frame_pointer passed from Beagle code for FP-chain stack walking
    pause_current_thread(frame_pointer, gc_return_addr, runtime);

    // Memory barrier to ensure all writes are visible before parking
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    while runtime.is_paused() {
        // Park can unpark itself even if I haven't called unpark
        thread::park();
    }

    // Apparently, I can't count on this not unparking
    // I need some other mechanism to know that things are ready
    unpause_current_thread(runtime);

    let pause_duration_ns = pause_start.elapsed().as_nanos() as u64;

    // Fire USDT probe for thread resuming
    usdt_probes::fire_thread_pause_exit(pause_duration_ns);
    usdt_probes::fire_thread_state(ThreadStateCode::Running);

    // Memory barrier to ensure all GC updates are visible before continuing
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    BuiltInTypes::null_value() as usize
}

fn pause_current_thread(frame_pointer: usize, gc_return_addr: usize, runtime: &mut Runtime) {
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    let stack_base = runtime.get_stack_base();
    // Store (stack_base, frame_pointer, gc_return_addr) for stack walking
    state.pause((stack_base, frame_pointer, gc_return_addr));
    condvar.notify_one();
    drop(state);
}

fn unpause_current_thread(runtime: &mut Runtime) {
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    state.unpause();
    condvar.notify_one();
}

pub extern "C" fn register_c_call(_stack_pointer: usize, frame_pointer: usize) -> usize {
    // Capture our return address - this is the safepoint in Beagle code
    let gc_return_addr = get_current_rust_frame_pointer();
    let gc_return_addr = unsafe { *((gc_return_addr + 8) as *const usize) };

    // Use frame_pointer passed from Beagle code for FP-chain stack walking
    let runtime = get_runtime().get_mut();
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    let stack_base = runtime.get_stack_base();
    state.register_c_call((stack_base, frame_pointer, gc_return_addr));
    condvar.notify_one();
    BuiltInTypes::null_value() as usize
}

pub extern "C" fn unregister_c_call() -> usize {
    let runtime = get_runtime().get_mut();
    let thread_state = runtime.thread_state.clone();
    let (lock, condvar) = &*thread_state;
    let mut state = lock.lock().unwrap();
    state.unregister_c_call();
    condvar.notify_one();
    while runtime.is_paused() {
        // Park can unpark itself even if I haven't called unpark
        thread::park();
    }
    BuiltInTypes::null_value() as usize
}

/// Called from a newly spawned thread to safely get a closure from a temporary root and call it.
/// This function:
/// 1. Unregisters from C-call (so this thread can participate in GC safepoints)
/// 2. Peeks the current closure value from the temporary root (which may have been updated by GC)
/// 3. Unregisters the temporary root
/// 4. Calls the closure via __call_fn
///
/// The temporary_root_id is passed as a tagged integer.
/// Get a value from a temporary root and unregister it.
/// This is called from Beagle code after entering a managed context.
pub extern "C" fn get_and_unregister_temp_root(temporary_root_id: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let root_id = BuiltInTypes::untag(temporary_root_id);
    // Read and unregister in one operation
    runtime.unregister_temporary_root(root_id)
}

/// Run a thread by calling the no-argument __run_thread_start function.
/// The Thread object is accessed via the get_my_thread_obj builtin inside Beagle code,
/// ensuring GC can update stack slots properly.
///
/// Key invariant: We are NOT counted in registered_thread_count until we hold gc_lock
/// and increment it. This prevents GC from waiting for us before we're ready.
///
/// Flow:
/// 1. Acquire gc_lock (GC can't start while we hold it)
/// 2. Increment registered_thread_count (now GC will count us)
/// 3. Release gc_lock
/// 4. Run Beagle code (first instruction is __pause, handles any pending GC)
/// 5. On cleanup: acquire gc_lock, decrement count, remove thread root, release
pub extern "C" fn run_thread(_unused: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let my_thread_id = thread::current().id();

    // === STARTUP ===
    // Acquire gc_lock and register ourselves.
    // We use blocking lock() here because we are NOT yet counted in registered_thread_count,
    // so GC won't wait for us - no deadlock possible.
    {
        let _guard = runtime.gc_lock.lock().unwrap();
        // Now register ourselves - GC will count us from this point on
        let new_count = runtime
            .registered_thread_count
            .fetch_add(1, std::sync::atomic::Ordering::Release)
            + 1;
        // Fire USDT probes for thread start and registration
        crate::gc::usdt_probes::fire_thread_register(new_count);
        crate::gc::usdt_probes::fire_thread_start();
        // Lock released here - GC can now start, and it will count us
    }

    // Enter Beagle code - __run_thread_start calls __pause as first instruction!
    // If GC started right after we released gc_lock, __pause will handle it.
    let result = unsafe { call_fn_0(runtime, "beagle.core/__run_thread_start") };

    // === CLEANUP ===
    // We're still registered but we're in C code now, not Beagle code.
    // If GC starts, it will wait for us to pause, but we can't pause from C.
    // Solution: register as c_calling so GC counts us and proceeds.
    {
        let (lock, condvar) = &*runtime.thread_state.clone();
        let mut state = lock.lock().unwrap();
        state.register_c_call((0, 0, 0)); // No stack to scan
        condvar.notify_one();
    }

    // Now any GC will count us as c_calling and proceed.
    // Wait for any in-progress GC to finish, then unregister everything.
    loop {
        while runtime.is_paused() {
            thread::yield_now();
        }

        match runtime.gc_lock.try_lock() {
            Ok(_guard) => {
                // While holding lock: unregister from c_calling, decrement count, remove root
                {
                    let (lock, condvar) = &*runtime.thread_state.clone();
                    let mut state = lock.lock().unwrap();
                    state.unregister_c_call();
                    condvar.notify_one();
                }
                let new_count = runtime
                    .registered_thread_count
                    .fetch_sub(1, std::sync::atomic::Ordering::Release)
                    - 1;
                // Thread object is in our GlobalObjectBlock - cleanup happens
                // when thread_globals entry is removed
                runtime
                    .memory
                    .thread_globals
                    .lock()
                    .unwrap()
                    .remove(&my_thread_id);
                // Fire USDT probes for thread unregistration and exit
                crate::gc::usdt_probes::fire_thread_unregister(new_count);
                crate::gc::usdt_probes::fire_thread_exit();
                break;
            }
            Err(_) => {
                thread::yield_now();
            }
        }
    }

    result
}

/// Get the current thread's Thread object from its GlobalObjectBlock.
/// Called from Beagle code in __run_thread_start.
/// Takes stack_pointer and frame_pointer so we can call __pause if needed.
///
/// CRITICAL: This function must return a pointer that won't become stale before
/// the caller can use it. We check is_paused() after releasing the lock to ensure
/// that if GC is about to run, we pause and get the updated pointer.
pub extern "C" fn get_my_thread_obj(stack_pointer: usize, frame_pointer: usize) -> usize {
    // CRITICAL: Save the gc context here so that if we call __pause,
    // the GC will have the correct return address pointing to Beagle code.
    save_gc_context!(stack_pointer, frame_pointer);

    let runtime = get_runtime().get_mut();
    let thread_id = thread::current().id();

    // Read the Thread object from our GlobalObjectBlock's reserved slot.
    // We need the gc_lock to ensure the value isn't stale during GC.
    let thread_obj = loop {
        // Check if GC needs us to pause first
        if runtime.is_paused() {
            unsafe { __pause(stack_pointer, frame_pointer) };
            continue;
        }

        // Try to get the lock
        let obj = match runtime.gc_lock.try_lock() {
            Ok(_guard) => {
                let thread_globals = runtime.memory.thread_globals.lock().unwrap();
                thread_globals
                    .get(&thread_id)
                    .map(|tg| tg.get_thread_object())
                    .expect("ThreadGlobal not found in get_my_thread_obj")
            }
            Err(_) => {
                thread::yield_now();
                continue;
            }
        };

        // After releasing the lock, check if GC is pending.
        if runtime.is_paused() {
            unsafe { __pause(stack_pointer, frame_pointer) };
            continue;
        }

        break obj;
    };

    if std::env::var("BEAGLE_THREAD_DEBUG").is_ok() {
        eprintln!(
            "[THREAD_DEBUG] get_my_thread_obj: thread_id={:?} thread_obj={:#x}",
            thread_id, thread_obj
        );
        if BuiltInTypes::is_heap_pointer(thread_obj) {
            let heap_obj = HeapObject::from_tagged(thread_obj);
            let closure_field = heap_obj.get_field(0);
            eprintln!(
                "[THREAD_DEBUG]   closure_field={:#x} is_heap_ptr={}",
                closure_field,
                BuiltInTypes::is_heap_pointer(closure_field)
            );
            if BuiltInTypes::is_heap_pointer(closure_field) {
                let closure_tag = BuiltInTypes::get_kind(closure_field);
                eprintln!("[THREAD_DEBUG]   closure_tag={:?}", closure_tag);
                if matches!(closure_tag, BuiltInTypes::Closure) {
                    let closure_obj = HeapObject::from_tagged(closure_field);
                    let fn_ptr = closure_obj.get_field(0);
                    let fn_ptr_untagged = BuiltInTypes::untag(fn_ptr);
                    if let Some(function) =
                        runtime.get_function_by_pointer(fn_ptr_untagged as *const u8)
                    {
                        eprintln!(
                            "[THREAD_DEBUG]   closure fn={} args={}",
                            function.name, function.number_of_args
                        );
                    } else {
                        eprintln!("[THREAD_DEBUG]   closure fn ptr not found: {:#x}", fn_ptr);
                        panic!("Closure function pointer not found in runtime");
                    }
                }
            }
        }
    }

    thread_obj
}

pub unsafe fn call_fn_0(runtime: &Runtime, function_name: &str) -> usize {
    print_call_builtin(
        runtime,
        format!("{} {}", "call_fn_0", function_name).as_str(),
    );

    // Save GC context before calling into Beagle - the Beagle code may call builtins
    // that update the saved GC context, making it point to now-deallocated frames
    let saved_ctx = save_current_gc_context();

    let save_volatile_registers = runtime
        .get_function_by_name("beagle.builtin/save_volatile_registers0")
        .unwrap();
    let save_volatile_registers = runtime.get_pointer(save_volatile_registers).unwrap();
    let save_volatile_registers: fn(usize) -> usize =
        unsafe { std::mem::transmute(save_volatile_registers) };

    let function = runtime.get_function_by_name(function_name).unwrap();
    let function = runtime.get_pointer(function).unwrap();
    let result = save_volatile_registers(function as usize);

    // Restore GC context after Beagle call returns
    restore_gc_context(saved_ctx);
    result
}

pub unsafe fn call_fn_1(runtime: &Runtime, function_name: &str, arg1: usize) -> usize {
    print_call_builtin(
        runtime,
        format!("{} {}", "call_fn_1", function_name).as_str(),
    );

    // Save GC context before calling into Beagle - the Beagle code may call builtins
    // that update the saved GC context, making it point to now-deallocated frames
    let saved_ctx = save_current_gc_context();

    let save_volatile_registers = runtime
        .get_function_by_name("beagle.builtin/save_volatile_registers1")
        .unwrap();
    let save_volatile_registers = runtime.get_pointer(save_volatile_registers).unwrap();
    let save_volatile_registers: fn(usize, usize) -> usize =
        unsafe { std::mem::transmute(save_volatile_registers) };

    let function = runtime.get_function_by_name(function_name).unwrap();
    let function = runtime.get_pointer(function).unwrap();
    let result = save_volatile_registers(arg1, function as usize);

    // Restore GC context after Beagle call returns
    restore_gc_context(saved_ctx);
    result
}

pub unsafe fn call_fn_2(runtime: &Runtime, function_name: &str, arg1: usize, arg2: usize) -> usize {
    print_call_builtin(
        runtime,
        format!("{} {}", "call_fn_2", function_name).as_str(),
    );

    // Save GC context before calling into Beagle - the Beagle code may call builtins
    // that update the saved GC context, making it point to now-deallocated frames
    let saved_ctx = save_current_gc_context();

    let save_volatile_registers = runtime
        .get_function_by_name("beagle.builtin/save_volatile_registers2")
        .unwrap();
    let save_volatile_registers = runtime.get_pointer(save_volatile_registers).unwrap();
    let save_volatile_registers: fn(usize, usize, usize) -> usize =
        unsafe { std::mem::transmute(save_volatile_registers) };

    let function = runtime.get_function_by_name(function_name).unwrap();
    let function = runtime.get_pointer(function).unwrap();
    let result = save_volatile_registers(arg1, arg2, function as usize);

    // Restore GC context after Beagle call returns
    restore_gc_context(saved_ctx);
    result
}

// Helper to call a Beagle function or closure with one argument
pub unsafe fn call_beagle_fn_ptr(runtime: &Runtime, fn_or_closure: usize, arg1: usize) {
    // Save GC context before calling into Beagle - the Beagle code may call builtins
    // that update the saved GC context, making it point to now-deallocated frames
    let saved_ctx = save_current_gc_context();

    // Check if this is a closure (has closure tag)
    let kind = BuiltInTypes::get_kind(fn_or_closure);

    if matches!(kind, BuiltInTypes::Closure) {
        // It's a closure - extract function pointer
        // Closure structure: [header][function_ptr][num_free_vars][free_vars...]
        let untagged = BuiltInTypes::untag(fn_or_closure);
        let closure = HeapObject::from_untagged(untagged as *const u8);
        // Function pointer is at field 0, but it's tagged so need to untag it
        let fp_tagged = closure.get_field(0);
        let function_pointer = BuiltInTypes::untag(fp_tagged);

        // Call with closure as first arg, exception as second
        // Closures MUST receive the closure object itself as the first argument
        let save_volatile_registers = runtime
            .get_function_by_name("beagle.builtin/save_volatile_registers2")
            .unwrap();
        let save_volatile_registers = runtime.get_pointer(save_volatile_registers).unwrap();
        let save_volatile_registers: fn(usize, usize, usize) -> usize =
            unsafe { std::mem::transmute(save_volatile_registers) };

        save_volatile_registers(fn_or_closure, arg1, function_pointer);
    } else {
        // It's a regular function pointer - just pass exception
        let function_pointer = BuiltInTypes::untag(fn_or_closure);

        let save_volatile_registers = runtime
            .get_function_by_name("beagle.builtin/save_volatile_registers1")
            .unwrap();
        let save_volatile_registers = runtime.get_pointer(save_volatile_registers).unwrap();
        let save_volatile_registers: fn(usize, usize) -> usize =
            unsafe { std::mem::transmute(save_volatile_registers) };

        save_volatile_registers(arg1, function_pointer);
    }

    // Restore GC context after Beagle call returns
    restore_gc_context(saved_ctx);
}

pub unsafe extern "C" fn load_library(name: usize) -> usize {
    let runtime = get_runtime().get_mut();
    let string = &runtime.get_string_literal(name);
    let lib = unsafe {
        libloading::Library::new(string)
            .unwrap_or_else(|e| panic!("Failed to load library '{}': {}", string, e))
    };
    let id = runtime.add_library(lib);

    unsafe {
        call_fn_1(
            runtime,
            "beagle.ffi/__make_lib_struct",
            BuiltInTypes::Int.tag(id as isize) as usize,
        )
    }
}

pub fn map_beagle_type_to_ffi_type(runtime: &Runtime, value: usize) -> Result<FFIType, String> {
    let heap_object = HeapObject::from_tagged(value);
    let struct_id = BuiltInTypes::untag(heap_object.get_struct_id());
    let struct_info = runtime.get_struct_by_id(struct_id);

    if struct_info.is_none() {
        return Err(format!("Could not find struct with id {}", struct_id));
    }

    let struct_info = struct_info.unwrap();
    let name = struct_info.name.as_str().split_once("/").unwrap().1;
    match name {
        "Type.U8" => Ok(FFIType::U8),
        "Type.U16" => Ok(FFIType::U16),
        "Type.U32" => Ok(FFIType::U32),
        "Type.U64" => Ok(FFIType::U64),
        "Type.I32" => Ok(FFIType::I32),
        "Type.F32" => Ok(FFIType::F32),
        "Type.Pointer" => Ok(FFIType::Pointer),
        "Type.MutablePointer" => Ok(FFIType::MutablePointer),
        "Type.String" => Ok(FFIType::String),
        "Type.Void" => Ok(FFIType::Void),
        "Type.Structure" => {
            let types = heap_object.get_field(0);
            let types = persistent_vector_to_vec(types);
            let fields: Result<Vec<FFIType>, String> = types
                .iter()
                .map(|t| map_beagle_type_to_ffi_type(runtime, *t))
                .collect();
            Ok(FFIType::Structure(fields?))
        }
        _ => Err(format!("Unknown type: {}", name)),
    }
}

fn persistent_vector_to_vec(vector: usize) -> Vec<usize> {
    use crate::collections::{GcHandle, PersistentVec};
    let vec_handle = GcHandle::from_tagged(vector);
    let count = PersistentVec::count(vec_handle);
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        result.push(PersistentVec::get(vec_handle, i));
    }
    result
}

/// Get a function from a dynamically loaded library and register it for FFI calls.
/// No longer uses libffi - just stores the function pointer and type information.
pub extern "C" fn get_function(
    stack_pointer: usize,
    frame_pointer: usize,
    library_struct: usize,
    function_name: usize,
    types: usize,
    return_type: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let library = runtime.get_library(library_struct);
    let function_name = runtime.get_string_literal(function_name);

    // Check if we already have this function registered
    if let Some(ffi_info_id) = runtime.find_ffi_info_by_name(&function_name) {
        let ffi_info_id = BuiltInTypes::Int.tag(ffi_info_id as isize) as usize;
        return unsafe { call_fn_1(runtime, "beagle.ffi/__create_ffi_function", ffi_info_id) };
    }

    // Get the function pointer from the library
    let func_ptr = unsafe {
        library
            .get::<fn()>(function_name.as_bytes())
            .unwrap_or_else(|e| panic!("Failed to get symbol '{}': {}", function_name, e))
    };
    let code_ptr = unsafe { func_ptr.try_as_raw_ptr().unwrap() };

    // Parse argument types
    let types: Vec<usize> = persistent_vector_to_vec(types);

    let beagle_ffi_types: Result<Vec<FFIType>, String> = types
        .iter()
        .map(|t| map_beagle_type_to_ffi_type(runtime, *t))
        .collect();
    let beagle_ffi_types = match beagle_ffi_types {
        Ok(types) => types,
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "FFIError", e);
        },
    };
    let number_of_arguments = beagle_ffi_types.len();

    // Parse return type
    let ffi_return_type = match map_beagle_type_to_ffi_type(runtime, return_type) {
        Ok(t) => t,
        Err(e) => unsafe {
            throw_runtime_error(stack_pointer, "FFIError", e);
        },
    };

    // Register the function info (no Cif needed anymore)
    let ffi_info_id = runtime.add_ffi_function_info(FFIInfo {
        name: function_name.to_string(),
        function: RawPtr::new(code_ptr as *const u8),
        number_of_arguments,
        argument_types: beagle_ffi_types,
        return_type: ffi_return_type,
    });
    runtime.add_ffi_info_by_name(function_name, ffi_info_id);
    let ffi_info_id = BuiltInTypes::Int.tag(ffi_info_id as isize) as usize;

    unsafe { call_fn_1(runtime, "beagle.ffi/__create_ffi_function", ffi_info_id) }
}

/// Marshal a Beagle value to a native u64 for FFI call.
/// This function converts tagged Beagle values to raw C-compatible values.
unsafe fn marshal_ffi_argument(
    runtime: &mut Runtime,
    stack_pointer: usize,
    argument: usize,
    ffi_type: &FFIType,
) -> u64 {
    let kind = BuiltInTypes::get_kind(argument);
    match kind {
        BuiltInTypes::Null => {
            // Null pointer
            0u64
        }
        BuiltInTypes::String => {
            // String literal - convert to C string
            let string = runtime.get_string_literal(argument);
            let c_string = runtime.memory.write_c_string(string);
            c_string as u64
        }
        BuiltInTypes::Int => match ffi_type {
            FFIType::U8 | FFIType::U16 | FFIType::U32 | FFIType::U64 | FFIType::I32 => {
                BuiltInTypes::untag(argument) as u64
            }
            FFIType::F32 => {
                // Convert integer to f32 and get its bit representation
                let int_val = BuiltInTypes::untag(argument) as i64;
                let f32_val = int_val as f32;
                f32_val.to_bits() as u64
            }
            FFIType::Pointer => {
                if BuiltInTypes::untag(argument) == 0 {
                    0u64
                } else {
                    let heap_object = HeapObject::from_tagged(argument);
                    let buffer = BuiltInTypes::untag(heap_object.get_field(0));
                    buffer as u64
                }
            }
            _ => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "FFIError",
                    format!("Expected integer for FFI type {:?}", ffi_type),
                );
            },
        },
        BuiltInTypes::HeapObject => match ffi_type {
            FFIType::Pointer | FFIType::MutablePointer => {
                let heap_object = HeapObject::from_tagged(argument);
                let buffer = BuiltInTypes::untag(heap_object.get_field(0));
                buffer as u64
            }
            FFIType::String => {
                let string = runtime.get_string(stack_pointer, argument);
                let c_string = runtime.memory.write_c_string(string);
                c_string as u64
            }
            FFIType::Structure(_) => {
                let heap_object = HeapObject::from_tagged(argument);
                let buffer = BuiltInTypes::untag(heap_object.get_field(0));
                buffer as u64
            }
            _ => unsafe {
                throw_runtime_error(
                    stack_pointer,
                    "FFIError",
                    format!(
                        "Got HeapObject but expected matching FFI type, got {:?}",
                        ffi_type
                    ),
                );
            },
        },
        _ => unsafe {
            runtime.print(argument);
            throw_runtime_error(
                stack_pointer,
                "FFIError",
                format!("Unsupported FFI type: {:?}", kind),
            );
        },
    }
}

/// Call a native function with the given number of arguments and return type.
/// Uses transmute to cast the function pointer to the appropriate signature.
/// Dispatches based on return type to avoid undefined behavior when calling
/// void functions or functions returning narrower types.
#[inline(never)]
unsafe fn call_native_function(
    func_ptr: *const u8,
    num_args: usize,
    args: [u64; 6],
    arg_types: &[FFIType],
    return_type: &FFIType,
) -> u64 {
    // Check if any arguments are floats
    let has_float = arg_types.iter().any(|t| matches!(t, FFIType::F32));

    if has_float {
        // Use float-aware calling for functions with float arguments
        // SAFETY: Same requirements as this function - func_ptr must be valid
        return unsafe {
            call_native_function_with_floats(func_ptr, num_args, args, arg_types, return_type)
        };
    }

    // Macro for non-void return types (all integer arguments)
    macro_rules! call_with_args {
        ($ret_type:ty) => {
            match num_args {
                0 => {
                    let f: extern "C" fn() -> $ret_type = transmute(func_ptr);
                    f() as u64
                }
                1 => {
                    let f: extern "C" fn(u64) -> $ret_type = transmute(func_ptr);
                    f(args[0]) as u64
                }
                2 => {
                    let f: extern "C" fn(u64, u64) -> $ret_type = transmute(func_ptr);
                    f(args[0], args[1]) as u64
                }
                3 => {
                    let f: extern "C" fn(u64, u64, u64) -> $ret_type = transmute(func_ptr);
                    f(args[0], args[1], args[2]) as u64
                }
                4 => {
                    let f: extern "C" fn(u64, u64, u64, u64) -> $ret_type = transmute(func_ptr);
                    f(args[0], args[1], args[2], args[3]) as u64
                }
                5 => {
                    let f: extern "C" fn(u64, u64, u64, u64, u64) -> $ret_type =
                        transmute(func_ptr);
                    f(args[0], args[1], args[2], args[3], args[4]) as u64
                }
                6 => {
                    let f: extern "C" fn(u64, u64, u64, u64, u64, u64) -> $ret_type =
                        transmute(func_ptr);
                    f(args[0], args[1], args[2], args[3], args[4], args[5]) as u64
                }
                _ => panic!("Too many arguments for FFI call: {}", num_args),
            }
        };
    }

    // Macro for void return type (returns 0)
    macro_rules! call_void {
        () => {
            match num_args {
                0 => {
                    let f: extern "C" fn() = transmute(func_ptr);
                    f();
                    0
                }
                1 => {
                    let f: extern "C" fn(u64) = transmute(func_ptr);
                    f(args[0]);
                    0
                }
                2 => {
                    let f: extern "C" fn(u64, u64) = transmute(func_ptr);
                    f(args[0], args[1]);
                    0
                }
                3 => {
                    let f: extern "C" fn(u64, u64, u64) = transmute(func_ptr);
                    f(args[0], args[1], args[2]);
                    0
                }
                4 => {
                    let f: extern "C" fn(u64, u64, u64, u64) = transmute(func_ptr);
                    f(args[0], args[1], args[2], args[3]);
                    0
                }
                5 => {
                    let f: extern "C" fn(u64, u64, u64, u64, u64) = transmute(func_ptr);
                    f(args[0], args[1], args[2], args[3], args[4]);
                    0
                }
                6 => {
                    let f: extern "C" fn(u64, u64, u64, u64, u64, u64) = transmute(func_ptr);
                    f(args[0], args[1], args[2], args[3], args[4], args[5]);
                    0
                }
                _ => panic!("Too many arguments for FFI call: {}", num_args),
            }
        };
    }

    // Dispatch based on return type
    unsafe {
        match return_type {
            FFIType::Void => call_void!(),
            FFIType::U8 => call_with_args!(u8),
            FFIType::U16 => call_with_args!(u16),
            FFIType::U32 => call_with_args!(u32),
            FFIType::I32 => call_with_args!(i32),
            FFIType::U64 | FFIType::Pointer | FFIType::MutablePointer | FFIType::String => {
                call_with_args!(u64)
            }
            FFIType::F32 => call_with_args!(f32),
            FFIType::Structure(fields) => {
                // For small structs (≤16 bytes on ARM64), they're returned in x0/x1
                // We call with a struct return type and pack the result
                call_struct_return(func_ptr, num_args, args, fields)
            }
        }
    }
}

/// A 16-byte struct for receiving struct returns from C functions.
/// On ARM64, structs up to 16 bytes are returned in x0 and x1.
#[repr(C)]
#[derive(Clone, Copy)]
struct StructReturn16 {
    low: u64,
    high: u64,
}

/// Call a function that returns a struct (up to 16 bytes).
/// Returns the struct packed as two u64 values in a single u128.
#[inline(never)]
unsafe fn call_struct_return(
    func_ptr: *const u8,
    num_args: usize,
    args: [u64; 6],
    _fields: &[FFIType],
) -> u64 {
    unsafe {
        // Call the function with struct return type
        let result: StructReturn16 = match num_args {
            0 => {
                let f: extern "C" fn() -> StructReturn16 = transmute(func_ptr);
                f()
            }
            1 => {
                let f: extern "C" fn(u64) -> StructReturn16 = transmute(func_ptr);
                f(args[0])
            }
            2 => {
                let f: extern "C" fn(u64, u64) -> StructReturn16 = transmute(func_ptr);
                f(args[0], args[1])
            }
            3 => {
                let f: extern "C" fn(u64, u64, u64) -> StructReturn16 = transmute(func_ptr);
                f(args[0], args[1], args[2])
            }
            4 => {
                let f: extern "C" fn(u64, u64, u64, u64) -> StructReturn16 = transmute(func_ptr);
                f(args[0], args[1], args[2], args[3])
            }
            5 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64) -> StructReturn16 =
                    transmute(func_ptr);
                f(args[0], args[1], args[2], args[3], args[4])
            }
            6 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64, u64) -> StructReturn16 =
                    transmute(func_ptr);
                f(args[0], args[1], args[2], args[3], args[4], args[5])
            }
            _ => panic!("Too many arguments for FFI call: {}", num_args),
        };

        // Store the struct in thread-local storage so unmarshal can access both parts
        STRUCT_RETURN_HIGH.with(|cell| cell.set(result.high));
        result.low
    }
}

/// Call a native function that has float arguments.
/// This creates function signatures with f32 in the appropriate positions
/// so that the Rust compiler places floats in FP registers per the C ABI.
#[inline(never)]
unsafe fn call_native_function_with_floats(
    func_ptr: *const u8,
    num_args: usize,
    args: [u64; 6],
    arg_types: &[FFIType],
    return_type: &FFIType,
) -> u64 {
    // Helper to convert u64 (containing f32 bits) to f32
    fn to_f32(v: u64) -> f32 {
        f32::from_bits(v as u32)
    }

    // Build a signature pattern: which positions are floats?
    // We encode this as a bitmask for efficient dispatch
    let mut float_mask: u8 = 0;
    for (i, t) in arg_types.iter().enumerate() {
        if matches!(t, FFIType::F32) {
            float_mask |= 1 << i;
        }
    }

    // For void return, we need separate handling
    let is_void = matches!(return_type, FFIType::Void);

    // Dispatch based on number of args and float positions
    // We handle common patterns explicitly
    unsafe {
        match (num_args, float_mask) {
            // 1 arg patterns
            (1, 0b0001) => {
                // f32
                let f: extern "C" fn(f32) -> u64 = transmute(func_ptr);
                if is_void {
                    f(to_f32(args[0]));
                    0
                } else {
                    f(to_f32(args[0]))
                }
            }

            // 2 arg patterns
            (2, 0b0001) => {
                // f32, u64
                let f: extern "C" fn(f32, u64) -> u64 = transmute(func_ptr);
                if is_void {
                    f(to_f32(args[0]), args[1]);
                    0
                } else {
                    f(to_f32(args[0]), args[1])
                }
            }
            (2, 0b0010) => {
                // u64, f32
                let f: extern "C" fn(u64, f32) -> u64 = transmute(func_ptr);
                if is_void {
                    f(args[0], to_f32(args[1]));
                    0
                } else {
                    f(args[0], to_f32(args[1]))
                }
            }
            (2, 0b0011) => {
                // f32, f32
                let f: extern "C" fn(f32, f32) -> u64 = transmute(func_ptr);
                if is_void {
                    f(to_f32(args[0]), to_f32(args[1]));
                    0
                } else {
                    f(to_f32(args[0]), to_f32(args[1]))
                }
            }

            // 3 arg patterns
            (3, 0b0001) => {
                let f: extern "C" fn(f32, u64, u64) -> u64 = transmute(func_ptr);
                if is_void {
                    f(to_f32(args[0]), args[1], args[2]);
                    0
                } else {
                    f(to_f32(args[0]), args[1], args[2])
                }
            }
            (3, 0b0010) => {
                let f: extern "C" fn(u64, f32, u64) -> u64 = transmute(func_ptr);
                if is_void {
                    f(args[0], to_f32(args[1]), args[2]);
                    0
                } else {
                    f(args[0], to_f32(args[1]), args[2])
                }
            }
            (3, 0b0100) => {
                let f: extern "C" fn(u64, u64, f32) -> u64 = transmute(func_ptr);
                if is_void {
                    f(args[0], args[1], to_f32(args[2]));
                    0
                } else {
                    f(args[0], args[1], to_f32(args[2]))
                }
            }

            // 4 arg patterns (common for raylib: int, int, float, color)
            (4, 0b0001) => {
                let f: extern "C" fn(f32, u64, u64, u64) -> u64 = transmute(func_ptr);
                if is_void {
                    f(to_f32(args[0]), args[1], args[2], args[3]);
                    0
                } else {
                    f(to_f32(args[0]), args[1], args[2], args[3])
                }
            }
            (4, 0b0010) => {
                let f: extern "C" fn(u64, f32, u64, u64) -> u64 = transmute(func_ptr);
                if is_void {
                    f(args[0], to_f32(args[1]), args[2], args[3]);
                    0
                } else {
                    f(args[0], to_f32(args[1]), args[2], args[3])
                }
            }
            (4, 0b0100) => {
                // DrawCircle pattern: int, int, float, color
                let f: extern "C" fn(u64, u64, f32, u64) -> u64 = transmute(func_ptr);
                if is_void {
                    f(args[0], args[1], to_f32(args[2]), args[3]);
                    0
                } else {
                    f(args[0], args[1], to_f32(args[2]), args[3])
                }
            }
            (4, 0b1000) => {
                let f: extern "C" fn(u64, u64, u64, f32) -> u64 = transmute(func_ptr);
                if is_void {
                    f(args[0], args[1], args[2], to_f32(args[3]));
                    0
                } else {
                    f(args[0], args[1], args[2], to_f32(args[3]))
                }
            }
            (4, 0b0011) => {
                let f: extern "C" fn(f32, f32, u64, u64) -> u64 = transmute(func_ptr);
                if is_void {
                    f(to_f32(args[0]), to_f32(args[1]), args[2], args[3]);
                    0
                } else {
                    f(to_f32(args[0]), to_f32(args[1]), args[2], args[3])
                }
            }
            (4, 0b0110) => {
                let f: extern "C" fn(u64, f32, f32, u64) -> u64 = transmute(func_ptr);
                if is_void {
                    f(args[0], to_f32(args[1]), to_f32(args[2]), args[3]);
                    0
                } else {
                    f(args[0], to_f32(args[1]), to_f32(args[2]), args[3])
                }
            }
            (4, 0b1100) => {
                let f: extern "C" fn(u64, u64, f32, f32) -> u64 = transmute(func_ptr);
                if is_void {
                    f(args[0], args[1], to_f32(args[2]), to_f32(args[3]));
                    0
                } else {
                    f(args[0], args[1], to_f32(args[2]), to_f32(args[3]))
                }
            }

            // 5 arg patterns
            (5, 0b00100) => {
                let f: extern "C" fn(u64, u64, f32, u64, u64) -> u64 = transmute(func_ptr);
                if is_void {
                    f(args[0], args[1], to_f32(args[2]), args[3], args[4]);
                    0
                } else {
                    f(args[0], args[1], to_f32(args[2]), args[3], args[4])
                }
            }

            // 6 arg patterns
            (6, 0b000100) => {
                let f: extern "C" fn(u64, u64, f32, u64, u64, u64) -> u64 = transmute(func_ptr);
                if is_void {
                    f(args[0], args[1], to_f32(args[2]), args[3], args[4], args[5]);
                    0
                } else {
                    f(args[0], args[1], to_f32(args[2]), args[3], args[4], args[5])
                }
            }

            _ => {
                panic!(
                    "Unsupported float argument pattern: {} args, float_mask=0b{:06b}. \
                     Add this pattern to call_native_function_with_floats.",
                    num_args, float_mask
                );
            }
        }
    }
}

/// Convert a native return value to a Beagle value.
unsafe fn unmarshal_ffi_return(
    runtime: &mut Runtime,
    stack_pointer: usize,
    result: u64,
    return_type: &FFIType,
) -> usize {
    unsafe {
        match return_type {
            FFIType::Void => BuiltInTypes::null_value() as usize,
            FFIType::U8 | FFIType::U16 | FFIType::U32 | FFIType::U64 => {
                BuiltInTypes::Int.tag(result as isize) as usize
            }
            FFIType::I32 => {
                // Sign-extend I32 to isize properly
                let signed_result = result as i32 as isize;
                BuiltInTypes::Int.tag(signed_result) as usize
            }
            FFIType::F32 => {
                // Convert f32 bits to integer for Beagle
                // The result is already the f32 bit pattern in the low 32 bits
                let f32_val = f32::from_bits(result as u32);
                BuiltInTypes::Int.tag(f32_val as i32 as isize) as usize
            }
            FFIType::Pointer | FFIType::MutablePointer => {
                let pointer_value = BuiltInTypes::Int.tag(result as isize) as usize;
                call_fn_1(runtime, "beagle.ffi/__make_pointer_struct", pointer_value)
            }
            FFIType::String => {
                if result == 0 {
                    return BuiltInTypes::null_value() as usize;
                }
                let c_string = CStr::from_ptr(result as *const i8);
                let string = c_string.to_str().unwrap();
                runtime
                    .allocate_string(stack_pointer, string.to_string())
                    .unwrap()
                    .into()
            }
            FFIType::Structure(fields) => {
                // Get the high part from thread-local storage
                let high = STRUCT_RETURN_HIGH.with(|cell| cell.get());

                // Create a Beagle array with the struct fields
                // For a 16-byte struct like Shader {id: u32, locs: *int}:
                // - low contains the first 8 bytes (id in lower 4 bytes)
                // - high contains the next 8 bytes (locs pointer)
                let mut offset = 0u64;
                let low = result;

                // Build an array of field values based on field types
                let mut field_values = Vec::with_capacity(fields.len());
                for field in fields {
                    let (value, size) = match field {
                        FFIType::U32 | FFIType::I32 => {
                            let v = if offset < 8 {
                                ((low >> (offset * 8)) & 0xFFFFFFFF) as u32
                            } else {
                                ((high >> ((offset - 8) * 8)) & 0xFFFFFFFF) as u32
                            };
                            (BuiltInTypes::Int.tag(v as isize) as usize, 4)
                        }
                        FFIType::Pointer | FFIType::MutablePointer | FFIType::U64 => {
                            let v = if offset < 8 { low } else { high };
                            // Align offset to 8 bytes for pointers
                            offset = (offset + 7) & !7;
                            (BuiltInTypes::Int.tag(v as isize) as usize, 8)
                        }
                        _ => {
                            // For other types, just use the raw value
                            let v = if offset < 8 { low } else { high };
                            (BuiltInTypes::Int.tag(v as isize) as usize, 8)
                        }
                    };
                    field_values.push(value);
                    offset += size as u64;
                }

                // Create a simple array/tuple to hold the struct fields
                // The caller can access fields by index
                call_fn_2(
                    runtime,
                    "beagle.ffi/__make_struct_return",
                    BuiltInTypes::Int.tag(low as isize) as usize,
                    BuiltInTypes::Int.tag(high as isize) as usize,
                )
            }
        }
    }
}

/// Call a foreign function through FFI.
/// This implementation uses direct function pointer calls via transmute,
/// eliminating the need for libffi.
pub unsafe extern "C" fn call_ffi_info(
    stack_pointer: usize,
    _frame_pointer: usize,
    ffi_info_id: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
    a6: usize,
) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        let ffi_info_id = BuiltInTypes::untag(ffi_info_id);

        // Extract function info
        let (func_ptr, number_of_arguments, argument_types, return_type) = {
            let ffi_info = runtime.get_ffi_info(ffi_info_id);
            (
                ffi_info.function.ptr,
                ffi_info.number_of_arguments,
                ffi_info.argument_types.clone(),
                ffi_info.return_type.clone(),
            )
        };

        let arguments = [a1, a2, a3, a4, a5, a6];
        let args = &arguments[..number_of_arguments];

        // Marshal arguments to native u64 values
        let mut native_args: [u64; 6] = [0; 6];
        for (i, (argument, ffi_type)) in args.iter().zip(argument_types.iter()).enumerate() {
            native_args[i] = marshal_ffi_argument(runtime, stack_pointer, *argument, ffi_type);
        }

        // Call the native function directly using transmute
        let result = call_native_function(
            func_ptr,
            number_of_arguments,
            native_args,
            &argument_types,
            &return_type,
        );

        // Unmarshal the return value
        let return_value = unmarshal_ffi_return(runtime, stack_pointer, result, &return_type);

        runtime.memory.clear_native_arguments();
        return_value
    }
}

pub unsafe extern "C" fn copy_object(
    stack_pointer: usize,
    frame_pointer: usize,
    object_pointer: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let object_pointer_id = runtime.register_temporary_root(object_pointer);
    let to_pointer = {
        let object = HeapObject::from_tagged(object_pointer);
        let header = object.get_header();
        let size = header.size as usize;
        let kind = BuiltInTypes::get_kind(object_pointer);
        runtime.allocate(size, stack_pointer, kind).unwrap()
    };
    let object_pointer = runtime.unregister_temporary_root(object_pointer_id);
    let mut to_object = HeapObject::from_tagged(to_pointer);
    let object = HeapObject::from_tagged(object_pointer);
    let result = runtime.copy_object(object, &mut to_object);
    if let Err(error) = result {
        let stack_pointer = get_current_stack_pointer();
        println!("Error: {:?}", error);
        unsafe { throw_error(stack_pointer, frame_pointer) };
    } else {
        result.unwrap()
    }
}

pub unsafe extern "C" fn copy_from_to_object(from: usize, to: usize) -> usize {
    let runtime = get_runtime().get_mut();
    if from == BuiltInTypes::null_value() as usize {
        return to;
    }
    let from = HeapObject::from_tagged(from);
    let mut to = HeapObject::from_tagged(to);
    runtime.copy_object_except_header(from, &mut to).unwrap();
    to.tagged_pointer()
}

pub unsafe extern "C" fn copy_array_range(
    from: usize,
    to: usize,
    start: usize,
    count: usize,
) -> usize {
    let from_obj = HeapObject::from_tagged(from);
    let mut to_obj = HeapObject::from_tagged(to);

    let from_fields = from_obj.get_fields();
    let to_fields = to_obj.get_fields_mut();

    let start_idx = BuiltInTypes::untag(start);
    let count_val = BuiltInTypes::untag(count);

    // Use ptr::copy_nonoverlapping for fast memcpy
    unsafe {
        std::ptr::copy_nonoverlapping(
            from_fields.as_ptr().add(start_idx),
            to_fields.as_mut_ptr().add(start_idx),
            count_val,
        );
    }

    to
}

unsafe extern "C" fn ffi_allocate(size: usize) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        // TODO: I intentionally don't want to manage this memory on the heap
        // I probably need a better answer than this
        // but for now we are just going to leak memory
        let size = BuiltInTypes::untag(size);

        let mut buffer: Vec<u8> = vec![0; size];
        let buffer_ptr: *mut c_void = buffer.as_mut_ptr() as *mut c_void;
        std::mem::forget(buffer);

        let buffer = BuiltInTypes::Int.tag(buffer_ptr as isize) as usize;
        let size = BuiltInTypes::Int.tag(size as isize) as usize;
        call_fn_2(runtime, "beagle.ffi/__make_buffer_struct", buffer, size)
    }
}

unsafe extern "C" fn ffi_deallocate(buffer: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let size = BuiltInTypes::untag(buffer_object.get_field(1));
        let _buffer = Vec::from_raw_parts(buffer, size, size);
        BuiltInTypes::null_value() as usize
    }
}

unsafe extern "C" fn ffi_get_u32(buffer: usize, offset: usize) -> usize {
    unsafe {
        // TODO: Make type safe
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset) as *const u32);
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

unsafe extern "C" fn ffi_set_u8(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value);
        assert!(value <= u8::MAX as usize);
        let value = value as u8;
        *(buffer.add(offset)) = value;
        BuiltInTypes::null_value() as usize
    }
}

unsafe extern "C" fn ffi_get_u8(buffer: usize, offset: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset));
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

unsafe extern "C" fn ffi_set_i32(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value) as i32;
        *(buffer.add(offset) as *mut i32) = value;
        BuiltInTypes::null_value() as usize
    }
}

unsafe extern "C" fn ffi_set_i16(buffer: usize, offset: usize, value: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let value = BuiltInTypes::untag(value) as i16;
        *(buffer.add(offset) as *mut i16) = value;
        BuiltInTypes::null_value() as usize
    }
}

unsafe extern "C" fn ffi_get_i32(buffer: usize, offset: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let value = *(buffer.add(offset) as *const i32);
        BuiltInTypes::Int.tag(value as isize) as usize
    }
}

unsafe extern "C" fn ffi_get_string(
    stack_pointer: usize,
    _frame_pointer: usize, // Frame pointer for GC stack walking
    buffer: usize,
    offset: usize,
    len: usize,
) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);
        let slice = std::slice::from_raw_parts(buffer.add(offset), len);
        let string = std::str::from_utf8(slice).unwrap();
        runtime
            .allocate_string(stack_pointer, string.to_string())
            .unwrap()
            .into()
    }
}

unsafe extern "C" fn ffi_create_array(
    stack_pointer: usize,
    _frame_pointer: usize, // Frame pointer for GC stack walking
    ffi_type: usize,
    array: usize,
) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        let ffi_type = match map_beagle_type_to_ffi_type(runtime, ffi_type) {
            Ok(t) => t,
            Err(e) => {
                throw_runtime_error(stack_pointer, "FFIError", e);
            }
        };
        let array = HeapObject::from_tagged(array);
        let fields = array.get_fields();
        let size = fields.len();
        let mut buffer: Vec<*mut i8> = Vec::with_capacity(size);
        for field in fields {
            match ffi_type {
                FFIType::U8 => todo!(),
                FFIType::U16 => todo!(),
                FFIType::U32 => todo!(),
                FFIType::U64 => todo!(),
                FFIType::I32 => todo!(),
                FFIType::F32 => todo!(),
                FFIType::Pointer => {
                    todo!()
                }
                FFIType::MutablePointer => {
                    todo!()
                }
                FFIType::Structure(_) => {
                    todo!()
                }
                FFIType::String => {
                    let string = runtime.get_string(stack_pointer, *field);
                    let string_pointer = runtime.memory.write_c_string(string);
                    buffer.push(string_pointer);
                }
                FFIType::Void => {
                    throw_runtime_error(
                        stack_pointer,
                        "FFIError",
                        "Cannot create array of void type".to_string(),
                    );
                }
            }
        }
        // null terminate array
        buffer.push(std::ptr::null_mut());

        // For now we are intentionally leaking memory

        let buffer_ptr: *mut c_void = buffer.as_mut_ptr() as *mut c_void;
        std::mem::forget(buffer);
        let buffer = BuiltInTypes::Int.tag(buffer_ptr as isize) as usize;
        call_fn_1(runtime, "beagle.ffi/__make_pointer_struct", buffer)
    }
}

// Copy bytes between FFI buffers with offsets
// ffi_copy_bytes(src, src_off, dst, dst_off, len) -> null
unsafe extern "C" fn ffi_copy_bytes(
    src: usize,
    src_off: usize,
    dst: usize,
    dst_off: usize,
    len: usize,
) -> usize {
    unsafe {
        let src_object = HeapObject::from_tagged(src);
        let src_ptr = BuiltInTypes::untag(src_object.get_field(0)) as *const u8;
        let src_off = BuiltInTypes::untag(src_off);

        let dst_object = HeapObject::from_tagged(dst);
        let dst_ptr = BuiltInTypes::untag(dst_object.get_field(0)) as *mut u8;
        let dst_off = BuiltInTypes::untag(dst_off);

        let len = BuiltInTypes::untag(len);

        std::ptr::copy_nonoverlapping(src_ptr.add(src_off), dst_ptr.add(dst_off), len);

        BuiltInTypes::null_value() as usize
    }
}

// Reallocate an FFI buffer to a new size
// ffi_realloc(buffer, new_size) -> new_buffer
unsafe extern "C" fn ffi_realloc(buffer: usize, new_size: usize) -> usize {
    unsafe {
        let runtime = get_runtime().get_mut();
        let buffer_object = HeapObject::from_tagged(buffer);
        let old_ptr = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let old_size = BuiltInTypes::untag(buffer_object.get_field(1));
        let new_size_val = BuiltInTypes::untag(new_size);

        // Create new buffer
        let mut new_buffer: Vec<u8> = vec![0; new_size_val];
        let new_ptr: *mut u8 = new_buffer.as_mut_ptr();

        // Copy old data
        let copy_len = std::cmp::min(old_size, new_size_val);
        std::ptr::copy_nonoverlapping(old_ptr, new_ptr, copy_len);

        // Free old buffer
        let _old_buffer = Vec::from_raw_parts(old_ptr, old_size, old_size);

        // Forget new buffer (we're taking ownership)
        std::mem::forget(new_buffer);

        let new_ptr_tagged = BuiltInTypes::Int.tag(new_ptr as isize) as usize;
        let new_size_tagged = BuiltInTypes::Int.tag(new_size_val as isize) as usize;
        call_fn_2(
            runtime,
            "beagle.ffi/__make_buffer_struct",
            new_ptr_tagged,
            new_size_tagged,
        )
    }
}

// Get the size of an FFI buffer
// ffi_buffer_size(buffer) -> size
unsafe extern "C" fn ffi_buffer_size(buffer: usize) -> usize {
    let buffer_object = HeapObject::from_tagged(buffer);
    let size = BuiltInTypes::untag(buffer_object.get_field(1));
    BuiltInTypes::Int.tag(size as isize) as usize
}

// Write from buffer at offset to a file descriptor
// ffi_write_buffer_offset(fd, buffer, offset, len) -> bytes_written
unsafe extern "C" fn ffi_write_buffer_offset(
    fd: usize,
    buffer: usize,
    offset: usize,
    len: usize,
) -> usize {
    unsafe {
        let fd = BuiltInTypes::untag(fd) as i32;
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer_ptr = BuiltInTypes::untag(buffer_object.get_field(0)) as *const u8;
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);

        let result = libc::write(fd, buffer_ptr.add(offset) as *const libc::c_void, len);

        BuiltInTypes::Int.tag(result as isize) as usize
    }
}

// Translate bytes in buffer using a 256-byte lookup table
// ffi_translate_bytes(buffer, offset, len, table) -> null
unsafe extern "C" fn ffi_translate_bytes(
    buffer: usize,
    offset: usize,
    len: usize,
    table: usize,
) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer_ptr = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);

        let table_object = HeapObject::from_tagged(table);
        let table_ptr = BuiltInTypes::untag(table_object.get_field(0)) as *const u8;

        for i in 0..len {
            let byte = *buffer_ptr.add(offset + i);
            let translated = *table_ptr.add(byte as usize);
            *buffer_ptr.add(offset + i) = translated;
        }

        BuiltInTypes::null_value() as usize
    }
}

// Reverse bytes in buffer in place
// ffi_reverse_bytes(buffer, offset, len) -> null
unsafe extern "C" fn ffi_reverse_bytes(buffer: usize, offset: usize, len: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer_ptr = BuiltInTypes::untag(buffer_object.get_field(0)) as *mut u8;
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);

        let mut left = 0;
        let mut right = len.saturating_sub(1);

        while left < right {
            let tmp = *buffer_ptr.add(offset + left);
            *buffer_ptr.add(offset + left) = *buffer_ptr.add(offset + right);
            *buffer_ptr.add(offset + right) = tmp;
            left += 1;
            right -= 1;
        }

        BuiltInTypes::null_value() as usize
    }
}

// Find first occurrence of a byte in buffer (like memchr)
// ffi_find_byte(buffer, offset, len, byte) -> index or -1
unsafe extern "C" fn ffi_find_byte(buffer: usize, offset: usize, len: usize, byte: usize) -> usize {
    unsafe {
        let buffer_object = HeapObject::from_tagged(buffer);
        let buffer_ptr = BuiltInTypes::untag(buffer_object.get_field(0)) as *const u8;
        let offset = BuiltInTypes::untag(offset);
        let len = BuiltInTypes::untag(len);
        let byte = BuiltInTypes::untag(byte) as u8;

        let slice = std::slice::from_raw_parts(buffer_ptr.add(offset), len);
        match slice.iter().position(|&b| b == byte) {
            Some(pos) => BuiltInTypes::Int.tag((offset + pos) as isize) as usize,
            None => BuiltInTypes::Int.tag(-1) as usize,
        }
    }
}

// Copy bytes from src to dst, skipping instances of skip_byte
// Returns number of bytes written to dst
// ffi_copy_bytes_filter(src, src_off, dst, dst_off, len, skip_byte) -> bytes_written
unsafe extern "C" fn ffi_copy_bytes_filter(
    src: usize,
    src_off: usize,
    dst: usize,
    dst_off: usize,
    len: usize,
    skip_byte: usize,
) -> usize {
    unsafe {
        let src_object = HeapObject::from_tagged(src);
        let src_ptr = BuiltInTypes::untag(src_object.get_field(0)) as *const u8;
        let src_off = BuiltInTypes::untag(src_off);

        let dst_object = HeapObject::from_tagged(dst);
        let dst_ptr = BuiltInTypes::untag(dst_object.get_field(0)) as *mut u8;
        let dst_off = BuiltInTypes::untag(dst_off);

        let len = BuiltInTypes::untag(len);
        let skip_byte = BuiltInTypes::untag(skip_byte) as u8;

        let mut written = 0;
        for i in 0..len {
            let byte = *src_ptr.add(src_off + i);
            if byte != skip_byte {
                *dst_ptr.add(dst_off + written) = byte;
                written += 1;
            }
        }

        BuiltInTypes::Int.tag(written as isize) as usize
    }
}

extern "C" fn placeholder() -> usize {
    BuiltInTypes::null_value() as usize
}

extern "C" fn wait_for_input(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let runtime = get_runtime().get_mut();
    let string = runtime.allocate_string(stack_pointer, input);
    string.unwrap().into()
}

// Get the ASCII code of the first character of a string
extern "C" fn char_code(stack_pointer: usize, frame_pointer: usize, string: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let string = runtime.get_string(stack_pointer, string);
    if let Some(ch) = string.chars().next() {
        BuiltInTypes::Int.tag(ch as isize) as usize
    } else {
        // Empty string returns -1
        BuiltInTypes::Int.tag(-1) as usize
    }
}

// Create a single-character string from an ASCII code
extern "C" fn char_from_code(stack_pointer: usize, frame_pointer: usize, code: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let code = BuiltInTypes::untag(code) as u8;
    let ch = code as char;
    let runtime = get_runtime().get_mut();
    runtime
        .allocate_string(stack_pointer, ch.to_string())
        .unwrap()
        .into()
}

// Read a line from stdin, stripping the trailing newline
// Returns null if EOF is reached
extern "C" fn read_line(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let mut input = String::new();
    match std::io::stdin().read_line(&mut input) {
        Ok(0) => {
            // EOF reached
            BuiltInTypes::null_value() as usize
        }
        Ok(_) => {
            // Remove trailing newline if present
            if input.ends_with('\n') {
                input.pop();
                if input.ends_with('\r') {
                    input.pop();
                }
            }
            let runtime = get_runtime().get_mut();
            let string = runtime.allocate_string(stack_pointer, input);
            string.unwrap().into()
        }
        Err(_) => {
            // Error - return null
            BuiltInTypes::null_value() as usize
        }
    }
}

extern "C" fn read_full_file(
    stack_pointer: usize,
    frame_pointer: usize,
    file_name: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let file_name = runtime.get_string(stack_pointer, file_name);
    let file = std::fs::read_to_string(file_name).unwrap();
    let string = runtime.allocate_string(stack_pointer, file);
    string.unwrap().into()
}

extern "C" fn eval(stack_pointer: usize, frame_pointer: usize, code: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let code = match BuiltInTypes::get_kind(code) {
        BuiltInTypes::String => runtime.get_string_literal(code),
        BuiltInTypes::HeapObject => {
            let code = HeapObject::from_tagged(code);
            if code.get_header().type_id != 2 {
                unsafe {
                    throw_runtime_error(
                        stack_pointer,
                        "TypeError",
                        format!(
                            "Expected string, got heap object with type_id {}",
                            code.get_header().type_id
                        ),
                    );
                }
            }
            let bytes = code.get_string_bytes();
            let code = std::str::from_utf8(bytes).unwrap();
            code.to_string()
        }
        _ => unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!("Expected string, got {:?}", BuiltInTypes::get_kind(code)),
            );
        },
    };
    let result = match runtime.compile_string(&code) {
        Ok(result) => result,
        Err(e) => unsafe {
            throw_runtime_error(
                stack_pointer,
                "CompileError",
                format!("Compilation failed: {}", e),
            );
        },
    };
    mem::forget(code);
    if result == 0 {
        return BuiltInTypes::null_value() as usize;
    }
    let f: fn() -> usize = unsafe { transmute(result) };
    f()
}

extern "C" fn sleep(time: usize) -> usize {
    let time = BuiltInTypes::untag(time);
    std::thread::sleep(std::time::Duration::from_millis(time as u64));
    BuiltInTypes::null_value() as usize
}

/// High-precision timer returning nanoseconds since an arbitrary epoch
/// Useful for benchmarking - subtract two values to get elapsed time
extern "C" fn time_now() -> usize {
    use std::time::Instant;
    // Use a static instant as the epoch to avoid overflow
    // This gives us relative timing which is what we need for benchmarks
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    let elapsed = start.elapsed().as_nanos() as isize;
    BuiltInTypes::Int.tag(elapsed) as usize
}

pub extern "C" fn register_extension(
    struct_name: usize,
    protocol_name: usize,
    method_name: usize,
    f: usize,
) -> usize {
    let runtime = get_runtime().get_mut();
    // TOOD: For right now I'm going to store these at the runtime level
    // But I think I actually want to store this information in the
    // protocol struct instead of out of band
    let struct_name = runtime.get_string_literal(struct_name);
    let protocol_name = runtime.get_string_literal(protocol_name);
    let method_name = runtime.get_string_literal(method_name);

    let struct_name = runtime.resolve(struct_name);
    // Don't resolve protocol names that contain type parameters (e.g., Handler<ns/Type>)
    // as they are already fully qualified
    let protocol_name = if protocol_name.contains('<') {
        protocol_name
    } else {
        runtime.resolve(protocol_name)
    };

    runtime.add_protocol_info(&protocol_name, &struct_name, &method_name, f);

    runtime.compile_protocol_method(&protocol_name, &method_name);

    BuiltInTypes::null_value() as usize
}

extern "C" fn hash(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let _ = stack_pointer; // Used by save_gc_context!
    print_call_builtin(get_runtime().get(), "hash");
    let runtime = get_runtime().get();
    let raw_hash = runtime.hash_value(value);
    BuiltInTypes::Int.tag(raw_hash as isize) as usize
}

pub extern "C" fn is_keyword(value: usize) -> usize {
    let tag = BuiltInTypes::get_kind(value);
    if tag != BuiltInTypes::HeapObject {
        return BuiltInTypes::construct_boolean(false) as usize;
    }
    let heap_object = HeapObject::from_tagged(value);
    let is_kw = heap_object.get_header().type_id == 3;
    BuiltInTypes::construct_boolean(is_kw) as usize
}

pub extern "C" fn keyword_to_string(
    stack_pointer: usize,
    frame_pointer: usize,
    keyword: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Check if it's a HeapObject before calling from_tagged
    let tag = BuiltInTypes::get_kind(keyword);
    if tag != BuiltInTypes::HeapObject {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                "keyword->string expects a keyword".to_string(),
            );
        }
    }

    let heap_object = HeapObject::from_tagged(keyword);

    if heap_object.get_header().type_id != 3 {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                "keyword->string expects a keyword".to_string(),
            );
        }
    }

    let bytes = heap_object.get_keyword_bytes();
    let keyword_text = unsafe { std::str::from_utf8_unchecked(bytes) };

    runtime
        .allocate_string(stack_pointer, keyword_text.to_string())
        .unwrap()
        .into()
}

pub extern "C" fn string_to_keyword(
    stack_pointer: usize,
    frame_pointer: usize,
    string_value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();
    let keyword_text = runtime.get_string(stack_pointer, string_value);

    // Use intern_keyword to ensure same text = same pointer
    runtime.intern_keyword(stack_pointer, keyword_text).unwrap()
}

pub extern "C" fn load_keyword_constant_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    index: usize,
) -> usize {
    use crate::types::BuiltInTypes;

    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Check heap-based PersistentMap first (survives GC relocation)
    let kw_ns = runtime.keyword_namespace_id();
    if kw_ns != 0 {
        let heap_ptr = runtime.get_heap_binding(kw_ns, index);
        if heap_ptr != BuiltInTypes::null_value() as usize {
            return heap_ptr;
        }
    }

    // Allocate and register in heap-based map
    let keyword_text = runtime.keyword_constants[index].str.clone();
    runtime.intern_keyword(stack_pointer, keyword_text).unwrap()
}

extern "C" fn many_args(
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
    a6: usize,
    a7: usize,
    a8: usize,
    a9: usize,
    a10: usize,
    a11: usize,
) -> usize {
    let a1 = BuiltInTypes::untag(a1);
    let a2 = BuiltInTypes::untag(a2);
    let a3 = BuiltInTypes::untag(a3);
    let a4 = BuiltInTypes::untag(a4);
    let a5 = BuiltInTypes::untag(a5);
    let a6 = BuiltInTypes::untag(a6);
    let a7 = BuiltInTypes::untag(a7);
    let a8 = BuiltInTypes::untag(a8);
    let a9 = BuiltInTypes::untag(a9);
    let a10 = BuiltInTypes::untag(a10);
    let a11 = BuiltInTypes::untag(a11);
    let result = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11;
    BuiltInTypes::Int.tag(result as isize) as usize
}

extern "C" fn pop_count(stack_pointer: usize, frame_pointer: usize, value: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "pop_count");
    let tag = BuiltInTypes::get_kind(value);
    match tag {
        BuiltInTypes::Int => {
            let value = BuiltInTypes::untag(value);
            let count = value.count_ones();
            BuiltInTypes::Int.tag(count as isize) as usize
        }
        _ => unsafe {
            throw_runtime_error(
                stack_pointer,
                "TypeError",
                format!("Expected int, got {:?}", tag),
            );
        },
    }
}

// Exception handling builtins
pub unsafe extern "C" fn push_exception_handler_runtime(
    handler_address: usize,
    result_local: isize,
    link_register: usize,
    stack_pointer: usize,
    frame_pointer: usize,
) -> usize {
    print_call_builtin(get_runtime().get(), "push_exception_handler");
    let runtime = get_runtime().get_mut();

    // All values are passed as parameters since we can't reliably read them
    // inside this function (x30 gets clobbered by the call, SP/FP might be modified)
    let handler = crate::runtime::ExceptionHandler {
        handler_address,
        stack_pointer,
        frame_pointer,
        link_register,
        result_local,
    };

    runtime.push_exception_handler(handler);
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn pop_exception_handler_runtime() -> usize {
    print_call_builtin(get_runtime().get(), "pop_exception_handler");
    let runtime = get_runtime().get_mut();
    runtime.pop_exception_handler();
    BuiltInTypes::null_value() as usize
}

pub unsafe extern "C" fn throw_exception(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "throw_exception");

    // Create exception object
    let exception = {
        let runtime = get_runtime().get_mut();
        runtime.create_exception(stack_pointer, value).unwrap()
    };

    // Pop handlers until we find one
    if let Some(handler) = {
        let runtime = get_runtime().get_mut();
        runtime.pop_exception_handler()
    } {
        // Restore stack, frame, and link register pointers
        let handler_address = handler.handler_address;
        let new_sp = handler.stack_pointer;
        let new_fp = handler.frame_pointer;
        let new_lr = handler.link_register;
        let result_local_offset = handler.result_local;

        let result_ptr = (new_fp as isize).wrapping_add(result_local_offset) as *mut usize;
        unsafe {
            *result_ptr = exception;
        }

        // Jump to handler with restored SP, FP, and LR
        unsafe {
            cfg_if::cfg_if! {
                if #[cfg(target_arch = "x86_64")] {
                    // x86-64: restore RSP, RBP, and jump
                    // The return address is already on the stack at [RBP + 8]
                    // so we don't need to push it (unlike ARM64 which uses LR register)
                    let _ = new_lr; // unused on x86-64, return addr is on stack
                    asm!(
                        "mov rsp, {0}",
                        "mov rbp, {1}",
                        "jmp {2}",
                        in(reg) new_sp,
                        in(reg) new_fp,
                        in(reg) handler_address,
                        options(noreturn)
                    );
                } else {
                    // ARM64: restore SP, X29 (FP), X30 (LR), and branch
                    asm!(
                        "mov sp, {0}",
                        "mov x29, {1}",
                        "mov x30, {2}",
                        "br {3}",
                        in(reg) new_sp,
                        in(reg) new_fp,
                        in(reg) new_lr,
                        in(reg) handler_address,
                        options(noreturn)
                    );
                }
            }
        }
    } else {
        // No try-catch handler found
        // Check per-thread uncaught exception handler (JVM-style)
        let thread_handler_fn = {
            let runtime = get_runtime().get();
            runtime.get_thread_exception_handler()
        };

        if let Some(handler_fn) = thread_handler_fn {
            // Call the per-thread uncaught exception handler
            let runtime = get_runtime().get();
            unsafe { call_beagle_fn_ptr(runtime, handler_fn, exception) };
            // Handler ran, now terminate since exception was uncaught
            println!("Uncaught exception after thread handler:");
            get_runtime().get_mut().println(exception).ok();
            unsafe { throw_error(stack_pointer, frame_pointer) };
        }

        // Check default (global) uncaught exception handler
        let default_handler_fn = {
            let runtime = get_runtime().get();
            runtime.default_exception_handler_fn
        };

        if let Some(handler_fn) = default_handler_fn {
            // Call the Beagle handler function
            let runtime = get_runtime().get();
            unsafe { call_beagle_fn_ptr(runtime, handler_fn, exception) };
            // Handler ran, now panic since exception was uncaught
            println!("Uncaught exception after default handler:");
            get_runtime().get_mut().println(exception).ok();
            unsafe { throw_error(stack_pointer, frame_pointer) };
        }

        // No handler at all - panic with stack trace
        println!("Uncaught exception:");
        get_runtime().get_mut().println(exception).ok();
        unsafe { throw_error(stack_pointer, frame_pointer) };
    }
}

pub unsafe extern "C" fn set_thread_exception_handler(handler_fn: usize) {
    let runtime = get_runtime().get_mut();
    runtime.set_thread_exception_handler(handler_fn);
}

pub unsafe extern "C" fn set_default_exception_handler(handler_fn: usize) {
    let runtime = get_runtime().get_mut();
    runtime.set_default_exception_handler(handler_fn);
}

/// Creates an Error struct on the heap
/// Returns tagged heap pointer to Error { kind, message, location }
pub unsafe extern "C" fn create_error(
    stack_pointer: usize,
    _frame_pointer: usize, // Frame pointer for GC stack walking (needed since create_struct can allocate)
    kind_str: usize, // Tagged string specifying the error variant (e.g., "StructError", "TypeError")
    message_str: usize, // Tagged string
    location_str: usize, // Tagged string or null
) -> usize {
    print_call_builtin(get_runtime().get(), "create_error");

    let runtime = get_runtime().get_mut();

    // Extract the error kind string to determine which variant to create
    // Use get_string which handles both string constants and heap-allocated strings
    let kind = runtime.get_string(stack_pointer, kind_str);

    // Use the general struct creation helper
    let fields = vec![message_str, location_str];
    runtime
        .create_struct(
            "beagle.core/SystemError",
            Some(&kind),
            &fields,
            stack_pointer,
        )
        .expect("Failed to create SystemError")
}

/// Helper to throw a runtime error with kind and message strings
/// This is a convenience function for Rust code to throw structured exceptions
pub unsafe fn throw_runtime_error(stack_pointer: usize, kind: &str, message: String) -> ! {
    // Get frame_pointer from thread-local (set by the builtin entry)
    let frame_pointer = get_saved_frame_pointer();

    // Allocate strings with proper GC root protection
    // kind_str must be rooted before allocating message_str since allocation can trigger GC
    let (kind_str, message_str) = {
        let runtime = get_runtime().get_mut();
        let kind_str: usize = runtime
            .allocate_string(stack_pointer, kind.to_string())
            .expect("Failed to allocate kind string")
            .into();

        // Register kind_str as a root before allocating message_str
        let kind_root_id = runtime.register_temporary_root(kind_str);

        let message_str: usize = runtime
            .allocate_string(stack_pointer, message)
            .expect("Failed to allocate message string")
            .into();

        // Get the potentially updated kind_str after GC
        let kind_str = runtime.unregister_temporary_root(kind_root_id);
        (kind_str, message_str)
    };
    // Runtime borrow is dropped here

    let null_location = BuiltInTypes::Null.tag(0) as usize;

    // Create the Error struct and throw it
    unsafe {
        let error = create_error(
            stack_pointer,
            frame_pointer,
            kind_str,
            message_str,
            null_location,
        );
        throw_exception(stack_pointer, frame_pointer, error);
    }
}

// ============================================================================
// Delimited Continuation Builtins
// ============================================================================

/// Push a prompt handler for delimited continuations.
/// Similar to push_exception_handler but for continuation capture.
pub unsafe extern "C" fn push_prompt_runtime(
    handler_address: usize,
    result_local: isize,
    link_register: usize,
    stack_pointer: usize,
    frame_pointer: usize,
) -> usize {
    print_call_builtin(get_runtime().get(), "push_prompt");
    let runtime = get_runtime().get_mut();

    // Generate a unique prompt ID to distinguish this handle block from others
    let prompt_id = runtime
        .prompt_id_counter
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    let handler = crate::runtime::PromptHandler {
        handler_address,
        stack_pointer,
        frame_pointer,
        link_register,
        result_local,
        prompt_id,
    };

    runtime.push_prompt_handler(handler);
    BuiltInTypes::null_value() as usize
}

/// Pop the current prompt handler.
/// If there's an invocation return point (continuation was invoked), returns via
/// return_from_shift_runtime to enable multi-shot continuations.
pub unsafe extern "C" fn pop_prompt_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    result_value: usize,
) -> usize {
    print_call_builtin(get_runtime().get(), "pop_prompt");
    let runtime = get_runtime().get_mut();
    let thread_id = std::thread::current().id();
    let debug_prompts = runtime.get_command_line_args().debug;

    // Get the current prompt's ID BEFORE popping - this tells us which handle block is completing
    let current_prompt_id = runtime
        .prompt_handlers
        .get(&thread_id)
        .and_then(|handlers| handlers.last())
        .map(|h| h.prompt_id);

    if debug_prompts {
        let rp_len = runtime
            .invocation_return_points
            .get(&thread_id)
            .map(|rps| rps.len())
            .unwrap_or(0);
        let top_rp = runtime
            .invocation_return_points
            .get(&thread_id)
            .and_then(|rps| rps.last())
            .map(|rp| {
                (
                    rp.stack_pointer,
                    rp.frame_pointer,
                    rp.return_address,
                    rp.prompt_id,
                )
            });
        eprintln!(
            "[pop_prompt] current_prompt_id={:?} return_points={} top={:?} sp={:#x} fp={:#x}",
            current_prompt_id, rp_len, top_rp, stack_pointer, frame_pointer
        );
    }

    // Check if there's an invocation return point for THIS handle block.
    //
    // For empty segments: the prompt was already popped by capture_continuation (perform/shift),
    // and NOT re-pushed by invoke_continuation. So current_prompt_id will be None.
    // But we still have an InvocationReturnPoint that we need to route through.
    //
    // For non-empty segments: the prompt was popped by capture_continuation, then RE-PUSHED
    // by invoke_continuation at the relocated location. So current_prompt_id will match.
    //
    // Strategy: Check for return points FIRST, regardless of current_prompt_id.
    // Match on the prompt_id stored in the return point itself.
    if let Some(return_points) = runtime.invocation_return_points.get(&thread_id)
        && let Some(top_point) = return_points.last()
    {
        let should_route =
            current_prompt_id.is_none() || current_prompt_id == Some(top_point.prompt_id);

        if should_route {
            runtime
                .return_from_shift_via_pop_prompt
                .insert(thread_id, true);
            unsafe {
                return_from_shift_runtime(
                    stack_pointer,
                    frame_pointer,
                    result_value,
                    BuiltInTypes::null_value() as usize,
                )
            };
        }
    }

    // Normal path - no matching invocation return point, just pop and return
    // Only pop if there's actually a prompt to pop
    if runtime.prompt_handler_count() > 0 {
        runtime.pop_prompt_handler();
    }

    // Clear invocation return points that belong to THIS handle block.
    // This ensures subsequent sequential handlers don't see stale return points.
    // NOTE: We do NOT manage continuation lifetime here - return_from_shift_runtime uses
    // the continuation pointer passed by the compiler.
    if let Some(prompt_id) = current_prompt_id
        && let Some(return_points) = runtime.invocation_return_points.get_mut(&thread_id) {
            // Remove all return points that were created for this prompt
            return_points.retain(|rp| rp.prompt_id != prompt_id);
        }

    // When all prompts are done, clear ALL continuation state for this thread
    if runtime.prompt_handler_count() == 0 {
        if let Some(return_points) = runtime.invocation_return_points.get_mut(&thread_id) {
            return_points.clear();
        }
        runtime.invocation_return_points.remove(&thread_id);
        runtime.prompt_handlers.remove(&thread_id);
        runtime.return_from_shift_via_pop_prompt.remove(&thread_id);
    }

    BuiltInTypes::null_value() as usize
}

/// Capture a continuation up to the nearest prompt.
/// This captures the stack segment and creates a continuation object.
///
/// Arguments:
/// - stack_pointer: current SP
/// - frame_pointer: current FP
/// - resume_address: where to resume when continuation is invoked
/// - result_local_offset: offset where the invoked value should be stored
///
/// Returns: a tagged continuation index that can be invoked later
pub unsafe extern "C" fn capture_continuation_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    resume_address: usize,
    result_local_offset: isize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "capture_continuation");
    let runtime = get_runtime().get_mut();
    let debug_prompts = runtime.get_command_line_args().debug;

    // Pop the prompt handler to get the delimiter information
    let prompt = match runtime.pop_prompt_handler() {
        Some(p) => p,
        None => {
            panic!(
                "shift/perform without enclosing reset/handle. This is a compiler bug - \
                 shift/perform must be inside a reset/handle block."
            );
        }
    };

    // NOTE: We intentionally do NOT clear InvocationReturnPoints here.
    // When nested performs happen (second perform inside first continuation body),
    // the outer return points are still valid - we need them when control unwinds.
    // The return_points form a proper stack that gets popped as continuations complete.
    let _thread_id = std::thread::current().id();

    // Calculate stack segment size (from current SP up to the prompt frame).
    //
    // NOTE: The prompt's stack_pointer is the bottom of the prompt frame
    // (after the function prologue reserves locals). If we only copy up to
    // prompt_sp, we exclude the prompt frame's locals and saved FP/LR.
    // That makes relocated continuations read locals from unmapped memory.
    //
    // To preserve locals and frame metadata, capture up to the prompt frame
    // pointer plus the saved FP/LR header.
    let prompt_sp = prompt.stack_pointer;
    let prompt_fp = prompt.frame_pointer;
    let frame_header_bytes = std::mem::size_of::<usize>() * 2; // saved FP + LR/return address
    let capture_top = prompt_fp.saturating_add(frame_header_bytes).max(prompt_sp);

    // Stack grows downward, so current SP < capture_top
    let stack_size = capture_top.saturating_sub(stack_pointer);

    // Allocate an opaque heap buffer for the stack segment
    let segment_words = stack_size.div_ceil(8);
    let segment_ptr = runtime
        .allocate(segment_words, stack_pointer, BuiltInTypes::HeapObject)
        .expect("Failed to allocate continuation segment");

    let mut segment_obj = HeapObject::from_tagged(segment_ptr);
    let is_large = segment_words > Header::MAX_INLINE_SIZE;
    segment_obj.writer_header_direct(Header {
        type_id: TYPE_ID_CONTINUATION_SEGMENT,
        type_data: stack_size as u32,
        size: if is_large { 0xFFFF } else { segment_words as u16 },
        opaque: true,
        marked: false,
        large: is_large,
    });
    if is_large {
        let size_ptr = (segment_obj.untagged() + 8) as *mut usize;
        unsafe { *size_ptr = segment_words };
    }

    if stack_size > 0 {
        let segment_bytes = segment_obj.get_opaque_bytes_mut();
        // SAFETY: stack_pointer points to valid stack memory of at least stack_size bytes
        unsafe {
            std::ptr::copy_nonoverlapping(
                stack_pointer as *const u8,
                segment_bytes.as_mut_ptr(),
                stack_size,
            );
        }
    }

    // Keep segment alive while allocating the continuation object
    let segment_root_id = runtime.register_temporary_root(segment_ptr);

    // Allocate the continuation heap object
    let cont_ptr = runtime
        .allocate(11, stack_pointer, BuiltInTypes::HeapObject)
        .expect("Failed to allocate continuation object");
    let segment_ptr = runtime.unregister_temporary_root(segment_root_id);

    let mut cont_obj = HeapObject::from_tagged(cont_ptr);
    ContinuationObject::initialize(
        &mut cont_obj,
        segment_ptr,
        stack_pointer,
        frame_pointer,
        resume_address,
        result_local_offset,
        &prompt,
    );

    if debug_prompts {
        eprintln!(
            "[capture_cont] prompt_id={} stack_size={} prompt_sp={:#x} prompt_fp={:#x} resume={:#x} cont_ptr={:#x}",
            prompt.prompt_id, stack_size, prompt_sp, prompt_fp, resume_address, cont_ptr
        );
    }

    // Return the continuation heap object pointer (tagged)
    cont_ptr
}

/// Return from shift body to the enclosing reset.
/// This pops the prompt and jumps to the prompt handler with the given value.
pub unsafe extern "C" fn return_from_shift_runtime(
    _stack_pointer: usize,
    _frame_pointer: usize,
    value: usize,
    cont_ptr: usize,
) -> ! {
    print_call_builtin(get_runtime().get(), "return_from_shift");

    let runtime = get_runtime().get_mut();
    let thread_id = std::thread::current().id();
    let debug_prompts = runtime.get_command_line_args().debug;

    // Check if we're being called from pop_prompt (via the flag).
    let from_pop_prompt = runtime
        .return_from_shift_via_pop_prompt
        .remove(&thread_id)
        .unwrap_or(false);

    // Check if this is a handler return (after `call-handler` in perform).
    // Handler returns should skip popping InvocationReturnPoints and use the passed continuation.
    let _is_handler_return = runtime
        .is_handler_return
        .remove(&thread_id)
        .unwrap_or(false);

    // Check if there's an invocation return point (multi-shot continuation case).
    // If a continuation was invoked via k(value), we should return to where k() was called,
    // not to the original prompt handler.
    //
    // ALWAYS check return_points first, even for handler returns.
    // For nested performs (second perform inside first continuation body),
    // we need to unwind through all the return_points to get back to outer callers.
    if let Some(return_points) = runtime.invocation_return_points.get_mut(&thread_id)
        && let Some(return_point) = return_points.pop()
    {
        if debug_prompts {
            eprintln!(
                "[return_from_shift] via_return_point from_pop_prompt={} remaining_after_pop={} rp_sp={:#x} rp_fp={:#x} ret_addr={:#x} prompt_id={}",
                from_pop_prompt,
                return_points.len(),
                return_point.stack_pointer,
                return_point.frame_pointer,
                return_point.return_address,
                return_point.prompt_id
            );
        }
        // For deep handler semantics: pop the prompt that was pushed by invoke_continuation.
        // Both empty and non-empty segments push a prompt that must be popped.
        let _ = runtime.pop_prompt_handler();

        // NOTE: We intentionally do NOT adjust continuation lifetime here.
        // The continuation pointer is carried by the shift body and used by return_from_shift.

        let new_sp = return_point.stack_pointer;
        let new_fp = return_point.frame_pointer;
        let return_address = return_point.return_address;
        let callee_saved = return_point.callee_saved_regs;

        // CRITICAL for multi-shot continuations: Restore the shift body's stack frame.
        // The continuation body may have written to stack locations that overlap with
        // the shift body's frame (e.g., the result_local slot where k is stored).
        // We must restore the original frame contents so that the shift body can
        // continue to access its local variables (including k for subsequent calls).
        let saved_frame = &return_point.saved_stack_frame;
        let _frame_src = if !saved_frame.is_empty() {
            saved_frame.as_ptr()
        } else {
            std::ptr::null()
        };
        let _frame_size = saved_frame.len();

        // Restore callee-saved registers and return to the call site
        // SAFETY: inline assembly for register/stack manipulation
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                // x86-64 callee-saved: rbx, r12-r15
                // Call the generated return_jump function to avoid optimizer issues
                // The function is generated by compile_continuation_trampolines in main.rs
                let runtime = get_runtime().get();
                let return_jump_fn = runtime
                    .get_function_by_name("beagle.builtin/return-jump")
                    .expect("return-jump function not found");
                let ptr: *const u8 = return_jump_fn.pointer.into();
                let return_jump_ptr: extern "C" fn(usize, usize, usize, usize, *const usize, *const u8, usize) -> ! =
                    unsafe { std::mem::transmute(ptr) };
                return_jump_ptr(new_sp, new_fp, value, return_address, callee_saved.as_ptr(), frame_src, frame_size);
            } else {
                // ARM64 callee-saved: x19-x28
                // Use explicit register constraints to avoid conflicts.
                // We put inputs in specific registers that won't be clobbered
                // before we use them.
                unsafe {
                    asm!(
                        // Restore callee-saved registers from array (x9 has the pointer)
                        "ldr x19, [x9]",
                        "ldr x20, [x9, #8]",
                        "ldr x21, [x9, #16]",
                        "ldr x22, [x9, #24]",
                        "ldr x23, [x9, #32]",
                        "ldr x24, [x9, #40]",
                        "ldr x25, [x9, #48]",
                        "ldr x26, [x9, #56]",
                        "ldr x27, [x9, #64]",
                        "ldr x28, [x9, #72]",
                        // Restore stack state and return
                        // x10=sp, x11=fp, x12=value, x13=return_addr
                        "mov sp, x10",
                        "mov x29, x11",
                        "mov x0, x12",
                        "br x13",
                        in("x9") callee_saved.as_ptr(),
                        in("x10") new_sp,
                        in("x11") new_fp,
                        in("x12") value,
                        in("x13") return_address,
                        options(noreturn)
                    );
                }
            }
        }
    }

    // No invocation return point - return to prompt handler using the passed continuation.
    let cont_ptr = if cont_ptr == BuiltInTypes::null_value() as usize {
        0
    } else {
        cont_ptr
    };

    if cont_ptr != 0 {
        let continuation = ContinuationObject::from_tagged(cont_ptr).unwrap_or_else(|| {
            panic!(
                "return_from_shift called with invalid continuation pointer {:#x}",
                cont_ptr
            );
        });
        let prompt = continuation.prompt_handler();
        if debug_prompts {
            eprintln!(
                "[return_from_shift] via_prompt from_pop_prompt={} prompt_sp={:#x} prompt_fp={:#x} handler={:#x} lr={:#x} prompt_id={}",
                from_pop_prompt,
                prompt.stack_pointer,
                prompt.frame_pointer,
                prompt.handler_address,
                prompt.link_register,
                prompt.prompt_id
            );
            // Debug: Check what's at the restored FP+8 (should be saved LR)
            let fp_plus_8 = (prompt.frame_pointer + 8) as *const usize;
            let saved_lr = unsafe { *fp_plus_8 };
            eprintln!(
                "[return_from_shift] via_prompt [FP+8]={:#x} (saved LR at stack)",
                saved_lr
            );
        }
        // Use the continuation's prompt - this contains the handle block's handler address
        // and stack state for the current handler being returned from.
        let handler_address = prompt.handler_address;
        let new_sp = prompt.stack_pointer;
        let new_fp = prompt.frame_pointer;
        let new_lr = prompt.link_register;
        let result_local_offset = prompt.result_local;

        // CRITICAL: Restore the stack segment before jumping!
        // The stack contents at capture time may have been overwritten by subsequent
        // code execution (function calls, etc.). We must restore the original stack data
        // for the epilogue to read correct FP/LR values.
        //
        // NOTE: We restore to continuation.original_sp (where the data was captured from),
        // NOT to prompt.stack_pointer (which is the push_prompt SP, above the capture region).
        let restore_sp = continuation.original_sp();
        let stack_segment_len = continuation.segment_len();
        if stack_segment_len > 0 {
            if debug_prompts {
                eprintln!(
                    "[return_from_shift] Restoring stack segment: {} bytes to original_sp={:#x} (prompt_sp={:#x})",
                    stack_segment_len,
                    restore_sp,
                    new_sp
                );
            }
            continuation.with_segment_bytes(|stack_segment| {
                // SAFETY: restore_sp points to valid stack memory that was captured earlier
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        stack_segment.as_ptr(),
                        restore_sp as *mut u8,
                        stack_segment_len,
                    );
                }
            });
        }

        // Store the value in the result local
        let result_ptr = (new_fp as isize).wrapping_add(result_local_offset) as *mut usize;
        // SAFETY: result_ptr points to a valid stack location
        if debug_prompts {
            eprintln!(
                "[return_from_shift] Writing value={:#x} to result_ptr={:#x}",
                value, result_ptr as usize
            );
        }
        unsafe {
            *result_ptr = value;
        }

        // Jump to the prompt handler with restored SP, FP, and LR
        // SAFETY: inline assembly for register/stack manipulation
        if debug_prompts {
            eprintln!(
                "[return_from_shift] via_prompt JUMPING: new_sp={:#x} new_fp={:#x} new_lr={:#x} handler={:#x} result_local_offset={} result_ptr={:#x}",
                new_sp, new_fp, new_lr, handler_address, result_local_offset, result_ptr as usize
            );
            // Check what's at key stack locations
            let fp_plus_0 = new_fp as *const usize;
            let fp_plus_8 = (new_fp + 8) as *const usize;
            let fp_minus_8 = (new_fp - 8) as *const usize;
            let saved_fp = unsafe { *fp_plus_0 };
            let saved_lr = unsafe { *fp_plus_8 };
            eprintln!(
                "[return_from_shift] Stack check: [FP+0]={:#x} [FP+8]={:#x} [FP-8]={:#x}",
                saved_fp,
                saved_lr,
                unsafe { *fp_minus_8 }
            );
            // Check what's at the caller's frame (main's frame)
            if saved_fp > 0x10000000 && saved_fp < 0x200000000 {
                let caller_fp_plus_0 = saved_fp as *const usize;
                let caller_fp_plus_8 = (saved_fp + 8) as *const usize;
                eprintln!(
                    "[return_from_shift] Caller frame check: [saved_FP+0]={:#x} [saved_FP+8]={:#x}",
                    unsafe { *caller_fp_plus_0 },
                    unsafe { *caller_fp_plus_8 }
                );
            }
            // Check the epilogue stack area - what's at SP + various offsets
            // The epilogue will do: add sp, sp, frame_size; ldp x29, x30, [sp, #0]
            // So we need to know what frame_size is to predict where the ldp reads from
            // For now, check what's at various offsets from SP
            let sp_plus_0 = new_sp as *const usize;
            let sp_plus_240 = (new_sp + 240) as *const usize; // common frame size
            eprintln!(
                "[return_from_shift] SP stack: [SP+0]={:#x} [SP+240]={:#x} [SP+248]={:#x}",
                unsafe { *sp_plus_0 },
                unsafe { *sp_plus_240 },
                unsafe { *((new_sp + 248) as *const usize) }
            );
        }
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                let _ = new_lr;
                unsafe {
                    asm!(
                        "mov rsp, {0}",
                        "mov rbp, {1}",
                        "jmp {2}",
                        in(reg) new_sp,
                        in(reg) new_fp,
                        in(reg) handler_address,
                        options(noreturn)
                    );
                }
            } else {
                // Note: We don't set LR (x30) here because:
                // 1. The first `bl` instruction in the handler code will overwrite it
                // 2. The function epilogue will load LR from the stack anyway
                // Setting it could cause issues if the value is stale
                let _ = new_lr;
                unsafe {
                    asm!(
                        "mov sp, {0}",
                        "mov x29, {1}",
                        "br {2}",
                        in(reg) new_sp,
                        in(reg) new_fp,
                        in(reg) handler_address,
                        options(noreturn)
                    );
                }
            }
        }
    }

    // No continuation found - this shouldn't happen in normal operation.
    panic!("return_from_shift called without captured continuation or return point");
}

/// Return from shift body to the enclosing reset, specifically for handler returns.
/// This is called after `call-handler` in perform. It sets the is_handler_return flag
/// so that return_from_shift_runtime skips popping InvocationReturnPoints.
/// This prevents nested handlers from incorrectly consuming outer handler return points.
pub unsafe extern "C" fn return_from_shift_handler_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
    cont_ptr: usize,
) -> ! {
    // SAFETY: We are in an unsafe function and caller guarantees valid stack/frame pointers
    unsafe {
        let runtime = get_runtime().get_mut();
        let thread_id = std::thread::current().id();
        runtime.is_handler_return.insert(thread_id, true);
        return_from_shift_runtime(stack_pointer, frame_pointer, value, cont_ptr)
    }
}

/// Invoke a captured continuation with a value.
/// This restores the stack segment and resumes execution.
/// The callee_saved_regs parameter contains the callee-saved registers that Beagle was using
/// when it called k() - these are saved at the very start of continuation_trampoline.
#[allow(improper_ctypes_definitions)]
pub unsafe extern "C" fn invoke_continuation_runtime(
    stack_pointer: usize,
    frame_pointer: usize,
    cont_ptr: usize,
    value: usize,
    callee_saved_regs: [usize; 10],
) -> ! {
    save_gc_context!(stack_pointer, frame_pointer);
    print_call_builtin(get_runtime().get(), "invoke_continuation");

    let runtime = get_runtime().get_mut();
    let debug_prompts = runtime.get_command_line_args().debug;

    let continuation = ContinuationObject::from_tagged(cont_ptr).unwrap_or_else(|| {
        panic!(
            "Invalid continuation pointer: {:#x}. This is a compiler bug - trying to invoke a continuation that doesn't exist.",
            cont_ptr
        );
    });

    // For multi-shot continuations: push an invocation return point.
    // When the continuation body completes (via return_from_shift_runtime),
    // it will pop this and return here with the result value.
    //
    // This is needed for BOTH empty and non-empty stack segments, because multi-shot
    // continuations need to preserve the shift body's stack frame (including local
    // variables like the continuation k itself) when resuming multiple times.
    let stack_segment_size = continuation.segment_len();
    let thread_id = std::thread::current().id();
    let prompt_id = continuation.prompt_id();

    // Create InvocationReturnPoint for ALL continuations (empty and non-empty).
    // This enables multi-shot and allows handlers to continue after calling resume().
    // For empty segments, we don't save the stack frame since there's nothing to relocate.

    // We need to get the return address for where k() was called in Beagle code.
    // The call chain is: Beagle code -> continuation_trampoline -> invoke_continuation_runtime
    //
    // Stack frame chain (on ARM64):
    // - invoke_continuation_runtime's FP (rust_fp) points to continuation_trampoline's FP
    // - continuation_trampoline's FP (trampoline_fp) points to Beagle's FP
    // - continuation_trampoline's FP+8 has the return address to Beagle code
    let rust_fp = get_current_rust_frame_pointer();
    // SAFETY: rust_fp points to valid stack frame
    let trampoline_fp = unsafe { *(rust_fp as *const usize) }; // continuation_trampoline's FP
    let beagle_fp = unsafe { *(trampoline_fp as *const usize) }; // Beagle caller's FP
    let beagle_return_address = unsafe { *((trampoline_fp + 8) as *const usize) }; // LR saved by trampoline
    // Beagle's SP when it called the closure is typically at trampoline's stack entry point
    // On ARM64, SP at function entry is FP + 16 (for the saved FP and LR)
    let beagle_sp = trampoline_fp + 16;

    // callee_saved_regs was already captured at the start of continuation_trampoline,
    // before any Rust code could clobber the registers

    // For NON-EMPTY segments: Save the shift body's stack frame for multi-shot continuations.
    // The frame contains local variables including the continuation k.
    // When the continuation body runs, it may write to stack locations that
    // overlap with this frame (e.g., the result_local), so we need to
    // save and restore the frame contents.
    //
    // For EMPTY segments: Save the HANDLER's stack frame (from beagle_sp to original_sp).
    // When the continuation runs at original_sp/fp, it may allocate stack space that
    // grows downward and overlaps with the handler's frame at beagle_sp/fp.
    // We need to save and restore the handler's frame to prevent corruption.
    let saved_stack_frame = if stack_segment_size > 0 {
        // Non-empty segment: Save the shift body's stack frame (beagle_sp to beagle_fp)
        let frame_size = beagle_fp.saturating_sub(beagle_sp);
        let mut saved_stack_frame = vec![0u8; frame_size];
        if frame_size > 0 {
            // SAFETY: beagle_sp points to valid stack memory
            unsafe {
                std::ptr::copy_nonoverlapping(
                    beagle_sp as *const u8,
                    saved_stack_frame.as_mut_ptr(),
                    frame_size,
                );
            }
        }
        saved_stack_frame
    } else {
        // Empty segment: Save the HANDLER's stack frame (beagle_sp to original_sp).
        // The continuation runs at original_sp/fp and may allocate stack downward,
        // potentially overwriting the handler's frame. Save it to prevent corruption.
        let frame_size = continuation.original_sp().saturating_sub(beagle_sp);
        let mut saved_stack_frame = vec![0u8; frame_size];
        if frame_size > 0 {
            // SAFETY: beagle_sp points to valid stack memory
            unsafe {
                std::ptr::copy_nonoverlapping(
                    beagle_sp as *const u8,
                    saved_stack_frame.as_mut_ptr(),
                    frame_size,
                );
            }
        }
        saved_stack_frame
    };

    let resume_address = continuation.resume_address();

    // Handle empty stack segment case (capture and prompt at same stack depth)
    // For empty segments, DON'T create InvocationReturnPoint - use single-shot semantics.
    if stack_segment_size == 0 {
        // Empty segment: perform and handle are at the same stack depth.
        // For single-shot use (handler calls resume once), we DON'T need
        // InvocationReturnPoints or the return trampoline - just jump back
        // to the continuation with the original link register.

        // Store the value in the result local (relative to the original FP)
        let result_ptr = (continuation.original_fp() as isize)
            .wrapping_add(continuation.result_local()) as *mut usize;
        // SAFETY: result_ptr points to valid stack location
        unsafe { *result_ptr = value };

        // Re-push the prompt so subsequent performs can find it.
        let relocated_prompt = continuation.prompt_handler();
        runtime.push_prompt_handler(relocated_prompt);

        // For empty segments, DON'T create InvocationReturnPoint and DON'T use return trampoline.
        // This is single-shot - the continuation will return normally through the handle block.

        if debug_prompts {
            eprintln!(
                "[invoke_cont] empty_segment prompt_id={} value={:#x} orig_sp={:#x} orig_fp={:#x} result_local_off={}",
                prompt_id,
                value,
                continuation.original_sp(),
                continuation.original_fp(),
                continuation.result_local()
            );
        }

        cfg_if::cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                let runtime = get_runtime().get();
                let jump_fn = runtime
                    .get_function_by_name("beagle.builtin/invoke-continuation-jump")
                    .expect("invoke-continuation-jump function not found");
                let ptr: *const u8 = jump_fn.pointer.into();
                let jump_ptr: extern "C" fn(usize, usize, usize, usize, usize, usize, usize) -> ! =
                    unsafe { std::mem::transmute(ptr) };
                jump_ptr(
                    continuation.original_fp(),
                    resume_address,
                    0,
                    continuation.original_sp(),
                    continuation.original_fp(),
                    result_ptr as usize,
                    value,
                );
            } else {
                // For ARM64: jump with original LR (not return_trampoline)
                // The continuation will return normally, eventually reaching pop_prompt_runtime
                let original_lr = continuation.prompt_link_register();
                let safe_sp = (stack_pointer - 16) & !0xF;
                unsafe {
                    asm!(
                        "mov sp, {0}",
                        "mov x29, {1}",
                        "mov x30, {2}",
                        "br {3}",
                        in(reg) safe_sp,
                        in(reg) continuation.original_fp(),
                        in(reg) original_lr,
                        in(reg) resume_address,
                        options(noreturn)
                    );
                }
            }
        }
    }

    // Non-empty stack segment - need to copy and relocate
    // For non-empty segments, create InvocationReturnPoint for multi-shot support.

    runtime
        .invocation_return_points
        .entry(thread_id)
        .or_default()
        .push(crate::runtime::InvocationReturnPoint {
            stack_pointer: beagle_sp,
            frame_pointer: beagle_fp,
            return_address: beagle_return_address,
            callee_saved_regs,
            saved_stack_frame,
            prompt_id,
        });

    if debug_prompts {
        let rp_len = runtime
            .invocation_return_points
            .get(&thread_id)
            .map(|rps| rps.len())
            .unwrap_or(0);
        eprintln!(
            "[invoke_cont] push_return_point prompt_id={} stack_seg={} beagle_sp={:#x} beagle_fp={:#x} ret_addr={:#x} rp_len={}",
            prompt_id, stack_segment_size, beagle_sp, beagle_fp, beagle_return_address, rp_len
        );
    }

    #[allow(unused_variables)]
    let return_trampoline = continuation_return_trampoline as usize;

    // Get the actual current RSP - we need to place the stack segment below this,
    // not below the Beagle SP, to avoid corrupting Rust's stack
    let actual_rsp: usize;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::asm!("mov {}, rsp", out(reg) actual_rsp);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        actual_rsp = stack_pointer; // Fall back to Beagle SP for other architectures
    }

    // Place the stack segment below the actual RSP with safety margin
    // IMPORTANT: Need a LARGE margin because Rust code (eprintln, etc.) uses the stack
    // and the stack segment includes high addresses that could overlap with Rust's stack.
    // The margin needs to be at least as large as the stack segment to avoid overlap.
    let safety_margin = stack_segment_size.max(4096) + 4096; // At least segment size + 4KB extra
    let new_sp = actual_rsp - stack_segment_size - safety_margin;
    let new_sp = new_sp & !0xF; // Align to 16 bytes

    // Copy the stack segment
    // SAFETY: new_sp points to valid stack memory below actual RSP
    continuation.with_segment_bytes(|stack_segment| {
        unsafe {
            std::ptr::copy_nonoverlapping(
                stack_segment.as_ptr(),
                new_sp as *mut u8,
                stack_segment_size,
            );
        }
    });

    // Calculate relocation offset for frame pointers
    let relocation_offset = (new_sp as isize) - (continuation.original_sp() as isize);

    // Adjust the frame pointer
    let new_fp = (continuation.original_fp() as isize + relocation_offset) as usize;

    // Check if new_fp is within the copied stack segment
    let stack_segment_end = new_sp + stack_segment_size;

    // Calculate result_ptr for later - the assembly will write the value
    let result_ptr = (new_fp as isize).wrapping_add(continuation.result_local()) as *mut usize;

    // Relocate the entire frame pointer chain within the copied stack segment
    let mut current_fp = new_fp;

    while current_fp >= new_sp && current_fp < stack_segment_end {
        let saved_fp_ptr = current_fp as *mut usize;
        // SAFETY: saved_fp_ptr points within valid stack segment
        let old_saved_fp = unsafe { *saved_fp_ptr };

        if old_saved_fp >= continuation.original_sp()
            && old_saved_fp < continuation.original_sp() + stack_segment_size
        {
            let new_saved_fp = (old_saved_fp as isize + relocation_offset) as usize;
            // SAFETY: saved_fp_ptr points within valid stack segment
            unsafe { *saved_fp_ptr = new_saved_fp };
            current_fp = new_saved_fp;
        } else {
            break;
        }
    }

    // Deep handler semantics: Push a relocated prompt handler for the non-empty stack segment case.
    // This enables sequential effects (multiple performs in one handle block).
    //
    // CRITICAL: The prompt's SP must be relocated by the same offset as the stack segment.
    // The original prompt SP marked the upper boundary of the captured stack.
    // After relocation, subsequent captures should capture from current SP up to
    // the relocated prompt SP (the new upper boundary).
    // IMPORTANT: Preserve the prompt_id so return points can match this prompt.
    let relocated_prompt_sp =
        (continuation.prompt_stack_pointer() as isize + relocation_offset) as usize;
    let relocated_prompt_fp =
        (continuation.prompt_frame_pointer() as isize + relocation_offset) as usize;

    let relocated_prompt = crate::runtime::PromptHandler {
        handler_address: continuation.handler_address(),
        stack_pointer: relocated_prompt_sp,
        frame_pointer: relocated_prompt_fp,
        link_register: continuation.prompt_link_register(),
        result_local: continuation.prompt_result_local(),
        prompt_id: continuation.prompt_id(),
    };
    runtime.push_prompt_handler(relocated_prompt);

    // SAFETY: inline assembly for stack/register manipulation
    // The assembly writes the result value AFTER switching stacks to avoid Rust stack corruption
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            // On x86-64, call the generated invoke_continuation_jump function
            let runtime = get_runtime().get();
            let jump_fn = runtime
                .get_function_by_name("beagle.builtin/invoke-continuation-jump")
                .expect("invoke-continuation-jump function not found");
            let ptr: *const u8 = jump_fn.pointer.into();
            let jump_ptr: extern "C" fn(usize, usize, usize, usize, usize, usize, usize) -> ! =
                unsafe { std::mem::transmute(ptr) };
            jump_ptr(
                continuation.original_fp,
                resume_address,
                stack_segment_size,  // Non-zero, so will use non-empty path
                new_sp,
                new_fp,
                result_ptr as usize,  // Where to write the result
                value,                // The result value to write
            );
        } else {
            // ARM64: Write result before switching stacks (TODO: move to assembly)
            unsafe { *result_ptr = value };
            unsafe {
                asm!(
                    "mov sp, {0}",
                    "mov x29, {1}",
                    "mov x30, {2}",
                    "br {3}",
                    in(reg) new_sp,
                    in(reg) new_fp,
                    in(reg) return_trampoline,
                    in(reg) resume_address,
                    options(noreturn)
                );
            }
        }
    }
}

// ============================================================================

/// Trampoline function for continuation closures.
/// When a continuation is captured, it's wrapped in a closure with this function as its body.
/// When called, it extracts the continuation pointer from the closure and invokes it.
///
/// Layout: closure_ptr points to a closure with:
/// - header (8 bytes)
/// - function pointer (8 bytes) - points to this trampoline
/// - cont_ptr (8 bytes) - the captured continuation heap object pointer (tagged)
///
/// Note: This is called as a regular closure body, so we receive (closure_ptr, value)
/// and need to get SP/FP ourselves.
#[allow(unused_variables)]
pub unsafe extern "C" fn continuation_trampoline(closure_ptr: usize, value: usize) -> ! {
    // Save callee-saved registers IMMEDIATELY before any Rust code runs
    // These are the registers Beagle was using when it called k()
    let mut saved_regs = [0usize; 10];
    // SAFETY: inline assembly to save callee-saved registers
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "aarch64")] {
            unsafe {
                std::arch::asm!(
                    "str x19, [{0}]",
                    "str x20, [{0}, #8]",
                    "str x21, [{0}, #16]",
                    "str x22, [{0}, #24]",
                    "str x23, [{0}, #32]",
                    "str x24, [{0}, #40]",
                    "str x25, [{0}, #48]",
                    "str x26, [{0}, #56]",
                    "str x27, [{0}, #64]",
                    "str x28, [{0}, #72]",
                    in(reg) saved_regs.as_mut_ptr(),
                );
            }
        } else if #[cfg(target_arch = "x86_64")] {
            unsafe {
                std::arch::asm!(
                    "mov [{0}], rbx",
                    "mov [{0} + 8], r12",
                    "mov [{0} + 16], r13",
                    "mov [{0} + 24], r14",
                    "mov [{0} + 32], r15",
                    in(reg) saved_regs.as_mut_ptr(),
                );
            }
        }
    }

    // Get current stack pointer and frame pointer
    let stack_pointer: usize;
    let frame_pointer: usize;

    // SAFETY: inline assembly to read SP/FP registers
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            unsafe {
                std::arch::asm!(
                    "mov {0}, rsp",
                    "mov {1}, rbp",
                    out(reg) stack_pointer,
                    out(reg) frame_pointer,
                );
            }
        } else {
            unsafe {
                std::arch::asm!(
                    "mov {0}, sp",
                    "mov {1}, x29",
                    out(reg) stack_pointer,
                    out(reg) frame_pointer,
                );
            }
        }
    }

    // Extract the continuation pointer from the closure's free variables
    // Closure layout: header(8) + fn_ptr(8) + num_free(8) + num_locals(8) + free_vars...
    // First free variable is at offset 32
    let untagged_closure = BuiltInTypes::untag(closure_ptr);

    // SAFETY: closure memory layout is known
    let cont_ptr = unsafe { *((untagged_closure + 32) as *const usize) };

    // Now invoke the continuation, passing the saved callee-saved registers
    // SAFETY: invoke_continuation_runtime is an unsafe function
    unsafe {
        invoke_continuation_runtime(stack_pointer, frame_pointer, cont_ptr, value, saved_regs)
    }
}

/// Return trampoline for multi-shot continuations.
/// When a continuation body returns, this trampoline is called to route the
/// return value through `return_from_shift_runtime` so that multi-shot
/// continuations work correctly.
///
/// On entry: the return value is in x0 (ARM64) or rax (x86-64)
/// This gets SP/FP and calls return_from_shift_runtime.
#[allow(unused_variables)]
pub unsafe extern "C" fn continuation_return_trampoline(value: usize) -> ! {
    // Get current stack pointer and frame pointer
    let stack_pointer: usize;
    let frame_pointer: usize;

    // SAFETY: inline assembly to read SP/FP registers
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            unsafe {
                std::arch::asm!(
                    "mov {0}, rsp",
                    "mov {1}, rbp",
                    out(reg) stack_pointer,
                    out(reg) frame_pointer,
                );
            }
        } else {
            unsafe {
                std::arch::asm!(
                    "mov {0}, sp",
                    "mov {1}, x29",
                    out(reg) stack_pointer,
                    out(reg) frame_pointer,
                );
            }
        }
    }

    // Route through return_from_shift_runtime so multi-shot works
    // SAFETY: return_from_shift_runtime is an unsafe function
    unsafe {
        return_from_shift_runtime(
            stack_pointer,
            frame_pointer,
            value,
            BuiltInTypes::null_value() as usize,
        )
    }
}

/// Dispatch a call to a multi-arity function.
/// Takes stack_pointer, frame_pointer, the multi-arity function object and the number of arguments.
/// Returns the function pointer for the matching arity, or throws an ArityError if no match found.
///
/// Layout of MultiArityFunction heap object:
/// - header (8 bytes)
/// - num_arities (8 bytes, tagged int)
/// - entries: [arity (tagged), fn_ptr (tagged Function), is_variadic (tagged bool)] * num_arities
pub extern "C" fn dispatch_multi_arity(
    stack_pointer: usize,
    frame_pointer: usize,
    multi_arity_obj: usize,
    arg_count: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);

    // Untag the heap object
    let ptr = BuiltInTypes::untag(multi_arity_obj);

    // Read num_arities from offset 1 (after header)
    let num_arities_raw = unsafe { *((ptr + 8) as *const usize) };
    let num_arities = BuiltInTypes::untag(num_arities_raw);

    // Untag the arg_count
    let arg_count = BuiltInTypes::untag(arg_count);

    // Collect available arities for error message
    let mut available_arities = Vec::new();

    // Search for matching arity - first try exact match for non-variadic
    for i in 0..num_arities {
        let base_offset = 16 + i * 24; // header(8) + num_arities(8) + i * (3 * 8)
        let arity = unsafe { *((ptr + base_offset) as *const usize) };
        let arity = BuiltInTypes::untag(arity);
        let fn_ptr = unsafe { *((ptr + base_offset + 8) as *const usize) };
        let is_variadic = unsafe { *((ptr + base_offset + 16) as *const usize) };
        let is_variadic = is_variadic == BuiltInTypes::true_value() as usize;

        available_arities.push(if is_variadic {
            format!("{}+", arity)
        } else {
            arity.to_string()
        });

        if !is_variadic && arity == arg_count {
            return fn_ptr;
        }
    }

    // Then try variadic match (arg_count >= min_arity)
    for i in 0..num_arities {
        let base_offset = 16 + i * 24;
        let min_arity = unsafe { *((ptr + base_offset) as *const usize) };
        let min_arity = BuiltInTypes::untag(min_arity);
        let fn_ptr = unsafe { *((ptr + base_offset + 8) as *const usize) };
        let is_variadic = unsafe { *((ptr + base_offset + 16) as *const usize) };
        let is_variadic = is_variadic == BuiltInTypes::true_value() as usize;

        if is_variadic && arg_count >= min_arity {
            return fn_ptr;
        }
    }

    // No matching arity found - throw a catchable ArityError
    unsafe {
        throw_runtime_error(
            stack_pointer,
            "ArityError",
            format!(
                "No matching arity for multi-arity function call with {} argument(s). Available arities: {}",
                arg_count,
                available_arities.join(", ")
            ),
        );
    }
}

// It is very important that this isn't inlined
// because other code is expecting that
// If we inline, we need to remove a skip frame
impl Runtime {
    pub fn install_builtins(&mut self) -> Result<(), Box<dyn Error>> {
        self.add_builtin_function(
            "beagle.__internal_test__/many_args",
            many_args as *const u8,
            false,
            11,
        )?;

        self.add_builtin_function("beagle.core/_println", println_value as *const u8, false, 1)?;

        self.add_builtin_function("beagle.core/_print", print_value as *const u8, false, 1)?;

        // Multi-arity dispatch builtin (needs stack/frame pointer for throwing ArityError)
        self.add_builtin_function_with_fp(
            "beagle.builtin/dispatch-multi-arity",
            dispatch_multi_arity as *const u8,
            true,
            true,
            4, // stack_pointer + frame_pointer + multi_arity_obj + arg_count
        )?;

        // to_string now takes (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp(
            "beagle.core/to-string",
            to_string as *const u8,
            true,
            true,
            3,
        )?;
        // to_number now takes (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp(
            "beagle.core/to-number",
            to_number as *const u8,
            true,
            true,
            3,
        )?;

        // allocate now takes (stack_pointer, frame_pointer, size)
        self.add_builtin_function_with_fp(
            "beagle.builtin/allocate",
            allocate as *const u8,
            true,
            true,
            3,
        )?;

        // allocate_zeroed for arrays (zeroes memory)
        self.add_builtin_function_with_fp(
            "beagle.builtin/allocate-zeroed",
            allocate_zeroed as *const u8,
            true,
            true,
            3,
        )?;

        // allocate_float now takes (stack_pointer, frame_pointer, size)
        self.add_builtin_function_with_fp(
            "beagle.builtin/allocate-float",
            allocate_float as *const u8,
            true,
            true,
            3,
        )?;

        self.add_builtin_function(
            "beagle.builtin/fill-object-fields",
            fill_object_fields as *const u8,
            false,
            2,
        )?;

        // copy_object now takes (stack_pointer, frame_pointer, object_pointer)
        self.add_builtin_function_with_fp(
            "beagle.builtin/copy-object",
            copy_object as *const u8,
            true,
            true,
            3,
        )?;

        self.add_builtin_function(
            "beagle.builtin/copy-from-to-object",
            copy_from_to_object as *const u8,
            false,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/copy-array-range",
            copy_array_range as *const u8,
            false,
            4,
        )?;

        // make_closure now takes (stack_pointer, frame_pointer, function, num_free, free_variable_pointer)
        self.add_builtin_function_with_fp(
            "beagle.builtin/make-closure",
            make_closure as *const u8,
            true,
            true,
            5,
        )?;

        // make_function_object takes (stack_pointer, frame_pointer, function)
        self.add_builtin_function_with_fp(
            "beagle.builtin/make-function-object",
            make_function_object as *const u8,
            true,
            true,
            3,
        )?;

        self.add_builtin_function(
            "beagle.builtin/property-access",
            property_access as *const u8,
            false,
            3,
        )?;

        self.add_builtin_function_with_fp(
            "beagle.builtin/protocol-dispatch",
            protocol_dispatch as *const u8,
            true,
            true,
            5, // stack_pointer, frame_pointer, first_arg, cache_location, dispatch_table_ptr
        )?;

        // type_of now takes (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp(
            "beagle.core/type-of",
            type_of as *const u8,
            true,
            true,
            3,
        )?;

        // get_os now takes (stack_pointer, frame_pointer)
        self.add_builtin_function_with_fp(
            "beagle.core/get-os",
            get_os as *const u8,
            true,
            true,
            2,
        )?;

        self.add_builtin_function("beagle.core/equal", equal as *const u8, false, 2)?;

        // write_field now takes (stack_pointer, frame_pointer, struct_pointer, str_constant_ptr, property_cache_location, value)
        self.add_builtin_function_with_fp(
            "beagle.builtin/write-field",
            write_field as *const u8,
            true,
            true,
            6, // stack_pointer + frame_pointer + 4 original args
        )?;

        // throw_error now takes (stack_pointer, frame_pointer)
        self.add_builtin_function_with_fp(
            "beagle.builtin/throw-error",
            throw_error as *const u8,
            true,
            true,
            2,
        )?;

        // throw_type_error takes (stack_pointer, frame_pointer) and throws catchable TypeError
        self.add_builtin_function_with_fp(
            "beagle.builtin/throw-type-error",
            throw_type_error as *const u8,
            true,
            true,
            2,
        )?;

        // check_arity now takes (stack_pointer, frame_pointer, function_pointer, expected_args)
        self.add_builtin_function_with_fp(
            "beagle.builtin/check-arity",
            check_arity as *const u8,
            true,
            true,
            4,
        )?;

        self.add_builtin_function(
            "beagle.builtin/is-function-variadic",
            is_function_variadic as *const u8,
            false,
            1, // function_pointer
        )?;

        self.add_builtin_function(
            "beagle.builtin/get-function-min-args",
            get_function_min_args as *const u8,
            false,
            1, // function_pointer
        )?;

        // pack_variadic_args_from_stack now takes (stack_pointer, frame_pointer, args_base, total_args, min_args)
        self.add_builtin_function_with_fp(
            "beagle.builtin/pack-variadic-args-from-stack",
            pack_variadic_args_from_stack as *const u8,
            true,
            true,
            5,
        )?;

        // build_rest_array_from_locals takes (stack_pointer, frame_pointer, arg_count, min_args, first_local_index)
        self.add_builtin_function_with_fp(
            "beagle.builtin/build-rest-array-from-locals",
            build_rest_array_from_locals as *const u8,
            true,
            true,
            5,
        )?;

        // call_variadic_function_value now takes (stack_pointer, frame_pointer, function_ptr, args_array, is_closure, closure_ptr)
        self.add_builtin_function_with_fp(
            "beagle.builtin/call-variadic-function-value",
            call_variadic_function_value as *const u8,
            true,
            true,
            6,
        )?;

        self.add_builtin_function(
            "beagle.builtin/push-exception-handler",
            push_exception_handler_runtime as *const u8,
            false,
            5, // handler_address, result_local, link_register, stack_pointer, frame_pointer
        )?;

        self.add_builtin_function(
            "beagle.builtin/pop-exception-handler",
            pop_exception_handler_runtime as *const u8,
            false,
            0,
        )?;

        // throw_exception now takes (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp(
            "beagle.builtin/throw-exception",
            throw_exception as *const u8,
            true,
            true,
            3,
        )?;

        // Delimited continuation builtins
        self.add_builtin_function(
            "beagle.builtin/push-prompt",
            push_prompt_runtime as *const u8,
            false,
            5, // handler_address, result_local, link_register, stack_pointer, frame_pointer
        )?;

        self.add_builtin_function(
            "beagle.builtin/pop-prompt",
            pop_prompt_runtime as *const u8,
            false,
            0,
        )?;

        // capture-continuation takes (stack_pointer, frame_pointer, resume_address, result_local_offset)
        self.add_builtin_function_with_fp(
            "beagle.builtin/capture-continuation",
            capture_continuation_runtime as *const u8,
            true,
            true,
            4,
        )?;

        // return-from-shift takes (stack_pointer, frame_pointer, value, cont_ptr)
        self.add_builtin_function_with_fp(
            "beagle.builtin/return-from-shift",
            return_from_shift_runtime as *const u8,
            true,
            true,
            4,
        )?;

        // return-from-shift-handler is for handler returns (after call-handler in perform)
        // It sets is_handler_return flag so return_from_shift skips InvocationReturnPoints
        self.add_builtin_function_with_fp(
            "beagle.builtin/return-from-shift-handler",
            return_from_shift_handler_runtime as *const u8,
            true,
            true,
            4,
        )?;

        // invoke-continuation takes (stack_pointer, frame_pointer, cont_ptr, value)
        self.add_builtin_function_with_fp(
            "beagle.builtin/invoke-continuation",
            invoke_continuation_runtime as *const u8,
            true,
            true,
            4,
        )?;

        // continuation-trampoline is the function body for continuation closures
        // takes (closure_ptr, value) - it's a closure body, not a regular builtin
        // NOTE: On x86-64, this trampoline will be replaced by a generated version
        // in main.rs to avoid optimizer issues with inline assembly in release builds
        self.add_builtin_function(
            "beagle.builtin/continuation-trampoline",
            continuation_trampoline as *const u8,
            false,
            2,
        )?;

        self.add_builtin_function(
            "beagle.core/set-thread-exception-handler!",
            set_thread_exception_handler as *const u8,
            false,
            1, // handler_fn
        )?;

        self.add_builtin_function(
            "beagle.core/set-default-exception-handler!",
            set_default_exception_handler as *const u8,
            false,
            1, // handler_fn
        )?;

        self.add_builtin_function(
            "beagle.builtin/create-error",
            create_error as *const u8,
            true,
            4, // stack_pointer, kind_str, message_str, location_str
        )?;

        self.add_builtin_function("beagle.builtin/assert!", placeholder as *const u8, false, 0)?;

        // gc needs both stack_pointer and frame_pointer
        // stack_pointer is arg 0, frame_pointer is arg 1
        self.add_builtin_function_with_fp("beagle.core/gc", gc as *const u8, true, true, 2)?;

        // Math builtins - all now take (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp(
            "beagle.core/sqrt",
            sqrt_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/floor",
            floor_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/ceil",
            ceil_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/abs",
            abs_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/round",
            round_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/truncate",
            truncate_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/max",
            max_builtin as *const u8,
            true,
            true,
            4,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/min",
            min_builtin as *const u8,
            true,
            true,
            4,
        )?;
        self.add_builtin_function("beagle.core/even?", is_even as *const u8, false, 1)?;
        self.add_builtin_function("beagle.core/odd?", is_odd as *const u8, false, 1)?;
        self.add_builtin_function("beagle.core/positive?", is_positive as *const u8, false, 1)?;
        self.add_builtin_function("beagle.core/negative?", is_negative as *const u8, false, 1)?;
        self.add_builtin_function("beagle.core/zero?", is_zero as *const u8, false, 1)?;
        self.add_builtin_function_with_fp(
            "beagle.core/clamp",
            clamp_builtin as *const u8,
            true,
            true,
            5,
        )?;
        self.add_builtin_function("beagle.core/gcd", gcd_builtin as *const u8, false, 2)?;
        self.add_builtin_function("beagle.core/lcm", lcm_builtin as *const u8, false, 2)?;
        self.add_builtin_function_with_fp(
            "beagle.core/random",
            random_builtin as *const u8,
            true,
            true,
            2,
        )?;
        self.add_builtin_function(
            "beagle.core/random-int",
            random_int_builtin as *const u8,
            false,
            1,
        )?;
        self.add_builtin_function(
            "beagle.core/random-range",
            random_range_builtin as *const u8,
            false,
            2,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/sin",
            sin_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/cos",
            cos_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/tan",
            tan_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/asin",
            asin_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/acos",
            acos_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/atan",
            atan_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/atan2",
            atan2_builtin as *const u8,
            true,
            true,
            4,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/exp",
            exp_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/log",
            log_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/log10",
            log10_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/log2",
            log2_builtin as *const u8,
            true,
            true,
            3,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/pow",
            pow_builtin as *const u8,
            true,
            true,
            4,
        )?;
        self.add_builtin_function_with_fp(
            "beagle.core/to-float",
            to_float_builtin as *const u8,
            true,
            true,
            3,
        )?;

        // new_thread now takes (stack_pointer, frame_pointer, function)
        self.add_builtin_function_with_fp(
            "beagle.core/thread",
            new_thread as *const u8,
            true,
            true,
            3,
        )?;

        self.add_builtin_function(
            "beagle.ffi/load-library",
            load_library as *const u8,
            false,
            1,
        )?;

        // get_function now takes (stack_pointer, frame_pointer, library_struct, function_name, types, return_type)
        self.add_builtin_function_with_fp(
            "beagle.ffi/get-function",
            get_function as *const u8,
            true,
            true,
            6,
        )?;

        self.add_builtin_function(
            "beagle.ffi/call-ffi-info",
            call_ffi_info as *const u8,
            true,
            8,
        )?;

        self.add_builtin_function("beagle.ffi/allocate", ffi_allocate as *const u8, false, 1)?;
        self.add_builtin_function(
            "beagle.ffi/deallocate",
            ffi_deallocate as *const u8,
            false,
            1,
        )?;

        self.add_builtin_function("beagle.ffi/get-u32", ffi_get_u32 as *const u8, false, 2)?;

        self.add_builtin_function("beagle.ffi/set-i16", ffi_set_i16 as *const u8, false, 3)?;

        self.add_builtin_function("beagle.ffi/set-i32", ffi_set_i32 as *const u8, false, 3)?;

        self.add_builtin_function("beagle.ffi/set-u8", ffi_set_u8 as *const u8, false, 3)?;

        self.add_builtin_function("beagle.ffi/get-u8", ffi_get_u8 as *const u8, false, 2)?;

        self.add_builtin_function("beagle.ffi/get-i32", ffi_get_i32 as *const u8, false, 2)?;

        self.add_builtin_function(
            "beagle.ffi/get-string",
            ffi_get_string as *const u8,
            true,
            4,
        )?;

        self.add_builtin_function(
            "beagle.ffi/create-array",
            ffi_create_array as *const u8,
            true,
            3,
        )?;

        self.add_builtin_function(
            "beagle.ffi/copy-bytes",
            ffi_copy_bytes as *const u8,
            false,
            5,
        )?;

        self.add_builtin_function("beagle.ffi/realloc", ffi_realloc as *const u8, false, 2)?;

        self.add_builtin_function(
            "beagle.ffi/buffer-size",
            ffi_buffer_size as *const u8,
            false,
            1,
        )?;

        self.add_builtin_function(
            "beagle.ffi/write-buffer-offset",
            ffi_write_buffer_offset as *const u8,
            false,
            4,
        )?;

        self.add_builtin_function(
            "beagle.ffi/translate-bytes",
            ffi_translate_bytes as *const u8,
            false,
            4,
        )?;

        self.add_builtin_function(
            "beagle.ffi/reverse-bytes",
            ffi_reverse_bytes as *const u8,
            false,
            3,
        )?;

        self.add_builtin_function("beagle.ffi/find-byte", ffi_find_byte as *const u8, false, 4)?;

        self.add_builtin_function(
            "beagle.ffi/copy-bytes_filter",
            ffi_copy_bytes_filter as *const u8,
            false,
            6,
        )?;

        // __pause needs both stack_pointer and frame_pointer for FP-chain walking
        self.add_builtin_function_with_fp(
            "beagle.builtin/__pause",
            __pause as *const u8,
            true,
            true,
            2,
        )?;

        // register_c_call needs both stack_pointer and frame_pointer for FP-chain walking
        self.add_builtin_function_with_fp(
            "beagle.builtin/__register_c_call",
            register_c_call as *const u8,
            true,
            true,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/__unregister_c_call",
            unregister_c_call as *const u8,
            false,
            0,
        )?;

        self.add_builtin_function(
            "beagle.builtin/__run_thread",
            run_thread as *const u8,
            false,
            1,
        )?;

        // __get_my_thread_obj needs stack_pointer and frame_pointer to call __pause
        self.add_builtin_function_with_fp(
            "beagle.builtin/__get_my_thread_obj",
            get_my_thread_obj as *const u8,
            true,
            true,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/__get_and_unregister_temp_root",
            get_and_unregister_temp_root as *const u8,
            false,
            1,
        )?;

        self.add_builtin_function(
            "beagle.builtin/update-binding",
            update_binding as *const u8,
            false,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/get-binding",
            get_binding as *const u8,
            false,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/set-current-namespace",
            set_current_namespace as *const u8,
            false,
            1,
        )?;

        // wait_for_input now takes (stack_pointer, frame_pointer)
        self.add_builtin_function_with_fp(
            "beagle.builtin/wait-for-input",
            wait_for_input as *const u8,
            true,
            true,
            2,
        )?;

        // read_line now takes (stack_pointer, frame_pointer)
        self.add_builtin_function_with_fp(
            "beagle.builtin/read-line",
            read_line as *const u8,
            true,
            true,
            2,
        )?;

        self.add_builtin_function(
            "beagle.builtin/print-byte",
            print_byte as *const u8,
            false,
            1,
        )?;

        // char_code now takes (stack_pointer, frame_pointer, string)
        self.add_builtin_function_with_fp(
            "beagle.builtin/char-code",
            char_code as *const u8,
            true,
            true,
            3,
        )?;

        // char_from_code now takes (stack_pointer, frame_pointer, code)
        self.add_builtin_function_with_fp(
            "beagle.builtin/char-from-code",
            char_from_code as *const u8,
            true,
            true,
            3,
        )?;

        // eval now takes (stack_pointer, frame_pointer, code)
        self.add_builtin_function_with_fp("beagle.core/eval", eval as *const u8, true, true, 3)?;

        self.add_builtin_function("beagle.core/sleep", sleep as *const u8, false, 1)?;

        self.add_builtin_function("beagle.core/time-now", time_now as *const u8, false, 0)?;

        self.add_builtin_function(
            "beagle.builtin/register-extension",
            register_extension as *const u8,
            false,
            4,
        )?;

        // get_string_index now takes (stack_pointer, frame_pointer, string, index)
        self.add_builtin_function_with_fp(
            "beagle.builtin/get-string-index",
            get_string_index as *const u8,
            true,
            true,
            4,
        )?;

        self.add_builtin_function(
            "beagle.builtin/get-string-length",
            get_string_length as *const u8,
            false,
            1,
        )?;

        // string_concat now takes (stack_pointer, frame_pointer, a, b)
        self.add_builtin_function_with_fp(
            "beagle.core/string-concat",
            string_concat as *const u8,
            true,
            true,
            4,
        )?;

        // substring now takes (stack_pointer, frame_pointer, string, start, length)
        self.add_builtin_function_with_fp(
            "beagle.core/substring",
            substring as *const u8,
            true,
            true,
            5,
        )?;
        // uppercase now takes (stack_pointer, frame_pointer, string)
        self.add_builtin_function_with_fp(
            "beagle.core/uppercase",
            uppercase as *const u8,
            true,
            true,
            3,
        )?;

        // lowercase now takes (stack_pointer, frame_pointer, string)
        self.add_builtin_function_with_fp(
            "beagle.core/lowercase",
            lowercase as *const u8,
            true,
            true,
            3,
        )?;

        // split now takes (stack_pointer, frame_pointer, string, delimiter)
        self.add_builtin_function_with_fp("beagle.core/split", split as *const u8, true, true, 4)?;

        // join now takes (stack_pointer, frame_pointer, array, separator)
        self.add_builtin_function_with_fp("beagle.core/join", join as *const u8, true, true, 4)?;

        // trim now takes (stack_pointer, frame_pointer, string)
        self.add_builtin_function_with_fp("beagle.core/trim", trim as *const u8, true, true, 3)?;

        // trim-left now takes (stack_pointer, frame_pointer, string)
        self.add_builtin_function_with_fp(
            "beagle.core/trim-left",
            trim_left as *const u8,
            true,
            true,
            3,
        )?;

        // trim-right now takes (stack_pointer, frame_pointer, string)
        self.add_builtin_function_with_fp(
            "beagle.core/trim-right",
            trim_right as *const u8,
            true,
            true,
            3,
        )?;

        // starts-with? now takes (stack_pointer, frame_pointer, string, prefix)
        self.add_builtin_function_with_fp(
            "beagle.core/starts-with?",
            starts_with as *const u8,
            true,
            true,
            4,
        )?;

        // ends-with? now takes (stack_pointer, frame_pointer, string, suffix)
        self.add_builtin_function_with_fp(
            "beagle.core/ends-with?",
            ends_with as *const u8,
            true,
            true,
            4,
        )?;

        // contains? now takes (stack_pointer, frame_pointer, string, substr)
        self.add_builtin_function_with_fp(
            "beagle.core/contains?",
            string_contains as *const u8,
            true,
            true,
            4,
        )?;

        // index-of now takes (stack_pointer, frame_pointer, string, substr)
        self.add_builtin_function_with_fp(
            "beagle.core/index-of",
            index_of as *const u8,
            true,
            true,
            4,
        )?;

        // last-index-of now takes (stack_pointer, frame_pointer, string, substr)
        self.add_builtin_function_with_fp(
            "beagle.core/last-index-of",
            last_index_of as *const u8,
            true,
            true,
            4,
        )?;

        // replace now takes (stack_pointer, frame_pointer, string, from, to)
        self.add_builtin_function_with_fp(
            "beagle.core/replace",
            replace_string as *const u8,
            true,
            true,
            5,
        )?;

        // blank? now takes (stack_pointer, frame_pointer, string)
        self.add_builtin_function_with_fp(
            "beagle.core/blank?",
            blank_string as *const u8,
            true,
            true,
            3,
        )?;

        // replace-first now takes (stack_pointer, frame_pointer, string, from, to)
        self.add_builtin_function_with_fp(
            "beagle.core/replace-first",
            replace_first_string as *const u8,
            true,
            true,
            5,
        )?;

        // pad-left now takes (stack_pointer, frame_pointer, string, width, pad_char)
        self.add_builtin_function_with_fp(
            "beagle.core/pad-left",
            pad_left_string as *const u8,
            true,
            true,
            5,
        )?;

        // pad-right now takes (stack_pointer, frame_pointer, string, width, pad_char)
        self.add_builtin_function_with_fp(
            "beagle.core/pad-right",
            pad_right_string as *const u8,
            true,
            true,
            5,
        )?;

        // lines now takes (stack_pointer, frame_pointer, string)
        self.add_builtin_function_with_fp(
            "beagle.core/lines",
            lines_string as *const u8,
            true,
            true,
            3,
        )?;

        // words now takes (stack_pointer, frame_pointer, string)
        self.add_builtin_function_with_fp(
            "beagle.core/words",
            words_string as *const u8,
            true,
            true,
            3,
        )?;

        // hash now takes (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp("beagle.builtin/hash", hash as *const u8, true, true, 3)?;

        self.add_builtin_function(
            "beagle.builtin/is-keyword",
            is_keyword as *const u8,
            false,
            1,
        )?;

        // keyword_to_string now takes (stack_pointer, frame_pointer, keyword)
        self.add_builtin_function_with_fp(
            "beagle.builtin/keyword-to-string",
            keyword_to_string as *const u8,
            true,
            true,
            3,
        )?;

        // string_to_keyword now takes (stack_pointer, frame_pointer, string_value)
        self.add_builtin_function_with_fp(
            "beagle.builtin/string-to-keyword",
            string_to_keyword as *const u8,
            true,
            true,
            3,
        )?;

        // load_keyword_constant_runtime now takes (stack_pointer, frame_pointer, index)
        self.add_builtin_function_with_fp(
            "beagle.builtin/load-keyword-constant-runtime",
            load_keyword_constant_runtime as *const u8,
            true,
            true,
            3,
        )?;

        // pop_count now takes (stack_pointer, frame_pointer, value)
        self.add_builtin_function_with_fp(
            "beagle.builtin/pop-count",
            pop_count as *const u8,
            true,
            true,
            3,
        )?;

        // read_full_file now takes (stack_pointer, frame_pointer, file_name)
        self.add_builtin_function_with_fp(
            "beagle.core/read-full-file",
            read_full_file as *const u8,
            true,
            true,
            3,
        )?;

        // Diagnostic system builtins
        self.add_builtin_function_with_fp(
            "beagle.core/diagnostics",
            diagnostics as *const u8,
            true,
            true,
            2, // stack_pointer, frame_pointer
        )?;

        self.add_builtin_function_with_fp(
            "beagle.core/diagnostics-for-file",
            diagnostics_for_file as *const u8,
            true,
            true,
            3, // stack_pointer, frame_pointer, file_path
        )?;

        self.add_builtin_function_with_fp(
            "beagle.core/files-with-diagnostics",
            files_with_diagnostics as *const u8,
            true,
            true,
            2, // stack_pointer, frame_pointer
        )?;

        self.add_builtin_function_with_fp(
            "beagle.core/clear-diagnostics",
            clear_diagnostics as *const u8,
            true,
            true,
            2, // stack_pointer, frame_pointer
        )?;

        self.install_rust_collection_builtins()?;
        self.install_regex_builtins()?;

        // Effect handler builtins
        self.add_builtin_function(
            "beagle.builtin/push-handler",
            push_handler_builtin as *const u8,
            false,
            2, // protocol_key_str, handler_instance
        )?;

        self.add_builtin_function(
            "beagle.builtin/pop-handler",
            pop_handler_builtin as *const u8,
            false,
            1, // protocol_key_str
        )?;

        // find-handler needs stack_pointer for get_string error handling
        self.add_builtin_function_with_fp(
            "beagle.builtin/find-handler",
            find_handler_builtin as *const u8,
            true,  // needs stack_pointer
            false, // doesn't need frame_pointer
            2,     // stack_pointer, protocol_key_str
        )?;

        // get-enum-type needs stack_pointer and frame_pointer for GC-safe string allocation
        self.add_builtin_function_with_fp(
            "beagle.builtin/get-enum-type",
            get_enum_type_builtin as *const u8,
            true,
            true,
            3, // stack_pointer, frame_pointer, value
        )?;

        // call-handler calls handler.handle(op, resume) using protocol dispatch
        self.add_builtin_function_with_fp(
            "beagle.builtin/call-handler",
            call_handler_builtin as *const u8,
            true,
            true,
            6, // stack_pointer, frame_pointer, handler, enum_type_ptr, op_value, resume
        )?;

        Ok(())
    }
}

// ============================================================================
// Effect Handler Builtins
// ============================================================================

/// Push a handler onto the thread-local handler stack
pub extern "C" fn push_handler_builtin(protocol_key_ptr: usize, handler_instance: usize) -> usize {
    let runtime = get_runtime().get();
    let protocol_key = runtime.get_string_literal(protocol_key_ptr);
    crate::runtime::push_handler(protocol_key.to_string(), handler_instance);
    BuiltInTypes::null_value() as usize
}

/// Pop a handler from the thread-local handler stack
pub extern "C" fn pop_handler_builtin(protocol_key_ptr: usize) -> usize {
    let runtime = get_runtime().get();
    let protocol_key = runtime.get_string_literal(protocol_key_ptr);
    crate::runtime::pop_handler(&protocol_key);
    BuiltInTypes::null_value() as usize
}

/// Find a handler in the thread-local handler stack
/// Returns the handler instance or null if not found
pub extern "C" fn find_handler_builtin(stack_pointer: usize, protocol_key_ptr: usize) -> usize {
    let runtime = get_runtime().get();
    // Use get_string which handles both string literals and heap-allocated strings
    let protocol_key = runtime.get_string(stack_pointer, protocol_key_ptr);
    match crate::runtime::find_handler(&protocol_key) {
        Some(handler) => handler,
        None => BuiltInTypes::null_value() as usize,
    }
}

/// Get the enum name for a value (by examining its struct_id/type_id)
/// Returns a string pointer to the enum name, or null if not an enum variant
pub extern "C" fn get_enum_type_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    value: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let _ = stack_pointer; // Used by save_gc_context!

    // Check if the value is a heap pointer
    if !BuiltInTypes::is_heap_pointer(value) {
        return BuiltInTypes::null_value() as usize;
    }

    // Get the struct_id from the heap object
    // For custom structs (including enum variants), header type_id is 0,
    // and the actual struct_id is stored separately
    let heap_obj = HeapObject::from_tagged(value);
    let header_type_id = heap_obj.get_type_id();

    // Only custom structs (type_id == 0) can be enum variants
    if header_type_id != 0 {
        return BuiltInTypes::null_value() as usize;
    }

    // Get struct_id - it's tagged, need to untag it
    let struct_id_tagged = heap_obj.get_struct_id();
    let struct_id = BuiltInTypes::untag(struct_id_tagged);

    // Look up the enum name for this struct_id
    let runtime = get_runtime().get_mut();
    match runtime.get_enum_name_for_variant(struct_id) {
        Some(enum_name) => {
            // Allocate a string for the enum name
            match runtime.allocate_string(stack_pointer, enum_name.to_string()) {
                Ok(string_ptr) => usize::from(string_ptr),
                Err(_) => BuiltInTypes::null_value() as usize,
            }
        }
        None => BuiltInTypes::null_value() as usize,
    }
}

/// Call the `handle` method on a handler instance with the given operation and resume continuation.
///
/// This is used by `perform` to dispatch to the handler's `handle(op, resume)` method.
/// The protocol key is constructed from the enum type: "Handler<{enum_type}>"
///
/// # Arguments
/// * `stack_pointer` - Stack pointer for GC safety
/// * `frame_pointer` - Frame pointer for GC safety
/// * `handler` - The handler instance (implements Handler(T))
/// * `enum_type_ptr` - String pointer to the enum type name (e.g., "myns/Async")
/// * `op_value` - The operation value (enum variant)
/// * `resume` - The continuation closure
///
/// # Returns
/// The result of calling handler.handle(op, resume)
pub extern "C" fn call_handler_builtin(
    stack_pointer: usize,
    frame_pointer: usize,
    handler: usize,
    enum_type_ptr: usize,
    op_value: usize,
    resume: usize,
) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);

    let runtime = get_runtime().get_mut();

    // Get the enum type string (can be heap-allocated or string literal)
    let enum_type = runtime.get_string(stack_pointer, enum_type_ptr);

    // Construct the protocol key: "beagle.effect/Handler<{enum_type}>"
    // Handler is always from beagle.effect (the core effect handler protocol)
    let protocol_key = format!("beagle.effect/Handler<{}>", enum_type);

    // The dispatch key is "beagle.effect/handle" since Handler is in beagle.effect
    let dispatch_key = "beagle.effect/handle".to_string();

    // Look up the dispatch table
    let dispatch_table_ptr = runtime.get_dispatch_table_ptr(&protocol_key, "handle");

    if dispatch_table_ptr.is_none() {
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "NoHandlerError",
                format!(
                    "No handler registered for protocol {}, dispatch key {}",
                    protocol_key, dispatch_key
                ),
            );
        }
    }

    let dispatch_table_ptr = dispatch_table_ptr.unwrap();
    let dispatch_table = unsafe { &*(dispatch_table_ptr as *const DispatchTable) };

    // Get the type_id from the handler to look up the function
    let type_id = if BuiltInTypes::is_heap_pointer(handler) {
        let heap_obj = HeapObject::from_tagged(handler);
        let header_type_id = heap_obj.get_type_id();

        if header_type_id == 0 {
            // Custom struct - use struct_id from type_data (tagged)
            heap_obj.get_struct_id()
        } else {
            // Built-in heap type
            0x8000_0000_0000_0000 | header_type_id
        }
    } else {
        // Tagged primitive
        let kind = BuiltInTypes::get_kind(handler);
        let tag = kind.get_tag() as usize;
        let primitive_index = if tag == 2 { 2 } else { tag + 16 };
        0x8000_0000_0000_0000 | primitive_index
    };

    // Look up function pointer in dispatch table
    let fn_ptr = if type_id & 0x8000_0000_0000_0000 != 0 {
        let primitive_index = type_id & 0x7FFF_FFFF_FFFF_FFFF;
        dispatch_table.lookup_primitive(primitive_index)
    } else {
        let struct_id = BuiltInTypes::untag(type_id);
        dispatch_table.lookup_struct(struct_id)
    };

    if fn_ptr == 0 {
        let handler_repr = runtime
            .get_repr(handler, 0)
            .unwrap_or_else(|| "<unknown>".to_string());
        unsafe {
            throw_runtime_error(
                stack_pointer,
                "NoHandlerError",
                format!(
                    "Handler type does not implement {} protocol. Handler: {}",
                    protocol_key, handler_repr
                ),
            );
        }
    }

    // Call the handle function: fn handle(self, op, resume) -> result
    // The function pointer is tagged, untag it
    let fn_ptr = BuiltInTypes::untag(fn_ptr as usize);

    // The handler is compiled Beagle code, NOT a builtin.
    // Beagle functions do NOT take stack_pointer and frame_pointer as explicit args.
    // The signature is just: fn handle(self, op, resume) -> result
    let func: extern "C" fn(usize, usize, usize) -> usize = unsafe { std::mem::transmute(fn_ptr) };

    func(handler, op_value, resume)
}

/// Allocates a Beagle struct from Rust using struct registry lookup.
///
/// WARNING: This function is NOT GC-safe if fields contain heap pointers!
/// The allocation can trigger GC, making the field values stale.
/// For GC-safe struct allocation, allocate first, then peek roots, then write fields.
///
/// # Arguments
/// * `struct_name` - Fully qualified name like "beagle.core/CompilerWarning"
/// * `fields` - Slice of pre-tagged Beagle values (must match struct field count)
///
/// # Returns
/// Tagged pointer to the allocated struct
#[allow(dead_code)]
unsafe fn allocate_struct(
    runtime: &mut Runtime,
    stack_pointer: usize,
    struct_name: &str,
    fields: &[usize],
) -> Result<usize, String> {
    // Look up struct definition from registry
    let (struct_id, struct_def) = runtime
        .get_struct(struct_name)
        .ok_or_else(|| format!("Struct {} not found", struct_name))?;

    let struct_id = BuiltInTypes::Int.tag(struct_id as isize) as usize;

    // Validate field count matches struct definition
    if fields.len() != struct_def.fields.len() {
        return Err(format!(
            "Expected {} fields for {}, got {}",
            struct_def.fields.len(),
            struct_name,
            fields.len()
        ));
    }

    // Allocate heap object (same as create_error line 1630-1632)
    let obj_ptr = runtime
        .allocate(fields.len(), stack_pointer, BuiltInTypes::HeapObject)
        .map_err(|e| format!("Allocation failed: {}", e))?;

    // Write struct_id to header's type_data field (same as create_error lines 1636-1653)
    let heap_obj = HeapObject::from_tagged(obj_ptr);

    let untagged = heap_obj.untagged();
    let header_ptr = untagged as *mut usize;

    // Write struct_id to type_data field (bytes 3-6) without changing other fields
    // Header layout (little-endian):
    //   Bits 0-7:   Byte 0 (flags)
    //   Bits 8-15:  Byte 1 (padding)
    //   Bits 16-23: Byte 2 (size) - MUST PRESERVE
    //   Bits 24-55: Bytes 3-6 (type_data) - WRITE HERE
    //   Bits 56-63: Byte 7 (type_id) - MUST PRESERVE
    unsafe {
        let current_header = *header_ptr;
        let mask = 0x00FFFFFFFF000000; // Mask for bits 24-55 (bytes 3-6, the type_data field)
        let shifted_type_id = struct_id << 24; // Shift to bit 24
        let new_header = (current_header & !mask) | shifted_type_id;
        *header_ptr = new_header;
    }
    // Write all fields (same as create_error lines 1656-1658)
    for (i, &field_value) in fields.iter().enumerate() {
        heap_obj.write_field(i as i32, field_value);
    }

    Ok(obj_ptr)
}

/// Converts a Diagnostic to a Beagle struct.
/// Wrapper that ensures temporary roots are always cleaned up.
unsafe fn diagnostic_to_struct(
    runtime: &mut Runtime,
    stack_pointer: usize,
    diagnostic: &crate::compiler::Diagnostic,
) -> Result<usize, String> {
    let mut temp_roots: Vec<usize> = Vec::new();

    // Do all the work that might fail
    // SAFETY: diagnostic_to_struct_impl is unsafe for the same reasons as this function
    let result =
        unsafe { diagnostic_to_struct_impl(runtime, stack_pointer, diagnostic, &mut temp_roots) };

    // Always clean up temporary roots, whether success or failure
    for root_id in temp_roots {
        runtime.unregister_temporary_root(root_id);
    }

    result
}

/// Inner implementation that does the actual work.
/// Any early return via ? will be caught by the wrapper which cleans up temp_roots.
unsafe fn diagnostic_to_struct_impl(
    runtime: &mut Runtime,
    stack_pointer: usize,
    diagnostic: &crate::compiler::Diagnostic,
    temp_roots: &mut Vec<usize>,
) -> Result<usize, String> {
    use crate::collections::{GcHandle, PersistentVec};

    // Helper macro to allocate, register as temp root, and return the root INDEX
    // We store root IDs and retrieve updated values before use (GC safety)
    macro_rules! alloc_and_root {
        ($expr:expr) => {{
            let val: usize = $expr;
            let root_id = runtime.register_temporary_root(val);
            temp_roots.push(root_id);
            temp_roots.len() - 1 // Return the index into temp_roots
        }};
    }

    // Create severity as a DiagnosticSeverity enum variant
    // Each variant is a zero-field struct named "beagle.core/DiagnosticSeverity.<variant>"
    let severity_variant_name = match diagnostic.severity {
        crate::compiler::Severity::Error => "beagle.core/DiagnosticSeverity.error",
        crate::compiler::Severity::Warning => "beagle.core/DiagnosticSeverity.warning",
        crate::compiler::Severity::Info => "beagle.core/DiagnosticSeverity.info",
        crate::compiler::Severity::Hint => "beagle.core/DiagnosticSeverity.hint",
    };
    let severity_root_idx = alloc_and_root!(
        unsafe { allocate_struct(runtime, stack_pointer, severity_variant_name, &[]) }
            .map_err(|e| format!("Failed to create severity enum variant: {}", e))?
    );

    // Create kind string
    let kind_root_idx = alloc_and_root!(
        runtime
            .allocate_string(stack_pointer, diagnostic.kind.clone())
            .map_err(|e| format!("Failed to create kind string: {}", e))?
            .into()
    );

    // Create file_name string
    let file_name_root_idx = alloc_and_root!(
        runtime
            .allocate_string(stack_pointer, diagnostic.file_name.clone())
            .map_err(|e| format!("Failed to create file_name string: {}", e))?
            .into()
    );

    // Create message string
    let message_root_idx = alloc_and_root!(
        runtime
            .allocate_string(stack_pointer, diagnostic.message.clone())
            .map_err(|e| format!("Failed to create message string: {}", e))?
            .into()
    );

    // Create line and column as tagged ints (no allocation, no rooting needed)
    let line_tagged = BuiltInTypes::Int.tag(diagnostic.line as isize) as usize;
    let column_tagged = BuiltInTypes::Int.tag(diagnostic.column as isize) as usize;

    // Handle optional enum_name field
    let enum_name_root_idx = if let Some(ref enum_name) = diagnostic.enum_name {
        Some(alloc_and_root!(
            runtime
                .allocate_string(stack_pointer, enum_name.clone())
                .map_err(|e| format!("Failed to create enum_name string: {}", e))?
                .into()
        ))
    } else {
        None
    };

    // Handle optional missing_variants field
    let missing_variants_root_idx = if let Some(ref missing_variants) = diagnostic.missing_variants {
        // Build persistent vector of variant strings
        let vec_handle = PersistentVec::empty(runtime, stack_pointer)
            .map_err(|e| format!("Failed to create empty vector: {}", e))?;
        let mut vec = vec_handle.as_tagged();
        let mut vec_root_id = runtime.register_temporary_root(vec);
        temp_roots.push(vec_root_id);
        let vec_root_index = temp_roots.len() - 1;

        for variant in missing_variants {
            let variant_str: usize = runtime
                .allocate_string(stack_pointer, variant.clone())
                .map_err(|e| format!("Failed to create variant string: {}", e))?
                .into();
            let variant_root_id = runtime.register_temporary_root(variant_str);
            vec = runtime.peek_temporary_root(vec_root_id);
            let vec_handle = GcHandle::from_tagged(vec);
            vec = PersistentVec::push(runtime, stack_pointer, vec_handle, variant_str)
                .map_err(|e| format!("Failed to push variant: {}", e))?
                .as_tagged();
            runtime.unregister_temporary_root(variant_root_id);
            runtime.unregister_temporary_root(vec_root_id);
            vec_root_id = runtime.register_temporary_root(vec);
            temp_roots[vec_root_index] = vec_root_id;
        }

        Some(vec_root_index)
    } else {
        None
    };

    // Allocate the struct FIRST (this can trigger GC)
    // Then peek all root values AFTER allocation
    let struct_ptr = {
        // Look up struct definition
        let (struct_id, struct_def) = runtime
            .get_struct("beagle.core/Diagnostic")
            .ok_or_else(|| "Struct beagle.core/Diagnostic not found".to_string())?;

        if struct_def.fields.len() != 8 {
            return Err(format!(
                "Expected 8 fields for Diagnostic, got {}",
                struct_def.fields.len()
            ));
        }

        let struct_id_tagged = BuiltInTypes::Int.tag(struct_id as isize) as usize;

        // Allocate the struct - this can trigger GC!
        let obj_ptr = runtime
            .allocate(8, stack_pointer, BuiltInTypes::HeapObject)
            .map_err(|e| format!("Allocation failed: {}", e))?;

        // Write struct_id to header
        let heap_obj = HeapObject::from_tagged(obj_ptr);
        let untagged = heap_obj.untagged();
        let header_ptr = untagged as *mut usize;
        unsafe {
            let current_header = *header_ptr;
            let mask = 0x00FFFFFFFF000000;
            let shifted_type_id = struct_id_tagged << 24;
            let new_header = (current_header & !mask) | shifted_type_id;
            *header_ptr = new_header;
        }

        obj_ptr
    };

    // NOW peek all values from roots - AFTER allocation/GC
    let severity_tagged = runtime.peek_temporary_root(temp_roots[severity_root_idx]);
    let kind_tagged = runtime.peek_temporary_root(temp_roots[kind_root_idx]);
    let file_name_tagged = runtime.peek_temporary_root(temp_roots[file_name_root_idx]);
    let message_tagged = runtime.peek_temporary_root(temp_roots[message_root_idx]);
    let enum_name_tagged = enum_name_root_idx
        .map(|idx| runtime.peek_temporary_root(temp_roots[idx]))
        .unwrap_or(BuiltInTypes::null_value() as usize);
    let missing_variants_tagged = missing_variants_root_idx
        .map(|idx| runtime.peek_temporary_root(temp_roots[idx]))
        .unwrap_or(BuiltInTypes::null_value() as usize);

    // Write all fields to the struct
    // Fields: severity, kind, file-name, line, column, message, enum-name, missing-variants
    let heap_obj = HeapObject::from_tagged(struct_ptr);
    heap_obj.write_field(0, severity_tagged);
    heap_obj.write_field(1, kind_tagged);
    heap_obj.write_field(2, file_name_tagged);
    heap_obj.write_field(3, line_tagged);
    heap_obj.write_field(4, column_tagged);
    heap_obj.write_field(5, message_tagged);
    heap_obj.write_field(6, enum_name_tagged);
    heap_obj.write_field(7, missing_variants_tagged);

    Ok(struct_ptr)
}

/// Returns all diagnostics across all files as a PersistentVec of Diagnostic structs
pub unsafe extern "C" fn diagnostics(stack_pointer: usize, frame_pointer: usize) -> usize {
    use crate::collections::{GcHandle, PersistentVec};

    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Clone diagnostics to avoid holding the lock while processing
    let all_diagnostics: Vec<crate::compiler::Diagnostic> = {
        let store_guard = runtime.diagnostic_store.lock().unwrap();
        store_guard.all().cloned().collect()
    };

    // Start with empty persistent vector using Rust API directly
    let vec_handle = match PersistentVec::empty(runtime, stack_pointer) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("diagnostics: Failed to create empty vector: {}", e);
            return BuiltInTypes::null_value() as usize;
        }
    };
    let mut vec = vec_handle.as_tagged();

    // Register vec as a temporary root to protect it from GC during the loop
    let mut vec_root_id = runtime.register_temporary_root(vec);

    // Convert each diagnostic to struct and add to persistent vector
    for diagnostic in all_diagnostics.iter() {
        match unsafe { diagnostic_to_struct(runtime, stack_pointer, diagnostic) } {
            Ok(diagnostic_struct) => {
                let diagnostic_root_id = runtime.register_temporary_root(diagnostic_struct);
                let vec_updated = runtime.peek_temporary_root(vec_root_id);
                let diagnostic_struct_updated = runtime.peek_temporary_root(diagnostic_root_id);

                let vec_handle = GcHandle::from_tagged(vec_updated);
                match PersistentVec::push(
                    runtime,
                    stack_pointer,
                    vec_handle,
                    diagnostic_struct_updated,
                ) {
                    Ok(new_vec) => {
                        vec = new_vec.as_tagged();
                    }
                    Err(e) => {
                        eprintln!("diagnostics: Failed to push diagnostic: {}", e);
                    }
                }

                runtime.unregister_temporary_root(diagnostic_root_id);
                runtime.unregister_temporary_root(vec_root_id);
                vec_root_id = runtime.register_temporary_root(vec);
            }
            Err(e) => {
                eprintln!("Warning: Failed to convert diagnostic: {}", e);
            }
        }
    }

    runtime.unregister_temporary_root(vec_root_id);
    vec
}

/// Returns diagnostics for a specific file as a PersistentVec of Diagnostic structs
pub unsafe extern "C" fn diagnostics_for_file(
    stack_pointer: usize,
    frame_pointer: usize,
    file_path: usize,
) -> usize {
    use crate::collections::{GcHandle, PersistentVec};

    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Get file path string
    let file_path_str = {
        let tag = BuiltInTypes::get_kind(file_path);
        if tag != BuiltInTypes::HeapObject {
            eprintln!("diagnostics_for_file: Invalid file path argument (not a heap object)");
            return BuiltInTypes::null_value() as usize;
        }
        let heap_object = HeapObject::from_tagged(file_path);
        if heap_object.get_type_id() != 2 {
            eprintln!("diagnostics_for_file: Invalid file path argument (not a string)");
            return BuiltInTypes::null_value() as usize;
        }
        let bytes = heap_object.get_string_bytes();
        unsafe { std::str::from_utf8_unchecked(bytes).to_string() }
    };

    // Clone diagnostics for the specific file
    let file_diagnostics: Vec<crate::compiler::Diagnostic> = {
        let store_guard = runtime.diagnostic_store.lock().unwrap();
        store_guard
            .for_file(&file_path_str).cloned()
            .unwrap_or_default()
    };

    // Build the result vector
    let vec_handle = match PersistentVec::empty(runtime, stack_pointer) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("diagnostics_for_file: Failed to create empty vector: {}", e);
            return BuiltInTypes::null_value() as usize;
        }
    };
    let mut vec = vec_handle.as_tagged();
    let mut vec_root_id = runtime.register_temporary_root(vec);

    for diagnostic in file_diagnostics.iter() {
        match unsafe { diagnostic_to_struct(runtime, stack_pointer, diagnostic) } {
            Ok(diagnostic_struct) => {
                let diagnostic_root_id = runtime.register_temporary_root(diagnostic_struct);
                let vec_updated = runtime.peek_temporary_root(vec_root_id);
                let diagnostic_struct_updated = runtime.peek_temporary_root(diagnostic_root_id);

                let vec_handle = GcHandle::from_tagged(vec_updated);
                match PersistentVec::push(
                    runtime,
                    stack_pointer,
                    vec_handle,
                    diagnostic_struct_updated,
                ) {
                    Ok(new_vec) => {
                        vec = new_vec.as_tagged();
                    }
                    Err(e) => {
                        eprintln!("diagnostics_for_file: Failed to push diagnostic: {}", e);
                    }
                }

                runtime.unregister_temporary_root(diagnostic_root_id);
                runtime.unregister_temporary_root(vec_root_id);
                vec_root_id = runtime.register_temporary_root(vec);
            }
            Err(e) => {
                eprintln!("Warning: Failed to convert diagnostic: {}", e);
            }
        }
    }

    runtime.unregister_temporary_root(vec_root_id);
    vec
}

/// Returns a list of file paths that have diagnostics
pub unsafe extern "C" fn files_with_diagnostics(
    stack_pointer: usize,
    frame_pointer: usize,
) -> usize {
    use crate::collections::{GcHandle, PersistentVec};

    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    // Get list of files
    let files: Vec<String> = {
        let store_guard = runtime.diagnostic_store.lock().unwrap();
        store_guard.files().cloned().collect()
    };

    // Build the result vector
    let vec_handle = match PersistentVec::empty(runtime, stack_pointer) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("files_with_diagnostics: Failed to create empty vector: {}", e);
            return BuiltInTypes::null_value() as usize;
        }
    };
    let mut vec = vec_handle.as_tagged();
    let mut vec_root_id = runtime.register_temporary_root(vec);

    for file in files.iter() {
        let file_str: usize = match runtime.allocate_string(stack_pointer, file.clone()) {
            Ok(s) => s.into(),
            Err(e) => {
                eprintln!("files_with_diagnostics: Failed to allocate string: {}", e);
                continue;
            }
        };
        let file_root_id = runtime.register_temporary_root(file_str);
        let vec_updated = runtime.peek_temporary_root(vec_root_id);
        let file_str_updated = runtime.peek_temporary_root(file_root_id);

        let vec_handle = GcHandle::from_tagged(vec_updated);
        match PersistentVec::push(runtime, stack_pointer, vec_handle, file_str_updated) {
            Ok(new_vec) => {
                vec = new_vec.as_tagged();
            }
            Err(e) => {
                eprintln!("files_with_diagnostics: Failed to push file: {}", e);
            }
        }

        runtime.unregister_temporary_root(file_root_id);
        runtime.unregister_temporary_root(vec_root_id);
        vec_root_id = runtime.register_temporary_root(vec);
    }

    runtime.unregister_temporary_root(vec_root_id);
    vec
}

/// Clears all diagnostics
pub unsafe extern "C" fn clear_diagnostics(_stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if let Ok(mut store) = runtime.diagnostic_store.lock() {
        store.clear_all();
    }

    BuiltInTypes::null_value() as usize
}

// ============================================================================
// Rust Collections Builtins
// ============================================================================

mod rust_collections {
    use super::*;
    use crate::collections::{GcHandle, PersistentMap, PersistentSet, PersistentVec};

    /// Create an empty persistent vector
    /// Signature: (stack_pointer, frame_pointer) -> tagged_ptr
    pub unsafe extern "C" fn rust_vec_empty(stack_pointer: usize, frame_pointer: usize) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        match PersistentVec::empty(runtime, stack_pointer) {
            Ok(handle) => handle.as_tagged(),
            Err(e) => {
                eprintln!("rust_vec_empty error: {}", e);
                BuiltInTypes::null_value() as usize
            }
        }
    }

    /// Get the count of a persistent vector
    /// Signature: (vec_ptr) -> tagged_int
    pub unsafe extern "C" fn rust_vec_count(vec_ptr: usize) -> usize {
        if !BuiltInTypes::is_heap_pointer(vec_ptr) {
            return BuiltInTypes::construct_int(0) as usize;
        }
        let vec = GcHandle::from_tagged(vec_ptr);
        let count = PersistentVec::count(vec);
        BuiltInTypes::construct_int(count as isize) as usize
    }

    /// Get a value from a persistent vector by index
    /// Signature: (vec_ptr, index) -> tagged_value
    pub unsafe extern "C" fn rust_vec_get(vec_ptr: usize, index: usize) -> usize {
        if !BuiltInTypes::is_heap_pointer(vec_ptr) {
            return BuiltInTypes::null_value() as usize;
        }
        let vec = GcHandle::from_tagged(vec_ptr);
        let idx = BuiltInTypes::untag(index);
        PersistentVec::get(vec, idx)
    }

    /// Push a value onto a persistent vector
    /// Signature: (stack_pointer, frame_pointer, vec_ptr, value) -> tagged_ptr
    pub unsafe extern "C" fn rust_vec_push(
        stack_pointer: usize,
        frame_pointer: usize,
        vec_ptr: usize,
        value: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        if !BuiltInTypes::is_heap_pointer(vec_ptr) {
            return BuiltInTypes::null_value() as usize;
        }

        let vec = GcHandle::from_tagged(vec_ptr);

        match PersistentVec::push(runtime, stack_pointer, vec, value) {
            Ok(handle) => handle.as_tagged(),
            Err(e) => {
                eprintln!("rust_vec_push error: {}", e);
                BuiltInTypes::null_value() as usize
            }
        }
    }

    /// Update a value at an index in a persistent vector
    /// Signature: (stack_pointer, frame_pointer, vec_ptr, index, value) -> tagged_ptr
    pub unsafe extern "C" fn rust_vec_assoc(
        stack_pointer: usize,
        frame_pointer: usize,
        vec_ptr: usize,
        index: usize,
        value: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        if !BuiltInTypes::is_heap_pointer(vec_ptr) {
            return BuiltInTypes::null_value() as usize;
        }

        let vec = GcHandle::from_tagged(vec_ptr);
        let idx = BuiltInTypes::untag(index);

        match PersistentVec::assoc(runtime, stack_pointer, vec, idx, value) {
            Ok(handle) => handle.as_tagged(),
            Err(e) => {
                eprintln!("rust_vec_assoc error: {}", e);
                BuiltInTypes::null_value() as usize
            }
        }
    }

    // ========== Map builtins ==========

    /// Create an empty persistent map
    /// Signature: (stack_pointer, frame_pointer) -> tagged_ptr
    pub unsafe extern "C" fn rust_map_empty(stack_pointer: usize, frame_pointer: usize) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        // Stay c_calling - HandleScope::allocate checks is_paused and participates in GC

        match PersistentMap::empty(runtime, stack_pointer) {
            Ok(handle) => handle.as_tagged(),
            Err(e) => {
                eprintln!("rust_map_empty error: {}", e);
                BuiltInTypes::null_value() as usize
            }
        }
    }

    /// Get the count of a persistent map
    /// Signature: (map_ptr) -> tagged_int
    pub unsafe extern "C" fn rust_map_count(map_ptr: usize) -> usize {
        if !BuiltInTypes::is_heap_pointer(map_ptr) {
            return BuiltInTypes::construct_int(0) as usize;
        }
        let map = GcHandle::from_tagged(map_ptr);
        let count = PersistentMap::count(map);
        BuiltInTypes::construct_int(count as isize) as usize
    }

    /// Get a value from a persistent map by key
    /// Signature: (map_ptr, key) -> tagged_value
    pub unsafe extern "C" fn rust_map_get(map_ptr: usize, key: usize) -> usize {
        if !BuiltInTypes::is_heap_pointer(map_ptr) {
            return BuiltInTypes::null_value() as usize;
        }
        let runtime = get_runtime().get();
        let map = GcHandle::from_tagged(map_ptr);
        PersistentMap::get(runtime, map, key)
    }

    /// Associate a key-value pair in a persistent map
    /// Signature: (stack_pointer, frame_pointer, map_ptr, key, value) -> tagged_ptr
    pub unsafe extern "C" fn rust_map_assoc(
        stack_pointer: usize,
        frame_pointer: usize,
        map_ptr: usize,
        key: usize,
        value: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        if !BuiltInTypes::is_heap_pointer(map_ptr) {
            return BuiltInTypes::null_value() as usize;
        }

        // Stay c_calling - HandleScope::allocate checks is_paused and participates in GC

        match PersistentMap::assoc(runtime, stack_pointer, map_ptr, key, value) {
            Ok(handle) => handle.as_tagged(),
            Err(e) => {
                eprintln!("rust_map_assoc error: {}", e);
                BuiltInTypes::null_value() as usize
            }
        }
    }

    /// Get all keys from a persistent map as a vector
    // Signature: (stack_pointer, frame_pointer, map_ptr) -> tagged_ptr (vector)
    pub unsafe extern "C" fn rust_map_keys(
        stack_pointer: usize,
        frame_pointer: usize,
        map_ptr: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        if !BuiltInTypes::is_heap_pointer(map_ptr) {
            // Empty map or invalid - return empty vector
            match PersistentVec::empty(runtime, stack_pointer) {
                Ok(handle) => return handle.as_tagged(),
                Err(e) => {
                    eprintln!("rust_map_keys empty vec error: {}", e);
                    return BuiltInTypes::null_value() as usize;
                }
            }
        }

        let map = GcHandle::from_tagged(map_ptr);

        // Stay c_calling - HandleScope::allocate checks is_paused and participates in GC

        match PersistentMap::keys(runtime, stack_pointer, map) {
            Ok(handle) => handle.as_tagged(),
            Err(e) => {
                eprintln!("rust_map_keys error: {}", e);
                BuiltInTypes::null_value() as usize
            }
        }
    }

    // ========== Set builtins ==========

    /// Create an empty persistent set
    /// Signature: (stack_pointer, frame_pointer) -> tagged_ptr
    pub unsafe extern "C" fn rust_set_empty(stack_pointer: usize, frame_pointer: usize) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        match PersistentSet::empty(runtime, stack_pointer) {
            Ok(handle) => handle.as_tagged(),
            Err(e) => {
                eprintln!("rust_set_empty error: {}", e);
                BuiltInTypes::null_value() as usize
            }
        }
    }

    /// Get the count of a persistent set
    /// Signature: (set_ptr) -> tagged_int
    pub unsafe extern "C" fn rust_set_count(set_ptr: usize) -> usize {
        if !BuiltInTypes::is_heap_pointer(set_ptr) {
            return BuiltInTypes::construct_int(0) as usize;
        }
        let set = GcHandle::from_tagged(set_ptr);
        let count = PersistentSet::count(set);
        BuiltInTypes::construct_int(count as isize) as usize
    }

    /// Check if an element is in a persistent set
    /// Signature: (set_ptr, element) -> tagged_bool
    pub unsafe extern "C" fn rust_set_contains(set_ptr: usize, element: usize) -> usize {
        if !BuiltInTypes::is_heap_pointer(set_ptr) {
            return BuiltInTypes::false_value() as usize;
        }
        let runtime = get_runtime().get();
        let set = GcHandle::from_tagged(set_ptr);
        if PersistentSet::contains(runtime, set, element) {
            BuiltInTypes::true_value() as usize
        } else {
            BuiltInTypes::false_value() as usize
        }
    }

    /// Add an element to a persistent set
    /// Signature: (stack_pointer, frame_pointer, set_ptr, element) -> tagged_ptr
    pub unsafe extern "C" fn rust_set_add(
        stack_pointer: usize,
        frame_pointer: usize,
        set_ptr: usize,
        element: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        if !BuiltInTypes::is_heap_pointer(set_ptr) {
            return BuiltInTypes::null_value() as usize;
        }

        match PersistentSet::add(runtime, stack_pointer, set_ptr, element) {
            Ok(handle) => handle.as_tagged(),
            Err(e) => {
                eprintln!("rust_set_add error: {}", e);
                BuiltInTypes::null_value() as usize
            }
        }
    }

    /// Get all elements from a persistent set as a vector
    /// Signature: (stack_pointer, frame_pointer, set_ptr) -> tagged_ptr (vector)
    pub unsafe extern "C" fn rust_set_elements(
        stack_pointer: usize,
        frame_pointer: usize,
        set_ptr: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        if !BuiltInTypes::is_heap_pointer(set_ptr) {
            // Empty set or invalid - return empty vector
            match PersistentVec::empty(runtime, stack_pointer) {
                Ok(handle) => return handle.as_tagged(),
                Err(e) => {
                    eprintln!("rust_set_elements empty vec error: {}", e);
                    return BuiltInTypes::null_value() as usize;
                }
            }
        }

        let set = GcHandle::from_tagged(set_ptr);

        match PersistentSet::elements(runtime, stack_pointer, set) {
            Ok(handle) => handle.as_tagged(),
            Err(e) => {
                eprintln!("rust_set_elements error: {}", e);
                BuiltInTypes::null_value() as usize
            }
        }
    }
}

impl Runtime {
    pub fn install_rust_collection_builtins(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use rust_collections::*;

        // Register the namespace so it can be imported
        self.reserve_namespace("beagle.collections".to_string());

        // rust-vec: Create empty vector (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/vec",
            rust_vec_empty as *const u8,
            true,
            true,
            2, // sp, fp (implicit to caller)
        )?;

        // rust-vec-count: Get count (no allocation needed)
        self.add_builtin_function(
            "beagle.collections/vec-count",
            rust_vec_count as *const u8,
            false,
            1, // vec
        )?;

        // rust-vec-get: Get by index (no allocation needed)
        self.add_builtin_function(
            "beagle.collections/vec-get",
            rust_vec_get as *const u8,
            false,
            2, // vec, index
        )?;

        // rust-vec-push: Push value (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/vec-push",
            rust_vec_push as *const u8,
            true,
            true,
            4, // sp, fp, vec, value
        )?;

        // rust-vec-assoc: Update at index (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/vec-assoc",
            rust_vec_assoc as *const u8,
            true,
            true,
            5, // sp, fp, vec, index, value
        )?;

        // ========== Map builtins ==========

        // rust-map: Create empty map (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/map",
            rust_map_empty as *const u8,
            true,
            true,
            2, // sp, fp
        )?;

        // rust-map-count: Get count (no allocation needed)
        self.add_builtin_function(
            "beagle.collections/map-count",
            rust_map_count as *const u8,
            false,
            1, // map
        )?;

        // rust-map-get: Get by key (no allocation needed)
        self.add_builtin_function(
            "beagle.collections/map-get",
            rust_map_get as *const u8,
            false,
            2, // map, key
        )?;

        // rust-map-assoc: Associate key-value (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/map-assoc",
            rust_map_assoc as *const u8,
            true,
            true,
            5, // sp, fp, map, key, value
        )?;

        // rust-map-keys: Get all keys as vector (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/map-keys",
            rust_map_keys as *const u8,
            true,
            true,
            3, // sp, fp, map
        )?;

        // ========== Set builtins ==========

        // rust-set: Create empty set (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/set",
            rust_set_empty as *const u8,
            true,
            true,
            2, // sp, fp
        )?;

        // rust-set-count: Get count (no allocation needed)
        self.add_builtin_function(
            "beagle.collections/set-count",
            rust_set_count as *const u8,
            false,
            1, // set
        )?;

        // rust-set-contains?: Check if element is in set (no allocation needed)
        self.add_builtin_function(
            "beagle.collections/set-contains?",
            rust_set_contains as *const u8,
            false,
            2, // set, element
        )?;

        // rust-set-add: Add element to set (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/set-add",
            rust_set_add as *const u8,
            true,
            true,
            4, // sp, fp, set, element
        )?;

        // rust-set-elements: Get all elements as vector (needs sp/fp for allocation)
        self.add_builtin_function_with_fp(
            "beagle.collections/set-elements",
            rust_set_elements as *const u8,
            true,
            true,
            3, // sp, fp, set
        )?;

        Ok(())
    }

    pub fn install_regex_builtins(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use regex_builtins::*;

        // Register the namespace so it can be imported
        self.reserve_namespace("beagle.regex".to_string());

        // regex/compile: Compile a regex pattern
        // Signature: (stack_pointer, frame_pointer, pattern_string) -> regex_handle
        self.add_builtin_function_with_fp(
            "beagle.regex/compile",
            regex_compile as *const u8,
            true,
            true,
            3, // sp, fp, pattern
        )?;

        // regex/matches?: Check if string matches regex
        // Signature: (regex, string) -> bool
        self.add_builtin_function(
            "beagle.regex/matches?",
            regex_matches as *const u8,
            false,
            2, // regex, string
        )?;

        // regex/find: Find first match in string
        // Signature: (stack_pointer, frame_pointer, regex, string) -> match_info or null
        self.add_builtin_function_with_fp(
            "beagle.regex/find",
            regex_find as *const u8,
            true,
            true,
            4, // sp, fp, regex, string
        )?;

        // regex/find-all: Find all matches in string
        // Signature: (stack_pointer, frame_pointer, regex, string) -> vector of matches
        self.add_builtin_function_with_fp(
            "beagle.regex/find-all",
            regex_find_all as *const u8,
            true,
            true,
            4, // sp, fp, regex, string
        )?;

        // regex/replace: Replace first match
        // Signature: (stack_pointer, frame_pointer, regex, string, replacement) -> new_string
        self.add_builtin_function_with_fp(
            "beagle.regex/replace",
            regex_replace as *const u8,
            true,
            true,
            5, // sp, fp, regex, string, replacement
        )?;

        // regex/replace-all: Replace all matches
        // Signature: (stack_pointer, frame_pointer, regex, string, replacement) -> new_string
        self.add_builtin_function_with_fp(
            "beagle.regex/replace-all",
            regex_replace_all as *const u8,
            true,
            true,
            5, // sp, fp, regex, string, replacement
        )?;

        // regex/split: Split string by regex
        // Signature: (stack_pointer, frame_pointer, regex, string) -> vector of strings
        self.add_builtin_function_with_fp(
            "beagle.regex/split",
            regex_split as *const u8,
            true,
            true,
            4, // sp, fp, regex, string
        )?;

        // regex/captures: Get capture groups from first match
        // Signature: (stack_pointer, frame_pointer, regex, string) -> vector of strings or null
        self.add_builtin_function_with_fp(
            "beagle.regex/captures",
            regex_captures as *const u8,
            true,
            true,
            4, // sp, fp, regex, string
        )?;

        // regex/is-regex?: Check if value is a regex
        // Signature: (value) -> bool
        self.add_builtin_function(
            "beagle.regex/is-regex?",
            is_regex as *const u8,
            false,
            1, // value
        )?;

        Ok(())
    }
}

mod regex_builtins {
    use super::*;
    use crate::collections::TYPE_ID_REGEX;
    use regex::Regex;

    /// Compile a regex pattern
    /// Signature: (stack_pointer, frame_pointer, pattern) -> regex_handle
    pub unsafe extern "C" fn regex_compile(
        stack_pointer: usize,
        frame_pointer: usize,
        pattern: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        // Get the pattern string
        let pattern_str = runtime.get_string(stack_pointer, pattern);

        // Try to compile the regex
        match Regex::new(&pattern_str) {
            Ok(regex) => {
                // Store the compiled regex and get its index
                let index = runtime.compiled_regexes.len();
                runtime.compiled_regexes.push(regex);

                // Allocate a heap object to hold the index
                // The object has 1 field: the index into compiled_regexes
                match runtime.allocate(1, stack_pointer, BuiltInTypes::HeapObject) {
                    Ok(ptr) => {
                        let mut heap_obj = HeapObject::from_tagged(ptr);
                        heap_obj.write_type_id(TYPE_ID_REGEX as usize);
                        heap_obj.write_field(0, BuiltInTypes::Int.tag(index as isize) as usize);
                        ptr
                    }
                    Err(e) => {
                        eprintln!("regex_compile allocation error: {}", e);
                        BuiltInTypes::null_value() as usize
                    }
                }
            }
            Err(e) => {
                // Return null on invalid regex (could also throw)
                eprintln!("regex compile error: {}", e);
                BuiltInTypes::null_value() as usize
            }
        }
    }

    /// Get the regex from a heap object
    fn get_regex(runtime: &Runtime, regex_ptr: usize) -> Option<&Regex> {
        if !BuiltInTypes::is_heap_pointer(regex_ptr) {
            return None;
        }
        let heap_obj = HeapObject::from_tagged(regex_ptr);
        if heap_obj.get_type_id() != TYPE_ID_REGEX as usize {
            return None;
        }
        let index = BuiltInTypes::untag(heap_obj.get_field(0));
        runtime.compiled_regexes.get(index)
    }

    /// Check if string matches regex
    /// Signature: (regex, string) -> bool
    pub unsafe extern "C" fn regex_matches(regex_ptr: usize, string: usize) -> usize {
        let runtime = get_runtime().get();

        let Some(regex) = get_regex(runtime, regex_ptr) else {
            return BuiltInTypes::false_value() as usize;
        };

        // Get the string - but we can't call get_string without stack_pointer
        // So we need to handle this differently
        let string_str = get_string_for_regex(runtime, string);
        let result = regex.is_match(&string_str);
        BuiltInTypes::construct_boolean(result) as usize
    }

    /// Helper to get a string without throwing (for non-allocating functions)
    fn get_string_for_regex(runtime: &Runtime, value: usize) -> String {
        let tag = BuiltInTypes::get_kind(value);
        if tag == BuiltInTypes::String {
            runtime.get_string_literal(value)
        } else if tag == BuiltInTypes::HeapObject {
            let heap_object = HeapObject::from_tagged(value);
            if heap_object.get_type_id() != 2 {
                return String::new();
            }
            let bytes = heap_object.get_string_bytes();
            unsafe { std::str::from_utf8_unchecked(bytes).to_string() }
        } else {
            String::new()
        }
    }

    /// Find first match in string
    /// Signature: (stack_pointer, frame_pointer, regex, string) -> match map or null
    pub unsafe extern "C" fn regex_find(
        stack_pointer: usize,
        frame_pointer: usize,
        regex_ptr: usize,
        string: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        let Some(regex) = get_regex(runtime, regex_ptr) else {
            return BuiltInTypes::null_value() as usize;
        };

        let string_str = runtime.get_string(stack_pointer, string);

        match regex.find(&string_str) {
            Some(m) => {
                // Return the matched string
                let matched = m.as_str().to_string();
                match runtime.allocate_string(stack_pointer, matched) {
                    Ok(ptr) => ptr.into(),
                    Err(_) => BuiltInTypes::null_value() as usize,
                }
            }
            None => BuiltInTypes::null_value() as usize,
        }
    }

    /// Find all matches in string
    /// Signature: (stack_pointer, frame_pointer, regex, string) -> vector of strings
    pub unsafe extern "C" fn regex_find_all(
        stack_pointer: usize,
        frame_pointer: usize,
        regex_ptr: usize,
        string: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        let Some(regex) = get_regex(runtime, regex_ptr) else {
            return BuiltInTypes::null_value() as usize;
        };

        let string_str = runtime.get_string(stack_pointer, string);

        // Collect all matches
        let matches: Vec<String> = regex
            .find_iter(&string_str)
            .map(|m| m.as_str().to_string())
            .collect();

        // Create a PersistentVec with all matches
        use crate::collections::PersistentVec;

        let mut vec = match PersistentVec::empty(runtime, stack_pointer) {
            Ok(v) => v,
            Err(_) => return BuiltInTypes::null_value() as usize,
        };

        for matched in matches {
            let str_ptr = match runtime.allocate_string(stack_pointer, matched) {
                Ok(ptr) => ptr.into(),
                Err(_) => return BuiltInTypes::null_value() as usize,
            };
            vec = match PersistentVec::push(runtime, stack_pointer, vec, str_ptr) {
                Ok(v) => v,
                Err(_) => return BuiltInTypes::null_value() as usize,
            };
        }

        vec.as_tagged()
    }

    /// Replace first match
    /// Signature: (stack_pointer, frame_pointer, regex, string, replacement) -> new_string
    pub unsafe extern "C" fn regex_replace(
        stack_pointer: usize,
        frame_pointer: usize,
        regex_ptr: usize,
        string: usize,
        replacement: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        let Some(regex) = get_regex(runtime, regex_ptr) else {
            return string; // Return original string if not a valid regex
        };

        let string_str = runtime.get_string(stack_pointer, string);
        let replacement_str = runtime.get_string(stack_pointer, replacement);

        let result = regex
            .replace(&string_str, replacement_str.as_str())
            .to_string();

        match runtime.allocate_string(stack_pointer, result) {
            Ok(ptr) => ptr.into(),
            Err(_) => BuiltInTypes::null_value() as usize,
        }
    }

    /// Replace all matches
    /// Signature: (stack_pointer, frame_pointer, regex, string, replacement) -> new_string
    pub unsafe extern "C" fn regex_replace_all(
        stack_pointer: usize,
        frame_pointer: usize,
        regex_ptr: usize,
        string: usize,
        replacement: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        let Some(regex) = get_regex(runtime, regex_ptr) else {
            return string; // Return original string if not a valid regex
        };

        let string_str = runtime.get_string(stack_pointer, string);
        let replacement_str = runtime.get_string(stack_pointer, replacement);

        let result = regex
            .replace_all(&string_str, replacement_str.as_str())
            .to_string();

        match runtime.allocate_string(stack_pointer, result) {
            Ok(ptr) => ptr.into(),
            Err(_) => BuiltInTypes::null_value() as usize,
        }
    }

    /// Split string by regex
    /// Signature: (stack_pointer, frame_pointer, regex, string) -> vector of strings
    pub unsafe extern "C" fn regex_split(
        stack_pointer: usize,
        frame_pointer: usize,
        regex_ptr: usize,
        string: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        let Some(regex) = get_regex(runtime, regex_ptr) else {
            return BuiltInTypes::null_value() as usize;
        };

        let string_str = runtime.get_string(stack_pointer, string);

        // Split the string
        let parts: Vec<&str> = regex.split(&string_str).collect();

        // Create a PersistentVec with all parts
        use crate::collections::PersistentVec;

        let mut vec = match PersistentVec::empty(runtime, stack_pointer) {
            Ok(v) => v,
            Err(_) => return BuiltInTypes::null_value() as usize,
        };

        for part in parts {
            let str_ptr = match runtime.allocate_string(stack_pointer, part.to_string()) {
                Ok(ptr) => ptr.into(),
                Err(_) => return BuiltInTypes::null_value() as usize,
            };
            vec = match PersistentVec::push(runtime, stack_pointer, vec, str_ptr) {
                Ok(v) => v,
                Err(_) => return BuiltInTypes::null_value() as usize,
            };
        }

        vec.as_tagged()
    }

    /// Get capture groups from first match
    /// Signature: (stack_pointer, frame_pointer, regex, string) -> vector of strings or null
    pub unsafe extern "C" fn regex_captures(
        stack_pointer: usize,
        frame_pointer: usize,
        regex_ptr: usize,
        string: usize,
    ) -> usize {
        save_gc_context!(stack_pointer, frame_pointer);
        let runtime = get_runtime().get_mut();

        let Some(regex) = get_regex(runtime, regex_ptr) else {
            return BuiltInTypes::null_value() as usize;
        };

        let string_str = runtime.get_string(stack_pointer, string);

        match regex.captures(&string_str) {
            Some(caps) => {
                use crate::collections::PersistentVec;

                let mut vec = match PersistentVec::empty(runtime, stack_pointer) {
                    Ok(v) => v,
                    Err(_) => return BuiltInTypes::null_value() as usize,
                };

                for i in 0..caps.len() {
                    let value = match caps.get(i) {
                        Some(m) => {
                            match runtime.allocate_string(stack_pointer, m.as_str().to_string()) {
                                Ok(ptr) => ptr.into(),
                                Err(_) => return BuiltInTypes::null_value() as usize,
                            }
                        }
                        None => BuiltInTypes::null_value() as usize,
                    };
                    vec = match PersistentVec::push(runtime, stack_pointer, vec, value) {
                        Ok(v) => v,
                        Err(_) => return BuiltInTypes::null_value() as usize,
                    };
                }

                vec.as_tagged()
            }
            None => BuiltInTypes::null_value() as usize,
        }
    }

    /// Check if value is a regex
    /// Signature: (value) -> bool
    pub unsafe extern "C" fn is_regex(value: usize) -> usize {
        if !BuiltInTypes::is_heap_pointer(value) {
            return BuiltInTypes::false_value() as usize;
        }
        let heap_obj = HeapObject::from_tagged(value);
        let is_regex = heap_obj.get_type_id() == TYPE_ID_REGEX as usize;
        BuiltInTypes::construct_boolean(is_regex) as usize
    }
}
