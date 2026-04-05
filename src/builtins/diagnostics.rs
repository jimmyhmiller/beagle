use super::*;
use crate::save_gc_context;

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
pub unsafe fn allocate_struct(
    runtime: &mut Runtime,
    stack_pointer: usize,
    struct_name: &str,
    fields: &[usize],
) -> Result<usize, String> {
    // Look up struct definition from registry
    let (struct_id, struct_def) = runtime
        .get_struct(struct_name)
        .ok_or_else(|| format!("Struct {} not found", struct_name))?;

    // struct_id is stored as raw value in header (not tagged)

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
pub unsafe fn diagnostic_to_struct(
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
pub unsafe fn diagnostic_to_struct_impl(
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
    let missing_variants_root_idx = if let Some(ref missing_variants) = diagnostic.missing_variants
    {
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

        // Allocate the struct - this can trigger GC!
        let obj_ptr = runtime
            .allocate(8, stack_pointer, BuiltInTypes::HeapObject)
            .map_err(|e| format!("Allocation failed: {}", e))?;

        // Write struct_id to header (as raw value, not tagged)
        let heap_obj = HeapObject::from_tagged(obj_ptr);
        let untagged = heap_obj.untagged();
        let header_ptr = untagged as *mut usize;
        unsafe {
            let current_header = *header_ptr;
            let mask = 0x00FFFFFFFF000000;
            let shifted_struct_id = struct_id << 24;
            let new_header = (current_header & !mask) | shifted_struct_id;
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
        let tid = heap_object.get_type_id();
        if tid != TYPE_ID_STRING as usize
            && tid != TYPE_ID_STRING_SLICE as usize
            && tid != TYPE_ID_CONS_STRING as usize
        {
            eprintln!("diagnostics_for_file: Invalid file path argument (not a string)");
            return BuiltInTypes::null_value() as usize;
        }
        let bytes = runtime.get_string_bytes_vec(file_path);
        unsafe { String::from_utf8_unchecked(bytes) }
    };

    // Clone diagnostics for the specific file
    let file_diagnostics: Vec<crate::compiler::Diagnostic> = {
        let store_guard = runtime.diagnostic_store.lock().unwrap();
        store_guard
            .for_file(&file_path_str)
            .cloned()
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
            eprintln!(
                "files_with_diagnostics: Failed to create empty vector: {}",
                e
            );
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
pub unsafe extern "C" fn clear_diagnostics(stack_pointer: usize, frame_pointer: usize) -> usize {
    save_gc_context!(stack_pointer, frame_pointer);
    let runtime = get_runtime().get_mut();

    if let Ok(mut store) = runtime.diagnostic_store.lock() {
        store.clear_all();
    }

    BuiltInTypes::null_value() as usize
}
