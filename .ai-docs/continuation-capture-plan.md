I didn't go with this plan. It is overly structured and fairly complicated. Instead, I reused the way in which we scan multiple stacks for multiple threads. AI produces documents like this that would convince someone so much better than I would of the implementation.


# Detailed Plan for Continuation Capture Implementation

## 1. Continuation Object Design

### A. Heap Object Structure
Create a new heap object type for continuations in `src/types.rs`:

```rust
// New heap object type ID for continuations  
pub const CONTINUATION_TYPE_ID: u32 = 100;

#[repr(C)]
pub struct ContinuationHeader {
    pub frame_count: u32,           // Number of captured frames
    pub total_stack_size: u32,      // Total bytes of stack data
    pub yield_return_address: u64,  // Where to resume execution
    pub yield_frame_pointer: u64,   // X29 at yield point
    pub yield_stack_pointer: u64,   // SP at yield point
}

// Heap layout:
// [HeapObject header] [ContinuationHeader] [FrameInfo array] [Stack data]
```

### B. Frame Metadata Structure
```rust
#[repr(C)]
pub struct FrameInfo {
    pub frame_size: u32,            // Size of this frame in bytes
    pub locals_count: u32,          // Number of local variables
    pub stack_size: u32,            // Current stack usage in frame
    pub return_address: u64,        // Saved X30 for this frame
    pub previous_frame_offset: u32, // Offset to previous frame data
}
```

## 2. Stack Capture Implementation

### A. Modify yield_continuation Function
Replace the current `compile_yield_continuation` in `src/main.rs` with a capture-based approach:

```rust
fn compile_capture_continuation(runtime: &mut Runtime) {
    let mut lang = LowLevelArm::new();
    lang.prelude();
    
    // Parameters: x0 = stack_pointer, x1 = yielded_value
    // 1. Save yielded value and current state
    // 2. Walk stack to calculate total capture size
    // 3. Allocate heap memory for continuation object
    // 4. Copy stack frames to heap
    // 5. Create continuation object with metadata
    // 6. Call handler with (yielded_value, continuation_object)
}
```

### B. Stack Walking and Size Calculation
Implement in `src/builtins.rs`:

```rust
pub unsafe extern "C" fn calculate_continuation_size(
    stack_pointer: usize
) -> (usize, Vec<FrameInfo>) {
    let mut total_size = 0;
    let mut frame_infos = Vec::new();
    
    // Walk stack frames using existing stack walker logic
    // For each frame from current to delimit boundary:
    //   - Calculate frame size (locals + stack usage + padding)
    //   - Record frame metadata
    //   - Accumulate total size
    
    (total_size, frame_infos)
}
```

### C. Stack Frame Copying
```rust
pub unsafe extern "C" fn copy_stack_to_continuation(
    heap_object: usize,
    stack_pointer: usize,
    frame_infos: &[FrameInfo]
) {
    let mut continuation = HeapObject::from_tagged(heap_object);
    let stack_data_section = continuation.get_raw_data_mut();
    
    // Copy each frame's data to heap:
    //   - Copy locals and stack variables
    //   - Preserve relative offsets within frames
    //   - Update any stack-relative pointers to be heap-relative
}
```

## 3. Handler Integration

### A. Modify Delimit/Handle Compilation
In `src/compiler.rs`, update delimit compilation to:

```rust
// Instead of setting continuation marker to handler address,
// set marker to a special "capture handler" function address
// The capture handler will:
//   1. Receive yielded value and captured continuation
//   2. Call user-defined handler with both parameters
//   3. Handle the return value appropriately
```

### B. New Handler Signature
Handlers now receive two parameters:
- `value`: The yielded value  
- `resume`: The captured continuation function

Example Beagle code:
```beagle
delimit {
    yield("test")
} handle (value, resume) {
    println("Caught:", value)
    resume()  // Can call the continuation
}
```

## 4. Continuation Restoration

### A. Continuation Call Implementation
Create a new builtin function:

```rust
pub unsafe extern "C" fn call_continuation(
    continuation_object: usize,
    return_value: usize  // Value to return from yield
) -> usize {
    // 1. Extract continuation metadata from heap object
    // 2. Allocate stack space for restoration  
    // 3. Copy frames from heap back to stack
    // 4. Restore frame pointers and return addresses
    // 5. Set return value in appropriate register
    // 6. Jump back to yield point
}
```

### B. Assembly Restoration Code
```rust
fn compile_restore_continuation(runtime: &mut Runtime) {
    let mut lang = LowLevelArm::new();
    
    // Parameters: x0 = continuation_object, x1 = return_value
    // 1. Parse continuation object structure
    // 2. Calculate new stack positions
    // 3. Copy frame data from heap to stack
    // 4. Restore X29 chain (frame pointers)
    // 5. Set up return addresses
    // 6. Place return value in X0
    // 7. Jump to saved yield return address
}
```

## 5. Memory Management and GC Integration

### A. Continuation Object Allocation
Add to `src/runtime.rs`:

```rust
impl Runtime {
    pub fn allocate_continuation(
        &mut self,
        stack_pointer: usize,
        frame_count: usize,
        total_stack_size: usize
    ) -> Result<usize, Box<dyn Error>> {
        let object_size = size_of::<ContinuationHeader>() 
            + frame_count * size_of::<FrameInfo>()
            + total_stack_size;
            
        let heap_object = self.allocate(
            object_size, 
            stack_pointer, 
            BuiltInTypes::HeapObject
        )?;
        
        // Initialize with continuation type ID
        let mut obj = HeapObject::from_tagged(heap_object);
        obj.set_type_id(CONTINUATION_TYPE_ID);
        
        Ok(heap_object)
    }
}
```

### B. GC Scanning for Continuations
Update `src/gc/` to handle continuation objects:

```rust
// In GC scan phase, when encountering continuation objects:
fn scan_continuation_object(obj: &HeapObject, callback: &mut dyn FnMut(usize)) {
    let header = obj.get_continuation_header();
    let stack_data = obj.get_continuation_stack_data();
    
    // Scan the copied stack data for heap pointers
    // This is similar to stack walking but on heap-allocated data
    for word in stack_data {
        if BuiltInTypes::is_heap_pointer(*word) {
            callback(*word);
        }
    }
}
```

### C. Pointer Updating During Compaction
```rust
// Update any heap pointers within captured stack frames
fn update_continuation_pointers(
    obj: &mut HeapObject, 
    old_to_new: &HashMap<usize, usize>
) {
    let stack_data = obj.get_continuation_stack_data_mut();
    
    for word in stack_data {
        if BuiltInTypes::is_heap_pointer(*word) {
            if let Some(&new_addr) = old_to_new.get(&BuiltInTypes::untag(*word)) {
                *word = BuiltInTypes::tag_heap_pointer(new_addr);
            }
        }
    }
}
```

## 6. Implementation Phases

### Phase 1: Basic Capture Infrastructure
1. **Add continuation object type** to type system
2. **Implement stack walking and size calculation** 
3. **Create heap allocation for continuations**
4. **Basic stack copying mechanism**
5. **Update tests to verify capture without restoration**

### Phase 2: Handler Integration  
1. **Modify delimit/handle compilation** to use capture
2. **Update handler calling convention** to pass continuation
3. **Create continuation call builtin function**
4. **Test basic yield/handle with captured continuations**

### Phase 3: Stack Restoration
1. **Implement continuation call mechanism**
2. **Stack restoration assembly code**
3. **Frame pointer and return address reconstruction** 
4. **Test complete capture and restore cycle**

### Phase 4: GC Integration
1. **Continuation object scanning in GC**
2. **Pointer updating during compaction**
3. **Test with all GC implementations** (mark-sweep, compacting, generational)
4. **Performance optimization**

## 7. Testing Strategy

### A. Test Cases
1. **Basic yield and capture**: Verify continuation objects are created
2. **Handler receives continuation**: Test new two-parameter handler signature  
3. **Simple continuation call**: Resume execution after yield
4. **Nested continuations**: Multiple yield/delimit levels
5. **GC interaction**: Continuations survive garbage collection
6. **Stack overflow prevention**: Large captured stacks

### B. Test Files to Create
```beagle
// test_continuation_capture.bg
namespace test_continuation_capture

fn test_basic_capture() {
    delimit {
        let x = 42
        let result = yield("captured")
        println("Resumed with:", result)
        x + result
    } handle (value, resume) {
        println("Handler got:", value)
        resume(100)  // Resume with value 100
    }
}

// test_nested_continuations.bg  
// test_continuation_gc.bg
```

## 8. Performance Considerations

### A. Optimization Opportunities
1. **Lazy copying**: Only copy frames that contain heap pointers
2. **Stack reuse**: Reuse stack space when possible
3. **Frame deduplication**: Share common frame data
4. **Continuation pooling**: Reuse continuation objects

### B. Memory Usage
- **Stack copying overhead**: Each continuation copies stack frames
- **GC pressure**: More heap objects to scan
- **Fragmentation**: Variable-sized continuation objects

## 9. Integration Points

### A. Compiler Changes (`src/compiler.rs`)
- Modify delimit/handle compilation for new signature
- Update continuation marker insertion
- Handle nested delimit scopes

### B. Runtime Changes (`src/runtime.rs`)  
- Add continuation allocation functions
- Register new builtin functions
- Update function resolution for continuations

### C. Type System (`src/types.rs`)
- Add continuation object type
- Define continuation header structures
- Update type checking for handlers

## 10. Current Implementation Analysis

### Current Yield Mechanism
The current yield mechanism in `src/main.rs` (lines 161-232) implements a **stack unwinding approach**:

- **Stack Walking**: Walks up the stack frame by frame from current frame pointer (X29)
- **Marker Detection**: Checks for continuation marker at offset `-16` from each frame pointer
- **Handler Jump**: When finding non-zero marker (handler address), restores stack to that frame and jumps
- **Value Passing**: Yielded value passed in X0 to handler

### Stack Frame Structure
From `prelude` function in `src/arm.rs`:

```
High Memory (Stack Base)
├─ [X29 + 0]     : Saved X29 (previous frame pointer)
├─ [X29 + 8]     : Saved X30 (return address)  
├─ [X29 - 8]     : Zero padding (first word)   ← CONTINUATION_MARKER_PADDING_SIZE
├─ [X29 - 16]    : Zero padding (second word)  ← Continuation marker slot
├─ [X29 - 24]    : Local variable 0
├─ [X29 - 32]    : Local variable 1
├─ ...           : More locals
└─ [SP]          : Current stack pointer
Low Memory
```

### Continuation Marker System
- **Padding Size**: `CONTINUATION_MARKER_PADDING_SIZE = 2` (from `src/ir.rs`)
- **Marker Location**: At `[X29 - 16]` (second zero word in padding)
- **Marker Content**: Raw handler address (untagged) when set, zero otherwise
- **Detection Logic**: `yield_continuation_v2` walks frames and checks this slot

### Limitations of Current System
Current system only supports **escape continuations** (one-shot, stack unwinding):
- No mechanism to **copy stack frames to heap**
- No way to **restore captured stack** later  
- Stack frames are **lost** during yield (can't resume after handler)
- Only supports **single-use continuations**

This comprehensive plan addresses all aspects of implementing true continuation capture: from the low-level stack copying and restoration mechanisms to high-level language integration and garbage collection. The phased approach allows for incremental development and testing at each stage.