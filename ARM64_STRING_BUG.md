# ARM64 String Literal Bug

## Symptom
On ARM64 macOS, certain string literals become `null` when used in println statements.

## Minimal Reproduction
```bash
cargo build --features backend-arm64
./target/debug/main resources/arm64_string_bug.bg
```

**Expected output:**
```
zip: test1
interleave: test2
interpose: test3
```

**Actual output on ARM64 macOS:**
```
zip: test1
null test2
interpose: test3
```

The string `"interleave:"` prints as `null`, while `"zip:"` and `"interpose:"` work fine.

## Analysis

### Root Cause
Bug in ARM64 ADR (Address Register) instruction encoding at:
- `src/machine_code/arm_codegen.rs` lines 748-755
- `src/arm.rs` lines 1118-1139 (patching logic)

### The Issue
The ADR instruction uses a 21-bit signed immediate split across:
- `immlo`: bits [30:29] (2 bits)
- `immhi`: bits [23:5] (19 bits)

When patching label addresses (`src/arm.rs:1132-1133`):
```rust
*immlo = byte_offset & 0x3;           // Lower 2 bits
*immhi = (byte_offset >> 2) & 0x7FFFF; // Upper 19 bits
```

When encoding (`src/machine_code/arm_codegen.rs:752-753`):
```rust
| (((*immlo as u32) & 0x3) << 29)
| (((*immhi as u32) & 0x7FFFF) << 5)
```

**Problem**: For certain negative offsets (when string literal is before the ADR instruction), the sign extension or bit manipulation causes the address to become invalid, resulting in null.

### Why Only Specific Strings Fail
- Most strings work fine
- `"interleave:"` specifically fails
- Likely related to:
  - The specific offset distance from the ADR instruction to the string data
  - Certain offset ranges where sign extension causes issues
  - Possibly edge cases around page boundaries or alignment

### Attempted Fix
Added masking in `arm_codegen.rs` to prevent sign extension:
```rust
| (((*immlo as u32) & 0x3) << 29)
| (((*immhi as u32) & 0x7FFFF) << 5)
```

This did NOT fix the issue - the problem may be deeper in the patching logic or how string constants are laid out in memory.

## Next Steps
1. Run the minimal reproducer on ARM64 Mac
2. Debug the actual instruction encoding and offset values
3. Compare working strings (like `"zip:"`) with failing string (`"interleave:"`)
4. Check if the issue is in:
   - ADR offset calculation
   - String constant placement in memory
   - Alignment issues
   - Sign extension in patching or encoding

## Workaround
Avoid using string literals with colons directly in println:
```rust
// Instead of:
println("interleave:", value)

// Use:
let result = value
println("interleave", result)
```
