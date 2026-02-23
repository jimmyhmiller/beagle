use crate::ir::{Ir, Value};

// I don't know if this is actually the setup I want
// But I want get some stuff down there
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BuiltInTypes {
    Int,
    Float,
    String,
    Bool,
    Function,
    Closure,
    HeapObject,
    Null,
}

impl BuiltInTypes {
    pub fn null_value() -> isize {
        0b111
    }

    pub fn true_value() -> isize {
        Self::construct_boolean(true)
    }

    pub fn false_value() -> isize {
        Self::construct_boolean(false)
    }

    pub fn is_true(value: usize) -> bool {
        value == Self::true_value() as usize
    }

    pub fn tag(&self, value: isize) -> isize {
        let value = value << 3;
        let tag = self.get_tag();
        value | tag
    }

    pub fn tagged(&self, value: usize) -> Tagged {
        if BuiltInTypes::is_heap_pointer(value) {
            Tagged(value)
        } else {
            Tagged(self.tag(value as isize) as usize)
        }
    }

    pub fn get_tag(&self) -> isize {
        match self {
            BuiltInTypes::Int => 0b000,
            BuiltInTypes::Float => 0b001,
            BuiltInTypes::String => 0b010,
            BuiltInTypes::Bool => 0b011,
            BuiltInTypes::Function => 0b100,
            BuiltInTypes::Closure => 0b101,
            BuiltInTypes::HeapObject => 0b110,
            BuiltInTypes::Null => 0b111,
        }
    }

    pub fn untag(value: usize) -> usize {
        value >> 3
    }

    pub fn untag_isize(value: isize) -> isize {
        value >> 3
    }

    pub fn get_kind(pointer: usize) -> Self {
        if pointer == Self::null_value() as usize {
            return BuiltInTypes::Null;
        }
        match pointer & 0b111 {
            0b000 => BuiltInTypes::Int,
            0b001 => BuiltInTypes::Float,
            0b010 => BuiltInTypes::String,
            0b011 => BuiltInTypes::Bool,
            0b100 => BuiltInTypes::Function,
            0b101 => BuiltInTypes::Closure,
            0b110 => BuiltInTypes::HeapObject,
            0b111 => BuiltInTypes::Null,
            _ => unreachable!("All 3-bit patterns are covered"),
        }
    }

    pub fn is_embedded(&self) -> bool {
        match self {
            BuiltInTypes::Int => true,
            BuiltInTypes::Float => true,
            BuiltInTypes::String => false,
            BuiltInTypes::Bool => true,
            BuiltInTypes::Function => false,
            BuiltInTypes::HeapObject => false,
            BuiltInTypes::Closure => false,
            BuiltInTypes::Null => true,
        }
    }

    pub fn construct_int(value: isize) -> isize {
        if value > isize::MAX >> 3 {
            panic!(
                "Integer overflow: {} exceeds maximum tagged integer value {}",
                value,
                isize::MAX >> 3
            )
        }
        BuiltInTypes::Int.tag(value)
    }

    pub fn construct_boolean(value: bool) -> isize {
        let bool = BuiltInTypes::Bool;
        if value { bool.tag(1) } else { bool.tag(0) }
    }

    pub fn tag_size() -> i32 {
        3
    }

    pub fn is_heap_pointer(value: usize) -> bool {
        // With proper FP-chain based stack walking, we only scan actual Beagle
        // locals, not garbage from Rust frames. So we can use the simple check.
        match BuiltInTypes::get_kind(value) {
            BuiltInTypes::Int => false,
            BuiltInTypes::Float | BuiltInTypes::Closure | BuiltInTypes::HeapObject => true,
            BuiltInTypes::String => false,
            BuiltInTypes::Bool => false,
            BuiltInTypes::Function => false,
            BuiltInTypes::Null => false,
        }
    }
}

#[test]
fn tag_and_untag() {
    let kinds = [
        BuiltInTypes::Int,
        BuiltInTypes::Float,
        BuiltInTypes::String,
        BuiltInTypes::Bool,
        BuiltInTypes::Function,
        BuiltInTypes::Closure,
        BuiltInTypes::HeapObject,
    ];
    for kind in kinds.iter() {
        let value = 123;
        let tagged = kind.tag(value);
        // assert_eq!(tagged & 0b111a, tag);
        assert_eq!(kind, &BuiltInTypes::get_kind(tagged as usize));
        assert_eq!(value as usize, BuiltInTypes::untag(tagged as usize));
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Header {
    pub type_id: u8,
    pub type_data: u32,
    pub size: u16, // Size in words. For large objects (>65535), this is 0xFFFF and actual size is in extended header
    pub opaque: bool,
    pub marked: bool,
    pub large: bool, // Large object flag - when set, actual size is stored in word after header
    pub type_flags: u8, // 4 bits (bits 4-7 of byte 0), per-type metadata flags
}

impl Header {
    // | Byte 7  | Bytes 3-6     | Bytes 1-2 | Byte 0                   |
    // |---------|---------------|-----------|--------------------------|
    // | Type    | Type Metadata | Size      | Flag bits                |
    // |         | (4 bytes)     | (2 bytes) | type_flags (bits 7-4)    |
    // |         |               |           | [reserved: fwd bit] (3)  |
    // |         |               |           | Large object (bit 2)     |
    // |         |               |           | Opaque object (bit 1)    |
    // |         |               |           | Marked (bit 0)           |
    //
    // For large objects (size > 65535 words), the large flag is set and the actual
    // size is stored in an 8-byte word immediately following the header.
    //
    // IMPORTANT: Bit 3 is reserved for the GC forwarding pointer marker.
    // During GC, the header word is overwritten with a forwarding tagged pointer
    // that has bit 3 set. No header flag may use bit 3.
    //
    // type_flags (bits 4-7): 4 per-type metadata bits. Each type can use these
    // for its own purposes. E.g., strings use bit 0 (header bit 4) for is_ascii.

    /// Position of the marked bit in the header.
    /// IMPORTANT: This MUST be in the 3 least significant bits (0, 1, or 2) for
    /// GC forwarding to work with 8-byte aligned pointers.
    const MARKED_BIT_POSITION: u32 = 0;

    /// Position of the opaque bit in the header.
    const OPAQUE_BIT_POSITION: u32 = 1;

    /// Position of the large object bit in the header.
    /// When set, the actual size is stored in the word after the header.
    pub const LARGE_OBJECT_BIT_POSITION: u32 = 2;

    // Bit 3 is RESERVED for GC forwarding pointer marker. Do not use.

    /// Maximum size that fits in the inline size field (u16)
    pub const MAX_INLINE_SIZE: usize = 0xFFFF;

    pub fn to_usize(self) -> usize {
        let mut data: usize = 0;
        data |= (self.type_id as usize) << 56;
        data |= (self.type_data as usize) << 24;
        data |= (self.size as usize) << 8; // Size now uses bytes 1-2
        data |= ((self.type_flags as usize) & 0xF) << 4; // type_flags in bits 4-7
        if self.opaque {
            data |= 1 << Self::OPAQUE_BIT_POSITION;
        }
        if self.marked {
            data |= 1 << Self::MARKED_BIT_POSITION;
        }
        if self.large {
            data |= 1 << Self::LARGE_OBJECT_BIT_POSITION;
        }
        data
    }

    pub fn from_usize(data: usize) -> Self {
        let _type = (data >> 56) as u8;
        let type_data = (data >> 24) as u32;
        let size = ((data >> 8) & 0xFFFF) as u16; // Extract 16 bits for size
        let type_flags = ((data >> 4) & 0xF) as u8; // Extract 4 bits for type_flags
        let opaque = (data & (1 << Self::OPAQUE_BIT_POSITION)) != 0;
        let marked = (data & (1 << Self::MARKED_BIT_POSITION)) != 0;
        let large = (data & (1 << Self::LARGE_OBJECT_BIT_POSITION)) != 0;
        Header {
            type_id: _type,
            type_data,
            size,
            opaque,
            marked,
            large,
            type_flags,
        }
    }

    pub fn type_id_offset() -> usize {
        7
    }

    fn type_data_offset() -> usize {
        3
    }

    pub fn size_offset() -> usize {
        1 // Size is now at bytes 1-2 (u16)
    }

    /// Get the bit mask for the marked bit
    pub const fn marked_bit_mask() -> usize {
        1 << Self::MARKED_BIT_POSITION
    }

    /// Set the marked bit in a raw header value, preserving other bits
    pub const fn set_marked_bit(header_value: usize) -> usize {
        header_value | Self::marked_bit_mask()
    }

    /// Clear the marked bit in a raw header value, preserving other bits
    pub const fn clear_marked_bit(header_value: usize) -> usize {
        header_value & !Self::marked_bit_mask()
    }

    /// Check if the marked bit is set in a raw header value
    pub const fn is_marked_bit_set(header_value: usize) -> bool {
        (header_value & Self::marked_bit_mask()) != 0
    }

    // === Large object bit operations ===

    /// Get the bit mask for the large object bit
    pub const fn large_object_bit_mask() -> usize {
        1 << Self::LARGE_OBJECT_BIT_POSITION
    }

    /// Check if the large object bit is set in a raw header value
    pub const fn is_large_object_bit_set(header_value: usize) -> bool {
        (header_value & Self::large_object_bit_mask()) != 0
    }

    // === Forwarding pointer operations ===
    //
    // When an object is forwarded during compacting GC, we store the new tagged
    // pointer in the object's header location. We need a bit to mark that this
    // has happened. We can't use bit 0-2 because those are used by type tags.
    // Bit 3 is the lowest bit of the shifted pointer value (tagged = raw << 3 | tag),
    // and since raw pointers are 8-byte aligned, bit 3 is always 0 in a valid
    // tagged pointer. So we use bit 3 as the forwarding marker.

    /// Position of the forwarding bit for tagged pointers
    const FORWARDING_BIT_POSITION: u32 = 3;

    /// Get the bit mask for the forwarding bit
    pub const fn forwarding_bit_mask() -> usize {
        1 << Self::FORWARDING_BIT_POSITION
    }

    /// Set the forwarding bit in a tagged pointer, preserving other bits
    pub const fn set_forwarding_bit(tagged_pointer: usize) -> usize {
        tagged_pointer | Self::forwarding_bit_mask()
    }

    /// Clear the forwarding bit in a tagged pointer, preserving other bits
    pub const fn clear_forwarding_bit(tagged_pointer: usize) -> usize {
        tagged_pointer & !Self::forwarding_bit_mask()
    }

    /// Check if the forwarding bit is set in a tagged pointer
    pub const fn is_forwarding_bit_set(value: usize) -> bool {
        (value & Self::forwarding_bit_mask()) != 0
    }
}

#[cfg(test)]
mod header_layout_tests {
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_marked_bit_position_compatibility() {
        // This test verifies that our marked bit position is compatible with 8-byte alignment
        assert!(
            Header::MARKED_BIT_POSITION < 3,
            "Marked bit must be in the 3 least significant bits for GC forwarding to work"
        );

        // Test that we can set and clear the marked bit correctly
        let test_value = 0xDEADBEEF_CAFEBABE_usize;

        let marked = Header::set_marked_bit(test_value);
        assert!(Header::is_marked_bit_set(marked));

        let unmarked = Header::clear_marked_bit(marked);
        assert!(!Header::is_marked_bit_set(unmarked));

        // Verify that setting/clearing doesn't affect other bits (except the marked bit)
        let expected_unmarked = test_value & !Header::marked_bit_mask();
        assert_eq!(unmarked, expected_unmarked);
    }

    #[test]
    fn test_header_bit_manipulation() {
        // Test that header conversion preserves bit manipulation
        let header = Header {
            type_id: 42,
            type_data: 0x12345678,
            size: 16,
            opaque: true,
            marked: false,
            large: false,
            type_flags: 0,
        };

        let header_value = header.to_usize();
        let marked_value = Header::set_marked_bit(header_value);
        let reconstructed = Header::from_usize(marked_value);

        assert!(reconstructed.marked);
        assert_eq!(reconstructed.type_id, header.type_id);
        assert_eq!(reconstructed.type_data, header.type_data);
        assert_eq!(reconstructed.size, header.size);
        assert_eq!(reconstructed.opaque, header.opaque);
    }
}

#[test]
fn header() {
    let header = Header {
        type_id: 0,
        type_data: 0,
        size: 0b0,
        opaque: true,
        marked: false,
        large: false,
        type_flags: 0,
    };
    let data = header.to_usize();
    // println the binary representation of the data
    println!("{:b}", data);
    assert!((data & 0b10) == 0b10);

    for t in 0..u8::MAX {
        for s in 0..256u16 {
            for sm in [true, false].iter() {
                for m in [true, false].iter() {
                    let header = Header {
                        type_id: t,
                        type_data: u32::MAX,
                        size: s,
                        opaque: *sm,
                        marked: *m,
                        large: false,
                        type_flags: 0,
                    };
                    let data = header.to_usize();
                    let new_header = Header::from_usize(data);
                    assert_eq!(header, new_header);
                }
            }
        }
    }
}

// This feels odd now that I have a header object
// I should think about a better way of representing this
pub struct HeapObject {
    pointer: usize,
    tagged: bool,
}

impl HeapObject {
    pub fn from_tagged(pointer: usize) -> Self {
        assert!(BuiltInTypes::is_heap_pointer(pointer));
        assert!(
            BuiltInTypes::untag(pointer).is_multiple_of(8),
            "Misaligned heap pointer: tagged={:#x}, untagged={:#x}, tag={}, untagged%8={}",
            pointer,
            BuiltInTypes::untag(pointer),
            pointer & 0b111,
            BuiltInTypes::untag(pointer) % 8
        );
        HeapObject {
            pointer,
            tagged: true,
        }
    }

    pub fn from_untagged(pointer: *const u8) -> Self {
        assert!((pointer as usize).is_multiple_of(8));
        HeapObject {
            pointer: pointer as usize,
            tagged: false,
        }
    }

    pub fn try_from_tagged(pointer: usize) -> Option<Self> {
        if BuiltInTypes::is_heap_pointer(pointer) {
            Some(HeapObject {
                pointer,
                tagged: true,
            })
        } else {
            None
        }
    }

    pub fn untagged(&self) -> usize {
        if self.tagged {
            BuiltInTypes::untag(self.pointer)
        } else {
            self.pointer
        }
    }

    pub fn get_object_type(&self) -> Option<BuiltInTypes> {
        if !self.tagged {
            return None;
        }
        Some(BuiltInTypes::get_kind(self.pointer))
    }

    pub fn mark(&self) {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let data: usize = unsafe { *pointer.cast::<usize>() };
        let marked_data = Header::set_marked_bit(data);
        unsafe { *pointer.cast::<usize>() = marked_data };
    }

    pub fn marked(&self) -> bool {
        self.get_header().marked
    }

    pub fn fields_size(&self) -> usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let data: usize = unsafe { *pointer };
        let header = Header::from_usize(data);

        if header.large {
            // For large objects, the actual size is in the word after the header
            let size_ptr = unsafe { pointer.add(1) };
            unsafe { *size_ptr * 8 }
        } else {
            header.size as usize * 8
        }
    }

    pub fn get_fields(&self) -> &[usize] {
        if self.is_opaque_object() {
            return &[];
        }
        let size = self.fields_size();
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let pointer = unsafe { pointer.add(self.header_size() / 8) };
        unsafe { std::slice::from_raw_parts(pointer, size / 8) }
    }

    /// String slice type ID (must match TYPE_ID_STRING_SLICE in type_ids.rs)
    const STRING_SLICE_TYPE_ID: u8 = 34;

    /// Cons string type ID (must match TYPE_ID_CONS_STRING in type_ids.rs)
    const CONS_STRING_TYPE_ID: u8 = 35;

    pub fn get_string_bytes(&self) -> &[u8] {
        let header = self.get_header();
        if header.type_id == Self::CONS_STRING_TYPE_ID {
            panic!(
                "get_string_bytes() called on cons string - use collect_string_bytes_into() or runtime.get_string_bytes_vec() instead"
            );
        }
        if header.type_id == Self::STRING_SLICE_TYPE_ID {
            // String slice: resolve through parent
            let parent_ptr = self.get_field(0);
            let offset = BuiltInTypes::untag(self.get_field(1));
            let length = header.type_data as usize;
            let parent = HeapObject::from_tagged(parent_ptr);
            // Parent is always a flat string (no nested slices).
            // The data lives on the GC heap (not in `parent`), so the reference
            // is valid for as long as the heap is stable (i.e., no GC).
            let parent_bytes = parent.get_flat_string_bytes();
            let parent_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(parent_bytes.as_ptr(), parent_bytes.len()) };
            &parent_bytes[offset..offset + length]
        } else {
            self.get_flat_string_bytes()
        }
    }

    /// Get bytes from a flat (non-slice) string. Skips header + cached hash word.
    #[inline]
    fn get_flat_string_bytes(&self) -> &[u8] {
        let header = self.get_header();
        let bytes = header.type_data as usize;
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        // Skip header + 8-byte cached hash
        let pointer = unsafe { pointer.add(self.header_size() + 8) };
        unsafe { std::slice::from_raw_parts(pointer, bytes) }
    }

    pub fn get_str_unchecked(&self) -> &str {
        let bytes = self.get_string_bytes();
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    /// Check if this is a string slice (vs flat string)
    #[inline]
    pub fn is_string_slice(&self) -> bool {
        self.get_header().type_id == Self::STRING_SLICE_TYPE_ID
    }

    /// Check if this is a cons string (lazy concatenation node)
    #[inline]
    pub fn is_cons_string(&self) -> bool {
        self.get_header().type_id == Self::CONS_STRING_TYPE_ID
    }

    /// Collect all bytes from this heap string (flat or slice) into a buffer.
    /// For cons strings, use `Runtime::get_string_bytes_vec()` instead.
    pub fn collect_flat_string_bytes_into(&self, buf: &mut Vec<u8>) {
        debug_assert!(
            !self.is_cons_string(),
            "Use Runtime::get_string_bytes_vec() for cons strings"
        );
        let bytes = self.get_string_bytes();
        buf.extend_from_slice(bytes);
    }

    /// Get the cached hash for a heap string (first 8 bytes after header).
    /// For string slices and cons strings, reads field 2 (stored as tagged int to avoid GC confusion).
    /// Returns 0 if hash has not been cached yet.
    #[inline]
    pub fn get_string_hash(&self) -> u64 {
        let type_id = self.get_header().type_id;
        if type_id == Self::STRING_SLICE_TYPE_ID || type_id == Self::CONS_STRING_TYPE_ID {
            // Field 2 stores hash as (hash << 3) — tagged int with tag 000.
            // Decode: shift right by 3. 0 → 0 (uncached).
            let raw = self.get_field(2);
            return (raw >> 3) as u64;
        }
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        let pointer = unsafe { pointer.add(self.header_size()) };
        unsafe { *(pointer as *const u64) }
    }

    /// Set the cached hash for a heap string.
    /// For string slices and cons strings, writes to field 2 (stored as tagged int to avoid GC confusion).
    #[inline]
    pub fn set_string_hash(&self, hash: u64) {
        let type_id = self.get_header().type_id;
        if type_id == Self::STRING_SLICE_TYPE_ID || type_id == Self::CONS_STRING_TYPE_ID {
            // Store as (hash << 3) so GC sees tag 000 (Int), not a heap pointer
            let encoded = (hash as usize) << 3;
            self.write_field(2, encoded);
            return;
        }
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        let pointer = unsafe { pointer.add(self.header_size()) };
        unsafe { *(pointer as *mut u64) = hash };
    }

    /// Get the cached hash for a keyword (first 8 bytes of keyword data)
    pub fn get_keyword_hash(&self) -> u64 {
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        let pointer = unsafe { pointer.add(self.header_size()) };
        unsafe { *(pointer as *const u64) }
    }

    /// Get keyword text bytes (skipping the 8-byte hash prefix)
    pub fn get_keyword_bytes(&self) -> &[u8] {
        let header = self.get_header();
        let text_len = header.type_data as usize;
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        // Skip header and hash (8 bytes)
        let pointer = unsafe { pointer.add(self.header_size() + 8) };
        unsafe { std::slice::from_raw_parts(pointer, text_len) }
    }

    pub fn get_full_object_data(&self) -> &[u8] {
        let size = self.full_size();
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        assert!(pointer.is_aligned());
        unsafe { std::slice::from_raw_parts(pointer, size) }
    }

    pub fn get_heap_references(&self) -> impl Iterator<Item = HeapObject> + '_ {
        let fields = self.get_fields();
        fields
            .iter()
            .filter(|_x| !self.is_opaque_object())
            .filter(|x| BuiltInTypes::is_heap_pointer(**x) && BuiltInTypes::untag(**x) != 0)
            .map(|&pointer| HeapObject::from_tagged(pointer))
    }

    pub fn get_fields_mut(&mut self) -> &mut [usize] {
        if self.is_opaque_object() {
            return &mut [];
        }
        let size = self.fields_size();
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let pointer = unsafe { pointer.add(self.header_size() / 8) };
        unsafe { std::slice::from_raw_parts_mut(pointer, size / 8) }
    }

    pub fn unmark(&self) {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let data: usize = unsafe { *pointer.cast::<usize>() };
        let unmarked_data = Header::clear_marked_bit(data);
        unsafe { *pointer.cast::<usize>() = unmarked_data };
    }

    pub fn full_size(&self) -> usize {
        self.fields_size() + self.header_size()
    }

    /// Header size for this object (8 for normal, 16 for large objects)
    pub fn header_size(&self) -> usize {
        let header = self.get_header();
        if header.large {
            16 // 8 bytes header + 8 bytes extended size
        } else {
            8
        }
    }

    /// Minimum header size constant (8 bytes)
    pub const MIN_HEADER_SIZE: usize = 8;

    pub fn writer_header_direct(&mut self, header: Header) {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        unsafe { *pointer.cast::<usize>() = header.to_usize() };
    }

    pub fn write_header(&mut self, field_size: Word) {
        assert!(field_size.to_bytes().is_multiple_of(8));
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;

        if !field_size.to_bytes().is_multiple_of(8) {
            panic!("Size is not aligned");
        }

        let words = field_size.to_words();
        let is_large = words > Header::MAX_INLINE_SIZE;

        let header = Header {
            type_id: 0,
            type_data: 0,
            size: if is_large { 0xFFFF } else { words as u16 },
            opaque: false,
            marked: false,
            large: is_large,
            type_flags: 0,
        };

        unsafe { *pointer.cast::<usize>() = header.to_usize() };

        // For large objects, write the actual size in the next word
        if is_large {
            let size_ptr = unsafe { pointer.add(1) };
            unsafe { *size_ptr = words };
        }
    }

    pub fn write_full_object(&mut self, data: &[u8]) {
        unsafe {
            let untagged = self.untagged();
            let pointer = untagged as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), pointer, data.len());
        }
    }

    pub fn copy_full_object(&self, to_object: &mut HeapObject) {
        let data = self.get_full_object_data();
        to_object.write_full_object(data);
    }

    pub fn copy_object_except_header(&self, to_object: &mut HeapObject) {
        let data = self.get_full_object_data();
        // Skip the actual header (8 or 16 bytes) when copying fields
        to_object.write_fields(&data[self.header_size()..]);
    }

    pub fn get_pointer(&self) -> *const u8 {
        let untagged = self.untagged();
        untagged as *const u8
    }

    pub fn write_field(&self, index: i32, value: usize) {
        debug_assert!(index < self.fields_size() as i32);
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let pointer = unsafe { pointer.add(index as usize + self.header_size() / 8) };
        unsafe { *pointer = value };
    }

    pub fn get_field(&self, arg: usize) -> usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let pointer = unsafe { pointer.add(arg + self.header_size() / 8) };
        unsafe { *pointer }
    }

    /// Get a mutable pointer to a field slot (for GC updates)
    pub fn get_field_ptr(&self, arg: usize) -> *mut usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        unsafe { pointer.add(arg + self.header_size() / 8) }
    }

    #[allow(unused)]
    pub fn get_type_id(&self) -> usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let header = unsafe { *pointer };
        let header = Header::from_usize(header);
        header.type_id as usize
    }

    pub fn write_type_id(&mut self, type_id: usize) {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let header = unsafe { *pointer };
        let header = Header::from_usize(header);
        let new_header = Header {
            type_id: type_id as u8,
            ..header
        };
        unsafe { *pointer = new_header.to_usize() };
    }

    pub fn write_struct_id(&mut self, struct_id: usize) {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let header = unsafe { *pointer };
        let header = Header::from_usize(header);
        let new_header = Header {
            type_data: struct_id as u32,
            ..header
        };
        unsafe { *pointer = new_header.to_usize() };
    }

    pub fn get_struct_id(&self) -> usize {
        self.get_type_data()
    }

    pub fn get_type_data(&self) -> usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let header = unsafe { *pointer };
        let header = Header::from_usize(header);
        header.type_data as usize
    }

    pub fn is_opaque_object(&self) -> bool {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let data: usize = unsafe { *pointer.cast::<usize>() };
        let header = Header::from_usize(data);
        header.opaque
    }

    pub fn get_header(&self) -> Header {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        assert!(pointer.is_aligned());
        let data: usize = unsafe { *pointer.cast::<usize>() };
        Header::from_usize(data)
    }

    pub fn tagged_pointer(&self) -> usize {
        if self.tagged {
            self.pointer
        } else {
            panic!("Not tagged");
        }
    }

    pub fn write_fields(&mut self, fields: &[u8]) {
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        let pointer = unsafe { pointer.add(self.header_size()) };
        unsafe { std::ptr::copy_nonoverlapping(fields.as_ptr(), pointer, fields.len()) };
    }

    /// Returns the byte length for opaque byte buffers that store length in header.type_data.
    /// This is intended for non-scanned byte blobs like continuation stack segments.
    pub fn get_opaque_bytes_len(&self) -> usize {
        let header = self.get_header();
        let len = header.type_data as usize;
        let max = self.fields_size();
        if len > max { max } else { len }
    }

    /// Get raw bytes for opaque byte buffers (length stored in header.type_data).
    pub fn get_opaque_bytes(&self) -> &[u8] {
        let len = self.get_opaque_bytes_len();
        let untagged = self.untagged();
        let pointer = untagged as *const u8;
        let pointer = unsafe { pointer.add(self.header_size()) };
        unsafe { std::slice::from_raw_parts(pointer, len) }
    }

    /// Get mutable raw bytes for opaque byte buffers (length stored in header.type_data).
    pub fn get_opaque_bytes_mut(&mut self) -> &mut [u8] {
        let len = self.get_opaque_bytes_len();
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        let pointer = unsafe { pointer.add(self.header_size()) };
        unsafe { std::slice::from_raw_parts_mut(pointer, len) }
    }

    pub fn is_zero_size(&self) -> bool {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let data: usize = unsafe { *pointer };
        let header = Header::from_usize(data);
        // Large objects are never zero-size
        !header.large && header.size == 0
    }
}

impl Ir {
    pub fn write_struct_id(&mut self, struct_pointer: Value, type_id: usize) {
        // Header layout (8 bytes, little-endian):
        // - Byte 0: flags (bits 0-7)
        // - Bytes 1-2: size (bits 8-23)
        // - Bytes 3-6: type_data/struct_id (bits 24-55)
        // - Byte 7: type_id (bits 56-63)
        //
        // Mask should preserve bytes 0-2 (flags+size) and byte 7 (type_id),
        // while clearing bytes 3-6 (type_data) so we can write the struct_id.
        let mask = 0xFF00_0000_00FF_FFFF_usize; // Preserve byte 7 and bytes 0-2
        // Use RawValue to avoid automatic tagging - the struct_id is a raw value,
        // not a tagged integer, and should be written as-is to the header.
        self.heap_store_byte_offset_masked(
            struct_pointer,
            Value::RawValue(type_id),
            0,
            Header::type_data_offset(),
            mask,
        );
    }

    pub fn read_struct_id(&mut self, struct_pointer: Value) -> Value {
        let byte_offset = Header::type_data_offset();
        let mask = 0x00FFFFFFFF000000; // Mask to isolate 4 bytes for type metadata
        let result = self.heap_load(struct_pointer);
        let masked = self.and_imm(result, mask);
        let tagged = self.tag(masked, BuiltInTypes::Int.get_tag());
        let shifted = self.shift_right_imm(tagged, (byte_offset * 8) as i32);

        self.untag(shifted)
    }
    pub fn write_type_id(&mut self, struct_pointer: Value, type_id: Value) {
        let byte_offset = Header::type_id_offset();
        let mask = !(0xFF << (byte_offset * 8));
        self.heap_store_byte_offset_masked(
            struct_pointer,
            type_id,
            0,
            Header::type_id_offset(),
            mask,
        );
    }

    pub fn read_type_id(&mut self, struct_pointer: Value) -> Value {
        let byte_offset = Header::type_id_offset();
        let mask = !(0xFF << (byte_offset * 8));
        let result = self.heap_load(struct_pointer);
        let masked = self.and_imm(result, mask);
        self.tag(masked, BuiltInTypes::Int.get_tag())
    }

    pub fn write_fields(&mut self, struct_pointer: Value, fields: &[Value]) {
        let offset = 1;
        for (i, field) in fields.iter().enumerate() {
            self.heap_store_offset(struct_pointer, *field, offset + i);
        }
    }

    pub fn write_field(&mut self, struct_pointer: Value, index: usize, field: Value) {
        let offset = 1;
        self.heap_store_offset(struct_pointer, field, offset + index);
    }

    pub fn read_field(&mut self, untagged_struct_pointer: Value, index: Value) -> Value {
        let incremented = self.add_int(index, 1);
        self.heap_load_with_reg_offset(untagged_struct_pointer, incremented)
    }

    pub fn write_field_dynamic(&mut self, struct_pointer: Value, index: Value, field: Value) {
        let incremented = self.add_int(index, 1);
        self.heap_store_with_reg_offset(struct_pointer, field, incremented);
    }

    pub fn write_small_object_header(&mut self, small_object_pointer: Value) {
        // We are going to set the least significant bits to 0b10
        // The 1 bit there will tell us that this object doesn't
        // have a size field, it is just a single 8 byte object following the header
        let header = Header {
            type_id: 0,
            type_data: 0,
            size: 1,
            opaque: true,
            marked: false,
            large: false,
            type_flags: 0,
        };
        self.heap_store(small_object_pointer, Value::RawValue(header.to_usize()));
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Word(usize);

impl Word {
    pub fn to_bytes(self) -> usize {
        self.0 * 8
    }

    pub fn from_word(size: usize) -> Word {
        Word(size)
    }

    pub fn from_bytes(len: usize) -> Word {
        Word(len / 8)
    }

    pub fn to_words(self) -> usize {
        self.0
    }
}

pub struct Tagged(usize);

impl From<Tagged> for usize {
    fn from(val: Tagged) -> Self {
        val.0
    }
}
