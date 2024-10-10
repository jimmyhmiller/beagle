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

    pub fn tag(&self, value: isize) -> isize {
        let value = value << 3;
        let tag = self.get_tag();
        value | tag
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

    pub fn untag(pointer: usize) -> usize {
        pointer >> 3
    }

    pub fn untag_isize(pointer: isize) -> isize {
        pointer >> 3
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
            _ => panic!("Invalid tag {}", pointer & 0b111),
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
            panic!("Integer overflow")
        }
        BuiltInTypes::Int.tag(value)
    }

    pub fn construct_boolean(value: bool) -> isize {
        let bool = BuiltInTypes::Bool;
        if value {
            bool.tag(1)
        } else {
            bool.tag(0)
        }
    }

    pub fn construct_float(x: f64) -> isize {
        let value = x.to_bits() as isize;
        BuiltInTypes::Float.tag(value)
    }

    pub fn tag_size() -> i32 {
        3
    }

    pub fn is_heap_pointer(value: usize) -> bool {
        match BuiltInTypes::get_kind(value) {
            BuiltInTypes::Int => false,
            BuiltInTypes::Float => true,
            BuiltInTypes::String => false,
            BuiltInTypes::Bool => false,
            BuiltInTypes::Function => false,
            BuiltInTypes::Closure => true,
            BuiltInTypes::HeapObject => true,
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
    pub size: u8,
    pub small: bool,
    pub marked: bool,
}

impl Header {
    // | Byte 7  | Bytes 3-6     | Byte 2 | Byte 1  | Byte 0               |
    // |---------|---------------|--------|---------|----------------------|
    // | Type    | Type Metadata | Size   | Padding | Flag bits            |
    // |         | (4 bytes)     |        |         | Small object (bit 1) |
    // |         |               |        |         | Marked (bit 0)       |

    fn to_usize(self) -> usize {
        let mut data: usize = 0;
        data |= (self.type_id as usize) << 56;
        data |= (self.type_data as usize) << 24;
        data |= (self.size as usize) << 16;
        if self.small {
            data |= 0b10;
        }
        if self.marked {
            data |= 0b1;
        }
        data
    }

    fn from_usize(data: usize) -> Self {
        let _type = (data >> 56) as u8;
        let type_data = (data >> 24) as u32;
        let size = (data >> 16) as u8;
        let small = (data & 0b10) == 0b10;
        let marked = (data & 0b1) == 0b1;
        Header {
            type_id: _type,
            type_data,
            size,
            small,
            marked,
        }
    }

    fn type_id_offset() -> usize {
        7
    }

    fn type_data_offset() -> usize {
        3
    }

    pub fn size_offset() -> usize {
        2
    }
}

#[test]
fn header() {
    let header = Header {
        type_id: 0,
        type_data: 0,
        size: 0b0,
        small: true,
        marked: false,
    };
    let data = header.to_usize();
    // println the binary representation of the data
    println!("{:b}", data);
    assert!((data & 0b10) == 0b10);

    for t in 0..u8::MAX {
        for s in 0..u8::MAX {
            for sm in [true, false].iter() {
                for m in [true, false].iter() {
                    let header = Header {
                        type_id: t,
                        type_data: u32::MAX,
                        size: s,
                        small: *sm,
                        marked: *m,
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

// TODO: Implement methods for writing the header of the heap object
// make sure we always use this representation everywhere so we can
// change things in one place
impl HeapObject {
    pub fn from_tagged(pointer: usize) -> Self {
        assert!(BuiltInTypes::is_heap_pointer(pointer));
        HeapObject {
            pointer,
            tagged: true,
        }
    }

    pub fn from_untagged(pointer: *const u8) -> Self {
        assert!(pointer as usize % 8 == 0);
        HeapObject {
            pointer: pointer as usize,
            tagged: false,
        }
    }

    pub fn untagged(&self) -> usize {
        if self.tagged {
            BuiltInTypes::untag(self.pointer)
        } else {
            self.pointer
        }
    }

    pub fn mark(&self) {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let mut data: usize = unsafe { *pointer.cast::<usize>() };

        // check right most bit
        if (data & 1) != 1 {
            data |= 1;
        }
        unsafe { *pointer.cast::<usize>() = data };
    }

    pub fn marked(&self) -> bool {
        let untagged = self.untagged();
        let pointer = untagged as *mut isize;
        let data: usize = unsafe { *pointer.cast::<usize>() };
        data & 1 == 1
    }

    pub fn fields_size(&self) -> usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut isize;
        let data: usize = unsafe { *pointer.cast::<usize>() };
        let header = Header::from_usize(data);
        // TODO: This is number of bytes, not number of fields
        // so is this wrong?
        if header.size % 8 != 0 {
            panic!("Size is not aligned");
        }
        header.size as usize
    }

    pub fn get_fields(&self) -> &[usize] {
        let size = self.fields_size();
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let pointer = unsafe { pointer.add(Self::header_size() / 8) };
        unsafe { std::slice::from_raw_parts(pointer, size / 8) }
    }

    pub fn get_full_object_data(&self) -> &[u8] {
        let size = self.full_size();
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        assert!(pointer as usize % 8 == 0);
        unsafe { std::slice::from_raw_parts(pointer, size) }
    }

    pub fn get_heap_references(&self) -> impl Iterator<Item = HeapObject> + '_ {
        let fields = self.get_fields();
        fields
            .iter()
            // Hack to make sure we don't get any references if this is a small object
            // Trying to do an empty in a condition makes the type checker complain
            .filter(|_x| !self.is_small_object())
            .filter(|x| BuiltInTypes::is_heap_pointer(**x))
            .map(|&pointer| HeapObject::from_tagged(pointer))
    }

    pub fn get_fields_mut(&mut self) -> &mut [usize] {
        if self.is_small_object() {
            return &mut [];
        }
        let size = self.fields_size();
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let pointer = unsafe { pointer.add(1) };
        unsafe { std::slice::from_raw_parts_mut(pointer, size / 8) }
    }

    pub fn unmark(&self) {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let mut data: usize = unsafe { *pointer.cast::<usize>() };
        // check right most bit
        if (data & 1) == 1 {
            data &= !1;
        }
        unsafe { *pointer.cast::<usize>() = data };
    }

    pub fn full_size(&self) -> usize {
        self.fields_size() + Self::header_size()
    }

    pub fn header_size() -> usize {
        8
    }

    pub fn write_header(&mut self, field_size: Word) {
        assert!(field_size.to_bytes() % 8 == 0);
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;

        if field_size.to_bytes() % 8 != 0 {
            panic!("Size is not aligned");
        }

        let header = Header {
            type_id: 0,
            type_data: 0,
            size: field_size.to_bytes() as u8,
            small: false,
            marked: false,
        };

        unsafe { *pointer.cast::<usize>() = header.to_usize() };
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
        to_object.write_fields(&data[Self::header_size()..]);
    }

    pub fn get_pointer(&self) -> *const u8 {
        let untagged = self.untagged();
        untagged as *const u8
    }

    pub fn write_field(&self, arg: i32, tagged_new: usize) {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let pointer = unsafe { pointer.add(arg as usize + Self::header_size() / 8) };
        unsafe { *pointer = tagged_new };
    }

    pub fn get_field(&self, arg: usize) -> usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let pointer = unsafe { pointer.add(arg + Self::header_size() / 8) };
        unsafe { *pointer }
    }

    #[allow(unused)]
    pub fn get_type_id(&self) -> usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let header = unsafe { *pointer };
        let header = Header::from_usize(header);
        header.type_id as usize
    }

    pub fn get_struct_id(&self) -> usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let header = unsafe { *pointer };
        let header = Header::from_usize(header);
        header.type_data as usize
    }

    pub fn is_small_object(&self) -> bool {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let data: usize = unsafe { *pointer.cast::<usize>() };
        let header = Header::from_usize(data);
        header.small
    }

    #[allow(unused)]
    pub fn get_header(&self) -> Header {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
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
    
    fn write_fields(&mut self, fields: &[u8])  {
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        let pointer = unsafe { pointer.add(Self::header_size()) };
        unsafe { std::ptr::copy_nonoverlapping(fields.as_ptr(), pointer, fields.len()) };
    }
}

impl Ir {
    pub fn write_struct_id(&mut self, struct_pointer: Value, type_id: usize) {
        // I need to understand this stuff better.
        // I really need to actually study some bit wise operations
        let mask = 0x000000FFFFFFFF;
        self.heap_store_byte_offset_masked(
            struct_pointer,
            type_id,
            0,
            Header::type_data_offset(),
            mask,
        );
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

    pub fn write_small_object_header(&mut self, small_object_pointer: Value) {
        // We are going to set the least significant bits to 0b10
        // The 1 bit there will tell us that this object doesn't
        // have a size field, it is just a single 8 byte object following the header
        let header = Header {
            type_id: 0,
            type_data: 0,
            size: 8,
            small: true,
            marked: false,
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
}
