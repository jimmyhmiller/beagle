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

    pub fn tag_size() -> i32 {
        3
    }

    pub fn is_heap_pointer(value: usize) -> bool {
        match BuiltInTypes::get_kind(value) {
            BuiltInTypes::Int => false,
            BuiltInTypes::Float => false,
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

pub struct HeapObject {
    pointer: usize,
    tagged: bool,
}

const SIZE_SHIFT: usize = 1;

// TODO: Implement methods for writing the header of the heap object
// make sure we always use this representation everywhere so we can
// change things in one place
impl HeapObject {
    pub fn from_tagged(pointer: usize) -> Self {
        assert!(pointer % 8 != 0);
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
        data >> SIZE_SHIFT
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
        unsafe { std::slice::from_raw_parts(pointer, size) }
    }

    pub fn get_heap_references(&self) -> impl Iterator<Item = HeapObject> + '_ {
        let fields = self.get_fields();
        fields
            .iter()
            .filter(|x| BuiltInTypes::is_heap_pointer(**x))
            .map(|&pointer| HeapObject::from_tagged(pointer))
    }

    pub fn get_fields_mut(&mut self) -> &mut [usize] {
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
        assert!(field_size.to_bytes() != 0);
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        // TODO: Don't mulitply so that we just store
        // words and can represent the size of the object
        // in a more compact way
        let data = field_size.to_bytes() << SIZE_SHIFT;
        assert!(data > 8);
        unsafe { *pointer.cast::<usize>() = data };
    }

    pub fn write_full_object(&mut self, data: &[u8]) {
        unsafe {
            let untagged = self.untagged();
            let pointer = untagged as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), pointer, data.len());
        }
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
        let pointer = unsafe { pointer.add(arg as usize + Self::header_size() / 8) };
        unsafe { *pointer }
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
