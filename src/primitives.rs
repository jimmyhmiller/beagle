use crate::{
    ast::AstCompiler,
    ir::{Condition, Value},
    types::{BuiltInTypes, Header},
};

// TODO: I'd rather this be on Ir I think?
impl<'a> AstCompiler<'a> {
    pub fn compile_inline_primitive_function(&mut self, name: &str, args: Vec<Value>) -> Value {
        match name {
            "beagle.primitive/deref" => {
                // self.ir.breakpoint();
                let pointer = args[0];
                let untagged = self.ir.untag(pointer);
                // TODO: I need a raw add that doesn't check for tags
                let offset = self.ir.add_int(untagged, Value::RawValue(8));
                let reg = self.ir.volatile_register();
                self.ir.atomic_load(reg.into(), offset)
            }
            "beagle.primitive/reset!" => {
                let pointer = args[0];
                let untagged = self.ir.untag(pointer);
                // TODO: I need a raw add that doesn't check for tags
                let offset = self.ir.add_int(untagged, Value::RawValue(8));
                let value = args[1];
                self.call_builtin("beagle.builtin/gc_add_root", vec![pointer]);
                self.ir.atomic_store(offset, value);
                args[1]
            }
            "beagle.primitive/compare_and_swap!" => {
                // self.ir.breakpoint();
                let pointer = args[0];
                let untagged = self.ir.untag(pointer);
                let offset = self.ir.add_int(untagged, Value::RawValue(8));
                let expected = args[1];
                let new = args[2];
                let expected_and_result = self.ir.assign_new_force(expected);
                self.ir
                    .compare_and_swap(expected_and_result.into(), new, offset);
                // TODO: I should do a conditional move here instead of a jump
                let label = self.ir.label("compare_and_swap");
                let result = self.ir.assign_new(Value::True);
                self.ir
                    .jump_if(label, Condition::Equal, expected_and_result, expected);
                self.ir.assign(result, Value::False);
                self.ir.write_label(label);
                result.into()
            }
            "beagle.primitive/breakpoint!" => {
                self.ir.breakpoint();
                Value::Null
            }
            "beagle.primitive/write_type_id" => {
                let pointer = args[0];
                let pointer = self.ir.untag(pointer);
                let type_id = args[1];
                let untagged_type_id = self.ir.untag(type_id);
                self.ir.write_type_id(pointer, untagged_type_id);
                Value::Null
            }
            "beagle.primitive/read_type_id" => {
                let pointer = args[0];
                let pointer = self.ir.untag(pointer);
                let header = self.ir.load_from_heap(pointer, 0);
                // mask and shift so we get the size
                let size_offset = Header::type_id_offset();
                let value = self.ir.shift_right_imm(header, (size_offset * 8) as i32);

                let value = self.ir.and_imm(value, 0x0000_0000_0000_FFFF);
                self.ir.tag(value, BuiltInTypes::Int.get_tag())
            }
            "beagle.primitive/write_field" => {
                // self.ir.breakpoint();
                let pointer = args[0];
                let pointer = self.ir.untag(pointer);
                let field = args[1];
                let field = self.ir.add_int(field, Value::TaggedConstant(1));
                let field = self.ir.mul(field, Value::RawValue(8));
                // let untagged_field = self.ir.untag(field);
                let value = args[2];
                self.ir.heap_store_with_reg_offset(pointer, value, field);
                self.call_builtin("beagle.builtin/gc_add_root", vec![pointer]);
                Value::Null
            }
            "beagle.primitive/read_field" => {
                let pointer = args[0];
                let pointer = self.ir.untag(pointer);
                let field = args[1];
                let field = self.ir.add_int(field, Value::TaggedConstant(1));
                let field = self.ir.mul(field, Value::RawValue(8));
                // self.ir.breakpoint();
                self.ir.heap_load_with_reg_offset(pointer, field)
            }
            "beagle.primitive/breakpoint" => {
                self.ir.breakpoint();
                Value::Null
            }
            "beagle.primitive/size" => {
                let pointer = args[0];
                let pointer = self.ir.untag(pointer);
                let header = self.ir.load_from_heap(pointer, 0);
                // mask and shift so we get the size
                let size_offset = Header::size_offset();
                let value = self.ir.shift_right_imm(header, (size_offset * 8) as i32);

                let value = self.ir.and_imm(value, 0x0000_0000_0000_FFFF);
                self.ir.tag(value, BuiltInTypes::Int.get_tag())
            }
            "beagle.primitive/panic" => {
                let message = args[0];
                // print the message then call throw_error
                self.call_builtin("beagle.core/println", vec![message]);
                self.call_builtin("beagle.builtin/throw_error", vec![]);
                Value::Null
            }
            "beagle.primitive/is_object" => {
                let pointer = args[0];
                // check the tag of the pointer
                let tag = self.ir.get_tag(pointer);
                let heap_object_tag = self
                    .ir
                    .assign_new(Value::RawValue(BuiltInTypes::HeapObject.get_tag() as usize));
                self.ir
                    .compare(tag, heap_object_tag.into(), Condition::Equal)
            }
            _ => panic!("Unknown inline primitive function {}", name),
        }
    }
}
