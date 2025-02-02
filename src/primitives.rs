use std::ops::Deref;

use crate::{
    ast::{Ast, AstCompiler},
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
                let value = self
                    .ir
                    .shift_right_imm_raw(header, (size_offset * 8) as i32);

                let value = self.ir.and_imm(value, 0x0000_0000_0000_FFFF);
                self.ir.tag(value, BuiltInTypes::Int.get_tag())
            }
            "beagle.primitive/read_struct_id" => {
                let pointer = args[0];
                let pointer = self.ir.untag(pointer);
                self.ir.read_struct_id(pointer)
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
            "beagle.primitive/is_string_constant" => {
                let pointer = args[0];
                // check the tag of the pointer
                let tag = self.ir.get_tag(pointer);
                let heap_object_tag = self
                    .ir
                    .assign_new(Value::RawValue(BuiltInTypes::String.get_tag() as usize));
                self.ir
                    .compare(tag, heap_object_tag.into(), Condition::Equal)
            }
            "beagle.primitive/set!" => {
                let pointer = args[0];
                let value = args[1];
                self.ir.heap_store(pointer, value);
                Value::Null
            }
            _ => panic!("Unknown inline primitive function {}", name),
        }
    }

    pub fn should_not_evaluate_arguments(&self, name: &str) -> bool {
        if name == "beagle.primitive/set!" {
            return true;
        }
        false
    }

    pub fn compile_macro_like_primitive(&mut self, name: String, args: Vec<Ast>) -> Value {
        if name != "beagle.primitive/set!" {
            panic!("Unknown macro-like primitive {}", name);
        }

        if args.len() != 2 {
            // TODO: Error handling properly
            panic!("set! expects 2 arguments, got {}", args.len());
        }

        let property_access = &args[0];
        if !matches!(property_access, Ast::PropertyAccess { .. }) {
            panic!("set! expects a property access as the first argument");
        };

        let Ast::PropertyAccess {
            object,
            property,
            token_range: _,
        } = property_access
        else {
            panic!("set! expects a property access as the first argument");
        };

        let object = object.deref();
        if !matches!(object, Ast::Identifier { .. }) {
            panic!("set! expects an identifier as the first argument for now");
        }

        let property = property.deref();
        if !matches!(property, Ast::Identifier { .. }) {
            panic!("set! expects an identifier as the second argument for now");
        }

        let object = self.call_compile(object);
        let value = self.call_compile(&args[1]);

        let object = self.ir.assign_new(object);
        self.call_builtin("beagle.builtin/gc_add_root", vec![object.into()]);
        let untagged_object = self.ir.untag(object.into());
        // self.ir.breakpoint();
        let struct_id = self.ir.read_struct_id(untagged_object);
        let property_location = Value::RawValue(self.compiler.add_property_lookup());
        let property_location = self.ir.assign_new(property_location);
        let property_value = self.ir.load_from_heap(property_location.into(), 0);
        let result = self.ir.assign_new(0);

        let exit_property_access = self.ir.label("exit_property_access");
        let slow_property_path = self.ir.label("slow_property_path");
        // self.ir.jump(slow_property_path);
        self.ir.jump_if(
            slow_property_path,
            Condition::NotEqual,
            struct_id,
            property_value,
        );

        let property_offset = self.ir.load_from_heap(property_location.into(), 1);
        self.ir
            .write_field_dynamic(untagged_object, property_offset, value);

        self.ir.jump(exit_property_access);

        self.ir.write_label(slow_property_path);
        let property = if let Ast::Identifier(name, _) = property {
            name.clone()
        } else {
            panic!("Expected identifier")
        };
        let constant_ptr = self.string_constant(property);
        let constant_ptr = self.ir.assign_new(constant_ptr);
        let call_result = self.call_builtin(
            "beagle.builtin/write_field",
            vec![
                object.into(),
                constant_ptr.into(),
                property_location.into(),
                value,
            ],
        );

        self.ir.assign(result, call_result);

        self.ir.write_label(exit_property_access);

        Value::Null
    }
}
