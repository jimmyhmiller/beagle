use std::ops::Deref;

use crate::{
    ast::{Ast, AstCompiler},
    ir::{Condition, Value},
    types::{BuiltInTypes, Header},
};

pub fn get_inline_primitive_arity(name: &str) -> usize {
    match name {
        "beagle.primitive/deref" => 1,
        "beagle.primitive/reset!" => 2,
        "beagle.primitive/compare-and-swap!" => 3,
        "beagle.primitive/breakpoint!" => 0,
        "beagle.primitive/write-type-id" => 2,
        "beagle.primitive/read-type-id" => 1,
        "beagle.primitive/read-struct-id" => 1,
        "beagle.primitive/write-field" => 3,
        "beagle.primitive/read-field" => 2,
        "beagle.primitive/breakpoint" => 0,
        "beagle.primitive/size" => 1,
        "beagle.primitive/panic" => 1,
        "beagle.primitive/is-object" => 1,
        "beagle.primitive/is-string-constant" => 1,
        "beagle.primitive/set!" => 2,
        "beagle.primitive/__get-my-thread-obj" => 0,
        _ => panic!("Unknown inline primitive: {}", name),
    }
}

// TODO: I'd rather this be on Ir I think?
impl AstCompiler<'_> {
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
                self.call_builtin("beagle.builtin/gc-add-root", vec![pointer]);
                self.ir.atomic_store(offset, value);
                args[1]
            }
            "beagle.primitive/compare-and-swap!" => {
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
            "beagle.primitive/write-type-id" => {
                let pointer = args[0];
                let pointer = self.ir.untag(pointer);
                let type_id = args[1];
                let untagged_type_id = self.ir.untag(type_id);
                self.ir.write_type_id(pointer, untagged_type_id);
                Value::Null
            }
            "beagle.primitive/read-type-id" => {
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
            "beagle.primitive/read-struct-id" => {
                let pointer = args[0];
                let pointer = self.ir.untag(pointer);
                self.ir.read_struct_id(pointer)
            }
            "beagle.primitive/write-field" => {
                let pointer = args[0];
                let untagged = self.ir.untag(pointer);
                let field = args[1];

                // Check large flag to determine header size (1 word or 2 words)
                let header = self.ir.load_from_heap(untagged, 0);
                let large_flag = self
                    .ir
                    .and_imm(header, 1 << Header::LARGE_OBJECT_BIT_POSITION);
                let zero = self.ir.assign_new(Value::RawValue(0));

                let not_large_label = self.ir.label("write_not_large");
                let done_label = self.ir.label("write_done");
                let offset_reg = self.ir.volatile_register();

                // If large flag is 0, jump to not_large
                self.ir
                    .jump_if(not_large_label, Condition::Equal, large_flag, zero);

                // Large object: add 2 to field (16-byte header)
                let field_large = self.ir.add_int(field, Value::TaggedConstant(2));
                let field_large = self.ir.mul(field_large, Value::RawValue(8));
                self.ir.assign(offset_reg, field_large);
                self.ir.jump(done_label);

                // Small object: add 1 to field (8-byte header)
                self.ir.write_label(not_large_label);
                let field_small = self.ir.add_int(field, Value::TaggedConstant(1));
                let field_small = self.ir.mul(field_small, Value::RawValue(8));
                self.ir.assign(offset_reg, field_small);

                self.ir.write_label(done_label);
                let value = args[2];
                self.ir
                    .heap_store_with_reg_offset(untagged, value, offset_reg.into());
                self.call_builtin("beagle.builtin/gc-add-root", vec![pointer]);
                Value::Null
            }
            "beagle.primitive/read-field" => {
                let pointer = args[0];
                let untagged = self.ir.untag(pointer);
                let field = args[1];

                // Check large flag to determine header size (1 word or 2 words)
                let header = self.ir.load_from_heap(untagged, 0);
                let large_flag = self
                    .ir
                    .and_imm(header, 1 << Header::LARGE_OBJECT_BIT_POSITION);
                let zero = self.ir.assign_new(Value::RawValue(0));

                let not_large_label = self.ir.label("read_not_large");
                let done_label = self.ir.label("read_done");
                let offset_reg = self.ir.volatile_register();

                // If large flag is 0, jump to not_large
                self.ir
                    .jump_if(not_large_label, Condition::Equal, large_flag, zero);

                // Large object: add 2 to field (16-byte header)
                let field_large = self.ir.add_int(field, Value::TaggedConstant(2));
                let field_large = self.ir.mul(field_large, Value::RawValue(8));
                self.ir.assign(offset_reg, field_large);
                self.ir.jump(done_label);

                // Small object: add 1 to field (8-byte header)
                self.ir.write_label(not_large_label);
                let field_small = self.ir.add_int(field, Value::TaggedConstant(1));
                let field_small = self.ir.mul(field_small, Value::RawValue(8));
                self.ir.assign(offset_reg, field_small);

                self.ir.write_label(done_label);
                self.ir
                    .heap_load_with_reg_offset(untagged, offset_reg.into())
            }
            "beagle.primitive/breakpoint" => {
                self.ir.breakpoint();
                Value::Null
            }
            "beagle.primitive/size" => {
                let pointer = args[0];
                let pointer = self.ir.untag(pointer);
                let header = self.ir.load_from_heap(pointer, 0);

                // Check if large flag (bit 2) is set
                let large_flag = self
                    .ir
                    .and_imm(header, 1 << Header::LARGE_OBJECT_BIT_POSITION);
                let zero = self.ir.assign_new(Value::RawValue(0));

                let not_large_label = self.ir.label("size_not_large");
                let done_label = self.ir.label("size_done");
                let result_reg = self.ir.volatile_register();

                // If large flag is 0, jump to not_large
                self.ir
                    .jump_if(not_large_label, Condition::Equal, large_flag, zero);

                // Large object: read size from second word (offset 1 = 8 bytes)
                let extended_size = self.ir.load_from_heap(pointer, 1);
                self.ir.assign(result_reg, extended_size);
                self.ir.jump(done_label);

                // Small object: extract inline size from header
                self.ir.write_label(not_large_label);
                let size_offset = Header::size_offset();
                let inline_size = self
                    .ir
                    .shift_right_imm_raw(header, (size_offset * 8) as i32);
                let inline_size = self.ir.and_imm(inline_size, 0x0000_0000_0000_FFFF);
                self.ir.assign(result_reg, inline_size);

                self.ir.write_label(done_label);
                self.ir.tag(result_reg.into(), BuiltInTypes::Int.get_tag())
            }
            "beagle.primitive/panic" => {
                let message = args[0];
                // print the message then call throw_error
                self.call_builtin("beagle.core/_println", vec![message]);
                self.call_builtin("beagle.builtin/throw-error", vec![]);
                Value::Null
            }
            "beagle.primitive/is-object" => {
                let pointer = args[0];
                // check the tag of the pointer
                let tag = self.ir.get_tag(pointer);
                let heap_object_tag = self
                    .ir
                    .assign_new(Value::RawValue(BuiltInTypes::HeapObject.get_tag() as usize));
                self.ir
                    .compare(tag, heap_object_tag.into(), Condition::Equal)
            }
            "beagle.primitive/is-string-constant" => {
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
            "beagle.primitive/__get-my-thread-obj" => {
                // Call the builtin that reads from thread_roots
                self.call_builtin("beagle.builtin/__get_my_thread_obj", vec![])
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
        self.call_builtin("beagle.builtin/gc-add-root", vec![object.into()]);
        let untagged_object = self.ir.untag(object.into());
        // self.ir.breakpoint();
        let struct_id = self.ir.read_struct_id(untagged_object);
        let property_location = Value::RawValue(self.compiler.add_property_lookup().unwrap());
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
