use std::ops::Deref;

use crate::{
    ast::{Ast, AstCompiler},
    builtins::write_barrier,
    collections::{TYPE_ID_RAW_ARRAY, TYPE_ID_STRING_BUILDER},
    compiler::CompileError,
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
        "beagle.primitive/string-builder-byte-at" | "beagle.string-builder/byte-at" => 2,
        "beagle.primitive/string-builder-set-byte-at!" | "beagle.string-builder/set-byte-at!" => 3,
        "beagle.mutable-array/get" => 2,
        "beagle.mutable-array/write-field" => 3,
        "beagle.mutable-array/swap" => 3,
        "beagle.mutable-array/read-field-unsafe" => 2,
        "beagle.primitive/breakpoint" => 0,
        "beagle.primitive/size" => 1,
        "beagle.primitive/panic" => 1,
        "beagle.primitive/is-object" => 1,
        "beagle.primitive/is-string-constant" => 1,
        "beagle.primitive/set!" => 2,
        "beagle.primitive/__get-my-thread-obj" => 0,
        "beagle.primitive/__pause" => 0,
        _ => panic!("Unknown inline primitive: {}", name),
    }
}

// TODO: I'd rather this be on Ir I think?
impl AstCompiler<'_> {
    fn compile_string_builder_type_guard(
        &mut self,
        sb: Value,
        invalid_label: crate::common::Label,
    ) -> Value {
        let tag = self.ir.get_tag(sb);
        self.ir.jump_if(
            invalid_label,
            Condition::NotEqual,
            tag,
            Value::RawValue(BuiltInTypes::HeapObject.get_tag() as usize),
        );

        let sb_ptr = self.ir.untag(sb);
        let header = self.ir.load_from_heap(sb_ptr, 0);
        let type_id_offset = Header::type_id_offset();
        let type_id = self
            .ir
            .shift_right_imm_raw(header, (type_id_offset * 8) as i32);
        let type_id = self.ir.and_imm(type_id, 0x0000_0000_0000_FFFF);
        self.ir.jump_if(
            invalid_label,
            Condition::NotEqual,
            type_id,
            Value::RawValue(TYPE_ID_STRING_BUILDER as usize),
        );
        sb_ptr
    }

    fn compile_string_builder_payload_base(&mut self, sb: Value) -> Value {
        let sb_ptr = self.ir.untag(sb);
        let storage_tagged = self.ir.load_from_heap(sb_ptr, 1);
        let storage_ptr = self.ir.untag(storage_tagged);

        let header = self.ir.load_from_heap(storage_ptr, 0);
        let large_flag = self
            .ir
            .and_imm(header, 1 << Header::LARGE_OBJECT_BIT_POSITION);
        let zero = self.ir.assign_new(Value::RawValue(0));
        let not_large = self.ir.label("sb_storage_not_large");
        let done = self.ir.label("sb_storage_payload_done");
        let payload_offset = self.ir.volatile_register();

        self.ir.assign(payload_offset, Value::RawValue(16));
        self.ir
            .jump_if(not_large, Condition::Equal, large_flag, zero);
        self.ir.jump(done);

        self.ir.write_label(not_large);
        self.ir.assign(payload_offset, Value::RawValue(8));

        self.ir.write_label(done);
        self.ir
            .add_int(storage_ptr, Value::Register(payload_offset))
    }

    /// Type-guard that `array` is a `RAW_ARRAY` heap object. On a miss, jumps
    /// to `slow` (the caller delegates to the real wrapper there, which throws
    /// on a non-array). Returns `(array_ptr, header, large_flag)` for the fast
    /// path. Shared by the inlined `swap`/`write-field` expansions; mirrors the
    /// guard inside the inlined `get`.
    fn array_type_guard(
        &mut self,
        array: Value,
        slow: crate::common::Label,
    ) -> (Value, Value, Value) {
        let tag = self.ir.get_tag(array);
        self.ir.jump_if(
            slow,
            Condition::NotEqual,
            tag,
            Value::RawValue(BuiltInTypes::HeapObject.get_tag() as usize),
        );
        let array_ptr = self.ir.untag(array);
        let header = self.ir.load_from_heap(array_ptr, 0);
        let type_id_offset = Header::type_id_offset();
        let type_id = self
            .ir
            .shift_right_imm_raw(header, (type_id_offset * 8) as i32);
        let type_id = self.ir.and_imm(type_id, 0x0000_0000_0000_FFFF);
        self.ir.jump_if(
            slow,
            Condition::NotEqual,
            type_id,
            Value::RawValue(TYPE_ID_RAW_ARRAY as usize),
        );
        let large_flag = self
            .ir
            .and_imm(header, 1 << Header::LARGE_OBJECT_BIT_POSITION);
        (array_ptr, header, large_flag)
    }

    /// Tagged-int element count of an array (large: extended-size word; small:
    /// inline header field).
    fn array_tagged_size(&mut self, array_ptr: Value, header: Value, large_flag: Value) -> Value {
        let zero = self.ir.assign_new(Value::RawValue(0));
        let not_large = self.ir.label("arr_size_not_large");
        let done = self.ir.label("arr_size_done");
        let size_reg = self.ir.volatile_register();
        self.ir
            .jump_if(not_large, Condition::Equal, large_flag, zero);
        let extended = self.ir.load_from_heap(array_ptr, 1);
        self.ir.assign(size_reg, extended);
        self.ir.jump(done);
        self.ir.write_label(not_large);
        let size_offset = Header::size_offset();
        let inline_size = self
            .ir
            .shift_right_imm_raw(header, (size_offset * 8) as i32);
        let inline_size = self.ir.and_imm(inline_size, 0x0000_0000_0000_FFFF);
        self.ir.assign(size_reg, inline_size);
        self.ir.write_label(done);
        self.ir
            .tag(Value::Register(size_reg), BuiltInTypes::Int.get_tag())
    }

    /// Byte offset (from the array pointer) of element `index_value` (a tagged
    /// int), accounting for the 1- vs 2-word header (`large_flag`).
    fn array_byte_offset(&mut self, index_value: Value, large_flag: Value) -> Value {
        let zero = self.ir.assign_new(Value::RawValue(0));
        let not_large = self.ir.label("arr_off_not_large");
        let done = self.ir.label("arr_off_done");
        let offset_reg = self.ir.volatile_register();
        self.ir
            .jump_if(not_large, Condition::Equal, large_flag, zero);
        let field_large = self.ir.add_int(index_value, Value::TaggedConstant(2));
        let field_large = self.ir.mul(field_large, Value::RawValue(8));
        self.ir.assign(offset_reg, field_large);
        self.ir.jump(done);
        self.ir.write_label(not_large);
        let field_small = self.ir.add_int(index_value, Value::TaggedConstant(1));
        let field_small = self.ir.mul(field_small, Value::RawValue(8));
        self.ir.assign(offset_reg, field_small);
        self.ir.write_label(done);
        Value::Register(offset_reg)
    }

    /// Emit the generational-GC card-marking write barrier for a store of
    /// `value` into the object at `array_ptr` (already untagged).
    fn array_write_barrier(&mut self, array_ptr: Value, value: Value) {
        let write_barrier_fn =
            Value::RawValue((write_barrier as usize) << BuiltInTypes::tag_size());
        self.ir
            .call_builtin(write_barrier_fn, vec![array_ptr, value]);
    }

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
                self.ir.atomic_store(offset, value);

                // Card marking for generational GC write barrier
                // The function pointer needs to be pre-shifted left because call_builtin
                // shifts right to "untag" the function pointer
                let write_barrier_fn =
                    Value::RawValue((write_barrier as usize) << BuiltInTypes::tag_size());
                self.ir
                    .call_builtin(write_barrier_fn, vec![untagged, value]);

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

                // Card marking for generational GC write barrier (on success)
                // Note: We mark the card even if CAS fails - this is safe (just extra work)
                let write_barrier_fn =
                    Value::RawValue((write_barrier as usize) << BuiltInTypes::tag_size());
                self.ir.call_builtin(write_barrier_fn, vec![untagged, new]);

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

                // Card marking for generational GC write barrier
                let write_barrier_fn =
                    Value::RawValue((write_barrier as usize) << BuiltInTypes::tag_size());
                self.ir
                    .call_builtin(write_barrier_fn, vec![untagged, value]);

                Value::Null
            }
            "beagle.primitive/read-field" | "beagle.mutable-array/read-field-unsafe" => {
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
            "beagle.mutable-array/get" => {
                let array = args[0];
                let index = args[1];

                let invalid = self.ir.label("array_get_invalid");
                let done = self.ir.label("array_get_done");
                let result = self.ir.assign_new(Value::Null);

                let tag = self.ir.get_tag(array);
                self.ir.jump_if(
                    invalid,
                    Condition::NotEqual,
                    tag,
                    Value::RawValue(BuiltInTypes::HeapObject.get_tag() as usize),
                );

                let array_ptr = self.ir.untag(array);
                let header = self.ir.load_from_heap(array_ptr, 0);
                let type_id_offset = Header::type_id_offset();
                let type_id = self
                    .ir
                    .shift_right_imm_raw(header, (type_id_offset * 8) as i32);
                let type_id = self.ir.and_imm(type_id, 0x0000_0000_0000_FFFF);
                self.ir.jump_if(
                    invalid,
                    Condition::NotEqual,
                    type_id,
                    Value::RawValue(TYPE_ID_RAW_ARRAY as usize),
                );

                let index_reg = self.ir.assign_new(index);
                let index_value = Value::Register(index_reg);
                self.ir.jump_if(
                    invalid,
                    Condition::LessThan,
                    index_value,
                    Value::TaggedConstant(0),
                );

                let large_flag = self
                    .ir
                    .and_imm(header, 1 << Header::LARGE_OBJECT_BIT_POSITION);
                let zero = self.ir.assign_new(Value::RawValue(0));
                let not_large_size = self.ir.label("array_get_not_large_size");
                let size_done = self.ir.label("array_get_size_done");
                let size_reg = self.ir.volatile_register();

                self.ir
                    .jump_if(not_large_size, Condition::Equal, large_flag, zero);
                let extended_size = self.ir.load_from_heap(array_ptr, 1);
                self.ir.assign(size_reg, extended_size);
                self.ir.jump(size_done);

                self.ir.write_label(not_large_size);
                let size_offset = Header::size_offset();
                let inline_size = self
                    .ir
                    .shift_right_imm_raw(header, (size_offset * 8) as i32);
                let inline_size = self.ir.and_imm(inline_size, 0x0000_0000_0000_FFFF);
                self.ir.assign(size_reg, inline_size);

                self.ir.write_label(size_done);
                let tagged_size = self
                    .ir
                    .tag(Value::Register(size_reg), BuiltInTypes::Int.get_tag());
                self.ir.jump_if(
                    invalid,
                    Condition::GreaterThanOrEqual,
                    index_value,
                    tagged_size,
                );

                let offset_reg = self.ir.volatile_register();
                let not_large = self.ir.label("array_get_not_large");
                let offset_done = self.ir.label("array_get_offset_done");

                self.ir
                    .jump_if(not_large, Condition::Equal, large_flag, zero);
                let field_large = self.ir.add_int(index_value, Value::TaggedConstant(2));
                let field_large = self.ir.mul(field_large, Value::RawValue(8));
                self.ir.assign(offset_reg, field_large);
                self.ir.jump(offset_done);

                self.ir.write_label(not_large);
                let field_small = self.ir.add_int(index_value, Value::TaggedConstant(1));
                let field_small = self.ir.mul(field_small, Value::RawValue(8));
                self.ir.assign(offset_reg, field_small);

                self.ir.write_label(offset_done);
                let value = self
                    .ir
                    .heap_load_with_reg_offset(array_ptr, Value::Register(offset_reg));
                self.ir.assign(result, value);
                self.ir.jump(done);

                self.ir.write_label(invalid);
                self.ir.assign(result, Value::Null);
                self.ir.write_label(done);
                Value::Register(result)
            }
            "beagle.mutable-array/write-field" => {
                // Fast path: array + in-bounds → inline store + write barrier.
                // Any miss (non-array or out-of-bounds) → call the real wrapper,
                // which throws on a non-array and returns null out-of-bounds —
                // exact semantics with no duplication.
                let array = args[0];
                let index = args[1];
                let value = args[2];

                let slow = self.ir.label("awf_slow");
                let done = self.ir.label("awf_done");
                let result = self.ir.assign_new(Value::Null);

                let (array_ptr, header, large_flag) = self.array_type_guard(array, slow);
                let index_reg = self.ir.assign_new(index);
                let index_value = Value::Register(index_reg);
                self.ir.jump_if(
                    slow,
                    Condition::LessThan,
                    index_value,
                    Value::TaggedConstant(0),
                );
                let tagged_size = self.array_tagged_size(array_ptr, header, large_flag);
                self.ir.jump_if(
                    slow,
                    Condition::GreaterThanOrEqual,
                    index_value,
                    tagged_size,
                );
                let offset = self.array_byte_offset(index_value, large_flag);
                self.ir.heap_store_with_reg_offset(array_ptr, value, offset);
                self.array_write_barrier(array_ptr, value);
                self.ir.assign(result, Value::Null);
                self.ir.jump(done);

                self.ir.write_label(slow);
                let slow_result = self
                    .call_builtin(
                        "beagle.mutable-array/write-field",
                        vec![array, index, value],
                    )
                    .expect("beagle.mutable-array/write-field must resolve for inline slow path");
                self.ir.assign(result, slow_result);
                self.ir.write_label(done);
                Value::Register(result)
            }
            "beagle.mutable-array/swap" => {
                // Fast path: array + both indices in-bounds → two inline reads +
                // two inline writes (+ barriers). Any miss → real wrapper (throws
                // on non-array, null out-of-bounds). `tmp`/offsets stay live
                // across the barrier calls; the GP allocator preserves them.
                let array = args[0];
                let i = args[1];
                let j = args[2];

                let slow = self.ir.label("asw_slow");
                let done = self.ir.label("asw_done");
                let result = self.ir.assign_new(Value::Null);

                let (array_ptr, header, large_flag) = self.array_type_guard(array, slow);
                let i_reg = self.ir.assign_new(i);
                let i_value = Value::Register(i_reg);
                let j_reg = self.ir.assign_new(j);
                let j_value = Value::Register(j_reg);
                let tagged_size = self.array_tagged_size(array_ptr, header, large_flag);
                self.ir
                    .jump_if(slow, Condition::LessThan, i_value, Value::TaggedConstant(0));
                self.ir
                    .jump_if(slow, Condition::GreaterThanOrEqual, i_value, tagged_size);
                self.ir
                    .jump_if(slow, Condition::LessThan, j_value, Value::TaggedConstant(0));
                self.ir
                    .jump_if(slow, Condition::GreaterThanOrEqual, j_value, tagged_size);

                let off_i = self.array_byte_offset(i_value, large_flag);
                let tmp = self.ir.heap_load_with_reg_offset(array_ptr, off_i);
                let tmp = self.ir.assign_new(tmp);
                let off_j = self.array_byte_offset(j_value, large_flag);
                let vj = self.ir.heap_load_with_reg_offset(array_ptr, off_j);
                let vj = self.ir.assign_new(vj);
                // perm[i] = perm[j]
                let off_i2 = self.array_byte_offset(i_value, large_flag);
                self.ir
                    .heap_store_with_reg_offset(array_ptr, Value::Register(vj), off_i2);
                self.array_write_barrier(array_ptr, Value::Register(vj));
                // perm[j] = tmp
                let off_j2 = self.array_byte_offset(j_value, large_flag);
                self.ir
                    .heap_store_with_reg_offset(array_ptr, Value::Register(tmp), off_j2);
                self.array_write_barrier(array_ptr, Value::Register(tmp));
                self.ir.assign(result, Value::Null);
                self.ir.jump(done);

                self.ir.write_label(slow);
                let slow_result = self
                    .call_builtin("beagle.mutable-array/swap", vec![array, i, j])
                    .expect("beagle.mutable-array/swap must resolve for inline slow path");
                self.ir.assign(result, slow_result);
                self.ir.write_label(done);
                Value::Register(result)
            }
            "beagle.primitive/string-builder-byte-at" | "beagle.string-builder/byte-at" => {
                let sb = args[0];
                let index = args[1];

                let out_of_bounds = self.ir.label("sb_byte_at_oob");
                let done = self.ir.label("sb_byte_at_done");
                let result = self.ir.assign_new(Value::Null);
                let sb_ptr = self.compile_string_builder_type_guard(sb, out_of_bounds);
                let len = self.ir.load_from_heap(sb_ptr, 2);
                let index_reg = self.ir.assign_new(index);
                let index_value = Value::Register(index_reg);

                self.ir.jump_if(
                    out_of_bounds,
                    Condition::LessThan,
                    index_value,
                    Value::TaggedConstant(0),
                );
                self.ir.jump_if(
                    out_of_bounds,
                    Condition::GreaterThanOrEqual,
                    index_value,
                    len,
                );

                let raw_index = self.ir.untag(index_value);
                let payload_base = self.compile_string_builder_payload_base(sb);
                let byte = self
                    .ir
                    .heap_load_byte_with_reg_offset(payload_base, raw_index);
                let tagged_byte = self.ir.tag(byte, BuiltInTypes::Int.get_tag());
                self.ir.assign(result, tagged_byte);
                self.ir.jump(done);

                self.ir.write_label(out_of_bounds);
                self.ir.assign(result, Value::Null);
                self.ir.write_label(done);
                Value::Register(result)
            }
            "beagle.primitive/string-builder-set-byte-at!"
            | "beagle.string-builder/set-byte-at!" => {
                let sb = args[0];
                let index = args[1];
                let byte = args[2];

                let out_of_bounds = self.ir.label("sb_set_byte_at_oob");
                let done = self.ir.label("sb_set_byte_at_done");
                let result = self.ir.assign_new(sb);
                let sb_ptr = self.compile_string_builder_type_guard(sb, out_of_bounds);
                let len = self.ir.load_from_heap(sb_ptr, 2);
                let index_reg = self.ir.assign_new(index);
                let index_value = Value::Register(index_reg);

                self.ir.jump_if(
                    out_of_bounds,
                    Condition::LessThan,
                    index_value,
                    Value::TaggedConstant(0),
                );
                self.ir.jump_if(
                    out_of_bounds,
                    Condition::GreaterThanOrEqual,
                    index_value,
                    len,
                );

                let raw_index = self.ir.untag(index_value);
                let raw_byte = self.ir.untag(byte);
                let raw_byte = self.ir.and_imm(raw_byte, 0xFF);
                let payload_base = self.compile_string_builder_payload_base(sb);
                self.ir
                    .heap_store_byte_with_reg_offset(payload_base, raw_byte, raw_index);
                self.ir.jump(done);

                self.ir.write_label(out_of_bounds);
                self.ir.assign(result, Value::Null);
                self.ir.write_label(done);
                Value::Register(result)
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
                let _ = self.call_builtin("beagle.core/_println", vec![message]);
                let _ = self.call_builtin("beagle.builtin/throw-error", vec![]);
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

                // Card marking for generational GC write barrier
                let untagged = self.ir.untag(pointer);
                let write_barrier_fn =
                    Value::RawValue((write_barrier as usize) << BuiltInTypes::tag_size());
                self.ir
                    .call_builtin(write_barrier_fn, vec![untagged, value]);

                Value::Null
            }
            "beagle.primitive/__get-my-thread-obj" => {
                // Call the builtin that reads from thread_roots
                // Pass stack_pointer and frame_pointer so it can call __pause if needed
                let stack_pointer = self.ir.get_stack_pointer_imm(0);
                let frame_pointer = self.ir.get_frame_pointer();
                let func = self
                    .compiler
                    .get_function_by_name("beagle.builtin/__get_my_thread_obj")
                    .unwrap();
                let func_ptr = self.compiler.get_function_pointer(func.clone()).unwrap();
                let func_ptr = self.ir.assign_new(func_ptr);
                self.ir
                    .call_builtin(func_ptr.into(), vec![stack_pointer, frame_pointer])
            }
            "beagle.primitive/__pause" => {
                // Call __pause with stack_pointer and frame_pointer
                let stack_pointer = self.ir.get_stack_pointer_imm(0);
                let frame_pointer = self.ir.get_frame_pointer();
                let pause_function = self
                    .compiler
                    .get_function_by_name("beagle.builtin/__pause")
                    .unwrap();
                let pause_function = self
                    .compiler
                    .get_function_pointer(pause_function.clone())
                    .unwrap();
                let pause_function = self.ir.assign_new(pause_function);
                self.ir
                    .call_builtin(pause_function.into(), vec![stack_pointer, frame_pointer]);
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

    pub fn compile_macro_like_primitive(
        &mut self,
        name: String,
        args: Vec<Ast>,
    ) -> Result<Value, CompileError> {
        if name != "beagle.primitive/set!" {
            return Err(CompileError::InternalError {
                message: format!("Unknown macro-like primitive {}", name),
            });
        }

        if args.len() != 2 {
            return Err(CompileError::ArityMismatch {
                function_name: "set!".to_string(),
                expected: 2,
                got: args.len(),
                is_variadic: false,
            });
        }

        let property_access = &args[0];
        let Ast::PropertyAccess {
            object,
            property,
            token_range: _,
        } = property_access
        else {
            return Err(CompileError::InvalidAssignment {
                reason: "set! expects a property access (e.g., obj.field) as the first argument"
                    .to_string(),
            });
        };

        let object = object.deref();
        if !matches!(object, Ast::Identifier { .. }) {
            return Err(CompileError::InvalidAssignment {
                reason:
                    "set! expects an identifier as the object (e.g., obj.field, not expr.field)"
                        .to_string(),
            });
        }

        let property = property.deref();
        if !matches!(property, Ast::Identifier { .. }) {
            return Err(CompileError::InvalidAssignment {
                reason: "set! expects an identifier as the property name".to_string(),
            });
        }

        let object = self.call_compile(object)?;
        let value = self.call_compile(&args[1])?;
        // Value must be in a register for write_field_dynamic and call_builtin
        let value = self.ir.assign_new(value);

        let object = self.ir.assign_new(object);
        let untagged_object = self.ir.untag(object.into());
        // self.ir.breakpoint();
        let struct_id_versioned = self.ir.read_struct_id_versioned(untagged_object);
        let property_location = Value::RawValue(self.compiler.add_property_lookup().unwrap());
        let property_location = self.ir.assign_new(property_location);
        let property_value = self.ir.load_from_heap(property_location.into(), 0);
        let result = self.ir.assign_new(0);

        let exit_property_access = self.ir.label("exit_property_access");
        let slow_property_path = self.ir.label("slow_property_path");

        // Check if cached struct_id+version matches
        self.ir.jump_if(
            slow_property_path,
            Condition::NotEqual,
            struct_id_versioned,
            property_value,
        );

        // Check if field is mutable (cache[2] == 1)
        // Cache layout: [struct_id, field_offset, is_mutable]
        let is_mutable = self.ir.load_from_heap(property_location.into(), 2);
        let one = self.ir.assign_new(Value::RawValue(1));
        self.ir
            .jump_if(slow_property_path, Condition::NotEqual, is_mutable, one);

        let property_offset = self.ir.load_from_heap(property_location.into(), 1);
        self.ir
            .write_field_dynamic(untagged_object, property_offset, Value::Register(value));

        let write_barrier_fn =
            Value::RawValue((write_barrier as usize) << BuiltInTypes::tag_size());
        self.ir.call_builtin(
            write_barrier_fn,
            vec![untagged_object, Value::Register(value)],
        );

        self.ir.jump(exit_property_access);

        self.ir.write_label(slow_property_path);
        let Ast::Identifier(property, _) = property else {
            return Err(CompileError::ExpectedIdentifier {
                got: format!("{:?}", property),
            });
        };
        let property = property.clone();
        let constant_ptr = self.string_constant(property);
        let constant_ptr = self.ir.assign_new(constant_ptr);
        let call_result = self.call_builtin(
            "beagle.builtin/write-field",
            vec![
                object.into(),
                constant_ptr.into(),
                property_location.into(),
                Value::Register(value),
            ],
        )?;

        self.ir.assign(result, call_result);

        self.ir.write_label(exit_property_access);

        Ok(Value::Null)
    }
}
