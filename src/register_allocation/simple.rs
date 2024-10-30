// Allow unused in whole file
#![allow(unused)]
use std::collections::{HashMap, HashSet};

use crate::{
    common::Label,
    ir::{Instruction, Ir, VirtualRegister},
};

struct SimpleRegisterAllocator {
    lifetimes: HashMap<VirtualRegister, (usize, usize)>,
    instructions: Vec<Instruction>,
    allocated_registers: HashMap<VirtualRegister, VirtualRegister>,
    free_registers: Vec<VirtualRegister>,
    last_allocated_register: usize,
}

impl SimpleRegisterAllocator {
    fn new(instructions: Vec<Instruction>) -> Self {
        let max_register = instructions
            .iter()
            .map(|instruction| {
                instruction
                    .get_registers()
                    .iter()
                    .map(|register| register.index)
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);

        let lifetimes = Self::get_register_lifetime(&instructions);

        SimpleRegisterAllocator {
            lifetimes,
            instructions,
            allocated_registers: HashMap::new(),
            free_registers: vec![],
            last_allocated_register: max_register,
        }
    }
    fn get_free_register(&mut self) -> VirtualRegister {
        if let Some(register) = self.free_registers.pop() {
            return register;
        }

        self.last_allocated_register += 1;
        VirtualRegister {
            argument: None,
            index: self.last_allocated_register,
            volatile: true,
        }
    }

    pub fn simplify_registers(&mut self) {
        let mut cloned_instructions = self.instructions.clone();
        for (instruction_index, instruction) in cloned_instructions.iter_mut().enumerate() {
            for register in instruction.get_registers() {
                if let Some(allocated_register) = self.allocated_registers.get(&register) {
                    instruction.replace_register(register, *allocated_register);
                }
                let lifetime = self.lifetimes.get(&register).cloned();

                if let Some((start, end)) = lifetime {
                    if start == instruction_index {
                        let new_register = self.get_free_register();
                        self.allocated_registers.insert(register, new_register);
                        instruction.replace_register(register, new_register);
                    }

                    if end == instruction_index {
                        let allocated_register = self.allocated_registers.get(&register).unwrap();
                        self.free_registers.push(*allocated_register);
                        self.allocated_registers.remove(&register);
                    }
                }
            }
        }
        self.instructions = cloned_instructions;
    }

    fn number_of_distinct_registers(&self) -> usize {
        self.instructions
            .iter()
            .flat_map(|instruction| instruction.get_registers())
            .collect::<HashSet<_>>()
            .iter()
            .collect::<Vec<_>>()
            .len()
    }

    fn get_register_lifetime(
        instructions: &[Instruction],
    ) -> HashMap<VirtualRegister, (usize, usize)> {
        let mut result: HashMap<VirtualRegister, (usize, usize)> = HashMap::new();
        for (index, instruction) in instructions.iter().enumerate().rev() {
            for register in instruction.get_registers() {
                if let Some((_start, end)) = result.get(&register) {
                    result.insert(register, (index, *end));
                } else {
                    result.insert(register, (index, index));
                }
            }
        }
        result
    }
}

// I'm not 100% sure what the output should be
// One way to try to think of this is that I am mapping
// between virtual registers and physical registers
// But I might need to change instructions as well
// Because, I need to spill some of these registers
// to memory, when I spill I need to update the instructions
// Given my setup, that will change not only instructions,
// but also the lifetimes of the registers.
// One way to get around this would be to have some instruction
// that just was this compound thing.
// Then the instructions don't change length
// Another answer would be to have a different register
// type so that way instructions when they get compiled
// would check the register type and do the right thing.

// Another thing I want to consider, is that I've found
// it useful to let instructions have some temporary registers.
// But I didn't express that in anyway here.
// I could make it so that all instructions have some field that says
// here are your extra registers. But that feels a bit gross.
// Doing it one the fly is nice, but then how can my register allocator know?

// I am currently doing register allocation on the fly as I compile to machine code
// But if I compiled to machine code, and then did register allocation, I would
// know all these things like auxiliary registers and stuff.

// I also need to account for argument registers
// Right now I am messing them up when it comes to call
// At this level here I'm not sure if I should be thinking about specific instructions
// or if I'm somehow able to abstract that away. The problem is that things like
// return value of course have to be allocated to a specific register.

// One option for this is just to remap between virtual registers and make sure
// that not too many are living at one time. Then my existing system should work.
// That doesn't answer the argument register problem. But that's why I think I can
// solve as well.

macro_rules! parse_lifetime {
    ($($register:ident {
        argument: $argument:expr,
        index: $index:expr,
        volatile: $volatile:expr,
    }: ($first:expr, $second:expr,),)*) => {
        {
            let mut map : HashMap<VirtualRegister, (usize, usize)> = HashMap::new();
            $(
                map.insert(
                    VirtualRegister {
                        argument: $argument,
                        index: $index,
                        volatile: $volatile,
                    },
                    ($first, $second,),
                );
            )*
            map
        }
    };
}
#[test]
fn big_test() {
    use crate::ir::Condition::*;
    use crate::ir::Instruction::*;
    use crate::ir::Value::*;
    let instructions = vec![
        Assign(
            VirtualRegister {
                argument: None,
                index: 0,
                volatile: false,
            },
            RawValue(6102964528),
        ),
        AtomicLoad(
            Register(VirtualRegister {
                argument: None,
                index: 1,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 0,
                volatile: false,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 2,
                volatile: false,
            },
            RawValue(0),
        ),
        JumpIf(
            Label { index: 1 },
            Equal,
            Register(VirtualRegister {
                argument: None,
                index: 1,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 2,
                volatile: false,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 3,
                volatile: false,
            },
            Pointer(6102963480),
        ),
        GetStackPointerImm(
            Register(VirtualRegister {
                argument: None,
                index: 4,
                volatile: true,
            }),
            0,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 5,
                volatile: false,
            },
            TaggedConstant(4364051508),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 6,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 5,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 3,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 4,
                    volatile: true,
                }),
            ],
            true,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 7,
                volatile: false,
            },
            StringConstantPtr(60),
        ),
        LoadConstant(
            Register(VirtualRegister {
                argument: None,
                index: 8,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 7,
                volatile: false,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 9,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 10,
                volatile: false,
            },
            Pointer(4550197368),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 11,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 10,
                volatile: false,
            }),
            0,
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 12,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 11,
                volatile: true,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 9,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 8,
                    volatile: true,
                }),
            ],
            true,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 13,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 12,
                volatile: true,
            }),
        ),
        StoreLocal(
            Local(0),
            Register(VirtualRegister {
                argument: None,
                index: 13,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 14,
                volatile: true,
            },
            Local(0),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 15,
                volatile: false,
            },
            StringConstantPtr(61),
        ),
        LoadConstant(
            Register(VirtualRegister {
                argument: None,
                index: 16,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 15,
                volatile: false,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 17,
                volatile: false,
            },
            RawValue(1),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 18,
                volatile: false,
            },
            RawValue(6),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 19,
                volatile: false,
            },
            TaggedConstant(4364051404),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 20,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 21,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 19,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 20,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 18,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 17,
                    volatile: false,
                }),
            ],
            false,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 22,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 21,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 23,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 22,
                volatile: true,
            }),
            0,
        ),
        AndImm(
            Register(VirtualRegister {
                argument: None,
                index: 24,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 23,
                volatile: true,
            }),
            72057594021150720,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 26,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 25,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 24,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 26,
                volatile: false,
            }),
        ),
        ShiftRightImm(
            Register(VirtualRegister {
                argument: None,
                index: 27,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 25,
                volatile: true,
            }),
            24,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 28,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 27,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 29,
                volatile: false,
            },
            RawValue(4550214208),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 30,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 29,
                volatile: false,
            }),
            0,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 31,
                volatile: false,
            },
            TaggedConstant(0),
        ),
        JumpIf(
            Label { index: 3 },
            NotEqual,
            Register(VirtualRegister {
                argument: None,
                index: 28,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 30,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 32,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 29,
                volatile: false,
            }),
            1,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 34,
                volatile: false,
            },
            TaggedConstant(1),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 33,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 32,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 34,
                volatile: false,
            }),
        ),
        HeapLoadReg(
            Register(VirtualRegister {
                argument: None,
                index: 35,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 22,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 33,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 31,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 35,
                volatile: true,
            }),
        ),
        Jump(Label { index: 2 }),
        Assign(
            VirtualRegister {
                argument: None,
                index: 36,
                volatile: false,
            },
            StringConstantPtr(62),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 37,
                volatile: false,
            },
            TaggedConstant(4364050776),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 38,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 39,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 37,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 38,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 21,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 36,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 29,
                    volatile: false,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 31,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 39,
                volatile: true,
            }),
        ),
        PushStack(Register(VirtualRegister {
            argument: None,
            index: 31,
            volatile: false,
        })),
        Assign(
            VirtualRegister {
                argument: None,
                index: 40,
                volatile: false,
            },
            TaggedConstant(4550064244),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 41,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 40,
                volatile: false,
            }),
            vec![],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 42,
                volatile: false,
            },
            TaggedConstant(4550046884),
        ),
        CurrentStackPosition(Register(VirtualRegister {
            argument: None,
            index: 43,
            volatile: true,
        })),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 44,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 43,
                volatile: true,
            }),
            1,
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 45,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 42,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 41,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 44,
                    volatile: true,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 41,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 45,
                volatile: true,
            }),
        ),
        PopStack(Register(VirtualRegister {
            argument: None,
            index: 46,
            volatile: true,
        })),
        Assign(
            VirtualRegister {
                argument: None,
                index: 47,
                volatile: false,
            },
            RawValue(1),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 48,
                volatile: false,
            },
            RawValue(6),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 49,
                volatile: false,
            },
            TaggedConstant(4364051404),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 50,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 51,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 49,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 50,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 48,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 47,
                    volatile: false,
                }),
            ],
            false,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 52,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 51,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 53,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 52,
                volatile: true,
            }),
            0,
        ),
        AndImm(
            Register(VirtualRegister {
                argument: None,
                index: 54,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 53,
                volatile: true,
            }),
            72057594021150720,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 56,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 55,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 54,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 56,
                volatile: false,
            }),
        ),
        ShiftRightImm(
            Register(VirtualRegister {
                argument: None,
                index: 57,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 55,
                volatile: true,
            }),
            24,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 58,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 57,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 59,
                volatile: false,
            },
            RawValue(4550214224),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 60,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 59,
                volatile: false,
            }),
            0,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 61,
                volatile: false,
            },
            TaggedConstant(0),
        ),
        JumpIf(
            Label { index: 5 },
            NotEqual,
            Register(VirtualRegister {
                argument: None,
                index: 58,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 60,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 62,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 59,
                volatile: false,
            }),
            1,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 64,
                volatile: false,
            },
            TaggedConstant(1),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 63,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 62,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 64,
                volatile: false,
            }),
        ),
        HeapLoadReg(
            Register(VirtualRegister {
                argument: None,
                index: 65,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 52,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 63,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 61,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 65,
                volatile: true,
            }),
        ),
        Jump(Label { index: 4 }),
        Assign(
            VirtualRegister {
                argument: None,
                index: 66,
                volatile: false,
            },
            StringConstantPtr(63),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 67,
                volatile: false,
            },
            TaggedConstant(4364050776),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 68,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 69,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 67,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 68,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 51,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 66,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 59,
                    volatile: false,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 61,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 69,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 70,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 71,
                volatile: false,
            },
            Pointer(4550197376),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 72,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 71,
                volatile: false,
            }),
            0,
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 73,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 72,
                volatile: true,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 70,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 14,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 16,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 41,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 61,
                    volatile: false,
                }),
            ],
            true,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 74,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 73,
                volatile: true,
            }),
        ),
        StoreLocal(
            Local(1),
            Register(VirtualRegister {
                argument: None,
                index: 74,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 75,
                volatile: true,
            },
            TaggedConstant(0),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 77,
                volatile: true,
            },
            Local(1),
        ),
        GetTag(
            Register(VirtualRegister {
                argument: None,
                index: 78,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 77,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 79,
                volatile: false,
            },
            RawValue(5),
        ),
        JumpIf(
            Label { index: 6 },
            NotEqual,
            Register(VirtualRegister {
                argument: None,
                index: 78,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 79,
                volatile: false,
            }),
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 80,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 77,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 81,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 80,
                volatile: true,
            }),
            1,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 76,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 81,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 82,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 80,
                volatile: true,
            }),
            2,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 84,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 83,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 82,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 84,
                volatile: false,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 85,
                volatile: true,
            },
            TaggedConstant(0),
        ),
        JumpIf(
            Label { index: 7 },
            GreaterThanOrEqual,
            Register(VirtualRegister {
                argument: None,
                index: 85,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 83,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 87,
                volatile: false,
            },
            TaggedConstant(4),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 86,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 85,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 87,
                volatile: false,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 89,
                volatile: false,
            },
            TaggedConstant(8),
        ),
        Mul(
            Register(VirtualRegister {
                argument: None,
                index: 88,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 86,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 89,
                volatile: false,
            }),
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 90,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 88,
                volatile: true,
            }),
        ),
        HeapLoadReg(
            Register(VirtualRegister {
                argument: None,
                index: 91,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 80,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 90,
                volatile: true,
            }),
        ),
        Sub(
            Register(VirtualRegister {
                argument: None,
                index: 92,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 83,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 85,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 93,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 80,
                volatile: true,
            }),
            3,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 95,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 94,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 93,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 95,
                volatile: false,
            }),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 96,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 92,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 94,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 98,
                volatile: false,
            },
            TaggedConstant(-8),
        ),
        Mul(
            Register(VirtualRegister {
                argument: None,
                index: 97,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 96,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 98,
                volatile: false,
            }),
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 99,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 97,
                volatile: true,
            }),
        ),
        GetStackPointerImm(
            Register(VirtualRegister {
                argument: None,
                index: 100,
                volatile: true,
            }),
            2,
        ),
        HeapStoreOffsetReg(
            Register(VirtualRegister {
                argument: None,
                index: 100,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 91,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 99,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 102,
                volatile: false,
            },
            TaggedConstant(1),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 101,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 102,
                volatile: false,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 85,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 85,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 101,
                volatile: true,
            }),
        ),
        Jump(Label { index: 8 }),
        ExtendLifeTime(Register(VirtualRegister {
            argument: None,
            index: 83,
            volatile: true,
        })),
        ExtendLifeTime(Register(VirtualRegister {
            argument: None,
            index: 85,
            volatile: true,
        })),
        ExtendLifeTime(Register(VirtualRegister {
            argument: None,
            index: 80,
            volatile: true,
        })),
        Assign(
            VirtualRegister {
                argument: None,
                index: 76,
                volatile: true,
            },
            Local(1),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 103,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 76,
                volatile: true,
            }),
            vec![Register(VirtualRegister {
                argument: None,
                index: 75,
                volatile: true,
            })],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 104,
                volatile: true,
            },
            Local(0),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 105,
                volatile: false,
            },
            StringConstantPtr(64),
        ),
        LoadConstant(
            Register(VirtualRegister {
                argument: None,
                index: 106,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 105,
                volatile: false,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 107,
                volatile: false,
            },
            RawValue(1),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 108,
                volatile: false,
            },
            RawValue(6),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 109,
                volatile: false,
            },
            TaggedConstant(4364051404),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 110,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 111,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 109,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 110,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 108,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 107,
                    volatile: false,
                }),
            ],
            false,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 112,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 111,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 113,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 112,
                volatile: true,
            }),
            0,
        ),
        AndImm(
            Register(VirtualRegister {
                argument: None,
                index: 114,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 113,
                volatile: true,
            }),
            72057594021150720,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 116,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 115,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 114,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 116,
                volatile: false,
            }),
        ),
        ShiftRightImm(
            Register(VirtualRegister {
                argument: None,
                index: 117,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 115,
                volatile: true,
            }),
            24,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 118,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 117,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 119,
                volatile: false,
            },
            RawValue(4550214240),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 120,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 119,
                volatile: false,
            }),
            0,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 121,
                volatile: false,
            },
            TaggedConstant(0),
        ),
        JumpIf(
            Label { index: 11 },
            NotEqual,
            Register(VirtualRegister {
                argument: None,
                index: 118,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 120,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 122,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 119,
                volatile: false,
            }),
            1,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 124,
                volatile: false,
            },
            TaggedConstant(1),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 123,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 122,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 124,
                volatile: false,
            }),
        ),
        HeapLoadReg(
            Register(VirtualRegister {
                argument: None,
                index: 125,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 112,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 123,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 121,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 125,
                volatile: true,
            }),
        ),
        Jump(Label { index: 10 }),
        Assign(
            VirtualRegister {
                argument: None,
                index: 126,
                volatile: false,
            },
            StringConstantPtr(65),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 127,
                volatile: false,
            },
            TaggedConstant(4364050776),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 128,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 129,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 127,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 128,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 111,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 126,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 119,
                    volatile: false,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 121,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 129,
                volatile: true,
            }),
        ),
        PushStack(Register(VirtualRegister {
            argument: None,
            index: 121,
            volatile: false,
        })),
        Assign(
            VirtualRegister {
                argument: None,
                index: 130,
                volatile: false,
            },
            RawValue(1),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 131,
                volatile: false,
            },
            RawValue(6),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 132,
                volatile: false,
            },
            TaggedConstant(4364051404),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 133,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 134,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 132,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 133,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 131,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 130,
                    volatile: false,
                }),
            ],
            false,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 135,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 134,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 136,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 135,
                volatile: true,
            }),
            0,
        ),
        AndImm(
            Register(VirtualRegister {
                argument: None,
                index: 137,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 136,
                volatile: true,
            }),
            72057594021150720,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 139,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 138,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 137,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 139,
                volatile: false,
            }),
        ),
        ShiftRightImm(
            Register(VirtualRegister {
                argument: None,
                index: 140,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 138,
                volatile: true,
            }),
            24,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 141,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 140,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 142,
                volatile: false,
            },
            RawValue(4550214256),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 143,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 142,
                volatile: false,
            }),
            0,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 144,
                volatile: false,
            },
            TaggedConstant(0),
        ),
        JumpIf(
            Label { index: 13 },
            NotEqual,
            Register(VirtualRegister {
                argument: None,
                index: 141,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 143,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 145,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 142,
                volatile: false,
            }),
            1,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 147,
                volatile: false,
            },
            TaggedConstant(1),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 146,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 145,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 147,
                volatile: false,
            }),
        ),
        HeapLoadReg(
            Register(VirtualRegister {
                argument: None,
                index: 148,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 135,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 146,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 144,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 148,
                volatile: true,
            }),
        ),
        Jump(Label { index: 12 }),
        Assign(
            VirtualRegister {
                argument: None,
                index: 149,
                volatile: false,
            },
            StringConstantPtr(66),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 150,
                volatile: false,
            },
            TaggedConstant(4364050776),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 151,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 152,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 150,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 151,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 134,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 149,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 142,
                    volatile: false,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 144,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 152,
                volatile: true,
            }),
        ),
        PushStack(Register(VirtualRegister {
            argument: None,
            index: 144,
            volatile: false,
        })),
        Assign(
            VirtualRegister {
                argument: None,
                index: 153,
                volatile: false,
            },
            RawValue(1),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 154,
                volatile: false,
            },
            RawValue(6),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 155,
                volatile: false,
            },
            TaggedConstant(4364051404),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 156,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 157,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 155,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 156,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 154,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 153,
                    volatile: false,
                }),
            ],
            false,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 158,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 157,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 159,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 158,
                volatile: true,
            }),
            0,
        ),
        AndImm(
            Register(VirtualRegister {
                argument: None,
                index: 160,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 159,
                volatile: true,
            }),
            72057594021150720,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 162,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 161,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 160,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 162,
                volatile: false,
            }),
        ),
        ShiftRightImm(
            Register(VirtualRegister {
                argument: None,
                index: 163,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 161,
                volatile: true,
            }),
            24,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 164,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 163,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 165,
                volatile: false,
            },
            RawValue(4550214272),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 166,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 165,
                volatile: false,
            }),
            0,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 167,
                volatile: false,
            },
            TaggedConstant(0),
        ),
        JumpIf(
            Label { index: 15 },
            NotEqual,
            Register(VirtualRegister {
                argument: None,
                index: 164,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 166,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 168,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 165,
                volatile: false,
            }),
            1,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 170,
                volatile: false,
            },
            TaggedConstant(1),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 169,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 168,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 170,
                volatile: false,
            }),
        ),
        HeapLoadReg(
            Register(VirtualRegister {
                argument: None,
                index: 171,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 158,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 169,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 167,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 171,
                volatile: true,
            }),
        ),
        Jump(Label { index: 14 }),
        Assign(
            VirtualRegister {
                argument: None,
                index: 172,
                volatile: false,
            },
            StringConstantPtr(67),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 173,
                volatile: false,
            },
            TaggedConstant(4364050776),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 174,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 175,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 173,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 174,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 157,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 172,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 165,
                    volatile: false,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 167,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 175,
                volatile: true,
            }),
        ),
        PushStack(Register(VirtualRegister {
            argument: None,
            index: 167,
            volatile: false,
        })),
        Assign(
            VirtualRegister {
                argument: None,
                index: 176,
                volatile: false,
            },
            RawValue(1),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 177,
                volatile: false,
            },
            RawValue(6),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 178,
                volatile: false,
            },
            TaggedConstant(4364051404),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 179,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 180,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 178,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 179,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 177,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 176,
                    volatile: false,
                }),
            ],
            false,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 181,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 180,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 182,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 181,
                volatile: true,
            }),
            0,
        ),
        AndImm(
            Register(VirtualRegister {
                argument: None,
                index: 183,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 182,
                volatile: true,
            }),
            72057594021150720,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 185,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 184,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 183,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 185,
                volatile: false,
            }),
        ),
        ShiftRightImm(
            Register(VirtualRegister {
                argument: None,
                index: 186,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 184,
                volatile: true,
            }),
            24,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 187,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 186,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 188,
                volatile: false,
            },
            RawValue(4550214288),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 189,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 188,
                volatile: false,
            }),
            0,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 190,
                volatile: false,
            },
            TaggedConstant(0),
        ),
        JumpIf(
            Label { index: 17 },
            NotEqual,
            Register(VirtualRegister {
                argument: None,
                index: 187,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 189,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 191,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 188,
                volatile: false,
            }),
            1,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 193,
                volatile: false,
            },
            TaggedConstant(1),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 192,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 191,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 193,
                volatile: false,
            }),
        ),
        HeapLoadReg(
            Register(VirtualRegister {
                argument: None,
                index: 194,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 181,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 192,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 190,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 194,
                volatile: true,
            }),
        ),
        Jump(Label { index: 16 }),
        Assign(
            VirtualRegister {
                argument: None,
                index: 195,
                volatile: false,
            },
            StringConstantPtr(68),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 196,
                volatile: false,
            },
            TaggedConstant(4364050776),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 197,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 198,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 196,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 197,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 180,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 195,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 188,
                    volatile: false,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 190,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 198,
                volatile: true,
            }),
        ),
        PushStack(Register(VirtualRegister {
            argument: None,
            index: 190,
            volatile: false,
        })),
        Assign(
            VirtualRegister {
                argument: None,
                index: 199,
                volatile: false,
            },
            RawValue(1),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 200,
                volatile: false,
            },
            RawValue(6),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 201,
                volatile: false,
            },
            TaggedConstant(4364051404),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 202,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 203,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 201,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 202,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 200,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 199,
                    volatile: false,
                }),
            ],
            false,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 204,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 203,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 205,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 204,
                volatile: true,
            }),
            0,
        ),
        AndImm(
            Register(VirtualRegister {
                argument: None,
                index: 206,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 205,
                volatile: true,
            }),
            72057594021150720,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 208,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 207,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 206,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 208,
                volatile: false,
            }),
        ),
        ShiftRightImm(
            Register(VirtualRegister {
                argument: None,
                index: 209,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 207,
                volatile: true,
            }),
            24,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 210,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 209,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 211,
                volatile: false,
            },
            RawValue(4550214304),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 212,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 211,
                volatile: false,
            }),
            0,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 213,
                volatile: false,
            },
            TaggedConstant(0),
        ),
        JumpIf(
            Label { index: 19 },
            NotEqual,
            Register(VirtualRegister {
                argument: None,
                index: 210,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 212,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 214,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 211,
                volatile: false,
            }),
            1,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 216,
                volatile: false,
            },
            TaggedConstant(1),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 215,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 214,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 216,
                volatile: false,
            }),
        ),
        HeapLoadReg(
            Register(VirtualRegister {
                argument: None,
                index: 217,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 204,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 215,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 213,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 217,
                volatile: true,
            }),
        ),
        Jump(Label { index: 18 }),
        Assign(
            VirtualRegister {
                argument: None,
                index: 218,
                volatile: false,
            },
            StringConstantPtr(69),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 219,
                volatile: false,
            },
            TaggedConstant(4364050776),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 220,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 221,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 219,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 220,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 203,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 218,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 211,
                    volatile: false,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 213,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 221,
                volatile: true,
            }),
        ),
        PushStack(Register(VirtualRegister {
            argument: None,
            index: 213,
            volatile: false,
        })),
        Assign(
            VirtualRegister {
                argument: None,
                index: 222,
                volatile: false,
            },
            RawValue(1),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 223,
                volatile: false,
            },
            RawValue(6),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 224,
                volatile: false,
            },
            TaggedConstant(4364051404),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 225,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 226,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 224,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 225,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 223,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 222,
                    volatile: false,
                }),
            ],
            false,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 227,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 226,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 228,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 227,
                volatile: true,
            }),
            0,
        ),
        AndImm(
            Register(VirtualRegister {
                argument: None,
                index: 229,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 228,
                volatile: true,
            }),
            72057594021150720,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 231,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 230,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 229,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 231,
                volatile: false,
            }),
        ),
        ShiftRightImm(
            Register(VirtualRegister {
                argument: None,
                index: 232,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 230,
                volatile: true,
            }),
            24,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 233,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 232,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 234,
                volatile: false,
            },
            RawValue(4550214320),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 235,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 234,
                volatile: false,
            }),
            0,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 236,
                volatile: false,
            },
            TaggedConstant(0),
        ),
        JumpIf(
            Label { index: 21 },
            NotEqual,
            Register(VirtualRegister {
                argument: None,
                index: 233,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 235,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 237,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 234,
                volatile: false,
            }),
            1,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 239,
                volatile: false,
            },
            TaggedConstant(1),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 238,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 237,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 239,
                volatile: false,
            }),
        ),
        HeapLoadReg(
            Register(VirtualRegister {
                argument: None,
                index: 240,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 227,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 238,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 236,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 240,
                volatile: true,
            }),
        ),
        Jump(Label { index: 20 }),
        Assign(
            VirtualRegister {
                argument: None,
                index: 241,
                volatile: false,
            },
            StringConstantPtr(70),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 242,
                volatile: false,
            },
            TaggedConstant(4364050776),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 243,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 244,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 242,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 243,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 226,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 241,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 234,
                    volatile: false,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 236,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 244,
                volatile: true,
            }),
        ),
        PushStack(Register(VirtualRegister {
            argument: None,
            index: 236,
            volatile: false,
        })),
        Assign(
            VirtualRegister {
                argument: None,
                index: 245,
                volatile: false,
            },
            TaggedConstant(4550064244),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 246,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 245,
                volatile: false,
            }),
            vec![],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 247,
                volatile: false,
            },
            TaggedConstant(4550046884),
        ),
        CurrentStackPosition(Register(VirtualRegister {
            argument: None,
            index: 248,
            volatile: true,
        })),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 249,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 248,
                volatile: true,
            }),
            6,
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 250,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 247,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 246,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 249,
                    volatile: true,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 246,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 250,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 251,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 248,
                volatile: true,
            }),
            5,
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 252,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 247,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 246,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 251,
                    volatile: true,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 246,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 252,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 253,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 248,
                volatile: true,
            }),
            4,
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 254,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 247,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 246,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 253,
                    volatile: true,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 246,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 254,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 255,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 248,
                volatile: true,
            }),
            3,
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 256,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 247,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 246,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 255,
                    volatile: true,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 246,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 256,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 257,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 248,
                volatile: true,
            }),
            2,
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 258,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 247,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 246,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 257,
                    volatile: true,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 246,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 258,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 259,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 248,
                volatile: true,
            }),
            1,
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 260,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 247,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 246,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 259,
                    volatile: true,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 246,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 260,
                volatile: true,
            }),
        ),
        PopStack(Register(VirtualRegister {
            argument: None,
            index: 261,
            volatile: true,
        })),
        PopStack(Register(VirtualRegister {
            argument: None,
            index: 262,
            volatile: true,
        })),
        PopStack(Register(VirtualRegister {
            argument: None,
            index: 263,
            volatile: true,
        })),
        PopStack(Register(VirtualRegister {
            argument: None,
            index: 264,
            volatile: true,
        })),
        PopStack(Register(VirtualRegister {
            argument: None,
            index: 265,
            volatile: true,
        })),
        PopStack(Register(VirtualRegister {
            argument: None,
            index: 266,
            volatile: true,
        })),
        Assign(
            VirtualRegister {
                argument: None,
                index: 267,
                volatile: false,
            },
            RawValue(1),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 268,
                volatile: false,
            },
            RawValue(6),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 269,
                volatile: false,
            },
            TaggedConstant(4364051404),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 270,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 271,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 269,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 270,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 268,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 267,
                    volatile: false,
                }),
            ],
            false,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 272,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 271,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 273,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 272,
                volatile: true,
            }),
            0,
        ),
        AndImm(
            Register(VirtualRegister {
                argument: None,
                index: 274,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 273,
                volatile: true,
            }),
            72057594021150720,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 276,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 275,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 274,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 276,
                volatile: false,
            }),
        ),
        ShiftRightImm(
            Register(VirtualRegister {
                argument: None,
                index: 277,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 275,
                volatile: true,
            }),
            24,
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 278,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 277,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 279,
                volatile: false,
            },
            RawValue(4550214336),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 280,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 279,
                volatile: false,
            }),
            0,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 281,
                volatile: false,
            },
            TaggedConstant(0),
        ),
        JumpIf(
            Label { index: 23 },
            NotEqual,
            Register(VirtualRegister {
                argument: None,
                index: 278,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 280,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 282,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 279,
                volatile: false,
            }),
            1,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 284,
                volatile: false,
            },
            TaggedConstant(1),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 283,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 282,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 284,
                volatile: false,
            }),
        ),
        HeapLoadReg(
            Register(VirtualRegister {
                argument: None,
                index: 285,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 272,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 283,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 281,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 285,
                volatile: true,
            }),
        ),
        Jump(Label { index: 22 }),
        Assign(
            VirtualRegister {
                argument: None,
                index: 286,
                volatile: false,
            },
            StringConstantPtr(71),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 287,
                volatile: false,
            },
            TaggedConstant(4364050776),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 288,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 289,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 287,
                volatile: false,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 288,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 271,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 286,
                    volatile: false,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 279,
                    volatile: false,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 281,
                volatile: false,
            },
            Register(VirtualRegister {
                argument: None,
                index: 289,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 290,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 291,
                volatile: false,
            },
            Pointer(4550197376),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 292,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 291,
                volatile: false,
            }),
            0,
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 293,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 292,
                volatile: true,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 290,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 104,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 106,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 246,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 281,
                    volatile: false,
                }),
            ],
            true,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 294,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 293,
                volatile: true,
            }),
        ),
        StoreLocal(
            Local(2),
            Register(VirtualRegister {
                argument: None,
                index: 294,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 295,
                volatile: false,
            },
            StringConstantPtr(72),
        ),
        LoadConstant(
            Register(VirtualRegister {
                argument: None,
                index: 296,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 295,
                volatile: false,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 297,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 296,
                volatile: true,
            }),
        ),
        StoreLocal(
            Local(3),
            Register(VirtualRegister {
                argument: None,
                index: 297,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 298,
                volatile: true,
            },
            Local(3),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 299,
                volatile: true,
            },
            TaggedConstant(100),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 300,
                volatile: true,
            },
            TaggedConstant(100),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 301,
                volatile: true,
            },
            TaggedConstant(640),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 302,
                volatile: true,
            },
            TaggedConstant(480),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 303,
                volatile: true,
            },
            TaggedConstant(0),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 305,
                volatile: true,
            },
            Local(2),
        ),
        GetTag(
            Register(VirtualRegister {
                argument: None,
                index: 306,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 305,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 307,
                volatile: false,
            },
            RawValue(5),
        ),
        JumpIf(
            Label { index: 24 },
            NotEqual,
            Register(VirtualRegister {
                argument: None,
                index: 306,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 307,
                volatile: false,
            }),
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 308,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 305,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 309,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 308,
                volatile: true,
            }),
            1,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 304,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 309,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 310,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 308,
                volatile: true,
            }),
            2,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 312,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 311,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 310,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 312,
                volatile: false,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 313,
                volatile: true,
            },
            TaggedConstant(0),
        ),
        JumpIf(
            Label { index: 25 },
            GreaterThanOrEqual,
            Register(VirtualRegister {
                argument: None,
                index: 313,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 311,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 315,
                volatile: false,
            },
            TaggedConstant(4),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 314,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 313,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 315,
                volatile: false,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 317,
                volatile: false,
            },
            TaggedConstant(8),
        ),
        Mul(
            Register(VirtualRegister {
                argument: None,
                index: 316,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 314,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 317,
                volatile: false,
            }),
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 318,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 316,
                volatile: true,
            }),
        ),
        HeapLoadReg(
            Register(VirtualRegister {
                argument: None,
                index: 319,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 308,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 318,
                volatile: true,
            }),
        ),
        Sub(
            Register(VirtualRegister {
                argument: None,
                index: 320,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 311,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 313,
                volatile: true,
            }),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 321,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 308,
                volatile: true,
            }),
            3,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 323,
                volatile: false,
            },
            RawValue(0),
        ),
        Tag(
            Register(VirtualRegister {
                argument: None,
                index: 322,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 321,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 323,
                volatile: false,
            }),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 324,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 320,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 322,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 326,
                volatile: false,
            },
            TaggedConstant(-8),
        ),
        Mul(
            Register(VirtualRegister {
                argument: None,
                index: 325,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 324,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 326,
                volatile: false,
            }),
        ),
        Untag(
            Register(VirtualRegister {
                argument: None,
                index: 327,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 325,
                volatile: true,
            }),
        ),
        GetStackPointerImm(
            Register(VirtualRegister {
                argument: None,
                index: 328,
                volatile: true,
            }),
            2,
        ),
        HeapStoreOffsetReg(
            Register(VirtualRegister {
                argument: None,
                index: 328,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 319,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 327,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 330,
                volatile: false,
            },
            TaggedConstant(1),
        ),
        AddInt(
            Register(VirtualRegister {
                argument: None,
                index: 329,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 330,
                volatile: false,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 313,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 313,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 329,
                volatile: true,
            }),
        ),
        Jump(Label { index: 26 }),
        ExtendLifeTime(Register(VirtualRegister {
            argument: None,
            index: 311,
            volatile: true,
        })),
        ExtendLifeTime(Register(VirtualRegister {
            argument: None,
            index: 313,
            volatile: true,
        })),
        ExtendLifeTime(Register(VirtualRegister {
            argument: None,
            index: 308,
            volatile: true,
        })),
        Assign(
            VirtualRegister {
                argument: None,
                index: 304,
                volatile: true,
            },
            Local(2),
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 331,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 304,
                volatile: true,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 298,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 299,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 300,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 301,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 302,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 303,
                    volatile: true,
                }),
            ],
            false,
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 332,
                volatile: true,
            },
            Register(VirtualRegister {
                argument: None,
                index: 331,
                volatile: true,
            }),
        ),
        StoreLocal(
            Local(4),
            Register(VirtualRegister {
                argument: None,
                index: 332,
                volatile: true,
            }),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 333,
                volatile: true,
            },
            Local(0),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 334,
                volatile: true,
            },
            Pointer(6102963480),
        ),
        Assign(
            VirtualRegister {
                argument: None,
                index: 335,
                volatile: false,
            },
            Pointer(4550197256),
        ),
        HeapLoad(
            Register(VirtualRegister {
                argument: None,
                index: 336,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 335,
                volatile: false,
            }),
            0,
        ),
        Call(
            Register(VirtualRegister {
                argument: None,
                index: 337,
                volatile: true,
            }),
            Register(VirtualRegister {
                argument: None,
                index: 336,
                volatile: true,
            }),
            vec![
                Register(VirtualRegister {
                    argument: None,
                    index: 334,
                    volatile: true,
                }),
                Register(VirtualRegister {
                    argument: None,
                    index: 333,
                    volatile: true,
                }),
            ],
            true,
        ),
        Ret(Register(VirtualRegister {
            argument: None,
            index: 337,
            volatile: true,
        })),
    ];

    let mut simple_register_allocator = SimpleRegisterAllocator::new(instructions);
    println!(
        "{:?}",
        simple_register_allocator.number_of_distinct_registers()
    );
    simple_register_allocator.simplify_registers();
    // println!("{:#?}", simple_register_allocator.instructions);
    println!(
        "{:?}",
        simple_register_allocator.number_of_distinct_registers()
    );
    let new_lifetimes =
        SimpleRegisterAllocator::get_register_lifetime(&simple_register_allocator.instructions);
    Ir::draw_lifetimes(&new_lifetimes);
    // This goes from 338 to 15
    // But ignoring argument registers we only have 10
    // We need to now add in logic where if we go over the amount we have,
    // we need to spill the register
    // We also should probably deal with argument registers and such.
    // But we are making progress
    // I could now that I have simplified things,
}
