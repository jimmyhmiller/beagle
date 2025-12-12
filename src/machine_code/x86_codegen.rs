//! x86-64 instruction encoder for Beagle
//!
//! This module provides types and functions for encoding x86-64 machine code.
//! It covers the subset of instructions needed by the Beagle compiler.

#![allow(dead_code)]
#![allow(clippy::identity_op)]

use std::ops::Shl;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Size {
    S8,
    S16,
    S32,
    S64,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct X86Register {
    pub size: Size,
    pub index: u8,
}

impl X86Register {
    pub fn encode(&self) -> u8 {
        self.index
    }

    pub fn from_index(index: usize) -> X86Register {
        X86Register {
            index: index as u8,
            size: Size::S64,
        }
    }

    /// Returns true if this register requires REX.B or REX.R extension
    pub fn needs_rex_ext(&self) -> bool {
        self.index >= 8
    }
}

impl Shl<u32> for &X86Register {
    type Output = u32;

    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}

impl Shl<u32> for X86Register {
    type Output = u32;

    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}

// 64-bit register constants
pub const RAX: X86Register = X86Register {
    size: Size::S64,
    index: 0,
};
pub const RCX: X86Register = X86Register {
    size: Size::S64,
    index: 1,
};
pub const RDX: X86Register = X86Register {
    size: Size::S64,
    index: 2,
};
pub const RBX: X86Register = X86Register {
    size: Size::S64,
    index: 3,
};
pub const RSP: X86Register = X86Register {
    size: Size::S64,
    index: 4,
};
pub const RBP: X86Register = X86Register {
    size: Size::S64,
    index: 5,
};
pub const RSI: X86Register = X86Register {
    size: Size::S64,
    index: 6,
};
pub const RDI: X86Register = X86Register {
    size: Size::S64,
    index: 7,
};
pub const R8: X86Register = X86Register {
    size: Size::S64,
    index: 8,
};
pub const R9: X86Register = X86Register {
    size: Size::S64,
    index: 9,
};
pub const R10: X86Register = X86Register {
    size: Size::S64,
    index: 10,
};
pub const R11: X86Register = X86Register {
    size: Size::S64,
    index: 11,
};
pub const R12: X86Register = X86Register {
    size: Size::S64,
    index: 12,
};
pub const R13: X86Register = X86Register {
    size: Size::S64,
    index: 13,
};
pub const R14: X86Register = X86Register {
    size: Size::S64,
    index: 14,
};
pub const R15: X86Register = X86Register {
    size: Size::S64,
    index: 15,
};

// 32-bit register aliases
pub const EAX: X86Register = X86Register {
    size: Size::S32,
    index: 0,
};
pub const ECX: X86Register = X86Register {
    size: Size::S32,
    index: 1,
};
pub const EDX: X86Register = X86Register {
    size: Size::S32,
    index: 2,
};
pub const EBX: X86Register = X86Register {
    size: Size::S32,
    index: 3,
};

// 8-bit register aliases
pub const AL: X86Register = X86Register {
    size: Size::S8,
    index: 0,
};
pub const CL: X86Register = X86Register {
    size: Size::S8,
    index: 1,
};

/// Generate REX prefix
/// W = 1 for 64-bit operand size
/// R = extension of ModR/M reg field
/// X = extension of SIB index field
/// B = extension of ModR/M r/m field, SIB base field, or opcode reg field
#[inline]
pub fn rex(w: bool, r: bool, x: bool, b: bool) -> u8 {
    0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8)
}

/// Generate REX.W prefix for 64-bit operand size
#[inline]
pub fn rex_w(reg: u8, rm: u8) -> u8 {
    rex(true, reg >= 8, false, rm >= 8)
}

/// Generate REX prefix without W bit (only if needed for extended registers)
#[inline]
pub fn rex_opt(reg: u8, rm: u8) -> Option<u8> {
    let r = reg >= 8;
    let b = rm >= 8;
    if r || b {
        Some(rex(false, r, false, b))
    } else {
        None
    }
}

/// Generate ModR/M byte
/// mod: 0b11 for register-register, 0b00 for [reg], 0b01 for [reg+disp8], 0b10 for [reg+disp32]
/// reg: register operand or opcode extension (3 bits)
/// rm: r/m operand (3 bits)
#[inline]
pub fn modrm(mod_: u8, reg: u8, rm: u8) -> u8 {
    (mod_ << 6) | ((reg & 0x7) << 3) | (rm & 0x7)
}

/// Generate SIB byte
/// scale: 0=1, 1=2, 2=4, 3=8
/// index: index register (3 bits)
/// base: base register (3 bits)
#[inline]
pub fn sib(scale: u8, index: u8, base: u8) -> u8 {
    (scale << 6) | ((index & 0x7) << 3) | (base & 0x7)
}

/// Condition codes for Jcc and SETcc instructions
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Condition {
    /// Equal / Zero (ZF=1)
    E = 0x4,
    /// Not Equal / Not Zero (ZF=0)
    NE = 0x5,
    /// Less (SF!=OF)
    L = 0xC,
    /// Less or Equal (ZF=1 or SF!=OF)
    LE = 0xE,
    /// Greater (ZF=0 and SF=OF)
    G = 0xF,
    /// Greater or Equal (SF=OF)
    GE = 0xD,
    /// Below / Carry (CF=1)
    B = 0x2,
    /// Below or Equal (CF=1 or ZF=1)
    BE = 0x6,
    /// Above (CF=0 and ZF=0)
    A = 0x7,
    /// Above or Equal / Not Carry (CF=0)
    AE = 0x3,
}

/// x86-64 instruction representation
#[derive(Debug, Clone)]
pub enum X86Asm {
    // === Move instructions ===
    /// MOV r64, r64
    MovRR {
        dest: X86Register,
        src: X86Register,
    },
    /// MOV r64, imm64
    MovRI {
        dest: X86Register,
        imm: i64,
    },
    /// MOV r64, imm32 (sign-extended)
    MovRI32 {
        dest: X86Register,
        imm: i32,
    },
    /// MOV r64, [base + offset]
    MovRM {
        dest: X86Register,
        base: X86Register,
        offset: i32,
    },
    /// MOV [base + offset], r64
    MovMR {
        base: X86Register,
        offset: i32,
        src: X86Register,
    },
    /// MOV r64, [base + index*1] - SIB addressing with scale=1
    MovRMIndexed {
        dest: X86Register,
        base: X86Register,
        index: X86Register,
    },
    /// MOV [base + index*1], r64 - SIB addressing with scale=1
    MovMRIndexed {
        base: X86Register,
        index: X86Register,
        src: X86Register,
    },
    /// LEA r64, [base + offset]
    Lea {
        dest: X86Register,
        base: X86Register,
        offset: i32,
    },
    /// LEA r64, [rip + offset] - RIP-relative addressing for label addresses
    LeaRipRel {
        dest: X86Register,
        label_index: usize, // Will be patched to actual offset
    },

    // === Arithmetic instructions ===
    /// ADD r64, r64
    AddRR {
        dest: X86Register,
        src: X86Register,
    },
    /// ADD r64, imm32
    AddRI {
        dest: X86Register,
        imm: i32,
    },
    /// SUB r64, r64
    SubRR {
        dest: X86Register,
        src: X86Register,
    },
    /// SUB r64, imm32
    SubRI {
        dest: X86Register,
        imm: i32,
    },
    /// IMUL r64, r64
    ImulRR {
        dest: X86Register,
        src: X86Register,
    },
    /// IMUL r64, r64, imm32
    ImulRRI {
        dest: X86Register,
        src: X86Register,
        imm: i32,
    },
    /// IDIV r64 (divides RDX:RAX by r64, quotient in RAX, remainder in RDX)
    Idiv {
        divisor: X86Register,
    },
    /// CQO (sign-extend RAX into RDX:RAX)
    Cqo,
    /// NEG r64
    Neg {
        reg: X86Register,
    },

    // === Bitwise instructions ===
    /// AND r64, r64
    AndRR {
        dest: X86Register,
        src: X86Register,
    },
    /// AND r64, imm32
    AndRI {
        dest: X86Register,
        imm: i32,
    },
    /// OR r64, r64
    OrRR {
        dest: X86Register,
        src: X86Register,
    },
    /// OR r64, imm32
    OrRI {
        dest: X86Register,
        imm: i32,
    },
    /// XOR r64, r64
    XorRR {
        dest: X86Register,
        src: X86Register,
    },
    /// XOR r64, imm32
    XorRI {
        dest: X86Register,
        imm: i32,
    },
    /// NOT r64
    Not {
        reg: X86Register,
    },

    // === Shift instructions ===
    /// SHL r64, imm8
    ShlRI {
        dest: X86Register,
        imm: u8,
    },
    /// SHL r64, CL
    ShlRCL {
        dest: X86Register,
    },
    /// SHR r64, imm8 (logical)
    ShrRI {
        dest: X86Register,
        imm: u8,
    },
    /// SHR r64, CL (logical)
    ShrRCL {
        dest: X86Register,
    },
    /// SAR r64, imm8 (arithmetic)
    SarRI {
        dest: X86Register,
        imm: u8,
    },
    /// SAR r64, CL (arithmetic)
    SarRCL {
        dest: X86Register,
    },

    // === Comparison instructions ===
    /// CMP r64, r64
    CmpRR {
        a: X86Register,
        b: X86Register,
    },
    /// CMP r64, imm32
    CmpRI {
        reg: X86Register,
        imm: i32,
    },
    /// TEST r64, r64
    TestRR {
        a: X86Register,
        b: X86Register,
    },
    /// TEST r64, imm32
    TestRI {
        reg: X86Register,
        imm: i32,
    },
    /// SETcc r8
    Setcc {
        dest: X86Register,
        cond: Condition,
    },

    // === Control flow instructions ===
    /// JMP rel32 (label index, patched later)
    Jmp {
        label_index: usize,
    },
    /// Jcc rel32 (conditional jump, label index)
    Jcc {
        label_index: usize,
        cond: Condition,
    },
    /// CALL r64
    CallR {
        target: X86Register,
    },
    /// CALL rel32 (label index)
    CallRel {
        label_index: usize,
    },
    /// RET
    Ret,

    // === Stack instructions ===
    /// PUSH r64
    Push {
        reg: X86Register,
    },
    /// POP r64
    Pop {
        reg: X86Register,
    },

    // === Floating-point instructions (SSE2) ===
    /// ADDSD xmm, xmm
    Addsd {
        dest: X86Register,
        src: X86Register,
    },
    /// SUBSD xmm, xmm
    Subsd {
        dest: X86Register,
        src: X86Register,
    },
    /// MULSD xmm, xmm
    Mulsd {
        dest: X86Register,
        src: X86Register,
    },
    /// DIVSD xmm, xmm
    Divsd {
        dest: X86Register,
        src: X86Register,
    },
    /// MOVSD xmm, xmm
    MovsdRR {
        dest: X86Register,
        src: X86Register,
    },
    /// MOVSD xmm, [base + offset]
    MovsdRM {
        dest: X86Register,
        base: X86Register,
        offset: i32,
    },
    /// MOVSD [base + offset], xmm
    MovsdMR {
        base: X86Register,
        offset: i32,
        src: X86Register,
    },
    /// MOVQ r64, xmm (move quadword from XMM to GP register)
    MovqRX {
        dest: X86Register,
        src: X86Register,
    },
    /// MOVQ xmm, r64 (move quadword from GP register to XMM)
    MovqXR {
        dest: X86Register,
        src: X86Register,
    },
    /// UCOMISD xmm, xmm (unordered compare scalar double)
    Ucomisd {
        a: X86Register,
        b: X86Register,
    },

    // === Atomic instructions ===
    /// MFENCE
    Mfence,
    /// LOCK CMPXCHG [base], r64
    LockCmpxchg {
        base: X86Register,
        src: X86Register,
    },

    // === Misc instructions ===
    /// INT3 (breakpoint)
    Int3,
    /// NOP
    Nop,

    // === Placeholder for label locations ===
    Label {
        index: usize,
    },
}

impl X86Asm {
    /// Encode this instruction to bytes
    pub fn encode(&self) -> Vec<u8> {
        match self {
            // === Move instructions ===
            X86Asm::MovRR { dest, src } => {
                // MOV r64, r64: REX.W + 89 /r (src -> dest)
                // or REX.W + 8B /r (dest <- src)
                let mut bytes = vec![rex_w(src.index, dest.index), 0x89];
                bytes.push(modrm(0b11, src.index, dest.index));
                bytes
            }

            X86Asm::MovRI { dest, imm } => {
                // MOV r64, imm64: REX.W + B8+rd io
                let mut bytes = vec![rex_w(0, dest.index), 0xB8 + (dest.index & 0x7)];
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }

            X86Asm::MovRI32 { dest, imm } => {
                // MOV r/m64, imm32: REX.W + C7 /0 id
                let mut bytes = vec![rex_w(0, dest.index), 0xC7];
                bytes.push(modrm(0b11, 0, dest.index));
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }

            X86Asm::MovRM { dest, base, offset } => {
                // MOV r64, [base + offset]: REX.W + 8B /r
                encode_mem_op(0x8B, dest.index, base.index, *offset, true)
            }

            X86Asm::MovMR { base, offset, src } => {
                // MOV [base + offset], r64: REX.W + 89 /r
                encode_mem_op(0x89, src.index, base.index, *offset, true)
            }

            X86Asm::MovRMIndexed { dest, base, index } => {
                // MOV r64, [base + index*1]: REX.W + 8B /r with SIB
                // ModRM: mod=00, reg=dest, rm=100 (SIB follows)
                // SIB: scale=00 (1x), index=index, base=base
                let base_idx = base.index;
                let index_idx = index.index;
                let dest_idx = dest.index;

                // REX prefix: W=1, R=(dest>>3), X=(index>>3), B=(base>>3)
                let rex = 0x48
                    | ((dest_idx >> 3) << 2)  // REX.R
                    | ((index_idx >> 3) << 1) // REX.X
                    | (base_idx >> 3); // REX.B

                // ModRM: mod=00, reg=dest[2:0], rm=100 (SIB)
                let modrm = ((dest_idx & 0b111) << 3) | 0b100;

                // SIB: scale=00, index=index[2:0], base=base[2:0]
                let sib_byte = ((index_idx & 0b111) << 3) | (base_idx & 0b111);

                // Handle RBP/R13 as base (requires disp8 of 0)
                if (base_idx & 0b111) == 5 {
                    // mod=01 with disp8=0
                    let modrm = 0b01_000_100 | ((dest_idx & 0b111) << 3);
                    vec![rex, 0x8B, modrm, sib_byte, 0x00]
                } else {
                    vec![rex, 0x8B, modrm, sib_byte]
                }
            }

            X86Asm::MovMRIndexed { base, index, src } => {
                // MOV [base + index*1], r64: REX.W + 89 /r with SIB
                let base_idx = base.index;
                let index_idx = index.index;
                let src_idx = src.index;

                let rex = 0x48
                    | ((src_idx >> 3) << 2)   // REX.R
                    | ((index_idx >> 3) << 1) // REX.X
                    | (base_idx >> 3); // REX.B

                let modrm = ((src_idx & 0b111) << 3) | 0b100;
                let sib_byte = ((index_idx & 0b111) << 3) | (base_idx & 0b111);

                if (base_idx & 0b111) == 5 {
                    let modrm = 0b01_000_100 | ((src_idx & 0b111) << 3);
                    vec![rex, 0x89, modrm, sib_byte, 0x00]
                } else {
                    vec![rex, 0x89, modrm, sib_byte]
                }
            }

            X86Asm::Lea { dest, base, offset } => {
                // LEA r64, [base + offset]: REX.W + 8D /r
                encode_mem_op(0x8D, dest.index, base.index, *offset, true)
            }

            X86Asm::LeaRipRel { dest, label_index } => {
                // LEA r64, [RIP + disp32]: REX.W + 8D + ModRM(00 reg 101) + disp32
                // mod=00, rm=101 means RIP-relative with 32-bit displacement
                let reg = dest.index;
                let rex = 0x48 | ((reg >> 3) << 2); // REX.W + REX.R if needed
                let modrm = ((reg & 0b111) << 3) | 0b101; // mod=00, rm=101 (RIP)
                let disp = *label_index as i32;
                let mut bytes = vec![rex, 0x8D, modrm];
                bytes.extend(&disp.to_le_bytes());
                bytes
            }

            // === Arithmetic instructions ===
            X86Asm::AddRR { dest, src } => {
                // ADD r64, r64: REX.W + 01 /r
                let mut bytes = vec![rex_w(src.index, dest.index), 0x01];
                bytes.push(modrm(0b11, src.index, dest.index));
                bytes
            }

            X86Asm::AddRI { dest, imm } => {
                // ADD r/m64, imm32: REX.W + 81 /0 id
                let mut bytes = vec![rex_w(0, dest.index), 0x81];
                bytes.push(modrm(0b11, 0, dest.index));
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }

            X86Asm::SubRR { dest, src } => {
                // SUB r64, r64: REX.W + 29 /r
                let mut bytes = vec![rex_w(src.index, dest.index), 0x29];
                bytes.push(modrm(0b11, src.index, dest.index));
                bytes
            }

            X86Asm::SubRI { dest, imm } => {
                // SUB r/m64, imm32: REX.W + 81 /5 id
                let mut bytes = vec![rex_w(0, dest.index), 0x81];
                bytes.push(modrm(0b11, 5, dest.index));
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }

            X86Asm::ImulRR { dest, src } => {
                // IMUL r64, r/m64: REX.W + 0F AF /r
                let mut bytes = vec![rex_w(dest.index, src.index), 0x0F, 0xAF];
                bytes.push(modrm(0b11, dest.index, src.index));
                bytes
            }

            X86Asm::ImulRRI { dest, src, imm } => {
                // IMUL r64, r/m64, imm32: REX.W + 69 /r id
                let mut bytes = vec![rex_w(dest.index, src.index), 0x69];
                bytes.push(modrm(0b11, dest.index, src.index));
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }

            X86Asm::Idiv { divisor } => {
                // IDIV r/m64: REX.W + F7 /7
                let mut bytes = vec![rex_w(0, divisor.index), 0xF7];
                bytes.push(modrm(0b11, 7, divisor.index));
                bytes
            }

            X86Asm::Cqo => {
                // CQO: REX.W + 99
                vec![0x48, 0x99]
            }

            X86Asm::Neg { reg } => {
                // NEG r/m64: REX.W + F7 /3
                let mut bytes = vec![rex_w(0, reg.index), 0xF7];
                bytes.push(modrm(0b11, 3, reg.index));
                bytes
            }

            // === Bitwise instructions ===
            X86Asm::AndRR { dest, src } => {
                // AND r64, r64: REX.W + 21 /r
                let mut bytes = vec![rex_w(src.index, dest.index), 0x21];
                bytes.push(modrm(0b11, src.index, dest.index));
                bytes
            }

            X86Asm::AndRI { dest, imm } => {
                // AND r/m64, imm32: REX.W + 81 /4 id
                let mut bytes = vec![rex_w(0, dest.index), 0x81];
                bytes.push(modrm(0b11, 4, dest.index));
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }

            X86Asm::OrRR { dest, src } => {
                // OR r64, r64: REX.W + 09 /r
                let mut bytes = vec![rex_w(src.index, dest.index), 0x09];
                bytes.push(modrm(0b11, src.index, dest.index));
                bytes
            }

            X86Asm::OrRI { dest, imm } => {
                // OR r/m64, imm32: REX.W + 81 /1 id
                let mut bytes = vec![rex_w(0, dest.index), 0x81];
                bytes.push(modrm(0b11, 1, dest.index));
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }

            X86Asm::XorRR { dest, src } => {
                // XOR r64, r64: REX.W + 31 /r
                let mut bytes = vec![rex_w(src.index, dest.index), 0x31];
                bytes.push(modrm(0b11, src.index, dest.index));
                bytes
            }

            X86Asm::XorRI { dest, imm } => {
                // XOR r/m64, imm32: REX.W + 81 /6 id
                let mut bytes = vec![rex_w(0, dest.index), 0x81];
                bytes.push(modrm(0b11, 6, dest.index));
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }

            X86Asm::Not { reg } => {
                // NOT r/m64: REX.W + F7 /2
                let mut bytes = vec![rex_w(0, reg.index), 0xF7];
                bytes.push(modrm(0b11, 2, reg.index));
                bytes
            }

            // === Shift instructions ===
            X86Asm::ShlRI { dest, imm } => {
                // SHL r/m64, imm8: REX.W + C1 /4 ib
                let mut bytes = vec![rex_w(0, dest.index), 0xC1];
                bytes.push(modrm(0b11, 4, dest.index));
                bytes.push(*imm);
                bytes
            }

            X86Asm::ShlRCL { dest } => {
                // SHL r/m64, CL: REX.W + D3 /4
                let mut bytes = vec![rex_w(0, dest.index), 0xD3];
                bytes.push(modrm(0b11, 4, dest.index));
                bytes
            }

            X86Asm::ShrRI { dest, imm } => {
                // SHR r/m64, imm8: REX.W + C1 /5 ib
                let mut bytes = vec![rex_w(0, dest.index), 0xC1];
                bytes.push(modrm(0b11, 5, dest.index));
                bytes.push(*imm);
                bytes
            }

            X86Asm::ShrRCL { dest } => {
                // SHR r/m64, CL: REX.W + D3 /5
                let mut bytes = vec![rex_w(0, dest.index), 0xD3];
                bytes.push(modrm(0b11, 5, dest.index));
                bytes
            }

            X86Asm::SarRI { dest, imm } => {
                // SAR r/m64, imm8: REX.W + C1 /7 ib
                let mut bytes = vec![rex_w(0, dest.index), 0xC1];
                bytes.push(modrm(0b11, 7, dest.index));
                bytes.push(*imm);
                bytes
            }

            X86Asm::SarRCL { dest } => {
                // SAR r/m64, CL: REX.W + D3 /7
                let mut bytes = vec![rex_w(0, dest.index), 0xD3];
                bytes.push(modrm(0b11, 7, dest.index));
                bytes
            }

            // === Comparison instructions ===
            X86Asm::CmpRR { a, b } => {
                // CMP r64, r64: REX.W + 39 /r
                let mut bytes = vec![rex_w(b.index, a.index), 0x39];
                bytes.push(modrm(0b11, b.index, a.index));
                bytes
            }

            X86Asm::CmpRI { reg, imm } => {
                // CMP r/m64, imm32: REX.W + 81 /7 id
                let mut bytes = vec![rex_w(0, reg.index), 0x81];
                bytes.push(modrm(0b11, 7, reg.index));
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }

            X86Asm::TestRR { a, b } => {
                // TEST r64, r64: REX.W + 85 /r
                let mut bytes = vec![rex_w(b.index, a.index), 0x85];
                bytes.push(modrm(0b11, b.index, a.index));
                bytes
            }

            X86Asm::TestRI { reg, imm } => {
                // TEST r/m64, imm32: REX.W + F7 /0 id
                let mut bytes = vec![rex_w(0, reg.index), 0xF7];
                bytes.push(modrm(0b11, 0, reg.index));
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }

            X86Asm::Setcc { dest, cond } => {
                // SETcc r/m8: 0F 9x /0
                let mut bytes = Vec::new();
                if let Some(rex) = rex_opt(0, dest.index) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x90 + (*cond as u8));
                bytes.push(modrm(0b11, 0, dest.index));
                bytes
            }

            // === Control flow instructions ===
            X86Asm::Jmp { label_index } => {
                // JMP rel32: E9 cd
                let rel = *label_index as i32;
                let mut bytes = vec![0xE9];
                bytes.extend_from_slice(&rel.to_le_bytes());
                bytes
            }

            X86Asm::Jcc { label_index, cond } => {
                // Jcc rel32: 0F 8x cd
                let rel = *label_index as i32;
                let mut bytes = vec![0x0F, 0x80 + (*cond as u8)];
                bytes.extend_from_slice(&rel.to_le_bytes());
                bytes
            }

            X86Asm::CallR { target } => {
                // CALL r64: FF /2
                let mut bytes = Vec::new();
                if target.needs_rex_ext() {
                    bytes.push(rex(false, false, false, true));
                }
                bytes.push(0xFF);
                bytes.push(modrm(0b11, 2, target.index));
                bytes
            }

            X86Asm::CallRel { label_index } => {
                // CALL rel32: E8 cd
                let rel = *label_index as i32;
                let mut bytes = vec![0xE8];
                bytes.extend_from_slice(&rel.to_le_bytes());
                bytes
            }

            X86Asm::Ret => {
                // RET: C3
                vec![0xC3]
            }

            // === Stack instructions ===
            X86Asm::Push { reg } => {
                // PUSH r64: 50+rd (or REX + 50+rd for R8-R15)
                if reg.needs_rex_ext() {
                    vec![rex(false, false, false, true), 0x50 + (reg.index & 0x7)]
                } else {
                    vec![0x50 + reg.index]
                }
            }

            X86Asm::Pop { reg } => {
                // POP r64: 58+rd (or REX + 58+rd for R8-R15)
                if reg.needs_rex_ext() {
                    vec![rex(false, false, false, true), 0x58 + (reg.index & 0x7)]
                } else {
                    vec![0x58 + reg.index]
                }
            }

            // === Floating-point instructions (SSE2) ===
            X86Asm::Addsd { dest, src } => {
                // ADDSD xmm, xmm: F2 0F 58 /r
                let mut bytes = vec![0xF2];
                if dest.needs_rex_ext() || src.needs_rex_ext() {
                    bytes.push(rex(false, dest.index >= 8, false, src.index >= 8));
                }
                bytes.extend_from_slice(&[0x0F, 0x58]);
                bytes.push(modrm(0b11, dest.index, src.index));
                bytes
            }

            X86Asm::Subsd { dest, src } => {
                // SUBSD xmm, xmm: F2 0F 5C /r
                let mut bytes = vec![0xF2];
                if dest.needs_rex_ext() || src.needs_rex_ext() {
                    bytes.push(rex(false, dest.index >= 8, false, src.index >= 8));
                }
                bytes.extend_from_slice(&[0x0F, 0x5C]);
                bytes.push(modrm(0b11, dest.index, src.index));
                bytes
            }

            X86Asm::Mulsd { dest, src } => {
                // MULSD xmm, xmm: F2 0F 59 /r
                let mut bytes = vec![0xF2];
                if dest.needs_rex_ext() || src.needs_rex_ext() {
                    bytes.push(rex(false, dest.index >= 8, false, src.index >= 8));
                }
                bytes.extend_from_slice(&[0x0F, 0x59]);
                bytes.push(modrm(0b11, dest.index, src.index));
                bytes
            }

            X86Asm::Divsd { dest, src } => {
                // DIVSD xmm, xmm: F2 0F 5E /r
                let mut bytes = vec![0xF2];
                if dest.needs_rex_ext() || src.needs_rex_ext() {
                    bytes.push(rex(false, dest.index >= 8, false, src.index >= 8));
                }
                bytes.extend_from_slice(&[0x0F, 0x5E]);
                bytes.push(modrm(0b11, dest.index, src.index));
                bytes
            }

            X86Asm::MovsdRR { dest, src } => {
                // MOVSD xmm, xmm: F2 0F 10 /r
                let mut bytes = vec![0xF2];
                if dest.needs_rex_ext() || src.needs_rex_ext() {
                    bytes.push(rex(false, dest.index >= 8, false, src.index >= 8));
                }
                bytes.extend_from_slice(&[0x0F, 0x10]);
                bytes.push(modrm(0b11, dest.index, src.index));
                bytes
            }

            X86Asm::MovsdRM { dest, base, offset } => {
                // MOVSD xmm, m64: F2 0F 10 /r
                let mut bytes = vec![0xF2];
                bytes.extend(encode_mem_op_no_rex(0x10, dest.index, base.index, *offset));
                bytes
            }

            X86Asm::MovsdMR { base, offset, src } => {
                // MOVSD m64, xmm: F2 0F 11 /r
                let mut bytes = vec![0xF2];
                bytes.extend(encode_mem_op_no_rex(0x11, src.index, base.index, *offset));
                bytes
            }

            X86Asm::MovqRX { dest, src } => {
                // MOVQ r64, xmm: 66 REX.W 0F 7E /r
                vec![
                    0x66,
                    rex_w(src.index, dest.index),
                    0x0F,
                    0x7E,
                    modrm(0b11, src.index, dest.index),
                ]
            }

            X86Asm::MovqXR { dest, src } => {
                // MOVQ xmm, r64: 66 REX.W 0F 6E /r
                vec![
                    0x66,
                    rex_w(dest.index, src.index),
                    0x0F,
                    0x6E,
                    modrm(0b11, dest.index, src.index),
                ]
            }

            X86Asm::Ucomisd { a, b } => {
                // UCOMISD xmm, xmm: 66 0F 2E /r
                let mut bytes = vec![0x66];
                if a.needs_rex_ext() || b.needs_rex_ext() {
                    bytes.push(rex(false, a.index >= 8, false, b.index >= 8));
                }
                bytes.extend_from_slice(&[0x0F, 0x2E, modrm(0b11, a.index, b.index)]);
                bytes
            }

            // === Atomic instructions ===
            X86Asm::Mfence => {
                // MFENCE: 0F AE F0
                vec![0x0F, 0xAE, 0xF0]
            }

            X86Asm::LockCmpxchg { base, src } => {
                // LOCK CMPXCHG [base], r64: F0 REX.W 0F B1 /r
                // Note: Must use encode_modrm_mem to handle RBP/R13 (index 5) correctly
                // since mod=00 rm=101 means RIP-relative, not [RBP]
                let mut bytes = vec![0xF0, rex_w(src.index, base.index), 0x0F, 0xB1];
                encode_modrm_mem(&mut bytes, src.index, base.index, 0);
                bytes
            }

            // === Misc instructions ===
            X86Asm::Int3 => {
                vec![0xCC]
            }

            X86Asm::Nop => {
                vec![0x90]
            }

            X86Asm::Label { index: _ } => {
                // Labels don't produce code, they're markers
                vec![]
            }
        }
    }

    /// Get the size of this instruction in bytes
    pub fn size(&self) -> usize {
        self.encode().len()
    }
}

/// Helper to encode memory operand with REX.W
fn encode_mem_op(opcode: u8, reg: u8, base: u8, offset: i32, rex_w_needed: bool) -> Vec<u8> {
    let mut bytes = Vec::new();

    // REX prefix
    if rex_w_needed {
        bytes.push(rex_w(reg, base));
    } else if reg >= 8 || base >= 8 {
        bytes.push(rex(false, reg >= 8, false, base >= 8));
    }

    // Opcode
    bytes.push(opcode);

    // ModRM and displacement
    encode_modrm_mem(&mut bytes, reg, base, offset);

    bytes
}

/// Helper to encode memory operand without automatic REX
fn encode_mem_op_no_rex(opcode: u8, reg: u8, base: u8, offset: i32) -> Vec<u8> {
    let mut bytes = Vec::new();

    // REX prefix only if needed for extended registers
    if reg >= 8 || base >= 8 {
        bytes.push(rex(false, reg >= 8, false, base >= 8));
    }

    // Two-byte opcode (0F xx)
    bytes.push(0x0F);
    bytes.push(opcode);

    // ModRM and displacement
    encode_modrm_mem(&mut bytes, reg, base, offset);

    bytes
}

/// Encode ModRM byte and displacement for memory operand
fn encode_modrm_mem(bytes: &mut Vec<u8>, reg: u8, base: u8, offset: i32) {
    let base_low = base & 0x7;

    // RSP/R12 (index 4) requires SIB byte
    let needs_sib = base_low == 4;

    // RBP/R13 (index 5) with zero offset still needs displacement
    let rbp_base = base_low == 5;

    if offset == 0 && !rbp_base {
        // [base] - no displacement
        bytes.push(modrm(0b00, reg, base));
        if needs_sib {
            bytes.push(sib(0, 4, base_low)); // SIB with no index
        }
    } else if (-128..=127).contains(&offset) {
        // [base + disp8]
        bytes.push(modrm(0b01, reg, base));
        if needs_sib {
            bytes.push(sib(0, 4, base_low));
        }
        bytes.push(offset as i8 as u8);
    } else {
        // [base + disp32]
        bytes.push(modrm(0b10, reg, base));
        if needs_sib {
            bytes.push(sib(0, 4, base_low));
        }
        bytes.extend_from_slice(&offset.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mov_rr() {
        // MOV RAX, RBX
        let instr = X86Asm::MovRR {
            dest: RAX,
            src: RBX,
        };
        assert_eq!(instr.encode(), vec![0x48, 0x89, 0xD8]);

        // MOV R8, R9
        let instr = X86Asm::MovRR { dest: R8, src: R9 };
        assert_eq!(instr.encode(), vec![0x4D, 0x89, 0xC8]);
    }

    #[test]
    fn test_add_rr() {
        // ADD RAX, RBX
        let instr = X86Asm::AddRR {
            dest: RAX,
            src: RBX,
        };
        assert_eq!(instr.encode(), vec![0x48, 0x01, 0xD8]);
    }

    #[test]
    fn test_push_pop() {
        // PUSH RAX
        assert_eq!(X86Asm::Push { reg: RAX }.encode(), vec![0x50]);
        // PUSH R8
        assert_eq!(X86Asm::Push { reg: R8 }.encode(), vec![0x41, 0x50]);
        // POP RBX
        assert_eq!(X86Asm::Pop { reg: RBX }.encode(), vec![0x5B]);
    }

    #[test]
    fn test_ret() {
        assert_eq!(X86Asm::Ret.encode(), vec![0xC3]);
    }

    #[test]
    fn test_lea_rsp_offset() {
        // LEA RDX, [RSP - 8]
        // RSP is index 4, requires SIB byte
        // REX.W + 8D + ModRM + SIB + disp8
        let instr = X86Asm::Lea {
            dest: RDX,
            base: RSP,
            offset: -8,
        };
        let encoded = instr.encode();
        // Debug print
        println!("LEA RDX, [RSP - 8] encoded as: {:02x?}", encoded);
        // Expected: 48 8D 54 24 F8
        // REX.W (0x48), 8D (LEA opcode), ModRM (01 010 100 = 0x54 for [RSP+disp8]),
        // SIB (00 100 100 = 0x24 for RSP no index), disp8 (-8 = 0xF8)
        assert_eq!(encoded, vec![0x48, 0x8D, 0x54, 0x24, 0xF8]);
    }
}
