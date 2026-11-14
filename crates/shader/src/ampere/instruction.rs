//! One 128-bit Ampere instruction and the operand fields its encodings share.
//!
//! Field positions follow Mesa NAK's SM70+ encoder (`sm70_encode.rs`), which Volta, Turing and
//! Ampere share. Bit 0 is the least significant bit of the first 64-bit word in program memory.

use crate::ir::{Pred, Reg};

/// Size of one instruction in bytes.
pub const INSTRUCTION_SIZE: u32 = 16;

/// A 128-bit Ampere (SM86) instruction. The low 64 bits are the word at the instruction's
/// address, the high 64 bits the word 8 bytes after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Instruction(pub u128);

/// Constant buffer selected by a constant-buffer operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CbufBank {
    /// Bound buffer slot `c[index]`.
    Index(u32),
    /// Bindless buffer whose handle is in a uniform register (`cx[UR]`).
    Ureg(Reg),
}

/// A constant-buffer operand `c[bank][offset]`, `offset` in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CbufOperand {
    pub bank: CbufBank,
    pub offset: u32,
}

impl Instruction {
    /// Builds an instruction from its two 64-bit words in memory order.
    pub const fn from_words(low: u64, high: u64) -> Instruction {
        Instruction(low as u128 | ((high as u128) << 64))
    }

    /// `count` bits (at most 64) starting at bit `offset`.
    pub const fn bits(self, offset: u32, count: u32) -> u64 {
        let value = (self.0 >> offset) as u64;
        if count >= 64 {
            value
        } else {
            value & ((1 << count) - 1)
        }
    }

    pub const fn bit(self, offset: u32) -> bool {
        (self.0 >> offset) & 1 != 0
    }

    /// `count` bits (1 to 64) starting at bit `offset`, sign-extended from the top one.
    pub const fn sbits(self, offset: u32, count: u32) -> i64 {
        let shift = 64 - count;
        ((self.bits(offset, count) << shift) as i64) >> shift
    }

    /// The 12-bit opcode: the 9-bit operation and the 3-bit form selector above it.
    pub const fn opcode(self) -> u32 {
        self.bits(0, 12) as u32
    }

    /// The form selector, bits 9..12: where the second and third sources live.
    pub const fn form_bits(self) -> u32 {
        self.bits(9, 3) as u32
    }

    /// Execution guard as a predicate register and its negation.
    pub const fn guard(self) -> (Pred, bool) {
        (Pred::p(self.bits(12, 3) as u8), self.bit(15))
    }

    /// Execution guard of a uniform-datapath instruction, which names a uniform predicate.
    pub const fn uniform_guard(self) -> (Pred, bool) {
        (Pred::up(self.bits(12, 3) as u8), self.bit(15))
    }

    /// Whether the guard is `!PT`: the instruction never executes (compiler padding).
    pub const fn never_executes(self) -> bool {
        self.bits(12, 4) == 15
    }

    /// The 8-bit general purpose register field at `offset`.
    pub const fn reg_at(self, offset: u32) -> Reg {
        Reg::r(self.bits(offset, 8) as u8)
    }

    /// The 6-bit uniform register field at `offset` (63 is URZ).
    pub const fn ureg_at(self, offset: u32) -> Reg {
        Reg::ur(self.bits(offset, 6) as u8)
    }

    pub const fn dst(self) -> Reg {
        self.reg_at(16)
    }

    pub const fn udst(self) -> Reg {
        self.ureg_at(16)
    }

    pub const fn src0(self) -> Reg {
        self.reg_at(24)
    }

    pub const fn usrc0(self) -> Reg {
        self.ureg_at(24)
    }

    /// The register at bits 32..40: the second source of the register-register form.
    pub const fn src1(self) -> Reg {
        self.reg_at(32)
    }

    /// The uniform register at bits 32..38, the uniform operand of a form with one.
    pub const fn usrc1(self) -> Reg {
        self.ureg_at(32)
    }

    /// The register at bits 64..72: the third source, or the second when the form puts an
    /// immediate, constant or uniform register in the third.
    pub const fn src2(self) -> Reg {
        self.reg_at(64)
    }

    pub const fn usrc2(self) -> Reg {
        self.ureg_at(64)
    }

    /// The 32-bit immediate of an immediate form, bits 32..64.
    pub const fn imm32(self) -> u32 {
        self.bits(32, 32) as u32
    }

    /// The constant buffer slot at 54..59. NVN2 programs read root table 0 (where the console
    /// keeps the blit parameter block) as slot 24, past Maxwell's 18; it is read as slot 17,
    /// which NVN2 never binds.
    pub const fn cbuf_slot(self) -> u32 {
        match self.bits(54, 5) as u32 {
            24 => 17,
            slot => slot,
        }
    }

    /// The constant-buffer operand of a constant-buffer form: offset at 38..54, slot at
    /// 54..59 ([`Self::cbuf_slot`]), or a uniform register handle at 32..38 when bit 91 is set.
    pub const fn cbuf(self) -> CbufOperand {
        let bank = if self.bit(91) {
            CbufBank::Ureg(self.ureg_at(32))
        } else {
            CbufBank::Index(self.cbuf_slot())
        };
        CbufOperand {
            bank,
            offset: self.bits(38, 16) as u32,
        }
    }

    pub const fn pred_at(self, offset: u32) -> Pred {
        Pred::p(self.bits(offset, 3) as u8)
    }

    pub const fn upred_at(self, offset: u32) -> Pred {
        Pred::up(self.bits(offset, 3) as u8)
    }

    /// A predicate source at `offset` with its negation bit.
    pub const fn pred_src(self, offset: u32, not_bit: u32) -> (Pred, bool) {
        (self.pred_at(offset), self.bit(not_bit))
    }

    /// A uniform predicate source at `offset` with its negation bit.
    pub const fn upred_src(self, offset: u32, not_bit: u32) -> (Pred, bool) {
        (self.upred_at(offset), self.bit(not_bit))
    }

    /// First predicate destination, bits 81..84.
    pub const fn pred_dst0(self) -> Pred {
        self.pred_at(81)
    }

    /// Second predicate destination, bits 84..87.
    pub const fn pred_dst1(self) -> Pred {
        self.pred_at(84)
    }

    /// The predicate source most instructions carry at bits 87..90, negated by bit 90.
    pub const fn pred_c(self) -> (Pred, bool) {
        self.pred_src(87, 90)
    }

    /// Byte offset of a relative branch target from the next instruction (bits 34..82,
    /// in words). BRA keeps its convergence mode in the two bits below.
    pub const fn branch_offset(self) -> i64 {
        self.sbits(34, 48) << 2
    }

    /// Byte target of an absolute transfer (JMP, CALL.ABS), bits 32..82.
    pub const fn absolute_target(self) -> u64 {
        self.bits(32, 50)
    }
}
