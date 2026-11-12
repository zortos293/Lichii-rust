//! Raw Maxwell instruction words and the fields most translators share.

use lichii_common::GpuVAddr;

use super::decode::decode;
use super::opcode::Opcode;
use crate::ir::{FlowTest, Pred, Reg};

/// One 64-bit Maxwell instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Instruction(pub u64);

impl Instruction {
    pub fn opcode(self) -> Option<Opcode> {
        decode(self.0)
    }

    /// `count` bits starting at bit `offset`, zero extended.
    pub const fn bits(self, offset: u32, count: u32) -> u64 {
        debug_assert!(count > 0 && offset + count <= 64);
        (self.0 >> offset) & (u64::MAX >> (64 - count))
    }

    /// `count` bits starting at bit `offset`, sign extended from the top bit of the field.
    pub const fn signed_bits(self, offset: u32, count: u32) -> i64 {
        debug_assert!(count > 0 && offset + count <= 64);
        ((self.0 << (64 - offset - count)) as i64) >> (64 - count)
    }

    pub const fn bit(self, offset: u32) -> bool {
        self.bits(offset, 1) != 0
    }

    /// Guard predicate (bits 16..19) and whether it is negated (bit 19).
    pub const fn pred(self) -> (Pred, bool) {
        (Pred::p(self.bits(16, 3) as u8), self.bit(19))
    }

    pub const fn dest_reg(self) -> Reg {
        Reg(self.bits(0, 8) as u16)
    }

    pub const fn src_a_reg(self) -> Reg {
        Reg(self.bits(8, 8) as u16)
    }

    pub const fn src_b_reg(self) -> Reg {
        Reg(self.bits(20, 8) as u16)
    }

    /// Constant buffer slot of a `c[index][offset]` operand (bits 34..39).
    pub const fn cbuf_index(self) -> u32 {
        self.bits(34, 5) as u32
    }

    /// Byte offset of a `c[index][offset]` operand: bits 20..34 in words.
    pub const fn cbuf_offset(self) -> u32 {
        self.bits(20, 14) as u32 * 4
    }

    /// The 20-bit immediate: 19 bits at 20 with the sign at bit 56, sign extended.
    pub const fn imm20(self) -> i32 {
        let magnitude = self.bits(20, 19) as i32;
        if self.bit(56) {
            magnitude - (1 << 19)
        } else {
            magnitude
        }
    }

    /// The 32-bit immediate at bits 20..52.
    pub const fn imm32(self) -> u32 {
        self.bits(20, 32) as u32
    }

    /// Relative branch displacement, signed 24 bits at 20, from the next instruction.
    pub const fn branch_offset(self) -> i32 {
        self.signed_bits(20, 24) as i32
    }

    /// Absolute branch target of JMP, JCAL and JMX: 32 bits at 20.
    pub const fn branch_absolute(self) -> u32 {
        self.bits(20, 32) as u32
    }

    /// A branch whose target comes from a constant buffer (bit 5).
    pub const fn branch_is_cbuf(self) -> bool {
        self.bit(5)
    }

    /// Condition-code test of a branch (bits 0..5).
    pub const fn flow_test(self) -> FlowTest {
        FlowTest::from_bits(self.bits(0, 5) as u32)
    }
}

/// Whether the word at `offset` bytes from `base` is a scheduling word: the first of every
/// 32-byte group holds the scheduling hints of the three instructions after it.
pub const fn is_scheduling_word(base: GpuVAddr, offset: u32) -> bool {
    base.wrapping_add(offset as u64).is_multiple_of(32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_fields() {
        let insn = Instruction(0xfedc_ba98_7654_3210);
        assert_eq!(insn.bits(0, 4), 0x0);
        assert_eq!(insn.bits(4, 8), 0x21);
        assert_eq!(insn.bits(0, 64), 0xfedc_ba98_7654_3210);
        assert_eq!(insn.signed_bits(60, 4), -1);
        assert_eq!(insn.signed_bits(0, 64), 0xfedc_ba98_7654_3210u64 as i64);
        assert_eq!(insn.signed_bits(4, 4), 1);
        assert!(insn.bit(63));
        assert!(!insn.bit(0));
    }

    #[test]
    fn guard_predicate() {
        assert_eq!(Instruction(0x7 << 16).pred(), (Pred::PT, false));
        assert_eq!(Instruction(0xf << 16).pred(), (Pred::PT, true));
        assert_eq!(Instruction(0x3 << 16).pred(), (Pred::p(3), false));
        assert_eq!(Instruction(0xa << 16).pred(), (Pred::p(2), true));
    }

    #[test]
    fn registers() {
        let insn = Instruction((0x12 << 20) | (0x34 << 8) | 0xff);
        assert_eq!(insn.dest_reg(), Reg::RZ);
        assert_eq!(insn.src_a_reg(), Reg::r(0x34));
        assert_eq!(insn.src_b_reg(), Reg::r(0x12));
    }

    #[test]
    fn constant_buffer_operand() {
        let insn = Instruction((3 << 34) | (0x10 << 20));
        assert_eq!(insn.cbuf_index(), 3);
        assert_eq!(insn.cbuf_offset(), 0x40);
        let max = Instruction((0x1f << 34) | (0x3fff << 20));
        assert_eq!(max.cbuf_index(), 31);
        assert_eq!(max.cbuf_offset(), 0xfffc);
    }

    #[test]
    fn immediates() {
        assert_eq!(Instruction(5 << 20).imm20(), 5);
        assert_eq!(Instruction((0x7ffff << 20) | (1 << 56)).imm20(), -1);
        assert_eq!(Instruction(1 << 56).imm20(), -(1 << 19));
        assert_eq!(Instruction(0x7ffff << 20).imm20(), 0x7ffff);
        assert_eq!(Instruction(0xdead_beef << 20).imm32(), 0xdead_beef);
    }

    #[test]
    fn branch_fields() {
        assert_eq!(Instruction(0x00_0010 << 20).branch_offset(), 0x10);
        assert_eq!(Instruction(0xff_fff8 << 20).branch_offset(), -8);
        assert_eq!(
            Instruction(0x1234_5678 << 20).branch_absolute(),
            0x1234_5678
        );
        assert!(Instruction(1 << 5).branch_is_cbuf());
        assert_eq!(Instruction(0xf).flow_test(), FlowTest::T);
        assert_eq!(Instruction(0x2).flow_test(), FlowTest::EQ);
        assert_eq!(Instruction(0x20).flow_test(), FlowTest::F);
    }

    #[test]
    fn scheduling_words() {
        assert!(is_scheduling_word(0x1000, 0));
        assert!(!is_scheduling_word(0x1000, 8));
        assert!(is_scheduling_word(0x1000, 32));
        assert!(is_scheduling_word(0x1050, 0x10));
        assert!(!is_scheduling_word(0x1050, 0));
    }
}
