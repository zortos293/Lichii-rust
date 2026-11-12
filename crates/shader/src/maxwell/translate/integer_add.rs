//! IADD and IADD32I: two-input integer add, ported from integer_add.cpp.

use super::{TResult, Translator};
use crate::ir::Value;

struct Modifiers {
    neg_a: bool,
    po: bool,
    sat: bool,
    x: bool,
    cc: bool,
}

fn iadd(t: &mut Translator<'_>, op_b: Value, m: Modifiers) -> TResult {
    if m.sat || (m.x && m.po) || (m.cc && (m.po || m.x)) {
        return Err(t.unsupported());
    }
    let insn = t.insn;
    let mut op_a = t.x(insn.src_a_reg());
    if m.neg_a {
        op_a = t.ir.ineg(op_a);
    }
    let mut result = t.ir.iadd(op_a, op_b);
    if m.x {
        let c = t.ir.get_c_flag();
        let carry = t.ir.select(c, Value::U32(1), Value::U32(0));
        result = t.ir.iadd(result, carry);
    }
    if m.po {
        result = t.ir.iadd(result, Value::U32(1));
    }
    if m.cc {
        let z = t.ir.get_zero_from_op(result);
        t.ir.set_z_flag(z);
        let s = t.ir.get_sign_from_op(result);
        t.ir.set_s_flag(s);
        let c = t.ir.get_carry_from_op(result);
        t.ir.set_c_flag(c);
        let o = t.ir.get_overflow_from_op(result);
        t.ir.set_o_flag(o);
    }
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

fn iadd_common(t: &mut Translator<'_>, op_b: Value) -> TResult {
    let insn = t.insn;
    let po = insn.bits(48, 2) == 3;
    let op_b = if !po && insn.bit(48) {
        t.ir.ineg(op_b)
    } else {
        op_b
    };
    let m = Modifiers {
        neg_a: insn.bit(49),
        po,
        sat: insn.bit(50),
        x: insn.bit(43),
        cc: insn.bit(47),
    };
    iadd(t, op_b, m)
}

pub(super) fn iadd_cbuf(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_cbuf()?;
    iadd_common(t, op_b)
}

pub(super) fn iadd_reg(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_reg20();
    iadd_common(t, op_b)
}

pub(super) fn iadd_imm(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_imm20();
    iadd_common(t, op_b)
}

pub(super) fn iadd32i(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let po = insn.bits(55, 2) == 3;
    let m = Modifiers {
        neg_a: !po && insn.bit(56),
        po,
        sat: insn.bit(54),
        x: insn.bit(53),
        cc: insn.bit(52),
    };
    let op_b = t.get_imm32();
    iadd(t, op_b, m)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn iadd_reg() {
        let insn = encode(Opcode::IaddReg, (1 << 8) | (2 << 20));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R1
  %2 = IAdd32 %1, %0
  SetRegister R0, %2
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iadd_reg_neg_a_cc() {
        let insn = encode(
            Opcode::IaddReg,
            (1 << 8) | (2 << 20) | (1 << 47) | (1 << 49),
        );
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R1
  %2 = INeg32 %1
  %3 = IAdd32 %2, %0
  %4 = GetZeroFromOp %3
  SetZFlag %4
  %5 = GetSignFromOp %3
  SetSFlag %5
  %6 = GetCarryFromOp %3
  SetCFlag %6
  %7 = GetOverflowFromOp %3
  SetOFlag %7
  SetRegister R0, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iadd_cbuf_x() {
        let insn = encode(
            Opcode::IaddCbuf,
            (1 << 8) | (1 << 34) | (4 << 20) | (1 << 43),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R1
  %2 = IAdd32 %1, %0
  %3 = GetCFlag
  %4 = SelectU32 %3, #0x1, #0x0
  %5 = IAdd32 %2, %4
  SetRegister R0, %5
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iadd_imm_po() {
        let insn = encode(Opcode::IaddImm, (1 << 8) | (5 << 20) | (3 << 48));
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = INeg32 %0
  %2 = IAdd32 %1, #0x5
  %3 = IAdd32 %2, #0x1
  SetRegister R0, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iadd_sat() {
        let insn = encode(Opcode::IaddReg, 1 << 50);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn iadd_x_cc() {
        let insn = encode(Opcode::IaddReg, (1 << 43) | (1 << 47));
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn iadd32i() {
        let insn = encode(Opcode::Iadd32i, (1 << 8) | (0x1234_5678 << 20));
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = IAdd32 %0, #0x12345678
  SetRegister R0, %1
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iadd32i_neg_a_cc() {
        let insn = encode(
            Opcode::Iadd32i,
            (1 << 8) | (0x10 << 20) | (1 << 52) | (1 << 56),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = INeg32 %0
  %2 = IAdd32 %1, #0x10
  %3 = GetZeroFromOp %2
  SetZFlag %3
  %4 = GetSignFromOp %2
  SetSFlag %4
  %5 = GetCarryFromOp %2
  SetCFlag %5
  %6 = GetOverflowFromOp %2
  SetOFlag %6
  SetRegister R0, %2
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }
}
