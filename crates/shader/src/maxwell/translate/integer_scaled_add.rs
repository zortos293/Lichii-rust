//! ISCADD and ISCADD32I: shift-and-add, ported from integer_scaled_add.cpp.

use super::{TResult, Translator};
use crate::ir::Value;

fn iscadd(
    t: &mut Translator<'_>,
    mut op_b: Value,
    cc: bool,
    neg_a: bool,
    neg_b: bool,
    scale: u32,
) -> TResult {
    let insn = t.insn;
    let po = neg_a && neg_b;
    let mut op_a = t.x(insn.src_a_reg());
    if po {
        op_b = t.ir.iadd(op_b, Value::U32(1));
    } else {
        if neg_a {
            op_a = t.ir.ineg(op_a);
        }
        if neg_b {
            op_b = t.ir.ineg(op_b);
        }
    }
    let scaled_a = t.ir.shift_left_logical(op_a, Value::U32(scale));
    let result = t.ir.iadd(scaled_a, op_b);
    t.set_x(insn.dest_reg(), result);
    if cc {
        let z = t.ir.get_zero_from_op(result);
        t.ir.set_z_flag(z);
        let s = t.ir.get_sign_from_op(result);
        t.ir.set_s_flag(s);
        let mut carry = t.ir.get_carry_from_op(result);
        let mut overflow = t.ir.get_overflow_from_op(result);
        if po {
            let carry_b = t.ir.get_carry_from_op(op_b);
            carry = t.ir.logical_or(carry, carry_b);
        }
        t.ir.set_c_flag(carry);
        if po {
            let overflow_b = t.ir.get_overflow_from_op(op_b);
            overflow = t.ir.logical_or(overflow, overflow_b);
        }
        t.ir.set_o_flag(overflow);
    }
    Ok(())
}

fn iscadd_common(t: &mut Translator<'_>, op_b: Value) -> TResult {
    let insn = t.insn;
    iscadd(
        t,
        op_b,
        insn.bit(47),
        insn.bit(49),
        insn.bit(48),
        insn.bits(39, 5) as u32,
    )
}

pub(super) fn iscadd_cbuf(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_cbuf()?;
    iscadd_common(t, op_b)
}

pub(super) fn iscadd_reg(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_reg20();
    iscadd_common(t, op_b)
}

pub(super) fn iscadd_imm(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_imm20();
    iscadd_common(t, op_b)
}

pub(super) fn iscadd32i(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let op_b = t.get_imm32();
    iscadd(t, op_b, insn.bit(52), false, false, insn.bits(53, 5) as u32)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn iscadd_reg() {
        let insn = encode(Opcode::IscaddReg, (1 << 8) | (2 << 20) | (3 << 39));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R1
  %2 = ShiftLeftLogical32 %1, #0x3
  %3 = IAdd32 %2, %0
  SetRegister R0, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iscadd_imm_po_cc() {
        let insn = encode(
            Opcode::IscaddImm,
            (1 << 8) | (5 << 20) | (3 << 39) | (1 << 47) | (3 << 48),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = IAdd32 #0x5, #0x1
  %2 = ShiftLeftLogical32 %0, #0x3
  %3 = IAdd32 %2, %1
  SetRegister R0, %3
  %4 = GetZeroFromOp %3
  SetZFlag %4
  %5 = GetSignFromOp %3
  SetSFlag %5
  %6 = GetCarryFromOp %3
  %7 = GetOverflowFromOp %3
  %8 = GetCarryFromOp %1
  %9 = LogicalOr %6, %8
  SetCFlag %9
  %10 = GetOverflowFromOp %1
  %11 = LogicalOr %7, %10
  SetOFlag %11
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iscadd_cbuf_neg_a() {
        let insn = encode(
            Opcode::IscaddCbuf,
            (1 << 8) | (1 << 34) | (4 << 20) | (3 << 39) | (1 << 49),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R1
  %2 = INeg32 %1
  %3 = ShiftLeftLogical32 %2, #0x3
  %4 = IAdd32 %3, %0
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iscadd32i_cc() {
        let insn = encode(
            Opcode::Iscadd32i,
            (1 << 8) | (0x40 << 20) | (1 << 52) | (2 << 53),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = ShiftLeftLogical32 %0, #0x2
  %2 = IAdd32 %1, #0x40
  SetRegister R0, %2
  %3 = GetZeroFromOp %2
  SetZFlag %3
  %4 = GetSignFromOp %2
  SetSFlag %4
  %5 = GetCarryFromOp %2
  %6 = GetOverflowFromOp %2
  SetCFlag %5
  SetOFlag %6
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }
}
