//! ISET: integer compare to a register mask or float, ported from integer_compare_and_set.cpp.

use super::{
    BooleanOp, CompareOp, TResult, Translator, extended_integer_compare, integer_compare,
    predicate_combine,
};
use crate::ir::{ICompare, Pred, Value};

fn iset(t: &mut Translator<'_>, src_b: Value) -> TResult {
    let insn = t.insn;
    let bop = BooleanOp::from_bits(insn.bits(45, 2)).ok_or_else(|| t.unsupported())?;
    let x = insn.bit(43);
    let bf = insn.bit(44);
    let cc = insn.bit(47);
    if cc && x {
        return Err(t.unsupported());
    }
    let src_a = t.x(insn.src_a_reg());
    let signed = insn.bit(48);
    let zero = Value::U32(0);
    let op = CompareOp::from_bits(insn.bits(49, 3));
    let cmp_result = if x {
        extended_integer_compare(&mut t.ir, src_a, src_b, op, signed)
    } else {
        integer_compare(&mut t.ir, src_a, src_b, op, signed)
    };
    let mut pred = t.ir.get_pred(Pred::p(insn.bits(39, 3) as u8), false);
    if insn.bit(42) {
        pred = t.ir.logical_not(pred);
    }
    let bop_result = predicate_combine(&mut t.ir, cmp_result, pred, bop);
    let pass_result = Value::U32(if bf { 0x3f80_0000 } else { u32::MAX });
    let result = t.ir.select(bop_result, pass_result, zero);
    t.set_x(insn.dest_reg(), result);
    if cc {
        let is_zero = t.ir.icompare(ICompare::Equal, result, zero);
        t.ir.set_z_flag(is_zero);
        if bf {
            t.reset_s_flag();
        } else {
            let not_zero = t.ir.logical_not(is_zero);
            t.ir.set_s_flag(not_zero);
        }
        t.reset_c_flag();
        t.reset_o_flag();
    }
    Ok(())
}

pub(super) fn iset_cbuf(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_cbuf()?;
    iset(t, src_b)
}

pub(super) fn iset_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_reg20();
    iset(t, src_b)
}

pub(super) fn iset_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_imm20();
    iset(t, src_b)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn iset_reg_lt() {
        let insn = encode(
            Opcode::IsetReg,
            (1 << 8) | (2 << 20) | (7 << 39) | (1 << 49),
        );
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R1
  %2 = ULessThan %1, %0
  %3 = LogicalAnd %2, #true
  %4 = SelectU32 %3, #0xffffffff, #0x0
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iset_imm_float_or_cc() {
        let insn = encode(
            Opcode::IsetImm,
            (1 << 8)
                | (5 << 20)
                | (7 << 39)
                | (1 << 42)
                | (1 << 44)
                | (1 << 45)
                | (1 << 47)
                | (2 << 49),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = IEqual %0, #0x5
  %2 = LogicalNot #true
  %3 = LogicalOr %1, %2
  %4 = SelectU32 %3, #0x3f800000, #0x0
  SetRegister R0, %4
  %5 = IEqual %4, #0x0
  SetZFlag %5
  SetSFlag #false
  SetCFlag #false
  SetOFlag #false
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iset_cbuf_extended_eq() {
        let insn = encode(
            Opcode::IsetCbuf,
            (1 << 8) | (1 << 34) | (4 << 20) | (7 << 39) | (1 << 43) | (1 << 48) | (2 << 49),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R1
  %2 = GetCFlag
  %3 = SelectU32 %2, #0x1, #0x0
  %4 = GetZFlag
  %5 = BitwiseNot32 %0
  %6 = IAdd32 %1, %5
  %7 = IAdd32 %6, %3
  %8 = IEqual %7, #0x0
  %9 = LogicalAnd %8, %4
  %10 = LogicalAnd %9, #true
  %11 = SelectU32 %10, #0xffffffff, #0x0
  SetRegister R0, %11
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iset_x_cc() {
        let insn = encode(Opcode::IsetReg, (1 << 43) | (1 << 47));
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
