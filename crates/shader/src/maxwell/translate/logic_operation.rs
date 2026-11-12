//! LOP and LOP32I: two-input bitwise operations, ported from logic_operation.cpp.

use super::{PredicateOp, TResult, Translator, predicate_operation};
use crate::ir::{ICompare, Pred, Value};

#[derive(Clone, Copy, PartialEq, Eq)]
enum LogicalOp {
    And,
    Or,
    Xor,
    PassB,
}

impl LogicalOp {
    const fn from_bits(bits: u64) -> LogicalOp {
        match bits & 3 {
            0 => LogicalOp::And,
            1 => LogicalOp::Or,
            2 => LogicalOp::Xor,
            _ => LogicalOp::PassB,
        }
    }
}

struct Modifiers {
    x: bool,
    cc: bool,
    inv_a: bool,
    inv_b: bool,
    op: LogicalOp,
    pred: Option<(PredicateOp, Pred)>,
}

fn lop(t: &mut Translator<'_>, mut op_b: Value, m: Modifiers) -> TResult {
    let insn = t.insn;
    if m.x {
        return Err(t.unsupported());
    }
    let mut op_a = t.x(insn.src_a_reg());
    if m.inv_a {
        op_a = t.ir.bitwise_not(op_a);
    }
    if m.inv_b {
        op_b = t.ir.bitwise_not(op_b);
    }
    let result = match m.op {
        LogicalOp::And => t.ir.bitwise_and(op_a, op_b),
        LogicalOp::Or => t.ir.bitwise_or(op_a, op_b),
        LogicalOp::Xor => t.ir.bitwise_xor(op_a, op_b),
        LogicalOp::PassB => op_b,
    };
    if let Some((pred_op, dest_pred)) = m.pred {
        let pred_result = predicate_operation(&mut t.ir, result, pred_op);
        t.ir.set_pred(dest_pred, pred_result);
    }
    if m.cc {
        if m.op == LogicalOp::PassB {
            let z = t.ir.icompare(ICompare::Equal, result, Value::U32(0));
            t.ir.set_z_flag(z);
            let s = t.ir.icompare(ICompare::SLessThan, result, Value::U32(0));
            t.ir.set_s_flag(s);
        } else {
            let z = t.ir.get_zero_from_op(result);
            t.ir.set_z_flag(z);
            let s = t.ir.get_sign_from_op(result);
            t.ir.set_s_flag(s);
        }
        t.reset_c_flag();
        t.reset_o_flag();
    }
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

fn lop_common(t: &mut Translator<'_>, op_b: Value) -> TResult {
    let insn = t.insn;
    let m = Modifiers {
        x: insn.bit(43),
        cc: insn.bit(47),
        inv_a: insn.bit(39),
        inv_b: insn.bit(40),
        op: LogicalOp::from_bits(insn.bits(41, 2)),
        pred: Some((
            PredicateOp::from_bits(insn.bits(44, 2)),
            Pred::p(insn.bits(48, 3) as u8),
        )),
    };
    lop(t, op_b, m)
}

pub(super) fn lop_cbuf(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_cbuf()?;
    lop_common(t, op_b)
}

pub(super) fn lop_reg(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_reg20();
    lop_common(t, op_b)
}

pub(super) fn lop_imm(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_imm20();
    lop_common(t, op_b)
}

pub(super) fn lop32i(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let m = Modifiers {
        x: insn.bit(57),
        cc: insn.bit(52),
        inv_a: insn.bit(55),
        inv_b: insn.bit(56),
        op: LogicalOp::from_bits(insn.bits(53, 2)),
        pred: None,
    };
    let op_b = t.get_imm32();
    lop(t, op_b, m)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn lop_reg_and_zero_pred() {
        let insn = encode(Opcode::LopReg, (1 << 8) | (2 << 20) | (2 << 44) | (1 << 48));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R1
  %2 = BitwiseAnd32 %1, %0
  %3 = IEqual %2, #0x0
  SetPred P1, %3
  SetRegister R0, %2
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn lop_imm_pass_b_inverted_cc() {
        let insn = encode(
            Opcode::LopImm,
            (1 << 8) | (5 << 20) | (1 << 40) | (3 << 41) | (1 << 47) | (7 << 48),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = BitwiseNot32 #0x5
  %2 = IEqual %1, #0x0
  SetZFlag %2
  %3 = SLessThan %1, #0x0
  SetSFlag %3
  SetCFlag #false
  SetOFlag #false
  SetRegister R0, %1
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn lop_cbuf_or_inverted_a() {
        let insn = encode(
            Opcode::LopCbuf,
            (1 << 8) | (1 << 34) | (4 << 20) | (1 << 39) | (1 << 41) | (7 << 48),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R1
  %2 = BitwiseNot32 %1
  %3 = BitwiseOr32 %2, %0
  SetRegister R0, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn lop32i_xor_cc() {
        let insn = encode(
            Opcode::Lop32i,
            (1 << 8) | (0xff << 20) | (1 << 52) | (2 << 53),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = BitwiseXor32 %0, #0xff
  %2 = GetZeroFromOp %1
  SetZFlag %2
  %3 = GetSignFromOp %1
  SetSFlag %3
  SetCFlag #false
  SetOFlag #false
  SetRegister R0, %1
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn lop_x() {
        let insn = encode(Opcode::LopReg, 1 << 43);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
