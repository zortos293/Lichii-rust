//! LOP3: three-input bitwise operation from an 8-bit truth table, ported from
//! logic_operation_three_input.cpp.

use super::{PredicateOp, TResult, Translator, predicate_operation};
use crate::TranslateError;
use crate::ir::{Pred, Value};

/// One step of a LUT expression in postfix order.
#[derive(Clone, Copy)]
enum Step {
    A,
    B,
    C,
    Zero,
    Ones,
    Not,
    And,
    Or,
    Xor,
}

use Step::{A, And, B, C, Not, Ones, Or, Xor, Zero};

/// The expression the C++ frontend emits for each truth table (its generated ApplyLUT
/// switch), in postfix order so the IR comes out in the same order.
const LUT: [&[Step]; 256] = [
    &[Zero],
    &[A, B, C, Or, Or, Not],
    &[C, A, B, Or, Not, And],
    &[A, B, Or, Not],
    &[B, A, C, Or, Not, And],
    &[A, C, Or, Not],
    &[A, Not, B, C, Xor, And],
    &[A, B, C, And, Or, Not],
    &[B, C, And, A, Not, And],
    &[A, B, C, Xor, Or, Not],
    &[C, A, Not, And],
    &[A, Not, C, B, Not, Or, And],
    &[B, A, Not, And],
    &[A, Not, B, C, Not, Or, And],
    &[A, Not, B, C, Or, And],
    &[A, Not],
    &[A, B, C, Or, Not, And],
    &[B, C, Or, Not],
    &[B, Not, A, C, Xor, And],
    &[B, A, C, And, Or, Not],
    &[C, Not, A, B, Xor, And],
    &[C, A, B, And, Or, Not],
    &[A, B, Or, C, A, B, And, Or, Xor],
    &[A, B, Xor, A, C, Xor, And, A, Not, Xor],
    &[A, B, Xor, A, C, Xor, And],
    &[A, B, And, B, C, Xor, Or, Not],
    &[C, B, Not, Or, A, C, Xor, And],
    &[A, C, Not, Or, B, C, Or, Xor],
    &[B, C, Not, Or, A, B, Xor, And],
    &[A, B, Not, Or, B, C, Or, Xor],
    &[A, B, C, Or, Xor],
    &[A, B, C, Or, And, Not],
    &[A, C, And, B, Not, And],
    &[B, A, C, Xor, Or, Not],
    &[C, B, Not, And],
    &[B, Not, C, A, Not, Or, And],
    &[A, B, Xor, B, C, Xor, And],
    &[A, B, And, A, C, Xor, Or, Not],
    &[C, A, Not, Or, B, C, Xor, And],
    &[A, C, Or, B, C, Not, Or, Xor],
    &[C, A, B, Xor, And],
    &[A, B, Or, A, B, And, C, Not, Or, Xor],
    &[C, A, B, And, Not, And],
    &[A, C, Not, Or, B, A, C, Xor, Or, Xor],
    &[B, C, Or, A, B, Xor, And],
    &[A, B, C, Not, Or, Xor],
    &[A, B, And, B, C, Or, Xor],
    &[C, B, Not, And, A, Not, Or],
    &[A, B, Not, And],
    &[B, Not, A, C, Not, Or, And],
    &[B, Not, A, C, Or, And],
    &[B, Not],
    &[A, C, Not, Or, A, B, Xor, And],
    &[A, C, Or, B, A, Not, Or, Xor],
    &[B, A, C, Or, Xor],
    &[B, A, C, Or, And, Not],
    &[A, C, Or, A, B, Xor, And],
    &[B, A, C, Not, Or, Xor],
    &[A, B, And, A, C, Or, Xor],
    &[C, A, Not, And, B, Not, Or],
    &[A, B, Xor],
    &[A, C, Or, Not, A, B, Xor, Or],
    &[C, A, Not, And, A, B, Xor, Or],
    &[A, B, And, Not],
    &[A, B, And, C, Not, And],
    &[C, A, B, Xor, Or, Not],
    &[A, C, Xor, B, C, Xor, And],
    &[A, C, And, A, B, Xor, Or, Not],
    &[B, C, Not, And],
    &[C, Not, B, A, Not, Or, And],
    &[B, A, Not, Or, B, C, Xor, And],
    &[A, B, Or, C, B, Not, Or, Xor],
    &[B, A, C, Xor, And],
    &[A, C, Or, A, C, And, B, Not, Or, Xor],
    &[B, C, Or, A, C, Xor, And],
    &[A, C, B, Not, Or, Xor],
    &[B, A, C, And, Not, And],
    &[A, B, Not, Or, C, A, B, Xor, Or, Xor],
    &[A, C, And, B, C, Or, Xor],
    &[B, C, Not, And, A, Not, Or],
    &[A, C, Not, And],
    &[C, Not, A, B, Not, Or, And],
    &[A, B, Not, Or, A, C, Xor, And],
    &[A, B, Or, C, A, Not, Or, Xor],
    &[C, Not, A, B, Or, And],
    &[C, Not],
    &[C, A, B, Or, Xor],
    &[C, A, B, Or, And, Not],
    &[A, B, Or, A, C, Xor, And],
    &[C, A, B, Not, Or, Xor],
    &[A, C, Xor],
    &[A, B, Or, Not, A, C, Xor, Or],
    &[A, C, And, A, B, Or, Xor],
    &[B, A, Not, And, C, Not, Or],
    &[B, A, Not, And, A, C, Xor, Or],
    &[A, C, And, Not],
    &[A, B, C, Xor, And],
    &[B, C, Or, B, C, And, A, Not, Or, Xor],
    &[A, C, Or, B, C, Xor, And],
    &[B, C, A, Not, Or, Xor],
    &[A, B, Or, B, C, Xor, And],
    &[C, B, A, Not, Or, Xor],
    &[B, C, Xor],
    &[A, B, Or, Not, B, C, Xor, Or],
    &[A, B, Or, C, A, B, And, Xor, And],
    &[A, Not, B, C, Xor, Xor],
    &[C, A, B, And, Xor],
    &[C, A, B, Or, And, A, B, Not, Xor, Xor],
    &[B, A, C, And, Xor],
    &[B, A, C, Or, And, A, C, Not, Xor, Xor],
    &[B, A, Not, And, B, C, Xor, Or],
    &[A, Not, B, C, Xor, Or],
    &[A, B, C, And, Not, And],
    &[B, A, Not, Or, C, A, B, Xor, Or, Xor],
    &[B, C, And, A, C, Or, Xor],
    &[A, C, Not, And, B, Not, Or],
    &[B, C, And, A, B, Or, Xor],
    &[A, B, Not, And, C, Not, Or],
    &[A, B, Not, And, B, C, Xor, Or],
    &[B, C, And, Not],
    &[A, B, C, And, Xor],
    &[A, B, C, Or, And, B, C, Not, Xor, Xor],
    &[A, B, Not, And, A, C, Xor, Or],
    &[B, Not, A, C, Xor, Or],
    &[A, C, Not, And, A, B, Xor, Or],
    &[C, Not, A, B, Xor, Or],
    &[A, B, Xor, A, C, Xor, Or],
    &[A, B, C, And, And, Not],
    &[A, B, C, And, And],
    &[A, B, Xor, A, C, Xor, Or, Not],
    &[C, A, B, Not, Xor, And],
    &[C, A, Not, Or, A, B, Not, Xor, And],
    &[B, A, C, Not, Xor, And],
    &[B, A, Not, Or, A, C, Not, Xor, And],
    &[B, C, Or, A, B, C, Xor, Xor, And],
    &[B, C, And, A, Not, Xor],
    &[B, C, And],
    &[B, A, Not, Or, B, C, Not, Xor, And],
    &[C, B, A, Not, Or, And],
    &[B, C, And, A, B, Or, Not, Or],
    &[B, C, A, Not, Or, And],
    &[B, C, And, A, C, Or, Not, Or],
    &[A, A, B, Xor, A, C, Xor, Or, Xor],
    &[B, C, And, A, Not, Or],
    &[A, B, C, Not, Xor, And],
    &[A, B, Not, Or, B, C, Not, Xor, And],
    &[A, C, Or, A, B, C, Xor, Xor, And],
    &[A, C, And, B, Not, Xor],
    &[A, B, Or, A, B, C, Xor, Xor, And],
    &[A, B, And, C, Not, Xor],
    &[A, B, C, Xor, Xor],
    &[A, B, Or, Not, A, B, C, Xor, Xor, Or],
    &[A, B, Or, B, C, Not, Xor, And],
    &[B, C, Not, Xor],
    &[C, A, B, Not, And, Xor],
    &[A, B, Or, B, C, Xor, And, Not],
    &[B, A, C, Not, And, Xor],
    &[A, C, Or, B, C, Xor, And, Not],
    &[B, C, And, A, B, C, Or, Xor, Or],
    &[A, B, C, Xor, And, Not],
    &[A, C, And],
    &[A, B, Not, Or, A, C, Not, Xor, And],
    &[C, A, B, Not, Or, And],
    &[A, C, And, A, B, Or, Not, Or],
    &[A, B, Or, A, C, Not, Xor, And],
    &[A, C, Not, Xor],
    &[C, B, A, Not, And, Xor],
    &[A, B, Or, A, C, Xor, And, Not],
    &[C, A, B, Or, And],
    &[C, Not, A, B, Or, Xor],
    &[C],
    &[C, A, B, Or, Not, Or],
    &[A, B, Or, C, A, Not, Or, And],
    &[B, C, And, A, C, Not, Xor, Or],
    &[C, B, A, Not, And, Or],
    &[C, A, Not, Or],
    &[A, C, B, Not, Or, And],
    &[A, C, And, B, C, Or, Not, Or],
    &[B, A, B, Xor, A, C, Xor, Or, Xor],
    &[A, C, And, B, Not, Or],
    &[A, B, C, Not, And, Xor],
    &[B, C, Or, A, C, Xor, And, Not],
    &[A, C, And, B, A, C, Or, Xor, Or],
    &[B, A, C, Xor, And, Not],
    &[A, B, Or, C, B, Not, Or, And],
    &[A, C, And, B, C, Not, Xor, Or],
    &[C, A, B, Not, And, Or],
    &[C, B, Not, Or],
    &[A, C, And, A, B, Xor, Or],
    &[A, B, Xor, A, C, Not, Xor, Or],
    &[C, A, B, Xor, Or],
    &[C, A, B, And, Not, Or],
    &[A, B, And],
    &[A, C, Not, Or, A, B, Not, Xor, And],
    &[A, C, Or, A, B, Not, Xor, And],
    &[A, B, Not, Xor],
    &[B, A, C, Not, Or, And],
    &[A, B, And, A, C, Or, Not, Or],
    &[B, C, A, Not, And, Xor],
    &[A, C, Or, A, B, Xor, And, Not],
    &[B, A, C, Or, And],
    &[B, Not, A, C, Or, Xor],
    &[A, C, Or, B, A, Not, Or, And],
    &[B, C, And, A, B, Not, Xor, Or],
    &[B],
    &[B, A, C, Or, Not, Or],
    &[B, C, A, Not, And, Or],
    &[B, A, Not, Or],
    &[A, B, C, Not, Or, And],
    &[A, B, And, B, C, Or, Not, Or],
    &[A, C, B, Not, And, Xor],
    &[B, C, Or, A, B, Xor, And, Not],
    &[C, A, B, Xor, A, C, Xor, Or, Xor],
    &[A, B, And, C, Not, Or],
    &[A, B, And, C, A, B, Or, Xor, Or],
    &[C, A, B, Xor, And, Not],
    &[A, C, Or, B, C, Not, Or, And],
    &[A, B, And, B, C, Not, Xor, Or],
    &[A, B, And, A, C, Xor, Or],
    &[A, C, Xor, A, B, Not, Xor, Or],
    &[B, A, C, Not, And, Or],
    &[B, C, Not, Or],
    &[B, A, C, Xor, Or],
    &[B, A, C, And, Not, Or],
    &[A, B, C, Or, And],
    &[A, Not, B, C, Or, Xor],
    &[A, B, Not, Or, B, C, Or, And],
    &[A, C, And, A, B, Not, Xor, Or],
    &[A, C, Not, Or, B, C, Or, And],
    &[A, B, And, A, C, Not, Xor, Or],
    &[A, B, And, B, C, Xor, Or],
    &[A, B, Not, Xor, B, C, Xor, Or],
    &[A, B, Or, C, A, B, And, Or, And],
    &[A, B, And, C, Not, A, B, Or, Xor, Or],
    &[C, A, B, And, Or],
    &[C, A, B, Not, Xor, Or],
    &[B, A, C, And, Or],
    &[B, A, C, Not, Xor, Or],
    &[B, C, Or],
    &[A, Not, B, C, Or, Or],
    &[A],
    &[A, B, C, Or, Not, Or],
    &[A, C, B, Not, And, Or],
    &[A, B, Not, Or],
    &[A, B, C, Not, And, Or],
    &[A, C, Not, Or],
    &[A, B, C, Xor, Or],
    &[A, B, C, And, Not, Or],
    &[A, B, C, And, Or],
    &[A, B, C, Not, Xor, Or],
    &[A, C, Or],
    &[B, Not, A, C, Or, Or],
    &[A, B, Or],
    &[C, Not, A, B, Or, Or],
    &[A, B, C, Or, Or],
    &[Ones],
];

fn apply_lut(t: &mut Translator<'_>, a: Value, b: Value, c: Value, ttbl: u8) -> Value {
    let mut stack = [Value::Void; 8];
    let mut depth = 0;
    for &step in LUT[ttbl as usize] {
        let value = match step {
            Step::A => a,
            Step::B => b,
            Step::C => c,
            Step::Zero => Value::U32(0),
            Step::Ones => Value::U32(u32::MAX),
            Step::Not => {
                depth -= 1;
                t.ir.bitwise_not(stack[depth])
            }
            Step::And | Step::Or | Step::Xor => {
                depth -= 2;
                let (lhs, rhs) = (stack[depth], stack[depth + 1]);
                match step {
                    Step::And => t.ir.bitwise_and(lhs, rhs),
                    Step::Or => t.ir.bitwise_or(lhs, rhs),
                    _ => t.ir.bitwise_xor(lhs, rhs),
                }
            }
        };
        stack[depth] = value;
        depth += 1;
    }
    stack[0]
}

fn lop3(
    t: &mut Translator<'_>,
    op_b: Value,
    op_c: Value,
    lut: u8,
) -> Result<Value, TranslateError> {
    let insn = t.insn;
    if insn.bit(47) {
        return Err(t.unsupported());
    }
    let op_a = t.x(insn.src_a_reg());
    let result = apply_lut(t, op_a, op_b, op_c, lut);
    t.set_x(insn.dest_reg(), result);
    Ok(result)
}

pub(super) fn lop3_reg(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    if insn.bit(38) {
        return Err(t.unsupported());
    }
    let op_b = t.get_reg20();
    let op_c = t.get_reg39();
    let result = lop3(t, op_b, op_c, insn.bits(28, 8) as u8)?;
    let pred_result =
        predicate_operation(&mut t.ir, result, PredicateOp::from_bits(insn.bits(36, 2)));
    t.ir.set_pred(Pred::p(insn.bits(48, 3) as u8), pred_result);
    Ok(())
}

pub(super) fn lop3_cbuf(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let op_b = t.get_cbuf()?;
    let op_c = t.get_reg39();
    lop3(t, op_b, op_c, insn.bits(48, 8) as u8).map(|_| ())
}

pub(super) fn lop3_imm(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let op_b = t.get_imm20();
    let op_c = t.get_reg39();
    lop3(t, op_b, op_c, insn.bits(48, 8) as u8).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use super::{LUT, Step};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn lop3_reg_xor3_nonzero_pred() {
        let insn = encode(
            Opcode::Lop3Reg,
            (1 << 8) | (2 << 20) | (3 << 39) | (0x96 << 28) | (3 << 36) | (2 << 48),
        );
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = GetRegister R1
  %3 = BitwiseXor32 %0, %1
  %4 = BitwiseXor32 %2, %3
  SetRegister R0, %4
  %5 = INotEqual %4, #0x0
  SetPred P2, %5
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn lop3_imm_and3() {
        let insn = encode(
            Opcode::Lop3Imm,
            (1 << 8) | (5 << 20) | (3 << 39) | (0x80 << 48),
        );
        let expected = "\
{0}:
  %0 = GetRegister R3
  %1 = GetRegister R1
  %2 = BitwiseAnd32 #0x5, %0
  %3 = BitwiseAnd32 %1, %2
  SetRegister R0, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn lop3_cbuf_zero() {
        let insn = encode(
            Opcode::Lop3Cbuf,
            (1 << 8) | (1 << 34) | (4 << 20) | (3 << 39),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R3
  %2 = GetRegister R1
  SetRegister R0, #0x0
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn lop3_x() {
        let insn = encode(Opcode::Lop3Reg, 1 << 38);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn lop3_cc() {
        let insn = encode(Opcode::Lop3Imm, 1 << 47);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn lut_expressions_match_their_truth_tables() {
        for (ttbl, steps) in LUT.iter().enumerate() {
            let mut stack = [0u8; 8];
            let mut depth = 0;
            for step in steps.iter() {
                let value = match step {
                    Step::A => 0xf0,
                    Step::B => 0xcc,
                    Step::C => 0xaa,
                    Step::Zero => 0,
                    Step::Ones => 0xff,
                    Step::Not => {
                        depth -= 1;
                        !stack[depth]
                    }
                    Step::And | Step::Or | Step::Xor => {
                        depth -= 2;
                        let (lhs, rhs) = (stack[depth], stack[depth + 1]);
                        match step {
                            Step::And => lhs & rhs,
                            Step::Or => lhs | rhs,
                            _ => lhs ^ rhs,
                        }
                    }
                };
                stack[depth] = value;
                depth += 1;
            }
            assert_eq!(depth, 1, "LUT {ttbl:#x}");
            assert_eq!(usize::from(stack[0]), ttbl, "LUT {ttbl:#x}");
        }
    }
}
