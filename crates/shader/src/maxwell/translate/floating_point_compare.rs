//! FCMP.

use super::floating_point_add::ftz_mode;
use super::{FpCompareOp, TResult, Translator, floating_point_compare};
use crate::ir::{FpControl, Value};

fn fcmp(t: &mut Translator<'_>, src_a: Value, operand: Value) -> TResult {
    let i = t.insn;
    let control = FpControl {
        fmz_mode: ftz_mode(i.bit(47)),
        ..FpControl::default()
    };
    let compare_op = FpCompareOp::from_bits(i.bits(48, 4));
    let cmp = floating_point_compare(&mut t.ir, operand, Value::F32(0.0), compare_op, control);
    let src_reg = t.x(i.src_a_reg());
    let result = t.ir.select(cmp, src_reg, src_a);
    t.set_x(i.dest_reg(), result);
    Ok(())
}

pub(super) fn fcmp_cr(t: &mut Translator<'_>) -> TResult {
    let src_a = t.get_cbuf()?;
    let operand = t.get_float_reg39();
    fcmp(t, src_a, operand)
}

pub(super) fn fcmp_rc(t: &mut Translator<'_>) -> TResult {
    let src_a = t.get_reg39();
    let operand = t.get_float_cbuf()?;
    fcmp(t, src_a, operand)
}

pub(super) fn fcmp_reg(t: &mut Translator<'_>) -> TResult {
    let src_a = t.get_reg20();
    let operand = t.get_float_reg39();
    fcmp(t, src_a, operand)
}

pub(super) fn fcmp_imm(t: &mut Translator<'_>) -> TResult {
    let sign = if t.insn.bit(56) { 1 << 31 } else { 0 };
    let src_a = Value::U32(((t.insn.bits(20, 19) as u32) << 12) | sign);
    let operand = t.get_float_reg39();
    fcmp(t, src_a, operand)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::check;
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn fcmp_reg_ftz() {
        check(
            encode(
                Opcode::FcmpReg,
                2 | 4 << 8 | 6 << 20 | 8 << 39 | 1 << 47 | 4 << 48,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R8",
                "%2 = BitCastF32U32 %1",
                "%3 = FPOrdGreaterThan32 %2, #0.0f (flags 0x10)",
                "%4 = GetRegister R4",
                "%5 = SelectU32 %3, %4, %0",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn fcmp_rc() {
        check(
            encode(
                Opcode::FcmpRc,
                2 | 4 << 8 | 1 << 34 | 3 << 20 | 8 << 39 | 1 << 48,
            ),
            &[
                "%0 = GetRegister R8",
                "%1 = GetCbufF32 #0x1, #0xc",
                "%2 = FPOrdLessThan32 %1, #0.0f (flags 0x30)",
                "%3 = GetRegister R4",
                "%4 = SelectU32 %2, %3, %0",
                "SetRegister R2, %4",
            ],
        );
    }

    #[test]
    fn fcmp_cr() {
        check(
            encode(
                Opcode::FcmpCr,
                2 | 4 << 8 | 1 << 34 | 3 << 20 | 8 << 39 | 13 << 48,
            ),
            &[
                "%0 = GetCbufU32 #0x1, #0xc",
                "%1 = GetRegister R8",
                "%2 = BitCastF32U32 %1",
                "%3 = FPUnordNotEqual32 %2, #0.0f (flags 0x30)",
                "%4 = GetRegister R4",
                "%5 = SelectU32 %3, %4, %0",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn fcmp_imm() {
        check(
            encode(
                Opcode::FcmpImm,
                2 | 4 << 8 | 0x3f800 << 20 | 1 << 56 | 8 << 39 | 7 << 48,
            ),
            &[
                "%0 = GetRegister R8",
                "%1 = BitCastF32U32 %0",
                "%2 = FPIsNan32 %1",
                "%3 = FPIsNan32 #0.0f",
                "%4 = LogicalNot %2",
                "%5 = LogicalNot %3",
                "%6 = LogicalAnd %4, %5",
                "%7 = GetRegister R4",
                "%8 = SelectU32 %6, %7, #0xbf800000",
                "SetRegister R2, %8",
            ],
        );
    }
}
