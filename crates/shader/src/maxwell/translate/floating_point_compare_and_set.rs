//! FSET.

use super::floating_point_add::ftz_mode;
use super::{
    BooleanOp, FpCompareOp, TResult, Translator, floating_point_compare, predicate_combine,
};
use crate::ir::{FpControl, FpRounding, ICompare, Pred, Value};

/// Writes the result of FSET or DSET: all ones or 1.0 (BF, bit 52) when `pass`, else 0, and the
/// condition codes when CC (bit 47) is set.
pub(super) fn write_set_result(t: &mut Translator<'_>, pass: Value) {
    let i = t.insn;
    let bf = i.bit(52);
    let pass_result = Value::U32(if bf { 0x3f80_0000 } else { u32::MAX });
    let zero = Value::U32(0);
    let result = t.ir.select(pass, pass_result, zero);
    t.set_x(i.dest_reg(), result);
    if i.bit(47) {
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
}

fn fset(t: &mut Translator<'_>, src_b: Value) -> TResult {
    let i = t.insn;
    let Some(bop) = BooleanOp::from_bits(i.bits(45, 2)) else {
        return Err(t.unsupported());
    };
    let src_a = t.f(i.src_a_reg());
    let op_a = t.ir.fp_abs_neg(src_a, i.bit(54), i.bit(43));
    let op_b = t.ir.fp_abs_neg(src_b, i.bit(44), i.bit(53));
    let control = FpControl {
        no_contraction: false,
        rounding: FpRounding::DontCare,
        fmz_mode: ftz_mode(i.bit(55)),
    };
    let pred = t.ir.get_pred(Pred::p(i.bits(39, 3) as u8), i.bit(42));
    let compare_op = FpCompareOp::from_bits(i.bits(48, 4));
    let cmp = floating_point_compare(&mut t.ir, op_a, op_b, compare_op, control);
    let result = predicate_combine(&mut t.ir, cmp, pred, bop);
    write_set_result(t, result);
    Ok(())
}

pub(super) fn fset_cbuf(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_cbuf()?;
    fset(t, src_b)
}

pub(super) fn fset_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_reg20();
    fset(t, src_b)
}

pub(super) fn fset_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_imm20();
    fset(t, src_b)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn fset_reg_mask_with_condition_codes() {
        check(
            encode(
                Opcode::FsetReg,
                2 | 4 << 8 | 6 << 20 | 1 << 39 | 1 << 42 | 1 << 45 | 1 << 47 | 1 << 48 | 1 << 55,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R4",
                "%3 = BitCastF32U32 %2",
                "%4 = GetPred P1",
                "%5 = LogicalNot %4",
                "%6 = FPOrdLessThan32 %3, %1 (flags 0x10)",
                "%7 = LogicalOr %6, %5",
                "%8 = SelectU32 %7, #0xffffffff, #0x0",
                "SetRegister R2, %8",
                "%9 = IEqual %8, #0x0",
                "SetZFlag %9",
                "%10 = LogicalNot %9",
                "SetSFlag %10",
                "SetCFlag #false",
                "SetOFlag #false",
            ],
        );
    }

    #[test]
    fn fset_cbuf_bf() {
        check(
            encode(
                Opcode::FsetCbuf,
                2 | 4 << 8 | 1 << 34 | 3 << 20 | 7 << 39 | 12 << 48 | 1 << 52,
            ),
            &[
                "%0 = GetCbufF32 #0x1, #0xc",
                "%1 = GetRegister R4",
                "%2 = BitCastF32U32 %1",
                "%3 = FPUnordGreaterThan32 %2, %0 (flags 0x30)",
                "%4 = LogicalAnd %3, #true",
                "%5 = SelectU32 %4, #0x3f800000, #0x0",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn fset_imm() {
        check(
            encode(
                Opcode::FsetImm,
                2 | 4 << 8 | 0x3f800 << 20 | 7 << 39 | 2 << 48,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPOrdEqual32 %1, #1.0f (flags 0x30)",
                "%3 = LogicalAnd %2, #true",
                "%4 = SelectU32 %3, #0xffffffff, #0x0",
                "SetRegister R2, %4",
            ],
        );
    }

    #[test]
    fn fset_rejects_reserved_bop() {
        rejects(encode(Opcode::FsetReg, 2 | 4 << 8 | 6 << 20 | 3 << 45));
    }
}
