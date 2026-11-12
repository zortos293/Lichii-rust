//! FMNMX.

use super::floating_point_add::ftz_mode;
use super::{TResult, Translator};
use crate::ir::{FpControl, FpRounding, Pred, Value};

fn fmnmx(t: &mut Translator<'_>, src_b: Value) -> TResult {
    let i = t.insn;
    if i.bit(47) {
        return Err(t.unsupported());
    }
    let pred = t.ir.get_pred(Pred::p(i.bits(39, 3) as u8), false);
    let src_a = t.f(i.src_a_reg());
    let op_a = t.ir.fp_abs_neg(src_a, i.bit(46), i.bit(48));
    let op_b = t.ir.fp_abs_neg(src_b, i.bit(49), i.bit(45));
    let control = FpControl {
        no_contraction: false,
        rounding: FpRounding::DontCare,
        fmz_mode: ftz_mode(i.bit(44)),
    };
    let mut max = t.ir.fp_max(op_a, op_b, control);
    let mut min = t.ir.fp_min(op_a, op_b, control);
    if i.bit(42) {
        std::mem::swap(&mut min, &mut max);
    }
    let value = t.ir.select(pred, min, max);
    t.set_f(i.dest_reg(), value);
    Ok(())
}

pub(super) fn fmnmx_cbuf(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_cbuf()?;
    fmnmx(t, src_b)
}

pub(super) fn fmnmx_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_reg20();
    fmnmx(t, src_b)
}

pub(super) fn fmnmx_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_imm20();
    fmnmx(t, src_b)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn fmnmx_reg_ftz_negated_predicate() {
        check(
            encode(
                Opcode::FmnmxReg,
                2 | 4 << 8 | 6 << 20 | 1 << 39 | 1 << 42 | 1 << 44 | 1 << 49,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = GetPred P1",
                "%3 = GetRegister R4",
                "%4 = BitCastF32U32 %3",
                "%5 = FPAbs32 %1",
                "%6 = FPMax32 %4, %5 (flags 0x10)",
                "%7 = FPMin32 %4, %5 (flags 0x10)",
                "%8 = SelectF32 %2, %6, %7",
                "%9 = BitCastU32F32 %8",
                "SetRegister R2, %9",
            ],
        );
    }

    #[test]
    fn fmnmx_cbuf() {
        check(
            encode(Opcode::FmnmxCbuf, 2 | 4 << 8 | 1 << 34 | 3 << 20 | 7 << 39),
            &[
                "%0 = GetCbufF32 #0x1, #0xc",
                "%1 = GetRegister R4",
                "%2 = BitCastF32U32 %1",
                "%3 = FPMax32 %2, %0 (flags 0x30)",
                "%4 = FPMin32 %2, %0 (flags 0x30)",
                "%5 = SelectF32 #true, %4, %3",
                "%6 = BitCastU32F32 %5",
                "SetRegister R2, %6",
            ],
        );
    }

    #[test]
    fn fmnmx_imm() {
        check(
            encode(
                Opcode::FmnmxImm,
                2 | 4 << 8 | 0x3f800 << 20 | 7 << 39 | 1 << 45 | 1 << 48,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPNeg32 %1",
                "%3 = FPNeg32 #1.0f",
                "%4 = FPMax32 %2, %3 (flags 0x30)",
                "%5 = FPMin32 %2, %3 (flags 0x30)",
                "%6 = SelectF32 #true, %5, %4",
                "%7 = BitCastU32F32 %6",
                "SetRegister R2, %7",
            ],
        );
    }

    #[test]
    fn fmnmx_rejects_cc() {
        rejects(encode(Opcode::FmnmxReg, 2 | 4 << 8 | 6 << 20 | 1 << 47));
    }
}
