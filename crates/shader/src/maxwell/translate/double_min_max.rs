//! DMNMX.

use super::{TResult, Translator};
use crate::ir::{FpControl, Pred, Value};

fn dmnmx(t: &mut Translator<'_>, src_b: Value) -> TResult {
    let i = t.insn;
    if i.bit(47) {
        return Err(t.unsupported());
    }
    let pred = t.ir.get_pred(Pred::p(i.bits(39, 3) as u8), false);
    let src_a = t.d(i.src_a_reg())?;
    let op_a = t.ir.fp_abs_neg(src_a, i.bit(46), i.bit(48));
    let op_b = t.ir.fp_abs_neg(src_b, i.bit(49), i.bit(45));
    let mut max = t.ir.fp_max(op_a, op_b, FpControl::default());
    let mut min = t.ir.fp_min(op_a, op_b, FpControl::default());
    if i.bit(42) {
        std::mem::swap(&mut min, &mut max);
    }
    let value = t.ir.select(pred, min, max);
    t.set_d(i.dest_reg(), value)
}

pub(super) fn dmnmx_cbuf(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_cbuf()?;
    dmnmx(t, src_b)
}

pub(super) fn dmnmx_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_reg20()?;
    dmnmx(t, src_b)
}

pub(super) fn dmnmx_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_imm20();
    dmnmx(t, src_b)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn dmnmx_reg_negated_predicate_swaps() {
        check(
            encode(
                Opcode::DmnmxReg,
                2 | 4 << 8 | 6 << 20 | 2 << 39 | 1 << 42 | 1 << 46,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R7",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = GetPred P2",
                "%5 = GetRegister R4",
                "%6 = GetRegister R5",
                "%7 = CompositeConstructU32x2 %5, %6",
                "%8 = PackDouble2x32 %7",
                "%9 = FPAbs64 %8",
                "%10 = FPMax64 %9, %3",
                "%11 = FPMin64 %9, %3",
                "%12 = SelectF64 %4, %10, %11",
                "%13 = UnpackDouble2x32 %12",
                "%14 = CompositeExtractU32x2 %13, #0x0",
                "SetRegister R2, %14",
                "%15 = CompositeExtractU32x2 %13, #0x1",
                "SetRegister R3, %15",
            ],
        );
    }

    #[test]
    fn dmnmx_cbuf() {
        check(
            encode(Opcode::DmnmxCbuf, 2 | 4 << 8 | 1 << 34 | 4 << 20 | 7 << 39),
            &[
                "%0 = GetCbufU32 #0x1, #0x14",
                "%1 = GetCbufU32 #0x1, #0x10",
                "%2 = CompositeConstructU32x2 %1, %0",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R4",
                "%5 = GetRegister R5",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = FPMax64 %7, %3",
                "%9 = FPMin64 %7, %3",
                "%10 = SelectF64 #true, %9, %8",
                "%11 = UnpackDouble2x32 %10",
                "%12 = CompositeExtractU32x2 %11, #0x0",
                "SetRegister R2, %12",
                "%13 = CompositeExtractU32x2 %11, #0x1",
                "SetRegister R3, %13",
            ],
        );
    }

    #[test]
    fn dmnmx_imm() {
        check(
            encode(Opcode::DmnmxImm, 2 | 4 << 8 | 0x3ff00 << 20 | 7 << 39),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = FPMax64 %3, #1.0d",
                "%5 = FPMin64 %3, #1.0d",
                "%6 = SelectF64 #true, %5, %4",
                "%7 = UnpackDouble2x32 %6",
                "%8 = CompositeExtractU32x2 %7, #0x0",
                "SetRegister R2, %8",
                "%9 = CompositeExtractU32x2 %7, #0x1",
                "SetRegister R3, %9",
            ],
        );
    }

    #[test]
    fn dmnmx_rejects_cc() {
        rejects(encode(Opcode::DmnmxReg, 2 | 4 << 8 | 6 << 20 | 1 << 47));
    }
}
