//! DFMA.

use super::floating_point_add::fp_rounding;
use super::{TResult, Translator};
use crate::ir::{FmzMode, FpControl, Reg, Value};

fn dfma(t: &mut Translator<'_>, src_b: Value, src_c: Value) -> TResult {
    let i = t.insn;
    if i.bit(47) {
        return Err(t.unsupported());
    }
    let src_a = t.d(i.src_a_reg())?;
    let op_b = t.ir.fp_abs_neg(src_b, false, i.bit(48));
    let op_c = t.ir.fp_abs_neg(src_c, false, i.bit(49));
    let control = FpControl {
        no_contraction: true,
        rounding: fp_rounding(i.bits(50, 2)),
        fmz_mode: FmzMode::None,
    };
    let value = t.ir.fp_fma(src_a, op_b, op_c, control);
    t.set_d(i.dest_reg(), value)
}

fn reg39(t: &mut Translator<'_>) -> Result<Value, crate::TranslateError> {
    t.d(Reg(t.insn.bits(39, 8) as u16))
}

pub(super) fn dfma_cr(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_cbuf()?;
    let src_c = reg39(t)?;
    dfma(t, src_b, src_c)
}

pub(super) fn dfma_rc(t: &mut Translator<'_>) -> TResult {
    let src_b = reg39(t)?;
    let src_c = t.get_double_cbuf()?;
    dfma(t, src_b, src_c)
}

pub(super) fn dfma_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_reg20()?;
    let src_c = reg39(t)?;
    dfma(t, src_b, src_c)
}

pub(super) fn dfma_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_imm20();
    let src_c = reg39(t)?;
    dfma(t, src_b, src_c)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn dfma_reg_rounding_and_negation() {
        check(
            encode(
                Opcode::DfmaReg,
                2 | 4 << 8 | 6 << 20 | 8 << 39 | 3 << 50 | 1 << 48 | 1 << 49,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R7",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R8",
                "%5 = GetRegister R9",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = GetRegister R4",
                "%9 = GetRegister R5",
                "%10 = CompositeConstructU32x2 %8, %9",
                "%11 = PackDouble2x32 %10",
                "%12 = FPNeg64 %3",
                "%13 = FPNeg64 %7",
                "%14 = FPFma64 %11, %12, %13 (flags 0x39)",
                "%15 = UnpackDouble2x32 %14",
                "%16 = CompositeExtractU32x2 %15, #0x0",
                "SetRegister R2, %16",
                "%17 = CompositeExtractU32x2 %15, #0x1",
                "SetRegister R3, %17",
            ],
        );
    }

    #[test]
    fn dfma_cr() {
        check(
            encode(Opcode::DfmaCr, 2 | 4 << 8 | 1 << 34 | 4 << 20 | 8 << 39),
            &[
                "%0 = GetCbufU32 #0x1, #0x14",
                "%1 = GetCbufU32 #0x1, #0x10",
                "%2 = CompositeConstructU32x2 %1, %0",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R8",
                "%5 = GetRegister R9",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = GetRegister R4",
                "%9 = GetRegister R5",
                "%10 = CompositeConstructU32x2 %8, %9",
                "%11 = PackDouble2x32 %10",
                "%12 = FPFma64 %11, %3, %7 (flags 0x33)",
                "%13 = UnpackDouble2x32 %12",
                "%14 = CompositeExtractU32x2 %13, #0x0",
                "SetRegister R2, %14",
                "%15 = CompositeExtractU32x2 %13, #0x1",
                "SetRegister R3, %15",
            ],
        );
    }

    #[test]
    fn dfma_rc() {
        check(
            encode(Opcode::DfmaRc, 2 | 4 << 8 | 1 << 34 | 4 << 20 | 8 << 39),
            &[
                "%0 = GetRegister R8",
                "%1 = GetRegister R9",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = GetCbufU32 #0x1, #0x14",
                "%5 = GetCbufU32 #0x1, #0x10",
                "%6 = CompositeConstructU32x2 %5, %4",
                "%7 = PackDouble2x32 %6",
                "%8 = GetRegister R4",
                "%9 = GetRegister R5",
                "%10 = CompositeConstructU32x2 %8, %9",
                "%11 = PackDouble2x32 %10",
                "%12 = FPFma64 %11, %3, %7 (flags 0x33)",
                "%13 = UnpackDouble2x32 %12",
                "%14 = CompositeExtractU32x2 %13, #0x0",
                "SetRegister R2, %14",
                "%15 = CompositeExtractU32x2 %13, #0x1",
                "SetRegister R3, %15",
            ],
        );
    }

    #[test]
    fn dfma_imm() {
        check(
            encode(Opcode::DfmaImm, 2 | 4 << 8 | 0x3ff00 << 20 | 8 << 39),
            &[
                "%0 = GetRegister R8",
                "%1 = GetRegister R9",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R4",
                "%5 = GetRegister R5",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = FPFma64 %7, #1.0d, %3 (flags 0x33)",
                "%9 = UnpackDouble2x32 %8",
                "%10 = CompositeExtractU32x2 %9, #0x0",
                "SetRegister R2, %10",
                "%11 = CompositeExtractU32x2 %9, #0x1",
                "SetRegister R3, %11",
            ],
        );
    }

    #[test]
    fn dfma_rejects_cc() {
        rejects(encode(
            Opcode::DfmaReg,
            2 | 4 << 8 | 6 << 20 | 8 << 39 | 1 << 47,
        ));
    }
}
