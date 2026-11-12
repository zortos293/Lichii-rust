//! DMUL.

use super::floating_point_add::fp_rounding;
use super::{TResult, Translator};
use crate::ir::{FmzMode, FpControl, Value};

fn dmul(t: &mut Translator<'_>, src_b: Value) -> TResult {
    let i = t.insn;
    if i.bit(47) {
        return Err(t.unsupported());
    }
    let src_a = t.d(i.src_a_reg())?;
    let op_a = t.ir.fp_abs_neg(src_a, false, i.bit(48));
    let control = FpControl {
        no_contraction: true,
        rounding: fp_rounding(i.bits(39, 2)),
        fmz_mode: FmzMode::None,
    };
    let value = t.ir.fp_mul(op_a, src_b, control);
    t.set_d(i.dest_reg(), value)
}

pub(super) fn dmul_cbuf(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_cbuf()?;
    dmul(t, src_b)
}

pub(super) fn dmul_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_reg20()?;
    dmul(t, src_b)
}

pub(super) fn dmul_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_imm20();
    dmul(t, src_b)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn dmul_reg_negated_rp() {
        check(
            encode(Opcode::DmulReg, 2 | 4 << 8 | 6 << 20 | 2 << 39 | 1 << 48),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R7",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R4",
                "%5 = GetRegister R5",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = FPNeg64 %7",
                "%9 = FPMul64 %8, %3 (flags 0x37)",
                "%10 = UnpackDouble2x32 %9",
                "%11 = CompositeExtractU32x2 %10, #0x0",
                "SetRegister R2, %11",
                "%12 = CompositeExtractU32x2 %10, #0x1",
                "SetRegister R3, %12",
            ],
        );
    }

    #[test]
    fn dmul_cbuf() {
        check(
            encode(Opcode::DmulCbuf, 2 | 4 << 8 | 1 << 34 | 4 << 20),
            &[
                "%0 = GetCbufU32 #0x1, #0x14",
                "%1 = GetCbufU32 #0x1, #0x10",
                "%2 = CompositeConstructU32x2 %1, %0",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R4",
                "%5 = GetRegister R5",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = FPMul64 %7, %3 (flags 0x33)",
                "%9 = UnpackDouble2x32 %8",
                "%10 = CompositeExtractU32x2 %9, #0x0",
                "SetRegister R2, %10",
                "%11 = CompositeExtractU32x2 %9, #0x1",
                "SetRegister R3, %11",
            ],
        );
    }

    #[test]
    fn dmul_imm() {
        check(
            encode(Opcode::DmulImm, 2 | 4 << 8 | 0x3ff00 << 20),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = FPMul64 %3, #1.0d (flags 0x33)",
                "%5 = UnpackDouble2x32 %4",
                "%6 = CompositeExtractU32x2 %5, #0x0",
                "SetRegister R2, %6",
                "%7 = CompositeExtractU32x2 %5, #0x1",
                "SetRegister R3, %7",
            ],
        );
    }

    #[test]
    fn dmul_rejects_cc() {
        rejects(encode(Opcode::DmulReg, 2 | 4 << 8 | 6 << 20 | 1 << 47));
    }
}
