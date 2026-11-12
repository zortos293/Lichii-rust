//! DADD.

use super::floating_point_add::fp_rounding;
use super::{TResult, Translator};
use crate::ir::{FmzMode, FpControl, Value};

fn dadd(t: &mut Translator<'_>, src_b: Value) -> TResult {
    let i = t.insn;
    if i.bit(47) {
        return Err(t.unsupported());
    }
    let src_a = t.d(i.src_a_reg())?;
    let op_a = t.ir.fp_abs_neg(src_a, i.bit(46), i.bit(48));
    let op_b = t.ir.fp_abs_neg(src_b, i.bit(49), i.bit(45));
    let control = FpControl {
        no_contraction: true,
        rounding: fp_rounding(i.bits(39, 2)),
        fmz_mode: FmzMode::None,
    };
    let value = t.ir.fp_add(op_a, op_b, control);
    t.set_d(i.dest_reg(), value)
}

pub(super) fn dadd_cbuf(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_cbuf()?;
    dadd(t, src_b)
}

pub(super) fn dadd_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_reg20()?;
    dadd(t, src_b)
}

pub(super) fn dadd_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_imm20();
    dadd(t, src_b)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn dadd_reg_rounding_and_modifiers() {
        check(
            encode(
                Opcode::DaddReg,
                2 | 4 << 8 | 6 << 20 | 1 << 39 | 1 << 45 | 1 << 46,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R7",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R4",
                "%5 = GetRegister R5",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = FPAbs64 %7",
                "%9 = FPNeg64 %3",
                "%10 = FPAdd64 %8, %9 (flags 0x35)",
                "%11 = UnpackDouble2x32 %10",
                "%12 = CompositeExtractU32x2 %11, #0x0",
                "SetRegister R2, %12",
                "%13 = CompositeExtractU32x2 %11, #0x1",
                "SetRegister R3, %13",
            ],
        );
    }

    #[test]
    fn dadd_cbuf() {
        check(
            encode(Opcode::DaddCbuf, 2 | 4 << 8 | 1 << 34 | 4 << 20),
            &[
                "%0 = GetCbufU32 #0x1, #0x14",
                "%1 = GetCbufU32 #0x1, #0x10",
                "%2 = CompositeConstructU32x2 %1, %0",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R4",
                "%5 = GetRegister R5",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = FPAdd64 %7, %3 (flags 0x33)",
                "%9 = UnpackDouble2x32 %8",
                "%10 = CompositeExtractU32x2 %9, #0x0",
                "SetRegister R2, %10",
                "%11 = CompositeExtractU32x2 %9, #0x1",
                "SetRegister R3, %11",
            ],
        );
    }

    #[test]
    fn dadd_imm() {
        check(
            encode(
                Opcode::DaddImm,
                2 | 4 << 8 | 0x3ff00 << 20 | 1 << 48 | 1 << 49,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = FPNeg64 %3",
                "%5 = FPAbs64 #1.0d",
                "%6 = FPAdd64 %4, %5 (flags 0x33)",
                "%7 = UnpackDouble2x32 %6",
                "%8 = CompositeExtractU32x2 %7, #0x0",
                "SetRegister R2, %8",
                "%9 = CompositeExtractU32x2 %7, #0x1",
                "SetRegister R3, %9",
            ],
        );
    }

    #[test]
    fn dadd_rejects_cc_and_odd_registers() {
        rejects(encode(Opcode::DaddReg, 2 | 4 << 8 | 6 << 20 | 1 << 47));
        rejects(encode(Opcode::DaddReg, 2 | 5 << 8 | 6 << 20));
    }
}
