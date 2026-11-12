//! FFMA and FFMA32I.

use super::floating_point_add::fp_rounding;
use super::{TResult, Translator};
use crate::ir::{Emitter, FmzMode, FpCompare, FpControl, FpRounding, Value};

/// Maxwell's 2-bit denormal field of FFMA and FMUL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DenormMode {
    Keep,
    Ftz,
    /// D3D9 multiply: FTZ plus `x * 0 == 0` for every x, emulated with a select.
    Fmz,
}

impl DenormMode {
    pub(super) const fn from_bits(bits: u64) -> Option<DenormMode> {
        match bits & 3 {
            0 => Some(DenormMode::Keep),
            1 => Some(DenormMode::Ftz),
            2 => Some(DenormMode::Fmz),
            _ => None,
        }
    }

    pub(super) const fn ir_mode(self) -> FmzMode {
        match self {
            DenormMode::Keep => FmzMode::None,
            DenormMode::Ftz | DenormMode::Fmz => FmzMode::Ftz,
        }
    }
}

/// Whether either factor compares (ordered) equal to zero, for the FMZ emulation.
pub(super) fn any_zero(ir: &mut Emitter<'_>, a: Value, b: Value, zero: Value) -> Value {
    let zero_a = ir.fp_compare(FpCompare::Equal, a, zero, true, FpControl::default());
    let zero_b = ir.fp_compare(FpCompare::Equal, b, zero, true, FpControl::default());
    ir.logical_or(zero_a, zero_b)
}

struct Ffma {
    neg_a: bool,
    neg_b: bool,
    neg_c: bool,
    sat: bool,
    cc: bool,
    fmz: Option<DenormMode>,
    rounding: FpRounding,
}

fn ffma(t: &mut Translator<'_>, src_b: Value, src_c: Value, m: Ffma) -> TResult {
    let (false, Some(fmz)) = (m.cc, m.fmz) else {
        return Err(t.unsupported());
    };
    let src_a = t.f(t.insn.src_a_reg());
    let op_a = t.ir.fp_abs_neg(src_a, false, m.neg_a);
    let op_b = t.ir.fp_abs_neg(src_b, false, m.neg_b);
    let op_c = t.ir.fp_abs_neg(src_c, false, m.neg_c);
    let control = FpControl {
        no_contraction: true,
        rounding: m.rounding,
        fmz_mode: fmz.ir_mode(),
    };
    let mut value = t.ir.fp_fma(op_a, op_b, op_c, control);
    if fmz == DenormMode::Fmz && !m.sat {
        let zero = any_zero(&mut t.ir, op_a, op_b, Value::F32(0.0));
        value = t.ir.select(zero, op_c, value);
    }
    if m.sat {
        value = t.ir.fp_saturate(value);
    }
    t.set_f(t.insn.dest_reg(), value);
    Ok(())
}

fn ffma_common(t: &mut Translator<'_>, src_b: Value, src_c: Value) -> TResult {
    let i = t.insn;
    let m = Ffma {
        neg_a: false,
        neg_b: i.bit(48),
        neg_c: i.bit(49),
        sat: i.bit(50),
        cc: i.bit(47),
        fmz: DenormMode::from_bits(i.bits(53, 2)),
        rounding: fp_rounding(i.bits(51, 2)),
    };
    ffma(t, src_b, src_c, m)
}

pub(super) fn ffma_cr(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_cbuf()?;
    let src_c = t.get_float_reg39();
    ffma_common(t, src_b, src_c)
}

pub(super) fn ffma_rc(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_reg39();
    let src_c = t.get_float_cbuf()?;
    ffma_common(t, src_b, src_c)
}

pub(super) fn ffma_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_reg20();
    let src_c = t.get_float_reg39();
    ffma_common(t, src_b, src_c)
}

pub(super) fn ffma_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_imm20();
    let src_c = t.get_float_reg39();
    ffma_common(t, src_b, src_c)
}

pub(super) fn ffma32i(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let m = Ffma {
        neg_a: i.bit(56),
        neg_b: false,
        neg_c: i.bit(57),
        sat: i.bit(55),
        cc: i.bit(52),
        fmz: DenormMode::from_bits(i.bits(53, 2)),
        rounding: FpRounding::Rn,
    };
    let src_b = t.get_float_imm32();
    let src_c = t.f(i.dest_reg());
    ffma(t, src_b, src_c, m)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn ffma_reg_fmz_emulation() {
        check(
            encode(
                Opcode::FfmaReg,
                2 | 4 << 8 | 6 << 20 | 8 << 39 | 2 << 53 | 1 << 48,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R8",
                "%3 = BitCastF32U32 %2",
                "%4 = GetRegister R4",
                "%5 = BitCastF32U32 %4",
                "%6 = FPNeg32 %1",
                "%7 = FPFma32 %5, %6, %3 (flags 0x13)",
                "%8 = FPOrdEqual32 %5, #0.0f",
                "%9 = FPOrdEqual32 %6, #0.0f",
                "%10 = LogicalOr %8, %9",
                "%11 = SelectF32 %10, %3, %7",
                "%12 = BitCastU32F32 %11",
                "SetRegister R2, %12",
            ],
        );
    }

    #[test]
    fn ffma_reg_fmz_sat_skips_emulation() {
        check(
            encode(
                Opcode::FfmaReg,
                2 | 4 << 8 | 6 << 20 | 8 << 39 | 2 << 53 | 1 << 50 | 1 << 51,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R8",
                "%3 = BitCastF32U32 %2",
                "%4 = GetRegister R4",
                "%5 = BitCastF32U32 %4",
                "%6 = FPFma32 %5, %1, %3 (flags 0x15)",
                "%7 = FPSaturate32 %6",
                "%8 = BitCastU32F32 %7",
                "SetRegister R2, %8",
            ],
        );
    }

    #[test]
    fn ffma_rc() {
        check(
            encode(
                Opcode::FfmaRc,
                2 | 4 << 8 | 1 << 34 | 3 << 20 | 8 << 39 | 1 << 49,
            ),
            &[
                "%0 = GetRegister R8",
                "%1 = BitCastF32U32 %0",
                "%2 = GetCbufF32 #0x1, #0xc",
                "%3 = GetRegister R4",
                "%4 = BitCastF32U32 %3",
                "%5 = FPNeg32 %2",
                "%6 = FPFma32 %4, %1, %5 (flags 0x33)",
                "%7 = BitCastU32F32 %6",
                "SetRegister R2, %7",
            ],
        );
    }

    #[test]
    fn ffma_cr() {
        check(
            encode(Opcode::FfmaCr, 2 | 4 << 8 | 1 << 34 | 3 << 20 | 8 << 39),
            &[
                "%0 = GetCbufF32 #0x1, #0xc",
                "%1 = GetRegister R8",
                "%2 = BitCastF32U32 %1",
                "%3 = GetRegister R4",
                "%4 = BitCastF32U32 %3",
                "%5 = FPFma32 %4, %0, %2 (flags 0x33)",
                "%6 = BitCastU32F32 %5",
                "SetRegister R2, %6",
            ],
        );
    }

    #[test]
    fn ffma_imm() {
        check(
            encode(
                Opcode::FfmaImm,
                2 | 4 << 8 | 0x3f800 << 20 | 8 << 39 | 1 << 53,
            ),
            &[
                "%0 = GetRegister R8",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R4",
                "%3 = BitCastF32U32 %2",
                "%4 = FPFma32 %3, #1.0f, %1 (flags 0x13)",
                "%5 = BitCastU32F32 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn ffma32i_reads_dest_as_addend() {
        check(
            encode(
                Opcode::Ffma32i,
                2 | 4 << 8 | 0x4000_0000 << 20 | 1 << 56 | 1 << 57,
            ),
            &[
                "%0 = GetRegister R2",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R4",
                "%3 = BitCastF32U32 %2",
                "%4 = FPNeg32 %3",
                "%5 = FPNeg32 %1",
                "%6 = FPFma32 %4, #2.0f, %5 (flags 0x33)",
                "%7 = BitCastU32F32 %6",
                "SetRegister R2, %7",
            ],
        );
    }

    #[test]
    fn ffma_rejects_cc_and_reserved_fmz() {
        rejects(encode(
            Opcode::FfmaReg,
            2 | 4 << 8 | 6 << 20 | 8 << 39 | 1 << 47,
        ));
        rejects(encode(
            Opcode::FfmaReg,
            2 | 4 << 8 | 6 << 20 | 8 << 39 | 3 << 53,
        ));
        rejects(encode(Opcode::Ffma32i, 2 | 4 << 8 | 1 << 52));
    }
}
