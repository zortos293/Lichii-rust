//! FMUL and FMUL32I.

use super::floating_point_add::fp_rounding;
use super::floating_point_fused_multiply_add::{DenormMode, any_zero};
use super::{TResult, Translator};
use crate::ir::{FpControl, FpRounding, Value};

struct Fmul {
    fmz: Option<DenormMode>,
    rounding: FpRounding,
    scale: u64,
    sat: bool,
    cc: bool,
    neg_b: bool,
}

fn fmul(t: &mut Translator<'_>, src_b: Value, m: Fmul) -> TResult {
    let (false, Some(fmz)) = (m.cc, m.fmz) else {
        return Err(t.unsupported());
    };
    let factor = match m.scale {
        0 => None,
        1 => Some(0.5),
        2 => Some(0.25),
        3 => Some(0.125),
        4 => Some(8.0),
        5 => Some(4.0),
        6 => Some(2.0),
        _ => return Err(t.unsupported()),
    };
    let mut op_a = t.f(t.insn.src_a_reg());
    if let Some(factor) = factor {
        if fmz != DenormMode::Ftz || m.rounding != FpRounding::Rn {
            return Err(t.unsupported());
        }
        op_a = t.ir.fp_mul(op_a, Value::F32(factor), FpControl::default());
    }
    let op_b = t.ir.fp_abs_neg(src_b, false, m.neg_b);
    let control = FpControl {
        no_contraction: true,
        rounding: m.rounding,
        fmz_mode: fmz.ir_mode(),
    };
    let mut value = t.ir.fp_mul(op_a, op_b, control);
    if fmz == DenormMode::Fmz && !m.sat {
        let zero = Value::F32(0.0);
        let is_zero = any_zero(&mut t.ir, op_a, op_b, zero);
        value = t.ir.select(is_zero, zero, value);
    }
    if m.sat {
        value = t.ir.fp_saturate(value);
    }
    t.set_f(t.insn.dest_reg(), value);
    Ok(())
}

fn fmul_common(t: &mut Translator<'_>, src_b: Value) -> TResult {
    let i = t.insn;
    let m = Fmul {
        fmz: DenormMode::from_bits(i.bits(44, 2)),
        rounding: fp_rounding(i.bits(39, 2)),
        scale: i.bits(41, 3),
        sat: i.bit(50),
        cc: i.bit(47),
        neg_b: i.bit(48),
    };
    fmul(t, src_b, m)
}

pub(super) fn fmul_cbuf(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_cbuf()?;
    fmul_common(t, src_b)
}

pub(super) fn fmul_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_reg20();
    fmul_common(t, src_b)
}

pub(super) fn fmul_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_imm20();
    fmul_common(t, src_b)
}

pub(super) fn fmul32i(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let m = Fmul {
        fmz: DenormMode::from_bits(i.bits(53, 2)),
        rounding: FpRounding::Rn,
        scale: 0,
        sat: i.bit(55),
        cc: i.bit(52),
        neg_b: false,
    };
    let src_b = t.get_float_imm32();
    fmul(t, src_b, m)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn fmul_reg_fmz_emulation() {
        check(
            encode(Opcode::FmulReg, 2 | 4 << 8 | 6 << 20 | 2 << 44 | 1 << 48),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R4",
                "%3 = BitCastF32U32 %2",
                "%4 = FPNeg32 %1",
                "%5 = FPMul32 %3, %4 (flags 0x13)",
                "%6 = FPOrdEqual32 %3, #0.0f",
                "%7 = FPOrdEqual32 %4, #0.0f",
                "%8 = LogicalOr %6, %7",
                "%9 = SelectF32 %8, #0.0f, %5",
                "%10 = BitCastU32F32 %9",
                "SetRegister R2, %10",
            ],
        );
    }

    #[test]
    fn fmul_reg_scale_d2() {
        check(
            encode(
                Opcode::FmulReg,
                2 | 4 << 8 | 6 << 20 | 1 << 41 | 1 << 44 | 1 << 50,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R4",
                "%3 = BitCastF32U32 %2",
                "%4 = FPMul32 %3, #0.5f",
                "%5 = FPMul32 %4, %1 (flags 0x13)",
                "%6 = FPSaturate32 %5",
                "%7 = BitCastU32F32 %6",
                "SetRegister R2, %7",
            ],
        );
    }

    #[test]
    fn fmul_cbuf() {
        check(
            encode(Opcode::FmulCbuf, 2 | 4 << 8 | 1 << 34 | 3 << 20 | 3 << 39),
            &[
                "%0 = GetCbufF32 #0x1, #0xc",
                "%1 = GetRegister R4",
                "%2 = BitCastF32U32 %1",
                "%3 = FPMul32 %2, %0 (flags 0x39)",
                "%4 = BitCastU32F32 %3",
                "SetRegister R2, %4",
            ],
        );
    }

    #[test]
    fn fmul_imm() {
        check(
            encode(Opcode::FmulImm, 2 | 4 << 8 | 0x3f800 << 20),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPMul32 %1, #1.0f (flags 0x33)",
                "%3 = BitCastU32F32 %2",
                "SetRegister R2, %3",
            ],
        );
    }

    #[test]
    fn fmul32i_fmz() {
        check(
            encode(Opcode::Fmul32i, 2 | 4 << 8 | 0x4000_0000 << 20 | 2 << 53),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPMul32 %1, #2.0f (flags 0x13)",
                "%3 = FPOrdEqual32 %1, #0.0f",
                "%4 = FPOrdEqual32 #2.0f, #0.0f",
                "%5 = LogicalOr %3, %4",
                "%6 = SelectF32 %5, #0.0f, %2",
                "%7 = BitCastU32F32 %6",
                "SetRegister R2, %7",
            ],
        );
    }

    #[test]
    fn fmul_rejects_invalid_modifiers() {
        rejects(encode(Opcode::FmulReg, 2 | 4 << 8 | 6 << 20 | 1 << 47));
        rejects(encode(Opcode::FmulReg, 2 | 4 << 8 | 6 << 20 | 3 << 44));
        rejects(encode(
            Opcode::FmulReg,
            2 | 4 << 8 | 6 << 20 | 7 << 41 | 1 << 44,
        ));
        rejects(encode(Opcode::FmulReg, 2 | 4 << 8 | 6 << 20 | 1 << 41));
        rejects(encode(
            Opcode::FmulReg,
            2 | 4 << 8 | 6 << 20 | 1 << 41 | 1 << 44 | 1 << 39,
        ));
        rejects(encode(Opcode::Fmul32i, 2 | 4 << 8 | 1 << 52));
    }
}
