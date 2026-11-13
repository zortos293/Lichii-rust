//! F2F: float to float conversion and rounding.

use super::floating_point_add::fp_rounding;
use super::half_floating_point_add::{Swizzle, extract};
use super::{TResult, Translator};
use crate::ir::{FmzMode, FpControl, FpRounding, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloatFormat {
    F16,
    F32,
    F64,
}

impl FloatFormat {
    const fn from_bits(bits: u64) -> Option<FloatFormat> {
        match bits & 3 {
            1 => Some(FloatFormat::F16),
            2 => Some(FloatFormat::F32),
            3 => Some(FloatFormat::F64),
            _ => None,
        }
    }

    const fn width(self) -> u32 {
        match self {
            FloatFormat::F16 => 16,
            FloatFormat::F32 => 32,
            FloatFormat::F64 => 64,
        }
    }
}

fn src_format(t: &Translator<'_>) -> Result<FloatFormat, crate::TranslateError> {
    FloatFormat::from_bits(t.insn.bits(10, 2)).ok_or_else(|| t.unsupported())
}

fn f16_zero(t: &mut Translator<'_>) -> Value {
    t.ir.fp_convert(16, Value::F32(0.0), FpControl::default())
}

fn f2f(t: &mut Translator<'_>, src: Value) -> TResult {
    let i = t.insn;
    let (Some(src_size), Some(dst_size)) = (
        FloatFormat::from_bits(i.bits(10, 2)),
        FloatFormat::from_bits(i.bits(8, 2)),
    ) else {
        return Err(t.unsupported());
    };
    let f16_f64 = matches!(
        (src_size, dst_size),
        (FloatFormat::F16, FloatFormat::F64) | (FloatFormat::F64, FloatFormat::F16)
    );
    if i.bit(47) || f16_f64 {
        return Err(t.unsupported());
    }
    let mut input = t.ir.fp_abs_neg(src, i.bit(49), i.bit(45));
    let any_fp64 = src_size == FloatFormat::F64 || dst_size == FloatFormat::F64;
    let mut control = FpControl {
        no_contraction: false,
        rounding: FpRounding::DontCare,
        fmz_mode: if i.bit(44) && !any_fp64 {
            FmzMode::Ftz
        } else {
            FmzMode::None
        },
    };
    if src_size != dst_size {
        control.rounding = fp_rounding(i.bits(39, 2));
        input = t.ir.fp_convert(dst_size.width(), input, control);
    } else {
        input = match i.bits(39, 4) & 0xb {
            0 | 3 => {
                let zero = match src_size {
                    FloatFormat::F16 => f16_zero(t),
                    FloatFormat::F32 => Value::F32(0.0),
                    FloatFormat::F64 => Value::F64(0.0),
                };
                t.ir.fp_add(input, zero, control)
            }
            8 => t.ir.fp_round_even(input, control),
            9 => t.ir.fp_floor(input, control),
            10 => t.ir.fp_ceil(input, control),
            11 => t.ir.fp_trunc(input, control),
            _ => return Err(t.unsupported()),
        };
    }
    if i.bit(50) && !any_fp64 {
        input = t.ir.fp_saturate(input);
    }
    match dst_size {
        FloatFormat::F16 => {
            let zero = f16_zero(t);
            let vector = t.ir.composite_construct(&[input, zero]);
            let packed = t.ir.pack_float_2x16(vector);
            t.set_x(i.dest_reg(), packed);
            Ok(())
        }
        FloatFormat::F32 => {
            t.set_f(i.dest_reg(), input);
            Ok(())
        }
        FloatFormat::F64 => t.set_d(i.dest_reg(), input),
    }
}

fn select_half(t: &mut Translator<'_>, packed: Value) -> Value {
    let (lhs, rhs) = extract(&mut t.ir, packed, Swizzle::H1H0);
    if t.insn.bit(41) { rhs } else { lhs }
}

pub(super) fn f2f_cbuf(t: &mut Translator<'_>) -> TResult {
    let src = match src_format(t)? {
        FloatFormat::F16 => {
            let packed = t.get_cbuf()?;
            select_half(t, packed)
        }
        FloatFormat::F32 => t.get_float_cbuf()?,
        FloatFormat::F64 => t.get_double_cbuf()?,
    };
    f2f(t, src)
}

pub(super) fn f2f_reg(t: &mut Translator<'_>) -> TResult {
    let src = match src_format(t)? {
        FloatFormat::F16 => {
            let packed = t.get_reg20();
            select_half(t, packed)
        }
        FloatFormat::F32 => t.get_float_reg20(),
        FloatFormat::F64 => t.get_double_reg20()?,
    };
    f2f(t, src)
}

pub(super) fn f2f_imm(t: &mut Translator<'_>) -> TResult {
    let src = match src_format(t)? {
        FloatFormat::F16 => {
            if t.insn.bit(56) {
                return Err(t.unsupported());
            }
            let imm = (t.insn.bits(20, 19) & 0xffff) as u32;
            let vector = t.ir.unpack_float_2x16(Value::U32(imm | (imm << 16)));
            t.ir.composite_extract(vector, u32::from(!t.insn.bit(41)))
        }
        FloatFormat::F32 => t.get_float_imm20(),
        FloatFormat::F64 => t.get_double_imm20(),
    };
    f2f(t, src)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn f2f_reg_f32_to_f16_rounds() {
        check(
            encode(Opcode::F2fReg, 2 | 6 << 20 | 1 << 8 | 2 << 10 | 3 << 39),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = ConvertF16F32 %1 (flags 0x38)",
                "%3 = ConvertF16F32 #0.0f",
                "%4 = CompositeConstructF16x2 %2, %3",
                "%5 = PackFloat2x16 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn f2f_reg_f32_floor_ftz_sat() {
        check(
            encode(
                Opcode::F2fReg,
                2 | 6 << 20 | 2 << 8 | 2 << 10 | 9 << 39 | 1 << 44 | 1 << 50,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = FPFloor32 %1 (flags 0x10)",
                "%3 = FPSaturate32 %2",
                "%4 = BitCastU32F32 %3",
                "SetRegister R2, %4",
            ],
        );
    }

    #[test]
    fn f2f_reg_f16_high_half_pass() {
        check(
            encode(
                Opcode::F2fReg,
                2 | 6 << 20 | 1 << 8 | 1 << 10 | 1 << 41 | 1 << 45,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = FPNeg16 %3",
                "%5 = ConvertF16F32 #0.0f",
                "%6 = FPAdd16 %4, %5 (flags 0x30)",
                "%7 = ConvertF16F32 #0.0f",
                "%8 = CompositeConstructF16x2 %6, %7",
                "%9 = PackFloat2x16 %8",
                "SetRegister R2, %9",
            ],
        );
    }

    #[test]
    fn f2f_cbuf_f64_to_f32() {
        check(
            encode(
                Opcode::F2fCbuf,
                2 | 1 << 34 | 4 << 20 | 2 << 8 | 3 << 10 | 1 << 44 | 1 << 50,
            ),
            &[
                "%0 = GetCbufU32 #0x1, #0x14",
                "%1 = GetCbufU32 #0x1, #0x10",
                "%2 = CompositeConstructU32x2 %1, %0",
                "%3 = PackDouble2x32 %2",
                "%4 = ConvertF32F64 %3 (flags 0x32)",
                "%5 = BitCastU32F32 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn f2f_imm_f16_selects_low_half() {
        check(
            encode(Opcode::F2fImm, 2 | 0x3c00 << 20 | 1 << 8 | 1 << 10),
            &[
                "%0 = UnpackFloat2x16 #0x3c003c00",
                "%1 = CompositeExtractF16x2 %0, #0x1",
                "%2 = ConvertF16F32 #0.0f",
                "%3 = FPAdd16 %1, %2 (flags 0x30)",
                "%4 = ConvertF16F32 #0.0f",
                "%5 = CompositeConstructF16x2 %3, %4",
                "%6 = PackFloat2x16 %5",
                "SetRegister R2, %6",
            ],
        );
    }

    #[test]
    fn f2f_imm_f32_trunc() {
        check(
            encode(
                Opcode::F2fImm,
                2 | 0x3f800 << 20 | 2 << 8 | 2 << 10 | 11 << 39,
            ),
            &[
                "%0 = FPTrunc32 #1.0f (flags 0x30)",
                "%1 = BitCastU32F32 %0",
                "SetRegister R2, %1",
            ],
        );
    }

    #[test]
    fn f2f_rejects_invalid_encodings() {
        rejects(encode(Opcode::F2fReg, 2 | 6 << 20 | 2 << 8));
        rejects(encode(Opcode::F2fReg, 2 | 6 << 20 | 3 << 8 | 1 << 10));
        rejects(encode(
            Opcode::F2fReg,
            2 | 6 << 20 | 2 << 8 | 2 << 10 | 1 << 39,
        ));
        rejects(encode(
            Opcode::F2fReg,
            2 | 6 << 20 | 2 << 8 | 2 << 10 | 1 << 47,
        ));
        rejects(encode(Opcode::F2fImm, 2 | 1 << 8 | 1 << 10 | 1 << 56));
        rejects(encode(Opcode::F2fReg, 2 | 5 << 20 | 2 << 8 | 3 << 10));
    }
}
