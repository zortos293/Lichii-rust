//! F2I: float to integer conversion with Maxwell's saturation and NaN results.

use super::{TResult, Translator};
use crate::ir::{Emitter, FmzMode, FpCompare, FpControl, FpRounding, Value};

/// Destination of an F2I conversion; the field is the log2 of its size in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DestFormat {
    I8,
    I16,
    I32,
    I64,
}

impl DestFormat {
    pub(crate) const fn from_bits(bits: u64) -> DestFormat {
        match bits & 3 {
            0 => DestFormat::I8,
            1 => DestFormat::I16,
            2 => DestFormat::I32,
            _ => DestFormat::I64,
        }
    }

    const fn bits(self) -> u32 {
        match self {
            DestFormat::I8 => 8,
            DestFormat::I16 => 16,
            DestFormat::I32 => 32,
            DestFormat::I64 => 64,
        }
    }

    /// Largest and smallest representable value.
    const fn clamp_bounds(self, signed: bool) -> (f64, f64) {
        match (self, signed) {
            (DestFormat::I8, true) => (i8::MAX as f64, i8::MIN as f64),
            (DestFormat::I16, true) => (i16::MAX as f64, i16::MIN as f64),
            (DestFormat::I32, true) => (i32::MAX as f64, i32::MIN as f64),
            (DestFormat::I64, true) => (i64::MAX as f64, i64::MIN as f64),
            (DestFormat::I8, false) => (u8::MAX as f64, 0.0),
            (DestFormat::I16, false) => (u16::MAX as f64, 0.0),
            (DestFormat::I32, false) => (u32::MAX as f64, 0.0),
            (DestFormat::I64, false) => (u64::MAX as f64, 0.0),
        }
    }
}

/// Source of an F2I conversion; the field is the log2 of its size in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SrcFormat {
    F16,
    F32,
    F64,
}

impl SrcFormat {
    pub(crate) const fn from_bits(bits: u64) -> Option<SrcFormat> {
        match bits & 3 {
            1 => Some(SrcFormat::F16),
            2 => Some(SrcFormat::F32),
            3 => Some(SrcFormat::F64),
            _ => None,
        }
    }
}

fn src_format(t: &Translator<'_>) -> Result<SrcFormat, crate::TranslateError> {
    SrcFormat::from_bits(t.insn.bits(10, 2)).ok_or_else(|| t.unsupported())
}

/// Largest F32 below 2^64 (0x1.fffffep63).
const MAX_F32_BELOW_2_POW_64: f32 = f32::from_bits(0x5f7f_ffff);

/// An F2I conversion: formats, signedness, the 2-bit rounding field (nearest even, floor, ceil,
/// trunc) and FTZ.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FloatToInt {
    pub src: SrcFormat,
    pub dest: DestFormat,
    pub signed: bool,
    pub rounding: u64,
    pub ftz: bool,
}

fn f2i(t: &mut Translator<'_>, src: SrcFormat, src_a: Value) -> TResult {
    let i = t.insn;
    if i.bit(47) {
        return Err(t.unsupported());
    }
    let conversion = FloatToInt {
        src,
        dest: DestFormat::from_bits(i.bits(8, 2)),
        signed: i.bit(12),
        rounding: i.bits(39, 2),
        ftz: i.bit(44),
    };
    let op_a = t.ir.fp_abs_neg(src_a, i.bit(45), i.bit(49));
    let result = float_to_int(&mut t.ir, conversion, op_a);
    if conversion.dest == DestFormat::I64 {
        t.set_l(i.dest_reg(), result)
    } else {
        t.set_x(i.dest_reg(), result);
        Ok(())
    }
}

/// Converts `op_a` (modifiers applied) with Maxwell's clamping and NaN results. A 64-bit
/// destination gives a U64, every other one a U32.
pub(crate) fn float_to_int(ir: &mut Emitter<'_>, c: FloatToInt, op_a: Value) -> Value {
    let (src, dest) = (c.src, c.dest);
    let denorm_cares = src == SrcFormat::F32 && dest != DestFormat::I64;
    let fmz_mode = match (denorm_cares, c.ftz) {
        (false, _) => FmzMode::DontCare,
        (true, true) => FmzMode::Ftz,
        (true, false) => FmzMode::None,
    };
    let control = FpControl {
        no_contraction: true,
        rounding: FpRounding::DontCare,
        fmz_mode,
    };
    let rounded = match c.rounding {
        0 => ir.fp_round_even(op_a, control),
        1 => ir.fp_floor(op_a, control),
        2 => ir.fp_ceil(op_a, control),
        _ => ir.fp_trunc(op_a, control),
    };
    let signed = c.signed;
    let (max_bound, min_bound) = dest.clamp_bounds(signed);
    let unsigned_f32_i64 = dest == DestFormat::I64 && src == SrcFormat::F32 && !signed;

    let intermediate = match src {
        SrcFormat::F16 => {
            let max_val = ir.fp_convert(16, Value::F32(max_bound as f32), FpControl::default());
            let min_val = ir.fp_convert(16, Value::F32(min_bound as f32), FpControl::default());
            let mut safe = rounded;
            if dest == DestFormat::I8 {
                let widened = ir.fp_convert(32, safe, FpControl::default());
                let nan = ir.fp_is_nan(op_a);
                let selected = ir.select(nan, Value::F32(min_bound as f32), widened);
                safe = ir.fp_convert(16, selected, FpControl::default());
            }
            ir.fp_clamp(safe, min_val, max_val)
        }
        SrcFormat::F32 => {
            let max_val = Value::F32(if unsigned_f32_i64 {
                MAX_F32_BELOW_2_POW_64
            } else {
                max_bound as f32
            });
            let min_val = Value::F32(min_bound as f32);
            let mut safe = rounded;
            if unsigned_f32_i64 || dest == DestFormat::I8 {
                let nan = ir.fp_is_nan(op_a);
                safe = ir.select(nan, min_val, safe);
            }
            ir.fp_clamp(safe, min_val, max_val)
        }
        SrcFormat::F64 if dest == DestFormat::I8 => {
            let max_val = Value::F32(max_bound as f32);
            let min_val = Value::F32(min_bound as f32);
            let narrowed = ir.fp_convert(32, rounded, FpControl::default());
            let nan = ir.fp_is_nan(op_a);
            let safe = ir.select(nan, min_val, narrowed);
            ir.fp_clamp(safe, min_val, max_val)
        }
        SrcFormat::F64 => ir.fp_clamp(rounded, Value::F64(min_bound), Value::F64(max_bound)),
    };

    let bitsize = dest.bits().max(32);
    let mut result = ir.convert_f_to_i(bitsize, signed, intermediate);
    if unsigned_f32_i64 {
        let ge = ir.fp_compare(
            FpCompare::GreaterThanEqual,
            op_a,
            Value::F32(18_446_744_073_709_551_616.0),
            true,
            FpControl::default(),
        );
        result = ir.select(ge, Value::U64(u64::MAX), result);
    }

    let special_nan = (src == SrcFormat::F64) != (dest == DestFormat::I64);
    let nan_value = match dest {
        DestFormat::I32 if special_nan => Some(Value::U32(0x8000_0000)),
        DestFormat::I64 if special_nan => Some(Value::U64(0x8000_0000_0000_0000)),
        _ if signed && bitsize == 64 => Some(Value::U64(0)),
        _ if signed => Some(Value::U32(0)),
        _ => None,
    };
    if let Some(nan_value) = nan_value {
        let nan = ir.fp_is_nan(op_a);
        result = ir.select(nan, nan_value, result);
    }

    result
}

/// The F64 operand of F2I (cbuf): only the high word is read, the low word is zero.
fn double_cbuf_high(t: &mut Translator<'_>) -> Result<Value, crate::TranslateError> {
    let binding = t.insn.bits(34, 5) as u32;
    let offset = t.insn.signed_bits(20, 14);
    if binding >= 18 || offset < 0 || offset % 2 != 0 {
        return Err(t.unsupported());
    }
    let byte_offset = Value::U32(offset as u32 * 4 + 4);
    let data = t.ir.get_cbuf(Value::U32(binding), byte_offset);
    let vector = t.ir.composite_construct(&[Value::U32(0), data]);
    Ok(t.ir.pack_double_2x32(vector))
}

pub(super) fn f2i_cbuf(t: &mut Translator<'_>) -> TResult {
    let src = src_format(t)?;
    let op_a = match src {
        SrcFormat::F16 => {
            let packed = t.get_cbuf()?;
            let vector = t.ir.unpack_float_2x16(packed);
            t.ir.composite_extract(vector, u32::from(t.insn.bit(41)))
        }
        SrcFormat::F32 => t.get_float_cbuf()?,
        SrcFormat::F64 => double_cbuf_high(t)?,
    };
    f2i(t, src, op_a)
}

pub(super) fn f2i_reg(t: &mut Translator<'_>) -> TResult {
    let src = src_format(t)?;
    let reg = t.insn.src_b_reg();
    let op_a = match src {
        SrcFormat::F16 => {
            let packed = t.x(reg);
            let vector = t.ir.unpack_float_2x16(packed);
            t.ir.composite_extract(vector, u32::from(t.insn.bit(41)))
        }
        SrcFormat::F32 => t.f(reg),
        SrcFormat::F64 => {
            let lo = t.x(reg);
            let hi = t.x(reg.offset(1));
            let vector = t.ir.composite_construct(&[lo, hi]);
            t.ir.pack_double_2x32(vector)
        }
    };
    f2i(t, src, op_a)
}

pub(super) fn f2i_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn f2i_reg_f32_to_s32_trunc() {
        check(
            encode(
                Opcode::F2iReg,
                2 | 6 << 20 | 2 << 8 | 2 << 10 | 1 << 12 | 3 << 39 | 1 << 44,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = FPTrunc32 %1 (flags 0x11)",
                "%3 = FPClamp32 %2, #-2147483600.0f, #2147483600.0f",
                "%4 = ConvertS32F32 %3",
                "%5 = FPIsNan32 %1",
                "%6 = SelectU32 %5, #0x0, %4",
                "SetRegister R2, %6",
            ],
        );
    }

    #[test]
    fn f2i_reg_f32_to_u64() {
        check(
            encode(Opcode::F2iReg, 2 | 6 << 20 | 3 << 8 | 2 << 10),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = FPRoundEven32 %1 (flags 0x1)",
                "%3 = FPIsNan32 %1",
                "%4 = SelectF32 %3, #0.0f, %2",
                "%5 = FPClamp32 %4, #0.0f, #1.8446743e19f",
                "%6 = ConvertU64F32 %5",
                "%7 = FPOrdGreaterThanEqual32 %1, #1.8446744e19f",
                "%8 = SelectU64 %7, #0xffffffffffffffffu64, %6",
                "%9 = FPIsNan32 %1",
                "%10 = SelectU64 %9, #0x8000000000000000u64, %8",
                "%11 = UnpackUint2x32 %10",
                "%12 = CompositeExtractU32x2 %11, #0x0",
                "SetRegister R2, %12",
                "%13 = CompositeExtractU32x2 %11, #0x1",
                "SetRegister R3, %13",
            ],
        );
    }

    #[test]
    fn f2i_reg_f16_to_s8_floor() {
        check(
            encode(
                Opcode::F2iReg,
                2 | 6 << 20 | 1 << 10 | 1 << 12 | 1 << 39 | 1 << 41 | 1 << 45,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x1",
                "%3 = FPAbs16 %2",
                "%4 = FPFloor16 %3 (flags 0x1)",
                "%5 = ConvertF16F32 #127.0f",
                "%6 = ConvertF16F32 #-128.0f",
                "%7 = ConvertF32F16 %4",
                "%8 = FPIsNan16 %3",
                "%9 = SelectF32 %8, #-128.0f, %7",
                "%10 = ConvertF16F32 %9",
                "%11 = FPClamp16 %10, %6, %5",
                "%12 = ConvertS32F16 %11",
                "%13 = FPIsNan16 %3",
                "%14 = SelectU32 %13, #0x0, %12",
                "SetRegister R2, %14",
            ],
        );
    }

    #[test]
    fn f2i_cbuf_f64_to_u32_ceil() {
        check(
            encode(
                Opcode::F2iCbuf,
                2 | 1 << 34 | 4 << 20 | 2 << 8 | 3 << 10 | 2 << 39 | 1 << 49,
            ),
            &[
                "%0 = GetCbufU32 #0x1, #0x14",
                "%1 = CompositeConstructU32x2 #0x0, %0",
                "%2 = PackDouble2x32 %1",
                "%3 = FPNeg64 %2",
                "%4 = FPCeil64 %3 (flags 0x1)",
                "%5 = FPClamp64 %4, #0.0d, #4294967295.0d",
                "%6 = ConvertU32F64 %5",
                "%7 = FPIsNan64 %3",
                "%8 = SelectU32 %7, #0x80000000, %6",
                "SetRegister R2, %8",
            ],
        );
    }

    #[test]
    fn f2i_rejects_invalid_encodings() {
        rejects(encode(Opcode::F2iImm, 2 | 4 << 8));
        rejects(encode(Opcode::F2iReg, 2 | 6 << 20 | 2 << 8));
        rejects(encode(
            Opcode::F2iReg,
            2 | 6 << 20 | 2 << 8 | 2 << 10 | 1 << 47,
        ));
        rejects(encode(Opcode::F2iReg, 3 | 6 << 20 | 3 << 8 | 2 << 10));
        rejects(encode(
            Opcode::F2iCbuf,
            2 | 1 << 34 | 3 << 20 | 2 << 8 | 3 << 10,
        ));
        rejects(encode(
            Opcode::F2iCbuf,
            2 | 1 << 34 | 0x2000 << 20 | 2 << 8 | 3 << 10,
        ));
    }
}
