//! I2F: integer to float conversion.

use super::floating_point_add::fp_rounding;
use super::floating_point_multi_function::has_native_instructions;
use super::{TResult, Translator};
use crate::ir::{Emitter, FmzMode, FpControl, FpRounding, ICompare, Type, Value};

/// Source of an I2F conversion; the field is the log2 of its size in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntFormat {
    U8,
    U16,
    U32,
    U64,
}

impl IntFormat {
    pub(crate) const fn from_bits(bits: u64) -> IntFormat {
        match bits & 3 {
            0 => IntFormat::U8,
            1 => IntFormat::U16,
            2 => IntFormat::U32,
            _ => IntFormat::U64,
        }
    }
}

fn int_format(t: &Translator<'_>) -> IntFormat {
    IntFormat::from_bits(t.insn.bits(10, 2))
}

/// Integer equality of two U32 or two U64 values; U64 compares both words.
fn iequal(ir: &mut Emitter<'_>, a: Value, b: Value) -> Value {
    if ir.ty(a) != Type::U64 {
        return ir.icompare(ICompare::Equal, a, b);
    }
    let (va, vb) = (ir.unpack_uint_2x32(a), ir.unpack_uint_2x32(b));
    let (lo_a, lo_b) = (ir.composite_extract(va, 0), ir.composite_extract(vb, 0));
    let lo = ir.icompare(ICompare::Equal, lo_a, lo_b);
    let (hi_a, hi_b) = (ir.composite_extract(va, 1), ir.composite_extract(vb, 1));
    let hi = ir.icompare(ICompare::Equal, hi_a, hi_b);
    ir.logical_and(lo, hi)
}

const fn least_value(bits: u32) -> u32 {
    (1u32 << (bits - 1)).wrapping_neg()
}

fn small_abs(ir: &mut Emitter<'_>, value: Value, bits: u32) -> Value {
    let mask = ir.shift_right_arithmetic(value, Value::U32(bits - 1));
    let sum = ir.iadd(value, mask);
    let absolute = ir.bitwise_xor(sum, mask);
    let is_least = ir.icompare(ICompare::Equal, value, Value::U32(least_value(bits)));
    ir.select(is_least, value, absolute)
}

fn word(ir: &mut Emitter<'_>, value: Value, index: u32) -> Value {
    let vector = ir.unpack_uint_2x32(value);
    ir.composite_extract(vector, index)
}

fn non_zero(ir: &mut Emitter<'_>, v: Value) -> Value {
    ir.icompare(ICompare::NotEqual, v, Value::U32(0))
}

/// Builds the F32 bits of a value whose most significant set bit is `msb`, from its top 24 bits.
fn float_bits(ir: &mut Emitter<'_>, msb: Value, top: Value) -> Value {
    let biased = ir.iadd(msb, Value::U32(127));
    let exponent = ir.shift_left_logical(biased, Value::U32(23));
    let mantissa = ir.bitwise_and(top, Value::U32(0x7f_ffff));
    ir.bitwise_or(exponent, mantissa)
}

/// Picks `bits` for values above 2^24 and the exact conversion of `low` below it.
fn select_exact(ir: &mut Emitter<'_>, msb: Value, bits: Value, low: Value) -> Value {
    let masked = ir.bitwise_and(low, Value::U32(0xff_ffff));
    let exact = ir.convert_i_to_f(32, 32, false, masked, FpControl::default());
    let large = ir.icompare(ICompare::UGreaterThan, msb, Value::U32(23));
    let bits = ir.bitcast_f32(bits);
    ir.select(large, bits, exact)
}

fn shift_below_24_bits(ir: &mut Emitter<'_>, msb: Value) -> Value {
    let wide = ir.imax(msb, Value::U32(23), false);
    ir.isub(wide, Value::U32(23))
}

fn convert_unsigned64_to_float(ir: &mut Emitter<'_>, source: Value, rounding: FpRounding) -> Value {
    let low = word(ir, source, 0);
    let high = word(ir, source, 1);
    let high_set = non_zero(ir, high);
    let high_msb = ir.find_u_msb(high);
    let high_msb = ir.iadd(high_msb, Value::U32(32));
    let low_set = non_zero(ir, low);
    let low_msb = ir.find_u_msb(low);
    let low_msb = ir.select(low_set, low_msb, Value::U32(0));
    let msb = ir.select(high_set, high_msb, low_msb);
    let shift = shift_below_24_bits(ir, msb);
    let retained = ir.shift_right_logical(source, shift);
    let top = word(ir, retained, 0);
    let mut bits = float_bits(ir, msb, top);
    let increment = match rounding {
        FpRounding::Rp => {
            let back = ir.shift_left_logical(retained, shift);
            let exact = iequal(ir, source, back);
            ir.logical_not(exact)
        }
        FpRounding::Rn => {
            let at_least_one = ir.imax(shift, Value::U32(1), false);
            let guard_shift = ir.isub(at_least_one, Value::U32(1));
            let guarded = ir.shift_right_logical(source, guard_shift);
            let guard_word = word(ir, guarded, 0);
            let guard_bit = ir.bitwise_and(guard_word, Value::U32(1));
            let guard = non_zero(ir, guard_bit);
            let back = ir.shift_left_logical(guarded, guard_shift);
            let exact = iequal(ir, source, back);
            let sticky = ir.logical_not(exact);
            let odd_bit = ir.bitwise_and(top, Value::U32(1));
            let odd = non_zero(ir, odd_bit);
            let round_up = ir.logical_or(sticky, odd);
            ir.logical_and(guard, round_up)
        }
        _ => Value::U1(false),
    };
    let one = ir.select(increment, Value::U32(1), Value::U32(0));
    bits = ir.iadd(bits, one);
    select_exact(ir, msb, bits, low)
}

fn convert_unsigned32_to_float_rz(ir: &mut Emitter<'_>, source: Value) -> Value {
    let set = non_zero(ir, source);
    let found = ir.find_u_msb(source);
    let msb = ir.select(set, found, Value::U32(0));
    let shift = shift_below_24_bits(ir, msb);
    let retained = ir.shift_right_logical(source, shift);
    let bits = float_bits(ir, msb, retained);
    select_exact(ir, msb, bits, source)
}

/// Signed I32 to F32 rounding towards +inf, fixed up from a round-to-nearest conversion.
fn convert_signed32_to_float_rp(ir: &mut Emitter<'_>, source: Value) -> Value {
    let converted = ir.convert_i_to_f(32, 32, true, source, FpControl::default());
    let abs = ir.fp_abs(converted);
    let rounded_magnitude = ir.convert_f_to_i(32, false, abs);
    let negative = ir.icompare(ICompare::SLessThan, source, Value::U32(0));
    let negated = ir.ineg(source);
    let magnitude = ir.select(negative, negated, source);
    let bits = ir.bitcast_u32(converted);
    let below = ir.icompare(ICompare::ULessThan, rounded_magnitude, magnitude);
    let up = ir.iadd(bits, Value::U32(1));
    let positive = ir.select(below, up, bits);
    let above = ir.icompare(ICompare::UGreaterThan, rounded_magnitude, magnitude);
    let down = ir.isub(bits, Value::U32(1));
    let negative_bits = ir.select(above, down, bits);
    let result = ir.select(negative, negative_bits, positive);
    ir.bitcast_f32(result)
}

/// An I2F conversion.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IntToFloat {
    pub format: IntFormat,
    /// Byte offset of an 8- or 16-bit source in its register.
    pub selector: u64,
    pub signed: bool,
    /// Integer absolute value of the source.
    pub abs: bool,
    /// Negation of the result.
    pub neg: bool,
    /// 16, 32 or 64.
    pub dst_bits: u32,
    pub rounding: FpRounding,
    /// Whether the program is native Ampere code, whose signed round-up conversion is exact.
    pub native: bool,
}

impl IntToFloat {
    /// Whether the selector names a byte the source format can start at.
    pub(crate) const fn valid_selector(&self) -> bool {
        match self.format {
            IntFormat::U8 => true,
            IntFormat::U16 => self.selector.is_multiple_of(2),
            IntFormat::U32 | IntFormat::U64 => self.selector == 0,
        }
    }
}

fn i2f(t: &mut Translator<'_>, src: Value) -> TResult {
    let i = t.insn;
    let dst_bits = match i.bits(8, 2) {
        1 => 16,
        2 => 32,
        3 => 64,
        _ => return Err(t.unsupported()),
    };
    let conversion = IntToFloat {
        format: int_format(t),
        selector: i.bits(41, 2),
        signed: i.bit(13),
        abs: i.bit(49),
        neg: i.bit(45),
        dst_bits,
        rounding: fp_rounding(i.bits(39, 2)),
        native: has_native_instructions(t),
    };
    if i.bit(47) || !conversion.valid_selector() {
        return Err(t.unsupported());
    }
    let value = int_to_float(&mut t.ir, conversion, src);
    match dst_bits {
        16 => {
            let zero = t.ir.fp_convert(16, Value::F32(0.0), FpControl::default());
            let vector = t.ir.composite_construct(&[value, zero]);
            let packed = t.ir.pack_float_2x16(vector);
            t.set_x(i.dest_reg(), packed);
            Ok(())
        }
        32 => {
            t.set_f(i.dest_reg(), value);
            Ok(())
        }
        _ => t.set_d(i.dest_reg(), value),
    }
}

/// Converts `src` (a U64 for 64-bit formats, else a U32) to a float of `c.dst_bits`. The
/// selector must be valid for the format.
pub(crate) fn int_to_float(ir: &mut Emitter<'_>, c: IntToFloat, mut src: Value) -> Value {
    let IntToFloat {
        format,
        selector,
        signed,
        abs,
        neg,
        dst_bits,
        rounding,
        native,
    } = c;
    let src_bits = match format {
        IntFormat::U8 | IntFormat::U16 => {
            let width = if format == IntFormat::U8 { 8 } else { 16 };
            let offset = Value::U32(selector as u32 * 8);
            src = ir.bit_field_extract(src, offset, Value::U32(width), signed);
            if abs {
                src = small_abs(ir, src, width);
            }
            width
        }
        IntFormat::U32 | IntFormat::U64 => {
            if abs && signed {
                src = ir.iabs(src);
            }
            if format == IntFormat::U64 { 64 } else { 32 }
        }
    };
    let conversion_bits = if format == IntFormat::U64 { 64 } else { 32 };
    let control = FpControl {
        no_contraction: false,
        rounding,
        fmz_mode: FmzMode::DontCare,
    };
    let mut value = if !signed && conversion_bits == 64 && dst_bits == 32 {
        convert_unsigned64_to_float(ir, src, rounding)
    } else if !signed && conversion_bits == 32 && dst_bits == 32 && rounding == FpRounding::Rz {
        convert_unsigned32_to_float_rz(ir, src)
    } else if native
        && signed
        && format == IntFormat::U32
        && dst_bits == 32
        && rounding == FpRounding::Rp
        && !abs
        && !neg
    {
        convert_signed32_to_float_rp(ir, src)
    } else {
        ir.convert_i_to_f(dst_bits, conversion_bits, signed, src, control)
    };
    if neg {
        if abs || !signed {
            value = ir.fp_neg(value);
        } else {
            let least = match src_bits {
                64 => Value::U64(i64::MIN as u64),
                bits => Value::U32(least_value(bits)),
            };
            let is_least = iequal(ir, src, least);
            let negated = ir.fp_neg(value);
            value = ir.select(is_least, value, negated);
        }
    }
    value
}

pub(super) fn i2f_cbuf(t: &mut Translator<'_>) -> TResult {
    let src = if int_format(t) == IntFormat::U64 {
        t.get_packed_cbuf()?
    } else {
        t.get_cbuf()?
    };
    i2f(t, src)
}

pub(super) fn i2f_reg(t: &mut Translator<'_>) -> TResult {
    let src = if int_format(t) == IntFormat::U64 {
        let reg = t.insn.src_b_reg();
        let lo = t.ir.get_reg(reg);
        let hi = t.ir.get_reg(reg.offset(1));
        let regs = t.ir.composite_construct(&[lo, hi]);
        t.ir.pack_uint_2x32(regs)
    } else {
        t.get_reg20()
    };
    i2f(t, src)
}

pub(super) fn i2f_imm(t: &mut Translator<'_>) -> TResult {
    let src = if int_format(t) == IntFormat::U64 {
        t.get_packed_imm20()
    } else {
        t.get_imm20()
    };
    i2f(t, src)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, check_native, rejects};
    use super::super::tests::encode;
    use super::super::tests::translate;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn i2f_reg_s32_to_f32() {
        check(
            encode(Opcode::I2fReg, 2 | 6 << 20 | 2 << 8 | 2 << 10 | 1 << 13),
            &[
                "%0 = GetRegister R6",
                "%1 = ConvertF32S32 %0 (flags 0x2)",
                "%2 = BitCastU32F32 %1",
                "SetRegister R2, %2",
            ],
        );
    }

    #[test]
    fn i2f_reg_s32_negated_keeps_least_value() {
        check(
            encode(
                Opcode::I2fReg,
                2 | 6 << 20 | 2 << 8 | 2 << 10 | 1 << 13 | 1 << 45,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = ConvertF32S32 %0 (flags 0x2)",
                "%2 = IEqual %0, #0x80000000",
                "%3 = FPNeg32 %1",
                "%4 = SelectF32 %2, %1, %3",
                "%5 = BitCastU32F32 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn i2f_reg_u32_rz() {
        check(
            encode(Opcode::I2fReg, 2 | 6 << 20 | 2 << 8 | 2 << 10 | 3 << 39),
            &[
                "%0 = GetRegister R6",
                "%1 = INotEqual %0, #0x0",
                "%2 = FindUMsb32 %0",
                "%3 = SelectU32 %1, %2, #0x0",
                "%4 = UMax32 %3, #0x17",
                "%5 = ISub32 %4, #0x17",
                "%6 = ShiftRightLogical32 %0, %5",
                "%7 = IAdd32 %3, #0x7f",
                "%8 = ShiftLeftLogical32 %7, #0x17",
                "%9 = BitwiseAnd32 %6, #0x7fffff",
                "%10 = BitwiseOr32 %8, %9",
                "%11 = BitwiseAnd32 %0, #0xffffff",
                "%12 = ConvertF32U32 %11",
                "%13 = UGreaterThan %3, #0x17",
                "%14 = BitCastF32U32 %10",
                "%15 = SelectF32 %13, %14, %12",
                "%16 = BitCastU32F32 %15",
                "SetRegister R2, %16",
            ],
        );
    }

    #[test]
    fn i2f_reg_s8_byte_abs() {
        check(
            encode(
                Opcode::I2fReg,
                2 | 6 << 20 | 2 << 8 | 1 << 13 | 2 << 41 | 1 << 49,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = BitFieldSExtract %0, #0x10, #0x8",
                "%2 = ShiftRightArithmetic32 %1, #0x7",
                "%3 = IAdd32 %1, %2",
                "%4 = BitwiseXor32 %3, %2",
                "%5 = IEqual %1, #0xffffff80",
                "%6 = SelectU32 %5, %1, %4",
                "%7 = ConvertF32S32 %6 (flags 0x2)",
                "%8 = BitCastU32F32 %7",
                "SetRegister R2, %8",
            ],
        );
    }

    #[test]
    fn i2f_reg_u64_rz() {
        check(
            encode(Opcode::I2fReg, 2 | 6 << 20 | 2 << 8 | 3 << 10 | 3 << 39),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R7",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackUint2x32 %2",
                "%4 = UnpackUint2x32 %3",
                "%5 = CompositeExtractU32x2 %4, #0x0",
                "%6 = UnpackUint2x32 %3",
                "%7 = CompositeExtractU32x2 %6, #0x1",
                "%8 = INotEqual %7, #0x0",
                "%9 = FindUMsb32 %7",
                "%10 = IAdd32 %9, #0x20",
                "%11 = INotEqual %5, #0x0",
                "%12 = FindUMsb32 %5",
                "%13 = SelectU32 %11, %12, #0x0",
                "%14 = SelectU32 %8, %10, %13",
                "%15 = UMax32 %14, #0x17",
                "%16 = ISub32 %15, #0x17",
                "%17 = ShiftRightLogical64 %3, %16",
                "%18 = UnpackUint2x32 %17",
                "%19 = CompositeExtractU32x2 %18, #0x0",
                "%20 = IAdd32 %14, #0x7f",
                "%21 = ShiftLeftLogical32 %20, #0x17",
                "%22 = BitwiseAnd32 %19, #0x7fffff",
                "%23 = BitwiseOr32 %21, %22",
                "%24 = SelectU32 #false, #0x1, #0x0",
                "%25 = IAdd32 %23, %24",
                "%26 = BitwiseAnd32 %5, #0xffffff",
                "%27 = ConvertF32U32 %26",
                "%28 = UGreaterThan %14, #0x17",
                "%29 = BitCastF32U32 %25",
                "%30 = SelectF32 %28, %29, %27",
                "%31 = BitCastU32F32 %30",
                "SetRegister R2, %31",
            ],
        );
    }

    #[test]
    fn i2f_cbuf_u16_to_f16() {
        check(
            encode(
                Opcode::I2fCbuf,
                2 | 1 << 34 | 3 << 20 | 1 << 8 | 1 << 10 | 2 << 41,
            ),
            &[
                "%0 = GetCbufU32 #0x1, #0xc",
                "%1 = BitFieldUExtract %0, #0x10, #0x10",
                "%2 = ConvertF16U32 %1 (flags 0x2)",
                "%3 = ConvertF16F32 #0.0f",
                "%4 = CompositeConstructF16x2 %2, %3",
                "%5 = PackFloat2x16 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn i2f_imm_s64_to_f64() {
        check(
            encode(Opcode::I2fImm, 2 | 5 << 20 | 3 << 8 | 3 << 10 | 1 << 13),
            &[
                "%0 = ConvertF64S64 #0x500000000u64 (flags 0x2)",
                "%1 = UnpackDouble2x32 %0",
                "%2 = CompositeExtractU32x2 %1, #0x0",
                "SetRegister R2, %2",
                "%3 = CompositeExtractU32x2 %1, #0x1",
                "SetRegister R3, %3",
            ],
        );
    }

    #[test]
    fn i2f_rejects_invalid_encodings() {
        rejects(encode(Opcode::I2fReg, 2 | 6 << 20 | 2 << 10));
        rejects(encode(
            Opcode::I2fReg,
            2 | 6 << 20 | 2 << 8 | 2 << 10 | 1 << 47,
        ));
        rejects(encode(
            Opcode::I2fReg,
            2 | 6 << 20 | 2 << 8 | 1 << 10 | 1 << 41,
        ));
        rejects(encode(
            Opcode::I2fReg,
            2 | 6 << 20 | 2 << 8 | 2 << 10 | 1 << 41,
        ));
        rejects(encode(Opcode::I2fReg, 3 | 6 << 20 | 3 << 8 | 2 << 10));
    }

    #[test]
    fn i2f_reg_u64_rp_rounds_inexact_values_up() {
        check(
            encode(Opcode::I2fReg, 2 | 6 << 20 | 2 << 8 | 3 << 10 | 2 << 39),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R7",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackUint2x32 %2",
                "%4 = UnpackUint2x32 %3",
                "%5 = CompositeExtractU32x2 %4, #0x0",
                "%6 = UnpackUint2x32 %3",
                "%7 = CompositeExtractU32x2 %6, #0x1",
                "%8 = INotEqual %7, #0x0",
                "%9 = FindUMsb32 %7",
                "%10 = IAdd32 %9, #0x20",
                "%11 = INotEqual %5, #0x0",
                "%12 = FindUMsb32 %5",
                "%13 = SelectU32 %11, %12, #0x0",
                "%14 = SelectU32 %8, %10, %13",
                "%15 = UMax32 %14, #0x17",
                "%16 = ISub32 %15, #0x17",
                "%17 = ShiftRightLogical64 %3, %16",
                "%18 = UnpackUint2x32 %17",
                "%19 = CompositeExtractU32x2 %18, #0x0",
                "%20 = IAdd32 %14, #0x7f",
                "%21 = ShiftLeftLogical32 %20, #0x17",
                "%22 = BitwiseAnd32 %19, #0x7fffff",
                "%23 = BitwiseOr32 %21, %22",
                "%24 = ShiftLeftLogical64 %17, %16",
                "%25 = UnpackUint2x32 %3",
                "%26 = UnpackUint2x32 %24",
                "%27 = CompositeExtractU32x2 %25, #0x0",
                "%28 = CompositeExtractU32x2 %26, #0x0",
                "%29 = IEqual %27, %28",
                "%30 = CompositeExtractU32x2 %25, #0x1",
                "%31 = CompositeExtractU32x2 %26, #0x1",
                "%32 = IEqual %30, %31",
                "%33 = LogicalAnd %29, %32",
                "%34 = LogicalNot %33",
                "%35 = SelectU32 %34, #0x1, #0x0",
                "%36 = IAdd32 %23, %35",
                "%37 = BitwiseAnd32 %5, #0xffffff",
                "%38 = ConvertF32U32 %37",
                "%39 = UGreaterThan %14, #0x17",
                "%40 = BitCastF32U32 %36",
                "%41 = SelectF32 %39, %40, %38",
                "%42 = BitCastU32F32 %41",
                "SetRegister R2, %42",
            ],
        );
    }

    #[test]
    fn i2f_reg_u64_rn_rounds_to_nearest_even() {
        let dump = translate(&[encode(Opcode::I2fReg, 2 | 6 << 20 | 2 << 8 | 3 << 10)]).unwrap();
        assert_eq!(dump.matches("ShiftRightLogical64").count(), 2);
        assert_eq!(dump.matches("ShiftLeftLogical64").count(), 1);
        assert!(dump.contains("LogicalAnd"));
    }

    #[test]
    fn i2f_reg_s32_rp_in_native_programs() {
        check_native(
            encode(
                Opcode::I2fReg,
                2 | 6 << 20 | 2 << 8 | 2 << 10 | 1 << 13 | 2 << 39,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = ConvertF32S32 %0",
                "%2 = FPAbs32 %1",
                "%3 = ConvertU32F32 %2",
                "%4 = SLessThan %0, #0x0",
                "%5 = INeg32 %0",
                "%6 = SelectU32 %4, %5, %0",
                "%7 = BitCastU32F32 %1",
                "%8 = ULessThan %3, %6",
                "%9 = IAdd32 %7, #0x1",
                "%10 = SelectU32 %8, %9, %7",
                "%11 = UGreaterThan %3, %6",
                "%12 = ISub32 %7, #0x1",
                "%13 = SelectU32 %11, %12, %7",
                "%14 = SelectU32 %4, %13, %10",
                "%15 = BitCastF32U32 %14",
                "%16 = BitCastU32F32 %15",
                "SetRegister R2, %16",
            ],
        );
    }

    #[test]
    fn i2f_reg_s32_rp_on_maxwell_uses_the_plain_conversion() {
        check(
            encode(
                Opcode::I2fReg,
                2 | 6 << 20 | 2 << 8 | 2 << 10 | 1 << 13 | 2 << 39,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = ConvertF32S32 %0 (flags 0x6)",
                "%2 = BitCastU32F32 %1",
                "SetRegister R2, %2",
            ],
        );
    }
}
