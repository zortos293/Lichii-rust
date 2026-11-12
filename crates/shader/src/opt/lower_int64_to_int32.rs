//! 64-bit integer arithmetic rewritten on pairs of 32-bit words, for hosts without
//! `shaderInt64`.

use super::constant_propagation::has_pseudo_op;
use crate::ir::{Emitter, FpControl, ICompare, InstId, Opcode, Program, Type, Value};

/// Rewrites U64 values as `U32x2` pairs (low word first) in reverse post order, so every
/// operand is lowered before its users. Fails on 64-bit arithmetic whose flags a
/// pseudo-instruction reads, which has no 32-bit form yet.
pub fn lower_int64_to_int32(prog: &mut Program) -> Result<(), String> {
    for i in (0..prog.post_order.len()).rev() {
        let mut cur = prog.block(prog.post_order[i]).first();
        while let Some(id) = cur {
            cur = prog.inst(id).next();
            lower(prog, id)?;
        }
    }
    Ok(())
}

fn lower(prog: &mut Program, id: InstId) -> Result<(), String> {
    use Opcode::*;
    let op = prog.op(id);
    let lowered = match op {
        Phi => {
            if Type::from_bits_truncate(prog.inst(id).flags) == Type::U64 {
                prog.set_flags(id, Type::U32X2.bits());
            }
            return Ok(());
        }
        PackUint2x32 | UnpackUint2x32 => Identity,
        SharedAtomicExchange64 => SharedAtomicExchange32x2,
        GlobalAtomicIAdd64 => GlobalAtomicIAdd32x2,
        GlobalAtomicSMin64 => GlobalAtomicSMin32x2,
        GlobalAtomicUMin64 => GlobalAtomicUMin32x2,
        GlobalAtomicSMax64 => GlobalAtomicSMax32x2,
        GlobalAtomicUMax64 => GlobalAtomicUMax32x2,
        GlobalAtomicAnd64 => GlobalAtomicAnd32x2,
        GlobalAtomicOr64 => GlobalAtomicOr32x2,
        GlobalAtomicXor64 => GlobalAtomicXor32x2,
        GlobalAtomicExchange64 => GlobalAtomicExchange32x2,
        StorageAtomicIAdd64 => StorageAtomicIAdd32x2,
        StorageAtomicSMin64 => StorageAtomicSMin32x2,
        StorageAtomicUMin64 => StorageAtomicUMin32x2,
        StorageAtomicSMax64 => StorageAtomicSMax32x2,
        StorageAtomicUMax64 => StorageAtomicUMax32x2,
        StorageAtomicAnd64 => StorageAtomicAnd32x2,
        StorageAtomicOr64 => StorageAtomicOr32x2,
        StorageAtomicXor64 => StorageAtomicXor32x2,
        StorageAtomicExchange64 => StorageAtomicExchange32x2,
        BitCastU64F64 => UnpackDouble2x32,
        BitCastF64U64 => PackDouble2x32,
        IAdd64
        | ISub64
        | INeg64
        | ShiftLeftLogical64
        | ShiftRightLogical64
        | ShiftRightArithmetic64
            if has_pseudo_op(prog, id) =>
        {
            return Err(format!("{} emulation with pseudo instructions", op.name()));
        }
        _ => {
            emulate(prog, id, op);
            return Ok(());
        }
    };
    prog.replace_opcode(id, lowered);
    Ok(())
}

fn emulate(prog: &mut Program, id: InstId, op: Opcode) {
    use Opcode::*;
    let arg = |i: usize| prog.arg(id, i);
    let value = match op {
        IAdd64 => {
            let (a, b) = (arg(0), arg(1));
            with(prog, id, |e| {
                let (a_lo, a_hi) = unpack(e, a);
                let (b_lo, b_hi) = unpack(e, b);
                let lo = e.iadd(a_lo, b_lo);
                let carry = carry_bit(e, lo);
                let hi = e.iadd(a_hi, b_hi);
                let hi = e.iadd(hi, carry);
                e.composite_construct(&[lo, hi])
            })
        }
        ISub64 => {
            let (a, b) = (arg(0), arg(1));
            with(prog, id, |e| {
                let (a_lo, a_hi) = unpack(e, a);
                let (b_lo, b_hi) = unpack(e, b);
                let lo = e.isub(a_lo, b_lo);
                let underflow = e.icompare(ICompare::UGreaterThan, lo, a_lo);
                let borrow = e.select(underflow, Value::U32(1), Value::U32(0));
                let hi = e.isub(a_hi, b_hi);
                let hi = e.isub(hi, borrow);
                e.composite_construct(&[lo, hi])
            })
        }
        INeg64 => {
            let a = arg(0);
            with(prog, id, |e| {
                let (lo, hi) = unpack(e, a);
                let (lo, hi) = negate(e, lo, hi);
                e.composite_construct(&[lo, hi])
            })
        }
        IAbs64 => {
            let a = arg(0);
            with(prog, id, |e| {
                let (lo, hi) = unpack(e, a);
                let (neg_lo, neg_hi) = negate(e, lo, hi);
                let sign = e.bitwise_and(hi, Value::U32(0x8000_0000));
                let negative = e.icompare(ICompare::NotEqual, sign, Value::U32(0));
                let lo = e.select(negative, neg_lo, lo);
                let hi = e.select(negative, neg_hi, hi);
                e.composite_construct(&[lo, hi])
            })
        }
        ShiftLeftLogical64 | ShiftRightLogical64 | ShiftRightArithmetic64 => {
            let (a, shift) = (arg(0), arg(1));
            with(prog, id, |e| shift64(e, op, a, shift))
        }
        SelectU64 => {
            let (cond, t, f) = (arg(0), arg(1), arg(2));
            with(prog, id, |e| {
                let (t_lo, t_hi) = unpack(e, t);
                let (f_lo, f_hi) = unpack(e, f);
                let lo = e.select(cond, t_lo, f_lo);
                let hi = e.select(cond, t_hi, f_hi);
                e.composite_construct(&[lo, hi])
            })
        }
        UndefU64 => with(prog, id, |e| {
            e.composite_construct(&[Value::U32(0), Value::U32(0)])
        }),
        ConvertU64U32 => {
            let a = arg(0);
            with(prog, id, |e| e.composite_construct(&[a, Value::U32(0)]))
        }
        ConvertU32U64 => {
            let a = arg(0);
            with(prog, id, |e| unpack(e, a).0)
        }
        ConvertS64F16 | ConvertS64F32 | ConvertS64F64 => {
            let a = arg(0);
            with(prog, id, |e| float_to_int64(e, a, true))
        }
        ConvertU64F16 | ConvertU64F32 | ConvertU64F64 => {
            let a = arg(0);
            with(prog, id, |e| float_to_int64(e, a, false))
        }
        ConvertF16S64 | ConvertF32S64 | ConvertF64S64 => {
            let a = arg(0);
            with(prog, id, |e| int64_to_float(e, a, true, float_bits(op)))
        }
        ConvertF16U64 | ConvertF32U64 | ConvertF64U64 => {
            let a = arg(0);
            with(prog, id, |e| int64_to_float(e, a, false, float_bits(op)))
        }
        _ => return,
    };
    prog.replace_uses_with(id, value);
}

fn with(prog: &mut Program, id: InstId, f: impl FnOnce(&mut Emitter) -> Value) -> Value {
    f(&mut Emitter::before(prog, id))
}

fn float_bits(op: Opcode) -> u32 {
    match op {
        Opcode::ConvertF16S64 | Opcode::ConvertF16U64 => 16,
        Opcode::ConvertF64S64 | Opcode::ConvertF64U64 => 64,
        _ => 32,
    }
}

fn unpack(e: &mut Emitter, packed: Value) -> (Value, Value) {
    match packed {
        Value::U64(v) => (Value::U32(v as u32), Value::U32((v >> 32) as u32)),
        _ => (
            e.composite_extract(packed, 0),
            e.composite_extract(packed, 1),
        ),
    }
}

fn carry_bit(e: &mut Emitter, sum: Value) -> Value {
    let carry = e.get_carry_from_op(sum);
    e.select(carry, Value::U32(1), Value::U32(0))
}

fn negate(e: &mut Emitter, lo: Value, hi: Value) -> (Value, Value) {
    let not_lo = e.bitwise_not(lo);
    let not_hi = e.bitwise_not(hi);
    let lo = e.iadd(not_lo, Value::U32(1));
    let carry = carry_bit(e, lo);
    (lo, e.iadd(not_hi, carry))
}

/// Shifts below 32 move bits across the word boundary, shifts of 32 or more move one word into
/// the other, and a shift of 0 keeps the value (the cross-word field would be 32 bits wide).
fn shift64(e: &mut Emitter, op: Opcode, value: Value, shift: Value) -> Value {
    let (lo, hi) = unpack(e, value);
    let inv_shift = e.isub(shift, Value::U32(32));
    let is_long = e.icompare(ICompare::SGreaterThanEqual, inv_shift, Value::U32(0));
    let is_zero = e.icompare(ICompare::Equal, shift, Value::U32(0));
    let shift_complement = e.isub(Value::U32(32), shift);
    let (short, long) = if op == Opcode::ShiftLeftLogical64 {
        let short_lo = e.shift_left_logical(lo, shift);
        let shifted_hi = e.shift_left_logical(hi, shift);
        let carried = e.bit_field_extract(lo, shift_complement, shift, false);
        let short_hi = e.bitwise_or(shifted_hi, carried);
        let long_hi = e.shift_left_logical(lo, inv_shift);
        ((short_lo, short_hi), (Value::U32(0), long_hi))
    } else {
        let arithmetic = op == Opcode::ShiftRightArithmetic64;
        let shift_hi = |e: &mut Emitter, by: Value| {
            if arithmetic {
                e.shift_right_arithmetic(hi, by)
            } else {
                e.shift_right_logical(hi, by)
            }
        };
        let shifted_lo = e.shift_right_logical(lo, shift);
        let short_hi = shift_hi(e, shift);
        let carried = e.bit_field_extract(hi, Value::U32(0), shift, false);
        let short_lo = e.bit_field_insert(shifted_lo, carried, shift_complement, shift);
        let long_lo = shift_hi(e, inv_shift);
        let long_hi = if arithmetic {
            e.shift_right_arithmetic(hi, Value::U32(31))
        } else {
            Value::U32(0)
        };
        ((short_lo, short_hi), (long_lo, long_hi))
    };
    let non_zero_lo = e.select(is_long, long.0, short.0);
    let non_zero_hi = e.select(is_long, long.1, short.1);
    let lo = e.select(is_zero, lo, non_zero_lo);
    let hi = e.select(is_zero, hi, non_zero_hi);
    e.composite_construct(&[lo, hi])
}

const TWO_POW_32: f32 = 4_294_967_296.0;

/// The arithmetic runs in F64 when the destination is F64 (the host then has float64), else
/// in F32.
fn int64_to_float(e: &mut Emitter, value: Value, signed: bool, dst_bits: u32) -> Value {
    let control = FpControl::default();
    let (lo, hi) = unpack(e, value);
    let (bits, two_pow_32) = match dst_bits {
        64 => (64, Value::F64(f64::from(TWO_POW_32))),
        _ => (32, Value::F32(TWO_POW_32)),
    };
    let low = e.convert_i_to_f(bits, 32, false, lo, control);
    let high = e.convert_i_to_f(bits, 32, signed, hi, control);
    let combined = e.fp_fma(high, two_pow_32, low, control);
    e.fp_convert(dst_bits, combined, control)
}

/// The arithmetic runs in F64 for an F64 source, else in F32.
fn float_to_int64(e: &mut Emitter, value: Value, signed: bool) -> Value {
    let control = FpControl::default();
    let (value, two_pow_32, inv_two_pow_32) = if e.ty(value) == Type::F64 {
        let two = f64::from(TWO_POW_32);
        (value, Value::F64(two), Value::F64(1.0 / two))
    } else {
        let value = e.fp_convert(32, value, control);
        (value, Value::F32(TWO_POW_32), Value::F32(1.0 / TWO_POW_32))
    };
    let scaled = e.fp_mul(value, inv_two_pow_32, control);
    let high_f = e.fp_floor(scaled, control);
    let hi = e.convert_f_to_i(32, signed, high_f);
    let minus_two_pow_32 = e.fp_neg(two_pow_32);
    let low_f = e.fp_fma(high_f, minus_two_pow_32, value, control);
    let lo = e.convert_f_to_i(32, false, low_f);
    e.composite_construct(&[lo, hi])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::BlockId;
    use crate::opt::eval_tests::eval;

    fn program() -> (Program, BlockId) {
        let mut prog = Program::new();
        let block = prog.new_block();
        prog.block_order = vec![block];
        prog.post_order = vec![block];
        (prog, block)
    }

    /// A U64 operand: an immediate, or a word pair the lowering has to unpack.
    fn operand(prog: &mut Program, b: BlockId, v: u64, immediate: bool) -> Value {
        if immediate {
            return Value::U64(v);
        }
        let pair = prog.append(
            b,
            Opcode::CompositeConstructU32x2,
            &[Value::U32(v as u32), Value::U32((v >> 32) as u32)],
            0,
        );
        Value::Inst(prog.append(b, Opcode::PackUint2x32, &[Value::Inst(pair)], 0))
    }

    #[test]
    fn double_conversions_stay_in_double_precision() {
        for op in [Opcode::ConvertS64F64, Opcode::ConvertF64U64] {
            let (mut prog, b) = program();
            let source = match op {
                Opcode::ConvertS64F64 => Value::F64(16_777_217.0),
                _ => operand(&mut prog, b, 16_777_217, false),
            };
            let conversion = prog.append(b, op, &[source], 0);
            prog.append(b, Opcode::Reference, &[Value::Inst(conversion)], 0);
            lower_int64_to_int32(&mut prog).unwrap();
            let ops: Vec<Opcode> = prog.insts_of(b).map(|id| prog.op(id)).collect();
            assert!(!ops.contains(&Opcode::ConvertF32F64), "{op:?}: {ops:?}");
            assert!(!ops.contains(&Opcode::FPFma32), "{op:?}: {ops:?}");
        }
    }

    fn run(op: Opcode, a: u64, b: Option<u64>, immediate: bool) -> u64 {
        let (mut prog, blk) = program();
        let mut args = vec![operand(&mut prog, blk, a, immediate)];
        match (op, b) {
            (
                Opcode::ShiftLeftLogical64
                | Opcode::ShiftRightLogical64
                | Opcode::ShiftRightArithmetic64,
                Some(b),
            ) => args.push(Value::U32(b as u32)),
            (_, Some(b)) => args.push(operand(&mut prog, blk, b, immediate)),
            (_, None) => {}
        }
        let result = prog.append(blk, op, &args, 0);
        prog.append(blk, Opcode::Reference, &[Value::Inst(result)], 0);
        lower_int64_to_int32(&mut prog).unwrap();
        assert_eq!(crate::opt::verify(&prog), Ok(()));
        eval(&prog, Value::Inst(result))
    }

    const VALUES: [u64; 6] = [
        0,
        1,
        0xffff_ffff,
        0x1_0000_0000,
        0x8000_0000_0000_0001,
        0xdead_beef_cafe_f00d,
    ];

    #[test]
    fn add_sub_neg_abs_match_u64_arithmetic() {
        for (&a, immediate) in VALUES.iter().zip([true, false].iter().cycle()) {
            for &b in &VALUES {
                assert_eq!(
                    run(Opcode::IAdd64, a, Some(b), *immediate),
                    a.wrapping_add(b)
                );
                assert_eq!(
                    run(Opcode::ISub64, a, Some(b), *immediate),
                    a.wrapping_sub(b)
                );
            }
            assert_eq!(run(Opcode::INeg64, a, None, *immediate), a.wrapping_neg());
            assert_eq!(
                run(Opcode::IAbs64, a, None, *immediate),
                (a as i64).wrapping_abs() as u64
            );
        }
    }

    #[test]
    fn shifts_match_u64_arithmetic_on_both_sides_of_32() {
        for &a in &VALUES {
            for shift in [0, 1, 16, 31, 32, 33, 63] {
                for immediate in [true, false] {
                    let sll = run(Opcode::ShiftLeftLogical64, a, Some(shift), immediate);
                    let srl = run(Opcode::ShiftRightLogical64, a, Some(shift), immediate);
                    let sra = run(Opcode::ShiftRightArithmetic64, a, Some(shift), immediate);
                    assert_eq!(sll, a << shift, "{a:#x} << {shift}");
                    assert_eq!(srl, a >> shift, "{a:#x} >> {shift}");
                    assert_eq!(sra, ((a as i64) >> shift) as u64, "{a:#x} >>a {shift}");
                }
            }
        }
    }

    #[test]
    fn selects_undefs_conversions_and_phis_become_word_pairs() {
        let (mut prog, b) = program();
        let phi = prog.append(b, Opcode::Phi, &[], Type::U64.bits());
        let x = prog.append(
            b,
            Opcode::GetRegister,
            &[Value::Reg(crate::ir::Reg::r(1))],
            0,
        );
        let wide = prog.append(b, Opcode::ConvertU64U32, &[Value::Inst(x)], 0);
        let undef = prog.append(b, Opcode::UndefU64, &[], 0);
        let pick = prog.append(
            b,
            Opcode::SelectU64,
            &[Value::U1(true), Value::Inst(wide), Value::Inst(undef)],
            0,
        );
        let narrow = prog.append(b, Opcode::ConvertU32U64, &[Value::Inst(pick)], 0);
        let float = prog.append(b, Opcode::ConvertF32U64, &[Value::Inst(pick)], 0);
        let unrelated = prog.append(b, Opcode::IAdd32, &[Value::Inst(x), Value::U32(1)], 0);
        prog.append(b, Opcode::Reference, &[Value::Inst(narrow)], 0);
        prog.append(b, Opcode::Reference, &[Value::Inst(float)], 0);
        lower_int64_to_int32(&mut prog).unwrap();
        assert_eq!(prog.type_of(Value::Inst(phi)), Type::U32X2);
        for id in [wide, undef, pick] {
            assert_eq!(prog.type_of(Value::Inst(id)), Type::U32X2);
        }
        assert_eq!(prog.type_of(Value::Inst(narrow)), Type::U32);
        assert_eq!(prog.type_of(Value::Inst(float)), Type::F32);
        assert_eq!(prog.op(unrelated), Opcode::IAdd32);
        assert_eq!(crate::opt::verify(&prog), Ok(()));
    }

    #[test]
    fn arithmetic_with_flag_readers_is_refused() {
        let (mut prog, b) = program();
        let sum = prog.append(b, Opcode::IAdd64, &[Value::U64(1), Value::U64(2)], 0);
        prog.append(b, Opcode::GetCarryFromOp, &[Value::Inst(sum)], 0);
        assert!(lower_int64_to_int32(&mut prog).is_err());
    }
}
