//! Double precision arithmetic rewritten as single precision, for hosts without
//! `shaderFloat64`.

use crate::ir::{Emitter, Opcode, Program, Type, Value};

/// Biased exponent deltas between doubles (bias 1023) and singles (bias 127).
const F64_TO_F32_EXP: i32 = 127 - 1023;
const F32_TO_F64_EXP: i32 = 1023 - 127;

/// Replaces every F64 opcode with its F32 counterpart, narrows F64 immediates and phis, and
/// rebuilds `PackDouble2x32`/`UnpackDouble2x32` from the IEEE bit layouts. Doubles outside
/// the single range overflow the exponent field, as on the C++ side.
pub fn lower_fp64_to_fp32(prog: &mut Program) {
    for i in 0..prog.block_order.len() {
        let mut cur = prog.block(prog.block_order[i]).first();
        while let Some(id) = cur {
            cur = prog.inst(id).next();
            match prog.op(id) {
                Opcode::Phi => {
                    let ty = Type::from_bits_truncate(prog.inst(id).flags);
                    prog.set_flags(id, retype(ty).bits());
                    narrow_immediates(prog, id);
                }
                Opcode::PackDouble2x32 => {
                    let packed = prog.arg(id, 0);
                    let value = packed_f64_to_f32(&mut Emitter::before(prog, id), packed);
                    prog.replace_uses_with(id, value);
                }
                Opcode::UnpackDouble2x32 => {
                    let raw = prog.arg(id, 0);
                    let value = f32_to_packed_f64(&mut Emitter::before(prog, id), raw);
                    prog.replace_uses_with(id, value);
                }
                op => {
                    if let Some(lowered) = replace(op) {
                        prog.replace_opcode(id, lowered);
                        narrow_immediates(prog, id);
                    }
                }
            }
        }
    }
}

fn narrow_immediates(prog: &mut Program, id: crate::ir::InstId) {
    for index in 0..prog.inst(id).num_args() {
        if let Value::F64(v) = prog.arg(id, index) {
            prog.set_arg(id, index, Value::F32(v as f32));
        }
    }
}

fn packed_f64_to_f32(e: &mut Emitter, packed: Value) -> Value {
    let field = |e: &mut Emitter, v: Value, offset: u32, count: u32| {
        e.bit_field_extract(v, Value::U32(offset), Value::U32(count), false)
    };
    let lo = e.composite_extract(packed, 0);
    let hi = e.composite_extract(packed, 1);
    let sign = field(e, hi, 31, 1);
    let exp = field(e, hi, 20, 11);
    let mantissa_hi = field(e, hi, 0, 20);
    let mantissa_lo = field(e, lo, 29, 3);
    let mantissa_hi = e.shift_left_logical(mantissa_hi, Value::U32(3));
    let mantissa = e.bitwise_or(mantissa_hi, mantissa_lo);
    let exp = rebias(e, exp, 0x7ff, 0xff, F64_TO_F32_EXP);
    let sign = e.shift_left_logical(sign, Value::U32(31));
    let exp = e.shift_left_logical(exp, Value::U32(23));
    let bits = e.bitwise_or(exp, mantissa);
    let bits = e.bitwise_or(sign, bits);
    e.bitcast_f32(bits)
}

fn f32_to_packed_f64(e: &mut Emitter, raw: Value) -> Value {
    let field = |e: &mut Emitter, v: Value, offset: u32, count: u32| {
        e.bit_field_extract(v, Value::U32(offset), Value::U32(count), false)
    };
    let value = e.bitcast_u32(raw);
    let sign = field(e, value, 31, 1);
    let exp = field(e, value, 23, 8);
    let mantissa = field(e, value, 0, 23);
    let mantissa_hi = field(e, mantissa, 3, 20);
    let mantissa_lo = field(e, mantissa, 0, 3);
    let exp = rebias(e, exp, 0xff, 0x7ff, F32_TO_F64_EXP);
    let lo = e.shift_left_logical(mantissa_lo, Value::U32(29));
    let sign = e.shift_left_logical(sign, Value::U32(31));
    let exp = e.shift_left_logical(exp, Value::U32(20));
    let hi = e.bitwise_or(exp, mantissa_hi);
    let hi = e.bitwise_or(sign, hi);
    e.composite_construct(&[lo, hi])
}

/// Moves a biased exponent to the other width, keeping zero (and denormals) at zero and
/// infinity/NaN at all ones.
fn rebias(e: &mut Emitter, exp: Value, from_max: u32, to_max: u32, delta: i32) -> Value {
    let is_zero = e.inst(Opcode::IEqual, &[exp, Value::U32(0)]);
    let shifted = e.iadd(exp, Value::U32(delta as u32));
    let finite = e.select(is_zero, Value::U32(0), shifted);
    let is_max = e.inst(Opcode::IEqual, &[exp, Value::U32(from_max)]);
    e.select(is_max, Value::U32(to_max), finite)
}

fn retype(ty: Type) -> Type {
    match ty {
        t if t == Type::F64 => Type::F32,
        t if t == Type::F64X2 => Type::F32X2,
        t if t == Type::F64X3 => Type::F32X3,
        t if t == Type::F64X4 => Type::F32X4,
        t => t,
    }
}

fn replace(op: Opcode) -> Option<Opcode> {
    use Opcode::*;
    Some(match op {
        FPAbs64 => FPAbs32,
        FPAdd64 => FPAdd32,
        FPCeil64 => FPCeil32,
        FPFloor64 => FPFloor32,
        FPFma64 => FPFma32,
        FPMax64 => FPMax32,
        FPMin64 => FPMin32,
        FPMul64 => FPMul32,
        FPNeg64 => FPNeg32,
        FPRecip64 => FPRecip32,
        FPRecipSqrt64 => FPRecipSqrt32,
        FPRoundEven64 => FPRoundEven32,
        FPSaturate64 => FPSaturate32,
        FPClamp64 => FPClamp32,
        FPTrunc64 => FPTrunc32,
        SelectF64 => SelectF32,
        CompositeConstructF64x2 => CompositeConstructF32x2,
        CompositeConstructF64x3 => CompositeConstructF32x3,
        CompositeConstructF64x4 => CompositeConstructF32x4,
        CompositeExtractF64x2 => CompositeExtractF32x2,
        CompositeExtractF64x3 => CompositeExtractF32x3,
        CompositeExtractF64x4 => CompositeExtractF32x4,
        CompositeInsertF64x2 => CompositeInsertF32x2,
        CompositeInsertF64x3 => CompositeInsertF32x3,
        CompositeInsertF64x4 => CompositeInsertF32x4,
        FPOrdEqual64 => FPOrdEqual32,
        FPUnordEqual64 => FPUnordEqual32,
        FPOrdNotEqual64 => FPOrdNotEqual32,
        FPUnordNotEqual64 => FPUnordNotEqual32,
        FPOrdLessThan64 => FPOrdLessThan32,
        FPUnordLessThan64 => FPUnordLessThan32,
        FPOrdGreaterThan64 => FPOrdGreaterThan32,
        FPUnordGreaterThan64 => FPUnordGreaterThan32,
        FPOrdLessThanEqual64 => FPOrdLessThanEqual32,
        FPUnordLessThanEqual64 => FPUnordLessThanEqual32,
        FPOrdGreaterThanEqual64 => FPOrdGreaterThanEqual32,
        FPUnordGreaterThanEqual64 => FPUnordGreaterThanEqual32,
        FPIsNan64 => FPIsNan32,
        ConvertS16F64 => ConvertS16F32,
        ConvertS32F64 => ConvertS32F32,
        ConvertS64F64 => ConvertS64F32,
        ConvertU16F64 => ConvertU16F32,
        ConvertU32F64 => ConvertU32F32,
        ConvertU64F64 => ConvertU64F32,
        ConvertF32F64 | ConvertF64F32 => Identity,
        ConvertF64S8 => ConvertF32S8,
        ConvertF64S16 => ConvertF32S16,
        ConvertF64S32 => ConvertF32S32,
        ConvertF64S64 => ConvertF32S64,
        ConvertF64U8 => ConvertF32U8,
        ConvertF64U16 => ConvertF32U16,
        ConvertF64U32 => ConvertF32U32,
        ConvertF64U64 => ConvertF32U64,
        _ => return None,
    })
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

    fn pack(bits: u64) -> u32 {
        let (mut prog, b) = program();
        let pair = prog.append(
            b,
            Opcode::CompositeConstructU32x2,
            &[Value::U32(bits as u32), Value::U32((bits >> 32) as u32)],
            0,
        );
        let double = prog.append(b, Opcode::PackDouble2x32, &[Value::Inst(pair)], 0);
        lower_fp64_to_fp32(&mut prog);
        eval(&prog, Value::Inst(double)) as u32
    }

    fn unpack(single: f32) -> u64 {
        let (mut prog, b) = program();
        let pair = prog.append(b, Opcode::UnpackDouble2x32, &[Value::F32(single)], 0);
        lower_fp64_to_fp32(&mut prog);
        eval(&prog, Value::Inst(pair))
    }

    const SAMPLES: [f32; 7] = [1.0, -2.25, 1.5e10, 3.0e-20, 0.0, f32::INFINITY, -0.0];

    #[test]
    fn packed_doubles_narrow_to_the_same_single() {
        for single in SAMPLES {
            assert_eq!(
                pack(f64::from(single).to_bits()),
                single.to_bits(),
                "{single}"
            );
        }
    }

    #[test]
    fn singles_widen_to_the_same_packed_double() {
        for single in SAMPLES {
            assert_eq!(unpack(single), f64::from(single).to_bits(), "{single}");
        }
    }

    #[test]
    fn double_ops_immediates_and_phis_become_single() {
        let (mut prog, b) = program();
        let phi = prog.append(b, Opcode::Phi, &[], Type::F64.bits());
        prog.add_phi_operand(phi, b, Value::F64(2.0));
        let sum = prog.append(b, Opcode::FPAdd64, &[Value::Inst(phi), Value::F64(0.5)], 0);
        let narrow = prog.append(b, Opcode::ConvertF32F64, &[Value::Inst(sum)], 0);
        let unrelated = prog.append(b, Opcode::IAdd32, &[Value::U32(1), Value::U32(2)], 0);
        lower_fp64_to_fp32(&mut prog);
        assert_eq!(prog.type_of(Value::Inst(phi)), Type::F32);
        assert_eq!(prog.arg(phi, 0), Value::F32(2.0));
        assert_eq!(prog.op(sum), Opcode::FPAdd32);
        assert_eq!(prog.arg(sum, 1), Value::F32(0.5));
        assert_eq!(prog.op(narrow), Opcode::Identity);
        assert_eq!(prog.op(unrelated), Opcode::IAdd32);
        assert_eq!(crate::opt::verify(&prog), Ok(()));
    }
}
