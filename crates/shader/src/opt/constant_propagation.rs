//! Constant folding and peephole rewrites of guest compiler idioms.
//!
//! One walk in reverse post order. Every instruction is folded once, after the instructions it
//! reads, so no fold ever revisits users. Folded instructions become identities; run identity
//! removal and dead code elimination afterwards.

use rustc_hash::FxHashMap;

use crate::ShaderEnvironment;
use crate::info::ReplaceConstant;
use crate::ir::{
    Attribute, Emitter, FmzMode, FpControl, FpRounding, InstId, Opcode, Program, TextureInstInfo,
    TextureType, Type, Value,
};

const PSEUDO_OPS: [Opcode; 6] = [
    Opcode::GetZeroFromOp,
    Opcode::GetSignFromOp,
    Opcode::GetCarryFromOp,
    Opcode::GetOverflowFromOp,
    Opcode::GetSparseFromOp,
    Opcode::GetInBoundsFromOp,
];

const MASK_UNKNOWN: u32 = 0;
const MASK_MINUS_ONE: u32 = 1 << 0;
const MASK_ZERO: u32 = 1 << 1;
const MASK_ONE: u32 = 1 << 2;
const MAX_MASK_DEPTH: u32 = 8;

/// Results shared by the folds of one walk, so repeated queries stay linear.
#[derive(Default)]
struct Memo {
    /// Value of `CompositeExtract(insert, index)` for each insert visited on a chain.
    extracts: FxHashMap<(InstId, u32), Option<Value>>,
    /// Predicate mask value sets that the depth limit did not cut short.
    masks: FxHashMap<InstId, u32>,
}

/// Folds immediates and the idioms guest compilers emit: XMAD multiplies, constant buffer
/// cancellation, quad swizzle derivatives, predicate mask normalisation and HLE draw
/// parameters.
pub fn constant_propagation(env: &dyn ShaderEnvironment, prog: &mut Program) {
    let mut memo = Memo::default();
    for i in (0..prog.post_order.len()).rev() {
        let mut cur = prog.block(prog.post_order[i]).first();
        while let Some(id) = cur {
            cur = prog.inst(id).next();
            fold(env, prog, &mut memo, id);
        }
    }
}

pub(super) fn has_pseudo_op(prog: &Program, id: InstId) -> bool {
    PSEUDO_OPS
        .iter()
        .any(|&op| prog.pseudo_op(id, op).is_some())
}

fn arg(prog: &Program, id: InstId, index: usize) -> Value {
    prog.resolve(prog.arg(id, index))
}

fn producer(prog: &Program, value: Value, op: Opcode) -> Option<InstId> {
    prog.resolve(value).inst().filter(|&p| prog.op(p) == op)
}

fn binary(prog: &Program, value: Value, op: Opcode) -> Option<(Value, Value)> {
    let p = producer(prog, value, op)?;
    Some((arg(prog, p, 0), arg(prog, p, 1)))
}

/// Base of `BitFieldUExtract(base, offset, count)`.
fn bfe(prog: &Program, value: Value, offset: u32, count: u32) -> Option<Value> {
    let p = producer(prog, value, Opcode::BitFieldUExtract)?;
    (arg(prog, p, 1) == Value::U32(offset) && arg(prog, p, 2) == Value::U32(count))
        .then(|| arg(prog, p, 0))
}

/// Base of `ShiftLeftLogical32(base, 16)`.
fn shl16(prog: &Program, value: Value) -> Option<Value> {
    let (base, shift) = binary(prog, value, Opcode::ShiftLeftLogical32)?;
    (shift == Value::U32(16)).then_some(base)
}

fn replace_with_new(prog: &mut Program, id: InstId, f: impl FnOnce(&mut Emitter) -> Value) {
    let value = f(&mut Emitter::before(prog, id));
    prog.replace_uses_with(id, value);
}

fn resolve_args(prog: &mut Program, id: InstId) {
    for index in 0..prog.inst(id).num_args() {
        let value = prog.arg(id, index);
        if let Value::Inst(target) = value
            && prog.op(target) == Opcode::Identity
        {
            let resolved = prog.resolve(value);
            prog.set_arg(id, index, resolved);
        }
    }
}

fn fold(env: &dyn ShaderEnvironment, prog: &mut Program, memo: &mut Memo, id: InstId) {
    use Opcode::*;
    let op = prog.op(id);
    if matches!(op, Phi | Identity | Void) || op.is_pseudo() {
        return;
    }
    resolve_args(prog, id);
    match op {
        GetRegister => {
            if let Value::Reg(reg) = prog.arg(id, 0)
                && reg.is_zero()
            {
                prog.replace_uses_with(id, Value::U32(0));
            }
        }
        GetPred => {
            if let Value::Pred(pred) = prog.arg(id, 0)
                && pred.is_true()
            {
                prog.replace_uses_with(id, Value::U1(true));
            }
        }
        IAdd32 => fold_add(prog, id, false),
        IAdd64 => fold_add(prog, id, true),
        ISub32 => fold_isub32(prog, id),
        SMax32 => fold_smax32(prog, memo, id),
        IMul32 => fold_imul32(prog, id),
        ShiftRightArithmetic32 => {
            fold_immediates(prog, id, |a| {
                let [base, shift] = u32s(a)?;
                (shift < 32).then(|| Value::U32(((base as i32) >> shift) as u32))
            });
        }
        BitCastF32U32 => fold_bitcast(prog, id, BitCastU32F32),
        BitCastU32F32 => fold_bitcast(prog, id, BitCastF32U32),
        PackHalf2x16 => fold_inverse(prog, id, UnpackHalf2x16),
        PackFloat2x16 => fold_inverse(prog, id, UnpackFloat2x16),
        UnpackFloat2x16 => fold_inverse(prog, id, PackFloat2x16),
        SelectU1 | SelectU8 | SelectU16 | SelectU32 | SelectU64 | SelectF16 | SelectF32
        | SelectF64 => {
            if let Value::U1(cond) = prog.arg(id, 0) {
                let chosen = prog.arg(id, if cond { 1 } else { 2 });
                prog.replace_uses_with(id, chosen);
            }
        }
        FPNeg32 => {
            fold_immediates(prog, id, |a| Some(Value::F32(-a[0].imm_f32()?)));
        }
        FPAdd32 => fold_fp_add32(prog, id),
        FPMul32 => fold_fp_mul32(prog, id),
        LogicalAnd => fold_logical(prog, id, false),
        LogicalOr => fold_logical(prog, id, true),
        LogicalNot => fold_logical_not(prog, id),
        BitFieldUExtract => {
            fold_immediates(prog, id, |a| {
                let [base, shift, count] = u32s(a)?;
                (shift + count <= 32).then(|| {
                    Value::U32(((u64::from(base) >> shift) & ((1u64 << count) - 1)) as u32)
                })
            });
        }
        BitFieldSExtract => {
            fold_immediates(prog, id, |a| {
                let [base, shift, count] = u32s(a)?;
                let left = 32u32.checked_sub(shift.checked_add(count)?)?;
                let right = 32u32.checked_sub(count)?;
                (left < 32 && right < 32)
                    .then(|| Value::U32((((base as i32) << left) >> right) as u32))
            });
        }
        BitFieldInsert => {
            fold_immediates(prog, id, |a| {
                let [base, insert, offset, bits] = u32s(a)?;
                (bits < 32 && offset < 32).then(|| {
                    let mask = !(!0u32 << bits) << offset;
                    Value::U32((base & !mask) | ((insert << offset) & mask))
                })
            });
        }
        CompositeExtractU32x2 => fold_extract(
            prog,
            memo,
            id,
            CompositeConstructU32x2,
            CompositeInsertU32x2,
        ),
        CompositeExtractU32x3 => fold_extract(
            prog,
            memo,
            id,
            CompositeConstructU32x3,
            CompositeInsertU32x3,
        ),
        CompositeExtractU32x4 => fold_extract(
            prog,
            memo,
            id,
            CompositeConstructU32x4,
            CompositeInsertU32x4,
        ),
        CompositeExtractF32x2 => fold_extract(
            prog,
            memo,
            id,
            CompositeConstructF32x2,
            CompositeInsertF32x2,
        ),
        CompositeExtractF32x3 => fold_extract(
            prog,
            memo,
            id,
            CompositeConstructF32x3,
            CompositeInsertF32x3,
        ),
        CompositeExtractF32x4 => fold_extract(
            prog,
            memo,
            id,
            CompositeConstructF32x4,
            CompositeInsertF32x4,
        ),
        CompositeExtractF16x2 => fold_extract(
            prog,
            memo,
            id,
            CompositeConstructF16x2,
            CompositeInsertF16x2,
        ),
        CompositeExtractF16x3 => fold_extract(
            prog,
            memo,
            id,
            CompositeConstructF16x3,
            CompositeInsertF16x3,
        ),
        CompositeExtractF16x4 => fold_extract(
            prog,
            memo,
            id,
            CompositeConstructF16x4,
            CompositeInsertF16x4,
        ),
        FSwizzleAdd => fold_fswizzle_add(prog, id),
        GetCbufU32 | GetCbufF32 => fold_cbuf(env, prog, id),
        BindlessImageSampleImplicitLod | BoundImageSampleImplicitLod | ImageSampleImplicitLod => {
            fold_image_sample_implicit_lod(prog, id)
        }
        _ => {
            if let Some(f) = u32_binop(op) {
                fold_u32_pair(prog, id, f);
            }
        }
    }
}

fn u32_binop(op: Opcode) -> Option<fn(u32, u32) -> Value> {
    use Opcode::*;
    let f: fn(u32, u32) -> Value = match op {
        SLessThan => |a, b| Value::U1((a as i32) < (b as i32)),
        ULessThan => |a, b| Value::U1(a < b),
        SLessThanEqual => |a, b| Value::U1((a as i32) <= (b as i32)),
        ULessThanEqual => |a, b| Value::U1(a <= b),
        SGreaterThan => |a, b| Value::U1((a as i32) > (b as i32)),
        UGreaterThan => |a, b| Value::U1(a > b),
        SGreaterThanEqual => |a, b| Value::U1((a as i32) >= (b as i32)),
        UGreaterThanEqual => |a, b| Value::U1(a >= b),
        IEqual => |a, b| Value::U1(a == b),
        INotEqual => |a, b| Value::U1(a != b),
        BitwiseAnd32 => |a, b| Value::U32(a & b),
        BitwiseOr32 => |a, b| Value::U32(a | b),
        BitwiseXor32 => |a, b| Value::U32(a ^ b),
        _ => return None,
    };
    Some(f)
}

fn u32s<const N: usize>(args: &[Value]) -> Option<[u32; N]> {
    if args.len() != N {
        return None;
    }
    let mut out = [0; N];
    for (slot, value) in out.iter_mut().zip(args) {
        *slot = value.imm_u32()?;
    }
    Some(out)
}

/// Replaces the instruction with `eval` of its arguments when all are immediates and no
/// pseudo-instruction reads its flags. `eval` returns `None` for results the guest leaves
/// undefined, which stay for the host to compute.
fn fold_immediates(
    prog: &mut Program,
    id: InstId,
    eval: impl FnOnce(&[Value]) -> Option<Value>,
) -> bool {
    let args = prog.inst(id).args();
    if !args.iter().all(|a| a.is_immediate()) || has_pseudo_op(prog, id) {
        return false;
    }
    let Some(result) = eval(args) else {
        return false;
    };
    prog.replace_uses_with(id, result);
    true
}

fn fold_u32_pair(prog: &mut Program, id: InstId, f: impl FnOnce(u32, u32) -> Value) -> bool {
    fold_immediates(prog, id, |a| {
        let [x, y] = u32s(a)?;
        Some(f(x, y))
    })
}

/// Folds two immediates, merges an immediate into a same-op operand that has one, and moves a
/// lone immediate to the right. Returns false when the instruction was replaced.
fn fold_commutative(
    prog: &mut Program,
    id: InstId,
    combine: impl Fn(Value, Value) -> Option<Value>,
) -> bool {
    let op = prog.op(id);
    let (lhs, rhs) = (prog.arg(id, 0), prog.arg(id, 1));
    let (imm, other, imm_is_lhs) = match (lhs.is_immediate(), rhs.is_immediate()) {
        (true, true) => {
            let Some(result) = combine(lhs, rhs) else {
                return true;
            };
            prog.replace_uses_with(id, result);
            return false;
        }
        (true, false) => (lhs, rhs, true),
        (false, true) => (rhs, lhs, false),
        (false, false) => return true,
    };
    if let Some((inner, inner_imm)) = binary(prog, other, op)
        && inner_imm.is_immediate()
        && let Some(combined) = combine(imm, inner_imm)
    {
        prog.set_arg(id, 0, inner);
        prog.set_arg(id, 1, combined);
    } else if imm_is_lhs {
        prog.set_arg(id, 0, rhs);
        prog.set_arg(id, 1, lhs);
    }
    true
}

fn fold_add(prog: &mut Program, id: InstId, wide: bool) {
    if has_pseudo_op(prog, id) {
        return;
    }
    let combine = |a: Value, b: Value| {
        Some(if wide {
            Value::U64(a.imm_u64()?.wrapping_add(b.imm_u64()?))
        } else {
            Value::U32(a.imm_u32()?.wrapping_add(b.imm_u32()?))
        })
    };
    if !fold_commutative(prog, id, combine) {
        return;
    }
    let rhs = prog.arg(id, 1);
    if rhs.is_immediate() && rhs.imm_u64() == Some(0) {
        let lhs = prog.arg(id, 0);
        prog.replace_uses_with(id, lhs);
        return;
    }
    if wide || fold_halves_recombine(prog, id) {
        return;
    }
    if let Some((a, b)) = xmad_multiply(prog, id) {
        replace_with_new(prog, id, |e| e.imul(a, b));
    } else if let Some((a, b, c)) = xmad_multiply_add(prog, id) {
        replace_with_new(prog, id, |e| {
            let product = e.imul(a, b);
            e.iadd(product, c)
        });
    }
}

/// `(BitFieldUExtract(a, 16, 16) << 16) + BitFieldUExtract(a, 0, 16)` is `a`: what an XMAD
/// multiplication by one leaves once its products by zero and one fold.
fn fold_halves_recombine(prog: &mut Program, id: InstId) -> bool {
    for k in 0..2 {
        if let Some(low) = bfe(prog, prog.arg(id, 1 - k), 0, 16)
            && let Some(high) = shl16(prog, prog.arg(id, k))
            && let Some(high_base) = bfe(prog, high, 16, 16)
            && high_base.same(low)
        {
            prog.replace_uses_with(id, low);
            return true;
        }
    }
    false
}

/// `(BitFieldUExtract(a, 16, 16) * b << 16) + BitFieldUExtract(a, 0, 16) * b` is `a * b`.
fn xmad_multiply(prog: &Program, id: InstId) -> Option<(Value, Value)> {
    let (lhs_bfe, factor_b) = binary(prog, shl16(prog, prog.arg(id, 0))?, Opcode::IMul32)?;
    let (rhs_bfe, rhs_factor_b) = binary(prog, prog.arg(id, 1), Opcode::IMul32)?;
    let factor_a = bfe(prog, lhs_bfe, 16, 16)?;
    (factor_b.same(rhs_factor_b) && bfe(prog, rhs_bfe, 0, 16)?.same(factor_a))
        .then_some((factor_a, factor_b))
}

/// The three-XMAD sequence NVIDIA's compiler emits for a 32-bit `a * b + c`:
///
/// ```text
/// %6  = BitFieldUExtract %b, #0, #16      %15 = BitFieldUExtract %b, #0, #16
/// %7  = BitFieldUExtract %a, #16, #16     %16 = BitFieldUExtract %a, #0, #16
/// %8  = IMul32 %6, %7                     %17 = IMul32 %15, %16
/// %10 = BitFieldUExtract %a, #0, #16      %18 = IAdd32 %17, %c
/// %11 = BitFieldInsert %8, %10, #16, #16  %22 = BitFieldUExtract %b, #16, #16
/// %23 = BitFieldUExtract %11, #16, #16    %24 = IMul32 %22, %23
/// %25 = ShiftLeftLogical32 %24, #16       %26 = ShiftLeftLogical32 %11, #16
/// %27 = IAdd32 %26, %18                   %result = IAdd32 %25, %27
/// ```
fn xmad_multiply_add(prog: &Program, id: InstId) -> Option<(Value, Value, Value)> {
    use Opcode::*;
    let (b_high, merged) = binary(prog, shl16(prog, prog.arg(id, 0))?, IMul32)?;
    let insert = producer(prog, bfe(prog, merged, 16, 16)?, BitFieldInsert)?;
    if arg(prog, insert, 2) != Value::U32(16) || arg(prog, insert, 3) != Value::U32(16) {
        return None;
    }
    let (b_low, a_high) = binary(prog, arg(prog, insert, 0), IMul32)?;
    let (shifted, low_sum) = binary(prog, prog.arg(id, 1), IAdd32)?;
    if prog.resolve(shl16(prog, shifted)?).inst() != Some(insert) {
        return None;
    }
    let (low_product, c) = binary(prog, low_sum, IAdd32)?;
    let (b_low2, a_low2) = binary(prog, low_product, IMul32)?;
    let a = [
        bfe(prog, a_high, 16, 16)?,
        bfe(prog, a_low2, 0, 16)?,
        bfe(prog, arg(prog, insert, 1), 0, 16)?,
    ];
    let b = [
        bfe(prog, b_high, 16, 16)?,
        bfe(prog, b_low, 0, 16)?,
        bfe(prog, b_low2, 0, 16)?,
    ];
    (all_equal(prog, &a) && all_equal(prog, &b)).then_some((a[0], b[1], c))
}

/// Values are equal, or read the same constant buffer word.
fn all_equal(prog: &Program, values: &[Value]) -> bool {
    values.windows(2).all(|w| {
        w[0].same(w[1])
            || matches!((w[0].inst(), w[1].inst()), (Some(x), Some(y)) if same_cbuf(prog, x, y))
    })
}

fn same_cbuf(prog: &Program, a: InstId, b: InstId) -> bool {
    prog.op(a) == Opcode::GetCbufU32
        && prog.op(b) == Opcode::GetCbufU32
        && arg(prog, a, 0).same(arg(prog, b, 0))
        && arg(prog, a, 1).same(arg(prog, b, 1))
}

/// Products by zero and one: an XMAD multiplication by a 16-bit immediate leaves them behind,
/// and they hide pointer arithmetic from storage buffer tracking.
fn fold_imul32(prog: &mut Program, id: InstId) {
    if fold_u32_pair(prog, id, |a, b| Value::U32(a.wrapping_mul(b))) {
        return;
    }
    for k in 0..2 {
        let replacement = match prog.arg(id, k) {
            Value::U32(0) => Value::U32(0),
            Value::U32(1) => prog.arg(id, 1 - k),
            _ => continue,
        };
        prog.replace_uses_with(id, replacement);
        return;
    }
}

/// `cbuf - cbuf` and `(x + cbuf) - cbuf` of the same constant buffer word.
fn fold_isub32(prog: &mut Program, id: InstId) {
    if fold_u32_pair(prog, id, |a, b| Value::U32(a.wrapping_sub(b))) {
        return;
    }
    let (Some(a), Some(b)) = (prog.arg(id, 0).inst(), prog.arg(id, 1).inst()) else {
        return;
    };
    if same_cbuf(prog, a, b) {
        prog.replace_uses_with(id, Value::U32(0));
        return;
    }
    if prog.op(a) != Opcode::IAdd32 || prog.op(b) != Opcode::GetCbufU32 {
        return;
    }
    let (mut x, mut y) = (arg(prog, a, 0), arg(prog, a, 1));
    if y.is_immediate() {
        std::mem::swap(&mut x, &mut y);
    }
    if let Some(add_cbuf) = y.inst()
        && same_cbuf(prog, add_cbuf, b)
    {
        prog.replace_uses_with(id, x);
    }
}

/// Folds an F32 operation of two immediates the way the guest computes it. Only
/// round-to-nearest folds (host arithmetic rounds that way); under FTZ and FMZ, denormal inputs
/// and results flush to zero, and under FMZ a zero factor gives zero.
fn fold_f32_pair(prog: &mut Program, id: InstId, is_mul: bool) -> bool {
    let control = FpControl::from_flags(prog.inst(id).flags);
    if !matches!(control.rounding, FpRounding::DontCare | FpRounding::Rn) {
        return false;
    }
    let flush = matches!(control.fmz_mode, FmzMode::Ftz | FmzMode::Fmz);
    let flushed = |x: f32| {
        if flush && x.is_subnormal() {
            0.0f32.copysign(x)
        } else {
            x
        }
    };
    fold_immediates(prog, id, |a| {
        let (x, y) = (flushed(a[0].imm_f32()?), flushed(a[1].imm_f32()?));
        let result = match is_mul {
            true if control.fmz_mode == FmzMode::Fmz && (x == 0.0 || y == 0.0) => 0.0,
            true => x * y,
            false => x + y,
        };
        Some(Value::F32(flushed(result)))
    })
}

fn fold_fp_add32(prog: &mut Program, id: InstId) {
    if fold_f32_pair(prog, id, false) {
        return;
    }
    let control = FpControl::from_flags(prog.inst(id).flags);
    if matches!(control.fmz_mode, FmzMode::Ftz | FmzMode::Fmz) {
        return;
    }
    let neutral = |v: Value| v.is_immediate() && v.imm_f32() == Some(0.0);
    let (lhs, rhs) = (prog.arg(id, 0), prog.arg(id, 1));
    if neutral(lhs) {
        prog.replace_uses_with(id, rhs);
    } else if neutral(rhs) {
        prog.replace_uses_with(id, lhs);
    }
}

/// Besides immediates: `YDirection * DPdyFine(x)` is the derivative itself, and
/// `(x * attr) * FPRecip32(attr)` of the same attribute is `x` (perspective interpolation).
fn fold_fp_mul32(prog: &mut Program, id: InstId) {
    use Opcode::*;
    if fold_f32_pair(prog, id, true) {
        return;
    }
    if FpControl::from_flags(prog.inst(id).flags).no_contraction {
        return;
    }
    let (lhs_value, rhs_value) = (prog.arg(id, 0), prog.arg(id, 1));
    let (Some(lhs), Some(rhs)) = (lhs_value.inst(), rhs_value.inst()) else {
        return;
    };
    match (prog.op(lhs), prog.op(rhs)) {
        (YDirection, DPdyFine) => return prog.replace_uses_with(id, rhs_value),
        (DPdyFine, YDirection) => return prog.replace_uses_with(id, lhs_value),
        (FPMul32, FPRecip32) => {}
        _ => return,
    }
    let attribute = |v: Value| producer(prog, v, GetAttribute).map(|p| arg(prog, p, 0));
    if let Some(a) = attribute(arg(prog, rhs, 0))
        && attribute(arg(prog, lhs, 1)) == Some(a)
    {
        let x = arg(prog, lhs, 0);
        prog.replace_uses_with(id, x);
    }
}

fn fold_logical(prog: &mut Program, id: InstId, is_or: bool) {
    let combine = |a: Value, b: Value| {
        let (a, b) = (a.imm_bool()?, b.imm_bool()?);
        Some(Value::U1(if is_or { a || b } else { a && b }))
    };
    if !fold_commutative(prog, id, combine) {
        return;
    }
    if let Value::U1(rhs) = prog.arg(id, 1) {
        let result = if rhs == is_or {
            Value::U1(rhs)
        } else {
            prog.arg(id, 0)
        };
        prog.replace_uses_with(id, result);
    }
}

fn fold_logical_not(prog: &mut Program, id: InstId) {
    let value = prog.arg(id, 0);
    if let Value::U1(v) = value {
        prog.replace_uses_with(id, Value::U1(!v));
    } else if let Some(inner) = producer(prog, value, Opcode::LogicalNot) {
        let original = arg(prog, inner, 0);
        prog.replace_uses_with(id, original);
    }
}

/// Folds immediates and round trips, turns a float view of a constant buffer word into a typed
/// read, and reads integer system values as integers: `ftou(itof(gl_InstanceID))` returns 0
/// on recent NVIDIA drivers.
fn fold_bitcast(prog: &mut Program, id: InstId, reverse: Opcode) {
    use Opcode::*;
    let op = prog.op(id);
    let value = prog.arg(id, 0);
    if value.is_immediate() {
        let Some(bits) = value.imm_u32() else {
            return;
        };
        let result = if op == BitCastF32U32 {
            Value::F32(f32::from_bits(bits))
        } else {
            Value::U32(bits)
        };
        prog.replace_uses_with(id, result);
        return;
    }
    let Some(source) = value.inst() else {
        return;
    };
    let source_op = prog.op(source);
    if source_op == reverse {
        let original = arg(prog, source, 0);
        prog.replace_uses_with(id, original);
        return;
    }
    let (typed, args) = match (op, source_op) {
        (BitCastF32U32, GetCbufU32) => (GetCbufF32, [arg(prog, source, 0), arg(prog, source, 1)]),
        (BitCastU32F32, GetAttribute) => {
            let attribute = arg(prog, source, 0);
            if !matches!(
                attribute,
                Value::Attribute(
                    Attribute::PRIMITIVE_ID
                        | Attribute::INSTANCE_ID
                        | Attribute::VERTEX_ID
                        | Attribute::BASE_VERTEX
                        | Attribute::BASE_INSTANCE
                        | Attribute::DRAW_ID
                )
            ) {
                return;
            }
            (GetAttributeU32, [attribute, arg(prog, source, 1)])
        }
        _ => return,
    };
    replace_with_new(prog, id, |e| e.inst(typed, &args));
}

fn fold_inverse(prog: &mut Program, id: InstId, reverse: Opcode) {
    if let Some(inner) = producer(prog, prog.arg(id, 0), reverse) {
        let original = arg(prog, inner, 0);
        prog.replace_uses_with(id, original);
    }
}

/// Follows a chain of inserts at other indices back to the insert or construct that wrote
/// `index`.
fn fold_extract(
    prog: &mut Program,
    memo: &mut Memo,
    id: InstId,
    construct: Opcode,
    insert: Opcode,
) {
    let Value::U32(index) = prog.arg(id, 1) else {
        return;
    };
    let mut path = Vec::new();
    let mut cur = prog.arg(id, 0);
    let result = loop {
        let Some(p) = prog.resolve(cur).inst() else {
            break None;
        };
        if let Some(&known) = memo.extracts.get(&(p, index)) {
            break known;
        }
        let op = prog.op(p);
        if op == construct {
            break ((index as usize) < prog.inst(p).num_args())
                .then(|| arg(prog, p, index as usize));
        }
        if op != insert {
            break None;
        }
        match arg(prog, p, 2) {
            Value::U32(at) if at == index => break Some(arg(prog, p, 1)),
            Value::U32(_) => {}
            _ => break None,
        }
        path.push(p);
        cur = arg(prog, p, 0);
    };
    for p in path {
        memo.extracts.insert((p, index), result);
    }
    if let Some(value) = result {
        prog.replace_uses_with(id, value);
    }
}

fn through_cast(prog: &Program, value: Value, cast: Opcode) -> Value {
    let value = prog.resolve(value);
    match producer(prog, value, cast) {
        Some(p) => arg(prog, p, 0),
        None => value,
    }
}

/// Lane XOR mask of a quad butterfly shuffle, 0 when `id` is not one.
fn quad_butterfly_mask(prog: &Program, id: InstId) -> u32 {
    match prog.op(id) {
        Opcode::QuadSwap => arg(prog, id, 1).imm_u32().map_or(0, |d| d.wrapping_add(1)),
        Opcode::ShuffleButterfly => match [1, 2, 3].map(|i| arg(prog, id, i).imm_u32()) {
            [Some(index), Some(3), Some(28)] => index,
            _ => 0,
        },
        _ => 0,
    }
}

/// `FSwizzleAdd(shuffle_xor(x, 1), x, 0x99)` is `DPdxFine(x)`; with lane mask 2 and swizzle
/// 0xa5 it is `DPdyFine(x)`.
fn fold_fswizzle_add(prog: &mut Program, id: InstId) {
    let Value::U32(swizzle) = prog.arg(id, 2) else {
        return;
    };
    let shuffled = through_cast(prog, prog.arg(id, 0), Opcode::BitCastF32U32);
    let other = through_cast(prog, prog.arg(id, 1), Opcode::BitCastF32U32);
    let Some(shuffle) = shuffled.inst() else {
        return;
    };
    let derivative = match (swizzle, quad_butterfly_mask(prog, shuffle)) {
        (0x99, 1) => Opcode::DPdxFine,
        (0xa5, 2) => Opcode::DPdyFine,
        _ => return,
    };
    let source = through_cast(prog, arg(prog, shuffle, 0), Opcode::BitCastU32F32);
    let same_bits = other.imm_u32().is_some() && other.imm_u32() == source.imm_u32();
    if !other.same(source) && !same_bits {
        return;
    }
    let x = prog.arg(id, 1);
    replace_with_new(prog, id, |e| e.inst(derivative, &[x]));
}

/// The float a quad broadcast or a lane 3 shuffle reads.
fn shuffled_source(prog: &Program, input: Value) -> Option<Value> {
    let shuffle = through_cast(prog, input, Opcode::BitCastF32U32).inst()?;
    match prog.op(shuffle) {
        Opcode::QuadBroadcast if arg(prog, shuffle, 1).is_immediate() => {}
        Opcode::ShuffleIndex => {
            let [index, clamp, segmentation] = [1, 2, 3].map(|i| arg(prog, shuffle, i));
            if !(index.is_immediate() && clamp.is_immediate() && segmentation.is_immediate()) {
                return None;
            }
            if index.imm_u32() != Some(3) && clamp.imm_u32() != Some(3) {
                return None;
            }
        }
        _ => return None,
    }
    Some(through_cast(
        prog,
        arg(prog, shuffle, 0),
        Opcode::BitCastU32F32,
    ))
}

fn swizzle_lanes(value: Value) -> Option<[u32; 4]> {
    let Value::U32(bits) = value else {
        return None;
    };
    Some([0, 1, 2, 3].map(|i| (bits >> (i * 2)) & 3))
}

/// Recovers `[coordinate, ddx, ddy]` from the nested quad swizzles NVN emits for one
/// coordinate of an implicit-LOD 3D sample.
fn gradient_3d_derivatives(prog: &Program, coord: Value) -> Option<[Value; 3]> {
    let outer = producer(prog, coord, Opcode::FSwizzleAdd)?;
    let (value_1, value_2) = (arg(prog, outer, 0), arg(prog, outer, 1));
    let lanes_a = swizzle_lanes(arg(prog, outer, 2))?;
    if value_1.is_immediate() || value_2.is_immediate() {
        return None;
    }
    let mut temporaries = [Value::Void; 3];
    let mut lanes_b = [0; 4];
    let mut resolve_pending = |v: Value, temporaries: &mut [Value; 3]| -> bool {
        let Some(inner) = producer(prog, v, Opcode::FSwizzleAdd) else {
            return false;
        };
        for (slot, index) in [(1, 0), (2, 1)] {
            match shuffled_source(prog, arg(prog, inner, index)) {
                Some(source) => temporaries[slot] = source,
                None => return false,
            }
        }
        match swizzle_lanes(arg(prog, inner, 2)) {
            Some(lanes) => {
                lanes_b = lanes;
                true
            }
            None => false,
        }
    };
    let mut found = false;
    let mut coordinate_index = 0;
    for (pending, coordinate, index) in [(value_1, value_2, 0), (value_2, value_1, 2)] {
        if resolve_pending(pending, &mut temporaries) {
            found = match shuffled_source(prog, coordinate) {
                Some(source) => {
                    temporaries[0] = source;
                    true
                }
                None => false,
            };
            coordinate_index = index;
        }
    }
    if !found {
        return None;
    }
    let mut zero_a = 0u32;
    let mut zero_b = 0u32;
    for i in 0..4 {
        if lanes_a[i] == 2 || lanes_b[i] == 2 {
            return None;
        }
        zero_a |= u32::from(lanes_a[i] == 3) << i;
        zero_b |= u32::from(lanes_b[i] == 3) << i;
    }
    const DDX: u32 = 0b1010;
    const DDX_INV: u32 = !DDX & 0b1111;
    if zero_a.count_ones() != 2 || zero_b.count_ones() != 2 || zero_a == zero_b {
        return None;
    }
    let is_ddx = |mask: u32| mask == DDX || mask == DDX_INV;
    let mut results = [temporaries[coordinate_index], Value::Void, Value::Void];
    if coordinate_index == 0 {
        if is_ddx(zero_b) {
            results[1] = temporaries[1];
            results[2] = temporaries[2];
        } else {
            results[2] = temporaries[1];
            results[1] = temporaries[2];
        }
    } else {
        for (value, mask) in [(temporaries[1], zero_b), (temporaries[0], zero_a)] {
            results[if is_ddx(mask) { 1 } else { 2 }] = value;
        }
    }
    Some(results)
}

/// An implicit-LOD 3D sample whose coordinates NVN built from quad swizzles becomes an explicit
/// gradient sample, with the third derivative pair in the offset slot (`num_derivatives` 3).
fn fold_image_sample_implicit_lod(prog: &mut Program, id: InstId) {
    let mut info = TextureInstInfo::from_flags(prog.inst(id).flags);
    if !info.ndv_is_active || info.ty != TextureType::Color3D {
        return;
    }
    let [handle, coords, bias_lc, offset] = [0, 1, 2, 3].map(|i| prog.arg(id, i));
    if matches!(offset, Value::Inst(_)) {
        return;
    }
    let Some(coords) = coords.inst() else {
        return;
    };
    if prog.inst(coords).num_args() < 3 {
        return;
    }
    let mut matrix = [[Value::Void; 3]; 3];
    for (i, row) in matrix.iter_mut().enumerate() {
        let Some(found) = gradient_3d_derivatives(prog, arg(prog, coords, i)) else {
            return;
        };
        *row = found;
    }
    let all_scalars = matrix.iter().flatten().all(|&v| {
        let ty = prog.type_of(v);
        ty == Type::F32 || ty == Type::U32
    });
    if !all_scalars {
        return;
    }
    let lod_clamp = match (info.has_lod_clamp, info.has_bias) {
        (false, _) => Value::Void,
        (true, false) => bias_lc,
        (true, true) => match bias_lc.inst() {
            Some(pair) if prog.op(pair) == Opcode::CompositeConstructF32x2 => arg(prog, pair, 1),
            _ => return,
        },
    };
    info.num_derivatives = 3;
    let op = match prog.op(id) {
        Opcode::ImageSampleImplicitLod => Opcode::ImageGradient,
        _ if handle.is_immediate() => Opcode::BoundImageGradient,
        _ => Opcode::BindlessImageGradient,
    };
    replace_with_new(prog, id, |e| {
        for value in matrix.iter_mut().flatten() {
            if e.ty(*value) == Type::U32 {
                *value = e.bitcast_f32(*value);
            }
        }
        let m = matrix;
        let coords = e.composite_construct(&[m[0][0], m[1][0], m[2][0]]);
        let derivatives = e.composite_construct(&[m[0][1], m[0][2], m[1][1], m[1][2]]);
        let derivatives_z = e.composite_construct(&[m[2][1], m[2][2]]);
        e.inst_flags(
            op,
            info.to_flags(),
            &[handle, coords, derivatives, derivatives_z, lod_clamp],
        )
    });
}

/// Draw parameters that HLE macros moved out of constant buffers become shader inputs, and
/// driver constant buffer reads become immediates on the proprietary driver.
fn fold_cbuf(env: &dyn ShaderEnvironment, prog: &mut Program, id: InstId) {
    let (Value::U32(bank), Value::U32(offset)) = (prog.arg(id, 0), prog.arg(id, 1)) else {
        return;
    };
    let is_u32 = prog.op(id) == Opcode::GetCbufU32;
    if env.has_hle_macro_state()
        && let Some(replacement) = env.replace_const_buffer(bank, offset)
    {
        let attribute = match replacement {
            ReplaceConstant::BaseInstance => Attribute::BASE_INSTANCE,
            ReplaceConstant::BaseVertex => Attribute::BASE_VERTEX,
            ReplaceConstant::DrawId => Attribute::DRAW_ID,
        };
        replace_with_new(prog, id, |e| {
            if is_u32 {
                e.get_attribute_u32(attribute, Value::U32(0))
            } else {
                e.get_attribute(attribute, Value::U32(0))
            }
        });
        return;
    }
    if env.is_proprietary_driver() && bank == 1 && offset < u32::from(u16::MAX) {
        let raw = env.read_cbuf(bank, offset);
        let value = if is_u32 {
            Value::U32(raw)
        } else {
            Value::F32(f32::from_bits(raw))
        };
        prog.replace_uses_with(id, value);
    }
}

fn is_mask_constant(value: Value) -> bool {
    matches!(value, Value::U32(0 | 1 | u32::MAX))
}

/// Guest shaders normalise predicate masks with `max(x, -x)`, which NVIDIA's compiler
/// miscompiles into a device loss. For a select of mask constants it is the select of their
/// absolute values; for any proven mask (-1, 0 or 1) it is `x & 1`.
fn fold_smax32(prog: &mut Program, memo: &mut Memo, id: InstId) {
    for k in 0..2 {
        let value = prog.arg(id, k);
        let Some(negation) = producer(prog, prog.arg(id, 1 - k), Opcode::INeg32) else {
            continue;
        };
        if !arg(prog, negation, 0).same(value) {
            continue;
        }
        if let Some(select) = producer(prog, value, Opcode::SelectU32)
            && is_mask_constant(arg(prog, select, 1))
            && is_mask_constant(arg(prog, select, 2))
        {
            let cond = prog.arg(select, 0);
            let abs = |v: Value| Value::U32(u32::from(v != Value::U32(0)));
            let (t, f) = (abs(arg(prog, select, 1)), abs(arg(prog, select, 2)));
            replace_with_new(prog, id, |e| e.select(cond, t, f));
            return;
        }
        if mask_values(prog, &mut memo.masks, value, 0).0 != MASK_UNKNOWN {
            replace_with_new(prog, id, |e| e.bitwise_and(value, Value::U32(1)));
            return;
        }
    }
}

/// Which of -1, 0 and 1 `value` can take, or `MASK_UNKNOWN` when it is not provably a mask,
/// and whether the depth limit cut the search short. Only complete answers are memoised, so
/// results match an unmemoised depth-limited search.
fn mask_values(
    prog: &Program,
    memo: &mut FxHashMap<InstId, u32>,
    value: Value,
    depth: u32,
) -> (u32, bool) {
    let id = match prog.resolve(value) {
        Value::U32(u32::MAX) => return (MASK_MINUS_ONE, false),
        Value::U32(0) => return (MASK_ZERO, false),
        Value::U32(1) => return (MASK_ONE, false),
        Value::Inst(id) => id,
        _ => return (MASK_UNKNOWN, false),
    };
    if let Some(&known) = memo.get(&id) {
        return (known, false);
    }
    if depth >= MAX_MASK_DEPTH {
        return (MASK_UNKNOWN, true);
    }
    let mut truncated = false;
    let mut sub = |v: Value| {
        let (mask, cut) = mask_values(prog, memo, v, depth + 1);
        truncated |= cut;
        mask
    };
    let result = match prog.op(id) {
        Opcode::SelectU32 => {
            let t = sub(arg(prog, id, 1));
            let f = sub(arg(prog, id, 2));
            if t != MASK_UNKNOWN && f != MASK_UNKNOWN {
                t | f
            } else {
                MASK_UNKNOWN
            }
        }
        Opcode::INeg32 => {
            let m = sub(arg(prog, id, 0));
            (m & MASK_ZERO) | ((m & MASK_MINUS_ONE) << 2) | ((m & MASK_ONE) >> 2)
        }
        Opcode::IAdd32 => {
            let (a, b) = (arg(prog, id, 0), arg(prog, id, 1));
            let operands = if b.is_immediate() {
                Some((a, b))
            } else if a.is_immediate() {
                Some((b, a))
            } else {
                None
            };
            match operands {
                Some((x, step)) => {
                    let m = sub(x);
                    match step {
                        Value::U32(1) if m & MASK_ONE == 0 => m << 1,
                        Value::U32(u32::MAX) if m & MASK_MINUS_ONE == 0 => m >> 1,
                        _ => MASK_UNKNOWN,
                    }
                }
                None => MASK_UNKNOWN,
            }
        }
        Opcode::Phi => {
            let mut acc = 0;
            for i in 0..prog.inst(id).num_args() {
                let m = sub(arg(prog, id, i));
                if m == MASK_UNKNOWN {
                    acc = MASK_UNKNOWN;
                    break;
                }
                acc |= m;
            }
            acc
        }
        _ => MASK_UNKNOWN,
    };
    if !truncated {
        memo.insert(id, result);
    }
    (result, truncated)
}

#[cfg(test)]
#[path = "constant_propagation_tests.rs"]
mod tests;
