//! TENSOR_MUL_ADD, from the software path of the C++ `emit_spirv_tensor.cpp`: HMMA.16816.F16
//! (depth 16) and HMMA.1688.F16 (depth 8) of a 32-lane warp, each lane holding its share of A
//! (four words), B and C (two words each) as packed halves. The lanes trade operands with
//! subgroup shuffles and accumulate in f32, rounding to half after every K=8 group as the
//! hardware does.

use rspirv::spirv::{
    Capability, Decoration, FunctionControl, GlslStd450Op, LoopControl, Scope, StorageClass, Word,
};

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::InstId;

/// The function index of a multiply depth: 0 for K=8, 1 for K=16.
fn variant(depth: u32) -> EmitResult<usize> {
    match depth {
        8 => Ok(0),
        16 => Ok(1),
        _ => Err(EmitError::Unsupported(
            "matrix multiply depth other than 8 or 16",
        )),
    }
}

pub(super) fn tensor_mul_add(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let variant = variant(ctx.flags(inst))?;
    if !ctx.supports_subgroups() || ctx.profile.has_broken_spirv_subgroup_shuffle {
        return Err(EmitError::Unsupported(
            "software HMMA without subgroup shuffles",
        ));
    }
    ctx.add_capability(Capability::GroupNonUniformShuffle);
    ctx.subgroup_invocation_id();
    if ctx.tensor_functions[variant] == 0 {
        ctx.tensor_functions[variant] = ctx.b.id();
    }
    let function = ctx.tensor_functions[variant];
    let a = ctx.def_arg(inst, 0)?;
    let b = ctx.def_arg(inst, 1)?;
    let c = ctx.def_arg(inst, 2)?;
    Ok(ctx.b.function_call(ctx.u32[2], None, function, [a, b, c])?)
}

/// Emits the bodies of the multiply-add functions [`tensor_mul_add`] called.
pub(crate) fn define_functions(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    for variant in 0..2 {
        let function = ctx.tensor_functions[variant];
        if function != 0 {
            define_function(ctx, function, variant == 1)?;
        }
    }
    Ok(())
}

fn imm_and(ctx: &mut EmitContext<'_>, value: Word, mask: u32) -> EmitResult<Word> {
    let mask = ctx.const_u32(mask);
    Ok(ctx.b.bitwise_and(ctx.u32[1], None, value, mask)?)
}

fn add(ctx: &mut EmitContext<'_>, a: Word, b: Word) -> EmitResult<Word> {
    Ok(ctx.b.i_add(ctx.u32[1], None, a, b)?)
}

fn add_imm(ctx: &mut EmitContext<'_>, a: Word, b: u32) -> EmitResult<Word> {
    let b = ctx.const_u32(b);
    add(ctx, a, b)
}

fn unpack(ctx: &mut EmitContext<'_>, packed: Word) -> EmitResult<Word> {
    ctx.glsl_inst(ctx.f32[2], GlslStd450Op::UnpackHalf2x16, &[packed])
}

fn at_least(ctx: &mut EmitContext<'_>, value: Word, bound: u32) -> EmitResult<Word> {
    let bound = ctx.const_u32(bound);
    Ok(ctx.b.u_greater_than_equal(ctx.u1, None, value, bound)?)
}

/// Rounds an f32 to the nearest half, ties to even, keeping it in f32. A round trip through
/// PackHalf2x16 would be the same, but drivers may eliminate it. Normal significands round
/// explicitly; half subnormals lie on the exactly representable 2^-24 grid.
fn round_half(ctx: &mut EmitContext<'_>, value: Word) -> EmitResult<Word> {
    let (u32_t, f32_t) = (ctx.u32[1], ctx.f32[1]);
    let bits = ctx.b.bitcast(u32_t, None, value)?;
    let magnitude = imm_and(ctx, bits, 0x7fff_ffff)?;
    let thirteen = ctx.const_u32(13);
    let shifted = ctx
        .b
        .shift_right_logical(u32_t, None, magnitude, thirteen)?;
    let tie = imm_and(ctx, shifted, 1)?;
    let biased = add_imm(ctx, magnitude, 0xfff)?;
    let biased = add(ctx, biased, tie)?;
    let rounded = imm_and(ctx, biased, !0x1fff)?;
    let overflow = at_least(ctx, rounded, 0x4780_0000)?;
    let infinity = ctx.const_u32(0x7f80_0000);
    let normal = ctx.b.select(u32_t, None, overflow, infinity, rounded)?;
    let magnitude_f = ctx.b.bitcast(f32_t, None, magnitude)?;
    let to_grid = ctx.const_f32(16_777_216.0);
    let scaled = ctx.b.f_mul(f32_t, None, magnitude_f, to_grid)?;
    let scaled = ctx.glsl_inst(f32_t, GlslStd450Op::RoundEven, &[scaled])?;
    let from_grid = ctx.const_f32(1.0 / 16_777_216.0);
    let subnormal = ctx.b.f_mul(f32_t, None, scaled, from_grid)?;
    let subnormal = ctx.b.bitcast(u32_t, None, subnormal)?;
    let is_normal = at_least(ctx, magnitude, 0x3880_0000)?;
    let finite = ctx.b.select(u32_t, None, is_normal, normal, subnormal)?;
    let is_special = at_least(ctx, magnitude, 0x7f80_0000)?;
    let result = ctx.b.select(u32_t, None, is_special, magnitude, finite)?;
    let sign = imm_and(ctx, bits, 0x8000_0000)?;
    let result = ctx.b.bitwise_or(u32_t, None, result, sign)?;
    Ok(ctx.b.bitcast(f32_t, None, result)?)
}

/// Which K=8 group's operand words a group reads: the low ones, or the high ones when the
/// condition holds.
#[derive(Clone, Copy)]
enum Half {
    Low,
    Select(Word),
}

impl Half {
    fn pick(self, ctx: &mut EmitContext<'_>, low: Word, high: Word) -> EmitResult<Word> {
        Ok(match self {
            Half::Low => low,
            Half::Select(upper) => ctx.b.select(ctx.u32[1], None, upper, high, low)?,
        })
    }
}

struct Lanes {
    scope: Word,
    a_base: Word,
    b_base: Word,
}

/// One K=8 group, unrolled so its shuffles overlap: latency-bound small dispatches need that,
/// while unrolling all of K=16 raises register pressure and slows large dispatches.
fn emit_group(
    ctx: &mut EmitContext<'_>,
    lanes: &Lanes,
    inputs: &[Word; 8],
    accumulators: &[Word; 4],
    half: Half,
) -> EmitResult<()> {
    let (u32_t, f32_t) = (ctx.u32[1], ctx.f32[1]);
    for component in 0..4 {
        let b = half.pick(ctx, inputs[4], inputs[5])?;
        let mut bs = [0; 2];
        for (col, slot) in bs.iter_mut().enumerate() {
            let lane = add_imm(ctx, lanes.b_base, component + col as u32 * 4)?;
            let shuffled = ctx
                .b
                .group_non_uniform_shuffle(u32_t, None, lanes.scope, b, lane)?;
            *slot = unpack(ctx, shuffled)?;
        }
        for row in 0..2 {
            let a = half.pick(ctx, inputs[row], inputs[row + 2])?;
            let lane = add_imm(ctx, lanes.a_base, component)?;
            let shuffled = ctx
                .b
                .group_non_uniform_shuffle(u32_t, None, lanes.scope, a, lane)?;
            let a = unpack(ctx, shuffled)?;
            for (col, &b) in bs.iter().enumerate() {
                let accumulator = accumulators[row * 2 + col];
                let mut value = ctx.b.load(f32_t, None, accumulator, None, [])?;
                for element in 0..2 {
                    let a = ctx.b.composite_extract(f32_t, None, a, [element])?;
                    let b = ctx.b.composite_extract(f32_t, None, b, [element])?;
                    value = ctx.glsl_inst(f32_t, GlslStd450Op::Fma, &[a, b, value])?;
                    ctx.decorate(value, Decoration::NoContraction, &[]);
                }
                ctx.b.store(accumulator, value, None, [])?;
            }
        }
    }
    for &accumulator in accumulators {
        let value = ctx.b.load(f32_t, None, accumulator, None, [])?;
        let value = round_half(ctx, value)?;
        ctx.b.store(accumulator, value, None, [])?;
    }
    Ok(())
}

fn define_function(ctx: &mut EmitContext<'_>, function: Word, full_depth: bool) -> EmitResult<()> {
    let (u1, u32_t, u32x2, u32x4) = (ctx.u1, ctx.u32[1], ctx.u32[2], ctx.u32[4]);
    let (f32_t, f32x2) = (ctx.f32[1], ctx.f32[2]);
    let function_type = ctx.b.type_function(u32x2, [u32x4, u32x2, u32x2]);
    ctx.b.begin_function(
        u32x2,
        Some(function),
        FunctionControl::DONT_INLINE,
        function_type,
    )?;
    let matrix_a = ctx.b.function_parameter(u32x4)?;
    let matrix_b = ctx.b.function_parameter(u32x2)?;
    let matrix_c = ctx.b.function_parameter(u32x2)?;
    ctx.b.begin_block(None)?;
    let function_u32 = ctx.type_pointer(StorageClass::Function, u32_t);
    let function_f32 = ctx.type_pointer(StorageClass::Function, f32_t);
    let counter = ctx
        .b
        .variable(function_u32, None, StorageClass::Function, None);
    let accumulators: [Word; 4] = std::array::from_fn(|_| {
        ctx.b
            .variable(function_f32, None, StorageClass::Function, None)
    });
    let mut inputs = [0; 8];
    for (i, input) in inputs[..4].iter_mut().enumerate() {
        *input = ctx.b.composite_extract(u32_t, None, matrix_a, [i as u32])?;
    }
    for i in 0..2 {
        inputs[4 + i] = ctx.b.composite_extract(u32_t, None, matrix_b, [i as u32])?;
        inputs[6 + i] = ctx.b.composite_extract(u32_t, None, matrix_c, [i as u32])?;
    }
    for row in 0..2 {
        let c = unpack(ctx, inputs[6 + row])?;
        for col in 0..2 {
            let value = ctx.b.composite_extract(f32_t, None, c, [col as u32])?;
            ctx.b.store(accumulators[row * 2 + col], value, None, [])?;
        }
    }
    ctx.b.store(counter, ctx.u32_zero_value, None, [])?;
    let lane = ctx
        .b
        .load(u32_t, None, ctx.subgroup_local_invocation_id, None, [])?;
    let warp_base = imm_and(ctx, lane, !31)?;
    let row_base = imm_and(ctx, lane, 28)?;
    let quad = imm_and(ctx, lane, 3)?;
    let three = ctx.const_u32(3);
    let col_base = ctx.b.shift_left_logical(u32_t, None, quad, three)?;
    let lanes = Lanes {
        scope: ctx.const_u32(Scope::Subgroup as u32),
        a_base: add(ctx, warp_base, row_base)?,
        b_base: add(ctx, warp_base, col_base)?,
    };
    if full_depth {
        let header = ctx.b.id();
        let body = ctx.b.id();
        let next = ctx.b.id();
        let end = ctx.b.id();
        ctx.b.branch(header)?;
        ctx.b.begin_block(Some(header))?;
        let group = ctx.b.load(u32_t, None, counter, None, [])?;
        let two = ctx.const_u32(2);
        let more = ctx.b.u_less_than(u1, None, group, two)?;
        ctx.b.loop_merge(end, next, LoopControl::DONT_UNROLL, [])?;
        ctx.b.branch_conditional(more, body, end, [])?;
        ctx.b.begin_block(Some(body))?;
        let one = ctx.const_u32(1);
        let upper = ctx.b.i_equal(u1, None, group, one)?;
        emit_group(ctx, &lanes, &inputs, &accumulators, Half::Select(upper))?;
        ctx.b.branch(next)?;
        ctx.b.begin_block(Some(next))?;
        let incremented = add(ctx, group, one)?;
        ctx.b.store(counter, incremented, None, [])?;
        ctx.b.branch(header)?;
        ctx.b.begin_block(Some(end))?;
    } else {
        emit_group(ctx, &lanes, &inputs, &accumulators, Half::Low)?;
    }
    let mut packed = [0; 2];
    for (row, slot) in packed.iter_mut().enumerate() {
        let low = ctx.b.load(f32_t, None, accumulators[row * 2], None, [])?;
        let high = ctx
            .b
            .load(f32_t, None, accumulators[row * 2 + 1], None, [])?;
        let pair = ctx.b.composite_construct(f32x2, None, [low, high])?;
        *slot = ctx.glsl_inst(u32_t, GlslStd450Op::PackHalf2x16, &[pair])?;
    }
    let result = ctx.b.composite_construct(u32x2, None, packed)?;
    ctx.b.ret_value(result)?;
    ctx.b.end_function()?;
    Ok(())
}
