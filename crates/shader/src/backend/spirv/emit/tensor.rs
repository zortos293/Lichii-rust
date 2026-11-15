//! TENSOR_MUL_ADD, from the software path of the C++ `emit_spirv_tensor.cpp`: HMMA.16816.F16
//! (depth 16) and HMMA.1688.F16 (depth 8) of a 32-lane warp, each lane holding its share of A
//! (four words), B and C (two words each) as packed halves. The lanes trade operands with
//! subgroup shuffles and accumulate in f32, rounding to half after every K=8 group as the
//! hardware does. TENSOR_MUL_ADD_F32 is the .F32 form: C and D are four f32 words each, and
//! nothing rounds to half.

use rspirv::spirv::{
    Capability, Decoration, FunctionControl, GlslStd450Op, LoopControl, MemorySemantics, Scope,
    StorageClass, Word,
};

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::InstId;

/// The function index of a multiply depth: 0 for K=8, 1 for K=16; 2 and 3 with F32
/// accumulators.
fn variant(depth: u32, f32_accumulators: bool) -> EmitResult<usize> {
    let depth = match depth {
        8 => 0,
        16 => 1,
        _ => {
            return Err(EmitError::Unsupported(
                "matrix multiply depth other than 8 or 16",
            ));
        }
    };
    Ok(depth + 2 * usize::from(f32_accumulators))
}

pub(super) fn tensor_mul_add(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    f32_accumulators: bool,
) -> EmitResult<Word> {
    let variant = variant(ctx.flags(inst), f32_accumulators)?;
    if !f32_accumulators && let Some(scratch) = ctx.tensor_scratch_word {
        return cooperative_mul_add(ctx, inst, scratch);
    }
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
    let result_type = ctx.u32[if f32_accumulators { 4 } else { 2 }];
    Ok(ctx
        .b
        .function_call(result_type, None, function, [a, b, c])?)
}

/// The pointer to shared memory word `index`.
fn shared_word(ctx: &mut EmitContext<'_>, index: Word) -> EmitResult<Word> {
    let (pointer_type, array) = (ctx.shared_u32, ctx.shared_memory_u32);
    Ok(if ctx.uses_explicit_workgroup_layout {
        let zero = ctx.u32_zero_value;
        ctx.b
            .access_chain(pointer_type, None, array, [zero, index])?
    } else {
        ctx.b.access_chain(pointer_type, None, array, [index])?
    })
}

fn subgroup_barrier(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    let scope = ctx.const_u32(Scope::Subgroup as u32);
    let semantics =
        ctx.barrier_semantics(MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY);
    let semantics = ctx.const_u32(semantics);
    Ok(ctx.b.control_barrier(scope, scope, semantics)?)
}

/// HMMA.F16 (K=16, or K=8 with zero upper halves) of a 32-lane warp as one cooperative-matrix
/// multiply-add on 16x16x16 fp16 matrices, for hosts without a 16x8x16 shape (AMD RDNA):
/// each lane stores its fragments to row-major A, column-major B (columns 8-15 zero) and
/// row-major C tiles in this subgroup's shared memory scratch, the subgroup multiplies, stores D
/// over C, and each lane reads its two D words back from where it wrote C.
fn cooperative_mul_add(ctx: &mut EmitContext<'_>, inst: InstId, scratch: u32) -> EmitResult<Word> {
    let types = ctx.cooperative_hmma();
    let a = ctx.def_arg(inst, 0)?;
    let b = ctx.def_arg(inst, 1)?;
    let c = ctx.def_arg(inst, 2)?;
    let (u32_t, u32x2) = (ctx.u32[1], ctx.u32[2]);
    let local = ctx.subgroup_invocation_id();
    let local = ctx.b.load(u32_t, None, local, None, [])?;
    let lane = imm_and(ctx, local, 31)?;
    let subgroup = ctx.b.load(u32_t, None, types.subgroup_id, None, [])?;
    let tile_words = ctx.const_u32(super::super::context::COOPERATIVE_HMMA_TILE_WORDS);
    let offset = ctx.b.i_mul(u32_t, None, subgroup, tile_words)?;
    let tile_a = add_imm(ctx, offset, scratch)?;
    let tile_b = add_imm(ctx, tile_a, 128)?;
    let tile_c = add_imm(ctx, tile_a, 256)?;
    let two = ctx.const_u32(2);
    let three = ctx.const_u32(3);
    let g = ctx.b.shift_right_logical(u32_t, None, lane, two)?;
    let t = imm_and(ctx, lane, 3)?;
    // Word of (row g, columns 2t and 2t+1) in a 16x16 fp16 tile of 8 words a row.
    let g8 = ctx.b.shift_left_logical(u32_t, None, g, three)?;
    let row_lo = add(ctx, g8, t)?;
    let row_hi = add_imm(ctx, row_lo, 64)?;
    let row_lo_k8 = add_imm(ctx, row_lo, 4)?;
    let row_hi_k8 = add_imm(ctx, row_hi, 4)?;
    let store = |ctx: &mut EmitContext<'_>, tile: Word, word: Word, value: Word| {
        let index = add(ctx, tile, word)?;
        let pointer = shared_word(ctx, index)?;
        ctx.store_shared(pointer, value)
    };
    let word = |ctx: &mut EmitContext<'_>, vector: Word, index: u32| {
        ctx.b.composite_extract(u32_t, None, vector, [index])
    };
    let a0 = word(ctx, a, 0)?;
    let a1 = word(ctx, a, 1)?;
    let a2 = word(ctx, a, 2)?;
    let a3 = word(ctx, a, 3)?;
    store(ctx, tile_a, row_lo, a0)?;
    store(ctx, tile_a, row_hi, a1)?;
    store(ctx, tile_a, row_lo_k8, a2)?;
    store(ctx, tile_a, row_hi_k8, a3)?;
    // B column g holds k pairs t (b0) and t+4 (b1); columns 8-15 (words 64-127) are zero.
    let b0 = word(ctx, b, 0)?;
    let b1 = word(ctx, b, 1)?;
    store(ctx, tile_b, row_lo, b0)?;
    store(ctx, tile_b, row_lo_k8, b1)?;
    let zero = ctx.u32_zero_value;
    let lane_64 = add_imm(ctx, lane, 64)?;
    let lane_96 = add_imm(ctx, lane, 96)?;
    store(ctx, tile_b, lane_64, zero)?;
    store(ctx, tile_b, lane_96, zero)?;
    let c0 = word(ctx, c, 0)?;
    let c1 = word(ctx, c, 1)?;
    store(ctx, tile_c, row_lo, c0)?;
    store(ctx, tile_c, row_hi, c1)?;
    subgroup_barrier(ctx)?;
    let stride = ctx.const_u32(8);
    let row_major = ctx.const_u32(0);
    let column_major = ctx.const_u32(1);
    let pointer_a = shared_word(ctx, tile_a)?;
    let pointer_b = shared_word(ctx, tile_b)?;
    let pointer_c = shared_word(ctx, tile_c)?;
    let (access, scope) = ctx.shared_access(false);
    let matrix_a = ctx.b.cooperative_matrix_load_khr(
        types.matrix_a,
        None,
        pointer_a,
        row_major,
        Some(stride),
        access,
        scope,
    )?;
    let (access, scope) = ctx.shared_access(false);
    let matrix_b = ctx.b.cooperative_matrix_load_khr(
        types.matrix_b,
        None,
        pointer_b,
        column_major,
        Some(stride),
        access,
        scope,
    )?;
    let (access, scope) = ctx.shared_access(false);
    let matrix_c = ctx.b.cooperative_matrix_load_khr(
        types.matrix_c,
        None,
        pointer_c,
        row_major,
        Some(stride),
        access,
        scope,
    )?;
    let matrix_d = ctx.b.cooperative_matrix_mul_add_khr(
        types.matrix_c,
        None,
        matrix_a,
        matrix_b,
        matrix_c,
        None,
    )?;
    let (access, scope) = ctx.shared_access(true);
    ctx.b.cooperative_matrix_store_khr(
        pointer_c,
        matrix_d,
        row_major,
        Some(stride),
        access,
        scope,
    )?;
    subgroup_barrier(ctx)?;
    let index_lo = add(ctx, tile_c, row_lo)?;
    let pointer_lo = shared_word(ctx, index_lo)?;
    let d0 = ctx.load_shared(u32_t, pointer_lo)?;
    let index_hi = add(ctx, tile_c, row_hi)?;
    let pointer_hi = shared_word(ctx, index_hi)?;
    let d1 = ctx.load_shared(u32_t, pointer_hi)?;
    Ok(ctx.b.composite_construct(u32x2, None, [d0, d1])?)
}

/// Emits the bodies of the multiply-add functions [`tensor_mul_add`] called.
pub(crate) fn define_functions(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    for variant in 0..4 {
        let function = ctx.tensor_functions[variant];
        if function != 0 {
            define_function(ctx, function, variant % 2 == 1, variant >= 2)?;
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
    inputs: &[Word; 6],
    accumulators: &[Word; 4],
    half: Half,
    round: bool,
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
    if !round {
        return Ok(());
    }
    for &accumulator in accumulators {
        let value = ctx.b.load(f32_t, None, accumulator, None, [])?;
        let value = round_half(ctx, value)?;
        ctx.b.store(accumulator, value, None, [])?;
    }
    Ok(())
}

fn define_function(
    ctx: &mut EmitContext<'_>,
    function: Word,
    full_depth: bool,
    f32_accumulators: bool,
) -> EmitResult<()> {
    let (u1, u32_t, u32x2, u32x4) = (ctx.u1, ctx.u32[1], ctx.u32[2], ctx.u32[4]);
    let (f32_t, f32x2) = (ctx.f32[1], ctx.f32[2]);
    let accumulator_type = if f32_accumulators { u32x4 } else { u32x2 };
    let function_type = ctx
        .b
        .type_function(accumulator_type, [u32x4, u32x2, accumulator_type]);
    ctx.b.begin_function(
        accumulator_type,
        Some(function),
        FunctionControl::DONT_INLINE,
        function_type,
    )?;
    let matrix_a = ctx.b.function_parameter(u32x4)?;
    let matrix_b = ctx.b.function_parameter(u32x2)?;
    let matrix_c = ctx.b.function_parameter(accumulator_type)?;
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
    let mut inputs = [0; 6];
    for (i, input) in inputs[..4].iter_mut().enumerate() {
        *input = ctx.b.composite_extract(u32_t, None, matrix_a, [i as u32])?;
    }
    for i in 0..2 {
        inputs[4 + i] = ctx.b.composite_extract(u32_t, None, matrix_b, [i as u32])?;
    }
    for (i, &accumulator) in accumulators.iter().enumerate() {
        let (row, col) = (i as u32 / 2, i as u32 % 2);
        let value = if f32_accumulators {
            let word = ctx.b.composite_extract(u32_t, None, matrix_c, [i as u32])?;
            ctx.b.bitcast(f32_t, None, word)?
        } else {
            let word = ctx.b.composite_extract(u32_t, None, matrix_c, [row])?;
            let pair = unpack(ctx, word)?;
            ctx.b.composite_extract(f32_t, None, pair, [col])?
        };
        ctx.b.store(accumulator, value, None, [])?;
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
        emit_group(
            ctx,
            &lanes,
            &inputs,
            &accumulators,
            Half::Select(upper),
            !f32_accumulators,
        )?;
        ctx.b.branch(next)?;
        ctx.b.begin_block(Some(next))?;
        let incremented = add(ctx, group, one)?;
        ctx.b.store(counter, incremented, None, [])?;
        ctx.b.branch(header)?;
        ctx.b.begin_block(Some(end))?;
    } else {
        emit_group(
            ctx,
            &lanes,
            &inputs,
            &accumulators,
            Half::Low,
            !f32_accumulators,
        )?;
    }
    if f32_accumulators {
        let mut words = [0; 4];
        for (word, &accumulator) in words.iter_mut().zip(&accumulators) {
            let value = ctx.b.load(f32_t, None, accumulator, None, [])?;
            *word = ctx.b.bitcast(u32_t, None, value)?;
        }
        let result = ctx.b.composite_construct(u32x4, None, words)?;
        ctx.b.ret_value(result)?;
        ctx.b.end_function()?;
        return Ok(());
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
