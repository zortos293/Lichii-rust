//! Prologue, epilogue and geometry stream instructions, from the C++ `emit_spirv_special.cpp`.

use rspirv::spirv::{SelectionControl, Word};
use smallvec::SmallVec;

use crate::ShaderStage;
use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Value};
use crate::runtime::{AttributeType, CompareFunction};

/// Output variables and the unscaled values they held before [`scale_noperspective_outputs`].
type ScaledOutputs = SmallVec<[(Word, Word); 8]>;

fn convert_depth_mode(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    let (f32_t, f32x4) = (ctx.f32[1], ctx.f32[4]);
    let half = ctx.const_f32(0.5);
    let position = ctx.b.load(f32x4, None, ctx.output_position, None, [])?;
    let z = ctx.b.composite_extract(f32_t, None, position, [2])?;
    let w = ctx.b.composite_extract(f32_t, None, position, [3])?;
    let sum = ctx.b.f_add(f32_t, None, z, w)?;
    let screen_depth = ctx.b.f_mul(f32_t, None, sum, half)?;
    let vector = ctx
        .b
        .composite_insert(f32x4, None, screen_depth, position, [2])?;
    Ok(ctx.b.store(ctx.output_position, vector, None, [])?)
}

/// Writes the screen-linear generic outputs (`RuntimeInfo::noperspective_outputs`) times
/// Position.w, returning the unscaled values so a geometry program can restore them after
/// EmitVertex.
fn scale_noperspective_outputs(ctx: &mut EmitContext<'_>) -> EmitResult<ScaledOutputs> {
    let mut saved = ScaledOutputs::new();
    let mask = ctx.runtime.noperspective_outputs;
    if mask == 0 || ctx.output_position == 0 {
        return Ok(saved);
    }
    let position = ctx
        .b
        .load(ctx.f32[4], None, ctx.output_position, None, [])?;
    let w = ctx.b.composite_extract(ctx.f32[1], None, position, [3])?;
    for index in 0..ctx.output_generics.len() {
        if mask & (1 << index) == 0 {
            continue;
        }
        let mut last = 0;
        for element in ctx.output_generics[index] {
            if element.id == 0 || element.id == last || element.num_components == 0 {
                continue;
            }
            last = element.id;
            let ty = ctx.f32[element.num_components as usize];
            let value = ctx.b.load(ty, None, element.id, None, [])?;
            let scaled = if element.num_components == 1 {
                ctx.b.f_mul(ty, None, value, w)?
            } else {
                ctx.b.vector_times_scalar(ty, None, value, w)?
            };
            ctx.b.store(element.id, scaled, None, [])?;
            saved.push((element.id, value));
        }
    }
    Ok(saved)
}

fn set_fixed_pipeline_point_size(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    if let Some(size) = ctx.runtime.fixed_state_point_size {
        let size = ctx.const_f32(size);
        ctx.b.store(ctx.output_point_size, size, None, [])?;
    }
    Ok(())
}

fn default_varying(
    ctx: &mut EmitContext<'_>,
    num_components: u32,
    element: u32,
    zero: Word,
    one: Word,
    default_vector: Word,
) -> Word {
    let last = |e: u32| if e == 3 { one } else { zero };
    match num_components {
        1 => last(element),
        2 => {
            let ty = ctx.f32[2];
            ctx.const_composite(ty, &[zero, last(element + 1)])
        }
        3 => {
            let ty = ctx.f32[3];
            ctx.const_composite(ty, &[zero, zero, last(element + 2)])
        }
        _ => default_vector,
    }
}

fn comparison(
    ctx: &mut EmitContext<'_>,
    function: CompareFunction,
    a: Word,
    b: Word,
) -> EmitResult<Word> {
    let u1 = ctx.u1;
    Ok(match function {
        CompareFunction::Never => ctx.false_value,
        CompareFunction::Less => ctx.b.f_ord_less_than(u1, None, a, b)?,
        CompareFunction::Equal => ctx.b.f_ord_equal(u1, None, a, b)?,
        CompareFunction::LessThanEqual => ctx.b.f_ord_less_than_equal(u1, None, a, b)?,
        CompareFunction::Greater => ctx.b.f_ord_greater_than(u1, None, a, b)?,
        CompareFunction::NotEqual => ctx.b.f_ord_not_equal(u1, None, a, b)?,
        CompareFunction::GreaterThanEqual => ctx.b.f_ord_greater_than_equal(u1, None, a, b)?,
        CompareFunction::Always => ctx.true_value,
    })
}

fn is_float_output(ctx: &EmitContext<'_>, index: usize) -> bool {
    !matches!(
        ctx.runtime.color_output_types[index],
        AttributeType::SignedInt | AttributeType::UnsignedInt
    )
}

fn alpha_test(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    let Some(function) = ctx.runtime.alpha_test_func else {
        return Ok(());
    };
    if function == CompareFunction::Always || ctx.frag_color[0] == 0 || !is_float_output(ctx, 0) {
        return Ok(());
    }
    let color = ctx.b.load(ctx.f32[4], None, ctx.frag_color[0], None, [])?;
    let alpha = ctx.b.composite_extract(ctx.f32[1], None, color, [3])?;
    let reference = ctx.const_f32(ctx.runtime.alpha_test_reference);
    let condition = comparison(ctx, function, alpha, reference)?;
    let true_label = ctx.b.id();
    let discard_label = ctx.b.id();
    ctx.b.selection_merge(true_label, SelectionControl::NONE)?;
    ctx.b
        .branch_conditional(condition, true_label, discard_label, [])?;
    ctx.b.begin_block(Some(discard_label))?;
    ctx.b.kill()?;
    ctx.b.begin_block(Some(true_label))?;
    Ok(())
}

pub(super) fn prologue(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    let zero = ctx.f32_zero_value;
    let one = ctx.const_f32(1.0);
    let f32x4 = ctx.f32[4];
    let default_vector = ctx.const_composite(f32x4, &[zero, zero, zero, one]);
    if ctx.stage == ShaderStage::Fragment && ctx.runtime.dual_source_blend {
        for index in 0..2 {
            if ctx.frag_color[index] != 0 && is_float_output(ctx, index) {
                ctx.b
                    .store(ctx.frag_color[index], default_vector, None, [])?;
            }
        }
    }
    if ctx.stage == ShaderStage::VertexB {
        ctx.b.store(ctx.output_position, default_vector, None, [])?;
        for index in 0..ctx.output_generics.len() {
            let generic = ctx.output_generics[index];
            if generic[0].num_components == 0 {
                continue;
            }
            let mut element = 0;
            while element < 4 {
                let info = generic[element as usize];
                let value =
                    default_varying(ctx, info.num_components, element, zero, one, default_vector);
                ctx.b.store(info.id, value, None, [])?;
                element += info.num_components;
            }
        }
        if ctx.clip_distances != 0 {
            for index in 0..ctx.profile.max_user_clip_distances.min(8) {
                let index = ctx.const_u32(index);
                let pointer =
                    ctx.b
                        .access_chain(ctx.output_f32, None, ctx.clip_distances, [index])?;
                ctx.b.store(pointer, zero, None, [])?;
            }
        }
    }
    if matches!(ctx.stage, ShaderStage::VertexB | ShaderStage::Geometry) {
        set_fixed_pipeline_point_size(ctx)?;
    }
    Ok(())
}

pub(super) fn epilogue(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    let convert_depth = ctx.runtime.convert_depth_mode && !ctx.profile.support_native_ndc;
    if ctx.stage == ShaderStage::VertexB && convert_depth {
        convert_depth_mode(ctx)?;
    }
    if matches!(
        ctx.stage,
        ShaderStage::VertexB | ShaderStage::TessellationEval
    ) {
        scale_noperspective_outputs(ctx)?;
    }
    if ctx.stage == ShaderStage::Fragment {
        alpha_test(ctx)?;
    }
    Ok(())
}

/// The stream of EmitVertex and EndPrimitive: `None` for stream 0, the plain instruction
/// that needs no GeometryStreams capability.
fn stream(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Option<Word>> {
    let stream = ctx.arg(inst, 0);
    if stream == Value::U32(0) {
        return Ok(None);
    }
    if !ctx.profile.support_geometry_streams {
        return Err(EmitError::Unsupported("geometry streams"));
    }
    if stream.is_immediate() {
        return ctx.def(stream).map(Some);
    }
    tracing::warn!("geometry stream is not an immediate");
    Ok(Some(ctx.u32_zero_value))
}

pub(super) fn emit_vertex(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    if ctx.runtime.convert_depth_mode && !ctx.profile.support_native_ndc {
        convert_depth_mode(ctx)?;
    }
    let scaled = scale_noperspective_outputs(ctx)?;
    match stream(ctx, inst)? {
        None => ctx.b.emit_vertex()?,
        Some(stream) => ctx.b.emit_stream_vertex(stream)?,
    }
    set_fixed_pipeline_point_size(ctx)?;
    for (id, value) in scaled {
        ctx.b.store(id, value, None, [])?;
    }
    Ok(())
}

pub(super) fn end_primitive(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    match stream(ctx, inst)? {
        None => ctx.b.end_primitive()?,
        Some(stream) => ctx.b.end_stream_primitive(stream)?,
    }
    Ok(())
}
