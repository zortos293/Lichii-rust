//! Context reads and writes: constant buffers, attributes, patches, fragment outputs, system
//! values and local memory, from the C++ `emit_spirv_context_get_set.cpp`.

use rspirv::spirv::{StorageClass, Word};
use smallvec::SmallVec;

use crate::ShaderStage;
use crate::backend::spirv::context::{CbufElement, EmitContext};
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{Attribute, InstId, Patch, Value};
use crate::runtime::AttributeType;

fn survived_ssa(ctx: &EmitContext<'_>, inst: InstId) -> EmitError {
    EmitError::InvalidIr(format!(
        "{:?} survived SSA construction",
        ctx.program.op(inst)
    ))
}

fn attribute_arg(ctx: &EmitContext<'_>, inst: InstId) -> EmitResult<Attribute> {
    match ctx.arg(inst, 0) {
        Value::Attribute(attribute) => Ok(attribute),
        v => Err(EmitError::InvalidIr(format!("{v:?} is not an attribute"))),
    }
}

fn patch_arg(ctx: &EmitContext<'_>, inst: InstId) -> EmitResult<Patch> {
    match ctx.arg(inst, 0) {
        Value::Patch(patch) => Ok(patch),
        v => Err(EmitError::InvalidIr(format!("{v:?} is not a patch"))),
    }
}

fn is_arrayed_input_stage(stage: ShaderStage) -> bool {
    matches!(
        stage,
        ShaderStage::TessellationControl | ShaderStage::TessellationEval | ShaderStage::Geometry
    )
}

/// Access chain into an input, through the vertex index in the stages whose inputs are arrays.
fn attr_pointer(
    ctx: &mut EmitContext<'_>,
    pointer_type: Word,
    vertex: Word,
    base: Word,
    indices: &[Word],
) -> EmitResult<Word> {
    let mut chain = SmallVec::<[Word; 3]>::new();
    if is_arrayed_input_stage(ctx.stage) {
        chain.push(vertex);
    }
    chain.extend_from_slice(indices);
    Ok(ctx.b.access_chain(pointer_type, None, base, chain)?)
}

/// Access chain into an output, through the invocation id in tessellation control programs.
fn output_access_chain(
    ctx: &mut EmitContext<'_>,
    result_type: Word,
    base: Word,
    index: Word,
) -> EmitResult<Word> {
    if ctx.stage == ShaderStage::TessellationControl {
        let invocation = ctx.b.load(ctx.u32[1], None, ctx.invocation_id, None, [])?;
        return Ok(ctx
            .b
            .access_chain(result_type, None, base, [invocation, index])?);
    }
    Ok(ctx.b.access_chain(result_type, None, base, [index])?)
}

/// The pointer an output attribute store writes, and the type to bitcast the F32 value to when
/// the output is an integer. `None` drops the store.
fn output_attr_pointer(
    ctx: &mut EmitContext<'_>,
    attr: Attribute,
) -> EmitResult<Option<(Word, Option<Word>)>> {
    if let Some(index) = attr.generic_index() {
        let element = u32::from(attr.component());
        let info = ctx.output_generics[index as usize][element as usize];
        if info.id == 0 {
            return Ok(None);
        }
        if info.num_components == 1 {
            return Ok(Some((info.id, None)));
        }
        let index = ctx.const_u32(element - info.first_element);
        let pointer = output_access_chain(ctx, ctx.output_f32, info.id, index)?;
        return Ok(Some((pointer, None)));
    }
    let stage_allows_layer =
        ctx.profile.support_viewport_index_layer_non_geometry || ctx.stage == ShaderStage::Geometry;
    match attr {
        Attribute::POINT_SIZE => Ok(Some((ctx.output_point_size, None))),
        Attribute::POSITION_X
        | Attribute::POSITION_Y
        | Attribute::POSITION_Z
        | Attribute::POSITION_W => {
            let element = ctx.const_u32(u32::from(attr.component()));
            let pointer = output_access_chain(ctx, ctx.output_f32, ctx.output_position, element)?;
            Ok(Some((pointer, None)))
        }
        a if (Attribute::CLIP_DISTANCE_0.0..Attribute::CLIP_DISTANCE_0.0 + 8).contains(&a.0) => {
            let index = u32::from(a.0 - Attribute::CLIP_DISTANCE_0.0);
            if index >= ctx.profile.max_user_clip_distances {
                tracing::warn!(
                    "ignoring clip distance store {index} >= {} supported",
                    ctx.profile.max_user_clip_distances
                );
                return Ok(None);
            }
            let index = ctx.const_u32(index);
            let pointer = output_access_chain(ctx, ctx.output_f32, ctx.clip_distances, index)?;
            Ok(Some((pointer, None)))
        }
        Attribute::LAYER if stage_allows_layer => Ok(Some((ctx.layer, Some(ctx.u32[1])))),
        Attribute::LAYER => Ok(None),
        Attribute::VIEWPORT_INDEX => {
            if !ctx.profile.support_multi_viewport {
                tracing::warn!("ignoring viewport index store on a host without multi viewport");
                return Ok(None);
            }
            if !stage_allows_layer {
                return Ok(None);
            }
            Ok(Some((ctx.viewport_index, Some(ctx.u32[1]))))
        }
        Attribute::VIEWPORT_MASK => {
            if !ctx.profile.support_viewport_mask {
                return Ok(None);
            }
            let pointer = ctx.b.access_chain(
                ctx.output_u32,
                None,
                ctx.viewport_mask,
                [ctx.u32_zero_value],
            )?;
            Ok(Some((pointer, Some(ctx.u32[1]))))
        }
        _ => Err(EmitError::InvalidIr(format!("store to attribute {attr}"))),
    }
}

/// Loads `element` from constant buffer `binding` at byte `offset`. Hardware reads the aligned
/// element (an LDC.U32 at offset 6 reads offset 4), so immediate offsets round down.
fn get_cbuf(
    ctx: &mut EmitContext<'_>,
    result_type: Word,
    element: CbufElement,
    element_size: u32,
    binding: Value,
    offset: Value,
    indirect_func: Word,
) -> EmitResult<Word> {
    let u32_t = ctx.u32[1];
    let buffer_offset = if let Some(offset) = offset.imm_u32() {
        ctx.const_u32(offset / element_size)
    } else if element_size > 1 {
        let shift = ctx.const_u32(element_size.trailing_zeros());
        let offset = ctx.def(offset)?;
        ctx.b.shift_right_logical(u32_t, None, offset, shift)?
    } else {
        ctx.def(offset)?
    };
    let Some(binding) = binding.imm_u32() else {
        let binding = ctx.def(binding)?;
        return Ok(ctx.b.function_call(
            result_type,
            None,
            indirect_func,
            [binding, buffer_offset],
        )?);
    };
    let uniform_type = ctx.uniform_type(element);
    let cbuf = ctx.cbuf(binding as usize, element);
    let pointer = ctx.b.access_chain(
        uniform_type,
        None,
        cbuf,
        [ctx.u32_zero_value, buffer_offset],
    )?;
    let value = ctx.b.load(result_type, None, pointer, None, [])?;
    if offset.is_immediate() || !ctx.profile.has_broken_robust {
        return Ok(value);
    }
    let limit = ctx.const_u32(0xffff);
    let in_bounds = ctx
        .b
        .u_less_than_equal(ctx.u1, None, buffer_offset, limit)?;
    let zero = ctx.const_null(result_type);
    Ok(ctx.b.select(result_type, None, in_bounds, value, zero)?)
}

fn get_cbuf_u32_view(ctx: &mut EmitContext<'_>, binding: Value, offset: Value) -> EmitResult<Word> {
    let (ty, func) = (ctx.u32[1], ctx.load_const_func_u32);
    get_cbuf(ctx, ty, CbufElement::U32, 4, binding, offset, func)
}

fn get_cbuf_u32x4_view(
    ctx: &mut EmitContext<'_>,
    binding: Value,
    offset: Value,
) -> EmitResult<Word> {
    let (ty, func) = (ctx.u32[4], ctx.load_const_func_u32x4);
    get_cbuf(ctx, ty, CbufElement::U32x4, 16, binding, offset, func)
}

/// Word `index_offset` past the one `offset` addresses, out of the vec4 holding it.
fn get_cbuf_element(
    ctx: &mut EmitContext<'_>,
    vector: Word,
    offset: Value,
    index_offset: u32,
) -> EmitResult<Word> {
    let u32_t = ctx.u32[1];
    if let Some(offset) = offset.imm_u32() {
        let element = (offset / 4) % 4 + index_offset;
        return Ok(ctx.b.composite_extract(u32_t, None, vector, [element])?);
    }
    let offset = ctx.def(offset)?;
    let two = ctx.const_u32(2);
    let three = ctx.const_u32(3);
    let shift = ctx.b.shift_right_logical(u32_t, None, offset, two)?;
    let mut element = ctx.b.bitwise_and(u32_t, None, shift, three)?;
    if index_offset > 0 {
        let index_offset = ctx.const_u32(index_offset);
        element = ctx.b.i_add(u32_t, None, element, index_offset)?;
    }
    Ok(ctx.b.vector_extract_dynamic(u32_t, None, vector, element)?)
}

/// The 32-bit word holding the narrow value at `offset`.
fn get_cbuf_word(ctx: &mut EmitContext<'_>, binding: Value, offset: Value) -> EmitResult<Word> {
    if ctx.profile.support_descriptor_aliasing {
        return get_cbuf_u32_view(ctx, binding, offset);
    }
    let vector = get_cbuf_u32x4_view(ctx, binding, offset)?;
    get_cbuf_element(ctx, vector, offset, 0)
}

/// `GetCbufU32` on explicit operands, for the context's own helper functions.
pub(crate) fn get_cbuf_u32_at(
    ctx: &mut EmitContext<'_>,
    binding: Value,
    offset: Value,
) -> EmitResult<Word> {
    get_cbuf_word(ctx, binding, offset)
}

/// `GetCbufU32x2` on explicit operands, for the context's own helper functions.
pub(crate) fn get_cbuf_u32x2_at(
    ctx: &mut EmitContext<'_>,
    binding: Value,
    offset: Value,
) -> EmitResult<Word> {
    if ctx.profile.support_descriptor_aliasing {
        let (ty, func) = (ctx.u32[2], ctx.load_const_func_u32x2);
        return get_cbuf(ctx, ty, CbufElement::U32x2, 8, binding, offset, func);
    }
    let vector = get_cbuf_u32x4_view(ctx, binding, offset)?;
    let low = get_cbuf_element(ctx, vector, offset, 0)?;
    let high = get_cbuf_element(ctx, vector, offset, 1)?;
    Ok(ctx.b.composite_construct(ctx.u32[2], None, [low, high])?)
}

#[derive(Clone, Copy)]
enum Narrow {
    U8,
    S8,
    U16,
    S16,
}

fn get_cbuf_narrow(ctx: &mut EmitContext<'_>, inst: InstId, narrow: Narrow) -> EmitResult<Word> {
    let binding = ctx.arg(inst, 0);
    let offset = ctx.arg(inst, 1);
    let profile = ctx.profile;
    let u32_t = ctx.u32[1];
    let (direct, ty, element, size, func, signed) = match narrow {
        Narrow::U8 | Narrow::S8 => (
            profile.support_int8 && profile.support_uniform_and_storage_buffer_8bit,
            if matches!(narrow, Narrow::U8) {
                ctx.u8
            } else {
                ctx.s8
            },
            if matches!(narrow, Narrow::U8) {
                CbufElement::U8
            } else {
                CbufElement::S8
            },
            1,
            ctx.load_const_func_u8,
            matches!(narrow, Narrow::S8),
        ),
        Narrow::U16 | Narrow::S16 => (
            profile.support_int16 && profile.support_uniform_and_storage_buffer_16bit,
            if matches!(narrow, Narrow::U16) {
                ctx.u16
            } else {
                ctx.s16
            },
            if matches!(narrow, Narrow::U16) {
                CbufElement::U16
            } else {
                CbufElement::S16
            },
            2,
            ctx.load_const_func_u16,
            matches!(narrow, Narrow::S16),
        ),
    };
    if profile.support_descriptor_aliasing && direct {
        let load = get_cbuf(ctx, ty, element, size, binding, offset, func)?;
        return Ok(if signed {
            ctx.b.s_convert(u32_t, None, load)?
        } else {
            ctx.b.u_convert(u32_t, None, load)?
        });
    }
    let word = get_cbuf_word(ctx, binding, offset)?;
    let bit_offset = if size == 1 {
        ctx.bit_offset8(offset)?
    } else {
        ctx.bit_offset16(offset)?
    };
    let count = ctx.const_u32(size * 8);
    Ok(if signed {
        ctx.b
            .bit_field_s_extract(u32_t, None, word, bit_offset, count)?
    } else {
        ctx.b
            .bit_field_u_extract(u32_t, None, word, bit_offset, count)?
    })
}

pub(super) fn get_register(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn set_register(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn get_pred(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn set_pred(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn get_goto_variable(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn set_goto_variable(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn get_indirect_branch_variable(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn set_indirect_branch_variable(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<()> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn get_cbuf_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    get_cbuf_narrow(ctx, inst, Narrow::U8)
}

pub(super) fn get_cbuf_s8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    get_cbuf_narrow(ctx, inst, Narrow::S8)
}

pub(super) fn get_cbuf_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    get_cbuf_narrow(ctx, inst, Narrow::U16)
}

pub(super) fn get_cbuf_s16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    get_cbuf_narrow(ctx, inst, Narrow::S16)
}

pub(super) fn get_cbuf_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (binding, offset) = (ctx.arg(inst, 0), ctx.arg(inst, 1));
    get_cbuf_word(ctx, binding, offset)
}

pub(super) fn get_cbuf_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (binding, offset) = (ctx.arg(inst, 0), ctx.arg(inst, 1));
    let f32_t = ctx.f32[1];
    if ctx.profile.support_descriptor_aliasing {
        let func = ctx.load_const_func_f32;
        return get_cbuf(ctx, f32_t, CbufElement::F32, 4, binding, offset, func);
    }
    let vector = get_cbuf_u32x4_view(ctx, binding, offset)?;
    let word = get_cbuf_element(ctx, vector, offset, 0)?;
    Ok(ctx.b.bitcast(f32_t, None, word)?)
}

pub(super) fn get_cbuf_u32x2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (binding, offset) = (ctx.arg(inst, 0), ctx.arg(inst, 1));
    get_cbuf_u32x2_at(ctx, binding, offset)
}

fn load_u32_as_f32(ctx: &mut EmitContext<'_>, variable: Word) -> EmitResult<Word> {
    let value = ctx.b.load(ctx.u32[1], None, variable, None, [])?;
    Ok(ctx.b.bitcast(ctx.f32[1], None, value)?)
}

fn load_instance_id(ctx: &mut EmitContext<'_>) -> EmitResult<Word> {
    let u32_t = ctx.u32[1];
    if ctx.profile.support_vertex_instance_id {
        return Ok(ctx.b.load(u32_t, None, ctx.instance_id, None, [])?);
    }
    let index = ctx.b.load(u32_t, None, ctx.instance_index, None, [])?;
    let base = ctx.b.load(u32_t, None, ctx.base_instance, None, [])?;
    Ok(ctx.b.i_sub(u32_t, None, index, base)?)
}

fn load_vertex_id(ctx: &mut EmitContext<'_>) -> EmitResult<Word> {
    let variable = if ctx.profile.support_vertex_instance_id {
        ctx.vertex_id
    } else {
        ctx.vertex_index
    };
    Ok(ctx.b.load(ctx.u32[1], None, variable, None, [])?)
}

pub(super) fn get_attribute(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let attr = attribute_arg(ctx, inst)?;
    let vertex = ctx.def_arg(inst, 1)?;
    let f32_t = ctx.f32[1];
    let element = u32::from(attr.component());
    if let Some(index) = attr.generic_index() {
        let generic = ctx.input_generics[index as usize];
        if generic.id == 0 {
            return Ok(ctx.const_f32(if element == 3 { 1.0 } else { 0.0 }));
        }
        let element = ctx.const_u32(element);
        let pointer = attr_pointer(ctx, generic.pointer_type, vertex, generic.id, &[element])?;
        let value = ctx
            .b
            .load(generic.component_type, None, pointer, None, [])?;
        return ctx.input_to_f32(generic.load_op, value);
    }
    match attr {
        Attribute::PRIMITIVE_ID => load_u32_as_f32(ctx, ctx.primitive_id),
        Attribute::LAYER => load_u32_as_f32(ctx, ctx.layer),
        Attribute::POSITION_X
        | Attribute::POSITION_Y
        | Attribute::POSITION_Z
        | Attribute::POSITION_W => {
            let element = ctx.const_u32(element);
            let pointer = if ctx.need_input_position_indirect {
                let zero = ctx.u32_zero_value;
                attr_pointer(
                    ctx,
                    ctx.input_f32,
                    vertex,
                    ctx.input_position,
                    &[zero, element],
                )?
            } else {
                attr_pointer(ctx, ctx.input_f32, vertex, ctx.input_position, &[element])?
            };
            Ok(ctx.b.load(f32_t, None, pointer, None, [])?)
        }
        Attribute::INSTANCE_ID => {
            let id = load_instance_id(ctx)?;
            Ok(ctx.b.bitcast(f32_t, None, id)?)
        }
        Attribute::VERTEX_ID => {
            let id = load_vertex_id(ctx)?;
            Ok(ctx.b.bitcast(f32_t, None, id)?)
        }
        Attribute::BASE_INSTANCE => load_u32_as_f32(ctx, ctx.base_instance),
        Attribute::BASE_VERTEX => load_u32_as_f32(ctx, ctx.base_vertex),
        Attribute::DRAW_ID => load_u32_as_f32(ctx, ctx.draw_index),
        Attribute::FRONT_FACE => {
            let front = ctx.b.load(ctx.u1, None, ctx.front_face, None, [])?;
            let all_ones = ctx.const_u32(u32::MAX);
            let all_ones = ctx.b.bitcast(f32_t, None, all_ones)?;
            Ok(ctx
                .b
                .select(f32_t, None, front, all_ones, ctx.f32_zero_value)?)
        }
        Attribute::POINT_SPRITE_S
        | Attribute::POINT_SPRITE_T
        | Attribute::TESS_EVAL_POINT_U
        | Attribute::TESS_EVAL_POINT_V => {
            let (variable, index) = match attr {
                Attribute::POINT_SPRITE_S => (ctx.point_coord, 0),
                Attribute::POINT_SPRITE_T => (ctx.point_coord, 1),
                Attribute::TESS_EVAL_POINT_U => (ctx.tess_coord, 0),
                _ => (ctx.tess_coord, 1),
            };
            let index = ctx.const_u32(index);
            let pointer = ctx.b.access_chain(ctx.input_f32, None, variable, [index])?;
            Ok(ctx.b.load(f32_t, None, pointer, None, [])?)
        }
        _ => Err(EmitError::InvalidIr(format!("read of attribute {attr}"))),
    }
}

pub(super) fn get_attribute_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let attr = attribute_arg(ctx, inst)?;
    let u32_t = ctx.u32[1];
    let variable = match attr {
        Attribute::PRIMITIVE_ID => ctx.primitive_id,
        Attribute::INSTANCE_ID => return load_instance_id(ctx),
        Attribute::VERTEX_ID => return load_vertex_id(ctx),
        Attribute::BASE_INSTANCE => ctx.base_instance,
        Attribute::BASE_VERTEX => ctx.base_vertex,
        Attribute::DRAW_ID => ctx.draw_index,
        _ => {
            return Err(EmitError::InvalidIr(format!(
                "U32 read of attribute {attr}"
            )));
        }
    };
    Ok(ctx.b.load(u32_t, None, variable, None, [])?)
}

pub(super) fn set_attribute(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let attr = attribute_arg(ctx, inst)?;
    let mut value = ctx.def_arg(inst, 1)?;
    let Some((pointer, ty)) = output_attr_pointer(ctx, attr)? else {
        return Ok(());
    };
    if let Some(ty) = ty {
        value = ctx.b.bitcast(ty, None, value)?;
    }
    Ok(ctx.b.store(pointer, value, None, [])?)
}

pub(super) fn get_attribute_indexed(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let offset = ctx.def_arg(inst, 0)?;
    let (f32_t, func) = (ctx.f32[1], ctx.indexed_load_func);
    if is_arrayed_input_stage(ctx.stage) {
        let vertex = ctx.def_arg(inst, 1)?;
        return Ok(ctx.b.function_call(f32_t, None, func, [offset, vertex])?);
    }
    Ok(ctx.b.function_call(f32_t, None, func, [offset])?)
}

pub(super) fn set_attribute_indexed(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let offset = ctx.def_arg(inst, 0)?;
    let value = ctx.def_arg(inst, 1)?;
    let (void, func) = (ctx.void_id, ctx.indexed_store_func);
    ctx.b.function_call(void, None, func, [offset, value])?;
    Ok(())
}

pub(super) fn get_patch(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let patch = patch_arg(ctx, inst)?;
    if !patch.is_generic() {
        return Err(EmitError::Unsupported("non-generic patch loads"));
    }
    let element = ctx.const_u32(u32::from(patch.generic_element()));
    let pointer_type = if ctx.stage == ShaderStage::TessellationControl {
        ctx.output_f32
    } else {
        ctx.input_f32
    };
    let variable = ctx.patches[patch.generic_index() as usize];
    let pointer = ctx
        .b
        .access_chain(pointer_type, None, variable, [element])?;
    Ok(ctx.b.load(ctx.f32[1], None, pointer, None, [])?)
}

pub(super) fn set_patch(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let patch = patch_arg(ctx, inst)?;
    let value = ctx.def_arg(inst, 1)?;
    let (variable, index) = if patch.is_generic() {
        (
            ctx.patches[patch.generic_index() as usize],
            u32::from(patch.generic_element()),
        )
    } else {
        match patch {
            Patch::LOD_LEFT | Patch::LOD_TOP | Patch::LOD_RIGHT | Patch::LOD_BOTTOM => {
                (ctx.output_tess_level_outer, u32::from(patch.0))
            }
            Patch::LOD_INTERIOR_U => (ctx.output_tess_level_inner, 0),
            Patch::LOD_INTERIOR_V => (ctx.output_tess_level_inner, 1),
            _ => return Err(EmitError::InvalidIr(format!("store to patch {patch}"))),
        }
    };
    let index = ctx.const_u32(index);
    let pointer = ctx
        .b
        .access_chain(ctx.output_f32, None, variable, [index])?;
    Ok(ctx.b.store(pointer, value, None, [])?)
}

pub(super) fn set_frag_color(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let index = ctx.imm_u32(inst, 0)? as usize;
    let component = ctx.imm_u32(inst, 1)?;
    let mut value = ctx.def_arg(inst, 2)?;
    let component = ctx.const_u32(component);
    let pointer_type = match ctx.runtime.color_output_types[index] {
        AttributeType::UnsignedInt => {
            value = ctx.b.bitcast(ctx.u32[1], None, value)?;
            ctx.output_u32
        }
        AttributeType::SignedInt => {
            value = ctx.b.bitcast(ctx.s32[1], None, value)?;
            ctx.type_pointer(StorageClass::Output, ctx.s32[1])
        }
        _ => ctx.output_f32,
    };
    let pointer = ctx
        .b
        .access_chain(pointer_type, None, ctx.frag_color[index], [component])?;
    Ok(ctx.b.store(pointer, value, None, [])?)
}

pub(super) fn set_sample_mask(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let value = ctx.def_arg(inst, 0)?;
    let pointer =
        ctx.b
            .access_chain(ctx.output_u32, None, ctx.sample_mask, [ctx.u32_zero_value])?;
    Ok(ctx.b.store(pointer, value, None, [])?)
}

pub(super) fn set_frag_depth(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let mut value = ctx.def_arg(inst, 0)?;
    if ctx.runtime.convert_depth_mode && !ctx.profile.support_native_ndc {
        let half = ctx.const_f32(0.5);
        value = ctx.glsl_inst(
            ctx.f32[1],
            rspirv::spirv::GlslStd450Op::Fma,
            &[value, half, half],
        )?;
    }
    Ok(ctx.b.store(ctx.frag_depth, value, None, [])?)
}

pub(super) fn get_z_flag(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn get_s_flag(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn get_c_flag(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn get_o_flag(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn set_z_flag(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn set_s_flag(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn set_c_flag(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn set_o_flag(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    Err(survived_ssa(ctx, inst))
}

pub(super) fn workgroup_id(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Ok(ctx.b.load(ctx.u32[3], None, ctx.workgroup_id, None, [])?)
}

pub(super) fn local_invocation_id(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Ok(ctx
        .b
        .load(ctx.u32[3], None, ctx.local_invocation_id, None, [])?)
}

pub(super) fn invocation_id(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Ok(ctx.b.load(ctx.u32[1], None, ctx.invocation_id, None, [])?)
}

/// Input vertices per primitive in bits 16 and up, as the guest's invocation info register.
pub(super) fn invocation_info(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    let u32_t = ctx.u32[1];
    match ctx.stage {
        ShaderStage::TessellationControl | ShaderStage::TessellationEval => {
            let vertices = ctx.b.load(u32_t, None, ctx.patch_vertices_in, None, [])?;
            let sixteen = ctx.const_u32(16);
            Ok(ctx.b.shift_left_logical(u32_t, None, vertices, sixteen)?)
        }
        ShaderStage::Geometry => Ok(ctx.const_u32(ctx.runtime.input_topology.vertices() << 16)),
        _ => {
            tracing::warn!("invocation info read outside tessellation and geometry (stubbed)");
            Ok(ctx.const_u32(0x00ff_0000))
        }
    }
}

pub(super) fn sample_id(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Ok(ctx.b.load(ctx.u32[1], None, ctx.sample_id, None, [])?)
}

pub(super) fn is_helper_invocation(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Ok(ctx
        .b
        .load(ctx.u1, None, ctx.is_helper_invocation, None, [])?)
}

pub(super) fn w_scale_factor_xy(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    tracing::warn!("SR_WSCALEFACTOR_XY read (stubbed)");
    Ok(ctx.const_u32(0x00ff_0000))
}

pub(super) fn w_scale_factor_z(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    tracing::warn!("SR_WSCALEFACTOR_Z read (stubbed)");
    Ok(ctx.const_u32(0x00ff_0000))
}

pub(super) fn y_direction(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Ok(ctx.const_f32(if ctx.runtime.y_negate { -1.0 } else { 1.0 }))
}

pub(super) fn resolution_down_factor(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    let f32_t = ctx.f32[1];
    if ctx.profile.unified_descriptor_binding {
        let pointer_type = ctx.type_pointer(StorageClass::PushConstant, f32_t);
        let index = ctx.const_u32(ctx.rescaling_downfactor_member_index);
        let pointer =
            ctx.b
                .access_chain(pointer_type, None, ctx.rescaling_push_constants, [index])?;
        return Ok(ctx.b.load(f32_t, None, pointer, None, [])?);
    }
    let composite = ctx
        .b
        .load(ctx.f32[4], None, ctx.rescaling_uniform_constant, None, [])?;
    Ok(ctx.b.composite_extract(f32_t, None, composite, [2])?)
}

pub(super) fn render_area(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    if !ctx.profile.unified_descriptor_binding {
        return Err(EmitError::Unsupported(
            "render area without unified descriptor binding",
        ));
    }
    let f32x4 = ctx.f32[4];
    let pointer_type = ctx.type_pointer(StorageClass::PushConstant, f32x4);
    let index = ctx.const_u32(ctx.render_area_member_index);
    let pointer = ctx
        .b
        .access_chain(pointer_type, None, ctx.render_area_push_constant, [index])?;
    Ok(ctx.b.load(f32x4, None, pointer, None, [])?)
}

fn local_pointer(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let word_offset = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.access_chain(
        ctx.private_u32,
        None,
        ctx.local_memory,
        [word_offset, ctx.u32_zero_value],
    )?)
}

pub(super) fn load_local(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let pointer = local_pointer(ctx, inst)?;
    Ok(ctx.b.load(ctx.u32[1], None, pointer, None, [])?)
}

pub(super) fn write_local(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let value = ctx.def_arg(inst, 1)?;
    let pointer = local_pointer(ctx, inst)?;
    Ok(ctx.b.store(pointer, value, None, [])?)
}
