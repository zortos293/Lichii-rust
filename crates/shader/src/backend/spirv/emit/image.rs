//! Texture and image sampling, fetches, reads, writes and queries, from the C++ `emit_spirv_image.cpp`.
//!
//! The texture pass turns every `Bound*` and `Bindless*` opcode into its `Image*` form with the
//! descriptor index in the flags and the array element (or `Void`) in argument 0, so only the
//! `Image*` opcodes reach emission.

use rspirv::dr::{InsertPoint, Instruction, Operand};
use rspirv::spirv::{
    Capability, Decoration, GlslStd450Op, ImageOperands as Mask, Op, StorageClass, Word,
};
use smallvec::SmallVec;

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{ImageFormat, InstId, Opcode, Program, TextureInstInfo, TextureType, Value};
use crate::{Profile, ShaderStage};

#[derive(Clone, Copy)]
enum NonUniformKind {
    SampledImage,
    StorageImage,
    UniformTexelBuffer,
    StorageTexelBuffer,
}

fn non_uniform_supported(profile: &Profile, kind: NonUniformKind) -> bool {
    match kind {
        NonUniformKind::SampledImage => profile.support_sampled_image_array_nonuniform_indexing,
        NonUniformKind::StorageImage => profile.support_storage_image_array_nonuniform_indexing,
        NonUniformKind::UniformTexelBuffer => {
            profile.support_uniform_texel_buffer_array_nonuniform_indexing
        }
        NonUniformKind::StorageTexelBuffer => {
            profile.support_storage_texel_buffer_array_nonuniform_indexing
        }
    }
}

pub(super) fn mark_non_uniform(ctx: &mut EmitContext<'_>, id: Word) {
    if ctx.non_uniform_ids.insert(id) {
        ctx.decorate(id, Decoration::NonUniform, &[]);
    }
}

pub(super) fn unlowered(ctx: &EmitContext<'_>, inst: InstId) -> EmitError {
    EmitError::InvalidIr(format!(
        "{:?} survived the texture pass",
        ctx.program.op(inst)
    ))
}

fn definition<T: Copy>(defs: &[T], info: TextureInstInfo) -> EmitResult<T> {
    defs.get(usize::from(info.descriptor_index))
        .copied()
        .ok_or_else(|| {
            EmitError::InvalidIr(format!(
                "descriptor index {} of a {:?} access has no definition",
                info.descriptor_index, info.ty
            ))
        })
}

/// The pointer to the descriptor `inst` accesses: the variable itself, or the element of a
/// descriptor array its argument 0 selects. The flag says whether the index is non-uniform, in
/// which case the pointer is decorated and every object loaded from it must be too.
fn descriptor_pointer(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    variable: Word,
    element_pointer: Word,
    count: u32,
    kind: NonUniformKind,
) -> EmitResult<(Word, bool)> {
    if count <= 1 {
        return Ok((variable, false));
    }
    let index = ctx.arg(inst, 0);
    let index_id = match index {
        Value::Void => ctx.u32_zero_value,
        index => ctx.def(index)?,
    };
    let non_uniform = matches!(index, Value::Inst(_)) && non_uniform_supported(ctx.profile, kind);
    if non_uniform {
        mark_non_uniform(ctx, index_id);
        match kind {
            NonUniformKind::SampledImage => ctx.uses_nonuniform_sampled_image = true,
            NonUniformKind::StorageImage => ctx.uses_nonuniform_storage_image = true,
            NonUniformKind::UniformTexelBuffer => ctx.uses_nonuniform_uniform_texel_buffer = true,
            NonUniformKind::StorageTexelBuffer => ctx.uses_nonuniform_storage_texel_buffer = true,
        }
    }
    let pointer = ctx
        .b
        .access_chain(element_pointer, None, variable, [index_id])?;
    if non_uniform {
        mark_non_uniform(ctx, pointer);
    }
    Ok((pointer, non_uniform))
}

fn load_descriptor(
    ctx: &mut EmitContext<'_>,
    ty: Word,
    pointer: Word,
    non_uniform: bool,
) -> EmitResult<Word> {
    let object = ctx.b.load(ty, None, pointer, None, [])?;
    if non_uniform {
        mark_non_uniform(ctx, object);
    }
    Ok(object)
}

/// The combined image sampler a sampling instruction reads.
fn texture(ctx: &mut EmitContext<'_>, inst: InstId, info: TextureInstInfo) -> EmitResult<Word> {
    let def = definition(&ctx.textures, info)?;
    let (pointer, non_uniform) = descriptor_pointer(
        ctx,
        inst,
        def.id,
        def.pointer_type,
        def.count,
        NonUniformKind::SampledImage,
    )?;
    load_descriptor(ctx, def.sampled_type, pointer, non_uniform)
}

/// The image of a texture (or uniform texel buffer), for fetches and queries.
fn texture_image(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    info: TextureInstInfo,
) -> EmitResult<Word> {
    if info.ty == TextureType::Buffer {
        let def = definition(&ctx.texture_buffers, info)?;
        let image_type = ctx.image_buffer_type;
        let element_pointer = ctx.type_pointer(StorageClass::UniformConstant, image_type);
        let (pointer, non_uniform) = descriptor_pointer(
            ctx,
            inst,
            def.id,
            element_pointer,
            def.count,
            NonUniformKind::UniformTexelBuffer,
        )?;
        return load_descriptor(ctx, image_type, pointer, non_uniform);
    }
    let def = definition(&ctx.textures, info)?;
    let (pointer, non_uniform) = descriptor_pointer(
        ctx,
        inst,
        def.id,
        def.pointer_type,
        def.count,
        NonUniformKind::SampledImage,
    )?;
    let sampled = load_descriptor(ctx, def.sampled_type, pointer, non_uniform)?;
    let image = ctx.b.image(def.image_type, None, sampled)?;
    if non_uniform {
        mark_non_uniform(ctx, image);
    }
    Ok(image)
}

/// A storage image or storage texel buffer: the pointer to it, its image type, whether that
/// pointer is non-uniform and whether its texels are integers.
pub(super) struct StorageImage {
    pub pointer: Word,
    pub image_type: Word,
    pub non_uniform: bool,
    pub is_integer: bool,
}

pub(super) fn storage_image(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    info: TextureInstInfo,
) -> EmitResult<StorageImage> {
    let (id, image_type, pointer_type, count, is_integer, kind) = if info.ty == TextureType::Buffer
    {
        let def = definition(&ctx.image_buffers, info)?;
        (
            def.id,
            def.image_type,
            def.pointer_type,
            def.count,
            def.is_integer,
            NonUniformKind::StorageTexelBuffer,
        )
    } else {
        let def = definition(&ctx.images, info)?;
        (
            def.id,
            def.image_type,
            def.pointer_type,
            def.count,
            def.is_integer,
            NonUniformKind::StorageImage,
        )
    };
    let (pointer, non_uniform) = descriptor_pointer(ctx, inst, id, pointer_type, count, kind)?;
    Ok(StorageImage {
        pointer,
        image_type,
        non_uniform,
        is_integer,
    })
}

fn load_storage_image(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    info: TextureInstInfo,
) -> EmitResult<(Word, bool)> {
    let image = storage_image(ctx, inst, info)?;
    let object = load_descriptor(ctx, image.image_type, image.pointer, image.non_uniform)?;
    Ok((object, image.is_integer))
}

fn is_texture_integer(ctx: &EmitContext<'_>, info: TextureInstInfo) -> EmitResult<bool> {
    if info.ty == TextureType::Buffer {
        return Ok(false);
    }
    Ok(definition(&ctx.textures, info)?.is_integer)
}

fn is_texture_msaa(ctx: &EmitContext<'_>, info: TextureInstInfo) -> EmitResult<bool> {
    if info.ty == TextureType::Buffer {
        return Ok(false);
    }
    Ok(definition(&ctx.textures, info)?.is_multisample)
}

fn color_type(ctx: &EmitContext<'_>, is_integer: bool) -> Word {
    if is_integer { ctx.u32[4] } else { ctx.f32[4] }
}

/// The IR returns texture colors as F32x4; integer textures are sampled as U32x4 and bitcast.
fn to_f32_color(ctx: &mut EmitContext<'_>, is_integer: bool, color: Word) -> EmitResult<Word> {
    if !is_integer {
        return Ok(color);
    }
    Ok(ctx.b.bitcast(ctx.f32[4], None, color)?)
}

/// The image operands mask and its operand ids, added in mask bit order as SPIR-V requires.
struct ImageOperands {
    mask: Mask,
    ids: SmallVec<[Word; 4]>,
}

impl ImageOperands {
    fn new() -> ImageOperands {
        ImageOperands {
            mask: Mask::NONE,
            ids: SmallVec::new(),
        }
    }

    fn add(&mut self, ctx: &mut EmitContext<'_>, mask: Mask, ids: &[Word]) {
        if mask == Mask::MIN_LOD {
            ctx.add_capability(Capability::MinLod);
        }
        self.mask |= mask;
        self.ids.extend_from_slice(ids);
    }

    /// Bias and lod clamp of an implicit-lod sample. With both, `bias_lc` is an F32x2 of the
    /// bias and the clamp.
    fn implicit_lod(
        ctx: &mut EmitContext<'_>,
        info: TextureInstInfo,
        bias_lc: Option<Word>,
        offset: Value,
    ) -> EmitResult<ImageOperands> {
        let mut operands = ImageOperands::new();
        let (bias, clamp) = bias_and_clamp(ctx, info, bias_lc)?;
        if let Some(bias) = bias {
            operands.add(ctx, Mask::BIAS, &[bias]);
        }
        operands.add_offset(ctx, offset, false)?;
        if let Some(clamp) = clamp {
            operands.add(ctx, Mask::MIN_LOD, &[clamp]);
        }
        Ok(operands)
    }

    /// An implicit lod as explicit gradients: the screen-space derivatives of `coords`
    /// without the array layer, scaled by 2^bias, keeping the lod clamp.
    fn implicit_gradient(
        ctx: &mut EmitContext<'_>,
        info: TextureInstInfo,
        coords: Word,
        bias_lc: Option<Word>,
        offset: Value,
    ) -> EmitResult<ImageOperands> {
        crate::fallback::note("TEX", "implicit_lod_as_explicit_gradients");
        let (bias, clamp) = bias_and_clamp(ctx, info, bias_lc)?;
        let (count, total) = match info.ty {
            TextureType::Color1D | TextureType::Buffer => (1, 1),
            TextureType::ColorArray1D => (1, 2),
            TextureType::Color2D | TextureType::Color2DRect => (2, 2),
            TextureType::ColorArray2D => (2, 3),
            TextureType::Color3D | TextureType::ColorCube => (3, 3),
            TextureType::ColorArrayCube => (3, 4),
        };
        let f32_t = ctx.f32[1];
        let ty = ctx.f32[count];
        let position = match count {
            _ if count == total => coords,
            1 => ctx.b.composite_extract(f32_t, None, coords, [0])?,
            _ => ctx
                .b
                .vector_shuffle(ty, None, coords, coords, 0..count as u32)?,
        };
        let mut dx = ctx.b.d_pdx(ty, None, position)?;
        let mut dy = ctx.b.d_pdy(ty, None, position)?;
        if let Some(bias) = bias {
            let scale = ctx.glsl_inst(f32_t, GlslStd450Op::Exp2, &[bias])?;
            if count == 1 {
                dx = ctx.b.f_mul(ty, None, dx, scale)?;
                dy = ctx.b.f_mul(ty, None, dy, scale)?;
            } else {
                dx = ctx.b.vector_times_scalar(ty, None, dx, scale)?;
                dy = ctx.b.vector_times_scalar(ty, None, dy, scale)?;
            }
        }
        let mut operands = ImageOperands::new();
        operands.add(ctx, Mask::GRAD, &[dx, dy]);
        operands.add_offset(ctx, offset, false)?;
        if let Some(clamp) = clamp {
            operands.add(ctx, Mask::MIN_LOD, &[clamp]);
        }
        Ok(operands)
    }

    fn explicit_lod(
        ctx: &mut EmitContext<'_>,
        lod: Word,
        offset: Value,
    ) -> EmitResult<ImageOperands> {
        let mut operands = ImageOperands::new();
        operands.add(ctx, Mask::LOD, &[lod]);
        operands.add_offset(ctx, offset, false)?;
        Ok(operands)
    }

    /// One offset, or the four constant offsets of a depth-plane gather (PTP) packed in two
    /// U32x4 composites. Non-constant PTP offsets only reach here on a sparse gather, which
    /// cannot be split per tap, so they are dropped as in the C++.
    fn gather(
        ctx: &mut EmitContext<'_>,
        offset: Value,
        offset2: Value,
    ) -> EmitResult<ImageOperands> {
        let mut operands = ImageOperands::new();
        if offset2 == Value::Void {
            operands.add_offset(ctx, offset, true)?;
            return Ok(operands);
        }
        let (Value::Inst(first), Value::Inst(second)) = (offset, offset2) else {
            return Err(EmitError::InvalidIr(
                "PTP offsets are not composites".into(),
            ));
        };
        let program = ctx.program;
        if !program.inst(first).are_all_args_immediates()
            || !program.inst(second).are_all_args_immediates()
        {
            tracing::warn!("not all PTP gather offsets are immediate, ignoring them");
            crate::fallback::note("TLD4", "ptp_non_immediate_offsets");
            return Ok(operands);
        }
        if program.op(first) != Opcode::CompositeConstructU32x4
            || program.op(second) != Opcode::CompositeConstructU32x4
        {
            return Err(EmitError::InvalidIr("invalid PTP gather offsets".into()));
        }
        let mut pairs = [0; 4];
        for (tap, pair) in pairs.iter_mut().enumerate() {
            let source = if tap < 2 { first } else { second };
            let base = (tap % 2) * 2;
            let x = ctx.imm_u32(source, base)? as i32;
            let y = ctx.imm_u32(source, base + 1)? as i32;
            *pair = ctx.const_s32_vec(&[x, y]);
        }
        let array = ctx.type_array(ctx.s32[2], 4);
        let offsets = ctx.const_composite(array, &pairs);
        operands.add(ctx, Mask::CONST_OFFSETS, &[offsets]);
        Ok(operands)
    }

    fn fetch(ctx: &mut EmitContext<'_>, lod: Option<Word>, ms: Option<Word>) -> ImageOperands {
        let mut operands = ImageOperands::new();
        if let Some(lod) = lod {
            operands.add(ctx, Mask::LOD, &[lod]);
        }
        if let Some(ms) = ms {
            operands.add(ctx, Mask::SAMPLE, &[ms]);
        }
        operands
    }

    /// Explicit gradients. The IR packs them as (dx0, dy0, dx1, dy1); three-component
    /// gradients take a second vector from the offset slot, holding (dx2, dy2).
    fn gradient(
        ctx: &mut EmitContext<'_>,
        info: TextureInstInfo,
        derivatives: Word,
        offset: Value,
        lod_clamp: Option<Word>,
    ) -> EmitResult<ImageOperands> {
        let f32_t = ctx.f32[1];
        let mut operands = ImageOperands::new();
        let (dx, dy) = match info.num_derivatives {
            0 => return Err(EmitError::InvalidIr("gradient without derivatives".into())),
            1 => (
                ctx.b.composite_extract(f32_t, None, derivatives, [0])?,
                ctx.b.composite_extract(f32_t, None, derivatives, [1])?,
            ),
            3 => {
                let second = ctx.def(offset)?;
                let mut dx = [0; 3];
                let mut dy = [0; 3];
                let sources = [(derivatives, 0), (derivatives, 2), (second, 0)];
                for (i, (source, component)) in sources.into_iter().enumerate() {
                    dx[i] = ctx.b.composite_extract(f32_t, None, source, [component])?;
                    dy[i] = ctx
                        .b
                        .composite_extract(f32_t, None, source, [component + 1])?;
                }
                (
                    ctx.b.composite_construct(ctx.f32[3], None, dx)?,
                    ctx.b.composite_construct(ctx.f32[3], None, dy)?,
                )
            }
            n => {
                let mut dx = SmallVec::<[Word; 3]>::new();
                let mut dy = SmallVec::<[Word; 3]>::new();
                for i in 0..u32::from(n) {
                    dx.push(ctx.b.composite_extract(f32_t, None, derivatives, [i * 2])?);
                    dy.push(
                        ctx.b
                            .composite_extract(f32_t, None, derivatives, [i * 2 + 1])?,
                    );
                }
                let ty = ctx.f32[usize::from(n)];
                (
                    ctx.b.composite_construct(ty, None, dx)?,
                    ctx.b.composite_construct(ty, None, dy)?,
                )
            }
        };
        operands.add(ctx, Mask::GRAD, &[dx, dy]);
        if info.num_derivatives != 3 {
            operands.add_offset(ctx, offset, false)?;
        }
        if info.has_lod_clamp {
            let clamp = lod_clamp.ok_or_else(|| {
                EmitError::InvalidIr("gradient with lod clamp but no clamp operand".into())
            })?;
            operands.add(ctx, Mask::MIN_LOD, &[clamp]);
        }
        Ok(operands)
    }

    /// A constant offset becomes ConstOffset. A runtime one is only legal on gathers in
    /// Vulkan; elsewhere it is dropped, as in the C++.
    fn add_offset(
        &mut self,
        ctx: &mut EmitContext<'_>,
        offset: Value,
        runtime_allowed: bool,
    ) -> EmitResult<()> {
        let id = match offset {
            Value::Void => return Ok(()),
            Value::Inst(id) => id,
            immediate => {
                let value = immediate
                    .imm_u32()
                    .ok_or_else(|| EmitError::InvalidIr(format!("texture offset {immediate:?}")))?;
                let offset = ctx.const_s32(value as i32);
                self.add(ctx, Mask::CONST_OFFSET, &[offset]);
                return Ok(());
            }
        };
        let constant = ctx.program.inst(id).are_all_args_immediates()
            && matches!(
                ctx.program.op(id),
                Opcode::CompositeConstructU32x2
                    | Opcode::CompositeConstructU32x3
                    | Opcode::CompositeConstructU32x4
            );
        if constant {
            let mut values = SmallVec::<[i32; 4]>::new();
            for index in 0..ctx.program.inst(id).num_args() {
                values.push(ctx.imm_u32(id, index)? as i32);
            }
            let offset = ctx.const_s32_vec(&values);
            self.add(ctx, Mask::CONST_OFFSET, &[offset]);
        } else if runtime_allowed {
            let offset = ctx.def(offset)?;
            self.add(ctx, Mask::OFFSET, &[offset]);
        } else {
            crate::fallback::note("TEX", "runtime_texel_offset_non_gather");
        }
        Ok(())
    }
}

/// Emits an image instruction, or its sparse form when the program reads the residency of the
/// result: the sparse struct's code defines the attached `GetSparseFromOp`.
fn emit_image_op(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    info: TextureInstInfo,
    (op, sparse_op): (Op, Op),
    result_type: Word,
    fixed: &[Word],
    operands: &ImageOperands,
) -> EmitResult<Word> {
    let sparse = ctx.program.pseudo_op(inst, Opcode::GetSparseFromOp);
    let (op, ty) = match sparse {
        Some(_) => (sparse_op, ctx.type_sparse_result(result_type)),
        None => (op, result_type),
    };
    let mut words: Vec<Operand> = fixed.iter().map(|&id| Operand::IdRef(id)).collect();
    if !operands.mask.is_empty() {
        words.push(Operand::ImageOperands(operands.mask));
        words.extend(operands.ids.iter().map(|&id| Operand::IdRef(id)));
    }
    let id = ctx.b.id();
    ctx.b.insert_into_block(
        InsertPoint::End,
        Instruction::new(op, Some(ty), Some(id), words),
    )?;
    if info.relaxed_precision {
        ctx.decorate(id, Decoration::RelaxedPrecision, &[]);
    }
    let Some(sparse) = sparse else {
        return Ok(id);
    };
    let code = ctx.b.composite_extract(ctx.u32[1], None, id, [0])?;
    let resident = ctx.b.image_sparse_texels_resident(ctx.u1, None, code)?;
    ctx.set_def(sparse, resident);
    Ok(ctx.b.composite_extract(result_type, None, id, [1])?)
}

fn is_dynamic_ptp(ctx: &EmitContext<'_>, inst: InstId, offset: Value, offset2: Value) -> bool {
    if offset == Value::Void
        || offset2 == Value::Void
        || ctx
            .program
            .pseudo_op(inst, Opcode::GetSparseFromOp)
            .is_some()
    {
        return false;
    }
    let constant = |value: Value| match value {
        Value::Inst(id) => ctx.program.inst(id).are_all_args_immediates(),
        _ => true,
    };
    !constant(offset) || !constant(offset2)
}

/// Per-tap offsets the IR could not fold to constants (ConstOffsets must be constant): one
/// gather per tap with a runtime Offset, keeping the i0,j0 texel (w) of each footprint, which
/// is the texel a per-tap gather returns for that tap.
fn gather_dynamic_ptp(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    info: TextureInstInfo,
    ops: (Op, Op),
    result_type: Word,
    fixed: &[Word],
    offsets: [Value; 2],
) -> EmitResult<Word> {
    crate::fallback::note("TLD4", "dynamic_ptp_per_tap_gather");
    let packed = [ctx.def(offsets[0])?, ctx.def(offsets[1])?];
    let component_type = if result_type == ctx.u32[4] {
        ctx.u32[1]
    } else {
        ctx.f32[1]
    };
    let mut texels = [0; 4];
    for (tap, texel) in texels.iter_mut().enumerate() {
        let vector = packed[tap / 2];
        let first = (tap as u32 % 2) * 2;
        let x = ctx.b.composite_extract(ctx.u32[1], None, vector, [first])?;
        let y = ctx
            .b
            .composite_extract(ctx.u32[1], None, vector, [first + 1])?;
        let x = ctx.b.bitcast(ctx.s32[1], None, x)?;
        let y = ctx.b.bitcast(ctx.s32[1], None, y)?;
        let tap_offset = ctx.b.composite_construct(ctx.s32[2], None, [x, y])?;
        let mut operands = ImageOperands::new();
        operands.add(ctx, Mask::OFFSET, &[tap_offset]);
        let color = emit_image_op(ctx, inst, info, ops, result_type, fixed, &operands)?;
        *texel = ctx.b.composite_extract(component_type, None, color, [3])?;
    }
    Ok(ctx.b.composite_construct(result_type, None, texels)?)
}

/// Nudges gather coordinates by 1/512 of a texel so AMD rounds them like Maxwell. Cube
/// coordinates are directions, so cubes are left alone (the C++ queried a cube's size as
/// three components, which is invalid).
fn gather_subpixel_offset(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    info: TextureInstInfo,
    coords: Word,
) -> EmitResult<Word> {
    let dim = match info.ty {
        TextureType::Color2D | TextureType::Color2DRect => 2,
        TextureType::ColorArray2D => 3,
        _ => {
            crate::fallback::note("TLD4", "gather_subpixel_offset_skipped");
            return Ok(coords);
        }
    };
    let image = texture_image(ctx, inst, info)?;
    let size = ctx
        .b
        .image_query_size_lod(ctx.u32[dim], None, image, ctx.u32_zero_value)?;
    let nudge = ctx.const_f32(1.0 / 512.0);
    let offset = if dim == 2 {
        ctx.const_composite(ctx.f32[2], &[nudge, nudge])
    } else {
        ctx.const_composite(ctx.f32[3], &[nudge, nudge, ctx.f32_zero_value])
    };
    let size = ctx.b.convert_u_to_f(ctx.f32[dim], None, size)?;
    let offset = ctx.b.f_div(ctx.f32[dim], None, offset, size)?;
    Ok(ctx.b.f_add(ctx.f32[dim], None, coords, offset)?)
}

fn add_offset_to_coordinates(
    ctx: &mut EmitContext<'_>,
    info: TextureInstInfo,
    coords: Word,
    offset: Option<Word>,
) -> EmitResult<Word> {
    let Some(offset) = offset else {
        return Ok(coords);
    };
    let zero = ctx.u32_zero_value;
    let (ty, offset) = match info.ty {
        TextureType::Buffer | TextureType::Color1D => (ctx.u32[1], offset),
        TextureType::ColorArray1D => (
            ctx.u32[2],
            ctx.b
                .composite_construct(ctx.u32[2], None, [offset, zero])?,
        ),
        TextureType::Color2D | TextureType::Color2DRect => (ctx.u32[2], offset),
        TextureType::ColorArray2D => {
            let x = ctx.b.composite_extract(ctx.u32[1], None, offset, [0])?;
            let y = ctx.b.composite_extract(ctx.u32[1], None, offset, [1])?;
            (
                ctx.u32[3],
                ctx.b.composite_construct(ctx.u32[3], None, [x, y, zero])?,
            )
        }
        TextureType::Color3D => (ctx.u32[3], offset),
        TextureType::ColorCube | TextureType::ColorArrayCube => {
            crate::fallback::note("TLD", "fetch_offset_cube");
            return Ok(coords);
        }
    };
    Ok(ctx.b.i_add(ty, None, coords, offset)?)
}

/// The bias and lod clamp `bias_lc` holds for `info`: an F32x2 of both, or the one present.
fn bias_and_clamp(
    ctx: &mut EmitContext<'_>,
    info: TextureInstInfo,
    bias_lc: Option<Word>,
) -> EmitResult<(Option<Word>, Option<Word>)> {
    let Some(bias_lc) = bias_lc else {
        if info.has_bias || info.has_lod_clamp {
            return Err(EmitError::InvalidIr(
                "sample with bias or lod clamp but no bias_lc operand".into(),
            ));
        }
        return Ok((None, None));
    };
    if !(info.has_bias && info.has_lod_clamp) {
        return Ok((
            info.has_bias.then_some(bias_lc),
            info.has_lod_clamp.then_some(bias_lc),
        ));
    }
    let f32_t = ctx.f32[1];
    Ok((
        Some(ctx.b.composite_extract(f32_t, None, bias_lc, [0])?),
        Some(ctx.b.composite_extract(f32_t, None, bias_lc, [1])?),
    ))
}

/// Whether a fragment sample of `inst` must use explicit gradients: it reads an element of a
/// descriptor array at an index that may differ within a quad, and the device leaves implicit
/// lods undefined then.
fn needs_explicit_gradients(
    ctx: &EmitContext<'_>,
    inst: InstId,
    info: TextureInstInfo,
) -> EmitResult<bool> {
    if ctx.profile.support_quad_divergent_implicit_lod {
        return Ok(false);
    }
    Ok(definition(&ctx.textures, info)?.count > 1
        && !is_draw_uniform(ctx.program, ctx.arg(inst, 0)))
}

/// Whether `value` is the same in every invocation of a draw: built only from constants and
/// constant buffer reads by integer arithmetic. Anything else, or a chain longer than a few
/// dozen instructions, counts as varying.
fn is_draw_uniform(prog: &Program, value: Value) -> bool {
    let mut pending = vec![value];
    let mut visited = 0;
    while let Some(value) = pending.pop() {
        let inst = match value {
            Value::Inst(inst) => inst,
            Value::Void
            | Value::U1(_)
            | Value::U8(_)
            | Value::U16(_)
            | Value::U32(_)
            | Value::U64(_) => continue,
            Value::F32(_) | Value::F64(_) => continue,
            _ => return false,
        };
        visited += 1;
        if visited > 64 {
            return false;
        }
        let uniform_op = matches!(
            prog.op(inst),
            Opcode::Identity
                | Opcode::GetCbufU8
                | Opcode::GetCbufS8
                | Opcode::GetCbufU16
                | Opcode::GetCbufS16
                | Opcode::GetCbufU32
                | Opcode::GetCbufF32
                | Opcode::GetCbufU32x2
                | Opcode::CompositeExtractU32x2
                | Opcode::BitCastU32F32
                | Opcode::BitCastF32U32
                | Opcode::SelectU32
                | Opcode::IAdd32
                | Opcode::ISub32
                | Opcode::IMul32
                | Opcode::INeg32
                | Opcode::IAbs32
                | Opcode::ShiftLeftLogical32
                | Opcode::ShiftRightLogical32
                | Opcode::ShiftRightArithmetic32
                | Opcode::BitwiseAnd32
                | Opcode::BitwiseOr32
                | Opcode::BitwiseXor32
                | Opcode::BitwiseNot32
                | Opcode::BitFieldInsert
                | Opcode::BitFieldSExtract
                | Opcode::BitFieldUExtract
                | Opcode::SMin32
                | Opcode::UMin32
                | Opcode::SMax32
                | Opcode::UMax32
                | Opcode::SClamp32
                | Opcode::UClamp32
                | Opcode::SLessThan
                | Opcode::ULessThan
                | Opcode::IEqual
                | Opcode::SLessThanEqual
                | Opcode::ULessThanEqual
                | Opcode::SGreaterThan
                | Opcode::UGreaterThan
                | Opcode::INotEqual
                | Opcode::SGreaterThanEqual
                | Opcode::UGreaterThanEqual
        );
        if !uniform_op {
            return false;
        }
        pending.extend((0..prog.op(inst).num_args()).map(|i| prog.arg(inst, i)));
    }
    true
}

fn explicit_ops(dref: Option<Word>) -> (Op, Op) {
    match dref {
        Some(_) => (
            Op::ImageSampleDrefExplicitLod,
            Op::ImageSparseSampleDrefExplicitLod,
        ),
        None => (Op::ImageSampleExplicitLod, Op::ImageSparseSampleExplicitLod),
    }
}

/// Implicit lods only exist in fragment programs; elsewhere Maxwell samples as if the lod
/// were zero, so other stages sample lod 0 explicitly (and drop bias and clamp, which have no
/// explicit-lod form).
fn sample_implicit(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    dref: Option<Word>,
    bias_lc: Option<Word>,
    offset: Value,
    result_type: Word,
) -> EmitResult<Word> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    let coords = ctx.def_arg(inst, 1)?;
    let (ops, operands) =
        if ctx.stage == ShaderStage::Fragment && needs_explicit_gradients(ctx, inst, info)? {
            let operands = ImageOperands::implicit_gradient(ctx, info, coords, bias_lc, offset)?;
            (explicit_ops(dref), operands)
        } else if ctx.stage == ShaderStage::Fragment {
            let operands = ImageOperands::implicit_lod(ctx, info, bias_lc, offset)?;
            let ops = match dref {
                Some(_) => (
                    Op::ImageSampleDrefImplicitLod,
                    Op::ImageSparseSampleDrefImplicitLod,
                ),
                None => (Op::ImageSampleImplicitLod, Op::ImageSparseSampleImplicitLod),
            };
            (ops, operands)
        } else {
            crate::fallback::note("TEX", "implicit_lod_non_fragment");
            let lod = ctx.f32_zero_value;
            let operands = ImageOperands::explicit_lod(ctx, lod, offset)?;
            (explicit_ops(dref), operands)
        };
    let texture = texture(ctx, inst, info)?;
    let mut fixed = SmallVec::<[Word; 3]>::from_slice(&[texture, coords]);
    fixed.extend(dref);
    emit_image_op(ctx, inst, info, ops, result_type, &fixed, &operands)
}

fn sample_explicit(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    dref: Option<Word>,
    lod: Word,
    offset: Value,
    result_type: Word,
) -> EmitResult<Word> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    let coords = ctx.def_arg(inst, 1)?;
    let operands = ImageOperands::explicit_lod(ctx, lod, offset)?;
    let texture = texture(ctx, inst, info)?;
    let mut fixed = SmallVec::<[Word; 3]>::from_slice(&[texture, coords]);
    fixed.extend(dref);
    let ops = explicit_ops(dref);
    emit_image_op(ctx, inst, info, ops, result_type, &fixed, &operands)
}

/// Vulkan gathers only from 2D and cube textures; `TLD4` encodes no other type on hardware.
fn gather(ctx: &mut EmitContext<'_>, inst: InstId, dref: Option<Word>) -> EmitResult<Word> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    if matches!(
        info.ty,
        TextureType::Color1D | TextureType::ColorArray1D | TextureType::Color3D
    ) {
        return Err(EmitError::Unsupported("gathers from 1D and 3D textures"));
    }
    let is_integer = is_texture_integer(ctx, info)?;
    let result_type = color_type(ctx, is_integer);
    let mut coords = ctx.def_arg(inst, 1)?;
    let offset = ctx.arg(inst, 2);
    let offset2 = ctx.arg(inst, 3);
    if ctx.profile.need_gather_subpixel_offset {
        coords = gather_subpixel_offset(ctx, inst, info, coords)?;
    }
    let texture = texture(ctx, inst, info)?;
    let (ops, third) = match dref {
        Some(dref) => ((Op::ImageDrefGather, Op::ImageSparseDrefGather), dref),
        None => (
            (Op::ImageGather, Op::ImageSparseGather),
            ctx.const_u32(u32::from(info.gather_component)),
        ),
    };
    let fixed = [texture, coords, third];
    let color = if is_dynamic_ptp(ctx, inst, offset, offset2) {
        let offsets = [offset, offset2];
        gather_dynamic_ptp(ctx, inst, info, ops, result_type, &fixed, offsets)?
    } else {
        let operands = ImageOperands::gather(ctx, offset, offset2)?;
        emit_image_op(ctx, inst, info, ops, result_type, &fixed, &operands)?
    };
    to_f32_color(ctx, is_integer, color)
}

pub(super) fn bindless_image_sample_implicit_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_sample_explicit_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_sample_dref_implicit_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_sample_dref_explicit_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_gather(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_gather_dref(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_fetch(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_query_dimensions(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_query_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_gradient(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_read(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_write(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_sample_implicit_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_sample_explicit_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_sample_dref_implicit_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_sample_dref_explicit_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_gather(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_gather_dref(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_fetch(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_query_dimensions(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_query_lod(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_gradient(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_read(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_write(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    Err(unlowered(ctx, inst))
}

pub(super) fn image_sample_implicit_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    let is_integer = is_texture_integer(ctx, info)?;
    let result_type = color_type(ctx, is_integer);
    let bias_lc = ctx.def_opt(ctx.arg(inst, 2))?;
    let offset = ctx.arg(inst, 3);
    let color = sample_implicit(ctx, inst, None, bias_lc, offset, result_type)?;
    to_f32_color(ctx, is_integer, color)
}

pub(super) fn image_sample_explicit_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    let is_integer = is_texture_integer(ctx, info)?;
    let result_type = color_type(ctx, is_integer);
    let lod = ctx.def_arg(inst, 2)?;
    let offset = ctx.arg(inst, 3);
    let color = sample_explicit(ctx, inst, None, lod, offset, result_type)?;
    to_f32_color(ctx, is_integer, color)
}

/// Vulkan has no depth compare on 3D textures.
fn reject_3d_depth_compare(ctx: &EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    if TextureInstInfo::from_flags(ctx.flags(inst)).ty == TextureType::Color3D {
        return Err(EmitError::Unsupported("depth compare on 3D textures"));
    }
    Ok(())
}

pub(super) fn image_sample_dref_implicit_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    reject_3d_depth_compare(ctx, inst)?;
    let dref = ctx.def_arg(inst, 2)?;
    let bias_lc = ctx.def_opt(ctx.arg(inst, 3))?;
    let offset = ctx.arg(inst, 4);
    let ty = ctx.f32[1];
    sample_implicit(ctx, inst, Some(dref), bias_lc, offset, ty)
}

pub(super) fn image_sample_dref_explicit_lod(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    reject_3d_depth_compare(ctx, inst)?;
    let dref = ctx.def_arg(inst, 2)?;
    let lod = ctx.def_arg(inst, 3)?;
    let offset = ctx.arg(inst, 4);
    let ty = ctx.f32[1];
    sample_explicit(ctx, inst, Some(dref), lod, offset, ty)
}

pub(super) fn image_gather(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    gather(ctx, inst, None)
}

pub(super) fn image_gather_dref(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let dref = ctx.def_arg(inst, 4)?;
    gather(ctx, inst, Some(dref))
}

pub(super) fn image_fetch(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    if matches!(
        info.ty,
        TextureType::ColorCube | TextureType::ColorArrayCube
    ) {
        return Err(EmitError::Unsupported("texel fetches from cube textures"));
    }
    let is_integer = is_texture_integer(ctx, info)?;
    let result_type = color_type(ctx, is_integer);
    let coords = ctx.def_arg(inst, 1)?;
    let offset = ctx.def_opt(ctx.arg(inst, 2))?;
    let ms = match ctx.def_opt(ctx.arg(inst, 4))? {
        None if is_texture_msaa(ctx, info)? => Some(ctx.u32_zero_value),
        ms => ms,
    };
    let lod = if info.ty == TextureType::Buffer || ms.is_some() {
        None
    } else {
        ctx.def_opt(ctx.arg(inst, 3))?
    };
    let coords = add_offset_to_coordinates(ctx, info, coords, offset)?;
    let operands = ImageOperands::fetch(ctx, lod, ms);
    let image = texture_image(ctx, inst, info)?;
    let ops = (Op::ImageFetch, Op::ImageSparseFetch);
    let color = emit_image_op(
        ctx,
        inst,
        info,
        ops,
        result_type,
        &[image, coords],
        &operands,
    )?;
    to_f32_color(ctx, is_integer, color)
}

/// Size, zeros and mip count as a U32x4. Buffers have no mip levels and SPIR-V cannot query
/// them (the C++ did, which is invalid), so they report one.
pub(super) fn image_query_dimensions(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    let image = texture_image(ctx, inst, info)?;
    let lod = ctx.def_arg(inst, 1)?;
    let skip_mips = ctx.imm_u32(inst, 2)? != 0;
    let zero = ctx.u32_zero_value;
    let mips = if skip_mips {
        zero
    } else if info.ty == TextureType::Buffer {
        crate::fallback::note("TXQ", "query_dimensions_buffer_mip_count");
        ctx.const_u32(1)
    } else {
        ctx.b.image_query_levels(ctx.u32[1], None, image)?
    };
    let uses_lod = !is_texture_msaa(ctx, info)? && info.ty != TextureType::Buffer;
    let components = match info.ty {
        TextureType::Color1D | TextureType::Buffer => 1,
        TextureType::ColorArray1D
        | TextureType::Color2D
        | TextureType::ColorCube
        | TextureType::Color2DRect => 2,
        TextureType::ColorArray2D | TextureType::Color3D | TextureType::ColorArrayCube => 3,
    };
    let size_type = ctx.u32[components];
    let size = if uses_lod {
        ctx.b.image_query_size_lod(size_type, None, image, lod)?
    } else {
        ctx.b.image_query_size(size_type, None, image)?
    };
    let mut constituents = SmallVec::<[Word; 4]>::from_slice(&[size]);
    constituents.extend(std::iter::repeat_n(zero, 3 - components));
    constituents.push(mips);
    Ok(ctx.b.composite_construct(ctx.u32[4], None, constituents)?)
}

/// `OpImageQuerySamples` on a multisampled texture; every other texture has one sample.
pub(super) fn image_query_samples(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    if !is_texture_msaa(ctx, info)? {
        return Ok(ctx.const_u32(1));
    }
    let image = texture_image(ctx, inst, info)?;
    Ok(ctx.b.image_query_samples(ctx.u32[1], None, image)?)
}

pub(super) fn image_query_lod(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    let coords = ctx.def_arg(inst, 1)?;
    let sampler = texture(ctx, inst, info)?;
    let lod = ctx.b.image_query_lod(ctx.f32[2], None, sampler, coords)?;
    let zero = ctx.f32_zero_value;
    Ok(ctx
        .b
        .composite_construct(ctx.f32[4], None, [lod, zero, zero])?)
}

pub(super) fn image_gradient(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    let is_integer = is_texture_integer(ctx, info)?;
    let result_type = color_type(ctx, is_integer);
    let coords = ctx.def_arg(inst, 1)?;
    let derivatives = ctx.def_arg(inst, 2)?;
    let offset = ctx.arg(inst, 3);
    let lod_clamp = ctx.def_opt(ctx.arg(inst, 4))?;
    let operands = ImageOperands::gradient(ctx, info, derivatives, offset, lod_clamp)?;
    let texture = texture(ctx, inst, info)?;
    let ops = explicit_ops(None);
    let color = emit_image_op(
        ctx,
        inst,
        info,
        ops,
        result_type,
        &[texture, coords],
        &operands,
    )?;
    to_f32_color(ctx, is_integer, color)
}

pub(super) fn image_read(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    if info.image_format == ImageFormat::Typeless && !ctx.profile.support_typeless_image_loads {
        tracing::warn!("typeless image read not supported by the host");
        crate::fallback::note("SULD", "typeless_image_read_unsupported");
        return Ok(ctx.const_null(ctx.u32[4]));
    }
    let coords = ctx.def_arg(inst, 1)?;
    let (image, is_integer) = load_storage_image(ctx, inst, info)?;
    let result_type = color_type(ctx, is_integer);
    let ops = (Op::ImageRead, Op::ImageSparseRead);
    let operands = ImageOperands::new();
    let color = emit_image_op(
        ctx,
        inst,
        info,
        ops,
        result_type,
        &[image, coords],
        &operands,
    )?;
    if is_integer {
        return Ok(color);
    }
    Ok(ctx.b.bitcast(ctx.u32[4], None, color)?)
}

pub(super) fn image_write(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    let coords = ctx.def_arg(inst, 1)?;
    let mut color = ctx.def_arg(inst, 2)?;
    let (image, is_integer) = load_storage_image(ctx, inst, info)?;
    if !is_integer {
        color = ctx.b.bitcast(ctx.f32[4], None, color)?;
    }
    Ok(ctx.b.image_write(image, coords, color, None, [])?)
}

/// Tests bit `index + base_index` of the rescaling bit set at `member_index` of the rescaling
/// push constant block. An immediate index folds the word and mask to constants.
fn is_scaled(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    member_index: u32,
    base_index: u32,
) -> EmitResult<Word> {
    let block = ctx.rescaling_push_constants;
    if block == 0 {
        return Err(EmitError::InvalidIr(
            "rescaling query without the rescaling push constants".into(),
        ));
    }
    let u32_t = ctx.u32[1];
    let (word_index, mask) = match ctx.arg(inst, 0).imm_u32() {
        Some(index) => {
            let index = index + base_index;
            (ctx.const_u32(index / 32), ctx.const_u32(1 << (index % 32)))
        }
        None => {
            let mut index = ctx.def_arg(inst, 0)?;
            if base_index != 0 {
                let base = ctx.const_u32(base_index);
                index = ctx.b.i_add(u32_t, None, index, base)?;
            }
            let five = ctx.const_u32(5);
            let thirty_one = ctx.const_u32(31);
            let one = ctx.const_u32(1);
            let word_index = ctx.b.shift_right_logical(u32_t, None, index, five)?;
            let bit_index = ctx.b.bitwise_and(u32_t, None, index, thirty_one)?;
            let mask = ctx.b.shift_left_logical(u32_t, None, one, bit_index)?;
            (word_index, mask)
        }
    };
    let pointer_type = ctx.type_pointer(StorageClass::PushConstant, u32_t);
    let member = ctx.const_u32(member_index);
    let pointer = ctx
        .b
        .access_chain(pointer_type, None, block, [member, word_index])?;
    let word = ctx.b.load(u32_t, None, pointer, None, [])?;
    let bit = ctx.b.bitwise_and(u32_t, None, word, mask)?;
    Ok(ctx.b.i_not_equal(ctx.u1, None, bit, ctx.u32_zero_value)?)
}

/// The non-unified (OpenGL) layout: the bit sets are the bit patterns of components of a vec4
/// uniform.
fn is_scaled_uniform(ctx: &mut EmitContext<'_>, inst: InstId, component: u32) -> EmitResult<Word> {
    let uniform = ctx.rescaling_uniform_constant;
    if uniform == 0 {
        return Err(EmitError::InvalidIr(
            "rescaling query without the rescaling uniform".into(),
        ));
    }
    let u32_t = ctx.u32[1];
    let composite = ctx.b.load(ctx.f32[4], None, uniform, None, [])?;
    let mask = ctx
        .b
        .composite_extract(ctx.f32[1], None, composite, [component])?;
    let mask = ctx.b.bitcast(u32_t, None, mask)?;
    let bit = ctx.def_arg(inst, 0)?;
    let shifted = ctx.b.shift_right_logical(u32_t, None, mask, bit)?;
    let one = ctx.const_u32(1);
    let value = ctx.b.bitwise_and(u32_t, None, shifted, one)?;
    Ok(ctx.b.i_not_equal(ctx.u1, None, value, ctx.u32_zero_value)?)
}

pub(super) fn is_texture_scaled(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    if !ctx.profile.unified_descriptor_binding {
        return is_scaled_uniform(ctx, inst, 0);
    }
    let (member, base) = (
        ctx.rescaling_textures_member_index,
        ctx.texture_rescaling_index,
    );
    is_scaled(ctx, inst, member, base)
}

pub(super) fn is_image_scaled(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    if !ctx.profile.unified_descriptor_binding {
        return is_scaled_uniform(ctx, inst, 1);
    }
    let (member, base) = (ctx.rescaling_images_member_index, ctx.image_rescaling_index);
    is_scaled(ctx, inst, member, base)
}
