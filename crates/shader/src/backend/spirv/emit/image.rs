//! Texture and image sampling, fetches, reads, writes and queries, from the C++ `emit_spirv_image.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Opcode};

pub(super) fn bindless_image_sample_implicit_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(
        Opcode::BindlessImageSampleImplicitLod,
    ))
}

pub(super) fn bindless_image_sample_explicit_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(
        Opcode::BindlessImageSampleExplicitLod,
    ))
}

pub(super) fn bindless_image_sample_dref_implicit_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(
        Opcode::BindlessImageSampleDrefImplicitLod,
    ))
}

pub(super) fn bindless_image_sample_dref_explicit_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(
        Opcode::BindlessImageSampleDrefExplicitLod,
    ))
}

pub(super) fn bindless_image_gather(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageGather))
}

pub(super) fn bindless_image_gather_dref(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageGatherDref))
}

pub(super) fn bindless_image_fetch(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageFetch))
}

pub(super) fn bindless_image_query_dimensions(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(
        Opcode::BindlessImageQueryDimensions,
    ))
}

pub(super) fn bindless_image_query_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageQueryLod))
}

pub(super) fn bindless_image_gradient(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageGradient))
}

pub(super) fn bindless_image_read(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageRead))
}

pub(super) fn bindless_image_write(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageWrite))
}

pub(super) fn bound_image_sample_implicit_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(
        Opcode::BoundImageSampleImplicitLod,
    ))
}

pub(super) fn bound_image_sample_explicit_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(
        Opcode::BoundImageSampleExplicitLod,
    ))
}

pub(super) fn bound_image_sample_dref_implicit_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(
        Opcode::BoundImageSampleDrefImplicitLod,
    ))
}

pub(super) fn bound_image_sample_dref_explicit_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(
        Opcode::BoundImageSampleDrefExplicitLod,
    ))
}

pub(super) fn bound_image_gather(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageGather))
}

pub(super) fn bound_image_gather_dref(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageGatherDref))
}

pub(super) fn bound_image_fetch(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageFetch))
}

pub(super) fn bound_image_query_dimensions(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageQueryDimensions))
}

pub(super) fn bound_image_query_lod(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageQueryLod))
}

pub(super) fn bound_image_gradient(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageGradient))
}

pub(super) fn bound_image_read(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageRead))
}

pub(super) fn bound_image_write(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::BoundImageWrite))
}

pub(super) fn image_sample_implicit_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageSampleImplicitLod))
}

pub(super) fn image_sample_explicit_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageSampleExplicitLod))
}

pub(super) fn image_sample_dref_implicit_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageSampleDrefImplicitLod))
}

pub(super) fn image_sample_dref_explicit_lod(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageSampleDrefExplicitLod))
}

pub(super) fn image_gather(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageGather))
}

pub(super) fn image_gather_dref(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageGatherDref))
}

pub(super) fn image_fetch(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageFetch))
}

pub(super) fn image_query_dimensions(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageQueryDimensions))
}

pub(super) fn image_query_lod(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageQueryLod))
}

pub(super) fn image_gradient(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageGradient))
}

pub(super) fn image_read(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageRead))
}

pub(super) fn image_write(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::ImageWrite))
}

pub(super) fn is_texture_scaled(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::IsTextureScaled))
}

pub(super) fn is_image_scaled(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::IsImageScaled))
}
