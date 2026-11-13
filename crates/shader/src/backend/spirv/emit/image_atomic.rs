//! Image atomics, from the C++ `emit_spirv_image_atomic.cpp`.
//!
//! Each atomic takes a texel pointer into the storage image (or storage texel buffer) and runs
//! at device scope with relaxed semantics. Unlike the C++, descriptor arrays are indexed.

use rspirv::dr::{Builder, Error};
use rspirv::spirv::{Scope, StorageClass, Word};

use super::image::{mark_non_uniform, storage_image, unlowered};
use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, TextureInstInfo};

type AtomicOp = fn(&mut Builder, Word, Option<Word>, Word, Word, Word, Word) -> Result<Word, Error>;

fn image_atomic(ctx: &mut EmitContext<'_>, inst: InstId, atomic: AtomicOp) -> EmitResult<Word> {
    let info = TextureInstInfo::from_flags(ctx.flags(inst));
    let coords = ctx.def_arg(inst, 1)?;
    let value = ctx.def_arg(inst, 2)?;
    let image = storage_image(ctx, inst, info)?;
    let u32_t = ctx.u32[1];
    let pointer_type = ctx.type_pointer(StorageClass::Image, u32_t);
    let zero = ctx.u32_zero_value;
    let texel = ctx
        .b
        .image_texel_pointer(pointer_type, None, image.pointer, coords, zero)?;
    if image.non_uniform {
        mark_non_uniform(ctx, texel);
    }
    let scope = ctx.const_u32(Scope::Device as u32);
    Ok(atomic(&mut ctx.b, u32_t, None, texel, scope, zero, value)?)
}

/// Maxwell's wrapping increment and decrement need a compare-exchange loop, which cannot be
/// emitted inside the block of one IR instruction, and an image texel pointer cannot be passed
/// to a helper function. The C++ leaves them unimplemented too.
fn wrapping_atomic() -> EmitError {
    EmitError::Unsupported("image atomic increment and decrement")
}

pub(super) fn bindless_image_atomic_i_add32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_atomic_i_add32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_atomic_s_min32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_atomic_s_min32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_atomic_u_min32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_atomic_u_min32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_atomic_s_max32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_atomic_s_max32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_atomic_u_max32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_atomic_u_max32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_atomic_inc32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_atomic_inc32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_atomic_dec32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_atomic_dec32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_atomic_and32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_atomic_and32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_atomic_or32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_atomic_or32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_atomic_xor32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_atomic_xor32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bindless_image_atomic_exchange32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn bound_image_atomic_exchange32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    Err(unlowered(ctx, inst))
}

pub(super) fn image_atomic_i_add32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    image_atomic(ctx, inst, Builder::atomic_i_add)
}

pub(super) fn image_atomic_s_min32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    image_atomic(ctx, inst, Builder::atomic_s_min)
}

pub(super) fn image_atomic_u_min32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    image_atomic(ctx, inst, Builder::atomic_u_min)
}

pub(super) fn image_atomic_s_max32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    image_atomic(ctx, inst, Builder::atomic_s_max)
}

pub(super) fn image_atomic_u_max32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    image_atomic(ctx, inst, Builder::atomic_u_max)
}

pub(super) fn image_atomic_inc32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(wrapping_atomic())
}

pub(super) fn image_atomic_dec32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(wrapping_atomic())
}

pub(super) fn image_atomic_and32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    image_atomic(ctx, inst, Builder::atomic_and)
}

pub(super) fn image_atomic_or32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    image_atomic(ctx, inst, Builder::atomic_or)
}

pub(super) fn image_atomic_xor32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    image_atomic(ctx, inst, Builder::atomic_xor)
}

pub(super) fn image_atomic_exchange32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    image_atomic(ctx, inst, Builder::atomic_exchange)
}
