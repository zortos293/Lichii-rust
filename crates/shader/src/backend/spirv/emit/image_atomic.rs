//! Image atomics, from the C++ `emit_spirv_image_atomic.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Opcode};

pub(super) fn bindless_image_atomic_i_add32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageAtomicIAdd32))
}

pub(super) fn bindless_image_atomic_s_min32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageAtomicSMin32))
}

pub(super) fn bindless_image_atomic_u_min32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageAtomicUMin32))
}

pub(super) fn bindless_image_atomic_s_max32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageAtomicSMax32))
}

pub(super) fn bindless_image_atomic_u_max32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageAtomicUMax32))
}

pub(super) fn bindless_image_atomic_inc32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageAtomicInc32))
}

pub(super) fn bindless_image_atomic_dec32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageAtomicDec32))
}

pub(super) fn bindless_image_atomic_and32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageAtomicAnd32))
}

pub(super) fn bindless_image_atomic_or32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageAtomicOr32))
}

pub(super) fn bindless_image_atomic_xor32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BindlessImageAtomicXor32))
}

pub(super) fn bindless_image_atomic_exchange32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(
        Opcode::BindlessImageAtomicExchange32,
    ))
}

pub(super) fn bound_image_atomic_i_add32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageAtomicIAdd32))
}

pub(super) fn bound_image_atomic_s_min32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageAtomicSMin32))
}

pub(super) fn bound_image_atomic_u_min32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageAtomicUMin32))
}

pub(super) fn bound_image_atomic_s_max32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageAtomicSMax32))
}

pub(super) fn bound_image_atomic_u_max32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageAtomicUMax32))
}

pub(super) fn bound_image_atomic_inc32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageAtomicInc32))
}

pub(super) fn bound_image_atomic_dec32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageAtomicDec32))
}

pub(super) fn bound_image_atomic_and32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageAtomicAnd32))
}

pub(super) fn bound_image_atomic_or32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageAtomicOr32))
}

pub(super) fn bound_image_atomic_xor32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageAtomicXor32))
}

pub(super) fn bound_image_atomic_exchange32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BoundImageAtomicExchange32))
}

pub(super) fn image_atomic_i_add32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageAtomicIAdd32))
}

pub(super) fn image_atomic_s_min32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageAtomicSMin32))
}

pub(super) fn image_atomic_u_min32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageAtomicUMin32))
}

pub(super) fn image_atomic_s_max32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageAtomicSMax32))
}

pub(super) fn image_atomic_u_max32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageAtomicUMax32))
}

pub(super) fn image_atomic_inc32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageAtomicInc32))
}

pub(super) fn image_atomic_dec32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageAtomicDec32))
}

pub(super) fn image_atomic_and32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageAtomicAnd32))
}

pub(super) fn image_atomic_or32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageAtomicOr32))
}

pub(super) fn image_atomic_xor32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageAtomicXor32))
}

pub(super) fn image_atomic_exchange32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ImageAtomicExchange32))
}
