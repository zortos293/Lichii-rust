//! Undefined values, from the C++ `emit_spirv_undefined.cpp`.
//!
//! Registers read before any write are zero rather than OpUndef, as NVIDIA hardware and drivers
//! give them; AMD's OpUndef is whatever the register held, which blended garbage into a
//! Switch 2 title's glow pass.

use rspirv::spirv::Word;

use crate::backend::spirv::EmitResult;
use crate::backend::spirv::context::EmitContext;
use crate::ir::InstId;

pub(super) fn undef_u1(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Ok(ctx.false_value)
}

pub(super) fn undef_u8(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u8;
    Ok(ctx.const_null(ty))
}

pub(super) fn undef_u16(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u16;
    Ok(ctx.const_null(ty))
}

pub(super) fn undef_u32(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Ok(ctx.u32_zero_value)
}

pub(super) fn undef_u64(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Ok(ctx.const_u64(0))
}
