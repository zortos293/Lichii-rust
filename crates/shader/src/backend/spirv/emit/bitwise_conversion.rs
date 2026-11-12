//! Bit casts and packing, from the C++ `emit_spirv_bitwise_conversion.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Opcode};

pub(super) fn bit_cast_u16_f16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitCastU16F16))
}

pub(super) fn bit_cast_u32_f32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitCastU32F32))
}

pub(super) fn bit_cast_u64_f64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitCastU64F64))
}

pub(super) fn bit_cast_f16_u16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitCastF16U16))
}

pub(super) fn bit_cast_f32_u32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitCastF32U32))
}

pub(super) fn bit_cast_f64_u64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitCastF64U64))
}

pub(super) fn pack_uint2x32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::PackUint2x32))
}

pub(super) fn unpack_uint2x32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::UnpackUint2x32))
}

pub(super) fn pack_float2x16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::PackFloat2x16))
}

pub(super) fn unpack_float2x16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::UnpackFloat2x16))
}

pub(super) fn pack_half2x16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::PackHalf2x16))
}

pub(super) fn unpack_half2x16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::UnpackHalf2x16))
}

pub(super) fn pack_double2x32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::PackDouble2x32))
}

pub(super) fn unpack_double2x32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::UnpackDouble2x32))
}
