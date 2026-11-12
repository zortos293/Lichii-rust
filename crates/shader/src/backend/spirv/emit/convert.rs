//! Numeric conversions, from the C++ `emit_spirv_convert.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Opcode};

pub(super) fn convert_s16_f16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertS16F16))
}

pub(super) fn convert_s16_f32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertS16F32))
}

pub(super) fn convert_s16_f64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertS16F64))
}

pub(super) fn convert_s32_f16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertS32F16))
}

pub(super) fn convert_s32_f32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertS32F32))
}

pub(super) fn convert_s32_f64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertS32F64))
}

pub(super) fn convert_s64_f16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertS64F16))
}

pub(super) fn convert_s64_f32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertS64F32))
}

pub(super) fn convert_s64_f64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertS64F64))
}

pub(super) fn convert_u16_f16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU16F16))
}

pub(super) fn convert_u16_f32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU16F32))
}

pub(super) fn convert_u16_f64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU16F64))
}

pub(super) fn convert_u32_f16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU32F16))
}

pub(super) fn convert_u32_f32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU32F32))
}

pub(super) fn convert_u32_f64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU32F64))
}

pub(super) fn convert_u64_f16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU64F16))
}

pub(super) fn convert_u64_f32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU64F32))
}

pub(super) fn convert_u64_f64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU64F64))
}

pub(super) fn convert_u64_u32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU64U32))
}

pub(super) fn convert_u32_u64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU32U64))
}

pub(super) fn convert_f16_f32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF16F32))
}

pub(super) fn convert_f32_f16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF32F16))
}

pub(super) fn convert_f32_f64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF32F64))
}

pub(super) fn convert_f64_f32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF64F32))
}

pub(super) fn convert_f16_s8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF16S8))
}

pub(super) fn convert_f16_s16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF16S16))
}

pub(super) fn convert_f16_s32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF16S32))
}

pub(super) fn convert_f16_s64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF16S64))
}

pub(super) fn convert_f16_u8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF16U8))
}

pub(super) fn convert_f16_u16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF16U16))
}

pub(super) fn convert_f16_u32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF16U32))
}

pub(super) fn convert_f16_u64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF16U64))
}

pub(super) fn convert_f32_s8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF32S8))
}

pub(super) fn convert_f32_s16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF32S16))
}

pub(super) fn convert_f32_s32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF32S32))
}

pub(super) fn convert_f32_s64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF32S64))
}

pub(super) fn convert_f32_u8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF32U8))
}

pub(super) fn convert_f32_u16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF32U16))
}

pub(super) fn convert_f32_u32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF32U32))
}

pub(super) fn convert_f32_u64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF32U64))
}

pub(super) fn convert_f64_s8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF64S8))
}

pub(super) fn convert_f64_s16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF64S16))
}

pub(super) fn convert_f64_s32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF64S32))
}

pub(super) fn convert_f64_s64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF64S64))
}

pub(super) fn convert_f64_u8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF64U8))
}

pub(super) fn convert_f64_u16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF64U16))
}

pub(super) fn convert_f64_u32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF64U32))
}

pub(super) fn convert_f64_u64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertF64U64))
}

pub(super) fn convert_u16_u32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU16U32))
}

pub(super) fn convert_u32_u16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU32U16))
}

pub(super) fn convert_u8_u32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU8U32))
}

pub(super) fn convert_u32_u8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertU32U8))
}

pub(super) fn convert_s32_s8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertS32S8))
}

pub(super) fn convert_s32_s16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ConvertS32S16))
}
