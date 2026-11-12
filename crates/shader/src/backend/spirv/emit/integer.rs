//! Integer arithmetic, bit operations and comparisons, from the C++ `emit_spirv_integer.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Opcode};

pub(super) fn i_add32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::IAdd32))
}

pub(super) fn i_add64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::IAdd64))
}

pub(super) fn i_sub32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ISub32))
}

pub(super) fn i_sub64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ISub64))
}

pub(super) fn i_mul32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::IMul32))
}

pub(super) fn s_div32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SDiv32))
}

pub(super) fn u_div32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::UDiv32))
}

pub(super) fn i_neg32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::INeg32))
}

pub(super) fn i_neg64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::INeg64))
}

pub(super) fn i_abs32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::IAbs32))
}

pub(super) fn i_abs64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::IAbs64))
}

pub(super) fn shift_left_logical32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ShiftLeftLogical32))
}

pub(super) fn shift_left_logical64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ShiftLeftLogical64))
}

pub(super) fn shift_right_logical32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ShiftRightLogical32))
}

pub(super) fn shift_right_logical64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ShiftRightLogical64))
}

pub(super) fn shift_right_arithmetic32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ShiftRightArithmetic32))
}

pub(super) fn shift_right_arithmetic64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ShiftRightArithmetic64))
}

pub(super) fn bitwise_and32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitwiseAnd32))
}

pub(super) fn bitwise_or32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitwiseOr32))
}

pub(super) fn bitwise_xor32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitwiseXor32))
}

pub(super) fn bit_field_insert(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitFieldInsert))
}

pub(super) fn bit_field_s_extract(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitFieldSExtract))
}

pub(super) fn bit_field_u_extract(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitFieldUExtract))
}

pub(super) fn bit_reverse32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitReverse32))
}

pub(super) fn bit_count32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitCount32))
}

pub(super) fn bitwise_not32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::BitwiseNot32))
}

pub(super) fn find_s_msb32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FindSMsb32))
}

pub(super) fn find_u_msb32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FindUMsb32))
}

pub(super) fn s_min32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SMin32))
}

pub(super) fn u_min32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::UMin32))
}

pub(super) fn s_max32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SMax32))
}

pub(super) fn u_max32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::UMax32))
}

pub(super) fn s_clamp32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SClamp32))
}

pub(super) fn u_clamp32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::UClamp32))
}

pub(super) fn s_less_than(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SLessThan))
}

pub(super) fn u_less_than(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ULessThan))
}

pub(super) fn i_equal(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::IEqual))
}

pub(super) fn s_less_than_equal(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SLessThanEqual))
}

pub(super) fn u_less_than_equal(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ULessThanEqual))
}

pub(super) fn s_greater_than(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SGreaterThan))
}

pub(super) fn u_greater_than(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::UGreaterThan))
}

pub(super) fn i_not_equal(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::INotEqual))
}

pub(super) fn s_greater_than_equal(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SGreaterThanEqual))
}

pub(super) fn u_greater_than_equal(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::UGreaterThanEqual))
}
