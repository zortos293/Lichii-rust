//! Subgroup votes, ballots, shuffles and derivatives, from the C++ `emit_spirv_warp.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Opcode};

pub(super) fn lane_id(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LaneId))
}

pub(super) fn vote_all(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::VoteAll))
}

pub(super) fn vote_any(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::VoteAny))
}

pub(super) fn vote_equal(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::VoteEqual))
}

pub(super) fn subgroup_ballot(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SubgroupBallot))
}

pub(super) fn subgroup_eq_mask(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SubgroupEqMask))
}

pub(super) fn subgroup_lt_mask(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SubgroupLtMask))
}

pub(super) fn subgroup_le_mask(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SubgroupLeMask))
}

pub(super) fn subgroup_gt_mask(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SubgroupGtMask))
}

pub(super) fn subgroup_ge_mask(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SubgroupGeMask))
}

pub(super) fn shuffle_index(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ShuffleIndex))
}

pub(super) fn shuffle_up(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ShuffleUp))
}

pub(super) fn shuffle_down(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ShuffleDown))
}

pub(super) fn shuffle_butterfly(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::ShuffleButterfly))
}

pub(super) fn quad_broadcast(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::QuadBroadcast))
}

pub(super) fn quad_swap(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::QuadSwap))
}

pub(super) fn f_swizzle_add(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FSwizzleAdd))
}

pub(super) fn d_pdx_fine(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::DPdxFine))
}

pub(super) fn d_pdy_fine(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::DPdyFine))
}

pub(super) fn d_pdx_coarse(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::DPdxCoarse))
}

pub(super) fn d_pdy_coarse(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::DPdyCoarse))
}
