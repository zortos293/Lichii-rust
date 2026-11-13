//! Subgroup votes, ballots, shuffles and derivatives, from the C++ `emit_spirv_warp.cpp`.
//!
//! The guest warp is 32 lanes. A host subgroup may be larger
//! (`Profile::warp_size_potentially_larger_than_guest`): each 32-lane slice of it then stands
//! for one guest warp, so ballots and masks read the slice's word and lane ids are taken
//! modulo 32. Stages without subgroup support see a single-lane warp.

use rspirv::spirv::{Scope, Word};

use crate::backend::spirv::EmitResult;
use crate::backend::spirv::context::EmitContext;
use crate::ir::{InstId, Opcode};

fn subgroup_scope(ctx: &mut EmitContext<'_>) -> Word {
    ctx.const_u32(Scope::Subgroup as u32)
}

/// The host subgroup invocation id, or zero in stages without subgroups.
fn thread_id(ctx: &mut EmitContext<'_>) -> EmitResult<Word> {
    if !ctx.supports_subgroups() {
        return Ok(ctx.u32_zero_value);
    }
    let (u32_t, id) = (ctx.u32[1], ctx.subgroup_invocation_id());
    Ok(ctx.b.load(u32_t, None, id, None, [])?)
}

/// The word of a 128-bit subgroup mask that covers this invocation's 32-lane slice.
fn warp_extract(ctx: &mut EmitContext<'_>, value: Word) -> EmitResult<Word> {
    let u32_t = ctx.u32[1];
    let thread = thread_id(ctx)?;
    let five = ctx.const_u32(5);
    let local_index = ctx.b.shift_right_arithmetic(u32_t, None, thread, five)?;
    if !ctx
        .profile
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic
    {
        return Ok(ctx
            .b
            .vector_extract_dynamic(u32_t, None, value, local_index)?);
    }
    let (u1, zero) = (ctx.u1, ctx.u32_zero_value);
    let mut result = zero;
    for word in 0..4 {
        let index = ctx.const_u32(word);
        let matches = ctx.b.i_equal(u1, None, local_index, index)?;
        let component = ctx.b.composite_extract(u32_t, None, value, [word])?;
        let selected = ctx.b.select(u32_t, None, matches, component, zero)?;
        result = if word == 0 {
            selected
        } else {
            ctx.b.bitwise_or(u32_t, None, result, selected)?
        };
    }
    Ok(result)
}

/// The guest-visible 32 bits of a subgroup ballot.
fn ballot_word(ctx: &mut EmitContext<'_>, ballot: Word) -> EmitResult<Word> {
    if ctx.profile.warp_size_potentially_larger_than_guest {
        return warp_extract(ctx, ballot);
    }
    let u32_t = ctx.u32[1];
    Ok(ctx.b.composite_extract(u32_t, None, ballot, [0])?)
}

fn ballot(ctx: &mut EmitContext<'_>, pred: Word) -> EmitResult<Word> {
    let (u32x4, scope) = (ctx.u32[4], subgroup_scope(ctx));
    Ok(ctx.b.group_non_uniform_ballot(u32x4, None, scope, pred)?)
}

/// The active-lane mask and the ballot of `pred` within this invocation's guest warp, for the
/// emulated votes.
fn emulated_vote_masks(ctx: &mut EmitContext<'_>, pred: Word) -> EmitResult<(Word, Word)> {
    let true_value = ctx.true_value;
    let active = ballot(ctx, true_value)?;
    let active = warp_extract(ctx, active)?;
    let votes = ballot(ctx, pred)?;
    let votes = warp_extract(ctx, votes)?;
    Ok((active, votes))
}

fn load_mask(ctx: &mut EmitContext<'_>, mask: Word) -> EmitResult<Word> {
    let u32x4 = ctx.u32[4];
    let value = ctx.b.load(u32x4, None, mask, None, [])?;
    ballot_word(ctx, value)
}

fn subgroup_mask(ctx: &mut EmitContext<'_>, mask: Word, single_lane: u32) -> EmitResult<Word> {
    if !ctx.supports_subgroups() {
        return Ok(ctx.const_u32(single_lane));
    }
    load_mask(ctx, mask)
}

fn set_in_bounds(ctx: &mut EmitContext<'_>, inst: InstId, in_range: Word) {
    if let Some(in_bounds) = ctx.program.pseudo_op(inst, Opcode::GetInBoundsFromOp) {
        ctx.set_def(in_bounds, in_range);
    }
}

/// The highest guest lane a shuffle may read: the segment base plus the clamp.
fn max_thread_id(
    ctx: &mut EmitContext<'_>,
    thread: Word,
    clamp: Word,
    segmentation_mask: Word,
) -> EmitResult<Word> {
    let u32_t = ctx.u32[1];
    let not_seg_mask = ctx.b.not(u32_t, None, segmentation_mask)?;
    let min_thread = ctx.b.bitwise_and(u32_t, None, thread, segmentation_mask)?;
    let clamped = ctx.b.bitwise_and(u32_t, None, clamp, not_seg_mask)?;
    Ok(ctx.b.bitwise_or(u32_t, None, min_thread, clamped)?)
}

/// A guest lane as a host invocation id, in this invocation's 32-lane slice.
fn host_thread_id(ctx: &mut EmitContext<'_>, lane: Word) -> EmitResult<Word> {
    if !ctx.profile.warp_size_potentially_larger_than_guest {
        return Ok(lane);
    }
    let u32_t = ctx.u32[1];
    let thread = thread_id(ctx)?;
    let five = ctx.const_u32(5);
    let partition = ctx.b.shift_right_logical(u32_t, None, thread, five)?;
    let base = ctx.b.shift_left_logical(u32_t, None, partition, five)?;
    Ok(ctx.b.i_add(u32_t, None, lane, base)?)
}

fn guest_lane(ctx: &mut EmitContext<'_>, index: Word) -> EmitResult<Word> {
    let u32_t = ctx.u32[1];
    let mask = ctx.const_u32(31);
    Ok(ctx.b.bitwise_and(u32_t, None, index, mask)?)
}

fn use_shuffle_fallback(ctx: &EmitContext<'_>) -> bool {
    ctx.profile.has_broken_spirv_subgroup_shuffle
}

/// Reads `value` from host invocation `src`. Hosts with broken shuffles broadcast every lane
/// and select the one asked for.
fn shuffle_absolute(ctx: &mut EmitContext<'_>, value: Word, src: Word) -> EmitResult<Word> {
    let (u1, u32_t, scope) = (ctx.u1, ctx.u32[1], subgroup_scope(ctx));
    if !use_shuffle_fallback(ctx) {
        return Ok(ctx
            .b
            .group_non_uniform_shuffle(u32_t, None, scope, value, src)?);
    }
    let mut result = ctx.u32_zero_value;
    for lane in 0..ctx.profile.max_subgroup_size {
        let lane = ctx.const_u32(lane);
        let read = ctx
            .b
            .group_non_uniform_broadcast(u32_t, None, scope, value, lane)?;
        let matches = ctx.b.i_equal(u1, None, src, lane)?;
        result = ctx.b.select(u32_t, None, matches, read, result)?;
    }
    Ok(result)
}

#[derive(Clone, Copy)]
enum Direction {
    Up,
    Down,
}

fn shuffle_relative(
    ctx: &mut EmitContext<'_>,
    value: Word,
    delta: Word,
    src: Word,
    direction: Direction,
) -> EmitResult<Word> {
    if !ctx.profile.support_shuffle_relative || use_shuffle_fallback(ctx) {
        let host_src = host_thread_id(ctx, src)?;
        return shuffle_absolute(ctx, value, host_src);
    }
    let (u32_t, scope) = (ctx.u32[1], subgroup_scope(ctx));
    Ok(match direction {
        Direction::Up => ctx
            .b
            .group_non_uniform_shuffle_up(u32_t, None, scope, value, delta)?,
        Direction::Down => ctx
            .b
            .group_non_uniform_shuffle_down(u32_t, None, scope, value, delta)?,
    })
}

/// Reads `value` from guest lane `lane` of this invocation's guest warp.
fn broadcast_lane(ctx: &mut EmitContext<'_>, value: Word, lane: u32) -> EmitResult<Word> {
    let (u1, u32_t, scope) = (ctx.u1, ctx.u32[1], subgroup_scope(ctx));
    let lane_id = ctx.const_u32(lane);
    let mut result = ctx
        .b
        .group_non_uniform_broadcast(u32_t, None, scope, value, lane_id)?;
    if !ctx.profile.warp_size_potentially_larger_than_guest {
        return Ok(result);
    }
    let thread = thread_id(ctx)?;
    let five = ctx.const_u32(5);
    let partition = ctx.b.shift_right_logical(u32_t, None, thread, five)?;
    for base in (32..ctx.profile.max_subgroup_size).step_by(32) {
        let lane_id = ctx.const_u32(base + lane);
        let read = ctx
            .b
            .group_non_uniform_broadcast(u32_t, None, scope, value, lane_id)?;
        let slice = ctx.const_u32(base >> 5);
        let matches = ctx.b.i_equal(u1, None, partition, slice)?;
        result = ctx.b.select(u32_t, None, matches, read, result)?;
    }
    Ok(result)
}

/// The operands of a relative shuffle: value, lane delta (or xor mask), this lane, and the
/// highest lane the shuffle may read.
fn shuffle_operands(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<[Word; 4]> {
    let value = ctx.def_arg(inst, 0)?;
    let index = ctx.def_arg(inst, 1)?;
    let clamp = ctx.def_arg(inst, 2)?;
    let segmentation_mask = ctx.def_arg(inst, 3)?;
    let delta = guest_lane(ctx, index)?;
    let thread = lane_id(ctx, inst)?;
    let max_thread = max_thread_id(ctx, thread, clamp, segmentation_mask)?;
    Ok([value, delta, thread, max_thread])
}

/// Returns `value` unchanged and out of bounds in stages without subgroups.
fn single_lane_shuffle(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Option<Word>> {
    if ctx.supports_subgroups() {
        return Ok(None);
    }
    let false_value = ctx.false_value;
    set_in_bounds(ctx, inst, false_value);
    Ok(Some(ctx.def_arg(inst, 0)?))
}

pub(super) fn lane_id(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    let id = thread_id(ctx)?;
    if !ctx.profile.warp_size_potentially_larger_than_guest {
        return Ok(id);
    }
    guest_lane(ctx, id)
}

pub(super) fn vote_all(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let pred = ctx.def_arg(inst, 0)?;
    if !ctx.supports_subgroups() {
        return Ok(pred);
    }
    let u1 = ctx.u1;
    if !ctx.profile.warp_size_potentially_larger_than_guest {
        let scope = subgroup_scope(ctx);
        return Ok(ctx.b.group_non_uniform_all(u1, None, scope, pred)?);
    }
    let (active, votes) = emulated_vote_masks(ctx, pred)?;
    let u32_t = ctx.u32[1];
    let lhs = ctx.b.bitwise_and(u32_t, None, votes, active)?;
    Ok(ctx.b.i_equal(u1, None, lhs, active)?)
}

pub(super) fn vote_any(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let pred = ctx.def_arg(inst, 0)?;
    if !ctx.supports_subgroups() {
        return Ok(pred);
    }
    let u1 = ctx.u1;
    if !ctx.profile.warp_size_potentially_larger_than_guest {
        let scope = subgroup_scope(ctx);
        return Ok(ctx.b.group_non_uniform_any(u1, None, scope, pred)?);
    }
    let (active, votes) = emulated_vote_masks(ctx, pred)?;
    let (u32_t, zero) = (ctx.u32[1], ctx.u32_zero_value);
    let lhs = ctx.b.bitwise_and(u32_t, None, votes, active)?;
    Ok(ctx.b.i_not_equal(u1, None, lhs, zero)?)
}

pub(super) fn vote_equal(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let pred = ctx.def_arg(inst, 0)?;
    if !ctx.supports_subgroups() {
        return Ok(ctx.true_value);
    }
    let u1 = ctx.u1;
    if !ctx.profile.warp_size_potentially_larger_than_guest {
        let scope = subgroup_scope(ctx);
        return Ok(ctx.b.group_non_uniform_all_equal(u1, None, scope, pred)?);
    }
    let (active, votes) = emulated_vote_masks(ctx, pred)?;
    let (u32_t, zero) = (ctx.u32[1], ctx.u32_zero_value);
    let lhs = ctx.b.bitwise_xor(u32_t, None, votes, active)?;
    let none = ctx.b.i_equal(u1, None, lhs, zero)?;
    let all = ctx.b.i_equal(u1, None, lhs, active)?;
    Ok(ctx.b.logical_or(u1, None, none, all)?)
}

pub(super) fn subgroup_ballot(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let pred = ctx.def_arg(inst, 0)?;
    if !ctx.supports_subgroups() {
        let (u32_t, one, zero) = (ctx.u32[1], ctx.const_u32(1), ctx.u32_zero_value);
        return Ok(ctx.b.select(u32_t, None, pred, one, zero)?);
    }
    let votes = ballot(ctx, pred)?;
    ballot_word(ctx, votes)
}

pub(super) fn subgroup_eq_mask(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    let mask = ctx.subgroup_mask_eq;
    subgroup_mask(ctx, mask, 1)
}

pub(super) fn subgroup_lt_mask(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    let mask = ctx.subgroup_mask_lt;
    subgroup_mask(ctx, mask, 0)
}

pub(super) fn subgroup_le_mask(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    let mask = ctx.subgroup_mask_le;
    subgroup_mask(ctx, mask, 1)
}

pub(super) fn subgroup_gt_mask(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    let mask = ctx.subgroup_mask_gt;
    subgroup_mask(ctx, mask, 0)
}

pub(super) fn subgroup_ge_mask(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    let mask = ctx.subgroup_mask_ge;
    subgroup_mask(ctx, mask, 1)
}

pub(super) fn shuffle_index(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    let index = ctx.def_arg(inst, 1)?;
    let clamp = ctx.def_arg(inst, 2)?;
    let segmentation_mask = ctx.def_arg(inst, 3)?;
    let (u1, u32_t) = (ctx.u1, ctx.u32[1]);
    let not_seg_mask = ctx.b.not(u32_t, None, segmentation_mask)?;
    let thread = lane_id(ctx, inst)?;
    let min_thread = ctx.b.bitwise_and(u32_t, None, thread, segmentation_mask)?;
    let clamped = ctx.b.bitwise_and(u32_t, None, clamp, not_seg_mask)?;
    let max_thread = ctx.b.bitwise_or(u32_t, None, min_thread, clamped)?;
    let lane = guest_lane(ctx, index)?;
    let lhs = ctx.b.bitwise_and(u32_t, None, lane, not_seg_mask)?;
    let src = ctx.b.bitwise_or(u32_t, None, lhs, min_thread)?;
    let in_range = ctx.b.s_less_than_equal(u1, None, src, max_thread)?;
    set_in_bounds(ctx, inst, in_range);
    if !ctx.supports_subgroups() {
        return Ok(value);
    }
    let shuffled = match (ctx.arg(inst, 1).imm_u32(), ctx.arg(inst, 3).imm_u32()) {
        (Some(lane), Some(0)) => broadcast_lane(ctx, value, lane & 31)?,
        _ => {
            let host_src = host_thread_id(ctx, src)?;
            shuffle_absolute(ctx, value, host_src)?
        }
    };
    Ok(ctx.b.select(u32_t, None, in_range, shuffled, value)?)
}

pub(super) fn shuffle_up(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    if let Some(value) = single_lane_shuffle(ctx, inst)? {
        return Ok(value);
    }
    let [value, delta, thread, max_thread] = shuffle_operands(ctx, inst)?;
    let (u1, u32_t) = (ctx.u1, ctx.u32[1]);
    let src = ctx.b.i_sub(u32_t, None, thread, delta)?;
    let in_range = ctx.b.s_greater_than_equal(u1, None, src, max_thread)?;
    set_in_bounds(ctx, inst, in_range);
    let shuffled = shuffle_relative(ctx, value, delta, src, Direction::Up)?;
    Ok(ctx.b.select(u32_t, None, in_range, shuffled, value)?)
}

pub(super) fn shuffle_down(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    if let Some(value) = single_lane_shuffle(ctx, inst)? {
        return Ok(value);
    }
    let [value, delta, thread, max_thread] = shuffle_operands(ctx, inst)?;
    let (u1, u32_t) = (ctx.u1, ctx.u32[1]);
    let src = ctx.b.i_add(u32_t, None, thread, delta)?;
    let in_range = ctx.b.s_less_than_equal(u1, None, src, max_thread)?;
    set_in_bounds(ctx, inst, in_range);
    let shuffled = shuffle_relative(ctx, value, delta, src, Direction::Down)?;
    Ok(ctx.b.select(u32_t, None, in_range, shuffled, value)?)
}

pub(super) fn shuffle_butterfly(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    if let Some(value) = single_lane_shuffle(ctx, inst)? {
        return Ok(value);
    }
    let [value, mask, thread, max_thread] = shuffle_operands(ctx, inst)?;
    let (u1, u32_t, scope) = (ctx.u1, ctx.u32[1], subgroup_scope(ctx));
    let src = ctx.b.bitwise_xor(u32_t, None, thread, mask)?;
    let in_range = ctx.b.s_less_than_equal(u1, None, src, max_thread)?;
    set_in_bounds(ctx, inst, in_range);
    let shuffled = ctx
        .b
        .group_non_uniform_shuffle_xor(u32_t, None, scope, value, mask)?;
    Ok(ctx.b.select(u32_t, None, in_range, shuffled, value)?)
}

pub(super) fn quad_broadcast(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    let lane = ctx.def_arg(inst, 1)?;
    let (u32_t, scope) = (ctx.u32[1], subgroup_scope(ctx));
    if ctx.profile.support_quad_shuffles {
        return Ok(ctx
            .b
            .group_non_uniform_quad_broadcast(u32_t, None, scope, value, lane)?);
    }
    let thread = thread_id(ctx)?;
    let quad_mask = ctx.const_u32(!3);
    let three = ctx.const_u32(3);
    let base = ctx.b.bitwise_and(u32_t, None, thread, quad_mask)?;
    let local_lane = ctx.b.bitwise_and(u32_t, None, lane, three)?;
    let src = ctx.b.bitwise_or(u32_t, None, base, local_lane)?;
    shuffle_absolute(ctx, value, src)
}

pub(super) fn quad_swap(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    let direction = ctx.def_arg(inst, 1)?;
    let (u32_t, scope) = (ctx.u32[1], subgroup_scope(ctx));
    if ctx.profile.support_quad_shuffles {
        return Ok(ctx
            .b
            .group_non_uniform_quad_swap(u32_t, None, scope, value, direction)?);
    }
    let one = ctx.const_u32(1);
    let xor_mask = ctx.b.i_add(u32_t, None, direction, one)?;
    Ok(ctx
        .b
        .group_non_uniform_shuffle_xor(u32_t, None, scope, value, xor_mask)?)
}

pub(super) fn f_swizzle_add(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let op_a = ctx.def_arg(inst, 0)?;
    let op_b = ctx.def_arg(inst, 1)?;
    let swizzle = ctx.def_arg(inst, 2)?;
    let (u32_t, f32_t) = (ctx.u32[1], ctx.f32[1]);
    let three = ctx.const_u32(3);
    let one = ctx.const_u32(1);
    let thread = thread_id(ctx)?;
    let quad_lane = ctx.b.bitwise_and(u32_t, None, thread, three)?;
    let shift = ctx.b.shift_left_logical(u32_t, None, quad_lane, one)?;
    let selector = ctx.b.shift_right_logical(u32_t, None, swizzle, shift)?;
    let mask = ctx.b.bitwise_and(u32_t, None, selector, three)?;
    let (lut_a, lut_b) = (ctx.fswzadd_lut_a, ctx.fswzadd_lut_b);
    let modifier_a = ctx.b.vector_extract_dynamic(f32_t, None, lut_a, mask)?;
    let modifier_b = ctx.b.vector_extract_dynamic(f32_t, None, lut_b, mask)?;
    let result_a = ctx.b.f_mul(f32_t, None, op_a, modifier_a)?;
    let result_b = ctx.b.f_mul(f32_t, None, op_b, modifier_b)?;
    Ok(ctx.b.f_add(f32_t, None, result_a, result_b)?)
}

pub(super) fn d_pdx_fine(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (f32_t, op) = (ctx.f32[1], ctx.def_arg(inst, 0)?);
    Ok(ctx.b.d_pdx_fine(f32_t, None, op)?)
}

pub(super) fn d_pdy_fine(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (f32_t, op) = (ctx.f32[1], ctx.def_arg(inst, 0)?);
    Ok(ctx.b.d_pdy_fine(f32_t, None, op)?)
}

pub(super) fn d_pdx_coarse(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (f32_t, op) = (ctx.f32[1], ctx.def_arg(inst, 0)?);
    Ok(ctx.b.d_pdx_coarse(f32_t, None, op)?)
}

pub(super) fn d_pdy_coarse(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (f32_t, op) = (ctx.f32[1], ctx.def_arg(inst, 0)?);
    Ok(ctx.b.d_pdy_coarse(f32_t, None, op)?)
}
