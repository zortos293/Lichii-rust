//! Votes, shuffles, matches and reductions across the warp.
//!
//! The reductions and matches over every lane read each lane with an index shuffle and mask
//! out the inactive ones with the ballot of the executing lanes, as the C++ lowering does.

use super::{TResult, Translator};
use crate::ShaderStage;
use crate::ampere::decode::Form;
use crate::ampere::opcode::Opcode;
use crate::ir::{ICompare, Opcode as Ir, Value};

/// Clamp 3 with a segment mask of 28: shuffles confined to a quad.
const QUAD_MASK: u32 = (28 << 8) | 3;

pub(super) fn translate(t: &mut Translator<'_>, op: Opcode) -> TResult {
    match op {
        Opcode::Vote => vote(t, false),
        Opcode::Voteu => vote(t, true),
        Opcode::Shfl => shfl(t),
        Opcode::Match => match_lanes(t),
        Opcode::Redux => redux(t),
        _ => Err(t.unsupported()),
    }
}

/// The ballot of the lanes executing the instruction.
fn active_lanes(t: &mut Translator<'_>) -> Value {
    t.ir.inst(Ir::SubgroupBallot, &[Value::U1(true)])
}

/// `value` as lane `lane` of the whole warp holds it.
fn read_lane(t: &mut Translator<'_>, value: Value, lane: Value) -> Value {
    t.ir.inst(
        Ir::ShuffleIndex,
        &[value, lane, Value::U32(31), Value::U32(0)],
    )
}

/// The lowest lane set in `lanes`.
fn lowest_lane(t: &mut Translator<'_>, lanes: Value) -> Value {
    let negated = t.ir.ineg(lanes);
    let lowest = t.ir.bitwise_and(lanes, negated);
    t.ir.find_u_msb(lowest)
}

/// `value` as the lowest active lane holds it.
pub(super) fn broadcast_first_active(t: &mut Translator<'_>, value: Value) -> Value {
    let active = active_lanes(t);
    let lane = lowest_lane(t, active);
    read_lane(t, value, lane)
}

/// Whether lane `lane` is set in `lanes`.
fn lane_set(t: &mut Translator<'_>, lanes: Value, lane: u32) -> Value {
    let bit = t.ir.bitwise_and(lanes, Value::U32(1 << lane));
    t.ir.icompare(ICompare::NotEqual, bit, Value::U32(0))
}

/// VOTE and VOTEU: the ALL, ANY or EQ vote (72..74) of the predicate at 87 into the predicate
/// at 81 and its ballot into the destination. VOTEU writes a uniform register and predicate.
fn vote(t: &mut Translator<'_>, uniform: bool) -> TResult {
    let insn = t.insn;
    let op = match insn.bits(72, 2) {
        0 => Ir::VoteAll,
        1 => Ir::VoteAny,
        2 => Ir::VoteEqual,
        _ => return Err(t.unsupported()),
    };
    let pred = t.pred(insn.pred_c());
    let (dst, dst_pred) = if uniform {
        (insn.udst(), insn.upred_at(81))
    } else {
        (insn.dst(), insn.pred_dst0())
    };
    if !dst_pred.is_true() {
        let result = t.ir.inst(op, &[pred]);
        t.ir.set_pred(dst_pred, result);
    }
    if !dst.is_zero() {
        let ballot = t.ir.inst(Ir::SubgroupBallot, &[pred]);
        t.set_x(dst, ballot);
    }
    Ok(())
}

/// SHFL: reads `a` from another lane in IDX, UP, DOWN or BFLY mode (58..60). The lane is a
/// register at 32 or an immediate at 53..58, the clamp and segment mask a register at 64 or an
/// immediate at 40..53; the predicate at 81 is whether the lane was in bounds. Fragment
/// shuffles inside a quad become quad operations.
fn shfl(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let lane = match t.form {
        Form::RegReg | Form::RegImm => {
            let reg = insn.src1();
            t.x(reg)
        }
        Form::ImmReg | Form::ImmImm => Value::U32(insn.bits(53, 5) as u32),
        _ => return Err(t.unsupported()),
    };
    let mask = match t.form {
        Form::RegReg | Form::ImmReg => {
            let reg = insn.src2();
            t.x(reg)
        }
        _ => Value::U32(insn.bits(40, 13) as u32),
    };
    let mode = insn.bits(58, 2);
    let in_bounds = insn.pred_dst0();
    let value = t.src_a();
    let dst = t.reg_d();
    if let (Value::U32(lane), Value::U32(QUAD_MASK)) = (lane, mask)
        && t.env.stage() == ShaderStage::Fragment
    {
        let quad = match (mode, lane) {
            (0, 0..=3) => Some((Ir::QuadBroadcast, lane)),
            (3, 1..=3) => Some((Ir::QuadSwap, lane - 1)),
            _ => None,
        };
        if let Some((op, operand)) = quad {
            let result = t.ir.inst(op, &[value, Value::U32(operand)]);
            t.set_x(dst, result);
            t.ir.set_pred(in_bounds, Value::U1(true));
            return Ok(());
        }
    }
    let (clamp, segment_mask) = match mask {
        Value::U32(mask) => (Value::U32(mask & 0x1f), Value::U32((mask >> 8) & 0x1f)),
        mask => (
            t.ir.bit_field_extract(mask, Value::U32(0), Value::U32(5), false),
            t.ir.bit_field_extract(mask, Value::U32(8), Value::U32(5), false),
        ),
    };
    let op = match mode {
        0 => Ir::ShuffleIndex,
        1 => Ir::ShuffleUp,
        2 => Ir::ShuffleDown,
        _ => Ir::ShuffleButterfly,
    };
    let result = t.ir.inst(op, &[value, lane, clamp, segment_mask]);
    if !in_bounds.is_true() {
        let flag = t.ir.get_in_bounds_from_op(result);
        t.ir.set_pred(in_bounds, flag);
    }
    t.set_x(dst, result);
    Ok(())
}

/// Whether lane `lane` holds the same 32-bit value, or register pair, as this one.
fn same_as_lane(t: &mut Translator<'_>, words: &[Value], lane: Value) -> Value {
    let mut equal = Value::U1(true);
    for &word in words {
        let other = read_lane(t, word, lane);
        let same = t.ir.icompare(ICompare::Equal, other, word);
        equal = match equal {
            Value::U1(true) => same,
            equal => t.ir.logical_and(equal, same),
        };
    }
    equal
}

/// MATCH: the active lanes holding the same value as this one (.ANY, 79), or with .ALL every
/// active lane when they all agree and none otherwise, that agreement going to the predicate
/// at 81. Bit 73 compares register pairs.
fn match_lanes(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let src = insn.src0();
    let wide = insn.bit(73);
    if wide && !(src.is_zero() || src.0.is_multiple_of(2)) {
        return Err(t.unsupported());
    }
    let low = t.x(src);
    let high = if wide {
        t.x(src.offset(1))
    } else {
        Value::Void
    };
    let words = [low, high];
    let words = &words[..1 + usize::from(wide)];
    let active = active_lanes(t);
    let mask = if insn.bit(79) {
        let mut mask = Value::U32(0);
        for lane in 0..32 {
            let same = same_as_lane(t, words, Value::U32(lane));
            let bit = t.ir.select(same, Value::U32(1 << lane), Value::U32(0));
            mask = match mask {
                Value::U32(0) => bit,
                mask => t.ir.bitwise_or(mask, bit),
            };
        }
        t.ir.bitwise_and(mask, active)
    } else {
        let first = lowest_lane(t, active);
        let same = same_as_lane(t, words, first);
        let all = t.ir.inst(Ir::VoteAll, &[same]);
        t.ir.set_pred(insn.pred_dst0(), all);
        t.ir.select(all, active, Value::U32(0))
    };
    let dst = insn.dst();
    t.set_x(dst, mask);
    Ok(())
}

/// REDUX: the AND, OR, XOR, SUM, MIN or MAX (78..81; MIN and MAX signed with 73) of a register
/// over the active lanes, into a uniform register.
fn redux(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let op = insn.bits(78, 3);
    let signed = insn.bit(73);
    if op > 5 || (signed && op < 4) {
        return Err(t.unsupported());
    }
    let identity = match (op, signed) {
        (0, _) | (4, false) => u32::MAX,
        (4, true) => i32::MAX as u32,
        (5, true) => i32::MIN as u32,
        _ => 0,
    };
    let value = t.src_a();
    let active = active_lanes(t);
    let mut result = Value::Void;
    for lane in 0..32 {
        let lane_value = read_lane(t, value, Value::U32(lane));
        let set = lane_set(t, active, lane);
        let term = t.ir.select(set, lane_value, Value::U32(identity));
        result = match (result, op) {
            (Value::Void, _) => term,
            (acc, 0) => t.ir.bitwise_and(acc, term),
            (acc, 1) => t.ir.bitwise_or(acc, term),
            (acc, 2) => t.ir.bitwise_xor(acc, term),
            (acc, 3) => t.ir.iadd(acc, term),
            (acc, 4) => t.ir.imin(acc, term, signed),
            (acc, _) => t.ir.imax(acc, term, signed),
        };
    }
    let dst = insn.udst();
    t.set_x(dst, result);
    Ok(())
}

#[cfg(test)]
#[path = "warp_tests.rs"]
mod tests;
