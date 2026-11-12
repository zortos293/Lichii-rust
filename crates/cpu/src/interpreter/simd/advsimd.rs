//! Decoding of the Advanced SIMD vector and scalar encoding classes.

use super::{Exec, Undefined, bit, bits, indexed, int, movement, shift};
use crate::ThreadContext;

/// Bits 31:28 are `0xx0`: an Advanced SIMD vector instruction.
pub(super) fn vector(ctx: &mut ThreadContext, inst: u32) -> Exec {
    if bit(inst, 24) {
        if !bit(inst, 10) {
            return indexed::indexed(ctx, inst, false).unwrap_or(Err(Undefined));
        }
        if bit(inst, 23) {
            return Err(Undefined);
        }
        if bits(inst, 22, 19) == 0 {
            return movement::modified_immediate(ctx, inst);
        }
        return shift::shift_immediate(ctx, inst, false);
    }
    if bit(inst, 21) {
        return match bits(inst, 11, 10) {
            0b00 => int::three_different(ctx, inst, false),
            0b10 => match bits(inst, 20, 17) {
                0b0000 => int::two_misc(ctx, inst, false).unwrap_or(Err(Undefined)),
                0b1000 => int::across_lanes(ctx, inst).unwrap_or(Err(Undefined)),
                _ => Err(Undefined),
            },
            _ if bits(inst, 15, 11) < 0b11000 => int::three_same(ctx, inst, false),
            _ => Err(Undefined),
        };
    }
    if bit(inst, 10) {
        if bit(inst, 15) {
            return int::three_same_extra(ctx, inst, false);
        }
        if bits(inst, 23, 22) == 0 {
            return movement::copy(ctx, inst, false);
        }
        return Err(Undefined);
    }
    if bit(inst, 15) {
        return Err(Undefined);
    }
    match (bit(inst, 29), bit(inst, 11)) {
        (true, _) => movement::extract(ctx, inst),
        (false, true) => movement::permute(ctx, inst),
        (false, false) => movement::table_lookup(ctx, inst),
    }
}

/// Bits 31:28 are `01x1`: an Advanced SIMD scalar instruction.
pub(super) fn scalar(ctx: &mut ThreadContext, inst: u32) -> Exec {
    if bit(inst, 24) {
        if !bit(inst, 10) {
            return indexed::indexed(ctx, inst, true).unwrap_or(Err(Undefined));
        }
        if bit(inst, 23) {
            return Err(Undefined);
        }
        return shift::shift_immediate(ctx, inst, true);
    }
    if bit(inst, 21) {
        return match bits(inst, 11, 10) {
            0b00 => int::three_different(ctx, inst, true),
            0b10 => match bits(inst, 20, 17) {
                0b0000 => int::two_misc(ctx, inst, true).unwrap_or(Err(Undefined)),
                0b1000 => int::scalar_pairwise(ctx, inst).unwrap_or(Err(Undefined)),
                _ => Err(Undefined),
            },
            _ if bits(inst, 15, 11) < 0b11000 => int::three_same(ctx, inst, true),
            _ => Err(Undefined),
        };
    }
    if bit(inst, 10) {
        if bit(inst, 15) {
            return int::three_same_extra(ctx, inst, true);
        }
        if bits(inst, 23, 22) == 0 {
            return movement::copy(ctx, inst, true);
        }
    }
    Err(Undefined)
}
