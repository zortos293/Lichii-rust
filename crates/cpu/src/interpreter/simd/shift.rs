//! Advanced SIMD shift by immediate, vector and scalar, including the fixed-point conversions.

use super::lanes::{commit_saturation, elem, insert, int, mask, sat, write, write_half};
use super::softfloat::{self as sf, FpEnv, Rounding};
use super::{Exec, Undefined, bit, bits, commit_flags};
use crate::ThreadContext;

pub(super) fn shift_immediate(ctx: &mut ThreadContext, inst: u32, scalar: bool) -> Exec {
    let (d, n) = (bits(inst, 4, 0) as usize, bits(inst, 9, 5) as usize);
    let (q, u) = (bit(inst, 30), bit(inst, 29));
    let immh = bits(inst, 22, 19);
    let immhb = bits(inst, 22, 16);
    let opcode = bits(inst, 15, 11);
    if immh == 0 {
        return Err(Undefined);
    }
    let esize = 8 << (31 - immh.leading_zeros());
    let datasize = if scalar {
        esize
    } else if q {
        128
    } else {
        64
    };
    let right = 2 * esize - immhb;
    let left = immhb - esize;
    let a = ctx.v[n];
    let acc = ctx.v[d];
    let mut saturated = false;
    match opcode {
        0b10000..=0b10011 => return narrowing(ctx, inst, scalar, esize, immhb),
        0b10100 => {
            if scalar || esize == 64 {
                return Err(Undefined);
            }
            let part = if q { 64 / esize } else { 0 };
            let wide = 2 * esize;
            ctx.v[d] = (0..64 / esize).fold(0u128, |r, i| {
                insert(
                    r,
                    wide,
                    i,
                    (int(elem(a, esize, i + part), esize, u) << left) as u64,
                )
            });
            return Ok(());
        }
        0b11100 | 0b11111 => return fixed_point(ctx, inst, scalar, esize, right),
        _ => {}
    }
    let legal = match (opcode, u) {
        (0b00000 | 0b00010 | 0b00100 | 0b00110 | 0b01010, _) | (0b01000, true) => {
            if scalar {
                esize == 64
            } else {
                esize < 64 || q
            }
        }
        (0b01110, _) | (0b01100, true) => scalar || esize < 64 || q,
        _ => false,
    };
    if !legal {
        return Err(Undefined);
    }
    let elements = datasize / esize;
    let result = (0..elements).fold(0u128, |r, i| {
        let x = elem(a, esize, i);
        let z = elem(acc, esize, i);
        let lane = match opcode {
            0b00000..=0b00110 => {
                let round = if opcode & 0b100 != 0 {
                    1i128 << (right - 1)
                } else {
                    0
                };
                let shifted = ((int(x, esize, u) + round) >> right) as u64;
                if opcode & 0b010 != 0 {
                    z.wrapping_add(shifted)
                } else {
                    shifted
                }
            }
            0b01000 => {
                let keep = if right == esize {
                    mask(esize)
                } else {
                    !(mask(esize) >> right)
                };
                (z & keep) | if right == esize { 0 } else { x >> right }
            }
            0b01010 if u => (z & !(mask(esize) << left)) | (x << left),
            0b01010 => x << left,
            0b01100 => sat(int(x, esize, false) << left, esize, true, &mut saturated),
            _ => sat(int(x, esize, u) << left, esize, u, &mut saturated),
        };
        insert(r, esize, i, lane)
    });
    write(ctx, d, result, datasize);
    commit_saturation(ctx, saturated);
    Ok(())
}

/// SHRN, RSHRN and the saturating narrowing right shifts.
fn narrowing(ctx: &mut ThreadContext, inst: u32, scalar: bool, esize: u32, immhb: u32) -> Exec {
    let (d, n) = (bits(inst, 4, 0) as usize, bits(inst, 9, 5) as usize);
    let (q, u) = (bit(inst, 30), bit(inst, 29));
    let opcode = bits(inst, 15, 11);
    if esize == 64 || (scalar && !u && opcode < 0b10010) {
        return Err(Undefined);
    }
    let shift = 2 * esize - immhb;
    let wide = 2 * esize;
    let round = if opcode & 1 != 0 {
        1i128 << (shift - 1)
    } else {
        0
    };
    let source_unsigned = u && opcode >= 0b10010;
    let a = ctx.v[n];
    let mut saturated = false;
    let count = if scalar { 1 } else { 64 / esize };
    let narrowed = (0..count).fold(0u128, |r, i| {
        let value = (int(elem(a, wide, i), wide, source_unsigned) + round) >> shift;
        let lane = match (opcode >> 1 & 1, u) {
            (0, false) => value as u64,
            (0, true) => sat(value, esize, true, &mut saturated),
            (_, unsigned) => sat(value, esize, unsigned, &mut saturated),
        };
        insert(r, esize, i, lane)
    });
    if scalar {
        write(ctx, d, narrowed, esize);
    } else {
        write_half(ctx, d, narrowed as u64, q);
    }
    commit_saturation(ctx, saturated);
    Ok(())
}

/// SCVTF, UCVTF, FCVTZS and FCVTZU with `fbits` fraction bits.
fn fixed_point(ctx: &mut ThreadContext, inst: u32, scalar: bool, esize: u32, fbits: u32) -> Exec {
    let q = bit(inst, 30);
    if esize == 8 || (!scalar && esize == 64 && !q) {
        return Err(Undefined);
    }
    match esize {
        16 => fixed_point_lanes::<16>(ctx, inst, scalar, fbits),
        32 => fixed_point_lanes::<32>(ctx, inst, scalar, fbits),
        _ => fixed_point_lanes::<64>(ctx, inst, scalar, fbits),
    }
}

fn fixed_point_lanes<const N: u32>(
    ctx: &mut ThreadContext,
    inst: u32,
    scalar: bool,
    fbits: u32,
) -> Exec {
    let (d, n) = (bits(inst, 4, 0) as usize, bits(inst, 9, 5) as usize);
    let unsigned = bit(inst, 29);
    let to_float = bits(inst, 15, 11) == 0b11100;
    let datasize = if scalar {
        N
    } else if bit(inst, 30) {
        128
    } else {
        64
    };
    let a = ctx.v[n];
    let mut env = FpEnv::new(ctx.fpcr);
    let rounding = env.rounding();
    let result = (0..datasize / N).fold(0u128, |r, i| {
        let x = elem(a, N, i);
        let lane = if to_float {
            sf::from_fixed::<N>(x, !unsigned, N, fbits, rounding, &mut env)
        } else {
            sf::to_fixed::<N>(x, fbits, unsigned, Rounding::Zero, N, &mut env)
        };
        insert(r, N, i, lane)
    });
    write(ctx, d, result, datasize);
    commit_flags(ctx, &env);
    Ok(())
}
