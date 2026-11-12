//! Lane access and saturation shared by the Advanced SIMD instructions. Lane `i` of `esize` bits
//! is bits `[i * esize, (i + 1) * esize)` of the 128-bit register.

use crate::ThreadContext;

pub(super) const QC: u32 = 1 << 27;

pub(super) fn mask(esize: u32) -> u64 {
    u64::MAX >> (64 - esize)
}

pub(super) fn elem(v: u128, esize: u32, i: u32) -> u64 {
    (v >> (i * esize)) as u64 & mask(esize)
}

pub(super) fn sext(x: u64, esize: u32) -> i64 {
    ((x << (64 - esize)) as i64) >> (64 - esize)
}

/// The lane as a signed or unsigned integer.
pub(super) fn int(x: u64, esize: u32, unsigned: bool) -> i128 {
    if unsigned {
        (x & mask(esize)) as i128
    } else {
        sext(x, esize) as i128
    }
}

pub(super) fn insert(v: u128, esize: u32, i: u32, x: u64) -> u128 {
    let lane_mask = (mask(esize) as u128) << (i * esize);
    (v & !lane_mask) | (((x & mask(esize)) as u128) << (i * esize))
}

/// Replicates the low `esize` bits of `x` across 128 bits.
pub(super) fn replicate(x: u64, esize: u32) -> u128 {
    let mut v = (x & mask(esize)) as u128;
    let mut width = esize;
    while width < 128 {
        v |= v << width;
        width *= 2;
    }
    v
}

/// `SatQ`: clamps to the `esize`-bit signed or unsigned range and records saturation.
pub(super) fn sat(x: i128, esize: u32, unsigned: bool, saturated: &mut bool) -> u64 {
    let (lo, hi) = if unsigned {
        (0, (1i128 << esize) - 1)
    } else {
        (-(1i128 << (esize - 1)), (1i128 << (esize - 1)) - 1)
    };
    let clamped = x.clamp(lo, hi);
    *saturated |= clamped != x;
    clamped as u64 & mask(esize)
}

/// Writes `datasize` bits of `value` to `Vd`, clearing the rest of the register.
pub(super) fn write(ctx: &mut ThreadContext, d: usize, value: u128, datasize: u32) {
    ctx.v[d] = if datasize == 128 {
        value
    } else {
        value & ((1 << datasize) - 1)
    };
}

/// `Vpart[d, part] = value`: the low half clears the high one, the high half keeps the low one.
pub(super) fn write_half(ctx: &mut ThreadContext, d: usize, value: u64, high: bool) {
    ctx.v[d] = if high {
        (ctx.v[d] as u64 as u128) | ((value as u128) << 64)
    } else {
        value as u128
    };
}

pub(super) fn commit_saturation(ctx: &mut ThreadContext, saturated: bool) {
    if saturated {
        ctx.fpsr |= QC;
    }
}

/// Carry-less multiplication.
pub(super) fn clmul(x: u64, y: u64) -> u128 {
    let mut r = 0u128;
    for i in 0..64 {
        if (y >> i) & 1 != 0 {
            r ^= (x as u128) << i;
        }
    }
    r
}

/// Shifts by a signed register amount as SSHL/SRSHL/SQSHL/SQRSHL and their unsigned forms do:
/// left for positive amounts, right (optionally rounding) for negative ones.
pub(super) fn shift_by(
    x: i128,
    shift: i32,
    esize: u32,
    unsigned: bool,
    rounding: bool,
    saturating: bool,
    saturated: &mut bool,
) -> u64 {
    let value = if shift >= 0 {
        if shift >= 64 {
            if x == 0 {
                return 0;
            }
            if !saturating {
                return 0;
            }
            x.signum() << 100
        } else {
            x << shift
        }
    } else {
        let s = (-shift).min(100) as u32;
        let round = if rounding { 1i128 << (s - 1) } else { 0 };
        (x + round) >> s
    };
    if saturating {
        sat(value, esize, unsigned, saturated)
    } else {
        value as u64 & mask(esize)
    }
}
