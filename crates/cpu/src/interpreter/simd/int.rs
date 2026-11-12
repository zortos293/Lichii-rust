//! Advanced SIMD integer arithmetic: three same, three different, three same extra (rounding
//! doubling multiply-accumulate and dot products), two-register miscellaneous, across lanes and
//! the scalar pairwise add, in their vector and scalar forms.

use super::lanes::{
    clmul, commit_saturation, elem, insert, int, mask, sat, sext, shift_by, write, write_half,
};
use super::{Exec, Undefined, bit, bits};
use crate::ThreadContext;

fn regs(inst: u32) -> (usize, usize, usize) {
    (
        bits(inst, 4, 0) as usize,
        bits(inst, 9, 5) as usize,
        bits(inst, 20, 16) as usize,
    )
}

fn ones_if(condition: bool) -> u64 {
    if condition { u64::MAX } else { 0 }
}

/// Polynomial multiplication of two bytes, truncated to a byte (PMUL).
fn pmul8(x: u64, y: u64) -> u64 {
    clmul(x & 0xff, y & 0xff) as u64 & 0xff
}

/// Advanced SIMD three same with an integer opcode (`opcode < 0b11000`).
pub(super) fn three_same(ctx: &mut ThreadContext, inst: u32, scalar: bool) -> Exec {
    let (d, n, m) = regs(inst);
    let (q, u) = (bit(inst, 30), bit(inst, 29));
    let size = bits(inst, 23, 22);
    let opcode = bits(inst, 15, 11);
    let esize = 8 << size;
    let datasize = if scalar {
        esize
    } else if q {
        128
    } else {
        64
    };
    let (a, b, acc) = (ctx.v[n], ctx.v[m], ctx.v[d]);
    if opcode == 0b00011 {
        if scalar {
            return Err(Undefined);
        }
        let result = match (u, size) {
            (false, 0) => a & b,
            (false, 1) => a & !b,
            (false, 2) => a | b,
            (false, _) => a | !b,
            (true, 0) => a ^ b,
            (true, 1) => (acc & a) | (!acc & b),
            (true, 2) => (acc & !b) | (a & b),
            (true, _) => (acc & b) | (a & !b),
        };
        write(ctx, d, result, datasize);
        return Ok(());
    }
    let no_d_unless_q = !(size == 3 && !q);
    let legal = match opcode {
        0b00001 | 0b00101 | 0b01001 | 0b01011 => scalar || no_d_unless_q,
        0b00110 | 0b00111 | 0b01000 | 0b01010 | 0b10000 | 0b10001 => {
            if scalar {
                size == 3
            } else {
                no_d_unless_q
            }
        }
        0b00000 | 0b00010 | 0b00100 | 0b01100 | 0b01101 | 0b01110 | 0b01111 | 0b10010 | 0b10100
        | 0b10101 => !scalar && size != 3,
        0b10011 => !scalar && if u { size == 0 } else { size != 3 },
        0b10110 => size == 1 || size == 2,
        0b10111 => !scalar && !u && no_d_unless_q,
        _ => false,
    };
    if !legal {
        return Err(Undefined);
    }
    let elements = datasize / esize;
    let pairwise = matches!(opcode, 0b10100 | 0b10101 | 0b10111);
    let pair = |j: u32| {
        if j < elements {
            elem(a, esize, j)
        } else {
            elem(b, esize, j - elements)
        }
    };
    let mut saturated = false;
    let mut result = 0u128;
    for i in 0..elements {
        let (x, y) = if pairwise {
            (pair(2 * i), pair(2 * i + 1))
        } else {
            (elem(a, esize, i), elem(b, esize, i))
        };
        let z = elem(acc, esize, i);
        let (ix, iy) = (int(x, esize, u), int(y, esize, u));
        let r = match opcode {
            0b00000 => ((ix + iy) >> 1) as u64,
            0b00001 => sat(ix + iy, esize, u, &mut saturated),
            0b00010 => ((ix + iy + 1) >> 1) as u64,
            0b00100 => ((ix - iy) >> 1) as u64,
            0b00101 => sat(ix - iy, esize, u, &mut saturated),
            0b00110 => ones_if(ix > iy),
            0b00111 => ones_if(ix >= iy),
            0b01000..=0b01011 => shift_by(
                ix,
                y as i8 as i32,
                esize,
                u,
                opcode & 0b10 != 0,
                opcode & 1 != 0,
                &mut saturated,
            ),
            0b01100 | 0b10100 => ix.max(iy) as u64,
            0b01101 | 0b10101 => ix.min(iy) as u64,
            0b01110 => (ix - iy).unsigned_abs() as u64,
            0b01111 => z.wrapping_add((ix - iy).unsigned_abs() as u64),
            0b10000 if u => x.wrapping_sub(y),
            0b10000 | 0b10111 => x.wrapping_add(y),
            0b10001 if u => ones_if(x == y),
            0b10001 => ones_if(x & y != 0),
            0b10010 if u => z.wrapping_sub(x.wrapping_mul(y)),
            0b10010 => z.wrapping_add(x.wrapping_mul(y)),
            0b10011 if u => pmul8(x, y),
            0b10011 => x.wrapping_mul(y),
            _ => {
                let (sx, sy) = (int(x, esize, false), int(y, esize, false));
                let round = if u { 1i128 << (esize - 1) } else { 0 };
                sat((2 * sx * sy + round) >> esize, esize, false, &mut saturated)
            }
        };
        result = insert(result, esize, i, r);
    }
    write(ctx, d, result, datasize);
    commit_saturation(ctx, saturated);
    Ok(())
}

/// Advanced SIMD three different (long, wide and narrowing high-half forms).
pub(super) fn three_different(ctx: &mut ThreadContext, inst: u32, scalar: bool) -> Exec {
    let (d, n, m) = regs(inst);
    let (q, u) = (bit(inst, 30), bit(inst, 29));
    let size = bits(inst, 23, 22);
    let opcode = bits(inst, 15, 12);
    let esize = 8 << size;
    let legal = match opcode {
        0b1110 => !scalar && !u && (size == 0 || size == 3),
        0b1001 | 0b1011 | 0b1101 => !u && (size == 1 || size == 2),
        0b1111 => false,
        _ => !scalar && size != 3,
    };
    if !legal {
        return Err(Undefined);
    }
    let (a, b, acc) = (ctx.v[n], ctx.v[m], ctx.v[d]);
    let part = if q && !scalar { 64 / esize } else { 0 };
    if opcode == 0b1110 {
        let result = if size == 3 {
            clmul((a >> (part * 64)) as u64, (b >> (part * 64)) as u64)
        } else {
            (0..8).fold(0u128, |r, i| {
                insert(
                    r,
                    16,
                    i,
                    clmul(elem(a, 8, i + part), elem(b, 8, i + part)) as u64,
                )
            })
        };
        ctx.v[d] = result;
        return Ok(());
    }
    let elements = if scalar { 1 } else { 64 / esize };
    let wide = 2 * esize;
    if opcode == 0b0100 || opcode == 0b0110 {
        let round = if u { 1u64 << (esize - 1) } else { 0 };
        let narrowed = (0..elements).fold(0u128, |r, i| {
            let (x, y) = (elem(a, wide, i), elem(b, wide, i));
            let sum = if opcode == 0b0110 {
                x.wrapping_sub(y)
            } else {
                x.wrapping_add(y)
            };
            let sum = (sum & mask(wide)) as u128 + round as u128;
            insert(r, esize, i, (sum >> esize) as u64)
        });
        write_half(ctx, d, narrowed as u64, q);
        return Ok(());
    }
    let widened_first = matches!(opcode, 0b0001 | 0b0011);
    let mut saturated = false;
    let mut result = 0u128;
    for i in 0..elements {
        let x = if widened_first {
            int(elem(a, wide, i), wide, u)
        } else {
            int(elem(a, esize, i + part), esize, u)
        };
        let y = int(elem(b, esize, i + part), esize, u);
        let z = elem(acc, wide, i);
        let r = match opcode {
            0b0000 | 0b0001 => (x + y) as u64,
            0b0010 | 0b0011 => (x - y) as u64,
            0b0101 => z.wrapping_add((x - y).unsigned_abs() as u64),
            0b0111 => (x - y).unsigned_abs() as u64,
            0b1000 => z.wrapping_add((x * y) as u64),
            0b1010 => z.wrapping_sub((x * y) as u64),
            0b1100 => (x * y) as u64,
            0b1101 => sat(2 * x * y, wide, false, &mut saturated),
            _ => {
                let product = sext(sat(2 * x * y, wide, false, &mut saturated), wide) as i128;
                let accumulator = sext(z, wide) as i128;
                let sum = if opcode == 0b1001 {
                    accumulator + product
                } else {
                    accumulator - product
                };
                sat(sum, wide, false, &mut saturated)
            }
        };
        result = insert(result, wide, i, r);
    }
    write(ctx, d, result, if scalar { wide } else { 128 });
    commit_saturation(ctx, saturated);
    Ok(())
}

/// `SQRDMLAH`/`SQRDMLSH` of one lane.
pub(super) fn rounding_doubling_mla(
    accumulator: u64,
    x: u64,
    y: u64,
    esize: u32,
    subtract: bool,
    saturated: &mut bool,
) -> u64 {
    let product = 2 * int(x, esize, false) * int(y, esize, false);
    let product = if subtract { -product } else { product };
    let sum = (int(accumulator, esize, false) << esize) + product + (1i128 << (esize - 1));
    sat(sum >> esize, esize, false, saturated)
}

/// Four-way 8-bit dot product accumulated into a 32-bit lane.
pub(super) fn dot(accumulator: u64, x: u64, y: u64, unsigned: bool) -> u64 {
    (0..4).fold(accumulator, |acc, k| {
        let product = int(x >> (8 * k), 8, unsigned) * int(y >> (8 * k), 8, unsigned);
        acc.wrapping_add(product as u64)
    }) & mask(32)
}

/// Advanced SIMD three same extra: SQRDMLAH, SQRDMLSH, SDOT and UDOT. The complex FP forms are
/// handled with the rest of the FP vector instructions.
pub(super) fn three_same_extra(ctx: &mut ThreadContext, inst: u32, scalar: bool) -> Exec {
    let (d, n, m) = regs(inst);
    let (q, u) = (bit(inst, 30), bit(inst, 29));
    let size = bits(inst, 23, 22);
    let opcode = bits(inst, 14, 11);
    let esize = 8 << size;
    let datasize = if scalar {
        esize
    } else if q {
        128
    } else {
        64
    };
    let (a, b, acc) = (ctx.v[n], ctx.v[m], ctx.v[d]);
    let mut saturated = false;
    let result = match opcode {
        0b0000 | 0b0001 if u && (size == 1 || size == 2) => {
            (0..datasize / esize).fold(0, |r, i| {
                let lane = rounding_doubling_mla(
                    elem(acc, esize, i),
                    elem(a, esize, i),
                    elem(b, esize, i),
                    esize,
                    opcode == 1,
                    &mut saturated,
                );
                insert(r, esize, i, lane)
            })
        }
        0b0010 if !scalar && size == 2 => (0..datasize / 32).fold(0, |r, i| {
            let lane = dot(elem(acc, 32, i), elem(a, 32, i), elem(b, 32, i), u);
            insert(r, 32, i, lane)
        }),
        _ => return Err(Undefined),
    };
    write(ctx, d, result, datasize);
    commit_saturation(ctx, saturated);
    Ok(())
}

/// Advanced SIMD two-register miscellaneous with an integer opcode. Returns `None` for the
/// floating-point opcodes.
pub(super) fn two_misc(ctx: &mut ThreadContext, inst: u32, scalar: bool) -> Option<Exec> {
    let (d, n, _) = regs(inst);
    let (q, u) = (bit(inst, 30), bit(inst, 29));
    let size = bits(inst, 23, 22);
    let opcode = bits(inst, 16, 12);
    let esize = 8 << size;
    let datasize = if scalar {
        esize
    } else if q {
        128
    } else {
        64
    };
    let no_d_unless_q = !(size == 3 && !q);
    let legal = match (opcode, u) {
        (0b00000, false) => !scalar && size < 3,
        (0b00000, true) => !scalar && size < 2,
        (0b00001, false) | (0b00101, false) => !scalar && size == 0,
        (0b00101, true) => !scalar && size < 2,
        (0b00010 | 0b00110 | 0b00100, _) => !scalar && size < 3,
        (0b00011 | 0b00111, _) => scalar || no_d_unless_q,
        (0b01000 | 0b01001, _) | (0b01010 | 0b01011, false) | (0b01011, true) => {
            if scalar {
                size == 3
            } else {
                no_d_unless_q
            }
        }
        (0b10010, false) => !scalar && size < 3,
        (0b10010, true) | (0b10100, _) => size < 3,
        (0b10011, true) => !scalar && size < 3,
        (0b01010, true) => false,
        _ => return None,
    };
    if !legal {
        return Some(Err(Undefined));
    }
    let a = ctx.v[n];
    let acc = ctx.v[d];
    let mut saturated = false;
    let elements = datasize / esize;
    let lanewise = |f: &mut dyn FnMut(u64, u64) -> u64| {
        (0..elements).fold(0u128, |r, i| {
            insert(r, esize, i, f(elem(a, esize, i), elem(acc, esize, i)))
        })
    };
    let result = match (opcode, u) {
        (0b00000 | 0b00001, _) => {
            let container = match (opcode, u) {
                (0b00000, false) => 64,
                (0b00000, true) => 32,
                _ => 16,
            };
            let per = container / esize;
            (0..elements).fold(0u128, |r, i| {
                let reversed = (i / per) * per + (per - 1 - i % per);
                insert(r, esize, i, elem(a, esize, reversed))
            })
        }
        (0b00010 | 0b00110, _) => {
            let wide = 2 * esize;
            (0..datasize / wide).fold(0u128, |r, i| {
                let sum =
                    int(elem(a, esize, 2 * i), esize, u) + int(elem(a, esize, 2 * i + 1), esize, u);
                let base = if opcode == 0b00110 {
                    elem(acc, wide, i)
                } else {
                    0
                };
                insert(r, wide, i, base.wrapping_add(sum as u64))
            })
        }
        (0b00011, _) => lanewise(&mut |x, z| {
            sat(
                int(z, esize, u) + int(x, esize, !u),
                esize,
                u,
                &mut saturated,
            )
        }),
        (0b00100, false) => lanewise(&mut |x, _| {
            let signs = ((x >> 1) ^ x) & mask(esize - 1);
            (signs.leading_zeros() - (64 - (esize - 1))) as u64
        }),
        (0b00100, true) => lanewise(&mut |x, _| (x.leading_zeros() - (64 - esize)) as u64),
        (0b00101, false) => lanewise(&mut |x, _| x.count_ones() as u64),
        (0b00101, true) if size == 0 => !a,
        (0b00101, true) => (0..datasize / 8).fold(0u128, |r, i| {
            insert(r, 8, i, (elem(a, 8, i) as u8).reverse_bits() as u64)
        }),
        (0b00111, false) => {
            lanewise(&mut |x, _| sat(int(x, esize, false).abs(), esize, false, &mut saturated))
        }
        (0b00111, true) => {
            lanewise(&mut |x, _| sat(-int(x, esize, false), esize, false, &mut saturated))
        }
        (0b01000, false) => lanewise(&mut |x, _| ones_if(int(x, esize, false) > 0)),
        (0b01000, true) => lanewise(&mut |x, _| ones_if(int(x, esize, false) >= 0)),
        (0b01001, false) => lanewise(&mut |x, _| ones_if(x == 0)),
        (0b01001, true) => lanewise(&mut |x, _| ones_if(int(x, esize, false) <= 0)),
        (0b01010, _) => lanewise(&mut |x, _| ones_if(int(x, esize, false) < 0)),
        (0b01011, false) => lanewise(&mut |x, _| int(x, esize, false).unsigned_abs() as u64),
        (0b01011, true) => lanewise(&mut |x, _| x.wrapping_neg()),
        (0b10011, _) => {
            let wide = 2 * esize;
            let part = if q { 64 / esize } else { 0 };
            let result = (0..64 / esize).fold(0u128, |r, i| {
                insert(r, wide, i, elem(a, esize, i + part) << esize)
            });
            ctx.v[d] = result;
            return Some(Ok(()));
        }
        _ => {
            let wide = 2 * esize;
            let count = if scalar { 1 } else { 64 / esize };
            let narrowed = (0..count).fold(0u128, |r, i| {
                let x = elem(a, wide, i);
                let lane = match (opcode, u) {
                    (0b10010, false) => x,
                    (0b10010, true) => sat(int(x, wide, false), esize, true, &mut saturated),
                    (_, false) => sat(int(x, wide, false), esize, false, &mut saturated),
                    (_, true) => sat(int(x, wide, true), esize, true, &mut saturated),
                };
                insert(r, esize, i, lane)
            });
            if scalar {
                write(ctx, d, narrowed, esize);
            } else {
                write_half(ctx, d, narrowed as u64, q);
            }
            commit_saturation(ctx, saturated);
            return Some(Ok(()));
        }
    };
    write(ctx, d, result, datasize);
    commit_saturation(ctx, saturated);
    Some(Ok(()))
}

/// Advanced SIMD across lanes with an integer opcode. Returns `None` for the FP opcodes.
pub(super) fn across_lanes(ctx: &mut ThreadContext, inst: u32) -> Option<Exec> {
    let (d, n, _) = regs(inst);
    let (q, u) = (bit(inst, 30), bit(inst, 29));
    let size = bits(inst, 23, 22);
    let opcode = bits(inst, 16, 12);
    if !matches!(
        (opcode, u),
        (0b00011 | 0b01010 | 0b11010, _) | (0b11011, false)
    ) {
        return None;
    }
    if size == 3 || (size == 2 && !q) {
        return Some(Err(Undefined));
    }
    let esize = 8 << size;
    let elements = if q { 128 } else { 64 } / esize;
    let a = ctx.v[n];
    let lanes = (0..elements).map(|i| int(elem(a, esize, i), esize, u));
    let (value, width) = match opcode {
        0b00011 => (lanes.sum::<i128>(), 2 * esize),
        0b01010 => (lanes.max().unwrap_or(0), esize),
        0b11010 => (lanes.min().unwrap_or(0), esize),
        _ => (lanes.sum::<i128>(), esize),
    };
    write(ctx, d, value as u128, width);
    Some(Ok(()))
}

/// Scalar pairwise ADDP; the FP pairwise forms return `None`.
pub(super) fn scalar_pairwise(ctx: &mut ThreadContext, inst: u32) -> Option<Exec> {
    let (d, n, _) = regs(inst);
    if bits(inst, 16, 12) != 0b11011 || bit(inst, 29) {
        return None;
    }
    if bits(inst, 23, 22) != 3 {
        return Some(Err(Undefined));
    }
    let a = ctx.v[n];
    write(
        ctx,
        d,
        (a as u64).wrapping_add((a >> 64) as u64) as u128,
        64,
    );
    Some(Ok(()))
}
