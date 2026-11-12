//! Advanced SIMD vector and scalar "x indexed element" forms.

use super::int::{dot, rounding_doubling_mla};
use super::lanes::{commit_saturation, elem, insert, int, sat, sext, write};
use super::{Exec, Undefined, bit, bits};
use crate::ThreadContext;

/// Integer by-element operations. Returns `None` for the floating-point opcodes.
pub(super) fn indexed(ctx: &mut ThreadContext, inst: u32, scalar: bool) -> Option<Exec> {
    let (q, u) = (bit(inst, 30), bit(inst, 29));
    let size = bits(inst, 23, 22);
    let opcode = bits(inst, 15, 12);
    let integer = matches!(
        (opcode, u),
        (0b0010 | 0b0110 | 0b1010 | 0b1110, _)
            | (0b0011 | 0b0111 | 0b1011 | 0b1100 | 0b1101 | 0b1000, false)
            | (0b0000 | 0b0100 | 0b1101 | 0b1111, true)
    );
    if !integer {
        return None;
    }
    let scalar_ok = matches!(
        (opcode, u),
        (0b0011 | 0b0111 | 0b1011 | 0b1100 | 0b1101, false) | (0b1101 | 0b1111, true)
    );
    let dot_product = opcode == 0b1110;
    let legal = (!scalar || scalar_ok)
        && if dot_product {
            size == 2 && !scalar
        } else {
            size == 1 || size == 2
        };
    if !legal {
        return Some(Err(Undefined));
    }
    let (h, l, mbit) = (
        bit(inst, 11) as u32,
        bit(inst, 21) as u32,
        bit(inst, 20) as u32,
    );
    let (index, m) = if size == 1 {
        ((h << 2) | (l << 1) | mbit, bits(inst, 19, 16) as usize)
    } else {
        ((h << 1) | l, bits(inst, 20, 16) as usize)
    };
    let (d, n) = (bits(inst, 4, 0) as usize, bits(inst, 9, 5) as usize);
    let esize = 8 << size;
    let (a, b, acc) = (ctx.v[n], ctx.v[m], ctx.v[d]);
    let mut saturated = false;
    let datasize = if scalar {
        esize
    } else if q {
        128
    } else {
        64
    };
    if dot_product {
        let y = elem(b, 32, index);
        let result = (0..datasize / 32).fold(0u128, |r, i| {
            insert(r, 32, i, dot(elem(acc, 32, i), elem(a, 32, i), y, u))
        });
        write(ctx, d, result, datasize);
        return Some(Ok(()));
    }
    let y = elem(b, esize, index);
    let long = matches!(opcode, 0b0010 | 0b0011 | 0b0110 | 0b0111 | 0b1010 | 0b1011);
    if long {
        let wide = 2 * esize;
        let part = if q && !scalar { 64 / esize } else { 0 };
        let elements = if scalar { 1 } else { 64 / esize };
        let iy = int(y, esize, u);
        let result = (0..elements).fold(0u128, |r, i| {
            let x = int(elem(a, esize, i + part), esize, u);
            let z = elem(acc, wide, i);
            let lane = match opcode {
                0b0010 => z.wrapping_add((x * iy) as u64),
                0b0110 => z.wrapping_sub((x * iy) as u64),
                0b1010 => (x * iy) as u64,
                0b1011 => sat(2 * x * iy, wide, false, &mut saturated),
                _ => {
                    let product = sext(sat(2 * x * iy, wide, false, &mut saturated), wide) as i128;
                    let accumulator = sext(z, wide) as i128;
                    let sum = if opcode == 0b0011 {
                        accumulator + product
                    } else {
                        accumulator - product
                    };
                    sat(sum, wide, false, &mut saturated)
                }
            };
            insert(r, wide, i, lane)
        });
        write(ctx, d, result, if scalar { wide } else { 128 });
        commit_saturation(ctx, saturated);
        return Some(Ok(()));
    }
    let result = (0..datasize / esize).fold(0u128, |r, i| {
        let x = elem(a, esize, i);
        let z = elem(acc, esize, i);
        let lane = match (opcode, u) {
            (0b1000, _) => x.wrapping_mul(y),
            (0b0000, _) => z.wrapping_add(x.wrapping_mul(y)),
            (0b0100, _) => z.wrapping_sub(x.wrapping_mul(y)),
            (0b1100 | 0b1101, false) => {
                let round = if opcode == 0b1101 {
                    1i128 << (esize - 1)
                } else {
                    0
                };
                let product = 2 * int(x, esize, false) * int(y, esize, false);
                sat((product + round) >> esize, esize, false, &mut saturated)
            }
            _ => rounding_doubling_mla(z, x, y, esize, opcode == 0b1111, &mut saturated),
        };
        insert(r, esize, i, lane)
    });
    write(ctx, d, result, datasize);
    commit_saturation(ctx, saturated);
    Some(Ok(()))
}
