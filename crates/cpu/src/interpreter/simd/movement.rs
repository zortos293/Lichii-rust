//! Advanced SIMD data movement: copy (DUP, INS, SMOV, UMOV), permute (UZP, TRN, ZIP), EXT,
//! table lookup (TBL, TBX) and modified immediate (MOVI, MVNI, ORR, BIC, FMOV).

use super::lanes::{elem, insert, mask, replicate, sext, write};
use super::scalar_fp::expand_imm;
use super::{Exec, Undefined, bit, bits, read_x, write_x};
use crate::ThreadContext;

fn regs(inst: u32) -> (usize, usize, usize) {
    (
        bits(inst, 4, 0) as usize,
        bits(inst, 9, 5) as usize,
        bits(inst, 20, 16) as usize,
    )
}

fn datasize(q: bool) -> u32 {
    if q { 128 } else { 64 }
}

pub(super) fn copy(ctx: &mut ThreadContext, inst: u32, scalar: bool) -> Exec {
    let (d, n, _) = regs(inst);
    let (q, op) = (bit(inst, 30), bit(inst, 29));
    let imm5 = bits(inst, 20, 16);
    let imm4 = bits(inst, 14, 11);
    let size = imm5.trailing_zeros();
    if size > 3 {
        return Err(Undefined);
    }
    let esize = 8 << size;
    let index = imm5 >> (size + 1);
    let a = ctx.v[n];
    if scalar {
        if op || imm4 != 0 {
            return Err(Undefined);
        }
        write(ctx, d, elem(a, esize, index) as u128, esize);
        return Ok(());
    }
    match (op, imm4) {
        (false, 0b0000 | 0b0001) => {
            if size == 3 && !q {
                return Err(Undefined);
            }
            let value = if imm4 == 0 {
                elem(a, esize, index)
            } else {
                read_x(ctx, n as u32)
            };
            write(ctx, d, replicate(value, esize), datasize(q));
        }
        (false, 0b0011) if q => {
            ctx.v[d] = insert(ctx.v[d], esize, index, read_x(ctx, n as u32));
        }
        (false, 0b0101) if size < 2 || (q && size == 2) => {
            let value = sext(elem(a, esize, index), esize) as u64;
            write_x(ctx, d as u32, if q { value } else { value & mask(32) });
        }
        (false, 0b0111) if (q && size == 3) || (!q && size < 3) => {
            write_x(ctx, d as u32, elem(a, esize, index));
        }
        (true, _) if q => {
            let source = imm4 >> size;
            ctx.v[d] = insert(ctx.v[d], esize, index, elem(a, esize, source));
        }
        _ => return Err(Undefined),
    }
    Ok(())
}

pub(super) fn permute(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let (d, n, m) = regs(inst);
    let q = bit(inst, 30);
    let size = bits(inst, 23, 22);
    let opcode = bits(inst, 14, 12);
    if (size == 3 && !q) || opcode & 0b011 == 0 {
        return Err(Undefined);
    }
    let esize = 8 << size;
    let elements = datasize(q) / esize;
    let half = elements / 2;
    let (a, b) = (ctx.v[n], ctx.v[m]);
    let upper = (opcode >> 2) as u32;
    let result = (0..elements).fold(0u128, |r, i| {
        let lane = match opcode & 0b011 {
            0b01 => {
                let j = 2 * i + upper;
                if j < elements {
                    elem(a, esize, j)
                } else {
                    elem(b, esize, j - elements)
                }
            }
            0b10 => {
                let source = if i % 2 == 0 { a } else { b };
                elem(source, esize, (i & !1) + upper)
            }
            _ => {
                let source = if i % 2 == 0 { a } else { b };
                elem(source, esize, upper * half + i / 2)
            }
        };
        insert(r, esize, i, lane)
    });
    write(ctx, d, result, datasize(q));
    Ok(())
}

pub(super) fn extract(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let (d, n, m) = regs(inst);
    let q = bit(inst, 30);
    let position = bits(inst, 14, 11);
    if bits(inst, 23, 22) != 0 || (!q && position >= 8) {
        return Err(Undefined);
    }
    let (a, b) = (ctx.v[n], ctx.v[m]);
    let result = if q {
        if position == 0 {
            a
        } else {
            (a >> (8 * position)) | (b << (128 - 8 * position))
        }
    } else {
        let concatenated = (a as u64 as u128) | ((b as u64 as u128) << 64);
        concatenated >> (8 * position)
    };
    write(ctx, d, result, datasize(q));
    Ok(())
}

pub(super) fn table_lookup(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let (d, n, m) = regs(inst);
    let q = bit(inst, 30);
    if bits(inst, 23, 22) != 0 {
        return Err(Undefined);
    }
    let registers = bits(inst, 14, 13) as usize + 1;
    let extension = bit(inst, 12);
    let indices = ctx.v[m];
    let original = ctx.v[d];
    let result = (0..datasize(q) / 8).fold(0u128, |r, i| {
        let index = elem(indices, 8, i) as usize;
        let byte = if index < 16 * registers {
            elem(ctx.v[(n + index / 16) % 32], 8, index as u32 % 16)
        } else if extension {
            elem(original, 8, i)
        } else {
            0
        };
        insert(r, 8, i, byte)
    });
    write(ctx, d, result, datasize(q));
    Ok(())
}

/// `AdvSIMDExpandImm` for the integer forms.
fn expand_integer(op: bool, cmode: u32, imm8: u64) -> u64 {
    let replicate64 = |x: u64, esize: u32| replicate(x, esize) as u64;
    match cmode >> 1 {
        0b000..=0b011 => replicate64(imm8 << (8 * (cmode >> 1)), 32),
        0b100 | 0b101 => replicate64(imm8 << (8 * ((cmode >> 1) & 1)), 16),
        0b110 if cmode & 1 == 0 => replicate64((imm8 << 8) | 0xff, 32),
        0b110 => replicate64((imm8 << 16) | 0xffff, 32),
        _ if op => (0..8).fold(0, |r, i| r | (((imm8 >> i) & 1) * 0xff) << (8 * i)),
        _ => replicate64(imm8, 8),
    }
}

pub(super) fn modified_immediate(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let d = bits(inst, 4, 0) as usize;
    let (q, op) = (bit(inst, 30), bit(inst, 29));
    let cmode = bits(inst, 15, 12);
    let imm8 = ((bits(inst, 18, 16) << 5) | bits(inst, 9, 5)) as u64;
    if bit(inst, 11) {
        if op || cmode != 0b1111 {
            return Err(Undefined);
        }
        write(
            ctx,
            d,
            replicate(expand_imm::<16>(imm8 as u32), 16),
            datasize(q),
        );
        return Ok(());
    }
    let imm64 = match (cmode, op) {
        (0b1111, false) => replicate(expand_imm::<32>(imm8 as u32), 32) as u64,
        (0b1111, true) if q => expand_imm::<64>(imm8 as u32),
        (0b1111, true) => return Err(Undefined),
        _ => expand_integer(op, cmode, imm8),
    };
    let imm = (imm64 as u128) | ((imm64 as u128) << 64);
    let old = ctx.v[d];
    let orr_or_bic = cmode < 0b1100 && cmode & 1 == 1;
    let result = match (orr_or_bic, op) {
        (true, false) => old | imm,
        (true, true) => old & !imm,
        (false, true) if cmode < 0b1110 => !imm,
        _ => imm,
    };
    write(ctx, d, result, datasize(q));
    Ok(())
}
