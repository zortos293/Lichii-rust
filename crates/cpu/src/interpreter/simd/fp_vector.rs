//! Advanced SIMD floating point: three same (including FP16 and the widening FMLAL/FMLSL), two-
//! register miscellaneous, across lanes, scalar pairwise, by element, and the complex FCMLA and
//! FCADD, in their vector and scalar forms.

use super::lanes::{elem, insert, write, write_half};
use super::softfloat::{self as sf, Fmt, FpEnv, Rounding};
use super::{Exec, Undefined, bit, bits, commit_flags};
use crate::ThreadContext;

type BinaryOp = fn(u64, u64, &mut FpEnv) -> u64;

fn regs(inst: u32) -> (usize, usize, usize) {
    (
        bits(inst, 4, 0) as usize,
        bits(inst, 9, 5) as usize,
        bits(inst, 20, 16) as usize,
    )
}

fn datasize(inst: u32, scalar: bool, esize: u32) -> u32 {
    if scalar {
        esize
    } else if bit(inst, 30) {
        128
    } else {
        64
    }
}

fn mask_if(condition: bool) -> u64 {
    if condition { u64::MAX } else { 0 }
}

macro_rules! with_width {
    ($esize:expr, $f:ident($($arg:expr),* $(,)?)) => {
        match $esize {
            16 => $f::<16>($($arg),*),
            32 => $f::<32>($($arg),*),
            _ => $f::<64>($($arg),*),
        }
    };
}

/// The element width of an FP encoding: FP16 encodings are always half precision, the others
/// select single or double with bit 22. Vector double precision needs Q = 1.
fn width(inst: u32, scalar: bool, half: bool) -> Result<u32, Undefined> {
    let esize = if half {
        16
    } else if bit(inst, 22) {
        64
    } else {
        32
    };
    if !scalar && esize == 64 && !bit(inst, 30) {
        return Err(Undefined);
    }
    Ok(esize)
}

/// Lane-wise map over `datasize / N` lanes, collecting the result and the FPSR flags.
fn lanes<const N: u32>(
    ctx: &mut ThreadContext,
    d: usize,
    datasize: u32,
    mut f: impl FnMut(u32, &mut FpEnv) -> u64,
) -> Exec {
    let mut env = FpEnv::new(ctx.fpcr);
    let result = (0..datasize / N).fold(0u128, |r, i| insert(r, N, i, f(i, &mut env)));
    write(ctx, d, result, datasize);
    commit_flags(ctx, &env);
    Ok(())
}

/// Three same with an FP opcode.
pub(super) fn three_same(ctx: &mut ThreadContext, inst: u32, scalar: bool, half: bool) -> Exec {
    let (u, a) = (bit(inst, 29), bit(inst, 23));
    let opcode = bits(inst, 13, 11);
    if !half && (opcode, u) == (0b101, false) || !half && (opcode, u) == (0b001, true) {
        if scalar || bit(inst, 22) {
            return Err(Undefined);
        }
        return widening_mla(ctx, inst, u, a, None);
    }
    let esize = width(inst, scalar, half)?;
    with_width!(esize, three_same_lanes(ctx, inst, scalar))
}

fn three_same_lanes<const N: u32>(ctx: &mut ThreadContext, inst: u32, scalar: bool) -> Exec {
    let (d, n, m) = regs(inst);
    let key = (bit(inst, 29), bit(inst, 23), bits(inst, 13, 11));
    let scalar_ok = matches!(
        key,
        (false, false, 0b011 | 0b100 | 0b111)
            | (false, true, 0b111)
            | (true, false, 0b100 | 0b101)
            | (true, true, 0b010 | 0b100 | 0b101)
    );
    let allocated = !matches!(
        key,
        (false, false, 0b101)
            | (false, true, 0b011..=0b101)
            | (true, _, 0b001)
            | (true, true, 0b011 | 0b111)
    );
    if !allocated || (scalar && !scalar_ok) {
        return Err(Undefined);
    }
    let datasize = datasize(inst, scalar, N);
    let elements = datasize / N;
    let (va, vb, vd) = (ctx.v[n], ctx.v[m], ctx.v[d]);
    let pairwise = matches!(key, (true, _, 0b000 | 0b110) | (true, false, 0b010));
    let abs = !Fmt::<N>::SIGN;
    lanes::<N>(ctx, d, datasize, |i, env| {
        let (x, y) = if pairwise {
            let pick = |j: u32| {
                if j < elements {
                    elem(va, N, j)
                } else {
                    elem(vb, N, j - elements)
                }
            };
            (pick(2 * i), pick(2 * i + 1))
        } else {
            (elem(va, N, i), elem(vb, N, i))
        };
        let z = elem(vd, N, i);
        match key {
            (_, false, 0b000) => sf::max_num::<N>(x, y, env),
            (_, true, 0b000) => sf::min_num::<N>(x, y, env),
            (false, false, 0b001) => sf::mul_add::<N>(z, x, y, env),
            (false, true, 0b001) => sf::mul_add::<N>(z, x ^ Fmt::<N>::SIGN, y, env),
            (_, false, 0b010) => sf::add::<N>(x, y, env),
            (false, true, 0b010) => sf::sub::<N>(x, y, env),
            (true, true, 0b010) => sf::sub::<N>(x, y, env) & abs,
            (false, false, 0b011) => sf::mulx::<N>(x, y, env),
            (true, false, 0b011) => sf::mul::<N>(x, y, env),
            (false, false, 0b100) => mask_if(sf::compare_eq::<N>(x, y, env)),
            (true, false, 0b100) => mask_if(sf::compare_ge::<N>(x, y, env)),
            (true, true, 0b100) => mask_if(sf::compare_gt::<N>(x, y, env)),
            (true, false, 0b101) => mask_if(sf::compare_ge::<N>(x & abs, y & abs, env)),
            (true, true, 0b101) => mask_if(sf::compare_gt::<N>(x & abs, y & abs, env)),
            (_, false, 0b110) => sf::max::<N>(x, y, env),
            (_, true, 0b110) => sf::min::<N>(x, y, env),
            (false, false, 0b111) => sf::recip_step::<N>(x, y, env),
            (false, true, 0b111) => sf::rsqrt_step::<N>(x, y, env),
            _ => sf::div::<N>(x, y, env),
        }
    })
}

/// FMLAL, FMLSL and their high-half forms; `index` selects the by-element form.
fn widening_mla(
    ctx: &mut ThreadContext,
    inst: u32,
    upper: bool,
    subtract: bool,
    index: Option<(usize, u32)>,
) -> Exec {
    let (d, n, m) = regs(inst);
    let datasize = datasize(inst, false, 32);
    let elements = datasize / 32;
    let part = if upper { elements } else { 0 };
    let (va, vd) = (ctx.v[n], ctx.v[d]);
    let negate = if subtract { 0x8000 } else { 0 };
    let by_element = index.map(|(m, i)| elem(ctx.v[m], 16, i));
    let vm = ctx.v[m];
    lanes::<32>(ctx, d, datasize, |e, env| {
        let x = elem(va, 16, e + part) ^ negate;
        let y = by_element.unwrap_or_else(|| elem(vm, 16, e + part));
        sf::mul_add_h(elem(vd, 32, e), x, y, env)
    })
}

/// Two-register miscellaneous with an FP opcode.
pub(super) fn two_misc(ctx: &mut ThreadContext, inst: u32, scalar: bool, half: bool) -> Exec {
    let (u, a) = (bit(inst, 29), bit(inst, 23));
    let opcode = bits(inst, 16, 12);
    if !half {
        match (opcode, u, a) {
            (0b10110, _, false) | (0b10111, false, false) => {
                return narrow_or_widen(ctx, inst, scalar);
            }
            (0b11100, _, true) => return unsigned_estimate(ctx, inst, scalar),
            _ => {}
        }
    }
    let esize = width(inst, scalar, half)?;
    with_width!(esize, two_misc_lanes(ctx, inst, scalar, half))
}

fn two_misc_lanes<const N: u32>(
    ctx: &mut ThreadContext,
    inst: u32,
    scalar: bool,
    half: bool,
) -> Exec {
    let (d, n, _) = regs(inst);
    let (u, a) = (bit(inst, 29), bit(inst, 23));
    let opcode = bits(inst, 16, 12);
    let fpcr_rounding = Rounding::from_rmode(ctx.fpcr >> 22);
    #[derive(Clone, Copy)]
    enum Op {
        Compare(u32),
        Abs,
        Neg,
        Sqrt,
        RoundInt(Rounding, bool),
        RoundIntN(Rounding, u32),
        ToInt(Rounding),
        FromInt,
        RecipEstimate,
        RsqrtEstimate,
        RecipExponent,
    }
    let op = match (opcode, u, a) {
        (0b01100 | 0b01101, _, true) | (0b01110, false, true) => Op::Compare(opcode),
        (0b01111, false, true) => Op::Abs,
        (0b01111, true, true) => Op::Neg,
        (0b11111, true, true) => Op::Sqrt,
        (0b11000, false, _) => Op::RoundInt(
            if a {
                Rounding::PlusInf
            } else {
                Rounding::TieEven
            },
            false,
        ),
        (0b11001, false, _) => Op::RoundInt(
            if a {
                Rounding::Zero
            } else {
                Rounding::MinusInf
            },
            false,
        ),
        (0b11000, true, false) => Op::RoundInt(Rounding::TieAway, false),
        (0b11001, true, _) => Op::RoundInt(fpcr_rounding, !a),
        (0b11110 | 0b11111, _, false) if !half => {
            let rounding = if u { fpcr_rounding } else { Rounding::Zero };
            Op::RoundIntN(rounding, if opcode == 0b11110 { 32 } else { 64 })
        }
        (0b11010, _, false) => Op::ToInt(Rounding::TieEven),
        (0b11010, _, true) => Op::ToInt(Rounding::PlusInf),
        (0b11011, _, false) => Op::ToInt(Rounding::MinusInf),
        (0b11011, _, true) => Op::ToInt(Rounding::Zero),
        (0b11100, _, false) => Op::ToInt(Rounding::TieAway),
        (0b11101, _, false) => Op::FromInt,
        (0b11101, false, true) => Op::RecipEstimate,
        (0b11101, true, true) => Op::RsqrtEstimate,
        (0b11111, false, true) => Op::RecipExponent,
        _ => return Err(Undefined),
    };
    let scalar_ok = matches!(
        op,
        Op::Compare(_)
            | Op::ToInt(_)
            | Op::FromInt
            | Op::RecipEstimate
            | Op::RsqrtEstimate
            | Op::RecipExponent
    );
    if scalar != scalar_ok && (scalar || matches!(op, Op::RecipExponent)) {
        return Err(Undefined);
    }
    let datasize = datasize(inst, scalar, N);
    let va = ctx.v[n];
    lanes::<N>(ctx, d, datasize, |i, env| {
        let x = elem(va, N, i);
        match op {
            Op::Compare(0b01100) if u => mask_if(sf::compare_ge::<N>(x, 0, env)),
            Op::Compare(0b01100) => mask_if(sf::compare_gt::<N>(x, 0, env)),
            Op::Compare(0b01101) if u => mask_if(sf::compare_ge::<N>(0, x, env)),
            Op::Compare(0b01101) => mask_if(sf::compare_eq::<N>(x, 0, env)),
            Op::Compare(_) => mask_if(sf::compare_gt::<N>(0, x, env)),
            Op::Abs => x & !Fmt::<N>::SIGN,
            Op::Neg => x ^ Fmt::<N>::SIGN,
            Op::Sqrt => sf::sqrt::<N>(x, env),
            Op::RoundInt(rounding, exact) => sf::round_int::<N>(x, rounding, exact, env),
            Op::RoundIntN(rounding, int_bits) => sf::round_int_n::<N>(x, rounding, int_bits, env),
            Op::ToInt(rounding) => sf::to_fixed::<N>(x, 0, u, rounding, N, env),
            Op::FromInt => {
                let rounding = env.rounding();
                sf::from_fixed::<N>(x, !u, N, 0, rounding, env)
            }
            Op::RecipEstimate => sf::recip_estimate::<N>(x, env),
            Op::RsqrtEstimate => sf::rsqrt_estimate::<N>(x, env),
            Op::RecipExponent => sf::recip_exponent::<N>(x, env),
        }
    })
}

/// FCVTN, FCVTXN and FCVTL.
fn narrow_or_widen(ctx: &mut ThreadContext, inst: u32, scalar: bool) -> Exec {
    let (d, n, _) = regs(inst);
    let (q, u, sz) = (bit(inst, 30), bit(inst, 29), bit(inst, 22));
    let widen = bits(inst, 16, 12) == 0b10111;
    let to_odd = u;
    if (to_odd && !sz) || (scalar && !to_odd) || (widen && scalar) {
        return Err(Undefined);
    }
    let va = ctx.v[n];
    let mut env = FpEnv::new(ctx.fpcr);
    let rounding = if to_odd {
        Rounding::Odd
    } else {
        env.rounding()
    };
    if widen {
        let (esize, part) = if sz {
            (32, if q { 2 } else { 0 })
        } else {
            (16, if q { 4 } else { 0 })
        };
        let result = (0..64 / esize).fold(0u128, |r, i| {
            let x = elem(va, esize, i + part);
            let wide = if sz {
                sf::convert::<32, 64>(x, rounding, &mut env)
            } else {
                sf::convert::<16, 32>(x, rounding, &mut env)
            };
            insert(r, 2 * esize, i, wide)
        });
        ctx.v[d] = result;
    } else {
        let wide = if sz { 64 } else { 32 };
        let count = if scalar { 1 } else { 128 / wide };
        let result = (0..count).fold(0u128, |r, i| {
            let x = elem(va, wide, i);
            let narrow = if sz {
                sf::convert::<64, 32>(x, rounding, &mut env)
            } else {
                sf::convert::<32, 16>(x, rounding, &mut env)
            };
            insert(r, wide / 2, i, narrow)
        });
        if scalar {
            write(ctx, d, result, 32);
        } else {
            write_half(ctx, d, result as u64, q);
        }
    }
    commit_flags(ctx, &env);
    Ok(())
}

/// URECPE and URSQRTE.
fn unsigned_estimate(ctx: &mut ThreadContext, inst: u32, scalar: bool) -> Exec {
    let (d, n, _) = regs(inst);
    if scalar || bit(inst, 22) {
        return Err(Undefined);
    }
    let sqrt = bit(inst, 29);
    let va = ctx.v[n];
    let datasize = datasize(inst, false, 32);
    let result = (0..datasize / 32).fold(0u128, |r, i| {
        let x = elem(va, 32, i);
        let lane = if sqrt && x >> 30 == 0 || !sqrt && x >> 31 == 0 {
            0xffff_ffff
        } else if sqrt {
            sf::recip_sqrt_estimate(x >> 23) << 23
        } else {
            sf::recip_estimate_u9(x >> 23) << 23
        };
        insert(r, 32, i, lane)
    });
    write(ctx, d, result, datasize);
    Ok(())
}

/// `Reduce`: halves are reduced separately and combined low op high.
fn reduce<const N: u32>(
    v: u128,
    first: u32,
    count: u32,
    op: fn(u64, u64, &mut FpEnv) -> u64,
    env: &mut FpEnv,
) -> u64 {
    if count == 1 {
        return elem(v, N, first);
    }
    let half = count / 2;
    let hi = reduce::<N>(v, first + half, half, op, env);
    let lo = reduce::<N>(v, first, half, op, env);
    op(lo, hi, env)
}

fn reduction<const N: u32>(inst: u32) -> Result<BinaryOp, Undefined> {
    Ok(match (bits(inst, 16, 12), bit(inst, 23)) {
        (0b01100, false) => sf::max_num::<N>,
        (0b01100, true) => sf::min_num::<N>,
        (0b01101, false) => sf::add::<N>,
        (0b01111, false) => sf::max::<N>,
        (0b01111, true) => sf::min::<N>,
        _ => return Err(Undefined),
    })
}

/// FMAXNMV, FMINNMV, FMAXV and FMINV.
pub(super) fn across_lanes(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let (d, n, _) = regs(inst);
    let (q, u) = (bit(inst, 30), bit(inst, 29));
    if bit(inst, 22) || bits(inst, 16, 12) == 0b01101 {
        return Err(Undefined);
    }
    let mut env = FpEnv::new(ctx.fpcr);
    let va = ctx.v[n];
    let result = if u {
        if !q {
            return Err(Undefined);
        }
        reduce::<32>(va, 0, 4, reduction::<32>(inst)?, &mut env)
    } else {
        reduce::<16>(
            va,
            0,
            if q { 8 } else { 4 },
            reduction::<16>(inst)?,
            &mut env,
        )
    };
    write(ctx, d, result as u128, if u { 32 } else { 16 });
    commit_flags(ctx, &env);
    Ok(())
}

/// Scalar FMAXNMP, FADDP, FMAXP, FMINNMP and FMINP.
pub(super) fn scalar_pairwise(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let (d, n, _) = regs(inst);
    let esize = if !bit(inst, 29) {
        if bit(inst, 22) {
            return Err(Undefined);
        }
        16
    } else if bit(inst, 22) {
        64
    } else {
        32
    };
    if bits(inst, 16, 12) == 0b01101 && bit(inst, 23) {
        return Err(Undefined);
    }
    let mut env = FpEnv::new(ctx.fpcr);
    let va = ctx.v[n];
    let result = with_width!(esize, pair_reduce(va, inst, &mut env))?;
    write(ctx, d, result as u128, esize);
    commit_flags(ctx, &env);
    Ok(())
}

fn pair_reduce<const N: u32>(v: u128, inst: u32, env: &mut FpEnv) -> Result<u64, Undefined> {
    let op = reduction::<N>(inst)?;
    Ok(op(elem(v, N, 0), elem(v, N, 1), env))
}

/// FP by element: FMLA, FMLS, FMUL, FMULX, FMLAL/FMLSL and FCMLA.
pub(super) fn indexed(ctx: &mut ThreadContext, inst: u32, scalar: bool) -> Exec {
    let u = bit(inst, 29);
    let size = bits(inst, 23, 22);
    let opcode = bits(inst, 15, 12);
    let (h, l, mbit) = (
        bit(inst, 11) as u32,
        bit(inst, 21) as u32,
        bit(inst, 20) as u32,
    );
    let m_low = bits(inst, 19, 16) as usize;
    let m_full = bits(inst, 20, 16) as usize;
    if u && opcode & 0b1001 == 0b0001 {
        if scalar {
            return Err(Undefined);
        }
        return match size {
            1 if h == 0 || bit(inst, 30) => {
                complex_mla::<16>(ctx, inst, Some((m_full, (h << 1) | l)))
            }
            2 if l == 0 && bit(inst, 30) => complex_mla::<32>(ctx, inst, Some((m_full, h))),
            _ => Err(Undefined),
        };
    }
    if matches!(
        (opcode, u),
        (0b0000 | 0b0100, false) | (0b1000 | 0b1100, true)
    ) {
        if scalar || size != 2 {
            return Err(Undefined);
        }
        return widening_mla(
            ctx,
            inst,
            u,
            opcode & 0b0100 != 0,
            Some((m_low, (h << 2) | (l << 1) | mbit)),
        );
    }
    if !matches!(
        (opcode, u),
        (0b0001 | 0b0101 | 0b1001, false) | (0b1001, true)
    ) {
        return Err(Undefined);
    }
    match size {
        0 => indexed_lanes::<16>(ctx, inst, scalar, m_low, (h << 2) | (l << 1) | mbit),
        2 => indexed_lanes::<32>(ctx, inst, scalar, m_full, (h << 1) | l),
        3 if l == 0 && (scalar || bit(inst, 30)) => {
            indexed_lanes::<64>(ctx, inst, scalar, m_full, h)
        }
        _ => Err(Undefined),
    }
}

fn indexed_lanes<const N: u32>(
    ctx: &mut ThreadContext,
    inst: u32,
    scalar: bool,
    m: usize,
    index: u32,
) -> Exec {
    let (d, n, _) = regs(inst);
    let datasize = datasize(inst, scalar, N);
    let y = elem(ctx.v[m], N, index);
    let (va, vd) = (ctx.v[n], ctx.v[d]);
    let (opcode, u) = (bits(inst, 15, 12), bit(inst, 29));
    lanes::<N>(ctx, d, datasize, |i, env| {
        let (x, z) = (elem(va, N, i), elem(vd, N, i));
        match (opcode, u) {
            (0b0001, _) => sf::mul_add::<N>(z, x, y, env),
            (0b0101, _) => sf::mul_add::<N>(z, x ^ Fmt::<N>::SIGN, y, env),
            (_, false) => sf::mul::<N>(x, y, env),
            (_, true) => sf::mulx::<N>(x, y, env),
        }
    })
}

/// Three same extra FCMLA and FCADD.
pub(super) fn complex(ctx: &mut ThreadContext, inst: u32, scalar: bool) -> Exec {
    let size = bits(inst, 23, 22);
    if scalar || !bit(inst, 29) || size == 0 || (size == 3 && !bit(inst, 30)) {
        return Err(Undefined);
    }
    let opcode = bits(inst, 14, 11);
    match opcode {
        0b1000..=0b1011 => with_width!(8 << size, complex_mla(ctx, inst, None)),
        0b1100 | 0b1110 => with_width!(8 << size, complex_add(ctx, inst)),
        _ => Err(Undefined),
    }
}

/// FCMLA; `index` gives `Vm` and the element pair of the by-element form.
fn complex_mla<const N: u32>(
    ctx: &mut ThreadContext,
    inst: u32,
    index: Option<(usize, u32)>,
) -> Exec {
    let (d, n, m) = regs(inst);
    let rotation = if index.is_some() {
        bits(inst, 14, 13)
    } else {
        bits(inst, 12, 11)
    };
    let datasize = datasize(inst, false, N);
    let (va, vd) = (ctx.v[n], ctx.v[d]);
    let vm = index.map_or(ctx.v[m], |(m, _)| ctx.v[m]);
    let sign = Fmt::<N>::SIGN;
    lanes::<N>(ctx, d, datasize, |i, env| {
        let pair = index.map_or(i & !1, |(_, index)| 2 * index);
        let (re2, im2) = (elem(vm, N, pair), elem(vm, N, pair + 1));
        let (re1, im1) = (elem(va, N, i & !1), elem(va, N, (i & !1) + 1));
        let (multiplier, multiplicand) = match (rotation, i & 1) {
            (0b00, 0) => (re2, re1),
            (0b00, _) => (im2, re1),
            (0b01, 0) => (im2 ^ sign, im1),
            (0b01, _) => (re2, im1),
            (0b10, 0) => (re2 ^ sign, re1),
            (0b10, _) => (im2 ^ sign, re1),
            (_, 0) => (im2, im1),
            (_, _) => (re2 ^ sign, im1),
        };
        sf::mul_add::<N>(elem(vd, N, i), multiplicand, multiplier, env)
    })
}

fn complex_add<const N: u32>(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let (d, n, m) = regs(inst);
    let rotate_270 = bit(inst, 12);
    let datasize = datasize(inst, false, N);
    let (va, vm) = (ctx.v[n], ctx.v[m]);
    let sign = Fmt::<N>::SIGN;
    lanes::<N>(ctx, d, datasize, |i, env| {
        let (re2, im2) = (elem(vm, N, i & !1), elem(vm, N, (i & !1) + 1));
        let addend = match (rotate_270, i & 1) {
            (false, 0) => im2 ^ sign,
            (false, _) => re2,
            (true, 0) => im2,
            (true, _) => re2 ^ sign,
        };
        sf::add::<N>(elem(va, N, i), addend, env)
    })
}
