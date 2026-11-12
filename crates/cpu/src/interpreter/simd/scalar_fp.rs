//! Scalar floating point: data processing with one, two and three sources, compares,
//! conditional compare and select, immediates, and conversions to and from integers and fixed
//! point.

use super::softfloat::{self as sf, Fmt, FpEnv, Rounding};
use super::{
    Exec, Undefined, bit, bits, commit_flags, condition_holds, read_scalar, read_x, set_nzcv,
    with_fp_width, write_scalar, write_x,
};
use crate::ThreadContext;

pub(super) fn execute(ctx: &mut ThreadContext, inst: u32) -> Exec {
    if bit(inst, 29) {
        return Err(Undefined);
    }
    if bit(inst, 24) {
        return without_sf(inst)
            .and_then(|()| with_fp_width!(bits(inst, 23, 22), fused(ctx, inst)));
    }
    if !bit(inst, 21) {
        return with_fp_width!(bits(inst, 23, 22), fixed_conversion(ctx, inst));
    }
    let ftype = bits(inst, 23, 22);
    match bits(inst, 11, 10) {
        0b01 => {
            without_sf(inst).and_then(|()| with_fp_width!(ftype, conditional_compare(ctx, inst)))
        }
        0b10 => without_sf(inst).and_then(|()| with_fp_width!(ftype, two_source(ctx, inst))),
        0b11 => {
            without_sf(inst).and_then(|()| with_fp_width!(ftype, conditional_select(ctx, inst)))
        }
        _ if bits(inst, 12, 10) == 0b100 => {
            without_sf(inst).and_then(|()| with_fp_width!(ftype, immediate(ctx, inst)))
        }
        _ if bits(inst, 13, 10) == 0b1000 => {
            without_sf(inst).and_then(|()| with_fp_width!(ftype, compare(ctx, inst)))
        }
        _ if bits(inst, 14, 10) == 0b10000 => {
            without_sf(inst).and_then(|()| with_fp_width!(ftype, one_source(ctx, inst)))
        }
        _ if bits(inst, 15, 10) == 0 => integer_conversion(ctx, inst),
        _ => Err(Undefined),
    }
}

fn without_sf(inst: u32) -> Exec {
    if bit(inst, 31) {
        Err(Undefined)
    } else {
        Ok(())
    }
}

fn finish(ctx: &mut ThreadContext, d: u32, value: u64, env: FpEnv) -> Exec {
    write_scalar(ctx, d, value);
    commit_flags(ctx, &env);
    Ok(())
}

fn fused<const N: u32>(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let (o1, o0) = (bit(inst, 21), bit(inst, 15));
    let negate_product = if o0 != o1 { Fmt::<N>::SIGN } else { 0 };
    let negate_addend = if o1 { Fmt::<N>::SIGN } else { 0 };
    let op1 = read_scalar::<N>(ctx, bits(inst, 9, 5)) ^ negate_product;
    let op2 = read_scalar::<N>(ctx, bits(inst, 20, 16));
    let addend = read_scalar::<N>(ctx, bits(inst, 14, 10)) ^ negate_addend;
    let mut env = FpEnv::new(ctx.fpcr);
    let result = sf::mul_add::<N>(addend, op1, op2, &mut env);
    finish(ctx, bits(inst, 4, 0), result, env)
}

fn two_source<const N: u32>(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let op1 = read_scalar::<N>(ctx, bits(inst, 9, 5));
    let op2 = read_scalar::<N>(ctx, bits(inst, 20, 16));
    let mut env = FpEnv::new(ctx.fpcr);
    let result = match bits(inst, 15, 12) {
        0b0000 => sf::mul::<N>(op1, op2, &mut env),
        0b0001 => sf::div::<N>(op1, op2, &mut env),
        0b0010 => sf::add::<N>(op1, op2, &mut env),
        0b0011 => sf::sub::<N>(op1, op2, &mut env),
        0b0100 => sf::max::<N>(op1, op2, &mut env),
        0b0101 => sf::min::<N>(op1, op2, &mut env),
        0b0110 => sf::max_num::<N>(op1, op2, &mut env),
        0b0111 => sf::min_num::<N>(op1, op2, &mut env),
        0b1000 => sf::mul::<N>(op1, op2, &mut env) ^ Fmt::<N>::SIGN,
        _ => return Err(Undefined),
    };
    finish(ctx, bits(inst, 4, 0), result, env)
}

fn one_source<const N: u32>(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let op = read_scalar::<N>(ctx, bits(inst, 9, 5));
    let mut env = FpEnv::new(ctx.fpcr);
    let rounding = env.rounding();
    let result = match bits(inst, 20, 15) {
        0b000000 => op,
        0b000001 => op & !Fmt::<N>::SIGN,
        0b000010 => op ^ Fmt::<N>::SIGN,
        0b000011 => sf::sqrt::<N>(op, &mut env),
        opcode @ 0b000100..=0b000111 => {
            let target = opcode & 3;
            if target == bits(inst, 23, 22) {
                return Err(Undefined);
            }
            match target {
                0 => sf::convert::<N, 32>(op, rounding, &mut env),
                1 => sf::convert::<N, 64>(op, rounding, &mut env),
                3 => sf::convert::<N, 16>(op, rounding, &mut env),
                _ => return Err(Undefined),
            }
        }
        0b001000 => sf::round_int::<N>(op, Rounding::TieEven, false, &mut env),
        0b001001 => sf::round_int::<N>(op, Rounding::PlusInf, false, &mut env),
        0b001010 => sf::round_int::<N>(op, Rounding::MinusInf, false, &mut env),
        0b001011 => sf::round_int::<N>(op, Rounding::Zero, false, &mut env),
        0b001100 => sf::round_int::<N>(op, Rounding::TieAway, false, &mut env),
        0b001110 => sf::round_int::<N>(op, rounding, true, &mut env),
        0b001111 => sf::round_int::<N>(op, rounding, false, &mut env),
        opcode @ 0b010000..=0b010011 if N != 16 => {
            let int_bits = if opcode & 2 == 0 { 32 } else { 64 };
            let rounding = if opcode & 1 == 0 {
                Rounding::Zero
            } else {
                rounding
            };
            sf::round_int_n::<N>(op, rounding, int_bits, &mut env)
        }
        _ => return Err(Undefined),
    };
    finish(ctx, bits(inst, 4, 0), result, env)
}

fn compare<const N: u32>(ctx: &mut ThreadContext, inst: u32) -> Exec {
    if bits(inst, 15, 14) != 0 || bits(inst, 2, 0) != 0 {
        return Err(Undefined);
    }
    let op1 = read_scalar::<N>(ctx, bits(inst, 9, 5));
    let op2 = if bit(inst, 3) {
        0
    } else {
        read_scalar::<N>(ctx, bits(inst, 20, 16))
    };
    let mut env = FpEnv::new(ctx.fpcr);
    let nzcv = sf::compare::<N>(op1, op2, bit(inst, 4), &mut env);
    set_nzcv(ctx, nzcv);
    commit_flags(ctx, &env);
    Ok(())
}

fn conditional_compare<const N: u32>(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let nzcv = if condition_holds(ctx.pstate, bits(inst, 15, 12)) {
        let op1 = read_scalar::<N>(ctx, bits(inst, 9, 5));
        let op2 = read_scalar::<N>(ctx, bits(inst, 20, 16));
        let mut env = FpEnv::new(ctx.fpcr);
        let nzcv = sf::compare::<N>(op1, op2, bit(inst, 4), &mut env);
        commit_flags(ctx, &env);
        nzcv
    } else {
        bits(inst, 3, 0)
    };
    set_nzcv(ctx, nzcv);
    Ok(())
}

fn conditional_select<const N: u32>(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let source = if condition_holds(ctx.pstate, bits(inst, 15, 12)) {
        bits(inst, 9, 5)
    } else {
        bits(inst, 20, 16)
    };
    let value = read_scalar::<N>(ctx, source);
    write_scalar(ctx, bits(inst, 4, 0), value);
    Ok(())
}

/// `VFPExpandImm`.
pub(super) fn expand_imm<const N: u32>(imm8: u32) -> u64 {
    let imm8 = imm8 as u64;
    let b6 = (imm8 >> 6) & 1;
    let replicated = if b6 == 1 {
        (1 << (Fmt::<N>::E - 3)) - 1
    } else {
        0
    };
    let exp = ((b6 ^ 1) << (Fmt::<N>::E - 1)) | (replicated << 2) | ((imm8 >> 4) & 3);
    let frac = (imm8 & 0xf) << (Fmt::<N>::F - 4);
    ((imm8 >> 7) << (N - 1)) | (exp << Fmt::<N>::F) | frac
}

fn immediate<const N: u32>(ctx: &mut ThreadContext, inst: u32) -> Exec {
    if bits(inst, 9, 5) != 0 {
        return Err(Undefined);
    }
    write_scalar(ctx, bits(inst, 4, 0), expand_imm::<N>(bits(inst, 20, 13)));
    Ok(())
}

fn fixed_conversion<const N: u32>(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let sf = bit(inst, 31);
    let scale = bits(inst, 15, 10);
    if !sf && scale < 32 {
        return Err(Undefined);
    }
    let int_bits = if sf { 64 } else { 32 };
    let fbits = 64 - scale;
    let (d, n) = (bits(inst, 4, 0), bits(inst, 9, 5));
    let mut env = FpEnv::new(ctx.fpcr);
    match bits(inst, 20, 16) {
        rmode_opcode @ (0b11000 | 0b11001) => {
            let op = read_scalar::<N>(ctx, n);
            let result = sf::to_fixed::<N>(
                op,
                fbits,
                rmode_opcode & 1 != 0,
                Rounding::Zero,
                int_bits,
                &mut env,
            );
            write_x(ctx, d, result);
        }
        rmode_opcode @ (0b00010 | 0b00011) => {
            let op = read_x(ctx, n);
            let rounding = env.rounding();
            let result = sf::from_fixed::<N>(
                op,
                rmode_opcode & 1 == 0,
                int_bits,
                fbits,
                rounding,
                &mut env,
            );
            write_scalar(ctx, d, result);
        }
        _ => return Err(Undefined),
    }
    commit_flags(ctx, &env);
    Ok(())
}

fn integer_conversion(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let sf = bit(inst, 31);
    let ftype = bits(inst, 23, 22);
    let rmode = bits(inst, 20, 19);
    let opcode = bits(inst, 18, 16);
    let (d, n) = (bits(inst, 4, 0), bits(inst, 9, 5));
    match (opcode, rmode) {
        (0b110 | 0b111, _) => return fmov_general(ctx, sf, ftype, rmode, opcode & 1 != 0, d, n),
        (0b000 | 0b001, _) | (0b100 | 0b101, 0) | (0b010 | 0b011, 0) => {}
        _ => return Err(Undefined),
    }
    with_fp_width!(ftype, convert_integer(ctx, inst))
}

fn convert_integer<const N: u32>(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let int_bits = if bit(inst, 31) { 64 } else { 32 };
    let opcode = bits(inst, 18, 16);
    let (d, n) = (bits(inst, 4, 0), bits(inst, 9, 5));
    let mut env = FpEnv::new(ctx.fpcr);
    let unsigned = opcode & 1 != 0;
    if opcode & 0b110 == 0b010 {
        let rounding = env.rounding();
        let result =
            sf::from_fixed::<N>(read_x(ctx, n), !unsigned, int_bits, 0, rounding, &mut env);
        write_scalar(ctx, d, result);
    } else {
        let rounding = if opcode & 0b100 != 0 {
            Rounding::TieAway
        } else {
            Rounding::from_rmode(bits(inst, 20, 19))
        };
        let op = read_scalar::<N>(ctx, n);
        let result = sf::to_fixed::<N>(op, 0, unsigned, rounding, int_bits, &mut env);
        write_x(ctx, d, result);
    }
    commit_flags(ctx, &env);
    Ok(())
}

fn fmov_general(
    ctx: &mut ThreadContext,
    sf: bool,
    ftype: u32,
    rmode: u32,
    to_vector: bool,
    d: u32,
    n: u32,
) -> Exec {
    if (sf, ftype, rmode, to_vector) == (false, 0b01, 0b11, false) {
        let mut env = FpEnv::new(ctx.fpcr);
        let (result, z) = sf::to_int32_js(ctx.v[n as usize] as u64, &mut env);
        write_x(ctx, d, result as u64);
        super::set_nzcv(ctx, (z as u32) << 2);
        commit_flags(ctx, &env);
        return Ok(());
    }
    let width = match (sf, ftype, rmode) {
        (false, 0b00, 0) => 32,
        (_, 0b11, 0) => 16,
        (true, 0b01, 0) => 64,
        (true, 0b10, 1) => {
            if to_vector {
                let low = ctx.v[d as usize] as u64 as u128;
                ctx.v[d as usize] = low | ((read_x(ctx, n) as u128) << 64);
            } else {
                write_x(ctx, d, (ctx.v[n as usize] >> 64) as u64);
            }
            return Ok(());
        }
        _ => return Err(Undefined),
    };
    let mask = u64::MAX >> (64 - width);
    if to_vector {
        write_scalar(ctx, d, read_x(ctx, n) & mask);
    } else {
        write_x(ctx, d, ctx.v[n as usize] as u64 & mask);
    }
    Ok(())
}
