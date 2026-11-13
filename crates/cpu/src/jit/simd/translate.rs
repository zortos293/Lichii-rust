//! Recognises the SIMD/FP instructions the backend compiles inline.

use super::super::ir::{Loc, Op, Value};
use super::super::translate::Translator;
use super::ir::{Arith, Fp, Fused, SimdOp};
use crate::decoder::Reg;

const fn field(word: u32, low: u32, bits: u32) -> u8 {
    ((word >> low) & ((1 << bits) - 1)) as u8
}

/// `VFPExpandImm`: the S or D value of an 8-bit FP immediate.
fn expand_imm(fp: Fp, imm8: u8) -> u64 {
    let (sign, b, low) = (imm8 as u64 >> 7, imm8 as u64 >> 6 & 1, imm8 as u64 & 0x3f);
    match fp {
        Fp::S => sign << 31 | (b ^ 1) << 30 | (b * 0x1f) << 25 | low << 19,
        Fp::D => sign << 63 | (b ^ 1) << 62 | (b * 0xff) << 54 | low << 48,
    }
}

/// Emits `word` as native ops and returns true, or emits nothing and returns false.
pub(in super::super) fn lower(t: &mut Translator, word: u32) -> bool {
    if super::vector::lower(t, word) || super::fp_vector::lower(t, word) {
        return true;
    }
    let (rd, rn, rm) = (field(word, 0, 5), field(word, 5, 5), field(word, 16, 5));
    let fp = match field(word, 22, 2) {
        0 => Fp::S,
        1 => Fp::D,
        _ => return false,
    };
    let vec = |t: &mut Translator, r: u8| t.op(Op::Get(Loc::V(r)));
    let set_vec = |t: &mut Translator, op: SimdOp| {
        let r = t.op(Op::Simd(op));
        t.set(Loc::V(rd), r);
    };
    let unary = |t: &mut Translator, op: fn(Fp, Value) -> SimdOp| {
        let a = vec(t, rn);
        set_vec(t, op(fp, a));
    };
    match word & 0xffff_fc00 {
        0x1e26_0000 | 0x9e66_0000 => {
            let v = vec(t, rn);
            let r = t.op(Op::Simd(SimdOp::ToGpr(fp, v)));
            t.set_x(Reg(rd), r);
            return true;
        }
        0x1e27_0000 | 0x9e67_0000 => {
            let x = t.x(Reg(rn));
            set_vec(t, SimdOp::FromGpr(fp, x));
            return true;
        }
        0x1e22_c000 | 0x1e62_4000 => {
            let a = vec(t, rn);
            set_vec(t, SimdOp::Convert { fp, a, word });
            return true;
        }
        _ => {}
    }
    let wide = word >> 31 != 0;
    if word & 0xdfbf_fc00 == 0x5e21_d800 {
        let v = vec(t, rn);
        let x = t.op(Op::Simd(SimdOp::ToGpr(fp, v)));
        set_vec(
            t,
            SimdOp::FromInt {
                fp,
                signed: word & 1 << 29 == 0,
                wide: fp == Fp::D,
                x,
                word,
            },
        );
        return true;
    }
    match word & 0x7f3f_fc00 {
        0x1e22_0000 | 0x1e23_0000 => {
            let x = t.x(Reg(rn));
            let signed = word & 0x1_0000 == 0;
            set_vec(
                t,
                SimdOp::FromInt {
                    fp,
                    signed,
                    wide,
                    x,
                    word,
                },
            );
            return true;
        }
        0x1e38_0000 | 0x1e39_0000 => {
            let a = vec(t, rn);
            let signed = word & 0x1_0000 == 0;
            let r = t.op(Op::Simd(SimdOp::ToInt {
                fp,
                signed,
                wide,
                a,
                word,
            }));
            t.set_x(Reg(rd), r);
            return true;
        }
        _ => {}
    }
    if word & 0xff20_0c00 == 0x1e20_0800 {
        let op = match field(word, 12, 4) {
            0 => Arith::Mul,
            1 => Arith::Div,
            2 => Arith::Add,
            3 => Arith::Sub,
            _ => return false,
        };
        let (a, b) = (vec(t, rn), vec(t, rm));
        set_vec(t, SimdOp::Arith { op, fp, a, b, word });
        return true;
    }
    if word & 0xff00_0000 == 0x1f00_0000 {
        let op = match (word >> 21 & 1, word >> 15 & 1) {
            (0, 0) => Fused::Madd,
            (0, _) => Fused::Msub,
            (_, 0) => Fused::Nmadd,
            _ => Fused::Nmsub,
        };
        let (n, m, a) = (vec(t, rn), vec(t, rm), vec(t, field(word, 10, 5)));
        set_vec(
            t,
            SimdOp::Fused {
                op,
                fp,
                n,
                m,
                a,
                word,
            },
        );
        return true;
    }
    if word & 0xff20_fc07 == 0x1e20_2000 {
        let a = vec(t, rn);
        let b = (word & 0x8 == 0).then(|| vec(t, rm));
        let r = t.op(Op::Simd(SimdOp::Compare { fp, a, b, word }));
        t.set(Loc::Nzcv, r);
        return true;
    }
    if word & 0xff20_1fe0 == 0x1e20_1000 {
        let bits = t.op(Op::Const(expand_imm(fp, field(word, 13, 8))));
        set_vec(t, SimdOp::FromGpr(fp, bits));
        return true;
    }
    match word & 0xff3f_fc00 {
        0x1e20_4000 => unary(t, SimdOp::Move),
        0x1e20_c000 => unary(t, SimdOp::Abs),
        0x1e21_4000 => unary(t, SimdOp::Neg),
        0x1e21_c000 => {
            let a = vec(t, rn);
            set_vec(t, SimdOp::Sqrt { fp, a, word });
        }
        _ if word & 0xff20_0c00 == 0x1e20_0c00 => {
            let c = t.cond(field(word, 12, 4));
            let (a, b) = (vec(t, rn), vec(t, rm));
            set_vec(t, SimdOp::Select(fp, c, a, b));
        }
        _ => return false,
    }
    true
}
