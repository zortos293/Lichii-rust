//! Recognises the SIMD/FP instructions the backend compiles inline.

use super::super::ir::{Loc, Op};
use super::super::translate::Translator;
use super::ir::{Fp, SimdOp};
use crate::decoder::Reg;

const fn field(word: u32, low: u32, bits: u32) -> u8 {
    ((word >> low) & ((1 << bits) - 1)) as u8
}

/// Emits `word` as native ops and returns true, or emits nothing and returns false.
pub(in super::super) fn lower(t: &mut Translator, word: u32) -> bool {
    let (rd, rn, rm) = (field(word, 0, 5), field(word, 5, 5), field(word, 16, 5));
    let fp = match field(word, 22, 2) {
        0 => Fp::S,
        1 => Fp::D,
        _ => return false,
    };
    let vec = |t: &mut Translator, r: u8| t.op(Op::Get(Loc::V(r)));
    let unary = |t: &mut Translator, op: fn(Fp, _) -> SimdOp| {
        let a = vec(t, rn);
        let r = t.op(Op::Simd(op(fp, a)));
        t.set(Loc::V(rd), r);
    };
    match word & 0xfffffc00 {
        0x1e26_0000 | 0x9e66_0000 => {
            let v = vec(t, rn);
            let r = t.op(Op::Simd(SimdOp::ToGpr(fp, v)));
            t.set_x(Reg(rd), r);
            return true;
        }
        0x1e27_0000 | 0x9e67_0000 => {
            let x = t.x(Reg(rn));
            let r = t.op(Op::Simd(SimdOp::FromGpr(fp, x)));
            t.set(Loc::V(rd), r);
            return true;
        }
        _ => {}
    }
    match word & 0xff3f_fc00 {
        0x1e20_4000 => unary(t, SimdOp::Move),
        0x1e20_c000 => unary(t, SimdOp::Abs),
        0x1e21_4000 => unary(t, SimdOp::Neg),
        _ if word & 0xff20_0c00 == 0x1e20_0c00 => {
            let c = t.cond(field(word, 12, 4));
            let (a, b) = (vec(t, rn), vec(t, rm));
            let r = t.op(Op::Simd(SimdOp::Select(fp, c, a, b)));
            t.set(Loc::V(rd), r);
        }
        _ => return false,
    }
    true
}
