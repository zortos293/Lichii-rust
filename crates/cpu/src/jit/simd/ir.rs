//! The SIMD/FP operations carried by `Op::Simd`.

use super::super::ir::{Ty, Value};
use super::fp_vector::Lanes;
use super::vector::VecOp;

/// Scalar floating-point element size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp {
    S,
    D,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arith {
    Add,
    Sub,
    Mul,
    Div,
    /// `FNMUL`: the rounded product, negated.
    Nmul,
    /// `FMAX` or `FMAXNM`, which differ only on NaNs.
    Max,
    /// `FMIN` or `FMINNM`, which differ only on NaNs.
    Min,
}

/// The fused multiply-adds, as `(n, m, a)` maps to the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fused {
    /// `a + n * m`.
    Madd,
    /// `a - n * m`.
    Msub,
    /// `-a - n * m`.
    Nmadd,
    /// `n * m - a`.
    Nmsub,
}

/// Each op that produces a vector leaves its result in the low element and zeroes the rest, as
/// A64 scalar writes do.
///
/// The ops with a `word` compute on the host and check the host's result: whenever it could
/// differ from the Arm ARM in value or FPSR flags (a NaN, a denormal, an underflow, any rounding
/// mode but nearest, and so on), they run `word` through `CallSimd` instead and take its result
/// from the context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdOp {
    /// The low element of a vector, zero-extended to 64 bits.
    ToGpr(Fp, Value),
    /// The low bits of a general-purpose value as a vector.
    FromGpr(Fp, Value),
    /// `FMOV` between vector registers.
    Move(Fp, Value),
    /// `FABS`: clears the sign bit, NaNs included.
    Abs(Fp, Value),
    /// `FNEG`: flips the sign bit, NaNs included.
    Neg(Fp, Value),
    /// `FCSEL`: `if c != 0 { a } else { b }`.
    Select(Fp, Value, Value, Value),
    Arith {
        op: Arith,
        fp: Fp,
        a: Value,
        b: Value,
        word: u32,
    },
    Fused {
        op: Fused,
        fp: Fp,
        n: Value,
        m: Value,
        a: Value,
        word: u32,
    },
    Sqrt {
        fp: Fp,
        a: Value,
        word: u32,
    },
    /// `FCVT` from H to S or D; `fp` is the destination size.
    FromHalf {
        fp: Fp,
        a: Value,
        word: u32,
    },
    /// `FCVT` between S and D; `fp` is the source size.
    Convert {
        fp: Fp,
        a: Value,
        word: u32,
    },
    /// `SCVTF`/`UCVTF` from a W (`wide` false) or X register.
    FromInt {
        fp: Fp,
        signed: bool,
        wide: bool,
        x: Value,
        word: u32,
    },
    /// `FCVTZS`/`FCVTZU` to a W (`wide` false) or X register, zero-extended; also `FCVTNS`,
    /// `FCVTPS`, `FCVTMS` and their unsigned forms, which round as `word`'s RMode field says first.
    ToInt {
        fp: Fp,
        signed: bool,
        wide: bool,
        a: Value,
        word: u32,
    },
    /// `FCCMP` or `FCCMPE`: the compare's NZCV when `c` is nonzero, else the immediate `nzcv`.
    CondCompare {
        fp: Fp,
        c: Value,
        a: Value,
        b: Value,
        nzcv: u8,
        word: u32,
    },
    /// `FCMP` or `FCMPE`: the NZCV result. `b` is `None` for the forms that compare with zero.
    /// They differ only on NaNs, which always fall back.
    Compare {
        fp: Fp,
        a: Value,
        b: Option<Value>,
        word: u32,
    },
    /// `SQXTN` and `SQXTN2` to B (`size` 0) or H (`size` 1) lanes: the lanes of `n` narrowed with
    /// signed saturation, in the low half over zeros, or for `SQXTN2` in the high half over the
    /// low half of `low`. It sets FPSR.QC itself when a lane saturates, so like a checked op it
    /// carries `word` and passes treat it as writing FPSR.
    SatNarrow {
        size: u8,
        n: Value,
        low: Option<Value>,
        word: u32,
    },
    /// Advanced SIMD integer and element-move ops.
    Vec(VecOp),
    /// Advanced SIMD FP arithmetic on S or D lanes, checked like the scalar ops.
    Lanes(Lanes),
}

impl SimdOp {
    pub fn ty(&self) -> Ty {
        match self {
            SimdOp::ToGpr(..) | SimdOp::ToInt { .. } => Ty::U64,
            SimdOp::Compare { .. } | SimdOp::CondCompare { .. } => Ty::Nzcv,
            SimdOp::Vec(op) => op.ty(),
            _ => Ty::U128,
        }
    }

    /// The instruction the op falls back to, which also updates FPSR.
    pub fn falls_back(&self) -> bool {
        self.word().is_some()
    }

    pub fn word(&self) -> Option<u32> {
        match *self {
            SimdOp::Arith { word, .. }
            | SimdOp::Fused { word, .. }
            | SimdOp::Sqrt { word, .. }
            | SimdOp::Convert { word, .. }
            | SimdOp::FromHalf { word, .. }
            | SimdOp::FromInt { word, .. }
            | SimdOp::ToInt { word, .. }
            | SimdOp::Compare { word, .. }
            | SimdOp::CondCompare { word, .. }
            | SimdOp::SatNarrow { word, .. } => Some(word),
            SimdOp::Lanes(lanes) => Some(lanes.word),
            _ => None,
        }
    }

    pub fn is_pure(&self) -> bool {
        !self.falls_back()
    }

    pub fn for_each_arg(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            SimdOp::ToGpr(_, a)
            | SimdOp::FromGpr(_, a)
            | SimdOp::Move(_, a)
            | SimdOp::Abs(_, a)
            | SimdOp::Neg(_, a)
            | SimdOp::Sqrt { a, .. }
            | SimdOp::Convert { a, .. }
            | SimdOp::FromHalf { a, .. }
            | SimdOp::FromInt { x: a, .. }
            | SimdOp::ToInt { a, .. } => f(a),
            SimdOp::Arith { a, b, .. } => {
                f(a);
                f(b);
            }
            SimdOp::Compare { a, b, .. } => {
                f(a);
                if let Some(b) = b {
                    f(b);
                }
            }
            SimdOp::Select(_, c, a, b) | SimdOp::CondCompare { c, a, b, .. } => {
                f(c);
                f(a);
                f(b);
            }
            SimdOp::Fused { n, m, a, .. } => {
                f(n);
                f(m);
                f(a);
            }
            SimdOp::SatNarrow { n, low, .. } => {
                f(n);
                if let Some(low) = low {
                    f(low);
                }
            }
            SimdOp::Vec(op) => op.for_each_arg(f),
            SimdOp::Lanes(lanes) => lanes.for_each_arg(f),
        }
    }
}
