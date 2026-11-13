//! The SIMD/FP operations carried by `Op::Simd`.

use super::super::ir::{Ty, Value};

/// Scalar floating-point element size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp {
    S,
    D,
}

/// Each op that produces a vector leaves its result in the low element and zeroes the rest, as
/// A64 scalar writes do.
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
}

impl SimdOp {
    pub fn ty(&self) -> Ty {
        match self {
            SimdOp::ToGpr(..) => Ty::U64,
            _ => Ty::U128,
        }
    }

    pub fn for_each_arg(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            SimdOp::ToGpr(_, a)
            | SimdOp::FromGpr(_, a)
            | SimdOp::Move(_, a)
            | SimdOp::Abs(_, a)
            | SimdOp::Neg(_, a) => f(a),
            SimdOp::Select(_, c, a, b) => {
                f(c);
                f(a);
                f(b);
            }
        }
    }
}
