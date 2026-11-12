//! Arm floating point in software.
//!
//! Every public function follows the Arm ARM pseudocode function of the same name (`FPAdd`,
//! `FPMulAdd`, `FPRound`, `FPToFixed`, ...) and is bit-exact with hardware for every FPCR
//! setting: rounding mode, flush-to-zero (FZ and FZ16), default NaN, alternative half precision
//! and the cumulative FPSR flags. Operands and results are raw encodings of `N` bits (16, 32 or
//! 64) held in a `u64`. Arithmetic is exact on 128-bit integers before one final rounding, so no
//! host floating point is involved and results do not depend on the host's MXCSR or FPU.

use std::cmp::Ordering;

pub(crate) const IOC: u32 = 1;
pub(crate) const DZC: u32 = 1 << 1;
pub(crate) const OFC: u32 = 1 << 2;
pub(crate) const UFC: u32 = 1 << 3;
pub(crate) const IXC: u32 = 1 << 4;
pub(crate) const IDC: u32 = 1 << 7;

const FPCR_AHP: u32 = 1 << 26;
const FPCR_DN: u32 = 1 << 25;
const FPCR_FZ: u32 = 1 << 24;
const FPCR_FZ16: u32 = 1 << 19;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rounding {
    TieEven,
    PlusInf,
    MinusInf,
    Zero,
    TieAway,
    Odd,
}

impl Rounding {
    /// Decodes an FPCR.RMode style two-bit field.
    pub(crate) fn from_rmode(rmode: u32) -> Self {
        match rmode & 3 {
            0 => Self::TieEven,
            1 => Self::PlusInf,
            2 => Self::MinusInf,
            _ => Self::Zero,
        }
    }
}

/// The FPCR an instruction executes under and the FPSR cumulative flags it raises.
pub(crate) struct FpEnv {
    pub(crate) fpcr: u32,
    pub(crate) flags: u32,
}

impl FpEnv {
    pub(crate) fn new(fpcr: u32) -> Self {
        Self { fpcr, flags: 0 }
    }

    pub(crate) fn raise(&mut self, flags: u32) {
        self.flags |= flags;
    }

    pub(crate) fn rounding(&self) -> Rounding {
        Rounding::from_rmode(self.fpcr >> 22)
    }

    fn dn(&self) -> bool {
        self.fpcr & FPCR_DN != 0
    }

    fn ahp(&self) -> bool {
        self.fpcr & FPCR_AHP != 0
    }

    fn fz<const N: u32>(&self) -> bool {
        if N == 16 {
            self.fpcr & FPCR_FZ16 != 0
        } else {
            self.fpcr & FPCR_FZ != 0
        }
    }
}

/// Encoding parameters of the `N`-bit IEEE format.
pub(crate) struct Fmt<const N: u32>;

impl<const N: u32> Fmt<N> {
    pub(crate) const E: u32 = match N {
        16 => 5,
        32 => 8,
        64 => 11,
        _ => panic!("unsupported floating-point width"),
    };
    pub(crate) const F: u32 = N - 1 - Self::E;
    pub(crate) const BIAS: i32 = (1 << (Self::E - 1)) - 1;
    pub(crate) const MIN_EXP: i32 = 1 - Self::BIAS;
    pub(crate) const EXP_ONES: u64 = (1 << Self::E) - 1;
    pub(crate) const FRAC_MASK: u64 = (1 << Self::F) - 1;
    pub(crate) const QUIET: u64 = 1 << (Self::F - 1);
    pub(crate) const SIGN: u64 = 1 << (N - 1);
    pub(crate) const MASK: u64 = u64::MAX >> (64 - N);
    pub(crate) const DEFAULT_NAN: u64 = (Self::EXP_ONES << Self::F) | Self::QUIET;

    pub(crate) const fn zero(sign: bool) -> u64 {
        if sign { Self::SIGN } else { 0 }
    }

    pub(crate) const fn infinity(sign: bool) -> u64 {
        Self::zero(sign) | (Self::EXP_ONES << Self::F)
    }

    pub(crate) const fn max_normal(sign: bool) -> u64 {
        Self::zero(sign) | ((Self::EXP_ONES - 1) << Self::F) | Self::FRAC_MASK
    }

    /// `(-1)^sign * 2^exp * (1 + frac / 2^F)` for a normal `exp`.
    pub(crate) const fn normal(sign: bool, exp: i32, frac: u64) -> u64 {
        Self::zero(sign) | (((exp + Self::BIAS) as u64) << Self::F) | frac
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Zero,
    Finite,
    Infinity,
    QNaN,
    SNaN,
}

/// An unpacked operand. A `Finite` value is `mant * 2^(exp - 63)` with bit 63 of `mant` set.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Value {
    pub(crate) kind: Kind,
    pub(crate) sign: bool,
    pub(crate) exp: i32,
    pub(crate) mant: u64,
}

impl Value {
    fn is_nan(&self) -> bool {
        matches!(self.kind, Kind::QNaN | Kind::SNaN)
    }

    fn is(&self, kind: Kind) -> bool {
        self.kind == kind
    }

    fn exact(&self) -> Exact {
        Exact {
            sign: self.sign,
            x: (self.mant as u128) << 62,
            e: self.exp - 125,
        }
    }
}

/// An exact intermediate `x * 2^e`; bit 0 of `x` may be a sticky bit standing for discarded
/// nonzero bits below it.
#[derive(Clone, Copy, Debug)]
struct Exact {
    sign: bool,
    x: u128,
    e: i32,
}

fn shr_sticky(x: u128, shift: u32) -> u128 {
    match shift {
        0 => x,
        1..=127 => (x >> shift) | ((x << (128 - shift) != 0) as u128),
        _ => (x != 0) as u128,
    }
}

impl Exact {
    fn product(a: &Value, b: &Value) -> Self {
        Self {
            sign: a.sign ^ b.sign,
            x: a.mant as u128 * b.mant as u128,
            e: a.exp + b.exp - 126,
        }
    }

    fn top_at_125(self) -> Self {
        let lz = self.x.leading_zeros() as i32;
        if lz >= 2 {
            Self {
                x: self.x << (lz - 2),
                e: self.e - (lz - 2),
                ..self
            }
        } else {
            Self {
                x: shr_sticky(self.x, (2 - lz) as u32),
                e: self.e + (2 - lz),
                ..self
            }
        }
    }

    fn add(self, other: Self) -> Self {
        if other.x == 0 {
            return self;
        }
        if self.x == 0 {
            return other;
        }
        let (a, b) = (self.top_at_125(), other.top_at_125());
        let (hi, lo) = if a.e >= b.e { (a, b) } else { (b, a) };
        let lo_x = shr_sticky(lo.x, (hi.e - lo.e) as u32);
        if hi.sign == lo.sign {
            Self {
                x: hi.x + lo_x,
                ..hi
            }
        } else if hi.x >= lo_x {
            Self {
                x: hi.x - lo_x,
                ..hi
            }
        } else {
            Self {
                sign: lo.sign,
                x: lo_x - hi.x,
                e: hi.e,
            }
        }
    }

    /// Rounds a nonzero value to `N` bits with the arithmetic flush-to-zero rules; an exact zero
    /// becomes the zero whose sign the rounding mode selects.
    fn round<const N: u32>(self, rounding: Rounding, env: &mut FpEnv) -> u64 {
        if self.x == 0 {
            return Fmt::<N>::zero(rounding == Rounding::MinusInf);
        }
        let lz = self.x.leading_zeros();
        let y = self.x << lz;
        let mant = (y >> 64) as u64 | (y as u64 != 0) as u64;
        round::<N>(self.sign, self.e + 127 - lz as i32, mant, rounding, env)
    }
}

fn unpack_base<const N: u32>(op: u64, fz: bool, ahp: bool, env: &mut FpEnv) -> Value {
    let sign = op & Fmt::<N>::SIGN != 0;
    let exp = (op >> Fmt::<N>::F) & Fmt::<N>::EXP_ONES;
    let frac = op & Fmt::<N>::FRAC_MASK;
    let special = |kind| Value {
        kind,
        sign,
        exp: 0,
        mant: 0,
    };
    if exp == 0 {
        if frac == 0 || fz {
            if frac != 0 && N != 16 {
                env.raise(IDC);
            }
            return special(Kind::Zero);
        }
        let lz = frac.leading_zeros();
        return Value {
            kind: Kind::Finite,
            sign,
            exp: Fmt::<N>::MIN_EXP - Fmt::<N>::F as i32 + 63 - lz as i32,
            mant: frac << lz,
        };
    }
    if exp == Fmt::<N>::EXP_ONES && !(N == 16 && ahp) {
        return special(if frac == 0 {
            Kind::Infinity
        } else if frac & Fmt::<N>::QUIET != 0 {
            Kind::QNaN
        } else {
            Kind::SNaN
        });
    }
    Value {
        kind: Kind::Finite,
        sign,
        exp: exp as i32 - Fmt::<N>::BIAS,
        mant: (frac | 1 << Fmt::<N>::F) << (63 - Fmt::<N>::F),
    }
}

/// `FPUnpack`: arithmetic operands, flushed under FZ/FZ16, never alternative half precision.
pub(crate) fn unpack<const N: u32>(op: u64, env: &mut FpEnv) -> Value {
    unpack_base::<N>(op, env.fz::<N>(), false, env)
}

/// `FPUnpackCV`: conversion operands, where FZ16 does not apply and FPCR.AHP does.
fn unpack_cv<const N: u32>(op: u64, env: &mut FpEnv) -> Value {
    let fz = N != 16 && env.fz::<N>();
    let ahp = env.ahp();
    unpack_base::<N>(op, fz, ahp, env)
}

/// `FPRoundBase` of `(-1)^sign * mant * 2^(exp - 63)`, `mant` normalised with bit 63 set.
fn round_base<const N: u32>(
    sign: bool,
    exp: i32,
    mant: u64,
    rounding: Rounding,
    fz: bool,
    ahp: bool,
    env: &mut FpEnv,
) -> u64 {
    let f = Fmt::<N>::F;
    let min_exp = Fmt::<N>::MIN_EXP;
    if fz && exp < min_exp {
        env.raise(UFC);
        return Fmt::<N>::zero(sign);
    }
    let (mut biased, shift) = if exp >= min_exp {
        (exp - min_exp + 1, 63 - f)
    } else {
        (0, (63 - f).saturating_add((min_exp - exp) as u32))
    };
    let (mut int_mant, half, rest) = if shift < 64 {
        (
            mant >> shift,
            (mant >> (shift - 1)) & 1 != 0,
            mant & ((1 << (shift - 1)) - 1) != 0,
        )
    } else if shift == 64 {
        (0, mant >> 63 != 0, mant << 1 != 0)
    } else {
        (0, false, mant != 0)
    };
    let mut inexact = half || rest;
    if biased == 0 && inexact {
        env.raise(UFC);
    }
    let (round_up, overflow_to_inf) = match rounding {
        Rounding::TieEven => (half && (rest || int_mant & 1 != 0), true),
        Rounding::PlusInf => (inexact && !sign, !sign),
        Rounding::MinusInf => (inexact && sign, sign),
        Rounding::Zero | Rounding::Odd => (false, false),
        Rounding::TieAway => (half, true),
    };
    if round_up {
        int_mant += 1;
        if int_mant == 1 << f {
            biased = 1;
        }
        if int_mant == 1 << (f + 1) {
            biased += 1;
            int_mant >>= 1;
        }
    }
    if rounding == Rounding::Odd && inexact {
        int_mant |= 1;
    }
    let packed = |biased: i32| {
        Fmt::<N>::zero(sign) | ((biased as u64) << f) | (int_mant & Fmt::<N>::FRAC_MASK)
    };
    let result = if N != 16 || !ahp {
        if biased >= Fmt::<N>::EXP_ONES as i32 {
            env.raise(OFC);
            inexact = true;
            if overflow_to_inf {
                Fmt::<N>::infinity(sign)
            } else {
                Fmt::<N>::max_normal(sign)
            }
        } else {
            packed(biased)
        }
    } else if biased > Fmt::<N>::EXP_ONES as i32 {
        env.raise(IOC);
        inexact = false;
        Fmt::<N>::zero(sign) | (Fmt::<N>::MASK >> 1)
    } else {
        packed(biased)
    };
    if inexact {
        env.raise(IXC);
    }
    result
}

/// `FPRound`.
fn round<const N: u32>(
    sign: bool,
    exp: i32,
    mant: u64,
    rounding: Rounding,
    env: &mut FpEnv,
) -> u64 {
    let fz = env.fz::<N>();
    round_base::<N>(sign, exp, mant, rounding, fz, false, env)
}

/// `FPRoundCV`: FZ16 does not apply, FPCR.AHP does.
fn round_cv<const N: u32>(
    sign: bool,
    exp: i32,
    mant: u64,
    rounding: Rounding,
    env: &mut FpEnv,
) -> u64 {
    let fz = N != 16 && env.fz::<N>();
    let ahp = env.ahp();
    round_base::<N>(sign, exp, mant, rounding, fz, ahp, env)
}

fn process_nan<const N: u32>(v: &Value, op: u64, env: &mut FpEnv) -> u64 {
    let mut result = op;
    if v.is(Kind::SNaN) {
        env.raise(IOC);
        result |= Fmt::<N>::QUIET;
    }
    if env.dn() {
        Fmt::<N>::DEFAULT_NAN
    } else {
        result
    }
}

fn process_nans<const N: u32>(
    a: &Value,
    op1: u64,
    b: &Value,
    op2: u64,
    env: &mut FpEnv,
) -> Option<u64> {
    for kind in [Kind::SNaN, Kind::QNaN] {
        if a.is(kind) {
            return Some(process_nan::<N>(a, op1, env));
        }
        if b.is(kind) {
            return Some(process_nan::<N>(b, op2, env));
        }
    }
    None
}

/// Orders two non-NaN operands by value; zeros of either sign are equal.
fn compare_values(a: &Value, b: &Value) -> Ordering {
    let signum = |v: &Value| match (v.kind, v.sign) {
        (Kind::Zero, _) => 0,
        (_, true) => -1,
        (_, false) => 1,
    };
    let (sa, sb) = (signum(a), signum(b));
    if sa != sb || sa == 0 {
        return sa.cmp(&sb);
    }
    let magnitude = |v: &Value| match v.kind {
        Kind::Infinity => (1, 0, 0),
        _ => (0, v.exp, v.mant),
    };
    let order = magnitude(a).cmp(&magnitude(b));
    if sa < 0 { order.reverse() } else { order }
}

pub(crate) fn add<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    add_sub::<N>(op1, op2, false, env)
}

pub(crate) fn sub<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    add_sub::<N>(op1, op2, true, env)
}

fn add_sub<const N: u32>(op1: u64, op2: u64, negate: bool, env: &mut FpEnv) -> u64 {
    let a = unpack::<N>(op1, env);
    let mut b = unpack::<N>(op2, env);
    if let Some(nan) = process_nans::<N>(&a, op1, &b, op2, env) {
        return nan;
    }
    b.sign ^= negate;
    let (inf1, inf2) = (a.is(Kind::Infinity), b.is(Kind::Infinity));
    if inf1 && inf2 && a.sign != b.sign {
        env.raise(IOC);
        return Fmt::<N>::DEFAULT_NAN;
    }
    if inf1 || inf2 {
        return Fmt::<N>::infinity(if inf1 { a.sign } else { b.sign });
    }
    if a.is(Kind::Zero) && b.is(Kind::Zero) && a.sign == b.sign {
        return Fmt::<N>::zero(a.sign);
    }
    let rounding = env.rounding();
    a.exact().add(b.exact()).round::<N>(rounding, env)
}

pub(crate) fn mul<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    mul_base::<N>(op1, op2, false, env)
}

/// `FPMulX`: infinity times zero is two instead of the default NaN.
pub(crate) fn mulx<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    mul_base::<N>(op1, op2, true, env)
}

fn mul_base<const N: u32>(op1: u64, op2: u64, mulx: bool, env: &mut FpEnv) -> u64 {
    let a = unpack::<N>(op1, env);
    let b = unpack::<N>(op2, env);
    if let Some(nan) = process_nans::<N>(&a, op1, &b, op2, env) {
        return nan;
    }
    let sign = a.sign ^ b.sign;
    let (inf1, inf2) = (a.is(Kind::Infinity), b.is(Kind::Infinity));
    let (zero1, zero2) = (a.is(Kind::Zero), b.is(Kind::Zero));
    if (inf1 && zero2) || (zero1 && inf2) {
        if mulx {
            return Fmt::<N>::normal(sign, 1, 0);
        }
        env.raise(IOC);
        return Fmt::<N>::DEFAULT_NAN;
    }
    if inf1 || inf2 {
        return Fmt::<N>::infinity(sign);
    }
    if zero1 || zero2 {
        return Fmt::<N>::zero(sign);
    }
    let rounding = env.rounding();
    Exact::product(&a, &b).round::<N>(rounding, env)
}

pub(crate) fn div<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    let a = unpack::<N>(op1, env);
    let b = unpack::<N>(op2, env);
    if let Some(nan) = process_nans::<N>(&a, op1, &b, op2, env) {
        return nan;
    }
    let sign = a.sign ^ b.sign;
    let (inf1, inf2) = (a.is(Kind::Infinity), b.is(Kind::Infinity));
    let (zero1, zero2) = (a.is(Kind::Zero), b.is(Kind::Zero));
    if (inf1 && inf2) || (zero1 && zero2) {
        env.raise(IOC);
        return Fmt::<N>::DEFAULT_NAN;
    }
    if inf1 || zero2 {
        if !inf1 {
            env.raise(DZC);
        }
        return Fmt::<N>::infinity(sign);
    }
    if zero1 || inf2 {
        return Fmt::<N>::zero(sign);
    }
    let dividend = (a.mant as u128) << 64;
    let divisor = b.mant as u128;
    let quotient = dividend / divisor;
    let sticky = !dividend.is_multiple_of(divisor) as u128;
    let rounding = env.rounding();
    Exact {
        sign,
        x: (quotient << 1) | sticky,
        e: a.exp - b.exp - 65,
    }
    .round::<N>(rounding, env)
}

pub(crate) fn sqrt<const N: u32>(op: u64, env: &mut FpEnv) -> u64 {
    let a = unpack::<N>(op, env);
    match a.kind {
        Kind::QNaN | Kind::SNaN => return process_nan::<N>(&a, op, env),
        Kind::Zero => return Fmt::<N>::zero(a.sign),
        Kind::Infinity if !a.sign => return Fmt::<N>::infinity(false),
        _ if a.sign => {
            env.raise(IOC);
            return Fmt::<N>::DEFAULT_NAN;
        }
        _ => {}
    }
    let (x, e) = if a.exp % 2 == 0 {
        ((a.mant as u128) << 63, a.exp - 126)
    } else {
        ((a.mant as u128) << 64, a.exp - 127)
    };
    let root = x.isqrt();
    let sticky = (root * root != x) as u128;
    let rounding = env.rounding();
    Exact {
        sign: false,
        x: (root << 1) | sticky,
        e: e / 2 - 1,
    }
    .round::<N>(rounding, env)
}

/// `FPMulAdd`: `addend + op1 * op2` with a single rounding.
pub(crate) fn mul_add<const N: u32>(addend: u64, op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    fused::<N, N>(addend, op1, op2, env)
}

/// `FPMulAddH` (FMLAL, FMLSL): half-precision multiplicands, single-precision addend and result.
pub(crate) fn mul_add_h(addend: u64, op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    fused::<32, 16>(addend, op1, op2, env)
}

/// `FPConvertNaN`: keeps the sign and the most significant payload bits, and sets the quiet bit.
fn convert_nan<const I: u32, const O: u32>(op: u64) -> u64 {
    let payload = op & (Fmt::<I>::FRAC_MASK >> 1);
    let payload = if Fmt::<I>::F > Fmt::<O>::F {
        payload >> (Fmt::<I>::F - Fmt::<O>::F)
    } else {
        payload << (Fmt::<O>::F - Fmt::<I>::F)
    };
    Fmt::<O>::infinity(op & Fmt::<I>::SIGN != 0) | Fmt::<O>::QUIET | payload
}

/// `addend + op1 * op2` with an `N`-bit addend and result and `P`-bit multiplicands.
fn fused<const N: u32, const P: u32>(addend: u64, op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    let c = unpack::<N>(addend, env);
    let a = unpack::<P>(op1, env);
    let b = unpack::<P>(op2, env);
    let (inf1, inf2) = (a.is(Kind::Infinity), b.is(Kind::Infinity));
    let (zero1, zero2) = (a.is(Kind::Zero), b.is(Kind::Zero));
    let mut nan = None;
    for kind in [Kind::SNaN, Kind::QNaN] {
        if nan.is_some() {
            break;
        }
        if c.is(kind) {
            nan = Some(process_nan::<N>(&c, addend, env));
        } else if a.is(kind) {
            nan = Some(convert_nan::<P, N>(process_nan::<P>(&a, op1, env)));
        } else if b.is(kind) {
            nan = Some(convert_nan::<P, N>(process_nan::<P>(&b, op2, env)));
        }
    }
    let inf_times_zero = (inf1 && zero2) || (zero1 && inf2);
    if c.is(Kind::QNaN) && inf_times_zero {
        env.raise(IOC);
        return Fmt::<N>::DEFAULT_NAN;
    }
    if let Some(nan) = nan {
        return nan;
    }
    let sign_p = a.sign ^ b.sign;
    let inf_p = inf1 || inf2;
    let zero_p = zero1 || zero2;
    let inf_c = c.is(Kind::Infinity);
    if inf_times_zero || (inf_c && inf_p && c.sign != sign_p) {
        env.raise(IOC);
        return Fmt::<N>::DEFAULT_NAN;
    }
    if inf_c || inf_p {
        return Fmt::<N>::infinity(if inf_c { c.sign } else { sign_p });
    }
    if c.is(Kind::Zero) && zero_p && c.sign == sign_p {
        return Fmt::<N>::zero(c.sign);
    }
    let rounding = env.rounding();
    c.exact()
        .add(Exact::product(&a, &b))
        .round::<N>(rounding, env)
}

/// `FPRecipStepFused`: `2 - op1 * op2`.
pub(crate) fn recip_step<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    let two = Exact {
        sign: false,
        x: 1 << 125,
        e: -124,
    };
    step_fused::<N>(op1, op2, two, Fmt::<N>::normal(false, 1, 0), 0, env)
}

/// `FPRSqrtStepFused`: `(3 - op1 * op2) / 2`.
pub(crate) fn rsqrt_step<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    let three = Exact {
        sign: false,
        x: 3 << 124,
        e: -124,
    };
    step_fused::<N>(
        op1,
        op2,
        three,
        Fmt::<N>::normal(false, 0, Fmt::<N>::QUIET),
        1,
        env,
    )
}

fn step_fused<const N: u32>(
    op1: u64,
    op2: u64,
    constant: Exact,
    inf_times_zero: u64,
    halve: i32,
    env: &mut FpEnv,
) -> u64 {
    let op1 = op1 ^ Fmt::<N>::SIGN;
    let a = unpack::<N>(op1, env);
    let b = unpack::<N>(op2, env);
    if let Some(nan) = process_nans::<N>(&a, op1, &b, op2, env) {
        return nan;
    }
    let (inf1, inf2) = (a.is(Kind::Infinity), b.is(Kind::Infinity));
    let (zero1, zero2) = (a.is(Kind::Zero), b.is(Kind::Zero));
    if (inf1 && zero2) || (zero1 && inf2) {
        return inf_times_zero;
    }
    if inf1 || inf2 {
        return Fmt::<N>::infinity(a.sign ^ b.sign);
    }
    let mut sum = constant.add(Exact::product(&a, &b));
    sum.e -= halve;
    let rounding = env.rounding();
    sum.round::<N>(rounding, env)
}

fn min_max<const N: u32>(op1: u64, op2: u64, max: bool, env: &mut FpEnv) -> u64 {
    let a = unpack::<N>(op1, env);
    let b = unpack::<N>(op2, env);
    if let Some(nan) = process_nans::<N>(&a, op1, &b, op2, env) {
        return nan;
    }
    let wanted = if max {
        Ordering::Greater
    } else {
        Ordering::Less
    };
    let (v, op) = if compare_values(&a, &b) == wanted {
        (a, op1)
    } else {
        (b, op2)
    };
    if v.is(Kind::Zero) {
        let sign = if max {
            a.sign && b.sign
        } else {
            a.sign || b.sign
        };
        return Fmt::<N>::zero(sign);
    }
    op
}

pub(crate) fn max<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    min_max::<N>(op1, op2, true, env)
}

pub(crate) fn min<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    min_max::<N>(op1, op2, false, env)
}

fn min_max_num<const N: u32>(mut op1: u64, mut op2: u64, max: bool, env: &mut FpEnv) -> u64 {
    let a = unpack::<N>(op1, env);
    let b = unpack::<N>(op2, env);
    let replacement = Fmt::<N>::infinity(max);
    if a.is(Kind::QNaN) && !b.is(Kind::QNaN) {
        op1 = replacement;
    } else if !a.is(Kind::QNaN) && b.is(Kind::QNaN) {
        op2 = replacement;
    }
    min_max::<N>(op1, op2, max, env)
}

pub(crate) fn max_num<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    min_max_num::<N>(op1, op2, true, env)
}

pub(crate) fn min_num<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> u64 {
    min_max_num::<N>(op1, op2, false, env)
}

/// `FPCompare`: the NZCV flags of `op1 - op2`.
pub(crate) fn compare<const N: u32>(op1: u64, op2: u64, signal_nans: bool, env: &mut FpEnv) -> u32 {
    match compare_with::<N>(op1, op2, signal_nans, env) {
        None => 0b0011,
        Some(Ordering::Equal) => 0b0110,
        Some(Ordering::Less) => 0b1000,
        Some(Ordering::Greater) => 0b0010,
    }
}

fn compare_with<const N: u32>(
    op1: u64,
    op2: u64,
    signal_nans: bool,
    env: &mut FpEnv,
) -> Option<Ordering> {
    let a = unpack::<N>(op1, env);
    let b = unpack::<N>(op2, env);
    if a.is_nan() || b.is_nan() {
        if a.is(Kind::SNaN) || b.is(Kind::SNaN) || signal_nans {
            env.raise(IOC);
        }
        return None;
    }
    Some(compare_values(&a, &b))
}

/// `FPCompareEQ`.
pub(crate) fn compare_eq<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> bool {
    compare_with::<N>(op1, op2, false, env) == Some(Ordering::Equal)
}

/// `FPCompareGE`.
pub(crate) fn compare_ge<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> bool {
    matches!(
        compare_with::<N>(op1, op2, true, env),
        Some(Ordering::Greater | Ordering::Equal)
    )
}

/// `FPCompareGT`.
pub(crate) fn compare_gt<const N: u32>(op1: u64, op2: u64, env: &mut FpEnv) -> bool {
    compare_with::<N>(op1, op2, true, env) == Some(Ordering::Greater)
}

/// Integer part of `|v| * 2^scale` (`u128::MAX` when it does not fit), and whether the
/// discarded fraction is at least one half and whether anything below the half is set.
fn split_integer(v: &Value, scale: i32) -> (u128, bool, bool) {
    if v.is(Kind::Zero) {
        return (0, false, false);
    }
    let k = v.exp.saturating_add(scale);
    let m = v.mant as u128;
    match k {
        i32::MIN..=-2 => (0, false, true),
        -1 => (0, true, v.mant << 1 != 0),
        0..=62 => (
            m >> (63 - k),
            (v.mant >> (62 - k)) & 1 != 0,
            v.mant & ((1 << (62 - k)) - 1) != 0,
        ),
        63..=126 => (m << (k - 63), false, false),
        _ => (u128::MAX, false, false),
    }
}

fn round_magnitude(
    magnitude: u128,
    half: bool,
    rest: bool,
    sign: bool,
    rounding: Rounding,
) -> u128 {
    let inexact = half || rest;
    let round_up = match rounding {
        Rounding::TieEven => half && (rest || magnitude & 1 != 0),
        Rounding::PlusInf => inexact && !sign,
        Rounding::MinusInf => inexact && sign,
        Rounding::Zero | Rounding::Odd => false,
        Rounding::TieAway => half,
    };
    magnitude.saturating_add(round_up as u128)
}

fn from_magnitude<const N: u32>(sign: bool, magnitude: u64, env: &mut FpEnv) -> u64 {
    let lz = magnitude.leading_zeros();
    round::<N>(sign, 63 - lz as i32, magnitude << lz, Rounding::Zero, env)
}

/// `FPRoundInt`: FRINTN/P/M/Z/A/I, and FRINTX when `exact`.
pub(crate) fn round_int<const N: u32>(
    op: u64,
    rounding: Rounding,
    exact: bool,
    env: &mut FpEnv,
) -> u64 {
    let v = unpack::<N>(op, env);
    match v.kind {
        Kind::QNaN | Kind::SNaN => return process_nan::<N>(&v, op, env),
        Kind::Infinity => return Fmt::<N>::infinity(v.sign),
        Kind::Zero => return Fmt::<N>::zero(v.sign),
        Kind::Finite => {}
    }
    if v.exp >= Fmt::<N>::F as i32 {
        return op;
    }
    let (magnitude, half, rest) = split_integer(&v, 0);
    let magnitude = round_magnitude(magnitude, half, rest, v.sign, rounding) as u64;
    let result = if magnitude == 0 {
        Fmt::<N>::zero(v.sign)
    } else {
        from_magnitude::<N>(v.sign, magnitude, env)
    };
    if exact && (half || rest) {
        env.raise(IXC);
    }
    result
}

/// `FPRoundIntN`: FRINT32Z/X and FRINT64Z/X.
pub(crate) fn round_int_n<const N: u32>(
    op: u64,
    rounding: Rounding,
    int_bits: u32,
    env: &mut FpEnv,
) -> u64 {
    let v = unpack::<N>(op, env);
    let overflow = Fmt::<N>::normal(true, int_bits as i32 - 1, 0);
    match v.kind {
        Kind::QNaN | Kind::SNaN | Kind::Infinity => {
            env.raise(IOC);
            return overflow;
        }
        Kind::Zero => return Fmt::<N>::zero(v.sign),
        Kind::Finite => {}
    }
    let (magnitude, half, rest) = split_integer(&v, 0);
    let magnitude = round_magnitude(magnitude, half, rest, v.sign, rounding);
    let limit = 1u128 << (int_bits - 1);
    if magnitude > limit || (magnitude == limit && !v.sign) {
        env.raise(IOC);
        return overflow;
    }
    let result = if magnitude == 0 {
        Fmt::<N>::zero(v.sign)
    } else {
        from_magnitude::<N>(v.sign, magnitude as u64, env)
    };
    if half || rest {
        env.raise(IXC);
    }
    result
}

/// `FPToFixed`: `op * 2^fbits` rounded to an `int_bits`-bit integer, saturating.
pub(crate) fn to_fixed<const N: u32>(
    op: u64,
    fbits: u32,
    unsigned: bool,
    rounding: Rounding,
    int_bits: u32,
    env: &mut FpEnv,
) -> u64 {
    let v = unpack::<N>(op, env);
    if v.is_nan() {
        env.raise(IOC);
        return 0;
    }
    let (magnitude, half, rest) = if v.is(Kind::Infinity) {
        (u128::MAX, false, false)
    } else {
        split_integer(&v, fbits as i32)
    };
    let magnitude = round_magnitude(magnitude, half, rest, v.sign, rounding);
    let mask = u64::MAX >> (64 - int_bits);
    let (result, overflow) = if unsigned {
        if v.sign && magnitude != 0 {
            (0, true)
        } else if magnitude > mask as u128 {
            (mask, true)
        } else {
            (magnitude as u64, false)
        }
    } else {
        let limit = 1u128 << (int_bits - 1);
        if v.sign {
            if magnitude > limit {
                (limit as u64, true)
            } else {
                ((magnitude as u64).wrapping_neg(), false)
            }
        } else if magnitude >= limit {
            (limit as u64 - 1, true)
        } else {
            (magnitude as u64, false)
        }
    };
    if overflow {
        env.raise(IOC);
    } else if half || rest {
        env.raise(IXC);
    }
    result & mask
}

/// `FPToFixedJS` (FJCVTZS): the 32-bit result and the Z flag.
pub(crate) fn to_int32_js(op: u64, env: &mut FpEnv) -> (u32, bool) {
    let v = unpack::<64>(op, env);
    let mut z = true;
    if v.is_nan() {
        env.raise(IOC);
        z = false;
    }
    if v.kind != Kind::Finite {
        if v.is(Kind::Infinity) {
            env.raise(IOC);
            z = false;
        } else if v.is(Kind::Zero) && (v.sign || op & Fmt::<64>::FRAC_MASK != 0) {
            z = false;
        }
        return (0, z);
    }
    let (magnitude, half, rest) = split_integer(&v, 0);
    let low = if v.exp > 126 { 0 } else { magnitude as u32 };
    let result = if v.sign { low.wrapping_neg() } else { low };
    let limit = 1u128 << 31;
    if magnitude > limit || (magnitude == limit && !v.sign) {
        env.raise(IOC);
        z = false;
    } else if half || rest {
        env.raise(IXC);
        z = false;
    }
    (result, z)
}

/// `FixedToFP`: the `int_bits`-bit integer `op` divided by `2^fbits`.
pub(crate) fn from_fixed<const N: u32>(
    op: u64,
    signed: bool,
    int_bits: u32,
    fbits: u32,
    rounding: Rounding,
    env: &mut FpEnv,
) -> u64 {
    let (sign, magnitude) = if signed {
        let value = ((op << (64 - int_bits)) as i64) >> (64 - int_bits);
        (value < 0, value.unsigned_abs())
    } else {
        (false, op & (u64::MAX >> (64 - int_bits)))
    };
    if magnitude == 0 {
        return Fmt::<N>::zero(false);
    }
    let lz = magnitude.leading_zeros();
    round::<N>(
        sign,
        63 - lz as i32 - fbits as i32,
        magnitude << lz,
        rounding,
        env,
    )
}

/// `FPConvert` from `I` bits to `O` bits.
pub(crate) fn convert<const I: u32, const O: u32>(
    op: u64,
    rounding: Rounding,
    env: &mut FpEnv,
) -> u64 {
    let v = unpack_cv::<I>(op, env);
    let alt_hp = O == 16 && env.ahp();
    match v.kind {
        Kind::QNaN | Kind::SNaN => {
            let result = if alt_hp {
                Fmt::<O>::zero(v.sign)
            } else if env.dn() {
                Fmt::<O>::DEFAULT_NAN
            } else {
                convert_nan::<I, O>(op)
            };
            if v.is(Kind::SNaN) || alt_hp {
                env.raise(IOC);
            }
            result
        }
        Kind::Infinity if alt_hp => {
            env.raise(IOC);
            Fmt::<O>::zero(v.sign) | (Fmt::<O>::MASK >> 1)
        }
        Kind::Infinity => Fmt::<O>::infinity(v.sign),
        Kind::Zero => Fmt::<O>::zero(v.sign),
        Kind::Finite => round_cv::<O>(v.sign, v.exp, v.mant, rounding, env),
    }
}

/// `RecipEstimate`: `a` in 256..512 is a fixed-point value in [0.5, 1).
pub(crate) fn recip_estimate_u9(a: u64) -> u64 {
    let a = a * 2 + 1;
    let b = (1 << 19) / a;
    b.div_ceil(2)
}

const fn recip_sqrt_estimate_u9(a: u64) -> u64 {
    let a = if a < 256 {
        a * 2 + 1
    } else {
        ((a >> 1) << 1) * 2 + 2
    };
    let mut b = 512;
    while a * (b + 1) * (b + 1) < (1 << 28) {
        b += 1;
    }
    b.div_ceil(2)
}

const RSQRT_TABLE: [u16; 384] = {
    let mut table = [0; 384];
    let mut i = 0;
    while i < 384 {
        table[i] = recip_sqrt_estimate_u9(i as u64 + 128) as u16;
        i += 1;
    }
    table
};

/// `RecipSqrtEstimate`: `a` in 128..512 is a fixed-point value in [0.25, 1).
pub(crate) fn recip_sqrt_estimate(a: u64) -> u64 {
    RSQRT_TABLE[a as usize - 128] as u64
}

/// The operand's fraction left-aligned in 52 bits, and its biased exponent.
fn fraction52<const N: u32>(op: u64) -> (u64, i32) {
    let frac = (op & Fmt::<N>::FRAC_MASK) << (52 - Fmt::<N>::F);
    (frac, ((op >> Fmt::<N>::F) & Fmt::<N>::EXP_ONES) as i32)
}

/// `FPRecipEstimate` (FRECPE).
pub(crate) fn recip_estimate<const N: u32>(op: u64, env: &mut FpEnv) -> u64 {
    let v = unpack::<N>(op, env);
    match v.kind {
        Kind::QNaN | Kind::SNaN => return process_nan::<N>(&v, op, env),
        Kind::Infinity => return Fmt::<N>::zero(v.sign),
        Kind::Zero => {
            env.raise(DZC);
            return Fmt::<N>::infinity(v.sign);
        }
        Kind::Finite => {}
    }
    if v.exp < -Fmt::<N>::BIAS - 1 {
        let overflow_to_inf = match env.rounding() {
            Rounding::PlusInf => !v.sign,
            Rounding::MinusInf => v.sign,
            Rounding::Zero => false,
            _ => true,
        };
        env.raise(OFC | IXC);
        return if overflow_to_inf {
            Fmt::<N>::infinity(v.sign)
        } else {
            Fmt::<N>::max_normal(v.sign)
        };
    }
    if env.fz::<N>() && v.exp >= Fmt::<N>::BIAS - 1 {
        env.raise(UFC);
        return Fmt::<N>::zero(v.sign);
    }
    let (mut fraction, mut exp) = fraction52::<N>(op);
    if exp == 0 {
        if fraction >> 51 == 0 {
            exp = -1;
            fraction = (fraction << 2) & ((1 << 52) - 1);
        } else {
            fraction = (fraction << 1) & ((1 << 52) - 1);
        }
    }
    let scaled = (1 << 8) | (fraction >> 44);
    let mut result_exp = 2 * Fmt::<N>::BIAS - 1 - exp;
    let estimate = recip_estimate_u9(scaled);
    let mut fraction = (estimate & 0xff) << 44;
    if result_exp == 0 {
        fraction = (1 << 51) | (fraction >> 1);
    } else if result_exp == -1 {
        fraction = (1 << 50) | (fraction >> 2);
        result_exp = 0;
    }
    Fmt::<N>::zero(v.sign) | ((result_exp as u64) << Fmt::<N>::F) | (fraction >> (52 - Fmt::<N>::F))
}

/// `FPRSqrtEstimate` (FRSQRTE).
pub(crate) fn rsqrt_estimate<const N: u32>(op: u64, env: &mut FpEnv) -> u64 {
    let v = unpack::<N>(op, env);
    match v.kind {
        Kind::QNaN | Kind::SNaN => return process_nan::<N>(&v, op, env),
        Kind::Zero => {
            env.raise(DZC);
            return Fmt::<N>::infinity(v.sign);
        }
        _ if v.sign => {
            env.raise(IOC);
            return Fmt::<N>::DEFAULT_NAN;
        }
        Kind::Infinity => return Fmt::<N>::zero(false),
        Kind::Finite => {}
    }
    let (mut fraction, mut exp) = fraction52::<N>(op);
    if exp == 0 {
        while fraction >> 51 == 0 {
            fraction <<= 1;
            exp -= 1;
        }
        fraction = (fraction << 1) & ((1 << 52) - 1);
    }
    let scaled = if exp & 1 == 0 {
        (1 << 8) | (fraction >> 44)
    } else {
        (1 << 7) | (fraction >> 45)
    };
    let result_exp = (3 * Fmt::<N>::BIAS - 1 - exp).div_euclid(2);
    let estimate = recip_sqrt_estimate(scaled);
    ((result_exp as u64) << Fmt::<N>::F) | ((estimate & 0xff) << (Fmt::<N>::F - 8))
}

/// `FPRecpX` (FRECPX).
pub(crate) fn recip_exponent<const N: u32>(op: u64, env: &mut FpEnv) -> u64 {
    let v = unpack::<N>(op, env);
    if v.is_nan() {
        return process_nan::<N>(&v, op, env);
    }
    let exp = (op >> Fmt::<N>::F) & Fmt::<N>::EXP_ONES;
    let result_exp = if exp == 0 {
        Fmt::<N>::EXP_ONES - 1
    } else {
        !exp & Fmt::<N>::EXP_ONES
    };
    Fmt::<N>::zero(v.sign) | (result_exp << Fmt::<N>::F)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_op(f: fn(u64, u64, &mut FpEnv) -> u64, a: f32, b: f32) -> (f32, u32) {
        let mut env = FpEnv::new(0);
        let r = f(a.to_bits() as u64, b.to_bits() as u64, &mut env);
        (f32::from_bits(r as u32), env.flags)
    }

    #[test]
    fn basic_arithmetic_matches_ieee() {
        assert_eq!(f32_op(add::<32>, 1.5, 2.25), (3.75, 0));
        assert_eq!(f32_op(sub::<32>, 1.0, 3.0), (-2.0, 0));
        assert_eq!(f32_op(mul::<32>, 3.0, -0.5), (-1.5, 0));
        assert_eq!(f32_op(div::<32>, 1.0, 3.0), (1.0 / 3.0, IXC));
        assert_eq!(f32_op(div::<32>, 1.0, 0.0), (f32::INFINITY, DZC));
    }

    #[test]
    fn rsqrt_table_endpoints() {
        assert_eq!(recip_sqrt_estimate(128), 511);
        assert_eq!(recip_sqrt_estimate(511), 256);
        assert_eq!(recip_estimate_u9(256), 511);
        assert_eq!(recip_estimate_u9(511), 256);
    }

    #[test]
    fn exhaustive_f16_sqrt_against_f64() {
        for bits in 0..0x7c00u64 {
            let mut env = FpEnv::new(0);
            let r = sqrt::<16>(bits, &mut env);
            let mut back = FpEnv::new(0);
            let wide = convert::<16, 64>(bits, Rounding::TieEven, &mut back);
            let expected = convert::<64, 16>(
                f64::from_bits(wide).sqrt().to_bits(),
                Rounding::TieEven,
                &mut back,
            );
            assert_eq!(r, expected, "sqrt({bits:#x})");
        }
    }
}
