//! Fixed-point arithmetic matching `Common::FixedPoint<I, F>` in the C++ renderer.
//!
//! [`Fx`] is the 64-bit family (`FixedPoint<64 - F, F>`), [`Fx32`] the 32-bit family
//! (`FixedPoint<32 - F, F>`). Multiplication and division widen to the next integer size and
//! truncate exactly like the C++ template does on x86-64, conversion from floats truncates toward
//! zero after scaling in the float's own precision, and `to_int` rounds by adding half of the
//! fractional part first.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 64-bit fixed-point value with `F` fractional bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Fx<const F: u32>(pub i64);

/// A 32-bit fixed-point value with `F` fractional bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Fx32<const F: u32>(pub i32);

impl<const F: u32> Fx<F> {
    pub const ONE: i64 = 1 << F;
    const FRACTION_MASK: i64 = Self::ONE - 1;

    pub const fn from_raw(raw: i64) -> Self {
        Self(raw)
    }

    pub const fn from_int(value: i64) -> Self {
        Self(value.wrapping_mul(Self::ONE))
    }

    pub fn from_f32(value: f32) -> Self {
        Self((value * Self::ONE as f32) as i64)
    }

    pub fn from_f64(value: f64) -> Self {
        Self((value * Self::ONE as f64) as i64)
    }

    pub const fn raw(self) -> i64 {
        self.0
    }

    pub const fn frac(self) -> i64 {
        self.0 & Self::FRACTION_MASK
    }

    pub const fn clear_int(self) -> Self {
        Self(self.0 & Self::FRACTION_MASK)
    }

    pub const fn to_int_floor(self) -> i32 {
        (self.0 >> F) as i32
    }

    pub const fn to_uint_floor(self) -> u32 {
        (self.0 >> F) as u32
    }

    pub const fn to_long_floor(self) -> i64 {
        self.0 >> F
    }

    const fn rounded(self) -> i64 {
        self.0.wrapping_add((self.0 & Self::FRACTION_MASK) >> 1) >> F
    }

    pub const fn to_int(self) -> i32 {
        self.rounded() as i32
    }

    pub const fn to_long(self) -> i64 {
        self.rounded()
    }

    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Self::ONE as f32
    }

    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::ONE as f64
    }
}

impl<const F: u32> Add for Fx<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl<const F: u32> Sub for Fx<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl<const F: u32> Mul for Fx<F> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(((self.0 as i128 * rhs.0 as i128) >> F) as i64)
    }
}

impl<const F: u32> Div for Fx<F> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if rhs.0 == 0 {
            return Self(0);
        }
        Self((((self.0 as i128) << F) / rhs.0 as i128) as i64)
    }
}

impl<const F: u32> Neg for Fx<F> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0.wrapping_neg())
    }
}

impl<const F: u32> AddAssign for Fx<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const F: u32> SubAssign for Fx<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const F: u32> MulAssign for Fx<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const F: u32> Fx32<F> {
    pub const ONE: i32 = 1 << F;

    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    pub const fn from_int(value: i32) -> Self {
        Self(value.wrapping_mul(Self::ONE))
    }

    pub fn from_f32(value: f32) -> Self {
        Self((value * Self::ONE as f32) as i32)
    }

    pub const fn raw(self) -> i32 {
        self.0
    }

    pub const fn to_int_floor(self) -> i32 {
        self.0 >> F
    }

    /// Promotes to the 64-bit family by reinterpreting the raw value, as mixed-width C++
    /// arithmetic does.
    pub const fn widen(self) -> Fx<F> {
        Fx(self.0 as i64)
    }
}

impl<const F: u32> Add for Fx32<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl<const F: u32> Sub for Fx32<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl<const F: u32> Mul for Fx32<F> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(((self.0 as i64 * rhs.0 as i64) >> F) as i32)
    }
}

impl<const F: u32> Div for Fx32<F> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if rhs.0 == 0 {
            return Self(0);
        }
        Self((((self.0 as i64) << F) / rhs.0 as i64) as i32)
    }
}

/// `FixedPoint<49, 15>`: sample-rate ratios, resampler fractions, limiter state.
pub type Q15 = Fx<15>;
/// `FixedPoint<50, 14>`: delay and reverb samples.
pub type Q14 = Fx<14>;
/// `FixedPoint<18, 14>`: delay effect gains.
pub type Q14x32 = Fx32<14>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_conversion_truncates_toward_zero() {
        assert_eq!(Q15::from_f32(0.5).raw(), 16384);
        assert_eq!(Q15::from_f32(1.3).raw(), 42598);
        assert_eq!(Q15::from_f32(-1.00001).raw(), -32768);
    }

    #[test]
    fn to_int_adds_half_the_fraction() {
        assert_eq!(Q15::from_raw(32768 + 16384).to_int(), 1);
        assert_eq!(Q15::from_raw(32768 + 21846).to_int(), 2);
        assert_eq!(Q15::from_raw(-32768 - 16384).to_int(), -2);
        assert_eq!(Q15::from_raw(-1).to_int_floor(), -1);
    }

    #[test]
    fn multiply_and_divide_widen_and_truncate() {
        let a = Q14::from_f32(1.5);
        let b = Q14::from_f32(-0.25);
        assert_eq!((a * b).raw(), -6144);
        assert_eq!((Q14::from_int(1) / Q14::from_int(3)).raw(), 5461);
        assert_eq!((Q14::from_int(-1) / Q14::from_int(3)).raw(), -5461);
        assert_eq!(
            (Q14x32::from_int(48) * Q14x32::from_f32(0.5)).raw(),
            24 << 14
        );
    }
}
