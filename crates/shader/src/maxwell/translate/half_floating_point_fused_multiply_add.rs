//! Stubs until ported from the C++ half_floating_point_fused_multiply_add.cpp.

use super::{TResult, Translator};

pub(super) fn hfma2_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hfma2_32i(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hfma2_cr(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hfma2_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hfma2_rc(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
