//! Stubs until ported from the C++ floating_point_compare.cpp.

use super::{TResult, Translator};

pub(super) fn fcmp_cr(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fcmp_rc(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fcmp_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fcmp_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
