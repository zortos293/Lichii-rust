//! Stubs until ported from the C++ half_floating_point_set_predicate.cpp.

use super::{TResult, Translator};

pub(super) fn hsetp2_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hsetp2_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hsetp2_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
