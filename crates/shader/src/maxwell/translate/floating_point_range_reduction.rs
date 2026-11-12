//! Stubs until ported from the C++ floating_point_range_reduction.cpp.

use super::{TResult, Translator};

pub(super) fn rro_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn rro_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn rro_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
