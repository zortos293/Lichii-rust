//! Stubs until ported from the C++ floating_point_min_max.cpp.

use super::{TResult, Translator};

pub(super) fn fmnmx_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fmnmx_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fmnmx_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
