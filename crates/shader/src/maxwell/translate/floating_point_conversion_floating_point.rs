//! Stubs until ported from the C++ floating_point_conversion_floating_point.cpp.

use super::{TResult, Translator};

pub(super) fn f2f_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn f2f_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn f2f_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
