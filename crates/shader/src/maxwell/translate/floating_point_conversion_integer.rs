//! Stubs until ported from the C++ floating_point_conversion_integer.cpp.

use super::{TResult, Translator};

pub(super) fn f2i_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn f2i_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn f2i_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
