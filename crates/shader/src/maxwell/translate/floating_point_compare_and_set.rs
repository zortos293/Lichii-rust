//! Stubs until ported from the C++ floating_point_compare_and_set.cpp.

use super::{TResult, Translator};

pub(super) fn fset_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fset_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fset_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
