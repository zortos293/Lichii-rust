//! Stubs until ported from the C++ floating_point_add.cpp.

use super::{TResult, Translator};

pub(super) fn fadd_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fadd_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fadd_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fadd32i(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
