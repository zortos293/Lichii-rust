//! Stubs until ported from the C++ floating_point_multiply.cpp.

use super::{TResult, Translator};

pub(super) fn fmul_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fmul_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fmul_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fmul32i(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
