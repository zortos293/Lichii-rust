//! Stubs until ported from the C++ output_geometry.cpp.

use super::{TResult, Translator};

pub(super) fn out_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn out_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn out_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
