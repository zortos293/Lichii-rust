//! Stubs until ported from the C++ double_min_max.cpp.

use super::{TResult, Translator};

pub(super) fn dmnmx_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dmnmx_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dmnmx_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
