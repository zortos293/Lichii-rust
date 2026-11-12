//! Stubs until ported from the C++ double_add.cpp.

use super::{TResult, Translator};

pub(super) fn dadd_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dadd_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dadd_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
