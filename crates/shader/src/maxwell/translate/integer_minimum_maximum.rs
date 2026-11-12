//! Stubs until ported from the C++ integer_minimum_maximum.cpp.

use super::{TResult, Translator};

pub(super) fn imnmx_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn imnmx_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn imnmx_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
