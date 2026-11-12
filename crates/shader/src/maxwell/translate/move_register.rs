//! Stubs until ported from the C++ move_register.cpp.

use super::{TResult, Translator};

pub(super) fn mov_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn mov_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn mov32i(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn mov_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
