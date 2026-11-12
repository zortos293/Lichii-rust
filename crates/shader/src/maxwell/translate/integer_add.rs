//! Stubs until ported from the C++ integer_add.cpp.

use super::{TResult, Translator};

pub(super) fn iadd_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn iadd_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn iadd_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn iadd32i(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
