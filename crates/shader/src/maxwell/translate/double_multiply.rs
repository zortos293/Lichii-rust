//! Stubs until ported from the C++ double_multiply.cpp.

use super::{TResult, Translator};

pub(super) fn dmul_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dmul_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dmul_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
