//! Stubs until ported from the C++ logic_operation.cpp.

use super::{TResult, Translator};

pub(super) fn lop_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn lop_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn lop_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn lop32i(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
