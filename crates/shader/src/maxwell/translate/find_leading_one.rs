//! Stubs until ported from the C++ find_leading_one.cpp.

use super::{TResult, Translator};

pub(super) fn flo_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn flo_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn flo_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
