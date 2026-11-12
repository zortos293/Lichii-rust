//! Stubs until ported from the C++ double_set_predicate.cpp.

use super::{TResult, Translator};

pub(super) fn dsetp_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dsetp_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dsetp_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
