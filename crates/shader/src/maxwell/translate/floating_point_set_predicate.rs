//! Stubs until ported from the C++ floating_point_set_predicate.cpp.

use super::{TResult, Translator};

pub(super) fn fsetp_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fsetp_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn fsetp_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
