//! Stubs until ported from the C++ integer_set_predicate.cpp.

use super::{TResult, Translator};

pub(super) fn isetp_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn isetp_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn isetp_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
