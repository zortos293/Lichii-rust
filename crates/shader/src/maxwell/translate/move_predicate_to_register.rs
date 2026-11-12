//! Stubs until ported from the C++ move_predicate_to_register.cpp.

use super::{TResult, Translator};

pub(super) fn p2r_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn p2r_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn p2r_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
