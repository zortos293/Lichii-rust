//! Stubs until ported from the C++ move_register_to_predicate.cpp.

use super::{TResult, Translator};

pub(super) fn r2p_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn r2p_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn r2p_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
