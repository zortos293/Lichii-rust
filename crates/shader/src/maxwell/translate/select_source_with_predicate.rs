//! Stubs until ported from the C++ select_source_with_predicate.cpp.

use super::{TResult, Translator};

pub(super) fn sel_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn sel_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn sel_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
