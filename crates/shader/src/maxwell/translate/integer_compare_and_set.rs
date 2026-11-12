//! Stubs until ported from the C++ integer_compare_and_set.cpp.

use super::{TResult, Translator};

pub(super) fn iset_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn iset_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn iset_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
