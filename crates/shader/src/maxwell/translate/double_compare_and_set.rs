//! Stubs until ported from the C++ double_compare_and_set.cpp.

use super::{TResult, Translator};

pub(super) fn dset_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dset_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dset_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
