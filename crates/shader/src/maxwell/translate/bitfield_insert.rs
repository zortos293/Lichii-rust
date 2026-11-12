//! Stubs until ported from the C++ bitfield_insert.cpp.

use super::{TResult, Translator};

pub(super) fn bfi_cr(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn bfi_rc(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn bfi_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn bfi_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
