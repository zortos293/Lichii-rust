//! Stubs until ported from the C++ bitfield_extract.cpp.

use super::{TResult, Translator};

pub(super) fn bfe_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn bfe_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn bfe_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
