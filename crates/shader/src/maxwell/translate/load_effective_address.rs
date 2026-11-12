//! Stubs until ported from the C++ load_effective_address.cpp.

use super::{TResult, Translator};

pub(super) fn lea_hi_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn lea_lo_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn lea_lo_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn lea_lo_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn lea_hi_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
