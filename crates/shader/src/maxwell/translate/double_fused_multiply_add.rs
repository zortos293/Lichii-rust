//! Stubs until ported from the C++ double_fused_multiply_add.cpp.

use super::{TResult, Translator};

pub(super) fn dfma_cr(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dfma_rc(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dfma_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn dfma_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
