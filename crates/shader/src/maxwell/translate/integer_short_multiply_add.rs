//! Stubs until ported from the C++ integer_short_multiply_add.cpp.

use super::{TResult, Translator};

pub(super) fn xmad_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn xmad_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn xmad_rc(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn xmad_cr(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
