//! Stubs until ported from the C++ half_floating_point_multiply.cpp.

use super::{TResult, Translator};

pub(super) fn hmul2_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hmul2_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hmul2_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hmul2_32i(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
