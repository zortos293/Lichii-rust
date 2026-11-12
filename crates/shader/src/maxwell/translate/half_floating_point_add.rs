//! Stubs until ported from the C++ half_floating_point_add.cpp.

use super::{TResult, Translator};

pub(super) fn hadd2_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hadd2_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hadd2_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hadd2_32i(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
