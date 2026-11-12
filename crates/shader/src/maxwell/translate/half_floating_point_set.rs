//! Stubs until ported from the C++ half_floating_point_set.cpp.

use super::{TResult, Translator};

pub(super) fn hset2_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hset2_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn hset2_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
