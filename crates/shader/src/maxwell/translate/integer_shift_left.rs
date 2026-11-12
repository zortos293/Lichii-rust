//! Stubs until ported from the C++ integer_shift_left.cpp.

use super::{TResult, Translator};

pub(super) fn shl_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn shl_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn shl_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
