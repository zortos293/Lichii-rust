//! Stubs until ported from the C++ integer_shift_right.cpp.

use super::{TResult, Translator};

pub(super) fn shr_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn shr_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn shr_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
