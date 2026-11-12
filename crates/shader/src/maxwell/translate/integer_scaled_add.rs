//! Stubs until ported from the C++ integer_scaled_add.cpp.

use super::{TResult, Translator};

pub(super) fn iscadd_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn iscadd_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn iscadd_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn iscadd32i(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
