//! Stubs until ported from the C++ integer_add_three_input.cpp.

use super::{TResult, Translator};

pub(super) fn iadd3_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn iadd3_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn iadd3_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
