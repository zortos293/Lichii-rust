//! Stubs until ported from the C++ logic_operation_three_input.cpp.

use super::{TResult, Translator};

pub(super) fn lop3_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn lop3_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn lop3_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
