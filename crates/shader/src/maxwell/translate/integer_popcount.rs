//! Stubs until ported from the C++ integer_popcount.cpp.

use super::{TResult, Translator};

pub(super) fn popc_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn popc_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn popc_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
