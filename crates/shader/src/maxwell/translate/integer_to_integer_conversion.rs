//! Stubs until ported from the C++ integer_to_integer_conversion.cpp.

use super::{TResult, Translator};

pub(super) fn i2i_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn i2i_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn i2i_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
