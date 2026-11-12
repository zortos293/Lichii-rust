//! Stubs until ported from the C++ integer_floating_point_conversion.cpp.

use super::{TResult, Translator};

pub(super) fn i2f_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn i2f_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn i2f_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
