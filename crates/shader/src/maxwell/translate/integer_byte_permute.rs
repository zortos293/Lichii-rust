//! Stubs until ported from the C++ integer_byte_permute.cpp.

use super::{TResult, Translator};

pub(super) fn prmt_cr(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn prmt_rc(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn prmt_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn prmt_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
