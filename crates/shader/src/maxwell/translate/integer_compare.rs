//! Stubs until ported from the C++ integer_compare.cpp.

use super::{TResult, Translator};

pub(super) fn icmp_cr(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn icmp_rc(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn icmp_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn icmp_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
