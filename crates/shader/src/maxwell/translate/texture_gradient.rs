//! Stubs until ported from the C++ texture_gradient.cpp.

use super::{TResult, Translator};

pub(super) fn txd(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn txd_b(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
