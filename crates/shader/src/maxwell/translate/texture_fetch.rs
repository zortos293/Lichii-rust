//! Stubs until ported from the C++ texture_fetch.cpp.

use super::{TResult, Translator};

pub(super) fn tex_b(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn tex(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
