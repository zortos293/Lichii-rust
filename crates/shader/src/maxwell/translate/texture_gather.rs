//! Stubs until ported from the C++ texture_gather.cpp.

use super::{TResult, Translator};

pub(super) fn tld4_b(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn tld4(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
