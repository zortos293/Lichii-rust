//! Stubs until ported from the C++ texture_load.cpp.

use super::{TResult, Translator};

pub(super) fn tld(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn tld_b(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
