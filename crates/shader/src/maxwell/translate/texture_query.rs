//! Stubs until ported from the C++ texture_query.cpp.

use super::{TResult, Translator};

pub(super) fn txq(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn txq_b(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
