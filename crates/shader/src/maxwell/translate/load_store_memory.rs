//! Stubs until ported from the C++ load_store_memory.cpp.

use super::{TResult, Translator};

pub(super) fn ldg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn stg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
