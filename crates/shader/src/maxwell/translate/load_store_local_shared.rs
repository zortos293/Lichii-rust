//! Stubs until ported from the C++ load_store_local_shared.cpp.

use super::{TResult, Translator};

pub(super) fn ldl(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn lds(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn stl(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn sts(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
