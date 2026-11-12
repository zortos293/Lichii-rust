//! Stubs until ported from the C++ load_store_attribute.cpp.

use super::{TResult, Translator};

pub(super) fn ald(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn ast(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn ipa(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
