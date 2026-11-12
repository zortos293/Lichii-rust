//! Stubs until ported from the C++ texture_mipmap_level.cpp.

use super::{TResult, Translator};

pub(super) fn tmml(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn tmml_b(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
