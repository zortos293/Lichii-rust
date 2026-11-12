//! Stubs until ported from the C++ texture_fetch_swizzled.cpp.

use super::{TResult, Translator};

pub(super) fn texs(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
