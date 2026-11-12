//! Stubs until ported from the C++ texture_gather_swizzled.cpp.

use super::{TResult, Translator};

pub(super) fn tld4s(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
