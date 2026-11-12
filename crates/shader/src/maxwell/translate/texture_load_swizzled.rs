//! Stubs until ported from the C++ texture_load_swizzled.cpp.

use super::{TResult, Translator};

pub(super) fn tlds(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
