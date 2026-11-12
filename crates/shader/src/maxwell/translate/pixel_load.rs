//! Stubs until ported from the C++ pixel_load.cpp.

use super::{TResult, Translator};

pub(super) fn pixld(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
