//! Stubs until ported from the C++ video_minimum_maximum.cpp.

use super::{TResult, Translator};

pub(super) fn vmnmx(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
