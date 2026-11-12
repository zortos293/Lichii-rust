//! Stubs until ported from the C++ video_set_predicate.cpp.

use super::{TResult, Translator};

pub(super) fn vsetp(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
