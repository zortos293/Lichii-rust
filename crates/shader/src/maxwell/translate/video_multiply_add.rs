//! Stubs until ported from the C++ video_multiply_add.cpp.

use super::{TResult, Translator};

pub(super) fn vmad(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
