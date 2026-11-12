//! Stubs until ported from the C++ load_constant.cpp.

use super::{TResult, Translator};

pub(super) fn ldc(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
