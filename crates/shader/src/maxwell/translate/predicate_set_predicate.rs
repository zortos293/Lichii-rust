//! Stubs until ported from the C++ predicate_set_predicate.cpp.

use super::{TResult, Translator};

pub(super) fn psetp(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
