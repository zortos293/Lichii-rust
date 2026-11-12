//! Stubs until ported from the C++ predicate_set_register.cpp.

use super::{TResult, Translator};

pub(super) fn pset(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
