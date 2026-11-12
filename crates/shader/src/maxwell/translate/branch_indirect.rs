//! Stubs until ported from the C++ branch_indirect.cpp.

use super::{TResult, Translator};

pub(super) fn brx(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn jmx(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
