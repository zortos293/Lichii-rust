//! Stubs until ported from the C++ surface_atomic_operations.cpp.

use super::{TResult, Translator};

pub(super) fn sured(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn suatom(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
