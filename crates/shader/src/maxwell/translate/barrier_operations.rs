//! Stubs until ported from the C++ barrier_operations.cpp.

use super::{TResult, Translator};

pub(super) fn bar(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn depbar(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn membar(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
