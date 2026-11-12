//! Stubs until ported from the C++ surface_load_store.cpp.

use super::{TResult, Translator};

pub(super) fn suld(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn sust(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
