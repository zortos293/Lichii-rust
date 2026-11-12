//! Stubs until ported from the C++ atomic_operations_global_memory.cpp.

use super::{TResult, Translator};

pub(super) fn red(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn atom_cas(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn atom(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
