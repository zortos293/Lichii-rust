//! Stubs until ported from the C++ atomic_operations_shared_memory.cpp.

use super::{TResult, Translator};

pub(super) fn atoms(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
