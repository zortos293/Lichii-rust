//! Stubs until ported from the C++ attribute_memory_to_physical.cpp.

use super::{TResult, Translator};

pub(super) fn al2p(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
