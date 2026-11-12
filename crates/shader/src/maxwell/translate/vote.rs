//! Stubs until ported from the C++ vote.cpp.

use super::{TResult, Translator};

pub(super) fn vote(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn vote_vtg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
