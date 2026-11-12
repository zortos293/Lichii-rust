//! Stubs until ported from the C++ warp_shuffle.cpp.

use super::{TResult, Translator};

pub(super) fn shfl(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
