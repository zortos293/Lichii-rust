//! Stubs until ported from the C++ floating_point_swizzled_add.cpp.

use super::{TResult, Translator};

pub(super) fn fswzadd(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
