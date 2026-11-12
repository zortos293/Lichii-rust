//! Stubs until ported from the C++ floating_point_multi_function.cpp.

use super::{TResult, Translator};

pub(super) fn mufu(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
