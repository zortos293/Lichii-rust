//! Stubs until ported from the C++ tensor_mul_add.cpp.

use super::{TResult, Translator};

pub(super) fn tensor_mul_add(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
