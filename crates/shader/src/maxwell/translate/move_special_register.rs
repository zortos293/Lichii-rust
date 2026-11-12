//! Stubs until ported from the C++ move_special_register.cpp.

use super::{TResult, Translator};

pub(super) fn s2r(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
