//! Stubs until ported from the C++ condition_code_set.cpp.

use super::{TResult, Translator};

pub(super) fn cset(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn csetp(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
