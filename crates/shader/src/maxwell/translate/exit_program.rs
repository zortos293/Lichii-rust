//! Stubs until ported from the C++ exit_program.cpp.

use super::{TResult, Translator};

pub(super) fn exit(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
