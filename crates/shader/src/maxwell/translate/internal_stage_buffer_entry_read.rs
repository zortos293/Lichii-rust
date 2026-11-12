//! Stubs until ported from the C++ internal_stage_buffer_entry_read.cpp.

use super::{TResult, Translator};

pub(super) fn isberd(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn isbe_nx2(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
