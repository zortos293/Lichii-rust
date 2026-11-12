//! Stubs until ported from the C++ integer_funnel_shift.cpp.

use super::{TResult, Translator};

pub(super) fn shf_lreg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn shf_rreg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn shf_limm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn shf_rimm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
