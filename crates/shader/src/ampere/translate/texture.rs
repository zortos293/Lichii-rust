//! Texture sampling, fetches and queries, and surface loads, stores and atomics.

use super::{TResult, Translator};
use crate::ampere::opcode::Opcode;

pub(super) fn translate(t: &mut Translator<'_>, op: Opcode) -> TResult {
    let _ = op;
    Err(t.unsupported())
}
