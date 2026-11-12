//! Splits a dual vertex program pair (VertexA then VertexB), ported from `dual_vertex_pass.cpp`.

use crate::ir::{Opcode, Program};

fn invalidate_first(prog: &mut Program, op: Opcode) {
    let found = prog
        .block_order
        .iter()
        .flat_map(|&block| prog.insts_of(block))
        .find(|&id| prog.op(id) == op);
    if let Some(id) = found {
        prog.invalidate(id);
    }
}

/// Drops the epilogue of the first program of a pair, which the second program continues.
pub fn vertex_a_transform(prog: &mut Program) {
    invalidate_first(prog, Opcode::Epilogue);
}

/// Drops the prologue of the second program of a pair, which continues the first.
pub fn vertex_b_transform(prog: &mut Program) {
    invalidate_first(prog, Opcode::Prologue);
}
