//! IR passes. Every pass is linear or n log n in program size.

mod collect_shader_info;
mod conditional_barrier;
mod dual_vertex;
mod geometry_compaction;
#[cfg(test)]
mod info_tests;
mod layer;
mod position;
mod ssa_rewrite;
#[cfg(test)]
mod tests;
mod vendor_workaround;

pub use collect_shader_info::collect_shader_info;
pub use conditional_barrier::conditional_barrier;
pub use dual_vertex::{vertex_a_transform, vertex_b_transform};
pub use geometry_compaction::geometry_compaction;
pub use layer::layer;
pub use position::{blit_constants, position};
pub use ssa_rewrite::ssa_rewrite;
pub use vendor_workaround::{has_broken_pattern, vendor_workaround};

use crate::ir::{BlockId, Opcode, Program, SyntaxNode, Value};

/// Computes `Program::post_order` by a depth-first walk of the block graph from the entry.
pub fn compute_post_order(prog: &mut Program) {
    let Some(&SyntaxNode::Block(entry)) = prog.syntax_list.first() else {
        prog.post_order.clear();
        return;
    };
    let mut visited = vec![false; prog.num_blocks()];
    let mut stack = vec![entry];
    let mut post_order = Vec::with_capacity(prog.num_blocks());
    visited[entry.0 as usize] = true;
    while let Some(block) = stack.pop() {
        let next = prog
            .block(block)
            .successors
            .iter()
            .copied()
            .find(|s| !visited[s.0 as usize]);
        match next {
            Some(succ) => {
                visited[succ.0 as usize] = true;
                stack.push(block);
                stack.push(succ);
            }
            None => post_order.push(block),
        }
    }
    prog.post_order = post_order;
}

/// Removes instructions with no uses and no side effects, walking backwards so whole chains
/// die in one pass.
pub fn dead_code_elimination(prog: &mut Program) {
    for i in 0..prog.post_order.len() {
        let mut cur = prog.block(prog.post_order[i]).last();
        while let Some(id) = cur {
            cur = prog.inst(id).prev();
            let inst = prog.inst(id);
            if !inst.has_uses() && !inst.op.may_have_side_effects() {
                prog.remove(id);
            }
        }
    }
}

/// Points arguments that name an identity at the forwarded value, then drops identities and
/// voided instructions.
pub fn identity_removal(prog: &mut Program) {
    let mut dropped = Vec::new();
    for i in 0..prog.block_order.len() {
        let block: BlockId = prog.block_order[i];
        let mut cur = prog.block(block).first();
        while let Some(id) = cur {
            cur = prog.inst(id).next();
            for arg in 0..prog.inst(id).num_args() {
                let value = prog.arg(id, arg);
                if let Value::Inst(target) = value
                    && prog.op(target) == Opcode::Identity
                {
                    let resolved = prog.resolve(value);
                    prog.set_arg(id, arg, resolved);
                }
            }
            if matches!(prog.op(id), Opcode::Identity | Opcode::Void) {
                prog.unlink(id);
                dropped.push(id);
            }
        }
    }
    for id in dropped {
        prog.invalidate(id);
    }
}
