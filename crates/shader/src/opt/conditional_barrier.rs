//! Drops barriers the host cannot run in conditional control flow, ported from
//! `conditional_barrier_pass.cpp`.

use crate::info::ShaderInfo;
use crate::ir::{Opcode, Program, SyntaxNode};

/// Removes every `Barrier` inside an if or loop, or after a return taken conditionally, and
/// sets `removed_conditional_barrier` when it removed one.
pub fn conditional_barrier(prog: &mut Program, info: &mut ShaderInfo) {
    let mut conditional_depth = 0i32;
    let mut conditional_returns = 0u32;
    for i in 0..prog.syntax_list.len() {
        match prog.syntax_list[i] {
            SyntaxNode::If { .. } | SyntaxNode::Loop { .. } => conditional_depth += 1,
            SyntaxNode::EndIf { .. } | SyntaxNode::Repeat { .. } => conditional_depth -= 1,
            SyntaxNode::Return | SyntaxNode::Unreachable => {
                if conditional_depth > 0 {
                    conditional_returns += 1;
                }
            }
            SyntaxNode::Block(block) if conditional_depth > 0 || conditional_returns > 0 => {
                let mut cur = prog.block(block).first();
                while let Some(id) = cur {
                    cur = prog.inst(id).next();
                    if prog.op(id) == Opcode::Barrier {
                        crate::fallback::note("BAR", "conditional_barrier_removed");
                        prog.remove(id);
                        info.removed_conditional_barrier = true;
                    }
                }
            }
            _ => {}
        }
    }
    debug_assert_eq!(conditional_depth, 0, "unbalanced structured control flow");
}
