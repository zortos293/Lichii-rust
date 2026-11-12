//! Text dump of a program, used by golden tests and debugging.

use std::fmt::Write as _;

use rustc_hash::FxHashMap;

use super::program::{InstId, Program, SyntaxNode, Value};
use super::types::Type;

impl Program {
    /// Renders every block in program order, numbering values in order of appearance, then the
    /// structured control flow.
    pub fn dump(&self) -> String {
        let mut names: FxHashMap<InstId, usize> = FxHashMap::default();
        let mut out = String::new();
        let order: Vec<_> = if self.block_order.is_empty() {
            (0..self.num_blocks() as u32).map(super::BlockId).collect()
        } else {
            self.block_order.clone()
        };
        for block in order {
            let _ = writeln!(out, "{{{}}}:", block.0);
            let mut cur = self.block(block).first();
            while let Some(id) = cur {
                let inst = self.inst(id);
                cur = inst.next();
                out.push_str("  ");
                if self.type_of(Value::Inst(id)) != Type::VOID {
                    let n = names.len();
                    let n = *names.entry(id).or_insert(n);
                    let _ = write!(out, "%{n} = ");
                }
                out.push_str(inst.op.name());
                for (i, &arg) in inst.args().iter().enumerate() {
                    out.push_str(if i == 0 { " " } else { ", " });
                    self.write_value(&mut out, &mut names, arg);
                    if inst.op == super::Opcode::Phi {
                        let _ = write!(out, " <- {{{}}}", inst.phi_block(i).0);
                    }
                }
                if inst.flags != 0 && inst.op != super::Opcode::Phi {
                    let _ = write!(out, " (flags {:#x})", inst.flags);
                }
                out.push('\n');
            }
        }
        if !self.syntax_list.is_empty() {
            out.push_str("syntax:\n");
            for node in &self.syntax_list {
                out.push_str("  ");
                match *node {
                    SyntaxNode::Block(b) => {
                        let _ = write!(out, "block {{{}}}", b.0);
                    }
                    SyntaxNode::If { cond, body, merge } => {
                        out.push_str("if ");
                        self.write_value(&mut out, &mut names, cond);
                        let _ = write!(out, " then {{{}}} merge {{{}}}", body.0, merge.0);
                    }
                    SyntaxNode::EndIf { merge } => {
                        let _ = write!(out, "endif {{{}}}", merge.0);
                    }
                    SyntaxNode::Loop {
                        body,
                        continue_block,
                        merge,
                    } => {
                        let _ = write!(
                            out,
                            "loop {{{}}} continue {{{}}} merge {{{}}}",
                            body.0, continue_block.0, merge.0
                        );
                    }
                    SyntaxNode::Repeat {
                        cond,
                        loop_header,
                        merge,
                    } => {
                        out.push_str("repeat ");
                        self.write_value(&mut out, &mut names, cond);
                        let _ = write!(out, " header {{{}}} merge {{{}}}", loop_header.0, merge.0);
                    }
                    SyntaxNode::Break { cond, merge, skip } => {
                        out.push_str("break ");
                        self.write_value(&mut out, &mut names, cond);
                        let _ = write!(out, " merge {{{}}} skip {{{}}}", merge.0, skip.0);
                    }
                    SyntaxNode::Return => out.push_str("return"),
                    SyntaxNode::Unreachable => out.push_str("unreachable"),
                }
                out.push('\n');
            }
        }
        out
    }

    fn write_value(&self, out: &mut String, names: &mut FxHashMap<InstId, usize>, value: Value) {
        let _ = match value {
            Value::Void => write!(out, "void"),
            Value::Inst(id) => {
                let n = names.len();
                let n = *names.entry(id).or_insert(n);
                write!(out, "%{n}")
            }
            Value::Reg(r) => write!(out, "{r}"),
            Value::Pred(p) => write!(out, "{p}"),
            Value::Attribute(a) => write!(out, "{a}"),
            Value::Patch(p) => write!(out, "{p}"),
            Value::U1(v) => write!(out, "#{v}"),
            Value::U8(v) => write!(out, "#{v}u8"),
            Value::U16(v) => write!(out, "#{v}u16"),
            Value::U32(v) => write!(out, "#{v:#x}"),
            Value::U64(v) => write!(out, "#{v:#x}u64"),
            Value::F32(v) => write!(out, "#{v:?}f"),
            Value::F64(v) => write!(out, "#{v:?}d"),
        };
    }
}
