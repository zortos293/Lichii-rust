use super::*;
use crate::ir::{InstId, Reg};

fn blocks(prog: &mut Program, n: usize) -> Vec<BlockId> {
    let blocks: Vec<BlockId> = (0..n).map(|_| prog.new_block()).collect();
    prog.block_order = blocks.clone();
    prog.syntax_list = blocks.iter().map(|&b| SyntaxNode::Block(b)).collect();
    blocks
}

fn set(prog: &mut Program, block: BlockId, value: u32) {
    prog.append(
        block,
        Opcode::SetRegister,
        &[Value::Reg(Reg::r(1)), Value::U32(value)],
        0,
    );
}

fn use_r1(prog: &mut Program, block: BlockId) -> InstId {
    let get = prog.append(block, Opcode::GetRegister, &[Value::Reg(Reg::r(1))], 0);
    prog.append(block, Opcode::Reference, &[Value::Inst(get)], 0)
}

fn optimize(prog: &mut Program) {
    compute_post_order(prog);
    ssa_rewrite(prog);
    identity_removal(prog);
    dead_code_elimination(prog);
}

fn phi_operands(prog: &Program, value: Value) -> Vec<Value> {
    let Value::Inst(phi) = value else {
        panic!("{value:?} is not a phi");
    };
    assert_eq!(prog.op(phi), Opcode::Phi);
    prog.inst(phi).args().to_vec()
}

#[test]
fn straight_line_read_forwards_the_write() {
    let mut prog = Program::new();
    let b = blocks(&mut prog, 1);
    set(&mut prog, b[0], 5);
    let r = use_r1(&mut prog, b[0]);
    optimize(&mut prog);
    assert_eq!(prog.arg(r, 0), Value::U32(5));
}

#[test]
fn diamond_merges_with_a_phi() {
    let mut prog = Program::new();
    let b = blocks(&mut prog, 4);
    prog.add_branch(b[0], b[1]);
    prog.add_branch(b[0], b[2]);
    prog.add_branch(b[1], b[3]);
    prog.add_branch(b[2], b[3]);
    set(&mut prog, b[1], 1);
    set(&mut prog, b[2], 2);
    let r = use_r1(&mut prog, b[3]);
    optimize(&mut prog);
    let mut ops = phi_operands(&prog, prog.arg(r, 0));
    ops.sort_by_key(|v| v.imm_u32());
    assert_eq!(ops, vec![Value::U32(1), Value::U32(2)]);
}

#[test]
fn one_sided_write_reads_undef_on_the_other_path() {
    let mut prog = Program::new();
    let b = blocks(&mut prog, 3);
    prog.add_branch(b[0], b[1]);
    prog.add_branch(b[0], b[2]);
    prog.add_branch(b[1], b[2]);
    set(&mut prog, b[1], 1);
    let r = use_r1(&mut prog, b[2]);
    optimize(&mut prog);
    let ops = phi_operands(&prog, prog.arg(r, 0));
    assert_eq!(ops.len(), 2);
    assert!(ops.contains(&Value::U32(1)));
    assert!(
        ops.iter()
            .any(|&v| matches!(v, Value::Inst(id) if prog.op(id) == Opcode::UndefU32))
    );
}

#[test]
fn a_merge_reached_from_an_unreachable_join_still_merges_the_reachable_writes() {
    let mut prog = Program::new();
    let b = blocks(&mut prog, 7);
    prog.add_branch(b[0], b[1]);
    prog.add_branch(b[0], b[2]);
    prog.add_branch(b[1], b[3]);
    prog.add_branch(b[2], b[3]);
    prog.add_branch(b[5], b[4]);
    prog.add_branch(b[6], b[4]);
    prog.add_branch(b[4], b[3]);
    set(&mut prog, b[1], 1);
    set(&mut prog, b[2], 2);
    let r = use_r1(&mut prog, b[3]);
    optimize(&mut prog);
    let ops = phi_operands(&prog, prog.arg(r, 0));
    assert!(
        ops.contains(&Value::U32(1)) && ops.contains(&Value::U32(2)),
        "{ops:?}"
    );
}

#[test]
fn an_unwritten_read_in_an_entry_block_with_a_back_edge_resolves_in_that_block() {
    let mut prog = Program::new();
    let b = blocks(&mut prog, 3);
    prog.add_branch(b[0], b[1]);
    prog.add_branch(b[1], b[0]);
    prog.add_branch(b[1], b[2]);
    prog.add_branch(b[2], b[0]);
    let r = use_r1(&mut prog, b[0]);
    optimize(&mut prog);
    let Value::Inst(value) = prog.arg(r, 0) else {
        panic!("{:?}", prog.arg(r, 0));
    };
    assert!(prog.insts_of(b[0]).any(|id| id == value), "{}", prog.dump());
}

#[test]
fn same_value_on_both_paths_needs_no_phi() {
    let mut prog = Program::new();
    let b = blocks(&mut prog, 4);
    set(&mut prog, b[0], 9);
    prog.add_branch(b[0], b[1]);
    prog.add_branch(b[0], b[2]);
    prog.add_branch(b[1], b[3]);
    prog.add_branch(b[2], b[3]);
    let r = use_r1(&mut prog, b[3]);
    optimize(&mut prog);
    assert_eq!(prog.arg(r, 0), Value::U32(9));
    assert!(
        b.iter()
            .all(|&blk| prog.insts_of(blk).all(|id| prog.op(id) != Opcode::Phi))
    );
}

#[test]
fn loop_header_gets_a_phi_from_the_back_edge() {
    let mut prog = Program::new();
    let b = blocks(&mut prog, 5);
    prog.add_branch(b[0], b[1]);
    prog.add_branch(b[1], b[2]);
    prog.add_branch(b[2], b[3]);
    prog.add_branch(b[3], b[1]);
    prog.add_branch(b[3], b[4]);
    set(&mut prog, b[0], 0);
    let header = use_r1(&mut prog, b[1]);
    set(&mut prog, b[2], 7);
    let exit = use_r1(&mut prog, b[4]);
    optimize(&mut prog);
    let mut ops = phi_operands(&prog, prog.arg(header, 0));
    ops.sort_by_key(|v| v.imm_u32());
    assert_eq!(ops, vec![Value::U32(0), Value::U32(7)]);
    assert_eq!(prog.arg(exit, 0), Value::U32(7));
}

#[test]
fn post_order_visits_successors_first() {
    let mut prog = Program::new();
    let b = blocks(&mut prog, 3);
    prog.add_branch(b[0], b[1]);
    prog.add_branch(b[1], b[2]);
    compute_post_order(&mut prog);
    assert_eq!(prog.post_order, vec![b[2], b[1], b[0]]);
}
