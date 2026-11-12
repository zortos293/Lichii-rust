use super::*;
use crate::ir::{Pred, Reg};
use crate::{ShaderArch, ShaderStage};

struct MockEnv {
    cbuf: Vec<u32>,
}

impl ShaderEnvironment for MockEnv {
    fn arch(&self) -> ShaderArch {
        ShaderArch::Maxwell
    }
    fn stage(&self) -> ShaderStage {
        ShaderStage::Compute
    }
    fn start_address(&self) -> lichii_common::GpuVAddr {
        0x1000
    }
    fn read_u64(&self, _: lichii_common::GpuVAddr) -> u64 {
        0
    }
    fn read_cbuf(&self, index: u32, offset: u32) -> u32 {
        assert_eq!(index, 2);
        self.cbuf[(offset / 4) as usize]
    }
    fn shared_memory_size(&self) -> u32 {
        0
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [1, 1, 1]
    }
}

struct MockIsa(Vec<FlowInst>);

impl FlowIsa for MockIsa {
    fn align(&self, offset: u32) -> u32 {
        offset
    }
    fn next(&self, pc: u32) -> u32 {
        pc + 8
    }
    fn prev(&self, pc: u32) -> u32 {
        pc - 8
    }
    fn classify(&self, _: &dyn ShaderEnvironment, pc: u32) -> Result<FlowInst, TranslateError> {
        Ok(self.0[(pc / 8) as usize])
    }
    fn track_indirect_branch(
        &self,
        _: &dyn ShaderEnvironment,
        _: u32,
        _: u32,
    ) -> Option<IndirectBranchTable> {
        Some(IndirectBranchTable {
            cbuf_index: 2,
            cbuf_offset: 0,
            num_entries: 3,
            branch_offset: 8,
            reg: Reg::r(4),
        })
    }
}

fn op(op: FlowOp) -> FlowInst {
    FlowInst {
        op,
        cond: Condition::TRUE,
    }
}

fn guarded(op: FlowOp, pred: u8) -> FlowInst {
    FlowInst {
        op,
        cond: Condition::pred(Pred::p(pred), false),
    }
}

const N: FlowOp = FlowOp::Normal;

fn build(program: Vec<FlowInst>) -> Cfg {
    build_with(program, false)
}

fn build_with(program: Vec<FlowInst>, dispatcher: bool) -> Cfg {
    let env = MockEnv {
        cbuf: vec![0, 16, 16, 32],
    };
    Cfg::build(&MockIsa(program), &env, 0, dispatcher).expect("cfg")
}

/// Begin, end, virtual, end class, true target begin, false target begin.
type Shape = (u32, u32, bool, EndClass, Option<u32>, Option<u32>);

fn shape(cfg: &Cfg, function: usize) -> Vec<Shape> {
    cfg.functions[function]
        .blocks()
        .map(|id| {
            let b = cfg.block(id);
            let target = |t: Option<BlockId>| t.map(|t| cfg.block(t).begin);
            (
                b.begin,
                b.end,
                b.is_virtual,
                b.end_class,
                target(b.branch_true),
                target(b.branch_false),
            )
        })
        .collect()
}

#[test]
fn straight_line_exit() {
    let cfg = build(vec![op(N), op(N), op(FlowOp::Exit)]);
    assert_eq!(
        shape(&cfg, 0),
        vec![(0, 24, false, EndClass::Exit, None, None)]
    );
}

#[test]
fn conditional_forward_branch() {
    let cfg = build(vec![
        op(N),
        guarded(FlowOp::Branch { target: 24 }, 0),
        op(N),
        op(FlowOp::Exit),
    ]);
    assert_eq!(
        shape(&cfg, 0),
        vec![
            (0, 8, false, EndClass::Branch, Some(24), Some(16)),
            (16, 24, false, EndClass::Branch, Some(24), None),
            (24, 32, false, EndClass::Exit, None, None),
        ]
    );
    let first = cfg.functions[0].blocks().next().unwrap();
    assert_eq!(cfg.block(first).cond, Condition::pred(Pred::p(0), false));
}

#[test]
fn predicated_instruction_gets_virtual_block() {
    let cfg = build(vec![guarded(N, 1), op(FlowOp::Exit)]);
    assert_eq!(
        shape(&cfg, 0),
        vec![
            (0, 0, true, EndClass::Branch, Some(0), Some(8)),
            (0, 8, false, EndClass::Branch, Some(8), None),
            (8, 16, false, EndClass::Exit, None, None),
        ]
    );
}

#[test]
fn predicated_instruction_mid_block_splits_first() {
    let cfg = build(vec![op(N), guarded(N, 2), op(FlowOp::Exit)]);
    assert_eq!(
        shape(&cfg, 0),
        vec![
            (0, 8, false, EndClass::Branch, Some(8), None),
            (8, 8, true, EndClass::Branch, Some(8), Some(16)),
            (8, 16, false, EndClass::Branch, Some(16), None),
            (16, 24, false, EndClass::Exit, None, None),
        ]
    );
}

#[test]
fn backward_branch_splits_visited_block() {
    let cfg = build(vec![
        op(N),
        op(N),
        guarded(FlowOp::Branch { target: 8 }, 0),
        op(FlowOp::Exit),
    ]);
    assert_eq!(
        shape(&cfg, 0),
        vec![
            (0, 8, false, EndClass::Branch, Some(8), None),
            (8, 16, false, EndClass::Branch, Some(8), Some(24)),
            (24, 32, false, EndClass::Exit, None, None),
        ]
    );
}

#[test]
fn ssy_sync_reconverges() {
    let cfg = build(vec![
        op(FlowOp::Push {
            token: Token::Ssy,
            target: 32,
        }),
        guarded(FlowOp::Branch { target: 24 }, 0),
        op(FlowOp::Pop { token: Token::Ssy }),
        op(N),
        op(FlowOp::Exit),
    ]);
    assert_eq!(
        shape(&cfg, 0),
        vec![
            (0, 8, false, EndClass::Branch, Some(24), Some(16)),
            (16, 16, false, EndClass::Branch, Some(32), None),
            (24, 32, false, EndClass::Branch, Some(32), None),
            (32, 40, false, EndClass::Exit, None, None),
        ]
    );
    let sync = cfg.functions[0].blocks().nth(1).unwrap();
    assert_eq!(cfg.block(sync).stack.peek(Token::Ssy), Some(32));
}

#[test]
fn unmatched_pop_is_an_error() {
    let env = MockEnv { cbuf: vec![] };
    let isa = MockIsa(vec![op(FlowOp::Pop { token: Token::Pbk })]);
    assert!(matches!(
        Cfg::build(&isa, &env, 0, false),
        Err(TranslateError::Unstructured { address: 0x1000 })
    ));
}

#[test]
fn indirect_branch_reads_jump_table() {
    let cfg = build(vec![
        op(FlowOp::IndirectBranch { absolute: false }),
        op(N),
        op(N),
        op(FlowOp::Exit),
        op(FlowOp::Exit),
        op(FlowOp::Exit),
    ]);
    let entry = cfg.block(cfg.functions[0].blocks().next().unwrap());
    assert_eq!(entry.end_class, EndClass::IndirectBranch);
    assert_eq!(entry.branch_reg, Reg::r(4));
    assert_eq!(entry.branch_offset, 8);
    let targets: Vec<u32> = entry.indirect_branches.iter().map(|&(_, t)| t).collect();
    assert_eq!(targets, vec![8, 24]);
}

#[test]
fn call_creates_function() {
    let cfg = build(vec![
        op(FlowOp::Call { target: 24 }),
        op(FlowOp::Exit),
        op(N),
        op(N),
        op(FlowOp::Return),
    ]);
    assert_eq!(cfg.functions.len(), 2);
    assert_eq!(cfg.functions[1].entry, 24);
    assert_eq!(
        shape(&cfg, 1),
        vec![(24, 32, false, EndClass::Return, None, None)]
    );
    let entry = cfg.block(cfg.functions[0].blocks().next().unwrap());
    assert_eq!(entry.end_class, EndClass::Call);
    assert_eq!(entry.function_call, 1);
    assert_eq!(entry.return_block.map(|r| cfg.block(r).begin), Some(8));
}

#[test]
fn exits_to_dispatcher() {
    let cfg = build_with(
        vec![guarded(FlowOp::Exit, 0), op(N), op(FlowOp::Exit)],
        true,
    );
    assert_eq!(
        shape(&cfg, 0),
        vec![
            (0, 0, false, EndClass::Branch, Some(24), Some(8)),
            (8, 16, false, EndClass::Branch, Some(24), None),
            (24, 24, false, EndClass::Exit, None, None),
        ]
    );
}

#[test]
fn dot_renders() {
    let cfg = build(vec![
        guarded(FlowOp::Branch { target: 16 }, 3),
        op(FlowOp::Kill),
        op(FlowOp::Exit),
    ]);
    let dot = cfg.dot();
    assert!(dot.starts_with("digraph shader {"));
    assert!(dot.contains("Kill"));
}

mod structured {
    use super::*;
    use crate::flow::structure::{StructureOptions, structure};
    use crate::ir::{Opcode, Program, Value};

    fn run(program: Vec<FlowInst>, options: StructureOptions) -> String {
        let cfg = build(program);
        let mut prog = Program::new();
        let mut translate = |prog: &mut Program, block, begin: u32, end: u32| {
            prog.append(
                block,
                Opcode::Reference,
                &[Value::U32(begin << 16 | end)],
                0,
            );
            Ok(())
        };
        structure(&mut prog, &cfg, options, &mut translate).expect("structure");
        prog.dump()
    }

    #[test]
    fn forward_branch_becomes_if() {
        let dump = run(
            vec![
                op(N),
                guarded(FlowOp::Branch { target: 24 }, 0),
                op(N),
                op(FlowOp::Exit),
            ],
            StructureOptions::default(),
        );
        assert!(dump.contains("syntax:\n  block {0}\n  if %"), "{dump}");
        assert!(dump.contains("Reference #0x8\n"), "{dump}");
        assert!(dump.contains("Reference #0x100018\n"), "{dump}");
        assert!(dump.contains("Reference #0x180020\n"), "{dump}");
        assert!(dump.contains("  endif {"), "{dump}");
        assert!(dump.trim_end().ends_with("return"), "{dump}");
        assert!(dump.contains("Prologue"), "{dump}");
        assert!(dump.contains("Epilogue"), "{dump}");
    }

    #[test]
    fn backward_branch_becomes_loop() {
        let dump = run(
            vec![
                op(N),
                op(N),
                guarded(FlowOp::Branch { target: 8 }, 1),
                op(FlowOp::Exit),
            ],
            StructureOptions::default(),
        );
        assert!(dump.contains("  loop {"), "{dump}");
        assert!(dump.contains("  repeat %"), "{dump}");
        assert!(dump.contains("GetPred P1"), "{dump}");
    }

    #[test]
    fn indirect_branch_compares_variable() {
        let dump = run(
            vec![
                op(FlowOp::IndirectBranch { absolute: false }),
                op(FlowOp::Exit),
                op(N),
                op(FlowOp::Exit),
            ],
            StructureOptions::default(),
        );
        assert!(dump.contains("SetIndirectBranchVariable"), "{dump}");
        assert!(dump.contains("GetIndirectBranchVariable"), "{dump}");
        assert!(dump.contains("IEqual"), "{dump}");
    }

    #[test]
    fn kill_demotes_and_reorders_before_return() {
        let program = vec![guarded(FlowOp::Kill, 2), op(N), op(FlowOp::Exit)];
        let dump = run(program.clone(), StructureOptions::default());
        assert!(dump.contains("DemoteToHelperInvocation"), "{dump}");
        let reordered = run(
            program,
            StructureOptions {
                reorder_demotes: true,
                ..Default::default()
            },
        );
        let syntax = reordered.split("syntax:").nth(1).unwrap();
        let demote_if = syntax.rfind("  if ").unwrap();
        let ret = syntax.rfind("return").unwrap();
        assert!(demote_if < ret, "{reordered}");
    }

    #[test]
    fn call_is_inlined() {
        let dump = run(
            vec![
                op(FlowOp::Call { target: 24 }),
                op(FlowOp::Exit),
                op(N),
                op(N),
                op(FlowOp::Return),
            ],
            StructureOptions::default(),
        );
        assert!(dump.contains("Reference #0x180020\n"), "{dump}");
        assert!(dump.contains("Reference #0x80010\n"), "{dump}");
    }

    #[test]
    fn converged_exits_share_one_return() {
        let dump = run(
            vec![guarded(FlowOp::Exit, 0), op(N), op(FlowOp::Exit)],
            StructureOptions {
                converge_exits: true,
                ..Default::default()
            },
        );
        assert_eq!(dump.matches("Epilogue").count(), 1, "{dump}");
    }
}
#[test]
fn zz_print() {
    let cfg = build(vec![
        op(N),
        op(N),
        guarded(FlowOp::Branch { target: 8 }, 1),
        op(FlowOp::Exit),
    ]);
    let mut prog = crate::ir::Program::new();
    let mut translate = |prog: &mut crate::ir::Program, block, begin: u32, end: u32| {
        prog.append(
            block,
            crate::ir::Opcode::Reference,
            &[crate::ir::Value::U32(begin << 16 | end)],
            0,
        );
        Ok(())
    };
    crate::flow::structure::structure(&mut prog, &cfg, Default::default(), &mut translate).unwrap();
    println!("{}", prog.dump());
}
