//! Structural checks of a program, for tests and debug builds.

use crate::ir::{InstId, Opcode, Program, Type, Value};

/// Checks what every pass relies on: argument types match the opcode signature, use counts
/// match the arguments that name each instruction, instructions other than phis only read
/// values defined earlier in program order, and phis lead their block.
pub fn verify(prog: &Program) -> Result<(), String> {
    let mut uses = vec![0u32; prog.num_insts()];
    let mut defined = vec![false; prog.num_insts()];
    for &block in &prog.block_order {
        let mut past_phis = false;
        for id in prog.insts_of(block) {
            let inst = prog.inst(id);
            defined[id.0 as usize] = true;
            let fail = |what: &str| Err(format!("{what} in block {}:\n{}", block.0, prog.dump()));
            if inst.op == Opcode::Phi {
                if past_phis {
                    return fail("interleaved phi nodes");
                }
                for arg in inst.args() {
                    if let Value::Inst(target) = *arg {
                        uses[target.0 as usize] += 1;
                    }
                }
                continue;
            }
            past_phis = true;
            let expected = inst.op.arg_types();
            for i in 0..inst.num_args().max(expected.len()) {
                let arg = inst.args().get(i).copied().unwrap_or(Value::Void);
                let want = expected.get(i).copied().unwrap_or(Type::VOID);
                if !prog.type_of(arg).compatible(want) {
                    return fail(&format!(
                        "argument {i} of {} is {}, expected {want}",
                        inst.op.name(),
                        prog.type_of(arg)
                    ));
                }
                if let Value::Inst(target) = arg {
                    if !defined[target.0 as usize] {
                        return fail(&format!("{} reads a later definition", inst.op.name()));
                    }
                    uses[target.0 as usize] += 1;
                }
            }
        }
    }
    for (index, &count) in uses.iter().enumerate() {
        let inst = prog.inst(InstId(index as u32));
        if count > 0 && inst.use_count() != count {
            return Err(format!(
                "{} has use count {} but {count} uses:\n{}",
                inst.op.name(),
                inst.use_count(),
                prog.dump()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BlockId, Reg};

    fn program(blocks: usize) -> (Program, Vec<BlockId>) {
        let mut prog = Program::new();
        let blocks: Vec<BlockId> = (0..blocks).map(|_| prog.new_block()).collect();
        prog.block_order = blocks.clone();
        prog.post_order = blocks.iter().rev().copied().collect();
        (prog, blocks)
    }

    fn register(prog: &mut Program, b: BlockId) -> Value {
        Value::Inst(prog.append(b, Opcode::GetRegister, &[Value::Reg(Reg::r(1))], 0))
    }

    #[test]
    fn well_formed_program_passes() {
        let (mut prog, b) = program(2);
        let x = register(&mut prog, b[0]);
        let phi = prog.append(b[1], Opcode::Phi, &[], Type::U32.bits());
        let sum = prog.append(b[1], Opcode::IAdd32, &[Value::Inst(phi), x], 0);
        prog.add_phi_operand(phi, b[0], x);
        prog.add_phi_operand(phi, b[1], Value::Inst(sum));
        prog.append(b[1], Opcode::Reference, &[Value::Inst(sum)], 0);
        assert_eq!(verify(&prog), Ok(()));
    }

    #[test]
    fn mismatched_argument_type_fails() {
        let (mut prog, b) = program(1);
        prog.append(b[0], Opcode::IAdd32, &[Value::F64(1.0), Value::U32(2)], 0);
        let err = verify(&prog).unwrap_err();
        assert!(err.contains("argument 0 of IAdd32 is F64"), "{err}");
    }

    #[test]
    fn missing_argument_fails() {
        let (mut prog, b) = program(1);
        prog.append(b[0], Opcode::IAdd32, &[Value::U32(2)], 0);
        assert!(verify(&prog).unwrap_err().contains("argument 1 of IAdd32"));
    }

    #[test]
    fn use_held_by_an_unlinked_instruction_fails() {
        let (mut prog, b) = program(1);
        let x = register(&mut prog, b[0]);
        prog.append(b[0], Opcode::Reference, &[x], 0);
        let dropped = prog.append(b[0], Opcode::Reference, &[x], 0);
        prog.unlink(dropped);
        assert!(
            verify(&prog)
                .unwrap_err()
                .contains("use count 2 but 1 uses")
        );
    }

    #[test]
    fn read_before_definition_fails() {
        let (mut prog, b) = program(2);
        let late = prog.append(b[1], Opcode::GetRegister, &[Value::Reg(Reg::r(1))], 0);
        prog.append(b[0], Opcode::Reference, &[Value::Inst(late)], 0);
        assert!(verify(&prog).unwrap_err().contains("later definition"));
    }

    #[test]
    fn phi_after_other_instructions_fails() {
        let (mut prog, b) = program(1);
        prog.append(b[0], Opcode::Phi, &[], Type::U32.bits());
        register(&mut prog, b[0]);
        prog.append(b[0], Opcode::Phi, &[], Type::U32.bits());
        assert!(verify(&prog).unwrap_err().contains("interleaved phi"));
    }
}
