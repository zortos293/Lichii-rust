//! Maxwell control flow classification, ported from `control_flow.cpp` and
//! `indirect_branch_table_track.cpp`.

use lichii_common::GpuVAddr;

use super::instruction::{Instruction, is_scheduling_word};
use super::opcode::Opcode;
use crate::flow::{FlowInst, FlowIsa, FlowOp, IndirectBranchTable, Token};
use crate::ir::Condition;
use crate::{ShaderEnvironment, TranslateError};

/// Control flow rules of a Maxwell program whose first instruction is at `base`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxwellFlow {
    pub base: GpuVAddr,
}

/// LDC `.B32` operand size.
const LDC_SIZE_B32: u64 = 4;
/// LDC addressing mode without `.IL`, `.IS` or `.ISL`.
const LDC_MODE_DEFAULT: u64 = 0;

impl MaxwellFlow {
    fn read(&self, env: &dyn ShaderEnvironment, pc: u32) -> Instruction {
        Instruction(env.read_u64(self.address(pc)))
    }

    fn address(&self, pc: u32) -> GpuVAddr {
        self.base.wrapping_add(u64::from(pc))
    }

    fn is_scheduling_word(&self, offset: u32) -> bool {
        is_scheduling_word(self.base, offset)
    }

    /// Target of a relative branch at `pc`, which counts from the next word.
    fn relative_target(&self, pc: u32, insn: Instruction) -> Result<u32, TranslateError> {
        let target = pc.wrapping_add(insn.branch_offset() as u32).wrapping_add(8);
        self.checked_target(target, insn, pc)
    }

    fn checked_target(
        &self,
        target: u32,
        insn: Instruction,
        pc: u32,
    ) -> Result<u32, TranslateError> {
        if !target.is_multiple_of(8) {
            return Err(TranslateError::Invalid {
                address: self.address(pc),
                raw: u128::from(insn.0),
            });
        }
        Ok(self.align(target))
    }

    /// Scans back from `*cursor` to `block_begin` for the newest instruction `matches`
    /// accepts, leaving `*cursor` on the instruction before it.
    fn track_back(
        &self,
        env: &dyn ShaderEnvironment,
        block_begin: u32,
        cursor: &mut Option<u32>,
        matches: impl Fn(Instruction) -> bool,
    ) -> Option<Instruction> {
        while let Some(pos) = *cursor {
            let insn = self.read(env, pos);
            *cursor = (pos > block_begin)
                .then(|| self.prev(pos))
                .filter(|&prev| prev >= block_begin);
            if matches(insn) {
                return Some(insn);
            }
        }
        None
    }
}

impl FlowIsa for MaxwellFlow {
    fn align(&self, offset: u32) -> u32 {
        if self.is_scheduling_word(offset) {
            offset + 8
        } else {
            offset
        }
    }

    fn next(&self, pc: u32) -> u32 {
        self.align(pc + 8)
    }

    /// Wraps below the program start; callers stop at their block's first instruction.
    fn prev(&self, pc: u32) -> u32 {
        let prev = pc.wrapping_sub(8);
        if self.is_scheduling_word(prev) {
            prev.wrapping_sub(8)
        } else {
            prev
        }
    }

    /// Classifies the instruction at `pc`. A flow instruction whose condition can never hold
    /// (guard `!PT` or flow test `F`) is `FlowOp::Normal` with that never-true condition.
    fn classify(&self, env: &dyn ShaderEnvironment, pc: u32) -> Result<FlowInst, TranslateError> {
        let insn = self.read(env, pc);
        let raw = u128::from(insn.0);
        let Some(opcode) = insn.opcode() else {
            return Err(TranslateError::Invalid {
                address: self.address(pc),
                raw,
            });
        };
        let unsupported = TranslateError::Unsupported {
            address: self.address(pc),
            opcode: raw,
        };
        let (pred, negated) = insn.pred();
        let tested = Condition::with_flow_test(insn.flow_test(), pred, negated);
        let (op, cond) = match opcode {
            Opcode::Bra
            | Opcode::Jmp
            | Opcode::Ret
            | Opcode::Sync
            | Opcode::Brk
            | Opcode::Cont
            | Opcode::Longjmp => {
                if insn.branch_is_cbuf() {
                    return Err(unsupported);
                }
                let op = match opcode {
                    Opcode::Bra => FlowOp::Branch {
                        target: self.relative_target(pc, insn)?,
                    },
                    Opcode::Jmp => FlowOp::Branch {
                        target: self.checked_target(insn.branch_absolute(), insn, pc)?,
                    },
                    Opcode::Ret => FlowOp::Return,
                    Opcode::Sync => FlowOp::Pop { token: Token::Ssy },
                    Opcode::Brk => FlowOp::Pop { token: Token::Pbk },
                    Opcode::Cont => FlowOp::Pop { token: Token::Pcnt },
                    _ => FlowOp::Pop {
                        token: Token::Plongjmp,
                    },
                };
                (op, tested)
            }
            Opcode::Ssy | Opcode::Pbk | Opcode::Pcnt | Opcode::Pexit | Opcode::Plongjmp => {
                let token = match opcode {
                    Opcode::Ssy => Token::Ssy,
                    Opcode::Pbk => Token::Pbk,
                    Opcode::Pcnt => Token::Pcnt,
                    Opcode::Pexit => Token::Pexit,
                    _ => Token::Plongjmp,
                };
                let target = self.relative_target(pc, insn)?;
                (FlowOp::Push { token, target }, Condition::TRUE)
            }
            Opcode::Cal => {
                let target = self.relative_target(pc, insn)?;
                (FlowOp::Call { target }, Condition::TRUE)
            }
            Opcode::Jcal => {
                let target = self.checked_target(insn.branch_absolute(), insn, pc)?;
                (FlowOp::Call { target }, Condition::TRUE)
            }
            Opcode::Brx => (FlowOp::IndirectBranch { absolute: false }, tested),
            Opcode::Jmx => (FlowOp::IndirectBranch { absolute: true }, tested),
            Opcode::Exit => (FlowOp::Exit, tested),
            Opcode::Kil => (FlowOp::Kill, tested),
            Opcode::Pret => return Err(unsupported),
            _ => (FlowOp::Normal, Condition::pred(pred, negated)),
        };
        let op = if cond.is_false() { FlowOp::Normal } else { op };
        Ok(FlowInst { op, cond })
    }

    /// Matches the `IMNMX imm -> SHL imm -> LDC.B32 -> BRX` bounds-checked jump table idiom.
    fn track_indirect_branch(
        &self,
        env: &dyn ShaderEnvironment,
        pc: u32,
        block_begin: u32,
    ) -> Option<IndirectBranchTable> {
        let brx = self.read(env, pc);
        if !matches!(brx.opcode(), Some(Opcode::Brx | Opcode::Jmx)) {
            return None;
        }
        let reg = brx.src_a_reg();
        let mut cursor = (pc >= block_begin).then_some(pc);
        let ldc = self.track_back(env, block_begin, &mut cursor, |insn| {
            insn.opcode() == Some(Opcode::Ldc)
                && insn.dest_reg() == reg
                && insn.bits(48, 3) == LDC_SIZE_B32
                && insn.bits(44, 2) == LDC_MODE_DEFAULT
        })?;
        let shl = self.track_back(env, block_begin, &mut cursor, |insn| {
            insn.opcode() == Some(Opcode::ShlImm) && insn.dest_reg() == ldc.src_a_reg()
        })?;
        let imnmx = self.track_back(env, block_begin, &mut cursor, |insn| {
            insn.opcode() == Some(Opcode::ImnmxImm) && insn.dest_reg() == shl.src_a_reg()
        })?;
        if imnmx.bit(56) {
            return None;
        }
        Some(IndirectBranchTable {
            cbuf_index: ldc.bits(36, 5) as u32,
            cbuf_offset: ldc.signed_bits(20, 16) as u32,
            num_entries: imnmx.bits(20, 19) as u32 + 1,
            branch_offset: brx.branch_offset() + 8,
            reg,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{FlowTest, Pred, Reg};
    use crate::{ShaderArch, ShaderStage};

    const BASE: GpuVAddr = 0x1000_0000;
    const PT: u64 = 7 << 16;
    const NEVER: u64 = 0xf << 16;
    const FLOW_T: u64 = 0xf;
    const SCHED: u64 = 0x001f_fc00_fd40_07ef;

    struct MockEnv {
        base: GpuVAddr,
        words: Vec<u64>,
    }

    impl MockEnv {
        /// Lays `insns` out with a scheduling word before every three instructions, as seen
        /// from an aligned `base`.
        fn new(insns: &[u64]) -> MockEnv {
            let mut words = Vec::new();
            for &insn in insns {
                if words.len() % 4 == 0 {
                    words.push(SCHED);
                }
                words.push(insn);
            }
            MockEnv { base: BASE, words }
        }

        fn single(insn: u64) -> MockEnv {
            MockEnv::new(&[insn])
        }
    }

    impl ShaderEnvironment for MockEnv {
        fn arch(&self) -> ShaderArch {
            ShaderArch::Maxwell
        }

        fn stage(&self) -> ShaderStage {
            ShaderStage::Fragment
        }

        fn start_address(&self) -> GpuVAddr {
            self.base
        }

        fn read_u64(&self, address: GpuVAddr) -> u64 {
            let index = ((address - self.base) / 8) as usize;
            self.words.get(index).copied().unwrap_or(0)
        }

        fn read_cbuf(&self, _index: u32, _offset: u32) -> u32 {
            0
        }

        fn shared_memory_size(&self) -> u32 {
            0
        }

        fn workgroup_size(&self) -> [u32; 3] {
            [1, 1, 1]
        }
    }

    fn word(opcode: Opcode, fields: u64) -> u64 {
        (u64::from(opcode.pattern().value) << 48) | fields
    }

    fn offset_field(offset: i32) -> u64 {
        (u64::from(offset as u32) & 0xff_ffff) << 20
    }

    fn flow() -> MaxwellFlow {
        MaxwellFlow { base: BASE }
    }

    fn classify(insn: u64) -> Result<FlowInst, TranslateError> {
        flow().classify(&MockEnv::single(insn), 8)
    }

    fn classify_ok(insn: u64) -> FlowInst {
        classify(insn).expect("classifies")
    }

    #[test]
    fn walks_around_scheduling_words() {
        let aligned = flow();
        assert_eq!(aligned.align(0), 8);
        assert_eq!(aligned.align(16), 16);
        assert_eq!(aligned.next(8), 16);
        assert_eq!(aligned.next(24), 40);
        assert_eq!(aligned.prev(40), 24);
        assert_eq!(aligned.prev(16), 8);

        let after_header = MaxwellFlow { base: BASE + 0x50 };
        assert_eq!(after_header.align(0), 0);
        assert_eq!(after_header.next(0), 8);
        assert_eq!(after_header.next(8), 24);
        assert_eq!(after_header.prev(24), 8);
        assert_eq!(after_header.align(16), 24);
    }

    #[test]
    fn bra() {
        let inst = classify_ok(word(Opcode::Bra, PT | offset_field(0x20) | FLOW_T));
        assert_eq!(inst.op, FlowOp::Branch { target: 0x30 });
        assert_eq!(inst.cond, Condition::TRUE);

        let back = classify_ok(word(Opcode::Bra, PT | offset_field(-0x10) | FLOW_T));
        assert_eq!(back.op, FlowOp::Branch { target: 8 });

        let onto_sched = classify_ok(word(Opcode::Bra, PT | offset_field(0x10) | FLOW_T));
        assert_eq!(onto_sched.op, FlowOp::Branch { target: 40 });
    }

    #[test]
    fn bra_conditions() {
        let guarded = classify_ok(word(
            Opcode::Bra,
            (0x8 | 2) << 16 | offset_field(8) | FLOW_T,
        ));
        assert_eq!(guarded.op, FlowOp::Branch { target: 24 });
        assert_eq!(guarded.cond, Condition::pred(Pred::p(2), true));

        let tested = classify_ok(word(Opcode::Bra, PT | offset_field(8) | 0x2));
        assert_eq!(
            tested.cond,
            Condition::with_flow_test(FlowTest::EQ, Pred::PT, false)
        );

        let never = classify_ok(word(Opcode::Bra, NEVER | offset_field(8) | FLOW_T));
        assert_eq!(never.op, FlowOp::Normal);
        assert!(never.cond.is_false());
    }

    #[test]
    fn unsupported_and_invalid_branches() {
        let cbuf = classify(word(Opcode::Bra, PT | (1 << 5) | FLOW_T));
        assert!(matches!(cbuf, Err(TranslateError::Unsupported { .. })));

        let misaligned = classify(word(Opcode::Bra, PT | offset_field(4) | FLOW_T));
        assert!(matches!(misaligned, Err(TranslateError::Invalid { .. })));

        let pret = classify(word(Opcode::Pret, PT | offset_field(8)));
        assert!(matches!(pret, Err(TranslateError::Unsupported { .. })));

        let undecodable = classify(u64::MAX);
        assert!(matches!(
            undecodable,
            Err(TranslateError::Invalid { address, raw }) if address == BASE + 8 && raw == u128::from(u64::MAX)
        ));
    }

    #[test]
    fn jmp_is_absolute() {
        let inst = classify_ok(word(Opcode::Jmp, PT | (0x100 << 20) | FLOW_T));
        assert_eq!(inst.op, FlowOp::Branch { target: 0x108 });
    }

    #[test]
    fn divergence_tokens() {
        let ssy = classify_ok(word(Opcode::Ssy, (0x3 << 16) | offset_field(0x40)));
        assert_eq!(
            ssy.op,
            FlowOp::Push {
                token: Token::Ssy,
                target: 0x50
            }
        );
        assert_eq!(ssy.cond, Condition::TRUE);

        let pbk = classify_ok(word(Opcode::Pbk, PT | offset_field(0x40)));
        assert_eq!(
            pbk.op,
            FlowOp::Push {
                token: Token::Pbk,
                target: 0x50
            }
        );

        let sync = classify_ok(word(Opcode::Sync, PT | FLOW_T));
        assert_eq!(sync.op, FlowOp::Pop { token: Token::Ssy });
        assert_eq!(sync.cond, Condition::TRUE);

        let brk = classify_ok(word(Opcode::Brk, (1 << 16) | FLOW_T));
        assert_eq!(brk.op, FlowOp::Pop { token: Token::Pbk });
        assert_eq!(brk.cond, Condition::pred(Pred::p(1), false));

        let cont = classify_ok(word(Opcode::Cont, PT | FLOW_T));
        assert_eq!(cont.op, FlowOp::Pop { token: Token::Pcnt });
    }

    #[test]
    fn exit_and_kill() {
        let exit = classify_ok(word(Opcode::Exit, PT | FLOW_T));
        assert_eq!(exit.op, FlowOp::Exit);
        assert_eq!(exit.cond, Condition::TRUE);

        let guarded = classify_ok(word(Opcode::Exit, (1 << 16) | FLOW_T));
        assert_eq!(guarded.op, FlowOp::Exit);
        assert_eq!(guarded.cond, Condition::pred(Pred::p(1), false));

        let never = classify_ok(word(Opcode::Exit, PT));
        assert_eq!(never.op, FlowOp::Normal);
        assert_eq!(never.cond.flow_test, FlowTest::F);

        let kil = classify_ok(word(Opcode::Kil, (0x8 | 4) << 16 | 0x5));
        assert_eq!(kil.op, FlowOp::Kill);
        assert_eq!(
            kil.cond,
            Condition::with_flow_test(FlowTest::NE, Pred::p(4), true)
        );
    }

    #[test]
    fn calls_and_returns() {
        let cal = classify_ok(word(Opcode::Cal, (2 << 16) | offset_field(0x100)));
        assert_eq!(cal.op, FlowOp::Call { target: 0x110 });
        assert_eq!(cal.cond, Condition::TRUE);

        let jcal = classify_ok(word(Opcode::Jcal, PT | (0x200 << 20)));
        assert_eq!(jcal.op, FlowOp::Call { target: 0x208 });

        let ret = classify_ok(word(Opcode::Ret, PT | FLOW_T));
        assert_eq!(ret.op, FlowOp::Return);
    }

    #[test]
    fn indirect_branches() {
        let brx = classify_ok(word(Opcode::Brx, PT | (3 << 8) | FLOW_T));
        assert_eq!(brx.op, FlowOp::IndirectBranch { absolute: false });
        assert_eq!(brx.cond, Condition::TRUE);

        let jmx = classify_ok(word(Opcode::Jmx, PT | (3 << 8) | FLOW_T));
        assert_eq!(jmx.op, FlowOp::IndirectBranch { absolute: true });
    }

    #[test]
    fn guarded_normal_instructions() {
        let fadd = classify_ok(word(Opcode::FaddReg, (0x8 | 2) << 16 | 0x0201));
        assert_eq!(fadd.op, FlowOp::Normal);
        assert_eq!(fadd.cond, Condition::pred(Pred::p(2), true));

        let unguarded = classify_ok(word(Opcode::FaddReg, PT | 0x0201));
        assert_eq!(
            unguarded,
            FlowInst {
                op: FlowOp::Normal,
                cond: Condition::TRUE
            }
        );
    }

    fn imnmx(dest: u64, src: u64, bound: u64) -> u64 {
        word(Opcode::ImnmxImm, PT | (bound << 20) | (src << 8) | dest)
    }

    fn shl(dest: u64, src: u64, shift: u64) -> u64 {
        word(Opcode::ShlImm, PT | (shift << 20) | (src << 8) | dest)
    }

    fn ldc(dest: u64, src: u64, index: u64, offset: u64, size: u64) -> u64 {
        word(
            Opcode::Ldc,
            (size << 48) | (index << 36) | (offset << 20) | PT | (src << 8) | dest,
        )
    }

    fn brx(src: u64, offset: i32) -> u64 {
        word(Opcode::Brx, PT | offset_field(offset) | (src << 8) | FLOW_T)
    }

    fn nop() -> u64 {
        word(Opcode::Nop, PT | 0x0f00)
    }

    fn jump_table_program() -> Vec<u64> {
        vec![
            imnmx(1, 0, 3),
            shl(2, 1, 2),
            nop(),
            ldc(3, 2, 2, 0x40, LDC_SIZE_B32),
            brx(3, 0x10),
        ]
    }

    const BRX_PC: u32 = 48;

    #[test]
    fn tracks_jump_table() {
        let env = MockEnv::new(&jump_table_program());
        let table = flow().track_indirect_branch(&env, BRX_PC, 8);
        assert_eq!(
            table,
            Some(IndirectBranchTable {
                cbuf_index: 2,
                cbuf_offset: 0x40,
                num_entries: 4,
                branch_offset: 0x18,
                reg: Reg::r(3),
            })
        );
    }

    #[test]
    fn jump_table_needs_the_whole_chain() {
        let flow = flow();
        let mut negative = jump_table_program();
        negative[0] |= 1 << 56;
        let env = MockEnv::new(&negative);
        assert_eq!(flow.track_indirect_branch(&env, BRX_PC, 8), None);

        let mut byte_load = jump_table_program();
        byte_load[3] = ldc(3, 2, 2, 0x40, 0);
        let env = MockEnv::new(&byte_load);
        assert_eq!(flow.track_indirect_branch(&env, BRX_PC, 8), None);

        let mut wrong_reg = jump_table_program();
        wrong_reg[1] = shl(5, 1, 2);
        let env = MockEnv::new(&wrong_reg);
        assert_eq!(flow.track_indirect_branch(&env, BRX_PC, 8), None);

        let env = MockEnv::new(&jump_table_program());
        assert_eq!(flow.track_indirect_branch(&env, BRX_PC, 16), None);
        assert_eq!(flow.track_indirect_branch(&env, 40, 8), None);
    }
}
