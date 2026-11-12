//! Control flow classification of Ampere programs.
//!
//! Volta and later GPUs reconverge with explicit convergence barriers (BSSY/BSYNC), so no
//! instruction pushes or pops a divergence token: every transfer is an explicit branch, call,
//! return or exit.

use lichii_common::GpuVAddr;

use super::decode::{Decoded, Form, decode};
use super::instruction::{INSTRUCTION_SIZE, Instruction};
use super::opcode::Opcode;
use crate::flow::{FlowInst, FlowIsa, FlowOp, IndirectBranchTable};
use crate::ir::{Condition, Pred, Reg};
use crate::{ShaderEnvironment, TranslateError};

/// BRA convergence mode (bits 32..34) .DIV: taken only when the warp has diverged. Threads run
/// converged in translated code, so it never is.
const BRANCH_DIVERGED: u64 = 2;
/// LDC/ULDC memory type of a 32-bit load (bits 73..76).
const MEM_B32: u64 = 4;
/// ISETP comparisons (bits 76..79) that make a select a clamp to its immediate.
const CMP_LT: u64 = 1;
const CMP_LE: u64 = 3;

/// Ampere control flow. Offsets are bytes from `base`, the program's first instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmpereFlow {
    pub base: GpuVAddr,
}

impl AmpereFlow {
    /// Reads the instruction at offset `pc`.
    pub fn read(&self, env: &dyn ShaderEnvironment, pc: u32) -> Instruction {
        let address = self.base + u64::from(pc);
        Instruction::from_words(env.read_u64(address), env.read_u64(address + 8))
    }

    fn invalid(&self, pc: u32, inst: Instruction) -> TranslateError {
        TranslateError::Invalid {
            address: self.base + u64::from(pc),
            raw: inst.0,
        }
    }

    fn target(&self, pc: u32, inst: Instruction, target: i64) -> Result<u32, TranslateError> {
        u32::try_from(target)
            .ok()
            .filter(|target| target % INSTRUCTION_SIZE == 0)
            .ok_or_else(|| self.invalid(pc, inst))
    }

    fn relative_target(&self, pc: u32, inst: Instruction) -> Result<u32, TranslateError> {
        let next = i64::from(pc) + i64::from(INSTRUCTION_SIZE);
        self.target(pc, inst, next + inst.branch_offset())
    }

    fn absolute_target(&self, pc: u32, inst: Instruction) -> Result<u32, TranslateError> {
        let target = i64::try_from(inst.absolute_target()).unwrap_or(i64::MAX);
        self.target(pc, inst, target)
    }

    /// Scans backwards from `*pos` to `block_begin` for the first instruction `matches`
    /// accepts, leaving `*pos` at it.
    fn find_back(
        &self,
        env: &dyn ShaderEnvironment,
        pos: &mut u32,
        block_begin: u32,
        matches: impl Fn(Instruction, Decoded) -> bool,
    ) -> Option<Instruction> {
        while *pos > block_begin {
            *pos -= INSTRUCTION_SIZE;
            let inst = self.read(env, *pos);
            if let Some(decoded) = decode(inst)
                && matches(inst, decoded)
            {
                return Some(inst);
            }
        }
        None
    }
}

/// The second condition control instructions carry, `None` when it is always true.
fn extra_condition(inst: Instruction, form: Form) -> Option<(Pred, bool)> {
    let (pred, negated) = if form == Form::UniformCondition {
        inst.upred_src(24, 27)
    } else {
        inst.pred_c()
    };
    (negated || !pred.is_true()).then_some((pred, negated))
}

impl FlowIsa for AmpereFlow {
    fn align(&self, offset: u32) -> u32 {
        offset.next_multiple_of(INSTRUCTION_SIZE)
    }

    fn next(&self, pc: u32) -> u32 {
        pc + INSTRUCTION_SIZE
    }

    fn prev(&self, pc: u32) -> u32 {
        pc.saturating_sub(INSTRUCTION_SIZE)
    }

    fn classify(&self, env: &dyn ShaderEnvironment, pc: u32) -> Result<FlowInst, TranslateError> {
        let inst = self.read(env, pc);
        let (pred, negated) = inst.guard();
        let mut cond = Condition::pred(pred, negated);
        if inst.never_executes() {
            return Ok(FlowInst {
                op: FlowOp::Normal,
                cond,
            });
        }
        let decoded = decode(inst).ok_or_else(|| self.invalid(pc, inst))?;
        let op = match decoded.opcode {
            Opcode::Bra if inst.bits(32, 2) == BRANCH_DIVERGED => FlowOp::Normal,
            Opcode::Bra => FlowOp::Branch {
                target: self.relative_target(pc, inst)?,
            },
            Opcode::Jmp => FlowOp::Branch {
                target: self.absolute_target(pc, inst)?,
            },
            Opcode::Brx | Opcode::Brxu => FlowOp::IndirectBranch { absolute: false },
            Opcode::Jmx | Opcode::Jmxu => FlowOp::IndirectBranch { absolute: true },
            Opcode::CallRel if inst.bit(86) && !cond.is_true() => FlowOp::Branch {
                target: self.relative_target(pc, inst)?,
            },
            Opcode::CallRel => FlowOp::Call {
                target: self.relative_target(pc, inst)?,
            },
            Opcode::CallAbs => FlowOp::Call {
                target: self.absolute_target(pc, inst)?,
            },
            Opcode::Ret => FlowOp::Return,
            Opcode::Exit => FlowOp::Exit,
            Opcode::Kill => FlowOp::Kill,
            _ => FlowOp::Normal,
        };
        if op != FlowOp::Normal {
            cond.extra = extra_condition(inst, decoded.form);
        }
        Ok(FlowInst { op, cond })
    }

    /// Recognises the jump table NVN2 emits for a switch:
    ///
    /// ```text
    /// IMNMX.U32 Ri, Rx, max, PT        | UISETP.LT.U32 UPn, PT, URx, max, PT
    ///                                  | USEL URi, URx, max, UPn
    /// IMAD.SHL.U32 Rj, Ri, 0x4, RZ     (or SHF.L.U32 Rj, Ri, 0x2, RZ, or LEA Rj, Ri, RZ, 0x2)
    /// LDC Rt, c[bank][Rj + offset]     | ULDC URt, c[bank][URj + offset]
    /// BRX Rt, rel                      | BRXU URt, rel
    /// ```
    ///
    /// The table holds `max + 1` offsets from the program start; the C++ lowering only accepts
    /// BRX whose `rel` cancels the next-instruction address.
    fn track_indirect_branch(
        &self,
        env: &dyn ShaderEnvironment,
        pc: u32,
        block_begin: u32,
    ) -> Option<IndirectBranchTable> {
        let branch = self.read(env, pc);
        let (relative, uniform) = match decode(branch)?.opcode {
            Opcode::Brx => (true, false),
            Opcode::Brxu => (true, true),
            Opcode::Jmx => (false, false),
            Opcode::Jmxu => (false, true),
            _ => return None,
        };
        let reg = if uniform {
            branch.usrc0()
        } else {
            branch.src0()
        };
        let bias = if relative {
            i64::from(INSTRUCTION_SIZE)
        } else {
            0
        };
        let branch_offset = i32::try_from(branch.branch_offset() + bias).ok()?;

        let mut pos = pc;
        let load = self.find_back(env, &mut pos, block_begin, |inst, d| {
            let indexed = if uniform {
                d.opcode == Opcode::Uldc && d.form == Form::UregIndexed && inst.udst() == reg
            } else {
                d.opcode == Opcode::Ldc
                    && d.form == Form::Bound
                    && inst.dst() == reg
                    && inst.bits(78, 2) == 0
            };
            indexed && inst.bits(73, 3) == MEM_B32
        })?;
        let index = if uniform { load.usrc0() } else { load.src0() };
        if index.is_zero() {
            return None;
        }

        let scale = self.find_back(env, &mut pos, block_begin, |inst, d| {
            destination(inst, uniform) == index && scaled_by_four(inst, d, uniform).is_some()
        })?;
        let unscaled = scaled_by_four(scale, decode(scale)?, uniform)?;

        let max = if uniform {
            let select = self.find_back(env, &mut pos, block_begin, |inst, d| {
                d == alu(Opcode::Usel, Form::ImmReg) && inst.udst() == unscaled
            })?;
            self.uniform_clamp(env, &mut pos, block_begin, select)?
        } else {
            let clamp = self.find_back(env, &mut pos, block_begin, |inst, d| {
                d == alu(Opcode::Imnmx, Form::ImmReg) && inst.dst() == unscaled
            })?;
            vector_clamp(clamp)?
        };

        Some(IndirectBranchTable {
            cbuf_index: load.bits(54, 5) as u32,
            cbuf_offset: load.bits(38, 16) as u32,
            num_entries: max.checked_add(1)?,
            branch_offset,
            reg,
        })
    }
}

impl AmpereFlow {
    /// The bound of `USEL URi, URx, max, UPn` when the UISETP that set `UPn` makes it
    /// `min(URx, max)`.
    fn uniform_clamp(
        &self,
        env: &dyn ShaderEnvironment,
        pos: &mut u32,
        block_begin: u32,
        select: Instruction,
    ) -> Option<u32> {
        let (cond, negated) = select.upred_src(87, 90);
        if negated || cond.is_true() {
            return None;
        }
        let max = select.imm32();
        let compare = self.find_back(env, pos, block_begin, |inst, d| {
            d.opcode == Opcode::Uisetp && inst.upred_at(81) == cond
        })?;
        let accumulate = compare.upred_src(87, 90);
        let clamps = decode(compare)? == alu(Opcode::Uisetp, Form::ImmReg)
            && compare.usrc0() == select.usrc0()
            && compare.imm32() == max
            && matches!(compare.bits(76, 3), CMP_LT | CMP_LE)
            && !compare.bit(73)
            && !compare.bit(72)
            && compare.bits(74, 2) == 0
            && accumulate == (Pred::UPT, false);
        clamps.then_some(max)
    }
}

const fn alu(opcode: Opcode, form: Form) -> Decoded {
    Decoded { opcode, form }
}

fn destination(inst: Instruction, uniform: bool) -> Reg {
    if uniform { inst.udst() } else { inst.dst() }
}

/// The register `scale` multiplies by four, when it is `IMAD.SHL Rd, Ra, 4, RZ`,
/// `SHF.L Rd, Ra, 2, RZ` or `LEA Rd, Ra, RZ, 2` (or their uniform forms).
fn scaled_by_four(scale: Instruction, d: Decoded, uniform: bool) -> Option<Reg> {
    let (src0, src1, src2) = if uniform {
        (scale.usrc0(), scale.usrc1(), scale.usrc2())
    } else {
        (scale.src0(), scale.src1(), scale.src2())
    };
    let scaled = match (d.opcode, d.form) {
        (Opcode::Imad | Opcode::Uimad, Form::ImmReg) => scale.imm32() == 4 && src2.is_zero(),
        (Opcode::Shf | Opcode::Ushf, Form::ImmReg) => {
            scale.imm32() == 2 && src2.is_zero() && !scale.bit(76) && !scale.bit(80)
        }
        (Opcode::Lea | Opcode::Ulea, Form::RegReg) => {
            src1.is_zero() && scale.bits(75, 5) == 2 && !scale.bit(80) && !scale.bit(74)
        }
        _ => false,
    };
    scaled.then_some(src0)
}

/// The bound of `IMNMX Rd, Ra, max, PT` (a minimum against a non-negative immediate).
fn vector_clamp(clamp: Instruction) -> Option<u32> {
    let max = clamp.imm32();
    (clamp.pred_c() == (Pred::PT, false) && max <= i32::MAX as u32).then_some(max)
}
