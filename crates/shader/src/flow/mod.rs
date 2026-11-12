//! Architecture-neutral control flow analysis.
//!
//! Each instruction set implements [`FlowIsa`], which tells the CFG builder how to walk program
//! memory and what every instruction does to control flow. The CFG, indirect branch resolution
//! and structurisation are then shared by Maxwell and Ampere.

pub mod cfg;
#[cfg(test)]
mod tests;

pub use cfg::{Block, BlockId, Cfg, EndClass, FunctionId};

use crate::ShaderEnvironment;
use crate::TranslateError;
use crate::ir::{Condition, Reg};

/// Maxwell's divergence stack tokens. Ampere programs never push tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Token {
    Ssy,
    Pbk,
    Pexit,
    Pret,
    Pcnt,
    Plongjmp,
}

/// What one instruction does to control flow. Offsets are bytes from the program start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowOp {
    /// Falls through to the next instruction.
    Normal,
    /// Direct branch (BRA, JMP, Ampere BRA, CALL.NOINC).
    Branch { target: u32 },
    /// Branch through a register (BRX, JMX). The target table is found with
    /// [`FlowIsa::track_indirect_branch`].
    IndirectBranch { absolute: bool },
    /// Ends the invocation.
    Exit,
    /// Discards the fragment (Maxwell KIL, Ampere KILL).
    Kill,
    /// Calls the function at `target` and resumes at the next instruction.
    Call { target: u32 },
    /// Returns from the current function.
    Return,
    /// Pushes a divergence token whose resume point is `target`.
    Push { token: Token, target: u32 },
    /// Pops the newest `token` and jumps to its resume point (SYNC, BRK, CONT, LONGJMP).
    Pop { token: Token },
}

/// A classified instruction: its control flow effect and its guard condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlowInst {
    pub op: FlowOp,
    /// Guard predicate, plus the flow test for Maxwell branches that have one.
    pub cond: Condition,
}

/// A jump table read by an indirect branch: `num_entries` words at `cbuf_offset` of constant
/// buffer `cbuf_index`. Each target offset is the entry plus `branch_offset`, plus the branch's
/// own offset when the branch is relative; `branch_offset` already includes the instruction set's
/// next-instruction bias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectBranchTable {
    pub cbuf_index: u32,
    pub cbuf_offset: u32,
    pub num_entries: u32,
    pub branch_offset: i32,
    pub reg: Reg,
}

/// The instruction-set specific half of control flow analysis.
pub trait FlowIsa {
    /// Offset of the first instruction of a program whose entry is at `offset`.
    fn align(&self, offset: u32) -> u32;
    /// Offset of the instruction after the one at `pc`.
    fn next(&self, pc: u32) -> u32;
    /// Offset of the instruction before the one at `pc`.
    fn prev(&self, pc: u32) -> u32;
    /// Classifies the instruction at `pc`.
    fn classify(&self, env: &dyn ShaderEnvironment, pc: u32) -> Result<FlowInst, TranslateError>;
    /// Finds the jump table of the indirect branch at `pc`, scanning no further back than
    /// `block_begin`.
    fn track_indirect_branch(
        &self,
        env: &dyn ShaderEnvironment,
        pc: u32,
        block_begin: u32,
    ) -> Option<IndirectBranchTable>;
}
