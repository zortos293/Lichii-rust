//! Reference A64 interpreter. It is the correctness oracle for the JIT and runs any block the JIT
//! cannot translate.
//!
//! Guest code is decoded once into blocks of [`Inst`] (straight-line runs that end at a branch,
//! an exception, or a page boundary) and cached by guest address, so hot code never goes back
//! through the decoder.

mod cache;
mod exec;
mod simd;
#[cfg(test)]
mod tests;

use lichii_common::VAddr;

use self::cache::BlockCache;
use self::exec::{Exec, Flow};
pub(crate) use self::exec::{atomic_cas, atomic_rmw, cas128, condition_holds};
use crate::decoder::{Inst, decode};
use crate::{CpuConfig, CpuCore, CpuFault, HaltReason, InterruptHandle, ThreadContext};

pub struct Interpreter {
    config: CpuConfig,
    interrupt: InterruptHandle,
    cache: BlockCache,
}

/// How a run of instructions ended.
pub(crate) enum Exit {
    /// `ctx.pc` holds the next instruction.
    Continue,
    Halt(HaltReason),
    Invalidate(VAddr, u64),
    InvalidateAll,
}

impl Interpreter {
    pub fn new(config: CpuConfig) -> Self {
        Self {
            config,
            interrupt: InterruptHandle::default(),
            cache: BlockCache::new(),
        }
    }

    /// Executes `insts`, which start at `ctx.pc`, until one leaves the sequence.
    #[inline(always)]
    pub(crate) fn execute(config: &CpuConfig, ctx: &mut ThreadContext, insts: &[Inst]) -> Exit {
        let mut exec = Exec {
            pc: ctx.pc,
            ctx,
            cfg: config,
        };
        for inst in insts {
            match exec.execute(inst) {
                Ok(Flow::Next) => exec.pc += 4,
                Ok(Flow::Jump(target)) => {
                    exec.ctx.pc = target;
                    return Exit::Continue;
                }
                Ok(Flow::Invalidate(addr, size)) => {
                    exec.ctx.pc = exec.pc + 4;
                    return Exit::Invalidate(addr, size);
                }
                Ok(Flow::InvalidateAll) => {
                    exec.ctx.pc = exec.pc + 4;
                    return Exit::InvalidateAll;
                }
                Err(halt) => {
                    let resume = matches!(halt, HaltReason::SupervisorCall(_));
                    exec.ctx.pc = exec.pc + if resume { 4 } else { 0 };
                    return Exit::Halt(halt);
                }
            }
        }
        exec.ctx.pc = exec.pc;
        Exit::Continue
    }

    fn apply(&mut self, exit: Exit) -> Option<HaltReason> {
        match exit {
            Exit::Continue => None,
            Exit::Halt(halt) => Some(halt),
            Exit::Invalidate(addr, size) => {
                self.cache.invalidate(addr, size);
                Some(HaltReason::InvalidateCode { addr, size })
            }
            Exit::InvalidateAll => {
                self.cache.clear();
                Some(HaltReason::InvalidateCode {
                    addr: 0,
                    size: u64::MAX,
                })
            }
        }
    }
}

impl CpuCore for Interpreter {
    fn run(&mut self, ctx: &mut ThreadContext) -> HaltReason {
        loop {
            if self.interrupt.is_signaled() && self.interrupt.take() {
                return HaltReason::Interrupted;
            }
            let block = match self.cache.get_or_translate(ctx.pc, &self.config.memory) {
                Ok(block) => block,
                Err(fault) => return HaltReason::Fault(fault),
            };
            let exit = Self::execute(&self.config, ctx, block);
            if let Some(halt) = self.apply(exit) {
                return halt;
            }
        }
    }

    fn step(&mut self, ctx: &mut ThreadContext) -> HaltReason {
        let pc = ctx.pc;
        let word = match self.config.memory.read::<u32>(pc) {
            Ok(word) if pc & 3 == 0 => word,
            _ => return HaltReason::Fault(CpuFault::Prefetch { pc }),
        };
        let exit = Self::execute(&self.config, ctx, &[decode(word)]);
        self.apply(exit).unwrap_or(HaltReason::StepComplete)
    }

    fn interrupt_handle(&self) -> InterruptHandle {
        self.interrupt.clone()
    }

    fn invalidate_code(&mut self, addr: VAddr, size: u64) {
        self.cache.invalidate(addr, size);
    }

    fn clear_code_cache(&mut self) {
        self.cache.clear();
    }

    fn clear_exclusive(&mut self) {
        self.config.monitor.clear(self.config.core_index);
    }
}
