//! Reference A64 interpreter. It is the correctness oracle for the JIT and runs any block the JIT
//! cannot translate.

mod simd;

use lichii_common::VAddr;

use crate::{CpuConfig, CpuCore, CpuFault, HaltReason, InterruptHandle, ThreadContext};

pub struct Interpreter {
    config: CpuConfig,
    interrupt: InterruptHandle,
}

impl Interpreter {
    pub fn new(config: CpuConfig) -> Self {
        Self {
            config,
            interrupt: InterruptHandle::default(),
        }
    }

    fn execute(&mut self, ctx: &mut ThreadContext) -> Option<HaltReason> {
        let pc = ctx.pc;
        let Ok(instruction) = self.config.memory.read::<u32>(pc) else {
            return Some(HaltReason::Fault(CpuFault::Prefetch { pc }));
        };
        match simd::execute(ctx, instruction) {
            Some(Ok(())) => {
                ctx.pc += 4;
                None
            }
            Some(Err(halt)) => Some(halt),
            None => Some(HaltReason::Fault(CpuFault::Undefined { pc, instruction })),
        }
    }
}

impl CpuCore for Interpreter {
    fn run(&mut self, ctx: &mut ThreadContext) -> HaltReason {
        loop {
            if self.interrupt.take() {
                return HaltReason::Interrupted;
            }
            if let Some(halt) = self.execute(ctx) {
                return halt;
            }
        }
    }

    fn step(&mut self, ctx: &mut ThreadContext) -> HaltReason {
        self.execute(ctx).unwrap_or(HaltReason::StepComplete)
    }

    fn interrupt_handle(&self) -> InterruptHandle {
        self.interrupt.clone()
    }

    fn invalidate_code(&mut self, _addr: VAddr, _size: u64) {}

    fn clear_code_cache(&mut self) {}

    fn clear_exclusive(&mut self) {
        self.config.monitor.clear(self.config.core_index);
    }
}
