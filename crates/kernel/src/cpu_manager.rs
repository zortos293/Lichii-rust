//! One host thread per emulated core: picks threads, runs them on a `CpuCore` and handles what
//! the core halts for.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use lichii_cpu::{CpuCore, HaltReason, ThreadContext};

use crate::KernelInner;
use crate::process::ProcessId;
use crate::scheduler::Preempt;
use crate::svc::{self, Flow, SvcContext};
use crate::thread::Thread;

struct Core {
    kernel: Arc<KernelInner>,
    index: usize,
    cpus: Vec<(ProcessId, Box<dyn CpuCore>)>,
    current_cpu: Option<ProcessId>,
    seen_exits: u64,
    ctx: ThreadContext,
}

pub(crate) fn run_core(kernel: Arc<KernelInner>, index: usize) {
    let mut core = Core {
        kernel,
        index,
        cpus: Vec::new(),
        current_cpu: None,
        seen_exits: 0,
        ctx: ThreadContext::default(),
    };
    while let Some(picked) = core.kernel.scheduler.next(index) {
        core.retire_exited_processes();
        let thread = picked.thread;
        if picked.terminate {
            core.kernel.exit_thread(index, &thread);
            continue;
        }
        core.ctx.clone_from(&thread.context.lock());
        svc::apply_resume(&mut core.ctx, picked.resume, picked.wake);
        core.run_thread(&thread);
    }
}

impl Core {
    /// Drops the CPU state (code cache, address space) kept for processes that exited.
    fn retire_exited_processes(&mut self) {
        let exits = self.kernel.process_exits.load(Ordering::Acquire);
        if exits == self.seen_exits {
            return;
        }
        self.seen_exits = exits;
        let live = self.kernel.processes.lock();
        self.cpus
            .retain(|(pid, _)| live.iter().any(|process| process.id() == *pid));
        if self
            .current_cpu
            .is_some_and(|current| !self.cpus.iter().any(|(pid, _)| *pid == current))
        {
            self.current_cpu = None;
        }
    }

    fn cpu_slot(&mut self, thread: &Thread) -> usize {
        let pid = thread.process().id();
        let slot = match self.cpus.iter().position(|(id, _)| *id == pid) {
            Some(slot) => slot,
            None => {
                let cpu = self.kernel.new_cpu(thread.process(), self.index);
                self.cpus.push((pid, cpu));
                self.cpus.len() - 1
            }
        };
        if self.current_cpu != Some(pid) {
            self.kernel
                .scheduler
                .set_interrupt(self.index, self.cpus[slot].1.interrupt_handle());
            self.current_cpu = Some(pid);
        }
        slot
    }

    fn apply_invalidations(&mut self) {
        for (pid, addr, size) in self.kernel.take_invalidations(self.index) {
            if let Some((_, cpu)) = self.cpus.iter_mut().find(|(id, _)| *id == pid) {
                cpu.invalidate_code(addr, size);
            }
        }
    }

    fn run_thread(&mut self, thread: &Arc<Thread>) {
        let slot = self.cpu_slot(thread);
        self.apply_invalidations();
        self.cpus[slot].1.clear_exclusive();
        loop {
            let halt = if self.kernel.scheduler.enter_guest(self.index) {
                let halt = self.cpus[slot].1.run(&mut self.ctx);
                self.kernel.scheduler.leave_guest(self.index);
                halt
            } else {
                HaltReason::Interrupted
            };
            let mut flow = match halt {
                HaltReason::SupervisorCall(imm) => {
                    let mut svc = SvcContext {
                        kernel: &self.kernel,
                        core: self.index,
                        thread,
                        ctx: &mut self.ctx,
                    };
                    svc::dispatch(&mut svc, imm)
                }
                HaltReason::InvalidateCode { addr, size } => {
                    self.kernel
                        .invalidate_code(thread.process().id(), addr, size);
                    Flow::Continue
                }
                HaltReason::Interrupted | HaltReason::StepComplete => {
                    self.kernel.live_pcs.record(self.index, thread, &self.ctx);
                    if self.kernel.scheduler.take_reschedule(self.index) {
                        thread.pin_if_preemption_disabled();
                    }
                    Flow::Yield
                }
                HaltReason::Breakpoint(imm) => {
                    tracing::error!(
                        process = thread.process().name(),
                        pc = format_args!("{:#x}", self.ctx.pc),
                        "guest BRK #{imm:#x}"
                    );
                    self.kernel.terminate_process(thread.process());
                    Flow::ExitThread
                }
                HaltReason::Fault(fault) => {
                    tracing::error!(
                        process = thread.process().name(),
                        thread = thread.id(),
                        lr = format_args!("{:#x}", self.ctx.x[30]),
                        "unhandled guest fault: {fault:?}"
                    );
                    self.kernel.terminate_process(thread.process());
                    Flow::ExitThread
                }
            };
            if flow == Flow::Continue && self.kernel.scheduler.take_reschedule(self.index) {
                flow = self.preempt(thread, false);
            } else if flow == Flow::Yield {
                flow = self.preempt(thread, matches!(halt, HaltReason::SupervisorCall(_)));
            }
            match flow {
                Flow::Continue | Flow::Yield => {
                    self.apply_invalidations();
                    continue;
                }
                Flow::Switch => return,
                Flow::ExitThread => {
                    self.kernel.exit_thread(self.index, thread);
                    return;
                }
            }
        }
    }

    /// Re-evaluates the running thread; `Continue` when it keeps the core.
    fn preempt(&mut self, thread: &Arc<Thread>, yielding: bool) -> Flow {
        let ctx = &self.ctx;
        let decision = self
            .kernel
            .scheduler
            .preempt(self.index, thread, yielding, || {
                *thread.context.lock() = ctx.clone();
            });
        match decision {
            Preempt::Stay => Flow::Continue,
            Preempt::Switch | Preempt::Pause => {
                if let Some((_, cpu)) = self
                    .cpus
                    .iter_mut()
                    .find(|(id, _)| *id == thread.process().id())
                {
                    cpu.clear_exclusive();
                }
                Flow::Switch
            }
            Preempt::Terminate => Flow::ExitThread,
        }
    }
}
