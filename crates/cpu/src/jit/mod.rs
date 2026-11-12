//! A64 JIT for x86-64 hosts.
//!
//! Each guest block is decoded, translated to the block-local IR in [`ir`], optimised by
//! [`passes`] and compiled by [`x64`]. Instructions the translator does not handle end the
//! block and run on the interpreter, which stays the reference: any decodable instruction runs,
//! translated or not. See `crates/cpu/JIT.md` for the design.

mod cache;
mod ir;
mod passes;
#[cfg(test)]
mod tests;
mod translate;
mod x64;

use lichii_common::{PAGE_MASK, VAddr};
use lichii_memory::AddressSpace;

use self::cache::{Entry, JitCache};
use self::ir::{Block, Terminal};
pub(crate) use self::passes::Passes;
use self::translate::{Step, Translator};
use self::x64::{EXIT_CONTINUE, EXIT_FAULT, SPILL_SLOTS, Target};
use crate::decoder::decode;
use crate::interpreter::{Exit, Interpreter};
use crate::{CpuConfig, CpuCore, CpuFault, HaltReason, InterruptHandle, ThreadContext};

/// Runtime state translated code reaches through `r13`.
#[repr(C, align(16))]
pub(super) struct JitState {
    /// Values passed between slow paths and their helpers.
    scratch: u128,
    save_xmm: [u128; 16],
    save_gpr: [u64; 16],
    spill: [u128; SPILL_SLOTS],
    memory: *const AddressSpace,
    fault: Option<CpuFault>,
}

// SAFETY: `memory` points into the `Arc<AddressSpace>` the owning `Jit` keeps in its config, and
// `AddressSpace` is `Sync`.
unsafe impl Send for JitState {}

impl JitState {
    /// # Safety
    /// Only called by slow paths of translated code, while the owning `Jit` is running it.
    unsafe fn memory(&self) -> &AddressSpace {
        // SAFETY: see the `Send` impl.
        unsafe { &*self.memory }
    }
}

extern "sysv64" fn read_slow(state: &mut JitState, addr: u64, pc: u64, size: u32) -> bool {
    // SAFETY: called from a slow path.
    let memory = unsafe { state.memory() };
    let value = match size {
        0 => memory.read::<u8>(addr).map(u128::from),
        1 => memory.read::<u16>(addr).map(u128::from),
        2 => memory.read::<u32>(addr).map(u128::from),
        3 => memory.read::<u64>(addr).map(u128::from),
        _ => memory.read::<u128>(addr),
    };
    match value {
        Ok(value) => {
            state.scratch = value;
            false
        }
        Err(f) => {
            state.fault = Some(CpuFault::Data {
                pc,
                addr: f.addr,
                write: false,
            });
            true
        }
    }
}

extern "sysv64" fn write_slow(state: &mut JitState, addr: u64, pc: u64, size: u32) -> bool {
    // SAFETY: called from a slow path.
    let memory = unsafe { state.memory() };
    let value = state.scratch;
    let result = match size {
        0 => memory.write(addr, value as u8),
        1 => memory.write(addr, value as u16),
        2 => memory.write(addr, value as u32),
        3 => memory.write(addr, value as u64),
        _ => memory.write(addr, value),
    };
    match result {
        Ok(()) => false,
        Err(f) => {
            state.fault = Some(CpuFault::Data {
                pc,
                addr: f.addr,
                write: true,
            });
            true
        }
    }
}

/// A [`CpuCore`] that translates guest blocks to x86-64 code and runs every instruction it does
/// not translate on the interpreter. Uses the address space's fastmem arena when it has one.
pub struct Jit {
    config: CpuConfig,
    interrupt: InterruptHandle,
    cache: JitCache,
    state: Box<JitState>,
    target: Target,
    arena: *mut u8,
    passes: Passes,
}

// SAFETY: `arena` is the base of the fastmem arena owned by `config.memory`, which is `Sync`; the
// cache's code memory is only touched by the thread running this core.
unsafe impl Send for Jit {}

impl Jit {
    /// `None` when executable memory cannot be mapped.
    pub fn new(config: CpuConfig) -> Option<Self> {
        Self::with_passes(config, Passes::ALL)
    }

    pub(crate) fn with_passes(config: CpuConfig, passes: Passes) -> Option<Self> {
        let cache = JitCache::new()?;
        let fastmem = config.memory.fastmem().filter(|_| {
            if !cache.handles_faults() {
                tracing::warn!("the JIT fault-site table is full; this core runs without fastmem");
            }
            cache.handles_faults()
        });
        if let Some(fastmem) = fastmem {
            fastmem.set_fault_handler(Some(cache::on_fault), 0);
        }
        let state = Box::new(JitState {
            scratch: 0,
            save_xmm: [0; 16],
            save_gpr: [0; 16],
            spill: [0; SPILL_SLOTS],
            memory: &*config.memory,
            fault: None,
        });
        Some(Self {
            target: Target {
                arena_size: fastmem.map(|f| f.size() as u64),
                read: read_slow as *const () as usize,
                write: write_slow as *const () as usize,
            },
            arena: fastmem.map_or(std::ptr::null_mut(), |f| f.base()),
            config,
            interrupt: InterruptHandle::default(),
            cache,
            state,
            passes,
        })
    }

    fn translate(&mut self, start: u64) -> Result<Entry, CpuFault> {
        if start & 3 != 0 {
            return Err(CpuFault::Prefetch { pc: start });
        }
        let mut t = Translator {
            block: Block {
                ops: Vec::new(),
                terminal: Terminal::Interpret(start),
            },
            config: &self.config,
        };
        let mut pc = start;
        let mut fallback = None;
        loop {
            let word = self
                .config
                .memory
                .read::<u32>(pc)
                .map_err(|_| CpuFault::Prefetch { pc: start })?;
            let inst = decode(word);
            let mark = t.block.ops.len();
            match t.translate(&inst, pc) {
                Step::Next => {}
                Step::End(terminal) => {
                    t.block.terminal = terminal;
                    break;
                }
                Step::Rejected => {
                    t.block.ops.truncate(mark);
                    t.block.terminal = Terminal::Interpret(pc);
                    fallback = Some(inst);
                    break;
                }
            }
            if (pc + 4) & PAGE_MASK == 0 {
                t.block.terminal = Terminal::Jump(pc + 4);
                break;
            }
            pc += 4;
        }
        let mut block = t.block;
        let len = pc + 4 - start;
        if block.terminal == Terminal::Interpret(start) {
            return Ok(self.cache.insert(start, len, None, fallback));
        }
        self.passes.run(&mut block);
        let compiled = x64::compile(&block, &self.target);
        Ok(self.cache.insert(start, len, Some(compiled), fallback))
    }

    fn apply(&mut self, exit: Exit) -> Option<HaltReason> {
        match exit {
            Exit::Continue => None,
            Exit::Halt(halt) => Some(halt),
            Exit::Invalidate(addr, size) => {
                self.cache.invalidate(addr, size);
                None
            }
            Exit::InvalidateAll => {
                self.cache.clear();
                None
            }
        }
    }
}

impl CpuCore for Jit {
    fn run(&mut self, ctx: &mut ThreadContext) -> HaltReason {
        loop {
            if self.interrupt.is_signaled() && self.interrupt.take() {
                return HaltReason::Interrupted;
            }
            let entry = match self.cache.lookup(ctx.pc) {
                Some(entry) => entry,
                None => match self.translate(ctx.pc) {
                    Ok(entry) => entry,
                    Err(fault) => return HaltReason::Fault(fault),
                },
            };
            if !entry.code.is_null() {
                let enter = self.cache.trampoline();
                // SAFETY: `entry.code` is a compiled block in this core's code memory; it only
                // touches `ctx`, the state, and the arena behind the bounds check.
                let exit = unsafe { enter(ctx, &mut *self.state, self.arena, entry.code) };
                match exit {
                    EXIT_CONTINUE => continue,
                    EXIT_FAULT => {
                        let fault = self
                            .state
                            .fault
                            .take()
                            .expect("a slow path recorded the fault");
                        if let CpuFault::Data { pc, .. } = fault {
                            ctx.pc = pc;
                        }
                        return HaltReason::Fault(fault);
                    }
                    _ => {}
                }
            }
            let inst = entry
                .fallback
                .expect("only blocks with a fallback exit to the interpreter");
            let exit = Interpreter::execute(&self.config, ctx, &[inst]);
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
        let exit = Interpreter::execute(&self.config, ctx, &[decode(word)]);
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
