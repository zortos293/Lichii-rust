//! A64 JIT for x86-64 hosts.
//!
//! Each guest block is decoded, translated to the block-local IR in [`ir`], optimised by
//! [`passes`] and compiled by [`x64`]. Instructions the translator does not handle end the
//! block and run on the interpreter, which stays the reference: any decodable instruction runs,
//! translated or not. See `crates/cpu/JIT.md` for the design.

mod cache;
mod ir;
mod passes;
mod simd;
#[cfg(test)]
mod tests;
mod translate;
mod x64;

use std::sync::atomic::AtomicBool;

use lichii_common::settings::CpuAccuracy;
use lichii_common::{PAGE_MASK, VAddr};
use lichii_memory::{AddressSpace, ExclusiveMonitor, ReservationSlot};

use self::cache::{Code, Entry, JitCache};
use self::ir::{Block, Terminal};
pub(crate) use self::passes::Passes;
use self::translate::{Step, Translator};
use self::x64::{EXIT_FAULT, SPILL_SLOTS, Target};
use crate::decoder::{AtomicOp, decode};
use crate::interpreter::{Exit, Interpreter, atomic_cas, atomic_rmw, cas128};
use crate::{CpuConfig, CpuCore, CpuFault, GuestClock, HaltReason, InterruptHandle, ThreadContext};

/// Runtime state translated code reaches through `r13`.
#[repr(C, align(16))]
pub(super) struct JitState {
    /// Values passed between slow paths and their helpers.
    scratch: u128,
    scratch2: u128,
    save_xmm: [u128; 16],
    save_gpr: [u64; 16],
    /// The high half of the last 128-bit exclusive load or compare-and-swap.
    high: u64,
    /// The registers a `cmpxchg16b` borrows while it runs.
    cx16_save: [u64; 2],
    spill: [u128; SPILL_SLOTS],
    memory: *const AddressSpace,
    clock: *const dyn GuestClock,
    monitor: *const ExclusiveMonitor,
    core_index: usize,
    /// This core's slot in `monitor`.
    reservation: *const ReservationSlot,
    /// The interrupt flag every block checks on entry.
    interrupt: *const AtomicBool,
    /// The indirect-branch table of the cache.
    table: *const (u64, usize),
    /// The `jmp` the last block left through, for the dispatcher to link; 0 when none.
    link: usize,
    fault: Option<CpuFault>,
}

// SAFETY: `memory`, `clock` and `interrupt` point into `Arc`s the owning `Jit` keeps, and all
// three are `Sync`; `table` is owned by its cache.
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

/// `AtomicOp` by discriminant, as `x64` encodes it in a slow path's `info`.
const ATOMIC_OPS: [AtomicOp; 9] = [
    AtomicOp::Add,
    AtomicOp::Clr,
    AtomicOp::Eor,
    AtomicOp::Set,
    AtomicOp::Smax,
    AtomicOp::Smin,
    AtomicOp::Umax,
    AtomicOp::Umin,
    AtomicOp::Swp,
];

/// `LD<op>`, `SWP` and `CAS` as the interpreter runs them: `info` holds the size in bits 7..0
/// and the operation above it; the operands are in `scratch` and `scratch2`.
extern "sysv64" fn atomic_slow(state: &mut JitState, addr: u64, pc: u64, info: u32) -> bool {
    let size = info as u8;
    if addr & ((1 << size) - 1) != 0 {
        state.fault = Some(CpuFault::Alignment { pc, addr });
        return true;
    }
    // SAFETY: called from a slow path.
    let Some((ptr, _)) = unsafe { state.memory() }.translate(addr) else {
        state.fault = Some(CpuFault::Data {
            pc,
            addr,
            write: true,
        });
        return true;
    };
    if size == 4 {
        // SAFETY: the pair is 16-byte aligned, so it lies in the one page `translate` returned.
        let old = unsafe { cas128(ptr, state.scratch, state.scratch2) };
        state.scratch = old;
        state.high = (old >> 64) as u64;
        return false;
    }
    let (operand, second) = (state.scratch as u64, state.scratch2 as u64);
    let op = (info >> 8) as i32;
    // SAFETY: the access is naturally aligned, so it stays inside the one mapped guest page that
    // `translate` returned.
    let old = unsafe {
        if op == x64::ATOMIC_CAS {
            atomic_cas(ptr, size, operand, second)
        } else {
            atomic_rmw(ptr, size, ATOMIC_OPS[op as usize], operand)
        }
    };
    state.scratch = old as u128;
    false
}

/// `LDXR` (`info` bit 8 clear) or `STXR` (set) of `1 << info` bytes through the monitor, as the
/// interpreter runs them; the stored value is in `scratch`, and the result goes back there.
extern "sysv64" fn exclusive_slow(state: &mut JitState, addr: u64, pc: u64, info: u32) -> bool {
    let bytes = 1u8 << (info as u8);
    let store = info & 1 << 8 != 0;
    if addr & (u64::from(bytes) - 1) != 0 {
        state.fault = Some(CpuFault::Alignment { pc, addr });
        return true;
    }
    // SAFETY: called from a slow path; see the `Send` impl.
    let (memory, monitor) = unsafe { (state.memory(), &*state.monitor) };
    let result = if store {
        let value = if bytes == 16 {
            state.scratch
        } else {
            state.scratch as u64 as u128
        };
        monitor
            .write(state.core_index, memory, addr, bytes, value)
            .map(|stored| u128::from(!stored))
    } else {
        monitor.read(state.core_index, memory, addr, bytes)
    };
    match result {
        Ok(value) => {
            state.scratch = value;
            state.high = (value >> 64) as u64;
            false
        }
        Err(f) => {
            state.fault = Some(CpuFault::Data {
                pc,
                addr: f.addr,
                write: store,
            });
            true
        }
    }
}

extern "sysv64" fn zero_slow(state: &mut JitState, addr: u64, pc: u64, _info: u32) -> bool {
    // SAFETY: called from a slow path.
    let memory = unsafe { state.memory() };
    match memory.fill(addr, 64, 0) {
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

extern "sysv64" fn ticks(state: &mut JitState) -> u64 {
    // SAFETY: see the `Send` impl.
    unsafe { (*state.clock).ticks() }
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
    /// `None` when the host cannot map JIT code, or under `CpuAccuracy::Paranoid`.
    pub fn new(config: CpuConfig) -> Option<Self> {
        if config.accuracy == CpuAccuracy::Paranoid {
            return None;
        }
        Self::with_passes(config, Passes::ALL)
    }

    pub(crate) fn with_passes(config: CpuConfig, passes: Passes) -> Option<Self> {
        let cache = JitCache::new()?;
        let interrupt = InterruptHandle::default();
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
            scratch2: 0,
            save_xmm: [0; 16],
            save_gpr: [0; 16],
            high: 0,
            cx16_save: [0; 2],
            spill: [0; SPILL_SLOTS],
            memory: &*config.memory,
            clock: &*config.clock,
            monitor: &*config.monitor,
            core_index: config.core_index,
            reservation: config.monitor.slot(config.core_index),
            interrupt: &*interrupt.0,
            table: cache.table(),
            link: 0,
            fault: None,
        });
        Some(Self {
            target: Target {
                arena_size: fastmem.map(|f| f.size() as u64),
                read: read_slow as *const () as usize,
                write: write_slow as *const () as usize,
                atomic: atomic_slow as *const () as usize,
                exclusive: exclusive_slow as *const () as usize,
                zero: zero_slow as *const () as usize,
                ticks: ticks as *const () as usize,
                simd: simd::call as *const () as usize,
                tsc_scale: config.clock.host_tsc_scale().map(std::ptr::from_ref),
                cmpxchg16b: std::arch::is_x86_feature_detected!("cmpxchg16b"),
                fma: std::arch::is_x86_feature_detected!("fma"),
                sse41: std::arch::is_x86_feature_detected!("sse4.1"),
                sse42: std::arch::is_x86_feature_detected!("sse4.2"),
            },
            arena: fastmem.map_or(std::ptr::null_mut(), |f| f.base()),
            config,
            interrupt,
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
                terminal: Terminal::Jump(start),
            },
            config: &self.config,
            cmpxchg16b: self.target.cmpxchg16b,
            sse41: self.target.sse41,
            sse42: self.target.sse42,
        };
        let mut pc = start;
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
                Step::Rejected if pc == start => {
                    return Ok(self.cache.insert(start, 4, Err(inst)));
                }
                Step::Rejected => {
                    t.block.ops.truncate(mark);
                    t.block.terminal = Terminal::Jump(pc);
                    pc -= 4;
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
        self.passes.run(&mut block);
        let compiled = x64::compile(&block, start, &self.target);
        Ok(self.cache.insert(start, len, Ok(compiled)))
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
        let _fp = simd::HostFp::enter();
        loop {
            let link = std::mem::take(&mut self.state.link);
            if self.interrupt.is_signaled() && self.interrupt.take() {
                return HaltReason::Interrupted;
            }
            let generation = self.cache.generation();
            let entry = match self.cache.lookup(ctx.pc) {
                Some(entry) => entry,
                None => match self.translate(ctx.pc) {
                    Ok(entry) => entry,
                    Err(fault) => return HaltReason::Fault(fault),
                },
            };
            let code = match entry.code {
                Code::Native(code) => code,
                Code::Interpret(inst) => {
                    let exit = Interpreter::execute(&self.config, ctx, &[inst]);
                    if let Some(halt) = self.apply(exit) {
                        return halt;
                    }
                    continue;
                }
            };
            if link != 0 && self.cache.generation() == generation {
                self.cache.link(link, ctx.pc, code);
            }
            self.cache.publish(ctx.pc, code);
            let enter = self.cache.trampoline();
            // SAFETY: `code` is a compiled block in this core's code memory; it only touches
            // `ctx`, the state, and the arena behind the bounds check.
            if unsafe { enter(ctx, &mut *self.state, self.arena, code) } == EXIT_FAULT {
                let fault = self
                    .state
                    .fault
                    .take()
                    .expect("a slow path recorded the fault");
                ctx.pc = match fault {
                    CpuFault::Data { pc, .. }
                    | CpuFault::Alignment { pc, .. }
                    | CpuFault::Prefetch { pc }
                    | CpuFault::Undefined { pc, .. } => pc,
                };
                return HaltReason::Fault(fault);
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
