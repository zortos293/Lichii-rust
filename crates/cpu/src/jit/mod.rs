//! A64 JIT for x86-64 hosts.
//!
//! Each guest block is decoded, translated to the block-local IR in [`ir`], optimised by
//! [`passes`] and compiled by [`x64`]. Instructions the translator does not handle end the
//! block and run on the interpreter, which stays the reference: any decodable instruction runs,
//! translated or not. See `crates/cpu/JIT.md` for the design.

mod cache;
mod ir;
mod map;
mod passes;
pub(crate) mod profile;
mod simd;
#[cfg(test)]
mod tests;
mod translate;
mod x64;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use lichii_common::settings::CpuAccuracy;
use lichii_common::{PAGE_MASK, VAddr};
use lichii_memory::{AddressSpace, ExclusiveMonitor, MemoryPermission, ReservationSlot};

use self::cache::{Code, Entry, JitCache};
use self::ir::{Block, Cmp, Terminal};
pub(crate) use self::passes::Passes;
use self::profile::{Flush, Profile};
use self::translate::{Step, Translator};
use self::x64::{EXIT_FAULT, EXIT_INTERRUPT, SPILL_SLOTS, Target};
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
    /// The highest counter an inline read on this core returned. Host cores' TSCs can disagree
    /// by a few cycles, so after the OS moves this core's thread to a lagging one, reads stay here
    /// until the counter passes it instead of stepping back.
    counter_floor: u64,
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
    /// NZCV a block left unpacked: the [`Cmp::lazy_code`] of the flags, 0 when the context holds
    /// them, and the operands they come from.
    lazy_nzcv: u32,
    lazy_a: u64,
    lazy_b: u64,
    /// `x64::materialize_stub` in the code cache.
    materialize: usize,
    /// Lazy NZCV write-backs the native stub made, counted only while profiling.
    materialized: u64,
    profile: Option<Box<Profile>>,
}

// SAFETY: `memory`, `clock` and `interrupt` point into `Arc`s the owning `Jit` keeps, and all
// three are `Sync`; `table` is owned by its cache.
unsafe impl Send for JitState {}

impl JitState {
    /// This core's counts since it last sent them, with the native stub's write-backs moved in.
    fn profile_counts(&mut self) -> Option<&mut Profile> {
        let native = std::mem::take(&mut self.materialized);
        let p = self.profile.as_deref_mut()?;
        p.materialized(native, 0);
        Some(p)
    }

    /// Packs NZCV a block left lazy into `ctx`, so the context is exact outside translated code.
    fn materialize_nzcv(&mut self, ctx: &mut ThreadContext) {
        let code = std::mem::take(&mut self.lazy_nzcv);
        if code != 0 {
            let (kind, w) = Cmp::from_lazy_code(code);
            let nzcv = passes::flags_of(kind, w, self.lazy_a, self.lazy_b) as u32;
            ctx.pstate = ctx.pstate & 0x0fff_ffff | nzcv << 28;
            if let Some(p) = self.profile.as_deref_mut() {
                p.materialized(0, 1);
            }
        }
    }

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
        0 => memory.guest_read::<u8>(addr).map(u128::from),
        1 => memory.guest_read::<u16>(addr).map(u128::from),
        2 => memory.guest_read::<u32>(addr).map(u128::from),
        3 => memory.guest_read::<u64>(addr).map(u128::from),
        _ => memory.guest_read::<u128>(addr),
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
    let (scratch, scratch2) = (state.scratch, state.scratch2);
    let op = (info >> 8) as i32;
    // SAFETY: called from a slow path.
    let memory = unsafe { state.memory() };
    let result = memory.write_with(addr, MemoryPermission::READ_WRITE, |ptr| {
        if size == 4 {
            // SAFETY: the pair is 16-byte aligned, so it lies in the one page of `ptr`.
            return unsafe { cas128(ptr, scratch, scratch2) };
        }
        let (operand, second) = (scratch as u64, scratch2 as u64);
        // SAFETY: the access is naturally aligned, so it stays inside the one mapped guest page
        // of `ptr`.
        u128::from(unsafe {
            if op == x64::ATOMIC_CAS {
                atomic_cas(ptr, size, operand, second)
            } else {
                atomic_rmw(ptr, size, ATOMIC_OPS[op as usize], operand)
            }
        })
    });
    let Ok(old) = result else {
        state.fault = Some(CpuFault::Data {
            pc,
            addr,
            write: true,
        });
        return true;
    };
    state.scratch = old;
    if size == 4 {
        state.high = (old >> 64) as u64;
    }
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
    match memory.guest_fill(addr, 64, 0) {
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
        0 => memory.guest_write(addr, value as u8),
        1 => memory.guest_write(addr, value as u16),
        2 => memory.guest_write(addr, value as u32),
        3 => memory.guest_write(addr, value as u64),
        _ => memory.guest_write(addr, value),
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
    /// When this core last sent its profile towards the process-wide total.
    flushed: Instant,
    /// Set while the profile this core sent last waits to be merged.
    queued: Arc<AtomicBool>,
    /// Blocks this core translated.
    translated: u64,
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
        Self::build(config, passes, profile::enabled())
    }

    fn build(config: CpuConfig, passes: Passes, profile: bool) -> Option<Self> {
        let fastmem = config.memory.fastmem();
        let features = [
            std::arch::is_x86_feature_detected!("cmpxchg16b"),
            std::arch::is_x86_feature_detected!("fma"),
            std::arch::is_x86_feature_detected!("sse4.1"),
            std::arch::is_x86_feature_detected!("sse4.2"),
        ];
        let [cmpxchg16b, fma, sse41, sse42] = features;
        let flavour = [
            passes.get_set,
            passes.fold,
            passes.fuse,
            passes.lazy,
            passes.dce,
            profile,
            fastmem.is_some(),
        ]
        .into_iter()
        .chain(features)
        .fold(0, |bits, on| bits << 1 | u64::from(on));
        let cache = JitCache::new(cache::Shared::of(&config.memory, flavour, profile)?);
        let interrupt = InterruptHandle::default();
        if let Some(fastmem) = fastmem {
            fastmem.set_fault_handler(Some(cache::on_fault), 0);
        }
        let state = Box::new(JitState {
            scratch: 0,
            scratch2: 0,
            save_xmm: [0; 16],
            save_gpr: [0; 16],
            high: 0,
            counter_floor: 0,
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
            lazy_nzcv: 0,
            lazy_a: 0,
            lazy_b: 0,
            materialize: cache.materialize(),
            materialized: 0,
            profile: profile.then(Box::default),
        });
        if profile {
            profile::register();
        }
        let (simd, simd_slow) = if profile {
            (
                simd::call_profiled as *const () as usize,
                simd::slow_profiled as *const () as usize,
            )
        } else {
            (
                simd::call as *const () as usize,
                simd::call as *const () as usize,
            )
        };
        Some(Self {
            target: Target {
                arena_size: fastmem.map(|f| f.size() as u64),
                read: read_slow as *const () as usize,
                write: write_slow as *const () as usize,
                atomic: atomic_slow as *const () as usize,
                exclusive: exclusive_slow as *const () as usize,
                zero: zero_slow as *const () as usize,
                ticks: ticks as *const () as usize,
                simd,
                simd_slow,
                tsc_scale: config.clock.host_tsc_scale().map(std::ptr::from_ref),
                cmpxchg16b,
                fma,
                sse41,
                sse42,
            },
            arena: fastmem.map_or(std::ptr::null_mut(), |f| f.base()),
            config,
            interrupt,
            cache,
            state,
            passes,
            flushed: Instant::now(),
            queued: Arc::default(),
            translated: 0,
        })
    }

    fn translate(&mut self, start: u64) -> Result<Entry, CpuFault> {
        if start & 3 != 0 {
            return Err(CpuFault::Prefetch { pc: start });
        }
        self.translated += 1;
        let started = self.state.profile.is_some().then(Instant::now);
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
                    let entry = self.cache.insert(start, 4, Err(inst));
                    if let (Some(p), Some(started)) = (self.state.profile.as_deref_mut(), started) {
                        p.translated(started.elapsed(), 0, 0);
                    }
                    return Ok(entry);
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
        let code_bytes = compiled.code.len();
        let mapped = map::enabled().then(|| compiled.code.clone());
        let generation = self.cache.generation();
        let entry = self.cache.insert(start, len, Ok(compiled));
        if let (Some(code), cache::Code::Native(host)) = (mapped, entry.code) {
            let words: Vec<u32> = (0..len / 4)
                .map(|i| self.config.memory.read::<u32>(start + 4 * i).unwrap_or(0))
                .collect();
            map::record(host as usize, code, start, words);
        }
        if let (Some(p), Some(started)) = (self.state.profile.as_deref_mut(), started) {
            p.translated(started.elapsed(), len / 4, code_bytes);
            if self.cache.generation() != generation {
                p.flush(Flush::Full);
            }
        }
        Ok(entry)
    }

    fn count_flush(&mut self, flush: Flush) {
        if let Some(p) = self.state.profile.as_deref_mut() {
            p.flush(flush);
        }
    }

    fn apply(&mut self, exit: Exit) -> Option<HaltReason> {
        match exit {
            Exit::Continue => None,
            Exit::Halt(halt) => Some(halt),
            Exit::Invalidate(addr, size) => {
                self.count_flush(Flush::Invalidate);
                self.cache.invalidate(addr, size);
                Some(HaltReason::InvalidateCode { addr, size })
            }
            Exit::InvalidateAll => {
                self.count_flush(Flush::Clear);
                self.cache.invalidate(0, u64::MAX);
                Some(HaltReason::InvalidateCode {
                    addr: 0,
                    size: u64::MAX,
                })
            }
        }
    }

    fn dispatch(&mut self, ctx: &mut ThreadContext) -> HaltReason {
        self.cache.enter();
        let halt = self.run_blocks(ctx);
        self.cache.leave();
        halt
    }

    /// The dispatcher loop. Only between `JitCache::enter` and `leave`, which keep the code it
    /// runs mapped.
    fn run_blocks(&mut self, ctx: &mut ThreadContext) -> HaltReason {
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
                    if let Some(p) = self.state.profile.as_deref_mut() {
                        p.fallback(&inst);
                    }
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
            self.state.materialize = self.cache.materialize();
            let enter = self.cache.trampoline();
            // SAFETY: `code` is a compiled block in this core's code memory; it only touches
            // `ctx`, the state, and the arena behind the bounds check.
            let exit = unsafe { enter(ctx, &mut *self.state, self.arena, code) };
            self.state.materialize_nzcv(ctx);
            let state = &mut *self.state;
            if let Some(p) = state.profile.as_deref_mut() {
                p.exit(match exit {
                    EXIT_FAULT => profile::Exit::Fault,
                    EXIT_INTERRUPT => profile::Exit::Interrupt,
                    _ if state.link != 0 => profile::Exit::Unlinked,
                    _ => profile::Exit::IndirectMiss,
                });
            }
            if exit == EXIT_FAULT {
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
}

impl CpuCore for Jit {
    fn run(&mut self, ctx: &mut ThreadContext) -> HaltReason {
        let halt = self.dispatch(ctx);
        if self.state.profile.is_some() && self.flushed.elapsed() >= profile::FLUSH_INTERVAL {
            self.flushed = Instant::now();
            if let Some(p) = self.state.profile_counts() {
                profile::send(&self.queued, p);
            }
        }
        halt
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
        self.count_flush(Flush::Invalidate);
        self.cache.invalidate(addr, size);
    }

    fn clear_code_cache(&mut self) {
        self.count_flush(Flush::Clear);
        self.cache.clear();
    }

    fn clear_exclusive(&mut self) {
        self.config.monitor.clear(self.config.core_index);
    }
}

impl Drop for Jit {
    fn drop(&mut self) {
        if let Some(p) = self.state.profile_counts() {
            profile::retire(std::mem::take(p));
            tracing::info!(
                core = self.config.core_index,
                translated = self.translated,
                shared = self.cache.adopted(),
                "JIT core totals"
            );
        }
        map::sync();
    }
}
