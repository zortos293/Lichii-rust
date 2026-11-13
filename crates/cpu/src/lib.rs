//! AArch64 (A64) guest CPU.
//!
//! The kernel owns one [`CpuCore`] per emulated core and swaps [`ThreadContext`]s on it.
//! [`CpuCore::run`] executes guest code until something needs the kernel (an `SVC`, a fault, an
//! interrupt request) and reports it as a [`HaltReason`].

pub mod decoder;
mod interpreter;
#[cfg(target_arch = "x86_64")]
mod jit;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use lichii_common::VAddr;
use lichii_common::settings::CpuBackend;
use lichii_memory::{AddressSpace, ExclusiveMonitor};

pub use interpreter::Interpreter;
#[cfg(target_arch = "x86_64")]
pub use jit::Jit;

/// Architectural state of one guest thread.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct ThreadContext {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    /// NZCV in bits 31..28.
    pub pstate: u32,
    pub fpcr: u32,
    pub fpsr: u32,
    pub v: [u128; 32],
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
}

/// Why [`CpuCore::run`] returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HaltReason {
    /// `SVC #imm`; `pc` already points past the instruction.
    SupervisorCall(u32),
    /// `BRK #imm`; `pc` points at the instruction.
    Breakpoint(u16),
    /// [`InterruptHandle::signal`] was called; resume with another `run`.
    Interrupted,
    /// [`CpuCore::step`] executed one instruction.
    StepComplete,
    Fault(CpuFault),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuFault {
    /// Data access to an unmapped or protected address.
    Data { pc: VAddr, addr: VAddr, write: bool },
    /// Instruction fetch from an unmapped address.
    Prefetch { pc: VAddr },
    /// The instruction at `pc` is unallocated or not implemented.
    Undefined { pc: VAddr, instruction: u32 },
    /// Misaligned exclusive or atomic access.
    Alignment { pc: VAddr, addr: VAddr },
}

/// The guest's system counter (`CNTPCT_EL0`).
pub trait GuestClock: Send + Sync {
    /// Ticks at [`GuestClock::frequency`] Hz.
    fn ticks(&self) -> u64;
    fn frequency(&self) -> u64 {
        19_200_000
    }
    /// How [`GuestClock::ticks`] follows the host TSC, so translated code can read the counter with
    /// `rdtsc` instead of calling `ticks`. `None` when the TSC is not invariant or the clock does
    /// not follow it. A core reads the scale on every counter access, so the clock may change it,
    /// for example on pause and resume, but only while no core runs guest code.
    fn host_tsc_scale(&self) -> Option<&TscScale> {
        None
    }
}

/// Guest ticks as a function of the host TSC: `offset + ((tsc * mul) >> shift)`, with a 128-bit
/// product and a wrapping add. `shift` is below 64.
#[derive(Debug, Default)]
#[repr(C)]
pub struct TscScale {
    pub mul: AtomicU64,
    pub shift: AtomicU64,
    pub offset: AtomicU64,
}

impl TscScale {
    pub fn new(mul: u64, shift: u32, offset: u64) -> Self {
        assert!(shift < 64, "TSC scale shift {shift}");
        Self {
            mul: AtomicU64::new(mul),
            shift: AtomicU64::new(shift.into()),
            offset: AtomicU64::new(offset),
        }
    }

    /// The guest ticks at host TSC value `tsc`.
    pub fn ticks(&self, tsc: u64) -> u64 {
        let product = u128::from(tsc) * u128::from(self.mul.load(Ordering::Relaxed));
        let scaled = (product >> self.shift.load(Ordering::Relaxed)) as u64;
        self.offset.load(Ordering::Relaxed).wrapping_add(scaled)
    }
}

/// Wakes a core out of [`CpuCore::run`] from another thread.
#[derive(Clone, Debug, Default)]
pub struct InterruptHandle(Arc<AtomicBool>);

impl InterruptHandle {
    pub fn signal(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Returns whether an interrupt was pending and clears it.
    pub fn take(&self) -> bool {
        self.0.swap(false, Ordering::AcqRel)
    }

    pub fn is_signaled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Everything a core needs from the rest of the system.
#[derive(Clone)]
pub struct CpuConfig {
    pub memory: Arc<AddressSpace>,
    pub monitor: Arc<ExclusiveMonitor>,
    pub clock: Arc<dyn GuestClock>,
    pub core_index: usize,
    pub ctr_el0: u32,
    pub dczid_el0: u32,
}

impl CpuConfig {
    pub const DEFAULT_CTR_EL0: u32 = 0x8444_c004;
    pub const DEFAULT_DCZID_EL0: u32 = 4;
}

pub trait CpuCore: Send {
    /// Runs guest code from `ctx.pc` until a [`HaltReason`].
    fn run(&mut self, ctx: &mut ThreadContext) -> HaltReason;
    /// Executes exactly one instruction.
    fn step(&mut self, ctx: &mut ThreadContext) -> HaltReason;
    fn interrupt_handle(&self) -> InterruptHandle;
    /// Drops translated code for guest addresses `[addr, addr + size)`.
    fn invalidate_code(&mut self, addr: VAddr, size: u64);
    fn clear_code_cache(&mut self);
    /// Forgets this core's exclusive reservation (on context switch).
    fn clear_exclusive(&mut self);
}

pub fn new_core(backend: CpuBackend, config: CpuConfig) -> Box<dyn CpuCore> {
    match backend {
        #[cfg(target_arch = "x86_64")]
        CpuBackend::Jit => match Jit::new(config.clone()) {
            Some(jit) => Box::new(jit),
            None => Box::new(Interpreter::new(config)),
        },
        _ => Box::new(Interpreter::new(config)),
    }
}

#[cfg(test)]
mod tests {
    use super::TscScale;

    #[test]
    fn tsc_scale_multiplies_shifts_and_offsets() {
        let scale = TscScale::new(3 << 32, 32, 10);
        assert_eq!(scale.ticks(7), 31);
        let wide = TscScale::new(u64::MAX, 63, u64::MAX);
        assert_eq!(
            wide.ticks(u64::MAX),
            ((u128::from(u64::MAX) * u128::from(u64::MAX)) >> 63) as u64 - 1
        );
    }
}
