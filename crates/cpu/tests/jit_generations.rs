//! Clearing the shared JIT code while another core stays in its dispatcher pins every retired
//! generation, until no fault-table slot is left for a fresh one. Its own process, because it
//! takes every slot of the process.
#![cfg(all(unix, target_arch = "x86_64"))]

use std::sync::Arc;

use lichii_common::settings::{CpuAccuracy, CpuBackend};
use lichii_cpu::{CpuConfig, GuestClock, HaltReason, InterruptHandle, ThreadContext, new_core};
use lichii_memory::{AddressSpace, ExclusiveMonitor, MemoryPermission, PhysicalMemory};

const CODE: u64 = 0x1000_0000;
const DATA: u64 = 0x2000_0000;

struct Clock;

impl GuestClock for Clock {
    fn ticks(&self) -> u64 {
        0
    }
}

struct SignalOnDrop(InterruptHandle);

impl Drop for SignalOnDrop {
    fn drop(&mut self) {
        self.0.signal();
    }
}

#[test]
fn clears_past_the_last_fault_table_slot_keep_every_core_running() {
    const MOV_X1_1: u32 = 0xd280_0021;
    const STR_X1_X2: u32 = 0xf900_0041;
    const B_SELF: u32 = 0x1400_0000;
    const MOV_X0_7: u32 = 0xd280_00e0;
    const BRK: u32 = 0xd420_0000;
    let physical = Arc::new(PhysicalMemory::new(1 << 16).unwrap());
    let mem = Arc::new(AddressSpace::with_fastmem(physical, 39).unwrap());
    mem.map(CODE, 0, 0x1000, MemoryPermission::READ_EXECUTE);
    mem.map(DATA, 0x1000, 0x1000, MemoryPermission::READ_WRITE);
    let code = [MOV_X1_1, STR_X1_X2, B_SELF, MOV_X0_7, BRK];
    for (i, word) in code.into_iter().enumerate() {
        mem.write(CODE + 4 * i as u64, word).unwrap();
    }
    let monitor = Arc::new(ExclusiveMonitor::new(2));
    let clock: Arc<dyn GuestClock> = Arc::new(Clock);
    let mut cores = (0..2).map(|core_index| {
        new_core(
            CpuBackend::Jit,
            CpuConfig {
                memory: mem.clone(),
                monitor: monitor.clone(),
                clock: clock.clone(),
                core_index,
                ctr_el0: CpuConfig::DEFAULT_CTR_EL0,
                dczid_el0: CpuConfig::DEFAULT_DCZID_EL0,
                accuracy: CpuAccuracy::Auto,
            },
        )
    });
    let (mut spinning, mut clearing) = (cores.next().unwrap(), cores.next().unwrap());
    let stop = SignalOnDrop(spinning.interrupt_handle());
    std::thread::scope(|scope| {
        let spinner = scope.spawn(move || {
            let mut ctx = ThreadContext {
                pc: CODE,
                ..Default::default()
            };
            ctx.x[2] = DATA;
            spinning.run(&mut ctx)
        });
        while mem.read::<u64>(DATA).unwrap() != 1 {
            std::thread::yield_now();
        }
        for _ in 0..300 {
            clearing.clear_code_cache();
        }
        let mut ctx = ThreadContext {
            pc: CODE + 12,
            ..Default::default()
        };
        assert_eq!(clearing.run(&mut ctx), HaltReason::Breakpoint(0));
        assert_eq!(ctx.x[0], 7);
        drop(stop);
        assert_eq!(spinner.join().unwrap(), HaltReason::Interrupted);
    });
    clearing.clear_code_cache();
    let mut ctx = ThreadContext {
        pc: CODE + 12,
        ..Default::default()
    };
    assert_eq!(clearing.run(&mut ctx), HaltReason::Breakpoint(0));
    assert_eq!(ctx.x[0], 7);
}
