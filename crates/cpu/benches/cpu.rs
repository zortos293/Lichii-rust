//! Interpreter and JIT throughput on synthetic guest loops, in guest MIPS.
//!
//! `cargo bench -p lichii-cpu --bench cpu`; `LICHII_BENCH_ITERATIONS` sets the loop count
//! (default 10M, zero is ignored).

use std::sync::Arc;
use std::time::Instant;

use lichii_common::settings::CpuBackend;
use lichii_cpu::{CpuConfig, GuestClock, HaltReason, ThreadContext, new_core};
use lichii_memory::{AddressSpace, ExclusiveMonitor, MemoryPermission, PhysicalMemory};

const CODE: u64 = 0x10000;
const DATA: u64 = 0x100000;

struct Clock;

impl GuestClock for Clock {
    fn ticks(&self) -> u64 {
        0
    }
}

/// A loop body that runs `per_iteration` instructions, counted down in X5.
struct Workload {
    name: &'static str,
    code: &'static [u32],
    per_iteration: u64,
}

const WORKLOADS: &[Workload] = &[
    Workload {
        name: "alu+memory loop",
        code: &[
            0xf8607823, // 1: ldr x3, [x1, x0, lsl #3]
            0x8b030042, // add x2, x2, x3
            0xca000044, // eor x4, x2, x0
            0xf8207824, // str x4, [x1, x0, lsl #3]
            0x91000400, // add x0, x0, #1
            0x92402400, // and x0, x0, #0x3ff
            0xf10004a5, // subs x5, x5, #1
            0x54ffff21, // b.ne 1b
            0xd4000001, // svc #0
        ],
        per_iteration: 8,
    },
    Workload {
        name: "call/return loop",
        code: &[
            0x94000004, // 1: bl 2f
            0xf10004a5, // subs x5, x5, #1
            0x54ffffc1, // b.ne 1b
            0xd4000001, // svc #0
            0x91000442, // 2: add x2, x2, #1
            0xb4000042, // cbz x2, 3f
            0xd65f03c0, // ret
            0xd65f03c0, // 3: ret
        ],
        per_iteration: 6,
    },
];

fn main() {
    let iterations: u64 = std::env::var("LICHII_BENCH_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(10_000_000);
    for (workload, backend) in WORKLOADS
        .iter()
        .flat_map(|w| [(w, CpuBackend::Interpreter), (w, CpuBackend::Jit)])
    {
        let mem = Arc::new(
            AddressSpace::with_fastmem(Arc::new(PhysicalMemory::new(1 << 20).unwrap()), 39)
                .unwrap(),
        );
        mem.map(CODE, 0, 0x1000, MemoryPermission::READ_EXECUTE);
        mem.map(DATA, 0x1000, 0x2000, MemoryPermission::READ_WRITE);
        for (i, word) in workload.code.iter().enumerate() {
            mem.write(CODE + 4 * i as u64, *word).unwrap();
        }
        let mut core = new_core(
            backend,
            CpuConfig {
                memory: mem,
                monitor: Arc::new(ExclusiveMonitor::new(1)),
                clock: Arc::new(Clock),
                core_index: 0,
                ctr_el0: CpuConfig::DEFAULT_CTR_EL0,
                dczid_el0: CpuConfig::DEFAULT_DCZID_EL0,
            },
        );
        let mut seconds = f64::MAX;
        for _ in 0..3 {
            let mut ctx = ThreadContext {
                pc: CODE,
                ..Default::default()
            };
            ctx.x[1] = DATA;
            ctx.x[5] = iterations;
            let start = Instant::now();
            assert_eq!(core.run(&mut ctx), HaltReason::SupervisorCall(0));
            seconds = seconds.min(start.elapsed().as_secs_f64());
        }
        let instructions = iterations * workload.per_iteration;
        println!(
            "{:<18} {:<12} {:>8.1} MIPS ({instructions} instructions in {seconds:.3} s, best of 3)",
            workload.name,
            format!("{backend:?}"),
            instructions as f64 / seconds / 1e6
        );
    }
}
