//! Interpreter and JIT throughput on synthetic guest loops, in guest MIPS.
//!
//! `cargo bench -p lichii-cpu --bench cpu`; `LICHII_BENCH_ITERATIONS` sets the loop count
//! (default 10M, zero is ignored).

use std::sync::Arc;
use std::time::Instant;

use lichii_common::settings::{CpuAccuracy, CpuBackend};
use lichii_cpu::{CpuConfig, GuestClock, HaltReason, ThreadContext, TscScale, new_core};
use lichii_memory::{AddressSpace, ExclusiveMonitor, MemoryPermission, PhysicalMemory};

const CODE: u64 = 0x10000;
const DATA: u64 = 0x100000;
/// Where `far` code goes: its addresses share indirect-table slots with `CODE`'s.
const FAR: u64 = CODE + 0x40000;

/// A 19.2 MHz counter whose `ticks` costs what `CoreTiming`'s does (an `Instant` read and a 128-bit
/// divide), with a TSC scale that lets the JIT read it inline instead.
struct Clock {
    start: Instant,
    scale: TscScale,
}

impl GuestClock for Clock {
    fn ticks(&self) -> u64 {
        (self.start.elapsed().as_nanos() * 19_200_000 / 1_000_000_000) as u64
    }

    fn host_tsc_scale(&self) -> Option<&TscScale> {
        Some(&self.scale)
    }
}

/// A loop body that runs `per_iteration` instructions, counted down in X5.
struct Workload {
    name: &'static str,
    code: &'static [u32],
    far: &'static [u32],
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
        far: &[],
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
        far: &[],
        per_iteration: 6,
    },
    Workload {
        name: "compare/select loop",
        code: &[
            0xeb01001f, // 1: cmp x0, x1
            0x9a813002, // csel x2, x0, x1, lo
            0xeb03005f, // cmp x2, x3
            0x9a84b484, // cinc x4, x4, ge
            0xf240001f, // tst x0, #1
            0x9a8000c6, // csel x6, x6, x0, eq
            0x91000c00, // add x0, x0, #3
            0xf10004a5, // subs x5, x5, #1
            0x54ffff01, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 9,
    },
    Workload {
        name: "atomic add loop",
        code: &[
            0xf8220023, // 1: ldadd x2, x3, [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54ffffc1, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 3,
    },
    Workload {
        name: "colliding returns",
        code: &[
            0x94000008, // 1: bl 3f
            0x1400ffff, // b 2f
            0, 0, 0, 0, 0, 0, 0x91000442, // 3: add x2, x2, #1
            0xd65f03c0, // ret
        ],
        far: &[
            0x97ff0008, // 2: bl 3b
            0xf10004a5, // subs x5, x5, #1
            0x54dfffc1, // b.ne 1b
            0xd4000001, // svc #0
        ],
        per_iteration: 9,
    },
    Workload {
        name: "counter read loop",
        code: &[
            0xd53be020, // 1: mrs x0, cntpct_el0
            0xf10004a5, // subs x5, x5, #1
            0x54ffffc1, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 3,
    },
    Workload {
        name: "exclusive add loop",
        code: &[
            0xc85f7c23, // 1: ldxr x3, [x1]
            0x91000463, // add x3, x3, #1
            0xc8047c23, // stxr w4, x3, [x1]
            0x35ffffa4, // cbnz w4, 1b
            0xf10004a5, // subs x5, x5, #1
            0x54ffff61, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 6,
    },
    Workload {
        name: "scalar fp loop",
        code: &[
            0x1c0001a0, // 1: ldr s0, 2f
            0x1c0001a1, // ldr s1, 3f
            0x1e210802, // fmul s2, s0, s1
            0x1e222863, // fadd s3, s3, s2
            0x1f011004, // fmadd s4, s0, s1, s4
            0x1e233885, // fsub s5, s4, s3
            0x1e2020a0, // fcmp s5, s0
            0x1e20cca6, // fcsel s6, s5, s0, gt
            0x1e22c0c7, // fcvt d7, s6
            0xfd000427, // str d7, [x1, #8]
            0xf10004a5, // subs x5, x5, #1
            0x54fffea1, // b.ne 1b
            0xd4000001, // svc #0
            0x3f8ccccd, // 2: 1.1
            0x3e99999a, // 3: 0.3
        ],
        far: &[],
        per_iteration: 12,
    },
    Workload {
        name: "exclusive pair loop",
        code: &[
            0xc87f1823, // 1: ldxp x3, x6, [x1]
            0x91000463, // add x3, x3, #1
            0xc8241823, // stxp w4, x3, x6, [x1]
            0x35ffffa4, // cbnz w4, 1b
            0xf10004a5, // subs x5, x5, #1
            0x54ffff61, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 6,
    },
    Workload {
        name: "vector int loop",
        code: &[
            0x4c407820, // 1: ld1 {v0.4s}, [x1]
            0x4e040c41, // dup v1.4s, w2
            0x4ea18400, // add v0.4s, v0.4s, v1.4s
            0x6e211c02, // eor v2.16b, v0.16b, v1.16b
            0x4f235442, // shl v2.4s, v2.4s, #3
            0x4ea03443, // cmgt v3.4s, v2.4s, v0.4s
            0x0e0c3c42, // umov w2, v2.s[1]
            0x4c007822, // st1 {v2.4s}, [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54fffee1, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 10,
    },
    Workload {
        name: "vector lane loop",
        code: &[
            0x0d409020, // 1: ld1 {v0.s}[1], [x1]
            0x4d40c821, // ld1r {v1.4s}, [x1]
            0x4ea18400, // add v0.4s, v0.4s, v1.4s
            0x4d608022, // ld2 {v2.s, v3.s}[2], [x1]
            0x6e231c42, // eor v2.16b, v2.16b, v3.16b
            0x0d009020, // st1 {v0.s}[1], [x1]
            0x4d004822, // st1 {v2.h}[5], [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54ffff01, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 9,
    },
    Workload {
        name: "vector fp loop",
        code: &[
            0x4f03f702, // fmov v2.4s, #1.5
            0x4f03f500, // fmov v0.4s, #0.75
            0x4fa21001, // 1: fmla v1.4s, v0.4s, v2.s[1]
            0x4f829003, // fmul v3.4s, v0.4s, v2.s[0]
            0x4e23d424, // fadd v4.4s, v1.4s, v3.4s
            0x4e20cc85, // fmla v5.4s, v4.4s, v0.4s
            0x6e22dca6, // fmul v6.4s, v5.4s, v2.4s
            0x4ea4d4c7, // fsub v7.4s, v6.4s, v4.4s
            0x4c007827, // st1 {v7.4s}, [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54ffff01, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 9,
    },
    Workload {
        name: "fp element loop",
        code: &[
            0x4f03f702, // fmov v2.4s, #1.5
            0x4f03f500, // fmov v0.4s, #0.75
            0x5fa29001, // 1: fmul s1, s0, v2.s[1]
            0x5f821824, // fmla s4, s1, v2.s[2]
            0x5fc25805, // fmls d5, d0, v2.d[1]
            0x5e21d803, // scvtf s3, s0
            0x1e232886, // fadd s6, s4, s3
            0x7e61d827, // ucvtf d7, d1
            0xbd000026, // str s6, [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54ffff01, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 9,
    },
    Workload {
        name: "vector struct loop",
        code: &[
            0x4c408820, // 1: ld2 {v0.4s, v1.4s}, [x1]
            0x4ea18402, // add v2.4s, v0.4s, v1.4s
            0x4c008821, // st2 {v1.4s, v2.4s}, [x1]
            0x4c400424, // ld4 {v4.8h, v5.8h, v6.8h, v7.8h}, [x1]
            0x6e271c84, // eor v4.16b, v4.16b, v7.16b
            0x0c004024, // st3 {v4.8b, v5.8b, v6.8b}, [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54ffff21, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 8,
    },
    Workload {
        name: "vector ext loop",
        code: &[
            0x4c407020, // 1: ld1 {v0.16b}, [x1]
            0x6e021801, // ext v1.16b, v0.16b, v2.16b, #3
            0x2e002822, // ext v2.8b, v1.8b, v0.8b, #5
            0x6e016043, // ext v3.16b, v2.16b, v1.16b, #12
            0x6e211c60, // eor v0.16b, v3.16b, v1.16b
            0x6e004004, // ext v4.16b, v0.16b, v0.16b, #8
            0x4c007024, // st1 {v4.16b}, [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54ffff01, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 9,
    },
    Workload {
        name: "half convert loop",
        code: &[
            0x7d400020, // 1: ldr h0, [x1]
            0x1ee24001, // fcvt s1, h0
            0x1ee2c002, // fcvt d2, h0
            0x1e20c023, // fabs s3, s1
            0x1e614044, // fneg d4, d2
            0x7d400425, // ldr h5, [x1, #2]
            0x1ee240a6, // fcvt s6, h5
            0x1e2600c7, // fmov w7, s6
            0xf10004a5, // subs x5, x5, #1
            0x54fffee1, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 10,
    },
    Workload {
        name: "fp compare loop",
        code: &[
            0x1c000180, // 1: ldr s0, 2f
            0x1c000181, // ldr s1, 3f
            0x1e218802, // fnmul s2, s0, s1
            0x1e212040, // fcmp s2, s1
            0x1e214404, // fccmp s0, s1, #4, mi
            0x1e208843, // fnmul s3, s2, s0
            0x1e200472, // fccmpe s3, s0, #2, eq
            0x1e23cc06, // fcsel s6, s0, s3, gt
            0xbd000026, // str s6, [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54fffec1, // b.ne 1b
            0xd4000001, // svc #0
            0x3f8ccccd, // 2: 1.1
            0x3e99999a, // 3: 0.3
        ],
        far: &[],
        per_iteration: 11,
    },
    Workload {
        name: "lane sum loop",
        code: &[
            0x4f03f700, // fmov v0.4s, #1.5
            0x1e2e1007, // fmov s7, #1.0
            0x1e211008, // fmov s8, #3.0
            0x1e2818e9, // fdiv s9, s7, s8
            0x5e0c0401, // 1: mov s1, v0.s[1]
            0x5e140402, // mov s2, v0.s[2]
            0x5e1c0403, // mov s3, v0.s[3]
            0x6e21d404, // faddp v4.4s, v0.4s, v1.4s
            0x2e22d485, // faddp v5.2s, v4.2s, v2.2s
            0x6e23d4a6, // faddp v6.4s, v5.4s, v3.4s
            0x4c007826, // st1 {v6.4s}, [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54ffff01, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 9,
    },
    Workload {
        name: "estimate step loop",
        code: &[
            0x4f03f700, // fmov v0.4s, #1.5
            0x4f03f687, // fmov v7.4s, #1.25
            0x4f000461, // movi v1.4s, #3
            0x1e2e1008, // fmov s8, #1.0
            0x1e201909, // fdiv s9, s8, s0
            0x4f03f502, // 1: fmov v2.4s, #0.75
            0x6e22dc43, // fmul v3.4s, v2.4s, v2.4s
            0x4ea0fc63, // frsqrts v3.4s, v3.4s, v0.4s
            0x4e22fce4, // frecps v4.4s, v7.4s, v2.4s
            0x4e21d825, // scvtf v5.4s, v1.4s
            0x6e25dc86, // fmul v6.4s, v4.4s, v5.4s
            0x4ea3fcc6, // frsqrts v6.4s, v6.4s, v3.4s
            0x4c007826, // st1 {v6.4s}, [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54fffee1, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 10,
    },
    Workload {
        name: "zero product loop",
        code: &[
            0x1e2703e0, // fmov s0, wzr
            0x1e2f1002, // fmov s2, #1.5
            0x1e2e1007, // fmov s7, #1.0
            0x1e211008, // fmov s8, #3.0
            0x1e2818e9, // fdiv s9, s7, s8
            0x1e220801, // 1: fmul s1, s0, s2
            0x1f020403, // fmadd s3, s0, s2, s1
            0x1e290844, // fmul s4, s2, s9
            0x1f000c85, // fmadd s5, s4, s0, s3
            0x1e248806, // fnmul s6, s0, s4
            0x1e22180a, // fdiv s10, s0, s2
            0x1f09184b, // fmadd s11, s2, s9, s6
            0xbd00002b, // str s11, [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54fffee1, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 10,
    },
    Workload {
        name: "saturating narrow loop",
        code: &[
            0x4f000460, // movi v0.4s, #3
            0x4f0367e1, // movi v1.4s, #0x7f, lsl #24
            0x0e614802, // 1: sqxtn v2.4h, v0.4s
            0x4e614822, // sqxtn2 v2.8h, v1.4s
            0x0e214843, // sqxtn v3.8b, v2.8h
            0x4e214843, // sqxtn2 v3.16b, v2.8h
            0x4c007023, // st1 {v3.16b}, [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54ffff41, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 7,
    },
    Workload {
        name: "popcount loop",
        code: &[
            0x4c407020, // 1: ld1 {v0.16b}, [x1]
            0x4e205801, // cnt v1.16b, v0.16b
            0x0e205802, // cnt v2.8b, v0.8b
            0x0e22bc43, // addp v3.8b, v2.8b, v2.8b
            0x4e63bc24, // addp v4.8h, v1.8h, v3.8h
            0x4ea0bc85, // addp v5.4s, v4.4s, v0.4s
            0x4ee1bca6, // addp v6.2d, v5.2d, v1.2d
            0x6e261c00, // eor v0.16b, v0.16b, v6.16b
            0x4c007020, // st1 {v0.16b}, [x1]
            0xf10004a5, // subs x5, x5, #1
            0x54fffec1, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 11,
    },
    Workload {
        name: "crc32 loop",
        code: &[
            0xf8607823, // 1: ldr x3, [x1, x0, lsl #3]
            0x1ac34042, // crc32b w2, w2, w3
            0x1ac34442, // crc32h w2, w2, w3
            0x1ac34842, // crc32w w2, w2, w3
            0x9ac34c42, // crc32x w2, w2, x3
            0x91000400, // add x0, x0, #1
            0x92402400, // and x0, x0, #0x3ff
            0xf10004a5, // subs x5, x5, #1
            0x54ffff01, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 9,
    },
    Workload {
        name: "crc32c loop",
        code: &[
            0xf8607823, // 1: ldr x3, [x1, x0, lsl #3]
            0x1ac35042, // crc32cb w2, w2, w3
            0x1ac35442, // crc32ch w2, w2, w3
            0x1ac35842, // crc32cw w2, w2, w3
            0x9ac35c42, // crc32cx w2, w2, x3
            0x91000400, // add x0, x0, #1
            0x92402400, // and x0, x0, #0x3ff
            0xf10004a5, // subs x5, x5, #1
            0x54ffff01, // b.ne 1b
            0xd4000001, // svc #0
        ],
        far: &[],
        per_iteration: 9,
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
        mem.map(FAR, 0x3000, 0x1000, MemoryPermission::READ_EXECUTE);
        for (base, words) in [(CODE, workload.code), (FAR, workload.far)] {
            for (i, word) in words.iter().enumerate() {
                mem.write(base + 4 * i as u64, *word).unwrap();
            }
        }
        let mut core = new_core(
            backend,
            CpuConfig {
                memory: mem,
                monitor: Arc::new(ExclusiveMonitor::new(1)),
                clock: Arc::new(Clock {
                    start: Instant::now(),
                    scale: TscScale::new(1 << 31, 32, 0),
                }),
                core_index: 0,
                ctr_el0: CpuConfig::DEFAULT_CTR_EL0,
                dczid_el0: CpuConfig::DEFAULT_DCZID_EL0,
                accuracy: CpuAccuracy::Auto,
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
