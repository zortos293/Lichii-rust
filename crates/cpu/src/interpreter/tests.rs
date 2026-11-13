use std::sync::Arc;
use std::time::Duration;

use lichii_common::settings::CpuAccuracy;
use lichii_memory::{AddressSpace, ExclusiveMonitor, MemoryPermission, PhysicalMemory};

use super::Interpreter;

use crate::{CpuConfig, CpuCore, CpuFault, GuestClock, HaltReason, ThreadContext};

const CODE: u64 = 0x10000;
const DATA: u64 = 0x80000;

struct FixedClock;

impl GuestClock for FixedClock {
    fn ticks(&self) -> u64 {
        123_456
    }
}

struct Harness {
    core: Interpreter,
    ctx: ThreadContext,
    mem: Arc<AddressSpace>,
}

fn harness(code: &[u32]) -> Harness {
    let mem = Arc::new(AddressSpace::new(
        Arc::new(PhysicalMemory::new(1 << 20).unwrap()),
        39,
    ));
    mem.map(CODE, 0, 0x4000, MemoryPermission::READ_EXECUTE);
    mem.map(DATA, 0x4000, 0x4000, MemoryPermission::READ_WRITE);
    for (i, word) in code.iter().enumerate() {
        mem.write(CODE + 4 * i as u64, *word).unwrap();
    }
    let core = Interpreter::new(CpuConfig {
        memory: mem.clone(),
        monitor: Arc::new(ExclusiveMonitor::new(1)),
        clock: Arc::new(FixedClock),
        core_index: 0,
        ctr_el0: CpuConfig::DEFAULT_CTR_EL0,
        dczid_el0: CpuConfig::DEFAULT_DCZID_EL0,
        accuracy: CpuAccuracy::Auto,
    });
    let ctx = ThreadContext {
        pc: CODE,
        sp: DATA + 0x4000,
        ..Default::default()
    };
    Harness { core, ctx, mem }
}

#[test]
fn runs_a_loop_until_svc() {
    let mut h = harness(&[
        0xd2800000, // mov x0, #0
        0xd2800c81, // mov x1, #100
        0x8b010000, // add x0, x0, x1
        0xf1000421, // subs x1, x1, #1
        0x54ffffc1, // b.ne -8
        0xd4000021, // svc #1
    ]);
    assert_eq!(h.core.run(&mut h.ctx), HaltReason::SupervisorCall(1));
    assert_eq!(h.ctx.x[0], 5050);
    assert_eq!(h.ctx.pc, CODE + 24);
    assert_eq!(h.ctx.pstate >> 28, 0b0110);
}

#[test]
fn exclusives_and_atomics_update_memory() {
    let mut h = harness(&[
        0x885ffc40, // ldaxr w0, [x2]
        0x11000400, // add w0, w0, #1
        0x8801fc40, // stlxr w1, w0, [x2]
        0x35ffffa1, // cbnz w1, -12
        0xb8230044, // ldadd w3, w4, [x2]
        0xd28000c5, // mov x5, #6
        0xd2800126, // mov x6, #9
        0x9100204e, // add x14, x2, #8
        0xc8e5fdc6, // casal x5, x6, [x14]
        0xf94001c7, // ldr x7, [x14]
        0x91004048, // add x8, x2, #16
        0x486afd0c, // caspal x10, x11, x12, x13, [x8]
        0xd4200060, // brk #3
    ]);
    h.mem.write(DATA, 41u32).unwrap();
    h.mem.write(DATA + 8, 6u64).unwrap();
    h.mem.write(DATA + 16, 0x1111u64).unwrap();
    h.mem.write(DATA + 24, 0x2222u64).unwrap();
    h.ctx.x[2] = DATA;
    h.ctx.x[3] = 100;
    h.ctx.x[10] = 0x1111;
    h.ctx.x[11] = 0x2222;
    h.ctx.x[12] = 0xaaaa;
    h.ctx.x[13] = 0xbbbb;
    assert_eq!(h.core.run(&mut h.ctx), HaltReason::Breakpoint(3));
    assert_eq!(h.ctx.pc, CODE + 48);
    assert_eq!(h.ctx.x[1], 0, "store-exclusive succeeded");
    assert_eq!(h.ctx.x[4], 42, "LDADD returns the old value");
    assert_eq!(h.mem.read::<u32>(DATA).unwrap(), 142);
    assert_eq!(h.ctx.x[5], 6, "CAS returns the old value");
    assert_eq!(h.ctx.x[7], 9);
    assert_eq!((h.ctx.x[10], h.ctx.x[11]), (0x1111, 0x2222));
    assert_eq!(
        h.mem.read::<u128>(DATA + 16).unwrap(),
        0xbbbb << 64 | 0xaaaa
    );
}

#[test]
fn reads_system_registers_and_zeroes_cache_lines() {
    let mut h = harness(&[
        0xd53be020, // mrs x0, cntpct_el0
        0xd53b0021, // mrs x1, ctr_el0
        0xd53b00e2, // mrs x2, dczid_el0
        0xd53be003, // mrs x3, cntfrq_el0
        0xd51bd044, // msr tpidr_el0, x4
        0xd53bd045, // mrs x5, tpidr_el0
        0xd53bd066, // mrs x6, tpidrro_el0
        0xd50b7427, // dc zva, x7
        0xd51b4208, // msr nzcv, x8
        0xd53b4209, // mrs x9, nzcv
        0xd4000001, // svc #0
    ]);
    h.mem.fill(DATA, 0x100, 0xff).unwrap();
    h.ctx.x[4] = 0xdead_beef;
    h.ctx.x[7] = DATA + 0x47;
    h.ctx.x[8] = 0xa000_0000;
    h.ctx.tpidrro_el0 = 0x5555;
    assert_eq!(h.core.run(&mut h.ctx), HaltReason::SupervisorCall(0));
    assert_eq!(h.ctx.x[0], 123_456);
    assert_eq!(h.ctx.x[1], CpuConfig::DEFAULT_CTR_EL0 as u64);
    assert_eq!(h.ctx.x[2], 4);
    assert_eq!(h.ctx.x[3], 19_200_000);
    assert_eq!(
        (h.ctx.tpidr_el0, h.ctx.x[5], h.ctx.x[6]),
        (0xdead_beef, 0xdead_beef, 0x5555)
    );
    assert_eq!(h.ctx.x[9], 0xa000_0000);
    assert_eq!(h.mem.read::<u8>(DATA + 0x3f).unwrap(), 0xff);
    assert!((0x40..0x80).all(|i| h.mem.read::<u8>(DATA + i).unwrap() == 0));
    assert_eq!(h.mem.read::<u8>(DATA + 0x80).unwrap(), 0xff);
}

#[test]
fn vector_structure_loads_and_stores() {
    let mut h = harness(&[
        0x4cdfa800, // ld1 {v0.4s, v1.4s}, [x0], #32
        0x4c408822, // ld2 {v2.4s, v3.4s}, [x1]
        0x4d9f9042, // st1 {v2.s}[3], [x2], #4
        0x4d40c404, // ld1r {v4.8h}, [x0]
        0xad409825, // ldp q5, q6, [x1, #16]
        0xfc008c46, // str d6, [x2, #8]!
        0xd4000001, // svc #0
    ]);
    for i in 0..16u32 {
        h.mem.write(DATA + 4 * i as u64, i).unwrap();
    }
    h.ctx.x[0] = DATA;
    h.ctx.x[1] = DATA;
    h.ctx.x[2] = DATA + 0x100;
    assert_eq!(h.core.run(&mut h.ctx), HaltReason::SupervisorCall(0));
    let lanes = |v: u128| {
        [
            v as u32,
            (v >> 32) as u32,
            (v >> 64) as u32,
            (v >> 96) as u32,
        ]
    };
    assert_eq!(lanes(h.ctx.v[0]), [0, 1, 2, 3]);
    assert_eq!(lanes(h.ctx.v[1]), [4, 5, 6, 7]);
    assert_eq!(lanes(h.ctx.v[2]), [0, 2, 4, 6]);
    assert_eq!(lanes(h.ctx.v[3]), [1, 3, 5, 7]);
    assert_eq!(h.ctx.x[0], DATA + 32);
    assert_eq!(h.mem.read::<u32>(DATA + 0x100).unwrap(), 6);
    assert_eq!(h.ctx.v[4], 0x0008_0008_0008_0008_0008_0008_0008_0008);
    assert_eq!(lanes(h.ctx.v[6]), [8, 9, 10, 11]);
    assert_eq!(h.ctx.x[2], DATA + 0x10c);
    assert_eq!(h.mem.read::<u64>(DATA + 0x10c).unwrap(), 8 | 9 << 32);
}

#[test]
fn faults_leave_pc_at_the_instruction() {
    let mut h = harness(&[
        0xf9400020, // ldr x0, [x1]
        0x00000000, // udf #0
    ]);
    h.ctx.x[1] = 0x1234_5000;
    let pc = h.ctx.pc;
    assert_eq!(
        h.core.run(&mut h.ctx),
        HaltReason::Fault(CpuFault::Data {
            pc,
            addr: 0x1234_5000,
            write: false
        })
    );
    assert_eq!(h.ctx.pc, pc);
    h.ctx.pc += 4;
    assert_eq!(
        h.core.run(&mut h.ctx),
        HaltReason::Fault(CpuFault::Undefined {
            pc: pc + 4,
            instruction: 0
        })
    );
    h.ctx.pc = u64::MAX;
    assert_eq!(
        h.core.run(&mut h.ctx),
        HaltReason::Fault(CpuFault::Prefetch { pc: u64::MAX })
    );
    h.ctx.pc = 0x7000_0000;
    assert_eq!(
        h.core.run(&mut h.ctx),
        HaltReason::Fault(CpuFault::Prefetch { pc: 0x7000_0000 })
    );
}

#[test]
fn step_executes_one_instruction() {
    let mut h = harness(&[
        0xd2800021, // mov x1, #1
        0xd2800042, // mov x2, #2
        0xd4000041, // svc #2
    ]);
    assert_eq!(h.core.step(&mut h.ctx), HaltReason::StepComplete);
    assert_eq!((h.ctx.x[1], h.ctx.x[2], h.ctx.pc), (1, 0, CODE + 4));
    assert_eq!(h.core.step(&mut h.ctx), HaltReason::StepComplete);
    assert_eq!(h.core.step(&mut h.ctx), HaltReason::SupervisorCall(2));
    assert_eq!(h.ctx.pc, CODE + 12);
}

#[test]
fn interrupt_stops_a_spinning_core() {
    let mut h = harness(&[0x14000000]); // b .
    let handle = h.core.interrupt_handle();
    let signaller = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        handle.signal();
    });
    assert_eq!(h.core.run(&mut h.ctx), HaltReason::Interrupted);
    signaller.join().unwrap();
    assert_eq!(h.ctx.pc, CODE);
}

#[test]
fn invalidated_code_is_decoded_again() {
    let mut h = harness(&[
        0xd2800540, // mov x0, #42
        0xd4000041, // svc #2
    ]);
    assert_eq!(h.core.run(&mut h.ctx), HaltReason::SupervisorCall(2));
    assert_eq!(h.ctx.x[0], 42);
    h.mem.write(CODE, 0xd2800020u32).unwrap(); // mov x0, #1
    h.ctx.pc = CODE;
    h.core.run(&mut h.ctx);
    assert_eq!(h.ctx.x[0], 42, "the cached block still runs");
    h.core.invalidate_code(CODE + 4, 4);
    h.ctx.pc = CODE;
    h.core.run(&mut h.ctx);
    assert_eq!(h.ctx.x[0], 1);
}

#[test]
fn guest_icache_maintenance_invalidates_patched_code() {
    let target = CODE + 0x100;
    let mut h = harness(&[
        0xb9400120, // ldr w0, [x9]
        0xb9000140, // str w0, [x10]
        0xd50b752a, // ic ivau, x10
        0xd5033b9f, // dsb ish
        0xd5033fdf, // isb
        0xd61f0140, // br x10
    ]);
    h.mem.protect(CODE, 0x1000, MemoryPermission(7));
    h.mem.write(target, 0xd4000041u32).unwrap(); // svc #2
    h.mem.write(target + 4, 0xd4000061u32).unwrap(); // svc #3
    h.ctx.pc = target;
    assert_eq!(h.core.run(&mut h.ctx), HaltReason::SupervisorCall(2));
    h.mem.write(DATA, 0xd503201fu32).unwrap(); // nop
    h.ctx.x[9] = DATA;
    h.ctx.x[10] = target;
    h.ctx.pc = CODE;
    assert_eq!(h.core.run(&mut h.ctx), HaltReason::SupervisorCall(3));
}
#[test]
fn narrow_signed_atomics_compare_signed_values() {
    let mut h = harness(&[
        0x38214062, // ldsmaxb w1, w2, [x3]
        0x782450c5, // ldsminh w4, w5, [x6]
        0xb8274128, // ldsmax w7, w8, [x9]
        0xb827712a, // ldumin w7, w10, [x9]
        0xd4200000, // brk #0
    ]);
    h.mem.write(DATA, 0xfeu8).unwrap();
    h.mem.write(DATA + 2, 1u16).unwrap();
    h.mem.write(DATA + 4, 0x8000_0000u32).unwrap();
    h.ctx.x[3] = DATA;
    h.ctx.x[4] = 0xffff;
    h.ctx.x[6] = DATA + 2;
    h.ctx.x[7] = 1;
    h.ctx.x[9] = DATA + 4;
    assert_eq!(h.core.run(&mut h.ctx), HaltReason::Breakpoint(0));
    assert_eq!((h.ctx.x[2], h.mem.read::<u8>(DATA).unwrap()), (0xfe, 0));
    assert_eq!(
        (h.ctx.x[5], h.mem.read::<u16>(DATA + 2).unwrap()),
        (1, 0xffff)
    );
    assert_eq!(h.ctx.x[8], 0x8000_0000);
    assert_eq!((h.ctx.x[10], h.mem.read::<u32>(DATA + 4).unwrap()), (1, 1));
}

#[test]
fn misaligned_exclusives_and_atomics_fault() {
    for (word, addr) in [
        (0xc85f7c20u32, DATA + 4), // ldxr x0, [x1]
        (0xb8200022, DATA + 2),    // ldadd w0, w2, [x1]
        (0x48227c24, DATA + 8),    // casp x2, x3, x4, x5, [x1]
    ] {
        let mut h = harness(&[word]);
        h.ctx.x[1] = addr;
        assert_eq!(
            h.core.run(&mut h.ctx),
            HaltReason::Fault(CpuFault::Alignment { pc: CODE, addr }),
            "{word:#010x}"
        );
    }
}
