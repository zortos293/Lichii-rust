//! The JIT against the interpreter: random instruction streams run on both, with every pass
//! combination, and every register, the flags, the vector registers, the halt reason and the
//! data region must match.
//!
//! `LICHII_JIT_CASES` sets the number of streams (default 400).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lichii_common::settings::CpuAccuracy;
use lichii_memory::{AddressSpace, ExclusiveMonitor, MemoryPermission, PhysicalMemory};

use super::{Jit, Passes};
use crate::decoder::{Inst, decode};
use crate::{CpuConfig, CpuCore, GuestClock, HaltReason, Interpreter, ThreadContext, TscScale};

const CODE: u64 = 0x10000;
const CODE_SIZE: u64 = 0x4000;
const DATA: u64 = 0x100000;
const DATA_SIZE: u64 = 0x8000;
/// Read-only data right after `DATA`; the page after it is unmapped.
const RODATA: u64 = DATA + DATA_SIZE;
const BRK: u32 = 0xd420_0000;
const BASES: u32 = 20;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn interesting(&mut self) -> u64 {
        const VALUES: [u64; 10] = [
            0,
            1,
            u64::MAX,
            0x7fff_ffff,
            0x8000_0000,
            0xffff_ffff,
            0x1_0000_0000,
            0x7fff_ffff_ffff_ffff,
            0x8000_0000_0000_0000,
            63,
        ];
        match self.below(4) {
            0 => VALUES[self.below(VALUES.len() as u64) as usize],
            1 => self.below(64),
            _ => self.next(),
        }
    }

    fn pattern(&mut self, bits: &str) -> u32 {
        let random = self.next() as u32;
        bits.bytes().enumerate().fold(0, |w, (i, c)| {
            let bit = 1 << (31 - i);
            match c {
                b'1' => w | bit,
                b'0' => w,
                _ => w | (random & bit),
            }
        })
    }
}

const ALU: [&str; 17] = [
    "xxx10000xxxxxxxxxxxxxxxxxxxxxxxx",
    "xxx100010xxxxxxxxxxxxxxxxxxxxxxx",
    "xxx100100xxxxxxxxxxxxxxxxxxxxxxx",
    "xxx100101xxxxxxxxxxxxxxxxxxxxxxx",
    "xxx100110xxxxxxxxxxxxxxxxxxxxxxx",
    "xxx100111xxxxxxxxxxxxxxxxxxxxxxx",
    "xxx01011xx0xxxxxxxxxxxxxxxxxxxxx",
    "xxx01011xx1xxxxxxxxxxxxxxxxxxxxx",
    "xxx11010000xxxxx000000xxxxxxxxxx",
    "xxx01010xxxxxxxxxxxxxxxxxxxxxxxx",
    "xx111010010xxxxxxxxxx0xxxxx0xxxx",
    "xx011010100xxxxxxxxxxxxxxxxxxxxx",
    "x1011010110000000000xxxxxxxxxxxx",
    "x0011010110xxxxxxxxxxxxxxxxxxxxx",
    "xxx11011xxxxxxxxxxxxxxxxxxxxxxxx",
    "11010101000x1011010000100000xxxx",
    "1101010100x1101111010000010xxxxx",
];

fn alu(rng: &mut Rng) -> u32 {
    loop {
        let class = ALU[rng.below(ALU.len() as u64) as usize];
        let mut w = rng.pattern(class);
        if rng.below(8) != 0 && w >> 22 != 0b1101010100 {
            w &= !0x10;
        }
        if !matches!(decode(w), Inst::Undefined { .. }) {
            return w;
        }
    }
}

/// A load or store of one register or a pair, based on `X20`..`X23` or `SP`, with data
/// registers `X0`..`X15` or any vector register.
fn memory(rng: &mut Rng) -> u32 {
    loop {
        let base = if rng.below(6) == 0 {
            31
        } else {
            BASES + rng.below(4) as u32
        };
        let rt = rng.below(16) as u32;
        let rt2 = rng.below(16) as u32;
        let vector = rng.below(4) == 0;
        let w = match rng.below(4) {
            0 => {
                let imm = rng.below(32) as u32;
                rng.pattern("xx111x01xx000000000000xxxxxxxxxx") | imm << 10
            }
            1 => {
                let imm = (rng.below(64) as i32 - 32) as u32 & 0x1ff;
                rng.pattern("xx111x00xx0000000000xxxxxxxxxxxx") | imm << 12
            }
            2 => {
                let index = if rng.below(2) == 0 { 24 } else { 26 };
                rng.pattern("xx111x00xx1xxxxxxxxx10xxxxxxxxxx") & !(0x1f << 16) | index << 16
            }
            _ => {
                let imm = (rng.below(16) as i32 - 8) as u32 & 0x7f;
                rng.pattern("xx101x0xxx0000000xxxxxxxxxxxxxxx") & !(0x1f << 10)
                    | imm << 15
                    | rt2 << 10
            }
        };
        let w = (w & !(1 << 26)) | (vector as u32) << 26;
        let w = (w & !0x3ff) | base << 5 | rt;
        if !matches!(
            decode(w),
            Inst::Undefined { .. }
                | Inst::Nop
                | Inst::StoreTag { .. }
                | Inst::LoadExclusive { .. }
                | Inst::StoreExclusive { .. }
        ) {
            return w;
        }
    }
}

/// An LSE `LD<op>`/`SWP`, a `CAS` or a load/store-exclusive on `X20`..`X23`.
fn atomic(rng: &mut Rng) -> u32 {
    loop {
        let class = match rng.below(8) {
            0 | 1 => "xx0010001x1xxxxxx11111xxxxxxxxxx",
            2..=4 => "xx0010000xxxxxxxxxxxxxxxxxxxxxxx",
            _ => "xx111000xx1xxxxxxxxx00xxxxxxxxxx",
        };
        let base = BASES + rng.below(4) as u32;
        let w = (rng.pattern(class) & !0x3f0) | base << 5;
        if matches!(
            decode(w),
            Inst::AtomicRmw { .. }
                | Inst::Cas { .. }
                | Inst::Casp { .. }
                | Inst::LoadExclusive { .. }
                | Inst::StoreExclusive { .. }
        ) {
            return w;
        }
    }
}

const FP: [&str; 8] = [
    "000111100x10000xx10000xxxxxxxxxx",
    "000111100x1xxxxxxxxx11xxxxxxxxxx",
    "x00111100x10011x000000xxxxxxxxxx",
    "x0x11110xx1xxxxxxxxxxxxxxxxxxxxx",
    "x0x11111xxxxxxxxxxxxxxxxxxxxxxxx",
    "01x11110xxxxxxxxxxxxxxxxxxxxxxxx",
    "0xx01110xxxxxxxxxxxxxxxxxxxxxxxx",
    "1101010100x110110100010000xxxxxx",
];

/// A defined SIMD/FP data-processing instruction, or an `MRS`/`MSR` of FPCR or FPSR.
fn simd(rng: &mut Rng) -> u32 {
    loop {
        let class = FP[rng.below(FP.len() as u64) as usize];
        let w = rng.pattern(class);
        let defined = match decode(w) {
            Inst::SimdFp { word } => {
                crate::interpreter::simd::execute(&mut ThreadContext::default(), word)
                    == Some(Ok(()))
            }
            inst => !matches!(inst, Inst::Undefined { .. }),
        };
        if defined {
            return w;
        }
    }
}

/// A forward branch from `from` to `to` (word indices).
fn branch(rng: &mut Rng, from: usize, to: usize) -> u32 {
    let words = (to - from) as u32;
    let rt = rng.below(16) as u32;
    match rng.below(4) {
        0 => 0x5400_0000 | (words & 0x7ffff) << 5 | rng.below(16) as u32,
        1 => rng.pattern("x011010x000000000000000000000000") | (words & 0x7ffff) << 5 | rt,
        2 => rng.pattern("x011011xxxxxx0000000000000000000") | (words & 0x3fff) << 5 | rt,
        _ => 0x1400_0000 | (words & 0x3ff_ffff),
    }
}

fn stream(rng: &mut Rng) -> Vec<u32> {
    let len = 1 + rng.below(40) as usize;
    let mut kinds: Vec<u64> = (0..len).map(|_| rng.below(16)).collect();
    if rng.below(4) == 0 {
        kinds.iter_mut().for_each(|k| *k %= 8);
    }
    (0..len)
        .map(|i| match kinds[i] {
            0..=7 => alu(rng),
            8..=11 => memory(rng),
            12 | 15 => simd(rng),
            14 => atomic(rng),
            13 => {
                let to = i + 1 + rng.below((len - i) as u64) as usize;
                branch(rng, i, to)
            }
            _ => alu(rng),
        })
        .chain([BRK])
        .collect()
}

/// Counts reads, so every `CNTPCT` read returns a new value in program order.
#[derive(Default)]
struct Clock(AtomicU64);

impl GuestClock for Clock {
    fn ticks(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

#[derive(Clone, Copy)]
enum Backend {
    Interpreter,
    Jit(Passes),
    /// The JIT on an address space without a fastmem arena.
    SlowJit,
    /// The JIT as on a host without `cmpxchg16b`.
    NoCmpxchg16b,
}

struct Machine {
    mem: Arc<AddressSpace>,
    core: Box<dyn CpuCore>,
}

impl Machine {
    fn new(backend: Backend) -> Self {
        let physical = Arc::new(PhysicalMemory::new(1 << 20).unwrap());
        let mem = Arc::new(match backend {
            Backend::SlowJit => AddressSpace::new(physical, 39),
            _ => AddressSpace::with_fastmem(physical, 39).unwrap(),
        });
        mem.map(CODE, 0, CODE_SIZE, MemoryPermission::READ_EXECUTE);
        mem.map(DATA, 0x10000, DATA_SIZE, MemoryPermission::READ_WRITE);
        mem.map(RODATA, 0x20000, 0x1000, MemoryPermission::READ);
        let config = CpuConfig {
            memory: mem.clone(),
            monitor: Arc::new(ExclusiveMonitor::new(1)),
            clock: Arc::new(Clock::default()),
            core_index: 0,
            ctr_el0: CpuConfig::DEFAULT_CTR_EL0,
            dczid_el0: CpuConfig::DEFAULT_DCZID_EL0,
            accuracy: CpuAccuracy::Auto,
        };
        let core: Box<dyn CpuCore> = match backend {
            Backend::Interpreter => Box::new(Interpreter::new(config)),
            Backend::Jit(passes) => Box::new(Jit::with_passes(config, passes).unwrap()),
            Backend::SlowJit => Box::new(Jit::new(config).unwrap()),
            Backend::NoCmpxchg16b => {
                let mut jit = Jit::new(config).unwrap();
                jit.target.cmpxchg16b = false;
                Box::new(jit)
            }
        };
        Self { mem, core }
    }

    fn run(&mut self, code: &[u32], ctx: &ThreadContext, data: &[u8]) -> Outcome {
        let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.mem.write_bytes(CODE, &bytes).unwrap();
        self.mem.write_bytes(DATA, data).unwrap();
        self.core.clear_code_cache();
        let mut ctx = ctx.clone();
        let halt = self.core.run(&mut ctx);
        let mut data = vec![0; DATA_SIZE as usize];
        self.mem.read_bytes(DATA, &mut data).unwrap();
        Outcome { halt, ctx, data }
    }
}

#[derive(Debug, PartialEq)]
struct Outcome {
    halt: HaltReason,
    ctx: ThreadContext,
    data: Vec<u8>,
}

fn initial(rng: &mut Rng) -> ThreadContext {
    let mut ctx = ThreadContext {
        x: std::array::from_fn(|_| rng.interesting()),
        sp: DATA + 0x4000 + 16 * rng.below(64),
        pc: CODE,
        pstate: (rng.below(16) as u32) << 28,
        v: std::array::from_fn(|_| (rng.next() as u128) << 64 | rng.next() as u128),
        tpidr_el0: rng.next(),
        fpcr: [
            0,
            0,
            1 << 24,
            1 << 25,
            1 << 19,
            3 << 22,
            1 << 22 | 1 << 24,
            0x07c0_0000,
        ][rng.below(8) as usize],
        fpsr: if rng.below(4) == 0 { 0x0800_009f } else { 0 },
        ..Default::default()
    };
    let align = if rng.below(2) == 0 { !15 } else { !0 };
    for b in 0..4 {
        ctx.x[(BASES + b) as usize] = (DATA + 0x2000 + rng.below(0x4000)) & align;
    }
    if rng.below(16) == 0 {
        ctx.x[BASES as usize] = match rng.below(4) {
            0 => RODATA + rng.below(0x1000),
            1 => RODATA + 0x1000 - rng.below(8),
            2 => 1 << 39,
            _ => rng.next(),
        };
    }
    ctx.x[24] = rng.below(64);
    ctx.x[26] = rng.below(64).wrapping_neg();
    ctx
}

fn describe(code: &[u32], expected: &Outcome, actual: &Outcome) -> String {
    let words: Vec<String> = code.iter().map(|w| format!("{w:08x}")).collect();
    let mut lines = vec![words.join(" ")];
    if expected.halt != actual.halt {
        lines.push(format!(
            "halt: interpreter {:?}, jit {:?}",
            expected.halt, actual.halt
        ));
    }
    let (e, a) = (&expected.ctx, &actual.ctx);
    for i in 0..31 {
        if e.x[i] != a.x[i] {
            lines.push(format!(
                "x{i}: interpreter {:#x}, jit {:#x}",
                e.x[i], a.x[i]
            ));
        }
    }
    for (name, e, a) in [
        ("sp", e.sp, a.sp),
        ("pc", e.pc, a.pc),
        ("tpidr", e.tpidr_el0, a.tpidr_el0),
    ] {
        if e != a {
            lines.push(format!("{name}: interpreter {e:#x}, jit {a:#x}"));
        }
    }
    if e.pstate != a.pstate || e.fpcr != a.fpcr || e.fpsr != a.fpsr {
        lines.push(format!(
            "pstate/fpcr/fpsr: interpreter {:#x}/{:#x}/{:#x}, jit {:#x}/{:#x}/{:#x}",
            e.pstate, e.fpcr, e.fpsr, a.pstate, a.fpcr, a.fpsr
        ));
    }
    for i in 0..32 {
        if e.v[i] != a.v[i] {
            lines.push(format!(
                "v{i}: interpreter {:#x}, jit {:#x}",
                e.v[i], a.v[i]
            ));
        }
    }
    if let Some(i) = (0..expected.data.len()).find(|&i| expected.data[i] != actual.data[i]) {
        lines.push(format!("memory differs first at {:#x}", DATA + i as u64));
    }
    lines.join("\n")
}

const ABLATIONS: [Passes; 5] = [
    Passes {
        get_set: false,
        fold: false,
        fuse: false,
        dce: false,
    },
    Passes {
        get_set: false,
        ..Passes::ALL
    },
    Passes {
        fold: false,
        ..Passes::ALL
    },
    Passes {
        fuse: false,
        ..Passes::ALL
    },
    Passes {
        dce: false,
        ..Passes::ALL
    },
];

fn compare(machines: &mut [(&str, Machine)], code: &[u32], ctx: &ThreadContext, data: &[u8]) {
    let (reference, rest) = machines.split_first_mut().unwrap();
    let expected = reference.1.run(code, ctx, data);
    for (name, machine) in rest {
        let actual = machine.run(code, ctx, data);
        if actual != expected {
            panic!("{name}\n{}", describe(code, &expected, &actual));
        }
    }
}

#[test]
fn random_streams_match_the_interpreter() {
    let cases = std::env::var("LICHII_JIT_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let mut machines = vec![("interpreter", Machine::new(Backend::Interpreter))];
    machines.push(("jit", Machine::new(Backend::Jit(Passes::ALL))));
    machines.push(("jit without fastmem", Machine::new(Backend::SlowJit)));
    for (i, passes) in ABLATIONS.iter().enumerate() {
        let name = [
            "jit without passes",
            "jit without get/set",
            "jit without folding",
            "jit without fusion",
            "jit without dce",
        ][i];
        machines.push((name, Machine::new(Backend::Jit(*passes))));
    }
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut data = vec![0u8; DATA_SIZE as usize];
    for _ in 0..cases {
        let code = stream(&mut rng);
        for chunk in data.chunks_mut(8) {
            chunk.copy_from_slice(&rng.next().to_le_bytes());
        }
        let ctx = initial(&mut rng);
        compare(&mut machines, &code, &ctx, &data);
    }
}

fn directed(code: &[u32], regs: &[(usize, u64)]) {
    let mut ctx = ThreadContext {
        pc: CODE,
        sp: DATA + 0x4000,
        ..Default::default()
    };
    for &(r, value) in regs {
        ctx.x[r] = value;
    }
    let mut code = code.to_vec();
    code.push(BRK);
    let data = vec![0x5a; DATA_SIZE as usize];
    let mut machines = vec![
        ("interpreter", Machine::new(Backend::Interpreter)),
        ("jit", Machine::new(Backend::Jit(Passes::ALL))),
        ("jit without fastmem", Machine::new(Backend::SlowJit)),
        (
            "jit without cmpxchg16b",
            Machine::new(Backend::NoCmpxchg16b),
        ),
    ];
    compare(&mut machines, &code, &ctx, &data);
}

#[test]
fn fused_conditions_match_the_interpreter() {
    const VALUES: [u64; 5] = [0, 1, 0x8000_0000, 1 << 63, u64::MAX];
    let setters = [0xeb01_001f, 0xab01_001f, 0xea01_001f];
    for setter in setters.into_iter().flat_map(|s| [s, s & !(1 << 31)]) {
        let selects = (0..14).map(|cond| 0x9a84_0060 | cond << 12 | (2 + cond));
        let code: Vec<u32> = [setter].into_iter().chain(selects).collect();
        for (a, b) in VALUES.into_iter().flat_map(|a| VALUES.map(|b| (a, b))) {
            let regs = [(0, a), (1, b), (3, 3), (4, 4)];
            directed(&code, &regs);
            for cond in 0..14 {
                directed(&[setter, 0x5400_0040 | cond, 0xd280_0025], &regs);
            }
        }
    }
}

#[test]
fn fused_selects_release_their_compare_operands() {
    let code: Vec<u32> = (0..300)
        .flat_map(|i| [0x9100_0140 | i << 10, 0xeb01_001f, 0x9a83_0042])
        .collect();
    directed(&code, &[(1, 77), (3, 5), (10, 3)]);
}

#[test]
fn faults_leave_the_interpreters_state() {
    const MOV_X0_1: u32 = 0xd280_0020;
    const MOV_X0_2: u32 = 0xd280_0040;
    const LDR_X1_X2: u32 = 0xf940_0041;
    const STR_X1_X2: u32 = 0xf900_0041;
    const LDR_Q0_X2: u32 = 0x3dc0_0040;
    const STR_W1_X2: u32 = 0xb900_0041;
    for addr in [
        0x5000_0000,
        RODATA + 0x1000,
        RODATA + 0x1000 - 4,
        1 << 39,
        (1 << 39) - 4,
        u64::MAX - 3,
        CODE,
    ] {
        for access in [LDR_X1_X2, STR_X1_X2, LDR_Q0_X2, STR_W1_X2] {
            directed(&[MOV_X0_1, access, MOV_X0_2], &[(2, addr)]);
        }
    }
}

#[test]
fn simd_runs_mid_block_on_the_live_context() {
    const FMOV_D0_X1: u32 = 0x9e67_0020;
    const ADD_X5_X1_1: u32 = 0x9100_0425;
    const FADD_D1_D0_D0: u32 = 0x1e60_2801;
    const FCMP_D1_D0: u32 = 0x1e60_2020;
    const CSET_X2_GT: u32 = 0x9a9f_d7e2;
    const FMOV_X3_D1: u32 = 0x9e66_0023;
    const ADD_X4_X3_X2: u32 = 0x8b02_0064;
    const ADD_X6_X5_X5: u32 = 0x8b05_00a6;
    for value in [1.5f64, -2.0, f64::NAN] {
        directed(
            &[
                FMOV_D0_X1,
                ADD_X5_X1_1,
                FADD_D1_D0_D0,
                FCMP_D1_D0,
                CSET_X2_GT,
                FMOV_X3_D1,
                ADD_X4_X3_X2,
                ADD_X6_X5_X5,
            ],
            &[(1, value.to_bits())],
        );
    }
}

#[test]
fn undefined_simd_faults_mid_block() {
    const MOV_X0_1: u32 = 0xd280_0020;
    const MOV_X0_2: u32 = 0xd280_0040;
    const UNALLOCATED_MOVI: u32 = 0x6f00_fc00;
    directed(&[MOV_X0_1, UNALLOCATED_MOVI, MOV_X0_2], &[]);
}

#[test]
fn stores_to_read_only_pages_fault_through_the_handler() {
    const STRB_W1_X2: u32 = 0x3900_0041;
    const STP_X1_X3_X2: u32 = 0xa900_0c41;
    directed(&[STRB_W1_X2], &[(1, 7), (2, RODATA + 5)]);
    directed(&[STP_X1_X3_X2], &[(2, RODATA - 8)]);
}

#[test]
fn invalidation_retranslates_changed_code() {
    let mut machine = Machine::new(Backend::Jit(Passes::ALL));
    let ctx = ThreadContext {
        pc: CODE,
        ..Default::default()
    };
    let data = vec![0; DATA_SIZE as usize];
    let first = machine.run(&[0xd280_0020, BRK], &ctx, &data);
    assert_eq!(first.ctx.x[0], 1);
    machine.mem.write(CODE, 0xd280_0040u32).unwrap();
    let mut again = ctx.clone();
    machine.core.run(&mut again);
    assert_eq!(
        again.x[0], 1,
        "the stale block still runs before invalidation"
    );
    machine.core.invalidate_code(CODE, 4);
    let mut fresh = ctx.clone();
    machine.core.run(&mut fresh);
    assert_eq!(fresh.x[0], 2);
}

#[test]
fn register_pressure_spills_and_reloads() {
    let mut code = Vec::new();
    for i in 0..15u32 {
        code.push(0x9100_0400 | i << 5 | i);
    }
    code.push(0xf940_0281);
    for i in 0..15u32 {
        code.push(0x8b00_0000 | (14 - i) << 16 | i << 5 | (15 + i));
    }
    let regs: Vec<(usize, u64)> = (0..31)
        .map(|r| (r, 0x1111_0000_0000 * r as u64 + r as u64))
        .chain([(20, DATA + 0x100)])
        .collect();
    directed(&code, &regs);
}

#[test]
fn vector_pressure_spills_and_reloads() {
    let mut code = Vec::new();
    for i in 0..20u32 {
        code.push(0x3dc0_0280 | i << 10 | i);
    }
    for i in 0..20u32 {
        code.push(0x3d80_0280 | (i + 20) << 10 | (19 - i));
    }
    directed(&code, &[(20, DATA + 0x200)]);
}

#[test]
fn slow_paths_keep_live_registers_of_both_classes() {
    let mut code = Vec::new();
    for i in 0..9u32 {
        code.push(0x9100_0400 | i << 5 | i);
    }
    for i in 0..15u32 {
        code.push(0x3dc0_0280 | i << 10 | i);
    }
    for i in 0..9u32 {
        code.push(0x8b00_0000 | (8 - i) << 16 | i << 5 | (10 + i));
    }
    for i in 0..15u32 {
        code.push(0x3d80_0280 | (i + 20) << 10 | (14 - i));
    }
    let regs: Vec<(usize, u64)> = (0..20)
        .map(|r| (r, 0x0101_0101_0101 * r as u64 + 3))
        .chain([(20, DATA + 0x200)])
        .collect();
    directed(&code, &regs);
}

#[test]
fn zeroing_a_block_matches_the_interpreter() {
    const DC_ZVA_X2: u32 = 0xd50b_7422;
    for addr in [
        DATA + 0x100,
        DATA + 0x13f,
        RODATA + 0x40,
        RODATA + 0x1000,
        1 << 39,
        u64::MAX,
    ] {
        directed(&[0xd280_0020, DC_ZVA_X2, 0xd280_0040], &[(2, addr)]);
    }
}

#[test]
fn counter_reads_call_the_clock_in_order() {
    const MRS_X0_CNTPCT: u32 = 0xd53b_e020;
    const MRS_X1_CNTVCT: u32 = 0xd53b_e041;
    const LDR_X3_X20: u32 = 0xf940_0283;
    directed(
        &[MRS_X0_CNTPCT, LDR_X3_X20, MRS_X1_CNTVCT, 0x8b01_0002],
        &[(20, DATA + 0x80), (9, 7), (10, 8)],
    );
}

#[test]
fn atomics_fault_like_the_interpreter() {
    const LDADD_X1_X3_X20: u32 = 0xf821_0283;
    const CAS_W1_W3_X20: u32 = 0x88a1_7e83;
    const LDSMAXAB_W1_W3_X20: u32 = 0x38a1_4283;
    const SWPLH_W1_W3_X20: u32 = 0x7861_8283;
    for addr in [
        DATA + 0x100,
        DATA + 0x101,
        DATA + 0x104,
        RODATA + 0x10,
        RODATA + 0x1000,
        1 << 39,
    ] {
        for access in [
            LDADD_X1_X3_X20,
            CAS_W1_W3_X20,
            LDSMAXAB_W1_W3_X20,
            SWPLH_W1_W3_X20,
        ] {
            directed(&[access], &[(20, addr), (1, 0x8000_0081)]);
        }
    }
}

/// Keeps a value live in the first allocatable register (`rbx`) until `ADD_X10_X22_1`, so the
/// address of a `cmpxchg16b` between them lands elsewhere.
const MOV_X9_X22: u32 = 0xaa16_03e9;
const ADD_X10_X22_1: u32 = 0x9100_06ca;

#[test]
fn exclusives_match_the_interpreter() {
    const LDXR_X1_X20: u32 = 0xc85f_7e81;
    const LDAXRB_W1_X20: u32 = 0x085f_fe81;
    const LDXP_W1_W2_X20: u32 = 0x887f_0a81;
    const STXR_W3_X4_X20: u32 = 0xc803_7e84;
    const STLXRB_W3_W4_X20: u32 = 0x0803_fe84;
    const STXP_W3_W4_W5_X20: u32 = 0x8823_1684;
    const STXR_W3_X4_X21: u32 = 0xc803_7ea4;
    const STR_X6_X20: u32 = 0xf900_0286;
    const CLREX: u32 = 0xd503_3f5f;
    const LDXP_X1_X2_X20: u32 = 0xc87f_0a81;
    const LDAXP_X1_X2_X20: u32 = 0xc87f_8a81;
    const STXP_W3_X4_X5_X20: u32 = 0xc823_1684;
    const STLXP_W3_X4_X5_X20: u32 = 0xc823_9684;
    let sequences: &[&[u32]] = &[
        &[LDXR_X1_X20, STXR_W3_X4_X20],
        &[LDXR_X1_X20, STXR_W3_X4_X20, STXR_W3_X4_X20],
        &[STXR_W3_X4_X20],
        &[LDXR_X1_X20, STXR_W3_X4_X21],
        &[LDXR_X1_X20, STR_X6_X20, STXR_W3_X4_X20],
        &[LDXR_X1_X20, CLREX, STXR_W3_X4_X20],
        &[LDAXRB_W1_X20, STLXRB_W3_W4_X20],
        &[LDAXRB_W1_X20, STXR_W3_X4_X20],
        &[LDXP_W1_W2_X20, STXP_W3_W4_W5_X20],
        &[LDXP_W1_W2_X20, STXR_W3_X4_X20],
        &[LDXP_X1_X2_X20, STXP_W3_X4_X5_X20],
        &[LDAXP_X1_X2_X20, STLXP_W3_X4_X5_X20],
        &[LDXP_X1_X2_X20, STR_X6_X20, STXP_W3_X4_X5_X20],
        &[LDXP_X1_X2_X20, STXR_W3_X4_X20],
        &[LDXR_X1_X20, STXP_W3_X4_X5_X20],
        &[STXP_W3_X4_X5_X20],
        &[MOV_X9_X22, LDXP_X1_X2_X20, STXP_W3_X4_X5_X20, ADD_X10_X22_1],
    ];
    for code in sequences {
        for addr in [
            DATA + 0x100,
            DATA + 0x101,
            DATA + 0x104,
            DATA + 0x108,
            RODATA + 0x10,
            1 << 39,
        ] {
            directed(
                code,
                &[
                    (20, addr),
                    (21, addr + 8),
                    (4, 0x1234_5678_9abc_def0),
                    (5, 7),
                    (6, 9),
                ],
            );
        }
    }
}

#[test]
fn casp_matches_the_interpreter() {
    const CASP_W2_W6_X20: u32 = 0x0822_7e86;
    const CASPAL_W2_W6_X20: u32 = 0x0862_fe86;
    const CASP_X2_X6_X20: u32 = 0x4822_7e86;
    for casp in [CASP_W2_W6_X20, CASPAL_W2_W6_X20, CASP_X2_X6_X20] {
        for code in [&[casp][..], &[MOV_X9_X22, casp, ADD_X10_X22_1]] {
            for addr in [
                DATA + 0x100,
                DATA + 0x104,
                DATA + 0x108,
                RODATA + 0x10,
                1 << 39,
            ] {
                for expected in [0x5a5a_5a5a, 0x5a5a_5a5b] {
                    directed(
                        code,
                        &[
                            (20, addr),
                            (2, expected),
                            (3, 0x5a5a_5a5a),
                            (6, 0x1111),
                            (7, 0x2222),
                            (22, 3),
                        ],
                    );
                }
            }
        }
    }
}

/// Words of a loop that calls a leaf §X5§ times, counting in §X2§: the call, the return and the
/// backward branch all go through linked or table exits.
const CALL_LOOP: [u32; 8] = [
    0x9400_0004,
    0xf100_04a5,
    0x54ff_ffc1,
    BRK,
    0x9100_0442,
    0xb400_0042,
    0xd65f_03c0,
    0xd65f_03c0,
];

#[test]
fn linked_loops_match_the_interpreter() {
    directed(&CALL_LOOP[..3], &[(5, 1000)]);
    let mut machines = [
        ("interpreter", Machine::new(Backend::Interpreter)),
        ("jit", Machine::new(Backend::Jit(Passes::ALL))),
    ];
    let ctx = ThreadContext {
        pc: CODE,
        x: std::array::from_fn(|r| if r == 5 { 5000 } else { 0 }),
        ..Default::default()
    };
    compare(
        &mut machines,
        &CALL_LOOP,
        &ctx,
        &vec![0; DATA_SIZE as usize],
    );
}

#[test]
fn a_linked_infinite_loop_stops_on_interrupt() {
    let mut machine = Machine::new(Backend::Jit(Passes::ALL));
    machine.mem.write(CODE, 0x1400_0000u32).unwrap();
    let interrupt = machine.core.interrupt_handle();
    let signaller = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        interrupt.signal();
    });
    let mut ctx = ThreadContext {
        pc: CODE,
        ..Default::default()
    };
    assert_eq!(machine.core.run(&mut ctx), HaltReason::Interrupted);
    assert_eq!(ctx.pc, CODE);
    signaller.join().unwrap();
}

#[test]
fn invalidating_a_linked_block_unlinks_it() {
    const BL_LEAF: u32 = 0x9400_0003;
    const MOV_X1_1: u32 = 0xd280_0021;
    const MOV_X1_2: u32 = 0xd280_0041;
    const MOV_X0_1: u32 = 0xd280_0020;
    const MOV_X0_2: u32 = 0xd280_0040;
    const RET: u32 = 0xd65f_03c0;
    let mut machine = Machine::new(Backend::Jit(Passes::ALL));
    let code = [BL_LEAF, MOV_X1_1, BRK, MOV_X0_1, RET];
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    machine.mem.write_bytes(CODE, &bytes).unwrap();
    let run = |machine: &mut Machine| {
        let mut ctx = ThreadContext::default();
        for _ in 0..3 {
            ctx.pc = CODE;
            assert_eq!(machine.core.run(&mut ctx), HaltReason::Breakpoint(0));
        }
        (ctx.x[0], ctx.x[1])
    };
    assert_eq!(run(&mut machine), (1, 1));
    machine.mem.write(CODE + 12, MOV_X0_2).unwrap();
    machine.mem.write(CODE + 4, MOV_X1_2).unwrap();
    assert_eq!(
        run(&mut machine),
        (1, 1),
        "stale code runs until it is invalidated"
    );
    machine.core.invalidate_code(CODE + 12, 4);
    assert_eq!(
        run(&mut machine),
        (2, 1),
        "the linked call target is retranslated"
    );
    machine.core.invalidate_code(CODE + 4, 4);
    assert_eq!(
        run(&mut machine),
        (2, 2),
        "the return target leaves the indirect table"
    );
}

#[test]
fn an_indirect_jump_to_the_empty_table_key_faults() {
    const BR_X0: u32 = 0xd61f_0000;
    for backend in [Backend::Interpreter, Backend::Jit(Passes::ALL)] {
        let mut machine = Machine::new(backend);
        machine.mem.write(CODE, BR_X0).unwrap();
        let mut ctx = ThreadContext {
            pc: CODE,
            x: std::array::from_fn(|r| if r == 0 { u64::MAX } else { 0 }),
            ..Default::default()
        };
        assert_eq!(
            machine.core.run(&mut ctx),
            HaltReason::Fault(crate::CpuFault::Prefetch { pc: u64::MAX })
        );
    }
}

struct TscClock(TscScale);

impl GuestClock for TscClock {
    fn ticks(&self) -> u64 {
        // SAFETY: `rdtsc` has no preconditions on x86-64.
        self.0.ticks(unsafe { std::arch::x86_64::_rdtsc() })
    }

    fn host_tsc_scale(&self) -> Option<&TscScale> {
        Some(&self.0)
    }
}

#[test]
fn counter_reads_follow_the_tsc_scale() {
    const MRS_X0_CNTPCT: u32 = 0xd53b_e020;
    const MRS_X1_CNTVCT: u32 = 0xd53b_e041;
    for scale in [
        TscScale::new(0, 0, 12345),
        TscScale::new(3 << 32, 32, 1000),
        TscScale::new(0x1234_5678_9abc, 40, u64::MAX - 5),
    ] {
        let clock = Arc::new(TscClock(scale));
        let physical = Arc::new(PhysicalMemory::new(1 << 20).unwrap());
        let mem = Arc::new(AddressSpace::with_fastmem(physical, 39).unwrap());
        mem.map(CODE, 0, CODE_SIZE, MemoryPermission::READ_EXECUTE);
        for (i, word) in [MRS_X0_CNTPCT, MRS_X1_CNTVCT, BRK].iter().enumerate() {
            mem.write(CODE + 4 * i as u64, *word).unwrap();
        }
        let mut jit = Jit::new(CpuConfig {
            memory: mem,
            monitor: Arc::new(ExclusiveMonitor::new(1)),
            clock: clock.clone(),
            core_index: 0,
            ctr_el0: CpuConfig::DEFAULT_CTR_EL0,
            dczid_el0: CpuConfig::DEFAULT_DCZID_EL0,
            accuracy: CpuAccuracy::Auto,
        })
        .unwrap();
        assert!(jit.target.tsc_scale.is_some());
        let mut ctx = ThreadContext {
            pc: CODE,
            ..Default::default()
        };
        let before = clock.ticks();
        assert_eq!(jit.run(&mut ctx), HaltReason::Breakpoint(0));
        let after = clock.ticks();
        let within = |t: u64| t.wrapping_sub(before) <= after.wrapping_sub(before);
        assert!(
            within(ctx.x[0]) && within(ctx.x[1]),
            "{before} {:?} {after}",
            &ctx.x[..2]
        );
        assert!(ctx.x[1].wrapping_sub(ctx.x[0]) <= after.wrapping_sub(before));
    }
}

#[test]
fn paranoid_accuracy_builds_no_jit() {
    let physical = Arc::new(PhysicalMemory::new(1 << 20).unwrap());
    let config = |accuracy| CpuConfig {
        memory: Arc::new(AddressSpace::with_fastmem(physical.clone(), 39).unwrap()),
        monitor: Arc::new(ExclusiveMonitor::new(1)),
        clock: Arc::new(Clock::default()),
        core_index: 0,
        ctr_el0: CpuConfig::DEFAULT_CTR_EL0,
        dczid_el0: CpuConfig::DEFAULT_DCZID_EL0,
        accuracy,
    };
    assert!(Jit::new(config(CpuAccuracy::Paranoid)).is_none());
    for accuracy in [
        CpuAccuracy::Auto,
        CpuAccuracy::Accurate,
        CpuAccuracy::Unsafe,
    ] {
        assert!(Jit::new(config(accuracy)).is_some());
    }
}
