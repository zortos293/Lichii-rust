//! The JIT against the interpreter: random instruction streams run on both, with every pass
//! combination, and every register, the flags, the vector registers, the halt reason and the
//! data region must match.
//!
//! `LICHII_JIT_CASES` sets the number of streams (default 400).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use lichii_common::settings::CpuAccuracy;
use lichii_memory::{
    AddressSpace, DeviceAddressSpace, ExclusiveMonitor, MemoryPermission, PhysicalMemory,
};

use super::{Jit, Passes, simd};
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

    /// A vector register value: random bits, S or D lanes holding small or edge-case values, or
    /// words from a small set so lanes of every size compare equal, differ in one half, and sit
    /// on the signed and unsigned boundaries.
    fn vector(&mut self) -> u128 {
        const WORDS: [u32; 6] = [0, 1, u32::MAX, 0x8000_0000, 0x7fff_ffff, 0xffff];
        match self.below(6) {
            0 => (self.next() as u128) << 64 | self.next() as u128,
            1..=3 => {
                let double = self.below(2) == 0;
                let (lanes, bits) = if double { (2, 64) } else { (4, 32) };
                (0..lanes).fold(0, |v, i| {
                    let small = (self.below(4096) as f32 - 2048.0) / 16.0;
                    let lane = match (double, self.below(4) == 0) {
                        (false, true) => FP_S[self.below(FP_S.len() as u64) as usize] as u64,
                        (false, false) => small.to_bits() as u64,
                        (true, true) => FP_D[self.below(FP_D.len() as u64) as usize],
                        (true, false) => (small as f64).to_bits(),
                    };
                    v | (lane as u128) << (bits * i)
                })
            }
            _ => (0..4).fold(0, |v, i| {
                let word = match self.below(8) {
                    0..=5 => WORDS[self.below(WORDS.len() as u64) as usize],
                    _ => self.next() as u32,
                };
                v | (word as u128) << (32 * i)
            }),
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

/// S values where host FP and the Arm ARM part ways: zeros, infinities, quiet and signalling NaNs,
/// denormals, the largest finite value, the smallest normal and the value just below 1.0 (whose product with it is tiny
/// before rounding but not after), and the integer conversion limits 2^31 and 2^32.
const FP_S: [u32; 16] = [
    0x7f7f_ffff,
    0,
    0x8000_0000,
    0x7f80_0000,
    0xff80_0000,
    0xffc0_0000,
    0x7fc0_0001,
    0x7f80_0001,
    0x0000_0001,
    0x807f_ffff,
    0x0080_0000,
    0x3f80_0000,
    0x3f7f_ffff,
    0x4f00_0000,
    0x4f80_0000,
    0xbf00_0000,
];

const FP_D: [u64; 16] = [
    0x7fef_ffff_ffff_ffff,
    0,
    0x8000_0000_0000_0000,
    0x7ff0_0000_0000_0000,
    0xfff0_0000_0000_0000,
    0xfff8_0000_0000_0000,
    0x7ff8_0000_0000_0001,
    0x7ff0_0000_0000_0001,
    1,
    0x800f_ffff_ffff_ffff,
    0x0010_0000_0000_0000,
    0x3ff0_0000_0000_0000,
    0x3fef_ffff_ffff_ffff,
    0x43e0_0000_0000_0000,
    0x41f0_0000_0000_0000,
    0xbfe0_0000_0000_0000,
];

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

const FP: [&str; 25] = [
    "000111100x1xxxxx01xx10xxxxxxxxxx",
    "0xx01110xx1xxxxx111001xxxxxxxxxx",
    "0x001110xx1xxxxx111111xxxxxxxxxx",
    "0xx011100x100001110110xxxxxxxxxx",
    "0x101110x01xxxxx110101xxxxxxxxxx",
    "000111100x1xxxxxxxxx01xxxxxxxxxx",
    "0001111011100010x10000xxxxxxxxxx",
    "01x111111xxxxxxxxx01x0xxxxxxxxxx",
    "01x111100x100001110110xxxxxxxxxx",
    "0xx01110xx1xxxxx110xx1xxxxxxxxxx",
    "0xx011111xxxxxxxxx01x0xxxxxxxxxx",
    "000111100x10000xx10000xxxxxxxxxx",
    "000111100x10001xx10000xxxxxxxxxx",
    "000111100x1xxxxx00xx10xxxxxxxxxx",
    "000111110xxxxxxxxxxxxxxxxxxxxxxx",
    "000111100x1xxxxx001000xxxxxxx000",
    "x00111100x1xx0xx000000xxxxxxxxxx",
    "000111100x1xxxxxxxx10000000xxxxx",
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

const VECTOR: [&str; 14] = [
    "0x001110xx100001010010xxxxxxxxxx",
    "01011110000xxxxx000001xxxxxxxxxx",
    "0x101110000xxxxx0xxxx0xxxxxxxxxx",
    "0xx01110xx1xxxxx100001xxxxxxxxxx",
    "0xx01110xx1xxxxx000111xxxxxxxxxx",
    "0xx01110xx1xxxxx0011x1xxxxxxxxxx",
    "0xx01110xx1xxxxx100011xxxxxxxxxx",
    "0xx01110xx1xxxxx100111xxxxxxxxxx",
    "0xx01110xx100000100x10xxxxxxxxxx",
    "0xx011110xxxxxxx0x0x01xxxxxxxxxx",
    "0xx01110000xxxxx0xxxx1xxxxxxxxxx",
    "0xx0111100000xxxxxxx01xxxxxxxxxx",
    "0x001110xx1xxxxx101111xxxxxxxxxx",
    "0x00111000100000010110xxxxxxxxxx",
];

/// An Advanced SIMD integer three-same, compare against zero, `CNT`, shift by immediate, copy or
/// modified-immediate instruction, or an `LD1`..`LD4`/`ST1`..`ST4` of one to four registers on `X20`..`X23` or
/// `SP`, offset or post-indexed by an immediate or `X24`/`X26`.
fn vector(rng: &mut Rng) -> u32 {
    if rng.below(3) == 0 {
        let base = if rng.below(6) == 0 {
            31
        } else {
            BASES + rng.below(4) as u32
        };
        let opcode =
            [0b0111, 0b1010, 0b0110, 0b0010, 0b1000, 0b0100, 0b0000][rng.below(7) as usize];
        let post = rng.below(2) as u32;
        let rm = post * [31, 24, 26][rng.below(3) as usize];
        if rng.below(2) == 0 {
            loop {
                let w = (rng.next() as u32 & (1 << 30 | 1 << 22 | 1 << 21 | 0x3f << 10 | 0x1f))
                    | 0x0d00_0000
                    | post << 23
                    | rm << 16
                    | base << 5;
                if matches!(
                    decode(w),
                    Inst::VecSingle { .. } | Inst::VecReplicate { .. }
                ) {
                    return w;
                }
            }
        }
        return (rng.next() as u32 & (1 << 30 | 1 << 22 | 3 << 10 | 0x1f))
            | 0x0c00_0000
            | post << 23
            | rm << 16
            | opcode << 12
            | base << 5;
    }
    loop {
        let class = VECTOR[rng.below(VECTOR.len() as u64) as usize];
        let mut w = rng.pattern(class);
        if rng.below(8) != 0 {
            w &= !0x10;
        }
        if crate::interpreter::simd::execute(&mut ThreadContext::default(), w) == Some(Ok(())) {
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
            12 => simd(rng),
            15 => vector(rng),
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
    /// The JIT as on a host without FMA3.
    NoFma,
    /// The JIT as on a host with SSE2 but neither SSE4.1 nor SSE4.2.
    NoSse4,
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
            Backend::NoFma => {
                let mut jit = Jit::new(config).unwrap();
                jit.target.fma = false;
                Box::new(jit)
            }
            Backend::NoSse4 => {
                let mut jit = Jit::new(config).unwrap();
                jit.target.sse41 = false;
                jit.target.sse42 = false;
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
        let mut data = vec![0; (DATA_SIZE + 0x1000) as usize];
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
        v: std::array::from_fn(|_| rng.vector()),
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
        fpsr: [0, 0x10, 0x10, 0x0800_009f][rng.below(4) as usize],
        ..Default::default()
    };
    let align = if rng.below(2) == 0 { !15 } else { !0 };
    for b in 0..4 {
        ctx.x[(BASES + b) as usize] = (DATA + 0x2000 + rng.below(0x4000)) & align;
    }
    if rng.below(16) == 0 {
        ctx.x[BASES as usize] = match rng.below(5) {
            0 => RODATA + rng.below(0x1000),
            1 => RODATA + 0x1000 - rng.below(8),
            2 => RODATA + 0x1000 - rng.below(72),
            3 => 1 << 39,
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

const ABLATIONS: [Passes; 6] = [
    Passes {
        get_set: false,
        fold: false,
        fuse: false,
        lazy: false,
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
        lazy: false,
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
    machines.push(("jit without fma", Machine::new(Backend::NoFma)));
    machines.push(("jit without sse4", Machine::new(Backend::NoSse4)));
    for (i, passes) in ABLATIONS.iter().enumerate() {
        let name = [
            "jit without passes",
            "jit without get/set",
            "jit without folding",
            "jit without fusion",
            "jit without lazy flags",
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
        ("jit without sse4", Machine::new(Backend::NoSse4)),
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
fn vector_structure_faults_leave_the_interpreters_state() {
    const MOV_X0_1: u32 = 0xd280_0020;
    const MOV_X0_2: u32 = 0xd280_0040;
    const LD1_16B_X4_X2_POST: u32 = 0x4cdf_2040;
    const ST1_16B_X4_X2_POST: u32 = 0x4c9f_2040;
    const LD1_2D_X2_X2: u32 = 0x4c40_ac40;
    const ST1_2D_X2_X2: u32 = 0x4c00_ac40;
    const LD1_8B_X3_X2_POST_X3: u32 = 0x0cc3_6040;
    const ST1_8B_X3_X2_POST_X3: u32 = 0x0c83_6040;
    const LD4_S1_X2_POST: u32 = 0x0dff_b040;
    const ST4_S1_X2_POST: u32 = 0x0dbf_b040;
    const LD3R_8H_X2: u32 = 0x4d40_e440;
    const LD2_D1_X2_POST_X3: u32 = 0x4de3_8441;
    const ST3_B13_X2_POST_X3: u32 = 0x4d83_3441;
    const LD4R_2D_X2_POST: u32 = 0x4dff_ec41;
    const LD4_4S_X2_POST: u32 = 0x4cdf_0840;
    const ST4_4S_X2_POST: u32 = 0x4c9f_0840;
    const LD3_8B_X2_POST_X3: u32 = 0x0cc3_4041;
    const ST2_8H_X2_POST_X3: u32 = 0x4c83_8441;
    const ST3_16B_X2: u32 = 0x4c00_4041;
    const LD2_2D_X2: u32 = 0x4c40_8c41;
    let movi = |rd: u32, imm: u32| 0x4f00_e400 | (imm >> 5) << 16 | (imm & 31) << 5 | rd;
    for addr in [
        RODATA + 0x1000 - 1,
        RODATA + 0x1000 - 16,
        RODATA + 0x1000 - 24,
        RODATA + 0x1000 - 40,
        RODATA + 0x1000 - 63,
        RODATA - 8,
        RODATA - 24,
        RODATA - 40,
        DATA - 8,
        (1 << 39) - 32,
        u64::MAX - 15,
    ] {
        for access in [
            LD1_16B_X4_X2_POST,
            ST1_16B_X4_X2_POST,
            LD1_2D_X2_X2,
            ST1_2D_X2_X2,
            LD1_8B_X3_X2_POST_X3,
            ST1_8B_X3_X2_POST_X3,
            LD4_S1_X2_POST,
            ST4_S1_X2_POST,
            LD3R_8H_X2,
            LD2_D1_X2_POST_X3,
            ST3_B13_X2_POST_X3,
            LD4R_2D_X2_POST,
            LD4_4S_X2_POST,
            ST4_4S_X2_POST,
            LD3_8B_X2_POST_X3,
            ST2_8H_X2_POST_X3,
            ST3_16B_X2,
            LD2_2D_X2,
        ] {
            let code = [
                movi(0, 0x11),
                movi(1, 0x22),
                movi(2, 0x33),
                movi(3, 0x44),
                MOV_X0_1,
                access,
                MOV_X0_2,
            ];
            directed(&code, &[(2, addr), (3, 8)]);
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

/// Every natively lowered FP instruction on every pair of edge-case inputs, under the FPCR
/// settings that change results (flush to zero, default NaN and a directed rounding mode), with
/// FPSR.IXC both clear and already set.
#[test]
fn fp_edge_cases_match_the_interpreter() {
    const OPS: [u32; 26] = [
        0x1e22_4820, // fmax s0, s1, s2
        0x1e22_5820, // fmin s0, s1, s2
        0x1e22_6820, // fmaxnm s0, s1, s2
        0x1e22_7820, // fminnm s0, s1, s2
        0x1e22_2820, // fadd s0, s1, s2
        0x1e22_3820, // fsub s0, s1, s2
        0x1e22_0820, // fmul s0, s1, s2
        0x1e22_1820, // fdiv s0, s1, s2
        0x1f02_0c20, // fmadd s0, s1, s2, s3
        0x1f02_8c20, // fmsub s0, s1, s2, s3
        0x1f22_0c20, // fnmadd s0, s1, s2, s3
        0x1f22_8c20, // fnmsub s0, s1, s2, s3
        0x1e21_c020, // fsqrt s0, s1
        0x1e22_c020, // fcvt d0, s1
        0x1e22_2020, // fcmp s1, s2
        0x1e22_2030, // fcmpe s1, s2
        0x1e20_2028, // fcmp s1, #0.0
        0x1e38_0020, // fcvtzs w0, s1
        0x1e39_0020, // fcvtzu w0, s1
        0x9e38_0020, // fcvtzs x0, s1
        0x9e39_0020, // fcvtzu x0, s1
        0x1e22_8820, // fnmul s0, s1, s2
        0x1e22_0425, // fccmp s1, s2, #5, eq
        0x1e22_1425, // fccmp s1, s2, #5, ne
        0x1e22_0435, // fccmpe s1, s2, #5, eq
        0x1e22_1435, // fccmpe s1, s2, #5, ne
    ];
    let machines = &mut [
        ("interpreter", Machine::new(Backend::Interpreter)),
        ("jit", Machine::new(Backend::Jit(Passes::ALL))),
        ("jit without fma", Machine::new(Backend::NoFma)),
    ];
    let data = vec![0u8; DATA_SIZE as usize];
    let fpcrs = [0, 1 << 24, 1 << 25, 3 << 22];
    let s_values: Vec<u64> = FP_S.iter().map(|&s| s as u64).collect();
    for (double, values) in [(false, s_values.as_slice()), (true, FP_D.as_slice())] {
        for op in OPS {
            let op = match (double, op) {
                (false, op) => op,
                (true, 0x1e22_c020) => 0x1e62_4020,
                (true, op) => op | 1 << 22,
            };
            for (i, &a) in values.iter().enumerate() {
                for (j, &b) in values.iter().enumerate() {
                    for (fpcr, fpsr) in fpcrs.into_iter().flat_map(|c| [(c, 0), (c, 0x10)]) {
                        let mut ctx = ThreadContext {
                            pc: CODE,
                            fpcr,
                            fpsr,
                            ..Default::default()
                        };
                        ctx.v[1] = a as u128;
                        ctx.v[2] = b as u128;
                        ctx.v[3] = values[(i * 7 + j) % values.len()] as u128;
                        compare(machines, &[op, BRK], &ctx, &data);
                    }
                }
            }
        }
    }
    let integers = [
        0,
        1,
        u64::MAX,
        1 << 63,
        (1 << 63) - 1,
        0xffff_ffff,
        0x8000_0001,
        1 << 53 | 1,
    ];
    for x in integers {
        for op in [
            0x1e22_0020,
            0x1e23_0020,
            0x9e22_0020,
            0x9e23_0020,
            0x5e21_d820,
            0x7e21_d820,
        ] {
            for (fp, fpsr) in [(0, 0), (0, 0x10), (1 << 22, 0), (1 << 22, 0x10)] {
                let mut ctx = ThreadContext {
                    pc: CODE,
                    fpsr,
                    ..Default::default()
                };
                ctx.x[1] = x;
                ctx.v[1] = 0x5a5a_5a5a_5a5a_5a5a << 64 | x as u128;
                compare(machines, &[op | fp, BRK], &ctx, &data);
            }
        }
    }
}

/// Every exponent of H, with mantissas at the ends, the middle and the quiet bit, converted to S and
/// D under the FPCR controls that matter: AHP, DN, FZ and FZ16.
#[test]
fn half_conversions_match_the_interpreter() {
    const FCVT_S0_H1: u32 = 0x1ee2_4020;
    const FCVT_D0_H1: u32 = 0x1ee2_c020;
    let machines = &mut [
        ("interpreter", Machine::new(Backend::Interpreter)),
        ("jit", Machine::new(Backend::Jit(Passes::ALL))),
    ];
    let data = vec![0u8; DATA_SIZE as usize];
    for sign in [0, 0x8000] {
        for exponent in 0..32 {
            for mantissa in [0, 1, 0x155, 0x200, 0x3ff] {
                let half = sign | exponent << 10 | mantissa;
                for fpcr in [0, 1 << 26, 1 << 25, 1 << 24, 1 << 19, 3 << 22] {
                    for op in [FCVT_S0_H1, FCVT_D0_H1] {
                        let mut ctx = ThreadContext {
                            pc: CODE,
                            fpcr,
                            ..Default::default()
                        };
                        ctx.v[1] = 0x5a5a_5a5a_5a5a_5a5a_5a5a_5a5a_5a5a << 16 | half as u128;
                        compare(machines, &[op, BRK], &ctx, &data);
                    }
                }
            }
        }
    }
}

/// FCCMP reads NZCV: from a flag op earlier in its block, and lazily from the previous block,
/// with operands that compare natively and NaNs that take the slow path.
#[test]
fn conditional_compares_read_the_live_flags() {
    const SUBS_X0_X1_X2: u32 = 0xeb02_0020;
    const B_NEXT: u32 = 0x1400_0001;
    const FCCMP_S1_S2_5_EQ: u32 = 0x1e22_0425;
    const FCCMPE_D1_D2_5_NE: u32 = 0x1e62_1435;
    let machines = &mut [
        ("interpreter", Machine::new(Backend::Interpreter)),
        ("jit", Machine::new(Backend::Jit(Passes::ALL))),
        ("jit without fastmem", Machine::new(Backend::SlowJit)),
    ];
    let data = vec![0u8; DATA_SIZE as usize];
    let cases: [(u32, [u64; 3], u64); 2] = [
        (
            FCCMP_S1_S2_5_EQ,
            [0x3f80_0000, 0x7fc0_0000, 0x7fa0_0000],
            0x4000_0000,
        ),
        (
            FCCMPE_D1_D2_5_NE,
            [0x3ff0 << 48, 0x7ff8 << 48, 0x7ff4 << 48],
            0x4000 << 48,
        ),
    ];
    for (fccmp, values, b) in cases {
        for code in [
            [SUBS_X0_X1_X2, fccmp, BRK, BRK],
            [SUBS_X0_X1_X2, B_NEXT, fccmp, BRK],
        ] {
            for x2 in [7, 8] {
                for a in values {
                    let mut ctx = ThreadContext {
                        pc: CODE,
                        ..Default::default()
                    };
                    ctx.x[1] = 7;
                    ctx.x[2] = x2;
                    ctx.v[1] = a as u128;
                    ctx.v[2] = b as u128;
                    compare(machines, &code, &ctx, &data);
                }
            }
        }
    }
}

#[test]
fn vector_fp_edge_cases_match_the_interpreter() {
    const S_OPS: [u32; 28] = [
        0x4e22_e420, // fcmeq v0.4s, v1.4s, v2.4s
        0x6e22_e420, // fcmge v0.4s, v1.4s, v2.4s
        0x6ea2_e420, // fcmgt v0.4s, v1.4s, v2.4s
        0x0e22_e420, // fcmeq v0.2s, v1.2s, v2.2s
        0x2e22_e420, // fcmge v0.2s, v1.2s, v2.2s
        0x2ea2_e420, // fcmgt v0.2s, v1.2s, v2.2s
        0x4e22_fc20, // frecps v0.4s, v1.4s, v2.4s
        0x4ea2_fc20, // frsqrts v0.4s, v1.4s, v2.4s
        0x0e22_fc20, // frecps v0.2s, v1.2s, v2.2s
        0x0ea2_fc20, // frsqrts v0.2s, v1.2s, v2.2s
        0x4e21_d820, // scvtf v0.4s, v1.4s
        0x0e21_d820, // scvtf v0.2s, v1.2s
        0x6e22_d420, // faddp v0.4s, v1.4s, v2.4s
        0x2e22_d420, // faddp v0.2s, v1.2s, v2.2s
        0x4e22_d420, // fadd v0.4s, v1.4s, v2.4s
        0x4ea2_d420, // fsub v0.4s, v1.4s, v2.4s
        0x6e22_dc20, // fmul v0.4s, v1.4s, v2.4s
        0x4e22_cc20, // fmla v0.4s, v1.4s, v2.4s
        0x4ea2_cc20, // fmls v0.4s, v1.4s, v2.4s
        0x4fa2_9820, // fmul v0.4s, v1.4s, v2.s[3]
        0x4fa2_1020, // fmla v0.4s, v1.4s, v2.s[1]
        0x4f82_5820, // fmls v0.4s, v1.4s, v2.s[2]
        0x0e22_d420, // fadd v0.2s, v1.2s, v2.2s
        0x0f82_1020, // fmla v0.2s, v1.2s, v2.s[0]
        0x0fa2_9820, // fmul v0.2s, v1.2s, v2.s[3]
        0x5fa2_9820, // fmul s0, s1, v2.s[3]
        0x5fa2_1020, // fmla s0, s1, v2.s[1]
        0x5f82_5820, // fmls s0, s1, v2.s[2]
    ];
    const D_OPS: [u32; 15] = [
        0x4e62_e420, // fcmeq v0.2d, v1.2d, v2.2d
        0x6e62_e420, // fcmge v0.2d, v1.2d, v2.2d
        0x6ee2_e420, // fcmgt v0.2d, v1.2d, v2.2d
        0x4e62_fc20, // frecps v0.2d, v1.2d, v2.2d
        0x4ee2_fc20, // frsqrts v0.2d, v1.2d, v2.2d
        0x6e62_d420, // faddp v0.2d, v1.2d, v2.2d
        0x4e62_d420, // fadd v0.2d, v1.2d, v2.2d
        0x6e62_dc20, // fmul v0.2d, v1.2d, v2.2d
        0x4e62_cc20, // fmla v0.2d, v1.2d, v2.2d
        0x4ee2_cc20, // fmls v0.2d, v1.2d, v2.2d
        0x4fc2_9820, // fmul v0.2d, v1.2d, v2.d[1]
        0x4fc2_1020, // fmla v0.2d, v1.2d, v2.d[0]
        0x5fc2_9820, // fmul d0, d1, v2.d[1]
        0x5fc2_1020, // fmla d0, d1, v2.d[0]
        0x5fc2_5820, // fmls d0, d1, v2.d[1]
    ];
    let machines = &mut [
        ("interpreter", Machine::new(Backend::Interpreter)),
        ("jit", Machine::new(Backend::Jit(Passes::ALL))),
        ("jit without fma", Machine::new(Backend::NoFma)),
    ];
    let data = vec![0u8; DATA_SIZE as usize];
    let fpcrs = [0, 1 << 24, 1 << 25, 3 << 22];
    let s_values: Vec<u64> = FP_S.iter().map(|&s| s as u64).collect();
    for (ops, values, bits) in [
        (S_OPS.as_slice(), s_values.as_slice(), 32),
        (&D_OPS, FP_D.as_slice(), 64),
    ] {
        let len = values.len();
        let vector = |f: &dyn Fn(usize) -> usize| {
            (0..128 / bits).fold(0u128, |v, k| v | (values[f(k) % len] as u128) << (bits * k))
        };
        for &op in ops {
            for i in 0..len {
                for j in 0..len {
                    for (fpcr, fpsr) in fpcrs.into_iter().flat_map(|c| [(c, 0), (c, 0x10)]) {
                        let mut ctx = ThreadContext {
                            pc: CODE,
                            fpcr,
                            fpsr,
                            ..Default::default()
                        };
                        ctx.v[0] = vector(&|k| i * 7 + j + k);
                        ctx.v[1] = vector(&|k| i + k);
                        ctx.v[2] = vector(&|k| j + 5 * k);
                        compare(machines, &[op, BRK], &ctx, &data);
                    }
                }
            }
        }
    }
}

/// SQXTN and SQXTN2 on lanes at and around the saturation bounds, with FPSR.QC clear and set.
#[test]
fn saturating_narrow_matches_the_interpreter() {
    const OPS: [u32; 4] = [
        0x0e21_4820, // sqxtn v0.8b, v1.8h
        0x4e21_4820, // sqxtn2 v0.16b, v1.8h
        0x0e61_4820, // sqxtn v0.4h, v1.4s
        0x4e61_4820, // sqxtn2 v0.8h, v1.4s
    ];
    const H: [u16; 8] = [0, 1, 0xffff, 0x7f, 0x80, 0xff80, 0xff7f, 0x8000];
    const S: [u32; 8] = [
        0,
        1,
        0xffff_ffff,
        0x7fff,
        0x8000,
        0xffff_8000,
        0xffff_7fff,
        0x8000_0000,
    ];
    let machines = &mut [
        ("interpreter", Machine::new(Backend::Interpreter)),
        ("jit", Machine::new(Backend::Jit(Passes::ALL))),
    ];
    let data = vec![0u8; DATA_SIZE as usize];
    for op in OPS {
        for start in 0..8 {
            for stride in 0..8 {
                let lane = |k: usize| (start + stride * k) % 8;
                let v1 = if op & 1 << 22 == 0 {
                    (0..8).fold(0u128, |v, k| v | (H[lane(k)] as u128) << (16 * k))
                } else {
                    (0..4).fold(0u128, |v, k| v | (S[lane(k)] as u128) << (32 * k))
                };
                for fpsr in [0, 1 << 27] {
                    let mut ctx = ThreadContext {
                        pc: CODE,
                        fpsr,
                        ..Default::default()
                    };
                    ctx.v[0] = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
                    ctx.v[1] = v1;
                    compare(machines, &[op, BRK], &ctx, &data);
                }
            }
        }
    }
}

/// Enough live results to put destinations in xmm8..xmm15, which the hand-encoded
/// FMLA/FMLS .2D reach through VEX.R.
#[test]
fn vector_fp_uses_high_registers() {
    const STREAM: [u32; 17] = [
        0x4e62_cc30,
        0x4ee4_cc71,
        0x4e66_ccb2,
        0x4ee8_ccf3,
        0x4e6a_cd34,
        0x4eec_cd75,
        0x4e6e_cdb6,
        0x4ee1_cdf7,
        0x4e63_cc58,
        0x4ee5_cc99,
        0x4e67_ccda,
        0x4ee9_cd1b,
        0x4e6b_cd5c,
        0x4eed_cd9d,
        0x4e6f_cdde,
        0x4ee9_cc3f,
        BRK,
    ];
    let machines = &mut [
        ("interpreter", Machine::new(Backend::Interpreter)),
        ("jit", Machine::new(Backend::Jit(Passes::ALL))),
    ];
    let mut ctx = ThreadContext {
        pc: CODE,
        fpsr: 0x10,
        ..Default::default()
    };
    for (r, v) in ctx.v.iter_mut().enumerate() {
        let x = (r as f64 + 1.0) * 0.375;
        *v = ((-x).to_bits() as u128) << 64 | x.to_bits() as u128;
    }
    compare(machines, &STREAM, &ctx, &vec![0u8; DATA_SIZE as usize]);
}

/// A by-element product that underflows to zero must fall back, for UFC, even where the same
/// lane of the element register is zero.
#[test]
fn vector_fp_underflow_to_zero_falls_back() {
    const FMUL_V0_V1_V2_S3: u32 = 0x4fa2_9820;
    let machines = &mut [
        ("interpreter", Machine::new(Backend::Interpreter)),
        ("jit", Machine::new(Backend::Jit(Passes::ALL))),
    ];
    let mut ctx = ThreadContext {
        pc: CODE,
        fpsr: 0x10,
        ..Default::default()
    };
    ctx.v[1] = 0x0000_0001_0000_0001_0000_0001_0000_0001;
    ctx.v[2] = 1 << 96;
    compare(
        machines,
        &[FMUL_V0_V1_V2_S3, BRK],
        &ctx,
        &vec![0u8; DATA_SIZE as usize],
    );
}

/// Inexact native ops rely on FPSR.IXC being set already: after an FPSR write clears it, or while
/// a saturating conversion falls back, the next op must still report IXC exactly.
#[test]
fn ixc_follows_fpsr_writes() {
    const FCVTZU_W0_D1: u32 = 0x1e79_0020;
    const FADD_S0_S1_S2: u32 = 0x1e22_2820;
    const FADD_S3_S4_S4: u32 = 0x1e24_2883;
    const MSR_FPSR_XZR: u32 = 0xd51b_443f;
    let machines = &mut [
        ("interpreter", Machine::new(Backend::Interpreter)),
        ("jit", Machine::new(Backend::Jit(Passes::ALL))),
    ];
    let data = vec![0u8; DATA_SIZE as usize];
    for (code, fpcr, a, b) in [
        (
            [FCVTZU_W0_D1, FADD_S3_S4_S4, BRK].as_slice(),
            0,
            0x41f0_0000_0008_0000,
            0,
        ),
        (
            &[FADD_S0_S1_S2, MSR_FPSR_XZR, FADD_S3_S4_S4, BRK],
            0,
            0x3f80_0000,
            0x3080_0000,
        ),
    ] {
        let mut ctx = ThreadContext {
            pc: CODE,
            fpcr,
            ..Default::default()
        };
        ctx.v[1] = a;
        ctx.v[2] = b;
        ctx.v[4] = 0x3f80_0000;
        compare(machines, code, &ctx, &data);
    }
}

/// Native FP must not inherit the caller's rounding, flush-to-zero or denormals-are-zero
/// controls, and the caller gets its MXCSR back.
#[test]
fn host_mxcsr_does_not_leak_into_guest_fp() {
    const FADD_S0_S1_S2: u32 = 0x1e22_2820;
    const FMUL_D3_D4_D5: u32 = 0x1e65_0883;
    const FADD_S6_S7_S7: u32 = 0x1e27_28e6;
    const FADD_S8_S9_S7: u32 = 0x1e27_2928;
    const ROUND_UP_FTZ_DAZ: u32 = 0x1f80 | 2 << 13 | 1 << 15 | 1 << 6;
    let mut ctx = ThreadContext {
        pc: CODE,
        fpsr: 0x10,
        ..Default::default()
    };
    ctx.v[1] = 0x3f80_0000;
    ctx.v[2] = 0x3380_0000;
    ctx.v[4] = 1;
    ctx.v[5] = 0x4330_0000_0000_0000;
    ctx.v[7] = 1;
    ctx.v[9] = 0x0080_0000;
    let code = [
        FADD_S0_S1_S2,
        FMUL_D3_D4_D5,
        FADD_S6_S7_S7,
        FADD_S8_S9_S7,
        BRK,
    ];
    let data = vec![0u8; DATA_SIZE as usize];
    let mut interpreter = Machine::new(Backend::Interpreter);
    let mut jit = Machine::new(Backend::Jit(Passes::ALL));
    let expected = interpreter.run(&code, &ctx, &data);
    let saved = simd::mxcsr();
    simd::set_mxcsr(ROUND_UP_FTZ_DAZ);
    let actual = jit.run(&code, &ctx, &data);
    let after = simd::mxcsr();
    simd::set_mxcsr(saved);
    assert_eq!(actual.ctx.v[0], expected.ctx.v[0]);
    assert_eq!(actual.ctx.v[3], expected.ctx.v[3]);
    assert_eq!(actual.ctx.v[6], expected.ctx.v[6]);
    assert_eq!(actual.ctx.v[8], expected.ctx.v[8]);
    assert_eq!(actual.ctx.fpsr, expected.ctx.fpsr);
    assert_eq!(after, ROUND_UP_FTZ_DAZ);
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
fn counter_reads_never_step_back_when_the_tsc_lags() {
    const MRS_X0_CNTPCT: u32 = 0xd53b_e020;
    let clock = Arc::new(TscClock(TscScale::new(1 << 31, 32, 1 << 40)));
    let physical = Arc::new(PhysicalMemory::new(1 << 20).unwrap());
    let mem = Arc::new(AddressSpace::with_fastmem(physical, 39).unwrap());
    mem.map(CODE, 0, CODE_SIZE, MemoryPermission::READ_EXECUTE);
    mem.write(CODE, MRS_X0_CNTPCT).unwrap();
    mem.write(CODE + 4, BRK).unwrap();
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
    let mut read = || {
        let mut ctx = ThreadContext {
            pc: CODE,
            ..Default::default()
        };
        assert_eq!(jit.run(&mut ctx), HaltReason::Breakpoint(0));
        ctx.x[0]
    };
    let ahead = read();
    clock.0.offset.fetch_sub(1 << 30, Ordering::Relaxed);
    let lagging = read();
    assert!(
        lagging >= ahead,
        "the counter went back from {ahead} to {lagging}"
    );
    clock.0.offset.fetch_add(1 << 31, Ordering::Relaxed);
    assert!(
        read() > ahead + (1 << 29),
        "the floor holds reads back only until the counter passes it"
    );
}

#[test]
fn icache_maintenance_reaches_the_other_cores() {
    const IC_IVAU_X10: u32 = 0xd50b_752a;
    const IC_IALLU: u32 = 0xd508_751f;
    const MOV_X0_1: u32 = 0xd280_0020;
    const MOV_X0_2: u32 = 0xd280_0040;
    const MOV_X0_3: u32 = 0xd280_0060;
    let target = CODE + 0x100;
    let physical = Arc::new(PhysicalMemory::new(1 << 20).unwrap());
    let mem = Arc::new(AddressSpace::with_fastmem(physical, 39).unwrap());
    mem.map(CODE, 0, CODE_SIZE, MemoryPermission::READ_EXECUTE);
    let monitor = Arc::new(ExclusiveMonitor::new(2));
    let mut cores: Vec<Jit> = (0..2)
        .map(|core_index| {
            Jit::new(CpuConfig {
                memory: mem.clone(),
                monitor: monitor.clone(),
                clock: Arc::new(Clock::default()),
                core_index,
                ctr_el0: CpuConfig::DEFAULT_CTR_EL0,
                dczid_el0: CpuConfig::DEFAULT_DCZID_EL0,
                accuracy: CpuAccuracy::Auto,
            })
            .unwrap()
        })
        .collect();
    for (i, word) in [IC_IVAU_X10, BRK, IC_IALLU, BRK].into_iter().enumerate() {
        mem.write(CODE + 4 * i as u64, word).unwrap();
    }
    mem.write(target + 4, BRK).unwrap();
    let run_target = |core: &mut Jit| {
        let mut ctx = ThreadContext {
            pc: target,
            ..Default::default()
        };
        assert_eq!(core.run(&mut ctx), HaltReason::Breakpoint(0));
        ctx.x[0]
    };
    let maintain = |core: &mut Jit, pc: u64| {
        let mut ctx = ThreadContext {
            pc,
            ..Default::default()
        };
        ctx.x[10] = target + 8;
        let halt = core.run(&mut ctx);
        assert_eq!(ctx.pc, pc + 4);
        assert_eq!(core.run(&mut ctx), HaltReason::Breakpoint(0));
        halt
    };
    let [first, second] = &mut cores[..] else {
        unreachable!()
    };
    mem.write(target, MOV_X0_1).unwrap();
    assert_eq!(run_target(first), 1);
    for (pc, mov, x0, range) in [
        (CODE, MOV_X0_2, 2, (target, 64)),
        (CODE + 8, MOV_X0_3, 3, (0, u64::MAX)),
    ] {
        mem.write(target, mov).unwrap();
        let halt = maintain(second, pc);
        assert_eq!(
            halt,
            HaltReason::InvalidateCode {
                addr: range.0,
                size: range.1
            }
        );
        let before = run_target(first);
        assert_ne!(before, x0, "the other core still runs the old block");
        first.invalidate_code(range.0, range.1);
        assert_eq!(run_target(first), x0);
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

#[test]
fn lazy_flags_cross_linked_blocks() {
    const SETTERS: [u32; 6] = [
        0xeb01_001f,
        0xab01_001f,
        0xea01_001f,
        0x6b01_001f,
        0x2b01_001f,
        0x6a01_001f,
    ];
    const VALUES: [u64; 4] = [0, 1, 0x8000_0000, u64::MAX];
    for setter in SETTERS {
        for cond in 0..14 {
            let code = [
                0x5400_0040 | cond,
                0x9100_0442,
                setter,
                0x8b07_0000,
                0xd100_0529,
                0xb5ff_ff69,
            ];
            for (a, b) in VALUES.into_iter().flat_map(|a| VALUES.map(|b| (a, b))) {
                for step in [1, 0x7fff_ffff_ffff_ffff] {
                    directed(&code, &[(0, a), (1, b), (7, step), (9, 4)]);
                }
            }
        }
    }
    const MSR_LOOP: [u32; 8] = [
        0xf100_0400,
        0x1400_0001,
        0xd51b_4203,
        0x1400_0001,
        0xd53b_4204,
        0x8b04_00a5,
        0xd100_0529,
        0xb5ff_ff29,
    ];
    for nzcv in [0, 0x6000_0000, 0xf000_0000] {
        directed(&MSR_LOOP, &[(0, 2), (3, nzcv), (9, 4)]);
    }
    const FCCMP_LOOP: [u32; 8] = [
        0xeb01_0000,
        0x1400_0001,
        0x1e61_0403,
        0x1400_0001,
        0xd53b_4204,
        0x8b04_00a5,
        0xd100_0529,
        0xb5ff_ff29,
    ];
    for (a, b) in [(5, 5), (5, 3), (3, 5)] {
        directed(&FCCMP_LOOP, &[(0, a), (1, b), (9, 4)]);
    }
}

#[test]
fn guest_stores_to_read_only_pages_fault_on_every_backend() {
    const STR_X1_X2: u32 = 0xf900_0041;
    const STADD_W1_X2: u32 = 0xb821_005f;
    const DC_ZVA_X2: u32 = 0xd50b_7422;
    let backends = [
        ("interpreter", Backend::Interpreter),
        ("jit", Backend::Jit(Passes::ALL)),
        ("jit without fastmem", Backend::SlowJit),
    ];
    let data = vec![0; DATA_SIZE as usize];
    for (name, backend) in backends {
        let mut machine = Machine::new(backend);
        for store in [STR_X1_X2, STADD_W1_X2, DC_ZVA_X2] {
            let mut ctx = ThreadContext {
                pc: CODE,
                ..Default::default()
            };
            ctx.x[1] = 7;
            ctx.x[2] = RODATA;
            let outcome = machine.run(&[store, BRK], &ctx, &data);
            assert_eq!(
                outcome.halt,
                HaltReason::Fault(crate::CpuFault::Data {
                    pc: CODE,
                    addr: RODATA,
                    write: true
                }),
                "{name}: {store:#010x}"
            );
            assert_eq!(
                machine.mem.read::<u64>(RODATA).unwrap(),
                0,
                "{name}: {store:#010x}"
            );
        }
    }
}

/// Runs `program` from `ctx` on a JIT without SSE4.1 and returns its halt, its final context and,
/// with `profile`, its profile table.
fn run_profiled(
    program: &[u32],
    mut ctx: ThreadContext,
    profile: bool,
) -> (HaltReason, ThreadContext, Option<String>) {
    let physical = Arc::new(PhysicalMemory::new(1 << 20).unwrap());
    let memory = Arc::new(AddressSpace::with_fastmem(physical, 39).unwrap());
    memory.map(CODE, 0, CODE_SIZE, MemoryPermission::READ_EXECUTE);
    memory.map(DATA, 0x10000, DATA_SIZE, MemoryPermission::READ_WRITE);
    let bytes: Vec<u8> = program.iter().flat_map(|w| w.to_le_bytes()).collect();
    memory.write_bytes(CODE, &bytes).unwrap();
    let config = CpuConfig {
        memory,
        monitor: Arc::new(ExclusiveMonitor::new(1)),
        clock: Arc::new(Clock::default()),
        core_index: 0,
        ctr_el0: CpuConfig::DEFAULT_CTR_EL0,
        dczid_el0: CpuConfig::DEFAULT_DCZID_EL0,
        accuracy: CpuAccuracy::Auto,
    };
    let mut jit = Jit::build(config, Passes::ALL, profile).unwrap();
    jit.target.sse41 = false;
    jit.clear_code_cache();
    let halt = jit.run(&mut ctx);
    let table = jit
        .state
        .profile_counts()
        .map(|p| p.table(1, Duration::ZERO));
    (halt, ctx, table)
}

/// A scalar product, fused product or quotient whose result is zero because a factor or the
/// dividend is zero stays native; an underflow to zero still falls back, and an inexact op with
/// FPSR.IXC clear is counted apart from it.
#[test]
fn zero_factor_products_stay_native() {
    const PROGRAM: [u32; 9] = [
        0x1e22_0820, // fmul s0, s1, s2
        0x1f02_1023, // fmadd s3, s1, s2, s4
        0x1e22_1825, // fdiv s5, s1, s2
        0x1e21_884b, // fnmul s11, s2, s1
        0x1e6d_098a, // fmul d10, d12, d13
        0x1e27_08e6, // fmul s6, s7, s7
        0xd51b_443f, // msr fpsr, xzr
        0x1e29_0848, // fmul s8, s2, s9
        BRK,
    ];
    let mut ctx = ThreadContext {
        pc: CODE,
        fpsr: 0x10,
        ..Default::default()
    };
    ctx.v[1] = 0x8000_0000;
    ctx.v[2] = 3.0f32.to_bits() as u128;
    ctx.v[4] = 0x8000_0000;
    ctx.v[7] = 2.0f32.powi(-100).to_bits() as u128;
    ctx.v[9] = 0.1f32.to_bits() as u128;
    ctx.v[12] = (-0.0f64).to_bits() as u128;
    ctx.v[13] = 3.0f64.to_bits() as u128;
    let machines = &mut [
        ("interpreter", Machine::new(Backend::Interpreter)),
        ("jit", Machine::new(Backend::Jit(Passes::ALL))),
    ];
    compare(machines, &PROGRAM, &ctx, &vec![0u8; DATA_SIZE as usize]);
    let table = run_profiled(&PROGRAM, ctx, true).2.unwrap();
    let fused_slow = u32::from(!std::arch::is_x86_feature_detected!("fma"));
    for line in [
        "native FP slow paths, FPSR.IXC clear: 1\n".to_string(),
        format!(
            "native FP slow paths, inputs or result: {}\n",
            1 + fused_slow
        ),
        "FP 2-source 0x1e200800".to_string(),
    ] {
        assert!(table.contains(&line), "{line:?} in\n{table}");
    }
}

#[test]
fn profile_counts_without_changing_results() {
    const PROGRAM: [u32; 10] = [
        0xeb01_001f,
        0x1400_0001,
        0x4e28_4820,
        0x1e62_2820,
        0xd100_0529,
        0xb5ff_ff69,
        0xd503_201f,
        0xd61f_0080,
        0x4c40_8040,
        BRK,
    ];
    let run = |profile: bool| {
        let mut ctx = ThreadContext {
            pc: CODE,
            fpcr: 1 << 24,
            ..Default::default()
        };
        ctx.x[0] = 3;
        ctx.x[1] = 5;
        ctx.x[2] = DATA;
        ctx.x[4] = CODE + 0x20;
        ctx.x[9] = 4;
        run_profiled(&PROGRAM, ctx, profile)
    };
    let (halt, ctx, table) = run(true);
    let (plain_halt, plain_ctx, plain_table) = run(false);
    assert_eq!((halt, &ctx), (plain_halt, &plain_ctx));
    assert!(plain_table.is_none());
    let table = table.unwrap();
    for line in [
        "translation: 3 blocks, 8 instructions,",
        "fault 0",
        "indirect-branch miss 1,",
        "code cache flushes: region full 0, clear 1, invalidate range 0",
        "interpreter fallbacks: 2\n",
        "LD2 (multiple structures)",
        "Brk",
        "CallSimd calls: 4\n",
        "AES 0x4e284800",
        "native FP slow paths, FPCR mode: 4\n",
        "FP 2-source 0x1e602800",
    ] {
        assert!(table.contains(line), "{line:?} in\n{table}");
    }
    let count = |label: &str| -> u64 {
        let rest = &table[table.find(label).unwrap() + label.len()..];
        rest[..rest.find(|c: char| !c.is_ascii_digit()).unwrap()]
            .parse()
            .unwrap()
    };
    assert!(count("unlinked jump ") > 0, "{table}");
    assert!(count("write-backs: native ") > 0, "{table}");
    assert!(count(", dispatcher ") > 0, "{table}");
}

#[test]
fn fp_selects_consume_fused_compares() {
    const SETTERS: [u32; 6] = [
        0xeb01_001f,
        0xab01_001f,
        0xea01_001f,
        0x6b01_001f,
        0x2b01_001f,
        0x6a01_001f,
    ];
    const VALUES: [u64; 4] = [0, 1, 0x8000_0000, u64::MAX];
    const FMOV_D1_X3: u32 = 0x9e67_0061;
    const FMOV_D2_X4: u32 = 0x9e67_0082;
    for setter in SETTERS {
        for cond in 0..16 {
            for fcsel in [0x1e22_0c20, 0x1e62_0c20] {
                let code = [FMOV_D1_X3, FMOV_D2_X4, setter, fcsel | cond << 12];
                for (a, b) in VALUES.into_iter().flat_map(|a| VALUES.map(|b| (a, b))) {
                    directed(
                        &code,
                        &[
                            (0, a),
                            (1, b),
                            (3, 0x3ff0_0000_0000_0001),
                            (4, 0x4000_0000_0000_0002),
                        ],
                    );
                }
            }
        }
    }
}

#[test]
fn cnt_and_addp_match_the_interpreter() {
    let addp = (0..8)
        .filter(|&a: &u32| a != 6)
        .map(|a| 0x0e20_bc00 | (a & 1) << 30 | (a >> 1) << 22);
    let words: Vec<u32> = [0x0e20_5800, 0x4e20_5800]
        .into_iter()
        .chain(addp)
        .flat_map(|w| {
            let rm = |r: u32| if w & 0x400 != 0 { r << 16 } else { 0 };
            [
                w | rm(3) | 2 << 5 | 1,
                w | rm(1) | 1 << 5 | 1,
                w | rm(4) | 4 << 5 | 5,
            ]
        })
        .collect();
    assert!(words.iter().all(|&w| {
        crate::interpreter::simd::execute(&mut ThreadContext::default(), w) == Some(Ok(()))
    }));
    let mut machines = vec![
        ("interpreter", Machine::new(Backend::Interpreter)),
        ("jit", Machine::new(Backend::Jit(Passes::ALL))),
        ("jit without fastmem", Machine::new(Backend::SlowJit)),
        ("jit without sse4", Machine::new(Backend::NoSse4)),
    ];
    let mut rng = Rng(0xc0ff_ee00_add0);
    let data = vec![0u8; DATA_SIZE as usize];
    for _ in 0..200 {
        let ctx = initial(&mut rng);
        for &w in &words {
            compare(&mut machines, &[w, BRK], &ctx, &data);
        }
        compare(
            &mut machines,
            &[0x0e20_5800, 0x0e20_bc00, 0x0e60_bc00, 0x0ea0_bc00, BRK],
            &ctx,
            &data,
        );
    }
}

fn crc(castagnoli: bool, size: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    let sf = if size == 3 { 0x8000_0000 } else { 0 };
    sf | 0x1ac0_4000 | rm << 16 | u32::from(castagnoli) << 12 | size << 10 | rn << 5 | rd
}

#[test]
fn crc32_matches_the_interpreter() {
    assert_eq!(crc(false, 0, 2, 2, 3), 0x1ac3_4042);
    assert_eq!(crc(true, 3, 2, 2, 3), 0x9ac3_5c42);
    let code = [
        crc(false, 0, 2, 2, 3),
        crc(true, 1, 4, 4, 3),
        crc(false, 2, 5, 2, 6),
        crc(true, 3, 7, 5, 3),
        crc(false, 3, 8, 8, 9),
        crc(true, 0, 10, 4, 9),
        crc(false, 1, 11, 11, 6),
        crc(true, 2, 12, 7, 12),
        crc(false, 3, 3, 3, 3),
        crc(true, 2, 31, 9, 31),
        0xd29b_ddcd,
        crc(false, 3, 14, 14, 13),
        crc(true, 3, 15, 15, 13),
        0x528b_4b50,
        crc(false, 0, 17, 16, 3),
        crc(true, 1, 18, 16, 6),
        crc(false, 2, 19, 16, 13),
    ];
    let mut rng = Rng(0x0c3c_3200_1eed_c0de);
    for _ in 0..300 {
        let regs: Vec<(usize, u64)> = (0..20).map(|r| (r, rng.interesting())).collect();
        directed(&code, &regs);
    }
}

#[test]
fn stores_to_gpu_cached_pages_land_and_are_reported() {
    const STR_X1_X2: u32 = 0xf900_0041;
    const STR_X1_X2_8: u32 = 0xf900_0441;
    const STP_X1_X3_X2: u32 = 0xa900_0c41;
    const LDADD_X1_X0_X2: u32 = 0xf821_0040;
    const LDXR_X0_X2: u32 = 0xc85f_7c40;
    const STXR_W4_X1_X2: u32 = 0xc804_7c41;
    const DC_ZVA_X2: u32 = 0xd50b_7422;
    const ST1_V0_X2: u32 = 0x4c00_7040;
    const DADDR: u64 = 0x1_0000_0000;
    const STR_Q0_X2: u32 = 0x3d80_0040;
    const ST2_V0_V1_X2: u32 = 0x4c00_8040;
    const SWP_X1_X0_X2: u32 = 0xf821_8040;
    const STLR_X1_X2: u32 = 0xc89f_fc41;
    let cases: [(&str, &[u32]); 10] = [
        ("str", &[STR_X1_X2]),
        ("str q", &[STR_Q0_X2]),
        ("st2", &[ST2_V0_V1_X2]),
        ("swp", &[SWP_X1_X0_X2]),
        ("stlr", &[STLR_X1_X2]),
        ("stp", &[STP_X1_X3_X2]),
        ("ldadd", &[LDADD_X1_X0_X2]),
        ("ldxr/stxr", &[LDXR_X0_X2, STXR_W4_X1_X2]),
        ("dc zva", &[DC_ZVA_X2]),
        ("st1", &[ST1_V0_X2]),
    ];
    let run = |backend: Backend, code: &[u32]| {
        let mut machine = Machine::new(backend);
        let device = DeviceAddressSpace::new(machine.mem.physical().clone(), 36);
        device.attach_process(machine.mem.clone());
        device
            .map_process(DADDR, &machine.mem, DATA, DATA_SIZE)
            .unwrap();
        let mut words = code.to_vec();
        words.push(BRK);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        machine.mem.write_bytes(CODE, &bytes).unwrap();
        machine.mem.write_bytes(DATA, &[0x5a; 0x1000]).unwrap();
        machine.core.clear_code_cache();
        let mut reported = Vec::new();
        let cached = device.cache_pages(DADDR, 0x1000);
        let mut ctx = ThreadContext {
            pc: CODE,
            ..Default::default()
        };
        ctx.x[1] = 0x1111_2222_3333_4444;
        ctx.x[2] = DATA + 0x100;
        ctx.x[3] = 0x5555_6666_7777_8888;
        let halt = machine.core.run(&mut ctx);
        device.take_cpu_writes(|daddr, size| reported.push((daddr, size)));

        let again = [STR_X1_X2_8, BRK];
        let bytes: Vec<u8> = again.iter().flat_map(|w| w.to_le_bytes()).collect();
        machine.mem.write_bytes(CODE, &bytes).unwrap();
        machine.core.clear_code_cache();
        let mut second = ctx.clone();
        second.pc = CODE;
        machine.core.run(&mut second);
        let mut later = Vec::new();
        device.take_cpu_writes(|daddr, size| later.push((daddr, size)));
        drop(cached);

        let mut data = vec![0; 0x1000];
        machine.mem.read_bytes(DATA, &mut data).unwrap();
        (halt, ctx, data, reported, later)
    };
    for (what, code) in cases {
        let expected = run(Backend::Interpreter, code);
        assert_eq!(expected.3, [(DADDR, 0x1000)], "interpreter {what}");
        assert!(
            expected.4.is_empty(),
            "interpreter {what}: the page stays untracked"
        );
        for (name, backend) in [
            ("jit", Backend::Jit(Passes::ALL)),
            ("jit without fastmem", Backend::SlowJit),
            ("jit without cmpxchg16b", Backend::NoCmpxchg16b),
        ] {
            let actual = run(backend, code);
            assert_eq!(actual.0, expected.0, "{name} {what}: halt");
            assert_eq!(actual.1, expected.1, "{name} {what}: registers");
            assert!(actual.2 == expected.2, "{name} {what}: memory");
            assert_eq!(actual.3, [(DADDR, 0x1000)], "{name} {what}: reported");
            assert!(
                actual.4.is_empty(),
                "{name} {what}: the page stays untracked"
            );
        }
    }
}
