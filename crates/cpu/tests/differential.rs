//! Differential tests against Unicorn (QEMU's A64 translator): random instruction streams per
//! encoding class run on both, then every general register, SP, NZCV, every vector register and
//! the whole data region are compared. Both the interpreter and the JIT run every stream.
//!
//! `LICHII_DIFF_CASES` sets the number of streams per class (default 300).

use std::sync::Arc;

use lichii_common::settings::{CpuAccuracy, CpuBackend};
use lichii_cpu::decoder::{Inst, decode};
use lichii_cpu::{CpuConfig, CpuCore, GuestClock, HaltReason, ThreadContext, new_core};
use lichii_memory::{AddressSpace, ExclusiveMonitor, MemoryPermission, PhysicalMemory};
use unicorn_engine::{Arch, Arm64CpuModel, Mode, Prot, RegisterARM64, Unicorn};

const CODE: u64 = 0x10000;
const CODE_SIZE: u64 = 0x1000;
const DATA: u64 = 0x100000;
const DATA_SIZE: u64 = 0x8000;
const BRK: u32 = 0xd420_0000;

/// Data registers are X0..X15, load/store bases X20..X23 (never written by memory streams),
/// index registers X24 (small positive) and X26 (small negative).
const BASES: u32 = 20;
const INDEX: u32 = 24;
const NEG_INDEX: u32 = 26;

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

    fn bit(&mut self) -> u32 {
        (self.next() & 1) as u32
    }

    fn interesting(&mut self) -> u64 {
        const VALUES: [u64; 12] = [
            0,
            1,
            2,
            u64::MAX,
            0x7fff_ffff,
            0x8000_0000,
            0xffff_ffff,
            0x1_0000_0000,
            0x7fff_ffff_ffff_ffff,
            0x8000_0000_0000_0000,
            0xffff_ffff_8000_0000,
            63,
        ];
        match self.below(4) {
            0 => VALUES[self.below(VALUES.len() as u64) as usize],
            1 => self.below(64),
            _ => self.next(),
        }
    }

    /// Fills every `x` of a 32-character pattern with random bits.
    fn pattern(&mut self, bits: &str) -> u32 {
        assert_eq!(bits.len(), 32);
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

    fn data_reg(&mut self) -> u32 {
        if self.below(16) == 0 {
            31
        } else {
            self.below(16) as u32
        }
    }

    fn base(&mut self) -> u32 {
        BASES + self.below(4) as u32
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct State {
    x: [u64; 31],
    sp: u64,
    nzcv: u32,
    tpidr: u64,
    v: [u128; 32],
}

struct Oracle {
    uc: Unicorn<'static, ()>,
}

fn x_reg(i: usize) -> i32 {
    match i {
        29 => RegisterARM64::X29 as i32,
        30 => RegisterARM64::X30 as i32,
        _ => RegisterARM64::X0 as i32 + i as i32,
    }
}

impl Oracle {
    fn new() -> Self {
        let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
        uc.ctl_set_cpu_model(Arm64CpuModel::MAX as i32).unwrap();
        uc.mem_map(CODE, CODE_SIZE, Prot::ALL).unwrap();
        uc.mem_map(DATA, DATA_SIZE, Prot::ALL).unwrap();
        Self { uc }
    }

    fn run(&mut self, code: &[u8], state: &State, data: &[u8]) -> Result<(State, Vec<u8>), String> {
        let uc = &mut self.uc;
        uc.mem_write(CODE, code).unwrap();
        uc.ctl_remove_cache(CODE, CODE + CODE_SIZE).unwrap();
        uc.mem_write(DATA, data).unwrap();
        for (i, &x) in state.x.iter().enumerate() {
            uc.reg_write(x_reg(i), x).unwrap();
        }
        uc.reg_write(RegisterARM64::SP, state.sp).unwrap();
        uc.reg_write(RegisterARM64::TPIDR_EL0, state.tpidr).unwrap();
        uc.reg_write(RegisterARM64::NZCV, (state.nzcv as u64) << 28)
            .unwrap();
        for (i, v) in state.v.iter().enumerate() {
            uc.reg_write_long(RegisterARM64::Q0 as i32 + i as i32, &v.to_le_bytes())
                .unwrap();
        }
        let end = CODE + code.len() as u64 - 4;
        uc.emu_start(CODE, end, 0, 10_000)
            .map_err(|e| format!("unicorn: {e:?}"))?;
        let pc = uc.reg_read(RegisterARM64::PC).unwrap();
        if pc != end {
            return Err(format!("unicorn stopped at {pc:#x}"));
        }
        let mut out = State {
            x: std::array::from_fn(|i| uc.reg_read(x_reg(i)).unwrap()),
            sp: uc.reg_read(RegisterARM64::SP).unwrap(),
            nzcv: (uc.reg_read(RegisterARM64::NZCV).unwrap() >> 28) as u32,
            tpidr: uc.reg_read(RegisterARM64::TPIDR_EL0).unwrap(),
            v: [0; 32],
        };
        for (i, v) in out.v.iter_mut().enumerate() {
            let bytes = uc
                .reg_read_long(RegisterARM64::Q0 as i32 + i as i32)
                .unwrap();
            *v = u128::from_le_bytes(bytes[..16].try_into().unwrap());
        }
        Ok((out, uc.mem_read_as_vec(DATA, DATA_SIZE as usize).unwrap()))
    }
}

struct Clock;

impl GuestClock for Clock {
    fn ticks(&self) -> u64 {
        0
    }
}

struct Ours {
    name: &'static str,
    core: Box<dyn CpuCore>,
    mem: Arc<AddressSpace>,
}

impl Ours {
    fn new(backend: CpuBackend) -> Self {
        let mem = Arc::new(
            AddressSpace::with_fastmem(Arc::new(PhysicalMemory::new(1 << 20).unwrap()), 39)
                .unwrap(),
        );
        mem.map(CODE, 0, CODE_SIZE, MemoryPermission::READ_EXECUTE);
        let pages = DATA_SIZE / 0x1000;
        for page in 0..pages {
            let pa = 0x10_0000 - (page + 1) * 0x1000;
            mem.map(
                DATA + page * 0x1000,
                pa,
                0x1000,
                MemoryPermission::READ_WRITE,
            );
        }
        let core = new_core(
            backend,
            CpuConfig {
                memory: mem.clone(),
                monitor: Arc::new(ExclusiveMonitor::new(1)),
                clock: Arc::new(Clock),
                core_index: 0,
                ctr_el0: CpuConfig::DEFAULT_CTR_EL0,
                dczid_el0: CpuConfig::DEFAULT_DCZID_EL0,
                accuracy: CpuAccuracy::Auto,
            },
        );
        let name = match backend {
            CpuBackend::Interpreter => "interpreter",
            CpuBackend::Jit => "jit",
        };
        Self { name, core, mem }
    }

    fn run(&mut self, code: &[u8], state: &State, data: &[u8]) -> Result<(State, Vec<u8>), String> {
        self.mem.write_bytes(CODE, code).unwrap();
        self.mem.write_bytes(DATA, data).unwrap();
        self.core.clear_code_cache();
        self.core.clear_exclusive();
        let mut ctx = ThreadContext {
            x: state.x,
            sp: state.sp,
            pc: CODE,
            pstate: state.nzcv << 28,
            tpidr_el0: state.tpidr,
            v: state.v,
            ..Default::default()
        };
        let interrupt = self.core.interrupt_handle();
        let (done, wait) = std::sync::mpsc::channel::<()>();
        let watchdog = std::thread::spawn(move || {
            if wait
                .recv_timeout(std::time::Duration::from_secs(5))
                .is_err()
            {
                interrupt.signal();
            }
        });
        let halt = self.core.run(&mut ctx);
        drop(done);
        watchdog.join().unwrap();
        self.core.interrupt_handle().take();
        let end = CODE + code.len() as u64 - 4;
        if halt != HaltReason::Breakpoint(0) || ctx.pc != end {
            return Err(format!("{}: {halt:?} at {:#x}", self.name, ctx.pc));
        }
        let mut data = vec![0; DATA_SIZE as usize];
        self.mem.read_bytes(DATA, &mut data).unwrap();
        Ok((
            State {
                x: ctx.x,
                sp: ctx.sp,
                nzcv: ctx.pstate >> 28,
                tpidr: ctx.tpidr_el0,
                v: ctx.v,
            },
            data,
        ))
    }
}

fn describe(initial: &State, expected: &State, actual: &State) -> String {
    let mut lines = Vec::new();
    for i in 0..31 {
        if expected.x[i] != actual.x[i] {
            lines.push(format!(
                "x{i}: initial {:#x}, unicorn {:#x}, ours {:#x}",
                initial.x[i], expected.x[i], actual.x[i]
            ));
        }
    }
    if expected.sp != actual.sp {
        lines.push(format!(
            "sp: initial {:#x}, unicorn {:#x}, ours {:#x}",
            initial.sp, expected.sp, actual.sp
        ));
    }
    if expected.nzcv != actual.nzcv {
        lines.push(format!(
            "nzcv: initial {:#x}, unicorn {:#x}, ours {:#x}",
            initial.nzcv, expected.nzcv, actual.nzcv
        ));
    }
    if expected.tpidr != actual.tpidr {
        lines.push(format!(
            "tpidr: unicorn {:#x}, ours {:#x}",
            expected.tpidr, actual.tpidr
        ));
    }
    for i in 0..32 {
        if expected.v[i] != actual.v[i] {
            lines.push(format!(
                "v{i}: initial {:#034x}, unicorn {:#034x}, ours {:#034x}",
                initial.v[i], expected.v[i], actual.v[i]
            ));
        }
    }
    lines.join("\n")
}

/// Builds one stream of instructions for a class.
type Generator = fn(&mut Rng, &mut Vec<u32>);

fn cases() -> usize {
    std::env::var("LICHII_DIFF_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}

fn initial_state(rng: &mut Rng) -> State {
    let mut state = State {
        x: std::array::from_fn(|_| rng.interesting()),
        sp: DATA + 0x4000,
        nzcv: rng.below(16) as u32,
        tpidr: rng.next(),
        v: std::array::from_fn(|_| (rng.next() as u128) << 64 | rng.next() as u128),
    };
    for b in 0..4 {
        state.x[(BASES + b) as usize] = DATA + 0x3000 + 16 * rng.below(0x200);
    }
    state.x[INDEX as usize] = rng.below(64);
    state.x[NEG_INDEX as usize] = rng.below(64).wrapping_neg();
    state
}

fn differential(name: &str, seed: u64, generate: Generator) {
    let mut rng = Rng(seed);
    let mut oracle = Oracle::new();
    let mut ours = backends();
    let mut data = vec![0u8; DATA_SIZE as usize];
    let mut stream = Vec::new();
    for case in 0..cases() {
        stream.clear();
        generate(&mut rng, &mut stream);
        for chunk in data.chunks_mut(8) {
            chunk.copy_from_slice(&rng.next().to_le_bytes());
        }
        let state = initial_state(&mut rng);
        compare(
            &mut oracle,
            &mut ours,
            &format!("{name} case {case}"),
            &stream,
            &state,
            &data,
        );
    }
}

/// Runs `stream` followed by `BRK #0` on Unicorn and on each of `ours`, and panics on any
/// difference.
fn compare(
    oracle: &mut Oracle,
    ours: &mut [Ours],
    name: &str,
    stream: &[u32],
    state: &State,
    data: &[u8],
) {
    let code: Vec<u8> = stream
        .iter()
        .chain([&BRK])
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let expected = oracle.run(&code, state, data);
    let words: Vec<String> = stream.iter().map(|w| format!("{w:08x}")).collect();
    for ours in ours {
        let actual = ours.run(&code, state, data);
        let context = format!("{name} ({}): {}", ours.name, words.join(" "));
        match (&expected, actual) {
            (Ok((es, em)), Ok((as_, am))) => {
                if as_ != *es {
                    panic!("{context}\n{}", describe(state, es, &as_));
                }
                if let Some(i) = (0..em.len()).find(|&i| em[i] != am[i]) {
                    panic!(
                        "{context}\nmemory differs at {:#x}: unicorn {:#x}, ours {:#x}",
                        DATA + i as u64,
                        em[i],
                        am[i]
                    );
                }
            }
            (e, a) => panic!(
                "{context}\nunicorn {:?}\nours {:?}",
                e.as_ref().map(|_| ()),
                a.map(|_| ())
            ),
        }
    }
}

fn backends() -> [Ours; 2] {
    [
        Ours::new(CpuBackend::Interpreter),
        Ours::new(CpuBackend::Jit),
    ]
}

/// Runs `stream` on both from a state where `regs` override the zeroed general registers.
fn directed(stream: &[u32], regs: &[(usize, u64)]) {
    let mut state = State {
        x: [0; 31],
        sp: DATA + 0x4000,
        nzcv: 0,
        tpidr: 0,
        v: [0; 32],
    };
    for &(r, value) in regs {
        state.x[r] = value;
    }
    let data = vec![0u8; DATA_SIZE as usize];
    compare(
        &mut Oracle::new(),
        &mut backends(),
        "directed",
        stream,
        &state,
        &data,
    );
}

/// Appends `count` random words matching `bits` that decode to something other than
/// `Undefined` and pass `keep`.
fn push_matching(rng: &mut Rng, out: &mut Vec<u32>, bits: &str, keep: fn(u32) -> bool) {
    loop {
        let w = rng.pattern(bits);
        if !matches!(decode(w), Inst::Undefined { .. }) && keep(w) {
            out.push(w);
            return;
        }
    }
}

fn alu_stream(rng: &mut Rng, out: &mut Vec<u32>, bits: &str, keep: fn(u32) -> bool) {
    for _ in 0..1 + rng.below(8) {
        push_matching(rng, out, bits, keep);
    }
}

fn any(_: u32) -> bool {
    true
}

macro_rules! alu_class {
    ($name:ident, $bits:literal) => {
        alu_class!($name, $bits, any);
    };
    ($name:ident, $bits:literal, $keep:expr) => {
        #[test]
        fn $name() {
            differential(
                stringify!($name),
                0x1234_5678 ^ line!() as u64,
                |rng, out| alu_stream(rng, out, $bits, $keep),
            );
        }
    };
}

alu_class!(pc_relative, "xxx10000xxxxxxxxxxxxxxxxxxxxxxxx");
alu_class!(add_sub_immediate, "xxx100010xxxxxxxxxxxxxxxxxxxxxxx");
alu_class!(logical_immediate, "xxx100100xxxxxxxxxxxxxxxxxxxxxxx");
alu_class!(move_wide, "xxx100101xxxxxxxxxxxxxxxxxxxxxxx");
alu_class!(bitfield, "xxx100110xxxxxxxxxxxxxxxxxxxxxxx");
alu_class!(extract, "xxx100111xxxxxxxxxxxxxxxxxxxxxxx");
alu_class!(add_sub_shifted, "xxx01011xx0xxxxxxxxxxxxxxxxxxxxx");
alu_class!(add_sub_extended, "xxx01011xx1xxxxxxxxxxxxxxxxxxxxx");
alu_class!(add_sub_carry, "xxx11010000xxxxx000000xxxxxxxxxx");
alu_class!(logical_shifted, "xxx01010xxxxxxxxxxxxxxxxxxxxxxxx");
alu_class!(conditional_compare, "xx111010010xxxxxxxxxx0xxxxx0xxxx");
alu_class!(conditional_select, "xx011010100xxxxxxxxxxxxxxxxxxxxx");
alu_class!(data_processing_1, "x1011010110000000000xxxxxxxxxxxx");
alu_class!(data_processing_2, "x0011010110xxxxxxxxxxxxxxxxxxxxx", |w| {
    !matches!((w >> 10) & 0x3f, 4 | 5 | 12)
});
alu_class!(data_processing_3, "xxx11011xxxxxxxxxxxxxxxxxxxxxxxx");
alu_class!(cnt, "0x00111000100000010110xxxxxxxxxx");
alu_class!(addp, "0x001110xx1xxxxx101111xxxxxxxxxx", |w| {
    w & 0x40c0_0000 != 0x00c0_0000
});
alu_class!(crc32, "x0011010110xxxxx010xxxxxxxxxxxxx");
alu_class!(vector_fp_arith, "0xx01110xx1xxxxx110xx1xxxxxxxxxx", |w| {
    matches!(
        w & 0xbfa0_fc00,
        0x0e20_d400 | 0x0ea0_d400 | 0x2e20_dc00 | 0x0e20_cc00 | 0x0ea0_cc00
    ) && w & 0x4040_0000 != 0x0040_0000
});
alu_class!(
    scalar_fp_by_element,
    "01x111111xxxxxxxxx01x0xxxxxxxxxx",
    |w| {
        matches!(w & 0xff80_f400, 0x5f80_1000 | 0x5f80_5000 | 0x5f80_9000)
            && w & 0x0060_0000 != 0x0060_0000
    }
);
alu_class!(int_to_fp_in_simd, "01x111100x100001110110xxxxxxxxxx");
alu_class!(fcvt_from_half, "0001111011100010x10000xxxxxxxxxx");
alu_class!(fp_conditional_compare, "000111100x1xxxxxxxxx01xxxxxxxxxx");
alu_class!(fnmul, "000111100x1xxxxx100010xxxxxxxxxx");
alu_class!(scalar_dup, "01011110000xxxxx000001xxxxxxxxxx", |w| w
    & 0xf_0000
    != 0);
alu_class!(faddp_vector, "0x101110x01xxxxx110101xxxxxxxxxx", |w| {
    w & 0x4040_0000 != 0x0040_0000
});
alu_class!(ext, "0x101110000xxxxx0xxxx0xxxxxxxxxx", |w| w & 0x4000_4000
    != 0x4000);
alu_class!(
    vector_fp_by_element,
    "0xx011111xxxxxxxxx01x0xxxxxxxxxx",
    |w| {
        matches!(w & 0xbf80_f400, 0x0f80_1000 | 0x0f80_5000 | 0x0f80_9000)
            && w & 0x4060_0000 != 0x0040_0000
            && w & 0x0060_0000 != 0x0060_0000
    }
);

#[test]
fn flags() {
    differential("flags", 0xf1a6, |rng, out| {
        for _ in 0..1 + rng.below(8) {
            let w = match rng.below(7) {
                0 => rng.pattern("10111010000xxxxxx00001xxxxx0xxxx"),
                1 => rng.pattern("00111010000000000x0010xxxxx01101"),
                2 => [0xd500401f, 0xd500403f, 0xd500405f][rng.below(3) as usize],
                3 => 0xd51b4200 | rng.below(16) as u32,
                4 => 0xd53b4200 | rng.below(16) as u32,
                5 => rng.pattern("xx111010010xxxxxxxxxx0xxxxx0xxxx"),
                _ => rng.pattern("xx1100010xxxxxxxxxxxxxxxxxxxxxxx"),
            };
            out.push(w);
        }
    });
}

#[test]
fn branches() {
    differential("branches", 0xb4a2, |rng, out| {
        // X16 holds the end of the stream for the indirect branches; fillers only write X0..X15.
        let len = 2 + rng.below(10) as usize;
        out.push(0x1000_0010 | (len as u32) << 5);
        for i in 1..len {
            let to_end = |at: usize| (len - at) as u64;
            let target = |rng: &mut Rng, at: usize| (1 + rng.below(to_end(at))) as u32;
            let w = match rng.below(8) {
                0 => 0x5400_0000 | (target(rng, i) << 5) | rng.below(16) as u32,
                1 => {
                    let base = if rng.bit() != 0 {
                        0x3500_0000
                    } else {
                        0x3400_0000
                    };
                    base | rng.bit() << 31 | (target(rng, i) << 5) | rng.below(16) as u32
                }
                2 => {
                    let b = rng.below(64) as u32;
                    let base = if rng.bit() != 0 {
                        0x3700_0000
                    } else {
                        0x3600_0000
                    };
                    base | (b >> 5) << 31
                        | (b & 31) << 19
                        | (target(rng, i) << 5)
                        | rng.below(16) as u32
                }
                3 => 0x1400_0000 | target(rng, i),
                4 => 0x9400_0000 | target(rng, i),
                5 => [0xd61f_0200, 0xd63f_0200, 0xd65f_0200][rng.below(3) as usize],
                _ => rng.pattern("xx1100010xxxxxxxxxxxxxxxxxx0xxxx"),
            };
            out.push(w);
        }
    });
}

/// A load/store register (any addressing form) with constrained registers and small offsets.
fn single_register(rng: &mut Rng) -> u32 {
    let (rt, rn) = (rng.data_reg(), rng.base());
    let head =
        (rng.below(4) as u32) << 30 | 0b111 << 27 | rng.bit() << 26 | (rng.below(4) as u32) << 22;
    match rng.below(3) {
        0 => head | 1 << 24 | (rng.below(64) as u32) << 10 | rn << 5 | rt,
        1 => {
            let imm9 = (rng.below(512) as u32).wrapping_sub(256) & 0x1ff;
            head | imm9 << 12 | (rng.below(4) as u32) << 10 | rn << 5 | rt
        }
        _ => {
            let (rm, option) = if rng.bit() != 0 {
                (NEG_INDEX, [6, 7][rng.below(2) as usize])
            } else {
                (INDEX, [2, 3, 6, 7][rng.below(4) as usize])
            };
            head | 1 << 21 | rm << 16 | option << 13 | rng.bit() << 12 | 0b10 << 10 | rn << 5 | rt
        }
    }
}

fn pair(rng: &mut Rng) -> Option<u32> {
    let (v, l, opc) = (rng.bit(), rng.bit(), rng.below(3) as u32);
    if v == 0 && opc == 1 && l == 0 {
        return None;
    }
    let rt = rng.below(16) as u32;
    let rt2 = (rt + 1 + rng.below(15) as u32) % 16;
    let imm7 = (rng.below(32) as u32).wrapping_sub(16) & 0x7f;
    Some(
        opc << 30
            | 0b101 << 27
            | v << 26
            | (rng.below(4) as u32) << 23
            | l << 22
            | imm7 << 15
            | rt2 << 10
            | rng.base() << 5
            | rt,
    )
}

#[test]
fn loads_and_stores() {
    differential("loads_and_stores", 0x10ad, |rng, out| {
        for _ in 0..1 + rng.below(10) {
            loop {
                let w = if rng.below(3) == 0 {
                    pair(rng)
                } else {
                    Some(single_register(rng))
                };
                if let Some(w) = w.filter(|&w| !matches!(decode(w), Inst::Undefined { .. })) {
                    out.push(w);
                    break;
                }
            }
        }
    });
}

#[test]
fn page_crossing_accesses() {
    differential("page_crossing", 0x9a6e, |rng, out| {
        for _ in 0..1 + rng.below(6) {
            let rn = rng.base();
            let back = 1 + rng.below(16) as u32;
            out.push(0xd100_0000 | (back + 0x1000 - 0x1000) << 10 | rn << 5 | rn);
            loop {
                let w = (rng.below(4) as u32) << 30
                    | 0b111 << 27
                    | rng.bit() << 26
                    | (rng.below(4) as u32) << 22
                    | rn << 5
                    | rng.data_reg();
                if !matches!(decode(w), Inst::Undefined { .. }) {
                    out.push(w);
                    break;
                }
            }
        }
    });
}

#[test]
fn load_literal() {
    differential("load_literal", 0x717e, |rng, out| {
        let len = 1 + rng.below(8) as usize;
        for i in 0..len {
            let offset = rng.below(len as u64 + 4) as i64 - i as i64;
            let imm19 = (offset as u32) & 0x7ffff;
            out.push(
                (rng.below(3) as u32) << 30
                    | 0b011 << 27
                    | rng.bit() << 26
                    | imm19 << 5
                    | rng.data_reg(),
            );
        }
        out.extend([0xd503201f; 4]);
    });
}

#[test]
fn exclusives() {
    differential("exclusives", 0xe8c1, |rng, out| {
        for _ in 0..1 + rng.below(4) {
            let rn = rng.base();
            let pair = rng.below(3) == 0;
            let size = if pair {
                2 + rng.below(2) as u32
            } else {
                rng.below(4) as u32
            };
            let (rt, rt2, rs) = (
                rng.below(5) as u32,
                5 + rng.below(5) as u32,
                10 + rng.below(5) as u32,
            );
            let (acquire, release) = (rng.bit(), rng.bit());
            let (head, p) = if pair {
                (1 << 31 | (size & 1) << 30, 1 << 21)
            } else {
                (size << 30, 0)
            };
            out.push(
                head | 0b001000 << 24
                    | 1 << 22
                    | p
                    | 31 << 16
                    | acquire << 15
                    | (if pair { rt2 } else { 31 }) << 10
                    | rn << 5
                    | rt,
            );
            match rng.below(4) {
                0 => out.push(0xd503_3f5f),
                1 => out.push(0xb800_0000 | rn << 5 | rng.below(16) as u32),
                _ => {}
            }
            out.push(
                head | 0b001000 << 24
                    | p
                    | rs << 16
                    | release << 15
                    | (if pair { rt2 } else { 31 }) << 10
                    | rn << 5
                    | rt2,
            );
        }
    });
}

#[test]
fn atomics() {
    differential("atomics", 0xa70c, |rng, out| {
        for _ in 0..1 + rng.below(8) {
            let rn = rng.base();
            let size = rng.below(4) as u32;
            let (a, r) = (rng.bit(), rng.bit());
            let (rs, rt) = (rng.below(16) as u32, rng.data_reg());
            let w = match rng.below(5) {
                0..=2 => {
                    let (o3, opc) = match rng.below(9) {
                        0 => (1, 0),
                        _ if size < 3 => (0, [0, 1, 2, 3, 6, 7][rng.below(6) as usize]),
                        _ => (0, rng.below(8) as u32),
                    };
                    size << 30
                        | 0b111 << 27
                        | a << 23
                        | r << 22
                        | 1 << 21
                        | rs << 16
                        | o3 << 15
                        | opc << 12
                        | rn << 5
                        | rt
                }
                3 => {
                    size << 30
                        | 0b001000 << 24
                        | 1 << 23
                        | a << 22
                        | 1 << 21
                        | rs << 16
                        | r << 15
                        | 31 << 10
                        | rn << 5
                        | rt
                }
                _ => {
                    let (s, t) = (2 * rng.below(8) as u32, 2 * rng.below(8) as u32);
                    (size & 1) << 30
                        | 0b001000 << 24
                        | a << 22
                        | 1 << 21
                        | s << 16
                        | r << 15
                        | 31 << 10
                        | rn << 5
                        | t
                }
            };
            out.push(w);
        }
    });
}

#[test]
fn acquire_release() {
    differential("acquire_release", 0xac9e, |rng, out| {
        for _ in 0..1 + rng.below(8) {
            let rn = rng.base();
            let size = rng.below(4) as u32;
            let rt = rng.data_reg();
            let w = match rng.below(4) {
                0 => {
                    size << 30
                        | 0b001000 << 24
                        | 1 << 23
                        | 1 << 22
                        | 31 << 16
                        | rng.bit() << 15
                        | 31 << 10
                        | rn << 5
                        | rt
                }
                1 => {
                    size << 30
                        | 0b001000 << 24
                        | 1 << 23
                        | 31 << 16
                        | rng.bit() << 15
                        | 31 << 10
                        | rn << 5
                        | rt
                }
                2 => {
                    size << 30
                        | 0b111 << 27
                        | 1 << 23
                        | 1 << 21
                        | 31 << 16
                        | 1 << 15
                        | 4 << 12
                        | rn << 5
                        | rt
                }
                _ => {
                    let imm = ((rng.below(32) as i32 - 16) << size) as u32 & 0x1ff;
                    size << 30
                        | 0b011001 << 24
                        | (rng.below(4) as u32) << 22
                        | imm << 12
                        | rn << 5
                        | rt
                }
            };
            if !matches!(decode(w), Inst::Undefined { .. }) {
                out.push(w);
            }
        }
    });
}

#[test]
fn vector_structures() {
    differential("vector_structures", 0x57c7, |rng, out| {
        for _ in 0..1 + rng.below(6) {
            loop {
                let rn = rng.base();
                let rm = if rng.bit() != 0 { 31 } else { INDEX };
                let post = rng.bit();
                let head = rng.bit() << 30 | rng.bit() << 22 | rng.below(32) as u32 | rn << 5;
                let w = if rng.bit() != 0 {
                    head | 0b0011000 << 23
                        | post << 23
                        | (if post != 0 { rm } else { 0 }) << 16
                        | (rng.below(16) as u32) << 12
                        | (rng.below(4) as u32) << 10
                } else {
                    head | 0b0011010 << 23
                        | post << 23
                        | rng.bit() << 21
                        | (if post != 0 { rm } else { 0 }) << 16
                        | (rng.below(8) as u32) << 13
                        | rng.bit() << 12
                        | (rng.below(4) as u32) << 10
                };
                if !matches!(decode(w), Inst::Undefined { .. }) {
                    out.push(w);
                    break;
                }
            }
        }
    });
}

#[test]
fn system() {
    differential("system", 0x5355, |rng, out| {
        for _ in 0..1 + rng.below(6) {
            let w = match rng.below(6) {
                0 => 0xd50b_7420 | rng.base(),
                1 => [
                    0xd503_201f,
                    0xd503_203f,
                    0xd503_3bbf,
                    0xd503_3f9f,
                    0xd503_3fdf,
                ][rng.below(5) as usize],
                2 => 0xd51b_d040 | rng.below(16) as u32,
                3 => 0xd53b_d040 | rng.below(16) as u32,
                4 => 0xd50b_7e20 | rng.base(),
                _ => rng.pattern("xx1100010xxxxxxxxxxxxxxxxxx0xxxx"),
            };
            out.push(w);
        }
    });
}

const MIN: u64 = 0x8000_0000_0000_0000;

#[test]
fn division_edges() {
    let program = [0x9ac20c20, 0x1ac50c83, 0x9ac808e6, 0x9ac80c29, 0x1ac8088a];
    directed(
        &program,
        &[
            (1, MIN),
            (2, u64::MAX),
            (4, 0x8000_0000),
            (5, 0xffff_ffff),
            (7, 77),
        ],
    );
}

#[test]
fn shift_amounts_wrap() {
    let program = [
        0x9ac22020, 0x9ac42823, 0x1ac42c25, 0x1ac22426, 0x937ffc27, 0x13817c28,
    ];
    for amount in [0, 31, 32, 63, 64, 65, u64::MAX] {
        directed(
            &program,
            &[
                (1, 0x8123_4567_89ab_cdef),
                (2, amount),
                (4, amount.wrapping_add(1)),
            ],
        );
    }
}

#[test]
fn flag_edges() {
    let program = [
        0xab0c0160, 0x6b0c03e1, 0xba0d01a2, 0x7a0e01c3, 0xba5f696f, 0xeb0b01bf,
    ];
    for (a, b) in [
        (i64::MAX as u64, 1),
        (MIN, u64::MAX),
        (0, 0),
        (u64::MAX, u64::MAX),
        (0x7fff_ffff, 1),
    ] {
        directed(&program, &[(11, a), (12, b), (13, a), (14, b)]);
    }
}

#[test]
fn bitfield_edges() {
    let program = [
        0x937ffc20, 0x53010022, 0xb340fc23, 0x33103c24, 0x93407c25, 0x13847c26,
    ];
    for value in [MIN, 1, u64::MAX, 0x8000_0000, 0x1234_5678_9abc_def0] {
        directed(&program, &[(1, value), (3, 0x5555), (4, 0xaaaa_aaaa)]);
    }
}
