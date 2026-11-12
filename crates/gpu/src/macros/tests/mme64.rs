use super::masm::*;
use super::{TestHost, engine_with};
use crate::macros::{
    MACRO_METHOD_BASE, MacroDataRam, MacroEngine, MacroError, MacroErrorKind, MacroIsa,
};

fn engine(program: &[Inst]) -> MacroEngine {
    let mut engine = engine_with(MacroIsa::Mme64, &assemble(program));
    engine.set_memo_enabled(false);
    engine
}

fn run(program: &[Inst], parameters: &[u32], host: &mut TestHost) -> Result<(), MacroError> {
    engine(program).execute(
        MACRO_METHOD_BASE,
        parameters,
        &mut MacroDataRam::new(),
        host,
    )
}

fn writes(program: &[Inst], parameters: &[u32]) -> Vec<(u32, u32)> {
    let mut host = TestHost::default();
    run(program, parameters, &mut host).expect("macro failed");
    host.writes
}

fn values(program: &[Inst], parameters: &[u32]) -> Vec<u32> {
    writes(program, parameters)
        .into_iter()
        .map(|(_, value)| value)
        .collect()
}

fn error(program: &[Inst], parameters: &[u32]) -> MacroErrorKind {
    run(program, parameters, &mut TestHost::default())
        .expect_err("macro succeeded")
        .kind
}

#[test]
fn loads_outputs_and_end_delay_slot() {
    let program = [
        inst(
            alu(ADD, 0, LOAD0, ZERO, 0x1100),
            alu(ADD, 1, LOAD1, ZERO, 0),
        )
        .out0(IMM0, ALU0)
        .out1(NONE, ALU1)
        .end(),
        inst(imm(0xA123), imm(0x42))
            .out0(NONE, IMMHIGH0)
            .out1(NONE, OUT_IMM32),
        inst(imm(0x99), nop()).out0(NONE, IMM0),
    ];
    assert_eq!(
        writes(&program, &[0xAA, 0xBB]),
        [
            (0x100, 0xAA),
            (0x101, 0xBB),
            (0x102, 0xA),
            (0x103, 0xA123_0042)
        ]
    );
}

#[test]
fn alu_operations() {
    let merge = (8 << 10) | (4 << 5) | 4;
    let cases: [(u8, u32, u32, u16, u32); 18] = [
        (ADD, 5, 7, 0, 12),
        (SUB, 5, 7, 0, 0xFFFF_FFFE),
        (MUL, (-3i32) as u32, 4, 0, (-12i32) as u32),
        (MULU, 0x1_0000, 0x1_0000, 0, 0),
        (CLZ, 1, 0, 0, 31),
        (SLL, 1, 33, 0, 2),
        (SRL, 0x8000_0000, 4, 0, 0x0800_0000),
        (SRA, 0x8000_0000, 4, 0, 0xF800_0000),
        (AND, 0b1100, 0b1010, 0, 0b1000),
        (NAND, 0b1100, 0b1010, 0, !0b1000),
        (OR, 0b1100, 0b1010, 0, 0b1110),
        (XOR, 0b1100, 0b1010, 0, 0b0110),
        (MERGE, 0xFFFF_FFFF, 0xA0, merge, 0xFFFF_FAFF),
        (SLT, u32::MAX, 1, 0, u32::MAX),
        (SLTU, u32::MAX, 1, 0, 0),
        (SLE, 1, 1, 0, u32::MAX),
        (SLEU, 2, 1, 0, 0),
        (SEQ, 3, 3, 0, u32::MAX),
    ];
    for (op, x, y, immediate, expected) in cases {
        let program = [
            inst(alu(ADD, 0, LOAD0, ZERO, 0), alu(ADD, 1, LOAD1, ZERO, 0)),
            inst(alu(op, 2, 0, 1, immediate), imm(0x200))
                .out0(IMM1, ALU0)
                .end(),
            inst(nop(), nop()),
        ];
        assert_eq!(writes(&program, &[x, y]), [(0x200, expected)], "op {op}");
    }
}

#[test]
fn carry_chains_between_lanes() {
    let program = [
        inst(alu(ADD, 0, LOAD0, ZERO, 0), alu(ADD, 1, LOAD1, ZERO, 0)),
        inst(alu(ADD, 2, 0, 1, 0x1200), alu(ADDC, 3, ZERO, ZERO, 0))
            .out0(IMM0, ALU0)
            .out1(NONE, ALU1),
        inst(alu(SUB, 2, 1, 0, 0), alu(SUBB, 3, ZERO, ZERO, 0))
            .out0(NONE, ALU0)
            .out1(NONE, ALU1),
        inst(alu(MULU, 2, 0, 1, 0), alu(MULH, 3, ZERO, ZERO, 0))
            .out0(NONE, ALU0)
            .out1(NONE, ALU1)
            .end(),
        inst(nop(), nop()),
    ];
    assert_eq!(
        values(&program, &[0xFFFF_FFFF, 2]),
        [1, 1, 3, 0xFFFF_FFFF, 0xFFFF_FFFE, 1]
    );
}

#[test]
fn operands_are_read_before_either_lane_writes() {
    let program = [
        inst(alu(ADD, 0, LOAD0, ZERO, 0), alu(ADD, 1, LOAD1, ZERO, 0)),
        inst(alu(ADD, 0, 0, IMM, 1), alu(ADD, 1, 0, ZERO, 0)),
        inst(alu(ADD, 2, 0, ZERO, 0x1300), alu(ADD, 3, 1, ZERO, 0))
            .out0(IMM0, ALU0)
            .out1(NONE, ALU1)
            .end(),
        inst(nop(), nop()),
    ];
    assert_eq!(writes(&program, &[5, 0]), [(0x300, 6), (0x301, 5)]);
}

#[test]
fn immediate_sources() {
    let program = [
        inst(
            alu(ADD, 0, IMMPAIR, ZERO, 0x1400),
            alu(ADD, 1, IMM32, ZERO, 0xFFFE),
        )
        .out0(IMM0, ALU0)
        .out1(NONE, ALU1),
        inst(alu(ADD, 2, IMM, ZERO, 0x8000), nop())
            .out0(NONE, ALU0)
            .end(),
        inst(nop(), nop()),
    ];
    assert_eq!(
        values(&program, &[]),
        [0xFFFF_FFFE, 0x1400_FFFE, 0xFFFF_8000]
    );
}

#[test]
fn predicated_lanes() {
    let alus = [
        inst(alu(ADD, 0, LOAD0, ZERO, 0), alu(ADD, 2, ZERO, ZERO, 0x1300)).out0(IMM1, NONE),
        inst(alu(ADD, 3, ZERO, ZERO, 0), nop()),
        inst(alu(ADD, 2, ZERO, IMM, 0x11), alu(ADD, 3, ZERO, IMM, 0x22)).pred(5, 0),
        inst(alu(ADD, 4, 2, ZERO, 0), alu(ADD, 5, 3, ZERO, 0))
            .out0(NONE, ALU0)
            .out1(NONE, ALU1)
            .end(),
        inst(nop(), nop()),
    ];
    assert_eq!(writes(&alus, &[0]), [(0x300, 0x11), (0x301, 0)]);
    assert_eq!(writes(&alus, &[7]), [(0x300, 0), (0x301, 0x22)]);

    let outputs = [
        inst(alu(ADD, 0, LOAD0, ZERO, 0), imm(0x400)).out0(IMM1, NONE),
        inst(imm(0xA), imm(0xB))
            .out0(NONE, IMM0)
            .out1(NONE, IMM1)
            .pred(9, 0)
            .end(),
        inst(nop(), nop()),
    ];
    assert_eq!(values(&outputs, &[0]), [0xA]);
    assert_eq!(values(&outputs, &[1]), [0xB]);
}

#[test]
fn loops() {
    let program = [
        inst(alu(ADD, 0, LOAD0, ZERO, 0), alu(ADD, 1, ZERO, ZERO, 0x400)).out0(IMM1, NONE),
        inst(alu(LOOP, ZERO, 0, ZERO, 3), nop()),
        inst(alu(ADD, 1, 1, IMM, 1), nop()).out0(NONE, ALU0),
        inst(nop(), nop()),
        inst(nop(), nop()).end(),
        inst(nop(), nop()),
    ];
    assert_eq!(values(&program, &[3]), [1, 2, 3]);
    assert_eq!(values(&program, &[1]), [1]);
    assert!(values(&program, &[0]).is_empty());

    let nested = [
        inst(alu(ADD, 0, LOAD0, ZERO, 0), nop()),
        inst(alu(LOOP, ZERO, 0, ZERO, 3), nop()),
        inst(alu(LOOP, ZERO, 0, ZERO, 2), nop()),
        inst(nop(), nop()),
        inst(nop(), nop()).end(),
        inst(nop(), nop()),
    ];
    assert_eq!(error(&nested, &[2]), MacroErrorKind::InvalidLoop);
    let short = [
        inst(alu(LOOP, ZERO, IMM, ZERO, 1), nop()).end(),
        inst(nop(), nop()),
    ];
    assert_eq!(error(&short, &[]), MacroErrorKind::InvalidLoop);
}

#[test]
fn conditional_branches_and_end() {
    let program = [
        inst(alu(ADD, 0, LOAD0, ZERO, 0), imm(0x500)).out0(IMM1, NONE),
        inst(alu(BEQ, ZERO, 0, ZERO, branch_imm(true, false, 3)), nop()),
        inst(imm(1), nop()).out0(NONE, IMM0),
        inst(imm(2), nop()).out0(NONE, IMM0).end(),
        inst(imm(3), nop()).out0(NONE, IMM0).end(),
        inst(imm(4), nop()).out0(NONE, IMM0),
        inst(imm(5), nop()).out0(NONE, IMM0),
    ];
    assert_eq!(values(&program, &[0]), [3, 4]);
    assert_eq!(values(&program, &[1]), [1, 2, 3]);
}

#[test]
fn back_edge_cancels_end() {
    let program = [
        inst(alu(ADD, 0, LOAD0, ZERO, 0), alu(ADD, 1, ZERO, ZERO, 0x600)).out0(IMM1, NONE),
        inst(alu(ADD, 1, 1, IMM, 1), nop()).out0(NONE, ALU0),
        inst(alu(BLT, ZERO, 1, 0, branch_imm(true, true, -1)), nop()).end(),
        inst(imm(0x99), nop()).out0(NONE, IMM0),
        inst(imm(0x77), nop()).out0(NONE, IMM0),
    ];
    assert_eq!(values(&program, &[3]), [1, 2, 3, 0x99]);
}

#[test]
fn stop_branch_and_jal() {
    let stop = [
        inst(
            alu(BEQ, ZERO, ZERO, ZERO, branch_imm(true, false, -4096)),
            imm(0x700),
        )
        .out0(IMM1, NONE)
        .out1(NONE, IMM1),
        inst(imm(5), nop()).out0(NONE, IMM0),
    ];
    assert_eq!(writes(&stop, &[]), [(0x700, 0x700)]);

    let jal = [
        inst(alu(JAL, ZERO, ZERO, ZERO, 0x8002), imm(0x700)).out0(IMM1, NONE),
        inst(imm(1), nop()).out0(NONE, IMM0),
        inst(imm(2), nop()).out0(NONE, IMM0).end(),
        inst(nop(), nop()),
    ];
    assert_eq!(writes(&jal, &[]), [(0x700, 2)]);
}

#[test]
fn data_ram_reads_and_writes() {
    let program = [
        inst(alu(DWRITE, ZERO, LOAD0, LOAD1, 0), nop()),
        inst(alu(DREAD, 0, IMM, ZERO, 5), imm(0x800)).out0(IMM1, NONE),
        inst(alu(ADD, 1, 0, ZERO, 0), nop()).out0(NONE, ALU0).end(),
        inst(nop(), nop()),
    ];
    let mut engine = engine(&program);
    let mut data_ram = MacroDataRam::new();
    data_ram.words_mut()[5] = 0x1234;
    let mut host = TestHost::default();
    engine
        .execute(MACRO_METHOD_BASE, &[0x10, 0xBEEF], &mut data_ram, &mut host)
        .unwrap();
    engine
        .execute(MACRO_METHOD_BASE, &[5, 0xCAFE], &mut data_ram, &mut host)
        .unwrap();
    assert_eq!(host.writes, [(0x800, 0x1234), (0x800, 0xCAFE)]);
    assert_eq!(data_ram.words()[0x10], 0xBEEF);
    let out_of_bounds = engine
        .execute(MACRO_METHOD_BASE, &[0xC00, 1], &mut data_ram, &mut host)
        .unwrap_err();
    assert_eq!(out_of_bounds.kind, MacroErrorKind::DataRamOutOfBounds);
    assert_eq!(out_of_bounds.pc, 0);
}

#[test]
fn data_ram_is_shared_by_a_channels_engines() {
    let mut graphics = engine(&[
        inst(alu(DWRITE, ZERO, IMM, LOAD0, 0x27), nop()).end(),
        inst(nop(), nop()),
    ]);
    let mut compute = engine(&[
        inst(alu(DREAD, 0, IMM, ZERO, 0x27), imm(0x900)).out0(IMM1, NONE),
        inst(alu(ADD, 1, 0, ZERO, 0), nop()).out0(NONE, ALU0).end(),
        inst(nop(), nop()),
    ]);
    let mut channel_a = MacroDataRam::new();
    let mut channel_b = MacroDataRam::new();
    let mut host = TestHost::default();
    graphics
        .execute(MACRO_METHOD_BASE, &[0xA], &mut channel_a, &mut host)
        .unwrap();
    graphics
        .execute(MACRO_METHOD_BASE, &[0xB], &mut channel_b, &mut host)
        .unwrap();
    compute
        .execute(MACRO_METHOD_BASE, &[], &mut channel_a, &mut host)
        .unwrap();
    compute
        .execute(MACRO_METHOD_BASE, &[], &mut channel_b, &mut host)
        .unwrap();
    assert_eq!(host.writes, [(0x900, 0xA), (0x900, 0xB)]);
}

#[test]
fn state_reads_registers_and_is_never_replayed() {
    let program = [
        inst(alu(STATE, 0, IMM, ZERO, 0x120), imm(0xA00)).out0(IMM1, NONE),
        inst(alu(ADD, 1, 0, ZERO, 0), nop()).out0(NONE, ALU0).end(),
        inst(nop(), nop()),
    ];
    let mut engine = engine_with(MacroIsa::Mme64, &assemble(&program));
    let mut host = TestHost::default();
    let mut data_ram = MacroDataRam::new();
    host.registers.insert(0x120, 0x5A);
    for expected in [0x5A, 0x5B] {
        engine
            .execute(MACRO_METHOD_BASE, &[], &mut data_ram, &mut host)
            .unwrap();
        assert_eq!(host.writes.last(), Some(&(0xA00, expected)));
        host.registers.insert(0x120, 0x5B);
    }
    assert_eq!(host.register_reads, 2);
    assert_eq!(engine.stats().replayed, 0);
}

fn dma_host(words: &[u32]) -> (TestHost, u64) {
    let mut host = TestHost::default();
    let gpu_va = (0x1_0001u64 << 32) | 0x1000;
    for (i, &word) in words.iter().enumerate() {
        host.memory.insert(gpu_va + 4 * i as u64, word);
    }
    (host, gpu_va)
}

#[test]
fn dma_read_into_data_ram() {
    let program = [
        inst(imm(0x1154), nop())
            .out0(IMM0, OUT_LOAD0)
            .out1(NONE, OUT_LOAD1),
        inst(imm(0x20), imm(3)).out0(NONE, IMM0).out1(NONE, IMM1),
        inst(alu(DREAD, 0, IMM, ZERO, 0x21), imm(0xB00)).out0(IMM1, NONE),
        inst(alu(ADD, 1, 0, ZERO, 0), nop()).out0(NONE, ALU0).end(),
        inst(nop(), nop()),
    ];
    let mut engine = engine_with(MacroIsa::Mme64, &assemble(&program));
    let (mut host, gpu_va) = dma_host(&[7, 8, 9]);
    let mut data_ram = MacroDataRam::new();
    for _ in 0..2 {
        engine
            .execute(
                MACRO_METHOD_BASE,
                &[0xFFFF_0001, 0x1000],
                &mut data_ram,
                &mut host,
            )
            .unwrap();
    }
    assert_eq!(data_ram.words()[0x20..0x23], [7, 8, 9]);
    assert_eq!(host.writes.last(), Some(&(0xB00, 8)));
    assert_eq!(host.dma_reads, [(gpu_va, 3), (gpu_va, 3)]);
    assert_eq!(engine.stats().replayed, 0);
}

fn fifoed_program(count: u16) -> [Inst; 5] {
    [
        inst(imm(0x1154), nop())
            .out0(IMM0, OUT_LOAD0)
            .out1(NONE, OUT_LOAD1),
        inst(imm(0x0158), imm(count)).out0(IMM0, IMM1),
        inst(alu(EXTENDED, ZERO, IMM, IMM, 0x1000), nop()),
        inst(
            alu(ADD, 0, LOAD0, ZERO, 0x1C00),
            alu(ADD, 1, LOAD1, ZERO, 0),
        )
        .out0(IMM0, ALU0)
        .out1(NONE, ALU1)
        .end(),
        inst(nop(), nop()),
    ]
}

#[test]
fn dma_read_fifoed_feeds_loads() {
    let (mut host, gpu_va) = dma_host(&[0x11, 0x22]);
    run(&fifoed_program(2), &[0x1_0001, 0x1000], &mut host).unwrap();
    assert_eq!(host.dma_reads, [(gpu_va, 2)]);
    assert_eq!(
        host.writes[host.writes.len() - 2..],
        [(0xC00, 0x11), (0xC01, 0x22)]
    );
}

#[test]
fn dma_read_fifoed_underflow() {
    let (mut host, _) = dma_host(&[0x11, 0x22]);
    let error = run(&fifoed_program(1), &[0x1_0001, 0x1000], &mut host).unwrap_err();
    assert_eq!(error.kind, MacroErrorKind::FifoUnderflow);
    assert_eq!(error.pc, 3);
}

#[test]
fn errors_never_panic() {
    let end = inst(nop(), nop());
    let cases = [
        (
            inst(alu(ADD, 0, 3, ZERO, 0), nop()),
            MacroErrorKind::UninitializedRegister,
        ),
        (
            inst(alu(ADD, 0, 30, ZERO, 0), nop()),
            MacroErrorKind::InvalidSource,
        ),
        (
            inst(alu(ADD, 25, ZERO, ZERO, 0), nop()),
            MacroErrorKind::InvalidDestination,
        ),
        (
            inst(nop(), nop()).out0(NONE, 7),
            MacroErrorKind::InvalidOutput,
        ),
        (
            inst(imm(0x0E00), imm(1)).out0(IMM0, IMM1),
            MacroErrorKind::MacroMethodEmit,
        ),
        (
            inst(alu(EXTENDED, ZERO, IMM, ZERO, 5), nop()),
            MacroErrorKind::UnsupportedExtended,
        ),
        (
            inst(alu(MERGE, 0, ZERO, ZERO, 40 << 10), nop()),
            MacroErrorKind::InvalidMerge,
        ),
        (
            inst(nop(), nop()).out0(NONE, OUT_LOAD0),
            MacroErrorKind::FifoUnderflow,
        ),
    ];
    for (instruction, kind) in cases {
        assert_eq!(error(&[instruction.end(), end], &[]), kind);
    }
    let spin = inst(alu(JAL, ZERO, ZERO, ZERO, 0x8000), nop());
    assert_eq!(error(&[spin], &[]), MacroErrorKind::StepLimit);

    let overflow = [
        inst(imm(0x0158), nop()).out0(IMM0, NONE),
        inst(alu(LOOP, ZERO, IMMPAIR, ZERO, 2), imm(10)),
        inst(nop(), imm(0x1000)).out0(NONE, IMM1),
        inst(nop(), nop()).end(),
        inst(nop(), nop()),
    ];
    assert_eq!(error(&overflow, &[]), MacroErrorKind::FifoOverflow);

    let mut engine = engine(&[inst(nop(), nop())]);
    let mut host = TestHost::default();
    let mut data_ram = MacroDataRam::new();
    let outside = engine
        .call(0x2AAA, &[], &mut data_ram, &mut host)
        .unwrap_err();
    assert_eq!(outside.kind, MacroErrorKind::OutsideCode);
    super::upload(&mut engine, 0x2AA9, &inst(nop(), nop()).encode());
    let run_off = engine
        .call(0x2AA9, &[], &mut data_ram, &mut host)
        .unwrap_err();
    assert_eq!(
        (run_off.kind, run_off.pc),
        (MacroErrorKind::OutsideCode, 0x2AAA)
    );
}
