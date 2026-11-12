use super::{Rng, TestHost, bind, fasm, masm, upload};
use crate::macros::{MACRO_METHOD_BASE, MacroDataRam, MacroEngine, MacroIsa};

/// Four MME64 macros: A depends on its parameters and data RAM words it reads (and writes data
/// RAM); B reads a register; C reads back a data RAM word it wrote; D takes a variable number
/// of parameters.
fn mme64_program() -> (Vec<u32>, [u32; 4]) {
    use masm::*;
    let program = [
        inst(
            alu(ADD, 0, LOAD0, ZERO, 0x1100),
            alu(ADD, 1, LOAD1, ZERO, 0),
        )
        .out0(IMM0, NONE),
        inst(alu(AND, 2, 0, IMM, 0xF), alu(AND, 3, 1, IMM, 0x7)),
        inst(alu(DREAD, 4, 2, ZERO, 0), nop()),
        inst(alu(ADD, 5, 4, 1, 0), nop()).out0(NONE, ALU0),
        inst(
            alu(DWRITE, ZERO, 3, 5, 0),
            alu(BEQ, ZERO, 4, ZERO, branch_imm(true, false, 2)),
        ),
        inst(imm(0x77), nop()).out0(NONE, IMM0),
        inst(alu(SLTU, 6, 0, 1, 0), nop()).out0(NONE, ALU0).end(),
        inst(nop(), nop()),
        inst(alu(STATE, 0, LOAD0, ZERO, 0), imm(0x200)).out0(IMM1, NONE),
        inst(alu(ADD, 1, 0, ZERO, 0), nop()).out0(NONE, ALU0).end(),
        inst(nop(), nop()),
        inst(alu(DWRITE, ZERO, IMM, LOAD0, 0x40), imm(0x300)).out0(IMM1, NONE),
        inst(alu(DREAD, 0, IMM, ZERO, 0x40), nop()),
        inst(alu(ADD, 1, 0, ZERO, 0), nop()).out0(NONE, ALU0).end(),
        inst(nop(), nop()),
        inst(alu(ADD, 0, LOAD0, ZERO, 0), imm(0x1400)).out0(IMM1, NONE),
        inst(alu(LOOP, ZERO, 0, ZERO, 2), nop()),
        inst(nop(), nop()).out0(NONE, OUT_LOAD0),
        inst(nop(), nop()).end(),
        inst(nop(), nop()),
    ];
    (assemble(&program), [0, 8, 11, 15])
}

/// Two Fermi macros: one that depends only on its parameters, one that reads a register.
fn fermi_program() -> (Vec<u32>, [u32; 4]) {
    use fasm::*;
    let code = vec![
        set_method(0x100, 1),
        addi(IGNORE_FETCH, 2, 0, 0),
        branch(true, true, 1, 3),
        alu(ADD, MOVE_SEND, 3, 1, 2),
        addi(MOVE_SEND, 0, 3, 7),
        alu(XOR, MOVE_SEND, 0, 1, 2),
        exit(addi(MOVE_SEND, 0, 2, 1)),
        nop(),
        set_method(0x200, 0),
        read(MOVE_SEND, 0, 1, 0x100),
        exit(nop()),
        nop(),
    ];
    (code, [0, 8, 0, 8])
}

fn parameters(rng: &mut Rng, isa: MacroIsa, kind: usize, out: &mut Vec<u32>) {
    out.clear();
    match (isa, kind) {
        (MacroIsa::Mme64, 0) => out.extend([rng.below(24), rng.below(12)]),
        (MacroIsa::Mme64, 1) => out.push(0x100 + rng.below(4)),
        (MacroIsa::Mme64, 2) => out.push(rng.below(4)),
        (MacroIsa::Mme64, _) => {
            let count = rng.below(4);
            out.push(count);
            out.extend((0..count).map(|_| rng.below(3)));
        }
        (MacroIsa::Fermi, 0 | 2) => out.extend([rng.below(3), rng.below(3)]),
        (MacroIsa::Fermi, _) => out.push(rng.below(3)),
    }
}

/// Drives the same random calls through a memo engine and a plain one and checks that the
/// emitted writes and the data RAM stay identical.
fn differential(isa: MacroIsa, memo_words: Option<usize>) -> MacroEngine {
    let (code, starts) = match isa {
        MacroIsa::Fermi => fermi_program(),
        MacroIsa::Mme64 => mme64_program(),
    };
    let mut engines = [MacroEngine::new(isa), MacroEngine::new(isa)];
    for engine in &mut engines {
        upload(engine, 0, &code);
        bind(engine, &starts);
    }
    engines[1].set_memo_enabled(false);
    if let Some(words) = memo_words {
        engines[0].set_memo_capacity(words);
    }
    let mut hosts = [TestHost::default(), TestHost::default()];
    let mut data_rams = [MacroDataRam::new(), MacroDataRam::new()];
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut call = Vec::new();
    for _ in 0..20_000 {
        if rng.below(64) == 0 {
            let (address, value) = (rng.below(16) as usize, rng.below(3));
            for data_ram in &mut data_rams {
                data_ram.words_mut()[address] = value;
            }
        }
        let kind = rng.below(4) as usize;
        let kind = if rng.below(3) == 0 { kind } else { 0 };
        parameters(&mut rng, isa, kind, &mut call);
        let method = MACRO_METHOD_BASE + 2 * kind as u32;
        let results: Vec<_> = (0..2)
            .map(|i| engines[i].execute(method, &call, &mut data_rams[i], &mut hosts[i]))
            .collect();
        assert_eq!(results[0], results[1], "call {call:?} of macro {kind}");
        assert_eq!(hosts[0].writes.len(), hosts[1].writes.len());
        assert_eq!(data_rams[0].words()[..0x48], data_rams[1].words()[..0x48]);
    }
    assert_eq!(hosts[0].writes, hosts[1].writes);
    assert_eq!(data_rams[0].words(), data_rams[1].words());
    assert_eq!(engines[1].stats().replayed, 0);
    let [memo, _] = engines;
    memo
}

#[test]
fn mme64_memo_matches_interpreter() {
    let stats = differential(MacroIsa::Mme64, None).stats();
    assert!(stats.replayed > 4_000, "{stats:?}");
    assert!(stats.memo.slot_hits > 0, "{stats:?}");
    assert!(stats.memo.guard_failures > 0, "{stats:?}");
    assert_eq!(stats.failed, 0);
}

#[test]
fn fermi_memo_matches_interpreter() {
    let stats = differential(MacroIsa::Fermi, None).stats();
    assert!(stats.replayed > 10_000, "{stats:?}");
    assert_eq!(stats.failed, 0);
}

#[test]
fn small_memo_compacts_and_stays_correct() {
    let stats = differential(MacroIsa::Mme64, Some(256)).stats();
    assert!(stats.memo.compactions > 0, "{stats:?}");
    assert!(stats.memo.arena_words <= 256, "{stats:?}");
    assert!(stats.replayed > 0, "{stats:?}");
}

#[test]
fn memo_switch_turns_replay_off() {
    let (code, starts) = mme64_program();
    let mut engine = MacroEngine::new(MacroIsa::Mme64);
    upload(&mut engine, 0, &code);
    bind(&mut engine, &starts);
    let mut host = TestHost::default();
    let mut data_ram = MacroDataRam::new();
    for _ in 0..3 {
        engine
            .execute(MACRO_METHOD_BASE, &[3, 4], &mut data_ram, &mut host)
            .unwrap();
    }
    assert_eq!(engine.stats().replayed, 2);
    engine.set_memo_enabled(false);
    assert!(!engine.memo_enabled());
    for _ in 0..3 {
        engine
            .execute(MACRO_METHOD_BASE, &[3, 4], &mut data_ram, &mut host)
            .unwrap();
    }
    assert_eq!(engine.stats().replayed, 2);
    assert_eq!(engine.stats().interpreted, 4);
}

fn call_once(engine: &mut MacroEngine, parameters: &[u32]) -> Vec<(u32, u32)> {
    let mut host = TestHost::default();
    engine
        .execute(
            MACRO_METHOD_BASE,
            parameters,
            &mut MacroDataRam::new(),
            &mut host,
        )
        .unwrap();
    host.writes
}

#[test]
fn mme64_upload_invalidates_decode_and_memo() {
    use masm::*;
    let emit = |value| {
        assemble(&[
            inst(imm(0x100), imm(value))
                .out0(IMM0, NONE)
                .out1(NONE, IMM1)
                .end(),
            inst(nop(), nop()),
        ])
    };
    let mut engine = MacroEngine::new(MacroIsa::Mme64);
    upload(&mut engine, 0, &emit(5));
    bind(&mut engine, &[0]);
    assert_eq!(call_once(&mut engine, &[]), [(0x100, 5)]);
    assert_eq!(call_once(&mut engine, &[]), [(0x100, 5)]);
    assert_eq!(engine.stats().replayed, 1);

    let generation = engine.code_generation();
    let changed = emit(6);
    assert_eq!(changed[1..], emit(5)[1..]);
    upload(&mut engine, 0, &changed[..1]);
    assert!(engine.code_generation() > generation);
    assert_eq!(call_once(&mut engine, &[]), [(0x100, 6)]);
    assert_eq!(engine.stats().replayed, 1);
    assert_eq!(engine.stats().memo.clears, 1);

    let generation = engine.code_generation();
    upload(&mut engine, 0, &changed);
    assert_eq!(engine.code_generation(), generation);
    assert_eq!(call_once(&mut engine, &[]), [(0x100, 6)]);
    assert_eq!(engine.stats().replayed, 2);
}

#[test]
fn fermi_upload_invalidates_decode_and_memo() {
    use fasm::*;
    let program = |value| [set_method(0x100, 0), exit(send_imm(value)), nop()];
    let mut engine = MacroEngine::new(MacroIsa::Fermi);
    upload(&mut engine, 0, &program(1));
    bind(&mut engine, &[0]);
    assert_eq!(call_once(&mut engine, &[0]), [(0x100, 1)]);
    assert_eq!(call_once(&mut engine, &[0]), [(0x100, 1)]);
    assert_eq!(engine.stats().replayed, 1);
    upload(&mut engine, 1, &program(2)[1..2]);
    assert_eq!(call_once(&mut engine, &[0]), [(0x100, 2)]);
    assert_eq!(call_once(&mut engine, &[0]), [(0x100, 2)]);
    assert_eq!(engine.stats().replayed, 2);
    assert_eq!(engine.stats().interpreted, 2);
}

#[test]
fn isa_for_class() {
    assert_eq!(MacroIsa::for_class(0xB197), MacroIsa::Fermi);
    assert_eq!(MacroIsa::for_class(0xB1C0), MacroIsa::Fermi);
    assert_eq!(MacroIsa::for_class(0xC597), MacroIsa::Mme64);
    assert_eq!(MacroIsa::for_class(0xC797), MacroIsa::Mme64);
    assert_eq!(MacroIsa::for_class(0xC7C0), MacroIsa::Mme64);
}
