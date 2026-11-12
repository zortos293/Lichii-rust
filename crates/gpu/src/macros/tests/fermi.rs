use super::fasm::*;
use super::{TestHost, engine_with};
use crate::macros::{MACRO_METHOD_BASE, MacroDataRam, MacroErrorKind, MacroIsa};

fn run(code: &[u32], parameters: &[u32]) -> (Result<(), crate::macros::MacroError>, TestHost) {
    let mut engine = engine_with(MacroIsa::Fermi, code);
    engine.set_memo_enabled(false);
    let mut host = TestHost::default();
    let result = engine.execute(
        MACRO_METHOD_BASE,
        parameters,
        &mut MacroDataRam::new(),
        &mut host,
    );
    (result, host)
}

fn writes(code: &[u32], parameters: &[u32]) -> Vec<(u32, u32)> {
    let (result, host) = run(code, parameters);
    result.expect("macro failed");
    host.writes
}

fn error(code: &[u32], parameters: &[u32]) -> MacroErrorKind {
    run(code, parameters).0.expect_err("macro succeeded").kind
}

#[test]
fn parameters_sends_and_exit_delay_slot() {
    let code = [
        set_method(0x100, 1),
        addi(MOVE_SEND, 0, 1, 0),
        addi(FETCH_SEND, 2, 0, 5),
        exit(addi(MOVE_SEND, 0, 2, 0)),
        addi(MOVE_SEND, 0, 2, 1),
        send_imm(0x777),
    ];
    assert_eq!(
        writes(&code, &[0xAA, 0xBB]),
        [(0x100, 0xAA), (0x101, 5), (0x102, 0xBB), (0x103, 0xBC)]
    );
}

#[test]
fn alu_operations_and_carry() {
    let code = [
        set_method(0x200, 1),
        addi(IGNORE_FETCH, 2, 0, 0),
        alu(ADD, MOVE_SEND, 3, 1, 2),
        alu(ADDC, MOVE_SEND, 0, 0, 0),
        alu(SUB, MOVE_SEND, 0, 2, 1),
        alu(SUBB, MOVE_SEND, 0, 2, 2),
        alu(SUB, MOVE_SEND, 0, 1, 2),
        alu(SUBB, MOVE_SEND, 0, 1, 2),
        alu(XOR, MOVE_SEND, 0, 1, 2),
        alu(OR, MOVE_SEND, 0, 1, 2),
        alu(AND, MOVE_SEND, 0, 1, 2),
        alu(ANDNOT, MOVE_SEND, 0, 1, 2),
        alu(NAND, MOVE_SEND, 0, 1, 2),
        addi(MOVE_SEND, 0, 3, 0),
        exit(nop()),
        nop(),
    ];
    let values: Vec<u32> = writes(&code, &[0xFFFF_FFFF, 1])
        .into_iter()
        .map(|(_, value)| value)
        .collect();
    assert_eq!(
        values,
        [
            0,
            1,
            2,
            0xFFFF_FFFF,
            0xFFFF_FFFE,
            0xFFFF_FFFE,
            0xFFFF_FFFE,
            0xFFFF_FFFF,
            1,
            0xFFFF_FFFE,
            0xFFFF_FFFE,
            0,
        ]
    );
}

#[test]
fn bitfield_operations() {
    let code = [
        set_method(0x300, 1),
        addi(IGNORE_FETCH, 2, 0, 0),
        addi(MOVE, 3, 0, 4),
        insert(MOVE_SEND, 0, 1, 2, 4, 8, 16),
        extract_shl_imm(MOVE_SEND, 0, 3, 2, 8, 8),
        extract_shl_reg(MOVE_SEND, 0, 3, 2, 8, 4),
        exit(nop()),
        nop(),
    ];
    assert_eq!(
        writes(&code, &[0xAABB_CCDD, 0x1234_5678]),
        [(0x300, 0xAA67_CCDD), (0x301, 0x6700), (0x302, 0x60)]
    );
}

#[test]
fn bitfield_positions_past_31_give_zero() {
    let code = [
        set_method(0x300, 1),
        addi(IGNORE_FETCH, 2, 0, 0),
        addi(MOVE, 3, 0, 32),
        addi(MOVE, 4, 0, 33),
        extract_shl_imm(MOVE_SEND, 0, 3, 2, 1, 0),
        extract_shl_reg(MOVE_SEND, 0, 3, 2, 0, 1),
        extract_shl_imm(MOVE_SEND, 0, 4, 2, 1, 0),
        extract_shl_reg(MOVE_SEND, 0, 4, 2, 0, 1),
        exit(nop()),
        nop(),
    ];
    assert_eq!(
        writes(&code, &[0, 1]),
        [(0x300, 0), (0x301, 0), (0x302, 0), (0x303, 0)]
    );
}

#[test]
fn result_operations_set_method_and_increment() {
    let code = [
        addi(MOVE_SET_METHOD_SEND, 0, 0, 0x50A0),
        addi(FETCH_SET_METHOD, 2, 0, 0x2400),
        addi(MOVE_SEND, 0, 2, 0),
        addi(MOVE_SET_METHOD_FETCH_SEND, 3, 0, 0x0410),
        addi(MOVE_SEND, 0, 3, 0),
        exit(nop()),
        nop(),
    ];
    assert_eq!(
        writes(&code, &[0, 0x11, 0x22]),
        [(0x0A0, 5), (0x400, 0x11), (0x410, 0x22), (0x410, 0x410)]
    );
}

#[test]
fn branch_delay_slot_and_annul() {
    let program = |annul| {
        [
            set_method(0x300, 1),
            branch(false, annul, 1, 3),
            send_imm(1),
            send_imm(2),
            exit(send_imm(3)),
            nop(),
        ]
    };
    let values = |code: &[u32], parameter| -> Vec<u32> {
        writes(code, &[parameter])
            .into_iter()
            .map(|(_, value)| value)
            .collect()
    };
    assert_eq!(values(&program(false), 1), [1, 3]);
    assert_eq!(values(&program(false), 0), [1, 2, 3]);
    assert_eq!(values(&program(true), 1), [3]);
    assert_eq!(values(&program(true), 0), [1, 2, 3]);
}

#[test]
fn backward_branch_loop() {
    let code = [
        set_method(0x400, 0),
        addi(MOVE_SEND, 1, 1, -1),
        branch(false, true, 1, -1),
        exit(nop()),
        nop(),
    ];
    assert_eq!(writes(&code, &[3]), [(0x400, 2), (0x400, 1), (0x400, 0)]);
}

#[test]
fn branch_if_zero() {
    let code = [
        set_method(0x400, 0),
        branch(true, true, 1, 2),
        send_imm(1),
        exit(send_imm(2)),
        nop(),
    ];
    assert_eq!(writes(&code, &[0]), [(0x400, 2)]);
    assert_eq!(writes(&code, &[5]), [(0x400, 1), (0x400, 2)]);
}

#[test]
fn register_read() {
    let code = [
        set_method(0x500, 0),
        read(MOVE_SEND, 2, 1, 3),
        exit(nop()),
        nop(),
    ];
    let mut engine = engine_with(MacroIsa::Fermi, &code);
    let mut host = TestHost::default();
    host.registers.insert(0x123, 0x55);
    let mut data_ram = MacroDataRam::new();
    for _ in 0..2 {
        engine
            .execute(MACRO_METHOD_BASE, &[0x120], &mut data_ram, &mut host)
            .unwrap();
    }
    assert_eq!(host.writes, [(0x500, 0x55), (0x500, 0x55)]);
    assert_eq!(host.register_reads, 2);
    assert_eq!(engine.stats().replayed, 0);
    assert_eq!(engine.stats().memo.stores, 0);
}

#[test]
fn errors_never_panic() {
    assert_eq!(
        error(&[addi(IGNORE_FETCH, 2, 0, 0), exit(nop()), nop()], &[1]),
        MacroErrorKind::FifoUnderflow
    );
    assert_eq!(
        error(&[branch(true, true, 0, 0)], &[1]),
        MacroErrorKind::StepLimit
    );
    assert_eq!(
        error(&[6, exit(nop()), nop()], &[1]),
        MacroErrorKind::InvalidInstruction
    );
    assert_eq!(
        error(&[alu(5, MOVE, 1, 1, 1), exit(nop()), nop()], &[1]),
        MacroErrorKind::InvalidInstruction
    );
    assert_eq!(
        error(&[exit(nop()), branch(true, false, 0, 0)], &[1]),
        MacroErrorKind::BranchInDelaySlot
    );
    assert_eq!(
        error(
            &[set_method(0xE00, 0), send_imm(1), exit(nop()), nop()],
            &[1]
        ),
        MacroErrorKind::MacroMethodEmit
    );

    let mut engine = engine_with(MacroIsa::Fermi, &[nop()]);
    let mut host = TestHost::default();
    let mut data_ram = MacroDataRam::new();
    let outside = engine
        .call(u32::MAX - 1, &[1], &mut data_ram, &mut host)
        .unwrap_err();
    assert_eq!(outside.kind, MacroErrorKind::OutsideCode);
    super::upload(&mut engine, 0x1FFE, &[nop(), nop()]);
    let run_off = engine
        .call(0x1FFE, &[1], &mut data_ram, &mut host)
        .unwrap_err();
    assert_eq!(run_off.kind, MacroErrorKind::OutsideCode);
    assert_eq!(run_off.pc, 0x2000);
    assert_eq!(engine.stats().failed, 2);
}

#[test]
fn macro_index_selects_start_address() {
    let mut engine = crate::macros::MacroEngine::new(MacroIsa::Fermi);
    super::upload(
        &mut engine,
        0x10,
        &[
            set_method(0x600, 0),
            exit(send_imm(1)),
            nop(),
            set_method(0x600, 0),
            exit(send_imm(2)),
            nop(),
        ],
    );
    super::bind(&mut engine, &[0x10, 0x13]);
    let mut host = TestHost::default();
    let mut data_ram = MacroDataRam::new();
    engine
        .execute(MACRO_METHOD_BASE + 2, &[0], &mut data_ram, &mut host)
        .unwrap();
    engine
        .execute(MACRO_METHOD_BASE, &[0], &mut data_ram, &mut host)
        .unwrap();
    assert_eq!(host.writes, [(0x600, 2), (0x600, 1)]);
}
