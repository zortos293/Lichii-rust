//! The at-exit handler NVN2 installs around geometry programs (C++ `geometry_exit.h`).
//!
//! The program opens with five setup instructions that load the handler's address and install
//! it with BMOV at-exit; every EXIT before the handler then runs it. The handler only
//! finalizes the output (`OUT RZ, R0, RZ`, which emits and cuts nothing) and exits, so once
//! the envelope is proven, translation skips the setup and lets each EXIT end the program.
//!
//! ```text
//! 0  LEPC R0
//! 1  MOV R3, handler
//! 2  IADD3 R0, P0, R0, R3, RZ
//! 3  IADD3.X R1, R1, RZ, RZ, P0, !PT   (or IMAD.MOV.U32 R1, RZ, RZ, R1)
//! 4  BMOV.32 ATEXIT_PC, R0
//!    body: no convergence-barrier state, no branch out of [5, handler)
//!    EXIT (or RET.REL to the program start)
//! handler:
//!    OUT RZ, R0, RZ
//!    BMOV.32 B?, ATEXIT_PC
//!    EXIT
//! ```

use super::instruction::{INSTRUCTION_SIZE, Instruction};

/// Instructions in front of the body that only install the handler.
pub const SETUP_INSTRUCTIONS: u32 = 5;

/// Why a program that opens with LEPC is not a geometry exit envelope: the offending
/// instruction's index and the reference's reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotAnEnvelope {
    pub index: usize,
    pub reason: &'static str,
}

/// Bits of the fourth word the envelope's instructions must leave clear: everything but the
/// scheduling controls.
const MODE_MASK: u32 = 0xc000_01ff;
/// Instructions a program may hold.
const MAX_INSTRUCTIONS: usize = 65536;

/// Proves the program `read` returns instruction by index is an envelope and returns the
/// handler's byte offset.
pub fn analyze(read: impl Fn(usize) -> Instruction) -> Result<u32, NotAnEnvelope> {
    let setup = SETUP_INSTRUCTIONS as usize;
    let fail = |index: usize, reason| Err(NotAnEnvelope { index, reason });
    let target = word(read(1), 1);
    let handler = (target / INSTRUCTION_SIZE) as usize;
    if target % INSTRUCTION_SIZE != 0 || handler <= setup + 1 || handler + 3 > MAX_INSTRUCTIONS {
        return fail(1, "native exit handler location");
    }
    let code: Vec<Instruction> = (0..handler + 3).map(read).collect();
    let is = |index: usize, words: [u32; 3]| {
        let inst = code[index];
        words.iter().zip(0..).all(|(&w, k)| word(inst, k) == w) && word(inst, 3) & MODE_MASK == 0
    };
    let bits = |index: usize, offset: u32, count: u32| code[index].bits(offset, count) as u32;
    let is_exit = |index: usize| {
        let inst = code[index];
        word(inst, 0) & !0xf000 == 0x94d
            && word(inst, 1) == 0
            && word(inst, 2) == 0x0380_0000
            && word(inst, 3) & MODE_MASK == 0
            && bits(index, 12, 4) != 15
    };
    let is_return = |index: usize| {
        let inst = code[index];
        word(inst, 0) & 0x00ff_ffff == 0x7950
            && bits(index, 24, 8) < 249
            && word(inst, 2) & !0x3ffff == 0x03c0_0000
            && word(inst, 3) & MODE_MASK == 0
            && (index as i64 + 1) * i64::from(INSTRUCTION_SIZE) + relative_target(inst) == 0
    };

    if !is(0, [0x734e, 0, 0])
        || !is(1, [0x37802, target, 0xf00])
        || !is(2, [0x7210, 3, 0x07f1_e0ff])
        || !(is(3, [0x0101_7210, 0xff, 0x007f_e4ff]) || is(3, [0xff01_7224, 0xff, 0x000e_0401]))
        || !is(4, [0x7357, 0, 0])
    {
        return fail(0, "native exit handler installation");
    }
    if !is(handler, [0x00ff_7324, 0xff, 0])
        || !is(handler + 1, [0x7957, 0, 0])
        || !is(handler + 2, [0x794d, 0, 0x0380_0000])
    {
        return fail(handler, "native exit handler body");
    }
    if !(is_exit(handler - 1) && bits(handler - 1, 12, 4) == 7) && !is_return(handler - 1) {
        return fail(handler - 1, "native exit handler reached by fallthrough");
    }

    for index in setup..handler {
        let opcode = bits(index, 0, 12);
        let family = opcode & 0x1ff;
        if family == 0x14e || (0x154..=0x157).contains(&family) {
            return fail(index, "native exit handler state access in body");
        }
        if opcode == 0x94d {
            if !is_exit(index) {
                return fail(index, "native body EXIT mode");
            }
            continue;
        }
        if is_return(index)
            || !(0x140..0x160).contains(&family)
            || matches!(opcode, 0x941 | 0x942 | 0x948)
        {
            continue;
        }
        let inst = code[index];
        let noinc_call = opcode == 0x944 && word(inst, 2) & 0x40_0000 != 0;
        if opcode != 0x945 && opcode != 0x947 && !noinc_call {
            return fail(index, "native control flow in exit envelope body");
        }
        let relative = if opcode == 0x945 {
            i64::from(word(inst, 1) as i32)
        } else {
            relative_target(inst)
        };
        let size = i64::from(INSTRUCTION_SIZE);
        let destination = (index as i64 + 1) * size + relative;
        if destination % size != 0
            || destination < setup as i64 * size
            || destination >= handler as i64 * size
        {
            return fail(index, "native branch leaves exit envelope body");
        }
    }

    let (mut live_registers, mut live_predicates) = (0b1011_u32, 0b1_u32);
    let mut index = setup;
    while live_registers != 0 || live_predicates != 0 {
        if index >= code.len() {
            return fail(index, "native instruction before exit setup state is dead");
        }
        let opcode = bits(index, 0, 12);
        let family = opcode & 0x1ff;
        let form = (opcode >> 9) & 7;
        let guard = bits(index, 12, 4);
        let (mut reads, mut read_predicates, mut writes, mut write_predicates) = (0, 0, 0, 0);
        let mut read_register = |reg: u32| {
            if reg < 32 {
                reads |= 1 << reg;
            }
        };
        let mut write_register = |reg: u32| {
            if reg < 32 {
                writes |= 1 << reg;
            }
        };
        let mut read_predicate = |pred: u32| {
            if pred < 7 {
                read_predicates |= 1 << pred;
            }
        };
        let mut write_predicate = |pred: u32| {
            if pred < 7 {
                write_predicates |= 1 << pred;
            }
        };
        let read_sources = |read_register: &mut dyn FnMut(u32), has_c: bool| {
            read_register(bits(index, 24, 8));
            if form == 1 {
                read_register(bits(index, 32, 8));
            }
            if matches!(form, 2 | 3 | 7) {
                read_register(bits(index, 64, 8));
            }
            if has_c && matches!(form, 1 | 4 | 5 | 6) {
                read_register(bits(index, 64, 8));
            }
        };
        read_predicate(guard & 7);
        if is_exit(index) {
            if guard != 7 {
                return fail(index, "native instruction before exit setup state is dead");
            }
            if live_registers & 1 != 0 || read_predicates & live_predicates != 0 {
                return fail(index, "native exit handler reads exit setup state");
            }
            break;
        }
        match opcode {
            0x919 | 0x923 => {
                if opcode == 0x923 {
                    read_register(bits(index, 24, 8));
                }
                write_register(bits(index, 16, 8));
            }
            0x805 | 0xab9 | 0x945 => {}
            0x381 | 0x981 => {
                let (address, size) = (bits(index, 24, 8), bits(index, 73, 3));
                if opcode == 0x381 {
                    read_predicate(7 - bits(index, 64, 3));
                }
                if address != 255 {
                    read_register(address);
                    read_register(address + 1);
                }
                let results = match size {
                    0..5 => 1,
                    5 => 2,
                    6 => 4,
                    _ => 0,
                };
                for i in 0..results {
                    write_register(bits(index, 16, 8) + i);
                }
            }
            _ if family == 0x002 && matches!(form, 1 | 4 | 5 | 6) => {
                if form == 1 {
                    read_register(bits(index, 32, 8));
                }
                write_register(bits(index, 16, 8));
            }
            _ if matches!(family, 0x024 | 0x016) && form != 0 => {
                read_sources(&mut read_register, true);
                if family == 0x024 && bits(index, 74, 1) != 0 {
                    read_predicate(bits(index, 87, 3));
                }
                write_register(bits(index, 16, 8));
            }
            _ if family == 0x105 && matches!(form, 1 | 5 | 6) => {
                if form == 1 {
                    read_register(bits(index, 32, 8));
                }
                write_register(bits(index, 16, 8));
            }
            _ if family == 0x011 && form != 0 => {
                read_sources(&mut read_register, true);
                if bits(index, 74, 1) != 0 {
                    read_predicate(bits(index, 87, 3));
                }
                write_register(bits(index, 16, 8));
                write_predicate(bits(index, 81, 3));
            }
            0xb82 => {
                if bits(index, 24, 8) != 255 {
                    read_register(bits(index, 24, 8));
                }
                write_register(bits(index, 16, 8));
                if bits(index, 73, 3) == 5 {
                    write_register(bits(index, 16, 8) + 1);
                }
            }
            _ if family == 0x012 && form != 0 => {
                read_sources(&mut read_register, true);
                write_register(bits(index, 16, 8));
                write_predicate(bits(index, 81, 3));
            }
            0x321 => {
                read_register(bits(index, 24, 8));
                read_register(bits(index, 32, 8));
                for i in 0..=bits(index, 74, 2) {
                    write_register(bits(index, 16, 8) + i);
                }
            }
            0x324 | 0x924 => {
                read_sources(&mut read_register, false);
                write_register(bits(index, 16, 8));
            }
            0x322 => {
                let value = bits(index, 32, 8);
                if value != 255 {
                    for i in 0..=bits(index, 74, 2) {
                        read_register(value + i);
                    }
                }
                read_register(bits(index, 64, 8));
            }
            _ if family == 0x00c && form != 0 && bits(index, 72, 1) == 0 => {
                read_sources(&mut read_register, false);
                read_predicate(bits(index, 87, 3));
                write_predicate(bits(index, 81, 3));
                write_predicate(bits(index, 84, 3));
            }
            _ => return fail(index, "native instruction before exit setup state is dead"),
        }
        if reads & live_registers != 0 || read_predicates & live_predicates != 0 {
            return fail(index, "native body reads exit setup state");
        }
        if guard == 7 {
            live_registers &= !writes;
            live_predicates &= !write_predicates;
        }
        index += 1;
    }
    Ok(target)
}

fn word(inst: Instruction, k: u32) -> u32 {
    (inst.0 >> (32 * k)) as u32
}

/// The signed 50-bit offset BRA, CALL and RET carry in bits 32..82.
fn relative_target(inst: Instruction) -> i64 {
    let raw = u64::from(word(inst, 1)) | (u64::from(word(inst, 2) & 0x3ffff) << 32);
    ((raw << 14) as i64) >> 14
}

/// `body` inside the envelope, followed by its EXIT and the handler.
#[cfg(test)]
pub(crate) fn envelope(body: &[Instruction]) -> Vec<Instruction> {
    let inst = |w0: u32, w1: u32, w2: u32| {
        Instruction(u128::from(w0) | u128::from(w1) << 32 | u128::from(w2) << 64)
    };
    let handler = SETUP_INSTRUCTIONS + body.len() as u32 + 1;
    let mut code = vec![
        inst(0x734e, 0, 0),
        inst(0x37802, handler * INSTRUCTION_SIZE, 0xf00),
        inst(0x7210, 3, 0x07f1_e0ff),
        inst(0x0101_7210, 0xff, 0x007f_e4ff),
        inst(0x7357, 0, 0),
    ];
    code.extend_from_slice(body);
    code.extend([
        inst(0x794d, 0, 0x0380_0000),
        inst(0x00ff_7324, 0xff, 0),
        inst(0x7957, 0, 0),
        inst(0x794d, 0, 0x0380_0000),
    ]);
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    const WRITE_R0: Instruction = Instruction(0x7802);
    const READ_R0: Instruction = Instruction(0x0004_7202);

    fn analyze_code(code: &[Instruction]) -> Result<u32, NotAnEnvelope> {
        analyze(|index| code.get(index).copied().unwrap_or(Instruction(0)))
    }

    #[test]
    fn the_envelope_yields_its_handler() {
        let code = envelope(&[WRITE_R0]);
        assert_eq!(analyze_code(&code), Ok(7 * INSTRUCTION_SIZE));
    }

    #[test]
    fn broken_envelopes_are_refused_with_the_reference_reason() {
        let reason = |code: &[Instruction]| analyze_code(code).map_err(|e| (e.index, e.reason));
        let mut setup = envelope(&[WRITE_R0]);
        setup[4] = Instruction(0x7918);
        assert_eq!(reason(&setup), Err((0, "native exit handler installation")));
        let mut handler = envelope(&[WRITE_R0]);
        handler[8] = Instruction(0x7918);
        assert_eq!(reason(&handler), Err((7, "native exit handler body")));
        let mut fallthrough = envelope(&[WRITE_R0]);
        fallthrough[6] = Instruction(0x7918);
        assert_eq!(
            reason(&fallthrough),
            Err((6, "native exit handler reached by fallthrough"))
        );
        assert_eq!(
            reason(&envelope(&[])),
            Err((1, "native exit handler location"))
        );
        let write_r4 = Instruction(0x0004_7802);
        assert_eq!(
            reason(&envelope(&[write_r4])),
            Err((6, "native exit handler reads exit setup state"))
        );
        assert_eq!(
            reason(&envelope(&[READ_R0, WRITE_R0])),
            Err((5, "native body reads exit setup state"))
        );
        let at_exit = Instruction(0x7357);
        assert_eq!(
            reason(&envelope(&[WRITE_R0, at_exit])),
            Err((6, "native exit handler state access in body"))
        );
    }
}
