//! Hand-assembled macros for both ISAs against a recording host.

mod fermi;
mod memo;
mod mme64;

use std::collections::HashMap;

use super::{
    LOAD_MME_INSTRUCTION_RAM, LOAD_MME_INSTRUCTION_RAM_POINTER, LOAD_MME_START_ADDRESS_RAM,
    LOAD_MME_START_ADDRESS_RAM_POINTER, MacroEngine, MacroHost, MacroIsa,
};

/// A register mirror that records every method write and serves DMA reads from a word map.
#[derive(Default)]
struct TestHost {
    registers: HashMap<u32, u32>,
    writes: Vec<(u32, u32)>,
    register_reads: u32,
    memory: HashMap<u64, u32>,
    dma_reads: Vec<(u64, usize)>,
}

impl MacroHost for TestHost {
    fn read_register(&mut self, method: u32) -> u32 {
        self.register_reads += 1;
        self.registers.get(&method).copied().unwrap_or(0)
    }

    fn write_method(&mut self, method: u32, argument: u32) {
        self.registers.insert(method, argument);
        self.writes.push((method, argument));
    }

    fn dma_read(&mut self, gpu_va: u64, out: &mut [u32]) {
        self.dma_reads.push((gpu_va, out.len()));
        for (i, word) in out.iter_mut().enumerate() {
            *word = self
                .memory
                .get(&(gpu_va + 4 * i as u64))
                .copied()
                .unwrap_or(0);
        }
    }
}

/// Uploads `code` at instruction `at` through the LOAD_MME methods.
fn upload(engine: &mut MacroEngine, at: u32, code: &[u32]) {
    assert!(engine.write_load_mme(LOAD_MME_INSTRUCTION_RAM_POINTER, at));
    for &word in code {
        assert!(engine.write_load_mme(LOAD_MME_INSTRUCTION_RAM, word));
    }
}

/// Binds macro slots 0.. to `starts`.
fn bind(engine: &mut MacroEngine, starts: &[u32]) {
    assert!(engine.write_load_mme(LOAD_MME_START_ADDRESS_RAM_POINTER, 0));
    for &start in starts {
        assert!(engine.write_load_mme(LOAD_MME_START_ADDRESS_RAM, start));
    }
}

fn engine_with(isa: MacroIsa, code: &[u32]) -> MacroEngine {
    let mut engine = MacroEngine::new(isa);
    upload(&mut engine, 0, code);
    bind(&mut engine, &[0]);
    engine
}

/// A tiny xorshift generator for reproducible random calls.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    fn below(&mut self, bound: u32) -> u32 {
        self.next() % bound
    }
}

/// Fermi assembler.
mod fasm {
    pub const ADD: u32 = 0;
    pub const ADDC: u32 = 1;
    pub const SUB: u32 = 2;
    pub const SUBB: u32 = 3;
    pub const XOR: u32 = 8;
    pub const OR: u32 = 9;
    pub const AND: u32 = 10;
    pub const ANDNOT: u32 = 11;
    pub const NAND: u32 = 12;

    pub const IGNORE_FETCH: u32 = 0;
    pub const MOVE: u32 = 1;
    pub const MOVE_SET_METHOD: u32 = 2;
    pub const FETCH_SEND: u32 = 3;
    pub const MOVE_SEND: u32 = 4;
    pub const FETCH_SET_METHOD: u32 = 5;
    pub const MOVE_SET_METHOD_FETCH_SEND: u32 = 6;
    pub const MOVE_SET_METHOD_SEND: u32 = 7;

    fn base(operation: u32, result: u32, dst: u32, a: u32, b: u32) -> u32 {
        operation | (result << 4) | (dst << 8) | (a << 11) | (b << 14)
    }

    pub fn alu(op: u32, result: u32, dst: u32, a: u32, b: u32) -> u32 {
        base(0, result, dst, a, b) | (op << 17)
    }

    pub fn addi(result: u32, dst: u32, a: u32, imm: i32) -> u32 {
        base(1, result, dst, a, 0) | ((imm as u32 & 0x3_FFFF) << 14)
    }

    pub fn insert(
        result: u32,
        dst: u32,
        a: u32,
        b: u32,
        src_bit: u32,
        size: u32,
        dst_bit: u32,
    ) -> u32 {
        base(2, result, dst, a, b) | (src_bit << 17) | (size << 22) | (dst_bit << 27)
    }

    pub fn extract_shl_imm(result: u32, dst: u32, a: u32, b: u32, size: u32, dst_bit: u32) -> u32 {
        base(3, result, dst, a, b) | (size << 22) | (dst_bit << 27)
    }

    pub fn extract_shl_reg(result: u32, dst: u32, a: u32, b: u32, src_bit: u32, size: u32) -> u32 {
        base(4, result, dst, a, b) | (src_bit << 17) | (size << 22)
    }

    pub fn read(result: u32, dst: u32, a: u32, imm: i32) -> u32 {
        base(5, result, dst, a, 0) | ((imm as u32 & 0x3_FFFF) << 14)
    }

    pub fn branch(if_zero: bool, annul: bool, a: u32, offset: i32) -> u32 {
        7 | (u32::from(!if_zero) << 4)
            | (u32::from(annul) << 5)
            | (a << 11)
            | ((offset as u32 & 0x3_FFFF) << 14)
    }

    pub fn exit(word: u32) -> u32 {
        word | 0x80
    }

    pub fn nop() -> u32 {
        addi(MOVE, 0, 0, 0)
    }

    pub fn set_method(method: u32, increment: u32) -> u32 {
        addi(MOVE_SET_METHOD, 0, 0, (method | (increment << 12)) as i32)
    }

    /// Sends `value` (an 18-bit signed immediate) to the current method.
    pub fn send_imm(value: i32) -> u32 {
        addi(MOVE_SEND, 0, 0, value)
    }
}

/// MME64 assembler: one instruction is two ALU slots, two output slots and a predicate.
mod masm {
    pub const ADD: u8 = 0;
    pub const ADDC: u8 = 1;
    pub const SUB: u8 = 2;
    pub const SUBB: u8 = 3;
    pub const MUL: u8 = 4;
    pub const MULH: u8 = 5;
    pub const MULU: u8 = 6;
    pub const EXTENDED: u8 = 7;
    pub const CLZ: u8 = 8;
    pub const SLL: u8 = 9;
    pub const SRL: u8 = 10;
    pub const SRA: u8 = 11;
    pub const AND: u8 = 12;
    pub const NAND: u8 = 13;
    pub const OR: u8 = 14;
    pub const XOR: u8 = 15;
    pub const MERGE: u8 = 16;
    pub const SLT: u8 = 17;
    pub const SLTU: u8 = 18;
    pub const SLE: u8 = 19;
    pub const SLEU: u8 = 20;
    pub const SEQ: u8 = 21;
    pub const STATE: u8 = 22;
    pub const LOOP: u8 = 23;
    pub const JAL: u8 = 24;
    pub const BLT: u8 = 25;
    pub const BEQ: u8 = 29;
    pub const DREAD: u8 = 30;
    pub const DWRITE: u8 = 31;

    pub const ZERO: u8 = 24;
    pub const IMM: u8 = 25;
    pub const IMMPAIR: u8 = 26;
    pub const IMM32: u8 = 27;
    pub const LOAD0: u8 = 28;
    pub const LOAD1: u8 = 29;

    pub const NONE: u8 = 0;
    pub const ALU0: u8 = 1;
    pub const ALU1: u8 = 2;
    pub const OUT_LOAD0: u8 = 3;
    pub const OUT_LOAD1: u8 = 4;
    pub const IMM0: u8 = 5;
    pub const IMM1: u8 = 6;
    pub const IMMHIGH0: u8 = 8;
    pub const OUT_IMM32: u8 = 10;

    #[derive(Clone, Copy)]
    pub struct Alu {
        op: u8,
        dst: u8,
        a: u8,
        b: u8,
        imm: u16,
    }

    /// `dst = a op b` with the slot's 16-bit immediate.
    pub fn alu(op: u8, dst: u8, a: u8, b: u8, imm: u16) -> Alu {
        Alu { op, dst, a, b, imm }
    }

    /// An ALU slot that does nothing but carry `imm` for the outputs.
    pub fn imm(imm: u16) -> Alu {
        alu(ADD, ZERO, ZERO, ZERO, imm)
    }

    pub fn nop() -> Alu {
        imm(0)
    }

    /// A branch immediate: taken when the condition is `when`, `offset` instructions away.
    pub fn branch_imm(when: bool, cancel_end: bool, offset: i32) -> u16 {
        (u16::from(when) << 15) | (u16::from(cancel_end) << 14) | (offset as u16 & 0x1FFF)
    }

    #[derive(Clone, Copy)]
    pub struct Inst {
        end: bool,
        pred_mode: u8,
        pred_reg: u8,
        alu: [Alu; 2],
        out: [(u8, u8); 2],
    }

    pub fn inst(alu0: Alu, alu1: Alu) -> Inst {
        Inst {
            end: false,
            pred_mode: 0,
            pred_reg: 0,
            alu: [alu0, alu1],
            out: [(NONE, NONE); 2],
        }
    }

    impl Inst {
        pub fn out0(mut self, method: u8, emit: u8) -> Self {
            self.out[0] = (method, emit);
            self
        }

        pub fn out1(mut self, method: u8, emit: u8) -> Self {
            self.out[1] = (method, emit);
            self
        }

        pub fn end(mut self) -> Self {
            self.end = true;
            self
        }

        pub fn pred(mut self, mode: u8, register: u8) -> Self {
            self.pred_mode = mode;
            self.pred_reg = register;
            self
        }

        /// The three code words in upload order (bits 95:64 first).
        pub fn encode(&self) -> [u32; 3] {
            let mut bits = u128::from(self.end)
                | (u128::from(self.pred_mode) << 1)
                | (u128::from(self.pred_reg) << 5);
            for (lane, alu) in self.alu.iter().enumerate() {
                let base = 10 + 36 * lane as u32;
                bits |= u128::from(alu.op) << base;
                bits |= u128::from(alu.dst) << (base + 5);
                bits |= u128::from(alu.a) << (base + 10);
                bits |= u128::from(alu.b) << (base + 15);
                bits |= u128::from(alu.imm) << (base + 20);
            }
            for (lane, (method, emit)) in self.out.iter().enumerate() {
                let base = 82 + 7 * lane as u32;
                bits |= u128::from(*method) << base;
                bits |= u128::from(*emit) << (base + 3);
            }
            [(bits >> 64) as u32, (bits >> 32) as u32, bits as u32]
        }
    }

    pub fn assemble(program: &[Inst]) -> Vec<u32> {
        program.iter().flat_map(Inst::encode).collect()
    }
}
