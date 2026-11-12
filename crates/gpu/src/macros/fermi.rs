//! The Fermi MME: 32-bit instructions, eight registers (r0 reads as zero), one parameter in r1
//! at entry, branches and exits with a delay slot.

use super::{CallRecord, Fault, MACRO_METHOD_BASE, MacroErrorKind, MacroHost};

/// Words of Fermi code RAM.
pub(super) const FERMI_CODE_WORDS: usize = 0x2000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operation {
    Alu(AluOperation),
    AddImmediate,
    ExtractInsert,
    ExtractShiftLeftImmediate,
    ExtractShiftLeftRegister,
    Read,
    Branch,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AluOperation {
    Add,
    AddWithCarry,
    Subtract,
    SubtractWithBorrow,
    Xor,
    Or,
    And,
    AndNot,
    Nand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResultOperation {
    IgnoreAndFetch,
    Move,
    MoveAndSetMethod,
    FetchAndSend,
    MoveAndSend,
    FetchAndSetMethod,
    MoveAndSetMethodFetchAndSend,
    MoveAndSetMethodSend,
}

/// One code word's fields, decoded when the word is uploaded.
#[derive(Debug, Clone, Copy)]
struct Instruction {
    operation: Operation,
    result: ResultOperation,
    dst: u8,
    src_a: u8,
    src_b: u8,
    src_bit: u8,
    dst_bit: u8,
    exit: bool,
    annul: bool,
    branch_if_zero: bool,
    mask: u32,
    immediate: u32,
}

impl Instruction {
    fn decode(word: u32) -> Self {
        let field = |low: u32, count: u32| (word >> low) & ((1 << count) - 1);
        let operation = match field(0, 3) {
            0 => match field(17, 5) {
                0 => Operation::Alu(AluOperation::Add),
                1 => Operation::Alu(AluOperation::AddWithCarry),
                2 => Operation::Alu(AluOperation::Subtract),
                3 => Operation::Alu(AluOperation::SubtractWithBorrow),
                8 => Operation::Alu(AluOperation::Xor),
                9 => Operation::Alu(AluOperation::Or),
                10 => Operation::Alu(AluOperation::And),
                11 => Operation::Alu(AluOperation::AndNot),
                12 => Operation::Alu(AluOperation::Nand),
                _ => Operation::Invalid,
            },
            1 => Operation::AddImmediate,
            2 => Operation::ExtractInsert,
            3 => Operation::ExtractShiftLeftImmediate,
            4 => Operation::ExtractShiftLeftRegister,
            5 => Operation::Read,
            7 => Operation::Branch,
            _ => Operation::Invalid,
        };
        let result = match field(4, 3) {
            0 => ResultOperation::IgnoreAndFetch,
            1 => ResultOperation::Move,
            2 => ResultOperation::MoveAndSetMethod,
            3 => ResultOperation::FetchAndSend,
            4 => ResultOperation::MoveAndSend,
            5 => ResultOperation::FetchAndSetMethod,
            6 => ResultOperation::MoveAndSetMethodFetchAndSend,
            _ => ResultOperation::MoveAndSetMethodSend,
        };
        Self {
            operation,
            result,
            dst: field(8, 3) as u8,
            src_a: field(11, 3) as u8,
            src_b: field(14, 3) as u8,
            src_bit: field(17, 5) as u8,
            dst_bit: field(27, 5) as u8,
            exit: field(7, 1) != 0,
            annul: field(5, 1) != 0,
            branch_if_zero: field(4, 1) == 0,
            mask: (1 << field(22, 5)) - 1,
            immediate: ((word as i32) >> 14) as u32,
        }
    }
}

/// Fermi code RAM and its decoded instructions, kept in step on every upload.
pub(super) struct FermiCode {
    words: Box<[u32]>,
    decoded: Box<[Instruction]>,
}

impl FermiCode {
    pub(super) fn new() -> Self {
        Self {
            words: vec![0; FERMI_CODE_WORDS].into_boxed_slice(),
            decoded: vec![Instruction::decode(0); FERMI_CODE_WORDS].into_boxed_slice(),
        }
    }

    /// Stores `word` at `address`; returns whether the code changed.
    pub(super) fn write(&mut self, address: u32, word: u32) -> bool {
        let Some(slot) = self.words.get_mut(address as usize) else {
            return false;
        };
        if *slot == word {
            return false;
        }
        *slot = word;
        self.decoded[address as usize] = Instruction::decode(word);
        true
    }
}

enum Flow {
    Next,
    Jump { target: u32, delay_slot: bool },
}

struct Machine<'a, H> {
    registers: [u32; 8],
    carry: bool,
    method: u32,
    increment: u32,
    parameters: &'a [u32],
    next_parameter: usize,
    host: &'a mut H,
    record: &'a mut CallRecord,
}

/// Runs the macro at `start`. The first parameter is in r1 at entry; the rest are fetched.
pub(super) fn run<H: MacroHost>(
    code: &FermiCode,
    start: u32,
    parameters: &[u32],
    host: &mut H,
    record: &mut CallRecord,
    step_limit: u32,
) -> Result<(), Fault> {
    let mut machine = Machine {
        registers: [0; 8],
        carry: false,
        method: 0,
        increment: 0,
        parameters,
        next_parameter: 1,
        host,
        record,
    };
    if let Some(&first) = parameters.first() {
        machine.registers[1] = first;
    } else {
        machine.next_parameter = 0;
    }
    let mut steps = 0u32;
    let mut fetch = |pc: u32| -> Result<Instruction, Fault> {
        if steps >= step_limit {
            return Err(Fault {
                kind: MacroErrorKind::StepLimit,
                pc,
            });
        }
        steps += 1;
        code.decoded.get(pc as usize).copied().ok_or(Fault {
            kind: MacroErrorKind::OutsideCode,
            pc,
        })
    };
    let mut pc = start;
    loop {
        let instruction = fetch(pc)?;
        let flow = machine
            .execute(&instruction, pc)
            .map_err(|kind| Fault { kind, pc })?;
        let (next, delay_slot, exit) = match flow {
            Flow::Jump { target, delay_slot } => (target, delay_slot, false),
            Flow::Next => (pc.wrapping_add(1), instruction.exit, instruction.exit),
        };
        if delay_slot {
            let slot = pc.wrapping_add(1);
            let delayed = fetch(slot)?;
            if delayed.operation == Operation::Branch {
                return Err(Fault {
                    kind: MacroErrorKind::BranchInDelaySlot,
                    pc: slot,
                });
            }
            machine
                .execute(&delayed, slot)
                .map_err(|kind| Fault { kind, pc: slot })?;
        }
        if exit {
            return Ok(());
        }
        pc = next;
    }
}

impl<H: MacroHost> Machine<'_, H> {
    #[inline]
    fn execute(&mut self, instruction: &Instruction, pc: u32) -> Result<Flow, MacroErrorKind> {
        let a = self.registers[instruction.src_a as usize];
        let b = self.registers[instruction.src_b as usize];
        let mask = instruction.mask;
        let result = match instruction.operation {
            Operation::Alu(operation) => self.alu(operation, a, b),
            Operation::AddImmediate => a.wrapping_add(instruction.immediate),
            Operation::ExtractInsert => {
                let dst_bit = instruction.dst_bit;
                (a & !(mask << dst_bit)) | (((b >> instruction.src_bit) & mask) << dst_bit)
            }
            Operation::ExtractShiftLeftImmediate if a > 31 => 0,
            Operation::ExtractShiftLeftImmediate => ((b >> a) & mask) << instruction.dst_bit,
            Operation::ExtractShiftLeftRegister if a > 31 => 0,
            Operation::ExtractShiftLeftRegister => ((b >> instruction.src_bit) & mask) << a,
            Operation::Read => {
                self.record.uncacheable();
                self.host
                    .read_register(a.wrapping_add(instruction.immediate))
            }
            Operation::Branch => {
                let taken = (a == 0) == instruction.branch_if_zero;
                return Ok(if taken {
                    Flow::Jump {
                        target: pc.wrapping_add(instruction.immediate),
                        delay_slot: !instruction.annul,
                    }
                } else {
                    Flow::Next
                });
            }
            Operation::Invalid => return Err(MacroErrorKind::InvalidInstruction),
        };
        self.process_result(instruction.result, instruction.dst, result)?;
        Ok(Flow::Next)
    }

    #[inline]
    fn alu(&mut self, operation: AluOperation, a: u32, b: u32) -> u32 {
        match operation {
            AluOperation::Add => {
                let (result, carry) = a.overflowing_add(b);
                self.carry = carry;
                result
            }
            AluOperation::AddWithCarry => {
                let sum = u64::from(a) + u64::from(b) + u64::from(self.carry);
                self.carry = sum > u64::from(u32::MAX);
                sum as u32
            }
            AluOperation::Subtract => {
                self.carry = a >= b;
                a.wrapping_sub(b)
            }
            AluOperation::SubtractWithBorrow => {
                let difference = u64::from(a)
                    .wrapping_sub(u64::from(b))
                    .wrapping_sub(u64::from(!self.carry));
                self.carry = difference <= u64::from(u32::MAX);
                difference as u32
            }
            AluOperation::Xor => a ^ b,
            AluOperation::Or => a | b,
            AluOperation::And => a & b,
            AluOperation::AndNot => a & !b,
            AluOperation::Nand => !(a & b),
        }
    }

    #[inline]
    fn process_result(
        &mut self,
        operation: ResultOperation,
        dst: u8,
        result: u32,
    ) -> Result<(), MacroErrorKind> {
        match operation {
            ResultOperation::IgnoreAndFetch => {
                let parameter = self.fetch_parameter()?;
                self.set_register(dst, parameter);
            }
            ResultOperation::Move => self.set_register(dst, result),
            ResultOperation::MoveAndSetMethod => {
                self.set_register(dst, result);
                self.set_method(result);
            }
            ResultOperation::FetchAndSend => {
                let parameter = self.fetch_parameter()?;
                self.set_register(dst, parameter);
                self.send(result)?;
            }
            ResultOperation::MoveAndSend => {
                self.set_register(dst, result);
                self.send(result)?;
            }
            ResultOperation::FetchAndSetMethod => {
                let parameter = self.fetch_parameter()?;
                self.set_register(dst, parameter);
                self.set_method(result);
            }
            ResultOperation::MoveAndSetMethodFetchAndSend => {
                self.set_register(dst, result);
                self.set_method(result);
                let parameter = self.fetch_parameter()?;
                self.send(parameter)?;
            }
            ResultOperation::MoveAndSetMethodSend => {
                self.set_register(dst, result);
                self.set_method(result);
                self.send((result >> 12) & 0x3F)?;
            }
        }
        Ok(())
    }

    #[inline]
    fn set_register(&mut self, register: u8, value: u32) {
        if register != 0 {
            self.registers[register as usize] = value;
        }
    }

    #[inline]
    fn set_method(&mut self, value: u32) {
        self.method = value & 0xFFF;
        self.increment = (value >> 12) & 0x3F;
    }

    #[inline]
    fn fetch_parameter(&mut self) -> Result<u32, MacroErrorKind> {
        let parameter = *self
            .parameters
            .get(self.next_parameter)
            .ok_or(MacroErrorKind::FifoUnderflow)?;
        self.next_parameter += 1;
        Ok(parameter)
    }

    #[inline]
    fn send(&mut self, value: u32) -> Result<(), MacroErrorKind> {
        if self.method >= MACRO_METHOD_BASE {
            return Err(MacroErrorKind::MacroMethodEmit);
        }
        self.record.write(self.method, value);
        self.host.write_method(self.method, value);
        self.method = (self.method + self.increment) & 0xFFF;
        Ok(())
    }
}
