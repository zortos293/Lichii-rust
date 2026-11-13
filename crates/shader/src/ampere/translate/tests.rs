//! Test harness for the Ampere translators: a synthetic environment, an encoder and the
//! every-operation property test.

use lichii_common::GpuVAddr;

use super::translate_block;
use crate::ampere::decode::{Decoded, Form, decode};
use crate::ampere::instruction::{INSTRUCTION_SIZE, Instruction};
use crate::ampere::opcode::Opcode;
use crate::ir::Program;
use crate::{ShaderArch, ShaderEnvironment, ShaderStage};

/// A synthetic Ampere program: instructions from `BASE`, a program header 0x80 bytes before it
/// and one constant buffer.
pub struct TestEnv {
    pub code: Vec<Instruction>,
    pub stage: ShaderStage,
    pub header: [u32; 20],
    pub cbuf: Vec<u32>,
    pub local_memory: u32,
}

impl TestEnv {
    pub const BASE: GpuVAddr = 0x10_0000;

    pub fn new(code: &[Instruction]) -> TestEnv {
        TestEnv {
            code: code.to_vec(),
            stage: ShaderStage::Compute,
            header: [0; 20],
            cbuf: Vec::new(),
            local_memory: 0,
        }
    }

    /// Offset one past the last instruction.
    pub fn end(&self) -> u32 {
        self.code.len() as u32 * INSTRUCTION_SIZE
    }
}

impl ShaderEnvironment for TestEnv {
    test_gpu_state!(@shared);
    fn local_memory_size(&self) -> u32 {
        self.local_memory
    }
    fn arch(&self) -> ShaderArch {
        ShaderArch::Ampere
    }
    fn stage(&self) -> ShaderStage {
        self.stage
    }
    fn start_address(&self) -> GpuVAddr {
        Self::BASE
    }
    fn read_u64(&self, address: GpuVAddr) -> u64 {
        if let Some(offset) = address.checked_sub(Self::BASE) {
            let index = (offset / u64::from(INSTRUCTION_SIZE)) as usize;
            return self.code.get(index).map_or(0, |inst| {
                if offset % 16 == 0 {
                    inst.0 as u64
                } else {
                    (inst.0 >> 64) as u64
                }
            });
        }
        let header = Self::BASE - crate::ampere::GRAPHICS_HEADER_SIZE;
        match address.checked_sub(header) {
            Some(offset) if offset < 0x50 => {
                let word = (offset / 4) as usize;
                u64::from(self.header[word]) | (u64::from(self.header[word + 1]) << 32)
            }
            _ => 0,
        }
    }
    fn read_cbuf(&self, _: u32, offset: u32) -> u32 {
        self.cbuf.get((offset / 4) as usize).copied().unwrap_or(0)
    }
    fn shared_memory_size(&self) -> u32 {
        0
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [1, 1, 1]
    }
}

/// The 12-bit opcode field that decodes to `op` in `form`.
pub fn opcode_bits(op: Opcode, form: Form) -> u128 {
    (0..4096u128)
        .find(|&bits| decode(Instruction(bits)) == Some(Decoded { opcode: op, form }))
        .unwrap_or_else(|| panic!("{op:?} has no {form:?} encoding"))
}

/// An instruction of `op` in `form` with the other bits from `fields` and an always-true
/// guard (`PT`).
pub fn encode(op: Opcode, form: Form, fields: u128) -> Instruction {
    let guard_pt = 7 << 12;
    Instruction((fields & !0xffff) | guard_pt | opcode_bits(op, form))
}

/// Translates `code` into a single block of a program of `stage` and returns the dump.
pub fn translate_stage(
    stage: ShaderStage,
    code: &[Instruction],
) -> Result<String, crate::TranslateError> {
    let mut env = TestEnv::new(code);
    env.stage = stage;
    let mut prog = Program::new();
    let block = prog.new_block();
    translate_block(&mut prog, block, &env, 0, env.end())?;
    Ok(prog.dump())
}

/// Translates `code` as a compute program and returns the dump.
pub fn translate(code: &[Instruction]) -> Result<String, crate::TranslateError> {
    translate_stage(ShaderStage::Compute, code)
}

/// Every encoding the decoder knows, with random operand bits, in every stage: translation
/// either succeeds or reports an error, and never panics.
#[test]
fn every_operation_translates_or_reports_an_error() {
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let stages = [
        ShaderStage::Compute,
        ShaderStage::VertexB,
        ShaderStage::TessellationControl,
        ShaderStage::TessellationEval,
        ShaderStage::Geometry,
        ShaderStage::Fragment,
    ];
    for bits in 0..4096u128 {
        if decode(Instruction(bits)).is_none() {
            continue;
        }
        for i in 0..48 {
            let fields = u128::from(next()) | (u128::from(next()) << 64);
            let inst = Instruction((fields & !0xfff) | bits);
            let _ = translate_stage(stages[i % stages.len()], &[inst]);
        }
    }
}

#[test]
fn compute_exit_translates_to_nothing() {
    let exit = encode(Opcode::Exit, Form::Fixed, 0);
    assert_eq!(translate(&[exit]).unwrap(), "{0}:\n");
}
