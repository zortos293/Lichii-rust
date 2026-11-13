//! Golden tests: hand-encoded instructions translated into one block and dumped.

use lichii_common::GpuVAddr;

use super::translate_block;
use crate::ir::Program;
use crate::{ShaderArch, ShaderEnvironment, ShaderStage};

/// A compute program at 0x1000 whose instruction slots hold `code` (scheduling words are
/// inserted automatically).
pub struct TestEnv {
    pub words: Vec<u64>,
    pub stage: ShaderStage,
    pub cbuf: Vec<u32>,
    pub local_memory: u32,
    pub base: GpuVAddr,
    /// Program header words read from the 0x50 bytes in front of `base`.
    pub header: [u32; 20],
}

impl TestEnv {
    pub const BASE: GpuVAddr = 0x1000;

    pub fn new(code: &[u64]) -> TestEnv {
        let mut words = Vec::new();
        for (i, &insn) in code.iter().enumerate() {
            if i % 3 == 0 {
                words.push(0);
            }
            words.push(insn);
        }
        TestEnv {
            words,
            stage: ShaderStage::Compute,
            cbuf: Vec::new(),
            local_memory: 0,
            base: Self::BASE,
            header: [0; 20],
        }
    }

    /// Offset one past the last instruction.
    pub fn end(&self) -> u32 {
        self.words.len() as u32 * 8
    }
}

impl ShaderEnvironment for TestEnv {
    test_gpu_state!(@shared);
    fn local_memory_size(&self) -> u32 {
        self.local_memory
    }
    fn arch(&self) -> ShaderArch {
        ShaderArch::Maxwell
    }
    fn stage(&self) -> ShaderStage {
        self.stage
    }
    fn start_address(&self) -> GpuVAddr {
        self.base
    }
    fn read_u64(&self, address: GpuVAddr) -> u64 {
        if let Some(i) = address
            .checked_sub(self.base - 0x50)
            .filter(|&offset| offset < 0x50)
            .map(|offset| offset as usize / 4)
        {
            return u64::from(self.header[i]) | (u64::from(self.header[i + 1]) << 32);
        }
        self.words
            .get((address.wrapping_sub(self.base) / 8) as usize)
            .copied()
            .unwrap_or(0)
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

/// Translates `code` into a single block and returns the dump.
pub fn translate(code: &[u64]) -> Result<String, crate::TranslateError> {
    let env = TestEnv::new(code);
    let mut prog = Program::new();
    let block = prog.new_block();
    translate_block(&mut prog, block, &env, 0, env.end())?;
    Ok(prog.dump())
}

/// Builds an instruction from the 16-bit opcode pattern of `op` and extra field bits.
pub fn encode(op: crate::maxwell::opcode::Opcode, fields: u64) -> u64 {
    let pattern = op.pattern();
    (u64::from(pattern.value) << 48) | (fields & !(u64::from(pattern.mask) << 48))
}

#[test]
fn every_opcode_translates_or_reports_unsupported() {
    use crate::maxwell::opcode::Opcode;
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    for &op in Opcode::ALL {
        for _ in 0..64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let insn = encode(op, rng);
            let _ = translate(&[insn]);
        }
    }
}

#[test]
fn ram_and_sam_translate_to_nothing() {
    use crate::maxwell::opcode::Opcode;
    for op in [Opcode::Ram, Opcode::Sam] {
        assert_eq!(translate(&[encode(op, 0)]).unwrap(), "{0}:\n", "{op:?}");
    }
}

#[test]
fn control_flow_markers_translate_to_nothing() {
    use crate::maxwell::opcode::Opcode;
    for op in [
        Opcode::Nop,
        Opcode::Cal,
        Opcode::Pbk,
        Opcode::Pcnt,
        Opcode::Ssy,
    ] {
        assert_eq!(translate(&[encode(op, 0xf)]).unwrap(), "{0}:\n", "{op:?}");
    }
}

#[test]
fn kil_translates_only_in_fragment_programs() {
    use crate::maxwell::opcode::Opcode;
    for (stage, translates) in [(ShaderStage::Fragment, true), (ShaderStage::VertexB, false)] {
        let mut env = TestEnv::new(&[encode(Opcode::Kil, 0xf)]);
        env.stage = stage;
        let mut prog = Program::new();
        let block = prog.new_block();
        let result = translate_block(&mut prog, block, &env, 0, env.end());
        assert_eq!(result.is_ok(), translates, "{stage:?}");
    }
}
