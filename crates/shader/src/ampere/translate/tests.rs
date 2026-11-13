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

/// Random Ampere programs with random program headers, through translation and `compile`,
/// for every stage: each either fails with an error or yields a module that passes spirv-val.
#[test]
fn random_programs_compile_to_valid_spirv_for_every_stage() {
    use crate::backend::Bindings;
    use crate::backend::spirv::EmitError;
    use crate::profile::HostTranslateInfo;
    use crate::translate::TranslateOptions;
    use crate::{Profile, RuntimeInfo, TranslateError};
    use lichii_common::settings::ResolutionScale;

    let mut state = 0x6a09_e667_f3bc_c908u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let encodings: Vec<u128> = (0..4096u128)
        .filter(|&bits| {
            decode(Instruction(bits)).is_some_and(|d| {
                !matches!(
                    d.opcode,
                    Opcode::Bra
                        | Opcode::Brx
                        | Opcode::Brxu
                        | Opcode::Jmp
                        | Opcode::Jmx
                        | Opcode::Jmxu
                        | Opcode::CallAbs
                        | Opcode::CallRel
                        | Opcode::Ret
                        | Opcode::Exit
                        | Opcode::Kill
                        | Opcode::Break
                )
            })
        })
        .collect();
    let native_host = HostTranslateInfo {
        support_float64: true,
        support_float16: true,
        support_int64: true,
        ..HostTranslateInfo::default()
    };
    let minimal_host = HostTranslateInfo::default();
    let native = Profile {
        support_int8: true,
        support_int16: true,
        support_int64: true,
        support_float16: true,
        support_float64: true,
        ..Profile::default()
    };
    let exit = encode(Opcode::Exit, Form::Fixed, 7 << 87);
    let stages = [
        ShaderStage::Compute,
        ShaderStage::VertexB,
        ShaderStage::TessellationControl,
        ShaderStage::TessellationEval,
        ShaderStage::Geometry,
        ShaderStage::Fragment,
    ];
    for stage in stages {
        let mut compiled = 0;
        for i in 0..1200 {
            let len = 1 + (next() % 4) as usize;
            let mut code: Vec<Instruction> = (0..len)
                .map(|_| {
                    let bits = encodings[(next() % encodings.len() as u64) as usize];
                    let mut fields = u128::from(next()) | (u128::from(next()) << 64);
                    if next() % 4 != 0 {
                        let plain_modifiers = (7 << 81) | (7 << 84) | (7 << 87);
                        fields = (fields & ((1 << 72) - 1)) | plain_modifiers;
                    }
                    Instruction((fields & !0xffff) | (7 << 12) | bits)
                })
                .collect();
            code.push(exit);
            let mut env = TestEnv::new(&code);
            env.stage = stage;
            for word in &mut env.header {
                *word = next() as u32;
            }
            let topology = [1, 6, 7][(next() % 3) as usize];
            env.header[3] = (env.header[3] & !(0xf << 24)) | (topology << 24);
            env.header[2] = (env.header[2] & 0x00ff_ffff) | ((1 + next() as u32 % 4) << 24);
            env.header[4] = (env.header[4] & !0xfff) | (1 + next() as u32 % 16);
            for byte in 0x18..0x38 {
                let mode = (next() % 4) as u32 * 0x55;
                let shift = byte % 4 * 8;
                env.header[byte / 4] = (env.header[byte / 4] & !(0xff << shift)) | (mode << shift);
            }
            let (profile, host) = if i % 2 == 0 {
                (&native, &native_host)
            } else {
                (&Profile::default(), &minimal_host)
            };
            let options = TranslateOptions {
                profile,
                host,
                resolution_scale: ResolutionScale::X1,
            };
            let mut bindings = Bindings::default();
            match crate::module::compile(&env, options, &RuntimeInfo::default(), &mut bindings) {
                Ok(module) => {
                    crate::backend::spirv::tests::validate(
                        &format!("ampere-random-{stage:?}-{i}"),
                        profile,
                        &module.spirv,
                    );
                    compiled += 1;
                }
                Err(TranslateError::InvalidIr(e)) => panic!("{stage:?} {code:x?}: {e}"),
                Err(TranslateError::Emit(EmitError::InvalidIr(e))) => {
                    panic!("{stage:?} {code:x?}: {e}")
                }
                Err(_) => {}
            }
        }
        assert!(
            compiled > 100,
            "{stage:?}: only {compiled} programs compiled"
        );
    }
}
