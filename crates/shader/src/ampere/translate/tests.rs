//! Test harness for the Ampere translators: a synthetic environment, an encoder and the
//! every-operation property test.

use lichii_common::GpuVAddr;

use lichii_common::settings::ResolutionScale;

use super::translate_block;
use crate::ampere::decode::{Decoded, Form, decode};
use crate::ampere::instruction::{INSTRUCTION_SIZE, Instruction};
use crate::ampere::opcode::Opcode;
use crate::backend::Bindings;
use crate::backend::spirv::EmitError;
use crate::ir::{Attribute, Program};
use crate::maxwell::header::ProgramHeader;
use crate::profile::HostTranslateInfo;
use crate::translate::TranslateOptions;
use crate::{
    MeshState, Profile, RuntimeInfo, ShaderArch, ShaderEnvironment, ShaderStage, TranslateError,
};

/// A synthetic Ampere program: instructions from `base` (`BASE` unless moved), a program header
/// 0x80 bytes before it and one constant buffer.
pub struct TestEnv {
    pub base: GpuVAddr,
    pub code: Vec<Instruction>,
    pub stage: ShaderStage,
    pub header: [u32; 20],
    pub cbuf: Vec<u32>,
    pub local_memory: u32,
    pub shared_memory: u32,
    pub workgroup: [u32; 3],
    pub mesh: Option<MeshState>,
    pub multisample: bool,
}

impl TestEnv {
    pub const BASE: GpuVAddr = 0x10_0000;

    pub fn new(code: &[Instruction]) -> TestEnv {
        TestEnv {
            base: Self::BASE,
            code: code.to_vec(),
            stage: ShaderStage::Compute,
            header: [0; 20],
            cbuf: Vec::new(),
            local_memory: 0,
            shared_memory: 0,
            workgroup: [1, 1, 1],
            mesh: None,
            multisample: false,
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
        self.base
    }
    fn read_u64(&self, address: GpuVAddr) -> u64 {
        if let Some(offset) = address.checked_sub(self.base) {
            let index = (offset / u64::from(INSTRUCTION_SIZE)) as usize;
            return self.code.get(index).map_or(0, |inst| {
                if offset % 16 == 0 {
                    inst.0 as u64
                } else {
                    (inst.0 >> 64) as u64
                }
            });
        }
        let header = self.base - crate::ampere::GRAPHICS_HEADER_SIZE;
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
        self.shared_memory
    }
    fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup
    }
    fn mesh_state(&self) -> Option<MeshState> {
        self.mesh
    }
    fn is_texture_multisample(&self, _: u32) -> bool {
        self.multisample
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

/// A xorshift64 generator seeded with the given state.
fn xorshift(mut state: u64) -> impl FnMut() -> u64 {
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }
}

/// The opcode bits of every encoding the decoder knows that does not branch, exit or kill.
fn straight_line_encodings() -> Vec<u128> {
    (0..4096u128)
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
        .collect()
}

/// One to four random instructions from the encodings, most with plain modifier fields, then
/// EXIT.
fn random_program(next: &mut impl FnMut() -> u64, encodings: &[u128]) -> Vec<Instruction> {
    let len = 1 + (next() % 4) as usize;
    let mut code: Vec<Instruction> = (0..len)
        .map(|_| {
            let bits = encodings[(next() % encodings.len() as u64) as usize];
            let mut fields = u128::from(next()) | (u128::from(next()) << 64);
            if !next().is_multiple_of(4) {
                let plain_modifiers = (7 << 81) | (7 << 84) | (7 << 87);
                fields = (fields & ((1 << 72) - 1)) | plain_modifiers;
            }
            Instruction((fields & !0xffff) | (7 << 12) | bits)
        })
        .collect();
    code.push(encode(Opcode::Exit, Form::Fixed, 7 << 87));
    code
}

/// Random Ampere programs with random program headers, through translation and `compile`,
/// for every stage: each either fails with an error or yields a module that passes spirv-val.
#[test]
fn random_programs_compile_to_valid_spirv_for_every_stage() {
    let mut next = xorshift(0x6a09_e667_f3bc_c908);
    let encodings = straight_line_encodings();
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
            let code = random_program(&mut next, &encodings);
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

/// Mesh state for a program of the given topology: 32 vertices, 16 primitives, 32 threads in
/// both stages and two payload lines.
fn mesh_state(topology: u32, primitive_header: ProgramHeader) -> MeshState {
    MeshState {
        shader_a: topology | (32 << 4) | (16 << 13),
        shader_b: 32 << 20,
        init_shader: 32 | (1 << 12) | (2 << 22),
        primitive_header,
    }
}

fn mesh_options<'a>(profile: &'a Profile, host: &'a HostTranslateInfo) -> TranslateOptions<'a> {
    TranslateOptions {
        profile,
        host,
        resolution_scale: ResolutionScale::X1,
    }
}

#[test]
fn task_and_mesh_programs_take_their_layout_from_the_mesh_state() {
    let (profile, host) = (Profile::default(), HostTranslateInfo::default());
    let exit = encode(Opcode::Exit, Form::Fixed, 7 << 87);
    let mut env = TestEnv::new(&[exit]);
    env.stage = ShaderStage::Mesh;
    env.header[0x34 / 4] |= (0x1 << 8) | (0xf << 16);
    let mut primitive_header = ProgramHeader::default();
    primitive_header.words[0x34 / 4] |= 0x10 << 16;
    env.mesh = Some(mesh_state(4, primitive_header));
    let program = crate::translate::translate(&env, mesh_options(&profile, &host)).unwrap();
    let mesh = &program.info.mesh;
    assert_eq!(
        (
            mesh.payload_words,
            mesh.topology,
            mesh.max_vertices,
            mesh.max_primitives
        ),
        (64, 4, 32, 16)
    );
    assert_eq!(program.workgroup_size, [32, 1, 1]);
    let generic = |index, element| u32::from(Attribute::generic(index, element).0);
    let primitive_id = u32::from(Attribute::PRIMITIVE_ID.0);
    assert_eq!(
        mesh.vertex_lines,
        [
            primitive_id,
            generic(0, 0),
            generic(0, 1),
            generic(0, 2),
            generic(0, 3)
        ]
    );
    assert_eq!(mesh.primitive_lines, [generic(1, 0)]);
    for line in mesh.vertex_lines.iter().chain(&mesh.primitive_lines) {
        assert!(program.info.stores.get(Attribute(*line as u16)), "{line}");
    }

    env.stage = ShaderStage::Task;
    let program = crate::translate::translate(&env, mesh_options(&profile, &host)).unwrap();
    assert_eq!(program.workgroup_size, [32, 1, 1]);
    assert_eq!(program.info.mesh.payload_words, 64);

    env.mesh = Some(mesh_state(5, ProgramHeader::default()));
    env.stage = ShaderStage::Mesh;
    assert!(matches!(
        crate::translate::translate(&env, mesh_options(&profile, &host)),
        Err(TranslateError::UnsupportedMesh(_))
    ));
    env.mesh = None;
    assert!(matches!(
        crate::translate::translate(&env, mesh_options(&profile, &host)),
        Err(TranslateError::MissingMeshState {
            stage: ShaderStage::Mesh
        })
    ));
}

/// Lavapipe draws nothing for a mesh program that stores an output vector one component at a
/// time, so the native epilogue stores each vertex's position and generics whole.
#[test]
fn native_mesh_outputs_are_stored_as_whole_vectors() {
    use rspirv::dr::Operand;
    use rspirv::spirv::{BuiltIn, Decoration, Op};

    let profile = Profile {
        supported_spirv: 0x0001_0600,
        support_mesh_shader: true,
        ..Profile::default()
    };
    let host = HostTranslateInfo::default();
    let exit = encode(Opcode::Exit, Form::Fixed, 7 << 87);
    let mut env = TestEnv::new(&[exit]);
    env.stage = ShaderStage::Mesh;
    env.header[0x34 / 4] |= (0xf << 12) | (0x7 << 16);
    env.mesh = Some(mesh_state(4, ProgramHeader::default()));
    let runtime = RuntimeInfo {
        convert_depth_mode: true,
        ..RuntimeInfo::default()
    };
    let module = crate::compile(
        &env,
        mesh_options(&profile, &host),
        &runtime,
        &mut Bindings::default(),
    )
    .unwrap();
    let position = u32::from(Attribute::POSITION_X.0);
    assert!(
        module
            .info
            .mesh
            .vertex_lines
            .starts_with(&[position, position + 1])
    );
    let spirv = rspirv::dr::load_words(&module.spirv).unwrap();
    let position = spirv
        .annotations
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands[1..]
                    == [
                        Operand::Decoration(Decoration::BuiltIn),
                        Operand::BuiltIn(BuiltIn::Position),
                    ]
        })
        .and_then(|inst| inst.operands[0].id_ref_any())
        .expect("a Position output");
    let outputs: Vec<u32> = spirv
        .types_global_values
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Variable
                && inst.operands[0] == Operand::StorageClass(rspirv::spirv::StorageClass::Output)
        })
        .filter_map(|inst| inst.result_id)
        .collect();
    assert!(outputs.contains(&position));
    let chains: Vec<_> = spirv.functions[0]
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .filter(|inst| {
            inst.class.opcode == Op::AccessChain
                && inst.operands[0]
                    .id_ref_any()
                    .is_some_and(|base| outputs.contains(&base))
        })
        .collect();
    assert!(
        chains
            .iter()
            .any(|inst| inst.operands[0].id_ref_any() == Some(position)),
        "position is written"
    );
    for chain in chains {
        assert_eq!(chain.operands.len(), 2, "{chain:?} selects a component");
    }
}

/// Random task and mesh programs through compile, natively and emulated: each either fails
/// with an error or yields a module that passes spirv-val.
#[test]
fn random_task_and_mesh_programs_compile_to_valid_spirv() {
    use crate::runtime::{MeshEmulation, MeshEmulationTask};

    let mut next = xorshift(0x3c6e_f372_fe94_f82b);
    let encodings = straight_line_encodings();
    let profile = Profile {
        supported_spirv: 0x0001_0600,
        support_mesh_shader: true,
        support_int8: true,
        support_int16: true,
        support_int64: true,
        support_float16: true,
        support_float64: true,
        ..Profile::default()
    };
    let host = HostTranslateInfo {
        support_float64: true,
        support_float16: true,
        support_int64: true,
        ..HostTranslateInfo::default()
    };
    let emulated = RuntimeInfo {
        mesh_emulation: Some(MeshEmulation {
            output_binding: 5,
            task: Some(MeshEmulationTask {
                payload_binding: 6,
                offsets_binding: 7,
                args_binding: 8,
                payload_stride: 64,
            }),
        }),
        ..RuntimeInfo::default()
    };
    for stage in [ShaderStage::Task, ShaderStage::Mesh] {
        let (mut native, mut emulation) = (0, 0);
        for i in 0..1200 {
            let code = random_program(&mut next, &encodings);
            let mut env = TestEnv::new(&code);
            env.stage = stage;
            let is_native = i % 2 == 0;
            let any_layout = is_native || next().is_multiple_of(4);
            let topology = if any_layout {
                [0, 1, 4][(next() % 3) as usize]
            } else {
                4
            };
            env.header[0x34 / 4] |= (next() as u32 & 0xffff) << 16;
            let mut primitive_header = ProgramHeader::default();
            if any_layout && next().is_multiple_of(2) {
                primitive_header.words[0x34 / 4] |= (next() as u32 & 0xff) << 16;
            }
            env.mesh = Some(mesh_state(topology, primitive_header));
            let runtime = if is_native {
                &RuntimeInfo::default()
            } else {
                &emulated
            };
            let mut bindings = Bindings::default();
            match crate::module::compile(
                &env,
                mesh_options(&profile, &host),
                runtime,
                &mut bindings,
            ) {
                Ok(module) => {
                    crate::backend::spirv::tests::validate(
                        &format!("ampere-random-{stage:?}-{i}"),
                        &profile,
                        &module.spirv,
                    );
                    *if is_native {
                        &mut native
                    } else {
                        &mut emulation
                    } += 1;
                }
                Err(TranslateError::InvalidIr(e)) => panic!("{stage:?} {code:x?}: {e}"),
                Err(TranslateError::Emit(EmitError::InvalidIr(e))) => {
                    panic!("{stage:?} {code:x?}: {e}")
                }
                Err(_) => {}
            }
        }
        assert!(
            native > 50 && emulation > 50,
            "{stage:?}: only {native} native and {emulation} emulated programs compiled"
        );
    }
}

#[test]
#[ignore]
fn diag_dump_ir_of_file() {
    let bytes = std::fs::read(std::env::var("DUMP").unwrap()).unwrap();
    let stage = match std::env::var("STAGE").as_deref() {
        Ok("compute") => ShaderStage::Compute,
        _ => ShaderStage::VertexB,
    };
    let code: Vec<Instruction> = bytes[0x80..]
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| Instruction(u128::from_le_bytes(*c)))
        .collect();
    let mut env = TestEnv::new(&code);
    env.stage = stage;
    for (i, w) in bytes[..0x50].as_chunks::<4>().0.iter().enumerate() {
        env.header[i] = u32::from_le_bytes(*w);
    }
    let profile = Profile::default();
    let host = HostTranslateInfo::default();
    let options = TranslateOptions {
        profile: &profile,
        host: &host,
        resolution_scale: ResolutionScale::X1,
    };
    let ir = crate::translate::translate(&env, options).unwrap().ir.dump();
    println!("{ir}");
}

