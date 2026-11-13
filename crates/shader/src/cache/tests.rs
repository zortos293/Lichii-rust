use lichii_common::GpuVAddr;
use lichii_common::settings::ResolutionScale;

use super::record::Logger;
use super::*;
use crate::ampere::translate::tests::{TestEnv as AmpereEnv, encode as ampere};
use crate::ampere::{Form, Opcode as Ampere};
use crate::info::{ReplaceConstant, TexturePixelFormat};
use crate::ir::TextureType;
use crate::maxwell::opcode::Opcode as Sass;
use crate::maxwell::translate::tests::{TestEnv, encode};
use crate::profile::HostTranslateInfo;
use crate::{MeshState, Profile, ShaderArch};

const PT: u64 = 7 << 16;

/// Answers every query from its fields; program memory holds `code` from `start` and
/// `header` words before it.
#[derive(Clone)]
struct Answers {
    arch: ShaderArch,
    stage: ShaderStage,
    start: GpuVAddr,
    code: Vec<u64>,
    cbuf: u32,
    texture_type: TextureType,
    pixel_format: TexturePixelFormat,
    integer: bool,
    multisample: bool,
    viewport: u32,
    bound_buffer: u32,
    local_memory: u32,
    shared_memory: u32,
    workgroup: [u32; 3],
    hle: bool,
    storage_pointer: bool,
    native: bool,
    indirect_pointer: bool,
    passthrough: [u32; 8],
    replace: Option<ReplaceConstant>,
    proprietary: bool,
    mesh: Option<MeshState>,
    code_reads: std::cell::Cell<usize>,
}

impl Default for Answers {
    fn default() -> Answers {
        Answers {
            arch: ShaderArch::Maxwell,
            stage: ShaderStage::Compute,
            start: 0x1000,
            code: vec![0x11, 0x22, 0x33],
            cbuf: 5,
            texture_type: TextureType::Color2D,
            pixel_format: TexturePixelFormat::A8b8g8r8Unorm,
            integer: false,
            multisample: false,
            viewport: 1,
            bound_buffer: 2,
            local_memory: 0,
            shared_memory: 0,
            workgroup: [1, 1, 1],
            hle: false,
            storage_pointer: true,
            native: false,
            indirect_pointer: true,
            passthrough: [0; 8],
            replace: None,
            proprietary: false,
            mesh: None,
            code_reads: Default::default(),
        }
    }
}

impl ShaderEnvironment for Answers {
    fn arch(&self) -> ShaderArch {
        self.arch
    }
    fn stage(&self) -> ShaderStage {
        self.stage
    }
    fn start_address(&self) -> GpuVAddr {
        self.start
    }
    fn read_u64(&self, address: GpuVAddr) -> u64 {
        self.code_reads.set(self.code_reads.get() + 1);
        let index = address.wrapping_sub(self.start) / 8;
        self.code.get(index as usize).copied().unwrap_or(0)
    }
    fn read_cbuf(&self, _: u32, _: u32) -> u32 {
        self.cbuf
    }
    fn shared_memory_size(&self) -> u32 {
        self.shared_memory
    }
    fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup
    }
    fn texture_type(&self, _: u32) -> TextureType {
        self.texture_type
    }
    fn texture_pixel_format(&self, _: u32) -> TexturePixelFormat {
        self.pixel_format
    }
    fn is_texture_pixel_format_integer(&self, _: u32) -> bool {
        self.integer
    }
    fn is_texture_multisample(&self, _: u32) -> bool {
        self.multisample
    }
    fn viewport_transform_state(&self) -> u32 {
        self.viewport
    }
    fn texture_bound_buffer(&self) -> u32 {
        self.bound_buffer
    }
    fn local_memory_size(&self) -> u32 {
        self.local_memory
    }
    fn has_hle_macro_state(&self) -> bool {
        self.hle
    }
    fn is_cbuf_storage_buffer_pointer(&self, _: u32, _: u32) -> bool {
        self.storage_pointer
    }
    fn is_native_program(&self) -> bool {
        self.native
    }
    fn is_cbuf_indirect_storage_buffer_pointer(&self, _: u32, _: u32, _: u32) -> bool {
        self.indirect_pointer
    }
    fn gp_passthrough_mask(&self) -> [u32; 8] {
        self.passthrough
    }
    fn replace_const_buffer(&self, _: u32, _: u32) -> Option<ReplaceConstant> {
        self.replace
    }
    fn is_proprietary_driver(&self) -> bool {
        self.proprietary
    }
    fn mesh_state(&self) -> Option<MeshState> {
        self.mesh
    }
}

type Ask = fn(&dyn ShaderEnvironment) -> String;
type Change = fn(&mut Answers);

#[test]
fn a_record_matches_only_the_answers_it_saw() {
    let cases: [(Ask, Change); 22] = [
        (
            |e| format!("{:?}", e.arch()),
            |a| a.arch = ShaderArch::Ampere,
        ),
        (
            |e| format!("{:?}", e.stage()),
            |a| a.stage = ShaderStage::Fragment,
        ),
        (|e| format!("{:?}", e.start_address()), |a| a.start += 8),
        (
            |e| format!("{:?}", e.read_u64(e.start_address() + 8)),
            |a| a.code[1] ^= 1,
        ),
        (|e| format!("{:?}", e.read_cbuf(1, 0x40)), |a| a.cbuf += 1),
        (
            |e| format!("{:?}", e.shared_memory_size()),
            |a| a.shared_memory = 4,
        ),
        (
            |e| format!("{:?}", e.workgroup_size()),
            |a| a.workgroup[2] = 2,
        ),
        (
            |e| format!("{:?}", e.texture_type(7)),
            |a| a.texture_type = TextureType::Color3D,
        ),
        (
            |e| format!("{:?}", e.texture_pixel_format(7)),
            |a| a.pixel_format = TexturePixelFormat::R32Uint,
        ),
        (
            |e| format!("{:?}", e.is_texture_pixel_format_integer(7)),
            |a| a.integer = true,
        ),
        (
            |e| format!("{:?}", e.is_texture_multisample(7)),
            |a| a.multisample = true,
        ),
        (
            |e| format!("{:?}", e.viewport_transform_state()),
            |a| a.viewport = 0,
        ),
        (
            |e| format!("{:?}", e.texture_bound_buffer()),
            |a| a.bound_buffer = 3,
        ),
        (
            |e| format!("{:?}", e.local_memory_size()),
            |a| a.local_memory = 0x100,
        ),
        (
            |e| format!("{:?}", e.has_hle_macro_state()),
            |a| a.hle = true,
        ),
        (
            |e| format!("{:?}", e.is_cbuf_storage_buffer_pointer(0, 8)),
            |a| a.storage_pointer = false,
        ),
        (
            |e| format!("{:?}", e.is_native_program()),
            |a| a.native = true,
        ),
        (
            |e| format!("{:?}", e.is_cbuf_indirect_storage_buffer_pointer(0, 8, 16)),
            |a| a.indirect_pointer = false,
        ),
        (
            |e| format!("{:?}", e.gp_passthrough_mask()),
            |a| a.passthrough[3] = 1,
        ),
        (
            |e| format!("{:?}", e.replace_const_buffer(0, 4)),
            |a| a.replace = Some(ReplaceConstant::BaseVertex),
        ),
        (
            |e| format!("{:?}", e.is_proprietary_driver()),
            |a| a.proprietary = true,
        ),
        (
            |e| format!("{:?}", e.mesh_state()),
            |a| a.mesh = Some(MeshState::default()),
        ),
    ];
    let base = Answers::default();
    for (i, (ask, change)) in cases.into_iter().enumerate() {
        let logger = Logger::new(&base);
        ask(&logger);
        let record = logger.into_record();
        assert!(
            record.matches(&base, 1, Some(&mut CodeCheck::default())),
            "case {i}"
        );
        let mut changed = base.clone();
        change(&mut changed);
        assert!(
            !record.matches(&changed, 1, Some(&mut CodeCheck::default())),
            "case {i}"
        );
    }
}

/// Texture queries by the words that hold the handle record the words' location and the
/// answer, not the words: a record and the module hash hold for any handle with the same answer.
#[test]
fn texture_queries_by_handle_words_hold_for_any_handle_with_the_same_answer() {
    use crate::hash::Recorder;
    use crate::info::HandleSource;
    const HANDLE: HandleSource = HandleSource {
        cbuf_index: 1,
        cbuf_offset: 0x40,
        shift_left: 0,
        secondary: None,
    };
    let cases: [(Ask, Change); 4] = [
        (
            |e| format!("{:?}", e.texture_type_of(HANDLE)),
            |a| a.texture_type = TextureType::Color3D,
        ),
        (
            |e| format!("{:?}", e.texture_pixel_format_of(HANDLE)),
            |a| a.pixel_format = TexturePixelFormat::R32Uint,
        ),
        (
            |e| format!("{:?}", e.is_texture_pixel_format_integer_of(HANDLE)),
            |a| a.integer = true,
        ),
        (
            |e| format!("{:?}", e.is_texture_multisample_of(HANDLE)),
            |a| a.multisample = true,
        ),
    ];
    let base = Answers::default();
    let other_handle = Answers {
        cbuf: base.cbuf + 1,
        ..base.clone()
    };
    for (i, (ask, change)) in cases.into_iter().enumerate() {
        let logger = Logger::new(&base);
        ask(&logger);
        let record = logger.into_record();
        assert!(
            record.matches(&other_handle, 1, Some(&mut CodeCheck::default())),
            "case {i}"
        );
        let mut changed = base.clone();
        change(&mut changed);
        assert!(
            !record.matches(&changed, 1, Some(&mut CodeCheck::default())),
            "case {i}"
        );

        let hash = |env: &Answers| {
            let recorder = Recorder::new(env);
            ask(&recorder);
            recorder.hasher().finish()
        };
        assert_eq!(hash(&base), hash(&other_handle), "case {i}");
        assert_ne!(hash(&base), hash(&changed), "case {i}");
    }
}

#[test]
fn program_memory_is_compared_relative_to_the_start() {
    let base = Answers::default();
    let logger = Logger::new(&base);
    logger.read_u64(base.start - 8);
    logger.read_u64(base.start + 16);
    let record = logger.into_record();
    assert_eq!(record.code_range(), (-8, 32));
    let moved = Answers {
        start: 0x9000,
        ..base.clone()
    };
    assert!(record.matches(&moved, 1, Some(&mut CodeCheck::default())));
    let mut edited = base.clone();
    edited.code[2] = 0;
    assert!(!record.matches(&edited, 1, Some(&mut CodeCheck::default())));
    assert!(
        record.matches(&edited, 1, None),
        "state only skips program memory"
    );
}

fn options<'a>(profile: &'a Profile, host: &'a HostTranslateInfo) -> TranslateOptions<'a> {
    TranslateOptions {
        profile,
        host,
        resolution_scale: ResolutionScale::X1,
    }
}

/// A compute program storing `value` to shared memory.
fn store(value: u64) -> TestEnv {
    TestEnv::new(&[
        encode(Sass::Mov32i, PT | (value << 20) | (0xf << 12) | 1),
        encode(Sass::Sts, PT | (4 << 48) | (0x10 << 20) | (0xff << 8) | 1),
        encode(Sass::Exit, PT | 0xf),
    ])
}

#[test]
fn an_entry_survives_the_file_and_matches_only_its_program() {
    let (profile, host) = (Profile::default(), HostTranslateInfo::default());
    let options = options(&profile, &host);
    let runtime = RuntimeInfo::default();
    let env = store(5);
    let mut bindings = Bindings::default();
    let entry = compile(&env, options, &runtime, &mut bindings).expect("translate");
    let direct = crate::compile(&env, options, &runtime, &mut Bindings::default()).unwrap();
    assert_eq!(entry.module().spirv, direct.spirv);
    assert_eq!(entry.bindings_after(), bindings);

    let mut file = header().to_vec();
    encode_module(entry.module(), &mut file);
    entry.encode(&mut file);
    let records = records(&file).expect("current header");
    let mut entries = entries(records);
    let loaded = entries.next().expect("one entry");
    assert!(entries.next().is_none());
    assert_eq!(entries.valid_len(), records.len());
    assert_eq!(loaded.module().spirv, direct.spirv);
    assert_eq!(loaded.module().hash, direct.hash);
    assert_eq!(loaded.module().info, direct.info);
    assert_eq!(loaded.module().resources, direct.resources);

    let key = options_key(options, &runtime, &Bindings::default());
    assert_eq!(loaded.probe_key(), probe_key(None, &env, key));
    assert!(loaded.matches(None, &env, key, &mut CodeCheck::default()));
    let mut moved = store(5);
    moved.base = 0x8000;
    assert!(
        loaded.matches(None, &moved, key, &mut CodeCheck::default()),
        "same code elsewhere"
    );
    assert!(
        !loaded.matches(None, &store(7), key, &mut CodeCheck::default()),
        "other code"
    );
    assert!(loaded.matches_state(None, &store(7), key), "state alone");
    let scaled = TranslateOptions {
        resolution_scale: ResolutionScale::X2,
        ..options
    };
    let other = options_key(scaled, &runtime, &Bindings::default());
    assert_ne!(other, key);
    assert!(
        !loaded.matches(None, &env, other, &mut CodeCheck::default()),
        "other options"
    );
    let mut shifted = Bindings::default();
    shifted.unified += 1;
    assert!(!loaded.matches(
        None,
        &env,
        options_key(options, &runtime, &shifted),
        &mut CodeCheck::default()
    ));
    assert_eq!(
        loaded.code_ranges(),
        [(0, 0), (8, 24)],
        "three instructions after a control word"
    );
}

/// An Ampere compute program at `base` storing its own address (LEPC) to shared memory.
fn store_own_address(base: GpuVAddr) -> AmpereEnv {
    let mut env = AmpereEnv::new(&[
        ampere(Ampere::Lepc, Form::Fixed, 4 << 16),
        ampere(
            Ampere::Sts,
            Form::Fixed,
            (255 << 24) | (4 << 32) | (0x10 << 40) | (5 << 73),
        ),
        ampere(Ampere::Exit, Form::Fixed, 7 << 87),
    ]);
    env.base = base;
    env.shared_memory = 0x100;
    env
}

#[test]
fn an_entry_holding_its_address_does_not_move() {
    let (profile, host) = (Profile::default(), HostTranslateInfo::default());
    let options = options(&profile, &host);
    let runtime = RuntimeInfo::default();
    let key = options_key(options, &runtime, &Bindings::default());
    let env = store_own_address(AmpereEnv::BASE);
    let entry = compile(&env, options, &runtime, &mut Bindings::default()).expect("translate");
    assert!(entry.module().info.reads_start_address);
    let mut file = header().to_vec();
    encode_module(entry.module(), &mut file);
    entry.encode(&mut file);
    let loaded = entries(records(&file).unwrap()).next().expect("entry");
    assert!(loaded.matches(None, &env, key, &mut CodeCheck::default()));

    let moved = store_own_address(AmpereEnv::BASE + 0x1000);
    assert_eq!(moved.base % 32, env.base % 32);
    assert!(!loaded.matches(None, &moved, key, &mut CodeCheck::default()));
    assert!(!loaded.matches_state(None, &moved, key));
    let recompiled = crate::compile(&moved, options, &runtime, &mut Bindings::default()).unwrap();
    assert_ne!(recompiled.spirv, loaded.module().spirv);
    assert_ne!(recompiled.hash, loaded.module().hash);
}

#[test]
fn dual_vertex_entries_need_both_programs() {
    let (profile, host) = (Profile::default(), HostTranslateInfo::default());
    let options = options(&profile, &host);
    let runtime = RuntimeInfo::default();
    let program = |stage, value| {
        let mut env = store(value);
        env.stage = stage;
        env
    };
    let a = program(ShaderStage::VertexA, 5);
    let b = program(ShaderStage::VertexB, 7);
    let entry = compile_dual_vertex(&a, &b, options, &runtime, &mut Bindings::default())
        .expect("translate");
    let mut file = header().to_vec();
    encode_module(entry.module(), &mut file);
    entry.encode(&mut file);
    let loaded = entries(records(&file).unwrap()).next().expect("entry");
    let key = options_key(options, &runtime, &Bindings::default());
    assert_eq!(loaded.probe_key(), probe_key(Some(&a), &b, key));
    assert!(loaded.matches(Some(&a), &b, key, &mut CodeCheck::default()));
    assert!(!loaded.matches(None, &b, key, &mut CodeCheck::default()));
    assert!(!loaded.matches(
        Some(&program(ShaderStage::VertexA, 6)),
        &b,
        key,
        &mut CodeCheck::default()
    ));
    assert!(!loaded.matches(
        Some(&a),
        &program(ShaderStage::VertexB, 6),
        key,
        &mut CodeCheck::default()
    ));
}

#[test]
fn stale_headers_and_damaged_records_are_refused() {
    let mut file = header().to_vec();
    assert_eq!(records(&file), Ok(&[][..]));
    assert_eq!(records(&file[..8]), Err(StaleCache::NotACache));
    file[0] ^= 1;
    assert_eq!(records(&file), Err(StaleCache::NotACache));
    let mut format = header();
    format[8] ^= 1;
    assert_eq!(
        records(&format),
        Err(StaleCache::Format(FORMAT_VERSION ^ 1))
    );
    let mut translator = header();
    translator[16] ^= 1;
    assert_eq!(records(&translator), Err(StaleCache::Translator));

    let (profile, host) = (Profile::default(), HostTranslateInfo::default());
    let entry = compile(
        &store(5),
        options(&profile, &host),
        &RuntimeInfo::default(),
        &mut Bindings::default(),
    )
    .unwrap();
    let mut bytes = Vec::new();
    encode_module(entry.module(), &mut bytes);
    entry.encode(&mut bytes);
    let one = bytes.len();
    entry.encode(&mut bytes);
    let mut truncated = entries(&bytes[..bytes.len() - 1]);
    assert_eq!(truncated.by_ref().count(), 1);
    assert_eq!(truncated.valid_len(), one);
    bytes[one + RECORD_HEADER_LEN + 3] ^= 0x40;
    let mut corrupt = entries(&bytes);
    assert_eq!(corrupt.by_ref().count(), 1);
    assert_eq!(corrupt.valid_len(), one);
}

#[test]
fn a_lookup_hashes_each_program_range_once() {
    let base = Answers {
        code: vec![1, 2, 3, 4],
        ..Answers::default()
    };
    let logger = Logger::new(&base);
    logger.read_u64(base.start);
    logger.read_u64(base.start + 24);
    let record = logger.into_record();
    let mut check = CodeCheck::default();
    base.code_reads.set(0);
    assert!(record.matches(&base, 1, Some(&mut check)));
    assert_eq!(base.code_reads.get(), 4, "the whole range, read once");
    assert!(record.matches(&base, 1, Some(&mut check)));
    assert_eq!(base.code_reads.get(), 4, "the second entry reuses the hash");
    let mut edited = base.clone();
    edited.code[1] = 0;
    assert!(
        !record.matches(&edited, 1, Some(&mut CodeCheck::default())),
        "a word between the ones read counts too"
    );
}

#[test]
fn entries_sharing_a_module_store_it_once() {
    let (profile, host) = (Profile::default(), HostTranslateInfo::default());
    let options = options(&profile, &host);
    let runtime = RuntimeInfo::default();
    let first = compile(&store(5), options, &runtime, &mut Bindings::default()).unwrap();
    let mut moved = store(5);
    moved.base = 0x8000;
    let second = compile(&moved, options, &runtime, &mut Bindings::default()).unwrap();
    assert_eq!(first.module().hash, second.module().hash);

    let mut file = header().to_vec();
    encode_module(first.module(), &mut file);
    first.encode(&mut file);
    let one = file.len();
    second.encode(&mut file);
    let mut module = Vec::new();
    encode_module(first.module(), &mut module);
    assert!(file.len() - one < module.len(), "no second module");

    let loaded: Vec<_> = entries(records(&file).unwrap()).collect();
    assert_eq!(loaded.len(), 2);
    assert!(Arc::ptr_eq(loaded[0].module(), loaded[1].module()));
}

#[test]
fn an_entry_without_its_module_ends_the_readable_part() {
    let (profile, host) = (Profile::default(), HostTranslateInfo::default());
    let entry = compile(
        &store(5),
        options(&profile, &host),
        &RuntimeInfo::default(),
        &mut Bindings::default(),
    )
    .unwrap();
    let mut bytes = Vec::new();
    entry.encode(&mut bytes);
    let mut orphan = entries(&bytes);
    assert_eq!(orphan.by_ref().count(), 0);
    assert_eq!(orphan.valid_len(), 0);
}
