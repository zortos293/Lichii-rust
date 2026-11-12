use std::cell::Cell;

use lichii_common::GpuVAddr;
use rustc_hash::FxHashMap;

use super::*;
use crate::info::StorageBufferDescriptor;
use crate::ir::SyntaxNode;
use crate::opt::compute_post_order;

const BOUND: u32 = 2;

#[derive(Default)]
struct Env {
    words: FxHashMap<(u32, u32), u32>,
    types: FxHashMap<u32, TextureType>,
    formats: FxHashMap<u32, TexturePixelFormat>,
    integers: Vec<u32>,
    stage: Option<ShaderStage>,
    cbuf_reads: Cell<u32>,
    type_reads: Cell<u32>,
}

impl Env {
    fn word(mut self, index: u32, offset: u32, value: u32) -> Env {
        self.words.insert((index, offset), value);
        self
    }

    fn ty(mut self, handle: u32, ty: TextureType) -> Env {
        self.types.insert(handle, ty);
        self
    }

    fn format(mut self, handle: u32, format: TexturePixelFormat) -> Env {
        self.formats.insert(handle, format);
        self
    }
}

impl ShaderEnvironment for Env {
    fn arch(&self) -> ShaderArch {
        ShaderArch::Ampere
    }
    fn stage(&self) -> ShaderStage {
        self.stage.unwrap_or(ShaderStage::Fragment)
    }
    fn start_address(&self) -> GpuVAddr {
        0
    }
    fn read_u64(&self, _: GpuVAddr) -> u64 {
        0
    }
    fn read_cbuf(&self, index: u32, offset: u32) -> u32 {
        self.cbuf_reads.set(self.cbuf_reads.get() + 1);
        self.words.get(&(index, offset)).copied().unwrap_or(0)
    }
    fn shared_memory_size(&self) -> u32 {
        0
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [1; 3]
    }
    fn texture_type(&self, handle: u32) -> TextureType {
        self.type_reads.set(self.type_reads.get() + 1);
        self.types
            .get(&handle)
            .copied()
            .unwrap_or(TextureType::Color2D)
    }
    fn texture_pixel_format(&self, handle: u32) -> TexturePixelFormat {
        self.formats
            .get(&handle)
            .copied()
            .unwrap_or(TexturePixelFormat::A8b8g8r8Unorm)
    }
    fn is_texture_pixel_format_integer(&self, handle: u32) -> bool {
        self.integers.contains(&handle)
    }
    fn viewport_transform_state(&self) -> u32 {
        1
    }
    fn texture_bound_buffer(&self) -> u32 {
        BOUND
    }
    fn local_memory_size(&self) -> u32 {
        0
    }
    fn has_hle_macro_state(&self) -> bool {
        false
    }
}

fn host(limit: u32) -> HostTranslateInfo {
    HostTranslateInfo {
        max_per_stage_descriptor_sampled_images: limit,
        max_per_stage_descriptor_samplers: limit,
        max_descriptor_set_samplers: limit,
        max_descriptor_set_sampled_images: limit,
        max_per_stage_resources: 1024,
        ..HostTranslateInfo::default()
    }
}

fn program(n: usize) -> (Program, Vec<BlockId>) {
    let mut prog = Program::new();
    let blocks: Vec<BlockId> = (0..n).map(|_| prog.new_block()).collect();
    prog.block_order = blocks.clone();
    prog.syntax_list = blocks.iter().map(|&b| SyntaxNode::Block(b)).collect();
    (prog, blocks)
}

fn info(ty: TextureType) -> TextureInstInfo {
    TextureInstInfo {
        ty,
        ..TextureInstInfo::default()
    }
}

fn sample(prog: &mut Program, block: BlockId, handle: Value) -> InstId {
    let op = if handle.is_immediate() {
        Opcode::BoundImageSampleImplicitLod
    } else {
        Opcode::BindlessImageSampleImplicitLod
    };
    let coords = prog.append(
        block,
        Opcode::CompositeConstructF32x2,
        &[Value::F32(0.5), Value::F32(0.25)],
        0,
    );
    let flags = info(TextureType::Color2D).to_flags();
    prog.append(
        block,
        op,
        &[handle, Value::Inst(coords), Value::Void, Value::Void],
        flags,
    )
}

fn cbuf(prog: &mut Program, block: BlockId, index: u32, offset: Value) -> Value {
    Value::Inst(prog.append(block, Opcode::GetCbufU32, &[Value::U32(index), offset], 0))
}

fn lane(prog: &mut Program, block: BlockId) -> Value {
    Value::Inst(prog.append(block, Opcode::LaneId, &[], 0))
}

fn run(env: &Env, prog: &mut Program, host: &HostTranslateInfo) -> ShaderInfo {
    let mut info = ShaderInfo::default();
    compute_post_order(prog);
    texture_pass(env, prog, &mut info, host).unwrap();
    info
}

fn flags_of(prog: &Program, id: InstId) -> TextureInstInfo {
    TextureInstInfo::from_flags(prog.inst(id).flags)
}

fn word_source(cbuf_index: u32, cbuf_offset: u32) -> HandleSource {
    HandleSource {
        cbuf_index,
        cbuf_offset,
        ..HandleSource::default()
    }
}

fn inst_of(prog: &Program, value: Value) -> InstId {
    prog.resolve(value).inst().expect("an instruction")
}

#[test]
fn bound_textures_share_a_descriptor_per_offset() {
    let (mut prog, b) = program(1);
    let first = sample(&mut prog, b[0], Value::U32(0x20));
    let second = sample(&mut prog, b[0], Value::U32(0x20));
    let third = sample(&mut prog, b[0], Value::U32(0x30));
    let info = run(&Env::default(), &mut prog, &host(1024));

    assert_eq!(info.texture_descriptors.len(), 2);
    assert_eq!(info.texture_descriptors[0].handle, word_source(BOUND, 0x20));
    assert_eq!(info.texture_descriptors[1].handle, word_source(BOUND, 0x30));
    for (id, index) in [(first, 0), (second, 0), (third, 1)] {
        assert_eq!(prog.op(id), Opcode::ImageSampleImplicitLod);
        assert_eq!(flags_of(&prog, id).descriptor_index, index);
        assert_eq!(prog.arg(id, 0), Value::Void);
    }
}

#[test]
fn bindless_handle_tracks_its_constant_buffer_word() {
    let (mut prog, b) = program(1);
    let word = cbuf(&mut prog, b[0], 1, Value::U32(0x40));
    let copy = prog.append(b[0], Opcode::Identity, &[word], 0);
    let id = sample(&mut prog, b[0], Value::Inst(copy));
    let info = run(&Env::default(), &mut prog, &host(1024));

    assert_eq!(prog.op(id), Opcode::ImageSampleImplicitLod);
    assert_eq!(info.texture_descriptors[0].handle, word_source(1, 0x40));
    assert_eq!(info.texture_descriptors[0].count, 1);
    assert!(!info.texture_descriptors[0].is_descriptor_array);
}

#[test]
fn separate_sampler_orders_its_words() {
    let (mut prog, b) = program(1);
    let sampler = cbuf(&mut prog, b[0], 3, Value::U32(8));
    let texture = cbuf(&mut prog, b[0], 3, Value::U32(4));
    let handle = prog.append(b[0], Opcode::BitwiseOr32, &[sampler, texture], 0);
    sample(&mut prog, b[0], Value::Inst(handle));
    let env = Env::default().word(3, 4, 0x10).word(3, 8, 0x200);
    let info = run(&env, &mut prog, &host(1024));

    let handle = info.texture_descriptors[0].handle;
    assert_eq!((handle.cbuf_index, handle.cbuf_offset), (3, 4));
    assert_eq!(handle.secondary, Some((3, 8, 0)));
}

#[test]
fn dynamic_offset_becomes_a_clamped_array() {
    let (mut prog, b) = program(1);
    let index = lane(&mut prog, b[0]);
    let scaled = prog.append(b[0], Opcode::ShiftLeftLogical32, &[index, Value::U32(3)], 0);
    let offset = prog.append(
        b[0],
        Opcode::IAdd32,
        &[Value::U32(0x40), Value::Inst(scaled)],
        0,
    );
    let handle = cbuf(&mut prog, b[0], 1, Value::Inst(offset));
    let id = sample(&mut prog, b[0], handle);
    let info = run(&Env::default(), &mut prog, &host(16));

    let desc = info.texture_descriptors[0];
    assert_eq!(desc.handle, word_source(1, 0x40));
    assert_eq!((desc.count, desc.size_shift), (16, 3));
    assert!(desc.is_descriptor_array);
    let clamp = inst_of(&prog, prog.arg(id, 0));
    assert_eq!(prog.op(clamp), Opcode::UMin32);
    assert_eq!(prog.arg(clamp, 1), Value::U32(15));
    let element = inst_of(&prog, prog.arg(clamp, 0));
    assert_eq!(prog.op(element), Opcode::ShiftRightLogical32);
    assert_eq!(
        prog.inst(element).args(),
        &[Value::Inst(scaled), Value::U32(3)]
    );
}

#[test]
fn texture_type_comes_from_the_descriptor() {
    let (mut prog, b) = program(1);
    let query = prog.append(
        b[0],
        Opcode::BoundImageQueryDimensions,
        &[Value::U32(0x10), Value::U32(0), Value::U1(false)],
        0,
    );
    let fetch = prog.append(
        b[0],
        Opcode::BoundImageFetch,
        &[
            Value::U32(0x14),
            Value::U32(7),
            Value::Void,
            Value::Void,
            Value::U32(1),
        ],
        info(TextureType::Color1D).to_flags(),
    );
    let env = Env::default()
        .word(BOUND, 0x10, 0x77)
        .ty(0x77, TextureType::Color3D)
        .word(BOUND, 0x14, 0x88)
        .ty(0x88, TextureType::Buffer);
    let info = run(&env, &mut prog, &host(1024));

    assert_eq!(flags_of(&prog, query).ty, TextureType::Color3D);
    assert_eq!(info.texture_descriptors[0].ty, TextureType::Color3D);
    assert_eq!(flags_of(&prog, fetch).ty, TextureType::Buffer);
    assert_eq!(prog.arg(fetch, 4), Value::Void);
    assert_eq!(info.texture_buffer_descriptors.len(), 1);
    assert_eq!(
        info.texture_buffer_descriptors[0].handle,
        word_source(BOUND, 0x14)
    );
}

#[test]
fn integer_formats_mark_samples_but_not_depth_compares() {
    let (mut prog, b) = program(1);
    let color = sample(&mut prog, b[0], Value::U32(0x8));
    let depth = prog.append(
        b[0],
        Opcode::BoundImageSampleDrefImplicitLod,
        &[
            Value::U32(0x8),
            Value::Void,
            Value::F32(0.5),
            Value::Void,
            Value::Void,
        ],
        TextureInstInfo {
            is_depth: true,
            ..info(TextureType::Color2D)
        }
        .to_flags(),
    );
    let mut env = Env::default().word(BOUND, 0x8, 0x99);
    env.integers.push(0x99);
    let info = run(&env, &mut prog, &host(1024));

    assert_eq!(info.texture_descriptors.len(), 2);
    assert!(flags_of(&prog, color).is_integer);
    assert!(info.texture_descriptors[0].is_integer);
    assert!(!flags_of(&prog, depth).is_integer);
    assert!(!info.texture_descriptors[1].is_integer);
}

#[test]
fn storage_images_merge_their_usage() {
    let (mut prog, b) = program(1);
    let flags = TextureInstInfo {
        image_format: ImageFormat::R32Uint,
        ..info(TextureType::Color2D)
    }
    .to_flags();
    let read = prog.append(
        b[0],
        Opcode::BoundImageRead,
        &[Value::U32(0x50), Value::U32(0)],
        flags,
    );
    let write = prog.append(
        b[0],
        Opcode::BoundImageWrite,
        &[Value::U32(0x50), Value::U32(0), Value::U32(1)],
        flags | TextureInstInfo::IMAGE_COHERENT,
    );
    let buffer = prog.append(
        b[0],
        Opcode::BoundImageWrite,
        &[Value::U32(0x60), Value::U32(0), Value::U32(1)],
        info(TextureType::Buffer).to_flags(),
    );
    let info = run(&Env::default(), &mut prog, &host(1024));

    assert_eq!(info.image_descriptors.len(), 1);
    let image = info.image_descriptors[0];
    assert!(image.is_read && image.is_written && !image.is_byte_addressed);
    assert_eq!((image.cbuf_index, image.cbuf_offset), (BOUND, 0x50));
    assert_eq!(prog.op(read), Opcode::ImageRead);
    assert_eq!(prog.op(write), Opcode::ImageWrite);
    assert_ne!(prog.inst(write).flags & TextureInstInfo::IMAGE_COHERENT, 0);
    assert_eq!(info.image_buffer_descriptors.len(), 1);
    assert!(info.image_buffer_descriptors[0].is_written);
    assert_eq!(prog.op(buffer), Opcode::ImageWrite);
}

fn byte_addressed_store(prog: &mut Program, block: BlockId, op: Opcode) -> (InstId, InstId) {
    let x = lane(prog, block);
    let coords = prog.append(
        block,
        Opcode::CompositeConstructU32x2,
        &[x, Value::U32(5)],
        0,
    );
    let flags = TextureInstInfo {
        image_format: ImageFormat::R32Uint,
        ..info(TextureType::Color2D)
    }
    .to_flags()
        | TextureInstInfo::IMAGE_BYTE_ADDRESS;
    let args = [Value::U32(0x70), Value::Inst(coords), Value::U32(1)];
    let args = if op == Opcode::BoundImageWrite {
        &args[..]
    } else {
        &args[..2]
    };
    (prog.append(block, op, args, flags), coords)
}

#[test]
fn byte_addressed_store_writes_the_texel_of_the_byte() {
    let (mut prog, b) = program(1);
    let (store, coords) = byte_addressed_store(&mut prog, b[0], Opcode::BoundImageWrite);
    let env = Env::default()
        .word(BOUND, 0x70, 0x5)
        .format(0x5, TexturePixelFormat::R32Uint);
    let info = run(&env, &mut prog, &host(1024));

    let image = info.image_descriptors[0];
    assert!(image.is_byte_addressed && image.is_integer);
    let texel = inst_of(&prog, prog.arg(store, 1));
    assert_eq!(prog.op(texel), Opcode::CompositeConstructU32x2);
    let x = inst_of(&prog, prog.arg(texel, 0));
    assert_eq!(prog.op(x), Opcode::ShiftRightArithmetic32);
    assert_eq!(prog.arg(x, 1), Value::U32(2));
    let extract = inst_of(&prog, prog.arg(x, 0));
    assert_eq!(
        prog.inst(extract).args(),
        &[Value::Inst(coords), Value::U32(0)]
    );
}

#[test]
fn byte_addressed_access_rejects_what_it_cannot_express() {
    let (mut prog, b) = program(1);
    byte_addressed_store(&mut prog, b[0], Opcode::BoundImageWrite);
    let env = Env::default()
        .word(BOUND, 0x70, 0x5)
        .format(0x5, TexturePixelFormat::B10g11r11Float);
    compute_post_order(&mut prog);
    let result = texture_pass(&env, &mut prog, &mut ShaderInfo::default(), &host(1024));
    assert!(matches!(
        result,
        Err(TextureError::ByteAddressedFormat {
            format: TexturePixelFormat::B10g11r11Float,
            ..
        })
    ));

    let (mut prog, b) = program(1);
    byte_addressed_store(&mut prog, b[0], Opcode::BoundImageRead);
    compute_post_order(&mut prog);
    let result = texture_pass(&env, &mut prog, &mut ShaderInfo::default(), &host(1024));
    assert_eq!(
        result,
        Err(TextureError::ByteAddressedMode(Opcode::ImageRead))
    );
}

#[test]
fn rectangle_sample_normalizes_its_coordinates() {
    let (mut prog, b) = program(1);
    let id = sample(&mut prog, b[0], Value::U32(0x10));
    let env = Env::default()
        .word(BOUND, 0x10, 0x3)
        .ty(0x3, TextureType::Color2DRect);
    run(&env, &mut prog, &host(1024));

    let coords = inst_of(&prog, prog.arg(id, 1));
    assert_eq!(prog.op(coords), Opcode::CompositeConstructF32x2);
    let x = inst_of(&prog, prog.arg(coords, 0));
    assert_eq!(prog.op(x), Opcode::FPMul32);
    let recip = inst_of(&prog, prog.arg(x, 1));
    assert_eq!(prog.op(recip), Opcode::FPRecip32);
    let width = inst_of(&prog, prog.arg(recip, 0));
    let size = inst_of(&prog, prog.arg(inst_of(&prog, prog.arg(width, 0)), 0));
    assert_eq!(prog.op(size), Opcode::ImageQueryDimensions);
    assert_eq!(prog.inst(size).flags, prog.inst(id).flags);
    assert_eq!(prog.arg(size, 0), Value::U32(0));
}

fn buffer_fetch(prog: &mut Program, block: BlockId) -> (InstId, InstId) {
    let fetch = prog.append(
        block,
        Opcode::BoundImageFetch,
        &[
            Value::U32(0x18),
            Value::U32(3),
            Value::Void,
            Value::Void,
            Value::Void,
        ],
        info(TextureType::Buffer).to_flags(),
    );
    let reference = prog.append(block, Opcode::Reference, &[Value::Inst(fetch)], 0);
    (fetch, reference)
}

#[test]
fn snorm_texel_buffer_fetch_converts_its_components() {
    let (mut prog, b) = program(1);
    let (fetch, reference) = buffer_fetch(&mut prog, b[0]);
    let env = Env::default()
        .word(BOUND, 0x18, 0x4)
        .format(0x4, TexturePixelFormat::R8Snorm);
    run(&env, &mut prog, &host(1024));

    assert_eq!(prog.op(fetch), Opcode::Identity);
    let converted = inst_of(&prog, prog.arg(reference, 0));
    assert_eq!(prog.op(converted), Opcode::CompositeConstructF32x4);
    let scaled = inst_of(&prog, prog.arg(converted, 3));
    assert_eq!(prog.op(scaled), Opcode::FPMul32);
    assert_eq!(prog.arg(scaled, 1), Value::F32(1.0 / 127.0));
    let fetches = prog
        .insts_of(b[0])
        .filter(|&id| prog.op(id) == Opcode::ImageFetch)
        .count();
    assert_eq!(fetches, 1);

    let (mut prog, b) = program(1);
    let (fetch, _) = buffer_fetch(&mut prog, b[0]);
    let snorm_host = HostTranslateInfo {
        support_snorm_render_buffer: true,
        ..host(1024)
    };
    run(&env, &mut prog, &snorm_host);
    assert_eq!(prog.op(fetch), Opcode::ImageFetch);
}

#[test]
fn untracked_bindless_handle_is_an_error() {
    let (mut prog, b) = program(1);
    let handle = lane(&mut prog, b[0]);
    sample(&mut prog, b[0], handle);
    compute_post_order(&mut prog);
    let result = texture_pass(
        &Env::default(),
        &mut prog,
        &mut ShaderInfo::default(),
        &host(1024),
    );
    assert_eq!(
        result,
        Err(TextureError::UntrackedHandle(
            Opcode::BindlessImageSampleImplicitLod
        ))
    );
}

fn loaded_handle(prog: &mut Program, block: BlockId) {
    let address = cbuf(prog, block, 1, Value::U32(0x30));
    let handle = prog.append(block, Opcode::LoadStorage32, &[Value::U32(0), address], 0);
    sample(prog, block, Value::Inst(handle));
}

#[test]
fn handle_loaded_through_a_pointer_binds_no_texture() {
    let (mut prog, b) = program(1);
    loaded_handle(&mut prog, b[0]);
    let info = run(&Env::default(), &mut prog, &host(1024));
    let desc = info.texture_descriptors[0];
    assert_eq!(desc.handle, word_source(1, 0x30));
    assert!(desc.handle_is_pointer_word);
    assert!(info.binds_unmapped_pointer_word);

    let (mut prog, b) = program(1);
    loaded_handle(&mut prog, b[0]);
    let mut info = ShaderInfo::default();
    info.storage_buffers_descriptors
        .push(StorageBufferDescriptor {
            cbuf_index: 1,
            cbuf_offset: 0x30,
            count: 1,
            is_written: false,
        });
    compute_post_order(&mut prog);
    texture_pass(&Env::default(), &mut prog, &mut info, &host(1024)).unwrap();
    assert!(info.texture_descriptors[0].handle_is_pointer_word);
    assert!(!info.binds_unmapped_pointer_word);
}

#[test]
fn unrelated_ir_is_left_alone() {
    let (mut prog, b) = program(1);
    let x = lane(&mut prog, b[0]);
    let word = cbuf(&mut prog, b[0], 1, Value::U32(0x10));
    let sum = prog.append(b[0], Opcode::IAdd32, &[x, word], 0);
    prog.append(b[0], Opcode::Reference, &[Value::Inst(sum)], 0);
    let before: Vec<(Opcode, Vec<Value>)> = prog
        .insts_of(b[0])
        .map(|id| (prog.op(id), prog.inst(id).args().to_vec()))
        .collect();
    let info = run(&Env::default(), &mut prog, &host(1024));
    let after: Vec<(Opcode, Vec<Value>)> = prog
        .insts_of(b[0])
        .map(|id| (prog.op(id), prog.inst(id).args().to_vec()))
        .collect();
    assert_eq!(before, after);
    assert_eq!(info, ShaderInfo::default());
}

#[test]
fn descriptor_reads_are_memoized_per_handle() {
    let (mut prog, b) = program(1);
    for _ in 0..8 {
        prog.append(
            b[0],
            Opcode::BoundImageQueryDimensions,
            &[Value::U32(0x10), Value::U32(0), Value::U1(false)],
            0,
        );
    }
    let word = cbuf(&mut prog, b[0], 1, Value::U32(0x20));
    for _ in 0..8 {
        prog.append(
            b[0],
            Opcode::BindlessImageQueryDimensions,
            &[word, Value::U32(0), Value::U1(false)],
            0,
        );
    }
    let env = Env::default();
    let info = run(&env, &mut prog, &host(1024));
    assert_eq!(info.texture_descriptors.len(), 2);
    assert_eq!(env.type_reads.get(), 2);
    assert_eq!(env.cbuf_reads.get(), 2);
}

#[test]
fn conditional_phi_takes_the_operand_of_the_same_condition() {
    let (mut prog, b) = program(5);
    let (first_if, first_body, second_if, second_body, end) = (b[0], b[1], b[2], b[3], b[4]);
    prog.add_branch(first_if, first_body);
    prog.add_branch(first_if, second_if);
    prog.add_branch(first_body, second_if);
    prog.add_branch(second_if, second_body);
    prog.add_branch(second_if, end);
    prog.add_branch(second_body, end);

    let x = lane(&mut prog, first_if);
    let cond = prog.append(first_if, Opcode::IEqual, &[x, Value::U32(0)], 0);
    let other = cbuf(&mut prog, first_if, 1, Value::U32(0x20));
    prog.append(first_if, Opcode::ConditionRef, &[Value::Inst(cond)], 0);
    let taken = cbuf(&mut prog, first_body, 1, Value::U32(0x10));
    let phi = prog.append(second_if, Opcode::Phi, &[], Type::U32.bits());
    prog.add_phi_operand(phi, first_body, taken);
    prog.add_phi_operand(phi, first_if, other);
    prog.append(second_if, Opcode::ConditionRef, &[Value::Inst(cond)], 0);
    sample(&mut prog, second_body, Value::Inst(phi));
    let info = run(&Env::default(), &mut prog, &host(1024));
    assert_eq!(info.texture_descriptors[0].handle, word_source(1, 0x10));

    let (mut prog, b) = program(1);
    let first = cbuf(&mut prog, b[0], 1, Value::U32(0x10));
    let second = cbuf(&mut prog, b[0], 1, Value::U32(0x20));
    let phi = prog.append(b[0], Opcode::Phi, &[], Type::U32.bits());
    prog.add_phi_operand(phi, b[0], first);
    prog.add_phi_operand(phi, b[0], second);
    sample(&mut prog, b[0], Value::Inst(phi));
    let info = run(&Env::default(), &mut prog, &host(1024));
    assert_eq!(info.texture_descriptors[0].handle, word_source(1, 0x20));
}

#[test]
fn limits_cap_texture_arrays_to_the_host() {
    let (mut prog, b) = program(1);
    let index = lane(&mut prog, b[0]);
    let offset = prog.append(b[0], Opcode::ShiftLeftLogical32, &[index, Value::U32(3)], 0);
    let handle = cbuf(&mut prog, b[0], 1, Value::Inst(offset));
    let id = sample(&mut prog, b[0], handle);
    let mut info = run(&Env::default(), &mut prog, &host(1024));
    assert_eq!(info.texture_descriptors[0].count, 1024);
    let clamp = prog.arg(id, 0);

    limit_sampled_texture_arrays(&mut prog, &mut info, &host(1024)).unwrap();
    assert_eq!(info.texture_descriptors[0].count, 1024);
    limit_sampled_texture_arrays(&mut prog, &mut info, &host(16)).unwrap();
    assert_eq!(info.texture_descriptors[0].count, 16);
    let recap = inst_of(&prog, prog.arg(id, 0));
    assert_eq!(prog.op(recap), Opcode::UMin32);
    assert_eq!(prog.inst(recap).args(), &[clamp, Value::U32(15)]);
}

fn texture(count: u32, is_descriptor_array: bool) -> TextureDescriptor {
    TextureDescriptor {
        ty: TextureType::Color2D,
        is_depth: false,
        is_multisample: false,
        is_integer: false,
        handle: word_source(1, 0x40),
        count,
        size_shift: 3,
        is_descriptor_array,
        handle_is_pointer_word: false,
    }
}

#[test]
fn limits_merge_an_array_cut_to_its_single_texture() {
    let (mut prog, b) = program(1);
    let element = lane(&mut prog, b[0]);
    let flags = |index| {
        TextureInstInfo {
            descriptor_index: index,
            ..info(TextureType::Color2D)
        }
        .to_flags()
    };
    let indexed = prog.append(
        b[0],
        Opcode::ImageSampleImplicitLod,
        &[element, Value::Void, Value::Void, Value::Void],
        flags(0),
    );
    let single = prog.append(
        b[0],
        Opcode::ImageSampleImplicitLod,
        &[Value::Void, Value::Void, Value::Void, Value::Void],
        flags(1),
    );
    let scaled = prog.append(b[0], Opcode::IsTextureScaled, &[Value::U32(1)], 0);
    compute_post_order(&mut prog);
    let mut info = ShaderInfo {
        texture_descriptors: vec![texture(8, true), texture(1, false)],
        ..ShaderInfo::default()
    };

    limit_sampled_texture_arrays(&mut prog, &mut info, &host(2)).unwrap();
    assert_eq!(
        info.texture_descriptors,
        vec![TextureDescriptor {
            is_descriptor_array: true,
            ..texture(1, false)
        }]
    );
    assert_eq!(prog.arg(indexed, 0), Value::Void);
    assert_eq!(flags_of(&prog, indexed).descriptor_index, 0);
    assert_eq!(flags_of(&prog, single).descriptor_index, 0);
    assert_eq!(prog.arg(scaled, 0), Value::U32(0));
}

#[test]
fn limits_reject_textures_that_cannot_fit() {
    let (mut prog, _) = program(1);
    compute_post_order(&mut prog);
    let mut info = ShaderInfo {
        texture_descriptors: vec![texture(4, true)],
        ..ShaderInfo::default()
    };
    for offset in [0x50, 0x58, 0x60] {
        info.texture_descriptors.push(TextureDescriptor {
            handle: word_source(1, offset),
            ..texture(1, false)
        });
    }
    let result = limit_sampled_texture_arrays(&mut prog, &mut info, &host(2));
    assert!(matches!(
        result,
        Err(TextureError::DescriptorLimits { samplers: 4, .. })
    ));
}
