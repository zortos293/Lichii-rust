use lichii_common::GpuVAddr;

use super::*;
use crate::info::ShaderInfo;
use crate::ir::{
    Attribute, FmzMode, FpControl, ImageFormat, InstId, Patch, Pred, Reg, TextureInstInfo,
    TextureType, Type,
};
use crate::profile::HostTranslateInfo;
use crate::{ShaderArch, ShaderEnvironment, ShaderStage};

const HEADER_BASE: GpuVAddr = 0x1000;

struct Env {
    arch: ShaderArch,
    stage: ShaderStage,
    viewport_transform: u32,
    header: [u8; 0x50],
}

impl Env {
    fn new(stage: ShaderStage) -> Env {
        Env {
            arch: ShaderArch::Maxwell,
            stage,
            viewport_transform: 1,
            header: [0; 0x50],
        }
    }
}

impl ShaderEnvironment for Env {
    fn arch(&self) -> ShaderArch {
        self.arch
    }

    fn stage(&self) -> ShaderStage {
        self.stage
    }

    fn start_address(&self) -> GpuVAddr {
        HEADER_BASE
            + match self.arch {
                ShaderArch::Maxwell => crate::maxwell::ProgramHeader::SIZE,
                ShaderArch::Ampere => crate::ampere::GRAPHICS_HEADER_SIZE,
            }
    }

    fn read_u64(&self, address: GpuVAddr) -> u64 {
        let at = (address - HEADER_BASE) as usize;
        self.header
            .get(at..at + 8)
            .map_or(0, |bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_cbuf(&self, _: u32, _: u32) -> u32 {
        0
    }

    fn shared_memory_size(&self) -> u32 {
        0
    }

    fn workgroup_size(&self) -> [u32; 3] {
        [1, 1, 1]
    }

    fn texture_type(&self, _: u32) -> TextureType {
        TextureType::Color2D
    }

    fn texture_pixel_format(&self, _: u32) -> crate::info::TexturePixelFormat {
        crate::info::TexturePixelFormat::A8b8g8r8Unorm
    }

    fn is_texture_pixel_format_integer(&self, _: u32) -> bool {
        false
    }

    fn viewport_transform_state(&self) -> u32 {
        self.viewport_transform
    }

    fn texture_bound_buffer(&self) -> u32 {
        2
    }

    fn local_memory_size(&self) -> u32 {
        0
    }

    fn has_hle_macro_state(&self) -> bool {
        false
    }
}

fn program(blocks: usize) -> (Program, Vec<BlockId>) {
    let mut prog = Program::new();
    let ids: Vec<BlockId> = (0..blocks).map(|_| prog.new_block()).collect();
    prog.block_order = ids.clone();
    prog.post_order = ids.iter().rev().copied().collect();
    prog.syntax_list = ids.iter().map(|&b| SyntaxNode::Block(b)).collect();
    (prog, ids)
}

fn reg(prog: &mut Program, block: BlockId, index: u8) -> Value {
    Value::Inst(prog.append(block, Opcode::GetRegister, &[Value::Reg(Reg::r(index))], 0))
}

fn add(prog: &mut Program, block: BlockId, op: Opcode, args: &[Value]) -> Value {
    Value::Inst(prog.append(block, op, args, 0))
}

fn inst(prog: &Program, value: Value) -> InstId {
    prog.resolve(value)
        .inst()
        .unwrap_or_else(|| panic!("{value:?} is not an instruction"))
}

fn collect(env: &Env, prog: &Program) -> ShaderInfo {
    let mut info = ShaderInfo::default();
    collect_shader_info(env, prog, &mut info);
    info
}

#[test]
fn collect_records_constant_buffer_reads() {
    let (mut prog, b) = program(1);
    let offset = reg(&mut prog, b[0], 0);
    add(&mut prog, b[0], Opcode::GetCbufU8, &[Value::U32(3), offset]);
    add(
        &mut prog,
        b[0],
        Opcode::GetCbufU32,
        &[Value::U32(1), Value::U32(0x24)],
    );
    add(
        &mut prog,
        b[0],
        Opcode::GetCbufU32x2,
        &[Value::U32(1), Value::U32(0x8)],
    );
    let info = collect(&Env::new(ShaderStage::VertexB), &prog);
    assert_eq!(info.constant_buffer_mask, 0b1010);
    let indices: Vec<u32> = info
        .constant_buffer_descriptors
        .iter()
        .map(|d| d.index)
        .collect();
    assert_eq!(indices, [1, 3]);
    assert_eq!(info.constant_buffer_used_sizes[1], 0x30);
    assert_eq!(info.constant_buffer_used_sizes[3], 0x1_0000);
    assert_eq!(
        info.used_constant_buffer_types,
        Type::U8 | Type::U32 | Type::U32X2
    );
    assert!(info.uses_int8);
    assert!(!info.uses_cbuf_indirect);
}

#[test]
fn collect_binds_every_indirect_cbuf_for_a_register_indexed_read() {
    let (mut prog, b) = program(1);
    let index = reg(&mut prog, b[0], 0);
    add(&mut prog, b[0], Opcode::GetCbufF32, &[index, Value::U32(0)]);
    let info = collect(&Env::new(ShaderStage::Fragment), &prog);
    assert!(info.uses_cbuf_indirect);
    assert_eq!(
        info.constant_buffer_descriptors.len(),
        ShaderInfo::MAX_INDIRECT_CBUFS
    );
    assert!(
        info.constant_buffer_used_sizes[..ShaderInfo::MAX_INDIRECT_CBUFS]
            .iter()
            .all(|&size| size == 0x1_0000)
    );
    assert_eq!(info.used_indirect_cbuf_types, Type::F32);
    assert_eq!(info.used_constant_buffer_types, Type::VOID);
    assert_eq!(info.nvn_buffer_used, u16::MAX);
}

#[test]
fn collect_marks_nvn_storage_descriptors_of_the_stage() {
    let (mut prog, b) = program(1);
    add(
        &mut prog,
        b[0],
        Opcode::GetCbufU32,
        &[Value::U32(0), Value::U32(0x130)],
    );
    add(
        &mut prog,
        b[0],
        Opcode::GetCbufU32,
        &[Value::U32(0), Value::U32(0x510)],
    );
    add(
        &mut prog,
        b[0],
        Opcode::GetCbufU32,
        &[Value::U32(1), Value::U32(0x110)],
    );
    let vertex = collect(&Env::new(ShaderStage::VertexB), &prog);
    assert_eq!(vertex.nvn_buffer_base, 0x110);
    assert_eq!(vertex.nvn_buffer_used, 1 << 2);
    let fragment = collect(&Env::new(ShaderStage::Fragment), &prog);
    assert_eq!(fragment.nvn_buffer_base, 0x510);
    assert_eq!(fragment.nvn_buffer_used, 1 << 0);

    let (mut prog, b) = program(1);
    let offset = reg(&mut prog, b[0], 0);
    add(
        &mut prog,
        b[0],
        Opcode::GetCbufU32,
        &[Value::U32(0), offset],
    );
    assert_eq!(
        collect(&Env::new(ShaderStage::Compute), &prog).nvn_buffer_used,
        u16::MAX
    );
}

#[test]
fn collect_records_attributes_patches_and_fragment_outputs() {
    let (mut prog, b) = program(1);
    let value = reg(&mut prog, b[0], 0);
    add(
        &mut prog,
        b[0],
        Opcode::GetAttribute,
        &[Value::Attribute(Attribute::POSITION_X), Value::U32(0)],
    );
    add(
        &mut prog,
        b[0],
        Opcode::SetAttribute,
        &[
            Value::Attribute(Attribute::generic(2, 1)),
            value,
            Value::U32(0),
        ],
    );
    add(
        &mut prog,
        b[0],
        Opcode::GetPatch,
        &[Value::Patch(Patch(Patch::COMPONENT_0.0 + 9))],
    );
    add(
        &mut prog,
        b[0],
        Opcode::SetPatch,
        &[Value::Patch(Patch::LOD_TOP), value],
    );
    add(
        &mut prog,
        b[0],
        Opcode::SetPatch,
        &[Value::Patch(Patch::LOD_INTERIOR_V), value],
    );
    add(
        &mut prog,
        b[0],
        Opcode::SetFragColor,
        &[Value::U32(1), Value::U32(2), value],
    );
    add(
        &mut prog,
        b[0],
        Opcode::SetFragColor,
        &[Value::U32(1), Value::U32(0), value],
    );
    add(&mut prog, b[0], Opcode::SetFragDepth, &[value]);
    add(&mut prog, b[0], Opcode::DemoteToHelperInvocation, &[]);
    let info = collect(&Env::new(ShaderStage::Fragment), &prog);
    assert!(info.loads.get(Attribute::POSITION_X));
    assert!(info.stores.generic_component(2, 1));
    assert!(info.uses_patches[2]);
    assert_eq!(info.uses_patches.iter().filter(|&&used| used).count(), 1);
    assert!(info.stores_tess_level_outer && info.stores_tess_level_inner);
    assert_eq!(
        info.stores_frag_color,
        [false, true, false, false, false, false, false, false]
    );
    assert_eq!(info.frag_color_components[1], 0b101);
    assert!(info.stores_frag_depth && !info.stores_sample_mask);
    assert!(info.uses_demote_to_helper_invocation);
}

#[test]
fn collect_records_float_modes_and_widths() {
    let (mut prog, b) = program(1);
    let value = reg(&mut prog, b[0], 0);
    let ftz = FpControl {
        fmz_mode: FmzMode::Ftz,
        ..FpControl::default()
    }
    .to_flags();
    let keep = FpControl {
        fmz_mode: FmzMode::None,
        ..FpControl::default()
    }
    .to_flags();
    prog.append(b[0], Opcode::FPAdd32, &[value, value], ftz);
    prog.append(b[0], Opcode::FPMul16, &[value, value], keep);
    add(&mut prog, b[0], Opcode::FPAdd64, &[value, value]);
    add(&mut prog, b[0], Opcode::ConvertS16F32, &[value]);
    let info = collect(&Env::new(ShaderStage::Compute), &prog);
    assert!(info.uses_fp32_denorms_flush && !info.uses_fp32_denorms_preserve);
    assert!(info.uses_fp16_denorms_preserve && !info.uses_fp16_denorms_flush);
    assert!(info.uses_fp16 && info.uses_fp64 && info.uses_int16);
    assert!(!info.uses_int8 && !info.uses_int64);
}

#[test]
fn collect_records_global_memory_and_atomics() {
    let (mut prog, b) = program(1);
    let address = reg(&mut prog, b[0], 0);
    prog.append(b[0], Opcode::LoadGlobal32, &[address], 1);
    add(&mut prog, b[0], Opcode::WriteGlobal32, &[address, address]);
    add(
        &mut prog,
        b[0],
        Opcode::StorageAtomicMaxF16x2,
        &[Value::U32(0), Value::U32(0), address],
    );
    add(
        &mut prog,
        b[0],
        Opcode::SharedAtomicInc32,
        &[address, address],
    );
    let info = collect(&Env::new(ShaderStage::Compute), &prog);
    assert!(info.uses_global_memory && info.stores_global_memory && info.uses_int64);
    assert!(info.requires_coherent_global_memory);
    assert_eq!(info.used_constant_buffer_types, Type::U32 | Type::U32X2);
    assert_eq!(
        info.used_storage_buffer_types,
        Type::U32 | Type::U32X2 | Type::U32X4
    );
    assert!(info.uses_atomic_f16x2_max && info.uses_fp16);
    assert!(info.uses_shared_increment);

    let (mut prog, b) = program(1);
    let address = reg(&mut prog, b[0], 0);
    add(&mut prog, b[0], Opcode::LoadGlobal32, &[address]);
    let info = collect(&Env::new(ShaderStage::Compute), &prog);
    assert!(info.uses_global_memory && !info.stores_global_memory);
    assert!(!info.requires_coherent_global_memory);
}

#[test]
fn collect_records_texture_and_image_features() {
    let (mut prog, b) = program(1);
    let coords = reg(&mut prog, b[0], 0);
    let depth_1d = TextureInstInfo {
        ty: TextureType::Color1D,
        is_depth: true,
        ..TextureInstInfo::default()
    };
    let sample = prog.append(
        b[0],
        Opcode::ImageSampleImplicitLod,
        &[Value::U32(0), coords],
        depth_1d.to_flags(),
    );
    add(
        &mut prog,
        b[0],
        Opcode::GetSparseFromOp,
        &[Value::Inst(sample)],
    );
    let buffer = TextureInstInfo {
        ty: TextureType::Buffer,
        image_format: ImageFormat::Typeless,
        ..TextureInstInfo::default()
    };
    prog.append(
        b[0],
        Opcode::ImageWrite,
        &[Value::U32(0), coords, coords],
        buffer.to_flags() | TextureInstInfo::IMAGE_COHERENT,
    );
    let info = collect(&Env::new(ShaderStage::Fragment), &prog);
    assert!(info.uses_sampled_1d && info.uses_shadow_lod && info.uses_sparse_residency);
    assert!(info.uses_image_buffers && info.uses_typeless_image_writes);
    assert!(info.requires_coherent_image_memory);
    assert!(!info.uses_image_1d && !info.uses_typeless_image_reads);
}

#[test]
fn collect_marks_task_and_mesh_invocation_ids() {
    let (prog, _) = program(1);
    let task = collect(&Env::new(ShaderStage::Task), &prog);
    assert!(task.uses_workgroup_id && task.uses_local_invocation_id);
    assert_eq!(task.nvn_buffer_base, 0x110);
    let mesh = collect(&Env::new(ShaderStage::Mesh), &prog);
    assert!(mesh.uses_workgroup_id && mesh.uses_local_invocation_id);
    assert_eq!(mesh.nvn_buffer_base, 0x310);
}

#[test]
fn collect_leaves_unrelated_programs_empty() {
    let (mut prog, b) = program(1);
    let value = reg(&mut prog, b[0], 0);
    add(&mut prog, b[0], Opcode::IAdd32, &[value, value]);
    let info = collect(&Env::new(ShaderStage::VertexB), &prog);
    assert_eq!(
        info,
        ShaderInfo {
            nvn_buffer_base: 0x110,
            ..ShaderInfo::default()
        }
    );
}

fn vtg_header() -> [u8; 0x50] {
    let mut header = [0u8; 0x50];
    header[0x17] = 0x10;
    header[0x18] = 0x0f;
    header[0x2b] = 0x40;
    header[0x35] = 0x02;
    header[0x36] = 0x10;
    header[0x48] = 0b101;
    header
}

fn indexed_attribute_program() -> Program {
    let (mut prog, b) = program(1);
    let address = reg(&mut prog, b[0], 0);
    add(
        &mut prog,
        b[0],
        Opcode::GetAttributeIndexed,
        &[address, Value::U32(0)],
    );
    add(
        &mut prog,
        b[0],
        Opcode::SetAttributeIndexed,
        &[address, address, Value::U32(0)],
    );
    prog
}

fn assert_vtg_header_maps(info: &ShaderInfo) {
    assert!((0..4).all(|c| info.loads.generic_component(0, c)));
    assert!(!info.loads.generic(1));
    assert!(info.loads.get(Attribute::POSITION_X) && !info.loads.get(Attribute::POSITION_Y));
    assert!(info.loads.get(Attribute::INSTANCE_ID));
    assert!(info.stores.generic_component(1, 0) && !info.stores.generic(0));
    assert!(info.stores.get(Attribute::LAYER));
    assert!(info.stores.get(Attribute::CLIP_DISTANCE_0));
    assert!(!info.stores.get(Attribute::CLIP_DISTANCE_0.offset(1)));
    assert!(info.stores.get(Attribute::CLIP_DISTANCE_0.offset(2)));
    assert_eq!(info.used_clip_distances, 3);
}

#[test]
fn collect_reads_indexed_attribute_maps_from_the_maxwell_header() {
    let mut env = Env::new(ShaderStage::VertexB);
    env.header = vtg_header();
    assert_vtg_header_maps(&collect(&env, &indexed_attribute_program()));
}

#[test]
fn collect_reads_indexed_attribute_maps_from_the_ampere_header() {
    let mut env = Env::new(ShaderStage::VertexB);
    env.arch = ShaderArch::Ampere;
    env.header = vtg_header();
    assert_vtg_header_maps(&collect(&env, &indexed_attribute_program()));
}

#[test]
fn collect_reads_fragment_input_maps_from_the_header() {
    let mut env = Env::new(ShaderStage::Fragment);
    env.header[0x18 + 3] = 0b10_00_01_00;
    let (mut prog, b) = program(1);
    let address = reg(&mut prog, b[0], 0);
    add(
        &mut prog,
        b[0],
        Opcode::GetAttributeIndexed,
        &[address, Value::U32(0)],
    );
    let info = collect(&env, &prog);
    let components: Vec<bool> = (0..4).map(|c| info.loads.generic_component(3, c)).collect();
    assert_eq!(components, [false, true, false, true]);
    assert!(!info.loads.generic(0));
}

#[test]
fn conditional_barrier_drops_barriers_under_conditions_only() {
    let (mut prog, b) = program(3);
    let kept = prog.append(b[0], Opcode::Barrier, &[], 0);
    let dropped = prog.append(b[1], Opcode::Barrier, &[], 0);
    let kept_after = prog.append(b[2], Opcode::Barrier, &[], 0);
    prog.syntax_list = vec![
        SyntaxNode::Block(b[0]),
        SyntaxNode::If {
            cond: Value::U1(true),
            body: b[1],
            merge: b[2],
        },
        SyntaxNode::Block(b[1]),
        SyntaxNode::EndIf { merge: b[2] },
        SyntaxNode::Block(b[2]),
        SyntaxNode::Return,
    ];
    let mut info = ShaderInfo::default();
    conditional_barrier(&mut prog, &mut info);
    assert!(info.removed_conditional_barrier);
    assert_eq!(prog.op(kept), Opcode::Barrier);
    assert_eq!(prog.op(kept_after), Opcode::Barrier);
    assert_eq!(prog.op(dropped), Opcode::Void);
    assert!(prog.block(b[1]).is_empty());
}

#[test]
fn conditional_barrier_drops_barriers_after_a_conditional_return() {
    let (mut prog, b) = program(3);
    let after = prog.append(b[2], Opcode::Barrier, &[], 0);
    prog.syntax_list = vec![
        SyntaxNode::Block(b[0]),
        SyntaxNode::If {
            cond: Value::U1(true),
            body: b[1],
            merge: b[2],
        },
        SyntaxNode::Block(b[1]),
        SyntaxNode::Return,
        SyntaxNode::EndIf { merge: b[2] },
        SyntaxNode::Block(b[2]),
    ];
    let mut info = ShaderInfo::default();
    conditional_barrier(&mut prog, &mut info);
    assert_eq!(prog.op(after), Opcode::Void);
    assert!(info.removed_conditional_barrier);
}

#[test]
fn conditional_barrier_keeps_unconditional_barriers() {
    let (mut prog, b) = program(1);
    let barrier = prog.append(b[0], Opcode::Barrier, &[], 0);
    let mut info = ShaderInfo::default();
    conditional_barrier(&mut prog, &mut info);
    assert_eq!(prog.op(barrier), Opcode::Barrier);
    assert!(!info.removed_conditional_barrier);
}

fn layer_program() -> (Program, InstId, InstId) {
    let (mut prog, b) = program(1);
    let value = reg(&mut prog, b[0], 0);
    let layer = prog.append(
        b[0],
        Opcode::SetAttribute,
        &[Value::Attribute(Attribute::LAYER), value, Value::U32(0)],
        0,
    );
    let position = prog.append(
        b[0],
        Opcode::SetAttribute,
        &[
            Value::Attribute(Attribute::POSITION_X),
            value,
            Value::U32(0),
        ],
        0,
    );
    (prog, layer, position)
}

#[test]
fn layer_moves_layer_stores_to_a_free_generic() {
    let (mut prog, layer_store, position_store) = layer_program();
    let mut info = ShaderInfo::default();
    info.stores.set(Attribute::generic(0, 2), true);
    info.stores.set(Attribute::LAYER, true);
    layer(
        &mut prog,
        ShaderStage::VertexB,
        &mut info,
        &HostTranslateInfo::default(),
    );
    let emulated = Attribute::generic(1, 0);
    assert_eq!(prog.arg(layer_store, 0), Value::Attribute(emulated));
    assert_eq!(
        prog.arg(position_store, 0),
        Value::Attribute(Attribute::POSITION_X)
    );
    assert!(info.requires_layer_emulation);
    assert_eq!(info.emulated_layer, Some(emulated));
    assert!(info.stores.get(emulated) && !info.stores.get(Attribute::LAYER));
}

#[test]
fn layer_keeps_native_layer_writes() {
    let host = HostTranslateInfo {
        support_viewport_index_layer: true,
        ..HostTranslateInfo::default()
    };
    for (stage, host) in [
        (ShaderStage::VertexB, &host),
        (ShaderStage::Geometry, &HostTranslateInfo::default()),
    ] {
        let (mut prog, layer_store, _) = layer_program();
        let mut info = ShaderInfo::default();
        layer(&mut prog, stage, &mut info, host);
        assert_eq!(prog.arg(layer_store, 0), Value::Attribute(Attribute::LAYER));
        assert!(!info.requires_layer_emulation);
    }
}

#[test]
fn position_maps_window_coordinates_to_ndc() {
    let (mut prog, b) = program(1);
    let value = Value::F32(3.0);
    let x = prog.append(
        b[0],
        Opcode::SetAttribute,
        &[
            Value::Attribute(Attribute::POSITION_X),
            value,
            Value::U32(0),
        ],
        0,
    );
    let y = prog.append(
        b[0],
        Opcode::SetAttribute,
        &[
            Value::Attribute(Attribute::POSITION_Y),
            value,
            Value::U32(0),
        ],
        0,
    );
    let z = prog.append(
        b[0],
        Opcode::SetAttribute,
        &[
            Value::Attribute(Attribute::POSITION_Z),
            value,
            Value::U32(0),
        ],
        0,
    );
    let mut env = Env::new(ShaderStage::VertexB);
    env.viewport_transform = 0;
    let mut info = ShaderInfo::default();
    position(&env, &mut prog, &mut info);
    assert!(info.uses_render_area);
    assert_eq!(prog.arg(z, 1), value);
    for (store, element) in [(x, 0), (y, 1)] {
        let fma = inst(&prog, prog.arg(store, 1));
        assert_eq!(prog.op(fma), Opcode::FPFma32);
        assert_eq!(prog.arg(fma, 0), value);
        assert_eq!(prog.arg(fma, 2), Value::F32(-1.0));
        let mul = inst(&prog, prog.arg(fma, 1));
        assert_eq!(prog.op(mul), Opcode::FPMul32);
        assert_eq!(prog.arg(mul, 1), Value::F32(2.0));
        let recip = inst(&prog, prog.arg(mul, 0));
        assert_eq!(prog.op(recip), Opcode::FPRecip32);
        let extract = inst(&prog, prog.arg(recip, 0));
        assert_eq!(prog.op(extract), Opcode::CompositeExtractF32x4);
        assert_eq!(prog.arg(extract, 1), Value::U32(element));
        assert_eq!(
            prog.op(inst(&prog, prog.arg(extract, 0))),
            Opcode::RenderArea
        );
    }
}

#[test]
fn position_needs_a_disabled_viewport_transform() {
    let (mut prog, b) = program(1);
    let x = prog.append(
        b[0],
        Opcode::SetAttribute,
        &[
            Value::Attribute(Attribute::POSITION_X),
            Value::F32(1.0),
            Value::U32(0),
        ],
        0,
    );
    let mut info = ShaderInfo::default();
    position(&Env::new(ShaderStage::VertexB), &mut prog, &mut info);
    assert_eq!(prog.arg(x, 1), Value::F32(1.0));
    assert!(!info.uses_render_area);
}

fn blit_read(prog: &mut Program, block: BlockId, op: Opcode, offset: u32) -> InstId {
    let read = prog.append(block, op, &[Value::U32(17), Value::U32(offset)], 0);
    prog.append(block, Opcode::Reference, &[Value::Inst(read)], 0);
    read
}

#[test]
fn blit_constants_synthesizes_the_parameter_block() {
    let (mut prog, b) = program(1);
    let size = blit_read(&mut prog, b[0], Opcode::GetCbufU32, 0x8);
    let scale = blit_read(&mut prog, b[0], Opcode::GetCbufF32, 0x18);
    let clamp = blit_read(&mut prog, b[0], Opcode::GetCbufU32x2, 0x38);
    let dither = blit_read(&mut prog, b[0], Opcode::GetCbufU32, 0x40);
    let mut info = ShaderInfo::default();
    blit_constants(&Env::new(ShaderStage::VertexB), &mut prog, &mut info);
    assert!(info.uses_render_area);
    let width = inst(&prog, Value::Inst(size));
    assert_eq!(prog.op(width), Opcode::BitCastU32F32);
    let extract = inst(&prog, prog.arg(width, 0));
    assert_eq!(prog.arg(extract, 1), Value::U32(0));
    assert_eq!(prog.resolve(Value::Inst(scale)), Value::F32(1.0));
    let pair = inst(&prog, Value::Inst(clamp));
    assert_eq!(prog.op(pair), Opcode::CompositeConstructU32x2);
    assert_eq!(prog.inst(pair).args(), [Value::U32(0x3f80_0000); 2]);
    assert_eq!(prog.resolve(Value::Inst(dither)), Value::U32(0));
}

#[test]
fn blit_constants_keeps_tables_the_game_bound() {
    for offsets in [&[0x8, 0x20][..], &[0x40, 0x44][..]] {
        let (mut prog, b) = program(1);
        let reads: Vec<InstId> = offsets
            .iter()
            .map(|&offset| blit_read(&mut prog, b[0], Opcode::GetCbufU32, offset))
            .collect();
        let mut info = ShaderInfo::default();
        blit_constants(&Env::new(ShaderStage::Fragment), &mut prog, &mut info);
        assert!(
            reads
                .iter()
                .all(|&read| prog.op(read) == Opcode::GetCbufU32)
        );
        assert!(!info.uses_render_area);
    }
}

#[test]
fn blit_constants_drops_the_native_dither_bias() {
    let (mut prog, b) = program(1);
    let color = reg(&mut prog, b[0], 0);
    let bits = add(&mut prog, b[0], Opcode::BitCastU32F32, &[color]);
    let bias = add(
        &mut prog,
        b[0],
        Opcode::GetCbufU32,
        &[Value::U32(17), Value::U32(0x14)],
    );
    let biased = add(&mut prog, b[0], Opcode::IAdd32, &[bias, bits]);
    let cast = prog.append(b[0], Opcode::BitCastF32U32, &[biased], 0);
    let mut env = Env::new(ShaderStage::Fragment);
    env.arch = ShaderArch::Ampere;
    blit_constants(&env, &mut prog, &mut ShaderInfo::default());
    assert_eq!(prog.arg(cast, 0), bits);

    let (mut prog, b) = program(1);
    let color = reg(&mut prog, b[0], 0);
    let bias = add(
        &mut prog,
        b[0],
        Opcode::GetCbufU32,
        &[Value::U32(17), Value::U32(0x14)],
    );
    let biased = add(&mut prog, b[0], Opcode::IAdd32, &[bias, color]);
    let cast = prog.append(b[0], Opcode::BitCastF32U32, &[biased], 0);
    blit_constants(&env, &mut prog, &mut ShaderInfo::default());
    assert_eq!(prog.arg(cast, 0), biased);
}

struct SlotProgram {
    prog: Program,
    blocks: Vec<BlockId>,
    pred: InstId,
    amount: Value,
    atomic: InstId,
    user: InstId,
}

fn slot_program() -> SlotProgram {
    let (mut prog, blocks) = program(1);
    let pred = prog.append(blocks[0], Opcode::GetPred, &[Value::Pred(Pred::p(0))], 0);
    let ballot = add(
        &mut prog,
        blocks[0],
        Opcode::SubgroupBallot,
        &[Value::Inst(pred)],
    );
    let amount = add(&mut prog, blocks[0], Opcode::BitCount32, &[ballot]);
    let atomic = prog.append(
        blocks[0],
        Opcode::StorageAtomicIAdd32,
        &[Value::U32(0), Value::U32(0x10), amount],
        0,
    );
    let user = prog.append(blocks[0], Opcode::Reference, &[Value::Inst(atomic)], 0);
    SlotProgram {
        prog,
        blocks,
        pred,
        amount,
        atomic,
        user,
    }
}

fn single_lane() -> HostTranslateInfo {
    HostTranslateInfo {
        single_lane_geometry_subgroups: true,
        ..HostTranslateInfo::default()
    }
}

#[test]
fn geometry_compaction_gives_each_primitive_its_slot() {
    let SlotProgram {
        mut prog,
        pred,
        amount,
        atomic,
        user,
        ..
    } = slot_program();
    geometry_compaction(&mut prog, ShaderStage::Geometry, 3, &single_lane());
    assert_eq!(prog.op(atomic), Opcode::Identity);
    let slot = inst(&prog, prog.arg(user, 0));
    assert_eq!(prog.op(slot), Opcode::IMul32);
    assert_eq!(prog.arg(slot, 1), amount);
    let key = inst(&prog, prog.arg(slot, 0));
    assert_eq!(prog.op(key), Opcode::IAdd32);
    let scaled = inst(&prog, prog.arg(key, 0));
    assert_eq!(prog.arg(scaled, 1), Value::U32(3));
    let primitive = inst(&prog, prog.arg(scaled, 0));
    assert_eq!(
        prog.arg(primitive, 0),
        Value::Attribute(Attribute::PRIMITIVE_ID)
    );
    assert_eq!(prog.op(inst(&prog, prog.arg(key, 1))), Opcode::InvocationId);
    let umax = prog
        .inst(slot)
        .next()
        .and_then(|id| prog.inst(id).next())
        .unwrap();
    assert_eq!(prog.op(umax), Opcode::StorageAtomicUMax32);
    assert_eq!(
        &prog.inst(umax).args()[..2],
        [Value::U32(0), Value::U32(0x10)]
    );
    let end = inst(&prog, prog.arg(umax, 2));
    assert_eq!(prog.inst(end).args(), [Value::Inst(slot), amount]);
    assert_eq!(prog.resolve(Value::Inst(pred)), Value::U1(true));
}

#[test]
fn geometry_compaction_rewrites_the_broadcast_slot_in_phis() {
    let SlotProgram {
        mut prog,
        mut blocks,
        atomic,
        ..
    } = slot_program();
    blocks.push(prog.new_block());
    blocks.push(prog.new_block());
    let (merge, other) = (blocks[1], blocks[2]);
    prog.add_branch(blocks[0], merge);
    prog.add_branch(other, merge);
    let amount = reg(&mut prog, other, 5);
    let shuffle = add(
        &mut prog,
        other,
        Opcode::ShuffleIndex,
        &[amount, Value::U32(0), Value::U32(31), Value::U32(0)],
    );
    let phi = prog.append(merge, Opcode::Phi, &[], Type::U32.bits());
    prog.add_phi_operand(phi, blocks[0], Value::Inst(atomic));
    prog.add_phi_operand(phi, other, shuffle);
    prog.post_order = vec![merge, other, blocks[0]];
    geometry_compaction(&mut prog, ShaderStage::Geometry, 1, &single_lane());
    let slot = inst(&prog, prog.arg(phi, 1));
    assert_eq!(prog.op(slot), Opcode::IMul32);
    assert_eq!(prog.inst(slot).block(), other);
    assert_eq!(prog.arg(slot, 1), amount);
    assert_eq!(
        prog.op(inst(&prog, prog.arg(slot, 0))),
        Opcode::GetAttributeU32
    );
    assert_eq!(prog.op(inst(&prog, prog.arg(phi, 0))), Opcode::IMul32);
}

#[test]
fn geometry_compaction_leaves_other_atomics_and_hosts_alone() {
    let SlotProgram {
        mut prog, atomic, ..
    } = slot_program();
    geometry_compaction(
        &mut prog,
        ShaderStage::Geometry,
        1,
        &HostTranslateInfo::default(),
    );
    assert_eq!(prog.op(atomic), Opcode::StorageAtomicIAdd32);
    geometry_compaction(&mut prog, ShaderStage::VertexB, 1, &single_lane());
    assert_eq!(prog.op(atomic), Opcode::StorageAtomicIAdd32);

    let (mut prog, b) = program(1);
    let amount = reg(&mut prog, b[0], 0);
    let plain = prog.append(
        b[0],
        Opcode::StorageAtomicIAdd32,
        &[Value::U32(0), Value::U32(0), amount],
        0,
    );
    geometry_compaction(&mut prog, ShaderStage::Geometry, 1, &single_lane());
    assert_eq!(prog.op(plain), Opcode::StorageAtomicIAdd32);
}

fn byte_swap(with_mul: bool, shift: u32) -> (Program, Value, InstId) {
    let (mut prog, b) = program(1);
    let a = reg(&mut prog, b[0], 0);
    let factor = reg(&mut prog, b[0], 1);
    let low = add(
        &mut prog,
        b[0],
        Opcode::BitFieldUExtract,
        &[a, Value::U32(0), Value::U32(16)],
    );
    let low = if with_mul {
        add(&mut prog, b[0], Opcode::IMul32, &[low, factor])
    } else {
        low
    };
    let shl = add(
        &mut prog,
        b[0],
        Opcode::ShiftLeftLogical32,
        &[low, Value::U32(shift)],
    );
    let high = add(
        &mut prog,
        b[0],
        Opcode::BitFieldUExtract,
        &[a, Value::U32(16), Value::U32(16)],
    );
    let sum = add(&mut prog, b[0], Opcode::IAdd32, &[shl, high]);
    let user = prog.append(b[0], Opcode::Reference, &[sum], 0);
    (prog, shl, user)
}

#[test]
fn vendor_workaround_turns_half_swap_adds_into_ors() {
    for with_mul in [true, false] {
        let (mut prog, shl, user) = byte_swap(with_mul, 16);
        vendor_workaround(&mut prog);
        let or = inst(&prog, prog.arg(user, 0));
        assert_eq!(prog.op(or), Opcode::BitwiseOr32);
        assert_eq!(prog.arg(or, 0), shl);
    }
}

#[test]
fn vendor_workaround_keeps_other_adds() {
    let (mut prog, _, user) = byte_swap(true, 8);
    vendor_workaround(&mut prog);
    assert_eq!(prog.op(inst(&prog, prog.arg(user, 0))), Opcode::IAdd32);
}

fn shuffled_attribute(attribute: Attribute) -> Program {
    let (mut prog, b) = program(1);
    let value = add(
        &mut prog,
        b[0],
        Opcode::GetAttribute,
        &[Value::Attribute(attribute), Value::U32(0)],
    );
    let bits = add(&mut prog, b[0], Opcode::BitCastU32F32, &[value]);
    add(
        &mut prog,
        b[0],
        Opcode::ShuffleButterfly,
        &[bits, Value::U32(1), Value::U32(31), Value::U32(0)],
    );
    prog
}

#[test]
fn has_broken_pattern_finds_shuffled_fragment_positions() {
    let prog = shuffled_attribute(Attribute::POSITION_Y);
    assert!(has_broken_pattern(&prog, ShaderStage::Fragment));
    assert!(!has_broken_pattern(&prog, ShaderStage::VertexB));
    let prog = shuffled_attribute(Attribute::POSITION_Z);
    assert!(!has_broken_pattern(&prog, ShaderStage::Fragment));
}

#[test]
fn dual_vertex_transforms_drop_the_joined_end() {
    let (mut prog, b) = program(2);
    let prologue = prog.append(b[0], Opcode::Prologue, &[], 0);
    let barrier = prog.append(b[0], Opcode::Barrier, &[], 0);
    let epilogue = prog.append(b[1], Opcode::Epilogue, &[], 0);
    let mut a = prog.clone();
    vertex_a_transform(&mut a);
    assert_eq!(
        [a.op(prologue), a.op(barrier), a.op(epilogue)],
        [Opcode::Prologue, Opcode::Barrier, Opcode::Void]
    );
    vertex_b_transform(&mut prog);
    assert_eq!(
        [prog.op(prologue), prog.op(barrier), prog.op(epilogue)],
        [Opcode::Void, Opcode::Barrier, Opcode::Epilogue]
    );
}
