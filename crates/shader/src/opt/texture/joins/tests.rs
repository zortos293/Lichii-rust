use super::*;
use crate::info::{
    HandleSource, ImageBufferDescriptor, ImageDescriptor, StorageBufferDescriptor,
    TextureBufferDescriptor, TextureDescriptor,
};
use crate::ir::{BlockId, ImageFormat, InstId, SyntaxNode};
use crate::opt::compute_post_order;

fn program() -> (Program, BlockId) {
    let mut prog = Program::new();
    let block = prog.new_block();
    prog.block_order = vec![block];
    prog.syntax_list = vec![SyntaxNode::Block(block)];
    (prog, block)
}

fn source(cbuf_offset: u32) -> HandleSource {
    HandleSource {
        cbuf_index: 1,
        cbuf_offset,
        ..HandleSource::default()
    }
}

fn texture(cbuf_offset: u32, count: u32) -> TextureDescriptor {
    TextureDescriptor {
        ty: TextureType::Color2D,
        is_depth: false,
        is_multisample: false,
        is_integer: false,
        handle: source(cbuf_offset),
        count,
        size_shift: 3,
        is_descriptor_array: count > 1,
        handle_is_pointer_word: false,
    }
}

fn texture_buffer(cbuf_offset: u32) -> TextureBufferDescriptor {
    TextureBufferDescriptor {
        handle: source(cbuf_offset),
        count: 1,
        size_shift: 3,
    }
}

fn image(cbuf_offset: u32) -> ImageDescriptor {
    ImageDescriptor {
        ty: TextureType::Color2D,
        format: ImageFormat::R32Uint,
        is_written: false,
        is_read: true,
        is_integer: false,
        cbuf_index: 1,
        cbuf_offset,
        count: 1,
        size_shift: 3,
        is_byte_addressed: false,
    }
}

fn image_buffer(cbuf_offset: u32) -> ImageBufferDescriptor {
    ImageBufferDescriptor {
        format: ImageFormat::R32Uint,
        is_written: false,
        is_read: true,
        is_integer: false,
        cbuf_index: 1,
        cbuf_offset,
        count: 1,
        size_shift: 3,
    }
}

fn flags(ty: TextureType, descriptor_index: u16) -> u32 {
    TextureInstInfo {
        descriptor_index,
        ty,
        ..TextureInstInfo::default()
    }
    .to_flags()
}

fn texture_inst(
    prog: &mut Program,
    block: BlockId,
    op: Opcode,
    element: Value,
    flags: u32,
) -> InstId {
    prog.append(
        block,
        op,
        &[element, Value::Void, Value::Void, Value::Void],
        flags,
    )
}

fn index_of(prog: &Program, id: InstId) -> u16 {
    TextureInstInfo::from_flags(prog.inst(id).flags).descriptor_index
}

#[test]
fn joined_descriptors_reuse_overlapping_ones_and_remap_the_second_program() {
    let mut base = ShaderInfo {
        texture_buffer_descriptors: vec![texture_buffer(0x10)],
        image_buffer_descriptors: vec![image_buffer(0x18)],
        texture_descriptors: vec![texture(0x20, 1), texture(0x28, 1)],
        image_descriptors: vec![image(0x30)],
        ..ShaderInfo::default()
    };
    let second = ShaderInfo {
        texture_buffer_descriptors: vec![texture_buffer(0x50), texture_buffer(0x10)],
        image_buffer_descriptors: vec![image_buffer(0x18)],
        texture_descriptors: vec![
            texture(0x60, 1),
            TextureDescriptor {
                is_integer: true,
                ..texture(0x28, 1)
            },
            texture(0x20, 1),
        ],
        image_descriptors: vec![
            ImageDescriptor {
                is_written: true,
                ..image(0x30)
            },
            image(0x70),
        ],
        requires_coherent_image_memory: true,
        ..ShaderInfo::default()
    };

    let joined = join_texture_info(&mut base, &second);
    assert_eq!(
        joined,
        JoinedDescriptors {
            texture_buffers: vec![1, 0],
            image_buffers: vec![0],
            textures: vec![2, 1, 0],
            images: vec![0, 1],
        }
    );
    assert_eq!(base.texture_buffer_descriptors.len(), 2);
    assert_eq!(base.image_buffer_descriptors.len(), 1);
    assert_eq!(base.texture_descriptors.len(), 3);
    assert_eq!(base.texture_descriptors[2].handle, source(0x60));
    assert!(base.texture_descriptors[1].is_integer);
    assert_eq!(base.image_descriptors.len(), 2);
    assert!(base.image_descriptors[0].is_written);
    assert!(base.requires_coherent_image_memory);

    let (mut prog, b) = program();
    let color = TextureType::Color2D;
    let buffer = TextureType::Buffer;
    let samples = [0, 1, 2].map(|index| {
        texture_inst(
            &mut prog,
            b,
            Opcode::ImageSampleImplicitLod,
            Value::Void,
            flags(color, index),
        )
    });
    let fetches = [0, 1].map(|index| {
        texture_inst(
            &mut prog,
            b,
            Opcode::ImageFetch,
            Value::Void,
            flags(buffer, index),
        )
    });
    let coherent = flags(color, 1) | TextureInstInfo::IMAGE_COHERENT;
    let read = texture_inst(&mut prog, b, Opcode::ImageRead, Value::Void, coherent);
    let buffer_read = texture_inst(
        &mut prog,
        b,
        Opcode::ImageRead,
        Value::Void,
        flags(buffer, 0),
    );
    let texture_scaled = prog.append(b, Opcode::IsTextureScaled, &[Value::U32(0)], 0);
    let image_scaled = prog.append(b, Opcode::IsImageScaled, &[Value::U32(0)], 0);
    compute_post_order(&mut prog);

    remap_texture_descriptors(&mut prog, &joined);
    assert_eq!(samples.map(|id| index_of(&prog, id)), [2, 1, 0]);
    assert_eq!(fetches.map(|id| index_of(&prog, id)), [1, 0]);
    assert_eq!(index_of(&prog, read), 1);
    assert_eq!(
        prog.inst(read).flags,
        flags(color, 1) | TextureInstInfo::IMAGE_COHERENT
    );
    assert_eq!(index_of(&prog, buffer_read), 0);
    assert_eq!(prog.arg(texture_scaled, 0), Value::U32(2));
    assert_eq!(prog.arg(image_scaled, 0), Value::U32(0));
}

#[test]
fn joined_storage_bindings_and_texture_indices_remap_together() {
    let storage = |cbuf_offset| StorageBufferDescriptor {
        cbuf_index: 0,
        cbuf_offset,
        count: 1,
        is_written: false,
        indirect_offset: StorageBufferDescriptor::DIRECT,
        write_extent: Default::default(),
    };
    let mut base = ShaderInfo {
        storage_buffers_descriptors: vec![storage(0x110)],
        texture_descriptors: vec![texture(0x20, 1)],
        ..ShaderInfo::default()
    };
    let second = ShaderInfo {
        storage_buffers_descriptors: vec![storage(0x120), storage(0x110)],
        texture_descriptors: vec![texture(0x20, 1)],
        ..ShaderInfo::default()
    };
    let (mut prog, b) = program();
    let load = prog.append(b, Opcode::LoadStorage32, &[Value::U32(0), Value::U32(0)], 0);
    let sample = texture_inst(
        &mut prog,
        b,
        Opcode::ImageSampleImplicitLod,
        Value::Void,
        flags(TextureType::Color2D, 0),
    );
    compute_post_order(&mut prog);

    let bindings = crate::opt::join_storage_info(&mut base, &second, 64).unwrap();
    let joined = join_texture_info(&mut base, &second);
    remap_texture_descriptors(&mut prog, &joined);
    crate::opt::remap_storage_bindings(&mut prog, &bindings);
    assert_eq!(bindings, vec![1, 0]);
    assert_eq!(prog.arg(load, 0), Value::U32(1));
    assert_eq!(index_of(&prog, sample), 0);
    assert_eq!(base.texture_descriptors.len(), 1);
}

#[test]
fn remapping_leaves_other_instructions_and_unknown_indices_alone() {
    let (mut prog, b) = program();
    let add = prog.append(b, Opcode::IAdd32, &[Value::U32(1), Value::U32(2)], 5);
    let past_end = texture_inst(
        &mut prog,
        b,
        Opcode::ImageSampleImplicitLod,
        Value::Void,
        flags(TextureType::Color2D, 7),
    );
    let dynamic_scale = prog.append(b, Opcode::IsTextureScaled, &[Value::Inst(add)], 0);
    compute_post_order(&mut prog);
    let joined = JoinedDescriptors {
        textures: vec![3],
        ..JoinedDescriptors::default()
    };

    remap_texture_descriptors(&mut prog, &joined);
    assert_eq!(prog.inst(add).flags, 5);
    assert_eq!(prog.inst(add).args(), &[Value::U32(1), Value::U32(2)]);
    assert_eq!(index_of(&prog, past_end), 7);
    assert_eq!(prog.arg(dynamic_scale, 0), Value::Inst(add));
}

fn host(limit: u32) -> HostTranslateInfo {
    HostTranslateInfo {
        max_per_stage_descriptor_sampled_images: 1024,
        max_per_stage_descriptor_samplers: 1024,
        max_descriptor_set_samplers: limit,
        max_descriptor_set_sampled_images: limit,
        max_per_stage_resources: 1024,
        ..HostTranslateInfo::default()
    }
}

/// A program sampling an 8-element texture array at a lane-dependent element and one plain
/// texture, with its info; returns the array sample too.
fn array_program(cbuf_offset: u32) -> (Program, ShaderInfo, InstId) {
    let (mut prog, b) = program();
    let element = Value::Inst(prog.append(b, Opcode::LaneId, &[], 0));
    let color = TextureType::Color2D;
    let indexed = texture_inst(
        &mut prog,
        b,
        Opcode::ImageSampleImplicitLod,
        element,
        flags(color, 0),
    );
    texture_inst(
        &mut prog,
        b,
        Opcode::ImageSampleImplicitLod,
        Value::Void,
        flags(color, 1),
    );
    compute_post_order(&mut prog);
    let info = ShaderInfo {
        texture_descriptors: vec![texture(cbuf_offset, 8), texture(cbuf_offset + 0x100, 1)],
        ..ShaderInfo::default()
    };
    (prog, info, indexed)
}

#[test]
fn pipeline_limits_split_the_set_budget_between_arrays() {
    let (mut vertex, mut vertex_info, vertex_sample) = array_program(0x20);
    let (mut fragment, mut fragment_info, fragment_sample) = array_program(0x40);
    let element = vertex.arg(vertex_sample, 0);

    limit_pipeline_sampled_texture_arrays(
        &mut [
            (&mut vertex, &mut vertex_info),
            (&mut fragment, &mut fragment_info),
        ],
        &host(18),
    )
    .unwrap();
    assert_eq!(vertex_info.texture_descriptors[0].count, 8);

    limit_pipeline_sampled_texture_arrays(
        &mut [
            (&mut vertex, &mut vertex_info),
            (&mut fragment, &mut fragment_info),
        ],
        &host(10),
    )
    .unwrap();
    assert_eq!(vertex_info.texture_descriptors[0].count, 4);
    assert_eq!(fragment_info.texture_descriptors[0].count, 4);
    assert_eq!(vertex_info.texture_descriptors[1].count, 1);
    for (prog, id) in [(&vertex, vertex_sample), (&fragment, fragment_sample)] {
        let clamp = prog.resolve(prog.arg(id, 0)).inst().unwrap();
        assert_eq!(prog.op(clamp), Opcode::UMin32);
        assert_eq!(prog.arg(clamp, 1), Value::U32(3));
    }
    let vertex_clamp = vertex.resolve(vertex.arg(vertex_sample, 0)).inst().unwrap();
    assert_eq!(vertex.arg(vertex_clamp, 0), element);
}

#[test]
fn pipeline_limits_reject_fixed_textures_past_the_set() {
    let (mut vertex, mut vertex_info, _) = array_program(0x20);
    let (mut fragment, mut fragment_info, _) = array_program(0x40);
    let result = limit_pipeline_sampled_texture_arrays(
        &mut [
            (&mut vertex, &mut vertex_info),
            (&mut fragment, &mut fragment_info),
        ],
        &host(3),
    );
    assert_eq!(
        result,
        Err(TextureError::PipelineDescriptorLimits {
            samplers: 4,
            sampled: 4,
        })
    );
}

#[test]
fn shared_stage_limits_count_other_resources_of_every_program() {
    let (mut task, mut task_info, _) = array_program(0x20);
    let (mut mesh, mut mesh_info, _) = array_program(0x40);
    task_info.image_descriptors = vec![image(0x80)];
    mesh_info.image_descriptors = vec![image(0x90)];
    let host = HostTranslateInfo {
        max_per_stage_resources: 12,
        ..host(1024)
    };

    limit_shared_stage_sampled_texture_arrays(
        &mut [(&mut task, &mut task_info), (&mut mesh, &mut mesh_info)],
        &host,
        2,
    )
    .unwrap();
    assert_eq!(task_info.texture_descriptors[0].count, 3);
    assert_eq!(mesh_info.texture_descriptors[0].count, 3);
}
