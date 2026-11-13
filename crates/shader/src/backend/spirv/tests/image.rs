//! Image and image atomic family tests: every opcode is emitted in a small synthetic program
//! and the module validated with spirv-val.

use rspirv::dr::{self, Operand};
use rspirv::spirv::{Capability, Decoration, Op};

use super::{Synthetic, count_ops, emit, validate};
use crate::backend::Bindings;
use crate::backend::spirv::{EmitError, emit_spirv};
use crate::info::{
    HandleSource, ImageBufferDescriptor, ImageDescriptor, TextureBufferDescriptor,
    TextureDescriptor,
};
use crate::ir::{BlockId, ImageFormat, Opcode, SyntaxNode, TextureInstInfo, TextureType, Value};
use crate::{Profile, RuntimeInfo, ShaderInfo, ShaderStage};

fn texture(ty: TextureType) -> TextureDescriptor {
    TextureDescriptor {
        ty,
        is_depth: false,
        is_multisample: false,
        is_integer: false,
        handle: HandleSource::default(),
        count: 1,
        size_shift: 3,
        is_descriptor_array: false,
        handle_is_pointer_word: false,
        handle_program: None,
    }
}

fn image(ty: TextureType, format: ImageFormat, is_integer: bool) -> ImageDescriptor {
    ImageDescriptor {
        ty,
        format,
        is_written: true,
        is_read: true,
        is_integer,
        cbuf_index: 0,
        cbuf_offset: 0,
        count: 1,
        size_shift: 3,
        is_byte_addressed: false,
        handle_program: None,
    }
}

fn image_buffer(format: ImageFormat, is_integer: bool, count: u32) -> ImageBufferDescriptor {
    ImageBufferDescriptor {
        format,
        is_written: true,
        is_read: true,
        is_integer,
        cbuf_index: 0,
        cbuf_offset: 0x20,
        count,
        size_shift: 3,
        handle_program: None,
    }
}

fn texture_buffer(count: u32) -> TextureBufferDescriptor {
    TextureBufferDescriptor {
        handle: HandleSource::default(),
        count,
        size_shift: 3,
        handle_program: None,
    }
}

fn flags(descriptor_index: u16, ty: TextureType) -> TextureInstInfo {
    TextureInstInfo {
        descriptor_index,
        ty,
        ..TextureInstInfo::default()
    }
}

fn finish(
    s: Synthetic,
    b: BlockId,
    stage: ShaderStage,
    info: ShaderInfo,
) -> crate::translate::TranslatedProgram {
    s.finish(vec![SyntaxNode::Block(b), SyntaxNode::Return], stage, info)
}

fn f32x2(s: &mut Synthetic, b: BlockId, x: f32, y: f32) -> Value {
    s.inst(
        b,
        Opcode::CompositeConstructF32x2,
        &[Value::F32(x), Value::F32(y)],
    )
}

fn u32x2(s: &mut Synthetic, b: BlockId, x: u32, y: u32) -> Value {
    s.inst(
        b,
        Opcode::CompositeConstructU32x2,
        &[Value::U32(x), Value::U32(y)],
    )
}

/// A dynamic u32: component 0 of the local invocation id (compute programs only).
fn dynamic_u32(s: &mut Synthetic, b: BlockId) -> Value {
    let local = s.inst(b, Opcode::LocalInvocationId, &[]);
    s.inst(b, Opcode::CompositeExtractU32x3, &[local, Value::U32(0)])
}

fn has_capability(module: &dr::Module, capability: Capability) -> bool {
    module
        .capabilities
        .iter()
        .any(|i| i.operands.first() == Some(&Operand::Capability(capability)))
}

fn count_decorations(module: &dr::Module, decoration: Decoration) -> usize {
    module
        .annotations
        .iter()
        .filter(|a| {
            a.class.opcode == Op::Decorate
                && a.operands.get(1) == Some(&Operand::Decoration(decoration))
        })
        .count()
}

fn compute_info() -> ShaderInfo {
    ShaderInfo {
        uses_local_invocation_id: true,
        ..ShaderInfo::default()
    }
}

#[test]
fn fragment_sampling_with_bias_clamp_offsets_and_gradients() {
    let mut s = Synthetic::new();
    let b = s.block();
    let coords = f32x2(&mut s, b, 0.5, 0.25);
    let bias_lc = f32x2(&mut s, b, 1.0, 2.0);
    let offset = u32x2(&mut s, b, 1, (-1i32) as u32);
    let color2d = flags(0, TextureType::Color2D);
    let both = TextureInstInfo {
        has_bias: true,
        has_lod_clamp: true,
        relaxed_precision: true,
        ..color2d
    };
    s.inst_flags(
        b,
        Opcode::ImageSampleImplicitLod,
        &[Value::Void, coords, bias_lc, offset],
        both.to_flags(),
    );
    let bias = TextureInstInfo {
        has_bias: true,
        ..color2d
    };
    s.inst_flags(
        b,
        Opcode::ImageSampleImplicitLod,
        &[Value::Void, coords, Value::F32(0.5), offset],
        bias.to_flags(),
    );
    let clamp = TextureInstInfo {
        has_lod_clamp: true,
        ..color2d
    };
    s.inst_flags(
        b,
        Opcode::ImageSampleImplicitLod,
        &[Value::Void, coords, Value::F32(1.0), Value::Void],
        clamp.to_flags(),
    );
    s.inst_flags(
        b,
        Opcode::ImageSampleExplicitLod,
        &[Value::Void, coords, Value::F32(2.0), offset],
        color2d.to_flags(),
    );
    let integer = flags(4, TextureType::Color2D).to_flags();
    s.inst_flags(
        b,
        Opcode::ImageSampleExplicitLod,
        &[Value::Void, coords, Value::F32(0.0), Value::Void],
        integer,
    );
    let depth = flags(1, TextureType::Color2D).to_flags();
    s.inst_flags(
        b,
        Opcode::ImageSampleDrefImplicitLod,
        &[
            Value::Void,
            coords,
            Value::F32(0.5),
            Value::Void,
            Value::Void,
        ],
        depth,
    );
    s.inst_flags(
        b,
        Opcode::ImageSampleDrefExplicitLod,
        &[
            Value::Void,
            coords,
            Value::F32(0.5),
            Value::F32(1.0),
            offset,
        ],
        depth,
    );
    s.inst_flags(
        b,
        Opcode::ImageQueryLod,
        &[Value::Void, coords],
        color2d.to_flags(),
    );
    let derivatives = s.inst(
        b,
        Opcode::CompositeConstructF32x4,
        &[
            Value::F32(0.1),
            Value::F32(0.2),
            Value::F32(0.3),
            Value::F32(0.4),
        ],
    );
    let grad2 = TextureInstInfo {
        num_derivatives: 2,
        has_lod_clamp: true,
        ..color2d
    };
    s.inst_flags(
        b,
        Opcode::ImageGradient,
        &[Value::Void, coords, derivatives, offset, Value::F32(1.0)],
        grad2.to_flags(),
    );
    let coords3 = s.inst(
        b,
        Opcode::CompositeConstructF32x3,
        &[Value::F32(0.1), Value::F32(0.2), Value::F32(0.3)],
    );
    let third = f32x2(&mut s, b, 0.5, 0.6);
    let grad3 = TextureInstInfo {
        num_derivatives: 3,
        ..flags(2, TextureType::Color3D)
    };
    s.inst_flags(
        b,
        Opcode::ImageGradient,
        &[Value::Void, coords3, derivatives, third, Value::Void],
        grad3.to_flags(),
    );
    let derivatives1 = f32x2(&mut s, b, 0.1, 0.2);
    let grad1 = TextureInstInfo {
        num_derivatives: 1,
        ..flags(3, TextureType::Color1D)
    };
    s.inst_flags(
        b,
        Opcode::ImageGradient,
        &[
            Value::Void,
            Value::F32(0.5),
            derivatives1,
            Value::U32(1),
            Value::Void,
        ],
        grad1.to_flags(),
    );
    let info = ShaderInfo {
        uses_sampled_1d: true,
        texture_descriptors: vec![
            texture(TextureType::Color2D),
            TextureDescriptor {
                is_depth: true,
                ..texture(TextureType::Color2D)
            },
            texture(TextureType::Color3D),
            texture(TextureType::Color1D),
            TextureDescriptor {
                is_integer: true,
                ..texture(TextureType::Color2D)
            },
        ],
        ..ShaderInfo::default()
    };
    let program = finish(s, b, ShaderStage::Fragment, info);
    let module = emit(
        "image_fragment_sampling",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::ImageSampleImplicitLod), 3);
    assert_eq!(count_ops(&module, Op::ImageSampleExplicitLod), 5);
    assert_eq!(count_ops(&module, Op::ImageSampleDrefImplicitLod), 1);
    assert_eq!(count_ops(&module, Op::ImageSampleDrefExplicitLod), 1);
    assert_eq!(count_ops(&module, Op::ImageQueryLod), 1);
    assert!(has_capability(&module, Capability::MinLod));
    assert_eq!(count_decorations(&module, Decoration::RelaxedPrecision), 1);
}

fn implicit_lod_program(stage: ShaderStage) -> crate::translate::TranslatedProgram {
    let mut s = Synthetic::new();
    let b = s.block();
    let coords = f32x2(&mut s, b, 0.5, 0.25);
    let offset = u32x2(&mut s, b, 1, 1);
    let both = TextureInstInfo {
        has_bias: true,
        has_lod_clamp: true,
        ..flags(0, TextureType::Color2D)
    };
    let bias_lc = f32x2(&mut s, b, 1.0, 2.0);
    s.inst_flags(
        b,
        Opcode::ImageSampleImplicitLod,
        &[Value::Void, coords, bias_lc, Value::Void],
        both.to_flags(),
    );
    let depth = flags(1, TextureType::Color2D).to_flags();
    s.inst_flags(
        b,
        Opcode::ImageSampleDrefImplicitLod,
        &[Value::Void, coords, Value::F32(0.5), Value::Void, offset],
        depth,
    );
    let info = ShaderInfo {
        texture_descriptors: vec![
            texture(TextureType::Color2D),
            TextureDescriptor {
                is_depth: true,
                ..texture(TextureType::Color2D)
            },
        ],
        ..ShaderInfo::default()
    };
    finish(s, b, stage, info)
}

#[test]
fn implicit_lods_sample_lod_zero_outside_fragment_programs() {
    for stage in [ShaderStage::Compute, ShaderStage::VertexB] {
        let program = implicit_lod_program(stage);
        let module = emit(
            "image_implicit_outside_fragment",
            &Profile::default(),
            &RuntimeInfo::default(),
            &program,
        );
        assert_eq!(count_ops(&module, Op::ImageSampleImplicitLod), 0);
        assert_eq!(count_ops(&module, Op::ImageSampleDrefImplicitLod), 0);
        assert_eq!(count_ops(&module, Op::ImageSampleExplicitLod), 1);
        assert_eq!(count_ops(&module, Op::ImageSampleDrefExplicitLod), 1);
        assert!(!has_capability(&module, Capability::MinLod));
    }
}

fn gather_program() -> crate::translate::TranslatedProgram {
    let mut s = Synthetic::new();
    let b = s.block();
    let coords = f32x2(&mut s, b, 0.5, 0.25);
    let dynamic = dynamic_u32(&mut s, b);
    let green = TextureInstInfo {
        gather_component: 1,
        ..flags(0, TextureType::Color2D)
    };
    let constant = u32x2(&mut s, b, 1, 2);
    s.inst_flags(
        b,
        Opcode::ImageGather,
        &[Value::Void, coords, constant, Value::Void],
        green.to_flags(),
    );
    let runtime = s.inst(
        b,
        Opcode::CompositeConstructU32x2,
        &[dynamic, Value::U32(1)],
    );
    s.inst_flags(
        b,
        Opcode::ImageGather,
        &[Value::Void, coords, runtime, Value::Void],
        green.to_flags(),
    );
    let ptp = |s: &mut Synthetic, first: Value| {
        s.inst(
            b,
            Opcode::CompositeConstructU32x4,
            &[first, Value::U32(1), Value::U32(2), Value::U32(3)],
        )
    };
    let ptp0 = ptp(&mut s, Value::U32(0));
    let ptp1 = ptp(&mut s, Value::U32((-2i32) as u32));
    s.inst_flags(
        b,
        Opcode::ImageGather,
        &[Value::Void, coords, ptp0, ptp1],
        green.to_flags(),
    );
    let dynamic_ptp = ptp(&mut s, dynamic);
    s.inst_flags(
        b,
        Opcode::ImageGather,
        &[Value::Void, coords, dynamic_ptp, ptp1],
        green.to_flags(),
    );
    let depth = flags(1, TextureType::Color2D).to_flags();
    s.inst_flags(
        b,
        Opcode::ImageGatherDref,
        &[Value::Void, coords, ptp0, ptp1, Value::F32(0.5)],
        depth,
    );
    s.inst_flags(
        b,
        Opcode::ImageGatherDref,
        &[Value::Void, coords, runtime, Value::Void, Value::F32(0.5)],
        depth,
    );
    s.inst_flags(
        b,
        Opcode::ImageGatherDref,
        &[Value::Void, coords, dynamic_ptp, ptp0, Value::F32(0.5)],
        depth,
    );
    let integer = flags(2, TextureType::Color2D).to_flags();
    s.inst_flags(
        b,
        Opcode::ImageGather,
        &[Value::Void, coords, dynamic_ptp, ptp0],
        integer,
    );
    let coords3 = s.inst(
        b,
        Opcode::CompositeConstructF32x3,
        &[Value::F32(0.1), Value::F32(0.2), Value::F32(1.0)],
    );
    s.inst_flags(
        b,
        Opcode::ImageGather,
        &[Value::Void, coords3, Value::Void, Value::Void],
        flags(3, TextureType::ColorArray2D).to_flags(),
    );
    s.inst_flags(
        b,
        Opcode::ImageGather,
        &[Value::Void, coords3, Value::Void, Value::Void],
        flags(4, TextureType::ColorCube).to_flags(),
    );
    let info = ShaderInfo {
        texture_descriptors: vec![
            texture(TextureType::Color2D),
            TextureDescriptor {
                is_depth: true,
                ..texture(TextureType::Color2D)
            },
            TextureDescriptor {
                is_integer: true,
                ..texture(TextureType::Color2D)
            },
            texture(TextureType::ColorArray2D),
            texture(TextureType::ColorCube),
        ],
        ..compute_info()
    };
    finish(s, b, ShaderStage::Compute, info)
}

#[test]
fn gathers_with_constant_runtime_and_per_tap_offsets() {
    let program = gather_program();
    let module = emit(
        "image_gathers",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::ImageGather), 3 + 4 + 4 + 2);
    assert_eq!(count_ops(&module, Op::ImageDrefGather), 2 + 4);
    assert_eq!(count_ops(&module, Op::ImageQuerySizeLod), 0);
    let nudged = Profile {
        need_gather_subpixel_offset: true,
        ..Profile::default()
    };
    let module = emit(
        "image_gathers_subpixel",
        &nudged,
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(
        count_ops(&module, Op::ImageQuerySizeLod),
        9,
        "every 2D and 2D array gather, never the cube"
    );
}

#[test]
fn sparse_residency_defines_the_sparse_pseudo_op() {
    let mut s = Synthetic::new();
    let b = s.block();
    let coords = f32x2(&mut s, b, 0.5, 0.25);
    let texel = u32x2(&mut s, b, 3, 4);
    let color2d = flags(0, TextureType::Color2D).to_flags();
    let sparse = |s: &mut Synthetic, op: Opcode, args: &[Value], flags: u32| {
        let value = s.inst_flags(b, op, args, flags);
        let resident = s.inst(b, Opcode::GetSparseFromOp, &[value]);
        s.inst(b, Opcode::LogicalNot, &[resident]);
    };
    sparse(
        &mut s,
        Opcode::ImageSampleImplicitLod,
        &[Value::Void, coords, Value::Void, Value::Void],
        color2d,
    );
    sparse(
        &mut s,
        Opcode::ImageSampleExplicitLod,
        &[Value::Void, coords, Value::F32(1.0), Value::Void],
        color2d,
    );
    let depth = flags(1, TextureType::Color2D).to_flags();
    sparse(
        &mut s,
        Opcode::ImageSampleDrefImplicitLod,
        &[
            Value::Void,
            coords,
            Value::F32(0.5),
            Value::Void,
            Value::Void,
        ],
        depth,
    );
    sparse(
        &mut s,
        Opcode::ImageSampleDrefExplicitLod,
        &[
            Value::Void,
            coords,
            Value::F32(0.5),
            Value::F32(0.0),
            Value::Void,
        ],
        depth,
    );
    sparse(
        &mut s,
        Opcode::ImageFetch,
        &[Value::Void, texel, Value::Void, Value::U32(0), Value::Void],
        color2d,
    );
    let ptp = s.inst(
        b,
        Opcode::CompositeConstructU32x4,
        &[Value::U32(0), Value::U32(1), Value::U32(2), Value::U32(3)],
    );
    sparse(
        &mut s,
        Opcode::ImageGather,
        &[Value::Void, coords, ptp, ptp],
        color2d,
    );
    sparse(
        &mut s,
        Opcode::ImageGatherDref,
        &[
            Value::Void,
            coords,
            Value::Void,
            Value::Void,
            Value::F32(0.5),
        ],
        depth,
    );
    let derivatives = s.inst(
        b,
        Opcode::CompositeConstructF32x4,
        &[
            Value::F32(0.1),
            Value::F32(0.2),
            Value::F32(0.3),
            Value::F32(0.4),
        ],
    );
    let grad = TextureInstInfo {
        num_derivatives: 2,
        ..flags(0, TextureType::Color2D)
    };
    sparse(
        &mut s,
        Opcode::ImageGradient,
        &[Value::Void, coords, derivatives, Value::Void, Value::Void],
        grad.to_flags(),
    );
    let storage = TextureInstInfo {
        image_format: ImageFormat::R32Uint,
        ..flags(0, TextureType::Color2D)
    };
    sparse(
        &mut s,
        Opcode::ImageRead,
        &[Value::Void, texel],
        storage.to_flags(),
    );
    let info = ShaderInfo {
        uses_sparse_residency: true,
        texture_descriptors: vec![
            texture(TextureType::Color2D),
            TextureDescriptor {
                is_depth: true,
                ..texture(TextureType::Color2D)
            },
        ],
        image_descriptors: vec![image(TextureType::Color2D, ImageFormat::R32Uint, true)],
        ..ShaderInfo::default()
    };
    let program = finish(s, b, ShaderStage::Fragment, info);
    let module = emit(
        "image_sparse",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::ImageSparseTexelsResident), 9);
    for op in [
        Op::ImageSparseSampleImplicitLod,
        Op::ImageSparseSampleDrefImplicitLod,
        Op::ImageSparseSampleDrefExplicitLod,
        Op::ImageSparseFetch,
        Op::ImageSparseGather,
        Op::ImageSparseDrefGather,
        Op::ImageSparseRead,
    ] {
        assert_eq!(count_ops(&module, op), 1, "{op:?}");
    }
    assert_eq!(count_ops(&module, Op::ImageSparseSampleExplicitLod), 2);
    assert_eq!(count_ops(&module, Op::ImageSampleImplicitLod), 0);
}

#[test]
fn fetches_and_dimension_queries_of_every_texture_type() {
    let mut s = Synthetic::new();
    let b = s.block();
    let dynamic = dynamic_u32(&mut s, b);
    let coords2 = s.inst(
        b,
        Opcode::CompositeConstructU32x2,
        &[dynamic, Value::U32(1)],
    );
    let coords3 = s.inst(
        b,
        Opcode::CompositeConstructU32x3,
        &[dynamic, Value::U32(1), Value::U32(2)],
    );
    let offset2 = s.inst(b, Opcode::CompositeConstructU32x2, &[dynamic, dynamic]);
    let offset3 = s.inst(
        b,
        Opcode::CompositeConstructU32x3,
        &[dynamic, dynamic, dynamic],
    );
    let types = [
        TextureType::Color1D,
        TextureType::ColorArray1D,
        TextureType::Color2D,
        TextureType::ColorArray2D,
        TextureType::Color3D,
        TextureType::ColorCube,
        TextureType::ColorArrayCube,
        TextureType::Color2DRect,
    ];
    let mut textures: Vec<TextureDescriptor> = types.iter().map(|&ty| texture(ty)).collect();
    textures.push(TextureDescriptor {
        is_multisample: true,
        ..texture(TextureType::Color2D)
    });
    textures.push(TextureDescriptor {
        is_integer: true,
        ..texture(TextureType::Color2D)
    });
    let fetch = |s: &mut Synthetic, index: u16, ty, coords, offset, lod, ms| {
        s.inst_flags(
            b,
            Opcode::ImageFetch,
            &[Value::Void, coords, offset, lod, ms],
            flags(index, ty).to_flags(),
        );
    };
    fetch(
        &mut s,
        0,
        TextureType::Color1D,
        dynamic,
        dynamic,
        Value::U32(1),
        Value::Void,
    );
    fetch(
        &mut s,
        1,
        TextureType::ColorArray1D,
        coords2,
        dynamic,
        Value::U32(0),
        Value::Void,
    );
    fetch(
        &mut s,
        2,
        TextureType::Color2D,
        coords2,
        offset2,
        dynamic,
        Value::Void,
    );
    fetch(
        &mut s,
        3,
        TextureType::ColorArray2D,
        coords3,
        offset2,
        Value::U32(0),
        Value::Void,
    );
    fetch(
        &mut s,
        4,
        TextureType::Color3D,
        coords3,
        offset3,
        Value::U32(0),
        Value::Void,
    );
    fetch(
        &mut s,
        7,
        TextureType::Color2DRect,
        coords2,
        Value::Void,
        Value::U32(0),
        Value::Void,
    );
    fetch(
        &mut s,
        8,
        TextureType::Color2D,
        coords2,
        Value::Void,
        Value::U32(0),
        Value::U32(3),
    );
    fetch(
        &mut s,
        9,
        TextureType::Color2D,
        coords2,
        Value::Void,
        Value::U32(0),
        Value::Void,
    );
    fetch(
        &mut s,
        0,
        TextureType::Buffer,
        dynamic,
        dynamic,
        Value::U32(0),
        Value::Void,
    );
    for (index, &ty) in types.iter().enumerate() {
        let skip = Value::U1(index % 2 == 0);
        s.inst_flags(
            b,
            Opcode::ImageQueryDimensions,
            &[Value::Void, dynamic, skip],
            flags(index as u16, ty).to_flags(),
        );
    }
    for skip in [false, true] {
        s.inst_flags(
            b,
            Opcode::ImageQueryDimensions,
            &[Value::Void, Value::U32(0), Value::U1(skip)],
            flags(8, TextureType::Color2D).to_flags(),
        );
        s.inst_flags(
            b,
            Opcode::ImageQueryDimensions,
            &[Value::Void, Value::U32(0), Value::U1(skip)],
            flags(0, TextureType::Buffer).to_flags(),
        );
    }
    let info = ShaderInfo {
        uses_sampled_1d: true,
        texture_descriptors: textures,
        texture_buffer_descriptors: vec![texture_buffer(1)],
        ..compute_info()
    };
    let program = finish(s, b, ShaderStage::Compute, info);
    let module = emit(
        "image_fetch_query",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::ImageFetch), 9);
    assert_eq!(count_ops(&module, Op::ImageQuerySizeLod), 8);
    assert_eq!(
        count_ops(&module, Op::ImageQuerySize),
        4,
        "multisample and buffer queries"
    );
    assert_eq!(count_ops(&module, Op::ImageQueryLevels), 4 + 1);
}

#[test]
fn storage_image_and_texel_buffer_reads_and_writes() {
    let mut s = Synthetic::new();
    let b = s.block();
    let dynamic = dynamic_u32(&mut s, b);
    let coords = s.inst(
        b,
        Opcode::CompositeConstructU32x2,
        &[dynamic, Value::U32(1)],
    );
    let coords3 = s.inst(
        b,
        Opcode::CompositeConstructU32x3,
        &[dynamic, Value::U32(1), Value::U32(2)],
    );
    let color = s.inst(
        b,
        Opcode::CompositeConstructU32x4,
        &[dynamic, Value::U32(1), Value::U32(2), Value::U32(3)],
    );
    let info_of = |index, ty, format| {
        TextureInstInfo {
            image_format: format,
            ..flags(index, ty)
        }
        .to_flags()
    };
    let integer = info_of(0, TextureType::Color2D, ImageFormat::R32Uint);
    let typeless = info_of(1, TextureType::Color2D, ImageFormat::Typeless);
    let array = info_of(2, TextureType::ColorArray2D, ImageFormat::R32G32B32A32Uint);
    let buffer = info_of(0, TextureType::Buffer, ImageFormat::R32Uint);
    let float_buffer = info_of(1, TextureType::Buffer, ImageFormat::Typeless);
    s.inst_flags(b, Opcode::ImageRead, &[Value::Void, coords], integer);
    s.inst_flags(
        b,
        Opcode::ImageWrite,
        &[Value::Void, coords, color],
        integer,
    );
    s.inst_flags(b, Opcode::ImageRead, &[Value::Void, coords], typeless);
    s.inst_flags(
        b,
        Opcode::ImageWrite,
        &[Value::Void, coords, color],
        typeless,
    );
    s.inst_flags(b, Opcode::ImageRead, &[Value::Void, coords3], array);
    s.inst_flags(b, Opcode::ImageWrite, &[Value::Void, coords3, color], array);
    s.inst_flags(b, Opcode::ImageRead, &[Value::Void, dynamic], buffer);
    s.inst_flags(
        b,
        Opcode::ImageWrite,
        &[Value::Void, dynamic, color],
        buffer,
    );
    s.inst_flags(
        b,
        Opcode::ImageWrite,
        &[Value::Void, dynamic, color],
        float_buffer,
    );
    let info = ShaderInfo {
        uses_typeless_image_reads: true,
        uses_typeless_image_writes: true,
        uses_image_buffers: true,
        image_descriptors: vec![
            image(TextureType::Color2D, ImageFormat::R32Uint, true),
            image(TextureType::Color2D, ImageFormat::Typeless, false),
            image(
                TextureType::ColorArray2D,
                ImageFormat::R32G32B32A32Uint,
                true,
            ),
        ],
        image_buffer_descriptors: vec![
            image_buffer(ImageFormat::R32Uint, true, 1),
            image_buffer(ImageFormat::Typeless, false, 1),
        ],
        ..compute_info()
    };
    let program = finish(s, b, ShaderStage::Compute, info);
    let module = emit(
        "image_storage_no_typeless_loads",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(
        count_ops(&module, Op::ImageRead),
        3,
        "the typeless read is a null constant"
    );
    assert_eq!(count_ops(&module, Op::ImageWrite), 5);
    let typeless_loads = Profile {
        support_typeless_image_loads: true,
        ..Profile::default()
    };
    let module = emit(
        "image_storage_typeless_loads",
        &typeless_loads,
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::ImageRead), 4);
    assert!(has_capability(
        &module,
        Capability::StorageImageReadWithoutFormat
    ));
}

fn descriptor_array_program() -> crate::translate::TranslatedProgram {
    let mut s = Synthetic::new();
    let b = s.block();
    let dynamic = dynamic_u32(&mut s, b);
    let coords = f32x2(&mut s, b, 0.5, 0.25);
    let texel = s.inst(
        b,
        Opcode::CompositeConstructU32x2,
        &[dynamic, Value::U32(1)],
    );
    let color = s.inst(
        b,
        Opcode::CompositeConstructU32x4,
        &[dynamic, Value::U32(1), Value::U32(2), Value::U32(3)],
    );
    let color2d = flags(0, TextureType::Color2D).to_flags();
    s.inst_flags(
        b,
        Opcode::ImageSampleExplicitLod,
        &[dynamic, coords, Value::F32(0.0), Value::Void],
        color2d,
    );
    s.inst_flags(
        b,
        Opcode::ImageSampleExplicitLod,
        &[Value::U32(3), coords, Value::F32(0.0), Value::Void],
        color2d,
    );
    s.inst_flags(
        b,
        Opcode::ImageFetch,
        &[dynamic, texel, Value::Void, Value::U32(0), Value::Void],
        color2d,
    );
    s.inst_flags(
        b,
        Opcode::ImageFetch,
        &[dynamic, dynamic, Value::Void, Value::U32(0), Value::Void],
        flags(0, TextureType::Buffer).to_flags(),
    );
    let storage = TextureInstInfo {
        image_format: ImageFormat::R32Uint,
        ..flags(0, TextureType::Color2D)
    }
    .to_flags();
    s.inst_flags(b, Opcode::ImageRead, &[dynamic, texel], storage);
    s.inst_flags(b, Opcode::ImageWrite, &[dynamic, texel, color], storage);
    let storage_buffer = TextureInstInfo {
        image_format: ImageFormat::R32Uint,
        ..flags(0, TextureType::Buffer)
    }
    .to_flags();
    s.inst_flags(
        b,
        Opcode::ImageWrite,
        &[dynamic, dynamic, color],
        storage_buffer,
    );
    s.inst_flags(
        b,
        Opcode::ImageAtomicIAdd32,
        &[dynamic, texel, Value::U32(1)],
        storage,
    );
    let info = ShaderInfo {
        uses_image_buffers: true,
        uses_atomic_image_u32: true,
        texture_descriptors: vec![TextureDescriptor {
            count: 4,
            is_descriptor_array: true,
            ..texture(TextureType::Color2D)
        }],
        texture_buffer_descriptors: vec![texture_buffer(2)],
        image_descriptors: vec![ImageDescriptor {
            count: 2,
            ..image(TextureType::Color2D, ImageFormat::R32Uint, true)
        }],
        image_buffer_descriptors: vec![image_buffer(ImageFormat::R32Uint, true, 3)],
        ..compute_info()
    };
    finish(s, b, ShaderStage::Compute, info)
}

#[test]
fn descriptor_arrays_index_their_elements_and_mark_non_uniform_indices() {
    let program = descriptor_array_program();
    let module = emit(
        "image_arrays_uniform",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_decorations(&module, Decoration::NonUniform), 0);
    assert_eq!(count_ops(&module, Op::AccessChain), 8);
    let non_uniform = Profile {
        support_sampled_image_array_nonuniform_indexing: true,
        support_storage_image_array_nonuniform_indexing: true,
        support_uniform_texel_buffer_array_nonuniform_indexing: true,
        support_storage_texel_buffer_array_nonuniform_indexing: true,
        ..Profile::default()
    };
    for (name, supported_spirv) in [
        ("image_arrays_non_uniform_1_3", 0x0001_0300),
        ("image_arrays_non_uniform_1_5", 0x0001_0500),
    ] {
        let profile = Profile {
            supported_spirv,
            ..non_uniform.clone()
        };
        let module = emit(name, &profile, &RuntimeInfo::default(), &program);
        assert!(count_decorations(&module, Decoration::NonUniform) > 0);
        for capability in [
            Capability::ShaderNonUniform,
            Capability::SampledImageArrayNonUniformIndexing,
            Capability::StorageImageArrayNonUniformIndexing,
            Capability::UniformTexelBufferArrayNonUniformIndexing,
            Capability::StorageTexelBufferArrayNonUniformIndexing,
        ] {
            assert!(has_capability(&module, capability), "{capability:?}");
        }
        assert_eq!(module.extensions.is_empty(), supported_spirv >= 0x0001_0500);
    }
}

/// A fragment program sampling descriptor arrays with implicit lods at one index: the lane
/// id, or arithmetic on constants.
fn table_sample_program(varying_index: bool) -> crate::translate::TranslatedProgram {
    let mut s = Synthetic::new();
    let b = s.block();
    let index = if varying_index {
        s.inst(b, Opcode::LaneId, &[])
    } else {
        let two = s.inst(
            b,
            Opcode::ShiftLeftLogical32,
            &[Value::U32(1), Value::U32(1)],
        );
        s.inst(b, Opcode::IAdd32, &[two, Value::U32(1)])
    };
    let coords = f32x2(&mut s, b, 0.5, 0.25);
    let both = TextureInstInfo {
        has_bias: true,
        has_lod_clamp: true,
        ..flags(0, TextureType::Color2D)
    };
    let bias_lc = f32x2(&mut s, b, 1.0, 2.0);
    s.inst_flags(
        b,
        Opcode::ImageSampleImplicitLod,
        &[index, coords, bias_lc, Value::Void],
        both.to_flags(),
    );
    let layered = s.inst(
        b,
        Opcode::CompositeConstructF32x3,
        &[Value::F32(0.5), Value::F32(0.25), Value::F32(1.0)],
    );
    s.inst_flags(
        b,
        Opcode::ImageSampleDrefImplicitLod,
        &[index, layered, Value::F32(0.5), Value::Void, Value::Void],
        flags(1, TextureType::ColorArray2D).to_flags(),
    );
    let table = |descriptor: TextureDescriptor| TextureDescriptor {
        count: 64,
        is_descriptor_array: true,
        ..descriptor
    };
    let info = ShaderInfo {
        uses_subgroup_invocation_id: true,
        texture_descriptors: vec![
            table(texture(TextureType::Color2D)),
            table(TextureDescriptor {
                is_depth: true,
                ..texture(TextureType::ColorArray2D)
            }),
        ],
        ..ShaderInfo::default()
    };
    finish(s, b, ShaderStage::Fragment, info)
}

#[test]
fn varying_table_indices_sample_with_explicit_gradients() {
    let profile = Profile {
        support_sampled_image_array_nonuniform_indexing: true,
        ..Profile::default()
    };
    let implicit_samples = |module: &dr::Module| {
        count_ops(module, Op::ImageSampleImplicitLod)
            + count_ops(module, Op::ImageSampleDrefImplicitLod)
    };

    let varying = table_sample_program(true);
    let module = emit(
        "table_gradients",
        &profile,
        &RuntimeInfo::default(),
        &varying,
    );
    assert_eq!(implicit_samples(&module), 0);
    assert_eq!(count_ops(&module, Op::ImageSampleExplicitLod), 1);
    assert_eq!(count_ops(&module, Op::ImageSampleDrefExplicitLod), 1);
    assert_eq!(count_ops(&module, Op::DPdx), 2);
    assert_eq!(count_ops(&module, Op::DPdy), 2);
    assert_eq!(
        count_ops(&module, Op::VectorTimesScalar),
        2,
        "bias scales one pair"
    );
    assert!(has_capability(&module, Capability::MinLod));

    let uniform = table_sample_program(false);
    let module = emit("table_uniform", &profile, &RuntimeInfo::default(), &uniform);
    assert_eq!(
        implicit_samples(&module),
        2,
        "a constant index keeps implicit lods"
    );
    assert_eq!(count_ops(&module, Op::DPdx), 0);

    let quad_divergent = Profile {
        support_quad_divergent_implicit_lod: true,
        ..profile
    };
    let module = emit(
        "table_quad_divergent",
        &quad_divergent,
        &RuntimeInfo::default(),
        &varying,
    );
    assert_eq!(
        implicit_samples(&module),
        2,
        "the device handles divergent lods"
    );
    assert_eq!(count_ops(&module, Op::DPdx), 0);
}

#[test]
fn image_atomics_on_images_and_texel_buffers() {
    let mut s = Synthetic::new();
    let b = s.block();
    let dynamic = dynamic_u32(&mut s, b);
    let texel = s.inst(
        b,
        Opcode::CompositeConstructU32x2,
        &[dynamic, Value::U32(1)],
    );
    let image_flags = TextureInstInfo {
        image_format: ImageFormat::R32Uint,
        ..flags(0, TextureType::Color2D)
    }
    .to_flags();
    let buffer_flags = TextureInstInfo {
        image_format: ImageFormat::R32Uint,
        ..flags(0, TextureType::Buffer)
    }
    .to_flags();
    let ops = [
        (Opcode::ImageAtomicIAdd32, Op::AtomicIAdd),
        (Opcode::ImageAtomicSMin32, Op::AtomicSMin),
        (Opcode::ImageAtomicUMin32, Op::AtomicUMin),
        (Opcode::ImageAtomicSMax32, Op::AtomicSMax),
        (Opcode::ImageAtomicUMax32, Op::AtomicUMax),
        (Opcode::ImageAtomicAnd32, Op::AtomicAnd),
        (Opcode::ImageAtomicOr32, Op::AtomicOr),
        (Opcode::ImageAtomicXor32, Op::AtomicXor),
        (Opcode::ImageAtomicExchange32, Op::AtomicExchange),
    ];
    for (opcode, _) in ops {
        s.inst_flags(b, opcode, &[Value::Void, texel, dynamic], image_flags);
        s.inst_flags(
            b,
            opcode,
            &[Value::Void, dynamic, Value::U32(7)],
            buffer_flags,
        );
    }
    let info = ShaderInfo {
        uses_atomic_image_u32: true,
        uses_image_buffers: true,
        image_descriptors: vec![image(TextureType::Color2D, ImageFormat::R32Uint, true)],
        image_buffer_descriptors: vec![image_buffer(ImageFormat::R32Uint, true, 1)],
        ..compute_info()
    };
    let program = finish(s, b, ShaderStage::Compute, info);
    let module = emit(
        "image_atomics",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::ImageTexelPointer), 18);
    for (_, op) in ops {
        assert_eq!(count_ops(&module, op), 2, "{op:?}");
    }
}

fn emit_error(op: Opcode, info: ShaderInfo, flags: u32, args: &[Value]) -> EmitError {
    let mut s = Synthetic::new();
    let b = s.block();
    s.inst_flags(b, op, args, flags);
    let program = finish(s, b, ShaderStage::Compute, info);
    emit_spirv(
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
        &mut Bindings::default(),
    )
    .expect_err("the opcode cannot be emitted")
}

#[test]
fn wrapping_image_atomics_and_unlowered_handles_are_errors() {
    let info = ShaderInfo {
        uses_atomic_image_u32: true,
        image_descriptors: vec![image(TextureType::Color2D, ImageFormat::R32Uint, true)],
        ..ShaderInfo::default()
    };
    let storage = TextureInstInfo {
        image_format: ImageFormat::R32Uint,
        ..flags(0, TextureType::Color2D)
    }
    .to_flags();
    for op in [Opcode::ImageAtomicInc32, Opcode::ImageAtomicDec32] {
        let error = emit_error(
            op,
            info.clone(),
            storage,
            &[Value::Void, Value::U32(0), Value::U32(1)],
        );
        assert!(matches!(error, EmitError::Unsupported(_)), "{error}");
    }
    let unlowered = [
        (
            Opcode::BoundImageQueryDimensions,
            vec![Value::U32(0), Value::U32(0), Value::U1(false)],
        ),
        (
            Opcode::BindlessImageRead,
            vec![Value::U32(0), Value::U32(0)],
        ),
        (
            Opcode::BoundImageAtomicIAdd32,
            vec![Value::U32(0), Value::U32(0), Value::U32(1)],
        ),
        (
            Opcode::BindlessImageAtomicExchange32,
            vec![Value::U32(0), Value::U32(0), Value::U32(1)],
        ),
    ];
    for (op, args) in unlowered {
        let error = emit_error(op, ShaderInfo::default(), 0, &args);
        assert!(
            matches!(&error, EmitError::InvalidIr(m) if m.contains("texture pass")),
            "{error}"
        );
    }
}

fn rescaling_program() -> crate::translate::TranslatedProgram {
    let mut s = Synthetic::new();
    let b = s.block();
    let index = dynamic_u32(&mut s, b);
    for op in [Opcode::IsTextureScaled, Opcode::IsImageScaled] {
        s.inst(b, op, &[Value::U32(37)]);
        s.inst(b, op, &[index]);
    }
    let info = ShaderInfo {
        uses_rescaling_uniform: true,
        ..compute_info()
    };
    finish(s, b, ShaderStage::Compute, info)
}

#[test]
fn rescaling_queries_test_the_push_constant_bit_sets() {
    let program = rescaling_program();
    let profile = Profile::default();
    for (name, texture_scaling_index, image_scaling_index) in
        [("image_rescaling", 0, 0), ("image_rescaling_based", 3, 1)]
    {
        let mut bindings = Bindings {
            texture_scaling_index,
            image_scaling_index,
            ..Bindings::default()
        };
        let code =
            emit_spirv(&profile, &RuntimeInfo::default(), &program, &mut bindings).expect("emit");
        validate(name, &profile, &code);
        let module = dr::load_words(&code).expect("parse");
        assert_eq!(count_ops(&module, Op::INotEqual), 4);
        assert_eq!(count_ops(&module, Op::AccessChain), 4);
        let adds = if texture_scaling_index == 0 { 0 } else { 2 };
        assert_eq!(count_ops(&module, Op::IAdd), adds);
    }
}

/// The OpenGL layout declares a non-opaque uniform constant, which Vulkan environments reject,
/// so this module is checked against the universal SPIR-V 1.3 rules.
#[test]
fn rescaling_queries_read_the_uniform_without_unified_bindings() {
    let program = rescaling_program();
    let profile = Profile {
        unified_descriptor_binding: false,
        ..Profile::default()
    };
    let code = emit_spirv(
        &profile,
        &RuntimeInfo::default(),
        &program,
        &mut Bindings::default(),
    )
    .expect("emit");
    validate_universal("image_rescaling_uniform", &code);
    let module = dr::load_words(&code).expect("parse");
    assert_eq!(count_ops(&module, Op::ShiftRightLogical), 4);
    assert_eq!(count_ops(&module, Op::INotEqual), 4);
    assert_eq!(count_ops(&module, Op::AccessChain), 0);
}

fn validate_universal(name: &str, code: &[u32]) {
    let path = std::env::temp_dir().join(format!("lichii-spirv-{name}-{}.spv", std::process::id()));
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&path, bytes).expect("write module");
    let output = std::process::Command::new("spirv-val")
        .arg("--target-env")
        .arg("spv1.3")
        .arg(&path)
        .output();
    let _ = std::fs::remove_file(&path);
    match output {
        Ok(output) => assert!(
            output.status.success(),
            "{name} failed validation:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("skipping SPIR-V validation of {name}: spirv-val not found");
        }
        Err(e) => panic!("running spirv-val: {e}"),
    }
}
