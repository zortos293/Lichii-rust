//! Maxwell register enums to Vulkan (port of `renderer_vulkan/maxwell_to_vk.cpp`).
//!
//! Every function maps each named guest value and returns `None` for values the hardware
//! documentation does not name, so callers pick their own fallback.

use ash::vk;
use lichii_gpu::engines::maxwell_3d::regs::{
    BlendEquation, BlendFactor, ComparisonOp, CullFace, FrontFace, IndexFormat, LogicOpOp,
    MsaaMode, PolygonMode, PrimitiveTopology, StencilOpOp, VertexAttributeSize,
    VertexAttributeType,
};

/// The host topology a guest topology draws with. Line loops draw as strips (the draw path
/// appends the closing index), quads and quad strips as triangle lists (the draw path rewrites
/// the indices) and polygons as fans.
pub fn primitive_topology(topology: PrimitiveTopology) -> Option<vk::PrimitiveTopology> {
    Some(match topology {
        PrimitiveTopology::POINTS => vk::PrimitiveTopology::POINT_LIST,
        PrimitiveTopology::LINES => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveTopology::LINE_LOOP | PrimitiveTopology::LINE_STRIP => {
            vk::PrimitiveTopology::LINE_STRIP
        }
        PrimitiveTopology::TRIANGLES | PrimitiveTopology::QUADS | PrimitiveTopology::QUAD_STRIP => {
            vk::PrimitiveTopology::TRIANGLE_LIST
        }
        PrimitiveTopology::TRIANGLE_STRIP => vk::PrimitiveTopology::TRIANGLE_STRIP,
        PrimitiveTopology::TRIANGLE_FAN | PrimitiveTopology::POLYGON => {
            vk::PrimitiveTopology::TRIANGLE_FAN
        }
        PrimitiveTopology::LINES_ADJACENCY => vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY,
        PrimitiveTopology::LINE_STRIP_ADJACENCY => vk::PrimitiveTopology::LINE_STRIP_WITH_ADJACENCY,
        PrimitiveTopology::TRIANGLES_ADJACENCY => {
            vk::PrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY
        }
        PrimitiveTopology::TRIANGLE_STRIP_ADJACENCY => {
            vk::PrimitiveTopology::TRIANGLE_STRIP_WITH_ADJACENCY
        }
        PrimitiveTopology::PATCHES => vk::PrimitiveTopology::PATCH_LIST,
        _ => return None,
    })
}

/// The topology a pipeline is built with when the topology is dynamic state: one per class
/// that dynamic topology can switch within (points, lines, triangles, their adjacency forms,
/// patches), so draws that only change the topology inside a class share a pipeline.
pub fn topology_class(topology: vk::PrimitiveTopology) -> vk::PrimitiveTopology {
    match topology {
        vk::PrimitiveTopology::LINE_STRIP => vk::PrimitiveTopology::LINE_LIST,
        vk::PrimitiveTopology::TRIANGLE_STRIP | vk::PrimitiveTopology::TRIANGLE_FAN => {
            vk::PrimitiveTopology::TRIANGLE_LIST
        }
        vk::PrimitiveTopology::LINE_STRIP_WITH_ADJACENCY => {
            vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY
        }
        vk::PrimitiveTopology::TRIANGLE_STRIP_WITH_ADJACENCY => {
            vk::PrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY
        }
        other => other,
    }
}

pub fn polygon_mode(mode: PolygonMode) -> Option<vk::PolygonMode> {
    Some(match mode {
        PolygonMode::POINT => vk::PolygonMode::POINT,
        PolygonMode::LINE => vk::PolygonMode::LINE,
        PolygonMode::FILL => vk::PolygonMode::FILL,
        _ => return None,
    })
}

pub fn cull_mode(face: CullFace) -> Option<vk::CullModeFlags> {
    Some(match face {
        CullFace::FRONT => vk::CullModeFlags::FRONT,
        CullFace::BACK => vk::CullModeFlags::BACK,
        CullFace::FRONT_AND_BACK => vk::CullModeFlags::FRONT_AND_BACK,
        _ => return None,
    })
}

pub fn front_face(face: FrontFace) -> Option<vk::FrontFace> {
    Some(match face {
        FrontFace::CLOCK_WISE => vk::FrontFace::CLOCKWISE,
        FrontFace::COUNTER_CLOCK_WISE => vk::FrontFace::COUNTER_CLOCKWISE,
        _ => return None,
    })
}

pub fn comparison_op(op: ComparisonOp) -> Option<vk::CompareOp> {
    Some(match op {
        ComparisonOp::NEVER_D3D | ComparisonOp::NEVER_GL => vk::CompareOp::NEVER,
        ComparisonOp::LESS_D3D | ComparisonOp::LESS_GL => vk::CompareOp::LESS,
        ComparisonOp::EQUAL_D3D | ComparisonOp::EQUAL_GL => vk::CompareOp::EQUAL,
        ComparisonOp::LESS_EQUAL_D3D | ComparisonOp::LESS_EQUAL_GL => vk::CompareOp::LESS_OR_EQUAL,
        ComparisonOp::GREATER_D3D | ComparisonOp::GREATER_GL => vk::CompareOp::GREATER,
        ComparisonOp::NOT_EQUAL_D3D | ComparisonOp::NOT_EQUAL_GL => vk::CompareOp::NOT_EQUAL,
        ComparisonOp::GREATER_EQUAL_D3D | ComparisonOp::GREATER_EQUAL_GL => {
            vk::CompareOp::GREATER_OR_EQUAL
        }
        ComparisonOp::ALWAYS_D3D | ComparisonOp::ALWAYS_GL => vk::CompareOp::ALWAYS,
        _ => return None,
    })
}

pub fn stencil_op(op: StencilOpOp) -> Option<vk::StencilOp> {
    Some(match op {
        StencilOpOp::KEEP_D3D | StencilOpOp::KEEP_GL => vk::StencilOp::KEEP,
        StencilOpOp::ZERO_D3D | StencilOpOp::ZERO_GL => vk::StencilOp::ZERO,
        StencilOpOp::REPLACE_D3D | StencilOpOp::REPLACE_GL => vk::StencilOp::REPLACE,
        StencilOpOp::INCR_SATURATE_D3D | StencilOpOp::INCR_SATURATE_GL => {
            vk::StencilOp::INCREMENT_AND_CLAMP
        }
        StencilOpOp::DECR_SATURATE_D3D | StencilOpOp::DECR_SATURATE_GL => {
            vk::StencilOp::DECREMENT_AND_CLAMP
        }
        StencilOpOp::INVERT_D3D | StencilOpOp::INVERT_GL => vk::StencilOp::INVERT,
        StencilOpOp::INCR_D3D | StencilOpOp::INCR_GL => vk::StencilOp::INCREMENT_AND_WRAP,
        StencilOpOp::DECR_D3D | StencilOpOp::DECR_GL => vk::StencilOp::DECREMENT_AND_WRAP,
        _ => return None,
    })
}

pub fn blend_equation(equation: BlendEquation) -> Option<vk::BlendOp> {
    Some(match equation {
        BlendEquation::ADD_D3D | BlendEquation::ADD_GL => vk::BlendOp::ADD,
        BlendEquation::SUBTRACT_D3D | BlendEquation::SUBTRACT_GL => vk::BlendOp::SUBTRACT,
        BlendEquation::REVERSE_SUBTRACT_D3D | BlendEquation::REVERSE_SUBTRACT_GL => {
            vk::BlendOp::REVERSE_SUBTRACT
        }
        BlendEquation::MIN_D3D | BlendEquation::MIN_GL => vk::BlendOp::MIN,
        BlendEquation::MAX_D3D | BlendEquation::MAX_GL => vk::BlendOp::MAX,
        _ => return None,
    })
}

/// D3D's `BOTH_SOURCE_ALPHA` pair maps to the constant alpha factors, as the C++ renderer does.
pub fn blend_factor(factor: BlendFactor) -> Option<vk::BlendFactor> {
    use vk::BlendFactor as F;
    Some(match factor {
        BlendFactor::ZERO_D3D | BlendFactor::ZERO_GL => F::ZERO,
        BlendFactor::ONE_D3D | BlendFactor::ONE_GL => F::ONE,
        BlendFactor::SOURCE_COLOR_D3D | BlendFactor::SOURCE_COLOR_GL => F::SRC_COLOR,
        BlendFactor::ONE_MINUS_SOURCE_COLOR_D3D | BlendFactor::ONE_MINUS_SOURCE_COLOR_GL => {
            F::ONE_MINUS_SRC_COLOR
        }
        BlendFactor::SOURCE_ALPHA_D3D | BlendFactor::SOURCE_ALPHA_GL => F::SRC_ALPHA,
        BlendFactor::ONE_MINUS_SOURCE_ALPHA_D3D | BlendFactor::ONE_MINUS_SOURCE_ALPHA_GL => {
            F::ONE_MINUS_SRC_ALPHA
        }
        BlendFactor::DEST_ALPHA_D3D | BlendFactor::DEST_ALPHA_GL => F::DST_ALPHA,
        BlendFactor::ONE_MINUS_DEST_ALPHA_D3D | BlendFactor::ONE_MINUS_DEST_ALPHA_GL => {
            F::ONE_MINUS_DST_ALPHA
        }
        BlendFactor::DEST_COLOR_D3D | BlendFactor::DEST_COLOR_GL => F::DST_COLOR,
        BlendFactor::ONE_MINUS_DEST_COLOR_D3D | BlendFactor::ONE_MINUS_DEST_COLOR_GL => {
            F::ONE_MINUS_DST_COLOR
        }
        BlendFactor::SOURCE_ALPHA_SATURATE_D3D | BlendFactor::SOURCE_ALPHA_SATURATE_GL => {
            F::SRC_ALPHA_SATURATE
        }
        BlendFactor::SOURCE1_COLOR_D3D | BlendFactor::SOURCE1_COLOR_GL => F::SRC1_COLOR,
        BlendFactor::ONE_MINUS_SOURCE1_COLOR_D3D | BlendFactor::ONE_MINUS_SOURCE1_COLOR_GL => {
            F::ONE_MINUS_SRC1_COLOR
        }
        BlendFactor::SOURCE1_ALPHA_D3D | BlendFactor::SOURCE1_ALPHA_GL => F::SRC1_ALPHA,
        BlendFactor::ONE_MINUS_SOURCE1_ALPHA_D3D | BlendFactor::ONE_MINUS_SOURCE1_ALPHA_GL => {
            F::ONE_MINUS_SRC1_ALPHA
        }
        BlendFactor::BLEND_FACTOR_D3D | BlendFactor::CONSTANT_COLOR_GL => F::CONSTANT_COLOR,
        BlendFactor::ONE_MINUS_BLEND_FACTOR_D3D | BlendFactor::ONE_MINUS_CONSTANT_COLOR_GL => {
            F::ONE_MINUS_CONSTANT_COLOR
        }
        BlendFactor::BOTH_SOURCE_ALPHA_D3D | BlendFactor::CONSTANT_ALPHA_GL => F::CONSTANT_ALPHA,
        BlendFactor::ONE_MINUS_BOTH_SOURCE_ALPHA_D3D | BlendFactor::ONE_MINUS_CONSTANT_ALPHA_GL => {
            F::ONE_MINUS_CONSTANT_ALPHA
        }
        _ => return None,
    })
}

/// The single-source factor a dual-source factor degrades to on devices without
/// `dualSrcBlend`.
pub fn single_source(factor: vk::BlendFactor) -> vk::BlendFactor {
    use vk::BlendFactor as F;
    match factor {
        F::SRC1_COLOR => F::SRC_COLOR,
        F::ONE_MINUS_SRC1_COLOR => F::ONE_MINUS_SRC_COLOR,
        F::SRC1_ALPHA => F::SRC_ALPHA,
        F::ONE_MINUS_SRC1_ALPHA => F::ONE_MINUS_SRC_ALPHA,
        other => other,
    }
}

pub fn logic_op(op: LogicOpOp) -> Option<vk::LogicOp> {
    use vk::LogicOp as L;
    Some(match op {
        LogicOpOp::CLEAR => L::CLEAR,
        LogicOpOp::AND => L::AND,
        LogicOpOp::AND_REVERSE => L::AND_REVERSE,
        LogicOpOp::COPY => L::COPY,
        LogicOpOp::AND_INVERTED => L::AND_INVERTED,
        LogicOpOp::NO_OP => L::NO_OP,
        LogicOpOp::XOR => L::XOR,
        LogicOpOp::OR => L::OR,
        LogicOpOp::NOR => L::NOR,
        LogicOpOp::EQUIV => L::EQUIVALENT,
        LogicOpOp::INVERT => L::INVERT,
        LogicOpOp::OR_REVERSE => L::OR_REVERSE,
        LogicOpOp::COPY_INVERTED => L::COPY_INVERTED,
        LogicOpOp::OR_INVERTED => L::OR_INVERTED,
        LogicOpOp::NAND => L::NAND,
        LogicOpOp::SET => L::SET,
        _ => return None,
    })
}

/// The vertex buffer format of a guest attribute; `None` for combinations the hardware does
/// not define. Whether the device can fetch it is checked when the pipeline is built.
pub fn vertex_format(ty: VertexAttributeType, size: VertexAttributeSize) -> Option<vk::Format> {
    use VertexAttributeSize as S;
    use vk::Format as F;
    let pick = |r8, rg8, rgb8, rgba8, r16, rg16, rgb16, rgba16, a2bgr10| match size {
        S::SIZE_R8 | S::SIZE_A8 => Some(r8),
        S::SIZE_R8_G8 | S::SIZE_G8_R8 => Some(rg8),
        S::SIZE_R8_G8_B8 => Some(rgb8),
        S::SIZE_R8_G8_B8_A8 | S::SIZE_X8_B8_G8_R8 => Some(rgba8),
        S::SIZE_R16 => Some(r16),
        S::SIZE_R16_G16 => Some(rg16),
        S::SIZE_R16_G16_B16 => Some(rgb16),
        S::SIZE_R16_G16_B16_A16 => Some(rgba16),
        S::SIZE_A2_B10_G10_R10 => Some(a2bgr10),
        _ => None,
    };
    let wide = |r32, rg32, rgb32, rgba32| match size {
        S::SIZE_R32 => Some(r32),
        S::SIZE_R32_G32 => Some(rg32),
        S::SIZE_R32_G32_B32 => Some(rgb32),
        S::SIZE_R32_G32_B32_A32 => Some(rgba32),
        _ => None,
    };
    match ty {
        VertexAttributeType::U_NORM => pick(
            F::R8_UNORM,
            F::R8G8_UNORM,
            F::R8G8B8_UNORM,
            F::R8G8B8A8_UNORM,
            F::R16_UNORM,
            F::R16G16_UNORM,
            F::R16G16B16_UNORM,
            F::R16G16B16A16_UNORM,
            F::A2B10G10R10_UNORM_PACK32,
        ),
        VertexAttributeType::S_NORM => pick(
            F::R8_SNORM,
            F::R8G8_SNORM,
            F::R8G8B8_SNORM,
            F::R8G8B8A8_SNORM,
            F::R16_SNORM,
            F::R16G16_SNORM,
            F::R16G16B16_SNORM,
            F::R16G16B16A16_SNORM,
            F::A2B10G10R10_SNORM_PACK32,
        ),
        VertexAttributeType::U_SCALED => pick(
            F::R8_USCALED,
            F::R8G8_USCALED,
            F::R8G8B8_USCALED,
            F::R8G8B8A8_USCALED,
            F::R16_USCALED,
            F::R16G16_USCALED,
            F::R16G16B16_USCALED,
            F::R16G16B16A16_USCALED,
            F::A2B10G10R10_USCALED_PACK32,
        ),
        VertexAttributeType::S_SCALED => pick(
            F::R8_SSCALED,
            F::R8G8_SSCALED,
            F::R8G8B8_SSCALED,
            F::R8G8B8A8_SSCALED,
            F::R16_SSCALED,
            F::R16G16_SSCALED,
            F::R16G16B16_SSCALED,
            F::R16G16B16A16_SSCALED,
            F::A2B10G10R10_SSCALED_PACK32,
        ),
        VertexAttributeType::U_INT => pick(
            F::R8_UINT,
            F::R8G8_UINT,
            F::R8G8B8_UINT,
            F::R8G8B8A8_UINT,
            F::R16_UINT,
            F::R16G16_UINT,
            F::R16G16B16_UINT,
            F::R16G16B16A16_UINT,
            F::A2B10G10R10_UINT_PACK32,
        )
        .or_else(|| {
            wide(
                F::R32_UINT,
                F::R32G32_UINT,
                F::R32G32B32_UINT,
                F::R32G32B32A32_UINT,
            )
        }),
        VertexAttributeType::S_INT => pick(
            F::R8_SINT,
            F::R8G8_SINT,
            F::R8G8B8_SINT,
            F::R8G8B8A8_SINT,
            F::R16_SINT,
            F::R16G16_SINT,
            F::R16G16B16_SINT,
            F::R16G16B16A16_SINT,
            F::A2B10G10R10_SINT_PACK32,
        )
        .or_else(|| {
            wide(
                F::R32_SINT,
                F::R32G32_SINT,
                F::R32G32B32_SINT,
                F::R32G32B32A32_SINT,
            )
        }),
        VertexAttributeType::FLOAT => match size {
            S::SIZE_R16 => Some(F::R16_SFLOAT),
            S::SIZE_R16_G16 => Some(F::R16G16_SFLOAT),
            S::SIZE_R16_G16_B16 => Some(F::R16G16B16_SFLOAT),
            S::SIZE_R16_G16_B16_A16 => Some(F::R16G16B16A16_SFLOAT),
            S::SIZE_B10_G11_R11 => Some(F::B10G11R11_UFLOAT_PACK32),
            _ => wide(
                F::R32_SFLOAT,
                F::R32G32_SFLOAT,
                F::R32G32B32_SFLOAT,
                F::R32G32B32A32_SFLOAT,
            ),
        },
        _ => None,
    }
}

/// 8-bit indices need `VK_EXT_index_type_uint8` (`DeviceFeatures::index_type_uint8`); the
/// draw path widens them otherwise.
pub fn index_type(format: IndexFormat) -> Option<vk::IndexType> {
    Some(match format {
        IndexFormat::UNSIGNED_BYTE => vk::IndexType::UINT8_EXT,
        IndexFormat::UNSIGNED_SHORT => vk::IndexType::UINT16,
        IndexFormat::UNSIGNED_INT => vk::IndexType::UINT32,
        _ => return None,
    })
}

pub fn sample_count(mode: MsaaMode) -> Option<vk::SampleCountFlags> {
    Some(match mode {
        MsaaMode::MSAA1X1 => vk::SampleCountFlags::TYPE_1,
        MsaaMode::MSAA2X1 | MsaaMode::MSAA2X1_D3D => vk::SampleCountFlags::TYPE_2,
        MsaaMode::MSAA2X2 | MsaaMode::MSAA2X2_VC4 | MsaaMode::MSAA2X2_VC12 => {
            vk::SampleCountFlags::TYPE_4
        }
        MsaaMode::MSAA4X2
        | MsaaMode::MSAA4X2_D3D
        | MsaaMode::MSAA4X2_VC8
        | MsaaMode::MSAA4X2_VC24 => vk::SampleCountFlags::TYPE_8,
        MsaaMode::MSAA4X4 => vk::SampleCountFlags::TYPE_16,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every raw value up to `limit` the guest enum names.
    fn named<T: Copy>(
        limit: u32,
        make: fn(u32) -> T,
        name: fn(T) -> Option<&'static str>,
    ) -> Vec<T> {
        (0..=limit)
            .map(make)
            .filter(|&v| name(v).is_some())
            .collect()
    }

    macro_rules! covers {
        ($ty:ident, $translate:expr, $count:expr) => {{
            let values = named(0x1_0000, $ty, $ty::name);
            assert_eq!(values.len(), $count, stringify!($ty));
            for value in values {
                assert!($translate(value).is_some(), "{value:?} has no translation");
            }
            for raw in [0x1_0001, u32::MAX] {
                assert!($translate($ty(raw)).is_none(), "{raw:#x}");
            }
        }};
    }

    #[test]
    fn every_named_guest_value_translates() {
        covers!(PrimitiveTopology, primitive_topology, 15);
        covers!(PolygonMode, polygon_mode, 3);
        covers!(CullFace, cull_mode, 3);
        covers!(FrontFace, front_face, 2);
        covers!(ComparisonOp, comparison_op, 16);
        covers!(StencilOpOp, stencil_op, 16);
        covers!(BlendEquation, blend_equation, 10);
        covers!(BlendFactor, blend_factor, 38);
        covers!(LogicOpOp, logic_op, 16);
        covers!(IndexFormat, index_type, 3);
        covers!(MsaaMode, sample_count, 11);
    }

    #[test]
    fn d3d_and_gl_encodings_agree() {
        assert_eq!(
            comparison_op(ComparisonOp::GREATER_EQUAL_D3D),
            comparison_op(ComparisonOp::GREATER_EQUAL_GL)
        );
        assert_eq!(stencil_op(StencilOpOp::ZERO_GL), Some(vk::StencilOp::ZERO));
        assert_eq!(
            stencil_op(StencilOpOp::INVERT_GL),
            stencil_op(StencilOpOp::INVERT_D3D)
        );
        assert_eq!(
            blend_factor(BlendFactor::ONE_MINUS_SOURCE1_ALPHA_GL),
            Some(vk::BlendFactor::ONE_MINUS_SRC1_ALPHA)
        );
        assert_eq!(
            blend_equation(BlendEquation::REVERSE_SUBTRACT_GL),
            Some(vk::BlendOp::REVERSE_SUBTRACT)
        );
        assert_eq!(logic_op(LogicOpOp::EQUIV), Some(vk::LogicOp::EQUIVALENT));
        assert_eq!(
            single_source(vk::BlendFactor::SRC1_ALPHA),
            vk::BlendFactor::SRC_ALPHA
        );
    }

    #[test]
    fn vertex_formats_cover_the_hardware_table() {
        let sizes = named(0x3f, VertexAttributeSize, VertexAttributeSize::name);
        let types = named(0x7, VertexAttributeType, VertexAttributeType::name);
        let mut defined = 0;
        for &size in &sizes {
            let mut any = false;
            for &ty in &types {
                if let Some(format) = vertex_format(ty, size) {
                    assert_ne!(format, vk::Format::UNDEFINED);
                    assert!(format.as_raw() < 256, "{format:?} must fit the key's byte");
                    defined += 1;
                    any = true;
                }
            }
            assert_eq!(any, size != VertexAttributeSize::INVALID, "{size:?}");
        }
        assert_eq!(defined, 12 * 4 + 16 * 2 + 9);
        assert_eq!(
            vertex_format(
                VertexAttributeType::FLOAT,
                VertexAttributeSize::SIZE_R32_G32_B32
            ),
            Some(vk::Format::R32G32B32_SFLOAT)
        );
        assert_eq!(
            vertex_format(VertexAttributeType::FLOAT, VertexAttributeSize::SIZE_R8),
            None
        );
    }

    #[test]
    fn topology_classes_keep_list_strip_and_fan_together() {
        for (topology, class) in [
            (
                PrimitiveTopology::LINE_STRIP,
                vk::PrimitiveTopology::LINE_LIST,
            ),
            (
                PrimitiveTopology::LINE_LOOP,
                vk::PrimitiveTopology::LINE_LIST,
            ),
            (
                PrimitiveTopology::TRIANGLE_FAN,
                vk::PrimitiveTopology::TRIANGLE_LIST,
            ),
            (
                PrimitiveTopology::QUAD_STRIP,
                vk::PrimitiveTopology::TRIANGLE_LIST,
            ),
            (
                PrimitiveTopology::TRIANGLE_STRIP_ADJACENCY,
                vk::PrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY,
            ),
            (
                PrimitiveTopology::PATCHES,
                vk::PrimitiveTopology::PATCH_LIST,
            ),
            (PrimitiveTopology::POINTS, vk::PrimitiveTopology::POINT_LIST),
        ] {
            assert_eq!(topology_class(primitive_topology(topology).unwrap()), class);
        }
    }
}
