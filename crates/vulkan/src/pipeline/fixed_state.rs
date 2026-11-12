//! The fixed-function part of a graphics pipeline key (port of `fixed_pipeline_state.cpp`).
//!
//! [`FixedStateTracker::refresh`] runs before every draw. It re-reads a register group only when
//! that group's dirty flag is set, skips groups the device leaves to dynamic state, and stores
//! values already translated to Vulkan, so building a pipeline reads the key directly.

use std::hash::{Hash, Hasher};
use std::mem::{offset_of, size_of};

use ash::vk;
use bytemuck::{Pod, Zeroable};
use lichii_gpu::engines::maxwell_3d::regs::{
    Blend, BlendPerTarget, DepthMode, PrimitiveTopology, ProvokingVertex, StencilOp,
    ViewportClipControlGeometryClip as GeometryClip,
};
use lichii_gpu::engines::maxwell_3d::{DirtyState, Regs};

use super::PipelineFeatures;
use super::maxwell_to_vk as mvk;

/// Dirty flags the tracker owns, allocated from `dirty::LAST_COMMON`. Their methods are
/// mapped through method table 0; table 1 stays free for the rasterizer's dynamic state.
pub mod flags {
    use lichii_gpu::engines::maxwell_3d::dirty::LAST_COMMON;

    /// Vertex attribute formats and per-stream instancing.
    pub const VERTEX_INPUT: u8 = LAST_COMMON;
    /// Blend state and colour masks.
    pub const BLENDING: u8 = LAST_COMMON + 1;
    /// Depth and stencil test state.
    pub const DEPTH_STENCIL: u8 = LAST_COMMON + 2;
    /// Culling, polygon mode, discard, restart, polygon offset, clip, provoking vertex, alpha
    /// to coverage, logic op and patch size.
    pub const RASTER: u8 = LAST_COMMON + 3;
    /// The first flag index after the tracker's.
    pub const NEXT_FREE: u8 = LAST_COMMON + 4;
}

pub(crate) const GROUP_DEPTH_STENCIL: u8 = 1 << 0;
pub(crate) const GROUP_EDS2: u8 = 1 << 1;
pub(crate) const GROUP_BLEND: u8 = 1 << 2;
pub(crate) const GROUP_VERTEX: u8 = 1 << 3;
pub(crate) const GROUP_STRIDES: u8 = 1 << 4;

pub(crate) const RASTER_DEPTH_CLAMP: u8 = 1 << 0;
pub(crate) const RASTER_NDC_MINUS_ONE_TO_ONE: u8 = 1 << 1;
pub(crate) const RASTER_PROVOKING_LAST: u8 = 1 << 2;
pub(crate) const RASTER_ALPHA_TO_COVERAGE: u8 = 1 << 3;
pub(crate) const RASTER_ALPHA_TO_ONE: u8 = 1 << 4;
pub(crate) const RASTER_LOGIC_OP: u8 = 1 << 5;

pub(crate) const EDS2_RASTERIZER_DISCARD: u8 = 1 << 0;
pub(crate) const EDS2_DEPTH_BIAS: u8 = 1 << 1;
pub(crate) const EDS2_PRIMITIVE_RESTART: u8 = 1 << 2;

pub(crate) const DEPTH_TEST: u8 = 1 << 0;
pub(crate) const DEPTH_WRITE: u8 = 1 << 1;
pub(crate) const DEPTH_BOUNDS: u8 = 1 << 2;
pub(crate) const STENCIL_TEST: u8 = 1 << 3;

/// Stencil operations of one face, as `vk::StencilOp` and `vk::CompareOp` values.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct StencilFace {
    pub(crate) fail: u8,
    pub(crate) depth_fail: u8,
    pub(crate) pass: u8,
    pub(crate) compare: u8,
}

/// One colour attachment's blending, as Vulkan values.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct BlendAttachment {
    /// `vk::ColorComponentFlags` bits.
    pub(crate) write_mask: u8,
    pub(crate) enable: u8,
    pub(crate) color_op: u8,
    pub(crate) src_color: u8,
    pub(crate) dst_color: u8,
    pub(crate) alpha_op: u8,
    pub(crate) src_alpha: u8,
    pub(crate) dst_alpha: u8,
}

/// One vertex attribute; `format` 0 (`vk::Format::UNDEFINED`) when it is disabled.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct VertexAttribute {
    pub(crate) format: u8,
    pub(crate) binding: u8,
    pub(crate) offset: u16,
}

/// Fixed-function state a pipeline is built with. Fields are Vulkan enum values; state the
/// device takes dynamically stays zero, and the trailing groups the device makes dynamic are
/// left out of hashing, comparison and the disk cache (see [`FixedPipelineState::as_bytes`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FixedPipelineState {
    pub(crate) groups: u8,
    pub(crate) topology: u8,
    pub(crate) patch_control_points: u8,
    pub(crate) polygon_mode: u8,
    pub(crate) logic_op: u8,
    pub(crate) raster: u8,
    pub(crate) eds2: u8,
    pub(crate) cull_mode: u8,
    pub(crate) front_face: u8,
    pub(crate) depth: u8,
    pub(crate) depth_compare: u8,
    _pad: u8,
    pub(crate) stencil_front: StencilFace,
    pub(crate) stencil_back: StencilFace,
    pub(crate) blend: [BlendAttachment; 8],
    pub(crate) attributes: [VertexAttribute; 32],
    /// Bit `i`: binding `i` advances per instance.
    pub(crate) instanced: u32,
    pub(crate) strides: [u16; 32],
}

impl FixedPipelineState {
    /// The bytes that identify the pipeline: the header always, then blending, vertex input
    /// and strides only while the device keeps them static. 20 bytes on a device with dynamic
    /// blending and vertex input, 280 with neither.
    pub fn as_bytes(&self) -> &[u8] {
        &bytemuck::bytes_of(self)[..self.len()]
    }

    pub(crate) fn len(&self) -> usize {
        if self.groups & GROUP_STRIDES != 0 {
            size_of::<Self>()
        } else if self.groups & GROUP_VERTEX != 0 {
            offset_of!(Self, strides)
        } else if self.groups & GROUP_BLEND != 0 {
            offset_of!(Self, attributes)
        } else {
            offset_of!(Self, blend)
        }
    }

    pub(crate) fn vk_topology(&self) -> vk::PrimitiveTopology {
        vk::PrimitiveTopology::from_raw(i32::from(self.topology))
    }
}

impl PartialEq for FixedPipelineState {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for FixedPipelineState {}

impl Hash for FixedPipelineState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(self.as_bytes());
    }
}

fn groups(features: &PipelineFeatures) -> u8 {
    let mut groups = 0;
    if !features.extended_dynamic_state {
        groups |= GROUP_DEPTH_STENCIL;
    }
    if !features.extended_dynamic_state2 {
        groups |= GROUP_EDS2;
    }
    if !features.dynamic_blend {
        groups |= GROUP_BLEND;
    }
    if !features.dynamic_vertex_input {
        groups |= GROUP_VERTEX;
        if !features.extended_dynamic_state {
            groups |= GROUP_STRIDES;
        }
    }
    groups
}

fn byte<T>(value: T) -> u8
where
    i32: From<T>,
{
    i32::from(value) as u8
}

fn word(offset: usize) -> u32 {
    (offset / 4) as u32
}

fn words<T>() -> u32 {
    (size_of::<T>() / 4) as u32
}

/// Builds the [`FixedPipelineState`] of the next draw from the 3D registers.
pub struct FixedStateTracker {
    features: PipelineFeatures,
    state: FixedPipelineState,
    topology: Option<PrimitiveTopology>,
}

impl FixedStateTracker {
    pub fn new(features: PipelineFeatures) -> Self {
        let mut state = FixedPipelineState::zeroed();
        state.groups = groups(&features);
        Self {
            features,
            state,
            topology: None,
        }
    }

    pub fn features(&self) -> &PipelineFeatures {
        &self.features
    }

    /// Maps every register the tracker reads to its flag in method table 0. Call once on each
    /// register file whose draws the tracker sees.
    pub fn register_dirty_flags(dirty: &mut DirtyState) {
        for (first, count, flag) in Self::methods() {
            dirty.fill(0, first, count, flag);
        }
    }

    /// `(first method, count, flag)` for every register range the tracker reads.
    pub(crate) fn methods() -> [(u32, u32, u8); 30] {
        use flags::*;
        let one = |offset: usize, flag: u8| (word(offset), 1, flag);
        [
            (
                word(offset_of!(Regs, vertex_attrib_format)),
                32,
                VERTEX_INPUT,
            ),
            (
                word(offset_of!(Regs, vertex_stream_instances)),
                32,
                VERTEX_INPUT,
            ),
            (word(offset_of!(Regs, blend)), words::<Blend>(), BLENDING),
            (
                word(offset_of!(Regs, blend_per_target)),
                words::<[BlendPerTarget; 8]>(),
                BLENDING,
            ),
            one(offset_of!(Regs, blend_per_target_enabled), BLENDING),
            (word(offset_of!(Regs, color_mask)), 8, BLENDING),
            one(offset_of!(Regs, color_mask_common), BLENDING),
            one(offset_of!(Regs, depth_test_enable), DEPTH_STENCIL),
            one(offset_of!(Regs, depth_write_enabled), DEPTH_STENCIL),
            one(offset_of!(Regs, depth_test_func), DEPTH_STENCIL),
            one(offset_of!(Regs, depth_bounds_enable), DEPTH_STENCIL),
            one(offset_of!(Regs, stencil_enable), DEPTH_STENCIL),
            one(offset_of!(Regs, stencil_two_side_enable), DEPTH_STENCIL),
            (
                word(offset_of!(Regs, stencil_front_op)),
                words::<StencilOp>(),
                DEPTH_STENCIL,
            ),
            (
                word(offset_of!(Regs, stencil_back_op)),
                words::<StencilOp>(),
                DEPTH_STENCIL,
            ),
            one(offset_of!(Regs, gl_cull_test_enabled), RASTER),
            one(offset_of!(Regs, gl_front_face), RASTER),
            one(offset_of!(Regs, gl_cull_face), RASTER),
            one(offset_of!(Regs, window_origin), RASTER),
            one(offset_of!(Regs, polygon_mode_front), RASTER),
            one(offset_of!(Regs, rasterize_enable), RASTER),
            one(offset_of!(Regs, primitive_restart.enabled), RASTER),
            (
                word(offset_of!(Regs, polygon_offset_point_enable)),
                3,
                RASTER,
            ),
            one(offset_of!(Regs, viewport_clip_control), RASTER),
            one(offset_of!(Regs, depth_mode), RASTER),
            one(offset_of!(Regs, provoking_vertex), RASTER),
            one(offset_of!(Regs, anti_alias_alpha_control), RASTER),
            one(offset_of!(Regs, logic_op.enable), RASTER),
            one(offset_of!(Regs, logic_op.op), RASTER),
            one(offset_of!(Regs, patch_vertices), RASTER),
        ]
    }

    /// The state for a draw of `topology` with `regs`, consuming the tracker's flags in
    /// `dirty`. Allocates nothing; with no flag set it reads no register group.
    pub fn refresh(
        &mut self,
        regs: &Regs,
        dirty: &mut DirtyState,
        topology: PrimitiveTopology,
    ) -> &FixedPipelineState {
        let topology_changed = self.topology != Some(topology);
        if topology_changed {
            self.topology = Some(topology);
            let host =
                mvk::primitive_topology(topology).unwrap_or(vk::PrimitiveTopology::TRIANGLE_LIST);
            let keyed = if self.features.extended_dynamic_state {
                mvk::topology_class(host)
            } else {
                host
            };
            self.state.topology = byte(keyed.as_raw());
        }
        if dirty.take(flags::RASTER) | topology_changed {
            self.refresh_raster(regs, topology);
        }
        if dirty.take(flags::DEPTH_STENCIL) && self.state.groups & GROUP_DEPTH_STENCIL != 0 {
            self.refresh_depth_stencil(regs);
        }
        if dirty.take(flags::BLENDING) && self.state.groups & GROUP_BLEND != 0 {
            self.refresh_blending(regs);
        }
        if dirty.take(flags::VERTEX_INPUT) && self.state.groups & GROUP_VERTEX != 0 {
            self.refresh_vertex_input(regs);
        }
        if self.state.groups & GROUP_STRIDES != 0 {
            for (stride, stream) in self.state.strides.iter_mut().zip(&regs.vertex_streams) {
                *stride = stream.stride() as u16;
            }
        }
        &self.state
    }

    fn refresh_raster(&mut self, regs: &Regs, topology: PrimitiveTopology) {
        let f = self.features;
        let s = &mut self.state;
        s.patch_control_points = if topology == PrimitiveTopology::PATCHES {
            regs.patch_vertices.clamp(1, 32) as u8
        } else {
            0
        };
        s.polygon_mode = if f.dynamic_polygon_mode || !f.fill_mode_non_solid {
            byte(vk::PolygonMode::FILL.as_raw())
        } else {
            byte(
                mvk::polygon_mode(regs.polygon_mode_front)
                    .unwrap_or(vk::PolygonMode::FILL)
                    .as_raw(),
            )
        };
        let logic_op_enabled = regs.logic_op.enable != 0;
        s.logic_op = if f.logic_op && (f.dynamic_logic_op_enable || logic_op_enabled) {
            byte(
                mvk::logic_op(regs.logic_op.op)
                    .unwrap_or(vk::LogicOp::COPY)
                    .as_raw(),
            )
        } else {
            0
        };

        let mut raster = 0;
        let clip = regs.viewport_clip_control.geometry_clip();
        let clamps = !matches!(
            clip,
            GeometryClip::PASSTHROUGH | GeometryClip::FRUSTUM_XYZ | GeometryClip::FRUSTUM_Z
        );
        let alpha = regs.anti_alias_alpha_control;
        for (set, bit) in [
            (
                !f.dynamic_depth_clamp && f.depth_clamp && clamps,
                RASTER_DEPTH_CLAMP,
            ),
            (
                !f.dynamic_depth_clip_negative_one_to_one
                    && f.depth_clip_control
                    && regs.depth_mode == DepthMode::MINUS_ONE_TO_ONE,
                RASTER_NDC_MINUS_ONE_TO_ONE,
            ),
            (
                !f.dynamic_provoking_vertex
                    && f.provoking_vertex_last
                    && regs.provoking_vertex == ProvokingVertex::LAST,
                RASTER_PROVOKING_LAST,
            ),
            (
                !f.dynamic_alpha_to_coverage && alpha.alpha_to_coverage() != 0,
                RASTER_ALPHA_TO_COVERAGE,
            ),
            (
                !f.dynamic_alpha_to_one && f.alpha_to_one && alpha.alpha_to_one() != 0,
                RASTER_ALPHA_TO_ONE,
            ),
            (
                !f.dynamic_logic_op_enable && f.logic_op && logic_op_enabled,
                RASTER_LOGIC_OP,
            ),
        ] {
            if set {
                raster |= bit;
            }
        }
        s.raster = raster;

        if s.groups & GROUP_EDS2 != 0 {
            let host =
                mvk::primitive_topology(topology).unwrap_or(vk::PrimitiveTopology::TRIANGLE_LIST);
            use vk::PrimitiveTopology as T;
            let offset_enable = match host {
                T::POINT_LIST => regs.polygon_offset_point_enable,
                T::LINE_LIST
                | T::LINE_STRIP
                | T::LINE_LIST_WITH_ADJACENCY
                | T::LINE_STRIP_WITH_ADJACENCY => regs.polygon_offset_line_enable,
                _ => regs.polygon_offset_fill_enable,
            };
            let restartable = matches!(
                host,
                T::LINE_STRIP
                    | T::TRIANGLE_STRIP
                    | T::TRIANGLE_FAN
                    | T::LINE_STRIP_WITH_ADJACENCY
                    | T::TRIANGLE_STRIP_WITH_ADJACENCY
            );
            let mut eds2 = 0;
            if regs.rasterize_enable == 0 {
                eds2 |= EDS2_RASTERIZER_DISCARD;
            }
            if offset_enable != 0 {
                eds2 |= EDS2_DEPTH_BIAS;
            }
            if restartable && regs.primitive_restart.enabled != 0 {
                eds2 |= EDS2_PRIMITIVE_RESTART;
            }
            s.eds2 = eds2;
        }

        if s.groups & GROUP_DEPTH_STENCIL != 0 {
            let cull = if regs.gl_cull_test_enabled != 0 {
                mvk::cull_mode(regs.gl_cull_face).unwrap_or(vk::CullModeFlags::BACK)
            } else {
                vk::CullModeFlags::NONE
            };
            s.cull_mode = cull.as_raw() as u8;
            let mut face =
                mvk::front_face(regs.gl_front_face).unwrap_or(vk::FrontFace::COUNTER_CLOCKWISE);
            if regs.window_origin.flip_y() != 0 {
                face = if face == vk::FrontFace::CLOCKWISE {
                    vk::FrontFace::COUNTER_CLOCKWISE
                } else {
                    vk::FrontFace::CLOCKWISE
                };
            }
            s.front_face = byte(face.as_raw());
        }
    }

    fn refresh_depth_stencil(&mut self, regs: &Regs) {
        let s = &mut self.state;
        let mut depth = 0;
        for (set, bit) in [
            (regs.depth_test_enable != 0, DEPTH_TEST),
            (regs.depth_write_enabled != 0, DEPTH_WRITE),
            (
                self.features.depth_bounds && regs.depth_bounds_enable != 0,
                DEPTH_BOUNDS,
            ),
            (regs.stencil_enable != 0, STENCIL_TEST),
        ] {
            if set {
                depth |= bit;
            }
        }
        s.depth = depth;
        s.depth_compare = byte(
            mvk::comparison_op(regs.depth_test_func)
                .unwrap_or(vk::CompareOp::ALWAYS)
                .as_raw(),
        );
        let face = |op: &StencilOp| {
            let stencil = |op| byte(mvk::stencil_op(op).unwrap_or(vk::StencilOp::KEEP).as_raw());
            StencilFace {
                fail: stencil(op.fail),
                depth_fail: stencil(op.zfail),
                pass: stencil(op.zpass),
                compare: byte(
                    mvk::comparison_op(op.func)
                        .unwrap_or(vk::CompareOp::ALWAYS)
                        .as_raw(),
                ),
            }
        };
        s.stencil_front = face(&regs.stencil_front_op);
        s.stencil_back = if regs.stencil_two_side_enable != 0 {
            face(&regs.stencil_back_op)
        } else {
            s.stencil_front
        };
    }

    fn refresh_blending(&mut self, regs: &Regs) {
        let dual_source = self.features.dual_src_blend;
        let factor = |factor| {
            let host = mvk::blend_factor(factor).unwrap_or(vk::BlendFactor::ONE);
            byte(
                if dual_source {
                    host
                } else {
                    mvk::single_source(host)
                }
                .as_raw(),
            )
        };
        let op = |op| byte(mvk::blend_equation(op).unwrap_or(vk::BlendOp::ADD).as_raw());
        let common_mask = regs.color_mask_common != 0;
        for (index, attachment) in self.state.blend.iter_mut().enumerate() {
            let mask = regs.color_mask[if common_mask { 0 } else { index }];
            let write_mask = (mask.r() | mask.g() << 1 | mask.b() << 2 | mask.a() << 3) as u8;
            *attachment = BlendAttachment {
                write_mask,
                ..Default::default()
            };
            if regs.blend.enable[index] == 0 {
                continue;
            }
            let b = &regs.blend;
            let t = &regs.blend_per_target[index];
            let (color_op, src_color, dst_color, alpha_op, src_alpha, dst_alpha) =
                if regs.blend_per_target_enabled != 0 {
                    (
                        t.color_op,
                        t.color_source,
                        t.color_dest,
                        t.alpha_op,
                        t.alpha_source,
                        t.alpha_dest,
                    )
                } else {
                    (
                        b.color_op,
                        b.color_source,
                        b.color_dest,
                        b.alpha_op,
                        b.alpha_source,
                        b.alpha_dest,
                    )
                };
            attachment.enable = 1;
            attachment.color_op = op(color_op);
            attachment.src_color = factor(src_color);
            attachment.dst_color = factor(dst_color);
            attachment.alpha_op = op(alpha_op);
            attachment.src_alpha = factor(src_alpha);
            attachment.dst_alpha = factor(dst_alpha);
        }
    }

    fn refresh_vertex_input(&mut self, regs: &Regs) {
        let s = &mut self.state;
        for (attribute, input) in s.attributes.iter_mut().zip(&regs.vertex_attrib_format) {
            let format = (input.constant() == 0)
                .then(|| mvk::vertex_format(input.type_(), input.size()))
                .flatten();
            *attribute = format.map_or_else(VertexAttribute::default, |format| VertexAttribute {
                format: byte(format.as_raw()),
                binding: input.buffer() as u8,
                offset: input.offset() as u16,
            });
        }
        s.instanced = regs
            .vertex_stream_instances
            .is_instanced
            .iter()
            .enumerate()
            .fold(0, |mask, (index, &on)| mask | u32::from(on != 0) << index);
    }
}

#[cfg(test)]
mod tests {
    use lichii_gpu::engines::maxwell_3d::RegisterState;
    use lichii_gpu::engines::maxwell_3d::regs::{
        BlendEquation, BlendFactor, ComparisonOp, CullFace, FrontFace, LogicOpOp, PolygonMode,
        StencilOpOp, VertexAttributeSize, VertexAttributeType,
    };

    use super::*;

    fn all_static() -> PipelineFeatures {
        PipelineFeatures {
            depth_clamp: true,
            depth_bounds: true,
            logic_op: true,
            alpha_to_one: true,
            fill_mode_non_solid: true,
            dual_src_blend: true,
            depth_clip_control: true,
            provoking_vertex_last: true,
            multi_viewport: true,
            ..Default::default()
        }
    }

    fn all_dynamic() -> PipelineFeatures {
        PipelineFeatures {
            extended_dynamic_state: true,
            extended_dynamic_state2: true,
            dynamic_vertex_input: true,
            dynamic_blend: true,
            dynamic_depth_clamp: true,
            dynamic_logic_op_enable: true,
            dynamic_alpha_to_coverage: true,
            dynamic_alpha_to_one: true,
            dynamic_polygon_mode: true,
            dynamic_depth_clip_negative_one_to_one: true,
            dynamic_provoking_vertex: true,
            ..all_static()
        }
    }

    /// A register file with the tracker's flags installed, every flag set.
    fn registers() -> RegisterState {
        let mut state = RegisterState::new();
        FixedStateTracker::register_dirty_flags(&mut state.dirty);
        state.dirty.set_all();
        state
    }

    fn extract(
        features: PipelineFeatures,
        regs: &Regs,
        topology: PrimitiveTopology,
    ) -> FixedPipelineState {
        let mut dirty = DirtyState::new();
        *FixedStateTracker::new(features).refresh(regs, &mut dirty, topology)
    }

    type Edit = (&'static str, fn(&mut Regs));

    /// One edit per field the key must react to, each against `Regs::with_defaults`.
    const EDITS: &[Edit] = &[
        ("polygon mode", |r| r.polygon_mode_front = PolygonMode::LINE),
        ("cull enable", |r| r.gl_cull_test_enabled = 1),
        ("front face", |r| {
            r.gl_front_face = FrontFace::COUNTER_CLOCK_WISE
        }),
        ("window flip", |r| r.window_origin.0 |= 1 << 4),
        ("depth test", |r| r.depth_test_enable = 1),
        ("depth write", |r| r.depth_write_enabled = 1),
        ("depth func", |r| r.depth_test_func = ComparisonOp::LESS_GL),
        ("depth bounds", |r| r.depth_bounds_enable = 1),
        ("stencil test", |r| r.stencil_enable = 1),
        ("stencil front fail", |r| {
            r.stencil_front_op.fail = StencilOpOp::ZERO_GL
        }),
        ("stencil front zfail", |r| {
            r.stencil_front_op.zfail = StencilOpOp::INCR_GL
        }),
        ("stencil front zpass", |r| {
            r.stencil_front_op.zpass = StencilOpOp::REPLACE_D3D
        }),
        ("stencil front func", |r| {
            r.stencil_front_op.func = ComparisonOp::EQUAL_GL
        }),
        ("stencil back", |r| {
            r.stencil_back_op.zpass = StencilOpOp::DECR_GL
        }),
        ("rasterizer discard", |r| r.rasterize_enable = 0),
        ("polygon offset fill", |r| r.polygon_offset_fill_enable = 1),
        ("depth clamp", |r| r.viewport_clip_control.0 = 1 << 11),
        ("depth mode", |r| r.depth_mode = DepthMode::ZERO_TO_ONE),
        ("provoking vertex", |r| {
            r.provoking_vertex = ProvokingVertex::LAST
        }),
        ("alpha to coverage", |r| r.anti_alias_alpha_control.0 = 1),
        ("alpha to one", |r| r.anti_alias_alpha_control.0 = 1 << 4),
        ("logic op enable", |r| r.logic_op.enable = 1),
        ("blend enable", |r| r.blend.enable[3] = 1),
        ("blend factor", |r| {
            r.blend.enable[0] = 1;
            r.blend.color_dest = BlendFactor::ONE_MINUS_SOURCE_ALPHA_GL;
        }),
        ("blend op", |r| {
            r.blend.enable[0] = 1;
            r.blend.alpha_op = BlendEquation::MAX_GL;
        }),
        ("per-target blend", |r| {
            r.blend.enable[1] = 1;
            r.blend_per_target_enabled = 1;
            r.blend_per_target[1].color_source = BlendFactor::SOURCE1_COLOR_GL;
        }),
        ("colour mask", |r| r.color_mask[2].0 = 0x0111),
        ("common colour mask", |r| {
            r.color_mask[0].0 = 0;
            r.color_mask_common = 1;
        }),
        ("attribute enable", |r| r.vertex_attrib_format[4] = float4()),
        ("attribute offset", |r| {
            r.vertex_attrib_format[0] = float4().with_bits(7, 14, 16);
        }),
        ("attribute format", |r| {
            r.vertex_attrib_format[0] =
                float4().with_bits(21, 6, VertexAttributeSize::SIZE_R16_G16.0)
        }),
        ("attribute binding", |r| {
            r.vertex_attrib_format[0] = float4().with_bits(0, 5, 3)
        }),
        ("instancing", |r| {
            r.vertex_stream_instances.is_instanced[2] = 1
        }),
        ("stride", |r| r.vertex_streams[1].bits.0 = 12),
    ];

    use lichii_gpu::engines::maxwell_3d::regs::VertexAttribute as GuestAttribute;

    fn float4() -> GuestAttribute {
        GuestAttribute(0)
            .with_bits(21, 6, VertexAttributeSize::SIZE_R32_G32_B32_A32.0)
            .with_bits(27, 3, VertexAttributeType::FLOAT.0)
    }

    fn base() -> Box<Regs> {
        let mut regs = Regs::with_defaults();
        regs.vertex_attrib_format[0] = float4();
        regs.stencil_two_side_enable = 1;
        regs
    }

    #[test]
    fn extraction_is_stable() {
        let regs = base();
        let a = extract(all_static(), &regs, PrimitiveTopology::TRIANGLES);
        let b = extract(all_static(), &regs, PrimitiveTopology::TRIANGLES);
        assert_eq!(bytemuck::bytes_of(&a), bytemuck::bytes_of(&b));
        assert_eq!(
            super::super::fx_hash(a.as_bytes()),
            super::super::fx_hash(b.as_bytes())
        );
        let mut tracker = FixedStateTracker::new(all_static());
        let mut dirty = DirtyState::new();
        let first = *tracker.refresh(&regs, &mut dirty, PrimitiveTopology::TRIANGLES);
        dirty.set_all();
        let again = *tracker.refresh(&regs, &mut dirty, PrimitiveTopology::TRIANGLES);
        assert_eq!(bytemuck::bytes_of(&first), bytemuck::bytes_of(&again));
        assert_eq!(first, a);
        let zeroed = extract(all_static(), &Regs::zeroed(), PrimitiveTopology::TRIANGLES);
        assert_eq!(
            zeroed,
            extract(all_static(), &Regs::zeroed(), PrimitiveTopology::TRIANGLES)
        );
    }

    #[test]
    fn key_reacts_to_every_static_field() {
        let base_regs = base();
        let base_key = extract(all_static(), &base_regs, PrimitiveTopology::TRIANGLE_STRIP);
        assert_eq!(base_key.as_bytes().len(), size_of::<FixedPipelineState>());
        for (name, edit) in EDITS {
            let mut regs = base_regs.clone();
            edit(&mut regs);
            let key = extract(all_static(), &regs, PrimitiveTopology::TRIANGLE_STRIP);
            assert_ne!(key, base_key, "{name} must change the key");
        }
        let mut restart = base_regs.clone();
        restart.primitive_restart.enabled = 1;
        assert_ne!(
            extract(all_static(), &restart, PrimitiveTopology::TRIANGLE_STRIP),
            base_key
        );
        assert_eq!(
            extract(all_static(), &restart, PrimitiveTopology::TRIANGLES),
            extract(all_static(), &base_regs, PrimitiveTopology::TRIANGLES),
            "lists cannot restart without primitiveTopologyListRestart"
        );
        let mut logic = base_regs.clone();
        logic.logic_op.enable = 1;
        let enabled = extract(all_static(), &logic, PrimitiveTopology::TRIANGLES);
        logic.logic_op.op = LogicOpOp::XOR;
        assert_ne!(
            extract(all_static(), &logic, PrimitiveTopology::TRIANGLES),
            enabled
        );
        let mut patches = base_regs.clone();
        patches.patch_vertices = 3;
        let three = extract(all_static(), &patches, PrimitiveTopology::PATCHES);
        patches.patch_vertices = 4;
        assert_ne!(
            extract(all_static(), &patches, PrimitiveTopology::PATCHES),
            three
        );
        assert_eq!(three.patch_control_points, 3);
    }

    #[test]
    fn dynamic_state_stays_out_of_the_key() {
        let base_regs = base();
        let base_key = extract(all_dynamic(), &base_regs, PrimitiveTopology::TRIANGLES);
        assert_eq!(base_key.as_bytes().len(), 20);
        for (name, edit) in EDITS {
            let mut regs = base_regs.clone();
            edit(&mut regs);
            let key = extract(all_dynamic(), &regs, PrimitiveTopology::TRIANGLES);
            assert_eq!(
                key, base_key,
                "{name} is dynamic and must not change the key"
            );
        }
        for topology in [
            PrimitiveTopology::TRIANGLE_STRIP,
            PrimitiveTopology::TRIANGLE_FAN,
            PrimitiveTopology::QUADS,
        ] {
            assert_eq!(
                extract(all_dynamic(), &base_regs, topology),
                base_key,
                "{topology:?}"
            );
        }
        assert_ne!(
            extract(all_dynamic(), &base_regs, PrimitiveTopology::LINES),
            base_key,
            "topology classes stay apart"
        );
        let mut logic = base_regs.clone();
        logic.logic_op.op = LogicOpOp::XOR;
        assert_ne!(
            extract(all_dynamic(), &logic, PrimitiveTopology::TRIANGLES),
            base_key,
            "the logic op itself is static (no VK_EXT_extended_dynamic_state2 logic op)"
        );
    }

    #[test]
    fn key_length_follows_the_dynamic_groups() {
        let regs = base();
        let len = |features| {
            extract(features, &regs, PrimitiveTopology::TRIANGLES)
                .as_bytes()
                .len()
        };
        assert_eq!(len(all_static()), 280);
        let eds = PipelineFeatures {
            extended_dynamic_state: true,
            extended_dynamic_state2: true,
            ..all_static()
        };
        assert_eq!(len(eds), 216);
        assert_eq!(
            len(PipelineFeatures {
                dynamic_vertex_input: true,
                ..eds
            }),
            84
        );
        assert_eq!(len(all_dynamic()), 20);
    }

    #[test]
    fn unsupported_features_are_cleared() {
        let mut regs = base();
        regs.polygon_mode_front = PolygonMode::POINT;
        regs.logic_op.enable = 1;
        regs.anti_alias_alpha_control.0 = 1 << 4;
        regs.depth_bounds_enable = 1;
        regs.blend.enable[0] = 1;
        regs.blend.color_source = BlendFactor::SOURCE1_ALPHA_D3D;
        let none = PipelineFeatures::default();
        let key = extract(none, &regs, PrimitiveTopology::TRIANGLES);
        assert_eq!(key.polygon_mode, byte(vk::PolygonMode::FILL.as_raw()));
        assert_eq!(key.logic_op, 0);
        assert_eq!(key.raster, 0);
        assert_eq!(key.depth & DEPTH_BOUNDS, 0);
        assert_eq!(
            key.blend[0].src_color,
            byte(vk::BlendFactor::SRC_ALPHA.as_raw())
        );
        let full = extract(all_static(), &regs, PrimitiveTopology::TRIANGLES);
        assert_eq!(full.polygon_mode, byte(vk::PolygonMode::POINT.as_raw()));
        assert_eq!(full.logic_op, byte(vk::LogicOp::COPY.as_raw()));
        assert_eq!(
            full.blend[0].src_color,
            byte(vk::BlendFactor::SRC1_ALPHA.as_raw())
        );
    }

    #[test]
    fn values_are_translated_into_the_key() {
        let mut regs = base();
        regs.gl_cull_test_enabled = 1;
        regs.gl_cull_face = CullFace::FRONT_AND_BACK;
        regs.depth_test_func = ComparisonOp::GREATER_D3D;
        regs.stencil_two_side_enable = 0;
        regs.stencil_front_op.zpass = StencilOpOp::INCR_SATURATE_GL;
        regs.blend.enable[0] = 1;
        regs.blend.color_op = BlendEquation::REVERSE_SUBTRACT_D3D;
        regs.vertex_stream_instances.is_instanced[5] = 1;
        regs.color_mask[0].0 = 0x1001;
        regs.color_mask[7].0 = 0x0110;
        let key = extract(all_static(), &regs, PrimitiveTopology::TRIANGLES);
        assert_eq!(key.blend[7].write_mask, 0b0110);
        regs.color_mask_common = 1;
        let common = extract(all_static(), &regs, PrimitiveTopology::TRIANGLES);
        assert_eq!(
            common.blend[7].write_mask, 0b1001,
            "the common mask is colour_mask[0]"
        );
        regs.color_mask_common = 0;
        regs.color_mask[0].0 = 0x1111;
        let key = extract(all_static(), &regs, PrimitiveTopology::TRIANGLES);
        assert_eq!(
            key.cull_mode,
            vk::CullModeFlags::FRONT_AND_BACK.as_raw() as u8
        );
        assert_eq!(key.front_face, byte(vk::FrontFace::CLOCKWISE.as_raw()));
        assert_eq!(key.depth_compare, byte(vk::CompareOp::GREATER.as_raw()));
        assert_eq!(key.stencil_back, key.stencil_front);
        assert_eq!(
            key.stencil_front.pass,
            byte(vk::StencilOp::INCREMENT_AND_CLAMP.as_raw())
        );
        assert_eq!(
            key.blend[0].color_op,
            byte(vk::BlendOp::REVERSE_SUBTRACT.as_raw())
        );
        assert_eq!(key.blend[0].write_mask, 0xf);
        assert_eq!(
            key.attributes[0].format,
            byte(vk::Format::R32G32B32A32_SFLOAT.as_raw())
        );
        assert_eq!(
            key.attributes[1],
            VertexAttribute::default(),
            "constant attributes are off"
        );
        assert_eq!(key.instanced, 1 << 5);
        assert_eq!(key.vk_topology(), vk::PrimitiveTopology::TRIANGLE_LIST);
    }

    #[test]
    fn only_dirty_groups_are_reread() {
        let mut state = registers();
        let mut tracker = FixedStateTracker::new(all_static());
        let topology = PrimitiveTopology::TRIANGLES;
        let first = *tracker.refresh(&state.regs, &mut state.dirty, topology);
        for flag in [
            flags::VERTEX_INPUT,
            flags::BLENDING,
            flags::DEPTH_STENCIL,
            flags::RASTER,
        ] {
            assert!(!state.dirty.is_set(flag));
        }

        state.regs.depth_test_enable = 1;
        state.regs.blend.enable[0] = 1;
        let unchanged = *tracker.refresh(&state.regs, &mut state.dirty, topology);
        assert_eq!(
            unchanged, first,
            "no flag set: the tracker must not re-read registers"
        );

        let depth = word(offset_of!(Regs, depth_test_enable));
        state.write(depth, 0);
        state.write(depth, 1);
        assert!(state.dirty.is_set(flags::DEPTH_STENCIL));
        assert!(!state.dirty.is_set(flags::BLENDING));
        let depth_only = *tracker.refresh(&state.regs, &mut state.dirty, topology);
        assert_eq!(depth_only.depth & DEPTH_TEST, DEPTH_TEST);
        assert_eq!(
            depth_only.blend, first.blend,
            "blending was not marked dirty"
        );

        let blend_enable = word(offset_of!(Regs, blend.enable));
        state.write(blend_enable, 0);
        state.write(blend_enable, 1);
        let blended = *tracker.refresh(&state.regs, &mut state.dirty, topology);
        assert_eq!(blended.blend[0].enable, 1);

        let attribute = word(offset_of!(Regs, vertex_attrib_format)) + 7;
        state.write(attribute, float4().0);
        let attributes = *tracker.refresh(&state.regs, &mut state.dirty, topology);
        assert_eq!(
            attributes.attributes[7].format,
            byte(vk::Format::R32G32B32A32_SFLOAT.as_raw())
        );

        let cull = word(offset_of!(Regs, gl_cull_test_enabled));
        state.write(cull, 1);
        let culled = *tracker.refresh(&state.regs, &mut state.dirty, topology);
        assert_eq!(culled.cull_mode, vk::CullModeFlags::BACK.as_raw() as u8);
    }

    #[test]
    fn tracker_methods_are_free_in_table_zero() {
        let common = DirtyState::new();
        let mut installed = DirtyState::new();
        FixedStateTracker::register_dirty_flags(&mut installed);
        for (first, count, flag) in FixedStateTracker::methods() {
            for method in first..first + count {
                assert_eq!(
                    common.flags_for(method)[0],
                    0,
                    "method {method:#x} is taken"
                );
                assert_eq!(installed.flags_for(method)[0], flag);
            }
        }
    }
}
