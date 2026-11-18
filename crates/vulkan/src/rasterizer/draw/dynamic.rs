//! Records the dynamic state commands a draw needs. A command runs only when a register
//! group or another input it reads changed since the last draw, and its arguments differ from
//! the ones the open rendering already set.

use std::mem::{offset_of, size_of};

use ash::vk;
use lichii_common::settings::ResolutionScale;
use lichii_gpu::engines::maxwell_3d::regs::{
    Blend, BlendPerTarget, DepthMode, ProvokingVertex, ScissorTest, StencilOp, SurfaceClip,
    ViewportClipControlGeometryClip as GeometryClip, ViewportTransform, WindowOriginMode,
};
use lichii_gpu::engines::maxwell_3d::{DirtyState, Regs};

use super::depth_bias;
use crate::device::DeviceFeatures;
use crate::pipeline::PipelineFeatures;
use crate::pipeline::fixed_state::flags::NEXT_FREE;
use crate::pipeline::maxwell_to_vk as mvk;

macro_rules! commands {
    ($lt:lifetime; $($slot:ident($args:ty),)*) => {
        /// The dynamic state commands a draw may record, one shadow slot each.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(super) enum Slot {
            $($slot,)*
        }

        /// A dynamic state command with its arguments.
        pub(super) enum Command<$lt> {
            $($slot($args),)*
        }

        impl Command<'_> {
            #[inline(always)]
            pub(super) fn slot(&self) -> Slot {
                match self {
                    $(Self::$slot(_) => Slot::$slot,)*
                }
            }

            /// The arguments as the words a shadow slot compares.
            #[inline(always)]
            pub(super) fn words(&self, out: &mut Vec<u32>) {
                match self {
                    $(Self::$slot(args) => args.words(out),)*
                }
            }
        }

        const SLOTS: usize = [$(Slot::$slot),*].len();
    };
}

/// One stencil face: fail, pass and depth-fail operations, then the compare operation.
pub(super) type StencilFace = (vk::StencilOp, vk::StencilOp, vk::StencilOp, vk::CompareOp);

commands! { 'a;
    LineWidth(f32),
    DepthBias((f32, f32, f32, vk::DepthBiasRepresentationEXT)),
    BlendConstants([f32; 4]),
    DepthBounds((f32, f32)),
    StencilCompareMask((u32, u32)),
    StencilWriteMask((u32, u32)),
    StencilReference((u32, u32)),
    Viewport(vk::Viewport),
    Scissor(vk::Rect2D),
    CullMode(vk::CullModeFlags),
    FrontFace(vk::FrontFace),
    PrimitiveTopology(vk::PrimitiveTopology),
    DepthTestEnable(bool),
    DepthWriteEnable(bool),
    DepthCompareOp(vk::CompareOp),
    DepthBoundsTestEnable(bool),
    StencilTestEnable(bool),
    StencilOp((StencilFace, StencilFace)),
    RasterizerDiscardEnable(bool),
    DepthBiasEnable(bool),
    PrimitiveRestartEnable(bool),
    VertexBuffers((&'a [vk::Buffer], &'a [u64], &'a [u64])),
    VertexInput((
        &'a [vk::VertexInputBindingDescription2EXT<'static>],
        &'a [vk::VertexInputAttributeDescription2EXT<'static>],
    )),
    ColorBlendEnable(&'a [vk::Bool32]),
    ColorBlendEquation(&'a [vk::ColorBlendEquationEXT]),
    ColorWriteMask(&'a [vk::ColorComponentFlags]),
    DepthClampEnable(bool),
    LogicOpEnable(bool),
    AlphaToCoverageEnable(bool),
    AlphaToOneEnable(bool),
    PolygonMode(vk::PolygonMode),
    DepthClipNegativeOneToOne(bool),
    ProvokingVertexMode(vk::ProvokingVertexModeEXT),
}

/// Register groups, one dirty flag each from [`NEXT_FREE`] on, mapped through method table 1.
const LINE_WIDTH: u32 = 1 << 0;
const DEPTH_BIAS: u32 = 1 << 1;
const BLEND_CONSTANTS: u32 = 1 << 2;
const DEPTH_BOUNDS: u32 = 1 << 3;
const STENCIL: u32 = 1 << 4;
const VIEWPORT: u32 = 1 << 5;
const SCISSOR: u32 = 1 << 6;
const CULL: u32 = 1 << 7;
const RASTER: u32 = 1 << 8;
const DEPTH: u32 = 1 << 9;
const VERTEX_INPUT: u32 = 1 << 10;
const BLEND: u32 = 1 << 11;
const FLAGGED: u8 = 12;
/// The draw's topology, which no register holds.
const TOPOLOGY: u32 = 1 << FLAGGED;
const ALL: u32 = (TOPOLOGY << 1) - 1;

const fn word(offset: usize) -> u32 {
    (offset / 4) as u32
}

const fn words<T>() -> u32 {
    (size_of::<T>() / 4) as u32
}

const fn one(offset: usize, group: u32) -> (u32, u32, u32) {
    (word(offset), 1, group)
}

/// `(first method, count, group)` for every register the dynamic state reads.
const METHODS: [(u32, u32, u32); 45] = [
    one(offset_of!(Regs, line_width_aliased), LINE_WIDTH),
    one(offset_of!(Regs, depth_bias), DEPTH_BIAS),
    one(offset_of!(Regs, depth_bias_clamp), DEPTH_BIAS),
    one(offset_of!(Regs, slope_scale_depth_bias), DEPTH_BIAS),
    (word(offset_of!(Regs, blend_color)), 4, BLEND_CONSTANTS),
    (word(offset_of!(Regs, depth_bounds)), 2, DEPTH_BOUNDS),
    one(offset_of!(Regs, stencil_enable), STENCIL),
    one(offset_of!(Regs, stencil_two_side_enable), STENCIL),
    (
        word(offset_of!(Regs, stencil_front_op)),
        words::<StencilOp>(),
        STENCIL,
    ),
    (
        word(offset_of!(Regs, stencil_back_op)),
        words::<StencilOp>(),
        STENCIL,
    ),
    one(offset_of!(Regs, stencil_front_ref), STENCIL),
    one(offset_of!(Regs, stencil_front_func_mask), STENCIL),
    one(offset_of!(Regs, stencil_front_mask), STENCIL),
    one(offset_of!(Regs, stencil_back_ref), STENCIL),
    one(offset_of!(Regs, stencil_back_mask), STENCIL),
    one(offset_of!(Regs, stencil_back_func_mask), STENCIL),
    (
        word(offset_of!(Regs, viewport_transform)),
        words::<ViewportTransform>(),
        VIEWPORT,
    ),
    (
        word(offset_of!(Regs, surface_clip)),
        words::<SurfaceClip>(),
        VIEWPORT,
    ),
    one(offset_of!(Regs, window_origin), VIEWPORT),
    one(offset_of!(Regs, depth_mode), VIEWPORT),
    (
        word(offset_of!(Regs, scissor_test)),
        words::<ScissorTest>(),
        SCISSOR,
    ),
    one(offset_of!(Regs, gl_cull_test_enabled), CULL),
    one(offset_of!(Regs, gl_cull_face), CULL),
    one(offset_of!(Regs, gl_front_face), CULL),
    one(offset_of!(Regs, rasterize_enable), RASTER),
    one(offset_of!(Regs, polygon_offset_point_enable), RASTER),
    one(offset_of!(Regs, polygon_offset_line_enable), RASTER),
    one(offset_of!(Regs, polygon_offset_fill_enable), RASTER),
    one(offset_of!(Regs, primitive_restart.enabled), RASTER),
    one(offset_of!(Regs, viewport_clip_control), RASTER),
    one(offset_of!(Regs, logic_op.enable), RASTER),
    one(offset_of!(Regs, anti_alias_alpha_control), RASTER),
    one(offset_of!(Regs, polygon_mode_front), RASTER),
    one(offset_of!(Regs, provoking_vertex), RASTER),
    one(offset_of!(Regs, depth_test_enable), DEPTH),
    one(offset_of!(Regs, depth_write_enabled), DEPTH),
    one(offset_of!(Regs, depth_test_func), DEPTH),
    one(offset_of!(Regs, depth_bounds_enable), DEPTH),
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
    (word(offset_of!(Regs, blend)), words::<Blend>(), BLEND),
    (
        word(offset_of!(Regs, blend_per_target)),
        words::<[BlendPerTarget; 8]>(),
        BLEND,
    ),
    one(offset_of!(Regs, blend_per_target_enabled), BLEND),
    (word(offset_of!(Regs, color_mask)), 8, BLEND),
    one(offset_of!(Regs, color_mask_common), BLEND),
];

fn flag(group: u32) -> u8 {
    NEXT_FREE + group.trailing_zeros() as u8
}

/// What the dynamic state reads besides the registers.
pub(super) struct DynamicInputs<'a> {
    pub(super) features: PipelineFeatures,
    pub(super) device: &'a DeviceFeatures,
    pub(super) mesh: bool,
    pub(super) topology: vk::PrimitiveTopology,
    pub(super) extent: vk::Extent2D,
    pub(super) scale: ResolutionScale,
    pub(super) vertex_buffers: &'a [vk::Buffer],
    pub(super) vertex_offsets: &'a [u64],
    pub(super) vertex_strides: &'a [u64],
    /// The format of each colour target, `None` where none is bound.
    pub(super) colors: [Option<vk::Format>; 8],
    /// The colour attachments of the rendering.
    pub(super) attachments: usize,
    /// The components each fragment output writes, as RGBA bits.
    pub(super) outputs: [u8; 8],
}

/// The inputs besides the registers that the last tracked draw read, compared so a change
/// marks the groups that read it.
#[derive(Clone, Copy, PartialEq)]
struct InputKey {
    features: PipelineFeatures,
    scale: ResolutionScale,
    extent: vk::Extent2D,
    topology: vk::PrimitiveTopology,
    /// The depth format the depth bias scales with, whose register method table 1 already maps
    /// to the render targets.
    zeta_format: u32,
    colors: [Option<vk::Format>; 8],
    attachments: usize,
    outputs: [u8; 8],
    vertex_streams: usize,
}

impl InputKey {
    fn new(inputs: &DynamicInputs<'_>, regs: &Regs) -> Self {
        Self {
            features: inputs.features,
            scale: inputs.scale,
            extent: inputs.extent,
            topology: inputs.topology,
            zeta_format: regs.zeta.format,
            colors: inputs.colors,
            attachments: inputs.attachments,
            outputs: inputs.outputs,
            vertex_streams: inputs.vertex_strides.len(),
        }
    }

    /// The groups that read an input in which `self` and `other` differ.
    fn changed(&self, other: &Self) -> u32 {
        if self == other {
            return 0;
        }
        if self.features != other.features {
            return ALL;
        }
        let mut groups = 0;
        if self.scale != other.scale {
            groups |= VIEWPORT | SCISSOR;
        }
        if self.extent != other.extent {
            groups |= SCISSOR;
        }
        if self.topology != other.topology {
            groups |= TOPOLOGY;
        }
        if self.zeta_format != other.zeta_format {
            groups |= DEPTH_BIAS;
        }
        if (self.colors, self.attachments, self.outputs)
            != (other.colors, other.attachments, other.outputs)
        {
            groups |= BLEND;
        }
        if self.vertex_streams != other.vertex_streams {
            groups |= VERTEX_INPUT;
        }
        groups
    }
}

/// The arguments of the dynamic state commands recorded in the open rendering, so a draw
/// records only the commands whose arguments changed. Command buffer state outlives a
/// rendering, but other work may bind pipelines between renderings, so a new rendering starts
/// empty; so does a switch between mesh and vertex pipelines, which declare different dynamic
/// states.
struct Shadow {
    slots: [Vec<u32>; SLOTS],
    scratch: Vec<u32>,
}

impl Default for Shadow {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| Vec::new()),
            scratch: Vec::new(),
        }
    }
}

impl Shadow {
    /// Runs `record` unless the slot of `command` already holds its arguments; returns whether
    /// it ran. Inlined so each call folds the matches on its command.
    #[inline(always)]
    fn set(&mut self, command: Command<'_>, record: &mut impl FnMut(&Command<'_>)) -> bool {
        self.scratch.clear();
        command.words(&mut self.scratch);
        let held = &mut self.slots[command.slot() as usize];
        let same = held.len() == self.scratch.len()
            && held
                .iter()
                .zip(&self.scratch)
                .all(|(held, new)| held == new);
        if !held.is_empty() && same {
            return false;
        }
        held.clear();
        held.extend_from_slice(&self.scratch);
        record(&command);
        true
    }
}

/// Argument lists the draw builds, kept so their capacity outlives the draw.
#[derive(Default)]
struct Lists {
    bindings: Vec<vk::VertexInputBindingDescription2EXT<'static>>,
    attributes: Vec<vk::VertexInputAttributeDescription2EXT<'static>>,
    blend_enables: Vec<vk::Bool32>,
    blend_equations: Vec<vk::ColorBlendEquationEXT>,
    write_masks: Vec<vk::ColorComponentFlags>,
}

/// The dynamic state of the open rendering and what the last draw set it from.
#[derive(Default)]
pub(super) struct DynamicShadow {
    shadow: Shadow,
    lists: Lists,
    mesh: bool,
    /// The shadow forgot its slots since the last tracked draw, so every group is due.
    stale: bool,
    key: Option<InputKey>,
}

impl DynamicShadow {
    /// Maps every register the dynamic state reads to its group's flag in method table 1 and
    /// sets those flags, unless `dirty` already maps them.
    pub(super) fn register_dirty_flags(dirty: &mut DirtyState) {
        let [(first, _, group), ..] = METHODS;
        if dirty.flags_for(first)[1] == flag(group) {
            return;
        }
        for (first, count, group) in METHODS {
            dirty.fill(1, first, count, flag(group));
        }
        for group in 0..FLAGGED {
            dirty.set(NEXT_FREE + group);
        }
    }

    /// Forgets every slot when the draw began a rendering or switches between mesh and vertex
    /// pipelines.
    pub(super) fn begin(&mut self, began: bool, mesh: bool) {
        if began || mesh != self.mesh {
            self.shadow.slots.iter_mut().for_each(Vec::clear);
            self.mesh = mesh;
            self.stale = true;
        }
    }

    /// Records the commands of the groups whose registers `dirty` flags or whose other inputs
    /// changed since the last draw, then clears the groups' flags. Allocates nothing once its
    /// lists reached the draw's sizes.
    pub(super) fn apply(
        &mut self,
        inputs: &DynamicInputs<'_>,
        regs: &Regs,
        dirty: &mut DirtyState,
        blendable: impl Fn(vk::Format) -> bool,
        record: &mut impl FnMut(&Command<'_>),
    ) {
        let key = InputKey::new(inputs, regs);
        let mut pending = match self.key {
            Some(last) if !self.stale => last.changed(&key),
            _ => ALL,
        };
        self.key = Some(key);
        self.stale = false;
        for group in 0..FLAGGED {
            if dirty.is_set(NEXT_FREE + group) {
                pending |= 1 << group;
            }
        }
        self.emit(pending, inputs, regs, blendable, record);
        for group in 0..FLAGGED {
            dirty.clear(NEXT_FREE + group);
        }
    }

    /// Runs `record` for each command `features` declares that reads a group in `pending`
    /// and whose arguments the shadow does not hold.
    pub(super) fn emit(
        &mut self,
        pending: u32,
        inputs: &DynamicInputs<'_>,
        regs: &Regs,
        blendable: impl Fn(vk::Format) -> bool,
        record: &mut impl FnMut(&Command<'_>),
    ) {
        let due = |groups: u32| pending & groups != 0;
        let shadow = &mut self.shadow;
        let lists = &mut self.lists;
        let features = &inputs.features;
        let (mesh, topology) = (inputs.mesh, inputs.topology);
        let factor = {
            let (num, den) = inputs.scale.ratio();
            num as f32 / den as f32
        };
        if due(LINE_WIDTH) {
            let width = if inputs.device.wide_lines {
                regs.line_width_aliased.max(1.0)
            } else {
                1.0
            };
            shadow.set(Command::LineWidth(width), record);
        }
        if due(DEPTH_BIAS) {
            shadow.set(Command::DepthBias(depth_bias(regs, inputs.device)), record);
        }
        if due(BLEND_CONSTANTS) {
            let color = &regs.blend_color;
            let constants = [color.r, color.g, color.b, color.a];
            shadow.set(Command::BlendConstants(constants), record);
        }
        if due(DEPTH_BOUNDS) {
            let min = regs.depth_bounds[0].clamp(0.0, 1.0);
            let max = regs.depth_bounds[1].clamp(0.0, 1.0);
            shadow.set(Command::DepthBounds((min, max.max(min))), record);
        }
        let two_sided = regs.stencil_two_side_enable != 0;
        if due(STENCIL) {
            let (back_ref, back_write, back_func) = if two_sided {
                (
                    regs.stencil_back_ref,
                    regs.stencil_back_mask,
                    regs.stencil_back_func_mask,
                )
            } else {
                (
                    regs.stencil_front_ref,
                    regs.stencil_front_mask,
                    regs.stencil_front_func_mask,
                )
            };
            let compare = (regs.stencil_front_func_mask, back_func);
            shadow.set(Command::StencilCompareMask(compare), record);
            let write = (regs.stencil_front_mask, back_write);
            shadow.set(Command::StencilWriteMask(write), record);
            let reference = (regs.stencil_front_ref, back_ref);
            shadow.set(Command::StencilReference(reference), record);
        }
        if due(VIEWPORT) {
            let src = &regs.viewport_transform[0];
            let mut x = (src.translate_x - src.scale_x) * factor;
            let mut width = src.scale_x * 2.0 * factor;
            if width < 0.0 {
                x += width;
                width = -width;
            }
            let mut y = (src.translate_y - src.scale_y) * factor;
            let mut height = src.scale_y * 2.0 * factor;
            if regs.window_origin.mode() != WindowOriginMode::UPPER_LEFT {
                y += regs.surface_clip.bits1.height() as f32 * factor;
                height = -height;
            }
            let reduce_z = if regs.depth_mode == DepthMode::MINUS_ONE_TO_ONE {
                1.0
            } else {
                0.0
            };
            let viewport = vk::Viewport {
                x,
                y,
                width: if width == 0.0 { 1.0 } else { width },
                height: if height == 0.0 { 1.0 } else { height },
                min_depth: (src.translate_z - src.scale_z * reduce_z).clamp(0.0, 1.0),
                max_depth: (src.translate_z + src.scale_z).clamp(0.0, 1.0),
            };
            shadow.set(Command::Viewport(viewport), record);
        }
        if due(SCISSOR) {
            let scissor_regs = &regs.scissor_test[0];
            let scissor = if scissor_regs.enable != 0 {
                let s = |v: u32| (v as f32 * factor) as u32;
                let (x0, x1) = (s(scissor_regs.min_x()), s(scissor_regs.max_x()));
                let (y0, y1) = (s(scissor_regs.min_y()), s(scissor_regs.max_y()));
                vk::Rect2D {
                    offset: vk::Offset2D {
                        x: x0.min(i32::MAX as u32) as i32,
                        y: y0.min(i32::MAX as u32) as i32,
                    },
                    extent: vk::Extent2D {
                        width: x1.saturating_sub(x0),
                        height: y1.saturating_sub(y0),
                    },
                }
            } else {
                vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: inputs.extent,
                }
            };
            shadow.set(Command::Scissor(scissor), record);
        }
        if features.extended_dynamic_state {
            if due(CULL) {
                let cull = if regs.gl_cull_test_enabled != 0 {
                    mvk::cull_mode(regs.gl_cull_face).unwrap_or(vk::CullModeFlags::BACK)
                } else {
                    vk::CullModeFlags::NONE
                };
                shadow.set(Command::CullMode(cull), record);
            }
            if due(CULL | VIEWPORT) {
                let mut face =
                    mvk::front_face(regs.gl_front_face).unwrap_or(vk::FrontFace::COUNTER_CLOCKWISE);
                if regs.window_origin.flip_y() != 0 {
                    face = if face == vk::FrontFace::CLOCKWISE {
                        vk::FrontFace::COUNTER_CLOCKWISE
                    } else {
                        vk::FrontFace::CLOCKWISE
                    };
                }
                shadow.set(Command::FrontFace(face), record);
            }
            if !mesh && due(TOPOLOGY) {
                shadow.set(Command::PrimitiveTopology(topology), record);
            }
            if due(DEPTH) {
                let depth_test = regs.depth_test_enable != 0;
                shadow.set(Command::DepthTestEnable(depth_test), record);
                let depth_write = regs.depth_write_enabled != 0;
                shadow.set(Command::DepthWriteEnable(depth_write), record);
                let depth_op =
                    mvk::comparison_op(regs.depth_test_func).unwrap_or(vk::CompareOp::ALWAYS);
                shadow.set(Command::DepthCompareOp(depth_op), record);
                let bounds_test = features.depth_bounds && regs.depth_bounds_enable != 0;
                shadow.set(Command::DepthBoundsTestEnable(bounds_test), record);
            }
            if due(STENCIL) {
                let stencil_test = regs.stencil_enable != 0;
                shadow.set(Command::StencilTestEnable(stencil_test), record);
                let op = |o: &StencilOp| {
                    let s = |op| mvk::stencil_op(op).unwrap_or(vk::StencilOp::KEEP);
                    (
                        s(o.fail),
                        s(o.zpass),
                        s(o.zfail),
                        mvk::comparison_op(o.func).unwrap_or(vk::CompareOp::ALWAYS),
                    )
                };
                let front = op(&regs.stencil_front_op);
                let back = if two_sided {
                    op(&regs.stencil_back_op)
                } else {
                    front
                };
                shadow.set(Command::StencilOp((front, back)), record);
            }
        }
        if features.extended_dynamic_state2 {
            use vk::PrimitiveTopology as T;
            if due(RASTER) {
                let discard = regs.rasterize_enable == 0;
                shadow.set(Command::RasterizerDiscardEnable(discard), record);
            }
            if due(RASTER | TOPOLOGY) {
                let offset = match topology {
                    T::POINT_LIST => regs.polygon_offset_point_enable,
                    T::LINE_LIST
                    | T::LINE_STRIP
                    | T::LINE_LIST_WITH_ADJACENCY
                    | T::LINE_STRIP_WITH_ADJACENCY => regs.polygon_offset_line_enable,
                    _ => regs.polygon_offset_fill_enable,
                } != 0;
                shadow.set(Command::DepthBiasEnable(offset), record);
            }
            if !mesh && due(RASTER | TOPOLOGY) {
                let restartable = matches!(
                    topology,
                    T::LINE_STRIP
                        | T::TRIANGLE_STRIP
                        | T::TRIANGLE_FAN
                        | T::LINE_STRIP_WITH_ADJACENCY
                        | T::TRIANGLE_STRIP_WITH_ADJACENCY
                );
                let restart = restartable && regs.primitive_restart.enabled != 0;
                shadow.set(Command::PrimitiveRestartEnable(restart), record);
            }
        }
        let buffers_changed = !inputs.vertex_buffers.is_empty()
            && shadow.set(
                Command::VertexBuffers((
                    inputs.vertex_buffers,
                    inputs.vertex_offsets,
                    inputs.vertex_strides,
                )),
                record,
            );
        if features.dynamic_vertex_input && !mesh && (due(VERTEX_INPUT) || buffers_changed) {
            lists.bindings.clear();
            lists.attributes.clear();
            for (location, a) in regs.vertex_attrib_format.iter().enumerate() {
                let Some(format) = (a.constant() == 0)
                    .then(|| mvk::vertex_format(a.type_(), a.size()))
                    .flatten()
                else {
                    continue;
                };
                lists.attributes.push(
                    vk::VertexInputAttributeDescription2EXT::default()
                        .location(location as u32)
                        .binding(a.buffer())
                        .format(format)
                        .offset(a.offset()),
                );
            }
            for (binding, &stride) in inputs.vertex_strides.iter().enumerate() {
                let instanced = regs.vertex_stream_instances.is_instanced[binding] != 0;
                lists.bindings.push(
                    vk::VertexInputBindingDescription2EXT::default()
                        .binding(binding as u32)
                        .stride(stride as u32)
                        .input_rate(if instanced {
                            vk::VertexInputRate::INSTANCE
                        } else {
                            vk::VertexInputRate::VERTEX
                        })
                        .divisor(1),
                );
            }
            let input = (lists.bindings.as_slice(), lists.attributes.as_slice());
            shadow.set(Command::VertexInput(input), record);
        }
        if features.dynamic_blend && inputs.attachments > 0 && due(BLEND) {
            lists.blend_enables.clear();
            lists.blend_equations.clear();
            lists.write_masks.clear();
            let common = regs.color_mask_common != 0;
            for index in 0..inputs.attachments {
                let mask = regs.color_mask[if common { 0 } else { index }];
                lists.write_masks.push(vk::ColorComponentFlags::from_raw(
                    (mask.r() | mask.g() << 1 | mask.b() << 2 | mask.a() << 3)
                        & u32::from(inputs.outputs[index]),
                ));
                let b = &regs.blend;
                let t = &regs.blend_per_target[index];
                let per_target = regs.blend_per_target_enabled != 0;
                let (color_op, src_color, dst_color, alpha_op, src_alpha, dst_alpha) = if per_target
                {
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
                let factor = |f| {
                    let host = mvk::blend_factor(f).unwrap_or(vk::BlendFactor::ONE);
                    if features.dual_src_blend {
                        host
                    } else {
                        mvk::single_source(host)
                    }
                };
                let op = |o| mvk::blend_equation(o).unwrap_or(vk::BlendOp::ADD);
                let blends = inputs.colors[index].is_some_and(&blendable);
                lists
                    .blend_enables
                    .push(vk::Bool32::from(regs.blend.enable[index] != 0 && blends));
                lists.blend_equations.push(vk::ColorBlendEquationEXT {
                    src_color_blend_factor: factor(src_color),
                    dst_color_blend_factor: factor(dst_color),
                    color_blend_op: op(color_op),
                    src_alpha_blend_factor: factor(src_alpha),
                    dst_alpha_blend_factor: factor(dst_alpha),
                    alpha_blend_op: op(alpha_op),
                });
            }
            shadow.set(Command::ColorBlendEnable(&lists.blend_enables), record);
            shadow.set(Command::ColorBlendEquation(&lists.blend_equations), record);
            shadow.set(Command::ColorWriteMask(&lists.write_masks), record);
        }
        let alpha = regs.anti_alias_alpha_control;
        if features.dynamic_depth_clamp && due(RASTER) {
            let clamps = !matches!(
                regs.viewport_clip_control.geometry_clip(),
                GeometryClip::PASSTHROUGH | GeometryClip::FRUSTUM_XYZ | GeometryClip::FRUSTUM_Z
            );
            let clamp = features.depth_clamp && clamps;
            shadow.set(Command::DepthClampEnable(clamp), record);
        }
        if features.dynamic_logic_op_enable && due(RASTER) {
            let logic = features.logic_op && regs.logic_op.enable != 0;
            shadow.set(Command::LogicOpEnable(logic), record);
        }
        if features.dynamic_alpha_to_coverage && due(RASTER) {
            let coverage = alpha.alpha_to_coverage() != 0;
            shadow.set(Command::AlphaToCoverageEnable(coverage), record);
        }
        if features.dynamic_alpha_to_one && due(RASTER) {
            let one = features.alpha_to_one && alpha.alpha_to_one() != 0;
            shadow.set(Command::AlphaToOneEnable(one), record);
        }
        if features.dynamic_polygon_mode && due(RASTER) {
            let mode = if features.fill_mode_non_solid {
                mvk::polygon_mode(regs.polygon_mode_front).unwrap_or(vk::PolygonMode::FILL)
            } else {
                vk::PolygonMode::FILL
            };
            shadow.set(Command::PolygonMode(mode), record);
        }
        if features.dynamic_depth_clip_negative_one_to_one && due(VIEWPORT) {
            let negative = regs.depth_mode == DepthMode::MINUS_ONE_TO_ONE;
            shadow.set(Command::DepthClipNegativeOneToOne(negative), record);
        }
        if features.dynamic_provoking_vertex && due(RASTER) {
            let provoking = if regs.provoking_vertex == ProvokingVertex::LAST
                && features.provoking_vertex_last
            {
                vk::ProvokingVertexModeEXT::LAST_VERTEX
            } else {
                vk::ProvokingVertexModeEXT::FIRST_VERTEX
            };
            shadow.set(Command::ProvokingVertexMode(provoking), record);
        }
    }
}

/// Command arguments as the words a shadow slot compares.
pub(super) trait Words {
    fn words(&self, out: &mut Vec<u32>);
}

impl Words for u32 {
    fn words(&self, out: &mut Vec<u32>) {
        out.push(*self);
    }
}

impl Words for u64 {
    fn words(&self, out: &mut Vec<u32>) {
        out.extend([*self as u32, (*self >> 32) as u32]);
    }
}

impl Words for f32 {
    fn words(&self, out: &mut Vec<u32>) {
        out.push(self.to_bits());
    }
}

impl Words for bool {
    fn words(&self, out: &mut Vec<u32>) {
        out.push(u32::from(*self));
    }
}

macro_rules! raw_words {
    ($($ty:ty),*) => {$(
        impl Words for $ty {
            fn words(&self, out: &mut Vec<u32>) {
                out.push(self.as_raw() as u32);
            }
        }
    )*};
}

raw_words!(
    vk::PrimitiveTopology,
    vk::FrontFace,
    vk::CompareOp,
    vk::StencilOp,
    vk::PolygonMode,
    vk::ProvokingVertexModeEXT,
    vk::DepthBiasRepresentationEXT,
    vk::BlendFactor,
    vk::BlendOp,
    vk::VertexInputRate,
    vk::Format,
    vk::CullModeFlags,
    vk::StencilFaceFlags,
    vk::ColorComponentFlags
);

impl Words for vk::Buffer {
    fn words(&self, out: &mut Vec<u32>) {
        ash::vk::Handle::as_raw(*self).words(out);
    }
}

impl Words for vk::Viewport {
    fn words(&self, out: &mut Vec<u32>) {
        (
            self.x,
            self.y,
            self.width,
            self.height,
            self.min_depth,
            self.max_depth,
        )
            .words(out);
    }
}

impl Words for vk::Rect2D {
    fn words(&self, out: &mut Vec<u32>) {
        let (o, e) = (self.offset, self.extent);
        (o.x as u32, o.y as u32, e.width, e.height).words(out);
    }
}

impl Words for vk::ColorBlendEquationEXT {
    fn words(&self, out: &mut Vec<u32>) {
        (
            self.src_color_blend_factor,
            self.dst_color_blend_factor,
            self.color_blend_op,
            self.src_alpha_blend_factor,
            self.dst_alpha_blend_factor,
            self.alpha_blend_op,
        )
            .words(out);
    }
}

impl Words for vk::VertexInputBindingDescription2EXT<'_> {
    fn words(&self, out: &mut Vec<u32>) {
        (self.binding, self.stride, self.input_rate, self.divisor).words(out);
    }
}

impl Words for vk::VertexInputAttributeDescription2EXT<'_> {
    fn words(&self, out: &mut Vec<u32>) {
        (self.location, self.binding, self.format, self.offset).words(out);
    }
}

impl<T: Words> Words for &[T] {
    fn words(&self, out: &mut Vec<u32>) {
        out.push(self.len() as u32);
        self.iter().for_each(|t| t.words(out));
    }
}

impl<T: Words, const N: usize> Words for [T; N] {
    fn words(&self, out: &mut Vec<u32>) {
        self.iter().for_each(|t| t.words(out));
    }
}

macro_rules! tuple_words {
    ($($name:ident),+) => {
        impl<$($name: Words),+> Words for ($($name,)+) {
            #[allow(non_snake_case)]
            fn words(&self, out: &mut Vec<u32>) {
                let ($($name,)+) = self;
                $($name.words(out);)+
            }
        }
    };
}

tuple_words!(A, B);
tuple_words!(A, B, C);
tuple_words!(A, B, C, D);
tuple_words!(A, B, C, D, E);
tuple_words!(A, B, C, D, E, F);

#[cfg(test)]
mod tests {
    use lichii_gpu::engines::maxwell_3d::RegisterState;
    use lichii_gpu::engines::maxwell_3d::dirty::NULL;
    use lichii_gpu::engines::maxwell_3d::regs::{
        BlendEquation, BlendFactor, ComparisonOp, CullFace, FrontFace, NUM_REGS, PolygonMode,
        StencilOpOp,
    };

    use super::*;

    #[test]
    fn slots_record_only_changed_arguments_until_a_rendering_begins() {
        let mut shadow = DynamicShadow::default();
        let mut recorded = 0;
        let mut set = |shadow: &mut DynamicShadow, command| {
            shadow
                .shadow
                .set(command, &mut |_: &Command<'_>| recorded += 1);
        };
        set(&mut shadow, Command::LineWidth(1.0));
        set(&mut shadow, Command::LineWidth(1.0));
        set(&mut shadow, Command::DepthBounds((1.0, 1.0)));
        set(&mut shadow, Command::LineWidth(2.0));
        shadow.begin(false, false);
        set(&mut shadow, Command::LineWidth(2.0));
        shadow.begin(true, false);
        set(&mut shadow, Command::LineWidth(2.0));
        shadow.begin(false, true);
        set(&mut shadow, Command::LineWidth(2.0));
        drop(set);
        assert_eq!(recorded, 5);
    }

    #[test]
    fn groups_take_only_methods_table_1_leaves_free() {
        let dirty = DirtyState::new();
        let mut claimed = vec![false; NUM_REGS];
        for (first, count, group) in METHODS {
            assert!(group.is_power_of_two() && group < TOPOLOGY);
            for method in first..first + count {
                assert_eq!(dirty.flags_for(method)[1], NULL, "method {method:#x}");
                assert!(!std::mem::replace(&mut claimed[method as usize], true));
            }
        }
    }

    /// Every register the dynamic state reads, listed apart from [`METHODS`] so a register the
    /// groups miss still changes in the equivalence test.
    const READS: [(usize, usize); 54] = [
        (offset_of!(Regs, line_width_aliased), 1),
        (offset_of!(Regs, depth_bias), 1),
        (offset_of!(Regs, depth_bias_clamp), 1),
        (offset_of!(Regs, slope_scale_depth_bias), 1),
        (offset_of!(Regs, zeta.format), 1),
        (offset_of!(Regs, blend_color), 4),
        (offset_of!(Regs, depth_bounds), 2),
        (offset_of!(Regs, stencil_two_side_enable), 1),
        (offset_of!(Regs, stencil_front_ref), 1),
        (offset_of!(Regs, stencil_front_mask), 1),
        (offset_of!(Regs, stencil_front_func_mask), 1),
        (offset_of!(Regs, stencil_back_ref), 1),
        (offset_of!(Regs, stencil_back_mask), 1),
        (offset_of!(Regs, stencil_back_func_mask), 1),
        (offset_of!(Regs, stencil_enable), 1),
        (offset_of!(Regs, stencil_front_op), 4),
        (offset_of!(Regs, stencil_back_op), 4),
        (offset_of!(Regs, viewport_transform), 16),
        (offset_of!(Regs, window_origin), 1),
        (offset_of!(Regs, surface_clip), 2),
        (offset_of!(Regs, depth_mode), 1),
        (offset_of!(Regs, scissor_test), 8),
        (offset_of!(Regs, gl_cull_test_enabled), 1),
        (offset_of!(Regs, gl_cull_face), 1),
        (offset_of!(Regs, gl_front_face), 1),
        (offset_of!(Regs, depth_test_enable), 1),
        (offset_of!(Regs, depth_write_enabled), 1),
        (offset_of!(Regs, depth_test_func), 1),
        (offset_of!(Regs, depth_bounds_enable), 1),
        (offset_of!(Regs, rasterize_enable), 1),
        (offset_of!(Regs, polygon_offset_point_enable), 1),
        (offset_of!(Regs, polygon_offset_line_enable), 1),
        (offset_of!(Regs, polygon_offset_fill_enable), 1),
        (offset_of!(Regs, primitive_restart), 2),
        (offset_of!(Regs, vertex_attrib_format), 4),
        (offset_of!(Regs, vertex_attrib_format) + 4 * 28, 4),
        (offset_of!(Regs, vertex_stream_instances), 4),
        (offset_of!(Regs, vertex_stream_instances) + 4 * 28, 4),
        (offset_of!(Regs, color_mask_common), 1),
        (offset_of!(Regs, color_mask), 8),
        (offset_of!(Regs, blend), 9),
        (offset_of!(Regs, blend.enable), 8),
        (offset_of!(Regs, blend_per_target), 8),
        (offset_of!(Regs, blend_per_target) + 4 * 8 * 7, 8),
        (offset_of!(Regs, blend_per_target_enabled), 1),
        (offset_of!(Regs, viewport_clip_control), 1),
        (offset_of!(Regs, logic_op), 2),
        (offset_of!(Regs, anti_alias_alpha_control), 1),
        (offset_of!(Regs, polygon_mode_front), 1),
        (offset_of!(Regs, provoking_vertex), 1),
        (offset_of!(Regs, rt), 4),
        (offset_of!(Regs, zeta), 2),
        (offset_of!(Regs, vertex_streams), 8),
        (offset_of!(Regs, rt_control), 1),
    ];

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn chance(&mut self, percent: u64) -> bool {
            self.next() % 100 < percent
        }

        fn pick<T: Copy>(&mut self, values: &[T]) -> T {
            values[self.below(values.len())]
        }
    }

    fn value(rng: &mut Rng, current: u32) -> u32 {
        let enums = [
            ComparisonOp::NEVER_GL.0,
            ComparisonOp::LESS_D3D.0,
            ComparisonOp::GREATER_EQUAL_GL.0,
            ComparisonOp::ALWAYS_GL.0,
            StencilOpOp::KEEP_GL.0,
            StencilOpOp::REPLACE_D3D.0,
            StencilOpOp::INVERT_GL.0,
            BlendEquation::ADD_GL.0,
            BlendEquation::SUBTRACT_D3D.0,
            BlendEquation::MAX_GL.0,
            BlendFactor::ZERO_GL.0,
            BlendFactor::ONE_D3D.0,
            BlendFactor::SOURCE_ALPHA_GL.0,
            BlendFactor::ONE_MINUS_SOURCE_ALPHA_D3D.0,
            CullFace::FRONT.0,
            CullFace::BACK.0,
            FrontFace::CLOCK_WISE.0,
            FrontFace::COUNTER_CLOCK_WISE.0,
            PolygonMode::POINT.0,
            PolygonMode::LINE.0,
            PolygonMode::FILL.0,
            0x0a,
            0x13,
            0x14,
            0x16,
            0x4 << 21 | 0x7 << 27,
            16 << 7 | 0x12 << 21 | 0x7 << 27 | 1,
            0x1 << 21 | 0x7 << 27 | 2,
            1280 << 16,
            720 << 16 | 16,
        ];
        let floats = [0.0f32, 0.5, 1.0, 2.0, -1.0, 360.0, 640.0, -320.0, 1e9];
        match rng.below(10) {
            0 | 1 => current,
            2..=4 => rng.below(4) as u32,
            5 | 6 => rng.pick(&enums),
            7 | 8 => rng.pick(&floats).to_bits(),
            _ => rng.next() as u32,
        }
    }

    fn all_features() -> PipelineFeatures {
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
            depth_clamp: true,
            depth_bounds: true,
            logic_op: true,
            alpha_to_one: true,
            fill_mode_non_solid: true,
            dual_src_blend: true,
            depth_clip_control: true,
            provoking_vertex_last: true,
            multi_viewport: true,
            fast_link: false,
        }
    }

    /// Features with each switch of `all_features` off one time in four.
    fn some_features(rng: &mut Rng) -> PipelineFeatures {
        let mut f = all_features();
        for switch in [
            &mut f.extended_dynamic_state,
            &mut f.extended_dynamic_state2,
            &mut f.dynamic_vertex_input,
            &mut f.dynamic_blend,
            &mut f.dynamic_depth_clamp,
            &mut f.dynamic_logic_op_enable,
            &mut f.dynamic_alpha_to_coverage,
            &mut f.dynamic_alpha_to_one,
            &mut f.dynamic_polygon_mode,
            &mut f.dynamic_depth_clip_negative_one_to_one,
            &mut f.dynamic_provoking_vertex,
            &mut f.depth_clamp,
            &mut f.depth_bounds,
            &mut f.logic_op,
            &mut f.alpha_to_one,
            &mut f.fill_mode_non_solid,
            &mut f.dual_src_blend,
            &mut f.provoking_vertex_last,
        ] {
            *switch = !rng.chance(25);
        }
        f
    }

    fn log(out: &mut Vec<(Slot, Vec<u32>)>, command: &Command<'_>) {
        let mut words = Vec::new();
        command.words(&mut words);
        out.push((command.slot(), words));
    }

    /// Draws the same random stream through the tracked path and through one that rebuilds
    /// every group, and returns how often each slot recorded. Panics with the round where the
    /// recorded commands differ.
    fn tracked_matches_full_rebuild(seed: u64, rounds: usize) -> [usize; SLOTS] {
        let mut rng = Rng(seed);
        let mut state = RegisterState::new();
        DynamicShadow::register_dirty_flags(&mut state.dirty);
        let mut tracked = DynamicShadow::default();
        let mut reference = DynamicShadow::default();
        let device = DeviceFeatures {
            wide_lines: true,
            depth_bias_float: true,
            ..DeviceFeatures::default()
        };
        let topologies = [
            vk::PrimitiveTopology::TRIANGLE_LIST,
            vk::PrimitiveTopology::TRIANGLE_STRIP,
            vk::PrimitiveTopology::LINE_STRIP,
            vk::PrimitiveTopology::POINT_LIST,
            vk::PrimitiveTopology::TRIANGLE_FAN,
        ];
        let scales = [
            ResolutionScale::X1,
            ResolutionScale::X1_2,
            ResolutionScale::X3_4,
            ResolutionScale::X2,
        ];
        let formats = [
            None,
            Some(vk::Format::R8G8B8A8_UNORM),
            Some(vk::Format::R16G16B16A16_SFLOAT),
            Some(vk::Format::R32_UINT),
        ];
        let blendable = |format| format != vk::Format::R32_UINT;
        let mut features = all_features();
        let mut mesh = false;
        let mut topology = topologies[0];
        let mut scale = ResolutionScale::X1;
        let mut extent = vk::Extent2D {
            width: 1280,
            height: 720,
        };
        let mut buffers: Vec<vk::Buffer> = Vec::new();
        let mut offsets: Vec<u64> = Vec::new();
        let mut strides: Vec<u64> = Vec::new();
        let mut colors = [None; 8];
        let mut attachments = 1;
        let mut outputs = [0xf; 8];
        let mut counts = [0; SLOTS];
        let (mut got, mut want) = (Vec::new(), Vec::new());
        for round in 0..rounds {
            for _ in 0..rng.below(4) {
                let method = if rng.chance(90) {
                    let (offset, count) = rng.pick(&READS);
                    (offset / 4 + rng.below(count)) as u32
                } else {
                    rng.below(NUM_REGS) as u32
                };
                if state.is_executable(method) {
                    continue;
                }
                let current = state.regs.words()[method as usize];
                state.write(method, value(&mut rng, current));
            }
            if rng.chance(2) {
                features = if rng.chance(50) {
                    all_features()
                } else {
                    some_features(&mut rng)
                };
            }
            if rng.chance(3) {
                mesh = !mesh;
            }
            if rng.chance(10) {
                topology = rng.pick(&topologies);
            }
            if rng.chance(3) {
                scale = rng.pick(&scales);
            }
            if rng.chance(3) {
                extent.width = rng.pick(&[640, 1280, 1920]);
            }
            if rng.chance(10) {
                let streams = rng.below(4);
                buffers.clear();
                offsets.clear();
                strides.clear();
                for _ in 0..streams {
                    buffers.push(ash::vk::Handle::from_raw(1 + rng.below(3) as u64));
                    offsets.push(rng.pick(&[0, 64]));
                    strides.push(rng.pick(&[0, 12, 16, 24]));
                }
            }
            if rng.chance(5) {
                attachments = rng.below(9);
                colors = std::array::from_fn(|_| rng.pick(&formats));
                outputs = std::array::from_fn(|_| rng.below(16) as u8);
            }
            if rng.chance(1) {
                state.dirty.set_all();
            }
            if rng.chance(5) {
                continue;
            }
            let began = rng.chance(8);
            tracked.begin(began, mesh);
            reference.begin(began, mesh);
            let inputs = DynamicInputs {
                features,
                device: &device,
                mesh,
                topology,
                extent,
                scale,
                vertex_buffers: &buffers,
                vertex_offsets: &offsets,
                vertex_strides: &strides,
                colors,
                attachments,
                outputs,
            };
            got.clear();
            want.clear();
            tracked.apply(
                &inputs,
                &state.regs,
                &mut state.dirty,
                blendable,
                &mut |c| log(&mut got, c),
            );
            reference.emit(ALL, &inputs, &state.regs, blendable, &mut |c| {
                log(&mut want, c)
            });
            assert_eq!(got, want, "seed {seed:#x}, round {round}");
            for (slot, _) in &want {
                counts[*slot as usize] += 1;
            }
        }
        counts
    }

    #[test]
    fn tracked_draws_record_what_a_full_rebuild_records() {
        let mut counts = [0; SLOTS];
        for seed in 1..=24u64 {
            let seen = tracked_matches_full_rebuild(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15), 4000);
            for (count, seen) in counts.iter_mut().zip(seen) {
                *count += seen;
            }
        }
        assert!(
            counts.iter().all(|&count| count >= 50),
            "every slot records often: {counts:?}"
        );
    }

    /// The stream of `measures_dynamic_state_cost` without a device: three vertex streams, two
    /// colour attachments, a new rendering every 40 draws and 0 to 3 toggled registers per
    /// draw, through the tracked path and through a full rebuild of every group, which is what
    /// each draw did before the groups had dirty flags.
    #[test]
    #[ignore = "benchmark"]
    fn measures_tracked_and_rebuilt_dynamic_state_cost() {
        use std::time::{Duration, Instant};

        let mut state = RegisterState::new();
        DynamicShadow::register_dirty_flags(&mut state.dirty);
        let word = |offset: usize| (offset / 4) as u32;
        for (i, attribute) in [0x12 << 21, 0x12 << 21 | 12 << 7, 0x9 << 21 | 1]
            .into_iter()
            .enumerate()
        {
            let method = word(offset_of!(Regs, vertex_attrib_format)) + i as u32;
            state.write(method, attribute | 0x7 << 27);
        }
        let buffers = [ash::vk::Handle::from_raw(1); 3];
        let device = DeviceFeatures::default();
        let toggles: [(u32, [u32; 2]); 13] = [
            (word(offset_of!(Regs, depth_write_enabled)), [0, 1]),
            (word(offset_of!(Regs, depth_test_enable)), [1, 0]),
            (word(offset_of!(Regs, depth_test_func)), [0x203, 0x207]),
            (word(offset_of!(Regs, blend.enable)), [0, 1]),
            (word(offset_of!(Regs, blend.color_source)), [0x4302, 0x4304]),
            (word(offset_of!(Regs, color_mask)), [0x1111, 0x0111]),
            (word(offset_of!(Regs, gl_cull_test_enabled)), [1, 0]),
            (word(offset_of!(Regs, polygon_offset_fill_enable)), [0, 1]),
            (word(offset_of!(Regs, depth_bias)), [0, 4f32.to_bits()]),
            (word(offset_of!(Regs, stencil_front_ref)), [0, 0x80]),
            (word(offset_of!(Regs, blend_color.r)), [0, 1f32.to_bits()]),
            (
                word(offset_of!(Regs, scissor_test)) + 1,
                [1280 << 16, 640 << 16],
            ),
            (
                word(offset_of!(Regs, viewport_transform)) + 3,
                [640f32.to_bits(), 320f32.to_bits()],
            ),
        ];
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        const DRAWS: usize = 4096;
        let stream: Vec<Vec<(u32, u32)>> = (0..DRAWS)
            .map(|_| {
                let changes = match rng.below(20) {
                    0..10 => 0,
                    10..15 => 1,
                    15..18 => 2,
                    _ => 3,
                };
                (0..changes)
                    .map(|_| {
                        let (method, values) = rng.pick(&toggles);
                        (method, values[rng.below(2)])
                    })
                    .collect()
            })
            .collect();
        let inputs = DynamicInputs {
            features: all_features(),
            device: &device,
            mesh: false,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            extent: vk::Extent2D {
                width: 1280,
                height: 720,
            },
            scale: ResolutionScale::X1,
            vertex_buffers: &buffers,
            vertex_offsets: &[0; 3],
            vertex_strides: &[24, 24, 8],
            colors: [Some(vk::Format::R8G8B8A8_UNORM); 8],
            attachments: 2,
            outputs: [0xf; 8],
        };
        let blendable = |_| true;
        let mut run = |tracked: bool| {
            let mut shadow = DynamicShadow::default();
            let mut recorded = 0usize;
            let mut elapsed = Duration::ZERO;
            let rounds = 400;
            for round in 0..=rounds {
                let start = Instant::now();
                for (i, writes) in stream.iter().enumerate() {
                    for &(method, value) in writes {
                        state.write(method, value);
                    }
                    shadow.begin(i % 40 == 0, false);
                    let record = &mut |_: &Command<'_>| recorded += 1;
                    if tracked {
                        shadow.apply(&inputs, &state.regs, &mut state.dirty, blendable, record);
                    } else {
                        shadow.emit(ALL, &inputs, &state.regs, blendable, record);
                    }
                }
                if round > 0 {
                    elapsed += start.elapsed();
                }
            }
            let draws = (rounds * DRAWS) as f64;
            (elapsed.as_secs_f64() * 1e9 / draws, recorded as f64 / draws)
        };
        let (rebuilt, rebuilt_commands) = run(false);
        let (tracked, tracked_commands) = run(true);
        eprintln!(
            "dynamic state per draw: rebuilt {rebuilt:.1} ns, tracked {tracked:.1} ns \
             ({rebuilt_commands:.2} and {tracked_commands:.2} commands)"
        );
    }
}
