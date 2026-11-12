//! The 3D engine's register file: current values, shadow RAM, dirty flags and the set of
//! methods that do more than store a value.
//!
//! [`RegisterState::write`] is the one write path. Both the back thread (which executes methods)
//! and the front thread's register mirror (which macros read, NX2_GPU_THREAD_SPLIT.md) run it, so
//! the two cannot disagree about shadow RAM or same-value writes. A write of the value a state
//! register already holds (about 60% of all 3D writes, NX2_NOTES row 135) costs one shadow RAM
//! step, a load and a compare.

use super::regs::{
    BlendEquation, BlendFactor, ComparisonOp, CullFace, FrontFace, NUM_REGS, PolygonMode, Regs,
    ShadowRamControl, StencilOpOp, ViewportSwizzle,
};
use crate::reg_index;

/// First method number that calls a macro instead of writing a register.
pub const MACRO_REGISTERS_START: u32 = NUM_REGS as u32;

/// NVN2 root tables (Ampere): selects one of eight 64-word tables and resets its write offset.
pub const ROOT_TABLE_SELECT: u32 = 0x504 / 4;
/// NVN2 root tables: stores one word at the selected table's offset and advances it.
pub const ROOT_TABLE_LOAD: u32 = 0x508 / 4;

impl Regs {
    /// The register values games expect at boot.
    pub fn with_defaults() -> Box<Self> {
        let mut regs: Box<Self> = bytemuck::zeroed_box();
        for viewport in &mut regs.viewports {
            viewport.depth_range_near = 0.0;
            viewport.depth_range_far = 1.0;
        }
        let swizzle = ViewportSwizzle::POSITIVE_X.0
            | ViewportSwizzle::POSITIVE_Y.0 << 4
            | ViewportSwizzle::POSITIVE_Z.0 << 8
            | ViewportSwizzle::POSITIVE_W.0 << 12;
        for transform in &mut regs.viewport_transform {
            transform.swizzle.0 = swizzle;
        }
        regs.blend.color_op = BlendEquation::ADD_D3D;
        regs.blend.color_source = BlendFactor::ONE_D3D;
        regs.blend.color_dest = BlendFactor::ZERO_D3D;
        regs.blend.alpha_op = BlendEquation::ADD_D3D;
        regs.blend.alpha_source = BlendFactor::ONE_D3D;
        regs.blend.alpha_dest = BlendFactor::ZERO_D3D;
        for blend in &mut regs.blend_per_target {
            blend.color_op = BlendEquation::ADD_D3D;
            blend.color_source = BlendFactor::ONE_D3D;
            blend.color_dest = BlendFactor::ZERO_D3D;
            blend.alpha_op = BlendEquation::ADD_D3D;
            blend.alpha_source = BlendFactor::ONE_D3D;
            blend.alpha_dest = BlendFactor::ZERO_D3D;
        }
        for op in [&mut regs.stencil_front_op, &mut regs.stencil_back_op] {
            op.fail = StencilOpOp::KEEP_D3D;
            op.zfail = StencilOpOp::KEEP_D3D;
            op.zpass = StencilOpOp::KEEP_D3D;
            op.func = ComparisonOp::ALWAYS_GL;
        }
        regs.stencil_front_func_mask = u32::MAX;
        regs.stencil_front_mask = u32::MAX;
        regs.stencil_back_func_mask = u32::MAX;
        regs.stencil_back_mask = u32::MAX;
        regs.stencil_two_side_enable = 1;
        regs.depth_test_func = ComparisonOp::ALWAYS_GL;
        regs.gl_front_face = FrontFace::CLOCK_WISE;
        regs.gl_cull_face = CullFace::BACK;
        regs.point_size = 1.0;
        for mask in &mut regs.color_mask {
            mask.0 = 0x1111;
        }
        for format in &mut regs.vertex_attrib_format {
            *format = format.with_bits(6, 1, 1);
        }
        regs.rasterize_enable = 1;
        regs.color_target_mrt_enable = 1;
        regs.framebuffer_srgb = 1;
        regs.line_width_aliased = 1.0;
        regs.line_width_smooth = 1.0;
        regs.polygon_mode_back = PolygonMode::FILL;
        regs.polygon_mode_front = PolygonMode::FILL;
        regs
    }

    /// The registers as the method array games write.
    #[inline]
    pub fn words(&self) -> &[u32; NUM_REGS] {
        bytemuck::cast_ref(self)
    }

    #[inline]
    pub fn words_mut(&mut self) -> &mut [u32; NUM_REGS] {
        bytemuck::cast_mut(self)
    }
}

/// Dirty flag indices shared by the GPU and the renderer. Index 0 is the null flag: methods
/// that map to it mark nothing. The renderer adds its own flags from [`dirty::LAST_COMMON`] on
/// with [`DirtyState::fill`].
pub mod dirty {
    pub const NULL: u8 = 0;
    pub const DESCRIPTORS: u8 = 1;
    pub const RENDER_TARGETS: u8 = 2;
    pub const RENDER_TARGET_CONTROL: u8 = 3;
    pub const COLOR_BUFFER0: u8 = 4;
    pub const ZETA_BUFFER: u8 = COLOR_BUFFER0 + 8;
    pub const RESCALE_VIEWPORTS: u8 = ZETA_BUFFER + 1;
    pub const RESCALE_SCISSORS: u8 = RESCALE_VIEWPORTS + 1;
    pub const VERTEX_BUFFERS: u8 = RESCALE_SCISSORS + 1;
    pub const VERTEX_BUFFER0: u8 = VERTEX_BUFFERS + 1;
    pub const INDEX_BUFFER: u8 = VERTEX_BUFFER0 + 32;
    pub const SHADERS: u8 = INDEX_BUFFER + 1;
    pub const DEPTH_BIAS_GLOBAL: u8 = SHADERS + 1;
    /// First flag index free for the renderer.
    pub const LAST_COMMON: u8 = DEPTH_BIAS_GLOBAL + 1;
}

/// 256 dirty flags and the two method-to-flag tables that set them.
pub struct DirtyState {
    flags: [u64; 4],
    tables: Box<[[u8; NUM_REGS]; 2]>,
}

impl Default for DirtyState {
    fn default() -> Self {
        Self::new()
    }
}

impl DirtyState {
    /// All flags set, with the common tables filled.
    pub fn new() -> Self {
        let mut state = Self {
            flags: [u64::MAX; 4],
            tables: bytemuck::zeroed_box(),
        };
        state.fill_common();
        state
    }

    #[inline]
    pub fn is_set(&self, flag: u8) -> bool {
        self.flags[usize::from(flag >> 6)] & (1 << (flag & 63)) != 0
    }

    #[inline]
    pub fn set(&mut self, flag: u8) {
        self.flags[usize::from(flag >> 6)] |= 1 << (flag & 63);
    }

    #[inline]
    pub fn clear(&mut self, flag: u8) {
        self.flags[usize::from(flag >> 6)] &= !(1 << (flag & 63));
    }

    /// Clears the flag and returns whether it was set.
    #[inline]
    pub fn take(&mut self, flag: u8) -> bool {
        let was = self.is_set(flag);
        self.clear(flag);
        was
    }

    pub fn set_all(&mut self) {
        self.flags = [u64::MAX; 4];
    }

    /// Makes writes to methods `[first, first + count)` set `flag` through table `table` (0 or 1).
    pub fn fill(&mut self, table: usize, first: u32, count: u32, flag: u8) {
        let first = first as usize;
        self.tables[table][first..first + count as usize].fill(flag);
    }

    /// The flags a write to `method` sets.
    pub fn flags_for(&self, method: u32) -> [u8; 2] {
        [
            self.tables[0][method as usize],
            self.tables[1][method as usize],
        ]
    }

    #[inline]
    fn mark(&mut self, method: usize) {
        let a = self.tables[0][method];
        let b = self.tables[1][method];
        self.set(a);
        self.set(b);
    }

    fn fill_common(&mut self) {
        use dirty::*;
        let words = |bytes: usize| (bytes / 4) as u32;
        let stream = words(size_of::<super::regs::VertexStream>());
        let limit = words(size_of::<super::regs::VertexStreamLimit>());
        for i in 0..32u32 {
            let flag = VERTEX_BUFFER0 + i as u8;
            let first = reg_index!(Regs, vertex_streams) + i * stream;
            self.fill(0, first, 3, flag);
            self.fill(1, first, 3, VERTEX_BUFFERS);
            self.fill(0, 0x600 / 4 + i * 2, 2, flag);
            self.fill(1, 0x600 / 4 + i * 2, 2, VERTEX_BUFFERS);
            let first = reg_index!(Regs, vertex_stream_limits) + i * limit;
            self.fill(0, first, limit, flag);
            self.fill(1, first, limit, VERTEX_BUFFERS);
        }
        self.fill(
            0,
            reg_index!(Regs, index_buffer),
            words(size_of::<super::regs::IndexBuffer>()),
            INDEX_BUFFER,
        );
        self.fill(
            0,
            reg_index!(Regs, tex_header),
            words(size_of::<super::regs::TexHeader>()),
            DESCRIPTORS,
        );
        self.fill(
            0,
            reg_index!(Regs, tex_sampler),
            words(size_of::<super::regs::TexSampler>()),
            DESCRIPTORS,
        );
        let rt = words(size_of::<super::regs::RenderTargetConfig>());
        for i in 0..8u32 {
            self.fill(
                0,
                reg_index!(Regs, rt) + i * rt,
                rt,
                COLOR_BUFFER0 + i as u8,
            );
        }
        self.fill(1, reg_index!(Regs, rt), rt * 8, RENDER_TARGETS);
        self.fill(
            0,
            reg_index!(Regs, surface_clip),
            words(size_of::<super::regs::SurfaceClip>()),
            RENDER_TARGETS,
        );
        self.fill(0, reg_index!(Regs, rt_control), 1, RENDER_TARGETS);
        self.fill(1, reg_index!(Regs, rt_control), 1, RENDER_TARGET_CONTROL);
        for (table, flag) in [(0, ZETA_BUFFER), (1, RENDER_TARGETS)] {
            self.fill(table, reg_index!(Regs, zeta_enable), 1, flag);
            self.fill(table, reg_index!(Regs, zeta_size.width), 1, flag);
            self.fill(table, reg_index!(Regs, zeta_size.height), 1, flag);
            self.fill(
                table,
                reg_index!(Regs, zeta),
                words(size_of::<super::regs::Zeta>()),
                flag,
            );
        }
        self.fill(
            0,
            reg_index!(Regs, pipelines),
            words(size_of::<[super::regs::Pipeline; 6]>()),
            SHADERS,
        );
        self.fill(1, super::regs::MESH_CONTROL as u32, 4, SHADERS);
    }
}

/// A set of methods, one bit each.
#[derive(Clone)]
pub struct MethodSet([u64; NUM_REGS / 64]);

impl MethodSet {
    pub const fn empty() -> Self {
        Self([0; NUM_REGS / 64])
    }

    #[inline]
    pub fn contains(&self, method: u32) -> bool {
        let m = method as usize;
        self.0[m >> 6] & (1 << (m & 63)) != 0
    }

    pub fn insert(&mut self, method: u32) {
        let m = method as usize;
        self.0[m >> 6] |= 1 << (m & 63);
    }

    pub fn insert_range(&mut self, first: u32, count: u32) {
        for m in first..first + count {
            self.insert(m);
        }
    }
}

/// Methods whose write triggers an action (a draw, an upload, a query, a macro bind) besides
/// storing the value. Writes to any other register only store and mark dirty flags.
pub fn executable_methods() -> MethodSet {
    let mut set = MethodSet::empty();
    for method in [
        reg_index!(Regs, draw.end),
        reg_index!(Regs, draw.begin),
        reg_index!(Regs, vertex_buffer.first),
        reg_index!(Regs, vertex_buffer.count),
        reg_index!(Regs, index_buffer.first),
        reg_index!(Regs, index_buffer.count),
        reg_index!(Regs, draw_inline_index),
        reg_index!(Regs, index_buffer32_subsequent),
        reg_index!(Regs, index_buffer16_subsequent),
        reg_index!(Regs, index_buffer8_subsequent),
        reg_index!(Regs, index_buffer32_first),
        reg_index!(Regs, index_buffer16_first),
        reg_index!(Regs, index_buffer8_first),
        reg_index!(Regs, inline_index_2x16.bits1),
        reg_index!(Regs, inline_index_4x8.bits1),
        ROOT_TABLE_SELECT,
        ROOT_TABLE_LOAD,
        reg_index!(Regs, vertex_array_instance_first),
        reg_index!(Regs, vertex_array_instance_subsequent),
        reg_index!(Regs, draw_texture.src_y0),
        reg_index!(Regs, wait_for_idle),
        reg_index!(Regs, shadow_ram_control),
        reg_index!(Regs, load_mme.instruction_ptr),
        reg_index!(Regs, load_mme.instruction),
        reg_index!(Regs, load_mme.start_address),
        reg_index!(Regs, falcon) + 4,
        reg_index!(Regs, topology_override),
        reg_index!(Regs, clear_surface),
        reg_index!(Regs, report_semaphore.query),
        reg_index!(Regs, render_enable.mode),
        reg_index!(Regs, clear_report_value),
        reg_index!(Regs, sync_info),
        reg_index!(Regs, launch_dma),
        reg_index!(Regs, inline_data),
        reg_index!(Regs, fragment_barrier),
        reg_index!(Regs, invalidate_texture_data_cache),
        reg_index!(Regs, tiled_cache_barrier),
    ] {
        set.insert(method);
    }
    set.insert_range(reg_index!(Regs, const_buffer.buffer), 16);
    let bind_group = (size_of::<super::regs::BindGroup>() / 4) as u32;
    for stage in 0..5 {
        set.insert(reg_index!(Regs, bind_groups) + stage * bind_group + 4);
    }
    set
}

/// The 3D register file with its shadow RAM and dirty flags.
pub struct RegisterState {
    pub regs: Box<Regs>,
    shadow: Box<Regs>,
    pub dirty: DirtyState,
    executable: MethodSet,
}

impl Default for RegisterState {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for RegisterState {
    fn clone(&self) -> Self {
        Self {
            regs: self.regs.clone(),
            shadow: self.shadow.clone(),
            dirty: DirtyState {
                flags: self.dirty.flags,
                tables: self.dirty.tables.clone(),
            },
            executable: self.executable.clone(),
        }
    }
}

impl RegisterState {
    pub fn new() -> Self {
        let regs = Regs::with_defaults();
        Self {
            shadow: regs.clone(),
            regs,
            dirty: DirtyState::new(),
            executable: executable_methods(),
        }
    }

    /// Whether a write to `method` (below [`MACRO_REGISTERS_START`]) needs the engine to act.
    #[inline]
    pub fn is_executable(&self, method: u32) -> bool {
        self.executable.contains(method)
    }

    /// The shadow RAM mode in effect.
    #[inline]
    pub fn shadow_ram_control(&self) -> ShadowRamControl {
        self.shadow.shadow_ram_control
    }

    /// The shadow RAM copy of a register.
    #[inline]
    pub fn shadow(&self, method: u32) -> u32 {
        self.shadow.words()[method as usize]
    }

    /// Writes `argument` to register `method` (below [`MACRO_REGISTERS_START`]) through shadow
    /// RAM. Returns the value the register now holds when the method is executable, so the caller
    /// runs its action with it; `None` when the write only stored state.
    #[inline]
    pub fn write(&mut self, method: u32, argument: u32) -> Option<u32> {
        let m = method as usize;
        let value = match self.shadow.shadow_ram_control {
            ShadowRamControl::TRACK | ShadowRamControl::TRACK_WITH_FILTER => {
                self.shadow.words_mut()[m] = argument;
                argument
            }
            ShadowRamControl::REPLAY => self.shadow.words()[m],
            _ => argument,
        };
        let executable = self.executable.contains(method);
        let current = &mut self.regs.words_mut()[m];
        if *current != value {
            *current = value;
            self.dirty.mark(m);
        } else if !executable {
            return None;
        }
        if !executable {
            return None;
        }
        if method == reg_index!(Regs, shadow_ram_control) {
            self.shadow.shadow_ram_control = ShadowRamControl(argument);
        }
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VIEWPORT_SCALE_X: u32 = reg_index!(Regs, viewport_transform);
    const SHADOW: u32 = reg_index!(Regs, shadow_ram_control);

    #[test]
    fn offsets_match_method_numbers() {
        assert_eq!(reg_index!(Regs, draw.begin), 0x586);
        assert_eq!(reg_index!(Regs, const_buffer.buffer), 0x8e4);
        assert_eq!(reg_index!(Regs, load_mme.instruction), 0x46);
        assert_eq!(reg_index!(Regs, rt), 0x200);
        assert_eq!(SHADOW, 0x49);
    }

    #[test]
    fn defaults_view_as_words() {
        let state = RegisterState::new();
        assert_eq!(
            state.regs.words()[reg_index!(Regs, rasterize_enable) as usize],
            1
        );
        assert_eq!(state.regs.viewports[3].depth_range_far, 1.0);
        assert_eq!(
            state.regs.viewport_transform[0].swizzle.y(),
            ViewportSwizzle::POSITIVE_Y
        );
        assert_eq!(state.regs.color_mask[7].a(), 1);
        assert_eq!(state.regs.vertex_attrib_format[0].constant(), 1);
    }

    #[test]
    fn same_value_state_writes_do_nothing() {
        let mut state = RegisterState::new();
        state.write(VIEWPORT_SCALE_X, 5.0f32.to_bits());
        state.dirty.flags = [0; 4];
        let rt_width = reg_index!(Regs, rt) + 2;
        state.dirty.fill(0, VIEWPORT_SCALE_X, 1, dirty::LAST_COMMON);
        assert_eq!(state.write(VIEWPORT_SCALE_X, 5.0f32.to_bits()), None);
        assert!(!state.dirty.is_set(dirty::LAST_COMMON));
        assert_eq!(state.write(rt_width, 640), None);
        assert!(state.dirty.is_set(dirty::COLOR_BUFFER0));
        assert!(state.dirty.is_set(dirty::RENDER_TARGETS));
        assert!(!state.dirty.is_set(dirty::COLOR_BUFFER0 + 1));
        assert_eq!(state.regs.rt[0].width, 640);
    }

    #[test]
    fn executable_writes_report_even_when_unchanged() {
        let mut state = RegisterState::new();
        for method in [
            reg_index!(Regs, draw.end),
            reg_index!(Regs, inline_index_2x16.bits1),
            reg_index!(Regs, inline_index_4x8.bits1),
            ROOT_TABLE_LOAD,
        ] {
            assert_eq!(state.write(method, 0), Some(0));
            assert_eq!(state.write(method, 0), Some(0));
        }
        assert_eq!(state.write(reg_index!(Regs, inline_index_2x16), 0), None);
    }

    #[test]
    fn shadow_ram_tracks_and_replays() {
        let mut state = RegisterState::new();
        let reg = reg_index!(Regs, depth_test_enable);
        state.write(SHADOW, ShadowRamControl::TRACK.0);
        state.write(reg, 1);
        assert_eq!(state.shadow(reg), 1);
        state.write(SHADOW, ShadowRamControl::PASSTHROUGH.0);
        state.write(reg, 0);
        assert_eq!(state.regs.depth_test_enable, 0);
        assert_eq!(state.shadow(reg), 1);
        state.write(SHADOW, ShadowRamControl::REPLAY.0);
        state.write(reg, 7);
        assert_eq!(state.regs.depth_test_enable, 1);
        assert_eq!(state.shadow_ram_control(), ShadowRamControl::REPLAY);
        assert_eq!(
            state.write(SHADOW, ShadowRamControl::TRACK.0),
            Some(ShadowRamControl::REPLAY.0)
        );
        assert_eq!(state.shadow_ram_control(), ShadowRamControl::TRACK);
    }

    #[test]
    fn vertex_stream_writes_mark_their_buffer() {
        let mut state = RegisterState::new();
        state.dirty.flags = [0; 4];
        let stream5_address = reg_index!(Regs, vertex_streams) + 5 * 4 + 1;
        state.write(stream5_address, 0x1234);
        assert!(state.dirty.take(dirty::VERTEX_BUFFER0 + 5));
        assert!(state.dirty.take(dirty::VERTEX_BUFFERS));
        assert!(!state.dirty.is_set(dirty::VERTEX_BUFFER0 + 5));
    }
}
