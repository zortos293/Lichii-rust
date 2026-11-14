//! The back-thread 3D engine: executes the method writes the front thread sends (macros already
//! expanded) in guest order.
//!
//! Every write goes through [`RegisterState::write`]. A write to a state register only stores the
//! value and sets dirty flags, and costs nothing more when the value is unchanged; the methods
//! in the executable set run their action here: draws, clears, constant buffer uploads and
//! binds, root tables, report semaphores, sync points and inline-to-memory uploads.
//!
//! Registers the engine changes on its own, which the front's register mirror must reproduce
//! (NX2_GPU_THREAD_SPLIT.md): `const_buffer.offset` (+4 per constant buffer data word) and
//! `shadow_scratch[0]` (set to 1 by FALCON[4]).

use super::const_buffers::{self, RootTableState};
use super::draw::{self, DrawState};
use super::queries;
use super::regs::{BindGroup, LaunchDmaLayout, MAX_SHADER_STAGE, Regs, ShadowRamControl};
use super::state::{
    MACRO_REGISTERS_START, ROOT_TABLE_LOAD, ROOT_TABLE_SELECT, RegisterState, dirty,
};
use crate::activity::RendererCall;
use crate::backend::{
    DrawParams, FenceAction, GraphicsConstBuffers, GraphicsContext, IndirectDraw, InlineIndices,
};
use crate::engines::upload::Upload;
use crate::engines::{EngineContext, EngineKind};
use crate::reg_index;

const WAIT_FOR_IDLE: u32 = reg_index!(Regs, wait_for_idle);
const FALCON4: u32 = reg_index!(Regs, falcon) + 4;
const CB_DATA_FIRST: u32 = reg_index!(Regs, const_buffer.buffer);
const CB_DATA_LAST: u32 = CB_DATA_FIRST + 15;
const BIND_GROUP_STRIDE: u32 = (size_of::<BindGroup>() / 4) as u32;
const BIND_GROUP_FIRST: u32 = reg_index!(Regs, bind_groups) + 4;
const BIND_GROUP_LAST: u32 = BIND_GROUP_FIRST + (MAX_SHADER_STAGE as u32 - 1) * BIND_GROUP_STRIDE;
const CLEAR_SURFACE: u32 = reg_index!(Regs, clear_surface);
const REPORT_SEMAPHORE_QUERY: u32 = reg_index!(Regs, report_semaphore.query);
const RENDER_ENABLE_MODE: u32 = reg_index!(Regs, render_enable.mode);
const CLEAR_REPORT_VALUE: u32 = reg_index!(Regs, clear_report_value);
const SYNC_INFO: u32 = reg_index!(Regs, sync_info);
const LAUNCH_DMA: u32 = reg_index!(Regs, launch_dma);
const INLINE_DATA: u32 = reg_index!(Regs, inline_data);
const FRAGMENT_BARRIER: u32 = reg_index!(Regs, fragment_barrier);
const INVALIDATE_TEXTURE_DATA_CACHE: u32 = reg_index!(Regs, invalidate_texture_data_cache);
const TILED_CACHE_BARRIER: u32 = reg_index!(Regs, tiled_cache_barrier);

/// The renderer's view of the engine for one call, borrowing fields one by one so that draw
/// parameters can borrow the engine's index buffer at the same time.
macro_rules! graphics {
    ($engine:expr, $ctx:expr) => {
        GraphicsContext {
            channel: $ctx.channel,
            memory: $ctx.memory,
            regs: &$engine.state.regs,
            dirty: &mut $engine.state.dirty,
            const_buffers: &$engine.const_buffers,
            root_tables: &$engine.root_tables.tables,
        }
    };
}

/// The 3D engine (Maxwell B197, Ampere C797) of one channel, on the GPU's back thread.
///
/// Draws may be held back so that the next instances of an instance loop join them: call
/// [`Maxwell3d::flush`] before another engine of the channel executes a method and at the end of
/// each submission.
pub struct Maxwell3d {
    /// Registers, shadow RAM and dirty flags.
    pub state: RegisterState,
    const_buffers: GraphicsConstBuffers,
    root_tables: RootTableState,
    draw: DrawState,
    upload: Upload,
    /// RENDER_ENABLE as evaluated when its mode was last written.
    render_enable: bool,
}

impl Maxwell3d {
    /// A 3D engine for 3D class `class`, with the registers at their boot values.
    pub fn new(class: u32) -> Self {
        debug_assert_eq!(EngineKind::from_class(class), Some(EngineKind::Maxwell3d));
        Self {
            state: RegisterState::new(),
            const_buffers: GraphicsConstBuffers::default(),
            root_tables: RootTableState::new(),
            draw: DrawState::new(),
            upload: Upload::default(),
            render_enable: true,
        }
    }

    /// Executes one method write. `method` is below the macro range: macros run on the front
    /// thread, which sends the methods they write instead.
    #[inline]
    pub fn call_method(&mut self, ctx: &mut EngineContext<'_>, method: u32, argument: u32) {
        debug_assert!(
            method < MACRO_REGISTERS_START,
            "macro method {method:#x} on the back"
        );
        if method >= MACRO_REGISTERS_START {
            return;
        }
        if self.draw.pending.is_some()
            && !draw::is_loop_method(method)
            && self.state.would_change(method, argument)
        {
            self.flush(ctx);
        }
        if let Some(value) = self.state.write(method, argument) {
            self.execute(ctx, method, value);
        }
    }

    /// Executes a run of writes to the same method (a non-incrementing method's data). Constant
    /// buffer data, inline-to-memory data, inline indices and root table loads are handled as
    /// one block; other methods are written one by one.
    pub fn call_multi(&mut self, ctx: &mut EngineContext<'_>, method: u32, arguments: &[u32]) {
        let Some(&last) = arguments.last() else {
            return;
        };
        let block = matches!(
            method,
            CB_DATA_FIRST
                ..=CB_DATA_LAST
                    | INLINE_DATA
                    | ROOT_TABLE_LOAD
                    | draw::DRAW_INLINE_INDEX
                    | draw::INLINE_INDEX_2X16
                    | draw::INLINE_INDEX_4X8
        );
        if !block || self.state.shadow_ram_control() == ShadowRamControl::REPLAY {
            for &argument in arguments {
                self.call_method(ctx, method, argument);
            }
            return;
        }
        if !draw::is_loop_method(method) {
            self.flush(ctx);
        }
        self.state.write(method, last);
        match method {
            CB_DATA_FIRST..=CB_DATA_LAST => {
                const_buffers::upload(&mut self.state.regs.const_buffer, ctx, arguments);
            }
            INLINE_DATA => {
                for &word in arguments {
                    self.upload.data(ctx, word);
                }
            }
            ROOT_TABLE_LOAD => self.load_root_table(arguments),
            _ => self.draw.push_inline(method, arguments),
        }
    }

    /// Draws the held-back draw, if any.
    pub fn flush(&mut self, ctx: &mut EngineContext<'_>) {
        if let Some(pending) = self.draw.pending.take() {
            self.render(ctx, &pending);
        }
    }

    /// Draws with parameters in GPU memory, with the current state.
    pub fn draw_indirect(&mut self, ctx: &mut EngineContext<'_>, draw: &IndirectDraw) {
        self.flush(ctx);
        crate::diag::add(if self.renders() {
            crate::diag::K::eng_indirect
        } else {
            crate::diag::K::eng_skip_indirect
        });
        if self.renders() {
            let draw = IndirectDraw {
                topology: draw::topology(&self.state.regs, draw.topology),
                ..*draw
            };
            let start = ctx.activity.begin_call(RendererCall::Draw);
            ctx.rasterizer
                .draw_indirect(&mut graphics!(self, ctx), &draw);
            ctx.activity.renderer_call(RendererCall::Draw, start);
        }
    }

    fn execute(&mut self, ctx: &mut EngineContext<'_>, method: u32, value: u32) {
        let regs = &self.state.regs;
        match method {
            draw::DRAW_BEGIN => self.draw.begin(regs.draw.begin),
            draw::DRAW_END => self.draw_end(ctx),
            draw::INDEX_BUFFER_COUNT => self.draw.indexed = true,
            draw::INDEX_BUFFER32_FIRST | draw::INDEX_BUFFER16_FIRST | draw::INDEX_BUFFER8_FIRST => {
                self.draw_index_small(ctx, value, false);
            }
            draw::INDEX_BUFFER32_SUBSEQUENT
            | draw::INDEX_BUFFER16_SUBSEQUENT
            | draw::INDEX_BUFFER8_SUBSEQUENT => self.draw_index_small(ctx, value, true),
            draw::VERTEX_ARRAY_INSTANCE_FIRST => self.draw_vertex_array(ctx, false),
            draw::VERTEX_ARRAY_INSTANCE_SUBSEQUENT => self.draw_vertex_array(ctx, true),
            draw::DRAW_INLINE_INDEX | draw::INLINE_INDEX_2X16 | draw::INLINE_INDEX_4X8 => {
                self.draw.push_inline(method, &[value]);
            }
            CB_DATA_FIRST..=CB_DATA_LAST => {
                const_buffers::upload(&mut self.state.regs.const_buffer, ctx, &[value]);
            }
            BIND_GROUP_FIRST..=BIND_GROUP_LAST
                if (method - BIND_GROUP_FIRST).is_multiple_of(BIND_GROUP_STRIDE) =>
            {
                let stage = ((method - BIND_GROUP_FIRST) / BIND_GROUP_STRIDE) as usize;
                let config = regs.bind_groups[stage].raw_config;
                if const_buffers::bind(&mut self.const_buffers, &regs.const_buffer, stage, config) {
                    self.state.dirty.set(dirty::CONST_BUFFER0 + stage as u8);
                }
            }
            ROOT_TABLE_SELECT => self.root_tables.select(value),
            ROOT_TABLE_LOAD => self.load_root_table(&[value]),
            CLEAR_SURFACE => {
                crate::diag::add(if self.renders() {
                    crate::diag::K::eng_clear
                } else {
                    crate::diag::K::eng_skip_clear
                });
                if self.renders() {
                    let start = ctx.activity.begin_call(RendererCall::Clear);
                    ctx.rasterizer.clear(&mut graphics!(self, ctx), 1);
                    ctx.activity.renderer_call(RendererCall::Clear, start);
                }
            }
            draw::DRAW_TEXTURE => {
                crate::diag::add(if self.renders() {
                    crate::diag::K::eng_texture
                } else {
                    crate::diag::K::eng_skip_texture
                });
                if self.renders() {
                    let texture = draw::texture(regs);
                    let start = ctx.activity.begin_call(RendererCall::Draw);
                    ctx.rasterizer
                        .draw_texture(&mut graphics!(self, ctx), &texture);
                    ctx.activity.renderer_call(RendererCall::Draw, start);
                }
            }
            REPORT_SEMAPHORE_QUERY => {
                if let Some(query) = queries::report(regs) {
                    use crate::diag::K;
                    crate::diag::add(match query.kind {
                        crate::backend::QueryKind::Payload => K::q_payload,
                        crate::backend::QueryKind::SamplesPassed
                            if query.size == crate::backend::ReleaseSize::Four =>
                        {
                            K::q_samples_short
                        }
                        crate::backend::QueryKind::SamplesPassed => K::q_samples,
                        crate::backend::QueryKind::PrimitivesGenerated => K::q_prims,
                        crate::backend::QueryKind::TransformFeedbackPrimitivesWritten => K::q_xfb,
                        crate::backend::QueryKind::Counter(_) => K::q_counter,
                    });
                    ctx.rasterizer.query(ctx.channel, ctx.memory, &query);
                }
            }
            CLEAR_REPORT_VALUE => {
                crate::diag::add(
                    if queries::counter(regs.clear_report_value)
                        == Some(crate::backend::QueryKind::SamplesPassed)
                    {
                        crate::diag::K::clear_samples
                    } else {
                        crate::diag::K::clear_other
                    },
                );
                if let Some(kind) = queries::counter(regs.clear_report_value) {
                    ctx.rasterizer.reset_counter(kind);
                }
            }
            RENDER_ENABLE_MODE => {
                self.render_enable = queries::render_enable(regs, ctx);
                use super::regs::RenderEnableMode as M;
                let mode = self.state.regs.render_enable.mode;
                crate::diag::add(match (mode, self.render_enable) {
                    (M::FALSE, _) => crate::diag::K::re_false,
                    (M::TRUE, _) => crate::diag::K::re_true,
                    (M::CONDITIONAL | M::IF_EQUAL | M::IF_NOT_EQUAL, true) => {
                        crate::diag::K::re_cond_true
                    }
                    (M::CONDITIONAL | M::IF_EQUAL | M::IF_NOT_EQUAL, false) => {
                        crate::diag::K::re_cond_false
                    }
                    _ => crate::diag::K::re_other,
                });
            }
            SYNC_INFO => {
                let id = regs.sync_info.sync_point();
                if ctx.syncpoints.host().increment(id).is_err() {
                    tracing::warn!(id, "3D sync point increments an invalid syncpoint");
                    return;
                }
                ctx.rasterizer
                    .signal_fence(FenceAction::IncrementSyncpoint(id));
            }
            FALCON4 => self.state.regs.shadow_scratch[0] = 1,
            LAUNCH_DMA => self.upload.exec(
                &regs.upload,
                regs.launch_dma.memory_layout() == LaunchDmaLayout::PITCH,
            ),
            INLINE_DATA => self.upload.data(ctx, value),
            WAIT_FOR_IDLE | FRAGMENT_BARRIER | TILED_CACHE_BARRIER => ctx.rasterizer.barrier(),
            INVALIDATE_TEXTURE_DATA_CACHE => {
                ctx.rasterizer.invalidate_gpu_caches();
                ctx.rasterizer.barrier();
            }
            _ => {}
        }
    }

    fn draw_end(&mut self, ctx: &mut EngineContext<'_>) {
        let regs = &self.state.regs;
        let topology = draw::topology(regs, self.draw.topology);
        let base_instance = regs
            .global_base_instance_index
            .wrapping_add(self.draw.instance);
        let base_vertex = regs.global_base_vertex_index as i32;
        if !self.draw.inline_indices.is_empty() {
            self.flush(ctx);
            crate::diag::add(if self.renders() {
                crate::diag::K::eng_draw_inline
            } else {
                crate::diag::K::eng_skip_draw
            });
            if self.renders() {
                let params = DrawParams {
                    topology,
                    indexed: true,
                    first: 0,
                    count: self.draw.inline_indices.len() as u32,
                    base_vertex,
                    base_instance,
                    instance_count: 1,
                    inline_indices: Some(InlineIndices::U32(&self.draw.inline_indices)),
                };
                let start = ctx.activity.begin_call(RendererCall::Draw);
                ctx.rasterizer.draw(&mut graphics!(self, ctx), &params);
                ctx.activity.renderer_call(RendererCall::Draw, start);
            }
            self.draw.inline_indices.clear();
            self.draw.indexed = false;
            self.state.dirty.set(dirty::INDEX_BUFFER);
            return;
        }
        let (first, count, base_vertex) = if self.draw.indexed {
            (
                regs.index_buffer.first,
                regs.index_buffer.count,
                base_vertex,
            )
        } else {
            (regs.vertex_buffer.first, regs.vertex_buffer.count, 0)
        };
        let indexed = self.draw.indexed;
        self.draw.indexed = false;
        self.queue(
            ctx,
            DrawParams {
                topology,
                indexed,
                first,
                count,
                base_vertex,
                base_instance,
                instance_count: 1,
                inline_indices: None,
            },
        );
    }

    fn draw_index_small(&mut self, ctx: &mut EngineContext<'_>, value: u32, subsequent: bool) {
        let small = super::regs::IndexBufferSmall(value);
        let instance = self.draw.next_instance(subsequent);
        let regs = &self.state.regs;
        let draw = DrawParams {
            topology: draw::topology(regs, small.topology()),
            indexed: true,
            first: small.first(),
            count: small.count(),
            base_vertex: regs.global_base_vertex_index as i32,
            base_instance: regs.global_base_instance_index.wrapping_add(instance),
            instance_count: 1,
            inline_indices: None,
        };
        self.queue(ctx, draw);
    }

    fn draw_vertex_array(&mut self, ctx: &mut EngineContext<'_>, subsequent: bool) {
        let instance = self.draw.next_instance(subsequent);
        let regs = &self.state.regs;
        let array = if subsequent {
            regs.vertex_array_instance_subsequent
        } else {
            regs.vertex_array_instance_first
        };
        let draw = DrawParams {
            topology: draw::topology(regs, array.topology()),
            indexed: false,
            first: array.start(),
            count: array.count(),
            base_vertex: 0,
            base_instance: regs.global_base_instance_index.wrapping_add(instance),
            instance_count: 1,
            inline_indices: None,
        };
        self.queue(ctx, draw);
    }

    fn queue(&mut self, ctx: &mut EngineContext<'_>, draw: DrawParams<'static>) {
        if draw.count == 0 {
            return;
        }
        if let Some(previous) = self.draw.queue(draw) {
            self.render(ctx, &previous);
        }
    }

    fn render(&mut self, ctx: &mut EngineContext<'_>, draw: &DrawParams<'_>) {
        crate::diag::add(if self.renders() {
            crate::diag::K::eng_draw
        } else {
            crate::diag::K::eng_skip_draw
        });
        match self.state.regs.render_enable_override {
            super::regs::RenderEnableOverride::NEVER_RENDER => {
                crate::diag::add(crate::diag::K::override_never)
            }
            super::regs::RenderEnableOverride::ALWAYS_RENDER => {
                crate::diag::add(crate::diag::K::override_always)
            }
            _ => {}
        }
        if self.renders() {
            let start = ctx.activity.begin_call(RendererCall::Draw);
            ctx.rasterizer.draw(&mut graphics!(self, ctx), draw);
            ctx.activity.renderer_call(RendererCall::Draw, start);
        }
    }

    fn load_root_table(&mut self, words: &[u32]) {
        if self.root_tables.load(words) {
            self.state.dirty.set(dirty::ROOT_TABLES);
        }
    }

    #[inline]
    fn renders(&self) -> bool {
        queries::renders(&self.state.regs, self.render_enable)
    }
}

#[cfg(test)]
mod tests;
