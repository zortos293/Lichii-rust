//! The renderer behind `lichii_gpu::backend::Rasterizer`, driven by the GPU's back thread.
//!
//! This milestone implements presentation (guest framebuffers read from device memory, deswizzled
//! and handed to the [`Presenter`](crate::present::Presenter), or taken from the texture cache
//! when the GPU rendered them), the fence and query protocol, clears (masked ones included)
//! through the [`TextureCache`], its collector once per presented frame, and draws and compute
//! dispatches (see [`draw`]).

mod draw;

use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;
use lichii_common::DAddr;
use lichii_common::settings::{AstcRecompression, GraphicsSettings, ResolutionScale};
use lichii_gpu::backend::ChannelId;
use lichii_gpu::backend::{
    ComputeContext, DrawParams, FenceAction, FramebufferConfig, GpuHandles, GraphicsContext,
    IndirectDraw, Query, QueryKind, Rasterizer,
};
use lichii_gpu::engines::maxwell_3d::Regs;
use lichii_gpu::engines::maxwell_3d::regs::NUM_RENDER_TARGETS;
use lichii_gpu::gmmu::Gmmu;
use lichii_texture::block_linear::{GobLayout, Tiling, unswizzle};
use lichii_texture::{Aspect, Numeric};

use crate::buffer_cache::BufferCache;
use crate::commands::{COLOR_RANGE, ImageAccess, transition};
use crate::device::Device;
use crate::error::Result;
use crate::format::{Conversion, host_format};
use crate::memory::{Image, ImageDesc};
use crate::pipeline::PipelineCache;
use crate::present::{FrameImage, FrameMailbox, GuestFrame};
use crate::scheduler::Scheduler;
use crate::staging::StagingRing;
use crate::texture_cache::{
    ClearValue, ImageId, ImageInfo, ImageKind, Rect, TextureCache, Transfer,
};

/// Frame images kept for reuse; one is being written while the presenter holds up to two.
const FRAME_POOL: usize = 4;

/// The Vulkan renderer. Owns the GPU thread's scheduler; shares only the device with the
/// presenter, and hands it frames through the mailbox.
pub struct VulkanRasterizer {
    device: Arc<Device>,
    handles: GpuHandles,
    scheduler: Scheduler,
    staging: StagingRing,
    mailbox: Arc<FrameMailbox>,
    gob_layout: GobLayout,
    frames: Vec<Arc<FrameImage>>,
    tiled: Vec<u8>,
    fences: VecDeque<(u64, FenceAction)>,
    unflushed: bool,
    texture_cache: TextureCache,
    buffer_cache: BufferCache,
    draw: draw::DrawState,
}

/// The rectangle a clear covers: the first scissor and the clear rectangle when
/// `clear_control` asks for them, in pixels.
fn clear_rect(regs: &Regs) -> Rect {
    let (mut x0, mut y0, mut x1, mut y1) = (0, 0, u32::MAX, u32::MAX);
    let mut clip = |min_x: u32, max_x: u32, min_y: u32, max_y: u32| {
        x0 = x0.max(min_x);
        y0 = y0.max(min_y);
        x1 = x1.min(max_x);
        y1 = y1.min(max_y);
    };
    let scissor = &regs.scissor_test[0];
    if regs.clear_control.use_scissor() != 0 && scissor.enable != 0 {
        clip(
            scissor.min_x(),
            scissor.max_x(),
            scissor.min_y(),
            scissor.max_y(),
        );
    }
    let rect = &regs.clear_rect;
    if regs.clear_control.use_clear_rect() != 0 {
        clip(rect.x_min(), rect.x_max(), rect.y_min(), rect.y_max());
    }
    Rect {
        x: x0,
        y: y0,
        width: x1.saturating_sub(x0),
        height: y1.saturating_sub(y0),
    }
}

/// `rect` in the sample grid of a multisampled image `info` describes, whose pixels are
/// `width` by `height`.
fn sample_rect(rect: Rect, info: &ImageInfo, width: u32, height: u32) -> Rect {
    let (gx, gy) = (info.width / width.max(1), info.height / height.max(1));
    Rect {
        x: rect.x.saturating_mul(gx),
        y: rect.y.saturating_mul(gy),
        width: rect.width.saturating_mul(gx),
        height: rect.height.saturating_mul(gy),
    }
}

/// Whether a clear of `layer_count` layers (3D slices) from `base_layer` inside `rect` writes
/// all of `info`.
fn covers(rect: Rect, base_layer: u32, layer_count: u32, info: &ImageInfo) -> bool {
    let layers = if info.kind == ImageKind::D3 {
        info.depth
    } else {
        info.layers
    };
    rect.x == 0
        && rect.y == 0
        && rect.width >= info.width
        && rect.height >= info.height
        && base_layer == 0
        && layer_count >= layers
}

impl VulkanRasterizer {
    /// `gob_layout` is the tiling of guest textures and render targets: `Switch2` for native
    /// Switch 2 titles. Framebuffers carry their own layout.
    pub fn new(
        device: &Arc<Device>,
        handles: GpuHandles,
        mailbox: Arc<FrameMailbox>,
        gob_layout: GobLayout,
    ) -> Result<Self> {
        let buffer_cache = BufferCache::new(device, handles.device.clone())?;
        let mut scheduler = Scheduler::new(device)?;
        let pipelines = PipelineCache::new(device, &GraphicsSettings::default(), None, None)?;
        let draw = draw::DrawState::new(device, pipelines, gob_layout, scheduler.record()?)?;
        Ok(Self {
            buffer_cache,
            draw,
            device: device.clone(),
            handles,
            scheduler,
            staging: StagingRing::new(device, 32 << 20)?,
            mailbox,
            gob_layout,
            frames: Vec::new(),
            tiled: Vec::new(),
            fences: VecDeque::new(),
            unflushed: true,
            texture_cache: TextureCache::new(device, gob_layout, AstcRecompression::Uncompressed),
        })
    }

    /// Applies the settings the renderer reads while running: the resolution scale render
    /// targets use from their next binding on, the ASTC fallback format of new textures, and
    /// how much device memory the texture cache may fill, sampler anisotropy and whether
    /// pipelines compile asynchronously.
    pub fn apply_settings(&mut self, settings: &GraphicsSettings) {
        self.draw.set_anisotropy(settings.anisotropic_filtering);
        self.draw
            .pipelines
            .set_async_compilation(settings.async_shader_compilation);
        self.draw
            .shaders
            .set_resolution_scale(settings.resolution_scale);
        self.texture_cache.set_vram_usage(settings.vram_usage);
        self.texture_cache
            .set_resolution_scale(settings.resolution_scale);
        self.texture_cache
            .set_astc_recompression(settings.astc_recompression);
    }

    fn clear_targets(&mut self, ctx: &GraphicsContext<'_>, layer_count: u32) -> Result<()> {
        let regs = ctx.regs;
        let surface = regs.clear_surface;
        let channels = [surface.r(), surface.g(), surface.b(), surface.a()];
        let use_color = channels.iter().any(|&c| c != 0);
        let (use_depth, use_stencil) = (surface.z() != 0, surface.s() != 0);
        if !use_color && !use_depth && !use_stencil {
            return Ok(());
        }
        let rect = clear_rect(regs);
        let base_layer = surface.layer();
        let msaa = regs.anti_alias_samples_mode.0;
        let index = surface.rt() as usize;
        let color_target = (use_color
            && index < (regs.rt_control.count() as usize).min(NUM_RENDER_TARGETS))
        .then(|| {
            let rt = &regs.rt[index];
            ImageInfo::from_render_target(rt, msaa, self.gob_layout)
                .map(|info| (rt.address(), info, rt.width, rt.height))
        })
        .flatten();
        if let Some((address, info, width, height)) = color_target {
            let mask = regs.color_mask[if regs.color_mask_common != 0 {
                0
            } else {
                index
            }];
            let enabled = [mask.r(), mask.g(), mask.b(), mask.a()];
            let written = channels.iter().zip(enabled).enumerate().fold(
                vk::ColorComponentFlags::empty(),
                |written, (i, (&c, m))| {
                    if c & m != 0 {
                        written | vk::ColorComponentFlags::from_raw(1 << i)
                    } else {
                        written
                    }
                },
            );
            if !written.is_empty() {
                let c = regs.clear_color;
                let value = match info.format.numeric() {
                    Numeric::Uint => vk::ClearColorValue {
                        uint32: c.map(f32::to_bits),
                    },
                    Numeric::Sint => vk::ClearColorValue {
                        int32: c.map(|v| v.to_bits() as i32),
                    },
                    _ => vk::ClearColorValue { float32: c },
                };
                let rect = sample_rect(rect, &info, width, height);
                let discard = written == vk::ColorComponentFlags::RGBA
                    && covers(rect, base_layer, layer_count, &info);
                let mut tx = Transfer {
                    scheduler: &mut self.scheduler,
                    staging: &mut self.staging,
                    memory: &self.handles.device,
                };
                if let Some(target) = self
                    .texture_cache
                    .render_target(&mut tx, ctx.memory, address, &info, discard)?
                {
                    self.texture_cache.clear(
                        &mut tx,
                        target,
                        ClearValue::Color {
                            value,
                            mask: written,
                        },
                        rect,
                        base_layer,
                        layer_count,
                    )?;
                }
            }
        }
        let depth_target = ((use_depth || use_stencil) && regs.zeta_enable != 0)
            .then(|| ImageInfo::from_zeta(&regs.zeta, &regs.zeta_size, msaa, self.gob_layout))
            .flatten();
        if let Some(info) = depth_target {
            let stencil_mask = if regs.clear_control.respect_stencil_mask() != 0 {
                regs.stencil_front_mask & 0xff
            } else {
                0xff
            };
            let depth = use_depth.then_some(regs.clear_depth);
            let stencil = (use_stencil && stencil_mask != 0).then_some(regs.clear_stencil as u8);
            let full_stencil = stencil.is_some() && stencil_mask == 0xff;
            let aspects_covered = match info.format.aspect() {
                Aspect::DepthStencil => depth.is_some() && full_stencil,
                Aspect::Depth => depth.is_some(),
                Aspect::Stencil => full_stencil,
                Aspect::Color => false,
            };
            if depth.is_some() || stencil.is_some() {
                let rect = sample_rect(rect, &info, regs.zeta_size.width, regs.zeta_size.height);
                let discard = aspects_covered && covers(rect, base_layer, layer_count, &info);
                let mut tx = Transfer {
                    scheduler: &mut self.scheduler,
                    staging: &mut self.staging,
                    memory: &self.handles.device,
                };
                if let Some(target) = self.texture_cache.render_target(
                    &mut tx,
                    ctx.memory,
                    regs.zeta.address(),
                    &info,
                    discard,
                )? {
                    self.texture_cache.clear(
                        &mut tx,
                        target,
                        ClearValue::DepthStencil {
                            depth,
                            stencil,
                            stencil_mask: stencil_mask as u8,
                        },
                        rect,
                        base_layer,
                        layer_count,
                    )?;
                }
            }
        }
        self.unflushed = true;
        Ok(())
    }

    /// Presents a framebuffer the GPU rendered into a cached image: copies it, at the image's
    /// scale, into a frame image on the GPU. `false` when the formats need the CPU path.
    fn present_cached(&mut self, fb: &FramebufferConfig, id: ImageId) -> Result<bool> {
        let Some(state) = self.texture_cache.state(id) else {
            return Ok(false);
        };
        let host = host_format(&self.device, fb.format, AstcRecompression::Uncompressed);
        let blit = vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST;
        let same_format = state.format == host.format;
        if !same_format
            && !(self.device.format_features(state.format).contains(blit)
                && self.device.format_features(host.format).contains(blit))
        {
            return Ok(false);
        }
        let scale = state.scale;
        let width = scale.scale(fb.width).min(state.extent.width);
        let height = scale.scale(fb.height).min(state.extent.height);
        let frame = self.frame_image(width, height, &host)?;
        let cmd = self.scheduler.record()?;
        let source = self
            .texture_cache
            .prepare(cmd, id, ImageAccess::TRANSFER_SRC);
        let image = frame.image.raw();
        transition(
            &self.device,
            cmd,
            image,
            COLOR_RANGE,
            ImageAccess::UNDEFINED,
            ImageAccess::TRANSFER_DST,
        );
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        // SAFETY: the source is in TRANSFER_SRC and the frame in TRANSFER_DST layout; the region
        // lies inside both.
        unsafe {
            if same_format {
                self.device.raw.cmd_copy_image(
                    cmd,
                    source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::ImageCopy {
                        src_subresource: layers,
                        dst_subresource: layers,
                        extent: vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        },
                        ..Default::default()
                    }],
                );
            } else {
                let corner = vk::Offset3D {
                    x: width as i32,
                    y: height as i32,
                    z: 1,
                };
                self.device.raw.cmd_blit_image(
                    cmd,
                    source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::ImageBlit {
                        src_subresource: layers,
                        src_offsets: [vk::Offset3D::default(), corner],
                        dst_subresource: layers,
                        dst_offsets: [vk::Offset3D::default(), corner],
                    }],
                    vk::Filter::NEAREST,
                );
            }
        }
        transition(
            &self.device,
            cmd,
            image,
            COLOR_RANGE,
            ImageAccess::TRANSFER_DST,
            ImageAccess::SHADER_READ,
        );
        let tick = self.scheduler.flush()?;
        self.unflushed = false;
        self.publish(fb, frame, tick, scale);
        Ok(true)
    }

    /// Hands a frame image holding `fb` (scaled by `scale`) to the presenter.
    fn publish(
        &self,
        fb: &FramebufferConfig,
        frame: Arc<FrameImage>,
        tick: u64,
        scale: ResolutionScale,
    ) {
        let (width, height) = (fb.width, fb.height);
        let (left, top, right, bottom) = fb
            .crop
            .map(|c| {
                (
                    c.left.min(width),
                    c.top.min(height),
                    c.right.min(width),
                    c.bottom.min(height),
                )
            })
            .filter(|&(left, top, right, bottom)| right > left && bottom > top)
            .unwrap_or((0, 0, width, height));
        let (w, h) = (width as f32, height as f32);
        let mut uv_rect = [
            left as f32 / w,
            top as f32 / h,
            (right - left) as f32 / w,
            (bottom - top) as f32 / h,
        ];
        if fb.flip_vertical {
            uv_rect[1] += uv_rect[3];
            uv_rect[3] = -uv_rect[3];
        }
        self.mailbox.publish(GuestFrame {
            image: frame,
            ready: (self.scheduler.shared_timeline(), tick),
            uv_rect,
            width: scale.scale(right - left),
            height: scale.scale(bottom - top),
        });
    }

    fn flush(&mut self) {
        if !self.unflushed {
            return;
        }
        self.unflushed = false;
        if let Err(error) = self.scheduler.flush() {
            tracing::error!(%error, "submitting GPU work failed");
        }
    }

    /// A free frame image of this size and format, or a new one.
    fn frame_image(
        &mut self,
        width: u32,
        height: u32,
        host: &crate::format::HostFormat,
    ) -> Result<Arc<FrameImage>> {
        let fits = |f: &Arc<FrameImage>| {
            let desc = f.image.desc();
            Arc::strong_count(f) == 1
                && desc.format == host.format
                && (desc.extent.width, desc.extent.height) == (width, height)
        };
        if let Some(frame) = self.frames.iter().find(|f| fits(f)) {
            return Ok(frame.clone());
        }
        if self.frames.len() >= FRAME_POOL
            && let Some(i) = self.frames.iter().position(|f| Arc::strong_count(f) == 1)
        {
            let old = self.frames.swap_remove(i);
            self.scheduler.retire(old);
        }
        let image = Image::new(
            &self.device,
            &ImageDesc::d2(
                host.format,
                width,
                height,
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            ),
            "guest frame",
        )?;
        let view = image.view(vk::ImageViewType::TYPE_2D, host.swizzle)?;
        let frame = Arc::new(FrameImage { image, view });
        self.frames.push(frame.clone());
        Ok(frame)
    }

    fn upload_frame(&mut self, fb: &FramebufferConfig) -> Result<()> {
        let info = fb.format.info();
        let host = host_format(&self.device, fb.format, AstcRecompression::Uncompressed);
        if info.compression.is_some() || !matches!(host.conversion, Conversion::None) {
            tracing::warn!(format = ?fb.format, "framebuffer format needs a conversion; not presented");
            return Ok(());
        }
        let (width, height) = (fb.width, fb.height);
        if width == 0 || height == 0 {
            return Ok(());
        }
        let row = (width * info.bytes_per_block) as usize;
        let slice = self
            .staging
            .alloc(&mut self.scheduler, (row * height as usize) as u64, 16)?;
        let read = match fb.block_height_log2 {
            Some(block_height_log2) => {
                let tiling = Tiling::new(
                    width,
                    height,
                    1,
                    info.bytes_per_block,
                    block_height_log2,
                    0,
                    fb.gob_layout,
                );
                self.tiled.resize(tiling.size_bytes(), 0);
                let read = self.handles.device.read_bytes(fb.address, &mut self.tiled);
                if read.is_ok() {
                    unswizzle(&tiling, &self.tiled, slice.data);
                }
                read
            }
            None => slice
                .data
                .chunks_exact_mut(row)
                .enumerate()
                .try_for_each(|(y, line)| {
                    self.handles
                        .device
                        .read_bytes(fb.address + y as DAddr * DAddr::from(fb.stride), line)
                }),
        };
        if let Err(fault) = read {
            tracing::warn!(
                ?fault,
                address = fb.address,
                "framebuffer is not mapped; not presented"
            );
            return Ok(());
        }
        let (buffer, offset) = (slice.buffer, slice.offset);
        let frame = self.frame_image(width, height, &host)?;
        let cmd = self.scheduler.record()?;
        let image = frame.image.raw();
        transition(
            &self.device,
            cmd,
            image,
            COLOR_RANGE,
            ImageAccess::UNDEFINED,
            ImageAccess::TRANSFER_DST,
        );
        let region = [vk::BufferImageCopy {
            buffer_offset: offset,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            ..Default::default()
        }];
        // SAFETY: the image is in TRANSFER_DST layout and the staged texels cover it exactly.
        unsafe {
            self.device.raw.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
            )
        };
        transition(
            &self.device,
            cmd,
            image,
            COLOR_RANGE,
            ImageAccess::TRANSFER_DST,
            ImageAccess::SHADER_READ,
        );
        let tick = self.scheduler.flush()?;
        self.unflushed = false;
        self.publish(fb, frame, tick, ResolutionScale::X1);
        Ok(())
    }
}

impl Rasterizer for VulkanRasterizer {
    fn draw(&mut self, ctx: &mut GraphicsContext<'_>, draw: &DrawParams<'_>) {
        if let Err(error) = self.draw_graphics(ctx, draw::DrawCall::Direct(draw)) {
            tracing::error!(%error, "recording a draw failed");
        }
    }

    fn draw_indirect(&mut self, ctx: &mut GraphicsContext<'_>, draw: &IndirectDraw) {
        if let Err(error) = self.draw_graphics(ctx, draw::DrawCall::Indirect(draw)) {
            tracing::error!(%error, "recording an indirect draw failed");
        }
    }

    fn clear(&mut self, ctx: &mut GraphicsContext<'_>, layer_count: u32) {
        if let Err(error) = self.clear_targets(ctx, layer_count) {
            tracing::error!(%error, "clearing render targets failed");
        }
    }

    fn dispatch_compute(&mut self, ctx: &ComputeContext<'_>) {
        if let Err(error) = self.dispatch(ctx) {
            tracing::error!(%error, "recording a compute dispatch failed");
        }
    }

    fn query(&mut self, _channel: ChannelId, memory: &Gmmu, query: &Query) {
        let Some(address) = memory.translate(query.address) else {
            tracing::warn!(address = query.address, "query to unmapped GPU memory");
            return;
        };
        let value = match query.kind {
            QueryKind::Payload => query.payload,
            _ => 0,
        };
        let action = FenceAction::WriteSemaphore {
            address,
            value,
            size: query.size,
        };
        if query.after_work {
            self.signal_fence(action);
        } else {
            action.execute(&self.handles);
        }
    }

    fn signal_fence(&mut self, action: FenceAction) {
        self.fences.push_back((self.scheduler.tick(), action));
        self.unflushed = true;
    }

    fn release_fences(&mut self, wait: bool) {
        if wait && let Some(&(tick, _)) = self.fences.back() {
            self.flush();
            if let Err(error) = self.scheduler.wait(tick) {
                tracing::error!(%error, "waiting for GPU work failed");
            }
        }
        let completed = self.scheduler.completed();
        while let Some(&(tick, action)) = self.fences.front() {
            if tick > completed {
                break;
            }
            self.fences.pop_front();
            action.execute(&self.handles);
        }
    }

    fn release_channel(&mut self, _channel: ChannelId) {
        self.texture_cache.invalidate_lookups();
    }

    fn invalidate_region(&mut self, addr: DAddr, size: u64) {
        self.buffer_cache.invalidate(addr, size);
        self.draw.shaders.invalidate(addr, size);
        let mut tx = Transfer {
            scheduler: &mut self.scheduler,
            staging: &mut self.staging,
            memory: &self.handles.device,
        };
        if let Err(error) = self.texture_cache.invalidate(&mut tx, addr, size) {
            tracing::error!(%error, "writing back GPU-modified images before a CPU write failed");
        }
    }

    fn flush_region(&mut self, addr: DAddr, size: u64) {
        if let Err(error) = self.buffer_cache.flush(&mut self.scheduler, addr, size) {
            tracing::error!(%error, "flushing GPU-written buffers failed");
        }
        let mut tx = Transfer {
            scheduler: &mut self.scheduler,
            staging: &mut self.staging,
            memory: &self.handles.device,
        };
        if let Err(error) = self.texture_cache.flush(&mut tx, addr, size) {
            tracing::error!(%error, "flushing GPU-modified images failed");
        }
    }

    fn unmap_region(&mut self, addr: DAddr, size: u64) {
        self.buffer_cache.invalidate(addr, size);
        self.draw.shaders.invalidate(addr, size);
        self.texture_cache.unmap(&mut self.scheduler, addr, size);
    }

    fn invalidate_gpu_caches(&mut self) {
        self.texture_cache.invalidate_lookups();
    }

    fn barrier(&mut self) {
        let cmd = match self.scheduler.record() {
            Ok(cmd) => cmd,
            Err(error) => {
                tracing::error!(%error, "recording a guest barrier failed");
                return;
            }
        };
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
        // SAFETY: `cmd` is recording outside any rendering; draws begin and end theirs.
        unsafe {
            self.device.raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[barrier]),
            );
        }
        self.unflushed = true;
    }

    fn flush_commands(&mut self) {
        self.flush();
    }

    fn wait_for_idle(&mut self) {
        self.flush();
        if let Err(error) = self.scheduler.finish() {
            tracing::error!(%error, "waiting for GPU idle failed");
        }
        self.release_fences(false);
    }

    fn present(&mut self, framebuffer: &FramebufferConfig) {
        self.handles.timestamp.frame_presented();
        let cached = match self.texture_cache.presentable(framebuffer) {
            Some(id) => self.present_cached(framebuffer, id),
            None => Ok(false),
        };
        let result = match cached {
            Ok(true) => Ok(()),
            Ok(false) => {
                let info = ImageInfo::from_framebuffer(framebuffer);
                self.flush_region(framebuffer.address, info.size_bytes());
                self.upload_frame(framebuffer)
            }
            Err(error) => Err(error),
        };
        if let Err(error) = result {
            tracing::error!(%error, "presenting a guest frame failed");
        }
        self.release_fences(false);
        let mut tx = Transfer {
            scheduler: &mut self.scheduler,
            staging: &mut self.staging,
            memory: &self.handles.device,
        };
        let device = &self.device;
        if let Err(error) = self
            .texture_cache
            .tick_frame(&mut tx, || device.memory_usage())
        {
            tracing::error!(%error, "collecting cached images failed");
        }
    }
}

impl Drop for VulkanRasterizer {
    fn drop(&mut self) {
        let _ = self.scheduler.finish();
        self.draw.destroy(&self.device);
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::GraphicsSettings;
    use lichii_gpu::backend::{ConstBufferBinding, CropRect, ReleaseSize};
    use lichii_gpu::engines::maxwell_3d::DirtyState;
    use lichii_gpu::engines::maxwell_3d::regs::{
        ClearControl, ClearSurface, ColorMask, MAX_CONST_BUFFERS, MAX_SHADER_STAGE,
        RenderTargetConfigBits, RtControl, ScissorTestBits, ScissorTestBits1, TileMode,
        ZetaSizeBits,
    };
    use std::time::Instant;

    use lichii_gpu::clock::{GpuTimestamp, GuestTime};
    use lichii_gpu::gmmu::BigPageSize;
    use lichii_gpu::host1x::Syncpoints;
    use lichii_memory::{DeviceAddressSpace, PhysicalMemory};
    use lichii_texture::PixelFormat;
    use lichii_texture::block_linear::swizzle;

    use super::*;
    use crate::memory::Buffer;
    use crate::{Instance, SurfaceSupport};

    const DRAM: u64 = 16 << 20;
    pub(super) const DADDR: DAddr = 0x1_0000_0000;

    pub(super) struct Fixture {
        pub(super) rasterizer: VulkanRasterizer,
        mailbox: Arc<FrameMailbox>,
        pub(super) memory: Arc<DeviceAddressSpace>,
        syncpoints: Arc<Syncpoints>,
    }

    pub(super) fn fixture(gob_layout: GobLayout) -> Option<Fixture> {
        let instance = match Instance::new(true, SurfaceSupport::None) {
            Ok(instance) => Arc::new(instance),
            Err(error) => {
                eprintln!("skipping: {error}");
                return None;
            }
        };
        let device = match Device::new(instance, &GraphicsSettings::default(), None) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping: {error}");
                return None;
            }
        };
        let physical = Arc::new(PhysicalMemory::new(DRAM).unwrap());
        physical.commit(0, DRAM).unwrap();
        let memory = Arc::new(DeviceAddressSpace::new(physical, 36));
        memory.map(DADDR, 0, DRAM);
        let syncpoints = Arc::new(Syncpoints::new());
        let handles = GpuHandles {
            device: memory.clone(),
            syncpoints: syncpoints.clone(),
            timestamp: Arc::new(GpuTimestamp::new(Arc::new(HostTime(Instant::now())), false)),
        };
        let mailbox = Arc::new(FrameMailbox::default());
        let rasterizer =
            VulkanRasterizer::new(&device, handles, mailbox.clone(), gob_layout).unwrap();
        Some(Fixture {
            rasterizer,
            mailbox,
            memory,
            syncpoints,
        })
    }

    impl Fixture {
        /// Reads the published frame's image back as tightly packed texels.
        fn read_frame(&mut self) -> (GuestFrame, Vec<u8>) {
            let frame = self.mailbox.latest().expect("a published frame");
            let r = &mut self.rasterizer;
            let desc = *frame.image.image.desc();
            let size = u64::from(desc.extent.width * desc.extent.height * 4);
            let readback = Buffer::new(
                &r.device,
                size,
                vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
                "rb",
            )
            .unwrap();
            let cmd = r.scheduler.record().unwrap();
            let image = frame.image.image.raw();
            transition(
                &r.device,
                cmd,
                image,
                COLOR_RANGE,
                ImageAccess::SHADER_READ,
                ImageAccess::TRANSFER_SRC,
            );
            let region = [vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_extent: desc.extent,
                ..Default::default()
            }];
            // SAFETY: the image is in TRANSFER_SRC layout and the buffer holds it exactly.
            unsafe {
                r.device.raw.cmd_copy_image_to_buffer(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.raw(),
                    &region,
                )
            };
            transition(
                &r.device,
                cmd,
                image,
                COLOR_RANGE,
                ImageAccess::TRANSFER_SRC,
                ImageAccess::SHADER_READ,
            );
            crate::commands::transfer_to_all_barrier(&r.device, cmd);
            r.scheduler.flush().unwrap();
            r.scheduler.finish().unwrap();
            (frame, readback.mapped().unwrap().to_vec())
        }

        pub(super) fn assert_no_validation_errors(&self) {
            assert_eq!(
                self.rasterizer.device.instance().validation_errors(),
                0,
                "validation reported errors"
            );
        }
    }

    pub(super) const GPU_VA: u64 = 0x20_0000_0000;

    pub(super) fn gmmu(f: &Fixture) -> Gmmu {
        let gmmu = Gmmu::new(f.memory.clone(), BigPageSize::Size64K);
        gmmu.map(GPU_VA, DADDR, DRAM, 0, false).unwrap();
        gmmu
    }

    /// 3D engine state with colour target 0 at `GPU_VA` (A8B8G8R8, block height `1 << bh`
    /// GOBs) and all channels selected for clearing.
    fn clear_regs(width: u32, height: u32, bh: u32, color: [f32; 4]) -> Box<Regs> {
        let mut regs = Box::new(<Regs as bytemuck::Zeroable>::zeroed());
        let rt = &mut regs.rt[0];
        rt.address_high = (GPU_VA >> 32) as u32;
        rt.address_low = GPU_VA as u32;
        rt.width = width;
        rt.height = height;
        rt.format = 0xd5;
        rt.tile_mode = TileMode(bh << 4);
        rt.bits = RenderTargetConfigBits(1);
        regs.rt_control = RtControl(1);
        regs.color_mask[0] = ColorMask(0x1111);
        regs.clear_surface = ClearSurface(0b1111 << 2);
        regs.clear_color = color;
        regs
    }

    fn clear(f: &mut Fixture, gmmu: &Gmmu, regs: &Regs) {
        clear_layers(f, gmmu, regs, 1);
    }

    fn clear_layers(f: &mut Fixture, gmmu: &Gmmu, regs: &Regs, layer_count: u32) {
        let mut dirty = DirtyState::new();
        let const_buffers = [[ConstBufferBinding::default(); MAX_CONST_BUFFERS]; MAX_SHADER_STAGE];
        let root_tables = [[0; 64]; 8];
        let mut ctx = GraphicsContext {
            channel: ChannelId(0),
            memory: gmmu,
            regs,
            dirty: &mut dirty,
            const_buffers: &const_buffers,
            root_tables: &root_tables,
        };
        f.rasterizer.clear(&mut ctx, layer_count);
    }

    /// Clears use the first scissor, set to `[x0, x1) x [y0, y1)`.
    fn scissor(regs: &mut Regs, x0: u32, x1: u32, y0: u32, y1: u32) {
        regs.clear_control = ClearControl(regs.clear_control.0 | 1 << 8);
        regs.scissor_test[0].enable = 1;
        regs.scissor_test[0].bits = ScissorTestBits(x0 | x1 << 16);
        regs.scissor_test[0].bits1 = ScissorTestBits1(y0 | y1 << 16);
    }

    pub(super) fn guest_linear(f: &Fixture, address: DAddr, tiling: &Tiling) -> Vec<u8> {
        let mut tiled = vec![0; tiling.size_bytes()];
        f.memory.read_bytes(address, &mut tiled).unwrap();
        let mut linear = vec![0; tiling.linear_size_bytes()];
        unswizzle(tiling, &tiled, &mut linear);
        linear
    }

    fn pattern(width: u32, height: u32) -> Vec<u8> {
        (0..width * height * 4)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect()
    }

    struct HostTime(Instant);

    impl GuestTime for HostTime {
        fn global_time_ns(&self) -> u64 {
            self.0.elapsed().as_nanos() as u64 + 1
        }
    }

    fn framebuffer(width: u32, height: u32) -> FramebufferConfig {
        FramebufferConfig {
            address: DADDR,
            width,
            height,
            stride: width * 4,
            format: PixelFormat::A8B8G8R8Unorm,
            block_height_log2: None,
            flip_vertical: false,
            crop: None,
            gob_layout: GobLayout::Tegra,
        }
    }

    #[test]
    fn presents_block_linear_framebuffers_in_both_gob_layouts() {
        for layout in [GobLayout::Tegra, GobLayout::Switch2] {
            let Some(mut f) = fixture(layout) else { return };
            let (width, height) = (70, 40);
            let linear = pattern(width, height);
            let tiling = Tiling::new(width, height, 1, 4, 2, 0, layout);
            let mut tiled = vec![0; tiling.size_bytes()];
            swizzle(&tiling, &linear, &mut tiled);
            f.memory.write_bytes(DADDR, &tiled).unwrap();
            let fb = FramebufferConfig {
                block_height_log2: Some(2),
                gob_layout: layout,
                ..framebuffer(width, height)
            };
            f.rasterizer.present(&fb);
            let (frame, texels) = f.read_frame();
            assert_eq!((frame.width, frame.height), (width, height));
            assert_eq!(frame.uv_rect, [0.0, 0.0, 1.0, 1.0]);
            assert!(texels == linear, "{layout:?}: deswizzled texels differ");
            f.assert_no_validation_errors();
        }
    }

    #[test]
    fn presents_pitch_linear_framebuffers_with_crop_and_flip() {
        let Some(mut f) = fixture(GobLayout::Tegra) else {
            return;
        };
        let (width, height, stride) = (33, 20, 33 * 4 + 60);
        let linear = pattern(width, height);
        for (y, row) in linear.chunks_exact(width as usize * 4).enumerate() {
            f.memory
                .write_bytes(DADDR + y as u64 * u64::from(stride), row)
                .unwrap();
        }
        let fb = FramebufferConfig {
            stride,
            flip_vertical: true,
            crop: Some(CropRect {
                left: 3,
                top: 2,
                right: 23,
                bottom: 12,
            }),
            ..framebuffer(width, height)
        };
        f.rasterizer.present(&fb);
        let (frame, texels) = f.read_frame();
        assert!(texels == linear, "pitch-linear texels differ");
        assert_eq!((frame.width, frame.height), (20, 10));
        let [x, y, w, h] = frame.uv_rect;
        let expect = [3.0 / 33.0, 12.0 / 20.0, 20.0 / 33.0, -10.0 / 20.0];
        assert!(
            [x, y, w, h]
                .iter()
                .zip(expect)
                .all(|(a, b)| (a - b).abs() < 1e-6),
            "{:?}",
            frame.uv_rect
        );
        f.rasterizer.present(&fb);
        f.rasterizer.present(&fb);
        assert!(f.rasterizer.frames.len() <= FRAME_POOL);
        for crop in [
            CropRect {
                left: 80,
                top: 0,
                right: 100,
                bottom: 20,
            },
            CropRect {
                left: 0,
                top: 30,
                right: 33,
                bottom: 40,
            },
            CropRect {
                left: 10,
                top: 5,
                right: 50,
                bottom: 50,
            },
        ] {
            f.rasterizer.present(&FramebufferConfig {
                crop: Some(crop),
                flip_vertical: false,
                ..fb
            });
            let frame = f.mailbox.latest().unwrap();
            let [x, y, w, h] = frame.uv_rect;
            assert!(
                w > 0.0 && h > 0.0 && x + w <= 1.0 && y + h <= 1.0,
                "{crop:?} -> {:?}",
                frame.uv_rect
            );
            assert!(frame.width <= width && frame.height <= height);
        }
        f.assert_no_validation_errors();
    }

    #[test]
    fn unmapped_framebuffers_are_skipped() {
        let Some(mut f) = fixture(GobLayout::Tegra) else {
            return;
        };
        f.rasterizer.present(&FramebufferConfig {
            address: 0x8_0000_0000,
            ..framebuffer(16, 16)
        });
        assert!(f.mailbox.latest().is_none());
    }

    #[test]
    fn clears_decode_engine_state_and_flush_as_swizzled_texels() {
        let Some(mut f) = fixture(GobLayout::Switch2) else {
            return;
        };
        let gmmu = gmmu(&f);
        let (width, height) = (96, 48);
        let mut regs = clear_regs(width, height, 2, [1.0, 0.5, 0.25, 0.0]);
        clear(&mut f, &gmmu, &regs);
        regs.clear_color = [0.0, 0.0, 1.0, 1.0];
        regs.clear_control = ClearControl(1 << 8);
        regs.scissor_test[0].enable = 1;
        regs.scissor_test[0].bits = ScissorTestBits(10 | 40 << 16);
        regs.scissor_test[0].bits1 = ScissorTestBits1(5 | 20 << 16);
        clear(&mut f, &gmmu, &regs);
        regs.clear_surface = ClearSurface(0b1101 << 2);
        regs.clear_control = ClearControl(0);
        clear(&mut f, &gmmu, &regs);

        let tiling = Tiling::new(width, height, 1, 4, 2, 0, GobLayout::Switch2);
        assert!(
            guest_linear(&f, DADDR, &tiling).iter().all(|&b| b == 0),
            "nothing reaches guest memory before a flush"
        );
        f.rasterizer.flush_region(DADDR, tiling.size_bytes() as u64);
        for (i, texel) in guest_linear(&f, DADDR, &tiling)
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
        {
            let (x, y) = (i as u32 % width, i as u32 / width);
            let inside = (10..40).contains(&x) && (5..20).contains(&y);
            let want: [u8; 4] = if inside {
                [0, 0, 255, 255]
            } else {
                [0, 128, 255, 255]
            };
            assert_eq!(*texel, want, "texel ({x}, {y})");
        }

        let depth_va = GPU_VA + 0x10_0000;
        regs.zeta_enable = 1;
        regs.zeta.address_high = (depth_va >> 32) as u32;
        regs.zeta.address_low = depth_va as u32;
        regs.zeta.format = 0x0a;
        regs.zeta.tile_mode = TileMode(2 << 4);
        regs.zeta_size.width = width;
        regs.zeta_size.height = height;
        regs.zeta_size.bits = ZetaSizeBits(1 | 1 << 16);
        regs.clear_surface = ClearSurface(1);
        regs.clear_depth = 0.25;
        clear(&mut f, &gmmu, &regs);
        f.rasterizer
            .flush_region(DADDR + 0x10_0000, tiling.size_bytes() as u64);
        assert!(
            guest_linear(&f, DADDR + 0x10_0000, &tiling)
                .as_chunks::<4>()
                .0
                .iter()
                .all(|&t| f32::from_le_bytes(t) == 0.25),
            "the depth target holds the clear depth"
        );
        f.assert_no_validation_errors();
    }

    #[test]
    fn presents_cleared_render_targets_from_the_cache_at_their_scale() {
        for scale in [ResolutionScale::X1, ResolutionScale::X2] {
            let Some(mut f) = fixture(GobLayout::Tegra) else {
                return;
            };
            f.rasterizer.apply_settings(&GraphicsSettings {
                resolution_scale: scale,
                ..GraphicsSettings::default()
            });
            let gmmu = gmmu(&f);
            let (width, height) = (64, 320);
            clear(
                &mut f,
                &gmmu,
                &clear_regs(width, height, 3, [0.0, 1.0, 1.0, 1.0]),
            );
            let fb = FramebufferConfig {
                block_height_log2: Some(3),
                ..framebuffer(width, height)
            };
            f.rasterizer.present(&fb);
            let (frame, texels) = f.read_frame();
            assert_eq!(
                (frame.width, frame.height),
                (scale.scale(width), scale.scale(height))
            );
            assert_eq!(frame.uv_rect, [0.0, 0.0, 1.0, 1.0]);
            assert!(
                texels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|t| *t == [0, 255, 255, 255]),
                "{scale:?}: the frame holds the cleared image"
            );
            let mut guest = vec![0; 64];
            f.memory.read_bytes(DADDR, &mut guest).unwrap();
            assert!(
                guest.iter().all(|&b| b == 0),
                "presented without a round trip through guest memory"
            );
            f.assert_no_validation_errors();
        }
    }

    #[test]
    fn fences_and_queries_complete_after_submitted_work() {
        let Some(mut f) = fixture(GobLayout::Tegra) else {
            return;
        };
        f.rasterizer
            .signal_fence(FenceAction::IncrementSyncpoint(3));
        assert_eq!(f.syncpoints.guest().value(3).unwrap(), 0);
        f.rasterizer.flush_commands();
        f.rasterizer.release_fences(true);
        assert_eq!(f.syncpoints.guest().value(3).unwrap(), 1);

        let gmmu = Gmmu::new(f.memory.clone(), BigPageSize::Size64K);
        let gpu_va = 0x10_0000;
        gmmu.map(gpu_va, DADDR + 0x80_0000, 0x1000, 0, false)
            .unwrap();
        let query = |payload, size, after_work| Query {
            address: gpu_va,
            kind: QueryKind::Payload,
            payload,
            size,
            after_work,
        };
        f.rasterizer.query(
            ChannelId(0),
            &gmmu,
            &query(0xdead_beef, ReleaseSize::Four, false),
        );
        assert_eq!(
            f.memory.read::<u32>(DADDR + 0x80_0000).unwrap(),
            0xdead_beef
        );
        f.rasterizer.query(
            ChannelId(0),
            &gmmu,
            &query(0x1234_5678_9abc, ReleaseSize::Sixteen, true),
        );
        assert_eq!(
            f.memory.read::<u32>(DADDR + 0x80_0000).unwrap(),
            0xdead_beef
        );
        f.rasterizer.wait_for_idle();
        assert_eq!(
            f.memory.read::<u64>(DADDR + 0x80_0000).unwrap(),
            0x1234_5678_9abc
        );
        assert!(
            f.memory.read::<u64>(DADDR + 0x80_0008).unwrap() > 0,
            "a GPU timestamp follows the value"
        );
        f.assert_no_validation_errors();
    }

    #[test]
    fn masked_colour_clears_keep_the_unwritten_channels() {
        // G and A written, through the colour mask (UNORM) and through the clear's own channel
        // selection (UINT, integer outputs).
        let cases = [
            (
                0xd5,
                ColorMask(0x1010),
                ClearSurface(0b1111 << 2),
                [1.0, 0.5, 0.25, 0.0],
                [0.0, 0.2, 1.0, 1.0],
                [255, 128, 64, 0],
                [255, 51, 64, 255],
            ),
            (
                0xd9,
                ColorMask(0x1111),
                ClearSurface(1 << 3 | 1 << 5),
                [10, 20, 30, 40].map(f32::from_bits),
                [1, 2, 3, 4].map(f32::from_bits),
                [10, 20, 30, 40],
                [10, 2, 30, 4],
            ),
        ];
        for (format, mask, surface, first, second, outside, inside) in cases {
            let Some(mut f) = fixture(GobLayout::Switch2) else {
                return;
            };
            let gmmu = gmmu(&f);
            let (width, height) = (64, 32);
            let mut regs = clear_regs(width, height, 2, first);
            regs.rt[0].format = format;
            clear(&mut f, &gmmu, &regs);
            regs.clear_color = second;
            regs.color_mask[0] = mask;
            regs.clear_surface = surface;
            scissor(&mut regs, 8, 40, 4, 20);
            clear(&mut f, &gmmu, &regs);
            let tiling = Tiling::new(width, height, 1, 4, 2, 0, GobLayout::Switch2);
            f.rasterizer.flush_region(DADDR, tiling.size_bytes() as u64);
            for (i, texel) in guest_linear(&f, DADDR, &tiling)
                .as_chunks::<4>()
                .0
                .iter()
                .enumerate()
            {
                let (x, y) = (i as u32 % width, i as u32 / width);
                let want = if (8..40).contains(&x) && (4..20).contains(&y) {
                    inside
                } else {
                    outside
                };
                assert_eq!(*texel, want, "format {format:#x}, texel ({x}, {y})");
            }
            f.assert_no_validation_errors();
        }
    }

    #[test]
    fn stencil_masked_clears_keep_the_masked_bits() {
        let Some(mut f) = fixture(GobLayout::Tegra) else {
            return;
        };
        let gmmu = gmmu(&f);
        let (width, height) = (32, 16);
        let mut regs = clear_regs(width, height, 1, [0.0; 4]);
        regs.rt_control = RtControl(0);
        regs.zeta_enable = 1;
        regs.zeta.address_high = (GPU_VA >> 32) as u32;
        regs.zeta.address_low = GPU_VA as u32;
        regs.zeta.format = 0x14;
        regs.zeta.tile_mode = TileMode(1 << 4);
        regs.zeta_size.width = width;
        regs.zeta_size.height = height;
        regs.zeta_size.bits = ZetaSizeBits(1 | 1 << 16);
        regs.clear_surface = ClearSurface(0b11);
        regs.clear_depth = 0.5;
        regs.clear_stencil = 0xa5;
        clear(&mut f, &gmmu, &regs);
        let tiling = Tiling::new(width, height, 1, 4, 1, 0, GobLayout::Tegra);
        let read = |f: &mut Fixture| -> Vec<u32> {
            f.rasterizer.flush_region(DADDR, tiling.size_bytes() as u64);
            guest_linear(f, DADDR, &tiling)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&t| u32::from_le_bytes(t))
                .collect()
        };
        let before = read(&mut f);
        assert!(before.iter().all(|&t| t & 0xff == 0xa5));

        regs.clear_surface = ClearSurface(0b10);
        regs.clear_control = ClearControl(1);
        regs.stencil_front_mask = 0x0f;
        regs.clear_stencil = 0x5a;
        clear(&mut f, &gmmu, &regs);
        let after = read(&mut f);
        for (i, (&old, &new)) in before.iter().zip(&after).enumerate() {
            assert_eq!(new & 0xff, 0xaa, "texel {i}: low stencil bits from 0x5a");
            assert_eq!(new >> 8, old >> 8, "texel {i}: depth kept");
        }

        regs.clear_surface = ClearSurface(0b11);
        regs.stencil_front_mask = 0xf0;
        regs.clear_stencil = 0x0f;
        regs.clear_depth = 1.0;
        scissor(&mut regs, 4, 20, 2, 10);
        clear(&mut f, &gmmu, &regs);
        for (i, (&old, &new)) in before.iter().zip(&read(&mut f)).enumerate() {
            let (x, y) = (i as u32 % width, i as u32 / width);
            let want = if (4..20).contains(&x) && (2..10).contains(&y) {
                0xff_ffff << 8 | 0x0a
            } else {
                old & !0xff | 0xaa
            };
            assert_eq!(new, want, "texel ({x}, {y})");
        }
        f.assert_no_validation_errors();
    }

    #[test]
    fn partial_3d_clears_write_only_their_slices() {
        let Some(mut f) = fixture(GobLayout::Switch2) else {
            return;
        };
        let gmmu = gmmu(&f);
        let (width, height, depth) = (32, 16, 4);
        let tiling = Tiling::new(width, height, depth, 4, 1, 0, GobLayout::Switch2);
        let linear = pattern(width, height * depth);
        let mut tiled = vec![0; tiling.size_bytes()];
        swizzle(&tiling, &linear, &mut tiled);
        f.memory.write_bytes(DADDR, &tiled).unwrap();
        let mut regs = clear_regs(width, height, 1, [0.0, 1.0, 0.0, 1.0]);
        regs.rt[0].tile_mode = TileMode(1 << 4 | 1 << 16);
        regs.rt[0].bits = RenderTargetConfigBits(depth);
        regs.clear_surface = ClearSurface(0b1111 << 2 | 1 << 10);
        clear_layers(&mut f, &gmmu, &regs, 2);
        regs.clear_surface = ClearSurface(0b0001 << 2 | 3 << 10);
        regs.clear_color = [1.0, 0.0, 0.0, 0.0];
        scissor(&mut regs, 4, 12, 2, 6);
        clear_layers(&mut f, &gmmu, &regs, 1);
        f.rasterizer.flush_region(DADDR, tiling.size_bytes() as u64);
        let got = guest_linear(&f, DADDR, &tiling);
        for (i, (texel, old)) in got
            .as_chunks::<4>()
            .0
            .iter()
            .zip(linear.as_chunks::<4>().0)
            .enumerate()
        {
            let i = i as u32;
            let (x, y, z) = (i % width, i / width % height, i / (width * height));
            let want = match z {
                1 | 2 => [0, 255, 0, 255],
                3 if (4..12).contains(&x) && (2..6).contains(&y) => [255, old[1], old[2], old[3]],
                _ => *old,
            };
            assert_eq!(*texel, want, "texel ({x}, {y}, {z})");
        }
        f.assert_no_validation_errors();
    }
}
