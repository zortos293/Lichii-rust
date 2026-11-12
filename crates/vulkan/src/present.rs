//! The contract between the renderer and the frontend, and the presenter behind it.
//!
//! The frontend owns the window and all UI logic (game list, settings, frame view) in egui. Each
//! host frame it hands the [`Presenter`] a [`UiFrame`]; the presenter draws the latest guest
//! frame (scaled and filtered per `GraphicsSettings`), then the UI on top, and presents.
//!
//! Guest frames arrive from the GPU thread through a [`FrameMailbox`]: the rasterizer publishes
//! each finished frame with the timeline value that completes it, and the presenter's submission
//! waits on that value on the GPU, so neither thread blocks the other.

mod passes;
mod ui;

use std::sync::Arc;
use std::time::{Duration, Instant};

use ash::vk;
use lichii_common::perf::{FrameSample, PerfStats};
use lichii_common::settings::{AntiAliasing, AspectRatio, GraphicsSettings, ScalingFilter};
use parking_lot::Mutex;

use crate::commands::{COLOR_RANGE, ImageAccess, transition};
use crate::descriptors::DescriptorAllocator;
use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::memory::{Image, ImageDesc, ImageView};
use crate::scheduler::{Scheduler, SemaphoreWait};
use crate::staging::StagingRing;
use crate::swapchain::Swapchain;
use passes::{MAX_INPUTS, Pass, PassParams, Passes};
use ui::UiRenderer;

/// egui output for one host frame, already tessellated.
pub struct UiFrame {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

/// Where the guest frame goes inside the window, in physical pixels. The frontend computes it
/// from the aspect-ratio setting and any UI chrome, usually with [`Viewport::fit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Viewport {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    /// The largest rectangle with `aspect` centred in an `area_width` x `area_height` area at
    /// (`x`, `y`); `Stretch` fills the area.
    pub fn fit(aspect: AspectRatio, x: i32, y: i32, area_width: u32, area_height: u32) -> Self {
        let ratio = match aspect {
            AspectRatio::R16x9 => 16.0 / 9.0,
            AspectRatio::R4x3 => 4.0 / 3.0,
            AspectRatio::R21x9 => 21.0 / 9.0,
            AspectRatio::R16x10 => 16.0 / 10.0,
            AspectRatio::Stretch => {
                return Self { x, y, width: area_width, height: area_height };
            }
        };
        let (w, h) = (area_width as f64, area_height as f64);
        let (width, height) = if w / h > ratio {
            ((h * ratio).round(), h)
        } else {
            (w, (w / ratio).round())
        };
        let (width, height) = (width as u32, height as u32);
        Self {
            x: x + ((area_width - width) / 2) as i32,
            y: y + ((area_height - height) / 2) as i32,
            width,
            height,
        }
    }
}

/// A host image holding one finished guest frame, in SHADER_READ_ONLY_OPTIMAL layout.
pub(crate) struct FrameImage {
    pub(crate) image: Image,
    pub(crate) view: ImageView,
}

/// A guest frame the GPU thread finished.
#[derive(Clone)]
pub struct GuestFrame {
    pub(crate) image: Arc<FrameImage>,
    /// Timeline semaphore and value that complete the frame's rendering.
    pub(crate) ready: (vk::Semaphore, u64),
    /// Visible part in UV space: origin and extent (negative height flips).
    pub(crate) uv_rect: [f32; 4],
    /// Size of the visible part in texels.
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// The newest guest frame, handed from the GPU thread to the presenter.
#[derive(Default)]
pub struct FrameMailbox {
    latest: Mutex<Option<GuestFrame>>,
}

impl FrameMailbox {
    pub(crate) fn publish(&self, frame: GuestFrame) {
        *self.latest.lock() = Some(frame);
    }

    fn latest(&self) -> Option<GuestFrame> {
        self.latest.lock().clone()
    }
}

/// A color target the presenter renders into between passes.
struct Target {
    image: Image,
    view: ImageView,
    extent: vk::Extent2D,
}

impl Target {
    fn new(device: &Arc<Device>, extent: vk::Extent2D, format: vk::Format, name: &str) -> Result<Self> {
        let image = Image::new(
            device,
            &ImageDesc::d2(
                format,
                extent.width,
                extent.height,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ),
            name,
        )?;
        let view = image.view(vk::ImageViewType::TYPE_2D)?;
        Ok(Self { image, view, extent })
    }
}

/// Keeps `slot` at `extent`, retiring a target of another size.
fn ensure_target<'a>(
    device: &Arc<Device>,
    scheduler: &mut Scheduler,
    slot: &'a mut Option<Target>,
    extent: vk::Extent2D,
    format: vk::Format,
    name: &str,
) -> Result<&'a Target> {
    if slot.as_ref().is_some_and(|t| t.extent != extent) {
        let old = slot.take().expect("checked");
        scheduler.retire((old.view, old.image));
    }
    if slot.is_none() {
        *slot = Some(Target::new(device, extent, format, name)?);
    }
    Ok(slot.as_ref().expect("just ensured"))
}

const INTERMEDIATE: vk::Format = vk::Format::R8G8B8A8_UNORM;
const FRAMES_IN_FLIGHT: usize = 2;
const VRAM_INTERVAL: Duration = Duration::from_millis(500);

/// FSR RCAS sharpness in stops from the 0..100 setting (100 = sharpest).
fn rcas_stops(sharpness: u8) -> f32 {
    2.0 * (1.0 - f32::from(sharpness.min(100)) / 100.0)
}

fn scaling_pass(filter: ScalingFilter) -> Pass {
    match filter {
        ScalingFilter::NearestNeighbor | ScalingFilter::Bilinear => Pass::Copy,
        ScalingFilter::Bicubic => Pass::Bicubic,
        ScalingFilter::Gaussian => Pass::Gaussian,
        ScalingFilter::Lanczos => Pass::Lanczos,
        ScalingFilter::ScaleForce => Pass::ScaleForce,
        ScalingFilter::Fsr => Pass::FsrEasu,
        ScalingFilter::Area => Pass::Area,
        ScalingFilter::Mmpx => Pass::Mmpx,
    }
}

/// Draws guest frames and the UI to the window. Lives on the frontend's thread and owns its own
/// scheduler; it shares only the device (and its queue lock) with the GPU thread.
pub struct Presenter {
    device: Arc<Device>,
    swapchain: Swapchain,
    scheduler: Scheduler,
    descriptors: DescriptorAllocator,
    staging: StagingRing,
    passes: Passes,
    ui: UiRenderer,
    mailbox: Arc<FrameMailbox>,
    perf: Arc<PerfStats>,
    settings: GraphicsSettings,
    nearest: vk::Sampler,
    linear: vk::Sampler,
    smaa_area: FrameImage,
    smaa_search: FrameImage,
    aa_target: Option<Target>,
    smaa_edges: Option<Target>,
    smaa_weights: Option<Target>,
    fsr_target: Option<Target>,
    in_flight: [(u64, Option<GuestFrame>); FRAMES_IN_FLIGHT],
    frame_slot: usize,
    timestamps: Option<vk::QueryPool>,
    last_present: Option<Instant>,
    last_vram: Option<Instant>,
}

impl Presenter {
    pub fn new(
        device: &Arc<Device>,
        swapchain: Swapchain,
        settings: &GraphicsSettings,
        mailbox: Arc<FrameMailbox>,
        perf: Arc<PerfStats>,
    ) -> Result<Self> {
        let mut scheduler = Scheduler::new(device)?;
        let mut staging = StagingRing::new(device, 16 << 20)?;
        let sampler = |filter: vk::Filter| {
            let info = vk::SamplerCreateInfo::default()
                .mag_filter(filter)
                .min_filter(filter)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(0.25);
            // SAFETY: valid create info.
            unsafe { device.raw.create_sampler(&info, None).call("vkCreateSampler") }
        };
        let nearest = sampler(vk::Filter::NEAREST)?;
        let linear = sampler(vk::Filter::LINEAR)?;
        let smaa_area = upload_lut(
            device,
            &mut scheduler,
            &mut staging,
            crate::shaders::SMAA_AREA_TEX,
            crate::shaders::SMAA_AREA_TEX_SIZE,
            vk::Format::R8G8_UNORM,
        )?;
        let smaa_search = upload_lut(
            device,
            &mut scheduler,
            &mut staging,
            crate::shaders::SMAA_SEARCH_TEX,
            crate::shaders::SMAA_SEARCH_TEX_SIZE,
            vk::Format::R8_UNORM,
        )?;
        let timestamps = (device.properties.timestamp_valid_bits > 0).then(|| {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2 * FRAMES_IN_FLIGHT as u32);
            // SAFETY: valid create info.
            unsafe { device.raw.create_query_pool(&info, None) }.ok()
        });
        Ok(Self {
            device: device.clone(),
            swapchain,
            descriptors: DescriptorAllocator::new(device)?,
            passes: Passes::new(device)?,
            ui: UiRenderer::new(device)?,
            scheduler,
            staging,
            mailbox,
            perf,
            settings: settings.clone(),
            nearest,
            linear,
            smaa_area,
            smaa_search,
            aa_target: None,
            smaa_edges: None,
            smaa_weights: None,
            fsr_target: None,
            in_flight: [(0, None), (0, None)],
            frame_slot: 0,
            timestamps: timestamps.flatten(),
            last_present: None,
            last_vram: None,
        })
    }

    /// Applies new graphics settings (filter, AA, sharpness, present mode) from the next frame.
    pub fn set_settings(&mut self, settings: &GraphicsSettings) {
        self.swapchain.set_present_mode(settings.present_mode);
        self.settings = settings.clone();
    }

    /// Draws one host frame into a `window_width` x `window_height` window: the newest guest
    /// frame into `viewport`, then `ui` over the whole window, and presents it.
    pub fn draw(&mut self, window_width: u32, window_height: u32, viewport: Viewport, ui: &UiFrame) -> Result<()> {
        let slot = self.frame_slot;
        self.frame_slot = (slot + 1) % FRAMES_IN_FLIGHT;
        self.scheduler.wait(self.in_flight[slot].0)?;
        let gpu_time = self.read_timestamps(slot);
        self.in_flight[slot].1 = None;

        self.ui.upload(ui, &mut self.scheduler, &mut self.staging)?;
        let Some(frame) = self.swapchain.acquire(window_width, window_height)? else {
            self.ui.free(ui, &mut self.scheduler);
            return Ok(());
        };
        let guest = self.mailbox.latest();
        let cmd = self.scheduler.record()?;
        if let Some(pool) = self.timestamps {
            // SAFETY: `cmd` is recording outside any rendering.
            unsafe {
                self.device.raw.cmd_reset_query_pool(cmd, pool, 2 * slot as u32, 2);
                self.device
                    .raw
                    .cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, pool, 2 * slot as u32);
            }
        }

        let mut waits = vec![SemaphoreWait {
            semaphore: frame.acquired,
            value: 0,
            stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        }];
        let scale_input = match &guest {
            Some(guest) => {
                waits.push(SemaphoreWait {
                    semaphore: guest.ready.0,
                    value: guest.ready.1,
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                });
                Some(self.prepare_source(cmd, guest, viewport)?)
            }
            None => None,
        };

        transition(&self.device, cmd, frame.image, COLOR_RANGE, ImageAccess::UNDEFINED, ImageAccess::COLOR_ATTACHMENT);
        begin_rendering(&self.device, cmd, frame.view, frame.extent, true);
        if let (Some((view, sampler, uv_rect, source_size)), Some(rect)) = (scale_input, clip_viewport(viewport, frame.extent)) {
            let pass = match scaling_pass(self.settings.scaling_filter) {
                Pass::FsrEasu => Pass::FsrRcas,
                pass => pass,
            };
            let set = self.input_set(&[(view, sampler)])?;
            let params = PassParams {
                src_rect: uv_rect,
                source_size,
                output_size: [rect.extent.width as f32, rect.extent.height as f32],
                sharpness: rcas_stops(self.settings.fsr_sharpness),
                pad: [0.0; 3],
            };
            self.passes.draw(cmd, pass, frame.format, set, rect, &params)?;
        }
        self.ui.draw(ui, cmd, frame.format, frame.extent, &mut self.scheduler, &mut self.descriptors)?;
        // SAFETY: matches the begin above.
        unsafe { self.device.raw.cmd_end_rendering(cmd) };
        transition(&self.device, cmd, frame.image, COLOR_RANGE, ImageAccess::COLOR_ATTACHMENT, ImageAccess::PRESENT);
        if let Some(pool) = self.timestamps {
            // SAFETY: as above.
            unsafe {
                self.device
                    .raw
                    .cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::BOTTOM_OF_PIPE, pool, 2 * slot as u32 + 1)
            };
        }
        let tick = self.scheduler.flush_with(&waits, &[frame.rendered])?;
        self.in_flight[slot] = (tick, guest.clone());
        self.swapchain.present(&frame)?;
        self.ui.free(ui, &mut self.scheduler);
        self.record_perf(gpu_time, guest.as_ref());
        Ok(())
    }

    /// Runs anti-aliasing and the first FSR pass into intermediate targets; returns what the
    /// final pass samples: view, sampler, UV window and its size in texels.
    fn prepare_source(
        &mut self,
        cmd: vk::CommandBuffer,
        guest: &GuestFrame,
        viewport: Viewport,
    ) -> Result<(vk::ImageView, vk::Sampler, [f32; 4], [f32; 2])> {
        let source_extent = vk::Extent2D { width: guest.width, height: guest.height };
        let source_size = [guest.width as f32, guest.height as f32];
        let mut view = guest.image.view.raw();
        let mut uv_rect = guest.uv_rect;
        let full = [0.0, 0.0, 1.0, 1.0];
        let params = |src_rect, output: vk::Extent2D| PassParams {
            src_rect,
            source_size,
            output_size: [output.width as f32, output.height as f32],
            sharpness: rcas_stops(self.settings.fsr_sharpness),
            pad: [0.0; 3],
        };
        match self.settings.anti_aliasing {
            AntiAliasing::None => {}
            AntiAliasing::Fxaa => {
                let set = self.input_set(&[(view, self.linear)])?;
                let target = ensure_target(&self.device, &mut self.scheduler, &mut self.aa_target, source_extent, INTERMEDIATE, "fxaa")?;
                let (target_image, target_view) = (target.image.raw(), target.view.raw());
                render_pass(&self.device, &mut self.passes, cmd, Pass::Fxaa, set, target_image, target_view, source_extent, &params(uv_rect, source_extent))?;
                view = target_view;
                uv_rect = full;
            }
            AntiAliasing::Smaa => {
                let input = (view, self.linear);
                let edges = ensure_target(&self.device, &mut self.scheduler, &mut self.smaa_edges, source_extent, INTERMEDIATE, "smaa edges")?;
                let (edges_image, edges_view) = (edges.image.raw(), edges.view.raw());
                let set = self.input_set(&[input])?;
                render_pass(&self.device, &mut self.passes, cmd, Pass::SmaaEdge, set, edges_image, edges_view, source_extent, &params(uv_rect, source_extent))?;
                let weights = ensure_target(&self.device, &mut self.scheduler, &mut self.smaa_weights, source_extent, INTERMEDIATE, "smaa weights")?;
                let (weights_image, weights_view) = (weights.image.raw(), weights.view.raw());
                let set = self.input_set(&[
                    (edges_view, self.linear),
                    (self.smaa_area.view.raw(), self.linear),
                    (self.smaa_search.view.raw(), self.nearest),
                ])?;
                render_pass(&self.device, &mut self.passes, cmd, Pass::SmaaBlend, set, weights_image, weights_view, source_extent, &params(full, source_extent))?;
                let out = ensure_target(&self.device, &mut self.scheduler, &mut self.aa_target, source_extent, INTERMEDIATE, "smaa output")?;
                let (out_image, out_view) = (out.image.raw(), out.view.raw());
                let set = self.input_set(&[input, (weights_view, self.linear)])?;
                render_pass(&self.device, &mut self.passes, cmd, Pass::SmaaNeighborhood, set, out_image, out_view, source_extent, &params(uv_rect, source_extent))?;
                view = out_view;
                uv_rect = full;
            }
        }
        let sampler = match self.settings.scaling_filter {
            ScalingFilter::NearestNeighbor | ScalingFilter::Mmpx => self.nearest,
            _ => self.linear,
        };
        if self.settings.scaling_filter != ScalingFilter::Fsr || viewport.width == 0 || viewport.height == 0 {
            return Ok((view, sampler, uv_rect, source_size));
        }
        let output = vk::Extent2D { width: viewport.width, height: viewport.height };
        let set = self.input_set(&[(view, self.linear)])?;
        let target = ensure_target(&self.device, &mut self.scheduler, &mut self.fsr_target, output, INTERMEDIATE, "fsr easu")?;
        let (target_image, target_view) = (target.image.raw(), target.view.raw());
        render_pass(&self.device, &mut self.passes, cmd, Pass::FsrEasu, set, target_image, target_view, output, &params(uv_rect, output))?;
        Ok((target_view, self.linear, full, [output.width as f32, output.height as f32]))
    }

    fn input_set(&mut self, inputs: &[(vk::ImageView, vk::Sampler)]) -> Result<vk::DescriptorSet> {
        let set = self.descriptors.allocate(&mut self.scheduler, self.passes.set_layout)?;
        let infos: Vec<[vk::DescriptorImageInfo; 1]> = (0..MAX_INPUTS)
            .map(|i| {
                let (view, sampler) = inputs.get(i).copied().unwrap_or(inputs[0]);
                [vk::DescriptorImageInfo {
                    sampler,
                    image_view: view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(info)
            })
            .collect();
        // SAFETY: the set is fresh; every view is live until this frame's tick completes.
        unsafe { self.device.raw.update_descriptor_sets(&writes, &[]) };
        Ok(set)
    }

    fn read_timestamps(&mut self, slot: usize) -> Option<Duration> {
        let pool = self.timestamps?;
        if self.in_flight[slot].0 == 0 {
            return None;
        }
        let mut values = [0u64; 2];
        // SAFETY: the slot's submission completed, so both queries are available.
        unsafe {
            self.device
                .raw
                .get_query_pool_results(pool, 2 * slot as u32, &mut values, vk::QueryResultFlags::TYPE_64)
        }
        .ok()?;
        let bits = self.device.properties.timestamp_valid_bits;
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        let ticks = values[1].wrapping_sub(values[0]) & mask;
        Some(Duration::from_nanos((ticks as f64 * f64::from(self.device.properties.timestamp_period)) as u64))
    }

    fn record_perf(&mut self, gpu_time: Option<Duration>, guest: Option<&GuestFrame>) {
        let now = Instant::now();
        if let Some(last) = self.last_present {
            self.perf.record_present(FrameSample {
                presented_at: now,
                frame_time: now - last,
                gpu_time,
            });
        }
        self.last_present = Some(now);
        if self.last_vram.is_none_or(|t| now - t >= VRAM_INTERVAL) {
            let usage = self.device.memory_usage();
            self.perf.set_vram(usage.used, usage.budget);
            self.last_vram = Some(now);
        }
        if let Some(guest) = guest {
            self.perf.set_render_size(guest.width, guest.height);
        }
    }
}

impl Drop for Presenter {
    fn drop(&mut self) {
        let _ = self.scheduler.finish();
        // SAFETY: all of the presenter's work finished.
        unsafe {
            if let Some(pool) = self.timestamps {
                self.device.raw.destroy_query_pool(pool, None);
            }
            self.device.raw.destroy_sampler(self.nearest, None);
            self.device.raw.destroy_sampler(self.linear, None);
        }
    }
}

/// The part of `viewport` inside the window, as a scissor-safe rectangle.
fn clip_viewport(viewport: Viewport, extent: vk::Extent2D) -> Option<vk::Rect2D> {
    let x0 = viewport.x.max(0);
    let y0 = viewport.y.max(0);
    let x1 = (viewport.x + viewport.width as i32).min(extent.width as i32);
    let y1 = (viewport.y + viewport.height as i32).min(extent.height as i32);
    (x1 > x0 && y1 > y0 && viewport.x >= 0 && viewport.y >= 0).then(|| vk::Rect2D {
        offset: vk::Offset2D { x: x0, y: y0 },
        extent: vk::Extent2D { width: (x1 - x0) as u32, height: (y1 - y0) as u32 },
    })
}

fn begin_rendering(device: &Device, cmd: vk::CommandBuffer, view: vk::ImageView, extent: vk::Extent2D, clear: bool) {
    let attachment = [vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(if clear { vk::AttachmentLoadOp::CLEAR } else { vk::AttachmentLoadOp::DONT_CARE })
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
        })];
    let info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent })
        .layer_count(1)
        .color_attachments(&attachment);
    // SAFETY: `view` is in COLOR_ATTACHMENT_OPTIMAL layout.
    unsafe { device.raw.cmd_begin_rendering(cmd, &info) };
}

/// Renders one fullscreen pass into a whole intermediate target and leaves the target ready to
/// be sampled.
#[allow(clippy::too_many_arguments)]
fn render_pass(
    device: &Device,
    passes: &mut Passes,
    cmd: vk::CommandBuffer,
    pass: Pass,
    set: vk::DescriptorSet,
    image: vk::Image,
    view: vk::ImageView,
    extent: vk::Extent2D,
    params: &PassParams,
) -> Result<()> {
    transition(device, cmd, image, COLOR_RANGE, ImageAccess::UNDEFINED, ImageAccess::COLOR_ATTACHMENT);
    begin_rendering(device, cmd, view, extent, false);
    let rect = vk::Rect2D { offset: vk::Offset2D::default(), extent };
    let drawn = passes.draw(cmd, pass, INTERMEDIATE, set, rect, params);
    // SAFETY: matches the begin above.
    unsafe { device.raw.cmd_end_rendering(cmd) };
    transition(device, cmd, image, COLOR_RANGE, ImageAccess::COLOR_ATTACHMENT, ImageAccess::SHADER_READ);
    drawn
}

/// Uploads a small lookup table (SMAA area/search textures) once.
fn upload_lut(
    device: &Arc<Device>,
    scheduler: &mut Scheduler,
    staging: &mut StagingRing,
    bytes: &[u8],
    (width, height): (u32, u32),
    format: vk::Format,
) -> Result<FrameImage> {
    let image = Image::new(
        device,
        &ImageDesc::d2(format, width, height, vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
        "smaa lut",
    )?;
    let view = image.view(vk::ImageViewType::TYPE_2D)?;
    let slice = staging.alloc(scheduler, bytes.len() as u64, 4)?;
    slice.data.copy_from_slice(bytes);
    let (buffer, offset) = (slice.buffer, slice.offset);
    let cmd = scheduler.record()?;
    transition(device, cmd, image.raw(), COLOR_RANGE, ImageAccess::UNDEFINED, ImageAccess::TRANSFER_DST);
    let region = [vk::BufferImageCopy {
        buffer_offset: offset,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D { width, height, depth: 1 },
    }];
    // SAFETY: the image is in TRANSFER_DST layout; the region covers it exactly.
    unsafe {
        device
            .raw
            .cmd_copy_buffer_to_image(cmd, buffer, image.raw(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, &region)
    };
    transition(device, cmd, image.raw(), COLOR_RANGE, ImageAccess::TRANSFER_DST, ImageAccess::SHADER_READ);
    Ok(FrameImage { image, view })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_letterboxes_and_pillarboxes() {
        assert_eq!(
            Viewport::fit(AspectRatio::R16x9, 0, 0, 1920, 1200),
            Viewport { x: 0, y: 60, width: 1920, height: 1080 }
        );
        assert_eq!(
            Viewport::fit(AspectRatio::R4x3, 10, 0, 1920, 1080),
            Viewport { x: 250, y: 0, width: 1440, height: 1080 }
        );
        assert_eq!(
            Viewport::fit(AspectRatio::Stretch, 0, 20, 800, 600),
            Viewport { x: 0, y: 20, width: 800, height: 600 }
        );
    }

    #[test]
    fn sharpness_maps_to_rcas_stops() {
        assert_eq!(rcas_stops(100), 0.0);
        assert_eq!(rcas_stops(0), 2.0);
        assert_eq!(rcas_stops(250), 0.0);
    }

    #[test]
    fn viewports_outside_the_window_are_clipped() {
        let extent = vk::Extent2D { width: 100, height: 50 };
        let inside = clip_viewport(Viewport { x: 10, y: 5, width: 200, height: 20 }, extent).unwrap();
        assert_eq!((inside.extent.width, inside.extent.height), (90, 20));
        assert!(clip_viewport(Viewport { x: 0, y: 0, width: 0, height: 10 }, extent).is_none());
    }
}
