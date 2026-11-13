//! The contract between the renderer and the frontend, and the presenter behind it.
//!
//! The frontend owns the window and all UI logic (game list, settings, frame view) in egui. Each
//! host frame it hands the [`Presenter`] a [`UiFrame`]; the presenter draws the latest guest
//! frame (scaled and filtered per `GraphicsSettings`), then the UI on top, and presents.
//!
//! Guest frames arrive from the GPU thread through a [`FrameMailbox`]: the rasterizer publishes
//! each finished frame with the timeline value that completes it, and the presenter's submission
//! waits on that value on the GPU, so neither thread blocks the other.

pub(crate) mod passes;
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
use crate::scheduler::{Scheduler, SemaphoreWait, Timeline};
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
                return Self {
                    x,
                    y,
                    width: area_width,
                    height: area_height,
                };
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
    pub(crate) ready: (Arc<Timeline>, u64),
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
    /// Replaces the frame the presenter draws next.
    pub fn publish(&self, frame: GuestFrame) {
        *self.latest.lock() = Some(frame);
    }

    pub(crate) fn latest(&self) -> Option<GuestFrame> {
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
    fn new(
        device: &Arc<Device>,
        extent: vk::Extent2D,
        format: vk::Format,
        name: &str,
    ) -> Result<Self> {
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
        let view = image.view(vk::ImageViewType::TYPE_2D, vk::ComponentMapping::default())?;
        Ok(Self {
            image,
            view,
            extent,
        })
    }
}

/// The intermediate targets the presenter renders passes into.
#[derive(Clone, Copy)]
enum Slot {
    Aa,
    SmaaInput,
    SmaaEdges,
    SmaaWeights,
    Fsr,
}

/// An intermediate target the previous frame sampled; rendering discards its contents.
const SAMPLED_LAST_FRAME: ImageAccess = ImageAccess {
    layout: vk::ImageLayout::UNDEFINED,
    ..ImageAccess::SHADER_READ
};

/// The image a frame is composed into: a swapchain image, or an offscreen target in tests.
#[derive(Clone, Copy)]
struct Output {
    image: vk::Image,
    view: vk::ImageView,
    format: vk::Format,
    extent: vk::Extent2D,
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
    smaa_input: Option<Target>,
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
            unsafe {
                device
                    .raw
                    .create_sampler(&info, None)
                    .call("vkCreateSampler")
            }
        };
        let nearest = sampler(vk::Filter::NEAREST)?;
        let linear = sampler(vk::Filter::LINEAR)?;
        let smaa_area = upload_image(
            device,
            &mut scheduler,
            &mut staging,
            crate::shaders::SMAA_AREA_TEX,
            (
                crate::shaders::SMAA_AREA_TEX_WIDTH,
                crate::shaders::SMAA_AREA_TEX_HEIGHT,
            ),
            crate::shaders::SMAA_AREA_TEX_FORMAT,
            vk::ComponentMapping::default(),
        )?;
        let smaa_search = upload_image(
            device,
            &mut scheduler,
            &mut staging,
            crate::shaders::SMAA_SEARCH_TEX,
            (
                crate::shaders::SMAA_SEARCH_TEX_WIDTH,
                crate::shaders::SMAA_SEARCH_TEX_HEIGHT,
            ),
            crate::shaders::SMAA_SEARCH_TEX_FORMAT,
            vk::ComponentMapping::default(),
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
            smaa_input: None,
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
    pub fn draw(
        &mut self,
        window_width: u32,
        window_height: u32,
        viewport: Viewport,
        mut ui: UiFrame,
    ) -> Result<()> {
        let drawn = self.draw_frame(window_width, window_height, viewport, &ui);
        self.ui.free(&mut ui, &mut self.scheduler);
        drawn
    }

    fn draw_frame(
        &mut self,
        window_width: u32,
        window_height: u32,
        viewport: Viewport,
        ui: &UiFrame,
    ) -> Result<()> {
        let slot = self.frame_slot;
        self.frame_slot = (slot + 1) % FRAMES_IN_FLIGHT;
        self.scheduler.wait(self.in_flight[slot].0)?;
        let gpu_time = self.read_timestamps(slot);
        self.in_flight[slot].1 = None;

        self.ui.upload(ui, &mut self.scheduler, &mut self.staging)?;
        let Some(frame) = self.swapchain.acquire(window_width, window_height)? else {
            return Ok(());
        };
        let guest = self.mailbox.latest();
        let cmd = self.scheduler.record()?;
        if let Some(pool) = self.timestamps {
            // SAFETY: `cmd` is recording outside any rendering.
            unsafe {
                self.device
                    .raw
                    .cmd_reset_query_pool(cmd, pool, 2 * slot as u32, 2);
                self.device.raw.cmd_write_timestamp2(
                    cmd,
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    pool,
                    2 * slot as u32,
                );
            }
        }

        let mut waits = vec![SemaphoreWait {
            semaphore: frame.acquired,
            value: 0,
            stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        }];
        if let Some(guest) = &guest {
            waits.push(SemaphoreWait {
                semaphore: guest.ready.0.raw(),
                value: guest.ready.1,
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            });
        }
        let output = Output {
            image: frame.image,
            view: frame.view,
            format: frame.format,
            extent: frame.extent,
        };
        self.compose(cmd, guest.as_ref(), viewport, ui, output)?;
        transition(
            &self.device,
            cmd,
            frame.image,
            COLOR_RANGE,
            ImageAccess::COLOR_ATTACHMENT,
            ImageAccess::PRESENT,
        );
        if let Some(pool) = self.timestamps {
            // SAFETY: as above.
            unsafe {
                self.device.raw.cmd_write_timestamp2(
                    cmd,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    pool,
                    2 * slot as u32 + 1,
                )
            };
        }
        let tick = self.scheduler.flush_with(&waits, &[frame.rendered])?;
        self.in_flight[slot] = (tick, guest.clone());
        self.swapchain.present(&frame)?;
        self.record_perf(gpu_time, guest.as_ref());
        Ok(())
    }

    /// Records the guest frame scaled into `viewport` and the UI over the whole of `output`,
    /// leaving `output` in COLOR_ATTACHMENT_OPTIMAL layout.
    fn compose(
        &mut self,
        cmd: vk::CommandBuffer,
        guest: Option<&GuestFrame>,
        viewport: Viewport,
        ui: &UiFrame,
        output: Output,
    ) -> Result<()> {
        let scale_input = guest
            .map(|guest| self.prepare_source(cmd, guest, viewport))
            .transpose()?;
        transition(
            &self.device,
            cmd,
            output.image,
            COLOR_RANGE,
            ImageAccess::UNDEFINED,
            ImageAccess::COLOR_ATTACHMENT,
        );
        begin_rendering(
            &self.device,
            cmd,
            output.view,
            output.extent,
            Some([0.0, 0.0, 0.0, 1.0]),
        );
        if let (Some((view, sampler, uv_rect, source_size)), Some(rect)) =
            (scale_input, clip_viewport(viewport, output.extent))
        {
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
            self.passes
                .draw(cmd, pass, output.format, set, rect, &params)?;
        }
        self.ui.draw(
            ui,
            cmd,
            output.format,
            output.extent,
            &mut self.scheduler,
            &mut self.descriptors,
        )?;
        // SAFETY: matches the begin above.
        unsafe { self.device.raw.cmd_end_rendering(cmd) };
        Ok(())
    }

    /// Runs anti-aliasing and the first FSR pass into intermediate targets; returns what the
    /// final pass samples: view, sampler, UV window and the bound image's size in texels.
    fn prepare_source(
        &mut self,
        cmd: vk::CommandBuffer,
        guest: &GuestFrame,
        viewport: Viewport,
    ) -> Result<(vk::ImageView, vk::Sampler, [f32; 4], [f32; 2])> {
        let image = guest.image.image.desc().extent;
        let crop = vk::Extent2D {
            width: guest.width,
            height: guest.height,
        };
        let crop_size = [crop.width as f32, crop.height as f32];
        let full = [0.0, 0.0, 1.0, 1.0];
        let sharpness = rcas_stops(self.settings.fsr_sharpness);
        let params = |src_rect, source_size, output: vk::Extent2D| PassParams {
            src_rect,
            source_size,
            output_size: [output.width as f32, output.height as f32],
            sharpness,
            pad: [0.0; 3],
        };
        let mut view = guest.image.view.raw();
        let mut uv_rect = guest.uv_rect;
        let mut size = [image.width as f32, image.height as f32];
        let (linear, nearest) = (self.linear, self.nearest);
        match self.settings.anti_aliasing {
            AntiAliasing::None => {}
            AntiAliasing::Fxaa => {
                let input = [(view, linear)];
                view = self.pass_into(
                    cmd,
                    Pass::Fxaa,
                    &input,
                    Slot::Aa,
                    crop,
                    &params(uv_rect, size, crop),
                )?;
                (uv_rect, size) = (full, crop_size);
            }
            AntiAliasing::Smaa => {
                if uv_rect != full {
                    let input = [(view, nearest)];
                    view = self.pass_into(
                        cmd,
                        Pass::Copy,
                        &input,
                        Slot::SmaaInput,
                        crop,
                        &params(uv_rect, size, crop),
                    )?;
                }
                let normalized = params(full, crop_size, crop);
                let edges = self.pass_into(
                    cmd,
                    Pass::SmaaEdge,
                    &[(view, linear)],
                    Slot::SmaaEdges,
                    crop,
                    &normalized,
                )?;
                let tables = [
                    (edges, linear),
                    (self.smaa_area.view.raw(), linear),
                    (self.smaa_search.view.raw(), nearest),
                ];
                let weights = self.pass_into(
                    cmd,
                    Pass::SmaaBlend,
                    &tables,
                    Slot::SmaaWeights,
                    crop,
                    &normalized,
                )?;
                let input = [(view, linear), (weights, linear)];
                view = self.pass_into(
                    cmd,
                    Pass::SmaaNeighborhood,
                    &input,
                    Slot::Aa,
                    crop,
                    &normalized,
                )?;
                (uv_rect, size) = (full, crop_size);
            }
        }
        let sampler = match self.settings.scaling_filter {
            ScalingFilter::NearestNeighbor | ScalingFilter::Mmpx => nearest,
            _ => linear,
        };
        if self.settings.scaling_filter != ScalingFilter::Fsr
            || viewport.width == 0
            || viewport.height == 0
        {
            return Ok((view, sampler, uv_rect, size));
        }
        let output = vk::Extent2D {
            width: viewport.width,
            height: viewport.height,
        };
        let input = [(view, linear)];
        let view = self.pass_into(
            cmd,
            Pass::FsrEasu,
            &input,
            Slot::Fsr,
            output,
            &params(uv_rect, size, output),
        )?;
        Ok((
            view,
            linear,
            full,
            [output.width as f32, output.height as f32],
        ))
    }

    /// Renders `pass` over the whole intermediate target in `slot`, resized to `extent`, and
    /// returns the target's view, ready to sample.
    #[allow(clippy::too_many_arguments)]
    fn pass_into(
        &mut self,
        cmd: vk::CommandBuffer,
        pass: Pass,
        inputs: &[(vk::ImageView, vk::Sampler)],
        slot: Slot,
        extent: vk::Extent2D,
        params: &PassParams,
    ) -> Result<vk::ImageView> {
        let set = self.input_set(inputs)?;
        let (target, name) = match slot {
            Slot::Aa => (&mut self.aa_target, "aa output"),
            Slot::SmaaInput => (&mut self.smaa_input, "smaa input"),
            Slot::SmaaEdges => (&mut self.smaa_edges, "smaa edges"),
            Slot::SmaaWeights => (&mut self.smaa_weights, "smaa weights"),
            Slot::Fsr => (&mut self.fsr_target, "fsr easu"),
        };
        let target = ensure_target(
            &self.device,
            &mut self.scheduler,
            target,
            extent,
            INTERMEDIATE,
            name,
        )?;
        let (image, view) = (target.image.raw(), target.view.raw());
        render_pass(
            &self.device,
            &mut self.passes,
            cmd,
            pass,
            set,
            image,
            view,
            extent,
            params,
        )?;
        Ok(view)
    }

    fn input_set(&mut self, inputs: &[(vk::ImageView, vk::Sampler)]) -> Result<vk::DescriptorSet> {
        let set = self
            .descriptors
            .allocate(&mut self.scheduler, self.passes.set_layout)?;
        let infos: [[vk::DescriptorImageInfo; 1]; MAX_INPUTS] = std::array::from_fn(|i| {
            let (view, sampler) = inputs.get(i).copied().unwrap_or(inputs[0]);
            [vk::DescriptorImageInfo {
                sampler,
                image_view: view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }]
        });
        let writes: [vk::WriteDescriptorSet; MAX_INPUTS] = std::array::from_fn(|binding| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&infos[binding])
        });
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
            self.device.raw.get_query_pool_results(
                pool,
                2 * slot as u32,
                &mut values,
                vk::QueryResultFlags::TYPE_64,
            )
        }
        .ok()?;
        let bits = self.device.properties.timestamp_valid_bits;
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let ticks = values[1].wrapping_sub(values[0]) & mask;
        Some(Duration::from_nanos(
            (ticks as f64 * f64::from(self.device.properties.timestamp_period)) as u64,
        ))
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
        extent: vk::Extent2D {
            width: (x1 - x0) as u32,
            height: (y1 - y0) as u32,
        },
    })
}

fn begin_rendering(
    device: &Device,
    cmd: vk::CommandBuffer,
    view: vk::ImageView,
    extent: vk::Extent2D,
    clear: Option<[f32; 4]>,
) {
    let attachment = [vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(if clear.is_some() {
            vk::AttachmentLoadOp::CLEAR
        } else {
            vk::AttachmentLoadOp::DONT_CARE
        })
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: clear.unwrap_or_default(),
            },
        })];
    let info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        })
        .layer_count(1)
        .color_attachments(&attachment);
    // SAFETY: `view` is in COLOR_ATTACHMENT_OPTIMAL layout.
    unsafe { device.raw.cmd_begin_rendering(cmd, &info) };
}

/// Renders one fullscreen pass into a whole intermediate target and leaves the target ready to
/// be sampled. SMAA's edge and weight passes discard empty pixels, so their targets start cleared.
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
    transition(
        device,
        cmd,
        image,
        COLOR_RANGE,
        SAMPLED_LAST_FRAME,
        ImageAccess::COLOR_ATTACHMENT,
    );
    let clear = matches!(pass, Pass::SmaaEdge | Pass::SmaaBlend).then_some([0.0; 4]);
    begin_rendering(device, cmd, view, extent, clear);
    let rect = vk::Rect2D {
        offset: vk::Offset2D::default(),
        extent,
    };
    let drawn = passes.draw(cmd, pass, INTERMEDIATE, set, rect, params);
    // SAFETY: matches the begin above.
    unsafe { device.raw.cmd_end_rendering(cmd) };
    transition(
        device,
        cmd,
        image,
        COLOR_RANGE,
        ImageAccess::COLOR_ATTACHMENT,
        ImageAccess::SHADER_READ,
    );
    drawn
}

/// Uploads a small sampled image (the SMAA area/search tables) once, viewed through `components`.
fn upload_image(
    device: &Arc<Device>,
    scheduler: &mut Scheduler,
    staging: &mut StagingRing,
    bytes: &[u8],
    (width, height): (u32, u32),
    format: vk::Format,
    components: vk::ComponentMapping,
) -> Result<FrameImage> {
    let image = Image::new(
        device,
        &ImageDesc::d2(
            format,
            width,
            height,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        ),
        "presenter image",
    )?;
    let view = image.view(vk::ImageViewType::TYPE_2D, components)?;
    let slice = staging.alloc(scheduler, bytes.len() as u64, 4)?;
    slice.data.copy_from_slice(bytes);
    let (buffer, offset) = (slice.buffer, slice.offset);
    let cmd = scheduler.record()?;
    transition(
        device,
        cmd,
        image.raw(),
        COLOR_RANGE,
        ImageAccess::UNDEFINED,
        ImageAccess::TRANSFER_DST,
    );
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
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    }];
    // SAFETY: the image is in TRANSFER_DST layout; the region covers it exactly.
    unsafe {
        device.raw.cmd_copy_buffer_to_image(
            cmd,
            buffer,
            image.raw(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &region,
        )
    };
    transition(
        device,
        cmd,
        image.raw(),
        COLOR_RANGE,
        ImageAccess::TRANSFER_DST,
        ImageAccess::SHADER_READ,
    );
    Ok(FrameImage { image, view })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Harness {
        presenter: Presenter,
    }

    impl Harness {
        fn new() -> Option<Self> {
            use crate::swapchain::Surface;
            use crate::{Instance, SurfaceSupport};
            let instance = match Instance::new(true, SurfaceSupport::Headless) {
                Ok(instance) => Arc::new(instance),
                Err(error) => {
                    eprintln!("skipping: {error}");
                    return None;
                }
            };
            if !instance.supports_headless_surface() {
                eprintln!("skipping: no VK_EXT_headless_surface");
                return None;
            }
            let surface = Surface::headless(&instance).ok()?;
            let device =
                match Device::new(instance, &GraphicsSettings::default(), Some(surface.raw())) {
                    Ok(device) => device,
                    Err(error) => {
                        eprintln!("skipping: {error}");
                        return None;
                    }
                };
            let swapchain = Swapchain::new(
                &device,
                surface,
                160,
                90,
                lichii_common::settings::PresentMode::Fifo,
            )
            .ok()?;
            let presenter = Presenter::new(
                &device,
                swapchain,
                &GraphicsSettings::default(),
                Arc::new(FrameMailbox::default()),
                Arc::new(PerfStats::default()),
            )
            .ok()?;
            Some(Self { presenter })
        }

        fn guest(
            &mut self,
            texels: &[u8],
            (width, height): (u32, u32),
            format: vk::Format,
            components: vk::ComponentMapping,
        ) -> GuestFrame {
            let p = &mut self.presenter;
            let image = upload_image(
                &p.device,
                &mut p.scheduler,
                &mut p.staging,
                texels,
                (width, height),
                format,
                components,
            )
            .unwrap();
            let tick = p.scheduler.flush().unwrap();
            GuestFrame {
                image: Arc::new(image),
                ready: (p.scheduler.shared_timeline(), tick),
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                width,
                height,
            }
        }

        /// Composes one frame into a `width` x `height` RGBA8 image and reads it back.
        fn compose(
            &mut self,
            guest: Option<&GuestFrame>,
            viewport: Viewport,
            mut ui: UiFrame,
            (width, height): (u32, u32),
        ) -> Vec<u8> {
            let p = &mut self.presenter;
            let format = vk::Format::R8G8B8A8_UNORM;
            let image = Image::new(
                &p.device,
                &ImageDesc::d2(
                    format,
                    width,
                    height,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                ),
                "test output",
            )
            .unwrap();
            let view = image
                .view(vk::ImageViewType::TYPE_2D, vk::ComponentMapping::default())
                .unwrap();
            let readback = crate::memory::Buffer::new(
                &p.device,
                u64::from(width * height * 4),
                vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
                "test readback",
            )
            .unwrap();
            p.ui.upload(&ui, &mut p.scheduler, &mut p.staging).unwrap();
            let cmd = p.scheduler.record().unwrap();
            let extent = vk::Extent2D { width, height };
            p.compose(
                cmd,
                guest,
                viewport,
                &ui,
                Output {
                    image: image.raw(),
                    view: view.raw(),
                    format,
                    extent,
                },
            )
            .unwrap();
            transition(
                &p.device,
                cmd,
                image.raw(),
                COLOR_RANGE,
                ImageAccess::COLOR_ATTACHMENT,
                ImageAccess::TRANSFER_SRC,
            );
            let region = [vk::BufferImageCopy {
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
            // SAFETY: the image is in TRANSFER_SRC layout and the buffer holds it exactly.
            unsafe {
                p.device.raw.cmd_copy_image_to_buffer(
                    cmd,
                    image.raw(),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.raw(),
                    &region,
                )
            };
            crate::commands::transfer_to_all_barrier(&p.device, cmd);
            p.scheduler.flush().unwrap();
            p.scheduler.finish().unwrap();
            p.ui.free(&mut ui, &mut p.scheduler);
            let pixels = readback.mapped().unwrap().to_vec();
            drop((view, image));
            pixels
        }

        fn assert_no_validation_errors(&self) {
            assert_eq!(
                self.presenter.device.instance().validation_errors(),
                0,
                "validation reported errors"
            );
        }
    }

    fn no_ui() -> UiFrame {
        UiFrame {
            primitives: Vec::new(),
            textures_delta: egui::TexturesDelta::default(),
            pixels_per_point: 1.0,
        }
    }

    fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * width + x) * 4) as usize;
        pixels[i..i + 4].try_into().unwrap()
    }

    fn assert_close(actual: [u8; 4], expected: [u8; 4], tolerance: u8, what: &str) {
        let close = actual
            .iter()
            .zip(&expected)
            .all(|(a, e)| a.abs_diff(*e) <= tolerance);
        assert!(close, "{what}: got {actual:?}, expected {expected:?}");
    }

    const QUADRANTS: [[u8; 4]; 4] = [
        [220, 30, 30, 255],
        [30, 220, 30, 255],
        [30, 30, 220, 255],
        [220, 220, 30, 255],
    ];

    fn quadrant_texels(width: u32, height: u32) -> Vec<u8> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    QUADRANTS[usize::from(y >= height / 2) * 2 + usize::from(x >= width / 2)]
                })
            })
            .flatten()
            .collect()
    }

    #[test]
    fn every_filter_and_aa_mode_scales_into_the_viewport() {
        let Some(mut h) = Harness::new() else { return };
        let source = (64, 36);
        let mut guest = h.guest(
            &quadrant_texels(source.0, source.1),
            source,
            vk::Format::R8G8B8A8_UNORM,
            vk::ComponentMapping::default(),
        );
        let (width, height) = (160, 90);
        let viewport = Viewport {
            x: 20,
            y: 10,
            width: 120,
            height: 68,
        };
        let at = |fx: f32, fy: f32| {
            (
                viewport.x as u32 + (fx * viewport.width as f32) as u32,
                viewport.y as u32 + (fy * viewport.height as f32) as u32,
            )
        };
        let filters = [
            ScalingFilter::NearestNeighbor,
            ScalingFilter::Bilinear,
            ScalingFilter::Bicubic,
            ScalingFilter::Gaussian,
            ScalingFilter::Lanczos,
            ScalingFilter::ScaleForce,
            ScalingFilter::Fsr,
            ScalingFilter::Area,
            ScalingFilter::Mmpx,
        ];
        for flip in [false, true] {
            guest.uv_rect = if flip {
                [0.0, 1.0, 1.0, -1.0]
            } else {
                [0.0, 0.0, 1.0, 1.0]
            };
            for filter in filters {
                for aa in [AntiAliasing::None, AntiAliasing::Fxaa, AntiAliasing::Smaa] {
                    let settings = GraphicsSettings {
                        scaling_filter: filter,
                        anti_aliasing: aa,
                        ..GraphicsSettings::default()
                    };
                    h.presenter.set_settings(&settings);
                    let pixels = h.compose(Some(&guest), viewport, no_ui(), (width, height));
                    let what = format!("{filter:?} {aa:?} flip={flip}");
                    for (i, &(fx, fy)) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
                        .iter()
                        .enumerate()
                    {
                        let (x, y) = at(fx, fy);
                        let expected = QUADRANTS[if flip { i ^ 2 } else { i }];
                        assert_close(
                            pixel(&pixels, width, x, y),
                            expected,
                            8,
                            &format!("{what} quadrant {i}"),
                        );
                    }
                    assert_eq!(
                        pixel(&pixels, width, 5, 5),
                        [0, 0, 0, 255],
                        "{what}: letterbox"
                    );
                    assert_eq!(
                        pixel(&pixels, width, 150, 85),
                        [0, 0, 0, 255],
                        "{what}: letterbox"
                    );
                }
            }
        }
        h.assert_no_validation_errors();
    }

    #[test]
    fn draws_published_frames_to_the_swapchain() {
        let Some(mut h) = Harness::new() else { return };
        let guest = h.guest(
            &quadrant_texels(64, 36),
            (64, 36),
            vk::Format::R8G8B8A8_UNORM,
            vk::ComponentMapping::default(),
        );
        h.presenter.mailbox.publish(guest);
        for (i, (w, hgt)) in [
            (160, 90),
            (160, 90),
            (320, 200),
            (320, 200),
            (100, 100),
            (0, 0),
        ]
        .into_iter()
        .enumerate()
        {
            let settings = GraphicsSettings {
                scaling_filter: if i % 2 == 0 {
                    ScalingFilter::Fsr
                } else {
                    ScalingFilter::Bicubic
                },
                anti_aliasing: if i % 2 == 0 {
                    AntiAliasing::Smaa
                } else {
                    AntiAliasing::Fxaa
                },
                ..GraphicsSettings::default()
            };
            h.presenter.set_settings(&settings);
            let viewport = Viewport::fit(AspectRatio::R16x9, 0, 0, w, hgt);
            h.presenter.draw(w, hgt, viewport, no_ui()).unwrap();
        }
        h.presenter.scheduler.finish().unwrap();
        let snapshot = h
            .presenter
            .perf
            .snapshot(Instant::now(), Duration::from_secs(60));
        assert!(snapshot.vram_budget > 0, "{snapshot:?}");
        h.assert_no_validation_errors();
    }

    fn random_texels(width: u32, height: u32) -> Vec<u8> {
        (0..width * height)
            .flat_map(|i| {
                let v = i.wrapping_mul(2_654_435_761);
                [(v >> 24) as u8, (v >> 16) as u8, (v >> 8) as u8, 255]
            })
            .collect()
    }

    #[test]
    fn interpolating_filters_are_exact_at_native_size() {
        let Some(mut h) = Harness::new() else { return };
        let (width, height) = (40, 24);
        let texels = random_texels(width, height);
        let guest = h.guest(
            &texels,
            (width, height),
            vk::Format::R8G8B8A8_UNORM,
            vk::ComponentMapping::default(),
        );
        let viewport = Viewport {
            x: 0,
            y: 0,
            width,
            height,
        };
        for filter in [
            ScalingFilter::NearestNeighbor,
            ScalingFilter::Bilinear,
            ScalingFilter::Bicubic,
            ScalingFilter::Lanczos,
            ScalingFilter::Area,
        ] {
            h.presenter.set_settings(&GraphicsSettings {
                scaling_filter: filter,
                anti_aliasing: AntiAliasing::None,
                ..GraphicsSettings::default()
            });
            let pixels = h.compose(Some(&guest), viewport, no_ui(), (width, height));
            for (i, (got, want)) in pixels
                .as_chunks::<4>()
                .0
                .iter()
                .zip(texels.as_chunks::<4>().0)
                .enumerate()
            {
                let close = got.iter().zip(want).all(|(g, w)| g.abs_diff(*w) <= 1);
                assert!(
                    close,
                    "{filter:?} texel {i}: got {got:?}, expected {want:?}"
                );
            }
        }
        h.assert_no_validation_errors();
    }

    #[test]
    fn cropped_frames_sample_the_bound_image() {
        let Some(mut h) = Harness::new() else { return };
        let (width, height) = (48, 32);
        let texels = random_texels(width, height);
        let mut guest = h.guest(
            &texels,
            (width, height),
            vk::Format::R8G8B8A8_UNORM,
            vk::ComponentMapping::default(),
        );
        let (left, top, cw, ch) = (12, 8, 24, 16);
        guest.uv_rect = [
            left as f32 / width as f32,
            top as f32 / height as f32,
            cw as f32 / width as f32,
            ch as f32 / height as f32,
        ];
        (guest.width, guest.height) = (cw, ch);
        let viewport = Viewport {
            x: 0,
            y: 0,
            width: cw,
            height: ch,
        };
        for (filter, aa) in [
            (ScalingFilter::Area, AntiAliasing::None),
            (ScalingFilter::Lanczos, AntiAliasing::None),
        ] {
            h.presenter.set_settings(&GraphicsSettings {
                scaling_filter: filter,
                anti_aliasing: aa,
                ..GraphicsSettings::default()
            });
            let pixels = h.compose(Some(&guest), viewport, no_ui(), (cw, ch));
            let mismatches = (0..ch)
                .flat_map(|y| (0..cw).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    let i = (((y + top) * width + x + left) * 4) as usize;
                    let want: [u8; 4] = texels[i..i + 4].try_into().unwrap();
                    let got = pixel(&pixels, cw, x, y);
                    got.iter().zip(&want).any(|(g, w)| g.abs_diff(*w) > 1)
                })
                .count();
            assert!(
                mismatches == 0,
                "{filter:?} {aa:?}: {mismatches} texels outside the crop window"
            );
        }
        h.assert_no_validation_errors();
    }

    #[test]
    fn smaa_resolves_cropped_and_flipped_frames_in_place() {
        let Some(mut h) = Harness::new() else { return };
        let (width, height) = (48, 32);
        let (left, top, cw, ch) = (12u32, 8u32, 24u32, 16u32);
        let inside = [200, 60, 20, 255];
        let texels: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let in_crop =
                        (left..left + cw).contains(&x) && (top..top + ch / 2).contains(&y);
                    if in_crop {
                        inside
                    } else if (left..left + cw).contains(&x) && (top..top + ch).contains(&y) {
                        [20, 60, 200, 255]
                    } else {
                        [0, 255, 0, 255]
                    }
                })
            })
            .flatten()
            .collect();
        let mut guest = h.guest(
            &texels,
            (width, height),
            vk::Format::R8G8B8A8_UNORM,
            vk::ComponentMapping::default(),
        );
        let (w, hgt) = (width as f32, height as f32);
        guest.uv_rect = [
            left as f32 / w,
            (top + ch) as f32 / hgt,
            cw as f32 / w,
            -(ch as f32) / hgt,
        ];
        (guest.width, guest.height) = (cw, ch);
        h.presenter.set_settings(&GraphicsSettings {
            scaling_filter: ScalingFilter::NearestNeighbor,
            anti_aliasing: AntiAliasing::Smaa,
            ..GraphicsSettings::default()
        });
        let pixels = h.compose(
            Some(&guest),
            Viewport {
                x: 0,
                y: 0,
                width: cw,
                height: ch,
            },
            no_ui(),
            (cw, ch),
        );
        for x in 1..cw - 1 {
            assert_close(
                pixel(&pixels, cw, x, 2),
                [20, 60, 200, 255],
                4,
                &format!("flipped bottom half at x={x}"),
            );
            assert_close(
                pixel(&pixels, cw, x, ch - 3),
                inside,
                4,
                &format!("flipped top half at x={x}"),
            );
        }
        h.assert_no_validation_errors();
    }

    #[test]
    fn frames_outlive_the_scheduler_that_rendered_them() {
        let Some(mut h) = Harness::new() else { return };
        let device = h.presenter.device.clone();
        let mut producer = Scheduler::new(&device).unwrap();
        let mut staging = StagingRing::new(&device, 1 << 20).unwrap();
        let texels = quadrant_texels(16, 16);
        let image = upload_image(
            &device,
            &mut producer,
            &mut staging,
            &texels,
            (16, 16),
            vk::Format::R8G8B8A8_UNORM,
            vk::ComponentMapping::default(),
        )
        .unwrap();
        let tick = producer.flush().unwrap();
        h.presenter.mailbox.publish(GuestFrame {
            image: Arc::new(image),
            ready: (producer.shared_timeline(), tick),
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            width: 16,
            height: 16,
        });
        drop(producer);
        drop(staging);
        for _ in 0..3 {
            h.presenter
                .draw(
                    160,
                    90,
                    Viewport::fit(AspectRatio::R16x9, 0, 0, 160, 90),
                    no_ui(),
                )
                .unwrap();
        }
        h.presenter.scheduler.finish().unwrap();
        h.assert_no_validation_errors();
    }

    #[test]
    fn guest_textures_sample_through_the_host_swizzle() {
        use lichii_texture::PixelFormat as F;
        let Some(mut h) = Harness::new() else { return };
        h.presenter.set_settings(&GraphicsSettings {
            scaling_filter: ScalingFilter::NearestNeighbor,
            anti_aliasing: AntiAliasing::None,
            ..GraphicsSettings::default()
        });
        let red_texels: [(F, u16); 5] = [
            (F::A1B5G5R5Unorm, 0x801f),
            (F::B5G6R5Unorm, 0x001f),
            (F::A4B4G4R4Unorm, 0xf00f),
            (F::A1R5G5B5Unorm, 0xfc00),
            (F::R5G6B5Unorm, 0xf800),
        ];
        let mut swizzled = 0;
        for (format, red) in red_texels {
            let host = crate::format::host_format(
                &h.presenter.device,
                format,
                lichii_common::settings::AstcRecompression::Uncompressed,
            );
            if !matches!(host.conversion, crate::format::Conversion::None) {
                continue;
            }
            let s = host.swizzle;
            swizzled += usize::from(
                [s.r, s.g, s.b, s.a]
                    .iter()
                    .any(|&c| c != vk::ComponentSwizzle::IDENTITY),
            );
            let texels: Vec<u8> = std::iter::repeat_n(red.to_le_bytes(), 16)
                .flatten()
                .collect();
            let guest = h.guest(&texels, (4, 4), host.format, host.swizzle);
            let pixels = h.compose(
                Some(&guest),
                Viewport {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 8,
                },
                no_ui(),
                (8, 8),
            );
            assert_eq!(
                pixel(&pixels, 8, 4, 4),
                [255, 0, 0, 255],
                "{format:?} as {:?} via {s:?}",
                host.format
            );
        }
        if swizzled == 0 {
            eprintln!("no swizzled host format on this device");
        }
        h.assert_no_validation_errors();
    }

    #[test]
    fn egui_draws_over_the_frame() {
        let Some(mut h) = Harness::new() else { return };
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(64.0, 48.0),
            )),
            ..Default::default()
        };
        let output = ctx.run_ui(input, |ui| {
            ui.label("Lichii");
            ui.painter().rect_filled(
                egui::Rect::from_min_size(egui::pos2(40.0, 30.0), egui::vec2(10.0, 10.0)),
                0.0,
                egui::Color32::from_rgb(0, 0, 255),
            );
        });
        let ui = UiFrame {
            primitives: ctx.tessellate(output.shapes, output.pixels_per_point),
            textures_delta: output.textures_delta,
            pixels_per_point: output.pixels_per_point,
        };
        assert!(
            !ui.textures_delta.set.is_empty(),
            "the font atlas uploads on the first frame"
        );
        let pixels = h.compose(
            None,
            Viewport {
                x: 0,
                y: 0,
                width: 64,
                height: 48,
            },
            ui,
            (64, 48),
        );
        assert_eq!(pixel(&pixels, 64, 45, 35), [0, 0, 255, 255]);
        assert_eq!(pixel(&pixels, 64, 60, 2), [0, 0, 0, 255]);
        let text = (0..20)
            .flat_map(|y| (0..40).map(move |x| (x, y)))
            .any(|(x, y)| pixel(&pixels, 64, x, y)[0] > 64);
        assert!(text, "the label rendered");
        h.assert_no_validation_errors();
    }

    #[test]
    fn viewport_letterboxes_and_pillarboxes() {
        assert_eq!(
            Viewport::fit(AspectRatio::R16x9, 0, 0, 1920, 1200),
            Viewport {
                x: 0,
                y: 60,
                width: 1920,
                height: 1080
            }
        );
        assert_eq!(
            Viewport::fit(AspectRatio::R4x3, 10, 0, 1920, 1080),
            Viewport {
                x: 250,
                y: 0,
                width: 1440,
                height: 1080
            }
        );
        assert_eq!(
            Viewport::fit(AspectRatio::Stretch, 0, 20, 800, 600),
            Viewport {
                x: 0,
                y: 20,
                width: 800,
                height: 600
            }
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
        let extent = vk::Extent2D {
            width: 100,
            height: 50,
        };
        let inside = clip_viewport(
            Viewport {
                x: 10,
                y: 5,
                width: 200,
                height: 20,
            },
            extent,
        )
        .unwrap();
        assert_eq!((inside.extent.width, inside.extent.height), (90, 20));
        assert!(
            clip_viewport(
                Viewport {
                    x: 0,
                    y: 0,
                    width: 0,
                    height: 10
                },
                extent
            )
            .is_none()
        );
    }
}
