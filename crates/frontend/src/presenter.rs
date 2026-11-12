//! A stand-in presenter until `lichii-vulkan` can present: it draws the frontend's [`UiFrame`]
//! with egui-wgpu on wgpu's Vulkan backend, over a black guest area. It consumes exactly the
//! renderer contract, so swapping in the real presenter only changes this module.

use std::sync::Arc;
use std::time::Instant;

use lichii_common::perf::{FrameSample, PerfStats};
use lichii_common::settings::PresentMode;
use lichii_vulkan::present::UiFrame;
use winit::window::Window;

pub struct Presenter {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    renderer: egui_wgpu::Renderer,
    adapter_name: String,
    perf: Arc<PerfStats>,
    last_present: Option<Instant>,
}

impl Presenter {
    pub fn new(
        window: Arc<Window>,
        mode: PresentMode,
        perf: Arc<PerfStats>,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let size = window.inner_size();
        let surface = instance
            .create_surface(window)
            .map_err(|e| format!("creating the window surface: {e}"))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .map_err(|e| format!("no Vulkan device can present to this window: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))
            .map_err(|e| format!("opening the Vulkan device: {e}"))?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or("the window surface supports no formats")?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: pick_present_mode(mode, &caps.present_modes),
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes.first().copied().unwrap_or_default(),
            view_formats: Vec::new(),
            color_space: Default::default(),
        };
        surface.configure(&device, &config);
        let renderer = egui_wgpu::Renderer::new(&device, format, Default::default());
        let adapter_name = adapter.get_info().name;
        tracing::info!(
            "presenting with {adapter_name} ({format:?}, {:?})",
            config.present_mode
        );
        Ok(Self {
            surface,
            device,
            queue,
            config,
            renderer,
            adapter_name,
            perf,
            last_present: None,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Draws `frame` over a black background and presents it.
    pub fn present(&mut self, mut frame: UiFrame) {
        for (id, deltas) in &frame.textures_delta.set {
            for delta in deltas {
                self.renderer
                    .update_texture(&self.device, &self.queue, *id, delta);
            }
        }
        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                self.surface.configure(&self.device, &self.config);
                t
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                self.free(&mut frame);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                self.free(&mut frame);
                return;
            }
        };
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: frame.pixels_per_point,
        };
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let callbacks = self.renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &frame.primitives,
            &screen,
        );
        let view = texture.texture.create_view(&Default::default());
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.renderer.render(&mut pass, &frame.primitives, &screen);
        }
        self.queue
            .submit(callbacks.into_iter().chain([encoder.finish()]));
        self.queue.present(texture);
        self.free(&mut frame);

        let now = Instant::now();
        if let Some(last) = self.last_present.replace(now) {
            self.perf.record_present(FrameSample {
                presented_at: now,
                frame_time: now - last,
                gpu_time: None,
            });
        }
    }

    fn free(&mut self, frame: &mut UiFrame) {
        for id in &frame.textures_delta.free {
            self.renderer.free_texture(id);
        }
        frame.textures_delta.clear();
    }
}

fn pick_present_mode(mode: PresentMode, supported: &[wgpu::PresentMode]) -> wgpu::PresentMode {
    let wanted = match mode {
        PresentMode::Fifo => wgpu::PresentMode::Fifo,
        PresentMode::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
        PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
        PresentMode::Immediate => wgpu::PresentMode::Immediate,
    };
    if supported.contains(&wanted) {
        wanted
    } else {
        wgpu::PresentMode::Fifo
    }
}
