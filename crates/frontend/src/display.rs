//! The window's Vulkan presenter with the instance, device and swapchain behind it, kept alive
//! together with the window they present to.

use std::sync::Arc;

use lichii_common::ConsoleGeneration;
use lichii_common::perf::PerfStats;
use lichii_common::settings::GraphicsSettings;
use lichii_core::RendererFactory;
use lichii_texture::block_linear::GobLayout;
use lichii_vulkan::present::{FrameMailbox, Presenter, UiFrame, Viewport};
use lichii_vulkan::rasterizer::VulkanRasterizer;
use lichii_vulkan::swapchain::{Surface, Swapchain};
use lichii_vulkan::{Device, Instance, SurfaceSupport};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

/// Field order is drop order: the presenter (and the surface it owns) goes before the window.
pub struct Display {
    presenter: Presenter,
    device: Arc<Device>,
    mailbox: Arc<FrameMailbox>,
    window: Arc<Window>,
}

impl Display {
    pub fn new(
        window: Arc<Window>,
        settings: &GraphicsSettings,
        perf: Arc<PerfStats>,
    ) -> Result<Self, String> {
        let display = window
            .display_handle()
            .map_err(|e| format!("reading the display handle: {e}"))?
            .as_raw();
        let handle = window
            .window_handle()
            .map_err(|e| format!("reading the window handle: {e}"))?
            .as_raw();
        let instance = Instance::new(settings.validation_layers, SurfaceSupport::Display(display))
            .map_err(|e| format!("loading Vulkan: {e}"))?;
        let instance = Arc::new(instance);
        // SAFETY: the surface lives in the presenter, which this struct drops before the window.
        let surface = unsafe { Surface::new(&instance, display, handle) }
            .map_err(|e| format!("creating the window surface: {e}"))?;
        let device = Device::new(instance, settings, Some(surface.raw()))
            .map_err(|e| format!("no Vulkan 1.3 device can present to this window: {e}"))?;
        let size = window.inner_size();
        let swapchain = Swapchain::new(
            &device,
            surface,
            size.width.max(1),
            size.height.max(1),
            settings.present_mode,
        )
        .map_err(|e| format!("creating the swapchain: {e}"))?;
        tracing::info!(
            "presenting with {} ({:?}, {:?})",
            device.properties.name,
            swapchain.format(),
            settings.present_mode
        );
        let mailbox = Arc::new(FrameMailbox::default());
        let presenter = Presenter::new(&device, swapchain, settings, mailbox.clone(), perf)
            .map_err(|e| format!("creating the presenter: {e}"))?;
        Ok(Self {
            presenter,
            device,
            mailbox,
            window,
        })
    }

    /// A Vulkan renderer for a title of generation `title` on this display's device, drawing
    /// guest frames into the presenter's mailbox. If the renderer cannot be created the title
    /// runs on the headless renderer and shows nothing.
    pub fn renderer(
        &self,
        title: ConsoleGeneration,
        settings: &GraphicsSettings,
    ) -> RendererFactory {
        let device = self.device.clone();
        let mailbox = self.mailbox.clone();
        let settings = settings.clone();
        let gob_layout = match title {
            ConsoleGeneration::Switch1 => GobLayout::Tegra,
            ConsoleGeneration::Switch2 => GobLayout::Switch2,
        };
        Box::new(move |handles| {
            match VulkanRasterizer::new(&device, handles.clone(), mailbox, gob_layout) {
                Ok(mut rasterizer) => {
                    rasterizer.apply_settings(&settings);
                    Box::new(rasterizer)
                }
                Err(e) => {
                    tracing::error!("creating the Vulkan renderer, drawing nothing: {e}");
                    lichii_core::headless_renderer()(handles)
                }
            }
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.device.properties.name
    }

    /// Applies filter, anti-aliasing, sharpness and present mode from the next frame.
    pub fn set_settings(&mut self, settings: &GraphicsSettings) {
        self.presenter.set_settings(settings);
    }

    /// Draws the newest guest frame into `viewport`, the UI over the whole window, and presents.
    pub fn draw(&mut self, viewport: Viewport, ui: UiFrame) {
        let size = self.window.inner_size();
        if let Err(e) = self.presenter.draw(size.width, size.height, viewport, ui) {
            tracing::error!("presenting: {e}");
        }
    }
}
