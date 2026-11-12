use std::sync::Arc;

use ash::{khr, vk};
use lichii_common::settings::PresentMode;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::device::Device;
use crate::error::{Error, Result, VkResultExt};
use crate::instance::Instance;

/// A presentation surface: a window, or a headless surface for tests.
pub struct Surface {
    instance: Arc<Instance>,
    raw: vk::SurfaceKHR,
}

impl Surface {
    /// # Safety
    /// The window and display handles must stay valid until the surface is dropped.
    pub unsafe fn new(
        instance: &Arc<Instance>,
        display: RawDisplayHandle,
        window: RawWindowHandle,
    ) -> Result<Self> {
        // SAFETY: the caller guarantees the handles outlive the surface.
        let raw = unsafe {
            ash_window::create_surface(&instance.entry, &instance.raw, display, window, None)
                .call("vkCreateSurfaceKHR")?
        };
        Ok(Self {
            instance: instance.clone(),
            raw,
        })
    }

    /// A surface with no window (`VK_EXT_headless_surface`).
    pub fn headless(instance: &Arc<Instance>) -> Result<Self> {
        let loader = instance.headless.as_ref().ok_or(Error::NoSurfaceFormat)?;
        // SAFETY: valid create info on an instance with the extension enabled.
        let raw = unsafe {
            loader
                .create_headless_surface(&vk::HeadlessSurfaceCreateInfoEXT::default(), None)
                .call("vkCreateHeadlessSurfaceEXT")?
        };
        Ok(Self {
            instance: instance.clone(),
            raw,
        })
    }

    pub fn raw(&self) -> vk::SurfaceKHR {
        self.raw
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        if let Some(loader) = &self.instance.surface {
            // SAFETY: the swapchain using this surface was destroyed first.
            unsafe { loader.destroy_surface(self.raw, None) };
        }
    }
}

/// An image acquired for drawing one host frame.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub index: u32,
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    /// Signalled when the image is ready to be drawn; wait on it before writing the image.
    pub acquired: vk::Semaphore,
    /// Signal it when drawing is done; presentation waits on it.
    pub rendered: vk::Semaphore,
}

/// The window's swapchain. It is recreated on resize, on `VK_ERROR_OUT_OF_DATE_KHR` or
/// suboptimal presents, and when the present mode setting changes.
pub struct Swapchain {
    device: Arc<Device>,
    surface: Surface,
    loader: khr::swapchain::Device,
    raw: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    mode: PresentMode,
    acquire_semaphores: Vec<vk::Semaphore>,
    next_acquire: usize,
    /// One per image: a present may still wait on it until that image is acquired again.
    rendered_semaphores: Vec<vk::Semaphore>,
    stale: bool,
}

fn vk_present_mode(mode: PresentMode) -> vk::PresentModeKHR {
    match mode {
        PresentMode::Fifo => vk::PresentModeKHR::FIFO,
        PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
        PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
    }
}

/// The present mode the surface supports closest to the setting; FIFO always exists.
pub fn choose_present_mode(
    wanted: PresentMode,
    supported: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    let preference: &[vk::PresentModeKHR] = match wanted {
        PresentMode::Mailbox => &[vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE],
        PresentMode::Immediate => &[vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::MAILBOX],
        PresentMode::FifoRelaxed => &[vk::PresentModeKHR::FIFO_RELAXED],
        PresentMode::Fifo => &[],
    };
    debug_assert!(preference.is_empty() || preference[0] == vk_present_mode(wanted));
    preference
        .iter()
        .copied()
        .find(|m| supported.contains(m))
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn choose_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
    const PREFERRED: [vk::Format; 3] = [
        vk::Format::B8G8R8A8_UNORM,
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::A2B10G10R10_UNORM_PACK32,
    ];
    PREFERRED
        .iter()
        .find_map(|&format| {
            formats
                .iter()
                .find(|f| f.format == format && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        })
        .or_else(|| formats.iter().find(|f| f.format != vk::Format::UNDEFINED))
        .copied()
        .or_else(|| {
            formats.first().map(|_| vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            })
        })
}

impl Swapchain {
    pub fn new(
        device: &Arc<Device>,
        surface: Surface,
        width: u32,
        height: u32,
        mode: PresentMode,
    ) -> Result<Self> {
        let loader = khr::swapchain::Device::new(&device.instance.raw, &device.raw);
        let mut swapchain = Self {
            device: device.clone(),
            surface,
            loader,
            raw: vk::SwapchainKHR::null(),
            images: Vec::new(),
            views: Vec::new(),
            format: vk::SurfaceFormatKHR::default(),
            extent: vk::Extent2D { width, height },
            mode,
            acquire_semaphores: Vec::new(),
            next_acquire: 0,
            rendered_semaphores: Vec::new(),
            stale: true,
        };
        swapchain.recreate(width, height)?;
        Ok(swapchain)
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn format(&self) -> vk::Format {
        self.format.format
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Changes the present mode; takes effect on the next acquire.
    pub fn set_present_mode(&mut self, mode: PresentMode) {
        if mode != self.mode {
            self.mode = mode;
            self.stale = true;
        }
    }

    fn surface_loader(&self) -> &khr::surface::Instance {
        self.device
            .instance
            .surface
            .as_ref()
            .expect("a swapchain needs an instance with surface support")
    }

    fn recreate(&mut self, width: u32, height: u32) -> Result<()> {
        let physical = self.device.physical;
        let surface = self.surface.raw;
        let loader = self.surface_loader();
        // SAFETY: valid physical device and surface.
        let (caps, formats, modes) = unsafe {
            (
                loader
                    .get_physical_device_surface_capabilities(physical, surface)
                    .call("vkGetPhysicalDeviceSurfaceCapabilitiesKHR")?,
                loader
                    .get_physical_device_surface_formats(physical, surface)
                    .call("vkGetPhysicalDeviceSurfaceFormatsKHR")?,
                loader
                    .get_physical_device_surface_present_modes(physical, surface)
                    .call("vkGetPhysicalDeviceSurfacePresentModesKHR")?,
            )
        };
        let format = choose_format(&formats).ok_or(Error::NoSurfaceFormat)?;
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        let present_mode = choose_present_mode(self.mode, &modes);
        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count != 0 {
            image_count = image_count.min(caps.max_image_count);
        }
        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::INHERIT,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        ]
        .into_iter()
        .find(|&mode| caps.supported_composite_alpha.contains(mode))
        .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST;
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage & caps.supported_usage_flags)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(self.raw);
        self.device.wait_idle();
        // SAFETY: the device is idle, so the old swapchain's images and semaphores are unused.
        let raw = unsafe {
            self.loader
                .create_swapchain(&info, None)
                .call("vkCreateSwapchainKHR")?
        };
        self.destroy_resources();
        self.raw = raw;
        self.format = format;
        self.extent = extent;
        // SAFETY: `raw` is a live swapchain.
        self.images = unsafe {
            self.loader
                .get_swapchain_images(raw)
                .call("vkGetSwapchainImagesKHR")?
        };
        let device = &self.device.raw;
        for &image in &self.images {
            let view = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            // SAFETY: valid image of this swapchain and create infos.
            unsafe {
                self.views.push(
                    device
                        .create_image_view(&view, None)
                        .call("vkCreateImageView")?,
                );
                self.rendered_semaphores.push(
                    device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .call("vkCreateSemaphore")?,
                );
                self.acquire_semaphores.push(
                    device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .call("vkCreateSemaphore")?,
                );
            }
        }
        // SAFETY: as above.
        self.acquire_semaphores.push(unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .call("vkCreateSemaphore")?
        });
        self.stale = false;
        tracing::debug!(
            "swapchain {}x{} {:?} {:?}, {} images",
            extent.width,
            extent.height,
            format.format,
            present_mode,
            self.images.len()
        );
        Ok(())
    }

    fn destroy_resources(&mut self) {
        let device = &self.device.raw;
        // SAFETY: callers idle the device first.
        unsafe {
            for view in self.views.drain(..) {
                device.destroy_image_view(view, None);
            }
            for semaphore in self
                .rendered_semaphores
                .drain(..)
                .chain(self.acquire_semaphores.drain(..))
            {
                device.destroy_semaphore(semaphore, None);
            }
            if self.raw != vk::SwapchainKHR::null() {
                self.loader.destroy_swapchain(self.raw, None);
            }
        }
        self.raw = vk::SwapchainKHR::null();
        self.images.clear();
        self.next_acquire = 0;
    }

    /// Acquires the next image for a window of `width` x `height`, recreating the swapchain
    /// when needed. `None` while the window has no area (minimised).
    ///
    /// The acquire semaphores rotate through one more than the image count, so a semaphore is
    /// reused only after the frame that waited on it was itself followed by a full swapchain
    /// cycle; the caller must not keep more frames in flight than there are images.
    pub fn acquire(&mut self, width: u32, height: u32) -> Result<Option<Frame>> {
        if width == 0 || height == 0 {
            return Ok(None);
        }
        if self.stale || width != self.extent.width || height != self.extent.height {
            self.recreate(width, height)?;
        }
        for _ in 0..2 {
            let semaphore = self.acquire_semaphores[self.next_acquire];
            // SAFETY: live swapchain; the semaphore is unsignalled (see the rotation above).
            let result = unsafe {
                self.loader
                    .acquire_next_image(self.raw, u64::MAX, semaphore, vk::Fence::null())
            };
            match result {
                Ok((index, suboptimal)) => {
                    self.next_acquire = (self.next_acquire + 1) % self.acquire_semaphores.len();
                    self.stale |= suboptimal;
                    return Ok(Some(Frame {
                        index,
                        image: self.images[index as usize],
                        view: self.views[index as usize],
                        extent: self.extent,
                        format: self.format.format,
                        acquired: semaphore,
                        rendered: self.rendered_semaphores[index as usize],
                    }));
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate(width, height)?,
                Err(result) => {
                    return Err(Error::Vulkan {
                        call: "vkAcquireNextImageKHR",
                        result,
                    });
                }
            }
        }
        Ok(None)
    }

    /// Queues `frame` for presentation once its `rendered` semaphore is signalled.
    pub fn present(&mut self, frame: &Frame) -> Result<()> {
        let swapchains = [self.raw];
        let indices = [frame.index];
        let waits = [frame.rendered];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: the queue lock serialises presentation with submissions from other threads.
        let result = self
            .device
            .with_queue(|queue| unsafe { self.loader.queue_present(queue, &info) });
        match result {
            Ok(suboptimal) => {
                self.stale |= suboptimal;
                Ok(())
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.stale = true;
                Ok(())
            }
            Err(result) => Err(Error::Vulkan {
                call: "vkQueuePresentKHR",
                result,
            }),
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.device.wait_idle();
        self.destroy_resources();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn present_mode_falls_back_to_fifo() {
        let fifo_only = [vk::PresentModeKHR::FIFO];
        assert_eq!(
            choose_present_mode(PresentMode::Mailbox, &fifo_only),
            vk::PresentModeKHR::FIFO
        );
        let all = [
            vk::PresentModeKHR::FIFO,
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::MAILBOX,
        ];
        assert_eq!(
            choose_present_mode(PresentMode::Mailbox, &all),
            vk::PresentModeKHR::MAILBOX
        );
        assert_eq!(
            choose_present_mode(PresentMode::Mailbox, &all[..2]),
            vk::PresentModeKHR::IMMEDIATE
        );
        assert_eq!(
            choose_present_mode(PresentMode::Fifo, &all),
            vk::PresentModeKHR::FIFO
        );
    }

    #[test]
    fn surface_format_prefers_unorm_bgra() {
        let formats = [
            vk::SurfaceFormatKHR {
                format: vk::Format::R16G16B16A16_SFLOAT,
                color_space: vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ];
        assert_eq!(
            choose_format(&formats).unwrap().format,
            vk::Format::B8G8R8A8_UNORM
        );
    }
}
