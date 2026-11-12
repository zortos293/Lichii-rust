//! The renderer behind `lichii_gpu::backend::Rasterizer`, driven by the GPU's back thread.
//!
//! This milestone implements presentation (guest framebuffers read from device memory, deswizzled
//! and handed to the [`Presenter`](crate::present::Presenter)) and the fence and query protocol.
//! Draws, clears and compute dispatches are accepted and dropped until the buffer, texture and
//! pipeline caches land.

use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;
use lichii_common::DAddr;
use lichii_common::settings::AstcRecompression;
use lichii_gpu::backend::ChannelId;
use lichii_gpu::backend::{
    ComputeContext, DrawParams, FenceAction, FramebufferConfig, GpuHandles, GraphicsContext,
    IndirectDraw, Query, QueryKind, Rasterizer,
};
use lichii_gpu::gmmu::Gmmu;
use lichii_texture::block_linear::{Tiling, unswizzle};

use crate::commands::{COLOR_RANGE, ImageAccess, transition};
use crate::device::Device;
use crate::error::Result;
use crate::format::{Conversion, host_format};
use crate::memory::{Image, ImageDesc};
use crate::present::{FrameImage, FrameMailbox, GuestFrame};
use crate::scheduler::Scheduler;
use crate::staging::StagingRing;

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
    frames: Vec<Arc<FrameImage>>,
    tiled: Vec<u8>,
    fences: VecDeque<(u64, FenceAction)>,
    unflushed: bool,
    warned_unimplemented: bool,
}

impl VulkanRasterizer {
    pub fn new(
        device: &Arc<Device>,
        handles: GpuHandles,
        mailbox: Arc<FrameMailbox>,
    ) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            handles,
            scheduler: Scheduler::new(device)?,
            staging: StagingRing::new(device, 32 << 20)?,
            mailbox,
            frames: Vec::new(),
            tiled: Vec::new(),
            fences: VecDeque::new(),
            unflushed: false,
            warned_unimplemented: false,
        })
    }

    fn unimplemented(&mut self, what: &str) {
        if !self.warned_unimplemented {
            self.warned_unimplemented = true;
            tracing::warn!("{what} and other rendering are not implemented yet; dropping them");
        }
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
            width: right - left,
            height: bottom - top,
        });
        Ok(())
    }
}

impl Rasterizer for VulkanRasterizer {
    fn draw(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &DrawParams<'_>) {
        self.unimplemented("draws");
    }

    fn draw_indirect(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &IndirectDraw) {
        self.unimplemented("indirect draws");
    }

    fn clear(&mut self, _ctx: &mut GraphicsContext<'_>, _layer_count: u32) {
        self.unimplemented("clears");
    }

    fn dispatch_compute(&mut self, _ctx: &ComputeContext<'_>) {
        self.unimplemented("compute dispatches");
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

    fn invalidate_region(&mut self, _addr: DAddr, _size: u64) {}

    fn flush_region(&mut self, _addr: DAddr, _size: u64) {}

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
        if let Err(error) = self.upload_frame(framebuffer) {
            tracing::error!(%error, "presenting a guest frame failed");
        }
        self.release_fences(false);
    }
}

impl Drop for VulkanRasterizer {
    fn drop(&mut self) {
        let _ = self.scheduler.finish();
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::GraphicsSettings;
    use lichii_gpu::backend::{CropRect, ReleaseSize};
    use std::time::Instant;

    use lichii_gpu::clock::{GpuTimestamp, GuestTime};
    use lichii_gpu::gmmu::BigPageSize;
    use lichii_gpu::host1x::Syncpoints;
    use lichii_memory::{DeviceAddressSpace, PhysicalMemory};
    use lichii_texture::PixelFormat;
    use lichii_texture::block_linear::{GobLayout, swizzle};

    use super::*;
    use crate::memory::Buffer;
    use crate::{Instance, SurfaceSupport};

    const DRAM: u64 = 16 << 20;
    const DADDR: DAddr = 0x1_0000_0000;

    struct Fixture {
        rasterizer: VulkanRasterizer,
        mailbox: Arc<FrameMailbox>,
        memory: Arc<DeviceAddressSpace>,
        syncpoints: Arc<Syncpoints>,
    }

    fn fixture() -> Option<Fixture> {
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
        let rasterizer = VulkanRasterizer::new(&device, handles, mailbox.clone()).unwrap();
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

        fn assert_no_validation_errors(&self) {
            assert_eq!(
                self.rasterizer.device.instance().validation_errors(),
                0,
                "validation reported errors"
            );
        }
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
            let Some(mut f) = fixture() else { return };
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
        let Some(mut f) = fixture() else {
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
        let Some(mut f) = fixture() else {
            return;
        };
        f.rasterizer.present(&FramebufferConfig {
            address: 0x8_0000_0000,
            ..framebuffer(16, 16)
        });
        assert!(f.mailbox.latest().is_none());
    }

    #[test]
    fn fences_and_queries_complete_after_submitted_work() {
        let Some(mut f) = fixture() else {
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
}
