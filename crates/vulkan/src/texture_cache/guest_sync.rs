//! Overlap syncs on the GPU. When an image's bytes are shared with overlaps the GPU modified
//! later and no rectangle copy relates them (another size, format or block layout over the same
//! memory), the image's own texels and then every newer overlap's are swizzled into a device
//! buffer of guest bytes, which is unswizzled back into the image. Nothing waits for the GPU and
//! no texel crosses to the host, where a write-back and re-upload through guest memory would.

use ash::vk;
use gpu_allocator::MemoryLocation;
use lichii_texture::PixelFormat;
use lichii_texture::block_linear::GobLayout;

use super::{
    CachedImage, GuestLayout, ImageId, ImageKind, ResolutionScale, TextureCache, Transfer,
};
use crate::commands::ImageAccess;
use crate::descriptors::DescriptorAllocator;
use crate::device::Device;
use crate::error::Result;
use crate::format::Conversion;
use crate::host_compute::HostCompute;
use crate::memory::{Buffer, aspect_of};
use crate::shaders;

/// The workgroup size of `guest_bytes.comp`.
const GROUP: [u32; 2] = [32, 8];

/// How a host texel's aspects make up the guest's 32-bit word, as `guest_bytes.comp` packs it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Packing {
    /// The host texel is the guest's bytes.
    Plain = 0,
    /// D24 in the low bits, S8 in the high byte (D24UnormS8Uint).
    DepthLowStencilHigh = 1,
    /// S8 in the low byte, D24 above it (S8UintD24Unorm).
    StencilLowDepthHigh = 2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    ToGuest = 0,
    FromGuest = 1,
}

/// The push constants of `guest_bytes.comp`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    row_words: u32,
    height: u32,
    row_gobs: u32,
    block_height_log2: u32,
    switch2_layout: u32,
    guest_offset: i32,
    guest_limit: u32,
    direction: u32,
    packing: u32,
    stencil_pitch: u32,
}

/// The pass and its buffers, grown to the largest image synced.
pub(super) struct GuestSync {
    compute: HostCompute,
    descriptors: DescriptorAllocator,
    guest: Option<Buffer>,
    linear: Option<Buffer>,
    stencil: Option<Buffer>,
}

impl GuestSync {
    fn new(device: &std::sync::Arc<Device>) -> Result<Self> {
        Ok(Self {
            compute: HostCompute::new(
                device,
                shaders::GUEST_BYTES_COMP,
                3,
                size_of::<Params>() as u32,
            )?,
            descriptors: DescriptorAllocator::new(device)?,
            guest: None,
            linear: None,
            stencil: None,
        })
    }
}

/// How `image` packs into guest words on the GPU, or `None` when the pass can't move it: only
/// 2D block-linear images at native scale whose texels are whole 32-bit words, laid out as the
/// guest's, or D24S8 kept as the host's D24S8.
fn packing(image: &CachedImage) -> Option<Packing> {
    let info = &image.info;
    let block_linear = matches!(
        info.layout,
        GuestLayout::BlockLinear {
            block_depth_log2: 0,
            ..
        }
    );
    let shape = block_linear
        && image.contiguous()
        && image.scale == ResolutionScale::X1
        && matches!(info.kind, ImageKind::D2)
        && info.depth == 1
        && info.samples == 1
        && info.format.block_width() == 1
        && info.format.block_height() == 1
        && info.format.bytes_per_block().is_multiple_of(4);
    if !shape {
        return None;
    }
    let host = image.host.format;
    match info.format {
        PixelFormat::D24UnormS8Uint if host == vk::Format::D24_UNORM_S8_UINT => {
            Some(Packing::DepthLowStencilHigh)
        }
        PixelFormat::S8UintD24Unorm if host == vk::Format::D24_UNORM_S8_UINT => {
            Some(Packing::StencilLowDepthHigh)
        }
        _ if image.host.conversion == Conversion::None
            && !aspect_of(host).contains(vk::ImageAspectFlags::STENCIL) =>
        {
            Some(Packing::Plain)
        }
        _ => None,
    }
}

fn barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
    // SAFETY: `cmd` is recording outside any rendering.
    unsafe {
        device.raw.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(&[barrier]),
        );
    }
}

/// Grows `slot` to at least `size` bytes, retiring the buffer it replaces.
fn ensure(
    device: &std::sync::Arc<Device>,
    tx: &mut Transfer<'_>,
    slot: &mut Option<Buffer>,
    size: u64,
    usage: vk::BufferUsageFlags,
    name: &'static str,
) -> Result<vk::Buffer> {
    if slot.as_ref().is_none_or(|b| b.size() < size) {
        let buffer = Buffer::new(
            device,
            size.next_power_of_two(),
            usage,
            MemoryLocation::GpuOnly,
            name,
        )?;
        if let Some(old) = slot.replace(buffer) {
            tx.scheduler.retire(old);
        }
    }
    Ok(slot.as_ref().expect("just ensured").raw())
}

impl TextureCache {
    /// Brings `id` up to date with `newer`, the overlaps modified after it last held the
    /// shared bytes, on the GPU. Returns false, having recorded nothing, when the pass can't
    /// move one of the images or `id` must take guest memory as it is (the CPU wrote it, or
    /// removed overlaps left their texels only there).
    pub(super) fn sync_on_gpu(
        &mut self,
        tx: &mut Transfer<'_>,
        id: ImageId,
        newer: &mut [ImageId],
    ) -> Result<bool> {
        let image = self.get(id).expect("a live image id");
        let base = image.base();
        let limit = image.info.size_bytes();
        let fits = |o: &CachedImage| {
            (o.base() as i64 - base as i64).unsigned_abs() + o.info.size_bytes() < 1 << 31
        };
        if image.cpu_dirty
            || !image.keep_out.is_empty()
            || packing(image).is_none()
            || limit >= 1 << 31
            || newer.iter().any(|&o| {
                let o = self.get(o).expect("overlaps are live");
                o.cpu_dirty || packing(o).is_none() || !fits(o)
            })
        {
            return Ok(false);
        }
        newer.sort_by_key(|&o| self.get(o).map_or(0, |o| o.modified));
        let mut linear = 4;
        let mut stencil = 4;
        for &o in newer.iter().chain(std::iter::once(&id)) {
            let info = &self.get(o).expect("a live image id").info;
            let (w, h, _) = info.level_extent(0);
            linear =
                linear.max(u64::from(w) * u64::from(h) * u64::from(info.format.bytes_per_block()));
            stencil = stencil.max((u64::from(w) * u64::from(h)).next_multiple_of(4));
        }
        if self.guest_sync.is_none() {
            self.guest_sync = Some(GuestSync::new(&self.device)?);
        }
        let device = self.device.clone();
        let sync = self.guest_sync.as_mut().expect("just created");
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        ensure(
            &device,
            tx,
            &mut sync.guest,
            limit.max(4),
            storage,
            "guest bytes",
        )?;
        ensure(
            &device,
            tx,
            &mut sync.linear,
            linear,
            storage,
            "guest sync texels",
        )?;
        ensure(
            &device,
            tx,
            &mut sync.stencil,
            stencil,
            storage,
            "guest sync stencil",
        )?;
        let cmd = tx.scheduler.record()?;
        self.paint(tx, cmd, id, base, limit, Direction::ToGuest)?;
        for &o in newer.iter() {
            self.paint(tx, cmd, o, base, limit, Direction::ToGuest)?;
        }
        self.paint(tx, cmd, id, base, limit, Direction::FromGuest)?;
        let tick = self.tick;
        let image = self.image(id);
        image.current = tick;
        image.gpu_modified = true;
        for alias in &mut image.aliases {
            alias.synced = 0;
        }
        self.traffic.gpu_syncs += 1;
        Ok(true)
    }

    /// Records the move of every subresource of `id` between the image and the guest bytes
    /// buffer, which holds `limit` bytes from device address `base`.
    fn paint(
        &mut self,
        tx: &mut Transfer<'_>,
        cmd: vk::CommandBuffer,
        id: ImageId,
        base: u64,
        limit: u64,
        direction: Direction,
    ) -> Result<()> {
        let device = self.device.clone();
        let image = self.get_mut(id).expect("a live image id");
        let packing = packing(image).expect("checked by sync_on_gpu");
        let info = image.info;
        let raw = image.image.raw();
        let levels = info.levels.min(image.image.desc().mip_levels);
        let host = image.host.format;
        let start = image.base() as i64 - base as i64;
        let access = match direction {
            Direction::ToGuest => ImageAccess::TRANSFER_SRC,
            Direction::FromGuest => ImageAccess::TRANSFER_DST,
        };
        image.transition(&device, cmd, access);
        let main_aspect = if packing == Packing::Plain {
            aspect_of(host)
        } else {
            vk::ImageAspectFlags::DEPTH
        };
        let sync = self.guest_sync.as_mut().expect("created by sync_on_gpu");
        let guest = sync.guest.as_ref().expect("ensured").raw();
        let linear = sync.linear.as_ref().expect("ensured").raw();
        let stencil = sync.stencil.as_ref().expect("ensured").raw();
        for layer in 0..info.layers {
            for level in 0..levels {
                let offset = start + info.subresource_offset(level, layer) as i64;
                if offset >= limit as i64 || offset + info.level_size(level) as i64 <= 0 {
                    continue;
                }
                let tiling = info.surface().level_tiling(level);
                let (w, h, _) = info.level_extent(level);
                let row_words = w * tiling.bytes_per_block / 4;
                let region = |aspect_mask| vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask,
                        mip_level: level,
                        base_array_layer: layer,
                        layer_count: 1,
                    },
                    image_extent: vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    },
                    ..Default::default()
                };
                let params = Params {
                    row_words,
                    height: h,
                    row_gobs: tiling.row_gobs,
                    block_height_log2: tiling.block_height_log2,
                    switch2_layout: u32::from(tiling.layout == GobLayout::Switch2),
                    guest_offset: offset as i32,
                    guest_limit: limit as u32,
                    direction: direction as u32,
                    packing: packing as u32,
                    stencil_pitch: w,
                };
                let set = sync
                    .descriptors
                    .allocate(tx.scheduler, sync.compute.set_layout)?;
                let buffers = [guest, linear, stencil].map(|buffer| vk::DescriptorBufferInfo {
                    buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                });
                let groups = [row_words.div_ceil(GROUP[0]), h.div_ceil(GROUP[1]), 1];
                let all = (
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                );
                // SAFETY: `cmd` is recording outside any rendering; the image is in the layout
                // `access` names and every buffer holds the subresource's texels.
                unsafe {
                    match direction {
                        Direction::ToGuest => {
                            device.raw.cmd_copy_image_to_buffer(
                                cmd,
                                raw,
                                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                                linear,
                                &[region(main_aspect)],
                            );
                            if packing != Packing::Plain {
                                device.raw.cmd_copy_image_to_buffer(
                                    cmd,
                                    raw,
                                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                                    stencil,
                                    &[region(vk::ImageAspectFlags::STENCIL)],
                                );
                            }
                            barrier(&device, cmd);
                            sync.compute.dispatch(
                                cmd,
                                set,
                                &buffers,
                                bytemuck::bytes_of(&params),
                                groups,
                                all,
                            );
                        }
                        Direction::FromGuest => {
                            if packing != Packing::Plain {
                                device
                                    .raw
                                    .cmd_fill_buffer(cmd, stencil, 0, vk::WHOLE_SIZE, 0);
                            }
                            barrier(&device, cmd);
                            sync.compute.dispatch(
                                cmd,
                                set,
                                &buffers,
                                bytemuck::bytes_of(&params),
                                groups,
                                all,
                            );
                            device.raw.cmd_copy_buffer_to_image(
                                cmd,
                                linear,
                                raw,
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                &[region(main_aspect)],
                            );
                            if packing != Packing::Plain {
                                device.raw.cmd_copy_buffer_to_image(
                                    cmd,
                                    stencil,
                                    raw,
                                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                    &[region(vk::ImageAspectFlags::STENCIL)],
                                );
                            }
                        }
                    }
                }
                barrier(&device, cmd);
            }
        }
        Ok(())
    }
}
