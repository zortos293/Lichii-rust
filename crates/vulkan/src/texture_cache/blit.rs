//! 2D engine blits on the GPU: one scaled `vkCmdBlitImage` between two cached images, for formats
//! the host blits as they are, or a gather of raw texels at the engine's own nearest sample
//! positions, which `vkCmdBlitImage` can't reproduce.

use ash::vk;
use lichii_common::GpuVAddr;
use lichii_gpu::gmmu::{Gmmu, RangeTarget};
use lichii_texture::{Numeric, PixelFormat};

use super::guest_sync::{barrier, ensure};
use super::{Found, ImageInfo, ImageKind, ResolutionScale, Segment, TextureCache, Transfer};
use crate::commands::ImageAccess;
use crate::descriptors::DescriptorAllocator;
use crate::device::Device;
use crate::error::Result;
use crate::format::{Conversion, HostFormat};
use crate::host_compute::HostCompute;
use crate::memory::{Buffer, aspect_of};
use crate::shaders;

/// One side of a blit: the guest image at `gpu_va` and the rectangle of it from (`rect[0]`,
/// `rect[1]`) to (`rect[2]`, `rect[3]`) in texels. A reversed pair mirrors the blit.
#[derive(Clone, Copy, Debug)]
pub struct BlitSide {
    pub gpu_va: GpuVAddr,
    pub info: ImageInfo,
    pub rect: [i32; 4],
}

/// Where a nearest-filtered blit samples its source, as the software blit does: destination
/// texel `i` along an axis, counted from the destination rectangle's start, takes source texel
/// `(start + i * step) >> 32`, clamped to the texels from the start position's (at least 0) up
/// to the end position's, `(start + count * step) >> 32`, at most `width` or `height`, the
/// source surface's extent. Starts and steps are 32.32 fixed point and wrap.
#[derive(Clone, Copy, Debug)]
pub struct NearestSampling {
    pub x0: i64,
    pub y0: i64,
    pub du: i64,
    pub dv: i64,
    pub width: u32,
    pub height: u32,
}

/// The workgroup size of `blit_gather.comp`.
const GATHER_GROUP: u32 = 8;

/// The push constants of `blit_gather.comp`. 64-bit values are low word first; `padding`
/// aligns them.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherParams {
    dst_width: u32,
    dst_height: u32,
    words: u32,
    box_width: u32,
    box_height: u32,
    padding: u32,
    x0: [u32; 2],
    du: [u32; 2],
    y0: [u32; 2],
    dv: [u32; 2],
}

/// The gather pass and its raw texel buffers, grown to the largest blit.
pub(super) struct Gather {
    compute: HostCompute,
    descriptors: DescriptorAllocator,
    src: Option<Buffer>,
    dst: Option<Buffer>,
}

impl Gather {
    fn new(device: &std::sync::Arc<Device>) -> Result<Self> {
        Ok(Self {
            compute: HostCompute::new(
                device,
                shaders::BLIT_GATHER_COMP,
                2,
                size_of::<GatherParams>() as u32,
            )?,
            descriptors: DescriptorAllocator::new(device)?,
            src: None,
            dst: None,
        })
    }
}

/// The source texels `[first, end)` an axis of `count` samples may read, as the software blit
/// bounds them, or `None` when that range is empty, as for a reversed or zero step.
fn axis_span(start: i64, step: i64, count: u32, extent: u32) -> Option<(u32, u32)> {
    let first = ((start >> 32) as i32).max(0);
    let end = (start.wrapping_add(step.wrapping_mul(i64::from(count))) >> 32) as i32;
    let end = end.min(extent.min(i32::MAX as u32) as i32);
    (first < end).then_some((first as u32, end as u32))
}

fn words(value: i64) -> [u32; 2] {
    [value as u32, (value >> 32) as u32]
}

impl TextureCache {
    /// Blits `src.rect` over `dst.rect` on the GPU, sampling the nearest texel.
    /// `Ok(false)` when the cache cannot, and the caller blits in software: a format the host
    /// does not blit as it is, a source no GPU work wrote (guest memory holds its texels, which
    /// the software blit reads without an upload), an image the lookup resolves to another
    /// format, a rectangle outside its image, or one image on both sides.
    pub fn blit(
        &mut self,
        tx: &mut Transfer<'_>,
        gmmu: &Gmmu,
        src: &BlitSide,
        dst: &BlitSide,
    ) -> Result<bool> {
        let src_host = self.host(src.info.format);
        let dst_host = self.host(dst.info.format);
        if src.info.kind != ImageKind::D2
            || dst.info.kind != ImageKind::D2
            || !self.blittable(src.info.format, src_host, dst.info.format, dst_host)
        {
            return Ok(false);
        }
        let Some((source, target)) = self.blit_images(tx, gmmu, src.gpu_va, &src.info, dst)? else {
            return Ok(false);
        };
        let (Some(src_offsets), Some(dst_offsets)) = (
            self.blit_offsets(source, src.info.format, src.rect),
            self.blit_offsets(target, dst.info.format, dst.rect),
        ) else {
            return Ok(false);
        };
        let device = self.device.clone();
        let cmd = tx.scheduler.record()?;
        let src_image = self.prepare_live(cmd, source.id, ImageAccess::TRANSFER_SRC);
        let dst_image = self.prepare_live(cmd, target.id, ImageAccess::TRANSFER_DST);
        let layers = |found: Found, format: vk::Format| vk::ImageSubresourceLayers {
            aspect_mask: aspect_of(format),
            mip_level: found.level,
            base_array_layer: found.layer,
            layer_count: 1,
        };
        let region = vk::ImageBlit {
            src_subresource: layers(source, src_host.format),
            src_offsets,
            dst_subresource: layers(target, dst_host.format),
            dst_offsets,
        };
        // SAFETY: the source is in TRANSFER_SRC and the target in TRANSFER_DST layout, they are
        // different images, and both offset pairs lie inside their subresources.
        unsafe {
            device.raw.cmd_blit_image(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
                vk::Filter::NEAREST,
            );
        }
        self.mark_modified(target.id);
        Ok(true)
    }

    /// Blits the source image at `src_va` over `dst.rect` on the GPU, sampling the nearest texel
    /// where `sampling` puts it, by moving raw host texels between images of one format.
    /// `Ok(false)` when the cache cannot, and the caller blits in software: another format on
    /// either side, a host format that converts, has a stencil plane or texels of other than whole
    /// 32-bit words, a source no GPU work wrote, an image the lookup resolves to another format
    /// or scale, a destination rectangle outside its image, or one image on both sides.
    pub fn gather(
        &mut self,
        tx: &mut Transfer<'_>,
        gmmu: &Gmmu,
        src_va: GpuVAddr,
        src: &ImageInfo,
        dst: &BlitSide,
        sampling: &NearestSampling,
    ) -> Result<bool> {
        let format = src.format;
        let host = self.host(format);
        let bytes = format.bytes_per_block();
        let [x0, y0, x1, y1] = dst.rect;
        if src.kind != ImageKind::D2
            || dst.info.kind != ImageKind::D2
            || dst.info.format != format
            || host.conversion != Conversion::None
            || aspect_of(host.format).contains(vk::ImageAspectFlags::STENCIL)
            || format.block_width() != 1
            || format.block_height() != 1
            || !bytes.is_multiple_of(4)
            || x0 < 0
            || y0 < 0
            || x1 <= x0
            || y1 <= y0
            || sampling.width == 0
            || sampling.height == 0
        {
            return Ok(false);
        }
        let (width, height) = ((x1 - x0) as u32, (y1 - y0) as u32);
        let (Some((box_x, box_x1)), Some((box_y, box_y1))) = (
            axis_span(sampling.x0, sampling.du, width, sampling.width),
            axis_span(sampling.y0, sampling.dv, height, sampling.height),
        ) else {
            return Ok(false);
        };
        let Some((source, target)) = self.blit_images(tx, gmmu, src_va, src, dst)? else {
            return Ok(false);
        };
        let holds = |found: Found, width: u32, height: u32| {
            self.get(found.id).is_some_and(|image| {
                let extent = image.host_extent(found.level);
                image.info.format == format
                    && image.scale == ResolutionScale::X1
                    && width <= extent.width
                    && height <= extent.height
            })
        };
        if !holds(source, box_x1, box_y1) || !holds(target, x1 as u32, y1 as u32) {
            return Ok(false);
        }
        let (box_width, box_height) = (box_x1 - box_x, box_y1 - box_y);
        let device = self.device.clone();
        if self.gather.is_none() {
            self.gather = Some(Gather::new(&device)?);
        }
        let gather = self.gather.as_mut().expect("just created");
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let src_bytes = u64::from(box_width) * u64::from(box_height) * u64::from(bytes);
        let dst_bytes = u64::from(width) * u64::from(height) * u64::from(bytes);
        let src_buffer = ensure(
            &device,
            tx,
            &mut gather.src,
            src_bytes,
            usage,
            "blit gather source",
        )?;
        let dst_buffer = ensure(
            &device,
            tx,
            &mut gather.dst,
            dst_bytes,
            usage,
            "blit gather target",
        )?;
        let set = gather
            .descriptors
            .allocate(tx.scheduler, gather.compute.set_layout)?;
        let cmd = tx.scheduler.record()?;
        let src_image = self.prepare_live(cmd, source.id, ImageAccess::TRANSFER_SRC);
        let dst_image = self.prepare_live(cmd, target.id, ImageAccess::TRANSFER_DST);
        // NVIDIA 610.88 copies stale texels out of a source written earlier in the command
        // buffer when another image's first upload sits between, although the image barrier
        // orders them; only a memory barrier into every stage's reads fixes it.
        barrier(&device, cmd);
        let aspect_mask = aspect_of(host.format);
        let subresource = |found: Found| vk::ImageSubresourceLayers {
            aspect_mask,
            mip_level: found.level,
            base_array_layer: found.layer,
            layer_count: 1,
        };
        let params = GatherParams {
            dst_width: width,
            dst_height: height,
            words: bytes / 4,
            box_width,
            box_height,
            padding: 0,
            x0: words(sampling.x0.wrapping_sub(i64::from(box_x) << 32)),
            du: words(sampling.du),
            y0: words(sampling.y0.wrapping_sub(i64::from(box_y) << 32)),
            dv: words(sampling.dv),
        };
        let buffers = [src_buffer, dst_buffer].map(|buffer| vk::DescriptorBufferInfo {
            buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        });
        // SAFETY: `cmd` is recording outside any rendering; the source is in TRANSFER_SRC and
        // the target in TRANSFER_DST layout, and both regions lie inside their subresources and
        // fit the buffers, which `ensure` grew to the box and the destination rectangle.
        unsafe {
            device.raw.cmd_copy_image_to_buffer(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_buffer,
                &[vk::BufferImageCopy {
                    buffer_row_length: box_width,
                    image_subresource: subresource(source),
                    image_offset: vk::Offset3D {
                        x: box_x as i32,
                        y: box_y as i32,
                        z: 0,
                    },
                    image_extent: vk::Extent3D {
                        width: box_width,
                        height: box_height,
                        depth: 1,
                    },
                    ..Default::default()
                }],
            );
            barrier(&device, cmd);
        }
        let gather = self.gather.as_ref().expect("just created");
        gather.compute.dispatch(
            cmd,
            set,
            &buffers,
            bytemuck::bytes_of(&params),
            [
                width.div_ceil(GATHER_GROUP),
                height.div_ceil(GATHER_GROUP),
                1,
            ],
            (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            ),
        );
        // SAFETY: as above; the dispatch's barrier made its writes visible to the copy.
        unsafe {
            device.raw.cmd_copy_buffer_to_image(
                cmd,
                dst_buffer,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy {
                    image_subresource: subresource(target),
                    image_offset: vk::Offset3D { x: x0, y: y0, z: 0 },
                    image_extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                    ..Default::default()
                }],
            );
        }
        self.mark_modified(target.id);
        Ok(true)
    }

    /// The source and target of a blit into `dst`: `None` when no GPU work wrote the source,
    /// either lookup fails, or one image is on both sides. The target is readied as a render
    /// target, discarding its contents when the blit covers all of it.
    fn blit_images(
        &mut self,
        tx: &mut Transfer<'_>,
        gmmu: &Gmmu,
        src_va: GpuVAddr,
        src: &ImageInfo,
        dst: &BlitSide,
    ) -> Result<Option<(Found, Found)>> {
        if !self.gpu_holds(gmmu, src_va, src.size_bytes()) {
            return Ok(None);
        }
        let Some(source) = self.find(tx, gmmu, src_va, src)? else {
            return Ok(None);
        };
        let Some(target) = self.lookup_or_create(tx, gmmu, dst.gpu_va, &dst.info)? else {
            return Ok(None);
        };
        if source.id == target.id {
            return Ok(None);
        }
        let whole = dst.rect == [0, 0, dst.info.width as i32, dst.info.height as i32];
        Ok(self
            .render_target(tx, gmmu, dst.gpu_va, &dst.info, whole)?
            .map(|target| (source, target)))
    }

    /// Whether a GPU-modified image overlaps the `size` bytes at `gpu_va`, so texels there are
    /// newer on the GPU than in guest memory.
    fn gpu_holds(&mut self, gmmu: &Gmmu, gpu_va: GpuVAddr, size: u64) -> bool {
        let mut ranges = std::mem::take(&mut self.segments);
        ranges.clear();
        gmmu.for_each_range(gpu_va, size, |range| {
            if let RangeTarget::Device(daddr) = range.target {
                ranges.push(Segment {
                    offset: 0,
                    daddr,
                    size: range.size,
                });
            }
        });
        self.collect_overlaps(&ranges);
        self.segments = ranges;
        self.ids
            .iter()
            .any(|&id| self.get(id).is_some_and(|image| image.gpu_modified))
    }

    /// Whether `vkCmdBlitImage` turns `src` texels into `dst` ones as the guest does: host
    /// formats stored as they are and blittable, colour formats of one integer class, and
    /// depth/stencil formats only into the same host format.
    fn blittable(
        &self,
        src: PixelFormat,
        src_host: HostFormat,
        dst: PixelFormat,
        dst_host: HostFormat,
    ) -> bool {
        let plain = |host: &HostFormat| {
            let s = host.swizzle;
            host.conversion == Conversion::None
                && [s.r, s.g, s.b, s.a]
                    .iter()
                    .all(|&c| c == vk::ComponentSwizzle::IDENTITY)
        };
        let class = |format: PixelFormat| match format.numeric() {
            Numeric::Uint => 1,
            Numeric::Sint => 2,
            _ => 0,
        };
        let compatible = if src.is_color() && dst.is_color() {
            class(src) == class(dst)
        } else {
            src_host.format == dst_host.format
        };
        compatible
            && plain(&src_host)
            && plain(&dst_host)
            && self.can_blit(src_host.format)
            && self.can_blit(dst_host.format)
    }

    /// `rect` of the found image's subresource in host texels, at the image's scale; `None`
    /// when the image holds another format or the rectangle is empty or leaves the image.
    fn blit_offsets(
        &self,
        found: Found,
        format: PixelFormat,
        rect: [i32; 4],
    ) -> Option<[vk::Offset3D; 2]> {
        let image = self.get(found.id)?;
        if image.info.format != format || rect[0] == rect[2] || rect[1] == rect[3] {
            return None;
        }
        let extent = image.host_extent(found.level);
        let (num, den) = image.scale.ratio();
        let scale = |v: i32, max: u32| {
            let v = u64::try_from(v).ok()? * u64::from(num) / u64::from(den);
            (v <= u64::from(max)).then_some(v as i32)
        };
        Some([
            vk::Offset3D {
                x: scale(rect[0], extent.width)?,
                y: scale(rect[1], extent.height)?,
                z: 0,
            },
            vk::Offset3D {
                x: scale(rect[2], extent.width)?,
                y: scale(rect[3], extent.height)?,
                z: 1,
            },
        ])
    }
}
