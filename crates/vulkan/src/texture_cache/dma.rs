//! Copy engine transfers out of images the GPU wrote: one `vkCmdCopyImageToBuffer` into the
//! buffer cache, so the texels stay on the GPU until the guest reads the destination.

use ash::vk;
use lichii_gpu::backend::ImageCopy;
use lichii_gpu::gmmu::{Gmmu, RangeTarget};
use lichii_texture::block_linear::Tiling;

use super::{GuestLayout, ImageKind, ResolutionScale, Segment, TextureCache, Transfer};
use crate::commands::{ImageAccess, transfer_to_all_barrier};
use crate::error::Result;
use crate::format::Conversion;

impl TextureCache {
    /// Copies a copy engine transfer from a block-linear surface into a pitch-linear one on the
    /// GPU, when a GPU-modified image starts at the source and tiles as the copy engine
    /// addresses the surface. The texels land in the buffer cache, marked GPU-written, until a
    /// read of the destination flushes them. `Ok(false)` when the cache cannot and the engine
    /// copies in software: no such image, a host format that is not the guest's bytes, a
    /// rectangle outside the image, or a destination that is not one device range clear of
    /// cached images.
    pub fn copy_to_buffer(
        &mut self,
        tx: &mut Transfer<'_>,
        gmmu: &Gmmu,
        copy: &ImageCopy,
    ) -> Result<bool> {
        let (src, dst) = (&copy.src, &copy.dst);
        let bpp = src.bytes_per_pixel;
        let Some((0, block_height_log2, block_depth_log2)) = src.block else {
            return Ok(false);
        };
        if src.depth > 1
            || src.layer != 0
            || dst.block.is_some()
            || dst.bytes_per_pixel != bpp
            || bpp == 0
            || copy.width == 0
            || copy.height == 0
            || u64::from(dst.pitch) < u64::from(copy.width) * u64::from(bpp)
        {
            return Ok(false);
        }
        let addressed = Tiling::new(
            src.width,
            src.height,
            1,
            bpp,
            block_height_log2,
            block_depth_log2,
            self.gob_layout,
        );
        let Some(source) = self.single_range(gmmu, src.address, addressed.size_bytes() as u64)
        else {
            return Ok(false);
        };
        let first = u64::from(copy.dst_origin.1) * u64::from(dst.pitch)
            + u64::from(copy.dst_origin.0) * u64::from(bpp);
        let span = u64::from(copy.height - 1) * u64::from(dst.pitch)
            + u64::from(copy.width) * u64::from(bpp);
        let Some(target) = self.single_range(gmmu, dst.address + first, span) else {
            return Ok(false);
        };
        self.collect_overlaps(&[target]);
        let clear = self.ids.iter().all(|&id| {
            self.get(id).is_none_or(|image| {
                image.segments.iter().all(|s| {
                    s.daddr >= target.daddr + target.size || target.daddr >= s.daddr + s.size
                })
            })
        });
        if !clear {
            return Ok(false);
        }
        let shape = |t: &Tiling| {
            (
                t.row_gobs,
                t.block_height_log2,
                t.block_depth_log2,
                t.layout,
            )
        };
        self.collect_overlaps(&[source]);
        let found = self.ids.iter().copied().find(|&id| {
            self.get(id).is_some_and(|image| {
                let info = &image.info;
                image.gpu_modified
                    && image.scale == ResolutionScale::X1
                    && image.contiguous()
                    && image.base() == source.daddr
                    && matches!(info.layout, GuestLayout::BlockLinear { .. })
                    && info.kind == ImageKind::D2
                    && info.format.is_color()
                    && info.format.block_width() == 1
                    && info.format.block_height() == 1
                    && image.host.conversion == Conversion::None
                    && shape(&info.surface().level_tiling(0)) == shape(&addressed)
            })
        });
        let Some(id) = found else {
            return Ok(false);
        };
        let image = self.get(id).expect("a live image id");
        let bpb = image.info.format.bytes_per_block();
        let (width, height, _) = image.info.level_extent(0);
        let (x_bytes, w_bytes) = (copy.src_origin.0 * bpp, copy.width * bpp);
        let (x, w) = (x_bytes / bpb, w_bytes / bpb);
        let fits = x_bytes.is_multiple_of(bpb)
            && w_bytes.is_multiple_of(bpb)
            && dst.pitch.is_multiple_of(bpb)
            && target.daddr.is_multiple_of(u64::from(bpb))
            && x + w <= width
            && copy.src_origin.1 + copy.height <= height;
        if !fits {
            return Ok(false);
        }
        self.refresh(tx, id)?;
        let binding = tx
            .buffers
            .acquire(tx.scheduler, tx.staging, target.daddr, target.size)?;
        let device = self.device.clone();
        let cmd = tx.scheduler.record()?;
        let raw = self.prepare_live(cmd, id, ImageAccess::TRANSFER_SRC);
        let before = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)];
        let region = vk::BufferImageCopy {
            buffer_offset: binding.offset,
            buffer_row_length: dst.pitch / bpb,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D {
                x: x as i32,
                y: copy.src_origin.1 as i32,
                z: 0,
            },
            image_extent: vk::Extent3D {
                width: w,
                height: copy.height,
                depth: 1,
            },
        };
        // SAFETY: `cmd` is recording outside any rendering, the image is in TRANSFER_SRC
        // layout, the region lies inside its level 0, and the buffer holds the destination bytes
        // from `binding.offset`.
        unsafe {
            device.raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&before),
            );
            device.raw.cmd_copy_image_to_buffer(
                cmd,
                raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                binding.buffer,
                &[region],
            );
        }
        transfer_to_all_barrier(&device, cmd);
        tx.buffers.mark_gpu_written(target.daddr, target.size);
        Ok(true)
    }

    /// The one device range `[gpu_va, +size)` maps to, or `None` when it is split or unmapped.
    fn single_range(
        &self,
        gmmu: &Gmmu,
        gpu_va: lichii_common::GpuVAddr,
        size: u64,
    ) -> Option<Segment> {
        let mut found = None;
        let mut pieces = 0;
        gmmu.for_each_range(gpu_va, size, |range| {
            pieces += 1;
            if let RangeTarget::Device(daddr) = range.target {
                found = Some(Segment {
                    offset: 0,
                    daddr,
                    size: range.size,
                });
            }
        });
        found.filter(|s| pieces == 1 && s.size == size)
    }
}
