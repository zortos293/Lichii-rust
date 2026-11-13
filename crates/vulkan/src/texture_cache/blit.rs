//! 2D engine blits on the GPU: one scaled `vkCmdBlitImage` between two cached images, for formats
//! the host blits as they are.

use ash::vk;
use lichii_common::GpuVAddr;
use lichii_gpu::gmmu::Gmmu;
use lichii_texture::{Numeric, PixelFormat};

use super::{Found, ImageInfo, ImageKind, TextureCache, Transfer};
use crate::commands::ImageAccess;
use crate::error::Result;
use crate::format::{Conversion, HostFormat};
use crate::memory::aspect_of;

/// One side of a blit: the guest image at `gpu_va` and the rectangle of it from (`rect[0]`,
/// `rect[1]`) to (`rect[2]`, `rect[3]`) in texels. A reversed pair mirrors the blit.
#[derive(Clone, Copy, Debug)]
pub struct BlitSide {
    pub gpu_va: GpuVAddr,
    pub info: ImageInfo,
    pub rect: [i32; 4],
}

impl TextureCache {
    /// Blits `src.rect` over `dst.rect` on the GPU, filtering linearly when `linear`.
    /// `Ok(false)` when the cache cannot, and the caller blits in software: a format the host
    /// does not blit as it is, an image the lookup resolves to another format, a rectangle
    /// outside its image, or one image on both sides.
    pub fn blit(
        &mut self,
        tx: &mut Transfer<'_>,
        gmmu: &Gmmu,
        src: &BlitSide,
        dst: &BlitSide,
        linear: bool,
    ) -> Result<bool> {
        let src_host = self.host(src.info.format);
        let dst_host = self.host(dst.info.format);
        if src.info.kind != ImageKind::D2
            || dst.info.kind != ImageKind::D2
            || !self.blittable(src.info.format, src_host, dst.info.format, dst_host)
        {
            return Ok(false);
        }
        let Some(source) = self.find(tx, gmmu, src.gpu_va, &src.info)? else {
            return Ok(false);
        };
        let whole = dst.rect == [0, 0, dst.info.width as i32, dst.info.height as i32];
        let Some(target) = self.render_target(tx, gmmu, dst.gpu_va, &dst.info, whole)? else {
            return Ok(false);
        };
        if source.id == target.id {
            return Ok(false);
        }
        let (Some(src_offsets), Some(dst_offsets)) = (
            self.blit_offsets(source, src.info.format, src.rect),
            self.blit_offsets(target, dst.info.format, dst.rect),
        ) else {
            return Ok(false);
        };
        let filter = if linear
            && src.info.format.is_color()
            && !src.info.format.is_integer()
            && self
                .device
                .format_features(src_host.format)
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            vk::Filter::LINEAR
        } else {
            vk::Filter::NEAREST
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
                filter,
            );
        }
        self.mark_modified(target.id);
        Ok(true)
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
