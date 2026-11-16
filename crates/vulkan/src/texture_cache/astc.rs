//! GPU decode of ASTC uploads for devices that cannot sample ASTC. Staging holds the
//! deswizzled blocks of each subresource; `astc_decode.comp` decodes them to RGBA8 in a
//! device-local scratch buffer, `bc_encode.comp` recompresses that to BC1 or BC3 beside it when
//! the host image is BC, and a buffer copy moves the result into the image.
//!
//! Subresources are cut into bands of block rows that fit the scratch buffer, so a whole mip
//! chain of any size goes through [`SCRATCH_BYTES`] of device memory (or the cap the caller sets): the bands that fit at once
//! decode, encode and copy together, with one barrier between each step, before the next batch
//! reuses the scratch.

use std::ops::Range;
use std::sync::Arc;

use ash::vk;
use lichii_texture::{Compression, PixelFormat};

use super::Transfer;
use super::guest_sync::ensure;
use crate::descriptors::DescriptorAllocator;
use crate::device::Device;
use crate::error::Result;
use crate::host_compute::HostCompute;
use crate::memory::Buffer;
use crate::shaders;

/// The most device memory decoded and recompressed bands take at once.
pub(super) const SCRATCH_BYTES: u64 = 32 << 20;
/// The workgroup size of both shaders, in blocks per axis.
const GROUP: u32 = 8;
const ASTC_BLOCK_BYTES: u64 = 16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DecodeParams {
    blocks_x: u32,
    blocks_y: u32,
    width: u32,
    height: u32,
    block_w: u32,
    block_h: u32,
    srgb: u32,
    in_offset: u32,
    out_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EncodeParams {
    blocks_x: u32,
    blocks_y: u32,
    width: u32,
    height: u32,
    bc3: u32,
    in_offset: u32,
    out_offset: u32,
}

/// One subresource whose ASTC blocks sit in staging: `depth` slices of tightly packed rows of
/// blocks from byte `staging` of the staging buffer, for host `level` of `layer`.
#[derive(Clone, Copy, Debug)]
pub(super) struct AstcLevel {
    pub staging: u64,
    pub layer: u32,
    pub level: u32,
    pub extent: (u32, u32, u32),
}

/// Block rows of one slice of an [`AstcLevel`] and where they decode in the scratch buffer.
#[derive(Clone, Copy, Debug, Default)]
struct Band {
    input: u64,
    blocks_x: u32,
    blocks_y: u32,
    width: u32,
    y: u32,
    rows: u32,
    z: u32,
    layer: u32,
    level: u32,
    rgba: u64,
    bc: u64,
}

impl Band {
    fn input_bytes(&self) -> u64 {
        u64::from(self.blocks_x) * u64::from(self.blocks_y) * ASTC_BLOCK_BYTES
    }

    fn rgba_bytes(&self) -> u64 {
        (u64::from(self.width) * u64::from(self.rows) * 4).next_multiple_of(16)
    }

    fn bc_bytes(&self, bc: Option<Compression>) -> u64 {
        bc.map_or(0, |bc| {
            u64::from(self.width.div_ceil(4))
                * u64::from(self.rows.div_ceil(4))
                * bc_block_bytes(bc)
        })
        .next_multiple_of(16)
    }
}

fn bc_block_bytes(bc: Compression) -> u64 {
    if bc == Compression::Bc1 { 8 } else { 16 }
}

/// The ASTC decode and BC encode pipelines, created once, and the scratch they share.
pub(super) struct AstcDecoder {
    device: Arc<Device>,
    decode: HostCompute,
    encode: HostCompute,
    descriptors: DescriptorAllocator,
    scratch: Option<Buffer>,
    bands: Vec<Band>,
    regions: Vec<vk::BufferImageCopy>,
}

impl AstcDecoder {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            decode: HostCompute::new(
                device,
                shaders::ASTC_DECODE_COMP,
                2,
                size_of::<DecodeParams>() as u32,
            )?,
            encode: HostCompute::new(
                device,
                shaders::BC_ENCODE_COMP,
                2,
                size_of::<EncodeParams>() as u32,
            )?,
            descriptors: DescriptorAllocator::new(device)?,
            scratch: None,
            bands: Vec::new(),
            regions: Vec::new(),
        })
    }

    /// Records the decode of `levels`, ASTC blocks of `format` in `staging` before byte
    /// `staging_end`, into `image` (in TRANSFER_DST layout): as RGBA8, or recompressed to `bc`,
    /// through at most `scratch_bytes` of scratch memory (more when one block row needs it).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        tx: &mut Transfer<'_>,
        staging: vk::Buffer,
        staging_end: u64,
        levels: &[AstcLevel],
        format: PixelFormat,
        bc: Option<Compression>,
        image: vk::Image,
        scratch_bytes: u64,
    ) -> Result<()> {
        let (bw, bh) = (format.block_width(), format.block_height());
        let granule = if bc.is_some() { 4 / gcd(bh, 4) } else { 1 };
        self.bands.clear();
        for level in levels {
            let (w, h, d) = level.extent;
            let (blocks_x, blocks_y) = (w.div_ceil(bw), h.div_ceil(bh));
            let row_cost = u64::from(w) * u64::from(bh) * 4
                + bc.map_or(0, |bc| {
                    u64::from(w.div_ceil(4)) * u64::from(bh.div_ceil(4)) * bc_block_bytes(bc)
                });
            let fit = (scratch_bytes / row_cost.max(1)).min(u64::from(blocks_y)) as u32;
            let step = (fit / granule * granule).max(granule);
            for z in 0..d {
                let slice = level.staging
                    + u64::from(z) * u64::from(blocks_x) * u64::from(blocks_y) * ASTC_BLOCK_BYTES;
                for first in (0..blocks_y).step_by(step as usize) {
                    let band_blocks = step.min(blocks_y - first);
                    let y = first * bh;
                    self.bands.push(Band {
                        input: slice + u64::from(first) * u64::from(blocks_x) * ASTC_BLOCK_BYTES,
                        blocks_x,
                        blocks_y: band_blocks,
                        width: w,
                        y,
                        rows: (band_blocks * bh).min(h - y),
                        z,
                        layer: level.layer,
                        level: level.level,
                        ..Band::default()
                    });
                }
            }
        }
        let (needed, largest) = self.bands.iter().fold((0, 0), |(sum, max), band| {
            let bytes = band.rgba_bytes() + band.bc_bytes(bc);
            (sum + bytes, u64::max(max, bytes))
        });
        if needed == 0 {
            return Ok(());
        }
        let size = needed.min(scratch_bytes).max(largest);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC;
        let device = self.device.clone();
        let scratch = ensure(
            &device,
            tx,
            &mut self.scratch,
            size,
            usage,
            "ASTC decode scratch",
        )?;
        let size = self.scratch.as_ref().expect("just ensured").size();
        let whole = vk::DescriptorBufferInfo {
            buffer: scratch,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let encode_set = match bc {
            Some(_) => {
                let set = self
                    .descriptors
                    .allocate(tx.scheduler, self.encode.set_layout)?;
                self.encode.write_set(set, &[whole, whole]);
                set
            }
            None => vk::DescriptorSet::null(),
        };
        let limits = &device.properties.limits;
        let window = Window {
            staging,
            end: staging_end,
            align: limits.min_storage_buffer_offset_alignment.max(16),
            max_range: u64::from(limits.max_storage_buffer_range) & !15,
            scratch: whole,
            start: 0,
            stop: 0,
            set: vk::DescriptorSet::null(),
        };
        let batch = Batch {
            tx,
            window,
            format,
            bc,
            scratch,
            encode_set,
            image,
        };
        self.record_batches(batch, size)
    }

    fn record_batches(&mut self, mut batch: Batch<'_, '_>, size: u64) -> Result<()> {
        let mut first = 0;
        let mut used = 0;
        for i in 0..self.bands.len() {
            let need = self.bands[i].rgba_bytes() + self.bands[i].bc_bytes(batch.bc);
            if used + need > size {
                self.flush(&mut batch, first..i)?;
                (first, used) = (i, 0);
            }
            let band = &mut self.bands[i];
            band.rgba = used;
            band.bc = used + band.rgba_bytes();
            used += need;
        }
        self.flush(&mut batch, first..self.bands.len())
    }

    /// Decodes, encodes and copies `bands`, which fit in the scratch buffer together.
    fn flush(&mut self, batch: &mut Batch<'_, '_>, bands: Range<usize>) -> Result<()> {
        let device = self.device.clone();
        let cmd = batch.tx.scheduler.record()?;
        let after_copies = barrier(
            vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::SHADER_STORAGE_READ,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        memory_barrier(&device, cmd, after_copies);
        let (bw, bh) = (batch.format.block_width(), batch.format.block_height());
        let mut bound = false;
        for band in &self.bands[bands.clone()] {
            let fresh = batch
                .window
                .cover(&mut self.descriptors, batch.tx, &self.decode, band)?;
            if fresh || !bound {
                self.decode.bind(cmd, batch.window.set);
                bound = true;
            }
            let params = DecodeParams {
                blocks_x: band.blocks_x,
                blocks_y: band.blocks_y,
                width: band.width,
                height: band.rows,
                block_w: bw,
                block_h: bh,
                srgb: u32::from(batch.format.is_srgb()),
                in_offset: ((band.input - batch.window.start) / ASTC_BLOCK_BYTES) as u32,
                out_offset: (band.rgba / 4) as u32,
            };
            self.decode.dispatch_bound(
                cmd,
                bytemuck::bytes_of(&params),
                [
                    band.blocks_x.div_ceil(GROUP),
                    band.blocks_y.div_ceil(GROUP),
                    1,
                ],
            );
        }
        let to_copy = barrier(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
        );
        if let Some(bc) = batch.bc {
            memory_barrier(
                &device,
                cmd,
                barrier(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                ),
            );
            self.encode.bind(cmd, batch.encode_set);
            for band in &self.bands[bands.clone()] {
                let (blocks_x, blocks_y) = (band.width.div_ceil(4), band.rows.div_ceil(4));
                let params = EncodeParams {
                    blocks_x,
                    blocks_y,
                    width: band.width,
                    height: band.rows,
                    bc3: u32::from(bc != Compression::Bc1),
                    in_offset: (band.rgba / 4) as u32,
                    out_offset: (band.bc / 8) as u32,
                };
                self.encode.dispatch_bound(
                    cmd,
                    bytemuck::bytes_of(&params),
                    [blocks_x.div_ceil(GROUP), blocks_y.div_ceil(GROUP), 1],
                );
            }
        }
        memory_barrier(&device, cmd, to_copy);
        self.regions.clear();
        self.regions
            .extend(self.bands[bands].iter().map(|band| vk::BufferImageCopy {
                buffer_offset: if batch.bc.is_some() {
                    band.bc
                } else {
                    band.rgba
                },
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: band.level,
                    base_array_layer: band.layer,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D {
                    x: 0,
                    y: band.y as i32,
                    z: band.z as i32,
                },
                image_extent: vk::Extent3D {
                    width: band.width,
                    height: band.rows,
                    depth: 1,
                },
                ..Default::default()
            }));
        if self.regions.is_empty() {
            return Ok(());
        }
        // SAFETY: `cmd` is recording outside any rendering, the image is in TRANSFER_DST layout,
        // every region lies inside its subresource and the barrier made the bands visible.
        unsafe {
            device.raw.cmd_copy_buffer_to_image(
                cmd,
                batch.scratch,
                batch.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &self.regions,
            )
        };
        Ok(())
    }
}

/// What one [`AstcDecoder::record`] call records into.
struct Batch<'t, 'a> {
    tx: &'t mut Transfer<'a>,
    window: Window,
    format: PixelFormat,
    bc: Option<Compression>,
    scratch: vk::Buffer,
    encode_set: vk::DescriptorSet,
    image: vk::Image,
}

/// The staging bytes `[start, stop)` the current decode set binds, within the device's storage
/// buffer offset alignment and range limits.
struct Window {
    staging: vk::Buffer,
    end: u64,
    align: u64,
    max_range: u64,
    scratch: vk::DescriptorBufferInfo,
    start: u64,
    stop: u64,
    set: vk::DescriptorSet,
}

impl Window {
    /// Makes the decode set bind `band`'s blocks; true when it took a new set.
    fn cover(
        &mut self,
        descriptors: &mut DescriptorAllocator,
        tx: &mut Transfer<'_>,
        decode: &HostCompute,
        band: &Band,
    ) -> Result<bool> {
        let end = band.input + band.input_bytes();
        if self.set != vk::DescriptorSet::null() && band.input >= self.start && end <= self.stop {
            return Ok(false);
        }
        self.start = band.input & !(self.align - 1);
        self.stop = self.end.min(self.start + self.max_range);
        self.set = descriptors.allocate(tx.scheduler, decode.set_layout)?;
        let input = vk::DescriptorBufferInfo {
            buffer: self.staging,
            offset: self.start,
            range: self.stop - self.start,
        };
        decode.write_set(self.set, &[input, self.scratch]);
        Ok(true)
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn barrier(
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) -> vk::MemoryBarrier2<'static> {
    vk::MemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
}

fn memory_barrier(device: &Device, cmd: vk::CommandBuffer, barrier: vk::MemoryBarrier2<'_>) {
    // SAFETY: `cmd` is recording outside any rendering.
    unsafe {
        device.raw.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(&[barrier]),
        )
    };
}
