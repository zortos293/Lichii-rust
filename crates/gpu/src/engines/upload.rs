//! Inline-to-memory transfers: data words written through a method land in GPU memory, pitch-
//! linear or block-linear. The KeplerMemory engine (A140), the 3D engine and the compute engine
//! all carry these registers.

use lichii_texture::block_linear::{GOB_HEIGHT, Region, Tiling, swizzle_region};

use super::EngineContext;
pub use super::maxwell_3d::regs::{UploadRegisters, UploadRegistersDest};

/// One transfer in progress. Its buffer is reused across transfers.
#[derive(Default)]
pub struct Upload {
    regs: Option<UploadRegisters>,
    linear: bool,
    buffer: Vec<u8>,
    size: usize,
    tiled: Vec<u8>,
}

impl Upload {
    /// LAUNCH_DMA / EXEC: starts a transfer of `line_length_in * line_count` bytes.
    pub fn exec(&mut self, regs: &UploadRegisters, linear: bool) {
        self.regs = Some(*regs);
        self.linear = linear;
        self.size = regs.line_length_in as usize * regs.line_count as usize;
        self.buffer.clear();
    }

    /// LOAD_INLINE_DATA: one data word. The transfer is written once all its bytes arrived.
    pub fn data(&mut self, ctx: &mut EngineContext<'_>, word: u32) {
        let left = self.size - self.buffer.len();
        if self.regs.is_none() || left == 0 {
            return;
        }
        self.buffer
            .extend_from_slice(&word.to_le_bytes()[..left.min(4)]);
        if self.buffer.len() == self.size {
            self.finish(ctx);
        }
    }

    fn finish(&mut self, ctx: &mut EngineContext<'_>) {
        let Some(regs) = self.regs.take() else {
            return;
        };
        let dest = regs.dest;
        let address = dest.address();
        if self.linear {
            let line = regs.line_length_in as usize;
            for (i, chunk) in self.buffer.chunks(line.max(1)).enumerate() {
                let line_address = address + i as u64 * u64::from(dest.pitch);
                if let Err(fault) = ctx.memory.write_block(line_address, chunk) {
                    tracing::warn!(?fault, "inline upload to unmapped GPU memory");
                }
            }
            let span = u64::from(dest.pitch) * u64::from(regs.line_count.saturating_sub(1))
                + u64::from(regs.line_length_in);
            ctx.rasterizer
                .inline_written(ctx.channel, ctx.memory, address, span);
            return;
        }
        let shift = [dest.width, regs.line_length_in, dest.x, address as u32]
            .into_iter()
            .fold(4, |shift, v| shift.min(v.trailing_zeros()));
        let bytes_per_pixel = 1 << shift;
        let block_height = dest.block_height();
        let tiling = Tiling::new(
            dest.width >> shift,
            dest.height,
            dest.depth.max(1),
            bytes_per_pixel,
            block_height,
            dest.block_depth(),
            ctx.gob_layout,
        );
        let region = Region {
            x: dest.x >> shift,
            y: dest.y,
            z: dest.layer.min(tiling.depth - 1),
            width: regs.line_length_in >> shift,
            height: regs.line_count,
            depth: 1,
        };
        if region.x + region.width > tiling.width || region.y + region.height > tiling.height {
            tracing::warn!(?region, "block-linear inline upload outside its surface");
            return;
        }
        let (base, band, band_region) = if tiling.depth == 1 {
            let rows = GOB_HEIGHT << block_height;
            let first = region.y / rows * rows;
            let last = (region.y + region.height).div_ceil(rows) * rows;
            let band = Tiling {
                height: last - first,
                ..tiling
            };
            let base = tiling.offset(0, first, 0) as u64;
            (
                base,
                band,
                Region {
                    y: region.y - first,
                    z: 0,
                    ..region
                },
            )
        } else {
            (0, tiling, region)
        };
        self.tiled.resize(band.size_bytes(), 0);
        let target = address + base;
        if let Err(fault) = ctx.memory.read_block(target, &mut self.tiled) {
            tracing::warn!(?fault, "block-linear inline upload to unmapped GPU memory");
        }
        swizzle_region(
            &band,
            band_region,
            &self.buffer,
            &mut self.tiled,
            Some(regs.line_length_in as usize),
        );
        if let Err(fault) = ctx.memory.write_block(target, &self.tiled) {
            tracing::warn!(?fault, "block-linear inline upload to unmapped GPU memory");
        }
        ctx.rasterizer
            .inline_written(ctx.channel, ctx.memory, target, self.tiled.len() as u64);
    }
}
