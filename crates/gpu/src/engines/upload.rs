//! Inline-to-memory transfers: data words written through a method land in GPU memory, pitch-
//! linear or block-linear. The KeplerMemory engine (A140), the 3D engine and the compute engine
//! all carry these registers.
//!
//! Data is written as it arrives, a bounded chunk at a time, straight to the bytes the transfer
//! covers: nothing around them is read back, and no buffer grows with the transfer size. The
//! renderer is told about every chunk as it is written, so a transfer the guest abandons halfway
//! leaves no written range unreported.

use lichii_common::GpuVAddr;
use lichii_texture::block_linear::{GOB_HEIGHT, Tiling};

use super::EngineContext;
pub use super::maxwell_3d::regs::{UploadRegisters, UploadRegistersDest};

/// Bytes buffered before they are written to GPU memory.
const CHUNK_BYTES: usize = 0x1000;
/// Bytes a block-linear write covers at most: one 16-byte unit of a GOB is contiguous.
const UNIT_BYTES: usize = 16;

#[derive(Clone, Copy)]
enum Target {
    Linear {
        pitch: u64,
    },
    BlockLinear {
        tiling: Tiling,
        x: u32,
        y: u32,
        z: u32,
    },
    /// Out of its surface: the data words are consumed and dropped.
    Discard,
}

/// One transfer in progress.
pub struct Upload {
    target: Target,
    address: GpuVAddr,
    line_length: usize,
    line_count: u32,
    line: u32,
    column: usize,
    remaining: usize,
    chunk: Vec<u8>,
}

impl Default for Upload {
    fn default() -> Self {
        Self {
            target: Target::Discard,
            address: 0,
            line_length: 0,
            line_count: 0,
            line: 0,
            column: 0,
            remaining: 0,
            chunk: Vec::with_capacity(CHUNK_BYTES),
        }
    }
}

impl Upload {
    /// LAUNCH_DMA / EXEC: starts a transfer of `line_length_in * line_count` bytes.
    pub fn exec(&mut self, regs: &UploadRegisters, linear: bool) {
        let dest = regs.dest;
        self.address = dest.address();
        self.line_length = regs.line_length_in as usize;
        self.line_count = regs.line_count;
        self.line = 0;
        self.column = 0;
        self.remaining = self.line_length * regs.line_count as usize;
        self.chunk.clear();
        self.target = if linear {
            Target::Linear {
                pitch: u64::from(dest.pitch),
            }
        } else {
            block_linear_target(regs).unwrap_or_else(|| {
                tracing::warn!(?dest, "block-linear inline upload outside its surface");
                Target::Discard
            })
        };
    }

    /// LOAD_INLINE_DATA: one data word. Its bytes past the end of the transfer are ignored.
    pub fn data(&mut self, ctx: &mut EngineContext<'_>, word: u32) {
        let bytes = word.to_le_bytes();
        let mut bytes = &bytes[..self.remaining.min(4)];
        self.remaining -= bytes.len();
        while !bytes.is_empty() {
            let line_left = self.line_length - self.column - self.chunk.len();
            let take = bytes
                .len()
                .min(line_left)
                .min(CHUNK_BYTES - self.chunk.len());
            self.chunk.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if take == line_left || self.chunk.len() == CHUNK_BYTES {
                self.write_chunk(ctx);
            }
        }
    }

    fn write_chunk(&mut self, ctx: &mut EngineContext<'_>) {
        match self.target {
            Target::Linear { pitch } => {
                let address = self.address + u64::from(self.line) * pitch + self.column as u64;
                if let Err(fault) = ctx.memory.write_block(address, &self.chunk) {
                    tracing::warn!(?fault, "inline upload to unmapped GPU memory");
                }
                ctx.rasterizer.inline_written(
                    ctx.channel,
                    ctx.memory,
                    address,
                    self.chunk.len() as u64,
                );
            }
            Target::BlockLinear { tiling, x, y, z } => {
                let tiling = Tiling {
                    layout: ctx.gob_layout,
                    ..tiling
                };
                let mut done = 0;
                while done < self.chunk.len() {
                    let byte_x = x + (self.column + done) as u32;
                    let len =
                        (UNIT_BYTES - byte_x as usize % UNIT_BYTES).min(self.chunk.len() - done);
                    let offset = tiling.offset(byte_x, y + self.line, z) as u64;
                    if let Err(fault) = ctx
                        .memory
                        .write_block(self.address + offset, &self.chunk[done..done + len])
                    {
                        tracing::warn!(?fault, "block-linear inline upload to unmapped GPU memory");
                    }
                    done += len;
                }
                let (start, size) = touched_block_rows(&tiling, y + self.line, z);
                ctx.rasterizer
                    .inline_written(ctx.channel, ctx.memory, self.address + start, size);
            }
            Target::Discard => {}
        }
        self.column += self.chunk.len();
        self.chunk.clear();
        if self.column == self.line_length {
            self.column = 0;
            self.line += 1;
        }
    }
}

/// The tiled bytes holding row `y` of slice `z`: its row of blocks, or the whole level when
/// blocks span several slices.
fn touched_block_rows(tiling: &Tiling, y: u32, z: u32) -> (u64, u64) {
    if tiling.depth > 1 {
        return (0, tiling.size_bytes() as u64);
    }
    let rows = GOB_HEIGHT << tiling.block_height_log2;
    let band = Tiling {
        height: rows,
        ..*tiling
    };
    (
        tiling.offset(0, y / rows * rows, z) as u64,
        band.size_bytes() as u64,
    )
}

/// The surface a block-linear transfer writes, in bytes (the GOB layout does not depend on the
/// element size), or `None` when the transfer does not fit inside it.
fn block_linear_target(regs: &UploadRegisters) -> Option<Target> {
    let dest = regs.dest;
    let depth = dest.depth.max(1);
    let z = dest.layer.min(depth - 1);
    let fits = dest.x.checked_add(regs.line_length_in)? <= dest.width
        && dest.y.checked_add(regs.line_count)? <= dest.height;
    fits.then(|| Target::BlockLinear {
        tiling: Tiling::new(
            dest.width,
            dest.height,
            depth,
            1,
            dest.block_height(),
            dest.block_depth(),
            Default::default(),
        ),
        x: dest.x,
        y: dest.y,
        z,
    })
}
