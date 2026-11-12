//! Shared walk over 4x4 block images: tightly packed blocks on one side, tightly packed texels on
//! the other, with partial blocks at the right and bottom edges.

/// A buffer too small for the image extent it was given.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ImageSizeError {
    #[error("input holds {actual} bytes, the image needs {required}")]
    Input { required: usize, actual: usize },
    #[error("output holds {actual} bytes, the image needs {required}")]
    Output { required: usize, actual: usize },
}

pub(crate) const BLOCK_DIM: usize = 4;

pub(crate) struct Extent {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub blocks_x: usize,
    pub blocks_y: usize,
}

impl Extent {
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        let (width, height) = (width as usize, height as usize);
        Self {
            width,
            height,
            depth: depth as usize,
            blocks_x: width.div_ceil(BLOCK_DIM),
            blocks_y: height.div_ceil(BLOCK_DIM),
        }
    }

    pub fn block_bytes(&self, bytes_per_block: usize) -> usize {
        self.blocks_x * self.blocks_y * self.depth * bytes_per_block
    }

    pub fn texel_bytes(&self, bytes_per_texel: usize) -> usize {
        self.width * self.height * self.depth * bytes_per_texel
    }
}

pub(crate) fn check_len(
    input: usize,
    input_required: usize,
    output: usize,
    output_required: usize,
) -> Result<(), ImageSizeError> {
    if input < input_required {
        return Err(ImageSizeError::Input {
            required: input_required,
            actual: input,
        });
    }
    if output < output_required {
        return Err(ImageSizeError::Output {
            required: output_required,
            actual: output,
        });
    }
    Ok(())
}

/// Decodes every block of an image with `decode`, which returns the block's 16 texels row-major,
/// each `T` bytes wide, and stores the texels that fall inside the image.
pub(crate) fn decode_image<const B: usize, const T: usize>(
    blocks: &[u8],
    width: u32,
    height: u32,
    depth: u32,
    out: &mut [u8],
    decode: impl Fn(&[u8; B]) -> [[u8; T]; 16],
) -> Result<(), ImageSizeError> {
    let extent = Extent::new(width, height, depth);
    let in_len = extent.block_bytes(B);
    let out_len = extent.texel_bytes(T);
    check_len(blocks.len(), in_len, out.len(), out_len)?;
    if out_len == 0 {
        return Ok(());
    }
    let row_pitch = extent.width * T;
    let (blocks, _) = blocks[..in_len].as_chunks::<B>();
    for (slice_blocks, slice_out) in blocks
        .chunks_exact(extent.blocks_x * extent.blocks_y)
        .zip(out[..out_len].chunks_exact_mut(row_pitch * extent.height))
    {
        for (row_blocks, rows_out) in slice_blocks
            .chunks_exact(extent.blocks_x)
            .zip(slice_out.chunks_mut(row_pitch * BLOCK_DIM))
        {
            for (bx, block) in row_blocks.iter().enumerate() {
                let texels = decode(block);
                let (texel_rows, _) = texels.as_chunks::<BLOCK_DIM>();
                let x = bx * BLOCK_DIM * T;
                for (src, out_row) in texel_rows.iter().zip(rows_out.chunks_exact_mut(row_pitch)) {
                    let src = src.as_flattened();
                    match out_row.get_mut(x..x + BLOCK_DIM * T) {
                        Some(dst) => dst.copy_from_slice(src),
                        None => out_row[x..].copy_from_slice(&src[..row_pitch - x]),
                    }
                }
            }
        }
    }
    Ok(())
}

macro_rules! image_decoder {
    ($(#[$doc:meta])* $name:ident, $block_fn:expr) => {
        $(#[$doc])*
        ///
        /// `blocks` holds `ceil(width / 4) * ceil(height / 4) * depth` blocks; `out` receives
        /// `width * height * depth` texels.
        pub fn $name(
            blocks: &[u8],
            width: u32,
            height: u32,
            depth: u32,
            out: &mut [u8],
        ) -> Result<(), $crate::block::ImageSizeError> {
            $crate::block::decode_image(blocks, width, height, depth, out, $block_fn)
        }
    };
}

pub(crate) use image_decoder;
