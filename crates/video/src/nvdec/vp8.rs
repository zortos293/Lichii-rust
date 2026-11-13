use super::picture::Vp8PictureInfo;
use super::{ComposedFrame, FrameDimensions, NvdecMemory, NvdecRegisters, SurfaceOffsets};

const CURRENT_SURFACE: usize = 3;

/// Composes VP8 frames: the RFC 6386 frame tag (and key frame start code and size) the guest's
/// driver stripped, followed by the guest's frame data.
pub struct Vp8Builder {
    info: Vp8PictureInfo,
    frame: Vec<u8>,
}

impl Default for Vp8Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Vp8Builder {
    /// A builder for a new stream.
    pub fn new() -> Self {
        Self {
            info: Vp8PictureInfo::default(),
            frame: Vec::new(),
        }
    }

    /// The picture parameters of the last built frame.
    pub fn info(&self) -> &Vp8PictureInfo {
        &self.info
    }

    /// Composes the frame the registers describe.
    pub fn build(
        &mut self,
        regs: &NvdecRegisters,
        memory: &(impl NvdecMemory + ?Sized),
    ) -> ComposedFrame<'_> {
        memory.read(
            regs.picture_info_address(),
            bytemuck::bytes_of_mut(&mut self.info),
        );
        let info = &self.info;
        let key_frame = info.key_frame == 1;
        let header_len = if key_frame { 10 } else { 3 };
        self.frame
            .resize(header_len + info.vld_buffer_size as usize, 0);

        let first_part_size = info.first_part_size;
        let header = &mut self.frame[..header_len];
        header[0] = u8::from(!key_frame) | (info.version & 7) << 1 | 1 << 4;
        header[0] |= ((first_part_size & 7) << 5) as u8;
        header[1] = (first_part_size >> 3) as u8;
        header[2] = (first_part_size >> 11) as u8;
        if key_frame {
            let [width_lo, width_hi] = info.frame_width.to_le_bytes();
            let [height_lo, height_hi] = info.frame_height.to_le_bytes();
            header[3..].copy_from_slice(&[
                0x9D,
                0x01,
                0x2A,
                width_lo,
                width_hi & 0x3F,
                height_lo,
                height_hi & 0x3F,
            ]);
        }
        memory.read(
            regs.frame_bitstream_address(),
            &mut self.frame[header_len..],
        );

        ComposedFrame {
            data: &self.frame,
            dimensions: FrameDimensions::new(
                i64::from(info.frame_width),
                i64::from(info.frame_height),
            ),
            hidden: false,
            offsets: SurfaceOffsets::Progressive {
                luma: regs.surface_luma_address(CURRENT_SURFACE),
                chroma: regs.surface_chroma_address(CURRENT_SURFACE),
            },
        }
    }
}
