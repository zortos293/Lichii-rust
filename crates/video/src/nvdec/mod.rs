//! NVDEC: the register file and the codec state builders that turn the picture parameters a guest
//! hands the engine into the compressed frames a software decoder takes.
//!
//! The builders run once per decoded frame on the NVDEC channel thread. Each keeps its output and
//! scratch buffers across frames, so the steady state allocates nothing.

mod bits;
mod h264;
pub mod picture;
pub mod registers;
mod vp8;
mod vp9;
mod vp9_tables;

#[cfg(test)]
mod tests;

pub use h264::H264Builder;
pub use registers::{NvdecRegisters, VideoCodec};
pub use vp8::Vp8Builder;
pub use vp9::Vp9Builder;

/// Read access to the device (GPU virtual) address space NVDEC's buffers live in.
pub trait NvdecMemory {
    /// Fills `out` with the bytes at device address `address`.
    fn read(&self, address: u64, out: &mut [u8]);
}

/// Size of a decoded picture in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameDimensions {
    pub width: u32,
    pub height: u32,
}

impl FrameDimensions {
    fn new(width: i64, height: i64) -> Option<Self> {
        (width > 0 && height > 0).then_some(Self {
            width: width as u32,
            height: height as u32,
        })
    }
}

/// Device addresses of the surface the decoded picture is written to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceOffsets {
    Progressive {
        luma: u64,
        chroma: u64,
    },
    /// A field picture: each field lives at its own offsets in the surface.
    Interlaced {
        luma_top: u64,
        luma_bottom: u64,
        chroma_top: u64,
        chroma_bottom: u64,
    },
}

impl SurfaceOffsets {
    /// The luma offset the frame is identified by (the top field for interlaced pictures).
    pub fn luma(&self) -> u64 {
        match *self {
            Self::Progressive { luma, .. } => luma,
            Self::Interlaced { luma_top, .. } => luma_top,
        }
    }
}

/// A frame ready for a software decoder, borrowed from the builder that composed it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComposedFrame<'a> {
    /// The complete compressed frame: H.264 Annex B (with SPS and PPS on the first frame and on
    /// every frame numbered 0), a VP8 frame with its frame tag, or a VP9 frame with its
    /// uncompressed and compressed headers.
    pub data: &'a [u8],
    /// Picture size, `None` when the guest left it zero or negative.
    pub dimensions: Option<FrameDimensions>,
    /// The frame is decoded but never shown (VP9 `show_frame == 0`); its output is dropped.
    pub hidden: bool,
    /// Where the guest expects the decoded picture.
    pub offsets: SurfaceOffsets,
}

/// The builder for whichever codec `SET_CODEC_ID` selected.
pub enum FrameBuilder {
    H264(Box<H264Builder>),
    Vp8(Box<Vp8Builder>),
    Vp9(Box<Vp9Builder>),
}

impl FrameBuilder {
    /// A builder for a new stream of `codec`.
    pub fn new(codec: VideoCodec) -> Self {
        match codec {
            VideoCodec::H264 => Self::H264(Box::default()),
            VideoCodec::Vp8 => Self::Vp8(Box::default()),
            VideoCodec::Vp9 => Self::Vp9(Box::default()),
        }
    }

    /// The codec this builder composes.
    pub fn codec(&self) -> VideoCodec {
        match self {
            Self::H264(_) => VideoCodec::H264,
            Self::Vp8(_) => VideoCodec::Vp8,
            Self::Vp9(_) => VideoCodec::Vp9,
        }
    }

    /// Composes the frame the registers describe.
    pub fn build(
        &mut self,
        regs: &NvdecRegisters,
        memory: &(impl NvdecMemory + ?Sized),
    ) -> ComposedFrame<'_> {
        match self {
            Self::H264(builder) => builder.build(regs, memory),
            Self::Vp8(builder) => builder.build(regs, memory),
            Self::Vp9(builder) => builder.build(regs, memory),
        }
    }
}

#[cfg(test)]
pub(crate) mod test_memory {
    use super::NvdecMemory;
    use bytemuck::Pod;

    /// Sparse guest memory: byte blocks at fixed device addresses; unmapped bytes read as 0.
    #[derive(Default)]
    pub(crate) struct TestMemory {
        blocks: Vec<(u64, Vec<u8>)>,
    }

    impl TestMemory {
        pub(crate) fn put(&mut self, address: u64, bytes: &[u8]) {
            self.blocks.retain(|(a, _)| *a != address);
            self.blocks.push((address, bytes.to_vec()));
        }

        pub(crate) fn put_pod(&mut self, address: u64, value: &impl Pod) {
            self.put(address, bytemuck::bytes_of(value));
        }
    }

    impl NvdecMemory for TestMemory {
        fn read(&self, address: u64, out: &mut [u8]) {
            out.fill(0);
            for (base, bytes) in &self.blocks {
                for (i, byte) in out.iter_mut().enumerate() {
                    let addr = address + i as u64;
                    if addr >= *base && addr < base + bytes.len() as u64 {
                        *byte = bytes[(addr - base) as usize];
                    }
                }
            }
        }
    }

    /// `len` deterministic payload bytes, distinct per `seed`.
    pub(crate) fn payload(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
            .collect()
    }
}
