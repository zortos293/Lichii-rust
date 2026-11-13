//! The NVDEC engine: rebuilds each compressed frame from the picture parameters the guest hands
//! the engine, decodes it with FFmpeg on the channel thread and keeps the picture in the shared
//! [`DecodedFrames`] under the luma address of the surface the guest decoded into.
//!
//! Every packet's pts names its slot in a ring of in-flight packets, which holds the surface it
//! decodes into. A decoder that returns a picture late (H.264 reordering, the VP9 builder's
//! one-frame lag) still files it under the surface of the packet that coded it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lichii_memory::DeviceAddressSpace;
use lichii_video::nvdec::registers::index;
use lichii_video::nvdec::{FrameBuilder, NvdecMemory, NvdecRegisters, VideoCodec};
use lichii_video::{Codec, Decoder, Frame, Latency, VideoError};

use super::channel::DECODE_TIME;
use super::frames::{DecodedFrames, SurfaceKeys};

/// Packets whose pictures may still come out of the decoder.
const IN_FLIGHT: usize = 64;

static NEXT_CHANNEL: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Default)]
struct Packet {
    pts: i64,
    /// `None` for a packet whose picture is never shown.
    surface: Option<SurfaceKeys>,
}

enum Backend {
    Closed,
    Open(Decoder),
    /// No FFmpeg, or none with this codec: jobs only take [`DECODE_TIME`].
    Unavailable,
}

impl Backend {
    fn open(&mut self, codec: VideoCodec) -> Option<&mut Decoder> {
        if let Self::Closed = self {
            let codec = match codec {
                VideoCodec::H264 => Codec::H264,
                VideoCodec::Vp8 => Codec::Vp8,
                VideoCodec::Vp9 => Codec::Vp9,
            };
            *self = match Decoder::new(codec, Latency::Immediate) {
                Ok(decoder) => Self::Open(decoder),
                Err(error) => {
                    if error != VideoError::Unavailable {
                        tracing::warn!(%error, "nvdec has no decoder");
                    }
                    Self::Unavailable
                }
            };
        }
        match self {
            Self::Open(decoder) => Some(decoder),
            _ => None,
        }
    }
}

struct DeviceReader<'a>(&'a DeviceAddressSpace);

impl NvdecMemory for DeviceReader<'_> {
    fn read(&self, address: u64, out: &mut [u8]) {
        if self.0.read_bytes(address, out).is_err() {
            tracing::debug!(address, len = out.len(), "nvdec read from unmapped memory");
            out.fill(0);
        }
    }
}

pub(super) struct Nvdec {
    regs: Box<NvdecRegisters>,
    builder: Option<FrameBuilder>,
    backend: Backend,
    memory: Arc<DeviceAddressSpace>,
    frames: Arc<DecodedFrames>,
    channel: u64,
    packets: [Packet; IN_FLIGHT],
    next_pts: i64,
    spare: Option<Frame>,
}

impl Nvdec {
    pub(super) fn new(memory: Arc<DeviceAddressSpace>, frames: Arc<DecodedFrames>) -> Self {
        Self {
            regs: Box::default(),
            builder: None,
            backend: Backend::Closed,
            memory,
            frames,
            channel: NEXT_CHANNEL.fetch_add(1, Ordering::Relaxed),
            packets: [Packet::default(); IN_FLIGHT],
            next_pts: 0,
            spare: None,
        }
    }

    /// An engine method write: NVDEC's registers are 64 bits wide, set from 32-bit arguments.
    pub(super) fn write(&mut self, method: u32, argument: u32) {
        let method = method as usize;
        if !self.regs.write(method, u64::from(argument)) {
            return;
        }
        match method {
            index::SET_CODEC_ID => self.select_codec(),
            index::EXECUTE => self.execute(),
            _ => {}
        }
    }

    fn select_codec(&mut self) {
        let Some(codec) = self.regs.codec_id() else {
            tracing::warn!(
                raw = self.regs.read(index::SET_CODEC_ID),
                "unknown nvdec codec"
            );
            return;
        };
        if self.builder.as_ref().is_some_and(|b| b.codec() == codec) {
            return;
        }
        tracing::info!(?codec, channel = self.channel, "nvdec stream opened");
        self.builder = Some(FrameBuilder::new(codec));
        if let Backend::Open(_) = self.backend {
            self.backend = Backend::Closed;
        }
    }

    fn execute(&mut self) {
        let Some(builder) = &mut self.builder else {
            std::thread::sleep(DECODE_TIME);
            return;
        };
        let Some(decoder) = self.backend.open(builder.codec()) else {
            std::thread::sleep(DECODE_TIME);
            return;
        };
        let frame = builder.build(&self.regs, &DeviceReader(&self.memory));
        let pts = self.next_pts;
        self.next_pts += 1;
        self.packets[pts as usize % IN_FLIGHT] = Packet {
            pts,
            surface: (!frame.hidden).then(|| frame.offsets.into()),
        };
        if let Err(error) = decoder.send(frame.data, pts) {
            tracing::debug!(%error, "nvdec packet rejected");
        }
        let mut spare = self.spare.take();
        while let Some(mut picture) = spare.take().or_else(|| self.frames.spare_frame()) {
            match decoder.receive(&mut picture) {
                Ok(true) => {}
                Ok(false) => {
                    spare = Some(picture);
                    break;
                }
                Err(error) => {
                    tracing::debug!(%error, "nvdec frame failed to decode");
                    spare = Some(picture);
                    break;
                }
            }
            let surface = picture.picture().and_then(|decoded| {
                let packet = self.packets[decoded.pts.rem_euclid(IN_FLIGHT as i64) as usize];
                packet.surface.filter(|_| packet.pts == decoded.pts)
            });
            match surface {
                Some(surface) => self.frames.insert(self.channel, surface, picture),
                None => spare = Some(picture),
            }
        }
        self.spare = spare;
    }
}

impl Drop for Nvdec {
    fn drop(&mut self) {
        self.frames.close(self.channel);
        if let Some(frame) = self.spare.take() {
            self.frames.recycle(frame);
        }
    }
}
