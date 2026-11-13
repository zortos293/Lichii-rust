//! The NVDEC engine: rebuilds each compressed frame from the picture parameters the guest hands
//! the engine, decodes it with FFmpeg on a decode thread of the channel and keeps the picture in
//! the shared [`DecodedFrames`] under the luma address of the surface the guest decoded into.
//!
//! Every packet's pts names its slot in a ring of in-flight packets, which holds the surface it
//! decodes into. A decoder that returns a picture late (H.264 reordering, the VP9 builder's
//! one-frame lag) still files it under the surface of the packet that coded it.
//!
//! The channel waits for each decode, but at most [`DECODE_TIMEOUT`]: a decode that has not
//! finished by then leaves the channel's later jobs completing without decoding, as they do
//! without FFmpeg, so a decoder that never returns cannot hold the title's video waits forever.
//! Its pictures are dropped and whatever it still decodes is not kept, since the guest may reuse
//! a surface as soon as the job completes.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::time::Duration;

use lichii_memory::DeviceAddressSpace;
use lichii_video::nvdec::registers::index;
use lichii_video::nvdec::{FrameBuilder, NvdecMemory, NvdecRegisters, VideoCodec};
use lichii_video::{Codec, Decoder, Frame, Latency, VideoError};

use super::frames::{DecodedFrames, SurfaceKeys};

/// Packets whose pictures may still come out of the decoder.
const IN_FLIGHT: usize = 64;
/// How long the channel waits for one decode before it gives up decoding on this channel.
pub(super) const DECODE_TIMEOUT: Duration = Duration::from_secs(1);

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
    /// No FFmpeg, or none with this codec: jobs complete without decoding.
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

/// A thread that runs one decode at a time for the channel, which waits for each at most
/// `timeout`. Register files travel between the two in a pair of boxes, so a job allocates
/// nothing.
pub(super) struct DecodeWorker {
    jobs: SyncSender<Box<NvdecRegisters>>,
    done: Receiver<Box<NvdecRegisters>>,
    /// The box the next job goes out in; `None` once a decode overran or the thread stopped.
    idle: Option<Box<NvdecRegisters>>,
    timeout: Duration,
    channel: u64,
}

impl DecodeWorker {
    pub(super) fn spawn(
        channel: u64,
        timeout: Duration,
        mut decode: impl FnMut(&NvdecRegisters) + Send + 'static,
    ) -> Self {
        let (jobs, received) = mpsc::sync_channel::<Box<NvdecRegisters>>(1);
        let (finished, done) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("host1x-nvdec-decode".into())
            .spawn(move || {
                for regs in received {
                    decode(&regs);
                    if finished.send(regs).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to start an nvdec decode thread");
        Self {
            jobs,
            done,
            idle: Some(Box::default()),
            timeout,
            channel,
        }
    }

    /// Decodes the job `regs` describes, waiting for it at most the worker's timeout. Returns
    /// false, from then on, once a decode has overrun it or the decode thread has stopped.
    pub(super) fn decode(&mut self, regs: &NvdecRegisters) -> bool {
        let Some(mut job) = self.idle.take() else {
            return false;
        };
        (*job).clone_from(regs);
        if self.jobs.send(job).is_err() {
            tracing::warn!(channel = self.channel, "nvdec decode thread stopped");
            return false;
        }
        match self.done.recv_timeout(self.timeout) {
            Ok(job) => {
                self.idle = Some(job);
                true
            }
            Err(RecvTimeoutError::Timeout) => {
                tracing::warn!(
                    channel = self.channel,
                    timeout_ms = self.timeout.as_millis() as u64,
                    "nvdec decode did not finish in time; this channel's jobs now complete without decoding"
                );
                false
            }
            Err(RecvTimeoutError::Disconnected) => {
                tracing::warn!(channel = self.channel, "nvdec decode thread stopped");
                false
            }
        }
    }
}

/// What the decode thread owns: the stream's frame builder and decoder and the packets in flight.
struct Decoding {
    builder: Option<FrameBuilder>,
    backend: Backend,
    memory: Arc<DeviceAddressSpace>,
    frames: Arc<DecodedFrames>,
    channel: u64,
    packets: [Packet; IN_FLIGHT],
    next_pts: i64,
    spare: Option<Arc<Frame>>,
}

impl Decoding {
    fn run(&mut self, regs: &NvdecRegisters) {
        self.select_codec(regs);
        let Some(builder) = &mut self.builder else {
            return;
        };
        let Some(decoder) = self.backend.open(builder.codec()) else {
            return;
        };
        let frame = builder.build(regs, &DeviceReader(&self.memory));
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
            let unshared = Arc::get_mut(&mut picture).expect("spare frames are unshared");
            match decoder.receive(unshared) {
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

    fn select_codec(&mut self, regs: &NvdecRegisters) {
        let Some(codec) = regs.codec_id() else {
            tracing::warn!(raw = regs.read(index::SET_CODEC_ID), "unknown nvdec codec");
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
}

impl Drop for Decoding {
    fn drop(&mut self) {
        self.frames.close(self.channel);
        if let Some(frame) = self.spare.take() {
            self.frames.recycle(frame);
        }
    }
}

pub(super) struct Nvdec {
    regs: Box<NvdecRegisters>,
    /// `None` without decoding: jobs complete without decoding.
    worker: Option<DecodeWorker>,
    frames: Arc<DecodedFrames>,
    channel: u64,
}

impl Nvdec {
    /// With `decode` false, jobs complete without decoding.
    pub(super) fn new(
        memory: Arc<DeviceAddressSpace>,
        frames: Arc<DecodedFrames>,
        decode: bool,
    ) -> Self {
        let channel = NEXT_CHANNEL.fetch_add(1, Ordering::Relaxed);
        let worker = decode.then(|| {
            let mut decoding = Decoding {
                builder: None,
                backend: Backend::Closed,
                memory,
                frames: frames.clone(),
                channel,
                packets: [Packet::default(); IN_FLIGHT],
                next_pts: 0,
                spare: None,
            };
            DecodeWorker::spawn(channel, DECODE_TIMEOUT, move |regs| decoding.run(regs))
        });
        Self {
            regs: Box::default(),
            worker,
            frames,
            channel,
        }
    }

    /// An engine method write: NVDEC's registers are 64 bits wide, set from 32-bit arguments.
    /// Returns true when the write ran a decode job.
    pub(super) fn write(&mut self, method: u32, argument: u32) -> bool {
        let method = method as usize;
        if !self.regs.write(method, u64::from(argument)) || method != index::EXECUTE {
            return false;
        }
        if let Some(worker) = &mut self.worker
            && !worker.decode(&self.regs)
        {
            self.worker = None;
            self.frames.revoke(self.channel);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Instant;

    use super::*;

    #[test]
    fn decodes_run_on_the_worker_and_return_their_registers() {
        let (seen, decoded) = mpsc::channel();
        let mut worker = DecodeWorker::spawn(1, DECODE_TIMEOUT, move |regs| {
            seen.send(regs.read(index::EXECUTE)).unwrap();
        });
        let mut regs = NvdecRegisters::default();
        for job in 1..=3 {
            regs.write(index::EXECUTE, job);
            assert!(worker.decode(&regs));
            assert_eq!(decoded.try_recv(), Ok(job));
        }
    }

    #[test]
    fn a_decode_that_overruns_its_timeout_stops_decoding() {
        let (release, blocked) = mpsc::channel::<()>();
        let timeout = Duration::from_millis(50);
        let mut worker = DecodeWorker::spawn(2, timeout, move |_| {
            let _ = blocked.recv();
        });
        let regs = NvdecRegisters::default();
        let started = Instant::now();
        assert!(!worker.decode(&regs));
        assert!(started.elapsed() >= timeout);
        let started = Instant::now();
        assert!(!worker.decode(&regs), "later jobs do not wait again");
        assert!(started.elapsed() < timeout);
        drop(release);
    }

    #[test]
    fn a_decode_thread_that_panicked_stops_decoding() {
        let mut worker = DecodeWorker::spawn(3, DECODE_TIMEOUT, |_| panic!("decoder bug"));
        let started = Instant::now();
        assert!(!worker.decode(&NvdecRegisters::default()));
        assert!(started.elapsed() < DECODE_TIMEOUT);
    }
}
