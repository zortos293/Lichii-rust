//! Host audio output.
//!
//! A [`Sink`] owns the host streams. Each [`SinkStream`] pulls interleaved `i16` samples at
//! 48 kHz (2 or 6 channels) from a ring the emulator thread fills with
//! [`SinkStream::append_buffer`], and reports how far it has played so `IAudioOut` can give
//! buffers back. The cpal sink plays through the host device named in
//! [`AudioSettings`]; the null sink consumes streams in real time without a device, for headless
//! runs and CI, and also drives every capture stream (they hear silence).

pub mod device;
mod host;
mod null;
pub mod result;
pub mod session;
mod stream;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use lichii_common::settings::{AudioOutput, AudioSettings};

pub use stream::{SinkStream, StreamType};

/// The sample rate of every stream.
pub const SAMPLE_RATE: u32 = 48_000;
/// Frames in one 5 ms block, the unit the audio systems work in.
pub const TARGET_SAMPLE_COUNT: u32 = 240;
/// The most channels a stream carries.
pub const MAX_CHANNELS: usize = 6;

/// Which backend a [`Sink`] plays through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinkKind {
    Cpal,
    Null,
}

/// State every stream of a sink reads.
pub(crate) struct SinkShared {
    host_volume: AtomicU32,
    device_volume: AtomicU32,
    paused: AtomicBool,
}

impl SinkShared {
    pub(crate) fn new(host_volume: f32) -> Self {
        Self {
            host_volume: AtomicU32::new(host_volume.to_bits()),
            device_volume: AtomicU32::new(1.0f32.to_bits()),
            paused: AtomicBool::new(false),
        }
    }
}

/// Host audio output: creates the streams the audio systems play through.
pub struct Sink {
    shared: Arc<SinkShared>,
    pacer: null::Pacer,
    host: Option<host::CpalOutput>,
}

impl Sink {
    /// The sink the settings ask for: cpal on the configured output device (falling back to the
    /// default device), or the null sink when the settings say so or no device can play 48 kHz.
    pub fn new(settings: &AudioSettings) -> Self {
        let host = match settings.output {
            AudioOutput::Null => None,
            AudioOutput::Auto | AudioOutput::Cpal => {
                host::CpalOutput::open(settings.output_device.as_deref())
            }
        };
        if host.is_none() {
            tracing::info!("audio: null sink");
        }
        Self {
            shared: Arc::new(SinkShared::new(host_volume(settings))),
            pacer: null::Pacer::start(),
            host,
        }
    }

    /// A null sink, whatever the settings say.
    pub fn null() -> Self {
        Self::new(&AudioSettings {
            output: AudioOutput::Null,
            ..AudioSettings::default()
        })
    }

    pub fn kind(&self) -> SinkKind {
        match self.host {
            Some(_) => SinkKind::Cpal,
            None => SinkKind::Null,
        }
    }

    /// Channels of the output device: 6 when it can play 5.1 at 48 kHz, else 2.
    pub fn device_channels(&self) -> usize {
        self.host.as_ref().map_or(2, host::CpalOutput::channels)
    }

    /// Opens a stream for `system_channels`-channel samples. Output streams play on the host
    /// device when there is one (a stream the device refuses falls back to the null clock);
    /// capture streams always hear silence.
    pub fn open_stream(&self, system_channels: usize, name: &str, ty: StreamType) -> SinkStream {
        let device_channels = self.device_channels();
        let state = Arc::new(stream::StreamState::new(
            ty,
            name,
            system_channels.clamp(1, MAX_CHANNELS),
            device_channels,
            self.shared.clone(),
        ));
        let host = match (&self.host, ty) {
            (Some(host), StreamType::Out | StreamType::Render) => host.build_stream(&state),
            _ => None,
        };
        if host.is_none() {
            self.pacer.add(&state);
        }
        SinkStream { state, host }
    }

    /// The device volume (`IAudioDevice` and `audctl` target volume), 0.0 to 1.0.
    pub fn device_volume(&self) -> f32 {
        load_f32(&self.shared.device_volume)
    }

    pub fn set_device_volume(&self, volume: f32) {
        store_f32(&self.shared.device_volume, volume);
    }

    /// The host-side volume from [`AudioSettings`]: `percent` is 0 to 200, and `muted` silences
    /// every stream. Takes effect for samples appended from now on.
    pub fn set_host_volume(&self, percent: u8, muted: bool) {
        store_f32(
            &self.shared.host_volume,
            host_volume(&AudioSettings {
                volume: percent,
                muted,
                ..AudioSettings::default()
            }),
        );
    }

    pub fn host_volume(&self) -> f32 {
        load_f32(&self.shared.host_volume)
    }

    /// Pauses every stream while emulation is paused: callbacks output silence and play
    /// positions stop advancing.
    pub fn set_paused(&self, paused: bool) {
        self.shared.paused.store(paused, Ordering::Release);
    }

    pub fn is_paused(&self) -> bool {
        self.shared.paused.load(Ordering::Acquire)
    }
}

/// The names of the host's output devices, for the settings UI.
pub fn output_device_names() -> Vec<String> {
    host::output_device_names()
}

fn host_volume(settings: &AudioSettings) -> f32 {
    if settings.muted {
        0.0
    } else {
        f32::from(settings.volume.min(200)) / 100.0
    }
}

pub(crate) fn load_f32(value: &AtomicU32) -> f32 {
    f32::from_bits(value.load(Ordering::Relaxed))
}

pub(crate) fn store_f32(value: &AtomicU32, new: f32) {
    value.store(new.to_bits(), Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    #[test]
    fn null_settings_give_a_null_sink() {
        let sink = Sink::null();
        assert_eq!(sink.kind(), SinkKind::Null);
        assert_eq!(sink.device_channels(), 2);
    }

    #[test]
    fn default_settings_open_a_sink_whose_streams_play() {
        let sink = Sink::new(&AudioSettings::default());
        let stream = sink.open_stream(6, "test", StreamType::Out);
        assert_eq!(stream.device_channels(), sink.device_channels());
        stream.append_buffer(240, &[0; 240 * 6]);
        stream.start();
        stream.stop();
    }

    #[test]
    fn host_volume_follows_the_settings() {
        let sink = Sink::new(&AudioSettings {
            output: AudioOutput::Null,
            volume: 150,
            ..AudioSettings::default()
        });
        assert_eq!(sink.host_volume(), 1.5);
        sink.set_host_volume(50, false);
        assert_eq!(sink.host_volume(), 0.5);
        sink.set_host_volume(50, true);
        assert_eq!(sink.host_volume(), 0.0);
    }

    #[test]
    fn the_null_sink_consumes_in_real_time() {
        let sink = Sink::null();
        let stream = sink.open_stream(2, "test", StreamType::Out);
        assert!(!stream.is_host_stream());
        let frames = u64::from(SAMPLE_RATE) / 10;
        let end = stream.append_buffer(frames, &vec![0; frames as usize * 2]);
        stream.start();
        let started = Instant::now();
        while stream.queue_size() != 0 || stream.has_pending_audio() {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "null sink stalled"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(90),
            "played 100 ms in {elapsed:?}"
        );
        assert!(stream.expected_played_sample_count() >= end);
    }

    #[test]
    fn a_paused_sink_holds_its_streams() {
        let sink = Sink::null();
        let stream = sink.open_stream(2, "test", StreamType::Out);
        stream.append_buffer(240, &[0; 480]);
        sink.set_paused(true);
        stream.start();
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(stream.queue_size(), 1);
        sink.set_paused(false);
        let started = Instant::now();
        while stream.queue_size() != 0 {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "null sink stalled"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}
