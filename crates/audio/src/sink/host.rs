//! The cpal sink: plays streams on a host output device.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{BufferSize, SampleFormat, StreamConfig};

use super::SAMPLE_RATE;
use super::stream::StreamState;

pub(super) struct CpalOutput {
    device: cpal::Device,
    name: String,
    channels: usize,
    format: SampleFormat,
}

impl CpalOutput {
    /// The device called `name` (or the default one when `name` is `None` or not found), if it
    /// can play 48 kHz `i16` or `f32` in stereo or 5.1.
    pub(super) fn open(name: Option<&str>) -> Option<Self> {
        let host = cpal::default_host();
        let named = name.and_then(|name| {
            let found = host
                .output_devices()
                .ok()?
                .find(|device| device_name(device).as_deref() == Some(name));
            if found.is_none() {
                tracing::warn!(
                    device = name,
                    "audio output device not found, using the default"
                );
            }
            found
        });
        let Some(device) = named.or_else(|| host.default_output_device()) else {
            tracing::warn!("no audio output device");
            return None;
        };
        let name = device_name(&device).unwrap_or_else(|| "unknown".into());
        let configs: Vec<_> = match device.supported_output_configs() {
            Ok(configs) => configs
                .filter(|config| {
                    (config.min_sample_rate()..=config.max_sample_rate()).contains(&SAMPLE_RATE)
                        && matches!(
                            config.sample_format(),
                            SampleFormat::I16 | SampleFormat::F32
                        )
                })
                .collect(),
            Err(err) => {
                tracing::warn!(device = %name, "querying audio output configs: {err}");
                return None;
            }
        };
        let pick = |channels: u16| {
            let usable = || configs.iter().filter(move |c| c.channels() == channels);
            usable()
                .find(|c| c.sample_format() == SampleFormat::I16)
                .or_else(|| usable().next())
                .map(|c| (usize::from(channels), c.sample_format()))
        };
        let Some((channels, format)) = pick(6).or_else(|| pick(2)) else {
            tracing::warn!(device = %name, "audio output device cannot play 48 kHz stereo");
            return None;
        };
        tracing::info!(device = %name, channels, ?format, "audio: cpal sink");
        Some(Self {
            device,
            name,
            channels,
            format,
        })
    }

    pub(super) fn channels(&self) -> usize {
        self.channels
    }

    /// A paused host stream that pulls from `state`, or `None` when the device refuses it.
    pub(super) fn build_stream(&self, state: &Arc<StreamState>) -> Option<cpal::Stream> {
        let config = StreamConfig {
            channels: self.channels as u16,
            sample_rate: SAMPLE_RATE,
            buffer_size: BufferSize::Default,
        };
        let reported = Arc::new(AtomicBool::new(false));
        let device = self.name.clone();
        let on_error = move |err: cpal::Error| {
            if !reported.swap(true, Ordering::Relaxed) {
                tracing::warn!(%device, "audio stream error: {err}");
            }
        };
        let state = state.clone();
        let built = match self.format {
            SampleFormat::I16 => self.device.build_output_stream::<i16, _, _>(
                config,
                move |data, _| state.process_output(data),
                on_error,
                None,
            ),
            _ => {
                let mut scratch = Vec::new();
                self.device.build_output_stream::<f32, _, _>(
                    config,
                    move |data, _| {
                        if scratch.len() < data.len() {
                            scratch.resize(data.len(), 0i16);
                        }
                        let samples = &mut scratch[..data.len()];
                        state.process_output(samples);
                        for (out, &sample) in data.iter_mut().zip(samples.iter()) {
                            *out = f32::from(sample) / 32768.0;
                        }
                    },
                    on_error,
                    None,
                )
            }
        };
        built
            .map_err(|err| {
                tracing::warn!(device = %self.name, "opening an audio stream, using the null clock: {err}");
            })
            .ok()
    }
}

fn device_name(device: &cpal::Device) -> Option<String> {
    device
        .description()
        .ok()
        .map(|description| description.name().to_string())
}

pub(super) fn output_device_names() -> Vec<String> {
    cpal::default_host()
        .output_devices()
        .map(|devices| devices.filter_map(|device| device_name(&device)).collect())
        .unwrap_or_default()
}
