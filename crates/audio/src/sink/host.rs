//! The cpal sink: plays streams on a host output device.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{BufferSize, HostId, SampleFormat, StreamConfig};

use super::SAMPLE_RATE;
use super::stream::StreamState;

pub(super) struct CpalOutput {
    device: cpal::Device,
    name: String,
    channels: usize,
    format: SampleFormat,
}

impl CpalOutput {
    /// An output on cpal host `host` and device `device`. A chosen host is the only one tried;
    /// otherwise every available host is, the default first.
    pub(super) fn open(host: Option<&str>, device: Option<&str>) -> Option<Self> {
        let available = cpal::available_hosts();
        let requested = host.and_then(|name| {
            let id = available.iter().copied().find(|id| id.name() == name);
            if id.is_none() {
                tracing::warn!(
                    host = name,
                    "audio host not available, choosing automatically"
                );
            }
            id
        });
        let default = cpal::default_host();
        let default_id = default.id();
        let mut default = Some(default);
        let hosts: Vec<cpal::Host> = candidate_hosts(requested, default_id, &available)
            .into_iter()
            .filter_map(|id| match default.take_if(|host| host.id() == id) {
                Some(host) => Some(host),
                None => cpal::host_from_id(id)
                    .map_err(|err| tracing::warn!(host = id.name(), "opening audio host: {err}"))
                    .ok(),
            })
            .collect();
        if let Some(name) = device {
            for host in &hosts {
                let found = host.output_devices().ok().and_then(|mut devices| {
                    devices.find(|device| device_name(device).as_deref() == Some(name))
                });
                if let Some(output) = found.and_then(|found| Self::from_device(host.id(), found)) {
                    return Some(output);
                }
            }
            tracing::warn!(
                device = name,
                "audio output device not found, using the default"
            );
        }
        for host in &hosts {
            match host.default_output_device() {
                Some(device) => {
                    if let Some(output) = Self::from_device(host.id(), device) {
                        return Some(output);
                    }
                }
                None => tracing::warn!(host = host.id().name(), "no default audio output device"),
            }
        }
        tracing::warn!("no audio output device");
        None
    }

    /// `device`, if it can play 48 kHz `i16` or `f32` in stereo or 5.1.
    fn from_device(host: HostId, device: cpal::Device) -> Option<Self> {
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
        tracing::info!(host = host.name(), device = %name, channels, ?format, "audio: cpal sink");
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

fn candidate_hosts(
    requested: Option<HostId>,
    default: HostId,
    available: &[HostId],
) -> Vec<HostId> {
    match requested {
        Some(id) => vec![id],
        None => std::iter::once(default)
            .chain(available.iter().copied().filter(|&id| id != default))
            .collect(),
    }
}

pub(super) fn host_names() -> Vec<&'static str> {
    let available = cpal::available_hosts();
    let default = cpal::default_host().id();
    candidate_hosts(None, default, &available)
        .into_iter()
        .map(|id| id.name())
        .collect()
}

pub(super) fn output_device_names(host: Option<&str>) -> Vec<String> {
    let host = host
        .and_then(|name| {
            cpal::available_hosts()
                .into_iter()
                .find(|id| id.name() == name)
        })
        .and_then(|id| cpal::host_from_id(id).ok())
        .unwrap_or_else(cpal::default_host);
    host.output_devices()
        .map(|devices| devices.filter_map(|device| device_name(&device)).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_tries_the_default_host_then_every_other_once() {
        let available = cpal::available_hosts();
        let Some(&default) = available.last() else {
            return;
        };
        let hosts = candidate_hosts(None, default, &available);
        assert_eq!(hosts[0], default);
        assert_eq!(hosts.len(), available.len());
        assert!(available.iter().all(|id| hosts.contains(id)));
    }

    #[test]
    fn a_requested_host_is_the_only_one_tried() {
        for &id in &cpal::available_hosts() {
            assert_eq!(
                candidate_hosts(
                    Some(id),
                    cpal::default_host().id(),
                    &cpal::available_hosts()
                ),
                [id]
            );
        }
    }
}
