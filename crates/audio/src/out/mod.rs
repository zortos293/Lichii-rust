//! Audio out: the `IAudioOut` sessions of `audout:u`, each playing guest PCM16 buffers on a
//! sink stream.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use lichii_common::HleResult;
use lichii_memory::AddressSpace;

use crate::sink::session::{BufferEvent, Session, SessionManager};
use crate::sink::{SAMPLE_RATE, Sink, StreamType, result};

pub use crate::sink::session::{ParameterInternal as AudioOutParameterInternal, State};

/// The only output device a console reports.
pub const DEVICE_NAME: &str = "DeviceOut";
/// Sessions open at once.
pub const MAX_SESSIONS: usize = 12;

/// What the guest asks for in `OpenAudioOut`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct AudioOutParameter {
    pub sample_rate: i32,
    pub channel_count: u16,
    pub reserved: u16,
}
const _: () = assert!(size_of::<AudioOutParameter>() == 0x8);

/// The guest's `nn::audio::AudioOutBuffer`, as `AppendAudioOutBuffer` passes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct AudioOutBuffer {
    pub next: u64,
    pub samples: u64,
    pub capacity: u64,
    pub size: u64,
    pub offset: u64,
}
const _: () = assert!(size_of::<AudioOutBuffer>() == 0x28);

/// An open `IAudioOut` session.
pub type AudioOut = Session;

/// The audio out manager behind `audout:u`.
pub struct Manager {
    sessions: SessionManager,
}

impl Manager {
    pub fn new(sink: Arc<Sink>) -> Self {
        Self {
            sessions: SessionManager::new(sink, StreamType::Out, MAX_SESSIONS),
        }
    }

    pub fn sink(&self) -> &Arc<Sink> {
        self.sessions.sink()
    }

    /// The device names `ListAudioOuts` reports.
    pub fn device_names(&self) -> &'static [&'static str] {
        &[DEVICE_NAME]
    }

    pub fn free_sessions(&self) -> usize {
        self.sessions.free_sessions()
    }

    /// Opens a session (`OpenAudioOut`) on `device_name` (empty for the default) whose buffers
    /// live in `memory`. `event` is signalled whenever buffers are released. The session closes
    /// when the returned `AudioOut` drops. As on Eden, running out of sessions is reported
    /// before an invalid configuration.
    pub fn open(
        &self,
        device_name: &str,
        parameter: AudioOutParameter,
        memory: Arc<AddressSpace>,
        applet_resource_user_id: u64,
        event: BufferEvent,
    ) -> HleResult<Arc<AudioOut>> {
        let name = if device_name.is_empty() {
            DEVICE_NAME
        } else {
            device_name
        };
        let channels = if parameter.channel_count <= 2 { 2 } else { 6 };
        let session = self
            .sessions
            .open(name, channels, memory, applet_resource_user_id, event)?;
        is_config_valid(device_name, parameter)?;
        Ok(session)
    }

    /// Sets the volume of every open session.
    pub fn set_all_volume(&self, volume: f32) {
        for session in self.sessions.open_sessions() {
            session.set_volume(volume);
        }
    }
}

fn is_config_valid(device_name: &str, parameter: AudioOutParameter) -> HleResult<()> {
    if !device_name.is_empty() && device_name != DEVICE_NAME {
        return Err(result::NOT_FOUND);
    }
    if parameter.sample_rate != SAMPLE_RATE as i32 && parameter.sample_rate > 0 {
        return Err(result::INVALID_SAMPLE_RATE);
    }
    match parameter.channel_count {
        0 | 2 | 6 => Ok(()),
        _ => Err(result::INVALID_CHANNEL_COUNT),
    }
}
