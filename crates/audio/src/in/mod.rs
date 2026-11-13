//! Audio in: the `IAudioIn` sessions of `audin:u`. There is no host capture: every session
//! records silence, delivered in real time.

use std::sync::Arc;

use lichii_common::HleResult;
use lichii_memory::AddressSpace;

use crate::sink::session::{BufferEvent, Session, SessionManager};
use crate::sink::{SAMPLE_RATE, Sink, StreamType, result};

pub use crate::out::{AudioOutBuffer as AudioInBuffer, AudioOutParameter as AudioInParameter};
pub use crate::sink::session::{ParameterInternal as AudioInParameterInternal, State};

/// The console's built-in microphone input.
pub const DEFAULT_DEVICE_NAME: &str = "BuiltInHeadset";
/// A USB audio class microphone, the device `ListAudioIns` reports.
pub const UAC_DEVICE_NAME: &str = "Uac";
/// Sessions open at once.
pub const MAX_SESSIONS: usize = 4;

/// An open `IAudioIn` session.
pub type AudioIn = Session;

/// The audio in manager behind `audin:u`.
pub struct Manager {
    sessions: SessionManager,
}

impl Manager {
    pub fn new(sink: Arc<Sink>) -> Self {
        Self {
            sessions: SessionManager::new(sink, StreamType::In, MAX_SESSIONS),
        }
    }

    /// The device names `ListAudioIns` reports: the silent capture device, as a USB microphone.
    pub fn device_names(&self) -> &'static [&'static str] {
        &[UAC_DEVICE_NAME]
    }

    pub fn free_sessions(&self) -> usize {
        self.sessions.free_sessions()
    }

    /// Opens a session (`OpenAudioIn`) on `device_name` (empty for the default) whose buffers
    /// live in `memory`. `event` is signalled whenever buffers are released. The configuration
    /// is checked before a session is taken.
    pub fn open(
        &self,
        device_name: &str,
        parameter: AudioInParameter,
        memory: Arc<AddressSpace>,
        applet_resource_user_id: u64,
        event: BufferEvent,
    ) -> HleResult<Arc<AudioIn>> {
        is_config_valid(device_name, parameter)?;
        let name = if device_name.is_empty() {
            DEFAULT_DEVICE_NAME
        } else {
            device_name
        };
        let channels = if parameter.channel_count <= 2 { 2 } else { 6 };
        self.sessions
            .open(name, channels, memory, applet_resource_user_id, event)
    }
}

/// Whether a session opened on `name` records from a USB microphone.
pub fn is_uac(name: &str) -> bool {
    name == UAC_DEVICE_NAME
}

fn is_config_valid(device_name: &str, parameter: AudioInParameter) -> HleResult<()> {
    if !device_name.is_empty()
        && device_name != DEFAULT_DEVICE_NAME
        && device_name != UAC_DEVICE_NAME
    {
        return Err(result::NOT_FOUND);
    }
    if parameter.sample_rate != SAMPLE_RATE as i32 && parameter.sample_rate > 0 {
        return Err(result::INVALID_SAMPLE_RATE);
    }
    Ok(())
}
