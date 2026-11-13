//! Audio services that do not need the renderer: `audout:u`, `audin:u`, `audctl` and `hwopus`.
//!
//! They share one [`Audio`]: the host sink `core::System` passes in [`ServiceContext::audio`]
//! and the audio out and audio in managers that play on it.

mod audctl;
mod audin;
mod audout;
mod hwopus;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use lichii_audio::sink::Sink;
use lichii_audio::{r#in, out};
use lichii_common::HleResult;
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;

use crate::ServiceContext;

/// Size of `nn::audio::AudioDeviceName`.
const DEVICE_NAME_SIZE: usize = 0x100;

/// The audio state every audio service shares.
pub(crate) struct Audio {
    pub sink: Arc<Sink>,
    pub out: out::Manager,
    pub input: r#in::Manager,
}

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    start(context).map(drop)
}

/// Registers every audio service on the context's sink.
fn start(context: &Arc<ServiceContext>) -> HleResult<Arc<Audio>> {
    let sink = context.audio.clone();
    let audio = Arc::new(Audio {
        out: out::Manager::new(sink.clone()),
        input: r#in::Manager::new(sink.clone()),
        sink,
    });
    audout::register(context, &audio)?;
    audin::register(context, &audio)?;
    audctl::register(context, &audio)?;
    hwopus::register(context)?;
    Ok(audio)
}

/// An `AudioDeviceName`: the name, NUL-padded to 0x100 bytes.
fn device_name(name: &str) -> [u8; DEVICE_NAME_SIZE] {
    let mut out = [0; DEVICE_NAME_SIZE];
    let len = name.len().min(DEVICE_NAME_SIZE - 1);
    out[..len].copy_from_slice(&name.as_bytes()[..len]);
    out
}

/// The `AudioDeviceName` in input buffer 0, or `None` when the buffer cannot hold one.
fn read_device_name(ctx: &HleRequestContext) -> Option<String> {
    let buffer = ctx.readable_buffer(0)?;
    if (buffer.size as usize) < DEVICE_NAME_SIZE {
        return None;
    }
    let mut name = [0; DEVICE_NAME_SIZE];
    ctx.memory().read_bytes(buffer.address, &mut name).ok()?;
    let len = name
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(DEVICE_NAME_SIZE);
    Some(String::from_utf8_lossy(&name[..len]).into_owned())
}

/// Writes `names` as `AudioDeviceName`s to output buffer 0, as many as fit, and returns how
/// many it wrote.
fn write_device_names(ctx: &HleRequestContext, names: &[&str]) -> u32 {
    let capacity = ctx
        .writable_buffer(0)
        .map_or(0, |buffer| buffer.size as usize / DEVICE_NAME_SIZE);
    let count = names.len().min(capacity);
    let data: Vec<u8> = names[..count]
        .iter()
        .flat_map(|name| device_name(name))
        .collect();
    if count > 0 && ctx.write_buffer(0, &data).is_err() {
        return 0;
    }
    count as u32
}

/// Whether output buffer 0 can hold an `AudioDeviceName`.
fn has_device_name_output(ctx: &HleRequestContext) -> bool {
    ctx.writable_buffer(0)
        .is_some_and(|buffer| buffer.size as usize >= DEVICE_NAME_SIZE)
}

/// A buffer event the audio sessions signal.
fn buffer_event() -> (Arc<Event>, lichii_audio::sink::session::BufferEvent) {
    let event = Arc::new(Event::new());
    let signal = event.clone();
    (event, Arc::new(move || signal.signal()))
}
