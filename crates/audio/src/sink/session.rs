//! The session machinery audio out and audio in share: a manager of numbered sessions, each a
//! ring of guest buffers playing on one sink stream, and the 5 ms tick that hands played buffers
//! back.

use std::sync::{Arc, Weak};
use std::time::Duration;

use lichii_common::HleResult;
use lichii_memory::AddressSpace;
use parking_lot::Mutex;

use super::device::{AudioBuffer, AudioBuffers, DeviceSession};
use super::{SAMPLE_RATE, Sink, StreamType, result};

/// How often a manager releases played buffers and registers waiting ones.
const TICK: Duration = Duration::from_millis(5);

/// `SampleFormat::PcmInt16`, the only format audio out and audio in carry.
pub const SAMPLE_FORMAT_PCM_INT16: u32 = 2;

/// Signals a session's buffer event; the services wrap a kernel event in it.
pub type BufferEvent = Arc<dyn Fn() + Send + Sync>;

/// A session's state (`GetAudioOutState`, `GetAudioInState`).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Started = 0,
    Stopped = 1,
}

/// What `OpenAudioOut` and `OpenAudioIn` answer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParameterInternal {
    pub sample_rate: u32,
    pub channel_count: u32,
    pub sample_format: u32,
    pub state: u32,
}
const _: () = assert!(size_of::<ParameterInternal>() == 0x10);

/// The free session ids, handed out in Eden's order (a ring of ids).
struct SessionIds {
    ids: Vec<usize>,
    next: usize,
    free: usize,
    available: usize,
}

impl SessionIds {
    fn new(count: usize) -> Self {
        Self {
            ids: (0..count).collect(),
            next: 0,
            free: 0,
            available: count,
        }
    }

    fn acquire(&mut self) -> Option<usize> {
        if self.available == 0 {
            return None;
        }
        let id = self.ids[self.next];
        self.next = (self.next + 1) % self.ids.len();
        self.available -= 1;
        Some(id)
    }

    fn release(&mut self, id: usize) {
        self.ids[self.free] = id;
        self.free = (self.free + 1) % self.ids.len();
        self.available += 1;
    }
}

struct Sessions {
    ids: SessionIds,
    open: Vec<Weak<Session>>,
    ticking: bool,
}

struct ManagerShared {
    ty: StreamType,
    sink: Arc<Sink>,
    sessions: Mutex<Sessions>,
}

impl ManagerShared {
    fn tick(&self, open: &mut [Option<Arc<Session>>]) {
        {
            let sessions = self.sessions.lock();
            for (slot, session) in open.iter_mut().zip(&sessions.open) {
                *slot = session.upgrade();
            }
        }
        for slot in open.iter_mut() {
            if let Some(session) = slot.take() {
                session.release_and_register();
            }
        }
    }
}

/// Numbered sessions of one kind (audio out or audio in) playing on one sink.
pub struct SessionManager {
    shared: Arc<ManagerShared>,
}

impl SessionManager {
    /// A manager of up to `max_sessions` sessions of type `ty`.
    pub fn new(sink: Arc<Sink>, ty: StreamType, max_sessions: usize) -> Self {
        Self {
            shared: Arc::new(ManagerShared {
                ty,
                sink,
                sessions: Mutex::new(Sessions {
                    ids: SessionIds::new(max_sessions),
                    open: (0..max_sessions).map(|_| Weak::new()).collect(),
                    ticking: false,
                }),
            }),
        }
    }

    pub fn sink(&self) -> &Arc<Sink> {
        &self.shared.sink
    }

    pub fn free_sessions(&self) -> usize {
        self.shared.sessions.lock().ids.available
    }

    /// Opens a session on device `name` with `channel_count` channels (2 or 6) whose buffers
    /// live in `memory`. `event` is signalled whenever buffers are released. The session closes
    /// when the returned `Session` drops.
    pub fn open(
        &self,
        name: &str,
        channel_count: u16,
        memory: Arc<AddressSpace>,
        applet_resource_user_id: u64,
        event: BufferEvent,
    ) -> HleResult<Arc<Session>> {
        let mut sessions = self.shared.sessions.lock();
        if !sessions.ticking {
            sessions.ticking = true;
            spawn_ticker(&self.shared);
        }
        let id = sessions.ids.acquire().ok_or_else(|| {
            tracing::error!(kind = ?self.shared.ty, "every audio session is in use");
            result::OUT_OF_SESSIONS
        })?;
        let session = Arc::new(Session {
            manager: self.shared.clone(),
            id,
            name: name.into(),
            channel_count,
            applet_resource_user_id,
            event,
            system: Mutex::new(System {
                state: State::Stopped,
                volume: 1.0,
                buffers: AudioBuffers::default(),
                session: DeviceSession::new(
                    format!("{name}-{id}"),
                    self.shared.ty,
                    usize::from(channel_count),
                    memory,
                ),
            }),
        });
        sessions.open[id] = Arc::downgrade(&session);
        Ok(session)
    }

    /// Every open session.
    pub fn open_sessions(&self) -> Vec<Arc<Session>> {
        let sessions = self.shared.sessions.lock();
        sessions.open.iter().filter_map(Weak::upgrade).collect()
    }
}

fn spawn_ticker(shared: &Arc<ManagerShared>) {
    let target = Arc::downgrade(shared);
    let name = match shared.ty {
        StreamType::In => "audio-in",
        _ => "audio-out",
    };
    let spawned = std::thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let mut open = Vec::new();
            while let Some(shared) = target.upgrade() {
                open.resize(shared.sessions.lock().open.len(), None);
                shared.tick(&mut open);
                drop(shared);
                std::thread::sleep(TICK);
            }
        });
    if let Err(err) = spawned {
        tracing::error!("starting the {name} manager: {err}");
    }
}

struct System {
    state: State,
    volume: f32,
    buffers: AudioBuffers,
    session: DeviceSession,
}

impl System {
    fn register(&mut self) -> usize {
        if self.state != State::Started {
            return 0;
        }
        let session = &mut self.session;
        self.buffers.register(|buffer| {
            if let Err(code) = session.append(buffer) {
                tracing::warn!("registering an audio buffer: {code:?}");
            }
        })
    }

    fn release(&mut self, force: bool) -> bool {
        let session = &self.session;
        self.buffers.release(
            force,
            |buffer| session.is_consumed(buffer),
            |buffer| session.release(buffer),
        )
    }
}

/// An open session (`IAudioOut` or `IAudioIn`).
pub struct Session {
    manager: Arc<ManagerShared>,
    id: usize,
    name: String,
    channel_count: u16,
    applet_resource_user_id: u64,
    event: BufferEvent,
    system: Mutex<System>,
}

impl Session {
    pub fn id(&self) -> usize {
        self.id
    }

    /// The device the session plays on (or records from).
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn channel_count(&self) -> u16 {
        self.channel_count
    }

    pub fn applet_resource_user_id(&self) -> u64 {
        self.applet_resource_user_id
    }

    pub fn state(&self) -> State {
        self.system.lock().state
    }

    /// What `OpenAudioOut` and `OpenAudioIn` answer for this session.
    pub fn parameter_internal(&self) -> ParameterInternal {
        ParameterInternal {
            sample_rate: SAMPLE_RATE,
            channel_count: u32::from(self.channel_count),
            sample_format: SAMPLE_FORMAT_PCM_INT16,
            state: self.state() as u32,
        }
    }

    /// Starts the stream and queues the buffers appended while stopped.
    pub fn start(&self) -> HleResult<()> {
        let mut system = self.system.lock();
        if system.state != State::Stopped {
            return Err(result::OPERATION_FAILED);
        }
        let volume = system.volume;
        system.session.start(&self.manager.sink);
        system.session.set_volume(volume);
        system.state = State::Started;
        let registered = system.register();
        system.session.set_ring_size(registered);
        Ok(())
    }

    /// Stops the stream and releases every registered buffer unplayed.
    pub fn stop(&self) -> HleResult<()> {
        let mut system = self.system.lock();
        if system.state == State::Started {
            system.session.stop();
            system.session.set_volume(0.0);
            system.session.clear_buffers();
            if system.release(true) {
                (self.event)();
            }
            system.state = State::Stopped;
        }
        Ok(())
    }

    /// Appends a guest buffer of `size` bytes at `samples`; `tag` identifies it in
    /// [`released_buffers`](Self::released_buffers).
    pub fn append_buffer(&self, samples: u64, size: u64, tag: u64) -> HleResult<()> {
        let mut system = self.system.lock();
        if !system.buffers.append(AudioBuffer::new(samples, size, tag)) {
            return Err(result::BUFFER_COUNT_REACHED);
        }
        system.register();
        Ok(())
    }

    /// The manager's tick: releases played buffers (signalling the event when one was released
    /// or none is left registered) and registers waiting ones.
    pub fn release_and_register(&self) {
        let mut system = self.system.lock();
        if system.state != State::Started {
            return;
        }
        if system.release(false) {
            (self.event)();
        }
        system.register();
    }

    /// Takes the released buffers: writes their tags and returns how many.
    pub fn released_buffers(&self, tags: &mut [u64]) -> u32 {
        self.system.lock().buffers.take_released(tags)
    }

    /// Releases every queued buffer unplayed. `false` when stopped.
    pub fn flush(&self) -> bool {
        let mut system = self.system.lock();
        if system.state != State::Started {
            return false;
        }
        if system.buffers.flush() > 0 {
            (self.event)();
        }
        true
    }

    pub fn contains_buffer(&self, tag: u64) -> bool {
        self.system.lock().buffers.contains(tag)
    }

    /// Appended and registered buffers.
    pub fn buffer_count(&self) -> u32 {
        self.system.lock().buffers.active() as u32
    }

    pub fn played_sample_count(&self) -> u64 {
        self.system.lock().session.played_sample_count()
    }

    pub fn volume(&self) -> f32 {
        self.system.lock().volume
    }

    pub fn set_volume(&self, volume: f32) {
        let mut system = self.system.lock();
        system.volume = volume;
        system.session.set_volume(volume);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let system = self.system.get_mut();
        system.session.stop();
        system.session.clear_buffers();
        let mut sessions = self.manager.sessions.lock();
        sessions.open[self.id] = Weak::new();
        sessions.ids.release(self.id);
        tracing::debug!(kind = ?self.manager.ty, session = self.id, "audio session closed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_ids_are_reused_in_release_order() {
        let mut ids = SessionIds::new(3);
        assert_eq!(
            [ids.acquire(), ids.acquire(), ids.acquire()],
            [Some(0), Some(1), Some(2)]
        );
        assert_eq!(ids.acquire(), None);
        ids.release(1);
        ids.release(0);
        assert_eq!([ids.acquire(), ids.acquire()], [Some(1), Some(0)]);
    }
}
