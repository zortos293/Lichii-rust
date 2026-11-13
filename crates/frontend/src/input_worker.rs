//! The input thread: owns the host input state and the SDL gamepad driver, resolves every
//! player's npad state at a fixed rate, publishes it to the `NpadHub`, forwards the guest's
//! rumble requests to the pads and reports controller hotkey combos.

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{RecvTimeoutError, Sender};
use lichii_common::settings::{ControlSettings, HotkeyAction, MAX_PLAYERS, NpadButton};
use lichii_input::{
    Gamepad, HostInput, InputMapper, Key, MouseButton, NpadButtons, NpadHub, NpadState, SdlInput,
    UPDATE_PERIOD,
};

/// How often the thread polls SDL, resolves and publishes.
const POLL: Duration = UPDATE_PERIOD;
/// Each rumble request lasts this long unless a newer one replaces it.
const RUMBLE_MS: u32 = 1000;
const TEST_RUMBLE_MS: u32 = 400;

pub enum InputMsg {
    Key(Key, bool),
    Mouse(MouseButton, bool),
    /// Releases every key and mouse button (focus lost, input handed back to the UI).
    ReleaseAll,
    /// Whether the guest sees input. While it doesn't, players publish as neutral, but host input
    /// is still tracked so controller hotkeys (e.g. Home+Plus to continue) keep working.
    GuestInput(bool),
    Configure {
        controls: Box<ControlSettings>,
        combos: Vec<(HotkeyAction, Vec<NpadButton>)>,
    },
    /// Copy the full host input into [`InputShared::host`] every poll (for click-to-bind).
    Mirror(bool),
    /// Rumble player `n`'s pad briefly at its configured strength.
    TestRumble(usize),
}

/// What the input thread reports back.
#[derive(Clone, Copy, Debug)]
pub enum InputEvent {
    Hotkey(HotkeyAction),
    /// A gamepad connected or disconnected; [`InputShared::devices`] changed.
    DevicesChanged,
}

/// State the UI reads from the input thread.
#[derive(Default)]
pub struct InputShared {
    /// Connected gamepads; `state` is not kept current here.
    pub devices: Vec<Gamepad>,
    /// The latest host input while mirroring is on.
    pub host: Option<HostInput>,
}

pub struct InputWorker {
    tx: Option<Sender<InputMsg>>,
    hub: Arc<NpadHub>,
    shared: Arc<Mutex<InputShared>>,
    thread: Option<JoinHandle<()>>,
}

impl InputWorker {
    pub fn spawn(
        controls: ControlSettings,
        combos: Vec<(HotkeyAction, Vec<NpadButton>)>,
        notify: impl Fn(InputEvent) + Send + 'static,
    ) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        let hub = Arc::new(NpadHub::default());
        let shared = Arc::new(Mutex::new(InputShared::default()));
        let (thread_hub, thread_shared) = (hub.clone(), shared.clone());
        let thread = std::thread::Builder::new()
            .name("input".into())
            .spawn(move || {
                let mut sdl = SdlInput::new()
                    .inspect_err(|e| tracing::warn!("gamepads unavailable: {e}"))
                    .ok();
                let mut host = HostInput::default();
                let mut controls = controls;
                let mut mapper = InputMapper::new(&controls, &host);
                let mut combos = combos;
                let mut held = vec![false; combos.len()];
                let mut rumble_seen = [0u64; MAX_PLAYERS];
                let mut mirror = false;
                let mut guest_input = false;
                loop {
                    let first = match rx.recv_timeout(POLL) {
                        Ok(msg) => Some(msg),
                        Err(RecvTimeoutError::Timeout) => None,
                        Err(RecvTimeoutError::Disconnected) => break,
                    };
                    let mut reconfigure = false;
                    for msg in first.into_iter().chain(rx.try_iter()) {
                        match msg {
                            InputMsg::Key(key, down) => host.keys.set(key, down),
                            InputMsg::Mouse(button, down) => host.set_mouse_button(button, down),
                            InputMsg::GuestInput(on) => guest_input = on,
                            InputMsg::ReleaseAll => {
                                host.keys.clear();
                                host.mouse_buttons = 0;
                            }
                            InputMsg::Configure {
                                controls: new_controls,
                                combos: new_combos,
                            } => {
                                controls = *new_controls;
                                held = vec![false; new_combos.len()];
                                combos = new_combos;
                                reconfigure = true;
                            }
                            InputMsg::Mirror(on) => {
                                mirror = on;
                                if !on {
                                    thread_shared.lock().unwrap().host = None;
                                }
                            }
                            InputMsg::TestRumble(player) => {
                                let target =
                                    mapper.players().get(player).and_then(|p| p.rumble_target());
                                if let (Some(sdl), Some((pad, strength))) = (&mut sdl, target) {
                                    let motor = (strength * 65535.0) as u16;
                                    sdl.rumble(pad, motor, motor, TEST_RUMBLE_MS);
                                }
                            }
                        }
                    }
                    if let Some(sdl) = &mut sdl
                        && sdl.poll(&mut host)
                    {
                        reconfigure = true;
                        thread_shared.lock().unwrap().devices = host
                            .gamepads
                            .iter()
                            .map(|g| Gamepad {
                                state: Default::default(),
                                ..g.clone()
                            })
                            .collect();
                        notify(InputEvent::DevicesChanged);
                    }
                    if reconfigure {
                        mapper = InputMapper::new(&controls, &host);
                    }
                    let mut player_one = NpadButtons::empty();
                    for player in 0..MAX_PLAYERS {
                        let state = mapper.resolve(player, &host);
                        if player == 0 {
                            player_one = state.buttons;
                        }
                        let published = if guest_input {
                            state
                        } else {
                            NpadState {
                                connected: state.connected,
                                controller: state.controller,
                                ..NpadState::default()
                            }
                        };
                        thread_hub.publish(player, &published);
                    }
                    if let Some(sdl) = &mut sdl {
                        for (player, seen) in rumble_seen.iter_mut().enumerate() {
                            let request = thread_hub.take_vibration(player, seen);
                            let target =
                                mapper.players().get(player).and_then(|p| p.rumble_target());
                            if let (Some(value), Some((pad, strength))) = (request, target) {
                                let (low, high) = value.to_rumble(strength);
                                sdl.rumble(pad, low, high, RUMBLE_MS);
                            }
                        }
                    }
                    if mirror {
                        let mut shared = thread_shared.lock().unwrap();
                        match &mut shared.host {
                            Some(copy) => copy.clone_from(&host),
                            None => shared.host = Some(host.clone()),
                        }
                    }
                    for ((action, combo), was_held) in combos.iter().zip(&mut held) {
                        let now = player_one.holds(combo);
                        if now && !*was_held {
                            notify(InputEvent::Hotkey(*action));
                        }
                        *was_held = now;
                    }
                }
            })
            .expect("spawning the input thread");
        Self {
            tx: Some(tx),
            hub,
            shared,
            thread: Some(thread),
        }
    }

    pub fn send(&self, msg: InputMsg) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(msg);
        }
    }

    /// The npad states the hid service reads.
    pub fn hub(&self) -> &Arc<NpadHub> {
        &self.hub
    }

    pub fn shared(&self) -> &Arc<Mutex<InputShared>> {
        &self.shared
    }
}

impl Drop for InputWorker {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lichii_common::settings::Settings;
    use std::time::Instant;

    fn wait_for(mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !done() {
            assert!(Instant::now() < deadline, "timed out");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn keys_reach_the_hub_and_fire_combos_once() {
        let settings = Settings::default();
        let fired = Arc::new(Mutex::new(Vec::new()));
        let sink = fired.clone();
        let worker = InputWorker::spawn(
            settings.controls.clone(),
            crate::hotkeys::combos(&settings.ui.hotkeys),
            move |event| {
                if let InputEvent::Hotkey(a) = event {
                    sink.lock().unwrap().push(a);
                }
            },
        );
        let key = |n| Key::from_name(n).unwrap();
        worker.send(InputMsg::Key(key("KeyC"), true));
        wait_for(|| worker.hub().read(0).connected);
        assert!(worker.hub().read(0).buttons.is_empty());
        worker.send(InputMsg::GuestInput(true));
        wait_for(|| worker.hub().read(0).buttons.contains(NpadButtons::A));
        worker.send(InputMsg::GuestInput(false));
        wait_for(|| worker.hub().read(0).buttons.is_empty());

        worker.send(InputMsg::Key(key("Backspace"), true));
        worker.send(InputMsg::Key(key("KeyM"), true));
        wait_for(|| !fired.lock().unwrap().is_empty());
        std::thread::sleep(POLL * 5);
        assert_eq!(*fired.lock().unwrap(), [HotkeyAction::ContinuePause]);

        worker.send(InputMsg::Mirror(true));
        wait_for(|| {
            worker
                .shared()
                .lock()
                .unwrap()
                .host
                .as_ref()
                .is_some_and(|h| h.keys.is_down(key("KeyC")))
        });
        worker.send(InputMsg::ReleaseAll);
        wait_for(|| worker.hub().read(0).buttons.is_empty());
        assert!(!worker.hub().read(1).connected);
    }
}
