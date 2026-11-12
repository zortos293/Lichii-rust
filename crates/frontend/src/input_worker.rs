//! The input thread: owns the host input state, resolves every player's npad state at a fixed
//! rate, publishes it to the `NpadHub` and reports controller hotkey combos.

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{RecvTimeoutError, Sender};
use lichii_common::settings::{ControlSettings, HotkeyAction, MAX_PLAYERS, NpadButton};
use lichii_input::{HostInput, InputMapper, Key, MouseButton, NpadHub};

/// How often the thread resolves and publishes when no event arrives (250 Hz, above hid's
/// sampling rate).
const POLL: Duration = Duration::from_millis(4);

pub enum InputMsg {
    Key(Key, bool),
    Mouse(MouseButton, bool),
    /// Releases every key and mouse button (focus lost, input handed back to the UI).
    ReleaseAll,
    Configure {
        controls: Box<ControlSettings>,
        combos: Vec<(HotkeyAction, Vec<NpadButton>)>,
    },
}

pub struct InputWorker {
    tx: Option<Sender<InputMsg>>,
    hub: Arc<NpadHub>,
    thread: Option<JoinHandle<()>>,
}

impl InputWorker {
    pub fn spawn(
        controls: ControlSettings,
        combos: Vec<(HotkeyAction, Vec<NpadButton>)>,
        on_hotkey: impl Fn(HotkeyAction) + Send + 'static,
    ) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        let hub = Arc::new(NpadHub::default());
        let thread_hub = hub.clone();
        let thread = std::thread::Builder::new()
            .name("input".into())
            .spawn(move || {
                let mut host = HostInput::default();
                let mut mapper = InputMapper::new(&controls, &host);
                let mut combos = combos;
                let mut held = vec![false; combos.len()];
                loop {
                    let first = match rx.recv_timeout(POLL) {
                        Ok(msg) => Some(msg),
                        Err(RecvTimeoutError::Timeout) => None,
                        Err(RecvTimeoutError::Disconnected) => break,
                    };
                    for msg in first.into_iter().chain(rx.try_iter()) {
                        match msg {
                            InputMsg::Key(key, down) => host.keys.set(key, down),
                            InputMsg::Mouse(button, down) => host.set_mouse_button(button, down),
                            InputMsg::ReleaseAll => {
                                host.keys.clear();
                                host.mouse_buttons = 0;
                            }
                            InputMsg::Configure {
                                controls,
                                combos: new_combos,
                            } => {
                                mapper = InputMapper::new(&controls, &host);
                                held = vec![false; new_combos.len()];
                                combos = new_combos;
                            }
                        }
                    }
                    for player in 0..MAX_PLAYERS {
                        thread_hub.publish(player, &mapper.resolve(player, &host));
                    }
                    let buttons = thread_hub.read(0).buttons;
                    for ((action, combo), was_held) in combos.iter().zip(&mut held) {
                        let now = buttons.holds(combo);
                        if now && !*was_held {
                            on_hotkey(*action);
                        }
                        *was_held = now;
                    }
                }
            })
            .expect("spawning the input thread");
        Self {
            tx: Some(tx),
            hub,
            thread: Some(thread),
        }
    }

    pub fn send(&self, msg: InputMsg) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(msg);
        }
    }

    /// The npad states the hid service reads.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "lichii-core's System takes the hub once it exists"
        )
    )]
    pub fn hub(&self) -> &Arc<NpadHub> {
        &self.hub
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
    use lichii_input::NpadButtons;
    use std::sync::Mutex;
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
            move |a| sink.lock().unwrap().push(a),
        );
        let key = |n| Key::from_name(n).unwrap();
        worker.send(InputMsg::Key(key("KeyC"), true));
        wait_for(|| worker.hub().read(0).buttons.contains(NpadButtons::A));

        worker.send(InputMsg::Key(key("Backspace"), true));
        worker.send(InputMsg::Key(key("KeyM"), true));
        wait_for(|| !fired.lock().unwrap().is_empty());
        std::thread::sleep(POLL * 5);
        assert_eq!(*fired.lock().unwrap(), [HotkeyAction::ContinuePause]);

        worker.send(InputMsg::ReleaseAll);
        wait_for(|| worker.hub().read(0).buttons.is_empty());
        assert!(!worker.hub().read(1).connected);
    }
}
