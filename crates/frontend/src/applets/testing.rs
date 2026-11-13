use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;

use egui::{Event, Key, Modifiers, RawInput};
use lichii_common::settings::ControlSettings;
use lichii_services::applets::AppletFrontend;

use super::AppletDialogs;

/// Drives [`AppletDialogs`] with headless egui frames while a second thread waits in the
/// [`AppletFrontend`], the way am's service thread does.
pub struct Harness {
    ctx: egui::Context,
    pub dialogs: AppletDialogs,
    wakes: Arc<AtomicUsize>,
    pub controls: ControlSettings,
    /// The controls the last frame returned, if any.
    pub applied: Option<ControlSettings>,
}

impl Harness {
    pub fn new() -> Self {
        let wakes = Arc::new(AtomicUsize::new(0));
        let counter = wakes.clone();
        Self {
            ctx: egui::Context::default(),
            dialogs: AppletDialogs::new("Lichii", move || {
                counter.fetch_add(1, Ordering::Relaxed);
            }),
            wakes,
            controls: ControlSettings::default(),
            applied: None,
        }
    }

    /// Calls the frontend on another thread and returns once it woke the UI.
    pub fn ask<T: Send + 'static>(
        &self,
        call: impl FnOnce(&dyn AppletFrontend) -> T + Send + 'static,
    ) -> JoinHandle<T> {
        let frontend = self.dialogs.frontend();
        let woken = self.wakes.load(Ordering::Relaxed);
        let handle = std::thread::spawn(move || call(frontend.as_ref()));
        while self.wakes.load(Ordering::Relaxed) == woken {
            std::thread::yield_now();
        }
        handle
    }

    pub fn frame(&mut self, events: Vec<Event>) {
        let input = RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 720.0),
            )),
            events,
            ..RawInput::default()
        };
        let (dialogs, controls) = (&mut self.dialogs, &self.controls);
        let mut applied = None;
        let mut output = self
            .ctx
            .run_ui(input, |ui| applied = dialogs.show(ui.ctx(), controls));
        output.textures_delta.clear();
        if applied.is_some() {
            self.applied = applied;
        }
    }

    pub fn typed(&mut self, text: &str) {
        self.frame(vec![Event::Text(text.to_owned())]);
    }

    pub fn key(&mut self, key: Key) {
        let press = |pressed| Event::Key {
            key,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: Modifiers::NONE,
        };
        self.frame(vec![press(true), press(false)]);
    }
}
