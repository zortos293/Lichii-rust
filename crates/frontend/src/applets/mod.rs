//! Library applets the running title opens, shown as egui dialogs. am asks on its service
//! thread and waits for the answer; the dialog answers from the UI thread. Once the dialogs are
//! dropped (the session stops), every question still waiting is answered as cancelled.

mod controller;
mod keyboard;
mod profile;
#[cfg(test)]
mod testing;

use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};
use lichii_common::settings::ControlSettings;
use lichii_services::applets::{
    AppletFrontend, ControllerAnswer, ControllerRequest, ErrorRequest, HeadlessApplets,
    ProfileSelectAnswer, ProfileSelectRequest, SoftwareKeyboardAnswer, SoftwareKeyboardRequest,
};

use self::controller::ControllerDialog;
use self::keyboard::KeyboardDialog;
use self::profile::ProfileDialog;

enum Request {
    Keyboard(SoftwareKeyboardRequest, Sender<SoftwareKeyboardAnswer>),
    Profile(ProfileSelectRequest, Sender<ProfileSelectAnswer>),
    Controller(ControllerRequest, Sender<ControllerAnswer>),
}

/// The [`AppletFrontend`] a session installs: it hands each request to [`AppletDialogs`] and wakes
/// the UI. Applets without a dialog yet answer like [`HeadlessApplets`].
struct UiApplets {
    requests: Sender<Request>,
    wake: Arc<dyn Fn() + Send + Sync>,
}

impl UiApplets {
    fn ask<T>(&self, request: impl FnOnce(Sender<T>) -> Request) -> Option<T> {
        let (reply, answer) = crossbeam_channel::bounded(1);
        self.requests.send(request(reply)).ok()?;
        (self.wake)();
        answer.recv().ok()
    }
}

impl AppletFrontend for UiApplets {
    fn show_error(&self, request: &ErrorRequest) {
        HeadlessApplets.show_error(request);
    }

    fn select_profile(&self, request: &ProfileSelectRequest) -> ProfileSelectAnswer {
        self.ask(|reply| Request::Profile(request.clone(), reply))
            .unwrap_or(ProfileSelectAnswer::Cancelled)
    }

    fn configure_controllers(&self, request: &ControllerRequest) -> ControllerAnswer {
        self.ask(|reply| Request::Controller(*request, reply))
            .unwrap_or(ControllerAnswer::Cancelled)
    }

    fn software_keyboard(&self, request: &SoftwareKeyboardRequest) -> SoftwareKeyboardAnswer {
        self.ask(|reply| Request::Keyboard(request.clone(), reply))
            .unwrap_or(SoftwareKeyboardAnswer::Cancelled)
    }
}

enum Dialog {
    Keyboard(KeyboardDialog),
    Profile(ProfileDialog),
    Controller(ControllerDialog),
}

/// The UI side of a session's applets: shows one applet dialog at a time, in the order the
/// title opened them.
pub struct AppletDialogs {
    sender: Sender<Request>,
    requests: Receiver<Request>,
    wake: Arc<dyn Fn() + Send + Sync>,
    /// The nickname of the user acc signed in, which the profile selector offers.
    username: String,
    open: Option<Dialog>,
}

impl AppletDialogs {
    /// Dialogs for a session whose user is `username`. Their frontend calls `wake` when a title
    /// opens an applet, so the UI redraws.
    pub fn new(username: impl Into<String>, wake: impl Fn() + Send + Sync + 'static) -> Self {
        let (sender, requests) = crossbeam_channel::unbounded();
        Self {
            sender,
            requests,
            wake: Arc::new(wake),
            username: username.into(),
            open: None,
        }
    }

    /// The frontend to install with `ServiceContext::set_applet_frontend`.
    pub fn frontend(&self) -> Arc<dyn AppletFrontend> {
        Arc::new(UiApplets {
            requests: self.sender.clone(),
            wake: self.wake.clone(),
        })
    }

    /// Whether an applet dialog is waiting for the user.
    pub fn is_open(&self) -> bool {
        self.open.is_some() || !self.requests.is_empty()
    }

    /// Shows the applet the title waits on. The controller applet starts from `controls` and
    /// returns them changed when the user confirmed a different setup.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        controls: &ControlSettings,
    ) -> Option<ControlSettings> {
        if self.open.is_none() {
            self.open = self.requests.try_recv().ok().map(|request| match request {
                Request::Keyboard(request, reply) => {
                    Dialog::Keyboard(KeyboardDialog::new(request, reply))
                }
                Request::Profile(request, reply) => {
                    Dialog::Profile(ProfileDialog::new(request, reply, &self.username))
                }
                Request::Controller(request, reply) => {
                    Dialog::Controller(ControllerDialog::new(request, reply, controls))
                }
            });
        }
        let changed = match self.open.as_mut()? {
            Dialog::Keyboard(dialog) => {
                let answer = dialog.show(ctx)?;
                let _ = dialog.reply.send(answer);
                None
            }
            Dialog::Profile(dialog) => {
                let answer = dialog.show(ctx)?;
                let _ = dialog.reply.send(answer);
                None
            }
            Dialog::Controller(dialog) => {
                let (answer, changed) = dialog.show(ctx)?;
                let _ = dialog.reply.send(answer);
                changed
            }
        };
        self.open = None;
        ctx.request_repaint();
        changed
    }
}
