//! The profile selector, after Lichii's QtProfileSelectionDialog. acc signs in one user per
//! boot, the one the Username setting names. As the reference skips its selector when there is
//! a single user, the title gets that user without a dialog; the dialog only shows when the
//! title does not allow it.

use crossbeam_channel::Sender;
use egui::{Key, RichText};
use lichii_services::applets::{ProfileSelectAnswer, ProfileSelectRequest};

/// The signed-in user, unless the title does not allow it.
pub(super) fn selectable_user(request: &ProfileSelectRequest) -> Option<[u8; 16]> {
    request
        .current_user
        .filter(|uid| *uid != [0; 16] && !request.invalid_users.contains(uid))
}

/// Tells the user that no user can be chosen, and cancels.
pub(super) struct ProfileDialog {
    request: ProfileSelectRequest,
    pub reply: Sender<ProfileSelectAnswer>,
}

impl ProfileDialog {
    pub fn new(request: ProfileSelectRequest, reply: Sender<ProfileSelectAnswer>) -> Self {
        Self { request, reply }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<ProfileSelectAnswer> {
        let mut cancel = ctx.input(|i| i.key_pressed(Key::Escape));
        egui::Modal::new(egui::Id::unique("profile selector")).show(ctx, |ui| {
            ui.set_width(360.0);
            ui.heading(title(self.request.mode));
            ui.label(instruction(self.request.purpose));
            ui.add_space(6.0);
            ui.label(
                RichText::new(
                    "No user can be chosen. Lichii signs in the user the Username setting \
                     names, and this title does not allow that user.",
                )
                .weak(),
            );
            ui.add_space(8.0);
            if ui.button("Cancel").clicked() {
                cancel = true;
            }
        });
        cancel.then_some(ProfileSelectAnswer::Cancelled)
    }
}

/// The window title for an `nn::account::UiMode`.
fn title(mode: u32) -> &'static str {
    match mode {
        1 | 5 => "Profile Creator",
        3 => "Profile Icon Editor",
        4 => "Profile Nickname Editor",
        _ => "Profile Selector",
    }
}

/// The question for an `nn::account::UserSelectionPurpose`.
fn instruction(purpose: u32) -> &'static str {
    match purpose {
        1 => "Who will receive the points?",
        2 => "Who is using Nintendo eShop?",
        3 => "Who is making this purchase?",
        4 => "Who is posting?",
        5 => "Select a user to link to a Nintendo Account.",
        6 => "Change settings for which user?",
        7 => "Format data for which user?",
        8 => "Which user will be transferred to another console?",
        9 => "Send save data for which user?",
        _ => "Select a user:",
    }
}

#[cfg(test)]
mod tests {
    use std::thread::JoinHandle;

    use super::super::testing::Harness;
    use super::*;

    const USER: [u8; 16] = [7; 16];

    fn select(ui: &Harness, request: ProfileSelectRequest) -> JoinHandle<ProfileSelectAnswer> {
        ui.ask(move |frontend| frontend.select_profile(&request))
    }

    fn request() -> ProfileSelectRequest {
        ProfileSelectRequest {
            current_user: Some(USER),
            ..ProfileSelectRequest::default()
        }
    }

    #[test]
    fn the_signed_in_user_is_selected_without_a_dialog() {
        let mut ui = Harness::new();
        let answer = ui.dialogs.frontend().select_profile(&request());
        assert_eq!(answer, ProfileSelectAnswer::Selected(USER));
        ui.frame(Vec::new());
        assert!(!ui.dialogs.is_open());
    }

    #[test]
    fn a_user_the_title_does_not_allow_cannot_be_chosen() {
        let mut ui = Harness::new();
        let mut invalid_users = [[0; 16]; 8];
        invalid_users[3] = USER;
        let answer = select(
            &ui,
            ProfileSelectRequest {
                invalid_users,
                ..request()
            },
        );
        ui.frame(Vec::new());
        ui.key(Key::Enter);
        assert!(ui.dialogs.is_open(), "OK needs a user");
        ui.key(Key::Escape);
        assert_eq!(answer.join().unwrap(), ProfileSelectAnswer::Cancelled);

        let answer = select(
            &ui,
            ProfileSelectRequest {
                current_user: None,
                ..request()
            },
        );
        ui.frame(Vec::new());
        ui.key(Key::Enter);
        assert!(ui.dialogs.is_open(), "no user is signed in");
        ui.key(Key::Escape);
        assert_eq!(answer.join().unwrap(), ProfileSelectAnswer::Cancelled);
    }

    #[test]
    fn titles_and_questions_follow_the_mode_and_purpose() {
        assert_eq!(title(0), "Profile Selector");
        assert_eq!(title(5), "Profile Creator");
        assert_eq!(instruction(0), "Select a user:");
        assert_eq!(instruction(2), "Who is using Nintendo eShop?");
        assert_eq!(instruction(99), "Select a user:");
    }
}
