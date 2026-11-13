mod app;
mod choices;
mod cli;
mod config;
mod display;
mod frame_view;
mod game_list;
mod hotkeys;
mod input_worker;
mod per_game;
mod session;
mod setup;
mod ui;

use std::path::Path;
use std::process::ExitCode;

use lichii_common::paths::UserPaths;
use lichii_common::settings::Settings;
use winit::event_loop::EventLoop;

use crate::app::{App, UserEvent};
use crate::cli::Command;

/// Reads the settings without logging, so the log filter they hold applies to their warnings. A
/// file that cannot be read or is not TOML is copied to `<file>.bak`, because the defaults that
/// replace it are saved; when that copy fails, saving is turned off (the returned `false`).
fn read_settings(path: &Path) -> (Settings, Vec<String>, bool) {
    let problem = match std::fs::read_to_string(path) {
        Ok(text) => match Settings::from_toml(&text) {
            Ok((settings, warnings)) => return (settings, warnings, true),
            Err(e) => format!("is not valid TOML ({e})"),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return (Settings::default(), vec![], true);
        }
        Err(e) => format!("cannot be read ({e})"),
    };
    let mut backup = path.as_os_str().to_owned();
    backup.push(".bak");
    match std::fs::copy(path, &backup) {
        Ok(_) => {
            let kept = backup.to_string_lossy();
            let warning =
                format!("{problem}; using default settings, the original is kept as {kept}");
            (Settings::default(), vec![warning], true)
        }
        Err(e) => {
            let warning = format!(
                "{problem}; using default settings without saving them, because backing the file up failed ({e})"
            );
            (Settings::default(), vec![warning], false)
        }
    }
}

fn main() -> ExitCode {
    let cli = match cli::parse(std::env::args_os().skip(1)) {
        Ok(Command::Run(cli)) => cli,
        Ok(Command::Help) => {
            println!("{}", cli::USAGE);
            return ExitCode::SUCCESS;
        }
        Ok(Command::Version) => {
            println!("lichii {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("lichii: {e}\n\n{}", cli::USAGE);
            return ExitCode::from(2);
        }
    };
    let paths = UserPaths::discover();
    let settings_path = cli.config.clone().unwrap_or_else(|| paths.settings_file());
    let (settings, warnings, writable) = read_settings(&settings_path);
    lichii_common::init_logging(&settings.debug.log_filter);
    tracing::info!("Lichii {}", env!("CARGO_PKG_VERSION"));
    for warning in warnings {
        tracing::warn!("{}: {warning}", settings_path.display());
    }

    let event_loop = match EventLoop::<UserEvent>::with_user_event().build() {
        Ok(event_loop) => event_loop,
        Err(e) => {
            eprintln!("lichii: cannot open a window: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut app = App::new(
        &paths,
        settings,
        writable.then_some(settings_path),
        cli,
        event_loop.create_proxy(),
    );
    match event_loop.run_app(&mut app) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lichii: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unreadable_settings_are_backed_up_or_never_saved() {
        let dir = std::env::temp_dir().join(format!("lichii-read-settings-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.toml");

        assert!(read_settings(&path).2);

        std::fs::write(&path, "not = [toml").unwrap();
        let (_, warnings, writable) = read_settings(&path);
        assert!(writable);
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            std::fs::read_to_string(dir.join("settings.toml.bak")).unwrap(),
            "not = [toml"
        );

        std::fs::remove_file(dir.join("settings.toml.bak")).unwrap();
        std::fs::create_dir(dir.join("settings.toml.bak")).unwrap();
        assert!(!read_settings(&path).2);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
