mod app;
mod choices;
mod cli;
mod game_list;
mod hotkeys;
mod input_worker;
mod presenter;
mod session;
mod ui;
mod viewport;

use std::path::Path;
use std::process::ExitCode;

use lichii_common::paths::UserPaths;
use lichii_common::settings::Settings;
use winit::event_loop::EventLoop;

use crate::app::{App, UserEvent};
use crate::cli::Command;

/// Reads the settings without logging, so the log filter they hold applies to their warnings. A
/// file that is not TOML is copied to `<file>.bak` first, because the defaults that replace it are
/// saved on exit.
fn read_settings(path: &Path) -> (Settings, Vec<String>) {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (Settings::default(), vec![]),
        Err(e) => {
            let warning = format!("reading {}: {e}", path.display());
            return (Settings::default(), vec![warning]);
        }
    };
    Settings::from_toml(&text).unwrap_or_else(|e| {
        let mut backup = path.as_os_str().to_owned();
        backup.push(".bak");
        let kept = match std::fs::copy(path, &backup) {
            Ok(_) => format!("the original is kept as {}", backup.to_string_lossy()),
            Err(copy) => format!("backing it up failed ({copy})"),
        };
        let warning = format!(
            "{} is not valid TOML ({e}); using default settings, {kept}",
            path.display()
        );
        (Settings::default(), vec![warning])
    })
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
    let (settings, warnings) = read_settings(&settings_path);
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
        settings_path,
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
