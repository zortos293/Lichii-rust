//! Boots a title without a window and logs its progress until it exits.
//!
//! `cargo run -p lichii-core --example headless -- <title> [--seconds N]`

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use lichii_common::paths::UserPaths;
use lichii_common::settings::Settings;
use lichii_core::{System, headless_renderer};

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    let mut title = None;
    let mut seconds = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seconds" => seconds = args.next().and_then(|s| s.parse::<u64>().ok()),
            _ => title = Some(PathBuf::from(arg)),
        }
    }
    let Some(title) = title else {
        eprintln!("usage: headless <title> [--seconds N]");
        return std::process::ExitCode::FAILURE;
    };

    let paths = UserPaths::discover();
    let settings = Settings::load(&paths.settings_file()).unwrap_or_else(|e| {
        eprintln!("{e}; using default settings");
        Settings::default()
    });
    lichii_common::init_logging(&settings.debug.log_filter);

    let mut system = match System::boot(Arc::new(settings), &title, headless_renderer()) {
        Ok(system) => system,
        Err(e) => {
            tracing::error!("boot failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    if let Err(e) = system.run() {
        tracing::error!("start failed: {e}");
        return std::process::ExitCode::FAILURE;
    }
    match system.wait_for_exit(seconds.map(Duration::from_secs)) {
        Some(reason) => tracing::info!(?reason, "title stopped"),
        None => tracing::info!("time limit reached"),
    }
    system.stop();
    std::process::ExitCode::SUCCESS
}
