use lichii_common::paths::UserPaths;
use lichii_common::settings::Settings;

fn main() {
    let paths = UserPaths::discover();
    let settings = Settings::load(&paths.settings_file()).unwrap_or_else(|e| {
        eprintln!("{e}; using default settings");
        Settings::default()
    });
    lichii_common::init_logging(&settings.debug.log_filter);
    tracing::info!("Lichii {}", env!("CARGO_PKG_VERSION"));
}
