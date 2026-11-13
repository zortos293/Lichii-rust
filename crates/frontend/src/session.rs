//! One emulation session: a booted [`System`] running the title, owned by the app and stopped when
//! dropped.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lichii_common::ConsoleGeneration;
use lichii_common::settings::{AudioSettings, ControlSettings, Settings};
use lichii_core::{ExitReason, Host, System, SystemError};
use lichii_crypto::KeyError;
use lichii_loader::LoadOptions;

use crate::applets::AppletDialogs;
use crate::game_list::GameKind;

#[derive(Debug)]
pub enum BootError {
    NotFound(PathBuf),
    Unsupported(PathBuf),
    System(SystemError),
}

impl std::fmt::Display for BootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(path) => write!(f, "{} does not exist", path.display()),
            Self::Unsupported(path) => write!(
                f,
                "{} is not an NSP, XCI, NCA, NRO, NSO or ExeFS directory",
                path.display()
            ),
            Self::System(err) => err.fmt(f),
        }
    }
}

impl BootError {
    /// The key a title needed and the loaded keys lack, when that is why it did not boot.
    pub fn missing_key(&self) -> Option<&KeyError> {
        match self {
            Self::System(SystemError::Loader(error)) => crate::setup::missing_key(error),
            _ => None,
        }
    }
}

impl From<SystemError> for BootError {
    fn from(err: SystemError) -> Self {
        Self::System(err)
    }
}

pub struct Session {
    pub path: PathBuf,
    pub title: String,
    /// The loaded program's id; 0 for homebrew and bare NSOs.
    pub program_id: u64,
    /// Dropped before `system`, so an applet the title is waiting on answers and the title can stop.
    applets: Option<AppletDialogs>,
    system: System,
}

impl Session {
    /// Loads the title at `path` with `options` (keys, installed updates and DLC) and starts it.
    /// Library applets show in `applets`; without them they answer as
    /// [`lichii_services::applets::HeadlessApplets`] does.
    /// `configure` picks the settings (a game's own configuration applies by its program id) and
    /// builds the renderer and shared state for its console generation, both known once the title
    /// is loaded.
    pub fn boot(
        path: &Path,
        options: LoadOptions,
        applets: Option<AppletDialogs>,
        configure: impl FnOnce(u64, ConsoleGeneration) -> (Arc<Settings>, Host),
    ) -> Result<Self, BootError> {
        if !path.exists() {
            return Err(BootError::NotFound(path.to_path_buf()));
        }
        let kind = GameKind::detect(path).ok_or_else(|| BootError::Unsupported(path.into()))?;
        tracing::info!("booting {} ({})", path.display(), kind.label());
        let image = lichii_loader::load(path, &options).map_err(SystemError::from)?;
        let program_id = image.program_id;
        let (settings, host) = configure(program_id, image.generation);
        let mut system = System::from_image(settings, image, host)?;
        if let Some(applets) = &applets {
            system.services().set_applet_frontend(applets.frontend());
        }
        system.run()?;
        Ok(Self {
            path: path.to_path_buf(),
            title: title(path, kind),
            program_id,
            applets,
            system,
        })
    }

    pub fn is_paused(&self) -> bool {
        self.system.is_paused()
    }

    pub fn pause(&mut self) {
        self.system.pause();
    }

    pub fn resume(&mut self) {
        if let Err(err) = self.system.run() {
            tracing::error!("resuming: {err}");
        }
    }

    pub fn toggle_pause(&mut self) {
        if self.is_paused() {
            self.resume();
        } else {
            self.pause();
        }
    }

    /// Applies volume and mute to the running title's audio.
    pub fn set_audio_volume(&self, audio: &AudioSettings) {
        self.system.set_audio_volume(audio);
    }

    /// Shows the library applet the title is waiting on, if any. Returns the controls when the
    /// controller applet changed them.
    pub fn show_applets(
        &mut self,
        ctx: &egui::Context,
        controls: &ControlSettings,
    ) -> Option<ControlSettings> {
        self.applets.as_mut()?.show(ctx, controls)
    }

    /// Whether the title is waiting on a library applet dialog.
    pub fn applet_open(&self) -> bool {
        self.applets.as_ref().is_some_and(AppletDialogs::is_open)
    }

    /// Why the title stopped, once it has.
    pub fn exit_reason(&self) -> Option<ExitReason> {
        self.system.exit_reason()
    }

    #[cfg(test)]
    fn system(&self) -> &System {
        &self.system
    }
}

/// The name a title shows: its file name without extension, or for an ExeFS directory the
/// directory name (the parent of an `exefs` folder).
fn title(path: &Path, kind: GameKind) -> String {
    let name = if kind == GameKind::ExeFs && path.ends_with("exefs") {
        path.parent().unwrap_or(path)
    } else {
        path
    };
    match kind {
        GameKind::ExeFs => name.file_name(),
        _ => name.file_stem(),
    }
    .map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into(),
    )
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::AudioOutput;

    use super::*;

    /// An NSO whose text is `text`, with empty rodata and data.
    fn nso(text: &[u8]) -> Vec<u8> {
        let mut file = vec![0u8; 0x100];
        file[..4].copy_from_slice(b"NSO0");
        let mut file_offset = 0x100u32;
        for (index, size) in [text.len() as u32, 0x10, 0x10].into_iter().enumerate() {
            for (at, value) in [
                (0x10 + index * 0x10, file_offset),
                (0x14 + index * 0x10, index as u32 * 0x1000),
                (0x18 + index * 0x10, size),
                (0x60 + index * 4, size),
            ] {
                file[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
            file_offset += size;
        }
        file.extend_from_slice(text);
        file.resize(file.len() + 0x20, 0);
        file
    }

    fn settings() -> Arc<Settings> {
        let mut settings = Settings::default();
        settings.system.rng_seed = Some(1);
        settings.audio.output = AudioOutput::Null;
        Arc::new(settings)
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("lichii-session-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn rejects_missing_unknown_and_unloadable_titles() {
        let missing = Path::new("/definitely/not/here.nsp");
        assert!(matches!(
            Session::boot(missing, LoadOptions::default(), None, |_, _| (settings(), Host::headless())),
            Err(BootError::NotFound(path)) if path == missing
        ));
        let dir = temp_dir("reject");
        let text = dir.join("notes.txt");
        std::fs::write(&text, "x").unwrap();
        assert!(matches!(
            Session::boot(&text, LoadOptions::default(), None, |_, _| (
                settings(),
                Host::headless()
            )),
            Err(BootError::Unsupported(_))
        ));
        let broken = dir.join("Broken.nro");
        std::fs::write(&broken, "x").unwrap();
        assert!(matches!(
            Session::boot(&broken, LoadOptions::default(), None, |_, _| (
                settings(),
                Host::headless()
            )),
            Err(BootError::System(SystemError::Loader(_)))
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_encrypted_title_without_keys_names_the_missing_key() {
        let dir = temp_dir("keys");
        let nca = dir.join("Game.nca");
        std::fs::write(&nca, vec![0u8; 0x4000]).unwrap();
        let error = Session::boot(&nca, LoadOptions::default(), None, |_, _| {
            (settings(), Host::headless())
        })
        .err()
        .expect("an NCA does not boot without keys");
        assert_eq!(
            error.missing_key(),
            Some(&KeyError::Missing("header_key".into())),
            "{error}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn titles_come_from_file_and_exefs_directory_names() {
        assert_eq!(
            title(Path::new("/g/Some Game.nro"), GameKind::Nro),
            "Some Game"
        );
        assert_eq!(title(Path::new("/g/Title/exefs"), GameKind::ExeFs), "Title");
        assert_eq!(title(Path::new("/g/Other"), GameKind::ExeFs), "Other");
    }

    #[test]
    fn pause_and_volume_reach_the_system() {
        let dir = temp_dir("run");
        let game = dir.join("Spin.nso");
        std::fs::write(&game, nso(&[0x14000000u32.to_le_bytes(), [0; 4]].concat())).unwrap();
        let host = Host::headless();
        let npad = host.npad.clone();
        let mut generation = None;
        let mut session = Session::boot(&game, LoadOptions::default(), None, |_, title| {
            generation = Some(title);
            (settings(), host)
        })
        .unwrap();
        assert_eq!(generation, Some(ConsoleGeneration::Switch1));
        assert_eq!(session.title, "Spin");
        assert!(Arc::ptr_eq(session.system().npad(), &npad));
        assert!(!session.is_paused());
        session.toggle_pause();
        assert!(session.is_paused());
        assert!(session.system().audio().is_paused());
        session.resume();
        assert!(!session.is_paused());
        assert!(!session.system().audio().is_paused());
        session.set_audio_volume(&AudioSettings {
            volume: 50,
            ..AudioSettings::default()
        });
        assert_eq!(session.system().audio().host_volume(), 0.5);
        assert_eq!(session.exit_reason(), None);
        drop(session);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_disk_shader_cache_opens_per_title_and_is_written_when_the_session_stops() {
        use lichii_vulkan::{Device, Instance, SurfaceSupport};

        let device = match Instance::new(false, SurfaceSupport::None)
            .and_then(|instance| Device::new(Arc::new(instance), &Default::default(), None))
        {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping: {error}");
                return;
            }
        };
        let dir = temp_dir("shader-cache");
        let game = dir.join("Spin.nso");
        std::fs::write(&game, nso(&[0x14000000u32.to_le_bytes(), [0; 4]].concat())).unwrap();
        let user = lichii_common::paths::UserPaths {
            config_dir: dir.join("config"),
            data_dir: dir.clone(),
            cache_dir: dir.join("cache"),
        };
        assert_eq!(crate::display::shader_cache_dir(&user, 0), None);
        let cache = crate::display::shader_cache_dir(&user, 0x0100_0000_0000_1000).unwrap();
        assert_eq!(cache, dir.join("shader").join("0100000000001000"));
        let boot = |disk_shader_cache: bool| {
            let mut settings = (*settings()).clone();
            settings.graphics.disk_shader_cache = disk_shader_cache;
            let renderer = crate::display::vulkan_renderer(
                device.clone(),
                Arc::default(),
                Arc::default(),
                ConsoleGeneration::Switch1,
                &settings.graphics,
                Some(cache.clone()),
            );
            let host = Host {
                renderer,
                ..Host::headless()
            };
            let session = Session::boot(&game, LoadOptions::default(), None, |_, _| {
                (Arc::new(settings), host)
            })
            .unwrap();
            drop(session);
        };

        boot(false);
        assert!(!cache.exists(), "a disabled disk cache writes nothing");
        boot(true);
        let mut files: Vec<_> = std::fs::read_dir(&cache)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        files.sort();
        assert_eq!(files, ["pipelines.bin", "shaders.bin"]);
        let written = std::fs::read(cache.join("pipelines.bin")).unwrap();
        boot(true);
        assert_eq!(
            std::fs::read(cache.join("pipelines.bin")).unwrap(),
            written,
            "a second boot with nothing new to cache keeps the file it found"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
