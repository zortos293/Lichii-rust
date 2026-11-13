//! One emulation session: a booted [`System`] running the title, owned by the app and stopped when
//! dropped.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lichii_common::ConsoleGeneration;
use lichii_common::settings::{AudioSettings, ControlSettings, Settings};
use lichii_core::{ExitReason, Handover, Host, PROGRAM_INDEX_MASK, System, SystemError};
use lichii_crypto::KeyError;
use lichii_loader::LoadOptions;

use crate::applets::{AppletDialogs, ControlsChange};
use crate::game_list::GameKind;

#[derive(Debug)]
pub enum BootError {
    NotFound(PathBuf),
    Unsupported(PathBuf),
    /// The title asked to start a program its file does not hold, or that its format cannot
    /// select.
    NoProgram(u8),
    System(SystemError),
}

impl std::fmt::Display for BootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(path) => write!(f, "{} does not exist", path.display()),
            Self::Unsupported(path) => write!(
                f,
                "{} is not an NSP, XCI, NCA, NUP, NRO, NSO or ExeFS directory",
                path.display()
            ),
            Self::NoProgram(index) => write!(f, "the title has no program {index} to start"),
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
    /// The title's name: its NACP name in the configured language, else the name the game list
    /// shows (a loose dump's game folder), else the file name.
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
        Self::start(path, options, None, applets, configure)
    }

    /// Stops this session and boots program `program_index` of the same title in its place,
    /// handing it the user channel this title left, as its `ExecuteProgram` asked.
    pub fn execute_program(
        self,
        program_index: u8,
        options: LoadOptions,
        applets: Option<AppletDialogs>,
        configure: impl FnOnce(u64, ConsoleGeneration) -> (Arc<Settings>, Host),
    ) -> Result<Self, BootError> {
        let path = self.path.clone();
        let handover = self.system.take_handover();
        drop(self);
        let options = LoadOptions {
            program_index,
            ..options
        };
        Self::start(&path, options, Some(handover), applets, configure)
    }

    fn start(
        path: &Path,
        options: LoadOptions,
        handover: Option<Handover>,
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
        let title = image
            .nacp
            .as_ref()
            .and_then(|nacp| nacp.preferred_title(options.language))
            .map(|entry| entry.name.clone())
            .filter(|name| !name.is_empty())
            .or_else(|| {
                lichii_loader::scan(path, &options)
                    .ok()
                    .map(|info| info.name)
                    .filter(|name| !name.is_empty())
            })
            .unwrap_or_else(|| file_title(path, kind));
        if handover.is_some() && program_id & PROGRAM_INDEX_MASK != u64::from(options.program_index)
        {
            return Err(BootError::NoProgram(options.program_index));
        }
        let (settings, host) = configure(program_id, image.generation);
        let mut system = System::from_image(settings, image, host)?;
        if let Some(applets) = &applets {
            system.services().set_applet_frontend(applets.frontend());
        }
        if let Some(handover) = handover {
            system.set_handover(handover);
        }
        system.run()?;
        Ok(Self {
            path: path.to_path_buf(),
            title,
            program_id,
            applets,
            system,
        })
    }

    /// The title's name with its program id, as the window title shows it.
    pub fn display_title(&self) -> String {
        match self.program_id {
            0 => self.title.clone(),
            id => format!("{} [{id:016X}]", self.title),
        }
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
    ) -> Option<ControlsChange> {
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
fn file_title(path: &Path, kind: GameKind) -> String {
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
    fn a_nup_container_boots_its_exefs() {
        use lichii_loader::nup::{META_TITLE_NAME, PackageKind};
        use lichii_loader::testing::{
            AcidLayout, Codec, NpdmSpec, NupPackage, build_npdm, build_nup, svc_mask, thread_info,
        };

        let program_id = 0x0100_0000_0001_0000;
        let mut capabilities = vec![thread_info(63, 16, 0, 3)];
        capabilities.extend((0..8).map(|index| svc_mask(index, 0xff_ffff)));
        let npdm = build_npdm(&NpdmSpec {
            program_id,
            layout: AcidLayout::Switch1,
            capabilities,
            ..NpdmSpec::default()
        });
        let main = nso(&[0x14000000u32.to_le_bytes(), [0; 4]].concat());
        let dir = temp_dir("nup");
        let game = dir.join("Kart.nup");
        std::fs::write(
            &game,
            build_nup(&[NupPackage::new(PackageKind::Base, program_id, 0)
                .file("exefs/main.npdm", npdm, Codec::Zstd)
                .file("exefs/main", main, Codec::Zstd)
                .file("romfs/data.bin", vec![7; 0x300], Codec::Raw)
                .meta(META_TITLE_NAME, "Synthetic Kart")]),
        )
        .unwrap();

        let mut booted_as = None;
        let session = Session::boot(&game, LoadOptions::default(), None, |id, generation| {
            booted_as = Some((id, generation));
            (settings(), Host::headless())
        })
        .unwrap();
        assert_eq!(booted_as, Some((program_id, ConsoleGeneration::Switch1)));
        assert_eq!(session.program_id, program_id);
        assert_eq!(session.title, "Synthetic Kart");
        assert_eq!(session.display_title(), "Synthetic Kart [0100000000010000]");
        assert_eq!(session.exit_reason(), None);
        drop(session);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_loose_dump_is_named_by_its_nacp_else_its_game_folder() {
        use lichii_fs::nacp::NacpLanguage;
        use lichii_loader::testing::{AcidLayout, NpdmSpec, build_npdm, svc_mask, thread_info};

        let program_id = 0x0400_0000_0003_0000;
        let mut capabilities = vec![thread_info(63, 16, 0, 3)];
        capabilities.extend((0..8).map(|index| svc_mask(index, 0xff_ffff)));
        let npdm = build_npdm(&NpdmSpec {
            program_id,
            layout: AcidLayout::Switch2,
            capabilities,
            ..NpdmSpec::default()
        });
        let dir = temp_dir("loose");
        let game = dir.join("SyntheticTennis_v1.1.1");
        let exefs = game.join("base/exefs");
        std::fs::create_dir_all(&exefs).unwrap();
        std::fs::write(exefs.join("main.npdm"), npdm).unwrap();
        std::fs::write(
            exefs.join("main"),
            nso(&[0x14000000u32.to_le_bytes(), [0; 4]].concat()),
        )
        .unwrap();
        let boot = |language| {
            let options = LoadOptions {
                language,
                ..LoadOptions::default()
            };
            Session::boot(&exefs, options, None, |_, _| (settings(), Host::headless())).unwrap()
        };

        let session = boot(NacpLanguage::AmericanEnglish);
        assert_eq!(
            session.title, "SyntheticTennis_v1.1.1",
            "not the storage folder"
        );
        assert_eq!(
            session.display_title(),
            "SyntheticTennis_v1.1.1 [0400000000030000]"
        );
        drop(session);

        let mut nacp = vec![0u8; lichii_fs::nacp::NACP_SIZE];
        for (language, name) in [
            (NacpLanguage::AmericanEnglish, "Synthetic Tennis"),
            (NacpLanguage::Japanese, "Gousei Tenisu"),
        ] {
            let at = language as usize * 0x300;
            nacp[at..at + name.len()].copy_from_slice(name.as_bytes());
        }
        std::fs::write(game.join("base/control.nacp"), nacp).unwrap();
        assert_eq!(
            boot(NacpLanguage::AmericanEnglish).title,
            "Synthetic Tennis"
        );
        assert_eq!(boot(NacpLanguage::Japanese).title, "Gousei Tenisu");
        assert_eq!(
            boot(NacpLanguage::German).display_title(),
            "Synthetic Tennis [0400000000030000]",
            "a language without a name falls back to American English"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn execute_program_restarts_at_the_new_program_with_the_user_channel() {
        use lichii_kernel::testing::{IpcClient, Request};
        use lichii_loader::testing::{
            AcidLayout, NpdmSpec, build_npdm, multi_program_nsp, pfs0, svc_mask, thread_info,
        };

        const APPLICATION: u64 = 0x0100_0000_0002_0000;
        const CURRENT_PROCESS: u32 = 0xffff_8001;
        let exefs = |index: u64| {
            let mut capabilities = vec![thread_info(63, 16, 0, 3)];
            capabilities.extend((0..8).map(|index| svc_mask(index, 0xff_ffff)));
            let npdm = build_npdm(&NpdmSpec {
                program_id: APPLICATION + index,
                layout: AcidLayout::Switch1,
                capabilities,
                ..NpdmSpec::default()
            });
            let main = nso(&[0x14000000u32.to_le_bytes(), [0; 4]].concat());
            pfs0(&[("main", main), ("main.npdm", npdm)])
        };
        let dir = temp_dir("execute-program");
        let game = dir.join("Multi.nsp");
        std::fs::write(&game, multi_program_nsp(APPLICATION, &[exefs(0), exefs(1)])).unwrap();
        let configure = |_, _| (settings(), Host::headless());

        let application = |session: &Session| {
            let system = session.system();
            let client = IpcClient::new(system.kernel(), system.process());
            let service = client.connect("appletOE").unwrap();
            let open = Request::new(0)
                .send_pid()
                .arg(0u64)
                .copy_handle(CURRENT_PROCESS);
            let proxy = client.send(service, &open).unwrap().move_handles[0];
            let interface = |command| {
                client
                    .send(proxy, &Request::new(command))
                    .unwrap()
                    .move_handles[0]
            };
            let (functions, creator) = (interface(20), interface(11));
            (client, functions, creator)
        };

        let session = Session::boot(&game, LoadOptions::default(), None, configure).unwrap();
        assert_eq!(session.program_id, APPLICATION);
        let (client, functions, creator) = application(&session);
        let storage = client
            .send(creator, &Request::new(10).arg(8i64))
            .unwrap()
            .move_handles[0];
        let accessor = client.send(storage, &Request::new(0)).unwrap().move_handles[0];
        let process = session.system().process();
        let input = process.page_table().set_heap_size(0x20_0000).unwrap();
        process.memory().write_bytes(input, b"handover").unwrap();
        client
            .send(accessor, &Request::new(10).arg(0i64).buffer_a(input, 8))
            .unwrap();
        let push = Request::new(122).move_handle(storage);
        assert_eq!(client.send(functions, &push).unwrap().result.0, 0);
        let execute = Request::new(120).arg(0u32).arg(1u64);
        assert_eq!(client.send(functions, &execute).unwrap().result.0, 0);
        assert_eq!(
            session.exit_reason(),
            Some(ExitReason::ExecuteProgram { program_index: 1 })
        );
        drop(client);

        let session = session
            .execute_program(1, LoadOptions::default(), None, configure)
            .unwrap();
        assert_eq!(session.program_id, APPLICATION + 1);
        assert_eq!(session.exit_reason(), None);
        let (client, functions, _) = application(&session);
        let popped = client.send(functions, &Request::new(1).arg(1u32)).unwrap();
        assert_eq!(popped.result.0, 0, "the user channel came along");
        let accessor = client
            .send(popped.move_handles[0], &Request::new(0))
            .unwrap()
            .move_handles[0];
        let process = session.system().process();
        let output = process.page_table().set_heap_size(0x20_0000).unwrap();
        client
            .send(accessor, &Request::new(11).arg(0i64).buffer_b(output, 8))
            .unwrap();
        let mut read = [0; 8];
        process.memory().read_bytes(output, &mut read).unwrap();
        assert_eq!(&read, b"handover");
        let mut previous = client.send(functions, &Request::new(123)).unwrap();
        assert_eq!(previous.pop::<i32>(), 0, "started by program 0");
        drop(client);
        drop(session);

        let single = dir.join("Single.nso");
        std::fs::write(
            &single,
            nso(&[0x14000000u32.to_le_bytes(), [0; 4]].concat()),
        )
        .unwrap();
        let session = Session::boot(&single, LoadOptions::default(), None, configure).unwrap();
        let (client, functions, _) = application(&session);
        assert_eq!(client.send(functions, &execute).unwrap().result.0, 0);
        drop(client);
        assert!(matches!(
            session.execute_program(1, LoadOptions::default(), None, configure),
            Err(BootError::NoProgram(1))
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn titles_come_from_file_and_exefs_directory_names() {
        assert_eq!(
            file_title(Path::new("/g/Some Game.nro"), GameKind::Nro),
            "Some Game"
        );
        assert_eq!(
            file_title(Path::new("/g/Title/exefs"), GameKind::ExeFs),
            "Title"
        );
        assert_eq!(file_title(Path::new("/g/Other"), GameKind::ExeFs), "Other");
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
