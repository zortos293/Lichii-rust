//! First-run setup: where the console keys and the firmware come from, and whether they work.
//! Lichii reads `prod.keys` and `title.keys` where the user keeps them; it never copies them and
//! never logs a key's value.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lichii_common::paths::UserPaths;
use lichii_common::settings::PathSettings;
use lichii_crypto::{KeyError, KeySet, LoadReport};
use lichii_fs::host::HostDir;
use lichii_fs::romfs::open_romfs;
use lichii_fs::{ContentIndex, FsError};
use lichii_fs::{DirectoryExt, FileExt};

/// The SystemVersion system data archive, which names the installed firmware.
const SYSTEM_VERSION: u64 = 0x0100_0000_0000_0809;
/// Master key revisions `prod.keys` can name (`master_key_00` to `master_key_1f`).
const MASTER_KEY_REVISIONS: usize = 0x20;

/// The keys folder in use: the configured override, else the profile's `keys` folder.
pub fn keys_dir(paths: &PathSettings, user: &UserPaths) -> PathBuf {
    paths.keys_dir.clone().unwrap_or_else(|| user.keys_dir())
}

/// The NAND folder in use: the configured override, else the profile's `nand` folder.
pub fn nand_dir(paths: &PathSettings, user: &UserPaths) -> PathBuf {
    paths.nand_dir.clone().unwrap_or_else(|| user.nand_dir())
}

/// What the keys folder holds.
pub struct KeyStatus {
    pub dir: PathBuf,
    /// The `prod.keys` report, `None` when the folder has no `prod.keys`.
    pub prod: Option<LoadReport>,
    /// The `title.keys` report, `None` when the folder has no `title.keys`.
    pub title: Option<LoadReport>,
    /// Why the folder could not be read at all.
    pub error: Option<String>,
    pub header_key: bool,
    /// The master key revisions present or derived, ascending.
    pub master_keys: Vec<usize>,
    pub keys: Arc<KeySet>,
}

impl KeyStatus {
    pub fn check(dir: PathBuf) -> Self {
        let (keys, reports, error) = match KeySet::load_dir(&dir) {
            Ok((keys, reports)) => (keys, reports, None),
            Err(e) => (KeySet::new(), Vec::new(), Some(e.to_string())),
        };
        let report = |file: &str| {
            reports
                .iter()
                .find(|(name, _)| name == file)
                .map(|(_, report)| report.clone())
        };
        Self {
            prod: report("prod.keys"),
            title: report("title.keys"),
            error,
            header_key: keys.header_key().is_ok(),
            master_keys: (0..MASTER_KEY_REVISIONS)
                .filter(|&revision| keys.master_key(revision).is_ok())
                .collect(),
            keys: Arc::new(keys),
            dir,
        }
    }

    /// Whether NSP, XCI and NCA titles can be opened: that needs the header key and the master
    /// keys. Loose ExeFS dumps and homebrew need neither.
    pub fn usable(&self) -> bool {
        self.header_key && !self.master_keys.is_empty()
    }
}

/// The firmware the NAND holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FirmwareStatus {
    Installed { version: String, title: String },
    Missing,
    Unreadable(String),
}

impl FirmwareStatus {
    /// Reads the SystemVersion archive from the NAND's registered system content. Opens the
    /// folder read-only, so checking a wrong folder changes nothing in it. Content that cannot
    /// be opened (a missing key, a damaged file) makes the firmware unreadable, not missing.
    pub fn check(nand: &Path, keys: &KeySet) -> Self {
        let registered = nand.join("system/Contents/registered");
        if !registered.is_dir() {
            return Self::Missing;
        }
        match read_system_version(&registered, keys) {
            Ok(Some(file)) => parse_system_version(&file)
                .unwrap_or_else(|| Self::Unreadable("SystemVersion is too short".into())),
            Ok(None) => Self::Missing,
            Err(e) => Self::Unreadable(e.to_string()),
        }
    }
}

fn read_system_version(registered: &Path, keys: &KeySet) -> Result<Option<Vec<u8>>, FsError> {
    let dir = HostDir::open(registered)?;
    let index = ContentIndex::from_registered(dir.as_ref(), &mut keys.clone())?;
    let Some(romfs) = index.data_romfs(SYSTEM_VERSION)? else {
        ContentIndex::from_dir(dir.as_ref(), &mut keys.clone())?;
        return Ok(None);
    };
    let root = open_romfs(&romfs)?;
    Ok(match root.open_file("file")? {
        Some(file) => Some(file.read_all()?),
        None => None,
    })
}

/// The display version (at 0x68) and display title (at 0x80) of a SystemVersion `file`.
fn parse_system_version(file: &[u8]) -> Option<FirmwareStatus> {
    let text = |range: std::ops::Range<usize>| {
        let bytes = file.get(range)?;
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        Some(String::from_utf8_lossy(&bytes[..end]).into_owned())
    };
    Some(FirmwareStatus::Installed {
        version: text(0x68..0x80)?,
        title: text(0x80..0x100)?,
    })
}

/// The missing key behind a failed boot, when keys are why it failed.
pub fn missing_key(error: &lichii_loader::LoaderError) -> Option<&KeyError> {
    use lichii_loader::LoaderError;
    match error {
        LoaderError::Key(key) | LoaderError::Fs(FsError::Key(key)) => Some(key),
        _ => None,
    }
}

/// `00–13` for a run of revisions, or each one when they have gaps.
pub fn revisions_label(revisions: &[usize]) -> String {
    let (Some(first), Some(last)) = (revisions.first(), revisions.last()) else {
        return "none".into();
    };
    if last - first + 1 == revisions.len() {
        return format!("{first:02x}\u{2013}{last:02x}");
    }
    revisions
        .iter()
        .map(|r| format!("{r:02x}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The keys and firmware window: first-run setup, and File → Keys and Firmware.
pub struct SetupWindow {
    keys: KeyStatus,
    firmware: FirmwareStatus,
}

/// What the window did this frame.
pub struct SetupOutput {
    /// The keys folder changed; save the settings.
    pub keys_changed: bool,
    /// The NAND folder changed; save the settings.
    pub nand_changed: bool,
    pub closed: bool,
}

impl SetupWindow {
    pub fn new(paths: &PathSettings, user: &UserPaths) -> Self {
        let keys = KeyStatus::check(keys_dir(paths, user));
        let firmware = FirmwareStatus::check(&nand_dir(paths, user), &keys.keys);
        Self { keys, firmware }
    }

    pub fn keys_usable(&self) -> bool {
        self.keys.usable()
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        paths: &mut PathSettings,
        user: &UserPaths,
    ) -> SetupOutput {
        let mut open = true;
        let mut keys_changed = false;
        let mut nand_changed = false;
        let mut close = false;
        egui::Window::new("Keys and Firmware")
            .open(&mut open)
            .collapsible(false)
            .default_width(560.0)
            .show(ctx, |ui| {
                ui.label(
                    "NSP, XCI and NCA titles are encrypted. Lichii decrypts them with keys dumped \
                     from your own console, read where you keep them: they are never copied or \
                     logged. Homebrew and extracted ExeFS folders need no keys.",
                );
                ui.add_space(8.0);
                ui.heading("Console keys");
                keys_changed = folder_row(
                    ui,
                    &self.keys.dir,
                    &mut paths.keys_dir,
                    |ui| {
                        let file = rfd::FileDialog::new()
                            .set_title("Choose prod.keys")
                            .add_filter("Console keys", &["keys"])
                            .pick_file();
                        ui.ctx().request_repaint();
                        file.and_then(|file| file.parent().map(Path::to_path_buf))
                    },
                    "Choose prod.keys…",
                );
                self.keys_ui(ui);
                ui.add_space(8.0);
                ui.heading("Firmware");
                let nand = nand_dir(paths, user);
                nand_changed = folder_row(
                    ui,
                    &nand,
                    &mut paths.nand_dir,
                    |_| {
                        rfd::FileDialog::new()
                            .set_title("Choose the NAND folder")
                            .pick_folder()
                    },
                    "Choose NAND folder…",
                );
                firmware_ui(ui, &self.firmware);
                ui.add_space(8.0);
                close = ui.button("Close").clicked();
            });
        if keys_changed || nand_changed {
            *self = Self::new(paths, user);
        }
        SetupOutput {
            keys_changed,
            nand_changed,
            closed: close || !open,
        }
    }

    fn keys_ui(&self, ui: &mut egui::Ui) {
        let k = &self.keys;
        if let Some(error) = &k.error {
            status(ui, Some(false), format!("Cannot read the folder: {error}"));
        }
        match &k.prod {
            Some(report) => status(ui, Some(true), format!("prod.keys: {} keys", report.loaded)),
            None => status(ui, Some(false), "prod.keys is not in this folder".into()),
        };
        status(
            ui,
            Some(k.header_key),
            if k.header_key {
                "Header key found"
            } else {
                "Header key missing"
            }
            .into(),
        );
        status(
            ui,
            Some(!k.master_keys.is_empty()),
            format!("Master keys: {}", revisions_label(&k.master_keys)),
        );
        match &k.title {
            Some(report) => status(
                ui,
                Some(true),
                format!("title.keys: {} title keys", report.loaded),
            ),
            None => status(
                ui,
                None,
                "No title.keys: only needed for titles whose ticket is not in the package".into(),
            ),
        };
        for (file, report) in [("prod.keys", &k.prod), ("title.keys", &k.title)] {
            for line in report.iter().flat_map(|r| &r.malformed) {
                status(
                    ui,
                    Some(false),
                    format!("{file} line {}: {} ({})", line.line, line.name, line.reason),
                );
            }
        }
    }
}

/// The folder in use, a button that picks another and one that returns to the default.
/// True when `choice` changed.
fn folder_row(
    ui: &mut egui::Ui,
    current: &Path,
    choice: &mut Option<PathBuf>,
    pick: impl FnOnce(&mut egui::Ui) -> Option<PathBuf>,
    label: &str,
) -> bool {
    let mut changed = false;
    ui.label(egui::RichText::new(current.display().to_string()).monospace());
    ui.horizontal(|ui| {
        if ui.button(label).clicked()
            && let Some(dir) = pick(ui)
        {
            *choice = Some(dir);
            changed = true;
        }
        if choice.is_some() && ui.button("Use the default folder").clicked() {
            *choice = None;
            changed = true;
        }
    });
    changed
}

fn firmware_ui(ui: &mut egui::Ui, firmware: &FirmwareStatus) {
    match firmware {
        FirmwareStatus::Installed { version, title } => {
            status(ui, Some(true), format!("Firmware {version}")).on_hover_text(title);
        }
        FirmwareStatus::Missing => {
            status(
                ui,
                None,
                "No firmware in this NAND (system/Contents/registered). Most titles run without \
                 it; system fonts and applets need it."
                    .into(),
            );
        }
        FirmwareStatus::Unreadable(error) => {
            status(
                ui,
                Some(false),
                format!("Cannot read the firmware: {error}"),
            );
        }
    }
}

/// One status line: `Some(true)` passed, `Some(false)` failed, `None` informational.
fn status(ui: &mut egui::Ui, ok: Option<bool>, text: String) -> egui::Response {
    let color = match ok {
        Some(true) => egui::Color32::from_rgb(90, 190, 110),
        Some(false) => egui::Color32::from_rgb(230, 90, 80),
        None => ui.visuals().weak_text_color(),
    };
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
        ui.painter().circle_filled(rect.center(), 4.0, color);
        ui.label(text);
    })
    .response
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("lichii-setup-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn hex(byte: u8, len: usize) -> String {
        format!("{byte:02x}").repeat(len)
    }

    #[test]
    fn an_empty_folder_has_no_usable_keys() {
        let dir = TempDir::new("empty");
        let status = KeyStatus::check(dir.0.clone());
        assert!(status.prod.is_none() && status.title.is_none() && status.error.is_none());
        assert!(!status.usable());
        assert_eq!(revisions_label(&status.master_keys), "none");
    }

    #[test]
    fn synthetic_keys_report_what_they_hold() {
        let dir = TempDir::new("keys");
        let prod = format!(
            "header_key = {}\nmaster_key_00 = {}\nmaster_key_01 = {}\nmaster_key_02 = nothex\n",
            hex(0x11, 32),
            hex(0x22, 16),
            hex(0x33, 16)
        );
        std::fs::write(dir.0.join("prod.keys"), prod).unwrap();
        std::fs::write(
            dir.0.join("title.keys"),
            format!("{} = {}\n", hex(0x01, 16), hex(0x44, 16)),
        )
        .unwrap();
        let status = KeyStatus::check(dir.0.clone());
        assert!(status.usable());
        assert_eq!(status.master_keys, [0, 1]);
        assert_eq!(revisions_label(&status.master_keys), "00\u{2013}01");
        let prod = status.prod.unwrap();
        assert_eq!(prod.malformed.len(), 1);
        assert_eq!(prod.malformed[0].name, "master_key_02");
        assert_eq!(status.title.unwrap().loaded, 1);
    }

    #[test]
    fn firmware_is_missing_without_registered_system_content() {
        let dir = TempDir::new("nand");
        assert_eq!(
            FirmwareStatus::check(&dir.0, &KeySet::new()),
            FirmwareStatus::Missing
        );
        assert!(
            !dir.0.join("system").exists(),
            "the check must not create folders"
        );
    }

    #[test]
    fn firmware_that_cannot_be_opened_is_unreadable() {
        let dir = TempDir::new("locked");
        let registered = dir.0.join("system/Contents/registered");
        std::fs::create_dir_all(&registered).unwrap();
        std::fs::write(registered.join("0123.nca"), vec![0u8; 0x4000]).unwrap();
        assert!(matches!(
            FirmwareStatus::check(&dir.0, &KeySet::new()),
            FirmwareStatus::Unreadable(_)
        ));
    }

    #[test]
    fn system_version_names_the_firmware() {
        let mut file = vec![0u8; 0x100];
        file[0x68..0x6E].copy_from_slice(b"20.1.0");
        file[0x80..0x85].copy_from_slice(b"NX 20");
        assert_eq!(
            parse_system_version(&file),
            Some(FirmwareStatus::Installed {
                version: "20.1.0".into(),
                title: "NX 20".into()
            })
        );
        assert_eq!(parse_system_version(&file[..0x90]), None);
    }

    #[test]
    fn gaps_in_revisions_are_listed() {
        assert_eq!(revisions_label(&[0, 1, 5]), "00, 01, 05");
    }
}
