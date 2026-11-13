//! File → Install Firmware, after Lichii's InstallFirmware: the NCAs of a dumped firmware, from a
//! folder or a ZIP, replace the NAND's system content (`system/Contents/registered`), which
//! holds the system archives Switch 1 titles need.

use std::ffi::OsStr;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crossbeam_channel::Receiver;
use lichii_crypto::KeySet;
use zip::ZipArchive;

use crate::setup::FirmwareStatus;

/// Where the firmware NCAs come from.
#[derive(Clone, Debug)]
pub enum FirmwareSource {
    /// The NCAs directly in this folder.
    Folder(PathBuf),
    /// The NCAs anywhere in this ZIP, since dumps are often zipped inside a subfolder.
    Zip(PathBuf),
}

#[derive(Debug)]
pub enum FirmwareError {
    /// The folder or ZIP could not be read.
    Source(String),
    NoNcas,
    FailedDelete,
    FailedCopy,
    Cancelled,
    /// The installed firmware has no readable SystemVersion archive.
    Unreadable,
}

impl fmt::Display for FirmwareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(e) => f.write_str(e),
            Self::NoNcas => f.write_str("Unable to locate potential firmware NCA files"),
            Self::FailedDelete => f.write_str(
                "Failed to delete one or more firmware files. The installed firmware was kept.",
            ),
            Self::FailedCopy => f.write_str(
                "One or more firmware files failed to copy into NAND. The installed firmware was \
                 kept.",
            ),
            Self::Cancelled => {
                f.write_str("Firmware installation cancelled. The installed firmware was kept.")
            }
            Self::Unreadable => f.write_str(
                "Firmware reported as present, but was unable to be read. Check for decryption \
                 keys and redump firmware if necessary.",
            ),
        }
    }
}

/// How far an install is, shared with the UI, which can also cancel it.
#[derive(Default)]
pub struct Progress {
    copied: AtomicU64,
    total: AtomicU64,
    cancel: AtomicBool,
}

impl Progress {
    /// The share of the NCA bytes copied so far.
    pub fn share(&self) -> f32 {
        let total = self.total.load(Ordering::Relaxed).max(1) as f32;
        (self.copied.load(Ordering::Relaxed) as f32 / total).min(1.0)
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

enum CopyError {
    Io(io::Error),
    Cancelled,
}

fn is_nca(name: &str) -> bool {
    Path::new(name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("nca"))
}

/// Copies one NCA to `dest`, counting its bytes and stopping when the install is cancelled.
fn copy_nca(reader: &mut dyn Read, dest: &Path, progress: &Progress) -> Result<(), CopyError> {
    let mut out = File::create(dest).map_err(CopyError::Io)?;
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        if progress.cancelled() {
            return Err(CopyError::Cancelled);
        }
        let n = reader.read(&mut buffer).map_err(CopyError::Io)?;
        if n == 0 {
            return out.flush().map_err(CopyError::Io);
        }
        out.write_all(&buffer[..n]).map_err(CopyError::Io)?;
        progress.copied.fetch_add(n as u64, Ordering::Relaxed);
    }
}

/// The install's error for a failed copy; any failure ends the install, which keeps the old
/// firmware.
fn copied(result: Result<(), CopyError>, name: &OsStr) -> Result<(), FirmwareError> {
    result.map_err(|e| match e {
        CopyError::Cancelled => FirmwareError::Cancelled,
        CopyError::Io(e) => {
            tracing::error!(file = %name.to_string_lossy(), "copying firmware into NAND: {e}");
            FirmwareError::FailedCopy
        }
    })
}

/// Whether `path` is `dir` or inside it, once both are resolved.
fn inside(path: &Path, dir: &Path) -> bool {
    match (path.canonicalize(), dir.canonicalize()) {
        (Ok(path), Ok(dir)) => path.starts_with(dir),
        _ => false,
    }
}

/// Replaces the NCAs in `registered` with those of `source`. The NCAs are copied into a staging
/// folder beside it first, so the installed firmware is kept unless every NCA was read and
/// written, and a cancel or a damaged source changes nothing.
pub fn install(
    source: &FirmwareSource,
    registered: &Path,
    progress: &Progress,
) -> Result<(), FirmwareError> {
    let (FirmwareSource::Folder(path) | FirmwareSource::Zip(path)) = source;
    if inside(path, registered) {
        return Err(FirmwareError::Source(
            "The firmware source is inside the NAND's system content folder, which the install \
             replaces. Choose a dump outside it."
                .into(),
        ));
    }
    let staging = registered.with_extension("installing");
    let fresh = || {
        if staging.exists() {
            std::fs::remove_dir_all(&staging)?;
        }
        std::fs::create_dir_all(&staging)
    };
    fresh().map_err(|e| {
        tracing::error!(dir = %staging.display(), "creating the firmware staging folder: {e}");
        FirmwareError::FailedCopy
    })?;
    if let Err(e) = copy_source(source, &staging, progress) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }
    swap(&staging, registered)
}

/// Moves `staging` into `registered`. The old firmware is set aside first and deleted only once
/// the new one is in place, so a failed move puts it back.
fn swap(staging: &Path, registered: &Path) -> Result<(), FirmwareError> {
    let backup = registered.with_extension("old");
    let set_aside = || {
        if backup.exists() {
            std::fs::remove_dir_all(&backup)?;
        }
        if registered.exists() {
            std::fs::rename(registered, &backup)?;
        }
        Ok::<_, io::Error>(())
    };
    if let Err(e) = set_aside() {
        tracing::error!(dir = %registered.display(), "setting the old firmware aside: {e}");
        let _ = std::fs::remove_dir_all(staging);
        return Err(FirmwareError::FailedDelete);
    }
    if let Err(e) = std::fs::rename(staging, registered) {
        tracing::error!(dir = %registered.display(), "moving the new firmware in place: {e}");
        let _ = std::fs::remove_dir_all(staging);
        if backup.exists()
            && let Err(e) = std::fs::rename(&backup, registered)
        {
            tracing::error!(dir = %backup.display(), "putting the old firmware back: {e}");
            return Err(FirmwareError::Source(format!(
                "The new firmware could not be moved into NAND, and the old firmware could not be \
                 put back. It is in {}; rename that folder to \"registered\" to restore it.",
                backup.display()
            )));
        }
        return Err(FirmwareError::FailedCopy);
    }
    if backup.exists()
        && let Err(e) = std::fs::remove_dir_all(&backup)
    {
        tracing::warn!(dir = %backup.display(), "removing the old firmware: {e}");
    }
    Ok(())
}

/// Copies the NCAs of `source` into `dest`.
fn copy_source(
    source: &FirmwareSource,
    dest: &Path,
    progress: &Progress,
) -> Result<(), FirmwareError> {
    match source {
        FirmwareSource::Folder(dir) => {
            let read =
                |e: io::Error| FirmwareError::Source(format!("Cannot read {}: {e}", dir.display()));
            let mut ncas = Vec::new();
            for entry in std::fs::read_dir(dir).map_err(read)? {
                let entry = entry.map_err(read)?;
                let meta = entry.metadata().map_err(read)?;
                if meta.is_file() && is_nca(&entry.file_name().to_string_lossy()) {
                    ncas.push((entry.path(), entry.file_name(), meta.len()));
                }
            }
            if ncas.is_empty() {
                return Err(FirmwareError::NoNcas);
            }
            progress.total.store(
                ncas.iter().map(|(_, _, size)| size).sum(),
                Ordering::Relaxed,
            );
            for (path, name, _) in ncas {
                let result = File::open(&path)
                    .map_err(CopyError::Io)
                    .and_then(|mut file| copy_nca(&mut file, &dest.join(&name), progress));
                copied(result, &name)?;
            }
        }
        FirmwareSource::Zip(path) => {
            let open = |e: &dyn fmt::Display| {
                FirmwareError::Source(format!("Cannot open {}: {e}", path.display()))
            };
            let file = File::open(path).map_err(|e| open(&e))?;
            let mut archive = ZipArchive::new(file).map_err(|e| open(&e))?;
            let mut ncas = Vec::new();
            for index in 0..archive.len() {
                let entry = archive.by_index_raw(index).map_err(|e| open(&e))?;
                let name = entry.name().map_err(|e| open(&e))?;
                let name = Path::new(name.as_ref())
                    .file_name()
                    .map(|n| n.to_os_string());
                if let Some(name) = name
                    && entry.is_file()
                    && is_nca(&name.to_string_lossy())
                {
                    ncas.push((index, name, entry.size()));
                }
            }
            if ncas.is_empty() {
                return Err(FirmwareError::NoNcas);
            }
            progress.total.store(
                ncas.iter().map(|(_, _, size)| size).sum(),
                Ordering::Relaxed,
            );
            for (index, name, _) in ncas {
                let result = match archive.by_index(index) {
                    Ok(mut entry) => copy_nca(&mut entry, &dest.join(&name), progress),
                    Err(e) => Err(CopyError::Io(io::Error::other(e))),
                };
                copied(result, &name)?;
            }
        }
    }
    Ok(())
}

/// Installs firmware into `nand` on a background thread, then reads back the version it installed.
pub struct FirmwareJob {
    progress: Arc<Progress>,
    done: Receiver<Result<String, FirmwareError>>,
}

impl FirmwareJob {
    /// Starts installing `source`; `wake` runs when the install has finished.
    pub fn start(
        source: FirmwareSource,
        nand: PathBuf,
        keys: Arc<KeySet>,
        wake: impl FnOnce() + Send + 'static,
    ) -> Self {
        let progress = Arc::new(Progress::default());
        let (tx, done) = crossbeam_channel::bounded(1);
        let job_progress = progress.clone();
        std::thread::Builder::new()
            .name("install firmware".into())
            .spawn(move || {
                tracing::info!(?source, "installing firmware");
                let registered = nand.join("system/Contents/registered");
                let result = install(&source, &registered, &job_progress).and_then(|()| {
                    match FirmwareStatus::check(&nand, &keys) {
                        FirmwareStatus::Installed { version, .. } => Ok(version),
                        status => {
                            tracing::error!(?status, "the installed firmware cannot be read");
                            Err(FirmwareError::Unreadable)
                        }
                    }
                });
                if let Err(e) = &result {
                    tracing::error!("firmware install failed: {e}");
                }
                let _ = tx.send(result);
                wake();
            })
            .expect("spawning the firmware installer");
        Self { progress, done }
    }

    pub fn progress(&self) -> &Progress {
        &self.progress
    }

    pub fn poll(&self) -> Option<Result<String, FirmwareError>> {
        self.done.try_recv().ok()
    }
}

/// The result dialog's title and message, worded like Lichii's.
pub fn result_message(result: &Result<String, FirmwareError>) -> (String, String) {
    match result {
        Ok(version) => (
            "Firmware Install Succeeded".into(),
            format!("Successfully installed firmware version {version}"),
        ),
        Err(e) => ("Firmware Install Failed".into(), e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use zip::write::{SimpleFileOptions, ZipWriter};

    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("lichii-firmware-{name}-{}", std::process::id()));
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

    fn names(dir: &Path) -> Vec<String> {
        let mut names: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    fn old_firmware(registered: &Path) {
        std::fs::create_dir_all(registered).unwrap();
        std::fs::write(registered.join("old.nca"), b"old").unwrap();
    }

    #[test]
    fn a_folder_replaces_the_registered_ncas() {
        let dir = TempDir::new("folder");
        let source = dir.0.join("dump");
        std::fs::create_dir_all(source.join("nested")).unwrap();
        std::fs::write(source.join("aa.nca"), vec![1u8; 3000]).unwrap();
        std::fs::write(source.join("bb.NCA"), vec![2u8; 1000]).unwrap();
        std::fs::write(source.join("readme.txt"), b"not firmware").unwrap();
        std::fs::write(source.join("nested/cc.nca"), b"folders are not searched").unwrap();
        let registered = dir.0.join("nand/system/Contents/registered");
        old_firmware(&registered);
        let progress = Progress::default();
        install(&FirmwareSource::Folder(source), &registered, &progress).unwrap();
        assert_eq!(names(&registered), ["aa.nca", "bb.NCA"]);
        assert_eq!(
            names(registered.parent().unwrap()),
            ["registered"],
            "no staging or backup left"
        );
        assert_eq!(
            std::fs::read(registered.join("aa.nca")).unwrap(),
            vec![1u8; 3000]
        );
        assert_eq!(progress.share(), 1.0);
    }

    #[test]
    fn a_zip_installs_its_ncas_from_any_folder() {
        let dir = TempDir::new("zip");
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        for (name, data) in [
            ("Firmware 20.1.0/aa.nca", vec![7u8; 5000]),
            ("Firmware 20.1.0/sub/bb.nca", vec![8u8; 10]),
            ("Firmware 20.1.0/info.txt", b"skip".to_vec()),
        ] {
            writer
                .start_file(name, SimpleFileOptions::default())
                .unwrap();
            writer.write_all(&data).unwrap();
        }
        writer
            .add_directory("Firmware 20.1.0/dir.nca/", SimpleFileOptions::default())
            .unwrap();
        let zip = dir.0.join("firmware.zip");
        std::fs::write(&zip, writer.finish().unwrap().into_inner()).unwrap();
        let registered = dir.0.join("registered");
        install(&FirmwareSource::Zip(zip), &registered, &Progress::default()).unwrap();
        assert_eq!(names(&registered), ["aa.nca", "bb.nca"]);
        assert_eq!(
            std::fs::read(registered.join("aa.nca")).unwrap(),
            vec![7u8; 5000]
        );
    }

    #[test]
    fn a_source_without_ncas_keeps_the_installed_firmware() {
        let dir = TempDir::new("empty");
        let source = dir.0.join("dump");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("readme.txt"), b"no firmware").unwrap();
        let registered = dir.0.join("registered");
        old_firmware(&registered);
        let result = install(
            &FirmwareSource::Folder(source),
            &registered,
            &Progress::default(),
        );
        assert!(matches!(result, Err(FirmwareError::NoNcas)));
        assert_eq!(names(&registered), ["old.nca"]);
        let broken = dir.0.join("broken.zip");
        std::fs::write(&broken, b"not a zip").unwrap();
        let result = install(
            &FirmwareSource::Zip(broken),
            &registered,
            &Progress::default(),
        );
        assert!(
            matches!(result, Err(FirmwareError::Source(_))),
            "{result:?}"
        );
        assert_eq!(names(&registered), ["old.nca"]);
    }

    #[test]
    fn a_source_inside_the_nand_is_refused() {
        let dir = TempDir::new("inside");
        let registered = dir.0.join("registered");
        old_firmware(&registered);
        let dump = registered.join("dump");
        std::fs::create_dir_all(&dump).unwrap();
        std::fs::write(dump.join("aa.nca"), b"new").unwrap();
        for source in [
            FirmwareSource::Folder(registered.clone()),
            FirmwareSource::Folder(dump.clone()),
        ] {
            let result = install(&source, &registered, &Progress::default());
            assert!(
                matches!(result, Err(FirmwareError::Source(_))),
                "{result:?}"
            );
        }
        assert_eq!(names(&registered), ["dump", "old.nca"]);
        assert!(dump.join("aa.nca").is_file());
    }

    #[test]
    fn a_damaged_zip_entry_keeps_the_installed_firmware() {
        let dir = TempDir::new("crc");
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        let stored =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        writer.start_file("aa.nca", stored).unwrap();
        writer.write_all(&[0x5a; 64]).unwrap();
        let mut bytes = writer.finish().unwrap().into_inner();
        let payload = bytes.windows(64).position(|w| w == [0x5a; 64]).unwrap();
        bytes[payload + 10] ^= 0xff;
        let zip = dir.0.join("firmware.zip");
        std::fs::write(&zip, bytes).unwrap();
        let registered = dir.0.join("registered");
        old_firmware(&registered);
        let result = install(&FirmwareSource::Zip(zip), &registered, &Progress::default());
        assert!(
            matches!(result, Err(FirmwareError::FailedCopy)),
            "{result:?}"
        );
        assert_eq!(names(&registered), ["old.nca"]);
        assert!(!dir.0.join("registered.installing").exists());
    }

    #[test]
    fn cancelling_stops_the_copy() {
        let dir = TempDir::new("cancel");
        let source = dir.0.join("dump");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("aa.nca"), vec![1u8; 100]).unwrap();
        let registered = dir.0.join("registered");
        old_firmware(&registered);
        let progress = Progress::default();
        progress.cancel();
        let result = install(&FirmwareSource::Folder(source), &registered, &progress);
        assert!(matches!(result, Err(FirmwareError::Cancelled)));
        assert_eq!(
            names(&registered),
            ["old.nca"],
            "a cancel keeps the old firmware"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .starts_with("Firmware installation cancelled")
        );
    }

    #[test]
    fn the_job_reports_firmware_it_cannot_read_back() {
        let dir = TempDir::new("job");
        let source = dir.0.join("dump");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("aa.nca"), vec![0u8; 0x4000]).unwrap();
        let (tx, rx) = crossbeam_channel::bounded(1);
        let job = FirmwareJob::start(
            FirmwareSource::Folder(source),
            dir.0.join("nand"),
            Arc::new(KeySet::new()),
            move || tx.send(()).unwrap(),
        );
        rx.recv().unwrap();
        let result = job.poll().expect("the job finished");
        assert!(
            matches!(result, Err(FirmwareError::Unreadable)),
            "{result:?}"
        );
        assert_eq!(result_message(&result).0, "Firmware Install Failed");
        assert!(
            dir.0
                .join("nand/system/Contents/registered/aa.nca")
                .is_file()
        );
    }

    #[test]
    fn success_names_the_version() {
        assert_eq!(
            result_message(&Ok("20.1.0".into())),
            (
                "Firmware Install Succeeded".into(),
                "Successfully installed firmware version 20.1.0".into()
            )
        );
    }
}
