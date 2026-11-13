//! File → Install Files to NAND: an NSP's updates and DLC copied into the NAND's user content
//! (`user/Contents/registered`) by [`install_registered`], where boots and the game list pick
//! them up. As in Lichii, base games are not installed; they run from the game list.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crossbeam_channel::Receiver;
use lichii_crypto::KeySet;
use lichii_fs::cnmt::ContentMetaType;
use lichii_fs::content::{InstallOutcome, InstalledTitle, install_registered};
use lichii_fs::host::HostFile;
use lichii_fs::{FsError, Nsp};

#[derive(Debug)]
pub enum InstallError {
    Fs(FsError),
    /// The package holds a base game.
    BaseGame,
    /// The package holds no update or DLC.
    NothingToInstall,
}

impl fmt::Display for InstallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fs(e) => e.fmt(f),
            Self::BaseGame => f.write_str(
                "it is a base game; to avoid conflicts, base games run from the game list",
            ),
            Self::NothingToInstall => f.write_str("it holds no update or DLC"),
        }
    }
}

impl From<FsError> for InstallError {
    fn from(e: FsError) -> Self {
        Self::Fs(e)
    }
}

/// Installs the updates and DLC of the NSP at `package` into `registered`; `copied` counts the
/// bytes written.
pub fn install(
    package: &Path,
    registered: &Path,
    keys: &KeySet,
    copied: &AtomicU64,
) -> Result<Vec<InstalledTitle>, InstallError> {
    let mut keys = keys.clone();
    let file = HostFile::open(package).map_err(FsError::from)?;
    let nsp = Nsp::open(&file, &mut keys)?;
    if nsp
        .index()
        .titles_of(ContentMetaType::Application)
        .next()
        .is_some()
    {
        return Err(InstallError::BaseGame);
    }
    let installed = install_registered(nsp.partition(), nsp.index(), registered, &mut keys, |n| {
        copied.fetch_add(n, Ordering::Relaxed);
    })?;
    if installed.is_empty() {
        return Err(InstallError::NothingToInstall);
    }
    Ok(installed)
}

/// One package's result.
pub type PackageResult = (PathBuf, Result<Vec<InstalledTitle>, InstallError>);

/// Installs packages one after another on a background thread.
pub struct InstallJob {
    files: Vec<PathBuf>,
    total: u64,
    copied: Arc<AtomicU64>,
    current: Arc<AtomicUsize>,
    done: Receiver<Vec<PackageResult>>,
}

impl InstallJob {
    /// Starts installing `files` (at least one) into `registered`; `wake` runs when they are all
    /// done.
    pub fn start(
        files: Vec<PathBuf>,
        registered: PathBuf,
        keys: Arc<KeySet>,
        wake: impl FnOnce() + Send + 'static,
    ) -> Self {
        let total = files
            .iter()
            .filter_map(|f| std::fs::metadata(f).ok())
            .map(|m| m.len())
            .sum();
        let copied = Arc::new(AtomicU64::new(0));
        let current = Arc::new(AtomicUsize::new(0));
        let (tx, done) = crossbeam_channel::bounded(1);
        let (job_files, job_copied, job_current) = (files.clone(), copied.clone(), current.clone());
        std::thread::Builder::new()
            .name("install to NAND".into())
            .spawn(move || {
                let results = job_files
                    .into_iter()
                    .enumerate()
                    .map(|(i, file)| {
                        job_current.store(i, Ordering::Relaxed);
                        let result = install(&file, &registered, &keys, &job_copied);
                        (file, result)
                    })
                    .collect();
                let _ = tx.send(results);
                wake();
            })
            .expect("spawning the installer");
        Self {
            files,
            total,
            copied,
            current,
            done,
        }
    }

    /// The package being installed, how many there are, and the share of bytes copied so far.
    pub fn progress(&self) -> (usize, usize, &Path, f32) {
        let current = self
            .current
            .load(Ordering::Relaxed)
            .min(self.files.len() - 1);
        let copied = self.copied.load(Ordering::Relaxed) as f32;
        let share = (copied / self.total.max(1) as f32).min(1.0);
        (current, self.files.len(), &self.files[current], share)
    }

    pub fn poll(&self) -> Option<Vec<PackageResult>> {
        self.done.try_recv().ok()
    }
}

/// The Install Results message, counted like Lichii's.
pub fn summary(results: &[PackageResult]) -> String {
    let titles: Vec<&InstalledTitle> = results
        .iter()
        .filter_map(|(_, r)| r.as_ref().ok())
        .flatten()
        .collect();
    let count =
        |want: fn(&InstallOutcome) -> bool| titles.iter().filter(|t| want(&t.outcome)).count();
    let mut lines = Vec::new();
    let new = count(|o| *o == InstallOutcome::New);
    if new > 0 {
        lines.push(format!("{new} title(s) were newly installed."));
    }
    let overwritten = count(|o| matches!(o, InstallOutcome::Overwrote { .. }));
    if overwritten > 0 {
        lines.push(format!("{overwritten} title(s) were overwritten."));
    }
    let kept = count(|o| matches!(o, InstallOutcome::KeptNewer { .. }));
    if kept > 0 {
        lines.push(format!(
            "{kept} title(s) were skipped because a newer version is installed."
        ));
    }
    for (file, result) in results {
        if let Err(e) = result {
            let name = file.file_name().map_or_else(
                || file.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            lines.push(format!("{name} was not installed: {e}."));
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_packages_are_reported_not_installed() {
        let dir = std::env::temp_dir().join(format!("lichii-install-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let junk = dir.join("junk.nsp");
        std::fs::write(&junk, vec![0u8; 0x100]).unwrap();
        let registered = dir.join("registered");
        let result = install(&junk, &registered, &KeySet::new(), &AtomicU64::new(0));
        assert!(matches!(result, Err(InstallError::Fs(_))));
        assert!(
            !registered.exists(),
            "nothing is written for a package that does not open"
        );
        let text = summary(&[(junk, result)]);
        assert!(text.starts_with("junk.nsp was not installed: "), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn results_are_counted_like_lichii() {
        let title = |outcome| InstalledTitle {
            title_id: 1,
            meta_type: ContentMetaType::Patch,
            version: 0,
            outcome,
        };
        let results = vec![
            (
                PathBuf::from("a.nsp"),
                Ok(vec![
                    title(InstallOutcome::New),
                    title(InstallOutcome::Overwrote { old_version: 0 }),
                ]),
            ),
            (
                PathBuf::from("b.nsp"),
                Ok(vec![title(InstallOutcome::KeptNewer { installed: 2 })]),
            ),
            (PathBuf::from("/g/base.nsp"), Err(InstallError::BaseGame)),
        ];
        assert_eq!(
            summary(&results),
            "1 title(s) were newly installed.\n\
             1 title(s) were overwritten.\n\
             1 title(s) were skipped because a newer version is installed.\n\
             base.nsp was not installed: it is a base game; to avoid conflicts, base games run from the game list."
        );
    }
}
