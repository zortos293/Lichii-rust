//! The per-title file of translated shaders (`lichii_shader::cache`): every entry loaded at
//! start-up and indexed by its probe key, and the entries translated since, appended in a
//! batch.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use lichii_shader::ShaderModule;
use lichii_shader::cache::{self, CacheEntry, HEADER_LEN};

use super::FxBuildHasher;

pub(crate) struct ShaderDisk {
    path: PathBuf,
    entries: HashMap<u64, Vec<CacheEntry>, FxBuildHasher>,
    /// Encoded entries not written yet.
    pending: Vec<u8>,
}

impl ShaderDisk {
    /// Loads the file at `path`. A stale file (another translator or format) is replaced by an
    /// empty one, and a damaged tail is cut so later appends follow the last good entry.
    pub(crate) fn open(path: PathBuf) -> io::Result<Self> {
        let file = match fs::read(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(error),
        };
        let mut entries: HashMap<u64, Vec<CacheEntry>, FxBuildHasher> = HashMap::default();
        match cache::records(&file) {
            Ok(records) => {
                let mut decoded = cache::entries(records);
                for entry in decoded.by_ref() {
                    entries.entry(entry.probe_key()).or_default().push(entry);
                }
                let valid = HEADER_LEN + decoded.valid_len();
                if valid < file.len() {
                    tracing::warn!(
                        "{}: dropping {} damaged bytes",
                        path.display(),
                        file.len() - valid
                    );
                    OpenOptions::new()
                        .write(true)
                        .open(&path)?
                        .set_len(valid as u64)?;
                }
            }
            Err(_) => {
                if !file.is_empty() {
                    tracing::info!("{}: stale shader cache, starting a new one", path.display());
                }
                fs::write(&path, cache::header())?;
            }
        }
        let count: usize = entries.values().map(Vec::len).sum();
        if count > 0 {
            tracing::info!("shader cache: {count} translated programs");
        }
        Ok(Self {
            path,
            entries,
            pending: Vec::new(),
        })
    }

    /// Entries translated from live state with this probe key: only candidates, which
    /// [`CacheEntry::matches`] decides.
    pub(crate) fn candidates(&self, probe_key: u64) -> &[CacheEntry] {
        self.entries.get(&probe_key).map_or(&[], Vec::as_slice)
    }

    /// Every module the file holds, by module hash, to build pipelines ahead of their draws.
    pub(crate) fn modules(&self) -> HashMap<u64, Arc<ShaderModule>, FxBuildHasher> {
        self.entries
            .values()
            .flatten()
            .map(|entry| (entry.module().hash, entry.module().clone()))
            .collect()
    }

    /// Queues `entry` for the next [`Self::flush`].
    pub(crate) fn append(&mut self, entry: &CacheEntry) {
        entry.encode(&mut self.pending);
    }

    /// Appends the queued entries to the file.
    pub(crate) fn flush(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        OpenOptions::new()
            .append(true)
            .open(&self.path)?
            .write_all(&self.pending)?;
        self.pending.clear();
        Ok(())
    }
}

impl Drop for ShaderDisk {
    fn drop(&mut self) {
        if let Err(error) = self.flush() {
            tracing::warn!("writing {}: {error}", self.path.display());
        }
    }
}
