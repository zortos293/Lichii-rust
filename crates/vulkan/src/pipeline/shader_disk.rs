//! The per-title file of translated shaders (`lichii_shader::cache`): every entry loaded at
//! start-up and indexed by its probe key, and the entries translated since, appended in a
//! batch.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::Arc;

use lichii_shader::ShaderModule;
use lichii_shader::cache::{self, CacheEntry, HEADER_LEN};

use super::FxBuildHasher;

pub(crate) struct ShaderDisk {
    path: PathBuf,
    entries: HashMap<u64, Vec<CacheEntry>, FxBuildHasher>,
    /// Bytes of the file up to its last complete record.
    len: u64,
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
        let len = match cache::records(&file) {
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
                valid
            }
            Err(_) => {
                if !file.is_empty() {
                    tracing::info!("{}: stale shader cache, starting a new one", path.display());
                }
                fs::write(&path, cache::header())?;
                HEADER_LEN
            }
        };
        let count: usize = entries.values().map(Vec::len).sum();
        if count > 0 {
            tracing::info!("shader cache: {count} translated programs");
        }
        Ok(Self {
            path,
            entries,
            len: len as u64,
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

    /// Appends the queued entries to the file. A write that failed part-way is cut off before
    /// the next one, so a retry never lands behind an incomplete record.
    pub(crate) fn flush(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let mut file = OpenOptions::new().write(true).open(&self.path)?;
        file.set_len(self.len)?;
        file.seek(SeekFrom::Start(self.len))?;
        file.write_all(&self.pending)?;
        self.len += self.pending.len() as u64;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flush_after_a_partial_write_starts_at_the_last_complete_record() {
        let dir = std::env::temp_dir().join(format!("lichii-partial-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shaders.bin");
        let mut disk = ShaderDisk::open(path.clone()).unwrap();
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0xaa; 5])
            .unwrap();
        disk.pending.extend_from_slice(b"record");
        disk.flush().unwrap();
        let mut want = cache::header().to_vec();
        want.extend_from_slice(b"record");
        assert_eq!(fs::read(&path).unwrap(), want);
        drop(disk);
        let _ = fs::remove_dir_all(&dir);
    }
}
