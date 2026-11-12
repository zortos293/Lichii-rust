//! NSP (Nintendo Submission Package): a PFS0 of NCAs, tickets and certificates.

use lichii_crypto::KeySet;

use crate::content::ContentIndex;
use crate::error::FsResult;
use crate::pfs::PartitionFs;
use crate::vfs::FileRef;

/// An opened NSP and the titles it holds.
pub struct Nsp {
    partition: PartitionFs,
    index: ContentIndex,
}

impl Nsp {
    /// Opens the package, imports its common tickets' title keys into `keys` and indexes its
    /// titles.
    pub fn open(file: &FileRef, keys: &mut KeySet) -> FsResult<Self> {
        let partition = PartitionFs::open(file)?;
        let index = ContentIndex::from_dir(&partition, keys)?;
        Ok(Self { partition, index })
    }

    pub fn partition(&self) -> &PartitionFs {
        &self.partition
    }

    pub fn index(&self) -> &ContentIndex {
        &self.index
    }

    pub fn into_index(self) -> ContentIndex {
        self.index
    }
}
