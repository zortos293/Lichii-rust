use std::io;

use lichii_crypto::KeyError;

/// Why a container could not be opened.
#[derive(Debug, thiserror::Error)]
pub enum FsError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("{format}: bad magic {found:02x?}")]
    BadMagic {
        format: &'static str,
        found: [u8; 4],
    },
    #[error("{format}: {reason}")]
    Corrupt {
        format: &'static str,
        reason: String,
    },
    #[error(transparent)]
    Key(#[from] KeyError),
    #[error("{0}")]
    Unsupported(String),
}

pub type FsResult<T> = Result<T, FsError>;

pub(crate) fn corrupt(format: &'static str, reason: impl Into<String>) -> FsError {
    FsError::Corrupt {
        format,
        reason: reason.into(),
    }
}

impl From<FsError> for io::Error {
    fn from(error: FsError) -> Self {
        match error {
            FsError::Io(error) => error,
            other => io::Error::new(io::ErrorKind::InvalidData, other),
        }
    }
}
