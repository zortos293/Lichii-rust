use std::io;

use lichii_crypto::KeyError;
use lichii_fs::FsError;

/// Why a program could not be loaded or a title not identified.
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Fs(#[from] FsError),
    #[error(transparent)]
    Key(#[from] KeyError),
    #[error("{0}: bad magic")]
    BadMagic(&'static str),
    #[error("{0}: {1}")]
    Corrupt(&'static str, &'static str),
    #[error("{module}: segment {segment} failed to decompress: {reason}")]
    Decompress {
        module: String,
        segment: usize,
        reason: String,
    },
    #[error("{module}: segment {segment} does not match its SHA-256")]
    HashMismatch { module: String, segment: usize },
    #[error("the ExeFS has no main.npdm")]
    MissingNpdm,
    #[error("the ExeFS has no main")]
    MissingMain,
    #[error("{0} is not a title Lichii can load")]
    NotATitle(String),
}

pub type LoaderResult<T> = Result<T, LoaderError>;
