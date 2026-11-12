//! `nn::fs` result codes (module 2) and how host I/O errors map onto them.

use std::io;

use lichii_common::ResultCode;

const fn fs(description: u32) -> ResultCode {
    ResultCode::new(2, description)
}

pub const PATH_NOT_FOUND: ResultCode = fs(1);
pub const PATH_ALREADY_EXISTS: ResultCode = fs(2);
pub const DIRECTORY_NOT_EMPTY: ResultCode = fs(8);
pub const USABLE_SPACE_NOT_ENOUGH: ResultCode = fs(30);
pub const TARGET_NOT_FOUND: ResultCode = fs(1002);
pub const PORT_SD_CARD_NO_DEVICE: ResultCode = fs(2001);
pub const NOT_IMPLEMENTED: ResultCode = fs(3001);
pub const OUT_OF_RANGE: ResultCode = fs(3005);
pub const INVALID_ARGUMENT: ResultCode = fs(6001);
pub const INVALID_PATH: ResultCode = fs(6002);
pub const INVALID_CHARACTER: ResultCode = fs(6004);
pub const INVALID_OFFSET: ResultCode = fs(6061);
pub const INVALID_SIZE: ResultCode = fs(6062);
pub const NULLPTR_ARGUMENT: ResultCode = fs(6063);
pub const INVALID_OPEN_MODE: ResultCode = fs(6072);
pub const FILE_EXTENSION_WITHOUT_OPEN_MODE_ALLOW_APPEND: ResultCode = fs(6201);
pub const READ_NOT_PERMITTED: ResultCode = fs(6202);
pub const WRITE_NOT_PERMITTED: ResultCode = fs(6203);
pub const UNSUPPORTED_OPERATE_RANGE_FOR_STORAGE: ResultCode = fs(6309);
pub const UNSUPPORTED_OPERATE_RANGE_FOR_FILE: ResultCode = fs(6362);
pub const UNSUPPORTED_WRITE_FOR_READ_ONLY_FILE_SYSTEM: ResultCode = fs(6369);
pub const UNSUPPORTED_WRITE_FOR_READ_ONLY_FILE: ResultCode = fs(6372);
/// What the C++ answers where it has no specific code (`ResultUnknown`).
pub const UNKNOWN: ResultCode = ResultCode(u32::MAX);

/// The result for a host I/O error. `read_only` is what a write to a read-only backend answers.
pub fn from_io(error: &io::Error, read_only: ResultCode) -> ResultCode {
    match error.kind() {
        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory => PATH_NOT_FOUND,
        io::ErrorKind::AlreadyExists => PATH_ALREADY_EXISTS,
        io::ErrorKind::DirectoryNotEmpty => DIRECTORY_NOT_EMPTY,
        io::ErrorKind::PermissionDenied | io::ErrorKind::ReadOnlyFilesystem => read_only,
        io::ErrorKind::InvalidInput | io::ErrorKind::InvalidFilename => INVALID_PATH,
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded => USABLE_SPACE_NOT_ENOUGH,
        _ => {
            tracing::warn!("host file system error: {error}");
            UNKNOWN
        }
    }
}
