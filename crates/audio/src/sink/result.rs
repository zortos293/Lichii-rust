//! Result codes of the audio services (`Service::Audio` module 153).

use lichii_common::ResultCode;

const MODULE: u32 = 153;

pub const NOT_FOUND: ResultCode = ResultCode::new(MODULE, 1);
pub const OPERATION_FAILED: ResultCode = ResultCode::new(MODULE, 2);
pub const INVALID_SAMPLE_RATE: ResultCode = ResultCode::new(MODULE, 3);
pub const INSUFFICIENT_BUFFER: ResultCode = ResultCode::new(MODULE, 4);
pub const OUT_OF_SESSIONS: ResultCode = ResultCode::new(MODULE, 5);
pub const BUFFER_COUNT_REACHED: ResultCode = ResultCode::new(MODULE, 8);
pub const INVALID_CHANNEL_COUNT: ResultCode = ResultCode::new(MODULE, 10);
pub const INVALID_UPDATE_INFO: ResultCode = ResultCode::new(MODULE, 41);
pub const INVALID_ADDRESS_INFO: ResultCode = ResultCode::new(MODULE, 42);
pub const NOT_SUPPORTED: ResultCode = ResultCode::new(MODULE, 513);
pub const INVALID_ARGUMENT: ResultCode = ResultCode::new(MODULE, 900);
pub const INVALID_HANDLE: ResultCode = ResultCode::new(MODULE, 1536);
pub const INVALID_REVISION: ResultCode = ResultCode::new(MODULE, 1537);

/// Eden's `ResultUnknown` (all bits set), which the audio services answer for missing process
/// handles and buffers.
pub const UNKNOWN: ResultCode = ResultCode(u32::MAX);
