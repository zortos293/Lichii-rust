//! Kernel result codes (module 1).

use lichii_common::ResultCode;

const fn kernel(description: u32) -> ResultCode {
    ResultCode::new(1, description)
}

pub const OUT_OF_SESSIONS: ResultCode = kernel(7);
pub const INVALID_ARGUMENT: ResultCode = kernel(14);
pub const NOT_IMPLEMENTED: ResultCode = kernel(33);
pub const NO_SYNCHRONIZATION_OBJECT: ResultCode = kernel(57);
pub const TERMINATION_REQUESTED: ResultCode = kernel(59);
pub const NO_EVENT: ResultCode = kernel(70);
pub const INVALID_SIZE: ResultCode = kernel(101);
pub const INVALID_ADDRESS: ResultCode = kernel(102);
pub const OUT_OF_RESOURCE: ResultCode = kernel(103);
pub const OUT_OF_MEMORY: ResultCode = kernel(104);
pub const OUT_OF_HANDLES: ResultCode = kernel(105);
pub const INVALID_CURRENT_MEMORY: ResultCode = kernel(106);
pub const INVALID_NEW_MEMORY_PERMISSION: ResultCode = kernel(108);
pub const INVALID_MEMORY_REGION: ResultCode = kernel(110);
pub const INVALID_PRIORITY: ResultCode = kernel(112);
pub const INVALID_CORE_ID: ResultCode = kernel(113);
pub const INVALID_HANDLE: ResultCode = kernel(114);
pub const INVALID_POINTER: ResultCode = kernel(115);
pub const INVALID_COMBINATION: ResultCode = kernel(116);
pub const TIMED_OUT: ResultCode = kernel(117);
pub const CANCELLED: ResultCode = kernel(118);
pub const OUT_OF_RANGE: ResultCode = kernel(119);
pub const INVALID_ENUM_VALUE: ResultCode = kernel(120);
pub const NOT_FOUND: ResultCode = kernel(121);
pub const BUSY: ResultCode = kernel(122);
pub const SESSION_CLOSED: ResultCode = kernel(123);
pub const NOT_HANDLED: ResultCode = kernel(124);
pub const INVALID_STATE: ResultCode = kernel(125);
pub const RESERVED_USED: ResultCode = kernel(126);
pub const NOT_SUPPORTED: ResultCode = kernel(127);
pub const PORT_CLOSED: ResultCode = kernel(131);
pub const LIMIT_REACHED: ResultCode = kernel(132);
pub const INVALID_MEMORY_POOL: ResultCode = kernel(133);
pub const RECEIVE_LIST_BROKEN: ResultCode = kernel(258);
pub const OUT_OF_ADDRESS_SPACE: ResultCode = kernel(259);
pub const MESSAGE_TOO_LARGE: ResultCode = kernel(260);
pub const INVALID_PROCESS_ID: ResultCode = kernel(517);
pub const INVALID_THREAD_ID: ResultCode = kernel(518);
pub const INVALID_ID: ResultCode = kernel(519);
pub const PROCESS_TERMINATED: ResultCode = kernel(520);
