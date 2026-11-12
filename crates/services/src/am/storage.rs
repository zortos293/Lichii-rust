//! `IStorage` and `IStorageAccessor`: byte buffers passed between applets.

use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use super::{INVALID_OFFSET, am};
use crate::{Service, ServiceContext};

const INVALID_STORAGE_TYPE: ResultCode = am(511);

pub(super) struct Storage {
    pub context: Arc<ServiceContext>,
    pub data: Arc<Mutex<Vec<u8>>>,
}

impl Service for Storage {
    fn name(&self) -> &str {
        "IStorage"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                ctx.push_interface(self.context.handler(StorageAccessor {
                    data: self.data.clone(),
                }));
                ResultCode::SUCCESS
            }
            1 => INVALID_STORAGE_TYPE,
            _ => return None,
        })
    }
}

struct StorageAccessor {
    data: Arc<Mutex<Vec<u8>>>,
}

impl Service for StorageAccessor {
    fn name(&self) -> &str {
        "IStorageAccessor"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                ctx.push(self.data.lock().len() as i64);
                ResultCode::SUCCESS
            }
            10 => {
                let offset: i64 = ctx.pop();
                let input = match ctx.read_buffer(0) {
                    Ok(input) => input,
                    Err(code) => return Some(code),
                };
                let mut data = self.data.lock();
                match range(offset, input.len(), data.len()) {
                    Some(range) => {
                        data[range].copy_from_slice(&input);
                        ResultCode::SUCCESS
                    }
                    None => INVALID_OFFSET,
                }
            }
            11 => {
                let offset: i64 = ctx.pop();
                let size = ctx
                    .writable_buffer(0)
                    .map_or(0, |buffer| buffer.size as usize);
                let data = self.data.lock();
                let Some(range) = range(offset, size, data.len()) else {
                    return Some(INVALID_OFFSET);
                };
                let chunk = data[range].to_vec();
                drop(data);
                match ctx.write_buffer(0, &chunk) {
                    Ok(_) => ResultCode::SUCCESS,
                    Err(code) => code,
                }
            }
            _ => return None,
        })
    }
}

fn range(offset: i64, len: usize, size: usize) -> Option<std::ops::Range<usize>> {
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(len)?;
    (end <= size).then_some(start..end)
}
