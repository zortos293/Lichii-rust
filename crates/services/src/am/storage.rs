//! `IStorage` and `IStorageAccessor`: byte buffers passed between applets.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use lichii_common::ResultCode;
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use super::{INVALID_OFFSET, am};
use crate::{Service, ServiceContext};

const INVALID_STORAGE_TYPE: ResultCode = am(511);

/// Bytes of guest-created storage an applet may hold at once.
pub(super) const LIVE_STORAGE_BUDGET: usize = 256 << 20;

/// A share of an applet's live-storage budget, given back when the storage is freed.
pub(super) struct Charge {
    budget: Arc<AtomicUsize>,
    bytes: usize,
}

impl Charge {
    pub fn take(budget: &Arc<AtomicUsize>, bytes: usize) -> Option<Self> {
        budget
            .try_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                live.checked_add(bytes)
                    .filter(|&total| total <= LIVE_STORAGE_BUDGET)
            })
            .ok()?;
        Some(Self {
            budget: budget.clone(),
            bytes,
        })
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.budget.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// The bytes an `IStorage` and its accessors share.
pub(super) struct StorageData {
    bytes: Mutex<Vec<u8>>,
    _charge: Option<Charge>,
}

impl StorageData {
    pub fn new(bytes: Vec<u8>, charge: Option<Charge>) -> Self {
        Self {
            bytes: Mutex::new(bytes),
            _charge: charge,
        }
    }

    /// The bytes, locked.
    pub fn bytes(&self) -> parking_lot::MutexGuard<'_, Vec<u8>> {
        self.bytes.lock()
    }
}

pub(super) struct Storage {
    pub context: Arc<ServiceContext>,
    pub data: Arc<StorageData>,
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
    data: Arc<StorageData>,
}

impl Service for StorageAccessor {
    fn name(&self) -> &str {
        "IStorageAccessor"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                ctx.push(self.data.bytes.lock().len() as i64);
                ResultCode::SUCCESS
            }
            10 => {
                let offset: i64 = ctx.pop();
                let input = match ctx.read_buffer(0) {
                    Ok(input) => input,
                    Err(code) => return Some(code),
                };
                let mut data = self.data.bytes.lock();
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
                let data = self.data.bytes.lock();
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
