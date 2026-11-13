//! `IStorage`, `IStorageAccessor` and `ITransferStorageAccessor`: data passed between
//! applets, held by am or lent by the title as transfer memory.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory_objects::TransferMemory;
use lichii_kernel::object::KObject;
use parking_lot::Mutex;

use super::{INVALID_OFFSET, UNKNOWN, am};
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

/// The data an `IStorage` and its accessors share.
pub(super) struct StorageData {
    backing: Backing,
    _charge: Option<Charge>,
}

enum Backing {
    Buffer(Mutex<Vec<u8>>),
    /// The title's transfer memory, read and written in place. A handle storage hands the
    /// transfer memory itself to whoever opens it, through `OpenTransferStorage`.
    TransferMemory {
        memory: Arc<TransferMemory>,
        size: usize,
        writable: bool,
        handle: bool,
    },
}

impl StorageData {
    pub fn new(bytes: Vec<u8>, charge: Option<Charge>) -> Self {
        Self {
            backing: Backing::Buffer(Mutex::new(bytes)),
            _charge: charge,
        }
    }

    /// `size` bytes of `memory`, as `CreateTransferMemoryStorage` (`handle` false) and
    /// `CreateHandleStorage` (`handle` true, always writable) lend it.
    pub fn transfer_memory(
        memory: Arc<TransferMemory>,
        size: usize,
        writable: bool,
        handle: bool,
    ) -> Self {
        Self {
            backing: Backing::TransferMemory {
                memory,
                size,
                writable,
                handle,
            },
            _charge: None,
        }
    }

    pub fn len(&self) -> usize {
        match &self.backing {
            Backing::Buffer(bytes) => bytes.lock().len(),
            Backing::TransferMemory { size, .. } => *size,
        }
    }

    /// Fills `out` from `offset`, which the caller checked against [`Self::len`].
    fn read(&self, offset: usize, out: &mut [u8]) -> HleResult<()> {
        match &self.backing {
            Backing::Buffer(bytes) => {
                out.copy_from_slice(&bytes.lock()[offset..offset + out.len()]);
                Ok(())
            }
            Backing::TransferMemory { memory, .. } => memory.read(offset as u64, out),
        }
    }

    /// Writes `data` at `offset`, which the caller checked against [`Self::len`].
    fn write(&self, offset: usize, data: &[u8]) -> HleResult<()> {
        match &self.backing {
            Backing::Buffer(bytes) => {
                bytes.lock()[offset..offset + data.len()].copy_from_slice(data);
                Ok(())
            }
            Backing::TransferMemory {
                writable: false, ..
            } => Err(UNKNOWN),
            Backing::TransferMemory { memory, .. } => memory.write(offset as u64, data),
        }
    }

    /// Every byte; unreadable transfer memory reads as zeroes, as the reference's `GetData`.
    pub fn to_vec(&self) -> Vec<u8> {
        let mut data = vec![0; self.len()];
        if let Err(code) = self.read(0, &mut data) {
            tracing::warn!(%code, "reading a transfer memory storage");
        }
        data
    }

    /// The transfer memory a handle storage lends.
    fn handle(&self) -> Option<&Arc<TransferMemory>> {
        match &self.backing {
            Backing::TransferMemory {
                memory,
                handle: true,
                ..
            } => Some(memory),
            _ => None,
        }
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
            0 if self.data.handle().is_some() => INVALID_STORAGE_TYPE,
            0 => {
                ctx.push_interface(self.context.handler(StorageAccessor {
                    data: self.data.clone(),
                }));
                ResultCode::SUCCESS
            }
            1 if self.data.handle().is_none() => INVALID_STORAGE_TYPE,
            1 => {
                ctx.push_interface(self.context.handler(TransferStorageAccessor {
                    data: self.data.clone(),
                }));
                ResultCode::SUCCESS
            }
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
        let result = match ctx.command() {
            0 => {
                ctx.push(self.data.len() as i64);
                Ok(())
            }
            10 => {
                let offset: i64 = ctx.pop();
                ctx.read_buffer(0).and_then(|input| {
                    let range =
                        range(offset, input.len(), self.data.len()).ok_or(INVALID_OFFSET)?;
                    self.data.write(range.start, &input)
                })
            }
            11 => {
                let offset: i64 = ctx.pop();
                let size = ctx
                    .writable_buffer(0)
                    .map_or(0, |buffer| buffer.size as usize);
                range(offset, size, self.data.len())
                    .ok_or(INVALID_OFFSET)
                    .and_then(|range| {
                        let mut chunk = vec![0; range.len()];
                        self.data.read(range.start, &mut chunk)?;
                        ctx.write_buffer(0, &chunk).map(drop)
                    })
            }
            _ => return None,
        };
        Some(result.err().unwrap_or(ResultCode::SUCCESS))
    }
}

/// `ITransferStorageAccessor`: the size and transfer memory of a handle storage.
struct TransferStorageAccessor {
    data: Arc<StorageData>,
}

impl Service for TransferStorageAccessor {
    fn name(&self) -> &str {
        "ITransferStorageAccessor"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push(self.data.len() as i64),
            1 => {
                ctx.push(self.data.len() as i64);
                let memory = self.data.handle()?.clone();
                ctx.push_copy_object(KObject::TransferMemory(memory));
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

fn range(offset: i64, len: usize, size: usize) -> Option<std::ops::Range<usize>> {
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(len)?;
    (end <= size).then_some(start..end)
}
