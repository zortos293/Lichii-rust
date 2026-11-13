//! The smaller ns interfaces: documents, application records, dynamic rights, content
//! management, download tasks, system update and vulnerability checks.

use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;

use super::Placeholder;
use super::application_manager::list_application_record;
use crate::{Service, ServiceContext};

const GIB: u64 = 1 << 30;
const NAND_USER_SIZE: u64 = 0x6_8000_0000;
const NAND_SYSTEM_SIZE: u64 = 0xa000_0000;
const NAND_TOTAL_SIZE: u64 = 0x7_47c0_0000;
/// The emulated SD card: the C++ rounds the used size up to the next 4 GiB block.
const SD_CARD_SIZE: u64 = 4 * GIB;

/// Total bytes of `FileSys::StorageId` `storage` (1 host, 3 system NAND, 4 user NAND,
/// 5 SD card).
pub(crate) fn storage_total_space(storage: u8) -> u64 {
    match storage {
        1 => NAND_TOTAL_SIZE,
        3 => NAND_SYSTEM_SIZE,
        4 => NAND_USER_SIZE,
        5 => SD_CARD_SIZE,
        _ => 0,
    }
}

/// Free bytes of `storage`. User NAND reports 1 MiB less than its size, since titles such as
/// BioShock check that it is not exactly 0x680000000.
pub(crate) fn storage_free_space(storage: u8) -> u64 {
    let user_free = NAND_USER_SIZE - 0x10_0000;
    match storage {
        1 => NAND_SYSTEM_SIZE + user_free,
        3 => NAND_SYSTEM_SIZE,
        4 => user_free,
        5 => SD_CARD_SIZE,
        _ => 0,
    }
}

pub(crate) struct Document {
    pub context: Arc<ServiceContext>,
}

impl Service for Document {
    fn name(&self) -> &str {
        "IDocumentInterface"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            23 => ResultCode::SUCCESS,
            92 => {
                ctx.push(self.context.program.program_id);
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

pub(crate) struct ReadOnlyApplicationRecord {
    pub context: Arc<ServiceContext>,
}

impl Service for ReadOnlyApplicationRecord {
    fn name(&self) -> &str {
        "IReadOnlyApplicationRecordInterface"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                ctx.push(1u8);
                ResultCode::SUCCESS
            }
            2 => {
                ctx.push(0u8);
                ResultCode::SUCCESS
            }
            3 => list_application_record(&self.context, ctx),
            _ => return None,
        })
    }
}

pub(crate) struct DynamicRights;

impl Service for DynamicRights {
    fn name(&self) -> &str {
        "DynamicRightsInterface"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            5 | 18 => ResultCode::SUCCESS,
            13 => {
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            26 => {
                ctx.push(0u8);
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

pub(crate) struct ContentManagement;

impl Service for ContentManagement {
    fn name(&self) -> &str {
        "IContentManagementInterface"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            11 => {
                for _ in 0..4 {
                    ctx.push(5u64);
                    ctx.push(8 * GIB);
                    ctx.push(2 * GIB);
                    ctx.push(12u64 << 20);
                }
                ResultCode::SUCCESS
            }
            43 => ResultCode::SUCCESS,
            47 => {
                let storage: u8 = ctx.pop();
                ctx.push(storage_total_space(storage) as i64);
                ResultCode::SUCCESS
            }
            48 => {
                let storage: u8 = ctx.pop();
                ctx.push(storage_free_space(storage) as i64);
                ResultCode::SUCCESS
            }
            71 => {
                ctx.push(0u64);
                ctx.push(0u64);
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

pub(crate) struct DownloadTask;

impl Service for DownloadTask {
    fn name(&self) -> &str {
        "IDownloadTaskInterface"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        matches!(ctx.command(), 707 | 708).then_some(ResultCode::SUCCESS)
    }
}

/// `nn::ns::detail::IAsyncResult`: Cancel signals the operation's event.
pub(crate) struct AsyncResult {
    pub event: Option<Arc<Event>>,
}

impl Service for AsyncResult {
    fn name(&self) -> &str {
        "nn::ns::detail::IAsyncResult"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        (ctx.command() == 1).then(|| {
            if let Some(event) = &self.event {
                event.signal();
            }
            ResultCode::SUCCESS
        })
    }
}

/// ns:su: no system update is ever available.
pub(crate) struct SystemUpdate {
    pub context: Arc<ServiceContext>,
    pub notification: Arc<Event>,
}

impl Service for SystemUpdate {
    fn name(&self) -> &str {
        "ns:su"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                ctx.push(0u8);
                ResultCode::SUCCESS
            }
            1 => {
                ctx.push_interface(self.context.handler(Placeholder("ISystemUpdateControl")));
                ResultCode::SUCCESS
            }
            9 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.notification.clone()));
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

/// ns:vm: the system never needs a vulnerability update.
pub(crate) struct VulnerabilityManager;

impl Service for VulnerabilityManager {
    fn name(&self) -> &str {
        "ns:vm"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            1200 => {
                ctx.push(0u8);
                ResultCode::SUCCESS
            }
            3100 => {
                ctx.push([0u8; 0x20]);
                ResultCode::SUCCESS
            }
            3102 => ResultCode::SUCCESS,
            _ => return None,
        })
    }
}
