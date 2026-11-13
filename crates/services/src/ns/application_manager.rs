//! IApplicationManagerInterface: application records, views, storage sizes and the control data
//! commands, for a console whose only application is the running title.

use std::sync::Arc;

use lichii_common::{ConsoleGeneration, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;

use super::control::{self, RESULT_UNKNOWN};
use super::interfaces::{AsyncResult, storage_free_space, storage_total_space};
use crate::{Service, ServiceContext};

const APPLICATION_RECORD_SIZE: usize = 0x18;
/// `ApplicationEvent::Installed`.
const EVENT_INSTALLED: u8 = 3;
const VIEW_VERSION: u32 = 0x70000;
const VIEW_FLAGS: u32 = 0x401f17;
const VIEW_V19_SIZE: usize = 0x50;
const VIEW_V20_SIZE: usize = 0x58;
const PROMOTION_INFO_SIZE: usize = 0x20;
const RIGHTS_SIZE: usize = 0x20;

pub(crate) struct ApplicationManager {
    context: Arc<ServiceContext>,
    record_update: Arc<Event>,
    sd_card_mount_status: Arc<Event>,
    game_card_update_detection: Arc<Event>,
    game_card_mount_failure: Arc<Event>,
    game_card_waken_ready: Arc<Event>,
    unknown: Arc<Event>,
}

impl ApplicationManager {
    pub fn new(context: Arc<ServiceContext>) -> Self {
        Self {
            context,
            record_update: Arc::new(Event::new()),
            sd_card_mount_status: Arc::new(Event::new()),
            game_card_update_detection: Arc::new(Event::new()),
            game_card_mount_failure: Arc::new(Event::new()),
            game_card_waken_ready: Arc::new(Event::new()),
            unknown: Arc::new(Event::new()),
        }
    }
}

/// The application records (u64 id, u8 event, u8 attributes, pad, s64 last updated) of the
/// installed applications from `offset`, as many as `capacity` bytes hold. The running title
/// is the only one, listed unless it is a system title or a sub-program.
pub(crate) fn application_records(
    context: &ServiceContext,
    offset: i32,
    capacity: usize,
) -> Vec<u8> {
    let id = context.program.program_id;
    let listed = id >= 0x0100_0000_0000_1fff && id & 0xfff == 0;
    let mut records = Vec::new();
    if listed && offset <= 0 && capacity >= APPLICATION_RECORD_SIZE {
        records.extend_from_slice(&id.to_le_bytes());
        records.push(EVENT_INSTALLED);
        records.resize(APPLICATION_RECORD_SIZE, 0);
    }
    records
}

/// ListApplicationRecord(s32 offset) -> s32 count, records in output buffer 0.
pub(crate) fn list_application_record(
    context: &ServiceContext,
    ctx: &mut HleRequestContext,
) -> ResultCode {
    let offset: i32 = ctx.pop();
    let capacity = ctx.writable_buffer(0).map_or(0, |b| b.size as usize);
    let records = application_records(context, offset, capacity);
    if !records.is_empty()
        && let Err(code) = ctx.write_buffer(0, &records)
    {
        return code;
    }
    ctx.push((records.len() / APPLICATION_RECORD_SIZE) as i32);
    ResultCode::SUCCESS
}

fn signal_and_push(ctx: &mut HleRequestContext, event: &Arc<Event>, signal: bool) -> ResultCode {
    if signal {
        event.signal();
    }
    ctx.push_copy_object(KObject::ReadableEvent(event.clone()));
    ResultCode::SUCCESS
}

impl ApplicationManager {
    /// The views of the application ids in input buffer 0: version 0x70000 and flags 0x401f17,
    /// the rest zero, `view_size` bytes each with `padding` zero bytes after each.
    fn views(&self, ctx: &HleRequestContext, view_size: usize, padding: usize) -> Vec<u8> {
        let ids = ctx.read_buffer(0).unwrap_or_default();
        let capacity = ctx.writable_buffer(0).map_or(0, |b| b.size as usize);
        let entry = view_size + padding;
        let mut out = Vec::new();
        for id in ids.as_chunks::<8>().0.iter().take(capacity / entry) {
            let start = out.len();
            out.extend_from_slice(id);
            out.extend_from_slice(&VIEW_VERSION.to_le_bytes());
            out.extend_from_slice(&VIEW_FLAGS.to_le_bytes());
            out.resize(start + entry, 0);
        }
        out
    }

    fn write_views(&self, ctx: &mut HleRequestContext, view_size: usize) -> ResultCode {
        let views = self.views(ctx, view_size, 0);
        if !views.is_empty()
            && let Err(code) = ctx.write_buffer(0, &views)
        {
            return code;
        }
        ResultCode::SUCCESS
    }
}

impl Service for ApplicationManager {
    fn name(&self) -> &str {
        "IApplicationManagerInterface"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let context = &self.context;
        Some(match ctx.command() {
            0 => list_application_record(context, ctx),
            2 => signal_and_push(ctx, &self.record_update, true),
            4 | 5 | 16 => {
                self.record_update.signal();
                ResultCode::SUCCESS
            }
            38 | 43 | 70 | 221 | 904 | 4053 | 4105 => ResultCode::SUCCESS,
            40 => {
                tracing::warn!("GetApplicationLogoData: no logo partition");
                RESULT_UNKNOWN
            }
            44 => signal_and_push(ctx, &self.sd_card_mount_status, false),
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
            52 => signal_and_push(ctx, &self.game_card_update_detection, false),
            55 => control::get_application_desired_language(context, ctx),
            59 => control::convert_application_language_to_language_code(ctx),
            71 => {
                let storage: u8 = ctx.pop();
                ctx.push(storage_total_space(storage) as i64);
                ctx.push(storage_free_space(storage) as i64);
                ResultCode::SUCCESS
            }
            400 => control::get_application_control_data(context, ctx),
            407 => control::list_application_title(context, ctx),
            408 => control::list_application_icon(context, ctx),
            419 => {
                self.unknown.signal();
                ResultCode::SUCCESS
            }
            505 => signal_and_push(ctx, &self.game_card_mount_failure, false),
            511 => signal_and_push(ctx, &self.game_card_waken_ready, false),
            512 => {
                ctx.push(0u8);
                ResultCode::SUCCESS
            }
            906 => {
                ctx.push(0u8);
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            936 | 4023 => {
                ctx.push(0u64);
                ResultCode::SUCCESS
            }
            1300 | 2520 => {
                ctx.push(1u8);
                ResultCode::SUCCESS
            }
            1701 => self.write_views(ctx, VIEW_V19_SIZE),
            1704 => {
                let view_size = match ctx.process().console() {
                    ConsoleGeneration::Switch2 => VIEW_V20_SIZE,
                    ConsoleGeneration::Switch1 => VIEW_V19_SIZE,
                };
                let views = self.views(ctx, view_size, PROMOTION_INFO_SIZE);
                if !views.is_empty()
                    && let Err(code) = ctx.write_buffer(0, &views)
                {
                    return Some(code);
                }
                ctx.push((views.len() / (view_size + PROMOTION_INFO_SIZE)) as u32);
                ResultCode::SUCCESS
            }
            1706 => self.write_views(ctx, VIEW_V20_SIZE),
            2050 => {
                let _flags: u32 = ctx.pop();
                let application_id: u64 = ctx.pop();
                let uid: [u8; 16] = ctx.pop();
                let capacity = ctx.writable_buffer(0).map_or(0, |b| b.size as usize);
                if capacity < RIGHTS_SIZE {
                    ctx.push(0u32);
                    return Some(ResultCode::SUCCESS);
                }
                let mut rights = [0u8; RIGHTS_SIZE];
                rights[..8].copy_from_slice(&application_id.to_le_bytes());
                rights[8..24].copy_from_slice(&uid);
                if let Err(code) = ctx.write_buffer(0, &rights) {
                    return Some(code);
                }
                ctx.push(1u32);
                ResultCode::SUCCESS
            }
            2100 => {
                ctx.push(ResultCode::SUCCESS.0);
                ResultCode::SUCCESS
            }
            4022 | 4088 => signal_and_push(ctx, &self.unknown, true),
            4042 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.unknown.clone()));
                ctx.push_interface(context.handler(AsyncResult {
                    event: Some(self.unknown.clone()),
                }));
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}
