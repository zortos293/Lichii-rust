//! IReadOnlyApplicationControlDataInterface: the running title's NACP and icon, its titles, and
//! the language to show it in. IApplicationManagerInterface answers the same commands.

use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory_objects::TransferMemory;
use lichii_kernel::object::KObject;
use lichii_loader::program::ProgramImage;

use super::language::ApplicationLanguage;
use crate::{Service, ServiceContext};

/// `ResultUnknown` (all bits set), what the C++ returns for a too small buffer.
pub(crate) const RESULT_UNKNOWN: ResultCode = ResultCode(u32::MAX);
/// ns ResultApplicationLanguageNotFound (16-300).
pub(crate) const APPLICATION_LANGUAGE_NOT_FOUND: ResultCode = ResultCode::new(16, 300);
const NACP_SIZE: usize = 0x4000;
const TITLE_ENTRY_SIZE: usize = 0x300;

pub(crate) struct ReadOnlyApplicationControlData {
    pub context: Arc<ServiceContext>,
}

impl Service for ReadOnlyApplicationControlData {
    fn name(&self) -> &str {
        "IReadOnlyApplicationControlDataInterface"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => get_application_control_data(&self.context, ctx),
            1 => get_application_desired_language(&self.context, ctx),
            2 => convert_application_language_to_language_code(ctx),
            5 => get_application_control_data_flags(&self.context, ctx, Layout::Packed),
            10 => list_application_icon(&self.context, ctx),
            13 => list_application_title(&self.context, ctx),
            19 | 23 => get_application_control_data_flags(&self.context, ctx, Layout::Flags),
            _ => return None,
        })
    }
}

/// The control data the title knows about: the running program's NACP. The icon is not part of
/// `ProgramImage` yet, so it is always empty.
fn control_nacp(program: &ProgramImage, application_id: u64) -> Option<&[u8; NACP_SIZE]> {
    if application_id != program.program_id {
        tracing::warn!(application_id, "no control data for this application");
        return None;
    }
    program.nacp.as_ref().map(|nacp| nacp.raw())
}

/// The NACP (zeroes without one) followed by the icon, padded with zeroes to `len`.
fn control_bytes(nacp: Option<&[u8; NACP_SIZE]>, icon: &[u8], len: usize) -> Vec<u8> {
    let mut data = vec![0u8; len.max(NACP_SIZE)];
    if let Some(nacp) = nacp {
        data[..NACP_SIZE].copy_from_slice(nacp);
    }
    let icon_len = icon.len().min(data.len() - NACP_SIZE);
    data[NACP_SIZE..NACP_SIZE + icon_len].copy_from_slice(&icon[..icon_len]);
    data.truncate(len);
    data
}

/// GetApplicationControlData(u8 source, u64 application_id) -> u32 size, NACP + icon.
pub(crate) fn get_application_control_data(
    context: &ServiceContext,
    ctx: &mut HleRequestContext,
) -> ResultCode {
    let _source: u8 = ctx.pop();
    let application_id: u64 = ctx.pop();
    let nacp = control_nacp(&context.program, application_id);
    let icon: &[u8] = &[];
    let total = NACP_SIZE + icon.len();
    let Some(buffer) = ctx.writable_buffer(0).filter(|b| b.size as usize >= total) else {
        tracing::error!("GetApplicationControlData: output buffer is too small");
        return RESULT_UNKNOWN;
    };
    if let Err(code) = ctx.write_buffer(0, &control_bytes(nacp, icon, buffer.size as usize)) {
        return code;
    }
    ctx.push(total as u32);
    ResultCode::SUCCESS
}

#[derive(Clone, Copy)]
enum Layout {
    /// Command 5: u64 (size << 32 | flag).
    Packed,
    /// Commands 19 and 23: u32 0x10001, u32 (0x10001 when the flag is 1), u32 size.
    Flags,
}

/// GetApplicationControlData(u8 source, u8 flag1, u8 flag2, u64 application_id) with a NACP and
/// as much of the icon as fits, the rest zeroed. Flag 1 asks for a resized JPEG; the icon is
/// passed through as is.
fn get_application_control_data_flags(
    context: &ServiceContext,
    ctx: &mut HleRequestContext,
    layout: Layout,
) -> ResultCode {
    let _source: u8 = ctx.pop();
    let flag1: u8 = ctx.pop();
    let _flag2: u8 = ctx.pop();
    let application_id: u64 = ctx.pop();
    let nacp = control_nacp(&context.program, application_id);
    let icon: &[u8] = &[];
    let Some(buffer) = ctx
        .writable_buffer(0)
        .filter(|b| b.size as usize >= NACP_SIZE)
    else {
        tracing::error!("GetApplicationControlData: output buffer is too small");
        return RESULT_UNKNOWN;
    };
    if let Err(code) = ctx.write_buffer(0, &control_bytes(nacp, icon, buffer.size as usize)) {
        return code;
    }
    let total = (NACP_SIZE + icon.len()) as u32;
    match layout {
        Layout::Packed => ctx.push(u64::from(total) << 32 | u64::from(flag1)),
        Layout::Flags => {
            ctx.push(0x10001u32);
            ctx.push(if flag1 == 1 { 0x10001u32 } else { 0 });
            ctx.push(total);
        }
    }
    ResultCode::SUCCESS
}

/// GetApplicationDesiredLanguage(u32 supported_languages) -> u8 language.
pub(crate) fn get_application_desired_language(
    context: &ServiceContext,
    ctx: &mut HleRequestContext,
) -> ResultCode {
    let supported: u32 = ctx.pop();
    let system = ApplicationLanguage::from_settings(context.settings.system.language);
    match system.desired(supported) {
        Some(language) => {
            ctx.push(language as u8);
            ResultCode::SUCCESS
        }
        None => {
            tracing::error!(supported, "no supported application language");
            APPLICATION_LANGUAGE_NOT_FOUND
        }
    }
}

/// ConvertApplicationLanguageToLanguageCode(u8 language) -> u64 language code.
pub(crate) fn convert_application_language_to_language_code(
    ctx: &mut HleRequestContext,
) -> ResultCode {
    let raw: u8 = ctx.pop();
    match ApplicationLanguage::from_raw(raw) {
        Some(language) => {
            ctx.push(language.language_code());
            ResultCode::SUCCESS
        }
        None => APPLICATION_LANGUAGE_NOT_FOUND,
    }
}

/// The application ids in input buffer 0.
fn application_ids(ctx: &HleRequestContext) -> Vec<u64> {
    ctx.read_buffer(0)
        .map(|bytes| {
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|id| u64::from_le_bytes(*id))
                .collect()
        })
        .unwrap_or_default()
}

/// The NACP title entry (name 0x200, publisher 0x100) in the system language, else the first
/// one with a name.
fn title_entry(context: &ServiceContext, application_id: u64) -> [u8; TITLE_ENTRY_SIZE] {
    let mut entry = [0u8; TITLE_ENTRY_SIZE];
    let Some(nacp) = context
        .program
        .nacp
        .as_ref()
        .filter(|_| application_id == context.program.program_id)
    else {
        return entry;
    };
    let index = ApplicationLanguage::from_settings(context.settings.system.language) as usize;
    let titles = nacp.titles();
    let title = titles
        .get(index)
        .filter(|t| !t.name.is_empty())
        .or_else(|| titles.iter().find(|t| !t.name.is_empty()));
    if let Some(title) = title {
        let name = &title.name.as_bytes()[..title.name.len().min(0x1ff)];
        let publisher = &title.publisher.as_bytes()[..title.publisher.len().min(0xff)];
        entry[..name.len()].copy_from_slice(name);
        entry[0x200..0x200 + publisher.len()].copy_from_slice(publisher);
    }
    entry
}

/// ListApplicationTitle(app ids, copy handle transfer memory) -> event, IAsyncValue: one title
/// entry per id at the start of the transfer memory.
pub(crate) fn list_application_title(
    context: &Arc<ServiceContext>,
    ctx: &mut HleRequestContext,
) -> ResultCode {
    let ids = application_ids(ctx);
    let memory = ctx.copy_handle::<TransferMemory>(0).ok();
    if let Some(memory) = memory.filter(|_| !ids.is_empty()) {
        for (i, &id) in ids.iter().enumerate() {
            let entry = title_entry(context, id);
            if let Err(code) = memory.write((i * TITLE_ENTRY_SIZE) as u64, &entry) {
                return code;
            }
        }
    }
    push_async_value(context, ctx, 0, (ids.len() * TITLE_ENTRY_SIZE) as i32);
    ResultCode::SUCCESS
}

/// ListApplicationIcon(app ids, copy handle transfer memory) -> event, IAsyncValue: the count,
/// each icon's size, then the icons. No title has an icon yet, so every size is 0.
pub(crate) fn list_application_icon(
    context: &Arc<ServiceContext>,
    ctx: &mut HleRequestContext,
) -> ResultCode {
    let ids = application_ids(ctx);
    let memory = ctx.copy_handle::<TransferMemory>(0).ok();
    let mut length = 0;
    if let Some(memory) = memory.filter(|_| !ids.is_empty()) {
        let mut header = Vec::with_capacity(8 * (ids.len() + 1));
        header.extend_from_slice(&(ids.len() as u64).to_le_bytes());
        header.resize(8 * (ids.len() + 1), 0);
        if let Err(code) = memory.write(0, &header) {
            return code;
        }
        length = header.len();
    }
    push_async_value(context, ctx, 0, length as i32);
    ResultCode::SUCCESS
}

fn push_async_value(
    context: &Arc<ServiceContext>,
    ctx: &mut HleRequestContext,
    offset: i32,
    size: i32,
) {
    let completion = Arc::new(Event::new());
    completion.signal();
    ctx.push_copy_object(KObject::ReadableEvent(completion));
    ctx.push_interface(context.handler(AsyncValue { offset, size }));
}

/// IAsyncValue: the result of a list command, already complete.
struct AsyncValue {
    offset: i32,
    size: i32,
}

impl Service for AsyncValue {
    fn name(&self) -> &str {
        "IAsyncValue"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                ctx.push(i64::from(self.size));
                ResultCode::SUCCESS
            }
            1 => match ctx.write_buffer(0, &self.offset.to_le_bytes()) {
                Ok(_) => ResultCode::SUCCESS,
                Err(code) => code,
            },
            2 | 3 => ResultCode::SUCCESS,
            _ => return None,
        })
    }
}
