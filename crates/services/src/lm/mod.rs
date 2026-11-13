//! `lm`: guest log packets, reassembled and written to tracing under the `lm` target.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

const HEADER_SIZE: usize = 0x18;
const HEAD: u8 = 1 << 0;
const TAIL: u8 = 1 << 1;
const ALL_DESTINATIONS: u32 = 0xffff;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("lm", "lm", |context| Lm {
        context: context.clone(),
    })?;
    context.register("lm:get", "lm", |_| LmGet)
}

struct Lm {
    context: Arc<ServiceContext>,
}

impl Service for Lm {
    fn name(&self) -> &str {
        "lm"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push_interface(self.context.handler(Logger::default())),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

struct LmGet;

impl Service for LmGet {
    fn name(&self) -> &str {
        "lm:get"
    }

    fn handle(&self, _ctx: &mut HleRequestContext) -> Option<ResultCode> {
        None
    }
}

/// Packets of one message share the sender and severity; the head packet starts a message and
/// the tail packet ends it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Sender {
    pid: u64,
    tid: u64,
    severity: u8,
    verbosity: u8,
}

/// `ILogger`.
struct Logger {
    pending: Mutex<HashMap<Sender, Vec<u8>>>,
    destination: AtomicU32,
}

impl Default for Logger {
    fn default() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            destination: AtomicU32::new(ALL_DESTINATIONS),
        }
    }
}

impl Logger {
    fn log(&self, packet: &[u8]) {
        let Some((header, payload)) = packet.split_at_checked(HEADER_SIZE) else {
            tracing::error!(
                size = packet.len(),
                "lm: log packet shorter than its header"
            );
            return;
        };
        let u64_at =
            |offset: usize| u64::from_le_bytes(header[offset..offset + 8].try_into().unwrap());
        let flags = header[0x10];
        let sender = Sender {
            pid: u64_at(0),
            tid: u64_at(8),
            severity: header[0x12],
            verbosity: header[0x13],
        };
        let mut pending = self.pending.lock();
        if flags & HEAD != 0 {
            pending.insert(sender, payload.to_vec());
        } else if let Some(message) = pending.get_mut(&sender) {
            message.extend_from_slice(payload);
        } else {
            tracing::error!("lm: log packet continues a message that never started");
            return;
        }
        if flags & TAIL != 0
            && let Some(message) = pending.remove(&sender)
        {
            emit(
                sender,
                self.destination.load(Ordering::Relaxed),
                &parse(&message),
            );
        }
    }
}

impl Service for Logger {
    fn name(&self) -> &str {
        "ILogger"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => match ctx.read_buffer(0) {
                Ok(packet) => self.log(&packet),
                Err(code) => tracing::error!(
                    code = format_args!("{:#x}", code.0),
                    "lm: unreadable log packet"
                ),
            },
            1 => self.destination.store(ctx.pop(), Ordering::Relaxed),
            2 | 3 => {}
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// The fields of one log message.
#[derive(Debug, Default, PartialEq)]
struct Message {
    text: Option<String>,
    line: Option<u32>,
    file: Option<String>,
    function: Option<String>,
    module: Option<String>,
    thread: Option<String>,
    dropped: Option<u64>,
    clock: Option<u64>,
    process: Option<String>,
}

fn parse(data: &[u8]) -> Message {
    let mut message = Message::default();
    let mut offset = 0;
    while offset < data.len() {
        let key = leb128(data, &mut offset);
        let size = leb128(data, &mut offset) as usize;
        let end = offset.saturating_add(size).min(data.len());
        let chunk = &data[offset..end];
        offset = end;
        let text = || (!chunk.is_empty()).then(|| String::from_utf8_lossy(chunk).into_owned());
        let value = || {
            let mut raw = [0u8; 8];
            let len = chunk.len().min(8);
            raw[..len].copy_from_slice(&chunk[..len]);
            u64::from_le_bytes(raw)
        };
        match key {
            2 => message.text = text(),
            3 => message.line = Some(value() as u32),
            4 => message.file = text(),
            5 => message.function = text(),
            6 => message.module = text(),
            7 => message.thread = text(),
            8 => message.dropped = Some(value()),
            9 => message.clock = Some(value()),
            10 => message.process = text(),
            _ => {}
        }
    }
    message
}

fn leb128(data: &[u8], offset: &mut usize) -> u64 {
    let mut value = 0;
    for shift in (0..64).step_by(7) {
        let Some(&byte) = data.get(*offset) else {
            break;
        };
        *offset += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            break;
        }
    }
    value
}

fn emit(sender: Sender, destination: u32, message: &Message) {
    let text = message.text.as_deref().unwrap_or("").trim_end();
    let process = message.process.as_deref().unwrap_or("");
    let module = message.module.as_deref().unwrap_or("");
    let file = message.file.as_deref().unwrap_or("");
    let function = message.function.as_deref().unwrap_or("");
    let line = message.line.unwrap_or(0);
    let thread = message.thread.as_deref().unwrap_or("");
    macro_rules! log {
        ($level:ident) => {
            tracing::$level!(
                target: "lm",
                pid = format_args!("{:#x}", sender.pid),
                tid = format_args!("{:#x}", sender.tid),
                process,
                module,
                file,
                function,
                line,
                thread,
                destination = format_args!("{destination:#x}"),
                "{text}"
            )
        };
    }
    match sender.severity {
        0 => log!(trace),
        1 => log!(info),
        2 => log!(warn),
        _ => log!(error),
    }
}

#[cfg(test)]
mod tests;
