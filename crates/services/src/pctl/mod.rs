//! `pctl`, `pctl:a`, `pctl:r` and `pctl:s`: parental controls, with no restrictions set.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

const MODULE: u32 = 142;
const NO_CAPABILITY: ResultCode = ResultCode::new(MODULE, 131);
const NO_RESTRICTION_ENABLED: ResultCode = ResultCode::new(MODULE, 181);

const APPLICATION: u32 = 1 << 0;
const SNS_POST: u32 = 1 << 1;
const RECOVERY: u32 = 1 << 6;
const STATUS: u32 = 1 << 8;
const STEREO_VISION: u32 = 1 << 9;
const SYSTEM: u32 = 1 << 15;

const PLAY_TIMER_NOT_CONFIGURED: u64 = 7;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let services = [
        ("pctl", APPLICATION | SNS_POST | STATUS | STEREO_VISION),
        ("pctl:a", 0),
        ("pctl:r", 0),
        ("pctl:s", 0),
    ];
    for (name, capability) in services {
        context.register(name, "pctl", move |context| Factory {
            name,
            capability,
            context: context.clone(),
        })?;
    }
    Ok(())
}

/// `IParentalControlServiceFactory`.
struct Factory {
    name: &'static str,
    capability: u32,
    context: Arc<ServiceContext>,
}

impl Service for Factory {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 | 1 => {
                ctx.push_interface(self.context.handler(ParentalControl::new(self.capability)))
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `IParentalControlService`. No PIN is set, so every permission check passes.
struct ParentalControl {
    capability: u32,
    play_timer_settings: Mutex<[u32; 17]>,
    synchronization: Arc<Event>,
    request_suspension: Arc<Event>,
    unlinked: Arc<Event>,
}

impl ParentalControl {
    fn new(capability: u32) -> Self {
        Self {
            capability,
            play_timer_settings: Mutex::new([0; 17]),
            synchronization: Arc::new(Event::new()),
            request_suspension: Arc::new(Event::new()),
            unlinked: Arc::new(Event::new()),
        }
    }

    fn has(&self, capability: u32) -> bool {
        self.capability & capability != 0
    }
}

impl Service for ParentalControl {
    fn name(&self) -> &str {
        "IParentalControlService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            1 if !self.has(APPLICATION | SYSTEM) => return Some(NO_CAPABILITY),
            1 | 1001..=1005 | 1013 | 1017 | 1018 | 1019 | 1064 | 1451 | 1452 => {}
            1006 | 1010 | 1403 | 1453 | 1455 | 1458 => ctx.push(0u8),
            1031 if !self.has(STATUS | RECOVERY) => {
                ctx.push(0u8);
                return Some(NO_CAPABILITY);
            }
            1031 => ctx.push(0u8),
            1032 | 1206 => ctx.push(0u32),
            1035 => ctx.push([0u8; 3]),
            1039 => ctx.push(4i32),
            1061 if !self.has(STEREO_VISION) => return Some(NO_CAPABILITY),
            1061 => return Some(NO_RESTRICTION_ENABLED),
            1062 if !self.has(STEREO_VISION) => {
                ctx.push(0u8);
                return Some(NO_CAPABILITY);
            }
            1062 => ctx.push(0u8),
            1063 if !self.has(STEREO_VISION) => return Some(NO_CAPABILITY),
            1063 => {}
            1065 => ctx.push(1u8),
            1432 => ctx.push_copy_object(KObject::ReadableEvent(self.synchronization.clone())),
            1454 => ctx.push(i32::MAX),
            1456 => ctx.push([0u32; 13]),
            1457 => ctx.push_copy_object(KObject::ReadableEvent(self.request_suspension.clone())),
            1459 | 1460 => ctx.push([PLAY_TIMER_NOT_CONFIGURED, 0, 0]),
            1473 => ctx.push_copy_object(KObject::ReadableEvent(self.unlinked.clone())),
            145601 => ctx.push(*self.play_timer_settings.lock()),
            195101 => *self.play_timer_settings.lock() = ctx.pop(),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

#[cfg(test)]
mod tests;
