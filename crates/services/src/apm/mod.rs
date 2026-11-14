//! `apm`, `apm:am`, `apm:p` and `apm:sys`: performance mode and configuration.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

const DEFAULT_CONFIGURATION: u32 = 0x0002_0003;
const NORMAL: i32 = 0;
const BOOST: i32 = 1;

const CONFIGURATIONS: [u32; 16] = [
    0x0001_0000,
    0x0001_0001,
    0x0001_0002,
    0x0002_0000,
    0x0002_0001,
    0x0002_0002,
    0x0002_0003,
    0x0002_0004,
    0x0002_0005,
    0x0002_0006,
    0x9222_0007,
    0x9222_0008,
    0x9222_0009,
    0x9222_000a,
    0x9222_000b,
    0x9222_000c,
];

const BOOST_MODE_CONFIGURATIONS: [u32; 3] = [DEFAULT_CONFIGURATION, 0x9222_0009, 0x9222_000b];

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let controller = Arc::new(Controller {
        docked: context.settings.system.docked,
        configurations: Mutex::new([DEFAULT_CONFIGURATION; 2]),
    });
    let _ = context.apm.set(controller.clone());
    for name in ["apm", "apm:am", "apm:p"] {
        let controller = controller.clone();
        context.register(name, "apm", move |context| Apm {
            name,
            context: context.clone(),
            controller: controller.clone(),
        })?;
    }
    context.register("apm:sys", "apm", move |context| ApmSys {
        context: context.clone(),
        controller: controller.clone(),
    })
}

/// The configuration of each performance mode. The mode itself follows the docked setting.
pub(crate) struct Controller {
    docked: bool,
    configurations: Mutex<[u32; 2]>,
}

impl Controller {
    fn mode(&self) -> i32 {
        if self.docked { BOOST } else { NORMAL }
    }

    fn set_configuration(&self, mode: i32, configuration: u32) {
        let Some(slot) = usize::try_from(mode).ok().filter(|&mode| mode < 2) else {
            tracing::warn!(mode, "apm: invalid performance mode");
            return;
        };
        if !CONFIGURATIONS.contains(&configuration) {
            tracing::warn!(
                configuration = format_args!("{configuration:#x}"),
                "apm: unknown performance configuration"
            );
            return;
        }
        self.configurations.lock()[slot] = configuration;
    }

    /// Sets the boost mode's configuration from a CPU boost mode, like apm:sys 6 and am's
    /// SetCpuBoostMode.
    pub(crate) fn set_cpu_boost_mode(&self, mode: u32) {
        match BOOST_MODE_CONFIGURATIONS.get(mode as usize) {
            Some(&configuration) => self.set_configuration(BOOST, configuration),
            None => tracing::warn!(mode, "apm: invalid CPU boost mode"),
        }
    }

    fn configuration(&self, mode: i32) -> u32 {
        usize::try_from(mode)
            .ok()
            .and_then(|mode| self.configurations.lock().get(mode).copied())
            .unwrap_or(DEFAULT_CONFIGURATION)
    }
}

struct Apm {
    name: &'static str,
    context: Arc<ServiceContext>,
    controller: Arc<Controller>,
}

impl Service for Apm {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push_interface(self.context.handler(Session(self.controller.clone()))),
            1 => ctx.push(self.controller.mode()),
            6 => ctx.push(0u8),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

struct ApmSys {
    context: Arc<ServiceContext>,
    controller: Arc<Controller>,
}

impl Service for ApmSys {
    fn name(&self) -> &str {
        "apm:sys"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            1 => ctx.push_interface(self.context.handler(Session(self.controller.clone()))),
            6 => self.controller.set_cpu_boost_mode(ctx.pop()),
            7 => {
                let configuration = self.controller.configuration(self.controller.mode());
                ctx.push(configuration);
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `ISession`.
struct Session(Arc<Controller>);

impl Service for Session {
    fn name(&self) -> &str {
        "ISession"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => {
                let mode: i32 = ctx.pop();
                let configuration: u32 = ctx.pop();
                self.0.set_configuration(mode, configuration);
            }
            1 => {
                let mode: i32 = ctx.pop();
                ctx.push(self.0.configuration(mode));
            }
            2 => {
                let enabled: u8 = ctx.pop();
                tracing::debug!(enabled, "apm: SetCpuOverclockEnabled");
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

#[cfg(test)]
mod tests;
