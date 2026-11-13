//! `psm`: battery and charger state. The console reports a full, healthy battery and no charger,
//! which is what the reference answers on a host without a battery, in the units and enum values
//! the guest's `psm` interface defines.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;

use crate::{Service, ServiceContext};

const BATTERY_PERCENTAGE: u32 = 100;
/// `ChargerType::Unplugged`.
const UNPLUGGED: u32 = 0;
/// `BatteryVoltageState::Normal`.
const VOLTAGE_NORMAL: u32 = 3;
/// A full battery, in the per-cent-mille units of `BatteryChargeInfoFields`.
const FULL_PCM: u32 = 100_000;
const CHARGE_INFO_SIZE: usize = 0x54;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("psm", "services", |context| Psm {
        context: context.clone(),
    })
}

struct Psm {
    context: Arc<ServiceContext>,
}

impl Service for Psm {
    fn name(&self) -> &str {
        "psm"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push(BATTERY_PERCENTAGE),
            1 => ctx.push(UNPLUGGED),
            7 => ctx.push_interface(self.context.handler(PsmSession::default())),
            12 => ctx.push(VOLTAGE_NORMAL),
            15 => ctx.push(1.0f64),
            17 => {
                let mut fields = [0u8; CHARGE_INFO_SIZE];
                fields[0x20..0x24].copy_from_slice(&FULL_PCM.to_le_bytes());
                fields[0x28..0x2c].copy_from_slice(&FULL_PCM.to_le_bytes());
                ctx.push_bytes(&fields);
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `IPsmSession`: the power state never changes, so its state change event is never signalled.
#[derive(Default)]
struct PsmSession {
    state_change: Arc<Event>,
}

impl Service for PsmSession {
    fn name(&self) -> &str {
        "IPsmSession"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push_copy_object(KObject::ReadableEvent(self.state_change.clone())),
            1..=4 => {}
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::Settings;
    use lichii_kernel::object::KObject;

    use crate::testing::{Request, SWITCH1_TITLE, booted};

    #[test]
    fn psm_reports_a_full_battery_without_a_charger() {
        let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
        let psm = client.connect("psm").unwrap();
        let call = |command: u32| {
            let reply = client.send(psm, &Request::new(command)).unwrap();
            assert_eq!(reply.result, ResultCode::SUCCESS, "psm {command}");
            reply
        };
        assert_eq!(call(0).pop::<u32>(), 100);
        assert_eq!(call(1).pop::<u32>(), 0);
        assert_eq!(call(12).pop::<u32>(), 3, "normal voltage");
        assert_eq!(call(15).pop::<f64>(), 1.0);
        let fields = call(17).data().to_vec();
        let field =
            |offset: usize| u32::from_le_bytes(fields[offset..offset + 4].try_into().unwrap());
        assert_eq!(field(0x20), 100_000, "full charge in per-cent-mille");
        assert_eq!(field(0x28), 100_000, "no battery wear in per-cent-mille");
        assert_eq!(fields[0x15], 0, "not charging");

        let session = call(7).move_handles[0];
        let reply = client.send(session, &Request::new(0)).unwrap();
        assert!(matches!(
            client.object(reply.copy_handles[0]),
            Some(KObject::ReadableEvent(event)) if !event.is_signaled()
        ));
        let enable = Request::new(2).arg(1u8);
        assert_eq!(
            client.send(session, &enable).unwrap().result,
            ResultCode::SUCCESS
        );
    }
}
