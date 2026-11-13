//! `notif:a`: the alarms an application registers with the notification service. Alarms live
//! until the emulator stops and never ring, and application parameters read back as zeroes, as
//! the reference keeps them.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

/// `nn::notification::AlarmSetting`, whose first two bytes are its `AlarmSettingId`.
const ALARM_SETTING_SIZE: usize = 0x40;
const APPLICATION_PARAMETER_SIZE: usize = 0x400;
/// The reference refuses a new alarm only once more than this many exist.
const MAX_ALARMS: usize = 8;
/// The reference's `ResultUnknown`.
const UNKNOWN: ResultCode = ResultCode(u32::MAX);

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let alarms = Arc::new(Mutex::new(Alarms::default()));
    context.register("notif:a", "services", move |_| Notif {
        alarms: alarms.clone(),
    })
}

type AlarmSetting = [u8; ALARM_SETTING_SIZE];

#[derive(Default)]
struct Alarms {
    settings: Vec<AlarmSetting>,
    next_id: u16,
}

fn alarm_id(setting: &AlarmSetting) -> u16 {
    u16::from_le_bytes([setting[0], setting[1]])
}

/// Input buffer 0 as an `AlarmSetting`, zero-padded when the title sends less.
fn alarm_setting(ctx: &HleRequestContext) -> HleResult<AlarmSetting> {
    let data = ctx.read_buffer(0)?;
    let mut setting = [0; ALARM_SETTING_SIZE];
    let len = data.len().min(ALARM_SETTING_SIZE);
    setting[..len].copy_from_slice(&data[..len]);
    Ok(setting)
}

struct Notif {
    alarms: Arc<Mutex<Alarms>>,
}

impl Notif {
    fn register_alarm(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let mut alarms = self.alarms.lock();
        if alarms.settings.len() > MAX_ALARMS {
            return Err(UNKNOWN);
        }
        let mut setting = alarm_setting(ctx)?;
        let id = alarms.next_id;
        alarms.next_id = id.wrapping_add(1);
        setting[..2].copy_from_slice(&id.to_le_bytes());
        alarms.settings.push(setting);
        ctx.push(id);
        Ok(())
    }

    fn update_alarm(&self, ctx: &HleRequestContext) -> HleResult<()> {
        let setting = alarm_setting(ctx)?;
        let mut alarms = self.alarms.lock();
        if let Some(alarm) = alarms
            .settings
            .iter_mut()
            .find(|alarm| alarm_id(alarm) == alarm_id(&setting))
        {
            *alarm = setting;
        }
        Ok(())
    }

    fn list_alarms(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let alarms = self.alarms.lock();
        let capacity = ctx
            .writable_buffer(0)
            .map_or(0, |buffer| buffer.size as usize);
        let count = alarms.settings.len().min(capacity / ALARM_SETTING_SIZE);
        if count > 0 {
            ctx.write_buffer(0, alarms.settings[..count].as_flattened())?;
        }
        ctx.push(count as i32);
        Ok(())
    }

    fn load_application_parameter(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let id = ctx.pop::<u16>();
        if !self
            .alarms
            .lock()
            .settings
            .iter()
            .any(|alarm| alarm_id(alarm) == id)
        {
            return Err(UNKNOWN);
        }
        ctx.write_buffer(0, &[0; APPLICATION_PARAMETER_SIZE])?;
        ctx.push(APPLICATION_PARAMETER_SIZE as u32);
        Ok(())
    }
}

impl Service for Notif {
    fn name(&self) -> &str {
        "notif:a"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let result = match ctx.command() {
            500 => self.register_alarm(ctx),
            510 => self.update_alarm(ctx),
            520 => self.list_alarms(ctx),
            530 => self.load_application_parameter(ctx),
            540 => {
                let id = ctx.pop::<u16>();
                self.alarms
                    .lock()
                    .settings
                    .retain(|alarm| alarm_id(alarm) != id);
                Ok(())
            }
            1000 => Ok(()),
            _ => return None,
        };
        Some(result.err().unwrap_or(ResultCode::SUCCESS))
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::Settings;
    use lichii_kernel::ipc::UNKNOWN_COMMAND_ID;

    use crate::testing::{Request, SWITCH1_TITLE, SWITCH2_TITLE, booted};

    fn setting(id: u16, kind: u8) -> Vec<u8> {
        let mut setting = vec![0; 0x40];
        setting[..2].copy_from_slice(&id.to_le_bytes());
        setting[2] = kind;
        setting[0x18..0x20].copy_from_slice(&0x0100_0000_0000_1000u64.to_le_bytes());
        setting
    }

    #[test]
    fn alarms_are_registered_listed_updated_and_deleted() {
        let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
        let notif = client.connect("notif:a").unwrap();
        let ok = |request: &Request| {
            let reply = client.send(notif, request).unwrap();
            assert_eq!(reply.result, ResultCode::SUCCESS);
            reply
        };
        ok(&Request::new(1000).send_pid().arg(0u64));
        let parameter = client.alloc(&[7; 0x10]);
        for (index, kind) in [1u8, 2].into_iter().enumerate() {
            let alarm = client.alloc(&setting(0xffff, kind));
            let register = Request::new(500)
                .buffer_a(alarm, 0x40)
                .buffer_a(parameter, 0x10);
            assert_eq!(ok(&register).pop::<u16>(), index as u16, "ids count up");
        }

        let alarm = client.alloc(&setting(1, 9));
        ok(&Request::new(510)
            .buffer_a(alarm, 0x40)
            .buffer_a(parameter, 0x10));
        let list = |capacity: u64| {
            let out = client.alloc(&vec![0xee; capacity as usize]);
            let count = ok(&Request::new(520).buffer_b(out, capacity)).pop::<i32>();
            (count, client.read(out, capacity as usize))
        };
        let (count, out) = list(0x80);
        assert_eq!(count, 2);
        assert_eq!(out[..2], 0u16.to_le_bytes());
        assert_eq!(out[2], 1);
        assert_eq!(out[0x40..0x42], 1u16.to_le_bytes());
        assert_eq!(out[0x42], 9, "updated");
        assert_eq!(list(0x40).0, 1, "limited by the buffer");

        let out = client.alloc(&[0xee; 0x400]);
        let mut reply = ok(&Request::new(530).arg(1u16).buffer_b(out, 0x400));
        assert_eq!(reply.pop::<u32>(), 0x400);
        assert!(client.read(out, 0x400).iter().all(|&byte| byte == 0));

        ok(&Request::new(540).arg(0u16));
        assert_eq!(list(0x80).0, 1);
        let reply = client
            .send(notif, &Request::new(530).arg(0u16).buffer_b(out, 0x400))
            .unwrap();
        assert_eq!(reply.result, ResultCode(u32::MAX), "deleted alarm");
    }

    #[test]
    fn a_ninth_alarm_is_the_last_one_accepted() {
        let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
        let notif = client.connect("notif:a").unwrap();
        let alarm = client.alloc(&setting(0, 0));
        let parameter = client.alloc(&[0; 4]);
        let register = Request::new(500)
            .buffer_a(alarm, 0x40)
            .buffer_a(parameter, 4);
        for _ in 0..9 {
            let reply = client.send(notif, &register).unwrap();
            assert_eq!(reply.result, ResultCode::SUCCESS);
        }
        let reply = client.send(notif, &register).unwrap();
        assert_eq!(reply.result, ResultCode(u32::MAX));
    }

    #[test]
    fn unhandled_commands_are_stubbed_only_for_switch2_titles() {
        for (title, expected) in [
            (SWITCH1_TITLE, UNKNOWN_COMMAND_ID),
            (SWITCH2_TITLE, ResultCode::SUCCESS),
        ] {
            let client = booted(title, Settings::default(), super::register);
            let notif = client.connect("notif:a").unwrap();
            assert_eq!(
                client.send(notif, &Request::new(1010)).unwrap().result,
                expected
            );
        }
    }
}
