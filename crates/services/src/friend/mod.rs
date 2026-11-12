//! friend:a/m/s/u/v and nd:app/nd:sys: an offline account with no friends, blocked users or
//! notifications. chat:u, which the C++ registers here too, belongs to the Switch 2 crew.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;

use crate::ns::Placeholder;
use crate::{Service, ServiceContext};

const THREAD: &str = "friends";
/// account ResultNoNotifications (124-15).
const NO_NOTIFICATIONS: ResultCode = ResultCode::new(124, 15);
const PRESENCE_VIEW_SIZE: usize = 0xe0;
const USER_SETTING_SIZE: usize = 0x800;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("nd:app", THREAD, |_| Placeholder("nd:app"))?;
    context.register("nd:sys", THREAD, |_| Placeholder("nd:sys"))?;
    for name in ["friend:a", "friend:m", "friend:s", "friend:u", "friend:v"] {
        context.register(name, THREAD, move |context| Friend {
            name,
            context: context.clone(),
        })?;
    }
    Ok(())
}

struct Friend {
    name: &'static str,
    context: Arc<ServiceContext>,
}

impl Service for Friend {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let handler = match ctx.command() {
            0 => self.context.handler(FriendService {
                completion: Arc::new(Event::new()),
            }),
            1 => self.context.handler(NotificationService {
                event: Arc::new(Event::new()),
            }),
            _ => return None,
        };
        ctx.push_interface(handler);
        Some(ResultCode::SUCCESS)
    }
}

/// `FriendsUserSetting` for `uid`: presence permission 2, play log permission 5, friend
/// requests accepted, friend code 0000-0000-0000.
fn user_setting(uid: [u8; 16]) -> [u8; USER_SETTING_SIZE] {
    let mut setting = [0u8; USER_SETTING_SIZE];
    setting[..0x10].copy_from_slice(&uid);
    setting[0x10..0x14].copy_from_slice(&2u32.to_le_bytes());
    setting[0x14..0x18].copy_from_slice(&5u32.to_le_bytes());
    setting[0x18..0x20].copy_from_slice(&1u64.to_le_bytes());
    setting[0x20..0x2e].copy_from_slice(b"0000-0000-0000");
    setting[0x40..0x48].copy_from_slice(&99_999_999_999u64.to_le_bytes());
    setting
}

struct FriendService {
    completion: Arc<Event>,
}

impl Service for FriendService {
    fn name(&self) -> &str {
        "IFriendService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                self.completion.signal();
                ctx.push_copy_object(KObject::ReadableEvent(self.completion.clone()));
                ResultCode::SUCCESS
            }
            1 | 10601 | 10610 | 10700 | 20104 | 20701 | 20900 => ResultCode::SUCCESS,
            10101 | 10400 | 20100 | 20101 | 20105 | 20108 | 22010 => {
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            10120 | 10420 => {
                ctx.push(1u8);
                ResultCode::SUCCESS
            }
            20200 => {
                ctx.push(0u32);
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            20600 | 20601 => match ctx.write_buffer(0, &[0; PRESENCE_VIEW_SIZE]) {
                Ok(_) => {
                    ctx.push(0u32);
                    ResultCode::SUCCESS
                }
                Err(code) => code,
            },
            20800 | 20802 => {
                let uid: [u8; 16] = ctx.pop();
                match ctx.write_buffer(0, &user_setting(uid)) {
                    Ok(_) => ResultCode::SUCCESS,
                    Err(code) => code,
                }
            }
            _ => return None,
        })
    }
}

/// INotificationService: no notification ever arrives.
struct NotificationService {
    event: Arc<Event>,
}

impl Service for NotificationService {
    fn name(&self) -> &str {
        "INotificationService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.event.clone()));
                ResultCode::SUCCESS
            }
            1 => ResultCode::SUCCESS,
            2 => NO_NOTIFICATIONS,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::Settings;

    use super::*;
    use crate::testing::{self, Guest, Request};

    fn guest() -> Guest {
        Guest::new(testing::image(), Settings::default(), |context| {
            register(context).unwrap()
        })
    }

    #[test]
    fn an_offline_account_without_friends() {
        let guest = guest();
        for name in [
            "friend:a", "friend:m", "friend:s", "friend:v", "nd:app", "nd:sys",
        ] {
            guest.service(name);
        }
        let friend = guest.service("friend:u");
        let service = guest.call(friend, Request::new(0)).ok().interface();
        let completion = guest.call(service, Request::new(0)).ok().handle();
        assert!(guest.is_signaled(completion));
        for command in [10101, 10400, 20100, 20101, 20105, 20108, 22010] {
            let reply = guest.call(service, Request::new(command)).ok();
            assert_eq!(reply.at::<u32>(0), 0, "command {command}");
        }
        for command in [10120, 10420] {
            let reply = guest
                .call(service, Request::new(command).arg([0u8; 16]))
                .ok();
            assert_eq!(reply.at::<u8>(0), 1);
        }
        let requests = guest.call(service, Request::new(20200).arg([0u8; 16])).ok();
        assert_eq!([requests.at::<u32>(0), requests.at::<u32>(4)], [0, 0]);
        for command in [1, 10601, 10610, 10700, 20104, 20701, 20900] {
            guest.call(service, Request::new(command)).ok();
        }

        let out = guest.heap;
        guest.write(out, &[0xaa; 0x800]);
        guest
            .call(
                service,
                Request::new(20601).arg([0u8; 16]).output(out, 0xe0),
            )
            .ok();
        assert!(guest.read(out, 0xe0).iter().all(|&b| b == 0));

        let uid = [7u8; 16];
        guest
            .call(service, Request::new(20802).arg(uid).output(out, 0x800))
            .ok();
        let setting = guest.read(out, 0x800);
        assert_eq!(setting[..16], uid);
        assert_eq!(setting[0x10..0x18], [2, 0, 0, 0, 5, 0, 0, 0]);
        assert_eq!(&setting[0x20..0x2e], b"0000-0000-0000");
    }

    #[test]
    fn notifications_never_arrive() {
        let guest = guest();
        let friend = guest.service("friend:u");
        let notifications = guest
            .call(friend, Request::new(1).arg([1u8; 16]))
            .ok()
            .interface();
        let event = guest.call(notifications, Request::new(0)).ok().handle();
        assert!(!guest.is_signaled(event));
        guest.call(notifications, Request::new(1)).ok();
        assert_eq!(
            guest.call(notifications, Request::new(2)).result,
            NO_NOTIFICATIONS
        );
    }
}
