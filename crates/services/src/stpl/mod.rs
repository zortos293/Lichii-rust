//! `stpl:u`, Switch 2 stream play (GameShare). Command 0 opens the user service the SDK keeps for
//! the process; without an object the first `nn::streamplay` call jumps through null. Its command 300
//! hands out the guest HID shared memory, which stays empty: no guest ever connects. Both are
//! `nx2:` interfaces, so the other commands get the Switch 2 auto-stub, as in the reference.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory::Pool;
use lichii_kernel::memory_objects::SharedMemory;
use lichii_kernel::object::KObject;
use lichii_memory::MemoryPermission;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

/// Size of the guest HID shared memory command 300 returns.
const GUEST_HID_SIZE: u64 = 0x9b000;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("stpl:u", "services", |context| StreamPlay {
        context: context.clone(),
    })
}

struct StreamPlay {
    context: Arc<ServiceContext>,
}

impl Service for StreamPlay {
    fn name(&self) -> &str {
        "nx2:stpl:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                tracing::warn!("(stubbed) stpl:u GetUserService");
                ctx.push_interface(self.context.handler(StreamPlayUser {
                    context: self.context.clone(),
                    guest_hid: Mutex::new(None),
                }));
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

struct StreamPlayUser {
    context: Arc<ServiceContext>,
    guest_hid: Mutex<Option<Arc<SharedMemory>>>,
}

impl StreamPlayUser {
    /// 300: (u64 size, u64 offset) and the guest HID shared memory, made on the first call.
    fn guest_hid_shared_memory(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let mut guest_hid = self.guest_hid.lock();
        let memory = match guest_hid.as_ref() {
            Some(memory) => memory.clone(),
            None => {
                let memory = match SharedMemory::new(
                    self.context.kernel.memory_manager().clone(),
                    None,
                    Pool::System,
                    GUEST_HID_SIZE,
                    MemoryPermission::READ_WRITE,
                    MemoryPermission::READ,
                ) {
                    Ok(memory) => Arc::new(memory),
                    Err(code) => return code,
                };
                tracing::info!(
                    size = GUEST_HID_SIZE,
                    "GameShare guest HID shared memory, no guests connected"
                );
                guest_hid.insert(memory).clone()
            }
        };
        ctx.push([GUEST_HID_SIZE, 0u64]);
        ctx.push_copy_object(KObject::SharedMemory(memory));
        ResultCode::SUCCESS
    }
}

impl Service for StreamPlayUser {
    fn name(&self) -> &str {
        "nx2:stpl:u::0"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            300 => self.guest_hid_shared_memory(ctx),
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::Settings;
    use lichii_kernel::object::KObject;
    use lichii_kernel::testing::Request;

    use super::GUEST_HID_SIZE;
    use crate::testing::{SWITCH2_TITLE, context};

    fn name(service: &str) -> u64 {
        let mut raw = [0u8; 8];
        raw[..service.len()].copy_from_slice(service.as_bytes());
        u64::from_le_bytes(raw)
    }

    #[test]
    fn stpl_u_opens_a_user_service_with_the_guest_hid_shared_memory() {
        let (context, client) = context(SWITCH2_TITLE, Settings::default());
        super::register(&context).unwrap();
        let stpl = client.connect("stpl:u").expect("stpl:u registered");
        let reply = client.send(stpl, &Request::tipc(0).arg(0x12u32)).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        let user = reply.move_handles[0];

        let mut first = client.send(user, &Request::tipc(300)).unwrap();
        assert_eq!(first.result, ResultCode::SUCCESS);
        assert_eq!(first.pop::<[u64; 2]>(), [GUEST_HID_SIZE, 0]);
        let Some(KObject::SharedMemory(memory)) = client.object(first.copy_handles[0]) else {
            panic!("no shared memory");
        };
        assert_eq!(memory.size(), GUEST_HID_SIZE);
        let again = client.send(user, &Request::tipc(300)).unwrap();
        let Some(KObject::SharedMemory(same)) = client.object(again.copy_handles[0]) else {
            panic!("no shared memory");
        };
        assert!(std::sync::Arc::ptr_eq(&memory, &same));
    }

    #[test]
    fn switch2_titles_get_auto_stubbed_sessions_for_services_nobody_registers() {
        let (context, client) = context(SWITCH2_TITLE, Settings::default());
        crate::stub_missing_switch2_services(&context).unwrap();
        let sm = client.connect_port("sm:").expect("sm: port");
        let init = client.send(sm, &Request::tipc(0).send_pid()).unwrap();
        assert_eq!(init.result, ResultCode::SUCCESS);
        for service in ["mnpp:app", "htc:tenv"] {
            let reply = client
                .send(sm, &Request::tipc(1).arg(name(service)))
                .unwrap();
            assert_eq!(reply.result, ResultCode::SUCCESS, "{service}");
            let session = reply.move_handles[0];
            let mut stubbed = client.send(session, &Request::tipc(7)).unwrap();
            assert_eq!(stubbed.result, ResultCode::SUCCESS, "{service} 7");
            assert_eq!(stubbed.pop::<u64>(), 0);
        }
    }
}
