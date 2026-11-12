//! The service manager (`sm:`): HLE services register here and the guest looks them up.
//!
//! Both the CMIF and the TIPC flavours (firmware 12+, every Switch 2 title) use the same command
//! ids. `GetServiceHandle` for a name nobody registered yet is deferred until it is registered,
//! as Horizon's sm does; the client thread stays blocked meanwhile (NX2_NOTES row 123).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lichii_common::{HleResult, ResultCode};
use parking_lot::Mutex;

use crate::ipc::{
    ClientSession, DeferredRequest, HleRequestContext, ServiceThread, SessionFactory,
    SessionHandler, UNKNOWN_COMMAND_ID,
};
use crate::object::KObject;

const fn sm(description: u32) -> ResultCode {
    ResultCode::new(21, description)
}

pub const INVALID_CLIENT: ResultCode = sm(2);
pub const ALREADY_REGISTERED: ResultCode = sm(4);
pub const INVALID_SERVICE_NAME: ResultCode = sm(6);
pub const NOT_REGISTERED: ResultCode = sm(7);
pub const NOT_ALLOWED: ResultCode = sm(8);

struct Service {
    thread: ServiceThread,
    factory: SessionFactory,
}

#[derive(Default)]
struct Registry {
    services: HashMap<u64, Service>,
    waiting: Vec<(u64, DeferredRequest)>,
}

/// The registry behind `sm:`.
#[derive(Default)]
pub struct ServiceManager {
    registry: Mutex<Registry>,
}

/// Packs a service name the way sm compares them: up to 8 bytes, zero padded.
fn encode_name(name: &str) -> HleResult<u64> {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > 8 || bytes.contains(&0) {
        return Err(INVALID_SERVICE_NAME);
    }
    let mut raw = [0u8; 8];
    raw[..bytes.len()].copy_from_slice(bytes);
    Ok(u64::from_le_bytes(raw))
}

/// A name as the guest sends it is valid when it is non-empty and nothing follows its first NUL.
fn is_valid_name(name: u64) -> bool {
    let bytes = name.to_le_bytes();
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(8);
    len != 0 && bytes[len..].iter().all(|&b| b == 0)
}

fn display_name(name: u64) -> String {
    let bytes = name.to_le_bytes();
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(8);
    String::from_utf8_lossy(&bytes[..len]).into_owned()
}

impl ServiceManager {
    /// Registers an HLE service. Each `GetServiceHandle` gets a new session whose handler
    /// `factory` makes, served by `thread`. Clients already waiting for the name are answered.
    pub fn register(
        &self,
        name: &str,
        thread: &ServiceThread,
        factory: SessionFactory,
    ) -> HleResult<()> {
        let key = encode_name(name)?;
        let ready = {
            let mut registry = self.registry.lock();
            if registry.services.contains_key(&key) {
                return Err(ALREADY_REGISTERED);
            }
            registry.services.insert(
                key,
                Service {
                    thread: thread.clone(),
                    factory,
                },
            );
            let (ready, waiting) = std::mem::take(&mut registry.waiting)
                .into_iter()
                .partition(|(name, _)| *name == key);
            registry.waiting = waiting;
            ready
        };
        for (_, request) in ready {
            request.retry();
        }
        Ok(())
    }

    pub fn unregister(&self, name: &str) -> HleResult<()> {
        let key = encode_name(name)?;
        self.registry
            .lock()
            .services
            .remove(&key)
            .map(drop)
            .ok_or(NOT_REGISTERED)
    }

    pub fn is_registered(&self, name: &str) -> bool {
        encode_name(name).is_ok_and(|key| self.registry.lock().services.contains_key(&key))
    }

    /// Names clients are waiting for, for diagnostics.
    pub fn waiting_for(&self) -> Vec<String> {
        let registry = self.registry.lock();
        registry
            .waiting
            .iter()
            .map(|(name, _)| display_name(*name))
            .collect()
    }

    /// Connects to `name` or, when it is not registered, defers the request until it is.
    fn get_service(&self, ctx: &mut HleRequestContext, name: u64) -> ResultCode {
        let mut registry = self.registry.lock();
        let Some(service) = registry.services.get(&name) else {
            tracing::info!(service = display_name(name), "waiting for service");
            let request = ctx.defer();
            registry.waiting.push((name, request));
            return ResultCode::SUCCESS;
        };
        let session: Arc<ClientSession> = service.thread.session((service.factory)());
        drop(registry);
        ctx.push_move_object(KObject::ClientSession(session));
        ResultCode::SUCCESS
    }

    /// A new session to the registered service `name`, as `GetService` would return.
    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn open(&self, name: &str) -> Option<Arc<ClientSession>> {
        let registry = self.registry.lock();
        let service = registry.services.get(&encode_name(name).ok()?)?;
        Some(service.thread.session((service.factory)()))
    }

    /// Drops every service and waiting request (their clients get `SessionClosed`). Service
    /// factories usually hold the services' shared state, which holds the kernel, so shutdown
    /// breaks that cycle here.
    pub(crate) fn clear(&self) {
        let registry = std::mem::take(&mut *self.registry.lock());
        drop(registry);
    }

    /// The handler of one `sm:` session.
    pub(crate) fn session_factory(self: &Arc<Self>) -> SessionFactory {
        let manager = self.clone();
        Arc::new(move || {
            Arc::new(SmSession {
                manager: manager.clone(),
                initialized: AtomicBool::new(false),
            }) as Arc<dyn SessionHandler>
        })
    }
}

struct SmSession {
    manager: Arc<ServiceManager>,
    initialized: AtomicBool,
}

impl SessionHandler for SmSession {
    fn name(&self) -> &str {
        "sm:"
    }

    fn handle_request(&self, ctx: &mut HleRequestContext) -> ResultCode {
        match ctx.command() {
            0 => {
                self.initialized.store(true, Ordering::Release);
                ResultCode::SUCCESS
            }
            1 => {
                if !self.initialized.load(Ordering::Acquire) {
                    return INVALID_CLIENT;
                }
                let name: u64 = ctx.pop();
                if !is_valid_name(name) {
                    return INVALID_SERVICE_NAME;
                }
                self.manager.get_service(ctx, name)
            }
            2 | 3 => {
                let name: u64 = ctx.pop();
                tracing::warn!(
                    service = display_name(name),
                    command = ctx.command(),
                    "guest service registration is not supported"
                );
                NOT_ALLOWED
            }
            command => {
                tracing::warn!(command, "unknown sm: command");
                UNKNOWN_COMMAND_ID
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_names() {
        assert_eq!(
            encode_name("sm:").unwrap(),
            u64::from_le_bytes(*b"sm:\0\0\0\0\0")
        );
        assert_eq!(
            encode_name("fsp-srv").unwrap(),
            u64::from_le_bytes(*b"fsp-srv\0")
        );
        assert_eq!(encode_name(""), Err(INVALID_SERVICE_NAME));
        assert_eq!(encode_name("123456789"), Err(INVALID_SERVICE_NAME));
        assert!(is_valid_name(u64::from_le_bytes(*b"appletOE")));
        assert!(!is_valid_name(0));
        assert!(!is_valid_name(u64::from_le_bytes(*b"ab\0c\0\0\0\0")));
        assert_eq!(display_name(u64::from_le_bytes(*b"vi:m\0\0\0\0")), "vi:m");
    }
}
