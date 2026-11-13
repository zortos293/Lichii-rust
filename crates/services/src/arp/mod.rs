//! arp:r and arp:w: the launch and control properties of running applications. The running
//! title is registered from its `ProgramImage` when the service starts, as am does when the C++
//! creates the process.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;

use crate::{Service, ServiceContext};

const INVALID_PROCESS_ID: ResultCode = ResultCode::new(157, 31);
const ALREADY_BOUND: ResultCode = ResultCode::new(157, 42);
const PROCESS_ID_NOT_REGISTERED: ResultCode = ResultCode::new(157, 102);
/// `FileSys::StorageId::Host`: titles loaded from the host's file system.
const STORAGE_HOST: u8 = 1;
const STORAGE_NONE: u8 = 0;

/// `ApplicationLaunchProperty`: u64 title id, u32 version, u8 base storage, u8 update storage,
/// u8 program index, u8 reserved.
type LaunchProperty = [u8; 0x10];

struct Entry {
    launch: LaunchProperty,
    control: Vec<u8>,
}

/// The registrations every arp session shares (the C++ ARPManager).
#[derive(Default)]
struct Manager {
    entries: Mutex<HashMap<u64, Entry>>,
}

impl Manager {
    fn register(&self, title_id: u64, launch: LaunchProperty, control: Vec<u8>) -> ResultCode {
        if title_id == 0 {
            return INVALID_PROCESS_ID;
        }
        let mut entries = self.entries.lock().unwrap();
        if entries.contains_key(&title_id) {
            return ALREADY_BOUND;
        }
        entries.insert(title_id, Entry { launch, control });
        ResultCode::SUCCESS
    }

    fn unregister(&self, title_id: u64) -> ResultCode {
        if title_id == 0 {
            return INVALID_PROCESS_ID;
        }
        match self.entries.lock().unwrap().remove(&title_id) {
            Some(_) => ResultCode::SUCCESS,
            None => PROCESS_ID_NOT_REGISTERED,
        }
    }

    fn with<T>(&self, title_id: u64, f: impl FnOnce(&Entry) -> T) -> HleResult<T> {
        if title_id == 0 {
            return Err(INVALID_PROCESS_ID);
        }
        let entries = self.entries.lock().unwrap();
        entries
            .get(&title_id)
            .map(f)
            .ok_or(PROCESS_ID_NOT_REGISTERED)
    }
}

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let manager = Arc::new(Manager::default());
    let program = &context.program;
    let mut launch = [0u8; 0x10];
    launch[..8].copy_from_slice(&program.program_id.to_le_bytes());
    launch[12] = STORAGE_HOST;
    launch[13] = STORAGE_NONE;
    let control = program
        .nacp
        .as_ref()
        .map(|nacp| nacp.raw().to_vec())
        .unwrap_or_default();
    let _ = manager.register(program.program_id, launch, control);

    let reader = manager.clone();
    context.register("arp:r", "glue", move |context| Reader {
        context: context.clone(),
        manager: reader.clone(),
    })?;
    context.register("arp:w", "glue", move |context| Writer {
        context: context.clone(),
        manager: manager.clone(),
    })
}

/// The program id of process `process_id`.
fn title_of(context: &ServiceContext, process_id: u64) -> Option<u64> {
    context
        .kernel
        .processes()
        .iter()
        .find(|p| p.id() == process_id)
        .map(|p| p.program_id())
}

struct Reader {
    context: Arc<ServiceContext>,
    manager: Arc<Manager>,
}

impl Reader {
    fn title(&self, ctx: &mut HleRequestContext, by_process: bool) -> HleResult<u64> {
        let id: u64 = ctx.pop();
        if by_process {
            title_of(&self.context, id).ok_or(PROCESS_ID_NOT_REGISTERED)
        } else {
            Ok(id)
        }
    }

    fn launch_property(&self, ctx: &mut HleRequestContext, by_process: bool) -> HleResult<()> {
        let title_id = self.title(ctx, by_process)?;
        let launch = self.manager.with(title_id, |entry| entry.launch)?;
        ctx.push(launch);
        Ok(())
    }

    fn control_property(&self, ctx: &mut HleRequestContext, by_process: bool) -> HleResult<()> {
        let title_id = self.title(ctx, by_process)?;
        let control = self.manager.with(title_id, |entry| entry.control.clone())?;
        ctx.write_buffer(0, &control)?;
        Ok(())
    }
}

impl Service for Reader {
    fn name(&self) -> &str {
        "arp:r"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let result = match ctx.command() {
            0 => self.launch_property(ctx, true),
            1 => self.launch_property(ctx, false),
            2 => self.control_property(ctx, true),
            3 => self.control_property(ctx, false),
            _ => return None,
        };
        Some(result.err().unwrap_or(ResultCode::SUCCESS))
    }
}

struct Writer {
    context: Arc<ServiceContext>,
    manager: Arc<Manager>,
}

impl Service for Writer {
    fn name(&self) -> &str {
        "arp:w"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                let registrar = Registrar {
                    context: self.context.clone(),
                    manager: self.manager.clone(),
                    state: Mutex::new(RegistrarState::default()),
                };
                ctx.push_interface(self.context.handler(registrar));
                ResultCode::SUCCESS
            }
            1 => {
                let process_id: u64 = ctx.pop();
                if process_id == 0 {
                    return Some(INVALID_PROCESS_ID);
                }
                match title_of(&self.context, process_id) {
                    Some(title_id) => self.manager.unregister(title_id),
                    None => PROCESS_ID_NOT_REGISTERED,
                }
            }
            _ => return None,
        })
    }
}

#[derive(Default)]
struct RegistrarState {
    issued: bool,
    launch: LaunchProperty,
    control: Vec<u8>,
}

/// IRegistrar: collects a launch and control property, then registers them for a process.
struct Registrar {
    context: Arc<ServiceContext>,
    manager: Arc<Manager>,
    state: Mutex<RegistrarState>,
}

impl Service for Registrar {
    fn name(&self) -> &str {
        "IRegistrar"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let mut state = self.state.lock().unwrap();
        Some(match ctx.command() {
            0 => {
                let process_id: u64 = ctx.pop();
                if process_id == 0 {
                    return Some(INVALID_PROCESS_ID);
                }
                if state.issued {
                    return Some(ALREADY_BOUND);
                }
                if let Some(title_id) = title_of(&self.context, process_id) {
                    let control = std::mem::take(&mut state.control);
                    let _ = self.manager.register(title_id, state.launch, control);
                }
                state.issued = true;
                ResultCode::SUCCESS
            }
            1 | 2 if state.issued => ALREADY_BOUND,
            1 => {
                state.launch = ctx.pop();
                ResultCode::SUCCESS
            }
            2 => match ctx.read_buffer(0) {
                Ok(control) => {
                    state.control = control;
                    ResultCode::SUCCESS
                }
                Err(code) => code,
            },
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::Settings;

    use super::*;
    use crate::testing::{self, Guest, PROGRAM_ID, ReplyExt, Request};

    fn guest() -> Guest {
        let mut image = testing::image();
        image.nacp = Some(testing::nacp("Arp", 1, 0));
        Guest::new(image, Settings::default(), |context| {
            register(context).unwrap()
        })
    }

    #[test]
    fn the_running_title_is_registered() {
        let guest = guest();
        let arp = guest.service("arp:r");
        let pid = guest.process.id();
        let mut expected = [0u8; 16];
        expected[..8].copy_from_slice(&PROGRAM_ID.to_le_bytes());
        expected[12] = STORAGE_HOST;
        for request in [Request::new(0).arg(pid), Request::new(1).arg(PROGRAM_ID)] {
            let launch = guest.call(arp, request).ok();
            assert_eq!(launch.at::<[u8; 16]>(0), expected);
        }

        let out = guest.heap;
        for request in [Request::new(2).arg(pid), Request::new(3).arg(PROGRAM_ID)] {
            guest.write(out, &[0; 4]);
            guest.call(arp, request.buffer_b(out, 0x4000)).ok();
            assert_eq!(&guest.read(out, 3), b"Arp");
        }

        assert_eq!(
            guest.call(arp, Request::new(0).arg(pid + 100)).result,
            PROCESS_ID_NOT_REGISTERED
        );
        assert_eq!(
            guest.call(arp, Request::new(1).arg(0u64)).result,
            INVALID_PROCESS_ID
        );
        assert_eq!(
            guest.call(arp, Request::new(1).arg(7u64)).result,
            PROCESS_ID_NOT_REGISTERED
        );
    }

    #[test]
    fn a_registrar_registers_once() {
        let guest = guest();
        let (reader, writer) = (guest.service("arp:r"), guest.service("arp:w"));
        let pid = guest.process.id();
        guest.call(writer, Request::new(1).arg(pid)).ok();
        assert_eq!(
            guest.call(reader, Request::new(0).arg(pid)).result,
            PROCESS_ID_NOT_REGISTERED
        );
        assert_eq!(
            guest.call(writer, Request::new(1).arg(0u64)).result,
            INVALID_PROCESS_ID
        );

        let registrar = guest.call(writer, Request::new(0)).ok().interface();
        let mut launch = [0u8; 16];
        launch[..8].copy_from_slice(&PROGRAM_ID.to_le_bytes());
        launch[8] = 7;
        guest.call(registrar, Request::new(1).arg(launch)).ok();
        let control = guest.heap;
        guest.write(control, b"control");
        guest
            .call(registrar, Request::new(2).buffer_a(control, 7))
            .ok();
        assert_eq!(
            guest.call(registrar, Request::new(0).arg(0u64)).result,
            INVALID_PROCESS_ID
        );
        guest.call(registrar, Request::new(0).arg(pid)).ok();
        assert_eq!(
            guest.call(registrar, Request::new(1).arg(launch)).result,
            ALREADY_BOUND
        );

        let read = guest.call(reader, Request::new(0).arg(pid)).ok();
        assert_eq!(read.at::<[u8; 16]>(0), launch);
        let out = guest.heap + 0x1000;
        guest
            .call(reader, Request::new(2).arg(pid).buffer_b(out, 0x100))
            .ok();
        assert_eq!(&guest.read(out, 7), b"control");
    }
}
