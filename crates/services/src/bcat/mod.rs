//! bcat:a/m/u/s: BCAT with the C++ null backend. Syncs finish at once with nothing delivered,
//! so the delivery cache has no directories or files.

use std::sync::{Arc, Mutex};

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;

use crate::{Service, ServiceContext};

const INVALID_ARGUMENT: ResultCode = ResultCode::new(122, 1);
const FAILED_OPEN_ENTITY: ResultCode = ResultCode::new(122, 2);
const NO_OPEN_ENTRY: ResultCode = ResultCode::new(122, 7);
/// `DeliveryCacheProgressStatus::Done`.
const STATUS_DONE: i32 = 9;
const PROGRESS_SIZE: usize = 0x200;
const NAME_SIZE: usize = 0x20;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    for name in ["bcat:a", "bcat:m", "bcat:u", "bcat:s"] {
        context.register(name, "bcat", move |context| ServiceCreator {
            name,
            context: context.clone(),
        })?;
    }
    Ok(())
}

/// Whether `name` is a NUL-terminated, non-empty run of letters, digits, `_` and `extra`.
fn valid_name(name: &[u8; NAME_SIZE], extra: u8) -> bool {
    let nulls = name.iter().filter(|&&c| c == 0).count();
    let bad = name
        .iter()
        .any(|&c| !c.is_ascii_alphanumeric() && c != b'_' && c != extra && c != 0);
    nulls != NAME_SIZE && nulls != 0 && !bad && name[NAME_SIZE - 1] == 0
}

struct ServiceCreator {
    name: &'static str,
    context: Arc<ServiceContext>,
}

impl Service for ServiceCreator {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let handler = match ctx.command() {
            0 => self.context.handler(BcatService {
                context: self.context.clone(),
                normal: Arc::new(Progress::default()),
                directory: Arc::new(Progress::default()),
            }),
            1 | 2 => self.context.handler(DeliveryCacheStorage {
                context: self.context.clone(),
            }),
            _ => return None,
        };
        ctx.push_interface(handler);
        Some(ResultCode::SUCCESS)
    }
}

/// One sync type's progress (`DeliveryCacheProgressImpl`) and update event.
#[derive(Default)]
struct Progress {
    event: Arc<Event>,
    state: Mutex<Option<[u8; PROGRESS_SIZE]>>,
}

impl Progress {
    /// What the null backend does: the download is done, successfully, with nothing in it.
    fn finish(&self) {
        let mut state = [0u8; PROGRESS_SIZE];
        state[..4].copy_from_slice(&STATUS_DONE.to_le_bytes());
        *self.state.lock().unwrap() = Some(state);
        self.event.signal();
    }

    fn snapshot(&self) -> [u8; PROGRESS_SIZE] {
        self.state.lock().unwrap().unwrap_or([0; PROGRESS_SIZE])
    }
}

struct BcatService {
    context: Arc<ServiceContext>,
    normal: Arc<Progress>,
    directory: Arc<Progress>,
}

impl BcatService {
    fn sync(&self, ctx: &mut HleRequestContext, progress: &Arc<Progress>) -> ResultCode {
        progress.finish();
        let service = DeliveryCacheProgress {
            progress: progress.clone(),
        };
        ctx.push_interface(self.context.handler(service));
        ResultCode::SUCCESS
    }
}

impl Service for BcatService {
    fn name(&self) -> &str {
        "IBcatService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            10100 => self.sync(ctx, &self.normal),
            10101 => self.sync(ctx, &self.directory),
            30100 => {
                let application_id: u64 = ctx.pop();
                let passphrase_len = ctx.readable_buffer(0).map_or(0, |b| b.size);
                if application_id == 0 || passphrase_len > 0x40 {
                    INVALID_ARGUMENT
                } else {
                    ResultCode::SUCCESS
                }
            }
            30300 => ResultCode::SUCCESS,
            90201 => {
                let application_id: u64 = ctx.pop();
                if application_id == 0 {
                    INVALID_ARGUMENT
                } else {
                    ResultCode::SUCCESS
                }
            }
            _ => return None,
        })
    }
}

struct DeliveryCacheProgress {
    progress: Arc<Progress>,
}

impl Service for DeliveryCacheProgress {
    fn name(&self) -> &str {
        "IDeliveryCacheProgressService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.progress.event.clone()));
                ResultCode::SUCCESS
            }
            1 => match ctx.write_buffer(0, &self.progress.snapshot()) {
                Ok(_) => ResultCode::SUCCESS,
                Err(code) => code,
            },
            _ => return None,
        })
    }
}

/// IDeliveryCacheStorageService over an empty delivery cache.
struct DeliveryCacheStorage {
    context: Arc<ServiceContext>,
}

impl Service for DeliveryCacheStorage {
    fn name(&self) -> &str {
        "IDeliveryCacheStorageService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                ctx.push_interface(self.context.handler(DeliveryCacheFile));
                ResultCode::SUCCESS
            }
            1 => {
                ctx.push_interface(self.context.handler(DeliveryCacheDirectory));
                ResultCode::SUCCESS
            }
            10 => {
                ctx.push(0i32);
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

/// IDeliveryCacheDirectoryService: every directory is missing.
struct DeliveryCacheDirectory;

impl Service for DeliveryCacheDirectory {
    fn name(&self) -> &str {
        "IDeliveryCacheDirectoryService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                let name: [u8; NAME_SIZE] = ctx.pop();
                if valid_name(&name, b'-') {
                    FAILED_OPEN_ENTITY
                } else {
                    INVALID_ARGUMENT
                }
            }
            1 | 2 => NO_OPEN_ENTRY,
            _ => return None,
        })
    }
}

/// IDeliveryCacheFileService: every file is missing.
struct DeliveryCacheFile;

impl Service for DeliveryCacheFile {
    fn name(&self) -> &str {
        "IDeliveryCacheFileService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                let directory: [u8; NAME_SIZE] = ctx.pop();
                let file: [u8; NAME_SIZE] = ctx.pop();
                if valid_name(&directory, b'-') && valid_name(&file, b'.') {
                    FAILED_OPEN_ENTITY
                } else {
                    INVALID_ARGUMENT
                }
            }
            1..=3 => NO_OPEN_ENTRY,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::Settings;

    use super::*;
    use crate::testing::{self, Guest, PROGRAM_ID, ReplyExt, Request};

    fn name(text: &str) -> [u8; NAME_SIZE] {
        let mut name = [0u8; NAME_SIZE];
        name[..text.len()].copy_from_slice(text.as_bytes());
        name
    }

    fn guest() -> Guest {
        Guest::new(testing::image(), Settings::default(), |context| {
            register(context).unwrap()
        })
    }

    #[test]
    fn syncs_finish_with_nothing_delivered() {
        let guest = guest();
        for service in ["bcat:a", "bcat:m", "bcat:s"] {
            guest.service(service);
        }
        let creator = guest.service("bcat:u");
        let bcat = guest
            .call(creator, Request::new(0).send_pid().arg(0u64))
            .ok()
            .interface();
        let out = guest.heap;
        for request in [Request::new(10100), Request::new(10101).arg(name("dir"))] {
            let progress = guest.call(bcat, request).ok().interface();
            let event = guest.call(progress, Request::new(0)).ok().handle();
            assert!(guest.is_signaled(event));
            guest.write(out, &[0xaa; PROGRESS_SIZE]);
            guest
                .call(
                    progress,
                    Request::new(1).buffer_b(out, PROGRESS_SIZE as u64),
                )
                .ok();
            let state = guest.read(out, PROGRESS_SIZE);
            assert_eq!(state[..4], STATUS_DONE.to_le_bytes());
            assert!(state[4..].iter().all(|&b| b == 0));
        }

        let passphrase = guest.heap + 0x1000;
        guest
            .call(
                bcat,
                Request::new(30100)
                    .arg(PROGRAM_ID)
                    .buffer_a(passphrase, 0x40),
            )
            .ok();
        let empty_id = guest.call(
            bcat,
            Request::new(30100).arg(0u64).buffer_a(passphrase, 0x40),
        );
        assert_eq!(empty_id.result, INVALID_ARGUMENT);
        let long = guest.call(
            bcat,
            Request::new(30100).arg(1u64).buffer_a(passphrase, 0x41),
        );
        assert_eq!(long.result, INVALID_ARGUMENT);
        guest.call(bcat, Request::new(30300)).ok();
        guest.call(bcat, Request::new(90201).arg(PROGRAM_ID)).ok();
        assert_eq!(
            guest.call(bcat, Request::new(90201).arg(0u64)).result,
            INVALID_ARGUMENT
        );
    }

    #[test]
    fn the_delivery_cache_is_empty() {
        let guest = guest();
        let creator = guest.service("bcat:u");
        let storage = guest
            .call(creator, Request::new(2).arg(PROGRAM_ID))
            .ok()
            .interface();
        let out = guest.heap;
        let directories = guest
            .call(storage, Request::new(10).buffer_b(out, 0x100))
            .ok();
        assert_eq!(directories.at::<i32>(0), 0);

        let directory = guest.call(storage, Request::new(1)).ok().interface();
        let open = |text| {
            guest
                .call(directory, Request::new(0).arg(name(text)))
                .result
        };
        assert_eq!(open("news-1"), FAILED_OPEN_ENTITY);
        assert_eq!(open("bad/name"), INVALID_ARGUMENT);
        assert_eq!(open(""), INVALID_ARGUMENT);
        assert_eq!(guest.call(directory, Request::new(2)).result, NO_OPEN_ENTRY);

        let file = guest.call(storage, Request::new(0)).ok().interface();
        let open = |dir, file_name| {
            guest
                .call(file, Request::new(0).arg(name(dir)).arg(name(file_name)))
                .result
        };
        assert_eq!(open("news", "a.bin"), FAILED_OPEN_ENTITY);
        assert_eq!(open("news", "a-bin"), INVALID_ARGUMENT);
        assert_eq!(guest.call(file, Request::new(2)).result, NO_OPEN_ENTRY);

        let storage = guest
            .call(creator, Request::new(1).send_pid().arg(0u64))
            .ok()
            .interface();
        guest.call(storage, Request::new(10)).ok();
    }
}
