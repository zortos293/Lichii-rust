//! ncm and lr: the content meta database and content storage, and the location resolvers, over
//! the running title and its add-ons. There is no installed NAND content, so placeholders (only
//! the system installer uses them) fail as the C++ does without a placeholder cache.

mod lr;

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::sync::{Arc, Mutex};

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;

use crate::ns::Placeholder;
use crate::{Service, ServiceContext};

const THREAD: &str = "ncm";
/// `ResultUnknown`: what the C++ placeholder commands return without a placeholder cache.
const RESULT_UNKNOWN: ResultCode = ResultCode(u32::MAX);
/// ncm ResultContentMetaNotFound (5-7).
const CONTENT_META_NOT_FOUND: ResultCode = ResultCode::new(5, 7);
const META_TYPE_APPLICATION: u8 = 0x80;
const META_TYPE_ADD_ON_CONTENT: u8 = 0x82;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("lr", THREAD, |context| lr::LocationResolverManager {
        context: context.clone(),
    })?;
    context.register("ncm", THREAD, |context| ContentManager {
        context: context.clone(),
    })?;
    context.register("ncm:v", THREAD, |_| Placeholder("ncm:v"))
}

struct ContentManager {
    context: Arc<ServiceContext>,
}

impl Service for ContentManager {
    fn name(&self) -> &str {
        "ncm"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let handler = match ctx.command() {
            4 => self.context.handler(ContentStorage),
            5 => self.context.handler(ContentMetaDatabase {
                context: self.context.clone(),
                pending: Mutex::new(Vec::new()),
            }),
            _ => return None,
        };
        ctx.push_interface(handler);
        Some(ResultCode::SUCCESS)
    }
}

/// IContentStorage: hands out placeholder ids; everything that would write content fails.
struct ContentStorage;

impl Service for ContentStorage {
    fn name(&self) -> &str {
        "IContentStorage"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                let state = RandomState::new();
                let mut id = [0u8; 16];
                id[..8].copy_from_slice(&state.hash_one(0u8).to_le_bytes());
                id[8..].copy_from_slice(&state.hash_one(1u8).to_le_bytes());
                ctx.push(id);
                ResultCode::SUCCESS
            }
            1 | 2 | 4 | 5 | 6 => {
                tracing::warn!(
                    command = ctx.command(),
                    "IContentStorage: no placeholder cache"
                );
                RESULT_UNKNOWN
            }
            _ => return None,
        })
    }
}

/// `ContentMetaKey`: u64 id, u32 version, u8 type, u8 install type, 2 bytes padding.
type ContentMetaKey = [u8; 0x10];

fn key_id(key: &ContentMetaKey) -> u64 {
    u64::from_le_bytes(key[..8].try_into().unwrap())
}

/// IContentMetaDatabase: the running title and its add-ons are installed; `Set` adds pending
/// entries until `Remove`.
struct ContentMetaDatabase {
    context: Arc<ServiceContext>,
    pending: Mutex<Vec<ContentMetaKey>>,
}

impl ContentMetaDatabase {
    /// The installed (version, meta type) of title `id`.
    fn installed(&self, id: u64) -> Option<(u32, u8)> {
        let program = &self.context.program;
        if id == program.program_id {
            return Some((0, META_TYPE_APPLICATION));
        }
        program
            .add_ons
            .iter()
            .find(|add_on| add_on.title_id == id)
            .map(|add_on| (add_on.version, META_TYPE_ADD_ON_CONTENT))
    }
}

impl Service for ContentMetaDatabase {
    fn name(&self) -> &str {
        "IContentMetaDatabase"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                let key: ContentMetaKey = ctx.pop();
                let mut pending = self.pending.lock().unwrap();
                if !pending.contains(&key) {
                    pending.push(key);
                }
                ResultCode::SUCCESS
            }
            2 => {
                let key: ContentMetaKey = ctx.pop();
                self.pending.lock().unwrap().retain(|entry| *entry != key);
                ResultCode::SUCCESS
            }
            6 => {
                let id: u64 = ctx.pop();
                let Some((version, meta_type)) = self.installed(id) else {
                    return Some(CONTENT_META_NOT_FOUND);
                };
                let mut key = [0u8; 0x10];
                key[..8].copy_from_slice(&id.to_le_bytes());
                key[8..12].copy_from_slice(&version.to_le_bytes());
                key[12] = meta_type;
                ctx.push(key);
                ResultCode::SUCCESS
            }
            8 => {
                let key: ContentMetaKey = ctx.pop();
                let has = self.pending.lock().unwrap().contains(&key)
                    || self.installed(key_id(&key)).is_some();
                ctx.push(u8::from(has));
                ResultCode::SUCCESS
            }
            15 => ResultCode::SUCCESS,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests;
