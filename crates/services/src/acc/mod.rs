//! `acc:u0`, `acc:u1`, `acc:su`, `acc:aa`, `acc:e*` and `dauth:0`: one user profile, named by
//! `settings.system.username`, opened when the title boots. No Nintendo Account is linked.
//!
//! Profiles are kept in the NAND's `profiles.dat`, as the C++ ProfileManager keeps them: the
//! user whose id or nickname matches the username is opened, else one is added; profile edits
//! are saved, and the other users in the file are kept as they are.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytemuck::{Pod, Zeroable};
use lichii_common::settings::write_atomic;
use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

const MODULE: u32 = 124;
const INVALID_USER_ID: ResultCode = ResultCode::new(MODULE, 20);
const INVALID_APPLICATION: ResultCode = ResultCode::new(MODULE, 22);
const INVALID_ARRAY_LENGTH: ResultCode = ResultCode::new(MODULE, 32);
const APPLICATION_INFO_ALREADY_INITIALIZED: ResultCode = ResultCode::new(MODULE, 41);
const UNKNOWN: ResultCode = ResultCode(u32::MAX);

const MAX_USERS: usize = 8;
const USER_DATA_SIZE: usize = 0x80;
const THUMBNAIL_SIZE: u64 = 0x24000;
const ID_TOKEN_CACHE_SIZE: usize = 0x100;
const NAS_USER_BASE_SIZE: usize = 0x68;
const NACP_USER_ACCOUNT_SWITCH_LOCK: usize = 0x3026;
const PROFILES_PATH: &str = "system/save/8000000000000010/su/avators/profiles.dat";

/// The avatar every profile shows: a 256x256 JPEG.
const AVATAR: &[u8] = include_bytes!("backup_avatar.jpg");

/// `nn::account::Uid`.
type Uid = [u8; 16];

/// A user id that stays the same for the same name across boots, so save data stays put.
fn user_id(username: &str) -> Uid {
    let fnv = |seed: u64| {
        username.bytes().fold(seed, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
        })
    };
    let mut id = [0; 16];
    id[..8].copy_from_slice(&fnv(0xcbf2_9ce4_8422_2325).to_le_bytes());
    id[8..].copy_from_slice(&(fnv(0x8422_2325_cbf2_9ce4) | 1).to_le_bytes());
    id
}

/// The network service account id titles see: `Common::UUID::Hash` of the user id.
fn account_id(uid: &Uid) -> u64 {
    let (upper, lower) = uid.split_at(8);
    let upper = u64::from_le_bytes(upper.try_into().unwrap());
    let lower = u64::from_le_bytes(lower.try_into().unwrap());
    upper ^ lower.rotate_left(1)
}

/// A user of `profiles.dat` (`UserRaw`); an all-zero uuid is a free slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UserRaw {
    uuid: Uid,
    uuid2: Uid,
    timestamp: u64,
    nickname: [u8; 32],
    user_data: [u8; USER_DATA_SIZE],
}

/// `profiles.dat` (`ProfileDataRaw`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProfileData {
    reserved: [u8; 0x10],
    users: [UserRaw; MAX_USERS],
}

const _: () = assert!(size_of::<ProfileData>() == 0x650);

/// The users of the file at `path`; none when it is missing or short.
fn read_profiles(path: &Path) -> ProfileData {
    let data = std::fs::read(path).unwrap_or_default();
    match data.get(..size_of::<ProfileData>()) {
        Some(data) => bytemuck::pod_read_unaligned(data),
        None => {
            if !data.is_empty() {
                tracing::warn!("{} is too short; starting without users", path.display());
            }
            ProfileData::zeroed()
        }
    }
}

/// The profile all `acc` interfaces share, and the file it is saved in.
struct Profile {
    uid: Uid,
    /// `nn::account::profile::ProfileBase`: uid, timestamp, 32-byte nickname.
    base: [u8; 0x38],
    user_data: [u8; USER_DATA_SIZE],
    open_context_stored: bool,
    path: PathBuf,
    data: ProfileData,
    /// The profile's user in `data`; none when all eight slots belong to other users.
    slot: Option<usize>,
}

impl Profile {
    /// Opens `username`'s user in `profiles.dat` under `nand`, adding (and saving) it when the
    /// file has no user with its id or nickname.
    fn open(username: &str, nand: &Path) -> Self {
        let path = nand.join(PROFILES_PATH);
        let data = read_profiles(&path);
        let uid = user_id(username);
        let mut nickname = [0; 32];
        let name = &username.as_bytes()[..username.len().min(32)];
        nickname[..name.len()].copy_from_slice(name);
        let found = data.users.iter().position(|user| {
            user.uuid != [0; 16] && (user.uuid == uid || user.nickname == nickname)
        });
        let slot = found.or_else(|| data.users.iter().position(|user| user.uuid == [0; 16]));
        let user = match found {
            Some(slot) => data.users[slot],
            None => UserRaw {
                uuid: uid,
                uuid2: uid,
                timestamp: 0,
                nickname,
                user_data: [0; USER_DATA_SIZE],
            },
        };
        let mut base = [0; 0x38];
        base[..16].copy_from_slice(&user.uuid);
        base[0x10..0x18].copy_from_slice(&user.timestamp.to_le_bytes());
        base[0x18..].copy_from_slice(&user.nickname);
        let mut profile = Self {
            uid: user.uuid,
            base,
            user_data: user.user_data,
            open_context_stored: false,
            path,
            data,
            slot,
        };
        match (found, slot) {
            (Some(_), _) => {}
            (None, Some(_)) => profile.save(),
            (None, None) => {
                tracing::warn!("profiles.dat has no free slot; {username} is not saved")
            }
        }
        profile
    }

    /// Writes the profile into its slot and saves `profiles.dat`.
    fn save(&mut self) {
        let Some(slot) = self.slot else {
            return;
        };
        self.data.users[slot] = UserRaw {
            uuid: self.uid,
            uuid2: self.uid,
            timestamp: u64::from_le_bytes(self.base[0x10..0x18].try_into().unwrap()),
            nickname: self.base[0x18..].try_into().unwrap(),
            user_data: self.user_data,
        };
        if let Err(e) = write_atomic(&self.path, bytemuck::bytes_of(&self.data)) {
            tracing::error!("saving {}: {e}", self.path.display());
        }
    }

    /// `ListAllUsers` and friends: the user ids, padded with zeros to `MAX_USERS`.
    fn user_list(&self, include: bool) -> [u8; 16 * MAX_USERS] {
        let mut list = [0; 16 * MAX_USERS];
        if include {
            list[..16].copy_from_slice(&self.uid);
        }
        list
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Interface {
    /// `acc:u0`, for applications.
    Application,
    /// `acc:u1`, for system applets.
    System,
    /// `acc:su`, for the system settings.
    Administrator,
}

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let profile = Profile::open(
        &context.settings.system.username,
        &crate::fs::nand_dir(&context.settings),
    );
    let profile = Arc::new(Mutex::new(profile));
    let services = [
        ("acc:u0", Interface::Application),
        ("acc:u1", Interface::System),
        ("acc:su", Interface::Administrator),
    ];
    for (name, interface) in services {
        let profile = profile.clone();
        context.register(name, "acc", move |context| AccountService {
            name,
            interface,
            context: context.clone(),
            profile: profile.clone(),
            application_info: AtomicBool::new(false),
        })?;
    }
    for name in ["acc:aa", "acc:e", "acc:e:u1", "acc:e:u2", "dauth:0"] {
        context.register(name, "acc", move |_| Stub { name })?;
    }
    Ok(())
}

/// `IAccountServiceForApplication`, `IAccountServiceForSystemService` and
/// `IAccountServiceForAdministrator`.
struct AccountService {
    name: &'static str,
    interface: Interface,
    context: Arc<ServiceContext>,
    profile: Arc<Mutex<Profile>>,
    application_info: AtomicBool,
}

impl AccountService {
    /// Writes the user list; `open_context` lists only users `StoreOpenContext` kept.
    fn write_users(&self, ctx: &mut HleRequestContext, open_context: bool) -> ResultCode {
        let profile = self.profile.lock();
        let list = profile.user_list(!open_context || profile.open_context_stored);
        drop(profile);
        match ctx.write_buffer(0, &list) {
            Ok(_) => ResultCode::SUCCESS,
            Err(result) => result,
        }
    }

    fn push_profile(&self, ctx: &mut HleRequestContext, editor: bool) {
        let uid: Uid = ctx.pop();
        let profile = self.profile.clone();
        ctx.push_interface(self.context.handler(ProfileService {
            uid,
            editor,
            profile,
        }));
    }

    fn initialize_application_info(&self) -> ResultCode {
        if self.application_info.swap(true, Ordering::Relaxed) {
            APPLICATION_INFO_ALREADY_INITIALIZED
        } else {
            ResultCode::SUCCESS
        }
    }

    fn store_thumbnail(ctx: &mut HleRequestContext, uid: Uid, program_id: u64) -> ResultCode {
        if program_id == 0 {
            return INVALID_APPLICATION;
        }
        if uid == [0; 16] {
            return INVALID_USER_ID;
        }
        if ctx.readable_buffer(0).map_or(0, |buffer| buffer.size) != THUMBNAIL_SIZE {
            return INVALID_ARRAY_LENGTH;
        }
        ResultCode::SUCCESS
    }
}

impl Service for AccountService {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let application = self.interface == Interface::Application;
        let administrator = self.interface == Interface::Administrator;
        match ctx.command() {
            0 => ctx.push(1u32),
            1 => {
                let uid: Uid = ctx.pop();
                ctx.push(u8::from(uid == self.profile.lock().uid));
            }
            2 | 3 => return Some(self.write_users(ctx, false)),
            141 if application => return Some(self.write_users(ctx, false)),
            140 if !application => return Some(self.write_users(ctx, false)),
            60 => return Some(self.write_users(ctx, true)),
            131 if application => return Some(self.write_users(ctx, true)),
            4 | 51 | 52 => ctx.push(self.profile.lock().uid),
            5 => self.push_profile(ctx, false),
            205 if administrator => self.push_profile(ctx, true),
            50 => ctx.push(1u8),
            100 | 140 if application => return Some(self.initialize_application_info()),
            160 if application => {}
            101 if application => ctx.push_interface(self.context.handler(ManagerForApplication {
                context: self.context.clone(),
                profile: self.profile.clone(),
                async_context: Arc::new(AsyncContext::new()),
            })),
            102 if !application => {
                let uid: Uid = ctx.pop();
                ctx.push_interface(self.context.handler(ManagerForSystemService { uid }));
            }
            110 => {
                let uid: Uid = ctx.pop();
                let program_id = if administrator { ctx.pop() } else { 1 };
                return Some(Self::store_thumbnail(ctx, uid, program_id));
            }
            150 if application => {
                let locked = self
                    .context
                    .program
                    .nacp
                    .as_ref()
                    .is_some_and(|nacp| nacp.raw()[NACP_USER_ACCOUNT_SWITCH_LOCK] != 0);
                ctx.push(u8::from(locked));
            }
            250 if administrator => {
                let _uid: Uid = ctx.pop();
                ctx.push_interface(self.context.handler(Administrator));
            }
            401 if !application => ctx.push(0u32),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `IProfile` and `IProfileEditor`.
struct ProfileService {
    uid: Uid,
    editor: bool,
    profile: Arc<Mutex<Profile>>,
}

impl ProfileService {
    fn store(&self, ctx: &mut HleRequestContext, user_data: Vec<u8>) -> ResultCode {
        let base: [u8; 0x38] = ctx.pop();
        let Some(user_data) = user_data.first_chunk::<USER_DATA_SIZE>() else {
            return INVALID_ARRAY_LENGTH;
        };
        let mut profile = self.profile.lock();
        if base[..16] != profile.uid || self.uid != profile.uid {
            return ResultCode::new(MODULE, 100);
        }
        profile.base = base;
        profile.user_data = *user_data;
        profile.save();
        ResultCode::SUCCESS
    }
}

impl Service for ProfileService {
    fn name(&self) -> &str {
        if self.editor {
            "IProfileEditor"
        } else {
            "IProfile"
        }
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 | 1 => {
                let profile = self.profile.lock();
                if self.uid != profile.uid {
                    return Some(UNKNOWN);
                }
                if ctx.command() == 0
                    && let Err(result) = ctx.write_buffer(0, &profile.user_data)
                {
                    return Some(result);
                }
                ctx.push(profile.base);
            }
            10 => ctx.push(AVATAR.len() as u32),
            11 => {
                if let Err(result) = ctx.write_buffer(0, AVATAR) {
                    return Some(result);
                }
                ctx.push(AVATAR.len() as u32);
            }
            20 | 21 | 30 => {}
            100 if self.editor => {
                let user_data = ctx.read_buffer(0).unwrap_or_default();
                return Some(self.store(ctx, user_data));
            }
            101 if self.editor => {
                let user_data = ctx
                    .buffers_x()
                    .first()
                    .map(|buffer| {
                        let mut data = vec![0; buffer.size as usize];
                        ctx.memory()
                            .read_bytes(buffer.address, &mut data)
                            .map(|()| data)
                    })
                    .and_then(Result::ok)
                    .unwrap_or_default();
                return Some(self.store(ctx, user_data));
            }
            110 if self.editor => {}
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `IManagerForApplication`: the Nintendo Account of the user, which is never linked.
struct ManagerForApplication {
    context: Arc<ServiceContext>,
    profile: Arc<Mutex<Profile>>,
    async_context: Arc<AsyncContext>,
}

impl Service for ManagerForApplication {
    fn name(&self) -> &str {
        "IManagerForApplication"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => {}
            1 => ctx.push(account_id(&self.profile.lock().uid)),
            2 => ctx.push_interface(
                self.context
                    .handler(AsyncContextService(self.async_context.clone())),
            ),
            3 => ctx.push(0u32),
            4 => {
                if let Err(result) = ctx.write_buffer(0, &[0; ID_TOKEN_CACHE_SIZE]) {
                    return Some(result);
                }
                ctx.push(ID_TOKEN_CACHE_SIZE as u32);
            }
            130 | 136 => {
                if let Err(result) = ctx.write_buffer(0, &[0; NAS_USER_BASE_SIZE]) {
                    return Some(result);
                }
                if let Some(buffer) = ctx.writable_buffer(1).filter(|buffer| buffer.size != 0) {
                    let zeros = vec![0; buffer.size as usize];
                    if let Err(result) = ctx.write_buffer(1, &zeros) {
                        return Some(result);
                    }
                }
                ctx.push(account_id(&self.profile.lock().uid));
            }
            150 => ctx.push_interface(self.context.handler(Stub {
                name: "IAuthorizationRequest",
            })),
            160 => self.profile.lock().open_context_stored = true,
            170 => ctx.push_interface(self.context.handler(Stub {
                name: "IAsyncNetworkServiceLicenseKindContext",
            })),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// An `IAsyncContext` that completed as soon as it was created.
struct AsyncContext {
    completion: Arc<Event>,
}

impl AsyncContext {
    fn new() -> Self {
        let completion = Arc::new(Event::new());
        completion.signal();
        Self { completion }
    }
}

struct AsyncContextService(Arc<AsyncContext>);

impl Service for AsyncContextService {
    fn name(&self) -> &str {
        "IAsyncContext"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push_copy_object(KObject::ReadableEvent(self.0.completion.clone())),
            1 | 3 => {}
            2 => ctx.push(1u8),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `IManagerForSystemService`.
struct ManagerForSystemService {
    uid: Uid,
}

impl Service for ManagerForSystemService {
    fn name(&self) -> &str {
        "IManagerForSystemService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 | 3 | 4 => {}
            1 => ctx.push(account_id(&self.uid)),
            143 => {
                ctx.push(0u32);
                ctx.push(0i64);
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `IAdministrator`.
struct Administrator;

impl Service for Administrator {
    fn name(&self) -> &str {
        "IAdministrator"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            250 => ctx.push(0u8),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// A service or interface with no implemented commands, as the C++ reference returns it.
struct Stub {
    name: &'static str,
}

impl Service for Stub {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, _: &mut HleRequestContext) -> Option<ResultCode> {
        None
    }
}

#[cfg(test)]
mod tests;
