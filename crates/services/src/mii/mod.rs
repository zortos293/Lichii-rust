//! `mii:u` and `mii:e`: the console's Mii database. Both hand out `IDatabaseService` sessions
//! over one shared database, and only `mii:e` (system) sessions may edit it. There is no saved
//! database, so it starts empty, as the reference does without a NAND database; the six built-in
//! Miis are always there under the `Default` source.

mod tables;
#[cfg(test)]
mod tests;
mod types;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bytemuck::bytes_of;
use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};
use types::{ALL, CharInfo, CoreData, Field, GENDER_ALL, Rng, StoreData, Ver3StoreData};

const fn mii(description: u32) -> ResultCode {
    ResultCode::new(126, description)
}

const INVALID_ARGUMENT: ResultCode = mii(1);
const INVALID_ARGUMENT_SIZE: ResultCode = mii(2);
const NOT_UPDATED: ResultCode = mii(3);
const NOT_FOUND: ResultCode = mii(4);
const DATABASE_FULL: ResultCode = mii(5);
const INVALID_CHAR_INFO: ResultCode = mii(100);
const INVALID_STORE_DATA: ResultCode = mii(109);
const INVALID_OPERATION: ResultCode = mii(202);
const PERMISSION_DENIED: ResultCode = mii(203);
const TEST_MODE_ONLY: ResultCode = mii(204);

/// `SourceFlag` bits: the database and the built-in Miis.
const SOURCE_FLAG_DATABASE: u32 = 1;
const SOURCE_FLAG_DEFAULT: u32 = 2;
/// `Source` of a returned element.
const SOURCE_DATABASE: u32 = 0;
const SOURCE_DEFAULT: u32 = 1;

const MAX_MIIS: usize = 100;
const DEFAULT_MII_COUNT: usize = 6;

/// UpdateLatest1, Move, AddOrReplace, Delete and IsBrokenDatabaseWithClearFlag.
const SYSTEM_COMMANDS: [u32; 5] = [10, 12, 13, 14, 20];

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let seed = context.settings.system.rng_seed.map_or_else(
        || {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |time| time.as_nanos() as u64)
        },
        u64::from,
    );
    let database = Arc::new(Mutex::new(Database {
        miis: Vec::new(),
        update_counter: 0,
        rng: Rng(seed),
    }));
    for (name, is_system) in [("mii:e", true), ("mii:u", false)] {
        let database = database.clone();
        context.register(name, "services", move |context| StaticService {
            name,
            context: context.clone(),
            database: database.clone(),
            is_system,
        })?;
    }
    Ok(())
}

/// The Miis every session shares.
///
/// Sessions never carry the special-Mii magic and there is no saved database to load, so a
/// special Mii can never get in: indices need none of the reference's special-Mii skipping.
struct Database {
    miis: Vec<StoreData>,
    /// Bumped by every change; sessions compare it to answer `IsUpdated`.
    update_counter: u64,
    rng: Rng,
}

impl Database {
    fn find(&self, create_id: &[u8; 16]) -> Option<usize> {
        self.miis.iter().position(|mii| mii.create_id == *create_id)
    }

    /// Records a change `session` made, which other sessions then see as an update.
    fn changed(&mut self, session: &mut Session) {
        self.update_counter += 1;
        session.update_counter = self.update_counter;
    }

    /// The reference's `DatabaseManager::AddOrReplace`.
    fn add_or_replace(
        &mut self,
        session: &mut Session,
        store: StoreData,
    ) -> Result<(), ResultCode> {
        if !store.is_valid() {
            return Err(INVALID_STORE_DATA);
        }
        if store.is_special() {
            return Err(INVALID_OPERATION);
        }
        match self.find(&store.create_id) {
            Some(index) => self.miis[index] = store,
            None if self.miis.len() >= MAX_MIIS => return Err(DATABASE_FULL),
            None => self.miis.push(store),
        }
        self.changed(session);
        Ok(())
    }

    /// The entry `UpdateLatest` answers for a Mii with `create_id` and type `ty`.
    fn latest(
        &self,
        session: &Session,
        source_flag: u32,
        is_valid: bool,
        create_id: &[u8; 16],
        ty: u8,
    ) -> Result<StoreData, ResultCode> {
        if source_flag & SOURCE_FLAG_DATABASE == 0 {
            return Err(NOT_FOUND);
        }
        if session.interface_version >= 1 && !is_valid {
            return Err(INVALID_CHAR_INFO);
        }
        let store = self.miis[self.find(create_id).ok_or(NOT_FOUND)?];
        if store.core.get(Field::Type) != ty {
            return Err(NOT_FOUND);
        }
        Ok(store)
    }
}

/// `IStaticService`.
struct StaticService {
    name: &'static str,
    context: Arc<ServiceContext>,
    database: Arc<Mutex<Database>>,
    is_system: bool,
}

impl Service for StaticService {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        if ctx.command() != 0 {
            return None;
        }
        let mut database = self.database.lock();
        database.update_counter += 1;
        let session = Session {
            interface_version: 0,
            update_counter: database.update_counter,
        };
        drop(database);
        ctx.push_interface(self.context.handler(DatabaseService {
            database: self.database.clone(),
            session: Mutex::new(session),
            is_system: self.is_system,
        }));
        Some(ResultCode::SUCCESS)
    }
}

/// The reference's `DatabaseSessionMetadata`.
struct Session {
    interface_version: u32,
    /// The database's update counter when this session last looked.
    update_counter: u64,
}

/// `IDatabaseService`.
struct DatabaseService {
    database: Arc<Mutex<Database>>,
    session: Mutex<Session>,
    /// Opened through `mii:e`, so it may edit the database.
    is_system: bool,
}

type Command = fn(&mut HleRequestContext, &mut Database, &mut Session) -> Result<(), ResultCode>;

impl Service for DatabaseService {
    fn name(&self) -> &str {
        "IDatabaseService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let command: Command = match ctx.command() {
            0 => is_updated,
            1 => |ctx, database, _| {
                ctx.push(u8::from(database.miis.len() == MAX_MIIS));
                Ok(())
            },
            2 => get_count,
            3 => |ctx, database, _| get(ctx, database, Element::SourcedCharInfo),
            4 => |ctx, database, _| get(ctx, database, Element::CharInfo),
            5 => update_latest,
            6 => build_random,
            7 => build_default,
            8 => |ctx, database, _| get(ctx, database, Element::SourcedStoreData),
            9 => |ctx, database, _| get(ctx, database, Element::StoreData),
            10 => update_latest_store_data,
            11 => find_index,
            12 => move_mii,
            13 => |ctx, database, session| database.add_or_replace(session, ctx.pop()),
            14 => delete,
            15..=17 => |_, _, _| Err(TEST_MODE_ONLY),
            20 => |ctx, _, _| {
                ctx.push(0u8);
                Ok(())
            },
            21 => get_index,
            22 => |ctx, _, session| {
                session.interface_version = ctx.pop();
                Ok(())
            },
            23 => convert_ver3,
            24 => convert_core_data,
            25 => convert_char_info,
            26 => append,
            _ => return None,
        };
        if SYSTEM_COMMANDS.contains(&ctx.command()) && !self.is_system {
            return Some(PERMISSION_DENIED);
        }
        let result = command(ctx, &mut self.database.lock(), &mut self.session.lock());
        Some(result.err().unwrap_or(ResultCode::SUCCESS))
    }
}

fn push_char_info(ctx: &mut HleRequestContext, store: &StoreData) {
    ctx.push_bytes(bytes_of(&CharInfo::from_store_data(store)));
}

fn is_updated(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    session: &mut Session,
) -> Result<(), ResultCode> {
    let updated = ctx.pop::<u32>() & SOURCE_FLAG_DATABASE != 0 && {
        let last = std::mem::replace(&mut session.update_counter, database.update_counter);
        last != database.update_counter
    };
    ctx.push(u8::from(updated));
    Ok(())
}

fn get_count(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    _: &mut Session,
) -> Result<(), ResultCode> {
    let source_flag = ctx.pop::<u32>();
    let mut count = 0;
    if source_flag & SOURCE_FLAG_DEFAULT != 0 {
        count += DEFAULT_MII_COUNT;
    }
    if source_flag & SOURCE_FLAG_DATABASE != 0 {
        count += database.miis.len();
    }
    ctx.push(count as u32);
    Ok(())
}

/// What a `Get` command fills its buffer with.
#[derive(Clone, Copy)]
enum Element {
    SourcedCharInfo,
    CharInfo,
    SourcedStoreData,
    StoreData,
}

/// `Get`, `Get1`, `Get2` and `Get3`: the database Miis, then the built-in ones, as the source
/// flag asks. Built-in Miis get a new create id every time, as in the reference.
fn get(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    element: Element,
) -> Result<(), ResultCode> {
    let source_flag = ctx.pop::<u32>();
    let mut entries = Vec::new();
    if source_flag & SOURCE_FLAG_DATABASE != 0 {
        entries.extend(database.miis.iter().map(|&mii| (SOURCE_DATABASE, mii)));
    }
    if source_flag & SOURCE_FLAG_DEFAULT != 0 {
        for index in 0..DEFAULT_MII_COUNT {
            let create_id = database.rng.create_id();
            entries.push((SOURCE_DEFAULT, StoreData::default_mii(index, create_id)));
        }
    }
    let size = match element {
        Element::SourcedCharInfo => size_of::<CharInfo>() + 4,
        Element::CharInfo => size_of::<CharInfo>(),
        Element::SourcedStoreData => size_of::<StoreData>() + 4,
        Element::StoreData => size_of::<StoreData>(),
    };
    let capacity = ctx
        .writable_buffer(0)
        .map_or(0, |buffer| buffer.size as usize)
        / size;
    if entries.len() > capacity {
        return Err(INVALID_ARGUMENT_SIZE);
    }
    let mut out = Vec::with_capacity(entries.len() * size);
    for (source, store) in &entries {
        match element {
            Element::SourcedCharInfo | Element::CharInfo => {
                out.extend_from_slice(bytes_of(&CharInfo::from_store_data(store)));
            }
            Element::SourcedStoreData | Element::StoreData => {
                out.extend_from_slice(bytes_of(store));
            }
        }
        if matches!(
            element,
            Element::SourcedCharInfo | Element::SourcedStoreData
        ) {
            out.extend_from_slice(&source.to_le_bytes());
        }
    }
    if !out.is_empty() {
        ctx.write_buffer(0, &out)?;
    }
    ctx.push(entries.len() as u32);
    Ok(())
}

fn update_latest(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    session: &mut Session,
) -> Result<(), ResultCode> {
    let info = ctx.pop::<CharInfo>();
    let source_flag = ctx.pop::<u32>();
    let ty = info.get(Field::Type);
    let store = database.latest(session, source_flag, info.is_valid(), &info.create_id, ty)?;
    let latest = CharInfo::from_store_data(&store);
    if info.matches(&latest) {
        return Err(NOT_UPDATED);
    }
    ctx.push_bytes(bytes_of(&latest));
    Ok(())
}

fn update_latest_store_data(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    session: &mut Session,
) -> Result<(), ResultCode> {
    let store = ctx.pop::<StoreData>();
    let source_flag = ctx.pop::<u32>();
    let ty = store.core.get(Field::Type);
    let latest = database.latest(session, source_flag, store.is_valid(), &store.create_id, ty)?;
    if store.matches(&latest) {
        return Err(NOT_UPDATED);
    }
    ctx.push_bytes(bytes_of(&latest));
    Ok(())
}

/// `BuildRandom` takes age, gender and race as one word each.
fn build_random(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    _: &mut Session,
) -> Result<(), ResultCode> {
    let (age, gender, race) = (ctx.pop::<u32>(), ctx.pop::<u32>(), ctx.pop::<u32>());
    if age > u32::from(ALL) || gender > u32::from(GENDER_ALL) || race > u32::from(ALL) {
        return Err(INVALID_ARGUMENT);
    }
    let store = StoreData::random(&mut database.rng, age as u8, gender as u8, race as u8);
    push_char_info(ctx, &store);
    Ok(())
}

fn build_default(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    _: &mut Session,
) -> Result<(), ResultCode> {
    let index = usize::try_from(ctx.pop::<i32>())
        .ok()
        .filter(|&index| index < DEFAULT_MII_COUNT)
        .ok_or(INVALID_ARGUMENT)?;
    let store = StoreData::default_mii(index, database.rng.create_id());
    push_char_info(ctx, &store);
    Ok(())
}

/// `FindIndex` answers -1 for a Mii that is not in the database.
fn find_index(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    _: &mut Session,
) -> Result<(), ResultCode> {
    let create_id = ctx.pop::<[u8; 16]>();
    let index = database.find(&create_id).map_or(-1, |index| index as i32);
    ctx.push(index);
    Ok(())
}

fn move_mii(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    session: &mut Session,
) -> Result<(), ResultCode> {
    let create_id = ctx.pop::<[u8; 16]>();
    let new_index = usize::try_from(ctx.pop::<i32>())
        .ok()
        .filter(|&index| index < database.miis.len())
        .ok_or(INVALID_ARGUMENT)?;
    let index = database.find(&create_id).ok_or(NOT_FOUND)?;
    if index == new_index {
        return Err(NOT_UPDATED);
    }
    let mii = database.miis.remove(index);
    database.miis.insert(new_index, mii);
    database.changed(session);
    Ok(())
}

fn delete(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    session: &mut Session,
) -> Result<(), ResultCode> {
    let index = database.find(&ctx.pop()).ok_or(NOT_FOUND)?;
    database.miis.remove(index);
    database.changed(session);
    Ok(())
}

fn get_index(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    _: &mut Session,
) -> Result<(), ResultCode> {
    let info = ctx.pop::<CharInfo>();
    if !info.is_valid() {
        return Err(INVALID_CHAR_INFO);
    }
    let index = database.find(&info.create_id).ok_or(NOT_FOUND)?;
    ctx.push(index as i32);
    Ok(())
}

/// `Convert`: a 3DS or Wii U Mii as a CharInfo under a new create id.
fn convert_ver3(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    _: &mut Session,
) -> Result<(), ResultCode> {
    let ver3 = ctx.pop::<Ver3StoreData>();
    if !ver3.is_valid() {
        return Err(INVALID_CHAR_INFO);
    }
    push_char_info(ctx, &ver3.to_store_data(database.rng.create_id()));
    Ok(())
}

fn convert_core_data(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    _: &mut Session,
) -> Result<(), ResultCode> {
    let core = ctx.pop::<CoreData>();
    if !core.is_valid() {
        return Err(INVALID_CHAR_INFO);
    }
    push_char_info(ctx, &StoreData::new(core, database.rng.create_id()));
    Ok(())
}

fn convert_char_info(
    ctx: &mut HleRequestContext,
    _: &mut Database,
    _: &mut Session,
) -> Result<(), ResultCode> {
    let info = ctx.pop::<CharInfo>();
    if !info.is_valid() {
        return Err(INVALID_CHAR_INFO);
    }
    ctx.push(CoreData::from_char_info(&info));
    Ok(())
}

/// `Append` adds a CharInfo under a new create id. Any failure answers `NotFound`, as in the
/// reference.
fn append(
    ctx: &mut HleRequestContext,
    database: &mut Database,
    session: &mut Session,
) -> Result<(), ResultCode> {
    let info = ctx.pop::<CharInfo>();
    if !info.is_valid() || info.get(Field::Type) == 1 {
        return Err(NOT_FOUND);
    }
    let core = CoreData::from_char_info(&info);
    let store = loop {
        let create_id = database.rng.create_id();
        if database.find(&create_id).is_none() {
            break StoreData::new(core, create_id);
        }
    };
    database
        .add_or_replace(session, store)
        .map_err(|_| NOT_FOUND)
}
